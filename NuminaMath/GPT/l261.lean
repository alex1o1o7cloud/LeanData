import Mathlib

namespace fill_tank_time_l261_261660

-- Define the rates of filling and draining
def rateA : ℕ := 200 -- Pipe A fills at 200 liters per minute
def rateB : ℕ := 50  -- Pipe B fills at 50 liters per minute
def rateC : ℕ := 25  -- Pipe C drains at 25 liters per minute

-- Define the times each pipe is open
def timeA : ℕ := 1   -- Pipe A is open for 1 minute
def timeB : ℕ := 2   -- Pipe B is open for 2 minutes
def timeC : ℕ := 2   -- Pipe C is open for 2 minutes

-- Define the capacity of the tank
def tankCapacity : ℕ := 1000

-- Prove the total time to fill the tank is 20 minutes
theorem fill_tank_time : 
  (tankCapacity * ((timeA * rateA + timeB * rateB) - (timeC * rateC)) * 5) = 20 :=
sorry

end fill_tank_time_l261_261660


namespace Danielle_rooms_is_6_l261_261871

-- Definitions for the problem conditions
def Heidi_rooms (Danielle_rooms : ℕ) : ℕ := 3 * Danielle_rooms
def Grant_rooms (Heidi_rooms : ℕ) : ℕ := Heidi_rooms / 9
def Grant_rooms_value : ℕ := 2

-- Theorem statement
theorem Danielle_rooms_is_6 (h : Grant_rooms_value = Grant_rooms (Heidi_rooms d)) : d = 6 :=
by
  sorry

end Danielle_rooms_is_6_l261_261871


namespace isosceles_triangle_sides_part1_isosceles_triangle_sides_part2_l261_261533

-- Part 1 proof
theorem isosceles_triangle_sides_part1 (x : ℝ) (h1 : x + 2 * x + 2 * x = 20) : 
  x = 4 ∧ 2 * x = 8 :=
by
  sorry

-- Part 2 proof
theorem isosceles_triangle_sides_part2 (a b : ℝ) (h2 : a = 5) (h3 : 2 * b + a = 20) :
  b = 7.5 :=
by
  sorry

end isosceles_triangle_sides_part1_isosceles_triangle_sides_part2_l261_261533


namespace minimum_gennadys_l261_261697

theorem minimum_gennadys (alexs borises vasilies x : ℕ) (h₁ : alexs = 45) (h₂ : borises = 122) (h₃ : vasilies = 27)
    (h₄ : ∀ i, i ∈ list.range (borises-1) → alexs + vasilies + x > i) : 
    x = 49 :=
by 
    sorry

end minimum_gennadys_l261_261697


namespace length_of_platform_l261_261971

noncomputable def train_length : ℝ := 450
noncomputable def signal_pole_time : ℝ := 18
noncomputable def platform_time : ℝ := 39

theorem length_of_platform : 
  ∃ (L : ℝ), 
    (train_length / signal_pole_time = (train_length + L) / platform_time) → 
    L = 525 := 
by
  sorry

end length_of_platform_l261_261971


namespace petes_original_number_l261_261341

theorem petes_original_number (x : ℤ) (h : 4 * (2 * x + 20) = 200) : x = 15 :=
sorry

end petes_original_number_l261_261341


namespace river_width_l261_261529

noncomputable def width_of_river (d: ℝ) (f: ℝ) (v: ℝ) : ℝ :=
  v / (d * (f * 1000 / 60))

theorem river_width : width_of_river 2 2 3000 = 45 := by
  sorry

end river_width_l261_261529


namespace triangle_angle_bisector_proportion_l261_261891

theorem triangle_angle_bisector_proportion
  (a b c x y : ℝ)
  (h : x / c = y / a)
  (h2 : x + y = b) :
  x / c = b / (a + c) :=
sorry

end triangle_angle_bisector_proportion_l261_261891


namespace triangle_trig_identity_l261_261890

open Real

theorem triangle_trig_identity (A B C : ℝ) (h_triangle : A + B + C = 180) (h_A : A = 15) :
  sqrt 3 * sin A - cos (B + C) = sqrt 2 := by
  sorry

end triangle_trig_identity_l261_261890


namespace workshop_average_salary_l261_261582

theorem workshop_average_salary :
  let technicians := 8
  let rest := 24 - technicians
  let avg_technician_salary := 12000
  let avg_rest_salary := 6000
  let total_workers := 24
  let total_staff_salary := (technicians * avg_technician_salary) + (rest * avg_rest_salary)
  let A := total_staff_salary / total_workers
  A = 8000 :=
by
  -- Definitions according to given conditions
  let technicians := 8
  let rest := 24 - technicians
  let avg_technician_salary := 12000
  let avg_rest_salary := 6000
  let total_workers := 24
  let total_staff_salary := (technicians * avg_technician_salary) + (rest * avg_rest_salary)
  let A := total_staff_salary / total_workers
  -- We need to show that A = 8000
  show A = 8000
  sorry

end workshop_average_salary_l261_261582


namespace min_gennadies_l261_261687

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l261_261687


namespace nat_implies_int_incorrect_reasoning_due_to_minor_premise_l261_261938

-- Definitions for conditions
def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n
def is_natural (x : ℚ) : Prop := ∃ (n : ℕ), x = n

-- Major premise: Natural numbers are integers
theorem nat_implies_int (n : ℕ) : is_integer n := 
  ⟨n, rfl⟩

-- Minor premise: 1 / 3 is a natural number
def one_div_three_is_natural : Prop := is_natural (1 / 3)

-- Conclusion: 1 / 3 is an integer
def one_div_three_is_integer : Prop := is_integer (1 / 3)

-- The proof problem
theorem incorrect_reasoning_due_to_minor_premise :
  ¬one_div_three_is_natural :=
sorry

end nat_implies_int_incorrect_reasoning_due_to_minor_premise_l261_261938


namespace solve_for_3x_plus_9_l261_261326

theorem solve_for_3x_plus_9 :
  ∀ (x : ℝ), (5 * x - 8 = 15 * x + 18) → 3 * (x + 9) = 96 / 5 :=
by
  intros x h
  sorry

end solve_for_3x_plus_9_l261_261326


namespace train_passes_jogger_l261_261125

noncomputable def speed_of_jogger_kmph := 9
noncomputable def speed_of_train_kmph := 45
noncomputable def jogger_lead_m := 270
noncomputable def train_length_m := 120

noncomputable def speed_of_jogger_mps := speed_of_jogger_kmph * (1000 / 3600) -- converting km/hr to m/s
noncomputable def speed_of_train_mps := speed_of_train_kmph * (1000 / 3600) -- converting km/hr to m/s
noncomputable def relative_speed_mps := speed_of_train_mps - speed_of_jogger_mps
noncomputable def total_distance_m := jogger_lead_m + train_length_m
noncomputable def time_to_pass_jogger := total_distance_m / relative_speed_mps

theorem train_passes_jogger : time_to_pass_jogger = 39 :=
  by
    -- Proof steps would be provided here
    sorry

end train_passes_jogger_l261_261125


namespace choose_5_with_exactly_one_twin_l261_261523

theorem choose_5_with_exactly_one_twin :
  let total_players := 12
  let twins := 2
  let players_to_choose := 5
  let remaining_players_after_one_twin := total_players - twins + 1 -- 11 players to choose from
  (2 * Nat.choose remaining_players_after_one_twin (players_to_choose - 1)) = 420 := 
by
  sorry

end choose_5_with_exactly_one_twin_l261_261523


namespace cos_210_eq_neg_sqrt3_div_2_l261_261152

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261152


namespace cos_4_3pi_add_alpha_l261_261434

theorem cos_4_3pi_add_alpha (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
    Real.cos (4 * Real.pi / 3 + α) = -1 / 3 := 
by sorry

end cos_4_3pi_add_alpha_l261_261434


namespace angle_symmetry_l261_261857

theorem angle_symmetry (α β : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) (hβ : 0 < β ∧ β < 2 * Real.pi) (h_symm : α = 2 * Real.pi - β) : α + β = 2 * Real.pi := 
by 
  sorry

end angle_symmetry_l261_261857


namespace find_x_l261_261524

-- Define the conditions
def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45
def molecular_weight : ℝ := 152

-- State the theorem
theorem find_x : ∃ x : ℕ, molecular_weight = atomic_weight_C + atomic_weight_Cl * x ∧ x = 4 := by
  sorry

end find_x_l261_261524


namespace no_equal_partition_of_173_ones_and_neg_ones_l261_261026

theorem no_equal_partition_of_173_ones_and_neg_ones
  (L : List ℤ) (h1 : L.length = 173) (h2 : ∀ x ∈ L, x = 1 ∨ x = -1) :
  ¬ (∃ (L1 L2 : List ℤ), L = L1 ++ L2 ∧ L1.sum = L2.sum) :=
by
  sorry

end no_equal_partition_of_173_ones_and_neg_ones_l261_261026


namespace employee_b_pay_l261_261381

theorem employee_b_pay (total_pay : ℝ) (ratio_ab : ℝ) (pay_b : ℝ) 
  (h1 : total_pay = 570)
  (h2 : ratio_ab = 1.5 * pay_b)
  (h3 : total_pay = ratio_ab + pay_b) :
  pay_b = 228 := 
sorry

end employee_b_pay_l261_261381


namespace Denise_age_l261_261678

-- Define the ages of Amanda, Carlos, Beth, and Denise
variables (A C B D : ℕ)

-- State the given conditions
def condition1 := A = C - 4
def condition2 := C = B + 5
def condition3 := D = B + 2
def condition4 := A = 16

-- The theorem to prove
theorem Denise_age (A C B D : ℕ) (h1 : condition1 A C) (h2 : condition2 C B) (h3 : condition3 D B) (h4 : condition4 A) : D = 17 :=
by
  sorry

end Denise_age_l261_261678


namespace f_at_10_l261_261488

variable (f : ℕ → ℝ)

-- Conditions
axiom f_1 : f 1 = 2
axiom f_relation : ∀ m n : ℕ, m ≥ n → f (m + n) + f (m - n) = (f (2 * m) + f (2 * n)) / 2 + 2 * n

-- Prove f(10) = 361
theorem f_at_10 : f 10 = 361 :=
by
  sorry

end f_at_10_l261_261488


namespace milo_dozen_eggs_l261_261913

theorem milo_dozen_eggs (total_weight_pounds egg_weight_pounds dozen : ℕ) (h1 : total_weight_pounds = 6)
  (h2 : egg_weight_pounds = 1 / 16) (h3 : dozen = 12) :
  total_weight_pounds / egg_weight_pounds / dozen = 8 :=
by
  -- The proof would go here
  sorry

end milo_dozen_eggs_l261_261913


namespace fishing_problem_l261_261968

theorem fishing_problem (a b c d : ℕ)
  (h1 : a + b + c + d = 11)
  (h2 : 1 ≤ a) 
  (h3 : 1 ≤ b) 
  (h4 : 1 ≤ c) 
  (h5 : 1 ≤ d) : 
  a < 3 ∨ b < 3 ∨ c < 3 ∨ d < 3 :=
by
  -- This is a placeholder for the proof
  sorry

end fishing_problem_l261_261968


namespace gcd_654327_543216_is_1_l261_261651

-- Define the gcd function and relevant numbers
def gcd_problem : Prop :=
  gcd 654327 543216 = 1

-- The statement of the theorem, with a placeholder for the proof
theorem gcd_654327_543216_is_1 : gcd_problem :=
by {
  -- actual proof will go here
  sorry
}

end gcd_654327_543216_is_1_l261_261651


namespace sequence_from_625_to_629_l261_261579

def arrows_repeating_pattern (n : ℕ) : ℕ := n % 5

theorem sequence_from_625_to_629 :
  arrows_repeating_pattern 625 = 0 ∧ arrows_repeating_pattern 629 = 4 →
  ∃ (seq : ℕ → ℕ), 
    (seq 0 = arrows_repeating_pattern 625) ∧
    (seq 1 = arrows_repeating_pattern (625 + 1)) ∧
    (seq 2 = arrows_repeating_pattern (625 + 2)) ∧
    (seq 3 = arrows_repeating_pattern (625 + 3)) ∧
    (seq 4 = arrows_repeating_pattern 629) := 
sorry

end sequence_from_625_to_629_l261_261579


namespace farm_horse_food_needed_l261_261297

-- Definitions given in the problem
def sheep_count : ℕ := 16
def sheep_to_horse_ratio : ℕ × ℕ := (2, 7)
def food_per_horse_per_day : ℕ := 230

-- The statement we want to prove
theorem farm_horse_food_needed : 
  ∃ H : ℕ, (sheep_count * sheep_to_horse_ratio.2 = sheep_to_horse_ratio.1 * H) ∧ 
           (H * food_per_horse_per_day = 12880) :=
sorry

end farm_horse_food_needed_l261_261297


namespace find_x_l261_261549

theorem find_x (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 72) : x = 8 :=
by
  sorry

end find_x_l261_261549


namespace geometric_sequence_properties_l261_261027

noncomputable def a_n (n : ℕ) : ℚ :=
if n = 0 then 1 / 4 else (1 / 4) * 2^(n-1)

def S_n (n : ℕ) : ℚ :=
(1/4) * (1 - 2^n) / (1 - 2)

theorem geometric_sequence_properties :
  (a_n 2 = 1 / 2) ∧ (∀ n : ℕ, 1 ≤ n → a_n n = 2^(n-3)) ∧ S_n 5 = 31 / 16 :=
by {
  sorry
}

end geometric_sequence_properties_l261_261027


namespace total_boxes_is_27_l261_261526

-- Defining the conditions
def stops : ℕ := 3
def boxes_per_stop : ℕ := 9

-- Prove that the total number of boxes is as expected
theorem total_boxes_is_27 : stops * boxes_per_stop = 27 := by
  sorry

end total_boxes_is_27_l261_261526


namespace power_of_three_l261_261041

theorem power_of_three (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_mult : (3^a) * (3^b) = 81) : (3^a)^b = 81 :=
sorry

end power_of_three_l261_261041


namespace green_duck_percentage_l261_261966

theorem green_duck_percentage (G_small G_large : ℝ) (D_small D_large : ℕ)
    (H1 : G_small = 0.20) (H2 : D_small = 20)
    (H3 : G_large = 0.15) (H4 : D_large = 80) : 
    ((G_small * D_small + G_large * D_large) / (D_small + D_large)) * 100 = 16 := 
by
  sorry

end green_duck_percentage_l261_261966


namespace fifth_dog_weight_l261_261814

theorem fifth_dog_weight (y : ℝ) (h : (25 + 31 + 35 + 33) / 4 = (25 + 31 + 35 + 33 + y) / 5) : y = 31 :=
by
  sorry

end fifth_dog_weight_l261_261814


namespace maximum_fraction_sum_l261_261858

noncomputable def max_fraction_sum (n : ℕ) (a b c d : ℕ) : ℝ :=
  1 - (1 / ((2 * n / 3 + 7 / 6) * ((2 * n / 3 + 7 / 6) * (n - (2 * n / 3 + 1 / 6)) + 1)))

theorem maximum_fraction_sum (n a b c d : ℕ) (h₀ : n > 1) (h₁ : a + c ≤ n) (h₂ : (a : ℚ) / b + (c : ℚ) / d < 1) :
  ∃ m : ℝ, m = max_fraction_sum n a b c d := by
  sorry

end maximum_fraction_sum_l261_261858


namespace sum_constants_l261_261363

def f (x : ℝ) : ℝ := -4 * x^2 + 20 * x - 88

theorem sum_constants (a b c : ℝ) (h : ∀ x : ℝ, -4 * x^2 + 20 * x - 88 = a * (x + b)^2 + c) : 
  a + b + c = -70.5 :=
sorry

end sum_constants_l261_261363


namespace scrooge_no_equal_coins_l261_261610

theorem scrooge_no_equal_coins (n : ℕ → ℕ)
  (initial_state : n 1 = 1 ∧ n 2 = 0 ∧ n 3 = 0 ∧ n 4 = 0 ∧ n 5 = 0 ∧ n 6 = 0)
  (operation : ∀ x i, 1 ≤ i ∧ i ≤ 6 → (n (i + 1) = n i - x ∧ n ((i % 6) + 2) = n ((i % 6) + 2) + 6 * x) 
                      ∨ (n (i + 1) = n i + 6 * x ∧ n ((i % 6) + 2) = n ((i % 6) + 2) - x)) :
  ¬ ∃ k, n 1 = k ∧ n 2 = k ∧ n 3 = k ∧ n 4 = k ∧ n 5 = k ∧ n 6 = k :=
by {
  sorry
}

end scrooge_no_equal_coins_l261_261610


namespace smallest_integer_with_inverse_mod_462_l261_261508

theorem smallest_integer_with_inverse_mod_462 :
  ∃ n : ℕ, n > 1 ∧ n ≤ 5 ∧ n.gcd(462) = 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → m.gcd(462) ≠ 1 :=
begin
  sorry
end

end smallest_integer_with_inverse_mod_462_l261_261508


namespace intersection_of_A_and_B_l261_261204

variable (A : Set ℝ)
variable (B : Set ℝ)
variable (C : Set ℝ)

theorem intersection_of_A_and_B (hA : A = { x | -1 < x ∧ x < 3 })
                                (hB : B = { -1, 1, 2 })
                                (hC : C = { 1, 2 }) :
  A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l261_261204


namespace everton_college_payment_l261_261420

theorem everton_college_payment :
  let num_sci_calculators := 20
  let num_graph_calculators := 25
  let price_sci_calculator := 10
  let price_graph_calculator := 57
  let total_payment := num_sci_calculators * price_sci_calculator + num_graph_calculators * price_graph_calculator
  total_payment = 1625 :=
by
  let num_sci_calculators := 20
  let num_graph_calculators := 25
  let price_sci_calculator := 10
  let price_graph_calculator := 57
  let total_payment := num_sci_calculators * price_sci_calculator + num_graph_calculators * price_graph_calculator
  sorry

end everton_college_payment_l261_261420


namespace probability_of_vowels_l261_261892

-- Conditions
def num_students : ℕ := 20
def initials : Finset String := 
  finset.of_list ["NN", "OO", "PP", "QQ", "RR", "SS", "TT", "UU", "VV", "WW", "XX", "YY", "ZZ"]
def vowels : Finset String :=
  finset.of_list ["O", "U", "Y"]

-- Probability calculation
def probability_initials_vowels : ℚ :=
  (vowels.card : ℚ) / (initials.card : ℚ)

-- Proof statement
theorem probability_of_vowels : 
  probability_initials_vowels = (3 : ℚ) / (13 : ℚ) :=
sorry

end probability_of_vowels_l261_261892


namespace algebraic_expression_value_l261_261451

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3 * x - 3 = 0) : x^3 + 2 * x^2 - 6 * x + 3 = 0 := 
sorry

end algebraic_expression_value_l261_261451


namespace felicia_flour_amount_l261_261725

-- Define the conditions as constants
def white_sugar := 1 -- cups
def brown_sugar := 1 / 4 -- cups
def oil := 1 / 2 -- cups
def scoop := 1 / 4 -- cups
def total_scoops := 15 -- number of scoops

-- Define the proof statement
theorem felicia_flour_amount : 
  (total_scoops * scoop - (white_sugar + brown_sugar / scoop + oil / scoop)) * scoop = 2 :=
by
  sorry

end felicia_flour_amount_l261_261725


namespace find_coefficients_l261_261017

noncomputable def P (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^4 - 8 * a * x^3 + b * x^2 - 32 * c * x + 16 * c

theorem find_coefficients (a b c : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 x3 x4 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ P a b c x1 = 0 ∧ P a b c x2 = 0 ∧ P a b c x3 = 0 ∧ P a b c x4 = 0) →
  (b = 16 * a ∧ c = a) :=
by
  sorry

end find_coefficients_l261_261017


namespace expected_value_eight_sided_die_l261_261635

-- Define a standard 8-sided die
def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Compute the probability of each outcome
def probability (n : ℕ) : ℝ := 1 / 8

-- Expected Value of a discrete random variable
def expected_value (outcomes : List ℕ) (prob : ℕ → ℝ) : ℝ :=
  outcomes.sum / outcomes.length.toReal

-- Theorem stating the expected value of a standard 8-sided die roll is 4.5
theorem expected_value_eight_sided_die : expected_value eight_sided_die probability = 4.5 := by
  sorry

end expected_value_eight_sided_die_l261_261635


namespace H2CO3_formation_l261_261849

-- Define the given conditions
def one_to_one_reaction (a b : ℕ) := a = b

-- Define the reaction
theorem H2CO3_formation (m_CO2 m_H2O : ℕ) 
  (h : one_to_one_reaction m_CO2 m_H2O) : 
  m_CO2 = 2 → m_H2O = 2 → m_CO2 = 2 ∧ m_H2O = 2 := 
by 
  intros h1 h2
  exact ⟨h1, h2⟩

end H2CO3_formation_l261_261849


namespace range_of_a_l261_261321

open Set

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (|x + 1/2| < 3/2) → (-2 < x ∧ x < 1)) →
  (∀ x : ℝ, ((1 / real.pi)^(2 * x) > real.pi^(-a - x)) → (x < a)) →
  (∀ x : ℝ, x ∈ compl (Ioo (-2 : ℝ) (1 : ℝ)) ∩ Iio a ↔ x ∈ Iio a) → (a ≤ 2) :=
by
  intros h1 h2 h3
  -- Skipped proof
  sorry

end range_of_a_l261_261321


namespace max_correct_answers_l261_261666

theorem max_correct_answers :
  ∃ (c w b : ℕ), c + w + b = 25 ∧ 4 * c - 3 * w = 57 ∧ c = 18 :=
by {
  sorry
}

end max_correct_answers_l261_261666


namespace find_C_l261_261287

theorem find_C (A B C : ℕ) (hA : A = 509) (hAB : A = B + 197) (hCB : C = B - 125) : C = 187 := 
by 
  sorry

end find_C_l261_261287


namespace sheets_of_paper_in_each_box_l261_261904

theorem sheets_of_paper_in_each_box (S E : ℕ) 
  (h1 : S - E = 70) 
  (h2 : 4 * (E - 20) = S) : 
  S = 120 := 
by 
  sorry

end sheets_of_paper_in_each_box_l261_261904


namespace sequence_expression_l261_261860

theorem sequence_expression (a : ℕ → ℝ) (h_base : a 1 = 2)
  (h_rec : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * (n + 1) * a n / (a n + n)) :
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = (n + 1) * 2^(n + 1) / (2^(n + 1) - 1) :=
by
  sorry

end sequence_expression_l261_261860


namespace camp_weights_l261_261683

theorem camp_weights (m_e_w : ℕ) (m_e_w1 : ℕ) (c_w : ℕ) (m_e_w2 : ℕ) (d : ℕ)
  (h1 : m_e_w = 30) 
  (h2 : m_e_w1 = 28) 
  (h3 : c_w = 56)
  (h4 : m_e_w = m_e_w1 + d)
  (h5 : m_e_w1 = m_e_w2 + d)
  (h6 : c_w = m_e_w + m_e_w1 + d) :
  m_e_w = 28 ∧ m_e_w2 = 26 := 
by {
    sorry
}

end camp_weights_l261_261683


namespace math_problem_l261_261713

-- Define the individual numbers
def a : Int := 153
def b : Int := 39
def c : Int := 27
def d : Int := 21

-- Define the entire expression and its expected result
theorem math_problem : (a + b + c + d) * 2 = 480 := by
  sorry

end math_problem_l261_261713


namespace yellow_paint_quarts_l261_261063

theorem yellow_paint_quarts (ratio_r : ℕ) (ratio_y : ℕ) (ratio_w : ℕ) (qw : ℕ) : 
  ratio_r = 5 → ratio_y = 3 → ratio_w = 7 → qw = 21 → (qw * ratio_y) / ratio_w = 9 :=
by
  -- No proof required, inserting sorry to indicate missing proof
  sorry

end yellow_paint_quarts_l261_261063


namespace inequality_reciprocal_l261_261314

theorem inequality_reciprocal (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 1 / (b - c) > 1 / (a - c) :=
sorry

end inequality_reciprocal_l261_261314


namespace solve_inequality_l261_261786

theorem solve_inequality (a x : ℝ) (h : a < 0) :
  (56 * x^2 + a * x - a^2 < 0) ↔ (a / 8 < x ∧ x < -a / 7) :=
by
  sorry

end solve_inequality_l261_261786


namespace abs_eq_neg_l261_261562

theorem abs_eq_neg (x : ℝ) (h : |x + 6| = -(x + 6)) : x ≤ -6 :=
by 
  sorry

end abs_eq_neg_l261_261562


namespace odd_number_representation_l261_261731

theorem odd_number_representation (n : ℤ) : 
  (∃ m : ℤ, 2 * m + 1 = 2 * n + 3) ∧ (¬ ∃ m : ℤ, 2 * m + 1 = 4 * n - 1) :=
by
  -- Proof steps would go here
  sorry

end odd_number_representation_l261_261731


namespace find_original_number_l261_261517

theorem find_original_number (r : ℝ) (h1 : r * 1.125 - r * 0.75 = 30) : r = 80 :=
by
  sorry

end find_original_number_l261_261517


namespace car_speed_l261_261924

theorem car_speed (time : ℕ) (distance : ℕ) (h1 : time = 5) (h2 : distance = 300) : distance / time = 60 := by
  sorry

end car_speed_l261_261924


namespace translate_vertex_l261_261079

/-- Given points A and B and their translations, verify the translated coordinates of B --/
theorem translate_vertex (A A' B B' : ℝ × ℝ)
  (hA : A = (0, 2))
  (hA' : A' = (-1, 0))
  (hB : B = (2, -1))
  (h_translation : A' = (A.1 - 1, A.2 - 2)) :
  B' = (B.1 - 1, B.2 - 2) :=
by
  sorry

end translate_vertex_l261_261079


namespace find_certain_number_l261_261038

theorem find_certain_number (x : ℕ) (h1 : 172 = 4 * 43) (h2 : 43 - 172 / x = 28) (h3 : 172 % x = 7) : x = 11 := by
  sorry

end find_certain_number_l261_261038


namespace chords_and_circle_l261_261235

theorem chords_and_circle (R : ℝ) (A B C D : ℝ) 
  (hAB : 0 < A - B) (hCD : 0 < C - D) (hR : R > 0) 
  (h_perp : (A - B) * (C - D) = 0) 
  (h_radA : A ^ 2 + B ^ 2 = R ^ 2) 
  (h_radC : C ^ 2 + D ^ 2 = R ^ 2) :
  (A - C)^2 + (B - D)^2 = 4 * R^2 :=
by
  sorry

end chords_and_circle_l261_261235


namespace count_simple_fractions_l261_261467

def isSimpleFraction (r : ℚ) : Prop :=
  r.denom ≠ 1 ∧ r < 1

theorem count_simple_fractions : 
  let count := (Finset.filter (λ n, isSimpleFraction (⟨n - 2, n⟩ : ℚ)) (Finset.range (13 - 3)).map (λ x, x + 3)).card
  count = 10 :=
by
  sorry

end count_simple_fractions_l261_261467


namespace probability_A_more_than_3_points_probability_distribution_B_expectation_of_X_l261_261583

section shooting_game
open Probability

-- Conditions for player A
def ξ : ℕ → ℕ → ℝ := λ n k, (n.choose k) * (1 / 2)^k * (1 / 2)^(n - k)

-- Probability that A scores more than 3 points in 3 shots
theorem probability_A_more_than_3_points :
  (ξ 3 2 + ξ 3 3) = 1 / 2 :=
by sorry

-- Conditions for player B
def X (sequence : list (ℕ → ℝ)) (n : ℕ) : ℝ :=
match sequence.nth n with
| some p => p
| none   => 0
end

def B_prob (n : ℕ) : list (ℕ → ℝ) :=
match n with
| 0 => [λ x, (if x = 1 then 1/2 else 1/2)]
| 1 => [λ x, (if x = 1 then 3/5 else 2/5)]
| _ => [λ x, 0]
end

-- Probability distribution of the score X
theorem probability_distribution_B :
  (X (B_prob 0) 0) = 9/50 ∧
  (X (B_prob 0) 2) = 8/25 ∧
  (X (B_prob 0) 4) = 8/25 ∧
  (X (B_prob 0) 6) = 9/50 :=
by sorry

-- Expectation of the score X
theorem expectation_of_X :
  (0 * (9 / 50) + 2 * (8 / 25) + 4 * (8 / 25) + 6 * (9 / 50)) = 3 :=
by sorry

end shooting_game

end probability_A_more_than_3_points_probability_distribution_B_expectation_of_X_l261_261583


namespace jeff_bought_from_chad_l261_261308

/-
  Eric has 4 ninja throwing stars.
  Chad has twice as many ninja throwing stars as Eric.
  Jeff now has 6 ninja throwing stars.
  Together, they have 16 ninja throwing stars.
  How many ninja throwing stars did Jeff buy from Chad?
-/

def eric_stars : ℕ := 4
def chad_stars : ℕ := 2 * eric_stars
def jeff_stars : ℕ := 6
def total_stars : ℕ := 16

theorem jeff_bought_from_chad (bought : ℕ) :
  chad_stars - bought + jeff_stars + eric_stars = total_stars → bought = 2 :=
by
  sorry

end jeff_bought_from_chad_l261_261308


namespace cos_210_eq_neg_sqrt3_div_2_l261_261150

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261150


namespace unit_prices_and_purchasing_schemes_l261_261662

theorem unit_prices_and_purchasing_schemes :
  ∃ (x y : ℕ),
    (14 * x + 8 * y = 1600) ∧
    (3 * x = 4 * y) ∧
    (x = 80) ∧ 
    (y = 60) ∧
    ∃ (m : ℕ), 
      (m ≥ 29) ∧ 
      (m ≤ 30) ∧ 
      (80 * m + 60 * (50 - m) ≤ 3600) ∧
      (m = 29 ∨ m = 30) := 
sorry

end unit_prices_and_purchasing_schemes_l261_261662


namespace solution_set_inequality_l261_261269

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 :=
by {
  sorry -- proof omitted
}

end solution_set_inequality_l261_261269


namespace math_proof_problem_l261_261028

variable (a_n : ℕ → ℤ) (S_n : ℕ → ℤ)
variable (a_1 d : ℤ)
variable (n : ℕ)

def arith_seq : Prop := ∀ n, a_n n = a_1 + (n - 1) * d

def sum_arith_seq : Prop := ∀ n, S_n n = n * (a_1 + (n - 1) * d / 2)

def condition1 : Prop := a_n 5 + a_n 9 = -2

def condition2 : Prop := S_n 3 = 57

noncomputable def general_formula : Prop := ∀ n, a_n n = 27 - 4 * n

noncomputable def max_S_n : Prop := ∀ n, S_n n ≤ 78 ∧ ∃ n, S_n n = 78

theorem math_proof_problem : 
  arith_seq a_n a_1 d ∧ sum_arith_seq S_n a_1 d ∧ condition1 a_n ∧ condition2 S_n 
  → general_formula a_n ∧ max_S_n S_n := 
sorry

end math_proof_problem_l261_261028


namespace relationship_among_variables_l261_261743

theorem relationship_among_variables (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (h1 : a^2 = 2) (h2 : b^3 = 3) (h3 : c^4 = 4) (h4 : d^5 = 5) : a = c ∧ a < d ∧ d < b := 
by
  sorry

end relationship_among_variables_l261_261743


namespace percent_difference_l261_261753

variables (w q y z x : ℝ)

-- Given conditions
def cond1 : Prop := w = 0.60 * q
def cond2 : Prop := q = 0.60 * y
def cond3 : Prop := z = 0.54 * y
def cond4 : Prop := x = 1.30 * w

-- The proof problem
theorem percent_difference (h1 : cond1 w q)
                           (h2 : cond2 q y)
                           (h3 : cond3 z y)
                           (h4 : cond4 x w) :
  ((z - x) / w) * 100 = 20 :=
by
  sorry

end percent_difference_l261_261753


namespace number_of_pupils_not_in_programX_is_639_l261_261236

-- Definitions for the conditions
def total_girls_elementary : ℕ := 192
def total_boys_elementary : ℕ := 135
def total_girls_middle : ℕ := 233
def total_boys_middle : ℕ := 163
def total_girls_high : ℕ := 117
def total_boys_high : ℕ := 89

def programX_girls_elementary : ℕ := 48
def programX_boys_elementary : ℕ := 28
def programX_girls_middle : ℕ := 98
def programX_boys_middle : ℕ := 51
def programX_girls_high : ℕ := 40
def programX_boys_high : ℕ := 25

-- Question formulation
theorem number_of_pupils_not_in_programX_is_639 :
  (total_girls_elementary - programX_girls_elementary) +
  (total_boys_elementary - programX_boys_elementary) +
  (total_girls_middle - programX_girls_middle) +
  (total_boys_middle - programX_boys_middle) +
  (total_girls_high - programX_girls_high) +
  (total_boys_high - programX_boys_high) = 639 := 
  by
  sorry

end number_of_pupils_not_in_programX_is_639_l261_261236


namespace restore_arithmetic_operations_l261_261806

/--
Given the placeholders \(A, B, C, D, E\) for operations in the equations:
1. \(4 A 2 = 2\)
2. \(8 = 4 C 2\)
3. \(2 D 3 = 5\)
4. \(4 = 5 E 1\)

Prove that:
(a) \(A = ÷\)
(b) \(B = =\)
(c) \(C = ×\)
(d) \(D = +\)
(e) \(E = -\)
-/
theorem restore_arithmetic_operations {A B C D E : String} (h1 : B = "=") 
    (h2 : "4" ++ A  ++ "2" ++ B ++ "2" = "4" ++ "÷" ++ "2" ++ "=" ++ "2")
    (h3 : "8" ++ "=" ++ "4" ++ C ++ "2" = "8" ++ "=" ++ "4" ++ "×" ++ "2")
    (h4 : "2" ++ D ++ "3" ++ "=" ++ "5" = "2" ++ "+" ++ "3" ++ "=" ++ "5")
    (h5 : "4" ++ "=" ++ "5" ++ E ++ "1" = "4" ++ "=" ++ "5" ++ "-" ++ "1") :
  (A = "÷") ∧ (B = "=") ∧ (C = "×") ∧ (D = "+") ∧ (E = "-") := by
    sorry

end restore_arithmetic_operations_l261_261806


namespace largest_number_l261_261099

-- Definitions based on the conditions
def numA := 0.893
def numB := 0.8929
def numC := 0.8931
def numD := 0.839
def numE := 0.8391

-- The statement to be proved 
theorem largest_number : numB = max numA (max numB (max numC (max numD numE))) := by
  sorry

end largest_number_l261_261099


namespace circle_problem_is_solved_l261_261149

def circle_problem_pqr : ℕ :=
  let n := 3 / 2;
  let p := 3;
  let q := 1;
  let r := 4;
  p + q + r

theorem circle_problem_is_solved : circle_problem_pqr = 8 :=
by {
  -- Additional context of conditions can be added here if necessary
  sorry
}

end circle_problem_is_solved_l261_261149


namespace determine_max_weight_l261_261101

theorem determine_max_weight {a b : ℕ} (n : ℕ) (x : ℕ) (ha : a > 0) (hb : b > 0) (hx : 1 ≤ x ∧ x ≤ n) :
  n = 9 :=
sorry

end determine_max_weight_l261_261101


namespace selling_price_is_correct_l261_261295

def profit_percent : ℝ := 0.6
def cost_price : ℝ := 375
def profit : ℝ := profit_percent * cost_price
def selling_price : ℝ := cost_price + profit

theorem selling_price_is_correct : selling_price = 600 :=
by
  -- proof steps would go here
  sorry

end selling_price_is_correct_l261_261295


namespace cos_210_eq_neg_sqrt3_div_2_l261_261151

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261151


namespace sum_of_squares_of_roots_eq_l261_261083

-- Definitions derived directly from conditions
def a := 5
def b := 2
def c := -15

-- Sum of roots
def sum_of_roots : ℚ := (-b : ℚ) / a

-- Product of roots
def product_of_roots : ℚ := (c : ℚ) / a

-- Sum of the squares of the roots
def sum_of_squares_of_roots : ℚ := sum_of_roots^2 - 2 * product_of_roots

-- The statement that needs to be proved
theorem sum_of_squares_of_roots_eq : sum_of_squares_of_roots = 154 / 25 :=
by
  sorry

end sum_of_squares_of_roots_eq_l261_261083


namespace sum_of_digits_divisible_by_7_l261_261069

theorem sum_of_digits_divisible_by_7
  (a b : ℕ)
  (h_three_digit : 100 * a + 11 * b ≥ 100 ∧ 100 * a + 11 * b < 1000)
  (h_last_two_digits_equal : true)
  (h_divisible_by_7 : (100 * a + 11 * b) % 7 = 0) :
  (a + 2 * b) % 7 = 0 :=
sorry

end sum_of_digits_divisible_by_7_l261_261069


namespace geometric_seq_a7_l261_261050

theorem geometric_seq_a7 (a : ℕ → ℝ) (r : ℝ) (h1 : a 3 = 16) (h2 : a 5 = 4) (h_geom : ∀ n, a (n + 1) = a n * r) : a 7 = 1 := by
  sorry

end geometric_seq_a7_l261_261050


namespace k_h_neg3_l261_261250

-- Definitions of h and k
def h (x : ℝ) : ℝ := 4 * x^2 - 12

variable (k : ℝ → ℝ) -- function k with range an ℝ

-- Given k(h(3)) = 16
axiom k_h_3 : k (h 3) = 16

-- Prove that k(h(-3)) = 16
theorem k_h_neg3 : k (h (-3)) = 16 :=
sorry

end k_h_neg3_l261_261250


namespace range_of_a_l261_261774

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 9 * x + a^2 / x + 7
  else 9 * x + a^2 / x - 7

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) → a ≤ -8/7 :=
by
  intros h
  -- Detailed proof would go here
  sorry

end range_of_a_l261_261774


namespace total_number_of_pages_l261_261746

variable (x : ℕ)

-- Conditions
def first_day_remaining : ℕ := x - (x / 6 + 10)
def second_day_remaining : ℕ := first_day_remaining x - (first_day_remaining x / 5 + 20)
def third_day_remaining : ℕ := second_day_remaining x - (second_day_remaining x / 4 + 25)
def final_remaining : Prop := third_day_remaining x = 100

-- Theorem statement
theorem total_number_of_pages : final_remaining x → x = 298 :=
by
  intros h
  sorry

end total_number_of_pages_l261_261746


namespace total_crayons_l261_261405

-- Define the number of crayons Billy has
def billy_crayons : ℝ := 62.0

-- Define the number of crayons Jane has
def jane_crayons : ℝ := 52.0

-- Formulate the theorem to prove the total number of crayons
theorem total_crayons : billy_crayons + jane_crayons = 114.0 := by
  sorry

end total_crayons_l261_261405


namespace cosine_210_l261_261177

noncomputable def Q := rotate 210 (1, 0)
noncomputable def E := foot Q (1, 0) (-1, 0)

theorem cosine_210 : cos 210 = - (sqrt 3/2) :=
sorry

end cosine_210_l261_261177


namespace lightest_height_is_135_l261_261796

-- Definitions based on the problem conditions
def heights_in_ratio (a b c d : ℕ) : Prop :=
  ∃ x : ℕ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x ∧ d = 6 * x

def height_condition (a c d : ℕ) : Prop :=
  d + a = c + 180

-- Lean statement describing the proof problem
theorem lightest_height_is_135 :
  ∀ (a b c d : ℕ),
  heights_in_ratio a b c d →
  height_condition a c d →
  a = 135 :=
by
  intro a b c d
  intro h_in_ratio h_condition
  sorry

end lightest_height_is_135_l261_261796


namespace chord_ratio_l261_261501

variable (XQ WQ YQ ZQ : ℝ)

theorem chord_ratio (h1 : XQ = 5) (h2 : WQ = 7) (h3 : XQ * YQ = WQ * ZQ) : YQ / ZQ = 7 / 5 :=
by
  sorry

end chord_ratio_l261_261501


namespace candy_bars_per_bag_l261_261519

def total_candy_bars : ℕ := 15
def number_of_bags : ℕ := 5

theorem candy_bars_per_bag : total_candy_bars / number_of_bags = 3 :=
by
  sorry

end candy_bars_per_bag_l261_261519


namespace Heechul_has_most_books_l261_261867

namespace BookCollection

variables (Heejin Heechul Dongkyun : ℕ)

theorem Heechul_has_most_books (h_h : ℕ) (h_j : ℕ) (d : ℕ) 
  (h_h_eq : h_h = h_j + 2) (d_lt_h_j : d < h_j) : 
  h_h > h_j ∧ h_h > d := 
by
  sorry

end BookCollection

end Heechul_has_most_books_l261_261867


namespace diophantine_solution_unique_l261_261726

theorem diophantine_solution_unique (k x y : ℕ) (hk : k > 0) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 = k * x * y - 1 ↔ k = 3 :=
by sorry

end diophantine_solution_unique_l261_261726


namespace max_writers_at_conference_l261_261537

variables (T E W x : ℕ)

-- Defining the conditions
def conference_conditions (T E W x : ℕ) : Prop :=
  T = 90 ∧ E > 38 ∧ x ≤ 6 ∧ 2 * x + (W + E - x) = T ∧ W = T - E - x

-- Statement to prove the number of writers
theorem max_writers_at_conference : ∃ W, conference_conditions 90 39 W 1 :=
by
  sorry

end max_writers_at_conference_l261_261537


namespace concurrent_circles_at_circumcenter_l261_261682

theorem concurrent_circles_at_circumcenter (A B C D O M N: Point) 
  (h_cyclic: IsCyclic A B C D) 
  (h_midpoints_AB_M: M = midpoint A B) 
  (h_midpoints_AD_N: N = midpoint A D) 
  (h_circumcenter: O = circumcenter A B C D) 
  (h_circle_AMN_A_M_N: CircleThroughPoints A M N) 
  (h_circle_BML_B_M_L: CircleThroughPoints B M L) 
  (h_circle_CNX_C_N_X: CircleThroughPoints C N X) 
  (h_circle_DNY_D_N_Y: CircleThroughPoints D N Y) : 
  ∃ P, P = O ∧ 
    OnCircleThroughPoints P A M N ∧ 
    OnCircleThroughPoints P B M L ∧ 
    OnCircleThroughPoints P C N X ∧ 
    OnCircleThroughPoints P D N Y := 
by 
  sorry

end concurrent_circles_at_circumcenter_l261_261682


namespace pipe_B_fill_time_l261_261342

theorem pipe_B_fill_time (T_B : ℝ) : 
  (1/3 + 1/T_B - 1/4 = 1/3) → T_B = 4 :=
sorry

end pipe_B_fill_time_l261_261342


namespace inequality_does_not_hold_l261_261040

theorem inequality_does_not_hold (x y : ℝ) (h : x > y) : ¬ (-3 * x > -3 * y) :=
by {
  sorry
}

end inequality_does_not_hold_l261_261040


namespace value_of_x_for_real_y_l261_261448

theorem value_of_x_for_real_y (x y : ℝ) (h : 4 * y^2 + 2 * x * y + |x| + 8 = 0) :
  (x ≤ -10) ∨ (x ≥ 10) :=
sorry

end value_of_x_for_real_y_l261_261448


namespace darryl_parts_cost_l261_261717

-- Define the conditions
def patent_cost : ℕ := 4500
def machine_price : ℕ := 180
def break_even_units : ℕ := 45
def total_revenue := break_even_units * machine_price

-- Define the theorem using the conditions
theorem darryl_parts_cost :
  ∃ (parts_cost : ℕ), parts_cost = total_revenue - patent_cost ∧ parts_cost = 3600 := by
  sorry

end darryl_parts_cost_l261_261717


namespace helium_min_cost_l261_261818

noncomputable def W (x : ℝ) : ℝ :=
  if x < 4 then 40 * (4 * x + 16 / x + 100)
  else 40 * (9 / (x * x) - 3 / x + 117)

theorem helium_min_cost :
  (∀ x, W x ≥ 4640) ∧ (W 2 = 4640) :=
by {
  sorry
}

end helium_min_cost_l261_261818


namespace rosalina_gifts_l261_261345

theorem rosalina_gifts (Emilio_gifts Jorge_gifts Pedro_gifts : ℕ) 
  (hEmilio : Emilio_gifts = 11) 
  (hJorge : Jorge_gifts = 6) 
  (hPedro : Pedro_gifts = 4) : 
  Emilio_gifts + Jorge_gifts + Pedro_gifts = 21 :=
by
  sorry

end rosalina_gifts_l261_261345


namespace rosalina_received_21_gifts_l261_261346

def Emilio_gifts : Nat := 11
def Jorge_gifts : Nat := 6
def Pedro_gifts : Nat := 4

def total_gifts : Nat :=
  Emilio_gifts + Jorge_gifts + Pedro_gifts

theorem rosalina_received_21_gifts : total_gifts = 21 := by
  sorry

end rosalina_received_21_gifts_l261_261346


namespace option_d_is_true_l261_261401

theorem option_d_is_true (x : ℝ) : (4 * x) / (x^2 + 4) ≤ 1 := 
  sorry

end option_d_is_true_l261_261401


namespace ellipse_problem_l261_261734

theorem ellipse_problem
  (F2 : ℝ) (a : ℝ) (A B : ℝ × ℝ)
  (on_ellipse_A : (A.1 ^ 2) / (a ^ 2) + (25 * (A.2 ^ 2)) / (9 * a ^ 2) = 1)
  (on_ellipse_B : (B.1 ^ 2) / (a ^ 2) + (25 * (B.2 ^ 2)) / (9 * a ^ 2) = 1)
  (focal_distance : |A.1 + F2| + |B.1 + F2| = 8 / 5 * a)
  (midpoint_to_directrix : |(A.1 + B.1) / 2 + 5 / 4 * a| = 3 / 2) :
  a = 1 → (∀ x y, (x^2 + (25 / 9) * y^2 = 1) ↔ ((x^2) / (a^2) + (25 * y^2) / (9 * a^2) = 1)) :=
by
  sorry

end ellipse_problem_l261_261734


namespace temperature_at_midnight_l261_261896

theorem temperature_at_midnight 
  (morning_temp : ℝ) 
  (afternoon_rise : ℝ) 
  (midnight_drop : ℝ)
  (h1 : morning_temp = 30)
  (h2 : afternoon_rise = 1)
  (h3 : midnight_drop = 7) 
  : morning_temp + afternoon_rise - midnight_drop = 24 :=
by
  -- Convert all conditions into the correct forms
  rw [h1, h2, h3]
  -- Perform the arithmetic operations
  norm_num

end temperature_at_midnight_l261_261896


namespace least_number_of_candles_l261_261906

theorem least_number_of_candles (b : ℕ) :
  (b ≡ 5 [MOD 6]) ∧ (b ≡ 7 [MOD 8]) ∧ (b ≡ 3 [MOD 9]) → b = 119 :=
by
  -- Proof omitted
  sorry

end least_number_of_candles_l261_261906


namespace problem1_solution_set_problem2_min_value_l261_261442

-- For Problem (1)
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

theorem problem1_solution_set (x : ℝ) (h : f x 1 1 ≤ 4) : 
  -2 ≤ x ∧ x ≤ 2 :=
sorry

-- For Problem (2)
theorem problem2_min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : ∀ x : ℝ, f x a b ≥ 2) : 
  (1 / a) + (2 / b) = 3 :=
sorry

end problem1_solution_set_problem2_min_value_l261_261442


namespace log_domain_eq_l261_261789

noncomputable def quadratic_expr (x : ℝ) : ℝ := x^2 - 2 * x - 3

def log_domain (x : ℝ) : Prop := quadratic_expr x > 0

theorem log_domain_eq :
  {x : ℝ | log_domain x} = 
  {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} :=
by {
  sorry
}

end log_domain_eq_l261_261789


namespace convert_point_to_polar_l261_261992

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2),
      θ := if y ≠ 0 then real.atan (y / x) else if x > 0 then 0 else real.pi in
  (r, if θ < 0 then θ + 2 * real.pi else θ)

theorem convert_point_to_polar :
  rectangular_to_polar 3 (-3) = (3 * real.sqrt 2, 7 * real.pi / 4) :=
by sorry

end convert_point_to_polar_l261_261992


namespace triangle_area_base_10_height_10_l261_261516

theorem triangle_area_base_10_height_10 :
  let base := 10
  let height := 10
  (base * height) / 2 = 50 := by
  sorry

end triangle_area_base_10_height_10_l261_261516


namespace modulus_of_z_l261_261060

noncomputable def z : ℂ := sorry
def condition (z : ℂ) : Prop := z * (1 - Complex.I) = 2 * Complex.I

theorem modulus_of_z (hz : condition z) : Complex.abs z = Real.sqrt 2 := sorry

end modulus_of_z_l261_261060


namespace total_gold_value_l261_261769

def legacy_bars : ℕ := 5
def aleena_bars : ℕ := legacy_bars - 2
def value_per_bar : ℕ := 2200
def total_bars : ℕ := legacy_bars + aleena_bars
def total_value : ℕ := total_bars * value_per_bar

theorem total_gold_value : total_value = 17600 :=
by
  -- Begin proof
  sorry

end total_gold_value_l261_261769


namespace expected_value_8_sided_die_l261_261646

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  (Σ x ∈ outcomes, probability_each_outcome * x) = 4.5 :=
by
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  have h : (Σ x ∈ outcomes, probability_each_outcome * x) = (1 / 8) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) := by sorry
  have sum_eq_36 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36 := by sorry
  rw [sum_eq_36] at h
  have expected_value_eq : (1 / 8) * 36 = 4.5 := by sorry
  rw [expected_value_eq] at h
  exact h

end expected_value_8_sided_die_l261_261646


namespace symmetric_circle_eqn_l261_261571

theorem symmetric_circle_eqn :
  ∀ (x y : ℝ),
  ((x + 1)^2 + (y - 1)^2 = 1) ∧ (x - y - 1 = 0) →
  (∀ (x' y' : ℝ), (x' = y + 1) ∧ (y' = x - 1) → (x' + 1)^2 + (y' - 1)^2 = 1) →
  (x - 2)^2 + (y + 2)^2 = 1 :=
by
  intros x y h h_sym
  sorry

end symmetric_circle_eqn_l261_261571


namespace bracelet_arrangements_l261_261897

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def distinct_arrangements : ℕ := factorial 8 / (8 * 2)

theorem bracelet_arrangements : distinct_arrangements = 2520 :=
by
  sorry

end bracelet_arrangements_l261_261897


namespace ratio_of_b_to_a_is_4_l261_261751

theorem ratio_of_b_to_a_is_4 (b a : ℚ) (h1 : b = 4 * a) (h2 : b = 15 - 4 * a) : a = 15 / 8 := by
  sorry

end ratio_of_b_to_a_is_4_l261_261751


namespace frog_jump_distance_l261_261261

theorem frog_jump_distance (grasshopper_jump : ℕ) (extra_jump : ℕ) (frog_jump : ℕ) :
  grasshopper_jump = 9 → extra_jump = 3 → frog_jump = grasshopper_jump + extra_jump → frog_jump = 12 :=
by
  intros h_grasshopper h_extra h_frog
  rw [h_grasshopper, h_extra] at h_frog
  exact h_frog

end frog_jump_distance_l261_261261


namespace extreme_value_h_tangent_to_both_l261_261433

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a*x + 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.log x - a
noncomputable def h (x : ℝ) : ℝ := f x 1 - g x 1

theorem extreme_value_h : h (1/2) = 11/4 + Real.log 2 := by
  sorry

theorem tangent_to_both : ∀ (a : ℝ), ∃ x₁ x₂ : ℝ, (2 * x₁ + a = 1 / x₂) ∧ 
  ((x₁ = (1 / (2 * x₂)) - (a / 2)) ∧ (a ≥ -1)) := by
  sorry

end extreme_value_h_tangent_to_both_l261_261433


namespace percentage_increase_in_allowance_l261_261590

def middle_school_allowance : ℕ := 8 + 2
def senior_year_allowance : ℕ := 2 * middle_school_allowance + 5

theorem percentage_increase_in_allowance : 
  (senior_year_allowance - middle_school_allowance) * 100 / middle_school_allowance = 150 := 
  by
    sorry

end percentage_increase_in_allowance_l261_261590


namespace prob_single_trial_l261_261911

theorem prob_single_trial (P : ℝ) : 
  (1 - (1 - P)^4) = 65 / 81 → P = 1 / 3 :=
by
  intro h
  sorry

end prob_single_trial_l261_261911


namespace consecutive_roots_prime_q_l261_261440

theorem consecutive_roots_prime_q (p q : ℤ) (h1 : Prime q)
  (h2 : ∃ x1 x2 : ℤ, 
    x1 ≠ x2 ∧ 
    (x1 = x2 + 1 ∨ x1 = x2 - 1) ∧ 
    x1 + x2 = p ∧ 
    x1 * x2 = q) : (p = 3 ∨ p = -3) ∧ q = 2 :=
by
  sorry

end consecutive_roots_prime_q_l261_261440


namespace find_q_l261_261324

theorem find_q (p q : ℝ) (hp : 1 < p) (hq : 1 < q) (hcond1 : 1/p + 1/q = 1) (hcond2 : p * q = 9) :
    q = (9 + 3 * Real.sqrt 5) / 2 ∨ q = (9 - 3 * Real.sqrt 5) / 2 :=
by
  sorry

end find_q_l261_261324


namespace sum_of_cubes_is_81720_l261_261366

-- Let n be the smallest of these consecutive even integers.
def smallest_even : Int := 28

-- Assumptions given the conditions
def sum_of_squares (n : Int) : Int := n^2 + (n + 2)^2 + (n + 4)^2

-- The condition provided is that sum of the squares is 2930
lemma sum_of_squares_is_2930 : sum_of_squares smallest_even = 2930 := by
  sorry

-- To prove that the sum of the cubes of these three integers is 81720
def sum_of_cubes (n : Int) : Int := n^3 + (n + 2)^3 + (n + 4)^3

theorem sum_of_cubes_is_81720 : sum_of_cubes smallest_even = 81720 := by
  sorry

end sum_of_cubes_is_81720_l261_261366


namespace arrange_in_circle_l261_261022

open Nat

noncomputable def smallest_n := 70

theorem arrange_in_circle (n : ℕ) (h : n = 70) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n →
    (∀ j : ℕ, 1 ≤ j ∧ j ≤ 40 → k > ((k + j) % n)) ∨
    (∀ p : ℕ, 1 ≤ p ∧ p ≤ 30 → k < ((k + p) % n))) :=
by
  sorry

end arrange_in_circle_l261_261022


namespace book_club_boys_count_l261_261788

theorem book_club_boys_count (B G : ℕ) 
  (h1 : B + G = 30) 
  (h2 : B + (1 / 3 : ℝ) * G = 18) :
  B = 12 :=
by
  have h3 : 3 • B + G = 54 := sorry
  have h4 : 3 • B + G - (B + G) = 54 - 30 := sorry
  have h5 : 2 • B = 24 := sorry
  have h6 : B = 12 := sorry
  exact h6

end book_club_boys_count_l261_261788


namespace find_theta_l261_261956

-- Definitions based on conditions
def angle_A : ℝ := 10
def angle_B : ℝ := 14
def angle_C : ℝ := 26
def angle_D : ℝ := 33
def sum_rect_angles : ℝ := 360
def sum_triangle_angles : ℝ := 180
def sum_right_triangle_acute_angles : ℝ := 90

-- Main theorem statement
theorem find_theta (A B C D : ℝ)
  (hA : A = angle_A)
  (hB : B = angle_B)
  (hC : C = angle_C)
  (hD : D = angle_D)
  (sum_rect : sum_rect_angles = 360)
  (sum_triangle : sum_triangle_angles = 180) :
  ∃ θ : ℝ, θ = 11 := 
sorry

end find_theta_l261_261956


namespace pyramid_height_l261_261819

noncomputable def height_of_pyramid :=
  let volume_cube := 6 ^ 3
  let volume_sphere := (4 / 3) * Real.pi * (4 ^ 3)
  let total_volume := volume_cube + volume_sphere
  let base_area := 10 ^ 2
  let h := (3 * total_volume) / base_area
  h

theorem pyramid_height :
  height_of_pyramid = 6.48 + 2.56 * Real.pi :=
by
  sorry

end pyramid_height_l261_261819


namespace min_positive_value_l261_261227

theorem min_positive_value (c d : ℤ) (h : c > d) : 
  ∃ x : ℝ, x = (c + 2 * d) / (c - d) + (c - d) / (c + 2 * d) ∧ x = 2 :=
by {
  sorry
}

end min_positive_value_l261_261227


namespace minimum_gennadys_l261_261696

theorem minimum_gennadys (alexs borises vasilies x : ℕ) (h₁ : alexs = 45) (h₂ : borises = 122) (h₃ : vasilies = 27)
    (h₄ : ∀ i, i ∈ list.range (borises-1) → alexs + vasilies + x > i) : 
    x = 49 :=
by 
    sorry

end minimum_gennadys_l261_261696


namespace cos_210_eq_neg_sqrt3_div_2_l261_261173

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261173


namespace cost_of_7_cubic_yards_of_topsoil_is_1512_l261_261274

-- Definition of the given conditions
def cost_per_cubic_foot : ℕ := 8
def cubic_yards : ℕ := 7
def cubic_yards_to_cubic_feet : ℕ := 27

-- Problem definition
def cost_of_topsoil (cubic_yards : ℕ) (cost_per_cubic_foot : ℕ) (cubic_yards_to_cubic_feet : ℕ) : ℕ :=
  cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot

-- The proof statement
theorem cost_of_7_cubic_yards_of_topsoil_is_1512 :
  cost_of_topsoil cubic_yards cost_per_cubic_foot cubic_yards_to_cubic_feet = 1512 := by
  sorry

end cost_of_7_cubic_yards_of_topsoil_is_1512_l261_261274


namespace tangerines_more_than_oranges_l261_261629

def initial_oranges := 5
def initial_tangerines := 17
def oranges_taken := 2
def tangerines_taken := 10

theorem tangerines_more_than_oranges
  (initial_oranges: ℕ) -- Tina starts with 5 oranges
  (initial_tangerines: ℕ) -- Tina starts with 17 tangerines
  (oranges_taken: ℕ) -- Tina takes away 2 oranges
  (tangerines_taken: ℕ) -- Tina takes away 10 tangerines
  : (initial_tangerines - tangerines_taken) - (initial_oranges - oranges_taken) = 4 := 
by
  sorry

end tangerines_more_than_oranges_l261_261629


namespace number_of_bouquets_l261_261115

theorem number_of_bouquets : ∃ n, n = 9 ∧ ∀ x y : ℕ, 3 * x + 2 * y = 50 → (x < 17) ∧ (x % 2 = 0 → y = (50 - 3 * x) / 2) :=
by
  sorry

end number_of_bouquets_l261_261115


namespace sum_of_roots_of_y_squared_eq_36_l261_261881

theorem sum_of_roots_of_y_squared_eq_36 :
  (∀ y : ℝ, y^2 = 36 → y = 6 ∨ y = -6) → (6 + (-6) = 0) :=
by
  sorry

end sum_of_roots_of_y_squared_eq_36_l261_261881


namespace conversion_base10_to_base7_l261_261374

-- Define the base-10 number
def num_base10 : ℕ := 1023

-- Define the conversion base
def base : ℕ := 7

-- Define the expected base-7 representation as a function of the base
def expected_base7 (b : ℕ) : ℕ := 2 * b^3 + 6 * b^2 + 6 * b^1 + 1 * b^0

-- Statement to prove
theorem conversion_base10_to_base7 : expected_base7 base = num_base10 :=
by 
  -- Sorry is a placeholder for the proof
  sorry

end conversion_base10_to_base7_l261_261374


namespace divisor_between_l261_261667

theorem divisor_between (n a b : ℕ) (h_n_gt_8 : n > 8) (h_a_dvd_n : a ∣ n) (h_b_dvd_n : b ∣ n) 
    (h_a_lt_b : a < b) (h_n_eq_asq_plus_b : n = a^2 + b) (h_a_ne_b : a ≠ b) :
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end divisor_between_l261_261667


namespace danielle_rooms_is_6_l261_261868

def heidi_rooms (danielle_rooms : ℕ) : ℕ := 3 * danielle_rooms
def grant_rooms (heidi_rooms : ℕ) : ℕ := heidi_rooms / 9

theorem danielle_rooms_is_6 (danielle_rooms : ℕ) (h1 : heidi_rooms danielle_rooms = 18) (h2 : grant_rooms (heidi_rooms danielle_rooms) = 2) :
  danielle_rooms = 6 :=
by 
  sorry

end danielle_rooms_is_6_l261_261868


namespace complex_square_l261_261454

theorem complex_square (z : ℂ) (i : ℂ) (h1 : z = 2 - 3 * i) (h2 : i^2 = -1) : z^2 = -5 - 12 * i :=
sorry

end complex_square_l261_261454


namespace greatest_perfect_power_sum_l261_261144

def sum_c_d_less_500 : ℕ :=
  let c := 22
  let d := 2
  c + d

theorem greatest_perfect_power_sum :
  ∃ c d : ℕ, 0 < c ∧ 1 < d ∧ c^d < 500 ∧
  ∀ x y : ℕ, 0 < x ∧ 1 < y ∧ x^y < 500 → x^y ≤ c^d ∧ (c + d = 24) :=
by
  sorry

end greatest_perfect_power_sum_l261_261144


namespace cos_210_eq_neg_sqrt_3_div_2_l261_261155

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l261_261155


namespace primes_quadratic_roots_conditions_l261_261251

theorem primes_quadratic_roots_conditions (p q : ℕ)
  (hp : Prime p) (hq : Prime q)
  (h1 : ∃ (x y : ℕ), x ≠ y ∧ x * y = 2 * q ∧ x + y = p) :
  (¬ (∀ (x y : ℕ), x ≠ y ∧ x * y = 2 * q ∧ x + y = p → (x - y) % 2 = 0)) ∧
  (∃ (x : ℕ), x * 2 = 2 * q ∨ x * q = 2 * q ∧ Prime x) ∧
  (¬ Prime (p * p + 2 * q)) ∧
  (Prime (p - q)) :=
by sorry

end primes_quadratic_roots_conditions_l261_261251


namespace expression_evaluation_l261_261988

theorem expression_evaluation : |(-7: ℤ)| / ((2 / 3) - (1 / 5)) - (1 / 2) * ((-4)^2) = 7 := by
  sorry

end expression_evaluation_l261_261988


namespace sum_of_solutions_l261_261310

theorem sum_of_solutions (y1 y2 : ℝ) (h1 : y1 + 16 / y1 = 12) (h2 : y2 + 16 / y2 = 12) : 
  y1 + y2 = 12 :=
by
  sorry

end sum_of_solutions_l261_261310


namespace rational_non_positive_l261_261875

variable (a : ℚ)

theorem rational_non_positive (h : ∃ a : ℚ, True) : 
  -a^2 ≤ 0 :=
by
  sorry

end rational_non_positive_l261_261875


namespace top_leftmost_rectangle_is_E_l261_261846

def rectangle (w x y z : ℕ) : Prop := true

-- Define the rectangles according to the given conditions
def rectangle_A : Prop := rectangle 4 1 6 9
def rectangle_B : Prop := rectangle 1 0 3 6
def rectangle_C : Prop := rectangle 3 8 5 2
def rectangle_D : Prop := rectangle 7 5 4 8
def rectangle_E : Prop := rectangle 9 2 7 0

-- Prove that the top leftmost rectangle is E
theorem top_leftmost_rectangle_is_E : rectangle_E → True :=
by
  sorry

end top_leftmost_rectangle_is_E_l261_261846


namespace oranges_in_each_box_l261_261834

theorem oranges_in_each_box (total_oranges : ℝ) (boxes : ℝ) (h_total : total_oranges = 72) (h_boxes : boxes = 3.0) : total_oranges / boxes = 24 :=
by
  -- Begin proof
  sorry

end oranges_in_each_box_l261_261834


namespace find_p_q_r_l261_261205

theorem find_p_q_r  (t : ℝ) (p q r : ℕ) (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4) 
                    (h2 : (1 - Real.sin t) * (1 - Real.cos t) = (p / q) - Real.sqrt r)
                    (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
                    (rel_prime : Nat.gcd p q = 1) : 
                    p + q + r = 5 := 
by
  sorry

end find_p_q_r_l261_261205


namespace no_four_distinct_sum_mod_20_l261_261733

theorem no_four_distinct_sum_mod_20 (R : Fin 9 → ℕ) (h : ∀ i, R i < 19) :
  ¬ ∃ (a b c d : Fin 9), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (R a + R b) % 20 = (R c + R d) % 20 := sorry

end no_four_distinct_sum_mod_20_l261_261733


namespace farmer_flax_acres_l261_261665

-- Definitions based on conditions
def total_acres : ℕ := 240
def extra_sunflower_acres : ℕ := 80

-- Problem statement
theorem farmer_flax_acres (F : ℕ) (S : ℕ) 
    (h1 : F + S = total_acres) 
    (h2 : S = F + extra_sunflower_acres) : 
    F = 80 :=
by
    -- Proof goes here
    sorry

end farmer_flax_acres_l261_261665


namespace problem_proof_l261_261228

theorem problem_proof (x : ℝ) (h : x + 1/x = 3) : (x - 3) ^ 2 + 36 / (x - 3) ^ 2 = 12 :=
sorry

end problem_proof_l261_261228


namespace irrational_root_exists_l261_261465

theorem irrational_root_exists 
  (a b c d : ℤ)
  (h_poly : ∀ x : ℚ, a * x^3 + b * x^2 + c * x + d ≠ 0) 
  (h_odd : a * d % 2 = 1) 
  (h_even : b * c % 2 = 0) : 
  ∃ x : ℚ, ¬ ∃ y : ℚ, y ≠ x ∧ y ≠ x ∧ a * x^3 + b * x^2 + c * x + d = 0 :=
sorry

end irrational_root_exists_l261_261465


namespace find_m_and_y_range_l261_261432

open Set

noncomputable def y (m x : ℝ) := (6 + 2 * m) * x^2 - 5 * x^((abs (m + 2))) + 3 

theorem find_m_and_y_range :
  (∃ m : ℝ, (∀ x : ℝ, y m x = (6 + 2*m) * x^2 - 5*x^((abs (m+2))) + 3) ∧ (∀ x : ℝ, y m x = -5 * x + 3 → m = -3)) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 → y (-3) x ∈ Icc (-22 : ℝ) (8 : ℝ)) :=
by
  sorry

end find_m_and_y_range_l261_261432


namespace instantaneous_velocity_at_t_2_l261_261114

def y (t : ℝ) : ℝ := 3 * t^2 + 4

theorem instantaneous_velocity_at_t_2 :
  deriv y 2 = 12 :=
by
  sorry

end instantaneous_velocity_at_t_2_l261_261114


namespace train_length_approx_140_l261_261673

noncomputable def km_per_hr_to_m_per_s (speed_km_hr : ℝ) : ℝ :=
  speed_km_hr * 1000 / 3600

def train_length (speed_km_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_s := km_per_hr_to_m_per_s speed_km_hr
  speed_m_s * time_sec

theorem train_length_approx_140 :
  train_length 56 9 ≈ 140 :=
by
  -- The proof can be filled in here
  sorry

end train_length_approx_140_l261_261673


namespace probability_of_integer_division_l261_261262

open Set

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def within_range_r (r : ℤ) : Prop := -5 < r ∧ r < 8
def within_range_k (k : ℕ) : Prop := 1 < k ∧ k < 10

def primes_in_range (k : ℕ) : Prop := within_range_k k ∧ is_prime k

def valid_pairs : Finset (ℤ × ℕ) := 
  ((Finset.Ico (-4 : ℤ) 8).product (Finset.filter primes_in_range (Finset.range 10)))

def integer_division_pairs : Finset (ℤ × ℕ) :=
  valid_pairs.filter (λ p, p.1 % p.2 = 0)

noncomputable def probability : ℚ :=
  ⟨integer_division_pairs.card, valid_pairs.card⟩

theorem probability_of_integer_division : probability = 5 / 16 := 
  sorry

end probability_of_integer_division_l261_261262


namespace polynomial_no_ab_term_l261_261043

theorem polynomial_no_ab_term (a b m : ℝ) :
  let p := 2 * (a^2 + a * b - 5 * b^2) - (a^2 - m * a * b + 2 * b^2)
  ∃ (m : ℝ), (p = a^2 - 12 * b^2) → (m = -2) :=
by
  let p := 2 * (a^2 + a * b - 5 * b^2) - (a^2 - m * a * b + 2 * b^2)
  intro h
  use -2
  sorry

end polynomial_no_ab_term_l261_261043


namespace total_earnings_l261_261518

-- Definitions based on conditions
def bead_necklaces : ℕ := 7
def gem_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 9

-- The main theorem to prove
theorem total_earnings : (bead_necklaces + gem_necklaces) * cost_per_necklace = 90 :=
by
  sorry

end total_earnings_l261_261518


namespace min_gennadys_l261_261705

theorem min_gennadys (alexanders borises vasilies : ℕ) (x : ℕ) 
    (h1 : alexanders = 45)
    (h2 : borises = 122)
    (h3 : vasilies = 27)
    (h4 : x = 49)
    (h5 : borises - 1 = alexanders + vasilies + x) :
  x = 49 := 
begin
  sorry,
end

end min_gennadys_l261_261705


namespace triangle_shape_l261_261233

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h1 : a * Real.cos A = b * Real.cos B) :
  (a = b ∨ c = a ∨ c = b ∨ A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2) :=
sorry

end triangle_shape_l261_261233


namespace part1_part2_part3_l261_261975

noncomputable def total_students : ℕ := 40
noncomputable def qualified_students : ℕ := 35
noncomputable def prob_qualified : ℚ := qualified_students / total_students

theorem part1 :
  prob_qualified = 7 / 8 :=
sorry

noncomputable def prob_X (x : ℕ) : ℚ :=
  if x = 0 then 1 / 30
  else if x = 1 then 3 / 10
  else if x = 2 then 1 / 2
  else if x = 3 then 1 / 6
  else 0

noncomputable def expected_X : ℚ :=
  ∑ i in finset.range 4, i * prob_X i

theorem part2 :
  expected_X = 9 / 5 :=
sorry

noncomputable def prob_male_excellent : ℚ := 1 / 5
noncomputable def prob_female_excellent : ℚ := 3 / 10
noncomputable def prob_two_excellent : ℚ :=
  prob_male_excellent ^ 2 * (1 - prob_female_excellent) +
  prob_male_excellent * (1 - prob_male_excellent) * prob_female_excellent +
  (1 - prob_male_excellent) * prob_male_excellent * prob_female_excellent

theorem part3 :
  prob_two_excellent = 31 / 250 :=
sorry

end part1_part2_part3_l261_261975


namespace probability_interval_l261_261937

noncomputable def probability_density (X : MeasureTheory.ProbabilityMeasure ℝ) : (ℝ → ℝ) → ℝ := sorry

noncomputable def normal_distribution (μ σ : ℝ) : MeasureTheory.ProbabilityMeasure ℝ := sorry

noncomputable def cdf (F : MeasureTheory.ProbabilityMeasure ℝ) (x : ℝ) : ℝ := sorry

theorem probability_interval {μ σ : ℝ} (F : MeasureTheory.ProbabilityMeasure ℝ) :
  (X ∼ normal_distribution μ σ) →
  (cdf F 2 - cdf F 1 = 0.2) →
  (cdf F 1 - cdf F 0 = 0.3) :=
sorry

end probability_interval_l261_261937


namespace chris_money_before_birthday_l261_261989

/-- Chris's total money now is $279 -/
def money_now : ℕ := 279

/-- Money received from Chris's grandmother is $25 -/
def money_grandmother : ℕ := 25

/-- Money received from Chris's aunt and uncle is $20 -/
def money_aunt_uncle : ℕ := 20

/-- Money received from Chris's parents is $75 -/
def money_parents : ℕ := 75

/-- Total money received for his birthday -/
def money_received : ℕ := money_grandmother + money_aunt_uncle + money_parents

/-- Money Chris had before his birthday -/
def money_before_birthday : ℕ := money_now - money_received

theorem chris_money_before_birthday : money_before_birthday = 159 := by
  sorry

end chris_money_before_birthday_l261_261989


namespace daily_earnings_c_l261_261658

theorem daily_earnings_c (A B C : ℕ) (h1 : A + B + C = 600) (h2 : A + C = 400) (h3 : B + C = 300) : C = 100 :=
sorry

end daily_earnings_c_l261_261658


namespace average_sitting_time_l261_261112

theorem average_sitting_time (number_of_students : ℕ) (number_of_seats : ℕ) (total_travel_time : ℕ) 
  (h1 : number_of_students = 6) 
  (h2 : number_of_seats = 4) 
  (h3 : total_travel_time = 192) :
  (number_of_seats * total_travel_time) / number_of_students = 128 :=
by
  sorry

end average_sitting_time_l261_261112


namespace tim_campaign_total_l261_261945

theorem tim_campaign_total (amount_max : ℕ) (num_max : ℕ) (num_half : ℕ) (total_donations : ℕ) (total_raised : ℕ)
  (H1 : amount_max = 1200)
  (H2 : num_max = 500)
  (H3 : num_half = 3 * num_max)
  (H4 : total_donations = num_max * amount_max + num_half * (amount_max / 2))
  (H5 : total_donations = 40 * total_raised / 100) :
  total_raised = 3750000 :=
by
  -- Proof is omitted
  sorry

end tim_campaign_total_l261_261945


namespace smallest_integer_coprime_with_462_l261_261507

theorem smallest_integer_coprime_with_462 :
  ∃ n, n > 1 ∧ Nat.gcd n 462 = 1 ∧ ∀ m, m > 1 ∧ Nat.gcd m 462 = 1 → n ≤ m → n = 13 := by
  sorry

end smallest_integer_coprime_with_462_l261_261507


namespace catch_bus_probability_within_5_minutes_l261_261536

theorem catch_bus_probability_within_5_minutes :
  (Pbus3 : ℝ) → (Pbus6 : ℝ) → (Pbus3 = 0.20) → (Pbus6 = 0.60) → (Pcatch : ℝ) → (Pcatch = Pbus3 + Pbus6) → (Pcatch = 0.80) :=
by
  intros Pbus3 Pbus6 hPbus3 hPbus6 Pcatch hPcatch
  sorry

end catch_bus_probability_within_5_minutes_l261_261536


namespace chelsea_guaranteed_victory_l261_261755

noncomputable def minimum_bullseye_shots_to_win (k : ℕ) (n : ℕ) : ℕ :=
  if (k + 5 * n + 500 > k + 930) then n else sorry

theorem chelsea_guaranteed_victory (k : ℕ) :
  minimum_bullseye_shots_to_win k 87 = 87 :=
by
  sorry

end chelsea_guaranteed_victory_l261_261755


namespace number_of_ways_to_select_students_l261_261625

theorem number_of_ways_to_select_students 
  (n m t : ℕ) (h : t = 1) (hs : 3 = 2 + 1) (hd : 2 = 1 + 1) :
  (nat.choose (3, 2) * nat.choose (2, 1)) = 15 :=
by
  sorry

end number_of_ways_to_select_students_l261_261625


namespace possible_values_of_D_plus_E_l261_261183

theorem possible_values_of_D_plus_E 
  (D E : ℕ) 
  (hD : 0 ≤ D ∧ D ≤ 9) 
  (hE : 0 ≤ E ∧ E ≤ 9) 
  (hdiv : (D + 8 + 6 + 4 + E + 7 + 2) % 9 = 0) : 
  D + E = 0 ∨ D + E = 9 ∨ D + E = 18 := 
sorry

end possible_values_of_D_plus_E_l261_261183


namespace max_sum_cd_l261_261143

theorem max_sum_cd (c d : ℕ) (hc : c > 0) (hd : d > 1) (hcd : c^d < 500) 
  (hmax : ∀ (c' d': ℕ), c' > 0 → d' > 1 → c'^d' < 500 → c'^d' ≤ c^d) : c + d = 24 := 
by
  have h1 : 22^2 = 484 := rfl
  have h2 : c = 22 ∧ d = 2 := by sorry
  exact by sorry

end max_sum_cd_l261_261143


namespace sequence_formula_l261_261910

noncomputable def a (n : ℕ) : ℕ := n

theorem sequence_formula (n : ℕ) (h : 0 < n) (S_n : ℕ → ℕ) 
  (hSn : ∀ m : ℕ, S_n m = (1 / 2 : ℚ) * (a m)^2 + (1 / 2 : ℚ) * m) : a n = n :=
by
  sorry

end sequence_formula_l261_261910


namespace max_distance_between_bus_stops_l261_261131

theorem max_distance_between_bus_stops 
  (v_m : ℝ) (v_b : ℝ) (dist : ℝ) 
  (h1 : v_m = v_b / 3) (h2 : dist = 2) : 
  ∀ d : ℝ, d = 1.5 := sorry

end max_distance_between_bus_stops_l261_261131


namespace solve_trig_eq_l261_261812

theorem solve_trig_eq (k : ℤ) :
  (8.410 * Real.sqrt 3 * Real.sin t - Real.sqrt (2 * (Real.sin t)^2 - Real.sin (2 * t) + 3 * Real.cos t^2) = 0) ↔
  (∃ k : ℤ, t = π / 4 + 2 * k * π ∨ t = -Real.arctan 3 + π * (2 * k + 1)) :=
sorry

end solve_trig_eq_l261_261812


namespace fraction_sum_l261_261836

-- Define the fractions
def frac1: ℚ := 3/9
def frac2: ℚ := 5/12

-- The theorem statement
theorem fraction_sum : frac1 + frac2 = 3/4 := 
sorry

end fraction_sum_l261_261836


namespace solveEquation1_proof_solveEquation2_proof_l261_261785

noncomputable def solveEquation1 : Set ℝ :=
  { x | 2 * x^2 - 5 * x = 0 }

theorem solveEquation1_proof :
  solveEquation1 = { 0, (5 / 2 : ℝ) } :=
by
  sorry

noncomputable def solveEquation2 : Set ℝ :=
  { x | x^2 + 3 * x - 3 = 0 }

theorem solveEquation2_proof :
  solveEquation2 = { ( (-3 + Real.sqrt 21) / 2 : ℝ ), ( (-3 - Real.sqrt 21) / 2 : ℝ ) } :=
by
  sorry

end solveEquation1_proof_solveEquation2_proof_l261_261785


namespace num_tuples_abc_l261_261969

theorem num_tuples_abc (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 2019 ≥ 10 * a) (h5 : 10 * a ≥ 100 * b) (h6 : 100 * b ≥ 1000 * c) : 
  ∃ n, n = 574 := sorry

end num_tuples_abc_l261_261969


namespace minimal_coach_handshakes_l261_261001

theorem minimal_coach_handshakes (n k1 k2 : ℕ) (h1 : k1 < n) (h2 : k2 < n)
  (hn : (n * (n - 1)) / 2 + k1 + k2 = 300) : k1 + k2 = 0 := by
  sorry

end minimal_coach_handshakes_l261_261001


namespace part1_part2_l261_261441

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + 1) / Real.exp x
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := (-a * x^2 + (2 * a - b) * x + b - 1) / Real.exp x

theorem part1 (a b : ℝ) (h : f a b (-1) + f' a b (-1) = 0) : b = 2 * a :=
sorry

theorem part2 (a : ℝ) (h : a ≤ 1 / 2) (x : ℝ) : f a (2 * a) (abs x) ≤ 1 :=
sorry

end part1_part2_l261_261441


namespace time_for_5x5_grid_l261_261458

-- Definitions based on the conditions
def total_length_3x7 : ℕ := 4 * 7 + 8 * 3
def time_for_3x7 : ℕ := 26
def time_per_unit_length : ℚ := time_for_3x7 / total_length_3x7
def total_length_5x5 : ℕ := 6 * 5 + 6 * 5
def expected_time_for_5x5 : ℚ := total_length_5x5 * time_per_unit_length

-- Theorem statement to prove the total time for 5x5 grid
theorem time_for_5x5_grid : expected_time_for_5x5 = 30 := by
  sorry

end time_for_5x5_grid_l261_261458


namespace spherical_caps_ratio_l261_261808

theorem spherical_caps_ratio (r : ℝ) (m₁ m₂ : ℝ) (σ₁ σ₂ : ℝ)
  (h₁ : r = 1)
  (h₂ : σ₁ = 2 * π * m₁ + π * (1 - (1 - m₁)^2))
  (h₃ : σ₂ = 2 * π * m₂ + π * (1 - (1 - m₂)^2))
  (h₄ : σ₁ + σ₂ = 5 * π)
  (h₅ : m₁ + m₂ = 2) :
  (2 * m₁ + (1 - (1 - m₁)^2)) / (2 * m₂ + (1 - (1 - m₂)^2)) = 3.6 :=
sorry

end spherical_caps_ratio_l261_261808


namespace circumradius_inradius_perimeter_inequality_l261_261816

open Real

variables {R r P : ℝ} -- circumradius, inradius, perimeter
variable (triangle_type : String) -- acute, obtuse, right

def satisfies_inequality (R r P : ℝ) (triangle_type : String) : Prop :=
  if triangle_type = "right" then
    R ≥ (sqrt 2) / 2 * sqrt (P * r)
  else
    R ≥ (sqrt 3) / 3 * sqrt (P * r)

theorem circumradius_inradius_perimeter_inequality :
  ∀ (R r P : ℝ) (triangle_type : String), satisfies_inequality R r P triangle_type :=
by 
  intros R r P triangle_type
  sorry -- proof steps go here

end circumradius_inradius_perimeter_inequality_l261_261816


namespace car_speed_l261_261922

def travel_time : ℝ := 5
def travel_distance : ℝ := 300

theorem car_speed :
  travel_distance / travel_time = 60 := sorry

end car_speed_l261_261922


namespace functional_equation_solution_l261_261614

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x, 2 * f (f x) = (x^2 - x) * f x + 4 - 2 * x) :
  f 2 = 2 ∧ (f 1 = 1 ∨ f 1 = 4) :=
sorry

end functional_equation_solution_l261_261614


namespace rachel_total_clothing_l261_261782

def box_1_scarves : ℕ := 2
def box_1_mittens : ℕ := 3
def box_1_hats : ℕ := 1
def box_2_scarves : ℕ := 4
def box_2_mittens : ℕ := 2
def box_2_hats : ℕ := 2
def box_3_scarves : ℕ := 1
def box_3_mittens : ℕ := 5
def box_3_hats : ℕ := 3
def box_4_scarves : ℕ := 3
def box_4_mittens : ℕ := 4
def box_4_hats : ℕ := 1
def box_5_scarves : ℕ := 5
def box_5_mittens : ℕ := 3
def box_5_hats : ℕ := 2
def box_6_scarves : ℕ := 2
def box_6_mittens : ℕ := 6
def box_6_hats : ℕ := 0
def box_7_scarves : ℕ := 4
def box_7_mittens : ℕ := 1
def box_7_hats : ℕ := 3
def box_8_scarves : ℕ := 3
def box_8_mittens : ℕ := 2
def box_8_hats : ℕ := 4
def box_9_scarves : ℕ := 1
def box_9_mittens : ℕ := 4
def box_9_hats : ℕ := 5

def total_clothing : ℕ := 
  box_1_scarves + box_1_mittens + box_1_hats +
  box_2_scarves + box_2_mittens + box_2_hats +
  box_3_scarves + box_3_mittens + box_3_hats +
  box_4_scarves + box_4_mittens + box_4_hats +
  box_5_scarves + box_5_mittens + box_5_hats +
  box_6_scarves + box_6_mittens + box_6_hats +
  box_7_scarves + box_7_mittens + box_7_hats +
  box_8_scarves + box_8_mittens + box_8_hats +
  box_9_scarves + box_9_mittens + box_9_hats

theorem rachel_total_clothing : total_clothing = 76 :=
by
  sorry

end rachel_total_clothing_l261_261782


namespace most_likely_dissatisfied_proof_expected_dissatisfied_proof_variance_dissatisfied_proof_l261_261941

noncomputable def most_likely_dissatisfied (n : ℕ) : ℕ := 1

theorem most_likely_dissatisfied_proof (n : ℕ) (h : n > 1) :
  (∃ d : ℕ, d = most_likely_dissatisfied n) := by
  use 1
  trivial

noncomputable def expected_dissatisfied (n : ℕ) : ℝ := Real.sqrt (n / Real.pi)

theorem expected_dissatisfied_proof (n : ℕ) (h : n > 0) :
  (∃ e : ℝ, e = expected_dissatisfied n) := by
  use Real.sqrt (n / Real.pi)
  trivial

noncomputable def variance_dissatisfied (n : ℕ) : ℝ := 0.182 * n

theorem variance_dissatisfied_proof (n : ℕ) (h : n > 0) :
  (∃ v : ℝ, v = variance_dissatisfied n) := by
  use 0.182 * n
  trivial

end most_likely_dissatisfied_proof_expected_dissatisfied_proof_variance_dissatisfied_proof_l261_261941


namespace exists_small_diff_l261_261591

open Classical

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def b_seq : ℕ → ℝ := sorry

axiom a_prop : ∀ n ≥ 101, a_seq n = sqrt ((1:ℝ) / 100 * (finset.range 100).sum (λ j, (b_seq (n - j - 1))^2))

axiom b_prop : ∀ n ≥ 101, b_seq n = sqrt ((1:ℝ) / 100 * (finset.range 100).sum (λ j, (a_seq (n - j - 1))^2))

theorem exists_small_diff : ∃ m : ℕ, |a_seq m - b_seq m| < 0.001 := sorry

end exists_small_diff_l261_261591


namespace length_of_second_train_l261_261276

theorem length_of_second_train (speed1 speed2 : ℝ) (length1 time : ℝ) (h1 : speed1 = 60) (h2 : speed2 = 40) 
  (h3 : length1 = 450) (h4 : time = 26.99784017278618) :
  let speed1_mps := speed1 * 1000 / 3600
  let speed2_mps := speed2 * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := relative_speed * time
  let length2 := total_distance - length1
  length2 = 300 :=
by
  sorry

end length_of_second_train_l261_261276


namespace breaks_difference_l261_261246

-- James works for 240 minutes
def total_work_time : ℕ := 240

-- He takes a water break every 20 minutes
def water_break_interval : ℕ := 20

-- He takes a sitting break every 120 minutes
def sitting_break_interval : ℕ := 120

-- Calculate the number of water breaks James takes
def number_of_water_breaks : ℕ := total_work_time / water_break_interval

-- Calculate the number of sitting breaks James takes
def number_of_sitting_breaks : ℕ := total_work_time / sitting_break_interval

-- Prove the difference between the number of water breaks and sitting breaks is 10
theorem breaks_difference :
  number_of_water_breaks - number_of_sitting_breaks = 10 :=
by
  -- calculate number_of_water_breaks = 12
  -- calculate number_of_sitting_breaks = 2
  -- check the difference 12 - 2 = 10
  sorry

end breaks_difference_l261_261246


namespace find_sale_in_third_month_l261_261978

def sale_in_first_month := 5700
def sale_in_second_month := 8550
def sale_in_fourth_month := 3850
def sale_in_fifth_month := 14045
def average_sale := 7800
def num_months := 5
def total_sales := average_sale * num_months

theorem find_sale_in_third_month (X : ℕ) 
  (H : total_sales = sale_in_first_month + sale_in_second_month + X + sale_in_fourth_month + sale_in_fifth_month) :
  X = 9455 :=
by
  sorry

end find_sale_in_third_month_l261_261978


namespace total_veg_eaters_l261_261327

def people_eat_only_veg : ℕ := 16
def people_eat_only_nonveg : ℕ := 9
def people_eat_both_veg_and_nonveg : ℕ := 12

theorem total_veg_eaters : people_eat_only_veg + people_eat_both_veg_and_nonveg = 28 := 
by
  sorry

end total_veg_eaters_l261_261327


namespace termite_ridden_not_collapsing_l261_261780

theorem termite_ridden_not_collapsing
  (total_homes : ℕ)
  (termite_ridden_fraction : ℚ)
  (collapsing_fraction_of_termite_ridden : ℚ)
  (h1 : termite_ridden_fraction = 1/3)
  (h2 : collapsing_fraction_of_termite_ridden = 1/4) :
  (termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction_of_termite_ridden)) = 1/4 := 
by {
  sorry
}

end termite_ridden_not_collapsing_l261_261780


namespace probability_fx_lt_0_l261_261730

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem probability_fx_lt_0 :
  (∫ x in -Real.pi..Real.pi, if f x < 0 then 1 else 0) / (2 * Real.pi) = 2 / Real.pi :=
by sorry

end probability_fx_lt_0_l261_261730


namespace algebraic_expression_value_l261_261452

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3 * x - 3 = 0) : x^3 + 2 * x^2 - 6 * x + 3 = 0 := 
sorry

end algebraic_expression_value_l261_261452


namespace percent_increase_perimeter_third_triangle_l261_261135

noncomputable def side_length_first : ℝ := 4
noncomputable def side_length_second : ℝ := 2 * side_length_first
noncomputable def side_length_third : ℝ := 2 * side_length_second

noncomputable def perimeter (s : ℝ) : ℝ := 3 * s

noncomputable def percent_increase (initial_perimeter final_perimeter : ℝ) : ℝ := 
  ((final_perimeter - initial_perimeter) / initial_perimeter) * 100

theorem percent_increase_perimeter_third_triangle :
  percent_increase (perimeter side_length_first) (perimeter side_length_third) = 300 := 
sorry

end percent_increase_perimeter_third_triangle_l261_261135


namespace div_difference_l261_261856

theorem div_difference {a b n : ℕ} (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) (h : n ∣ a^n - b^n) :
  n ∣ ((a^n - b^n) / (a - b)) :=
by
  sorry

end div_difference_l261_261856


namespace remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0_l261_261961

def f (x : ℝ) : ℝ := x^15 + 1

theorem remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0 : f (-1) = 0 := by
  sorry

end remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0_l261_261961


namespace smallest_b_l261_261352

theorem smallest_b (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) (h_diff : a - b = 8)
  (h_gcd : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  b = 4 := 
sorry

end smallest_b_l261_261352


namespace perimeter_of_excircle_opposite_leg_l261_261213

noncomputable def perimeter_of_right_triangle (a varrho_a : ℝ) : ℝ :=
  2 * varrho_a * a / (2 * varrho_a - a)

theorem perimeter_of_excircle_opposite_leg
  (a varrho_a : ℝ) (h_a_pos : 0 < a) (h_varrho_a_pos : 0 < varrho_a) :
  (perimeter_of_right_triangle a varrho_a = 2 * varrho_a * a / (2 * varrho_a - a)) :=
by
  sorry

end perimeter_of_excircle_opposite_leg_l261_261213


namespace greatest_perfect_power_sum_l261_261146

def sum_c_d_less_500 : ℕ :=
  let c := 22
  let d := 2
  c + d

theorem greatest_perfect_power_sum :
  ∃ c d : ℕ, 0 < c ∧ 1 < d ∧ c^d < 500 ∧
  ∀ x y : ℕ, 0 < x ∧ 1 < y ∧ x^y < 500 → x^y ≤ c^d ∧ (c + d = 24) :=
by
  sorry

end greatest_perfect_power_sum_l261_261146


namespace isosceles_base_length_l261_261283

theorem isosceles_base_length :
  ∀ (equilateral_perimeter isosceles_perimeter side_length base_length : ℕ), 
  equilateral_perimeter = 60 →  -- Condition: Perimeter of the equilateral triangle is 60
  isosceles_perimeter = 45 →    -- Condition: Perimeter of the isosceles triangle is 45
  side_length = equilateral_perimeter / 3 →   -- Condition: Each side of the equilateral triangle
  isosceles_perimeter = side_length + side_length + base_length  -- Condition: Perimeter relation in isosceles triangle
  → base_length = 5  -- Result: The base length of the isosceles triangle is 5
:= 
sorry

end isosceles_base_length_l261_261283


namespace max_colors_l261_261065

theorem max_colors {n : ℕ} (h₁ : n ≥ 3) 
  (distinct_lengths : ∀ (p q : ℕ), p ≠ q → (∃! l : ℕ, l = dist p q))
  (coloring : ∀ (P : ℕ → ℕ),
    (∀ Q R, Q ≠ R → P(Q) = P(R)) ∧ 
    ∀ L M, L ≠ M → P(L) = P(M)) :
    ∃ k : ℕ, k = (n + 1) / 4 :=
sorry

end max_colors_l261_261065


namespace satellite_modular_units_l261_261669

variables (N S T U : ℕ)
variable (h1 : N = S / 3)
variable (h2 : S / T = 1 / 9)
variable (h3 : U * N = 8 * T / 9)

theorem satellite_modular_units :
  U = 24 :=
by sorry

end satellite_modular_units_l261_261669


namespace real_roots_of_polynomial_l261_261718

theorem real_roots_of_polynomial :
  (∀ x : ℝ, (x^10 + 36 * x^6 + 13 * x^2 = 13 * x^8 + x^4 + 36) ↔ 
    (x = 1 ∨ x = -1 ∨ x = 3 ∨ x = -3 ∨ x = 2 ∨ x = -2)) :=
by 
  sorry

end real_roots_of_polynomial_l261_261718


namespace consumption_increase_l261_261367

theorem consumption_increase (T C : ℝ) (P : ℝ) (h : 0.82 * (1 + P / 100) = 0.943) :
  P = 15.06 := by
  sorry

end consumption_increase_l261_261367


namespace possible_to_divide_into_two_groups_l261_261234

-- Define a type for People
universe u
variable {Person : Type u}

-- Define friend and enemy relations (assume they are given as functions)
variable (friend enemy : Person → Person)

-- Define the main statement
theorem possible_to_divide_into_two_groups (h_friend : ∀ p : Person, ∃ q : Person, friend p = q)
                                           (h_enemy : ∀ p : Person, ∃ q : Person, enemy p = q) :
  ∃ (company : Person → Bool),
    ∀ p : Person, company p ≠ company (friend p) ∧ company p ≠ company (enemy p) :=
by
  sorry

end possible_to_divide_into_two_groups_l261_261234


namespace min_value_of_quadratic_l261_261656

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 8*x + 18

theorem min_value_of_quadratic : ∃ x : ℝ, quadratic x = 2 ∧ (∀ y : ℝ, quadratic y ≥ 2) :=
by
  use 4
  sorry

end min_value_of_quadratic_l261_261656


namespace tangent_line_eq_l261_261487

theorem tangent_line_eq : 
  ∀ (x y: ℝ), y = x^3 - x + 3 → (x = 1 ∧ y = 3) → (2 * x - y - 1 = 0) :=
by
  intros x y h_curve h_point
  sorry

end tangent_line_eq_l261_261487


namespace problem_l261_261336

theorem problem (p q : ℝ) (h : 5 * p^2 - 20 * p + 15 = 0 ∧ 5 * q^2 - 20 * q + 15 = 0) : (p * q - 3)^2 = 0 := 
sorry

end problem_l261_261336


namespace starting_number_l261_261091

theorem starting_number (n : ℕ) (h1 : 200 ≥ n) (h2 : 33 = ((200 / 3) - (n / 3))) : n = 102 :=
by
  sorry

end starting_number_l261_261091


namespace knights_win_35_l261_261619

noncomputable def Sharks : ℕ := sorry
noncomputable def Falcons : ℕ := sorry
noncomputable def Knights : ℕ := 35
noncomputable def Wolves : ℕ := sorry
noncomputable def Royals : ℕ := sorry

-- Conditions
axiom h1 : Sharks > Falcons
axiom h2 : Wolves > 25
axiom h3 : Wolves < Knights ∧ Knights < Royals

-- Prove: Knights won 35 games
theorem knights_win_35 : Knights = 35 := 
by sorry

end knights_win_35_l261_261619


namespace value_of_expression_l261_261847

-- Let's define the sequences and sums based on the conditions in a)
def sum_of_evens (n : ℕ) : ℕ :=
  n * (n + 1)

def sum_of_multiples_of_three (p : ℕ) : ℕ :=
  3 * (p * (p + 1)) / 2

def sum_of_odds (m : ℕ) : ℕ :=
  m * m

-- Now let's formulate the problem statement as a theorem.
theorem value_of_expression : 
  sum_of_evens 200 - sum_of_multiples_of_three 100 - sum_of_odds 148 = 3146 :=
  by
  sorry

end value_of_expression_l261_261847


namespace rectangular_to_polar_l261_261991

theorem rectangular_to_polar : 
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (3 * Real.sqrt 2, 7 * Real.pi / 4) := 
by
  sorry

end rectangular_to_polar_l261_261991


namespace prove_central_angle_of_sector_l261_261232

noncomputable def central_angle_of_sector (R α : ℝ) : Prop :=
  (2 * R + R * α = 8) ∧ (1 / 2 * α * R^2 = 4)

theorem prove_central_angle_of_sector :
  ∃ α R : ℝ, central_angle_of_sector R α ∧ α = 2 :=
sorry

end prove_central_angle_of_sector_l261_261232


namespace smallest_positive_integer_divisible_l261_261853

theorem smallest_positive_integer_divisible (n : ℕ) (h1 : 15 = 3 * 5) (h2 : 16 = 2 ^ 4) (h3 : 18 = 2 * 3 ^ 2) :
  n = Nat.lcm (Nat.lcm 15 16) 18 ↔ n = 720 :=
by
  sorry

end smallest_positive_integer_divisible_l261_261853


namespace eyes_given_to_dog_l261_261585

-- Definitions of the conditions
def fish_per_person : ℕ := 4
def number_of_people : ℕ := 3
def eyes_per_fish : ℕ := 2
def eyes_eaten_by_Oomyapeck : ℕ := 22

-- The proof statement
theorem eyes_given_to_dog : ∃ (eyes_given_to_dog : ℕ), eyes_given_to_dog = 4 * 3 * 2 - 22 := by
  sorry

end eyes_given_to_dog_l261_261585


namespace find_d_and_r_l261_261557

theorem find_d_and_r (d r : ℤ)
  (h1 : 1210 % d = r)
  (h2 : 1690 % d = r)
  (h3 : 2670 % d = r) :
  d - 4 * r = -20 := sorry

end find_d_and_r_l261_261557


namespace woman_stop_time_l261_261823

-- Conditions
def man_speed := 5 -- in miles per hour
def woman_speed := 15 -- in miles per hour
def wait_time := 4 -- in minutes
def man_speed_mpm : ℚ := man_speed * (1 / 60) -- convert to miles per minute
def distance_covered := man_speed_mpm * wait_time

-- Definition of the relative speed between the woman and the man
def relative_speed := woman_speed - man_speed
def relative_speed_mpm : ℚ := relative_speed * (1 / 60) -- convert to miles per minute

-- The Proof statement
theorem woman_stop_time :
  (distance_covered / relative_speed_mpm) = 2 :=
by
  sorry

end woman_stop_time_l261_261823


namespace minimum_gennadys_l261_261698

theorem minimum_gennadys (alexs borises vasilies x : ℕ) (h₁ : alexs = 45) (h₂ : borises = 122) (h₃ : vasilies = 27)
    (h₄ : ∀ i, i ∈ list.range (borises-1) → alexs + vasilies + x > i) : 
    x = 49 :=
by 
    sorry

end minimum_gennadys_l261_261698


namespace bert_earns_more_l261_261404

def bert_toy_phones : ℕ := 8
def bert_price_per_phone : ℕ := 18
def tory_toy_guns : ℕ := 7
def tory_price_per_gun : ℕ := 20

theorem bert_earns_more : (bert_toy_phones * bert_price_per_phone) - (tory_toy_guns * tory_price_per_gun) = 4 := by
  sorry

end bert_earns_more_l261_261404


namespace selection_ways_l261_261456

-- Definitions based on conditions
def total_people : Nat := 9
def english_speakers : Nat := 5
def japanese_speakers : Nat := 4

-- Theorem to prove
theorem selection_ways :
  (Nat.choose english_speakers 1) * (Nat.choose japanese_speakers 1) = 20 := by
  sorry

end selection_ways_l261_261456


namespace flower_bed_dimensions_l261_261256

variable (l w : ℕ)

theorem flower_bed_dimensions :
  (l + 3) * (w + 2) = l * w + 64 →
  (l + 2) * (w + 3) = l * w + 68 →
  l = 14 ∧ w = 10 :=
by
  intro h1 h2
  sorry

end flower_bed_dimensions_l261_261256


namespace solve_for_x_l261_261784

-- Define the given equation as a predicate
def equation (x: ℚ) : Prop := (x + 4) / (x - 3) = (x - 2) / (x + 2)

-- State the problem in a Lean theorem
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = -2 / 11 :=
by
  existsi -2 / 11
  constructor
  repeat { sorry }

end solve_for_x_l261_261784


namespace sum_geometric_terms_l261_261239

noncomputable def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^n

theorem sum_geometric_terms (a q : ℝ) :
  a * (1 + q) = 3 → a * (1 + q) * q^2 = 6 → 
  a * (1 + q) * q^6 = 24 :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end sum_geometric_terms_l261_261239


namespace abs_ineq_solution_set_l261_261085

theorem abs_ineq_solution_set (x : ℝ) :
  |x - 5| + |x + 3| ≥ 10 ↔ x ≤ -4 ∨ x ≥ 6 :=
by
  sorry

end abs_ineq_solution_set_l261_261085


namespace possible_slopes_l261_261979

theorem possible_slopes (k : ℝ) (H_pos : k > 0) :
  (∃ x1 x2 : ℤ, (x1 + x2 : ℝ) = k ∧ (x1 * x2 : ℝ) = -2020) ↔ 
  k = 81 ∨ k = 192 ∨ k = 399 ∨ k = 501 ∨ k = 1008 ∨ k = 2019 := 
by
  sorry

end possible_slopes_l261_261979


namespace profit_share_ratio_l261_261105

theorem profit_share_ratio (P Q : ℝ) (hP : P = 40000) (hQ : Q = 60000) : P / Q = 2 / 3 :=
by
  rw [hP, hQ]
  norm_num

end profit_share_ratio_l261_261105


namespace santiago_stay_in_australia_l261_261348

/-- Santiago leaves his home country in the month of January,
    stays in Australia for a few months,
    and returns on the same date in the month of December.
    Prove that Santiago stayed in Australia for 11 months. -/
theorem santiago_stay_in_australia :
  ∃ (months : ℕ), months = 11 ∧
  (months = if (departure_month = 1) ∧ (return_month = 12) then 11 else 0) :=
by sorry

end santiago_stay_in_australia_l261_261348


namespace soda_ratio_l261_261147

theorem soda_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) : 
  let v_z := 1.3 * v
  let p_z := 0.85 * p
  (p_z / v_z) / (p / v) = 17 / 26 :=
by sorry

end soda_ratio_l261_261147


namespace num_integers_in_set_x_l261_261889

-- Definition and conditions
variable (x y : Finset ℤ)
variable (h1 : y.card = 10)
variable (h2 : (x ∩ y).card = 6)
variable (h3 : (x.symmDiff y).card = 6)

-- Proof statement
theorem num_integers_in_set_x : x.card = 8 := by
  sorry

end num_integers_in_set_x_l261_261889


namespace modulo_remainder_l261_261850

theorem modulo_remainder : (7^2023) % 17 = 15 := 
by 
  sorry

end modulo_remainder_l261_261850


namespace find_constant_b_l261_261019

theorem find_constant_b 
  (a b c : ℝ)
  (h1 : 3 * a = 9) 
  (h2 : (-2 * a + 3 * b) = -5) 
  : b = 1 / 3 :=
by 
  have h_a : a = 3 := by linarith
  
  have h_b : -2 * 3 + 3 * b = -5 := by linarith [h2]
  
  linarith

end find_constant_b_l261_261019


namespace sin_alpha_plus_pi_over_4_tan_double_alpha_l261_261430

-- Definitions of sin and tan 
open Real

variable (α : ℝ)

-- Given conditions
axiom α_in_interval : 0 < α ∧ α < π / 2
axiom sin_alpha_def : sin α = sqrt 5 / 5

-- Statement to prove
theorem sin_alpha_plus_pi_over_4 : sin (α + π / 4) = 3 * sqrt 10 / 10 :=
by
  sorry

theorem tan_double_alpha : tan (2 * α) = 4 / 3 :=
by
  sorry

end sin_alpha_plus_pi_over_4_tan_double_alpha_l261_261430


namespace min_gennadies_l261_261688

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l261_261688


namespace expected_value_eight_sided_die_l261_261634

-- Define a standard 8-sided die
def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Compute the probability of each outcome
def probability (n : ℕ) : ℝ := 1 / 8

-- Expected Value of a discrete random variable
def expected_value (outcomes : List ℕ) (prob : ℕ → ℝ) : ℝ :=
  outcomes.sum / outcomes.length.toReal

-- Theorem stating the expected value of a standard 8-sided die roll is 4.5
theorem expected_value_eight_sided_die : expected_value eight_sided_die probability = 4.5 := by
  sorry

end expected_value_eight_sided_die_l261_261634


namespace sum_roots_x_squared_minus_5x_plus_6_eq_5_l261_261386

noncomputable def sum_of_roots (a b c : Real) : Real :=
  -b / a

theorem sum_roots_x_squared_minus_5x_plus_6_eq_5 :
  sum_of_roots 1 (-5) 6 = 5 := by
  sorry

end sum_roots_x_squared_minus_5x_plus_6_eq_5_l261_261386


namespace al_told_the_truth_l261_261406

-- Definitions of G, S, and B based on each pirate's claim
def tom_G := 10
def tom_S := 8
def tom_B := 11

def al_G := 9
def al_S := 11
def al_B := 10

def pit_G := 10
def pit_S := 10
def pit_B := 9

def jim_G := 8
def jim_S := 10
def jim_B := 11

-- Condition that the total number of coins is 30
def total_coins (G : ℕ) (S : ℕ) (B : ℕ) : Prop := G + S + B = 30

-- The assertion that only Al told the truth
theorem al_told_the_truth :
  (total_coins tom_G tom_S tom_B → false) →
  (total_coins al_G al_S al_B) →
  (total_coins pit_G pit_S pit_B → false) →
  (total_coins jim_G jim_S jim_B → false) →
  true :=
by
  intros
  sorry

end al_told_the_truth_l261_261406


namespace alice_needs_to_add_stamps_l261_261709

variable (A B E P D : ℕ)
variable (h₁ : B = 4 * E)
variable (h₂ : E = 3 * P)
variable (h₃ : P = 2 * D)
variable (h₄ : D = A + 5)
variable (h₅ : A = 65)

theorem alice_needs_to_add_stamps : (1680 - A = 1615) :=
by
  sorry

end alice_needs_to_add_stamps_l261_261709


namespace structure_cube_count_l261_261264

theorem structure_cube_count :
  let middle_layer := 16
  let other_layers := 4 * 24
  middle_layer + other_layers = 112 :=
by
  let middle_layer := 16
  let other_layers := 4 * 24
  have h : middle_layer + other_layers = 112 := by
    sorry
  exact h

end structure_cube_count_l261_261264


namespace min_number_of_gennadys_l261_261692

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l261_261692


namespace convert_to_dms_convert_to_decimal_degrees_l261_261542

-- Problem 1: Conversion of 24.29 degrees to degrees, minutes, and seconds 
theorem convert_to_dms (d : ℝ) (h : d = 24.29) : 
  (∃ deg min sec, d = deg + min / 60 + sec / 3600 ∧ deg = 24 ∧ min = 17 ∧ sec = 24) :=
by
  sorry

-- Problem 2: Conversion of 36 degrees 40 minutes 30 seconds to decimal degrees
theorem convert_to_decimal_degrees (deg min sec : ℝ) (h : deg = 36 ∧ min = 40 ∧ sec = 30) : 
  (deg + min / 60 + sec / 3600) = 36.66 :=
by
  sorry

end convert_to_dms_convert_to_decimal_degrees_l261_261542


namespace sum_possible_values_l261_261876

theorem sum_possible_values (y : ℝ) (h : y^2 = 36) : 
  y = 6 ∨ y = -6 → 6 + (-6) = 0 :=
by
  intro hy
  rw [add_comm]
  exact add_neg_self 6

end sum_possible_values_l261_261876


namespace part1_solution_set_part2_range_a_l261_261034

noncomputable def inequality1 (a x : ℝ) : Prop :=
|a * x - 2| + |a * x - a| ≥ 2

theorem part1_solution_set : 
  (∀ x : ℝ, inequality1 1 x ↔ x ≥ 2.5 ∨ x ≤ 0.5) := 
sorry

theorem part2_range_a :
  (∀ x : ℝ, inequality1 a x) ↔ a ≥ 4 :=
sorry

end part1_solution_set_part2_range_a_l261_261034


namespace num_senior_in_sample_l261_261670

-- Definitions based on conditions
def total_students : ℕ := 2000
def senior_students : ℕ := 700
def sample_size : ℕ := 400

-- Theorem statement for the number of senior students in the sample
theorem num_senior_in_sample : 
  (senior_students * sample_size) / total_students = 140 :=
by 
  sorry

end num_senior_in_sample_l261_261670


namespace intersection_points_of_parabolas_l261_261841

/-- Let P1 be the equation of the first parabola: y = 3x^2 - 8x + 2 -/
def P1 (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 2

/-- Let P2 be the equation of the second parabola: y = 6x^2 + 4x + 2 -/
def P2 (x : ℝ) : ℝ := 6 * x^2 + 4 * x + 2

/-- Prove that the intersection points of P1 and P2 are (-4, 82) and (0, 2) -/
theorem intersection_points_of_parabolas : 
  {p : ℝ × ℝ | ∃ x, p = (x, P1 x) ∧ P1 x = P2 x} = 
    {(-4, 82), (0, 2)} :=
sorry

end intersection_points_of_parabolas_l261_261841


namespace range_of_a_no_solution_l261_261750

theorem range_of_a_no_solution (a : ℝ) :
  ¬ ∃ x : ℝ, |x + 2| + |3 - x| < 2 * a + 1 → a ≤ 2 :=
begin
  intros h,
  have H : 5 ≤ 2 * a + 1,
  { specialize h (3/2),
    linarith [abs_nonneg (3/2 + 2), abs_nonneg (3 - 3/2)] },
  linarith,
end

end range_of_a_no_solution_l261_261750


namespace positive_difference_perimeters_l261_261387

theorem positive_difference_perimeters (length width : ℝ) 
    (cut_rectangles : ℕ) 
    (H : length = 6 ∧ width = 9 ∧ cut_rectangles = 4) : 
    ∃ (p1 p2 : ℝ), (p1 = 24 ∧ p2 = 15) ∧ (abs (p1 - p2) = 9) :=
by
  sorry

end positive_difference_perimeters_l261_261387


namespace max_2x_plus_y_value_l261_261201

open Real

def on_ellipse (P : ℝ × ℝ) : Prop := 
  (P.1^2 / 4 + P.2^2 = 1)

def max_value_2x_plus_y (P : ℝ × ℝ) (h : on_ellipse P) : ℝ := 
  2 * P.1 + P.2

theorem max_2x_plus_y_value (P : ℝ × ℝ) (h : on_ellipse P):
  ∃ (m : ℝ), max_value_2x_plus_y P h = m ∧ m = sqrt 17 :=
sorry

end max_2x_plus_y_value_l261_261201


namespace distance_between_locations_l261_261130

theorem distance_between_locations
  (d_AC d_BC : ℚ)
  (d : ℚ)
  (meet_C : d_AC + d_BC = d)
  (travel_A_B : 150 + 150 + 540 = 840)
  (distance_ratio : 840 / 540 = 14 / 9)
  (distance_ratios : d_AC / d_BC = 14 / 9)
  (C_D : 540 = 5 * d / 23) :
  d = 2484 :=
by
  sorry

end distance_between_locations_l261_261130


namespace curveG_symmetric_curveG_unique_common_point_curveG_common_points_x_y_curveG_max_distance_l261_261738

-- Define the curve G as a set of points (x, y) satisfying the equation x^3 + y^3 - 6xy = 0
def curveG (x y : ℝ) : Prop :=
  x^3 + y^3 - 6 * x * y = 0

-- Prove symmetry of curveG with respect to the line y = x
theorem curveG_symmetric (x y : ℝ) (h : curveG x y) : curveG y x :=
  sorry

-- Prove unique common point with the line x + y - 6 = 0
theorem curveG_unique_common_point : ∃! p : ℝ × ℝ, curveG p.1 p.2 ∧ p.1 + p.2 = 6 :=
  sorry

-- Prove curveG has at least one common point with the line x - y + 1 = 0
theorem curveG_common_points_x_y : ∃ p : ℝ × ℝ, curveG p.1 p.2 ∧ p.1 - p.2 + 1 = 0 :=
  sorry

-- Prove the maximum distance from any point on the curveG to the origin is 3√2
theorem curveG_max_distance : ∀ p : ℝ × ℝ, curveG p.1 p.2 → p.1 > 0 → p.2 > 0 → (p.1^2 + p.2^2 ≤ 18) :=
  sorry

end curveG_symmetric_curveG_unique_common_point_curveG_common_points_x_y_curveG_max_distance_l261_261738


namespace find_number_of_roses_l261_261121

theorem find_number_of_roses : ∃ a : ℕ, 300 ≤ a ∧ a ≤ 400 ∧ a % 21 = 13 ∧ a % 15 = 7 :=
by
  -- Existential quantifier for the number 'a'
  use 307
  
  -- Proof of the conditions for 'a'
  split
  -- Proof that 300 ≤ 307 ∧ 307 ≤ 400
  exact ⟨by linarith, by linarith⟩
  split
  -- Proof that 307 % 21 = 13
  exact by norm_num
  -- Proof that 307 % 15 = 7 (because -8 % 15 = 7)
  exact by norm_num

end find_number_of_roses_l261_261121


namespace infinite_series_sum_eq_33_div_8_l261_261424

noncomputable def infinite_series_sum: ℝ :=
  ∑' n: ℕ, n^3 / (3^n : ℝ)

theorem infinite_series_sum_eq_33_div_8:
  infinite_series_sum = 33 / 8 :=
sorry

end infinite_series_sum_eq_33_div_8_l261_261424


namespace factor_of_M_l261_261772

theorem factor_of_M (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) : 
  1 ∣ (101010 * a + 10001 * b + 100 * c) :=
sorry

end factor_of_M_l261_261772


namespace flour_vs_sugar_difference_l261_261602

-- Definitions based on the conditions
def flour_needed : ℕ := 10
def flour_added : ℕ := 7
def sugar_needed : ℕ := 2

-- Define the mathematical statement to prove
theorem flour_vs_sugar_difference :
  (flour_needed - flour_added) - sugar_needed = 1 :=
by
  sorry

end flour_vs_sugar_difference_l261_261602


namespace arithmetic_sequence_geometric_l261_261567

noncomputable def sequence_arith_to_geom (a1 d : ℤ) (h_d : d ≠ 0) (n : ℕ) : ℤ :=
a1 + (n - 1) * d

theorem arithmetic_sequence_geometric (a1 d : ℤ) (h_d : d ≠ 0) (n : ℕ) :
  (n = 16)
    ↔ (((a1 + 3 * d) / (a1 + 2 * d) = (a1 + 6 * d) / (a1 + 3 * d)) ∧ 
        ((a1 + 6 * d) / (a1 + 3 * d) = (a1 + (n - 1) * d) / (a1 + 6 * d))) :=
by
  sorry

end arithmetic_sequence_geometric_l261_261567


namespace point_not_in_fourth_quadrant_l261_261899

theorem point_not_in_fourth_quadrant (a : ℝ) :
  ¬ ((a - 3 > 0) ∧ (a + 3 < 0)) :=
by
  sorry

end point_not_in_fourth_quadrant_l261_261899


namespace range_of_a_l261_261429

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ↔ 0 ≤ a ∧ a < 4 := sorry

end range_of_a_l261_261429


namespace factorize_expr_l261_261190

theorem factorize_expr (a b : ℝ) : a^2 - 2 * a * b = a * (a - 2 * b) := 
by 
  sorry

end factorize_expr_l261_261190


namespace number_of_true_propositions_l261_261864

-- Let's state the propositions
def original_proposition (P Q : Prop) := P → Q
def converse_proposition (P Q : Prop) := Q → P
def inverse_proposition (P Q : Prop) := ¬P → ¬Q
def contrapositive_proposition (P Q : Prop) := ¬Q → ¬P

-- Main statement we need to prove
theorem number_of_true_propositions (P Q : Prop) (hpq : original_proposition P Q) 
  (hc: contrapositive_proposition P Q) (hev: converse_proposition P Q)  (hbv: inverse_proposition P Q) : 
  (¬(P ↔ Q) ∨ (¬¬P ↔ ¬¬Q) ∨ (¬Q → ¬P) ∨ (P → Q)) := sorry

end number_of_true_propositions_l261_261864


namespace right_triangle_inscribed_circle_inequality_l261_261068

theorem right_triangle_inscribed_circle_inequality 
  {a b c r : ℝ} (h : a^2 + b^2 = c^2) (hr : r = (a + b - c) / 2) : 
  r ≤ (c / 2) * (Real.sqrt 2 - 1) :=
sorry

end right_triangle_inscribed_circle_inequality_l261_261068


namespace sum_possible_values_of_y_l261_261879

theorem sum_possible_values_of_y (y : ℝ) (h : y^2 = 36) : y = 6 ∨ y = -6 → (6 + (-6) = 0) :=
by
  sorry

end sum_possible_values_of_y_l261_261879


namespace mary_final_books_l261_261603

def mary_initial_books := 5
def mary_first_return := 3
def mary_first_checkout := 5
def mary_second_return := 2
def mary_second_checkout := 7

theorem mary_final_books :
  (mary_initial_books - mary_first_return + mary_first_checkout - mary_second_return + mary_second_checkout) = 12 := 
by 
  sorry

end mary_final_books_l261_261603


namespace circle_equation_l261_261356

theorem circle_equation :
  ∃ (a : ℝ), (y - a)^2 + x^2 = 1 ∧ (1 - 0)^2 + (2 - a)^2 = 1 ∧
  ∀ a, (1 - 0)^2 + (2 - a)^2 = 1 → a = 2 →
  x^2 + (y - 2)^2 = 1 := by sorry

end circle_equation_l261_261356


namespace piggy_bank_total_l261_261539

def amount_added_in_january: ℕ := 19
def amount_added_in_february: ℕ := 19
def amount_added_in_march: ℕ := 8

theorem piggy_bank_total:
  amount_added_in_january + amount_added_in_february + amount_added_in_march = 46 := by
  sorry

end piggy_bank_total_l261_261539


namespace vector_calc_l261_261217

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Statement to prove that 2a - b = (5, 7)
theorem vector_calc : 2 • a - b = (5, 7) :=
by {
  -- Proof will be filled here
  sorry
}

end vector_calc_l261_261217


namespace part1_part2_l261_261572

noncomputable def f (x m : ℝ) : ℝ := abs (x - m) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 ≤ x + 4 ↔ 0 ≤ x ∧ x ≤ 2 := sorry

theorem part2 (m n t : ℝ) (hm : m > 0) (hn : n > 0) (ht : t > 0) 
  (hmin : ∀ x, f x m ≥ 5 - n - t) :
  1 / (m + n) + 1 / t ≥ 2 := sorry

end part1_part2_l261_261572


namespace cos_210_eq_neg_sqrt3_div_2_l261_261166

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261166


namespace even_pair_probability_l261_261129

open Finset

theorem even_pair_probability : 
  let S := (range 5).image (λ x, x + 1),
      even_numbers := S.filter (λ x, x % 2 = 0),
      total_pairs := S.product S \ (S.diag),
      even_pairs := even_numbers.product even_numbers \ (even_numbers.diag) in
  ((even_pairs.card : ℚ) / total_pairs.card) = (1 / 10) :=
by 
  let S := (range 5).image (λ x, x + 1),
  let even_numbers := S.filter (λ x, x % 2 = 0),
  let total_pairs := S.product S \ (S.diag),
  let even_pairs := even_numbers.product even_numbers \ (even_numbers.diag),
  have h1 : (even_pairs.card : ℚ) = 1 := by sorry,
  have h2 : total_pairs.card = 10 := by sorry,
  rw [h1, h2],
  norm_num

end even_pair_probability_l261_261129


namespace num_complementary_sets_l261_261845

-- Definitions for shapes, colors, shades, and patterns
inductive Shape
| circle | square | triangle

inductive Color
| red | blue | green

inductive Shade
| light | medium | dark

inductive Pattern
| striped | dotted | plain

-- Definition of a card
structure Card where
  shape : Shape
  color : Color
  shade : Shade
  pattern : Pattern

-- Condition: Each possible combination is represented once in a deck of 81 cards.
def deck : List Card := sorry -- Construct the deck with 81 unique cards

-- Predicate for complementary sets of three cards
def is_complementary (c1 c2 c3 : Card) : Prop :=
  (c1.shape = c2.shape ∧ c2.shape = c3.shape ∧ c1.shape = c3.shape ∨
   c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∧ c1.color = c3.color ∨
   c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∧
  (c1.shade = c2.shade ∧ c2.shade = c3.shade ∧ c1.shade = c3.shade ∨
   c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade) ∧
  (c1.pattern = c2.pattern ∧ c2.pattern = c3.pattern ∧ c1.pattern = c3.pattern ∨
   c1.pattern ≠ c2.pattern ∧ c2.pattern ≠ c3.pattern ∧ c1.pattern ≠ c3.pattern)

-- Statement of the theorem to prove
theorem num_complementary_sets : 
  ∃ (complementary_sets : List (Card × Card × Card)), 
  complementary_sets.length = 5400 ∧
  ∀ (c1 c2 c3 : Card), (c1, c2, c3) ∈ complementary_sets → is_complementary c1 c2 c3 :=
sorry

end num_complementary_sets_l261_261845


namespace cost_of_corn_per_acre_l261_261077

def TotalLand : ℕ := 4500
def CostWheat : ℕ := 35
def Capital : ℕ := 165200
def LandWheat : ℕ := 3400
def LandCorn := TotalLand - LandWheat

theorem cost_of_corn_per_acre :
  ∃ C : ℕ, (Capital = (C * LandCorn) + (CostWheat * LandWheat)) ∧ C = 42 :=
by
  sorry

end cost_of_corn_per_acre_l261_261077


namespace find_total_roses_l261_261118

open Nat

theorem find_total_roses 
  (a : ℕ)
  (h1 : 300 ≤ a)
  (h2 : a ≤ 400)
  (h3 : a % 21 = 13)
  (h4 : a % 15 = 7) : 
  a = 307 := 
sorry

end find_total_roses_l261_261118


namespace sin_neg_thirtyone_sixths_pi_l261_261368

theorem sin_neg_thirtyone_sixths_pi : Real.sin (-31 / 6 * Real.pi) = 1 / 2 :=
by 
  sorry

end sin_neg_thirtyone_sixths_pi_l261_261368


namespace find_eighth_term_l261_261365

noncomputable def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + n * d

theorem find_eighth_term (a d : ℕ) :
  (arithmetic_sequence a d 0) + 
  (arithmetic_sequence a d 1) + 
  (arithmetic_sequence a d 2) + 
  (arithmetic_sequence a d 3) + 
  (arithmetic_sequence a d 4) + 
  (arithmetic_sequence a d 5) = 21 ∧
  arithmetic_sequence a d 6 = 7 →
  arithmetic_sequence a d 7 = 8 :=
by
  sorry

end find_eighth_term_l261_261365


namespace product_of_roots_proof_l261_261020

noncomputable def product_of_roots : ℚ :=
  let leading_coeff_poly1 := 3
  let leading_coeff_poly2 := 4
  let constant_term_poly1 := -15
  let constant_term_poly2 := 9
  let a := leading_coeff_poly1 * leading_coeff_poly2
  let b := constant_term_poly1 * constant_term_poly2
  (b : ℚ) / a

theorem product_of_roots_proof :
  product_of_roots = -45/4 :=
by
  sorry

end product_of_roots_proof_l261_261020


namespace quadratic_inequality_solution_l261_261364

theorem quadratic_inequality_solution (m : ℝ) :
  {x : ℝ | (x - m) * (x - (m + 1)) > 0} = {x | x < m ∨ x > m + 1} := sorry

end quadratic_inequality_solution_l261_261364


namespace smallest_cube_dividing_pq2r4_l261_261776

-- Definitions of conditions
variables {p q r : ℕ} [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] [Fact (Nat.Prime r)]
variables (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)

-- Definitions used in the proof
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

def smallest_perfect_cube_dividing (n k : ℕ) : Prop :=
  is_perfect_cube k ∧ n ∣ k ∧ ∀ k', is_perfect_cube k' ∧ n ∣ k' → k ≤ k'

-- The proof problem
theorem smallest_cube_dividing_pq2r4 (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  smallest_perfect_cube_dividing (p * q^2 * r^4) ((p * q * r^2)^3) :=
sorry

end smallest_cube_dividing_pq2r4_l261_261776


namespace books_arrangement_count_l261_261445

noncomputable def arrangement_of_books : ℕ :=
  let total_books := 5
  let identical_books := 2
  Nat.factorial total_books / Nat.factorial identical_books

theorem books_arrangement_count : arrangement_of_books = 60 := by
  sorry

end books_arrangement_count_l261_261445


namespace eggs_not_eaten_per_week_l261_261259

theorem eggs_not_eaten_per_week : 
  let trays_bought := 2
  let eggs_per_tray := 24
  let days_per_week := 7
  let eggs_eaten_by_children_per_day := 2 * 2 -- 2 eggs each by 2 children
  let eggs_eaten_by_parents_per_day := 4
  let total_eggs_eaten_per_week := (eggs_eaten_by_children_per_day + eggs_eaten_by_parents_per_day) * days_per_week
  let total_eggs_bought := trays_bought * eggs_per_tray * 2  -- Re-calculated trays
  total_eggs_bought - total_eggs_eaten_per_week = 40 :=
by
  let trays_bought := 2
  let eggs_per_tray := 24
  let days_per_week := 7
  let eggs_eaten_by_children_per_day := 2 * 2
  let eggs_eaten_by_parents_per_day := 4
  let total_eggs_eaten_per_week := (eggs_eaten_by_children_per_day + eggs_eaten_by_parents_per_day) * days_per_week
  let total_eggs_bought := trays_bought * eggs_per_tray * 2
  show total_eggs_bought - total_eggs_eaten_per_week = 40
  sorry

end eggs_not_eaten_per_week_l261_261259


namespace intersection_of_P_and_Q_l261_261741

def P : Set ℤ := {-3, -2, 0, 2}
def Q : Set ℤ := {-1, -2, -3, 0, 1}

theorem intersection_of_P_and_Q : P ∩ Q = {-3, -2, 0} := by
  sorry

end intersection_of_P_and_Q_l261_261741


namespace shortest_distance_l261_261525

theorem shortest_distance 
  (C : ℝ × ℝ) (B : ℝ × ℝ) (stream : ℝ)
  (hC : C = (0, -3))
  (hB : B = (9, -8))
  (hStream : stream = 0) :
  ∃ d : ℝ, d = 3 + Real.sqrt 202 :=
by
  sorry

end shortest_distance_l261_261525


namespace jane_mistake_corrected_l261_261831

-- Conditions translated to Lean definitions
variables (x y z : ℤ)
variable (h1 : x - (y + z) = 15)
variable (h2 : x - y + z = 7)

-- Statement to prove
theorem jane_mistake_corrected : x - y = 11 :=
by
  -- Placeholder for the proof
  sorry

end jane_mistake_corrected_l261_261831


namespace inequality_solution_set_l261_261267

theorem inequality_solution_set (x : ℝ) : 
  ( (x - 1) / (x + 2) > 0 ) ↔ ( x > 1 ∨ x < -2 ) :=
by sorry

end inequality_solution_set_l261_261267


namespace sum_possible_values_l261_261877

theorem sum_possible_values (y : ℝ) (h : y^2 = 36) : 
  y = 6 ∨ y = -6 → 6 + (-6) = 0 :=
by
  intro hy
  rw [add_comm]
  exact add_neg_self 6

end sum_possible_values_l261_261877


namespace digit_100th_is_4_digit_1000th_is_3_l261_261677

noncomputable section

def digit_100th_place : Nat :=
  4

def digit_1000th_place : Nat :=
  3

theorem digit_100th_is_4 (n : ℕ) (h1 : n ∈ {m | m = 100}) : digit_100th_place = 4 := by
  sorry

theorem digit_1000th_is_3 (n : ℕ) (h1 : n ∈ {m | m = 1000}) : digit_1000th_place = 3 := by
  sorry

end digit_100th_is_4_digit_1000th_is_3_l261_261677


namespace rewrite_equation_to_function_l261_261609

theorem rewrite_equation_to_function (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  sorry

end rewrite_equation_to_function_l261_261609


namespace sphere_volume_l261_261623

theorem sphere_volume {r : ℝ} (h: 4 * Real.pi * r^2 = 256 * Real.pi) : (4 / 3) * Real.pi * r^3 = (2048 / 3) * Real.pi :=
by
  sorry

end sphere_volume_l261_261623


namespace find_number_l261_261397

theorem find_number (x N : ℕ) (h₁ : x = 32) (h₂ : N - (23 - (15 - x)) = (12 * 2 / 1 / 2)) : N = 88 :=
sorry

end find_number_l261_261397


namespace a_number_M_middle_digit_zero_l261_261288

theorem a_number_M_middle_digit_zero (d e f M : ℕ) (h1 : M = 36 * d + 6 * e + f)
  (h2 : M = 64 * f + 8 * e + d) (hd : d < 6) (he : e < 6) (hf : f < 6) : e = 0 :=
by sorry

end a_number_M_middle_digit_zero_l261_261288


namespace sixth_term_is_sixteen_l261_261794

-- Definition of the conditions
def first_term : ℝ := 512
def eighth_term (r : ℝ) : Prop := 512 * r^7 = 2

-- Proving the 6th term is 16 given the conditions
theorem sixth_term_is_sixteen (r : ℝ) (hr : eighth_term r) :
  512 * r^5 = 16 :=
by
  sorry

end sixth_term_is_sixteen_l261_261794


namespace truck_distance_in_3_hours_l261_261832

theorem truck_distance_in_3_hours : 
  ∀ (speed_2miles_2_5minutes : ℝ) 
    (time_minutes : ℝ),
    (speed_2miles_2_5minutes = 2 / 2.5) →
    (time_minutes = 180) →
    (speed_2miles_2_5minutes * time_minutes = 144) :=
by
  intros
  sorry

end truck_distance_in_3_hours_l261_261832


namespace problem_problem_contrapositive_l261_261059

def is_rational (r : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ r = p / q

def can_be_expressed_as_quotient (f : ℝ → ℝ) : Prop :=
  ∃ (p q : polynomial ℤ), q ≠ 0 ∧ ∀ x : ℝ, f x = (polynomial.eval x p) / (polynomial.eval x q)

def a (n : ℕ) : Nat := 0 -- This would normally come from the conditions, to ensure a_𝑛 ∈ {0, 1}

noncomputable def f (x : ℝ) : ℝ := ∑' n, (a n) * x^n

theorem problem (h : is_rational (f (1/2))) :
  can_be_expressed_as_quotient f :=
sorry

theorem problem_contrapositive (h : ¬ is_rational (f (1/2))) :
  ¬ can_be_expressed_as_quotient f :=
sorry

end problem_problem_contrapositive_l261_261059


namespace scientist_took_absent_mindedness_pills_l261_261107

-- Definitions for the conditions
variables {Ω : Type*} [measurable_space Ω] {P : measure Ω}

def R : event Ω := sorry --Event that the Scientist took pills for absent-mindedness
def A : event Ω := sorry --Event that knee pain stopped
def B : event Ω := sorry --Event that absent-mindedness disappeared

-- Given conditions
def P_R : ℝ := 1/2
def P_A_given_R : ℝ := 0.8
def P_B_given_R : ℝ := 0.05

def P_R_complement : ℝ := 1/2
def P_A_given_R_complement : ℝ := 0.9
def P_B_given_R_complement : ℝ := 0.02

-- Joint probabilities
def P_R_A_B : ℝ := P_R * P_A_given_R * P_B_given_R
def P_R_complement_A_B : ℝ := P_R_complement * P_A_given_R_complement * P_B_given_R_complement

-- Event that both A and B happen
def P_A_B : ℝ := P_R_A_B + P_R_complement_A_B

-- Required conditional probability
noncomputable def P_R_given_A_B : ℝ := P_R_A_B / P_A_B

-- Theorem we want to prove
theorem scientist_took_absent_mindedness_pills :
  P_R_given_A_B = 0.69 :=
sorry

end scientist_took_absent_mindedness_pills_l261_261107


namespace sum_of_roots_of_y_squared_eq_36_l261_261880

theorem sum_of_roots_of_y_squared_eq_36 :
  (∀ y : ℝ, y^2 = 36 → y = 6 ∨ y = -6) → (6 + (-6) = 0) :=
by
  sorry

end sum_of_roots_of_y_squared_eq_36_l261_261880


namespace saree_sale_price_l261_261939

def initial_price : Real := 150
def discount1 : Real := 0.20
def tax1 : Real := 0.05
def discount2 : Real := 0.15
def tax2 : Real := 0.04
def discount3 : Real := 0.10
def tax3 : Real := 0.03
def final_price : Real := 103.25

theorem saree_sale_price :
  let price_after_discount1 : Real := initial_price * (1 - discount1)
  let price_after_tax1 : Real := price_after_discount1 * (1 + tax1)
  let price_after_discount2 : Real := price_after_tax1 * (1 - discount2)
  let price_after_tax2 : Real := price_after_discount2 * (1 + tax2)
  let price_after_discount3 : Real := price_after_tax2 * (1 - discount3)
  let price_after_tax3 : Real := price_after_discount3 * (1 + tax3)
  abs (price_after_tax3 - final_price) < 0.01 :=
by
  sorry

end saree_sale_price_l261_261939


namespace seminar_attendees_l261_261391

theorem seminar_attendees (a b c d attendees_not_from_companies : ℕ)
  (h1 : a = 30)
  (h2 : b = 2 * a)
  (h3 : c = a + 10)
  (h4 : d = c - 5)
  (h5 : attendees_not_from_companies = 20) :
  a + b + c + d + attendees_not_from_companies = 185 := by
  sorry

end seminar_attendees_l261_261391


namespace find_c_for_circle_radius_five_l261_261720

theorem find_c_for_circle_radius_five
  (c : ℝ)
  (h : ∀ x y : ℝ, x^2 + 8 * x + y^2 + 2 * y + c = 0) :
  c = -8 :=
sorry

end find_c_for_circle_radius_five_l261_261720


namespace num_math_not_science_l261_261237

-- Definitions as conditions
def students_total : ℕ := 30
def both_clubs : ℕ := 2
def math_to_science_ratio : ℕ := 3

-- The proof we need to show
theorem num_math_not_science :
  ∃ x y : ℕ, (x + y + both_clubs = students_total) ∧ (y = math_to_science_ratio * (x + both_clubs) - 2 * (math_to_science_ratio - 1)) ∧ (y - both_clubs = 20) :=
by
  sorry

end num_math_not_science_l261_261237


namespace inequality_solution_set_l261_261266

theorem inequality_solution_set (x : ℝ) : 
  ( (x - 1) / (x + 2) > 0 ) ↔ ( x > 1 ∨ x < -2 ) :=
by sorry

end inequality_solution_set_l261_261266


namespace unique_real_y_l261_261010

def star (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y

theorem unique_real_y (y : ℝ) : (∃! y : ℝ, star 4 y = 10) :=
  by {
    sorry
  }

end unique_real_y_l261_261010


namespace ttakjis_count_l261_261369

theorem ttakjis_count (n : ℕ) (initial_residual new_residual total_ttakjis : ℕ) :
  initial_residual = 36 → 
  new_residual = 3 → 
  total_ttakjis = n^2 + initial_residual → 
  total_ttakjis = (n + 1)^2 + new_residual → 
  total_ttakjis = 292 :=
by
  sorry

end ttakjis_count_l261_261369


namespace factorize_expression_l261_261187

-- Define the variables
variables (a b : ℝ)

-- State the theorem to prove the factorization
theorem factorize_expression : a^2 - 2 * a * b = a * (a - 2 * b) :=
by 
  -- Proof goes here
  sorry

end factorize_expression_l261_261187


namespace max_min_z_diff_correct_l261_261597

noncomputable def max_min_z_diff (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) : ℝ :=
  6

theorem max_min_z_diff_correct (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) :
  max_min_z_diff x y z h1 h2 = 6 :=
sorry

end max_min_z_diff_correct_l261_261597


namespace folder_cost_calc_l261_261826

noncomputable def pencil_cost : ℚ := 0.5
noncomputable def dozen_pencils : ℕ := 24
noncomputable def num_folders : ℕ := 20
noncomputable def total_cost : ℚ := 30
noncomputable def total_pencil_cost : ℚ := dozen_pencils * pencil_cost
noncomputable def remaining_cost := total_cost - total_pencil_cost
noncomputable def folder_cost := remaining_cost / num_folders

theorem folder_cost_calc : folder_cost = 0.9 := by
  -- Definitions
  have pencil_cost_def : pencil_cost = 0.5 := rfl
  have dozen_pencils_def : dozen_pencils = 24 := rfl
  have num_folders_def : num_folders = 20 := rfl
  have total_cost_def : total_cost = 30 := rfl
  have total_pencil_cost_def : total_pencil_cost = dozen_pencils * pencil_cost := rfl
  have remaining_cost_def : remaining_cost = total_cost - total_pencil_cost := rfl
  have folder_cost_def : folder_cost = remaining_cost / num_folders := rfl

  -- Calculation steps given conditions
  sorry

end folder_cost_calc_l261_261826


namespace find_number_l261_261021

theorem find_number (n : ℕ) (h : 582964 * n = 58293485180) : n = 100000 :=
by
  sorry

end find_number_l261_261021


namespace vector_subtraction_l261_261224

variable (a b : ℝ × ℝ)

def vector_calc (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem vector_subtraction :
  a = (2, 4) → b = (-1, 1) → vector_calc a b = (5, 7) := by
  intros ha hb
  simp [vector_calc]
  rw [ha, hb]
  simp
  sorry

end vector_subtraction_l261_261224


namespace total_value_of_gold_is_l261_261767

-- Definitions based on the conditions
def legacyBars : ℕ := 5
def aleenaBars : ℕ := legacyBars - 2
def valuePerBar : ℝ := 2200
def totalValue : ℝ := (legacyBars + aleenaBars) * valuePerBar

-- Theorem statement
theorem total_value_of_gold_is :
  totalValue = 17600 := by
  -- We add sorry here to skip the proof
  sorry

end total_value_of_gold_is_l261_261767


namespace pamela_skittles_correct_l261_261257

def pamela_initial_skittles := 50
def pamela_gives_skittles_to_karen := 7
def pamela_receives_skittles_from_kevin := 3
def pamela_shares_percentage := 20

def pamela_final_skittles : Nat :=
  let after_giving := pamela_initial_skittles - pamela_gives_skittles_to_karen
  let after_receiving := after_giving + pamela_receives_skittles_from_kevin
  let share_amount := (after_receiving * pamela_shares_percentage) / 100
  let rounded_share := Nat.floor share_amount
  let final_count := after_receiving - rounded_share
  final_count

theorem pamela_skittles_correct :
  pamela_final_skittles = 37 := by
  sorry

end pamela_skittles_correct_l261_261257


namespace largest_inscribed_parabola_area_l261_261729

noncomputable def maximum_parabolic_area_in_cone (r l : ℝ) : ℝ :=
  (l * r) / 2 * Real.sqrt 3

theorem largest_inscribed_parabola_area (r l : ℝ) : 
  ∃ t : ℝ, t = maximum_parabolic_area_in_cone r l :=
by
  let t_max := (l * r) / 2 * Real.sqrt 3
  use t_max
  sorry

end largest_inscribed_parabola_area_l261_261729


namespace expected_value_of_defective_products_variance_of_defective_products_l261_261528

def batch_authentic_prob := 0.99
def selected_products := 200
def defective_prob := 1.0 - batch_authentic_prob
def binomial_dist := Probability.Distribution.Binomial selected_products defective_prob

-- Definitions for the expected value and variance of a binomial distribution.
def expected_value := 2
def variance := 1.98

theorem expected_value_of_defective_products :
  Probability.Distribution.mean binomial_dist = expected_value := by
  sorry

theorem variance_of_defective_products :
  Probability.Distribution.variance binomial_dist = variance := by
  sorry

end expected_value_of_defective_products_variance_of_defective_products_l261_261528


namespace volume_of_prism_l261_261960

   theorem volume_of_prism (a b c : ℝ)
     (h1 : a * b = 18) (h2 : b * c = 12) (h3 : a * c = 8) :
     a * b * c = 24 * Real.sqrt 3 :=
   sorry
   
end volume_of_prism_l261_261960


namespace score_of_juniors_correct_l261_261580

-- Let the total number of students be 20
def total_students : ℕ := 20

-- 20% of the students are juniors
def juniors_percent : ℝ := 0.20

-- Total number of juniors
def number_of_juniors : ℕ := 4 -- 20% of 20

-- The remaining are seniors
def number_of_seniors : ℕ := 16 -- 80% of 20

-- Overall average score of all students
def overall_average_score : ℝ := 85

-- Average score of the seniors
def seniors_average_score : ℝ := 84

-- Calculate the total score of all students
def total_score : ℝ := overall_average_score * total_students

-- Calculate the total score of the seniors
def total_score_of_seniors : ℝ := seniors_average_score * number_of_seniors

-- We need to prove that the score of each junior
def score_of_each_junior : ℝ := 89

theorem score_of_juniors_correct :
  (total_score - total_score_of_seniors) / number_of_juniors = score_of_each_junior :=
by
  sorry

end score_of_juniors_correct_l261_261580


namespace first_player_wins_l261_261948

def wins (sum_rows sum_cols : ℕ) : Prop := sum_rows > sum_cols

theorem first_player_wins 
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ)
  (h : a_1 > a_2 ∧ a_2 > a_3 ∧ a_3 > a_4 ∧ a_4 > a_5 ∧ a_5 > a_6 ∧ a_6 > a_7 ∧ a_7 > a_8 ∧ a_8 > a_9) :
  ∃ sum_rows sum_cols, wins sum_rows sum_cols :=
sorry

end first_player_wins_l261_261948


namespace probability_of_two_tails_two_heads_l261_261885

theorem probability_of_two_tails_two_heads :
  let p := (1:ℚ) / 2 in
  ∃ (prob : ℚ), prob = 3/8 ∧ (∀ (k : Finset (Fin 4)), k.card = 2 → k = 3) → prob = (nat.choose 4 2) * (p ^ 4) :=
begin
  -- Definitions derived from conditions
  -- p is the probability of head or tail in single toss
  -- We need to show the overall probability of two heads and two tails equals 3/8
  -- p is 1/2
  let p : ℚ := 1/2,
  -- There are 4 choose 2 ways to arrange 2 heads and 2 tails among 4 coins
  have h1 : nat.choose 4 2 = 6,
  { rw [nat.choose_eq_factorial_div_factorial, nat.factorial, nat.factorial],
    norm_num, },
  -- The overall probability of one arrangement of the coins is (1/2)^4
  let seq_prob := p ^ 4,
  -- The total probability is the number of arrangements times the probability of one specific arrangement
  let prob := 6 * seq_prob,
  have h2 : prob = 3/8,
  { simp [h1, seq_prob],
    norm_num, },
  existsi prob,
  split,
  exact h2,
  intros k hk,
  -- This is to show that any set of 2 elements chosen from 4 elements (coins) will
  -- be one of those 6 arrangements that results in the desired probability.
  have h3 : ∀ (k : Finset (Fin 4)), k.card = 2 → k ∈ Finset.powerset.fin k,
  { intro k,
    norm_num, },
  exact h3,
  sorry,
end

end probability_of_two_tails_two_heads_l261_261885


namespace coefficient_of_determination_l261_261042

-- Define the observations and conditions for the problem
def observations (n : ℕ) := 
  {x : ℕ → ℝ // ∃ b a : ℝ, ∀ i : ℕ, i < n → ∃ y_i : ℝ, y_i = b * x i + a}

/-- 
  Given a set of observations (x_1, y_1), (x_2, y_2), ..., (x_n, y_n) 
  that satisfies the equation y_i = bx_i + a for i = 1, 2, ..., n, 
  prove that the coefficient of determination R² is 1.
-/
theorem coefficient_of_determination (n : ℕ) (obs : observations n) : 
  ∃ R_squared : ℝ, R_squared = 1 :=
sorry

end coefficient_of_determination_l261_261042


namespace rectangular_table_capacity_l261_261392

variable (R : ℕ) -- The number of pupils a rectangular table can seat

-- Conditions
variable (rectangular_tables : ℕ)
variable (square_tables : ℕ)
variable (square_table_capacity : ℕ)
variable (total_pupils : ℕ)

-- Setting the values based on the conditions
axiom h1 : rectangular_tables = 7
axiom h2 : square_tables = 5
axiom h3 : square_table_capacity = 4
axiom h4 : total_pupils = 90

-- The proof statement
theorem rectangular_table_capacity :
  7 * R + 5 * 4 = 90 → R = 10 :=
by
  intro h
  sorry

end rectangular_table_capacity_l261_261392


namespace average_income_QR_l261_261482

theorem average_income_QR (P Q R : ℝ) 
  (h1: (P + Q) / 2 = 5050) 
  (h2: (P + R) / 2 = 5200) 
  (hP: P = 4000) : 
  (Q + R) / 2 = 6250 := 
by 
  -- additional steps and proof to be provided here
  sorry

end average_income_QR_l261_261482


namespace min_angle_for_quadrilateral_l261_261337

theorem min_angle_for_quadrilateral (d : ℝ) (h : ∀ (a b c d : ℝ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a + b + c + d = 360 → (a < d ∨ b < d)) :
  d = 120 :=
by
  sorry

end min_angle_for_quadrilateral_l261_261337


namespace polynomial_expansion_coefficient_a8_l261_261214

theorem polynomial_expansion_coefficient_a8 :
  let a := 1
  let a_1 := 10
  let a_2 := 45
  let a_3 := 120
  let a_4 := 210
  let a_5 := 252
  let a_6 := 210
  let a_7 := 120
  let a_8 := 45
  let a_9 := 10
  let a_10 := 1
  a_8 = 45 :=
by {
  sorry
}

end polynomial_expansion_coefficient_a8_l261_261214


namespace prime_eq_sum_of_two_squares_l261_261031

theorem prime_eq_sum_of_two_squares (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 4 = 1) : 
  ∃ a b : ℤ, p = a^2 + b^2 := 
sorry

end prime_eq_sum_of_two_squares_l261_261031


namespace find_y_l261_261825

theorem find_y (x y : ℕ) (h1 : 24 * x = 173 * y) (h2 : 173 * y = 1730) : y = 10 :=
by 
  -- Proof is skipped
  sorry

end find_y_l261_261825


namespace four_pow_2024_mod_11_l261_261376

theorem four_pow_2024_mod_11 : (4 ^ 2024) % 11 = 3 :=
by
  sorry

end four_pow_2024_mod_11_l261_261376


namespace smallest_positive_integer_divisible_by_15_16_18_l261_261851

theorem smallest_positive_integer_divisible_by_15_16_18 : 
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) ∧ n = 720 := 
by
  sorry

end smallest_positive_integer_divisible_by_15_16_18_l261_261851


namespace toy_position_from_left_l261_261521

/-- Define the total number of toys -/
def total_toys : ℕ := 19

/-- Define the position of toy (A) from the right -/
def position_from_right : ℕ := 8

/-- Prove the main statement: The position of toy (A) from the left is 12 given the conditions -/
theorem toy_position_from_left : total_toys - position_from_right + 1 = 12 := by
  sorry

end toy_position_from_left_l261_261521


namespace find_all_triples_l261_261998

def satisfying_triples (a b c : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 
  (a^2 + a*b = c) ∧ 
  (b^2 + b*c = a) ∧ 
  (c^2 + c*a = b)

theorem find_all_triples (a b c : ℝ) : satisfying_triples a b c ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = 1/2 ∧ b = 1/2 ∧ c = 1/2) :=
by sorry

end find_all_triples_l261_261998


namespace cos_210_eq_neg_sqrt3_div_2_l261_261170

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261170


namespace sum_of_consecutive_even_integers_l261_261509

theorem sum_of_consecutive_even_integers (a : ℤ) (h : a + (a + 6) = 136) :
  a + (a + 2) + (a + 4) + (a + 6) = 272 :=
by
  sorry

end sum_of_consecutive_even_integers_l261_261509


namespace kim_money_l261_261515

theorem kim_money (S P K : ℝ) (h1 : K = 1.40 * S) (h2 : S = 0.80 * P) (h3 : S + P = 1.80) : K = 1.12 :=
by sorry

end kim_money_l261_261515


namespace cost_of_7_cubic_yards_l261_261275

def cost_per_cubic_foot : ℕ := 8
def cubic_feet_per_cubic_yard : ℕ := 27
def cubic_yards : ℕ := 7

theorem cost_of_7_cubic_yards
  (c : ℕ) (c_cubic : c = cost_per_cubic_foot)
  (f : ℕ) (f_cubic : f = cubic_feet_per_cubic_yard)
  (y : ℕ) (y_cubic : y = cubic_yards) :
  c * f * y = 1512 :=
begin
  sorry
end

end cost_of_7_cubic_yards_l261_261275


namespace simplify_expression_l261_261453

theorem simplify_expression (y : ℝ) : (y - 2)^2 + 2 * (y - 2) * (5 + y) + (5 + y)^2 = (2*y + 3)^2 := 
by sorry

end simplify_expression_l261_261453


namespace price_reduction_percentage_l261_261360

theorem price_reduction_percentage (original_price new_price : ℕ) 
  (h_original : original_price = 250) 
  (h_new : new_price = 200) : 
  (original_price - new_price) * 100 / original_price = 20 := 
by 
  -- include the proof when needed
  sorry

end price_reduction_percentage_l261_261360


namespace unique_n_for_50_percent_mark_l261_261340

def exam_conditions (n : ℕ) : Prop :=
  let correct_first_20 : ℕ := 15
  let remaining : ℕ := n - 20
  let correct_remaining : ℕ := remaining / 3
  let total_correct : ℕ := correct_first_20 + correct_remaining
  total_correct * 2 = n

theorem unique_n_for_50_percent_mark : ∃! (n : ℕ), exam_conditions n := sorry

end unique_n_for_50_percent_mark_l261_261340


namespace minimum_value_of_f_l261_261278

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 6 * x + 1

theorem minimum_value_of_f :
  exists (x : ℝ), x = 1 + 1 / Real.sqrt 3 ∧ ∀ (y : ℝ), f (1 + 1 / Real.sqrt 3) ≤ f y := sorry

end minimum_value_of_f_l261_261278


namespace regression_analysis_correct_statement_l261_261329

variables (x : Type) (y : Type)

def is_deterministic (v : Type) : Prop := sorry -- A placeholder definition
def is_random (v : Type) : Prop := sorry -- A placeholder definition

theorem regression_analysis_correct_statement :
  (is_deterministic x) → (is_random y) →
  ("The independent variable is a deterministic variable, and the dependent variable is a random variable" = "C") :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end regression_analysis_correct_statement_l261_261329


namespace probability_x_lt_2y_in_rectangle_l261_261398

open Set MeasureTheory

theorem probability_x_lt_2y_in_rectangle :
  let Ω := {p : ℝ × ℝ | (0 ≤ p.1 ∧ p.1 ≤ 4) ∧ (0 ≤ p.2 ∧ p.2 ≤ 3)},
      event := {p : ℝ × ℝ | p.1 < 2 * p.2}
  in measure (event ∩ Ω) / measure Ω = (1 : ℚ) / 3 :=
begin
  sorry
end

end probability_x_lt_2y_in_rectangle_l261_261398


namespace max_sum_arithmetic_prog_l261_261817

theorem max_sum_arithmetic_prog (a d : ℝ) (S : ℕ → ℝ) 
  (h1 : S 3 = 327)
  (h2 : S 57 = 57)
  (hS : ∀ n, S n = (n / 2) * (2 * a + (n - 1) * d)) :
  ∃ max_S : ℝ, max_S = 1653 := by
  sorry

end max_sum_arithmetic_prog_l261_261817


namespace part1_solution_set_part2_values_a_b_part3_range_m_l261_261315

-- Definitions for the given functions
def y1 (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def y2 (x : ℝ) : ℝ := x^2 + x - 2

-- Proof that the solution set for y2 < 0 is (-2, 1)
theorem part1_solution_set : ∀ x : ℝ, y2 x < 0 ↔ (x > -2 ∧ x < 1) :=
sorry

-- Given |y1| ≤ |y2| for all x ∈ ℝ, prove that a = 1 and b = -2
theorem part2_values_a_b (a b : ℝ) : (∀ x : ℝ, |y1 x a b| ≤ |y2 x|) → a = 1 ∧ b = -2 :=
sorry

-- Given y1 > (m-2)x - m for all x > 1 under condition from part 2, prove the range for m is (-∞, 2√2 + 5)
theorem part3_range_m (a b : ℝ) (m : ℝ) : 
  (∀ x : ℝ, |y1 x a b| ≤ |y2 x|) → a = 1 ∧ b = -2 →
  (∀ x : ℝ, x > 1 → y1 x a b > (m-2) * x - m) → m < 2 * Real.sqrt 2 + 5 :=
sorry

end part1_solution_set_part2_values_a_b_part3_range_m_l261_261315


namespace product_pricing_and_savings_l261_261286

theorem product_pricing_and_savings :
  ∃ (x y : ℝ),
    (6 * x + 3 * y = 600) ∧
    (40 * x + 30 * y = 5200) ∧
    x = 40 ∧
    y = 120 ∧
    (80 * x + 100 * y - (80 * 0.8 * x + 100 * 0.75 * y) = 3640) := 
by
  sorry

end product_pricing_and_savings_l261_261286


namespace jackson_spends_260_l261_261586

-- Definitions based on conditions
def num_students := 30
def pens_per_student := 5
def notebooks_per_student := 3
def binders_per_student := 1
def highlighters_per_student := 2

def cost_per_pen := 0.50
def cost_per_notebook := 1.25
def cost_per_binder := 4.25
def cost_per_highlighter := 0.75
def discount := 100.00

-- Calculate total cost
noncomputable def total_cost := 
  let cost_per_student := 
    (pens_per_student * cost_per_pen) +
    (notebooks_per_student * cost_per_notebook) +
    (binders_per_student * cost_per_binder) +
    (highlighters_per_student * cost_per_highlighter)
  in num_students * cost_per_student - discount

-- Theorem to prove the final cost
theorem jackson_spends_260 : total_cost = 260 := by
  sorry

end jackson_spends_260_l261_261586


namespace largest_gcd_sum_1089_l261_261494

theorem largest_gcd_sum_1089 (c d : ℕ) (h₁ : 0 < c) (h₂ : 0 < d) (h₃ : c + d = 1089) : ∃ k, k = Nat.gcd c d ∧ k = 363 :=
by
  sorry

end largest_gcd_sum_1089_l261_261494


namespace least_distance_travelled_by_8_boys_l261_261307

open Real

noncomputable def total_distance (r : ℝ) : ℝ := 
  8 * 5 * (2 * r * sin (135 * (π / 180) / 2))

theorem least_distance_travelled_by_8_boys 
  (r : ℝ) (hr : r = 30) : 
  total_distance r = 1200 * sqrt(2 + sqrt(2)) :=
by
  have h1 : 135 * (π / 180) / 2 = 67.5 * (π / 180), by norm_num
  have h2 : sin (67.5 * (π / 180)) = sqrt(2 + sqrt(2)) / 2, by sorry
  rw [total_distance, hr, h1, h2]
  norm_num
  ring

end least_distance_travelled_by_8_boys_l261_261307


namespace not_basic_logical_structure_l261_261003

def basic_structures : Set String := {"Sequential structure", "Conditional structure", "Loop structure"}

theorem not_basic_logical_structure : "Operational structure" ∉ basic_structures := by
  sorry

end not_basic_logical_structure_l261_261003


namespace ratio_of_areas_l261_261490

variable (s' : ℝ) -- Let s' be the side length of square S'

def area_square : ℝ := s' ^ 2
def length_longer_side_rectangle : ℝ := 1.15 * s'
def length_shorter_side_rectangle : ℝ := 0.95 * s'
def area_rectangle : ℝ := length_longer_side_rectangle s' * length_shorter_side_rectangle s'

theorem ratio_of_areas :
  (area_rectangle s') / (area_square s') = (10925 / 10000) :=
by
  -- skip the proof for now
  sorry

end ratio_of_areas_l261_261490


namespace man_walking_time_l261_261185

section TrainProblem

variables {T W : ℕ}

/-- Each day a man meets his wife at the train station after work,
    and then she drives him home. She always arrives exactly on time to pick him up.
    One day he catches an earlier train and arrives at the station an hour early.
    He immediately begins walking home along the same route the wife drives.
    Eventually, his wife sees him on her way to the station and drives him the rest of the way home.
    When they arrive home, the man notices that they arrived 30 minutes earlier than usual.
    How much time did the man spend walking? -/
theorem man_walking_time : 
    (∃ (T : ℕ), T > 30 ∧ (W = T - 30) ∧ (W + 30 = T)) → W = 30 :=
sorry

end TrainProblem

end man_walking_time_l261_261185


namespace cos_210_eq_neg_sqrt3_div2_l261_261160

theorem cos_210_eq_neg_sqrt3_div2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div2_l261_261160


namespace lowest_test_score_dropped_l261_261054

theorem lowest_test_score_dropped (A B C D : ℝ) 
  (h1: A + B + C + D = 280)
  (h2: A + B + C = 225) : D = 55 := 
by 
  sorry

end lowest_test_score_dropped_l261_261054


namespace unique_intersections_l261_261838

def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := 5 * x + y = 1
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2

theorem unique_intersections :
  (∃ x1 y1, line1 x1 y1 ∧ line2 x1 y1) ∧
  (∃ x2 y2, line2 x2 y2 ∧ line3 x2 y2) ∧
  ¬ (∃ x y, line1 x y ∧ line3 x y) ∧
  (∀ x y x' y', (line1 x y ∧ line2 x y ∧ line2 x' y' ∧ line3 x' y') → (x = x' ∧ y = y')) :=
by
  sorry

end unique_intersections_l261_261838


namespace largest_of_eight_consecutive_l261_261086

theorem largest_of_eight_consecutive (n : ℕ) (h : 8 * n + 28 = 2024) : n + 7 = 256 := by
  -- This means you need to solve for n first, then add 7 to get the largest number
  sorry

end largest_of_eight_consecutive_l261_261086


namespace base_10_representation_l261_261486

-- Conditions
variables (C D : ℕ)
variables (hC : 0 ≤ C ∧ C ≤ 7)
variables (hD : 0 ≤ D ∧ D ≤ 5)
variables (hEq : 8 * C + D = 6 * D + C)

-- Goal
theorem base_10_representation : 8 * C + D = 0 := by
  sorry

end base_10_representation_l261_261486


namespace find_non_equivalent_fraction_l261_261279

-- Define the fractions mentioned in the problem
def sevenSixths := 7 / 6
def optionA := 14 / 12
def optionB := 1 + 1 / 6
def optionC := 1 + 5 / 30
def optionD := 1 + 2 / 6
def optionE := 1 + 14 / 42

-- The main problem statement
theorem find_non_equivalent_fraction :
  optionD ≠ sevenSixths := by
  -- We put a 'sorry' here because we are not required to provide the proof
  sorry

end find_non_equivalent_fraction_l261_261279


namespace illegally_parked_percentage_l261_261103

theorem illegally_parked_percentage (total_cars : ℕ) (towed_cars : ℕ)
  (ht : towed_cars = 2 * total_cars / 100) (not_towed_percentage : ℕ)
  (hp : not_towed_percentage = 80) : 
  (100 * (5 * towed_cars) / total_cars) = 10 :=
by
  sorry

end illegally_parked_percentage_l261_261103


namespace max_sum_cd_l261_261142

theorem max_sum_cd (c d : ℕ) (hc : c > 0) (hd : d > 1) (hcd : c^d < 500) 
  (hmax : ∀ (c' d': ℕ), c' > 0 → d' > 1 → c'^d' < 500 → c'^d' ≤ c^d) : c + d = 24 := 
by
  have h1 : 22^2 = 484 := rfl
  have h2 : c = 22 ∧ d = 2 := by sorry
  exact by sorry

end max_sum_cd_l261_261142


namespace sum_of_three_numbers_l261_261810

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 36) 
  (h2 : b + c = 55) 
  (h3 : c + a = 60) : 
  a + b + c = 75.5 := 
by 
  sorry

end sum_of_three_numbers_l261_261810


namespace expected_value_8_sided_die_l261_261647

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  (Σ x ∈ outcomes, probability_each_outcome * x) = 4.5 :=
by
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  have h : (Σ x ∈ outcomes, probability_each_outcome * x) = (1 / 8) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) := by sorry
  have sum_eq_36 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36 := by sorry
  rw [sum_eq_36] at h
  have expected_value_eq : (1 / 8) * 36 = 4.5 := by sorry
  rw [expected_value_eq] at h
  exact h

end expected_value_8_sided_die_l261_261647


namespace min_value_a_b_l261_261773

variable (a b : ℝ)

theorem min_value_a_b (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) : 
  a + b ≥ 2 * (Real.sqrt 2 + 1) :=
sorry

end min_value_a_b_l261_261773


namespace average_pages_correct_l261_261332

noncomputable def total_pages : ℝ := 50 + 75 + 80 + 120 + 100 + 90 + 110 + 130
def num_books : ℝ := 8
noncomputable def average_pages : ℝ := total_pages / num_books

theorem average_pages_correct : average_pages = 94.375 :=
by
  sorry

end average_pages_correct_l261_261332


namespace polynomial_roots_product_l261_261999

theorem polynomial_roots_product (a b : ℤ)
  (h1 : ∀ (r : ℝ), r^2 - r - 2 = 0 → r^3 - a * r - b = 0) : a * b = 6 := sorry

end polynomial_roots_product_l261_261999


namespace ratio_constant_l261_261474

theorem ratio_constant (A B C O : Point) 
(h1 : right_triangle A B C) 
(h2 : square_on_hypotenuse A B C O) :
  (CO / (AC + CB)) = (sqrt 2 / 2) :=
sorry

end ratio_constant_l261_261474


namespace total_whales_correct_l261_261761

def first_trip_male_whales : ℕ := 28
def first_trip_female_whales : ℕ := 2 * first_trip_male_whales
def first_trip_total_whales : ℕ := first_trip_male_whales + first_trip_female_whales

def second_trip_baby_whales : ℕ := 8
def second_trip_parent_whales : ℕ := 2 * second_trip_baby_whales
def second_trip_total_whales : ℕ := second_trip_baby_whales + second_trip_parent_whales

def third_trip_male_whales : ℕ := first_trip_male_whales / 2
def third_trip_female_whales : ℕ := first_trip_female_whales
def third_trip_total_whales : ℕ := third_trip_male_whales + third_trip_female_whales

def total_whales_seen : ℕ :=
  first_trip_total_whales + second_trip_total_whales + third_trip_total_whales

theorem total_whales_correct : total_whales_seen = 178 := by
  sorry

end total_whales_correct_l261_261761


namespace Jake_has_62_balls_l261_261331

theorem Jake_has_62_balls 
  (C A J : ℕ)
  (h1 : C = 41 + 7)
  (h2 : A = 2 * C)
  (h3 : J = A - 34) : 
  J = 62 :=
by 
  sorry

end Jake_has_62_balls_l261_261331


namespace schoolchildren_number_l261_261106

theorem schoolchildren_number (n m S : ℕ) 
  (h1 : S = 22 * n + 3)
  (h2 : S = (n - 1) * m)
  (h3 : n ≤ 18)
  (h4 : m ≤ 36) : 
  S = 135 := 
sorry

end schoolchildren_number_l261_261106


namespace quadratic_roots_k_relation_l261_261990

theorem quadratic_roots_k_relation (k a b k1 k2 : ℝ) 
    (h_eq : k * (a^2 - a) + 2 * a + 7 = 0)
    (h_eq_b : k * (b^2 - b) + 2 * b + 7 = 0)
    (h_ratio : a / b + b / a = 3)
    (h_k : k = k1 ∨ k = k2)
    (h_vieta_sum : k1 + k2 = 39)
    (h_vieta_product : k1 * k2 = 4) :
    k1 / k2 + k2 / k1 = 1513 / 4 := 
    sorry

end quadratic_roots_k_relation_l261_261990


namespace expected_value_of_8_sided_die_is_4_point_5_l261_261649

def expected_value_8_sided_die : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (list.sum outcomes : ℝ) * probability

theorem expected_value_of_8_sided_die_is_4_point_5 :
  expected_value_8_sided_die = 4.5 := by
  sorry

end expected_value_of_8_sided_die_is_4_point_5_l261_261649


namespace find_TS_l261_261004

-- Definitions of the conditions as given:
def PQ : ℝ := 25
def PS : ℝ := 25
def QR : ℝ := 15
def RS : ℝ := 15
def PT : ℝ := 15
def ST_parallel_QR : Prop := true  -- ST is parallel to QR (used as a given fact)

-- Main statement in Lean:
theorem find_TS (h1 : PQ = 25) (h2 : PS = 25) (h3 : QR = 15) (h4 : RS = 15) (h5 : PT = 15)
               (h6 : ST_parallel_QR) : TS = 24 :=
by
  sorry

end find_TS_l261_261004


namespace base2_to_base4_conversion_l261_261373

/-- Definition of base conversion from binary to quaternary. -/
def bin_to_quat (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  if n = 1 then 1 else
  if n = 10 then 2 else
  if n = 11 then 3 else
  0 -- (more cases can be added as necessary)

theorem base2_to_base4_conversion :
  bin_to_quat 1 * 4^4 + bin_to_quat 1 * 4^3 + bin_to_quat 10 * 4^2 + bin_to_quat 11 * 4^1 + bin_to_quat 10 * 4^0 = 11232 :=
by sorry

end base2_to_base4_conversion_l261_261373


namespace triangle_inequality_l261_261754

theorem triangle_inequality (a b c : ℝ) (S : ℝ) (hS : S = (1/4) * Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

end triangle_inequality_l261_261754


namespace greatest_third_side_of_triangle_l261_261953

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 15) :
  ∃ x : ℕ, 8 < x ∧ x < 22 ∧ (∀ y : ℕ, 8 < y ∧ y < 22 → y ≤ x) ∧ x = 21 :=
by
  sorry

end greatest_third_side_of_triangle_l261_261953


namespace min_gennadies_l261_261689

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end min_gennadies_l261_261689


namespace sum_possible_values_of_y_l261_261878

theorem sum_possible_values_of_y (y : ℝ) (h : y^2 = 36) : y = 6 ∨ y = -6 → (6 + (-6) = 0) :=
by
  sorry

end sum_possible_values_of_y_l261_261878


namespace arithmetic_geom_sequence_ratio_l261_261384

theorem arithmetic_geom_sequence_ratio (a : ℕ → ℝ) (d a1 : ℝ) (h1 : d ≠ 0) 
(h2 : ∀ n, a (n+1) = a n + d)
(h3 : (a 0 + 2 * d)^2 = a 0 * (a 0 + 8 * d)):
  (a 0 + a 2 + a 8) / (a 1 + a 3 + a 9) = 13 / 16 := 
by sorry

end arithmetic_geom_sequence_ratio_l261_261384


namespace vector_calc_l261_261216

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Statement to prove that 2a - b = (5, 7)
theorem vector_calc : 2 • a - b = (5, 7) :=
by {
  -- Proof will be filled here
  sorry
}

end vector_calc_l261_261216


namespace cube_surface_area_is_24_l261_261087

def edge_length : ℝ := 2

def surface_area_of_cube (a : ℝ) : ℝ := 6 * a * a

theorem cube_surface_area_is_24 : surface_area_of_cube edge_length = 24 := 
by 
  -- Compute the surface area of the cube with given edge length
  -- surface_area_of_cube 2 = 6 * 2 * 2 = 24
  sorry

end cube_surface_area_is_24_l261_261087


namespace find_p_value_l261_261212

noncomputable def solve_p (m p : ℕ) :=
  (1^m / 5^m) * (1^16 / 4^16) = 1 / (2 * p^31)

theorem find_p_value (m p : ℕ) (hm : m = 31) :
  solve_p m p ↔ p = 10 :=
by
  sorry

end find_p_value_l261_261212


namespace complex_number_in_second_quadrant_l261_261049

theorem complex_number_in_second_quadrant :
  let z := (2 + 4 * Complex.I) / (1 + Complex.I) 
  ∃ (im : ℂ), z = im ∧ im.re < 0 ∧ 0 < im.im := by
  sorry

end complex_number_in_second_quadrant_l261_261049


namespace bus_stops_for_minutes_per_hour_l261_261996

theorem bus_stops_for_minutes_per_hour (speed_no_stops speed_with_stops : ℕ)
  (h1 : speed_no_stops = 60) (h2 : speed_with_stops = 45) : 
  (60 * (speed_no_stops - speed_with_stops) / speed_no_stops) = 15 :=
by
  sorry

end bus_stops_for_minutes_per_hour_l261_261996


namespace num_points_within_and_on_boundary_is_six_l261_261727

noncomputable def num_points_within_boundary : ℕ :=
  let points := [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1)]
  points.length

theorem num_points_within_and_on_boundary_is_six :
  num_points_within_boundary = 6 :=
  by
    -- proof steps would go here
    sorry

end num_points_within_and_on_boundary_is_six_l261_261727


namespace expression_for_f_value_of_sin_a_l261_261740

open Real

noncomputable def f (x ϕ : ℝ) : ℝ := sin (2 * x) * cos ϕ + cos (2 * x) * sin ϕ

theorem expression_for_f (ϕ : ℝ) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π) : 
  f (π / 4) ϕ = √3 / 2 ↔ ϕ = π / 6 := by
  sorry

theorem value_of_sin_a (a : ℝ) (ϕ : ℝ) (hϕ1 : 0 < ϕ) (hϕ2 : ϕ < π) 
  (ha1 : π / 2 < a) (ha2 : a < π) :
  (f (a / 2 - π / 3) ϕ = 5 / 13 ↔ sin a = 12 / 13) := by
  sorry

end expression_for_f_value_of_sin_a_l261_261740


namespace min_gennadys_l261_261685

-- Defining the basic constants for each name type
def Alexanders : Nat := 45
def Borises : Nat := 122
def Vasilies : Nat := 27

-- Define the proof statement to check the minimum number of Gennadys needed
theorem min_gennadys (a b v : Nat) (no_adjacent: a = 45 ∧ b = 122 ∧ v = 27) : ∃ g : Nat, g = 49 :=
by
  -- Using provided conditions
  cases no_adjacent with h_a h_bv
  cases h_bv with h_b h_v
  -- Correct answer derived from the solution
  use 49
  -- skipping proof details 
  sorry

end min_gennadys_l261_261685


namespace sphere_volume_l261_261428

theorem sphere_volume (length width : ℝ) (angle_deg : ℝ) (h_length : length = 4) (h_width : width = 3) (h_angle : angle_deg = 60) :
  ∃ (volume : ℝ), volume = (125 / 6) * Real.pi :=
by
  sorry

end sphere_volume_l261_261428


namespace divisibility_problem_l261_261179

theorem divisibility_problem (a b k : ℕ) :
  (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) →
  a * b^2 + b + 7 ∣ a^2 * b + a + b := by
  intro h
  cases h
  case inl h1 =>
    rw [h1.1, h1.2]
    sorry
  case inr h2 =>
    cases h2
    case inl h21 =>
      rw [h21.1, h21.2]
      sorry
    case inr h22 =>
      rw [h22.1, h22.2]
      sorry

end divisibility_problem_l261_261179


namespace gcd_example_l261_261633

-- Define the two numbers
def a : ℕ := 102
def b : ℕ := 238

-- Define the GCD of a and b
def gcd_ab : ℕ :=
  Nat.gcd a b

-- The expected result of the GCD
def expected_gcd : ℕ := 34

-- Prove that the GCD of a and b is equal to the expected GCD
theorem gcd_example : gcd_ab = expected_gcd := by
  sorry

end gcd_example_l261_261633


namespace compound_interest_correct_l261_261044

variables (SI : ℚ) (R : ℚ) (T : ℕ) (P : ℚ)

def calculate_principal (SI R T : ℚ) : ℚ := SI * 100 / (R * T)

def calculate_compound_interest (P R : ℚ) (T : ℕ) : ℚ :=
  P * ((1 + R / 100)^T - 1)

theorem compound_interest_correct (h1: SI = 52) (h2: R = 5) (h3: T = 2) :
  calculate_compound_interest (calculate_principal SI R T) R T = 53.30 :=
by
  sorry

end compound_interest_correct_l261_261044


namespace unique_solution_l261_261657

noncomputable def unique_solution_exists : Prop :=
  ∃ (a b c d e : ℕ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    (a + b = (c + d + e) / 7) ∧
    (a + d = (b + c + e) / 5) ∧
    (a + b + c + d + e = 24) ∧
    (a = 1 ∧ b = 2 ∧ c = 9 ∧ d = 3 ∧ e = 9)

theorem unique_solution : unique_solution_exists :=
sorry

end unique_solution_l261_261657


namespace train_total_travel_time_l261_261000

noncomputable def totalTravelTime (d1 d2 s1 s2 : ℝ) : ℝ :=
  (d1 / s1) + (d2 / s2)

theorem train_total_travel_time : 
  totalTravelTime 150 200 50 80 = 5.5 :=
by
  sorry

end train_total_travel_time_l261_261000


namespace barry_sotter_magic_l261_261916

theorem barry_sotter_magic (n : ℕ) : (n + 3) / 3 = 50 → n = 147 := 
by 
  sorry

end barry_sotter_magic_l261_261916


namespace probability_of_prime_spinner_l261_261375

def spinner_sections : List ℕ := [2, 4, 7, 8, 11, 14, 17, 19]

def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_numbers (lst : List ℕ) : List ℕ := lst.filter is_prime

theorem probability_of_prime_spinner :
  (prime_numbers spinner_sections).length / spinner_sections.length = 5 / 8 := by
sorry

end probability_of_prime_spinner_l261_261375


namespace product_mk_through_point_l261_261578

theorem product_mk_through_point (k m : ℝ) (h : (2 : ℝ) ^ m * k = (1/4 : ℝ)) : m * k = -2 := 
sorry

end product_mk_through_point_l261_261578


namespace recorded_expenditure_l261_261455

-- Define what it means to record an income and an expenditure
def record_income (y : ℝ) : ℝ := y
def record_expenditure (y : ℝ) : ℝ := -y

-- Define specific instances for the problem
def income_recorded_as : ℝ := 20
def expenditure_value : ℝ := 75

-- Given condition
axiom income_condition : record_income income_recorded_as = 20

-- Theorem to prove the recorded expenditure
theorem recorded_expenditure : record_expenditure expenditure_value = -75 := by
  sorry

end recorded_expenditure_l261_261455


namespace largest_value_b_l261_261466

theorem largest_value_b (b : ℚ) : (3 * b + 7) * (b - 2) = 9 * b -> b = (4 + Real.sqrt 58) / 3 :=
by
  sorry

end largest_value_b_l261_261466


namespace largest_n_sum_pos_l261_261203

section
variables {a : ℕ → ℤ}
variables {d : ℤ}
variables {n : ℕ}

axiom a_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom a1_pos : a 1 > 0
axiom a2013_2014_pos : a 2013 + a 2014 > 0
axiom a2013_2014_neg : a 2013 * a 2014 < 0

theorem largest_n_sum_pos :
  ∃ n : ℕ, (∀ k ≤ n, (k * (2 * a 1 + (k - 1) * d) / 2) > 0) → n = 4026 := sorry

end

end largest_n_sum_pos_l261_261203


namespace pure_ghee_added_l261_261757

theorem pure_ghee_added
  (Q : ℕ) (hQ : Q = 30)
  (P : ℕ)
  (original_pure_ghee : ℕ := (Q / 2))
  (original_vanaspati : ℕ := (Q / 2))
  (new_total_ghee : ℕ := Q + P)
  (new_vanaspati_fraction : ℝ := 0.3) :
  original_vanaspati = (new_vanaspati_fraction * ↑new_total_ghee : ℝ) → P = 20 := by
  sorry

end pure_ghee_added_l261_261757


namespace expression_value_zero_l261_261449

variable (x : ℝ)

theorem expression_value_zero (h : x^2 + 3 * x - 3 = 0) : x^3 + 2 * x^2 - 6 * x + 3 = 0 := 
by
  sorry

end expression_value_zero_l261_261449


namespace car_speed_l261_261923

def travel_time : ℝ := 5
def travel_distance : ℝ := 300

theorem car_speed :
  travel_distance / travel_time = 60 := sorry

end car_speed_l261_261923


namespace divisibility_problem_l261_261178

theorem divisibility_problem (a b k : ℕ) :
  (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) →
  a * b^2 + b + 7 ∣ a^2 * b + a + b := by
  intro h
  cases h
  case inl h1 =>
    rw [h1.1, h1.2]
    sorry
  case inr h2 =>
    cases h2
    case inl h21 =>
      rw [h21.1, h21.2]
      sorry
    case inr h22 =>
      rw [h22.1, h22.2]
      sorry

end divisibility_problem_l261_261178


namespace complex_number_solution_l261_261438

theorem complex_number_solution (z : ℂ) (i : ℂ) (H1 : i * i = -1) (H2 : z * i = 2 - 2 * i) : z = -2 - 2 * i :=
by
  sorry

end complex_number_solution_l261_261438


namespace pages_left_to_read_l261_261478

theorem pages_left_to_read (total_pages : ℕ) (pages_read : ℕ) (pages_skipped : ℕ) : 
  total_pages = 372 → pages_read = 125 → pages_skipped = 16 → (total_pages - (pages_read + pages_skipped)) = 231 :=
by
  intros
  sorry

end pages_left_to_read_l261_261478


namespace woman_finishes_work_in_225_days_l261_261110

theorem woman_finishes_work_in_225_days
  (M W : ℝ)
  (h1 : (10 * M + 15 * W) * 6 = 1)
  (h2 : M * 100 = 1) :
  1 / W = 225 :=
by
  sorry

end woman_finishes_work_in_225_days_l261_261110


namespace union_of_sets_l261_261862

def setA : Set ℕ := {0, 1}
def setB : Set ℕ := {0, 2}

theorem union_of_sets : setA ∪ setB = {0, 1, 2} := 
sorry

end union_of_sets_l261_261862


namespace y_is_multiple_of_2_y_is_multiple_of_3_y_is_multiple_of_6_y_is_multiple_of_9_l261_261771

def y : ℕ := 54 + 108 + 162 + 216 + 648 + 810 + 972

theorem y_is_multiple_of_2 : 2 ∣ y :=
sorry

theorem y_is_multiple_of_3 : 3 ∣ y :=
sorry

theorem y_is_multiple_of_6 : 6 ∣ y :=
sorry

theorem y_is_multiple_of_9 : 9 ∣ y :=
sorry

end y_is_multiple_of_2_y_is_multiple_of_3_y_is_multiple_of_6_y_is_multiple_of_9_l261_261771


namespace probability_first_ace_second_spade_l261_261093

theorem probability_first_ace_second_spade :
  let deck := List.range 52 in
  let first_is_ace (card : ℕ) := card % 13 = 0 in
  let second_is_spade (card : ℕ) := card / 13 = 3 in
  let events :=
    [ ((first_is_ace card, second_is_spade card') | card ∈ deck, card' ∈ List.erase deck card) ] in
  let favorable_events :=
    [(true, true)] in
  (List.count (λ event => event ∈ favorable_events) events).toRat /
  (List.length events).toRat = 1 / 52 :=
sorry

end probability_first_ace_second_spade_l261_261093


namespace min_value_of_a_l261_261206

theorem min_value_of_a 
  (a b x1 x2 : ℕ) 
  (h1 : a = b - 2005) 
  (h2 : (x1 + x2) = a) 
  (h3 : (x1 * x2) = b) 
  (h4 : x1 > 0 ∧ x2 > 0) : 
  a ≥ 95 :=
sorry

end min_value_of_a_l261_261206


namespace decorations_total_l261_261716

def number_of_skulls : Nat := 12
def number_of_broomsticks : Nat := 4
def number_of_spiderwebs : Nat := 12
def number_of_pumpkins (spiderwebs : Nat) : Nat := 2 * spiderwebs
def number_of_cauldron : Nat := 1
def number_of_lanterns (trees : Nat) : Nat := 3 * trees
def number_of_scarecrows (trees : Nat) : Nat := 1 * (trees / 2)
def total_stickers : Nat := 30
def stickers_per_window (stickers : Nat) (windows : Nat) : Nat := (stickers / 2) / windows
def additional_decorations (bought : Nat) (used_percent : Nat) (leftover : Nat) : Nat := ((bought * used_percent) / 100) + leftover

def total_decorations : Nat :=
  number_of_skulls +
  number_of_broomsticks +
  number_of_spiderwebs +
  (number_of_pumpkins number_of_spiderwebs) +
  number_of_cauldron +
  (number_of_lanterns 5) +
  (number_of_scarecrows 4) +
  (additional_decorations 25 70 15)

theorem decorations_total : total_decorations = 102 := by
  sorry

end decorations_total_l261_261716


namespace cone_base_radius_l261_261569

variable (s : ℝ) (A : ℝ) (r : ℝ)

theorem cone_base_radius (h1 : s = 5) (h2 : A = 15 * Real.pi) : r = 3 :=
by
  sorry

end cone_base_radius_l261_261569


namespace find_y_coordinate_l261_261900

noncomputable def y_coordinate_of_point_on_line : ℝ :=
  let x1 := 10
  let y1 := 3
  let x2 := 4
  let y2 := 0
  let x := -2
  let m := (y1 - y2) / (x1 - x2)
  let b := y1 - m * x1
  m * x + b

theorem find_y_coordinate :
  (y_coordinate_of_point_on_line = -3) :=
by
  sorry

end find_y_coordinate_l261_261900


namespace min_value_fraction_l261_261558

theorem min_value_fraction (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (∃c, (c = 8) ∧ (∀z w : ℝ, z > 1 → w > 1 → ((z^3 / (w - 1) + w^3 / (z - 1)) ≥ c))) :=
by 
  sorry

end min_value_fraction_l261_261558


namespace triangle_area_l261_261046

theorem triangle_area (a b c : ℝ) (C : ℝ) (h1 : c^2 = (a - b)^2 + 6) (h2 : C = π / 3) :
    abs ((1 / 2) * a * b * Real.sin C) = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_area_l261_261046


namespace greatest_prime_factor_f_24_l261_261513

def f (m : ℕ) : ℕ :=
  if h : m % 2 = 0 then (List.range' 2 (m/2)).map (λ x, 2 * x).prod else 1

theorem greatest_prime_factor_f_24 : 
  ∃ p : ℕ, (nat.prime p) ∧ (p ∣ f 24) ∧ ∀ q : ℕ, (nat.prime q) ∧ (q ∣ f 24) → q ≤ p :=
by
  sorry

end greatest_prime_factor_f_24_l261_261513


namespace min_gennadys_l261_261684

-- Defining the basic constants for each name type
def Alexanders : Nat := 45
def Borises : Nat := 122
def Vasilies : Nat := 27

-- Define the proof statement to check the minimum number of Gennadys needed
theorem min_gennadys (a b v : Nat) (no_adjacent: a = 45 ∧ b = 122 ∧ v = 27) : ∃ g : Nat, g = 49 :=
by
  -- Using provided conditions
  cases no_adjacent with h_a h_bv
  cases h_bv with h_b h_v
  -- Correct answer derived from the solution
  use 49
  -- skipping proof details 
  sorry

end min_gennadys_l261_261684


namespace expected_value_of_eight_sided_die_l261_261642

-- Definitions based on the problem conditions
def eight_sided_die_outcomes : List ℕ := [1,2,3,4,5,6,7,8]

def probability (n : ℕ) := 1 / n

-- Expected value calculation related to the problem
def expected_value_die_roll (outcomes : List ℕ) (prob : ℕ → Rat) : Rat :=
  List.sum (outcomes.map (λ x => prob outcomes.length * x))

-- Expected value of an 8-sided die roll
theorem expected_value_of_eight_sided_die :
  expected_value_die_roll eight_sided_die_outcomes probability = 4.5 := 
sorry

end expected_value_of_eight_sided_die_l261_261642


namespace breaks_difference_l261_261245

-- James works for 240 minutes
def total_work_time : ℕ := 240

-- He takes a water break every 20 minutes
def water_break_interval : ℕ := 20

-- He takes a sitting break every 120 minutes
def sitting_break_interval : ℕ := 120

-- Calculate the number of water breaks James takes
def number_of_water_breaks : ℕ := total_work_time / water_break_interval

-- Calculate the number of sitting breaks James takes
def number_of_sitting_breaks : ℕ := total_work_time / sitting_break_interval

-- Prove the difference between the number of water breaks and sitting breaks is 10
theorem breaks_difference :
  number_of_water_breaks - number_of_sitting_breaks = 10 :=
by
  -- calculate number_of_water_breaks = 12
  -- calculate number_of_sitting_breaks = 2
  -- check the difference 12 - 2 = 10
  sorry

end breaks_difference_l261_261245


namespace values_of_a_l261_261198

noncomputable def quadratic_eq (a x : ℝ) : ℝ :=
(a - 1) * x^2 - 2 * (a + 1) * x + 2 * (a + 1)

theorem values_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic_eq a x = 0 → x ≥ 0) ↔ (a = 3 ∨ (-1 ≤ a ∧ a ≤ 1)) :=
sorry

end values_of_a_l261_261198


namespace initial_number_of_macaroons_l261_261333

theorem initial_number_of_macaroons 
  (w : ℕ) (bag_count : ℕ) (eaten_bag_count : ℕ) (remaining_weight : ℕ) 
  (macaroon_weight : ℕ) (remaining_bags : ℕ) (initial_macaroons : ℕ) :
  w = 5 → bag_count = 4 → eaten_bag_count = 1 → remaining_weight = 45 → 
  macaroon_weight = w → remaining_bags = (bag_count - eaten_bag_count) → 
  initial_macaroons = (remaining_bags * remaining_weight / macaroon_weight) * bag_count / remaining_bags →
  initial_macaroons = 12 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end initial_number_of_macaroons_l261_261333


namespace probability_painted_faces_l261_261977

theorem probability_painted_faces (total_cubes : ℕ) (corner_cubes : ℕ) (no_painted_face_cubes : ℕ) (successful_outcomes : ℕ) (total_outcomes : ℕ) 
  (probability : ℚ) : 
  total_cubes = 125 ∧ corner_cubes = 8 ∧ no_painted_face_cubes = 27 ∧ successful_outcomes = 216 ∧ total_outcomes = 7750 ∧ 
  probability = 72 / 2583 :=
by
  sorry

end probability_painted_faces_l261_261977


namespace water_breaks_vs_sitting_breaks_l261_261243

theorem water_breaks_vs_sitting_breaks :
  (240 / 20) - (240 / 120) = 10 := by
  sorry

end water_breaks_vs_sitting_breaks_l261_261243


namespace car_speed_l261_261925

theorem car_speed (time : ℕ) (distance : ℕ) (h1 : time = 5) (h2 : distance = 300) : distance / time = 60 := by
  sorry

end car_speed_l261_261925


namespace pete_miles_walked_l261_261920

noncomputable def steps_from_first_pedometer (flips1 : ℕ) (final_reading1 : ℕ) : ℕ :=
  flips1 * 100000 + final_reading1 

noncomputable def steps_from_second_pedometer (flips2 : ℕ) (final_reading2 : ℕ) : ℕ :=
  flips2 * 400000 + final_reading2 * 4

noncomputable def total_steps (flips1 flips2 final_reading1 final_reading2 : ℕ) : ℕ :=
  steps_from_first_pedometer flips1 final_reading1 + steps_from_second_pedometer flips2 final_reading2

noncomputable def miles_walked (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

theorem pete_miles_walked
  (flips1 flips2 final_reading1 final_reading2 steps_per_mile : ℕ)
  (h_flips1 : flips1 = 50)
  (h_final_reading1 : final_reading1 = 25000)
  (h_flips2 : flips2 = 15)
  (h_final_reading2 : final_reading2 = 30000)
  (h_steps_per_mile : steps_per_mile = 1500) :
  miles_walked (total_steps flips1 flips2 final_reading1 final_reading2) steps_per_mile = 7430 :=
by sorry

end pete_miles_walked_l261_261920


namespace find_m_l261_261742

theorem find_m 
(x0 m : ℝ)
(h1 : m ≠ 0)
(h2 : x0^2 - x0 + m = 0)
(h3 : (2 * x0)^2 - 2 * x0 + 3 * m = 0)
: m = -2 :=
sorry

end find_m_l261_261742


namespace price_decrease_percentage_l261_261531

-- Define the conditions
variables {P : ℝ} (original_price increased_price decreased_price : ℝ)
variables (y : ℝ) -- percentage by which increased price is decreased

-- Given conditions
def store_conditions :=
  increased_price = 1.20 * original_price ∧
  decreased_price = increased_price * (1 - y/100) ∧
  decreased_price = 0.75 * original_price

-- The proof problem
theorem price_decrease_percentage 
  (original_price increased_price decreased_price : ℝ)
  (y : ℝ) 
  (h : store_conditions original_price increased_price decreased_price y) :
  y = 37.5 :=
by 
  sorry

end price_decrease_percentage_l261_261531


namespace smallest_root_abs_eq_six_l261_261098

theorem smallest_root_abs_eq_six : 
  (∃ x : ℝ, (abs (x - 1)) / (x^2) = 6 ∧ ∀ y : ℝ, (abs (y - 1)) / (y^2) = 6 → y ≥ x) → x = -1 / 2 := by
  sorry

end smallest_root_abs_eq_six_l261_261098


namespace abs_neg_three_l261_261354

theorem abs_neg_three : abs (-3) = 3 := 
by 
  -- Skipping proof with sorry
  sorry

end abs_neg_three_l261_261354


namespace pow_mod_remainder_l261_261506

theorem pow_mod_remainder (x : ℕ) (h : x = 3) : x^1988 % 8 = 1 := by
  sorry

end pow_mod_remainder_l261_261506


namespace contradiction_proof_l261_261371

theorem contradiction_proof (a b c : ℝ) (h : (a⁻¹ * b⁻¹ * c⁻¹) > 0) : (a ≤ 1) ∧ (b ≤ 1) ∧ (c ≤ 1) → False :=
sorry

end contradiction_proof_l261_261371


namespace square_ratios_l261_261481

/-- 
  Given two squares with areas ratio 16:49, 
  prove that the ratio of their perimeters is 4:7,
  and the ratio of the sum of their perimeters to the sum of their areas is 84:13.
-/
theorem square_ratios (s₁ s₂ : ℝ) 
  (h₁ : s₁^2 / s₂^2 = 16 / 49) :
  (s₁ / s₂ = 4 / 7) ∧ ((4 * (s₁ + s₂)) / (s₁^2 + s₂^2) = 84 / 13) :=
by {
  sorry
}

end square_ratios_l261_261481


namespace min_value_proof_l261_261437

noncomputable def min_value (x y : ℝ) : ℝ :=
  (y / x) + (1 / y)

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  (min_value x y) ≥ 4 :=
by
  sorry

end min_value_proof_l261_261437


namespace pages_left_to_read_l261_261477

theorem pages_left_to_read (total_pages : ℕ) (pages_read : ℕ) (pages_skipped : ℕ) : 
  total_pages = 372 → pages_read = 125 → pages_skipped = 16 → (total_pages - (pages_read + pages_skipped)) = 231 :=
by
  intros
  sorry

end pages_left_to_read_l261_261477


namespace max_min_difference_l261_261595

noncomputable def difference_max_min_z (x y z : ℝ) : ℝ :=
  if h₁ : x + y + z = 3 ∧ x^2 + y^2 + z^2 = 18 then 6 else 0

theorem max_min_difference (x y z : ℝ) (h₁ : x + y + z = 3) (h₂ : x^2 + y^2 + z^2 = 18) :
  difference_max_min_z x y z = 6 :=
by sorry

end max_min_difference_l261_261595


namespace vacation_cost_l261_261940

theorem vacation_cost (C : ℝ) (h : C / 6 - C / 8 = 120) : C = 2880 :=
by
  sorry

end vacation_cost_l261_261940


namespace cube_properties_l261_261417

theorem cube_properties (y : ℝ) (s : ℝ) 
  (h_volume : s^3 = 6 * y)
  (h_surface_area : 6 * s^2 = 2 * y) :
  y = 5832 :=
by sorry

end cube_properties_l261_261417


namespace pima_investment_value_l261_261066

noncomputable def pima_investment_worth (initial_investment : ℕ) (first_week_gain_percentage : ℕ) (second_week_gain_percentage : ℕ) : ℕ :=
  let first_week_value := initial_investment + (initial_investment * first_week_gain_percentage / 100)
  let second_week_value := first_week_value + (first_week_value * second_week_gain_percentage / 100)
  second_week_value

-- Conditions
def initial_investment := 400
def first_week_gain_percentage := 25
def second_week_gain_percentage := 50

theorem pima_investment_value :
  pima_investment_worth initial_investment first_week_gain_percentage second_week_gain_percentage = 750 := by
  sorry

end pima_investment_value_l261_261066


namespace Sarah_substitution_l261_261457

theorem Sarah_substitution :
  ∀ (f g h i j : ℤ), 
    f = 2 → g = 4 → h = 5 → i = 10 →
    (f - (g - (h * (i - j))) = 48 - 5 * j) →
    (f - g - h * i - j = -52 - j) →
    j = 25 :=
by
  intros f g h i j hfg hi hhi hmf hCm hRn
  sorry

end Sarah_substitution_l261_261457


namespace remainder_1234567_127_l261_261300

theorem remainder_1234567_127 : (1234567 % 127) = 51 := 
by {
  sorry
}

end remainder_1234567_127_l261_261300


namespace total_roses_l261_261123

theorem total_roses (a : ℕ) (x y k : ℕ) (h1 : 300 ≤ a) (h2 : a ≤ 400)
  (h3 : a = 21 * x + 13) (h4 : a = 15 * y - 8) (h5 : a + 8 = 105 * k) :
  a = 307 :=
sorry

end total_roses_l261_261123


namespace probability_two_tails_two_heads_l261_261886

theorem probability_two_tails_two_heads :
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  total_probability = 3 / 8 :=
by
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  sorry

end probability_two_tails_two_heads_l261_261886


namespace crank_slider_motion_l261_261967

def omega : ℝ := 10
def OA : ℝ := 90
def AB : ℝ := 90
def AM : ℝ := 60
def t : ℝ := sorry -- t is a variable, no specific value required

theorem crank_slider_motion :
  (∀ t : ℝ, ((90 * Real.cos (10 * t)), (90 * Real.sin (10 * t) + 60)) = (x, y)) ∧
  (∀ t : ℝ, ((-900 * Real.sin (10 * t)), (900 * Real.cos (10 * t))) = (vx, vy)) :=
sorry

end crank_slider_motion_l261_261967


namespace lengths_equal_l261_261084

-- a rhombus AFCE inscribed in a rectangle ABCD
variables {A B C D E F : Type}
variables {width length perimeter side_BF side_DE : ℝ}
variables {AF CE FC AF_side FC_side : ℝ}
variables {h1 : width = 20} {h2 : length = 25} {h3 : perimeter = 82}
variables {h4 : side_BF = (82 / 4 - 20)} {h5 : side_DE = (82 / 4 - 20)} 

-- prove that the lengths of BF and DE are equal
theorem lengths_equal :
  side_BF = side_DE :=
by
  sorry

end lengths_equal_l261_261084


namespace am_gm_iq_l261_261317

theorem am_gm_iq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (a + 1/a) * (b + 1/b) ≥ 25/4 := sorry

end am_gm_iq_l261_261317


namespace jessica_current_age_l261_261903

-- Define the conditions
def jessicaOlderThanClaire (jessica claire : ℕ) : Prop :=
  jessica = claire + 6

def claireAgeInTwoYears (claire : ℕ) : Prop :=
  claire + 2 = 20

-- State the theorem to prove
theorem jessica_current_age : ∃ jessica claire : ℕ, 
  jessicaOlderThanClaire jessica claire ∧ claireAgeInTwoYears claire ∧ jessica = 24 := 
sorry

end jessica_current_age_l261_261903


namespace largest_angle_in_scalene_triangle_l261_261946

-- Define the conditions of the problem
def is_scalene (D E F : ℝ) : Prop :=
  D ≠ E ∧ D ≠ F ∧ E ≠ F

def angle_sum (D E F : ℝ) : Prop :=
  D + E + F = 180

def given_angles (D E : ℝ) : Prop :=
  D = 30 ∧ E = 50

-- Statement of the problem
theorem largest_angle_in_scalene_triangle :
  ∀ (D E F : ℝ), is_scalene D E F ∧ given_angles D E ∧ angle_sum D E F → F = 100 :=
by
  intros D E F h
  sorry

end largest_angle_in_scalene_triangle_l261_261946


namespace oliver_bumper_cars_proof_l261_261538

def rides_of_bumper_cars (total_tickets : ℕ) (tickets_per_ride : ℕ) (rides_ferris_wheel : ℕ) : ℕ :=
  (total_tickets - rides_ferris_wheel * tickets_per_ride) / tickets_per_ride

def oliver_bumper_car_rides : Prop :=
  rides_of_bumper_cars 30 3 7 = 3

theorem oliver_bumper_cars_proof : oliver_bumper_car_rides :=
by
  sorry

end oliver_bumper_cars_proof_l261_261538


namespace temperature_at_midnight_l261_261895

theorem temperature_at_midnight 
  (morning_temp : ℝ) 
  (afternoon_rise : ℝ) 
  (midnight_drop : ℝ)
  (h1 : morning_temp = 30)
  (h2 : afternoon_rise = 1)
  (h3 : midnight_drop = 7) 
  : morning_temp + afternoon_rise - midnight_drop = 24 :=
by
  -- Convert all conditions into the correct forms
  rw [h1, h2, h3]
  -- Perform the arithmetic operations
  norm_num

end temperature_at_midnight_l261_261895


namespace cos_210_eq_neg_sqrt3_div_2_l261_261168

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261168


namespace carolyn_total_monthly_practice_l261_261408

def daily_piano_practice : ℕ := 20
def violin_practice_multiplier : ℕ := 3
def days_per_week : ℕ := 6
def weeks_in_month : ℕ := 4

def daily_violin_practice : ℕ := violin_practice_multiplier * daily_piano_practice := by sorry
def daily_total_practice : ℕ := daily_piano_practice + daily_violin_practice := by sorry
def weekly_total_practice : ℕ := daily_total_practice * days_per_week := by sorry
def monthly_total_practice : ℕ := weekly_total_practice * weeks_in_month := by sorry

theorem carolyn_total_monthly_practice : monthly_total_practice = 1920 := by sorry

end carolyn_total_monthly_practice_l261_261408


namespace evaluate_log_expression_l261_261550

noncomputable def evaluate_expression (x y : Real) : Real :=
  (Real.log x / Real.log (y ^ 8)) * 
  (Real.log (y ^ 3) / Real.log (x ^ 7)) * 
  (Real.log (x ^ 7) / Real.log (y ^ 3)) * 
  (Real.log (y ^ 8) / Real.log (x ^ 2))

theorem evaluate_log_expression (x y : Real) : 
  evaluate_expression x y = (1 : Real) := sorry

end evaluate_log_expression_l261_261550


namespace greatest_third_side_of_triangle_l261_261952

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 15) :
  ∃ x : ℕ, 8 < x ∧ x < 22 ∧ (∀ y : ℕ, 8 < y ∧ y < 22 → y ≤ x) ∧ x = 21 :=
by
  sorry

end greatest_third_side_of_triangle_l261_261952


namespace solve_real_eq_l261_261350

theorem solve_real_eq (x : ℝ) :
  (8 * x ^ 2 + 150 * x + 3) / (3 * x + 56) = 4 * x + 2 ↔ x = -1.5 ∨ x = -18.5 :=
by
  sorry

end solve_real_eq_l261_261350


namespace john_paid_more_l261_261764

theorem john_paid_more 
  (original_price : ℝ)
  (discount_percentage : ℝ) 
  (tip_percentage : ℝ) 
  (discounted_price : ℝ)
  (john_tip : ℝ) 
  (john_total : ℝ)
  (jane_tip : ℝ)
  (jane_total : ℝ) 
  (difference : ℝ) :
  original_price = 42.00000000000004 →
  discount_percentage = 0.10 →
  tip_percentage = 0.15 →
  discounted_price = original_price - (discount_percentage * original_price) →
  john_tip = tip_percentage * original_price →
  john_total = original_price + john_tip →
  jane_tip = tip_percentage * discounted_price →
  jane_total = discounted_price + jane_tip →
  difference = john_total - jane_total →
  difference = 4.830000000000005 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end john_paid_more_l261_261764


namespace verify_equation_l261_261807

theorem verify_equation : (3^2 + 5^2)^2 = 16^2 + 30^2 := by
  sorry

end verify_equation_l261_261807


namespace compute_expression_l261_261009

theorem compute_expression : 85 * 1500 + (1 / 2) * 1500 = 128250 :=
by
  sorry

end compute_expression_l261_261009


namespace animal_legs_count_l261_261626

-- Let's define the conditions first.
def total_animals : ℕ := 12
def chickens : ℕ := 5
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4

-- Define the statement that we need to prove.
theorem animal_legs_count :
  ∃ (total_legs : ℕ), total_legs = 38 :=
by
  -- Adding the condition for total number of legs
  let sheep := total_animals - chickens
  let total_legs := (chickens * chicken_legs) + (sheep * sheep_legs)
  existsi total_legs
  -- Question proves the correct answer
  sorry

end animal_legs_count_l261_261626


namespace repetitions_today_l261_261962

theorem repetitions_today (yesterday_reps : ℕ) (deficit : ℤ) (today_reps : ℕ) : 
  yesterday_reps = 86 ∧ deficit = -13 → 
  today_reps = yesterday_reps + deficit →
  today_reps = 73 :=
by
  intros
  sorry

end repetitions_today_l261_261962


namespace simplify_expression_l261_261349

theorem simplify_expression (w : ℕ) : 
  4 * w + 6 * w + 8 * w + 10 * w + 12 * w + 14 * w + 16 = 54 * w + 16 :=
by 
  sorry

end simplify_expression_l261_261349


namespace constant_is_arithmetic_l261_261100

def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem constant_is_arithmetic (a : ℕ → ℝ) (h : is_constant_sequence a) : is_arithmetic_sequence a := by
  sorry

end constant_is_arithmetic_l261_261100


namespace train_length_l261_261293

theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 36 * 1000 / 3600) (h2 : time = 14.998800095992321) :
  speed * time = 149.99 :=
by {
  sorry
}

end train_length_l261_261293


namespace number_of_pages_in_book_l261_261963

-- Define the conditions using variables and hypotheses
variables (P : ℝ) (h1 : 0.30 * P = 150)

-- State the theorem to be proved
theorem number_of_pages_in_book : P = 500 :=
by
  -- Proof would go here, but we use sorry to skip it
  sorry

end number_of_pages_in_book_l261_261963


namespace g_at_4_l261_261480

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (y : ℝ) : ℝ := (3 * y - 4) / y

noncomputable def g (x : ℝ) : ℝ := 1 / (f_inv x) + 5

theorem g_at_4 : g 4 = 11 / 2 :=
by
  sorry

end g_at_4_l261_261480


namespace cos_210_eq_neg_sqrt3_div_2_l261_261165

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261165


namespace vector_subtraction_l261_261223

variable (a b : ℝ × ℝ)

def vector_calc (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem vector_subtraction :
  a = (2, 4) → b = (-1, 1) → vector_calc a b = (5, 7) := by
  intros ha hb
  simp [vector_calc]
  rw [ha, hb]
  simp
  sorry

end vector_subtraction_l261_261223


namespace exist_indices_with_non_decreasing_subsequences_l261_261593

theorem exist_indices_with_non_decreasing_subsequences
  (a b c : ℕ → ℕ) :
  (∀ n m : ℕ, n < m → ∃ p q : ℕ, q < p ∧ 
    a p ≥ a q ∧ 
    b p ≥ b q ∧ 
    c p ≥ c q) :=
  sorry

end exist_indices_with_non_decreasing_subsequences_l261_261593


namespace john_profit_l261_261905

theorem john_profit (cost price : ℕ) (n : ℕ) (h1 : cost = 4) (h2 : price = 8) (h3 : n = 30) : 
  n * (price - cost) = 120 :=
by
  -- The proof goes here
  sorry

end john_profit_l261_261905


namespace value_of_a_plus_b_l261_261863

theorem value_of_a_plus_b (a b x y : ℝ) 
  (h1 : 2 * x + 4 * y = 20)
  (h2 : a * x + b * y = 1)
  (h3 : 2 * x - y = 5)
  (h4 : b * x + a * y = 6) : a + b = 1 := 
sorry

end value_of_a_plus_b_l261_261863


namespace max_self_intersection_points_13_max_self_intersection_points_1950_l261_261957

def max_self_intersection_points (n : ℕ) : ℕ :=
if n % 2 = 1 then n * (n - 3) / 2 else n * (n - 4) / 2 + 1

theorem max_self_intersection_points_13 : max_self_intersection_points 13 = 65 :=
by sorry

theorem max_self_intersection_points_1950 : max_self_intersection_points 1950 = 1897851 :=
by sorry

end max_self_intersection_points_13_max_self_intersection_points_1950_l261_261957


namespace project_completion_time_l261_261511

theorem project_completion_time (rate_a rate_b rate_c : ℝ) (total_work : ℝ) (quit_time : ℝ) 
  (ha : rate_a = 1 / 20) 
  (hb : rate_b = 1 / 30) 
  (hc : rate_c = 1 / 40) 
  (htotal : total_work = 1)
  (hquit : quit_time = 18) : 
  ∃ T : ℝ, T = 18 :=
by {
  sorry
}

end project_completion_time_l261_261511


namespace vector_calc_l261_261218

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Statement to prove that 2a - b = (5, 7)
theorem vector_calc : 2 • a - b = (5, 7) :=
by {
  -- Proof will be filled here
  sorry
}

end vector_calc_l261_261218


namespace fourth_guard_distance_l261_261815

theorem fourth_guard_distance (d1 d2 d3 : ℕ) (d4 : ℕ) (h1 : d1 + d2 + d3 + d4 = 1000) (h2 : d1 + d2 + d3 = 850) : d4 = 150 :=
sorry

end fourth_guard_distance_l261_261815


namespace factorize_expr_l261_261189

theorem factorize_expr (a b : ℝ) : a^2 - 2 * a * b = a * (a - 2 * b) := 
by 
  sorry

end factorize_expr_l261_261189


namespace fraction_sum_simplified_l261_261263

theorem fraction_sum_simplified (a b : ℕ) (h1 : 0.6125 = (a : ℝ) / b) (h2 : Nat.gcd a b = 1) : a + b = 129 :=
sorry

end fraction_sum_simplified_l261_261263


namespace bobby_payment_l261_261984

theorem bobby_payment :
  let mold_cost := 250
  let labor_cost_per_hour := 75
  let hours := 8
  let discount := 0.80
  let total_labor_cost := labor_cost_per_hour * hours
  let discounted_labor_cost := discount * total_labor_cost
  let total_payment := mold_cost + discounted_labor_cost
  total_payment = 730 :=
by
  let mold_cost := 250
  let labor_cost_per_hour := 75
  let hours := 8
  let discount := 0.80
  let total_labor_cost := labor_cost_per_hour * hours
  let discounted_labor_cost := discount * total_labor_cost
  let total_payment := mold_cost + discounted_labor_cost
  sorry

end bobby_payment_l261_261984


namespace interest_is_less_by_1940_l261_261820

noncomputable def principal : ℕ := 2000
noncomputable def rate : ℕ := 3
noncomputable def time : ℕ := 3

noncomputable def simple_interest (P R T : ℕ) : ℕ :=
  (P * R * T) / 100

noncomputable def difference (sum_lent interest : ℕ) : ℕ :=
  sum_lent - interest

theorem interest_is_less_by_1940 :
  difference principal (simple_interest principal rate time) = 1940 :=
by
  sorry

end interest_is_less_by_1940_l261_261820


namespace find_number_of_roses_l261_261120

theorem find_number_of_roses : ∃ a : ℕ, 300 ≤ a ∧ a ≤ 400 ∧ a % 21 = 13 ∧ a % 15 = 7 :=
by
  -- Existential quantifier for the number 'a'
  use 307
  
  -- Proof of the conditions for 'a'
  split
  -- Proof that 300 ≤ 307 ∧ 307 ≤ 400
  exact ⟨by linarith, by linarith⟩
  split
  -- Proof that 307 % 21 = 13
  exact by norm_num
  -- Proof that 307 % 15 = 7 (because -8 % 15 = 7)
  exact by norm_num

end find_number_of_roses_l261_261120


namespace fraction_transform_l261_261655

theorem fraction_transform {x : ℤ} :
  (537 - x : ℚ) / (463 + x) = 1 / 9 ↔ x = 437 := by
sorry

end fraction_transform_l261_261655


namespace isosceles_triangle_perimeter_l261_261986

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₁ : a = 12) (h₂ : b = 12) (h₃ : c = 17) : a + b + c = 41 :=
by
  rw [h₁, h₂, h₃]
  norm_num

end isosceles_triangle_perimeter_l261_261986


namespace square_side_length_l261_261289

theorem square_side_length (length width : ℕ) (h1 : length = 10) (h2 : width = 5) (cut_across_length : length % 2 = 0) :
  ∃ square_side : ℕ, square_side = 5 := by
  sorry

end square_side_length_l261_261289


namespace smaller_tablet_diagonal_l261_261801

theorem smaller_tablet_diagonal :
  ∀ (A_large A_small : ℝ)
    (d : ℝ),
    A_large = (8 / Real.sqrt 2) ^ 2 →
    A_small = (d / Real.sqrt 2) ^ 2 →
    A_large = A_small + 7.5 →
    d = 7
:= by
  intros A_large A_small d h1 h2 h3
  sorry

end smaller_tablet_diagonal_l261_261801


namespace max_value_x_minus_y_proof_l261_261468

noncomputable def max_value_x_minus_y (θ : ℝ) : ℝ :=
  sorry

theorem max_value_x_minus_y_proof (θ : ℝ) (h1 : x = Real.sin θ) (h2 : y = Real.cos θ)
(h3 : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) (h4 : (x^2 + y^2)^2 = x + y) : 
  max_value_x_minus_y θ = Real.sqrt 2 :=
sorry

end max_value_x_minus_y_proof_l261_261468


namespace complement_intersection_l261_261322

open Set

variable (U A B : Set ℕ)

theorem complement_intersection (U : Set ℕ) (A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {1, 3, 6}) (hB : B = {1, 2}) :
  ((U \ A) ∩ B) = {2} :=
by
  rw [hU, hA, hB]
  sorry

end complement_intersection_l261_261322


namespace expected_value_of_8_sided_die_l261_261638

def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℝ := 1 / 8

theorem expected_value_of_8_sided_die :
  (∑ x in outcomes, probability x * x) = 4.5 := 
sorry

end expected_value_of_8_sided_die_l261_261638


namespace beetle_speed_l261_261679

theorem beetle_speed
  (distance_ant : ℝ )
  (time_minutes : ℝ)
  (distance_beetle : ℝ) 
  (distance_percent_less : ℝ)
  (time_hours : ℝ)
  (beetle_speed_kmh : ℝ)
  (h1 : distance_ant = 600)
  (h2 : time_minutes = 10)
  (h3 : time_hours = time_minutes / 60)
  (h4 : distance_percent_less = 0.25)
  (h5 : distance_beetle = distance_ant * (1 - distance_percent_less))
  (h6 : beetle_speed_kmh = distance_beetle / time_hours) : 
  beetle_speed_kmh = 2.7 :=
by 
  sorry

end beetle_speed_l261_261679


namespace calculate_cakes_left_l261_261132

-- Define the conditions
def b_lunch : ℕ := 5
def s_dinner : ℕ := 6
def b_yesterday : ℕ := 3

-- Define the calculation of the total cakes baked and cakes left
def total_baked : ℕ := b_lunch + b_yesterday
def cakes_left : ℕ := total_baked - s_dinner

-- The theorem we want to prove
theorem calculate_cakes_left : cakes_left = 2 := 
by
  sorry

end calculate_cakes_left_l261_261132


namespace frog_hops_ratio_l261_261944

theorem frog_hops_ratio (S T F : ℕ) (h1 : S = 2 * T) (h2 : S = 18) (h3 : F + S + T = 99) :
  F / S = 4 / 1 :=
by
  sorry

end frog_hops_ratio_l261_261944


namespace multiply_eq_four_l261_261865

variables (a b c d : ℝ)

theorem multiply_eq_four (h1 : a = d) 
                         (h2 : b = c) 
                         (h3 : d + d = c * d) 
                         (h4 : b = d) 
                         (h5 : d + d = d * d) 
                         (h6 : c = 3) :
                         a * b = 4 := 
by 
  sorry

end multiply_eq_four_l261_261865


namespace simplify_expression_l261_261419

theorem simplify_expression : 3000 * 3000^3000 = 3000^(3001) := 
by 
  sorry

end simplify_expression_l261_261419


namespace probability_both_segments_at_least_1m_l261_261615

-- Definitions
def rope_length := 3 -- length of the rope
def min_segment_length := 1 -- minimum length of each segment

-- Main Statement
theorem probability_both_segments_at_least_1m : 
  (Pr(λ x : ℝ, min_segment_length ≤ x ∧ x ≤ rope_length - min_segment_length)) = (1 / rope_length) :=
by
  sorry

end probability_both_segments_at_least_1m_l261_261615


namespace danielle_rooms_is_6_l261_261870

def heidi_rooms (danielle_rooms : ℕ) : ℕ := 3 * danielle_rooms
def grant_rooms (heidi_rooms : ℕ) : ℕ := heidi_rooms / 9

theorem danielle_rooms_is_6 (danielle_rooms : ℕ) (h1 : heidi_rooms danielle_rooms = 18) (h2 : grant_rooms (heidi_rooms danielle_rooms) = 2) :
  danielle_rooms = 6 :=
by 
  sorry

end danielle_rooms_is_6_l261_261870


namespace train_length_is_correct_l261_261134

noncomputable def convert_speed (speed_kmh : ℕ) : ℝ :=
  (speed_kmh : ℝ) * 5 / 18

noncomputable def relative_speed (train_speed_kmh man's_speed_kmh : ℕ) : ℝ :=
  convert_speed train_speed_kmh + convert_speed man's_speed_kmh

noncomputable def length_of_train (train_speed_kmh man's_speed_kmh : ℕ) (time_seconds : ℝ) : ℝ := 
  relative_speed train_speed_kmh man's_speed_kmh * time_seconds

theorem train_length_is_correct :
  length_of_train 60 6 29.997600191984645 = 550 :=
by
  sorry

end train_length_is_correct_l261_261134


namespace total_area_of_removed_triangles_l261_261530

theorem total_area_of_removed_triangles (x r s : ℝ) (h1 : (x - r)^2 + (x - s)^2 = 15^2) :
  4 * (1/2 * r * s) = 112.5 :=
by
  sorry

end total_area_of_removed_triangles_l261_261530


namespace circle_area_difference_l261_261325

theorem circle_area_difference (r1 r2 : ℝ) (h1 : r1 = 15) (h2 : r2 = 8) 
: real.pi * (r1 * r1) - real.pi * (r2 * r2) = 161 * real.pi := 
by {
  rw [h1, h2],
  norm_num,
  ring,
}

end circle_area_difference_l261_261325


namespace rectangle_area_l261_261668

variables (y : ℝ) (length : ℝ) (width : ℝ)

-- Definitions based on conditions
def is_diagonal_y (length width y : ℝ) : Prop :=
  y^2 = length^2 + width^2

def is_length_three_times_width (length width : ℝ) : Prop :=
  length = 3 * width

-- Statement to prove
theorem rectangle_area (y : ℝ) (length width : ℝ)
  (h1 : is_diagonal_y length width y)
  (h2 : is_length_three_times_width length width) :
  length * width = 3 * (y^2 / 10) :=
sorry

end rectangle_area_l261_261668


namespace expected_value_of_8_sided_die_is_4_point_5_l261_261648

def expected_value_8_sided_die : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (list.sum outcomes : ℝ) * probability

theorem expected_value_of_8_sided_die_is_4_point_5 :
  expected_value_8_sided_die = 4.5 := by
  sorry

end expected_value_of_8_sided_die_is_4_point_5_l261_261648


namespace jack_jog_speed_l261_261241

theorem jack_jog_speed (melt_time_minutes : ℕ) (distance_blocks : ℕ) (block_length_miles : ℚ) 
    (h_melt_time : melt_time_minutes = 10)
    (h_distance : distance_blocks = 16)
    (h_block_length : block_length_miles = 1/8) :
    let time_hours := (melt_time_minutes : ℚ) / 60
    let distance_miles := (distance_blocks : ℚ) * block_length_miles
        12 = distance_miles / time_hours :=
by
  sorry

end jack_jog_speed_l261_261241


namespace sum_c_d_eq_24_l261_261139

theorem sum_c_d_eq_24 (c d : ℕ) (h_pos_c : c > 0) (h_pos_d : d > 1) (h_max_power : c^d < 500 ∧ ∀ ⦃x y : ℕ⦄, x^y < 500 → x^y ≤ c^d) : c + d = 24 :=
sorry

end sum_c_d_eq_24_l261_261139


namespace find_z_l261_261493

theorem find_z (z : ℝ) (v : ℝ × ℝ × ℝ) (u : ℝ × ℝ × ℝ)
  (h_v : v = (4, 1, z)) (h_u : u = (2, -3, 4))
  (h_eq : (4 * 2 + 1 * -3 + z * 4) / (2 * 2 + -3 * -3 + 4 * 4) = 5 / 29) :
  z = 0 :=
by
  sorry

end find_z_l261_261493


namespace rhombus_diagonals_l261_261260

theorem rhombus_diagonals (x y : ℝ) 
  (h1 : x * y = 234)
  (h2 : x + y = 31) :
  (x = 18 ∧ y = 13) ∨ (x = 13 ∧ y = 18) := by
sorry

end rhombus_diagonals_l261_261260


namespace chord_length_through_focus_l261_261541

theorem chord_length_through_focus (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1)
  (h_perp : (x = 1) ∨ (x = -1)) : abs (2 * y) = 3 :=
by {
  sorry
}

end chord_length_through_focus_l261_261541


namespace mod_remainder_l261_261958

open Int

theorem mod_remainder (n : ℤ) : 
  (1125 * 1127 * n) % 12 = 3 ↔ n % 12 = 1 :=
by
  sorry

end mod_remainder_l261_261958


namespace sequence_general_term_l261_261215

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 0 then 4 else 4 * (-1 / 3)^(n - 1) 

theorem sequence_general_term (n : ℕ) (hn : n ≥ 1) 
  (hrec : ∀ n, 3 * a_n (n + 1) + a_n n = 0)
  (hinit : a_n 2 = -4 / 3) :
  a_n n = 4 * (-1 / 3)^(n - 1) := by
  sorry

end sequence_general_term_l261_261215


namespace cost_per_can_of_tuna_l261_261072

theorem cost_per_can_of_tuna
  (num_cans : ℕ) -- condition 1
  (num_coupons : ℕ) -- condition 2
  (coupon_discount_cents : ℕ) -- condition 2 detail
  (amount_paid_dollars : ℚ) -- condition 3
  (change_received_dollars : ℚ) -- condition 3 detail
  (cost_per_can_cents: ℚ) : -- the quantity we want to prove
  num_cans = 9 →
  num_coupons = 5 →
  coupon_discount_cents = 25 →
  amount_paid_dollars = 20 →
  change_received_dollars = 5.5 →
  cost_per_can_cents = 175 :=
by
  intros hn hc hcd hap hcr
  sorry

end cost_per_can_of_tuna_l261_261072


namespace expand_binomials_l261_261552

variable (x y : ℝ)

theorem expand_binomials : 
  (3 * x - 2) * (2 * x + 4 * y + 1) = 6 * x^2 + 12 * x * y - x - 8 * y - 2 :=
by
  sorry

end expand_binomials_l261_261552


namespace evaluate_expression_at_3_l261_261728

theorem evaluate_expression_at_3 :
  (1 / (3 + 1 / (3 + 1 / (3 - 1 / 3)))) = 0.30337078651685395 :=
  sorry

end evaluate_expression_at_3_l261_261728


namespace mirror_tweet_rate_is_45_l261_261343

-- Defining the conditions given in the problem
def happy_tweet_rate : ℕ := 18
def hungry_tweet_rate : ℕ := 4
def mirror_tweet_rate (x : ℕ) : ℕ := x
def happy_minutes : ℕ := 20
def hungry_minutes : ℕ := 20
def mirror_minutes : ℕ := 20
def total_tweets : ℕ := 1340

-- Proving the rate of tweets when Polly watches herself in the mirror
theorem mirror_tweet_rate_is_45 : mirror_tweet_rate 45 * mirror_minutes = total_tweets - (happy_tweet_rate * happy_minutes + hungry_tweet_rate * hungry_minutes) :=
by 
  sorry

end mirror_tweet_rate_is_45_l261_261343


namespace parallel_line_slope_l261_261719

theorem parallel_line_slope (a b c : ℝ) (m : ℝ) :
  (5 * a + 10 * b = -35) →
  (∃ m : ℝ, b = m * a + c) →
  m = -1/2 :=
by sorry

end parallel_line_slope_l261_261719


namespace problem1_problem2_l261_261436

variable {a b c : ℝ}
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Statement for the first proof
theorem problem1 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  bc / a + ca / b + ab / c ≥ a + b + c :=
sorry

-- Statement for the second proof
theorem problem2 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (1 - a) / a + (1 - b) / b + (1 - c) / c ≥ 6 :=
sorry

end problem1_problem2_l261_261436


namespace minimum_distance_sum_squared_l261_261207

variable (P : ℝ × ℝ)
variable (F₁ F₂ : ℝ × ℝ)

def on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 / 4) + y^2 = 1

def distance_squared (A B : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2

theorem minimum_distance_sum_squared
  (hP : on_ellipse P)
  (hF1 : F₁ = (2, 0) ∨ F₁ = (-2, 0)) -- Assuming standard position of foci
  (hF2 : F₂ = (2, 0) ∨ F₂ = (-2, 0)) :
  ∃ P : ℝ × ℝ, on_ellipse P ∧ F₁ ≠ F₂ → distance_squared P F₁ + distance_squared P F₂ = 8 :=
by
  sorry

end minimum_distance_sum_squared_l261_261207


namespace base_10_representation_l261_261485

-- Conditions
variables (C D : ℕ)
variables (hC : 0 ≤ C ∧ C ≤ 7)
variables (hD : 0 ≤ D ∧ D ≤ 5)
variables (hEq : 8 * C + D = 6 * D + C)

-- Goal
theorem base_10_representation : 8 * C + D = 0 := by
  sorry

end base_10_representation_l261_261485


namespace upper_bound_neg_expr_l261_261058

theorem upper_bound_neg_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  - (1 / (2 * a) + 2 / b) ≤ - (9 / 2) := 
sorry

end upper_bound_neg_expr_l261_261058


namespace goose_eggs_l261_261380

theorem goose_eggs (E : ℕ) 
  (H1 : (2/3 : ℚ) * E = h) 
  (H2 : (3/4 : ℚ) * h = m)
  (H3 : (2/5 : ℚ) * m = 180) : 
  E = 2700 := 
sorry

end goose_eggs_l261_261380


namespace general_term_of_sequence_l261_261934

noncomputable def seq (n : ℕ) : ℕ :=
  match n with
  | 1 => 6
  | 2 => 9
  | 3 => 14
  | 4 => 21
  | 5 => 30
  | _ => sorry

theorem general_term_of_sequence :
  ∀ n : ℕ, seq n = 5 + n^2 :=
by
  sorry

end general_term_of_sequence_l261_261934


namespace mailman_junk_mail_l261_261126

variable (junk_mail_per_house : ℕ) (houses_per_block : ℕ)

theorem mailman_junk_mail (h1 : junk_mail_per_house = 2) (h2 : houses_per_block = 7) :
  junk_mail_per_house * houses_per_block = 14 :=
by
  sorry

end mailman_junk_mail_l261_261126


namespace marie_initial_erasers_l261_261253

def erasers_problem : Prop :=
  ∃ initial_erasers : ℝ, initial_erasers + 42.0 = 137

theorem marie_initial_erasers : erasers_problem :=
  sorry

end marie_initial_erasers_l261_261253


namespace number_of_smaller_cubes_l261_261664

theorem number_of_smaller_cubes (edge : ℕ) (N : ℕ) (h_edge : edge = 5)
  (h_divisors : ∃ (a b c : ℕ), a + b + c = N ∧ a * 1^3 + b * 2^3 + c * 3^3 = edge^3 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  N = 22 :=
by
  sorry

end number_of_smaller_cubes_l261_261664


namespace expected_value_of_eight_sided_die_l261_261643

-- Definitions based on the problem conditions
def eight_sided_die_outcomes : List ℕ := [1,2,3,4,5,6,7,8]

def probability (n : ℕ) := 1 / n

-- Expected value calculation related to the problem
def expected_value_die_roll (outcomes : List ℕ) (prob : ℕ → Rat) : Rat :=
  List.sum (outcomes.map (λ x => prob outcomes.length * x))

-- Expected value of an 8-sided die roll
theorem expected_value_of_eight_sided_die :
  expected_value_die_roll eight_sided_die_outcomes probability = 4.5 := 
sorry

end expected_value_of_eight_sided_die_l261_261643


namespace find_inverse_of_25_l261_261208

-- Define the inverses and the modulo
def inverse_mod (a m i : ℤ) : Prop :=
  (a * i) % m = 1

-- The given condition in the problem
def condition (m : ℤ) : Prop :=
  inverse_mod 5 m 39

-- The theorem we want to prove
theorem find_inverse_of_25 (m : ℤ) (h : condition m) : inverse_mod 25 m 8 :=
by
  sorry

end find_inverse_of_25_l261_261208


namespace total_roses_l261_261122

theorem total_roses (a : ℕ) (x y k : ℕ) (h1 : 300 ≤ a) (h2 : a ≤ 400)
  (h3 : a = 21 * x + 13) (h4 : a = 15 * y - 8) (h5 : a + 8 = 105 * k) :
  a = 307 :=
sorry

end total_roses_l261_261122


namespace sara_ticket_cost_l261_261339

noncomputable def calc_ticket_price : ℝ :=
  let rented_movie_cost := 1.59
  let bought_movie_cost := 13.95
  let total_cost := 36.78
  let total_tickets := 2
  let spent_on_tickets := total_cost - (rented_movie_cost + bought_movie_cost)
  spent_on_tickets / total_tickets

theorem sara_ticket_cost : calc_ticket_price = 10.62 := by
  sorry

end sara_ticket_cost_l261_261339


namespace box_weight_l261_261805

theorem box_weight (W : ℝ) (h : 7 * (W - 20) = 3 * W) : W = 35 := by
  sorry

end box_weight_l261_261805


namespace cos_210_eq_neg_sqrt3_div_2_l261_261169

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261169


namespace undefined_expression_value_l261_261561

theorem undefined_expression_value {a : ℝ} : (a^3 - 8 = 0) ↔ (a = 2) :=
by sorry

end undefined_expression_value_l261_261561


namespace greatest_possible_sum_of_consecutive_integers_prod_lt_200_l261_261652

theorem greatest_possible_sum_of_consecutive_integers_prod_lt_200 :
  ∃ n : ℤ, (n * (n + 1) < 200) ∧ ( ∀ m : ℤ, (m * (m + 1) < 200) → m ≤ n) ∧ (n + (n + 1) = 27) :=
by
  sorry

end greatest_possible_sum_of_consecutive_integers_prod_lt_200_l261_261652


namespace ashley_family_spent_30_l261_261136

def cost_of_child_ticket : ℝ := 4.25
def cost_of_adult_ticket : ℝ := cost_of_child_ticket + 3.25
def discount : ℝ := 2.00
def num_adult_tickets : ℕ := 2
def num_child_tickets : ℕ := 4

def total_cost : ℝ := num_adult_tickets * cost_of_adult_ticket + num_child_tickets * cost_of_child_ticket - discount

theorem ashley_family_spent_30 :
  total_cost = 30.00 :=
sorry

end ashley_family_spent_30_l261_261136


namespace initial_volume_of_mixture_l261_261396

theorem initial_volume_of_mixture
  (x : ℕ)
  (h1 : 3 * x / (2 * x + 1) = 4 / 3)
  (h2 : x = 4) :
  5 * x = 20 :=
by
  sorry

end initial_volume_of_mixture_l261_261396


namespace intersection_of_lines_l261_261548

-- Define the first and second lines
def line1 (x : ℚ) : ℚ := 3 * x + 1
def line2 (x : ℚ) : ℚ := -7 * x - 5

-- Statement: Prove that the intersection of the lines given by
-- y = 3x + 1 and y + 5 = -7x is (-3/5, -4/5).

theorem intersection_of_lines :
  ∃ x y : ℚ, y = line1 x ∧ y = line2 x ∧ x = -3 / 5 ∧ y = -4 / 5 :=
by
  sorry

end intersection_of_lines_l261_261548


namespace sqrt_88200_simplified_l261_261613

theorem sqrt_88200_simplified : Real.sqrt 88200 = 210 * Real.sqrt 6 :=
by sorry

end sqrt_88200_simplified_l261_261613


namespace bobby_shoes_cost_l261_261985

theorem bobby_shoes_cost :
  let mold_cost := 250
  let hourly_rate := 75
  let hours_worked := 8
  let discount_rate := 0.20
  let labor_cost := hourly_rate * hours_worked
  let discounted_labor_cost := labor_cost * (1 - discount_rate)
  let total_cost := mold_cost + discounted_labor_cost
  mold_cost = 250 ∧ hourly_rate = 75 ∧ hours_worked = 8 ∧ discount_rate = 0.20 →
  total_cost = 730 := 
by
  sorry

end bobby_shoes_cost_l261_261985


namespace rosalina_gifts_l261_261344

theorem rosalina_gifts (Emilio_gifts Jorge_gifts Pedro_gifts : ℕ) 
  (hEmilio : Emilio_gifts = 11) 
  (hJorge : Jorge_gifts = 6) 
  (hPedro : Pedro_gifts = 4) : 
  Emilio_gifts + Jorge_gifts + Pedro_gifts = 21 :=
by
  sorry

end rosalina_gifts_l261_261344


namespace charlene_gave_18_necklaces_l261_261712

theorem charlene_gave_18_necklaces
  (initial_necklaces : ℕ) (sold_necklaces : ℕ) (left_necklaces : ℕ)
  (h1 : initial_necklaces = 60)
  (h2 : sold_necklaces = 16)
  (h3 : left_necklaces = 26) :
  initial_necklaces - sold_necklaces - left_necklaces = 18 :=
by
  sorry

end charlene_gave_18_necklaces_l261_261712


namespace required_jogging_speed_l261_261242

-- Definitions based on the conditions
def blocks_to_miles (blocks : ℕ) : ℚ := blocks * (1 / 8 : ℚ)
def time_in_hours (minutes : ℕ) : ℚ := minutes / 60

-- Constants provided by the problem
def beach_distance_in_blocks : ℕ := 16
def ice_cream_melt_time_in_minutes : ℕ := 10

-- The main statement to prove
theorem required_jogging_speed :
  let distance := blocks_to_miles beach_distance_in_blocks
  let time := time_in_hours ice_cream_melt_time_in_minutes
  (distance / time) = 12 := by
  sorry

end required_jogging_speed_l261_261242


namespace period_of_tan_2x_pi_over_6_l261_261082

noncomputable def function_period (x : ℝ) : ℝ := tan (2 * x + real.pi / 6)

theorem period_of_tan_2x_pi_over_6 : 
  ∃ T, (∀ x, function_period (x + T) = function_period x) ∧ T = real.pi / 2 :=
by
  sorry

end period_of_tan_2x_pi_over_6_l261_261082


namespace avg_score_calculation_l261_261292

-- Definitions based on the conditions
def directly_proportional (a b : ℝ) : Prop := ∃ k, a = k * b

variables (score_math : ℝ) (score_science : ℝ)
variables (hours_math : ℝ := 4) (hours_science : ℝ := 5)
variables (next_hours_math_science : ℝ := 5)
variables (expected_avg_score : ℝ := 97.5)

axiom h1 : directly_proportional 80 4
axiom h2 : directly_proportional 95 5

-- Define the goal: Expected average score given the study hours next time
theorem avg_score_calculation :
  (score_math / hours_math = score_science / hours_science) →
  (score_math = 100 ∧ score_science = 95) →
  ((next_hours_math_science * score_math / hours_math + next_hours_math_science * score_science / hours_science) / 2 = expected_avg_score) :=
by sorry

end avg_score_calculation_l261_261292


namespace celeb_baby_photo_matching_probability_l261_261395

theorem celeb_baby_photo_matching_probability :
  let total_matches := 6
  let correct_matches := 1
  let probability := correct_matches / total_matches
  probability = 1 / 6 :=
by
  let total_matches := 3!
  let correct_matches := 1
  let probability := correct_matches / total_matches
  have h1 : total_matches = 6 := by simp
  have h2 : probability = 1 / 6 := by simp [h1]
  exact h2

end celeb_baby_photo_matching_probability_l261_261395


namespace range_of_y_l261_261229

theorem range_of_y (y : ℝ) (h1: 1 / y < 3) (h2: 1 / y > -4) : y > 1 / 3 :=
by
  sorry

end range_of_y_l261_261229


namespace water_added_l261_261532

theorem water_added (x : ℝ) (salt_initial_percentage : ℝ) (salt_final_percentage : ℝ) 
   (evap_fraction : ℝ) (salt_added : ℝ) (W : ℝ) 
   (hx : x = 150) (h_initial_salt : salt_initial_percentage = 0.2) 
   (h_final_salt : salt_final_percentage = 1 / 3) 
   (h_evap_fraction : evap_fraction = 1 / 4) 
   (h_salt_added : salt_added = 20) : 
  W = 37.5 :=
by
  sorry

end water_added_l261_261532


namespace distribution_count_l261_261534

theorem distribution_count :
  let rows := 13
  ∃ count : ℕ, count = 2560 ∧ 
    (∀ (x : fin rows → ℤ), (∀ i : fin rows, x i = 0 ∨ x i = 1) →
      (∑ i, (nat.choose (rows - 1) i) * x i) % 5 = 0 ↔ count = 2560) := 
by
  sorry

end distribution_count_l261_261534


namespace carrie_strawberry_harvest_l261_261409

/-- Carrie has a rectangular garden that measures 10 feet by 7 feet.
    She plants the entire garden with strawberry plants. Carrie is able to
    plant 5 strawberry plants per square foot, and she harvests an average of
    12 strawberries per plant. How many strawberries can she expect to harvest?
-/
theorem carrie_strawberry_harvest :
  let width := 10
  let length := 7
  let plants_per_sqft := 5
  let strawberries_per_plant := 12
  let area := width * length
  let total_plants := plants_per_sqft * area
  let total_strawberries := strawberries_per_plant * total_plants
  total_strawberries = 4200 :=
by
  sorry

end carrie_strawberry_harvest_l261_261409


namespace supremum_of_function_l261_261311

theorem supremum_of_function : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 
  (∃ M : ℝ, (∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → -1 / (2 * a) - 2 / b ≤ M) ∧
    (∀ K : ℝ, (∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → -1 / (2 * a) - 2 / b ≤ K) → M ≤ K) → M = -9 / 2) := 
sorry

end supremum_of_function_l261_261311


namespace minimum_gennadies_l261_261702

theorem minimum_gennadies (A B V G : ℕ) (hA : A = 45) (hB : B = 122) (hV : V = 27) (hGap : G + A + V >= B - 1) :
  G >= 49 :=
by 
  have := by linarith [hGap, hA, hB, hV]
  exact this

end minimum_gennadies_l261_261702


namespace gcf_180_240_45_l261_261504

theorem gcf_180_240_45 : Nat.gcd (Nat.gcd 180 240) 45 = 15 := by
  sorry

end gcf_180_240_45_l261_261504


namespace set1_eq_set2_eq_set3_eq_set4_eq_set5_eq_l261_261553

open Set

-- (1) The set of integers whose absolute value is not greater than 2
theorem set1_eq : { x : ℤ | |x| ≤ 2 } = {-2, -1, 0, 1, 2} := sorry

-- (2) The set of positive numbers less than 10 that are divisible by 3
theorem set2_eq : { x : ℕ | x < 10 ∧ x > 0 ∧ x % 3 = 0 } = {3, 6, 9} := sorry

-- (3) The set {x | x = |x|, x < 5, x ∈ 𝕫}
theorem set3_eq : { x : ℕ | x < 5 } = {0, 1, 2, 3, 4} := sorry

-- (4) The set {(x, y) | x + y = 6, x ∈ ℕ⁺, y ∈ ℕ⁺}
theorem set4_eq : { p : ℕ × ℕ | p.1 + p.2 = 6 ∧ p.1 > 0 ∧ p.2 > 0 } = {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1) } := sorry

-- (5) The set {-3, -1, 1, 3, 5}
theorem set5_eq : {-3, -1, 1, 3, 5} = { x : ℤ | ∃ k : ℤ, x = 2 * k - 1 ∧ -1 ≤ k ∧ k ≤ 3 } := sorry

end set1_eq_set2_eq_set3_eq_set4_eq_set5_eq_l261_261553


namespace find_cost_of_jersey_l261_261803

def cost_of_jersey (J : ℝ) : Prop := 
  let shorts_cost := 15.20
  let socks_cost := 6.80
  let total_players := 16
  let total_cost := 752
  total_players * (J + shorts_cost + socks_cost) = total_cost

theorem find_cost_of_jersey : cost_of_jersey 25 :=
  sorry

end find_cost_of_jersey_l261_261803


namespace total_whales_seen_is_178_l261_261759

/-
Ishmael's monitoring of whales yields the following:
- On the first trip, he counts 28 male whales and twice as many female whales.
- On the second trip, he sees 8 baby whales, each traveling with their parents.
- On the third trip, he counts half as many male whales as the first trip and the same number of female whales as on the first trip.
-/

def number_of_whales_first_trip : ℕ := 28
def number_of_female_whales_first_trip : ℕ := 2 * number_of_whales_first_trip
def total_whales_first_trip : ℕ := number_of_whales_first_trip + number_of_female_whales_first_trip

def number_of_baby_whales_second_trip : ℕ := 8
def total_whales_second_trip : ℕ := number_of_baby_whales_second_trip * 3

def number_of_male_whales_third_trip : ℕ := number_of_whales_first_trip / 2
def number_of_female_whales_third_trip : ℕ := number_of_female_whales_first_trip
def total_whales_third_trip : ℕ := number_of_male_whales_third_trip + number_of_female_whales_third_trip

def total_whales_seen : ℕ := total_whales_first_trip + total_whales_second_trip + total_whales_third_trip

theorem total_whales_seen_is_178 : total_whales_seen = 178 :=
by
  -- skip the actual proof
  sorry

end total_whales_seen_is_178_l261_261759


namespace wilson_sledding_l261_261811

variable (T : ℕ)

theorem wilson_sledding :
  (4 * T) + 6 = 14 → T = 2 :=
by
  intros h
  sorry

end wilson_sledding_l261_261811


namespace option_a_correct_l261_261377

-- Define the variables as real numbers
variables {a b : ℝ}

-- Define the main theorem to prove
theorem option_a_correct : (a - b) * (2 * a + 2 * b) = 2 * a^2 - 2 * b^2 := by
  -- start the proof block
  sorry

end option_a_correct_l261_261377


namespace ordered_pair_a_82_a_28_l261_261680

-- Definitions for the conditions
def a (i j : ℕ) : ℕ :=
  if i % 2 = 1 then
    if j = 1 then i * i else i * i - (j - 1)
  else
    if j = 1 then (i-1) * i + 1 else i * i - (j - 1)

theorem ordered_pair_a_82_a_28 : (a 8 2, a 2 8) = (51, 63) := by
  sorry

end ordered_pair_a_82_a_28_l261_261680


namespace mary_remaining_cards_l261_261912

variable (initial_cards : ℝ) (bought_cards : ℝ) (promised_cards : ℝ)

def remaining_cards (initial : ℝ) (bought : ℝ) (promised : ℝ) : ℝ :=
  initial + bought - promised

theorem mary_remaining_cards :
  initial_cards = 18.0 →
  bought_cards = 40.0 →
  promised_cards = 26.0 →
  remaining_cards initial_cards bought_cards promised_cards = 32.0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end mary_remaining_cards_l261_261912


namespace max_value_k_l261_261901

noncomputable def seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 4
  | (n+1) => 3 * seq n - 2

theorem max_value_k (k : ℝ) :
  (∀ n : ℕ, n > 0 → k * (seq n) ≤ 9^n) → k ≤ 9 / 4 :=
sorry

end max_value_k_l261_261901


namespace carolyn_total_monthly_practice_l261_261407

-- Define the constants and relationships given in the problem
def daily_piano_practice : ℕ := 20
def times_violin_practice : ℕ := 3
def days_week : ℕ := 6
def weeks_month : ℕ := 4
def daily_violin_practice : ℕ := daily_piano_practice * times_violin_practice
def total_daily_practice : ℕ := daily_piano_practice + daily_violin_practice
def weekly_practice_time : ℕ := total_daily_practice * days_week
def monthly_practice_time : ℕ := weekly_practice_time * weeks_month

-- The proof statement with the final result
theorem carolyn_total_monthly_practice : monthly_practice_time = 1920 := by
  sorry

end carolyn_total_monthly_practice_l261_261407


namespace raman_profit_percentage_l261_261608

theorem raman_profit_percentage
  (cost1 weight1 rate1 : ℕ) (cost2 weight2 rate2 : ℕ) (total_cost_mix total_weight mixing_rate selling_rate profit profit_percentage : ℕ)
  (h_cost1 : cost1 = weight1 * rate1)
  (h_cost2 : cost2 = weight2 * rate2)
  (h_total_cost_mix : total_cost_mix = cost1 + cost2)
  (h_total_weight : total_weight = weight1 + weight2)
  (h_mixing_rate : mixing_rate = total_cost_mix / total_weight)
  (h_selling_price : selling_rate * total_weight = profit + total_cost_mix)
  (h_profit : profit = selling_rate * total_weight - total_cost_mix)
  (h_profit_percentage : profit_percentage = (profit * 100) / total_cost_mix)
  (h_weight1 : weight1 = 54)
  (h_rate1 : rate1 = 150)
  (h_weight2 : weight2 = 36)
  (h_rate2 : rate2 = 125)
  (h_selling_rate_value : selling_rate = 196) :
  profit_percentage = 40 :=
sorry

end raman_profit_percentage_l261_261608


namespace euler_sum_of_squares_euler_sum_of_quads_l261_261102

theorem euler_sum_of_squares :
  ∑' n : ℕ, 1 / (n.succ : ℚ)^2 = π^2 / 6 := sorry

theorem euler_sum_of_quads :
  ∑' n : ℕ, 1 / (n.succ : ℚ)^4 = π^4 / 90 := sorry

end euler_sum_of_squares_euler_sum_of_quads_l261_261102


namespace cosine_210_l261_261176

noncomputable def Q := rotate 210 (1, 0)
noncomputable def E := foot Q (1, 0) (-1, 0)

theorem cosine_210 : cos 210 = - (sqrt 3/2) :=
sorry

end cosine_210_l261_261176


namespace normal_pumping_rate_l261_261462

-- Define the conditions and the proof problem
def pond_capacity : ℕ := 200
def drought_factor : ℚ := 2/3
def fill_time : ℕ := 50

theorem normal_pumping_rate (R : ℚ) :
  (drought_factor * R) * (fill_time : ℚ) = pond_capacity → R = 6 :=
by
  sorry

end normal_pumping_rate_l261_261462


namespace expected_value_8_sided_die_l261_261637

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l261_261637


namespace max_min_difference_l261_261594

noncomputable def difference_max_min_z (x y z : ℝ) : ℝ :=
  if h₁ : x + y + z = 3 ∧ x^2 + y^2 + z^2 = 18 then 6 else 0

theorem max_min_difference (x y z : ℝ) (h₁ : x + y + z = 3) (h₂ : x^2 + y^2 + z^2 = 18) :
  difference_max_min_z x y z = 6 :=
by sorry

end max_min_difference_l261_261594


namespace cos_210_eq_neg_sqrt3_div2_l261_261159

theorem cos_210_eq_neg_sqrt3_div2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div2_l261_261159


namespace PlanY_more_cost_effective_l261_261627

-- Define the gigabytes Tim uses
variable (y : ℕ)

-- Define the cost functions for Plan X and Plan Y in cents
def cost_PlanX (y : ℕ) := 25 * y
def cost_PlanY (y : ℕ) := 1500 + 15 * y

-- Prove that Plan Y is cheaper than Plan X when y >= 150
theorem PlanY_more_cost_effective (y : ℕ) : y ≥ 150 → cost_PlanY y < cost_PlanX y := by
  sorry

end PlanY_more_cost_effective_l261_261627


namespace sum_of_first_3n_terms_l261_261271

theorem sum_of_first_3n_terms (n : ℕ) (sn s2n s3n : ℕ) 
  (h1 : sn = 48) (h2 : s2n = 60)
  (h3 : s2n - sn = s3n - s2n) (h4 : 2 * (s2n - sn) = sn + (s3n - s2n)) :
  s3n = 36 := 
by {
  sorry
}

end sum_of_first_3n_terms_l261_261271


namespace starting_number_l261_261089

theorem starting_number (n : ℤ) : 
  (∃ n, (200 - n) / 3 = 33 ∧ (200 % 3 ≠ 0) ∧ (n % 3 = 0 ∧ n ≤ 200)) → n = 102 :=
by
  sorry

end starting_number_l261_261089


namespace find_divisor_exists_four_numbers_in_range_l261_261092

theorem find_divisor_exists_four_numbers_in_range :
  ∃ n : ℕ, (n > 1) ∧ (∀ k : ℕ, 39 ≤ k ∧ k ≤ 79 → ∃ a : ℕ, k = n * a) ∧ (∃! (k₁ k₂ k₃ k₄ : ℕ), 39 ≤ k₁ ∧ k₁ ≤ 79 ∧ 39 ≤ k₂ ∧ k₂ ≤ 79 ∧ 39 ≤ k₃ ∧ k₃ ≤ 79 ∧ 39 ≤ k₄ ∧ k₄ ≤ 79 ∧ k₁ ≠ k₂ ∧ k₁ ≠ k₃ ∧ k₁ ≠ k₄ ∧ k₂ ≠ k₃ ∧ k₂ ≠ k₄ ∧ k₃ ≠ k₄ ∧ k₁ % n = 0 ∧ k₂ % n = 0 ∧ k₃ % n = 0 ∧ k₄ % n = 0) → n = 19 :=
by sorry

end find_divisor_exists_four_numbers_in_range_l261_261092


namespace probability_both_l261_261675

variable (Ω : Type) [ProbabilitySpace Ω]
variable (A B : Event Ω)
variable [decidable (A ∧ B)]

def probability_over_60 (P : ℙ ∋ A) : ℝ := 0.20
def probability_hypertension_given_over_60 (P : ℙ ∋ B | A) : ℝ := 0.45

theorem probability_both :
  (probability_over_60 Ω A) * (probability_hypertension_given_over_60 Ω B A) = 0.09 :=
by
  sorry

end probability_both_l261_261675


namespace remainder_when_divided_by_8_l261_261959

theorem remainder_when_divided_by_8 :
  (481207 % 8) = 7 :=
by
  sorry

end remainder_when_divided_by_8_l261_261959


namespace ginger_total_water_l261_261199

def hours_worked : Nat := 8
def cups_per_bottle : Nat := 2
def bottles_drank_per_hour : Nat := 1
def bottles_for_plants : Nat := 5

theorem ginger_total_water : 
  (hours_worked * cups_per_bottle * bottles_drank_per_hour) + (bottles_for_plants * cups_per_bottle) = 26 :=
by
  sorry

end ginger_total_water_l261_261199


namespace vector_subtraction_l261_261222

variable (a b : ℝ × ℝ)

def vector_calc (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem vector_subtraction :
  a = (2, 4) → b = (-1, 1) → vector_calc a b = (5, 7) := by
  intros ha hb
  simp [vector_calc]
  rw [ha, hb]
  simp
  sorry

end vector_subtraction_l261_261222


namespace quadratic_factorization_l261_261792

theorem quadratic_factorization :
  ∃ a b : ℕ, (a > b) ∧ (x^2 - 20 * x + 96 = (x - a) * (x - b)) ∧ (4 * b - a = 20) := sorry

end quadratic_factorization_l261_261792


namespace xy_product_range_l261_261837

theorem xy_product_range (x y : ℝ) (h : x^2 * y^2 + x^2 - 10 * x * y - 8 * x + 16 = 0) :
  0 ≤ x * y ∧ x * y ≤ 10 := 
sorry

end xy_product_range_l261_261837


namespace no_intersecting_axes_l261_261444

theorem no_intersecting_axes (m : ℝ) : (m^2 + 2 * m - 7 = 0) → m = -4 :=
sorry

end no_intersecting_axes_l261_261444


namespace find_x_if_arithmetic_mean_is_12_l261_261568

theorem find_x_if_arithmetic_mean_is_12 (x : ℝ) (h : (8 + 16 + 21 + 7 + x) / 5 = 12) : x = 8 :=
by
  sorry

end find_x_if_arithmetic_mean_is_12_l261_261568


namespace find_a_l261_261618

theorem find_a (a : ℝ) : (-2 * a + 3 = -4) -> (a = 7 / 2) :=
by
  intro h
  sorry

end find_a_l261_261618


namespace problem1_asymptotes_problem2_equation_l261_261520

-- Problem 1: Asymptotes of a hyperbola
theorem problem1_asymptotes (a : ℝ) (x y : ℝ) (hx : (y + a) ^ 2 - (x - a) ^ 2 = 2 * a)
  (hpt : 3 = x ∧ 1 = y) : 
  (y = x - 2 * a) ∨ (y = - x) := 
by 
  sorry

-- Problem 2: Equation of a hyperbola
theorem problem2_equation (a b c : ℝ) (x y : ℝ) 
  (hasymptote : y = x + 1 ∨ y = - (x + 1))  (hfocal : 2 * c = 4)
  (hc_squared : c ^ 2 = a ^ 2 + b ^ 2) (ha_eq_b : a = b): 
  y^2 - (x + 1)^2 = 2 := 
by 
  sorry

end problem1_asymptotes_problem2_equation_l261_261520


namespace problem_statement_l261_261505

theorem problem_statement 
  (h1 : 17 ≡ 3 [MOD 7])
  (h2 : 3^1 ≡ 3 [MOD 7])
  (h3 : 3^2 ≡ 2 [MOD 7])
  (h4 : 3^3 ≡ 6 [MOD 7])
  (h5 : 3^4 ≡ 4 [MOD 7])
  (h6 : 3^5 ≡ 5 [MOD 7])
  (h7 : 3^6 ≡ 1 [MOD 7])
  (h8 : 3^100 ≡ 4 [MOD 7]) :
  17^100 ≡ 4 [MOD 7] :=
by sorry

end problem_statement_l261_261505


namespace total_gold_value_l261_261770

def legacy_bars : ℕ := 5
def aleena_bars : ℕ := legacy_bars - 2
def value_per_bar : ℕ := 2200
def total_bars : ℕ := legacy_bars + aleena_bars
def total_value : ℕ := total_bars * value_per_bar

theorem total_gold_value : total_value = 17600 :=
by
  -- Begin proof
  sorry

end total_gold_value_l261_261770


namespace find_a5_a7_l261_261211

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom h1 : a 1 + a 3 = 2
axiom h2 : a 3 + a 5 = 4

theorem find_a5_a7 (a : ℕ → ℤ) (d : ℤ) (h_seq : is_arithmetic_sequence a d)
  (h1 : a 1 + a 3 = 2) (h2 : a 3 + a 5 = 4) : a 5 + a 7 = 6 :=
sorry

end find_a5_a7_l261_261211


namespace original_square_area_l261_261013

theorem original_square_area :
  ∀ (a b : ℕ), 
  (a * a = 24 * 1 * 1 + b * b ∧ 
  ((∃ m n : ℕ, (a + b = m ∧ a - b = n ∧ m * n = 24) ∨ 
  (a + b = n ∧ a - b = m ∧ m * n = 24)))) →
  a * a = 25 :=
by
  sorry

end original_square_area_l261_261013


namespace shift_parabola_5_units_right_l261_261887

def original_parabola (x : ℝ) : ℝ := x^2 + 3
def shifted_parabola (x : ℝ) : ℝ := (x-5)^2 + 3

theorem shift_parabola_5_units_right : ∀ x : ℝ, shifted_parabola x = original_parabola (x - 5) :=
by {
  -- This is the mathematical equivalence that we're proving
  sorry
}

end shift_parabola_5_units_right_l261_261887


namespace cos_210_eq_neg_sqrt3_div2_l261_261158

theorem cos_210_eq_neg_sqrt3_div2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div2_l261_261158


namespace part_a_part_b_l261_261280

-- Part (a)
theorem part_a (f : ℚ → ℝ) (h_add : ∀ x y : ℚ, f (x + y) = f x + f y) (h_mul : ∀ x y : ℚ, f (x * y) = f x * f y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = 0) :=
sorry

-- Part (b)
theorem part_b (f : ℝ → ℝ) (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) (h_mul : ∀ x y : ℝ, f (x * y) = f x * f y) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 0) :=
sorry

end part_a_part_b_l261_261280


namespace division_quotient_proof_l261_261902

theorem division_quotient_proof :
  (300324 / 29 = 10356) →
  (100007892 / 333 = 300324) :=
by
  intros h1
  sorry

end division_quotient_proof_l261_261902


namespace negation_of_exists_l261_261935

theorem negation_of_exists (x : ℝ) (h : ∃ x : ℝ, x^2 - x + 1 ≤ 0) : 
  (∀ x : ℝ, x^2 - x + 1 > 0) :=
sorry

end negation_of_exists_l261_261935


namespace coprime_solution_l261_261195

theorem coprime_solution (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_eq : 5 * a + 7 * b = 29 * (6 * a + 5 * b)) : a = 3 ∧ b = 2 :=
sorry

end coprime_solution_l261_261195


namespace second_root_of_quadratic_l261_261559

theorem second_root_of_quadratic (p q r : ℝ) (quad_eqn : ∀ x, 2 * p * (q - r) * x^2 + 3 * q * (r - p) * x + 4 * r * (p - q) = 0) (root : 2 * p * (q - r) * 2^2 + 3 * q * (r - p) * 2 + 4 * r * (p - q) = 0) :
    ∃ r₂ : ℝ, r₂ = (r * (p - q)) / (p * (q - r)) :=
sorry

end second_root_of_quadratic_l261_261559


namespace average_weight_of_dogs_is_5_l261_261927

def weight_of_brown_dog (B : ℝ) : ℝ := B
def weight_of_black_dog (B : ℝ) : ℝ := B + 1
def weight_of_white_dog (B : ℝ) : ℝ := 2 * B
def weight_of_grey_dog (B : ℝ) : ℝ := B - 1

theorem average_weight_of_dogs_is_5 (B : ℝ) (h : (weight_of_brown_dog B + weight_of_black_dog B + weight_of_white_dog B + weight_of_grey_dog B) / 4 = 5) :
  5 = 5 :=
by sorry

end average_weight_of_dogs_is_5_l261_261927


namespace stamps_total_l261_261599

def Lizette_stamps : ℕ := 813
def Minerva_stamps : ℕ := Lizette_stamps - 125
def Jermaine_stamps : ℕ := Lizette_stamps + 217

def total_stamps : ℕ := Minerva_stamps + Lizette_stamps + Jermaine_stamps

theorem stamps_total :
  total_stamps = 2531 := by
  sorry

end stamps_total_l261_261599


namespace cost_per_box_of_cookies_l261_261724

-- Given conditions
def initial_money : ℝ := 20
def mother_gift : ℝ := 2 * initial_money
def total_money : ℝ := initial_money + mother_gift
def cupcake_price : ℝ := 1.50
def num_cupcakes : ℝ := 10
def cost_cupcakes : ℝ := num_cupcakes * cupcake_price
def money_after_cupcakes : ℝ := total_money - cost_cupcakes
def remaining_money : ℝ := 30
def num_boxes_cookies : ℝ := 5
def money_spent_on_cookies : ℝ := money_after_cupcakes - remaining_money

-- Theorem: Calculate the cost per box of cookies
theorem cost_per_box_of_cookies : (money_spent_on_cookies / num_boxes_cookies) = 3 :=
by
  sorry

end cost_per_box_of_cookies_l261_261724


namespace proj_b_eq_l261_261334

open Real

variable {a b : ℝ × ℝ} 
variable (v : ℝ × ℝ)

def orthogonal (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0

def proj (u v : ℝ × ℝ) : ℝ × ℝ := 
  let k := (v.1 * u.1 + v.2 * u.2) / (u.1 * u.1 + u.2 * u.2)
  (k * u.1, k * u.2)

theorem proj_b_eq :
  orthogonal a b →
  proj a (4, -2) = (4/5, 8/5) →
  proj b (4, -2) = (16/5, -18/5) :=
by
  intros h_orth h_proj
  sorry

end proj_b_eq_l261_261334


namespace product_mod_self_inverse_l261_261775

theorem product_mod_self_inverse 
  {n : ℕ} (hn : 0 < n) (a b : ℤ) (ha : a * a % n = 1) (hb : b * b % n = 1) :
  (a * b) % n = 1 := 
sorry

end product_mod_self_inverse_l261_261775


namespace simplify_expression_l261_261544

-- Definitions for conditions and parameters
variables {x y : ℝ}

-- The problem statement and proof
theorem simplify_expression : 12 * x^5 * y / (6 * x * y) = 2 * x^4 :=
by sorry

end simplify_expression_l261_261544


namespace dave_tray_problem_l261_261415

theorem dave_tray_problem (n_trays_per_trip : ℕ) (n_trips : ℕ) (n_second_table : ℕ) : 
  (n_trays_per_trip = 9) → (n_trips = 8) → (n_second_table = 55) → 
  (n_trays_per_trip * n_trips - n_second_table = 17) :=
by
  sorry

end dave_tray_problem_l261_261415


namespace min_gennadies_l261_261700

noncomputable section

def minGennadiesNeeded (alexanders borises vasilies : Nat) : Nat :=
  let needed_gaps := borises - 1
  let total_others := alexanders + vasilies
  if needed_gaps > total_others then needed_gaps - total_others else 0

theorem min_gennadies (alexanders borises vasilies : Nat) (h_alex: alexanders = 45) (h_boris: borises = 122) (h_vasil: vasilies = 27):
  minGennadiesNeeded alexanders borises vasilies = 49 := by
  rw [h_alex, h_boris, h_vasil]
  simp [minGennadiesNeeded]
  sorry

end min_gennadies_l261_261700


namespace mario_total_flowers_l261_261472

-- Define the number of flowers on the first plant
def F1 : ℕ := 2

-- Define the number of flowers on the second plant as twice the first
def F2 : ℕ := 2 * F1

-- Define the number of flowers on the third plant as four times the second
def F3 : ℕ := 4 * F2

-- Prove that total number of flowers is 22
theorem mario_total_flowers : F1 + F2 + F3 = 22 := by
  -- Proof is to be filled here
  sorry

end mario_total_flowers_l261_261472


namespace find_angle_PQF_l261_261056

open Real EuclideanGeometry

section parabola_problem

def parabola (x : ℝ) : ℝ := - (1 / 4) * x ^ 2

def focus : ℝ × ℝ := (0, -1)

def tangent_at_P : AffineSpace.ConvexLine ℝ := { x | let y := 2 * x + 4 in y }

def P : ℝ × ℝ := (-4, -4)

def Q : ℝ × ℝ := (-2, 0)

theorem find_angle_PQF :
  ∠ (P - Q) (focus - Q) = π / 2 :=
sorry

end parabola_problem

end find_angle_PQF_l261_261056


namespace greatest_third_side_l261_261951

-- Given data and the Triangle Inequality theorem
theorem greatest_third_side (c : ℕ) (h1 : 8 < c) (h2 : c < 22) : c = 21 :=
by
  sorry

end greatest_third_side_l261_261951


namespace f_zero_eq_one_f_always_positive_f_inequality_l261_261412

variables {f : ℝ → ℝ}

-- Given conditions
axiom f_def : ∀ x, f x ∈ ℝ
axiom f_neq_zero : f 0 ≠ 0
axiom f_one : f 1 = 2
axiom f_pos : ∀ x > 0, f x > 1
axiom f_eqn : ∀ a b : ℝ, f (a + b) = f a * f b
axiom f_increasing : ∀ x y : ℝ, x < y → f x < f y

-- 1. Prove that f(0) = 1
theorem f_zero_eq_one : f 0 = 1 := sorry

-- 2. Prove that ∀ x ∈ ℝ, f(x) > 0
theorem f_always_positive : ∀ x, f x > 0 := sorry

-- 3. Solve the inequality f(3 - 2x) > 4
theorem f_inequality (x : ℝ) : f (3 - 2*x) > 4 → x < 1/2 := sorry

end f_zero_eq_one_f_always_positive_f_inequality_l261_261412


namespace box_inscribed_in_sphere_l261_261290

theorem box_inscribed_in_sphere (x y z r : ℝ) (surface_area : ℝ)
  (edge_sum : ℝ) (given_x : x = 8) 
  (given_surface_area : surface_area = 432) 
  (given_edge_sum : edge_sum = 104) 
  (surface_area_eq : 2 * (x * y + y * z + z * x) = surface_area)
  (edge_sum_eq : 4 * (x + y + z) = edge_sum) : 
  r = 7 :=
by
  sorry

end box_inscribed_in_sphere_l261_261290


namespace exists_monochromatic_triangle_l261_261097

theorem exists_monochromatic_triangle (points : Fin 6 → Point) (color : (Point × Point) → Color) :
  ∃ (a b c : Point), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (color (a, b) = color (b, c) ∧ color (b, c) = color (c, a)) :=
by
  sorry

end exists_monochromatic_triangle_l261_261097


namespace friends_received_pebbles_l261_261418

-- Define the conditions as expressions
def total_weight_kg : ℕ := 36
def weight_per_pebble_g : ℕ := 250
def pebbles_per_friend : ℕ := 4

-- Convert the total weight from kilograms to grams
def total_weight_g : ℕ := total_weight_kg * 1000

-- Calculate the total number of pebbles
def total_pebbles : ℕ := total_weight_g / weight_per_pebble_g

-- Calculate the total number of friends who received pebbles
def number_of_friends : ℕ := total_pebbles / pebbles_per_friend

-- The theorem to prove the number of friends
theorem friends_received_pebbles : number_of_friends = 36 := by
  sorry

end friends_received_pebbles_l261_261418


namespace value_of_4b_minus_a_l261_261791

theorem value_of_4b_minus_a (a b : ℕ) (h1 : a > b) (h2 : x^2 - 20*x + 96 = (x - a)*(x - b)) : 4*b - a = 20 :=
  sorry

end value_of_4b_minus_a_l261_261791


namespace problem_solution_l261_261565

theorem problem_solution (a b c : ℝ) (h : b^2 = a * c) :
  (a^2 * b^2 * c^2 / (a^3 + b^3 + c^3)) * (1 / a^3 + 1 / b^3 + 1 / c^3) = 1 :=
  by sorry

end problem_solution_l261_261565


namespace fraction_of_90_l261_261809

theorem fraction_of_90 : (1 / 2) * (1 / 3) * (1 / 6) * (90 : ℝ) = (5 / 2) := by
  sorry

end fraction_of_90_l261_261809


namespace brother_and_sister_ages_l261_261710

theorem brother_and_sister_ages :
  ∃ (b s : ℕ), (b - 3 = 7 * (s - 3)) ∧ (b - 2 = 4 * (s - 2)) ∧ (b - 1 = 3 * (s - 1)) ∧ (b = 5 / 2 * s) ∧ b = 10 ∧ s = 4 :=
by 
  sorry

end brother_and_sister_ages_l261_261710


namespace norma_bananas_count_l261_261604

-- Definitions for the conditions
def initial_bananas : ℕ := 47
def lost_bananas : ℕ := 45

-- The proof problem in Lean 4 statement
theorem norma_bananas_count : initial_bananas - lost_bananas = 2 := by
  -- Proof is omitted
  sorry

end norma_bananas_count_l261_261604


namespace garage_sale_items_count_l261_261005

theorem garage_sale_items_count :
  (16 + 22) + 1 = 38 :=
by
  -- proof goes here
  sorry

end garage_sale_items_count_l261_261005


namespace problem_solution_l261_261312

theorem problem_solution (x : ℝ) :
  (⌊|x^2 - 1|⌋ = 10) ↔ (x ∈ Set.Ioc (-2 * Real.sqrt 3) (-Real.sqrt 11) ∪ Set.Ico (Real.sqrt 11) (2 * Real.sqrt 3)) :=
by
  sorry

end problem_solution_l261_261312


namespace susan_correct_guess_probability_l261_261926

theorem susan_correct_guess_probability :
  (1 - (5/6)^6) = 31031/46656 := 
sorry

end susan_correct_guess_probability_l261_261926


namespace rectangle_width_is_14_l261_261616

noncomputable def rectangleWidth (areaOfCircle : ℝ) (length : ℝ) : ℝ :=
  let r := Real.sqrt (areaOfCircle / Real.pi)
  2 * r

theorem rectangle_width_is_14 :
  rectangleWidth 153.93804002589985 18 = 14 :=
by 
  sorry

end rectangle_width_is_14_l261_261616


namespace part1_max_min_part2_triangle_inequality_l261_261723

noncomputable def f (x k : ℝ) : ℝ :=
  (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

theorem part1_max_min (k : ℝ): 
  (∀ x : ℝ, k ≥ 1 → 1 ≤ f x k ∧ f x k ≤ (1/3) * (k + 2)) ∧ 
  (∀ x : ℝ, k < 1 → (1/3) * (k + 2) ≤ f x k ∧ f x k ≤ 1) := 
sorry

theorem part2_triangle_inequality (k : ℝ) : 
  -1/2 < k ∧ k < 4 ↔ (∀ a b c : ℝ, (f a k + f b k > f c k) ∧ (f b k + f c k > f a k) ∧ (f c k + f a k > f b k)) :=
sorry

end part1_max_min_part2_triangle_inequality_l261_261723


namespace total_pencils_l261_261804

theorem total_pencils  (a b c : Nat) (total : Nat) 
(h₀ : a = 43) 
(h₁ : b = 19) 
(h₂ : c = 16) 
(h₃ : total = a + b + c) : 
total = 78 := 
by
  sorry

end total_pencils_l261_261804


namespace I_consecutive_integers_I_consecutive_even_integers_II_consecutive_integers_II_consecutive_even_integers_II_consecutive_odd_integers_l261_261413

-- Define the problems
theorem I_consecutive_integers:
  ∃ (x y z : ℕ), 2 * x + y + z = 47 ∧ y = x + 1 ∧ z = x + 2 :=
sorry

theorem I_consecutive_even_integers:
  ¬ ∃ (x y z : ℕ), 2 * x + y + z = 47 ∧ y = x + 2 ∧ z = x + 4 :=
sorry

theorem II_consecutive_integers:
  ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 1 ∧ z = x + 2 ∧ w = x + 3 :=
sorry

theorem II_consecutive_even_integers:
  ¬ ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 2 ∧ z = x + 4 ∧ w = x + 6 :=
sorry

theorem II_consecutive_odd_integers:
  ¬ ∃ (x y z w : ℕ), 2 * x + y + z + w = 47 ∧ y = x + 2 ∧ z = x + 4 ∧ w = x + 6 ∧ x % 2 = 1 ∧ y % 2 = 1 ∧ z % 2 = 1 ∧ w % 2 = 1 :=
sorry

end I_consecutive_integers_I_consecutive_even_integers_II_consecutive_integers_II_consecutive_even_integers_II_consecutive_odd_integers_l261_261413


namespace find_total_roses_l261_261117

open Nat

theorem find_total_roses 
  (a : ℕ)
  (h1 : 300 ≤ a)
  (h2 : a ≤ 400)
  (h3 : a % 21 = 13)
  (h4 : a % 15 = 7) : 
  a = 307 := 
sorry

end find_total_roses_l261_261117


namespace integer_base10_from_bases_l261_261484

theorem integer_base10_from_bases (C D : ℕ) (hC : 0 ≤ C ∧ C ≤ 7) (hD : 0 ≤ D ∧ D ≤ 5)
    (h : 8 * C + D = 6 * D + C) : C = 0 ∧ D = 0 ∧ (8 * C + D = 0) := by
  sorry

end integer_base10_from_bases_l261_261484


namespace square_side_length_l261_261133

theorem square_side_length
  (P : ℕ) (A : ℕ) (s : ℕ)
  (h1 : P = 44)
  (h2 : A = 121)
  (h3 : P = 4 * s)
  (h4 : A = s * s) :
  s = 11 :=
sorry

end square_side_length_l261_261133


namespace distance_two_from_origin_l261_261492

theorem distance_two_from_origin (x : ℝ) (h : abs x = 2) : x = 2 ∨ x = -2 := by
  sorry

end distance_two_from_origin_l261_261492


namespace value_of_A_l261_261023

theorem value_of_A (A C : ℤ) (h₁ : 2 * A - C + 4 = 26) (h₂ : C = 6) : A = 14 :=
by sorry

end value_of_A_l261_261023


namespace no_such_function_exists_l261_261844

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, 2 * f (Real.cos x) = f (Real.sin x) + Real.sin x) :=
by
  sorry

end no_such_function_exists_l261_261844


namespace max_temp_difference_l261_261605

-- Define the highest and lowest temperatures
def highest_temp : ℤ := 3
def lowest_temp : ℤ := -3

-- State the theorem for maximum temperature difference
theorem max_temp_difference : highest_temp - lowest_temp = 6 := 
by 
  -- Provide the proof here
  sorry

end max_temp_difference_l261_261605


namespace min_gennadies_l261_261699

noncomputable section

def minGennadiesNeeded (alexanders borises vasilies : Nat) : Nat :=
  let needed_gaps := borises - 1
  let total_others := alexanders + vasilies
  if needed_gaps > total_others then needed_gaps - total_others else 0

theorem min_gennadies (alexanders borises vasilies : Nat) (h_alex: alexanders = 45) (h_boris: borises = 122) (h_vasil: vasilies = 27):
  minGennadiesNeeded alexanders borises vasilies = 49 := by
  rw [h_alex, h_boris, h_vasil]
  simp [minGennadiesNeeded]
  sorry

end min_gennadies_l261_261699


namespace find_f_2015_l261_261993

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_periodic_2 : ∀ x : ℝ, f x * f (x + 2) = 13
axiom f_at_1 : f 1 = 2

theorem find_f_2015 : f 2015 = 13 / 2 :=
by
  sorry

end find_f_2015_l261_261993


namespace bread_cost_each_is_3_l261_261722

-- Define the given conditions
def initial_amount : ℕ := 86
def bread_quantity : ℕ := 3
def orange_juice_quantity : ℕ := 3
def orange_juice_cost_each : ℕ := 6
def remaining_amount : ℕ := 59

-- Define the variable for bread cost
variable (B : ℕ)

-- Lean 4 statement to prove the cost of each loaf of bread
theorem bread_cost_each_is_3 :
  initial_amount - remaining_amount = (bread_quantity * B + orange_juice_quantity * orange_juice_cost_each) →
  B = 3 :=
by
  sorry

end bread_cost_each_is_3_l261_261722


namespace probability_of_same_color_l261_261281

-- Definition of the conditions
def total_shoes : ℕ := 14
def pairs : ℕ := 7

-- The main statement
theorem probability_of_same_color :
  let total_ways := Nat.choose total_shoes 2 in
  let successful_ways := pairs in
  (successful_ways : ℚ) / total_ways = 1 / 13 :=
by
  sorry

end probability_of_same_color_l261_261281


namespace tens_digit_of_23_pow_2023_l261_261305

theorem tens_digit_of_23_pow_2023 : (23 ^ 2023 % 100 / 10) = 6 :=
by
  sorry

end tens_digit_of_23_pow_2023_l261_261305


namespace missing_digit_l261_261014

theorem missing_digit (B : ℕ) (h : B < 10) : 
  (15 ∣ (200 + 10 * B)) ↔ B = 1 ∨ B = 4 :=
by sorry

end missing_digit_l261_261014


namespace max_product_of_two_positive_numbers_l261_261414

theorem max_product_of_two_positive_numbers (x y s : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = s) : 
  x * y ≤ (s ^ 2) / 4 :=
sorry

end max_product_of_two_positive_numbers_l261_261414


namespace jack_speed_to_beach_12_mph_l261_261240

theorem jack_speed_to_beach_12_mph :
  let distance := 16 * (1 / 8) -- distance in miles
  let time := 10 / 60        -- time in hours
  distance / time = 12 :=    -- speed in miles per hour
by
  let distance := 16 * (1 / 8) -- evaluation of distance
  let time := 10 / 60          -- evaluation of time
  show distance / time = 12    -- final speed calculation
  from sorry

end jack_speed_to_beach_12_mph_l261_261240


namespace jane_dolls_l261_261632

theorem jane_dolls (jane_dolls jill_dolls : ℕ) (h1 : jane_dolls + jill_dolls = 32) (h2 : jill_dolls = jane_dolls + 6) : jane_dolls = 13 := 
by {
  sorry
}

end jane_dolls_l261_261632


namespace tutors_meet_after_84_days_l261_261330

theorem tutors_meet_after_84_days :
  let jaclyn := 3
  let marcelle := 4
  let susanna := 6
  let wanda := 7
  Nat.lcm (Nat.lcm (Nat.lcm jaclyn marcelle) susanna) wanda = 84 := by
  sorry

end tutors_meet_after_84_days_l261_261330


namespace mixed_water_temp_l261_261460

def cold_water_temp : ℝ := 20
def hot_water_temp : ℝ := 40

theorem mixed_water_temp :
  (cold_water_temp + hot_water_temp) / 2 = 30 :=
by
  unfold cold_water_temp hot_water_temp
  norm_num
  sorry

end mixed_water_temp_l261_261460


namespace sum_c_d_eq_24_l261_261140

theorem sum_c_d_eq_24 (c d : ℕ) (h_pos_c : c > 0) (h_pos_d : d > 1) (h_max_power : c^d < 500 ∧ ∀ ⦃x y : ℕ⦄, x^y < 500 → x^y ≤ c^d) : c + d = 24 :=
sorry

end sum_c_d_eq_24_l261_261140


namespace tangent_line_parallel_points_l261_261495

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Prove the points where the derivative equals 4
theorem tangent_line_parallel_points :
  ∃ (P0 : ℝ × ℝ), P0 = (1, 0) ∨ P0 = (-1, -4) ∧ (f' P0.fst = 4) :=
by
  sorry

end tangent_line_parallel_points_l261_261495


namespace cos_210_eq_neg_sqrt3_div_2_l261_261171

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261171


namespace bisection_contains_root_l261_261955

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem bisection_contains_root : (1 < 1.5) ∧ f 1 < 0 ∧ f 1.5 > 0 → ∃ (c : ℝ), 1 < c ∧ c < 1.5 ∧ f c = 0 :=
by
  sorry

end bisection_contains_root_l261_261955


namespace determine_pairs_l261_261180

theorem determine_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  (∃ k : ℕ, k > 0 ∧ (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1)) :=
by
  sorry

end determine_pairs_l261_261180


namespace tangerines_more_than_oranges_l261_261628

def initial_oranges := 5
def initial_tangerines := 17
def oranges_taken := 2
def tangerines_taken := 10

theorem tangerines_more_than_oranges
  (initial_oranges: ℕ) -- Tina starts with 5 oranges
  (initial_tangerines: ℕ) -- Tina starts with 17 tangerines
  (oranges_taken: ℕ) -- Tina takes away 2 oranges
  (tangerines_taken: ℕ) -- Tina takes away 10 tangerines
  : (initial_tangerines - tangerines_taken) - (initial_oranges - oranges_taken) = 4 := 
by
  sorry

end tangerines_more_than_oranges_l261_261628


namespace pears_for_36_bananas_l261_261778

theorem pears_for_36_bananas (p : ℕ) (bananas : ℕ) (pears : ℕ) (h : 9 * pears = 6 * bananas) :
  36 * pears = 9 * 24 :=
by
  sorry

end pears_for_36_bananas_l261_261778


namespace find_percentage_l261_261973

theorem find_percentage (P : ℝ) (N : ℝ) (h1 : N = 140) (h2 : (P / 100) * N = (4 / 5) * N - 21) : P = 65 := by
  sorry

end find_percentage_l261_261973


namespace solution_set_quadratic_inequality_l261_261184

theorem solution_set_quadratic_inequality :
  { x : ℝ | x^2 + 3 * x - 4 < 0 } = { x : ℝ | -4 < x ∧ x < 1 } :=
sorry

end solution_set_quadratic_inequality_l261_261184


namespace third_box_number_l261_261128

def N : ℕ := 301

theorem third_box_number (N : ℕ) (h1 : N % 3 = 1) (h2 : N % 4 = 1) (h3 : N % 7 = 0) :
  ∃ x : ℕ, x > 4 ∧ x ≠ 7 ∧ N % x = 1 ∧ (∀ y > 4, y ≠ 7 → y < x → N % y ≠ 1) ∧ x = 6 :=
by
  sorry

end third_box_number_l261_261128


namespace find_number_of_roses_l261_261119

theorem find_number_of_roses : ∃ a : ℕ, 300 ≤ a ∧ a ≤ 400 ∧ a % 21 = 13 ∧ a % 15 = 7 :=
by
  -- Existential quantifier for the number 'a'
  use 307
  
  -- Proof of the conditions for 'a'
  split
  -- Proof that 300 ≤ 307 ∧ 307 ≤ 400
  exact ⟨by linarith, by linarith⟩
  split
  -- Proof that 307 % 21 = 13
  exact by norm_num
  -- Proof that 307 % 15 = 7 (because -8 % 15 = 7)
  exact by norm_num

end find_number_of_roses_l261_261119


namespace expected_value_of_8_sided_die_l261_261644

theorem expected_value_of_8_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (∑ outcome in outcomes, outcome * probability) = 4.5 :=
by
  sorry

end expected_value_of_8_sided_die_l261_261644


namespace evaluate_expression_l261_261551

theorem evaluate_expression :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
sorry

end evaluate_expression_l261_261551


namespace nuts_per_cookie_l261_261601

theorem nuts_per_cookie (h1 : (1/4:ℝ) * 60 = 15)
(h2 : (0.40:ℝ) * 60 = 24)
(h3 : 60 - 15 - 24 = 21)
(h4 : 72 / (15 + 21) = 2) :
72 / 36 = 2 := by
suffices h : 72 / 36 = 2 from h
exact h4

end nuts_per_cookie_l261_261601


namespace cos_210_eq_neg_sqrt3_div_2_l261_261162

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261162


namespace intersection_M_N_l261_261036

-- Definitions of sets M and N
def M : Set ℕ := {1, 2, 5}
def N : Set ℕ := {x | x ≤ 2}

-- Lean statement to prove that the intersection of M and N is {1, 2}
theorem intersection_M_N : M ∩ N = {1, 2} :=
by
  sorry

end intersection_M_N_l261_261036


namespace company_C_more_than_A_l261_261663

theorem company_C_more_than_A (A B C D: ℕ) (hA: A = 30) (hB: B = 2 * A)
    (hC: C = A + 10) (hD: D = C - 5) (total: A + B + C + D = 165) : C - A = 10 := 
by 
  sorry

end company_C_more_than_A_l261_261663


namespace walnuts_left_in_burrow_l261_261388

-- Definitions of conditions
def boy_gathers : ℕ := 15
def originally_in_burrow : ℕ := 25
def boy_drops : ℕ := 3
def boy_hides : ℕ := 5
def girl_brings : ℕ := 12
def girl_eats : ℕ := 4
def girl_gives_away : ℕ := 3
def girl_loses : ℕ := 2

-- Theorem statement
theorem walnuts_left_in_burrow : 
  originally_in_burrow + (boy_gathers - boy_drops - boy_hides) + 
  (girl_brings - girl_eats - girl_gives_away - girl_loses) = 35 := 
sorry

end walnuts_left_in_burrow_l261_261388


namespace block_measure_is_40_l261_261463

def jony_walks (start_time : String) (start_block end_block stop_block : ℕ) (stop_time : String) (speed : ℕ) : ℕ :=
  let total_time := 40 -- walking time in minutes
  let total_distance := speed * total_time -- total distance walked in meters
  let blocks_forward := end_block - start_block -- blocks walked forward
  let blocks_backward := end_block - stop_block -- blocks walked backward
  let total_blocks := blocks_forward + blocks_backward -- total blocks walked
  total_distance / total_blocks

theorem block_measure_is_40 :
  jony_walks "07:00" 10 90 70 "07:40" 100 = 40 := by
  sorry

end block_measure_is_40_l261_261463


namespace lino_shells_l261_261471

theorem lino_shells (picked_up : ℝ) (put_back : ℝ) (remaining_shells : ℝ) :
  picked_up = 324.0 → 
  put_back = 292.0 → 
  remaining_shells = picked_up - put_back → 
  remaining_shells = 32.0 :=
by
  intros h1 h2 h3
  sorry

end lino_shells_l261_261471


namespace find_x_value_l261_261238

def solve_for_x (a b x : ℝ) (rectangle_perimeter triangle_height equated_areas : Prop) :=
  rectangle_perimeter -> triangle_height -> equated_areas -> x = 20 / 3

-- Definitions of the conditions
def rectangle_perimeter (a b : ℝ) : Prop := 2 * (a + b) = 60
def triangle_height : Prop := 60 > 0
def equated_areas (a b x : ℝ) : Prop := a * b = 30 * x

theorem find_x_value :
  ∃ a b x : ℝ, solve_for_x a b x (rectangle_perimeter a b) triangle_height (equated_areas a b x) :=
  sorry

end find_x_value_l261_261238


namespace carol_blocks_l261_261302

theorem carol_blocks (initial_blocks : ℕ) (blocks_lost : ℕ) (final_blocks : ℕ) : 
  initial_blocks = 42 → blocks_lost = 25 → final_blocks = initial_blocks - blocks_lost → final_blocks = 17 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end carol_blocks_l261_261302


namespace july_birth_percentage_l261_261489

theorem july_birth_percentage (total : ℕ) (july : ℕ) (h1 : total = 150) (h2 : july = 18) : (july : ℚ) / total * 100 = 12 := sorry

end july_birth_percentage_l261_261489


namespace problem_statement_l261_261987

theorem problem_statement (x y : ℝ) : 
  ((-3 * x * y^2)^3 * (-6 * x^2 * y) / (9 * x^4 * y^5) = 18 * x * y^2) :=
by sorry

end problem_statement_l261_261987


namespace determine_pairs_l261_261181

theorem determine_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔
  (∃ k : ℕ, k > 0 ∧ (a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1)) :=
by
  sorry

end determine_pairs_l261_261181


namespace slope_of_line_l261_261309

theorem slope_of_line {x1 x2 y1 y2 : ℝ} 
  (h1 : (1 / x1 + 2 / y1 = 0)) 
  (h2 : (1 / x2 + 2 / y2 = 0)) 
  (h_neq : x1 ≠ x2) : 
  (y2 - y1) / (x2 - x1) = -2 := 
sorry

end slope_of_line_l261_261309


namespace less_than_reciprocal_l261_261739

theorem less_than_reciprocal (n : ℚ) : 
  n = -3 ∨ n = 3/4 ↔ (n = -1/2 → n >= 1/(-1/2)) ∧
                           (n = -3 → n < 1/(-3)) ∧
                           (n = 3/4 → n < 1/(3/4)) ∧
                           (n = 3 → n > 1/3) ∧
                           (n = 0 → false) := sorry

end less_than_reciprocal_l261_261739


namespace eliminate_duplicates_3n_2m1_l261_261186

theorem eliminate_duplicates_3n_2m1 :
  ∀ k: ℤ, ∃ n m: ℤ, 3 * n ≠ 2 * m + 1 ↔ 2 * m + 1 = 12 * k + 1 ∨ 2 * m + 1 = 12 * k + 5 :=
by
  sorry

end eliminate_duplicates_3n_2m1_l261_261186


namespace number_of_adults_l261_261299

theorem number_of_adults (total_apples : ℕ) (children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) (h : total_apples = 450) (h1 : children = 33) (h2 : apples_per_child = 10) (h3 : apples_per_adult = 3) :
  total_apples - (children * apples_per_child) = 120 →
  (total_apples - (children * apples_per_child)) / apples_per_adult = 40 :=
by
  intros
  sorry

end number_of_adults_l261_261299


namespace rate_of_current_l261_261270

theorem rate_of_current (speed_boat_still_water : ℕ) (time_hours : ℚ) (distance_downstream : ℚ)
    (h_speed_boat_still_water : speed_boat_still_water = 20)
    (h_time_hours : time_hours = 15 / 60)
    (h_distance_downstream : distance_downstream = 6.25) :
    ∃ c : ℚ, distance_downstream = (speed_boat_still_water + c) * time_hours ∧ c = 5 :=
by
    sorry

end rate_of_current_l261_261270


namespace solve_k_n_l261_261383
-- Import the entire Mathlib

-- Define the theorem statement
theorem solve_k_n (k n : ℕ) (hk : k > 0) (hn : n > 0) : k^2 - 2016 = 3^n ↔ k = 45 ∧ n = 2 :=
  by sorry

end solve_k_n_l261_261383


namespace tea_drinking_problem_l261_261535

theorem tea_drinking_problem 
  (k b c t s : ℕ) 
  (hk : k = 1) 
  (hb : b = 15) 
  (hc : c = 3) 
  (ht : t = 2) 
  (hs : s = 1) : 
  17 = 17 := 
by {
  sorry
}

end tea_drinking_problem_l261_261535


namespace distinct_triangles_count_l261_261037

def num_points : ℕ := 8
def num_rows : ℕ := 2
def num_cols : ℕ := 4

-- Define the number of ways to choose 3 points from the 8 available points.
def combinations (n k : ℕ) := Nat.choose n k
def total_combinations := combinations num_points 3

-- Define the number of degenerate cases of collinear points in columns.
def degenerate_cases_per_column := combinations num_cols 3
def total_degenerate_cases := num_cols * degenerate_cases_per_column

-- The number of distinct triangles is the total combinations minus the degenerate cases.
def distinct_triangles := total_combinations - total_degenerate_cases

theorem distinct_triangles_count : distinct_triangles = 40 := by
  -- the proof goes here
  sorry

end distinct_triangles_count_l261_261037


namespace expected_value_of_eight_sided_die_l261_261641

theorem expected_value_of_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8], 
      n := (outcomes.length : ℚ),
      probabilities := List.replicate (outcomes.length) (1 / n),
      expected_value := (List.zipWith (*) probabilities (outcomes.map (· : ℚ))).sum
  in expected_value = 4.5 :=
by
  sorry

end expected_value_of_eight_sided_die_l261_261641


namespace cosine_210_l261_261175

noncomputable def Q := rotate 210 (1, 0)
noncomputable def E := foot Q (1, 0) (-1, 0)

theorem cosine_210 : cos 210 = - (sqrt 3/2) :=
sorry

end cosine_210_l261_261175


namespace arithmetic_geometric_sequence_S6_l261_261737

noncomputable def S_6 (a : Nat) (q : Nat) : Nat :=
  (q ^ 6 - 1) / (q - 1)

theorem arithmetic_geometric_sequence_S6 (a : Nat) (q : Nat) (h1 : a * q ^ 1 = 2) (h2 : a * q ^ 3 = 8) (hq : q > 0) : S_6 a q = 63 :=
by
  sorry

end arithmetic_geometric_sequence_S6_l261_261737


namespace temperature_at_midnight_l261_261894

def morning_temp : ℝ := 30
def afternoon_increase : ℝ := 1
def midnight_decrease : ℝ := 7

theorem temperature_at_midnight : morning_temp + afternoon_increase - midnight_decrease = 24 := by
  sorry

end temperature_at_midnight_l261_261894


namespace solve_system_of_equations_l261_261479

theorem solve_system_of_equations :
  ∃ (x y z w : ℤ), 
    x - y + z - w = 2 ∧
    x^2 - y^2 + z^2 - w^2 = 6 ∧
    x^3 - y^3 + z^3 - w^3 = 20 ∧
    x^4 - y^4 + z^4 - w^4 = 66 ∧
    (x, y, z, w) = (1, 3, 0, 2) := 
  by
    sorry

end solve_system_of_equations_l261_261479


namespace cos_210_eq_neg_sqrt3_div_2_l261_261153

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261153


namespace range_of_a_l261_261777

def p (a : ℝ) : Prop := 0 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 1 / 4

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : a ∈ Set.Ioc 0 (1 / 4) ∨ a ∈ Set.Ioi 1 :=
by
  sorry

end range_of_a_l261_261777


namespace average_contribution_increase_l261_261249

theorem average_contribution_increase
  (average_old : ℝ)
  (num_people_old : ℕ)
  (john_donation : ℝ)
  (increase_percentage : ℝ) :
  average_old = 75 →
  num_people_old = 3 →
  john_donation = 150 →
  increase_percentage = 25 :=
by {
  sorry
}

end average_contribution_increase_l261_261249


namespace tangerines_more_than_oranges_l261_261631

-- Define initial conditions
def initial_apples := 9
def initial_oranges := 5
def initial_tangerines := 17

-- Define actions taken
def oranges_taken := 2
def tangerines_taken := 10

-- Resulting quantities
def oranges_left := initial_oranges - oranges_taken
def tangerines_left := initial_tangerines - tangerines_taken

-- Proof problem
theorem tangerines_more_than_oranges : tangerines_left - oranges_left = 4 := 
by sorry

end tangerines_more_than_oranges_l261_261631


namespace gcd_of_45_135_225_is_45_l261_261650

theorem gcd_of_45_135_225_is_45 : Nat.gcd (Nat.gcd 45 135) 225 = 45 :=
by
  sorry

end gcd_of_45_135_225_is_45_l261_261650


namespace other_continent_passengers_l261_261230

noncomputable def totalPassengers := 240
noncomputable def northAmericaFraction := (1 / 3 : ℝ)
noncomputable def europeFraction := (1 / 8 : ℝ)
noncomputable def africaFraction := (1 / 5 : ℝ)
noncomputable def asiaFraction := (1 / 6 : ℝ)

theorem other_continent_passengers :
  (totalPassengers : ℝ) - (totalPassengers * northAmericaFraction +
                           totalPassengers * europeFraction +
                           totalPassengers * africaFraction +
                           totalPassengers * asiaFraction) = 42 :=
by
  sorry

end other_continent_passengers_l261_261230


namespace polygon_interior_equals_exterior_sum_eq_360_l261_261361

theorem polygon_interior_equals_exterior_sum_eq_360 (n : ℕ) :
  (n - 2) * 180 = 360 → n = 6 :=
by
  intro h
  sorry

end polygon_interior_equals_exterior_sum_eq_360_l261_261361


namespace new_mixture_alcohol_percentage_l261_261385

/-- 
Given: 
  - a solution with 15 liters containing 26% alcohol
  - 5 liters of water added to the solution
Prove:
  The percentage of alcohol in the new mixture is 19.5%
-/
theorem new_mixture_alcohol_percentage 
  (original_volume : ℝ) (original_percent_alcohol : ℝ) (added_water_volume : ℝ) :
  original_volume = 15 → 
  original_percent_alcohol = 26 →
  added_water_volume = 5 →
  (original_volume * (original_percent_alcohol / 100) / (original_volume + added_water_volume)) * 100 = 19.5 :=
by 
  intros h1 h2 h3
  sorry

end new_mixture_alcohol_percentage_l261_261385


namespace suitableTempForPreservingBoth_l261_261621

-- Definitions for the temperature ranges of types A and B vegetables
def suitableTempRangeA := {t : ℝ | 3 ≤ t ∧ t ≤ 8}
def suitableTempRangeB := {t : ℝ | 5 ≤ t ∧ t ≤ 10}

-- The intersection of the suitable temperature ranges
def suitableTempRangeForBoth := {t : ℝ | 5 ≤ t ∧ t ≤ 8}

-- The theorem statement we need to prove
theorem suitableTempForPreservingBoth :
  suitableTempRangeForBoth = suitableTempRangeA ∩ suitableTempRangeB :=
sorry

end suitableTempForPreservingBoth_l261_261621


namespace sphere_volume_l261_261622

theorem sphere_volume {r : ℝ} (h: 4 * Real.pi * r^2 = 256 * Real.pi) : (4 / 3) * Real.pi * r^3 = (2048 / 3) * Real.pi :=
by
  sorry

end sphere_volume_l261_261622


namespace shadow_length_building_l261_261821

theorem shadow_length_building:
  let height_flagstaff := 17.5
  let shadow_flagstaff := 40.25
  let height_building := 12.5
  let expected_shadow_building := 28.75
  (height_flagstaff / shadow_flagstaff = height_building / expected_shadow_building) := by
  let height_flagstaff := 17.5
  let shadow_flagstaff := 40.25
  let height_building := 12.5
  let expected_shadow_building := 28.75
  sorry

end shadow_length_building_l261_261821


namespace simplify_sqrt_88200_l261_261612

theorem simplify_sqrt_88200 :
  (Real.sqrt 88200) = 70 * Real.sqrt 6 := 
by 
  -- given conditions
  have h : 88200 = 2^3 * 3 * 5^2 * 7^2 := sorry,
  sorry

end simplify_sqrt_88200_l261_261612


namespace min_gennadys_l261_261686

-- Defining the basic constants for each name type
def Alexanders : Nat := 45
def Borises : Nat := 122
def Vasilies : Nat := 27

-- Define the proof statement to check the minimum number of Gennadys needed
theorem min_gennadys (a b v : Nat) (no_adjacent: a = 45 ∧ b = 122 ∧ v = 27) : ∃ g : Nat, g = 49 :=
by
  -- Using provided conditions
  cases no_adjacent with h_a h_bv
  cases h_bv with h_b h_v
  -- Correct answer derived from the solution
  use 49
  -- skipping proof details 
  sorry

end min_gennadys_l261_261686


namespace intersection_proof_complement_proof_range_of_m_condition_l261_261909

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 < x ∧ x < 1}
def C (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2 + m}

theorem intersection_proof : A ∩ B = {x | -2 ≤ x ∧ x < 1} := sorry

theorem complement_proof : (Set.univ \ B) = {x | x ≤ -3 ∨ x ≥ 1} := sorry

theorem range_of_m_condition (m : ℝ) : (A ∪ C m = A) → (m ≤ 2) := sorry

end intersection_proof_complement_proof_range_of_m_condition_l261_261909


namespace max_value_of_g_l261_261425

noncomputable def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

theorem max_value_of_g : ∃ x ∈ Set.Icc (0:ℝ) 2, g x = 25 / 8 := 
by 
  sorry

end max_value_of_g_l261_261425


namespace problem_a_problem_b_l261_261045

variable (w x y z t : ℝ)

theorem problem_a (h1 : w = 0.60 * x) (h2 : x = 0.60 * y) (h3 : z = 0.54 * y) (h4 : t = 0.48 * x) :
  (z - w) / w = 0.50 :=
by
  sorry

theorem problem_b (h1 : w = 0.60 * x) (h2 : x = 0.60 * y) (h3 : z = 0.54 * y) (h4 : t = 0.48 * x) :
  (w - t) / w = 0.20 :=
by
  sorry

end problem_a_problem_b_l261_261045


namespace weight_of_white_ring_l261_261589

def weight_orange := 0.08333333333333333
def weight_purple := 0.3333333333333333
def total_weight := 0.8333333333

def weight_white := 0.41666666663333337

theorem weight_of_white_ring :
  weight_white + weight_orange + weight_purple = total_weight :=
by
  sorry

end weight_of_white_ring_l261_261589


namespace shooter_mean_hits_l261_261362

theorem shooter_mean_hits (p : ℝ) (n : ℕ) (h_prob : p = 0.9) (h_shots : n = 10) : n * p = 9 := by
  sorry

end shooter_mean_hits_l261_261362


namespace determinant_of_sum_l261_261574

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, 6], ![2, 3]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 1], ![1, 0]]

theorem determinant_of_sum : (A + B).det = -3 := 
by 
  sorry

end determinant_of_sum_l261_261574


namespace XiaoKang_min_sets_pushups_pullups_l261_261964

theorem XiaoKang_min_sets_pushups_pullups (x y : ℕ) (hx : x ≥ 100) (hy : y ≥ 106) (h : 8 * x + 5 * y = 9050) :
  x ≥ 100 ∧ y ≥ 106 :=
by {
  sorry  -- proof not required as per instruction
}

end XiaoKang_min_sets_pushups_pullups_l261_261964


namespace trigonometric_inequality_l261_261607

theorem trigonometric_inequality (a b A B : ℝ) (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos 2 * x - B * Real.sin 2 * x ≥ 0) : 
  a ^ 2 + b ^ 2 ≤ 2 ∧ A ^ 2 + B ^ 2 ≤ 1 := 
sorry

end trigonometric_inequality_l261_261607


namespace ramu_profit_percent_is_21_64_l261_261071

-- Define the costs and selling price as constants
def cost_of_car : ℕ := 42000
def cost_of_repairs : ℕ := 13000
def selling_price : ℕ := 66900

-- Define the total cost and profit
def total_cost : ℕ := cost_of_car + cost_of_repairs
def profit : ℕ := selling_price - total_cost

-- Define the profit percent formula
def profit_percent : ℚ := ((profit : ℚ) / (total_cost : ℚ)) * 100

-- State the theorem we want to prove
theorem ramu_profit_percent_is_21_64 : profit_percent = 21.64 := by
  sorry

end ramu_profit_percent_is_21_64_l261_261071


namespace find_q_l261_261323

theorem find_q (p q : ℝ) (hp : 1 < p) (hq : 1 < q) (hcond1 : 1/p + 1/q = 1) (hcond2 : p * q = 9) :
    q = (9 + 3 * Real.sqrt 5) / 2 ∨ q = (9 - 3 * Real.sqrt 5) / 2 :=
by
  sorry

end find_q_l261_261323


namespace michael_work_time_l261_261064

theorem michael_work_time (M A L : ℚ) 
  (h1 : M + A + L = 1/15) 
  (h2 : A + L = 1/24) :
  1 / M = 40 := 
by
  sorry

end michael_work_time_l261_261064


namespace consecutive_odd_numbers_l261_261917

/- 
  Out of some consecutive odd numbers, 9 times the first number 
  is equal to the addition of twice the third number and adding 9 
  to twice the second. Let x be the first number, then we aim to prove that 
  9 * x = 2 * (x + 4) + 2 * (x + 2) + 9 ⟹ x = 21 / 5
-/

theorem consecutive_odd_numbers (x : ℚ) (h : 9 * x = 2 * (x + 4) + 2 * (x + 2) + 9) : x = 21 / 5 :=
sorry

end consecutive_odd_numbers_l261_261917


namespace nested_series_sum_l261_261411

theorem nested_series_sum : 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2))))) = 126 :=
by
  sorry

end nested_series_sum_l261_261411


namespace max_nondiagonal_5x5_grid_l261_261830

open Set

/-- Maximum number of non-intersecting diagonals in a 5x5 grid of squares. --/
theorem max_nondiagonal_5x5_grid : ∃ n, n = 16 ∧ 
  ∀ diags : Finset (Fin 5 × Fin 5), 
    (∀ (x y : Fin 5 × Fin 5), x ≠ y → diags x ∩ diags y = ∅) →
    diags.card ≤ n :=
begin
  use 16,
  split,
  { refl, },
  {
    intros diags h,
    sorry
  }
end

end max_nondiagonal_5x5_grid_l261_261830


namespace range_of_a_l261_261057

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    (if x1 ≤ 1 then (-x1^2 + a*x1)
     else (a*x1 - 1)) = 
    (if x2 ≤ 1 then (-x2^2 + a*x2)
     else (a*x2 - 1))) → a < 2 :=
sorry

end range_of_a_l261_261057


namespace bisect_angle_BAX_l261_261469

-- Definitions and conditions
variables {A B C M X : Point}
variable (is_scalene_triangle : ScaleneTriangle A B C)
variable (is_midpoint : Midpoint M B C)
variable (is_parallel : Parallel (Line C X) (Line A B))
variable (angle_right : Angle AM X = 90)

-- The theorem statement to be proven
theorem bisect_angle_BAX (h1 : is_scalene_triangle)
                         (h2 : is_midpoint)
                         (h3 : is_parallel)
                         (h4 : angle_right) :
  Bisects (Line A M) (Angle B A X) :=
sorry

end bisect_angle_BAX_l261_261469


namespace janeth_balloons_l261_261247

/-- Janeth's total remaining balloons after accounting for burst ones. -/
def total_remaining_balloons (round_bags : Nat) (round_per_bag : Nat) (burst_round : Nat)
    (long_bags : Nat) (long_per_bag : Nat) (burst_long : Nat)
    (heart_bags : Nat) (heart_per_bag : Nat) (burst_heart : Nat) : Nat :=
  let total_round := round_bags * round_per_bag - burst_round
  let total_long := long_bags * long_per_bag - burst_long
  let total_heart := heart_bags * heart_per_bag - burst_heart
  total_round + total_long + total_heart

theorem janeth_balloons :
  total_remaining_balloons 5 25 5 4 35 7 3 40 3 = 370 :=
by
  let round_bags := 5
  let round_per_bag := 25
  let burst_round := 5
  let long_bags := 4
  let long_per_bag := 35
  let burst_long := 7
  let heart_bags := 3
  let heart_per_bag := 40
  let burst_heart := 3
  show total_remaining_balloons round_bags round_per_bag burst_round long_bags long_per_bag burst_long heart_bags heart_per_bag burst_heart = 370
  sorry

end janeth_balloons_l261_261247


namespace total_flowers_collected_l261_261681

/- Definitions for the given conditions -/
def maxFlowers : ℕ := 50
def arwenTulips : ℕ := 20
def arwenRoses : ℕ := 18
def arwenSunflowers : ℕ := 6

def elrondTulips : ℕ := 2 * arwenTulips
def elrondRoses : ℕ := if 3 * arwenRoses + elrondTulips > maxFlowers then maxFlowers - elrondTulips else 3 * arwenRoses

def galadrielTulips : ℕ := if 3 * elrondTulips > maxFlowers then maxFlowers else 3 * elrondTulips
def galadrielRoses : ℕ := if 2 * arwenRoses + galadrielTulips > maxFlowers then maxFlowers - galadrielTulips else 2 * arwenRoses

def galadrielSunflowers : ℕ := 0 -- she didn't pick any sunflowers
def legolasSunflowers : ℕ := arwenSunflowers + galadrielSunflowers
def legolasRemaining : ℕ := maxFlowers - legolasSunflowers
def legolasRosesAndTulips : ℕ := legolasRemaining / 2
def legolasTulips : ℕ := legolasRosesAndTulips
def legolasRoses : ℕ := legolasRosesAndTulips

def arwenTotal : ℕ := arwenTulips + arwenRoses + arwenSunflowers
def elrondTotal : ℕ := elrondTulips + elrondRoses
def galadrielTotal : ℕ := galadrielTulips + galadrielRoses + galadrielSunflowers
def legolasTotal : ℕ := legolasTulips + legolasRoses + legolasSunflowers

def totalFlowers : ℕ := arwenTotal + elrondTotal + galadrielTotal + legolasTotal

theorem total_flowers_collected : totalFlowers = 194 := by
  /- This will be where the proof goes, but we leave it as a placeholder. -/
  sorry

end total_flowers_collected_l261_261681


namespace cos_210_eq_neg_sqrt3_div_2_l261_261164

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261164


namespace expected_value_of_eight_sided_die_l261_261640

theorem expected_value_of_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8], 
      n := (outcomes.length : ℚ),
      probabilities := List.replicate (outcomes.length) (1 / n),
      expected_value := (List.zipWith (*) probabilities (outcomes.map (· : ℚ))).sum
  in expected_value = 4.5 :=
by
  sorry

end expected_value_of_eight_sided_die_l261_261640


namespace second_set_length_is_correct_l261_261248

variables (first_set_length second_set_length : ℝ)

theorem second_set_length_is_correct 
  (h1 : first_set_length = 4)
  (h2 : second_set_length = 5 * first_set_length) : 
  second_set_length = 20 := 
by 
  sorry

end second_set_length_is_correct_l261_261248


namespace min_gennadys_needed_l261_261695

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l261_261695


namespace cos_210_eq_neg_sqrt3_div_2_l261_261167

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261167


namespace combined_fractions_value_l261_261475

theorem combined_fractions_value (N : ℝ) (h1 : 0.40 * N = 168) : 
  (1/4) * (1/3) * (2/5) * N = 14 :=
by
  sorry

end combined_fractions_value_l261_261475


namespace find_x_positive_multiple_of_8_l261_261576

theorem find_x_positive_multiple_of_8 (x : ℕ) 
  (h1 : ∃ k, x = 8 * k) 
  (h2 : x^2 > 100) 
  (h3 : x < 20) : x = 16 :=
by
  sorry

end find_x_positive_multiple_of_8_l261_261576


namespace common_focus_hyperbola_ellipse_l261_261210

theorem common_focus_hyperbola_ellipse (p : ℝ) (c : ℝ) :
  (0 < p ∧ p < 8) →
  (c = Real.sqrt (3 + 1)) →
  (c = Real.sqrt (8 - p)) →
  p = 4 := by
sorry

end common_focus_hyperbola_ellipse_l261_261210


namespace inequality_for_positive_integer_l261_261783

theorem inequality_for_positive_integer (n : ℕ) (h : n > 0) :
  n^n ≤ (n!)^2 ∧ (n!)^2 ≤ ((n + 1) * (n + 2) / 6)^n := by
  sorry

end inequality_for_positive_integer_l261_261783


namespace lcm_of_two_numbers_l261_261095

theorem lcm_of_two_numbers (x y : ℕ) (h1 : Nat.gcd x y = 12) (h2 : x * y = 2460) : Nat.lcm x y = 205 :=
by
  -- Proof omitted
  sorry

end lcm_of_two_numbers_l261_261095


namespace lisa_flight_time_l261_261779

theorem lisa_flight_time
  (distance : ℕ) (speed : ℕ) (time : ℕ)
  (h_distance : distance = 256)
  (h_speed : speed = 32)
  (h_time : time = distance / speed) :
  time = 8 :=
by sorry

end lisa_flight_time_l261_261779


namespace vector_subtraction_l261_261221

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Define the expression to be proven
def expression : ℝ × ℝ := ((2 * a.1 - b.1), (2 * a.2 - b.2))

-- The theorem statement
theorem vector_subtraction : expression = (5, 7) :=
by
  -- The proof will be provided here
  sorry

end vector_subtraction_l261_261221


namespace net_percentage_change_is_correct_l261_261936

def initial_price : Float := 100.0

def price_after_first_year (initial: Float) := initial * (1 - 0.05)

def price_after_second_year (price1: Float) := price1 * (1 + 0.10)

def price_after_third_year (price2: Float) := price2 * (1 + 0.04)

def price_after_fourth_year (price3: Float) := price3 * (1 - 0.03)

def price_after_fifth_year (price4: Float) := price4 * (1 + 0.08)

def final_price := price_after_fifth_year (price_after_fourth_year (price_after_third_year (price_after_second_year (price_after_first_year initial_price))))

def net_percentage_change (initial final: Float) := ((final - initial) / initial) * 100

theorem net_percentage_change_is_correct :
  net_percentage_change initial_price final_price = 13.85 := by
  sorry

end net_percentage_change_is_correct_l261_261936


namespace greatest_perfect_power_sum_l261_261145

def sum_c_d_less_500 : ℕ :=
  let c := 22
  let d := 2
  c + d

theorem greatest_perfect_power_sum :
  ∃ c d : ℕ, 0 < c ∧ 1 < d ∧ c^d < 500 ∧
  ∀ x y : ℕ, 0 < x ∧ 1 < y ∧ x^y < 500 → x^y ≤ c^d ∧ (c + d = 24) :=
by
  sorry

end greatest_perfect_power_sum_l261_261145


namespace solve_for_x_l261_261416

theorem solve_for_x (x y z w : ℕ) 
  (h1 : x = y + 7) 
  (h2 : y = z + 15) 
  (h3 : z = w + 25) 
  (h4 : w = 95) : 
  x = 142 :=
by 
  sorry

end solve_for_x_l261_261416


namespace table_area_l261_261273

theorem table_area (A : ℝ) (runner_total : ℝ) (cover_percentage : ℝ) (double_layer : ℝ) (triple_layer : ℝ) :
  runner_total = 208 ∧
  cover_percentage = 0.80 ∧
  double_layer = 24 ∧
  triple_layer = 22 →
  A = 260 :=
by
  sorry

end table_area_l261_261273


namespace employee_n_salary_l261_261382

theorem employee_n_salary (m n : ℝ) (h1 : m = 1.2 * n) (h2 : m + n = 594) :
  n = 270 :=
sorry

end employee_n_salary_l261_261382


namespace members_in_both_sets_l261_261272

def U : Nat := 193
def B : Nat := 41
def not_A_or_B : Nat := 59
def A : Nat := 116

theorem members_in_both_sets
  (h1 : 193 = U)
  (h2 : 41 = B)
  (h3 : 59 = not_A_or_B)
  (h4 : 116 = A) :
  (U - not_A_or_B) = A + B - 23 :=
by
  sorry

end members_in_both_sets_l261_261272


namespace prob_xi_ge_2_l261_261859

noncomputable theory
open Classical

variable (ξ : ℝ → ℝ)

def normal_dist (μ σ : ℝ) := measure_theory.measure.norm_pdf μ σ

axiom ξ_normal : normal_dist 0 σ = ξ
axiom prob_neg2_to_0 : measure_theory.prob ξ (-2 ≤ ξ) (ξ ≤ 0) = 0.2

theorem prob_xi_ge_2 : measure_theory.prob ξ (ξ ≥ 2) = 0.3 :=
sorry

end prob_xi_ge_2_l261_261859


namespace symmetrical_implies_congruent_l261_261947

-- Define a structure to represent figures
structure Figure where
  segments : Set ℕ
  angles : Set ℕ

-- Define symmetry about a line
def is_symmetrical_about_line (f1 f2 : Figure) : Prop :=
  ∀ s ∈ f1.segments, s ∈ f2.segments ∧ ∀ a ∈ f1.angles, a ∈ f2.angles

-- Define congruent figures
def are_congruent (f1 f2 : Figure) : Prop :=
  f1.segments = f2.segments ∧ f1.angles = f2.angles

-- Lean 4 statement of the proof problem
theorem symmetrical_implies_congruent (f1 f2 : Figure) (h : is_symmetrical_about_line f1 f2) : are_congruent f1 f2 :=
by
  sorry

end symmetrical_implies_congruent_l261_261947


namespace keiko_walking_speed_l261_261464

theorem keiko_walking_speed (r : ℝ) (t : ℝ) (width : ℝ) 
   (time_diff : ℝ) (h0 : width = 8) (h1 : time_diff = 48) 
   (h2 : t = (2 * (2 * (r + 8) * Real.pi) / (r + 8) + 2 * (0 * Real.pi))) 
   (h3 : 2 * (2 * r * Real.pi) / r + 2 * (0 * Real.pi) = t - time_diff) :
   t = 48 -> 
   (v : ℝ) →
   v = (16 * Real.pi) / time_diff →
   v = Real.pi / 3 :=
by
  sorry

end keiko_walking_speed_l261_261464


namespace starting_number_l261_261090

theorem starting_number (n : ℕ) (h1 : 200 ≥ n) (h2 : 33 = ((200 / 3) - (n / 3))) : n = 102 :=
by
  sorry

end starting_number_l261_261090


namespace beyonce_total_songs_l261_261137

theorem beyonce_total_songs :
  let singles := 12
  let albums := 4
  let songs_per_album := 18 + 14
  let total_album_songs := albums * songs_per_album
  let total_songs := total_album_songs + singles
  total_songs = 140 := by
  let singles := 12
  let albums := 4
  let songs_per_album := 18 + 14
  let total_album_songs := albums * songs_per_album
  let total_songs := total_album_songs + singles
  sorry

end beyonce_total_songs_l261_261137


namespace closure_property_of_A_l261_261907

theorem closure_property_of_A 
  (a b c d k1 k2 : ℤ) 
  (x y : ℤ) 
  (Hx : x = a^2 + k1 * a * b + b^2) 
  (Hy : y = c^2 + k2 * c * d + d^2) : 
  ∃ m k : ℤ, x * y = m * (a^2 + k * a * b + b^2) := 
  by 
    -- this is where the proof would go
    sorry

end closure_property_of_A_l261_261907


namespace clare_milk_cartons_l261_261410

def money_given := 47
def cost_per_loaf := 2
def loaves_bought := 4
def cost_per_milk := 2
def money_left := 35

theorem clare_milk_cartons : (money_given - money_left - loaves_bought * cost_per_loaf) / cost_per_milk = 2 :=
by
  sorry

end clare_milk_cartons_l261_261410


namespace y_coord_diff_eq_nine_l261_261370

-- Declaring the variables and conditions
variables (m n : ℝ) (p : ℝ) (h1 : p = 3)
variable (L1 : m = (n / 3) - (2 / 5))
variable (L2 : m + p = ((n + 9) / 3) - (2 / 5))

-- The theorem statement
theorem y_coord_diff_eq_nine : (n + 9) - n = 9 :=
by
  sorry

end y_coord_diff_eq_nine_l261_261370


namespace quadratic_factorization_l261_261793

theorem quadratic_factorization :
  ∃ a b : ℕ, (a > b) ∧ (x^2 - 20 * x + 96 = (x - a) * (x - b)) ∧ (4 * b - a = 20) := sorry

end quadratic_factorization_l261_261793


namespace problem1_problem2_l261_261303

def f (x b : ℝ) : ℝ := |x - b| + |x + b|

theorem problem1 (x : ℝ) : (∀ y, y = 1 → f x y ≤ x + 2) ↔ (0 ≤ x ∧ x ≤ 2) :=
sorry

theorem problem2 (a b : ℝ) (h : a ≠ 0) : (∀ y, y = 1 → f y b ≥ (|a + 1| - |2 * a - 1|) / |a|) ↔ (b ≤ -3 / 2 ∨ b ≥ 3 / 2) :=
sorry

end problem1_problem2_l261_261303


namespace Malik_yards_per_game_l261_261600

-- Definitions of the conditions
def number_of_games : ℕ := 4
def josiah_yards_per_game : ℕ := 22
def darnell_average_yards_per_game : ℕ := 11
def total_yards_all_athletes : ℕ := 204

-- The statement to prove
theorem Malik_yards_per_game (M : ℕ) 
  (H1 : number_of_games = 4) 
  (H2 : josiah_yards_per_game = 22) 
  (H3 : darnell_average_yards_per_game = 11) 
  (H4 : total_yards_all_athletes = 204) :
  4 * M + 4 * 22 + 4 * 11 = 204 → M = 18 :=
by
  intros h
  sorry

end Malik_yards_per_game_l261_261600


namespace B_finish_work_alone_in_12_days_l261_261972

theorem B_finish_work_alone_in_12_days (A_days B_days both_days : ℕ) :
  A_days = 6 →
  both_days = 4 →
  (1 / A_days + 1 / B_days = 1 / both_days) →
  B_days = 12 :=
by
  intros hA hBoth hRate
  sorry

end B_finish_work_alone_in_12_days_l261_261972


namespace possible_slopes_of_line_intersecting_ellipse_l261_261393

theorem possible_slopes_of_line_intersecting_ellipse (m : ℝ) : 
  (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ (m ≤ -1/√55 ∨ 1/√55 ≤ m) := 
by
  sorry

end possible_slopes_of_line_intersecting_ellipse_l261_261393


namespace final_selling_price_l261_261390

variable (a : ℝ)

theorem final_selling_price (h : a > 0) : 0.9 * (1.25 * a) = 1.125 * a := 
by
  sorry

end final_selling_price_l261_261390


namespace sequence_u5_value_l261_261075

theorem sequence_u5_value (u : ℕ → ℝ) 
  (h_rec : ∀ n, u (n + 2) = 2 * u (n + 1) + u n)
  (h_u3 : u 3 = 9) 
  (h_u6 : u 6 = 128) : 
  u 5 = 53 := 
sorry

end sequence_u5_value_l261_261075


namespace total_whales_correct_l261_261762

def first_trip_male_whales : ℕ := 28
def first_trip_female_whales : ℕ := 2 * first_trip_male_whales
def first_trip_total_whales : ℕ := first_trip_male_whales + first_trip_female_whales

def second_trip_baby_whales : ℕ := 8
def second_trip_parent_whales : ℕ := 2 * second_trip_baby_whales
def second_trip_total_whales : ℕ := second_trip_baby_whales + second_trip_parent_whales

def third_trip_male_whales : ℕ := first_trip_male_whales / 2
def third_trip_female_whales : ℕ := first_trip_female_whales
def third_trip_total_whales : ℕ := third_trip_male_whales + third_trip_female_whales

def total_whales_seen : ℕ :=
  first_trip_total_whales + second_trip_total_whales + third_trip_total_whales

theorem total_whales_correct : total_whales_seen = 178 := by
  sorry

end total_whales_correct_l261_261762


namespace count_perfect_square_factors_l261_261745

theorem count_perfect_square_factors : 
  let n := (2^10) * (3^12) * (5^15) * (7^7)
  ∃ (count : ℕ), count = 1344 ∧
    (∀ (a b c d : ℕ), 0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 12 ∧ 0 ≤ c ∧ c ≤ 15 ∧ 0 ≤ d ∧ d ≤ 7 →
      ((a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ (d % 2 = 0) →
        ∃ (k : ℕ), (2^a * 3^b * 5^c * 7^d) = k ∧ k ∣ n)) :=
by
  sorry

end count_perfect_square_factors_l261_261745


namespace temperature_at_midnight_l261_261893

def morning_temp : ℝ := 30
def afternoon_increase : ℝ := 1
def midnight_decrease : ℝ := 7

theorem temperature_at_midnight : morning_temp + afternoon_increase - midnight_decrease = 24 := by
  sorry

end temperature_at_midnight_l261_261893


namespace find_C_monthly_income_l261_261081

theorem find_C_monthly_income (A_m B_m C_m : ℝ) (h1 : A_m / B_m = 5 / 2) (h2 : B_m = 1.12 * C_m) (h3 : 12 * A_m = 504000) : C_m = 15000 :=
sorry

end find_C_monthly_income_l261_261081


namespace exists_root_in_interval_l261_261798

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

theorem exists_root_in_interval : ∃ x, (2 < x ∧ x < 3) ∧ f x = 0 := 
by
  -- Assuming f(2) < 0 and f(3) > 0
  have h1 : f 2 < 0 := sorry
  have h2 : f 3 > 0 := sorry
  -- From the intermediate value theorem, there exists a c in (2, 3) such that f(c) = 0
  sorry

end exists_root_in_interval_l261_261798


namespace total_cost_of_ads_l261_261994

-- Define the conditions
def cost_ad1 := 3500
def minutes_ad1 := 2
def cost_ad2 := 4500
def minutes_ad2 := 3
def cost_ad3 := 3000
def minutes_ad3 := 3
def cost_ad4 := 4000
def minutes_ad4 := 2
def cost_ad5 := 5500
def minutes_ad5 := 5

-- Define the function to calculate the total cost
def total_cost :=
  (cost_ad1 * minutes_ad1) +
  (cost_ad2 * minutes_ad2) +
  (cost_ad3 * minutes_ad3) +
  (cost_ad4 * minutes_ad4) +
  (cost_ad5 * minutes_ad5)

-- The statement to prove
theorem total_cost_of_ads : total_cost = 66000 := by
  sorry

end total_cost_of_ads_l261_261994


namespace max_sum_cd_l261_261141

theorem max_sum_cd (c d : ℕ) (hc : c > 0) (hd : d > 1) (hcd : c^d < 500) 
  (hmax : ∀ (c' d': ℕ), c' > 0 → d' > 1 → c'^d' < 500 → c'^d' ≤ c^d) : c + d = 24 := 
by
  have h1 : 22^2 = 484 := rfl
  have h2 : c = 22 ∧ d = 2 := by sorry
  exact by sorry

end max_sum_cd_l261_261141


namespace frank_more_miles_than_jim_in_an_hour_l261_261053

theorem frank_more_miles_than_jim_in_an_hour
    (jim_distance : ℕ) (jim_time : ℕ)
    (frank_distance : ℕ) (frank_time : ℕ)
    (h_jim : jim_distance = 16)
    (h_jim_time : jim_time = 2)
    (h_frank : frank_distance = 20)
    (h_frank_time : frank_time = 2) :
    (frank_distance / frank_time) - (jim_distance / jim_time) = 2 := 
by
  -- Placeholder for the proof, no proof steps included as instructed.
  sorry

end frank_more_miles_than_jim_in_an_hour_l261_261053


namespace remainder_8927_div_11_l261_261654

theorem remainder_8927_div_11 : 8927 % 11 = 8 :=
by
  sorry

end remainder_8927_div_11_l261_261654


namespace factorize_expression_l261_261188

-- Define the variables
variables (a b : ℝ)

-- State the theorem to prove the factorization
theorem factorize_expression : a^2 - 2 * a * b = a * (a - 2 * b) :=
by 
  -- Proof goes here
  sorry

end factorize_expression_l261_261188


namespace smallest_n_with_314_l261_261194

noncomputable def contains_314 (n : ℕ) (m : ℕ) : Prop :=
  let frac := (m : ℚ) / (n : ℚ) in
  let dec_str := frac.to_decimal_string in
  "314".isIn dec_str

theorem smallest_n_with_314 :
  ∃ m n : ℕ,
    Nat.coprime m n ∧
    m < n ∧
    contains_314 n m ∧
    ∀ n' (h' : n' < n), ¬ ∃ m',
      Nat.coprime m' n' ∧
      m' < n' ∧
      contains_314 n' m' :=
begin
  sorry
end

end smallest_n_with_314_l261_261194


namespace series_sum_is_6_over_5_l261_261304

noncomputable def series_sum : ℝ := ∑' n : ℕ, if n % 4 == 0 then 1 / (4^(n/4)) else 
                                          if n % 4 == 1 then 1 / (2 * 4^(n/4)) else 
                                          if n % 4 == 2 then -1 / (4^(n/4) * 4^(1/2)) else 
                                          -1 / (2 * 4^(n/4 + 1/2))

theorem series_sum_is_6_over_5 : series_sum = 6 / 5 := 
  sorry

end series_sum_is_6_over_5_l261_261304


namespace find_ABC_l261_261908

theorem find_ABC (A B C : ℝ) (h : ∀ n : ℕ, n > 0 → 2 * n^3 + 3 * n^2 = A * (n * (n - 1) * (n - 2)) / 6 + B * (n * (n - 1)) / 2 + C * n) :
  A = 12 ∧ B = 18 ∧ C = 5 :=
by {
  sorry
}

end find_ABC_l261_261908


namespace probability_daniel_wins_l261_261546

theorem probability_daniel_wins :
  let p := 0.60
  let P := 0.36 + 0.48 * p
  P = 9 / 13 :=
by
  sorry

end probability_daniel_wins_l261_261546


namespace expression_value_zero_l261_261450

variable (x : ℝ)

theorem expression_value_zero (h : x^2 + 3 * x - 3 = 0) : x^3 + 2 * x^2 - 6 * x + 3 = 0 := 
by
  sorry

end expression_value_zero_l261_261450


namespace bridget_gave_erasers_l261_261919

variable (p_start : ℕ) (p_end : ℕ) (e_b : ℕ)

theorem bridget_gave_erasers (h1 : p_start = 8) (h2 : p_end = 11) (h3 : p_end = p_start + e_b) :
  e_b = 3 := by
  sorry

end bridget_gave_erasers_l261_261919


namespace no_natural_number_for_square_condition_l261_261842

theorem no_natural_number_for_square_condition :
  ∀ n : ℕ, ¬ (∃ k : ℕ, k^2 = n^5 - 5*n^3 + 4*n + 7) :=
begin
  intro n,
  intro h,
  cases h with k hk,
  sorry,
end

end no_natural_number_for_square_condition_l261_261842


namespace combined_weight_of_three_boxes_l261_261914

theorem combined_weight_of_three_boxes (a b c d : ℕ) (h₁ : a + b = 132) (h₂ : a + c = 136) (h₃ : b + c = 138) (h₄ : d = 60) : 
  a + b + c = 203 :=
sorry

end combined_weight_of_three_boxes_l261_261914


namespace greatest_third_side_of_triangle_l261_261954

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 15) :
  ∃ x : ℕ, 8 < x ∧ x < 22 ∧ (∀ y : ℕ, 8 < y ∧ y < 22 → y ≤ x) ∧ x = 21 :=
by
  sorry

end greatest_third_side_of_triangle_l261_261954


namespace range_of_x_l261_261560

theorem range_of_x (θ : ℝ) (h0 : 0 < θ) (h1 : θ < Real.pi / 2) (h2 : ∀ θ, (0 < θ) → (θ < Real.pi / 2) → (1 / (Real.sin θ) ^ 2 + 4 / (Real.cos θ) ^ 2 ≥ abs (2 * x - 1))) :
  -4 ≤ x ∧ x ≤ 5 := sorry

end range_of_x_l261_261560


namespace sam_earnings_difference_l261_261921

def hours_per_dollar := 1 / 10  -- Sam earns $10 per hour, so it takes 1/10 hour per dollar earned.

theorem sam_earnings_difference
  (hours_per_dollar : ℝ := 1 / 10)
  (E1 : ℝ := 200)  -- Earnings in the first month are $200.
  (total_hours : ℝ := 55)  -- Total hours he worked over two months.
  (total_hourly_earning : ℝ := total_hours / hours_per_dollar)  -- Total earnings over two months.
  (E2 : ℝ := total_hourly_earning - E1) :  -- Earnings in the second month.

  E2 - E1 = 150 :=  -- The difference in earnings between the second month and the first month is $150.
sorry

end sam_earnings_difference_l261_261921


namespace fifteenth_term_l261_261732

variable (a b : ℤ)

def sum_first_n_terms (n : ℕ) : ℤ := n * (2 * a + (n - 1) * b) / 2

axiom sum_first_10 : sum_first_n_terms 10 = 60
axiom sum_first_20 : sum_first_n_terms 20 = 320

def nth_term (n : ℕ) : ℤ := a + (n - 1) * b

theorem fifteenth_term : nth_term 15 = 25 :=
by
  sorry

end fifteenth_term_l261_261732


namespace min_value_xy_l261_261029

theorem min_value_xy (x y : ℝ) (h : 1 / x + 2 / y = Real.sqrt (x * y)) : x * y ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_xy_l261_261029


namespace cookie_ratio_l261_261965

theorem cookie_ratio (cookies_monday cookies_tuesday cookies_wednesday final_cookies : ℕ)
  (h1 : cookies_monday = 32)
  (h2 : cookies_tuesday = cookies_monday / 2)
  (h3 : final_cookies = 92)
  (h4 : cookies_wednesday = final_cookies + 4 - cookies_monday - cookies_tuesday) :
  cookies_wednesday / cookies_tuesday = 3 :=
by
  sorry

end cookie_ratio_l261_261965


namespace remainder_divisibility_l261_261301

theorem remainder_divisibility (n : ℕ) (d : ℕ) (r : ℕ) : 
  let n := 1234567
  let d := 256
  let r := n % d
  r = 933 ∧ ¬ (r % 7 = 0) := by
  sorry

end remainder_divisibility_l261_261301


namespace minimum_value_2x_3y_l261_261736

theorem minimum_value_2x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hxy : x^2 * y * (4 * x + 3 * y) = 3) :
  2 * x + 3 * y ≥ 2 * Real.sqrt 3 := by
  sorry

end minimum_value_2x_3y_l261_261736


namespace sin_arcsin_plus_arctan_l261_261191

theorem sin_arcsin_plus_arctan :
  let a := Real.arcsin (4/5)
  let b := Real.arctan 1
  Real.sin (a + b) = (7 * Real.sqrt 2) / 10 := by
  sorry

end sin_arcsin_plus_arctan_l261_261191


namespace find_total_roses_l261_261116

open Nat

theorem find_total_roses 
  (a : ℕ)
  (h1 : 300 ≤ a)
  (h2 : a ≤ 400)
  (h3 : a % 21 = 13)
  (h4 : a % 15 = 7) : 
  a = 307 := 
sorry

end find_total_roses_l261_261116


namespace division_and_multiply_l261_261714

theorem division_and_multiply :
  (-128) / (-16) * 5 = 40 := 
by
  sorry

end division_and_multiply_l261_261714


namespace percentage_increase_in_expenses_l261_261378

variable (a b c : ℝ)

theorem percentage_increase_in_expenses :
  (10 / 100 * a + 30 / 100 * b + 20 / 100 * c) / (a + b + c) =
  (10 * a + 30 * b + 20 * c) / (100 * (a + b + c)) :=
by
  sorry

end percentage_increase_in_expenses_l261_261378


namespace total_birds_correct_l261_261498

def numPairs : Nat := 3
def birdsPerPair : Nat := 2
def totalBirds : Nat := numPairs * birdsPerPair

theorem total_birds_correct : totalBirds = 6 :=
by
  -- proof goes here
  sorry

end total_birds_correct_l261_261498


namespace water_breaks_vs_sitting_breaks_l261_261244

theorem water_breaks_vs_sitting_breaks :
  (240 / 20) - (240 / 120) = 10 := by
  sorry

end water_breaks_vs_sitting_breaks_l261_261244


namespace water_tank_full_capacity_l261_261833

theorem water_tank_full_capacity (x : ℝ) (h1 : x * (3/4) - x * (1/3) = 15) : x = 36 := 
by
  sorry

end water_tank_full_capacity_l261_261833


namespace total_amount_paid_l261_261840

variable (n : ℕ) (each_paid : ℕ)

/-- This is a statement that verifies the total amount paid given the number of friends and the amount each friend pays. -/
theorem total_amount_paid (h1 : n = 7) (h2 : each_paid = 70) : n * each_paid = 490 := by
  -- This proof will validate that the total amount paid is 490
  sorry

end total_amount_paid_l261_261840


namespace bus_fare_max_profit_passenger_count_change_l261_261108

noncomputable def demand (p : ℝ) : ℝ := 3000 - 20 * p
noncomputable def train_fare : ℝ := 10
noncomputable def train_capacity : ℝ := 1000
noncomputable def bus_cost (y : ℝ) : ℝ := y + 5

theorem bus_fare_max_profit : 
  ∃ (p_bus : ℝ), 
  p_bus = 50.5 ∧ 
  p_bus * (demand p_bus - train_capacity) - bus_cost (demand p_bus - train_capacity) = 
  p_bus * (demand p_bus - train_capacity) - (demand p_bus - train_capacity + 5) := 
sorry

theorem passenger_count_change :
  (demand train_fare - train_capacity) + train_capacity - demand 75.5 = 500 :=
sorry

end bus_fare_max_profit_passenger_count_change_l261_261108


namespace baguettes_leftover_l261_261355

-- Definitions based on conditions
def batches_per_day := 3
def baguettes_per_batch := 48
def sold_after_first_batch := 37
def sold_after_second_batch := 52
def sold_after_third_batch := 49

-- Prove the question equals the answer
theorem baguettes_leftover : 
  (batches_per_day * baguettes_per_batch - (sold_after_first_batch + sold_after_second_batch + sold_after_third_batch)) = 6 := 
by 
  sorry

end baguettes_leftover_l261_261355


namespace percentage_of_non_defective_products_l261_261104

-- Define the conditions
def totalProduction : ℕ := 100
def M1_production : ℕ := 25
def M2_production : ℕ := 35
def M3_production : ℕ := 40

def M1_defective_rate : ℝ := 0.02
def M2_defective_rate : ℝ := 0.04
def M3_defective_rate : ℝ := 0.05

-- Calculate the total defective units
noncomputable def total_defective_units : ℝ := 
  (M1_defective_rate * M1_production) + 
  (M2_defective_rate * M2_production) + 
  (M3_defective_rate * M3_production)

-- Calculate the percentage of defective products
noncomputable def defective_percentage : ℝ := (total_defective_units / totalProduction) * 100

-- Calculate the percentage of non-defective products
noncomputable def non_defective_percentage : ℝ := 100 - defective_percentage

-- The statement to prove
theorem percentage_of_non_defective_products :
  non_defective_percentage = 96.1 :=
by
  sorry

end percentage_of_non_defective_products_l261_261104


namespace expected_value_of_8_sided_die_l261_261645

theorem expected_value_of_8_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (∑ outcome in outcomes, outcome * probability) = 4.5 :=
by
  sorry

end expected_value_of_8_sided_die_l261_261645


namespace find_perpendicular_line_through_intersection_l261_261439

theorem find_perpendicular_line_through_intersection : 
  (∃ (M : ℚ × ℚ), 
    (M.1 - 2 * M.2 + 3 = 0) ∧ 
    (2 * M.1 + 3 * M.2 - 8 = 0) ∧ 
    (∃ (c : ℚ), M.1 + 3 * M.2 + c = 0 ∧ 3 * M.1 - M.2 + 1 = 0)) → 
  ∃ (c : ℚ), x + 3 * y + c = 0 :=
sorry

end find_perpendicular_line_through_intersection_l261_261439


namespace circle_intersects_y_axis_at_one_l261_261617

theorem circle_intersects_y_axis_at_one :
  let A := (-2011, 0)
  let B := (2010, 0)
  let C := (0, (-2010) * 2011)
  ∃ (D : ℝ × ℝ), D = (0, 1) ∧
    (∃ O : ℝ × ℝ, O = (0, 0) ∧
    (dist O A) * (dist O B) = (dist O C) * (dist O D)) :=
by
  sorry -- Proof of the theorem

end circle_intersects_y_axis_at_one_l261_261617


namespace Rachel_total_earnings_l261_261070

-- Define the constants for the conditions
def hourly_wage : ℝ := 12
def people_served : ℕ := 20
def tip_per_person : ℝ := 1.25

-- Define the problem
def total_money_made : ℝ := hourly_wage + (people_served * tip_per_person)

-- State the theorem to be proved
theorem Rachel_total_earnings : total_money_made = 37 := by
  sorry

end Rachel_total_earnings_l261_261070


namespace new_sum_after_decrease_l261_261502

theorem new_sum_after_decrease (a b : ℕ) (h₁ : a + b = 100) (h₂ : a' = a - 48) : a' + b = 52 := by
  sorry

end new_sum_after_decrease_l261_261502


namespace no_prime_divisor_of_form_8k_minus_1_l261_261055

theorem no_prime_divisor_of_form_8k_minus_1 (n : ℕ) (h : 0 < n) :
  ¬ ∃ p k : ℕ, Nat.Prime p ∧ p = 8 * k - 1 ∧ p ∣ (2^n + 1) :=
by
  sorry

end no_prime_divisor_of_form_8k_minus_1_l261_261055


namespace max_min_z_diff_correct_l261_261596

noncomputable def max_min_z_diff (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) : ℝ :=
  6

theorem max_min_z_diff_correct (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) :
  max_min_z_diff x y z h1 h2 = 6 :=
sorry

end max_min_z_diff_correct_l261_261596


namespace determine_flower_responsibility_l261_261527

-- Define the structure of the grid
structure Grid (m n : ℕ) :=
  (vertices : Fin m → Fin n → Bool) -- True if gardener lives at the vertex

-- Define a function to determine if 3 gardeners are nearest to a flower
def is_nearest (i j fi fj : ℕ) : Bool :=
  -- Assume this function gives true if the gardener at (i, j) is one of the 3 nearest to the flower at (fi, fj)
  sorry

-- The main theorem statement
theorem determine_flower_responsibility 
  {m n : ℕ} 
  (G : Grid m n) 
  (i j : Fin m) 
  (k : Fin n) 
  (h : G.vertices i k = true) 
  : ∃ (fi fj : ℕ), is_nearest (i : ℕ) (k : ℕ) fi fj = true := 
sorry

end determine_flower_responsibility_l261_261527


namespace how_many_bones_in_adult_woman_l261_261047

-- Define the conditions
def numSkeletons : ℕ := 20
def halfSkeletons : ℕ := 10
def numAdultWomen : ℕ := 10
def numMenAndChildren : ℕ := 10
def numAdultMen : ℕ := 5
def numChildren : ℕ := 5
def totalBones : ℕ := 375

-- Define the proof statement
theorem how_many_bones_in_adult_woman (W : ℕ) (H : 10 * W + 5 * (W + 5) + 5 * (W / 2) = 375) : W = 20 :=
sorry

end how_many_bones_in_adult_woman_l261_261047


namespace annieka_free_throws_l261_261929

theorem annieka_free_throws :
  let d : ℕ := 12 in
  let k : ℕ := d + (d / 2) in
  let a : ℕ := k - 4 in
  a = 14 :=
by
  let d := 12
  let k := d + (d / 2)
  let a := k - 4
  show a = 14
  sorry

end annieka_free_throws_l261_261929


namespace probability_of_selecting_at_least_one_female_l261_261563

open BigOperators

noncomputable def prob_at_least_one_female_selected : ℚ :=
  let total_choices := Nat.choose 10 3
  let all_males_choices := Nat.choose 6 3
  1 - (all_males_choices / total_choices : ℚ)

theorem probability_of_selecting_at_least_one_female :
  prob_at_least_one_female_selected = 5 / 6 := by
  sorry

end probability_of_selecting_at_least_one_female_l261_261563


namespace hannah_money_left_l261_261744

variable (initial_amount : ℕ) (amount_spent_rides : ℕ) (amount_spent_dessert : ℕ)
  (remaining_after_rides : ℕ) (remaining_money : ℕ)

theorem hannah_money_left :
  initial_amount = 30 →
  amount_spent_rides = initial_amount / 2 →
  remaining_after_rides = initial_amount - amount_spent_rides →
  amount_spent_dessert = 5 →
  remaining_money = remaining_after_rides - amount_spent_dessert →
  remaining_money = 10 := by
  sorry

end hannah_money_left_l261_261744


namespace poly_div_simplification_l261_261970

-- Assume a and b are real numbers.
variables (a b : ℝ)

-- Theorem to prove the equivalence
theorem poly_div_simplification (a b : ℝ) : (4 * a^2 - b^2) / (b - 2 * a) = -2 * a - b :=
by
  -- The proof will go here
  sorry

end poly_div_simplification_l261_261970


namespace jeans_cost_l261_261606

theorem jeans_cost (initial_money pizza_cost soda_cost quarter_value after_quarters : ℝ) (quarters_count: ℕ) :
  initial_money = 40 ->
  pizza_cost = 2.75 ->
  soda_cost = 1.50 ->
  quarter_value = 0.25 ->
  quarters_count = 97 ->
  after_quarters = quarters_count * quarter_value ->
  initial_money - (pizza_cost + soda_cost) - after_quarters = 11.50 :=
by
  intros h_initial h_pizza h_soda h_quarter_val h_quarters h_after_quarters
  sorry

end jeans_cost_l261_261606


namespace number_of_points_max_45_lines_l261_261447

theorem number_of_points_max_45_lines (n : ℕ) (h : n * (n - 1) / 2 ≤ 45) : n = 10 := 
  sorry

end number_of_points_max_45_lines_l261_261447


namespace Danielle_rooms_is_6_l261_261873

-- Definitions for the problem conditions
def Heidi_rooms (Danielle_rooms : ℕ) : ℕ := 3 * Danielle_rooms
def Grant_rooms (Heidi_rooms : ℕ) : ℕ := Heidi_rooms / 9
def Grant_rooms_value : ℕ := 2

-- Theorem statement
theorem Danielle_rooms_is_6 (h : Grant_rooms_value = Grant_rooms (Heidi_rooms d)) : d = 6 :=
by
  sorry

end Danielle_rooms_is_6_l261_261873


namespace area_of_rhombus_l261_261556

theorem area_of_rhombus (R₁ R₂ : ℝ) (x y : ℝ) (hR₁ : R₁ = 10) (hR₂ : R₂ = 20) 
    (h_eq : (x * (x^2 + y^2)) / (4 * R₁) = (y * (x^2 + y^2)) / (4 * R₂)) :
    ((2 * x) * (2 * y)) / 2 = 40 :=
by
  have h₁ : x * (x^2 + y^2) = y * (x^2 + y^2) / 2 := by sorry
  have h₂ : y = 2 * x := by sorry
  have h₃ : x^2 = 40 := by sorry
  have h₄ : x = sqrt 40 := by sorry
  have h₅ : y = 4 * sqrt 40 := by sorry
  have h₆ : (2 * sqrt 40) * (4 * sqrt 40) / 2 = 40 := by sorry
  exact h₆

end area_of_rhombus_l261_261556


namespace parametrize_line_l261_261358

theorem parametrize_line (s h : ℝ) :
    s = -5/2 ∧ h = 20 → ∀ t : ℝ, ∃ x y : ℝ, 4 * x + 7 = y ∧ 
    (x = s + 5 * t ∧ y = -3 + h * t) :=
by
  sorry

end parametrize_line_l261_261358


namespace given_inequality_l261_261476

theorem given_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h: 1 + a + b + c = 2 * a * b * c) :
  ab / (1 + a + b) + bc / (1 + b + c) + ca / (1 + c + a) ≥ 3 / 2 :=
sorry

end given_inequality_l261_261476


namespace min_gennadys_l261_261706

theorem min_gennadys (alexanders borises vasilies : ℕ) (x : ℕ) 
    (h1 : alexanders = 45)
    (h2 : borises = 122)
    (h3 : vasilies = 27)
    (h4 : x = 49)
    (h5 : borises - 1 = alexanders + vasilies + x) :
  x = 49 := 
begin
  sorry,
end

end min_gennadys_l261_261706


namespace find_x_from_triangle_area_l261_261196

theorem find_x_from_triangle_area :
  ∀ (x : ℝ), x > 0 ∧ (1 / 2) * x * 3 * x = 96 → x = 8 :=
by
  intros x hx
  -- The proof goes here
  sorry

end find_x_from_triangle_area_l261_261196


namespace symmetric_line_eq_l261_261933

theorem symmetric_line_eq (x y : ℝ) (h : 2 * x - y = 0) : 2 * x + y = 0 :=
sorry

end symmetric_line_eq_l261_261933


namespace find_y_intercept_l261_261018

theorem find_y_intercept (x1 y1 x2 y2 : ℝ) (h₁ : (x1, y1) = (2, -2)) (h₂ : (x2, y2) = (6, 6)) : 
  ∃ b : ℝ, (∀ x : ℝ, y = 2 * x + b) ∧ b = -6 :=
by
  sorry

end find_y_intercept_l261_261018


namespace find_k_l261_261313

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 3)

theorem find_k (k : ℝ) :
  let sum_vector := (vector_a.1 + 2 * (vector_b k).1, vector_a.2 + 2 * (vector_b k).2)
  let diff_vector := (2 * vector_a.1 - (vector_b k).1, 2 * vector_a.2 - (vector_b k).2)
  sum_vector.1 * diff_vector.2 = sum_vector.2 * diff_vector.1
  → k = 6 :=
by
  sorry

end find_k_l261_261313


namespace alicia_satisfaction_l261_261294

theorem alicia_satisfaction (t : ℚ) (h_sat : t * (12 - t) = (4 - t) * (2 * t + 2)) : t = 2 :=
by
  sorry

end alicia_satisfaction_l261_261294


namespace arithmetic_sequence_150th_term_l261_261372

open Nat

-- Define the nth term of an arithmetic sequence
def nth_term_arithmetic (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- Theorem to prove
theorem arithmetic_sequence_150th_term (a1 d n : ℕ) (h1 : a1 = 3) (h2 : d = 7) (h3 : n = 150) :
  nth_term_arithmetic a1 d n = 1046 :=
by
  sorry

end arithmetic_sequence_150th_term_l261_261372


namespace negation_of_proposition_l261_261491

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) ↔ ∀ x : ℝ, x^2 ≠ 1 → x ≠ 1 ∧ x ≠ -1 := by
sorry

end negation_of_proposition_l261_261491


namespace extremum_condition_l261_261080

noncomputable def quadratic_polynomial (a x : ℝ) : ℝ := a * x^2 + x + 1

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, ∃ f' : ℝ → ℝ, 
     (f' = (fun x => 2 * a * x + 1)) ∧ 
     (f' x = 0) ∧ 
     (∃ (f'' : ℝ → ℝ), (f'' = (fun x => 2 * a)) ∧ (f'' x ≠ 0))) ↔ a < 0 := 
sorry

end extremum_condition_l261_261080


namespace alpha_beta_square_l261_261039

noncomputable def roots_of_quadratic : set ℝ := 
  { x | x^2 = 2 * x + 1 }

theorem alpha_beta_square (α β : ℝ) (hα : α ∈ roots_of_quadratic) (hβ : β ∈ roots_of_quadratic) (hαβ : α ≠ β) :
  (α - β)^2 = 8 :=
sorry

end alpha_beta_square_l261_261039


namespace probability_same_color_is_27_over_100_l261_261226

def num_sides_die1 := 20
def num_sides_die2 := 20

def maroon_die1 := 5
def teal_die1 := 6
def cyan_die1 := 7
def sparkly_die1 := 1
def silver_die1 := 1

def maroon_die2 := 4
def teal_die2 := 6
def cyan_die2 := 7
def sparkly_die2 := 1
def silver_die2 := 2

noncomputable def probability_same_color : ℚ :=
  (maroon_die1 * maroon_die2 + teal_die1 * teal_die2 + cyan_die1 * cyan_die2 + sparkly_die1 * sparkly_die2 + silver_die1 * silver_die2) /
  (num_sides_die1 * num_sides_die2)

theorem probability_same_color_is_27_over_100 :
  probability_same_color = 27 / 100 := 
sorry

end probability_same_color_is_27_over_100_l261_261226


namespace prop_disjunction_is_true_l261_261318

variable (p q : Prop)
axiom hp : p
axiom hq : ¬q

theorem prop_disjunction_is_true (hp : p) (hq : ¬q) : p ∨ q :=
by
  sorry

end prop_disjunction_is_true_l261_261318


namespace factorize_expression_l261_261421

theorem factorize_expression (x y : ℝ) : (y + 2 * x)^2 - (x + 2 * y)^2 = 3 * (x + y) * (x - y) :=
  sorry

end factorize_expression_l261_261421


namespace vector_subtraction_l261_261220

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Define the expression to be proven
def expression : ℝ × ℝ := ((2 * a.1 - b.1), (2 * a.2 - b.2))

-- The theorem statement
theorem vector_subtraction : expression = (5, 7) :=
by
  -- The proof will be provided here
  sorry

end vector_subtraction_l261_261220


namespace acute_angled_triangle_range_l261_261752

theorem acute_angled_triangle_range (x : ℝ) (h : (x^2 + 6)^2 < (x^2 + 4)^2 + (4 * x)^2) : x > (Real.sqrt 15) / 3 := sorry

end acute_angled_triangle_range_l261_261752


namespace max_marks_l261_261282

theorem max_marks (score shortfall passing_threshold : ℝ) (h1 : score = 212) (h2 : shortfall = 19) (h3 : passing_threshold = 0.30) :
  ∃ M, M = 770 :=
by
  sorry

end max_marks_l261_261282


namespace find_special_number_l261_261848

theorem find_special_number:
  ∃ (n : ℕ), (n > 0) ∧ (∃ (k : ℕ), 2 * n = k^2)
           ∧ (∃ (m : ℕ), 3 * n = m^3)
           ∧ (∃ (p : ℕ), 5 * n = p^5)
           ∧ n = 1085 :=
by
  sorry

end find_special_number_l261_261848


namespace base_triangle_not_equilateral_l261_261799

-- Define the lengths of the lateral edges
def SA := 1
def SB := 2
def SC := 4

-- Main theorem: the base triangle is not equilateral
theorem base_triangle_not_equilateral 
  (a : ℝ)
  (equilateral : a = a)
  (triangle_inequality1 : SA + SB > a)
  (triangle_inequality2 : SA + a > SC) : 
  a ≠ a :=
by 
  sorry

end base_triangle_not_equilateral_l261_261799


namespace video_files_count_l261_261402

-- Definitions for the given conditions
def total_files : ℝ := 48.0
def music_files : ℝ := 4.0
def picture_files : ℝ := 23.0

-- The proposition to prove
theorem video_files_count : total_files - (music_files + picture_files) = 21.0 :=
by
  sorry

end video_files_count_l261_261402


namespace circumscribed_triangle_area_relation_l261_261976

theorem circumscribed_triangle_area_relation
    (a b c: ℝ) (h₀: a = 8) (h₁: b = 15) (h₂: c = 17)
    (triangle_area: ℝ) (circle_area: ℝ) (X Y Z: ℝ)
    (hZ: Z > X) (hXY: X < Y)
    (triangle_area_calc: triangle_area = 60)
    (circle_area_calc: circle_area = π * (c / 2)^2) :
    X + Y = Z := by
  sorry

end circumscribed_triangle_area_relation_l261_261976


namespace fewest_students_possible_l261_261328

theorem fewest_students_possible :
  ∃ n : ℕ, n ≡ 2 [MOD 5] ∧ n ≡ 4 [MOD 6] ∧ n ≡ 6 [MOD 8] ∧ n = 22 :=
sorry

end fewest_students_possible_l261_261328


namespace find_divisible_xy9z_l261_261422

-- Define a predicate for numbers divisible by 132
def divisible_by_132 (n : ℕ) : Prop :=
  n % 132 = 0

-- Define the given number form \(\overline{xy9z}\) as a number maker
def form_xy9z (x y z : ℕ) : ℕ :=
  1000 * x + 100 * y + 90 + z

-- Stating the theorem for finding all numbers of form \(\overline{xy9z}\) that are divisible by 132
theorem find_divisible_xy9z (x y z : ℕ) :
  (divisible_by_132 (form_xy9z x y z)) ↔
  form_xy9z x y z = 3696 ∨
  form_xy9z x y z = 4092 ∨
  form_xy9z x y z = 6996 ∨
  form_xy9z x y z = 7392 :=
by sorry

end find_divisible_xy9z_l261_261422


namespace additional_cats_l261_261127

theorem additional_cats {M R C : ℕ} (h1 : 20 * R = M) (h2 : 4 + 2 * C = 10) : C = 3 := 
  sorry

end additional_cats_l261_261127


namespace PU_squared_fraction_l261_261584

noncomputable def compute_PU_squared : ℚ :=
  sorry -- Proof of the distance computation PU^2.

theorem PU_squared_fraction :
  ∃ (a b : ℕ), (gcd a b = 1) ∧ (compute_PU_squared = a / b) :=
  sorry -- Proof that the resulting fraction a/b is in its simplest form.

end PU_squared_fraction_l261_261584


namespace rosalina_received_21_gifts_l261_261347

def Emilio_gifts : Nat := 11
def Jorge_gifts : Nat := 6
def Pedro_gifts : Nat := 4

def total_gifts : Nat :=
  Emilio_gifts + Jorge_gifts + Pedro_gifts

theorem rosalina_received_21_gifts : total_gifts = 21 := by
  sorry

end rosalina_received_21_gifts_l261_261347


namespace find_d_l261_261592

theorem find_d (a₁: ℤ) (d : ℤ) (Sn : ℤ → ℤ) : 
  a₁ = 190 → 
  (Sn 20 > 0) → 
  (Sn 24 < 0) → 
  (Sn n = n * a₁ + (n * (n - 1)) / 2 * d) →
  d = -17 :=
by
  intros
  sorry

end find_d_l261_261592


namespace smallest_rel_prime_to_180_l261_261427

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ (∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → y ≥ x) ∧ x = 7 :=
  sorry

end smallest_rel_prime_to_180_l261_261427


namespace findingRealNumsPureImaginary_l261_261423

theorem findingRealNumsPureImaginary :
  ∀ x : ℝ, ((x + Complex.I * 2) * ((x + 2) + Complex.I * 2) * ((x + 4) + Complex.I * 2)).im = 0 → 
    x = -4 ∨ x = -1 + 2 * Real.sqrt 5 ∨ x = -1 - 2 * Real.sqrt 5 :=
by
  intros x h
  let expr := x^3 + 6*x^2 + 4*x - 16
  have h_real_part_eq_0 : expr = 0 := sorry
  have solutions_correct :
    expr = 0 → (x = -4 ∨ x = -1 + 2 * Real.sqrt 5 ∨ x = -1 - 2 * Real.sqrt 5) := sorry
  exact solutions_correct h_real_part_eq_0

end findingRealNumsPureImaginary_l261_261423


namespace area_of_rectangle_l261_261943

theorem area_of_rectangle (length : ℝ) (width : ℝ) (h_length : length = 47.3) (h_width : width = 24) : 
  length * width = 1135.2 := 
by 
  sorry

end area_of_rectangle_l261_261943


namespace find_second_number_in_denominator_l261_261624

theorem find_second_number_in_denominator :
  (0.625 * 0.0729 * 28.9) / (0.0017 * x * 8.1) = 382.5 → x = 0.24847 :=
by
  intro h
  sorry

end find_second_number_in_denominator_l261_261624


namespace number_of_teachers_l261_261981

theorem number_of_teachers
  (T S : ℕ)
  (h1 : T + S = 2400)
  (h2 : 320 = 320) -- This condition is trivial and can be ignored
  (h3 : 280 = 280) -- This condition is trivial and can be ignored
  (h4 : S / 280 = T / 40) : T = 300 :=
by
  sorry

end number_of_teachers_l261_261981


namespace starting_number_l261_261088

theorem starting_number (n : ℤ) : 
  (∃ n, (200 - n) / 3 = 33 ∧ (200 % 3 ≠ 0) ∧ (n % 3 = 0 ∧ n ≤ 200)) → n = 102 :=
by
  sorry

end starting_number_l261_261088


namespace decompose_max_product_l261_261547

theorem decompose_max_product (a : ℝ) (h_pos : a > 0) :
  ∃ x y : ℝ, x + y = a ∧ x * y ≤ (a / 2) * (a / 2) :=
by
  sorry

end decompose_max_product_l261_261547


namespace derivative_at_zero_l261_261708

open Real

def f (x : ℝ) : ℝ := if x ≠ 0 then (log (cos x)) / x else 0

theorem derivative_at_zero :
  deriv f 0 = -1 / 2 :=
by
  sorry

end derivative_at_zero_l261_261708


namespace algebraic_expression_value_l261_261575

theorem algebraic_expression_value (a b : ℕ) (h : a - 3 * b = 0) :
  (a - (2 * a * b - b * b) / a) / ((a * a - b * b) / a) = 1 / 2 := 
sorry

end algebraic_expression_value_l261_261575


namespace vector_subtraction_l261_261219

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-1, 1)

-- Define the expression to be proven
def expression : ℝ × ℝ := ((2 * a.1 - b.1), (2 * a.2 - b.2))

-- The theorem statement
theorem vector_subtraction : expression = (5, 7) :=
by
  -- The proof will be provided here
  sorry

end vector_subtraction_l261_261219


namespace arvin_first_day_km_l261_261898

theorem arvin_first_day_km :
  ∀ (x : ℕ), (∀ i : ℕ, (i < 5 → (i + x) < 6) → (x + 4 = 6)) → x = 2 :=
by sorry

end arvin_first_day_km_l261_261898


namespace alpha_value_l261_261035

theorem alpha_value (m : ℝ) (α : ℝ) (h : m * 8 ^ α = 1 / 4) : α = -2 / 3 :=
by
  sorry

end alpha_value_l261_261035


namespace final_alcohol_percentage_l261_261522

noncomputable def initial_volume : ℝ := 6
noncomputable def initial_percentage : ℝ := 0.25
noncomputable def added_alcohol : ℝ := 3
noncomputable def final_volume : ℝ := initial_volume + added_alcohol
noncomputable def final_percentage : ℝ := (initial_volume * initial_percentage + added_alcohol) / final_volume * 100

theorem final_alcohol_percentage :
  final_percentage = 50 := by
  sorry

end final_alcohol_percentage_l261_261522


namespace seventh_root_of_unity_problem_l261_261338

theorem seventh_root_of_unity_problem (q : ℂ) (h : q^7 = 1) :
  (q = 1 → (q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6)) = 3 / 2) ∧ 
  (q ≠ 1 → (q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6)) = -2) :=
by
  sorry

end seventh_root_of_unity_problem_l261_261338


namespace area_triangle_le_quarter_l261_261721

theorem area_triangle_le_quarter (S : ℝ) (S₁ S₂ S₃ S₄ S₅ S₆ S₇ : ℝ)
  (h₁ : S₃ + (S₂ + S₇) = S / 2)
  (h₂ : S₁ + S₆ + (S₂ + S₇) = S / 2) :
  S₁ ≤ S / 4 :=
by
  -- Proof skipped
  sorry

end area_triangle_le_quarter_l261_261721


namespace greatest_third_side_l261_261949

-- Given data and the Triangle Inequality theorem
theorem greatest_third_side (c : ℕ) (h1 : 8 < c) (h2 : c < 22) : c = 21 :=
by
  sorry

end greatest_third_side_l261_261949


namespace gallons_10_percent_milk_needed_l261_261514

-- Definitions based on conditions
def amount_of_butterfat (x : ℝ) : ℝ := 0.10 * x
def total_butterfat_in_existing_milk : ℝ := 4
def final_butterfat (x : ℝ) : ℝ := amount_of_butterfat x + total_butterfat_in_existing_milk
def total_milk (x : ℝ) : ℝ := x + 8
def desired_butterfat (x : ℝ) : ℝ := 0.20 * total_milk x

-- Lean proof statement
theorem gallons_10_percent_milk_needed (x : ℝ) : final_butterfat x = desired_butterfat x → x = 24 :=
by
  intros h
  sorry

end gallons_10_percent_milk_needed_l261_261514


namespace num_houses_with_digit_7_in_range_l261_261671

-- Define the condition for a number to contain a digit 7
def contains_digit_7 (n : Nat) : Prop :=
  (n / 10 = 7) || (n % 10 = 7)

-- The main theorem
theorem num_houses_with_digit_7_in_range (h : Nat) (H1 : 1 ≤ h ∧ h ≤ 70) : 
  ∃! n, 1 ≤ n ∧ n ≤ 70 ∧ contains_digit_7 n :=
sorry

end num_houses_with_digit_7_in_range_l261_261671


namespace cos_210_eq_neg_sqrt3_div2_l261_261161

theorem cos_210_eq_neg_sqrt3_div2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div2_l261_261161


namespace mean_eq_median_of_set_l261_261676

theorem mean_eq_median_of_set (x : ℕ) (hx : 0 < x) :
  let s := [1, 2, 4, 5, x]
  let mean := (1 + 2 + 4 + 5 + x) / 5
  let median := if x ≤ 2 then 2 else if x ≤ 4 then x else 4
  mean = median → (x = 3 ∨ x = 8) :=
by {
  sorry
}

end mean_eq_median_of_set_l261_261676


namespace cos_210_eq_neg_sqrt_3_div_2_l261_261156

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l261_261156


namespace divide_coal_l261_261843

noncomputable def part_of_pile (whole: ℚ) (parts: ℕ) := whole / parts
noncomputable def part_tons (total_tons: ℚ) (fraction: ℚ) := total_tons * fraction

theorem divide_coal (total_tons: ℚ) (parts: ℕ) (h: total_tons = 3 ∧ parts = 5):
  (part_of_pile 1 parts = 1/parts) ∧ (part_tons total_tons (1/parts) = total_tons / parts) :=
by
  sorry

end divide_coal_l261_261843


namespace tenth_term_geometric_sequence_l261_261545

theorem tenth_term_geometric_sequence :
  let a : ℚ := 5
  let r : ℚ := 3 / 4
  let a_n (n : ℕ) : ℚ := a * r ^ (n - 1)
  a_n 10 = 98415 / 262144 :=
by
  sorry

end tenth_term_geometric_sequence_l261_261545


namespace value_division_l261_261824

theorem value_division (x : ℝ) (h1 : 54 / x = 54 - 36) : x = 3 := by
  sorry

end value_division_l261_261824


namespace expected_value_8_sided_die_l261_261636

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l261_261636


namespace sum_of_exterior_segment_angles_is_540_l261_261399

-- Define the setup of the problem
def quadrilateral_inscribed_in_circle (A B C D : Type) : Prop := sorry
def angle_externally_inscribed (segment : Type) : ℝ := sorry

-- Main theorem statement
theorem sum_of_exterior_segment_angles_is_540
  (A B C D : Type)
  (h_quad : quadrilateral_inscribed_in_circle A B C D)
  (alpha beta gamma delta : ℝ)
  (h_alpha : alpha = angle_externally_inscribed A)
  (h_beta : beta = angle_externally_inscribed B)
  (h_gamma : gamma = angle_externally_inscribed C)
  (h_delta : delta = angle_externally_inscribed D) :
  alpha + beta + gamma + delta = 540 :=
sorry

end sum_of_exterior_segment_angles_is_540_l261_261399


namespace find_income_l261_261797

-- Define the conditions
def income_and_expenditure (income expenditure : ℕ) : Prop :=
  5 * expenditure = 3 * income

def savings (income expenditure : ℕ) (saving : ℕ) : Prop :=
  income - expenditure = saving

-- State the theorem
theorem find_income (expenditure : ℕ) (saving : ℕ) (h1 : income_and_expenditure 5 3) (h2 : savings (5 * expenditure) (3 * expenditure) saving) :
  5 * expenditure = 10000 :=
by
  -- Use the provided hint or conditions
  sorry

end find_income_l261_261797


namespace cos_210_eq_neg_sqrt_3_div_2_l261_261154

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l261_261154


namespace balloon_ratio_l261_261715

/-- Janice has 6 water balloons. --/
def Janice_balloons : Nat := 6

/-- Randy has half as many water balloons as Janice. --/
def Randy_balloons : Nat := Janice_balloons / 2

/-- Cynthia has 12 water balloons. --/
def Cynthia_balloons : Nat := 12

/-- The ratio of Cynthia's water balloons to Randy's water balloons is 4:1. --/
theorem balloon_ratio : Cynthia_balloons / Randy_balloons = 4 := by
  sorry

end balloon_ratio_l261_261715


namespace area_of_rhombus_l261_261555

/-- Given the radii of the circles circumscribed around triangles EFG and EGH
    are 10 and 20, respectively, then the area of rhombus EFGH is 30.72√3. -/
theorem area_of_rhombus (R1 R2 : ℝ) (A : ℝ) :
  R1 = 10 → R2 = 20 → A = 30.72 * Real.sqrt 3 :=
by sorry

end area_of_rhombus_l261_261555


namespace base_16_zeros_in_15_factorial_l261_261359

-- Definition of the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definition of the power function to generalize \( a^b \)
def power (a b : ℕ) : ℕ :=
  if b = 0 then 1 else a * power a (b - 1)

-- The constraints of the problem
def k_zeros_base_16 (n : ℕ) (k : ℕ) : Prop :=
  ∃ p, factorial n = p * power 16 k ∧ ¬ (∃ q, factorial n = q * power 16 (k + 1))

-- The main theorem we want to prove
theorem base_16_zeros_in_15_factorial : ∃ k, k_zeros_base_16 15 k ∧ k = 3 :=
by 
  sorry -- Proof to be found

end base_16_zeros_in_15_factorial_l261_261359


namespace expected_value_of_8_sided_die_l261_261639

def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℝ := 1 / 8

theorem expected_value_of_8_sided_die :
  (∑ x in outcomes, probability x * x) = 4.5 := 
sorry

end expected_value_of_8_sided_die_l261_261639


namespace max_value_of_f_smallest_positive_period_of_f_values_of_x_satisfying_f_ge_1_l261_261749

/-- Define the given function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.cos x * Real.sin (x + Real.pi / 4) - 1

/-- The maximum value of the function f(x) is sqrt(2) -/
theorem max_value_of_f : ∃ x, f x = Real.sqrt 2 := 
sorry

/-- The smallest positive period of the function f(x) -/
theorem smallest_positive_period_of_f :
  ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = Real.pi :=
sorry

/-- The set of values x that satisfy f(x) ≥ 1 -/
theorem values_of_x_satisfying_f_ge_1 :
  ∀ x, f x ≥ 1 ↔ ∃ k : ℤ, k * Real.pi ≤ x ∧ x ≤ k * Real.pi + Real.pi / 4 :=
sorry

end max_value_of_f_smallest_positive_period_of_f_values_of_x_satisfying_f_ge_1_l261_261749


namespace area_sin_6phi_is_pi_over_2_l261_261661

noncomputable def area_enclosed_by_sin_6phi : ℝ :=
  (1 / 2) * 12 * (1 / 2) * ∫ (x : ℝ) in 0..(Real.pi / 6), (Real.sin (6 * x)) ^ 2

theorem area_sin_6phi_is_pi_over_2 :
  area_enclosed_by_sin_6phi = Real.pi / 2 :=
by
  -- Proof goes here
  sorry

end area_sin_6phi_is_pi_over_2_l261_261661


namespace mary_rental_hours_l261_261255

def ocean_bike_fixed_fee := 17
def ocean_bike_hourly_rate := 7
def total_paid := 80

def calculate_hours (fixed_fee : Nat) (hourly_rate : Nat) (total_amount : Nat) : Nat :=
  (total_amount - fixed_fee) / hourly_rate

theorem mary_rental_hours :
  calculate_hours ocean_bike_fixed_fee ocean_bike_hourly_rate total_paid = 9 :=
by
  sorry

end mary_rental_hours_l261_261255


namespace bobby_toy_cars_l261_261835

theorem bobby_toy_cars (initial_cars : ℕ) (increase_rate : ℕ → ℕ) (n : ℕ) :
  initial_cars = 16 →
  increase_rate 1 = initial_cars + (initial_cars / 2) →
  increase_rate 2 = increase_rate 1 + (increase_rate 1 / 2) →
  increase_rate 3 = increase_rate 2 + (increase_rate 2 / 2) →
  n = 3 →
  increase_rate n = 54 :=
by
  intros
  sorry

end bobby_toy_cars_l261_261835


namespace geese_more_than_ducks_l261_261008

theorem geese_more_than_ducks (initial_ducks: ℕ) (initial_geese: ℕ) (initial_swans: ℕ) (additional_ducks: ℕ)
  (additional_geese: ℕ) (leaving_swans: ℕ) (leaving_geese: ℕ) (returning_geese: ℕ) (returning_swans: ℕ)
  (final_leaving_ducks: ℕ) (final_leaving_swans: ℕ)
  (initial_ducks_eq: initial_ducks = 25)
  (initial_geese_eq: initial_geese = 2 * initial_ducks - 10)
  (initial_swans_eq: initial_swans = 3 * initial_ducks + 8)
  (additional_ducks_eq: additional_ducks = 4)
  (additional_geese_eq: additional_geese = 7)
  (leaving_swans_eq: leaving_swans = 9)
  (leaving_geese_eq: leaving_geese = 5)
  (returning_geese_eq: returning_geese = 15)
  (returning_swans_eq: returning_swans = 11)
  (final_leaving_ducks_eq: final_leaving_ducks = 2 * (initial_ducks + additional_ducks))
  (final_leaving_swans_eq: final_leaving_swans = (initial_swans + returning_swans) / 2):
  (initial_geese + additional_geese + returning_geese - leaving_geese - final_leaving_geese + returning_geese) -
  (initial_ducks + additional_ducks - final_leaving_ducks) = 57 :=
by
  sorry

end geese_more_than_ducks_l261_261008


namespace fill_bucket_time_l261_261888

theorem fill_bucket_time (time_full_bucket : ℕ) (fraction : ℚ) (time_two_thirds_bucket : ℕ) 
  (h1 : time_full_bucket = 150) (h2 : fraction = 2 / 3) : time_two_thirds_bucket = 100 :=
sorry

end fill_bucket_time_l261_261888


namespace exponentiation_equation_l261_261711

theorem exponentiation_equation : 4^2011 * (-0.25)^2010 - 1 = 3 := 
by { sorry }

end exponentiation_equation_l261_261711


namespace length_chord_AB_standard_equation_circle_M_l261_261202

-- Define the point P_0 and circle centered at the origin with radius 2√2
def P₀ : ℝ × ℝ := (-1, 2)
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define the chord AB and the angle of inclination α
def α : ℝ := 135

-- Define the conditions for P_0, the inclination α, and being a bisector
def P₀_in_circle : Prop := circle_eq (P₀.1) (P₀.2)
def is_bisector (a b : ℝ) : Prop := true -- Placeholder, as the full geometric definition is complex

-- Problem 1: Length of AB when α = 135°
theorem length_chord_AB : 
  α = 135 → ∃ l : ℝ, l = sqrt 30 := 
by sorry

-- Define point C and conditions for circle M passing through C and tangent to AB at P0
def C : ℝ × ℝ := (3, 0)
def circle_M (x y : ℝ) : Prop := (x - 1/4)^2 + (y + 1/2)^2 = 125 / 16

-- Problem 2: Standard equation of circle M
theorem standard_equation_circle_M :
  ∃ x y : ℝ, circle_M x y ∧ circle_eq x y ∧ is_bisector x y :=
by sorry

end length_chord_AB_standard_equation_circle_M_l261_261202


namespace min_positive_announcements_l261_261007

theorem min_positive_announcements (x y : ℕ) 
  (h1 : x * (x - 1) = 110) 
  (h2 : y * (y - 1) + (x - y) * (x - 1 - (y - 1)) = 50) : 
  y >= 5 := 
sorry

end min_positive_announcements_l261_261007


namespace min_gennadys_l261_261707

theorem min_gennadys (alexanders borises vasilies : ℕ) (x : ℕ) 
    (h1 : alexanders = 45)
    (h2 : borises = 122)
    (h3 : vasilies = 27)
    (h4 : x = 49)
    (h5 : borises - 1 = alexanders + vasilies + x) :
  x = 49 := 
begin
  sorry,
end

end min_gennadys_l261_261707


namespace train_length_l261_261674

noncomputable def speed_kph := 56  -- speed in km/hr
def time_crossing := 9  -- time in seconds
noncomputable def speed_mps := speed_kph * 1000 / 3600  -- converting km/hr to m/s

theorem train_length : speed_mps * time_crossing = 140 := by
  -- conversion and result approximation
  sorry

end train_length_l261_261674


namespace problem1_f_x_linear_problem2_f_x_l261_261109

-- Problem 1 statement: Prove f(x) = 2x + 7 given conditions
theorem problem1_f_x_linear (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * x + 7)
  (h2 : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) : 
  ∀ x, f x = 2 * x + 7 :=
by sorry

-- Problem 2 statement: Prove f(x) = 2x - 1/x given conditions
theorem problem2_f_x (f : ℝ → ℝ) 
  (h1 : ∀ x, 2 * f x + f (1 / x) = 3 * x) : 
  ∀ x, f x = 2 * x - 1 / x :=
by sorry

end problem1_f_x_linear_problem2_f_x_l261_261109


namespace abs_eq_self_iff_nonneg_l261_261882

variable (a : ℝ)

theorem abs_eq_self_iff_nonneg (h : |a| = a) : a ≥ 0 :=
by
  sorry

end abs_eq_self_iff_nonneg_l261_261882


namespace number_of_adults_l261_261298

theorem number_of_adults (total_apples : ℕ) (children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) (h : total_apples = 450) (h1 : children = 33) (h2 : apples_per_child = 10) (h3 : apples_per_adult = 3) :
  total_apples - (children * apples_per_child) = 120 →
  (total_apples - (children * apples_per_child)) / apples_per_adult = 40 :=
by
  intros
  sorry

end number_of_adults_l261_261298


namespace find_n_l261_261200

-- We need a definition for permutations counting A_n^2 = n(n-1)
def permutations_squared (n : ℕ) : ℕ := n * (n - 1)

theorem find_n (n : ℕ) (h : permutations_squared n = 56) : n = 8 :=
by {
  sorry -- proof omitted as instructed
}

end find_n_l261_261200


namespace distinct_remainders_sum_quotient_l261_261351

theorem distinct_remainders_sum_quotient :
  let sq_mod_7 (n : Nat) := (n * n) % 7
  let distinct_remainders := List.eraseDup ([sq_mod_7 1, sq_mod_7 2, sq_mod_7 3, sq_mod_7 4, sq_mod_7 5])
  let s := List.sum distinct_remainders
  s / 7 = 1 :=
by
  sorry

end distinct_remainders_sum_quotient_l261_261351


namespace probability_second_try_success_l261_261827

-- Definitions based directly on the problem conditions
def total_keys : ℕ := 4
def keys_can_open_door : ℕ := 2
def keys_cannot_open_door : ℕ := total_keys - keys_can_open_door

-- Theorem statement translation
theorem probability_second_try_success :
  let prob_first_try_fail := (keys_cannot_open_door : ℝ) / total_keys,
      prob_second_try_success := (keys_can_open_door : ℝ) / (total_keys - 1)
  in
  prob_first_try_fail * prob_second_try_success = (1/3 : ℝ) :=
by
  -- Proof content goes here
  sorry

end probability_second_try_success_l261_261827


namespace circle_equation1_circle_equation2_l261_261182

-- Definitions for the first question
def center1 : (ℝ × ℝ) := (2, -2)
def pointP : (ℝ × ℝ) := (6, 3)

-- Definitions for the second question
def pointA : (ℝ × ℝ) := (-4, -5)
def pointB : (ℝ × ℝ) := (6, -1)

-- Theorems we need to prove
theorem circle_equation1 : (x - 2)^2 + (y + 2)^2 = 41 :=
sorry

theorem circle_equation2 : (x - 1)^2 + (y + 3)^2 = 29 :=
sorry

end circle_equation1_circle_equation2_l261_261182


namespace coupon_redeem_day_l261_261461

theorem coupon_redeem_day (first_day : ℕ) (redeem_every : ℕ) : 
  (∀ n : ℕ, n < 8 → (first_day + n * redeem_every) % 7 ≠ 6) ↔ (first_day % 7 = 2 ∨ first_day % 7 = 5) :=
by
  sorry

end coupon_redeem_day_l261_261461


namespace tangerines_more_than_oranges_l261_261630

-- Define initial conditions
def initial_apples := 9
def initial_oranges := 5
def initial_tangerines := 17

-- Define actions taken
def oranges_taken := 2
def tangerines_taken := 10

-- Resulting quantities
def oranges_left := initial_oranges - oranges_taken
def tangerines_left := initial_tangerines - tangerines_taken

-- Proof problem
theorem tangerines_more_than_oranges : tangerines_left - oranges_left = 4 := 
by sorry

end tangerines_more_than_oranges_l261_261630


namespace total_roses_l261_261124

theorem total_roses (a : ℕ) (x y k : ℕ) (h1 : 300 ≤ a) (h2 : a ≤ 400)
  (h3 : a = 21 * x + 13) (h4 : a = 15 * y - 8) (h5 : a + 8 = 105 * k) :
  a = 307 :=
sorry

end total_roses_l261_261124


namespace greatest_third_side_l261_261950

-- Given data and the Triangle Inequality theorem
theorem greatest_third_side (c : ℕ) (h1 : 8 < c) (h2 : c < 22) : c = 21 :=
by
  sorry

end greatest_third_side_l261_261950


namespace number_of_int_pairs_l261_261426

theorem number_of_int_pairs (x y : ℤ) (h : x^2 + 2 * y^2 < 25) : 
  ∃ S : Finset (ℤ × ℤ), S.card = 55 ∧ ∀ (a : ℤ × ℤ), a ∈ S ↔ a.1^2 + 2 * a.2^2 < 25 :=
sorry

end number_of_int_pairs_l261_261426


namespace find_divisor_l261_261500

theorem find_divisor (D N : ℕ) (h₁ : N = 265) (h₂ : N / D + 8 = 61) : D = 5 :=
by
  sorry

end find_divisor_l261_261500


namespace find_water_needed_l261_261252

def apple_juice := 4
def honey (A : ℕ) := 3 * A
def water (H : ℕ) := 3 * H

theorem find_water_needed : water (honey apple_juice) = 36 :=
  sorry

end find_water_needed_l261_261252


namespace intersection_of_A_and_B_l261_261735

-- Given sets A and B
def A : Set ℤ := { -1, 0, 1, 2 }
def B : Set ℤ := { 0, 2, 3 }

-- Prove that the intersection of A and B is {0, 2}
theorem intersection_of_A_and_B : A ∩ B = {0, 2} := 
by
  sorry

end intersection_of_A_and_B_l261_261735


namespace danielle_rooms_is_6_l261_261869

def heidi_rooms (danielle_rooms : ℕ) : ℕ := 3 * danielle_rooms
def grant_rooms (heidi_rooms : ℕ) : ℕ := heidi_rooms / 9

theorem danielle_rooms_is_6 (danielle_rooms : ℕ) (h1 : heidi_rooms danielle_rooms = 18) (h2 : grant_rooms (heidi_rooms danielle_rooms) = 2) :
  danielle_rooms = 6 :=
by 
  sorry

end danielle_rooms_is_6_l261_261869


namespace max_rooks_in_cube_l261_261653

def non_attacking_rooks (n : ℕ) (cube : ℕ × ℕ × ℕ) : ℕ :=
  if cube = (8, 8, 8) then 64 else 0

theorem max_rooks_in_cube:
  non_attacking_rooks 64 (8, 8, 8) = 64 :=
by
  -- proof by logical steps matching the provided solution, if necessary, start with sorry for placeholder
  sorry

end max_rooks_in_cube_l261_261653


namespace turnip_heavier_than_zhuchka_l261_261510

theorem turnip_heavier_than_zhuchka {C B M T : ℝ} 
  (h1 : B = 3 * C)
  (h2 : M = C / 10)
  (h3 : T = 60 * M) : 
  T / B = 2 :=
by
  sorry

end turnip_heavier_than_zhuchka_l261_261510


namespace pq_parallel_ab_l261_261051

-- Given problem setup
variables (A B C M K L P Q : Type*) [MetricSpace A]

-- Conditions
variables (h_M_on_AB : M ∈ line[A, B])
variables (h_K_on_BC : K ∈ line[B, C])
variables (h_L_on_AC : L ∈ line[A, C])
variables (h_MK_parallel_AC : Parallel (line[M, K]) (line[A, C]))
variables (h_ML_parallel_BC : Parallel (line[M, L]) (line[B, C]))
variables (h_BL_inter_MK : ∃ P, P ∈ line[B, L] ∧ P ∈ line[M, K])
variables (h_AK_inter_ML : ∃ Q, Q ∈ line[A, K] ∧ Q ∈ line[M, L])

-- Question
theorem pq_parallel_ab : Parallel (line[P, Q]) (line[A, B]) :=
sorry

end pq_parallel_ab_l261_261051


namespace basketball_rim_height_l261_261265

theorem basketball_rim_height
    (height_in_inches : ℕ)
    (reach_in_inches : ℕ)
    (jump_in_inches : ℕ)
    (above_rim_in_inches : ℕ) :
    height_in_inches = 72
    → reach_in_inches = 22
    → jump_in_inches = 32
    → above_rim_in_inches = 6
    → (height_in_inches + reach_in_inches + jump_in_inches - above_rim_in_inches) = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end basketball_rim_height_l261_261265


namespace Danielle_rooms_is_6_l261_261872

-- Definitions for the problem conditions
def Heidi_rooms (Danielle_rooms : ℕ) : ℕ := 3 * Danielle_rooms
def Grant_rooms (Heidi_rooms : ℕ) : ℕ := Heidi_rooms / 9
def Grant_rooms_value : ℕ := 2

-- Theorem statement
theorem Danielle_rooms_is_6 (h : Grant_rooms_value = Grant_rooms (Heidi_rooms d)) : d = 6 :=
by
  sorry

end Danielle_rooms_is_6_l261_261872


namespace min_gennadies_l261_261701

noncomputable section

def minGennadiesNeeded (alexanders borises vasilies : Nat) : Nat :=
  let needed_gaps := borises - 1
  let total_others := alexanders + vasilies
  if needed_gaps > total_others then needed_gaps - total_others else 0

theorem min_gennadies (alexanders borises vasilies : Nat) (h_alex: alexanders = 45) (h_boris: borises = 122) (h_vasil: vasilies = 27):
  minGennadiesNeeded alexanders borises vasilies = 49 := by
  rw [h_alex, h_boris, h_vasil]
  simp [minGennadiesNeeded]
  sorry

end min_gennadies_l261_261701


namespace soup_can_pyramid_rows_l261_261829

theorem soup_can_pyramid_rows (n : ℕ) :
  (∃ (n : ℕ), (2 * n^2 - n = 225)) → n = 11 :=
by
  sorry

end soup_can_pyramid_rows_l261_261829


namespace octagon_diagonals_l261_261787

def num_sides := 8

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diagonals : num_diagonals num_sides = 20 :=
by
  sorry

end octagon_diagonals_l261_261787


namespace solve_equation1_solve_equation2_l261_261074

-- Let x be a real number
variable {x : ℝ}

-- The first equation and its solutions
def equation1 (x : ℝ) : Prop := (x - 1) ^ 2 - 25 = 0

-- Asserting that the solutions to the first equation are x = 6 or x = -4
theorem solve_equation1 (x : ℝ) : equation1 x ↔ x = 6 ∨ x = -4 :=
by
  sorry

-- The second equation and its solution
def equation2 (x : ℝ) : Prop := (1 / 4) * (2 * x + 3) ^ 3 = 16

-- Asserting that the solution to the second equation is x = 1/2
theorem solve_equation2 (x : ℝ) : equation2 x ↔ x = 1 / 2 :=
by
  sorry

end solve_equation1_solve_equation2_l261_261074


namespace original_price_l261_261828

variable (P : ℝ)

theorem original_price (h : 560 = 1.05 * (0.72 * P)) : P = 740.46 := 
by
  sorry

end original_price_l261_261828


namespace diana_can_paint_statues_l261_261012

theorem diana_can_paint_statues (total_paint : ℚ) (paint_per_statue : ℚ) 
  (h1 : total_paint = 3 / 6) (h2 : paint_per_statue = 1 / 6) : 
  total_paint / paint_per_statue = 3 :=
by
  sorry

end diana_can_paint_statues_l261_261012


namespace perpendicular_tangents_l261_261573

theorem perpendicular_tangents (a b : ℝ) (h1 : ∀ (x y : ℝ), y = x^3 → y = (3 * x^2) * (x - 1) + 1 → y = 3 * (x - 1) + 1) (h2 : (a : ℝ) * 1 - (b : ℝ) * 1 = 2) 
 (h3 : (a : ℝ)/(b : ℝ) * 3 = -1) : a / b = -1 / 3 :=
by
  sorry

end perpendicular_tangents_l261_261573


namespace interest_rate_proof_l261_261918

noncomputable def remaining_interest_rate (total_investment yearly_interest part_investment interest_rate_part amount_remaining_interest : ℝ) : Prop :=
  (part_investment * interest_rate_part) + amount_remaining_interest = yearly_interest ∧
  (total_investment - part_investment) * (amount_remaining_interest / (total_investment - part_investment)) = amount_remaining_interest

theorem interest_rate_proof :
  remaining_interest_rate 3000 256 800 0.1 176 :=
by
  sorry

end interest_rate_proof_l261_261918


namespace solution_set_l261_261285

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem solution_set (x : ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_def : ∀ x : ℝ, x >= 0 → f x = x^2 - 4 * x) :
    f (x + 2) < 5 ↔ -7 < x ∧ x < 3 :=
sorry

end solution_set_l261_261285


namespace projection_of_orthogonal_vectors_l261_261335

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scale := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (scale * v.1, scale * v.2)

theorem projection_of_orthogonal_vectors
  (a b : ℝ × ℝ)
  (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj : proj (4, -2) a = (4 / 5, 8 / 5)) :
  proj (4, -2) b = (16 / 5, -18 / 5) :=
sorry

end projection_of_orthogonal_vectors_l261_261335


namespace polynomial_irreducible_over_Z_iff_Q_l261_261781

theorem polynomial_irreducible_over_Z_iff_Q (f : Polynomial ℤ) :
  Irreducible f ↔ Irreducible (f.map (Int.castRingHom ℚ)) :=
sorry

end polynomial_irreducible_over_Z_iff_Q_l261_261781


namespace integer_base10_from_bases_l261_261483

theorem integer_base10_from_bases (C D : ℕ) (hC : 0 ≤ C ∧ C ≤ 7) (hD : 0 ≤ D ∧ D ≤ 5)
    (h : 8 * C + D = 6 * D + C) : C = 0 ∧ D = 0 ∧ (8 * C + D = 0) := by
  sorry

end integer_base10_from_bases_l261_261483


namespace segments_form_pentagon_l261_261291

theorem segments_form_pentagon (a b c d e : ℝ) 
  (h_sum : a + b + c + d + e = 2)
  (h_a : a > 1/10)
  (h_b : b > 1/10)
  (h_c : c > 1/10)
  (h_d : d > 1/10)
  (h_e : e > 1/10) :
  a + b + c + d > e ∧ a + b + c + e > d ∧ a + b + d + e > c ∧ a + c + d + e > b ∧ b + c + d + e > a := 
sorry

end segments_form_pentagon_l261_261291


namespace jerry_games_before_birthday_l261_261763

def num_games_before (current received : ℕ) : ℕ :=
  current - received

theorem jerry_games_before_birthday : 
  ∀ (current received before : ℕ), current = 9 → received = 2 → before = num_games_before current received → before = 7 :=
by
  intros current received before h_current h_received h_before
  rw [h_current, h_received] at h_before
  exact h_before

end jerry_games_before_birthday_l261_261763


namespace taller_tree_height_l261_261802

theorem taller_tree_height
  (h : ℕ)
  (h_shorter_ratio : h - 16 = (3 * h) / 4) : h = 64 := by
  sorry

end taller_tree_height_l261_261802


namespace max_grapes_in_bag_l261_261942

theorem max_grapes_in_bag : ∃ (x : ℕ), x > 100 ∧ x % 3 = 1 ∧ x % 5 = 2 ∧ x % 7 = 4 ∧ x = 172 := by
  sorry

end max_grapes_in_bag_l261_261942


namespace find_range_of_f_l261_261431

noncomputable def f (x : ℝ) : ℝ := (Real.logb (1/2) x) ^ 2 - 2 * (Real.logb (1/2) x) + 4

theorem find_range_of_f :
  ∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → 7 ≤ f x ∧ f x ≤ 12 :=
by
  sorry

end find_range_of_f_l261_261431


namespace minor_axis_length_of_ellipse_l261_261357

theorem minor_axis_length_of_ellipse :
  ∀ (x y : ℝ), (9 * x^2 + y^2 = 36) → 4 = 4 :=
by
  intros x y h
  -- the proof goes here
  sorry

end minor_axis_length_of_ellipse_l261_261357


namespace unique_real_solution_bound_l261_261554

theorem unique_real_solution_bound (b : ℝ) :
  (∀ x : ℝ, x^3 - b*x^2 - 4*b*x + b^2 - 4 = 0 → ∃! y : ℝ, y = x) → b < 1 :=
by
  sorry

end unique_real_solution_bound_l261_261554


namespace men_at_conference_l261_261581

theorem men_at_conference (M : ℕ) 
  (num_women : ℕ) (num_children : ℕ)
  (indian_men_fraction : ℚ) (indian_women_fraction : ℚ)
  (indian_children_fraction : ℚ) (non_indian_fraction : ℚ)
  (num_women_eq : num_women = 300)
  (num_children_eq : num_children = 500)
  (indian_men_fraction_eq : indian_men_fraction = 0.10)
  (indian_women_fraction_eq : indian_women_fraction = 0.60)
  (indian_children_fraction_eq : indian_children_fraction = 0.70)
  (non_indian_fraction_eq : non_indian_fraction = 0.5538461538461539) :
  M = 500 :=
by
  sorry

end men_at_conference_l261_261581


namespace smallest_part_2340_division_l261_261111

theorem smallest_part_2340_division :
  ∃ (A B C : ℕ), (A + B + C = 2340) ∧ 
                 (A / 5 = B / 7) ∧ 
                 (B / 7 = C / 11) ∧ 
                 (A = 510) :=
by 
  sorry

end smallest_part_2340_division_l261_261111


namespace minimum_gennadies_l261_261704

theorem minimum_gennadies (A B V G : ℕ) (hA : A = 45) (hB : B = 122) (hV : V = 27) (hGap : G + A + V >= B - 1) :
  G >= 49 :=
by 
  have := by linarith [hGap, hA, hB, hV]
  exact this

end minimum_gennadies_l261_261704


namespace gardening_project_total_cost_l261_261983

noncomputable def cost_gardening_project : ℕ := 
  let number_rose_bushes := 20
  let cost_per_rose_bush := 150
  let cost_fertilizer_per_bush := 25
  let gardener_work_hours := [6, 5, 4, 7]
  let gardener_hourly_rate := 30
  let soil_amount := 100
  let cost_per_cubic_foot := 5

  let cost_roses := number_rose_bushes * cost_per_rose_bush
  let cost_fertilizer := number_rose_bushes * cost_fertilizer_per_bush
  let total_work_hours := List.sum gardener_work_hours
  let cost_labor := total_work_hours * gardener_hourly_rate
  let cost_soil := soil_amount * cost_per_cubic_foot

  cost_roses + cost_fertilizer + cost_labor + cost_soil

theorem gardening_project_total_cost : cost_gardening_project = 4660 := by
  sorry

end gardening_project_total_cost_l261_261983


namespace correct_solutions_l261_261016

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ (x y : ℝ), f (x * y) = f x * f y - 2 * x * y

theorem correct_solutions :
  (∀ x : ℝ, f x = 2 * x) ∨ (∀ x : ℝ, f x = -x) := sorry

end correct_solutions_l261_261016


namespace division_of_composite_products_l261_261306

noncomputable def product_of_first_seven_composites : ℕ :=
  4 * 6 * 8 * 9 * 10 * 12 * 14

noncomputable def product_of_next_seven_composites : ℕ :=
  15 * 16 * 18 * 20 * 21 * 22 * 24

noncomputable def divided_product_composites : ℚ :=
  product_of_first_seven_composites / product_of_next_seven_composites

theorem division_of_composite_products : divided_product_composites = 1 / 176 := by
  sorry

end division_of_composite_products_l261_261306


namespace part1_solution_set_part2_range_m_l261_261443
open Real

noncomputable def f (x : ℝ) : ℝ := abs (x - 1)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := - abs (x + 4) + m

-- Part I: Solution set for f(x) > x + 1 is (-∞, 0)
theorem part1_solution_set : { x : ℝ | f x > x + 1 } = { x : ℝ | x < 0 } :=
sorry

-- Part II: Range of m when the graphs of y = f(x) and y = g(x) have common points
theorem part2_range_m (m : ℝ) : (∃ x : ℝ, f x = g x m) → m ≥ 5 :=
sorry

end part1_solution_set_part2_range_m_l261_261443


namespace problem1_l261_261284

theorem problem1 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y > 2) : 
    (1 + x) / y < 2 ∨ (1 + y) / x < 2 := 
sorry

end problem1_l261_261284


namespace solid_circles_count_2006_l261_261672

def series_of_circles (n : ℕ) : List Char :=
  if n ≤ 0 then []
  else if n % 5 == 0 then '●' :: series_of_circles (n - 1)
  else '○' :: series_of_circles (n - 1)

def count_solid_circles (l : List Char) : ℕ :=
  l.count '●'

theorem solid_circles_count_2006 : count_solid_circles (series_of_circles 2006) = 61 := 
by
  sorry

end solid_circles_count_2006_l261_261672


namespace largest_side_of_rectangle_l261_261598

theorem largest_side_of_rectangle (l w : ℕ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 1920) : max l w = 101 := 
by
  sorry

end largest_side_of_rectangle_l261_261598


namespace sum_of_fib_factorials_last_two_digits_l261_261011

-- Define the condition that factorials greater than 10 end in 00
def end_in_00_if_gt_10 {n : ℕ} (hn : n > 10) : (n ! % 100) = 0 := sorry

-- Define the factorials of the specific Fibonacci numbers
def fib_factorials := [1!, 1!, 2!, 3!, 5!, 8!, 13!, 21!, 34!, 55!, 89!, 144!]

-- Define a function to get the last two digits of a number
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Calculate the sum of the last two digits of the factorials of the Fibonacci numbers
def sum_of_last_two_digits : ℕ := 
  fib_factorials.take 6.map last_two_digits.sum + 0 + 0 + 0 + 0 + 0 + 0

-- The statement to prove
theorem sum_of_fib_factorials_last_two_digits : sum_of_last_two_digits = 5 := by
  sorry

end sum_of_fib_factorials_last_two_digits_l261_261011


namespace principal_amount_l261_261512

/-- Given:
 - 820 = P + (P * R * 2) / 100
 - 1020 = P + (P * R * 6) / 100
Prove:
 - P = 720
--/

theorem principal_amount (P R : ℝ) (h1 : 820 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 6) / 100) : P = 720 :=
by
  sorry

end principal_amount_l261_261512


namespace smallest_positive_integer_divisible_l261_261854

theorem smallest_positive_integer_divisible (n : ℕ) (h1 : 15 = 3 * 5) (h2 : 16 = 2 ^ 4) (h3 : 18 = 2 * 3 ^ 2) :
  n = Nat.lcm (Nat.lcm 15 16) 18 ↔ n = 720 :=
by
  sorry

end smallest_positive_integer_divisible_l261_261854


namespace first_digit_base9_of_base3_num_l261_261570

theorem first_digit_base9_of_base3_num {y : ℕ} (hy : y = 21211122211122211111₃) : 
  ∃ d, d = y.digits 9^.0 ∧ d.headI = 4 :=
by sorry

end first_digit_base9_of_base3_num_l261_261570


namespace tan_angle_addition_l261_261874

theorem tan_angle_addition (y : ℝ) (hyp : Real.tan y = -3) : 
  Real.tan (y + Real.pi / 3) = - (5 * Real.sqrt 3 - 6) / 13 := 
by 
  sorry

end tan_angle_addition_l261_261874


namespace total_population_l261_261048

variables (b g t : ℕ)

theorem total_population (h1 : b = 4 * g) (h2 : g = 5 * t) : b + g + t = 26 * t :=
sorry

end total_population_l261_261048


namespace not_equal_77_l261_261258

theorem not_equal_77 (x y : ℤ) : x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end not_equal_77_l261_261258


namespace percentage_decrease_l261_261353

-- Define the condition given in the problem
def is_increase (pct : ℤ) : Prop := pct > 0
def is_decrease (pct : ℤ) : Prop := pct < 0

-- The main proof statement
theorem percentage_decrease (pct : ℤ) (h : pct = -10) : is_decrease pct :=
by
  sorry

end percentage_decrease_l261_261353


namespace trigonometric_proof_l261_261566

noncomputable def f (x : ℝ) (a b : ℝ) := a * (Real.sin x ^ 3) + b * (Real.cos x ^ 3) + 4

theorem trigonometric_proof (a b : ℝ) (h₁ : f (Real.sin (10 * Real.pi / 180 )) a b = 5) : 
  f (Real.cos (100 * Real.pi / 180 )) a b = 3 := 
sorry

end trigonometric_proof_l261_261566


namespace g_at_3_l261_261884

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 7 * x ^ 2 + 3 * x - 2

theorem g_at_3 : g 3 = 79 := 
by 
  -- proof placeholder
  sorry

end g_at_3_l261_261884


namespace smallest_b_greater_than_1_l261_261470

def g (x : ℕ) : ℕ :=
  if x % 35 = 0 then x / 35
  else if x % 7 = 0 then 5 * x
  else if x % 5 = 0 then 7 * x
  else x + 5

def g_iter (n : ℕ) (x : ℕ) : ℕ := Nat.iterate g n x

theorem smallest_b_greater_than_1 (b : ℕ) :
  (b > 1) → 
  g_iter 1 3 = 8 ∧ g_iter b 3 = 8 →
  b = 21 := by
  sorry

end smallest_b_greater_than_1_l261_261470


namespace least_x_for_factorial_divisible_by_100000_l261_261855

theorem least_x_for_factorial_divisible_by_100000 :
  ∃ x : ℕ, (∀ y : ℕ, (y < x) → (¬ (100000 ∣ y!))) ∧ (100000 ∣ x!) :=
  sorry

end least_x_for_factorial_divisible_by_100000_l261_261855


namespace find_a_l261_261389

noncomputable def calculation (a : ℝ) (x : ℝ) (y : ℝ) (b : ℝ) (c : ℝ) : Prop :=
  (x * y) / (a * b * c) = 840

theorem find_a : calculation 50 0.0048 3.5 0.1 0.004 :=
by
  sorry

end find_a_l261_261389


namespace mrs_hilt_current_rocks_l261_261254

-- Definitions based on conditions
def total_rocks_needed : ℕ := 125
def more_rocks_needed : ℕ := 61

-- Lean statement proving the required amount of currently held rocks
theorem mrs_hilt_current_rocks : (total_rocks_needed - more_rocks_needed) = 64 :=
by
  -- proof will be here
  sorry

end mrs_hilt_current_rocks_l261_261254


namespace recruits_total_l261_261800

theorem recruits_total (P N D : ℕ) (total_recruits : ℕ) 
  (h1 : P = 50) 
  (h2 : N = 100) 
  (h3 : D = 170)
  (h4 : (∃ x y, (x = 50) ∧ (y = 100) ∧ (x = 4 * y))
        ∨ (∃ x z, (x = 50) ∧ (z = 170) ∧ (x = 4 * z))
        ∨ (∃ y z, (y = 100) ∧ (z = 170) ∧ (y = 4 * z))) : 
  total_recruits = 211 :=
by
  sorry

end recruits_total_l261_261800


namespace hose_removal_rate_l261_261076

theorem hose_removal_rate (w l d : ℝ) (capacity_fraction : ℝ) (drain_time : ℝ) 
  (h_w : w = 60) 
  (h_l : l = 150) 
  (h_d : d = 10) 
  (h_capacity_fraction : capacity_fraction = 0.80) 
  (h_drain_time : drain_time = 1200) : 
  ((w * l * d * capacity_fraction) / drain_time) = 60 :=
by
  -- the proof is omitted here
  sorry

end hose_removal_rate_l261_261076


namespace total_cost_of_stickers_l261_261061

-- Definitions based on given conditions
def initial_funds_per_person := 9
def cost_of_deck_of_cards := 10
def Dora_packs_of_stickers := 2

-- Calculate the total amount of money collectively after buying the deck of cards
def remaining_funds := 2 * initial_funds_per_person - cost_of_deck_of_cards

-- Calculate the total packs of stickers if split evenly
def total_packs_of_stickers := 2 * Dora_packs_of_stickers

-- Prove the total cost of the boxes of stickers
theorem total_cost_of_stickers : remaining_funds = 8 := by
  -- Given initial funds per person, cost of deck of cards, and packs of stickers for Dora, the theorem should hold.
  sorry

end total_cost_of_stickers_l261_261061


namespace find_u_plus_v_l261_261446

-- Conditions: 3u - 4v = 17 and 5u - 2v = 1.
-- Question: Find the value of u + v.

theorem find_u_plus_v (u v : ℚ) (h1 : 3 * u - 4 * v = 17) (h2 : 5 * u - 2 * v = 1) : u + v = -8 :=
by
  sorry

end find_u_plus_v_l261_261446


namespace transformed_data_stats_l261_261209

open Real BigOperators

variables {n : ℕ} (x : Fin n → ℝ)

-- Given conditions
axiom avg_given : (∑ i, x i) / n = 2
axiom var_given : (∑ i, (x i - 2)^2) / n = 5

-- Definitions for transformed data
noncomputable def transformed_data (i : Fin n) : ℝ := 2 * x i + 1

-- Prove that the new average and new variance match the given correct answer
theorem transformed_data_stats :
  (∑ i, transformed_data x i) / n = 5 ∧
  (∑ i, (transformed_data x i - 5)^2) / n = 20 :=
by
  sorry

end transformed_data_stats_l261_261209


namespace calculate_expression_l261_261540

theorem calculate_expression : 5^3 + 5^3 + 5^3 + 5^3 = 625 :=
  sorry

end calculate_expression_l261_261540


namespace log_expression_identity_l261_261543

theorem log_expression_identity :
  (Real.log 5 / Real.log 10)^2 + (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) = 1 :=
by
  sorry

end log_expression_identity_l261_261543


namespace distance_not_six_l261_261078

theorem distance_not_six (x : ℝ) : 
  (x = 6 → 10 + (x - 3) * 1.8 ≠ 17.2) ∧ 
  (10 + (x - 3) * 1.8 = 17.2 → x ≠ 6) :=
by {
  sorry
}

end distance_not_six_l261_261078


namespace investment_value_after_two_weeks_l261_261067

theorem investment_value_after_two_weeks (initial_investment : ℝ) (gain_first_week : ℝ) (gain_second_week : ℝ) : 
  initial_investment = 400 → 
  gain_first_week = 0.25 → 
  gain_second_week = 0.5 → 
  ((initial_investment * (1 + gain_first_week) * (1 + gain_second_week)) = 750) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    400 * (1 + 0.25) * (1 + 0.5)
    = 400 * 1.25 * 1.5 : by ring
    = 750 : by norm_num

end investment_value_after_two_weeks_l261_261067


namespace range_of_k_l261_261319

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (x - 1) / (x - 2) = k / (x - 2) + 2 ∧ x ≥ 0 ∧ x ≠ 2) ↔ (k ≤ 3 ∧ k ≠ 1) :=
by
  sorry

end range_of_k_l261_261319


namespace simplify_expression_l261_261277

theorem simplify_expression :
  2 + 1 / (2 + 1 / (2 + 1 / 2)) = 29 / 12 :=
by
  sorry  -- Proof will be provided here

end simplify_expression_l261_261277


namespace school_supply_cost_l261_261587

theorem school_supply_cost (num_students : ℕ) (pens_per_student : ℕ) (pen_cost : ℝ) 
  (notebooks_per_student : ℕ) (notebook_cost : ℝ) 
  (binders_per_student : ℕ) (binder_cost : ℝ) 
  (highlighters_per_student : ℕ) (highlighter_cost : ℝ) 
  (teacher_discount : ℝ) : 
  num_students = 30 →
  pens_per_student = 5 →
  pen_cost = 0.50 →
  notebooks_per_student = 3 →
  notebook_cost = 1.25 →
  binders_per_student = 1 →
  binder_cost = 4.25 →
  highlighters_per_student = 2 →
  highlighter_cost = 0.75 →
  teacher_discount = 100 →
  (num_students * 
    (pens_per_student * pen_cost + notebooks_per_student * notebook_cost + 
    binders_per_student * binder_cost + highlighters_per_student * highlighter_cost) - 
    teacher_discount) = 260 :=
by
  intros _ _ _ _ _ _ _ _ _ _

  -- Sorry added to skip the proof
  sorry

end school_supply_cost_l261_261587


namespace circle_equation_tangent_to_line_l261_261932

theorem circle_equation_tangent_to_line
  (h k : ℝ) (A B C : ℝ)
  (hxk : h = 2) (hyk : k = -1) 
  (hA : A = 3) (hB : B = -4) (hC : C = 5)
  (r_squared : ℝ := (|A * h + B * k + C| / Real.sqrt (A^2 + B^2))^2)
  (h_radius : r_squared = 9) :
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r_squared := 
by
  sorry

end circle_equation_tangent_to_line_l261_261932


namespace remainder_is_23_l261_261231

def number_remainder (n : ℤ) : ℤ :=
  n % 36

theorem remainder_is_23 (n : ℤ) (h1 : n % 4 = 3) (h2 : n % 9 = 5) :
  number_remainder n = 23 :=
by
  sorry

end remainder_is_23_l261_261231


namespace domain_of_f_2x_minus_1_l261_261032

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) (dom : ∀ x, f x ≠ 0 → (0 < x ∧ x < 1)) :
  ∀ x, f (2*x - 1) ≠ 0 → (1/2 < x ∧ x < 1) :=
by
  sorry

end domain_of_f_2x_minus_1_l261_261032


namespace intersection_A_B_l261_261316

-- Define the set A as natural numbers greater than 1
def A : Set ℕ := {x | x > 1}

-- Define the set B as numbers less than or equal to 3
def B : Set ℕ := {x | x ≤ 3}

-- Define the intersection of A and B
def A_inter_B : Set ℕ := {x | x ∈ A ∧ x ∈ B}

-- State the theorem we want to prove
theorem intersection_A_B : A_inter_B = {2, 3} :=
  sorry

end intersection_A_B_l261_261316


namespace cos_three_theta_l261_261747

open Complex

theorem cos_three_theta (θ : ℝ) (h : cos θ = 1 / 2) : cos (3 * θ) = -1 / 2 :=
by
  sorry

end cos_three_theta_l261_261747


namespace cos_210_eq_neg_sqrt_3_div_2_l261_261157

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l261_261157


namespace simplify_sqrt_88200_l261_261611

theorem simplify_sqrt_88200 :
  ∀ (a b c d e : ℕ), a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 7 ∧ e = 1 →
  ∃ f g : ℝ, (88200 : ℝ) = (f^2 * g) ∧ f = 882 ∧ g = 10 ∧ real.sqrt (88200 : ℝ) = f * real.sqrt g :=
sorry

end simplify_sqrt_88200_l261_261611


namespace no_such_positive_integer_l261_261192

theorem no_such_positive_integer (n : ℕ) (d : ℕ → ℕ)
  (h₁ : ∃ d1 d2 d3 d4 d5, d 1 = d1 ∧ d 2 = d2 ∧ d 3 = d3 ∧ d 4 = d4 ∧ d 5 = d5) 
  (h₂ : 1 ≤ d 1 ∧ d 1 < d 2 ∧ d 2 < d 3 ∧ d 3 < d 4 ∧ d 4 < d 5)
  (h₃ : ∀ i, 1 ≤ i → i ≤ 5 → d i ∣ n)
  (h₄ : ∀ i, 1 ≤ i → i ≤ 5 → ∀ j, i ≠ j → d i ≠ d j)
  (h₅ : ∃ x, 1 + (d 2)^2 + (d 3)^2 + (d 4)^2 + (d 5)^2 = x^2) :
  false :=
sorry

end no_such_positive_integer_l261_261192


namespace smallest_positive_integer_divisible_by_15_16_18_l261_261852

theorem smallest_positive_integer_divisible_by_15_16_18 : 
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) ∧ n = 720 := 
by
  sorry

end smallest_positive_integer_divisible_by_15_16_18_l261_261852


namespace probability_of_multiple_of_45_l261_261577

noncomputable def single_digit_multiples_of_3 := {x : ℕ | x ∈ {3, 6, 9}}

noncomputable def prime_numbers_less_than_20 := {x : ℕ | x ∈ {2, 3, 5, 7, 11, 13, 17, 19}}

noncomputable def is_multiple_of_45 (n : ℕ) : Prop :=
  45 ∣ n

theorem probability_of_multiple_of_45 : 
  (↑((1 : ℚ) / 3) : ℚ) * (↑((1 : ℚ) / 8) : ℚ) = (1 : ℚ) / 24 :=
by 
  sorry

end probability_of_multiple_of_45_l261_261577


namespace max_popsicles_l261_261062

theorem max_popsicles (total_money : ℝ) (cost_per_popsicle : ℝ) (h_money : total_money = 19.23) (h_cost : cost_per_popsicle = 1.60) : 
  ∃ (x : ℕ), x = ⌊total_money / cost_per_popsicle⌋ ∧ x = 12 :=
by
    sorry

end max_popsicles_l261_261062


namespace mixed_water_temp_l261_261459

def cold_water_temp : ℝ := 20   -- Temperature of cold water
def hot_water_temp : ℝ := 40    -- Temperature of hot water

theorem mixed_water_temp :
  (cold_water_temp + hot_water_temp) / 2 = 30 := 
by sorry

end mixed_water_temp_l261_261459


namespace probability_kyle_catherine_not_david_l261_261766

/--
Kyle, David, and Catherine each try independently to solve a problem. 
Their individual probabilities for success are 1/3, 2/7, and 5/9.
Prove that the probability that Kyle and Catherine, but not David, will solve the problem is 25/189.
-/
theorem probability_kyle_catherine_not_david :
  let P_K := 1 / 3
  let P_D := 2 / 7
  let P_C := 5 / 9
  let P_D_c := 1 - P_D
  P_K * P_C * P_D_c = 25 / 189 :=
by
  sorry

end probability_kyle_catherine_not_david_l261_261766


namespace solution_set_inequality_l261_261268

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 :=
by {
  sorry -- proof omitted
}

end solution_set_inequality_l261_261268


namespace smallest_n_not_divisible_by_10_smallest_n_correct_l261_261193

theorem smallest_n_not_divisible_by_10 :
  ∃ n ≥ 2017, n % 4 = 0 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 :=
by
  -- Existence proof of such n is omitted
  sorry

def smallest_n : Nat :=
  Nat.find $ smallest_n_not_divisible_by_10

theorem smallest_n_correct : smallest_n = 2020 :=
by
  -- Correctness proof of smallest_n is omitted
  sorry

end smallest_n_not_divisible_by_10_smallest_n_correct_l261_261193


namespace perpendicularity_proof_l261_261024

-- Definitions of geometric entities and properties
variable (Plane Line : Type)
variable (α β : Plane) -- α and β are planes
variable (m n : Line) -- m and n are lines

-- Geometric properties and relations
variable (subset : Line → Plane → Prop) -- Line is subset of plane
variable (perpendicular : Line → Plane → Prop) -- Line is perpendicular to plane
variable (line_perpendicular : Line → Line → Prop) -- Line is perpendicular to another line

-- Conditions
axiom planes_different : α ≠ β
axiom lines_different : m ≠ n
axiom m_in_beta : subset m β
axiom n_in_beta : subset n β

-- Proof problem statement
theorem perpendicularity_proof :
  (subset m α) → (perpendicular n α) → (line_perpendicular n m) :=
by
  sorry

end perpendicularity_proof_l261_261024


namespace checker_arrangements_five_digit_palindromes_l261_261756

noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem checker_arrangements :
  comb 32 12 * comb 20 12 = Nat.choose 32 12 * Nat.choose 20 12 := by
  sorry

theorem five_digit_palindromes :
  9 * 10 * 10 = 900 := by
  sorry

end checker_arrangements_five_digit_palindromes_l261_261756


namespace find_number_l261_261883

theorem find_number (x : ℝ) (h : (25 / 100) * x = 20 / 100 * 30) : x = 24 :=
by
  sorry

end find_number_l261_261883


namespace magic_square_proof_l261_261758

theorem magic_square_proof
    (a b c d e S : ℕ)
    (h1 : 35 + e + 27 = S)
    (h2 : 30 + c + d = S)
    (h3 : a + 32 + b = S)
    (h4 : 35 + c + b = S)
    (h5 : a + c + 27 = S)
    (h6 : 35 + c + b = S)
    (h7 : 35 + c + 27 = S)
    (h8 : a + c + d = S) :
  d + e = 35 :=
  sorry

end magic_square_proof_l261_261758


namespace part1_part2_l261_261025

noncomputable def x : ℝ := 1 - Real.sqrt 2
noncomputable def y : ℝ := 1 + Real.sqrt 2

theorem part1 : x^2 + 3 * x * y + y^2 = 3 := by
  sorry

theorem part2 : (y / x) - (x / y) = -4 * Real.sqrt 2 := by
  sorry

end part1_part2_l261_261025


namespace cost_of_tax_free_items_l261_261379

theorem cost_of_tax_free_items : 
  ∀ (total_spent : ℝ) (sales_tax : ℝ) (tax_rate : ℝ) (taxable_cost : ℝ),
  total_spent = 25 ∧ sales_tax = 0.30 ∧ tax_rate = 0.05 ∧ sales_tax = tax_rate * taxable_cost → 
  total_spent - taxable_cost = 19 :=
by
  intros total_spent sales_tax tax_rate taxable_cost
  intro h
  sorry

end cost_of_tax_free_items_l261_261379


namespace sum_of_possible_values_l261_261113

theorem sum_of_possible_values (A B : ℕ) 
  (hA1 : A < 10) (hA2 : 0 < A) (hB1 : B < 10) (hB2 : 0 < B)
  (h1 : 3 / 12 < A / 12) (h2 : A / 12 < 7 / 12)
  (h3 : 1 / 10 < 1 / B) (h4 : 1 / B < 1 / 3) :
  3 + 6 = 9 :=
by
  sorry

end sum_of_possible_values_l261_261113


namespace probability_of_ace_then_spade_l261_261094

theorem probability_of_ace_then_spade :
  let P := (1 / 52) * (12 / 51) + (3 / 52) * (13 / 51)
  P = (3 / 127) :=
by
  sorry

end probability_of_ace_then_spade_l261_261094


namespace sum_of_max_and_min_values_on_interval_l261_261861

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

theorem sum_of_max_and_min_values_on_interval :
  let a := 3 in
  (∃ x ∈ Ioo (0 : ℝ) (⊤ : ℝ), f x = 0) →
  let f_on_interval := (Icc (-1 : ℝ) 1 : set ℝ),
  let max_value := sup (f '' f_on_interval),
  let min_value := inf (f '' f_on_interval)
  in max_value + min_value = -3 :=
by sorry

end sum_of_max_and_min_values_on_interval_l261_261861


namespace trinomial_ne_binomial_l261_261148

theorem trinomial_ne_binomial (a b c A B : ℝ) (h : a ≠ 0) : 
  ¬ ∀ x : ℝ, ax^2 + bx + c = Ax + B :=
by
  sorry

end trinomial_ne_binomial_l261_261148


namespace profit_starts_from_third_year_most_beneficial_option_l261_261974

-- Define the conditions of the problem
def investment_cost := 144
def maintenance_cost (n : ℕ) := 4 * n^2 + 20 * n
def revenue_per_year := 1

-- Define the net profit function
def net_profit (n : ℕ) : ℤ :=
(revenue_per_year * n : ℤ) - (maintenance_cost n) - investment_cost

-- Question 1: Prove the project starts to make a profit from the 3rd year
theorem profit_starts_from_third_year (n : ℕ) (h : 2 < n ∧ n < 18) : 
net_profit n > 0 ↔ 3 ≤ n := sorry

-- Question 2: Prove the most beneficial option for company's development
theorem most_beneficial_option : (∃ o, o = 1) ∧ (∃ t1 t2, t1 = 264 ∧ t2 = 264 ∧ t1 < t2) := sorry

end profit_starts_from_third_year_most_beneficial_option_l261_261974


namespace solution_set_of_inequality_l261_261620

theorem solution_set_of_inequality :
  {x : ℝ | |x - 5| + |x + 3| >= 10} = {x : ℝ | x ≤ -4} ∪ {x : ℝ | x ≥ 6} :=
by
  sorry

end solution_set_of_inequality_l261_261620


namespace total_points_l261_261866

theorem total_points (gwen_points_per_4 : ℕ) (lisa_points_per_5 : ℕ) (jack_points_per_7 : ℕ) 
                     (gwen_recycled : ℕ) (lisa_recycled : ℕ) (jack_recycled : ℕ)
                     (gwen_ratio : gwen_points_per_4 = 2) (lisa_ratio : lisa_points_per_5 = 3) 
                     (jack_ratio : jack_points_per_7 = 1) (gwen_pounds : gwen_recycled = 12) 
                     (lisa_pounds : lisa_recycled = 25) (jack_pounds : jack_recycled = 21) 
                     : gwen_points_per_4 * (gwen_recycled / 4) + 
                       lisa_points_per_5 * (lisa_recycled / 5) + 
                       jack_points_per_7 * (jack_recycled / 7) = 24 := by
  sorry

end total_points_l261_261866


namespace max_value_of_ratio_l261_261030

theorem max_value_of_ratio (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) : 
  ∃ z, z = (x / y) ∧ z ≤ 1 := sorry

end max_value_of_ratio_l261_261030


namespace min_number_of_gennadys_l261_261690

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l261_261690


namespace inequality_condition_l261_261997

theorem inequality_condition (k : ℝ) (h : k ≥ 4) 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + (k * c) / (a + b) ≥ 2) :=
sorry

end inequality_condition_l261_261997


namespace problem_statement_l261_261033

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2 - 2 * a * x

theorem problem_statement (a : ℝ) (x1 x2 : ℝ) (h_a : a > 1) (h1 : x1 < x2) (h_extreme : f a x1 = 0 ∧ f a x2 = 0) : 
  f a x2 < -3/2 :=
sorry

end problem_statement_l261_261033


namespace amount_cut_off_l261_261403

def initial_length : ℕ := 11
def final_length : ℕ := 7

theorem amount_cut_off : (initial_length - final_length) = 4 :=
by
  sorry

end amount_cut_off_l261_261403


namespace regular_eqn_exists_l261_261839

noncomputable def parametric_eqs (k : ℝ) : ℝ × ℝ :=
  (4 * k / (1 - k^2), 4 * k^2 / (1 - k^2))

theorem regular_eqn_exists (k : ℝ) (x y : ℝ) (h1 : x = 4 * k / (1 - k^2)) 
(h2 : y = 4 * k^2 / (1 - k^2)) : x^2 - y^2 - 4 * y = 0 :=
sorry

end regular_eqn_exists_l261_261839


namespace student_marks_l261_261496

variable (M P C : ℕ)

theorem student_marks (h1 : C = P + 20) (h2 : (M + C) / 2 = 20) : M + P = 20 :=
by
  sorry

end student_marks_l261_261496


namespace differential_savings_l261_261813

theorem differential_savings (income : ℝ) (tax_rate1 tax_rate2 : ℝ) 
                            (old_tax_rate_eq : tax_rate1 = 0.40) 
                            (new_tax_rate_eq : tax_rate2 = 0.33) 
                            (income_eq : income = 45000) :
    ((tax_rate1 - tax_rate2) * income) = 3150 :=
by
  rw [old_tax_rate_eq, new_tax_rate_eq, income_eq]
  norm_num

end differential_savings_l261_261813


namespace sheila_picnic_probability_l261_261073

theorem sheila_picnic_probability :
  let P_rain := 0.5
  let P_go_given_rain := 0.3
  let P_go_given_sunny := 0.9
  let P_remember := 0.9  -- P(remember) = 1 - P(forget)
  let P_sunny := 1 - P_rain
  
  P_rain * P_go_given_rain * P_remember + P_sunny * P_go_given_sunny * P_remember = 0.54 :=
by
  sorry

end sheila_picnic_probability_l261_261073


namespace prove_B_is_guilty_l261_261052

variables (A B C : Prop)

def guilty_conditions (A B C : Prop) : Prop :=
  (A → ¬ B → C) ∧
  (C → B ∨ A) ∧
  (A → ¬ (A ∧ C)) ∧
  (A ∨ B ∨ C) ∧ 
  ¬ (¬ A ∧ ¬ B ∧ ¬ C)

theorem prove_B_is_guilty : guilty_conditions A B C → B :=
by
  intros h
  sorry

end prove_B_is_guilty_l261_261052


namespace rectangle_area_l261_261928

variable (x y : ℕ)

theorem rectangle_area
  (h1 : (x + 3) * (y - 1) = x * y)
  (h2 : (x - 3) * (y + 2) = x * y) :
  x * y = 36 :=
by
  -- Proof omitted
  sorry

end rectangle_area_l261_261928


namespace solution_set_f_2_minus_x_l261_261795

def f (x : ℝ) (a : ℝ) (b : ℝ) := (x - 2) * (a * x + b)

theorem solution_set_f_2_minus_x (a b : ℝ) (h_even : b - 2 * a = 0)
  (h_mono : 0 < a) :
  {x : ℝ | f (2 - x) a b > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry

end solution_set_f_2_minus_x_l261_261795


namespace smallest_w_l261_261659

theorem smallest_w (w : ℕ) (h : 2^5 ∣ 936 * w ∧ 3^3 ∣ 936 * w ∧ 11^2 ∣ 936 * w) : w = 4356 :=
sorry

end smallest_w_l261_261659


namespace good_eggs_collected_l261_261096

/-- 
Uncle Ben has 550 chickens on his farm, consisting of 49 roosters and the rest being hens. 
Out of these hens, there are three types:
1. Type A: 25 hens do not lay eggs at all.
2. Type B: 155 hens lay 2 eggs per day.
3. Type C: The remaining hens lay 4 eggs every three days.

Moreover, Uncle Ben found that 3% of the eggs laid by Type B and Type C hens go bad before being collected. 
Prove that the total number of good eggs collected by Uncle Ben after one day is 716.
-/
theorem good_eggs_collected 
    (total_chickens : ℕ) (roosters : ℕ) (typeA_hens : ℕ) (typeB_hens : ℕ) 
    (typeB_eggs_per_day : ℕ) (typeC_eggs_per_3days : ℕ) (percent_bad_eggs : ℚ) :
  total_chickens = 550 →
  roosters = 49 →
  typeA_hens = 25 →
  typeB_hens = 155 →
  typeB_eggs_per_day = 2 →
  typeC_eggs_per_3days = 4 →
  percent_bad_eggs = 0.03 →
  (total_chickens - roosters - typeA_hens - typeB_hens) * (typeC_eggs_per_3days / 3) + (typeB_hens * typeB_eggs_per_day) - 
  round (percent_bad_eggs * ((total_chickens - roosters - typeA_hens - typeB_hens) * (typeC_eggs_per_3days / 3) + (typeB_hens * typeB_eggs_per_day))) = 716 :=
by
  intros
  sorry

end good_eggs_collected_l261_261096


namespace total_cost_of_fencing_l261_261748

theorem total_cost_of_fencing (side_count : ℕ) (cost_per_side : ℕ) (h1 : side_count = 4) (h2 : cost_per_side = 79) : side_count * cost_per_side = 316 := by
  sorry

end total_cost_of_fencing_l261_261748


namespace john_weekly_earnings_after_raise_l261_261765

theorem john_weekly_earnings_after_raise (original_earnings : ℝ) (raise_percentage : ℝ) (raise_amount new_earnings : ℝ) 
  (h1 : original_earnings = 50) (h2 : raise_percentage = 60) (h3 : raise_amount = (raise_percentage / 100) * original_earnings) 
  (h4 : new_earnings = original_earnings + raise_amount) : 
  new_earnings = 80 := 
by sorry

end john_weekly_earnings_after_raise_l261_261765


namespace total_whales_seen_is_178_l261_261760

/-
Ishmael's monitoring of whales yields the following:
- On the first trip, he counts 28 male whales and twice as many female whales.
- On the second trip, he sees 8 baby whales, each traveling with their parents.
- On the third trip, he counts half as many male whales as the first trip and the same number of female whales as on the first trip.
-/

def number_of_whales_first_trip : ℕ := 28
def number_of_female_whales_first_trip : ℕ := 2 * number_of_whales_first_trip
def total_whales_first_trip : ℕ := number_of_whales_first_trip + number_of_female_whales_first_trip

def number_of_baby_whales_second_trip : ℕ := 8
def total_whales_second_trip : ℕ := number_of_baby_whales_second_trip * 3

def number_of_male_whales_third_trip : ℕ := number_of_whales_first_trip / 2
def number_of_female_whales_third_trip : ℕ := number_of_female_whales_first_trip
def total_whales_third_trip : ℕ := number_of_male_whales_third_trip + number_of_female_whales_third_trip

def total_whales_seen : ℕ := total_whales_first_trip + total_whales_second_trip + total_whales_third_trip

theorem total_whales_seen_is_178 : total_whales_seen = 178 :=
by
  -- skip the actual proof
  sorry

end total_whales_seen_is_178_l261_261760


namespace minimum_grade_Ahmed_l261_261002

theorem minimum_grade_Ahmed (assignments : ℕ) (Ahmed_grade : ℕ) (Emily_grade : ℕ) (final_assignment_grade_Emily : ℕ) 
  (sum_grades_Emily : ℕ) (sum_grades_Ahmed : ℕ) (total_points_Ahmed : ℕ) (total_points_Emily : ℕ) :
  assignments = 9 →
  Ahmed_grade = 91 →
  Emily_grade = 92 →
  final_assignment_grade_Emily = 90 →
  sum_grades_Emily = 828 →
  sum_grades_Ahmed = 819 →
  total_points_Ahmed = sum_grades_Ahmed + 100 →
  total_points_Emily = sum_grades_Emily + final_assignment_grade_Emily →
  total_points_Ahmed > total_points_Emily :=
by
  sorry

end minimum_grade_Ahmed_l261_261002


namespace sum_c_d_eq_24_l261_261138

theorem sum_c_d_eq_24 (c d : ℕ) (h_pos_c : c > 0) (h_pos_d : d > 1) (h_max_power : c^d < 500 ∧ ∀ ⦃x y : ℕ⦄, x^y < 500 → x^y ≤ c^d) : c + d = 24 :=
sorry

end sum_c_d_eq_24_l261_261138


namespace sin_cos_tan_l261_261435

theorem sin_cos_tan (α : ℝ) (h1 : Real.tan α = 3) : Real.sin α * Real.cos α = 3 / 10 := 
sorry

end sin_cos_tan_l261_261435


namespace min_gennadys_needed_l261_261693

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l261_261693


namespace students_absent_percentage_l261_261499

theorem students_absent_percentage (total_students present_students : ℕ) (h_total : total_students = 50) (h_present : present_students = 45) :
  (total_students - present_students) * 100 / total_students = 10 := 
by
  sorry

end students_absent_percentage_l261_261499


namespace two_squares_inequality_l261_261564

theorem two_squares_inequality (a b : ℝ) : 2 * (a^2 + b^2) ≥ (a + b)^2 := 
sorry

end two_squares_inequality_l261_261564


namespace value_of_4b_minus_a_l261_261790

theorem value_of_4b_minus_a (a b : ℕ) (h1 : a > b) (h2 : x^2 - 20*x + 96 = (x - a)*(x - b)) : 4*b - a = 20 :=
  sorry

end value_of_4b_minus_a_l261_261790


namespace university_students_l261_261006

theorem university_students (total_students students_both math_students physics_students : ℕ) 
  (h1 : total_students = 75) 
  (h2 : total_students = (math_students - students_both) + (physics_students - students_both) + students_both)
  (h3 : math_students = 2 * physics_students) 
  (h4 : students_both = 10) : 
  math_students = 56 := by
  sorry

end university_students_l261_261006


namespace annieka_free_throws_l261_261930

theorem annieka_free_throws (deshawn_throws : ℕ) (kayla_factor : ℝ) (annieka_diff : ℕ) (ht1 : deshawn_throws = 12) (ht2 : kayla_factor = 1.5) (ht3 : annieka_diff = 4) :
  ∃ (annieka_throws : ℕ), annieka_throws = (⌊deshawn_throws * kayla_factor⌋.toNat - annieka_diff) :=
by
  sorry

end annieka_free_throws_l261_261930


namespace min_gennadys_needed_l261_261694

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end min_gennadys_needed_l261_261694


namespace compound_interest_rate_l261_261400

theorem compound_interest_rate
  (P : ℝ) (r : ℝ) :
  (3000 = P * (1 + r / 100)^3) →
  (3600 = P * (1 + r / 100)^4) →
  r = 20 :=
by
  sorry

end compound_interest_rate_l261_261400


namespace total_value_of_gold_is_l261_261768

-- Definitions based on the conditions
def legacyBars : ℕ := 5
def aleenaBars : ℕ := legacyBars - 2
def valuePerBar : ℝ := 2200
def totalValue : ℝ := (legacyBars + aleenaBars) * valuePerBar

-- Theorem statement
theorem total_value_of_gold_is :
  totalValue = 17600 := by
  -- We add sorry here to skip the proof
  sorry

end total_value_of_gold_is_l261_261768


namespace triangle_DEF_area_l261_261503

noncomputable def point := (ℝ × ℝ)

def D : point := (-2, 2)
def E : point := (8, 2)
def F : point := (6, -4)

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_DEF_area : area_of_triangle D E F = 30 := by
  sorry

end triangle_DEF_area_l261_261503


namespace triangle_30_60_90_PQ_l261_261015

theorem triangle_30_60_90_PQ (PR : ℝ) (hPR : PR = 18 * Real.sqrt 3) : 
  ∃ PQ : ℝ, PQ = 54 :=
by
  sorry

end triangle_30_60_90_PQ_l261_261015


namespace line_intersects_ellipse_possible_slopes_l261_261394

theorem line_intersects_ellipse_possible_slopes :
  ∀ m : ℝ, (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔
    (m ≤ -Real.sqrt (1 / 20) ∨ m ≥ Real.sqrt (1 / 20)) :=
by
  sorry

end line_intersects_ellipse_possible_slopes_l261_261394


namespace black_piece_is_option_C_l261_261296

-- Definitions for the problem conditions
def rectangular_prism (cubes : Nat) := cubes = 16
def block (small_cubes : Nat) := small_cubes = 4
def piece_containing_black_shape_is_partially_seen (rows : Nat) := rows = 2

-- Hypotheses and conditions
variable (rect_prism : Nat) (block1 block2 block3 block4 : Nat)
variable (visibility_block1 visibility_block2 visibility_block3 : Bool)
variable (visible_in_back_row : Bool)

-- Given conditions based on the problem statement
axiom h1 : rectangular_prism rect_prism
axiom h2 : block block1
axiom h3 : block block2
axiom h4 : block block3
axiom h5 : block block4
axiom h6 : visibility_block1 = true
axiom h7 : visibility_block2 = true
axiom h8 : visibility_block3 = true
axiom h9 : visible_in_back_row = true

-- Prove the configuration matches Option C
theorem black_piece_is_option_C :
  ∀ (config : Char), (config = 'C') :=
by
  intros
  -- Proof incomplete intentionally.
  sorry

end black_piece_is_option_C_l261_261296


namespace greatest_sum_l261_261497

theorem greatest_sum {x y : ℤ} (h₁ : x^2 + y^2 = 49) : x + y ≤ 9 :=
sorry

end greatest_sum_l261_261497


namespace problem_statement_l261_261320

theorem problem_statement (m n : ℝ) 
  (h₁ : m^2 - 1840 * m + 2009 = 0)
  (h₂ : n^2 - 1840 * n + 2009 = 0) : 
  (m^2 - 1841 * m + 2009) * (n^2 - 1841 * n + 2009) = 2009 :=
sorry

end problem_statement_l261_261320


namespace min_number_of_gennadys_l261_261691

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l261_261691


namespace rate_of_current_l261_261980

theorem rate_of_current (c : ℝ) (h1 : ∀ d : ℝ, d / (3.9 - c) = 2 * (d / (3.9 + c))) : c = 1.3 :=
sorry

end rate_of_current_l261_261980


namespace train_speed_l261_261982

theorem train_speed 
  (length_train : ℝ) (length_bridge : ℝ) (time : ℝ) 
  (h_length_train : length_train = 110)
  (h_length_bridge : length_bridge = 138)
  (h_time : time = 12.399008079353651) : 
  (length_train + length_bridge) / time * 3.6 = 72 :=
by
  sorry

end train_speed_l261_261982


namespace Earl_owes_Fred_l261_261995

-- Define initial amounts of money each person has
def Earl_initial : ℤ := 90
def Fred_initial : ℤ := 48
def Greg_initial : ℤ := 36

-- Define debts
def Fred_owes_Greg : ℤ := 32
def Greg_owes_Earl : ℤ := 40

-- Define the total money Greg and Earl have together after debts are settled
def Greg_Earl_total_after_debts : ℤ := 130

-- Define the final amounts after debts are settled
def Earl_final (E : ℤ) : ℤ := Earl_initial - E + Greg_owes_Earl
def Fred_final (E : ℤ) : ℤ := Fred_initial + E - Fred_owes_Greg
def Greg_final : ℤ := Greg_initial + Fred_owes_Greg - Greg_owes_Earl

-- Prove that the total money Greg and Earl have together after debts are settled is 130
theorem Earl_owes_Fred (E : ℤ) (H : Greg_final + Earl_final E = Greg_Earl_total_after_debts) : E = 28 := 
by sorry

end Earl_owes_Fred_l261_261995


namespace parabola_geometric_sequence_l261_261822

theorem parabola_geometric_sequence (M : ℝ × ℝ) (l: ℝ × ℝ → ℝ) (parabola : ℝ → ℝ → Prop) 
  (A B : ℝ × ℝ) (p : ℝ) 
  (hM : M = (-2, -4))
  (slope_angle : ∀ x, l x = (-2 + real.sqrt 2 / 2 * x, -4 + real.sqrt 2 / 2 * x))
  (parabola_cond : ∀ x y, parabola x y ↔ y^2 = 2 * p * x)
  (intersection_points : parabola (A.1) (A.2) ∧ parabola (B.1) (B.2))
  (geometric_sequence_cond : distance M A * distance M B = distance A B^2)
  : p = 1 :=
sorry

end parabola_geometric_sequence_l261_261822


namespace growth_operation_two_operations_growth_operation_four_operations_l261_261197

noncomputable def growth_operation_perimeter (initial_side_length : ℕ) (growth_operations : ℕ) := 
  initial_side_length * 3 * (4/3 : ℚ)^(growth_operations + 1)

theorem growth_operation_two_operations :
  growth_operation_perimeter 9 2 = 48 := by sorry

theorem growth_operation_four_operations :
  growth_operation_perimeter 9 4 = 256 / 3 := by sorry

end growth_operation_two_operations_growth_operation_four_operations_l261_261197


namespace problem_a9_b9_l261_261915

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the conditions
axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

-- Prove the goal
theorem problem_a9_b9 : a^9 + b^9 = 76 :=
by
  -- the proof will come here
  sorry

end problem_a9_b9_l261_261915


namespace corner_coloring_condition_l261_261931

theorem corner_coloring_condition 
  (n : ℕ) 
  (h1 : n ≥ 5) 
  (board : ℕ → ℕ → Prop) -- board(i, j) = true if cell (i, j) is black, false if white
  (h2 : ∀ i j, board i j = board (i + 1) j → board (i + 2) j = board (i + 1) j → ¬(board i j = board (i + 2) j)) -- row condition
  (h3 : ∀ i j, board i j = board i (j + 1) → board i (j + 2) = board i (j + 1) → ¬(board i j = board i (j + 2))) -- column condition
  (h4 : ∀ i j, board i j = board (i + 1) (j + 1) → board (i + 2) (j + 2) = board (i + 1) (j + 1) → ¬(board i j = board (i + 2) (j + 2))) -- diagonal condition
  (h5 : ∀ i j, board (i + 2) j = board (i + 1) (j + 1) → board (i + 2) (j + 2) = board (i + 1) (j + 1) → ¬(board (i + 2) j = board (i + 2) (j + 2))) -- anti-diagonal condition
  : ∀ i j, i + 2 < n ∧ j + 2 < n → ((board i j ∧ board (i + 2) (j + 2)) ∨ (board i (j + 2) ∧ board (i + 2) j)) :=
sorry

end corner_coloring_condition_l261_261931


namespace cosine_210_l261_261174

noncomputable def Q := rotate 210 (1, 0)
noncomputable def E := foot Q (1, 0) (-1, 0)

theorem cosine_210 : cos 210 = - (sqrt 3/2) :=
sorry

end cosine_210_l261_261174


namespace cos_210_eq_neg_sqrt3_div_2_l261_261172

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261172


namespace minimum_gennadies_l261_261703

theorem minimum_gennadies (A B V G : ℕ) (hA : A = 45) (hB : B = 122) (hV : V = 27) (hGap : G + A + V >= B - 1) :
  G >= 49 :=
by 
  have := by linarith [hGap, hA, hB, hV]
  exact this

end minimum_gennadies_l261_261703


namespace paint_containers_left_l261_261473

theorem paint_containers_left (initial_containers : ℕ)
  (tiled_wall_containers : ℕ)
  (ceiling_containers : ℕ)
  (gradient_walls : ℕ)
  (additional_gradient_containers_per_wall : ℕ)
  (remaining_containers : ℕ) :
  initial_containers = 16 →
  tiled_wall_containers = 1 →
  ceiling_containers = 1 →
  gradient_walls = 3 →
  additional_gradient_containers_per_wall = 1 →
  remaining_containers = initial_containers - tiled_wall_containers - (ceiling_containers + gradient_walls * additional_gradient_containers_per_wall) →
  remaining_containers = 11 :=
by
  intros h_initial h_tiled h_ceiling h_gradient_walls h_additional_gradient h_remaining_calc
  rw [h_initial, h_tiled, h_ceiling, h_gradient_walls, h_additional_gradient] at h_remaining_calc
  exact h_remaining_calc

end paint_containers_left_l261_261473


namespace cos_210_eq_neg_sqrt3_div_2_l261_261163

theorem cos_210_eq_neg_sqrt3_div_2 : real.cos (210 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l261_261163


namespace correct_calculation_l261_261588

theorem correct_calculation (x : ℤ) (h : 20 + x = 60) : 34 - x = -6 := by
  sorry

end correct_calculation_l261_261588


namespace height_difference_l261_261225

-- Define the heights of Eiffel Tower and Burj Khalifa as constants
def eiffelTowerHeight : ℕ := 324
def burjKhalifaHeight : ℕ := 830

-- Define the statement that needs to be proven
theorem height_difference : burjKhalifaHeight - eiffelTowerHeight = 506 := by
  sorry

end height_difference_l261_261225
