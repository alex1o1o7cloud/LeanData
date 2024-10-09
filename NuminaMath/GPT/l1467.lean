import Mathlib

namespace john_gym_hours_l1467_146734

theorem john_gym_hours :
  (2 * (1 + 1/3)) + (2 * (1 + 1/2)) + (1.5 + 3/4) = 7.92 :=
by
  sorry

end john_gym_hours_l1467_146734


namespace find_num_large_envelopes_l1467_146736

def numLettersInSmallEnvelopes : Nat := 20
def totalLetters : Nat := 150
def totalLettersInMediumLargeEnvelopes := totalLetters - numLettersInSmallEnvelopes -- 130
def lettersPerLargeEnvelope : Nat := 5
def lettersPerMediumEnvelope : Nat := 3
def numLargeEnvelopes (L : Nat) : Prop := 5 * L + 6 * L = totalLettersInMediumLargeEnvelopes

theorem find_num_large_envelopes : ∃ L : Nat, numLargeEnvelopes L ∧ L = 11 := by
  sorry

end find_num_large_envelopes_l1467_146736


namespace everyone_can_cross_l1467_146740

-- Define each agent
inductive Agent
| C   -- Princess Sonya
| K (i : Fin 8) -- Knights numbered 1 to 7

open Agent

-- Define friendships
def friends (a b : Agent) : Prop :=
  match a, b with
  | C, (K 4) => False
  | (K 4), C => False
  | _, _ => (∃ i : Fin 8, a = K i ∧ b = K (i+1)) ∨ (∃ i : Fin 7, a = K (i+1) ∧ b = K i) ∨ a = C ∨ b = C

-- Define the crossing conditions
def boatCanCarry : List Agent → Prop
| [a, b] => friends a b
| [a, b, c] => friends a b ∧ friends b c ∧ friends a c
| _ => False

-- The main statement to prove
theorem everyone_can_cross (agents : List Agent) (steps : List (List Agent)) :
  agents = [C, K 0, K 1, K 2, K 3, K 4, K 5, K 6, K 7] →
  (∀ step ∈ steps, boatCanCarry step) →
  (∃ final_state : List (List Agent), final_state = [[C, K 0, K 1, K 2, K 3, K 4, K 5, K 6, K 7]]) :=
by 
  -- The proof is omitted.
  sorry

end everyone_can_cross_l1467_146740


namespace remainder_2468135792_mod_101_l1467_146792

theorem remainder_2468135792_mod_101 : 
  2468135792 % 101 = 47 := 
sorry

end remainder_2468135792_mod_101_l1467_146792


namespace range_of_b_l1467_146708

theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) → (5 < b ∧ b < 7) :=
sorry

end range_of_b_l1467_146708


namespace line_equation_through_point_and_area_l1467_146791

theorem line_equation_through_point_and_area (k b : ℝ) :
  (∃ (P : ℝ × ℝ), P = (4/3, 2)) ∧
  (∀ (A B : ℝ × ℝ), A = (- b / k, 0) ∧ B = (0, b) → 
  1 / 2 * abs ((- b / k) * b) = 6) →
  (y = k * x + b ↔ (y = -3/4 * x + 3 ∨ y = -3 * x + 6)) :=
by
  sorry

end line_equation_through_point_and_area_l1467_146791


namespace bus_rent_proof_l1467_146781

theorem bus_rent_proof (r1 r2 : ℝ) (r1_rent_eq : r1 + 2 * r2 = 2800) (r2_mult : r2 = 1.25 * r1) :
  r1 = 800 ∧ r2 = 1000 := 
by
  sorry

end bus_rent_proof_l1467_146781


namespace solve_for_y_l1467_146703

noncomputable def find_angle_y : Prop :=
  let AB_CD_are_straight_lines : Prop := True
  let angle_AXB : ℕ := 70
  let angle_BXD : ℕ := 40
  let angle_CYX : ℕ := 100
  let angle_YXZ := 180 - angle_AXB - angle_BXD
  let angle_XYZ := 180 - angle_CYX
  let y := 180 - angle_YXZ - angle_XYZ
  y = 30

theorem solve_for_y : find_angle_y :=
by
  trivial

end solve_for_y_l1467_146703


namespace other_root_of_quadratic_l1467_146778

theorem other_root_of_quadratic (k : ℝ) (h : -2 * 1 = -2) (h_eq : x^2 + k * x - 2 = 0) :
  1 * -2 = -2 :=
by
  sorry

end other_root_of_quadratic_l1467_146778


namespace gratuity_percentage_l1467_146766

open Real

theorem gratuity_percentage (num_bankers num_clients : ℕ) (total_bill per_person_cost : ℝ) 
    (h1 : num_bankers = 4) (h2 : num_clients = 5) (h3 : total_bill = 756) 
    (h4 : per_person_cost = 70) : 
    ((total_bill - (num_bankers + num_clients) * per_person_cost) / 
     ((num_bankers + num_clients) * per_person_cost)) = 0.2 :=
by 
  sorry

end gratuity_percentage_l1467_146766


namespace second_hand_travel_distance_l1467_146779

theorem second_hand_travel_distance (radius : ℝ) (time_minutes : ℕ) (C : ℝ) (distance : ℝ) 
    (h1 : radius = 8) (h2 : time_minutes = 45) 
    (h3 : C = 2 * Real.pi * radius) 
    (h4 : distance = time_minutes * C)
    : distance = 720 * Real.pi := 
by 
  rw [h1, h2, h3] at *
  sorry

end second_hand_travel_distance_l1467_146779


namespace complex_division_l1467_146726

theorem complex_division (i : ℂ) (h : i * i = -1) : 3 / (1 - i) ^ 2 = (3 / 2) * i :=
by
  sorry

end complex_division_l1467_146726


namespace probability_of_selecting_GEARS_letter_l1467_146735

def bag : List Char := ['A', 'L', 'G', 'E', 'B', 'R', 'A', 'S']
def target_word : List Char := ['G', 'E', 'A', 'R', 'S']

theorem probability_of_selecting_GEARS_letter :
  (6 : ℚ) / 8 = 3 / 4 :=
by
  sorry

end probability_of_selecting_GEARS_letter_l1467_146735


namespace solve_inequality_l1467_146784

theorem solve_inequality : { x : ℝ | 3 * x^2 - 1 > 13 - 5 * x } = { x : ℝ | x < -7 ∨ x > 2 } :=
by
  sorry

end solve_inequality_l1467_146784


namespace FGH_supermarkets_total_l1467_146704

theorem FGH_supermarkets_total 
  (us_supermarkets : ℕ)
  (ca_supermarkets : ℕ)
  (h1 : us_supermarkets = 41)
  (h2 : us_supermarkets = ca_supermarkets + 22) :
  us_supermarkets + ca_supermarkets = 60 :=
by
  sorry

end FGH_supermarkets_total_l1467_146704


namespace number_of_valid_3_digit_numbers_l1467_146785

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end number_of_valid_3_digit_numbers_l1467_146785


namespace solve_y_l1467_146727

theorem solve_y : ∀ y : ℚ, (9 * y^2 + 8 * y - 2 = 0) ∧ (27 * y^2 + 62 * y - 8 = 0) → y = 1 / 9 :=
by
  intro y h
  cases h
  sorry

end solve_y_l1467_146727


namespace leak_empty_time_l1467_146794

theorem leak_empty_time (P L : ℝ) (h1 : P = 1 / 6) (h2 : P - L = 1 / 12) : 1 / L = 12 :=
by
  -- Proof to be provided
  sorry

end leak_empty_time_l1467_146794


namespace max_x_minus_2y_l1467_146746

open Real

theorem max_x_minus_2y (x y : ℝ) (h : (x^2) / 16 + (y^2) / 9 = 1) : 
  ∃ t : ℝ, t = 2 * sqrt 13 ∧ x - 2 * y = t := 
sorry

end max_x_minus_2y_l1467_146746


namespace royal_children_count_l1467_146788

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l1467_146788


namespace log_property_l1467_146774

theorem log_property (x : ℝ) (h₁ : Real.log x > 0) (h₂ : x > 1) : x > Real.exp 1 := by 
  sorry

end log_property_l1467_146774


namespace parabola_chord_midpoint_l1467_146764

/-- 
If the point (3, 1) is the midpoint of a chord of the parabola y^2 = 2px, 
and the slope of the line containing this chord is 2, then p = 2. 
-/
theorem parabola_chord_midpoint (p : ℝ) :
    (∃ (m : ℝ), (m = 2) ∧ ∀ (x y : ℝ), y = 2 * x - 5 → y^2 = 2 * p * x → 
        ((x1 = 0 ∧ y1 = 0 ∧ x2 = 6 ∧ y2 = 6) → 
            (x1 + x2 = 6) ∧ (y1 + y2 = 2) ∧ (p = 2))) :=
sorry

end parabola_chord_midpoint_l1467_146764


namespace parallel_lines_eq_a_l1467_146793

theorem parallel_lines_eq_a (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a + 1) * x - a * y = 0) → (a = -3/2 ∨ a = 0) :=
by sorry

end parallel_lines_eq_a_l1467_146793


namespace B_div_A_75_l1467_146705

noncomputable def find_ratio (A B : ℝ) (x : ℝ) :=
  (A / (x + 3) + B / (x * (x - 9)) = (x^2 - 3*x + 15) / (x * (x + 3) * (x - 9)))

theorem B_div_A_75 :
  ∀ (A B : ℝ), (∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 9 → find_ratio A B x) → 
  B/A = 7.5 :=
by
  sorry

end B_div_A_75_l1467_146705


namespace triangle_shading_probability_l1467_146741

theorem triangle_shading_probability (n_triangles: ℕ) (n_shaded: ℕ) (h1: n_triangles > 4) (h2: n_shaded = 4) (h3: n_triangles = 10) :
  (n_shaded / n_triangles) = 2 / 5 := 
by
  sorry

end triangle_shading_probability_l1467_146741


namespace slope_range_PA2_l1467_146753

-- Define the given conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

def A1 : ℝ × ℝ := (-2, 0)
def A2 : ℝ × ℝ := (2, 0)
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.fst P.snd

-- Define the range of the slope of line PA1
def slope_range_PA1 (k_PA1 : ℝ) : Prop := -2 ≤ k_PA1 ∧ k_PA1 ≤ -1

-- Main theorem
theorem slope_range_PA2 (x0 y0 k_PA1 k_PA2 : ℝ) (h1 : on_ellipse (x0, y0)) (h2 : slope_range_PA1 k_PA1) :
  k_PA1 = (y0 / (x0 + 2)) →
  k_PA2 = (y0 / (x0 - 2)) →
  - (3 / 4) = k_PA1 * k_PA2 →
  (3 / 8) ≤ k_PA2 ∧ k_PA2 ≤ (3 / 4) :=
by
  sorry

end slope_range_PA2_l1467_146753


namespace min_value_a_plus_2b_l1467_146754

theorem min_value_a_plus_2b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b + 2 * a * b = 8) :
  a + 2 * b ≥ 4 :=
sorry

end min_value_a_plus_2b_l1467_146754


namespace mean_of_squares_eq_l1467_146749

noncomputable def sum_of_squares (n : ℕ) : ℚ := (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def arithmetic_mean_of_squares (n : ℕ) : ℚ := sum_of_squares n / n

theorem mean_of_squares_eq (n : ℕ) (h : n ≠ 0) : arithmetic_mean_of_squares n = ((n + 1) * (2 * n + 1)) / 6 :=
by
  sorry

end mean_of_squares_eq_l1467_146749


namespace polynomial_integer_roots_k_zero_l1467_146722

theorem polynomial_integer_roots_k_zero :
  (∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℤ, (x - a) * (x - b) * (x - c) = x^3 - x + 0) ∨
  (∀ x : ℤ, (x - a) * (x - b) * (x - c) = x^3 - x + k)) →
  k = 0 :=
sorry

end polynomial_integer_roots_k_zero_l1467_146722


namespace probability_of_selecting_double_l1467_146762

-- Define the conditions and the question
def total_integers : ℕ := 13

def number_of_doubles : ℕ := total_integers

def total_pairings : ℕ := 
  (total_integers * (total_integers + 1)) / 2

def probability_double : ℚ := 
  number_of_doubles / total_pairings

-- Statement to be proved 
theorem probability_of_selecting_double : 
  probability_double = 1/7 := 
sorry

end probability_of_selecting_double_l1467_146762


namespace harrison_annual_croissant_expenditure_l1467_146712

-- Define the different costs and frequency of croissants.
def cost_regular_croissant : ℝ := 3.50
def cost_almond_croissant : ℝ := 5.50
def cost_chocolate_croissant : ℝ := 4.50
def cost_ham_cheese_croissant : ℝ := 6.00

def frequency_regular_croissant : ℕ := 52
def frequency_almond_croissant : ℕ := 52
def frequency_chocolate_croissant : ℕ := 52
def frequency_ham_cheese_croissant : ℕ := 26

-- Calculate annual expenditure for each type of croissant.
def annual_expenditure (cost : ℝ) (frequency : ℕ) : ℝ :=
  cost * frequency

-- Total annual expenditure on croissants.
def total_annual_expenditure : ℝ :=
  annual_expenditure cost_regular_croissant frequency_regular_croissant +
  annual_expenditure cost_almond_croissant frequency_almond_croissant +
  annual_expenditure cost_chocolate_croissant frequency_chocolate_croissant +
  annual_expenditure cost_ham_cheese_croissant frequency_ham_cheese_croissant

-- The theorem to prove.
theorem harrison_annual_croissant_expenditure :
  total_annual_expenditure = 858 := by
  sorry

end harrison_annual_croissant_expenditure_l1467_146712


namespace range_of_f_l1467_146771

noncomputable def f (x : ℝ) : ℝ := 3^(x - 2)

theorem range_of_f : Set.Icc 1 9 = {y : ℝ | ∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ f x = y} :=
by
  sorry

end range_of_f_l1467_146771


namespace range_of_a_l1467_146737

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → (x^2 + 2*x + a) / x > 0) ↔ a > -3 :=
by
  sorry

end range_of_a_l1467_146737


namespace muffin_cost_l1467_146787

theorem muffin_cost (m : ℝ) :
  let fruit_cup_cost := 3
  let francis_cost := 2 * m + 2 * fruit_cup_cost
  let kiera_cost := 2 * m + 1 * fruit_cup_cost
  let total_cost := 17
  (francis_cost + kiera_cost = total_cost) → m = 2 :=
by
  intro h
  sorry

end muffin_cost_l1467_146787


namespace kim_hard_correct_l1467_146701

-- Definitions
def points_per_easy := 2
def points_per_average := 3
def points_per_hard := 5
def easy_correct := 6
def average_correct := 2
def total_points := 38

-- Kim's correct answers in the hard round is 4
theorem kim_hard_correct : (total_points - (easy_correct * points_per_easy + average_correct * points_per_average)) / points_per_hard = 4 :=
by
  sorry

end kim_hard_correct_l1467_146701


namespace total_surface_area_hemisphere_l1467_146750

theorem total_surface_area_hemisphere (A : ℝ) (r : ℝ) : (A = 100 * π) → (r = 10) → (2 * π * r^2 + A = 300 * π) :=
by
  intro hA hr
  sorry

end total_surface_area_hemisphere_l1467_146750


namespace sum_of_repeating_decimals_l1467_146761

/-- Definitions of the repeating decimals as real numbers. --/
def x : ℝ := 0.3 -- This actually represents 0.\overline{3} in Lean
def y : ℝ := 0.04 -- This actually represents 0.\overline{04} in Lean
def z : ℝ := 0.005 -- This actually represents 0.\overline{005} in Lean

/-- The theorem stating that the sum of these repeating decimals is a specific fraction. --/
theorem sum_of_repeating_decimals : x + y + z = (14 : ℝ) / 37 := 
by 
  sorry -- Placeholder for the proof

end sum_of_repeating_decimals_l1467_146761


namespace min_value_fraction_sum_l1467_146752

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_collinear : 3 * a + 2 * b = 1)

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_collinear : 3 * a + 2 * b = 1) : 
  (3 / a + 1 / b) = 11 + 6 * Real.sqrt 2 :=
by
  sorry

end min_value_fraction_sum_l1467_146752


namespace composite_2011_2014_composite_2012_2015_l1467_146786

theorem composite_2011_2014 :
  let N := 2011 * 2012 * 2013 * 2014 + 1
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ N = a * b := 
by
  let N := 2011 * 2012 * 2013 * 2014 + 1
  sorry
  
theorem composite_2012_2015 :
  let N := 2012 * 2013 * 2014 * 2015 + 1
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ N = a * b := 
by
  let N := 2012 * 2013 * 2014 * 2015 + 1
  sorry

end composite_2011_2014_composite_2012_2015_l1467_146786


namespace total_amount_spent_l1467_146751

theorem total_amount_spent (tax_paid : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) (total_spent : ℝ) :
  tax_paid = 30 → tax_rate = 0.06 → tax_free_cost = 19.7 →
  total_spent = 30 / 0.06 + 19.7 :=
by
  -- Definitions for assumptions
  intro h1 h2 h3
  -- Skip the proof here
  sorry

end total_amount_spent_l1467_146751


namespace f_of_6_l1467_146796

noncomputable def f (u : ℝ) : ℝ := 
  let x := (u + 2) / 4
  x^3 - x + 2

theorem f_of_6 : f 6 = 8 :=
by
  sorry

end f_of_6_l1467_146796


namespace competition_results_l1467_146759

variables (x : ℝ) (freq1 freq3 freq4 freq5 freq2 : ℝ)

/-- Axiom: Given frequencies of groups and total frequency, determine the total number of participants and the probability of an excellent score -/
theorem competition_results :
  freq1 = 0.30 ∧
  freq3 = 0.15 ∧
  freq4 = 0.10 ∧
  freq5 = 0.05 ∧
  freq2 = 40 / x ∧
  (freq1 + freq2 + freq3 + freq4 + freq5 = 1) ∧
  (x * freq2 = 40) →
  x = 100 ∧ (freq4 + freq5 = 0.15) := sorry

end competition_results_l1467_146759


namespace michael_payment_correct_l1467_146797

def suit_price : ℕ := 430
def suit_discount : ℕ := 100
def shoes_price : ℕ := 190
def shoes_discount : ℕ := 30
def shirt_price : ℕ := 80
def tie_price: ℕ := 50
def combined_discount : ℕ := (shirt_price + tie_price) * 20 / 100

def total_price_paid : ℕ :=
    suit_price - suit_discount + shoes_price - shoes_discount + (shirt_price + tie_price - combined_discount)

theorem michael_payment_correct :
    total_price_paid = 594 :=
by
    -- skipping the proof
    sorry

end michael_payment_correct_l1467_146797


namespace robin_earns_30_percent_more_than_erica_l1467_146742

variable (E R C : ℝ)

theorem robin_earns_30_percent_more_than_erica
  (h1 : C = 1.60 * E)
  (h2 : C = 1.23076923076923077 * R) :
  R = 1.30 * E :=
by
  sorry

end robin_earns_30_percent_more_than_erica_l1467_146742


namespace percentage_correct_l1467_146724

noncomputable def part : ℝ := 172.8
noncomputable def whole : ℝ := 450.0
noncomputable def percentage (part whole : ℝ) := (part / whole) * 100

theorem percentage_correct : percentage part whole = 38.4 := by
  sorry

end percentage_correct_l1467_146724


namespace intersection_of_M_and_complementN_l1467_146782

def UniversalSet := Set ℝ
def setM : Set ℝ := {-1, 0, 1, 3}
def setN : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def complementSetN : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_M_and_complementN :
  setM ∩ complementSetN = {0, 1} :=
sorry

end intersection_of_M_and_complementN_l1467_146782


namespace calculate_value_l1467_146715

def f (x : ℝ) : ℝ := 9 - x
def g (x : ℝ) : ℝ := x - 9

theorem calculate_value : g (f 15) = -15 := by
  sorry

end calculate_value_l1467_146715


namespace domain_of_f_i_l1467_146744

variable (f : ℝ → ℝ)

theorem domain_of_f_i (h : ∀ x, -1 ≤ x + 1 ∧ x + 1 ≤ 1) : ∀ x, -2 ≤ x ∧ x ≤ 0 :=
by
  intro x
  specialize h x
  sorry

end domain_of_f_i_l1467_146744


namespace shorter_piece_length_l1467_146702

theorem shorter_piece_length : ∃ (x : ℕ), (x + (x + 2) = 30) ∧ x = 14 :=
by {
  sorry
}

end shorter_piece_length_l1467_146702


namespace baker_number_of_eggs_l1467_146777

theorem baker_number_of_eggs (flour cups eggs : ℕ) (h1 : eggs = 3 * (flour / 2)) (h2 : flour = 6) : eggs = 9 :=
by
  sorry

end baker_number_of_eggs_l1467_146777


namespace cube_root_floor_equality_l1467_146760

theorem cube_root_floor_equality (n : ℕ) : 
  (⌊(n : ℝ)^(1/3) + (n+1 : ℝ)^(1/3)⌋ : ℝ) = ⌊(8*n + 3 : ℝ)^(1/3)⌋ :=
sorry

end cube_root_floor_equality_l1467_146760


namespace union_M_N_eq_M_l1467_146700

-- Define set M
def M : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

-- Define set N
def N : Set ℝ := { y | ∃ x : ℝ, y = Real.log (x - 1) }

-- Statement to prove that M ∪ N = M
theorem union_M_N_eq_M : M ∪ N = M := by
  sorry

end union_M_N_eq_M_l1467_146700


namespace total_dogs_at_center_l1467_146773

structure PawsitiveTrainingCenter :=
  (sit : Nat)
  (stay : Nat)
  (fetch : Nat)
  (roll_over : Nat)
  (sit_stay : Nat)
  (sit_fetch : Nat)
  (sit_roll_over : Nat)
  (stay_fetch : Nat)
  (stay_roll_over : Nat)
  (fetch_roll_over : Nat)
  (sit_stay_fetch : Nat)
  (sit_stay_roll_over : Nat)
  (sit_fetch_roll_over : Nat)
  (stay_fetch_roll_over : Nat)
  (all_four : Nat)
  (none : Nat)

def PawsitiveTrainingCenter.total_dogs (p : PawsitiveTrainingCenter) : Nat :=
  p.sit + p.stay + p.fetch + p.roll_over
  - p.sit_stay - p.sit_fetch - p.sit_roll_over - p.stay_fetch - p.stay_roll_over - p.fetch_roll_over
  + p.sit_stay_fetch + p.sit_stay_roll_over + p.sit_fetch_roll_over + p.stay_fetch_roll_over
  - p.all_four + p.none

theorem total_dogs_at_center (p : PawsitiveTrainingCenter) (h : 
  p.sit = 60 ∧
  p.stay = 35 ∧
  p.fetch = 45 ∧
  p.roll_over = 40 ∧
  p.sit_stay = 20 ∧
  p.sit_fetch = 15 ∧
  p.sit_roll_over = 10 ∧
  p.stay_fetch = 5 ∧
  p.stay_roll_over = 8 ∧
  p.fetch_roll_over = 6 ∧
  p.sit_stay_fetch = 4 ∧
  p.sit_stay_roll_over = 3 ∧
  p.sit_fetch_roll_over = 2 ∧
  p.stay_fetch_roll_over = 1 ∧
  p.all_four = 2 ∧
  p.none = 12
) : PawsitiveTrainingCenter.total_dogs p = 135 := by
  sorry

end total_dogs_at_center_l1467_146773


namespace initial_time_for_train_l1467_146720

theorem initial_time_for_train (S : ℝ)
  (length_initial : ℝ := 12 * 15)
  (length_detached : ℝ := 11 * 15)
  (time_detached : ℝ := 16.5)
  (speed_constant : S = length_detached / time_detached) :
  (length_initial / S = 18) :=
by
  sorry

end initial_time_for_train_l1467_146720


namespace avg_words_per_hour_l1467_146721

theorem avg_words_per_hour (words hours : ℝ) (h_words : words = 40000) (h_hours : hours = 80) :
  words / hours = 500 :=
by
  rw [h_words, h_hours]
  norm_num
  done

end avg_words_per_hour_l1467_146721


namespace find_length_of_sheet_l1467_146756

noncomputable section

-- Axioms regarding the conditions
def width_of_sheet : ℝ := 36       -- The width of the metallic sheet is 36 meters
def side_of_square : ℝ := 7        -- The side length of the square cut off from each corner is 7 meters
def volume_of_box : ℝ := 5236      -- The volume of the resulting box is 5236 cubic meters

-- Define the length of the metallic sheet as L
def length_of_sheet (L : ℝ) : Prop :=
  let new_length := L - 2 * side_of_square
  let new_width := width_of_sheet - 2 * side_of_square
  let height := side_of_square
  volume_of_box = new_length * new_width * height

-- The condition to prove
theorem find_length_of_sheet : ∃ L : ℝ, length_of_sheet L ∧ L = 48 :=
by
  sorry

end find_length_of_sheet_l1467_146756


namespace factor_example_solve_equation_example_l1467_146772

-- Factorization proof problem
theorem factor_example (m a b : ℝ) : 
  (m * a ^ 2 - 4 * m * b ^ 2) = m * (a + 2 * b) * (a - 2 * b) :=
sorry

-- Solving the equation proof problem
theorem solve_equation_example (x : ℝ) (hx1: x ≠ 2) (hx2: x ≠ 0) : 
  (1 / (x - 2) = 3 / x) ↔ x = 3 :=
sorry

end factor_example_solve_equation_example_l1467_146772


namespace proof1_proof2a_proof2b_l1467_146714

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (S : ℝ)

-- Given conditions for Question 1
def condition1 := (a = 3 * Real.cos C ∧ b = 1)

-- Proof statement for Question 1
theorem proof1 : condition1 a b C → Real.tan C = 2 * Real.tan B :=
by sorry

-- Given conditions for Question 2a
def condition2a := (S = 1 / 2 * a * b * Real.sin C ∧ S = 1 / 2 * 3 * Real.cos C * 1 * Real.sin C)

-- Proof statement for Question 2a
theorem proof2a : condition2a a b C S → Real.cos (2 * B) = 3 / 5 :=
by sorry

-- Given conditions for Question 2b
def condition2b := (c = Real.sqrt 10 / 2)

-- Proof statement for Question 2b
theorem proof2b : condition1 a b C → condition2b c → Real.cos (2 * B) = 3 / 5 :=
by sorry

end proof1_proof2a_proof2b_l1467_146714


namespace trumpet_cost_l1467_146732

/-
  Conditions:
  1. Cost of the music tool: $9.98
  2. Cost of the song book: $4.14
  3. Total amount Joan spent at the music store: $163.28

  Prove that the cost of the trumpet is $149.16
-/

theorem trumpet_cost :
  let c_mt := 9.98
  let c_sb := 4.14
  let t_sp := 163.28
  let c_trumpet := t_sp - (c_mt + c_sb)
  c_trumpet = 149.16 :=
by
  sorry

end trumpet_cost_l1467_146732


namespace completing_the_square_l1467_146765

theorem completing_the_square (x : ℝ) : x^2 + 8 * x + 9 = 0 → (x + 4)^2 = 7 :=
by 
  intro h
  sorry

end completing_the_square_l1467_146765


namespace route_Y_is_quicker_l1467_146733

noncomputable def route_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

def route_X_distance : ℝ := 8
def route_X_speed : ℝ := 40

def route_Y_total_distance : ℝ := 7
def route_Y_construction_distance : ℝ := 1
def route_Y_construction_speed : ℝ := 20
def route_Y_regular_speed_distance : ℝ := 6
def route_Y_regular_speed : ℝ := 50

noncomputable def route_X_time : ℝ :=
  route_time route_X_distance route_X_speed * 60  -- converting to minutes

noncomputable def route_Y_time : ℝ :=
  (route_time route_Y_regular_speed_distance route_Y_regular_speed +
  route_time route_Y_construction_distance route_Y_construction_speed) * 60 -- converting to minutes

theorem route_Y_is_quicker : route_X_time - route_Y_time = 1.8 :=
  by
    sorry

end route_Y_is_quicker_l1467_146733


namespace shortest_is_Bob_l1467_146723

variable {Person : Type}
variable [LinearOrder Person]

variable (Amy Bob Carla Dan Eric : Person)

-- Conditions
variable (h1 : Amy > Carla)
variable (h2 : Dan < Eric)
variable (h3 : Dan > Bob)
variable (h4 : Eric < Carla)

theorem shortest_is_Bob : ∀ p : Person, p = Bob :=
by
  intro p
  sorry

end shortest_is_Bob_l1467_146723


namespace interest_rate_proof_l1467_146718
noncomputable def interest_rate_B (P : ℝ) (rA : ℝ) (t : ℝ) (gain_B : ℝ) : ℝ := 
  (P * rA * t + gain_B) / (P * t)

theorem interest_rate_proof
  (P : ℝ := 3500)
  (rA : ℝ := 0.10)
  (t : ℝ := 3)
  (gain_B : ℝ := 210) :
  interest_rate_B P rA t gain_B = 0.12 :=
sorry

end interest_rate_proof_l1467_146718


namespace greatest_root_of_f_one_is_root_of_f_l1467_146748

def f (x : ℝ) : ℝ := 16 * x^6 - 15 * x^4 + 4 * x^2 - 1

theorem greatest_root_of_f :
  ∀ x : ℝ, f x = 0 → x ≤ 1 :=
sorry

theorem one_is_root_of_f :
  f 1 = 0 :=
sorry

end greatest_root_of_f_one_is_root_of_f_l1467_146748


namespace range_of_a_l1467_146747

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := 
sorry

end range_of_a_l1467_146747


namespace solve_triplet_l1467_146713

theorem solve_triplet (x y z : ℕ) (h : 2^x * 3^y + 1 = 7^z) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 2) :=
 by sorry

end solve_triplet_l1467_146713


namespace geom_seq_ratio_l1467_146710

variable {a_1 r : ℚ}
variable {S : ℕ → ℚ}

-- The sum of the first n terms of a geometric sequence
def geom_sum (a_1 r : ℚ) (n : ℕ) : ℚ := a_1 * (1 - r^n) / (1 - r)

-- Given conditions
axiom Sn_def : ∀ n, S n = geom_sum a_1 r n
axiom condition : S 10 / S 5 = 1 / 2

-- Theorem to prove
theorem geom_seq_ratio (h : r ≠ 1) : S 15 / S 5 = 3 / 4 :=
by
  -- proof omitted
  sorry

end geom_seq_ratio_l1467_146710


namespace shaded_region_area_l1467_146757

theorem shaded_region_area (r : ℝ) (n : ℕ) (shaded_area : ℝ) (h_r : r = 3) (h_n : n = 6) :
  shaded_area = 27 * Real.pi - 54 := by
  sorry

end shaded_region_area_l1467_146757


namespace symmetric_diff_cardinality_l1467_146798

theorem symmetric_diff_cardinality (X Y : Finset ℤ) 
  (hX : X.card = 8) 
  (hY : Y.card = 10) 
  (hXY : (X ∩ Y).card = 6) : 
  (X \ Y ∪ Y \ X).card = 6 := 
by
  sorry

end symmetric_diff_cardinality_l1467_146798


namespace simple_interest_rate_l1467_146728

variables (P R T SI : ℝ)

theorem simple_interest_rate :
  T = 10 →
  SI = (2 / 5) * P →
  SI = (P * R * T) / 100 →
  R = 4 :=
by
  intros hT hSI hFormula
  sorry

end simple_interest_rate_l1467_146728


namespace average_value_of_items_in_loot_box_l1467_146730

-- Definitions as per the given conditions
def cost_per_loot_box : ℝ := 5
def total_spent : ℝ := 40
def total_loss : ℝ := 12

-- Proving the average value of items inside each loot box
theorem average_value_of_items_in_loot_box :
  (total_spent - total_loss) / (total_spent / cost_per_loot_box) = 3.50 := by
  sorry

end average_value_of_items_in_loot_box_l1467_146730


namespace math_problem_modulo_l1467_146716

theorem math_problem_modulo :
    (245 * 15 - 20 * 8 + 5) % 17 = 1 := 
by
  sorry

end math_problem_modulo_l1467_146716


namespace ratio_new_radius_l1467_146739

theorem ratio_new_radius (r R h : ℝ) (h₀ : π * r^2 * h = 6) (h₁ : π * R^2 * h = 186) : R / r = Real.sqrt 31 :=
by
  sorry

end ratio_new_radius_l1467_146739


namespace incorrect_calculation_l1467_146707

theorem incorrect_calculation (a : ℝ) : (2 * a) ^ 3 ≠ 6 * a ^ 3 :=
by {
  sorry
}

end incorrect_calculation_l1467_146707


namespace value_of_k_l1467_146729

theorem value_of_k (m n k : ℝ) (h1 : 3 ^ m = k) (h2 : 5 ^ n = k) (h3 : 1 / m + 1 / n = 2) : k = Real.sqrt 15 :=
  sorry

end value_of_k_l1467_146729


namespace prime_solution_l1467_146763

theorem prime_solution (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ (p = 2 ∨ p = 3 ∨ p = 7) :=
by
  sorry

end prime_solution_l1467_146763


namespace area_of_field_l1467_146706

theorem area_of_field (L W A : ℕ) (h₁ : L = 20) (h₂ : L + 2 * W = 80) : A = 600 :=
by
  sorry

end area_of_field_l1467_146706


namespace money_total_l1467_146709

theorem money_total (s j m : ℝ) (h1 : 3 * s = 80) (h2 : j / 2 = 70) (h3 : 2.5 * m = 100) :
  s + j + m = 206.67 :=
sorry

end money_total_l1467_146709


namespace flour_more_than_sugar_l1467_146770

-- Define the conditions.
def sugar_needed : ℕ := 9
def total_flour_needed : ℕ := 14
def salt_needed : ℕ := 40
def flour_already_added : ℕ := 4

-- Define the target proof statement.
theorem flour_more_than_sugar :
  (total_flour_needed - flour_already_added) - sugar_needed = 1 :=
by
  -- sorry is used here to skip the proof.
  sorry

end flour_more_than_sugar_l1467_146770


namespace find_ax5_plus_by5_l1467_146789

variable (a b x y : ℝ)

-- Conditions
axiom h1 : a * x + b * y = 3
axiom h2 : a * x^2 + b * y^2 = 7
axiom h3 : a * x^3 + b * y^3 = 16
axiom h4 : a * x^4 + b * y^4 = 42

-- Theorem (what we need to prove)
theorem find_ax5_plus_by5 : a * x^5 + b * y^5 = 20 :=
sorry

end find_ax5_plus_by5_l1467_146789


namespace negative_square_inequality_l1467_146711

theorem negative_square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end negative_square_inequality_l1467_146711


namespace an_gt_bn_l1467_146790

theorem an_gt_bn (a b : ℕ → ℕ) (h₁ : a 1 = 2013) (h₂ : ∀ n, a (n + 1) = 2013^(a n))
                            (h₃ : b 1 = 1) (h₄ : ∀ n, b (n + 1) = 2013^(2012 * (b n))) :
  ∀ n, a n > b n := 
sorry

end an_gt_bn_l1467_146790


namespace abs_inequality_solution_l1467_146799

theorem abs_inequality_solution (x : ℝ) : 
  abs (2 * x - 1) < abs x + 1 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end abs_inequality_solution_l1467_146799


namespace sin_double_angle_tan_double_angle_l1467_146725

-- Step 1: Define the first problem in Lean 4.
theorem sin_double_angle (α : ℝ) (h1 : Real.sin α = 12 / 13) (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  Real.sin (2 * α) = -120 / 169 := 
sorry

-- Step 2: Define the second problem in Lean 4.
theorem tan_double_angle (α : ℝ) (h1 : Real.tan α = 1 / 2) :
  Real.tan (2 * α) = 4 / 3 := 
sorry

end sin_double_angle_tan_double_angle_l1467_146725


namespace triangle_angle_B_eq_60_l1467_146745

theorem triangle_angle_B_eq_60 {A B C : ℝ} (h1 : B = 2 * A) (h2 : C = 3 * A) (h3 : A + B + C = 180) : B = 60 :=
by sorry

end triangle_angle_B_eq_60_l1467_146745


namespace AndrewAge_l1467_146775

variable (a f g : ℚ)
axiom h1 : f = 8 * a
axiom h2 : g = 3 * f
axiom h3 : g - a = 72

theorem AndrewAge : a = 72 / 23 :=
by
  sorry

end AndrewAge_l1467_146775


namespace min_area_after_fold_l1467_146780

theorem min_area_after_fold (A : ℝ) (h_A : A = 1) (c : ℝ) (h_c : 0 ≤ c ∧ c ≤ 1) : 
  ∃ (m : ℝ), m = min_area ∧ m = 2 / 3 :=
by
  sorry

end min_area_after_fold_l1467_146780


namespace general_term_arithmetic_sequence_l1467_146769

variable {α : Type*}
variables (a_n a : ℕ → ℕ) (d a_1 a_2 a_3 a_4 n : ℕ)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a_n : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a_n (n + 1) = a_n n + d

-- Define the inequality solution condition 
def inequality_solution_set (a_1 a_2 : ℕ) (x : ℕ) :=
  a_1 ≤ x ∧ x ≤ a_2

theorem general_term_arithmetic_sequence :
  arithmetic_sequence a_n d ∧ (d ≠ 0) ∧ 
  (∀ x, x^2 - a_3 * x + a_4 ≤ 0 ↔ inequality_solution_set a_1 a_2 x) →
  a_n = 2 * n :=
by
  sorry

end general_term_arithmetic_sequence_l1467_146769


namespace solve_for_y_l1467_146719

theorem solve_for_y (y : ℝ) (h : y + 81 / (y - 3) = -12) : y = -6 ∨ y = -3 :=
sorry

end solve_for_y_l1467_146719


namespace parabola_vertex_l1467_146783

theorem parabola_vertex (c d : ℝ) (h : ∀ x : ℝ, - x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ x ≥ 6)) :
  (∃ v : ℝ × ℝ, v = (5, 1)) :=
sorry

end parabola_vertex_l1467_146783


namespace positive_solution_of_x_l1467_146743

theorem positive_solution_of_x :
  ∃ x y z : ℝ, (x * y = 6 - 2 * x - 3 * y) ∧ (y * z = 6 - 4 * y - 2 * z) ∧ (x * z = 30 - 4 * x - 3 * z) ∧ x > 0 ∧ x = 3 :=
by
  sorry

end positive_solution_of_x_l1467_146743


namespace value_of_M_l1467_146758

theorem value_of_M (M : ℕ) : (32^3) * (16^3) = 2^M → M = 27 :=
by
  sorry

end value_of_M_l1467_146758


namespace initial_candy_bobby_l1467_146767

-- Definitions given conditions
def initial_candy (x : ℕ) : Prop :=
  (x + 42 = 70)

-- Theorem statement
theorem initial_candy_bobby : ∃ x : ℕ, initial_candy x ∧ x = 28 :=
by {
  sorry
}

end initial_candy_bobby_l1467_146767


namespace ratio_of_X_to_Y_l1467_146717

theorem ratio_of_X_to_Y (total_respondents : ℕ) (preferred_X : ℕ)
    (h_total : total_respondents = 250)
    (h_X : preferred_X = 200) :
    preferred_X / (total_respondents - preferred_X) = 4 := by
  sorry

end ratio_of_X_to_Y_l1467_146717


namespace initial_water_amount_gallons_l1467_146795

theorem initial_water_amount_gallons 
  (cup_capacity_oz : ℕ)
  (rows : ℕ)
  (chairs_per_row : ℕ)
  (water_left_oz : ℕ)
  (oz_per_gallon : ℕ)
  (total_gallons : ℕ)
  (h1 : cup_capacity_oz = 6)
  (h2 : rows = 5)
  (h3 : chairs_per_row = 10)
  (h4 : water_left_oz = 84)
  (h5 : oz_per_gallon = 128)
  (h6 : total_gallons = (rows * chairs_per_row * cup_capacity_oz + water_left_oz) / oz_per_gallon) :
  total_gallons = 3 := 
by sorry

end initial_water_amount_gallons_l1467_146795


namespace find_a_2b_3c_value_l1467_146768

-- Problem statement and conditions
theorem find_a_2b_3c_value (a b c : ℝ)
  (h : ∀ x : ℝ, (x < -1 ∨ abs (x - 10) ≤ 2) ↔ (x - a) * (x - b) / (x - c) ≤ 0)
  (h_ab : a < b) : a + 2 * b + 3 * c = 29 := 
sorry

end find_a_2b_3c_value_l1467_146768


namespace largest_constant_l1467_146731

theorem largest_constant (x y z : ℝ) : (x^2 + y^2 + z^2 + 3 ≥ 2 * (x + y + z)) :=
by
  sorry

end largest_constant_l1467_146731


namespace percentage_decrease_l1467_146738

theorem percentage_decrease (original_salary new_salary decreased_salary : ℝ) (p : ℝ) (D : ℝ) : 
  original_salary = 4000.0000000000005 →
  p = 10 →
  new_salary = original_salary * (1 + p/100) →
  decreased_salary = 4180 →
  decreased_salary = new_salary * (1 - D / 100) →
  D = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_decrease_l1467_146738


namespace train_truck_load_l1467_146755

variables (x y : ℕ)

def transport_equations (x y : ℕ) : Prop :=
  (2 * x + 5 * y = 120) ∧ (8 * x + 10 * y = 440)

def tonnage (x y : ℕ) : ℕ :=
  5 * x + 8 * y

theorem train_truck_load
  (x y : ℕ)
  (h : transport_equations x y) :
  tonnage x y = 282 :=
sorry

end train_truck_load_l1467_146755


namespace total_letters_received_l1467_146776

theorem total_letters_received 
  (Brother_received Greta_received Mother_received : ℕ) 
  (h1 : Greta_received = Brother_received + 10)
  (h2 : Brother_received = 40)
  (h3 : Mother_received = 2 * (Greta_received + Brother_received)) :
  Brother_received + Greta_received + Mother_received = 270 := 
sorry

end total_letters_received_l1467_146776
