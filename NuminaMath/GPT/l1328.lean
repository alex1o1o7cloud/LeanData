import Mathlib

namespace NUMINAMATH_GPT_dragons_at_meeting_l1328_132877

def dragon_meeting : Prop :=
  ∃ (x y : ℕ), 
    (2 * x + 7 * y = 26) ∧ 
    (x + y = 8)

theorem dragons_at_meeting : dragon_meeting :=
by
  sorry

end NUMINAMATH_GPT_dragons_at_meeting_l1328_132877


namespace NUMINAMATH_GPT_tan_sum_identity_l1328_132828

-- Definitions
def quadratic_eq (x : ℝ) : Prop := 6 * x^2 - 5 * x + 1 = 0
def tan_roots (α β : ℝ) : Prop := quadratic_eq (Real.tan α) ∧ quadratic_eq (Real.tan β)

-- Problem statement
theorem tan_sum_identity (α β : ℝ) (hαβ : tan_roots α β) : Real.tan (α + β) = 1 :=
sorry

end NUMINAMATH_GPT_tan_sum_identity_l1328_132828


namespace NUMINAMATH_GPT_fraction_to_decimal_l1328_132827

theorem fraction_to_decimal : (7 : ℚ) / 16 = 4375 / 10000 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1328_132827


namespace NUMINAMATH_GPT_directrix_of_parabola_l1328_132869

theorem directrix_of_parabola (y x : ℝ) (h_eq : y^2 = 8 * x) :
  x = -2 :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1328_132869


namespace NUMINAMATH_GPT_maximum_profit_l1328_132824

noncomputable def profit (x : ℝ) : ℝ :=
  5.06 * x - 0.15 * x^2 + 2 * (15 - x)

theorem maximum_profit : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 15 ∧ profit x = 45.6 :=
by
  sorry

end NUMINAMATH_GPT_maximum_profit_l1328_132824


namespace NUMINAMATH_GPT_unique_root_iff_l1328_132885

def has_unique_solution (a : ℝ) : Prop :=
  ∃ (x : ℝ), ∀ (y : ℝ), (a * y^2 + 2 * y - 1 = 0 ↔ y = x)

theorem unique_root_iff (a : ℝ) : has_unique_solution a ↔ (a = 0 ∨ a = 1) := 
sorry

end NUMINAMATH_GPT_unique_root_iff_l1328_132885


namespace NUMINAMATH_GPT_range_of_x2_plus_y2_l1328_132840

theorem range_of_x2_plus_y2 (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f x = -f (-x))
  (h_increasing : ∀ x y : ℝ, x < y → f x < f y)
  (x y : ℝ)
  (h_inequality : f (x^2 - 6 * x) + f (y^2 - 8 * y + 24) < 0) :
  16 < x^2 + y^2 ∧ x^2 + y^2 < 36 :=
sorry

end NUMINAMATH_GPT_range_of_x2_plus_y2_l1328_132840


namespace NUMINAMATH_GPT_norb_age_is_47_l1328_132881

section NorbAge

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def exactlyHalfGuessesTooLow (guesses : List ℕ) (age : ℕ) : Prop :=
  (guesses.filter (λ x => x < age)).length = (guesses.length / 2)

def oneGuessOffByTwo (guesses : List ℕ) (age : ℕ) : Prop :=
  guesses.any (λ x => x = age + 2 ∨ x = age - 2)

def validAge (guesses : List ℕ) (age : ℕ) : Prop :=
  exactlyHalfGuessesTooLow guesses age ∧ oneGuessOffByTwo guesses age ∧ isPrime age

theorem norb_age_is_47 : validAge [23, 29, 33, 35, 39, 41, 46, 48, 50, 54] 47 :=
sorry

end NorbAge

end NUMINAMATH_GPT_norb_age_is_47_l1328_132881


namespace NUMINAMATH_GPT_expression_for_f_in_positive_domain_l1328_132817

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def given_f (x : ℝ) : ℝ :=
  if x < 0 then 3 * Real.sin x + 4 * Real.cos x + 1 else 0 -- temp def for Lean proof

theorem expression_for_f_in_positive_domain (f : ℝ → ℝ) (h_odd : is_odd_function f)
  (h_neg : ∀ x : ℝ, x < 0 → f x = 3 * Real.sin x + 4 * Real.cos x + 1) :
  ∀ x : ℝ, x > 0 → f x = 3 * Real.sin x - 4 * Real.cos x - 1 :=
by
  intros x hx_pos
  sorry

end NUMINAMATH_GPT_expression_for_f_in_positive_domain_l1328_132817


namespace NUMINAMATH_GPT_max_sum_unit_hexagons_l1328_132821

theorem max_sum_unit_hexagons (k : ℕ) (hk : k ≥ 3) : 
  ∃ S, S = 6 + (3 * k - 9) * k * (k + 1) / 2 + (3 * (k^2 - 2)) * (k * (k + 1) * (2 * k + 1) / 6) / 6 ∧
       S = 3 * (k * k - 14 * k + 33 * k - 28) / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_unit_hexagons_l1328_132821


namespace NUMINAMATH_GPT_spring_festival_scientific_notation_l1328_132835

noncomputable def scientific_notation := (260000000: ℝ) = (2.6 * 10^8)

theorem spring_festival_scientific_notation : scientific_notation :=
by
  -- proof logic goes here
  sorry

end NUMINAMATH_GPT_spring_festival_scientific_notation_l1328_132835


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1328_132838

theorem solution_set_of_inequality (x : ℝ) : 
  (x ≠ 0 ∧ (x * (x - 1)) ≤ 0) ↔ 0 < x ∧ x ≤ 1 :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1328_132838


namespace NUMINAMATH_GPT_reciprocal_of_one_twentieth_l1328_132867

theorem reciprocal_of_one_twentieth : (1 / (1 / 20 : ℝ)) = 20 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_one_twentieth_l1328_132867


namespace NUMINAMATH_GPT_ratio_won_to_lost_l1328_132844

-- Define the total number of games and the number of games won
def total_games : Nat := 30
def games_won : Nat := 18

-- Define the number of games lost
def games_lost : Nat := total_games - games_won

-- Define the ratio of games won to games lost as a pair
def ratio : Nat × Nat := (games_won / Nat.gcd games_won games_lost, games_lost / Nat.gcd games_won games_lost)

-- The theorem to be proved
theorem ratio_won_to_lost : ratio = (3, 2) :=
  by
    -- Skipping the proof here
    sorry

end NUMINAMATH_GPT_ratio_won_to_lost_l1328_132844


namespace NUMINAMATH_GPT_problem1_problem2_l1328_132810

-- Define the base types and expressions
variables (x m : ℝ)

-- Proofs of the given expressions
theorem problem1 : (x^7 / x^3) * x^4 = x^8 :=
by sorry

theorem problem2 : m * m^3 + ((-m^2)^3 / m^2) = 0 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1328_132810


namespace NUMINAMATH_GPT_sum_x_y_eq_two_l1328_132802

theorem sum_x_y_eq_two (x y : ℝ) 
  (h1 : (x-1)^3 + 2003*(x-1) = -1) 
  (h2 : (y-1)^3 + 2003*(y-1) = 1) : 
  x + y = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_x_y_eq_two_l1328_132802


namespace NUMINAMATH_GPT_kate_retirement_fund_value_l1328_132889

theorem kate_retirement_fund_value 
(initial_value decrease final_value : ℝ) 
(h1 : initial_value = 1472)
(h2 : decrease = 12)
(h3 : final_value = initial_value - decrease) : 
final_value = 1460 := 
by
  sorry

end NUMINAMATH_GPT_kate_retirement_fund_value_l1328_132889


namespace NUMINAMATH_GPT_max_3cosx_4sinx_l1328_132814

theorem max_3cosx_4sinx (x : ℝ) : (3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧ (∃ y : ℝ, 3 * Real.cos y + 4 * Real.sin y = 5) :=
  sorry

end NUMINAMATH_GPT_max_3cosx_4sinx_l1328_132814


namespace NUMINAMATH_GPT_tower_surface_area_l1328_132812

noncomputable def total_visible_surface_area (volumes : List ℕ) : ℕ := sorry

theorem tower_surface_area :
  total_visible_surface_area [512, 343, 216, 125, 64, 27, 8, 1] = 882 :=
sorry

end NUMINAMATH_GPT_tower_surface_area_l1328_132812


namespace NUMINAMATH_GPT_tan_alpha_eq_3_l1328_132815

theorem tan_alpha_eq_3 (α : ℝ) (h1 : 0 < α ∧ α < (π / 2))
  (h2 : (Real.sin α)^2 + Real.cos ((π / 2) + 2 * α) = 3 / 10) : Real.tan α = 3 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_3_l1328_132815


namespace NUMINAMATH_GPT_no_square_remainder_2_infinitely_many_squares_remainder_3_l1328_132887

theorem no_square_remainder_2 :
  ∀ n : ℤ, (n * n) % 6 ≠ 2 :=
by sorry

theorem infinitely_many_squares_remainder_3 :
  ∀ k : ℤ, ∃ n : ℤ, n = 6 * k + 3 ∧ (n * n) % 6 = 3 :=
by sorry

end NUMINAMATH_GPT_no_square_remainder_2_infinitely_many_squares_remainder_3_l1328_132887


namespace NUMINAMATH_GPT_remainder_of_x_l1328_132830

theorem remainder_of_x (x : ℤ) (h : 2 * x - 3 = 7) : x % 2 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_of_x_l1328_132830


namespace NUMINAMATH_GPT_percentage_less_than_l1328_132811

theorem percentage_less_than (T F S : ℝ) 
  (hF : F = 0.70 * T) 
  (hS : S = 0.63 * T) : 
  ((T - S) / T) * 100 = 37 := 
by
  sorry

end NUMINAMATH_GPT_percentage_less_than_l1328_132811


namespace NUMINAMATH_GPT_move_line_up_l1328_132879

/-- Define the original line equation as y = 3x - 2 -/
def original_line (x : ℝ) : ℝ := 3 * x - 2

/-- Define the resulting line equation as y = 3x + 4 -/
def resulting_line (x : ℝ) : ℝ := 3 * x + 4

theorem move_line_up (x : ℝ) : resulting_line x = original_line x + 6 :=
by
  sorry

end NUMINAMATH_GPT_move_line_up_l1328_132879


namespace NUMINAMATH_GPT_probability_of_pink_flower_is_five_over_nine_l1328_132805

-- Definitions as per the conditions
def flowersInBagA := 9
def pinkFlowersInBagA := 3
def flowersInBagB := 9
def pinkFlowersInBagB := 7
def probChoosingBag := (1:ℚ) / 2

-- Definition of the probabilities
def probPinkFromA := pinkFlowersInBagA / flowersInBagA
def probPinkFromB := pinkFlowersInBagB / flowersInBagB

-- Total probability calculation using the law of total probability
def probPink := probPinkFromA * probChoosingBag + probPinkFromB * probChoosingBag

-- Statement to be proved
theorem probability_of_pink_flower_is_five_over_nine : probPink = (5:ℚ) / 9 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_pink_flower_is_five_over_nine_l1328_132805


namespace NUMINAMATH_GPT_system_of_equations_solution_exists_l1328_132886

theorem system_of_equations_solution_exists :
  ∃ (x y : ℝ), 
    (4 * x^2 + 8 * x * y + 16 * y^2 + 2 * x + 20 * y = -7) ∧
    (2 * x^2 - 16 * x * y + 8 * y^2 - 14 * x + 20 * y = -11) ∧
    (x = 1/2) ∧ (y = -3/4) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_exists_l1328_132886


namespace NUMINAMATH_GPT_math_problem_l1328_132836

theorem math_problem
  (m : ℕ) (h₁ : m = 8^126) :
  (m * 16) / 64 = 16^94 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1328_132836


namespace NUMINAMATH_GPT_dexter_total_cards_l1328_132807

theorem dexter_total_cards 
  (boxes_basketball : ℕ) 
  (cards_per_basketball_box : ℕ) 
  (boxes_football : ℕ) 
  (cards_per_football_box : ℕ) 
   (h1 : boxes_basketball = 15)
   (h2 : cards_per_basketball_box = 20)
   (h3 : boxes_football = boxes_basketball - 7)
   (h4 : cards_per_football_box = 25) 
   : boxes_basketball * cards_per_basketball_box + boxes_football * cards_per_football_box = 500 := by 
sorry

end NUMINAMATH_GPT_dexter_total_cards_l1328_132807


namespace NUMINAMATH_GPT_proof_inequality_l1328_132822

noncomputable def proof_problem (a b c d : ℝ) (h_ab : a * b + b * c + c * d + d * a = 1) : Prop :=
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1 / 3

theorem proof_inequality (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (h_ab : a * b + b * c + c * d + d * a = 1) : 
  proof_problem a b c d h_ab := 
by
  sorry

end NUMINAMATH_GPT_proof_inequality_l1328_132822


namespace NUMINAMATH_GPT_total_leaves_correct_l1328_132800

-- Definitions based on conditions
def basil_pots := 3
def rosemary_pots := 9
def thyme_pots := 6

def basil_leaves_per_pot := 4
def rosemary_leaves_per_pot := 18
def thyme_leaves_per_pot := 30

-- Calculate the total number of leaves
def total_leaves : Nat :=
  (basil_pots * basil_leaves_per_pot) +
  (rosemary_pots * rosemary_leaves_per_pot) +
  (thyme_pots * thyme_leaves_per_pot)

-- The statement to prove
theorem total_leaves_correct : total_leaves = 354 := by
  sorry

end NUMINAMATH_GPT_total_leaves_correct_l1328_132800


namespace NUMINAMATH_GPT_polygon_sides_l1328_132863

theorem polygon_sides (n : ℕ) (h₁ : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 156 = (180 * (n - 2)) / n) : n = 15 := sorry

end NUMINAMATH_GPT_polygon_sides_l1328_132863


namespace NUMINAMATH_GPT_general_term_formula_sum_of_geometric_sequence_l1328_132804

-- Definitions for the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = 3

def conditions_1 (a : ℕ → ℤ) : Prop :=
  a 2 + a 4 = 14

-- Definitions for the geometric sequence
def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ q, ∀ n, b (n + 1) = b n * q

def conditions_2 (a b : ℕ → ℤ) : Prop := 
  b 2 = a 2 ∧ 
  b 4 = a 6

-- The main theorem statements for part (I) and part (II)
theorem general_term_formula (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : conditions_1 a) : 
  ∀ n, a n = 3 * n - 2 := 
sorry

theorem sum_of_geometric_sequence (a b : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) - a n = 3)
  (h2 : a 2 + a 4 = 14)
  (h3 : b 2 = a 2)
  (h4 : b 4 = a 6)
  (h5 : geometric_sequence b) :
  ∃ (S7 : ℤ), S7 = 254 ∨ S7 = -86 :=
sorry

end NUMINAMATH_GPT_general_term_formula_sum_of_geometric_sequence_l1328_132804


namespace NUMINAMATH_GPT_solve_for_x_l1328_132888

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 5) * x = 2) : x = 17.5 :=
by
  -- Here we acknowledge the initial condition and conclusion without proving
  sorry

end NUMINAMATH_GPT_solve_for_x_l1328_132888


namespace NUMINAMATH_GPT_gcd_8154_8640_l1328_132831

theorem gcd_8154_8640 : Nat.gcd 8154 8640 = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_8154_8640_l1328_132831


namespace NUMINAMATH_GPT_solve_x_minus_y_l1328_132809

theorem solve_x_minus_y :
  (2 = 0.25 * x) → (2 = 0.1 * y) → (x - y = -12) :=
by
  sorry

end NUMINAMATH_GPT_solve_x_minus_y_l1328_132809


namespace NUMINAMATH_GPT_right_triangle_side_length_l1328_132847

theorem right_triangle_side_length (x : ℝ) (hx : x > 0) (h_area : (1 / 2) * x * (3 * x) = 108) :
  x = 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_right_triangle_side_length_l1328_132847


namespace NUMINAMATH_GPT_sum_of_intercepts_l1328_132823

theorem sum_of_intercepts (x y : ℝ) (h : 3 * x - 4 * y - 12 = 0) :
    (y = -3 ∧ x = 4) → x + y = 1 :=
by
  intro h'
  obtain ⟨hy, hx⟩ := h'
  rw [hy, hx]
  norm_num
  done

end NUMINAMATH_GPT_sum_of_intercepts_l1328_132823


namespace NUMINAMATH_GPT_find_n_l1328_132890

theorem find_n (x y : ℝ) (n : ℝ) (h1 : x / (2 * y) = 3 / n) (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) : n = 2 := by
  sorry

end NUMINAMATH_GPT_find_n_l1328_132890


namespace NUMINAMATH_GPT_probability_l1328_132861

def total_chips : ℕ := 15
def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4

def probability_of_different_colors : ℚ :=
  (blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips) +
  (red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips) +
  (yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips)

theorem probability : probability_of_different_colors = 148 / 225 :=
by
  unfold probability_of_different_colors
  sorry

end NUMINAMATH_GPT_probability_l1328_132861


namespace NUMINAMATH_GPT_katherine_fruit_count_l1328_132843

variables (apples pears bananas total_fruit : ℕ)

theorem katherine_fruit_count (h1 : apples = 4) 
  (h2 : pears = 3 * apples)
  (h3 : total_fruit = 21) 
  (h4 : total_fruit = apples + pears + bananas) : bananas = 5 := 
by sorry

end NUMINAMATH_GPT_katherine_fruit_count_l1328_132843


namespace NUMINAMATH_GPT_arithmetic_expression_l1328_132870

theorem arithmetic_expression : (5 * 7 - 6 + 2 * 12 + 2 * 6 + 7 * 3) = 86 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l1328_132870


namespace NUMINAMATH_GPT_ab_product_eq_four_l1328_132868

theorem ab_product_eq_four (a b : ℝ) (h1: 0 < a) (h2: 0 < b) 
  (h3: (1/2) * (4 / a) * (6 / b) = 3) : 
  a * b = 4 :=
by 
  sorry

end NUMINAMATH_GPT_ab_product_eq_four_l1328_132868


namespace NUMINAMATH_GPT_smallest_n_terminating_decimal_l1328_132846

-- Define the given condition: n + 150 must be expressible as 2^a * 5^b.
def has_terminating_decimal_property (n : ℕ) := ∃ a b : ℕ, n + 150 = 2^a * 5^b

-- We want to prove that the smallest n satisfying the property is 50.
theorem smallest_n_terminating_decimal :
  (∀ n : ℕ, n > 0 ∧ has_terminating_decimal_property n → n ≥ 50) ∧ (has_terminating_decimal_property 50) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_terminating_decimal_l1328_132846


namespace NUMINAMATH_GPT_male_student_number_l1328_132871

theorem male_student_number (year class_num student_num : ℕ) (h_year : year = 2011) (h_class : class_num = 6) (h_student : student_num = 23) : 
  (100000 * year + 1000 * class_num + 10 * student_num + 1 = 116231) :=
by
  sorry

end NUMINAMATH_GPT_male_student_number_l1328_132871


namespace NUMINAMATH_GPT_fraction_operation_correct_l1328_132862

theorem fraction_operation_correct {a b : ℝ} :
  (0.2 * a + 0.5 * b) ≠ 0 →
  (2 * a + 5 * b) ≠ 0 →
  (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_fraction_operation_correct_l1328_132862


namespace NUMINAMATH_GPT_inheritance_division_l1328_132816

variables {M P Q R : ℝ} {p q r : ℕ}

theorem inheritance_division (hP : P < 99 * (p : ℝ))
                             (hR : R > 10000 * (r : ℝ))
                             (hM : M = P + Q + R)
                             (hRichPoor : R ≥ P) : 
                             R ≥ 100 * P := 
sorry

end NUMINAMATH_GPT_inheritance_division_l1328_132816


namespace NUMINAMATH_GPT_train_crossing_time_l1328_132851

noncomputable def relative_speed_kmh (speed_train : ℕ) (speed_man : ℕ) : ℕ := speed_train + speed_man

noncomputable def kmh_to_mps (speed_kmh : ℕ) : ℝ := speed_kmh * 1000 / 3600

noncomputable def crossing_time (length_train : ℕ) (speed_train_kmh : ℕ) (speed_man_kmh : ℕ) : ℝ :=
  let relative_speed_kmh := relative_speed_kmh speed_train_kmh speed_man_kmh
  let relative_speed_mps := kmh_to_mps relative_speed_kmh
  length_train / relative_speed_mps

theorem train_crossing_time :
  crossing_time 210 25 2 = 28 :=
  by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1328_132851


namespace NUMINAMATH_GPT_buns_problem_l1328_132856

theorem buns_problem (N : ℕ) (x y u v : ℕ) 
  (h1 : 3 * x + 5 * y = 25)
  (h2 : 3 * u + 5 * v = 35)
  (h3 : x + y = N)
  (h4 : u + v = N) : 
  N = 7 := 
sorry

end NUMINAMATH_GPT_buns_problem_l1328_132856


namespace NUMINAMATH_GPT_digit_a_for_divisibility_l1328_132875

theorem digit_a_for_divisibility (a : ℕ) (h1 : (8 * 10^3 + 7 * 10^2 + 5 * 10 + a) % 6 = 0) : a = 4 :=
sorry

end NUMINAMATH_GPT_digit_a_for_divisibility_l1328_132875


namespace NUMINAMATH_GPT_find_sin_cos_of_perpendicular_vectors_l1328_132825

theorem find_sin_cos_of_perpendicular_vectors 
  (θ : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (h_a : a = (Real.sin θ, -2)) 
  (h_b : b = (1, Real.cos θ)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) 
  (h_theta_range : 0 < θ ∧ θ < Real.pi / 2) : 
  Real.sin θ = 2 * Real.sqrt 5 / 5 ∧ Real.cos θ = Real.sqrt 5 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_sin_cos_of_perpendicular_vectors_l1328_132825


namespace NUMINAMATH_GPT_least_possible_lcm_l1328_132859

-- Definitions of the least common multiples given the conditions
variable (a b c : ℕ)
variable (h₁ : Nat.lcm a b = 20)
variable (h₂ : Nat.lcm b c = 28)

-- The goal is to prove the least possible value of lcm(a, c) given the conditions
theorem least_possible_lcm (a b c : ℕ) (h₁ : Nat.lcm a b = 20) (h₂ : Nat.lcm b c = 28) : Nat.lcm a c = 35 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_lcm_l1328_132859


namespace NUMINAMATH_GPT_school_badminton_rackets_l1328_132866

theorem school_badminton_rackets :
  ∃ (x y : ℕ), x + y = 30 ∧ 50 * x + 40 * y = 1360 ∧ x = 16 ∧ y = 14 :=
by
  sorry

end NUMINAMATH_GPT_school_badminton_rackets_l1328_132866


namespace NUMINAMATH_GPT_noah_class_size_l1328_132893

theorem noah_class_size :
  ∀ n : ℕ, (n = 39 + 39 + 1) → n = 79 :=
by
  intro n
  intro h
  exact h

end NUMINAMATH_GPT_noah_class_size_l1328_132893


namespace NUMINAMATH_GPT_river_length_GSA_AWRA_l1328_132852

-- Define the main problem statement
noncomputable def river_length_estimate (GSA_length AWRA_length GSA_error AWRA_error error_prob : ℝ) : Prop :=
  (GSA_length = 402) ∧ (AWRA_length = 403) ∧ 
  (GSA_error = 0.5) ∧ (AWRA_error = 0.5) ∧ 
  (error_prob = 0.04) ∧ 
  (abs (402.5 - GSA_length) ≤ GSA_error) ∧ 
  (abs (402.5 - AWRA_length) ≤ AWRA_error) ∧ 
  (error_prob = 1 - (2 * 0.02))

-- The main theorem statement
theorem river_length_GSA_AWRA :
  river_length_estimate 402 403 0.5 0.5 0.04 :=
by
  sorry

end NUMINAMATH_GPT_river_length_GSA_AWRA_l1328_132852


namespace NUMINAMATH_GPT_edric_hourly_rate_l1328_132864

theorem edric_hourly_rate
  (monthly_salary : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (H1 : monthly_salary = 576)
  (H2 : hours_per_day = 8)
  (H3 : days_per_week = 6)
  (H4 : weeks_per_month = 4) :
  monthly_salary / weeks_per_month / days_per_week / hours_per_day = 3 := by
  sorry

end NUMINAMATH_GPT_edric_hourly_rate_l1328_132864


namespace NUMINAMATH_GPT_correct_option_is_B_l1328_132829

-- Define the Pythagorean theorem condition for right-angled triangles
def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Conditions given in the problem
def option_A : Prop := ¬is_right_angled_triangle 1 2 2
def option_B : Prop := is_right_angled_triangle 1 (Real.sqrt 3) 2
def option_C : Prop := ¬is_right_angled_triangle 4 5 6
def option_D : Prop := ¬is_right_angled_triangle 1 1 (Real.sqrt 3)

-- The formal proof problem statement
theorem correct_option_is_B : option_A ∧ option_B ∧ option_C ∧ option_D :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_B_l1328_132829


namespace NUMINAMATH_GPT_minimize_theta_abs_theta_val_l1328_132834

noncomputable def theta (k : ℤ) : ℝ := -11 / 4 * Real.pi + 2 * k * Real.pi

theorem minimize_theta_abs (k : ℤ) :
  ∃ θ : ℝ, (θ = -11 / 4 * Real.pi + 2 * k * Real.pi) ∧
           (∀ η : ℝ, (η = -11 / 4 * Real.pi + 2 * (k + 1) * Real.pi) →
             |θ| ≤ |η|) :=
  sorry

theorem theta_val : ∃ θ : ℝ, θ = -3 / 4 * Real.pi :=
  ⟨ -3 / 4 * Real.pi, rfl ⟩

end NUMINAMATH_GPT_minimize_theta_abs_theta_val_l1328_132834


namespace NUMINAMATH_GPT_unique_third_rectangle_exists_l1328_132848

-- Define the given rectangles.
def rect1_length : ℕ := 3
def rect1_width : ℕ := 8
def rect2_length : ℕ := 2
def rect2_width : ℕ := 5

-- Define the areas of the given rectangles.
def area_rect1 : ℕ := rect1_length * rect1_width
def area_rect2 : ℕ := rect2_length * rect2_width

-- Define the total area covered by the two given rectangles.
def total_area_without_third : ℕ := area_rect1 + area_rect2

-- We need to prove that there exists one unique configuration for the third rectangle.
theorem unique_third_rectangle_exists (a b : ℕ) : 
  (total_area_without_third + a * b = 34) → 
  (a * b = 4) → 
  (a = 4 ∧ b = 1 ∨ a = 1 ∧ b = 4) :=
by sorry

end NUMINAMATH_GPT_unique_third_rectangle_exists_l1328_132848


namespace NUMINAMATH_GPT_collinearity_necessary_but_not_sufficient_l1328_132801

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (u v : V) : Prop := ∃ (a : ℝ), v = a • u

def equal (u v : V) : Prop := u = v

theorem collinearity_necessary_but_not_sufficient (u v : V) :
  (collinear u v → equal u v) ∧ (equal u v → collinear u v) → collinear u v ∧ ¬(collinear u v ↔ equal u v) :=
sorry

end NUMINAMATH_GPT_collinearity_necessary_but_not_sufficient_l1328_132801


namespace NUMINAMATH_GPT_smallest_base_converted_l1328_132860

def convert_to_decimal_base_3 (n : ℕ) : ℕ :=
  1 * 3^3 + 0 * 3^2 + 0 * 3^1 + 2 * 3^0

def convert_to_decimal_base_6 (n : ℕ) : ℕ :=
  2 * 6^2 + 1 * 6^1 + 0 * 6^0

def convert_to_decimal_base_4 (n : ℕ) : ℕ :=
  1 * 4^3 + 0 * 4^2 + 0 * 4^1 + 0 * 4^0

def convert_to_decimal_base_2 (n : ℕ) : ℕ :=
  1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem smallest_base_converted :
  min (convert_to_decimal_base_3 1002) 
      (min (convert_to_decimal_base_6 210) 
           (min (convert_to_decimal_base_4 1000) 
                (convert_to_decimal_base_2 111111))) = convert_to_decimal_base_3 1002 :=
by sorry

end NUMINAMATH_GPT_smallest_base_converted_l1328_132860


namespace NUMINAMATH_GPT_base_7_divisibility_l1328_132872

theorem base_7_divisibility (y : ℕ) :
  (934 + 7 * y) % 19 = 0 ↔ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_base_7_divisibility_l1328_132872


namespace NUMINAMATH_GPT_coin_flip_probability_l1328_132832

theorem coin_flip_probability :
  let total_outcomes := 2^5
  let successful_outcomes := 2 * 2^2
  total_outcomes > 0 → (successful_outcomes / total_outcomes) = (1 / 4) :=
by
  intros
  sorry

end NUMINAMATH_GPT_coin_flip_probability_l1328_132832


namespace NUMINAMATH_GPT_isosceles_triangle_angle_B_l1328_132873

theorem isosceles_triangle_angle_B :
  ∀ (A B C : ℝ), (B = C) → (C = 3 * A) → (A + B + C = 180) → (B = 540 / 7) :=
by
  intros A B C h1 h2 h3
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_B_l1328_132873


namespace NUMINAMATH_GPT_john_free_throws_l1328_132849

theorem john_free_throws 
  (hit_rate : ℝ) 
  (shots_per_foul : ℕ) 
  (fouls_per_game : ℕ) 
  (total_games : ℕ) 
  (percentage_played : ℝ) 
  : hit_rate = 0.7 → 
    shots_per_foul = 2 → 
    fouls_per_game = 5 → 
    total_games = 20 → 
    percentage_played = 0.8 → 
    ∃ (total_free_throws : ℕ), total_free_throws = 112 := 
by
  intros
  sorry

end NUMINAMATH_GPT_john_free_throws_l1328_132849


namespace NUMINAMATH_GPT_isosceles_triangle_legs_length_l1328_132876

-- Define the given conditions in Lean
def perimeter (L B: ℕ) : ℕ := 2 * L + B
def base_length : ℕ := 8
def given_perimeter : ℕ := 20

-- State the theorem to be proven
theorem isosceles_triangle_legs_length :
  ∃ (L : ℕ), perimeter L base_length = given_perimeter ∧ L = 6 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_legs_length_l1328_132876


namespace NUMINAMATH_GPT_ab_value_l1328_132826

theorem ab_value (a b : ℝ) (h1 : a + b = 4) (h2 : a^2 + b^2 = 18) : a * b = -1 :=
by sorry

end NUMINAMATH_GPT_ab_value_l1328_132826


namespace NUMINAMATH_GPT_line_through_points_l1328_132813

theorem line_through_points (a b : ℝ) (h₁ : 1 = a * 3 + b) (h₂ : 13 = a * 7 + b) : a - b = 11 := 
  sorry

end NUMINAMATH_GPT_line_through_points_l1328_132813


namespace NUMINAMATH_GPT_farm_cows_l1328_132841

theorem farm_cows (c h : ℕ) 
  (legs_eq : 5 * c + 2 * h = 20 + 2 * (c + h)) : 
  c = 6 :=
by 
  sorry

end NUMINAMATH_GPT_farm_cows_l1328_132841


namespace NUMINAMATH_GPT_dance_pairs_exist_l1328_132857

variable {Boy Girl : Type} 

-- Define danced_with relation
variable (danced_with : Boy → Girl → Prop)

-- Given conditions
variable (H1 : ∀ (b : Boy), ∃ (g : Girl), ¬ danced_with b g)
variable (H2 : ∀ (g : Girl), ∃ (b : Boy), danced_with b g)

-- Proof that desired pairs exist
theorem dance_pairs_exist :
  ∃ (M1 M2 : Boy) (D1 D2 : Girl),
    danced_with M1 D1 ∧
    danced_with M2 D2 ∧
    ¬ danced_with M1 D2 ∧
    ¬ danced_with M2 D1 :=
sorry

end NUMINAMATH_GPT_dance_pairs_exist_l1328_132857


namespace NUMINAMATH_GPT_city_grid_sinks_l1328_132818

-- Define the main conditions of the grid city
def cell_side_meter : Int := 500
def max_travel_km : Int := 1

-- Total number of intersections in a 100x100 grid
def total_intersections : Int := (100 + 1) * (100 + 1)

-- Number of sinks that need to be proven
def required_sinks : Int := 1300

-- Lean theorem statement to prove that given the conditions,
-- there are at least 1300 sinks (intersections that act as sinks)
theorem city_grid_sinks :
  ∀ (city_grid : Matrix (Fin 101) (Fin 101) IntersectionType),
  (∀ i j, i < 100 → j < 100 → cell_side_meter ≤ max_travel_km * 1000) →
  ∃ (sinks : Finset (Fin 101 × Fin 101)), 
  (sinks.card ≥ required_sinks) := sorry

end NUMINAMATH_GPT_city_grid_sinks_l1328_132818


namespace NUMINAMATH_GPT_antonella_purchase_l1328_132820

theorem antonella_purchase
  (total_coins : ℕ)
  (coin_value : ℕ → ℕ)
  (num_toonies : ℕ)
  (initial_loonies : ℕ)
  (initial_toonies : ℕ)
  (total_value : ℕ)
  (amount_spent : ℕ)
  (amount_left : ℕ)
  (H1 : total_coins = 10)
  (H2 : coin_value 1 = 1)
  (H3 : coin_value 2 = 2)
  (H4 : initial_toonies = 4)
  (H5 : initial_loonies = total_coins - initial_toonies)
  (H6 : total_value = initial_loonies * coin_value 1 + initial_toonies * coin_value 2)
  (H7 : amount_spent = 3)
  (H8 : amount_left = total_value - amount_spent)
  (H9 : amount_left = 11) :
  ∃ (used_loonies used_toonies : ℕ), used_loonies = 1 ∧ used_toonies = 1 ∧ (used_loonies * coin_value 1 + used_toonies * coin_value 2 = amount_spent) :=
by
  sorry

end NUMINAMATH_GPT_antonella_purchase_l1328_132820


namespace NUMINAMATH_GPT_moving_circle_passes_focus_l1328_132839

noncomputable def parabola (x : ℝ) : Set (ℝ × ℝ) := {p | p.2 ^ 2 = 8 * p.1}
def is_tangent (c : ℝ × ℝ) (r : ℝ) : Prop := c.1 = -2 ∨ c.1 = -2 + 2 * r

theorem moving_circle_passes_focus
  (center : ℝ × ℝ) (H1 : center ∈ parabola center.1)
  (H2 : is_tangent center 2) :
  ∃ focus : ℝ × ℝ, focus = (2, 0) ∧ ∃ r : ℝ, ∀ p ∈ parabola center.1, dist center p = r := sorry

end NUMINAMATH_GPT_moving_circle_passes_focus_l1328_132839


namespace NUMINAMATH_GPT_heroes_on_the_back_l1328_132837

theorem heroes_on_the_back (total_heroes front_heroes : ℕ) (h1 : total_heroes = 9) (h2 : front_heroes = 2) :
  total_heroes - front_heroes = 7 := by
  sorry

end NUMINAMATH_GPT_heroes_on_the_back_l1328_132837


namespace NUMINAMATH_GPT_weird_fraction_implies_weird_power_fraction_l1328_132874

theorem weird_fraction_implies_weird_power_fraction 
  (a b c : ℝ) (k : ℕ) 
  (h1 : (1/a) + (1/b) + (1/c) = (1/(a + b + c))) 
  (h2 : Odd k) : 
  (1 / (a^k) + 1 / (b^k) + 1 / (c^k) = 1 / (a^k + b^k + c^k)) := 
by 
  sorry

end NUMINAMATH_GPT_weird_fraction_implies_weird_power_fraction_l1328_132874


namespace NUMINAMATH_GPT_suzie_reads_pages_hour_l1328_132806

-- Declaration of the variables and conditions
variables (S : ℕ) -- S is the number of pages Suzie reads in an hour
variables (L : ℕ) -- L is the number of pages Liza reads in an hour

-- Conditions given in the problem
def reads_per_hour_Liza : L = 20 := sorry
def reads_more_pages : L * 3 = S * 3 + 15 := sorry

-- The statement we want to prove:
theorem suzie_reads_pages_hour : S = 15 :=
by
  -- Proof steps needed here (omitted due to the instruction)
  sorry

end NUMINAMATH_GPT_suzie_reads_pages_hour_l1328_132806


namespace NUMINAMATH_GPT_sin_alpha_cos_squared_beta_range_l1328_132858

theorem sin_alpha_cos_squared_beta_range (α β : ℝ) 
  (h : Real.sin α + Real.sin β = 1) : 
  ∃ y, y = Real.sin α - Real.cos β ^ 2 ∧ (-1/4 ≤ y ∧ y ≤ 0) :=
sorry

end NUMINAMATH_GPT_sin_alpha_cos_squared_beta_range_l1328_132858


namespace NUMINAMATH_GPT_find_values_l1328_132892

theorem find_values (x y z : ℝ) :
  (x + y + z = 1) →
  (x^2 * y + y^2 * z + z^2 * x = x * y^2 + y * z^2 + z * x^2) →
  (x^3 + y^2 + z = y^3 + z^2 + x) →
  ( (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ 
    (x = 0 ∧ y = 0 ∧ z = 1) ∨
    (x = 2/3 ∧ y = -1/3 ∧ z = 2/3) ∨
    (x = 0 ∧ y = 1 ∧ z = 0) ∨
    (x = 1 ∧ y = 0 ∧ z = 0) ∨
    (x = -1 ∧ y = 1 ∧ z = 1) ) := 
sorry

end NUMINAMATH_GPT_find_values_l1328_132892


namespace NUMINAMATH_GPT_rectangle_ratio_l1328_132884

theorem rectangle_ratio (s x y : ℝ) 
  (h1 : 4 * (x * y) + s^2 = 9 * s^2)
  (h2 : x + s = 3 * s)
  (h3 : s + 2 * y = 3 * s) :
  x / y = 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l1328_132884


namespace NUMINAMATH_GPT_solve_trig_eqn_solution_set_l1328_132882

theorem solve_trig_eqn_solution_set :
  {x : ℝ | ∃ k : ℤ, x = 3 * k * Real.pi + Real.pi / 4 ∨ x = 3 * k * Real.pi + 5 * Real.pi / 4} =
  {x : ℝ | 2 * Real.sin ((2 / 3) * x) = 1} :=
by
  sorry

end NUMINAMATH_GPT_solve_trig_eqn_solution_set_l1328_132882


namespace NUMINAMATH_GPT_sum_of_reciprocals_is_3_over_8_l1328_132854

theorem sum_of_reciprocals_is_3_over_8 (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  (1 / x + 1 / y) = 3 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_is_3_over_8_l1328_132854


namespace NUMINAMATH_GPT_total_onions_l1328_132808

-- Define the number of onions grown by each individual
def nancy_onions : ℕ := 2
def dan_onions : ℕ := 9
def mike_onions : ℕ := 4

-- Proposition: The total number of onions grown is 15
theorem total_onions : (nancy_onions + dan_onions + mike_onions) = 15 := 
by sorry

end NUMINAMATH_GPT_total_onions_l1328_132808


namespace NUMINAMATH_GPT_total_amount_l1328_132853

theorem total_amount {B C : ℝ} 
  (h1 : C = 1600) 
  (h2 : 4 * B = 16 * C) : 
  B + C = 2000 :=
sorry

end NUMINAMATH_GPT_total_amount_l1328_132853


namespace NUMINAMATH_GPT_appropriate_sampling_method_l1328_132898

theorem appropriate_sampling_method (total_staff teachers admin_staff logistics_personnel sample_size : ℕ)
  (h1 : total_staff = 160)
  (h2 : teachers = 120)
  (h3 : admin_staff = 16)
  (h4 : logistics_personnel = 24)
  (h5 : sample_size = 20) :
  (sample_method : String) -> sample_method = "Stratified sampling" :=
sorry

end NUMINAMATH_GPT_appropriate_sampling_method_l1328_132898


namespace NUMINAMATH_GPT_manny_gave_2_marbles_l1328_132891

-- Define the total number of marbles
def total_marbles : ℕ := 36

-- Define the ratio parts for Mario and Manny
def mario_ratio : ℕ := 4
def manny_ratio : ℕ := 5

-- Define the total ratio parts
def total_ratio : ℕ := mario_ratio + manny_ratio

-- Define the number of marbles Manny has after giving some away
def manny_marbles_now : ℕ := 18

-- Calculate the marbles per part based on the ratio and total marbles
def marbles_per_part : ℕ := total_marbles / total_ratio

-- Calculate the number of marbles Manny originally had
def manny_marbles_original : ℕ := manny_ratio * marbles_per_part

-- Formulate the theorem
theorem manny_gave_2_marbles : manny_marbles_original - manny_marbles_now = 2 := by
  sorry

end NUMINAMATH_GPT_manny_gave_2_marbles_l1328_132891


namespace NUMINAMATH_GPT_hexagon_largest_angle_measure_l1328_132895

theorem hexagon_largest_angle_measure (x : ℝ) (a b c d e f : ℝ)
  (h_ratio: a = 2 * x) (h_ratio2: b = 3 * x)
  (h_ratio3: c = 3 * x) (h_ratio4: d = 4 * x)
  (h_ratio5: e = 4 * x) (h_ratio6: f = 6 * x)
  (h_sum: a + b + c + d + e + f = 720) :
  f = 2160 / 11 :=
by
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_hexagon_largest_angle_measure_l1328_132895


namespace NUMINAMATH_GPT_zongzi_unit_price_l1328_132880

theorem zongzi_unit_price (uA uB : ℝ) (pA pB : ℝ) : 
  pA = 1200 → pB = 800 → uA = 2 * uB → pA / uA = pB / uB - 50 → uB = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_zongzi_unit_price_l1328_132880


namespace NUMINAMATH_GPT_reciprocal_of_neg_two_thirds_l1328_132833

-- Definition for finding the reciprocal
def reciprocal (x : ℚ) : ℚ := 1 / x

-- The proof problem statement
theorem reciprocal_of_neg_two_thirds : reciprocal (-2 / 3) = -3 / 2 :=
sorry

end NUMINAMATH_GPT_reciprocal_of_neg_two_thirds_l1328_132833


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_l1328_132803

noncomputable def radius_inscribed_circle (AB BC AC : ℝ) (s : ℝ) (K : ℝ) : ℝ := K / s

theorem radius_of_inscribed_circle (AB BC AC : ℝ) (h1: AB = 8) (h2: BC = 8) (h3: AC = 10) :
  radius_inscribed_circle AB BC AC 13 (5 * Real.sqrt 39) = (5 * Real.sqrt 39) / 13 :=
  by
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_l1328_132803


namespace NUMINAMATH_GPT_initial_books_count_l1328_132842

-- Definitions of the given conditions
def shelves : ℕ := 9
def books_per_shelf : ℕ := 9
def books_remaining : ℕ := shelves * books_per_shelf
def books_sold : ℕ := 39

-- Statement of the proof problem
theorem initial_books_count : books_remaining + books_sold = 120 := 
by {
  sorry
}

end NUMINAMATH_GPT_initial_books_count_l1328_132842


namespace NUMINAMATH_GPT_largest_value_of_c_l1328_132894

theorem largest_value_of_c : ∃ c, (∀ x : ℝ, x^2 - 6 * x + c = 1 → c ≤ 10) :=
sorry

end NUMINAMATH_GPT_largest_value_of_c_l1328_132894


namespace NUMINAMATH_GPT_range_of_a_squared_minus_2b_l1328_132878

variable (a b : ℝ)

def quadratic_has_two_real_roots_in_01 (a b : ℝ) : Prop :=
  b ≥ 0 ∧ 1 + a + b ≥ 0 ∧ -2 ≤ a ∧ a ≤ 0 ∧ a^2 - 4 * b ≥ 0

theorem range_of_a_squared_minus_2b (a b : ℝ)
  (h : quadratic_has_two_real_roots_in_01 a b) : 0 ≤ a^2 - 2 * b ∧ a^2 - 2 * b ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_squared_minus_2b_l1328_132878


namespace NUMINAMATH_GPT_dogs_sold_l1328_132845

theorem dogs_sold (cats_sold : ℕ) (h1 : cats_sold = 16) (ratio : ℕ × ℕ) (h2 : ratio = (2, 1)) : ∃ dogs_sold : ℕ, dogs_sold = 8 := by
  sorry

end NUMINAMATH_GPT_dogs_sold_l1328_132845


namespace NUMINAMATH_GPT_total_area_for_building_l1328_132897

theorem total_area_for_building (num_sections : ℕ) (area_per_section : ℝ) (open_space_percentage : ℝ) :
  num_sections = 7 →
  area_per_section = 9473 →
  open_space_percentage = 0.15 →
  (num_sections * (area_per_section * (1 - open_space_percentage))) = 56364.35 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_area_for_building_l1328_132897


namespace NUMINAMATH_GPT_find_m_of_pure_imaginary_l1328_132865

theorem find_m_of_pure_imaginary (m : ℝ) (h1 : (m^2 + m - 2) = 0) (h2 : (m^2 - 1) ≠ 0) : m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_of_pure_imaginary_l1328_132865


namespace NUMINAMATH_GPT_find_larger_number_l1328_132855

theorem find_larger_number (x y : ℕ) (h1 : y - x = 1365) (h2 : y = 4 * x + 15) : y = 1815 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l1328_132855


namespace NUMINAMATH_GPT_Eleanor_books_l1328_132883

theorem Eleanor_books (h p : ℕ) : 
    h + p = 12 ∧ 28 * h + 18 * p = 276 → h = 6 :=
by
  intro hp
  sorry

end NUMINAMATH_GPT_Eleanor_books_l1328_132883


namespace NUMINAMATH_GPT_counterexample_to_conjecture_l1328_132896

theorem counterexample_to_conjecture (n : ℕ) (h : n > 5) : 
  ¬ (∃ a b c : ℕ, (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (a + b + c = n)) ∨
  ¬ (∃ a b c : ℕ, (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (a + b + c = n)) :=
sorry

end NUMINAMATH_GPT_counterexample_to_conjecture_l1328_132896


namespace NUMINAMATH_GPT_parabola_directrix_l1328_132819

theorem parabola_directrix (x y : ℝ) (h_parabola : x^2 = (1/2) * y) : y = - (1/8) :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l1328_132819


namespace NUMINAMATH_GPT_impossible_rearrange_reverse_l1328_132899

theorem impossible_rearrange_reverse :
  ∀ (tokens : ℕ → ℕ), 
    (∀ i, (i % 2 = 1 ∧ i < 99 → tokens i = tokens (i + 2)) 
      ∧ (i % 2 = 0 ∧ i < 99 → tokens i = tokens (i + 2))) → ¬(∀ i, tokens i = 100 + 1 - tokens (i - 1)) :=
by
  intros tokens h
  sorry

end NUMINAMATH_GPT_impossible_rearrange_reverse_l1328_132899


namespace NUMINAMATH_GPT_tabitha_honey_nights_l1328_132850

def servings_per_cup := 1
def cups_per_night := 2
def ounces_per_container := 16
def servings_per_ounce := 6
def total_servings := servings_per_ounce * ounces_per_container
def servings_per_night := servings_per_cup * cups_per_night
def number_of_nights := total_servings / servings_per_night

theorem tabitha_honey_nights : number_of_nights = 48 :=
by
  -- Proof to be provided.
  sorry

end NUMINAMATH_GPT_tabitha_honey_nights_l1328_132850
