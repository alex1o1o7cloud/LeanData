import Mathlib

namespace NUMINAMATH_GPT_part1_l167_16759

theorem part1 (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) : 2 * x^2 + y^2 > x^2 + x * y := 
sorry

end NUMINAMATH_GPT_part1_l167_16759


namespace NUMINAMATH_GPT_smallest_positive_angle_l167_16739

theorem smallest_positive_angle (α : ℝ) (h : α = 2012) : ∃ β : ℝ, 0 < β ∧ β < 360 ∧ β = α % 360 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_angle_l167_16739


namespace NUMINAMATH_GPT_packs_needed_is_six_l167_16774

variable (l_bedroom l_bathroom l_kitchen l_basement : ℕ)

def total_bulbs_needed := l_bedroom + l_bathroom + l_kitchen + l_basement
def garage_bulbs_needed := total_bulbs_needed / 2
def total_bulbs_with_garage := total_bulbs_needed + garage_bulbs_needed
def packs_needed := total_bulbs_with_garage / 2

theorem packs_needed_is_six
    (h1 : l_bedroom = 2)
    (h2 : l_bathroom = 1)
    (h3 : l_kitchen = 1)
    (h4 : l_basement = 4) :
    packs_needed l_bedroom l_bathroom l_kitchen l_basement = 6 := by
  sorry

end NUMINAMATH_GPT_packs_needed_is_six_l167_16774


namespace NUMINAMATH_GPT_factorize_poly1_factorize_poly2_l167_16766

-- Define y substitution for first problem
def poly1_y := fun (x : ℝ) => x^2 + 2*x
-- Define y substitution for second problem
def poly2_y := fun (x : ℝ) => x^2 - 4*x

-- Define the given polynomial expressions 
def poly1 := fun (x : ℝ) => (x^2 + 2*x)*(x^2 + 2*x + 2) + 1
def poly2 := fun (x : ℝ) => (x^2 - 4*x)*(x^2 - 4*x + 8) + 16

theorem factorize_poly1 (x : ℝ) : poly1 x = (x + 1) ^ 4 := sorry

theorem factorize_poly2 (x : ℝ) : poly2 x = (x - 2) ^ 4 := sorry

end NUMINAMATH_GPT_factorize_poly1_factorize_poly2_l167_16766


namespace NUMINAMATH_GPT_initial_marbles_count_l167_16720

theorem initial_marbles_count (g y : ℕ) 
  (h1 : (g + 3) * 4 = g + y + 3) 
  (h2 : 3 * g = g + y + 4) : 
  g + y = 8 := 
by 
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_initial_marbles_count_l167_16720


namespace NUMINAMATH_GPT_number_of_girls_l167_16705

theorem number_of_girls (B G : ℕ) (h1 : B * 5 = G * 8) (h2 : B + G = 1040) : G = 400 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l167_16705


namespace NUMINAMATH_GPT_lily_final_balance_l167_16703

noncomputable def initial_balance : ℝ := 55
noncomputable def shirt_cost : ℝ := 7
noncomputable def shoes_cost : ℝ := 3 * shirt_cost
noncomputable def book_cost : ℝ := 4
noncomputable def books_amount : ℝ := 5
noncomputable def gift_fraction : ℝ := 0.20

noncomputable def remaining_balance : ℝ :=
  initial_balance - 
  shirt_cost - 
  shoes_cost - 
  books_amount * book_cost - 
  gift_fraction * (initial_balance - shirt_cost - shoes_cost - books_amount * book_cost)

theorem lily_final_balance : remaining_balance = 5.60 := 
by 
  sorry

end NUMINAMATH_GPT_lily_final_balance_l167_16703


namespace NUMINAMATH_GPT_find_side_b_in_triangle_l167_16714

theorem find_side_b_in_triangle 
  (A B : ℝ) (a : ℝ)
  (h_cosA : Real.cos A = -1/2)
  (h_B : B = Real.pi / 4)
  (h_a : a = 3) :
  ∃ b, b = Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_find_side_b_in_triangle_l167_16714


namespace NUMINAMATH_GPT_passenger_gets_ticket_l167_16785

variables (p1 p2 p3 p4 p5 p6 : ℝ)

-- Conditions:
axiom h_sum_eq_one : p1 + p2 + p3 = 1
axiom h_p1_nonneg : 0 ≤ p1
axiom h_p2_nonneg : 0 ≤ p2
axiom h_p3_nonneg : 0 ≤ p3
axiom h_p4_nonneg : 0 ≤ p4
axiom h_p4_le_one : p4 ≤ 1
axiom h_p5_nonneg : 0 ≤ p5
axiom h_p5_le_one : p5 ≤ 1
axiom h_p6_nonneg : 0 ≤ p6
axiom h_p6_le_one : p6 ≤ 1

-- Theorem:
theorem passenger_gets_ticket :
  (p1 * (1 - p4) + p2 * (1 - p5) + p3 * (1 - p6)) = (p1 * (1 - p4) + p2 * (1 - p5) + p3 * (1 - p6)) :=
by sorry

end NUMINAMATH_GPT_passenger_gets_ticket_l167_16785


namespace NUMINAMATH_GPT_maximum_combined_power_l167_16787

theorem maximum_combined_power (x1 x2 x3 : ℝ) (hx : x1 < 1 ∧ x2 < 1 ∧ x3 < 1) 
    (hcond : 2 * (x1 + x2 + x3) + 4 * (x1 * x2 * x3) = 3 * (x1 * x2 + x1 * x3 + x2 * x3) + 1) : 
    x1 + x2 + x3 ≤ 3 / 4 := 
sorry

end NUMINAMATH_GPT_maximum_combined_power_l167_16787


namespace NUMINAMATH_GPT_proposition_q_must_be_true_l167_16794

theorem proposition_q_must_be_true (p q : Prop) (h1 : p ∨ q) (h2 : ¬ p) : q :=
by
  sorry

end NUMINAMATH_GPT_proposition_q_must_be_true_l167_16794


namespace NUMINAMATH_GPT_car_price_difference_l167_16758

variable (original_paid old_car_proceeds : ℝ)
variable (new_car_price additional_amount : ℝ)

theorem car_price_difference :
  old_car_proceeds = new_car_price - additional_amount →
  old_car_proceeds = 0.8 * original_paid →
  additional_amount = 4000 →
  new_car_price = 30000 →
  (original_paid - new_car_price) = 2500 :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_car_price_difference_l167_16758


namespace NUMINAMATH_GPT_no_integers_satisfy_eq_l167_16728

theorem no_integers_satisfy_eq (m n : ℤ) : m^2 ≠ n^5 - 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_no_integers_satisfy_eq_l167_16728


namespace NUMINAMATH_GPT_positive_n_of_single_solution_l167_16781

theorem positive_n_of_single_solution (n : ℝ) (h : ∃ x : ℝ, (9 * x^2 + n * x + 36) = 0 ∧ (∀ y : ℝ, (9 * y^2 + n * y + 36) = 0 → y = x)) : n = 36 :=
sorry

end NUMINAMATH_GPT_positive_n_of_single_solution_l167_16781


namespace NUMINAMATH_GPT_xyz_inequality_l167_16718

theorem xyz_inequality (x y z : ℝ) (h : x + y + z = 0) : 
  6 * (x^3 + y^3 + z^3)^2 ≤ (x^2 + y^2 + z^2)^3 := 
by sorry

end NUMINAMATH_GPT_xyz_inequality_l167_16718


namespace NUMINAMATH_GPT_teammates_score_is_correct_l167_16777

-- Definitions based on the given conditions
def Lizzie_score : ℕ := 4
def Nathalie_score : ℕ := Lizzie_score + 3
def Combined_score : ℕ := Lizzie_score + Nathalie_score
def Aimee_score : ℕ := 2 * Combined_score
def Total_score : ℕ := Lizzie_score + Nathalie_score + Aimee_score
def Whole_team_score : ℕ := 50
def Teammates_score : ℕ := Whole_team_score - Total_score

-- Proof statement
theorem teammates_score_is_correct : Teammates_score = 17 := by
  sorry

end NUMINAMATH_GPT_teammates_score_is_correct_l167_16777


namespace NUMINAMATH_GPT_cyclic_inequality_l167_16786

variables {a b c : ℝ}

theorem cyclic_inequality (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  (ab / (a + b + 2 * c) + bc / (b + c + 2 * a) + ca / (c + a + 2 * b)) ≤ (a + b + c) / 4 :=
sorry

end NUMINAMATH_GPT_cyclic_inequality_l167_16786


namespace NUMINAMATH_GPT_tiles_touching_walls_of_room_l167_16712

theorem tiles_touching_walls_of_room (length width : Nat) 
    (hl : length = 10) (hw : width = 5) : 
    2 * length + 2 * width - 4 = 26 := by
  sorry

end NUMINAMATH_GPT_tiles_touching_walls_of_room_l167_16712


namespace NUMINAMATH_GPT_cost_price_600_l167_16745

variable (CP SP : ℝ)

theorem cost_price_600 
  (h1 : SP = 1.08 * CP) 
  (h2 : SP = 648) : 
  CP = 600 := 
by
  sorry

end NUMINAMATH_GPT_cost_price_600_l167_16745


namespace NUMINAMATH_GPT_mass_of_barium_sulfate_l167_16782

-- Definitions of the chemical equation and molar masses
def barium_molar_mass : ℝ := 137.327
def sulfur_molar_mass : ℝ := 32.065
def oxygen_molar_mass : ℝ := 15.999
def molar_mass_BaSO4 : ℝ := barium_molar_mass + sulfur_molar_mass + 4 * oxygen_molar_mass

-- Given conditions
def moles_BaBr2 : ℝ := 4
def moles_BaSO4_produced : ℝ := moles_BaBr2 -- from balanced equation

-- Calculate mass of BaSO4 produced
def mass_BaSO4 : ℝ := moles_BaSO4_produced * molar_mass_BaSO4

-- Mass of Barium sulfate produced
theorem mass_of_barium_sulfate : mass_BaSO4 = 933.552 :=
by 
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_mass_of_barium_sulfate_l167_16782


namespace NUMINAMATH_GPT_rectangle_area_l167_16757

theorem rectangle_area (l w : ℕ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 120) : l * w = 800 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_rectangle_area_l167_16757


namespace NUMINAMATH_GPT_base8_subtraction_correct_l167_16761

theorem base8_subtraction_correct : (453 - 326 : ℕ) = 125 :=
by sorry

end NUMINAMATH_GPT_base8_subtraction_correct_l167_16761


namespace NUMINAMATH_GPT_xy_value_l167_16764

theorem xy_value (x y : ℝ) (h1 : (x + y) / 3 = 1.222222222222222) : x + y = 3.666666666666666 :=
by
  sorry

end NUMINAMATH_GPT_xy_value_l167_16764


namespace NUMINAMATH_GPT_vertical_asymptote_sum_l167_16726

theorem vertical_asymptote_sum :
  (∀ x : ℝ, 4*x^2 + 6*x + 3 = 0 → x = -1 / 2 ∨ x = -1) →
  (-1 / 2 + -1) = -3 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_vertical_asymptote_sum_l167_16726


namespace NUMINAMATH_GPT_inequality_proof_l167_16713

variable {a b : ℕ → ℝ}

-- Conditions: {a_n} is a geometric sequence with positive terms, {b_n} is an arithmetic sequence, a_6 = b_8
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

axiom a_pos_terms : ∀ n : ℕ, a n > 0
axiom a_geom_seq : is_geometric a
axiom b_arith_seq : is_arithmetic b
axiom a6_eq_b8 : a 6 = b 8

-- Prove: a_3 + a_9 ≥ b_9 + b_7
theorem inequality_proof : a 3 + a 9 ≥ b 9 + b 7 :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l167_16713


namespace NUMINAMATH_GPT_option_C_incorrect_l167_16791

structure Line := (point1 point2 : ℝ × ℝ × ℝ)
structure Plane := (point : ℝ × ℝ × ℝ) (normal : ℝ × ℝ × ℝ)

variables (m n : Line) (α β : Plane)

def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular_to_plane (p1 p2 : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def lines_parallel (l1 l2 : Line) : Prop := sorry
def lines_perpendicular (l1 l2 : Line) : Prop := sorry
def planes_parallel (p1 p2 : Plane) : Prop := sorry

theorem option_C_incorrect 
  (h1 : line_in_plane m α)
  (h2 : line_parallel_to_plane n α)
  (h3 : lines_parallel m n) :
  false :=
sorry

end NUMINAMATH_GPT_option_C_incorrect_l167_16791


namespace NUMINAMATH_GPT_distinct_factors_count_l167_16754

-- Given conditions
def eight_squared : ℕ := 8^2
def nine_cubed : ℕ := 9^3
def seven_fifth : ℕ := 7^5
def number : ℕ := eight_squared * nine_cubed * seven_fifth

-- Proving the number of natural-number factors of the given number
theorem distinct_factors_count : 
  (number.factors.count 1 = 294) := sorry

end NUMINAMATH_GPT_distinct_factors_count_l167_16754


namespace NUMINAMATH_GPT_right_triangle_short_leg_l167_16793

theorem right_triangle_short_leg (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 65) (h_int : ∃ x y z : ℕ, a = x ∧ b = y ∧ c = z) :
  a = 39 ∨ b = 39 :=
sorry

end NUMINAMATH_GPT_right_triangle_short_leg_l167_16793


namespace NUMINAMATH_GPT_xy_sum_of_squares_l167_16751

theorem xy_sum_of_squares (x y : ℝ) (h1 : x - y = 5) (h2 : -x * y = 4) : x^2 + y^2 = 17 := 
sorry

end NUMINAMATH_GPT_xy_sum_of_squares_l167_16751


namespace NUMINAMATH_GPT_find_m_if_polynomial_is_perfect_square_l167_16762

theorem find_m_if_polynomial_is_perfect_square (m : ℝ) :
  (∃ a b : ℝ, (a * x + b)^2 = x^2 + m * x + 4) → (m = 4 ∨ m = -4) :=
sorry

end NUMINAMATH_GPT_find_m_if_polynomial_is_perfect_square_l167_16762


namespace NUMINAMATH_GPT_tina_mother_age_l167_16709

variable {x : ℕ}

theorem tina_mother_age (h1 : 10 + x = 2 * x - 20) : 2010 + x = 2040 :=
by 
  sorry

end NUMINAMATH_GPT_tina_mother_age_l167_16709


namespace NUMINAMATH_GPT_man_speed_still_water_l167_16743

noncomputable def speed_in_still_water (U D : ℝ) : ℝ := (U + D) / 2

theorem man_speed_still_water :
  let U := 45
  let D := 55
  speed_in_still_water U D = 50 := by
  sorry

end NUMINAMATH_GPT_man_speed_still_water_l167_16743


namespace NUMINAMATH_GPT_sarah_numbers_sum_l167_16733

-- Definition of x and y being integers with their respective ranges
def isTwoDigit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99
def isThreeDigit (y : ℕ) : Prop := 100 ≤ y ∧ y ≤ 999

-- The condition relating x and y
def formedNumber (x y : ℕ) : Prop := 1000 * x + y = 7 * x * y

-- The Lean 4 statement for the proof problem
theorem sarah_numbers_sum (x y : ℕ) (H1 : isTwoDigit x) (H2 : isThreeDigit y) (H3 : formedNumber x y) : x + y = 1074 :=
  sorry

end NUMINAMATH_GPT_sarah_numbers_sum_l167_16733


namespace NUMINAMATH_GPT_original_number_is_perfect_square_l167_16778

variable (n : ℕ)

theorem original_number_is_perfect_square
  (h1 : n = 1296)
  (h2 : ∃ m : ℕ, (n + 148) = m^2) : ∃ k : ℕ, n = k^2 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_perfect_square_l167_16778


namespace NUMINAMATH_GPT_chairs_per_row_l167_16763

theorem chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) 
  (h_total_chairs : total_chairs = 432) (h_num_rows : num_rows = 27) : 
  total_chairs / num_rows = 16 :=
by
  sorry

end NUMINAMATH_GPT_chairs_per_row_l167_16763


namespace NUMINAMATH_GPT_num_five_letter_words_correct_l167_16737

-- Define the number of letters in the alphabet
def num_letters : ℕ := 26

-- Define the number of vowels
def num_vowels : ℕ := 5

-- Define a function that calculates the number of valid five-letter words
def num_five_letter_words : ℕ :=
  num_letters * num_vowels * num_letters * num_letters

-- The theorem statement we need to prove
theorem num_five_letter_words_correct : num_five_letter_words = 87700 :=
by
  -- The proof is omitted; it should equate the calculated value to 87700
  sorry

end NUMINAMATH_GPT_num_five_letter_words_correct_l167_16737


namespace NUMINAMATH_GPT_total_interest_is_68_l167_16792

-- Definitions of the initial conditions
def amount_2_percent : ℝ := 600
def amount_4_percent : ℝ := amount_2_percent + 800
def interest_rate_2_percent : ℝ := 0.02
def interest_rate_4_percent : ℝ := 0.04
def invested_total_1 : ℝ := amount_2_percent
def invested_total_2 : ℝ := amount_4_percent

-- The total interest calculation
def interest_2_percent : ℝ := invested_total_1 * interest_rate_2_percent
def interest_4_percent : ℝ := invested_total_2 * interest_rate_4_percent

-- Claim: The total interest earned is $68
theorem total_interest_is_68 : interest_2_percent + interest_4_percent = 68 := by
  sorry

end NUMINAMATH_GPT_total_interest_is_68_l167_16792


namespace NUMINAMATH_GPT_ab_geq_3_plus_cd_l167_16719

theorem ab_geq_3_plus_cd (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 13) (h5 : a^2 + b^2 + c^2 + d^2 = 43) :
  a * b ≥ 3 + c * d := 
sorry

end NUMINAMATH_GPT_ab_geq_3_plus_cd_l167_16719


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l167_16790

theorem distance_between_parallel_lines : 
  ∀ (x y : ℝ), 
  (3 * x - 4 * y - 3 = 0) ∧ (6 * x - 8 * y + 5 = 0) → 
  ∃ d : ℝ, d = 11 / 10 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l167_16790


namespace NUMINAMATH_GPT_best_model_l167_16724

theorem best_model (R1 R2 R3 R4 : ℝ) (h1 : R1 = 0.55) (h2 : R2 = 0.65) (h3 : R3 = 0.79) (h4 : R4 = 0.95) :
  R4 > R3 ∧ R4 > R2 ∧ R4 > R1 :=
by {
  sorry
}

end NUMINAMATH_GPT_best_model_l167_16724


namespace NUMINAMATH_GPT_product_of_sums_of_squares_l167_16736

-- Given conditions as definitions
def sum_of_squares (a b : ℤ) : ℤ := a^2 + b^2

-- Prove that the product of two sums of squares is also a sum of squares
theorem product_of_sums_of_squares (a b n k : ℤ) (K P : ℤ) (hK : K = sum_of_squares a b) (hP : P = sum_of_squares n k) :
    K * P = (a * n + b * k)^2 + (a * k - b * n)^2 := 
by
  sorry

end NUMINAMATH_GPT_product_of_sums_of_squares_l167_16736


namespace NUMINAMATH_GPT_maximum_value_of_func_l167_16742

noncomputable def func (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

def domain_x (x : ℝ) : Prop := (1/3 : ℝ) ≤ x ∧ x ≤ (2/5 : ℝ)
def domain_y (y : ℝ) : Prop := (1/2 : ℝ) ≤ y ∧ y ≤ (5/8 : ℝ)

theorem maximum_value_of_func :
  ∀ (x y : ℝ), domain_x x → domain_y y → func x y ≤ (20 / 21 : ℝ) ∧ 
  (∃ (x y : ℝ), domain_x x ∧ domain_y y ∧ func x y = (20 / 21 : ℝ)) :=
by sorry

end NUMINAMATH_GPT_maximum_value_of_func_l167_16742


namespace NUMINAMATH_GPT_agatha_amount_left_l167_16747

noncomputable def initial_amount : ℝ := 60
noncomputable def frame_cost : ℝ := 15 * (1 - 0.10)
noncomputable def wheel_cost : ℝ := 25 * (1 - 0.05)
noncomputable def seat_cost : ℝ := 8 * (1 - 0.15)
noncomputable def handlebar_tape_cost : ℝ := 5
noncomputable def bell_cost : ℝ := 3
noncomputable def hat_cost : ℝ := 10 * (1 - 0.25)

noncomputable def total_cost : ℝ :=
  frame_cost + wheel_cost + seat_cost + handlebar_tape_cost + bell_cost + hat_cost

noncomputable def amount_left : ℝ := initial_amount - total_cost

theorem agatha_amount_left : amount_left = 0.45 :=
by
  -- interim calculations would go here
  sorry

end NUMINAMATH_GPT_agatha_amount_left_l167_16747


namespace NUMINAMATH_GPT_min_value_proof_l167_16700

noncomputable def min_value_condition (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 / (a + 3) + 1 / (b + 3) = 1 / 4)

theorem min_value_proof : ∃ a b : ℝ, min_value_condition a b ∧ a + 3 * b = 4 + 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_min_value_proof_l167_16700


namespace NUMINAMATH_GPT_least_alpha_condition_l167_16779

variables {a b α : ℝ}

theorem least_alpha_condition (a_gt_1 : a > 1) (b_gt_0 : b > 0) : 
  ∀ x, (x ≥ α) → (a + b) ^ x ≥ a ^ x + b ↔ α = 1 :=
by
  sorry

end NUMINAMATH_GPT_least_alpha_condition_l167_16779


namespace NUMINAMATH_GPT_non_empty_solution_set_l167_16702

theorem non_empty_solution_set (a : ℝ) (h : a > 0) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 :=
by
  sorry

end NUMINAMATH_GPT_non_empty_solution_set_l167_16702


namespace NUMINAMATH_GPT_series_sum_solution_l167_16770

noncomputable def series_sum (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a > b) (h₄ : b > c) : ℝ :=
  ∑' n : ℕ, (1 / ((n * c - (n - 1) * b) * ((n + 1) * c - n * b)))

theorem series_sum_solution (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a > b) (h₄ : b > c) :
  series_sum a b c h₀ h₁ h₂ h₃ h₄ = 1 / ((c - b) * c) := 
  sorry

end NUMINAMATH_GPT_series_sum_solution_l167_16770


namespace NUMINAMATH_GPT_binom_15_4_l167_16721

theorem binom_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_GPT_binom_15_4_l167_16721


namespace NUMINAMATH_GPT_correctness_check_l167_16760

noncomputable def questionD (x y : ℝ) : Prop := 
  3 * x^2 * y - 2 * y * x^2 = x^2 * y

theorem correctness_check (x y : ℝ) : questionD x y :=
by 
  sorry

end NUMINAMATH_GPT_correctness_check_l167_16760


namespace NUMINAMATH_GPT_enclosed_area_eq_two_l167_16788

noncomputable def enclosed_area : ℝ :=
  -∫ x in (2 * Real.pi / 3)..Real.pi, (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem enclosed_area_eq_two : enclosed_area = 2 := 
  sorry

end NUMINAMATH_GPT_enclosed_area_eq_two_l167_16788


namespace NUMINAMATH_GPT_division_proof_l167_16784

def dividend : ℕ := 144
def inner_divisor_num : ℕ := 12
def inner_divisor_denom : ℕ := 2
def final_divisor : ℕ := inner_divisor_num / inner_divisor_denom
def expected_result : ℕ := 24

theorem division_proof : (dividend / final_divisor) = expected_result := by
  sorry

end NUMINAMATH_GPT_division_proof_l167_16784


namespace NUMINAMATH_GPT_john_took_more_chickens_l167_16735

theorem john_took_more_chickens (john_chickens mary_chickens ray_chickens : ℕ)
  (h_ray : ray_chickens = 10)
  (h_mary : mary_chickens = ray_chickens + 6)
  (h_john : john_chickens = mary_chickens + 5) :
  john_chickens - ray_chickens = 11 := by
  sorry

end NUMINAMATH_GPT_john_took_more_chickens_l167_16735


namespace NUMINAMATH_GPT_k_range_l167_16780

noncomputable def range_of_k (k : ℝ): Prop :=
  ∀ x : ℤ, (x - 2) * (x + 1) > 0 ∧ (2 * x + 5) * (x + k) < 0 → x = -2

theorem k_range:
  (∃ k : ℝ, range_of_k k) ↔ -3 ≤ k ∧ k < 2 :=
by
  sorry

end NUMINAMATH_GPT_k_range_l167_16780


namespace NUMINAMATH_GPT_son_present_age_l167_16768

variable (S F : ℕ)

-- Given conditions
def father_age := F = S + 34
def future_age_rel := F + 2 = 2 * (S + 2)

-- Theorem to prove the son's current age
theorem son_present_age (h₁ : father_age S F) (h₂ : future_age_rel S F) : S = 32 := by
  sorry

end NUMINAMATH_GPT_son_present_age_l167_16768


namespace NUMINAMATH_GPT_problem_solution_l167_16744

theorem problem_solution :
  { x : ℝ // (x / 4 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x)) } = { x : ℝ // x ∈ Set.Ico (-4 : ℝ) (-(3 / 2) : ℝ) } :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l167_16744


namespace NUMINAMATH_GPT_total_population_expression_l167_16717

variables (b g t: ℕ)

-- Assuming the given conditions
def condition1 := b = 4 * g
def condition2 := g = 8 * t

-- The theorem to prove
theorem total_population_expression (h1 : condition1 b g) (h2 : condition2 g t) :
    b + g + t = 41 * b / 32 := sorry

end NUMINAMATH_GPT_total_population_expression_l167_16717


namespace NUMINAMATH_GPT_find_a_plus_b_plus_c_l167_16775

noncomputable def parabola_satisfies_conditions (a b c : ℝ) : Prop :=
  (∀ x, a * x ^ 2 + b * x + c ≥ 61) ∧
  (a * (1:ℝ) ^ 2 + b * (1:ℝ) + c = 0) ∧
  (a * (3:ℝ) ^ 2 + b * (3:ℝ) + c = 0)

theorem find_a_plus_b_plus_c (a b c : ℝ) 
  (h_minimum : parabola_satisfies_conditions a b c) :
  a + b + c = 0 := 
sorry

end NUMINAMATH_GPT_find_a_plus_b_plus_c_l167_16775


namespace NUMINAMATH_GPT_beadshop_wednesday_profit_l167_16746

theorem beadshop_wednesday_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ) :
  monday_fraction = 1/3 → tuesday_fraction = 1/4 → total_profit = 1200 →
  let monday_profit := monday_fraction * total_profit;
  let tuesday_profit := tuesday_fraction * total_profit;
  let wednesday_profit := total_profit - monday_profit - tuesday_profit;
  wednesday_profit = 500 :=
sorry

end NUMINAMATH_GPT_beadshop_wednesday_profit_l167_16746


namespace NUMINAMATH_GPT_b_ne_d_l167_16711

-- Conditions
def P (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def Q (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

def PQ_eq_QP_no_real_roots (a b c d : ℝ) : Prop := 
  ∀ (x : ℝ), P (Q x c d) a b ≠ Q (P x a b) c d

-- Goal
theorem b_ne_d (a b c d : ℝ) (h : PQ_eq_QP_no_real_roots a b c d) : b ≠ d := 
sorry

end NUMINAMATH_GPT_b_ne_d_l167_16711


namespace NUMINAMATH_GPT_projectile_height_30_in_2_seconds_l167_16740

theorem projectile_height_30_in_2_seconds (t y : ℝ) : 
  (y = -5 * t^2 + 25 * t ∧ y = 30) → t = 2 :=
by
  sorry

end NUMINAMATH_GPT_projectile_height_30_in_2_seconds_l167_16740


namespace NUMINAMATH_GPT_sweet_apples_percentage_is_75_l167_16753

noncomputable def percentage_sweet_apples 
  (price_sweet : ℝ) 
  (price_sour : ℝ) 
  (total_apples : ℕ) 
  (total_earnings : ℝ) 
  (percentage_sweet_expr : ℝ) :=
  price_sweet * percentage_sweet_expr + price_sour * (total_apples - percentage_sweet_expr) = total_earnings

theorem sweet_apples_percentage_is_75 :
  percentage_sweet_apples 0.5 0.1 100 40 75 :=
by
  unfold percentage_sweet_apples
  sorry

end NUMINAMATH_GPT_sweet_apples_percentage_is_75_l167_16753


namespace NUMINAMATH_GPT_roots_theorem_l167_16749

-- Definitions and Conditions
def root1 (a b p : ℝ) : Prop := 
  a + b = -p ∧ a * b = 1

def root2 (b c q : ℝ) : Prop := 
  b + c = -q ∧ b * c = 2

-- The theorem to prove
theorem roots_theorem (a b c p q : ℝ) (h1 : root1 a b p) (h2 : root2 b c q) : 
  (b - a) * (b - c) = p * q - 6 :=
sorry

end NUMINAMATH_GPT_roots_theorem_l167_16749


namespace NUMINAMATH_GPT_polynomial_without_xy_l167_16799

theorem polynomial_without_xy (k : ℝ) (x y : ℝ) :
  ¬(∃ c : ℝ, (x^2 + k * x * y + 4 * x - 2 * x * y + y^2 - 1 = c * x * y)) → k = 2 := by
  sorry

end NUMINAMATH_GPT_polynomial_without_xy_l167_16799


namespace NUMINAMATH_GPT_shaded_area_correct_l167_16723

def diameter := 3 -- inches
def pattern_length := 18 -- inches equivalent to 1.5 feet

def radius := diameter / 2 -- radius calculation

noncomputable def area_of_one_circle := Real.pi * (radius ^ 2)
def number_of_circles := pattern_length / diameter
noncomputable def total_shaded_area := number_of_circles * area_of_one_circle

theorem shaded_area_correct :
  total_shaded_area = 13.5 * Real.pi :=
  by
  sorry

end NUMINAMATH_GPT_shaded_area_correct_l167_16723


namespace NUMINAMATH_GPT_first_sequence_correct_second_sequence_correct_l167_16752

theorem first_sequence_correct (a1 a2 a3 a4 a5 : ℕ) (h1 : a1 = 12) (h2 : a2 = a1 + 4) (h3 : a3 = a2 + 4) (h4 : a4 = a3 + 4) (h5 : a5 = a4 + 4) :
  a4 = 24 ∧ a5 = 28 :=
by sorry

theorem second_sequence_correct (b1 b2 b3 b4 b5 : ℕ) (h1 : b1 = 2) (h2 : b2 = b1 * 2) (h3 : b3 = b2 * 2) (h4 : b4 = b3 * 2) (h5 : b5 = b4 * 2) :
  b4 = 16 ∧ b5 = 32 :=
by sorry

end NUMINAMATH_GPT_first_sequence_correct_second_sequence_correct_l167_16752


namespace NUMINAMATH_GPT_simplify_fraction_l167_16741

variable (a b y : ℝ)
variable (h1 : y = (a + 2 * b) / a)
variable (h2 : a ≠ -2 * b)
variable (h3 : a ≠ 0)

theorem simplify_fraction : (2 * a + 2 * b) / (a - 2 * b) = (y + 1) / (3 - y) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l167_16741


namespace NUMINAMATH_GPT_max_MB_value_l167_16798

open Real

-- Define the conditions of the problem
variables {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : sqrt 6 / 3 = sqrt (1 - b^2 / a^2))

-- Define the point M and the vertex B on the ellipse
variables (M : ℝ × ℝ) (hM : (M.1)^2 / (a)^2 + (M.2)^2 / (b)^2 = 1)
def B : ℝ × ℝ := (0, -b)

-- The task is to prove the maximum value of |MB| given the conditions
theorem max_MB_value : ∃ (maxMB : ℝ), maxMB = (3 * sqrt 2 / 2) * b :=
sorry

end NUMINAMATH_GPT_max_MB_value_l167_16798


namespace NUMINAMATH_GPT_faster_speed_l167_16708

theorem faster_speed (x : ℝ) (h1 : 40 = 8 * 5) (h2 : 60 = x * 5) : x = 12 :=
sorry

end NUMINAMATH_GPT_faster_speed_l167_16708


namespace NUMINAMATH_GPT_balance_after_6_months_l167_16701

noncomputable def final_balance : ℝ :=
  let balance_m1 := 5000 * (1 + 0.04 / 12)
  let balance_m2 := (balance_m1 + 1000) * (1 + 0.042 / 12)
  let balance_m3 := balance_m2 * (1 + 0.038 / 12)
  let balance_m4 := (balance_m3 - 1500) * (1 + 0.05 / 12)
  let balance_m5 := (balance_m4 + 750) * (1 + 0.052 / 12)
  let balance_m6 := (balance_m5 - 1000) * (1 + 0.045 / 12)
  balance_m6

theorem balance_after_6_months : final_balance = 4371.51 := sorry

end NUMINAMATH_GPT_balance_after_6_months_l167_16701


namespace NUMINAMATH_GPT_neither_rain_nor_snow_l167_16734

theorem neither_rain_nor_snow 
  (p_rain : ℚ)
  (p_snow : ℚ)
  (independent : Prop) 
  (h_rain : p_rain = 4/10)
  (h_snow : p_snow = 1/5)
  (h_independent : independent)
  : (1 - p_rain) * (1 - p_snow) = 12 / 25 := 
by
  sorry

end NUMINAMATH_GPT_neither_rain_nor_snow_l167_16734


namespace NUMINAMATH_GPT_students_remaining_l167_16755

theorem students_remaining (n : ℕ) (h1 : n = 1000)
  (h_beach : n / 2 = 500)
  (h_home : (n - n / 2) / 2 = 250) :
  n - (n / 2 + (n - n / 2) / 2) = 250 :=
by
  sorry

end NUMINAMATH_GPT_students_remaining_l167_16755


namespace NUMINAMATH_GPT_probability_green_face_l167_16707

def faces : ℕ := 6
def green_faces : ℕ := 3

theorem probability_green_face : (green_faces : ℚ) / (faces : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_green_face_l167_16707


namespace NUMINAMATH_GPT_steven_needs_more_seeds_l167_16756

def apple_seeds : Nat := 6
def pear_seeds : Nat := 2
def grape_seeds : Nat := 3
def apples_set_aside : Nat := 4
def pears_set_aside : Nat := 3
def grapes_set_aside : Nat := 9
def seeds_required : Nat := 60

theorem steven_needs_more_seeds : 
  seeds_required - (apples_set_aside * apple_seeds + pears_set_aside * pear_seeds + grapes_set_aside * grape_seeds) = 3 := by
  sorry

end NUMINAMATH_GPT_steven_needs_more_seeds_l167_16756


namespace NUMINAMATH_GPT_find_c_l167_16732

theorem find_c (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 310) : C = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l167_16732


namespace NUMINAMATH_GPT_range_of_a_l167_16797

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, x^2 - a * x + 1 = 0) ↔ -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l167_16797


namespace NUMINAMATH_GPT_Terrence_earns_l167_16796

theorem Terrence_earns :
  ∀ (J T E : ℝ), J + T + E = 90 ∧ J = T + 5 ∧ E = 25 → T = 30 :=
by
  intro J T E
  intro h
  obtain ⟨h₁, h₂, h₃⟩ := h
  sorry -- proof steps go here

end NUMINAMATH_GPT_Terrence_earns_l167_16796


namespace NUMINAMATH_GPT_find_m_l167_16783

def vector_collinear {α : Type*} [Field α] (a b : α × α) : Prop :=
  ∃ k : α, b = (k * (a.1), k * (a.2))

theorem find_m (m : ℝ) : 
  let a := (2, 3)
  let b := (-1, 2)
  vector_collinear (2 * m - 4, 3 * m + 8) (4, -1) → m = -2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_m_l167_16783


namespace NUMINAMATH_GPT_first_of_five_consecutive_sums_60_l167_16706

theorem first_of_five_consecutive_sums_60 (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 60) : n = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_first_of_five_consecutive_sums_60_l167_16706


namespace NUMINAMATH_GPT_min_value_abs_diff_l167_16767

-- Definitions of conditions
def is_in_interval (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 4

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  (b^2 - a^2 = 2) ∧ (c^2 - b^2 = 2)

-- Main statement
theorem min_value_abs_diff (x y z : ℝ)
  (h1 : is_in_interval x)
  (h2 : is_in_interval y)
  (h3 : is_in_interval z)
  (h4 : is_arithmetic_progression x y z) :
  |x - y| + |y - z| = 4 - 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_abs_diff_l167_16767


namespace NUMINAMATH_GPT_average_of_possible_values_of_x_l167_16748

theorem average_of_possible_values_of_x (x : ℝ) (h : (2 * x^2 + 3) = 21) : (x = 3 ∨ x = -3) → (3 + -3) / 2 = 0 := by
  sorry

end NUMINAMATH_GPT_average_of_possible_values_of_x_l167_16748


namespace NUMINAMATH_GPT_larger_integer_is_7sqrt14_l167_16729

theorem larger_integer_is_7sqrt14 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a / b = 7 / 3) (h2 : a * b = 294) : max a b = 7 * Real.sqrt 14 :=
by 
  sorry

end NUMINAMATH_GPT_larger_integer_is_7sqrt14_l167_16729


namespace NUMINAMATH_GPT_total_stickers_l167_16710

-- Definitions for the given conditions
def stickers_per_page : ℕ := 10
def number_of_pages : ℕ := 22

-- The theorem to be proven
theorem total_stickers : stickers_per_page * number_of_pages = 220 := by
  sorry

end NUMINAMATH_GPT_total_stickers_l167_16710


namespace NUMINAMATH_GPT_complex_pure_imaginary_l167_16731

theorem complex_pure_imaginary (a : ℝ) : 
  ((a^2 - 3*a + 2) = 0) → (a = 2) := 
  by 
  sorry

end NUMINAMATH_GPT_complex_pure_imaginary_l167_16731


namespace NUMINAMATH_GPT_correct_transformation_l167_16776

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : 
  (a / b) = ((a + 2 * a) / (b + 2 * b)) :=
by 
  sorry

end NUMINAMATH_GPT_correct_transformation_l167_16776


namespace NUMINAMATH_GPT_total_listening_days_l167_16730

-- Definitions
variables {x y z t : ℕ}

-- Problem statement
theorem total_listening_days (x y z t : ℕ) : (x + y + z) * t = ((x + y + z) * t) :=
by sorry

end NUMINAMATH_GPT_total_listening_days_l167_16730


namespace NUMINAMATH_GPT_ec_value_l167_16704

theorem ec_value (AB AD : ℝ) (EFGH1 EFGH2 : ℝ) (x : ℝ)
  (h1 : AB = 2)
  (h2 : AD = 1)
  (h3 : EFGH1 = 1 / 2 * AB)
  (h4 : EFGH2 = 1 / 2 * AD)
  (h5 : 1 + 2 * x = 1)
  : x = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_ec_value_l167_16704


namespace NUMINAMATH_GPT_market_value_of_10_percent_yielding_8_percent_stock_l167_16789

/-- 
Given:
1. The stock yields 8%.
2. It is a 10% stock, meaning the annual dividend per share is 10% of the face value.
3. Assume the face value of the stock is $100.

Prove:
The market value of the stock is $125.
-/
theorem market_value_of_10_percent_yielding_8_percent_stock
    (annual_dividend_per_share : ℝ)
    (face_value : ℝ)
    (dividend_yield : ℝ)
    (market_value_per_share : ℝ) 
    (h1 : face_value = 100)
    (h2 : annual_dividend_per_share = 0.10 * face_value)
    (h3 : dividend_yield = 8) :
    market_value_per_share = 125 := 
by
  /-
  Here, the following conditions are already given:
  1. face_value = 100
  2. annual_dividend_per_share = 0.10 * 100 = 10
  3. dividend_yield = 8
  
  We need to prove: market_value_per_share = 125
  -/
  sorry

end NUMINAMATH_GPT_market_value_of_10_percent_yielding_8_percent_stock_l167_16789


namespace NUMINAMATH_GPT_ratio_of_oranges_l167_16715

def num_good_oranges : ℕ := 24
def num_bad_oranges : ℕ := 8
def ratio_good_to_bad : ℕ := num_good_oranges / num_bad_oranges

theorem ratio_of_oranges : ratio_good_to_bad = 3 := by
  show 24 / 8 = 3
  sorry

end NUMINAMATH_GPT_ratio_of_oranges_l167_16715


namespace NUMINAMATH_GPT_length_real_axis_hyperbola_l167_16765

theorem length_real_axis_hyperbola (a : ℝ) (h : a^2 = 4) : 2 * a = 4 := by
  sorry

end NUMINAMATH_GPT_length_real_axis_hyperbola_l167_16765


namespace NUMINAMATH_GPT_intercepts_congruence_l167_16722

theorem intercepts_congruence (m : ℕ) (h : m = 29) (x0 y0 : ℕ) (hx : 0 ≤ x0 ∧ x0 < m) (hy : 0 ≤ y0 ∧ y0 < m) 
  (h1 : 5 * x0 % m = (2 * 0 + 3) % m)  (h2 : (5 * 0) % m = (2 * y0 + 3) % m) : 
  x0 + y0 = 31 := by
  sorry

end NUMINAMATH_GPT_intercepts_congruence_l167_16722


namespace NUMINAMATH_GPT_pencil_groups_l167_16772

theorem pencil_groups (total_pencils number_per_group number_of_groups : ℕ) 
  (h_total: total_pencils = 25) 
  (h_group: number_per_group = 5) 
  (h_eq: total_pencils = number_per_group * number_of_groups) : 
  number_of_groups = 5 :=
by
  sorry

end NUMINAMATH_GPT_pencil_groups_l167_16772


namespace NUMINAMATH_GPT_wine_with_cork_cost_is_2_10_l167_16716

noncomputable def cork_cost : ℝ := 0.05
noncomputable def wine_without_cork_cost : ℝ := cork_cost + 2.00
noncomputable def wine_with_cork_cost : ℝ := wine_without_cork_cost + cork_cost

theorem wine_with_cork_cost_is_2_10 : wine_with_cork_cost = 2.10 :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_wine_with_cork_cost_is_2_10_l167_16716


namespace NUMINAMATH_GPT_plates_difference_l167_16738

noncomputable def num_pots_angela : ℕ := 20
noncomputable def num_plates_angela (P : ℕ) := P
noncomputable def num_cutlery_angela (P : ℕ) := P / 2
noncomputable def num_pots_sharon : ℕ := 10
noncomputable def num_plates_sharon (P : ℕ) := 3 * P - 20
noncomputable def num_cutlery_sharon (P : ℕ) := P
noncomputable def total_kitchen_supplies_sharon (P : ℕ) := 
  num_pots_sharon + num_plates_sharon P + num_cutlery_sharon P

theorem plates_difference (P : ℕ) 
  (hP: num_plates_angela P > 3 * num_pots_angela) 
  (h_supplies: total_kitchen_supplies_sharon P = 254) :
  P - 3 * num_pots_angela = 6 := 
sorry

end NUMINAMATH_GPT_plates_difference_l167_16738


namespace NUMINAMATH_GPT_complement_union_example_l167_16750

open Set

theorem complement_union_example :
  ∀ (U A B : Set ℕ), 
  U = {1, 2, 3, 4, 5, 6, 7, 8} → 
  A = {1, 3, 5, 7} → 
  B = {2, 4, 5} → 
  (U \ (A ∪ B)) = {6, 8} := by 
  intros U A B hU hA hB
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_complement_union_example_l167_16750


namespace NUMINAMATH_GPT_fractions_addition_l167_16725

theorem fractions_addition : (1 / 6 - 5 / 12 + 3 / 8) = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fractions_addition_l167_16725


namespace NUMINAMATH_GPT_cubes_identity_l167_16795

theorem cubes_identity (a b c : ℝ) (h₁ : a + b + c = 15) (h₂ : ab + ac + bc = 40) : 
    a^3 + b^3 + c^3 - 3 * a * b * c = 1575 :=
by 
  sorry

end NUMINAMATH_GPT_cubes_identity_l167_16795


namespace NUMINAMATH_GPT_const_seq_is_arithmetic_not_geometric_l167_16727

-- Define the sequence
def const_seq (n : ℕ) : ℕ := 0

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, seq (n + 1) = seq n + d

-- Define what it means for a sequence to be geometric
def is_geometric_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = seq n * r

-- The proof statement
theorem const_seq_is_arithmetic_not_geometric :
  is_arithmetic_sequence const_seq ∧ ¬ is_geometric_sequence const_seq :=
by
  sorry

end NUMINAMATH_GPT_const_seq_is_arithmetic_not_geometric_l167_16727


namespace NUMINAMATH_GPT_harrys_total_cost_l167_16773

def cost_large_pizza : ℕ := 14
def cost_per_topping : ℕ := 2
def number_of_pizzas : ℕ := 2
def number_of_toppings_per_pizza : ℕ := 3
def tip_percentage : ℚ := 0.25

def total_cost (c_pizza c_topping tip_percent : ℚ) (n_pizza n_topping : ℕ) : ℚ :=
  let inital_cost := (c_pizza + c_topping * n_topping) * n_pizza
  let tip := inital_cost * tip_percent
  inital_cost + tip

theorem harrys_total_cost : total_cost 14 2 0.25 2 3 = 50 := 
  sorry

end NUMINAMATH_GPT_harrys_total_cost_l167_16773


namespace NUMINAMATH_GPT_line_intersects_y_axis_at_point_intersection_at_y_axis_l167_16771

theorem line_intersects_y_axis_at_point :
  ∃ y, 5 * 0 - 7 * y = 35 := sorry

theorem intersection_at_y_axis :
  (∃ y, 5 * 0 - 7 * y = 35) → 0 - 7 * (-5) = 35 := sorry

end NUMINAMATH_GPT_line_intersects_y_axis_at_point_intersection_at_y_axis_l167_16771


namespace NUMINAMATH_GPT_fraction_equiv_l167_16769

def repeating_decimal := 0.4 + (37 / 1000) / (1 - 1 / 1000)

theorem fraction_equiv : repeating_decimal = 43693 / 99900 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equiv_l167_16769
