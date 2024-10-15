import Mathlib

namespace NUMINAMATH_GPT_number_of_lucky_tickets_l361_36105

def is_leningrad_lucky (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₁ + a₂ + a₃ = a₄ + a₅ + a₆

def is_moscow_lucky (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₂ + a₄ + a₆ = a₁ + a₃ + a₅

def is_symmetric (a₂ a₅ : ℕ) : Prop :=
  a₂ = a₅

def is_valid_ticket (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  is_leningrad_lucky a₁ a₂ a₃ a₄ a₅ a₆ ∧
  is_moscow_lucky a₁ a₂ a₃ a₄ a₅ a₆ ∧
  is_symmetric a₂ a₅

theorem number_of_lucky_tickets : 
  ∃ n : ℕ, n = 6700 ∧ 
  (∀ a₁ a₂ a₃ a₄ a₅ a₆ : ℕ, 
    0 ≤ a₁ ∧ a₁ ≤ 9 ∧
    0 ≤ a₂ ∧ a₂ ≤ 9 ∧
    0 ≤ a₃ ∧ a₃ ≤ 9 ∧
    0 ≤ a₄ ∧ a₄ ≤ 9 ∧
    0 ≤ a₅ ∧ a₅ ≤ 9 ∧
    0 ≤ a₆ ∧ a₆ ≤ 9 →
    is_valid_ticket a₁ a₂ a₃ a₄ a₅ a₆ →
    n = 6700) := sorry

end NUMINAMATH_GPT_number_of_lucky_tickets_l361_36105


namespace NUMINAMATH_GPT_intersection_of_line_with_x_axis_l361_36194

theorem intersection_of_line_with_x_axis 
  (k : ℝ) 
  (h : ∀ x y : ℝ, y = k * x + 4 → (x = -1 ∧ y = 2)) 
  : ∃ x : ℝ, (2 : ℝ) * x + 4 = 0 ∧ x = -2 :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_line_with_x_axis_l361_36194


namespace NUMINAMATH_GPT_number_div_by_3_l361_36171

theorem number_div_by_3 (x : ℕ) (h : 54 = x - 39) : x / 3 = 31 :=
by
  sorry

end NUMINAMATH_GPT_number_div_by_3_l361_36171


namespace NUMINAMATH_GPT_brownies_cut_into_pieces_l361_36174

theorem brownies_cut_into_pieces (total_amount_made : ℕ) (pans : ℕ) (cost_per_brownie : ℕ) (brownies_sold : ℕ) 
  (h1 : total_amount_made = 32) (h2 : pans = 2) (h3 : cost_per_brownie = 2) (h4 : brownies_sold = total_amount_made / cost_per_brownie) :
  16 = brownies_sold :=
by
  sorry

end NUMINAMATH_GPT_brownies_cut_into_pieces_l361_36174


namespace NUMINAMATH_GPT_steven_owes_jeremy_l361_36128

-- Definitions for the conditions
def base_payment_per_room := (13 : ℚ) / 3
def rooms_cleaned := (5 : ℚ) / 2
def additional_payment_per_room := (1 : ℚ) / 2

-- Define the total amount of money Steven owes Jeremy
def total_payment (base_payment_per_room rooms_cleaned additional_payment_per_room : ℚ) : ℚ :=
  let base_payment := base_payment_per_room * rooms_cleaned
  let additional_payment := if rooms_cleaned > 2 then additional_payment_per_room * rooms_cleaned else 0
  base_payment + additional_payment

-- The statement to prove
theorem steven_owes_jeremy :
  total_payment base_payment_per_room rooms_cleaned additional_payment_per_room = 145 / 12 :=
by
  sorry

end NUMINAMATH_GPT_steven_owes_jeremy_l361_36128


namespace NUMINAMATH_GPT_number_of_truthful_dwarfs_l361_36121

def num_dwarfs : Nat := 10

def likes_vanilla : Nat := num_dwarfs

def likes_chocolate : Nat := num_dwarfs / 2

def likes_fruit : Nat := 1

theorem number_of_truthful_dwarfs : 
  ∃ t l : Nat, 
  t + l = num_dwarfs ∧  -- total number of dwarfs
  t + 2 * l = likes_vanilla + likes_chocolate + likes_fruit ∧  -- total number of hand raises
  t = 4 :=  -- number of truthful dwarfs
  sorry

end NUMINAMATH_GPT_number_of_truthful_dwarfs_l361_36121


namespace NUMINAMATH_GPT_tournament_player_count_l361_36153

theorem tournament_player_count (n : ℕ) :
  (∃ points_per_game : ℕ, points_per_game = (n * (n - 1)) / 2) →
  (∃ T : ℕ, T = 90) →
  (n * (n - 1)) / 4 = 90 →
  n = 19 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_tournament_player_count_l361_36153


namespace NUMINAMATH_GPT_sum_four_digit_integers_l361_36169

def a := 1000
def l := 9999
def n := 9999 - 1000 + 1
def S (n : ℕ) (a : ℕ) (l : ℕ) := n / 2 * (a + l)

theorem sum_four_digit_integers : S n a l = 49495500 :=
by
  sorry

end NUMINAMATH_GPT_sum_four_digit_integers_l361_36169


namespace NUMINAMATH_GPT_measure_diagonal_without_pythagorean_theorem_l361_36190

variables (a b c : ℝ)

-- Definition of the function to measure the diagonal distance
def diagonal_method (a b c : ℝ) : ℝ :=
  -- by calculating the hypotenuse scaled by sqrt(3), we ignore using the Pythagorean theorem directly
  sorry

-- Calculate distance by arranging bricks
theorem measure_diagonal_without_pythagorean_theorem (distance_extreme_corners : ℝ) :
  distance_extreme_corners = (diagonal_method a b c) :=
  sorry

end NUMINAMATH_GPT_measure_diagonal_without_pythagorean_theorem_l361_36190


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_l361_36141

theorem radius_of_inscribed_circle (a b x : ℝ) (hx : 0 < x) 
  (h_side_length : a > 20) 
  (h_TM : a = x + 8) 
  (h_OM : b = x + 9) 
  (h_Pythagorean : (a - 8)^2 + (b - 9)^2 = x^2) :
  x = 29 :=
by
  -- Assume all conditions and continue to the proof part.
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_l361_36141


namespace NUMINAMATH_GPT_contractor_engaged_days_l361_36198

theorem contractor_engaged_days
  (earnings_per_day : ℤ)
  (fine_per_day : ℤ)
  (total_earnings : ℤ)
  (absent_days : ℤ)
  (days_worked : ℤ) 
  (h1 : earnings_per_day = 25)
  (h2 : fine_per_day = 15 / 2)
  (h3 : total_earnings = 620)
  (h4 : absent_days = 4)
  (h5 : total_earnings = earnings_per_day * days_worked - fine_per_day * absent_days) :
  days_worked = 26 := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_contractor_engaged_days_l361_36198


namespace NUMINAMATH_GPT_cheerleaders_uniforms_l361_36160

theorem cheerleaders_uniforms (total_cheerleaders : ℕ) (size_6_cheerleaders : ℕ) (half_size_6_cheerleaders : ℕ) (size_2_cheerleaders : ℕ) : 
  total_cheerleaders = 19 →
  size_6_cheerleaders = 10 →
  half_size_6_cheerleaders = size_6_cheerleaders / 2 →
  size_2_cheerleaders = total_cheerleaders - (size_6_cheerleaders + half_size_6_cheerleaders) →
  size_2_cheerleaders = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cheerleaders_uniforms_l361_36160


namespace NUMINAMATH_GPT_no_solutions_in_domain_l361_36146

-- Define the function g
def g (x : ℝ) : ℝ := -0.5 * x^2 + x + 3

-- Define the condition on the domain of g
def in_domain (x : ℝ) : Prop := x ≥ -3 ∧ x ≤ 3

-- State the theorem to be proved
theorem no_solutions_in_domain :
  ∀ x : ℝ, in_domain x → ¬ (g (g x) = 3) :=
by
  -- Provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_no_solutions_in_domain_l361_36146


namespace NUMINAMATH_GPT_sophie_perceived_height_in_mirror_l361_36138

noncomputable def inch_to_cm : ℝ := 2.5

noncomputable def sophie_height_in_inches : ℝ := 50

noncomputable def sophie_height_in_cm := sophie_height_in_inches * inch_to_cm

noncomputable def perceived_height := sophie_height_in_cm * 2

theorem sophie_perceived_height_in_mirror : perceived_height = 250 :=
by
  unfold perceived_height
  unfold sophie_height_in_cm
  unfold sophie_height_in_inches
  unfold inch_to_cm
  sorry

end NUMINAMATH_GPT_sophie_perceived_height_in_mirror_l361_36138


namespace NUMINAMATH_GPT_common_difference_is_7_l361_36163

-- Define the arithmetic sequence with common difference d
def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Define the conditions
variables (a1 d : ℕ)

-- Define the conditions provided in the problem
def condition1 := (arithmetic_seq a1 d 3) + (arithmetic_seq a1 d 6) = 11
def condition2 := (arithmetic_seq a1 d 5) + (arithmetic_seq a1 d 8) = 39

-- Prove that the common difference d is 7
theorem common_difference_is_7 : condition1 a1 d → condition2 a1 d → d = 7 :=
by
  intros cond1 cond2
  sorry

end NUMINAMATH_GPT_common_difference_is_7_l361_36163


namespace NUMINAMATH_GPT_find_set_B_l361_36158

def A : Set ℕ := {1, 2}
def B : Set (Set ℕ) := { x | x ⊆ A }

theorem find_set_B : B = { ∅, {1}, {2}, {1, 2} } :=
by
  sorry

end NUMINAMATH_GPT_find_set_B_l361_36158


namespace NUMINAMATH_GPT_exterior_angle_hexagon_l361_36108

theorem exterior_angle_hexagon (θ : ℝ) (hθ : θ = 60) (h_sum : θ * 6 = 360) : n = 6 :=
sorry

end NUMINAMATH_GPT_exterior_angle_hexagon_l361_36108


namespace NUMINAMATH_GPT_symbols_invariance_l361_36199

def final_symbol_invariant (symbols : List Char) : Prop :=
  ∀ (erase : List Char → List Char), 
  (∀ (l : List Char), 
    (erase l = List.cons '+' (List.tail (List.tail l)) ∨ 
    erase l = List.cons '-' (List.tail (List.tail l))) → 
    erase (erase l) = List.cons '+' (List.tail (List.tail (erase l))) ∨ 
    erase (erase l) = List.cons '-' (List.tail (List.tail (erase l)))) →
  (symbols = []) ∨ (symbols = ['+']) ∨ (symbols = ['-'])

theorem symbols_invariance (symbols : List Char) (h : final_symbol_invariant symbols) : 
  ∃ (s : Char), s = '+' ∨ s = '-' :=
  sorry

end NUMINAMATH_GPT_symbols_invariance_l361_36199


namespace NUMINAMATH_GPT_minimum_value_exists_l361_36135

-- Definitions of the components
noncomputable def quadratic_expression (k x y : ℝ) : ℝ := 
  9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 9 * y + 12

theorem minimum_value_exists (k : ℝ) :
  (∃ x y : ℝ, quadratic_expression k x y = 0) ↔ k = 2 := 
sorry

end NUMINAMATH_GPT_minimum_value_exists_l361_36135


namespace NUMINAMATH_GPT_sushi_cost_l361_36142

variable (x : ℕ)

theorem sushi_cost (h1 : 9 * x = 180) : x + (9 * x) = 200 :=
by 
  sorry

end NUMINAMATH_GPT_sushi_cost_l361_36142


namespace NUMINAMATH_GPT_probability_first_third_fifth_correct_probability_exactly_three_hits_correct_l361_36176

noncomputable def probability_first_third_fifth_hit : ℚ :=
  (3 / 5) * (2 / 5) * (3 / 5) * (2 / 5) * (3 / 5)

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  ↑(Nat.factorial n) / (↑(Nat.factorial k) * ↑(Nat.factorial (n - k)))

noncomputable def probability_exactly_three_hits : ℚ :=
  binomial_coefficient 5 3 * (3 / 5)^3 * (2 / 5)^2

theorem probability_first_third_fifth_correct :
  probability_first_third_fifth_hit = 108 / 3125 :=
by sorry

theorem probability_exactly_three_hits_correct :
  probability_exactly_three_hits = 216 / 625 :=
by sorry

end NUMINAMATH_GPT_probability_first_third_fifth_correct_probability_exactly_three_hits_correct_l361_36176


namespace NUMINAMATH_GPT_max_value_of_xyz_l361_36167

theorem max_value_of_xyz (x y z : ℝ) (h : x + 3 * y + z = 5) : xy + xz + yz ≤ 125 / 4 := 
sorry

end NUMINAMATH_GPT_max_value_of_xyz_l361_36167


namespace NUMINAMATH_GPT_part_I_part_II_l361_36137

-- Let the volume V of the tetrahedron ABCD be given
def V : ℝ := sorry

-- Areas of the faces opposite vertices A, B, C, D
def S_A : ℝ := sorry
def S_B : ℝ := sorry
def S_C : ℝ := sorry
def S_D : ℝ := sorry

-- Definitions of the edge lengths and angles
def a : ℝ := sorry -- BC
def a' : ℝ := sorry -- DA
def b : ℝ := sorry -- CA
def b' : ℝ := sorry -- DB
def c : ℝ := sorry -- AB
def c' : ℝ := sorry -- DC
def alpha : ℝ := sorry -- Angle between BC and DA
def beta : ℝ := sorry -- Angle between CA and DB
def gamma : ℝ := sorry -- Angle between AB and DC

theorem part_I : 
  S_A^2 + S_B^2 + S_C^2 + S_D^2 = 
  (1 / 4) * ((a * a' * Real.sin alpha)^2 + (b * b' * Real.sin beta)^2 + (c * c' * Real.sin gamma)^2) := 
  sorry

theorem part_II : 
  S_A^2 + S_B^2 + S_C^2 + S_D^2 ≥ 9 * (3 * V^4)^(1/3) :=
  sorry

end NUMINAMATH_GPT_part_I_part_II_l361_36137


namespace NUMINAMATH_GPT_cost_combination_exists_l361_36151

/-!
Given:
- Nadine spent a total of $105.
- The table costs $34.
- The mirror costs $15.
- The lamp costs $6.
- The total cost of the 2 chairs and 3 decorative vases is $50.

Prove:
- There are multiple combinations of individual chair cost (C) and individual vase cost (V) such that 2 * C + 3 * V = 50.
-/

theorem cost_combination_exists :
  ∃ (C V : ℝ), 2 * C + 3 * V = 50 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_combination_exists_l361_36151


namespace NUMINAMATH_GPT_algebraic_expression_value_l361_36119

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 - x - 1 = 5) : 6 * x^2 - 3 * x - 9 = 9 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l361_36119


namespace NUMINAMATH_GPT_joe_first_lift_weight_l361_36156

variables (x y : ℕ)

theorem joe_first_lift_weight (h1 : x + y = 600) (h2 : 2 * x = y + 300) : x = 300 :=
by
  sorry

end NUMINAMATH_GPT_joe_first_lift_weight_l361_36156


namespace NUMINAMATH_GPT_value_of_a_minus_b_l361_36140

theorem value_of_a_minus_b (a b : ℝ) (h1 : (a + b)^2 = 49) (h2 : ab = 6) : a - b = 5 ∨ a - b = -5 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l361_36140


namespace NUMINAMATH_GPT_total_books_l361_36120

def number_of_zoology_books : ℕ := 16
def number_of_botany_books : ℕ := 4 * number_of_zoology_books

theorem total_books : number_of_zoology_books + number_of_botany_books = 80 := by
  sorry

end NUMINAMATH_GPT_total_books_l361_36120


namespace NUMINAMATH_GPT_pentagon_largest_angle_l361_36182

variable (F G H I J : ℝ)

-- Define the conditions given in the problem
axiom angle_sum : F + G + H + I + J = 540
axiom angle_F : F = 80
axiom angle_G : G = 100
axiom angle_HI : H = I
axiom angle_J : J = 2 * H + 20

-- Statement that the largest angle in the pentagon is 190°
theorem pentagon_largest_angle : max F (max G (max H (max I J))) = 190 :=
sorry

end NUMINAMATH_GPT_pentagon_largest_angle_l361_36182


namespace NUMINAMATH_GPT_possible_values_of_sum_l361_36185

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end NUMINAMATH_GPT_possible_values_of_sum_l361_36185


namespace NUMINAMATH_GPT_systematic_sampling_40th_number_l361_36100

theorem systematic_sampling_40th_number
  (total_students sample_size : ℕ)
  (first_group_start first_group_end selected_first_group_number steps : ℕ)
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_group_start = 1)
  (h4 : first_group_end = 20)
  (h5 : selected_first_group_number = 15)
  (h6 : steps = total_students / sample_size)
  (h7 : first_group_end - first_group_start + 1 = steps)
  : (selected_first_group_number + steps * (40 - 1)) = 795 :=
sorry

end NUMINAMATH_GPT_systematic_sampling_40th_number_l361_36100


namespace NUMINAMATH_GPT_price_increase_needed_l361_36118

theorem price_increase_needed (P : ℝ) (hP : P > 0) : (100 * ((P / (0.85 * P)) - 1)) = 17.65 :=
by
  sorry

end NUMINAMATH_GPT_price_increase_needed_l361_36118


namespace NUMINAMATH_GPT_number_of_regions_on_sphere_l361_36165

theorem number_of_regions_on_sphere (n : ℕ) (h : ∀ {a b c: ℤ}, a ≠ b → b ≠ c → a ≠ c → True) : 
  ∃ a_n, a_n = n^2 - n + 2 := 
by
  sorry

end NUMINAMATH_GPT_number_of_regions_on_sphere_l361_36165


namespace NUMINAMATH_GPT_fewer_onions_than_tomatoes_and_corn_l361_36157

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

theorem fewer_onions_than_tomatoes_and_corn :
  (tomatoes + corn - onions) = 5200 :=
by
  sorry

end NUMINAMATH_GPT_fewer_onions_than_tomatoes_and_corn_l361_36157


namespace NUMINAMATH_GPT_value_of_b_l361_36116

theorem value_of_b (a b : ℝ) (h1 : 2 * a + 1 = 1) (h2 : b - a = 1) : b = 1 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_b_l361_36116


namespace NUMINAMATH_GPT_total_movie_hours_l361_36186

-- Definitions
def JoyceMovie : ℕ := 12 -- Joyce's favorite movie duration in hours
def MichaelMovie : ℕ := 10 -- Michael's favorite movie duration in hours
def NikkiMovie : ℕ := 30 -- Nikki's favorite movie duration in hours
def RynMovie : ℕ := 24 -- Ryn's favorite movie duration in hours

-- Condition translations
def Joyce_movie_condition : Prop := JoyceMovie = MichaelMovie + 2
def Nikki_movie_condition : Prop := NikkiMovie = 3 * MichaelMovie
def Ryn_movie_condition : Prop := RynMovie = (4 * NikkiMovie) / 5
def Nikki_movie_given : Prop := NikkiMovie = 30

-- The theorem to prove
theorem total_movie_hours : Joyce_movie_condition ∧ Nikki_movie_condition ∧ Ryn_movie_condition ∧ Nikki_movie_given → 
  (JoyceMovie + MichaelMovie + NikkiMovie + RynMovie = 76) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_total_movie_hours_l361_36186


namespace NUMINAMATH_GPT_log5_6_identity_l361_36180

noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.log 10 / Real.log 3

theorem log5_6_identity :
  Real.log 6 / Real.log 5 = ((a * b) + 1) / (b - (a * b)) :=
by sorry

end NUMINAMATH_GPT_log5_6_identity_l361_36180


namespace NUMINAMATH_GPT_james_money_left_no_foreign_currency_needed_l361_36148

noncomputable def JameMoneyLeftAfterPurchase : ℝ :=
  let usd_bills := 50 + 20 + 5 + 1 + 20 + 10 -- USD bills and coins
  let euro_in_usd := 5 * 1.20               -- €5 bill to USD
  let pound_in_usd := 2 * 1.35 - 0.8 / 100 * (2 * 1.35) -- £2 coin to USD after fee
  let yen_in_usd := 100 * 0.009 - 1.5 / 100 * (100 * 0.009) -- ¥100 coin to USD after fee
  let franc_in_usd := 2 * 1.08 - 1 / 100 * (2 * 1.08) -- 2₣ coins to USD after fee
  let total_usd := usd_bills + euro_in_usd + pound_in_usd + yen_in_usd + franc_in_usd
  let present_cost_with_tax := 88 * 1.08   -- Present cost after 8% tax
  total_usd - present_cost_with_tax        -- Amount left after purchasing the present

theorem james_money_left :
  JameMoneyLeftAfterPurchase = 22.6633 :=
by
  sorry

theorem no_foreign_currency_needed :
  (0 : ℝ)  = 0 :=
by
  sorry

end NUMINAMATH_GPT_james_money_left_no_foreign_currency_needed_l361_36148


namespace NUMINAMATH_GPT_local_minimum_at_2_l361_36173

noncomputable def f (x : ℝ) : ℝ := (2 / x) + Real.log x

theorem local_minimum_at_2 : ∃ δ > 0, ∀ y, abs (y - 2) < δ → f y ≥ f 2 := by
  sorry

end NUMINAMATH_GPT_local_minimum_at_2_l361_36173


namespace NUMINAMATH_GPT_average_marks_of_all_students_l361_36139

theorem average_marks_of_all_students :
  (22 * 40 + 28 * 60) / (22 + 28) = 51.2 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_of_all_students_l361_36139


namespace NUMINAMATH_GPT_least_area_in_rectangle_l361_36181

theorem least_area_in_rectangle
  (x y : ℤ)
  (h1 : 2 * (x + y) = 150)
  (h2 : x > 0)
  (h3 : y > 0) :
  ∃ x y : ℤ, (2 * (x + y) = 150) ∧ (x * y = 74) := by
  sorry

end NUMINAMATH_GPT_least_area_in_rectangle_l361_36181


namespace NUMINAMATH_GPT_arc_length_calc_l361_36154

-- Defining the conditions
def circle_radius := 12 -- radius OR
def angle_RIP := 30 -- angle in degrees

-- Defining the goal
noncomputable def arc_length_RP := 4 * Real.pi -- length of arc RP

-- The statement to prove
theorem arc_length_calc :
  arc_length_RP = 4 * Real.pi :=
sorry

end NUMINAMATH_GPT_arc_length_calc_l361_36154


namespace NUMINAMATH_GPT_james_needs_to_sell_12_coins_l361_36161

theorem james_needs_to_sell_12_coins:
  ∀ (num_coins : ℕ) (initial_price new_price : ℝ),
  num_coins = 20 ∧ initial_price = 15 ∧ new_price = initial_price + (2 / 3) * initial_price →
  (num_coins * initial_price) / new_price = 12 :=
by
  intros num_coins initial_price new_price h
  obtain ⟨hc1, hc2, hc3⟩ := h
  sorry

end NUMINAMATH_GPT_james_needs_to_sell_12_coins_l361_36161


namespace NUMINAMATH_GPT_order_of_f_l361_36125

-- Define the function f
variables {f : ℝ → ℝ}

-- Definition of even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Definition of monotonic increasing function on [0, +∞)
def monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y, (0 ≤ x ∧ 0 ≤ y ∧ x ≤ y) → f x ≤ f y

-- The main problem statement
theorem order_of_f (h_even : even_function f) (h_mono : monotonically_increasing_on_nonneg f) :
  f (-π) > f 3 ∧ f 3 > f (-2) :=
  sorry

end NUMINAMATH_GPT_order_of_f_l361_36125


namespace NUMINAMATH_GPT_find_y_l361_36162

theorem find_y (y : ℝ) (h : (y + 10 + (5 * y) + 4 + (3 * y) + 12) / 3 = 6 * y - 8) :
  y = 50 / 9 := by
  sorry

end NUMINAMATH_GPT_find_y_l361_36162


namespace NUMINAMATH_GPT_shorter_leg_of_right_triangle_l361_36112

theorem shorter_leg_of_right_triangle (a b : ℕ) (h1 : a < b)
    (h2 : a^2 + b^2 = 65^2) : a = 16 :=
sorry

end NUMINAMATH_GPT_shorter_leg_of_right_triangle_l361_36112


namespace NUMINAMATH_GPT_probability_segments_length_l361_36164

theorem probability_segments_length (x y : ℝ) : 
    80 ≥ x ∧ x ≥ 20 ∧ 80 ≥ y ∧ y ≥ 20 ∧ 80 ≥ 80 - x - y ∧ 80 - x - y ≥ 20 → 
    (∃ (s : ℝ), s = (200 / 3200) ∧ s = (1 / 16)) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_probability_segments_length_l361_36164


namespace NUMINAMATH_GPT_standard_equation_of_tangent_circle_l361_36101

theorem standard_equation_of_tangent_circle (r h k : ℝ)
  (h_r : r = 1) 
  (h_k : k = 1) 
  (h_center_quadrant : h > 0 ∧ k > 0)
  (h_tangent_x_axis : k = r) 
  (h_tangent_line : r = abs (4 * h - 3) / 5)
  : (x - 2)^2 + (y - 1)^2 = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_standard_equation_of_tangent_circle_l361_36101


namespace NUMINAMATH_GPT_find_g_l361_36147

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g (g : ℝ → ℝ)
  (H : ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = x + y + 4) :
  g = fun x => x + 5 :=
by
  sorry

end NUMINAMATH_GPT_find_g_l361_36147


namespace NUMINAMATH_GPT_quadratic_roots_prime_distinct_l361_36110

theorem quadratic_roots_prime_distinct (a α β m : ℕ) (h1: α ≠ β) (h2: Nat.Prime α) (h3: Nat.Prime β) (h4: α + β = m / a) (h5: α * β = 1996 / a) :
    a = 2 := by
  sorry

end NUMINAMATH_GPT_quadratic_roots_prime_distinct_l361_36110


namespace NUMINAMATH_GPT_certain_number_is_negative_425_l361_36102

theorem certain_number_is_negative_425 (x : ℝ) :
  (3 - (1/5) * x = 88) ∧ (4 - (1/7) * 210 = -26) → x = -425 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_negative_425_l361_36102


namespace NUMINAMATH_GPT_Julia_watch_collection_l361_36193

section
variable (silver_watches : ℕ) (bronze_watches : ℕ) (gold_watches : ℕ) (total_watches : ℕ)

theorem Julia_watch_collection :
  silver_watches = 20 →
  bronze_watches = 3 * silver_watches →
  gold_watches = 10 * (silver_watches + bronze_watches) / 100 →
  total_watches = silver_watches + bronze_watches + gold_watches →
  total_watches = 88 :=
by
  intros
  sorry
end

end NUMINAMATH_GPT_Julia_watch_collection_l361_36193


namespace NUMINAMATH_GPT_number_of_C_animals_l361_36136

-- Define the conditions
def A : ℕ := 45
def B : ℕ := 32
def C : ℕ := 5

-- Define the theorem that we need to prove
theorem number_of_C_animals : B + C = A - 8 :=
by
  -- placeholder to complete the proof (not part of the problem's requirement)
  sorry

end NUMINAMATH_GPT_number_of_C_animals_l361_36136


namespace NUMINAMATH_GPT_complement_U_A_l361_36150

def U : Set ℝ := { x | x^2 ≤ 4 }
def A : Set ℝ := { x | abs (x + 1) ≤ 1 }

theorem complement_U_A :
  (U \ A) = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_complement_U_A_l361_36150


namespace NUMINAMATH_GPT_complement_intersection_l361_36132

def U : Set ℝ := Set.univ
def M : Set ℝ := {y | 0 ≤ y ∧ y ≤ 2}
def N : Set ℝ := {x | (x < -3) ∨ (x > 0)}

theorem complement_intersection :
  (Set.univ \ M) ∩ N = {x | x < -3 ∨ x > 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l361_36132


namespace NUMINAMATH_GPT_not_possible_to_fill_grid_l361_36149

theorem not_possible_to_fill_grid :
  ¬ ∃ (f : Fin 7 → Fin 7 → ℝ), ∀ i j : Fin 7,
    ((if j > 0 then f i (j - 1) else 0) +
     (if j < 6 then f i (j + 1) else 0) +
     (if i > 0 then f (i - 1) j else 0) +
     (if i < 6 then f (i + 1) j else 0)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_not_possible_to_fill_grid_l361_36149


namespace NUMINAMATH_GPT_stream_speed_l361_36195

theorem stream_speed (v : ℝ) (boat_speed : ℝ) (distance : ℝ) (time : ℝ) 
    (h1 : boat_speed = 10) 
    (h2 : distance = 54) 
    (h3 : time = 3) 
    (h4 : distance = (boat_speed + v) * time) : 
    v = 8 :=
by
  sorry

end NUMINAMATH_GPT_stream_speed_l361_36195


namespace NUMINAMATH_GPT_days_collected_money_l361_36123

-- Defining constants and parameters based on the conditions
def households_per_day : ℕ := 20
def money_per_pair : ℕ := 40
def total_money_collected : ℕ := 2000
def money_from_households : ℕ := (households_per_day / 2) * money_per_pair

-- The theorem that needs to be proven
theorem days_collected_money :
  (total_money_collected / money_from_households) = 5 :=
sorry -- Proof not provided

end NUMINAMATH_GPT_days_collected_money_l361_36123


namespace NUMINAMATH_GPT_not_prime_1001_base_l361_36172

theorem not_prime_1001_base (n : ℕ) (h : n ≥ 2) : ¬ Nat.Prime (n^3 + 1) :=
sorry

end NUMINAMATH_GPT_not_prime_1001_base_l361_36172


namespace NUMINAMATH_GPT_f_of_5_l361_36178

/- The function f(x) is defined by f(x) = x^2 - x. Prove that f(5) = 20. -/
def f (x : ℤ) : ℤ := x^2 - x

theorem f_of_5 : f 5 = 20 := by
  sorry

end NUMINAMATH_GPT_f_of_5_l361_36178


namespace NUMINAMATH_GPT_minimum_area_for_rectangle_l361_36109

theorem minimum_area_for_rectangle 
(length width : ℝ) 
(h_length_min : length = 4 - 0.5) 
(h_width_min : width = 5 - 1) :
length * width = 14 := 
by 
  simp [h_length_min, h_width_min]
  sorry

end NUMINAMATH_GPT_minimum_area_for_rectangle_l361_36109


namespace NUMINAMATH_GPT_orange_juice_fraction_l361_36168

theorem orange_juice_fraction 
    (capacity1 capacity2 : ℕ)
    (orange_fraction1 orange_fraction2 : ℚ)
    (h_capacity1 : capacity1 = 800)
    (h_capacity2 : capacity2 = 700)
    (h_orange_fraction1 : orange_fraction1 = 1/4)
    (h_orange_fraction2 : orange_fraction2 = 1/3) :
    (capacity1 * orange_fraction1 + capacity2 * orange_fraction2) / (capacity1 + capacity2) = 433.33 / 1500 :=
by sorry

end NUMINAMATH_GPT_orange_juice_fraction_l361_36168


namespace NUMINAMATH_GPT_compare_fractions_l361_36129

variable {a b : ℝ}

theorem compare_fractions (h1 : 3 * a > b) (h2 : b > 0) :
  (a / b) > ((a + 1) / (b + 3)) :=
by
  sorry

end NUMINAMATH_GPT_compare_fractions_l361_36129


namespace NUMINAMATH_GPT_cos_min_sin_eq_neg_sqrt_seven_half_l361_36133

variable (θ : ℝ)

theorem cos_min_sin_eq_neg_sqrt_seven_half (h1 : Real.sin θ + Real.cos θ = 0.5)
    (h2 : π / 2 < θ ∧ θ < π) : Real.cos θ - Real.sin θ = - Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_min_sin_eq_neg_sqrt_seven_half_l361_36133


namespace NUMINAMATH_GPT_find_divisor_l361_36124

theorem find_divisor (d q r : ℕ) :
  (919 = d * q + r) → (q = 17) → (r = 11) → d = 53 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l361_36124


namespace NUMINAMATH_GPT_evaluate_expression_l361_36189

theorem evaluate_expression (x : ℤ) (h : x = 3) : x^6 - 6 * x^2 + 7 * x = 696 :=
by
  have hx : x = 3 := h
  sorry

end NUMINAMATH_GPT_evaluate_expression_l361_36189


namespace NUMINAMATH_GPT_find_B_l361_36106

variable (A B : Set ℤ)
variable (U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 6})

theorem find_B (hU : U = {x | 0 ≤ x ∧ x ≤ 6})
               (hA_complement_B : A ∩ (U \ B) = {1, 3, 5}) :
  B = {0, 2, 4, 6} :=
sorry

end NUMINAMATH_GPT_find_B_l361_36106


namespace NUMINAMATH_GPT_sara_initial_savings_l361_36145

-- Given conditions as definitions
def save_rate_sara : ℕ := 10
def save_rate_jim : ℕ := 15
def weeks : ℕ := 820

-- Prove that the initial savings of Sara is 4100 dollars given the conditions
theorem sara_initial_savings : 
  ∃ S : ℕ, S + save_rate_sara * weeks = save_rate_jim * weeks → S = 4100 := 
sorry

end NUMINAMATH_GPT_sara_initial_savings_l361_36145


namespace NUMINAMATH_GPT_jill_peaches_l361_36192

variable (S J : ℕ)

theorem jill_peaches (h1 : S = 19) (h2 : S = J + 13) : J = 6 :=
by
  sorry

end NUMINAMATH_GPT_jill_peaches_l361_36192


namespace NUMINAMATH_GPT_employee_pay_l361_36134

variable (X Y Z : ℝ)

-- Conditions
def X_pay (Y : ℝ) := 1.2 * Y
def Z_pay (X : ℝ) := 0.75 * X

-- Proof statement
theorem employee_pay (h1 : X = X_pay Y) (h2 : Z = Z_pay X) (total_pay : X + Y + Z = 1540) : 
  X + Y + Z = 1540 :=
by
  sorry

end NUMINAMATH_GPT_employee_pay_l361_36134


namespace NUMINAMATH_GPT_problem_l361_36152

theorem problem (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 - x) = x^2 + 1) : f (-1) = 5 := 
  sorry

end NUMINAMATH_GPT_problem_l361_36152


namespace NUMINAMATH_GPT_tractor_brigades_l361_36187
noncomputable def brigade_plowing : Prop :=
∃ x y : ℝ,
  x * y = 240 ∧
  (x + 3) * (y + 2) = 324 ∧
  x > 20 ∧
  (x + 3) > 20 ∧
  x = 24 ∧
  (x + 3) = 27

theorem tractor_brigades:
  brigade_plowing :=
sorry

end NUMINAMATH_GPT_tractor_brigades_l361_36187


namespace NUMINAMATH_GPT_ratio_of_speeds_l361_36122

variable (x y n : ℝ)

-- Conditions
def condition1 : Prop := 3 * (x - y) = n
def condition2 : Prop := 2 * (x + y) = n

-- Problem Statement
theorem ratio_of_speeds (h1 : condition1 x y n) (h2 : condition2 x y n) : x = 5 * y :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l361_36122


namespace NUMINAMATH_GPT_inequality_proof_l361_36130

theorem inequality_proof (a b c d : ℝ) (hnonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (hsum : a + b + c + d = 1) :
  abcd + bcda + cdab + dabc ≤ 1/27 + (176/27) * abcd :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l361_36130


namespace NUMINAMATH_GPT_plant_supplier_earnings_l361_36111

theorem plant_supplier_earnings :
  let orchids_price := 50
  let orchids_sold := 20
  let money_plant_price := 25
  let money_plants_sold := 15
  let worker_wage := 40
  let workers := 2
  let pot_cost := 150
  let total_earnings := (orchids_price * orchids_sold) + (money_plant_price * money_plants_sold)
  let total_expense := (worker_wage * workers) + pot_cost
  total_earnings - total_expense = 1145 :=
by
  sorry

end NUMINAMATH_GPT_plant_supplier_earnings_l361_36111


namespace NUMINAMATH_GPT_factorize_polynomial_l361_36183

theorem factorize_polynomial (x y : ℝ) : 
  (2 * x^2 * y - 4 * x * y^2 + 2 * y^3) = 2 * y * (x - y)^2 :=
sorry

end NUMINAMATH_GPT_factorize_polynomial_l361_36183


namespace NUMINAMATH_GPT_find_value_of_z_l361_36103

open Complex

-- Define the given complex number z and imaginary unit i
def z : ℂ := sorry
def i : ℂ := Complex.I

-- Given condition
axiom condition : z / (1 - i) = i ^ 2019

-- Proof that z equals -1 - i
theorem find_value_of_z : z = -1 - i :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_z_l361_36103


namespace NUMINAMATH_GPT_chef_used_apples_l361_36159

theorem chef_used_apples (initial_apples remaining_apples used_apples : ℕ) 
  (h1 : initial_apples = 40) 
  (h2 : remaining_apples = 39) 
  (h3 : used_apples = initial_apples - remaining_apples) : 
  used_apples = 1 := 
  sorry

end NUMINAMATH_GPT_chef_used_apples_l361_36159


namespace NUMINAMATH_GPT_captain_age_l361_36179

noncomputable def whole_team_age : ℕ := 253
noncomputable def remaining_players_age : ℕ := 198
noncomputable def captain_and_wicket_keeper_age : ℕ := whole_team_age - remaining_players_age
noncomputable def wicket_keeper_age (C : ℕ) : ℕ := C + 3

theorem captain_age (C : ℕ) (whole_team : whole_team_age = 11 * 23) (remaining_players : remaining_players_age = 9 * 22) 
    (sum_ages : captain_and_wicket_keeper_age = 55) (wicket_keeper : wicket_keeper_age C = C + 3) : C = 26 := 
  sorry

end NUMINAMATH_GPT_captain_age_l361_36179


namespace NUMINAMATH_GPT_bicycle_final_price_l361_36143

theorem bicycle_final_price : 
  let original_price := 200 
  let weekend_discount := 0.40 * original_price 
  let price_after_weekend_discount := original_price - weekend_discount 
  let wednesday_discount := 0.20 * price_after_weekend_discount 
  let final_price := price_after_weekend_discount - wednesday_discount 
  final_price = 96 := 
by 
  sorry

end NUMINAMATH_GPT_bicycle_final_price_l361_36143


namespace NUMINAMATH_GPT_tangent_line_ln_l361_36191

theorem tangent_line_ln (x y : ℝ) (h_curve : y = Real.log (x + 1)) (h_point : (1, Real.log 2) = (1, y)) :
  x - 2 * y - 1 + 2 * Real.log 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_ln_l361_36191


namespace NUMINAMATH_GPT_find_stream_speed_l361_36104

-- Define the conditions
def boat_speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def upstream_time : ℝ := 1.5
def speed_of_stream (v : ℝ) : Prop :=
  let downstream_speed := boat_speed_in_still_water + v
  let upstream_speed := boat_speed_in_still_water - v
  (downstream_speed * downstream_time) = (upstream_speed * upstream_time)

-- Define the theorem to prove
theorem find_stream_speed : ∃ v, speed_of_stream v ∧ v = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_stream_speed_l361_36104


namespace NUMINAMATH_GPT_min_minutes_for_B_cheaper_l361_36155

-- Define the relevant constants and costs associated with each plan
def cost_A (x : ℕ) : ℕ := 12 * x
def cost_B (x : ℕ) : ℕ := 2500 + 6 * x
def cost_C (x : ℕ) : ℕ := 9 * x

-- Lean statement for the proof problem
theorem min_minutes_for_B_cheaper : ∃ (x : ℕ), x = 834 ∧ cost_B x < cost_A x ∧ cost_B x < cost_C x := 
sorry

end NUMINAMATH_GPT_min_minutes_for_B_cheaper_l361_36155


namespace NUMINAMATH_GPT_LineChart_characteristics_and_applications_l361_36126

-- Definitions related to question and conditions
def LineChart : Type := sorry
def represents_amount (lc : LineChart) : Prop := sorry
def reflects_increase_or_decrease (lc : LineChart) : Prop := sorry

-- Theorem related to the correct answer
theorem LineChart_characteristics_and_applications (lc : LineChart) :
  represents_amount lc ∧ reflects_increase_or_decrease lc :=
sorry

end NUMINAMATH_GPT_LineChart_characteristics_and_applications_l361_36126


namespace NUMINAMATH_GPT_production_cost_per_performance_l361_36131

theorem production_cost_per_performance
  (overhead : ℕ)
  (revenue_per_performance : ℕ)
  (num_performances : ℕ)
  (production_cost : ℕ)
  (break_even : num_performances * revenue_per_performance = overhead + num_performances * production_cost) :
  production_cost = 7000 :=
by
  have : num_performances = 9 := by sorry
  have : revenue_per_performance = 16000 := by sorry
  have : overhead = 81000 := by sorry
  exact sorry

end NUMINAMATH_GPT_production_cost_per_performance_l361_36131


namespace NUMINAMATH_GPT_vampire_count_after_two_nights_l361_36144

noncomputable def vampire_growth : Nat :=
  let first_night_new_vampires := 3 * 7
  let total_vampires_after_first_night := first_night_new_vampires + 3
  let second_night_new_vampires := total_vampires_after_first_night * (7 + 1)
  second_night_new_vampires + total_vampires_after_first_night

theorem vampire_count_after_two_nights : vampire_growth = 216 :=
by
  -- Skipping the detailed proof steps for now
  sorry

end NUMINAMATH_GPT_vampire_count_after_two_nights_l361_36144


namespace NUMINAMATH_GPT_find_sequence_term_l361_36197

noncomputable def sequence_sum (n : ℕ) : ℚ :=
  (2 / 3) * n^2 - (1 / 3) * n

def sequence_term (n : ℕ) : ℚ :=
  if n = 1 then (1 / 3) else (4 / 3) * n - 1

theorem find_sequence_term (n : ℕ) : sequence_term n = (sequence_sum n - sequence_sum (n - 1)) :=
by
  unfold sequence_sum
  unfold sequence_term
  sorry

end NUMINAMATH_GPT_find_sequence_term_l361_36197


namespace NUMINAMATH_GPT_sum_of_digits_of_m_eq_nine_l361_36196

theorem sum_of_digits_of_m_eq_nine
  (m : ℕ)
  (h1 : m * 3 / 2 - 72 = m) :
  1 + (m / 10 % 10) + (m % 10) = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_m_eq_nine_l361_36196


namespace NUMINAMATH_GPT_num_cows_correct_l361_36114

-- Definitions from the problem's conditions
def total_animals : ℕ := 500
def percentage_chickens : ℤ := 10
def remaining_animals := total_animals - (percentage_chickens * total_animals / 100)
def goats (cows: ℕ) : ℕ := 2 * cows

-- Statement to prove
theorem num_cows_correct : ∃ cows, remaining_animals = cows + goats cows ∧ 3 * cows = 450 :=
by
  sorry

end NUMINAMATH_GPT_num_cows_correct_l361_36114


namespace NUMINAMATH_GPT_subtraction_example_l361_36175

theorem subtraction_example :
  145.23 - 0.07 = 145.16 :=
sorry

end NUMINAMATH_GPT_subtraction_example_l361_36175


namespace NUMINAMATH_GPT_Kylie_uses_3_towels_in_one_month_l361_36113

-- Define the necessary variables and conditions
variable (daughters_towels : Nat) (husband_towels : Nat) (loads : Nat) (towels_per_load : Nat)
variable (K : Nat) -- number of bath towels Kylie uses

-- Given conditions
axiom h1 : daughters_towels = 6
axiom h2 : husband_towels = 3
axiom h3 : loads = 3
axiom h4 : towels_per_load = 4
axiom h5 : (K + daughters_towels + husband_towels) = (loads * towels_per_load)

-- Prove that K = 3
theorem Kylie_uses_3_towels_in_one_month : K = 3 :=
by
  sorry

end NUMINAMATH_GPT_Kylie_uses_3_towels_in_one_month_l361_36113


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l361_36177

theorem necessary_but_not_sufficient_condition (a b c d : ℝ) : 
  (a + b < c + d) → (a < c ∨ b < d) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l361_36177


namespace NUMINAMATH_GPT_B_spends_85_percent_salary_l361_36115

theorem B_spends_85_percent_salary (A_s B_s : ℝ) (A_savings : ℝ) :
  A_s + B_s = 2000 →
  A_s = 1500 →
  A_savings = 0.05 * A_s →
  (B_s - (B_s * (1 - 0.05))) = A_savings →
  (1 - 0.85) * B_s = 0.15 * B_s := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_B_spends_85_percent_salary_l361_36115


namespace NUMINAMATH_GPT_solve_equation_l361_36127

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by
sorry

end NUMINAMATH_GPT_solve_equation_l361_36127


namespace NUMINAMATH_GPT_initial_men_in_camp_l361_36117

theorem initial_men_in_camp (days_initial men_initial : ℕ) (days_plus_thirty men_plus_thirty : ℕ)
(h1 : days_initial = 20)
(h2 : men_plus_thirty = men_initial + 30)
(h3 : days_plus_thirty = 5)
(h4 : (men_initial * days_initial) = (men_plus_thirty * days_plus_thirty)) :
  men_initial = 10 :=
by sorry

end NUMINAMATH_GPT_initial_men_in_camp_l361_36117


namespace NUMINAMATH_GPT_train_platform_length_l361_36166

theorem train_platform_length 
  (speed_train_kmph : ℕ) 
  (time_cross_platform : ℕ) 
  (time_cross_man : ℕ) 
  (L_platform : ℕ) :
  speed_train_kmph = 72 ∧ 
  time_cross_platform = 34 ∧ 
  time_cross_man = 18 ∧ 
  L_platform = 320 :=
by
  sorry

end NUMINAMATH_GPT_train_platform_length_l361_36166


namespace NUMINAMATH_GPT_convex_polyhedron_P_T_V_sum_eq_34_l361_36188

theorem convex_polyhedron_P_T_V_sum_eq_34
  (F : ℕ) (V : ℕ) (E : ℕ) (T : ℕ) (P : ℕ) 
  (hF : F = 32)
  (hT1 : 3 * T + 5 * P = 960)
  (hT2 : 2 * E = V * (T + P))
  (hT3 : T + P - 2 = 60)
  (hT4 : F + V - E = 2) :
  P + T + V = 34 := by
  sorry

end NUMINAMATH_GPT_convex_polyhedron_P_T_V_sum_eq_34_l361_36188


namespace NUMINAMATH_GPT_distinct_solutions_for_quadratic_l361_36107

theorem distinct_solutions_for_quadratic (n : ℕ) : ∃ (xs : Finset ℤ), xs.card = n ∧ ∀ x ∈ xs, ∃ y : ℤ, x^2 + 2^(n + 1) = y^2 :=
by sorry

end NUMINAMATH_GPT_distinct_solutions_for_quadratic_l361_36107


namespace NUMINAMATH_GPT_sum_of_integers_l361_36184

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 10) (h2 : x * y = 80) (hx_pos : 0 < x) (hy_pos : 0 < y) : x + y = 20 := by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l361_36184


namespace NUMINAMATH_GPT_geometric_seq_ratio_l361_36170

theorem geometric_seq_ratio : 
  ∀ (a : ℕ → ℝ) (q : ℝ), 
    (∀ n, a (n+1) = a n * q) → 
    q > 1 → 
    a 1 + a 6 = 8 → 
    a 3 * a 4 = 12 → 
    a 2018 / a 2013 = 3 :=
by
  intros a q h_geom h_q_pos h_sum_eq h_product_eq
  sorry

end NUMINAMATH_GPT_geometric_seq_ratio_l361_36170
