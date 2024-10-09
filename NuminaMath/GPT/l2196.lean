import Mathlib

namespace square_side_is_8_l2196_219620

-- Definitions based on problem conditions
def rectangle_width : ℝ := 4
def rectangle_length : ℝ := 16
def rectangle_area : ℝ := rectangle_width * rectangle_length

def square_side_length (s : ℝ) : Prop := s^2 = rectangle_area

-- The theorem we need to prove
theorem square_side_is_8 (s : ℝ) : square_side_length s → s = 8 := by
  -- Proof to be filled in
  sorry

end square_side_is_8_l2196_219620


namespace correct_calculation_l2196_219653

theorem correct_calculation : -Real.sqrt ((-5)^2) = -5 := 
by 
  sorry

end correct_calculation_l2196_219653


namespace pentagon_ABEDF_area_l2196_219637

theorem pentagon_ABEDF_area (BD_diagonal : ∀ (ABCD : Nat) (BD : Nat),
                            ABCD = BD^2 / 2 → BD = 20) 
                            (BDFE_is_rectangle : ∀ (BDFE : Nat), BDFE = 2 * BD) 
                            : ∃ (area : Nat), area = 300 :=
by
  -- Placeholder for the actual proof
  sorry

end pentagon_ABEDF_area_l2196_219637


namespace union_M_N_l2196_219688

-- Definitions for the sets M and N
def M : Set ℝ := { x | x^2 = x }
def N : Set ℝ := { x | Real.log x / Real.log 2 ≤ 0 }

-- Proof problem statement
theorem union_M_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end union_M_N_l2196_219688


namespace not_prime_3999991_l2196_219621

   theorem not_prime_3999991 : ¬ Nat.Prime 3999991 :=
   by
     -- Provide the factorization proof
     sorry
   
end not_prime_3999991_l2196_219621


namespace alia_markers_count_l2196_219647

theorem alia_markers_count :
  ∀ (Alia Austin Steve Bella : ℕ),
  (Alia = 2 * Austin) →
  (Austin = (1 / 3) * Steve) →
  (Steve = 60) →
  (Bella = (3 / 2) * Alia) →
  Alia = 40 :=
by
  intros Alia Austin Steve Bella H1 H2 H3 H4
  sorry

end alia_markers_count_l2196_219647


namespace tagged_fish_in_second_catch_l2196_219669

theorem tagged_fish_in_second_catch :
  let N := 500
  let total_tagged := 50
  let total_caught := 50
  (total_tagged / N) * total_caught = 5 :=
by
  let N := 500
  let total_tagged := 50
  let total_caught := 50
  show (total_tagged / N) * total_caught = 5
  sorry

end tagged_fish_in_second_catch_l2196_219669


namespace rectangle_perimeter_l2196_219668

theorem rectangle_perimeter :
  ∃ (a b : ℕ), (a ≠ b) ∧ (a * b = 2 * (a + b) - 4) ∧ (2 * (a + b) = 26) :=
by {
  sorry
}

end rectangle_perimeter_l2196_219668


namespace find_C_and_D_l2196_219635

theorem find_C_and_D (C D : ℚ) (h1 : 5 * C + 3 * D - 4 = 47) (h2 : C = D + 2) : 
  C = 57 / 8 ∧ D = 41 / 8 :=
by 
  sorry

end find_C_and_D_l2196_219635


namespace hyperbola_equation_standard_form_l2196_219679

noncomputable def point_on_hyperbola_asymptote (A : ℝ × ℝ) (C : ℝ) : Prop :=
  let x := A.1
  let y := A.2
  (4 * y^2 - x^2 = C) ∧
  (y = (1/2) * x ∨ y = -(1/2) * x)

theorem hyperbola_equation_standard_form
  (A : ℝ × ℝ)
  (hA : A = (2 * Real.sqrt 2, 2))
  (asymptote1 asymptote2 : ℝ → ℝ)
  (hasymptote1 : ∀ x, asymptote1 x = (1/2) * x)
  (hasymptote2 : ∀ x, asymptote2 x = -(1/2) * x) :
  (∃ C : ℝ, point_on_hyperbola_asymptote A C) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (4 * (A.2)^2 - (A.1)^2 = 8) ∧ 
    (∀ x y : ℝ, (4 * y^2 - x^2 = 8) ↔ ((y^2) / a - (x^2) / b = 1))) :=
by
  sorry

end hyperbola_equation_standard_form_l2196_219679


namespace sandy_initial_books_l2196_219682

-- Define the initial conditions as given.
def books_tim : ℕ := 33
def books_lost : ℕ := 24
def books_after_loss : ℕ := 19

-- Define the equation for the total books before Benny's loss and solve for Sandy's books.
def books_total_before_loss : ℕ := books_after_loss + books_lost
def books_sandy_initial : ℕ := books_total_before_loss - books_tim

-- Assert the proof statement:
def proof_sandy_books : Prop :=
  books_sandy_initial = 10

theorem sandy_initial_books : proof_sandy_books := by
  -- Placeholder for the actual proof.
  sorry

end sandy_initial_books_l2196_219682


namespace expressions_equal_iff_conditions_l2196_219603

theorem expressions_equal_iff_conditions (a b c : ℝ) :
  (2 * a + 3 * b * c = (a + 2 * b) * (2 * a + 3 * c)) ↔ (a = 0 ∨ a + 2 * b + 1.5 * c = 0) :=
by
  sorry

end expressions_equal_iff_conditions_l2196_219603


namespace work_completion_l2196_219602

theorem work_completion (a b : ℝ) 
  (h1 : a + b = 6) 
  (h2 : a = 10) : 
  a + b = 6 :=
by sorry

end work_completion_l2196_219602


namespace crow_eats_nuts_l2196_219618

theorem crow_eats_nuts (time_fifth_nuts : ℕ) (time_quarter_nuts : ℕ) (h : time_fifth_nuts = 8) :
  time_quarter_nuts = 10 :=
sorry

end crow_eats_nuts_l2196_219618


namespace find_three_digit_number_l2196_219671

theorem find_three_digit_number (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9) (h₂ : 0 ≤ b ∧ b ≤ 9) (h₃ : 0 ≤ c ∧ c ≤ 9)
    (h₄ : (10 * a + b) / 99 + (100 * a + 10 * b + c) / 999 = 33 / 37) :
    100 * a + 10 * b + c = 447 :=
sorry

end find_three_digit_number_l2196_219671


namespace cubic_inequality_l2196_219643

theorem cubic_inequality (a b : ℝ) : (a > b) ↔ (a^3 > b^3) := sorry

end cubic_inequality_l2196_219643


namespace simplify_polynomial_l2196_219660

variable {R : Type*} [CommRing R]

theorem simplify_polynomial (x : R) :
  (12 * x ^ 10 + 9 * x ^ 9 + 5 * x ^ 8) + (2 * x ^ 12 + x ^ 10 + 2 * x ^ 9 + 3 * x ^ 8 + 4 * x ^ 4 + 6 * x ^ 2 + 9) =
  2 * x ^ 12 + 13 * x ^ 10 + 11 * x ^ 9 + 8 * x ^ 8 + 4 * x ^ 4 + 6 * x ^ 2 + 9 :=
  sorry

end simplify_polynomial_l2196_219660


namespace sum_sequences_l2196_219615

theorem sum_sequences : 
  (1 + 12 + 23 + 34 + 45) + (10 + 20 + 30 + 40 + 50) = 265 := by
  sorry

end sum_sequences_l2196_219615


namespace remainder_seven_power_twenty_seven_l2196_219693

theorem remainder_seven_power_twenty_seven :
  (7^27) % 1000 = 543 := 
sorry

end remainder_seven_power_twenty_seven_l2196_219693


namespace pascal_triangle_41st_number_42nd_row_l2196_219681

open Nat

theorem pascal_triangle_41st_number_42nd_row :
  Nat.choose 42 40 = 861 := by
  sorry

end pascal_triangle_41st_number_42nd_row_l2196_219681


namespace volume_of_tetrahedron_l2196_219630

theorem volume_of_tetrahedron 
(angle_ABC_BCD : Real := 45 * Real.pi / 180)
(area_ABC : Real := 150)
(area_BCD : Real := 90)
(length_BC : Real := 10) :
  let h := 2 * area_BCD / length_BC
  let height_perpendicular := h * Real.sin angle_ABC_BCD
  let volume := (1 / 3 : Real) * area_ABC * height_perpendicular
  volume = 450 * Real.sqrt 2 :=
by
  sorry

end volume_of_tetrahedron_l2196_219630


namespace two_absent_one_present_probability_l2196_219601

-- Define the probabilities
def probability_absent_normal : ℚ := 1 / 15

-- Given that the absence rate on Monday increases by 10%
def monday_increase_factor : ℚ := 1.1

-- Calculate the probability of being absent on Monday
def probability_absent_monday : ℚ := probability_absent_normal * monday_increase_factor

-- Calculate the probability of being present on Monday
def probability_present_monday : ℚ := 1 - probability_absent_monday

-- Define the probability that exactly two students are absent and one present
def probability_two_absent_one_present : ℚ :=
  3 * (probability_absent_monday ^ 2) * probability_present_monday

-- Convert the probability to a percentage and round to the nearest tenth
def probability_as_percent : ℚ := round (probability_two_absent_one_present * 100 * 10) / 10

theorem two_absent_one_present_probability : probability_as_percent = 1.5 := by sorry

end two_absent_one_present_probability_l2196_219601


namespace prove_proposition_l2196_219694

-- Define the propositions p and q
def p : Prop := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
def q : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2

-- Define the main theorem to prove
theorem prove_proposition : (¬ p) ∨ q :=
by { sorry }

end prove_proposition_l2196_219694


namespace find_a_l2196_219652

theorem find_a (a b c d : ℤ) 
  (h1 : d + 0 = 2)
  (h2 : c + 2 = 2)
  (h3 : b + 0 = 4)
  (h4 : a + 4 = 0) : 
  a = -4 := 
sorry

end find_a_l2196_219652


namespace contrapositive_inequality_l2196_219633

theorem contrapositive_inequality (a b : ℝ) :
  (a > b → a - 5 > b - 5) ↔ (a - 5 ≤ b - 5 → a ≤ b) := by
sorry

end contrapositive_inequality_l2196_219633


namespace tire_usage_is_25714_l2196_219644

-- Definitions based on conditions
def car_has_six_tires : Prop := (4 + 2 = 6)
def used_equally_over_miles (total_miles : ℕ) (number_of_tires : ℕ) : Prop := 
  (total_miles * 4) / number_of_tires = 25714

-- Theorem statement based on proof
theorem tire_usage_is_25714 (miles_driven : ℕ) (num_tires : ℕ) 
  (h1 : car_has_six_tires) 
  (h2 : miles_driven = 45000)
  (h3 : num_tires = 7) :
  used_equally_over_miles miles_driven num_tires :=
by
  sorry

end tire_usage_is_25714_l2196_219644


namespace only_book_A_l2196_219665

variable (numA numB numBoth numOnlyB x : ℕ)
variable (h1 : numA = 2 * numB)
variable (h2 : numBoth = 500)
variable (h3 : numBoth = 2 * numOnlyB)
variable (h4 : numB = numOnlyB + numBoth)
variable (h5 : x = numA - numBoth)

theorem only_book_A : 
  x = 1000 := 
by
  sorry

end only_book_A_l2196_219665


namespace max_mx_plus_ny_l2196_219673

theorem max_mx_plus_ny 
  (m n x y : ℝ) 
  (h1 : m^2 + n^2 = 6) 
  (h2 : x^2 + y^2 = 24) : 
  mx + ny ≤ 12 :=
sorry

end max_mx_plus_ny_l2196_219673


namespace simplify_expr_l2196_219677

theorem simplify_expr (a b x : ℝ) (h₁ : x = a^3 / b^3) (h₂ : a ≠ b) (h₃ : b ≠ 0) : 
  (a^3 + b^3) / (a^3 - b^3) = (x + 1) / (x - 1) := 
by 
  sorry

end simplify_expr_l2196_219677


namespace coffee_shop_sold_lattes_l2196_219661

theorem coffee_shop_sold_lattes (T L : ℕ) (h1 : T = 6) (h2 : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_sold_lattes_l2196_219661


namespace greater_savings_on_hat_l2196_219606

theorem greater_savings_on_hat (savings_shoes spent_shoes savings_hat sale_price_hat : ℝ) 
  (h1 : savings_shoes = 3.75)
  (h2 : spent_shoes = 42.25)
  (h3 : savings_hat = 1.80)
  (h4 : sale_price_hat = 18.20) :
  ((savings_hat / (sale_price_hat + savings_hat)) * 100) > ((savings_shoes / (spent_shoes + savings_shoes)) * 100) :=
by
  sorry

end greater_savings_on_hat_l2196_219606


namespace fill_grid_power_of_two_l2196_219631

theorem fill_grid_power_of_two (n : ℕ) (h : ∃ m : ℕ, n = 2^m) :
  ∃ f : ℕ → ℕ → ℕ, 
    (∀ i j : ℕ, i < n → j < n → 1 ≤ f i j ∧ f i j ≤ 2 * n - 1) ∧
    (∀ k, 1 ≤ k ∧ k ≤ n → (∀ i, i < n → ∀ j, j < n → i ≠ j → f i k ≠ f j k))
:= by
  sorry

end fill_grid_power_of_two_l2196_219631


namespace probability_sunglasses_to_hat_l2196_219692

variable (S H : Finset ℕ) -- S: set of people wearing sunglasses, H: set of people wearing hats
variable (num_S : Nat) (num_H : Nat) (num_SH : Nat)
variable (prob_hat_to_sunglasses : ℚ)

-- Conditions
def condition1 : num_S = 80 := sorry
def condition2 : num_H = 50 := sorry
def condition3 : prob_hat_to_sunglasses = 3 / 5 := sorry
def condition4 : num_SH = (3/5) * 50 := sorry

-- Question: Prove that the probability a person wearing sunglasses is also wearing a hat
theorem probability_sunglasses_to_hat :
  (num_SH : ℚ) / num_S = 3 / 8 :=
sorry

end probability_sunglasses_to_hat_l2196_219692


namespace angle_C_value_sides_a_b_l2196_219698

variables (A B C : ℝ) (a b c : ℝ)

-- First part: Proving the value of angle C
theorem angle_C_value
  (h1 : 2*Real.cos (A/2)^2 + (Real.cos B - Real.sqrt 3 * Real.sin B) * Real.cos C = 1)
  : C = Real.pi / 3 :=
sorry

-- Second part: Proving the values of a and b given c and the area
theorem sides_a_b
  (c : ℝ)
  (h2 : c = 2)
  (h3 : C = Real.pi / 3)
  (area : ℝ)
  (h4 : area = Real.sqrt 3)
  (h5 : 1/2 * a * b * Real.sin C = Real.sqrt 3)
  : a = 2 ∧ b = 2 :=
sorry

end angle_C_value_sides_a_b_l2196_219698


namespace find_ABC_l2196_219656

noncomputable def g (x : ℝ) (A B C : ℝ) : ℝ := 
  x^2 / (A * x^2 + B * x + C)

theorem find_ABC : 
  (∀ x : ℝ, x > 5 → g x 2 (-2) (-24) > 0.5) ∧
  (A = 2) ∧
  (B = -2) ∧
  (C = -24) ∧
  (∀ x, A * x^2 + B * x + C = A * (x + 3) * (x - 4)) → 
  A + B + C = -24 := 
by
  sorry

end find_ABC_l2196_219656


namespace find_a_from_polynomial_factor_l2196_219648

theorem find_a_from_polynomial_factor (a b : ℤ)
  (h: ∀ x : ℝ, x*x - x - 1 = 0 → a*x^5 + b*x^4 + 1 = 0) : a = 3 :=
sorry

end find_a_from_polynomial_factor_l2196_219648


namespace handshakes_meeting_l2196_219619

theorem handshakes_meeting (x : ℕ) (h : x * (x - 1) / 2 = 66) : x = 12 := 
by 
  sorry

end handshakes_meeting_l2196_219619


namespace dan_present_age_l2196_219622

-- Let x be Dan's present age
variable (x : ℤ)

-- Condition: Dan's age after 18 years will be 8 times his age 3 years ago
def condition (x : ℤ) : Prop :=
  x + 18 = 8 * (x - 3)

-- The goal is to prove that Dan's present age is 6
theorem dan_present_age (x : ℤ) (h : condition x) : x = 6 :=
by
  sorry

end dan_present_age_l2196_219622


namespace smartphone_price_l2196_219612

/-
Question: What is the sticker price of the smartphone, given the following conditions?
Conditions:
1: Store A offers a 20% discount on the sticker price, followed by a $120 rebate. Prices include an 8% sales tax applied after all discounts and fees.
2: Store B offers a 30% discount on the sticker price but adds a $50 handling fee. Prices include an 8% sales tax applied after all discounts and fees.
3: Natalie saves $27 by purchasing the smartphone at store A instead of store B.

Proof Problem:
Prove that given the above conditions, the sticker price of the smartphone is $1450.
-/

theorem smartphone_price (p : ℝ) :
  (1.08 * (0.7 * p + 50) - 1.08 * (0.8 * p - 120)) = 27 ->
  p = 1450 :=
by
  sorry

end smartphone_price_l2196_219612


namespace original_price_of_sarees_l2196_219687

theorem original_price_of_sarees (P : ℝ) (h : 0.92 * 0.90 * P = 331.2) : P = 400 :=
by
  sorry

end original_price_of_sarees_l2196_219687


namespace inner_square_area_l2196_219670

theorem inner_square_area (side_ABCD : ℝ) (dist_BI : ℝ) (area_IJKL : ℝ) :
  side_ABCD = Real.sqrt 72 →
  dist_BI = 2 →
  area_IJKL = 39 :=
by
  sorry

end inner_square_area_l2196_219670


namespace sum_invested_l2196_219650

theorem sum_invested (P R: ℝ) (h1: SI₁ = P * R * 20 / 100) (h2: SI₂ = P * (R + 10) * 20 / 100) (h3: SI₂ = SI₁ + 3000) : P = 1500 :=
by
  sorry

end sum_invested_l2196_219650


namespace smallest_units_C_union_D_l2196_219604

-- Definitions for the sets C and D and their sizes
def C_units : ℝ := 25.5
def D_units : ℝ := 18.0

-- Definition stating the inclusion-exclusion principle for sets C and D
def C_union_D (C_units D_units C_intersection_units : ℝ) : ℝ :=
  C_units + D_units - C_intersection_units

-- Statement to prove the minimum units in C union D
theorem smallest_units_C_union_D : ∃ h, h ≤ C_union_D C_units D_units D_units ∧ h = 25.5 := by
  sorry

end smallest_units_C_union_D_l2196_219604


namespace triangle_area_l2196_219684

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area : 
  ∀ (A B C : (ℝ × ℝ)),
  (A = (3, 3)) →
  (B = (4.5, 7.5)) →
  (C = (7.5, 4.5)) →
  area_of_triangle A B C = 8.625 :=
by
  intros A B C hA hB hC
  rw [hA, hB, hC]
  unfold area_of_triangle
  norm_num
  sorry

end triangle_area_l2196_219684


namespace pattern_E_cannot_be_formed_l2196_219610

-- Define the basic properties of the tile and the patterns
inductive Tile
| rhombus (diag_coloring : Bool) -- representing black-and-white diagonals

inductive Pattern
| optionA
| optionB
| optionC
| optionD
| optionE

-- The given tile is a rhombus with a certain coloring scheme
def given_tile : Tile := Tile.rhombus true

-- The statement to prove
theorem pattern_E_cannot_be_formed : 
  ¬ (∃ f : Pattern → Tile, f Pattern.optionE = given_tile) :=
sorry

end pattern_E_cannot_be_formed_l2196_219610


namespace average_of_pqrs_l2196_219607

variable (p q r s : ℝ)

theorem average_of_pqrs
  (h : (5 / 4) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 4 :=
by
  sorry

end average_of_pqrs_l2196_219607


namespace find_n_in_arithmetic_sequence_l2196_219611

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 4 then 7 else
  if n = 5 then 16 - 7 else sorry

-- Define the arithmetic sequence and the given conditions
theorem find_n_in_arithmetic_sequence (n : ℕ) (a : ℕ → ℕ) 
  (h1 : a 4 = 7) 
  (h2 : a 3 + a 6 = 16) 
  (h3 : a n = 31) :
  n = 16 :=
by
  sorry

end find_n_in_arithmetic_sequence_l2196_219611


namespace find_y_when_x_is_8_l2196_219690

theorem find_y_when_x_is_8 : 
  ∃ k, (70 * 5 = k ∧ 8 * 25 = k) := 
by
  -- The proof will be filled in here
  sorry

end find_y_when_x_is_8_l2196_219690


namespace fourth_game_water_correct_fourth_game_sports_drink_l2196_219613

noncomputable def total_bottled_water_cases : ℕ := 10
noncomputable def total_sports_drink_cases : ℕ := 5
noncomputable def bottles_per_case_water : ℕ := 20
noncomputable def bottles_per_case_sports_drink : ℕ := 15
noncomputable def initial_bottled_water : ℕ := total_bottled_water_cases * bottles_per_case_water
noncomputable def initial_sports_drinks : ℕ := total_sports_drink_cases * bottles_per_case_sports_drink

noncomputable def first_game_water : ℕ := 70
noncomputable def first_game_sports_drink : ℕ := 30
noncomputable def second_game_water : ℕ := 40
noncomputable def second_game_sports_drink : ℕ := 20
noncomputable def third_game_water : ℕ := 50
noncomputable def third_game_sports_drink : ℕ := 25

noncomputable def total_consumed_water : ℕ := first_game_water + second_game_water + third_game_water
noncomputable def total_consumed_sports_drink : ℕ := first_game_sports_drink + second_game_sports_drink + third_game_sports_drink

noncomputable def remaining_water_before_fourth_game : ℕ := initial_bottled_water - total_consumed_water
noncomputable def remaining_sports_drink_before_fourth_game : ℕ := initial_sports_drinks - total_consumed_sports_drink

noncomputable def remaining_water_after_fourth_game : ℕ := 20
noncomputable def remaining_sports_drink_after_fourth_game : ℕ := 10

noncomputable def fourth_game_water_consumed : ℕ := remaining_water_before_fourth_game - remaining_water_after_fourth_game

theorem fourth_game_water_correct : fourth_game_water_consumed = 20 :=
by
  unfold fourth_game_water_consumed remaining_water_before_fourth_game
  sorry

theorem fourth_game_sports_drink : false :=
by
  sorry

end fourth_game_water_correct_fourth_game_sports_drink_l2196_219613


namespace consecutive_differences_equal_l2196_219689

-- Define the set and the condition
def S : Set ℕ := {n : ℕ | n > 0}

-- Condition that for any two numbers a and b in S with a > b, at least one of a + b or a - b is also in S
axiom h_condition : ∀ a b : ℕ, a ∈ S → b ∈ S → a > b → (a + b ∈ S ∨ a - b ∈ S)

-- The main theorem that we want to prove
theorem consecutive_differences_equal (a : ℕ) (s : Fin 2003 → ℕ) 
  (hS : ∀ i, s i ∈ S)
  (h_ordered : ∀ i j, i < j → s i < s j) :
  ∃ (d : ℕ), ∀ i, i < 2002 → (s (i + 1)) - (s i) = d :=
sorry

end consecutive_differences_equal_l2196_219689


namespace number_of_terms_l2196_219676

noncomputable def Sn (n : ℕ) : ℝ := sorry

def an_arithmetic_seq (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

theorem number_of_terms {a : ℕ → ℝ}
  (h_arith : an_arithmetic_seq a)
  (cond1 : a 1 + a 2 + a 3 + a 4 = 1)
  (cond2 : a 5 + a 6 + a 7 + a 8 = 2)
  (cond3 : Sn = 15) :
  ∃ n, n = 16 :=
sorry

end number_of_terms_l2196_219676


namespace anne_cleaning_time_l2196_219666

theorem anne_cleaning_time (B A : ℝ) 
  (h₁ : 4 * (B + A) = 1) 
  (h₂ : 3 * (B + 2 * A) = 1) : 
  1 / A = 12 :=
sorry

end anne_cleaning_time_l2196_219666


namespace shark_sightings_l2196_219672

theorem shark_sightings (x : ℕ) 
  (h1 : 26 = 5 + 3 * x) : x = 7 :=
by
  sorry

end shark_sightings_l2196_219672


namespace min_button_presses_l2196_219624

theorem min_button_presses :
  ∃ (a b : ℤ), 9 * a - 20 * b = 13 ∧  a + b = 24 := 
by
  sorry

end min_button_presses_l2196_219624


namespace inverse_proposition_is_false_l2196_219629

theorem inverse_proposition_is_false (a : ℤ) (h : a = 6) : ¬ (|a| = 6 → a = 6) :=
sorry

end inverse_proposition_is_false_l2196_219629


namespace gcd_lcm_ratio_l2196_219697

theorem gcd_lcm_ratio (A B : ℕ) (k : ℕ) (h1 : Nat.lcm A B = 200) (h2 : 2 * k = A) (h3 : 5 * k = B) : Nat.gcd A B = k :=
by
  sorry

end gcd_lcm_ratio_l2196_219697


namespace geometric_sequence_value_of_b_l2196_219632

-- Definitions
def is_geometric_sequence (a b c : ℝ) := 
  ∃ r : ℝ, a * r = b ∧ b * r = c

-- Theorem statement
theorem geometric_sequence_value_of_b (b : ℝ) (h : b > 0) 
  (h_seq : is_geometric_sequence 15 b 1) : b = Real.sqrt 15 :=
by
  sorry

end geometric_sequence_value_of_b_l2196_219632


namespace consecutive_coeff_sum_l2196_219626

theorem consecutive_coeff_sum (P : Polynomial ℕ) (hdeg : P.degree = 699)
  (hP : P.eval 1 ≤ 2022) :
  ∃ k : ℕ, k < 700 ∧ (P.coeff (k + 1) + P.coeff k) = 22 ∨
                    (P.coeff (k + 1) + P.coeff k) = 55 ∨
                    (P.coeff (k + 1) + P.coeff k) = 77 :=
by
  sorry

end consecutive_coeff_sum_l2196_219626


namespace imaginary_unit_squared_in_set_l2196_219685

-- Conditions of the problem
def imaginary_unit (i : ℂ) : Prop := i^2 = -1
def S : Set ℂ := {-1, 0, 1}

-- The statement to prove
theorem imaginary_unit_squared_in_set {i : ℂ} (hi : imaginary_unit i) : i^2 ∈ S := sorry

end imaginary_unit_squared_in_set_l2196_219685


namespace percentage_income_diff_l2196_219663

variable (A B : ℝ)

-- Condition that B's income is 33.33333333333333% greater than A's income
def income_relation (A B : ℝ) : Prop :=
  B = (4 / 3) * A

-- Proof statement to show that A's income is 25% less than B's income
theorem percentage_income_diff : 
  income_relation A B → 
  ((B - A) / B) * 100 = 25 :=
by
  intros h
  rw [income_relation] at h
  sorry

end percentage_income_diff_l2196_219663


namespace fourth_pentagon_has_31_dots_l2196_219658

-- Conditions representing the sequence of pentagons
def first_pentagon_dots : ℕ := 1

def second_pentagon_dots : ℕ := first_pentagon_dots + 5

def nth_layer_dots (n : ℕ) : ℕ := 5 * (n - 1)

def nth_pentagon_dots (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc k => acc + nth_layer_dots (k+1)) first_pentagon_dots

-- Question and proof statement
theorem fourth_pentagon_has_31_dots : nth_pentagon_dots 4 = 31 :=
  sorry

end fourth_pentagon_has_31_dots_l2196_219658


namespace percentage_two_sections_cleared_l2196_219699

noncomputable def total_candidates : ℕ := 1200
def pct_cleared_all_sections : ℝ := 0.05
def pct_cleared_none_sections : ℝ := 0.05
def pct_cleared_one_section : ℝ := 0.25
def pct_cleared_four_sections : ℝ := 0.20
def cleared_three_sections : ℕ := 300

theorem percentage_two_sections_cleared :
  (total_candidates - total_candidates * (pct_cleared_all_sections + pct_cleared_none_sections + pct_cleared_one_section + pct_cleared_four_sections) - cleared_three_sections) / total_candidates * 100 = 20 := by
  sorry

end percentage_two_sections_cleared_l2196_219699


namespace incorrect_counting_of_students_l2196_219623

open Set

theorem incorrect_counting_of_students
  (total_students : ℕ)
  (english_only : ℕ)
  (german_only : ℕ)
  (french_only : ℕ)
  (english_german : ℕ)
  (english_french : ℕ)
  (german_french : ℕ)
  (all_three : ℕ)
  (reported_total : ℕ)
  (h_total_students : total_students = 100)
  (h_english_only : english_only = 30)
  (h_german_only : german_only = 23)
  (h_french_only : french_only = 50)
  (h_english_german : english_german = 10)
  (h_english_french : english_french = 8)
  (h_german_french : german_french = 20)
  (h_all_three : all_three = 5)
  (h_reported_total : reported_total = 100) :
  (english_only + german_only + french_only + english_german +
   english_french + german_french - 2 * all_three) ≠ reported_total :=
by
  sorry

end incorrect_counting_of_students_l2196_219623


namespace arithmetic_sequence_nth_term_639_l2196_219608

theorem arithmetic_sequence_nth_term_639 :
  ∀ (x n : ℕ) (a₁ a₂ a₃ aₙ : ℤ),
  a₁ = 3 * x - 5 →
  a₂ = 7 * x - 17 →
  a₃ = 4 * x + 3 →
  aₙ = a₁ + (n - 1) * (a₂ - a₁) →
  aₙ = 4018 →
  n = 639 :=
by
  intros x n a₁ a₂ a₃ aₙ h₁ h₂ h₃ hₙ hₙ_eq
  sorry

end arithmetic_sequence_nth_term_639_l2196_219608


namespace rectangle_area_integer_length_width_l2196_219646

theorem rectangle_area_integer_length_width (l w : ℕ) (h1 : w = l / 2) (h2 : 2 * l + 2 * w = 200) :
  l * w = 2178 :=
by
  sorry

end rectangle_area_integer_length_width_l2196_219646


namespace proportion_is_equation_l2196_219600

/-- A proportion containing unknowns is an equation -/
theorem proportion_is_equation (P : Prop) (contains_equality_sign: Prop)
  (indicates_equality : Prop)
  (contains_unknowns : Prop) : (contains_equality_sign ∧ indicates_equality ∧ contains_unknowns ↔ True) := by
  sorry

end proportion_is_equation_l2196_219600


namespace raffle_prize_l2196_219641

theorem raffle_prize (P : ℝ) :
  (0.80 * P = 80) → (P = 100) :=
by
  intro h1
  sorry

end raffle_prize_l2196_219641


namespace total_selling_price_of_toys_l2196_219659

/-
  Prove that the total selling price (TSP) for 18 toys,
  given that each toy costs Rs. 1100 and the man gains the cost price of 3 toys, is Rs. 23100.
-/
theorem total_selling_price_of_toys :
  let CP := 1100
  let TCP := 18 * CP
  let G := 3 * CP
  let TSP := TCP + G
  TSP = 23100 :=
by
  let CP := 1100
  let TCP := 18 * CP
  let G := 3 * CP
  let TSP := TCP + G
  sorry

end total_selling_price_of_toys_l2196_219659


namespace minimum_height_l2196_219639

theorem minimum_height (x : ℝ) (h : ℝ) (A : ℝ) :
  (h = x + 4) →
  (A = 6*x^2 + 16*x) →
  (A ≥ 120) →
  (x ≥ 2) →
  h = 6 :=
by
  intros h_def A_def A_geq min_x
  sorry

end minimum_height_l2196_219639


namespace jenny_spent_180_minutes_on_bus_l2196_219674

noncomputable def jennyBusTime : ℕ :=
  let timeAwayFromHome := 9 * 60  -- in minutes
  let classTime := 5 * 45  -- 5 classes each lasting 45 minutes
  let lunchTime := 45  -- in minutes
  let extracurricularTime := 90  -- 1 hour and 30 minutes
  timeAwayFromHome - (classTime + lunchTime + extracurricularTime)

theorem jenny_spent_180_minutes_on_bus : jennyBusTime = 180 :=
  by
  -- We need to prove that the total time Jenny was away from home minus time spent in school activities is 180 minutes.
  sorry  -- Proof to be completed.

end jenny_spent_180_minutes_on_bus_l2196_219674


namespace unattainable_y_l2196_219628

theorem unattainable_y (x : ℝ) (hx : x ≠ -2 / 3) : ¬ (∃ x, y = (x - 3) / (3 * x + 2) ∧ y = 1 / 3) := by
  sorry

end unattainable_y_l2196_219628


namespace difference_between_median_and_mean_is_five_l2196_219617

noncomputable def mean_score : ℝ :=
  0.20 * 60 + 0.20 * 75 + 0.40 * 85 + 0.20 * 95

noncomputable def median_score : ℝ := 85

theorem difference_between_median_and_mean_is_five :
  abs (median_score - mean_score) = 5 :=
by
  unfold mean_score median_score
  -- median_score - mean_score = 85 - 80
  -- thus the absolute value of the difference is 5
  sorry

end difference_between_median_and_mean_is_five_l2196_219617


namespace abs_a_k_le_fractional_l2196_219651

variable (a : ℕ → ℝ) (n : ℕ)

-- Condition 1: a_0 = a_(n+1) = 0
axiom a_0 : a 0 = 0
axiom a_n1 : a (n + 1) = 0

-- Condition 2: |a_{k-1} - 2a_k + a_{k+1}| ≤ 1 for k = 1, 2, ..., n
axiom abs_diff_ineq (k : ℕ) (h : 1 ≤ k ∧ k ≤ n) : 
  |a (k - 1) - 2 * a k + a (k + 1)| ≤ 1

-- Theorem statement
theorem abs_a_k_le_fractional (k : ℕ) (h : 0 ≤ k ∧ k ≤ n + 1) : 
  |a k| ≤ k * (n + 1 - k) / 2 := sorry

end abs_a_k_le_fractional_l2196_219651


namespace find_b_plus_k_l2196_219655

open Real

noncomputable def semi_major_axis (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) : ℝ :=
  dist p f1 + dist p f2

def c_squared (a : ℝ) (b : ℝ) : ℝ :=
  a ^ 2 - b ^ 2

theorem find_b_plus_k :
  ∀ (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) (h k : ℝ) (a b : ℝ),
  f1 = (-2, 0) →
  f2 = (2, 0) →
  p = (6, 0) →
  (∃ a b, semi_major_axis f1 f2 p = 2 * a ∧ c_squared a b = 4) →
  h = 0 →
  k = 0 →
  b = 4 * sqrt 2 →
  b + k = 4 * sqrt 2 :=
by
  intros f1 f2 p h k a b f1_def f2_def p_def maj_axis_def h_def k_def b_def
  rw [b_def, k_def]
  exact add_zero (4 * sqrt 2)

end find_b_plus_k_l2196_219655


namespace days_to_complete_job_l2196_219645

theorem days_to_complete_job (m₁ m₂ d₁ d₂ total_man_days : ℝ)
    (h₁ : m₁ = 30)
    (h₂ : d₁ = 8)
    (h₃ : total_man_days = 240)
    (h₄ : total_man_days = m₁ * d₁)
    (h₅ : m₂ = 40) :
    d₂ = total_man_days / m₂ := by
  sorry

end days_to_complete_job_l2196_219645


namespace pen_shorter_than_pencil_l2196_219664

-- Definitions of the given conditions
def P (R : ℕ) := R + 3
def L : ℕ := 12
def total_length (R : ℕ) := R + P R + L

-- The theorem to be proven
theorem pen_shorter_than_pencil (R : ℕ) (h : total_length R = 29) : L - P R = 2 :=
by
  sorry

end pen_shorter_than_pencil_l2196_219664


namespace exists_hexagon_in_square_l2196_219680

structure Point (α : Type*) :=
(x : α)
(y : α)

def is_in_square (p : Point ℕ) : Prop :=
p.x ≤ 4 ∧ p.y ≤ 4

def area_of_hexagon (vertices : List (Point ℕ)) : ℝ :=
-- placeholder for actual area calculation of a hexagon
sorry

theorem exists_hexagon_in_square : ∃ (p1 p2 : Point ℕ), 
  is_in_square p1 ∧ is_in_square p2 ∧ 
  area_of_hexagon [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 0⟩, ⟨4, 4⟩, p1, p2] = 6 :=
sorry

end exists_hexagon_in_square_l2196_219680


namespace factorization_option_D_l2196_219649

-- Define variables
variables (x y : ℝ)

-- Define the expressions
def left_side_D := -4 * x^2 + 12 * x * y - 9 * y^2
def right_side_D := -(2 * x - 3 * y)^2

-- Theorem statement
theorem factorization_option_D : left_side_D x y = right_side_D x y :=
sorry

end factorization_option_D_l2196_219649


namespace sum_of_positive_factors_of_72_l2196_219642

/-- Define the divisor sum function based on the given formula -/
def divisor_sum (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 3
  | 3 => 4
  | 4 => 7
  | 6 => 12
  | 8 => 15
  | 12 => 28
  | 18 => 39
  | 24 => 60
  | 36 => 91
  | 48 => 124
  | 60 => 168
  | 72 => 195
  | _ => 0 -- This is not generally correct, just handles given problem specifically

theorem sum_of_positive_factors_of_72 :
  divisor_sum 72 = 195 :=
sorry

end sum_of_positive_factors_of_72_l2196_219642


namespace point_B_coordinates_l2196_219695

variable (A : ℝ × ℝ)

def move_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

def move_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

theorem point_B_coordinates : 
  (move_left (move_up (-3, -5) 4) 3) = (-6, -1) :=
by
  sorry

end point_B_coordinates_l2196_219695


namespace seashells_total_l2196_219654

theorem seashells_total (joan_seashells jessica_seashells : ℕ)
  (h_joan : joan_seashells = 6)
  (h_jessica : jessica_seashells = 8) :
  joan_seashells + jessica_seashells = 14 :=
by 
  sorry

end seashells_total_l2196_219654


namespace tree_initial_height_l2196_219686

noncomputable def initial_tree_height (H : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ := 
  H + growth_rate * years

theorem tree_initial_height :
  ∀ (H : ℝ), 
  (∀ (years : ℕ), ∃ h : ℝ, h = initial_tree_height H 0.5 years) →
  initial_tree_height H 0.5 6 = initial_tree_height H 0.5 4 * (7 / 6) →
  H = 4 :=
by
  intro H height_increase condition
  sorry

end tree_initial_height_l2196_219686


namespace range_of_a_l2196_219638

-- Lean statement that represents the proof problem
theorem range_of_a 
  (h1 : ∀ x y : ℝ, x^2 - 2 * x + Real.log (2 * y^2 - y) = 0 → x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0)
  (h2 : ∀ b : ℝ, 2 * b^2 - b > 0) :
  (∀ a : ℝ, x^2 - 2 * x + Real.log (2 * a^2 - a) = 0 → (- (1:ℝ) / 2) < a ∧ a < 0 ∨ (1 / 2) < a ∧ a < 1) :=
sorry

end range_of_a_l2196_219638


namespace num_distinct_integers_formed_l2196_219662

theorem num_distinct_integers_formed (digits : Multiset ℕ) (h : digits = {2, 2, 3, 3, 3}) : 
  Multiset.card (Multiset.powerset digits).attach = 10 := 
by {
  sorry
}

end num_distinct_integers_formed_l2196_219662


namespace john_tour_days_l2196_219616

noncomputable def numberOfDaysInTourProgram (d e : ℕ) : Prop :=
  d * e = 800 ∧ (d + 7) * (e - 5) = 800

theorem john_tour_days :
  ∃ (d e : ℕ), numberOfDaysInTourProgram d e ∧ d = 28 :=
by
  sorry

end john_tour_days_l2196_219616


namespace polynomial_simplified_l2196_219657

def polynomial (x : ℝ) : ℝ := 4 - 6 * x - 8 * x^2 + 12 - 14 * x + 16 * x^2 - 18 + 20 * x + 24 * x^2

theorem polynomial_simplified (x : ℝ) : polynomial x = 32 * x^2 - 2 :=
by
  sorry

end polynomial_simplified_l2196_219657


namespace speed_in_still_water_l2196_219625

-- Definitions for the conditions
def upstream_speed : ℕ := 30
def downstream_speed : ℕ := 60

-- Prove that the speed of the man in still water is 45 kmph
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 45 := by
  sorry

end speed_in_still_water_l2196_219625


namespace evaluate_expression_l2196_219609

theorem evaluate_expression : ((5^2 + 3)^2 - (5^2 - 3)^2)^3 = 27000000 :=
by
  sorry

end evaluate_expression_l2196_219609


namespace cost_of_remaining_ingredients_l2196_219627

theorem cost_of_remaining_ingredients :
  let cocoa_required := 0.4
  let sugar_required := 0.6
  let cake_weight := 450
  let given_cocoa := 259
  let cost_per_lb_cocoa := 3.50
  let cost_per_lb_sugar := 0.80
  let total_cocoa_needed := cake_weight * cocoa_required
  let total_sugar_needed := cake_weight * sugar_required
  let remaining_cocoa := max 0 (total_cocoa_needed - given_cocoa)
  let remaining_sugar := total_sugar_needed
  let total_cost := remaining_cocoa * cost_per_lb_cocoa + remaining_sugar * cost_per_lb_sugar
  total_cost = 216 := by
  sorry

end cost_of_remaining_ingredients_l2196_219627


namespace ideal_point_distance_y_axis_exists_ideal_point_linear_range_of_t_l2196_219667

variable (a b : ℝ)
variable (m x : ℝ)
variable (t : ℝ)
variable (A B C : ℝ)

-- Define ideal points
def is_ideal_point (p : ℝ × ℝ) := p.snd = 2 * p.fst

-- Define the conditions for question 1
def distance_from_y_axis (a : ℝ) := abs a = 2

-- Question 1: Prove that M(2, 4) or M(-2, -4)
theorem ideal_point_distance_y_axis (a b : ℝ) (h1 : is_ideal_point (a, b)) (h2 : distance_from_y_axis a) :
  (a = 2 ∧ b = 4) ∨ (a = -2 ∧ b = -4) := sorry

-- Define the linear function
def linear_func (m x : ℝ) : ℝ := 3 * m * x - 1

-- Question 2: Prove or disprove the existence of ideal points in y = 3mx - 1
theorem exists_ideal_point_linear (m x : ℝ) (hx : is_ideal_point (x, linear_func m x)) :
  (m ≠ 2/3 → ∃ x, linear_func m x = 2 * x) ∧ (m = 2/3 → ¬ ∃ x, linear_func m x = 2 * x) := sorry

-- Question 3 conditions
def quadratic_func (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def quadratic_conditions (a b c : ℝ) : Prop :=
  (quadratic_func a b c 0 = 5 * a + 1) ∧ (quadratic_func a b c (-2) = 5 * a + 1)

-- Question 3: Prove the range of t = a^2 + a + 1 given the quadratic conditions
theorem range_of_t (a b c t : ℝ) (h1 : is_ideal_point (x, quadratic_func a b c x))
  (h2 : quadratic_conditions a b c) (ht : t = a^2 + a + 1) :
    3 / 4 ≤ t ∧ t ≤ 21 / 16 ∧ t ≠ 1 := sorry

end ideal_point_distance_y_axis_exists_ideal_point_linear_range_of_t_l2196_219667


namespace fraction_of_short_students_l2196_219636

theorem fraction_of_short_students 
  (total_students tall_students average_students : ℕ) 
  (htotal : total_students = 400) 
  (htall : tall_students = 90) 
  (haverage : average_students = 150) : 
  (total_students - (tall_students + average_students)) / total_students = 2 / 5 :=
by
  sorry

end fraction_of_short_students_l2196_219636


namespace platform_length_is_350_l2196_219634

variables (L : ℕ)

def train_length := 300
def time_to_cross_pole := 18
def time_to_cross_platform := 39

-- Speed of the train when crossing the pole
def speed_cross_pole : ℚ := train_length / time_to_cross_pole

-- Speed of the train when crossing the platform
def speed_cross_platform (L : ℕ) : ℚ := (train_length + L) / time_to_cross_platform

-- The main goal is to prove that the length of the platform is 350 meters
theorem platform_length_is_350 (L : ℕ) (h : speed_cross_pole = speed_cross_platform L) : L = 350 := sorry

end platform_length_is_350_l2196_219634


namespace tom_monthly_fluid_intake_l2196_219691

-- Define the daily fluid intake amounts
def daily_soda_intake := 5 * 12
def daily_water_intake := 64
def daily_juice_intake := 3 * 8
def daily_sports_drink_intake := 2 * 16
def additional_weekend_smoothie := 32

-- Define the weekdays and weekend days in a month
def weekdays_in_month := 5 * 4
def weekend_days_in_month := 2 * 4

-- Calculate the total daily intake
def daily_intake := daily_soda_intake + daily_water_intake + daily_juice_intake + daily_sports_drink_intake
def weekend_daily_intake := daily_intake + additional_weekend_smoothie

-- Calculate the total monthly intake
def total_fluid_intake_in_month := (daily_intake * weekdays_in_month) + (weekend_daily_intake * weekend_days_in_month)

-- Statement to prove
theorem tom_monthly_fluid_intake : total_fluid_intake_in_month = 5296 :=
by
  unfold total_fluid_intake_in_month
  unfold daily_intake weekend_daily_intake
  unfold weekdays_in_month weekend_days_in_month
  unfold daily_soda_intake daily_water_intake daily_juice_intake daily_sports_drink_intake additional_weekend_smoothie
  sorry

end tom_monthly_fluid_intake_l2196_219691


namespace maximum_profit_l2196_219678

noncomputable def L1 (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
noncomputable def L2 (x : ℝ) : ℝ := 2 * x

theorem maximum_profit :
  (∀ (x1 x2 : ℝ), x1 + x2 = 15 → L1 x1 + L2 x2 ≤ 45.6) := sorry

end maximum_profit_l2196_219678


namespace sum_of_p_q_r_s_t_l2196_219640

theorem sum_of_p_q_r_s_t (p q r s t : ℤ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)
  (h_product : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = 120) : 
  p + q + r + s + t = 32 := 
sorry

end sum_of_p_q_r_s_t_l2196_219640


namespace arrange_squares_l2196_219675

theorem arrange_squares (n : ℕ) (h : n ≥ 5) :
  ∃ arrangement : Fin n → Fin n × Fin n, 
    (∀ i j : Fin n, i ≠ j → 
      (arrangement i).fst + (arrangement i).snd = (arrangement j).fst + (arrangement j).snd
      ∨ (arrangement i).fst = (arrangement j).fst
      ∨ (arrangement i).snd = (arrangement j).snd) :=
sorry

end arrange_squares_l2196_219675


namespace distance_to_angle_bisector_l2196_219683

theorem distance_to_angle_bisector 
  (P : ℝ × ℝ) 
  (h_hyperbola : P.1^2 - P.2^2 = 9) 
  (h_distance_to_line_neg_x : abs (P.1 + P.2) = 2016 * Real.sqrt 2) : 
  abs (P.1 - P.2) / Real.sqrt 2 = 448 :=
sorry

end distance_to_angle_bisector_l2196_219683


namespace palace_to_airport_distance_l2196_219605

-- Let I be the distance from the palace to the airport
-- Let v be the speed of the Emir's car
-- Let t be the time taken to travel from the palace to the airport

theorem palace_to_airport_distance (v t I : ℝ) 
    (h1 : v = I / t) 
    (h2 : v + 20 = I / (t - 2 / 60)) 
    (h3 : v - 20 = I / (t + 3 / 60)) : 
    I = 20 := by
  sorry

end palace_to_airport_distance_l2196_219605


namespace stella_weeks_l2196_219614

-- Define the constants used in the conditions
def rolls_per_bathroom_per_day : ℕ := 1
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def rolls_per_pack : ℕ := 12
def packs_bought : ℕ := 14

-- Define the total number of rolls Stella uses per day and per week
def rolls_per_day := rolls_per_bathroom_per_day * bathrooms
def rolls_per_week := rolls_per_day * days_per_week

-- Calculate the total number of rolls bought
def total_rolls_bought := packs_bought * rolls_per_pack

-- Calculate the number of weeks Stella bought toilet paper for
def weeks := total_rolls_bought / rolls_per_week

theorem stella_weeks : weeks = 4 := by
  sorry

end stella_weeks_l2196_219614


namespace arithmetic_sum_ratio_l2196_219696

variable (a_n : ℕ → ℤ) -- the arithmetic sequence
variable (S : ℕ → ℤ) -- sum of the first n terms of the sequence
variable (d : ℤ) (a₁ : ℤ) -- common difference and first term of the sequence

-- Definition of the sum of the first n terms in an arithmetic sequence
def arithmetic_sum (n : ℕ) : ℤ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

-- Given condition
axiom h1 : (S 6) / (S 3) = 3

-- Definition of S_n in terms of the given formula
axiom S_def : ∀ n, S n = arithmetic_sum n

-- The main goal to prove
theorem arithmetic_sum_ratio : S 12 / S 9 = 5 / 3 := by
  sorry

end arithmetic_sum_ratio_l2196_219696
