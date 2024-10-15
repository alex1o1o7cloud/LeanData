import Mathlib

namespace NUMINAMATH_GPT_new_average_weight_l2137_213763

theorem new_average_weight (avg_weight_19_students : ℝ) (new_student_weight : ℝ) (num_students_initial : ℕ) : 
  avg_weight_19_students = 15 → new_student_weight = 7 → num_students_initial = 19 → 
  let total_weight_with_new_student := (avg_weight_19_students * num_students_initial + new_student_weight) 
  let new_num_students := num_students_initial + 1 
  let new_avg_weight := total_weight_with_new_student / new_num_students 
  new_avg_weight = 14.6 :=
by
  intros h1 h2 h3
  let total_weight := avg_weight_19_students * num_students_initial
  let total_weight_with_new_student := total_weight + new_student_weight
  let new_num_students := num_students_initial + 1
  let new_avg_weight := total_weight_with_new_student / new_num_students
  have h4 : total_weight = 285 := by sorry
  have h5 : total_weight_with_new_student = 292 := by sorry
  have h6 : new_num_students = 20 := by sorry
  have h7 : new_avg_weight = 292 / 20 := by sorry
  have h8 : new_avg_weight = 14.6 := by sorry
  exact h8

end NUMINAMATH_GPT_new_average_weight_l2137_213763


namespace NUMINAMATH_GPT_algebraic_expression_value_l2137_213715

theorem algebraic_expression_value
  (a : ℝ) 
  (h : a^2 + 2 * a - 1 = 0) : 
  -a^2 - 2 * a + 8 = 7 :=
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2137_213715


namespace NUMINAMATH_GPT_dragon_jewels_end_l2137_213741

-- Given conditions
variables (D : ℕ) (jewels_taken_by_king jewels_taken_from_king new_jewels final_jewels : ℕ)

-- Conditions corresponding to the problem
axiom h1 : jewels_taken_by_king = 3
axiom h2 : jewels_taken_from_king = 2 * jewels_taken_by_king
axiom h3 : new_jewels = jewels_taken_from_king
axiom h4 : new_jewels = D / 3

-- Equation derived from the problem setting
def number_of_jewels_initial := D
def number_of_jewels_after_king_stole := number_of_jewels_initial - jewels_taken_by_king
def number_of_jewels_final := number_of_jewels_after_king_stole + jewels_taken_from_king

-- Final proof obligation
theorem dragon_jewels_end : ∃ (D : ℕ), number_of_jewels_final D 3 6 = 21 :=
by
  sorry

end NUMINAMATH_GPT_dragon_jewels_end_l2137_213741


namespace NUMINAMATH_GPT_min_ab_square_is_four_l2137_213719

noncomputable def min_ab_square : Prop :=
  ∃ a b : ℝ, (a^2 + b^2 = 4 ∧ ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0)

theorem min_ab_square_is_four : min_ab_square :=
  sorry

end NUMINAMATH_GPT_min_ab_square_is_four_l2137_213719


namespace NUMINAMATH_GPT_solve_equation_l2137_213714

theorem solve_equation (x : ℝ) : (x + 4)^2 = 5 * (x + 4) ↔ (x = -4 ∨ x = 1) :=
by sorry

end NUMINAMATH_GPT_solve_equation_l2137_213714


namespace NUMINAMATH_GPT_inequality_may_not_hold_l2137_213780

theorem inequality_may_not_hold (a b c : ℝ) (h : a > b) : (c < 0) → ¬ (a/c > b/c) := 
sorry

end NUMINAMATH_GPT_inequality_may_not_hold_l2137_213780


namespace NUMINAMATH_GPT_shortest_side_of_right_triangle_l2137_213712

theorem shortest_side_of_right_triangle 
  (a b : ℕ) (ha : a = 7) (hb : b = 10) (c : ℝ) (hright : a^2 + b^2 = c^2) :
  min a b = 7 :=
by
  sorry

end NUMINAMATH_GPT_shortest_side_of_right_triangle_l2137_213712


namespace NUMINAMATH_GPT_binomial_expansion_of_110_minus_1_l2137_213779

theorem binomial_expansion_of_110_minus_1:
  110^5 - 5 * 110^4 + 10 * 110^3 - 10 * 110^2 + 5 * 110 - 1 = 109^5 :=
by
  -- We will use the binomial theorem: (a - b)^n = ∑ (k in range(n+1)), C(n, k) * a^(n-k) * (-b)^k
  -- where C(n, k) are the binomial coefficients.
  sorry

end NUMINAMATH_GPT_binomial_expansion_of_110_minus_1_l2137_213779


namespace NUMINAMATH_GPT_probability_of_other_girl_l2137_213769

theorem probability_of_other_girl (A B : Prop) (P : Prop → ℝ) 
    (hA : P A = 3 / 4) 
    (hAB : P (A ∧ B) = 1 / 4) : 
    P (B ∧ A) / P A = 1 / 3 := by 
  -- The proof is skipped using the sorry keyword.
  sorry

end NUMINAMATH_GPT_probability_of_other_girl_l2137_213769


namespace NUMINAMATH_GPT_combined_earnings_l2137_213790

theorem combined_earnings (dwayne_earnings brady_earnings : ℕ) (h1 : dwayne_earnings = 1500) (h2 : brady_earnings = dwayne_earnings + 450) : 
  dwayne_earnings + brady_earnings = 3450 :=
by 
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_combined_earnings_l2137_213790


namespace NUMINAMATH_GPT_mystical_swamp_l2137_213776

/-- 
In a mystical swamp, there are two species of talking amphibians: toads, whose statements are always true, and frogs, whose statements are always false. 
Five amphibians: Adam, Ben, Cara, Dan, and Eva make the following statements:
Adam: "Eva and I are different species."
Ben: "Cara is a frog."
Cara: "Dan is a frog."
Dan: "Of the five of us, at least three are toads."
Eva: "Adam is a toad."
Given these statements, prove that the number of frogs is 3.
-/
theorem mystical_swamp :
  (∀ α β : Prop, α ∨ ¬β) ∧ -- Adam's statement: "Eva and I are different species."
  (Cara = "frog") ∧          -- Ben's statement: "Cara is a frog."
  (Dan = "frog") ∧         -- Cara's statement: "Dan is a frog."
  (∃ t, t = nat → t ≥ 3) ∧ -- Dan's statement: "Of the five of us, at least three are toads."
  (Adam = "toad")               -- Eva's statement: "Adam is a toad."
  → num_frogs = 3 := sorry       -- Number of frogs is 3.

end NUMINAMATH_GPT_mystical_swamp_l2137_213776


namespace NUMINAMATH_GPT_factor_expression_l2137_213731

theorem factor_expression (z : ℤ) : 55 * z^17 + 121 * z^34 = 11 * z^17 * (5 + 11 * z^17) := 
by sorry

end NUMINAMATH_GPT_factor_expression_l2137_213731


namespace NUMINAMATH_GPT_inequality_one_solution_l2137_213778

theorem inequality_one_solution (a : ℝ) :
  (∀ x : ℝ, |x^2 + 2 * a * x + 4 * a| ≤ 4 → x = -a) ↔ a = 2 :=
by sorry

end NUMINAMATH_GPT_inequality_one_solution_l2137_213778


namespace NUMINAMATH_GPT_g_675_eq_42_l2137_213748

noncomputable def g : ℕ → ℕ := sorry

axiom gxy : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g15 : g 15 = 18
axiom g45 : g 45 = 24

theorem g_675_eq_42 : g 675 = 42 :=
sorry

end NUMINAMATH_GPT_g_675_eq_42_l2137_213748


namespace NUMINAMATH_GPT_time_to_clear_l2137_213783

def length_train1 := 121 -- in meters
def length_train2 := 153 -- in meters
def speed_train1 := 80 * 1000 / 3600 -- converting km/h to meters/s
def speed_train2 := 65 * 1000 / 3600 -- converting km/h to meters/s

def total_distance := length_train1 + length_train2
def relative_speed := speed_train1 + speed_train2

theorem time_to_clear : 
  (total_distance / relative_speed : ℝ) = 6.80 :=
by
  sorry

end NUMINAMATH_GPT_time_to_clear_l2137_213783


namespace NUMINAMATH_GPT_solve_for_xy_l2137_213718

theorem solve_for_xy (x y : ℝ) 
  (h1 : 0.05 * x + 0.07 * (30 + x) = 14.9)
  (h2 : 0.03 * y - 5.6 = 0.07 * x) : 
  x = 106.67 ∧ y = 435.567 := 
  by 
  sorry

end NUMINAMATH_GPT_solve_for_xy_l2137_213718


namespace NUMINAMATH_GPT_profit_function_l2137_213720

def cost_per_unit : ℝ := 8

def daily_sales_quantity (x : ℝ) : ℝ := -x + 30

def profit_per_unit (x : ℝ) : ℝ := x - cost_per_unit

def total_profit (x : ℝ) : ℝ := (profit_per_unit x) * (daily_sales_quantity x)

theorem profit_function (x : ℝ) : total_profit x = -x^2 + 38*x - 240 :=
  sorry

end NUMINAMATH_GPT_profit_function_l2137_213720


namespace NUMINAMATH_GPT_oranges_worth_as_much_as_bananas_l2137_213758

-- Define the given conditions
def worth_same_bananas_oranges (bananas oranges : ℕ) : Prop :=
  (3 / 4 * 12 : ℝ) = 9 ∧ 9 = 6

/-- Prove how many oranges are worth as much as (2 / 3) * 9 bananas,
    given that (3 / 4) * 12 bananas are worth 6 oranges. -/
theorem oranges_worth_as_much_as_bananas :
  worth_same_bananas_oranges 12 6 →
  (2 / 3 * 9 : ℝ) = 4 :=
by
  sorry

end NUMINAMATH_GPT_oranges_worth_as_much_as_bananas_l2137_213758


namespace NUMINAMATH_GPT_photos_ratio_l2137_213744

theorem photos_ratio (L R C : ℕ) (h1 : R = L) (h2 : C = 12) (h3 : R = C + 24) :
  L / C = 3 :=
by 
  sorry

end NUMINAMATH_GPT_photos_ratio_l2137_213744


namespace NUMINAMATH_GPT_part1_part2_l2137_213771

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * x^2 - Real.log x

theorem part1 (a : ℝ) (h : 0 < a) (hf'1 : (1 - 2 * a * 1 - 1) = -2) :
  a = 1 ∧ (∀ x y : ℝ, y = -2 * (x - 1) → 2 * x + y - 2 = 0) :=
by
  sorry

theorem part2 {a : ℝ} (ha : a ≥ 1 / 8) :
  ∀ x : ℝ, (1 - 2 * a * x - 1 / x) ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2137_213771


namespace NUMINAMATH_GPT_fraction_of_male_fish_l2137_213787

def total_fish : ℕ := 45
def female_fish : ℕ := 15
def male_fish := total_fish - female_fish

theorem fraction_of_male_fish : (male_fish : ℚ) / total_fish = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_of_male_fish_l2137_213787


namespace NUMINAMATH_GPT_tom_saves_promotion_l2137_213798

open Nat

theorem tom_saves_promotion (price : ℕ) (disc_percent : ℕ) (discount_amount : ℕ) 
    (promotion_x_cost second_pair_cost_promo_x promotion_y_cost promotion_savings : ℕ) 
    (h1 : price = 50)
    (h2 : disc_percent = 40)
    (h3 : discount_amount = 15)
    (h4 : second_pair_cost_promo_x = price - (price * disc_percent / 100))
    (h5 : promotion_x_cost = price + second_pair_cost_promo_x)
    (h6 : promotion_y_cost = price + (price - discount_amount))
    (h7 : promotion_savings = promotion_y_cost - promotion_x_cost) :
  promotion_savings = 5 :=
by
  sorry

end NUMINAMATH_GPT_tom_saves_promotion_l2137_213798


namespace NUMINAMATH_GPT_circle_through_ABC_l2137_213762

-- Define points A, B, and C
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (1, 4)

-- Define the circle equation components to be proved
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3*y - 3 = 0

-- The theorem statement that we need to prove
theorem circle_through_ABC : 
  ∃ (D E F : ℝ), (∀ x y, (x, y) = A ∨ (x, y) = B ∨ (x, y) = C → x^2 + y^2 + D*x + E*y + F = 0) 
  → circle_eqn x y :=
sorry

end NUMINAMATH_GPT_circle_through_ABC_l2137_213762


namespace NUMINAMATH_GPT_correlation_coefficient_l2137_213791

theorem correlation_coefficient (variation_explained_by_height : ℝ)
    (variation_explained_by_errors : ℝ)
    (total_variation : variation_explained_by_height + variation_explained_by_errors = 1)
    (percentage_explained_by_height : variation_explained_by_height = 0.71) :
  variation_explained_by_height = 0.71 := 
by
  sorry

end NUMINAMATH_GPT_correlation_coefficient_l2137_213791


namespace NUMINAMATH_GPT_abs_inequality_solution_l2137_213706

theorem abs_inequality_solution (x : ℝ) : (|2 * x - 1| - |x - 2| < 0) ↔ (-1 < x ∧ x < 1) := 
sorry

end NUMINAMATH_GPT_abs_inequality_solution_l2137_213706


namespace NUMINAMATH_GPT_squares_have_consecutive_digits_generalized_squares_have_many_consecutive_digits_l2137_213736

theorem squares_have_consecutive_digits (n : ℕ) (h : ∃ j : ℕ, n = 33330 + j ∧ j < 10) :
    ∃ (a b : ℕ), n ^ 2 / 10 ^ a % 10 = n ^ 2 / 10 ^ (a + 1) % 10 :=
by
  sorry

theorem generalized_squares_have_many_consecutive_digits (k : ℕ) (n : ℕ)
  (h1 : k ≥ 4)
  (h2 : ∃ j : ℕ, n = 33333 * 10 ^ (k - 4) + j ∧ j < 10 ^ (k - 4)) :
    ∃ m, ∃ l : ℕ, ∀ i < m, n^2 / 10 ^ (l + i) % 10 = n^2 / 10 ^ l % 10 :=
by
  sorry

end NUMINAMATH_GPT_squares_have_consecutive_digits_generalized_squares_have_many_consecutive_digits_l2137_213736


namespace NUMINAMATH_GPT_marble_probability_l2137_213782

theorem marble_probability
  (total_marbles : ℕ)
  (blue_marbles : ℕ)
  (green_marbles : ℕ)
  (draws : ℕ)
  (prob_first_green : ℚ)
  (prob_second_blue_given_green : ℚ)
  (total_prob : ℚ)
  (h_total : total_marbles = 10)
  (h_blue : blue_marbles = 4)
  (h_green : green_marbles = 6)
  (h_draws : draws = 2)
  (h_prob_first_green : prob_first_green = 3 / 5)
  (h_prob_second_blue_given_green : prob_second_blue_given_green = 4 / 9)
  (h_total_prob : total_prob = 4 / 15) :
  prob_first_green * prob_second_blue_given_green = total_prob := sorry

end NUMINAMATH_GPT_marble_probability_l2137_213782


namespace NUMINAMATH_GPT_area_inside_rectangle_outside_circles_is_4_l2137_213754

-- Specify the problem in Lean 4
theorem area_inside_rectangle_outside_circles_is_4 :
  let CD := 3
  let DA := 5
  let radius_A := 1
  let radius_B := 2
  let radius_C := 3
  let area_rectangle := CD * DA
  let area_circles := (radius_A^2 + radius_B^2 + radius_C^2) * Real.pi / 4
  abs (area_rectangle - area_circles - 4) < 1 :=
by
  repeat { sorry }

end NUMINAMATH_GPT_area_inside_rectangle_outside_circles_is_4_l2137_213754


namespace NUMINAMATH_GPT_range_of_a_l2137_213773

theorem range_of_a (a : ℝ) (h1 : 2 * a + 1 < 17) (h2 : 2 * a + 1 > 7) : 3 < a ∧ a < 8 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2137_213773


namespace NUMINAMATH_GPT_price_of_cashew_nuts_l2137_213713

theorem price_of_cashew_nuts 
  (C : ℝ)  -- price per kilo of cashew nuts
  (P_p : ℝ := 130)  -- price per kilo of peanuts
  (cashew_kilos : ℝ := 3)  -- kilos of cashew nuts bought
  (peanut_kilos : ℝ := 2)  -- kilos of peanuts bought
  (total_kilos : ℝ := 5)  -- total kilos of nuts bought
  (total_price_per_kilo : ℝ := 178)  -- total price per kilo of all nuts
  (h_total_cost : cashew_kilos * C + peanut_kilos * P_p = total_kilos * total_price_per_kilo) :
  C = 210 :=
sorry

end NUMINAMATH_GPT_price_of_cashew_nuts_l2137_213713


namespace NUMINAMATH_GPT_x_squared_minus_y_squared_l2137_213793

-- Define the given conditions as Lean definitions
def x_plus_y : ℚ := 8 / 15
def x_minus_y : ℚ := 1 / 45

-- State the proof problem in Lean 4
theorem x_squared_minus_y_squared : (x_plus_y * x_minus_y = 8 / 675) := 
by
  sorry

end NUMINAMATH_GPT_x_squared_minus_y_squared_l2137_213793


namespace NUMINAMATH_GPT_number_of_cages_l2137_213727

-- Definitions based on the conditions
def parrots_per_cage := 2
def parakeets_per_cage := 6
def total_birds := 72

-- Goal: Prove the number of cages
theorem number_of_cages : 
  (parrots_per_cage + parakeets_per_cage) * x = total_birds → x = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cages_l2137_213727


namespace NUMINAMATH_GPT_find_x1_l2137_213729

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) :
  x1 = 4 / 5 := 
sorry

end NUMINAMATH_GPT_find_x1_l2137_213729


namespace NUMINAMATH_GPT_bike_owners_without_car_l2137_213766

variable (T B C : ℕ) (H1 : T = 500) (H2 : B = 450) (H3 : C = 200)

theorem bike_owners_without_car (total bike_owners car_owners : ℕ) 
  (h_total : total = 500) (h_bike_owners : bike_owners = 450) (h_car_owners : car_owners = 200) : 
  (bike_owners - (bike_owners + car_owners - total)) = 300 := by
  sorry

end NUMINAMATH_GPT_bike_owners_without_car_l2137_213766


namespace NUMINAMATH_GPT_sin_double_angle_identity_l2137_213789

theorem sin_double_angle_identity: 2 * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_identity_l2137_213789


namespace NUMINAMATH_GPT_relationship_between_vars_l2137_213761

variable {α : Type*} [LinearOrderedAddCommGroup α]

theorem relationship_between_vars (a b : α) 
  (h1 : a + b < 0) 
  (h2 : b > 0) : a < -b ∧ -b < b ∧ b < -a :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_vars_l2137_213761


namespace NUMINAMATH_GPT_midpoint_square_sum_l2137_213724

theorem midpoint_square_sum (x y : ℝ) :
  (4, 1) = ((2 + x) / 2, (6 + y) / 2) → x^2 + y^2 = 52 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_square_sum_l2137_213724


namespace NUMINAMATH_GPT_product_of_consecutive_multiples_of_4_divisible_by_768_l2137_213709

theorem product_of_consecutive_multiples_of_4_divisible_by_768 (n : ℤ) :
  (4 * n) * (4 * (n + 1)) * (4 * (n + 2)) % 768 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_of_consecutive_multiples_of_4_divisible_by_768_l2137_213709


namespace NUMINAMATH_GPT_sequence_sum_l2137_213764

noncomputable def a₁ : ℝ := sorry
noncomputable def a₂ : ℝ := sorry
noncomputable def a₃ : ℝ := sorry
noncomputable def a₄ : ℝ := sorry
noncomputable def a₅ : ℝ := sorry
noncomputable def a₆ : ℝ := sorry
noncomputable def a₇ : ℝ := sorry
noncomputable def a₈ : ℝ := sorry
noncomputable def q : ℝ := sorry

axiom condition_1 : a₁ + a₂ + a₃ + a₄ = 1
axiom condition_2 : a₅ + a₆ + a₇ + a₈ = 2
axiom condition_3 : q^4 = 2

theorem sequence_sum : q = (2:ℝ)^(1/4) → a₁ + a₂ + a₃ + a₄ = 1 → 
  (a₁ * q^16 + a₂ * q^17 + a₃ * q^18 + a₄ * q^19) = 16 := 
by
  intros hq hsum_s4
  sorry

end NUMINAMATH_GPT_sequence_sum_l2137_213764


namespace NUMINAMATH_GPT_functional_equation_solution_l2137_213772

-- Define ℕ* (positive integers) as a subtype of ℕ
def Nat.star := {n : ℕ // n > 0}

-- Define the problem statement
theorem functional_equation_solution (f : Nat.star → Nat.star) :
  (∀ m n : Nat.star, m.val ^ 2 + (f n).val ∣ m.val * (f m).val + n.val) →
  (∀ n : Nat.star, f n = n) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l2137_213772


namespace NUMINAMATH_GPT_total_carrots_l2137_213735

theorem total_carrots (sally_carrots fred_carrots mary_carrots : ℕ)
  (h_sally : sally_carrots = 6)
  (h_fred : fred_carrots = 4)
  (h_mary : mary_carrots = 10) :
  sally_carrots + fred_carrots + mary_carrots = 20 := 
by sorry

end NUMINAMATH_GPT_total_carrots_l2137_213735


namespace NUMINAMATH_GPT_min_value_fraction_l2137_213784

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) : 
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ (∀ (x : ℝ) (hx : x = 1 / a + 1 / b), x ≥ m) := 
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l2137_213784


namespace NUMINAMATH_GPT_Eva_arts_marks_difference_l2137_213781

noncomputable def marks_difference_in_arts : ℕ := 
  let M1 := 90
  let A2 := 90
  let S1 := 60
  let M2 := 80
  let A1 := A2 - 75
  let S2 := 90
  A2 - A1

theorem Eva_arts_marks_difference : marks_difference_in_arts = 75 := by
  sorry

end NUMINAMATH_GPT_Eva_arts_marks_difference_l2137_213781


namespace NUMINAMATH_GPT_alloy_problem_l2137_213792

theorem alloy_problem (x : ℝ) (h1 : 0.12 * x + 0.08 * 30 = 0.09333333333333334 * (x + 30)) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_alloy_problem_l2137_213792


namespace NUMINAMATH_GPT_inequality_holds_if_and_only_if_c_lt_0_l2137_213757

theorem inequality_holds_if_and_only_if_c_lt_0 (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (c < 0) :=
sorry

end NUMINAMATH_GPT_inequality_holds_if_and_only_if_c_lt_0_l2137_213757


namespace NUMINAMATH_GPT_coordinates_of_A_in_second_quadrant_l2137_213721

noncomputable def coordinates_A (m : ℤ) : ℤ × ℤ :=
  (7 - 2 * m, 5 - m)

theorem coordinates_of_A_in_second_quadrant (m : ℤ) (h1 : 7 - 2 * m < 0) (h2 : 5 - m > 0) :
  coordinates_A m = (-1, 1) := 
sorry

end NUMINAMATH_GPT_coordinates_of_A_in_second_quadrant_l2137_213721


namespace NUMINAMATH_GPT_sum_and_ratio_implies_difference_l2137_213733

theorem sum_and_ratio_implies_difference (a b : ℚ) (h1 : a + b = 500) (h2 : a / b = 0.8) : b - a = 55.55555555555556 := by
  sorry

end NUMINAMATH_GPT_sum_and_ratio_implies_difference_l2137_213733


namespace NUMINAMATH_GPT_total_hangers_l2137_213756

theorem total_hangers (pink green blue yellow orange purple red : ℕ) 
  (h_pink : pink = 7)
  (h_green : green = 4)
  (h_blue : blue = green - 1)
  (h_yellow : yellow = blue - 1)
  (h_orange : orange = 2 * pink)
  (h_purple : purple = yellow + 3)
  (h_red : red = purple / 2) :
  pink + green + blue + yellow + orange + purple + red = 37 :=
sorry

end NUMINAMATH_GPT_total_hangers_l2137_213756


namespace NUMINAMATH_GPT_servant_received_amount_l2137_213799

def annual_salary := 900
def uniform_price := 100
def fraction_of_year_served := 3 / 4

theorem servant_received_amount :
  annual_salary * fraction_of_year_served + uniform_price = 775 := by
  sorry

end NUMINAMATH_GPT_servant_received_amount_l2137_213799


namespace NUMINAMATH_GPT_cos_double_angle_l2137_213751

theorem cos_double_angle (α : ℝ) (h : Real.sin α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l2137_213751


namespace NUMINAMATH_GPT_series_converges_l2137_213702

theorem series_converges (u : ℕ → ℝ) (h : ∀ n, u n = n / (3 : ℝ)^n) :
  ∃ l, 0 ≤ l ∧ l < 1 ∧ ∑' n, u n = l := by
  sorry

end NUMINAMATH_GPT_series_converges_l2137_213702


namespace NUMINAMATH_GPT_dino_remaining_money_l2137_213726

-- Definitions of the conditions
def hours_gig_1 : ℕ := 20
def hourly_rate_gig_1 : ℕ := 10

def hours_gig_2 : ℕ := 30
def hourly_rate_gig_2 : ℕ := 20

def hours_gig_3 : ℕ := 5
def hourly_rate_gig_3 : ℕ := 40

def expenses : ℕ := 500

-- The theorem to be proved: Dino's remaining money at the end of the month
theorem dino_remaining_money : 
  (hours_gig_1 * hourly_rate_gig_1 + hours_gig_2 * hourly_rate_gig_2 + hours_gig_3 * hourly_rate_gig_3) - expenses = 500 := by
  sorry

end NUMINAMATH_GPT_dino_remaining_money_l2137_213726


namespace NUMINAMATH_GPT_sequence_term_a1000_l2137_213701

theorem sequence_term_a1000 :
  ∃ (a : ℕ → ℕ), a 1 = 1007 ∧ a 2 = 1008 ∧
  (∀ n, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n) ∧
  a 1000 = 1673 :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_a1000_l2137_213701


namespace NUMINAMATH_GPT_sum_of_cube_edges_l2137_213723

/-- A cube has 12 edges. Each edge of a cube is of equal length. Given the length of one
edge as 15 cm, the sum of the lengths of all the edges of the cube is 180 cm. -/
theorem sum_of_cube_edges (edge_length : ℝ) (num_edges : ℕ) (h1 : edge_length = 15) (h2 : num_edges = 12) :
  num_edges * edge_length = 180 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cube_edges_l2137_213723


namespace NUMINAMATH_GPT_sum_of_digits_of_greatest_prime_divisor_l2137_213745

-- Define the number 32767
def number : ℕ := 32767

-- Assert that 32767 is 2^15 - 1
lemma number_def : number = 2^15 - 1 := by
  sorry

-- State that 151 is the greatest prime divisor of 32767
lemma greatest_prime_divisor : Nat.Prime 151 ∧ ∀ p : ℕ, Nat.Prime p → p ∣ number → p ≤ 151 := by
  sorry

-- Calculate the sum of the digits of 151
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Conclude the sum of the digits of the greatest prime divisor is 7
theorem sum_of_digits_of_greatest_prime_divisor : sum_of_digits 151 = 7 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_greatest_prime_divisor_l2137_213745


namespace NUMINAMATH_GPT_charles_pictures_after_work_l2137_213788

variable (initial_papers : ℕ)
variable (draw_today : ℕ)
variable (draw_yesterday_morning : ℕ)
variable (papers_left : ℕ)

theorem charles_pictures_after_work :
    initial_papers = 20 →
    draw_today = 6 →
    draw_yesterday_morning = 6 →
    papers_left = 2 →
    initial_papers - (draw_today + draw_yesterday_morning + 6) = papers_left →
    6 = (initial_papers - draw_today - draw_yesterday_morning - papers_left) := 
by
  intros h1 h2 h3 h4 h5
  exact sorry

end NUMINAMATH_GPT_charles_pictures_after_work_l2137_213788


namespace NUMINAMATH_GPT_adult_tickets_sold_l2137_213765

theorem adult_tickets_sold (A S : ℕ) (h1 : S = 3 * A) (h2 : A + S = 600) : A = 150 :=
by
  sorry

end NUMINAMATH_GPT_adult_tickets_sold_l2137_213765


namespace NUMINAMATH_GPT_grandfather_older_than_xiaoming_dad_age_when_twenty_times_xiaoming_l2137_213749

-- Definition of the conditions
def grandfather_age (gm_age dad_age : ℕ) := gm_age = 2 * dad_age
def dad_age_eight_times_xiaoming (dad_age xm_age : ℕ) := dad_age = 8 * xm_age
def grandfather_age_61 (gm_age : ℕ) := gm_age = 61
def twenty_times_xiaoming (gm_age xm_age : ℕ) := gm_age = 20 * xm_age

-- Question 1: Proof that Grandpa is 57 years older than Xiaoming 
theorem grandfather_older_than_xiaoming (gm_age dad_age xm_age : ℕ) 
  (h1 : grandfather_age gm_age dad_age) (h2 : dad_age_eight_times_xiaoming dad_age xm_age)
  (h3 : grandfather_age_61 gm_age)
  : gm_age - xm_age = 57 := 
sorry

-- Question 2: Proof that Dad is 31 years old when Grandpa's age is twenty times Xiaoming's age
theorem dad_age_when_twenty_times_xiaoming (gm_age dad_age xm_age : ℕ) 
  (h1 : twenty_times_xiaoming gm_age xm_age)
  (hm : grandfather_age gm_age dad_age)
  : dad_age = 31 :=
sorry

end NUMINAMATH_GPT_grandfather_older_than_xiaoming_dad_age_when_twenty_times_xiaoming_l2137_213749


namespace NUMINAMATH_GPT_number_is_two_l2137_213722

theorem number_is_two 
  (N : ℝ)
  (h1 : N = 4 * 1 / 2)
  (h2 : (1 / 2) * N = 1) :
  N = 2 :=
sorry

end NUMINAMATH_GPT_number_is_two_l2137_213722


namespace NUMINAMATH_GPT_find_line_equation_l2137_213743

theorem find_line_equation :
  ∃ (m : ℝ), ∃ (b : ℝ), (∀ x y : ℝ,
  (x + 3 * y - 2 = 0 → y = -1/3 * x + 2/3) ∧
  (x = 3 → y = 0) →
  y = m * x + b) ∧
  (m = 3 ∧ b = -9) :=
  sorry

end NUMINAMATH_GPT_find_line_equation_l2137_213743


namespace NUMINAMATH_GPT_valid_starting_day_count_l2137_213728

-- Defining the structure of the 30-day month and conditions
def days_in_month : Nat := 30

-- A function to determine the number of each weekday in a month which also checks if the given day is valid as per conditions
def valid_starting_days : List Nat :=
  [1] -- '1' represents Tuesday being the valid starting day corresponding to equal number of Tuesdays and Thursdays

-- The theorem we want to prove
-- The goal is to prove that there is only 1 valid starting day for the 30-day month to have equal number of Tuesdays and Thursdays
theorem valid_starting_day_count (days : Nat) (valid_days : List Nat) : 
  days = days_in_month → valid_days = valid_starting_days :=
by
  -- Sorry to skip full proof implementation
  sorry

end NUMINAMATH_GPT_valid_starting_day_count_l2137_213728


namespace NUMINAMATH_GPT_problem_statement_l2137_213759

theorem problem_statement (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
    (h3 : m + 5 < n) 
    (h4 : (m + 3 + m + 7 + m + 13 + n + 4 + n + 5 + 2 * n + 3) / 6 = n + 3)
    (h5 : (↑((m + 13) + (n + 4)) / 2 : ℤ) = n + 3) : 
  m + n = 37 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2137_213759


namespace NUMINAMATH_GPT_cos_540_eq_neg1_l2137_213711

theorem cos_540_eq_neg1 : Real.cos (540 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_GPT_cos_540_eq_neg1_l2137_213711


namespace NUMINAMATH_GPT_unique_triplet_satisfying_conditions_l2137_213760

theorem unique_triplet_satisfying_conditions :
  ∃! (a b c: ℕ), 1 < a ∧ 1 < b ∧ 1 < c ∧
                 (c ∣ a * b + 1) ∧
                 (b ∣ c * a + 1) ∧
                 (a ∣ b * c + 1) ∧
                 a = 2 ∧ b = 3 ∧ c = 7 :=
by
  sorry

end NUMINAMATH_GPT_unique_triplet_satisfying_conditions_l2137_213760


namespace NUMINAMATH_GPT_card_area_after_one_inch_shortening_l2137_213753

def initial_length := 5
def initial_width := 7
def new_area_shortened_side_two := 21
def shorter_side_reduction := 2
def longer_side_reduction := 1

theorem card_area_after_one_inch_shortening :
  (initial_length - shorter_side_reduction) * initial_width = new_area_shortened_side_two →
  initial_length * (initial_width - longer_side_reduction) = 30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_card_area_after_one_inch_shortening_l2137_213753


namespace NUMINAMATH_GPT_intersection_eq_set_l2137_213707

def M : Set ℤ := { x | -4 < (x : Int) ∧ x < 2 }
def N : Set Int := { x | (x : ℝ) ^ 2 < 4 }
def intersection := M ∩ N

theorem intersection_eq_set : intersection = {-1, 0, 1} := 
sorry

end NUMINAMATH_GPT_intersection_eq_set_l2137_213707


namespace NUMINAMATH_GPT_average_calculation_l2137_213734

def average_two (a b : ℚ) : ℚ := (a + b) / 2
def average_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem average_calculation :
  average_three (average_three 2 2 0) (average_two 1 2) 1 = 23 / 18 :=
by sorry

end NUMINAMATH_GPT_average_calculation_l2137_213734


namespace NUMINAMATH_GPT_range_of_3a_minus_b_l2137_213703

theorem range_of_3a_minus_b (a b : ℝ) (ha : -5 < a) (ha' : a < 2) (hb : 1 < b) (hb' : b < 4) : 
  -19 < 3 * a - b ∧ 3 * a - b < 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_3a_minus_b_l2137_213703


namespace NUMINAMATH_GPT_sum_of_a2_and_a3_l2137_213797

theorem sum_of_a2_and_a3 (S : ℕ → ℕ) (hS : ∀ n, S n = 3^n + 1) :
  S 3 - S 1 = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_a2_and_a3_l2137_213797


namespace NUMINAMATH_GPT_complement_M_eq_interval_l2137_213704

-- Definition of the set M
def M : Set ℝ := { x | x * (x - 3) > 0 }

-- Universal set is ℝ
def U : Set ℝ := Set.univ

-- Theorem to prove the complement of M in ℝ is [0, 3]
theorem complement_M_eq_interval :
  U \ M = { x | 0 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end NUMINAMATH_GPT_complement_M_eq_interval_l2137_213704


namespace NUMINAMATH_GPT_value_of_f_at_3_l2137_213747

def f (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 - 3 * x + 7

theorem value_of_f_at_3 : f 3 = 196 := by
  sorry

end NUMINAMATH_GPT_value_of_f_at_3_l2137_213747


namespace NUMINAMATH_GPT_probability_at_least_four_girls_l2137_213777

noncomputable def binomial_probability (n : ℕ) (k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_at_least_four_girls
  (n : ℕ)
  (p : ℝ)
  (q : ℝ)
  (h_pq : p + q = 1)
  (h_p : p = 0.55)
  (h_q : q = 0.45)
  (h_n : n = 7) :
  (binomial_probability n 4 p) + (binomial_probability n 5 p) + (binomial_probability n 6 p) + (binomial_probability n 7 p) = 0.59197745 :=
sorry

end NUMINAMATH_GPT_probability_at_least_four_girls_l2137_213777


namespace NUMINAMATH_GPT_elder_age_is_33_l2137_213730

-- Define the conditions
variables (y e : ℕ)

def age_difference_condition : Prop :=
  e = y + 20

def age_reduced_condition : Prop :=
  e - 8 = 5 * (y - 8)

-- State the theorem to prove the age of the elder person
theorem elder_age_is_33 (h1 : age_difference_condition y e) (h2 : age_reduced_condition y e): e = 33 :=
  sorry

end NUMINAMATH_GPT_elder_age_is_33_l2137_213730


namespace NUMINAMATH_GPT_profit_difference_l2137_213785

theorem profit_difference
  (p1 p2 : ℝ)
  (h1 : p1 > p2)
  (h2 : p1 + p2 = 3635000)
  (h3 : p2 = 442500) :
  p1 - p2 = 2750000 :=
by 
  sorry

end NUMINAMATH_GPT_profit_difference_l2137_213785


namespace NUMINAMATH_GPT_long_jump_record_l2137_213708

theorem long_jump_record 
  (standard_distance : ℝ)
  (jump1 : ℝ)
  (jump2 : ℝ)
  (record1 : ℝ)
  (record2 : ℝ)
  (h1 : standard_distance = 4.00)
  (h2 : jump1 = 4.22)
  (h3 : jump2 = 3.85)
  (h4 : record1 = jump1 - standard_distance)
  (h5 : record2 = jump2 - standard_distance)
  : record2 = -0.15 := 
sorry

end NUMINAMATH_GPT_long_jump_record_l2137_213708


namespace NUMINAMATH_GPT_findAngleC_findPerimeter_l2137_213750

noncomputable def triangleCondition (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m := (b+c, Real.sin A)
  let n := (a+b, Real.sin C - Real.sin B)
  m.1 * n.2 = m.2 * n.1 -- m parallel to n

noncomputable def lawOfSines (a b c A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

noncomputable def areaOfTriangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C -- Area calculation by a, b, and angle between them

theorem findAngleC (a b c A B C : ℝ) : 
  triangleCondition a b c A B C ∧ lawOfSines a b c A B C → 
  Real.cos C = -1/2 :=
sorry

theorem findPerimeter (a b c A B C : ℝ) : 
  b = 4 ∧ areaOfTriangle a b c A B C = 4 * Real.sqrt 3 → 
  a = 4 ∧ b = 4 ∧ c = 4 * Real.sqrt 3 ∧ a + b + c = 8 + 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_findAngleC_findPerimeter_l2137_213750


namespace NUMINAMATH_GPT_polynomial_divisibility_l2137_213717

theorem polynomial_divisibility (n : ℕ) : 120 ∣ (n^5 - 5*n^3 + 4*n) :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l2137_213717


namespace NUMINAMATH_GPT_log_term_evaluation_l2137_213725

theorem log_term_evaluation : (Real.log 2)^2 + (Real.log 5)^2 + 2 * (Real.log 2) * (Real.log 5) = 1 := by
  sorry

end NUMINAMATH_GPT_log_term_evaluation_l2137_213725


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l2137_213738

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 2 * a 5 = (a 4) ^ 2)
  (h4 : d ≠ 0) : d = -1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l2137_213738


namespace NUMINAMATH_GPT_S15_eq_l2137_213755

-- Definitions in terms of the geometric sequence and given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions given in the problem
axiom geom_seq (n : ℕ) : S n = (a 0) * (1 - (a 1) ^ n) / (1 - (a 1))
axiom S5_eq : S 5 = 10
axiom S10_eq : S 10 = 50

-- The problem statement to prove
theorem S15_eq : S 15 = 210 :=
by sorry

end NUMINAMATH_GPT_S15_eq_l2137_213755


namespace NUMINAMATH_GPT_find_k_intersect_lines_l2137_213732

theorem find_k_intersect_lines :
  ∃ (k : ℚ), ∀ (x y : ℚ), 
  (2 * x + 3 * y + 8 = 0) → (x - y - 1 = 0) → (x + k * y = 0) → k = -1/2 :=
by sorry

end NUMINAMATH_GPT_find_k_intersect_lines_l2137_213732


namespace NUMINAMATH_GPT_base_conversion_unique_b_l2137_213752

theorem base_conversion_unique_b (b : ℕ) (h_b_pos : 0 < b) :
  (1 * 5^2 + 3 * 5^1 + 2 * 5^0) = (2 * b^2 + b) → b = 4 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_unique_b_l2137_213752


namespace NUMINAMATH_GPT_general_admission_tickets_l2137_213774

-- Define the number of student tickets and general admission tickets
variables {S G : ℕ}

-- Define the conditions
def tickets_sold (S G : ℕ) : Prop := S + G = 525
def amount_collected (S G : ℕ) : Prop := 4 * S + 6 * G = 2876

-- The theorem to prove that the number of general admission tickets is 388
theorem general_admission_tickets : 
  ∀ (S G : ℕ), tickets_sold S G → amount_collected S G → G = 388 :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_general_admission_tickets_l2137_213774


namespace NUMINAMATH_GPT_find_n_l2137_213705

theorem find_n (n : ℤ) (h₁ : 50 ≤ n ∧ n ≤ 120)
               (h₂ : n % 8 = 0)
               (h₃ : n % 12 = 4)
               (h₄ : n % 7 = 4) : 
  n = 88 :=
sorry

end NUMINAMATH_GPT_find_n_l2137_213705


namespace NUMINAMATH_GPT_polygon_diagonals_l2137_213796

theorem polygon_diagonals (n : ℕ) (h : n - 3 = 4) : n = 7 :=
sorry

end NUMINAMATH_GPT_polygon_diagonals_l2137_213796


namespace NUMINAMATH_GPT_fee_difference_l2137_213768

-- Defining the given conditions
def stadium_capacity : ℕ := 2000
def fraction_full : ℚ := 3 / 4
def entry_fee : ℚ := 20

-- Statement to prove
theorem fee_difference :
  let people_at_three_quarters := stadium_capacity * fraction_full
  let total_fees_at_three_quarters := people_at_three_quarters * entry_fee
  let total_fees_full := stadium_capacity * entry_fee
  total_fees_full - total_fees_at_three_quarters = 10000 :=
by
  sorry

end NUMINAMATH_GPT_fee_difference_l2137_213768


namespace NUMINAMATH_GPT_sum_of_remainders_l2137_213770

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 11) : (n % 4) + (n % 5) = 4 :=
by
  -- sorry is here to skip the actual proof as per instructions
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l2137_213770


namespace NUMINAMATH_GPT_longest_sequence_positive_integer_x_l2137_213786

theorem longest_sequence_positive_integer_x :
  ∃ x : ℤ, 0 < x ∧ 34 * x - 10500 > 0 ∧ 17000 - 55 * x > 0 ∧ x = 309 :=
by
  use 309
  sorry

end NUMINAMATH_GPT_longest_sequence_positive_integer_x_l2137_213786


namespace NUMINAMATH_GPT_track_length_eq_900_l2137_213737

/-- 
Bruce and Bhishma are running on a circular track. 
The speed of Bruce is 30 m/s and that of Bhishma is 20 m/s.
They start from the same point at the same time in the same direction.
They meet again for the first time after 90 seconds. 
Prove that the length of the track is 900 meters.
-/
theorem track_length_eq_900 :
  let speed_bruce := 30 -- [m/s]
  let speed_bhishma := 20 -- [m/s]
  let time_meet := 90 -- [s]
  let distance_bruce := speed_bruce * time_meet
  let distance_bhishma := speed_bhishma * time_meet
  let track_length := distance_bruce - distance_bhishma
  track_length = 900 :=
by
  let speed_bruce := 30
  let speed_bhishma := 20
  let time_meet := 90
  let distance_bruce := speed_bruce * time_meet
  let distance_bhishma := speed_bhishma * time_meet
  let track_length := distance_bruce - distance_bhishma
  have : track_length = 900 := by
    sorry
  exact this

end NUMINAMATH_GPT_track_length_eq_900_l2137_213737


namespace NUMINAMATH_GPT_prize_distribution_l2137_213746

/--
In a best-of-five competition where two players of equal level meet in the final, 
with a score of 2:1 after the first three games and the total prize money being 12,000 yuan, 
the prize awarded to the player who has won 2 games should be 9,000 yuan.
-/
theorem prize_distribution (prize_money : ℝ) 
  (A_wins : ℕ) (B_wins : ℕ) (prob_A : ℝ) (prob_B : ℝ) (total_games : ℕ) : 
  total_games = 5 → 
  prize_money = 12000 → 
  A_wins = 2 → 
  B_wins = 1 → 
  prob_A = 1/2 → 
  prob_B = 1/2 → 
  ∃ prize_for_A : ℝ, prize_for_A = 9000 :=
by
  intros
  sorry

end NUMINAMATH_GPT_prize_distribution_l2137_213746


namespace NUMINAMATH_GPT_product_of_number_and_sum_of_digits_l2137_213794

-- Definitions according to the conditions
def units_digit (a b : ℕ) : Prop := b = a + 2
def number_equals_24 (a b : ℕ) : Prop := 10 * a + b = 24

-- The main statement to prove the product of the number and the sum of its digits
theorem product_of_number_and_sum_of_digits :
  ∃ (a b : ℕ), units_digit a b ∧ number_equals_24 a b ∧ (24 * (a + b) = 144) :=
sorry

end NUMINAMATH_GPT_product_of_number_and_sum_of_digits_l2137_213794


namespace NUMINAMATH_GPT_problem_statement_l2137_213716

variable { a b c x y z : ℝ }

theorem problem_statement 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) : 
  a * x + b * y + c * z ≥ 0 :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l2137_213716


namespace NUMINAMATH_GPT_lily_calculation_l2137_213700

theorem lily_calculation (a b c : ℝ) (h1 : a - 2 * b - 3 * c = 2) (h2 : a - 2 * (b - 3 * c) = 14) :
  a - 2 * b = 6 :=
by
  sorry

end NUMINAMATH_GPT_lily_calculation_l2137_213700


namespace NUMINAMATH_GPT_driver_schedule_l2137_213775

-- Definitions based on the conditions
def one_way_trip_time := 160 -- in minutes (2 hours 40 minutes)
def round_trip_time := 320  -- in minutes (5 hours 20 minutes)
def rest_time := 60         -- in minutes (1 hour)

def Driver := ℕ

def A := 1
def B := 2
def C := 3
def D := 4

noncomputable def return_time_A := 760 -- 12:40 PM in minutes from day start (12 * 60 + 40)
noncomputable def earliest_departure_A := 820 -- 13:40 PM in minutes from day start (13 * 60 + 40)
noncomputable def departure_time_D := 785 -- 13:05 PM in minutes from day start (13 * 60 + 5)
noncomputable def second_trip_departure_time := 640 -- 10:40 AM in minutes from day start (10 * 60 + 40)

-- Problem statement
theorem driver_schedule : 
  ∃ (n : ℕ), n = 4 ∧ (∀ i : Driver, i = B → second_trip_departure_time = 640) :=
by
  -- Adding sorry to skip proof
  sorry

end NUMINAMATH_GPT_driver_schedule_l2137_213775


namespace NUMINAMATH_GPT_sixtieth_term_of_arithmetic_sequence_l2137_213795

theorem sixtieth_term_of_arithmetic_sequence (a1 a15 : ℚ) (d : ℚ) (h1 : a1 = 7) (h2 : a15 = 37)
  (h3 : a15 = a1 + 14 * d) : a1 + 59 * d = 134.5 := by
  sorry

end NUMINAMATH_GPT_sixtieth_term_of_arithmetic_sequence_l2137_213795


namespace NUMINAMATH_GPT_trisect_angle_l2137_213739

noncomputable def can_trisect_with_ruler_and_compasses (n : ℕ) : Prop :=
  ¬(3 ∣ n) → ∃ a b : ℤ, 3 * a + n * b = 1

theorem trisect_angle (n : ℕ) (h : ¬(3 ∣ n)) :
  can_trisect_with_ruler_and_compasses n :=
sorry

end NUMINAMATH_GPT_trisect_angle_l2137_213739


namespace NUMINAMATH_GPT_same_color_points_distance_2004_l2137_213767

noncomputable def exists_same_color_points_at_distance_2004 (color : ℝ × ℝ → ℕ) : Prop :=
  ∃ (p q : ℝ × ℝ), (p ≠ q) ∧ (color p = color q) ∧ (dist p q = 2004)

/-- The plane is colored in two colors. Prove that there exist two points of the same color at a distance of 2004 meters. -/
theorem same_color_points_distance_2004 {color : ℝ × ℝ → ℕ}
  (hcolor : ∀ p, color p = 1 ∨ color p = 2) :
  exists_same_color_points_at_distance_2004 color :=
sorry

end NUMINAMATH_GPT_same_color_points_distance_2004_l2137_213767


namespace NUMINAMATH_GPT_time_left_to_use_exerciser_l2137_213742

-- Definitions based on the conditions
def total_time : ℕ := 2 * 60  -- Total time in minutes (120 minutes)
def piano_time : ℕ := 30  -- Time spent on piano
def writing_music_time : ℕ := 25  -- Time spent on writing music
def history_time : ℕ := 38  -- Time spent on history

-- The theorem statement that Joan has 27 minutes left
theorem time_left_to_use_exerciser : 
  total_time - (piano_time + writing_music_time + history_time) = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_time_left_to_use_exerciser_l2137_213742


namespace NUMINAMATH_GPT_binary_operation_l2137_213740

def b11001 := 25  -- binary 11001 is 25 in decimal
def b1101 := 13   -- binary 1101 is 13 in decimal
def b101 := 5     -- binary 101 is 5 in decimal
def b100111010 := 314 -- binary 100111010 is 314 in decimal

theorem binary_operation : (b11001 * b1101 - b101) = b100111010 := by
  -- provide implementation details to prove the theorem
  sorry

end NUMINAMATH_GPT_binary_operation_l2137_213740


namespace NUMINAMATH_GPT_production_value_decreased_by_10_percent_l2137_213710

variable (a : ℝ)

def production_value_in_January : ℝ := a

def production_value_in_February (a : ℝ) : ℝ := 0.9 * a

theorem production_value_decreased_by_10_percent (a : ℝ) :
  production_value_in_February a = 0.9 * production_value_in_January a := 
by
  sorry

end NUMINAMATH_GPT_production_value_decreased_by_10_percent_l2137_213710
