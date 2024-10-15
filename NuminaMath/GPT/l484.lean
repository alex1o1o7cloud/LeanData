import Mathlib

namespace NUMINAMATH_GPT_probability_of_girls_under_18_l484_48441

theorem probability_of_girls_under_18
  (total_members : ℕ)
  (girls : ℕ)
  (boys : ℕ)
  (underaged_girls : ℕ)
  (two_members_chosen : ℕ)
  (total_ways_to_choose_two : ℕ)
  (ways_to_choose_two_girls : ℕ)
  (ways_to_choose_at_least_one_underaged : ℕ)
  (prob : ℚ)
  : 
  total_members = 15 →
  girls = 8 →
  boys = 7 →
  underaged_girls = 3 →
  two_members_chosen = 2 →
  total_ways_to_choose_two = (Nat.choose total_members two_members_chosen) →
  ways_to_choose_two_girls = (Nat.choose girls two_members_chosen) →
  ways_to_choose_at_least_one_underaged = 
    (Nat.choose underaged_girls 1 * Nat.choose (girls - underaged_girls) 1 + Nat.choose underaged_girls 2) →
  prob = (ways_to_choose_at_least_one_underaged : ℚ) / (total_ways_to_choose_two : ℚ) →
  prob = 6 / 35 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_of_girls_under_18_l484_48441


namespace NUMINAMATH_GPT_distinct_prime_factors_90_l484_48445

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end NUMINAMATH_GPT_distinct_prime_factors_90_l484_48445


namespace NUMINAMATH_GPT_pascal_50_5th_element_is_22050_l484_48438

def pascal_fifth_element (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_50_5th_element_is_22050 :
  pascal_fifth_element 50 4 = 22050 :=
by
  -- Calculation steps would go here
  sorry

end NUMINAMATH_GPT_pascal_50_5th_element_is_22050_l484_48438


namespace NUMINAMATH_GPT_solutions_to_quadratic_l484_48419

noncomputable def a : ℝ := (6 + Real.sqrt 92) / 2
noncomputable def b : ℝ := (6 - Real.sqrt 92) / 2

theorem solutions_to_quadratic :
  a ≥ b ∧ ((∀ x : ℝ, x^2 - 6 * x + 11 = 25 → x = a ∨ x = b) → 3 * a + 2 * b = 15 + Real.sqrt 92 / 2) := by
  sorry

end NUMINAMATH_GPT_solutions_to_quadratic_l484_48419


namespace NUMINAMATH_GPT_count_four_digit_numbers_l484_48492

open Int

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def valid_combination (a b c d : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  a ≠ 0 ∧
  a + b + c + d = 10 ∧
  (a + c) - (b + d) % 11 = 0

theorem count_four_digit_numbers :
  ∃ n, n > 0 ∧ ∀ a b c d : ℕ, valid_combination a b c d → n = sorry :=
sorry

end NUMINAMATH_GPT_count_four_digit_numbers_l484_48492


namespace NUMINAMATH_GPT_complex_expression_simplified_l484_48498

theorem complex_expression_simplified :
  let z1 := (1 + 3 * Complex.I) / (1 - 3 * Complex.I)
  let z2 := (1 - 3 * Complex.I) / (1 + 3 * Complex.I)
  let z3 := 1 / (8 * Complex.I^3)
  z1 + z2 + z3 = -1.6 + 0.125 * Complex.I := 
by
  sorry

end NUMINAMATH_GPT_complex_expression_simplified_l484_48498


namespace NUMINAMATH_GPT_divisibility_by_six_l484_48409

theorem divisibility_by_six (n : ℤ) : 6 ∣ (n^3 - n) := 
sorry

end NUMINAMATH_GPT_divisibility_by_six_l484_48409


namespace NUMINAMATH_GPT_radius_of_larger_circle_l484_48414

theorem radius_of_larger_circle (r : ℝ) (r_pos : r > 0)
    (ratio_condition : ∀ (rs : ℝ), rs = 3 * r)
    (diameter_condition : ∀ (ac : ℝ), ac = 6 * r)
    (chord_tangent_condition : ∀ (ab : ℝ), ab = 12) :
     (radius : ℝ) = 3 * r :=
by
  sorry

end NUMINAMATH_GPT_radius_of_larger_circle_l484_48414


namespace NUMINAMATH_GPT_thirty_percent_greater_l484_48402

theorem thirty_percent_greater (x : ℝ) (h : x = 1.3 * 88) : x = 114.4 :=
sorry

end NUMINAMATH_GPT_thirty_percent_greater_l484_48402


namespace NUMINAMATH_GPT_num_integers_satisfying_inequality_l484_48421

theorem num_integers_satisfying_inequality : 
  ∃ (xs : Finset ℤ), (∀ x ∈ xs, -6 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9) ∧ xs.card = 5 := 
by 
  sorry

end NUMINAMATH_GPT_num_integers_satisfying_inequality_l484_48421


namespace NUMINAMATH_GPT_problem_f_prime_at_zero_l484_48459

noncomputable def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5) + 6

theorem problem_f_prime_at_zero : deriv f 0 = 120 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_problem_f_prime_at_zero_l484_48459


namespace NUMINAMATH_GPT_no_natural_n_for_perfect_square_l484_48440

theorem no_natural_n_for_perfect_square :
  ¬ ∃ n : ℕ, ∃ k : ℕ, 2007 + 4^n = k^2 :=
by {
  sorry  -- Proof omitted
}

end NUMINAMATH_GPT_no_natural_n_for_perfect_square_l484_48440


namespace NUMINAMATH_GPT_sum_of_squares_of_coefficients_l484_48448

theorem sum_of_squares_of_coefficients :
  ∃ (a b c d e f : ℤ), (∀ x : ℤ, 8 * x ^ 3 + 64 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) ∧ 
  (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 + e ^ 2 + f ^ 2 = 356) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_coefficients_l484_48448


namespace NUMINAMATH_GPT_remainder_of_x_pow_77_eq_6_l484_48474

theorem remainder_of_x_pow_77_eq_6 (x : ℤ) (h : x^77 % 7 = 6) : x^77 % 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_x_pow_77_eq_6_l484_48474


namespace NUMINAMATH_GPT_Chloe_initial_picked_carrots_l484_48467

variable (x : ℕ)

theorem Chloe_initial_picked_carrots :
  (x - 45 + 42 = 45) → (x = 48) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Chloe_initial_picked_carrots_l484_48467


namespace NUMINAMATH_GPT_candy_in_one_bowl_l484_48466

theorem candy_in_one_bowl (total_candies : ℕ) (eaten_candies : ℕ) (bowls : ℕ) (taken_per_bowl : ℕ) 
  (h1 : total_candies = 100) (h2 : eaten_candies = 8) (h3 : bowls = 4) (h4 : taken_per_bowl = 3) :
  (total_candies - eaten_candies) / bowls - taken_per_bowl = 20 :=
by
  sorry

end NUMINAMATH_GPT_candy_in_one_bowl_l484_48466


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l484_48443

theorem average_of_remaining_two_numbers :
  ∀ (a b c d e f : ℝ),
    (a + b + c + d + e + f) / 6 = 3.95 →
    (a + b) / 2 = 3.6 →
    (c + d) / 2 = 3.85 →
    ((e + f) / 2 = 4.4) :=
by
  intros a b c d e f h1 h2 h3
  have h4 : a + b + c + d + e + f = 23.7 := sorry
  have h5 : a + b = 7.2 := sorry
  have h6 : c + d = 7.7 := sorry
  have h7 : e + f = 8.8 := sorry
  exact sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l484_48443


namespace NUMINAMATH_GPT_new_person_weight_l484_48487

-- Define the initial conditions
def initial_average_weight (w : ℕ) := 6 * w -- The total weight of 6 persons

-- Define the scenario where the average weight increases by 2 kg
def total_weight_increase := 6 * 2 -- The total increase in weight due to an increase of 2 kg in average weight

def person_replaced := 75 -- The weight of the person being replaced

-- Define the expected condition on the weight of the new person
theorem new_person_weight (w_new : ℕ) :
  initial_average_weight person_replaced + total_weight_increase = initial_average_weight (w_new / 6) →
  w_new = 87 :=
sorry

end NUMINAMATH_GPT_new_person_weight_l484_48487


namespace NUMINAMATH_GPT_dasha_meeting_sasha_l484_48424

def stripes_on_zebra : ℕ := 360

variables {v : ℝ} -- speed of Masha
def dasha_speed (v : ℝ) : ℝ := 2 * v -- speed of Dasha (twice Masha's speed)

def masha_distance_before_meeting_sasha : ℕ := 180
def total_stripes_met : ℕ := stripes_on_zebra
def relative_speed_masha_sasha (v : ℝ) : ℝ := v + v -- combined speed of Masha and Sasha
def relative_speed_dasha_sasha (v : ℝ) : ℝ := 3 * v -- combined speed of Dasha and Sasha

theorem dasha_meeting_sasha (v : ℝ) (hv : 0 < v) :
  ∃ t' t'', 
  (t'' = 120 / v) ∧ (dasha_speed v * t' = 240) :=
by {
  sorry
}

end NUMINAMATH_GPT_dasha_meeting_sasha_l484_48424


namespace NUMINAMATH_GPT_william_farm_tax_l484_48425

theorem william_farm_tax :
  let total_tax_collected := 3840
  let william_land_percentage := 0.25
  william_land_percentage * total_tax_collected = 960 :=
by sorry

end NUMINAMATH_GPT_william_farm_tax_l484_48425


namespace NUMINAMATH_GPT_bottles_purchased_l484_48429

/-- Given P bottles can be bought for R dollars, determine how many bottles can be bought for M euros
    if 1 euro is worth 1.2 dollars and there is a 10% discount when buying with euros. -/
theorem bottles_purchased (P R M : ℝ) (hR : R > 0) (hP : P > 0) :
  let euro_to_dollars := 1.2
  let discount := 0.9
  let dollars := euro_to_dollars * M * discount
  (P / R) * dollars = (1.32 * P * M) / R :=
by
  sorry

end NUMINAMATH_GPT_bottles_purchased_l484_48429


namespace NUMINAMATH_GPT_find_number_l484_48470

theorem find_number (x : ℝ) (h : x - (3/5 : ℝ) * x = 60) : x = 150 :=
sorry

end NUMINAMATH_GPT_find_number_l484_48470


namespace NUMINAMATH_GPT_min_value_expression_l484_48442

theorem min_value_expression (x y : ℝ) : 
  (∃ (x_min y_min : ℝ), 
  (x_min = 1/2 ∧ y_min = 0) ∧ 
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = 39/4) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l484_48442


namespace NUMINAMATH_GPT_vector_parallel_y_value_l484_48481

theorem vector_parallel_y_value (y : ℝ) 
  (a : ℝ × ℝ := (3, 2)) 
  (b : ℝ × ℝ := (6, y)) 
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  y = 4 :=
by sorry

end NUMINAMATH_GPT_vector_parallel_y_value_l484_48481


namespace NUMINAMATH_GPT_euclid_middle_school_math_students_l484_48408

theorem euclid_middle_school_math_students
  (students_Germain : ℕ)
  (students_Newton : ℕ)
  (students_Young : ℕ)
  (students_Euler : ℕ)
  (h_Germain : students_Germain = 12)
  (h_Newton : students_Newton = 10)
  (h_Young : students_Young = 7)
  (h_Euler : students_Euler = 6) :
  students_Germain + students_Newton + students_Young + students_Euler = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_euclid_middle_school_math_students_l484_48408


namespace NUMINAMATH_GPT_range_of_a_l484_48412

variable (a b c : ℝ)

def condition1 := a^2 - b * c - 8 * a + 7 = 0

def condition2 := b^2 + c^2 + b * c - 6 * a + 6 = 0

theorem range_of_a (h1 : condition1 a b c) (h2 : condition2 a b c) : 1 ≤ a ∧ a ≤ 9 := 
  sorry

end NUMINAMATH_GPT_range_of_a_l484_48412


namespace NUMINAMATH_GPT_find_mn_l484_48446

variable (OA OB OC : EuclideanSpace ℝ (Fin 3))
variable (AOC BOC : ℝ)

axiom length_OA : ‖OA‖ = 2
axiom length_OB : ‖OB‖ = 2
axiom length_OC : ‖OC‖ = 2 * Real.sqrt 3
axiom tan_angle_AOC : Real.tan AOC = 3 * Real.sqrt 3
axiom angle_BOC : BOC = Real.pi / 3

theorem find_mn : ∃ m n : ℝ, OC = m • OA + n • OB ∧ m = 5 / 3 ∧ n = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_find_mn_l484_48446


namespace NUMINAMATH_GPT_union_inter_complement_l484_48477

open Set

variable (U : Set ℝ := univ)
variable (A : Set ℝ := {x | abs (x - 2) > 3})
variable (B : Set ℝ := {x | x * (-2 - x) > 0})

theorem union_inter_complement 
  (C_U_A : Set ℝ := compl A)
  (A_def : A = {x | abs (x - 2) > 3})
  (B_def : B = {x | x * (-2 - x) > 0})
  (C_U_A_def : C_U_A = compl A) :
  (A ∪ B = {x : ℝ | x < 0} ∪ {x : ℝ | x > 5}) ∧ 
  ((C_U_A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 0}) :=
by
  sorry

end NUMINAMATH_GPT_union_inter_complement_l484_48477


namespace NUMINAMATH_GPT_find_m_intersection_points_l484_48464

theorem find_m (m : ℝ) (hp : 2^2 + 2 * m + m^2 - 3 = 4) (h_pos : m > 0) : m = 1 := 
by
  sorry

theorem intersection_points (m : ℝ) (hp : 2^2 + 2 * m + m^2 - 3 = 4) (h_pos : m > 0) 
  (hm : m = 1) : ∃ x1 x2 : ℝ, (x^2 + x - 2 = 0) ∧ x1 ≠ x2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_intersection_points_l484_48464


namespace NUMINAMATH_GPT_pieces_from_rod_l484_48437

theorem pieces_from_rod (length_of_rod : ℝ) (length_of_piece : ℝ) 
  (h_rod : length_of_rod = 42.5) 
  (h_piece : length_of_piece = 0.85) :
  length_of_rod / length_of_piece = 50 :=
by
  rw [h_rod, h_piece]
  calc
    42.5 / 0.85 = 50 := by norm_num

end NUMINAMATH_GPT_pieces_from_rod_l484_48437


namespace NUMINAMATH_GPT_peter_total_spent_l484_48435

/-
Peter bought a scooter for a certain sum of money. He spent 5% of the cost on the first round of repairs, another 10% on the second round of repairs, and 7% on the third round of repairs. After this, he had to pay a 12% tax on the original cost. Also, he offered a 15% holiday discount on the scooter's selling price. Despite the discount, he still managed to make a profit of $2000. How much did he spend in total, including repairs, tax, and discount if his profit percentage was 30%?
-/

noncomputable def total_spent (C S P : ℝ) : Prop :=
    (0.3 * C = P) ∧
    (0.85 * S = 1.34 * C + P) ∧
    (C = 2000 / 0.3) ∧
    (1.34 * C = 8933.33)

theorem peter_total_spent
  (C S P : ℝ)
  (h1 : 0.3 * C = P)
  (h2 : 0.85 * S = 1.34 * C + P)
  (h3 : C = 2000 / 0.3)
  : 1.34 * C = 8933.33 := by 
  sorry

end NUMINAMATH_GPT_peter_total_spent_l484_48435


namespace NUMINAMATH_GPT_sin_zero_range_valid_m_l484_48497

noncomputable def sin_zero_range (m : ℝ) : Prop :=
  ∀ f : ℝ → ℝ, 
    (∀ x : ℝ, f x = Real.sin (2 * x - Real.pi / 6) - m) →
    (∃ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ Real.pi / 2) ∧ (0 ≤ x2 ∧ x2 ≤ Real.pi / 2) ∧ x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0)

theorem sin_zero_range_valid_m : 
  ∀ m : ℝ, sin_zero_range m ↔ (1 / 2 ≤ m ∧ m < 1) :=
sorry

end NUMINAMATH_GPT_sin_zero_range_valid_m_l484_48497


namespace NUMINAMATH_GPT_find_missing_dimension_l484_48469

-- Definitions based on conditions
def is_dimension_greatest_area (x : ℝ) : Prop :=
  max (2 * x) (max (3 * x) 6) = 15

-- The final statement to prove
theorem find_missing_dimension (x : ℝ) (h1 : is_dimension_greatest_area x) : x = 5 :=
sorry

end NUMINAMATH_GPT_find_missing_dimension_l484_48469


namespace NUMINAMATH_GPT_sum_of_angles_l484_48400

theorem sum_of_angles (a b : ℝ) (ha : a = 45) (hb : b = 225) : a + b = 270 :=
by
  rw [ha, hb]
  norm_num -- Lean's built-in tactic to normalize numerical expressions

end NUMINAMATH_GPT_sum_of_angles_l484_48400


namespace NUMINAMATH_GPT_zero_points_of_gx_l484_48420

noncomputable def fx (a x : ℝ) : ℝ := (1 / 2) * x^2 - abs (x - 2 * a)
noncomputable def gx (a x : ℝ) : ℝ := 4 * a * x^2 + 2 * x + 1

theorem zero_points_of_gx (a : ℝ) (h : -1 / 4 ≤ a ∧ a ≤ 1 / 4) : 
  ∃ n, (n = 1 ∨ n = 2) ∧ (∃ x1 x2, gx a x1 = 0 ∧ gx a x2 = 0) := 
sorry

end NUMINAMATH_GPT_zero_points_of_gx_l484_48420


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l484_48434

def polynomial (x : ℝ) := x^5 + 2 * x^3 - x + 4

theorem remainder_when_divided_by_x_minus_2 :
  polynomial 2 = 50 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l484_48434


namespace NUMINAMATH_GPT_unique_digits_addition_l484_48433

theorem unique_digits_addition :
  ∃ (X Y B M C : ℕ), 
    -- Conditions
    X ≠ 0 ∧ Y ≠ 0 ∧ B ≠ 0 ∧ M ≠ 0 ∧ C ≠ 0 ∧
    X ≠ Y ∧ X ≠ B ∧ X ≠ M ∧ X ≠ C ∧ Y ≠ B ∧ Y ≠ M ∧ Y ≠ C ∧ B ≠ M ∧ B ≠ C ∧ M ≠ C ∧
    -- Addition equation with distinct digits
    (X * 1000 + Y * 100 + 70) + (B * 100 + M * 10 + C) = (B * 1000 + M * 100 + C * 10 + 0) ∧
    -- Correct Answer
    X = 9 ∧ Y = 8 ∧ B = 3 ∧ M = 8 ∧ C = 7 :=
sorry

end NUMINAMATH_GPT_unique_digits_addition_l484_48433


namespace NUMINAMATH_GPT_parabola_equation_l484_48427

-- Defining the point F and the line
def F : ℝ × ℝ := (0, 4)

def line_eq (y : ℝ) : Prop := y = -5

-- Defining the condition that point M is closer to F(0, 4) than to the line y = -5 by less than 1
def condition (M : ℝ × ℝ) : Prop :=
  let dist_to_F := (M.1 - F.1)^2 + (M.2 - F.2)^2
  let dist_to_line := abs (M.2 - (-5))
  abs (dist_to_F - dist_to_line) < 1

-- The equation we need to prove under the given condition
theorem parabola_equation (M : ℝ × ℝ) (h : condition M) : M.1^2 = 16 * M.2 := 
sorry

end NUMINAMATH_GPT_parabola_equation_l484_48427


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l484_48450

theorem isosceles_triangle_base_length
  (perimeter : ℝ)
  (side1 side2 base : ℝ)
  (h_perimeter : perimeter = 18)
  (h_side1 : side1 = 4)
  (h_isosceles : side1 = side2 ∨ side1 = base ∨ side2 = base)
  (h_triangle : side1 + side2 + base = 18) :
  base = 7 := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l484_48450


namespace NUMINAMATH_GPT_rogers_spending_l484_48447

theorem rogers_spending (B m p : ℝ) (H1 : m = 0.25 * (B - p)) (H2 : p = 0.10 * (B - m)) : 
  m + p = (4 / 13) * B :=
sorry

end NUMINAMATH_GPT_rogers_spending_l484_48447


namespace NUMINAMATH_GPT_evaluate_expression_l484_48489

theorem evaluate_expression : 202 - 101 + 9 = 110 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l484_48489


namespace NUMINAMATH_GPT_other_root_eq_six_l484_48452

theorem other_root_eq_six (a : ℝ) (x1 : ℝ) (x2 : ℝ) 
  (h : x1 = -2) 
  (eqn : ∀ x, x^2 - a * x - 3 * a = 0 → (x = x1 ∨ x = x2)) :
  x2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_other_root_eq_six_l484_48452


namespace NUMINAMATH_GPT_father_current_age_l484_48491

variable (M F : ℕ)

/-- The man's current age is (2 / 5) of the age of his father. -/
axiom man_age : M = (2 / 5) * F

/-- After 12 years, the man's age will be (1 / 2) of his father's age. -/
axiom age_relation_in_12_years : (M + 12) = (1 / 2) * (F + 12)

/-- Prove that the father's current age, F, is 60. -/
theorem father_current_age : F = 60 :=
by
  sorry

end NUMINAMATH_GPT_father_current_age_l484_48491


namespace NUMINAMATH_GPT_find_s_for_g_neg1_zero_l484_48499

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem find_s_for_g_neg1_zero (s : ℝ) : g (-1) s = 0 ↔ s = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_s_for_g_neg1_zero_l484_48499


namespace NUMINAMATH_GPT_sequence_eventually_periodic_l484_48488

open Nat

noncomputable def sum_prime_factors_plus_one (K : ℕ) : ℕ := 
  (K.factors.sum) + 1

theorem sequence_eventually_periodic (K : ℕ) (hK : K ≥ 9) :
  ∃ m n : ℕ, m ≠ n ∧ sum_prime_factors_plus_one^[m] K = sum_prime_factors_plus_one^[n] K := 
sorry

end NUMINAMATH_GPT_sequence_eventually_periodic_l484_48488


namespace NUMINAMATH_GPT_initial_incorrect_average_l484_48451

theorem initial_incorrect_average (S_correct S_wrong : ℝ) :
  (S_correct = S_wrong - 26 + 36) →
  (S_correct / 10 = 19) →
  (S_wrong / 10 = 18) :=
by
  sorry

end NUMINAMATH_GPT_initial_incorrect_average_l484_48451


namespace NUMINAMATH_GPT_total_persimmons_l484_48482

-- Definitions based on conditions in a)
def totalWeight (kg : ℕ) := kg = 3
def weightPerFivePersimmons (kg : ℕ) := kg = 1

-- The proof problem
theorem total_persimmons (k : ℕ) (w : ℕ) (x : ℕ) (h1 : totalWeight k) (h2 : weightPerFivePersimmons w) : x = 15 :=
by
  -- With the definitions totalWeight and weightPerFivePersimmons given in the conditions
  -- we aim to prove that the number of persimmons, x, is 15.
  sorry

end NUMINAMATH_GPT_total_persimmons_l484_48482


namespace NUMINAMATH_GPT_probability_of_odd_score_l484_48485

noncomputable def dartboard : Type := sorry

variables (r_inner r_outer : ℝ)
variables (inner_values outer_values : Fin 3 → ℕ)
variables (P_odd : ℚ)

-- Conditions
def dartboard_conditions (r_inner r_outer : ℝ) (inner_values outer_values : Fin 3 → ℕ) : Prop :=
  r_inner = 4 ∧ r_outer = 8 ∧
  inner_values 0 = 3 ∧ inner_values 1 = 1 ∧ inner_values 2 = 1 ∧
  outer_values 0 = 3 ∧ outer_values 1 = 2 ∧ outer_values 2 = 2

-- Correct Answer
def correct_odds_probability (P_odd : ℚ) : Prop :=
  P_odd = 4 / 9

-- Main Statement
theorem probability_of_odd_score (r_inner r_outer : ℝ) (inner_values outer_values : Fin 3 → ℕ) (P_odd : ℚ) :
  dartboard_conditions r_inner r_outer inner_values outer_values →
  correct_odds_probability P_odd :=
sorry

end NUMINAMATH_GPT_probability_of_odd_score_l484_48485


namespace NUMINAMATH_GPT_jose_profit_share_l484_48493

def investment_share (toms_investment : ℕ) (jose_investment : ℕ) 
  (toms_duration : ℕ) (jose_duration : ℕ) (total_profit : ℕ) : ℕ :=
  let toms_capital_months := toms_investment * toms_duration
  let jose_capital_months := jose_investment * jose_duration
  let total_capital_months := toms_capital_months + jose_capital_months
  let jose_share_ratio := jose_capital_months / total_capital_months
  jose_share_ratio * total_profit

theorem jose_profit_share 
  (toms_investment : ℕ := 3000)
  (jose_investment : ℕ := 4500)
  (toms_duration : ℕ := 12)
  (jose_duration : ℕ := 10)
  (total_profit : ℕ := 6300) :
  investment_share toms_investment jose_investment toms_duration jose_duration total_profit = 3500 := 
sorry

end NUMINAMATH_GPT_jose_profit_share_l484_48493


namespace NUMINAMATH_GPT_range_of_a_l484_48484

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x > 2 ↔ x < 2 / (a - 1)) → a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l484_48484


namespace NUMINAMATH_GPT_percentage_increase_is_20_l484_48454

-- Defining the original cost and new cost
def original_cost := 200
def new_total_cost := 480

-- Doubling the capacity means doubling the original cost
def doubled_old_cost := 2 * original_cost

-- The increase in cost
def increase_cost := new_total_cost - doubled_old_cost

-- The percentage increase in cost
def percentage_increase := (increase_cost / doubled_old_cost) * 100

-- The theorem we need to prove
theorem percentage_increase_is_20 : percentage_increase = 20 :=
  by
  sorry

end NUMINAMATH_GPT_percentage_increase_is_20_l484_48454


namespace NUMINAMATH_GPT_probability_of_at_least_one_accurate_forecast_l484_48471

theorem probability_of_at_least_one_accurate_forecast (PA PB : ℝ) (hA : PA = 0.8) (hB : PB = 0.75) :
  1 - ((1 - PA) * (1 - PB)) = 0.95 :=
by
  rw [hA, hB]
  sorry

end NUMINAMATH_GPT_probability_of_at_least_one_accurate_forecast_l484_48471


namespace NUMINAMATH_GPT_cube_volume_l484_48495

-- Define the surface area constant
def surface_area : ℝ := 725.9999999999998

-- Define the formula for surface area of a cube and solve for volume given the conditions
theorem cube_volume (SA : ℝ) (h : SA = surface_area) : 11^3 = 1331 :=
by sorry

end NUMINAMATH_GPT_cube_volume_l484_48495


namespace NUMINAMATH_GPT_second_tap_empties_cistern_l484_48473

theorem second_tap_empties_cistern (t_fill: ℝ) (x: ℝ) (t_net: ℝ) : 
  (1 / 6) - (1 / x) = (1 / 12) → x = 12 := 
by
  sorry

end NUMINAMATH_GPT_second_tap_empties_cistern_l484_48473


namespace NUMINAMATH_GPT_snowboard_final_price_l484_48494

noncomputable def original_price : ℝ := 200
noncomputable def discount_friday : ℝ := 0.40
noncomputable def discount_monday : ℝ := 0.25

noncomputable def price_after_friday_discount (orig : ℝ) (discount : ℝ) : ℝ :=
  (1 - discount) * orig

noncomputable def final_price (price_friday : ℝ) (discount : ℝ) : ℝ :=
  (1 - discount) * price_friday

theorem snowboard_final_price :
  final_price (price_after_friday_discount original_price discount_friday) discount_monday = 90 := 
sorry

end NUMINAMATH_GPT_snowboard_final_price_l484_48494


namespace NUMINAMATH_GPT_zero_in_interval_l484_48411

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem zero_in_interval : 
  ∃ x₀, f x₀ = 0 ∧ (2 : ℝ) < x₀ ∧ x₀ < (3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_zero_in_interval_l484_48411


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l484_48455

theorem sum_of_squares_of_roots :
  (∃ x1 x2 : ℝ, 5 * x1^2 - 3 * x1 - 11 = 0 ∧ 5 * x2^2 - 3 * x2 - 11 = 0 ∧ x1 ≠ x2) →
  (x1 + x2 = 3 / 5 ∧ x1 * x2 = -11 / 5) →
  (x1^2 + x2^2 = 119 / 25) :=
by intro h1 h2; sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l484_48455


namespace NUMINAMATH_GPT_center_coordinates_l484_48475

noncomputable def center_of_circle (x y : ℝ) : Prop := 
  x^2 + y^2 + 2*x - 4*y = 0

theorem center_coordinates : center_of_circle (-1) 2 :=
by sorry

end NUMINAMATH_GPT_center_coordinates_l484_48475


namespace NUMINAMATH_GPT_missing_digit_first_digit_l484_48462

-- Definitions derived from conditions
def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_divisible_by_six (n : ℕ) : Prop := n % 6 = 0
def multiply_by_two (d : ℕ) : ℕ := 2 * d

-- Main statement to prove
theorem missing_digit_first_digit (d : ℕ) (n : ℕ) 
  (h1 : multiply_by_two d = n) 
  (h2 : is_three_digit_number n) 
  (h3 : is_divisible_by_six n)
  (h4 : d = 2)
  : n / 100 = 2 :=
sorry

end NUMINAMATH_GPT_missing_digit_first_digit_l484_48462


namespace NUMINAMATH_GPT_a_n_div_3_sum_two_cubes_a_n_div_3_not_sum_two_squares_l484_48430

def a_n (n : ℕ) : ℕ := 10^(3*n+2) + 2 * 10^(2*n+1) + 2 * 10^(n+1) + 1

theorem a_n_div_3_sum_two_cubes (n : ℕ) : ∃ x y : ℤ, (x > 0) ∧ (y > 0) ∧ (a_n n / 3 = x^3 + y^3) := sorry

theorem a_n_div_3_not_sum_two_squares (n : ℕ) : ¬ (∃ x y : ℤ, a_n n / 3 = x^2 + y^2) := sorry

end NUMINAMATH_GPT_a_n_div_3_sum_two_cubes_a_n_div_3_not_sum_two_squares_l484_48430


namespace NUMINAMATH_GPT_width_of_first_sheet_l484_48422

theorem width_of_first_sheet (w : ℝ) (h : 2 * (w * 17) = 2 * (8.5 * 11) + 100) : w = 287 / 34 :=
by
  sorry

end NUMINAMATH_GPT_width_of_first_sheet_l484_48422


namespace NUMINAMATH_GPT_capacity_of_each_bucket_in_second_case_final_proof_l484_48453

def tank_volume (buckets: ℕ) (bucket_capacity: ℝ) : ℝ := buckets * bucket_capacity

theorem capacity_of_each_bucket_in_second_case
  (total_volume: ℝ)
  (first_case_buckets : ℕ)
  (first_case_capacity : ℝ)
  (second_case_buckets : ℕ) :
  first_case_buckets * first_case_capacity = total_volume → 
  (total_volume / second_case_buckets) = 9 :=
by
  intros h
  sorry

-- Given the conditions:
noncomputable def total_volume := tank_volume 28 13.5

theorem final_proof :
  (tank_volume 28 13.5 = total_volume) → 
  (total_volume / 42 = 9) :=
by
  intro h
  exact capacity_of_each_bucket_in_second_case total_volume 28 13.5 42 h

end NUMINAMATH_GPT_capacity_of_each_bucket_in_second_case_final_proof_l484_48453


namespace NUMINAMATH_GPT_cost_price_per_meter_l484_48418

-- Definitions for conditions
def total_length : ℝ := 9.25
def total_cost : ℝ := 416.25

-- The theorem to be proved
theorem cost_price_per_meter : total_cost / total_length = 45 := by
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_l484_48418


namespace NUMINAMATH_GPT_average_billboards_per_hour_l484_48496

-- Define the number of billboards seen in each hour
def billboards_first_hour := 17
def billboards_second_hour := 20
def billboards_third_hour := 23

-- Define the number of hours
def total_hours := 3

-- Prove that the average number of billboards per hour is 20
theorem average_billboards_per_hour : 
  (billboards_first_hour + billboards_second_hour + billboards_third_hour) / total_hours = 20 :=
by
  sorry

end NUMINAMATH_GPT_average_billboards_per_hour_l484_48496


namespace NUMINAMATH_GPT_solve_for_x_l484_48449

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (5 * x + 2)) (h2 : y = 2) : x = -3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l484_48449


namespace NUMINAMATH_GPT_quarters_to_dollars_l484_48439

theorem quarters_to_dollars (total_quarters : ℕ) (quarters_per_dollar : ℕ) (h1 : total_quarters = 8) (h2 : quarters_per_dollar = 4) : total_quarters / quarters_per_dollar = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_quarters_to_dollars_l484_48439


namespace NUMINAMATH_GPT_line_circle_intersect_l484_48483

theorem line_circle_intersect (a : ℝ) (h : a > 1) :
  ∃ x y : ℝ, (x - a)^2 + (y - 1)^2 = 2 ∧ x - a * y - 2 = 0 :=
sorry

end NUMINAMATH_GPT_line_circle_intersect_l484_48483


namespace NUMINAMATH_GPT_quadratic_roots_condition_l484_48461

theorem quadratic_roots_condition (a : ℝ) :
  (∃ α : ℝ, 5 * α = -(a - 4) ∧ 4 * α^2 = a - 5) ↔ (a = 7 ∨ a = 5) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_condition_l484_48461


namespace NUMINAMATH_GPT_embankment_height_bounds_l484_48486

theorem embankment_height_bounds
  (a : ℝ) (b : ℝ) (h : ℝ)
  (a_eq : a = 5)
  (b_lower_bound : 2 ≤ b)
  (vol_lower_bound : 400 ≤ (25 * (a^2 - b^2)))
  (vol_upper_bound : (25 * (a^2 - b^2)) ≤ 500) :
  1 ≤ h ∧ h ≤ (5 - Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_embankment_height_bounds_l484_48486


namespace NUMINAMATH_GPT_triangle_statements_l484_48472

-- Define the fundamental properties of the triangle
noncomputable def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a = 45 ∧ a = 2 ∧ b = 2 * Real.sqrt 2 ∧ 
  (a - b = c * Real.cos B - c * Real.cos A)

-- Statement A
def statement_A (A B C a b c : ℝ) (h : triangle A B C a b c) : Prop :=
  ∃ B, Real.sin B = 1

-- Statement B
def statement_B (A B C : ℝ) (v_AC v_AB : ℝ) : Prop :=
  v_AC * v_AB > 0 → Real.cos A > 0

-- Statement C
def statement_C (A B : ℝ) (a b : ℝ) : Prop :=
  Real.sin A > Real.sin B → a > b

-- Statement D
def statement_D (A B C a b c : ℝ) (h : triangle A B C a b c) : Prop :=
  (a - b = c * Real.cos B - c * Real.cos A) →
  (a = b ∨ c^2 = a^2 + b^2)

-- Final proof statement
theorem triangle_statements (A B C a b c : ℝ) (v_AC v_AB : ℝ) 
  (h_triangle : triangle A B C a b c) :
  (statement_A A B C a b c h_triangle) ∧
  ¬(statement_B A B C v_AC v_AB) ∧
  (statement_C A B a b) ∧
  (statement_D A B C a b c h_triangle) :=
by sorry

end NUMINAMATH_GPT_triangle_statements_l484_48472


namespace NUMINAMATH_GPT_charles_earnings_l484_48468

def housesit_rate : ℝ := 15
def dog_walk_rate : ℝ := 22
def hours_housesit : ℝ := 10
def num_dogs : ℝ := 3

theorem charles_earnings :
  housesit_rate * hours_housesit + dog_walk_rate * num_dogs = 216 :=
by
  sorry

end NUMINAMATH_GPT_charles_earnings_l484_48468


namespace NUMINAMATH_GPT_central_angle_of_spherical_sector_l484_48416

theorem central_angle_of_spherical_sector (R α r m : ℝ) (h1 : R * Real.pi * r = 2 * R * Real.pi * m) (h2 : R^2 = r^2 + (R - m)^2) :
  α = 2 * Real.arccos (3 / 5) :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_spherical_sector_l484_48416


namespace NUMINAMATH_GPT_flyers_left_l484_48456

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (h1 : total_flyers = 1236) (h2 : jack_flyers = 120) (h3 : rose_flyers = 320) : (total_flyers - (jack_flyers + rose_flyers) = 796) := 
by
  sorry

end NUMINAMATH_GPT_flyers_left_l484_48456


namespace NUMINAMATH_GPT_solution_quadrant_I_l484_48458

theorem solution_quadrant_I (c x y : ℝ) :
  (x - y = 2 ∧ c * x + y = 3 ∧ x > 0 ∧ y > 0) ↔ (-1 < c ∧ c < 3/2) := by
  sorry

end NUMINAMATH_GPT_solution_quadrant_I_l484_48458


namespace NUMINAMATH_GPT_divisibility_problem_l484_48415

theorem divisibility_problem (q : ℕ) (hq : Nat.Prime q) (hq2 : q % 2 = 1) :
  ¬((q + 2)^(q - 3) + 1) % (q - 4) = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % q = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % (q + 6) = 0 ∧
  ¬((q + 2)^(q - 3) + 1) % (q + 3) = 0 := sorry

end NUMINAMATH_GPT_divisibility_problem_l484_48415


namespace NUMINAMATH_GPT_fourth_person_height_is_82_l484_48407

theorem fourth_person_height_is_82 (H : ℕ)
    (h1: (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 76)
    (h_diff1: H + 2 - H = 2)
    (h_diff2: H + 4 - (H + 2) = 2)
    (h_diff3: H + 10 - (H + 4) = 6) :
  (H + 10) = 82 := 
sorry

end NUMINAMATH_GPT_fourth_person_height_is_82_l484_48407


namespace NUMINAMATH_GPT_solve_system_1_solve_system_2_l484_48480

theorem solve_system_1 (x y : ℤ) (h1 : y = 2 * x - 3) (h2 : 3 * x + 2 * y = 8) : x = 2 ∧ y = 1 :=
by {
  sorry
}

theorem solve_system_2 (x y : ℤ) (h1 : 2 * x + 3 * y = 7) (h2 : 3 * x - 2 * y = 4) : x = 2 ∧ y = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_system_1_solve_system_2_l484_48480


namespace NUMINAMATH_GPT_last_two_digits_of_7_pow_5_pow_6_l484_48403

theorem last_two_digits_of_7_pow_5_pow_6 : (7 ^ (5 ^ 6)) % 100 = 7 := 
  sorry

end NUMINAMATH_GPT_last_two_digits_of_7_pow_5_pow_6_l484_48403


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l484_48476

-- Definitions
def line_eq1 (x y : ℝ) : Prop := 3 * x - 4 * y - 12 = 0
def line_eq2 (x y : ℝ) : Prop := 6 * x - 8 * y + 11 = 0

-- Statement of the problem
theorem distance_between_parallel_lines :
  (∀ x y : ℝ, line_eq1 x y ↔ line_eq2 x y) →
  (∃ d : ℝ, d = 7 / 2) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l484_48476


namespace NUMINAMATH_GPT_singers_in_choir_l484_48478

variable (X : ℕ)

/-- In the first verse, only half of the total singers sang -/ 
def first_verse_not_singing (X : ℕ) : ℕ := X / 2

/-- In the second verse, a third of the remaining singers joined in -/
def second_verse_joining (X : ℕ) : ℕ := (X / 2) / 3

/-- In the final third verse, 10 people joined so that the whole choir sang together -/
def remaining_singers_after_second_verse (X : ℕ) : ℕ := first_verse_not_singing X - second_verse_joining X

def final_verse_joining_condition (X : ℕ) : Prop := remaining_singers_after_second_verse X = 10

theorem singers_in_choir : ∃ (X : ℕ), final_verse_joining_condition X ∧ X = 30 :=
by
  sorry

end NUMINAMATH_GPT_singers_in_choir_l484_48478


namespace NUMINAMATH_GPT_find_other_number_l484_48426

theorem find_other_number (a b : ℕ) (h1 : (a + b) / 2 = 7) (h2 : a = 5) : b = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l484_48426


namespace NUMINAMATH_GPT_total_dividends_received_l484_48431

theorem total_dividends_received
  (investment : ℝ)
  (share_price : ℝ)
  (nominal_value : ℝ)
  (dividend_rate_year1 : ℝ)
  (dividend_rate_year2 : ℝ)
  (dividend_rate_year3 : ℝ)
  (num_shares : ℝ)
  (total_dividends : ℝ) :
  investment = 14400 →
  share_price = 120 →
  nominal_value = 100 →
  dividend_rate_year1 = 0.07 →
  dividend_rate_year2 = 0.09 →
  dividend_rate_year3 = 0.06 →
  num_shares = investment / share_price → 
  total_dividends = (dividend_rate_year1 * nominal_value * num_shares) +
                    (dividend_rate_year2 * nominal_value * num_shares) +
                    (dividend_rate_year3 * nominal_value * num_shares) →
  total_dividends = 2640 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_total_dividends_received_l484_48431


namespace NUMINAMATH_GPT_rectangle_ratio_l484_48410

theorem rectangle_ratio {l w : ℕ} (h_w : w = 5) (h_A : 50 = l * w) : l / w = 2 := by 
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l484_48410


namespace NUMINAMATH_GPT_probability_same_color_two_dice_l484_48432

theorem probability_same_color_two_dice :
  let total_sides : ℕ := 30
  let maroon_sides : ℕ := 5
  let teal_sides : ℕ := 10
  let cyan_sides : ℕ := 12
  let sparkly_sides : ℕ := 3
  (maroon_sides / total_sides)^2 + (teal_sides / total_sides)^2 + (cyan_sides / total_sides)^2 + (sparkly_sides / total_sides)^2 = 139 / 450 :=
by
  sorry

end NUMINAMATH_GPT_probability_same_color_two_dice_l484_48432


namespace NUMINAMATH_GPT_inverse_of_true_implies_negation_true_l484_48428

variable (P : Prop)
theorem inverse_of_true_implies_negation_true (h : ¬ P) : ¬ P :=
by 
  exact h

end NUMINAMATH_GPT_inverse_of_true_implies_negation_true_l484_48428


namespace NUMINAMATH_GPT_liangliang_distance_to_school_l484_48463

theorem liangliang_distance_to_school :
  (∀ (t : ℕ), (40 * t = 50 * (t - 5)) → (40 * 25 = 1000)) :=
sorry

end NUMINAMATH_GPT_liangliang_distance_to_school_l484_48463


namespace NUMINAMATH_GPT_count_triples_l484_48490

open Set

theorem count_triples 
  (A B C : Set ℕ) 
  (h_union : A ∪ B ∪ C = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
  (h_inter : A ∩ B ∩ C = ∅) :
  (∃ n : ℕ, n = 60466176) :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_count_triples_l484_48490


namespace NUMINAMATH_GPT_smallest_f1_value_l484_48413

noncomputable def polynomial := 
  fun (f : ℝ → ℝ) (r s : ℝ) => 
    f = λ x => (x - r) * (x - s) * (x - ((r + s)/2))

def distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ (r s : ℝ), r ≠ s ∧ polynomial f r s ∧ 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (f ∘ f) a = 0 ∧ (f ∘ f) b = 0 ∧ (f ∘ f) c = 0)

theorem smallest_f1_value
  (f : ℝ → ℝ)
  (hf : distinct_real_roots f) :
  ∃ r s : ℝ, r ≠ s ∧ f 1 = 3/8 :=
sorry

end NUMINAMATH_GPT_smallest_f1_value_l484_48413


namespace NUMINAMATH_GPT_percentage_increase_l484_48406

theorem percentage_increase (original_price new_price : ℝ) (h₁ : original_price = 300) (h₂ : new_price = 480) :
  ((new_price - original_price) / original_price) * 100 = 60 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_increase_l484_48406


namespace NUMINAMATH_GPT_parts_drawn_l484_48436

-- Given that a sample of 30 parts is drawn and each part has a 25% chance of being drawn,
-- prove that the total number of parts N is 120.

theorem parts_drawn (N : ℕ) (h : (30 : ℚ) / N = 0.25) : N = 120 :=
sorry

end NUMINAMATH_GPT_parts_drawn_l484_48436


namespace NUMINAMATH_GPT_find_other_integer_l484_48404

theorem find_other_integer (x y : ℤ) (h1 : 4 * x + 3 * y = 150) (h2 : x = 15 ∨ y = 15) : y = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_other_integer_l484_48404


namespace NUMINAMATH_GPT_find_number_l484_48457

-- Define the conditions.
def condition (x : ℚ) : Prop := x - (1 / 3) * x = 16 / 3

-- Define the theorem from the translated (question, conditions, correct answer) tuple
theorem find_number : ∃ x : ℚ, condition x ∧ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l484_48457


namespace NUMINAMATH_GPT_yarn_length_proof_l484_48423

def green_length := 156
def total_length := 632

noncomputable def red_length (x : ℕ) := green_length * x + 8

theorem yarn_length_proof (x : ℕ) (green_length_eq : green_length = 156)
  (total_length_eq : green_length + red_length x = 632) : x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_yarn_length_proof_l484_48423


namespace NUMINAMATH_GPT_range_of_a_l484_48444

-- Define the negation of the original proposition as a function
def negated_prop (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 4 * x + a > 0

-- State the theorem to be proven
theorem range_of_a (a : ℝ) (h : ¬∃ x : ℝ, a * x^2 + 4 * x + a ≤ 0) : a > 2 :=
  by
  -- Using the assumption to conclude the negated proposition holds
  let h_neg : negated_prop a := sorry
  
  -- Prove the range of a based on h_neg
  sorry

end NUMINAMATH_GPT_range_of_a_l484_48444


namespace NUMINAMATH_GPT_parabola_vertex_above_x_axis_l484_48401

theorem parabola_vertex_above_x_axis (k : ℝ) (h : k > 9 / 4) : 
  ∃ y : ℝ, ∀ x : ℝ, y = (x - 3 / 2) ^ 2 + k - 9 / 4 ∧ y > 0 := 
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_above_x_axis_l484_48401


namespace NUMINAMATH_GPT_slope_intercept_form_of_line_l484_48417

theorem slope_intercept_form_of_line :
  ∀ (x y : ℝ), (∀ (a b : ℝ), (a, b) = (0, 4) ∨ (a, b) = (3, 0) → y = - (4 / 3) * x + 4) := 
by
  sorry

end NUMINAMATH_GPT_slope_intercept_form_of_line_l484_48417


namespace NUMINAMATH_GPT_water_tank_capacity_l484_48460

theorem water_tank_capacity
  (tank_capacity : ℝ)
  (h : 0.30 * tank_capacity = 0.90 * tank_capacity - 54) :
  tank_capacity = 90 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_water_tank_capacity_l484_48460


namespace NUMINAMATH_GPT_base_subtraction_problem_l484_48479

theorem base_subtraction_problem (b : ℕ) (C_b : ℕ) (hC : C_b = 12) : 
  b = 15 :=
by
  sorry

end NUMINAMATH_GPT_base_subtraction_problem_l484_48479


namespace NUMINAMATH_GPT_meeting_time_eqn_l484_48405

-- Mathematical definitions derived from conditions:
def distance := 270 -- Cities A and B are 270 kilometers apart.
def speed_fast_train := 120 -- Speed of the fast train is 120 km/h.
def speed_slow_train := 75 -- Speed of the slow train is 75 km/h.
def time_head_start := 1 -- Slow train departs 1 hour before the fast train.

-- Let x be the number of hours it takes for the two trains to meet after the fast train departs
def x : Real := sorry

-- Proving the equation representing the situation:
theorem meeting_time_eqn : 75 * 1 + (120 + 75) * x = 270 :=
by
  sorry

end NUMINAMATH_GPT_meeting_time_eqn_l484_48405


namespace NUMINAMATH_GPT_number_of_students_l484_48465

variable (F S J R T : ℕ)

axiom freshman_more_than_junior : F = (5 * J) / 4
axiom sophomore_fewer_than_freshman : S = 9 * F / 10
axiom total_students : T = F + S + J + R
axiom seniors_total : R = T / 5
axiom given_sophomores : S = 144

theorem number_of_students (T : ℕ) : T = 540 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_students_l484_48465
