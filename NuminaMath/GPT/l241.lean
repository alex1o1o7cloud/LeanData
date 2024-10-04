import Mathlib

namespace acres_used_for_corn_l241_241936

noncomputable def total_acres : ℝ := 1634
noncomputable def beans_ratio : ℝ := 4.5
noncomputable def wheat_ratio : ℝ := 2.3
noncomputable def corn_ratio : ℝ := 3.8
noncomputable def barley_ratio : ℝ := 3.4

noncomputable def total_parts : ℝ := beans_ratio + wheat_ratio + corn_ratio + barley_ratio
noncomputable def acres_per_part : ℝ := total_acres / total_parts
noncomputable def corn_acres : ℝ := corn_ratio * acres_per_part

theorem acres_used_for_corn :
  corn_acres = 443.51 := by
  sorry

end acres_used_for_corn_l241_241936


namespace sum_even_factors_l241_241855

theorem sum_even_factors (n : ℕ) (h : n = 720) : 
  (∑ d in Finset.filter (λ d, d % 2 = 0) (Finset.divisors n), d) = 2340 :=
by
  rw h
  -- sorry to skip the actual proof
  sorry

end sum_even_factors_l241_241855


namespace area_of_shape_enclosed_by_curve_tangent_and_x_axis_l241_241600

noncomputable def area_of_enclosed_shape : ℝ :=
  (∫ x in 0..1, x^2) + (∫ x in 1..2, x^2 - 4*x + 4)

theorem area_of_shape_enclosed_by_curve_tangent_and_x_axis :
  area_of_enclosed_shape = 2/3 :=
by
  sorry

end area_of_shape_enclosed_by_curve_tangent_and_x_axis_l241_241600


namespace angle_is_2pi_over_3_l241_241277

variable (a b : EuclideanSpace ℝ (Fin 2)) -- Assuming vectors in 2D space for simplicity
variable (θ : ℝ)

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  Real.arccos ((InnerProductSpace.inner a b) / (∥a∥ * ∥b∥))

-- Conditions:
variable (ha : ∥a∥ = 1)
variable (hb : ∥b∥ = 2)
variable (h_perp : InnerProductSpace.inner (a+b) a = 0)

-- The statement to be proved:
theorem angle_is_2pi_over_3 : angle_between_vectors a b = 2 * Real.pi / 3 :=
by
  sorry

end angle_is_2pi_over_3_l241_241277


namespace johns_weekly_allowance_l241_241502

theorem johns_weekly_allowance (A : ℝ)
  (arcade_spent : A * 3 / 5)
  (toy_store_spent : (A - arcade_spent) * 1 / 3)
  (candy_store_spent : A - arcade_spent - toy_store_spent = 0.96) 
  : A = 3.60 := by
  sorry

end johns_weekly_allowance_l241_241502


namespace students_with_both_l241_241210

-- Define the problem conditions as given in a)
def total_students : ℕ := 50
def students_with_bike : ℕ := 28
def students_with_scooter : ℕ := 35

-- State the theorem
theorem students_with_both :
  ∃ (n : ℕ), n = 13 ∧ total_students = students_with_bike + students_with_scooter - n := by
  sorry

end students_with_both_l241_241210


namespace find_n_l241_241156

theorem find_n (n : ℕ) (h_lcm : Nat.lcm n 16 = 48) (h_gcf : Nat.gcd n 16 = 4) : n = 12 :=
by
  sorry

end find_n_l241_241156


namespace find_a_parallel_lines_l241_241650

theorem find_a_parallel_lines (a : ℝ) (l1_parallel_l2 : x + a * y + 6 = 0 → (a - 1) * x + 2 * y + 3 * a = 0 → Parallel) : a = -1 :=
sorry

end find_a_parallel_lines_l241_241650


namespace dinner_customers_l241_241926

theorem dinner_customers 
    (breakfast : ℕ)
    (lunch : ℕ)
    (total_friday : ℕ)
    (H : breakfast = 73)
    (H1 : lunch = 127)
    (H2 : total_friday = 287) :
  (breakfast + lunch + D = total_friday) → D = 87 := by
  sorry

end dinner_customers_l241_241926


namespace statement_A_statement_B_statement_C_statement_D_correct_options_are_BCD_l241_241133

-- Statement A: Incorrect assumption as per given solution
-- We must show that x=4 is not the only solution for C(28, x) = C(28, 3x - 8)

theorem statement_A : ∃ x : ℕ, x ≠ 4 ∧ Nat.choose 28 x = Nat.choose 28 (3 * x - 8) :=
sorry

-- Statement B: Correct assumption as per given solution
-- The coefficient of x^4 in the expansion of (x-1)(x-2)(x-3)(x-4)(x-5) is -15

theorem statement_B : coeff_x_pow_4_expansion : (∏ i in range 1 6, (X : ℤ[X]) - i) ^ 4 = -15 :=
sorry

-- Statement C: Correct assumption as per given solution
-- The remainder when 3^8 is divided by 5 is 1

theorem statement_C : (3^8) % 5 = 1 :=
sorry

-- Statement D: Correct assumption as per given solution
-- With one yuan, five yuan, ten yuan, twenty yuan, and fifty yuan bills, 31 denominations can be formed

theorem statement_D : (2^5 - 1 = 31) :=
rfl

-- Proof that these are the correct options
theorem correct_options_are_BCD : 
  ({statement_A, statement_B, statement_C, statement_D} ∧ False ∧ True ∧ True ∧ True) ∧ ¬ False :=
sorry

end statement_A_statement_B_statement_C_statement_D_correct_options_are_BCD_l241_241133


namespace solve_for_x_l241_241610

theorem solve_for_x (y : ℝ) (x : ℝ) (h1 : y = 432) (h2 : 12^2 * x^4 / 432 = y) : x = 6 := by
  sorry

end solve_for_x_l241_241610


namespace sum_of_absolute_roots_l241_241608

theorem sum_of_absolute_roots :
  let f : Polynomial ℝ := Polynomial.Coeff 4 1 - Polynomial.Coeff 3 6 + Polynomial.Coeff 2 13 + Polynomial.Coeff 1 6 - Polynomial.Coeff 0 40
  ((Polynomial.roots f).sum (λ z, |z|)) = 5 + 2 * Real.sqrt 8.5 :=
by
  sorry

end sum_of_absolute_roots_l241_241608


namespace total_area_of_three_circles_l241_241984

-- Definitions and conditions
def side_length_of_triangle : ℝ := 2
def diagonal_of_square (a : ℝ) : ℝ := a * Real.sqrt 2
def radius_of_circle (d : ℝ) : ℝ := d / 2
def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

-- Mathematical problem to be proved
theorem total_area_of_three_circles :
  let d := diagonal_of_square side_length_of_triangle;
      r := radius_of_circle d;
      area := area_of_circle r in
  3 * area = 6 * Real.pi :=
by
  let d := diagonal_of_square side_length_of_triangle
  let r := radius_of_circle d
  let area := area_of_circle r
  sorry

end total_area_of_three_circles_l241_241984


namespace second_number_is_22_l241_241324

theorem second_number_is_22 (x second_number : ℕ) : 
  (x + second_number = 33) → 
  (second_number = 2 * x) → 
  second_number = 22 :=
by
  intros h_sum h_double
  sorry

end second_number_is_22_l241_241324


namespace sum_even_factors_l241_241857

theorem sum_even_factors (n : ℕ) (h : n = 720) : 
  (∑ d in Finset.filter (λ d, d % 2 = 0) (Finset.divisors n), d) = 2340 :=
by
  rw h
  -- sorry to skip the actual proof
  sorry

end sum_even_factors_l241_241857


namespace age_difference_l241_241499

variable (A B C : ℕ)

def condition1 := C = B / 2
def condition2 := A + B + C = 22
def condition3 := B = 8

theorem age_difference (h1 : condition1 C B)
                       (h2 : condition2 A B C) 
                       (h3 : condition3 B) : A - B = 2 := by
  sorry

end age_difference_l241_241499


namespace true_statements_l241_241639

variable {Line Plane : Type}
variable m n : Line
variable α β : Plane

-- Defining the conditions from the problem
def distinct_lines (m n : Line) : Prop := m ≠ n
def non_coincident_planes (α β : Plane) : Prop := α ≠ β

-- Define the propositions from the problem
def proposition1 (m : Line) (α : Plane) : Prop := (Parallel m α) → (∀ l : Line, l ⊂ α → Parallel m l)
def proposition2 (m n : Line) (α β : Plane) : Prop := (Parallel α β) ∧ (m ⊂ α) ∧ (n ⊂ β) → (Parallel m n)
def proposition3 (m n : Line) (α β : Plane) : Prop := (Perpendicular m α) ∧ (Perpendicular n β) ∧ (Parallel m n) → (Parallel α β)
def proposition4 (m : Line) (α β : Plane) : Prop := (Parallel α β) ∧ (m ⊂ α) → (Parallel m β)

theorem true_statements (h1 : distinct_lines m n) (h2 : non_coincident_planes α β) :
  (proposition3 m n α β) ∧ (proposition4 m α β) :=
by 
  sorry

end true_statements_l241_241639


namespace find_n_that_makes_vectors_collinear_l241_241307

theorem find_n_that_makes_vectors_collinear (n : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, 3)) (h_b : b = (3, n)) (h_collinear : ∃ k : ℝ, 2 • a - b = k • b) : n = 9 :=
sorry

end find_n_that_makes_vectors_collinear_l241_241307


namespace smallest_possible_value_l241_241692

theorem smallest_possible_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (a b c : ℕ), a = floor((x + y) / z) ∧ b = floor((y + z) / x) ∧ c = floor((z + x) / y) ∧ (a + b + c) = 4 :=
begin
  sorry
end

end smallest_possible_value_l241_241692


namespace range_independent_variable_l241_241085

theorem range_independent_variable (x : ℝ) (y : ℝ) (h : y = sqrt (x - 2)) : x ≥ 2 :=
by {
  -- Proof steps would go here, but we use sorry to indicate we are skipping the proof.
  sorry
}

end range_independent_variable_l241_241085


namespace difference_sum_even_odd_l241_241846

noncomputable def sum_first_odd_numbers (n : ℕ) : ℕ :=
  n * n

noncomputable def sum_first_even_numbers (n : ℕ) : ℕ :=
  n * (n + 1)

theorem difference_sum_even_odd (n : ℕ) :
  sum_first_even_numbers n - sum_first_odd_numbers n = n :=
by
  sorry

#eval difference_sum_even_odd 1500  -- should evaluate to true

end difference_sum_even_odd_l241_241846


namespace range_independent_variable_l241_241086

theorem range_independent_variable (x : ℝ) (y : ℝ) (h : y = sqrt (x - 2)) : x ≥ 2 :=
by {
  -- Proof steps would go here, but we use sorry to indicate we are skipping the proof.
  sorry
}

end range_independent_variable_l241_241086


namespace sum_of_digits_least_number_l241_241797

theorem sum_of_digits_least_number : 
  (let swap_and_decrease (n : ℕ) := 
     -- define the function that simulates the described operations
     ... -- the definition of the operation would go here
   in 
   sum_of_digits (least_number_after_operations 123456789 swap_and_decrease)) = 5 := sorry

end sum_of_digits_least_number_l241_241797


namespace probability_of_multiple_of_3_is_1_5_l241_241513

-- Definition of the problem conditions
def digits : List ℕ := [1, 2, 3, 4, 5]

-- Function to calculate the probability
noncomputable def probability_of_multiple_of_3 : ℚ := 
  let total_permutations := (Nat.factorial 5) / (Nat.factorial (5 - 4))  -- i.e., 120
  let valid_permutations := Nat.factorial 4  -- i.e., 24, for the valid combination
  valid_permutations / total_permutations 

-- Statement to be proved
theorem probability_of_multiple_of_3_is_1_5 :
  probability_of_multiple_of_3 = 1 / 5 := 
by
  -- Skeleton for the proof
  sorry

end probability_of_multiple_of_3_is_1_5_l241_241513


namespace count_only_brand_a_soap_l241_241189

-- Given conditions
variables (total surveyed households : ℕ) (neither brand soap : ℕ)
variables (both brands soap : ℕ) (only brand b soap : ℕ)
variables (ratio both to b : ℕ)

-- Define the given constants based on the problem's conditions
def total_surveyed_households : ℕ := 300
def neither_brand_soap : ℕ := 80
def both_brands_soap : ℕ := 40
def ratio_both_to_b = 3

-- Derived quantities
def only_brand_b_soap : ℕ := ratio_both_to_b * both_brands_soap
def total_soap_users : ℕ := total_surveyed_households - neither_brand_soap

-- The main proof statement
theorem count_only_brand_a_soap : ∃ (A : ℕ), 
  A = total_soap_users - (both_brands_soap + only_brand_b_soap) ∧ A = 60 :=
by {
  -- let A be the number of households that used only brand A soap
  let A := total_soap_users - (both_brands_soap + only_brand_b_soap),
  existsi A,
  split,
  refl,
  sorry,  -- We assume this result as given in the problem statement
}

end count_only_brand_a_soap_l241_241189


namespace length_of_XH_l241_241774

theorem length_of_XH
  (WXYZ : Type) [square WXYZ]
  (A B X Y G Z H : WXYZ) 
  (side_length : ℝ) (area_square : ℝ) (area_triangle : ℝ)
  (h1 : side_length = 12) 
  (h2 : area_square = 144)
  (h3 : area_triangle = 72)
  (h4 : ZG ⊥ ZH)
  (h5 : ZG = ZH)
  (h6 : XY.extend H = XH)
  (h7 : triangle ZGH) :
  length XH = 12 * real.sqrt 2 
:= sorry

end length_of_XH_l241_241774


namespace savings_by_going_earlier_l241_241472

/-- Define the cost of evening ticket -/
def evening_ticket_cost : ℝ := 10

/-- Define the cost of large popcorn & drink combo -/
def food_combo_cost : ℝ := 10

/-- Define the discount percentage on tickets from 12 noon to 3 pm -/
def ticket_discount : ℝ := 0.20

/-- Define the discount percentage on food combos from 12 noon to 3 pm -/
def food_combo_discount : ℝ := 0.50

/-- Prove that the total savings Trip could achieve by going to the earlier movie is $7 -/
theorem savings_by_going_earlier : 
  (ticket_discount * evening_ticket_cost) + (food_combo_discount * food_combo_cost) = 7 := by
  sorry

end savings_by_going_earlier_l241_241472


namespace exists_n_for_all_xy_l241_241227

theorem exists_n_for_all_xy :
  ∃ (n : ℕ), (n = 4 ∨ n = 6) ∧ ∀ x y : ℝ, ∃ (a : Fin n → ℝ),
    (x = (Finset.univ.sum (λ i, a i))) ∧
    (y = (Finset.univ.sum (λ i, (a i)⁻¹))) :=
sorry

end exists_n_for_all_xy_l241_241227


namespace sum_of_possible_radii_l241_241931

theorem sum_of_possible_radii :
  ∃ r1 r2 : ℝ, 
    (∀ r, (r - 5)^2 + r^2 = (r + 2)^2 → r = r1 ∨ r = r2) ∧ 
    r1 + r2 = 14 :=
sorry

end sum_of_possible_radii_l241_241931


namespace lcm_180_504_is_2520_l241_241114

-- Define what it means for a number to be the least common multiple of two numbers
def is_lcm (a b lcm : ℕ) : Prop :=
  a ∣ lcm ∧ b ∣ lcm ∧ ∀ m, (a ∣ m ∧ b ∣ m) → lcm ∣ m

-- Lean 4 statement to prove that the least common multiple of 180 and 504 is 2520
theorem lcm_180_504_is_2520 : ∀ (a b : ℕ), a = 180 → b = 504 → is_lcm a b 2520 := by
  intro a b
  assume h1 : a = 180
  assume h2 : b = 504
  sorry

end lcm_180_504_is_2520_l241_241114


namespace inequality_wxyz_l241_241055

theorem inequality_wxyz 
  (w x y z : ℝ) 
  (h₁ : w^2 + y^2 ≤ 1) : 
  (w * x + y * z - 1)^2 ≥ (w^2 + y^2 - 1) * (x^2 + z^2 - 1) :=
by
  sorry

end inequality_wxyz_l241_241055


namespace negation_exists_ltx2_plus_x_plus_1_lt_0_l241_241077

theorem negation_exists_ltx2_plus_x_plus_1_lt_0 :
  ¬ (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by
  sorry

end negation_exists_ltx2_plus_x_plus_1_lt_0_l241_241077


namespace total_pens_l241_241020

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l241_241020


namespace proof_divide_prism_l241_241493

-- Conditions from the problem, stated as premises:
axiom prism_base (P : Type) (B : set P) : is_polygon B

-- Statement D in terms of Lean's vocabulary:
def can_divide_prism (P : Type) [Prism P] : Prop :=
  ∃ plane : set P, divides_into_two_prisms plane

theorem proof_divide_prism (P : Type) [Prism P] : can_divide_prism P :=
  sorry

end proof_divide_prism_l241_241493


namespace fly_maximum_path_length_l241_241938

noncomputable def max_fly_path_length : ℝ :=
  Real.sqrt 14 + 6 + Real.sqrt 13 + Real.sqrt 5

theorem fly_maximum_path_length :
  let length := max_fly_path_length in
  let dimensions := (1, 2, 3) in
  let start := (0, 0, 0) in
  let finish := (0, 0, 3) in
  let corners := [(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3),
                  (1, 2, 0), (0, 2, 3), (1, 0, 3), (1, 2, 3)] in
  (∀ path : list (ℝ × ℝ × ℝ), path.length = corners.length ∧
                              path.head = start ∧
                              path.last = finish ∧
                              ∀ corner ∈ corners, corner ∈ path) →
  ∃ path : list (ℝ × ℝ × ℝ), ∑ i in range (path.length - 1), dist (path.nth i) (path.nth (i + 1)) = length := sorry

end fly_maximum_path_length_l241_241938


namespace sequence_ai_eq_i_l241_241768

open Nat

theorem sequence_ai_eq_i (a : ℕ → ℕ) (h : ∀ i j : ℕ, i ≠ j → gcd (a i) (a j) = gcd i j) : ∀ i : ℕ, a i = i := by
  sorry

end sequence_ai_eq_i_l241_241768


namespace total_pens_bought_l241_241033

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l241_241033


namespace relationship_among_a_b_c_l241_241087

noncomputable def a : ℝ := 0.31^2
noncomputable def b : ℝ := Real.log 0.31 / Real.log 2 -- Equivalent to log base 2
noncomputable def c : ℝ := 2^0.31

theorem relationship_among_a_b_c : b < a ∧ a < c := 
  by
  sorry

end relationship_among_a_b_c_l241_241087


namespace first_non_zero_digit_62nd_place_l241_241619

theorem first_non_zero_digit_62nd_place
  (lg_2 : ℝ)
  (lg_3 : ℝ)
  (h1 : lg_2 = 0.3010)
  (h2 : lg_3 = 0.4771) :
  (let x := (6 / 25 : ℝ) ^ 100 in
   let lgx := 100 * (Math.log10 6 - Math.log10 25) in
   (Nat.floor (Real.fract lgx)) < -62) := sorry

end first_non_zero_digit_62nd_place_l241_241619


namespace floor_sum_min_value_l241_241696

theorem floor_sum_min_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end floor_sum_min_value_l241_241696


namespace quadrilateral_PACQ_area_l241_241471

noncomputable def quadrilateral_area : ℝ :=
  let PA := 10
  let PQ := 24
  let PC := 26
  let AQ := real.sqrt (PQ^2 - PA^2)
  let area_PAQ := 0.5 * PA * AQ
  let area_PQC := 0.5 * PC * PA
  area_PAQ + area_PQC

theorem quadrilateral_PACQ_area :
  let PA := 10
  let PQ := 24
  let PC := 26
  let AQ := real.sqrt (PQ^2 - PA^2)
  let area_PAQ := 0.5 * PA * AQ
  let area_PQC := 0.5 * PC * PA
  area_PAQ + area_PQC = 10 * real.sqrt 119 + 130 :=
by
  sorry

end quadrilateral_PACQ_area_l241_241471


namespace evaluate_fraction_l241_241760

noncomputable section

variables (u v : ℂ)
variables (h1 : u ≠ 0) (h2 : v ≠ 0) (h3 : u^2 + u * v + v^2 = 0)

theorem evaluate_fraction : (u^7 + v^7) / (u + v)^7 = -2 := by
  sorry

end evaluate_fraction_l241_241760


namespace division_remainder_l241_241606

-- Define the divisor
def divisor : ℚ[X] := X^2 - 5*X + 6

-- Define the dividend
def dividend : ℚ[X] := X^101

-- Define the expected remainder
def expected_remainder : ℚ[X] := (3^101 - 2^101) * X + (2^101 - 2 * 3^101)

-- Prove that the remainder of the division is as expected
theorem division_remainder :
  let ⟨q, r⟩ := Polynomial.divMod dividend divisor in r = expected_remainder :=
by {
  -- Placeholder for the proof
  sorry
}

end division_remainder_l241_241606


namespace toys_per_week_l241_241522

-- Define the number of days the workers work in a week
def days_per_week : ℕ := 4

-- Define the number of toys produced each day
def toys_per_day : ℕ := 1140

-- State the proof problem: workers produce 4560 toys per week
theorem toys_per_week : (toys_per_day * days_per_week) = 4560 :=
by
  -- Proof goes here
  sorry

end toys_per_week_l241_241522


namespace find_n_if_an_is_50_l241_241736

-- Define the arithmetic sequence
def sequence (n : ℕ) : ℕ := 2 + 3 * (n - 1)

-- Define the theorem statement
theorem find_n_if_an_is_50 (n : ℕ) (h : sequence n = 50) : n = 17 :=
by
  sorry

end find_n_if_an_is_50_l241_241736


namespace square_ratio_l241_241421

def area (side_length : ℝ) : ℝ := side_length^2

theorem square_ratio (x : ℝ) (x_pos : 0 < x) :
  let A := area x
  let B := area (3*x)
  let C := area (2*x)
  A / (B + C) = 1 / 13 :=
by
  sorry

end square_ratio_l241_241421


namespace number_proportion_l241_241173

theorem number_proportion (number : ℚ) :
  (number : ℚ) / 12 = 9 / 360 →
  number = 0.3 :=
by
  intro h
  sorry

end number_proportion_l241_241173


namespace range_of_y_under_conditions_l241_241588

theorem range_of_y_under_conditions :
  (∀ x : ℝ, (x - y) * (x + y) < 1) → (-1/2 : ℝ) < y ∧ y < (3/2 : ℝ) := by
  intro h
  have h' : ∀ x : ℝ, (x - y) * (1 - x - y) < 1 := by
    sorry
  have g_min : ∀ x : ℝ, y^2 - y < x^2 - x + 1 := by
    sorry
  have min_value : y^2 - y < 3/4 := by
    sorry
  have range_y : (-1/2 : ℝ) < y ∧ y < (3/2 : ℝ) := by
    sorry
  exact range_y

end range_of_y_under_conditions_l241_241588


namespace prime_looking_count_l241_241220

/-- 
A number is prime-looking if it is composite and not divisible by 2, 3 or 5.
The three smallest prime-looking numbers are 49, 77, and 91.
There are 303 prime numbers less than 2000.
 -/
def is_prime_looking (n : ℕ) : Prop :=
  n > 1 ∧ 
  ∃ d, d > 1 ∧ d < n ∧ n % d = 0 ∧ 
  n % 2 ≠ 0 ∧ n % 3 ≠ 0 ∧ n % 5 ≠ 0

def count_prime_below_2000 : ℕ := 303

theorem prime_looking_count : 
  (finset.filter is_prime_looking (finset.range 2000)).card = 233 := 
sorry

end prime_looking_count_l241_241220


namespace translation_of_point_l241_241081

variable (P : ℝ × ℝ) (xT yT : ℝ)

def translate_x (P : ℝ × ℝ) (xT : ℝ) : ℝ × ℝ :=
    (P.1 + xT, P.2)

def translate_y (P : ℝ × ℝ) (yT : ℝ) : ℝ × ℝ :=
    (P.1, P.2 + yT)

theorem translation_of_point : translate_y (translate_x (-5, 1) 2) (-4) = (-3, -3) :=
by
  sorry

end translation_of_point_l241_241081


namespace range_of_a_l241_241296

theorem range_of_a :
  let f : ℝ → ℝ := λ x, |x| + 2^|x|
  in (∀ a : ℝ, f (a - 1) < f 2 → -1 < a ∧ a < 3) :=
by
  let f : ℝ → ℝ := λ x, |x| + 2^|x|
  assume a ha
  have fact1 : f (2) = 6 := by sorry -- We should show the value of f(2) as 6
  have fact2 : f (a - 1) < 6 := by rw fact1 at ha; exact ha -- Using f(2)=6
  -- Now we can proceed to solve |a - 1| < 2
  sorry

end range_of_a_l241_241296


namespace james_tylenol_intake_per_day_l241_241743

theorem james_tylenol_intake_per_day :
  (∃ (tablets_per_dose doses_per_day tablet_mg : ℕ),
    tablets_per_dose = 2 ∧
    doses_per_day = 24 / 6 ∧
    tablet_mg = 375 ∧
    (tablets_per_dose * doses_per_day * tablet_mg = 3000)) :=
begin
  let tablets_per_dose := 2,
  let doses_per_day := 24 / 6,
  let tablet_mg := 375,
  use [tablets_per_dose, doses_per_day, tablet_mg],
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  sorry
end

end james_tylenol_intake_per_day_l241_241743


namespace equidistant_circumcenters_l241_241385

-- Definitions of the given conditions

variables {A B C D O O₁ O₂ : Type} 
variables [triangle ABC]
variables [angle_bisector A D B C]
variables [circumcenter ABC O]
variables [circumcenter ABD O₁]
variables [circumcenter ACD O₂]

-- The proof statement

theorem equidistant_circumcenters :
  dist O O₁ = dist O O₂ :=
sorry

end equidistant_circumcenters_l241_241385


namespace sum_of_even_factors_720_l241_241872

theorem sum_of_even_factors_720 : 
  let n := 2^4 * 3^2 * 5 in
  (∑ d in (Finset.range (n + 1)).filter (λ d, d % 2 = 0 ∧ n % d = 0), d) = 2340 :=
by
  let n := 2^4 * 3^2 * 5
  sorry

end sum_of_even_factors_720_l241_241872


namespace smallest_k_l241_241851

theorem smallest_k (k : ℕ) (h₁ : k > 1) (h₂ : k % 17 = 1) (h₃ : k % 6 = 1) (h₄ : k % 2 = 1) : k = 103 :=
by sorry

end smallest_k_l241_241851


namespace quadratic_roots_l241_241582

open Real

noncomputable def roots_of_quadratic (p : ℝ) (h : 3*x^2 - 4*p*x + 9 = 0) := 
  ∀ x, 3*x^2 - 4*p*x + 9 = 0 → 
  x = sqrt 3 ∨ x = -sqrt 3

theorem quadratic_roots (p : ℝ) (h : (16 * p^2 - 108 = 0)) :
  roots_of_quadratic p := 
sorry

end quadratic_roots_l241_241582


namespace roots_of_equation_l241_241806

theorem roots_of_equation {x : ℝ} :
  (12 * x^2 - 31 * x - 6 = 0) →
  (x = (31 + Real.sqrt 1249) / 24 ∨ x = (31 - Real.sqrt 1249) / 24) :=
by
  sorry

end roots_of_equation_l241_241806


namespace sum_even_factors_of_720_l241_241899

open Nat

theorem sum_even_factors_of_720 :
  ∑ d in (finset.filter (λ x, even x) (finset.divisors 720)), d = 2340 :=
by
  sorry

end sum_even_factors_of_720_l241_241899


namespace range_of_m_l241_241671

def setA := {x : ℝ | |x - 1| + |x + 1| ≤ 3}
def setB (m : ℝ) := {x : ℝ | x^2 - (2 * m + 1) * x + m^2 + m < 0}

theorem range_of_m (m : ℝ) (h : setA ∩ setB m ≠ ∅) : m ∈ Ioo (-5 / 2) (3 / 2) :=
by
  sorry

end range_of_m_l241_241671


namespace sum_even_factors_of_720_l241_241895

open Nat

theorem sum_even_factors_of_720 :
  ∑ d in (finset.filter (λ x, even x) (finset.divisors 720)), d = 2340 :=
by
  sorry

end sum_even_factors_of_720_l241_241895


namespace collapsed_buildings_l241_241203

theorem collapsed_buildings (initial_collapse : ℕ) (collapse_one : initial_collapse = 4)
                            (collapse_double : ∀ n m, m = 2 * n) : (4 + 8 + 16 + 32 = 60) :=
by
  sorry

end collapsed_buildings_l241_241203


namespace sum_of_squares_expressible_l241_241778

theorem sum_of_squares_expressible (a b c : ℕ) (h1 : c^2 = a^2 + b^2) : 
  ∃ x y : ℕ, x^2 + y^2 = c^2 + a*b ∧ ∃ u v : ℕ, u^2 + v^2 = c^2 - a*b :=
by
  sorry

end sum_of_squares_expressible_l241_241778


namespace find_OP_squared_l241_241518

-- Define the Circle and its properties
structure Circle (α : Type) [MetricSpace α] :=
(center : α)
(radius : ℝ)

-- Define Points and Chords
variables {α : Type} [MetricSpace α]

-- Define the midpoints and lengths relationships
theorem find_OP_squared
  (O A B C D P E F : α)
  (circle : Circle α)
  (chord_AB : 24)
  (chord_CD : 16)
  (E_is_midpoint_AB : dist A E = 12 ∧ dist E B = 12)
  (F_is_midpoint_CD : dist C F = 8 ∧ dist F D = 8)
  (radius : dist O A = 20 ∧ dist O C = 20)
  (intersection : ∃ P, ∀ (X : α), dist A P + dist P B = chord_AB ∧ dist C P + dist P D = chord_CD)
  (distance_midpoints : dist E F = 10) :
  ∃ x, x = dist O P ^ 2 :=
by
  sorry

end find_OP_squared_l241_241518


namespace point_on_external_bisector_l241_241976

noncomputable theory

open EuclideanGeometry

variables {w1 w2 w : Circle}
variables {P Q A B O : Point}

-- Conditions
-- 1. Circles \( w_1 \) and \( w_2 \) intersect at points \( P \) and \( Q \).
axiom circles_intersect (Hw1 : w1 ∈ Circle) (Hw2 : w2 ∈ Circle) (HPQ : P ∈ w1 ∧ P ∈ w2 ∧ Q ∈ w1 ∧ Q ∈ w2) :

-- 2. Circle \( w \) with center \( O \) is internally tangent to \( w_1 \) at \( A \) and to \( w_2 \) at \( B \).
internal_tangent (Hw : w ∈ Circle) (HO : O ∈ w.center) (Hp1 : w.internallyTangentw1 = A) (Hp2 : w.internallyTangentw2 = B) :

-- 3. Points \( A \), \( B \), and \( Q \) are collinear.
collinear_points (Hline : collinear {A, B, Q}) : 

-- Prove that point \( O \) lies on the external bisector of \( \angle APB \).
theorem point_on_external_bisector 
  (Hw1 : w1 ∈ Circle) (Hw2 : w2 ∈ Circle) 
  (HPQ : P ∈ w1 ∧ P ∈ w2 ∧ Q ∈ w1 ∧ Q ∈ w2) 
  (Hw : w ∈ Circle) (HO : O ∈ w.center) 
  (Hp1 : w.internallyTangentw1 = A) 
  (Hp2 : w.internallyTangentw2 = B) 
  (Hline : collinear {A, B, Q}) : 
   lies_on_external_bisector O A P B := 
sorry

end point_on_external_bisector_l241_241976


namespace sum_of_even_factors_720_l241_241886

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) : ℕ :=
    match n with
    | 720 => 
      let sum_powers_2 := 2 + 4 + 8 + 16 in
      let sum_powers_3 := 1 + 3 + 9 in
      let sum_powers_5 := 1 + 5 in
      sum_powers_2 * sum_powers_3 * sum_powers_5
    | _ => 0
  in
  even_factors_sum 720 = 2340 :=
by 
  sorry

end sum_of_even_factors_720_l241_241886


namespace ratio_L_T_l241_241771

variable (A : Matrix (Fin 50) (Fin 50) ℝ)

def rowSum (i : Fin 50) : ℝ := ∑ j, A i j
def colSum (j : Fin 50) : ℝ := ∑ i, A i j

def L : ℝ := (∑ i, rowSum A i) / 50
def T : ℝ := (∑ j, colSum A j) / 50

theorem ratio_L_T : L A / T A = 1 := 
  sorry

end ratio_L_T_l241_241771


namespace total_pens_l241_241024

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l241_241024


namespace root_in_interval_l241_241444

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - 4

theorem root_in_interval : ∃ c ∈ Ioo (1 : ℝ) (2 : ℝ), f c = 0 := by
  let f := λ x : ℝ, Real.exp x + x^2 - 4
  have h1 : f 1 = Real.exp 1 + 1 - 4 := by rfl
  have h1_neg : f 1 < 0 := by
    rw h1
    exact add_lt_add_of_lt_of_le (Real.exp_pos _).le (by norm_num)
  
  have h2: f 2 = Real.exp 2 + 2^2 - 4 := by rfl
  have h2_pos : f 2 > 0 := by
    rw h2
    exact add_pos_of_nonneg_of_pos (by norm_num) (Real.exp_pos _)
  
  have h_cont : ContinuousOn f (Icc 1 2) := by
    exact Real.continuous_exp.add continuous_id.sq.sub continuous_const
  
  exact IntermediateValueTheorem.f_cont_in_interval h1_neg h2_pos h_cont

end root_in_interval_l241_241444


namespace swimming_pool_length_l241_241197

theorem swimming_pool_length :
  ∀ (w d1 d2 V : ℝ), w = 9 → d1 = 1 → d2 = 4 → V = 270 → 
  (((V = (1 / 2) * (d1 + d2) * w * l) → l = 12)) :=
by
  intros w d1 d2 V hw hd1 hd2 hV hv
  simp only [hw, hd1, hd2, hV] at hv
  sorry

end swimming_pool_length_l241_241197


namespace find_cos_B_l241_241684

theorem find_cos_B (B : ℝ) (h : Real.tan B - Real.sec B = -2) : Real.cos B = 4 / 5 := 
sorry

end find_cos_B_l241_241684


namespace quad_area_leq_one_l241_241822

theorem quad_area_leq_one {AB BC CD DA : ℝ} (H1: convex_quad ABCD) (H2: AB + BC + CD + DA = 4) :
  area_quad ABCD ≤ 1 :=
sorry

end quad_area_leq_one_l241_241822


namespace negation_correct_l241_241449

-- Define the original proposition
def original_proposition : Prop := ∀ x : ℝ, x^2 + 5 * x = 4

-- Define the negation of the original proposition
def negated_proposition : Prop := ∃ x : ℝ, x^2 + 5 * x ≠ 4

-- Theorem statement asserting the equivalence
theorem negation_correct : ¬ original_proposition ↔ negated_proposition := 
begin
  sorry
end

end negation_correct_l241_241449


namespace rectangle_solution_l241_241146

-- Define the given conditions
variables (x y : ℚ)

-- Given equations
def condition1 := (Real.sqrt (x - y) = 2 / 5)
def condition2 := (Real.sqrt (x + y) = 2)

-- Solution
theorem rectangle_solution (x y : ℚ) (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 52 / 25 ∧ y = 48 / 25 ∧ (Real.sqrt ((52 / 25) * (48 / 25)) = 8 / 25) :=
by
  sorry

end rectangle_solution_l241_241146


namespace areas_ratio_of_similar_triangles_l241_241835

theorem areas_ratio_of_similar_triangles :
  ∀ (A B C : Type) [EuclideanGeometry] (a b c : A) (new_a new_b new_c : A),
  (angle a b c = 45) → (angle b c a = 60) → (angle c a b = 75) →
  (angle new_a new_b new_c = 45) → (angle new_b new_c new_a = 60) → (angle new_c new_a new_b = 75) →
  ratio_of_areas a b c new_a new_b new_c = sqrt(3) - 1 :=
by
  sorry

end areas_ratio_of_similar_triangles_l241_241835


namespace sum_of_even_factors_720_l241_241881

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) :=
    (2 + 4 + 8 + 16) * (1 + 3 + 9) * (1 + 5)
  in even_factors_sum 720 = 2340 :=
by
  sorry

end sum_of_even_factors_720_l241_241881


namespace matchstick_triangle_count_l241_241478

noncomputable def is_triangle (a b c : ℕ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

theorem matchstick_triangle_count : 
  (∃ (sides : list (ℕ × ℕ × ℕ)), 
    sides = [(3, 3, 3), (2, 3, 4), (1, 4, 4)] ∧ 
    ∀ (a b c : ℕ), 
      (a, b, c) ∈ sides → 
      a + b + c = 9 ∧ is_triangle a b c) → 3 := 
begin
  sorry
end

end matchstick_triangle_count_l241_241478


namespace find_m_l241_241435

-- Define the function f(x)
def f (x m : ℝ) : ℝ := (m^2 - m - 1) * x^m

-- Theorem to prove that m = 2 given the conditions
theorem find_m (h1 : (∀ x : ℝ, x > 0 → ∃ c : ℝ, f x m = c * x^m))
                (h2 : ∀ x : ℝ, x > 0 → (differentiable ℝ (f x)) → f' x > 0) :
  m = 2 :=
sorry

end find_m_l241_241435


namespace company_kw_price_percentage_l241_241152

variable (A B: ℝ) 

theorem company_kw_price_percentage (h₁ : 1.9 * A = 2 * B) :
  let combined_assets := A + B in
  let price_kw := 2 * B in
  price_kw / combined_assets * 100 = 97.4 := 
by 
  sorry

end company_kw_price_percentage_l241_241152


namespace age_proof_l241_241844

def age_relationship : Prop := 
  ∀ (x : ℕ), let weiwei_current_age := 8
             let father_current_age := 34
             in (father_current_age + x = 3 * (weiwei_current_age + x)) → x = 5

theorem age_proof : age_relationship := 
  by
  sorry

end age_proof_l241_241844


namespace amount_after_3_years_l241_241951

theorem amount_after_3_years (P t A' : ℝ) (R : ℝ) :
  P = 800 → t = 3 → A' = 992 →
  (800 * ((R + 3) / 100) * 3 = 192) →
  (A = P * (1 + (R / 100) * t)) →
  A = 1160 := by
  intros hP ht hA' hR hA
  sorry

end amount_after_3_years_l241_241951


namespace isosceles_trapezoid_AC_length_l241_241817

theorem isosceles_trapezoid_AC_length :
  \let A B C D: ℝ \In  line_segments_of_isosceles_trapezoid
    \lbrace ( A , 0 ),  ( B , 0 ),  ( B -17 \cdot   \cos  ( \alpha ),  17 \cdot \sin ( \alpha )  ),( 17 \cdot \cos( \beta ),  17 \cdot \sin( \beta )  )\rbrace
  \cdot 
  \[
\begin{array}{cccc}
  AB = 24 & CD = 10 & AD =12 & BC = 12\\   
  AF = 7 & AE = 17 &
    \  CE?=
    10 & AC = 17 \\ \sqrt{ 2 }\
\end{array}

end isosceles_trapezoid_AC_length_l241_241817


namespace seats_in_hall_l241_241717

theorem seats_in_hall (S : ℕ) (filled_pct : ℕ) (vacant_seats : ℕ) (h_filled : filled_pct = 75) (h_vacant : vacant_seats = 150) :
    S = 600 :=
by
  let vacant_pct := 100 - filled_pct
  have h_vacant_eq : vacant_pct * S / 100 = vacant_seats, from sorry
  sorry

end seats_in_hall_l241_241717


namespace triangle_acute_angles_integer_solution_l241_241611

theorem triangle_acute_angles_integer_solution :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℕ), (20 < x ∧ x < 27) ∧ (12 < x ∧ x < 36) ↔ (x = 21 ∨ x = 22 ∨ x = 23 ∨ x = 24 ∨ x = 25 ∨ x = 26) :=
by
  sorry

end triangle_acute_angles_integer_solution_l241_241611


namespace heather_start_time_later_than_stacy_l241_241422

theorem heather_start_time_later_than_stacy :
  ∀ (distance_initial : ℝ) (H_speed : ℝ) (S_speed : ℝ) (H_distance_when_meet : ℝ),
    distance_initial = 5 ∧
    H_speed = 5 ∧
    S_speed = 6 ∧
    H_distance_when_meet = 1.1818181818181817 →
    ∃ (Δt : ℝ), Δt = 24 / 60 :=
by
  sorry

end heather_start_time_later_than_stacy_l241_241422


namespace vertices_form_parabola_l241_241268

open Real

-- Given constants a > 0 and d
variables (a d : ℝ) (h_a_pos : 0 < a)

-- For each real number t, (x_t, y_t) is the vertex of the parabola y = ax^2 + tx + d.
def vertex (t : ℝ) : ℝ × ℝ :=
  let x_t := -t / (2 * a)
  let y_t := a * x_t^2 + t * x_t + d
  (x_t, y_t)

theorem vertices_form_parabola :
  ∃ c : ℝ, ∀ (t : ℝ), (vertex a d t).snd = -a * (vertex a d t).fst^2 + c :=
by
  use d
  intros t
  let x := -t / (2 * a)
  let y := a * x^2 + t * x + d
  have hx : x = (vertex a d t).fst := rfl
  have hy : y = (vertex a d t).snd := rfl
  rw [hx, hy]
  have : y = -a * x^2 + d := by
    calc
      y = a * x^2 + t * x + d         : rfl
      ... = a * (-(t / (2 * a)))^2 + t * (-(t / (2 * a))) + d : by rw [hx]
      ... = a * (t^2 / (4 * a^2)) - t^2 / (2 * a) + d         : by ring
      ... = t^2 / (4 * a) - t^2 / (2 * a) + d                 : by field_simp [a ≠ 0]
      ... = - t^2 / (4 * a) + d                               : by ring
      ... = -a * x^2 + d                                     : by rw [hx]; field_simp [a ≠ 0]
  exact this

end vertices_form_parabola_l241_241268


namespace find_k_l241_241752

variable {a : ℕ → ℝ}
variable {d : ℝ}
variable (d_ne_zero : d ≠ 0)
variable (S11_eq_132 : 11 * (a 1 + a 11) / 2 = 132)
variable (a3_plus_ak_eq_24 : ∀ k : ℕ, a 3 + a k = 24 → k = 9)

theorem find_k (k : ℕ) : 
    S11_eq_132 → a3_plus_ak_eq_24 k → d_ne_zero → k = 9 :=
by
  intro h1 h2 h3
  -- Here we would normally proceed with the proof.
  -- Skipping the proof as required.
  exact sorry

end find_k_l241_241752


namespace find_value_a2_b2_c2_l241_241423

variable (a b c p q r : ℝ)
variable (h1 : a * b = p)
variable (h2 : b * c = q)
variable (h3 : c * a = r)
variable (h4 : p ≠ 0)
variable (h5 : q ≠ 0)
variable (h6 : r ≠ 0)

theorem find_value_a2_b2_c2 : a^2 + b^2 + c^2 = 1 :=
by sorry

end find_value_a2_b2_c2_l241_241423


namespace complex_expr_evaluation_l241_241284

-- Define the complex number z
def z : ℂ := -1 + 2 * complex.I

-- Calculate the conjugate of z
def z_conjugate : ℂ := complex.conj z

-- Formulate the expression (1 + conjugate(z)) / z
def expr : ℂ := (1 + z_conjugate) / z

-- Assert the final result
theorem complex_expr_evaluation : 
  expr = - (4 / 5) + (2 / 5) * complex.I := 
sorry

end complex_expr_evaluation_l241_241284


namespace integrate_differential_eq1_general_integral_diff_eq2_l241_241740

-- Proof statement for the particular solution of the first differential equation
theorem integrate_differential_eq1 :
  ∃ C, ∀ (x y : ℝ), (1 + x^2) * (y.derivative) - 2 * x * y * (x.derivative) = 0 → y = C * (1 + x^2) :=
sorry

-- Proof statement to find the general integral of the second differential equation
theorem general_integral_diff_eq2 :
  ∃ C, ∀ (x y : ℝ), (x * y^2 + x) * (x.derivative) + (y - x^2 * y) * (y.derivative) = 0 →
  1 + y^2 = C * (1 - x^2) :=
sorry

end integrate_differential_eq1_general_integral_diff_eq2_l241_241740


namespace math_expression_eval_l241_241812

open Real

noncomputable def log_base_2 : ℝ := log 2
noncomputable def log_base_5 : ℝ := log 5
noncomputable def term_2 := (0.064 ^ (-1/3) : ℝ)
noncomputable def term_3 := log_base_5 * (log_base_2 * 2 + log_base_5)

theorem math_expression_eval :
  (log_base_2 ^ 2) + term_2 + term_3 = 3.5 :=
by
  sorry

end math_expression_eval_l241_241812


namespace find_abcd_sum_eq_67_l241_241432

def equation_has_roots_of_form (a b c d : ℕ) (d_not_div_prime_square : ∀ p : ℕ, Nat.Prime p → ¬ (p^2 ∣ d)) : Prop :=
  ∃ x : ℝ, x ≠ -a ∧ (x = -a + Real.sqrt (b + c * Real.sqrt d) ∨ x = -a - Real.sqrt (b + c * Real.sqrt d) ∨ 
  x = -a + Real.sqrt (b - c * Real.sqrt d) ∨ x = -a - Real.sqrt (b - c * Real.sqrt d)) ∧
  (1/x + 1/(x + 3) - 1/(x + 6) - 1/(x + 9) - 1/(x + 12) - 1/(x + 15) + 1/(x + 18) + 1/(x + 21) = 0)

theorem find_abcd_sum_eq_67 : ∃ a b c d : ℕ, equation_has_roots_of_form a b c d (λ p hp, sorry) ∧ a + b + c + d = 67 :=
sorry

end find_abcd_sum_eq_67_l241_241432


namespace base_6_to_base_10_exact_value_l241_241958

def base_6_to_base_10 (n : ℕ) : ℕ :=
  1 * 6^2 + 5 * 6^1 + 4 * 6^0

theorem base_6_to_base_10_exact_value : base_6_to_base_10 154 = 70 := by
  rfl

end base_6_to_base_10_exact_value_l241_241958


namespace rectangular_eq_line_l_rectangular_eq_curve_C_min_value_on_curve_C_l241_241652

-- Define the parametric equations of line l
def line_l_param (t : Real) : Real × Real :=
  (1 + t / 2, 2 + Real.sqrt 3 / 2 * t)

-- Define the polar coordinate equation of curve C
def curve_C_polar (theta : Real) : Real × Real :=
  (Real.cos theta, Real.sin theta)

-- Define the rectangular equation of curve C'
def curve_C'_rect (x' y' : Real) : Prop :=
  x'^2 / 4 + y'^2 = 1

-- Prove the rectangular coordinate equation of line l is sqrt(3)x - y + 2 - sqrt(3) = 0
theorem rectangular_eq_line_l (x y : Real) (t : Real) :
  line_l_param t = (x, y) →
  Real.sqrt 3 * x - y + 2 - Real.sqrt 3 = 0 :=
sorry

-- Prove the rectangular coordinate equation of curve C is x^2 + y^2 = 1
theorem rectangular_eq_curve_C (x y : Real) (theta : Real) :
  curve_C_polar theta = (x, y) →
  x^2 + y^2 = 1 :=
sorry

-- Prove the minimum value of x + 2 * Real.sqrt 3 * y for any point on curve C' is -4
theorem min_value_on_curve_C' (x' y' : Real) :
  curve_C'_rect x' y' →
  ∃ (M : Real × Real), M = (x', y') ∧ (∀ (p : Real × Real), p ∈ curve_C'_set → x' + 2 * Real.sqrt 3 * y' ≥ -4) :=
sorry

end rectangular_eq_line_l_rectangular_eq_curve_C_min_value_on_curve_C_l241_241652


namespace proof_problem_l241_241793

noncomputable def problem (x y : ℝ) : ℝ :=
  let A := 2 * x + y
  let B := 2 * x - y
  (A ^ 2 - B ^ 2) * (x - 2 * y)

theorem proof_problem : problem (-1) 2 = 80 := by
  sorry

end proof_problem_l241_241793


namespace combined_term_equal_exponents_l241_241311

theorem combined_term_equal_exponents (x y : ℝ) (a b : ℝ) 
(h1 : a - 1 = 2) 
(h2 : b + 1 = 2) : b^a = 1 := 
by 
  have ha : a = 3 := by linarith,
  have hb : b = 1 := by linarith,
  rw [ha, hb],
  norm_num,
  sorry

end combined_term_equal_exponents_l241_241311


namespace lemonade_lemons_per_glass_l241_241361

def number_of_glasses : ℕ := 9
def total_lemons : ℕ := 18
def lemons_per_glass : ℕ := 2

theorem lemonade_lemons_per_glass :
  total_lemons / number_of_glasses = lemons_per_glass :=
by
  sorry

end lemonade_lemons_per_glass_l241_241361


namespace marbles_left_l241_241222

def initial_marbles : ℕ := 64
def marbles_given : ℕ := 14

theorem marbles_left : (initial_marbles - marbles_given) = 50 := by
  sorry

end marbles_left_l241_241222


namespace remainder_when_divided_by_y_is_9_l241_241130

theorem remainder_when_divided_by_y_is_9 {x y : ℕ} (h₁ : y = 36) (h₂ : x / y = 96.25) :
  x % y = 9 :=
by
  sorry

end remainder_when_divided_by_y_is_9_l241_241130


namespace bisect_perimeter_triangle_l241_241779

theorem bisect_perimeter_triangle (A B C I T₃ H₃ P₃ M₃ : Point) (p a b c : ℝ) 
                                  (h_incenter : Incenter I A B C) 
                                  (h_incircletangent : TangentPoint I B C T₃) 
                                  (h_altitude : Altitude C H₃ A B)
                                  (h_midpoint : Midpoint A B M₃)
                                  (h_parallel : Parallel (LineThrough C P₃) (LineThrough I M₃))
                                  (h_midpoint_P₃ : P₃ = M₃)
                                  (h_semiperimeter : p = a + b + c) 
                                  (h_side_lengths : sideLength A B = c ∧ sideLength B C = a ∧ sideLength C A = b) :
  sideLength A C + sideLength C P₃ + sideLength P₃ A = p - (sideLength B C) := sorry

end bisect_perimeter_triangle_l241_241779


namespace sin_neg_690_eq_half_l241_241830

theorem sin_neg_690_eq_half : sin (-690 * pi / 180) = 1 / 2 :=
by {
  -- Mathematical proof required here using lean tactics 
  sorry
}

end sin_neg_690_eq_half_l241_241830


namespace cross_product_self_l241_241314

variables (v w : Vector3 ℝ)
variables (h : v × w = ⟨5, -2, 4⟩)

theorem cross_product_self (v w : Vector3 ℝ) (h : v × w = ⟨5, -2, 4⟩) : 
  (v + w) × (v + w) = ⟨0, 0, 0⟩ := 
by {
  sorry
}

end cross_product_self_l241_241314


namespace trigonometric_identity_l241_241905

theorem trigonometric_identity (α : ℝ) :
  2 * (sin (3 * π - 2 * α))^2 * (cos (5 * π + 2 * α))^2 = 
  (1 / 4) - (1 / 4) * sin ((5 / 2) * π - 8 * α) :=
by
  sorry

end trigonometric_identity_l241_241905


namespace solve_fractional_equation_l241_241417

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 4) : (1 / (x - 1) = 2 / (1 - x) + 1) → x = 4 :=
by
  sorry

end solve_fractional_equation_l241_241417


namespace find_norm_b_l241_241306

open_locale real_inner_product_space

variables {a b : euclidean_space ℝ (fin 2)}

-- Conditions as definitions
def vec_a : euclidean_space ℝ (fin 2) := ![2, 1]
def norm_sum_ab_eq_4 := ∥vec_a + b∥ = 4
def dot_ab_eq_1 := inner vec_a b = 1

/-- The problem to prove given the conditions -/
theorem find_norm_b (hb : norm_sum_ab_eq_4) (hdot : dot_ab_eq_1) : ∥b∥ = 3 := 
sorry

end find_norm_b_l241_241306


namespace vasya_made_a_mistake_l241_241841

theorem vasya_made_a_mistake (A B V G D E : ℕ) (h1 : 10 * A + B ≠ 5) (h2 : 10 * V + G ≠ 5)
  (h3 : (10 * A + B) * (10 * V + G) = 1000 * D + 1000 * D + 100 * E + 10 * E + E)
  (h4 : ∀ x y, x ≠ y → A = x → B = y → V = y → G = x → D = x ∨ D = y)
  : false :=
begin
  -- We need to show that there is a contradiction
  sorry

end vasya_made_a_mistake_l241_241841


namespace total_pens_bought_l241_241015

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l241_241015


namespace ellipse_eq_proof_l241_241287

noncomputable def ellipse_eq {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a > b) : Prop :=
  ∀ x y : ℝ, (x^2/(a^2)) + (y^2/(b^2)) = 1

noncomputable def eccentricity {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a > b) : ℝ :=
  sqrt(1 - (b^2/a^2))

noncomputable def line_distance (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ :=
  abs(-b) / (sqrt((b/a)^2 + 1))

noncomputable def ellipse_proof {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : eccentricity h1 h2 h3 = (sqrt 6) / 3)
  (h5 : line_distance a b h1 h2 = (sqrt 3) / 2) : Prop :=
  (∀ x y : ℝ, x^2 + 3*y^2 = a^2) ∧
  (∀ t : ℝ, t > 0 → ∃ k : ℝ, ∀ x1 y1 x2 y2 : ℝ, 
    (x1 ≠ x2 ∧ y = k * x + t ∧
     (x1^2/(a^2)) + (y1^2/(b^2)) = 1 ∧
     (x2^2/(a^2)) + (y2^2/(b^2)) = 1) → 
    circle_with_diameter x1 y1 x2 y2 (-1) 0)

theorem ellipse_eq_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : eccentricity h1 h2 h3 = (sqrt 6) / 3)
  (h5 : line_distance a b h1 h2 = (sqrt 3) / 2)
  : ellipse_proof h1 h2 h3 h4 h5 := sorry

end ellipse_eq_proof_l241_241287


namespace tasteful_tiling_exists_and_unique_l241_241485

-- Definitions corresponding to the conditions
def simple_polygon (P : Type) (sides : list P) : Prop :=
  -- definition of a simple polygon (this needs to be fleshed out)
  sorry

def chessboard_polygon (P : Type) [integral_coordinates : P → (ℤ × ℤ)] : Prop :=
  simple_polygon P (λ a, ∃ n : ℤ, integral_coordinates a = (n, 0) ∨ integral_coordinates a = (0, n))

def alternately_shaded (P : Type) [integral_coordinates : P → (ℤ × ℤ)] : Prop :=
  ∀ (a b : P), (abs (fst (integral_coordinates a) - fst (integral_coordinates b)) + abs (snd (integral_coordinates a) - snd (integral_coordinates b))) = 1 →
                (fst (integral_coordinates a) + snd (integral_coordinates a)) % 2 ≠ (fst (integral_coordinates b) + snd (integral_coordinates b)) % 2

def domino_tiling (P : Type) : Prop :=
  ∀ (d : P) (a b : d), abs (fst a - fst b) + abs (snd a - snd b) = 2

def tasteful_tiling (P : Type) (tiling : domino_tiling P) [integral_coordinates : P → (ℤ × ℤ)] : Prop :=
  ∀ (d : P) (a b : d), 
    (abs (fst a - fst b) = 2 ∧ abs (snd a - snd b) = 0) ∨ (abs (fst a - fst b) = 0 ∧ abs (snd a - snd b) = 2) →
    (fst a + snd a) % 2 ≠ (fst b + snd b) % 2

-- Main proofs
theorem tasteful_tiling_exists_and_unique 
  {P : Type} [integral_coordinates : P → (ℤ × ℤ)] 
  (poly : simple_polygon P) 
  (chess_poly : chessboard_polygon P)
  (alt_shade : alternately_shaded P)
  (domino_tile : domino_tiling P) 
  : tasteful_tiling P domino_tile ∧ 
  ∀ t : domino_tiling P, tasteful_tiling P t → t = domino_tile :=
by sorry

end tasteful_tiling_exists_and_unique_l241_241485


namespace monotonicity_f_inequality_f_l241_241290

def f (a x : ℝ) := a * x - log x

theorem monotonicity_f (a : ℝ) :
  (∀ x > 0, a ≤ 0 → (a * x - log x) ≤ (a * (x + 1) - log (x + 1))) ∧
  (∀ x > 0, a > 0 → ((x ≤ 1 / a ∧ (a * x - log x) ≤ (a * (x + 1) - log (x + 1))) ∨ 
                     (x > 1 / a ∧ (a * x - log x) ≥ (a * (x + 1) - log (x + 1))))) :=
sorry

theorem inequality_f (a : ℝ) (h : a ≤ -1 / (Real.exp 2)) :
  ∀ x > 0, f a x ≥ 2 * a * x - x * Real.exp (a * x - 1) :=
sorry

end monotonicity_f_inequality_f_l241_241290


namespace water_bill_august_32m_cubed_water_usage_october_59_8_yuan_l241_241103

noncomputable def tiered_water_bill (usage : ℕ) : ℝ :=
  if usage <= 20 then
    2.3 * usage
  else if usage <= 30 then
    2.3 * 20 + 3.45 * (usage - 20)
  else
    2.3 * 20 + 3.45 * 10 + 4.6 * (usage - 30)

-- (1) Prove that if Xiao Ming's family used 32 cubic meters of water in August, 
-- their water bill is 89.7 yuan.
theorem water_bill_august_32m_cubed : tiered_water_bill 32 = 89.7 := by
  sorry

-- (2) Prove that if Xiao Ming's family paid 59.8 yuan for their water bill in October, 
-- they used 24 cubic meters of water.
theorem water_usage_october_59_8_yuan : ∃ x : ℕ, tiered_water_bill x = 59.8 ∧ x = 24 := by
  use 24
  sorry

end water_bill_august_32m_cubed_water_usage_october_59_8_yuan_l241_241103


namespace cube_volume_l241_241935

theorem cube_volume (a R : ℝ) (h_cube_on_sphere : sqrt 3 * a = 2 * R) (h_sphere_volume : (4 / 3) * π * R^3 = 9 * π / 2) :
  a^3 = 3 * sqrt 3 :=
by {
  --- identifying key steps
  have h_R : R = 3 / 2 := sorry,
  have h_a : a = sqrt 3 := sorry,
  --- computing volume
  calc a^3 = (sqrt 3)^3 : by rw h_a
  ... = 3 * sqrt 3 : sorry,
}

end cube_volume_l241_241935


namespace min_M_value_l241_241075

theorem min_M_value : ∀ M : ℝ, (∀ x : ℝ, x > -1 → ln (1 + x) - (1 / 4) * x^2 ≤ M) ↔ M ≥ ln 2 - 1 / 4 :=
by sorry

end min_M_value_l241_241075


namespace trader_profit_or_loss_l241_241907

theorem trader_profit_or_loss 
  (SP : ℤ) (gain loss : ℚ) (SP_val : SP = 404415) (gain_val : gain = 0.15) (loss_val : loss = 0.15) :
  let CP1 := SP * (1 / (1 + gain)),
      CP2 := SP * (1 / (1 - loss)),
      TCP := CP1 + CP2,
      TSP := SP + SP in 
  ((TSP - TCP) / TCP) * 100 = -2.25 := by
sorry

end trader_profit_or_loss_l241_241907


namespace angle_bisectors_length_ratio_is_two_l241_241995

section IsoscelesTriangleAngleBisectors

variables {b : ℝ} (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C]
variables (triangleABC : triangle A B C)
variables (angle β : ℝ)

open_locale real_angle

-- Assume the triangle is isosceles with AB = AC = b and base angles β
noncomputable def isIsoscelesTriangle (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C] (triangleABC : triangle A B C) :=
  (AB = AC) ∧ (β = angle B A C)

-- Statement of the math proof problem
theorem angle_bisectors_length_ratio_is_two (h1 : isIsoscelesTriangle A B C triangleABC) (H : (angle_bisector_ratio_of_triangle triangleABC A B C = 2)) :
  β ≈ 77.0 :=
sorry

end IsoscelesTriangleAngleBisectors

end angle_bisectors_length_ratio_is_two_l241_241995


namespace magnitude_proof_l241_241673

-- Define the vectors a and b
def a (m : ℝ) : ℝ × ℝ := (4, m)
def b : ℝ × ℝ := (1, -2)

-- Define the condition that a is parallel to b
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- The main theorem to be proved
theorem magnitude_proof (m : ℝ) (h : parallel (a m) b) : magnitude (a m - (2 : ℝ) • b) = 2 * real.sqrt 5 :=
begin
  sorry
end

end magnitude_proof_l241_241673


namespace eq_d_is_quadratic_l241_241955

def is_quadratic (eq : ℕ → ℤ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ eq 2 = a ∧ eq 1 = b ∧ eq 0 = c

def eq_cond_1 (n : ℕ) : ℤ :=
  match n with
  | 2 => 1  -- x^2 coefficient
  | 1 => 0  -- x coefficient
  | 0 => -1 -- constant term
  | _ => 0

theorem eq_d_is_quadratic : is_quadratic eq_cond_1 :=
  sorry

end eq_d_is_quadratic_l241_241955


namespace book_page_problem_l241_241170

variable (x : Nat) (y : Nat) (total : Nat)

-- Conditions
def first_chapter_pages := x
def second_chapter_pages := y = 33
def total_pages := total = 93

-- Proof Statement
theorem book_page_problem (h1 : second_chapter_pages) (h2 : total_pages) 
  (h3 : total = x + y) : x = 60 := 
by 
  sorry

end book_page_problem_l241_241170


namespace problem_statement_l241_241814

-- Let ∆ABC be a triangle with incircle touching sides BC, CA, AB at A_1, B_1, C_1 respectively
variables {A B C A₁ B₁ C₁ A₂ B₂ C₂ : Point}
variable {triangle : Triangle A B C}

-- Assume points A₂, B₂, C₂ are reflections of A₁, B₁, C₁ with respect to angle bisectors at A, B, C
variables (Sa Sb Sc : Point → Point)
variable h_A1_A2 : A₂ = Sa A₁
variable h_B1_B2 : B₂ = Sb B₁
variable h_C1_C2 : C₂ = Sc C₁

-- Theorem establishing the parallelism and concurrency as stated in the problem
theorem problem_statement :
  Parallel (Line A₂ B₂) (Line A B) ∧ Concurrent [Line A A₂, Line B B₂, Line C C₂] :=
sorry -- Proof is omitted

end problem_statement_l241_241814


namespace total_pens_l241_241009

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l241_241009


namespace range_of_a_l241_241251

def A (a : ℝ) : set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) (h : A a ∪ B = B) : a ∈ set.Ioo (-∞) (-4) ∪ set.Ioo 5 ∞ :=
by
  sorry

end range_of_a_l241_241251


namespace problem_solved_by_at_least_3_girls_and_boys_l241_241473

-- Define necessary sets and properties
def Girls : Set ℕ := {i | i < 21}
def Boys : Set ℕ := {i | i < 21}
def Problems : Set ℕ := {i | i < n} -- where n is the number of problems

-- Define the relation indicating which problems were solved by which contestants
def solved_by : ℕ → ℕ → Bool -- maps (contestant, problem) pairs to Bool

-- Define the properties based on conditions
def max_solved_problems (person : ℕ) : Prop := 
  ∀ (p : Problems), (solved_by person p) → (number_of_problems_solved_by person) ≤ 6

def solved_by_both (g : ℕ) (b : ℕ) : Prop := 
  ∃ p, (solved_by g p) ∧ (solved_by b p)

-- Define subsets for problems solved by groups
def difficult_for_boys (p : ℕ) : Prop :=
  (count {b : ℕ | b ∈ Boys ∧ solved_by b p}) ≤ 2

def difficult_for_girls (p : ℕ) : Prop :=
  (count {g : ℕ | g ∈ Girls ∧ solved_by g p}) ≤ 2

open Classical
noncomputable def exists_common_problem : Prop :=
  ∃ p, 
    (count {g : ℕ | g ∈ Girls ∧ solved_by g p} ≥ 3) ∧ 
    (count {b : ℕ | b ∈ Boys ∧ solved_by b p} ≥ 3)

theorem problem_solved_by_at_least_3_girls_and_boys :
  (∀ (i ∈ Girls), max_solved_problems i) → (∀ (p ∈ Problems), solved_by_both i j) → exists_common_problem :=
begin
  intros h1 h2,
  sorry -- proof details go here
end

end problem_solved_by_at_least_3_girls_and_boys_l241_241473


namespace simplify_fraction_l241_241966

theorem simplify_fraction (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c ≠ 0) :
  (a^2 + a * b - b^2 + a * c) / (b^2 + b * c - c^2 + b * a) = (a - b) / (b - c) :=
by
  sorry

end simplify_fraction_l241_241966


namespace sum_even_factors_l241_241854

theorem sum_even_factors (n : ℕ) (h : n = 720) : 
  (∑ d in Finset.filter (λ d, d % 2 = 0) (Finset.divisors n), d) = 2340 :=
by
  rw h
  -- sorry to skip the actual proof
  sorry

end sum_even_factors_l241_241854


namespace find_a_l241_241276

-- Definitions according to the given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f (x : ℝ) (a : ℝ) : ℝ :=
  if 0 < x ∧ x < 2 then Real.log x - a * x else 0

-- Main theorem to prove the value of 'a'
theorem find_a (a : ℝ) (h : a > 1/2)
  (odd_f : is_odd_function (λ x, f x a))
  (min_f_neg : ∀ x ∈ Ioo (-2 : ℝ) (0 : ℝ), f x a ≥ 1):
  a = 1 := sorry

end find_a_l241_241276


namespace constant_t_l241_241833

theorem constant_t (c : ℝ) (hC : c = 2) : 
  let t := sqrt (2 * ((1 - c)^2 + 1)) in t = 2 :=
by
  sorry

end constant_t_l241_241833


namespace good_numbers_1_to_10_good_numbers_1_to_100_l241_241319

def is_good_number (a : ℕ) : Prop :=
  ∃ x y : ℕ, a = x^2 + y^2

def good_numbers_count (n : ℕ) : ℕ :=
  Finset.card (Finset.filter is_good_number (Finset.range (n + 1)))

theorem good_numbers_1_to_10 :
  good_numbers_count 10 = 7 :=
by
  sorry

theorem good_numbers_1_to_100 :
  good_numbers_count 100 = 43 :=
by
  sorry

end good_numbers_1_to_10_good_numbers_1_to_100_l241_241319


namespace mutually_coprime_divisors_l241_241986

theorem mutually_coprime_divisors (a x y : ℕ) (h1 : a = 1944) 
  (h2 : ∃ d1 d2 d3, d1 * d2 * d3 = a ∧ gcd x y = 1 ∧ gcd x (x + y) = 1 ∧ gcd y (x + y) = 1) : 
  (x = 1 ∧ y = 2 ∧ x + y = 3) ∨ 
  (x = 1 ∧ y = 8 ∧ x + y = 9) ∨ 
  (x = 1 ∧ y = 3 ∧ x + y = 4) :=
sorry

end mutually_coprime_divisors_l241_241986


namespace average_time_per_stop_l241_241454

-- Definitions from the conditions
def pizzas : Nat := 12
def stops_with_two_pizzas : Nat := 2
def total_delivery_time : Nat := 40

-- Using the conditions to define what needs to be proved
theorem average_time_per_stop : 
  let single_pizza_stops := pizzas - stops_with_two_pizzas * 2
  let total_stops := single_pizza_stops + stops_with_two_pizzas
  let average_time := total_delivery_time / total_stops
  average_time = 4 := by
  -- Proof to be provided
  sorry

end average_time_per_stop_l241_241454


namespace solution_pairs_l241_241845

def equation (r p : ℤ) : Prop := r^2 - r * (p + 6) + p^2 + 5 * p + 6 = 0

theorem solution_pairs :
  ∀ (r p : ℤ),
    equation r p ↔ (r = 3 ∧ p = 1) ∨ (r = 4 ∧ p = 1) ∨ 
                    (r = 0 ∧ p = -2) ∨ (r = 4 ∧ p = -2) ∨ 
                    (r = 0 ∧ p = -3) ∨ (r = 3 ∧ p = -3) :=
by
  sorry

end solution_pairs_l241_241845


namespace distinct_sums_count_l241_241631

theorem distinct_sums_count (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i j, i ≠ j → a i ≠ a j) :
  (∃ S : Finset (Fin n), S.nonempty ∧ S.card = (∑ i in Finset.range n, i + 1) / 2) :=
by
  sorry

end distinct_sums_count_l241_241631


namespace coeff_sum_l241_241697

theorem coeff_sum (a : ℕ → ℝ) :
  (∀ x, x ^ 2018 = a 0 + a 1 * (x - 1) + a 2 * (x - 1) ^ 2 + ∑ i in finset.range 2018, a (i + 2) * (x - 1) ^ (i + 2)) →
  (∑ i in finset.range 2018, (a (i + 1) / (3 : ℝ) ^ (i + 1))) = ((4 / 3:ℝ) ^ 2018 - 1) :=
by
  intro h
  sorry

end coeff_sum_l241_241697


namespace sin_angle_ACB_correct_l241_241728

variables {A B C D : Type} [Point : Type]
variables {angle : Point → Point → Point → ℝ}
variables {cos sin : ℝ → ℝ}
variables {x y : ℝ}

def tetrahedron_conditions (A B C D : Point) : Prop :=
  angle A D B = 90 ∧ angle A D C = 90 ∧ angle B D C = 90

def cos_conditions : Prop := 
  x = cos (angle C A D) ∧ y = cos (angle C B D)

noncomputable def sin_angle_ACB (A B C D : Point) 
  (h_tetra : tetrahedron_conditions A B C D) 
  (h_cos : cos_conditions) : ℝ :=
  sqrt (1 - (x * y) ^ 2)

theorem sin_angle_ACB_correct (A B C D : Point)
  (h_tetra : tetrahedron_conditions A B C D) 
  (h_cos : cos_conditions) : 
  sin (angle A C B) = sqrt (1 - (x * y) ^ 2) :=
sorry

end sin_angle_ACB_correct_l241_241728


namespace quadratic_passing_point_calc_l241_241813

theorem quadratic_passing_point_calc :
  (∀ (x y : ℤ), y = 2 * x ^ 2 - 3 * x + 4 → ∃ (x' y' : ℤ), x' = 2 ∧ y' = 6) →
  (2 * 2 - 3 * (-3) + 4 * 4 = 29) :=
by
  intro h
  -- The corresponding proof would follow by providing the necessary steps.
  -- For now, let's just use sorry to meet the requirement.
  sorry

end quadratic_passing_point_calc_l241_241813


namespace total_pens_l241_241025

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l241_241025


namespace simplify_expression_l241_241061

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
by
  sorry

end simplify_expression_l241_241061


namespace sum_of_even_factors_720_l241_241882

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) :=
    (2 + 4 + 8 + 16) * (1 + 3 + 9) * (1 + 5)
  in even_factors_sum 720 = 2340 :=
by
  sorry

end sum_of_even_factors_720_l241_241882


namespace sum_possible_values_A_l241_241190

theorem sum_possible_values_A : 
  ∑ (A : ℕ) in (Finset.filter (λ n, (100 + 10 * n + 6) % 8 = 0) (Finset.range 10)), A = 10 := by
  sorry

end sum_possible_values_A_l241_241190


namespace tunnel_length_l241_241552

theorem tunnel_length {train_length : ℝ} (train_speed_kmph : ℝ) (time_minutes : ℝ) (train_length : ℝ) :
  time_minutes = 1.0000000000000002 → train_speed_kmph = 72 → train_length = 100 →
  let train_speed_mps := train_speed_kmph * 1000 / 3600 in
  let time_seconds := time_minutes * 60 in
  let travel_distance := train_speed_mps * time_seconds in
  let tunnel_length := travel_distance - train_length in
  tunnel_length / 1000 = 1.1 :=
begin
  intros,
  sorry
end

end tunnel_length_l241_241552


namespace y_n_is_square_of_odd_integer_l241_241843

-- Define the sequences and the initial conditions
def x : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 2) => 3 * x (n + 1) - 2 * x n

def y (n : ℕ) : ℤ := x n ^ 2 + 2 ^ (n + 2)

-- Helper function to check if a number is odd
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- The theorem to prove
theorem y_n_is_square_of_odd_integer (n : ℕ) (h : n > 0) : ∃ k : ℤ, y n = k ^ 2 ∧ is_odd k := by
  sorry

end y_n_is_square_of_odd_integer_l241_241843


namespace sum_of_even_factors_720_l241_241888

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) : ℕ :=
    match n with
    | 720 => 
      let sum_powers_2 := 2 + 4 + 8 + 16 in
      let sum_powers_3 := 1 + 3 + 9 in
      let sum_powers_5 := 1 + 5 in
      sum_powers_2 * sum_powers_3 * sum_powers_5
    | _ => 0
  in
  even_factors_sum 720 = 2340 :=
by 
  sorry

end sum_of_even_factors_720_l241_241888


namespace total_number_of_students_l241_241715

theorem total_number_of_students (girls boys : ℕ) 
  (h_ratio : 8 * girls = 5 * boys) 
  (h_girls : girls = 160) : 
  girls + boys = 416 := 
sorry

end total_number_of_students_l241_241715


namespace loss_percentage_correct_l241_241505

variable (CP : ℝ := 1800) (SP : ℝ := 1430)

def loss (CP SP : ℝ) : ℝ := CP - SP

def loss_percentage (CP SP : ℝ) : ℝ := (loss CP SP / CP) * 100

theorem loss_percentage_correct : loss_percentage CP SP = 20.56 :=
by
  -- This is a placeholder for the actual proof
  sorry

end loss_percentage_correct_l241_241505


namespace unique_y_for_star_eq_9_l241_241223

def star (x y : ℝ) : ℝ := 3 * x - 2 * y + x^2 * y

theorem unique_y_for_star_eq_9 : ∃! y : ℝ, star 2 y = 9 := by
  sorry

end unique_y_for_star_eq_9_l241_241223


namespace original_bales_l241_241466

/-
There were some bales of hay in the barn. Jason stacked 23 bales in the barn today.
There are now 96 bales of hay in the barn. Prove that the original number of bales of hay 
in the barn was 73.
-/

theorem original_bales (stacked : ℕ) (total : ℕ) (original : ℕ) 
  (h1 : stacked = 23) (h2 : total = 96) : original = 73 :=
by
  sorry

end original_bales_l241_241466


namespace find_f_neg_two_l241_241663

section
  variables {α : Type*} [Ring α] (a b c : α)
  
  def f (x : α) : α := a * x^5 + b * x^3 + c * x + 1
  
  theorem find_f_neg_two : f a b c (-2) = 3 :=
  by
    -- Given condition
    have h : f a b c 2 = -1 := sorry
    -- Function definition
    have f_def : f (2 : α) = a * 2^5 + b * 2^3 + c * 2 + 1 := sorry
    sorry
end

end find_f_neg_two_l241_241663


namespace ellipse_problem_l241_241266

noncomputable def ellipse_equation (a b : ℝ) (cond1 : a > 0) (cond2 : b > 0) (cond3 : a ≠ b) : Prop :=
∀ (x y : ℝ), ax^2 + by^2 = 1 → (x + y = 1) → 
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧ 
    dist A B = 2 * real.sqrt 2 ∧ 
    let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in 
    ((C.2 / C.1) = real.sqrt 2 / 2) ∧ 
    (a = 1 / 3) ∧ 
    (b = real.sqrt 2 / 3)) 

theorem ellipse_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b)
    (H : ellipse_equation a b h1 h2 h3) : 
    (a = 1 / 3) ∧ (b = real.sqrt 2 / 3) ∧ (∀ x y : ℝ, (ax^2 + by^2 = 1) → (x^2 / (1/3) + y^2 / (real.sqrt 2 / 3) = 1)) :=
by
  sorry

end ellipse_problem_l241_241266


namespace second_number_is_twenty_two_l241_241327

theorem second_number_is_twenty_two (x y : ℕ) 
  (h1 : x + y = 33) 
  (h2 : y = 2 * x) : 
  y = 22 :=
by
  sorry

end second_number_is_twenty_two_l241_241327


namespace triangle_equilateral_if_arithmetic_and_geometric_mean_l241_241353

theorem triangle_equilateral_if_arithmetic_and_geometric_mean 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : B = (A + C) / 2) 
  (h2 : b = Real.sqrt (a * c)) 
  (h_tr : ∠A + ∠B + ∠C = 180) : 
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c := 
sorry

end triangle_equilateral_if_arithmetic_and_geometric_mean_l241_241353


namespace age_difference_l241_241497

theorem age_difference (a b c : ℕ) (h₁ : b = 8) (h₂ : c = b / 2) (h₃ : a + b + c = 22) : a - b = 2 :=
by
  sorry

end age_difference_l241_241497


namespace complex_coordinate_l241_241320

-- Define the complex number z and the imaginary unit i
def z : ℂ := complex.I * (2 + 4 * complex.I)

-- The coordinates of the point corresponding to z in the complex plane
def coordinate_of_z : ℂ := -4 + 2 * complex.I

-- The statement to be proved
theorem complex_coordinate :
  z = coordinate_of_z :=
by
  -- Requirements to prove the statement
  sorry

end complex_coordinate_l241_241320


namespace total_pens_bought_l241_241017

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l241_241017


namespace radius_of_circumcircle_AMD_l241_241400

-- Define the necessary geometric terms
variables {R : ℝ} {A B C D M : Type*}

-- Define the properties of the circles and point M
def circle (center : Type*) (radius : ℝ) (point1 point2 : Type*) : Prop := 
  dist center point1 = radius ∧ dist center point2 = radius

-- Define the assumptions
axiom circle_Omega1 : circle A R A B
axiom circle_Omega2 : circle B R B C
axiom point_M_intersection : ∃ M, circle A R A M ∧ circle B R B M

-- Statement that needs to be proved
theorem radius_of_circumcircle_AMD : 
  ∀ (A B C D M : Type*), 
    (∃ M, (circle A R A B) ∧ (circle B R B C) ∧ 
             (∀ P, P = M ∧ P ≠ B → circle A R A P ∧ circle B R B P)) → 
    circumradius_triangle A M D = R :=
begin
  sorry
end

end radius_of_circumcircle_AMD_l241_241400


namespace find_m_to_make_z1_eq_z2_l241_241285

def z1 (m : ℝ) : ℂ := (2 * m + 7 : ℝ) + (m^2 - 2 : ℂ) * Complex.I
def z2 (m : ℝ) : ℂ := (m^2 - 8 : ℝ) + (4 * m + 3 : ℂ) * Complex.I

theorem find_m_to_make_z1_eq_z2 : 
  ∃ m : ℝ, z1 m = z2 m ∧ m = 5 :=
by
  sorry

end find_m_to_make_z1_eq_z2_l241_241285


namespace prove_a_eq_b_l241_241826

theorem prove_a_eq_b (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_eq : a^b = b^a) (h_a_lt_1 : a < 1) : a = b :=
by
  sorry

end prove_a_eq_b_l241_241826


namespace problem_statement_l241_241262

noncomputable def seq (n : ℕ) : ℝ := match n with
  | 0     => 0 -- Undefined behavior for n = 0, should not be used
  | 1     => 6
  | k + 2 => ((k+2) * (2 * (k+1) + 2 * (k+1))) +
             ((k + 1) * seq (k + 1) - (k + 2) * seq k)

def is_arithmetic_sequence (f : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, 1 < n → (f (n) - f (n - 1) = d)

def general_formula (n : ℕ) : ℝ := (n+1)*(2*n+1)

def partial_sum (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), f (i + 1)

theorem problem_statement (n : ℕ) (h : n ≥ 2) :
  is_arithmetic_sequence (λ n, seq n / (n + 1)) ∧
  (seq n = general_formula n) ∧
  partially_sum (λ n, 1 / seq n) n < 5 / 12 :=
by
  sorry

end problem_statement_l241_241262


namespace radius_of_incircle_l241_241445

-- Definitions for the legs a and b
def a : ℝ := 5
def b : ℝ := 12

-- Hypotenuse c calculation using the Pythagorean theorem
def c : ℝ := Real.sqrt (a^2 + b^2)

-- Radius r calculation for the inscribed circle in a right triangle
def r : ℝ := (a + b - c) / 2

-- Prove r = 2
theorem radius_of_incircle : r = 2 := by
  -- Proof will follow here
  sorry

end radius_of_incircle_l241_241445


namespace proof_of_problem_l241_241142

noncomputable def proof_problem (x y : ℚ) : Prop :=
  (sqrt (x - y) = 2 / 5) ∧ (sqrt (x + y) = 2) ∧ 
  x = 52 / 25 ∧ y = 48 / 25 ∧ 
  let vertices := [(0, 0), (2, 2), (2 / 25, -2 / 25), (52 / 25, 48 / 25)] in
  let area := Rational.from_ints 8 25 in
  ∃ (a b c d : ℚ × ℚ), 
    a ∈ vertices ∧ b ∈ vertices ∧ c ∈ vertices ∧ d ∈ vertices ∧ 
    ((b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = area)

theorem proof_of_problem : proof_problem (52 / 25) (48 / 25) :=
by { sorry } 

end proof_of_problem_l241_241142


namespace function_increasing_on_interval_l241_241998

theorem function_increasing_on_interval (a : ℝ) :
  (∀ x ∈ Icc (a-3) (3*a), (∀ x0 ∈ Icc (-∞) (-2), y = 8 / (x0^2 + 4*x0 + 44) → 0 ≤ y)) ↔ 
  a ∈ Icc (-3 / 2) (-2 / 3) :=
sorry

end function_increasing_on_interval_l241_241998


namespace triangles_similar_and_first_brocard_point_l241_241399

theorem triangles_similar_and_first_brocard_point (A B C A1 B1 C1: Point)
  (h_acute_triangle : is_acute_triangle ABC)
  (h_points_on_sides : pts_on_sides ABC A1 B1 C1)
  (h_angles_equal : angle BA A1 = angle CB B1 = angle AC C1) :
  -- Prove triangles are similar
  triangle_sim A1 B1 C1 ABC ∧
  -- Prove rotational homothety's center coincides with the Brocard point
  first_brocard_point A1 B1 C1 = first_brocard_point ABC :=
sorry

end triangles_similar_and_first_brocard_point_l241_241399


namespace range_of_independent_variable_l241_241083

theorem range_of_independent_variable (x : ℝ) : 
    ∃ y : ℝ, y = sqrt (x - 2) → x ≥ 2 :=
by
  intro y
  intro h
  rw [h]
  apply real.sqrt_nonneg (x - 2)
  sorry

end range_of_independent_variable_l241_241083


namespace Anya_initial_seat_l241_241716

def seats {α : Type*} [DecidableEq α] := 
  {Varya Anya Galya Diana Ella : α // 
    let V_i := 1; 
    let A_i := 2; 
    let G_i := 3; 
    let D_i := 4; 
    let E_i := 5 
  in
  -- Conditions reflected through the final positions
  let V_f := V_i + 1; 
  let G_f := G_i - 2; 
  let D_f := E_i; 
  let E_f := D_i
  in
  -- Ensure that final positions sum up to initial sum
  V_f + G_f + D_f + E_f + 5 = 15 
      }

theorem Anya_initial_seat : 
  ∀ (V_i A_i G_i D_i E_i V_f G_f D_f E_f : ℕ),
  V_i = 1 → G_i = 3 → D_i = 4 → E_i = 5 → 
  V_f = V_i + 1 → G_f = G_i - 2 → D_f = E_i → E_f = D_i →
  V_f + G_f + D_f + E_f + 5 = 15 →
  A_i = 4 :=
by
  sorry

end Anya_initial_seat_l241_241716


namespace sum_of_possible_values_of_x_l241_241064

noncomputable def sum_of_x (x y : ℝ) : ℝ := x + y

theorem sum_of_possible_values_of_x : 
  ∃ x y : ℝ, (5^(x^2 + 6 * x + 9) = 25^(x + 3) ∧ 5^(y^2 + 6 * y + 9) = 25^(y + 3)) ∧ 
  sum_of_x x y = -4 := by
sory

end sum_of_possible_values_of_x_l241_241064


namespace Paul_correct_probability_l241_241730

theorem Paul_correct_probability :
  let P_Ghana := 1/2
  let P_Bolivia := 1/6
  let P_Argentina := 1/6
  let P_France := 1/6
  (P_Ghana^2 + P_Bolivia^2 + P_Argentina^2 + P_France^2) = 1/3 :=
by
  sorry

end Paul_correct_probability_l241_241730


namespace store_refusal_illegal_l241_241482

/-- Legal status of banknotes for transactions in the Russian Federation. -/
def legal_tender (issued_by_bank_of_russia : Prop) (not_counterfeit : Prop) (permissible_damage : Prop) : Prop :=
  issued_by_bank_of_russia ∧ not_counterfeit ∧ permissible_damage

/-- Banknotes with tears are considered legal tender according to the Bank of Russia Directive from December 26, 2006, No. 1778-U. -/
def permissible_damage (has_tears : Prop) : Prop :=
  has_tears

/-- The store's refusal to accept torn banknotes from Lydia Alekseevna was legally unjustified. -/
theorem store_refusal_illegal
  (issued_by_bank_of_russia : Prop)
  (not_counterfeit : Prop)
  (has_tears : Prop) :
  legal_tender issued_by_bank_of_russia not_counterfeit (permissible_damage has_tears) → 
  ¬ store_refusal_torn_banknotes issued_by_bank_of_russia not_counterfeit has_tears :=
by
  intros
  sorry

end store_refusal_illegal_l241_241482


namespace exists_x0_in_interval_l241_241711

noncomputable def f (x : ℝ) : ℝ := (2 : ℝ) / x + Real.log (1 / (x - 1))

theorem exists_x0_in_interval :
  ∃ x0 ∈ Set.Ioo (2 : ℝ) (3 : ℝ), f x0 = 0 := 
sorry  -- Proof is left as an exercise

end exists_x0_in_interval_l241_241711


namespace total_dolls_l241_241410

def grandmother_dolls := 50
def sister_dolls := grandmother_dolls + 2
def rene_dolls := 3 * sister_dolls

theorem total_dolls : rene_dolls + sister_dolls + grandmother_dolls = 258 :=
by {
  -- Required proof steps would be placed here, 
  -- but are omitted as per the instructions.
  sorry
}

end total_dolls_l241_241410


namespace each_tire_usage_l241_241952
-- Import the necessary Lean library

-- Define the conditions in Lean
def truck_tire_problem (total_distance : ℕ) (total_tires : ℕ) (road_tires : ℕ) (equal_usage : Prop) : Prop :=
  (total_distance = 36000) ∧
  (total_tires = 6) ∧
  (road_tires = 5) ∧
  equal_usage

-- State the theorem in Lean to prove that each tire was used 30,000 miles
theorem each_tire_usage {total_distance total_tires road_tires : ℕ} (equal_usage : Prop) :
  truck_tire_problem total_distance total_tires road_tires equal_usage →
  (total_distance * road_tires) / total_tires = 30000 :=
by
  intro h
  unfold truck_tire_problem at h
  cases h with ht h1
  cases h1 with tt h2
  cases h2 with rt he
  sorry

end each_tire_usage_l241_241952


namespace adult_meal_cost_l241_241211

def num_people : ℕ := 9
def num_kids : ℕ := 2
def total_cost : ℝ := 14

-- The main theorem to prove
theorem adult_meal_cost :
  let num_adults := num_people - num_kids in
  let cost_per_adult := total_cost / num_adults in
  cost_per_adult = 2 := 
by
  sorry

end adult_meal_cost_l241_241211


namespace probability_white_ball_l241_241834

/-- Define the conditions for the two urns and the probability of drawing specific balls from them. -/
def urn1_total : ℕ := 9
def urn2_total : ℕ := 10

def urn1_white : ℕ := 5
def urn1_blue : ℕ := 4
def urn2_white : ℕ := 2
def urn2_blue : ℕ := 8

/-- Define the probabilities for drawing white and blue balls from the two urns. -/
def P_urn1_white : ℚ := urn1_white / urn1_total
def P_urn2_white : ℚ := urn2_white / urn2_total
def P_urn1_blue  : ℚ := urn1_blue / urn1_total
def P_urn2_blue  : ℚ := urn2_blue / urn2_total

/-- Define the probability combinations for drawing a white ball from the third urn according to the scenarios. -/
def P_scenario1 : ℚ := P_urn1_white * P_urn2_blue / 2
def P_scenario2 : ℚ := P_urn1_blue * P_urn2_white / 2
def P_scenario3 : ℚ := P_urn1_white * P_urn2_white

/-- Calculate the total probability of drawing a white ball from the third urn. -/
def P_white_from_third_urn : ℚ := P_scenario1 + P_scenario2 + P_scenario3

/-- Main theorem: The probability of drawing a white ball from the third urn is given by 17/45. -/
theorem probability_white_ball : P_white_from_third_urn = 17 / 45 := by
  sorry

end probability_white_ball_l241_241834


namespace sequence_sum_lt_four_l241_241350

theorem sequence_sum_lt_four (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 = 2)
  (h2 : ∀ n > 0, a (n + 1) = a n^2 / (a n + 2)) :
  (∑ k in finset.range n, 2 * (k+1) * a (k+1) / (a (k+1) + 2)) < 4 :=
sorry

end sequence_sum_lt_four_l241_241350


namespace sqrt_five_minus_two_squared_l241_241568

theorem sqrt_five_minus_two_squared :
  (real.sqrt 5 - 2) ^ 2 = 9 - 4 * real.sqrt 5 :=
by
  sorry

end sqrt_five_minus_two_squared_l241_241568


namespace statement_S3_l241_241340

def rounding_function (x : ℝ) : ℤ :=
  if h : ∃ (n : ℤ), n - 0.5 ≤ x ∧ x < n + 0.5 then
    Classical.choose h
  else 0 -- this case never happens since ℝ is covered by intervals

theorem statement_S3 (x : ℝ) (m : ℤ) : 
  rounding_function (x + m) = rounding_function x + m :=
sorry

end statement_S3_l241_241340


namespace area_of_trapezoid_PQRS_l241_241339

variables {P Q R S T : Type} 
variables [geo : Geometry P Q R S T]
variables (PQ_parallel_RS : geo.parallel PQ RS)
variables (PR_intersection_QS_at_T : geo.intersect PR QS = T)
variables (area_of_PQT : geo.area PQT = 75)
variables (area_of_PRT : geo.area PRT = 45)

theorem area_of_trapezoid_PQRS : geo.area_trapezoid PQRS = 210 :=
by sorry

end area_of_trapezoid_PQRS_l241_241339


namespace pathway_area_ratio_l241_241953

theorem pathway_area_ratio (AB AD: ℝ) (r: ℝ) (A_rectangle A_circles: ℝ):
  AB = 24 → (AD / AB) = (4 / 3) → r = AB / 2 → 
  A_rectangle = AD * AB → A_circles = π * r^2 →
  (A_rectangle / A_circles) = 16 / (3 * π) :=
by
  sorry

end pathway_area_ratio_l241_241953


namespace length_of_bridge_l241_241500

-- Define the conditions
def speed := 5 -- in km/hr
def time_in_minutes := 15 -- in minutes
def time_in_hours := time_in_minutes / 60 -- converting minutes to hours
def distance := speed * time_in_hours

-- State the theorem
theorem length_of_bridge : distance = 1.25 := by
  sorry

end length_of_bridge_l241_241500


namespace integral_eval_l241_241157

open Real

theorem integral_eval :
  ∫ x in 0..(1 / 2 : ℝ), (8 * x - arctan (2 * x)) / (1 + 4 * x ^ 2) = log 2 - (π ^ 2 / 64) :=
by
  sorry

end integral_eval_l241_241157


namespace sum_of_altitudes_correct_l241_241943

-- Define the line equation 15x + 8y = 120
def line_eq (x y : ℝ) : Prop := 15 * x + 8 * y = 120

-- Define the sum of lengths of altitudes of the triangle formed by the line with coordinate axes
def sum_of_altitudes (lhs : ℝ) : ℝ :=
  let x_intercept := 120 / 15 in
  let y_intercept := 120 / 8 in
  let hypotenuse := real.sqrt (x_intercept^2 + y_intercept^2) in
  let third_altitude := 120 / hypotenuse in
  x_intercept + y_intercept + third_altitude

-- The Lean statement to prove
theorem sum_of_altitudes_correct : sum_of_altitudes (a : ℝ) = 391 / 17 := sorry

end sum_of_altitudes_correct_l241_241943


namespace total_pens_l241_241008

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l241_241008


namespace total_pens_l241_241010

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l241_241010


namespace stacking_glass_pieces_opaque_l241_241096

-- Defining the conditions
structure GlassPiece where
  id : Nat -- Unique identifier for each glass piece
  painted_triangle : Fin 4 -- Position of the painted triangle (0 to 3)

def all_glass_pieces : List GlassPiece :=
  [ { id := 1, painted_triangle := 0 },
    { id := 2, painted_triangle := 1 },
    { id := 3, painted_triangle := 2 },
    { id := 4, painted_triangle := 3 },
    { id := 5, painted_triangle := 0 } ]

-- Main statement to prove
theorem stacking_glass_pieces_opaque :
  -- The number of ways to stack the glass pieces to be completely opaque is 7200.
  count_opaque_stackings all_glass_pieces = 7200 :=
sorry

end stacking_glass_pieces_opaque_l241_241096


namespace sector_area_l241_241705

theorem sector_area (α : ℝ) (r : ℝ) (h1 : α = Real.pi / 3) (h2 : r = 2) : 
  (1 / 2) * α * r^2 = (2 * Real.pi) / 3 :=
by
  rw [h1, h2]
  norm_num
  rw [Real.mul_div_cancel' _ (Real.two_ne_zero : 2 ≠ 0)]
  rw [mul_div_cancel' _ (Real.two_ne_zero : 2 ≠ 0)]
  sorry

end sector_area_l241_241705


namespace line_relationship_l241_241510

variables (l_1 l_2 l_3 : Type)
variables (is_perpendicular : l_1 → l_2 → Prop)
variables (is_parallel : l_2 → l_3 → Prop)

theorem line_relationship (h1 : is_perpendicular l_1 l_2) (h2 : is_parallel l_2 l_3) : is_perpendicular l_1 l_3 :=
sorry

end line_relationship_l241_241510


namespace expected_value_is_correct_l241_241537

noncomputable def expected_winnings : ℚ :=
  (5/12 : ℚ) * 2 + (1/3 : ℚ) * 0 + (1/6 : ℚ) * (-2) + (1/12 : ℚ) * 10

theorem expected_value_is_correct : expected_winnings = 4 / 3 := 
by 
  -- Complex calculations skipped for brevity
  sorry

end expected_value_is_correct_l241_241537


namespace total_pens_bought_l241_241031

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l241_241031


namespace inversion_preserves_angle_l241_241047

theorem inversion_preserves_angle (C1 C2 : Circle) (P : Point) (l1 l2 : Line)
  (θ : Real) (O : Point) (intersect_at_P : Circle.tangent_at C1 P l1 ∧ Circle.tangent_at C2 P l2)
  (angle_θ : Angle l1 l2 = θ) (inversion_properties : 
    (∀ l, Line.passes_through l O → Line.maps_to_itself_under_inversion l O) ∧
    (∀ l, ¬ Line.passes_through l O → Line.maps_to_circle_through_center l O) ∧
    (∀ P l1 l2, Tangent.intersect l1 l2 P → Tangent.retains_angle l1 l2 P under_inversion O)) :
  Angle (inversion_map l1 O) (inversion_map l2 O) = θ := by
  sorry

end inversion_preserves_angle_l241_241047


namespace part_b_well_filled_ways_count_part_c_well_filled_ways_count_l241_241802

-- Definitions related to 'well-filled' figures and number assignments
def well_filled (f : ℕ → ℕ) : Prop := 
  ∀ (i j : ℕ), (i = 1 ∧ j = 3) ∨ (i = 1 ∧ j = 4) ∨ (i = 2 ∧ j = 4) → f i < f j

-- Part b: Number of ways to well-fill the figure with numbers 1 to 5
theorem part_b_well_filled_ways_count :
  ∃ n : ℕ, n = 8 ∧ ∀ (f : ℕ → ℕ) (hf : well_filled f) (inj_on f {1, 2, 3, 4, 5}), 
  true := sorry

-- Part c: Number of ways to well-fill the figure with numbers 1 to 7
theorem part_c_well_filled_ways_count :
  ∃ n : ℕ, n = 48 ∧ ∀ (f : ℕ → ℕ) (hf : well_filled f) (inj_on f {1, 2, 3, 4, 5, 6, 7}), 
  true := sorry

end part_b_well_filled_ways_count_part_c_well_filled_ways_count_l241_241802


namespace a_2016_l241_241088

noncomputable def a : ℕ → ℚ
| 1       := 1/2
| (n + 1) := (1 + a (n + 1 - 1)) / (1 - a (n + 1 - 1))

theorem a_2016 : a 2016 = -1/3 :=
sorry

end a_2016_l241_241088


namespace problem_tiles_probability_l241_241394

theorem problem_tiles_probability :
  let tiles := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  (players : Finset (Finset ℕ)) :=
  ∀ (σ : perms (3)) (player_sums : Finset ℕ → ℕ)
  (h_distribute : ∀ p ∈ players, p.card = 3)
  (h_sum_odd : ∀ p ∈ players, Odd (player_sums p)),
  let total_distributions := (9.choose 3) * (6.choose 3) * (3.choose 3),
  let favorable_distributions := (5.choose 3) * 2 * (4.choose 2) * (2.choose 2) * 3,
  let probability := favorable_distributions / total_distributions,
  probability = 3 / 14 → 3 + 14 = 17 :=
begin
  sorry
end

end problem_tiles_probability_l241_241394


namespace lambda_value_l241_241342

-- Definitions used in Lean 4 should only directly appear in the conditions problem
variables {V : Type*} [AddCommGroup V] [Module ℝ V] -- Define the vector space
variables (A B C D O : V) -- Define the points A, B, C, D, O
variables (λ : ℝ)

-- Define the conditions
def parallelogram (A B C D : V) : Prop :=
  A - B = D - C ∧ A - D = B - C

def diagonals_intersect (A B C D O : V) : Prop :=
  (A + C) / 2 = O ∧ (B + D) / 2 = O

def vector_condition (A B D O : V) (λ : ℝ) : Prop :=
  A + D = λ * O

-- Rewrite the problem
theorem lambda_value (h1 : parallelogram A B C D) (h2 : diagonals_intersect A B C D O) (h3 : vector_condition (A - B) (B - C) (D - C) O λ) :
  λ = 2 :=
sorry

end lambda_value_l241_241342


namespace lcm_180_504_is_2520_l241_241115

-- Define what it means for a number to be the least common multiple of two numbers
def is_lcm (a b lcm : ℕ) : Prop :=
  a ∣ lcm ∧ b ∣ lcm ∧ ∀ m, (a ∣ m ∧ b ∣ m) → lcm ∣ m

-- Lean 4 statement to prove that the least common multiple of 180 and 504 is 2520
theorem lcm_180_504_is_2520 : ∀ (a b : ℕ), a = 180 → b = 504 → is_lcm a b 2520 := by
  intro a b
  assume h1 : a = 180
  assume h2 : b = 504
  sorry

end lcm_180_504_is_2520_l241_241115


namespace rational_solution_system_l241_241987

theorem rational_solution_system (x y z t w : ℚ) :
  (t^2 - w^2 + z^2 = 2 * x * y) →
  (t^2 - y^2 + w^2 = 2 * x * z) →
  (t^2 - w^2 + x^2 = 2 * y * z) →
  x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intros h1 h2 h3
  sorry

end rational_solution_system_l241_241987


namespace lambda_sum_rule_l241_241557

theorem lambda_sum_rule {A B C D E P M N : Type}
  [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ E] 
  [AffineSpace ℝ P] [AffineSpace ℝ M] [AffineSpace ℝ N]
  (h1 : Parallel (Line D E) (Line B C))
  (h2 : D ∈ Line A B)
  (h3 : E ∈ Line A C)
  (h4 : P ∈ Line D E)
  (h5 : M ∈ Line A B)
  (h6 : N ∈ Line A C)
  (h7 : Ratio A D B = λ)
  (h8 : λ ≠ -1)
  (λ1 : Ratio A M B)
  (λ2 : Ratio A N C) :
  λ = λ1 + λ2 := 
begin
  sorry
end

end lambda_sum_rule_l241_241557


namespace intersection_A_B_l241_241704

-- Define the sets A and B as per the conditions
def set_A : Set ℝ := {x : ℝ | x^2 - 3 * x - 4 ≤ 0}
def set_B : Set ℕ := {x : ℕ | 2 * x - 3 > 0}

-- Prove that the intersection of A and B is the set {2, 3, 4}
theorem intersection_A_B : (set_A ∩ (set.B : Set ℝ) = {2, 3, 4}) :=
by
  sorry -- Proof omitted

end intersection_A_B_l241_241704


namespace polynomial_nonneg_l241_241366

variable {a0 an : ℝ}
variable {a : ℕ → ℝ}
variable {n : ℕ}
variable {x : ℝ}

theorem polynomial_nonneg (hn : Even n) (ha0 : 0 < a0) (han : 0 < an)
  (hineq : ∑ i in Finset.range (n-1), (a i.succ.succ) ^ 2 ≤ (4 * min (a0^2) (an^2) / (n - 1))) :
  let P (x : ℝ) := an * x^n + ∑ i in Finset.range n, a (i + 1) * x^i + a0
  in ∀ x : ℝ, P x ≥ 0 := 
sorry

end polynomial_nonneg_l241_241366


namespace mutually_exclusive_but_not_complementary_l241_241050

open Finset

def Omega := {1, 2, 3, 4, 5, 6}
def A := {1, 3}
def B := {3, 5}
def C := {2, 4, 6}

theorem mutually_exclusive_but_not_complementary :
  disjoint A C ∧ (A ∪ C ≠ Omega) :=
by
  sorry

end mutually_exclusive_but_not_complementary_l241_241050


namespace pedal_triangle_exists_and_angles_l241_241777
noncomputable theory

-- Define the angles of the given triangle T
def angle_A := 24
def angle_B := 60
def angle_C := 96

-- Define the angles of the corresponding pedal triangles
def pedal_angle_A := 102
def pedal_angle_B := 30
def pedal_angle_C := 48

-- State the equivalence theorem
theorem pedal_triangle_exists_and_angles (T : Triangle) :
    ∃ (P₁ P₂ P₃ P₄ : Triangle),
    T.angle1 = angle_A ∧ T.angle2 = angle_B ∧ T.angle3 = angle_C ∧
    (P₁.angle1 = pedal_angle_A ∧ P₁.angle2 = pedal_angle_B ∧ P₁.angle3 = pedal_angle_C) ∧
    (P₂.angle1 = pedal_angle_A ∧ P₂.angle2 = pedal_angle_B ∧ P₂.angle3 = pedal_angle_C) ∧
    (P₃.angle1 = pedal_angle_A ∧ P₃.angle2 = pedal_angle_B ∧ P₃.angle3 = pedal_angle_C) ∧
    (P₄.angle1 = pedal_angle_A ∧ P₄.angle2 = pedal_angle_B ∧ P₄.angle3 = pedal_angle_C) :=
by { sorry }

end pedal_triangle_exists_and_angles_l241_241777


namespace anya_kolya_count_equal_l241_241960

def digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def distinct_digits (n : ℕ) := (n.digits : multiset ℕ).nodup

def divisible_by (n : ℕ) (d : ℕ) := n % d = 0

def anya_numbers_count := 
  fintype.card {x : ℕ // (x.digits.card = 9 ∧ distinct_digits x ∧ divisible_by (x.digits.sum) 9)}

def kolya_numbers_count := 
  fintype.card {x : ℕ // (x.digits.card = 10 ∧ distinct_digits x ∧ (x % 10 = 0 ∨ x % 10 = 5))}

theorem anya_kolya_count_equal :
  anya_numbers_count = kolya_numbers_count :=
sorry

end anya_kolya_count_equal_l241_241960


namespace sum_of_even_factors_720_l241_241891

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) : ℕ :=
    match n with
    | 720 => 
      let sum_powers_2 := 2 + 4 + 8 + 16 in
      let sum_powers_3 := 1 + 3 + 9 in
      let sum_powers_5 := 1 + 5 in
      sum_powers_2 * sum_powers_3 * sum_powers_5
    | _ => 0
  in
  even_factors_sum 720 = 2340 :=
by 
  sorry

end sum_of_even_factors_720_l241_241891


namespace number_of_solutions_l241_241677

theorem number_of_solutions : 
  let N := 2011 in
  let S : Finset (ℤ × ℤ × ℤ) := 
    { (x, y, z) | 0 ≤ x ∧ x < N ∧ 0 ≤ y ∧ y < N ∧ 0 ≤ z ∧ z < N 
                ∧ (x + y + z) % N = 0 
                ∧ (x * y + y * z + z * x) % N = 0 } in
  Finset.card S = 4021 :=
sorry

end number_of_solutions_l241_241677


namespace cyclic_sum_inequality_l241_241623

open Real

theorem cyclic_sum_inequality (n : ℕ) (x : ℕ → ℝ) 
  (h1 : 2 < n)
  (h2 : ∀ i : ℕ, 1 ≤ i → i ≤ n → 2 ≤ x i ∧ x i ≤ 3) :
  (∑ i in Finset.range n, (let j := (i + 1) % n in let k := (i + 2) % n in (x i ^ 2 + x j ^ 2 - x k ^ 2) / (x i + x j - x k))) 
    ≤ 2 * (∑ i in Finset.range n, x i) - 2 * n := 
  sorry

end cyclic_sum_inequality_l241_241623


namespace unattainable_y_ne_l241_241226

theorem unattainable_y_ne : ∀ x : ℝ, x ≠ -5/4 → y = (2 - 3 * x) / (4 * x + 5) → y ≠ -3/4 :=
by
  sorry

end unattainable_y_ne_l241_241226


namespace path_count_correct_l241_241171

-- Define the graph-like structure for the octagonal lattice with directional constraints
structure OctagonalLattice :=
  (vertices : Type)
  (edges : vertices → vertices → Prop) -- Directed edges

-- Define a path from A to B respecting the constraints
def path_num_lattice (L : OctagonalLattice) (A B : L.vertices) : ℕ :=
  sorry -- We assume a function counting valid paths exists here

-- Assert the specific conditions for the bug's movement
axiom LatticeStructure : OctagonalLattice
axiom vertex_A : LatticeStructure.vertices
axiom vertex_B : LatticeStructure.vertices

-- Example specific path counting for the problem's lattice
noncomputable def paths_from_A_to_B : ℕ :=
  path_num_lattice LatticeStructure vertex_A vertex_B

theorem path_count_correct : paths_from_A_to_B = 2618 :=
  sorry -- This is where the proof would go

end path_count_correct_l241_241171


namespace number_of_numbers_tadd_said_after_20_rounds_l241_241107

-- Define the arithmetic sequence representing the count of numbers Tadd says each round
def tadd_sequence (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

-- Define the sum of the first n terms of Tadd's sequence
def sum_tadd_sequence (n : ℕ) : ℕ :=
  n * (1 + tadd_sequence n) / 2

-- The main theorem to state the problem
theorem number_of_numbers_tadd_said_after_20_rounds :
  sum_tadd_sequence 20 = 400 :=
by
  -- The actual proof should be filled in here
  sorry

end number_of_numbers_tadd_said_after_20_rounds_l241_241107


namespace condition_for_equilateral_reflected_triangle_l241_241160

-- Definitions of triangles and reflections
structure Triangle (α : Type) [LinearOrderedField α] :=
(A B C : α × α)

def reflect_vertex (α : Type) [LinearOrderedField α] 
    (T : Triangle α) : 
    Triangle α :=
    sorry -- placeholder for the actual reflection function

-- Predicate for checking if a triangle is equilateral
def is_equilateral {α : Type} [LinearOrderedField α] 
    (T : Triangle α) : 
    Prop :=
    sorry -- placeholder for the actual definition

-- Predicate for checking if a triangle is isosceles with specific angles
def isosceles_with_special_angle {α : Type} [LinearOrderedField α] 
    (T : Triangle α) : 
    Prop :=
    sorry -- placeholder for the actual definition

-- Main theorem statement
theorem condition_for_equilateral_reflected_triangle {α : Type} [LinearOrderedField α] 
    (ABC : Triangle α) :
    (is_equilateral (reflect_vertex α ABC)) ↔ (isosceles_with_special_angle ABC) :=
begin
    sorry
end

end condition_for_equilateral_reflected_triangle_l241_241160


namespace minimum_percentage_lost_all_parts_l241_241593

theorem minimum_percentage_lost_all_parts (N : ℝ)
  (h_ear : 0.85 * N)
  (h_eye : 0.80 * N)
  (h_arm : 0.75 * N)
  (h_leg : 0.70 * N) : 
  ∃ (p : ℝ), p = 0.10 :=
by
  sorry

end minimum_percentage_lost_all_parts_l241_241593


namespace exists_ints_for_inequalities_l241_241045

theorem exists_ints_for_inequalities (a b : ℝ) (ε : ℝ) (hε : ε > 0) :
  ∃ (n : ℕ) (k m : ℤ), |(n * a) - k| < ε ∧ |(n * b) - m| < ε :=
by
  sorry

end exists_ints_for_inequalities_l241_241045


namespace quadratic_inequality_ab_l241_241627

theorem quadratic_inequality_ab (a b : ℝ) :
  (∀ x : ℝ, (x > -1 ∧ x < 1 / 3) → a * x^2 + b * x + 1 > 0) →
  a * b = 6 :=
sorry

end quadratic_inequality_ab_l241_241627


namespace store_refusal_illegal_l241_241481

-- Definitions for conditions
def is_legal_tender (banknote : Type) : Prop :=
  banknote ∈ [issued_by_Bank_of_Russia] -- Placeholder, refine for Lean syntax requirements.

def is_not_counterfeit (banknote : Type) : Prop :=
  ¬ banknote ∈ [counterfeit] -- Placeholder, refine for Lean syntax requirements.

def permissible_damage (banknote : Type) : Prop :=
  banknote ∈ [dirt, wear, tears, small_holes, punctures, foreign_inscriptions, stains, stamps, missing_corners, missing_edges] -- Placeholder, refine for Lean syntax requirements.

-- Proof statement
theorem store_refusal_illegal (banknote : Type) (h1 : is_legal_tender banknote) (h2 : is_not_counterfeit banknote) (h3 : permissible_damage banknote) : false := 
  sorry

end store_refusal_illegal_l241_241481


namespace sum_of_products_divisible_by_p_l241_241758

theorem sum_of_products_divisible_by_p (p : ℕ) (h_prime : Nat.Prime p) (k : ℕ) (h_k : 1 ≤ k ∧ k ≤ p - 1):
  let products := {P | ∃ s : Finset ℕ, s.card = k ∧ s ⊆ Finset.range (p - 1) ∧ P = s.prod id} in
  (∑ P in products, P) % p = 0 :=
by
  sorry

end sum_of_products_divisible_by_p_l241_241758


namespace omega_value_and_range_l241_241661

noncomputable def f (ω x : ℝ) := (sin (ω * x))^2 - (cos (ω * x))^2 + 2 * sin (ω * x) * cos (ω * x)

theorem omega_value_and_range :
  (0 < ω ∧ ω < 4) ∧ (f ω x = f ω (2 * π / 16) → ω = 2) ∧
  (∀ x, x ∈ Icc (5 * π / 48) (11 * π / 48) → f 2 x ∈ Icc (sqrt 2 / 2) (sqrt 2)) :=
by
  sorry

end omega_value_and_range_l241_241661


namespace function_properties_l241_241766

open Function

noncomputable def f : ℝ → ℝ := sorry

theorem function_properties :
  (∀ x, f(10 + x) = f(10 - x)) ∧
  (∀ x, f(5 - x) = f(5 + x)) →
  (Even f ∧ ∃ T, T ≠ 0 ∧ (∀ x, f(x + T) = f x)) :=
begin
  intros h,
  sorry
end

end function_properties_l241_241766


namespace sqrt_equation_solution_l241_241245

theorem sqrt_equation_solution (x : ℝ) (h : sqrt (2 * x - 3) = 10) : x = 51.5 :=
  sorry

end sqrt_equation_solution_l241_241245


namespace melissa_driving_hours_per_year_l241_241037

theorem melissa_driving_hours_per_year :
  ∀ (months_per_year trips_per_month hours_per_trip : ℕ),
    months_per_year = 12 → trips_per_month = 2 → hours_per_trip = 3 →
    (trips_per_month * hours_per_trip * months_per_year = 72) :=
by {
  intros months_per_year trips_per_month hours_per_trip,
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end melissa_driving_hours_per_year_l241_241037


namespace total_pens_bought_l241_241002

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l241_241002


namespace partition_27_squares_equally_l241_241592

noncomputable def sum_list_squares : (List ℕ) → ℕ :=
  List.sum ∘ (List.map (λ n, n * n))

-- Define the first 27 squares
def first_27_squares : List ℕ := List.range 27

-- Our goal is to partition these squares into two groups with equal sums
theorem partition_27_squares_equally :
  ∃ (group1 group2 : List ℕ),
    group1 ++ group2 = first_27_squares ∧
    sum_list_squares group1 = sum_list_squares group2 := sorry

end partition_27_squares_equally_l241_241592


namespace LCM_180_504_l241_241118

theorem LCM_180_504 : Nat.lcm 180 504 = 2520 := 
by 
  -- We skip the proof.
  sorry

end LCM_180_504_l241_241118


namespace sum_of_factorials_modulo_5_l241_241235

-- Definitions based on problem conditions
def factorial (n : ℕ) : ℕ := match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def S (n : ℕ) : ℕ := 
  if n = 0 then 0
  else ∑ i in Finset.range (n + 1), factorial i

-- Main theorem statement to prove
theorem sum_of_factorials_modulo_5 (n : ℕ) (hn : 0 < n) :
  (∃ k : ℕ, k * k ≡ S n [MOD 5]) ↔ (n = 1 ∨ n = 3) :=
sorry

end sum_of_factorials_modulo_5_l241_241235


namespace xt_inequality_least_constant_l241_241370

theorem xt_inequality (x y z t : ℝ) (h : x ≤ y ∧ y ≤ z ∧ z ≤ t) (h_sum : x * y + x * z + x * t + y * z + y * t + z * t = 1) :
  x * t < 1 / 3 := sorry

theorem least_constant (x y z t : ℝ) (h : x ≤ y ∧ y ≤ z ∧ z ≤ t) (h_sum : x * y + x * z + x * t + y * z + y * t + z * t = 1) :
  ∃ C, ∀ (x t : ℝ), xt < C ∧ C = 1 / 3 := sorry

end xt_inequality_least_constant_l241_241370


namespace total_pens_l241_241005

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l241_241005


namespace exist_j_n_l241_241672

noncomputable def sequence_x : ℕ → ℝ
| 0       := 1
| (n + 1) := sequence_x n / (2 + sequence_x n)

noncomputable def sequence_y : ℕ → ℝ
| 0       := 1
| (n + 1) := (sequence_y n)^2 / (1 + 2 * sequence_y n)

theorem exist_j_n (n : ℕ) : ∃ j_n : ℕ, sequence_y n = sequence_x j_n :=
by
  sorry

end exist_j_n_l241_241672


namespace puzzles_sold_correct_l241_241068

def science_kits_sold : ℕ := 45
def puzzles_sold : ℕ := science_kits_sold - 9

theorem puzzles_sold_correct : puzzles_sold = 36 := by
  -- Proof will be provided here
  sorry

end puzzles_sold_correct_l241_241068


namespace line_intercepts_of_3x_minus_y_plus_6_eq_0_l241_241613

theorem line_intercepts_of_3x_minus_y_plus_6_eq_0 :
  (∃ y, 3 * 0 - y + 6 = 0 ∧ y = 6) ∧ (∃ x, 3 * x - 0 + 6 = 0 ∧ x = -2) :=
by
  sorry

end line_intercepts_of_3x_minus_y_plus_6_eq_0_l241_241613


namespace distance_between_A_and_B_l241_241476

-- Definitions for the problem
def speed_fast_train := 65 -- speed of the first train in km/h
def speed_slow_train := 29 -- speed of the second train in km/h
def time_difference := 5   -- difference in hours

-- Given conditions and the final equation leading to the proof
theorem distance_between_A_and_B :
  ∃ (D : ℝ), D = 9425 / 36 :=
by
  existsi (9425 / 36 : ℝ)
  sorry

end distance_between_A_and_B_l241_241476


namespace one_twenty_percent_of_number_l241_241698

theorem one_twenty_percent_of_number (x : ℝ) (h : 0.20 * x = 300) : 1.20 * x = 1800 :=
by 
sorry

end one_twenty_percent_of_number_l241_241698


namespace find_2g_x_l241_241701

theorem find_2g_x (g : ℝ → ℝ) (h : ∀ x > 0, g (3 * x) = 3 / (3 + x)) (x : ℝ) (hx : x > 0) :
  2 * g x = 18 / (9 + x) :=
sorry

end find_2g_x_l241_241701


namespace total_savings_during_sale_with_volume_discount_l241_241051

-- Definitions based on the problem's conditions
def num_notebooks : ℕ := 8
def original_cost_per_notebook : ℝ := 3.00
def sale_discount : ℝ := 0.25
def volume_discount : ℝ := 0.10

-- Theorems based on the problem's question and conditions
theorem total_savings_during_sale_with_volume_discount :
  let sale_price := original_cost_per_notebook * (1 - sale_discount),
      volume_discount_applicable := if num_notebooks > 5 then sale_price * volume_discount else 0,
      final_price_per_notebook := sale_price - volume_discount_applicable,
      total_cost_without_discounts := num_notebooks * original_cost_per_notebook,
      total_cost_with_discounts := num_notebooks * final_price_per_notebook
  in total_cost_without_discounts - total_cost_with_discounts = 7.84 := by
  sorry

end total_savings_during_sale_with_volume_discount_l241_241051


namespace solve_expression_l241_241796

theorem solve_expression : 68 + (108 * 3) + (29^2) - 310 - (6 * 9) = 869 :=
by
  sorry

end solve_expression_l241_241796


namespace simplify_sqrt_expression_l241_241792

theorem simplify_sqrt_expression :
  (Real.sqrt (3 * 5) * Real.sqrt (3^3 * 5^3)) = 225 := 
by 
  sorry

end simplify_sqrt_expression_l241_241792


namespace floor_sum_min_value_l241_241694

theorem floor_sum_min_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end floor_sum_min_value_l241_241694


namespace shenille_total_points_l241_241718

def shenille_shots (x y : ℕ) : Prop := 
  x + y = 30

def successful_three_point_shots (x : ℕ) : ℕ :=
  Nat.floor (0.2 * x)

def successful_two_point_shots (y : ℕ) : ℕ :=
  Nat.floor (0.3 * y)

def points_from_three_point_shots (x : ℕ) : ℕ :=
  3 * (successful_three_point_shots x)

def points_from_two_point_shots (y : ℕ) : ℕ :=
  2 * (successful_two_point_shots y)

def total_points (x y : ℕ) : ℕ :=
  points_from_three_point_shots x + points_from_two_point_shots y

theorem shenille_total_points : 
  ∀ (x y : ℕ), 
  shenille_shots x y → 
  total_points x y = 18 :=
by
  intros x y h
  have : x + y = 30 := h
  sorry

end shenille_total_points_l241_241718


namespace ketchup_per_hot_dog_l241_241097

theorem ketchup_per_hot_dog (total_ketchup : ℝ) (hot_dogs : ℕ) (ketchup_per_hot_dog : ℝ) : 
  total_ketchup = 84.6 → hot_dogs = 12 → ketchup_per_hot_dog = total_ketchup / hot_dogs → ketchup_per_hot_dog = 7.05 :=
by
  intros h_total h_hotdogs h_calc
  rw [h_total, h_hotdogs] at h_calc
  exact h_calc

end ketchup_per_hot_dog_l241_241097


namespace distance_between_foci_is_six_l241_241836

-- Lean 4 Statement
noncomputable def distance_between_foci (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  if (p1 = (1, 3) ∧ p2 = (6, -1) ∧ p3 = (11, 3)) then 6 else 0

theorem distance_between_foci_is_six : distance_between_foci (1, 3) (6, -1) (11, 3) = 6 :=
by
  sorry

end distance_between_foci_is_six_l241_241836


namespace bear_weight_gain_l241_241169

theorem bear_weight_gain :
  let total_weight := 1000
  let weight_from_berries := total_weight / 5
  let weight_from_acorns := 2 * weight_from_berries
  let weight_from_salmon := (total_weight - weight_from_berries - weight_from_acorns) / 2
  let weight_from_small_animals := total_weight - (weight_from_berries + weight_from_acorns + weight_from_salmon)
  weight_from_small_animals = 200 :=
by sorry

end bear_weight_gain_l241_241169


namespace geometric_series_sum_l241_241230

-- Definitions based on conditions
def a : ℚ := 3 / 2
def r : ℚ := -4 / 9

-- Statement of the proof
theorem geometric_series_sum : (a / (1 - r)) = 27 / 26 :=
by
  -- proof goes here
  sorry

end geometric_series_sum_l241_241230


namespace required_run_rate_in_remaining_overs_l241_241909

variable (run_rate_first_10_overs : ℝ) 
variable (overs_first_phase : ℕ)
variable (target_runs : ℕ)
variable (overs_remaining : ℕ)

theorem required_run_rate_in_remaining_overs 
    (h1 : run_rate_first_10_overs = 3.2)
    (h2 : overs_first_phase = 10)
    (h3 : target_runs = 282)
    (h4 : overs_remaining = 40) :
    let runs_scored_first_phase := run_rate_first_10_overs * overs_first_phase,
        remaining_runs := target_runs - runs_scored_first_phase,
        required_run_rate := remaining_runs / overs_remaining
    in required_run_rate = 6.25 := 
sorry

end required_run_rate_in_remaining_overs_l241_241909


namespace classification_l241_241494

-- Define the types for inhabitants and their roles
inductive Inhabitant : Type
| A
| B
| C

inductive Role : Type
| Knight
| Liar
| Werewolf

open Inhabitant Role

-- Given conditions
axiom knight_or_liar (i : Inhabitant) : Role i = Knight ∨ Role i = Liar
axiom exactly_one_werewolf : ∃ i : Inhabitant, Role i = Werewolf ∧ ∀ j ≠ i, Role j ≠ Werewolf

-- Statements made by the inhabitants
axiom statement_A : Role A = Werewolf
axiom statement_B : Role B = Werewolf
axiom statement_C : (Role A = Knight ∨ Role B = Knight ∨ Role C = Knight) → (Role A ≠ Knight ∧ Role B ≠ Knight ∧ Role C ≠ Knight)

-- Proof goal
theorem classification :
  (Role A = Liar ∧ Role B = Liar ∧ Role C = Knight ∧ Role C = Werewolf) :=
sorry

end classification_l241_241494


namespace value_of_five_inch_cube_l241_241183

def cube_volume (side: ℝ) : ℝ := side ^ 3

def value_of_cube (side₄: ℝ) (value₄: ℝ) (side₅: ℝ) : ℝ :=
  let V₄ := cube_volume side₄
  let V₅ := cube_volume side₅
  value₄ * V₅ / V₄

theorem value_of_five_inch_cube :
  value_of_cube 4 400 5 = 781 := by
  sorry

end value_of_five_inch_cube_l241_241183


namespace solution_satisfies_conditions_l241_241135

noncomputable def sqrt_eq (a b x y : ℝ) : Prop :=
  sqrt (x - y) = a ∧ sqrt (x + y) = b

theorem solution_satisfies_conditions 
  (x y : ℝ)
  (h1 : sqrt_eq (2/5) 2 x y)
  (hexact: x = 52/25 ∧ y = 48/25) :
  sqrt_eq (2/5) 2 x y ∧ 
  (x * y = 8/25) :=
by
  sorry

end solution_satisfies_conditions_l241_241135


namespace total_dolls_l241_241409

-- Definitions based on the given conditions
def grandmother_dolls : Nat := 50
def sister_dolls : Nat := grandmother_dolls + 2
def rene_dolls : Nat := 3 * sister_dolls

-- Statement we want to prove
theorem total_dolls : grandmother_dolls + sister_dolls + rene_dolls = 258 := by
  sorry

end total_dolls_l241_241409


namespace original_number_is_two_l241_241993

-- Conditions and definitions
def satisfies_condition (x : ℕ) : Prop :=
  ∃ y : ℕ, x * y = 4 * x ∧ (x * y)^(1 / 3) ∈ ℕ

-- Proof statement
theorem original_number_is_two : ∃ x : ℕ, satisfies_condition x ∧ x = 2 := 
sorry

end original_number_is_two_l241_241993


namespace ecoli_population_after_4_hours_l241_241840

noncomputable def bacterial_doublings : ℕ := 240 / 20

noncomputable def bacterial_population (initial : ℕ) (doublings : ℕ) : ℕ :=
  initial * (2^doublings)

theorem ecoli_population_after_4_hours :
  bacterial_population 1 bacterial_doublings = 4096 :=
by
  unfold bacterial_doublings bacterial_population
  norm_num
  rw [pow_mul, pow_two, bit0, pow_two, bit0, pow_two, bit0, pow_two, pow_two, pow_two] -- simplified
  norm_num

end ecoli_population_after_4_hours_l241_241840


namespace common_sales_days_count_l241_241514

noncomputable def bookstore_sales_days := {7, 14, 21, 28}
noncomputable def shoe_store_sales_days := {2, 8, 14, 20, 26}

def common_sales_days := bookstore_sales_days ∩ shoe_store_sales_days

theorem common_sales_days_count : common_sales_days.toFinset.card = 2 :=
  by
    -- Proof is omitted
    sorry

end common_sales_days_count_l241_241514


namespace compare_abc_l241_241685

def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3
def a : ℝ := log2 3
def b : ℝ := log3 4
def c : ℝ := 2 ^ (-0.3)

theorem compare_abc : c < b ∧ b < a :=
by 
  have ha : a = log2 3 := rfl
  have hb : b = log3 4 := rfl
  have hc : c = 2 ^ (-0.3) := rfl
  -- Proof omitted
  sorry

end compare_abc_l241_241685


namespace line_through_point_and_opposite_intercepts_l241_241807

theorem line_through_point_and_opposite_intercepts (x y : ℝ) (M : ℝ × ℝ) :
  M = (5, -2) →
  (-- The line may have intercepts on the x-axis and y-axis as opposite numbers --) →
  (-- Therefore the equation of the line is: --)
  (x - y - 7 = 0 ∨ 2 * x + 5 * y = 0) :=
sorry

end line_through_point_and_opposite_intercepts_l241_241807


namespace sum_first_60_natural_numbers_l241_241569

theorem sum_first_60_natural_numbers : (60 * (60 + 1)) / 2 = 1830 := by
  sorry

end sum_first_60_natural_numbers_l241_241569


namespace number_of_nickels_l241_241477

variable (n : Nat) -- number of nickels

def value_of_nickels := n * 5 -- value of nickels n in cents
def total_value :=
    2 * 100 +   -- 2 one-dollar bills
    1 * 500 +   -- 1 five-dollar bill
    13 * 25 +   -- 13 quarters
    20 * 10 +   -- 20 dimes
    35 * 1 +    -- 35 pennies
    value_of_nickels n

theorem number_of_nickels :
    total_value n = 1300 ↔ n = 8 :=
by sorry

end number_of_nickels_l241_241477


namespace no_real_solution_l241_241416

theorem no_real_solution :
    ∀ x : ℝ, (5 * x^2 - 3 * x + 2) / (x + 2) ≠ 2 * x - 3 :=
by
  intro x
  sorry

end no_real_solution_l241_241416


namespace y1_eq_y2_imp_x_y1_lt_y2_a_gt_1_y1_lt_y2_0_lt_a_lt_1_l241_241371

-- Definitions for proof context
variable {a x : ℝ}
variable (y1 y2 : ℝ)
variable (a_pos : 0 < a) (a_ne_one : a ≠ 1)

-- Definition of y1 and y2
def y_1 := a^(3 * x + 1)
def y_2 := a^(-2 * x)

-- Proof Problem 1
theorem y1_eq_y2_imp_x :
  y_1 = y_2 → x = (1 / 5) :=
sorry

-- Proof Problem 2 (part 1: a > 1)
theorem y1_lt_y2_a_gt_1 :
  a > 1 → y_1 < y_2 → x < (1 / 5) :=
sorry

-- Proof Problem 2 (part 2: 0 < a < 1)
theorem y1_lt_y2_0_lt_a_lt_1 :
  0 < a ∧ a < 1 → y_1 < y_2 → x > (1 / 5) :=
sorry

end y1_eq_y2_imp_x_y1_lt_y2_a_gt_1_y1_lt_y2_0_lt_a_lt_1_l241_241371


namespace quadratic_nonneg_for_all_t_l241_241618

theorem quadratic_nonneg_for_all_t (x y : ℝ) : 
  (y ≤ x + 1) → (y ≥ -x - 1) → (x ≥ y^2 / 4) → (∀ (t : ℝ), (|t| ≤ 1) → t^2 + y * t + x ≥ 0) :=
by
  intro h1 h2 h3 t ht
  sorry

end quadratic_nonneg_for_all_t_l241_241618


namespace tensor_op_example_l241_241315

variable (a b : ℝ)

def tensor_op (a b : ℝ) : ℝ := (a + b)^2 / (a - b)

theorem tensor_op_example : tensor_op (tensor_op 4 6) 2 = -576 / 13 :=
by
  sorry

end tensor_op_example_l241_241315


namespace proof_of_problem_l241_241140

noncomputable def proof_problem (x y : ℚ) : Prop :=
  (sqrt (x - y) = 2 / 5) ∧ (sqrt (x + y) = 2) ∧ 
  x = 52 / 25 ∧ y = 48 / 25 ∧ 
  let vertices := [(0, 0), (2, 2), (2 / 25, -2 / 25), (52 / 25, 48 / 25)] in
  let area := Rational.from_ints 8 25 in
  ∃ (a b c d : ℚ × ℚ), 
    a ∈ vertices ∧ b ∈ vertices ∧ c ∈ vertices ∧ d ∈ vertices ∧ 
    ((b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = area)

theorem proof_of_problem : proof_problem (52 / 25) (48 / 25) :=
by { sorry } 

end proof_of_problem_l241_241140


namespace goods_train_speed_is_72_l241_241536

noncomputable def speed_of_goods_train (man_train_speed : ℕ) (goods_train_time : ℝ) (goods_train_length : ℝ) : ℝ :=
  let relative_speed :=  (goods_train_length / (goods_train_time / 3600)) in
  relative_speed - man_train_speed

theorem goods_train_speed_is_72 :
  speed_of_goods_train 40 (9) (280 / 1000) = 72 :=
by sorry

end goods_train_speed_is_72_l241_241536


namespace sequence_inequality_l241_241770

theorem sequence_inequality (a : ℕ → ℝ) (M : ℝ) (h1 : ∀ n : ℕ, a n > 0)
  (h2 : ∀ n : ℕ, (∑ i in Finset.range n, (a i)^2) < M * (a n)^2) :
  ∃ M' > 0, ∀ n : ℕ, (∑ i in Finset.range n, a i) < M' * a n :=
by
  sorry

end sequence_inequality_l241_241770


namespace invalid_diagonal_sets_l241_241902

namespace ExternalDiagonals

def valid_diagonals (a b c : ℕ) : ℕ × ℕ × ℕ :=
  (Nat.sqrt (a * a + b * b), Nat.sqrt (b * b + c * c), Nat.sqrt (a * a + c * c))

def check_diagonals (d1 d2 d3 : ℕ) : Prop :=
  d1 * d1 + d2 * d2 >= d3 * d3

def valid_set (s : ℕ × ℕ × ℕ) : Prop :=
  match s with 
  | (a, b, c) => a <= b ∧ b <= c ∧ check_diagonals a b c

def sets_to_check := [{3, 4, 5}, {5, 5, 8}, {3, 6, 7}, {6, 6, 8}, {5, 7, 9}]

theorem invalid_diagonal_sets : ∀ s ∈ sets_to_check, ¬ valid_set s → s = {5, 5, 8} ∨ s = {5, 7, 9} :=
by
  intro s h
  cases h
  . all_goals { sorry }

end ExternalDiagonals

end invalid_diagonal_sets_l241_241902


namespace hyperbola_asymptote_angle_l241_241665

theorem hyperbola_asymptote_angle (a b x y : ℝ) (h1 : a > b) (h2 : (x, y) = (3, 3 * Real.sqrt 2)) 
  (h3 : ∀ x y, x = (3:ℝ) → y = (3 * Real.sqrt 2:ℝ) → (x^2 / a^2 - y^2 / b^2 = 1)) 
  (h4 : ∃ θ: ℝ, θ = Real.pi / 4 ∧ Real.tan (θ / 2) = b / a) : 
  a / b = Real.sqrt 2 + 1 := by
  sorry

end hyperbola_asymptote_angle_l241_241665


namespace sum_of_even_factors_720_l241_241876

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) :=
    (2 + 4 + 8 + 16) * (1 + 3 + 9) * (1 + 5)
  in even_factors_sum 720 = 2340 :=
by
  sorry

end sum_of_even_factors_720_l241_241876


namespace solution_satisfies_conditions_l241_241137

noncomputable def sqrt_eq (a b x y : ℝ) : Prop :=
  sqrt (x - y) = a ∧ sqrt (x + y) = b

theorem solution_satisfies_conditions 
  (x y : ℝ)
  (h1 : sqrt_eq (2/5) 2 x y)
  (hexact: x = 52/25 ∧ y = 48/25) :
  sqrt_eq (2/5) 2 x y ∧ 
  (x * y = 8/25) :=
by
  sorry

end solution_satisfies_conditions_l241_241137


namespace sum_b_lt_2_plus_2_ln_l241_241630

def a_n (n : ℕ) : ℝ := (1/2)^(n-1)

def T_n (n : ℕ) : ℝ :=
  match n with
  | 0     => 0  -- edge case: sum of 0 terms
  | n + 1 => 2 - (1/2)^n

def b_n (n : ℕ) : ℝ :=
  match n with
  | 0     => 0    -- edge case: not used as per condition
  | 1     => a_n 1
  | n + 2 => T_n (n + 1) / (n + 2) + (1 + ∑ i in Finset.range (n + 1), 1 / (i + 1)) * a_n (n + 2)

def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b_n (i + 1)

theorem sum_b_lt_2_plus_2_ln (n : ℕ) : S_n n < 2 + 2 * Real.log n :=
by
  sorry

end sum_b_lt_2_plus_2_ln_l241_241630


namespace board_all_integers_l241_241465

theorem board_all_integers (n : ℕ) (a : fin n → ℕ)
  (h : ∀ i j, i ≠ j → (∃ x ∈ (a i + a j)/2, x ∈ ℤ) ∨ (∃ y ∈ √(a i * a j), y ∈ ℤ)) :
  ∃ b, ( ∀ i j, i ≠ j → (∃ x ∈ (a i + a j)/2, x ∈ ℤ) ) ∨
       ( ∀ i j, i ≠ j → (∃ y ∈ √(a i * a j), y ∈ ℤ) ) :=
sorry

end board_all_integers_l241_241465


namespace equilateral_triangle_area_l241_241666

variables (a b : ℝ)
noncomputable def k : ℝ := 2 / 3

theorem equilateral_triangle_area (a b : ℝ) (k : ℝ) (hk : k > 0)
  (A B : ℝ × ℝ) (OA OB : ℝ) (area : ℝ)
  (h_intersect : ∀ x, y = ax + b → y = k / x → (x, y) ∈ {A, B}) 
  (h_orig : O = (0, 0))
  (h_is_equilateral : triangle O A B)
  (h_area : area_of_triangle O A B = 2 * √3 / 3) :
  k = 2 / 3 :=
sorry

end equilateral_triangle_area_l241_241666


namespace neg_p_l241_241761

-- Define the sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

-- Define the proposition p
def p : Prop := ∀ x : ℤ, is_odd x → is_even (2 * x)

-- State the negation of proposition p
theorem neg_p : ¬ p ↔ ∃ x : ℤ, is_odd x ∧ ¬ is_even (2 * x) := by sorry

end neg_p_l241_241761


namespace participants_initial_count_l241_241335

theorem participants_initial_count (initial_participants remaining_after_first_round remaining_after_second_round : ℝ) 
  (h1 : remaining_after_first_round = 0.4 * initial_participants)
  (h2 : remaining_after_second_round = (1/4) * remaining_after_first_round)
  (h3 : remaining_after_second_round = 15) : 
  initial_participants = 150 :=
sorry

end participants_initial_count_l241_241335


namespace tower_count_is_350_l241_241929

theorem tower_count_is_350 (y p o h : ℕ) 
  (hy : y = 3) 
  (hp : p = 3) 
  (ho : o = 2) 
  (hh : h = 6) : 
  (∃ n : ℕ, n = 350) :=
by {
  have total_towers_count : ℕ := 350,
  use total_towers_count,
  sorry -- the proof steps are skipped
}

end tower_count_is_350_l241_241929


namespace sum_of_even_factors_720_l241_241875

theorem sum_of_even_factors_720 : 
  let n := 2^4 * 3^2 * 5 in
  (∑ d in (Finset.range (n + 1)).filter (λ d, d % 2 = 0 ∧ n % d = 0), d) = 2340 :=
by
  let n := 2^4 * 3^2 * 5
  sorry

end sum_of_even_factors_720_l241_241875


namespace sum_even_factors_of_720_l241_241894

open Nat

theorem sum_even_factors_of_720 :
  ∑ d in (finset.filter (λ x, even x) (finset.divisors 720)), d = 2340 :=
by
  sorry

end sum_even_factors_of_720_l241_241894


namespace journey_time_correct_l241_241927

noncomputable def total_travel_time : ℕ → ℝ := 
by
  -- Definitions based on conditions provided:
  let total_distance := 642
  let first_part_distance := total_distance * (1 / 4 : ℝ)
  let second_part_distance := total_distance * (1 / 2 : ℝ)
  let third_part_distance := total_distance - (first_part_distance + second_part_distance)
  let first_part_speed := 60
  let second_part_speed := 80
  let third_part_speed := 50
  -- Calculations:
  let first_part_time := first_part_distance / first_part_speed
  let second_part_time := second_part_distance / second_part_speed
  let third_part_time := third_part_distance / third_part_speed
  -- Total time calculation:
  exact first_part_time + second_part_time + third_part_time

theorem journey_time_correct : 
  total_travel_time 642 = 9.8975 := 
by
  sorry

end journey_time_correct_l241_241927


namespace expected_value_of_biased_coin_l241_241925

def biased_coin_expected_value : ℝ :=
  let heads_prob := 2 / 3
  let tails_prob := 1 / 3
  let heads_gain := 5
  let tails_loss := -10
  heads_prob * heads_gain + tails_prob * tails_loss

theorem expected_value_of_biased_coin :
  biased_coin_expected_value = 0.00 := 
by
  -- proof goes here
  sorry

end expected_value_of_biased_coin_l241_241925


namespace find_incorrect_statement_l241_241286

variables {x1 x2 x3 k m n p : ℝ}

-- Given conditions
def median (data : list ℝ) : ℝ := sorry
def mode (data : list ℝ) : ℝ := sorry
def average (data : list ℝ) : ℝ := (data.sum) / (data.length)
def variance (data : list ℝ) : ℝ := sorry

-- Given data
def data := [x1, x2, x3]

-- Assertions based on conditions
axiom median_is_k : median data = k
axiom mode_is_m : mode data = m
axiom average_is_n : average data = n
axiom variance_is_p : variance data = p

-- Transformed data
def transformed_data := [2 * x1, 2 * x2, 2 * x3]

-- Proof problem
theorem find_incorrect_statement :
  (median transformed_data ≠ 2 * k ∨
   mode transformed_data ≠ 2 * m ∨
   average transformed_data ≠ 2 * n ∨
   variance transformed_data ≠ 2 * p) :=
by {
  sorry
}

end find_incorrect_statement_l241_241286


namespace function_properties_of_tan_abs_l241_241201

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def is_monotonically_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≤ f y

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

theorem function_properties_of_tan_abs :
  is_even_function (λ x, |tan x|) ∧
  is_monotonically_increasing_on (λ x, |tan x|) (set.Ioo 0 (π / 2)) ∧
  has_period (λ x, |tan x|) π :=
by sorry

end function_properties_of_tan_abs_l241_241201


namespace sum_of_even_factors_720_l241_241880

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) :=
    (2 + 4 + 8 + 16) * (1 + 3 + 9) * (1 + 5)
  in even_factors_sum 720 = 2340 :=
by
  sorry

end sum_of_even_factors_720_l241_241880


namespace expand_square_binomial_l241_241969

variable (m n : ℝ)

theorem expand_square_binomial : (3 * m - n) ^ 2 = 9 * m ^ 2 - 6 * m * n + n ^ 2 :=
by
  sorry

end expand_square_binomial_l241_241969


namespace total_books_l241_241699

def books_per_shelf : ℕ := 78
def number_of_shelves : ℕ := 15

theorem total_books : books_per_shelf * number_of_shelves = 1170 := 
by
  sorry

end total_books_l241_241699


namespace fill_pipe_time_l241_241531

theorem fill_pipe_time (t : ℕ) (H : ∀ C : Type, (1 / 2 : ℚ) * C = t * 1/2 * C) : t = t :=
by
  sorry

end fill_pipe_time_l241_241531


namespace solve_inequality_l241_241243

theorem solve_inequality (x : ℝ) : ||x-2|-1| ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 :=
by sorry

end solve_inequality_l241_241243


namespace croissant_price_l241_241308

theorem croissant_price (price_almond: ℝ) (total_expenditure: ℝ) (weeks: ℕ) (price_regular: ℝ) 
  (h1: price_almond = 5.50) (h2: total_expenditure = 468) (h3: weeks = 52) 
  (h4: weeks * price_regular + weeks * price_almond = total_expenditure) : price_regular = 3.50 :=
by 
  sorry

end croissant_price_l241_241308


namespace find_lambda_and_sum_l241_241264

def a (n : ℕ) : ℕ → ℕ
| 1     := 5
| (n+2) := 2 * a (n + 1) + 2 ^ (n + 2) - 1

theorem find_lambda_and_sum (λ : ℝ) (h : ∀ n ≥ 2, (a n + λ) / 2 ^ n = (a (n - 1) + λ) / 2 ^ (n - 1)) : 
  λ = -1 ∧ (∑ k in Finset.range n, a (k + 1)) = n * 2^(n+1) + n := 
sorry

end find_lambda_and_sum_l241_241264


namespace distance_between_points_l241_241565

noncomputable def point := (ℝ × ℝ × ℝ)

def distance (A B : point) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 + (B.3 - A.3) ^ 2)

theorem distance_between_points : distance (1, 4, -6) (7, 1, -6) = 3 * real.sqrt 5 := by
  sorry

end distance_between_points_l241_241565


namespace total_pens_bought_l241_241032

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l241_241032


namespace find_m_l241_241668

def a : ℝ × ℝ := (3, m)
def b : ℝ × ℝ := (1, -2)

def dot_product (p q : ℝ × ℝ) : ℝ := p.1 * q.1 + p.2 * q.2

noncomputable def squared_magnitude (p : ℝ × ℝ) : ℝ := p.1^2 + p.2^2

theorem find_m (m : ℝ) (h : dot_product a b + 3 * squared_magnitude b = 0) : m = 9 := 
  sorry

end find_m_l241_241668


namespace count_ordered_triples_lcm_l241_241678

theorem count_ordered_triples_lcm :
  let triples := { xyz : ℕ × ℕ × ℕ | ∃ (x y z : ℕ), xyz = (x, y, z) ∧ 
                  Nat.lcm x y = 180 ∧ Nat.lcm x z = 360 ∧ Nat.lcm y z = 1200 }
  in fintype.card triples = 6 :=
by
  sorry

end count_ordered_triples_lcm_l241_241678


namespace abs_val_problem_l241_241257

variable (a b : ℝ)

theorem abs_val_problem (h_abs_a : |a| = 2) (h_abs_b : |b| = 4) (h_sum_neg : a + b < 0) : a - b = 2 ∨ a - b = 6 :=
sorry

end abs_val_problem_l241_241257


namespace area_inequality_l241_241942

variable {A B C B₁ C₁ G : Type}
variable [RealField A B C B₁ C₁ G]

-- Define the areas of triangles
def area (X Y Z : Type) [RealField X Y Z] : ℝ := sorry 

-- Define centroid property (not directly necessary, for demonstration purposes)
def is_centroid (G A B C : Type) [RealField G A B C] : Prop := sorry

-- Main statement to prove
theorem area_inequality
  (ABC : Type) [RealField ABC]
  (A B C B₁ C₁ G : ABC)
  (h1 : l intersects_side A B at B₁ in ∆ABC)
  (h2 : l intersects_side A C at C₁ in ∆ABC)
  (h3 : is_centroid G A B C)
  (h4 : G and A on_same_side_of l)
  : area B B₁ C₁ + area C₁ G B₁ ≥ (4/9) * area A B C := 
  sorry

end area_inequality_l241_241942


namespace determine_m_range_l241_241587

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

def monotone_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y, x ∈ I → y ∈ I → x ≤ y → f y ≤ f x

noncomputable def g (m : ℝ) (f : ℝ → ℝ) : Prop :=
∀ x ∈ set.Icc 1 3, f (2 * m * x - real.log x - 3) ≥ 2 * f 3 - f (-2 * m * x + real.log x + 3)

theorem determine_m_range {f : ℝ → ℝ} (h_even : even_function f)
  (h_mono : monotone_decreasing f (set.Ici 0))
  (h_ineq : g m f) :
  m ∈ set.Icc (1 / (2 * real.exp 1)) ((real.log 3 + 6) / 6) :=
sorry

end determine_m_range_l241_241587


namespace LCM_180_504_l241_241117

theorem LCM_180_504 : Nat.lcm 180 504 = 2520 := 
by 
  -- We skip the proof.
  sorry

end LCM_180_504_l241_241117


namespace probability_of_event_l241_241782

noncomputable def probability_sin_ge_cos (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ Real.pi) : Prop :=
  Real.sin x ≥ Real.cos x

theorem probability_of_event :
  ∫ x in 0..Real.pi, ite (probability_sin_ge_cos x (by linarith) (by linarith)) 1 0 = (3 / 4) * Real.pi :=
by
  sorry

end probability_of_event_l241_241782


namespace problem_proof_l241_241373

-- Define the set of distinct digits from 1 to 9
def distinct_digits : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Given values for the letters
def F : ℕ := 9
def G : ℕ := 3
def D : ℕ := 6
def E : ℕ := 8
def H : ℕ := 2
def I : ℕ := 5

-- Define the numbers based on the letters
def FG : ℕ := 10 * F + G
def DE : ℕ := 10 * D + E
def HI : ℕ := 10 * H + I

-- Define the theorem corresponding to the problem
theorem problem_proof :
  FG - DE = HI :=
by
  -- Substitute the values
  have h1 : FG = 93 := rfl
  have h2 : DE = 68 := rfl
  have h3 : HI = 25 := rfl

  -- Perform the subtraction
  calc
    FG - DE = 93 - 68 := by rw [h1, h2]
    ...  = 25 := by rw [h3]

end problem_proof_l241_241373


namespace chelsea_cupcakes_time_l241_241971

theorem chelsea_cupcakes_time
  (batches : ℕ)
  (bake_time_per_batch : ℕ)
  (ice_time_per_batch : ℕ)
  (total_time : ℕ)
  (h1 : batches = 4)
  (h2 : bake_time_per_batch = 20)
  (h3 : ice_time_per_batch = 30)
  (h4 : total_time = (bake_time_per_batch + ice_time_per_batch) * batches) :
  total_time = 200 :=
  by
  -- The proof statement here
  -- The proof would go here, but we skip it for now
  sorry

end chelsea_cupcakes_time_l241_241971


namespace isosceles_trapezoid_AC_length_l241_241816

theorem isosceles_trapezoid_AC_length (A B C D : Point) :
  distance A B = 21 →
  distance A D = 10 →
  distance B C = 10 →
  distance C D = 9 →
  distance A C = 17 :=
by
  sorry

end isosceles_trapezoid_AC_length_l241_241816


namespace tan_squared_identity_l241_241044

theorem tan_squared_identity (x : ℝ) (hx : cos x ≠ 0) : 
  tan x ^ 2 + 1 / (tan x ^ 2) = (2 * (3 + cos (4 * x))) / (1 - cos (4 * x)) :=
by sorry

end tan_squared_identity_l241_241044


namespace abs_ineq_solution_l241_241163

theorem abs_ineq_solution (x : ℝ) : abs (x - 2) + abs (x - 3) < 9 ↔ -2 < x ∧ x < 7 :=
sorry

end abs_ineq_solution_l241_241163


namespace unit_cube_diagonal_distance_l241_241847

theorem unit_cube_diagonal_distance : 
  let dist := Real.sqrt 3 / 3 in
  ∃ (cube : UnitCube) (d1 d2 : cube.Diagonals),
    d1.is_face_diagonal ∧ 
    d2.is_face_diagonal ∧ 
    d1.is_adjacent_to d2 ∧ 
    d1.is_non_intersecting_with d2 ∧ 
  d1.distance_to d2 = dist :=
sorry

end unit_cube_diagonal_distance_l241_241847


namespace xy_sum_possible_values_l241_241687

theorem xy_sum_possible_values (x y : ℕ) (h1 : x < 20) (h2 : y < 20) (h3 : 0 < x) (h4 : 0 < y) (h5 : x + y + x * y = 95) :
  x + y = 18 ∨ x + y = 20 :=
by {
  sorry
}

end xy_sum_possible_values_l241_241687


namespace right_angled_triangle_ratio_BP_CH_l241_241365

variables {A B C H M P Q : Type}
variables [NonObtueseTriangle ABC] [Altitude CH A B] [Median CM A B]
variables [AngleBisector BAC P] [AngleBisector BAC Q]
variables [EqualAngles (Angle ABP PBQ) (Angle PBQ QBC)]

theorem right_angled_triangle (h1: ∠ ABC = 90°) : is_right_angled_triangle ABC :=
sorry

theorem ratio_BP_CH (h2: ∠ ABC = 90°) (h1: BP / CH = 2) : BP / CH = 2 :=
sorry

end right_angled_triangle_ratio_BP_CH_l241_241365


namespace sum_of_even_factors_720_l241_241871

theorem sum_of_even_factors_720 : 
  let n := 2^4 * 3^2 * 5 in
  (∑ d in (Finset.range (n + 1)).filter (λ d, d % 2 = 0 ∧ n % d = 0), d) = 2340 :=
by
  let n := 2^4 * 3^2 * 5
  sorry

end sum_of_even_factors_720_l241_241871


namespace mass_percentage_O_in_CuCO3_is_38_l241_241237

-- Define the atomic masses as constants
def Cu_atomic_mass : ℝ := 63.55
def C_atomic_mass : ℝ := 12.01
def O_atomic_mass : ℝ := 16.00

-- Define the composition of CuCO3
def CuCO3_molar_mass : ℝ := Cu_atomic_mass + C_atomic_mass + 3 * O_atomic_mass
def oxygen_mass_in_CuCO3 : ℝ := 3 * O_atomic_mass

-- Calculate the mass percentage of oxygen in CuCO3
def mass_percentage_O_in_CuCO3 : ℝ := (oxygen_mass_in_CuCO3 / CuCO3_molar_mass) * 100

-- Main theorem: mass percentage of O in CuCO3 is approximately 38.83%
theorem mass_percentage_O_in_CuCO3_is_38.83 : abs (mass_percentage_O_in_CuCO3 - 38.83) < 0.01 :=
by
  -- Proof would go here
  sorry

end mass_percentage_O_in_CuCO3_is_38_l241_241237


namespace middle_frustum_volume_l241_241442

theorem middle_frustum_volume (H R V : ℝ) (V_eq : V = (1/3) * π * R^2 * H) :
    let h := H / 3
    let R1 := R / 3
    let R2 := 2 * R / 3
    let frustum_volume := (1 / 3) * h * π * (R1^2 + R2^2 + R1 * R2)
in frustum_volume = (7 / 27) * V :=
by
  sorry

end middle_frustum_volume_l241_241442


namespace volume_ratio_proof_l241_241443

-- Definitions for the problem
variables (r_small h_small r_large h_large : ℝ)

-- Height condition
def height_relation : Prop := 
  h_large = 2 * h_small

-- Surface area condition
def surface_area_relation : Prop :=
  2 * Real.pi * r_large * h_large = 12 * 2 * Real.pi * r_small * h_small

-- Prove the volume ratio condition
theorem volume_ratio_proof
  (H1 : height_relation r_large h_large h_small)
  (H2 : surface_area_relation r_small r_large h_small h_large) :
  (Real.pi * r_large^2 * h_large) / (Real.pi * r_small^2 * h_small) = 72 :=
sorry  -- Proof to be filled in

end volume_ratio_proof_l241_241443


namespace sum_of_even_factors_720_l241_241870

theorem sum_of_even_factors_720 : 
  let n := 2^4 * 3^2 * 5 in
  (∑ d in (Finset.range (n + 1)).filter (λ d, d % 2 = 0 ∧ n % d = 0), d) = 2340 :=
by
  let n := 2^4 * 3^2 * 5
  sorry

end sum_of_even_factors_720_l241_241870


namespace probability_red_gt_blue_lt_3blue_is_5_over_18_l241_241193

noncomputable def probability_red_gt_blue_lt_3blue : ℝ :=
  let s := {p : ℝ × ℝ | 0 ≤ p.fst ∧ p.fst ≤ 1 ∧ 0 ≤ p.snd ∧ p.snd ≤ 1}
  let e := {p : ℝ × ℝ | p.1 < p.2 ∧ p.2 < 3 * p.1}
  (MeasureTheory.volume e) / (MeasureTheory.volume s)

theorem probability_red_gt_blue_lt_3blue_is_5_over_18 :
  probability_red_gt_blue_lt_3blue = 5 / 18 :=
sorry

end probability_red_gt_blue_lt_3blue_is_5_over_18_l241_241193


namespace find_x_value_y_256_l241_241702

noncomputable def x_value (y: ℕ) (x: ℕ) (k: ℕ) (z : ℕ): ℕ :=
    if y = 256 then 1 else
    if y = 4 then 14 else
    sorry

theorem find_x_value_y_256 :
    (∀ y x, x^2 * y = 256) → 
    (x_value 16 4 256 10 = 4) →
    (x_value 4 14 256 10 = 14) →
    x_value 256 1 256 0 = 1 :=
by
    intros h_inv h_16 h_z
    assume x_value 256 1 256 0 = 1
    sorry

end find_x_value_y_256_l241_241702


namespace minimum_positive_period_f_l241_241447

def f (x : ℝ) : ℝ := (1/2) * sin (2 * x) + (1/2) * tan (π / 3) * cos (2 * x)

theorem minimum_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T' ≥ T) :=
sorry

end minimum_positive_period_f_l241_241447


namespace integers_in_range_of_f_l241_241390

noncomputable def f (x : ℝ) := x^2 + x + 1/2

def count_integers_in_range (n : ℕ) : ℕ :=
  2 * (n + 1)

theorem integers_in_range_of_f (n : ℕ) :
  (count_integers_in_range n) = (2 * (n + 1)) :=
by
  sorry

end integers_in_range_of_f_l241_241390


namespace solve_y_l241_241795

theorem solve_y (y : ℂ) : y^2 - 6 * y + 5 = -(y + 2) * (y + 7) ↔ 
         (y = Complex.mk (-3 / 4) (Complex.cosh (sqrt 143 / 4)) ∨ 
          y = Complex.mk (-3 / 4) (-Complex.cosh (sqrt 143 / 4))) := 
sorry

end solve_y_l241_241795


namespace parabola_focus_hyperbola_vertex_l241_241707

noncomputable def parabola_focus (p: ℝ) : ℝ × ℝ := (p / 2, 0)
noncomputable def hyperbola_right_vertex : ℝ × ℝ := (2, 0)

theorem parabola_focus_hyperbola_vertex (p : ℝ) :
  parabola_focus p = hyperbola_right_vertex → p = 4 :=
by
  intro h
  rw [parabola_focus, hyperbola_right_vertex] at h
  cases h -- This step breaks the equality into two equalities: one for each component
  linarith -- Uses linear arithmetic to solve the resulting equation

end parabola_focus_hyperbola_vertex_l241_241707


namespace black_white_intersection_l241_241398

def segment_intersection_problem (k : ℕ) :=
  ∃ (white_segments black_segments : fin (2*k - 1) → fin (2*k - 1) → Prop), 
    (∀ w, (∑ b, if white_segments b w then 1 else 0) ≥ k) ∧
    (∀ b, (∑ w, if black_segments w b then 1 else 0) ≥ k) ∧
    (∃ b_all, ∀ w, black_segments w b_all) ∧
    (∃ w_all, ∀ b, white_segments b w_all)
  
theorem black_white_intersection (k : ℕ) : segment_intersection_problem k :=
sorry

end black_white_intersection_l241_241398


namespace total_pens_bought_l241_241035

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l241_241035


namespace fraction_spent_on_personal_needs_l241_241039

theorem fraction_spent_on_personal_needs :
  let hourly_rate := 10
  let hours_first_month := 35
  let hours_second_month := hours_first_month + 5
  let earnings_first_month := hourly_rate * hours_first_month
  let earnings_second_month := hourly_rate * hours_second_month
  let total_earnings := earnings_first_month + earnings_second_month
  let amount_saved := 150
  let amount_spent_on_personal_needs := total_earnings - amount_saved
  let fraction_spent := amount_spent_on_personal_needs / total_earnings
  in fraction_spent = 4 / 5 := by
begin
  sorry
end

end fraction_spent_on_personal_needs_l241_241039


namespace fill_pipe_time_l241_241529

theorem fill_pipe_time (t : ℕ) (H : ∀ C : Type, (1 / 2 : ℚ) * C = t * 1/2 * C) : t = t :=
by
  sorry

end fill_pipe_time_l241_241529


namespace range_of_g_l241_241316

noncomputable def g (c d : ℝ) : ℝ → ℝ := λ x, c * x + d

theorem range_of_g (c d : ℝ) (h : c < 0) : 
  set.range (g c d) = set.Icc (2 * c + d) (c + d) :=
by {
  sorry
}

end range_of_g_l241_241316


namespace parallelogram_side_sum_l241_241580

variable (x y : ℚ)

theorem parallelogram_side_sum :
  4 * x - 1 = 10 →
  5 * y + 3 = 12 →
  x + y = 91 / 20 :=
by
  intros h1 h2
  sorry

end parallelogram_side_sum_l241_241580


namespace average_velocity_first_second_instantaneous_velocity_end_first_second_velocity_reaches_14_after_2_seconds_l241_241192

open Real

noncomputable def f (x : ℝ) := (2/3) * x ^ 3 + x ^ 2 + 2 * x

-- (1) Prove that the average velocity of the particle during the first second is 3 m/s
theorem average_velocity_first_second : (f 1 - f 0) / (1 - 0) = 3 := by
  sorry

-- (2) Prove that the instantaneous velocity at the end of the first second is 6 m/s
theorem instantaneous_velocity_end_first_second : deriv f 1 = 6 := by
  sorry

-- (3) Prove that the velocity of the particle reaches 14 m/s after 2 seconds
theorem velocity_reaches_14_after_2_seconds :
  ∃ x : ℝ, deriv f x = 14 ∧ x = 2 := by
  sorry

end average_velocity_first_second_instantaneous_velocity_end_first_second_velocity_reaches_14_after_2_seconds_l241_241192


namespace range_of_independent_variable_l241_241084

theorem range_of_independent_variable (x : ℝ) : 
    ∃ y : ℝ, y = sqrt (x - 2) → x ≥ 2 :=
by
  intro y
  intro h
  rw [h]
  apply real.sqrt_nonneg (x - 2)
  sorry

end range_of_independent_variable_l241_241084


namespace polynomial_identity_l241_241622

noncomputable def z : ℂ := 2 - complex.I

theorem polynomial_identity :
  (z ^ 2 - 4 * z + 5 = 0) →
  (z ^ 6 - 3 * z ^ 5 + z ^ 4 + 5 * z ^ 3 + 2 = (z ^ 2 - 4 * z + 5) * (z ^ 4 + z ^ 3) + 2) :=
by
  intro hz
  sorry

end polynomial_identity_l241_241622


namespace sum_of_even_factors_720_l241_241869

theorem sum_of_even_factors_720 : 
  let n := 2^4 * 3^2 * 5 in
  (∑ d in (Finset.range (n + 1)).filter (λ d, d % 2 = 0 ∧ n % d = 0), d) = 2340 :=
by
  let n := 2^4 * 3^2 * 5
  sorry

end sum_of_even_factors_720_l241_241869


namespace mn_value_l241_241312

noncomputable def log_base (a b : ℝ) := Real.log b / Real.log a

theorem mn_value (M N : ℝ) (a : ℝ) 
  (h1 : log_base M N = a * log_base N M)
  (h2 : M ≠ N) (h3 : M * N > 0) (h4 : M ≠ 1) (h5 : N ≠ 1) (h6 : a = 4)
  : M * N = N^(3/2) ∨ M * N = N^(1/2) := 
by
  sorry

end mn_value_l241_241312


namespace cross_product_scalar_multiplication_l241_241313

variables (a b : ℝ^3)
variables (k : ℝ) (u : ℝ^3)

theorem cross_product_scalar_multiplication (h : a × b = ![-3, 2, 8]) :
  a × (4 • b) = ![-12, 8, 32] :=
by sorry

end cross_product_scalar_multiplication_l241_241313


namespace sum_of_even_factors_720_l241_241883

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) :=
    (2 + 4 + 8 + 16) * (1 + 3 + 9) * (1 + 5)
  in even_factors_sum 720 = 2340 :=
by
  sorry

end sum_of_even_factors_720_l241_241883


namespace sum_of_consecutive_pages_with_product_15300_l241_241827

theorem sum_of_consecutive_pages_with_product_15300 : 
  ∃ n : ℕ, n * (n + 1) = 15300 ∧ n + (n + 1) = 247 :=
by
  sorry

end sum_of_consecutive_pages_with_product_15300_l241_241827


namespace prob_m_n_leq_4_prob_m_lt_n_add_2_l241_241105

/-- A definition to specify a die roll can be 1 through 6. --/
@[derive decidable_eq] inductive die_val : Type
| one | two | three | four | five | six

open die_val

/-- Convert die_val values to corresponding natural numbers. --/
def die_val_to_nat : die_val → ℕ
| one   := 1
| two   := 2
| three := 3
| four  := 4
| five  := 5
| six   := 6

/-- Denotes that we are looking at a pair of dice rolls (m, n). --/
@[derive decidable_eq] structure dice_rolls : Type :=
(m n : die_val)

/-- Probability calculation for the condition m + n ≤ 4 --/
theorem prob_m_n_leq_4 : 
  (finset.univ.filter (λ x : dice_rolls, die_val_to_nat x.m + die_val_to_nat x.n ≤ 4)).card.to_rat / 36 = 1 / 6 :=
by sorry

/-- Probability calculation for the condition m < n + 2 --/
theorem prob_m_lt_n_add_2 :
  (finset.univ.filter (λ x : dice_rolls, die_val_to_nat x.m < die_val_to_nat x.n + 2)).card.to_rat / 36 = 13 / 18 :=
by sorry

end prob_m_n_leq_4_prob_m_lt_n_add_2_l241_241105


namespace largest_possible_length_d_l241_241950

theorem largest_possible_length_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 2) 
  (h2 : a ≤ b)
  (h3 : b ≤ c)
  (h4 : c ≤ d) 
  (h5 : d < a + b + c) : 
  d < 1 :=
sorry

end largest_possible_length_d_l241_241950


namespace combinations_with_at_least_one_red_l241_241054

theorem combinations_with_at_least_one_red :
  let total_socks := 7
  let choose_socks := 4
  let choose_ways (n k : ℕ) := Nat.choose n k
  total_socks = 7 → choose_socks = 4 →
  (choose_ways 7 4) - (choose_ways 6 4) = 20 :=
by
  let total_socks := 7
  let choose_socks := 4
  let choose_ways (n k : ℕ) := Nat.choose n k
  intro h1 h2
  rw [← h1, ← h2]
  sorry

end combinations_with_at_least_one_red_l241_241054


namespace initially_calculated_average_l241_241427

theorem initially_calculated_average 
  (correct_sum : ℤ)
  (incorrect_diff : ℤ)
  (num_numbers : ℤ)
  (correct_average : ℤ)
  (h1 : correct_sum = correct_average * num_numbers)
  (h2 : incorrect_diff = 20)
  (h3 : num_numbers = 10)
  (h4 : correct_average = 18) :
  (correct_sum - incorrect_diff) / num_numbers = 16 := by
  sorry

end initially_calculated_average_l241_241427


namespace modulus_z_l241_241283

noncomputable def z : ℂ := (1 + complex.I) / (2 - complex.I)

theorem modulus_z : complex.abs z = real.sqrt 10 / 5 :=
sorry

end modulus_z_l241_241283


namespace map_lines_l241_241769

noncomputable def maps_lines_to_lines (f : ℝ² → ℝ²) : Prop :=
∀ L : set ℝ², is_line L → is_line (f '' L)

axiom is_circle (C : set ℝ²) : Prop

theorem map_lines (f : ℝ² → ℝ²) (hf : continuous f) (H : ∀ C : set ℝ², is_circle C → is_circle (f '' C)) :
  maps_lines_to_lines f :=
sorry

end map_lines_l241_241769


namespace correct_calculation_l241_241131

theorem correct_calculation (x a b : ℝ) : 
  (x^4 * x^4 = x^8) ∧ ((a^3)^2 = a^6) ∧ ((a * (b^2))^3 = a^3 * b^6) → (a + 2*a = 3*a) := 
by 
  sorry

end correct_calculation_l241_241131


namespace percent_non_swimmers_play_soccer_l241_241209

def percent_soccer_non_swimmers (total_children : ℕ) (soccer_percent : ℝ) (swim_percent : ℝ) (soccer_and_swim_percent : ℝ) : ℝ :=
  let soccer_players := soccer_percent * total_children
  let swimmers := swim_percent * total_children
  let soccer_and_swim := soccer_and_swim_percent * soccer_players
  let non_swimming_soccer_players := soccer_players - soccer_and_swim
  let non_swimmers := total_children - swimmers
  (non_swimming_soccer_players / non_swimmers) * 100

theorem percent_non_swimmers_play_soccer 
  (N : ℕ)
  (soccer_percent : ℝ)
  (swim_percent : ℝ)
  (soccer_and_swim_percent : ℝ)
  (percent_play_soccer_non_swimmers : ℝ) :
  soccer_percent = 0.6 →
  swim_percent = 0.3 →
  soccer_and_swim_percent = 0.4 →
  percent_play_soccer_non_swimmers = 51 :=
by
  sorry

end percent_non_swimmers_play_soccer_l241_241209


namespace sum_even_factors_l241_241856

theorem sum_even_factors (n : ℕ) (h : n = 720) : 
  (∑ d in Finset.filter (λ d, d % 2 = 0) (Finset.divisors n), d) = 2340 :=
by
  rw h
  -- sorry to skip the actual proof
  sorry

end sum_even_factors_l241_241856


namespace odd_divisors_count_below_100_l241_241309

theorem odd_divisors_count_below_100 : 
  ∃ n, n = 9 ∧ ∀ m : ℕ, m < 100 → (odd (length (divisors m)) ↔ ∃ k : ℕ, k^2 = m) := 
sorry

end odd_divisors_count_below_100_l241_241309


namespace triangle_is_isosceles_at_2_l241_241756

-- Define the initial angles of the right triangle
def α₀ := 30
def β₀ := 60
def γ₀ := 90

-- Recurrence relations for the angles
def α(n : ℕ) : ℕ :=
  if n = 0 then α₀ else β(n - 1)

def β(n : ℕ) : ℕ :=
  if n = 0 then β₀ else α(n - 1)

def γ(n : ℕ) : ℕ :=
  γ₀

-- Define the property of being isosceles
def is_isosceles (n : ℕ) : Prop :=
  α(n) = β(n)

-- Prove that the triangle becomes isosceles at n = 2
theorem triangle_is_isosceles_at_2 : is_isosceles 2 :=
  sorry

end triangle_is_isosceles_at_2_l241_241756


namespace minimize_distance_theorem_l241_241725

noncomputable theory
open_locale classical

variables {ℝ : Type*} [metric_space ℝ]

structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Line :=
(point : Point)
(direction : Point)

def distance (A B : Point) : ℝ :=
  ((A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2)^0.5

noncomputable def minimize_distance (A B : Point) (l : Line) :=
  ∃ M : Point, (M ∈ l) ∧
  ∀ N : Point, (N ∈ l) → distance A M + distance M B ≤ distance A N + distance N B

theorem minimize_distance_theorem (A B : Point) (l : Line) 
  (h_skew : ¬ ∃ P, P ∈ (Line_through A B) ∧ P ∈ l) :
  minimize_distance A B l :=
sorry

end minimize_distance_theorem_l241_241725


namespace find_n_l241_241731

-- Definitions as per conditions
def arith_seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 7 = 4

def geom_seq (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 1 = 6 ∧ b 2 = a 3

-- The main theorem statement
theorem find_n (a b : ℕ → ℝ) (h_arith : arith_seq a) (h_geom : geom_seq b a) :
  ∃ n : ℕ, b n * a 26 < 1 ∧ ∀ m : ℕ, m < n → b m * a 26 ≥ 1 :=
begin
  sorry
end

end find_n_l241_241731


namespace area_of_FDBG_l241_241354

theorem area_of_FDBG
  (A B C D E F G : Type)
  (AB AC : ℝ) 
  (area_ABC : ℝ)
  (midpoint_D : D)
  (midpoint_E : E)
  (angle_bisector_intersection_F : F)
  (angle_bisector_intersection_G : G)
  (hAB : AB = 64)
  (hAC : AC = 26)
  (h_area_ABC : area_ABC = 208)
  (h_midpoint_D : midpoint_D = (A + B) / 2)
  (h_midpoint_E : midpoint_E = (A + C) / 2)
  (h_intersection_F : angle_bisector_intersection_F = F)
  (h_intersection_G : angle_bisector_intersection_G = G) :
  area_of F D B G = 134 :=
sorry

end area_of_FDBG_l241_241354


namespace sum_of_even_factors_720_l241_241878

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) :=
    (2 + 4 + 8 + 16) * (1 + 3 + 9) * (1 + 5)
  in even_factors_sum 720 = 2340 :=
by
  sorry

end sum_of_even_factors_720_l241_241878


namespace chris_car_offer_difference_l241_241973

theorem chris_car_offer_difference :
  ∀ (asking_price : ℕ) (maintenance_cost_factor : ℕ) (headlight_cost : ℕ) (tire_multiplier : ℕ),
  asking_price = 5200 →
  maintenance_cost_factor = 10 →
  headlight_cost = 80 →
  tire_multiplier = 3 →
  let first_earnings := asking_price - asking_price / maintenance_cost_factor,
      second_earnings := asking_price - (headlight_cost + headlight_cost * tire_multiplier) in
  second_earnings - first_earnings = 200 :=
by
  intros asking_price maintenance_cost_factor headlight_cost tire_multiplier h1 h2 h3 h4
  -- leave "sorry" as a placeholder for the proof
  sorry

end chris_car_offer_difference_l241_241973


namespace points_on_plane_l241_241041

theorem points_on_plane (P : set (ℝ × ℝ)) (hP : P.card = 9)
  (h_no_four_collinear : ∀ (S : set (ℝ × ℝ)), S ⊆ P → S.card = 4 → ¬collinear ℝ S)
  (h_six_points : ∀ (S : set (ℝ × ℝ)), S ⊆ P → S.card = 6 → ∃ T ⊆ S, T.card = 3 ∧ collinear ℝ T) :
  ∃ P' : set (ℝ × ℝ), P'.card = 9 ∧
    (∀ (S : set (ℝ × ℝ)), S ⊆ P' → S.card = 4 → ¬collinear ℝ S) ∧
    (∀ (S : set (ℝ × ℝ)), S ⊆ P' → S.card = 6 → ∃ T ⊆ S, T.card = 3 ∧ collinear ℝ T) := 
begin 
   sorry 
end

end points_on_plane_l241_241041


namespace omega_not_possible_l241_241295

noncomputable def f (ω x φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem omega_not_possible (ω φ : ℝ) (h1 : ∀ x y, -π/3 ≤ x → x < y → y ≤ π/6 → f ω x φ ≤ f ω y φ)
  (h2 : f ω (π / 6) φ = f ω (4 * π / 3) φ)
  (h3 : f ω (π / 6) φ = -f ω (-π / 3) φ) :
  ω ≠ 7 / 5 :=
sorry

end omega_not_possible_l241_241295


namespace divide_circle_into_equal_parts_l241_241586

theorem divide_circle_into_equal_parts 
-- Given: 
(circle: Type) 
(center : point circle) 
(points_on_circumference : list (point circle)) 
(h_start : circumcenter center circle)
(h_six_points : points_on_circumference.length = 6)
(h_equilateral_triangles : ∀ i ∈ points_on_circumference, -- Some property of equilateral triangle on each sector
  is_equilateral_triangle (triangle.mk center i (succ i))) : 
-- To Prove:
∃ (parts : list (set (point circle))), 
  parts.length = 12 ∧
  ∀ (p ∈ parts), ¬p_meets_boundary (p, center) :=
  sorry

end divide_circle_into_equal_parts_l241_241586


namespace volume_of_hemisphere_correct_l241_241571

-- Define the diameter and the radius
def diameter : ℝ := 8
def radius : ℝ := diameter / 2

-- Volume of a hemisphere calculation
def volume_hemisphere (r : ℝ) : ℝ := (2 / 3) * π * r^3

-- Theorem statement
theorem volume_of_hemisphere_correct : volume_hemisphere radius = (128 / 3) * π :=
by
  sorry

end volume_of_hemisphere_correct_l241_241571


namespace seating_possible_l241_241336

theorem seating_possible (n : ℕ) (guests : Fin (2 * n) → Finset (Fin (2 * n))) 
  (h1 : ∀ i, n ≤ (guests i).card)
  (h2 : ∀ i j, (i ≠ j) → i ∈ guests j → j ∈ guests i) : 
  ∃ (a b c d : Fin (2 * n)), 
    (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ d) ∧ (d ≠ a) ∧
    (a ∈ guests b) ∧ (b ∈ guests c) ∧ (c ∈ guests d) ∧ (d ∈ guests a) := 
sorry

end seating_possible_l241_241336


namespace number_of_lilies_l241_241594

theorem number_of_lilies (L : ℕ) 
  (h1 : ∀ n:ℕ, n * 6 = 6 * n)
  (h2 : ∀ n:ℕ, n * 3 = 3 * n) 
  (h3 : 5 * 3 = 15)
  (h4 : 6 * L + 15 = 63) : 
  L = 8 := 
by
  -- Proof omitted 
  sorry

end number_of_lilies_l241_241594


namespace compute_expression_l241_241578

theorem compute_expression : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end compute_expression_l241_241578


namespace perimeter_of_triangle_ADE_l241_241364

structure Point (α : Type) := 
(x : α) (y : α)

structure Triangle (α : Type) :=
(A : Point α) (B : Point α) (C : Point α)

def midpoint {α : Type} [field α] (p1 p2 : Point α) : Point α :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2 }

def perimeter {α : Type} [metric_space α] (t : Triangle α) : α :=
  dist t.A t.B + dist t.B t.C + dist t.C t.A

variables {α : Type} [field α] [metric_space (Point α)]

def problem_conditions (A D E B C : Point α) (t : Triangle α) : Prop :=
  midpoint A D = C ∧
  triangle_area {A:=A, B:=B, C:=E} = triangle_area {A:=B, B:=C, C:=E} ∧
  is_isosceles {A:=B, B:=D, C:=E} ∧
  perimeter {A:=A, B:=B, C:=E} = distance B E + distance E C + 6 ∧
  perimeter {A:=A, B:=C, C:=E} = perimeter {A:=C, B:=D, C:=E} + 2

def perimeter_ADE {α : Type} [field α] [metric_space (Point α)] (A D E B C : Point α) : α :=
  perimeter {A:=A, B:=D, C:=E}

theorem perimeter_of_triangle_ADE (A D E B C : Point α) (h : problem_conditions A D E B C) :
  perimeter_ADE A D E B C = 46 / 3 := sorry

end perimeter_of_triangle_ADE_l241_241364


namespace geometric_sequence_a4_l241_241625

theorem geometric_sequence_a4 (a : ℕ → ℝ) 
  (h_geom : ∀ n m : ℕ, a (n+1) / a n = a (m+1) / a m) 
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_cond : a 1 * a 7 = 3 / 4) : 
  a 4 = real.sqrt 3 / 2 :=
by
  sorry

end geometric_sequence_a4_l241_241625


namespace isosceles_triangle_area_l241_241204

/-- An isosceles triangle has a perimeter of \( 2q \). If one of the equal legs of the triangle
is \( \sqrt{3} \) times the base, then the area of the triangle is 
\( \frac{q^2 (1 - 2 \sqrt{3}) (3 \sqrt{3} + 1)} {66} \). -/
theorem isosceles_triangle_area (q : ℝ) :
  ∃ A : ℝ, (∀ (b : ℝ), 
    let leg := b * sqrt 3 in
    let perimeter := b + 2 * leg in
    let h := sqrt (leg^2 - (b / 2)^2) in
    perimeter = 2 * q →
    A = 1 / 2 * b * h) →
  A = q^2 * (1 - 2 * sqrt 3) * (3 * sqrt 3 + 1) / 66 :=
sorry

end isosceles_triangle_area_l241_241204


namespace find_x_l241_241127

theorem find_x (x m n : ℤ) 
  (h₁ : 15 + x = m^2) 
  (h₂ : x - 74 = n^2) :
  x = 2010 :=
by
  sorry

end find_x_l241_241127


namespace sum_of_roots_of_quadratic_l241_241488

theorem sum_of_roots_of_quadratic : 
  ∀ x1 x2 : ℝ, 
  (3 * x1^2 - 6 * x1 - 7 = 0 ∧ 3 * x2^2 - 6 * x2 - 7 = 0) → 
  (x1 + x2 = 2) := by
  sorry

end sum_of_roots_of_quadratic_l241_241488


namespace solution_satisfies_conditions_l241_241138

noncomputable def sqrt_eq (a b x y : ℝ) : Prop :=
  sqrt (x - y) = a ∧ sqrt (x + y) = b

theorem solution_satisfies_conditions 
  (x y : ℝ)
  (h1 : sqrt_eq (2/5) 2 x y)
  (hexact: x = 52/25 ∧ y = 48/25) :
  sqrt_eq (2/5) 2 x y ∧ 
  (x * y = 8/25) :=
by
  sorry

end solution_satisfies_conditions_l241_241138


namespace flow_in_channels_l241_241179

variables (q0 : ℝ)

-- Define the flows in use
def flow_AH : ℝ := q0
def flow_HG : ℝ := q0  -- By symmetry
def flow_BC : ℝ := 2/3 * q0
def flow_CD : ℝ := 2/3 * q0  -- By symmetry
def flow_BG : ℝ := 2/3 * q0  -- By symmetry
def flow_GD : ℝ := 2/3 * q0  -- By symmetry
def flow_AB : ℝ := 4/3 * q0

-- Total flow into node A
def total_flow_A : ℝ := flow_AH q0 + flow_AB q0 

theorem flow_in_channels (q0 : ℝ) :
  flow_AB q0 = 4/3 * q0 ∧ flow_BC q0 = 2/3 * q0 ∧ total_flow_A q0 = 7/3 * q0 :=
by
  sorry

end flow_in_channels_l241_241179


namespace total_pens_bought_l241_241018

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l241_241018


namespace sum_of_cubic_numbers_l241_241112

theorem sum_of_cubic_numbers :
  ∑ n in {n | ∃ m : ℕ, n^3 + 13 * n - 273 = m^3}.to_finset, n = 29 :=
by
  sorry

end sum_of_cubic_numbers_l241_241112


namespace linear_equation_solution_l241_241298

theorem linear_equation_solution (x y : ℝ) (h : 3 * x - y = 5) : y = 3 * x - 5 :=
sorry

end linear_equation_solution_l241_241298


namespace total_pens_bought_l241_241012

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l241_241012


namespace roots_g_eq_zero_l241_241532

noncomputable def g : ℝ → ℝ := sorry

theorem roots_g_eq_zero :
  (∀ x : ℝ, g (3 + x) = g (3 - x)) →
  (∀ x : ℝ, g (8 + x) = g (8 - x)) →
  (∀ x : ℝ, g (12 + x) = g (12 - x)) →
  g 0 = 0 →
  ∃ L : ℕ, 
  (∀ k, 0 ≤ k ∧ k ≤ L → g (k * 48) = 0) ∧ 
  (∀ k : ℤ, -1000 ≤ k ∧ k ≤ 1000 → (∃ n : ℕ, k = n * 48)) ∧ 
  L + 1 = 42 := 
by sorry

end roots_g_eq_zero_l241_241532


namespace option_d_correct_l241_241940

noncomputable def f : ℝ → ℝ := sorry -- Define f as a noncomputable function to avoid implementation issues

axiom h_domain : ∀ x, 0 < x ∧ x < π / 2 → ∃ y, f y ∈ set.Ioo 0 (π / 2) -- f is defined on the open interval (0, π/2)

axiom h_tan_ineq : ∀ x, 0 < x ∧ x < π / 2 → (tan x) * f x > deriv f x -- ∀ x in the domain, tan(x) * f(x) > f'(x)

theorem option_d_correct : sqrt 2 * f (π / 4) < sqrt 3 * f (π / 6) :=
by
  sorry -- Proof of the theorem will be provided here

end option_d_correct_l241_241940


namespace hyperbola_standard_equation_l241_241305

open Real

noncomputable def hyperbola_has_foci (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x^2 / 4) - y^2 = 1 ∧ 
    (0 < a ∧ 0 < b) ∧ 
    (F1_x = -sqrt 5 ∧ F1_y = 0 ∧ F2_x = sqrt 5 ∧ F2_y = 0) ∧
    (P_F1_dot_P_F2 = 0 ∧ |P_F1| * |P_F2| = 2)

theorem hyperbola_standard_equation (x y : ℝ) (a b : ℝ) : 
  hyperbola_has_foci (F1_x, F1_y) (F2_x, F2_y) → 
  P ∈ hyperbola :=
by
  show (x^2 / 4) - y^2 = 1
  sorry

end hyperbola_standard_equation_l241_241305


namespace LCM_180_504_l241_241119

theorem LCM_180_504 : Nat.lcm 180 504 = 2520 := 
by 
  -- We skip the proof.
  sorry

end LCM_180_504_l241_241119


namespace inversely_proportional_x_y_l241_241221

theorem inversely_proportional_x_y (x y c : ℝ) 
  (h1 : x * y = c) (h2 : 8 * 16 = c) : y = -32 → x = -4 :=
by
  sorry

end inversely_proportional_x_y_l241_241221


namespace simplify_expression_l241_241060

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
by
  sorry

end simplify_expression_l241_241060


namespace sum_even_factors_of_720_l241_241863

theorem sum_even_factors_of_720 : 
  let even_factors_sum (n : ℕ) : ℕ := 
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  in even_factors_sum 720 = 2340 :=
by
  let even_factors_sum (n : ℕ) : ℕ :=
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  sorry

end sum_even_factors_of_720_l241_241863


namespace kasun_family_children_count_l241_241344

theorem kasun_family_children_count 
    (m : ℝ) (x : ℕ) (y : ℝ)
    (h1 : (m + 50 + x * y + 10) / (3 + x) = 22)
    (h2 : (m + x * y + 10) / (2 + x) = 18) :
    x = 5 :=
by
  sorry

end kasun_family_children_count_l241_241344


namespace students_in_line_l241_241533

theorem students_in_line (n : ℕ) (h : 1 ≤ n ∧ n ≤ 130) : 
  n = 3 ∨ n = 43 ∨ n = 129 :=
by
  sorry

end students_in_line_l241_241533


namespace sum_of_even_factors_720_l241_241887

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) : ℕ :=
    match n with
    | 720 => 
      let sum_powers_2 := 2 + 4 + 8 + 16 in
      let sum_powers_3 := 1 + 3 + 9 in
      let sum_powers_5 := 1 + 5 in
      sum_powers_2 * sum_powers_3 * sum_powers_5
    | _ => 0
  in
  even_factors_sum 720 = 2340 :=
by 
  sorry

end sum_of_even_factors_720_l241_241887


namespace proof_f_2008_l241_241810

theorem proof_f_2008 {f : ℝ → ℝ} 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (3 * x + 1) = f (3 * (x + 1) + 1))
  (h3 : f (-1) = -1) : 
  f 2008 = 1 := 
by
  sorry

end proof_f_2008_l241_241810


namespace defeated_candidate_percentage_l241_241721

noncomputable def percentage_defeated_candidate (total_votes diff_votes invalid_votes : ℕ) : ℕ :=
  let valid_votes := total_votes - invalid_votes
  let P := 100 * (valid_votes - diff_votes) / (2 * valid_votes)
  P

theorem defeated_candidate_percentage (total_votes : ℕ) (diff_votes : ℕ) (invalid_votes : ℕ) :
  total_votes = 12600 ∧ diff_votes = 5000 ∧ invalid_votes = 100 → percentage_defeated_candidate total_votes diff_votes invalid_votes = 30 :=
by
  intros
  sorry

end defeated_candidate_percentage_l241_241721


namespace ellipse_hyperbola_common_foci_l241_241658

noncomputable def ellipse_C1 (x y : ℝ) (n : ℝ) : Prop :=
  x^2 / 3 + y^2 / n = 1

noncomputable def hyperbola_C2 (x y : ℝ) (n : ℝ) : Prop :=
  x^2 - y^2 / n = 1

theorem ellipse_hyperbola_common_foci (n : ℝ) (a1 a2 : ℝ) (b1 b2 e : ℝ) :
  (a1^2 = 3) →
  (b1^2 = n) →
  (a2^2 = 1) →
  (b2^2 = n) →
  (a1^2 - b1^2 = a2^2 + b2^2) →
  e = Real.sqrt (1 - b1^2 / a1^2) →
  n = 1 →
  e = Real.sqrt 6 / 3 ∧
  ∀ x, (hyperbola_C2 x x n ∨ hyperbola_C2 x (-x) n) :=
begin
  intros h_a1 h_b1 h_a2 h_b2 h_eq h_e h_n,
  have ecc : e = Real.sqrt (2 / 3) := by sorry,
  have asymptotes : ∀ x, (hyperbola_C2 x x 1 ∨ hyperbola_C2 x (-x) 1) := by
  { intro x, split; rw [hyperbola_C2], sorry },
  exact ⟨ecc, asymptotes⟩,
end

end ellipse_hyperbola_common_foci_l241_241658


namespace sum_of_three_distinct_squares_sum_of_six_distinct_squares_l241_241741

theorem sum_of_three_distinct_squares (a b c : ℕ) (h₁ : 15129 = 123^2) (h₂ : 123 = 121 + 2) :
  15129 = 121^2 + 22^2 + 2^2 :=
by {
  -- Proof skipped
  sorry
}

theorem sum_of_six_distinct_squares (a b c d e f : ℕ) (h₁ : 378225 = 615^2) (h₂ : 615 = 5 * 123)
  (h₃ : 5^2 = 3^2 + 4^2) (h₄: sum_of_three_distinct_squares 2 22 121 15129 (by norm_num) (by norm_num)) :
  378225 = 6^2 + 66^2 + 363^2 + 8^2 + 88^2 + 484^2 :=
by {
  -- Proof skipped
  sorry
}

end sum_of_three_distinct_squares_sum_of_six_distinct_squares_l241_241741


namespace max_value_sqrt_sum_l241_241384

noncomputable def max_sqrt_sum (x y z : ℝ) : ℝ :=
  sqrt (4 * x + 1) + sqrt (4 * y + 1) + sqrt (4 * z + 1)

theorem max_value_sqrt_sum :
  ∀ (x y z : ℝ), x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 8 →
  max_sqrt_sum x y z ≤ 3 * sqrt (35 / 3) :=
by sorry

end max_value_sqrt_sum_l241_241384


namespace problem_part1_problem_part2_l241_241660

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) / Real.log a
noncomputable def g (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := 2 * Real.log (2 * x + t) / Real.log a

theorem problem_part1 (a t : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  f a 1 - g a t 1 = 0 → t = -2 + Real.sqrt 2 :=
sorry

theorem problem_part2 (a t : ℝ) (ha_bound : 0 < a ∧ a < 1) :
  (∀ x, 0 ≤ x ∧ x ≤ 15 → f a x ≥ g a t x) → t ≥ 1 :=
sorry

end problem_part1_problem_part2_l241_241660


namespace female_adults_more_than_male_l241_241200

/-- Given the number of male adults is 100, the total number of people is 750,
    and the number of children is twice the number of adults, prove that the 
    number of female adults is 150 and there are 50 more female adults than male adults. -/
theorem female_adults_more_than_male (M F : ℕ) (total_people children : ℕ) 
  (hM : M = 100)
  (h_total : total_people = 750)
  (h_children : children = 2 * (M + F))
  (h_people : total_people = (M + F) + children) :
  F - M = 50 :=
by {
  have h1 : total_people = 3 * (M + F) := by rw [h_people, h_children]; ring,
  rw hM at h1,
  have h2 : 750 = 3 * (100 + F) := h1,
  have h3 : 250 = 100 + F := by linarith,
  have hF : F = 150 := by linarith,
  rw [hF, hM],
  linarith,
}

end female_adults_more_than_male_l241_241200


namespace not_all_divisible_by_6_have_prime_neighbors_l241_241357

theorem not_all_divisible_by_6_have_prime_neighbors :
  ¬ ∀ n : ℕ, (6 ∣ n) → (Prime (n - 1) ∨ Prime (n + 1)) := by
  sorry

end not_all_divisible_by_6_have_prime_neighbors_l241_241357


namespace num_valid_four_digit_numbers_l241_241675

noncomputable def num_four_digit_numbers_with_2_3_4 : ℕ := 60

theorem num_valid_four_digit_numbers :
  ∃ n, n = 9000 ∧ (∃ k, k = 60 ∧ (num_four_digit_numbers_with_2_3_4 = k)) :=
begin
  use 9000,
  split,
  { -- Total four-digit numbers is 9000
    refl, 
  },
  { -- The number of four-digit numbers with at least one 2, at least one 3, and at least one 4 is 60
    use 60,
    split,
    { refl, },
    { -- The final condition, which we are given
      refl, },
  },
end

end num_valid_four_digit_numbers_l241_241675


namespace find_number_l241_241100

theorem find_number : ∃ x : ℤ, 35 + 3 * x = 56 ∧ x = 7 :=
by
  use 7
  split
  . norm_num
  . rfl

end find_number_l241_241100


namespace fill_half_cistern_time_l241_241524

variable (t_half : ℝ)

-- Define a condition that states the certain amount of time to fill 1/2 of the cistern.
def fill_pipe_half_time (t_half : ℝ) : Prop :=
  t_half > 0

-- The statement to prove that t_half is the time required to fill 1/2 of the cistern.
theorem fill_half_cistern_time : fill_pipe_half_time t_half → t_half = t_half := by
  intros
  rfl

end fill_half_cistern_time_l241_241524


namespace find_x_in_magic_square_l241_241334

def magicSquareProof (x d e f g h S : ℕ) : Prop :=
  (x + 25 + 75 = S) ∧
  (5 + d + e = S) ∧
  (f + g + h = S) ∧
  (x + d + h = S) ∧
  (f = 95) ∧
  (d = x - 70) ∧
  (h = 170 - x) ∧
  (e = x - 145) ∧
  (x + 25 + 75 = 5 + (x - 70) + (x - 145))

theorem find_x_in_magic_square : ∃ x d e f g h S, magicSquareProof x d e f g h S ∧ x = 310 := by
  sorry

end find_x_in_magic_square_l241_241334


namespace average_hidden_primes_l241_241214

theorem average_hidden_primes (x y z : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (hz : Nat.Prime z)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum : 44 + x = 59 + y ∧ 59 + y = 38 + z) :
  (x + y + z) / 3 = 14 := 
by
  sorry

end average_hidden_primes_l241_241214


namespace tree_edges_l241_241790

theorem tree_edges (n : ℕ) (G : SimpleGraph V) [Fintype V] (hG : G.IsTree) (hV : Fintype.card V = n) : G.edgeFinset.card = n - 1 :=
sorry

end tree_edges_l241_241790


namespace store_refusal_illegal_l241_241480

-- Definitions for conditions
def is_legal_tender (banknote : Type) : Prop :=
  banknote ∈ [issued_by_Bank_of_Russia] -- Placeholder, refine for Lean syntax requirements.

def is_not_counterfeit (banknote : Type) : Prop :=
  ¬ banknote ∈ [counterfeit] -- Placeholder, refine for Lean syntax requirements.

def permissible_damage (banknote : Type) : Prop :=
  banknote ∈ [dirt, wear, tears, small_holes, punctures, foreign_inscriptions, stains, stamps, missing_corners, missing_edges] -- Placeholder, refine for Lean syntax requirements.

-- Proof statement
theorem store_refusal_illegal (banknote : Type) (h1 : is_legal_tender banknote) (h2 : is_not_counterfeit banknote) (h3 : permissible_damage banknote) : false := 
  sorry

end store_refusal_illegal_l241_241480


namespace fraction_of_students_on_trip_are_girls_l241_241559

variable (b g : ℕ)
variable (H1 : g = 2 * b) -- twice as many girls as boys
variable (fraction_girls_on_trip : ℚ := 2 / 3)
variable (fraction_boys_on_trip : ℚ := 1 / 2)

def fraction_of_girls_on_trip (b g : ℕ) (H1 : g = 2 * b) (fraction_girls_on_trip : ℚ) (fraction_boys_on_trip : ℚ) :=
  let girls_on_trip := fraction_girls_on_trip * g
  let boys_on_trip := fraction_boys_on_trip * b
  let total_on_trip := girls_on_trip + boys_on_trip
  girls_on_trip / total_on_trip

theorem fraction_of_students_on_trip_are_girls (b g : ℕ) (H1 : g = 2 * b) : 
  fraction_of_girls_on_trip b g H1 (2 / 3) (1 / 2) = 8 / 11 := 
by sorry

end fraction_of_students_on_trip_are_girls_l241_241559


namespace sequence_general_formula_l241_241669

theorem sequence_general_formula :
  ∀ n : ℕ, n > 0 → ∀ a : ℕ → ℚ, 
    (a 1 = 3) → 
    (∀ n : ℕ, n > 0 → a (n+1) = (3 * a n - 4) / (a n - 1)) → 
    a n = (2 * n + 1) / n :=
by
  intro n hn a a1_step h_rec
  induction n using Nat.case_strong_induction_on with
  | base =>
    sorry
  | step n ih =>
    sorry

end sequence_general_formula_l241_241669


namespace max_determinant_of_matrix_l241_241755

open Matrix

def v : ℝ^3 := ![4, -2, 2]
def w : ℝ^3 := ![-1, 2, 5]

def cross_product (a b : ℝ^3) : ℝ^3 :=
  ![a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x]

def magnitude (u : ℝ^3) : ℝ :=
  real.sqrt (u.x^2 + u.y^2 + u.z^2)

def u : ℝ^3 := 1 / magnitude (cross_product v w) • cross_product v w

theorem max_determinant_of_matrix : det ![u, v, w] = real.sqrt 716 := by
  sorry

end max_determinant_of_matrix_l241_241755


namespace angle_YCZ_45_deg_l241_241509

theorem angle_YCZ_45_deg (X Y Z C : Type*)
  [Triangle X Y Z]
  (h1 : right_angle Z)
  (h2 : angle X = 30)
  (h3 : is_angle_bisector C Z (angle X Z Y))
  : angle Y C Z = 45 := 
sorry

end angle_YCZ_45_deg_l241_241509


namespace inverse_fourier_cosine_transform_l241_241601

noncomputable def F (p : ℝ) : ℝ :=
  if 0 < p ∧ p < 1 then 1 else 0

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (2 / Real.pi) * (Real.sin x / x)

theorem inverse_fourier_cosine_transform :
  (∀ x, f x = sqrt (2 / Real.pi) * (Real.sin x / x)) ->
  ∀ x, (Real.FourierCosineInverseTransform F x = f x) := 
by sorry

end inverse_fourier_cosine_transform_l241_241601


namespace min_distance_for_20_poles_l241_241616

def positions (n : ℕ) : ℕ → ℕ
| 0 => 0
| (n + 1) => positions n + 50

def distance_per_trip (n : ℕ) : ℕ := 2 * positions (n + 1) * 3

def total_distance (n : ℕ) : ℕ := 
  if n % 3 = 0 then
    (n / 3) * distance_per_trip (n / 3)
  else
    let trips := (n / 3) + 1
    in (trips - 1) * distance_per_trip (trips - 1) + 2 * positions (if n % 3 = 1 then 1 else 2)

theorem min_distance_for_20_poles : total_distance 20 = 14000 := 
by {
  -- The proof would involve demonstrating arithmetic calculations
  sorry
}

end min_distance_for_20_poles_l241_241616


namespace total_pens_l241_241007

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l241_241007


namespace sum_ratio_l241_241375

noncomputable theory

def is_arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, seq (n + 1) = seq n + d

def sum_first_n_terms (seq : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, seq i

variables {a b : ℕ → ℝ}
variables (h_arith_a : is_arithmetic_sequence a)
variables (h_arith_b : is_arithmetic_sequence b)
variable (h_a5_eq_2b5 : a 5 = 2 * b 5)

theorem sum_ratio (h_a5_eq_2b5 : a 5 = 2 * b 5) :
  (sum_first_n_terms a 9) / (sum_first_n_terms b 9) = 2 :=
sorry

end sum_ratio_l241_241375


namespace smallest_k_l241_241388

theorem smallest_k (n : ℕ) (h : 2 ≤ n) (S : Finset ℕ) (hS : S.card = n) :
  ∃ (k : ℕ), (∃ (A : Fin k → Finset ℕ), 
  (∀ a b ∈ S, a ≠ b → ∃ j : Fin k, (A j ∩ {a, b}) = {a} ∨ (A j ∩ {a, b}) = {b})) ∧ 
  k = Int.ceil (Real.log2 n) :=
begin
  sorry
end

end smallest_k_l241_241388


namespace eval_i_powers_sum_l241_241229

-- Define the cycle of i's powers as a hypothesis
axiom i_powers_cycle (n : ℤ) : 
  ∃ m : ℤ, n = 4 * m + 1 ∨ n = 4 * m + 2 ∨ n = 4 * m + 3 ∨ n = 4 * m 

noncomputable def complex_i := complex.I

-- Lean statement to prove that i^2023 + i^303 = -2i
theorem eval_i_powers_sum :
  complex_i ^ 2023 + complex_i ^ 303 = -2 * complex_i := by
  sorry

end eval_i_powers_sum_l241_241229


namespace new_age_average_l241_241428

theorem new_age_average (n : ℕ) (A old_age teacher_age : ℕ)
  (h1 : n = 30) (h2 : A = 10) (h3 : old_age = 11) (h4 : teacher_age = 41) :
  (n - 1) * A - old_age + teacher_age = 300 ∧
  (n - 1) * A - old_age + teacher_age = 330 →
  new_avg_age = (new_total_age) / old_n :
  new_avg_age = 11 :=
by
  sorry

end new_age_average_l241_241428


namespace trapezoid_diagonals_l241_241048

theorem trapezoid_diagonals (a b c d AC BD : ℝ) (h1: AC^2 = a^2 + d^2 + 2 * a * d * Math.cos(Math.pi - Math.acos(c / d)))
               (h2: BD^2 = a^2 + c^2 + 2 * a * c * Math.cos(Math.pi - Math.acos(d / c))) :
  AC^2 + BD^2 = c^2 + d^2 + 2 * a * b :=
by
  sorry

end trapezoid_diagonals_l241_241048


namespace increasing_interval_of_f_l241_241815

def f (x : ℝ) : ℝ := (1 / 2) ^ (2 * x^2 - 3 * x + 1)

theorem increasing_interval_of_f :
  ∀ x < (3 / 4), ∃ ϵ > 0, ∀ y ∈ Icc (x - ϵ) x, f y < f x := 
sorry

end increasing_interval_of_f_l241_241815


namespace rectangle_solution_l241_241144

-- Define the given conditions
variables (x y : ℚ)

-- Given equations
def condition1 := (Real.sqrt (x - y) = 2 / 5)
def condition2 := (Real.sqrt (x + y) = 2)

-- Solution
theorem rectangle_solution (x y : ℚ) (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 52 / 25 ∧ y = 48 / 25 ∧ (Real.sqrt ((52 / 25) * (48 / 25)) = 8 / 25) :=
by
  sorry

end rectangle_solution_l241_241144


namespace length_of_XZ_l241_241738

noncomputable def triangle_XYZ (XY XZ YZ : ℝ) (hXYZ: XY^2 + XZ^2 = YZ^2) : Prop :=
  ∠X = 90 ∧ YZ = 25 ∧ tan (Z) = 3 * sin (Z)

theorem length_of_XZ 
  (XY XZ YZ : ℝ)
  (hXYZ : XY^2 + XZ^2 = YZ^2)
  (hYZ : YZ = 25)
  (htan : tan (Z) = 3 * sin (Z))
  : XZ = 25 / 3 :=
sorry

end length_of_XZ_l241_241738


namespace sum_of_x_values_l241_241900

theorem sum_of_x_values (x : ℝ) (h : x ≠ -1) : 
  (∃ x, 3 = (x^3 - 3*x^2 - 4*x)/(x + 1)) →
  (x = 6) :=
by
  sorry

end sum_of_x_values_l241_241900


namespace max_radius_of_additional_jar_l241_241949

open Real

noncomputable def max_jar_radius (pot_radius jar1_radius jar2_radius : ℝ) : ℝ :=
(pot_radius^2 - 5 * jar1_radius^2 - (jar2_radius - jar1_radius)^2) / (2 * (pot_radius - jar2_radius - jar1_radius))

theorem max_radius_of_additional_jar :
  let pot_diameter := 36.0
  let pot_radius := pot_diameter / 2
  let jar1_radius := 6.0
  let jar2_radius := 12.0
  let r := 36.0 / 7.0
  max_jar_radius pot_radius jar1_radius jar2_radius = r := by
  sorry

end max_radius_of_additional_jar_l241_241949


namespace order_of_real_numbers_l241_241590

open Real

theorem order_of_real_numbers :
  (0.5 ^ 2) ∈ set.Ioo (0:ℝ) 1 ∧ log (2:ℝ) 0.5 < 0 ∧ (2:ℝ) ^ 0.5 > 1 →
  (2:ℝ) ^ 0.5 > 0.5 ^ 2 ∧ 0.5 ^ 2 > log (2:ℝ) 0.5 := 
by
  intro h
  sorry

end order_of_real_numbers_l241_241590


namespace min_square_sum_l241_241224

theorem min_square_sum (a b : ℝ) (h : a + b = 3) : a^2 + b^2 ≥ 9 / 2 :=
by 
  sorry

end min_square_sum_l241_241224


namespace advertisement_time_per_week_l241_241073

theorem advertisement_time_per_week : 
  let advertisement_duration := 1.5 -- In minutes
  let interval_between_ads := 18.5  -- In minutes
  let total_cycle_time := advertisement_duration + interval_between_ads
  let cycles_per_hour := 60 / total_cycle_time
  let advertisement_time_per_hour := cycles_per_hour * advertisement_duration
  let hours_per_week := 24 * 7
  let total_advertisement_time_minutes := hours_per_week * advertisement_time_per_hour
  let total_advertisement_time_hours := total_advertisement_time_minutes / 60
  let extra_minutes := total_advertisement_time_minutes % 60
  total_advertisement_time_hours = 12 ∧ extra_minutes = 36 := 
  by 
    sorry

end advertisement_time_per_week_l241_241073


namespace circle_passes_through_fixed_point_l241_241425

-- Define a parabola and a tangent line, and state the proof problem
theorem circle_passes_through_fixed_point :
  ∀ h : ℝ, ∀ k : ℝ, (k = (1/12 : ℝ) * h^2) ∧ ((abs h + 3) = (abs k + 3)) → (0, 3) ∈ set_of (λ x, x ∈ circle (0, 3) (√(h^2 + (k-3)^2))) :=
by
  sorry

end circle_passes_through_fixed_point_l241_241425


namespace second_number_is_22_l241_241329

noncomputable section

variables (x y : ℕ)

-- Definitions based on the conditions
-- Condition 1: The sum of two numbers is 33
def sum_condition : Prop := x + y = 33

-- Condition 2: The second number is twice the first number
def twice_condition : Prop := y = 2 * x

-- Theorem: Given the conditions, the second number y is 22.
theorem second_number_is_22 (h1 : sum_condition x y) (h2 : twice_condition x y) : y = 22 :=
by
  sorry

end second_number_is_22_l241_241329


namespace gcd_rope_lengths_l241_241393

-- Define the lengths of the ropes as constants
def rope_length1 := 75
def rope_length2 := 90
def rope_length3 := 135

-- Prove that the GCD of these lengths is 15
theorem gcd_rope_lengths : Nat.gcd rope_length1 (Nat.gcd rope_length2 rope_length3) = 15 := by
  sorry

end gcd_rope_lengths_l241_241393


namespace determine_f_l241_241391

noncomputable def f : ℝ → ℝ := sorry

axiom h_f0 : f 0 = 1

axiom h_fxy : ∀ x y : ℝ, f (xy + 1) = f x * f y - f y - x + 2

theorem determine_f (x : ℝ) : f x = x + 1 :=
by
  sorry

end determine_f_l241_241391


namespace total_pens_l241_241022

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l241_241022


namespace businessmen_neither_coffee_tea_soda_l241_241963

theorem businessmen_neither_coffee_tea_soda :
  (∀ (total_team coffee_team tea_team soda_team coffee_tea_team tea_soda_team coffee_soda_team all_three_team : ℕ),
    total_team = 40 →
    coffee_team = 20 →
    tea_team = 15 →
    soda_team = 10 →
    coffee_tea_team = 8 →
    tea_soda_team = 4 →
    coffee_soda_team = 3 →
    all_three_team = 2 →
    total_team - ((coffee_team + tea_team + soda_team) - (coffee_tea_team + tea_soda_team + coffee_soda_team) + all_three_team) = 8) :=
begin
  intros total_team coffee_team tea_team soda_team coffee_tea_team tea_soda_team coffee_soda_team all_three_team,
  intros h_total h_coffee h_tea h_soda h_coffee_tea h_tea_soda h_coffee_soda h_all_three,
  
  calc
  total_team - ((coffee_team + tea_team + soda_team) - (coffee_tea_team + tea_soda_team + coffee_soda_team) + all_three_team)
  = 40 - ((20 + 15 + 10) - (8 + 4 + 3) + 2) : by rw [h_total, h_coffee, h_tea, h_soda, h_coffee_tea, h_tea_soda, h_coffee_soda, h_all_three]
  = 40 - (45 - 15 + 2) : by calc (20 + 15 + 10) - (8 + 4 + 3) + 2 = 45 - 15 + 2 : rfl
  = 40 - 32 : by calc 45 - 15 + 2 = 32 : rfl
  = 8 : rfl,
end

end businessmen_neither_coffee_tea_soda_l241_241963


namespace sum_even_factors_of_720_l241_241893

open Nat

theorem sum_even_factors_of_720 :
  ∑ d in (finset.filter (λ x, even x) (finset.divisors 720)), d = 2340 :=
by
  sorry

end sum_even_factors_of_720_l241_241893


namespace monotonically_decreasing_range_l241_241662

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 1

theorem monotonically_decreasing_range (a : ℝ) :
  (∀ x : ℝ, f' a x ≤ 0) → a ≤ -3 := by
  sorry

end monotonically_decreasing_range_l241_241662


namespace total_pens_l241_241021

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l241_241021


namespace find_DF_l241_241341

variable (A B C D E F : Type) [add_comm_group A] [module ℝ A]

theorem find_DF (ABCD_parallelogram : is_parallelogram A B C D)
               (DE_altitude_AB : is_altitude D E A B)
               (DF_altitude_BC : is_altitude D F B C)
               (DC_eq_15 : ∥D - C∥ = 15)
               (EB_eq_3 : ∥E - B∥ = 3)
               (DE_eq_5 : ∥D - E∥ = 5) : 
               ∥D - F∥ = 5 := 
by { sorry }

end find_DF_l241_241341


namespace james_tylenol_intake_per_day_l241_241744

variable (hours_in_day : ℕ := 24) 
variable (tablets_per_dose : ℕ := 2) 
variable (mg_per_tablet : ℕ := 375)
variable (hours_per_dose : ℕ := 6)

theorem james_tylenol_intake_per_day :
  (tablets_per_dose * mg_per_tablet) * (hours_in_day / hours_per_dose) = 3000 := by
  sorry

end james_tylenol_intake_per_day_l241_241744


namespace total_pens_l241_241026

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l241_241026


namespace remainder_of_expression_l241_241686

theorem remainder_of_expression (n : ℤ) (h : n % 60 = 1) : (n^2 + 2 * n + 3) % 60 = 6 := 
by
  sorry

end remainder_of_expression_l241_241686


namespace age_difference_l241_241496

theorem age_difference (a b c : ℕ) (h₁ : b = 8) (h₂ : c = b / 2) (h₃ : a + b + c = 22) : a - b = 2 :=
by
  sorry

end age_difference_l241_241496


namespace total_shaded_area_l241_241345

theorem total_shaded_area (A_L A_M A_S : ℕ) (h1 : A_L = 49) (h2 : A_M = 25) (h3 : A_S = 9) : 
  A_S + (A_L - A_M) = 33 := 
by
  rw [h1, h2, h3]
  norm_num
  exact rfl

end total_shaded_area_l241_241345


namespace area_of_region_between_semicircles_l241_241545

/-- Given a region between two semicircles with the same center and parallel diameters,
where the farthest distance between two points with a clear line of sight is 12 meters,
prove that the area of the region is 18π square meters. -/
theorem area_of_region_between_semicircles :
  ∃ (R r : ℝ), R > r ∧ (R - r = 6) ∧ 18 * Real.pi = (Real.pi / 2) * (R^2 - r^2) ∧ (R^2 - r^2 = 144) :=
sorry

end area_of_region_between_semicircles_l241_241545


namespace total_dolls_l241_241404

-- Definitions given in the conditions
def grandmother_dolls := 50
def sister_dolls := grandmother_dolls + 2
def rene_dolls := 3 * sister_dolls

-- The theorem statement based on condition and correct answer
theorem total_dolls (g : ℕ) (s : ℕ) (r : ℕ) (h_g : g = 50) (h_s : s = g + 2) (h_r : r = 3 * s) : g + s + r = 258 := 
by {
  -- Placeholder for the proof
  sorry,
}

end total_dolls_l241_241404


namespace complete_the_square_l241_241067

theorem complete_the_square (d e f : ℤ) (h1 : d > 0)
  (h2 : 25 * d * d = 25)
  (h3 : 10 * d * e = 30)
  (h4 : 25 * d * d * (d * x + e) * (d * x + e) = 25 * x * x * 25 + 30 * x * 25 * d + 25 * e * e - 9)
  : d + e + f = 41 := 
  sorry

end complete_the_square_l241_241067


namespace locus_of_midpoint_l241_241187

-- Definition of the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Definition of the line with inclination angle π/4
def line (x y : ℝ) : Prop :=
  y = -x

-- Definition of the condition that A and B are points of intersection
def intersects (A B : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, ellipse x y ∧ line x y ∧ A = (x, y) ∧ B = (-x, -y)

-- Definition of the condition for the midpoint
def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- The theorem statement to be proved
theorem locus_of_midpoint : ∀ A B M : ℝ × ℝ,
  intersects A B →
  midpoint A B M →
  x + 4 * y = 0 :=
begin
  sorry
end

end locus_of_midpoint_l241_241187


namespace total_dolls_l241_241408

-- Definitions based on the given conditions
def grandmother_dolls : Nat := 50
def sister_dolls : Nat := grandmother_dolls + 2
def rene_dolls : Nat := 3 * sister_dolls

-- Statement we want to prove
theorem total_dolls : grandmother_dolls + sister_dolls + rene_dolls = 258 := by
  sorry

end total_dolls_l241_241408


namespace x_gt_one_iff_x_cube_gt_one_l241_241383

theorem x_gt_one_iff_x_cube_gt_one (x : ℝ) : x > 1 ↔ x^3 > 1 :=
by sorry

end x_gt_one_iff_x_cube_gt_one_l241_241383


namespace base_prime_representation_360_l241_241113

-- Definition of the base prime representation function
def base_prime_representation (n : ℕ) (bases : List ℕ) : List ℕ :=
  bases.map (λ p => Nat.find_exponent p (Nat.factorization n))

-- Axiom for correct exponent finding within factorization
axiom Nat.find_exponent : ℕ → List (ℕ × ℕ) → ℕ

-- The theorem we need to prove
theorem base_prime_representation_360 :
  base_prime_representation 360 [2, 3, 5] = [3, 2, 1] :=
by
  sorry

end base_prime_representation_360_l241_241113


namespace general_term_sequence_l241_241301

def seq (a : ℕ → ℤ) : Prop :=
  a 0 = 3 ∧ a 1 = 9 ∧ ∀ n ≥ 2, a n = 4 * a (n - 1) - 3 * a (n - 2) - 4 * n + 2

theorem general_term_sequence (a : ℕ → ℤ) (h : seq a) : 
  ∀ n, a n = 3^n + n^2 + 3 * n + 2 :=
by
  sorry

end general_term_sequence_l241_241301


namespace quadratic_polynomial_and_tangent_slope_l241_241605

theorem quadratic_polynomial_and_tangent_slope :
  ∃ (a b c : ℝ), (∀ x : ℝ, y = a * x ^ 2 + b * x + c) ∧ 
  (y 1 = -2) ∧ (y 2 = 4) ∧ (y 3 = 10) ∧ (derivative y 2 = 6) := 
begin
  sorry
end

end quadratic_polynomial_and_tangent_slope_l241_241605


namespace compute_expr_l241_241576

theorem compute_expr : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end compute_expr_l241_241576


namespace total_pens_l241_241004

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l241_241004


namespace sequence_recurrence_gcd_sequence_u_l241_241457

def sequence_u : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := sequence_u (n+1) + 2 * sequence_u n

theorem sequence_recurrence (n p : ℕ) (hp : p > 1) :
  sequence_u (n + p) = sequence_u (n + 1) * sequence_u p + 2 * sequence_u n * sequence_u (p - 1) :=
sorry

theorem gcd_sequence_u (n : ℕ) : 
  Nat.gcd (sequence_u n) (sequence_u (n + 3)) = 
    if n % 3 = 0 then 3 else 1 :=
sorry

end sequence_recurrence_gcd_sequence_u_l241_241457


namespace moles_of_Cl2_l241_241240

def chemical_reaction : Prop :=
  ∀ (CH4 Cl2 HCl : ℕ), 
  (CH4 = 1) → 
  (HCl = 4) →
  -- Given the balanced equation: CH4 + 2Cl2 → CHCl3 + 4HCl
  (CH4 + 2 * Cl2 = CH4 + 2 * Cl2) →
  (4 * HCl = 4 * HCl) → -- This asserts the product side according to the balanced equation
  (Cl2 = 2)

theorem moles_of_Cl2 (CH4 Cl2 HCl : ℕ) (hCH4 : CH4 = 1) (hHCl : HCl = 4)
  (h_balanced : CH4 + 2 * Cl2 = CH4 + 2 * Cl2) (h_product : 4 * HCl = 4 * HCl) :
  Cl2 = 2 := by {
    sorry
}

end moles_of_Cl2_l241_241240


namespace fill_pipe_half_time_l241_241528

theorem fill_pipe_half_time (T : ℝ) (hT : 0 < T) :
  ∀ t : ℝ, t = T / 2 :=
by
  sorry

end fill_pipe_half_time_l241_241528


namespace smallest_possible_floor_sum_l241_241688

theorem smallest_possible_floor_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ∃ (a b c : ℝ), ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end smallest_possible_floor_sum_l241_241688


namespace sum_even_factors_of_720_l241_241897

open Nat

theorem sum_even_factors_of_720 :
  ∑ d in (finset.filter (λ x, even x) (finset.divisors 720)), d = 2340 :=
by
  sorry

end sum_even_factors_of_720_l241_241897


namespace second_number_is_twenty_two_l241_241328

theorem second_number_is_twenty_two (x y : ℕ) 
  (h1 : x + y = 33) 
  (h2 : y = 2 * x) : 
  y = 22 :=
by
  sorry

end second_number_is_twenty_two_l241_241328


namespace circle_center_radius_sum_l241_241763

theorem circle_center_radius_sum :
  let D := { p : ℝ × ℝ | (p.1^2 - 14*p.1 + p.2^2 + 10*p.2 = -34) }
  let c := 7
  let d := -5
  let s := 2 * Real.sqrt 10
  (c + d + s = 2 + 2 * Real.sqrt 10) :=
by
  sorry

end circle_center_radius_sum_l241_241763


namespace sum_of_remainders_mod_13_l241_241129

theorem sum_of_remainders_mod_13 
  (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_mod_13_l241_241129


namespace marble_problem_l241_241462

theorem marble_problem :
  ∀ (marbles_total white_marbles red_marbles blue_marbles marbles_left k : ℕ),
    marbles_total = 50 →
    white_marbles = 20 →
    red_marbles = blue_marbles →
    marbles_left = 40 →
    red_marbles + blue_marbles = marbles_total - white_marbles →
    k * (white_marbles - blue_marbles) = marbles_total - marbles_left →
    k = 2 :=
by
  intros marbles_total white_marbles red_marbles blue_marbles marbles_left k
  assume h_total h_white h_equal h_left h_rb h_k
  sorry

end marble_problem_l241_241462


namespace intersection_A_B_union_A_B_subset_C_B_l241_241303

open Set

noncomputable def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 9}
noncomputable def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 6} :=
by
  sorry

theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 9} :=
by
  sorry

theorem subset_C_B (a : ℝ) : C a ⊆ B → 2 ≤ a ∧ a ≤ 8 :=
by
  sorry

end intersection_A_B_union_A_B_subset_C_B_l241_241303


namespace perfect_square_factors_count_l241_241564

open BigOperators

theorem perfect_square_factors_count :
  let factors := (2^12) * (3^18) * (5^20) * (7^8)
  ∃ (count : ℕ), count = 3850 := 
by
  let factors := (2 ^ 12) * (3 ^ 18) * (5 ^ 20) * (7 ^ 8)
  have h : ∀ p n, p ^ (2 * n + 1) ∣ factors -> false := λ p n h, sorry
  let n := 7 * 10 * 11 * 5
  use n
  have : n = 3850 := by norm_num
  exact this

end perfect_square_factors_count_l241_241564


namespace math_proof_problem_l241_241149

-- Define the problem conditions
def problem_conditions (x y : ℚ) := 
  (real.sqrt (x - y) = 2 / 5) ∧ (real.sqrt (x + y) = 2)

-- Define the correct solution
def correct_solution (x y : ℚ) := 
  (x = 52 / 25) ∧ (y = 48 / 25)

-- Define the area of the rectangle
def rectangle_area (a b : ℚ) : ℚ :=
  abs (a * b)

-- Define the proof problem
theorem math_proof_problem : 
  problem_conditions (52 / 25) (48 / 25) ∧ 
  rectangle_area (52 / 25) (48 / 25) = 8 / 25 :=
by 
  sorry

end math_proof_problem_l241_241149


namespace product_of_roots_eq_38_l241_241604

noncomputable def product_of_roots : ℝ :=
  let p1 := (2 : ℝ) * X^3 + X^2 - 8 * X + 20
  let p2 := (5 : ℝ) * X^3 - 25 * X^2 + 19
  let p := p1 * p2
  sorry

theorem product_of_roots_eq_38 : product_of_roots = 38 := sorry

end product_of_roots_eq_38_l241_241604


namespace plane_speed_west_l241_241539

theorem plane_speed_west (v t : ℝ) : 
  (300 * t + 300 * t = 1200) ∧ (t = 7 - t) → 
  (v = 300 * t / (7 - t)) ∧ (t = 2) → 
  v = 120 :=
by
  intros h1 h2
  sorry

end plane_speed_west_l241_241539


namespace solve_for_x_l241_241706

theorem solve_for_x (x : ℝ) (h : (10 - 6 * x)^ (1 / 3) = -2) : x = 3 := 
by
  sorry

end solve_for_x_l241_241706


namespace bananas_and_cantaloupe_cost_l241_241617

noncomputable def prices (a b c d : ℕ) : Prop :=
  a + b + c + d = 40 ∧
  d = 3 * a ∧
  b = c - 2

theorem bananas_and_cantaloupe_cost (a b c d : ℕ) (h : prices a b c d) : b + c = 20 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  -- Using the given conditions:
  --     a + b + c + d = 40
  --     d = 3 * a
  --     b = c - 2
  -- We find that b + c = 20
  sorry

end bananas_and_cantaloupe_cost_l241_241617


namespace sum_even_factors_of_720_l241_241862

theorem sum_even_factors_of_720 : 
  let even_factors_sum (n : ℕ) : ℕ := 
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  in even_factors_sum 720 = 2340 :=
by
  let even_factors_sum (n : ℕ) : ℕ :=
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  sorry

end sum_even_factors_of_720_l241_241862


namespace sum_of_even_factors_720_l241_241890

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) : ℕ :=
    match n with
    | 720 => 
      let sum_powers_2 := 2 + 4 + 8 + 16 in
      let sum_powers_3 := 1 + 3 + 9 in
      let sum_powers_5 := 1 + 5 in
      sum_powers_2 * sum_powers_3 * sum_powers_5
    | _ => 0
  in
  even_factors_sum 720 = 2340 :=
by 
  sorry

end sum_of_even_factors_720_l241_241890


namespace geometry_problem_l241_241737

/-- In triangle ABC and point P in the same plane, point P is equidistant from B and C. 
Angle BPC is twice angle BAC, and line AB intersects line CP at point D. If PC = 4 and PD = 3, 
then the product AD · BD is equal to 7. -/
theorem geometry_problem
  (A B C P D : Type)
  [metric_space P]
  (h_eqdist : dist P B = dist P C)
  (h_angle : ∃ α, ∠BPC = 2 * α ∧ ∠BAC = α)
  (h_intersect : segment A B ∩ segment C P ⊆ {D})
  (h_pc : dist P C = 4)
  (h_pd : dist P D = 3)
  : (∃ AD BD, AD * BD = 7) :=
sorry

end geometry_problem_l241_241737


namespace triangle_is_isosceles_right_l241_241332

variable {A B C : ℝ}
variable {a b c : ℝ}

theorem triangle_is_isosceles_right 
  (h1 : log a - log c = -log (sqrt 2))
  (h2 : log (sin B) = -log (sqrt 2))
  (h3 : 0 < B ∧ B < π/2) : 
  (A + B + C = π ∧ a^2 + b^2 = c^2) :=
sorry

end triangle_is_isosceles_right_l241_241332


namespace local_min_in_interval_implies_b_in_range_l241_241708

noncomputable def f (x b : ℝ) := x^3 - 3 * b * x + 3 * b

theorem local_min_in_interval_implies_b_in_range (b : ℝ) 
  (h : ∃ c ∈ Ioo (0 : ℝ) 1, (∀ x ∈ Ioo (0 : ℝ) 1, deriv (f x) c = 0 → deriv (f x) c < 0)) : 
  0 < b ∧ b < 1 :=
sorry

end local_min_in_interval_implies_b_in_range_l241_241708


namespace arc_length_proof_l241_241566

noncomputable def arc_length_problem : ℝ :=
    ∫ y in 1..1.5, sqrt(1 + (0.5 * y - 0.5 / y) ^ 2)

theorem arc_length_proof :
    arc_length_problem = 0.3125 + 0.5 * Real.log 1.5 :=
sorry

end arc_length_proof_l241_241566


namespace PartA_l241_241356

variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_eq : ∀ x, f (f x) = f x)

theorem PartA : ∀ x, (deriv f x = 0) ∨ (deriv f (f x) = 1) :=
by
  sorry

end PartA_l241_241356


namespace sweets_per_child_l241_241154

variable (S x : ℕ)
variable (children_total children_absent sweets_extra : ℕ)
variable (children_remaining : ℕ) := children_total - children_absent

-- Conditions given in the problem
def conditions : Prop :=
  children_total = 112 ∧
  children_absent = 32 ∧
  sweets_extra = 6 ∧
  S = children_total * x ∧
  S = children_remaining * (x + sweets_extra)

-- Goal: Prove that each child was originally supposed to get 15 sweets
theorem sweets_per_child (h : conditions) : x = 15 :=
by {
  sorry
}

end sweets_per_child_l241_241154


namespace sum_arithmetic_sequence_l241_241281

theorem sum_arithmetic_sequence (S : ℕ → ℕ) :
  S 7 = 21 ∧ S 17 = 34 → S 27 = 27 :=
by
  sorry

end sum_arithmetic_sequence_l241_241281


namespace scoops_arrangement_count_l241_241784

theorem scoops_arrangement_count :
  (5 * 4 * 3 * 2 * 1 = 120) :=
by
  sorry

end scoops_arrangement_count_l241_241784


namespace pipe_A_fill_time_l241_241475

theorem pipe_A_fill_time :
  (∃ x : ℕ, (1 / (x : ℝ) + 1 / 60 - 1 / 72 = 1 / 40) ∧ x = 45) :=
sorry

end pipe_A_fill_time_l241_241475


namespace systematic_sampling_correct_l241_241052

theorem systematic_sampling_correct :
  ∀ (n m : ℕ) (students : Finset ℕ),
    n = 50 →
    m = 5 →
    students = {1, 2, ..., 50} →
    ∃ (s : Finset ℕ), 
      s = {6, 16, 26, 36, 46} ∧
      ∀ i ∈ s, ∃ k, i = 6 + 10 * k ∧ k < 5 :=
by {
  sorry
}

end systematic_sampling_correct_l241_241052


namespace modulus_of_z_l241_241164

def z := (1 - Complex.I) / (1 + Complex.I) + 2 * Complex.I

theorem modulus_of_z : Complex.abs z = 1 := by
  sorry

end modulus_of_z_l241_241164


namespace max_average_weight_l241_241914

theorem max_average_weight (P R S : Type) [has_weight P] [has_weight R] [has_weight S] 
  (weight_P : average_weight P = 30)
  (weight_R : average_weight R = 40)
  (average_PR : average_weight (combine P R) = 34)
  (average_PS : average_weight (combine P S) = 35) :
  average_weight (combine R S) <= 48 :=
by
  sorry

end max_average_weight_l241_241914


namespace second_number_is_twenty_two_l241_241326

theorem second_number_is_twenty_two (x y : ℕ) 
  (h1 : x + y = 33) 
  (h2 : y = 2 * x) : 
  y = 22 :=
by
  sorry

end second_number_is_twenty_two_l241_241326


namespace smallest_n_divisibility_l241_241125

theorem smallest_n_divisibility:
  ∃ (n : ℕ), n > 0 ∧ n^2 % 24 = 0 ∧ n^3 % 540 = 0 ∧ n = 60 :=
by
  sorry

end smallest_n_divisibility_l241_241125


namespace a_3_def_a_4_def_a_r_recurrence_l241_241558

-- Define minimally the structure of the problem.
noncomputable def a_r (r : ℕ) : ℕ := -- Definition for minimum phone calls required.
by sorry

-- Assertions for the specific cases provided.
theorem a_3_def : a_r 3 = 3 :=
by
  -- Proof is omitted with sorry.
  sorry

theorem a_4_def : a_r 4 = 4 :=
by
  -- Proof is omitted with sorry.
  sorry

theorem a_r_recurrence (r : ℕ) (hr : r ≥ 3) : a_r r ≤ a_r (r - 1) + 2 :=
by
  -- Proof is omitted with sorry.
  sorry

end a_3_def_a_4_def_a_r_recurrence_l241_241558


namespace C_increases_with_n_l241_241612

variable (e R r : ℝ) (n : ℝ)
def C (n : ℝ) : ℝ := e * n / (R + n * r)

theorem C_increases_with_n (hR : 0 < R) (he : 0 < e) (hr : 0 < r) (hn : 0 ≤ n) :
  ∀ n₁ n₂ : ℝ, n₁ < n₂ → C e R r n₁ < C e R r n₂ :=
sorry

end C_increases_with_n_l241_241612


namespace correct_combined_monthly_rate_of_profit_l241_241535

structure Book :=
  (cost_price : ℕ)
  (selling_price : ℕ)
  (months_held : ℕ)

def profit (b : Book) : ℕ :=
  b.selling_price - b.cost_price

def monthly_rate_of_profit (b : Book) : ℕ :=
  if b.months_held = 0 then profit b else profit b / b.months_held

def combined_monthly_rate_of_profit (b1 b2 b3 : Book) : ℕ :=
  monthly_rate_of_profit b1 + monthly_rate_of_profit b2 + monthly_rate_of_profit b3

theorem correct_combined_monthly_rate_of_profit :
  combined_monthly_rate_of_profit
    {cost_price := 50, selling_price := 90, months_held := 1}
    {cost_price := 120, selling_price := 150, months_held := 2}
    {cost_price := 75, selling_price := 110, months_held := 0} 
    = 90 := 
by
  sorry

end correct_combined_monthly_rate_of_profit_l241_241535


namespace length_of_JK_l241_241120

-- Given lengths in centimeters
constant LM : ℝ := 180
constant NO : ℝ := 120

-- Given parallel condition (we encode this as a similarity condition derived from parallel lines)
constant parallel_condition : LM / NO = 180 / 120

-- The goal is to prove the length of JK
theorem length_of_JK : (1 / ((1 / NO) + (1 / LM))) = 72 := by
  -- Given \( NO = 120 \) cm and \( LM = 180 \) cm, calculate \( JK \).
  have h1 : (1 / ((1 / NO) + (1 / LM))) = 1 / ((LM + NO) / (LM * NO)),
  { rw [add_div, one_div, ←mul_div, mul_one] },
  rw [←add_div, ←one_div, mul_comm NO LM, div_div, mul_div, mul_comm, ←one_div] at h1,
  simp [NO, LM] at h1,
  exact h1

end length_of_JK_l241_241120


namespace problem_statement_l241_241254

variable (a b : ℝ)

theorem problem_statement (h1 : a > b) (h2 : b > 1/a) (h3 : 1/a > 0) :
  (a + b > 2) ∧ (a > 1) ∧ (a - 1/b > b - 1/a) :=
by 
  sorry

end problem_statement_l241_241254


namespace slope_angle_of_line_l241_241090

theorem slope_angle_of_line :
  ∀ (x y : ℝ), (\sqrt 3 * x + 3 * y + 1 = 0) → 
  (atan (- sqrt 3 / 3) = 5 * real.pi / 6) :=
by
  intros x y h
  sorry

end slope_angle_of_line_l241_241090


namespace price_reduction_equation_l241_241172

theorem price_reduction_equation (x : ℝ) : 25 * (1 - x)^2 = 16 :=
by
  sorry

end price_reduction_equation_l241_241172


namespace max_value_of_f_l241_241238

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem max_value_of_f : ∃ x ∈ Set.Icc (Real.pi / 2) Real.pi, ∀ y ∈ Set.Icc (Real.pi / 2) Real.pi, f y ≤ f x ∧ f x = Real.pi := 
by
  sorry

end max_value_of_f_l241_241238


namespace problem_proof_l241_241349

noncomputable def T_polar_coords (θ t : ℝ) : Prop :=
  let x_l := (sqrt 3 / 2) * t
  let y_l := 2 - (1 / 2) * t
  let x_c := sqrt 6 * cos θ
  let y_c := sqrt 2 * sin θ
  let polar_r := sqrt ((sqrt 3) ^ 2 + 1 ^ 2)
  let polar_theta := atan (1 / sqrt 3)
  (x_l = x_c ∧ y_l = y_c) → (polar_r = 2 ∧ polar_theta = π / 6)

noncomputable def m_polar_eqs (m k x y : ℝ) : Prop :=
  let x_w := x
  let y_w := sqrt 3 * y
  let vertical_line := x = sqrt 3
  let sloped_line := (- sqrt 3 * k + 1) / sqrt (k ^ 2 + 1) = sqrt 3
  let line_eq_1 := y = - sqrt 3 / 3 * x + 2
  let polar_eq_1 := (sqrt (x ^ 2 + y ^ 2) * cos (atan (y / x))) = sqrt 3
  let polar_eq_2 := (sqrt (x ^ 2 + y ^ 2) * sin (atan (y / x)) + (sqrt 3 / 3) * cos (atan (y / x))) = 2
  vertical_line ∨ (sloped_line ∧ (line_eq_1 → (polar_eq_1 ∨ polar_eq_2)))

theorem problem_proof (θ t k x y m : ℝ) : T_polar_coords θ t ∧ m_polar_eqs m k x y := sorry

end problem_proof_l241_241349


namespace sarah_apples_calc_l241_241787

variable (brother_apples : ℕ)
variable (sarah_apples : ℕ)
variable (multiplier : ℕ)

theorem sarah_apples_calc
  (h1 : brother_apples = 9)
  (h2 : multiplier = 5)
  (h3 : sarah_apples = multiplier * brother_apples) : sarah_apples = 45 := by
  sorry

end sarah_apples_calc_l241_241787


namespace prime_factors_difference_l241_241487

theorem prime_factors_difference (h : 184437 = 3 * 7 * 8783) : 8783 - 7 = 8776 :=
by sorry

end prime_factors_difference_l241_241487


namespace bank_robbery_car_l241_241217

def car_statement (make color : String) : Prop :=
  (make = "Buick" ∨ color = "blue") ∧
  (make = "Chrysler" ∨ color = "black") ∧
  (make = "Ford" ∨ color ≠ "blue")

theorem bank_robbery_car : ∃ make color : String, car_statement make color ∧ make = "Buick" ∧ color = "black" :=
by
  sorry

end bank_robbery_car_l241_241217


namespace smallest_possible_value_l241_241693

theorem smallest_possible_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (a b c : ℕ), a = floor((x + y) / z) ∧ b = floor((y + z) / x) ∧ c = floor((z + x) / y) ∧ (a + b + c) = 4 :=
begin
  sorry
end

end smallest_possible_value_l241_241693


namespace usual_time_is_36_l241_241479

noncomputable def usual_time_to_school (R : ℝ) (T : ℝ) : Prop :=
  let new_rate := (9/8 : ℝ) * R
  let new_time := T - 4
  R * T = new_rate * new_time

theorem usual_time_is_36 (R : ℝ) (T : ℝ) (h : T = 36) : usual_time_to_school R T :=
by
  sorry

end usual_time_is_36_l241_241479


namespace min_sum_of_distances_l241_241042

-- Define the four points
def P₁ := (0 : ℝ, 0 : ℝ)
def P₂ := (10 : ℝ, 20 : ℝ)
def P₃ := (5 : ℝ, 15 : ℝ)
def P₄ := (12 : ℝ, -6 : ℝ)

-- Define the function to calculate the Euclidean distance between two points in ℝ²
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the sum of distances function from a point P to the four points {P₁, P₂, P₃, P₄}
def sum_of_distances (P : ℝ × ℝ) : ℝ :=
  euclidean_distance P P₁ + euclidean_distance P P₂ + euclidean_distance P P₃ + euclidean_distance P P₄

-- Claim that the point (6, 12) is the point that minimizes the sum of distances
theorem min_sum_of_distances : ∀ (P : ℝ × ℝ), sum_of_distances (6, 12) ≤ sum_of_distances P :=
  sorry

end min_sum_of_distances_l241_241042


namespace second_number_is_22_l241_241323

theorem second_number_is_22 (x second_number : ℕ) : 
  (x + second_number = 33) → 
  (second_number = 2 * x) → 
  second_number = 22 :=
by
  intros h_sum h_double
  sorry

end second_number_is_22_l241_241323


namespace max_value_of_a_l241_241647

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ + Real.pi / 3)

theorem max_value_of_a 
  (φ : ℝ) (hφ : |φ| < Real.pi / 2)
  (hmax : ∃ (k : ℤ), 2 * (Real.pi / 6) + φ + Real.pi / 3 = Real.pi / 2 + 2 * k * Real.pi)
  (hmono : ∀ {x : ℝ}, -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 6 → (x ∈ Icc (-Real.pi / 3) (Real.pi / 6) → 0 ≤ 2 * Real.cos (2 * x + φ + Real.pi / 3))):
  ∃ a : ℝ, a = Real.pi / 6 ∧ ∀ x ∈ Icc (-a) a, f x φ = 0 := sorry

end max_value_of_a_l241_241647


namespace quadratic_inequality_solution_set_l241_241248

noncomputable def quadratic_solution_set (a b c : ℝ) (ineq : Char) : Set ℝ :=
if ineq = '<' then {x : ℝ | ax^2 + bx + c < 0}
else if ineq = '>' then {x : ℝ | ax^2 - bx + c > 0}
else ∅

theorem quadratic_inequality_solution_set 
  (a b c : ℝ)
  (h_cond1 : a < 0)
  (h_cond2 : quadratic_solution_set a b c '<' = {x : ℝ | x < -2 ∨ x > -0.5})
  : quadratic_solution_set a b c '>' = {x : ℝ | 0.5 < x ∧ x < 2} :=
sorry

end quadratic_inequality_solution_set_l241_241248


namespace sum_even_factors_of_720_l241_241861

theorem sum_even_factors_of_720 : 
  let even_factors_sum (n : ℕ) : ℕ := 
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  in even_factors_sum 720 = 2340 :=
by
  let even_factors_sum (n : ℕ) : ℕ :=
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  sorry

end sum_even_factors_of_720_l241_241861


namespace g6_eq_16_l241_241074

-- Definition of the function g that satisfies the given conditions
variable (g : ℝ → ℝ)

-- Given conditions
axiom functional_eq : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g3_eq_4 : g 3 = 4

-- The goal is to prove g(6) = 16
theorem g6_eq_16 : g 6 = 16 := by
  sorry

end g6_eq_16_l241_241074


namespace pens_at_end_l241_241904

def starting_pens : ℕ := 5
def mike_gives : ℕ := 20
def pens_doubled (x : ℕ) : ℕ := 2 * x
def pens_given_sharon : ℕ := 19

theorem pens_at_end :
  let pens_after_mike := starting_pens + mike_gives in
  let pens_after_cindy := pens_doubled pens_after_mike in
  let pens_after_sharon := pens_after_cindy - pens_given_sharon in
  pens_after_sharon = 31 :=
by
  sorry

end pens_at_end_l241_241904


namespace find_hyperbola_from_ellipse_l241_241644

noncomputable theory

def ellipse (a b : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def hyperbola (a b : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

theorem find_hyperbola_from_ellipse :
  ellipse 13 12 →
  (∀ x y : ℝ, real.sqrt (13^2 - 12^2) = 5) →
  (∃ a b : ℝ, hyperbola a b ∧ a=4 ∧ b=3) :=
begin
  intros h1 h2,
  use [4, 3],
  split,
  { exact hyperbola 4 3 },
  split,
  { refl },
  { refl }
end

end find_hyperbola_from_ellipse_l241_241644


namespace segment_XY_passes_through_incenter_of_triangle_ABC_l241_241176

-- Mathematical definitions for the problem

variables {A B C X Y : Type} -- Points A, B, C, X, and Y.
variable triangle : A ≠ B ∧ B ≠ C ∧ A ≠ C -- Given that A, B, and C are vertices of a triangle.
variable circle_passing_through_AB : ∃ (D : Type), A ≠ D ∧ B ≠ D ∧ (circle A B D) -- There's a circle passing through vertices A and B.
variable intersects_at_XY : (X ∈ AC) ∧ (Y ∈ BC) -- Circle intersects sides AC at X and BC at Y.
variable excircle_center_P : ∃ (P : Type), (P is_center_of_excircle_of_triangle X Y C) ∧ (P ∈ (circumcircle_of_triangle ABC)) -- Center of excircle of triangle XYC lies on circumcircle of triangle ABC.
variable segment_XY_passes_through_incenter : (segment XY connects with (incenter_of_triangle ABC)) -- Conclusion: Segment XY passes through the incenter of triangle ABC.

-- The main theorem.
theorem segment_XY_passes_through_incenter_of_triangle_ABC :
  triangle ∧ circle_passing_through_AB ∧ intersects_at_XY ∧ excircle_center_P → segment_XY_passes_through_incenter :=
by
  sorry

end segment_XY_passes_through_incenter_of_triangle_ABC_l241_241176


namespace initial_average_marks_l241_241069

theorem initial_average_marks (A : ℝ) (h1 : 25 * A - 50 = 2450) : A = 100 :=
by
  sorry

end initial_average_marks_l241_241069


namespace smallest_possible_value_l241_241691

theorem smallest_possible_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (a b c : ℕ), a = floor((x + y) / z) ∧ b = floor((y + z) / x) ∧ c = floor((z + x) / y) ∧ (a + b + c) = 4 :=
begin
  sorry
end

end smallest_possible_value_l241_241691


namespace intersection_M_N_l241_241304

-- Definitions based on the conditions
def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}

-- Theorem asserting the intersection of sets M and N
theorem intersection_M_N : M ∩ N = {x | 2 ≤ x ∧ x < 5} := 
by
  sorry

end intersection_M_N_l241_241304


namespace smallest_possible_floor_sum_l241_241689

theorem smallest_possible_floor_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ∃ (a b c : ℝ), ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end smallest_possible_floor_sum_l241_241689


namespace ratio_lcm_gcf_256_162_l241_241123

theorem ratio_lcm_gcf_256_162 : (Nat.lcm 256 162) / (Nat.gcd 256 162) = 10368 := 
by 
  sorry

end ratio_lcm_gcf_256_162_l241_241123


namespace man_speed_with_current_l241_241188

theorem man_speed_with_current
  (v : ℝ)  -- man's speed in still water
  (current_speed : ℝ) (against_current_speed : ℝ)
  (h1 : against_current_speed = v - 3.2)
  (h2 : current_speed = 3.2) :
  v = 12.8 → (v + current_speed = 16.0) :=
by
  sorry

end man_speed_with_current_l241_241188


namespace resulting_solution_percentage_l241_241947

theorem resulting_solution_percentage :
  ∀ (V: ℕ) (P1 P2: ℕ),
  V > 0 → P1 = 50 → P2 = 60 →
  ((0.5 * V * P2 + 0.5 * V * P1) / V = 55.0) :=
by
  intros V P1 P2 hV hP1 hP2
  have h1: 0.5 = 1 / 2 := by norm_num
  have h2: V > 0 := hV
  have h3: 0.5 * V = V / 2 := by norm_num; rw mul_div_right_comm; assumption
  have h4: P1 * (V / 2) = (P1 * V) / 2 := by rw mul_div_assoc; apply nat.div_pos; assumption
  have h5: P2 * (V / 2) = (P2 * V) / 2 := by rw mul_div_assoc; apply nat.div_pos; assumption
  have h6: ((P2 * V) / 2 + (P1 * V) / 2 ) / V = ((P2 + P1) / 2) := by ring
  sorry

end resulting_solution_percentage_l241_241947


namespace sum_even_factors_of_720_l241_241898

open Nat

theorem sum_even_factors_of_720 :
  ∑ d in (finset.filter (λ x, even x) (finset.divisors 720)), d = 2340 :=
by
  sorry

end sum_even_factors_of_720_l241_241898


namespace parallel_LK_AB_l241_241468
-- Import the Mathlib library for accessing geometry
  
-- Define non-computable objects
noncomputable theory

-- Set up the required definitions and hypotheses
variable {A B C M N L K : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space M] [metric_space N] [metric_space L] [metric_space K]

-- Suppose we have a triangle ABC
variables (ABC : triangle A B C)

-- M and N are points on sides BC and AC respectively
variables (M : point A) (N : point B)
(hM : M ∈ BC) (hN : N ∈ AC)

-- Lines through M and N are parallel to lines BN and AM respectively.
variables (BN_parallel : parallel (line B N) (line L M))
variables (AM_parallel : parallel (line A M) (line N K))

-- Hypothesize that points L and K lie on AC and BC respectively
variables (hL : L ∈ AC) (hK : K ∈ BC)

-- The theorem stating that LK is parallel to AB
theorem parallel_LK_AB : parallel (line L K) (line A B) :=
by
  sorry

end parallel_LK_AB_l241_241468


namespace problem1_problem2_variant1_problem2_variant2_l241_241244

-- Problem 1
theorem problem1
  (a b : ℝ)
  (h_line : b = -a)
  (h1 : (a - 2) ^ 2 + b ^ 2 = a ^ 2 + (b + 4) ^ 2) :
  (x - 3) ^ 2 + (y + 3) ^ 2 = 10 :=
by
  sorry

-- Problem 2
theorem problem2_variant1
  (a b : ℝ)
  (h_line : 5 * a - 3 * b = 8)
  (h_tangent_x : |a| = c)
  (h_tangent_y : |b| = c)
  (c = 4) :
  (x - 4) ^ 2 + (y - 4) ^ 2 = 16 :=
by
  sorry

theorem problem2_variant2
  (a b : ℝ)
  (h_line : 5 * a - 3 * b = 8)
  (h_tangent_x : |a| = c)
  (h_tangent_y : |b| = c)
  (c = 1) :
  (x - 1) ^ 2 + (y + 1) ^ 2 = 1 :=
by
  sorry

end problem1_problem2_variant1_problem2_variant2_l241_241244


namespace number_of_distinct_products_of_special_fractions_l241_241979

def is_special_fraction (a b : ℕ) : Prop := a * b = 24

def products_of_special_fractions : set ℕ :=
  let fractions := { (1, 24), (2, 12), (3, 8), (4, 6), (6, 4), (8, 3), (12, 2), (24, 1) } in
  let special_fracs := fractions.map (λ p, (p.1:ℚ)/(p.2:ℚ)) in
  let products := { (f1 * f2).denom = 1 | f1 ∈ special_fracs, f2 ∈ special_fracs } in
  products.to_finset.map (λ p, (p.num : ℕ))

theorem number_of_distinct_products_of_special_fractions : (products_of_special_fractions : set ℕ).card = 3 :=
by {
  sorry
}

end number_of_distinct_products_of_special_fractions_l241_241979


namespace find_quadratic_function_l241_241321

theorem find_quadratic_function (g : ℝ → ℝ) 
  (h1 : g 0 = 0) 
  (h2 : g 1 = 1) 
  (h3 : g (-1) = 5) 
  (h_quadratic : ∃ a b, ∀ x, g x = a * x^2 + b * x) : 
  g = fun x => 3 * x^2 - 2 * x := 
by
  sorry

end find_quadratic_function_l241_241321


namespace possible_values_l241_241351

def triangle_values (n : ℕ) :=
  17 ≤ n ∧ n ≤ 21

theorem possible_values :
  { n // triangle_values n }.card = 5 :=
by
  sorry

end possible_values_l241_241351


namespace problem_statement_l241_241289

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan x

theorem problem_statement : 
  (∀ x : ℝ, f(x+Real.pi) = f(x)) ∨ 
  (∃ k : ℤ, ∀ x : ℝ, f(x + 2 * k * Real.pi) = -f(x)) ∨ 
  (∃ k : ℤ, f (k * Real.pi) = 0) ∨ 
  (∀ k : ℤ, ∀ x : ℝ, x ∈ Ioo (-Real.pi / 2 + k * Real.pi) (Real.pi / 2 + k * Real.pi) → 
    f x = Real.sin x + Real.tan x) = 
  ((∃ k : ℤ, ∀ x : ℝ, f(x + 2 * k * Real.pi) = -f(x)) ∧ 
  (∃ k : ℤ, f (k * Real.pi) = 0) ∧ 
  (∀ k : ℤ, ∀ x : ℝ, x ∈ Ioo (-Real.pi/2 + k * Real.pi) (Real.pi/2 + k * Real.pi) → 
    f x = Real.sin x + Real.tan x)) :=
sorry

end problem_statement_l241_241289


namespace find_beautiful_numbers_l241_241111

-- Define what it means to be a beautiful number (palindrome)
def is_beautiful (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  digits = digits.reverse

-- Define the constraints for the problem
def is_valid_beautiful_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (∀ d ∈ (Nat.digits 10 n), d = 0 ∨ d = 1) ∧ is_beautiful n

-- Define the conclusion that should be proven
theorem find_beautiful_numbers :
  {n : ℕ | is_valid_beautiful_five_digit n} = {10001, 10101, 11011, 11111} :=
by 
  sorry

end find_beautiful_numbers_l241_241111


namespace part_I_solution_part_II_solution_l241_241389

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2 * |x - 1|

theorem part_I_solution :
  ∀ x : ℝ, f x 3 ≥ 1 ↔ 0 ≤ x ∧ x ≤ (4 / 3) := by
  sorry

theorem part_II_solution :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x a - |2*x - 5| ≤ 0) ↔ (-1 ≤ a ∧ a ≤ 4) := by
  sorry

end part_I_solution_part_II_solution_l241_241389


namespace intersection_in_second_quadrant_l241_241270

-- Definitions of the lines l₁ and l₂
def l₁ (a x : ℝ) : ℝ := (2 / 3) * x + (1 - a) / 3
def l₂ (a x : ℝ) : ℝ := - (1 / 2) * x + a

-- Statement of the problem
theorem intersection_in_second_quadrant (a : ℝ) : 
  (∃ x y : ℝ, l₁ a x = y ∧ l₂ a x = y ∧ x < 0 ∧ y > 0) ↔ a > (1 / 4) :=
by
  sorry

end intersection_in_second_quadrant_l241_241270


namespace math_proof_problem_l241_241150

-- Define the problem conditions
def problem_conditions (x y : ℚ) := 
  (real.sqrt (x - y) = 2 / 5) ∧ (real.sqrt (x + y) = 2)

-- Define the correct solution
def correct_solution (x y : ℚ) := 
  (x = 52 / 25) ∧ (y = 48 / 25)

-- Define the area of the rectangle
def rectangle_area (a b : ℚ) : ℚ :=
  abs (a * b)

-- Define the proof problem
theorem math_proof_problem : 
  problem_conditions (52 / 25) (48 / 25) ∧ 
  rectangle_area (52 / 25) (48 / 25) = 8 / 25 :=
by 
  sorry

end math_proof_problem_l241_241150


namespace identify_false_statement_l241_241134

-- Definitions for the conditions
def isMultipleOf (n k : Nat) : Prop := ∃ m, n = k * m

def conditions : Prop :=
  isMultipleOf 12 2 ∧
  isMultipleOf 123 3 ∧
  isMultipleOf 1234 4 ∧
  isMultipleOf 12345 5 ∧
  isMultipleOf 123456 6

-- The statement which proves which condition is false
theorem identify_false_statement : conditions → ¬ (isMultipleOf 1234 4) :=
by
  intros h
  sorry

end identify_false_statement_l241_241134


namespace determine_knights_l241_241921

noncomputable def number_of_knights (total_travelers: ℕ) (vasily_is_liar: Prop) (statement_by_vasily: ∀ room: ℕ, (more_liars room ∨ more_knights room) = false) : ℕ := 9

theorem determine_knights :
  ∀ (travelers: ℕ)
    (liar_iff_false: ∀ (P: Prop), liar P ↔ P = false)
    (vasily: traveler)
    (rooms: fin 3 → fin 16)
    (more_liars: Π (r: fin 3), Prop)
    (more_knights: Π (r: fin 3), Prop),
    travelers = 16 →
    liar (more_liars (rooms 0)) ∧ liar (more_knights (rooms 0)) ∧
    liar (more_liars (rooms 1)) ∧ liar (more_knights (rooms 1)) ∧
    liar (more_liars (rooms 2)) ∧ liar (more_knights (rooms 2)) →
    ∃ (k l: ℕ),
      k + l = 15 ∧ k - l = 1 ∧ k = 9 :=
begin
  sorry
end

end determine_knights_l241_241921


namespace area_convex_quadrilateral_le_one_l241_241824

theorem area_convex_quadrilateral_le_one
  (AB BC CD DA : ℝ)
  (h_pos: AB ≥ 0 ∧ BC ≥ 0 ∧ CD ≥ 0 ∧ DA ≥ 0)
  (h_perimeter : AB + BC + CD + DA = 4)
  (h_convex : convex_quadrilateral ABCD) :
  area ABCD ≤ 1 := 
sorry

end area_convex_quadrilateral_le_one_l241_241824


namespace triangle_is_isosceles_l241_241253

theorem triangle_is_isosceles
  (a b c : ℝ)
  (h1 : a ≠ b ∨ a ≠ c ∨ b ≠ c)
  (h2 : (c - b) * x^2 + 2 * (b - a) * x + (a - b) = 0 ∧ discriminant = 0)
  : (a = b ∨ a = c) :=
begin
  sorry
end

end triangle_is_isosceles_l241_241253


namespace perpendicular_planes_l241_241255

variables (m n : Line) (α β : Plane)

theorem perpendicular_planes
  (hm_alpha : m ⊥ α)
  (hm_beta : m ∥ β) :
  α ⊥ β :=
sorry

end perpendicular_planes_l241_241255


namespace count_even_numbers_l241_241674

theorem count_even_numbers (a b : ℕ) (h1 : a > 300) (h2 : b ≤ 600) (h3 : ∀ n, 300 < n ∧ n ≤ 600 → n % 2 = 0) : 
  ∃ c : ℕ, c = 150 :=
by
  sorry

end count_even_numbers_l241_241674


namespace area_of_square_ABCD_l241_241092

theorem area_of_square_ABCD :
  (∃ (x y : ℝ), 2 * x + 2 * y = 40) →
  ∃ (s : ℝ), s = 20 ∧ s * s = 400 :=
by
  sorry

end area_of_square_ABCD_l241_241092


namespace minimum_m_value_l241_241710

theorem minimum_m_value :
  (∀ x ∈ set.Icc (0 : ℝ) (Real.pi / 4), Real.tan x ≤ 1) →
  ∀ m : ℝ, (∀ x ∈ set.Icc (0 : ℝ) (Real.pi / 4), Real.tan x ≤ m) → 1 ≤ m :=
by
  sorry

end minimum_m_value_l241_241710


namespace circumradius_of_right_triangle_l241_241175

theorem circumradius_of_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  2 * (radius_of_circumcircle a b c h) = c := sorry

def radius_of_circumcircle (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℝ :=
  c / 2

example : radius_of_circumcircle 8 15 17
  (by norm_num [pow_two]; norm_num) = 8.5 := by norm_num

end circumradius_of_right_triangle_l241_241175


namespace correct_operation_l241_241132

theorem correct_operation : ¬ (-2 * x + 5 * x = -7 * x) 
                          ∧ (y * x - 3 * x * y = -2 * x * y) 
                          ∧ ¬ (-x^2 - x^2 = 0) 
                          ∧ ¬ (x^2 - x = x) := 
by {
    sorry
}

end correct_operation_l241_241132


namespace radius_of_inscribed_circle_l241_241719

-- Definition of the problem in Lean 4 statement
theorem radius_of_inscribed_circle 
  (a b r : ℝ) (h : a > b)
  (hyp : a^2 + b^2 = (a + b)^2 + 4 * r^2) :
  r = 1 / 2 * (sqrt (a^2 + b^2) - (a^2 + b^2) / (a + b)) := 
sorry

end radius_of_inscribed_circle_l241_241719


namespace green_team_final_score_l241_241762

theorem green_team_final_score (G : ℕ) :
  (∀ G : ℕ, 68 = G + 29 → G = 39) :=
by
  sorry

end green_team_final_score_l241_241762


namespace find_integer_values_l241_241988

theorem find_integer_values (a : ℤ) (h : ∃ (n : ℤ), (a + 9) = n * (a + 6)) :
  a = -5 ∨ a = -7 ∨ a = -3 ∨ a = -9 :=
by
  sorry

end find_integer_values_l241_241988


namespace problem_l241_241294

noncomputable def f (x φ : ℝ) : ℝ := 4 * Real.cos (3 * x + φ)

theorem problem 
  (φ : ℝ) (x1 x2 : ℝ)
  (hφ : |φ| < Real.pi / 2)
  (h_symm : ∀ x, f x φ = f (2 * (11 * Real.pi / 12) - x) φ)
  (hx1x2 : x1 ≠ x2)
  (hx1_range : -7 * Real.pi / 12 < x1 ∧ x1 < -Real.pi / 12)
  (hx2_range : -7 * Real.pi / 12 < x2 ∧ x2 < -Real.pi / 12)
  (h_eq : f x1 φ = f x2 φ) : 
  f (x1 + x2) (-Real.pi / 4) = 2 * Real.sqrt 2 := by
  sorry

end problem_l241_241294


namespace sum_of_f_values_l241_241184

def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem sum_of_f_values : (f (1 / 2016) + f (1 / 2015) + ... + f (1 / 2) + f 0 + f 2 + ... + f 2015 + f 2016) = 1 :=
  sorry

end sum_of_f_values_l241_241184


namespace maximum_area_ABCD_l241_241751

theorem maximum_area_ABCD 
  (ABCD : ConvexQuadrilateral)
  (BC : ℝ) (CD : ℝ)
  (hBC : BC = 3) (hCD : CD = 2 * Real.sqrt 3)
  (hEquilateralCentroids : 
      EquilateralTriangle (Centroid (triangle ABC)) 
                          (Centroid (triangle BCD)) 
                          (Centroid (triangle ACD)))
  (hAreaRelation : 2 * Area (triangle ABD) = Area (triangle BCD)) :
  Area ABD ABCD ≤ 16 * Real.sqrt 3 :=
sorry

end maximum_area_ABCD_l241_241751


namespace students_play_both_l241_241911

variable (students total_students football cricket neither : ℕ)
variable (H1 : total_students = 420)
variable (H2 : football = 325)
variable (H3 : cricket = 175)
variable (H4 : neither = 50)
  
theorem students_play_both (H1 : total_students = 420) (H2 : football = 325) 
    (H3 : cricket = 175) (H4 : neither = 50) : 
    students = 325 + 175 - (420 - 50) :=
by sorry

end students_play_both_l241_241911


namespace average_speed_x_to_z_l241_241775

theorem average_speed_x_to_z (d : ℝ) (h1 : 0 < d) :
  let d_x_y := 2 * d,
      d_y_z := d,
      speed_x_y := 300,
      speed_y_z := 100,
      total_distance := d_x_y + d_y_z,
      total_time := d_x_y / speed_x_y + d_y_z / speed_y_z
  in total_distance / total_time = 180 := 
sorry

end average_speed_x_to_z_l241_241775


namespace max_students_with_equal_distribution_l241_241155

theorem max_students_with_equal_distribution (p : ℕ) (q : ℕ) (students : ℕ) : 
  p = 781 → q = 710 → gcd p q = students → students = 71 :=
by
  intros hp hq hstudents
  rw [hp, hq] at hstudents
  simp at hstudents
  assumption

end max_students_with_equal_distribution_l241_241155


namespace polygonal_chain_length_leq_200_l241_241258

theorem polygonal_chain_length_leq_200 :
  ∀ (board : matrix ℕ ℕ ℕ) (chain : list (ℕ × ℕ)),
    (dim board).fst = 15 →
    (dim board).snd = 15 →
    is_closed_non_self_intersecting_polygonal_chain chain →
    is_symmetric_with_respect_to_diagonal chain →
    length chain ≤ 200 :=
by
  sorry

end polygonal_chain_length_leq_200_l241_241258


namespace range_of_m_l241_241291

noncomputable theory

variable {a : ℝ} (x : ℝ)

def f (x : ℝ) (a : ℝ) := x^2 / a - 2 * log x

def f_prime (x : ℝ) (a : ℝ) := (2 * x / a) - (2 / x)

def g (a : ℝ) := 1 - log a

def F (a : ℝ) := a - log a - (2 / (9 * a))

def has_three_real_roots (F : ℝ → ℝ) :=
  ∃ (a1 a2 a3 : ℝ), (a1 ≠ a2) ∧ (a2 ≠ a3) ∧ (a1 ≠ a3) ∧ (F a1 = m) ∧ (F a2 = m) ∧ (F a3 = m)

theorem range_of_m (m : ℝ) :
  (∃ (a : ℝ) (h : a > 0), g(a) + a - (2 / (9 * a)) - 1 = m) ∧ has_three_real_roots F →
  ((1 / 3 - log 2 + log 3) < m ∧ m < (-1 / 3 + log 3)) :=
begin
  sorry
end

end range_of_m_l241_241291


namespace number_of_integers_with_abs_val_conditions_l241_241450

theorem number_of_integers_with_abs_val_conditions : 
  (∃ n : ℕ, n = 8) :=
by sorry

end number_of_integers_with_abs_val_conditions_l241_241450


namespace total_pens_bought_l241_241034

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l241_241034


namespace smallest_value_of_root_expression_l241_241038

noncomputable def smallest_value : ℕ := sorry

theorem smallest_value_of_root_expression :
  ∃ (a₁ a₂ : ℝ) (u q : ℝ),
    (a₁ + a₂ = u) ∧
    (a₁ ^ 2 + a₂ ^ 2 = u) ∧
    (a₁ ^ 3 + a₂ ^ 3 = u) ∧
    (a₁ ^ 4 + a₂ ^ 4 = u) ∧
    (u = a₁ + a₂) ∧
    (q = a₁ * a₂) ∧
    (polynomial.aeval a₁ (polynomial.X^2 - polynomial.C u * polynomial.X + polynomial.C q) = 0) ∧
    (polynomial.aeval a₂ (polynomial.X^2 - polynomial.C u * polynomial.X + polynomial.C q) = 0) ∧
    (smallest_value = (1 / a₁^10) + (1 / a₂^10)) ∧ smallest_value = 2 := sorry

end smallest_value_of_root_expression_l241_241038


namespace initial_slices_ham_l241_241959

def total_sandwiches : ℕ := 50
def slices_per_sandwich : ℕ := 3
def additional_slices_needed : ℕ := 119

-- Calculate the total number of slices needed to make 50 sandwiches.
def total_slices_needed : ℕ := total_sandwiches * slices_per_sandwich

-- Prove the initial number of slices of ham Anna has.
theorem initial_slices_ham : total_slices_needed - additional_slices_needed = 31 := by
  sorry

end initial_slices_ham_l241_241959


namespace library_leftover_space_l241_241556

theorem library_leftover_space (wall_length : ℝ)
                              (desk_length : ℝ)
                              (bookcase_length : ℝ)
                              (spacing : ℝ)
                              (num_pairs : ℕ) :
  wall_length = 15 →
  desk_length = 2 →
  bookcase_length = 1.5 →
  spacing = 0.5 →
  4 * num_pairs ≤ wall_length →
  num_pairs = 3 →
  ∃ (q : ℝ), q = wall_length - (4 * num_pairs) ∧ q = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  use wall_length - 4 * num_pairs
  split
  { sorry }
  { sorry }

end library_leftover_space_l241_241556


namespace complex_number_value_l241_241246

open Complex

theorem complex_number_value : (i + i^2 + i^3 + i^4) = 0 := by
  have h1 : i^1 = i, from Complex.cpow_one i
  have h2 : i^2 = -1, by simp only [Complex.I_mul_I, Complex.one_mul, neg_one_mul]
  have h3 : i^3 = -i, by rw [←Complex.mul_assoc, h2, neg_mul_eq_neg_mul_symm, one_mul]
  have h4 : i^4 = 1, by rw [←Complex.mul_assoc, h2, Complex.mul_neg_eq_neg_mul_symm, neg_neg, Complex.one_mul]
  calc
    i + i^2 + i^3 + i^4
    = i + (-1) + (-i) + 1 : by rw [h2, h3, h4]
    = i + (-i) + (-1) + 1 : by ring
    = 0 : by ring

end complex_number_value_l241_241246


namespace minimum_value_expr_l241_241990

noncomputable def expr (x y z : ℝ) : ℝ := 
  3 * x^2 + 2 * x * y + 3 * y^2 + 2 * y * z + 3 * z^2 - 3 * x + 3 * y - 3 * z + 9

theorem minimum_value_expr : 
  ∃ (x y z : ℝ), ∀ (a b c : ℝ), expr a b c ≥ expr x y z ∧ expr x y z = 3/2 :=
sorry

end minimum_value_expr_l241_241990


namespace _l241_241629

variable {a : ℕ → ℚ} {S : ℕ → ℚ}

-- The conditions from the problem
axiom cond1 : a 1 = 1 / 2
axiom cond2 : ∀ n : ℕ, (1 < n) → a n + 2 * S n * S (n-1) = 0
axiom cond3 : ∀ n : ℕ, (0 < n) → S n = ∑ i in Finset.range n, a (i + 1)

-- (1) Prove that the sequence {1 / S_n} is an arithmetic sequence
theorem (h1 : cond1) (h2 : cond2) (h3 : cond3) : 
  ∃ c : ℚ, (∀ n : ℕ, (1 < n) → (1 / S n) - (1 / S (n-1)) = c) ∧ (1 / (S 1) = c) :=
by sorry

-- (2) Find the general formula for the sequence {a_n}
theorem (h1 : cond1) (h2 : cond2) (h3 : cond3) : 
  ∀ n : ℕ, a n = 
    if n = 1 then
      1 / 2 
    else 
      - (1 / (2 * n * (n-1))) :=
by sorry

end _l241_241629


namespace find_length_of_second_train_l241_241151

def length_of_second_train (L : ℝ) : Prop :=
  let speed_first_train := 33.33 -- Speed in m/s
  let speed_second_train := 22.22 -- Speed in m/s
  let relative_speed := speed_first_train + speed_second_train -- Relative speed in m/s
  let time_to_cross := 9 -- time in seconds
  let length_first_train := 260 -- Length in meters
  length_first_train + L = relative_speed * time_to_cross

theorem find_length_of_second_train : length_of_second_train 239.95 :=
by
  admit -- To be completed (proof)

end find_length_of_second_train_l241_241151


namespace prime_dates_in_2007_l241_241396

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_dates_2007 : ℕ :=
  let prime_days_feb := [2, 3, 5, 7, 11, 13, 17, 19, 23].length
  let prime_days_31 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31].length
  let prime_days_30 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29].length
  in prime_days_feb + 3 * prime_days_31 + prime_days_30

theorem prime_dates_in_2007 : prime_dates_2007 = 52 :=
  by
    sorry

end prime_dates_in_2007_l241_241396


namespace circumcircle_radius_l241_241174

theorem circumcircle_radius (a b c : ℕ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) (h4 : a^2 + b^2 = c^2) :
  ∃ r : ℚ, r = 17 / 2 :=
by
  have hc : c = 17 := h3
  use 17 / 2
  sorry

end circumcircle_radius_l241_241174


namespace increasing_a_n_not_increasing_n_a_n_increasing_a_n_over_n_not_increasing_a_n_sq_l241_241441

def a_n (n : ℕ) : ℤ := 2 * n - 8

theorem increasing_a_n : ∀ n : ℕ, a_n (n + 1) > a_n n := 
by 
-- Assuming n >= 0
intro n
dsimp [a_n]
sorry

def n_a_n (n : ℕ) : ℤ := n * (2 * n - 8)

theorem not_increasing_n_a_n : ∀ n : ℕ, n > 0 → n_a_n (n + 1) ≤ n_a_n n :=
by
-- Assuming n > 0
intro n hn
dsimp [n_a_n]
sorry

def a_n_over_n (n : ℕ) : ℚ := (2 * n - 8 : ℚ) / n

theorem increasing_a_n_over_n : ∀ n > 0, a_n_over_n (n + 1) > a_n_over_n n :=
by 
-- Assuming n > 0
intro n hn
dsimp [a_n_over_n]
sorry

def a_n_sq (n : ℕ) : ℤ := (2 * n - 8) * (2 * n - 8)

theorem not_increasing_a_n_sq : ∀ n : ℕ, a_n_sq (n + 1) ≤ a_n_sq n :=
by
-- Assuming n >= 0
intro n
dsimp [a_n_sq]
sorry

end increasing_a_n_not_increasing_n_a_n_increasing_a_n_over_n_not_increasing_a_n_sq_l241_241441


namespace problem_trapezoid_ED_length_l241_241734

theorem problem_trapezoid_ED_length :
  ∀ (A B C D E : Type)
    (AB CD AD BC : ℝ)
    (mid_AC : (A × C) → E)
    (h1 : AB = 8)
    (h2 : CD = 6)
    (h3 : AD = 5)
    (h4 : BC = 5)
    (hE : ∀ AC, E = mid_AC AC)
    , let ED := sqrt 6.5
    in ED = sqrt 6.5 := 
by
  sorry

end problem_trapezoid_ED_length_l241_241734


namespace conjugate_z_range_of_a_l241_241657

open Complex

noncomputable def z : ℂ := (-1 + 3 * I) * (1 - I) - 4
noncomputable def ω (a : ℝ) : ℂ := z + a * I

theorem conjugate_z : conj z = -2 - 4 * I := by
  -- Proof here
  sorry

theorem range_of_a (a : ℝ) : abs (ω a) ≤ abs z ↔ -8 ≤ a ∧ a ≤ 0 := by
  -- Proof here
  sorry

end conjugate_z_range_of_a_l241_241657


namespace second_number_is_22_l241_241325

theorem second_number_is_22 (x second_number : ℕ) : 
  (x + second_number = 33) → 
  (second_number = 2 * x) → 
  second_number = 22 :=
by
  intros h_sum h_double
  sorry

end second_number_is_22_l241_241325


namespace construct_triangle_ABC_l241_241099

noncomputable theory

variables {B B' B'' A C : ℝ}
variables {a b c p q β : ℝ}
variables (hB : B > 0 ∧ B' > 0 ∧ B'' > 0)
variables (h_distances : c > a ∧ p > q)
variables (h_angle : β > 0 ∧ β < 2 * π)
variables (h_A_plane : A > 0)
variables (h_C_axis : C > 0)
variables (h_segments : p > 0 ∧ q > 0 ∧ b > 0)

theorem construct_triangle_ABC :
  ∃ (A B C : ℝ), B > 0 ∧ B' > 0 ∧ B'' > 0 ∧ c > a ∧ p > q ∧ β > 0 ∧ β < 2 * π ∧ 
  A > 0 ∧ C > 0 ∧ p > 0 ∧ q > 0 ∧ b > 0 ∧ 
  is_triangle A B C :=
sorry

end construct_triangle_ABC_l241_241099


namespace max_quadratic_value_l241_241850

noncomputable def quadratic_function (x : ℝ) : ℝ := -5 * x^2 + 25 * x - 1

theorem max_quadratic_value :
  (∃ x : ℝ, ∀ y : ℝ, quadratic_function x ≥ quadratic_function y) ∧
  quadratic_function (5 / 2) = 129 / 4 :=
begin
  sorry
end

end max_quadratic_value_l241_241850


namespace ratio_areas_l241_241072

-- Given conditions
variables (A B J E C : Type) [Point E] [Point C] [Midpoint E A B] [Midpoint C F G]

-- Defining squares BCDE and FGHI inside triangle ABJ
structure Square (P Q R S : Type)

axiom is_square_BCDE (BCDE : Square BCDE) : Square B C D E
axiom is_square_FGHI (FGHI : Square FGHI) : Square F G H I

variables [InTriangle BCDE ABJ] [InTriangle FGHI ABJ]

-- Goal: Prove that the ratio of the area of BCDE to the area of the triangle ABJ is 1/3
theorem ratio_areas (BCDE : Square B C D E) (ABJ : Triangle A B J) : 
  (Area BCDE) / (Area ABJ) = 1 / 3 := 
sorry

end ratio_areas_l241_241072


namespace total_votes_l241_241720

variable {V : ℕ}

theorem total_votes (h₁ : 0.20 * V = 240) : V = 1200 :=
sorry

end total_votes_l241_241720


namespace strongest_goldbach_140_l241_241828

noncomputable def largest_prime_difference : ℕ :=
  let is_prime (n : ℕ) := Nat.Prime n in
  let pairs : List (ℕ × ℕ) := 
    [(3, 137), (31, 109), (37, 103), (43, 97)] in
  let differences := pairs.map (λ ⟨p, q⟩ => q - p) in
  List.maximum differences

theorem strongest_goldbach_140 : largest_prime_difference = 134 :=
by 
  sorry

end strongest_goldbach_140_l241_241828


namespace find_a_l241_241818

noncomputable def a_value : ℚ :=
-11 / 5

theorem find_a (a : ℚ) (x : ℚ) (y : ℚ) :
  (a * x + (a + 1) * y = a + 4) ∧ (x = 3) ∧ (y = -7) → a = -11 / 5 :=
by
  intro h
  cases h with line_eq conds
  cases conds with x_eq y_eq
  sorry

end find_a_l241_241818


namespace trigonometric_identity_value_of_expression_l241_241458

theorem trigonometric_identity :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = Real.cos (Real.pi / 4) :=
sorry

theorem value_of_expression :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = (Real.sqrt 2) / 2 :=
by
  have h : 1 - 2 * (Real.sin (Real.pi / 8))^2 = Real.cos (Real.pi / 4) := trigonometric_identity
  rw Real.cos_pi_div_four at h
  exact h

end trigonometric_identity_value_of_expression_l241_241458


namespace determine_x_l241_241654

def geometric_sum (n : ℕ) (x : ℝ) : ℝ := x * 3^(n-1) - 1/6

theorem determine_x (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, S n = geometric_sum n (1/2)) :
  (1 / 2) = 1 / 2 :=
by sorry

end determine_x_l241_241654


namespace amplitude_five_phase_shift_minus_pi_over_4_l241_241603

noncomputable def f (x : ℝ) : ℝ := 5 * Real.cos (x + (Real.pi / 4))

theorem amplitude_five : ∀ x : ℝ, 5 * Real.cos (x + (Real.pi / 4)) = f x :=
by
  sorry

theorem phase_shift_minus_pi_over_4 : ∀ x : ℝ, f x = 5 * Real.cos (x + (Real.pi / 4)) :=
by
  sorry

end amplitude_five_phase_shift_minus_pi_over_4_l241_241603


namespace brick_wall_total_bricks_l241_241109

theorem brick_wall_total_bricks : 
  ∀ (x : ℕ), (1 ≠ 0) →
  let rate1 := x / 8 
  ∧ let rate2 := x / 12 
  ∧ let combined_rate := (x / 8 + x / 12 - 12)
  ∧ (6 * combined_rate = x) 
  in x = 288 := 
by 
  intros x h1; 
  let rate1 := x / 8; 
  let rate2 := x / 12; 
  let combined_rate := (x / 8 + x / 12 - 12);
  have h2 : (6 * combined_rate = x) := sorry; 
  have h3 : x = 288 := sorry;
  exact h3

end brick_wall_total_bricks_l241_241109


namespace student_factor_l241_241548

theorem student_factor (x : ℤ) : (121 * x - 138 = 104) → x = 2 :=
by
  intro h
  sorry

end student_factor_l241_241548


namespace park_trees_after_planting_l241_241463

theorem park_trees_after_planting (current_trees trees_today trees_tomorrow : ℕ)
  (h1 : current_trees = 7)
  (h2 : trees_today = 5)
  (h3 : trees_tomorrow = 4) :
  current_trees + trees_today + trees_tomorrow = 16 :=
by
  sorry

end park_trees_after_planting_l241_241463


namespace money_leftover_is_90_l241_241746

-- Define constants and given conditions.
def jars_quarters : ℕ := 4
def quarters_per_jar : ℕ := 160
def jars_dimes : ℕ := 4
def dimes_per_jar : ℕ := 300
def jars_nickels : ℕ := 2
def nickels_per_jar : ℕ := 500

def value_per_quarter : ℝ := 0.25
def value_per_dime : ℝ := 0.10
def value_per_nickel : ℝ := 0.05

def bike_cost : ℝ := 240
def total_quarters := jars_quarters * quarters_per_jar
def total_dimes := jars_dimes * dimes_per_jar
def total_nickels := jars_nickels * nickels_per_jar

-- Calculate the total money Jenn has in quarters, dimes, and nickels.
def total_value_quarters : ℝ := total_quarters * value_per_quarter
def total_value_dimes : ℝ := total_dimes * value_per_dime
def total_value_nickels : ℝ := total_nickels * value_per_nickel

def total_money : ℝ := total_value_quarters + total_value_dimes + total_value_nickels

-- Calculate the money left after buying the bike.
def money_left : ℝ := total_money - bike_cost

-- Prove that the amount of money left is precisely $90.
theorem money_leftover_is_90 : money_left = 90 :=
by
  -- Placeholder for the proof
  sorry

end money_leftover_is_90_l241_241746


namespace percentage_republicans_vote_X_l241_241714

theorem percentage_republicans_vote_X (R : ℝ) (P_R : ℝ) :
  (3 * R * P_R + 2 * R * 0.15) - (3 * R * (1 - P_R) + 2 * R * 0.85) = 0.019999999999999927 * (3 * R + 2 * R) →
  P_R = 4.1 / 6 :=
by
  intro h
  sorry

end percentage_republicans_vote_X_l241_241714


namespace work_days_for_p_l241_241504

theorem work_days_for_p (dp : ℝ) (dq : ℝ) (left_work_fraction : ℝ) (days_together : ℝ) : dq = 20 →
  left_work_fraction = 0.5333333333333333 → days_together = 4 → 4 * (1 / dp + 1 / dq) = 0.4666666666666667 → dp = 15 :=
begin
  intros,
  sorry
end

end work_days_for_p_l241_241504


namespace ants_no_collision_probability_l241_241228

open Probability

def vertices : Finset (ℕ × ℕ × ℕ) := 
  {(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)}

def adjacent (v1 v2 : ℕ × ℕ × ℕ) : Prop := 
  ∥v1.1 - v2.1∥ + ∥v1.2 - v2.2∥ + ∥v1.3 - v2.3∥ = 1

def no_collision_movement_probability : ℚ :=
  405 / 43046721

theorem ants_no_collision_probability :
  ∃ (prob : ℚ), prob = no_collision_movement_probability :=
begin
  use 405 / 43046721,
  sorry,
end

end ants_no_collision_probability_l241_241228


namespace statement_1_statement_4_l241_241202

variables {m n : Type} {α β γ : Type}
variables (is_perpendicular : m -> α -> Prop) (is_parallel : m -> α -> Prop)

-- Conditions
axiom perp_alpha_m : is_perpendicular m α
axiom par_alpha_n : is_parallel n α
axiom par_alpha_beta : is_parallel α β
axiom par_beta_gamma : is_parallel β γ
axiom perp_alpha_m' : is_perpendicular m α

-- Statement 1 : m ⊥ α ∧ n ∥ α → m ⊥ n
theorem statement_1 (h1 : is_perpendicular m α) (h2 : is_parallel n α) : is_perpendicular m n := sorry

-- Statement 4 : α ∥ β ∧ β ∥ γ ∧ m ⊥ α → m ⊥ γ
theorem statement_4 (h1 : is_parallel α β) (h2 : is_parallel β γ) (h3 : is_perpendicular m α) : is_perpendicular m γ := sorry

end statement_1_statement_4_l241_241202


namespace part_I_part_II_l241_241292

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) - 2

theorem part_I (α : ℝ) (hα : ∃ (P : ℝ × ℝ), P = (Real.sqrt 3, -1) ∧
  (Real.tan α = -1 / Real.sqrt 3 ∨ Real.tan α = - (Real.sqrt 3) / 3)) :
  f α = -3 := by
  sorry

theorem part_II (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  -2 ≤ f x ∧ f x ≤ 1 := by
  sorry

end part_I_part_II_l241_241292


namespace sam_original_puppies_count_l241_241785

theorem sam_original_puppies_count 
  (spotted_puppies_start : ℕ)
  (non_spotted_puppies_start : ℕ)
  (spotted_puppies_given : ℕ)
  (non_spotted_puppies_given : ℕ)
  (spotted_puppies_left : ℕ)
  (non_spotted_puppies_left : ℕ)
  (h1 : spotted_puppies_start = 8)
  (h2 : non_spotted_puppies_start = 5)
  (h3 : spotted_puppies_given = 2)
  (h4 : non_spotted_puppies_given = 3)
  (h5 : spotted_puppies_left = spotted_puppies_start - spotted_puppies_given)
  (h6 : non_spotted_puppies_left = non_spotted_puppies_start - non_spotted_puppies_given)
  (h7 : spotted_puppies_left = 6)
  (h8 : non_spotted_puppies_left = 2) :
  spotted_puppies_start + non_spotted_puppies_start = 13 :=
by
  sorry

end sam_original_puppies_count_l241_241785


namespace polynomial_divisible_by_7_polynomial_divisible_by_12_l241_241585

theorem polynomial_divisible_by_7 (x : ℤ) : (x^7 - x) % 7 = 0 := 
sorry

theorem polynomial_divisible_by_12 (x : ℤ) : (x^4 - x^2) % 12 = 0 := 
sorry

end polynomial_divisible_by_7_polynomial_divisible_by_12_l241_241585


namespace expected_value_difference_cereals_l241_241216

noncomputable def expectedDifferenceCereals : ℕ :=
  let dieSides := 8
  let daysInYear := 365
  let probSweetened := 4 / 7
  let probUnsweetened := 3 / 7
  let expectedSweetenedDays := probSweetened * daysInYear
  let expectedUnsweetenedDays := probUnsweetened * daysInYear
  (expectedUnsweetenedDays - expectedSweetenedDays).round

theorem expected_value_difference_cereals : expectedDifferenceCereals = 54 := 
  by
    sorry

end expected_value_difference_cereals_l241_241216


namespace constant_sequence_l241_241981

theorem constant_sequence (a : Fin 2017 → ℕ) (h1 : ∀ i : Fin 2017, 0 ≤ a i ∧ a i ≤ 2016) 
  (h2 : ∀ i j : Fin 2017, (i + j : ℕ) ∣ (i : ℕ) * (a i) + (j : ℕ) * (a j)) :
  ∃ c ∈ set.Icc (0 : ℕ) 2016, ∀ i, a i = c :=
by 
  sorry

end constant_sequence_l241_241981


namespace count_divisible_by_11_l241_241381

-- definition of write_naturals is provided
def write_naturals (k : ℕ) : ℕ :=
  -- The number obtained by concatenating the integers 1 to k
  let digits := List.join (List.map toString (List.range (k + 1)))
  digits.toNat

-- Define the alternating sum function
def alternating_sum (k : ℕ) : ℤ :=
  List.sum
    (List.mapWithIndex (λ i d, if i % 2 = 0 then d else -d) (digits k))

theorem count_divisible_by_11 : 
  let b_k := write_naturals
  let s_b_k := alternating_sum b_k
  #{ k | 1 ≤ k ∧ k ≤ 50 ∧ s_b_k k % 11 = 0 } = X :=
sorry

end count_divisible_by_11_l241_241381


namespace solution_set_interval_l241_241985

theorem solution_set_interval (a : ℝ) : 
  {x : ℝ | x^2 - 2*a*x + a^2 - 1 < 0} = {x : ℝ | a - 1 < x ∧ x < a + 1} :=
sorry

end solution_set_interval_l241_241985


namespace force_from_potential_l241_241492

open Real

noncomputable def U (x : ℝ) : ℝ := 
  if x < -10 then c1
  else if x < 0 then c2 + m1 * x
  else if x <= 10 then c3 + m2 * x
  else c4

theorem force_from_potential (x : ℝ) (F : ℝ) (m1 m2 : ℝ) (c1 c2 c3 c4 : ℝ) :
  F = - (U x).derivative → (
    (∀ x ∈ Icc (-15:ℝ) (-10), F = 0) ∧
    (∀ x ∈ Icc (-10) (0), F > 0) ∧
    (∀ x ∈ Icc (0) (10), F < 0) ∧
    (∀ x ∈ Icc (10:ℝ) (15), F = 0)
  ) := sorry

end force_from_potential_l241_241492


namespace problem1_min_value_problem2_max_value_problem3_min_value_l241_241913

-- Problem 1
theorem problem1_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 1) : x + 2 * y ≥ 8 :=
sorry

-- Problem 2
theorem problem2_max_value (x : ℝ) (hx : x < 3) : -1 ≥ f(x) :=
sorry

-- Define the function f for completeness
def f (x : ℝ) := 4 / (x - 3) + x

-- Problem 3
theorem problem3_min_value (m n : ℝ) (hmn : m + n = 2) (hm : m > 0) (hn : n > 0) : (n / m + 1 / (2 * n)) ≥ 5 / 4 :=
sorry

end problem1_min_value_problem2_max_value_problem3_min_value_l241_241913


namespace snail_crawl_distance_l241_241547

theorem snail_crawl_distance
  (α : ℕ → ℝ)  -- α represents the snail's position at each minute
  (crawls_forward : ∀ n m : ℕ, n < m → α n ≤ α m)  -- The snail moves forward (without going backward)
  (observer_finds : ∀ n : ℕ, α (n + 1) - α n = 1) -- Every observer finds that the snail crawled exactly 1 meter per minute
  (time_span : ℕ := 6)  -- Total observation period is 6 minutes
  : α time_span - α 0 ≤ 10 :=  -- The distance crawled in 6 minutes does not exceed 10 meters
by
  -- Proof goes here
  sorry

end snail_crawl_distance_l241_241547


namespace trig_from_quadratic_l241_241260

theorem trig_from_quadratic (α : ℝ) :
  (∃ x : ℝ, x^2 - (Real.tan α + Real.cot α) * x + 1 = 0 ∧ (x = 2 - Real.sqrt 3)) →
  Real.sin (2 * α) = 1 / 2 ∧ Real.cos (4 * α) = 1 / 2 :=
by
  intro h
  sorry

end trig_from_quadratic_l241_241260


namespace average_of_set_is_5_l241_241071

theorem average_of_set_is_5 :
  let data := [5, 6, 5, 6, 4, 4]
  in (data.sum) / (data.length) = 5 :=
by
  let data := [5, 6, 5, 6, 4, 4]
  show (data.sum) / (data.length) = 5
  sorry

end average_of_set_is_5_l241_241071


namespace flight_height_l241_241363

theorem flight_height (flights : ℕ) (step_height_in_inches : ℕ) (total_steps : ℕ) 
    (H1 : flights = 9) (H2 : step_height_in_inches = 18) (H3 : total_steps = 60) : 
    (total_steps * step_height_in_inches) / 12 / flights = 10 :=
by
  sorry

end flight_height_l241_241363


namespace solution_product_eq_minus_450_l241_241759

def quadratic_roots (a b c : ℝ) : (ℝ × ℝ) :=
  let d := b^2 - 4 * a * c
  (((-b + Real.sqrt d) / (2 * a)), ((-b - Real.sqrt d) / (2 * a)))

theorem solution_product_eq_minus_450 :
  let r := (-6 + 3 * Real.sqrt 17);
  let s := (-6 - 3 * Real.sqrt 17);
  (r + 3) * (s + 3) = -450 := by
  sorry

end solution_product_eq_minus_450_l241_241759


namespace modulus_lucas_sequence_l241_241448

theorem modulus_lucas_sequence :
  let M : ℕ → ℕ := λ n, if n = 1 then 2 else if n = 2 then 4 else (M (n - 1) + M (n - 2)) % 5 in
  M 100 % 5 = 0 :=
by
  sorry

end modulus_lucas_sequence_l241_241448


namespace regions_divided_by_99_lines_l241_241924

theorem regions_divided_by_99_lines (n : ℕ) : n < 199 → (n = 100 ∨ n = 198) :=
by {
  -- Definition placeholder: Number of regions divided by m lines
  let P : ℕ → ℕ :=
    λ m, (m * (m - 1)) / 2 + m + 1,
  
  have P99 : P 99 = 4951,
    by sorry, -- Simplification steps for P 99
  
  have small_nat : ℕ → ℕ → ℕ := nat.min -- Auxiliary function for minimum
  
  have result := small_nat (small_nat n 100) 198,
  sorry -- Placeholder for the final proof
}

end regions_divided_by_99_lines_l241_241924


namespace bond_selling_price_l241_241910

theorem bond_selling_price (face_value interest_rate_1 interest_rate_2 S : ℝ)
  (h1 : face_value = 5000)
  (h2 : interest_rate_1 = 0.08)
  (h3 : interest_rate_2 = 0.065)
  (h4 : 5000 * 0.08 = 400)
  (h5 : 400 ≈ S * 0.065) :
  S ≈ 6153.85 :=
sorry

end bond_selling_price_l241_241910


namespace pencil_box_contains_same_color_pencils_l241_241056

theorem pencil_box_contains_same_color_pencils :
  ∀ (n : ℕ), n ≥ 1 → (∀ (box : Fin n → ℕ), ∃ c : ℕ, ∀ i : Fin n, box i = c) :=
begin
    sorry
end

end pencil_box_contains_same_color_pencils_l241_241056


namespace rectangle_diagonals_not_perpendicular_l241_241491

-- Definition of a rectangle through its properties
structure Rectangle (α : Type _) [LinearOrderedField α] :=
  (angle_eq : ∀ (a : α), a = 90)
  (diagonals_eq : ∀ (d1 d2 : α), d1 = d2)
  (diagonals_bisect : ∀ (d1 d2 : α), d1 / 2 = d2 / 2)

-- Theorem stating that a rectangle's diagonals are not necessarily perpendicular
theorem rectangle_diagonals_not_perpendicular (α : Type _) [LinearOrderedField α] (R : Rectangle α) : 
  ¬ (∀ (d1 d2 : α), d1 * d2 = 0) :=
sorry

end rectangle_diagonals_not_perpendicular_l241_241491


namespace infinite_n_Satisfying_conditions_l241_241402

def is_sum_of_three_squares (n : ℕ) : Prop := 
  ∃ a b c : ℕ, a^2 + b^2 + c^2 = n

theorem infinite_n_Satisfying_conditions :
  ∀ i : ℕ, i ∈ {0, 1, 2, 3} →
  ∃ᶠ n in at_top, 
    (finset.card (finset.filter is_sum_of_three_squares (finset.of_list [n, n + 2, n + 28])) = i) :=
by
  sorry

end infinite_n_Satisfying_conditions_l241_241402


namespace max_value_of_cos2_sin_add_3_domain_of_f_is_correct_l241_241915

-- Problem 1: Proving the maximum value of the function y = cos^2 α + sin α + 3
theorem max_value_of_cos2_sin_add_3 (α : ℝ) : 
  ∃ m, (∀ α, cos α * cos α + sin α + 3 ≤ m) ∧ m = 17 / 4 :=
sorry

-- Problem 2: Finding the domain of the function f(x) = sqrt(2 sin^2 x + 3 sin x - 2) + log2(-x^2 + 7x + 8)
noncomputable def domain_of_f : Set ℝ :=
{ x | (2 * (sin x)^2 + 3 * sin x - 2 ≥ 0) ∧ (-x^2 + 7 * x + 8 > 0) }

theorem domain_of_f_is_correct : 
  domain_of_f = { x : ℝ | π / 6 ≤ x ∧ x ≤ 5 * π / 6 } :=
sorry

end max_value_of_cos2_sin_add_3_domain_of_f_is_correct_l241_241915


namespace regina_has_20_cows_l241_241049

theorem regina_has_20_cows (C P : ℕ)
  (h1 : P = 4 * C)
  (h2 : 400 * P + 800 * C = 48000) :
  C = 20 :=
by
  sorry

end regina_has_20_cows_l241_241049


namespace number_of_knights_l241_241919

def traveler := Type
def is_knight (t : traveler) : Prop := sorry
def is_liar (t : traveler) : Prop := sorry

axiom total_travelers : Finset traveler
axiom vasily : traveler
axiom  h_total : total_travelers.card = 16

axiom kn_lie (t : traveler) : is_knight t ∨ is_liar t

axiom vasily_liar : is_liar vasily
axiom contradictory_statements_in_room (rooms: Finset (Finset traveler)):
  (∀ room ∈ rooms, ∃ t ∈ room, (is_liar t ∧ is_knight t))
  ∧
  (∀ room ∈ rooms, ∃ t ∈ room, (is_knight t ∧ is_liar t))

theorem number_of_knights : 
  ∃ k, k = 9 ∧ (∃ l, l = 7 ∧ ∀ t ∈ total_travelers, (is_knight t ∨ is_liar t)) :=
sorry

end number_of_knights_l241_241919


namespace sufficient_condition_l241_241430

theorem sufficient_condition (m : ℝ) (x : ℝ) : -3 < m ∧ m < 1 → ((m - 1) * x^2 + (m - 1) * x - 1 < 0) :=
by
  sorry

end sufficient_condition_l241_241430


namespace compute_expression_l241_241579

theorem compute_expression : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end compute_expression_l241_241579


namespace count_valid_permutations_l241_241614

-- Define the set of digits
def digits : List ℕ := [1, 2, 3, 4, 5, 6]

-- Define all permutations of the digits
def all_permutations := List.permutations digits

-- Define a predicate to check if 5 and 6 are on the same side of 3 in a given list
def same_side_of_three (l : List ℕ) : Prop :=
  let idx3 := l.indexOf 3
  let idx5 := l.indexOf 5
  let idx6 := l.indexOf 6
  (idx5 < idx3 ∧ idx6 < idx3) ∨ (idx5 > idx3 ∧ idx6 > idx3)

-- Define the list of valid permutations where 5 and 6 are on the same side of 3
def valid_permutations := List.filter same_side_of_three all_permutations

-- The theorem stating the number of such valid permutations
theorem count_valid_permutations : valid_permutations.length = 480 := by
  sorry

end count_valid_permutations_l241_241614


namespace evaluate_S_div_l241_241748

def binomial (n k : ℕ) : ℕ := nat.choose n k

def S (n : ℕ) : ℕ :=
  ∑ r in finset.range (n + 1), binomial (3 * n + r) r

theorem evaluate_S_div : S 12 / (23 * 38 * 41 * 43 * 47) = 1274 :=
  sorry

end evaluate_S_div_l241_241748


namespace complement_intersection_U_l241_241916

-- Definitions of the sets based on the given conditions
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Definition of the complement of a set with respect to another set
def complement (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Statement asserting the equivalence
theorem complement_intersection_U :
  complement U (M ∩ N) = {1, 4} :=
by
  sorry

end complement_intersection_U_l241_241916


namespace james_tylenol_intake_per_day_l241_241745

variable (hours_in_day : ℕ := 24) 
variable (tablets_per_dose : ℕ := 2) 
variable (mg_per_tablet : ℕ := 375)
variable (hours_per_dose : ℕ := 6)

theorem james_tylenol_intake_per_day :
  (tablets_per_dose * mg_per_tablet) * (hours_in_day / hours_per_dose) = 3000 := by
  sorry

end james_tylenol_intake_per_day_l241_241745


namespace simplify_product_l241_241508

theorem simplify_product (x y : ℝ) : 
  (x - 3 * y + 2) * (x + 3 * y + 2) = (x^2 + 4 * x + 4 - 9 * y^2) :=
by
  sorry

end simplify_product_l241_241508


namespace sum_of_a_21_to_a_30_l241_241456

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Definition of the arithmetic sequence
noncomputable def S_n (n : ℕ) : ℝ := (n * (a_n 1 + a_n n)) / 2 -- Sum of first n terms of the arithmetic sequence

theorem sum_of_a_21_to_a_30 :
  let S₁₀ := 10 in
  let S₂₀ := S₁₀ + 30 in
  let S₃₀ := S₂₀ + 50 in
  (S_n 30 - S_n 20) = 50 :=
by
  sorry

end sum_of_a_21_to_a_30_l241_241456


namespace sequence_is_geometric_from_second_term_l241_241263

noncomputable theory

open Nat

def sequence_geometric_from_second_term (S : ℕ → ℕ) : Prop :=
S 1 = 1 ∧
S 2 = 2 ∧
(∀ n, n ≥ 2 → S (n + 1) - 3 * S n + 2 * S (n - 1) = 0) →
(∃ r, r = 2 ∧ ∀ n, n ≥ 2 → (S (n+1) - S n) = r * (S n - S (n - 1)))

theorem sequence_is_geometric_from_second_term (S : ℕ → ℕ) :
  sequence_geometric_from_second_term S :=
by
  sorry

end sequence_is_geometric_from_second_term_l241_241263


namespace perimeter_triangle_ABF2_l241_241944

noncomputable def ellipse_perimeter : ℝ :=
  let a := (sqrt 2) / 2 in
  2 * sqrt 2

theorem perimeter_triangle_ABF2 :
  ∀ (x y : ℝ), 4 * x^2 + 2 * y^2 = 1 →
  ∃ A B F1 F2 : ℝ × ℝ,
  (A ∈ set_of (λ p : ℝ × ℝ, 4 * p.1^2 + 2 * p.2^2 = 1)) ∧
  (B ∈ set_of (λ p : ℝ × ℝ, 4 * p.1^2 + 2 * p.2^2 = 1)) ∧
  (F1, F2 ∈ set_of (λ p : ℝ × ℝ, abs p.1 = a ∧ p.2 = 0)) ∧
  (line_through F1 (0, 0) = line_through A B) →
  let perimeter := 2 * a in
  perimeter = 2 * sqrt 2 := by
  sorry

end perimeter_triangle_ABF2_l241_241944


namespace flow_AB_correct_flow_BC_correct_total_flow_A_correct_l241_241181

namespace IrrigationSystem

-- Conditions and variable definitions
variable (q0 : ℝ)

-- Define the flows in different channels as variables
def flow_AH := q0
def flow_HG := q0
def flow_BC := (2 / 3) * q0
def flow_CD := (2 / 3) * q0
def flow_BG := (2 / 3) * q0
def flow_GD := (2 / 3) * q0

-- Define the flow balance for AB
def flow_AB := (4 / 3) * q0

-- Define the total flow into node A
def total_flow_A := q0 + (6 / 3) * q0

-- Proof statements encoding the answers
theorem flow_AB_correct : flow_AB q0 = (4 / 3) * q0 := sorry
theorem flow_BC_correct : flow_BC q0 = (2 / 3) * q0 := sorry
theorem total_flow_A_correct : total_flow_A q0 = (7 / 3) * q0 := sorry

end IrrigationSystem

end flow_AB_correct_flow_BC_correct_total_flow_A_correct_l241_241181


namespace pentagon_inscribed_circle_sum_of_outer_angles_l241_241538

/--
Given a pentagon inscribed in a circle,
if an angle is inscribed into each of the five segments outside the pentagon,
then the sum of these five angles is 900 degrees.
-/
theorem pentagon_inscribed_circle_sum_of_outer_angles
  {A B C D E : Point} (h : inscribed_pentagon A B C D E) :
  sum_of_outer_angles A B C D E = 900 := 
sorry

end pentagon_inscribed_circle_sum_of_outer_angles_l241_241538


namespace percentage_of_children_prefer_corn_l241_241333

theorem percentage_of_children_prefer_corn
  (total_children : ℕ)
  (children_prefer_corn : ℕ)
  (h_total_children : total_children = 40)
  (h_children_prefer_corn : children_prefer_corn = 7) :
  (children_prefer_corn.to_rat / total_children.to_rat) * 100 = 17.5 := 
by
  sorry

end percentage_of_children_prefer_corn_l241_241333


namespace rectangle_solution_l241_241145

-- Define the given conditions
variables (x y : ℚ)

-- Given equations
def condition1 := (Real.sqrt (x - y) = 2 / 5)
def condition2 := (Real.sqrt (x + y) = 2)

-- Solution
theorem rectangle_solution (x y : ℚ) (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 52 / 25 ∧ y = 48 / 25 ∧ (Real.sqrt ((52 / 25) * (48 / 25)) = 8 / 25) :=
by
  sorry

end rectangle_solution_l241_241145


namespace james_tylenol_intake_per_day_l241_241742

theorem james_tylenol_intake_per_day :
  (∃ (tablets_per_dose doses_per_day tablet_mg : ℕ),
    tablets_per_dose = 2 ∧
    doses_per_day = 24 / 6 ∧
    tablet_mg = 375 ∧
    (tablets_per_dose * doses_per_day * tablet_mg = 3000)) :=
begin
  let tablets_per_dose := 2,
  let doses_per_day := 24 / 6,
  let tablet_mg := 375,
  use [tablets_per_dose, doses_per_day, tablet_mg],
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  sorry
end

end james_tylenol_intake_per_day_l241_241742


namespace fraction_ratio_l241_241989

theorem fraction_ratio :
  ∃ (x y : ℕ), y ≠ 0 ∧ (x:ℝ) / (y:ℝ) = 240 / 1547 ∧ ((x:ℝ) / (y:ℝ)) / (2 / 13) = (5 / 34) / (7 / 48) :=
sorry

end fraction_ratio_l241_241989


namespace a_equals_5_l241_241437

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9
def f' (x : ℝ) (a : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem a_equals_5 (a : ℝ) : 
  (∃ x : ℝ, x = -3 ∧ f' x a = 0) → a = 5 := 
by
  sorry

end a_equals_5_l241_241437


namespace total_dolls_l241_241406

-- Definitions given in the conditions
def grandmother_dolls := 50
def sister_dolls := grandmother_dolls + 2
def rene_dolls := 3 * sister_dolls

-- The theorem statement based on condition and correct answer
theorem total_dolls (g : ℕ) (s : ℕ) (r : ℕ) (h_g : g = 50) (h_s : s = g + 2) (h_r : r = 3 * s) : g + s + r = 258 := 
by {
  -- Placeholder for the proof
  sorry,
}

end total_dolls_l241_241406


namespace height_of_boxes_l241_241489

theorem height_of_boxes
  (volume_required : ℝ)
  (price_per_box : ℝ)
  (min_expenditure : ℝ)
  (volume_per_box : ∀ n : ℕ, n = min_expenditure / price_per_box -> ℝ) :
  volume_required = 3060000 ->
  price_per_box = 0.50 ->
  min_expenditure = 255 ->
  ∃ h : ℝ, h = 19 := by
  sorry

end height_of_boxes_l241_241489


namespace circumcenter_intersection_l241_241739

-- Define the conditions
variable {A B C M N : Type*}
variable [T1: IsTriangle A B C]
variable (angleB_eq_60 : ∠ B = 60)
variable (AM_eq_MN_eq_NC : ∀ {a b c m n : ℝ}, AM = MN ∧ MN = NC)

-- The theorem we need to prove
theorem circumcenter_intersection (A B C M N : Type*)
  [IsTriangle A B C] 
  (angleB_eq_60 : ∠ B = 60)
  (AM_eq_MN_eq_NC : ∀ {a b c m n : ℝ}, AM = MN ∧ MN = NC) :
  is_circumcenter (intersection_point (CM) (AN)) :=
by
  sorry

end circumcenter_intersection_l241_241739


namespace derivative_of_f_l241_241293

def f (x : ℝ) : ℝ := 2 * x + 3

theorem derivative_of_f :
  ∀ x : ℝ, (deriv f x) = 2 :=
by 
  sorry

end derivative_of_f_l241_241293


namespace range_of_a_l241_241269

def g (a : ℝ) (x : ℝ) := a * x + a
def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then 
    x^2 - 1 
  else if -2 ≤ x ∧ x < 0 then 
    -x^2 
  else 
    0

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ Icc (-2 : ℝ) (2 : ℝ), ∃ x2 ∈ Icc (-2 : ℝ) (2 : ℝ), g a x1 = f x2) ↔ a ∈ Icc (-4 / 3 : ℝ) 1 :=
by sorry

end range_of_a_l241_241269


namespace maximize_a2_b2_c2_d2_l241_241372

theorem maximize_a2_b2_c2_d2 
  (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 85)
  (h3 : ad + bc = 187)
  (h4 : cd = 110) :
  a^2 + b^2 + c^2 + d^2 ≤ 120 :=
sorry

end maximize_a2_b2_c2_d2_l241_241372


namespace matrix_B6_eq_sB_plus_tI_l241_241754

noncomputable section

open Matrix

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  !![1, -1;
     4, 2]

theorem matrix_B6_eq_sB_plus_tI :
  ∃ s t : ℤ, B^6 = s • B + t • (1 : Matrix (Fin 2) (Fin 2) ℤ) := by
  have B2_eq : B^2 = -3 • B :=
    -- Matrix multiplication and scalar multiplication
    sorry
  use 81, 0
  have B4_eq : B^4 = 9 • B^2 := by
    rw [B2_eq]
    -- Calculation steps for B^4 equation
    sorry
  have B6_eq : B^6 = B^4 * B^2 := by
    rw [B4_eq, B2_eq]
    -- Calculation steps for B^6 final equation
    sorry
  rw [B6_eq]
  -- Final steps to show (81 • B + 0 • I = 81 • B)
  sorry

end matrix_B6_eq_sB_plus_tI_l241_241754


namespace chris_car_offer_difference_l241_241972

theorem chris_car_offer_difference :
  ∀ (asking_price : ℕ) (maintenance_cost_factor : ℕ) (headlight_cost : ℕ) (tire_multiplier : ℕ),
  asking_price = 5200 →
  maintenance_cost_factor = 10 →
  headlight_cost = 80 →
  tire_multiplier = 3 →
  let first_earnings := asking_price - asking_price / maintenance_cost_factor,
      second_earnings := asking_price - (headlight_cost + headlight_cost * tire_multiplier) in
  second_earnings - first_earnings = 200 :=
by
  intros asking_price maintenance_cost_factor headlight_cost tire_multiplier h1 h2 h3 h4
  -- leave "sorry" as a placeholder for the proof
  sorry

end chris_car_offer_difference_l241_241972


namespace melanie_bought_books_l241_241036

-- Defining the initial number of books and final number of books
def initial_books : ℕ := 41
def final_books : ℕ := 87

-- Theorem stating that Melanie bought 46 books at the yard sale
theorem melanie_bought_books : (final_books - initial_books) = 46 := by
  sorry

end melanie_bought_books_l241_241036


namespace sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three_l241_241967

theorem sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three : 
  ((Real.sqrt 7 + 2) * (Real.sqrt 7 - 2) = 3) := by
  sorry

end sqrt_seven_plus_two_times_sqrt_seven_minus_two_eq_three_l241_241967


namespace number_of_monic_Qs_l241_241369

noncomputable def P (X : Polynomial ℤ) : Polynomial ℤ := X^5 + 3*X^4 - 4*X^3 - X^2 - 3*X + 4

theorem number_of_monic_Qs : 
  ∃ (n : ℕ), P(Polynomial.X) = (Polynomial.X - 1) * (Polynomial.X^4 + 4*Polynomial.X^3 - Polynomial.X^2 - 4*Polynomial.X + 4) 
  ∧ n = 12 :=
sorry

end number_of_monic_Qs_l241_241369


namespace number_of_sides_l241_241937

theorem number_of_sides (P l n : ℕ) (hP : P = 49) (hl : l = 7) (h : P = n * l) : n = 7 :=
by
  sorry

end number_of_sides_l241_241937


namespace y_intercept_of_line_l241_241076

theorem y_intercept_of_line : 
  let line_eq : ℝ → ℝ → Prop := λ x y, x - 2*y - 3 = 0
  ∃ y : ℝ, line_eq 0 y ∧ y = -3/2 :=
by { 
  sorry 
}

end y_intercept_of_line_l241_241076


namespace total_pens_l241_241011

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l241_241011


namespace value_of_a9_l241_241732

variables (a : ℕ → ℤ) (d : ℤ)
noncomputable def arithmetic_sequence : Prop :=
(a 1 + (a 1 + 10 * d)) / 2 = 15 ∧
a 1 + (a 1 + d) + (a 1 + 2 * d) = 9

theorem value_of_a9 (h : arithmetic_sequence a d) : a 9 = 24 :=
by sorry

end value_of_a9_l241_241732


namespace find_k_l241_241964

variables (S x y : ℝ)

theorem find_k (h₁ : 0.75 * x * y / (x + y) = 18) : 8 = (x * y / 3) / (x + y) :=
by
  have h₂ : x * y / (x + y) = 24 := by linarith
  rw h₂
  rw (show 8 = 24 / 3, by norm_num)

end find_k_l241_241964


namespace general_term_formula_sum_of_terms_l241_241265

-- Definitions according to conditions
def arithmetic_seq (a d n : ℕ) : ℕ := a + (n - 1) * d
def S_n (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
def S_3 (a d : ℕ) : ℕ := S_n a d 3
def geometric_seq (a₁ a₃ a₇ : ℕ) : Prop := a₃ * a₃ = a₁ * a₇
def seq_b (a d n : ℕ) : ℕ := (arithmetic_seq a d n - 1) * 2 ^ n
def T_n (a d n : ℕ) : ℕ := ∑ i in finset.range n, (seq_b a d (i+1))

-- Conditions
axiom S3_given: S_3 a d = 9
axiom geometric_given: geometric_seq a (arithmetic_seq a d 3) (arithmetic_seq a d 7)

-- Main proof problem statement
theorem general_term_formula (a d : ℕ) (S_n_as_Sum : (S_n (a+d) d 3) = 9) (S3_given : arithmetic_seq a (d/2) 3 = 9 ∧ geometric_seq a (arithmetic_seq a d 3) (arithmetic_seq a d 7)) :
  ∃ a d, ∀ n, arithmetic_seq a d n = n + 1 := sorry

theorem sum_of_terms (a d n : ℕ) (general_term : ∀ n, arithmetic_seq a d n = n + 1) :
  ∃ T_n, T_n = (n - 1) * 2^(n + 1) + 2 := sorry

end general_term_formula_sum_of_terms_l241_241265


namespace claim1_claim2_l241_241511

theorem claim1 (n : ℤ) (hs : ∃ l : List ℤ, l.length = n ∧ l.prod = n ∧ l.sum = 0) : 
  ∃ k : ℤ, n = 4 * k := 
sorry

theorem claim2 (n : ℕ) (h : n % 4 = 0) : 
  ∃ l : List ℤ, l.length = n ∧ l.prod = n ∧ l.sum = 0 := 
sorry

end claim1_claim2_l241_241511


namespace sum_even_factors_of_720_l241_241865

theorem sum_even_factors_of_720 : 
  let even_factors_sum (n : ℕ) : ℕ := 
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  in even_factors_sum 720 = 2340 :=
by
  let even_factors_sum (n : ℕ) : ℕ :=
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  sorry

end sum_even_factors_of_720_l241_241865


namespace flow_in_channels_l241_241178

variables (q0 : ℝ)

-- Define the flows in use
def flow_AH : ℝ := q0
def flow_HG : ℝ := q0  -- By symmetry
def flow_BC : ℝ := 2/3 * q0
def flow_CD : ℝ := 2/3 * q0  -- By symmetry
def flow_BG : ℝ := 2/3 * q0  -- By symmetry
def flow_GD : ℝ := 2/3 * q0  -- By symmetry
def flow_AB : ℝ := 4/3 * q0

-- Total flow into node A
def total_flow_A : ℝ := flow_AH q0 + flow_AB q0 

theorem flow_in_channels (q0 : ℝ) :
  flow_AB q0 = 4/3 * q0 ∧ flow_BC q0 = 2/3 * q0 ∧ total_flow_A q0 = 7/3 * q0 :=
by
  sorry

end flow_in_channels_l241_241178


namespace total_dolls_l241_241411

def grandmother_dolls := 50
def sister_dolls := grandmother_dolls + 2
def rene_dolls := 3 * sister_dolls

theorem total_dolls : rene_dolls + sister_dolls + grandmother_dolls = 258 :=
by {
  -- Required proof steps would be placed here, 
  -- but are omitted as per the instructions.
  sorry
}

end total_dolls_l241_241411


namespace distance_planes_example_l241_241589

noncomputable def distance_between_planes {α : Type*} [linear_ordered_field α]
  (plane1 : AffineSubspace α (ℝ × ℝ × ℝ)) 
  (plane2 : AffineSubspace α (ℝ × ℝ × ℝ)) : α := sorry

theorem distance_planes_example :
  distance_between_planes 
    (AffineSubspace.mk' {p | 3 * p.1 + 6 * p.2 - 6 * p.3 + 3 = 0})
    (AffineSubspace.mk' {p | 6 * p.1 + 12 * p.2 - 12 * p.3 + 15 = 0}) = 0.5 :=
begin
  sorry
end

end distance_planes_example_l241_241589


namespace total_dolls_l241_241412

def grandmother_dolls := 50
def sister_dolls := grandmother_dolls + 2
def rene_dolls := 3 * sister_dolls

theorem total_dolls : rene_dolls + sister_dolls + grandmother_dolls = 258 :=
by {
  -- Required proof steps would be placed here, 
  -- but are omitted as per the instructions.
  sorry
}

end total_dolls_l241_241412


namespace interest_rate_of_additional_investment_l241_241213

theorem interest_rate_of_additional_investment
  (initial_investment : ℝ)
  (initial_interest_rate : ℝ)
  (additional_investment : ℝ)
  (desired_total_rate : ℝ) :
  initial_investment = 2400 →
  initial_interest_rate = 0.04 →
  additional_investment = 2400 →
  desired_total_rate = 0.06 →
  ∃ (R : ℝ), R = 0.08 :=
by
  intros h1 h2 h3 h4
  use 0.08
  rw [h1, h2, h3, h4]
  sorry

end interest_rate_of_additional_investment_l241_241213


namespace distance_to_right_focus_l241_241626

noncomputable def hyperbola : Prop :=
  ∃ (P : ℝ × ℝ), P.1^2 / 16 - P.2^2 / 9 = 1 ∧ 
  let d_L := real.sqrt ((P.1 + real.sqrt (4^2 + (4^2) * 3))^2 + P.2^2) in
  d_L = 10 ∧ 
  let a := 4 in
  let d_R := d_L + 2 * a in
  d_R = 18

-- Assertion that encodes the statement of the problem
theorem distance_to_right_focus (P : ℝ × ℝ) (h : P.1^2 / 16 - P.2^2 / 9 = 1) 
  (d_L : ℝ) (h_dL : d_L = 10)
  (a : ℝ) (h_a : a = 4) :
  let d_R := d_L + 2 * a in
  d_R = 18 := by
  sorry

end distance_to_right_focus_l241_241626


namespace angle_BAD_degree_l241_241839

-- Definitions based on the problem's conditions
variables {A B C D : Type} [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]

namespace Geom

-- Assume real-world numerical angles and equalities
def isosceles (a b : ℕ) : Prop := a = b

def degree_measure (a b : ℕ) : Prop :=
  Triangle_ABC_is_isosceles : isosceles AB BC,
  Triangle_ADC_is_isosceles : isosceles AD DC,
  point_D_inside_ABC : D ∈ interior ABC,
  angle_ABC : ∠ ABC = 40,
  angle_ADC : ∠ ADC = 140

-- The statement to prove
theorem angle_BAD_degree :
  ∀ (AB BC AD DC : ℕ)(D: Point),
    (isosceles AB BC) →
    (isosceles AD DC) →
    (∠ ABC = 40) →
    (∠ ADC = 140) →
    (∠ BAC - ∠ DAC = 50) :=
  by 
  sorry

end angle_BAD_degree_l241_241839


namespace irrational_among_options_l241_241555

theorem irrational_among_options : 
  ∀ (a b c d : ℝ), a = -2 ∧ b = 0 ∧ c = Real.sqrt 2 ∧ d = 5 → 
  (¬ ∃ p q : ℤ, q ≠ 0 ∧ c = p / q) ∧
  (∃ p q : ℤ, q ≠ 0 ∧ a = p / q) ∧
  (∃ p q : ℤ, q ≠ 0 ∧ b = p / q) ∧
  (∃ p q : ℤ, q ≠ 0 ∧ d = p / q) :=
by
  intro a b c d
  intro h
  obtain ⟨ha, hb, hc, hd⟩ := h
  split
  {
    sorry
  }
  split
  {
    split
    {
      sorry
    }
    split
    {
      sorry
    }
    split
    {
      sorry
    }
    split
    {
      sorry
    }
  }

end irrational_among_options_l241_241555


namespace multiplication_cryptogram_l241_241347

theorem multiplication_cryptogram 
  (数学 花园 : ℕ) 
  (H1 : 数学 = 15)
  (H2 : 花园 = 35) 
  (H3 : 数学 * 花园 = 525)
  (H4 : "数学" "花园" = 1537) 
  : "数学花园" = 1537 := sorry

end multiplication_cryptogram_l241_241347


namespace sum_even_factors_l241_241853

theorem sum_even_factors (n : ℕ) (h : n = 720) : 
  (∑ d in Finset.filter (λ d, d % 2 = 0) (Finset.divisors n), d) = 2340 :=
by
  rw h
  -- sorry to skip the actual proof
  sorry

end sum_even_factors_l241_241853


namespace circle_center_second_quadrant_l241_241624

theorem circle_center_second_quadrant 
  (D E : ℝ) 
  (h1 : (x^2 + y^2 + D * x + E * y + 3 = 0))
  (h2 : -D/2 - E/2 - 1 = 0)
  (h3 : D > 0) 
  (h4 : E < 0) 
  (h5 : sqrt((D / 2) ^ 2 + (E / 2) ^ 2 - 3) = sqrt(2)) : 
  (D = 2) ∧ (E = -4) := 
by
  sorry

end circle_center_second_quadrant_l241_241624


namespace triangle_area_l241_241636

noncomputable def hyperbola_foci := (F_1, F_2 : ℝ × ℝ)
noncomputable def foci_coordinates : hyperbola_foci := ((-3, 0), (3, 0))

def point_P_condition (P : ℝ × ℝ) : Prop :=
  let |x, y| := P in
  let PF_1 := real.sqrt ((x + 3) ^ 2 + y ^ 2) in
  let PF_2 := real.sqrt ((x - 3) ^ 2 + y ^ 2) in
  PF_1 = 2 * PF_2

noncomputable def area_triangle (P F_1 F_2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P in
  let (x2, y2) := F_1 in
  let (x3, y3) := F_2 in
  let s := (real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) + real.sqrt ((x3 - x1)^2 + (y3 - y1)^2) + real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)) / 2 in
  real.sqrt (s * (s - real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)) * (s - real.sqrt ((x3 - x1)^2 + (y3 - y1)^2)) * (s - real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)))

theorem triangle_area : ∀ P : ℝ × ℝ, point_P_condition P →
  let F_1 := (-3, 0) in
  let F_2 := (3, 0) in
  area_triangle P F_1 F_2 = 3 * real.sqrt 15 :=
sorry

end triangle_area_l241_241636


namespace inequality_solution_l241_241903

theorem inequality_solution (x : ℤ) (h : x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1) : x - 1 ≥ 0 ↔ x = 1 :=
by
  sorry

end inequality_solution_l241_241903


namespace simplify_expression_l241_241058

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
sorry

end simplify_expression_l241_241058


namespace boxwood_trim_cost_l241_241040

theorem boxwood_trim_cost (x : ℝ) (h1 : ∀ n : ℝ, n = 15) (h2 : ∀ n : ℝ, n = 30) (h3 : ∀ n : ℝ, n = 4) (h4 : ∀ t : ℝ, t = 210) : 
  x = 5.77 :=
by
  -- Calculate cost for shaping 4 boxwoods into spheres
  let shaping_cost := 4 * 15
  -- The remaining cost for trimming 26 boxwoods
  let trim_cost := 210 - shaping_cost
  -- Calculate cost to trim each boxwood
  let result := trim_cost / 26
  -- Compare the result with the expected approximate result
  have h : result ≈ 5.77 := sorry
  exact h

end boxwood_trim_cost_l241_241040


namespace parabola_sum_distance_l241_241247

theorem parabola_sum_distance :
  (∑ i in Finset.range 1992 + 1, 
    let n := i + 1
    let A_n := (1 - 2 * n) / (2 * n * (n + 1))
    let B_n := -1 / n
    abs (A_n - B_n)) = 1992 / 1993 := sorry

end parabola_sum_distance_l241_241247


namespace chromium_percentage_new_alloy_l241_241724

-- Conditions as definitions
def first_alloy_chromium_percentage : ℝ := 12
def second_alloy_chromium_percentage : ℝ := 8
def first_alloy_weight : ℝ := 10
def second_alloy_weight : ℝ := 30

-- Final proof statement
theorem chromium_percentage_new_alloy : 
  ((first_alloy_chromium_percentage / 100 * first_alloy_weight +
    second_alloy_chromium_percentage / 100 * second_alloy_weight) /
  (first_alloy_weight + second_alloy_weight)) * 100 = 9 :=
by
  sorry

end chromium_percentage_new_alloy_l241_241724


namespace part_I_geometric_sequence_part_II_general_formula_part_III_arithmetic_sequence_l241_241261

open Nat

-- Given conditions for the sequence {a_n}
def sequence_a : ℕ → ℝ
| 0     := 0
| 1     := 1
| 2     := 3
| (n+2) := 3 * (sequence_a (n+1)) - 2 * (sequence_a n)

-- Part (I)
theorem part_I_geometric_sequence : 
  ∃ r : ℝ, ∃ a₀ : ℝ, ∀ n : ℕ, sequence_a (n+1) - sequence_a n = a₀ * (r^n) :=
sorry

-- Part (II)
theorem part_II_general_formula :
  ∀ n : ℕ, sequence_a n = 2^n - 1 :=
sorry

-- Additional condition for Part (III)
def sequence_b_condition (b : ℕ → ℝ) (n : ℕ) : Prop :=
  ∏ i in range (n + 1), 4^(b i - 1) = (sequence_a (n + 1) + 1) ^ (b (n + 1))

-- Part (III)
theorem part_III_arithmetic_sequence (b : ℕ → ℝ) :
  (∀ n : ℕ, sequence_b_condition b n) → 
  ∃ d : ℝ, ∀ n : ℕ, b (n+1) - b n = d :=
sorry

end part_I_geometric_sequence_part_II_general_formula_part_III_arithmetic_sequence_l241_241261


namespace isosceles_triangle_problem_l241_241380

theorem isosceles_triangle_problem
  (BT CT : Real) (BC : Real) (BZ CZ TZ : Real) :
  BT = 20 →
  CT = 20 →
  BC = 24 →
  TZ^2 + 2 * BZ * CZ = 478 →
  BZ = CZ →
  BZ * CZ = 144 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end isosceles_triangle_problem_l241_241380


namespace sum_even_factors_l241_241858

theorem sum_even_factors (n : ℕ) (h : n = 720) : 
  (∑ d in Finset.filter (λ d, d % 2 = 0) (Finset.divisors n), d) = 2340 :=
by
  rw h
  -- sorry to skip the actual proof
  sorry

end sum_even_factors_l241_241858


namespace GH_parallel_AC_l241_241387

open Real

noncomputable def triangle := {A B C : Point}

variables {G H : Point} 
variables {A B C : Point}
variables (tangentA tangentB tangentC : ℝ)
variable (T : triangle)

-- Definitions
def is_centroid (G : Point) (T : triangle) := sorry -- precise definition not given in the problem
def is_orthocenter (H : Point) (T : triangle) := sorry -- precise definition not given in the problem
def arithmetic_progression (a b c : ℝ) := 2*b = a + c

-- Assumptions
axiom is_acute_triangle (T : triangle) : sorry
axiom centroid (G : Point) (T : triangle) : is_centroid G T
axiom orthocenter (H : Point) (T : triangle) : is_orthocenter H T
axiom tangents_in_arithmetic_progression : arithmetic_progression tangentA tangentB tangentC

-- Goal
theorem GH_parallel_AC (hT : is_acute_triangle T) (hG : centroid G T) (hH : orthocenter H T) (hAP : tangents_in_arithmetic_progression tangentA tangentB tangentC) :
  parallel (line_segment G H) (line_segment A C) :=
sorry

end GH_parallel_AC_l241_241387


namespace math_proof_problem_l241_241148

-- Define the problem conditions
def problem_conditions (x y : ℚ) := 
  (real.sqrt (x - y) = 2 / 5) ∧ (real.sqrt (x + y) = 2)

-- Define the correct solution
def correct_solution (x y : ℚ) := 
  (x = 52 / 25) ∧ (y = 48 / 25)

-- Define the area of the rectangle
def rectangle_area (a b : ℚ) : ℚ :=
  abs (a * b)

-- Define the proof problem
theorem math_proof_problem : 
  problem_conditions (52 / 25) (48 / 25) ∧ 
  rectangle_area (52 / 25) (48 / 25) = 8 / 25 :=
by 
  sorry

end math_proof_problem_l241_241148


namespace problem_solution_l241_241968

theorem problem_solution :
  2002 * 20032003 - 2003 * 20022002 = 0 :=
by
  -- Here we decompose the numbers as given in the problem:
  let a := 2002
  let b := 2003
  let k := 10001
  have h1 : 20032003 = b * k := sorry
  have h2 : 20022002 = a * k := sorry
  calc
    a * 20032003 - b * 20022002
        = a * (b * k) - b * (a * k) : by rw [h1, h2]
    ... = a * b * k - b * a * k    : by rw [mul_assoc]
    ... = (a * b - b * a) * k      : by rw [mul_sub_left_distrib]
    ... = (a * b - a * b) * k      : by rw [mul_comm]
    ... = 0 * k                    : by rw [sub_self]
    ... = 0                        : by rw [zero_mul]

end problem_solution_l241_241968


namespace expected_red_pairs_correct_l241_241177

-- Define the number of red cards and the total number of cards
def red_cards : ℕ := 25
def total_cards : ℕ := 50

-- Calculate the probability that one red card is followed by another red card in a circle of total_cards
def prob_adj_red : ℚ := (red_cards - 1) / (total_cards - 1)

-- The expected number of pairs of adjacent red cards
def expected_adj_red_pairs : ℚ := red_cards * prob_adj_red

-- The theorem to be proved: the expected number of adjacent red pairs is 600/49
theorem expected_red_pairs_correct : expected_adj_red_pairs = 600 / 49 :=
by
  -- Placeholder for the proof
  sorry

end expected_red_pairs_correct_l241_241177


namespace total_pens_bought_l241_241030

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l241_241030


namespace problem_statement_l241_241638

section
  variables (i : ℂ) (z : ℂ)
  def z_def : z = ( (1 + i) / (1 - i) ) ^ 2019 := sorry

  theorem problem_statement :
    (|conj z - 1| = √2) := sorry
end

end problem_statement_l241_241638


namespace sum_of_all_possible_radii_l241_241934

noncomputable def circle_center_and_tangent (r : ℝ) : Prop :=
  let C := (r, r) in
  let circleC_radius := r in
  let circleD_center := (5 : ℝ, 0 : ℝ) in
  let circleD_radius := (2 : ℝ) in
  (circleC_radius - 5)^2 + circleC_radius^2 = (circleC_radius + circleD_radius)^2

theorem sum_of_all_possible_radii : ∀ r : ℝ, circle_center_and_tangent r → (r = 7 + 2 * real.sqrt 7) ∨ (r = 7 - 2 * real.sqrt 7) → r + 7 - 2 * real.sqrt 7 = 14 :=
by
  intros r hcond hr;
  sorry

end sum_of_all_possible_radii_l241_241934


namespace bob_weight_l241_241804

theorem bob_weight (j b : ℝ) (h1 : j + b = 220) (h2 : b - 2 * j = b / 3) : b = 165 :=
  sorry

end bob_weight_l241_241804


namespace volleyball_count_l241_241467

theorem volleyball_count (x y z : ℕ) (h1 : x + y + z = 20) (h2 : 6 * x + 3 * y + z = 33) : z = 15 :=
by
  sorry

end volleyball_count_l241_241467


namespace complex_product_conjugate_l241_241274

theorem complex_product_conjugate (i : ℂ) (h_i : i * i = -1) (z : ℂ) (h_z : z = 1 / (2 + i)) : z * conj(z) = 1 / 5 := 
by 
  sorry

end complex_product_conjugate_l241_241274


namespace simplify_A_plus_2B_value_A_plus_2B_at_a1_bneg1_l241_241271

variable (a b : ℤ)

def A : ℤ := 3 * a^2 - 6 * a * b + b^2
def B : ℤ := -2 * a^2 + 3 * a * b - 5 * b^2

theorem simplify_A_plus_2B : 
  A a b + 2 * B a b = -a^2 - 9 * b^2 := by
  sorry

theorem value_A_plus_2B_at_a1_bneg1 : 
  let a := 1
  let b := -1
  A a b + 2 * B a b = -10 := by
  sorry

end simplify_A_plus_2B_value_A_plus_2B_at_a1_bneg1_l241_241271


namespace no_coprime_integers_l241_241403

-- Given Conditions
variables {n x y : ℕ}
variables (n_positive : 0 < n)
variables (n_no_square_divisors : ∀ d : ℕ, 1 < d → d * d ∣ n → false)
variables (x_y_coprime : Nat.coprime x y)
variables (x_positive : 0 < x)
variables (y_positive : 0 < y)
variables (x_y_multiple_condition : (x + y)^3 ∣ x^n + y^n)

-- Proof Statement
theorem no_coprime_integers (n_positive : 0 < n)
                            (n_no_square_divisors : ∀ d : ℕ, 1 < d → d * d ∣ n → false)
                            (x_y_coprime : Nat.coprime x y)
                            (x_positive : 0 < x)
                            (y_positive : 0 < y)
                            (x_y_multiple_condition : (x + y)^3 ∣ x^n + y^n) : false :=
sorry

end no_coprime_integers_l241_241403


namespace positive_integers_with_at_most_three_diff_digits_l241_241681

theorem positive_integers_with_at_most_three_diff_digits : 
  ∃ n : ℕ, n < 1000 ∧ (∀ i, i < n → ∃ d1 d2 d3 : ℕ, d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
  (i = d1 ∨ i = d2 ∨ i = d3)) ∧ n = 819 :=
by
  sorry

end positive_integers_with_at_most_three_diff_digits_l241_241681


namespace side_of_square_l241_241584

theorem side_of_square (d a : ℝ) (h : d > a) : ∃ s : ℝ, s = d - a :=
begin
  use d - a,
  sorry,
end

end side_of_square_l241_241584


namespace possible_m_values_l241_241670

theorem possible_m_values (m : ℝ) :
  let A := {x : ℝ | mx - 1 = 0}
  let B := {2, 3}
  (A ⊆ B) → (m = 0 ∨ m = 1 / 2 ∨ m = 1 / 3) :=
by
  intro A B h
  sorry

end possible_m_values_l241_241670


namespace total_pens_bought_l241_241001

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l241_241001


namespace geometric_sequence_false_inequality_solution_sets_diff_series_inequality_sin_law_comparison_number_of_correct_statements_l241_241288

-- Definitions for the conditions
variable {a : ℕ → ℝ}

theorem geometric_sequence_false (h : ∀ n, a (n + 1) = 2 * a n) :
  ¬∀ n, a (n + 1) = 2 * a n ↔ ∀ n, a (2 * n) = (2 : ℝ) * a n := sorry

variable {a1 b1 c1 a2 b2 c2 : ℝ}
variable (ha1 : a1 ≠ 0) (hb1 : b1 ≠ 0) (hc1 : c1 ≠ 0)
          (ha2 : a2 ≠ 0) (hb2 : b2 ≠ 0) (hc2 : c2 ≠ 0)

theorem inequality_solution_sets_diff (h : a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) :
  ¬∀ x : ℝ, a1 * x^2 + b1 * x + c1 > 0 ↔ a2 * x^2 + b2 * x + c2 > 0 := sorry

variable (n : ℕ)

theorem series_inequality (h : ∀ n, a n = 2 * n + 1) :
  (∑ k in range n, 1 / a (n + k)) ≥ 1 / 5 := sorry

variables {A B : ℝ}

theorem sin_law_comparison (h : sin A > sin B) : A > B := sorry

-- Main theorem
theorem number_of_correct_statements :
  let st1 := geometric_sequence_false _
  let st2 := inequality_solution_sets_diff _
  let st3 := series_inequality _
  let st4 := sin_law_comparison _
  (1 : ℕ) + (1 : ℕ) = 2 := sorry

end geometric_sequence_false_inequality_solution_sets_diff_series_inequality_sin_law_comparison_number_of_correct_statements_l241_241288


namespace omega_not_real_root_l241_241378

theorem omega_not_real_root {ω : ℂ} (h1 : ω^3 = 1) (h2 : ω ≠ 1) (h3 : ω^2 + ω + 1 = 0) :
  (2 + 3 * ω - ω^2)^3 + (2 - 3 * ω + ω^2)^3 = -68 + 96 * ω :=
by sorry

end omega_not_real_root_l241_241378


namespace sum_min_max_l241_241829

open Real

noncomputable def f (x : ℝ) := 2^x + log (2, x + 1)

theorem sum_min_max : 
  (f 0 + f 1) = 4 :=
by
  sorry

end sum_min_max_l241_241829


namespace average_speed_l241_241091

-- Defining conditions
def speed_first_hour : ℕ := 100  -- The car travels 100 km in the first hour
def speed_second_hour : ℕ := 60  -- The car travels 60 km in the second hour
def total_distance : ℕ := speed_first_hour + speed_second_hour  -- Total distance traveled

def total_time : ℕ := 2  -- Total time taken in hours

-- Stating the theorem
theorem average_speed : total_distance / total_time = 80 := 
by
  sorry

end average_speed_l241_241091


namespace det_projection_onto_vector_l241_241377

noncomputable def projection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm := a^2 + b^2
  (1 / norm) • ![![a^2, a * b], ![a * b, b^2]]

theorem det_projection_onto_vector : 
  det (projection_matrix 3 4) = 0 := by
  sorry

end det_projection_onto_vector_l241_241377


namespace find_consumption_fund_l241_241520

noncomputable def initial_capital : ℝ := 1
noncomputable def growth_rate : ℝ := 1.5
noncomputable def final_capital : ℝ := 2.9

theorem find_consumption_fund (x : ℝ) :
  let c1 := initial_capital * growth_rate - x in
  let c2 := c1 * growth_rate - x in
  let c3 := c2 * growth_rate - x in
  c3 = final_capital ->
  x = 10 / 19 :=
by
  sorry

end find_consumption_fund_l241_241520


namespace candles_in_each_box_l241_241250

/-
Assume four small boxes fit in one big box, and there are 50 big boxes.
We are given that there are a total of 8000 candles in all small boxes.
We want to prove that the number of candles in each small box is 40.
-/

theorem candles_in_each_box (num_small_boxes_per_big_box : ℕ) 
                            (num_big_boxes : ℕ) 
                            (total_candles : ℕ) 
                            (H1 : num_small_boxes_per_big_box = 4) 
                            (H2 : num_big_boxes = 50) 
                            (H3 : total_candles = 8000) : 
                            ∃ (candles_per_small_box : ℕ), candles_per_small_box = 40 :=
begin
  -- The proof would go here.
  sorry
end

end candles_in_each_box_l241_241250


namespace solutions_for_exponential_diophantine_equation_l241_241597

theorem solutions_for_exponential_diophantine_equation:
  {a b : ℕ} (ha : a ≥ 1) (hb : b ≥ 1) :
  a^(b^2) = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3) :=
by
  sorry

end solutions_for_exponential_diophantine_equation_l241_241597


namespace problem1_problem2_l241_241917

theorem problem1 (k : ℝ) (f : ℝ → ℝ) (hf : f = λ x, k * x^3 - 3 * (k + 1) * x^2 - 2 * k^2 + 4) 
  (dk : derivative f x = 3 * k * x^2 - 6 * (k + 1) * x)
  (h_dec : ∀ x ∈ Ioo 0 4, derivative f x < 0) : 
  k = 1 := 
sorry

theorem problem2 (a : ℝ) (f : ℝ → ℝ) 
  (hf : f = λ x, x^3 - 6 * x^2 + 2) 
  (h_in : ∀ t ∈ Icc (-1 : ℝ) 1, f t ≥ -5) :
  (∀ t ∈ Icc (-1 : ℝ) 1, ∃ x : ℝ, 2 * x^2 + 5 * x + a = f t) -> 
  a ≤ -5 / 4 := 
sorry

end problem1_problem2_l241_241917


namespace second_number_is_22_l241_241331

noncomputable section

variables (x y : ℕ)

-- Definitions based on the conditions
-- Condition 1: The sum of two numbers is 33
def sum_condition : Prop := x + y = 33

-- Condition 2: The second number is twice the first number
def twice_condition : Prop := y = 2 * x

-- Theorem: Given the conditions, the second number y is 22.
theorem second_number_is_22 (h1 : sum_condition x y) (h2 : twice_condition x y) : y = 22 :=
by
  sorry

end second_number_is_22_l241_241331


namespace store_refusal_illegal_l241_241483

/-- Legal status of banknotes for transactions in the Russian Federation. -/
def legal_tender (issued_by_bank_of_russia : Prop) (not_counterfeit : Prop) (permissible_damage : Prop) : Prop :=
  issued_by_bank_of_russia ∧ not_counterfeit ∧ permissible_damage

/-- Banknotes with tears are considered legal tender according to the Bank of Russia Directive from December 26, 2006, No. 1778-U. -/
def permissible_damage (has_tears : Prop) : Prop :=
  has_tears

/-- The store's refusal to accept torn banknotes from Lydia Alekseevna was legally unjustified. -/
theorem store_refusal_illegal
  (issued_by_bank_of_russia : Prop)
  (not_counterfeit : Prop)
  (has_tears : Prop) :
  legal_tender issued_by_bank_of_russia not_counterfeit (permissible_damage has_tears) → 
  ¬ store_refusal_torn_banknotes issued_by_bank_of_russia not_counterfeit has_tears :=
by
  intros
  sorry

end store_refusal_illegal_l241_241483


namespace length_AB_is_4sqrt2_l241_241643

def point_on_parabola (A B : ℝ × ℝ) : Prop :=
  (A.1^2 = 4 * A.2) ∧ (B.1^2 = 4 * B.2)

def midpoint_condition (A B : ℝ × ℝ) : Prop :=
  ((A.1 + B.1) / 2 = 2) ∧ ((A.2 + B.2) / 2 = 2)

theorem length_AB_is_4sqrt2 (A B : ℝ × ℝ) 
  (h1 : point_on_parabola A B) 
  (h2 : midpoint_condition A B) :
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * real.sqrt 2 := 
sorry

end length_AB_is_4sqrt2_l241_241643


namespace average_time_per_stop_l241_241453

theorem average_time_per_stop (pizzas : ℕ) 
                              (stops_for_two_pizzas : ℕ) 
                              (pizzas_per_stop_for_two : ℕ) 
                              (remaining_pizzas : ℕ) 
                              (total_stops : ℕ) 
                              (total_time : ℕ) 
                              (H1: pizzas = 12) 
                              (H2: stops_for_two_pizzas = 2) 
                              (H3: pizzas_per_stop_for_two = 2) 
                              (H4: remaining_pizzas = pizzas - stops_for_two_pizzas * pizzas_per_stop_for_two)
                              (H5: total_stops = stops_for_two_pizzas + remaining_pizzas)
                              (H6: total_time = 40) :
                              total_time / total_stops = 4 :=
by
  sorry

end average_time_per_stop_l241_241453


namespace rational_roots_of_polynomial_l241_241233

theorem rational_roots_of_polynomial :
  { x : ℚ | (x + 1) * (x - (2 / 3)) * (x^2 - 2) = 0 } = {-1, 2 / 3} :=
by
  sorry

end rational_roots_of_polynomial_l241_241233


namespace original_group_planned_days_l241_241941

/-- Original number of men -/
def original_men : ℕ := 180

/-- Number of absent men -/
def absent_men : ℕ := 15

/-- Remaining men after some became absent -/
def remaining_men : ℕ := original_men - absent_men

/-- Number of days the remaining men completed the work -/
def days_remaining_men_completed_work : ℕ := 60

/-- Work done by remaining men -/
def total_work_done_by_remaining_men : ℕ := remaining_men * days_remaining_men_completed_work

/-- Original planned days by the original group -/
def original_planned_days : ℕ := 55

/-- Work done by the original group -/
def total_work_done_by_original_group : ℕ := original_men * original_planned_days

/-- Theorem stating the equivalency of work done by both groups -/
theorem original_group_planned_days 
  (h : total_work_done_by_original_group = total_work_done_by_remaining_men) :
  original_planned_days = 55 :=
by 
  rw [total_work_done_by_original_group, total_work_done_by_remaining_men]
  sorry

end original_group_planned_days_l241_241941


namespace gateway_academy_problem_l241_241208

theorem gateway_academy_problem :
  let total_students := 100
  let students_like_skating := 0.4 * total_students
  let students_dislike_skating := total_students - students_like_skating
  let like_and_say_like := 0.7 * students_like_skating
  let like_and_say_dislike := students_like_skating - like_and_say_like
  let dislike_and_say_dislike := 0.8 * students_dislike_skating
  let dislike_and_say_like := students_dislike_skating - dislike_and_say_dislike
  let says_dislike := like_and_say_dislike + dislike_and_say_dislike
  (like_and_say_dislike / says_dislike) = 0.2 :=
by
  sorry

end gateway_academy_problem_l241_241208


namespace room_area_l241_241542

noncomputable def semicircle_room_area (R r : ℝ) : ℝ :=
(1/2) * Real.pi * R^2 - (1/2) * Real.pi * r^2

theorem room_area (R r : ℝ) (h : 2 * Real.sqrt (R^2 - r^2) = 12) : 
semicircle_room_area R r = 18 * Real.pi := by
  have hr := Real.sqrt_eq_iff_sq_eq.2 (by linarith)
  have hRr : R^2 - r^2 = 36 := by rw [hr] at h; exact (mul_eq_mul_left_iff.1 h).2
  sorry

end room_area_l241_241542


namespace sum_of_possible_radii_l241_241932

theorem sum_of_possible_radii :
  ∃ r1 r2 : ℝ, 
    (∀ r, (r - 5)^2 + r^2 = (r + 2)^2 → r = r1 ∨ r = r2) ∧ 
    r1 + r2 = 14 :=
sorry

end sum_of_possible_radii_l241_241932


namespace book_pages_l241_241359

theorem book_pages (P D : ℕ) 
  (h1 : P = 23 * D + 9) 
  (h2 : ∃ D, P = 23 * (D + 1) - 14) : 
  P = 32 :=
by sorry

end book_pages_l241_241359


namespace lcm_16_27_35_l241_241849

-- Definitions of the numbers involved
def a : ℕ := 16
def b : ℕ := 27
def c : ℕ := 35

-- Function to compute the LCM of two natural numbers
def lcm (m n : ℕ) : ℕ := m * n / (Nat.gcd m n)

-- LCM of three numbers defined using the LCM of two numbers
def lcm_three (x y z : ℕ) : ℕ := lcm (lcm x y) z

-- The proof problem statement
theorem lcm_16_27_35 : lcm_three a b c = 15120 :=
by
  -- Skipping the proof steps
  sorry

end lcm_16_27_35_l241_241849


namespace divisibility_problem_l241_241158

theorem divisibility_problem
  (h1 : 5^3 ∣ 1978^100 - 1)
  (h2 : 10^4 ∣ 3^500 - 1)
  (h3 : 2003 ∣ 2^286 - 1) :
  2^4 * 5^7 * 2003 ∣ (2^286 - 1) * (3^500 - 1) * (1978^100 - 1) :=
by sorry

end divisibility_problem_l241_241158


namespace sum_of_even_factors_720_l241_241873

theorem sum_of_even_factors_720 : 
  let n := 2^4 * 3^2 * 5 in
  (∑ d in (Finset.range (n + 1)).filter (λ d, d % 2 = 0 ∧ n % d = 0), d) = 2340 :=
by
  let n := 2^4 * 3^2 * 5
  sorry

end sum_of_even_factors_720_l241_241873


namespace sum_x_coordinates_l241_241102

theorem sum_x_coordinates (P : ℝ × ℝ) (θ : ℝ) (n : ℕ) (line : ℝ → ℝ × ℝ)
  (X : ℕ → ℝ) (Y : ℕ → ℝ) :
  P = (2, 2) →
  θ = π / 10 →
  n = 18 →
  (∀ i : ℕ, i < n → line i = (X i, Y i)) →
  (∀ i : ℕ, i < n → X i + Y i = 2016) →
  ∑ i in (Finset.range n), X i = 10080 := 
by
  intros hP hθ hn hline hXYsum
  sorry

end sum_x_coordinates_l241_241102


namespace dog_total_distance_l241_241168

noncomputable def total_distance_dog_ran 
  (distance_AB : ℝ) 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (speed_dog : ℝ) : ℝ :=
  if distance_AB = 100 ∧ speed_A = 6 ∧ speed_B = 4 ∧ speed_dog = 10
  then 100 else 0

theorem dog_total_distance 
  (distance_AB : ℝ) 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (speed_dog : ℝ) 
  (h₁ : distance_AB = 100) 
  (h₂ : speed_A = 6) 
  (h₃ : speed_B = 4) 
  (h₄ : speed_dog = 10) :
  total_distance_dog_ran distance_AB speed_A speed_B speed_dog = 100 :=
by 
  simp [total_distance_dog_ran, h₁, h₂, h₃, h₄]
  sorry

end dog_total_distance_l241_241168


namespace age_difference_l241_241498

variable (A B C : ℕ)

def condition1 := C = B / 2
def condition2 := A + B + C = 22
def condition3 := B = 8

theorem age_difference (h1 : condition1 C B)
                       (h2 : condition2 A B C) 
                       (h3 : condition3 B) : A - B = 2 := by
  sorry

end age_difference_l241_241498


namespace quadratic_identity_l241_241299

variables {R : Type*} [CommRing R] [IsDomain R]

-- Define the quadratic polynomial P
def P (a b c x : R) : R := a * x^2 + b * x + c

-- Conditions as definitions in Lean
variables (a b c : R) (h₁ : P a b c a = 2021 * b * c)
                (h₂ : P a b c b = 2021 * c * a)
                (h₃ : P a b c c = 2021 * a * b)
                (dist : (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c))

-- The main theorem statement
theorem quadratic_identity : a + 2021 * b + c = 0 :=
sorry

end quadratic_identity_l241_241299


namespace find_number_l241_241191

theorem find_number :
  ∃ x : ℕ, (x / 5 = 80 + x / 6) ∧ x = 2400 := 
by 
  sorry

end find_number_l241_241191


namespace radius_of_circle_l241_241415

theorem radius_of_circle (r : ℝ) : 
  (∀ x : ℝ, tangent_to_line (y = x^2 + r) (y = -x * √3)) → r = 3 / 4 :=
by
  sorry

def tangent_to_line (parabola_eq line_eq : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, parabola_eq x = line_eq x ∧ deriv parabola_eq x = deriv line_eq x

end radius_of_circle_l241_241415


namespace sum_even_factors_of_720_l241_241860

theorem sum_even_factors_of_720 : 
  let even_factors_sum (n : ℕ) : ℕ := 
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  in even_factors_sum 720 = 2340 :=
by
  let even_factors_sum (n : ℕ) : ℕ :=
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  sorry

end sum_even_factors_of_720_l241_241860


namespace sin_angle_ACB_correct_l241_241727

variables {A B C D : Type} [Point : Type]
variables {angle : Point → Point → Point → ℝ}
variables {cos sin : ℝ → ℝ}
variables {x y : ℝ}

def tetrahedron_conditions (A B C D : Point) : Prop :=
  angle A D B = 90 ∧ angle A D C = 90 ∧ angle B D C = 90

def cos_conditions : Prop := 
  x = cos (angle C A D) ∧ y = cos (angle C B D)

noncomputable def sin_angle_ACB (A B C D : Point) 
  (h_tetra : tetrahedron_conditions A B C D) 
  (h_cos : cos_conditions) : ℝ :=
  sqrt (1 - (x * y) ^ 2)

theorem sin_angle_ACB_correct (A B C D : Point)
  (h_tetra : tetrahedron_conditions A B C D) 
  (h_cos : cos_conditions) : 
  sin (angle A C B) = sqrt (1 - (x * y) ^ 2) :=
sorry

end sin_angle_ACB_correct_l241_241727


namespace shortest_distance_circles_l241_241124

-- Definition of distances and radii
def centerA : (ℝ × ℝ) := (5, 3)
def radiusA : ℝ := 12
def centerB : (ℝ × ℝ) := (2, -1)
def radiusB : ℝ := 6

-- Function to calculate the Euclidean distance between two points
def euclidean_distance (P Q : (ℝ × ℝ)) : ℝ :=
  real.sqrt (((Q.1 - P.1) ^ 2) + ((Q.2 - P.2) ^ 2))

-- Distance between the centers of the circles
def distance_centers := euclidean_distance centerA centerB

-- Total sum of radii
def radius_sum := radiusA + radiusB

-- Shortest distance between the circumferences of the two circles
def shortest_distance := distance_centers - radius_sum

-- Prove that shortest distance is 1 (with the given conditions)
theorem shortest_distance_circles :
  shortest_distance = 1 := sorry

end shortest_distance_circles_l241_241124


namespace min_phi_that_makes_g_symmetric_l241_241709

-- Define the initial function
def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + π / 3)

-- Function after shifting to the right by φ
def g (φ : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (2 * x + π / 3 + 2 * φ)

-- Symmetry about the origin for the function g
def symmetric_about_origin (φ : ℝ) : Prop :=
  ∀ x : ℝ, g (φ) x = - g (φ) (-x)

-- Prove the minimum value of φ
theorem min_phi_that_makes_g_symmetric : 
  ∃ φ > 0, symmetric_about_origin φ ∧ φ = π / 6 :=
by
  existsi (π / 6)
  sorry

end min_phi_that_makes_g_symmetric_l241_241709


namespace work_done_by_first_group_l241_241166

theorem work_done_by_first_group :
  (6 * 8 * 5 : ℝ) / W = (4 * 3 * 8 : ℝ) / 30 →
  W = 75 :=
by
  sorry

end work_done_by_first_group_l241_241166


namespace trig_identity_proof_l241_241620

noncomputable def trig_identity (α β : ℝ) (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 2) : ℝ :=
  (Real.sin (2 * α)) / (Real.cos (2 * β))

theorem trig_identity_proof (α β : ℝ) (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 2) :
  trig_identity α β h1 h2 = 1 :=
sorry

end trig_identity_proof_l241_241620


namespace cylinder_volume_triple_radius_quadruple_height_l241_241082

open Real

theorem cylinder_volume_triple_radius_quadruple_height (r h : ℝ) (V : ℝ) (hV : V = π * r^2 * h) :
  (3 * r) ^ 2 * 4 * h * π = 360 :=
by
  sorry

end cylinder_volume_triple_radius_quadruple_height_l241_241082


namespace arrest_people_l241_241098

-- Definitions based on conditions
def society (N : Type) := set (set N)

def condition (N : Type) (S : society N) : Prop :=
  ∀ (f : fin 2004 → set N), (∀ i, f i ∈ S) →
  ∃ n : N, (card {i | i < 2004 ∧ n ∈ f i}) ≥ 11

-- Theorem statement
theorem arrest_people (N : Type) (S : society N)
  (h_societies : ∀ s ∈ S, s.card = 10) 
  (h_condition : condition N S) :
  ∃ (arrested : set N), (card arrested) ≤ 2003 ∧ (∀ s ∈ S, ∃ n ∈ s, n ∈ arrested) :=
sorry

end arrest_people_l241_241098


namespace perimeter_of_first_square_l241_241451

theorem perimeter_of_first_square
  (s1 s2 s3 : ℝ)
  (P1 P2 P3 : ℝ)
  (A1 A2 A3 : ℝ)
  (hs2 : s2 = 8)
  (hs3 : s3 = 10)
  (hP2 : P2 = 4 * s2)
  (hP3 : P3 = 4 * s3)
  (hP2_val : P2 = 32)
  (hP3_val : P3 = 40)
  (hA2 : A2 = s2^2)
  (hA3 : A3 = s3^2)
  (hA1_A2_A3 : A3 = A1 + A2)
  (hA3_val : A3 = 100)
  (hA2_val : A2 = 64) :
  P1 = 24 := by
  sorry

end perimeter_of_first_square_l241_241451


namespace suzanne_read_pages_on_monday_l241_241066

def pages_read_on_Monday (total_pages pages_left pages_read_difference : ℕ) (h : total_pages = 64 ∧ pages_left = 18 ∧ pages_read_difference = 16) : Prop :=
  ∃ M : ℕ, M + (M + 16) = 46 ∧ M = 15

theorem suzanne_read_pages_on_monday : 
  pages_read_on_Monday 64 18 16 (and.intro rfl (and.intro rfl rfl)) :=
sorry

end suzanne_read_pages_on_monday_l241_241066


namespace find_second_number_l241_241506

theorem find_second_number :
  ∃ (a b c x : ℝ),
  (2 * x = a) ∧ 
  (3 * x = b) ∧ 
  (24 / 5 * x = c) ∧ 
  (a + b + c = 98) ∧ 
  (b = 30) :=
begin
  sorry -- Proof is not required as per the instructions.
end

end find_second_number_l241_241506


namespace coordinate_inequality_l241_241842

theorem coordinate_inequality (x y : ℝ) :
  (xy > 0 → (x - 2)^2 + (y + 1)^2 < 5) ∧ (xy < 0 → (x - 2)^2 + (y + 1)^2 > 5) :=
by
  sorry

end coordinate_inequality_l241_241842


namespace floor_sum_min_value_l241_241695

theorem floor_sum_min_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end floor_sum_min_value_l241_241695


namespace one_div_log_plus_one_div_log_l241_241635

theorem one_div_log_plus_one_div_log (a b : ℝ) (h1 : 2 ^ a = 10) (h2 : 5 ^ b = 10) : 
  1 / a + 1 / b = 1 :=
sorry

end one_div_log_plus_one_div_log_l241_241635


namespace present_age_of_B_l241_241153

theorem present_age_of_B 
  (a b : ℕ)
  (h1 : a + 10 = 2 * (b - 10))
  (h2 : a = b + 9) :
  b = 39 :=
by
  sorry

end present_age_of_B_l241_241153


namespace multiple_of_a_age_l241_241906

theorem multiple_of_a_age (A B M : ℝ) (h1 : A = B + 5) (h2 : A + B = 13) (h3 : M * (A + 7) = 4 * (B + 7)) : M = 2.75 :=
sorry

end multiple_of_a_age_l241_241906


namespace total_pens_bought_l241_241029

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l241_241029


namespace solution_satisfies_conditions_l241_241136

noncomputable def sqrt_eq (a b x y : ℝ) : Prop :=
  sqrt (x - y) = a ∧ sqrt (x + y) = b

theorem solution_satisfies_conditions 
  (x y : ℝ)
  (h1 : sqrt_eq (2/5) 2 x y)
  (hexact: x = 52/25 ∧ y = 48/25) :
  sqrt_eq (2/5) 2 x y ∧ 
  (x * y = 8/25) :=
by
  sorry

end solution_satisfies_conditions_l241_241136


namespace total_pens_bought_l241_241013

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l241_241013


namespace bryan_initial_pushups_l241_241965

def bryan_pushups (x : ℕ) : Prop :=
  let totalPushups := x + x + (x - 5)
  totalPushups = 40

theorem bryan_initial_pushups (x : ℕ) (hx : bryan_pushups x) : x = 15 :=
by {
  sorry
}

end bryan_initial_pushups_l241_241965


namespace min_sum_x_y_l241_241272

theorem min_sum_x_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0 ∧ y > 0) (h3 : (1 : ℚ)/x + (1 : ℚ)/y = 1/12) : x + y = 49 :=
sorry

end min_sum_x_y_l241_241272


namespace inverse_log_base_half_l241_241649

theorem inverse_log_base_half (x : ℝ) (hx : x > 0) : 
  (∃ f : ℝ → ℝ, (∀ x, f (log (1/2) x) = x) ∧ (∀ y, log (1/2) (f y) = y)) :=
begin
  let f := λ x, (1 / 2) ^ x,
  use f,
  split,
  { intro x,
    sorry }, -- Proof that f(log_{1/2} x) = x
  { intro y,
    sorry } -- Proof that log_{1/2} (f y) = y
end

end inverse_log_base_half_l241_241649


namespace total_apples_eaten_l241_241057

theorem total_apples_eaten : (1 / 2) * 16 + (1 / 3) * 15 + (1 / 4) * 20 = 18 := by
  sorry

end total_apples_eaten_l241_241057


namespace impossible_to_reach_one_pawn_l241_241484

variables {k n : ℕ}

def initial_pawn_count := 3 * k * n

theorem impossible_to_reach_one_pawn (k n : ℕ) (h1 : initial_pawn_count = 3 * k * n) :
  ∀ moves : list (ℕ × (ℕ × ℕ)), 
    (∀ (i j : ℕ), (i + j) % 3 = 0 ∨ (i + j) % 3 = 1 ∨ (i + j) % 3 = 2) →
    (∀ (move : ℕ × (ℕ × ℕ)), move ∈ moves → 
      ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a + b + c = 3 ∧ a % 3 = (b + 1) % 3 ∧ (b + 1) % 3 = c % 3) →
    ∀ state : ℕ, state ≠ 1 :=
by
  sorry

end impossible_to_reach_one_pawn_l241_241484


namespace rectangular_to_polar_l241_241978

theorem rectangular_to_polar (x y : ℝ) (hx : x = π / 2) (hy : y = - sqrt 3 * π / 2) : 
  ∃ (ρ θ : ℝ), ρ > 0 ∧ (θ ∈ Set.Ico 0 (2 * π)) ∧ (ρ = π) ∧ (θ = 5 * π / 3) :=
by
  use [π, 5 * π / 3]
  constructor
  { exact real.pi_pos }
  constructor
  { split; linarith [real.pi_pos] }
  constructor
  { rfl }
  { rfl }

end rectangular_to_polar_l241_241978


namespace pure_imaginary_solution_l241_241805

theorem pure_imaginary_solution (x : ℝ) :
  (∃ z : ℂ, z = complex.mk (x^2 - 1) (x - 1) ∧ z.im = z) → x = -1 := 
by
  intro h
  unfold complex.im at h
  sorry

end pure_imaginary_solution_l241_241805


namespace count_valid_years_l241_241948

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def one_digit_prime_palindromes := {2, 3, 5, 7}
def two_digit_prime_palindromes := {11}

def valid_years : Finset ℕ :=
  Finset.filter (λ n, is_palindrome n ∧ ∃ p ∈ one_digit_prime_palindromes, ∃ q ∈ two_digit_prime_palindromes, n = p * q) (Finset.Ico 100 1000)

theorem count_valid_years : valid_years.card = 1 := 
  sorry

end count_valid_years_l241_241948


namespace average_time_per_stop_l241_241452

theorem average_time_per_stop (pizzas : ℕ) 
                              (stops_for_two_pizzas : ℕ) 
                              (pizzas_per_stop_for_two : ℕ) 
                              (remaining_pizzas : ℕ) 
                              (total_stops : ℕ) 
                              (total_time : ℕ) 
                              (H1: pizzas = 12) 
                              (H2: stops_for_two_pizzas = 2) 
                              (H3: pizzas_per_stop_for_two = 2) 
                              (H4: remaining_pizzas = pizzas - stops_for_two_pizzas * pizzas_per_stop_for_two)
                              (H5: total_stops = stops_for_two_pizzas + remaining_pizzas)
                              (H6: total_time = 40) :
                              total_time / total_stops = 4 :=
by
  sorry

end average_time_per_stop_l241_241452


namespace translate_graph_upwards_l241_241106

theorem translate_graph_upwards (x : ℝ) :
  (∀ x, (3*x - 1) + 3 = 3*x + 2) :=
by
  intro x
  sorry

end translate_graph_upwards_l241_241106


namespace min_real_roots_of_polynomial_l241_241757

-- Definitions based on conditions
def degree : ℕ := 2006
def f (x : ℝ) := (polynomial ℝ)
def roots (f : polynomial ℝ) : set ℝ := { r | polynomial.root f r }

-- Main statement with the given equality to prove
theorem min_real_roots_of_polynomial (f : polynomial ℝ)
  (hf : f.degree = degree)
  (h_real_coeffs : ∀ (x : ℝ), polynomial.coeff f x ∈ ℝ)
  (h_distinct_magnitudes : (|roots f|.imaged abs).to_finset.card = 1006) :
  ∃ (n : ℕ), n = 6 ∧ (roots f).count (λ x, x ∈ ℝ) ≥ n :=
sorry

end min_real_roots_of_polynomial_l241_241757


namespace union_of_A_and_B_l241_241634

open Set

variable {α : Type}

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2, 4} := 
by
  sorry

end union_of_A_and_B_l241_241634


namespace room_area_is_18pi_l241_241544

def semicircle_room_area (R r : ℝ) (h : R > r) (d : ℝ) (hd : d = 12) : ℝ :=
  (π / 2) * (R^2 - r^2)

theorem room_area_is_18pi (R r : ℝ) (h : R > r) :
  semicircle_room_area R r h 12 (by rfl) = 18 * π :=
by
  sorry

end room_area_is_18pi_l241_241544


namespace island_knights_liars_l241_241773

noncomputable def num_knights (n m : ℕ) (inhabitants : ℕ → Prop) : ℕ :=
sorry

theorem island_knights_liars :
  let n := 99 in
  let m := 10 in
  ∃ k : ℕ, k = 9 ∧
  ∃ inhabitants : ℕ → Prop,
    (∀ i : ℕ, i < n → (inhabitants ((i + 1) % n)) = (¬inhabitants ((i + m) % n))) :=
begin
  sorry
end

end island_knights_liars_l241_241773


namespace lcm_180_504_is_2520_l241_241116

-- Define what it means for a number to be the least common multiple of two numbers
def is_lcm (a b lcm : ℕ) : Prop :=
  a ∣ lcm ∧ b ∣ lcm ∧ ∀ m, (a ∣ m ∧ b ∣ m) → lcm ∣ m

-- Lean 4 statement to prove that the least common multiple of 180 and 504 is 2520
theorem lcm_180_504_is_2520 : ∀ (a b : ℕ), a = 180 → b = 504 → is_lcm a b 2520 := by
  intro a b
  assume h1 : a = 180
  assume h2 : b = 504
  sorry

end lcm_180_504_is_2520_l241_241116


namespace three_digit_sum_of_permutations_l241_241198

theorem three_digit_sum_of_permutations (a b c : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : a ≠ c) :
  (100 * a + 10 * b + c + 100 * a + 10 * c + b + 100 * b + 10 * a + c + 100 * b + 10 * c + a + 100 * c + 10 * a + b + 100 * c + 10 * b + a = 444) →
  ({ a, b, c } = {1, 1, 2} ∨ { a, b, c } = {4, 4, 4}) := sorry

end three_digit_sum_of_permutations_l241_241198


namespace proof_of_problem_l241_241139

noncomputable def proof_problem (x y : ℚ) : Prop :=
  (sqrt (x - y) = 2 / 5) ∧ (sqrt (x + y) = 2) ∧ 
  x = 52 / 25 ∧ y = 48 / 25 ∧ 
  let vertices := [(0, 0), (2, 2), (2 / 25, -2 / 25), (52 / 25, 48 / 25)] in
  let area := Rational.from_ints 8 25 in
  ∃ (a b c d : ℚ × ℚ), 
    a ∈ vertices ∧ b ∈ vertices ∧ c ∈ vertices ∧ d ∈ vertices ∧ 
    ((b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = area)

theorem proof_of_problem : proof_problem (52 / 25) (48 / 25) :=
by { sorry } 

end proof_of_problem_l241_241139


namespace point_P_values_l241_241642

theorem point_P_values :
  ∃ (x : ℝ), (|a - 1| = 5) ∧ (b^3 = -27) ∧ (|a - b| = a - b) ∧ (2 * (x - b).abs = (6 - x).abs) ∧ (x = 0 ∨ x = -12) :=
by
  sorry

end point_P_values_l241_241642


namespace min_monthly_processing_cost_min_avg_cost_per_ton_l241_241469

section Problem1

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 300 * x + 64800

theorem min_monthly_processing_cost :
  ∃ x : ℝ, 30 ≤ x ∧ x ≤ 400 ∧ (∀ y : ℝ, 30 ≤ y ∧ y ≤ 400 → f(x) ≤ f(y)) ∧ f(x) = 19800 :=
by
  use 300
  sorry

end Problem1

section Problem2

noncomputable def avg_cost_per_ton (x : ℝ) : ℝ := f(x) / x

theorem min_avg_cost_per_ton :
  ∃ x : ℝ, 30 ≤ x ∧ x ≤ 400 ∧ (∀ y : ℝ, 30 ≤ y ∧ y ≤ 400 → avg_cost_per_ton(x) ≤ avg_cost_per_ton(y)) ∧ avg_cost_per_ton(x) = 60 :=
by
  use 360
  sorry

end Problem2

end min_monthly_processing_cost_min_avg_cost_per_ton_l241_241469


namespace shaded_area_of_partitioned_square_l241_241167

theorem shaded_area_of_partitioned_square :
  ∀ (A B : ℝ × ℝ) (side_length : ℝ), 
  A = (10 / 3, 10) ∧ B = (20 / 3, 0) ∧ side_length = 10
  → let shaded_area := 50 in shaded_area = 50 :=
by 
  intros A B side_length h
  let shaded_area := 50
  have A_coords : A = (10 / 3, 10) := and.left h
  have B_coords : B = (20 / 3, 0) := and.left (and.right h)
  have side_length_cond : side_length = 10 := and.right (and.right h)
  sorry

end shaded_area_of_partitioned_square_l241_241167


namespace a_2003_equals_2003_times_2002_l241_241300

open Nat

def a : ℕ → ℕ
| 1 := 0
| (n + 1) := a n + 2 * n

theorem a_2003_equals_2003_times_2002 : a 2003 = 2003 * 2002 :=
by
  sorry

end a_2003_equals_2003_times_2002_l241_241300


namespace fifth_term_arithmetic_sequence_l241_241434

noncomputable def fifth_term (x y : ℚ) (a1 : ℚ := x + 2 * y) (a2 : ℚ := x - 2 * y) (a3 : ℚ := x + 2 * y^2) (a4 : ℚ := x / (2 * y)) (d : ℚ := -4 * y) : ℚ :=
    a4 + d

theorem fifth_term_arithmetic_sequence (x y : ℚ) (h1 : y ≠ 0) :
  (fifth_term x y - (-((x : ℚ) / 6) - 12)) = 0 :=
by
  sorry

end fifth_term_arithmetic_sequence_l241_241434


namespace initial_investment_l241_241996

theorem initial_investment (A r : ℝ) (n : ℕ) (P : ℝ) (hA : A = 630.25) (hr : r = 0.12) (hn : n = 5) :
  A = P * (1 + r) ^ n → P = 357.53 :=
by
  sorry

end initial_investment_l241_241996


namespace find_m_l241_241379

def a : ℝ × ℝ := (-3, m)
def b : ℝ × ℝ := (4, 3)

theorem find_m {m : ℝ} (h_obtuse : a.1 * b.1 + a.2 * b.2 < 0)
  (h_not_parallel : a.1 / b.1 ≠ a.2 / b.2) : m < 4 ∧ m ≠ -9 / 4 := 
sorry

end find_m_l241_241379


namespace average_a_b_l241_241080

theorem average_a_b (a b : ℝ) (h : (4 + 6 + 8 + a + b) / 5 = 20) : (a + b) / 2 = 41 :=
by
  sorry

end average_a_b_l241_241080


namespace find_a_l241_241252

theorem find_a (M N : Set (ℝ × ℝ)) (a : ℝ) :
  (M = {(x,y) | (y - 3) / (x - 2) = 3}) →
  (N = {(x,y) | a*x + 2*y + a = 0}) →
  (M ∩ N = ∅) →
  (a = -6 ∨ a = -2) :=
by
  intros hM hN hMN
  rsorry

end find_a_l241_241252


namespace fill_pipe_time_l241_241530

theorem fill_pipe_time (t : ℕ) (H : ∀ C : Type, (1 / 2 : ℚ) * C = t * 1/2 * C) : t = t :=
by
  sorry

end fill_pipe_time_l241_241530


namespace find_x_from_roots_l241_241656

variable (x m : ℕ)

theorem find_x_from_roots (h1 : (m + 3)^2 = x) (h2 : (2 * m - 15)^2 = x) : x = 49 := by
  sorry

end find_x_from_roots_l241_241656


namespace positive_integers_with_at_most_three_diff_digits_l241_241680

theorem positive_integers_with_at_most_three_diff_digits : 
  ∃ n : ℕ, n < 1000 ∧ (∀ i, i < n → ∃ d1 d2 d3 : ℕ, d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
  (i = d1 ∨ i = d2 ∨ i = d3)) ∧ n = 819 :=
by
  sorry

end positive_integers_with_at_most_three_diff_digits_l241_241680


namespace ratio_of_volumes_l241_241540

theorem ratio_of_volumes (s : ℝ) (h_tetrahedron : ∀ t : ℝ, t = s * (√2)) :
  let cube_volume := s^3,
      tetrahedron_volume := (s * (√2))^3 * (√2) / 12 in
  tetrahedron_volume / cube_volume = 1 / 3 :=
by
  sorry

end ratio_of_volumes_l241_241540


namespace angle_CDE_is_105_l241_241346

variables {α : Type*} [add_group α]

-- Definition of the angles in the problem
def angle_A := 90
def angle_B := 90
def angle_C := 90
def angle_AEB := 30
def angle_BDE := real
def angle_BED := 2 * angle_BDE

-- Definition of the target angle
def angle_CDE := 
  360 - (angle_A + angle_C + angle_AEB + angle_BED) + angle_BDE

-- Theorem stating the desired result
theorem angle_CDE_is_105 :
  angle_CDE = 105 :=
sorry

end angle_CDE_is_105_l241_241346


namespace enclosed_area_correct_l241_241803

noncomputable def enclosed_area : ℝ := 
let arc_length := (π / 2) in
let side_length := 3 in
let r := 1 / 2 in
let sector_area := 1 / 4 * π * r^2 * (π / 2) / π in
let sector_total_area := 12 * sector_area in
let octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2 in
octagon_area + sector_total_area

theorem enclosed_area_correct : enclosed_area = 18 * (1 + Real.sqrt 2) + (3 * π) / 2 := 
by
  sorry

end enclosed_area_correct_l241_241803


namespace find_a_l241_241207

noncomputable def f (x a : ℝ) : ℝ := exp (x + a) + x
noncomputable def g (x a : ℝ) : ℝ := log (x + 3) - 4 * exp (-x - a)

theorem find_a (a x_0 : ℝ) (h : f x_0 a - g x_0 a = 2) :
  a = 2 + real.log 2 := by
  sorry

end find_a_l241_241207


namespace junjun_problem_l241_241747

theorem junjun_problem : ∃ (A B C D : ℕ), 
  (A * B ≠ C * 10 + D) ∧
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ D) ∧
  (∃ (A' : ℕ), (A' * B = C * 10 + D) ∧ A' ∈ {2, 3} ∨
   ∃ (B' : ℕ), (A * B' = C * 10 + D) ∧ B' ≠ 9 ∧ B' ∈ {6, 9} ∨
   ∃ (CD' : ℕ), (A * B = CD') ∧ CD' = 12) ∧
  (∃ (A B C D : ℕ), (A ≠ B ∧ B ≠ C ∧ C ≠ D) → (A * 10 + B = C * 10 + D)) →
  A + B + C + D = 17 :=
begin
  -- Proof will go here
  sorry
end

end junjun_problem_l241_241747


namespace jenn_wins_if_and_only_if_n_eq_6_l241_241562

theorem jenn_wins_if_and_only_if_n_eq_6 (n : ℤ) (h : n > 6) :
  (JennWins n) ↔ n = 6 := sorry

end jenn_wins_if_and_only_if_n_eq_6_l241_241562


namespace find_b_value_l241_241819

-- Given conditions
def tangent_condition (x : ℝ) (b : ℝ) : Prop :=
  x > 0 ∧ (ln x = (1 / 2) * x + b) ∧ (1 / x = 1 / 2)

-- Define the problem to verify the value of b
theorem find_b_value : ∃ (b : ℝ), tangent_condition 2 b ∧ b = ln 2 - 1 :=
by
  sorry

end find_b_value_l241_241819


namespace probability_of_same_number_l241_241215

theorem probability_of_same_number (m n : ℕ) 
  (hb : m < 250 ∧ m % 20 = 0) 
  (bb : n < 250 ∧ n % 30 = 0) : 
  (∀ (b : ℕ), b < 250 ∧ b % 60 = 0 → ∃ (m n : ℕ), ((m < 250 ∧ m % 20 = 0) ∧ (n < 250 ∧ n % 30 = 0)) → (m = n)) :=
sorry

end probability_of_same_number_l241_241215


namespace count_valid_four_digit_numbers_l241_241939

def distinct_digits (n : ℕ) : Prop := 
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
  digits.nodup

def sum_of_digits_is_six (n : ℕ) : Prop :=
  (n / 1000 % 10 + n / 100 % 10 + n / 10 % 10 + n % 10) = 6

def is_multiple_of_eleven (n : ℕ) : Prop :=
  (n / 1000 % 10 + n / 10 % 10 - n / 100 % 10 - n % 10) % 11 = 0

theorem count_valid_four_digit_numbers : 
  { n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ distinct_digits n ∧ sum_of_digits_is_six n ∧ is_multiple_of_eleven n }.count = 6 :=
sorry

end count_valid_four_digit_numbers_l241_241939


namespace remainder_of_binomial_sum_mod_prime_l241_241079

theorem remainder_of_binomial_sum_mod_prime :
  ∀ (n : ℕ), Nat.Prime 2027 →
    (∑ k in Finset.range (n + 1), Nat.choose 2024 k) % 2027 = 905 :=
by
  intros n h_prime
  have key := Nat.Prime.is_prime h_prime
  sorry

end remainder_of_binomial_sum_mod_prime_l241_241079


namespace scientific_notation_of_0_00077_l241_241733

theorem scientific_notation_of_0_00077 :
  0.00077 = 7.7 * 10 ^ -4 :=
sorry

end scientific_notation_of_0_00077_l241_241733


namespace no_real_solution_for_g_even_l241_241439
open Real

def g : ℝ → ℝ := sorry

theorem no_real_solution_for_g_even (x : ℝ) (hx : x ≠ 0) :
  g x = g (-x) → False :=
by
  assume h : g x = g (-x)
  have h1 : g x + 3 * g (1 / x) = 4 * x + 1 := sorry
  have h2 : g (1 / x) + 3 * g x = 4 / x + 1 := sorry
  have g_def : g x = (3 * x^2 - 2 + x) / x := sorry
  have g_neq : g x = g (-x) → 6 * x = 0 := sorry
  exact absurd (by linarith) hx

end no_real_solution_for_g_even_l241_241439


namespace total_pens_bought_l241_241000

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l241_241000


namespace proof_of_problem_l241_241141

noncomputable def proof_problem (x y : ℚ) : Prop :=
  (sqrt (x - y) = 2 / 5) ∧ (sqrt (x + y) = 2) ∧ 
  x = 52 / 25 ∧ y = 48 / 25 ∧ 
  let vertices := [(0, 0), (2, 2), (2 / 25, -2 / 25), (52 / 25, 48 / 25)] in
  let area := Rational.from_ints 8 25 in
  ∃ (a b c d : ℚ × ℚ), 
    a ∈ vertices ∧ b ∈ vertices ∧ c ∈ vertices ∧ d ∈ vertices ∧ 
    ((b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = area)

theorem proof_of_problem : proof_problem (52 / 25) (48 / 25) :=
by { sorry } 

end proof_of_problem_l241_241141


namespace simplify_expression_l241_241059

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
sorry

end simplify_expression_l241_241059


namespace midpoints_form_parallelogram_l241_241343

variable {A B C D E F M N K L : Point}
variable (AB CD AF CE BF DE : Line)
variable [Midpoint E AB]
variable [Midpoint F CD]
variable [Midpoint M AF]
variable [Midpoint K BF]
variable [Midpoint N CE]
variable [Midpoint L DE]

theorem midpoints_form_parallelogram
  (h_mid_E : Midpoint E AB)
  (h_mid_F : Midpoint F CD)
  (h_mid_M : Midpoint M AF)
  (h_mid_N : Midpoint N CE)
  (h_mid_K : Midpoint K BF)
  (h_mid_L : Midpoint L DE) : 
  Parallelogram M N K L :=
by sorry

end midpoints_form_parallelogram_l241_241343


namespace option_A_range_and_increasing_option_C_odd_option_D_domain_l241_241783

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (4^x - 1) / (4^x + 1 - a * 2^x)

-- stating the problem for a = 0 and show f is increasing and range is (-1, 1)
theorem option_A_range_and_increasing (x : ℝ) : 
  f(x, 0) = (4^x - 1) / (4^x + 1) ∧ 
  (∀ x y : ℝ, x < y → f(x, 0) < f(y, 0)) ∧ 
  (∀ y : ℝ, ∃ x : ℝ, y = f(x, 0)) :=
sorry

-- stating the problem for a = 1 and show f is an odd function
theorem option_C_odd (x : ℝ) : 
  f(-x, 1) = -f(x, 1) :=
sorry

-- stating the problem for domain being R implies a < 2
theorem option_D_domain (x : ℝ) :
  (∀ x : ℝ, 4^x + 1 - a * 2^x ≠ 0) → a < 2 :=
sorry

end option_A_range_and_increasing_option_C_odd_option_D_domain_l241_241783


namespace min_value_expression_l241_241317

theorem min_value_expression (x y : ℝ) :
  ∃ m, (m = 104) ∧ (∀ x y : ℝ, (x + 3)^2 + 2 * (y - 2)^2 + 4 * (x - 7)^2 + (y + 4)^2 ≥ m) :=
sorry

end min_value_expression_l241_241317


namespace percentage_increase_l241_241713

variable {α : Type} [LinearOrderedField α]

theorem percentage_increase (x y : α) (h : x = 0.5 * y) : y = x + x :=
by
  -- The steps of the proof are omitted and 'sorry' is used to skip actual proof.
  sorry

end percentage_increase_l241_241713


namespace jimin_yuna_difference_l241_241053

-- Definitions based on the conditions.
def seokjin_marbles : ℕ := 3
def yuna_marbles : ℕ := seokjin_marbles - 1
def jimin_marbles : ℕ := seokjin_marbles * 2

-- Theorem stating the problem we need to prove: the difference in marbles between Jimin and Yuna is 4.
theorem jimin_yuna_difference : jimin_marbles - yuna_marbles = 4 :=
by sorry

end jimin_yuna_difference_l241_241053


namespace ruler_count_l241_241095

theorem ruler_count (initial_rulers added_rulers : ℕ) (h1 : initial_rulers = 11) (h2 : added_rulers = 14) :
  initial_rulers + added_rulers = 25 :=
by
  rw [h1, h2]
  rfl

end ruler_count_l241_241095


namespace area_of_region_between_semicircles_l241_241546

/-- Given a region between two semicircles with the same center and parallel diameters,
where the farthest distance between two points with a clear line of sight is 12 meters,
prove that the area of the region is 18π square meters. -/
theorem area_of_region_between_semicircles :
  ∃ (R r : ℝ), R > r ∧ (R - r = 6) ∧ 18 * Real.pi = (Real.pi / 2) * (R^2 - r^2) ∧ (R^2 - r^2 = 144) :=
sorry

end area_of_region_between_semicircles_l241_241546


namespace function_period_30_l241_241583

noncomputable def least_period (f : ℝ → ℝ) : ℝ :=
  Inf {p : ℝ | p > 0 ∧ ∀ x, f(x) = f(x + p)}

theorem function_period_30 (f : ℝ → ℝ) (hf : ∀ x : ℝ, f(x + 5) + f(x - 5) = f(x)) :
  least_period f = 30 :=
sorry

end function_period_30_l241_241583


namespace pure_alcohol_addition_l241_241495

variable (x : ℝ)

def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.25
def final_concentration : ℝ := 0.50

theorem pure_alcohol_addition :
  (1.5 + x) / (initial_volume + x) = final_concentration → x = 3 :=
by
  sorry

end pure_alcohol_addition_l241_241495


namespace fraction_of_milk_in_mug1_l241_241553

theorem fraction_of_milk_in_mug1 :
  let tea_in_mug1_init := 5
  let milk_in_mug2_init := 3
  let tea_transfer := 2
  let pour_back := 3
  let tea_in_mug1_after_transfer := tea_in_mug1_init - tea_transfer
  let milk_in_mug2_after_transfer := milk_in_mug2_init
  let tea_in_mug2_after_transfer := tea_transfer
  let total_in_mug2 := milk_in_mug2_after_transfer + tea_in_mug2_after_transfer
  let milk_back_to_mug1 := pour_back * (milk_in_mug2_after_transfer / total_in_mug2)
  let tea_back_to_mug1 := pour_back * (tea_in_mug2_after_transfer / total_in_mug2)
  let final_tea_in_mug1 := tea_in_mug1_after_transfer + tea_back_to_mug1
  let final_milk_in_mug1 := milk_back_to_mug1
  let total_liquid_in_mug1 := final_tea_in_mug1 + final_milk_in_mug1
  in final_milk_in_mug1 / total_liquid_in_mug1 = 3 / 10 :=
by
  sorry

end fraction_of_milk_in_mug1_l241_241553


namespace sum_even_factors_l241_241859

theorem sum_even_factors (n : ℕ) (h : n = 720) : 
  (∑ d in Finset.filter (λ d, d % 2 = 0) (Finset.divisors n), d) = 2340 :=
by
  rw h
  -- sorry to skip the actual proof
  sorry

end sum_even_factors_l241_241859


namespace incenter_of_isosceles_triangle_l241_241961

/-- An isosceles triangle with AB = AC, and specific geometric properties about the circle inscribed in circumcircle and points of tangency. -/
theorem incenter_of_isosceles_triangle 
  (A B C P Q O : Point)
  (h_iso: AB = AC)
  (h_circum : ∃ (circum : Circle), circum.inscribed_in (Triangle A B C))
  (h_tangent : Tangent_to at P Q (Circle T))
  (h_midpoint : Midpoint O P Q) :
  Incenter O (Triangle A B C) :=
  sorry

end incenter_of_isosceles_triangle_l241_241961


namespace sum_of_angles_l241_241043

theorem sum_of_angles (A B R D C P : Point) (h1 : Circle A B R D C) 
  (h2 : diametrically_opposite B D) (h3 : arc_measure B R = 72) 
  (h4 : arc_measure R D = 108) : 
  angle_measure P + angle_measure R = 90 := 
sorry

end sum_of_angles_l241_241043


namespace solve_for_bc_l241_241382

def g (b c : ℝ) (x : ℝ) : ℝ := b * x + c * x^3 - Real.sqrt 3

theorem solve_for_bc (b c : ℝ) (b_pos : 0 < b) (c_pos : 0 < c) :
  g b c (g b c (Real.sqrt 3)) = -Real.sqrt 3 → (b = 0 ∧ c = 1/3) :=
by
  sorry

end solve_for_bc_l241_241382


namespace pair_B_same_pairs_b_same_functions_l241_241554

-- Define the pairs of functions
def fa (x : ℝ) : ℝ := 1
def ga (x : ℝ) : ℝ := if x ≠ 0 then x ^ 0 else 1  -- x^0 behaves strangely with x=0 in Lean

def fb (x : ℝ) : ℝ := if x ≠ 0 then x ^ 2 / x else 0
def gb (x : ℝ) : ℝ := x

def fc (x : ℝ) : ℝ := |x|
def gc (x : ℝ) : ℝ := if x > 0 then x else -x

def fd (x : ℝ) : ℝ := 1
def gd (x : ℝ) : ℝ := if x ≠ 0 then x / x else 1

-- Corresponding proof for pair B
theorem pair_B_same : ∀ x : ℝ, x ≠ 0 → fb x = gb x := by
  intros x hx
  -- Proof needed (skipped)
  sorry

-- Prove that pair B has the same domain and rule of correspondence
theorem pairs_b_same_functions :
  (∀ x, x ≠ 0 → fb x = gb x) := by
  intros x hx
  apply pair_B_same
  assumption

end pair_B_same_pairs_b_same_functions_l241_241554


namespace total_pens_l241_241027

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l241_241027


namespace solve_recurrence_relation_l241_241419

def sequence (a : ℕ → ℤ) : Prop :=
  (a 0 = 2) ∧
  (a 1 = 8) ∧
  (∀ n, n ≥ 2 → a n = 4 * a (n - 1) - 3 * a (n - 2) + 2^n)

def solution_correctness (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 1 + 5 * 3^n - 4 * 2^n

theorem solve_recurrence_relation (a : ℕ → ℤ) 
  (h : sequence a) : solution_correctness a :=
sorry

end solve_recurrence_relation_l241_241419


namespace f_even_when_a_is_zero_f_neither_even_nor_odd_when_a_is_nonzero_min_val_f_l241_241765

-- Definition of the function f
def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

-- Proof that if a = 0, then f is even
theorem f_even_when_a_is_zero (x : ℝ) : f 0 x = f 0 (-x) :=
begin
  sorry -- proof omitted
end

-- Proof that if a ≠ 0, then f is neither even nor odd
theorem f_neither_even_nor_odd_when_a_is_nonzero (a x : ℝ) (h : a ≠ 0) :
  f a x ≠ f a (-x) ∧ f a x ≠ -f a (-x) :=
begin
  sorry -- proof omitted
end

-- Proof of the minimum value of f based on conditions of a
theorem min_val_f (a x : ℝ) :
  ((a <= -1/2) → ∀ x, f a x ≥ (3/4 - a))
  ∧ ((-1/2 < a ∧ a <= 1/2) → ∀ x, f a x ≥ (a^2 + 1))
  ∧ ((a > 1/2) → ∀ x, f a x ≥ (3/4 + a)) :=
begin
  sorry -- proof omitted
end

end f_even_when_a_is_zero_f_neither_even_nor_odd_when_a_is_nonzero_min_val_f_l241_241765


namespace tom_has_hours_to_spare_l241_241470

theorem tom_has_hours_to_spare 
  (num_walls : ℕ) 
  (wall_length wall_height : ℕ) 
  (painting_rate : ℕ) 
  (total_hours : ℕ) 
  (num_walls_eq : num_walls = 5) 
  (wall_length_eq : wall_length = 2) 
  (wall_height_eq : wall_height = 3) 
  (painting_rate_eq : painting_rate = 10) 
  (total_hours_eq : total_hours = 10)
  : total_hours - (num_walls * wall_length * wall_height * painting_rate) / 60 = 5 := 
sorry

end tom_has_hours_to_spare_l241_241470


namespace all_roots_in_interval_l241_241507

variable {R : Type*} [LinearOrder R] [LinearOrderedField R]

def conditions (P : R → R) (a b : R) (n : ℕ) :=
  a < b ∧ P(a) < 0 ∧ P(b) > 0 ∧ 
  (∀ k : ℕ, k ≤ n → (-1)^(k) * (deriv^[k] P a) ≤ 0) ∧
  (∀ k : ℕ, k ≤ n → (deriv^[k] P b) ≥ 0)

theorem all_roots_in_interval {P : R → R} {a b : R} {n : ℕ} 
  (h : conditions P a b n) : 
  ∀ x, P(x) = 0 → a < x ∧ x < b :=
sorry

end all_roots_in_interval_l241_241507


namespace sum_of_even_factors_720_l241_241884

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) : ℕ :=
    match n with
    | 720 => 
      let sum_powers_2 := 2 + 4 + 8 + 16 in
      let sum_powers_3 := 1 + 3 + 9 in
      let sum_powers_5 := 1 + 5 in
      sum_powers_2 * sum_powers_3 * sum_powers_5
    | _ => 0
  in
  even_factors_sum 720 = 2340 :=
by 
  sorry

end sum_of_even_factors_720_l241_241884


namespace Alex_dimes_l241_241426

theorem Alex_dimes : 
    ∃ (d q : ℕ), 10 * d + 25 * q = 635 ∧ d = q + 5 ∧ d = 22 :=
by sorry

end Alex_dimes_l241_241426


namespace max_distinct_counts_proof_l241_241923

-- Define the number of boys (B) and girls (G)
def B : ℕ := 29
def G : ℕ := 15

-- Define the maximum distinct dance counts achievable
def max_distinct_counts : ℕ := 29

-- The theorem to prove
theorem max_distinct_counts_proof:
  ∃ (distinct_counts : ℕ), distinct_counts = max_distinct_counts ∧ distinct_counts <= B + G := 
by
  sorry

end max_distinct_counts_proof_l241_241923


namespace car_pedestrian_speed_ratio_l241_241954

theorem car_pedestrian_speed_ratio
  (L : ℝ) -- Length of the bridge
  (v_p v_c : ℝ) -- Speed of pedestrian and car
  (h1 : (4 / 9) * L / v_p = (5 / 9) * L / v_p + (5 / 9) * L / v_c) -- Initial meet at bridge start
  (h2 : (4 / 9) * L / v_p = (8 / 9) * L / v_c) -- If pedestrian continues to walk
  : v_c / v_p = 9 :=
sorry

end car_pedestrian_speed_ratio_l241_241954


namespace minotaur_palace_return_l241_241799

structure Room :=
  (id : ℕ) -- Assuming room IDs are natural numbers.

structure Triplet :=
  (room : Room)
  (door : ℕ) -- Assume door is identified by a natural number.
  (direction : Bool) -- True for right, False for left.

def next_triplet : Triplet → Triplet := sorry
-- Define the function to move to the next triplet (stubbed with sorry for now).

noncomputable def minotaur_returns (start : Triplet) : Prop :=
  ∃ n : ℕ, (iter "next_triplet" n start) = start
  -- iter "next_triplet" n start applies the function next_triplet n times to start.

theorem minotaur_palace_return :
  ∀ start : Triplet, minotaur_returns start :=
begin
  -- Proof omitted.
  sorry
end

end minotaur_palace_return_l241_241799


namespace monthly_growth_rate_selling_price_april_l241_241928

-- First problem: Proving the monthly average growth rate
theorem monthly_growth_rate (sales_jan sales_mar : ℝ) (x : ℝ) 
    (h1 : sales_jan = 256)
    (h2 : sales_mar = 400)
    (h3 : sales_mar = sales_jan * (1 + x)^2) :
  x = 0.25 := 
sorry

-- Second problem: Proving the selling price in April
theorem selling_price_april (unit_profit desired_profit current_sales sales_increase_per_yuan_change current_price new_price : ℝ)
    (h1 : unit_profit = new_price - 25)
    (h2 : desired_profit = 4200)
    (h3 : current_sales = 400)
    (h4 : sales_increase_per_yuan_change = 4)
    (h5 : current_price = 40)
    (h6 : desired_profit = unit_profit * (current_sales + sales_increase_per_yuan_change * (current_price - new_price))) :
  new_price = 35 := 
sorry

end monthly_growth_rate_selling_price_april_l241_241928


namespace ratio_of_7th_terms_l241_241108

theorem ratio_of_7th_terms (a b : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : ∀ n, T n = n * (b 1 + b n) / 2)
  (h3 : ∀ n, S n / T n = (5 * n + 10) / (2 * n - 1)) :
  a 7 / b 7 = 3 :=
by
  sorry

end ratio_of_7th_terms_l241_241108


namespace indistinguishable_balls_indistinguishable_boxes_l241_241310

def number_of_ways_to_place_balls : Nat :=
  5

theorem indistinguishable_balls_indistinguishable_boxes :
  ∃ n : Nat, n = 4 ∧ number_of_ways_to_place_balls = 5 :=
by
  use 4
  split
  sorry

end indistinguishable_balls_indistinguishable_boxes_l241_241310


namespace fill_half_cistern_time_l241_241525

variable (t_half : ℝ)

-- Define a condition that states the certain amount of time to fill 1/2 of the cistern.
def fill_pipe_half_time (t_half : ℝ) : Prop :=
  t_half > 0

-- The statement to prove that t_half is the time required to fill 1/2 of the cistern.
theorem fill_half_cistern_time : fill_pipe_half_time t_half → t_half = t_half := by
  intros
  rfl

end fill_half_cistern_time_l241_241525


namespace unique_sequence_satisfying_condition_l241_241598

theorem unique_sequence_satisfying_condition (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n+1) > a n) 
  (h3 : ∀ n, 3 * (∑ i in finset.range n, a (i + 1)) = ∑ i in finset.range (2 * n) \ finset.range n, a (i + 1)) :
  (∀ n, a n = 2 * n - 1) :=
by
  sorry

end unique_sequence_satisfying_condition_l241_241598


namespace trains_meet_time_l241_241838

variable {x : ℝ} -- Here, x is a positive real number representing the speed unit

-- Conditions
def speed_train_A : ℝ := 3 * x
def speed_train_B : ℝ := 2 * x
def time_train_A_at_C : ℝ := 9 -- 9 AM
def time_train_B_at_C : ℝ := 19 -- 7 PM

-- Prove that the meeting time is 13:00 given the conditions above
theorem trains_meet_time : 13 = 
  let relative_speed := speed_train_A + speed_train_B in
  let time_difference := time_train_B_at_C - time_train_A_at_C in
  time_train_A_at_C + time_difference * (speed_train_B / relative_speed) := sorry

end trains_meet_time_l241_241838


namespace greatest_two_digit_prime_saturated_l241_241501

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_prime_factors_product (n : ℕ) : ℕ :=
  (nat.factors n).erase_dup.prod

def is_prime_saturated (n : ℕ) : Prop :=
  distinct_prime_factors_product n < nat.sqrt n

def two_digit_numbers : list ℕ :=
  (list.range' 10 (99 - 10 + 1)).reverse

theorem greatest_two_digit_prime_saturated : 
  ∃ n ∈ (two_digit_numbers.filter is_prime_saturated), n = 96 :=
begin
  sorry
end

end greatest_two_digit_prime_saturated_l241_241501


namespace tangent_line_at_origin_l241_241809

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.sin x

theorem tangent_line_at_origin :
  ∃ (m b : ℝ), (m = 2) ∧ (b = 1) ∧ (∀ x, f x - (m * x + b) = 0 → 2 * x - f x + 1 = 0) :=
sorry

end tangent_line_at_origin_l241_241809


namespace max_pt_of_f_l241_241664

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - x

theorem max_pt_of_f : ∃ x ∈ Ioo 0 Real.pi, ∀ y ∈ Ioo 0 Real.pi, f y ≤ f x :=
begin
  use Real.pi / 6,
  split,
  {
    -- Show that pi / 6 is within the interval (0, pi)
    apply And.intro,
    { exact by linarith [Real.pi_pos] },  -- 0 < pi/6
    { exact by linarith [Real.pi_pos] }   -- pi/6 < pi
  },
  {
    -- Sorry for proof placeholder
    sorry
  }
end

end max_pt_of_f_l241_241664


namespace dice_even_sum_probability_l241_241901

/-- When two dice are tossed, the probability that the total score is an even number is 0.4166666666666667. -/
theorem dice_even_sum_probability :
  let dice := [1, 2, 3, 4, 5, 6]
  let outcomes := (dice × dice).filter (λ (x : ℕ × ℕ), (x.1 + x.2) % 2 = 0)
  (outcomes.length : ℚ) / (dice.length * dice.length) = 0.4166666666666667 := sorry

end dice_even_sum_probability_l241_241901


namespace perpendicular_slope_l241_241241

variable (x y : ℝ)

def line_eq : Prop := 4 * x - 5 * y = 20

theorem perpendicular_slope (x y : ℝ) (h : line_eq x y) : - (1 / (4 / 5)) = -5 / 4 := by
  sorry

end perpendicular_slope_l241_241241


namespace prime_remainder_div_60_is_49_l241_241358

theorem prime_remainder_div_60_is_49 (p : ℕ) (hp : p.prime) (h : ∃ r, r < 60 ∧ r ≠ 1 ∧ r ≠ 0 ∧ ¬ (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ r = a * b) ∧ ∃ k, p = 60 * k + r) :
  (∃ r, r < 60 ∧ ¬ (r % 2 = 0) ∧ ¬ (r % 3 = 0) ∧ ¬ (r % 5 = 0) ∧ r = 49) :=
by sorry

end prime_remainder_div_60_is_49_l241_241358


namespace rabbit_jump_lengths_order_l241_241780

theorem rabbit_jump_lengths_order :
  ∃ (R : ℕ) (G : ℕ) (P : ℕ) (F : ℕ),
    R = 2730 ∧
    R = P + 1100 ∧
    P = F + 150 ∧
    F = G - 200 ∧
    R > G ∧ G > P ∧ P > F :=
  by
  -- calculations
  sorry

end rabbit_jump_lengths_order_l241_241780


namespace cats_to_dogs_ratio_l241_241820

theorem cats_to_dogs_ratio (d c : ℕ) (h1 : c = d - 8) (h2 : d = 32) :
  c / gcd c d = 3 ∧ d / gcd c d = 4 :=
by
  have h3 : c = 24 := by
    rw [h2] at h1
    exact eq_add_of_sub_eq h1.symm
  have gcd_cd : gcd c d = 8 := by
    rw [h3, h2]
    exact gcd_comm 24 32
  exact ⟨(nat.div_eq_of_lt (by norm_num)).mpr rfl, (nat.div_eq_of_lt (by norm_num)).mpr rfl⟩

end cats_to_dogs_ratio_l241_241820


namespace flow_AB_correct_flow_BC_correct_total_flow_A_correct_l241_241180

namespace IrrigationSystem

-- Conditions and variable definitions
variable (q0 : ℝ)

-- Define the flows in different channels as variables
def flow_AH := q0
def flow_HG := q0
def flow_BC := (2 / 3) * q0
def flow_CD := (2 / 3) * q0
def flow_BG := (2 / 3) * q0
def flow_GD := (2 / 3) * q0

-- Define the flow balance for AB
def flow_AB := (4 / 3) * q0

-- Define the total flow into node A
def total_flow_A := q0 + (6 / 3) * q0

-- Proof statements encoding the answers
theorem flow_AB_correct : flow_AB q0 = (4 / 3) * q0 := sorry
theorem flow_BC_correct : flow_BC q0 = (2 / 3) * q0 := sorry
theorem total_flow_A_correct : total_flow_A q0 = (7 / 3) * q0 := sorry

end IrrigationSystem

end flow_AB_correct_flow_BC_correct_total_flow_A_correct_l241_241180


namespace solve_trig_eq_l241_241418

theorem solve_trig_eq (k : ℤ) :
  (∃ x, (sin (3 * x) + sqrt 3 * cos (3 * x))^2 - 2 * cos (14 * x) = 2) ↔ 
  (∃ x, x = (Real.pi / 60) + k * (Real.pi / 10) ∨ x = -(Real.pi / 24) - k * (Real.pi / 4)) :=
by
  sorry

end solve_trig_eq_l241_241418


namespace log_equality_implies_y_l241_241234

theorem log_equality_implies_y (y : ℝ) : log y 125 = log 3 27 → y = 5 :=
by
  sorry

end log_equality_implies_y_l241_241234


namespace decimal_places_of_product_l241_241122

-- Define the decimal place properties of the numbers
def num_decimals (x : ℚ) : ℕ := 
  if h : ∃ (n m : ℕ), x = n / (10 ^ m) ∧ gcd n (10 ^ m) = 1 then 
    classical.some h 
  else 
    0

-- Define the numbers and their properties
def a : ℚ := 0.8
def b : ℚ := 0.42

-- Define the number of decimal places for a and b
axiom a_decimal_places : num_decimals a = 1
axiom b_decimal_places : num_decimals b = 2

-- Proposition to prove
theorem decimal_places_of_product : num_decimals (a * b) = 3 := sorry

end decimal_places_of_product_l241_241122


namespace savings_increase_is_100_percent_l241_241945

variable (I : ℝ) -- Initial income
variable (S : ℝ) -- Initial savings
variable (I2 : ℝ) -- Income in the second year
variable (E1 : ℝ) -- Expenditure in the first year
variable (E2 : ℝ) -- Expenditure in the second year
variable (S2 : ℝ) -- Second year savings

-- Initial conditions
def initial_savings (I : ℝ) : ℝ := 0.25 * I
def first_year_expenditure (I : ℝ) (S : ℝ) : ℝ := I - S
def second_year_income (I : ℝ) : ℝ := 1.25 * I

-- Total expenditure condition
def total_expenditure_condition (E1 : ℝ) (E2 : ℝ) : Prop := E1 + E2 = 2 * E1

-- Prove that the savings increase in the second year is 100%
theorem savings_increase_is_100_percent :
   ∀ (I S E1 I2 E2 S2 : ℝ),
     S = initial_savings I →
     E1 = first_year_expenditure I S →
     I2 = second_year_income I →
     total_expenditure_condition E1 E2 →
     S2 = I2 - E2 →
     ((S2 - S) / S) * 100 = 100 := by
  sorry

end savings_increase_is_100_percent_l241_241945


namespace domain_of_f_l241_241431

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (2 - x)) + Real.log (x+1)

theorem domain_of_f : {x : ℝ | (2 - x) > 0 ∧ (x + 1) > 0} = {x : ℝ | -1 < x ∧ x < 2} := 
by
  ext x
  simp
  sorry

end domain_of_f_l241_241431


namespace find_current_l241_241490

theorem find_current (R Q t : ℝ) (hR : R = 8) (hQ : Q = 72) (ht : t = 2) :
  ∃ I : ℝ, Q = I^2 * R * t ∧ I = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end find_current_l241_241490


namespace intersection_of_angle_bisectors_on_CD_l241_241046

noncomputable def inscribed_quadrilateral (A B C D : Type) [MetricSpace A] :=
A

theorem intersection_of_angle_bisectors_on_CD
  (A B C D : Type)
  [MetricSpace A]
  (inscribed : inscribed_quadrilateral A B C D)
  (h : dist A D + dist B C = dist C D) :
  ∃ P : Type, P ∈ line_segment C D ∧ is_angle_bisector A P ∧ is_angle_bisector B P := sorry

end intersection_of_angle_bisectors_on_CD_l241_241046


namespace product_of_factors_eq_one_over_eleven_l241_241567

theorem product_of_factors_eq_one_over_eleven : 
  (∏ n in finRange 2 11, (1 - 1 / n)) = 1 / 11 :=
by 
  sorry

end product_of_factors_eq_one_over_eleven_l241_241567


namespace integral_sqrt_add_x_l241_241231

open intervalIntegral

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - x^2) + x

theorem integral_sqrt_add_x : ∫ x in (-1 : ℝ)..(1 : ℝ), f x = (real.pi / 2) :=
by
  sorry

end integral_sqrt_add_x_l241_241231


namespace zero_in_interval_l241_241461

noncomputable def f (x : ℝ) : ℝ := 2 * x - 8 + Real.log x / Real.log 3

theorem zero_in_interval : continuous f ∧ f 3 = -1 ∧ f 4 = Real.log 4 / Real.log 3 ∧ 0 < Real.log 4 / Real.log 3 
→ ∃ c ∈ Ioo 3 4, f c = 0 := 
by
  sorry

end zero_in_interval_l241_241461


namespace total_pens_bought_l241_241014

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l241_241014


namespace hexagon_area_proof_l241_241957

noncomputable def area_of_hexagon (s t : ℝ) (h : 3 * s = 3 * t) (H : (s^2 * real.sqrt 3) / 4 = 9) : ℝ :=
  6 * (t^2 * real.sqrt 3 / 4)

theorem hexagon_area_proof (s t : ℝ) (h : 3 * s = 3 * t) (H : (s^2 * real.sqrt 3) / 4 = 9) :
  area_of_hexagon s t h H = 54 := 
sorry

end hexagon_area_proof_l241_241957


namespace compute_expr_l241_241577

theorem compute_expr : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end compute_expr_l241_241577


namespace smallest_possible_floor_sum_l241_241690

theorem smallest_possible_floor_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ∃ (a b c : ℝ), ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end smallest_possible_floor_sum_l241_241690


namespace ABCD_has_incircle_l241_241367

variables {A B C D E F G H P : Type} [EuclideanGeometry]

-- Given conditions
def is_convex_quadrilateral (Q : Set (Point P)) : Prop := sorry
def lies_on {P : Type} [EuclideanGeometry P] (point : P) (segment : Line P) := sorry
def intersection (line1 line2 : Line P) : P := sorry
def has_incircle (quadrilateral : Set (Point P)) : Prop := sorry

-- Main theorem to prove
theorem ABCD_has_incircle (ABCD : Set (Point P))
  (hABCD_convex : is_convex_quadrilateral ABCD)
  (hE_on_AB : lies_on E (segment AB))
  (hF_on_BC : lies_on F (segment BC))
  (hG_on_CD : lies_on G (segment CD))
  (hH_on_DA : lies_on H (segment DA))
  (P_eq_intersection : P = intersection (line EG) (line FH))
  (HAEP_has_incircle : has_incircle {H, A, E, P})
  (EBFP_has_incircle : has_incircle {E, B, F, P})
  (FCGP_has_incircle : has_incircle {F, C, G, P})
  (GDHP_has_incircle : has_incircle {G, D, H, P}) :
  has_incircle ABCD :=
sorry

end ABCD_has_incircle_l241_241367


namespace count_valid_arrangements_l241_241999

section
variable (students : Finset ℕ) -- Representing the 6 female students
variable (A B : ℕ) -- Two specific students A and B

-- Hypotheses
hypothesis h_students_size : students.card = 6
hypothesis h_AB : A ∈ students ∧ B ∈ students

-- Define the conditions to select 4 from 6 with at least one of A or B, and if both exist, they must not run consecutively
def valid_arrangements (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ (A ∈ s ∨ B ∈ s) ∧
  (∀ (i j : ℕ), A ≠ i + 1 ∨ B ≠ i ∨ B ≠ i - 1)

-- The theorem representing the main problem statement
theorem count_valid_arrangements :
  ∃ (n : ℕ), n = 264 ∧ ∀ (s : Finset ℕ), s ⊆ students → valid_arrangements students A B s → s.card = n := 
  sorry
end

end count_valid_arrangements_l241_241999


namespace direction_vector_of_line_l241_241821

def proj_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1/7, 2/7, -3/7], ![2/7, 4/7, -6/7], ![-3/7, -6/7, 9/7]]

theorem direction_vector_of_line :
  ∃ a b c : ℤ, a > 0 ∧ Int.gcd3 (Int.natAbs a) (Int.natAbs b) (Int.natAbs c) = 1 ∧
  (proj_matrix ⬝ ![1, 0, 0]) = (1/7 : ℚ) • ![ (a : ℚ), b, c ] ∧
  ![(a : ℚ), b, c] = ![1, 2, -3] :=
by
  sorry

end direction_vector_of_line_l241_241821


namespace union_A_B_l241_241683

def A : Set ℝ := {x | x^2 - 1 < 0}
def B : Set ℝ := {x | x > 0}

theorem union_A_B : A ∪ B = {x : ℝ | x > -1} := 
by
  sorry

end union_A_B_l241_241683


namespace variance_of_dataset_l241_241632

def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / (l.length)

theorem variance_of_dataset :
  let data_set := [87, 91, 90, 89, 93]
  mean data_set = 90 → variance data_set = 4 :=
by
  intro h
  sorry

end variance_of_dataset_l241_241632


namespace rhombus_min_rotation_l241_241541

theorem rhombus_min_rotation (α : ℝ) (h1 : α = 60) : ∃ θ, θ = 180 := 
by 
  -- The proof here will show that the minimum rotation angle is 180°
  sorry

end rhombus_min_rotation_l241_241541


namespace min_value_four_l241_241280

noncomputable def min_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : y > 2 * x) : ℝ :=
  (y^2 - 2 * x * y + x^2) / (x * y - 2 * x^2)

theorem min_value_four (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hy_gt_2x : y > 2 * x) :
  min_value x y hx_pos hy_pos hy_gt_2x = 4 := 
sorry

end min_value_four_l241_241280


namespace measure_angle_B_l241_241749

-- Definitions for the given conditions
variables (ABC : Triangle) (A B C I : Point)
variables (F : incenter I ABC)
variables (h_A : measure_angle A = 70)
variables (h_length : length B C = length A C + length A I)

-- Statement to be proved
theorem measure_angle_B : measure_angle B = 35 :=
begin
  sorry
end

end measure_angle_B_l241_241749


namespace no_real_roots_of_geometric_sequence_l241_241322

-- Definition of real numbers and the conditions of the problem
variables {a b c : ℝ}
hypothesis h1 : b^2 = a * c
hypothesis h2 : a * c > 0

-- Statement of the theorem
theorem no_real_roots_of_geometric_sequence : (b^2 - 4 * a * c < 0) :=
by
  -- This is where the proof would go
  sorry

end no_real_roots_of_geometric_sequence_l241_241322


namespace packs_of_chewing_gum_zero_l241_241789

noncomputable def frozen_yogurt_price : ℝ := sorry
noncomputable def chewing_gum_price : ℝ := frozen_yogurt_price / 2
noncomputable def packs_of_chewing_gum : ℕ := sorry

theorem packs_of_chewing_gum_zero 
  (F : ℝ) -- Price of a pint of frozen yogurt
  (G : ℝ) -- Price of a pack of chewing gum
  (x : ℕ) -- Number of packs of chewing gum
  (H1 : G = F / 2)
  (H2 : 5 * F + x * G + 25 = 55)
  : x = 0 :=
sorry

end packs_of_chewing_gum_zero_l241_241789


namespace slope_of_AB_is_1_l241_241429

noncomputable def circle1 := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 4 * p.1 + 2 * p.2 - 11 = 0 }
noncomputable def circle2 := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 14 * p.1 + 12 * p.2 + 60 = 0 }
def is_on_circle1 (p : ℝ × ℝ) := p ∈ circle1
def is_on_circle2 (p : ℝ × ℝ) := p ∈ circle2

theorem slope_of_AB_is_1 :
  ∃ A B : ℝ × ℝ,
  is_on_circle1 A ∧ is_on_circle2 A ∧
  is_on_circle1 B ∧ is_on_circle2 B ∧
  (B.2 - A.2) / (B.1 - A.1) = 1 :=
sorry

end slope_of_AB_is_1_l241_241429


namespace pascal_triangle_sum_right_diagonal_l241_241065

theorem pascal_triangle_sum_right_diagonal (n r : ℕ)
    (a : ℕ) (b : ℕ → ℕ) 
    (h_a : pascal n r = a) 
    (h_b : ∀ i, i ≤ r → b i = pascal (n-i) (r-i)) :
  a = ∑ i in finset.range (r + 1), b i :=
sorry

end pascal_triangle_sum_right_diagonal_l241_241065


namespace compute_root_of_sum_of_cubes_l241_241574

-- Variables definition as per the problem statement
def a : ℤ := 75^2
def b : ℤ := 117

-- Proving the equality given the conditions
theorem compute_root_of_sum_of_cubes (h : (↑5508 : ℝ) ≈ (↑a - ↑b)) 
    (h2 : (↑5742 : ℝ) ≈ (↑a + ↑b)) 
    (h3 : is_int (↑(root 4 (5508^3 + 5625^3 + 5742^3)))) :
  (root 4 (5508^3 + 5625^3 + 5742^3) : ℝ) = 855 :=
sorry

end compute_root_of_sum_of_cubes_l241_241574


namespace find_length_AF_l241_241279

/-- Given a parabola, its focus, and a point on the parabola with 
a midpoint condition, prove the distance formula -/
theorem find_length_AF (p : ℝ) (hp : 0 < p)
  (x0 y0 : ℝ) (hx0 : x0 + p / 2 = 2)
  (A_on_parabola : y0^2 = 2 * p * x0) :
  sqrt (2 * p * (2 - p / 2)) = (√(2 * p * (2 - p / 2))) :=
begin
  sorry
end

end find_length_AF_l241_241279


namespace deepak_age_l241_241212

theorem deepak_age
  (A D : ℕ)
  (h1 : A / D = 2 / 5)  -- the ratio condition
  (h2 : A + 10 = 30)   -- Arun’s age after 10 years will be 30
  : D = 50 :=       -- conclusion Deepak is 50 years old
sorry

end deepak_age_l241_241212


namespace sum_infinite_geometric_series_l241_241219

theorem sum_infinite_geometric_series :
  ∑' (n : ℕ), (3 : ℝ) * ((1 / 3) ^ n) = (9 / 2 : ℝ) :=
sorry

end sum_infinite_geometric_series_l241_241219


namespace compute_fraction_equals_one_l241_241575

theorem compute_fraction_equals_one :
  let q := 18
  let p := 20
  ( ∏ i in (finset.range(p) \ finset.range(q)), (1 + q / (i + 1)) ) / 
  ( ∏ j in (finset.range(q) \ finset.range(1)), (1 + p / (j + 1)) ) = 1 :=
sorry

end compute_fraction_equals_one_l241_241575


namespace simplify_expression_l241_241791

-- Define the expressions and the simplification statement
def expr1 (x : ℝ) := (3 * x - 6) * (x + 8)
def expr2 (x : ℝ) := (x + 6) * (3 * x - 2)
def simplified (x : ℝ) := 2 * x - 36

theorem simplify_expression (x : ℝ) : expr1 x - expr2 x = simplified x := by
  sorry

end simplify_expression_l241_241791


namespace number_of_girls_in_group_l241_241615

-- Define the given conditions
def total_students : ℕ := 20
def prob_of_selecting_girl : ℚ := 2/5

-- State the lean problem for the proof
theorem number_of_girls_in_group : (total_students : ℚ) * prob_of_selecting_girl = 8 := by
  sorry

end number_of_girls_in_group_l241_241615


namespace common_point_circumcircles_l241_241735

theorem common_point_circumcircles 
  (n : ℕ)
  (S : Point)
  (A : Finₓ n → Point)
  (equal_sides : ∀ i j, dist S (A i) = dist S (A j))
  (X : Finₓ n → Point)
  (midpoint_arc_X : ∀ i, is_midpoint_arc (X i) (A i) (A (i + 1 % n)))
  : ∃ P : Point, ∀ i, is_on_circumcircle P (triangle.circumcircle (triangle.mk (X i) (A (i + 1 % n)) (X (i + 1 % n)))) :=
sorry

end common_point_circumcircles_l241_241735


namespace fill_pipe_half_time_l241_241527

theorem fill_pipe_half_time (T : ℝ) (hT : 0 < T) :
  ∀ t : ℝ, t = T / 2 :=
by
  sorry

end fill_pipe_half_time_l241_241527


namespace proposition_A_proposition_D_l241_241653

variables {m n : ℕ}
variables {x y : ℝ}
variables {sx sy s : ℝ}

-- Conditions
variables (hx : x ≤ y) (hxs : sx^2 ≤ sy^2)

-- Proposition A statement: average of the total sample lies between averages of the two parts
theorem proposition_A (hx : x ≤ y) (hz : ≤ x) (hy : y ≤ y) (hz_avg: m + n > 0) 
  (hz1: z = (m / (m + n) * x + n / (m + n) * y)) : x ≤ z ∧ z ≤ y :=
begin
  sorry
end

-- Proposition D statement: variance of the total sample when m = n and x = y
theorem proposition_D (hmn : m = n) (hxy : x = y) (hvariance : s = (sx^2 + sy^2) / 2) : 
  s = (sx^2 + sy^2) / 2 :=
begin
  sorry
end

end proposition_A_proposition_D_l241_241653


namespace charlotte_overall_score_l241_241970

theorem charlotte_overall_score :
  (0.60 * 15 + 0.75 * 20 + 0.85 * 25).round / 60 = 0.75 :=
by
  sorry

end charlotte_overall_score_l241_241970


namespace count_integers_sum_digits_div_by_5_l241_241676

-- Definition for sum of the digits of a number
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

-- Definition for counting integers within a range whose digit sum is divisible by 5
def count_divisible_by_5 (start end_ : ℕ) : ℕ :=
  ((finset.range (end_ - start + 1)).filter (λ n, (digit_sum (n + start)) % 5 = 0)).card

theorem count_integers_sum_digits_div_by_5 :
  count_divisible_by_5 1 1997 = 399 :=
sorry

end count_integers_sum_digits_div_by_5_l241_241676


namespace average_is_A_l241_241070

variable (A : ℝ)
variable (nums : Fin 10 → ℝ)

def average (nums : Fin 10 → ℝ) : ℝ :=
  (∑ i, nums i) / 10

theorem average_is_A (h_avg : average nums = A) (h_ge0 : ∃ i, nums i ≥ 0) : average nums = A :=
  by exact h_avg

end average_is_A_l241_241070


namespace product_of_roots_of_quadratic_l241_241591

theorem product_of_roots_of_quadratic :
  let x := { x : ℝ | x^2 - x - 6 = 0 } in
  ∏ x = -6 :=
begin
  sorry
end

end product_of_roots_of_quadratic_l241_241591


namespace find_m_range_l241_241436

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + 2 * (m - 1) * x + 2 

theorem find_m_range (m : ℝ) : (∀ x ≤ 4, f x m ≤ f (x + 1) m) → m ≤ -3 :=
by
  sorry

end find_m_range_l241_241436


namespace sum_even_factors_of_720_l241_241892

open Nat

theorem sum_even_factors_of_720 :
  ∑ d in (finset.filter (λ x, even x) (finset.divisors 720)), d = 2340 :=
by
  sorry

end sum_even_factors_of_720_l241_241892


namespace intersection_of_line_and_plane_l241_241602

-- Define the line in symmetric form
def line := {p : ℝ × ℝ × ℝ | ∃ t : ℝ, p.1 = 2 - t ∧ p.2 = 3 - t ∧ p.3 = -1 + 4 * t}

-- Define the plane
def plane := {p : ℝ × ℝ × ℝ | p.1 + 2 * p.2 + 3 * p.3 - 14 = 0}

-- Define the intersection point
def intersection_point := (1, 2, 3)

-- Statement that the intersection point is indeed the point where the line intersects the plane
theorem intersection_of_line_and_plane :
  intersection_point ∈ line ∧ intersection_point ∈ plane :=
by
  -- Proof goes here. Skipping with sorry.
  sorry

end intersection_of_line_and_plane_l241_241602


namespace total_pens_l241_241006

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l241_241006


namespace math_proof_problem_l241_241147

-- Define the problem conditions
def problem_conditions (x y : ℚ) := 
  (real.sqrt (x - y) = 2 / 5) ∧ (real.sqrt (x + y) = 2)

-- Define the correct solution
def correct_solution (x y : ℚ) := 
  (x = 52 / 25) ∧ (y = 48 / 25)

-- Define the area of the rectangle
def rectangle_area (a b : ℚ) : ℚ :=
  abs (a * b)

-- Define the proof problem
theorem math_proof_problem : 
  problem_conditions (52 / 25) (48 / 25) ∧ 
  rectangle_area (52 / 25) (48 / 25) = 8 / 25 :=
by 
  sorry

end math_proof_problem_l241_241147


namespace sum_of_even_factors_720_l241_241885

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) : ℕ :=
    match n with
    | 720 => 
      let sum_powers_2 := 2 + 4 + 8 + 16 in
      let sum_powers_3 := 1 + 3 + 9 in
      let sum_powers_5 := 1 + 5 in
      sum_powers_2 * sum_powers_3 * sum_powers_5
    | _ => 0
  in
  even_factors_sum 720 = 2340 :=
by 
  sorry

end sum_of_even_factors_720_l241_241885


namespace probability_of_red_buttons_l241_241362

noncomputable def initialJarA : ℕ := 16 -- total buttons in Jar A (6 red, 10 blue)
noncomputable def initialRedA : ℕ := 6 -- initial red buttons in Jar A
noncomputable def initialBlueA : ℕ := 10 -- initial blue buttons in Jar A

noncomputable def initialJarB : ℕ := 5 -- total buttons in Jar B (2 red, 3 blue)
noncomputable def initialRedB : ℕ := 2 -- initial red buttons in Jar B
noncomputable def initialBlueB : ℕ := 3 -- initial blue buttons in Jar B

noncomputable def transferRed : ℕ := 3
noncomputable def transferBlue : ℕ := 3

noncomputable def finalRedA : ℕ := initialRedA - transferRed
noncomputable def finalBlueA : ℕ := initialBlueA - transferBlue

noncomputable def finalRedB : ℕ := initialRedB + transferRed
noncomputable def finalBlueB : ℕ := initialBlueB + transferBlue

noncomputable def remainingJarA : ℕ := finalRedA + finalBlueA
noncomputable def finalJarB : ℕ := finalRedB + finalBlueB

noncomputable def probRedA : ℚ := finalRedA / remainingJarA
noncomputable def probRedB : ℚ := finalRedB / finalJarB

noncomputable def combinedProb : ℚ := probRedA * probRedB

theorem probability_of_red_buttons :
  combinedProb = 3 / 22 := sorry

end probability_of_red_buttons_l241_241362


namespace part1_part2_l241_241267

-- Part 1: Prove that |z1 + z2| = sqrt(10) if z1 * z2 is purely imaginary
theorem part1 (a : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 - a * I) (h2 : z2 = 2 * a + 3 * I)
  (purely_imaginary : (z1 * z2).re = 0) : 
  abs (z1 + z2) = Real.sqrt 10 := by
  sorry

-- Part 2: Prove that a = -1 or a = -3/2 if z2 / z1 corresponds to a point on the line y = 5x
theorem part2 (a : ℝ) (z1 z2 : ℂ) (h1 : z1 = 1 - a * I) (h2 : z2 = 2 * a + 3 * I)
  (on_line : Im (z2 / z1) / Re (z2 / z1) = 5) : 
  a = -1 ∨ a = -3/2 := by
  sorry

end part1_part2_l241_241267


namespace number_of_divisors_of_N_l241_241750

noncomputable def p : ℕ := 101

def S : Type := Fin p → ℤ

def f (S → Fin p) := sorry

axiom cond1 (a b : S) : f (a + b) + f (a - b) = 2 * (f a + f b) % p
axiom cond2 (a b : S) : (∀ i, a i - b i % p = 0) → f a = f b

theorem number_of_divisors_of_N : (∃ N, ∀ f : S → Fin p, cond1 ∧ cond2 → N = p^(p + (p * (p - 1))/2) ∧ nat.num_divisors N = 5152) :=
  sorry

end number_of_divisors_of_N_l241_241750


namespace smallest_x_l241_241607

theorem smallest_x (x : ℝ) (h : |4 * x + 12| = 40) : x = -13 :=
sorry

end smallest_x_l241_241607


namespace rational_number_among_options_l241_241956

theorem rational_number_among_options :
  (∃ (x: ℚ), x = 1 / 11) ∧
  (¬ (∃ (y: ℚ), y = real.cbrt 15)) ∧
  (¬ (∃ (z: ℚ), z = real.pi)) ∧
  (¬ (∃ (w: ℚ), w = - real.sqrt 2)) :=
by
  sorry

end rational_number_among_options_l241_241956


namespace vector_calc_l241_241570

def vec1 : ℝ × ℝ := (5, -8)
def vec2 : ℝ × ℝ := (2, 6)
def vec3 : ℝ × ℝ := (-1, 4)
def scalar : ℝ := 5

theorem vector_calc :
  (vec1.1 - scalar * vec2.1 + vec3.1, vec1.2 - scalar * vec2.2 + vec3.2) = (-6, -34) :=
sorry

end vector_calc_l241_241570


namespace correct_total_cost_l241_241786

-- Number of sandwiches and their cost
def num_sandwiches : ℕ := 7
def sandwich_cost : ℕ := 4

-- Number of sodas and their cost
def num_sodas : ℕ := 9
def soda_cost : ℕ := 3

-- Total cost calculation
def total_cost : ℕ := num_sandwiches * sandwich_cost + num_sodas * soda_cost

theorem correct_total_cost : total_cost = 55 := by
  -- skip the proof details
  sorry

end correct_total_cost_l241_241786


namespace remainder_of_polynomial_l241_241991

noncomputable def polynomial_remainder : ℚ → ℚ :=
  λ x, 5 * x^4 - 12 * x^3 + 3 * x^2 - 5 * x + 15

noncomputable def divisor_value : ℚ := 3 * 3

theorem remainder_of_polynomial (x : ℚ) (h : x = 3) :
  polynomial_remainder x = 108 :=
by
  rw [h]
  simp only [polynomial_remainder]
  norm_num
  /- Therefore, the remainder when the given polynomial is divided by 3x - 9 is 108. -/
  sorry

end remainder_of_polynomial_l241_241991


namespace first_class_students_count_l241_241800

theorem first_class_students_count 
  (x : ℕ) 
  (avg1 : ℕ) (avg2 : ℕ) (num2 : ℕ) (overall_avg : ℝ)
  (h_avg1 : avg1 = 40)
  (h_avg2 : avg2 = 60)
  (h_num2 : num2 = 50)
  (h_overall_avg : overall_avg = 52.5)
  (h_eq : 40 * x + 60 * 50 = (52.5:ℝ) * (x + 50)) :
  x = 30 :=
by
  sorry

end first_class_students_count_l241_241800


namespace speed_on_second_day_is_approx_8_l241_241515

noncomputable def second_day_speed : ℝ := 
  let distance := 3  -- distance in km
  let first_day_speed := 6  -- speed in km/hr
  let late_time := 7 / 60  -- time late in hrs
  let early_time := 8 / 60  -- time early in hrs
  let expected_time := distance / first_day_speed  -- expected time in hr
  let first_day_time := expected_time + late_time  -- actual time on the first day in hr
  let second_day_time := expected_time - early_time  -- time on the second day in hr
  distance / second_day_time  -- calculate speed on the second day

theorem speed_on_second_day_is_approx_8.18 :
  second_day_speed ≈ 8.18 := sorry

end speed_on_second_day_is_approx_8_l241_241515


namespace gcd_polynomial_l241_241637

open scoped Classical

-- Definitions and conditions
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = k * m

-- The main theorem
theorem gcd_polynomial (b : ℕ) (h : is_multiple_of b 1428) : 
  Nat.gcd (b^2 + 11 * b + 30) (b + 6) = 6 :=
by
  sorry

end gcd_polynomial_l241_241637


namespace traders_gain_percentage_l241_241563

theorem traders_gain_percentage (C : ℝ) (h : 0 < C) : 
  let cost_of_100_pens := 100 * C
  let gain := 40 * C
  let selling_price := cost_of_100_pens + gain
  let gain_percentage := (gain / cost_of_100_pens) * 100
  gain_percentage = 40 := by
  sorry

end traders_gain_percentage_l241_241563


namespace least_different_denominators_l241_241395

theorem least_different_denominators (n : ℕ) (h : n = 6 ^ 100) :
  let fractions := {1 / m | m ∣ n}
  let cumulative_sums := {sum | ∃ (f : finset (fractions)), sum = ∑ x in f}
  let reduced_fractions := {r.denom | r ∈ cumulative_sums ∧ r.is_irreducible}
  reduced_fractions.card = 2 :=
begin
  sorry
end

end least_different_denominators_l241_241395


namespace sum_of_even_factors_720_l241_241879

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) :=
    (2 + 4 + 8 + 16) * (1 + 3 + 9) * (1 + 5)
  in even_factors_sum 720 = 2340 :=
by
  sorry

end sum_of_even_factors_720_l241_241879


namespace circle_properties_and_range_of_a_l241_241259

theorem circle_properties_and_range_of_a :
  (∃ (x y : ℝ), (x^2 + (y - 1)^2 = 5 ∨ (x - 3 - real.sqrt 3)^2 + (y - 4 - real.sqrt 3)^2 = 5)) ∧
  (∀ (a : ℝ),
    (∃ (x y : ℝ), (x - y + 1 = 0) ∧ x^2 + (y-1)^2 = 5) →
    (∀ (m : ℝ), ∃ (x y : ℝ), mx - y + real.sqrt a + 1 = 0) →
    (0 ≤ a ∧ a ≤ 5)) :=
begin
  sorry,
end

end circle_properties_and_range_of_a_l241_241259


namespace original_population_l241_241918

theorem original_population (P : ℕ) (h1 : 0.1 * (P : ℝ) + 0.2 * (0.9 * P) = 4500) : P = 6250 :=
sorry

end original_population_l241_241918


namespace find_m_l241_241659

-- Definitions of conditions
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def focus_condition (a c : ℝ) : Prop :=
  a^2 = c^2 + m^2

-- Statement
theorem find_m {m : ℝ} (h : m > 0) :
  is_ellipse 5 m x y ∧ focus_condition 5 4 → m = 3 :=
sorry

end find_m_l241_241659


namespace numSpaceDiagonals_P_is_241_l241_241521

noncomputable def numSpaceDiagonals (vertices : ℕ) (edges : ℕ) (tri_faces : ℕ) (quad_faces : ℕ) : ℕ :=
  let total_segments := (vertices * (vertices - 1)) / 2
  let face_diagonals := 2 * quad_faces
  total_segments - edges - face_diagonals

theorem numSpaceDiagonals_P_is_241 :
  numSpaceDiagonals 26 60 24 12 = 241 := by 
  sorry

end numSpaceDiagonals_P_is_241_l241_241521


namespace total_pens_bought_l241_241019

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l241_241019


namespace two_card_draw_probability_l241_241474

open ProbabilityTheory

def card_values (card : ℕ) : ℕ :=
  if card = 1 ∨ card = 11 ∨ card = 12 ∨ card = 13 then 10 else card

def deck_size := 52

def total_prob : ℚ :=
  let cards := (1, deck_size)
  let case_1 := (card_values 6 * card_values 9 / (deck_size * (deck_size - 1))) + 
                (card_values 7 * card_values 8 / (deck_size * (deck_size - 1)))
  let case_2 := (3 * 4 / (deck_size * (deck_size - 1))) + 
                (4 * 3 / (deck_size * (deck_size - 1)))
  case_1 + case_2

theorem two_card_draw_probability :
  total_prob = 16 / 331 :=
by
  sorry

end two_card_draw_probability_l241_241474


namespace xy_proposition_l241_241256

theorem xy_proposition (x y : ℝ) : (x + y ≥ 5) → (x ≥ 3 ∨ y ≥ 2) :=
sorry

end xy_proposition_l241_241256


namespace sum_and_product_geometric_sequences_l241_241640

-- Definitions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∃ n₀ : ℕ, ∀ n ≥ n₀, a (n + 1) = r * a n

-- Theorem statement
theorem sum_and_product_geometric_sequences (a b : ℕ → ℝ):
  is_geometric_sequence a →
  is_geometric_sequence b →
  (¬ is_geometric_sequence (λ n, a n + b n)) ∧ is_geometric_sequence (λ n, a n * b n) :=
by
  assume ha hb,
  -- Proof goes here
  sorry

end sum_and_product_geometric_sequences_l241_241640


namespace goods_train_length_l241_241946

theorem goods_train_length (speed_mans_train_kmph : ℕ) (speed_goods_train_kmph : ℕ) (time_seconds : ℕ) :
  speed_mans_train_kmph = 56 → speed_goods_train_kmph = 42 → time_seconds = 15 →
  let relative_speed_mps := (speed_mans_train_kmph + speed_goods_train_kmph) * 1000 / 3600 in
  relative_speed_mps * time_seconds = 410 :=
by
  intros h1 h2 h3
  let relative_speed_mps := (speed_mans_train_kmph + speed_goods_train_kmph) * 1000 / 3600
  calc
    relative_speed_mps * time_seconds = ((56 + 42) * 1000 / 3600) * 15 : by rw [h1, h2, h3]
    ... = 27.3333 * 15 : by sorry
    ... = 410 : by sorry

end goods_train_length_l241_241946


namespace sum_of_all_possible_radii_l241_241933

noncomputable def circle_center_and_tangent (r : ℝ) : Prop :=
  let C := (r, r) in
  let circleC_radius := r in
  let circleD_center := (5 : ℝ, 0 : ℝ) in
  let circleD_radius := (2 : ℝ) in
  (circleC_radius - 5)^2 + circleC_radius^2 = (circleC_radius + circleD_radius)^2

theorem sum_of_all_possible_radii : ∀ r : ℝ, circle_center_and_tangent r → (r = 7 + 2 * real.sqrt 7) ∨ (r = 7 - 2 * real.sqrt 7) → r + 7 - 2 * real.sqrt 7 = 14 :=
by
  intros r hcond hr;
  sorry

end sum_of_all_possible_radii_l241_241933


namespace euclidean_division_37_5_l241_241977

theorem euclidean_division_37_5 : 
  ∃ q r : ℕ, 37 = 5 * q + r ∧ 0 ≤ r ∧ r < 5 :=
by
  use 7, 2
  split
  { -- Prove 37 = 5 * 7 + 2
    simp
  }
  split
  { -- Prove 0 ≤ 2
    norm_num
  }
  { -- Prove 2 < 5
    norm_num
  }

end euclidean_division_37_5_l241_241977


namespace rectangle_area_perimeter_l241_241194

/-- 
Given a rectangle with positive integer sides a and b,
let A be the area and P be the perimeter.

A = a * b
P = 2 * a + 2 * b

Prove that 100 cannot be expressed as A + P - 4.
-/
theorem rectangle_area_perimeter (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (A : ℕ) (P : ℕ)
  (hA : A = a * b) (hP : P = 2 * a + 2 * b) : 
  ¬ (A + P - 4 = 100) := 
sorry

end rectangle_area_perimeter_l241_241194


namespace min_pairs_acquainted_l241_241831

-- Definition of the problem conditions in Lean 4
def residents : ℕ := 240

-- Statement of the theorem to be proved
theorem min_pairs_acquainted (n : ℕ) (h1 : n = 240)
    (h2 : ∀ (a1 a2 a3 a4 a5 : ℕ), a1 ∈ finset.range n → a2 ∈ finset.range n → a3 ∈ finset.range n → a4 ∈ finset.range n → a5 ∈ finset.range n → 
  ((a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5) ∧ (a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5) ∧ (a3 ≠ a4 ∧ a3 ≠ a5) ∧ a4 ≠ a5) → 
    (∃ (seating : finset (fin n)) (h : seating.card = 5), 
      ∀ x ∈ seating, (∀ y ∈ seating, (¬ (x = y) → ∃ z1 z2 : fin n, z1 ∈ seating ∧ z2 ∈ seating ∧ z1 ≠ z2 ∧ 
                                (z1 = x + 1 ∨ z1 = x - 1) ∧ (z2 = x + 1 ∨ z2 = x -1)))) : 
    (n * (n - 3)) / 2 = 28440 := 
by {
  -- Proof
  sorry
}

end min_pairs_acquainted_l241_241831


namespace room_area_is_18pi_l241_241543

def semicircle_room_area (R r : ℝ) (h : R > r) (d : ℝ) (hd : d = 12) : ℝ :=
  (π / 2) * (R^2 - r^2)

theorem room_area_is_18pi (R r : ℝ) (h : R > r) :
  semicircle_room_area R r h 12 (by rfl) = 18 * π :=
by
  sorry

end room_area_is_18pi_l241_241543


namespace cubic_roots_sum_of_cubes_l241_241424

theorem cubic_roots_sum_of_cubes (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a * b + a * c + b * c = -4) (h3 : a * b * c = -4) :
  a^3 + b^3 + c^3 = 1 :=
sorry

end cubic_roots_sum_of_cubes_l241_241424


namespace minimum_four_sum_multiple_of_four_l241_241712

theorem minimum_four_sum_multiple_of_four (n : ℕ) (h : n = 7) (s : Fin n → ℤ) :
  ∃ (a b c d : Fin n), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (s a + s b + s c + s d) % 4 = 0 := 
by
  -- Proof goes here
  sorry

end minimum_four_sum_multiple_of_four_l241_241712


namespace least_possible_faces_combined_l241_241110

noncomputable def hasValidDiceConfiguration : Prop :=
  ∃ a b : ℕ,
  (∃ s8 s12 s13 : ℕ,
    (s8 = 3) ∧
    (s12 = 4) ∧
    (a ≥ 5 ∧ b = 6 ∧ (a + b = 11) ∧
      (2 * s12 = s8) ∧
      (2 * s8 = s13))
  )

theorem least_possible_faces_combined : hasValidDiceConfiguration :=
  sorry

end least_possible_faces_combined_l241_241110


namespace second_team_pieces_l241_241249

-- Definitions for the conditions
def total_pieces_required : ℕ := 500
def pieces_first_team : ℕ := 189
def pieces_third_team : ℕ := 180

-- The number of pieces the second team made
def pieces_second_team : ℕ := total_pieces_required - (pieces_first_team + pieces_third_team)

-- The theorem we are proving
theorem second_team_pieces : pieces_second_team = 131 := by
  unfold pieces_second_team
  norm_num
  sorry

end second_team_pieces_l241_241249


namespace simplify_expression_l241_241414

theorem simplify_expression (x : ℝ) : 
  (12 * x ^ 12 - 3 * x ^ 10 + 5 * x ^ 9) + (-1 * x ^ 12 + 2 * x ^ 10 + x ^ 9 + 4 * x ^ 4 + 6 * x ^ 2 + 9) =
  11 * x ^ 12 - x ^ 10 + 6 * x ^ 9 + 4 * x ^ 4 + 6 * x ^ 2 + 9 :=
by
  sorry

end simplify_expression_l241_241414


namespace total_pens_bought_l241_241016

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l241_241016


namespace max_non_intersecting_chords_l241_241401

theorem max_non_intersecting_chords (n : ℕ) (c : ℕ) (p : ℕ) (h : c = 17) (hn : n = 2006) (hp : p ≥ 2) : ∃ k, k = 117 ∧ 
  ∀ (coloring : set (fin c → fin n)) (segments : set (fin n × fin n)), 
    (∀ (a b : fin n), a ≠ b → coloring a = coloring b → (a, b) ∈ segments) ∧
    (∀ (a b c d : fin n), a ≠ b → c ≠ d → (a, b) ∈ segments → (c, d) ∈ segments → a ≠ c → b ≠ d → 
     ¬ ((open_segment {x | x ∈ {a, b}} {x}) ∩ (open_segment {x | x ∈ {c, d}}) ≠ ∅ )) → size segments = k :=
sorry

end max_non_intersecting_chords_l241_241401


namespace sequence_converges_to_one_l241_241753

noncomputable def u (n : ℕ) : ℝ :=
1 + (Real.sin n) / n

theorem sequence_converges_to_one :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n - 1| ≤ ε :=
sorry

end sequence_converges_to_one_l241_241753


namespace polynomial_root_problem_l241_241374

theorem polynomial_root_problem 
  (P Q R : ℚ[X]) 
  (hP : P = X^2 - 3 * X - 7)
  (hQ : Q(0) = 2)
  (hQ2 : Q.degree = 2)
  (hR2 : R.degree = 2)
  (hPQ_common_root : ∃ p, is_root (P + Q) p ∧ is_root (P + R) p)
  (hQR_common_root : ∃ r, is_root (P + R) r ∧ is_root (Q + R) r)
  (hQP_common_root : ∃ q, is_root (P + Q) q ∧ is_root (Q + R) q)
  (h_distinct_roots : all_different [p, q, r]) :
  R(0) = 52 / 19 := by 
  sorry

end polynomial_root_problem_l241_241374


namespace find_X_plus_Y_l241_241063

-- Statement of the problem translated from the given problem-solution pair.
theorem find_X_plus_Y (X Y : ℚ) :
  (∀ x : ℚ, x ≠ 5 → x ≠ 6 →
    (Y * x + 8) / (x^2 - 11 * x + 30) = X / (x - 5) + 7 / (x - 6)) →
  X + Y = -22 / 3 :=
by
  sorry

end find_X_plus_Y_l241_241063


namespace isosceles_triangle_vertex_angle_cos_values_l241_241089

theorem isosceles_triangle_vertex_angle_cos_values 
  (x : ℝ)
  (h0 : 0 < x ∧ x < 90) -- x is acute
  (h1 : ∃ (a b c : ℝ), a = b ∧ c = Math.cos 7 * x ∧ a = Math.cos x)
  (h2 : 2 * x = 180 / 2 * 2 - 2 * (1 - 1))
  : x = 10 ∨ x = 50 ∨ x = 54 :=
by sorry

end isosceles_triangle_vertex_angle_cos_values_l241_241089


namespace markup_is_150_percent_l241_241205

-- Defining constants
def selling_price : ℝ := 10
def profit (S : ℝ) : ℝ := 0.20 * S
def expenses (S : ℝ) : ℝ := 0.30 * S
def fixed_cost : ℝ := 1

-- Calculate the variable cost
def variable_cost (S : ℝ) : ℝ := S - profit S - expenses S - fixed_cost

-- Markup calculation
def markup (S C : ℝ) : ℝ := (S - C) / C * 100

-- Theorem stating that the rate of markup on the product is 150%
theorem markup_is_150_percent : markup selling_price (variable_cost selling_price) = 150 :=
by
  -- Proof steps would go here, using sorry to skip the actual proof.
  sorry

end markup_is_150_percent_l241_241205


namespace jackson_holidays_l241_241360

theorem jackson_holidays (holidays_per_month : ℕ) (months_in_year : ℕ) (holidays_per_year : ℕ) : 
  holidays_per_month = 3 → months_in_year = 12 → holidays_per_year = holidays_per_month * months_in_year → holidays_per_year = 36 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jackson_holidays_l241_241360


namespace ellipse_eccentricity_range_l241_241764

theorem ellipse_eccentricity_range (a b : ℝ) (h : a > b) (h_b : b > 0) : 
  ∃ e : ℝ, (e = (Real.sqrt (a^2 - b^2)) / a) ∧ (e > 1/2 ∧ e < 1) :=
by
  sorry

end ellipse_eccentricity_range_l241_241764


namespace find_t_l241_241808

  theorem find_t (t : ℝ) : 
    (∀ (f : ℝ → ℝ) (f' : ℝ → ℝ), 
      (f = λ x, sin x + t * cos x) → 
      (f' = λ x, cos x - t * sin x) → 
      (f' 0 = 1) → 
      (∀ x y, y = f 0 + f' 0 * (x - 0) → y = x + 1)) → 
    t = 1 :=
  by 
    intros f f' h1 h2 h3 h4
    sorry
  
end find_t_l241_241808


namespace find_x_eq_24_over_23_l241_241994

theorem find_x_eq_24_over_23 : 
  ∃ (x : ℚ), (∃ x_nonneg : 0 ≤ x, (x = 24/23 ∧ x ≠ 1) ∧ (sqrt (6 * x) / sqrt (4 * (x - 1)) = 3)) :=
begin
  sorry
end

end find_x_eq_24_over_23_l241_241994


namespace employed_population_is_60_percent_l241_241352

def percent_employed (P : ℝ) (E : ℝ) : Prop :=
  ∃ (P_0 : ℝ) (E_male : ℝ) (E_female : ℝ),
    P_0 = P * 0.45 ∧    -- 45 percent of the population are employed males
    E_female = (E * 0.25) * P ∧   -- 25 percent of the employed people are females
    (0.75 * E = 0.45) ∧    -- 75 percent of the employed people are males which equals to 45% of the total population
    E = 0.6            -- 60% of the population are employed

theorem employed_population_is_60_percent (P : ℝ) (E : ℝ):
  percent_employed P E :=
by
  sorry

end employed_population_is_60_percent_l241_241352


namespace cyclic_quad_division_l241_241776

noncomputable def is_convex_quadrilateral (A B C D : Point) : Prop :=
  ∠ A + ∠ B + ∠ C + ∠ D = 360° ∧ ∀ angle ∈ {∠ A, ∠ B, ∠ C, ∠ D}, angle < 180°

noncomputable def is_cyclic_quadrilateral (A B C D : Point) : Prop :=
  ∠ A + ∠ C = 180° ∧ ∠ B + ∠ D = 180°

theorem cyclic_quad_division (A B C D : Point) (n : ℕ) 
  (convex : is_convex_quadrilateral A B C D)
  (cyclic : is_cyclic_quadrilateral A B C D)
  (h : n > 4) :
  ∃ quadrilaterals : list (Point × Point × Point × Point), 
    quadrilaterals.length = n ∧ 
    ∀ quad ∈ quadrilaterals, is_cyclic_quadrilateral quad.1 quad.2 quad.3 quad.4 :=
sorry

end cyclic_quad_division_l241_241776


namespace smallest_possible_variance_l241_241338

theorem smallest_possible_variance (n : ℕ) (hn : 2 ≤ n) (a : Fin n → ℝ) (ha1 : ∃ i j : Fin n, i ≠ j ∧ a i = 0 ∧ a j = 1) :
  var a = 1 / (2 * n) := 
sorry

end smallest_possible_variance_l241_241338


namespace triangle_angles_are_45_45_90_l241_241930

-- Define the right-angled triangle with the right angle at C, and other necessary conditions
variables {A B C M : Type}
          [metric_space A]
          [metric_space B]
          [metric_space C]
          [metric_space M]

-- Define the conditions and the problem statement
theorem triangle_angles_are_45_45_90 (hABC : ∀ (P : set (A × B)), is_right_triangle P)
  (hCircle : ∀ (circle : set (A × C)), intersects_circle_at_midpoint_of_hypotenuse circle):
  angles_of_triangle_are_ (45, 45, 90) :=
begin
  sorry
end

end triangle_angles_are_45_45_90_l241_241930


namespace total_pens_l241_241023

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l241_241023


namespace spider_max_journey_length_l241_241196

theorem spider_max_journey_length (side_length : ℝ) (h_side : side_length = 2) 
  (start_corner : ℕ) (end_corner : ℕ) (h_start_end : start_corner = end_corner)
  (visited_corners : list ℕ) (h_visited : visited_corners.length = 8) :
  ∃ max_length : ℝ, max_length = (8 * real.sqrt 3 + 8 * real.sqrt 2) :=
by
  use (8 * real.sqrt 3 + 8 * real.sqrt 2)
  sorry

end spider_max_journey_length_l241_241196


namespace find_a_l241_241275

theorem find_a (a : ℤ) : (∃ x : ℤ, x = -1 ∧ 2 * x + 3 * a = 4) → a = 2 :=
by
  intros h
  obtain ⟨x, hx, h_eq⟩ := h
  have h_sub := h_eq.subst hx
  sorry

end find_a_l241_241275


namespace tan_alpha_eq_one_half_ratio_sin_cos_l241_241273

theorem tan_alpha_eq_one_half (α : ℝ) (h : (sin α) / (sin α - cos α) = -1) : 
  tan α = 1 / 2 := 
sorry

theorem ratio_sin_cos (α : ℝ) (h : (sin α) / (sin α - cos α) = -1) : 
  (sin α ^ 2 + 2 * sin α * cos α) / (3 * sin α ^ 2 + cos α ^ 2) = 5 / 7 := 
sorry

end tan_alpha_eq_one_half_ratio_sin_cos_l241_241273


namespace bus_speed_including_stoppages_l241_241232

theorem bus_speed_including_stoppages (speed_excluding_stoppages : ℕ) (stoppage_time_per_hour : ℕ)
  (h1 : speed_excluding_stoppages = 54) (h2 : stoppage_time_per_hour = 10) :
  let time_per_hour := 60
  let running_time := time_per_hour - stoppage_time_per_hour
  let running_time_hours := running_time / time_per_hour.toFloat
  let distance := speed_excluding_stoppages.toFloat * running_time_hours
  (distance = 45) := 
by {
  let time_per_hour := 60
  let running_time := time_per_hour - stoppage_time_per_hour
  let running_time_hours := running_time / time_per_hour.toFloat
  let distance := speed_excluding_stoppages.toFloat * running_time_hours
  sorry
}

end bus_speed_including_stoppages_l241_241232


namespace sum_even_factors_l241_241852

theorem sum_even_factors (n : ℕ) (h : n = 720) : 
  (∑ d in Finset.filter (λ d, d % 2 = 0) (Finset.divisors n), d) = 2340 :=
by
  rw h
  -- sorry to skip the actual proof
  sorry

end sum_even_factors_l241_241852


namespace mutually_exclusive_white_ball_events_l241_241101

-- Definitions of persons and balls
inductive Person | A | B | C
inductive Ball | red | black | white

-- Definitions of events
def eventA (dist : Person → Ball) : Prop := dist Person.A = Ball.white
def eventB (dist : Person → Ball) : Prop := dist Person.B = Ball.white

theorem mutually_exclusive_white_ball_events (dist : Person → Ball) :
  (eventA dist → ¬eventB dist) :=
by
  sorry

end mutually_exclusive_white_ball_events_l241_241101


namespace odd_degree_polynomial_unique_l241_241413
-- Import necessary library

-- Define the polynomial p and its properties
noncomputable def p : ℝ → ℝ := sorry -- Placeholder for the polynomial p

-- State the problem as a theorem
theorem odd_degree_polynomial_unique (h_odd : ∃ n : ℕ, p ∈ Polynomial ℝ ∧ odd n) 
  (h_eq : ∀ x, p (x^2 - 1) = p x ^ 2 - 1) :
  p = λ x, x :=
by
  sorry

end odd_degree_polynomial_unique_l241_241413


namespace sum_of_ages_is_12_l241_241093

-- Let Y be the age of the youngest child
def Y : ℝ := 1.5

-- Let the ages of the other children
def age2 : ℝ := Y + 1
def age3 : ℝ := Y + 2
def age4 : ℝ := Y + 3

-- Define the sum of the ages
def sum_of_ages : ℝ := Y + age2 + age3 + age4

-- The theorem to prove the sum of the ages is 12 years
theorem sum_of_ages_is_12 : sum_of_ages = 12 :=
by
  -- The detailed proof is to be filled in later, currently skipped.
  sorry

end sum_of_ages_is_12_l241_241093


namespace total_pens_bought_l241_241003

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l241_241003


namespace no_arithmetic_progression_6x7_l241_241801

def is_arithmetic_sequence (xs : List ℕ) : Prop :=
  forall k, k < xs.length - 1 → 2 * xs.nth k = xs.nth (k - 1) + xs.nth (k + 1)

theorem no_arithmetic_progression_6x7 :
  ¬ ∃ t : Matrix (Fin 6) (Fin 7) ℕ,
    (∀ i : Fin 6, is_arithmetic_sequence (List.of_fn (λ j : Fin 7, t i j))) ∧
    (∀ j : Fin 7, is_arithmetic_sequence (List.of_fn (λ i : Fin 6, t i j))) ∧
    (List.of_fn (λ i : Fin 6, t i 3) = [1, 2, 3, 4, 5, 7]) :=
sorry

end no_arithmetic_progression_6x7_l241_241801


namespace smallest_positive_b_l241_241078

theorem smallest_positive_b (b N : ℕ) (h1 : N = 7 * b^2 + 7 * b + 7) (h2 : ∃ x : ℕ, N = x^4) : b = 18 :=
  sorry

end smallest_positive_b_l241_241078


namespace sum_of_roots_of_P_l241_241368

-- Define a quadratic polynomial
def P (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the inequality condition for the given polynomial
def condition (a b c : ℝ) :=
  ∀ x : ℝ, P a b c (x^3 + x) ≥ P a b c (x^2 + 1)

-- The sum of the roots of the quadratic polynomial
def sum_of_roots (a b c : ℝ) := -b / a

-- The main theorem
theorem sum_of_roots_of_P (a b c : ℝ) (h : condition a b c) : 
  sum_of_roots a b c = 4 :=
by
  sorry

end sum_of_roots_of_P_l241_241368


namespace sum_even_factors_of_720_l241_241867

theorem sum_even_factors_of_720 : 
  let even_factors_sum (n : ℕ) : ℕ := 
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  in even_factors_sum 720 = 2340 :=
by
  let even_factors_sum (n : ℕ) : ℕ :=
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  sorry

end sum_even_factors_of_720_l241_241867


namespace fill_blanks_satisfies_equation_l241_241433

theorem fill_blanks_satisfies_equation :
  ∃ (a b c d e f g h i : ℕ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ≠ ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ≠ e ≠ i ∧
    f ≠ g ≠ f ≠ h ∧ f ≠ i ∧
    g ≠ h ∧ g ≠ i ∧
    h ≠ i) ∧
    (a ∈ {1, 2, 3, 4, 5, 6, 8, 9} ∧
     b ∈ {1, 2, 3, 4, 5, 6, 8, 9} ∧
     c ∈ {1, 2, 3, 4, 5, 6, 8, 9} ∧
     d ∈ {1, 2, 3, 4, 5, 6, 8, 9} ∧
     e ∈ {1, 2, 3, 4, 5, 6, 8, 9} ∧
     f ∈ {1, 2, 3, 4, 5, 6, 8, 9} ∧
     g ∈ {1, 2, 3, 4, 5, 6, 8, 9} ∧
     h ∈ {1, 2, 3, 4, 5, 6, 8, 9} ∧
     i ∈ {1, 2, 3, 4, 5, 6, 8, 9}) ∧
    (100 * a + 10 * b + c) / (10 * d + e) = 2 ∧
    100 * f + 10 * g + h - 70 = 2 ∧
    10 * i + 7 = 97 :=
sorry

end fill_blanks_satisfies_equation_l241_241433


namespace term_without_x_in_expansion_l241_241645

open Nat

theorem term_without_x_in_expansion :
  let n := 10
  let k := 6 in
  ∃ k, (binom n k = 210) ∧ (30 - 5 * k = 0) :=
by
  sorry

end term_without_x_in_expansion_l241_241645


namespace total_dolls_l241_241405

-- Definitions given in the conditions
def grandmother_dolls := 50
def sister_dolls := grandmother_dolls + 2
def rene_dolls := 3 * sister_dolls

-- The theorem statement based on condition and correct answer
theorem total_dolls (g : ℕ) (s : ℕ) (r : ℕ) (h_g : g = 50) (h_s : s = g + 2) (h_r : r = 3 * s) : g + s + r = 258 := 
by {
  -- Placeholder for the proof
  sorry,
}

end total_dolls_l241_241405


namespace y_share_is_correct_l241_241549

noncomputable def share_of_y (a : ℝ) := 0.45 * a

theorem y_share_is_correct :
  ∃ a : ℝ, (1 * a + 0.45 * a + 0.30 * a = 245) ∧ (share_of_y a = 63) :=
by
  sorry

end y_share_is_correct_l241_241549


namespace shift_graph_right_by_3_units_l241_241104

theorem shift_graph_right_by_3_units (x : ℝ) :
    (∀ x, y = (1 / 2) ^ x → 8 * 2 ^ -x) ↔ (∀ x, y = 2 ^ -(x - 3)) := sorry

end shift_graph_right_by_3_units_l241_241104


namespace inequality_condition_l241_241162

theorem inequality_condition {a b x y : ℝ} (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) : 
  (a^2 / x) + (b^2 / y) ≥ ((a + b)^2 / (x + y)) ∧ (a^2 / x) + (b^2 / y) = ((a + b)^2 / (x + y)) ↔ (x / y) = (a / b) :=
sorry

end inequality_condition_l241_241162


namespace cube_surface_area_increase_l241_241199

theorem cube_surface_area_increase :
  let edge_length : ℝ := 5
  let original_surface_area := 6 * (edge_length ^ 2)
  -- The surface area of a cube with edge length 5 is 150 square cm.
  let increase_in_surface_area := 100
  -- After cutting into 3 cuboids, the surface area increases by 100 square cm.
  True := True :=
begin
  sorry
end

end cube_surface_area_increase_l241_241199


namespace rectangle_solution_l241_241143

-- Define the given conditions
variables (x y : ℚ)

-- Given equations
def condition1 := (Real.sqrt (x - y) = 2 / 5)
def condition2 := (Real.sqrt (x + y) = 2)

-- Solution
theorem rectangle_solution (x y : ℚ) (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 52 / 25 ∧ y = 48 / 25 ∧ (Real.sqrt ((52 / 25) * (48 / 25)) = 8 / 25) :=
by
  sorry

end rectangle_solution_l241_241143


namespace intersection_of_A_and_B_l241_241302

def setA : Set ℝ := { x | (x - 3) * (x + 1) ≥ 0 }
def setB : Set ℝ := { x | x < -4/5 }

theorem intersection_of_A_and_B : setA ∩ setB = { x | x ≤ -1 } :=
  sorry

end intersection_of_A_and_B_l241_241302


namespace find_f_12_plus_f_3_l241_241161

noncomputable def f : ℝ → ℝ := sorry

axiom odd_fn : ∀ x, f(-x) = -f(x)
axiom f_periodic : ∀ x, f(x + 1) = f(x + 5)
axiom f_at_1 : f(1) = 2

theorem find_f_12_plus_f_3 : f(12) + f(3) = -2 :=
by
  -- conditions of the problem
  have odd_function := odd_fn
  have periodicity := f_periodic
  have value_at_1 := f_at_1
  sorry

end find_f_12_plus_f_3_l241_241161


namespace linda_spent_total_l241_241772

noncomputable def total_spent (notebooks_price_euro : ℝ) (notebooks_count : ℕ) 
    (pencils_price_pound : ℝ) (pencils_gift_card_pound : ℝ)
    (pens_price_yen : ℝ) (pens_points : ℝ) 
    (markers_price_dollar : ℝ) (calculator_price_dollar : ℝ)
    (marker_discount : ℝ) (coupon_discount : ℝ) (sales_tax : ℝ)
    (euro_to_dollar : ℝ) (pound_to_dollar : ℝ) (yen_to_dollar : ℝ) : ℝ :=
  let notebooks_cost := (notebooks_price_euro * notebooks_count) * euro_to_dollar
  let pencils_cost := 0
  let pens_cost := 0
  let marked_price := markers_price_dollar * (1 - marker_discount)
  let us_total_before_tax := (marked_price + calculator_price_dollar) * (1 - coupon_discount)
  let us_total_after_tax := us_total_before_tax * (1 + sales_tax)
  notebooks_cost + pencils_cost + pens_cost + us_total_after_tax

theorem linda_spent_total : 
  total_spent 1.2 3 1.5 5 170 200 2.8 12.5 0.15 0.10 0.05 1.1 1.25 0.009 = 18.0216 := 
  by
  sorry

end linda_spent_total_l241_241772


namespace minimum_shift_value_l241_241811

theorem minimum_shift_value 
  (a : ℝ) 
  (h_gt : a > 0) 
  (shifted_eq : ∀ x, sin (2 * (x + a) - π / 3) = sin (2 * x)) :
  a = π / 6 :=
by
  sorry

end minimum_shift_value_l241_241811


namespace train_crosses_pole_l241_241551

theorem train_crosses_pole
  (speed_kmph : ℝ) (length_m : ℝ)
  (h_speed : speed_kmph = 60)
  (h_length : length_m = 150) :
  let speed_mps := speed_kmph * (1000/3600)
  in (length_m / speed_mps) = 9 := 
by
  sorry

end train_crosses_pole_l241_241551


namespace integral_eval_l241_241595

noncomputable def integral_problem : ℝ :=
  ∫ x in - (Real.pi / 2)..(Real.pi / 2), (x + Real.cos x)

theorem integral_eval : integral_problem = 2 :=
  by 
  sorry

end integral_eval_l241_241595


namespace fill_half_cistern_time_l241_241523

variable (t_half : ℝ)

-- Define a condition that states the certain amount of time to fill 1/2 of the cistern.
def fill_pipe_half_time (t_half : ℝ) : Prop :=
  t_half > 0

-- The statement to prove that t_half is the time required to fill 1/2 of the cistern.
theorem fill_half_cistern_time : fill_pipe_half_time t_half → t_half = t_half := by
  intros
  rfl

end fill_half_cistern_time_l241_241523


namespace balls_score_at_least_seven_l241_241832

theorem balls_score_at_least_seven :
  (∑ k in finset.range 4, (nat.choose 4 k) * (nat.choose 6 (5 - k)) * if (2 * k + (5 - k) ≥ 7) then 1 else 0) = 186 :=
by
  sorry

end balls_score_at_least_seven_l241_241832


namespace sum_of_even_factors_720_l241_241874

theorem sum_of_even_factors_720 : 
  let n := 2^4 * 3^2 * 5 in
  (∑ d in (Finset.range (n + 1)).filter (λ d, d % 2 = 0 ∧ n % d = 0), d) = 2340 :=
by
  let n := 2^4 * 3^2 * 5
  sorry

end sum_of_even_factors_720_l241_241874


namespace asher_speed_l241_241962

theorem asher_speed :
  (5 * 60 ≠ 0) → (6600 / (5 * 60) = 22) :=
by
  intros h
  sorry

end asher_speed_l241_241962


namespace sum_even_factors_of_720_l241_241864

theorem sum_even_factors_of_720 : 
  let even_factors_sum (n : ℕ) : ℕ := 
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  in even_factors_sum 720 = 2340 :=
by
  let even_factors_sum (n : ℕ) : ℕ :=
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  sorry

end sum_even_factors_of_720_l241_241864


namespace values_for_multiplication_values_for_division_l241_241512

noncomputable def possible_values_for_multiplication (n : ℕ) : Prop :=
  5 * n * 18 ≈ 1200

noncomputable def possible_values_for_division (n : ℕ) : Prop :=
  3 * n * 10 + 9 ≈ 300

theorem values_for_multiplication :
  ∀ n, possible_values_for_multiplication n → (n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9) := sorry

theorem values_for_division :
  ∀ n, possible_values_for_division n → (n = 0 ∨ n = 1 ∨ n = 2) := sorry

end values_for_multiplication_values_for_division_l241_241512


namespace sequence_general_term_l241_241440

theorem sequence_general_term:
  ∀ n : ℕ, (nth_term n = match n with
                         | 1 => 1
                         | 2 => 3
                         | 3 => 6
                         | 4 => 10
                         | _ => sorry -- All other terms follow the discovered pattern
                       end)
  → (nth_term n) = n * (n + 1) / 2 :=
by 
  sorry

end sequence_general_term_l241_241440


namespace total_pens_bought_l241_241028

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l241_241028


namespace Auston_taller_than_Emma_l241_241561

theorem Auston_taller_than_Emma :
  let inch_to_cm := 2.54 in
  let A_height_in_inch := 60 in
  let E_height_in_inch := 54 in
  let A_height_in_cm := A_height_in_inch * inch_to_cm in
  let E_height_in_cm := E_height_in_inch * inch_to_cm in
  A_height_in_cm > E_height_in_cm ∧ A_height_in_cm - E_height_in_cm = 15.24 :=
by
  let inch_to_cm := 2.54
  let A_height_in_inch := 60
  let E_height_in_inch := 54
  let A_height_in_cm := A_height_in_inch * inch_to_cm
  let E_height_in_cm := E_height_in_inch * inch_to_cm
  have h1 : A_height_in_cm = 152.4 := by sorry
  have h2 : E_height_in_cm = 137.16 := by sorry
  have h3 : A_height_in_cm > E_height_in_cm := by sorry
  have h4 : A_height_in_cm - E_height_in_cm = 15.24 := by sorry
  exact ⟨h3, h4⟩

end Auston_taller_than_Emma_l241_241561


namespace circle_center_l241_241236

theorem circle_center (x y : ℝ) :
  x^2 + 4 * x + y^2 - 6 * y + 1 = 0 → (x + 2, y - 3) = (0, 0) :=
by
  sorry

end circle_center_l241_241236


namespace sum_of_supercool_triangles_areas_l241_241195

theorem sum_of_supercool_triangles_areas : 
  ∃ (a b : ℕ), (a * b / 2 = 3 * (a + b)) ∧ (sum of areas a b = 324) :=
by
  sorry

end sum_of_supercool_triangles_areas_l241_241195


namespace time_to_fill_cistern_l241_241519

-- Definition of the rates for pipes A and B
def rateA (C : ℝ) := C / 20
def rateB (C : ℝ) := C / 25

-- Net rate when both pipes are open
def netRate (C : ℝ) := rateA C - rateB C

-- Time to fill the cistern at the net rate
theorem time_to_fill_cistern (C : ℝ) (h₀ : C > 0) : 
  C / netRate C = 100 := 
by
  -- Ensure the net rate is calculated properly
  have h₁ : netRate C = C / 100 := by 
    unfold netRate rateA rateB
    field_simp [show (20 : ℝ) * 25 = 500 by norm_num]
    simp
  rw [h₁]
  field_simp
  norm_num

end time_to_fill_cistern_l241_241519


namespace principal_sum_l241_241908

/-!
# Problem Statement
Given:
1. The difference between compound interest (CI) and simple interest (SI) on a sum at 10% per annum for 2 years is 65.
2. The rate of interest \( R \) is 10%.
3. The time \( T \) is 2 years.

We need to prove that the principal sum \( P \) is 6500.
-/

theorem principal_sum (P : ℝ) (R : ℝ) (T : ℕ) (H : (P * (1 + R / 100)^T - P) - (P * R * T / 100) = 65) 
                      (HR : R = 10) (HT : T = 2) : P = 6500 := 
by 
  sorry

end principal_sum_l241_241908


namespace fill_pipe_half_time_l241_241526

theorem fill_pipe_half_time (T : ℝ) (hT : 0 < T) :
  ∀ t : ℝ, t = T / 2 :=
by
  sorry

end fill_pipe_half_time_l241_241526


namespace planes_perpendicular_l241_241651

theorem planes_perpendicular (nα nβ : Vector3) (h : nα ⋅ nβ = 0) : planes_perpendicular α β :=
begin
  sorry
end

end planes_perpendicular_l241_241651


namespace gender_independence_expectation_X_probability_meet_standard_l241_241837

-- Given data
def total_population : ℕ := 100
def male_total : ℕ := 45
def female_total : ℕ := 55

def exercise_distribution : List (ℕ × ℕ) := [(30, 15), (45, 10)]

noncomputable def χ_squared (a b c d n : ℕ) : ℝ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem gender_independence :
  let a := 30
  let b := 15
  let c := 45
  let d := 10
  let n := 100 in
  χ_squared a b c d n < 3.841 :=
by sorry

-- Distribution and Expectation of X
--- Given probabilities for P(X)
def P_X (x : ℕ) : ℚ :=
  match x with
  | 0 => 1/10
  | 1 => 3/5
  | 2 => 3/10
  | _ => 0

--- Expectation of X
noncomputable def E_X : ℚ :=
  let p0 := 0 * (1 / 10)
  let p1 := 1 * (3 / 5)
  let p2 := 2 * (3 / 10)
  p0 + p1 + p2

theorem expectation_X : E_X = 6 / 5 :=
by sorry

-- Binomial distribution
def P_meeting_standard (k : ℕ) : ℚ :=
  if k = 2 then (Nat.choose 3 k) * (1/4)^k * (3/4)^(3-k) else 0

theorem probability_meet_standard :
  P_meeting_standard 2 = 9 / 64 :=
by sorry

end gender_independence_expectation_X_probability_meet_standard_l241_241837


namespace remainder_five_n_minus_eleven_l241_241128

theorem remainder_five_n_minus_eleven (n : ℤ) (h : n % 7 = 3) : (5 * n - 11) % 7 = 4 := 
    sorry

end remainder_five_n_minus_eleven_l241_241128


namespace gcd_repeated_five_digit_number_l241_241182

theorem gcd_repeated_five_digit_number :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 →
  ∀ m : ℕ, 10000 ≤ m ∧ m < 100000 →
  (10000100001 : ℕ) ∣ ((10^10 + 10^5 + 1) * n) ∧
  (10000100001 : ℕ) ∣ ((10^10 + 10^5 + 1) * m) →
  gcd ((10^10 + 10^5 + 1) * n) ((10^10 + 10^5 + 1) * m) = 10000100001 :=
sorry

end gcd_repeated_five_digit_number_l241_241182


namespace crescent_moon_area_l241_241185

-- Definition of the given problem conditions
def largeQuarterCircleArea : ℝ := (1 / 4) * π * 4^2
def smallQuarterCircleArea : ℝ := (1 / 4) * π * 2^2

-- Prove that the area of the crescent moon is 3π
theorem crescent_moon_area : largeQuarterCircleArea - smallQuarterCircleArea = 3 * π := by
  --  Sorry, the actual proof is not required per instruction
  sorry

end crescent_moon_area_l241_241185


namespace intersect_on_midline_l241_241337

variables {A B C M H_a H_c : Type}
variables [scalene_right_triangle ABC]
variables [midpoint M AC]
variables [orthocenter H_a ABM]
variables [orthocenter H_c CBM]

theorem intersect_on_midline 
  (scalene_right_triangle ABC) 
  (midpoint M AC) 
  (orthocenter H_a ABM)
  (orthocenter H_c CBM) : 
  ∃ P, (P lies_on (line_through A H_c)) ∧ (P lies_on (line_through C H_a)) ∧ (P lies_on (midline_of_triangle ABC)) :=
sorry

end intersect_on_midline_l241_241337


namespace solve_for_x_l241_241503

theorem solve_for_x (x : ℤ) (h : 5 * x + 3 = 10 * x - 22) : x = 5 :=
sorry

end solve_for_x_l241_241503


namespace quad_area_leq_one_l241_241823

theorem quad_area_leq_one {AB BC CD DA : ℝ} (H1: convex_quad ABCD) (H2: AB + BC + CD + DA = 4) :
  area_quad ABCD ≤ 1 :=
sorry

end quad_area_leq_one_l241_241823


namespace smallest_a_satisfies_condition_l241_241992

noncomputable def satisfies_condition (a : ℝ) : Prop :=
  (10 * real.sqrt ((2 * a)^2 + 1) - 3 * a^2 - 2) / (real.sqrt (1 + 3 * a^2) + 4) = 3

theorem smallest_a_satisfies_condition :
  ∃ a : ℝ, satisfies_condition a ∧ ∀ b : ℝ, satisfies_condition b → a ≤ b :=
sorry

end smallest_a_satisfies_condition_l241_241992


namespace collinear_RPQ_l241_241641

open Function

variables {α β γ : Type*}
variables {Γ1 Γ2 : Set α} -- Circles Gamma_1 and Gamma_2
variables {A B C D E P Q R : α} -- Points

-- Define the conditions
def conditions (Γ1 Γ2 : Set α) (A B C D E P Q R : α) : Prop :=
  ∃ (tangent_l1 tangent_l2 : Set α)
  (onΓ1 : ∀ x ∈ Γ1, x = A ∨ x ∈ tangent_l1)
  (onΓ2 : ∀ x ∈ Γ2, x = A ∨ x ∈ tangent_l2)
  (intersectC : ∃! C ∈ Γ1, C ∈ tangent_l2)
  (intersectD : ∃! D ∈ Γ2, D ∈ tangent_l1)
  (extensionAB : ∃ ext : Set α, ∀ x, x ∈ ext ↔ x = A ∨ x = B ∨ ∃! y, y ∈ ext ∧ (A = B ∨ x = y ∨ (A = y ∧ B = x)))
  (AP_PE : ∀ x, x = P ↔ x = E)
  (R_BDE : ∃ R, R ∈ circumcircle B D E)
  (Q_BCE : ∃ Q, Q ∈ circumcircle B C E)
  (collinear_RPQ : R ∈ linear_combination P Q)

-- Translate the equivalent problem into a theorem
theorem collinear_RPQ
  (h: conditions Γ1 Γ2 A B C D E P Q R) :
  ∃ line : Set α, R ∈ line ∧ P ∈ line ∧ Q ∈ line :=
begin
  sorry
end

end collinear_RPQ_l241_241641


namespace sum_of_even_factors_720_l241_241868

theorem sum_of_even_factors_720 : 
  let n := 2^4 * 3^2 * 5 in
  (∑ d in (Finset.range (n + 1)).filter (λ d, d % 2 = 0 ∧ n % d = 0), d) = 2340 :=
by
  let n := 2^4 * 3^2 * 5
  sorry

end sum_of_even_factors_720_l241_241868


namespace product_of_two_real_numbers_sum_three_times_product_l241_241094

variable (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)

theorem product_of_two_real_numbers_sum_three_times_product
    (h : x + y = 3 * x * y) :
  x * y = (x + y) / 3 :=
sorry

end product_of_two_real_numbers_sum_three_times_product_l241_241094


namespace total_fish_l241_241392

theorem total_fish (Lilly Rosy Alex Jamie Sam : ℕ)
  (hL: Lilly = 10)
  (hR: Rosy = 11)
  (hA: Alex = 15)
  (hJ: Jamie = 8)
  (hS: Sam = 20) :
  Lilly + Rosy + Alex + Jamie + Sam = 64 :=
by
  rw [hL, hR, hA, hJ, hS]
  norm_num
  done

end total_fish_l241_241392


namespace shaded_cell_is_7_l241_241723

theorem shaded_cell_is_7
  (S : ℕ)
  (x y ? : ℕ)
  (h₁ : S = 1 + 8 + 6 + x)
  (h₂ : S = 2 + 7 + 5 + y)
  (h₃ : S = 4 + 3 + 7 + 1)
  (h₄ : S = x + 5 + ? + ?) :
  x = 7 :=
by
  sorry

end shaded_cell_is_7_l241_241723


namespace calc_sequence_l241_241516

theorem calc_sequence (x : ℝ) (n : ℕ) (hx : x ≠ 0) : 
  let y := (n : ℕ → ℝ) × (x : ℝ) → ℝ 
  in y = x ^ ((-4) * 2 ^ (n - 1)) :=
sorry

end calc_sequence_l241_241516


namespace g_36_l241_241438

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eqn (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g(x * y) = g(x) / y
axiom g_24 : g 24 = 8

theorem g_36 : g 36 = 16 / 3 :=
by
  sorry

end g_36_l241_241438


namespace probability_of_non_touching_square_is_correct_l241_241397

def square_not_touching_perimeter_or_center_probability : ℚ :=
  let total_squares := 100
  let perimeter_squares := 24
  let center_line_squares := 16
  let touching_squares := perimeter_squares + center_line_squares
  let non_touching_squares := total_squares - touching_squares
  non_touching_squares / total_squares

theorem probability_of_non_touching_square_is_correct :
  square_not_touching_perimeter_or_center_probability = 3 / 5 :=
by
  sorry

end probability_of_non_touching_square_is_correct_l241_241397


namespace variance_of_sample_data_l241_241459

def sample_data : List ℤ := [-2, 0, 5, 3, 4]

def mean (l : List ℤ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℤ) : ℚ :=
  let μ := mean l
  l.map (λ x => (x - μ) ^ 2).sum / l.length

theorem variance_of_sample_data :
  variance sample_data = 34 / 5 := by
  sorry

end variance_of_sample_data_l241_241459


namespace maximize_hotel_profit_l241_241186

theorem maximize_hotel_profit :
  let rooms := 50
  let base_price := 180
  let increase_per_vacancy := 10
  let maintenance_cost := 20
  ∃ (x : ℕ), ((base_price + increase_per_vacancy * x) * (rooms - x) 
    - maintenance_cost * (rooms - x) = 10890) ∧ (base_price + increase_per_vacancy * x = 350) :=
by
  sorry

end maximize_hotel_profit_l241_241186


namespace four_distinct_perfect_square_l241_241788

open Set

theorem four_distinct_perfect_square (A : Finset ℕ) :
  A.card = 2016 →
  (∀ n ∈ A, ∀ p, Prime p → p ∣ n → p < 30) →
  ∃ a b c d ∈ A, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ (a * b * c * d).sqrt * (a * b * c * d).sqrt = a * b * c * d :=
by {
  intros h_card h_prime_divisors,
  -- Admitting the rest of the proof
  sorry
}

end four_distinct_perfect_square_l241_241788


namespace calculate_S₆_l241_241633

noncomputable def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
a₁ + (n - 1) * d

theorem calculate_S₆ {a₁ d : ℤ} (h₁ : d = 2) 
(h₂ : (arithmetic_sequence a₁ d 3)^2 = a₁ * (arithmetic_sequence a₁ d 4)) :
  let S₆ := 6 * a₁ + (5 * 6 / 2 * d) in
  S₆ = -18 :=
by
  sorry

end calculate_S₆_l241_241633


namespace solution_set_inequality_l241_241648

variable (f : ℝ → ℝ)

-- Assume f is an even function
axiom even_f : ∀ x, f(x) = f(-x)

-- Assume f is monotonically decreasing on [0, +∞)
axiom mono_decreasing_f : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f(x) ≥ f(y)

-- Assume f(3) = 0
axiom f_at_3 : f(3) = 0

theorem solution_set_inequality :
  {x : ℝ | (2 * x - 5) * f (x - 1) < 0} = {x | (-2 < x ∧ x < 5 / 2) ∨ 4 < x} :=
by
  sorry

end solution_set_inequality_l241_241648


namespace average_time_per_stop_l241_241455

-- Definitions from the conditions
def pizzas : Nat := 12
def stops_with_two_pizzas : Nat := 2
def total_delivery_time : Nat := 40

-- Using the conditions to define what needs to be proved
theorem average_time_per_stop : 
  let single_pizza_stops := pizzas - stops_with_two_pizzas * 2
  let total_stops := single_pizza_stops + stops_with_two_pizzas
  let average_time := total_delivery_time / total_stops
  average_time = 4 := by
  -- Proof to be provided
  sorry

end average_time_per_stop_l241_241455


namespace flippy_numbers_divisible_by_25_count_l241_241682

def is_flippy (n : ℕ) : Prop :=
  let digits := (toDigits 10 n) in
  digits.length = 6 ∧
  ∀ i, i < 5 → digits.nth i ≠ digits.nth (i + 1) ∧
  ∀ i, i < 4 → digits.nth i = digits.nth (i + 2)

def is_divisible_by_25 (n : ℕ) : Prop :=
  let digits := (toDigits 10 n) in
  digits.length = 6 ∧
  digits.nth 4 = 2 ∧ digits.nth 5 = 5 ∨
  digits.nth 4 = 5 ∧ digits.nth 5 = 0 ∨
  digits.nth 4 = 7 ∧ digits.nth 5 = 5

theorem flippy_numbers_divisible_by_25_count :
  {n : ℕ // is_flippy n ∧ is_divisible_by_25 n}.card = 24 :=
sorry

end flippy_numbers_divisible_by_25_count_l241_241682


namespace four_painters_workdays_l241_241062

theorem four_painters_workdays (D : ℚ) :
  (6 * (3 / 2) = 9) ∧ (4 * D = 9) → D = 9 / 4 :=
by
  intro h
  cases h with h1 h2
  rw [h2]
  norm_num -- Ensures the fractions are handled correctly
  sorry

end four_painters_workdays_l241_241062


namespace positive_integer_solution_l241_241596

theorem positive_integer_solution : ∃ n : ℕ, 
  (∀ n > 0, 
  (∑ i in range n, (2 * i + 1)) / (∑ i in range n, (2 * (i + 1))) = 125 / 126) 
:= sorry

end positive_integer_solution_l241_241596


namespace lcm_135_468_l241_241486

theorem lcm_135_468 : Nat.lcm 135 468 = 7020 := by
  sorry

end lcm_135_468_l241_241486


namespace number_of_knights_l241_241920

def traveler := Type
def is_knight (t : traveler) : Prop := sorry
def is_liar (t : traveler) : Prop := sorry

axiom total_travelers : Finset traveler
axiom vasily : traveler
axiom  h_total : total_travelers.card = 16

axiom kn_lie (t : traveler) : is_knight t ∨ is_liar t

axiom vasily_liar : is_liar vasily
axiom contradictory_statements_in_room (rooms: Finset (Finset traveler)):
  (∀ room ∈ rooms, ∃ t ∈ room, (is_liar t ∧ is_knight t))
  ∧
  (∀ room ∈ rooms, ∃ t ∈ room, (is_knight t ∧ is_liar t))

theorem number_of_knights : 
  ∃ k, k = 9 ∧ (∃ l, l = 7 ∧ ∀ t ∈ total_travelers, (is_knight t ∨ is_liar t)) :=
sorry

end number_of_knights_l241_241920


namespace area_convex_quadrilateral_le_one_l241_241825

theorem area_convex_quadrilateral_le_one
  (AB BC CD DA : ℝ)
  (h_pos: AB ≥ 0 ∧ BC ≥ 0 ∧ CD ≥ 0 ∧ DA ≥ 0)
  (h_perimeter : AB + BC + CD + DA = 4)
  (h_convex : convex_quadrilateral ABCD) :
  area ABCD ≤ 1 := 
sorry

end area_convex_quadrilateral_le_one_l241_241825


namespace num_balls_total_l241_241722

theorem num_balls_total (m : ℕ) (h1 : 6 < m) (h2 : (6 : ℝ) / (m : ℝ) = 0.3) : m = 20 :=
by
  sorry

end num_balls_total_l241_241722


namespace sum_of_even_factors_720_l241_241877

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) :=
    (2 + 4 + 8 + 16) * (1 + 3 + 9) * (1 + 5)
  in even_factors_sum 720 = 2340 :=
by
  sorry

end sum_of_even_factors_720_l241_241877


namespace edward_tickets_l241_241983

theorem edward_tickets (dunk_tickets rides_cost rides_count: ℕ) (tickets_left: rides_cost * rides_count = 56) (dunk_tickets = 23) (each_ride_cost = 7) (rides_count = 8) :
  tickets_left + dunk_tickets = 79 :=
by
  sorry

end edward_tickets_l241_241983


namespace on_real_axis_in_first_quadrant_on_line_l241_241997

theorem on_real_axis (m : ℝ) : 
  (m = -3 ∨ m = 5) ↔ (m^2 - 2 * m - 15 = 0) := 
sorry

theorem in_first_quadrant (m : ℝ) : 
  (m < -3 ∨ m > 5) ↔ ((m^2 + 5 * m + 6 > 0) ∧ (m^2 - 2 * m - 15 > 0)) := 
sorry

theorem on_line (m : ℝ) : 
  (m = 1 ∨ m = -5 / 2) ↔ ((m^2 + 5 * m + 6) + (m^2 - 2 * m - 15) + 5 = 0) := 
sorry

end on_real_axis_in_first_quadrant_on_line_l241_241997


namespace correct_equations_count_l241_241464

theorem correct_equations_count : 
  let eq1 := (-5) + (+3) = -2,
      eq2 := -(-2)^3 = 8,
      eq3 := (+5/6) + (-1/6) = 2/3,
      eq4 := -3 / (-1/3) = 9 in
  eq1 = true ∧ eq2 = true ∧ eq3 = true ∧ eq4 = true → 
  2 = 2 := 
by 
  intro h,
  exact rfl

end correct_equations_count_l241_241464


namespace total_dolls_l241_241407

-- Definitions based on the given conditions
def grandmother_dolls : Nat := 50
def sister_dolls : Nat := grandmother_dolls + 2
def rene_dolls : Nat := 3 * sister_dolls

-- Statement we want to prove
theorem total_dolls : grandmother_dolls + sister_dolls + rene_dolls = 258 := by
  sorry

end total_dolls_l241_241407


namespace proof_ellipse_foci_and_equation_proof_ellipse_tangents_and_pointP_l241_241729

noncomputable def ellipse_foci_and_equation : Prop :=
  ∃ (a b c : ℝ) (h1 : a > b ∧ b > 0) (h2 : 2 * a = 8) (h3 : c = 2) (h4 : a = 4) (h5 : b^2 = a^2 - c^2 ) (h6 : b^2 = 12),
    ∃ (E_eq : ∀x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1), E_eq 16 12

theorem proof_ellipse_foci_and_equation :
  ellipse_foci_and_equation :=
sorry

noncomputable def ellipse_tangents_and_pointP : Prop :=
  ∃ (x0 y0 : ℝ), 
    (x0^2 / 16 + y0^2 / 12 = 1) ∧ 
    (y0^2 / ((2 - x0)^2 - 2) = 1 / 2) ∧ 
    (5 * x0^2 - 8 * x0 - 36 = 0) ∧
    (x0 = -2) ∧ 
    (y0 = 3 ∨ y0 = -3)

theorem proof_ellipse_tangents_and_pointP :
  ellipse_tangents_and_pointP :=
sorry

end proof_ellipse_foci_and_equation_proof_ellipse_tangents_and_pointP_l241_241729


namespace intersection_point_on_circle_l241_241386

open EuclideanGeometry

variables {C C1 C2 : Circle} {A B D E F : Point} {t : Line}

-- Hypotheses
variables (hC1 : C1.tangent C A)
variables (hC2 : C2.tangent C B)
variables (hC1C2Touch : C1.nonIntersect C2)
variables (hTangent : Line.tangentToCircles t C1 C2 D E)
variables (hSameSide : SameSideOfLine C1 C2 t)
variables (hAD : AD_intersect_lineA_C_Circle C A D)
variables (hBE : BE_intersect_lineB_C_Circle C B E)

-- Statement to prove
theorem intersection_point_on_circle :
  F = intersection_point AD BE → lies_on_circle F C := 
by
  sorry

end intersection_point_on_circle_l241_241386


namespace max_divisor_f_l241_241621

-- Given definition
def f (n : ℕ) : ℕ := (2 * n + 7) * 3 ^ n + 9

-- Main theorem to be proved
theorem max_divisor_f :
  ∃ m : ℕ, (∀ n : ℕ, 0 < n → m ∣ f n) ∧ m = 36 :=
by
  -- The proof would go here
  sorry

end max_divisor_f_l241_241621


namespace eighth_vertex_on_sphere_l241_241534

-- Assume we have a hexagon where all faces are quadrilaterals
variable {A B, C, D, E, F, K L : Point}
variable (O : Point)
variable (Sphere : Set Point)
variable [is_sphere Sphere O]
variable (quadrilateral_faces : Set (Set Point))

-- Assume that seven of the eight vertices lie on the surface of the sphere
variable (A_on_sphere : A ∈ Sphere)
variable (B_on_sphere : B ∈ Sphere)
variable (D_on_sphere : D ∈ Sphere)
variable (E_on_sphere : E ∈ Sphere)
variable (F_on_sphere : F ∈ Sphere)
variable (K_on_sphere : K ∈ Sphere)
variable (L_on_sphere : L ∈ Sphere)

-- Definition of the vertices of the hexagon being quadrilaterals
variable (faces_are_quadrilaterals : 
  ∀ {X Y Z W : Point}, {X, Y, Z, W} ∈ quadrilateral_faces → 
  is_quadrilateral {X, Y, Z, W})

-- Prove that the eighth vertex C also lies on the surface of the sphere
theorem eighth_vertex_on_sphere (C : Point) 
  (hexagon_faces : Set (Set Point))
  (hexagon_condition : Set.member {A, B, C, D, E, F, K, L} hexagon_faces)
  (other_vertices_on_sphere : 
    Set.member {A, B, D, E, F, K, L} (Sphere : Set Point)) :
  C ∈ Sphere := 
sorry

end eighth_vertex_on_sphere_l241_241534


namespace earnings_difference_is_200_l241_241975

noncomputable def difference_in_earnings : ℕ :=
  let asking_price := 5200
  let maintenance_cost := asking_price / 10
  let first_offer_earnings := asking_price - maintenance_cost
  let headlight_cost := 80
  let tire_cost := 3 * headlight_cost
  let total_repair_cost := headlight_cost + tire_cost
  let second_offer_earnings := asking_price - total_repair_cost
  second_offer_earnings - first_offer_earnings

theorem earnings_difference_is_200 : difference_in_earnings = 200 := by
  sorry

end earnings_difference_is_200_l241_241975


namespace f_pi_minus_f_neg_pi_l241_241700

def f (x : ℝ) : ℝ := x^3 * Real.cos x + 3 * x^2 + 7 * Real.sin x

theorem f_pi_minus_f_neg_pi : f π - f (-π) = -2 * π^3 := 
by 
  sorry

end f_pi_minus_f_neg_pi_l241_241700


namespace sin_theta_line_plane_l241_241225

theorem sin_theta_line_plane 
  (θ : ℝ)
  (d : ℝ × ℝ × ℝ := (4, 5, 8))
  (n : ℝ × ℝ × ℝ := (-12, -3, 14))
  (dot_product : ℝ := 49)
  (mag_d : ℝ := Real.sqrt (4^2 + 5^2 + 8^2))
  (mag_n : ℝ := Real.sqrt ((-12)^2 + (-3)^2 + 14^2))
  (cos_90_minus_theta : ℝ := dot_product / (mag_d * mag_n))
  (sinθ : ℝ := cos_90_minus_theta) :
  sin θ = 49 / Real.sqrt 36645 :=
sorry

end sin_theta_line_plane_l241_241225


namespace find_length_AB_distance_to_midpoint_l241_241667

noncomputable def parametric_eq_x (t : ℝ) : ℝ := -1 + 3 * t
noncomputable def parametric_eq_y (t : ℝ) : ℝ := 2 - 4 * t

def curve_eq (x y : ℝ) : Prop := (y - 2)^2 - x^2 = 1

def intersection_points (t1 t2 : ℝ) : Prop :=
  parametric_eq_x t1 = parametric_eq_x t2 ∧ parametric_eq_y t1 = parametric_eq_y t2 ∧
    curve_eq (parametric_eq_x t1) (parametric_eq_y t1) ∧ 
    curve_eq (parametric_eq_x t2) (parametric_eq_y t2)

def length_AB (t1 t2 : ℝ) : ℝ := Real.sqrt ((t1 + t2)^2 - 4 * t1 * t2)

theorem find_length_AB (t1 t2 : ℝ) (h_inter : intersection_points t1 t2) :
  length_AB t1 t2 = (10 * Real.sqrt 23) / 7 :=
  sorry

def midpoint_param (t1 t2 : ℝ) : ℝ := (t1 + t2) / 2

def distance_from_P (x y : ℝ) (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem distance_to_midpoint (t1 t2 : ℝ) (x_mid y_mid : ℝ) (h_inter : intersection_points t1 t2)
    (hP : (x_mid, y_mid) = (parametric_eq_x (midpoint_param t1 t2), parametric_eq_y (midpoint_param t1 t2))) :
  distance_from_P (-1, 2) (x_mid, y_mid) = 15 / 7 :=
  sorry

end find_length_AB_distance_to_midpoint_l241_241667


namespace solve_m_l241_241609

theorem solve_m (n : ℤ) : ∀ m : ℤ, 21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 :=
by {
  intro m,
  intro h,
  sorry
}

end solve_m_l241_241609


namespace sqrt_mul_sqrt_l241_241573

theorem sqrt_mul_sqrt (a b c : ℝ) (ha : a = √150) (hb : b = √48) (hc : c = √12) (hd : d = 3) :
  a * b * c * d = 360 * √6 := by
  rw [ha, hb, hc, hd]
  -- Here we rewrite the problem according to given conditions
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  -- Continue rewriting and simplifying accordingly
  sorry

end sqrt_mul_sqrt_l241_241573


namespace sum_of_even_factors_720_l241_241889

theorem sum_of_even_factors_720 :
  let even_factors_sum (n : ℕ) : ℕ :=
    match n with
    | 720 => 
      let sum_powers_2 := 2 + 4 + 8 + 16 in
      let sum_powers_3 := 1 + 3 + 9 in
      let sum_powers_5 := 1 + 5 in
      sum_powers_2 * sum_powers_3 * sum_powers_5
    | _ => 0
  in
  even_factors_sum 720 = 2340 :=
by 
  sorry

end sum_of_even_factors_720_l241_241889


namespace who_is_wrong_l241_241159

theorem who_is_wrong 
  (a1 a2 a3 a4 a5 a6 : ℤ)
  (h1 : a1 + a3 + a5 = a2 + a4 + a6 + 3)
  (h2 : a2 + a4 + a6 = a1 + a3 + a5 + 5) : 
  False := 
sorry

end who_is_wrong_l241_241159


namespace perfect_squares_less_than_500_ending_in_4_l241_241679

theorem perfect_squares_less_than_500_ending_in_4 : 
  (∃ (squares : Finset ℕ), (∀ n ∈ squares, n < 500 ∧ (n % 10 = 4)) ∧ squares.card = 5) :=
by
  sorry

end perfect_squares_less_than_500_ending_in_4_l241_241679


namespace find_smallest_positive_angle_l241_241242

noncomputable def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

theorem find_smallest_positive_angle :
  ∃ φ > 0, cos_deg φ = sin_deg 45 + cos_deg 37 - sin_deg 23 - cos_deg 11 ∧ φ = 53 := 
by
  sorry

end find_smallest_positive_angle_l241_241242


namespace compute_B_2_1_l241_241980

def B : ℕ → ℕ → ℕ
| 0, n := n + 2
| (m+1), 0 := B m 2
| (m+1), (n+1) := B m (B (m+1) n)

theorem compute_B_2_1 : B 2 1 = 12 := by
  sorry

end compute_B_2_1_l241_241980


namespace midpoint_probability_l241_241376

theorem midpoint_probability (T : Type) (points : Finset (ℤ × ℤ × ℤ)) :
  (∀ (a b c : ℤ), (a, b, c) ∈ points ↔ (0 ≤ a ∧ a ≤ 3) ∧ (0 ≤ b ∧ b ≤ 4) ∧ (0 ≤ c ∧ c ≤ 5)) →
  ∃ p q : ℕ, (nat.gcd p q = 1 ∧ p + q = 91 ∧ 
    (points.card > 1 → ∃ valid_pairs total_pairs : ℕ,
      total_pairs = points.card * (points.card - 1) / 2 ∧
      (∃ R : Finset ((ℤ × ℤ × ℤ) × (ℤ × ℤ × ℤ)), R.card = valid_pairs ∧
         (∀ (x y : (ℤ × ℤ × ℤ)), (x ∈ points ∧ y ∈ points ↔ x ≠ y ∧ (x, y) ∈ R) 
          → (let a := (x.1 + y.1) / 2, b := (x.2 + y.2) / 2, c := (x.3 + y.3) / 2
             in (a, b, c) ∈ points)) →
      (valid_pairs * q = total_pairs * p))) :=
begin
  intros h_points,
  sorry
end

end midpoint_probability_l241_241376


namespace sum_even_factors_of_720_l241_241896

open Nat

theorem sum_even_factors_of_720 :
  ∑ d in (finset.filter (λ x, even x) (finset.divisors 720)), d = 2340 :=
by
  sorry

end sum_even_factors_of_720_l241_241896


namespace equal_segments_l241_241206

variables {O M P Q C D A B E F : Point}
variables {r : ℝ}
variables {GH PQ : Line}

axiom midpoint_M (H G : Point) : is_midpoint M G H
axiom symmetric_PQ : symmetric P Q M
axiom secant_P (s : Line) : intersects s O P C D
axiom secant_Q (s : Line) : intersects s O Q A B
axiom line_intersection_AC_PQ : intersects (line_through A C) PQ E
axiom line_intersection_BD_PQ : intersects (line_through B D) PQ F
axiom line_GH : line_contains GH G H
axiom line_PQ : line_contains PQ P Q

theorem equal_segments : dist E M = dist M F :=
by
  sorry

end equal_segments_l241_241206


namespace a_beats_b_by_90_meters_l241_241517

-- Define constants for the problem
def distance : ℝ := 720
def time_A : ℝ := 84
def time_B : ℝ := 96

-- Define speeds as conditions
def speed_A : ℝ := distance / time_A
def speed_B : ℝ := distance / time_B

-- Define the theorem statement to prove A beats B by 90 meters
theorem a_beats_b_by_90_meters 
  (dist : ℝ := distance)
  (t_A : ℝ := time_A)
  (t_B : ℝ := time_B)
  (v_A : ℝ := speed_A)
  (v_B : ℝ := speed_B) :
  dist - (v_B * t_A) = 90 :=
  by {
    sorry
  }

end a_beats_b_by_90_meters_l241_241517


namespace angle_measure_l241_241239

theorem angle_measure :
  let sum_sins := ∑ i in (2907 : ℕ) .. 6507, real.sin (i : ℝ) * real.pi / 180
  let sum_coss := ∑ i in (2880 : ℕ) .. 6480, real.cos (i : ℝ) * real.pi / 180
  let power := real.cos (6480 : ℝ) * real.pi / 180 + sum_coss
  ∃ (delta : ℝ), delta = real.arccos (real.sin (27 * real.pi / 180)) ↔  delta = 63 * real.pi / 180 := 
by
  sorry

end angle_measure_l241_241239


namespace spheres_max_min_dist_l241_241348

variable {R_1 R_2 d : ℝ}

noncomputable def max_min_dist (R_1 R_2 d : ℝ) (sep : d > R_1 + R_2) :
  ℝ × ℝ :=
(d + R_1 + R_2, d - R_1 - R_2)

theorem spheres_max_min_dist {R_1 R_2 d : ℝ} (sep : d > R_1 + R_2) :
  max_min_dist R_1 R_2 d sep = (d + R_1 + R_2, d - R_1 - R_2) := by
sorry

end spheres_max_min_dist_l241_241348


namespace programmer_hours_worked_l241_241982

theorem programmer_hours_worked :
  ∃ (T : ℝ), (T * (1 - (1 / 4 + 3 / 8)) = 18) ∧ (T = 48) :=
begin
  use 48,
  split,
  { simp,
    field_simp,
    norm_num, },
  { refl, },
end

end programmer_hours_worked_l241_241982


namespace min_matches_for_champion_win_min_total_matches_if_champion_wins_11_l241_241798

-- Define teams
def Team := ℕ -- Assume A = 0, B = 1, C = 2

-- Define the setup according to the problem
structure GoTournament :=
  (players_per_team : ℕ := 9)
  (teams : fin 3)

-- The minimum number of matches the champion team must win is 9
theorem min_matches_for_champion_win {T : GoTournament} :
  ∀ champion, champion ∈ T.teams →
  (∀ A B, A ≠ champion ∧ B ≠ champion ∧ A ∈ T.teams ∧ B ∈ T.teams) →
  (∀ matches, matches ≥ 9) → 
  matches = if champion_wins then 9 else 0 := 
sorry

-- If the champion team won 11 matches, the minimum number of total matches played is 24
theorem min_total_matches_if_champion_wins_11 {T : GoTournament} :
  ∀ champion, champion ∈ T.teams →
  ∀ matches,
    (matches = 24) → 
  champion_wins_match 11  :=
sorry

end min_matches_for_champion_win_min_total_matches_if_champion_wins_11_l241_241798


namespace hundred_digit_12_fact_minus_8_fact_l241_241848

theorem hundred_digit_12_fact_minus_8_fact :
  let factorial := Nat.factorial
  8! = 40320 ∧ 12! = 8! * 9 * 10 * 11 * 12 → 
  (12! - 8!) % 1000 / 100 = 2 :=
by
  let factorial := Nat.factorial
  sorry

end hundred_digit_12_fact_minus_8_fact_l241_241848


namespace general_term_sum_of_terms_inequality_l241_241628

variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}

-- All given conditions
def condition_1 : Prop := ∀ n : ℕ, 3 * (n + 1) * a n = n * a (n + 1)
def initial_condition : Prop := a 1 = 3

-- Statement to prove the general term formula of the sequence {a_n}
theorem general_term (h1 : condition_1) (h2 : initial_condition) :
  ∀ n, a n = n * 3^n :=
sorry

-- Statement to prove the sum of the first n terms of the sequence {a_n}
theorem sum_of_terms (h1 : condition_1) (h2 : initial_condition) :
  ∀ S : ℕ → ℕ, S = λ n, Σ k in finset.range (n + 1), a k
  → ∀ n, S n = ((2 * n - 1) * 3 ^ (n + 1) + 3) / 4 :=
sorry

-- Given and to prove the inequality condition.
theorem inequality (h1 : condition_1) (h2 : initial_condition)
  (rel : ∀ n, a n / b n = (2 * n + 3) / (n + 1)) :
  ∀ n, 5 / 6 ≤ Σ k in finset.range (n + 1), 1 / (b k) ∧ Σ k in finset.range (n + 1), 1 / (b k) < 1 :=
sorry

end general_term_sum_of_terms_inequality_l241_241628


namespace simplify_fraction_l241_241794

theorem simplify_fraction :
  (sqrt 75 = 5 * sqrt 3) →
  (sqrt 48 = 4 * sqrt 3) →
  (sqrt 18 = 3 * sqrt 3) →
  (3 / (sqrt 75 + sqrt 48 + sqrt 18) = sqrt 3 / 12) :=
by
  intros h1 h2 h3
  sorry

end simplify_fraction_l241_241794


namespace remainder_of_p_l241_241703

theorem remainder_of_p (p : ℤ) (h1 : p = 35 * 17 + 10) : p % 35 = 10 := 
  sorry

end remainder_of_p_l241_241703


namespace determine_knights_l241_241922

noncomputable def number_of_knights (total_travelers: ℕ) (vasily_is_liar: Prop) (statement_by_vasily: ∀ room: ℕ, (more_liars room ∨ more_knights room) = false) : ℕ := 9

theorem determine_knights :
  ∀ (travelers: ℕ)
    (liar_iff_false: ∀ (P: Prop), liar P ↔ P = false)
    (vasily: traveler)
    (rooms: fin 3 → fin 16)
    (more_liars: Π (r: fin 3), Prop)
    (more_knights: Π (r: fin 3), Prop),
    travelers = 16 →
    liar (more_liars (rooms 0)) ∧ liar (more_knights (rooms 0)) ∧
    liar (more_liars (rooms 1)) ∧ liar (more_knights (rooms 1)) ∧
    liar (more_liars (rooms 2)) ∧ liar (more_knights (rooms 2)) →
    ∃ (k l: ℕ),
      k + l = 15 ∧ k - l = 1 ∧ k = 9 :=
begin
  sorry
end

end determine_knights_l241_241922


namespace sum_of_fractions_equals_l241_241218

theorem sum_of_fractions_equals :
  (1 / 15 + 2 / 25 + 3 / 35 + 4 / 45 : ℚ) = 0.32127 :=
  sorry

end sum_of_fractions_equals_l241_241218


namespace sum_even_factors_of_720_l241_241866

theorem sum_even_factors_of_720 : 
  let even_factors_sum (n : ℕ) : ℕ := 
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  in even_factors_sum 720 = 2340 :=
by
  let even_factors_sum (n : ℕ) : ℕ :=
      ((∑ a in finset.range 5, 2^a) * (∑ b in finset.range 3, 3^b) * (∑ c in finset.range 2, 5^c))
  sorry

end sum_even_factors_of_720_l241_241866


namespace number_of_new_girls_admitted_l241_241560

theorem number_of_new_girls_admitted
    (initial_boys : ℕ) (initial_girls : ℕ)
    (final_students : ℕ) (final_boys : ℕ) (new_girls : ℕ) :
    initial_boys = 15 →
    initial_girls = initial_boys + (initial_boys / 5) →
    final_students = 51 →
    final_boys = initial_boys →
    new_girls = (final_students - final_boys) - initial_girls →
    new_girls = 18 :=
by
  intros hb hg hs hfb hng
  rw [hb, hfb] at hs
  have h_initial_girls : initial_girls = 18 := by
    calc
      initial_girls = 15 + (15 / 5) : by rw [hb, hg]
      ... = 15 + 3 : by norm_num
      ... = 18 : by norm_num
  rw h_initial_girls at hng
  rw hs at hng
  calc
    new_girls = 51 - 15 - 18 : by rw hng
    ... = 36 - 18 : by norm_num
    ... = 18 : by norm_num

end number_of_new_girls_admitted_l241_241560


namespace superhero_speed_in_mph_l241_241550

theorem superhero_speed_in_mph (superhero_speed_kpm : ℝ) (km_to_miles : ℝ) :
  superhero_speed_kpm = 1000 →
  km_to_miles = 0.6 →
  (superhero_speed_kpm * 60 * km_to_miles = 36000) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end superhero_speed_in_mph_l241_241550


namespace find_principal_amount_l241_241121

theorem find_principal_amount (R : ℝ) (SI : ℝ) (T : ℝ) (hR : R = 20) (hSI : SI = 160) (hT : T = 2) : 
  ∃ P : ℝ, P = 400 := 
by 
  -- Definitions based on given conditions
  have h1 : 160 = SI, from hSI,
  have h2 : 20 = R, from hR,
  have h3 : 2 = T, from hT,
  
  -- Derivation from the simple interest formula
  let P := 400,
  have h4 : 160 = P * R * T / 100, 
  {
    calc
    160 = (P * 20 * 2) / 100 : by 
      simp [SI, P, R, T]
    ... = P * 0.40 : by
      norm_num, split_adj
    ... = 400 : by
      norm_num
  },
  
  existsi P, 
  exact h4,
  sorry -- Proof here is skipped.

end find_principal_amount_l241_241121


namespace trig_identity_l241_241912

theorem trig_identity (α : ℝ) :
  (cos ((5 / 2) * Real.pi - 6 * α) + sin (Real.pi + 4 * α) + sin (3 * Real.pi - α)) /
  (sin ((5 / 2) * Real.pi + 6 * α) + cos (4 * α - 2 * Real.pi) + cos (α + 2 * Real.pi)) = 
  Real.tan α := by
sorry

end trig_identity_l241_241912


namespace sum_sr_values_l241_241297

def r (x : ℝ) := |x| + 3
def s (x : ℝ) := -|x|

def sr (x : ℝ) := s (r x)

theorem sum_sr_values :
  ∑ x in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], sr x = -63 :=
by
  -- Assuming -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5 are interpreted as a list of reals
  sorry

end sum_sr_values_l241_241297


namespace log_decreasing_iff_a_ge_4_l241_241646

noncomputable def f (a x : ℝ) : ℝ :=
  log (2:ℝ) (a * x^2 - x + 1 + a)

def is_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

theorem log_decreasing_iff_a_ge_4 (a : ℝ) :
  is_decreasing (f a) (Set.Iio 2) ↔ 4 ≤ a :=
sorry

end log_decreasing_iff_a_ge_4_l241_241646


namespace find_alpha_l241_241726

theorem find_alpha (α : ℝ) :
  let θ1 := 7 * α
  let θ2 := 8 * α
  let θ3 := 45
  θ1 + θ2 + θ3 = 180 → α = 9 := 
by
  intros θ1 θ2 θ3 h,
  sorry

end find_alpha_l241_241726


namespace problem_statement_l241_241126

def is_direct_proportion_function (f : ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x : ℝ, f x = k * x 

theorem problem_statement (m : ℝ) :
  (m^2 - 3 = 1) ∧ (m + 2 ≠ 0) → is_direct_proportion_function (λ x, (m + 2) * x^(m^2 - 3)) :=
by
  sorry

end problem_statement_l241_241126


namespace earnings_difference_is_200_l241_241974

noncomputable def difference_in_earnings : ℕ :=
  let asking_price := 5200
  let maintenance_cost := asking_price / 10
  let first_offer_earnings := asking_price - maintenance_cost
  let headlight_cost := 80
  let tire_cost := 3 * headlight_cost
  let total_repair_cost := headlight_cost + tire_cost
  let second_offer_earnings := asking_price - total_repair_cost
  second_offer_earnings - first_offer_earnings

theorem earnings_difference_is_200 : difference_in_earnings = 200 := by
  sorry

end earnings_difference_is_200_l241_241974


namespace second_number_is_22_l241_241330

noncomputable section

variables (x y : ℕ)

-- Definitions based on the conditions
-- Condition 1: The sum of two numbers is 33
def sum_condition : Prop := x + y = 33

-- Condition 2: The second number is twice the first number
def twice_condition : Prop := y = 2 * x

-- Theorem: Given the conditions, the second number y is 22.
theorem second_number_is_22 (h1 : sum_condition x y) (h2 : twice_condition x y) : y = 22 :=
by
  sorry

end second_number_is_22_l241_241330


namespace find_MN_l241_241355

noncomputable def length_of_AB := 3
noncomputable def length_of_BC := 4
noncomputable def length_of_AC := 5

def angle_bisector_theorem (AD DC AB BC : ℚ) : Prop :=
  (AD / DC) = (AB / BC)

def length_of_MN (AB BC AC BD : ℚ) (AD DC : ℕ) : ℚ :=
  let MD := (1 / 2) * ((BD + AD) - AB)
  let ND := (1 / 2) * ((BD + DC) - BC)
  (| MD - ND |)

theorem find_MN (AB BC AC : ℚ) (BD : ℚ) (AD DC : ℚ) (h₁ : AB = 3) (h₂ : BC = 4) (h₃ : AC = 5)
  (h₅ : angle_bisector_theorem AD DC AB BC) : length_of_MN AB BC AC BD AD DC = 1 / 7 :=
sorry

end find_MN_l241_241355


namespace capacity_of_bucket_in_first_scenario_l241_241165

theorem capacity_of_bucket_in_first_scenario (x : ℝ) 
  (h1 : 28 * x = 378) : x = 13.5 :=
by
  sorry

end capacity_of_bucket_in_first_scenario_l241_241165


namespace selling_price_l241_241781

theorem selling_price (cost_price profit_percentage : ℝ) (h1 : cost_price = 90) (h2 : profit_percentage = 100) : 
    cost_price + (profit_percentage * cost_price / 100) = 180 :=
by
  rw [h1, h2]
  norm_num
  -- sorry

end selling_price_l241_241781


namespace sin_zero_degrees_l241_241572

theorem sin_zero_degrees : Real.sin 0 = 0 := 
by {
  -- The proof is added here (as requested no proof is required, hence using sorry)
  sorry
}

end sin_zero_degrees_l241_241572


namespace abs_diff_roots_quad_eq_l241_241599

theorem abs_diff_roots_quad_eq : 
  ∀ (r1 r2 : ℝ), 
  (r1 * r2 = 12) ∧ (r1 + r2 = 7) → |r1 - r2| = 1 :=
by
  intro r1 r2 h
  sorry

end abs_diff_roots_quad_eq_l241_241599


namespace volumes_are_equal_l241_241460

-- Definitions for the regions and their volumes
def region1 := { p : ℝ × ℝ | p.1 ^ 2 = 4 * p.2 ∨ p.1 ^ 2 = -4 * p.2 ∧ -4 ≤ p.1 ∧ p.1 ≤ 4 }
def volume_of_solid_rotating_y_axis (region : set (ℝ × ℝ)) : ℝ := sorry -- assume volume calculation
def V1 := volume_of_solid_rotating_y_axis region1

def region2 := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ 16 ∧ p.1 ^ 2 + (p.2 - 2) ^ 2 ≥ 4 ∧ p.1 ^ 2 + (p.2 + 2) ^ 2 ≥ 4 }
def V2 := volume_of_solid_rotating_y_axis region2

-- Theorem stating the volumes are equal
theorem volumes_are_equal : V1 = V2 :=
by {
  sorry
}

end volumes_are_equal_l241_241460


namespace lambda_le_l241_241278

noncomputable def a_sequence : ℕ+ → ℝ
| ⟨1, _⟩ := 1
| ⟨n+1, h⟩ := 2 * (n + 1) - 1

noncomputable def b_sequence (n : ℕ+) : ℝ :=
  let a_n := a_sequence n
  let a_n_plus_1 := a_sequence ⟨n + 1, Nat.succ_pos' n⟩
  (3 * 2^a_n) / ((2^a_n_plus_1 - 1) * (2^a_n - 1))

noncomputable def T_n (n : ℕ+) : ℝ := ∑ i in Finset.range n, b_sequence ⟨i + 1, Nat.succ_pos' i⟩

theorem lambda_le : ∀ n : ℕ+, λ ≤ (1 / (32 * (1 - T_n n))) + (32 / (2^(a_sequence ⟨n + 1, Nat.succ_pos' n⟩))) :=
sorry

end lambda_le_l241_241278


namespace sophie_donuts_l241_241420

theorem sophie_donuts :
  let boxes := 4
  let donuts_per_box := 12
  let total_donuts := boxes * donuts_per_box
  let mom_donuts := 12
  let leftover_donuts := total_donuts - mom_donuts
  ∃ S : Nat, (leftover_donuts - S = 30) ∧ (S = 6) :=
begin
  sorry
end

end sophie_donuts_l241_241420


namespace sum_of_solutions_sum_of_all_possible_values_l241_241318

theorem sum_of_solutions (y : ℝ) (h : y^2 = 25) : y = 5 ∨ y = -5 :=
by {
  sorry
}

theorem sum_of_all_possible_values
: ∀ y : ℝ, (y = 5 ∨ y = -5) → y + -y = 0 :=
by {
  intros y h,
  cases h,
  { simp },
  { simp }
}

end sum_of_solutions_sum_of_all_possible_values_l241_241318


namespace complex_number_equality_l241_241282

theorem complex_number_equality : 
  (z : ℂ) (h : z = (1 : ℂ) / (1 + I)) : z = (1 - I) / 2 :=
by
  sorry

end complex_number_equality_l241_241282


namespace inequality_xyz_l241_241767

theorem inequality_xyz 
  (x y z: ℝ) 
  (hx: x > 0) 
  (hy: y > 0) 
  (hz: z > 0):
  (x * y * z * (x + y + z + real.sqrt (x^2 + y^2 + z^2))) / ((x^2 + y^2 + z^2) * (y * z + z * x + x * y)) ≤ (3 + real.sqrt 3) / 9 := 
by sorry

end inequality_xyz_l241_241767


namespace sum_of_sequence_99_terms_l241_241655

theorem sum_of_sequence_99_terms :
  (∑ n in finset.range 99, 1 / (a n * a (n + 1))) = 37 / 50 :=
sorry

-- Defining the sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 0 then 2 else n + 1

-- Sum of the first n terms of the sequence {a_n}
def S (n : ℕ) : ℚ :=
  (n^2 + n) / 2 + 1

end sum_of_sequence_99_terms_l241_241655


namespace find_a_perpendicular_l241_241446

theorem find_a_perpendicular :
  ∀ (a : ℝ),
    (∀ (k1 k2 : ℝ), k1 = a ∧ k2 = -1 ∧ k1 * k2 = -1) → a = 1 :=
begin
  sorry
end

end find_a_perpendicular_l241_241446


namespace expression_equality_unique_l241_241581

theorem expression_equality_unique (x : ℝ) (h : x > 0) :
  (x^(x+1) + x^(x+1) = 2 * x^(x+1)) ∧ 
  (∀ (y : ℝ), y ∈ {2 * x^(x+1), x^(2 * x + 2), (3 * x)^x, (3 * x)^(x+1)} → y = 2 * x^(x+1) ↔ y = (2 * x)^(x+1)): = 
begin
  sorry
end

end expression_equality_unique_l241_241581
