import Mathlib

namespace NUMINAMATH_GPT_ratio_and_tangent_l969_96979

-- Definitions for the problem
def acute_triangle (A B C : Point) : Prop := 
  -- acute angles condition
  sorry

def is_diameter (A B C D : Point) : Prop := 
  -- D is midpoint of BC condition
  sorry

def divide_in_half (A B C : Point) (D : Point) : Prop := 
  -- D divides BC in half condition
  sorry

def divide_in_ratio (A B C : Point) (D : Point) (ratio : ℚ) : Prop := 
  -- D divides AC in the given ratio condition
  sorry

def tan (angle : ℝ) : ℝ := 
  -- Tangent function
  sorry

def angle (A B C : Point) : ℝ := 
  -- Angle at B of triangle ABC
  sorry

-- The statement of the problem in Lean
theorem ratio_and_tangent (A B C D : Point) :
  acute_triangle A B C →
  is_diameter A B C D →
  divide_in_half A B C D →
  (divide_in_ratio A B C D (1 / 3) ↔ tan (angle A B C) = 2 * tan (angle A C B)) :=
by sorry

end NUMINAMATH_GPT_ratio_and_tangent_l969_96979


namespace NUMINAMATH_GPT_multiple_of_3_l969_96987

theorem multiple_of_3 (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 2^n + 1) : 3 ∣ n :=
sorry

end NUMINAMATH_GPT_multiple_of_3_l969_96987


namespace NUMINAMATH_GPT_fido_leash_yard_reach_area_product_l969_96926

noncomputable def fido_leash_yard_fraction : ℝ :=
  let a := 2 + Real.sqrt 2
  let b := 8
  a * b

theorem fido_leash_yard_reach_area_product :
  ∃ (a b : ℝ), 
  (fido_leash_yard_fraction = (a * b)) ∧ 
  (1 > a) ∧ -- Regular Octagon computation constraints
  (b = 8) ∧ 
  a = 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_fido_leash_yard_reach_area_product_l969_96926


namespace NUMINAMATH_GPT_total_cars_for_sale_l969_96949

-- Define the conditions given in the problem
def salespeople : Nat := 10
def cars_per_salesperson_per_month : Nat := 10
def months : Nat := 5

-- Statement to prove the total number of cars for sale
theorem total_cars_for_sale : (salespeople * cars_per_salesperson_per_month) * months = 500 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_cars_for_sale_l969_96949


namespace NUMINAMATH_GPT_remaining_budget_correct_l969_96906

def cost_item1 := 13
def cost_item2 := 24
def last_year_remaining_budget := 6
def this_year_budget := 50

theorem remaining_budget_correct :
    (last_year_remaining_budget + this_year_budget - (cost_item1 + cost_item2) = 19) :=
by
  -- This is the statement only, with the proof omitted
  sorry

end NUMINAMATH_GPT_remaining_budget_correct_l969_96906


namespace NUMINAMATH_GPT_coffee_processing_completed_l969_96931

-- Define the initial conditions
def CoffeeBeansProcessed (m n : ℕ) : Prop :=
  let mass: ℝ := 1
  let days_single_machine: ℕ := 5
  let days_both_machines: ℕ := 4
  let half_mass: ℝ := mass / 2
  let total_ground_by_June_10 := (days_single_machine * m + days_both_machines * (m + n)) = half_mass
  total_ground_by_June_10

-- Define the final proof problem
theorem coffee_processing_completed (m n : ℕ) (h: CoffeeBeansProcessed m n) : ∃ d : ℕ, d = 15 := by
  -- Processed in 15 working days
  sorry

end NUMINAMATH_GPT_coffee_processing_completed_l969_96931


namespace NUMINAMATH_GPT_ella_incorrect_answers_l969_96960

theorem ella_incorrect_answers
  (marion_score : ℕ)
  (ella_score : ℕ)
  (total_items : ℕ)
  (h1 : marion_score = 24)
  (h2 : marion_score = (ella_score / 2) + 6)
  (h3 : total_items = 40) : 
  total_items - ella_score = 4 :=
by
  sorry

end NUMINAMATH_GPT_ella_incorrect_answers_l969_96960


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l969_96913

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
    (h₁ : ∀ n, a (n + 1) = a n + d)
    (h₂ : a 3 + a 5 + a 7 + a 9 + a 11 = 20) : a 1 + a 13 = 8 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l969_96913


namespace NUMINAMATH_GPT_largest_integer_x_divisible_l969_96956

theorem largest_integer_x_divisible (x : ℤ) : 
  (∃ x : ℤ, (x^2 + 3 * x + 8) % (x - 2) = 0 ∧ x ≤ 1) → x = 1 :=
sorry

end NUMINAMATH_GPT_largest_integer_x_divisible_l969_96956


namespace NUMINAMATH_GPT_solve_equation_l969_96984

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x^2 - 1 ≠ 0) : (x / (x - 1) = 2 / (x^2 - 1)) → (x = -2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l969_96984


namespace NUMINAMATH_GPT_vector_sum_l969_96995

-- Define the vectors a and b according to the conditions.
def a : (ℝ × ℝ) := (2, 1)
def b : (ℝ × ℝ) := (-3, 4)

-- Prove that the vector sum a + b is (-1, 5).
theorem vector_sum : (a.1 + b.1, a.2 + b.2) = (-1, 5) :=
by
  -- include the proof later
  sorry

end NUMINAMATH_GPT_vector_sum_l969_96995


namespace NUMINAMATH_GPT_fraction_zero_solution_l969_96937

theorem fraction_zero_solution (x : ℝ) (h1 : x - 5 = 0) (h2 : 4 * x^2 - 1 ≠ 0) : x = 5 :=
by {
  sorry -- The proof
}

end NUMINAMATH_GPT_fraction_zero_solution_l969_96937


namespace NUMINAMATH_GPT_value_of_f_at_2_l969_96981

def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem value_of_f_at_2 : f 2 = -2 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_f_at_2_l969_96981


namespace NUMINAMATH_GPT_tulips_for_each_eye_l969_96993

theorem tulips_for_each_eye (R : ℕ) : 2 * R + 18 + 9 * 18 = 196 → R = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tulips_for_each_eye_l969_96993


namespace NUMINAMATH_GPT_f_value_third_quadrant_l969_96998

noncomputable def f (α : ℝ) : ℝ :=
  (Real.cos (Real.pi / 2 + α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.sin (-Real.pi - α) * Real.sin (3 * Real.pi / 2 + α))

theorem f_value_third_quadrant (α : ℝ) (h1 : (3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)) (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 :=
sorry

end NUMINAMATH_GPT_f_value_third_quadrant_l969_96998


namespace NUMINAMATH_GPT_range_of_a_l969_96955

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x = 3 ∧ 3 * x - (a * x + 1) / 2 < 4 * x / 3) → a > 3 :=
by
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  sorry

end NUMINAMATH_GPT_range_of_a_l969_96955


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_6_l969_96950

-- Define the smallest four-digit number
def smallest_four_digit_number := 1000

-- Define divisibility conditions
def divisible_by_2 (n : Nat) := n % 2 = 0
def divisible_by_3 (n : Nat) := n % 3 = 0
def divisible_by_6 (n : Nat) := divisible_by_2 n ∧ divisible_by_3 n

-- Prove that the smallest four-digit number divisible by 6 is 1002
theorem smallest_four_digit_divisible_by_6 : ∃ n : Nat, n ≥ smallest_four_digit_number ∧ divisible_by_6 n ∧ ∀ m : Nat, m ≥ smallest_four_digit_number ∧ divisible_by_6 m → n ≤ m :=
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_6_l969_96950


namespace NUMINAMATH_GPT_combined_weight_of_three_parcels_l969_96947

theorem combined_weight_of_three_parcels (x y z : ℕ)
  (h1 : x + y = 112) (h2 : y + z = 146) (h3 : z + x = 132) :
  x + y + z = 195 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_of_three_parcels_l969_96947


namespace NUMINAMATH_GPT_common_ratio_geometric_series_l969_96983

theorem common_ratio_geometric_series 
  (a : ℚ) (b : ℚ) (r : ℚ)
  (h_a : a = 4 / 5)
  (h_b : b = -5 / 12)
  (h_r : r = b / a) :
  r = -25 / 48 :=
by sorry

end NUMINAMATH_GPT_common_ratio_geometric_series_l969_96983


namespace NUMINAMATH_GPT_max_distinct_numbers_example_l969_96916

def max_distinct_numbers (a b c d e : ℕ) : ℕ := sorry

theorem max_distinct_numbers_example
  (A B : ℕ) :
  max_distinct_numbers 100 200 400 A B = 64 := sorry

end NUMINAMATH_GPT_max_distinct_numbers_example_l969_96916


namespace NUMINAMATH_GPT_children_getting_on_bus_l969_96986

theorem children_getting_on_bus (a b c: ℕ) (ha : a = 64) (hb : b = 78) (hc : c = b - a) : c = 14 :=
by
  sorry

end NUMINAMATH_GPT_children_getting_on_bus_l969_96986


namespace NUMINAMATH_GPT_yen_per_pound_l969_96988

theorem yen_per_pound 
  (pounds_initial : ℕ) 
  (euros : ℕ) 
  (yen_initial : ℕ) 
  (pounds_per_euro : ℕ) 
  (yen_total : ℕ) 
  (hp : pounds_initial = 42) 
  (he : euros = 11) 
  (hy : yen_initial = 3000) 
  (hpe : pounds_per_euro = 2) 
  (hy_total : yen_total = 9400) 
  : (yen_total - yen_initial) / (pounds_initial + euros * pounds_per_euro) = 100 := 
by
  sorry

end NUMINAMATH_GPT_yen_per_pound_l969_96988


namespace NUMINAMATH_GPT_smallest_value_of_3a_plus_2_l969_96990

variable (a : ℝ)

theorem smallest_value_of_3a_plus_2 (h : 5 * a^2 + 7 * a + 2 = 1) : 3 * a + 2 = -1 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_3a_plus_2_l969_96990


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l969_96970

theorem necessary_but_not_sufficient (x: ℝ) :
  (1 < x ∧ x < 4) → (1 < x ∧ x < 3) := by
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l969_96970


namespace NUMINAMATH_GPT_relatively_prime_pair_count_l969_96991

theorem relatively_prime_pair_count :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m + n = 190 ∧ Nat.gcd m n = 1) →
  (∃! k : ℕ, k = 26) :=
by
  sorry

end NUMINAMATH_GPT_relatively_prime_pair_count_l969_96991


namespace NUMINAMATH_GPT_crayons_per_box_l969_96980

theorem crayons_per_box (total_crayons : ℝ) (total_boxes : ℝ) (h1 : total_crayons = 7.0) (h2 : total_boxes = 1.4) : total_crayons / total_boxes = 5 :=
by
  sorry

end NUMINAMATH_GPT_crayons_per_box_l969_96980


namespace NUMINAMATH_GPT_solve_for_x_l969_96966

theorem solve_for_x (x y : ℤ) (h1 : x + 2 * y = 10) (h2 : y = 3) : x = 4 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l969_96966


namespace NUMINAMATH_GPT_op_4_neg3_eq_neg28_l969_96917

def op (x y : Int) : Int := x * (y + 2) + 2 * x * y

theorem op_4_neg3_eq_neg28 : op 4 (-3) = -28 := by
  sorry

end NUMINAMATH_GPT_op_4_neg3_eq_neg28_l969_96917


namespace NUMINAMATH_GPT_cyclic_quadrilateral_sides_equal_l969_96953

theorem cyclic_quadrilateral_sides_equal
  (A B C D P : ℝ) -- Points represented as reals for simplicity
  (AB CD BC AD : ℝ) -- Lengths of sides AB, CD, BC, AD
  (a b c d e θ : ℝ) -- Various lengths and angle as given in the solution
  (h1 : a + e = b + c + d)
  (h2 : (1 / 2) * a * e * Real.sin θ = (1 / 2) * b * e * Real.sin θ + (1 / 2) * c * d * Real.sin θ) :
  c = e ∨ d = e := sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_sides_equal_l969_96953


namespace NUMINAMATH_GPT_difference_between_numbers_l969_96967

theorem difference_between_numbers :
  ∃ X Y : ℕ, 
    100 ≤ X ∧ X < 1000 ∧
    100 ≤ Y ∧ Y < 1000 ∧
    X + Y = 999 ∧
    1000 * X + Y = 6 * (1000 * Y + X) ∧
    (X - Y = 715 ∨ Y - X = 715) :=
by
  sorry

end NUMINAMATH_GPT_difference_between_numbers_l969_96967


namespace NUMINAMATH_GPT_calc_expression_l969_96948

theorem calc_expression : (113^2 - 104^2) / 9 = 217 := by
  sorry

end NUMINAMATH_GPT_calc_expression_l969_96948


namespace NUMINAMATH_GPT_fraction_of_salary_on_rent_l969_96920

theorem fraction_of_salary_on_rent
  (S : ℝ) (food_fraction : ℝ) (clothes_fraction : ℝ) (remaining_amount : ℝ) (approx_salary : ℝ)
  (food_fraction_eq : food_fraction = 1 / 5)
  (clothes_fraction_eq : clothes_fraction = 3 / 5)
  (remaining_amount_eq : remaining_amount = 19000)
  (approx_salary_eq : approx_salary = 190000) :
  ∃ (H : ℝ), H = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_salary_on_rent_l969_96920


namespace NUMINAMATH_GPT_volume_of_solid_rotation_l969_96934

noncomputable def volume_of_solid := 
  (∫ y in (0:ℝ)..(1:ℝ), (y^(2/3) - y^2)) * Real.pi 

theorem volume_of_solid_rotation :
  volume_of_solid = (4 * Real.pi / 15) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_solid_rotation_l969_96934


namespace NUMINAMATH_GPT_repeating_sequence_length_1_over_221_l969_96974

theorem repeating_sequence_length_1_over_221 : ∃ n : ℕ, (10 ^ n ≡ 1 [MOD 221]) ∧ (∀ m : ℕ, (10 ^ m ≡ 1 [MOD 221]) → (n ≤ m)) ∧ n = 48 :=
by
  sorry

end NUMINAMATH_GPT_repeating_sequence_length_1_over_221_l969_96974


namespace NUMINAMATH_GPT_sample_size_correct_l969_96977

def total_students (freshmen sophomores juniors : ℕ) : ℕ :=
  freshmen + sophomores + juniors

def sample_size (total : ℕ) (prob : ℝ) : ℝ :=
  total * prob

theorem sample_size_correct (f : ℕ) (s : ℕ) (j : ℕ) (p : ℝ) (h_f : f = 400) (h_s : s = 320) (h_j : j = 280) (h_p : p = 0.2) :
  sample_size (total_students f s j) p = 200 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_correct_l969_96977


namespace NUMINAMATH_GPT_number_of_authors_l969_96969

/-- Define the number of books each author has and the total number of books. -/
def books_per_author : ℕ := 33
def total_books : ℕ := 198

/-- Main theorem stating that the number of authors Jack has is derived by dividing total books by the number of books per author. -/
theorem number_of_authors (n : ℕ) (h : total_books = n * books_per_author) : n = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_authors_l969_96969


namespace NUMINAMATH_GPT_find_b_l969_96929

noncomputable def P (x a b c : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + c

theorem find_b (a b c : ℝ) (h1: P 0 a b c = 12)
  (h2: (-c / 2) * 1 = -6)
  (h3: (2 + a + b + c) = -6)
  (h4: a + b + 14 = -6) : b = -56 :=
sorry

end NUMINAMATH_GPT_find_b_l969_96929


namespace NUMINAMATH_GPT_solution_set_f_pos_l969_96928

noncomputable def f : ℝ → ℝ := sorry -- Definition of the function f(x)

-- Conditions
axiom h1 : ∀ x, f (-x) = -f x     -- f(x) is odd
axiom h2 : f 2 = 0                -- f(2) = 0
axiom h3 : ∀ x > 0, 2 * f x + x * (deriv f x) > 0 -- 2f(x) + xf'(x) > 0 for x > 0

-- Theorem to prove
theorem solution_set_f_pos : { x : ℝ | f x > 0 } = { x : ℝ | x > 2 ∨ (-2 < x ∧ x < 0) } :=
sorry

end NUMINAMATH_GPT_solution_set_f_pos_l969_96928


namespace NUMINAMATH_GPT_y_at_x_eq_120_l969_96910

@[simp] def custom_op (a b : ℕ) : ℕ := List.prod (List.map (λ i => a + i) (List.range b))

theorem y_at_x_eq_120 {x y : ℕ}
  (h1 : custom_op x (custom_op y 2) = 420)
  (h2 : x = 4)
  (h3 : y = 2) :
  custom_op y x = 120 := by
  sorry

end NUMINAMATH_GPT_y_at_x_eq_120_l969_96910


namespace NUMINAMATH_GPT_volleyball_team_starters_l969_96939

-- Define the team and the triplets
def total_players : ℕ := 14
def triplet_count : ℕ := 3
def remaining_players : ℕ := total_players - triplet_count

-- Define the binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the problem
theorem volleyball_team_starters : 
  C total_players 6 - C remaining_players 3 = 2838 :=
by sorry

end NUMINAMATH_GPT_volleyball_team_starters_l969_96939


namespace NUMINAMATH_GPT_benny_kids_l969_96940

theorem benny_kids (total_money : ℕ) (cost_per_apple : ℕ) (apples_per_kid : ℕ) (total_apples : ℕ) (kids : ℕ) :
  total_money = 360 →
  cost_per_apple = 4 →
  apples_per_kid = 5 →
  total_apples = total_money / cost_per_apple →
  kids = total_apples / apples_per_kid →
  kids = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_benny_kids_l969_96940


namespace NUMINAMATH_GPT_determinant_matrix_A_l969_96952

open Matrix

def matrix_A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, 0, -2], ![1, 3, 4], ![0, -1, 1]]

theorem determinant_matrix_A :
  det matrix_A = 33 :=
by
  sorry

end NUMINAMATH_GPT_determinant_matrix_A_l969_96952


namespace NUMINAMATH_GPT_negation_of_proposition_l969_96903

theorem negation_of_proposition (a b : ℝ) : 
  ¬ (∀ a b : ℝ, (a = 1 → a + b = 1)) ↔ (∃ a b : ℝ, a = 1 ∧ a + b ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l969_96903


namespace NUMINAMATH_GPT_max_reflections_l969_96999

theorem max_reflections (angle_increase : ℕ := 10) (max_angle : ℕ := 90) :
  ∃ n : ℕ, 10 * n ≤ max_angle ∧ ∀ m : ℕ, (10 * (m + 1) > max_angle → m < n) := 
sorry

end NUMINAMATH_GPT_max_reflections_l969_96999


namespace NUMINAMATH_GPT_initial_sum_l969_96919

theorem initial_sum (P : ℝ) (compound_interest : ℝ) (r1 r2 r3 r4 r5 : ℝ) 
  (h1 : r1 = 0.06) (h2 : r2 = 0.08) (h3 : r3 = 0.07) (h4 : r4 = 0.09) (h5 : r5 = 0.10)
  (interest_sum : compound_interest = 4016.25) :
  P = 4016.25 / ((1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5) - 1) :=
by
  sorry

end NUMINAMATH_GPT_initial_sum_l969_96919


namespace NUMINAMATH_GPT_capsule_cost_difference_l969_96978

theorem capsule_cost_difference :
  let cost_per_capsule_r := 6.25 / 250
  let cost_per_capsule_t := 3.00 / 100
  cost_per_capsule_t - cost_per_capsule_r = 0.005 := by
  sorry

end NUMINAMATH_GPT_capsule_cost_difference_l969_96978


namespace NUMINAMATH_GPT_min_abs_sum_l969_96943

theorem min_abs_sum (a b c : ℝ) (h₁ : a + b + c = -2) (h₂ : a * b * c = -4) :
  ∃ (m : ℝ), m = min (abs a + abs b + abs c) 6 :=
sorry

end NUMINAMATH_GPT_min_abs_sum_l969_96943


namespace NUMINAMATH_GPT_dan_violet_marbles_l969_96904

def InitMarbles : ℕ := 128
def MarblesGivenMary : ℕ := 24
def MarblesGivenPeter : ℕ := 16
def MarblesReceived : ℕ := 10

def FinalMarbles : ℕ := InitMarbles - MarblesGivenMary - MarblesGivenPeter + MarblesReceived

theorem dan_violet_marbles : FinalMarbles = 98 := 
by 
  sorry

end NUMINAMATH_GPT_dan_violet_marbles_l969_96904


namespace NUMINAMATH_GPT_hyperbola_through_focus_and_asymptotes_l969_96942

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 1

def asymptotes_holds (x y : ℝ) : Prop :=
  (x + y = 0) ∨ (x - y = 0)

theorem hyperbola_through_focus_and_asymptotes :
  hyperbola parabola_focus.1 parabola_focus.2 ∧ asymptotes_holds parabola_focus.1 parabola_focus.2 :=
sorry

end NUMINAMATH_GPT_hyperbola_through_focus_and_asymptotes_l969_96942


namespace NUMINAMATH_GPT_range_of_p_l969_96944

def p (x : ℝ) : ℝ := x^6 + 6 * x^3 + 9

theorem range_of_p : Set.Ici 9 = { y | ∃ x ≥ 0, p x = y } :=
by
  -- We skip the proof to only provide the statement as requested.
  sorry

end NUMINAMATH_GPT_range_of_p_l969_96944


namespace NUMINAMATH_GPT_monthly_increase_per_ticket_l969_96900

variable (x : ℝ)

theorem monthly_increase_per_ticket
    (initial_premium : ℝ := 50)
    (percent_increase_per_accident : ℝ := 0.10)
    (tickets : ℕ := 3)
    (final_premium : ℝ := 70) :
    initial_premium * (1 + percent_increase_per_accident) + tickets * x = final_premium → x = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_monthly_increase_per_ticket_l969_96900


namespace NUMINAMATH_GPT_b_n_expression_l969_96912

-- Define sequence a_n as an arithmetic sequence with given conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + d * (n - 1)

-- Define the conditions for the sequence a_n
def a_conditions (a : ℕ → ℤ) : Prop :=
  a 2 = 8 ∧ a 8 = 26

-- Define the new sequence b_n based on the terms of a_n
def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  a (3^n)

theorem b_n_expression (a : ℕ → ℤ) (n : ℕ)
  (h_arith : is_arithmetic_sequence a)
  (h_conditions : a_conditions a) :
  b a n = 3^(n + 1) + 2 := 
sorry

end NUMINAMATH_GPT_b_n_expression_l969_96912


namespace NUMINAMATH_GPT_find_divisor_l969_96964

/-- Given a dividend of 15698, a quotient of 89, and a remainder of 14, find the divisor. -/
theorem find_divisor :
  ∃ D : ℕ, 15698 = 89 * D + 14 ∧ D = 176 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l969_96964


namespace NUMINAMATH_GPT_intersection_point_sum_l969_96997

noncomputable def h : ℝ → ℝ := sorry
noncomputable def j : ℝ → ℝ := sorry

axiom h2 : h 2 = 2
axiom j2 : j 2 = 2
axiom h4 : h 4 = 6
axiom j4 : j 4 = 6
axiom h6 : h 6 = 12
axiom j6 : j 6 = 12
axiom h8 : h 8 = 12
axiom j8 : j 8 = 12

theorem intersection_point_sum :
  (∃ x, h (x + 2) = j (2 * x)) →
  (h (2 + 2) = j (2 * 2) ∨ h (4 + 2) = j (2 * 4)) →
  (h (4) = 6 ∧ j (4) = 6 ∧ h 6 = 12 ∧ j 8 = 12) →
  (∃ x, (x = 2 ∧ (x + h (x + 2) = 8) ∨ x = 4 ∧ (x + h (x + 2) = 16))) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_sum_l969_96997


namespace NUMINAMATH_GPT_range_of_k_for_distinct_real_roots_l969_96901

theorem range_of_k_for_distinct_real_roots (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 - 2 * x + 1 = 0) → (k < 2 ∧ k ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_for_distinct_real_roots_l969_96901


namespace NUMINAMATH_GPT_find_other_number_l969_96941

theorem find_other_number (a b lcm hcf : ℕ) (h_lcm : lcm = 2310) (h_hcf : hcf = 61) (h_first_number : a = 210) :
  a * b = lcm * hcf → b = 671 :=
by 
  -- setup
  sorry

end NUMINAMATH_GPT_find_other_number_l969_96941


namespace NUMINAMATH_GPT_solve_for_a_minus_b_l969_96957

theorem solve_for_a_minus_b (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 7) (h3 : |a + b| = a + b) : a - b = -2 := 
sorry

end NUMINAMATH_GPT_solve_for_a_minus_b_l969_96957


namespace NUMINAMATH_GPT_inequality_solution_l969_96992

theorem inequality_solution
  : {x : ℝ | (x^2 / (x + 2)^2) ≥ 0} = {x : ℝ | x ≠ -2} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l969_96992


namespace NUMINAMATH_GPT_find_divisor_l969_96975

variable (dividend quotient remainder divisor : ℕ)

theorem find_divisor (h1 : dividend = 52) (h2 : quotient = 16) (h3 : remainder = 4) (h4 : dividend = divisor * quotient + remainder) : 
  divisor = 3 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l969_96975


namespace NUMINAMATH_GPT_line_slope_intercept_l969_96905

theorem line_slope_intercept :
  ∃ k b, (∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 → y = k * x + b) ∧ k = 2/3 ∧ b = 2 :=
by
  sorry

end NUMINAMATH_GPT_line_slope_intercept_l969_96905


namespace NUMINAMATH_GPT_smallest_N_l969_96989

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end NUMINAMATH_GPT_smallest_N_l969_96989


namespace NUMINAMATH_GPT_speed_ratio_correct_l969_96925

noncomputable def boat_speed_still_water := 12 -- Boat's speed in still water (in mph)
noncomputable def current_speed := 4 -- Current speed of the river (in mph)

-- Calculate the downstream speed
noncomputable def downstream_speed := boat_speed_still_water + current_speed

-- Calculate the upstream speed
noncomputable def upstream_speed := boat_speed_still_water - current_speed

-- Assume a distance for the trip (1 mile each up and down)
noncomputable def distance := 1

-- Calculate time for downstream
noncomputable def time_downstream := distance / downstream_speed

-- Calculate time for upstream
noncomputable def time_upstream := distance / upstream_speed

-- Calculate total time for the round trip
noncomputable def total_time := time_downstream + time_upstream

-- Calculate total distance for the round trip
noncomputable def total_distance := 2 * distance

-- Calculate the average speed for the round trip
noncomputable def avg_speed_trip := total_distance / total_time

-- Calculate the ratio of average speed to speed in still water
noncomputable def speed_ratio := avg_speed_trip / boat_speed_still_water

theorem speed_ratio_correct : speed_ratio = 8/9 := by
  sorry

end NUMINAMATH_GPT_speed_ratio_correct_l969_96925


namespace NUMINAMATH_GPT_count_possible_third_side_lengths_l969_96965

theorem count_possible_third_side_lengths : ∀ (n : ℤ), 2 < n ∧ n < 14 → ∃ s : Finset ℤ, s.card = 11 ∧ ∀ x ∈ s, 2 < x ∧ x < 14 := by
  sorry

end NUMINAMATH_GPT_count_possible_third_side_lengths_l969_96965


namespace NUMINAMATH_GPT_P_and_S_could_not_be_fourth_l969_96972

-- Define the relationships between the runners using given conditions
variables (P Q R S T U : ℕ)

axiom P_beats_Q : P < Q
axiom Q_beats_R : Q < R
axiom R_beats_S : R < S
axiom T_after_P_before_R : P < T ∧ T < R
axiom U_before_R_after_S : S < U ∧ U < R

-- Prove that P and S could not be fourth
theorem P_and_S_could_not_be_fourth : ¬((Q < U ∧ U < P) ∨ (Q > S ∧ S < P)) :=
by sorry

end NUMINAMATH_GPT_P_and_S_could_not_be_fourth_l969_96972


namespace NUMINAMATH_GPT_rem_sum_a_b_c_l969_96961

theorem rem_sum_a_b_c (a b c : ℤ) (h1 : a * b * c ≡ 1 [ZMOD 5]) (h2 : 3 * c ≡ 1 [ZMOD 5]) (h3 : 4 * b ≡ 1 + b [ZMOD 5]) : 
  (a + b + c) % 5 = 3 := by 
  sorry

end NUMINAMATH_GPT_rem_sum_a_b_c_l969_96961


namespace NUMINAMATH_GPT_gold_copper_alloy_ratio_l969_96930

theorem gold_copper_alloy_ratio {G C A : ℝ} (hC : C = 9) (hA : A = 18) (hG : 9 < G ∧ G < 18) :
  ∃ x : ℝ, 18 = x * G + (1 - x) * 9 :=
by
  sorry

end NUMINAMATH_GPT_gold_copper_alloy_ratio_l969_96930


namespace NUMINAMATH_GPT_cuboid_properties_l969_96908

-- Given definitions from conditions
variables (l w h : ℝ)
variables (h_edge_length : 4 * (l + w + h) = 72)
variables (h_ratio : l / w = 3 / 2 ∧ w / h = 2 / 1)

-- Define the surface area and volume based on the given conditions
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)
def volume (l w h : ℝ) : ℝ := l * w * h

-- Theorem statement
theorem cuboid_properties :
  surface_area l w h = 198 ∧ volume l w h = 162 :=
by
  -- Code to provide the proof goes here
  sorry

end NUMINAMATH_GPT_cuboid_properties_l969_96908


namespace NUMINAMATH_GPT_gcd_three_digit_numbers_l969_96927

theorem gcd_three_digit_numbers (a b c : ℕ) (h1 : b = a + 1) (h2 : c = a + 2) :
  ∃ k, (∀ n, n = 100 * a + 10 * b + c + 100 * c + 10 * b + a → n = 212 * k) :=
by
  sorry

end NUMINAMATH_GPT_gcd_three_digit_numbers_l969_96927


namespace NUMINAMATH_GPT_polygon_sides_l969_96994

theorem polygon_sides (h : 1440 = (n - 2) * 180) : n = 10 := 
by {
  -- Here, the proof would show the steps to solve the equation h and confirm n = 10
  sorry
}

end NUMINAMATH_GPT_polygon_sides_l969_96994


namespace NUMINAMATH_GPT_discount_percentage_l969_96962

variable (P : ℝ) (r : ℝ) (S : ℝ)

theorem discount_percentage (hP : P = 20) (hr : r = 30 / 100) (hS : S = 13) :
  (P * (1 + r) - S) / (P * (1 + r)) * 100 = 50 := 
sorry

end NUMINAMATH_GPT_discount_percentage_l969_96962


namespace NUMINAMATH_GPT_value_of_m_l969_96922

theorem value_of_m : (∀ x : ℝ, (1 + 2 * x) ^ 3 = 1 + 6 * x + m * x ^ 2 + 8 * x ^ 3 → m = 12) := 
by {
  -- This is where the proof would go
  sorry
}

end NUMINAMATH_GPT_value_of_m_l969_96922


namespace NUMINAMATH_GPT_proof_calculate_expr_l969_96938

def calculate_expr : Prop :=
  (4 + 4 + 6) / 3 - 2 / 3 = 4

theorem proof_calculate_expr : calculate_expr := 
by 
  sorry

end NUMINAMATH_GPT_proof_calculate_expr_l969_96938


namespace NUMINAMATH_GPT_total_logs_in_both_stacks_l969_96918

-- Define the number of logs in the first stack
def first_stack_logs : Nat :=
  let bottom_row := 15
  let top_row := 4
  let number_of_terms := bottom_row - top_row + 1
  let average_logs := (bottom_row + top_row) / 2
  average_logs * number_of_terms

-- Define the number of logs in the second stack
def second_stack_logs : Nat :=
  let bottom_row := 5
  let top_row := 10
  let number_of_terms := top_row - bottom_row + 1
  let average_logs := (bottom_row + top_row) / 2
  average_logs * number_of_terms

-- Prove the total number of logs in both stacks
theorem total_logs_in_both_stacks : first_stack_logs + second_stack_logs = 159 := by
  sorry

end NUMINAMATH_GPT_total_logs_in_both_stacks_l969_96918


namespace NUMINAMATH_GPT_rearrangements_of_abcde_l969_96976

def is_adjacent (c1 c2 : Char) : Bool :=
  (c1 == 'a' ∧ c2 == 'b') ∨ 
  (c1 == 'b' ∧ c1 == 'a') ∨ 
  (c1 == 'b' ∧ c2 == 'c') ∨ 
  (c1 == 'c' ∧ c2 == 'b') ∨ 
  (c1 == 'c' ∧ c2 == 'd') ∨ 
  (c1 == 'd' ∧ c2 == 'c') ∨ 
  (c1 == 'd' ∧ c2 == 'e') ∨ 
  (c1 == 'e' ∧ c2 == 'd')

def is_valid_rearrangement (lst : List Char) : Bool :=
  match lst with
  | [] => true
  | [_] => true
  | c1 :: c2 :: rest => 
    ¬is_adjacent c1 c2 ∧ is_valid_rearrangement (c2 :: rest)

def count_valid_rearrangements (chars : List Char) : Nat :=
  chars.permutations.filter is_valid_rearrangement |>.length

theorem rearrangements_of_abcde : count_valid_rearrangements ['a', 'b', 'c', 'd', 'e'] = 8 := 
by
  sorry

end NUMINAMATH_GPT_rearrangements_of_abcde_l969_96976


namespace NUMINAMATH_GPT_angle_B_magnitude_value_of_b_l969_96982
open Real

theorem angle_B_magnitude (B : ℝ) (h : 2 * sin B - 2 * sin B ^ 2 - cos (2 * B) = sqrt 3 - 1) :
  B = π / 3 ∨ B = 2 * π / 3 := sorry

theorem value_of_b (a B S : ℝ) (hB : B = π / 3) (ha : a = 6) (hS : S = 6 * sqrt 3) :
  let c := 4
  let b := 2 * sqrt 7
  let half_angle_B := 1 / 2 * a * c * sin B
  half_angle_B = S :=
by
  sorry

end NUMINAMATH_GPT_angle_B_magnitude_value_of_b_l969_96982


namespace NUMINAMATH_GPT_adoption_event_l969_96936

theorem adoption_event (c : ℕ) 
  (h1 : ∀ d : ℕ, d = 8) 
  (h2 : ∀ fees_dog : ℕ, fees_dog = 15) 
  (h3 : ∀ fees_cat : ℕ, fees_cat = 13)
  (h4 : ∀ donation : ℕ, donation = 53)
  (h5 : fees_dog * 8 + fees_cat * c = 159) :
  c = 3 :=
by 
  sorry

end NUMINAMATH_GPT_adoption_event_l969_96936


namespace NUMINAMATH_GPT_find_pairs_l969_96932

theorem find_pairs (x y p : ℕ)
  (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h3 : x ≤ y) (h4 : Prime p) :
  (x = 3 ∧ y = 5 ∧ p = 7) ∨ (x = 1 ∧ ∃ q, Prime q ∧ y = q + 1 ∧ p = q ∧ q ≠ 7) ↔
  (x + y) * (x * y - 1) / (x * y + 1) = p := 
sorry

end NUMINAMATH_GPT_find_pairs_l969_96932


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l969_96909

theorem algebraic_expression_evaluation (a b : ℝ) (h₁ : a ≠ b) 
  (h₂ : a^2 - 8 * a + 5 = 0) (h₃ : b^2 - 8 * b + 5 = 0) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l969_96909


namespace NUMINAMATH_GPT_white_square_area_l969_96946

theorem white_square_area
    (edge_length : ℝ)
    (total_paint : ℝ)
    (total_surface_area : ℝ)
    (green_paint_per_face : ℝ)
    (white_square_area_per_face: ℝ) :
    edge_length = 12 →
    total_paint = 432 →
    total_surface_area = 6 * (edge_length ^ 2) →
    green_paint_per_face = total_paint / 6 →
    white_square_area_per_face = (edge_length ^ 2) - green_paint_per_face →
    white_square_area_per_face = 72
:= sorry

end NUMINAMATH_GPT_white_square_area_l969_96946


namespace NUMINAMATH_GPT_P_has_real_root_l969_96923

def P : ℝ → ℝ := sorry
variables (a1 a2 a3 b1 b2 b3 : ℝ)

axiom a1_nonzero : a1 ≠ 0
axiom a2_nonzero : a2 ≠ 0
axiom a3_nonzero : a3 ≠ 0

axiom functional_eq (x : ℝ) :
  P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)

theorem P_has_real_root :
  ∃ x : ℝ, P x = 0 :=
sorry

end NUMINAMATH_GPT_P_has_real_root_l969_96923


namespace NUMINAMATH_GPT_ellen_lost_legos_l969_96951

theorem ellen_lost_legos (L_initial L_final : ℕ) (h1 : L_initial = 2080) (h2 : L_final = 2063) : L_initial - L_final = 17 := by
  sorry

end NUMINAMATH_GPT_ellen_lost_legos_l969_96951


namespace NUMINAMATH_GPT_whole_number_N_l969_96958

theorem whole_number_N (N : ℤ) : (9 < N / 4 ∧ N / 4 < 10) ↔ (N = 37 ∨ N = 38 ∨ N = 39) := 
by sorry

end NUMINAMATH_GPT_whole_number_N_l969_96958


namespace NUMINAMATH_GPT_parabola_unique_solution_l969_96945

theorem parabola_unique_solution (b c : ℝ) :
  (∀ x y : ℝ, (x, y) = (-2, -8) ∨ (x, y) = (4, 28) ∨ (x, y) = (1, 4) →
    (y = x^2 + b * x + c)) →
  b = 4 ∧ c = -1 :=
by
  intro h
  have h₁ := h (-2) (-8) (Or.inl rfl)
  have h₂ := h 4 28 (Or.inr (Or.inl rfl))
  have h₃ := h 1 4 (Or.inr (Or.inr rfl))
  sorry

end NUMINAMATH_GPT_parabola_unique_solution_l969_96945


namespace NUMINAMATH_GPT_lipstick_cost_is_correct_l969_96911

noncomputable def cost_of_lipstick (palette_cost : ℝ) (num_palettes : ℝ) (hair_color_cost : ℝ) (num_hair_colors : ℝ) (total_paid : ℝ) (num_lipsticks : ℝ) : ℝ :=
  let total_palette_cost := num_palettes * palette_cost
  let total_hair_color_cost := num_hair_colors * hair_color_cost
  let remaining_amount := total_paid - (total_palette_cost + total_hair_color_cost)
  remaining_amount / num_lipsticks

theorem lipstick_cost_is_correct :
  cost_of_lipstick 15 3 4 3 67 4 = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_lipstick_cost_is_correct_l969_96911


namespace NUMINAMATH_GPT_expression_range_l969_96959

theorem expression_range (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2)
  (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
  4 * Real.sqrt 2 ≤ (Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2)
  + Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)) ∧ 
  (Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2) + Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)) ≤ 8 :=
sorry

end NUMINAMATH_GPT_expression_range_l969_96959


namespace NUMINAMATH_GPT_k_sq_geq_25_over_4_l969_96915

theorem k_sq_geq_25_over_4
  (a1 a2 a3 a4 a5 k : ℝ)
  (h1 : |a1 - a2| ≥ 1 ∧ |a1 - a3| ≥ 1 ∧ |a1 - a4| ≥ 1 ∧ |a1 - a5| ≥ 1 ∧
       |a2 - a3| ≥ 1 ∧ |a2 - a4| ≥ 1 ∧ |a2 - a5| ≥ 1 ∧
       |a3 - a4| ≥ 1 ∧ |a3 - a5| ≥ 1 ∧
       |a4 - a5| ≥ 1)
  (h2 : a1 + a2 + a3 + a4 + a5 = 2 * k)
  (h3 : a1^2 + a2^2 + a3^2 + a4^2 + a5^2 = 2 * k^2) :
  k^2 ≥ 25 / 4 :=
sorry

end NUMINAMATH_GPT_k_sq_geq_25_over_4_l969_96915


namespace NUMINAMATH_GPT_message_hours_needed_l969_96996

-- Define the sequence and the condition
def S (n : ℕ) : ℕ := 2^(n + 1) - 2

theorem message_hours_needed : ∃ n : ℕ, S n > 55 ∧ n = 5 := by
  sorry

end NUMINAMATH_GPT_message_hours_needed_l969_96996


namespace NUMINAMATH_GPT_find_k_l969_96973

def equation (k : ℝ) (x : ℝ) : Prop := 2 * x^2 + 3 * x - k = 0

theorem find_k (k : ℝ) (h : equation k 7) : k = 119 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l969_96973


namespace NUMINAMATH_GPT_four_four_four_digits_eight_eight_eight_digits_l969_96963

theorem four_four_four_digits_eight_eight_eight_digits (n : ℕ) :
  (4 * (10 ^ (n + 1) - 1) * (10 ^ n) + 8 * (10^n - 1) + 9) = 
  (6 * 10^n + 7) * (6 * 10^n + 7) :=
sorry

end NUMINAMATH_GPT_four_four_four_digits_eight_eight_eight_digits_l969_96963


namespace NUMINAMATH_GPT_eval_expression_l969_96933

theorem eval_expression : 9^9 * 3^3 / 3^30 = 1 / 19683 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l969_96933


namespace NUMINAMATH_GPT_range_of_2a_plus_3b_inequality_between_expressions_l969_96924

-- First proof problem
theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b) (h2 : a + b ≤ 1) (h3 : -1 ≤ a - b) (h4 : a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
sorry

-- Second proof problem
theorem inequality_between_expressions (a b c : ℝ) (h : a^2 + b^2 + c^2 = 6) :
  (1 / (a^2 + 1) + 1 / (b^2 + 2)) > (1 / 2 - 1 / (c^2 + 3)) :=
sorry

end NUMINAMATH_GPT_range_of_2a_plus_3b_inequality_between_expressions_l969_96924


namespace NUMINAMATH_GPT_least_subtracted_correct_second_num_correct_l969_96935

-- Define the given numbers
def given_num : ℕ := 1398
def remainder : ℕ := 5
def num1 : ℕ := 7
def num2 : ℕ := 9
def num3 : ℕ := 11

-- Least number to subtract to satisfy the condition
def least_subtracted : ℕ := 22

-- Second number in the sequence
def second_num : ℕ := 2069

-- Define the hypotheses and statements to be proved
theorem least_subtracted_correct : given_num - least_subtracted ≡ remainder [MOD num1]
∧ given_num - least_subtracted ≡ remainder [MOD num2]
∧ given_num - least_subtracted ≡ remainder [MOD num3] := sorry

theorem second_num_correct : second_num ≡ remainder [MOD num1 * num2 * num3] := sorry

end NUMINAMATH_GPT_least_subtracted_correct_second_num_correct_l969_96935


namespace NUMINAMATH_GPT_total_games_l969_96954

-- Define the conditions
def games_this_year : ℕ := 4
def games_last_year : ℕ := 9

-- Define the proposition that we want to prove
theorem total_games : games_this_year + games_last_year = 13 := by
  sorry

end NUMINAMATH_GPT_total_games_l969_96954


namespace NUMINAMATH_GPT_find_number_l969_96921

theorem find_number (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 := 
sorry

end NUMINAMATH_GPT_find_number_l969_96921


namespace NUMINAMATH_GPT_evaluate_expression_l969_96902

def operation (x y : ℚ) : ℚ := x^2 / y

theorem evaluate_expression : 
  (operation (operation 3 4) 2) - (operation 3 (operation 4 2)) = 45 / 32 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l969_96902


namespace NUMINAMATH_GPT_black_marbles_count_l969_96985

theorem black_marbles_count :
  ∀ (white_marbles total_marbles : ℕ), 
  white_marbles = 19 → total_marbles = 37 → total_marbles - white_marbles = 18 :=
by
  intros white_marbles total_marbles h_white h_total
  sorry

end NUMINAMATH_GPT_black_marbles_count_l969_96985


namespace NUMINAMATH_GPT_quadratic_real_roots_iff_range_of_a_l969_96971

theorem quadratic_real_roots_iff_range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a = 0) ↔ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_iff_range_of_a_l969_96971


namespace NUMINAMATH_GPT_balls_in_boxes_l969_96907

theorem balls_in_boxes : (3^4 = 81) :=
by
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l969_96907


namespace NUMINAMATH_GPT_remainder_div_2468135790_101_l969_96968

theorem remainder_div_2468135790_101 : 2468135790 % 101 = 50 :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_2468135790_101_l969_96968


namespace NUMINAMATH_GPT_sum_of_primes_is_prime_l969_96914

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

theorem sum_of_primes_is_prime (P Q : ℕ) :
  is_prime P → is_prime Q → is_prime (P - Q) → is_prime (P + Q) →
  ∃ n : ℕ, n = P + Q + (P - Q) + (P + Q) ∧ is_prime n := by
  sorry

end NUMINAMATH_GPT_sum_of_primes_is_prime_l969_96914
