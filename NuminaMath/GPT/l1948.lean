import Mathlib

namespace find_n_l1948_194847

theorem find_n (n : ℕ) (h1 : 0 < n) (h2 : n < 11) (h3 : (18888 - n) % 11 = 0) : n = 1 :=
by
  sorry

end find_n_l1948_194847


namespace range_of_a_l1948_194801

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l1948_194801


namespace overlap_length_in_mm_l1948_194809

theorem overlap_length_in_mm {sheets : ℕ} {length_per_sheet : ℝ} {perimeter : ℝ} 
  (h_sheets : sheets = 12)
  (h_length_per_sheet : length_per_sheet = 18)
  (h_perimeter : perimeter = 210) : 
  (length_per_sheet * sheets - perimeter) / sheets * 10 = 5 := by
  sorry

end overlap_length_in_mm_l1948_194809


namespace relationship_among_a_b_c_l1948_194892

theorem relationship_among_a_b_c (a b c : ℝ) (h₁ : a = 0.09) (h₂ : -2 < b ∧ b < -1) (h₃ : 1 < c ∧ c < 2) : b < a ∧ a < c := 
by 
  -- proof will involve but we only need to state this
  sorry

end relationship_among_a_b_c_l1948_194892


namespace count_positive_solutions_of_eq_l1948_194830

theorem count_positive_solutions_of_eq : 
  (∃ x : ℝ, x^2 = -6 * x + 9 ∧ x > 0) ∧ (¬ ∃ y : ℝ, y^2 = -6 * y + 9 ∧ y > 0 ∧ y ≠ -3 + 3 * Real.sqrt 2) :=
sorry

end count_positive_solutions_of_eq_l1948_194830


namespace reciprocal_of_neg_five_l1948_194803

theorem reciprocal_of_neg_five : ∃ b : ℚ, (-5) * b = 1 ∧ b = -1/5 :=
by
  sorry

end reciprocal_of_neg_five_l1948_194803


namespace quadratic_has_integer_solutions_l1948_194872

theorem quadratic_has_integer_solutions : 
  ∃ (s : Finset ℕ), ∀ a : ℕ, a ∈ s ↔ (1 ≤ a ∧ a ≤ 50 ∧ ((∃ n : ℕ, 4 * a + 1 = n^2))) ∧ s.card = 6 := 
  sorry

end quadratic_has_integer_solutions_l1948_194872


namespace convex_ngon_sides_l1948_194897

theorem convex_ngon_sides (n : ℕ) (h : (n * (n - 3)) / 2 = 27) : n = 9 :=
by
  -- Proof omitted
  sorry

end convex_ngon_sides_l1948_194897


namespace passed_candidates_l1948_194824

theorem passed_candidates (P F : ℕ) (h1 : P + F = 120) (h2 : 39 * P + 15 * F = 35 * 120) : P = 100 :=
by
  sorry

end passed_candidates_l1948_194824


namespace maxwell_distance_l1948_194895

-- Define the given conditions
def distance_between_homes : ℝ := 65
def maxwell_speed : ℝ := 2
def brad_speed : ℝ := 3

-- The statement we need to prove
theorem maxwell_distance :
  ∃ (x t : ℝ), 
    x = maxwell_speed * t ∧
    distance_between_homes - x = brad_speed * t ∧
    x = 26 := by sorry

end maxwell_distance_l1948_194895


namespace base_eight_seventeen_five_is_one_two_five_l1948_194841

def base_eight_to_base_ten (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 8^2 + ((n % 100) / 10) * 8^1 + (n % 10) * 8^0

theorem base_eight_seventeen_five_is_one_two_five :
  base_eight_to_base_ten 175 = 125 :=
by
  sorry

end base_eight_seventeen_five_is_one_two_five_l1948_194841


namespace find_quotient_l1948_194856

-- Definitions based on given conditions
def remainder : ℕ := 8
def dividend : ℕ := 997
def divisor : ℕ := 23

-- Hypothesis based on the division formula
def quotient_formula (q : ℕ) : Prop :=
  dividend = (divisor * q) + remainder

-- Statement of the problem
theorem find_quotient (q : ℕ) (h : quotient_formula q) : q = 43 :=
sorry

end find_quotient_l1948_194856


namespace sequence_x_values_3001_l1948_194886

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n = (a (n - 1) * a (n + 1) - 1)

theorem sequence_x_values_3001 {a : ℕ → ℝ} (x : ℝ) (h₁ : a 1 = x) (h₂ : a 2 = 3000) :
  (∃ n, a n = 3001) ↔ x = 1 ∨ x = 9005999 ∨ x = 3001 / 9005999 :=
sorry

end sequence_x_values_3001_l1948_194886


namespace polynomial_remainder_l1948_194837

noncomputable def h (x : ℕ) := x^5 + x^4 + x^3 + x^2 + x + 1

theorem polynomial_remainder (x : ℕ) : (h (x^10)) % (h x) = 5 :=
sorry

end polynomial_remainder_l1948_194837


namespace common_ratio_l1948_194863

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geom : ∀ n, a (n+1) = r * a n)
variable (h1 : a 5 * a 11 = 3)
variable (h2 : a 3 + a 13 = 4)

theorem common_ratio (h_geom : ∀ n, a (n+1) = r * a n) (h1 : a 5 * a 11 = 3) (h2 : a 3 + a 13 = 4) :
  (r = 3 ∨ r = -3) :=
by
  sorry

end common_ratio_l1948_194863


namespace two_hours_charge_l1948_194899

def charge_condition_1 (F A : ℕ) : Prop :=
  F = A + 35

def charge_condition_2 (F A : ℕ) : Prop :=
  F + 4 * A = 350

theorem two_hours_charge (F A : ℕ) (h1 : charge_condition_1 F A) (h2 : charge_condition_2 F A) : 
  F + A = 161 := 
sorry

end two_hours_charge_l1948_194899


namespace trees_to_plant_total_l1948_194890

def trees_chopped_first_half := 200
def trees_chopped_second_half := 300
def trees_to_plant_per_tree_chopped := 3

theorem trees_to_plant_total : 
  (trees_chopped_first_half + trees_chopped_second_half) * trees_to_plant_per_tree_chopped = 1500 :=
by
  sorry

end trees_to_plant_total_l1948_194890


namespace element_in_set_l1948_194800

open Set

theorem element_in_set : -7 ∈ ({1, -7} : Set ℤ) := by
  sorry

end element_in_set_l1948_194800


namespace eliminate_y_substitution_l1948_194881

theorem eliminate_y_substitution (x y : ℝ) (h1 : y = x - 5) (h2 : 3 * x - y = 8) : 3 * x - x + 5 = 8 := 
by
  sorry

end eliminate_y_substitution_l1948_194881


namespace area_ratio_of_circles_l1948_194825

theorem area_ratio_of_circles 
  (CX : ℝ)
  (CY : ℝ)
  (RX RY : ℝ)
  (hX : CX = 2 * π * RX)
  (hY : CY = 2 * π * RY)
  (arc_length_equality : (90 / 360) * CX = (60 / 360) * CY) :
  (π * RX^2) / (π * RY^2) = 9 / 4 :=
by
  sorry

end area_ratio_of_circles_l1948_194825


namespace final_price_is_correct_l1948_194840

def initial_price : ℝ := 15
def first_discount_rate : ℝ := 0.2
def second_discount_rate : ℝ := 0.25

def first_discount : ℝ := initial_price * first_discount_rate
def price_after_first_discount : ℝ := initial_price - first_discount

def second_discount : ℝ := price_after_first_discount * second_discount_rate
def final_price : ℝ := price_after_first_discount - second_discount

theorem final_price_is_correct :
  final_price = 9 :=
by
  -- The actual proof steps will go here.
  sorry

end final_price_is_correct_l1948_194840


namespace inequality_and_equality_condition_l1948_194838

variable (a b c t : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem inequality_and_equality_condition :
  abc * (a^t + b^t + c^t) ≥ a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ∧ 
  (abc * (a^t + b^t + c^t) = a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ↔ a = b ∧ b = c) :=
sorry

end inequality_and_equality_condition_l1948_194838


namespace asymptote_hyperbola_condition_l1948_194894

theorem asymptote_hyperbola_condition : 
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1 → y = 4/3 * x ∨ y = -4/3 * x)) ∧
  ¬(∀ x y : ℝ, (y = 4/3 * x ∨ y = -4/3 * x → x^2 / 9 - y^2 / 16 = 1)) :=
by sorry

end asymptote_hyperbola_condition_l1948_194894


namespace heptagon_triangulation_count_l1948_194804

/-- The number of ways to divide a regular heptagon (7-sided polygon) 
    into 5 triangles using non-intersecting diagonals is 4. -/
theorem heptagon_triangulation_count : ∃ (n : ℕ), n = 4 ∧ ∀ (p : ℕ), (p = 7 ∧ (∀ (k : ℕ), k = 5 → (n = 4))) :=
by {
  -- The proof is non-trivial and omitted here
  sorry
}

end heptagon_triangulation_count_l1948_194804


namespace telephone_charge_l1948_194885

theorem telephone_charge (x : ℝ) (h1 : ∀ t : ℝ, t = 18.70 → x + 39 * 0.40 = t) : x = 3.10 :=
by
  sorry

end telephone_charge_l1948_194885


namespace koschei_never_escapes_l1948_194843

-- Define a structure for the initial setup
structure Setup where
  koschei_initial_room : Nat -- Initial room of Koschei
  guard_positions : List (Bool) -- Guards' positions, True for West, False for East

-- Example of the required setup:
def initial_setup : Setup :=
  { koschei_initial_room := 1, guard_positions := [true, false, true] }

-- Function to simulate the movement of guards
def move_guards (guards : List Bool) (room : Nat) : List Bool :=
  guards.map (λ g => not g)

-- Function to check if all guards are on the same wall
def all_guards_same_wall (guards : List Bool) : Bool :=
  List.all guards id ∨ List.all guards (λ g => ¬g)

-- Main statement: 
theorem koschei_never_escapes (setup : Setup) :
  ∀ room : Nat, ¬(all_guards_same_wall (move_guards setup.guard_positions room)) :=
  sorry

end koschei_never_escapes_l1948_194843


namespace range_of_a_l1948_194898

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x < -1 ↔ x ≤ a) ↔ a < -1 :=
by
  sorry

end range_of_a_l1948_194898


namespace sum_tens_units_digit_9_pow_1001_l1948_194826

-- Define a function to extract the last two digits of a number
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Define a function to extract the tens digit
def tens_digit (n : ℕ) : ℕ := (last_two_digits n) / 10

-- Define a function to extract the units digit
def units_digit (n : ℕ) : ℕ := (last_two_digits n) % 10

-- The main theorem
theorem sum_tens_units_digit_9_pow_1001 :
  tens_digit (9 ^ 1001) + units_digit (9 ^ 1001) = 9 :=
by
  sorry

end sum_tens_units_digit_9_pow_1001_l1948_194826


namespace printing_time_l1948_194854

-- Definitions based on the problem conditions
def printer_rate : ℕ := 25 -- Pages per minute
def total_pages : ℕ := 325 -- Total number of pages to be printed

-- Statement of the problem rewritten as a Lean 4 statement
theorem printing_time : total_pages / printer_rate = 13 := by
  sorry

end printing_time_l1948_194854


namespace largest_real_number_condition_l1948_194874

theorem largest_real_number_condition (x : ℝ) (hx : ⌊x⌋ / x = 7 / 8) : x ≤ 48 / 7 :=
by
  sorry

end largest_real_number_condition_l1948_194874


namespace gcd_135_81_l1948_194845

-- Define the numbers
def a : ℕ := 135
def b : ℕ := 81

-- State the goal: greatest common divisor of a and b is 27
theorem gcd_135_81 : Nat.gcd a b = 27 := by
  sorry

end gcd_135_81_l1948_194845


namespace sum_of_roots_of_quadratic_l1948_194846

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h_eq : a = 3 ∧ b = 6 ∧ c = -9) :
  (-b / a) = -2 :=
by
  rcases h_eq with ⟨ha, hb, hc⟩
  -- Proof goes here, but we can use sorry to skip it
  sorry

end sum_of_roots_of_quadratic_l1948_194846


namespace max_pies_without_ingredients_l1948_194810

theorem max_pies_without_ingredients (total_pies half_chocolate two_thirds_marshmallows three_fifths_cayenne one_eighth_peanuts : ℕ) 
  (h1 : total_pies = 48) 
  (h2 : half_chocolate = total_pies / 2)
  (h3 : two_thirds_marshmallows = 2 * total_pies / 3) 
  (h4 : three_fifths_cayenne = 3 * total_pies / 5)
  (h5 : one_eighth_peanuts = total_pies / 8) : 
  ∃ pies_without_any_ingredients, pies_without_any_ingredients = 16 :=
  by 
    sorry

end max_pies_without_ingredients_l1948_194810


namespace total_slices_l1948_194839

theorem total_slices {slices_per_pizza pizzas : ℕ} (h1 : slices_per_pizza = 2) (h2 : pizzas = 14) : 
  slices_per_pizza * pizzas = 28 :=
by
  -- This is where the proof would go, but we are omitting it as instructed.
  sorry

end total_slices_l1948_194839


namespace irreducible_fraction_denominator_l1948_194857

theorem irreducible_fraction_denominator :
  let num := 201920192019
  let denom := 191719171917
  let gcd_num_denom := Int.gcd num denom
  let irreducible_denom := denom / gcd_num_denom
  irreducible_denom = 639 :=
by
  sorry

end irreducible_fraction_denominator_l1948_194857


namespace ufo_convention_attendees_l1948_194813

theorem ufo_convention_attendees (f m total : ℕ) 
  (h1 : m = 62) 
  (h2 : m = f + 4) : 
  total = 120 :=
by
  sorry

end ufo_convention_attendees_l1948_194813


namespace chlorine_weight_is_35_l1948_194883

def weight_Na : Nat := 23
def weight_O : Nat := 16
def molecular_weight : Nat := 74

theorem chlorine_weight_is_35 (Cl : Nat) 
  (h : molecular_weight = weight_Na + Cl + weight_O) : 
  Cl = 35 := by
  -- Proof placeholder
  sorry

end chlorine_weight_is_35_l1948_194883


namespace age_of_first_person_added_l1948_194884

theorem age_of_first_person_added :
  ∀ (T A x : ℕ),
    (T = 7 * A) →
    (T + x = 8 * (A + 2)) →
    (T + 15 = 8 * (A - 1)) →
    x = 39 :=
by
  intros T A x h1 h2 h3
  sorry

end age_of_first_person_added_l1948_194884


namespace solve_quadratic_equation_l1948_194867

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 6 * x - 3 = 0 ↔ x = 3 + 2 * Real.sqrt 3 ∨ x = 3 - 2 * Real.sqrt 3 :=
by
  sorry

end solve_quadratic_equation_l1948_194867


namespace find_number_of_small_branches_each_branch_grows_l1948_194882

theorem find_number_of_small_branches_each_branch_grows :
  ∃ x : ℕ, 1 + x + x^2 = 43 ∧ x = 6 :=
by {
  sorry
}

end find_number_of_small_branches_each_branch_grows_l1948_194882


namespace symmetric_function_value_l1948_194806

noncomputable def f (x a : ℝ) := (|x - 2| + a) / (Real.sqrt (4 - x^2))

theorem symmetric_function_value :
  ∃ a : ℝ, (∀ x : ℝ, f x a = (|x - 2| + a) / (Real.sqrt (4 - x^2)) ∧ f x a = -f (-x) a) →
  f (a / 2) a = (Real.sqrt 3) / 3 :=
by
  sorry

end symmetric_function_value_l1948_194806


namespace quadratic_polynomials_exist_l1948_194834

-- Definitions of the polynomials
def p1 (x : ℝ) := (x - 10)^2 - 1
def p2 (x : ℝ) := x^2 - 1
def p3 (x : ℝ) := (x + 10)^2 - 1

-- The theorem to prove
theorem quadratic_polynomials_exist :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ p1 x1 = 0 ∧ p1 x2 = 0) ∧
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ p2 y1 = 0 ∧ p2 y2 = 0) ∧
  (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ p3 z1 = 0 ∧ p3 z2 = 0) ∧
  (∀ x : ℝ, p1 x + p2 x ≠ 0 ∧ p1 x + p3 x ≠ 0 ∧ p2 x + p3 x ≠ 0) :=
by
  sorry

end quadratic_polynomials_exist_l1948_194834


namespace fraction_to_decimal_l1948_194860

theorem fraction_to_decimal (numer: ℚ) (denom: ℕ) (h_denom: denom = 2^5 * 5^1) :
  numer.den = 160 → numer.num = 59 → numer == 0.36875 :=
by
  intros
  sorry  

end fraction_to_decimal_l1948_194860


namespace consumption_increase_l1948_194832

variable (T C C' : ℝ)
variable (h1 : 0.8 * T * C' = 0.92 * T * C)

theorem consumption_increase (T C C' : ℝ) (h1 : 0.8 * T * C' = 0.92 * T * C) : C' = 1.15 * C :=
by
  sorry

end consumption_increase_l1948_194832


namespace line_equation_through_point_and_area_l1948_194844

theorem line_equation_through_point_and_area (b S x y : ℝ) 
  (h1 : ∀ y, (x, y) = (-2*b, 0) → True) 
  (h2 : ∀ p1 p2 p3 : ℝ × ℝ, p1 = (-2*b, 0) → p2 = (0, 0) → 
        ∃ k, p3 = (0, k) ∧ S = 1/2 * (2*b) * k) : 2*S*x - b^2*y + 4*b*S = 0 :=
sorry

end line_equation_through_point_and_area_l1948_194844


namespace ratio_first_term_common_diff_l1948_194812

theorem ratio_first_term_common_diff {a d : ℤ} 
  (S_20 : ℤ) (S_10 : ℤ)
  (h1 : S_20 = 10 * (2 * a + 19 * d))
  (h2 : S_10 = 5 * (2 * a + 9 * d))
  (h3 : S_20 = 6 * S_10) :
  a / d = 2 :=
by
  sorry

end ratio_first_term_common_diff_l1948_194812


namespace arithmetic_seq_sum_a3_a15_l1948_194891

theorem arithmetic_seq_sum_a3_a15 (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_eq : a 1 - a 5 + a 9 - a 13 + a 17 = 117) :
  a 3 + a 15 = 234 :=
sorry

end arithmetic_seq_sum_a3_a15_l1948_194891


namespace minimum_score_to_win_l1948_194858

namespace CompetitionPoints

-- Define points awarded for each position
def points_first : ℕ := 5
def points_second : ℕ := 3
def points_third : ℕ := 1

-- Define the number of competitions
def competitions : ℕ := 3

-- Total points in one competition
def total_points_one_competition : ℕ := points_first + points_second + points_third

-- Total points in all competitions
def total_points_all_competitions : ℕ := total_points_one_competition * competitions

theorem minimum_score_to_win : ∃ m : ℕ, m = 13 ∧ (∀ s : ℕ, s < 13 → ¬ ∃ c1 c2 c3 : ℕ, 
  c1 ≤ competitions ∧ c2 ≤ competitions ∧ c3 ≤ competitions ∧ 
  ((c1 * points_first) + (c2 * points_second) + (c3 * points_third)) = s) :=
by {
  sorry
}

end CompetitionPoints

end minimum_score_to_win_l1948_194858


namespace find_t_l1948_194850

theorem find_t (t : ℝ) (h : (1 / (t + 2) + 2 * t / (t + 2) - 3 / (t + 2) = 3)) : t = -8 := 
by 
  sorry

end find_t_l1948_194850


namespace factorize_expression_l1948_194870

theorem factorize_expression (x : ℝ) : 2 * x ^ 3 - 4 * x ^ 2 - 6 * x = 2 * x * (x - 3) * (x + 1) :=
by
  sorry

end factorize_expression_l1948_194870


namespace room_length_difference_l1948_194827

def width := 19
def length := 20
def difference := length - width

theorem room_length_difference : difference = 1 := by
  sorry

end room_length_difference_l1948_194827


namespace setB_can_form_triangle_l1948_194831

theorem setB_can_form_triangle : 
  let a := 8
  let b := 6
  let c := 4
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  let a := 8
  let b := 6
  let c := 4
  have h1 : a + b > c := by sorry
  have h2 : a + c > b := by sorry
  have h3 : b + c > a := by sorry
  exact ⟨h1, h2, h3⟩

end setB_can_form_triangle_l1948_194831


namespace Nero_speed_is_8_l1948_194817

-- Defining the conditions
def Jerome_time := 6 -- in hours
def Nero_time := 3 -- in hours
def Jerome_speed := 4 -- in miles per hour

-- Calculation step
def Distance := Jerome_speed * Jerome_time

-- The theorem we need to prove (Nero's speed)
theorem Nero_speed_is_8 :
  (Distance / Nero_time) = 8 := by
  sorry

end Nero_speed_is_8_l1948_194817


namespace math_problem_l1948_194873

theorem math_problem (a b : ℝ) (h : Real.sqrt (a + 2) + |b - 1| = 0) : (a + b) ^ 2023 = -1 := 
by
  sorry

end math_problem_l1948_194873


namespace cubic_polynomial_unique_l1948_194877

-- Define the polynomial q(x)
def q (x : ℝ) : ℝ := -x^3 + 4*x^2 - 7*x - 4

-- State the conditions
theorem cubic_polynomial_unique :
  q 1 = -8 ∧
  q 2 = -10 ∧
  q 3 = -16 ∧
  q 4 = -32 :=
by
  -- Expand the function definition for the given inputs.
  -- Add these expansions in the proof part.
  sorry

end cubic_polynomial_unique_l1948_194877


namespace problem_statement_l1948_194851

theorem problem_statement (x y a : ℝ) (h1 : x + a < y + a) (h2 : a * x > a * y) : x < y ∧ a < 0 :=
sorry

end problem_statement_l1948_194851


namespace probability_A_and_B_same_county_l1948_194836

/-
We have four experts and three counties. We need to assign the experts to the counties such 
that each county has at least one expert. We need to prove that the probability of experts 
A and B being dispatched to the same county is 1/6.
-/

def num_experts : Nat := 4
def num_counties : Nat := 3

def total_possible_events : Nat := 36
def favorable_events : Nat := 6

theorem probability_A_and_B_same_county :
  (favorable_events : ℚ) / total_possible_events = 1 / 6 := by sorry

end probability_A_and_B_same_county_l1948_194836


namespace remainder_when_divided_by_x_minus_2_l1948_194814

def f (x : ℝ) : ℝ := x^5 - 4 * x^4 + 6 * x^3 + 25 * x^2 - 20 * x - 24

theorem remainder_when_divided_by_x_minus_2 : f 2 = 52 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l1948_194814


namespace tapanga_corey_candies_l1948_194861

theorem tapanga_corey_candies (corey_candies : ℕ) (tapanga_candies : ℕ) 
                              (h1 : corey_candies = 29) 
                              (h2 : tapanga_candies = corey_candies + 8) : 
                              corey_candies + tapanga_candies = 66 :=
by
  rw [h1, h2]
  sorry

end tapanga_corey_candies_l1948_194861


namespace percent_blue_marbles_l1948_194875

theorem percent_blue_marbles (total_items buttons red_marbles : ℝ) 
  (H1 : buttons = 0.30 * total_items)
  (H2 : red_marbles = 0.50 * (total_items - buttons)) :
  (total_items - buttons - red_marbles) / total_items = 0.35 :=
by 
  sorry

end percent_blue_marbles_l1948_194875


namespace solution_set_f_x_gt_0_l1948_194816

theorem solution_set_f_x_gt_0 (b : ℝ)
  (h_eq : ∀ x : ℝ, (x + 1) * (x - 3) = 0 → b = -2) :
  {x : ℝ | (x - 1)^2 > 0} = {x : ℝ | x ≠ 1} :=
by
  sorry

end solution_set_f_x_gt_0_l1948_194816


namespace positive_diff_probability_fair_coin_l1948_194880

theorem positive_diff_probability_fair_coin :
  let p1 := (Nat.choose 5 3) * (1 / 2)^5
  let p2 := (1 / 2)^5
  p1 - p2 = 9 / 32 :=
by
  sorry

end positive_diff_probability_fair_coin_l1948_194880


namespace willie_cream_l1948_194833

theorem willie_cream : ∀ (total_cream needed_cream: ℕ), total_cream = 300 → needed_cream = 149 → (total_cream - needed_cream) = 151 :=
by
  intros total_cream needed_cream h1 h2
  sorry

end willie_cream_l1948_194833


namespace transformed_curve_l1948_194853

variables (x y x' y' : ℝ)

def original_curve := (x^2) / 4 - y^2 = 1
def transformation_x := x' = (1/2) * x
def transformation_y := y' = 2 * y

theorem transformed_curve : original_curve x y → transformation_x x x' → transformation_y y y' → x^2 - (y^2) / 4 = 1 := 
sorry

end transformed_curve_l1948_194853


namespace negative_values_count_l1948_194835

theorem negative_values_count (n : ℕ) : (n < 13) → (n^2 < 150) → ∃ (k : ℕ), k = 12 :=
by
  sorry

end negative_values_count_l1948_194835


namespace pure_imaginary_complex_number_l1948_194822

variable (a : ℝ)

theorem pure_imaginary_complex_number:
  (a^2 + 2*a - 3 = 0) ∧ (a^2 + a - 6 ≠ 0) → a = 1 := by
  sorry

end pure_imaginary_complex_number_l1948_194822


namespace num_common_points_of_three_lines_l1948_194896

def three_planes {P : Type} [AddCommGroup P] (l1 l2 l3 : Set P) : Prop :=
  let p12 := Set.univ \ (l1 ∪ l2)
  let p13 := Set.univ \ (l1 ∪ l3)
  let p23 := Set.univ \ (l2 ∪ l3)
  ∃ (pl12 pl13 pl23 : Set P), 
    p12 = pl12 ∧ p13 = pl13 ∧ p23 = pl23

theorem num_common_points_of_three_lines (l1 l2 l3 : Set ℝ) 
  (h : three_planes l1 l2 l3) : ∃ n : ℕ, n = 0 ∨ n = 1 := by
  sorry

end num_common_points_of_three_lines_l1948_194896


namespace identify_smart_person_l1948_194889

theorem identify_smart_person (F S : ℕ) (h_total : F + S = 30) (h_max_fools : F ≤ 8) : S ≥ 1 :=
by {
  sorry
}

end identify_smart_person_l1948_194889


namespace fisherman_total_fish_l1948_194887

theorem fisherman_total_fish :
  let bass : Nat := 32
  let trout : Nat := bass / 4
  let blue_gill : Nat := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  show bass + trout + blue_gill = 104
  sorry

end fisherman_total_fish_l1948_194887


namespace sqrt_sqrt_16_eq_pm2_l1948_194807

theorem sqrt_sqrt_16_eq_pm2 (h : Real.sqrt 16 = 4) : Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm2_l1948_194807


namespace prod_of_real_roots_equation_l1948_194818

theorem prod_of_real_roots_equation :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → x = 0 ∨ x = -(4 / 7) → (0 * (-(4 / 7)) = 0) :=
by sorry

end prod_of_real_roots_equation_l1948_194818


namespace tan_degree_identity_l1948_194805

theorem tan_degree_identity (k : ℝ) (hk : Real.cos (Real.pi * -80 / 180) = k) : 
  Real.tan (Real.pi * 100 / 180) = - (Real.sqrt (1 - k^2) / k) := 
by 
  sorry

end tan_degree_identity_l1948_194805


namespace range_of_m_l1948_194811

-- Define the sets A and B
def setA := {x : ℝ | abs (x - 1) < 2}
def setB (m : ℝ) := {x : ℝ | x >= m}

-- State the theorem
theorem range_of_m : ∀ (m : ℝ), (setA ∩ setB m = setA) → m <= -1 :=
by
  sorry

end range_of_m_l1948_194811


namespace missed_questions_proof_l1948_194859

def num_missed_questions : ℕ := 180

theorem missed_questions_proof (F : ℕ) (h1 : 5 * F + F = 216) : F = 36 ∧ 5 * F = num_missed_questions :=
by {
  sorry
}

end missed_questions_proof_l1948_194859


namespace div_polynomial_l1948_194879

noncomputable def f (x : ℝ) : ℝ := x^4 + 4*x^3 + 6*x^2 + 4*x + 2
noncomputable def g (x : ℝ) (p q s t : ℝ) : ℝ := x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*s*x + t

theorem div_polynomial 
  (p q s t : ℝ) 
  (h : ∀ x : ℝ, f x = 0 → g x p q s t = 0) : 
  (p + q + s) * t = -6 :=
by
  sorry

end div_polynomial_l1948_194879


namespace tourist_tax_l1948_194869

theorem tourist_tax (total_value : ℕ) (non_taxable_amount : ℕ) (tax_rate : ℚ) (tax : ℚ) : 
  total_value = 1720 → 
  non_taxable_amount = 600 → 
  tax_rate = 0.12 → 
  tax = (total_value - non_taxable_amount : ℕ) * tax_rate → 
  tax = 134.40 := 
by 
  intros total_value_eq non_taxable_amount_eq tax_rate_eq tax_eq
  sorry

end tourist_tax_l1948_194869


namespace triangle_angle_sum_l1948_194829

-- Definitions of the given angles and relationships
def angle_BAC := 95
def angle_ABC := 55
def angle_ABD := 125

-- We need to express the configuration of points and the measure of angle ACB
noncomputable def angle_ACB (angle_BAC angle_ABC angle_ABD : ℝ) : ℝ :=
  180 - angle_BAC - angle_ABC

-- The formalization of the problem statement in Lean 4
theorem triangle_angle_sum (angle_BAC angle_ABC angle_ABD : ℝ) :
  angle_BAC = 95 → angle_ABC = 55 → angle_ABD = 125 → angle_ACB angle_BAC angle_ABC angle_ABD = 30 :=
by
  intros h_BAC h_ABC h_ABD
  rw [h_BAC, h_ABC, h_ABD]
  sorry

end triangle_angle_sum_l1948_194829


namespace minimum_value_of_xy_l1948_194848

theorem minimum_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy = 64 :=
sorry

end minimum_value_of_xy_l1948_194848


namespace simplify_problem_1_simplify_problem_2_l1948_194865

-- Problem 1: Statement of Simplification Proof
theorem simplify_problem_1 :
  (- (99 + (71 / 72)) * 36 = - (3599 + 1 / 2)) :=
by sorry

-- Problem 2: Statement of Simplification Proof
theorem simplify_problem_2 :
  (-3 * (1 / 4) - 2.5 * (-2.45) + (7 / 2) * (1 / 4) = 6 + 1 / 4) :=
by sorry

end simplify_problem_1_simplify_problem_2_l1948_194865


namespace lines_parallel_l1948_194871

/--
Given two lines represented by the equations \(2x + my - 2m + 4 = 0\) and \(mx + 2y - m + 2 = 0\), 
prove that the value of \(m\) that makes these two lines parallel is \(m = -2\).
-/
theorem lines_parallel (m : ℝ) : 
    (∀ x y : ℝ, 2 * x + m * y - 2 * m + 4 = 0) ∧ (∀ x y : ℝ, m * x + 2 * y - m + 2 = 0) 
    → m = -2 :=
by
  sorry

end lines_parallel_l1948_194871


namespace find_m_l1948_194808

theorem find_m 
(x0 m : ℝ)
(h1 : m ≠ 0)
(h2 : x0^2 - x0 + m = 0)
(h3 : (2 * x0)^2 - 2 * x0 + 3 * m = 0)
: m = -2 :=
sorry

end find_m_l1948_194808


namespace test_score_range_l1948_194820

theorem test_score_range
  (mark_score : ℕ) (least_score : ℕ) (highest_score : ℕ)
  (twice_least_score : mark_score = 2 * least_score)
  (mark_fixed : mark_score = 46)
  (highest_fixed : highest_score = 98) :
  (highest_score - least_score) = 75 :=
by
  sorry

end test_score_range_l1948_194820


namespace Tara_loss_point_l1948_194878

theorem Tara_loss_point :
  ∀ (clarinet_cost initial_savings book_price total_books_sold additional_books books_sold_to_goal) 
  (H1 : initial_savings = 10)
  (H2 : clarinet_cost = 90)
  (H3 : book_price = 5)
  (H4 : total_books_sold = 25)
  (H5 : books_sold_to_goal = (clarinet_cost - initial_savings) / book_price)
  (H6 : additional_books = total_books_sold - books_sold_to_goal),
  additional_books * book_price = 45 :=
by
  intros clarinet_cost initial_savings book_price total_books_sold additional_books books_sold_to_goal
  intros H1 H2 H3 H4 H5 H6
  sorry

end Tara_loss_point_l1948_194878


namespace intersection_of_A_and_B_l1948_194888

def setA : Set ℕ := {1, 2, 3}
def setB : Set ℕ := {2, 4, 6}

theorem intersection_of_A_and_B : setA ∩ setB = {2} :=
by
  sorry

end intersection_of_A_and_B_l1948_194888


namespace percentage_of_divisible_l1948_194868

def count_divisible (n m : ℕ) : ℕ :=
(n / m)

def calculate_percentage (part total : ℕ) : ℚ :=
(part * 100 : ℚ) / (total : ℚ)

theorem percentage_of_divisible (n : ℕ) (k : ℕ) (h₁ : n = 150) (h₂ : k = 6) :
  calculate_percentage (count_divisible n k) n = 16.67 :=
by
  sorry

end percentage_of_divisible_l1948_194868


namespace find_second_number_l1948_194828

variable (A B : ℕ)

def is_LCM (a b lcm : ℕ) := Nat.lcm a b = lcm
def is_HCF (a b hcf : ℕ) := Nat.gcd a b = hcf

theorem find_second_number (h_lcm : is_LCM 330 B 2310) (h_hcf : is_HCF 330 B 30) : B = 210 := by
  sorry

end find_second_number_l1948_194828


namespace mike_total_time_spent_l1948_194815

theorem mike_total_time_spent : 
  let hours_watching_tv_per_day := 4
  let days_per_week := 7
  let days_playing_video_games := 3
  let hours_playing_video_games_per_day := hours_watching_tv_per_day / 2
  let total_hours_watching_tv := hours_watching_tv_per_day * days_per_week
  let total_hours_playing_video_games := hours_playing_video_games_per_day * days_playing_video_games
  let total_time_spent := total_hours_watching_tv + total_hours_playing_video_games
  total_time_spent = 34 :=
by
  sorry

end mike_total_time_spent_l1948_194815


namespace graph_of_function_does_not_pass_through_first_quadrant_l1948_194802

theorem graph_of_function_does_not_pass_through_first_quadrant (k : ℝ) (h : k < 0) : 
  ¬(∃ x y : ℝ, y = k * (x - k) ∧ x > 0 ∧ y > 0) :=
sorry

end graph_of_function_does_not_pass_through_first_quadrant_l1948_194802


namespace problem1_problem2_l1948_194821

-- Define the first problem
theorem problem1 (x : ℝ) : (x - 2) ^ 2 = 2 * x - 4 ↔ (x = 2 ∨ x = 4) := 
by 
  sorry

-- Define the second problem using completing the square method
theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) := 
by 
  sorry

end problem1_problem2_l1948_194821


namespace value_of_x_plus_2y_l1948_194862

theorem value_of_x_plus_2y :
  let x := 3
  let y := 1
  x + 2 * y = 5 :=
by
  sorry

end value_of_x_plus_2y_l1948_194862


namespace invest_in_yourself_examples_l1948_194876

theorem invest_in_yourself_examples (example1 example2 example3 : String)
  (benefit1 benefit2 benefit3 : String)
  (h1 : example1 = "Investment in Education")
  (h2 : benefit1 = "Spending money on education improves knowledge and skills, leading to better job opportunities and higher salaries. Education appreciates over time, providing financial stability.")
  (h3 : example2 = "Investment in Physical Health")
  (h4 : benefit2 = "Spending on sports activities, fitness programs, or healthcare prevents chronic diseases, saves future medical expenses, and enhances overall well-being.")
  (h5 : example3 = "Time Spent on Reading Books")
  (h6 : benefit3 = "Reading books expands knowledge, improves vocabulary and cognitive abilities, develops critical thinking and analytical skills, and fosters creativity and empathy."):
  "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." = "Investments in oneself, such as education, physical health, and reading, provide long-term benefits and can significantly improve one's quality of life and financial stability." :=
by
  sorry

end invest_in_yourself_examples_l1948_194876


namespace combined_weight_l1948_194842

theorem combined_weight (mary_weight : ℝ) (jamison_weight : ℝ) (john_weight : ℝ) :
  mary_weight = 160 ∧ jamison_weight = mary_weight + 20 ∧ john_weight = mary_weight + (0.25 * mary_weight) →
  john_weight + mary_weight + jamison_weight = 540 :=
by
  intros h
  obtain ⟨hm, hj, hj'⟩ := h
  rw [hm, hj, hj']
  norm_num
  sorry

end combined_weight_l1948_194842


namespace parallel_vectors_x_value_l1948_194823

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

-- Define the condition that vectors are parallel
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- State the problem: if a and b are parallel, then x = 1/2
theorem parallel_vectors_x_value (x : ℝ) (h : is_parallel a (b x)) : x = 1/2 :=
by
  sorry

end parallel_vectors_x_value_l1948_194823


namespace average_speed_calculation_l1948_194864

def average_speed (s1 s2 t1 t2 : ℕ) : ℕ :=
  (s1 * t1 + s2 * t2) / (t1 + t2)

theorem average_speed_calculation :
  average_speed 40 60 1 3 = 55 :=
by
  -- skipping the proof
  sorry

end average_speed_calculation_l1948_194864


namespace quadratic_equation_solution_l1948_194819

theorem quadratic_equation_solution (m : ℝ) :
  (m - 3) * x ^ (m^2 - 7) - x + 3 = 0 → m^2 - 7 = 2 → m ≠ 3 → m = -3 :=
by
  intros h_eq h_power h_nonzero
  sorry

end quadratic_equation_solution_l1948_194819


namespace possible_values_a_possible_values_m_l1948_194849

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a + 1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 2 = 0}

theorem possible_values_a (a : ℝ) : 
  (A ∪ B a = A) → a = 2 ∨ a = 3 := sorry

theorem possible_values_m (m : ℝ) : 
  (A ∩ C m = C m) → (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := sorry

end possible_values_a_possible_values_m_l1948_194849


namespace modulus_product_l1948_194855

open Complex

theorem modulus_product :
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  (Complex.abs (z1 * z2)) = Real.sqrt 8450 :=
by
  let z1 := mk 7 (-4)
  let z2 := mk 3 11
  sorry

end modulus_product_l1948_194855


namespace sin_double_angle_plus_pi_over_2_l1948_194866

theorem sin_double_angle_plus_pi_over_2 (θ : ℝ) (h : Real.cos θ = -1/3) :
  Real.sin (2 * θ + Real.pi / 2) = -7/9 :=
sorry

end sin_double_angle_plus_pi_over_2_l1948_194866


namespace abs_eq_zero_iff_l1948_194893

theorem abs_eq_zero_iff {a : ℝ} (h : |a + 3| = 0) : a = -3 :=
sorry

end abs_eq_zero_iff_l1948_194893


namespace roots_condition_implies_m_range_l1948_194852

theorem roots_condition_implies_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 1 ∧ x₂ > 1 ∧ (x₁^2 + (m-1)*x₁ + m^2 - 2 = 0) ∧ (x₂^2 + (m-1)*x₂ + m^2 - 2 = 0))
  → -2 < m ∧ m < 1 :=
by
  sorry

end roots_condition_implies_m_range_l1948_194852
