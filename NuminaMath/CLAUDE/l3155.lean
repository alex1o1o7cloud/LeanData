import Mathlib

namespace quadratic_sum_l3155_315574

theorem quadratic_sum (x : ℝ) : 
  let f : ℝ → ℝ := λ x => -3 * x^2 + 27 * x - 153
  ∃ (a b c : ℝ), (∀ x, f x = a * (x + b)^2 + c) ∧ (a + b + c = -99.75) := by
  sorry

end quadratic_sum_l3155_315574


namespace nine_integer_chords_l3155_315533

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Counts the number of integer-length chords through P -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem nine_integer_chords :
  let c := CircleWithPoint.mk 20 12
  count_integer_chords c = 9 := by
  sorry

end nine_integer_chords_l3155_315533


namespace four_even_cards_different_suits_count_l3155_315577

/-- Represents a standard playing card suit -/
inductive Suit
| hearts
| diamonds
| clubs
| spades

/-- Represents an even-numbered card (including face cards) -/
inductive EvenCard
| two
| four
| six
| eight
| ten
| queen

/-- The number of suits in a standard deck -/
def number_of_suits : Nat := 4

/-- The number of even-numbered cards in each suit -/
def even_cards_per_suit : Nat := 6

/-- A function to calculate the number of ways to choose 4 cards from a standard deck
    under the given conditions -/
def choose_four_even_cards_different_suits : Nat :=
  number_of_suits * even_cards_per_suit ^ 4

/-- The theorem stating that the number of ways to choose 4 cards from a standard deck,
    where all four cards are of different suits, each card is even-numbered,
    and the order doesn't matter, is equal to 1296 -/
theorem four_even_cards_different_suits_count :
  choose_four_even_cards_different_suits = 1296 := by
  sorry


end four_even_cards_different_suits_count_l3155_315577


namespace f_satisfies_equation_l3155_315587

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x + 1

-- State the theorem
theorem f_satisfies_equation : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1 := by
  sorry

end f_satisfies_equation_l3155_315587


namespace cylinder_to_cone_volume_l3155_315570

/-- Given a cylindrical block carved into the largest possible cone, 
    if the volume of the part removed is 25.12 cubic centimeters, 
    then the volume of the original cylindrical block is 37.68 cubic centimeters 
    and the volume of the cone-shaped block is 12.56 cubic centimeters. -/
theorem cylinder_to_cone_volume (removed_volume : ℝ) 
  (h : removed_volume = 25.12) : 
  ∃ (cylinder_volume cone_volume : ℝ),
    cylinder_volume = 37.68 ∧ 
    cone_volume = 12.56 ∧
    removed_volume = cylinder_volume - cone_volume := by
  sorry

end cylinder_to_cone_volume_l3155_315570


namespace one_real_root_condition_l3155_315549

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (k : ℝ) (x : ℝ) : Prop := lg (k * x) = 2 * lg (x + 1)

-- Theorem statement
theorem one_real_root_condition (k : ℝ) : 
  (∃! x : ℝ, equation k x) ↔ (k = 4 ∨ k < 0) :=
sorry

end one_real_root_condition_l3155_315549


namespace line_segment_ratio_l3155_315569

/-- Given five points P, Q, R, S, T on a line in that order, with specified distances between them,
    prove that the ratio of PR to ST is 9/10. -/
theorem line_segment_ratio (P Q R S T : ℝ) : 
  P < Q ∧ Q < R ∧ R < S ∧ S < T →  -- Points are in order
  Q - P = 3 →                      -- PQ = 3
  R - Q = 6 →                      -- QR = 6
  S - R = 4 →                      -- RS = 4
  T - S = 10 →                     -- ST = 10
  T - P = 30 →                     -- Total distance PT = 30
  (R - P) / (T - S) = 9 / 10 :=    -- Ratio of PR to ST
by
  sorry

end line_segment_ratio_l3155_315569


namespace factor_difference_of_squares_l3155_315595

theorem factor_difference_of_squares (x : ℝ) : x^2 - 144 = (x - 12) * (x + 12) := by
  sorry

end factor_difference_of_squares_l3155_315595


namespace sqrt_x_minus_2_real_iff_x_geq_2_l3155_315531

theorem sqrt_x_minus_2_real_iff_x_geq_2 (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_x_minus_2_real_iff_x_geq_2_l3155_315531


namespace l_shape_surface_area_l3155_315584

/-- Represents the 'L' shaped solid described in the problem -/
structure LShape where
  base_cubes : Nat
  column_cubes : Nat
  base_length : Nat
  base_width : Nat
  extension_length : Nat

/-- Calculates the surface area of the 'L' shaped solid -/
def surface_area (shape : LShape) : Nat :=
  let base_area := shape.base_cubes
  let top_exposed := shape.base_cubes - 1
  let column_sides := 4 * shape.column_cubes
  let column_top := 1
  let base_perimeter := 2 * (shape.base_length + shape.base_width + 2 * shape.extension_length)
  top_exposed + column_sides + column_top + base_perimeter

/-- The specific 'L' shape described in the problem -/
def problem_shape : LShape := {
  base_cubes := 8
  column_cubes := 7
  base_length := 3
  base_width := 2
  extension_length := 2
}

theorem l_shape_surface_area :
  surface_area problem_shape = 58 := by sorry

end l_shape_surface_area_l3155_315584


namespace jake_has_eleven_apples_l3155_315526

-- Define the number of peaches and apples for Steven
def steven_peaches : ℕ := 9
def steven_apples : ℕ := 8

-- Define Jake's peaches and apples in relation to Steven's
def jake_peaches : ℕ := steven_peaches - 13
def jake_apples : ℕ := steven_apples + 3

-- Theorem to prove
theorem jake_has_eleven_apples : jake_apples = 11 := by
  sorry

end jake_has_eleven_apples_l3155_315526


namespace function_value_at_alpha_l3155_315507

theorem function_value_at_alpha (α : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.cos x ^ 4 + Real.sin x ^ 4
  Real.sin (2 * α) = 2 / 3 →
  f α = 7 / 9 := by
sorry

end function_value_at_alpha_l3155_315507


namespace first_equation_is_double_root_second_equation_double_root_condition_l3155_315550

/-- Definition of a double root equation -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ (a * x^2 + b * x + c = 0) ∧ (a * y^2 + b * y + c = 0) ∧ (x = 2 * y ∨ y = 2 * x)

/-- Theorem for the first part of the problem -/
theorem first_equation_is_double_root : is_double_root_equation 1 (-6) 8 := by sorry

/-- Theorem for the second part of the problem -/
theorem second_equation_double_root_condition (n : ℝ) : 
  is_double_root_equation 1 (-8 - n) (8 * n) → n = 4 ∨ n = 16 := by sorry

end first_equation_is_double_root_second_equation_double_root_condition_l3155_315550


namespace afternoon_fish_count_l3155_315523

/-- Proves that the number of fish caught in the afternoon is 3 --/
theorem afternoon_fish_count (morning_a : ℕ) (morning_b : ℕ) (total : ℕ)
  (h1 : morning_a = 4)
  (h2 : morning_b = 3)
  (h3 : total = 10) :
  total - (morning_a + morning_b) = 3 := by
  sorry

end afternoon_fish_count_l3155_315523


namespace investment_rate_proof_l3155_315521

/-- Calculates the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that the given investment scenario results in a 10% annual interest rate -/
theorem investment_rate_proof (principal : ℝ) (final_amount : ℝ) (time : ℕ) 
  (h1 : principal = 5000)
  (h2 : final_amount = 6050.000000000001)
  (h3 : time = 2) :
  ∃ (rate : ℝ), compound_interest principal rate time = final_amount ∧ rate = 0.1 := by
  sorry

#check investment_rate_proof

end investment_rate_proof_l3155_315521


namespace relationship_between_exponents_l3155_315538

theorem relationship_between_exponents 
  (a b c : ℝ) (x y q z : ℝ) 
  (h1 : a^x = c^q) (h2 : a^x = b^2) (h3 : c^q = b^2)
  (h4 : c^y = a^z) (h5 : c^y = b^3) (h6 : a^z = b^3) :
  x * q = y * z := by
  sorry

end relationship_between_exponents_l3155_315538


namespace part1_part2_l3155_315529

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + a*x + 3

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2) 2, f a x ≥ a) ↔ a ∈ Set.Icc (-7) 2 :=
sorry

-- Part 2
theorem part2 (x : ℝ) :
  (∀ a ∈ Set.Icc 4 6, f a x ≥ 0) ↔ 
  x ∈ Set.Iic (-3 - Real.sqrt 6) ∪ Set.Ici (-3 + Real.sqrt 6) :=
sorry

end part1_part2_l3155_315529


namespace problem_solution_l3155_315578

/-- The function f(x) as defined in the problem -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (1/2) * (t * Real.log (x + 2) - Real.log (x - 2))

/-- The function F(x) as defined in the problem -/
noncomputable def F (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := a * Real.log (x - 1) - f t x

/-- Theorem stating the main results of the problem -/
theorem problem_solution :
  ∃ (t : ℝ),
    (∀ x : ℝ, f t x ≥ f t 4) ∧
    (t = 3) ∧
    (∀ x ∈ Set.Icc 3 7, f t x ≤ f t 7) ∧
    (∀ a : ℝ, (∀ x > 2, Monotone (F a t)) ↔ a ≥ 1) :=
by sorry

end problem_solution_l3155_315578


namespace even_polynomial_iff_product_with_negation_l3155_315548

/-- A polynomial over the complex numbers -/
def ComplexPolynomial := ℂ → ℂ

/-- Definition of an even polynomial -/
def IsEvenPolynomial (P : ComplexPolynomial) : Prop :=
  ∀ z : ℂ, P z = P (-z)

/-- The main theorem -/
theorem even_polynomial_iff_product_with_negation (P : ComplexPolynomial) :
  IsEvenPolynomial P ↔ ∃ Q : ComplexPolynomial, ∀ z : ℂ, P z = Q z * Q (-z) := by
  sorry

end even_polynomial_iff_product_with_negation_l3155_315548


namespace a_bounded_by_two_l3155_315545

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem a_bounded_by_two
  (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (a : ℝ)
  (h_ineq : ∀ x : ℝ, f (a * 2^x) - f (4^x + 1) ≤ 0) :
  -2 ≤ a ∧ a ≤ 2 :=
sorry

end a_bounded_by_two_l3155_315545


namespace y_derivative_l3155_315557

noncomputable def y (x : ℝ) : ℝ := 
  (Real.cos (Real.tan (1/3)) * (Real.sin (15*x))^2) / (15 * Real.cos (30*x))

theorem y_derivative (x : ℝ) : 
  deriv y x = (Real.cos (Real.tan (1/3)) * Real.tan (30*x)) / Real.cos (30*x) :=
by sorry

end y_derivative_l3155_315557


namespace probability_three_primes_in_six_rolls_l3155_315599

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def count_primes_on_12_sided_die : ℕ := 5

def probability_prime_on_12_sided_die : ℚ := 5 / 12

def probability_not_prime_on_12_sided_die : ℚ := 7 / 12

def number_of_ways_to_choose_3_out_of_6 : ℕ := 20

theorem probability_three_primes_in_six_rolls : 
  (probability_prime_on_12_sided_die ^ 3 * 
   probability_not_prime_on_12_sided_die ^ 3 * 
   number_of_ways_to_choose_3_out_of_6 : ℚ) = 3575 / 124416 := by sorry

end probability_three_primes_in_six_rolls_l3155_315599


namespace johns_age_l3155_315590

/-- Given that John is 30 years younger than his dad and the sum of their ages is 80 years, 
    prove that John is 25 years old. -/
theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 := by
  sorry

end johns_age_l3155_315590


namespace maria_towels_problem_l3155_315524

theorem maria_towels_problem (green_towels white_towels given_to_mother : ℝ) 
  (h1 : green_towels = 124.5)
  (h2 : white_towels = 67.7)
  (h3 : given_to_mother = 85.35) :
  green_towels + white_towels - given_to_mother = 106.85 := by
sorry

end maria_towels_problem_l3155_315524


namespace profit_increase_l3155_315552

theorem profit_increase (m : ℝ) : 
  (m + 8) / 0.92 = m + 10 → m = 15 := by
  sorry

end profit_increase_l3155_315552


namespace special_triangle_sides_l3155_315576

/-- A triangle with sides a, b, and c satisfying specific conditions -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  perimeter_eq : a + b + c = 18
  sum_eq_double_c : a + b = 2 * c
  b_eq_double_a : b = 2 * a

/-- Theorem stating the unique side lengths of the special triangle -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 4 ∧ t.b = 8 ∧ t.c = 6 := by
  sorry

end special_triangle_sides_l3155_315576


namespace symmetrical_circle_l3155_315517

/-- Given a circle with equation x² + y² + 2x = 0, 
    its symmetrical circle with respect to the y-axis 
    has the equation x² + y² - 2x = 0 -/
theorem symmetrical_circle (x y : ℝ) : 
  (x^2 + y^2 + 2*x = 0) → 
  ∃ (x' y' : ℝ), (x'^2 + y'^2 - 2*x' = 0 ∧ 
                  x' = -x ∧ 
                  y' = y) :=
by sorry

end symmetrical_circle_l3155_315517


namespace ellipse_focus_distance_l3155_315508

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- Distance from a point to a focus -/
noncomputable def distance_to_focus (x y : ℝ) (fx fy : ℝ) : ℝ :=
  Real.sqrt ((x - fx)^2 + (y - fy)^2)

/-- The statement to prove -/
theorem ellipse_focus_distance 
  (x y : ℝ) 
  (h_on_ellipse : is_on_ellipse x y) 
  (f1x f1y f2x f2y : ℝ) 
  (h_focus1 : distance_to_focus x y f1x f1y = 7) :
  distance_to_focus x y f2x f2y = 3 :=
sorry

end ellipse_focus_distance_l3155_315508


namespace next_perfect_square_sum_l3155_315561

def children_ages : List ℕ := [1, 3, 5, 7, 9, 11, 13]

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def sum_ages (years_later : ℕ) : ℕ :=
  List.sum (List.map (· + years_later) children_ages)

theorem next_perfect_square_sum :
  (∃ (x : ℕ), x > 0 ∧ 
    is_perfect_square (sum_ages x) ∧
    (∀ y : ℕ, 0 < y ∧ y < x → ¬is_perfect_square (sum_ages y))) →
  (∃ (x : ℕ), x = 21 ∧ 
    is_perfect_square (sum_ages x) ∧
    (List.head! children_ages + x) + sum_ages x = 218) :=
by sorry

end next_perfect_square_sum_l3155_315561


namespace chebyshev_properties_l3155_315593

/-- Chebyshev polynomial of the first kind -/
def T : ℕ → (Real → Real)
| 0 => λ _ => 1
| 1 => λ x => x
| (n + 2) => λ x => 2 * x * T (n + 1) x - T n x

/-- Chebyshev polynomial of the second kind -/
def U : ℕ → (Real → Real)
| 0 => λ _ => 1
| 1 => λ x => 2 * x
| (n + 2) => λ x => 2 * x * U (n + 1) x - U n x

/-- Theorem: Chebyshev polynomials satisfy their initial conditions and recurrence relations -/
theorem chebyshev_properties :
  (∀ x, T 0 x = 1) ∧
  (∀ x, T 1 x = x) ∧
  (∀ n x, T (n + 1) x = 2 * x * T n x - T (n - 1) x) ∧
  (∀ x, U 0 x = 1) ∧
  (∀ x, U 1 x = 2 * x) ∧
  (∀ n x, U (n + 1) x = 2 * x * U n x - U (n - 1) x) := by
  sorry

end chebyshev_properties_l3155_315593


namespace simplify_expression_1_simplify_expression_2_l3155_315536

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  2 * (2*b - 3*a) + 3 * (2*a - 3*b) = -5*b := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  4*a^2 + 2*(3*a*b - 2*a^2) - (7*a*b - 1) = -a*b + 1 := by sorry

end simplify_expression_1_simplify_expression_2_l3155_315536


namespace hash_five_neg_one_l3155_315532

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * (y + 2) + x * y

-- Theorem statement
theorem hash_five_neg_one : hash 5 (-1) = 0 := by
  sorry

end hash_five_neg_one_l3155_315532


namespace jills_age_l3155_315546

theorem jills_age (henry_age jill_age : ℕ) : 
  (henry_age + jill_age = 48) →
  (henry_age - 9 = 2 * (jill_age - 9)) →
  jill_age = 19 :=
by
  sorry

end jills_age_l3155_315546


namespace tech_ownership_1995_l3155_315588

/-- The percentage of families owning computers, tablets, and smartphones in City X in 1995 -/
def tech_ownership (pc_1992 : ℝ) (pc_increase_1993 : ℝ) (family_increase_1993 : ℝ)
                   (tablet_adoption_1994 : ℝ) (smartphone_adoption_1995 : ℝ) : ℝ :=
  let pc_1993 := pc_1992 * (1 + pc_increase_1993)
  let pc_tablet_1994 := pc_1993 * tablet_adoption_1994
  pc_tablet_1994 * smartphone_adoption_1995

theorem tech_ownership_1995 :
  tech_ownership 0.6 0.5 0.03 0.4 0.3 = 0.108 := by
  sorry

end tech_ownership_1995_l3155_315588


namespace condition_necessary_not_sufficient_l3155_315535

theorem condition_necessary_not_sufficient :
  (∀ x y : ℝ, x = 1 ∧ y = 2 → x + y = 3) ∧
  (∃ x y : ℝ, x + y = 3 ∧ (x ≠ 1 ∨ y ≠ 2)) := by
  sorry

end condition_necessary_not_sufficient_l3155_315535


namespace semicircle_perimeter_approx_l3155_315580

/-- The perimeter of a semicircle with radius 10 is approximately 51.4 -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |((10 : ℝ) * Real.pi + 20) - 51.4| < ε := by
  sorry

end semicircle_perimeter_approx_l3155_315580


namespace absent_workers_l3155_315525

theorem absent_workers (total_workers : ℕ) (original_days : ℕ) (actual_days : ℕ) 
  (h1 : total_workers = 42)
  (h2 : original_days = 12)
  (h3 : actual_days = 14) :
  ∃ (absent : ℕ), 
    absent = 6 ∧ 
    (total_workers * original_days = (total_workers - absent) * actual_days) :=
by sorry

end absent_workers_l3155_315525


namespace special_sequence_a6_l3155_315560

/-- A sequence where a₂ = 3, a₄ = 15, and {aₙ + 1} is a geometric sequence -/
def special_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 3 ∧ a 4 = 15 ∧ ∃ q : ℝ, ∀ n : ℕ, (a (n + 1) + 1) = (a n + 1) * q

theorem special_sequence_a6 (a : ℕ → ℝ) (h : special_sequence a) : a 6 = 63 := by
  sorry

end special_sequence_a6_l3155_315560


namespace football_field_lap_time_l3155_315542

-- Define the field dimensions
def field_length : ℝ := 100
def field_width : ℝ := 50

-- Define the number of laps and obstacles
def num_laps : ℕ := 6
def num_obstacles : ℕ := 2

-- Define the additional distance per obstacle
def obstacle_distance : ℝ := 20

-- Define the average speed of the player
def average_speed : ℝ := 4

-- Theorem to prove
theorem football_field_lap_time :
  let perimeter : ℝ := 2 * (field_length + field_width)
  let total_obstacle_distance : ℝ := num_obstacles * obstacle_distance
  let lap_distance : ℝ := perimeter + total_obstacle_distance
  let total_distance : ℝ := num_laps * lap_distance
  let time_taken : ℝ := total_distance / average_speed
  time_taken = 510 := by sorry

end football_field_lap_time_l3155_315542


namespace rational_combination_equals_24_l3155_315581

theorem rational_combination_equals_24 :
  ∃ (f : List ℚ → ℚ),
    f [-1, -2, -3, -4] = 24 ∧
    (∀ x y z w, f [x, y, z, w] = ((x + y + z) * w) ∨
                f [x, y, z, w] = ((x + y + z) / w) ∨
                f [x, y, z, w] = ((x + y - z) * w) ∨
                f [x, y, z, w] = ((x + y - z) / w) ∨
                f [x, y, z, w] = ((x - y + z) * w) ∨
                f [x, y, z, w] = ((x - y + z) / w) ∨
                f [x, y, z, w] = ((x - y - z) * w) ∨
                f [x, y, z, w] = ((x - y - z) / w)) :=
by
  sorry

end rational_combination_equals_24_l3155_315581


namespace problem_statement_l3155_315543

theorem problem_statement : 
  ∃ d : ℝ, 5^(Real.log 30) * (1/3)^(Real.log 0.5) = d ∧ d = 30 := by
  sorry

end problem_statement_l3155_315543


namespace alpha_necessary_not_sufficient_l3155_315522

-- Define the conditions
def α (x : ℝ) : Prop := x^2 = 4
def β (x : ℝ) : Prop := x = 2

-- State the theorem
theorem alpha_necessary_not_sufficient :
  (∀ x, β x → α x) ∧ (∃ x, α x ∧ ¬β x) := by sorry

end alpha_necessary_not_sufficient_l3155_315522


namespace factoring_expression_l3155_315540

theorem factoring_expression (y : ℝ) : 5 * y * (y + 2) + 9 * (y + 2) = (5 * y + 9) * (y + 2) := by
  sorry

end factoring_expression_l3155_315540


namespace election_winner_percentage_l3155_315510

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 868 → 
  margin = 336 → 
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 62 / 100 := by
  sorry

end election_winner_percentage_l3155_315510


namespace total_age_is_877_l3155_315558

def family_gathering (T : ℕ) : Prop :=
  ∃ (father mother brother sister elder_cousin younger_cousin grandmother uncle aunt kaydence : ℕ),
    father = 60 ∧
    mother = father - 2 ∧
    brother = father / 2 ∧
    sister = 40 ∧
    elder_cousin = brother + 2 * sister ∧
    younger_cousin = elder_cousin / 2 + 3 ∧
    grandmother = 3 * mother - 5 ∧
    uncle = 5 * younger_cousin - 10 ∧
    aunt = 2 * mother + 7 ∧
    5 * kaydence = 2 * aunt ∧
    T = father + mother + brother + sister + elder_cousin + younger_cousin + grandmother + uncle + aunt + kaydence

theorem total_age_is_877 : family_gathering 877 := by
  sorry

end total_age_is_877_l3155_315558


namespace gcd_linear_combination_l3155_315596

theorem gcd_linear_combination (a b : ℤ) : Int.gcd (5*a + 3*b) (13*a + 8*b) = Int.gcd a b := by
  sorry

end gcd_linear_combination_l3155_315596


namespace picture_area_l3155_315511

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h_frame_area : (2 * x + 3) * (y + 2) - x * y = 34) : x * y = 8 := by
  sorry

end picture_area_l3155_315511


namespace factor_x_squared_minus_81_l3155_315592

theorem factor_x_squared_minus_81 (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := by
  sorry

end factor_x_squared_minus_81_l3155_315592


namespace consecutive_sum_equals_fourteen_l3155_315563

theorem consecutive_sum_equals_fourteen (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) = 14 → n = 2 := by
  sorry

end consecutive_sum_equals_fourteen_l3155_315563


namespace max_a_for_monotonic_f_l3155_315575

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem max_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y → f a x ≤ f a y) →
  a ≤ 0 ∧ ∀ b : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y → f b x ≤ f b y) → b ≤ a :=
sorry

end max_a_for_monotonic_f_l3155_315575


namespace parabola_through_fixed_point_l3155_315582

-- Define the line equation as a function of a
def line_equation (a x y : ℝ) : Prop := (a - 1) * x - y + 2 * a + 1 = 0

-- Define the fixed point P
def fixed_point : ℝ × ℝ := (-2, 3)

-- Define the two possible parabola equations
def parabola1 (x y : ℝ) : Prop := y^2 = -9/2 * x
def parabola2 (x y : ℝ) : Prop := x^2 = 4/3 * y

-- State the theorem
theorem parabola_through_fixed_point :
  (∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2)) →
  (parabola1 (fixed_point.1) (fixed_point.2) ∨ parabola2 (fixed_point.1) (fixed_point.2)) :=
sorry

end parabola_through_fixed_point_l3155_315582


namespace room_width_l3155_315544

/-- Proves that a rectangular room with given volume, length, and height has a specific width -/
theorem room_width (volume : ℝ) (length : ℝ) (height : ℝ) (width : ℝ) 
  (h_volume : volume = 10000)
  (h_length : length = 100)
  (h_height : height = 10)
  (h_relation : volume = length * width * height) :
  width = 10 := by
  sorry

end room_width_l3155_315544


namespace trinomial_square_l3155_315506

theorem trinomial_square (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 9*x^2 + 21*x + a = (3*x + b)^2) → a = 49/4 := by
  sorry

end trinomial_square_l3155_315506


namespace expression_factorization_l3155_315556

theorem expression_factorization (x y z : ℝ) :
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / 
  ((x - y)^3 + (y - z)^3 + (z - x)^3) = 
  (x + y) * (y + z) * (z + x) :=
by sorry

end expression_factorization_l3155_315556


namespace ramu_car_profit_percent_l3155_315518

/-- Calculates the profit percent given the purchase price, repair cost, and selling price of a car. -/
def profit_percent (purchase_price repair_cost selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem stating that the profit percent for Ramu's car transaction is 29.8% -/
theorem ramu_car_profit_percent :
  profit_percent 42000 8000 64900 = 29.8 :=
by sorry

end ramu_car_profit_percent_l3155_315518


namespace total_savings_three_months_l3155_315519

def savings (n : ℕ) : ℕ := 10 + 30 * n

theorem total_savings_three_months :
  savings 0 + savings 1 + savings 2 = 120 := by
  sorry

end total_savings_three_months_l3155_315519


namespace price_reduction_problem_l3155_315514

theorem price_reduction_problem (x : ℝ) : 
  (∀ (P : ℝ), P > 0 → 
    P * (1 - x / 100) * (1 - 20 / 100) = P * (1 - 40 / 100)) → 
  x = 25 := by
  sorry

end price_reduction_problem_l3155_315514


namespace existence_of_unachievable_fraction_l3155_315597

/-- Given an odd prime p, this theorem proves the existence of a specific fraction that cannot be achieved by any coloring of integers. -/
theorem existence_of_unachievable_fraction (p : Nat) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ a : Nat, 0 < a ∧ a < p ∧
  ∀ (coloring : Nat → Bool) (N : Nat),
    N = (p^3 - p) / 4 - 1 →
    ∀ n : Nat, 0 < n ∧ n ≤ N →
      (Finset.filter (fun i => coloring i) (Finset.range n)).card ≠ n * a / p :=
by sorry

end existence_of_unachievable_fraction_l3155_315597


namespace square_division_impossible_l3155_315586

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane -/
structure Square where
  sideLength : ℝ
  center : Point

/-- Represents a division of a square by two internal points -/
structure SquareDivision where
  square : Square
  point1 : Point
  point2 : Point

/-- Checks if a point is inside a square -/
def isPointInsideSquare (s : Square) (p : Point) : Prop :=
  abs (p.x - s.center.x) ≤ s.sideLength / 2 ∧ abs (p.y - s.center.y) ≤ s.sideLength / 2

/-- Checks if a square division results in 9 equal parts -/
def isDividedIntoNineEqualParts (sd : SquareDivision) : Prop :=
  ∃ (areas : Finset ℝ), areas.card = 9 ∧ 
  (∀ a ∈ areas, a = sd.square.sideLength^2 / 9) ∧
  (isPointInsideSquare sd.square sd.point1) ∧
  (isPointInsideSquare sd.square sd.point2)

/-- Theorem stating that it's impossible to divide a square into 9 equal parts
    by connecting two internal points to its vertices -/
theorem square_division_impossible :
  ¬ ∃ (sd : SquareDivision), isDividedIntoNineEqualParts sd :=
sorry

end square_division_impossible_l3155_315586


namespace sum_x_y_equals_two_l3155_315505

theorem sum_x_y_equals_two (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 1))
  (h2 : (5 : ℝ) ^ (2 * y) = 25 ^ (x - 2)) : 
  x + y = 2 := by
sorry

end sum_x_y_equals_two_l3155_315505


namespace treats_calculation_l3155_315572

/-- Calculates the number of treats per child per house -/
def treats_per_child_per_house (num_children : ℕ) (num_hours : ℕ) (houses_per_hour : ℕ) (total_treats : ℕ) : ℚ :=
  (total_treats : ℚ) / ((num_children : ℚ) * (num_hours * houses_per_hour))

/-- Theorem: Given the conditions from the problem, the number of treats per child per house is 3 -/
theorem treats_calculation :
  let num_children : ℕ := 3
  let num_hours : ℕ := 4
  let houses_per_hour : ℕ := 5
  let total_treats : ℕ := 180
  treats_per_child_per_house num_children num_hours houses_per_hour total_treats = 3 := by
  sorry

#eval treats_per_child_per_house 3 4 5 180

end treats_calculation_l3155_315572


namespace product_divisibility_l3155_315598

theorem product_divisibility (a b c : ℤ) 
  (h1 : (a + b + c)^2 = -(a*b + a*c + b*c))
  (h2 : a + b ≠ 0)
  (h3 : b + c ≠ 0)
  (h4 : a + c ≠ 0) :
  (∃ k : ℤ, (a + b) * (a + c) = k * (b + c)) ∧
  (∃ k : ℤ, (a + b) * (b + c) = k * (a + c)) ∧
  (∃ k : ℤ, (a + c) * (b + c) = k * (a + b)) :=
sorry

end product_divisibility_l3155_315598


namespace sams_initial_dimes_l3155_315554

/-- The problem of determining Sam's initial number of dimes -/
theorem sams_initial_dimes :
  ∀ (initial_dimes current_dimes : ℕ),
    initial_dimes - 4 = current_dimes →
    current_dimes = 4 →
    initial_dimes = 8 := by
  sorry

end sams_initial_dimes_l3155_315554


namespace max_candy_count_l3155_315583

/-- Represents the state of the board and candy count -/
structure BoardState where
  numbers : List Nat
  candy_count : Nat

/-- Combines two numbers on the board and updates the candy count -/
def combine_numbers (state : BoardState) (i j : Nat) : BoardState :=
  { numbers := (state.numbers.removeNth i).removeNth j ++ [state.numbers[i]! + state.numbers[j]!],
    candy_count := state.candy_count + state.numbers[i]! * state.numbers[j]! }

/-- Theorem: The maximum number of candies Karlson can eat is 300 -/
theorem max_candy_count :
  ∃ (final_state : BoardState),
    (final_state.numbers.length = 1) ∧
    (final_state.candy_count = 300) ∧
    (∃ (initial_state : BoardState),
      (initial_state.numbers = List.replicate 25 1) ∧
      (∃ (moves : List (Nat × Nat)),
        moves.length = 24 ∧
        final_state = moves.foldl (fun state (i, j) => combine_numbers state i j) initial_state)) :=
by
  sorry

#check max_candy_count

end max_candy_count_l3155_315583


namespace book_distribution_l3155_315509

theorem book_distribution (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 6 →
  k = 3 →
  m = 2 →
  (Nat.choose n m) * (Nat.choose (n - m) m) * (Nat.choose (n - 2*m) m) = 90 :=
by sorry

end book_distribution_l3155_315509


namespace roots_of_equation_l3155_315527

theorem roots_of_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ = -1 ∧ x₂ = 0) ∧ 
  (∀ x : ℝ, (x + 1) * x = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end roots_of_equation_l3155_315527


namespace parallel_vectors_m_value_l3155_315547

/-- Given vectors a and b, if a + 2b is parallel to ma + b, then m = 1/2 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (ha : a = (2, 3)) 
  (hb : b = (1, 2)) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a + 2 • b = k • (m • a + b)) : 
  m = 1/2 := by
sorry

end parallel_vectors_m_value_l3155_315547


namespace parking_lot_car_difference_l3155_315564

theorem parking_lot_car_difference (initial_cars : ℕ) (cars_left : ℕ) (current_cars : ℕ) : 
  initial_cars = 80 → cars_left = 13 → current_cars = 85 → 
  (current_cars - initial_cars) + cars_left = 18 := by
  sorry

end parking_lot_car_difference_l3155_315564


namespace point_Q_in_first_quadrant_l3155_315534

-- Define the conditions for point P
def fourth_quadrant (a b : ℝ) : Prop := a > 0 ∧ b < 0

-- Define the condition |a| > |b|
def magnitude_condition (a b : ℝ) : Prop := abs a > abs b

-- Define what it means for a point to be in the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem point_Q_in_first_quadrant (a b : ℝ) 
  (h1 : fourth_quadrant a b) (h2 : magnitude_condition a b) : 
  first_quadrant (a + b) (a - b) := by
  sorry

end point_Q_in_first_quadrant_l3155_315534


namespace polynomial_factorization_l3155_315589

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 8*x^6 + 24*x^4 - 32*x^2 + 16 = (x - Real.sqrt 2)^4 * (x + Real.sqrt 2)^4 := by
  sorry

end polynomial_factorization_l3155_315589


namespace solve_linear_equation_l3155_315591

theorem solve_linear_equation (x : ℝ) (h : 5*x - 7 = 15*x + 13) : 3*(x+10) = 24 := by
  sorry

end solve_linear_equation_l3155_315591


namespace joseph_cards_percentage_l3155_315515

theorem joseph_cards_percentage (initial_cards : ℕ) 
  (brother_fraction : ℚ) (friend_cards : ℕ) : 
  initial_cards = 16 →
  brother_fraction = 3/8 →
  friend_cards = 2 →
  (initial_cards - (initial_cards * brother_fraction).floor - friend_cards) / initial_cards * 100 = 50 := by
  sorry

end joseph_cards_percentage_l3155_315515


namespace find_q_l3155_315571

-- Define the polynomial g(x)
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p*x^4 + q*x^3 + r*x^2 + s*x + t

-- State the theorem
theorem find_q :
  ∀ p q r s t : ℝ,
  (∀ x : ℝ, g p q r s t x = 0 ↔ x = -2 ∨ x = 0 ∨ x = 1 ∨ x = 3) →
  g p q r s t 2 = -24 →
  q = 12 := by
sorry


end find_q_l3155_315571


namespace triangle_ratio_l3155_315565

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  A = Real.pi / 3 →
  b = 1 →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 :=
by sorry

end triangle_ratio_l3155_315565


namespace star_commutative_star_not_distributive_l3155_315539

/-- Binary operation ⋆ -/
def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

/-- Commutativity of ⋆ -/
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

/-- Non-distributivity of ⋆ over addition -/
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

end star_commutative_star_not_distributive_l3155_315539


namespace max_root_sum_l3155_315541

theorem max_root_sum (a b c : ℝ) : 
  (a^3 - 4 * Real.sqrt 3 * a^2 + 13 * a - 2 * Real.sqrt 3 = 0) →
  (b^3 - 4 * Real.sqrt 3 * b^2 + 13 * b - 2 * Real.sqrt 3 = 0) →
  (c^3 - 4 * Real.sqrt 3 * c^2 + 13 * c - 2 * Real.sqrt 3 = 0) →
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  max (a + b - c) (max (a - b + c) (-a + b + c)) = 2 * Real.sqrt 3 + 2 * Real.sqrt 2 :=
by sorry

end max_root_sum_l3155_315541


namespace polynomial_decrease_l3155_315513

theorem polynomial_decrease (b : ℝ) :
  let P : ℝ → ℝ := fun x ↦ -2 * x + b
  ∀ x : ℝ, P (x + 1) = P x - 2 := by
sorry

end polynomial_decrease_l3155_315513


namespace morning_fliers_fraction_l3155_315568

theorem morning_fliers_fraction (total : ℕ) (remaining : ℕ) : 
  total = 2500 → remaining = 1500 → 
  ∃ x : ℚ, x > 0 ∧ x < 1 ∧ 
  (1 - x) * total - (1 - x) * total / 4 = remaining ∧
  x = 1/5 := by
sorry

end morning_fliers_fraction_l3155_315568


namespace smallest_k_is_2010_l3155_315537

/-- A sequence of natural numbers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ n, 1005 ∣ a n ∨ 1006 ∣ a n) ∧
  (∀ n, ¬(97 ∣ a n))

/-- The difference between consecutive terms is at most k -/
def BoundedDifference (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n, a (n + 1) - a n ≤ k

/-- The theorem stating the smallest possible k -/
theorem smallest_k_is_2010 :
  (∃ a, ValidSequence a ∧ BoundedDifference a 2010) ∧
  (∀ k < 2010, ¬∃ a, ValidSequence a ∧ BoundedDifference a k) :=
sorry

end smallest_k_is_2010_l3155_315537


namespace min_balls_to_draw_theorem_l3155_315567

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to draw to ensure at least 15 of one color -/
def minBallsToDraw (counts : BallCounts) : Nat :=
  sorry

/-- The theorem stating the minimum number of balls to draw -/
theorem min_balls_to_draw_theorem (counts : BallCounts) 
  (h1 : counts.red = 28)
  (h2 : counts.green = 20)
  (h3 : counts.yellow = 13)
  (h4 : counts.blue = 19)
  (h5 : counts.white = 11)
  (h6 : counts.black = 9)
  (h_total : counts.red + counts.green + counts.yellow + counts.blue + counts.white + counts.black = 100) :
  minBallsToDraw counts = 76 :=
sorry

end min_balls_to_draw_theorem_l3155_315567


namespace lucas_raspberry_candies_l3155_315502

-- Define the variables
def original_raspberry : ℕ := sorry
def original_lemon : ℕ := sorry

-- Define the conditions
axiom initial_ratio : original_raspberry = 3 * original_lemon
axiom after_giving_away : original_raspberry - 5 = 4 * (original_lemon - 5)

-- Theorem to prove
theorem lucas_raspberry_candies : original_raspberry = 45 := by
  sorry

end lucas_raspberry_candies_l3155_315502


namespace square_area_l3155_315551

/-- The parabola function -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 4

/-- The line function -/
def g (x : ℝ) : ℝ := 3

/-- The theorem stating the area of the square -/
theorem square_area : 
  ∃ (x₁ x₂ : ℝ), 
    f x₁ = g x₁ ∧ 
    f x₂ = g x₂ ∧ 
    x₁ ≠ x₂ ∧
    (x₂ - x₁)^2 = 8 :=
sorry

end square_area_l3155_315551


namespace gcd_6273_14593_l3155_315579

theorem gcd_6273_14593 : Nat.gcd 6273 14593 = 3 := by sorry

end gcd_6273_14593_l3155_315579


namespace marias_green_towels_l3155_315594

theorem marias_green_towels :
  ∀ (green_towels : ℕ),
  (green_towels + 21 : ℕ) - 34 = 22 →
  green_towels = 35 :=
by
  sorry

end marias_green_towels_l3155_315594


namespace provisions_problem_l3155_315585

/-- The initial number of men given the conditions of the problem -/
def initial_men : ℕ := 1000

/-- The number of days the provisions last for the initial group -/
def initial_days : ℕ := 20

/-- The number of additional men that join the group -/
def additional_men : ℕ := 650

/-- The number of days the provisions last after additional men join -/
def final_days : ℚ := 12121212121212121 / 1000000000000000

theorem provisions_problem :
  initial_men * initial_days = (initial_men + additional_men) * final_days :=
sorry

end provisions_problem_l3155_315585


namespace alpha_beta_equivalence_l3155_315573

theorem alpha_beta_equivalence (α β : ℝ) :
  (α > β) ↔ (α + Real.sin α * Real.cos β > β + Real.sin β * Real.cos α) := by
  sorry

end alpha_beta_equivalence_l3155_315573


namespace total_pups_is_91_l3155_315500

/-- Represents the number of pups for each dog breed --/
structure DogBreedPups where
  huskies : Nat
  pitbulls : Nat
  goldenRetrievers : Nat
  germanShepherds : Nat
  bulldogs : Nat
  poodles : Nat

/-- Calculates the total number of pups from all dog breeds --/
def totalPups (d : DogBreedPups) : Nat :=
  d.huskies + d.pitbulls + d.goldenRetrievers + d.germanShepherds + d.bulldogs + d.poodles

/-- Theorem stating that the total number of pups is 91 --/
theorem total_pups_is_91 :
  let numHuskies := 5
  let numPitbulls := 2
  let numGoldenRetrievers := 4
  let numGermanShepherds := 3
  let numBulldogs := 2
  let numPoodles := 3
  let huskiePups := 4
  let pitbullPups := 3
  let goldenRetrieverPups := huskiePups + 2
  let germanShepherdPups := pitbullPups + 3
  let bulldogPups := 4
  let poodlePups := bulldogPups + 1
  let d := DogBreedPups.mk
    (numHuskies * huskiePups)
    (numPitbulls * pitbullPups)
    (numGoldenRetrievers * goldenRetrieverPups)
    (numGermanShepherds * germanShepherdPups)
    (numBulldogs * bulldogPups)
    (numPoodles * poodlePups)
  totalPups d = 91 := by
  sorry

end total_pups_is_91_l3155_315500


namespace bacteria_growth_days_l3155_315553

def initial_bacteria : ℕ := 5
def growth_rate : ℕ := 3
def target_bacteria : ℕ := 200

def bacteria_count (days : ℕ) : ℕ :=
  initial_bacteria * growth_rate ^ days

theorem bacteria_growth_days :
  (∀ k : ℕ, k < 4 → bacteria_count k ≤ target_bacteria) ∧
  bacteria_count 4 > target_bacteria :=
sorry

end bacteria_growth_days_l3155_315553


namespace x_value_possibilities_l3155_315562

theorem x_value_possibilities (x y p q : ℝ) (h1 : y ≠ 0) (h2 : q ≠ 0) 
  (h3 : |x / y| < |p| / q^2) :
  ∃ (x_neg x_zero x_pos : ℝ), 
    (x_neg < 0 ∧ |x_neg / y| < |p| / q^2) ∧
    (x_zero = 0 ∧ |x_zero / y| < |p| / q^2) ∧
    (x_pos > 0 ∧ |x_pos / y| < |p| / q^2) :=
by sorry

end x_value_possibilities_l3155_315562


namespace exp_sum_rule_l3155_315555

theorem exp_sum_rule (a b : ℝ) : Real.exp a * Real.exp b = Real.exp (a + b) := by
  sorry

end exp_sum_rule_l3155_315555


namespace modulus_of_z_l3155_315516

theorem modulus_of_z (z : ℂ) (h : z * (4 - 3*I) = 1) : Complex.abs z = 1/5 := by
  sorry

end modulus_of_z_l3155_315516


namespace backpack_cost_is_fifteen_l3155_315512

def total_spent : ℝ := 32
def pens_cost : ℝ := 1
def pencils_cost : ℝ := 1
def notebook_cost : ℝ := 3
def notebook_count : ℕ := 5

def backpack_cost : ℝ := total_spent - (pens_cost + pencils_cost + notebook_cost * notebook_count)

theorem backpack_cost_is_fifteen : backpack_cost = 15 := by
  sorry

end backpack_cost_is_fifteen_l3155_315512


namespace roots_sum_of_squares_l3155_315559

theorem roots_sum_of_squares (a b : ℝ) : 
  (∀ x, x^2 - 8*x + 8 = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = 48 := by
  sorry

end roots_sum_of_squares_l3155_315559


namespace total_people_in_park_l3155_315501

/-- The number of lines formed by people in the park -/
def num_lines : ℕ := 4

/-- The number of people in each line -/
def people_per_line : ℕ := 8

/-- The total number of people doing gymnastics in the park -/
def total_people : ℕ := num_lines * people_per_line

theorem total_people_in_park : total_people = 32 := by
  sorry

end total_people_in_park_l3155_315501


namespace n_in_interval_l3155_315528

def is_repeating_decimal (d : ℚ) (period : ℕ) : Prop :=
  ∃ (k : ℕ), d * 10^period - d.floor = k / (10^period - 1)

theorem n_in_interval (n : ℕ) (hn : n < 1000) 
  (h1 : is_repeating_decimal (1 / n) 3)
  (h2 : is_repeating_decimal (1 / (n + 4)) 6) :
  n ∈ Set.Icc 1 150 := by
  sorry

end n_in_interval_l3155_315528


namespace unique_solution_mod_37_l3155_315520

theorem unique_solution_mod_37 :
  ∃! (a b c d : ℤ),
    (a^2 + b*c) % 37 = a % 37 ∧
    (b*(a + d)) % 37 = b % 37 ∧
    (c*(a + d)) % 37 = c % 37 ∧
    (b*c + d^2) % 37 = d % 37 ∧
    (a*d - b*c) % 37 = 1 % 37 := by
  sorry

end unique_solution_mod_37_l3155_315520


namespace orange_count_indeterminate_l3155_315503

/-- Represents Philip's fruit collection -/
structure FruitCollection where
  banana_count : ℕ
  banana_groups : ℕ
  bananas_per_group : ℕ
  orange_groups : ℕ

/-- Predicate to check if the banana count is consistent with the groups and bananas per group -/
def banana_count_consistent (collection : FruitCollection) : Prop :=
  collection.banana_count = collection.banana_groups * collection.bananas_per_group

/-- Theorem stating that the number of oranges cannot be determined -/
theorem orange_count_indeterminate (collection : FruitCollection)
  (h1 : collection.banana_count = 290)
  (h2 : collection.banana_groups = 2)
  (h3 : collection.bananas_per_group = 145)
  (h4 : collection.orange_groups = 93)
  (h5 : banana_count_consistent collection) :
  ¬∃ (orange_count : ℕ), ∀ (other_collection : FruitCollection),
    collection.banana_count = other_collection.banana_count ∧
    collection.banana_groups = other_collection.banana_groups ∧
    collection.bananas_per_group = other_collection.bananas_per_group ∧
    collection.orange_groups = other_collection.orange_groups →
    orange_count = (other_collection.orange_groups : ℕ) * (orange_count / other_collection.orange_groups) :=
sorry

end orange_count_indeterminate_l3155_315503


namespace well_digging_payment_l3155_315504

/-- Calculates the total payment for a group of workers given their daily work hours and hourly rate -/
def totalPayment (numWorkers : ℕ) (dailyHours : List ℕ) (hourlyRate : ℕ) : ℕ :=
  numWorkers * (dailyHours.sum * hourlyRate)

/-- Proves that the total payment for 3 workers working 12, 10, 8, and 14 hours on four days at $15 per hour is $1980 -/
theorem well_digging_payment :
  totalPayment 3 [12, 10, 8, 14] 15 = 1980 := by
  sorry

end well_digging_payment_l3155_315504


namespace mary_fruit_change_l3155_315530

/-- The change Mary received after buying fruits -/
theorem mary_fruit_change (berries_cost peaches_cost payment : ℚ) 
  (h1 : berries_cost = 719 / 100)
  (h2 : peaches_cost = 683 / 100)
  (h3 : payment = 20) :
  payment - (berries_cost + peaches_cost) = 598 / 100 := by
  sorry

end mary_fruit_change_l3155_315530


namespace total_balls_in_box_l3155_315566

theorem total_balls_in_box (white_balls black_balls : ℕ) : 
  white_balls = 6 * black_balls →
  black_balls = 8 →
  white_balls + black_balls = 56 := by
sorry

end total_balls_in_box_l3155_315566
