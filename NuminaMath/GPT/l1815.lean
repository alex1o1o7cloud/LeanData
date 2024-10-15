import Mathlib

namespace NUMINAMATH_GPT_split_cube_l1815_181575

theorem split_cube (m : ℕ) (hm : m > 1) (h : ∃ k, ∃ l, l > 0 ∧ (3 + 2 * (k - 1)) = 59 ∧ (k + l = (m * (m - 1)) / 2)) : m = 8 :=
sorry

end NUMINAMATH_GPT_split_cube_l1815_181575


namespace NUMINAMATH_GPT_ab_greater_than_a_plus_b_l1815_181596

theorem ab_greater_than_a_plus_b (a b : ℝ) (h₁ : a ≥ 2) (h₂ : b > 2) : a * b > a + b :=
  sorry

end NUMINAMATH_GPT_ab_greater_than_a_plus_b_l1815_181596


namespace NUMINAMATH_GPT_sarah_math_homework_pages_l1815_181591

theorem sarah_math_homework_pages (x : ℕ) 
  (h1 : ∀ page, 4 * page = 4 * 6 + 4 * x)
  (h2 : 40 = 4 * 6 + 4 * x) : 
  x = 4 :=
by 
  sorry

end NUMINAMATH_GPT_sarah_math_homework_pages_l1815_181591


namespace NUMINAMATH_GPT_hours_practicing_l1815_181554

theorem hours_practicing (W : ℕ) (hours_weekday : ℕ) 
  (h1 : hours_weekday = W + 17)
  (h2 : W + hours_weekday = 33) :
  W = 8 :=
sorry

end NUMINAMATH_GPT_hours_practicing_l1815_181554


namespace NUMINAMATH_GPT_percentage_increase_l1815_181570

theorem percentage_increase (original new : ℝ) (h_original : original = 50) (h_new : new = 75) : 
  (new - original) / original * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1815_181570


namespace NUMINAMATH_GPT_sin2θ_value_l1815_181530

theorem sin2θ_value (θ : Real) (h1 : Real.sin θ = 4/5) (h2 : Real.sin θ - Real.cos θ > 1) : Real.sin (2*θ) = -24/25 := 
by 
  sorry

end NUMINAMATH_GPT_sin2θ_value_l1815_181530


namespace NUMINAMATH_GPT_parity_of_f_minimum_value_of_f_l1815_181552

noncomputable def f (x a : ℝ) : ℝ := x^2 + |x - a| - 1

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem parity_of_f (a : ℝ) :
  (a = 0 → is_even_function (f a)) ∧
  (a ≠ 0 → ¬is_even_function (f a) ∧ ¬is_odd_function (f a)) := 
by sorry

theorem minimum_value_of_f (a : ℝ) :
  (a ≤ -1/2 → ∀ x : ℝ, f x a ≥ -a - 5 / 4) ∧
  (-1/2 < a ∧ a ≤ 1/2 → ∀ x : ℝ, f x a ≥ a^2 - 1) ∧
  (a > 1/2 → ∀ x : ℝ, f x a ≥ a - 5 / 4) :=
by sorry

end NUMINAMATH_GPT_parity_of_f_minimum_value_of_f_l1815_181552


namespace NUMINAMATH_GPT_probability_sum_of_digits_eq_10_l1815_181517

theorem probability_sum_of_digits_eq_10 (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1): 
  let P := m / n
  let valid_numbers := 120
  let total_numbers := 2020
  (P = valid_numbers / total_numbers) → (m = 6) → (n = 101) → (m + n = 107) :=
by 
  sorry

end NUMINAMATH_GPT_probability_sum_of_digits_eq_10_l1815_181517


namespace NUMINAMATH_GPT_seating_arrangements_l1815_181578

-- Define the conditions and the proof problem
theorem seating_arrangements (children : Finset (Fin 6)) 
  (is_sibling_pair : (Fin 6) -> (Fin 6) -> Prop)
  (no_siblings_next_to_each_other : (Fin 6) -> (Fin 6) -> Bool)
  (no_sibling_directly_in_front : (Fin 6) -> (Fin 6) -> Bool) :
  -- Statement: There are 96 valid seating arrangements
  ∃ (arrangements : Finset (Fin 6 -> Fin (2 * 3))),
  arrangements.card = 96 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_seating_arrangements_l1815_181578


namespace NUMINAMATH_GPT_compute_expression_l1815_181510

theorem compute_expression :
  (-9 * 5 - (-7 * -2) + (-11 * -4)) = -15 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1815_181510


namespace NUMINAMATH_GPT_regular_price_of_tire_l1815_181547

theorem regular_price_of_tire (x : ℝ) (h : 3 * x + 10 = 250) : x = 80 :=
sorry

end NUMINAMATH_GPT_regular_price_of_tire_l1815_181547


namespace NUMINAMATH_GPT_calculate_expression_l1815_181537

theorem calculate_expression : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1815_181537


namespace NUMINAMATH_GPT_complex_solution_l1815_181558

theorem complex_solution (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (Complex.mk a b)^2 = Complex.mk 3 4) :
  Complex.mk a b = Complex.mk 2 1 :=
sorry

end NUMINAMATH_GPT_complex_solution_l1815_181558


namespace NUMINAMATH_GPT_solve_for_b_l1815_181534

def p (x : ℝ) : ℝ := 2 * x - 5
def q (x : ℝ) (b : ℝ) : ℝ := 3 * x - b

theorem solve_for_b (b : ℝ) : p (q 5 b) = 11 → b = 7 := by
  sorry

end NUMINAMATH_GPT_solve_for_b_l1815_181534


namespace NUMINAMATH_GPT_find_a_l1815_181573

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y + 1 = 0 → 
     ∀ (x' y' : ℝ), (x' = x - 2 * (x - a * y + 2) / (1 + a^2)) ∧ (y' = y - 2 * a * (x - a * y + 2) / (1 + a^2)) → 
     (x'^2 + y'^2 + 2 * x' - 4 * y' + 1 = 0)) → 
  (a = -1 / 2) := 
sorry

end NUMINAMATH_GPT_find_a_l1815_181573


namespace NUMINAMATH_GPT_equivalent_statement_l1815_181518

variable (R G : Prop)

theorem equivalent_statement (h : ¬ R → ¬ G) : G → R := by
  intro hG
  by_contra hR
  exact h hR hG

end NUMINAMATH_GPT_equivalent_statement_l1815_181518


namespace NUMINAMATH_GPT_bracelet_cost_l1815_181544

theorem bracelet_cost (B : ℝ)
  (H1 : 5 = 5)
  (H2 : 3 = 3)
  (H3 : 2 * B + 5 + B + 3 = 20) : B = 4 :=
by
  sorry

end NUMINAMATH_GPT_bracelet_cost_l1815_181544


namespace NUMINAMATH_GPT_nell_more_ace_cards_than_baseball_l1815_181525

-- Definitions based on conditions
def original_baseball_cards : ℕ := 239
def original_ace_cards : ℕ := 38
def current_ace_cards : ℕ := 376
def current_baseball_cards : ℕ := 111

-- The statement we need to prove
theorem nell_more_ace_cards_than_baseball :
  current_ace_cards - current_baseball_cards = 265 :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_nell_more_ace_cards_than_baseball_l1815_181525


namespace NUMINAMATH_GPT_intersection_A_B_intersection_CA_B_intersection_CA_CB_l1815_181594

-- Set definitions
def A := {x : ℝ | -5 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | x < -2 ∨ x > 4}
def C_A := {x : ℝ | x < -5 ∨ x > 3}  -- Complement of A
def C_B := {x : ℝ | -2 ≤ x ∧ x ≤ 4}  -- Complement of B

-- Lean statements proving the intersections
theorem intersection_A_B : {x : ℝ | -5 ≤ x ∧ x ≤ 3} ∩ {x : ℝ | x < -2 ∨ x > 4} = {x : ℝ | -5 ≤ x ∧ x < -2} :=
by sorry

theorem intersection_CA_B : {x : ℝ | x < -5 ∨ x > 3} ∩ {x : ℝ | x < -2 ∨ x > 4} = {x : ℝ | x < -5 ∨ x > 4} :=
by sorry

theorem intersection_CA_CB : {x : ℝ | x < -5 ∨ x > 3} ∩ {x : ℝ | -2 ≤ x ∧ x ≤ 4} = {x : ℝ | 3 < x ∧ x ≤ 4} :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_intersection_CA_B_intersection_CA_CB_l1815_181594


namespace NUMINAMATH_GPT_find_three_digit_number_l1815_181519

def is_valid_three_digit_number (M G U : ℕ) : Prop :=
  M ≠ G ∧ G ≠ U ∧ M ≠ U ∧ 
  0 ≤ M ∧ M ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ U ∧ U ≤ 9 ∧
  100 * M + 10 * G + U = (M + G + U) * (M + G + U - 2)

theorem find_three_digit_number : ∃ (M G U : ℕ), 
  is_valid_three_digit_number M G U ∧
  100 * M + 10 * G + U = 195 :=
by
  sorry

end NUMINAMATH_GPT_find_three_digit_number_l1815_181519


namespace NUMINAMATH_GPT_find_x_l1815_181582

theorem find_x (x : ℝ) (h1 : 3 * Real.sin (2 * x) = 2 * Real.sin x) (h2 : 0 < x ∧ x < Real.pi) :
  x = Real.arccos (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1815_181582


namespace NUMINAMATH_GPT_solve_system_of_equations_l1815_181565

theorem solve_system_of_equations:
  ∃ (x y z : ℝ), 
  x + y - z = 4 ∧
  x^2 + y^2 - z^2 = 12 ∧
  x^3 + y^3 - z^3 = 34 ∧
  ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 2 ∧ z = 1)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1815_181565


namespace NUMINAMATH_GPT_unpainted_cubes_count_l1815_181587

noncomputable def num_unpainted_cubes : ℕ :=
  let total_cubes := 216
  let painted_on_faces := 16 * 6 / 1  -- Central 4x4 areas on each face
  let shared_edges := ((4 * 4) * 6) / 2  -- Shared edges among faces
  let shared_corners := (4 * 6) / 3  -- Shared corners among faces
  let total_painted := painted_on_faces - shared_edges - shared_corners
  total_cubes - total_painted

theorem unpainted_cubes_count : num_unpainted_cubes = 160 := sorry

end NUMINAMATH_GPT_unpainted_cubes_count_l1815_181587


namespace NUMINAMATH_GPT_units_digit_of_expression_l1815_181583

theorem units_digit_of_expression :
  (8 * 18 * 1988 - 8^4) % 10 = 6 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_of_expression_l1815_181583


namespace NUMINAMATH_GPT_equilateral_triangle_not_centrally_symmetric_l1815_181520

-- Definitions for the shapes
def is_centrally_symmetric (shape : Type) : Prop := sorry
def Parallelogram : Type := sorry
def LineSegment : Type := sorry
def EquilateralTriangle : Type := sorry
def Rhombus : Type := sorry

-- Main theorem statement
theorem equilateral_triangle_not_centrally_symmetric :
  ¬ is_centrally_symmetric EquilateralTriangle ∧
  is_centrally_symmetric Parallelogram ∧
  is_centrally_symmetric LineSegment ∧
  is_centrally_symmetric Rhombus :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_not_centrally_symmetric_l1815_181520


namespace NUMINAMATH_GPT_A_is_5_years_older_than_B_l1815_181562

-- Given conditions
variables (A B : ℕ) -- A and B are the current ages
variables (x y : ℕ) -- x is the current age of A, y is the current age of B
variables 
  (A_was_B_age : A = y)
  (B_was_10_when_A_was_B_age : B = 10)
  (B_will_be_A_age : B = x)
  (A_will_be_25_when_B_will_be_A_age : A = 25)

-- Define the theorem to prove that A is 5 years older than B: A = B + 5
theorem A_is_5_years_older_than_B (x y : ℕ) (A B : ℕ) 
  (A_was_B_age : x = y) 
  (B_was_10_when_A_was_B_age : y = 10) 
  (B_will_be_A_age : y = x) 
  (A_will_be_25_when_B_will_be_A_age : x = 25): 
  x - y = 5 := 
by sorry

end NUMINAMATH_GPT_A_is_5_years_older_than_B_l1815_181562


namespace NUMINAMATH_GPT_tan_difference_l1815_181584

open Real

noncomputable def tan_difference_intermediate (θ : ℝ) : ℝ :=
  (tan θ - tan (π / 4)) / (1 + tan θ * tan (π / 4))

theorem tan_difference (θ : ℝ) (h1 : cos θ = -12 / 13) (h2 : π < θ ∧ θ < 3 * π / 2) :
  tan (θ - π / 4) = -7 / 17 :=
by
  sorry

end NUMINAMATH_GPT_tan_difference_l1815_181584


namespace NUMINAMATH_GPT_problem_solution_l1815_181566

noncomputable def area_triangle_ABC
  (R : ℝ) 
  (angle_BAC : ℝ) 
  (angle_DAC : ℝ) : ℝ :=
  let α := angle_DAC
  let β := angle_BAC
  2 * R^2 * (Real.sin α) * (Real.sin β) * (Real.sin (α + β))

theorem problem_solution :
  ∀ (R : ℝ) (angle_BAC : ℝ) (angle_DAC : ℝ),
  R = 3 →
  angle_BAC = (Real.pi / 4) →
  angle_DAC = (5 * Real.pi / 12) →
  area_triangle_ABC R angle_BAC angle_DAC = 10 :=
by intros R angle_BAC angle_DAC hR hBAC hDAC
   sorry

end NUMINAMATH_GPT_problem_solution_l1815_181566


namespace NUMINAMATH_GPT_line_bisects_circle_l1815_181521

theorem line_bisects_circle (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, l x y ↔ x - y = 0) → 
  (∀ x y : ℝ, C x y ↔ x^2 + y^2 = 1) → 
  ∀ x y : ℝ, (x - y = 0) ∨ (x + y = 0) → l x y ∧ C x y → l x y = (x - y = 0) := by
  sorry

end NUMINAMATH_GPT_line_bisects_circle_l1815_181521


namespace NUMINAMATH_GPT_gathering_handshakes_l1815_181543

theorem gathering_handshakes :
  let N := 12       -- twelve people, six couples
  let shakes_per_person := 9   -- each person shakes hands with 9 others
  let total_shakes := (N * shakes_per_person) / 2
  total_shakes = 54 := 
by
  sorry

end NUMINAMATH_GPT_gathering_handshakes_l1815_181543


namespace NUMINAMATH_GPT_binary_multiplication_l1815_181574

theorem binary_multiplication :
  0b1101 * 0b110 = 0b1011110 := 
sorry

end NUMINAMATH_GPT_binary_multiplication_l1815_181574


namespace NUMINAMATH_GPT_remainder_division_l1815_181590

theorem remainder_division (x r : ℕ) (h₁ : 1650 - x = 1390) (h₂ : 1650 = 6 * x + r) : r = 90 := by
  sorry

end NUMINAMATH_GPT_remainder_division_l1815_181590


namespace NUMINAMATH_GPT_derrick_has_34_pictures_l1815_181516

-- Assume Ralph has 26 pictures of wild animals
def ralph_pictures : ℕ := 26

-- Derrick has 8 more pictures than Ralph
def derrick_pictures : ℕ := ralph_pictures + 8

-- Prove that Derrick has 34 pictures of wild animals
theorem derrick_has_34_pictures : derrick_pictures = 34 := by
  sorry

end NUMINAMATH_GPT_derrick_has_34_pictures_l1815_181516


namespace NUMINAMATH_GPT_find_a7_over_b7_l1815_181529

-- Definitions of the sequences and the arithmetic properties
variable {a b: ℕ → ℕ}  -- sequences a_n and b_n
variable {S T: ℕ → ℕ}  -- sums of the first n terms

-- Problem conditions
def is_arithmetic_sequence (seq: ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) - seq n = d

def sum_of_first_n_terms (seq: ℕ → ℕ) (sum_fn: ℕ → ℕ) : Prop :=
  ∀ n, sum_fn n = n * (seq 1 + seq n) / 2

-- Given conditions
axiom h1: is_arithmetic_sequence a
axiom h2: is_arithmetic_sequence b
axiom h3: sum_of_first_n_terms a S
axiom h4: sum_of_first_n_terms b T
axiom h5: ∀ n, S n / T n = (3 * n + 2) / (2 * n)

-- Main theorem to prove
theorem find_a7_over_b7 : (a 7) / (b 7) = (41 / 26) :=
sorry

end NUMINAMATH_GPT_find_a7_over_b7_l1815_181529


namespace NUMINAMATH_GPT_line_perp_to_plane_contains_line_implies_perp_l1815_181593

variables {Point Line Plane : Type}
variables (m n : Line) (α : Plane)
variables (contains : Plane → Line → Prop) (perp : Line → Line → Prop) (perp_plane : Line → Plane → Prop)

-- Given: 
-- m and n are two different lines
-- α is a plane
-- m ⊥ α (m is perpendicular to the plane α)
-- n ⊂ α (n is contained in the plane α)
-- Prove: m ⊥ n
theorem line_perp_to_plane_contains_line_implies_perp (hm : perp_plane m α) (hn : contains α n) : perp m n :=
sorry

end NUMINAMATH_GPT_line_perp_to_plane_contains_line_implies_perp_l1815_181593


namespace NUMINAMATH_GPT_quadratic_roots_identity_l1815_181588

theorem quadratic_roots_identity :
  ∀ (x1 x2 : ℝ), (x1^2 - 3 * x1 - 4 = 0) ∧ (x2^2 - 3 * x2 - 4 = 0) →
  (x1^2 - 2 * x1 * x2 + x2^2) = 25 :=
by
  intros x1 x2 h
  sorry

end NUMINAMATH_GPT_quadratic_roots_identity_l1815_181588


namespace NUMINAMATH_GPT_how_many_years_younger_l1815_181553

-- Define conditions
def age_ratio (sandy_age moll_age : ℕ) := sandy_age * 9 = moll_age * 7
def sandy_age := 70

-- Define the theorem to prove
theorem how_many_years_younger 
  (molly_age : ℕ) 
  (h1 : age_ratio sandy_age molly_age) 
  (h2 : sandy_age = 70) : molly_age - sandy_age = 20 := 
sorry

end NUMINAMATH_GPT_how_many_years_younger_l1815_181553


namespace NUMINAMATH_GPT_solve_quadratic_eq_solve_equal_squares_l1815_181542

theorem solve_quadratic_eq (x : ℝ) : 
    (4 * x^2 - 2 * x - 1 = 0) ↔ 
    (x = (1 + Real.sqrt 5) / 4 ∨ x = (1 - Real.sqrt 5) / 4) := 
by
  sorry

theorem solve_equal_squares (y : ℝ) :
    ((y + 1)^2 = (3 * y - 1)^2) ↔ 
    (y = 1 ∨ y = 0) := 
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_solve_equal_squares_l1815_181542


namespace NUMINAMATH_GPT_complex_number_multiplication_l1815_181536

theorem complex_number_multiplication (i : ℂ) (hi : i * i = -1) : i * (1 + i) = -1 + i :=
by sorry

end NUMINAMATH_GPT_complex_number_multiplication_l1815_181536


namespace NUMINAMATH_GPT_exist_odd_distinct_integers_l1815_181595

theorem exist_odd_distinct_integers (n : ℕ) (h1 : n % 2 = 1) (h2 : n > 3) (h3 : n % 3 ≠ 0) : 
  ∃ a b c : ℕ, a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  3 / (n : ℚ) = 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) :=
sorry

end NUMINAMATH_GPT_exist_odd_distinct_integers_l1815_181595


namespace NUMINAMATH_GPT_small_fries_number_l1815_181511

variables (L S : ℕ)

axiom h1 : L + S = 24
axiom h2 : L = 5 * S

theorem small_fries_number : S = 4 :=
by sorry

end NUMINAMATH_GPT_small_fries_number_l1815_181511


namespace NUMINAMATH_GPT_min_weighings_to_determine_counterfeit_l1815_181513

/-- 
  Given 2023 coins with two counterfeit coins and 2021 genuine coins, 
  and using a balance scale, determine whether the counterfeit coins 
  are heavier or lighter. Prove that the minimum number of weighings 
  required is 3. 
-/
theorem min_weighings_to_determine_counterfeit (n : ℕ) (k : ℕ) (l : ℕ) 
  (h : n = 2023) (h₁ : k = 2) (h₂ : l = 2021) 
  (w₁ w₂ : ℕ → ℝ) -- weights of coins
  (h_fake : ∀ i j, w₁ i = w₁ j) -- counterfeits have same weight
  (h_fake_diff : ∀ i j, i ≠ j → w₁ i ≠ w₂ j) -- fake different from genuine
  (h_genuine : ∀ i j, w₂ i = w₂ j) -- genuines have same weight
  (h_total : ∀ i, i ≤ l + k) -- total coins condition
  : ∃ min_weighings : ℕ, min_weighings = 3 :=
by
  sorry

end NUMINAMATH_GPT_min_weighings_to_determine_counterfeit_l1815_181513


namespace NUMINAMATH_GPT_sum_of_fractions_l1815_181506

theorem sum_of_fractions:
  (7 / 12) + (11 / 15) = 79 / 60 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l1815_181506


namespace NUMINAMATH_GPT_square_side_length_l1815_181555

theorem square_side_length (p : ℝ) (h : p = 17.8) : (p / 4) = 4.45 := by
  sorry

end NUMINAMATH_GPT_square_side_length_l1815_181555


namespace NUMINAMATH_GPT_Mary_avg_speed_l1815_181507

def Mary_uphill_distance := 1.5 -- km
def Mary_uphill_time := 45.0 / 60.0 -- hours
def Mary_downhill_distance := 1.5 -- km
def Mary_downhill_time := 15.0 / 60.0 -- hours

def total_distance := Mary_uphill_distance + Mary_downhill_distance
def total_time := Mary_uphill_time + Mary_downhill_time

theorem Mary_avg_speed : 
  (total_distance / total_time) = 3.0 := by
  sorry

end NUMINAMATH_GPT_Mary_avg_speed_l1815_181507


namespace NUMINAMATH_GPT_max_sum_at_n_is_6_l1815_181561

-- Assuming an arithmetic sequence a_n where a_1 = 4 and d = -5/7
def arithmetic_seq (n : ℕ) : ℚ := (33 / 7) - (5 / 7) * n

-- Sum of the first n terms (S_n) of the arithmetic sequence {a_n}
def sum_arithmetic_seq (n : ℕ) : ℚ := (n / 2) * (2 * (arithmetic_seq 1) + (n - 1) * (-5 / 7))

theorem max_sum_at_n_is_6 
  (a_1 : ℚ) (d : ℚ) (h1 : a_1 = 4) (h2 : d = -5/7) :
  ∀ n : ℕ, sum_arithmetic_seq n ≤ sum_arithmetic_seq 6 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_at_n_is_6_l1815_181561


namespace NUMINAMATH_GPT_factorize_expression_l1815_181528

theorem factorize_expression (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4 * x * y = (x * y - 1 + x + y) * (x * y - 1 - x - y) :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l1815_181528


namespace NUMINAMATH_GPT_concentration_after_5500_evaporates_l1815_181527

noncomputable def concentration_after_evaporation 
  (V₀ Vₑ : ℝ) (C₀ : ℝ) : ℝ := 
  let sodium_chloride := C₀ * V₀
  let remaining_volume := V₀ - Vₑ
  100 * sodium_chloride / remaining_volume

theorem concentration_after_5500_evaporates 
  : concentration_after_evaporation 10000 5500 0.05 = 11.11 := 
by
  -- Formalize the calculations as we have derived
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_concentration_after_5500_evaporates_l1815_181527


namespace NUMINAMATH_GPT_angle_ABC_40_degrees_l1815_181505

theorem angle_ABC_40_degrees (ABC ABD CBD : ℝ) 
    (h1 : CBD = 90) 
    (h2 : ABD = 60)
    (h3 : ABC + ABD + CBD = 190) : 
    ABC = 40 := 
by {
  sorry
}

end NUMINAMATH_GPT_angle_ABC_40_degrees_l1815_181505


namespace NUMINAMATH_GPT_b_days_solve_l1815_181523

-- Definitions from the conditions
variable (b_days : ℝ)
variable (a_rate : ℝ) -- work rate of a
variable (b_rate : ℝ) -- work rate of b

-- Condition 1: a is twice as fast as b
def twice_as_fast_as_b : Prop :=
  a_rate = 2 * b_rate

-- Condition 2: a and b together can complete the work in 3.333333333333333 days
def combined_completion_time : Prop :=
  1 / (a_rate + b_rate) = 10 / 3

-- The number of days b alone can complete the work should satisfy this equation
def b_alone_can_complete_in_b_days : Prop :=
  b_rate = 1 / b_days

-- The actual theorem we want to prove:
theorem b_days_solve (b_rate a_rate : ℝ) (h1 : twice_as_fast_as_b a_rate b_rate) (h2 : combined_completion_time a_rate b_rate) : b_days = 10 :=
by
  sorry

end NUMINAMATH_GPT_b_days_solve_l1815_181523


namespace NUMINAMATH_GPT_temperature_rise_l1815_181592

variable (t : ℝ)

theorem temperature_rise (initial final : ℝ) (h : final = t) : final = 5 + t := by
  sorry

end NUMINAMATH_GPT_temperature_rise_l1815_181592


namespace NUMINAMATH_GPT_compute_fraction_sum_l1815_181581

-- Define the equation whose roots are a, b, c
def cubic_eq (x : ℝ) : Prop := x^3 - 6*x^2 + 11*x = 12

-- State the main theorem
theorem compute_fraction_sum 
  (a b c : ℝ) 
  (ha : cubic_eq a) 
  (hb : cubic_eq b) 
  (hc : cubic_eq c) :
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → 
  ∃ (r : ℝ), r = -23/12 ∧ (ab/c + bc/a + ca/b) = r := 
  sorry

end NUMINAMATH_GPT_compute_fraction_sum_l1815_181581


namespace NUMINAMATH_GPT_bob_distance_walked_l1815_181586

theorem bob_distance_walked
    (dist : ℕ)
    (yolanda_rate : ℕ)
    (bob_rate : ℕ)
    (hour_diff : ℕ)
    (meet_time_bob: ℕ) :

    dist = 31 → yolanda_rate = 1 → bob_rate = 2 → hour_diff = 1 → meet_time_bob = 10 →
    (bob_rate * meet_time_bob) = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bob_distance_walked_l1815_181586


namespace NUMINAMATH_GPT_increasing_on_1_to_infinity_max_and_min_on_1_to_4_l1815_181514

noncomputable def f (x : ℝ) : ℝ := x + (1 / x)

theorem increasing_on_1_to_infinity : ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → (1 ≤ x2) → f x1 < f x2 := by
  sorry

theorem max_and_min_on_1_to_4 : 
  (∀ (x : ℝ), 1 ≤ x → x ≤ 4 → f x ≤ f 4) ∧ 
  (∀ (x : ℝ), 1 ≤ x → x ≤ 4 → f 1 ≤ f x) := by
  sorry

end NUMINAMATH_GPT_increasing_on_1_to_infinity_max_and_min_on_1_to_4_l1815_181514


namespace NUMINAMATH_GPT_choir_min_students_l1815_181515

theorem choir_min_students : ∃ n : ℕ, (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) ∧ n = 990 :=
by
  sorry

end NUMINAMATH_GPT_choir_min_students_l1815_181515


namespace NUMINAMATH_GPT_compute_c_plus_d_l1815_181526

theorem compute_c_plus_d (c d : ℕ) (h1 : d = c^3) (h2 : d - c = 435) : c + d = 520 :=
sorry

end NUMINAMATH_GPT_compute_c_plus_d_l1815_181526


namespace NUMINAMATH_GPT_x_intercept_rotation_30_degrees_eq_l1815_181535

noncomputable def x_intercept_new_line (x0 y0 : ℝ) (θ : ℝ) (a b c : ℝ) : ℝ :=
  let m := a / b
  let m' := (m + θ.tan) / (1 - m * θ.tan)
  let x_intercept := x0 - (y0 * (b - m * c)) / (m' * (b - m * c) - a)
  x_intercept

theorem x_intercept_rotation_30_degrees_eq :
  x_intercept_new_line 7 4 (Real.pi / 6) 4 (-7) 28 = 7 - (4 * (7 * Real.sqrt 3 - 4) / (4 * Real.sqrt 3 + 7)) :=
by 
  -- detailed math proof goes here 
  sorry

end NUMINAMATH_GPT_x_intercept_rotation_30_degrees_eq_l1815_181535


namespace NUMINAMATH_GPT_ratio_second_to_first_l1815_181557

-- Condition 1: The first bell takes 50 pounds of bronze
def first_bell_weight : ℕ := 50

-- Condition 2: The second bell is a certain size compared to the first bell
variable (x : ℕ) -- the ratio of the size of the second bell to the first bell
def second_bell_weight := first_bell_weight * x

-- Condition 3: The third bell is four times the size of the second bell
def third_bell_weight := 4 * second_bell_weight x

-- Condition 4: The total weight of bronze required is 550 pounds
def total_weight : ℕ := 550

-- Define the proof problem
theorem ratio_second_to_first (x : ℕ) (h : 50 + 50 * x + 200 * x = 550) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_second_to_first_l1815_181557


namespace NUMINAMATH_GPT_smaller_fraction_is_l1815_181549

theorem smaller_fraction_is
  (x y : ℝ)
  (h₁ : x + y = 7 / 8)
  (h₂ : x * y = 1 / 12) :
  min x y = (7 - Real.sqrt 17) / 16 :=
sorry

end NUMINAMATH_GPT_smaller_fraction_is_l1815_181549


namespace NUMINAMATH_GPT_smallest_perimeter_consecutive_integers_triangle_l1815_181508

theorem smallest_perimeter_consecutive_integers_triangle :
  ∃ (a b c : ℕ), 
    1 < a ∧ a + 1 = b ∧ b + 1 = c ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    a + b + c = 12 :=
by
  -- proof placeholder
  sorry

end NUMINAMATH_GPT_smallest_perimeter_consecutive_integers_triangle_l1815_181508


namespace NUMINAMATH_GPT_at_least_one_not_less_than_one_l1815_181512

open Real

theorem at_least_one_not_less_than_one (x : ℝ) :
  let a := x^2 + 1/2
  let b := 2 - x
  let c := x^2 - x + 1
  a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1 :=
by
  -- Definitions of a, b, and c
  let a := x^2 + 1/2
  let b := 2 - x
  let c := x^2 - x + 1
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_one_l1815_181512


namespace NUMINAMATH_GPT_divisible_by_24_l1815_181524

theorem divisible_by_24 (n : ℕ) (hn : n > 0) : 24 ∣ n * (n + 2) * (5 * n - 1) * (5 * n + 1) := 
by sorry

end NUMINAMATH_GPT_divisible_by_24_l1815_181524


namespace NUMINAMATH_GPT_cake_and_tea_cost_l1815_181545

theorem cake_and_tea_cost (cost_of_milk_tea : ℝ) (cost_of_cake : ℝ)
    (h1 : cost_of_cake = (3 / 4) * cost_of_milk_tea)
    (h2 : cost_of_milk_tea = 2.40) :
    2 * cost_of_cake + cost_of_milk_tea = 6.00 := 
sorry

end NUMINAMATH_GPT_cake_and_tea_cost_l1815_181545


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1815_181560

open Set

def univeral_set : Set ℕ := { x | x + 1 ≤ 0 ∨ 0 ≤ x - 5 }

def A : Set ℕ := {1, 2, 4}

noncomputable def complement_U_A : Set ℕ := {0, 3}

theorem complement_of_A_in_U : (compl A ∩ univeral_set) = complement_U_A := 
by 
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1815_181560


namespace NUMINAMATH_GPT_batsman_average_l1815_181539

theorem batsman_average
  (avg_20_matches : ℕ → ℕ → ℕ)
  (avg_10_matches : ℕ → ℕ → ℕ)
  (total_1st_20 : ℕ := avg_20_matches 20 30)
  (total_next_10 : ℕ := avg_10_matches 10 15) :
  (total_1st_20 + total_next_10) / 30 = 25 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_l1815_181539


namespace NUMINAMATH_GPT_floor_length_l1815_181589

/-- Given the rectangular tiles of size 50 cm by 40 cm, which are laid on a rectangular floor
without overlap and with a maximum of 9 tiles. Prove the floor length is 450 cm. -/
theorem floor_length (tiles_max : ℕ) (tile_length tile_width floor_length floor_width : ℕ)
  (Htile_length : tile_length = 50) (Htile_width : tile_width = 40)
  (Htiles_max : tiles_max = 9)
  (Hconditions : (∀ m n : ℕ, (m * n = tiles_max) → 
                  (floor_length = m * tile_length ∨ floor_length = m * tile_width)))
  : floor_length = 450 :=
by 
  sorry

end NUMINAMATH_GPT_floor_length_l1815_181589


namespace NUMINAMATH_GPT_books_loaned_out_l1815_181538

theorem books_loaned_out (initial_books : ℕ) (returned_percentage : ℝ) (end_books : ℕ) (x : ℝ) :
    initial_books = 75 →
    returned_percentage = 0.70 →
    end_books = 63 →
    0.30 * x = (initial_books - end_books) →
    x = 40 := by
  sorry

end NUMINAMATH_GPT_books_loaned_out_l1815_181538


namespace NUMINAMATH_GPT_production_line_B_units_l1815_181551

theorem production_line_B_units
  (total_units : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
  (h_total_units : total_units = 5000)
  (h_ratio : ratio_A = 1 ∧ ratio_B = 2 ∧ ratio_C = 2) :
  (2 * (total_units / (ratio_A + ratio_B + ratio_C))) = 2000 :=
by
  sorry

end NUMINAMATH_GPT_production_line_B_units_l1815_181551


namespace NUMINAMATH_GPT_machines_in_first_scenario_l1815_181559

theorem machines_in_first_scenario (x : ℕ) (hx : x ≠ 0) : 
  ∃ n : ℕ, (∀ m : ℕ, (∀ r1 r2 : ℚ, r1 = (x:ℚ) / (6 * n) → r2 = (3 * x:ℚ) / (6 * 12) → r1 = r2 → m = 12 → 3 * n = 12) → n = 4) :=
by
  sorry

end NUMINAMATH_GPT_machines_in_first_scenario_l1815_181559


namespace NUMINAMATH_GPT_larry_substituted_value_l1815_181585

theorem larry_substituted_value :
  ∀ (a b c d e : ℤ), a = 5 → b = 3 → c = 4 → d = 2 → e = 2 → 
  (a + b - c + d - e = a + (b - (c + (d - e)))) :=
by
  intros a b c d e ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end NUMINAMATH_GPT_larry_substituted_value_l1815_181585


namespace NUMINAMATH_GPT_algebra_expression_l1815_181598

theorem algebra_expression (a b : ℝ) (h : a = b + 1) : 3 + 2 * a - 2 * b = 5 :=
sorry

end NUMINAMATH_GPT_algebra_expression_l1815_181598


namespace NUMINAMATH_GPT_right_triangle_ratio_l1815_181568

theorem right_triangle_ratio (x : ℝ) :
  let AB := 3 * x
  let BC := 4 * x
  let AC := (AB ^ 2 + BC ^ 2).sqrt
  let h := AC
  let AD := 16 / 21 * h / (16 / 21 + 1)
  let CD := h / (16 / 21 + 1)
  (CD / AD) = 21 / 16 :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_ratio_l1815_181568


namespace NUMINAMATH_GPT_geometric_loci_l1815_181541

noncomputable def quadratic_discriminant (x y : ℝ) : ℝ :=
  x^2 + 4 * y^2 - 4

-- Conditions:
def real_and_distinct (x y : ℝ) := 
  ((x^2) / 4 + y^2 > 1) 

def equal_and_real (x y : ℝ) := 
  ((x^2) / 4 + y^2 = 1) 

def complex_roots (x y : ℝ) := 
  ((x^2) / 4 + y^2 < 1)

def both_roots_positive (x y : ℝ) := 
  (x < 0) ∧ (-1 < y) ∧ (y < 1)

def both_roots_negative (x y : ℝ) := 
  (x > 0) ∧ (-1 < y) ∧ (y < 1)

def opposite_sign_roots (x y : ℝ) := 
  (y > 1) ∨ (y < -1)

theorem geometric_loci (x y : ℝ) :
  (real_and_distinct x y ∨ equal_and_real x y ∨ complex_roots x y) ∧ 
  ((real_and_distinct x y ∧ both_roots_positive x y) ∨
   (real_and_distinct x y ∧ both_roots_negative x y) ∨
   (real_and_distinct x y ∧ opposite_sign_roots x y)) := 
sorry

end NUMINAMATH_GPT_geometric_loci_l1815_181541


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1815_181532

theorem simplify_and_evaluate_expression (a : ℂ) (h: a^2 + 4 * a + 1 = 0) :
  ( ( (a + 2) / (a^2 - 2 * a) + 8 / (4 - a^2) ) / ( (a^2 - 4) / a ) ) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1815_181532


namespace NUMINAMATH_GPT_least_positive_integer_l1815_181540

theorem least_positive_integer :
  ∃ (a : ℕ), (a ≡ 1 [MOD 3]) ∧ (a ≡ 2 [MOD 4]) ∧ (∀ b, (b ≡ 1 [MOD 3]) → (b ≡ 2 [MOD 4]) → b ≥ a → b = a) :=
sorry

end NUMINAMATH_GPT_least_positive_integer_l1815_181540


namespace NUMINAMATH_GPT_sales_tax_paid_l1815_181503

variable (total_cost : ℝ)
variable (tax_rate : ℝ)
variable (tax_free_cost : ℝ)

theorem sales_tax_paid (h_total : total_cost = 25) (h_rate : tax_rate = 0.10) (h_free : tax_free_cost = 21.7) :
  ∃ (X : ℝ), 21.7 + X + (0.10 * X) = 25 ∧ (0.10 * X = 0.3) := 
by
  sorry

end NUMINAMATH_GPT_sales_tax_paid_l1815_181503


namespace NUMINAMATH_GPT_abs_neg_two_l1815_181556

theorem abs_neg_two : abs (-2) = 2 := by
  sorry

end NUMINAMATH_GPT_abs_neg_two_l1815_181556


namespace NUMINAMATH_GPT_complement_of_union_l1815_181576

-- Define the universal set U, set M, and set N as given:
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

-- Define the complement of a set relative to the universal set U
def complement_U (A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

-- Prove that the complement of M ∪ N with respect to U is {1, 6}
theorem complement_of_union : complement_U (M ∪ N) = {1, 6} :=
  sorry -- proof goes here

end NUMINAMATH_GPT_complement_of_union_l1815_181576


namespace NUMINAMATH_GPT_leak_empties_tank_in_30_hours_l1815_181500

-- Define the known rates based on the problem conditions
def rate_pipe_a : ℚ := 1 / 12
def combined_rate : ℚ := 1 / 20

-- Define the rate at which the leak empties the tank
def rate_leak : ℚ := rate_pipe_a - combined_rate

-- Define the time it takes for the leak to empty the tank
def time_to_empty_tank : ℚ := 1 / rate_leak

-- The theorem that needs to be proved
theorem leak_empties_tank_in_30_hours : time_to_empty_tank = 30 :=
sorry

end NUMINAMATH_GPT_leak_empties_tank_in_30_hours_l1815_181500


namespace NUMINAMATH_GPT_sum_of_three_digits_eq_nine_l1815_181567

def horizontal_segments (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 0
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 2
  | 6 => 1
  | 7 => 1
  | 8 => 3
  | 9 => 2
  | _ => 0  -- Invalid digit

def vertical_segments (n : ℕ) : ℕ :=
  match n with
  | 0 => 4
  | 1 => 2
  | 2 => 3
  | 3 => 3
  | 4 => 3
  | 5 => 2
  | 6 => 3
  | 7 => 2
  | 8 => 4
  | 9 => 3
  | _ => 0  -- Invalid digit

theorem sum_of_three_digits_eq_nine :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
             (horizontal_segments a + horizontal_segments b + horizontal_segments c = 5) ∧ 
             (vertical_segments a + vertical_segments b + vertical_segments c = 10) ∧
             (a + b + c = 9) :=
sorry

end NUMINAMATH_GPT_sum_of_three_digits_eq_nine_l1815_181567


namespace NUMINAMATH_GPT_combined_average_age_l1815_181531

theorem combined_average_age :
  (8 * 35 + 6 * 30) / (8 + 6) = 33 :=
by
  sorry

end NUMINAMATH_GPT_combined_average_age_l1815_181531


namespace NUMINAMATH_GPT_meetings_percentage_l1815_181597

theorem meetings_percentage
  (workday_hours : ℕ)
  (first_meeting_minutes : ℕ)
  (second_meeting_factor : ℕ)
  (third_meeting_factor : ℕ)
  (total_minutes : ℕ)
  (total_meeting_minutes : ℕ) :
  workday_hours = 9 →
  first_meeting_minutes = 30 →
  second_meeting_factor = 2 →
  third_meeting_factor = 3 →
  total_minutes = workday_hours * 60 →
  total_meeting_minutes = first_meeting_minutes + second_meeting_factor * first_meeting_minutes + third_meeting_factor * first_meeting_minutes →
  (total_meeting_minutes : ℚ) / (total_minutes : ℚ) * 100 = 33.33 :=
by
  sorry

end NUMINAMATH_GPT_meetings_percentage_l1815_181597


namespace NUMINAMATH_GPT_min_tiles_needed_l1815_181522

-- Definitions for the problem
def tile_width : ℕ := 3
def tile_height : ℕ := 4

def region_width_ft : ℕ := 2
def region_height_ft : ℕ := 5

def inches_in_foot : ℕ := 12

-- Conversion
def region_width_in := region_width_ft * inches_in_foot
def region_height_in := region_height_ft * inches_in_foot

-- Calculations
def region_area := region_width_in * region_height_in
def tile_area := tile_width * tile_height

-- Theorem statement
theorem min_tiles_needed : region_area / tile_area = 120 := 
  sorry

end NUMINAMATH_GPT_min_tiles_needed_l1815_181522


namespace NUMINAMATH_GPT_calculation_results_in_a_pow_5_l1815_181548

variable (a : ℕ)

theorem calculation_results_in_a_pow_5 : a^3 * a^2 = a^5 := 
  by sorry

end NUMINAMATH_GPT_calculation_results_in_a_pow_5_l1815_181548


namespace NUMINAMATH_GPT_real_roots_iff_le_one_l1815_181546

theorem real_roots_iff_le_one (k : ℝ) : (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) → k ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_real_roots_iff_le_one_l1815_181546


namespace NUMINAMATH_GPT_junior_score_calculation_l1815_181501

variable {total_students : ℕ}
variable {junior_score senior_average : ℕ}
variable {junior_ratio senior_ratio : ℚ}
variable {class_average total_average : ℚ}

-- Hypotheses from the conditions
theorem junior_score_calculation (h1 : junior_ratio = 0.2)
                               (h2 : senior_ratio = 0.8)
                               (h3 : class_average = 82)
                               (h4 : senior_average = 80)
                               (h5 : total_students = 10)
                               (h6 : total_average * total_students = total_students * class_average)
                               (h7 : total_average = (junior_ratio * junior_score + senior_ratio * senior_average))
                               : junior_score = 90 :=
sorry

end NUMINAMATH_GPT_junior_score_calculation_l1815_181501


namespace NUMINAMATH_GPT_math_problem_proof_l1815_181550

theorem math_problem_proof :
    24 * (243 / 3 + 49 / 7 + 16 / 8 + 4 / 2 + 2) = 2256 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_math_problem_proof_l1815_181550


namespace NUMINAMATH_GPT_value_of_larger_denom_eq_10_l1815_181571

/-- Anna has 12 bills in her wallet, and the total value is $100. 
    She has 4 $5 bills and 8 bills of a larger denomination.
    Prove that the value of the larger denomination bill is $10. -/
theorem value_of_larger_denom_eq_10 (n : ℕ) (b : ℤ) (total_value : ℤ) (five_bills : ℕ) (larger_bills : ℕ):
    (total_value = 100) ∧ 
    (five_bills = 4) ∧ 
    (larger_bills = 8) ∧ 
    (n = five_bills + larger_bills) ∧ 
    (n = 12) → 
    (b = 10) :=
by
  sorry

end NUMINAMATH_GPT_value_of_larger_denom_eq_10_l1815_181571


namespace NUMINAMATH_GPT_solution_set_line_l1815_181509

theorem solution_set_line (x y : ℝ) : x - 2 * y = 1 → y = (x - 1) / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solution_set_line_l1815_181509


namespace NUMINAMATH_GPT_least_number_of_table_entries_l1815_181577

-- Given conditions
def num_towns : ℕ := 6

-- Theorem statement
theorem least_number_of_table_entries : (num_towns * (num_towns - 1)) / 2 = 15 := by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_least_number_of_table_entries_l1815_181577


namespace NUMINAMATH_GPT_possible_integer_roots_l1815_181569

theorem possible_integer_roots (x : ℤ) :
  x^3 + 3 * x^2 - 4 * x - 13 = 0 →
  x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13 :=
by sorry

end NUMINAMATH_GPT_possible_integer_roots_l1815_181569


namespace NUMINAMATH_GPT_total_donation_correct_l1815_181599

-- Define the donations to each orphanage
def first_orphanage_donation : ℝ := 175.00
def second_orphanage_donation : ℝ := 225.00
def third_orphanage_donation : ℝ := 250.00

-- State the total donation
def total_donation : ℝ := 650.00

-- The theorem statement to be proved
theorem total_donation_correct :
  first_orphanage_donation + second_orphanage_donation + third_orphanage_donation = total_donation :=
by
  sorry

end NUMINAMATH_GPT_total_donation_correct_l1815_181599


namespace NUMINAMATH_GPT_time_to_fill_pond_l1815_181564

-- Conditions:
def pond_capacity : ℕ := 200
def normal_pump_rate : ℕ := 6
def drought_factor : ℚ := 2 / 3

-- The current pumping rate:
def current_pump_rate : ℚ := normal_pump_rate * drought_factor

-- We need to prove the time it takes to fill the pond is 50 minutes:
theorem time_to_fill_pond : 
  (pond_capacity : ℚ) / current_pump_rate = 50 := 
sorry

end NUMINAMATH_GPT_time_to_fill_pond_l1815_181564


namespace NUMINAMATH_GPT_least_score_to_play_final_l1815_181502

-- Definitions based on given conditions
def num_teams := 2021

def match_points (outcome : String) : ℕ :=
  match outcome with
  | "win"  => 3
  | "draw" => 1
  | "loss" => 0
  | _      => 0

def brazil_won_first_match : Prop := True

def ties_advantage (bfc_score other_team_score : ℕ) : Prop :=
  bfc_score = other_team_score

-- Theorem statement
theorem least_score_to_play_final (bfc_has_tiebreaker : (bfc_score other_team_score : ℕ) → ties_advantage bfc_score other_team_score)
  (bfc_first_match_won : brazil_won_first_match) :
  ∃ (least_score : ℕ), least_score = 2020 := sorry

end NUMINAMATH_GPT_least_score_to_play_final_l1815_181502


namespace NUMINAMATH_GPT_triangle_area_ordering_l1815_181533

variable (m n p : ℚ)

theorem triangle_area_ordering (hm : m = 15 / 2) (hn : n = 13 / 2) (hp : p = 7) : n < p ∧ p < m := by
  sorry

end NUMINAMATH_GPT_triangle_area_ordering_l1815_181533


namespace NUMINAMATH_GPT_multiplication_of_935421_and_625_l1815_181563

theorem multiplication_of_935421_and_625 :
  935421 * 625 = 584638125 :=
by sorry

end NUMINAMATH_GPT_multiplication_of_935421_and_625_l1815_181563


namespace NUMINAMATH_GPT_age_ratio_l1815_181579

theorem age_ratio (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 10) (h3 : a + b + c = 27) : b / c = 2 := by
  sorry

end NUMINAMATH_GPT_age_ratio_l1815_181579


namespace NUMINAMATH_GPT_bleaching_process_percentage_decrease_l1815_181572

noncomputable def total_percentage_decrease (L B : ℝ) : ℝ :=
  let area1 := (0.80 * L) * (0.90 * B)
  let area2 := (0.85 * (0.80 * L)) * (0.95 * (0.90 * B))
  let area3 := (0.90 * (0.85 * (0.80 * L))) * (0.92 * (0.95 * (0.90 * B)))
  ((L * B - area3) / (L * B)) * 100

theorem bleaching_process_percentage_decrease (L B : ℝ) :
  total_percentage_decrease L B = 44.92 :=
by
  sorry

end NUMINAMATH_GPT_bleaching_process_percentage_decrease_l1815_181572


namespace NUMINAMATH_GPT_number_of_true_propositions_l1815_181580

open Classical

axiom real_numbers (a b : ℝ): Prop

noncomputable def original_proposition (a b : ℝ) : Prop := a > b → a * abs a > b * abs b
noncomputable def converse_proposition (a b : ℝ) : Prop := a * abs a > b * abs b → a > b
noncomputable def negation_proposition (a b : ℝ) : Prop := a ≤ b → a * abs a ≤ b * abs b
noncomputable def contrapositive_proposition (a b : ℝ) : Prop := a * abs a ≤ b * abs b → a ≤ b

theorem number_of_true_propositions (a b : ℝ) (h₁: original_proposition a b) 
  (h₂: converse_proposition a b) (h₃: negation_proposition a b)
  (h₄: contrapositive_proposition a b) : ∃ n, n = 4 := 
by
  -- The proof would go here, proving that ∃ n, n = 4 is true.
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_l1815_181580


namespace NUMINAMATH_GPT_jack_change_l1815_181504

def cost_per_sandwich : ℕ := 5
def number_of_sandwiches : ℕ := 3
def payment : ℕ := 20

theorem jack_change : payment - (cost_per_sandwich * number_of_sandwiches) = 5 := 
by
  sorry

end NUMINAMATH_GPT_jack_change_l1815_181504
