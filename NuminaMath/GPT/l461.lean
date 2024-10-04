import Mathlib

namespace headlight_counts_l461_461401

-- Definitions related to the problem conditions
def total_cars := 800
def hybrid_fraction := 0.6
def non_hybrid_fraction := 1 - hybrid_fraction

-- Number of hybrid and non-hybrid cars
def hybrid_cars := hybrid_fraction * total_cars
def non_hybrid_cars := non_hybrid_fraction * total_cars

-- Fractions for hybrid cars
def hybrid_one_headlight_fraction := 0.4
def hybrid_upgraded_led_fraction := 0.15
def hybrid_luxury_headlights_fraction := 0.2
def hybrid_exclusive_edition_fraction := 0.05

-- Fractions for non-hybrid cars
def non_hybrid_upgraded_led_fraction := 0.3
def non_hybrid_luxury_headlights_fraction := 0.1
def non_hybrid_solar_powered_fraction := 0.05

-- Number of each type of headlight for hybrid cars
def hybrid_one_headlight_cars := hybrid_one_headlight_fraction * hybrid_cars
def hybrid_upgraded_led_cars := hybrid_upgraded_led_fraction * hybrid_cars
def hybrid_luxury_headlights_cars := hybrid_luxury_headlights_fraction * hybrid_cars
def hybrid_exclusive_edition_cars := hybrid_exclusive_edition_fraction * hybrid_cars

-- Number of each type of headlight for non-hybrid cars
def non_hybrid_upgraded_led_cars := non_hybrid_upgraded_led_fraction * non_hybrid_cars
def non_hybrid_luxury_headlights_cars := non_hybrid_luxury_headlights_fraction * non_hybrid_cars
def non_hybrid_solar_powered_cars := non_hybrid_solar_powered_fraction * non_hybrid_cars

-- The theorem to be proven
theorem headlight_counts :
  hybrid_one_headlight_cars = 192 ∧
  hybrid_upgraded_led_cars = 72 ∧
  hybrid_luxury_headlights_cars = 96 ∧
  hybrid_exclusive_edition_cars = 24 ∧
  non_hybrid_upgraded_led_cars = 96 ∧
  non_hybrid_luxury_headlights_cars = 32 ∧
  non_hybrid_solar_powered_cars = 16 :=
by
  sorry

end headlight_counts_l461_461401


namespace find_ax5_by5_l461_461385

theorem find_ax5_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 3) 
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16) 
  (h4 : a * x^4 + b * y^4 = 42) : 
  ax5_by5 : a * x^5 + b * y^5 = 20 :=
sorry

end find_ax5_by5_l461_461385


namespace determine_knight_liar_l461_461615

-- Definitions for the problem
def native : Type := ℕ
def knight (n : native) : Prop := sorry -- Define the condition for being a knight
def liar (n : native) : Prop := sorry -- Define the condition for being a liar

variable (ages : native → ℕ) -- Function to represent the ages of the natives

-- Conditions according to the problem statement
axiom native_type : ∀ n, native n ∧ (knight n ∨ liar n)
axiom correct_statement : ∀ n (k : knight n), 
  ages (n.left_neighbor) = correct_left_age n ∧ ages (n.right_neighbor) = correct_right_age n
axiom incorrect_statement : ∀ n (l : liar n), 
  (ages (n.left_neighbor) = incorrect_left_age n ∧ ages (n.right_neighbor) = incorrect_right_age n) ∨
  (ages (n.left_neighbor) = incorrect_right_age n ∧ ages (n.right_neighbor) = incorrect_left_age n)

-- Main theorem to prove
theorem determine_knight_liar : ∀ n, ∃ d : bool, (d = true → knight n) ∧ (d = false → liar n) := sorry

end determine_knight_liar_l461_461615


namespace assignment_schemes_count_l461_461008

-- Definitions of the problem's conditions
def students := {a, b, c, d}
def venues := {A, B, C}
def student_not_in_venue_A (assignment : a → venues) : Prop := 
  assignment a ≠ A

noncomputable def count_assignments : ℕ :=
  let assignments := {f : students → venues // (∀ v ∈ venues, ∃ s ∈ students, f s = v) ∧ student_not_in_venue_A f}
  in assignments.fintype.card

-- The theorem statement
theorem assignment_schemes_count : count_assignments = 24 := 
by sorry

end assignment_schemes_count_l461_461008


namespace simplify_expression_l461_461556

theorem simplify_expression :
  (Real.sqrt 5 * 5^(1/2) + 20 / 4 * 3 - 9^(3/2)) = -7 :=
by sorry

end simplify_expression_l461_461556


namespace number_of_men_l461_461497

-- Definitions based on conditions
def num_men : ℕ := 9
def expenditure_8_men : ℕ := 8 * 3
def total_expenditure : ℚ := 29.25
def avg_expenditure : ℕ → ℚ := λ n, total_expenditure / n
def ninth_expenditure (n : ℕ) : ℚ := (avg_expenditure n) + 2

-- Theorem to prove the number of men
theorem number_of_men :
  (expenditure_8_men + ninth_expenditure num_men = total_expenditure) → num_men = 9 :=
by
  sorry

end number_of_men_l461_461497


namespace two_roses_more_than_three_carnations_l461_461036

variable {x y : ℝ}

theorem two_roses_more_than_three_carnations
  (h1 : 6 * x + 3 * y > 24)
  (h2 : 4 * x + 5 * y < 22) :
  2 * x > 3 * y := 
by 
  sorry

end two_roses_more_than_three_carnations_l461_461036


namespace f_value_f_maximum_l461_461589

def f (x : ℝ) : ℝ := sin x ^ 2 - sin x ^ 2

theorem f_value : f() = 0 :=
by sorry

theorem f_maximum (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π) : f(x) = 0 :=
by sorry

end f_value_f_maximum_l461_461589


namespace rosie_pies_l461_461891

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end rosie_pies_l461_461891


namespace count_coprime_to_15_eq_8_l461_461267

def is_coprime_to_15 (a : ℕ) : Prop := Nat.gcd a 15 = 1

def count_coprime_to_15 (n : ℕ) : ℕ :=
  (Finset.filter (λ a, is_coprime_to_15 a) (Finset.range n)).card

theorem count_coprime_to_15_eq_8 : count_coprime_to_15 15 = 8 := by
  sorry

end count_coprime_to_15_eq_8_l461_461267


namespace floor_e_equals_2_l461_461692

theorem floor_e_equals_2 : Int.floor Real.exp = 2 := 
sorry

end floor_e_equals_2_l461_461692


namespace kelly_points_l461_461826

theorem kelly_points (K : ℕ) 
  (h1 : 12 + 2 * 12 + K + 2 * K + 12 / 2 = 69) : K = 9 := by
  sorry

end kelly_points_l461_461826


namespace number_of_triangles_with_perimeter_20_l461_461926

-- Declare the condition: number of triangles with integer side lengths and perimeter of 20
def integerTrianglesWithPerimeter (n : ℕ) : ℕ :=
  (Finset.range (n/2 + 1)).card

/-- Prove that the number of triangles with integer side lengths and a perimeter of 20 is 8. -/
theorem number_of_triangles_with_perimeter_20 : integerTrianglesWithPerimeter 20 = 8 := 
  sorry

end number_of_triangles_with_perimeter_20_l461_461926


namespace vector_addition_l461_461381

variable (a b : ℝ × ℝ)

def vector_a := (5, -3 : ℝ)
def vector_b := (-6, 4 : ℝ)

theorem vector_addition : vector_a + vector_b = (-1, 1 : ℝ) := by
  sorry

end vector_addition_l461_461381


namespace not_equal_7_6_l461_461978

theorem not_equal_7_6 :
  ¬ (1 + 2 / 8 = 7 / 6) :=
by
  -- Conditions given in the problem
  let A := (14 / 12)
  let B := (1 + 1 / 6)
  let C := (1 + 2 / 12)
  let D := (1 + 2 / 8)
  let E := (1 + 14 / 84)
  
  -- Correct answer derived from comparison
  have h1 : 7 / 6 = A := by sorry
  have h2 : 7 / 6 = B := by sorry
  have h3 : 7 / 6 = C := by sorry
  have h4 : 5 / 4 = D := by sorry
  have h5 : 7 / 6 = E := by sorry
  
  -- Proving the statement
  show ¬ (1 + 2 / 8 = 7 / 6), by
    rw [←h4]
    exact ne_of_lt (lt_of_le_of_ne (le_refl (5 / 4)) (ne.symm (ne_of_lt (by norm_num1))))

end not_equal_7_6_l461_461978


namespace abs_difference_of_numbers_l461_461137

theorem abs_difference_of_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 391) :
  |x - y| = 6 :=
sorry

end abs_difference_of_numbers_l461_461137


namespace product_of_axes_l461_461479

-- Definitions of geometric elements
variables (O A B C D F : Type)
variables (OA OB OC OD OF: ℝ)

-- Given conditions
def is_center_of_ellipse (O: Type) (A B C D: Type) := 
OF = 8 ∧
2 * (OC - 4) = 4 ∧
(OA = OB) ∧
(OC = OD) 

-- Prove the product (AB)(CD) is 240
theorem product_of_axes (O A B C D F : Type) (OB OC OA OD: ℝ) (of : 8) (diam : 4):
is_center_of_ellipse O A B C D → 
(AB*CD = 240) :=
begin
    sorry
end

end product_of_axes_l461_461479


namespace system_solution_l461_461246

theorem system_solution (u v w : ℚ) 
  (h1 : 3 * u - 4 * v + w = 26)
  (h2 : 6 * u + 5 * v - 2 * w = -17) :
  u + v + w = 101 / 3 :=
sorry

end system_solution_l461_461246


namespace rosie_can_make_nine_pies_l461_461895

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end rosie_can_make_nine_pies_l461_461895


namespace problem1_line_distance_problem2_cos_angle_between_lines_l461_461728

-- Definition of lines l1 and l2
def l1 (x y : ℝ) := 2*x + y + 6 = 0
def l2 (x y : ℝ) := 3*x - 4*y - 6 = 0

-- Problem 1: Find the equation of line l that is 2 units away from l2
theorem problem1_line_distance (x y : ℝ) (l : ℝ → ℝ → Prop) 
  (h1 : ∀ (x y : ℝ), l x y = (3*x - 4*y - 6 = 0)) :
  (∀ (x y : ℝ), l x y = (3*x - 4*y - 16 = 0) ∨ l x y = (3*x - 4*y + 4 = 0)) ↔ 
  (∃ m, ∀ (x y : ℝ), l x y = (3*x - 4*y + m = 0) ∧ abs (-6 - m) / sqrt (3^2 + 4^2) = 2) :=
sorry

-- Problem 2: Find the cosine of the angle between l1 and l2
theorem problem2_cos_angle_between_lines :
  (cos_theta : ℝ) (h2 : ∀ (θ : ℝ), cos θ =  2√5 / 25):
  ∃ θ, cos θ = cos_theta ∧ (cos θ = cos_theta ∧ θ ≠ 0) :=
sorry

end problem1_line_distance_problem2_cos_angle_between_lines_l461_461728


namespace hexagon_circumradius_l461_461483

theorem hexagon_circumradius
  (A B C D E F : Type)
  (hexagon : convex_hexagon A B C D E F)
  (side_length : ∀ (P Q : Type), P ≠ Q → dist P Q = 1) :
  ∃ (T : triangle), (T = triangle A C E ∨ T = triangle B D F) ∧ circumradius T ≤ 1 :=
sorry

structure convex_hexagon (A B C D E F : Type) :=
(convex : ∀ (P Q R S T U V W X Y Z : Type), convex_hull {P, Q, R, S, T, U, V, W, X, Y, Z} ≤ affine_hull {A, B, C, D, E, F})

structure triangle (A B C : Type) :=
(vertices : (A → B → C → Type))

def dist (P Q : Type) : Type := by sorry

def circumradius (T : triangle) : ℝ := by sorry

end hexagon_circumradius_l461_461483


namespace abelian_if_order_is_power_of_two_l461_461844

variable {G : Type*} [Group G]
variable (G_prop : ∀ (a b : G), a^2 * b = b * a^2 → a * b = b * a)

theorem abelian_if_order_is_power_of_two (hn : ∃ (n : ℕ), Fintype.card G = 2^n) :
  ∀ (a b : G), a * b = b * a :=
by
  sorry

end abelian_if_order_is_power_of_two_l461_461844


namespace positive_integers_between_300_and_1000_squared_l461_461797

theorem positive_integers_between_300_and_1000_squared :
  ∃ n : ℕ, 300 < n^2 ∧ n^2 < 1000 → ∃ m : ℕ, m = 14 := sorry

end positive_integers_between_300_and_1000_squared_l461_461797


namespace ellipse_equation_line_equation_l461_461744

-- Definitions based on given conditions
def ellipse (x y : ℝ) : ℝ :=
  x^2 / 4 + y^2

def focus1 := (-real.sqrt 3, 0)
def pointM := (real.sqrt 3, 1 / 2)
def pointP := (1, 0)
def area_triangle := 4 / 5

-- Theorem statements
theorem ellipse_equation :
  ellipse focus1.1 focus1.2 = 1 ∧ ellipse pointM.1 pointM.2 = 1 :=
by
  sorry

theorem line_equation (A B : ℝ × ℝ) :
  (∃ m : ℝ, (((A.1 = m * A.2 + 1) ∧ (B.1 = m * B.2 + 1)) ∧
  (((A.1 - B.1) / 2) * ((A.2 - B.2) / 2) = area_triangle)) ∧
  (A ≠ B ∧ ellipse A.1 A.2 = 1 ∧ ellipse B.1 B.2 = 1))) →
  ((∃ x y : ℝ, (A.1 = x ∧ A.2 = y) ∨ (A.1 = -x ∧ A.2 = -y)) ∨
  (∃ m : ℝ, (m = 1 ∨ m = -1))) :=
by
  sorry

end ellipse_equation_line_equation_l461_461744


namespace sum_first_2005_terms_l461_461939

theorem sum_first_2005_terms : (1 + 2 + 3 + 4) * 501 + 1 = 5011 := by
  -- Sum of one block (1, 2, 3, 4) is 10
  have sum_block : 1 + 2 + 3 + 4 = 10 := by
    exact rfl
  -- Number of complete blocks in the first 2005 terms
  have num_blocks : 2005 / 4 = 501 := by
    exact rfl
  -- Theorem result
  show (1 + 2 + 3 + 4) * 501 + 1 = 5011
  calc
    (1 + 2 + 3 + 4) * 501 + 1 = 10 * 501 + 1 : by rw [sum_block]
    ... = 5010 + 1 : rfl
    ... = 5011 : rfl

end sum_first_2005_terms_l461_461939


namespace valid_paths_count_l461_461468
open Nat

-- Define the starting and destination coordinates
def start : (ℕ × String) := (5, "A St")
def destination : (ℕ × String) := (1, "F St")

-- Define the constraint coordinate to avoid
def avoid : (ℕ × String) := (4, "B St")

-- The expected number of valid paths
def expected_paths : ℕ := 56

-- Proving the number of shortest paths avoiding the avoided intersection equals the expected number
theorem valid_paths_count : 
  (shortest_paths start destination avoid) = expected_paths := 
  sorry -- Proof to be filled in

end valid_paths_count_l461_461468


namespace f_fixed_point_l461_461045

-- Definitions and conditions based on the problem statement
def g (n : ℕ) : ℕ := sorry
def f (n : ℕ) : ℕ := sorry

-- Helper functions for the repeated application of f
noncomputable def f_iter (n x : ℕ) : ℕ := 
    Nat.iterate f (x^2023) n

axiom g_bijective : Function.Bijective g
axiom f_repeated : ∀ x : ℕ, f_iter x x = x
axiom f_div_g : ∀ (x y : ℕ), x ∣ y → f x ∣ g y

-- Main theorem statement
theorem f_fixed_point : ∀ x : ℕ, f x = x := by
  sorry

end f_fixed_point_l461_461045


namespace coefficient_of_x6_in_expansion_l461_461965

theorem coefficient_of_x6_in_expansion :
  let a := 1
  let b := -3 * (x : ℝ) ^ 3
  let n := 4
  let k := 2
  (1 - 3 * (x : ℝ) ^ 3) ^ 4 = ∑ k in finset.range (n + 1), 
    (nat.choose n k) * a ^ (n - k) * b ^ k →
  is_term_of_degree (1 - 3 * (x : ℝ) ^ 3) ^ 4 x 6 (54 * x ^ 6) :=
by
  sorry

end coefficient_of_x6_in_expansion_l461_461965


namespace sin_value_of_alpha_tan_and_sin_cos_ratio_l461_461765

variables {α : ℝ}

-- Condition: The terminal side of angle α passes through point P(1, 3)
def is_on_terminal_side (x y : ℝ) : Prop := (x, y) = (1, 3)

-- Condition: tan(π - α) = -2
def tan_supplement (α : ℝ) : Prop := Real.tan(π - α) = -2

-- Question 1: prove sin α = 3√10 / 10
theorem sin_value_of_alpha (h : is_on_terminal_side 1 3) : Real.sin α = 3 * Real.sqrt 10 / 10 := 
sorry

-- Question 2: prove (2sin α - cos α) / (sin α + 2cos α) = 3/4
theorem tan_and_sin_cos_ratio (h1 : Real.sin α = 3 * Real.sqrt 10 / 10) (h2 : tan_supplement α) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := 
sorry

end sin_value_of_alpha_tan_and_sin_cos_ratio_l461_461765


namespace count_integers_satisfying_condition_l461_461794

theorem count_integers_satisfying_condition :
  ({n : ℕ | 300 < n^2 ∧ n^2 < 1000}.card = 14) :=
by
  sorry

end count_integers_satisfying_condition_l461_461794


namespace b_2006_eq_4_l461_461070

noncomputable def b : ℕ → ℚ
| 1       := 3
| 2       := 4
| (n + 3) := b (n + 2) / b (n + 1)

theorem b_2006_eq_4 : b 2006 = 4 := sorry

end b_2006_eq_4_l461_461070


namespace min_a_squared_plus_b_squared_l461_461327

theorem min_a_squared_plus_b_squared (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 := 
sorry

end min_a_squared_plus_b_squared_l461_461327


namespace bugs_meet_again_at_P_l461_461155

def radius_large_circle := 6
def radius_small_circle := 3
def speed_large_bug := 4 * Real.pi
def speed_small_bug := 3 * Real.pi

def circumference_large_circle := 2 * radius_large_circle * Real.pi
def circumference_small_circle := 2 * radius_small_circle * Real.pi

def time_large_bug := circumference_large_circle / speed_large_bug
def time_small_bug := circumference_small_circle / speed_small_bug

theorem bugs_meet_again_at_P :
  Nat.lcm (Int.natAbs time_large_bug) (Int.natAbs time_small_bug) = 6 := by
  sorry

end bugs_meet_again_at_P_l461_461155


namespace problem1_part1_problem1_part2_l461_461780

theorem problem1_part1 (a b : ℝ)
  (h1 : 8 * a + 2 * b = 2)
  (h2 : 12 * a + b = 9) :
  a * b = -3 :=
sorry

theorem problem1_part2 (a b : ℝ)
  (ha : a = 1)
  (hb : b = -3) :
  set.range (λ x : ℝ, a * x^3 + b * x) = set.Icc (-2 : ℝ) (18 : ℝ) :=
sorry

end problem1_part1_problem1_part2_l461_461780


namespace floor_e_equals_two_l461_461686

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end floor_e_equals_two_l461_461686


namespace coprime_integers_lt_15_l461_461276

theorem coprime_integers_lt_15 : ∃ (S : Finset ℕ), S.card = 8 ∧ (∀ a ∈ S, a < 15 ∧ Nat.gcd a 15 = 1) :=
by
  sorry

end coprime_integers_lt_15_l461_461276


namespace total_phones_in_Delaware_l461_461520

def population : ℕ := 974000
def phones_per_1000 : ℕ := 673

theorem total_phones_in_Delaware : (population / 1000) * phones_per_1000 = 655502 := by
  sorry

end total_phones_in_Delaware_l461_461520


namespace sin_shift_right_pi_over_3_l461_461543

theorem sin_shift_right_pi_over_3 :
  ∀ x : ℝ,
    -sin (2 * x) = sin (2 * (x - (π / 3)) - π / 3) → 
    ∃ c : ℝ, c = π / 3 :=
by
  sorry

end sin_shift_right_pi_over_3_l461_461543


namespace maria_total_money_l461_461081

theorem maria_total_money
    (initial_dimes : ℕ := 4)
    (initial_quarters : ℕ := 4)
    (initial_nickels : ℕ := 7)
    (additional_quarters : ℕ := 5) :
    let total_quarters := initial_quarters + additional_quarters
    let amount_from_quarters := total_quarters * 0.25
    let amount_from_dimes := initial_dimes * 0.10
    let amount_from_nickels := initial_nickels * 0.05
    amount_from_quarters + amount_from_dimes + amount_from_nickels = 3.00 := 
by
    sorry

end maria_total_money_l461_461081


namespace number_of_invertible_integers_mod_15_l461_461271

theorem number_of_invertible_integers_mod_15 :
  (finset.card {a ∈ finset.range 15 | Int.gcd a 15 = 1}) = 8 := by
  sorry

end number_of_invertible_integers_mod_15_l461_461271


namespace number_of_integers_satisfying_l461_461790

theorem number_of_integers_satisfying (k1 k2 : ℕ) (hk1 : k1 = 300) (hk2 : k2 = 1000) :
  ∃ m : ℕ, m = 14 ∧ ∀ n : ℕ, 300 < n^2 → n^2 < 1000 → 18 ≤ n ∧ n ≤ 31 :=
by
  use 14
  sorry

end number_of_integers_satisfying_l461_461790


namespace coprime_integers_lt_15_l461_461275

theorem coprime_integers_lt_15 : ∃ (S : Finset ℕ), S.card = 8 ∧ (∀ a ∈ S, a < 15 ∧ Nat.gcd a 15 = 1) :=
by
  sorry

end coprime_integers_lt_15_l461_461275


namespace triangle_has_side_property_l461_461833

theorem triangle_has_side_property (a b c : ℝ) (A B C : ℝ) 
  (h₀ : 3 * b * Real.cos C + 3 * c * Real.cos B = a^2)
  (h₁ : A + B + C = Real.pi)
  (h₂ : a = 3) :
  a = 3 := 
sorry

end triangle_has_side_property_l461_461833


namespace number_of_integers_satisfying_l461_461789

theorem number_of_integers_satisfying (k1 k2 : ℕ) (hk1 : k1 = 300) (hk2 : k2 = 1000) :
  ∃ m : ℕ, m = 14 ∧ ∀ n : ℕ, 300 < n^2 → n^2 < 1000 → 18 ≤ n ∧ n ≤ 31 :=
by
  use 14
  sorry

end number_of_integers_satisfying_l461_461789


namespace complex_addition_identity_l461_461325

theorem complex_addition_identity (a b : ℝ) (h : a + b * complex.i = (11 - 7 * complex.i) / (1 - 2 * complex.i)) : a + b = 8 :=
sorry

end complex_addition_identity_l461_461325


namespace garden_area_l461_461573

theorem garden_area (length width : ℕ) (h_len : length = 12) (h_wid : width = 5) :
  length * width = 60 :=
by
  rw [h_len, h_wid]
  norm_num
  sorry

end garden_area_l461_461573


namespace apples_left_l461_461630

theorem apples_left (initial_apples : ℕ) (ricki_removes : ℕ) (samson_removes : ℕ) 
  (h1 : initial_apples = 74) 
  (h2 : ricki_removes = 14) 
  (h3 : samson_removes = 2 * ricki_removes) : 
  initial_apples - (ricki_removes + samson_removes) = 32 := 
by
  sorry

end apples_left_l461_461630


namespace triangle_AB_range_l461_461830

theorem triangle_AB_range (A B C : Type) [MetricSpace A] [MetricSpace B] 
  [MetricSpace C] (a b : ℝ) (h1 : ∠ACB = 120) (h2 : a + b = 2) 
  (h3 : AB^2 = a^2 + b^2 + a * b) : 
  √3 ≤ AB ∧ AB ≤ 2 :=
sorry

end triangle_AB_range_l461_461830


namespace proof_A_value_l461_461458

def largest_int_not_greater (x : ℝ) : ℤ := floor x

def A : ℤ :=
  largest_int_not_greater (
    (2008 * 80 + 2009 * 130 + 2010 * 180 : ℝ) /
    (2008 * 15 + 2009 * 25 + 2010 * 35 : ℝ))

theorem proof_A_value : A = 5 :=
  by
  sorry

end proof_A_value_l461_461458


namespace option_A_option_C_option_D_l461_461835

variable {A B C a b c : ℝ}

theorem option_A 
  (h1: 3 * b * Real.cos C + 3 * c * Real.cos B = a^2)
  (h2: A + B + C = Real.pi) : 
  a = 3 := sorry

theorem option_C 
  (h1: B = Real.pi - A - C) 
  (h2: C = 2 * A) 
  (h3: 0 < A) (h4: A < Real.pi / 2) 
  (h5: 0 < B) (h6: B < Real.pi / 2)
  (h7: 0 < C) (h8: C < Real.pi / 2) :
  3 * Real.sqrt 2 < c ∧ c < 3 * Real.sqrt 3 :=
  sorry

theorem option_D 
  (h1: A = 2 * C) 
  (h2: Real.sin B = 2 * Real.sin C) 
  (h3: B = Real.pi - A - C) 
  (O : Type) 
  [is_incenter_triangle_O ABC] : 
  area (triangle AOB) = (3 * Real.sqrt 3 - 3) / 4 :=
  sorry

end option_A_option_C_option_D_l461_461835


namespace bread_recipe_l461_461542

theorem bread_recipe (water_per_300_flour milk_per_300_flour flour_used : ℕ)
  (h_water : water_per_300_flour = 75)
  (h_milk : milk_per_300_flour = 60)
  (h_flour : flour_used = 900) :
  let portions := flour_used / 300 in
  let total_water := portions * water_per_300_flour in
  let total_milk := portions * milk_per_300_flour in
  total_water = 225 ∧ total_milk = 180 :=
by
  sorry

end bread_recipe_l461_461542


namespace fruit_needed_per_batch_l461_461914

theorem fruit_needed_per_batch (cost_milk_per_liter cost_fruit_per_kg : ℝ) 
  (milk_needed_per_batch total_cost_three_batches : ℝ) 
  (h_cost_milk : cost_milk_per_liter = 1.5) 
  (h_cost_fruit : cost_fruit_per_kg = 2) 
  (h_milk_needed : milk_needed_per_batch = 10) 
  (h_total_cost : total_cost_three_batches = 63)
  : 
    let cost_milk_per_batch := milk_needed_per_batch * cost_milk_per_liter in
    let total_cost_milk := 3 * cost_milk_per_batch in
    let cost_fruit_three_batches := total_cost_three_batches - total_cost_milk in
    let cost_fruit_per_batch := cost_fruit_three_batches / 3 in
    let fruit_needed_per_batch := cost_fruit_per_batch / cost_fruit_per_kg in
    fruit_needed_per_batch = 3 := 
by 
  sorry

end fruit_needed_per_batch_l461_461914


namespace number_and_sum_f_one_half_l461_461053

noncomputable def f : ℝ → ℝ := sorry

axiom f_property1 : f 1 = 2
axiom f_property2 : ∀ x y : ℝ, f (x * y + f x) = x * f y + 2 * f x

-- The statement we need to prove
theorem number_and_sum_f_one_half :
  let n := {y : ℝ | ∃ z : ℝ, y = f (1 / 2) ∧ f_property1 ∧ f_property2}.to_finset.card,
      s := {y : ℝ | ∃ z : ℝ, y = f (1 / 2) ∧ f_property1 ∧ f_property2}.to_finset.sum id in
  n * s = 1 :=
sorry

end number_and_sum_f_one_half_l461_461053


namespace smaller_number_is_72_l461_461950

theorem smaller_number_is_72
  (x : ℝ)
  (h1 : (3 * x - 24) / (8 * x - 24) = 4 / 9)
  : 3 * x = 72 :=
sorry

end smaller_number_is_72_l461_461950


namespace total_groups_of_two_marbles_l461_461148

-- Define the given conditions as constants
constant red_marble : Type
constant green_marble : Type
constant blue_marble : Type
constant purple_marble : Type
constant yellow_marble : Type

constant identical_yellow_marbles : ℕ

-- Prove the total number of different groups of two marbles
theorem total_groups_of_two_marbles :
  (identical_yellow_marbles = 4) →
  let marbles := [red_marble, green_marble, blue_marble, purple_marble, yellow_marble, yellow_marble, yellow_marble, yellow_marble] in
  (count_distinct_groups_of_two marbles = 11) :=
begin
  intros h,
  let marbles := [red_marble, green_marble, blue_marble, purple_marble, yellow_marble, yellow_marble, yellow_marble, yellow_marble],
  sorry -- proof omitted
end

end total_groups_of_two_marbles_l461_461148


namespace average_of_remaining_numbers_l461_461911

theorem average_of_remaining_numbers (s : ℝ) (a b c d e f : ℝ)
  (h1: (a + b + c + d + e + f) / 6 = 3.95)
  (h2: (a + b) / 2 = 4.4)
  (h3: (c + d) / 2 = 3.85) :
  ((e + f) / 2 = 3.6) :=
by
  sorry

end average_of_remaining_numbers_l461_461911


namespace probability_eventA_prob_three_vertices_same_color_l461_461096

/-- Define the event A that "three vertices on the same face are of the same color". -/
def eventA (tetra : ℕ) (colors : ℕ) : Prop :=
  ∃ (v1 v2 v3 v4 : ℕ), v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4 ∧
    (∀ (f : fin 4), ∃ (c : fin colors), ∀ (v : fin 3), v ∈ f → v < colors → c)

/-- Calculate the probability of complementary event. -/
def complementary_prob (total_colors : ℕ) : ℚ :=
  let total_cases : ℕ := total_colors ^ 4 in
  let favorable_cases : ℕ := 6 in
  (favorable_cases : ℚ) / (total_cases : ℚ)

/-- Calculate the probability of eventA using complementary probability. -/
theorem probability_eventA : complementary_prob 2 = 3 / 8 :=
  by
    sorry

/-- The probability that "three vertices on the same face are of the same color" is 5/8. -/
theorem prob_three_vertices_same_color : 1 - 3 / 8 = 5 / 8 :=
  by
    sorry


end probability_eventA_prob_three_vertices_same_color_l461_461096


namespace impossible_to_place_circles_l461_461094

theorem impossible_to_place_circles (A : ℝ) (n : ℕ) (r : ℕ → ℝ) 
  (hA : A = 1) 
  (h_non_overlap : ∀ i j, i ≠ j → disjoint (metric.closed_ball (0 : ℝ × ℝ) (r i)) (metric.closed_ball (0 : ℝ × ℝ) (r j)))
  (h_sum_radii : (finset.range n).sum r = 1962) : 
  false :=
sorry

end impossible_to_place_circles_l461_461094


namespace determine_n_l461_461626

-- All the terms used in the conditions
variables (S C M : ℝ)
variables (n : ℝ)

-- Define the conditions as hypotheses
def condition1 := M = 1 / 3 * S
def condition2 := M = 1 / n * C

-- The main theorem statement
theorem determine_n (S C M : ℝ) (n : ℝ) (h1 : condition1 S M) (h2 : condition2 M n C) : n = 2 :=
by sorry

end determine_n_l461_461626


namespace decipher_code_probability_l461_461477

theorem decipher_code_probability 
  (p : ℝ)
  (h1 : p = 0.3) : (1 - (1 - p) * (1 - p) = 0.51) :=
by
  -- Assume p = 0.3
  rw [h1]
  -- Compute (1 - p) = 0.7 as in conditions.
  let q := 1 - 0.3
  -- Evaluate the probability that neither can decipher the code
  have h2 : q * q = 0.49 := by norm_num
  -- Complement of this probability is the required probability we need to prove.
  show 1 - (q * q) = 0.51
  rw h2
  norm_num
  sorry

end decipher_code_probability_l461_461477


namespace cryptarithmetic_problem_l461_461012

theorem cryptarithmetic_problem :
  ∀ (A B C D : ℕ), A + B + C = 11 →
                    B + A + D = 10 →
                    A + D = 4 →
                    A ≠ B →
                    A ≠ C →
                    A ≠ D →
                    B ≠ C →
                    B ≠ D →
                    C ≠ D →
                    0 ≤ A ∧ A ≤ 9 →
                    0 ≤ B ∧ B ≤ 9 →
                    0 ≤ C ∧ C ≤ 9 →
                    0 ≤ D ∧ D ≤ 9 →
                    C = 4 :=
by { intros, sorry }

#eval cryptarithmetic_problem 1 6 4 3 sorry sorry sorry sorry sorry sorry sorry sorry sorry

end cryptarithmetic_problem_l461_461012


namespace monotonic_intervals_tangent_line_equation_l461_461771

section 
variable (e : ℝ)
noncomputable def f (x : ℝ) : ℝ := x^2 - 2*e*real.log x

theorem monotonic_intervals (e : ℝ) :
  (∀ x ∈ set.Ioo 0 (real.sqrt e), deriv (f e) x < 0) ∧
  (∀ x ∈ set.Ioi (real.sqrt e), deriv (f e) x > 0) :=
sorry

theorem tangent_line_equation (e : ℝ) :
  ∃ a b c : ℝ, a = (2*e - 2) ∧ b = 1 ∧ c = 1 - 2*e ∧
  ∀ x y, y = f e x → (a * x + b * y + c = 0) :=
sorry
end

end monotonic_intervals_tangent_line_equation_l461_461771


namespace no_integer_solutions_l461_461278

theorem no_integer_solutions (x y : ℤ) : 2^(2 * x) - 5^(2 * y) ≠ 79 :=
by
  sorry

end no_integer_solutions_l461_461278


namespace solve_inequalities_l461_461496

theorem solve_inequalities (x : ℝ) :
  (1 / x < 1 ∧ |4 * x - 1| > 2) →
  (x < -1/4 ∨ x > 1) :=
by
  sorry

end solve_inequalities_l461_461496


namespace gain_percent_is_approx_30_11_l461_461201

-- Definitions for cost price (CP) and selling price (SP)
def CP : ℕ := 930
def SP : ℕ := 1210

-- Definition for gain percent
noncomputable def gain_percent : ℚ :=
  ((SP - CP : ℚ) / CP) * 100

-- Statement to prove the gain percent is approximately 30.11%
theorem gain_percent_is_approx_30_11 :
  abs (gain_percent - 30.11) < 0.01 := by
  sorry

end gain_percent_is_approx_30_11_l461_461201


namespace eq_areas_quadrilaterals_l461_461443

open EuclideanGeometry

theorem eq_areas_quadrilaterals (ABC : Triangle) (AC_ne_BC : AC ≠ BC)
    (K : Point) (K_foot : foot_of_altitude(C, AB)) 
    (O : Point) (O_circumcenter : circumcenter(ABC, O)) :
  area(quadrilateral(A, K, O, C)) = area(quadrilateral(B, K, O, C)) := 
sorry

end eq_areas_quadrilaterals_l461_461443


namespace round_nearest_tenth_l461_461898

theorem round_nearest_tenth (x : ℝ) (hx : x = 4.76) : (round (x * 10) / 10) = 4.8 := by
  sorry

end round_nearest_tenth_l461_461898


namespace floor_e_eq_2_l461_461698

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_eq_2_l461_461698


namespace sum_cubes_coeffs_l461_461510

theorem sum_cubes_coeffs :
  ∃ a b c d e : ℤ, 
  (1000 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
  (a + b + c + d + e = 92) :=
sorry

end sum_cubes_coeffs_l461_461510


namespace inverse_44_mod_53_l461_461752

theorem inverse_44_mod_53 : (44 * 22) % 53 = 1 :=
by
-- Given condition: 19's inverse modulo 53 is 31
have h: (19 * 31) % 53 = 1 := by sorry
-- We should prove the required statement using the given condition.
sorry

end inverse_44_mod_53_l461_461752


namespace decrypt_messages_correctness_l461_461281

theorem decrypt_messages_correctness :
  ∃ algorithm : string → string,
    (algorithm "Шцлиц эцадгнюизх 15 гдхчжх ю шыжгйзт 13 ъыацчжх" = "Вахта закончится 12 ноября и вернусь 10 декабря") ∧
    (algorithm "101 идгг ъдчсидя гыкию ъцбю -40 еждмыгидш уккыаиюшгдзию ацеюицбдшбдьыгюя" = "50 тонн добытой нефти дали -28 процентов эффективности капитальных вложений") ∧
    (algorithm "Эцъцяиы задждзит шжцпыгюх адбыгнцидшд шцбц ш 124 ежю йщбы -63" = "Задайте скорость вращения коленчатого вала в 67 при угле -39") :=
by
  sorry

end decrypt_messages_correctness_l461_461281


namespace square_construction_unique_l461_461653

noncomputable def construct_square_with_distances (A B D O : Point) (OB OD: ℝ) : Prop :=
  (dist A B = dist A D) ∧ (dist B O = OB) ∧ (dist D O = OD) ->
  ∃ C : Point, is_square A B C D

theorem square_construction_unique (A B D O : Point) (OB OD: ℝ)
  (h1 : A ≠ O)
  (h2 : dist O B = OB)
  (h3 : dist O D = OD) :
  ∃ C : Point, is_square A B C D ∧ ∃ O': Point, (dist A O' = dist A O) ∧ ∠OAO' = 90 :=
sorry

end square_construction_unique_l461_461653


namespace functional_equation_solution_l461_461758

theorem functional_equation_solution (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)) →
  ∀ x : ℝ, f x = a * x^2 + b * x :=
by
  intro h
  intro x
  have : ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y) := h
  sorry

end functional_equation_solution_l461_461758


namespace angle_PQR_is_90_degrees_l461_461023

-- Conditions
variable (P Q R S : Point)
variable (PQ PS SQ : ℝ)
variable (h1 : PQ = 5)
variable (h2 : PS = 7)
variable (h3 : SQ = 7)
variable (angleQSP : Angle)
variable (h4 : angleQSP = 70)
variable (h5 : S - Q - R)

-- Goal
theorem angle_PQR_is_90_degrees (h1 : PQ = 5) (h2 : PS = 7) (h3 : SQ = 7) (h4 : angleQSP = 70) (h5 : S - Q - R) : anglePQR = 90 := 
by sorry

end angle_PQR_is_90_degrees_l461_461023


namespace smile_area_correct_l461_461900

noncomputable def C := (0, 0) -- Center of the semicircle
noncomputable def A := (-2, 0) -- Endpoint of the semicircle on the left
noncomputable def B := (2, 0) -- Endpoint of the semicircle on the right
noncomputable def D := (0, 2) -- Point D on the semicircle perpendicular to AB

noncomputable def E := (2, 3) -- Endpoint E extended from BD with arc radius 3
noncomputable def F := (-2, 4) -- Endpoint F extended from AD with arc radius 4

-- Define the sectors and semicircle
noncomputable def sector_ABE := (3^2 * Real.pi/4) / 2
noncomputable def sector_ABF := (4^2 * Real.pi/4) / 2
noncomputable def sector_DEF := (sqrt(2)^2 * Real.pi/4)

noncomputable def semicircle_ABD := Real.pi * 2^2 / 2

-- Area of the smile AEFBDA
noncomputable def area_smile := sector_ABE + sector_ABF - semicircle_ABD + sector_DEF

-- Theorem statement to be proved
theorem smile_area_correct : area_smile = 13 * Real.pi / 4 := 
by 
  sorry

end smile_area_correct_l461_461900


namespace distances_product_correct_l461_461020

noncomputable def circle_parametric (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

def line_parametric (t : ℝ) : ℝ × ℝ :=
  (1 + (Real.sqrt 3) / 2 * t, 1 + 1 / 2 * t)

def distances_product (c : ℝ → ℝ × ℝ) (l : ℝ → ℝ × ℝ) (P : ℝ × ℝ) : Bool :=
  let intersect (t : ℝ) : Bool :=
    let (x1, y1) := c t
    let (x2, y2) := find t (λ t, l t)
    x1 == x2 ∧ y1 == y2
  let ts := [t | t ∈ ℝ ∧ intersect t]
  let (t1, t2) := (ts.head, ts.tail.head)
  abs t1 * abs t2 == 2

theorem distances_product_correct : distances_product circle_parametric line_parametric (1, 1) = 2 :=
  sorry

end distances_product_correct_l461_461020


namespace number_of_distinct_ordered_pairs_l461_461662

-- Define the conditions as functions/properties in Lean
def satisfies_equation1 (x y : ℕ) : Prop :=
  (x * y)^4 - 2 * (x * y)^2 + 1 = 0

def satisfies_equation2 (x y : ℕ) : Prop :=
  x + y = 4

-- State the proof problem
theorem number_of_distinct_ordered_pairs : 
  {p : ℕ × ℕ // satisfies_equation1 p.1 p.2 ∧ satisfies_equation2 p.1 p.2}.card = 2 :=
by
  sorry

end number_of_distinct_ordered_pairs_l461_461662


namespace money_left_l461_461867

def entry_tickets_zoo := 5
def entry_tickets_aquarium := 7
def entry_tickets_show := 4
def bus_fare_per_transfer := 1.5
def total_transfers := 4
def souvenir_budget := 20
def amount_brought := 100
def noah_lunch := 10
def ava_lunch := 8
def beverage_cost := 3

theorem money_left (n_av_cost n_zoo n_aquarium n_show n_bus n_souv n_total_expense : ℝ) 
  (n_amount_left : ℝ) :
  n_av_cost = n_zoo + n_aquarium + n_show ∧
  n_zoo = 2 * entry_tickets_zoo ∧
  n_aquarium = 2 * entry_tickets_aquarium ∧
  n_show = 2 * entry_tickets_show ∧
  n_bus = 2 * 4 * bus_fare_per_transfer ∧
  n_souv = souvenir_budget ∧
  n_total_expense = n_av_cost + n_bus + n_souv ∧
  n_amount_left = amount_brought - n_total_expense ∧
  n_amount_left = 12 :=
sorry

end money_left_l461_461867


namespace compute_expression_l461_461242

theorem compute_expression : 12 * ((1/3 + 1/4 + 1/6)⁻¹) = 16 :=
by
  sorry

end compute_expression_l461_461242


namespace ratio_AF_FD_l461_461863

-- Definitions based on given conditions
variables (A B C D E F O: Type) [MetricSpace A] [MetricSpace O]
variables [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (circle : Set O) (line_AB : Set A) (line_CD : Set C) (line_AD : Set A) (line_OE : Set O)

-- Circle and points on the circle
variables (isCircle : MetricSpace.Circle O circle)
variables (tangentPoint_BC : B ∈ circle ∧ C ∈ circle)
variables (diameter_BD : ∃ D ∈ circle, (BD : line) is Diameter)

-- Intersection points
variables (meet_E : ∃ E, E ∈ (AB : line) ∧ E ∈ (CD : line))
variables (meet_F : ∃ F, F ∈ (AD : line) ∧ F ∈ (OE : line))

-- Lean statement to prove the problem
theorem ratio_AF_FD : 
  ∀ (A B C D E F O : Type)
  [MetricSpace A] [MetricSpace O]
  [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (circle : Set O) (line_AB : Set A) (line_CD : Set C) (line_AD : Set A) (line_OE : Set O)
  (isCircle : MetricSpace.Circle O circle)
  (tangentPoint_BC : B ∈ circle ∧ C ∈ circle)
  (diameter_BD : ∃ D ∈ circle, (BD : line) is Diameter)
  (meet_E : ∃ E, E ∈ (AB : line) ∧ E ∈ (CD : line))
  (meet_F : ∃ F, F ∈ (AD : line) ∧ F ∈ (OE : line)),
  |AF| / |FD| = 1 / 2 :=
by
  sorry

end ratio_AF_FD_l461_461863


namespace rosie_pies_l461_461887

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end rosie_pies_l461_461887


namespace part1_part2_l461_461355

-- Definition of the function f.
def f (x: ℝ) : ℝ := 2 * Real.log x + 1

-- Definition of the function g.
def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

-- Part 1: Prove that c ≥ -1 given f(x) ≤ 2x + c.
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
by
  -- Proof is omitted.
  sorry

-- Part 2: Prove that g(x) is monotonically decreasing on (0, a) and (a, +∞) given a > 0.
theorem part2 (a : ℝ) : a > 0 → (∀ x : ℝ, x > 0 → x ≠ a → 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo 0 a → x2 ∈ Ioo 0 a → x1 < x2 → g x2 a < g x1 a) ∧ 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x2 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x1 < x2 → g x2 a < g x1 a)) :=
by
  -- Proof is omitted.
  sorry

end part1_part2_l461_461355


namespace max_ratio_l461_461019

-- Definitions for points A, B and the circle
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (1, -1)
def P (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Definition of PB, PA and the ratio
def PB (x y : ℝ) : ℝ := real.sqrt ((x - 1)^2 + (y + 1)^2)
def PA (x y : ℝ) : ℝ := real.sqrt (x^2 + (y + 2)^2)
def ratio (x y : ℝ) : ℝ := PB x y / PA x y

-- Theorem statement
theorem max_ratio (x y : ℝ) (h : P x y) : ∃ t, t = ratio x y ∧ t ≤ 2 := sorry

end max_ratio_l461_461019


namespace sum_of_lunes_area_equals_triangle_l461_461243

theorem sum_of_lunes_area_equals_triangle (A B C : Point) (h_right_angle : ∠ BAC = 90°) :
  let AB := dist A B
  let AC := dist A C
  let BC := dist B C
  let area_triangle := (1 / 2) * AB * AC
  let area_semicircle_BC := (1 / 2) * Real.pi * (BC / 2)^2
  let area_semicircle_AB := (1 / 2) * Real.pi * (AB / 2)^2
  let area_semicircle_AC := (1 / 2) * Real.pi * (AC / 2)^2
  let sum_of_lunes := area_semicircle_AB + area_semicircle_AC
  sum_of_lunes - (area_semicircle_BC - area_triangle) = area_triangle :=
by
  sorry

end sum_of_lunes_area_equals_triangle_l461_461243


namespace floor_e_eq_two_l461_461703

theorem floor_e_eq_two
  (e_approx : Real ≈ 2.718) :
  ⌊e⌋ = 2 :=
sorry

end floor_e_eq_two_l461_461703


namespace num_people_in_5_years_l461_461228

def seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 12
  | (k+1) => 4 * seq k - 18

theorem num_people_in_5_years : seq 5 = 6150 :=
  sorry

end num_people_in_5_years_l461_461228


namespace angle_DBC_is_60_degrees_l461_461659

theorem angle_DBC_is_60_degrees
  (A B C D : Type)
  [IsTriangle A B C]
  (h1 : ∠ BAC = 50)
  (h2 : ∠ DAB = 10)
  (h3 : ∠ DCA = 30)
  (h4 : ∠ DBA = 20) :
  ∠ DBC = 60 :=
by
  sorry

end angle_DBC_is_60_degrees_l461_461659


namespace find_missing_number_l461_461110

theorem find_missing_number (x : ℝ) :
  (20 + 40 + 60) / 3 = (10 + x + 35) / 3 + 5 → x = 60 :=
by
  sorry

end find_missing_number_l461_461110


namespace min_a_value_l461_461778

noncomputable def f (x : ℝ) (m a : ℝ) : ℝ := m * x - a * Real.log x - m

noncomputable def g (x : ℝ) : ℝ := x / Real.exp (x - 1)

theorem min_a_value (a : ℝ) (h_a : a < 0) : 
  (∀ x1 x2 ∈ Icc (3 : ℝ) (4 : ℝ), x1 ≠ x2 → 
    |f x2 1 a - f x1 1 a| < |(1 / g x2) - (1 / g x1)|)
  → a ≥ 3 - (2 / 3) * Real.exp 2 :=
by
  sorry

end min_a_value_l461_461778


namespace intersection_of_sets_l461_461072

open Set

def A : Set ℝ := { x | 2^x ≤ 4 }
def B : Set ℝ := { x | ∃ y, y = Real.log10(x - 1) }

theorem intersection_of_sets : A ∩ B = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_of_sets_l461_461072


namespace rhombus_obtuse_angle_l461_461518

theorem rhombus_obtuse_angle (perimeter height : ℝ) (h_perimeter : perimeter = 8) (h_height : height = 1) : 
  ∃ θ : ℝ, θ = 150 :=
by
  sorry

end rhombus_obtuse_angle_l461_461518


namespace min_value_x2_minus_x1_l461_461333

noncomputable def f (x : ℝ) := 2 * Real.sin (Real.pi / 2 * x + Real.pi / 5)

theorem min_value_x2_minus_x1 :
  (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) → |x2 - x1| = 2 :=
sorry

end min_value_x2_minus_x1_l461_461333


namespace max_value_sum_products_l461_461533

theorem max_value_sum_products (a b c d : ℕ) (h1 : a ∈ {1, 3, 5, 7})
                                (h2 : b ∈ {1, 3, 5, 7})
                                (h3 : c ∈ {1, 3, 5, 7})
                                (h4 : d ∈ {1, 3, 5, 7})
                                (h5 : a + b + c + d = 16)
                                (h6 : a^2 + b^2 + c^2 + d^2 = 84) :
  ab + ac + bd + cd ≤ 64 := 
sorry

end max_value_sum_products_l461_461533


namespace clara_quarters_l461_461240

theorem clara_quarters :
  ∃ q : ℕ, 8 < q ∧ q < 80 ∧ q % 3 = 1 ∧ q % 4 = 1 ∧ q % 5 = 1 ∧ q = 61 :=
by
  sorry

end clara_quarters_l461_461240


namespace part1_part2_l461_461344

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.log x + 1

-- Part (1): Prove c ≥ -1 given f(x) ≤ 2x + c
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
  sorry

-- Define g with a > 0
def g (x a : ℝ) : ℝ := (f x - f a) / (x - a)

-- Part (2): Prove g is monotonically decreasing on (0, a) and (a, +∞)
theorem part2 (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, x > 0 → x ≠ a → g x a < g a a) :=
  sorry

end part1_part2_l461_461344


namespace simplify_trig_expression_l461_461103

theorem simplify_trig_expression (x : ℝ) :
  (cos (2 * x + π / 2) * sin (3 * π / 2 - 3 * x) - cos (2 * x - 5 * π) * cos (3 * x + 3 * π / 2)) /
  (sin (5 * π / 2 - x) * cos (4 * x) + sin x * cos (5 * π / 2 + 4 * x))
  = tan (5 * x) :=
sorry

end simplify_trig_expression_l461_461103


namespace unique_P_l461_461288

noncomputable def P (x : ℝ) : ℝ := x^n / (2^(2*n - 1) - 1)

theorem unique_P (P : ℝ → ℝ)
  (hP : ∀ (x y z t : ℕ), 
          x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t ∧
          x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ 
          x^2 + y^2 + z^2 = 2 * t^2 ∧ 
          Nat.gcd (Nat.gcd (Nat.gcd x y) z) t = 1 → 
          2 * (P t)^2 + 2 * P (x * y + y * z + z * x) = (P (x + y + z))^2) :
  ∀ x, P x = x^n / (2^(2*n - 1) - 1) :=
by
  sorry

end unique_P_l461_461288


namespace min_value_4a_plus_c_l461_461431

theorem min_value_4a_plus_c 
  (a c : ℝ) 
  (h_angle_eq : ∠ (ABC) = 120)
  (h_bisector : angle_bisector (ABC) (D) AC)
  (h_bd_eq : BD = 1) 
  (h_triangle : triangle ABC a) 
  (h_triangle : triangle ABC c)
  (h_eq : ac = a + c) : 
  4 * a + c = 9 :=
sorry

end min_value_4a_plus_c_l461_461431


namespace part1_c_range_part2_monotonicity_l461_461341

noncomputable def f (x : ℝ) := 2 * Real.log x + 1

theorem part1_c_range (c : ℝ) (x : ℝ) (h : a > 0) : f x ≤ 2 * x + c → c ≥ -1 :=
sorry

noncomputable def g (x a : ℝ) := (f x - f a) / (x - a)

theorem part2_monotonicity (a : ℝ) (h : a > 0) : monotone_decreasing_on g (0, a) ∧ monotone_decreasing_on g (a, +∞) :=
sorry

end part1_c_range_part2_monotonicity_l461_461341


namespace domain_of_f_l461_461644

variable (x : ℝ)

def f (x : ℝ) : ℝ := real.sqrt (3 - real.sqrt (5 - real.sqrt x))

theorem domain_of_f :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 25 ↔ f x = real.sqrt (3 - real.sqrt (5 - real.sqrt x))) :=
sorry

end domain_of_f_l461_461644


namespace right_triangle_area_outside_circle_l461_461415

theorem right_triangle_area_outside_circle (r : ℝ) :
  let AB := r,
      AC := r * real.sqrt 3,
      BC := 2 * r,
      area_triangle := (r * (r * real.sqrt 3)) / 2,
      area_circle := r^2 * ((π / 2) - (1)),
      area_outside := area_triangle - area_circle in
  (area_outside / area_triangle) = (real.sqrt 3 / 2) - (π / (2 * real.sqrt 3)) :=
sorry

end right_triangle_area_outside_circle_l461_461415


namespace soccer_and_volleyball_unit_prices_max_soccer_balls_l461_461414

-- Define the conditions and the problem
def unit_price_soccer_ball (x : ℕ) (y : ℕ) : Prop :=
  x = y + 15 ∧ 480 / x = 390 / y

def school_purchase (m : ℕ) : Prop :=
  m ≤ 70 ∧ 80 * m + 65 * (100 - m) ≤ 7550

-- Proof statement for the unit prices of soccer balls and volleyballs
theorem soccer_and_volleyball_unit_prices (x y : ℕ) (h : unit_price_soccer_ball x y) :
  x = 80 ∧ y = 65 :=
by
  sorry

-- Proof statement for the maximum number of soccer balls the school can purchase
theorem max_soccer_balls (m : ℕ) :
  school_purchase m :=
by
  sorry

end soccer_and_volleyball_unit_prices_max_soccer_balls_l461_461414


namespace circle_tangent_problem_l461_461196

theorem circle_tangent_problem (O A B C : EuclideanGeometry.Point)
  (radius : ℝ) (hO : EuclideanGeometry.is_center O)
  (hA : EuclideanGeometry.is_on_circle A O 2)
  (hTangent : EuclideanGeometry.is_tangent_segment A B O)
  (theta : ℝ) (s : ℝ) (c : ℝ) (hAOBAngle : EuclideanGeometry.angle O A B = theta)
  (hC_on_OA : EuclideanGeometry.is_on_segment C O A)
  (hBC_bisect_AOB : EuclideanGeometry.bisects C B (EuclideanGeometry.angle A B O))
  (hSinTheta : s = Real.sin theta) (hCosTheta : c = Real.cos theta) :
  EuclideanGeometry.distance O C = 2 * s / (1 + s) :=
sorry

end circle_tangent_problem_l461_461196


namespace distance_from_focus_to_asymptotes_parabola_hyperbola_l461_461714

noncomputable def distance_focus_to_asymptote : ℝ := 
  let focus := (1 : ℝ, 0) in
  let asymptote1 := (3 : ℝ, 4, 0) in -- representing 3x + 4y = 0 as (a, b, c)
  let asymptote2 := (3 : ℝ, -4, 0) in -- representing 3x - 4y = 0 as (a, b, c)
  min (abs (asymptote1.1 * focus.1 + asymptote1.2 * focus.2 + asymptote1.3) / real.sqrt (asymptote1.1 ^ 2 + asymptote1.2 ^ 2))
      (abs (asymptote2.1 * focus.1 + asymptote2.2 * focus.2 + asymptote2.3) / real.sqrt (asymptote2.1 ^ 2 + asymptote2.2 ^ 2))

theorem distance_from_focus_to_asymptotes_parabola_hyperbola : distance_focus_to_asymptote = 3 / 5 := 
by 
  sorry

end distance_from_focus_to_asymptotes_parabola_hyperbola_l461_461714


namespace seven_by_seven_grid_more_dark_than_light_l461_461646

theorem seven_by_seven_grid_more_dark_than_light : 
  ∀ (grid : Matrix Bool (Fin 7) (Fin 7)), 
  (∀ i j, ((i + j) % 2 = 0) → grid i j = tt) → 
  (∀ i j, ((i + j) % 2 ≠ 0) → grid i j = ff) → 
  (count (λ b, b = tt) (grid.flatten) - count (λ b, b = ff) (grid.flatten) = 1) :=
by
  sorry

end seven_by_seven_grid_more_dark_than_light_l461_461646


namespace log_condition_sufficient_not_necessary_log_condition_not_necessary_l461_461915

theorem log_condition_sufficient_not_necessary (x : ℝ) :
  (log 2 (2 * x - 3) < 1) → (x > 3 / 2) :=
by
  sorry

theorem log_condition_not_necessary (x : ℝ) :
  (∃ x, x > 3 / 2 ∧ ¬(log 2 (2 * x - 3) < 1)) :=
by
  sorry

end log_condition_sufficient_not_necessary_log_condition_not_necessary_l461_461915


namespace cost_of_kid_ticket_l461_461600

theorem cost_of_kid_ticket (total_people kids adults : ℕ) 
  (adult_ticket_cost kid_ticket_cost : ℕ) 
  (total_sales : ℕ) 
  (h_people : total_people = kids + adults)
  (h_adult_cost : adult_ticket_cost = 28)
  (h_kids : kids = 203)
  (h_total_sales : total_sales = 3864)
  (h_calculate_sales : adults * adult_ticket_cost + kids * kid_ticket_cost = total_sales)
  : kid_ticket_cost = 12 :=
by
  sorry -- Proof will be filled in

end cost_of_kid_ticket_l461_461600


namespace patty_heavier_before_losing_weight_l461_461876

theorem patty_heavier_before_losing_weight {w_R w_P w_P' x : ℝ}
  (h1 : w_R = 100)
  (h2 : w_P = 100 * x)
  (h3 : w_P' = w_P - 235)
  (h4 : w_P' = w_R + 115) :
  x = 4.5 :=
by
  sorry

end patty_heavier_before_losing_weight_l461_461876


namespace total_subscription_amount_l461_461619

theorem total_subscription_amount (x : ℕ) (h₁ : A = x + 9000) (h₂ : B = x + 5000) 
                                  (h₃ : 14700 / A = 35000 / (A + B + C)) 
                                  (C := x) :
  A + B + C = 50000 :=
by
  have h₄ : A * (A + B + C) = 14700 * 35000,
  { sorry },
  have h₅ : A = x + 9000,
  { sorry },
  have h₆ : B = x + 5000,
  { sorry },
  have h₇ : 14700 = 35000 * (x + 9000) / (3x + 14000),
  { sorry },
  have h₈ : x = 12000,
  { sorry },
  have h₉ : A = 21000,
  { sorry },
  have h₁₀ : B = 17000,
  { sorry },
  have h₁₁ : C = 12000,
  { sorry },
  have h₁₂ : A + B + C = 50000,
  { sorry },
  exact h₁₂

end total_subscription_amount_l461_461619


namespace false_proposition_C_l461_461168

theorem false_proposition_C : ¬ (∀ x : ℝ, 2^|x| > 1) :=
by
  sorry

end false_proposition_C_l461_461168


namespace part1_part2_part3_l461_461021

variables {A B C H N1 N2 N3 M1 M2 M3 D1 D2 D3 H1 H2 H3 : Type*}
variables [acute_angle_triangle A B C H]
variables [reflected_over H (segment B C) N1]
variables [reflected_over H (segment C A) N2]
variables [reflected_over H (segment A B) N3]
variables [midpoint M1 (segment B C)]
variables [midpoint M2 (segment C A)]
variables [midpoint M3 (segment A B)]
variables [reflected_over M1 (segment B C) D1]
variables [reflected_over M2 (segment C A) D2]
variables [reflected_over M3 (segment A B) D3]
variables [orthic_triangle H1 H2 H3 A B C]

theorem part1 : concyclic {N1, N2, N3, D1, D2, D3} := sorry
theorem part2 : area (triangle D1 D2 D3) = area (triangle A B C) := sorry
theorem part3 : area (triangle N1 N2 N3) = 4 * area (triangle H1 H2 H3) := sorry

end part1_part2_part3_l461_461021


namespace floor_e_eq_two_l461_461702

theorem floor_e_eq_two
  (e_approx : Real ≈ 2.718) :
  ⌊e⌋ = 2 :=
sorry

end floor_e_eq_two_l461_461702


namespace coefficient_x_6_in_expansion_l461_461970

-- Define the variable expressions and constraints of the problem
def expansion_expr : ℕ → ℤ := λ k, Nat.choose 4 k * 1^(4 - k) * (-3)^(k)
def term_coefficient_of_x_pow_6 (k : ℕ) : ℕ := if (3 * k = 6) then Nat.choose 4 k * 9 else 0

-- Prove that the coefficient of x^6 in the expansion of (1-3x^3)^4 is 54
theorem coefficient_x_6_in_expansion : term_coefficient_of_x_pow_6 2 = 54 := by
  -- Simplify the expression for the term coefficient of x^6 when k = 2
  simp only [term_coefficient_of_x_pow_6]
  split_ifs
  simp [Nat.choose, Nat.factorial]
  sorry -- one could continue simplifying this manually or provide arithmetic through Lean library

end coefficient_x_6_in_expansion_l461_461970


namespace find_square_constant_l461_461910

noncomputable def square_constant : ℝ :=
  let s := 38 / 4 in
  let a := s^2 in
  let p := 4 * s in
  let c := a - 2 * p in
  c

-- Proof statement
theorem find_square_constant (a p : ℝ) (h1 : a = (38 / 4)^2) (h2 : p = 38) (h3 : a = 2 * p + 14.25) : square_constant = 14.25 :=
by
  unfold square_constant
  rw [h1, h2, h3]
  norm_num
  exact eq.symm (real.eq_of_sub_eq_zero rfl)

end find_square_constant_l461_461910


namespace box_width_l461_461128

theorem box_width (
  length : ℝ,
  lowering_height : ℝ,
  removed_volume_gallons : ℝ,
  gallons_per_cubic_foot : ℝ,
  removed_volume_cubic_feet : ℝ
) : 
  length = 64 ∧
  lowering_height = 0.5 ∧
  removed_volume_gallons = 6000 ∧
  gallons_per_cubic_foot = 7.48052 ∧
  removed_volume_cubic_feet = removed_volume_gallons / gallons_per_cubic_foot →
  802.039 = length * lowering_height * 25.0637 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h_rest
  cases h_rest with h4 h5
  have V : removed_volume_cubic_feet = 802.039, by linarith,
  sorry

end box_width_l461_461128


namespace log_inequality_solution_l461_461301

theorem log_inequality_solution (x : ℝ) (h : log x (2 * x^2 + x - 1) > log x 2 - 1) : x > 1 ∧ x ≠ 1 :=
sorry

end log_inequality_solution_l461_461301


namespace floor_e_equals_two_l461_461687

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end floor_e_equals_two_l461_461687


namespace blocks_add_remove_l461_461620

theorem blocks_add_remove (A R : ℕ) (initial_blocks final_blocks added_blocks removed_blocks : ℕ) 
  (h1 : initial_blocks = 35) (h2 : final_blocks = 65) 
  (h3 : final_blocks = initial_blocks + added_blocks - removed_blocks) : 
  added_blocks - removed_blocks = 30 := 
by {
  rw [h1, h2] at h3,
  sorry
}

end blocks_add_remove_l461_461620


namespace part1_if_perpendicular_then_equal_part2_if_equal_then_perpendicular_l461_461063

noncomputable theory
open_locale classical

variables {A B C O Q E F : Type} [inner_product_space A] [inner_product_space B]
variables [is_isosceles_triangle : A B = A C]
variables [midpoint_M : ∃ M, M = (B + C) / 2]
variables [point_on_extension_O : ∃ O1, O1 ∈ (line (A, midpoint_M))]
variables [OB_perpendicular_AB : O1 = ⟨O1A, O1B⟩ ∈ orthogonal (line (B, A))]
variables [Q_on_BC : Q ∈ segment (B, C) ∧ Q ≠ B ∧ Q ≠ C]
variables [collinear_EQF : collinear {E, Q, F} ∧ E ≠ Q ∧ Q ≠ F]

/- (I) If \( OQ \perp EF \), then \( QE = QF \)-/
theorem part1_if_perpendicular_then_equal :
  ∀ {O Q E F}, (O, Q) ∈ orthogonal (line (E, F)) → (distance Q E = distance Q F) :=
begin
  sorry
end

/- (II) If \( QE = QF \), then \( OQ \perp EF \)-/
theorem part2_if_equal_then_perpendicular :
  ∀ {Q E F}, (distance Q E = distance Q F) → (Q, E) ∈ orthogonal (line (O, F)) :=
begin
  sorry
end

end part1_if_perpendicular_then_equal_part2_if_equal_then_perpendicular_l461_461063


namespace estimated_germination_probability_l461_461133

-- This definition represents the conditions of the problem in Lean.
def germination_data : List (ℕ × ℕ × Real) :=
  [(2, 2, 1.000), (5, 4, 0.800), (10, 9, 0.900), (50, 44, 0.880), (100, 92, 0.920),
   (500, 463, 0.926), (1000, 928, 0.928), (1500, 1396, 0.931), (2000, 1866, 0.933), (3000, 2794, 0.931)]

-- The theorem states that the germination probability is approximately 0.93.
theorem estimated_germination_probability (data : List (ℕ × ℕ × Real)) (h : data = germination_data) :
  ∃ p : Real, p = 0.93 ∧ ∀ n m r, (n, m, r) ∈ data → |r - p| < 0.01 :=
by
  -- Placeholder for proof
  sorry

end estimated_germination_probability_l461_461133


namespace number_of_invertible_integers_mod_15_l461_461268

theorem number_of_invertible_integers_mod_15 :
  (finset.card {a ∈ finset.range 15 | Int.gcd a 15 = 1}) = 8 := by
  sorry

end number_of_invertible_integers_mod_15_l461_461268


namespace train_cross_tree_time_l461_461189

/--
  Prove that the time it takes for the train to cross the tree is 120 seconds, 
  given the conditions that the train is 1200 meters long and it takes 
  200 seconds to pass an 800 meters long platform.
-/
theorem train_cross_tree_time :
  ∀ (train_length platform_length platform_time : ℕ), 
  train_length = 1200 ∧ platform_length = 800 ∧ platform_time = 200 →
  let v := (train_length + platform_length) / platform_time in
  let t := train_length / v in
  t = 120 :=
by
  intros train_length platform_length platform_time h
  rcases h with ⟨tl_eq, pl_eq, pt_eq⟩
  simp [tl_eq, pl_eq, pt_eq]
  sorry

end train_cross_tree_time_l461_461189


namespace CEMC_additional_employees_l461_461232

variable (t : ℝ)

def initialEmployees (t : ℝ) := t + 40

def finalEmployeesMooseJaw (t : ℝ) := 1.25 * t

def finalEmployeesOkotoks : ℝ := 26

def finalEmployeesTotal (t : ℝ) := finalEmployeesMooseJaw t + finalEmployeesOkotoks

def netChangeInEmployees (t : ℝ) := finalEmployeesTotal t - initialEmployees t

theorem CEMC_additional_employees (t : ℝ) (h : t = 120) : 
    netChangeInEmployees t = 16 := 
by
    sorry

end CEMC_additional_employees_l461_461232


namespace minimum_f_value_l461_461753

noncomputable def f (x y : ℝ) : ℝ :=
  y / x + 16 * x / (2 * x + y)

theorem minimum_f_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, (∀ x y, f x y ≥ t) ∧ t = 6 := sorry

end minimum_f_value_l461_461753


namespace quadrilateral_count_is_423_l461_461378

noncomputable def tetrahedron_quadrilateral_count : ℕ := 
  let vertices : ℕ := 4
  let midpoints : ℕ := 6
  let total_points : ℕ := vertices + midpoints
  -- The number of ways to choose 4 points out of 10 for quadrilaterals
  (finset.univ.card.choose 4)

theorem quadrilateral_count_is_423 :
  total_points == 10 → tetrahedron_quadrilateral_count = 423 := 
begin
  intros h_total_points,
  rw h_total_points,
  sorry
end

end quadrilateral_count_is_423_l461_461378


namespace fiona_correct_answers_l461_461908

-- 5 marks for each correct answer in Questions 1-15
def marks_questions_1_to_15 (correct1 : ℕ) : ℕ := 5 * correct1

-- 6 marks for each correct answer in Questions 16-25
def marks_questions_16_to_25 (correct2 : ℕ) : ℕ := 6 * correct2

-- 1 mark penalty for incorrect answers in Questions 16-20
def penalty_questions_16_to_20 (incorrect1 : ℕ) : ℕ := incorrect1

-- 2 mark penalty for incorrect answers in Questions 21-25
def penalty_questions_21_to_25 (incorrect2 : ℕ) : ℕ := 2 * incorrect2

-- Total marks given correct and incorrect answers
def total_marks (correct1 correct2 incorrect1 incorrect2 : ℕ) : ℕ :=
  marks_questions_1_to_15 correct1 +
  marks_questions_16_to_25 correct2 -
  penalty_questions_16_to_20 incorrect1 -
  penalty_questions_21_to_25 incorrect2

-- Fiona's total score
def fionas_total_score : ℕ := 80

-- The proof problem: Fiona answered 16 questions correctly
theorem fiona_correct_answers (correct1 correct2 incorrect1 incorrect2 : ℕ) :
  total_marks correct1 correct2 incorrect1 incorrect2 = fionas_total_score → 
  (correct1 + correct2 = 16) := sorry

end fiona_correct_answers_l461_461908


namespace sum_of_sequence_l461_461741

-- Definitions based on conditions
def a (n : ℕ) := 2 * n - 1
def b (n : ℕ) := 2^(a n) + n
def S (n : ℕ) := (Finset.range n).sum (λ i => b (i + 1))

-- The theorem assertion / problem statement
theorem sum_of_sequence (n : ℕ) : 
  S n = (2 * (4^n - 1)) / 3 + n * (n + 1) / 2 := 
sorry

end sum_of_sequence_l461_461741


namespace A_and_C_mutually_exclusive_l461_461723

/-- Definitions for the problem conditions. -/
def A (all_non_defective : Prop) : Prop := all_non_defective
def B (all_defective : Prop) : Prop := all_defective
def C (at_least_one_defective : Prop) : Prop := at_least_one_defective

/-- Theorem stating that A and C are mutually exclusive. -/
theorem A_and_C_mutually_exclusive (all_non_defective at_least_one_defective : Prop) :
  A all_non_defective ∧ C at_least_one_defective → false :=
  sorry

end A_and_C_mutually_exclusive_l461_461723


namespace hosting_schedules_count_l461_461188

theorem hosting_schedules_count :
  let n_universities := 6
  let n_years := 8
  let total_ways := 6 * 5 * 4^6
  let excluding_one := 6 * 5 * 4 * 3^6
  let excluding_two := 15 * 4 * 3 * 2^6
  let excluding_three := 20 * 3 * 2 * 1^6
  total_ways - excluding_one + excluding_two - excluding_three = 46080 := 
by
  sorry

end hosting_schedules_count_l461_461188


namespace max_sin_angle_APF_proof_l461_461748

noncomputable def max_sin_angle_APF (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (he : e = 1/5) : Prop :=
  let c := e * a in
  let A := (-a, 0) in
  let F := (-c, 0) in
  ∀ P : ℝ × ℝ, ∃ xq : ℝ × ℝ,
    P = (c, xq.snd) ∧
    (∀ θ : ℝ, θ = angle A P F →
      abs (sin θ) ≤ 1/2)

theorem max_sin_angle_APF_proof (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (he : e = 1/5) :
  max_sin_angle_APF a b h e he :=
sorry

end max_sin_angle_APF_proof_l461_461748


namespace pies_from_36_apples_l461_461881

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l461_461881


namespace collinear_and_concurrent_l461_461582

/-- 
  Construct three squares PBTV, TCRS, QRAP with the sides of ΔPTR as their diagonals.
  Construct three squares ADEB, BFGC, CHIA externally along the sides of ΔABC.
  Define intersection points:
    BH ∩ CE = M,
    CD ∩ AG = N,
    AF ∩ BI = O.
  
  Prove that:
  1. A, M, T are collinear.
  2. B, N, R are collinear.
  3. C, O, P are collinear.
  4. The lines AMT, BNR, and COP are concurrent at the orthocenter of ΔABC.
-/
theorem collinear_and_concurrent :
  ∀ (A B C D E F G H I M N O P Q R S T V: Point),
    square P B T V ∧ 
    square T C R S ∧
    square Q R A P ∧ 
    square A D E B ∧
    square B F G C ∧
    square C H I A ∧ 
    intersect (line B H) (line C E) = M ∧
    intersect (line C D) (line A G) = N ∧
    intersect (line A F) (line B I) = O →
    collinear A M T ∧
    collinear B N R ∧
    collinear C O P ∧
    concurrent (line A M T) (line B N R) (line C O P) :=
  begin
    sorry
  end

end collinear_and_concurrent_l461_461582


namespace wire_ratio_is_one_l461_461219

theorem wire_ratio_is_one (a b : ℝ) (h1 : a = b) : a / b = 1 := by
  -- The proof goes here
  sorry

end wire_ratio_is_one_l461_461219


namespace grid_min_max_open_doors_l461_461999

theorem grid_min_max_open_doors (n : ℕ) (rooms : matrix (fin 5) (fin 5) ℕ)
  (h_grid : ∀ i j, 0 ≤ rooms i j ∧ rooms i j ≤ 4)
  (h_doors : ∀ i, 0 ≤ n i ∧ n i ≤ 4)
  (h3_doors : ∀ i j, (∑ k, rooms i k) + (∑ l, rooms l j) - rooms i j = 3) :
  5 ≤ n ∧ n ≤ 19 :=
by
  sorry

end grid_min_max_open_doors_l461_461999


namespace sarah_marriage_age_l461_461668

theorem sarah_marriage_age
  (current_age : ℕ)
  (birth_month : ℕ)
  (name_length : ℕ) :
  current_age = 9 →
  birth_month = 7 →
  name_length = 5 →
  ((name_length + 2 * current_age) * birth_month = 161) :=
by
  intros hca hbm hnl
  simp [hca, hbm, hnl]
  sorry

end sarah_marriage_age_l461_461668


namespace inequality_proof_l461_461730

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (n : ℕ) (hn : 0 < n) : 
  (x / (n * x + y + z) + y / (x + n * y + z) + z / (x + y + n * z)) ≤ 3 / (n + 2) :=
sorry

end inequality_proof_l461_461730


namespace least_n_condition_l461_461038

-- Define the conditions and the question in Lean 4
def jackson_position (n : ℕ) : ℕ := sorry  -- Defining the position of Jackson after n steps

def expected_value (n : ℕ) : ℝ := sorry  -- Defining the expected value E_n

theorem least_n_condition : ∃ n : ℕ, (1 / expected_value n > 2017) ∧ (∀ m < n, 1 / expected_value m ≤ 2017) ∧ n = 13446 :=
by {
  -- Jackson starts at position 1
  -- The conditions described in the problem will be formulated here
  -- We need to show that the least n such that 1 / E_n > 2017 is 13446
  sorry
}

end least_n_condition_l461_461038


namespace point_after_rotation_l461_461429

-- Definitions based on conditions
def point_N : ℝ × ℝ := (-1, -2)
def origin_O : ℝ × ℝ := (0, 0)
def rotation_180 (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- The statement to be proved
theorem point_after_rotation :
  rotation_180 point_N = (1, 2) :=
by
  sorry

end point_after_rotation_l461_461429


namespace find_point_on_BC_l461_461049

variable {A B C P : Point}
variable {PA XY PB PC AB AC : ℝ}
variable [Geometry]

-- Define a triangle ABC
def is_triangle (A B C : Point) : Prop := 
  ∃ Δ : Triangle, Δ.vertices = {A, B, C}

-- Define point P on line segment BC
def point_on_segment (P B C : Point) : Prop := 
  ∃ t : ℝ, t ∈ Icc 0 1 ∧ P = (1 - t) • B + t • C

-- Define the property involving PA and XY
def property (PA XY PB PC AB AC : ℝ) : Prop :=
  (PA / XY)^2 + (PB * PC) / (AB * AC) = 1

theorem find_point_on_BC 
  (h_triangle : is_triangle A B C)
  (h_segment : point_on_segment P B C)
  (h_property : property PA XY PB PC AB AC) : 
  PB = (ab / (b + c)) ∨ PB = (ac / (b + c)) :=
sorry

end find_point_on_BC_l461_461049


namespace decrease_in_demand_l461_461595

theorem decrease_in_demand (init_price new_price demand : ℝ) (init_demand : ℕ) (price_increase : ℝ) (original_revenue new_demand : ℝ) :
  init_price = 20 ∧ init_demand = 500 ∧ price_increase = 5 ∧ demand = init_price + price_increase ∧ 
  original_revenue = init_price * init_demand ∧ new_demand ≤ init_demand ∧ 
  new_demand * demand ≥ original_revenue → 
  init_demand - new_demand = 100 :=
by 
  sorry

end decrease_in_demand_l461_461595


namespace cosine_dihedral_angle_l461_461629

-- Definitions of the geometric objects and their properties
variable (A B C D E: Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (AB AD AE: ℝ)

-- Given conditions
def conditions : Prop :=
  AB = 2 ∧ AD = 4 ∧ AE = 3 ∧ AD - AE = 1 ∧ ∃ O, BD = 2 * Real.sqrt 5 ∧ O ∈ Line(B, D) ∧ O ∈ Line(E, C) ∧ BD.perp EC

-- The dihedral angle's cosine calculation
theorem cosine_dihedral_angle (h : conditions A B C D E AB AD AE) : 
  cos (dihedral_angle D B E C) = 7 / 8 :=
sorry

end cosine_dihedral_angle_l461_461629


namespace desks_in_classroom_l461_461198

theorem desks_in_classroom (d c : ℕ) (h1 : c = 4 * d) (h2 : 4 * c + 6 * d = 728) : d = 33 :=
by
  -- The proof is omitted, this placeholder is to indicate that it is required to complete the proof.
  sorry

end desks_in_classroom_l461_461198


namespace decrypt_messages_correctness_l461_461280

theorem decrypt_messages_correctness :
  ∃ algorithm : string → string,
    (algorithm "Шцлиц эцадгнюизх 15 гдхчжх ю шыжгйзт 13 ъыацчжх" = "Вахта закончится 12 ноября и вернусь 10 декабря") ∧
    (algorithm "101 идгг ъдчсидя гыкию ъцбю -40 еждмыгидш уккыаиюшгдзию ацеюицбдшбдьыгюя" = "50 тонн добытой нефти дали -28 процентов эффективности капитальных вложений") ∧
    (algorithm "Эцъцяиы задждзит шжцпыгюх адбыгнцидшд шцбц ш 124 ежю йщбы -63" = "Задайте скорость вращения коленчатого вала в 67 при угле -39") :=
by
  sorry

end decrypt_messages_correctness_l461_461280


namespace at_least_one_greater_than_16000_l461_461959

open Nat

theorem at_least_one_greater_than_16000 (seq : Fin 20 → ℕ)
  (h_distinct : ∀ i j : Fin 20, i ≠ j → seq i ≠ seq j)
  (h_perfect_square : ∀ i : Fin 19, ∃ k : ℕ, (seq i) * (seq (i + 1)) = k^2)
  (h_first : seq 0 = 42) : ∃ i : Fin 20, seq i > 16000 :=
by
  sorry

end at_least_one_greater_than_16000_l461_461959


namespace length_of_rectangle_l461_461525

noncomputable def rectangle_length (b l : ℝ) : Prop :=
  (2 * (l + b) / b = 5) ∧ (l * b = 216) ∧ l = 18

theorem length_of_rectangle :
  ∃ (b l : ℝ), rectangle_length b l :=
by
  use 12, 18
  dsimp [rectangle_length]
  split
  { norm_num }
  split
  { norm_num }
  { norm_num }


end length_of_rectangle_l461_461525


namespace find_lambda_l461_461380

noncomputable def vector_a := (-1, 3) : ℝ × ℝ
noncomputable def vector_b := (2, 1) : ℝ × ℝ
def lambda_value : ℝ := Real.sqrt 2

def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem find_lambda (λ : ℝ) (h : λ > 0) :
  is_perpendicular (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2)
                   (vector_a.1 - λ * vector_b.1, vector_a.2 - λ * vector_b.2) →
  λ = λ_value :=
by
  -- Proof steps go here
  sorry

end find_lambda_l461_461380


namespace recurring_decimal_sum_l461_461285

theorem recurring_decimal_sum :
  let x := 0.\overline{123} in
  let y := 0.\overline{000123} in
  x + y = (154 : ℚ) / (1001 : ℚ) :=
sorry

end recurring_decimal_sum_l461_461285


namespace coordinate_equation_solution_l461_461672

theorem coordinate_equation_solution (x y : ℝ) :
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 →
  (y = -x - 2) ∨ (y = -2 * x + 1) :=
by
  sorry

end coordinate_equation_solution_l461_461672


namespace probability_of_drawing_red_ball_l461_461992

noncomputable def probability_of_red_ball (total_balls red_balls : ℕ) : ℚ :=
  red_balls / total_balls

theorem probability_of_drawing_red_ball:
  probability_of_red_ball 5 3 = 3 / 5 :=
by
  unfold probability_of_red_ball
  norm_num

end probability_of_drawing_red_ball_l461_461992


namespace proof_problem_l461_461726

variables {a b c d e : ℝ}

theorem proof_problem (h1 : a * b^2 * c^3 * d^4 * e^5 < 0) (h2 : b^2 ≥ 0) (h3 : d^4 ≥ 0) :
  a * b^2 * c * d^4 * e < 0 :=
sorry

end proof_problem_l461_461726


namespace composite_sum_of_powers_l461_461482

theorem composite_sum_of_powers (a b c d : ℕ) (h : 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d) (h_prod : a * b = c * d) :
    ¬(nat.prime (a ^ 1984 + b ^ 1984 + c ^ 1984 + d ^ 1984)) :=
sorry

end composite_sum_of_powers_l461_461482


namespace train_crosses_bridge_in_approx_39_6_sec_l461_461787

noncomputable def train_crossing_time (train_length : ℕ) (bridge_length : ℕ) (speed_km_per_hr : ℕ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600
  total_distance / speed_m_per_s

theorem train_crosses_bridge_in_approx_39_6_sec : 
  train_crossing_time 250 300 50 ≈ 39.6 := 
  sorry

end train_crosses_bridge_in_approx_39_6_sec_l461_461787


namespace product_units_tens_digit_not_divisible_by_4_but_by_3_l461_461869

theorem product_units_tens_digit_not_divisible_by_4_but_by_3 :
  ∃ n ∈ {3684, 3704, 3714, 3732, 3882},
    (¬ (n % 100 % 4 = 0) ∧ (List.sum (Int.toNat <$> n.digits 10) % 3 = 0)) →
    (n % 10 * (n / 10 % 10) = 4) :=
by
  -- Here list the elements that satisfy the given condition
  let eligible_numbers := [3714]

  -- Provide the introduction to the numbers under consideration
  have h3714 : (¬ (3714 % 100 % 4 = 0) ∧ (List.sum (Int.toNat <$> 3714.digits 10) % 3 = 0)) :=
    by sorry

  -- Check that the product of the units digit and the tens digit of 3714 is 4
  have product_3714 : 3714 % 10 * (3714 / 10 % 10) = 4 := 
    by sorry
  
  -- Conclude the theorem using the provided checks
  use 3714
  split
  . right
  . exact rfl
  . exact ⟨h3714, product_3714⟩

end product_units_tens_digit_not_divisible_by_4_but_by_3_l461_461869


namespace pies_from_36_apples_l461_461877

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l461_461877


namespace smallest_positive_period_of_f_l461_461664

noncomputable def f (x : ℝ) : ℝ := (Real.tan x) / (1 + (Real.tan x)^2)

theorem smallest_positive_period_of_f :
  Function.periodic f π :=
sorry

end smallest_positive_period_of_f_l461_461664


namespace mod_equiv_l461_461846

theorem mod_equiv (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (25 * m + 3 * n) % 83 = 0 ↔ (3 * m + 7 * n) % 83 = 0 :=
by
  sorry

end mod_equiv_l461_461846


namespace transform_sin_graph_l461_461545

theorem transform_sin_graph :
  let original_func := λ x : ℝ, (1/2) * sin (2 * x),
      target_func := λ x : ℝ, (1/4) * sin x in
  (∀ x : ℝ, target_func x = ((1/2) * (original_func (x / 2)))) :=
by
  -- The actual proof will be provided here
  sorry

end transform_sin_graph_l461_461545


namespace irrational_S_concatenation_l461_461574

def S (n : ℕ) : ℕ :=
  (n.digits 10).sum

def concatenated_S : ℝ :=
  ∑' n, S(n) / 10^n

theorem irrational_S_concatenation : irrational concatenated_S := sorry

end irrational_S_concatenation_l461_461574


namespace area_square_PAED_l461_461872

variables (A B C D E P : Point)
variables (k1 k2 k3 : ℝ)
variables [Field ℝ]

noncomputable def distance (X Y : Point) : ℝ := sorry
noncomputable def angle (X Y Z : Point) : ℝ := sorry
noncomputable def area_of_triangle (X Y Z : Point) : ℝ := sorry
noncomputable def area_of_quadrilateral (X Y Z W : Point) : ℝ := sorry

-- Given conditions
axiom h1 : collinear A E C
axiom h2 : collinear B E D
axiom h3 : distance A E = 1
axiom h4 : distance B E = 4
axiom h5 : distance C E = 3
axiom h6 : distance D E = 2
axiom h7 : angle A E B = 60
axiom h8 : ∃ P, intersects (line_through A B) (line_through C D) = P

-- Proof objective
theorem area_square_PAED (m n : ℕ) (h9 : gcd m n = 1) : 
  (area_of_quadrilateral P A E D) ^ 2 = (867 / 100) * 3 
  ∧ m + n = 967 :=
sorry

end area_square_PAED_l461_461872


namespace petya_cards_numbers_l461_461478

theorem petya_cards_numbers (cards : Finset ℕ) (h : cards = {1, 2, 3, 4}) :
  (∃ n ∈ permutations cards, to_nat n > 2222) → (n > 2222 ∧ count n = 16) :=
by {
  sorry
}

end petya_cards_numbers_l461_461478


namespace inscribed_cone_sphere_radius_l461_461316

theorem inscribed_cone_sphere_radius (O : Type) [normed_group O] [normed_space ℝ O] :
  ∀ (radius : ℝ), (radius = 2) → 
  ∃ (cone_volume_max : ℝ),
  (radius_of_sphere_inscribed_in_cone cone_volume_max = (4 * (real.sqrt 3 - 1)) / 3) :=
by sorry

end inscribed_cone_sphere_radius_l461_461316


namespace arrangements_l461_461509

def digits := [1, 2, 3, 4, 5, 6, 7, 8]

-- Function to check divisibility by k
def is_divisible_by (n k : ℕ) : Prop := n % k = 0

-- Function representing the arrangement in a grid
-- ... (details depending on specific approach of arrangement which we abstract here)

-- Main theorem
theorem arrangements (k : ℕ) : k ∈ [2, 3, 4, 5, 6] →
  (k = 2 ∨ k = 3 → 
    ∃ (arrangement : list (list ℕ)), -- using list to represent the grid and numbers
    (∀ nums ∈ arrangement, is_divisible_by nums k)) ∧ 
  ((k = 4 ∨ k = 5 ∨ k = 6) → 
    ¬ ∃ (arrangement : list (list ℕ)),
    (∀ nums ∈ arrangement, is_divisible_by nums k)) :=
by
  -- proof goes here
  sorry

end arrangements_l461_461509


namespace cost_price_is_100_l461_461387

noncomputable def cost_price_of_article (C : ℝ) : Prop :=
  let SP1 := 345
  let SP2 := 350
  ∃ G : ℝ,
    SP1 = C + (G / 100) * C ∧
    SP2 = C + ((G + 5) / 100) * C ∧
    C = 100

theorem cost_price_is_100 : ∃ C : ℝ, cost_price_of_article C :=
  by {
    use 100,
    sorry
  }

end cost_price_is_100_l461_461387


namespace arithmetic_seq_sum_l461_461821

-- Define arithmetic sequence for natural numbers
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ (n : ℕ), a n = a 0 + n * (a 1 - a 0)

-- Given conditions
variables (a : ℕ → ℤ) [arithmetic_sequence a] (h : a 3 + a 4 + a 5 + a 7 + a 8 + a 9 = 300)

-- Theorem statement
theorem arithmetic_seq_sum : a 2 + a 10 = 100 :=
by sorry

end arithmetic_seq_sum_l461_461821


namespace fraction_of_orange_juice_is_correct_l461_461546

noncomputable def fraction_of_orange_juice_in_mixture (V1 V2 juice1_ratio juice2_ratio : ℚ) : ℚ :=
  let juice1 := V1 * juice1_ratio
  let juice2 := V2 * juice2_ratio
  let total_juice := juice1 + juice2
  let total_volume := V1 + V2
  total_juice / total_volume

theorem fraction_of_orange_juice_is_correct :
  fraction_of_orange_juice_in_mixture 800 500 (1/4) (1/3) = 7 / 25 :=
by sorry

end fraction_of_orange_juice_is_correct_l461_461546


namespace popcorn_kernel_count_l461_461840

theorem popcorn_kernel_count :
  ∀ (distance : ℕ) (interval : ℕ) (fraction : ℚ) 
    (dropped_kernels squirrel_eaten remaining : ℕ),
    distance = 5000 →
    interval = 25 →
    fraction = 1/4 →
    dropped_kernels = distance / interval →
    squirrel_eaten = (fraction * dropped_kernels : ℚ).toNat →
    remaining = dropped_kernels - squirrel_eaten →
    remaining = 150 :=
by
  intros distance interval fraction dropped_kernels squirrel_eaten remaining h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end popcorn_kernel_count_l461_461840


namespace triangle_area_is_correct_l461_461426

noncomputable def isosceles_triangle_area : Prop :=
  let side_large_square := 6 -- sides of the large square WXYZ
  let area_large_square := side_large_square * side_large_square
  let side_small_square := 2 -- sides of the smaller squares
  let BC := side_large_square - 2 * side_small_square -- length of BC
  let height_AM := side_large_square / 2 + side_small_square -- height of the triangle from A to M
  let area_ABC := (BC * height_AM) / 2 -- area of the triangle ABC
  area_large_square = 36 ∧ BC = 2 ∧ height_AM = 5 ∧ area_ABC = 5

theorem triangle_area_is_correct : isosceles_triangle_area := sorry

end triangle_area_is_correct_l461_461426


namespace greatest_possible_perimeter_l461_461408

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem greatest_possible_perimeter :
  ∃ x : ℕ, 6 ≤ x ∧ x < 17 ∧ is_triangle x (2 * x) 17 ∧ (x + 2 * x + 17 = 65) := by
  sorry

end greatest_possible_perimeter_l461_461408


namespace max_marks_l461_461467

theorem max_marks (M : ℝ) (score passing shortfall : ℝ)
  (h_score : score = 212)
  (h_shortfall : shortfall = 44)
  (h_passing : passing = score + shortfall)
  (h_pass_cond : passing = 0.4 * M) :
  M = 640 :=
by
  sorry

end max_marks_l461_461467


namespace frac_b4_b3_a2_a1_l461_461457

-- Define the conditions
variables {x y a1 a2 a3 b1 b2 b3 b4 : ℝ}
variable h1 : x ≠ y
variable ha : a3 = a2 + (a2 - a1) ∧ a2 = a1 + (a1 - x) ∧ y = a3 + (a3 - a2)
variable hb : b4 = b3 + 2 * (b3 - b2) ∧ b3 = b2 + (b2 - x) ∧ y = b4 + (b4 - b3)

-- Define the theorem to be proven
theorem frac_b4_b3_a2_a1 : x ≠ y ∧ 
  (a3 = a2 + (a2 - a1)) ∧ (a2 = a1 + (a1 - x)) ∧ (y = a3 + (a3 - a2)) ∧ 
  (b4 = b3 + 2 * (b3 - b2)) ∧ (b3 = b2 + (b2 - x)) ∧ (y = b4 + (b4 - b3)) 
  → (b4 - b3) / (a2 - a1) = 8 / 3 := by sorry

end frac_b4_b3_a2_a1_l461_461457


namespace billiard_table_path_length_invariance_l461_461631

theorem billiard_table_path_length_invariance {n : ℕ} (hn : 2 < 2 * n) :
  ∀ (P : ℝ × ℝ) (u : ℝ × ℝ),
    P ∈ perimeter_points_of_regular_2n_gon n →
    (u = small_diagonal_vector_of_regular_2n_gon n) →
    path_length_of_ball_in_regular_2n_gon P u =
    path_length_of_ball_in_regular_2n_gon some_initial_point u := sorry

end billiard_table_path_length_invariance_l461_461631


namespace floor_e_equals_two_l461_461684

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end floor_e_equals_two_l461_461684


namespace geometric_seq_arithmetic_seq_l461_461011

theorem geometric_seq_arithmetic_seq (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = q * a n)
  (h2 : a 1 > 0)
  (h3 : a 1, (1 / 2) * a 3, 2 * a 2 form_arithmetic_sequence) :
  (a 5 / a 3 = 3 + 2 * Real.sqrt 2) :=
begin
  sorry
end

end geometric_seq_arithmetic_seq_l461_461011


namespace probability_of_three_blue_beans_l461_461983

-- Define the conditions
def red_jellybeans : ℕ := 10 
def blue_jellybeans : ℕ := 10 
def total_jellybeans : ℕ := red_jellybeans + blue_jellybeans 
def draws : ℕ := 3 

-- Define the events
def P_first_blue : ℚ := blue_jellybeans / total_jellybeans 
def P_second_blue : ℚ := (blue_jellybeans - 1) / (total_jellybeans - 1) 
def P_third_blue : ℚ := (blue_jellybeans - 2) / (total_jellybeans - 2) 
def P_all_three_blue : ℚ := P_first_blue * P_second_blue * P_third_blue 

-- Define the correct answer
def correct_probability : ℚ := 1 / 9.5 

-- State the theorem
theorem probability_of_three_blue_beans : 
  P_all_three_blue = correct_probability := 
sorry

end probability_of_three_blue_beans_l461_461983


namespace incenter_and_centroid_coordinates_l461_461031

-- Define the sides of the triangle
def x := 13
def y := 15
def z := 6

-- Define the conditions for the coordinates of I (Incenter)
def I (a b c : ℝ) (X Y Z : ℝ) := a * X + b * Y + c * Z
def incenter_coords : Prop := a + b + c = 1

-- Define the conditions for the coordinates of G (Centroid)
def G (p q r : ℝ) (X Y Z : ℝ) := p * X + q * Y + r * Z
def centroid_coords : Prop := p + q + r = 1

theorem incenter_and_centroid_coordinates
  (a b c p q r X Y Z : ℝ) :
  incenter_coords a b c ∧ centroid_coords p q r →
  a = (13 : ℝ) / (13 + 15 + 6) ∧
  b = (15 : ℝ) / (13 + 15 + 6) ∧
  c = (6 : ℝ) / (13 + 15 + 6) ∧
  p = (1 : ℝ) / 3 ∧
  q = (1 : ℝ) / 3 ∧
  r = (1 : ℝ) / 3 :=
by
  sorry

end incenter_and_centroid_coordinates_l461_461031


namespace circle_area_from_diameter_endpoints_l461_461090

theorem circle_area_from_diameter_endpoints :
  let C := (-2, 3)
  let D := (4, -1)
  let diameter := Real.sqrt ((4 - (-2))^2 + ((-1) - 3)^2)
  let radius := diameter / 2
  let area := Real.pi * radius^2
  C = (-2, 3) ∧ D = (4, -1) → area = 13 * Real.pi := by
    sorry

end circle_area_from_diameter_endpoints_l461_461090


namespace plane_split_into_four_regions_l461_461249

theorem plane_split_into_four_regions {x y : ℝ} :
  (y = 3 * x) ∨ (y = (1 / 3) * x - (2 / 3)) →
  ∃ r : ℕ, r = 4 :=
by
  intro h
  -- We must show that these lines split the plane into 4 regions
  sorry

end plane_split_into_four_regions_l461_461249


namespace count_coprime_to_15_eq_8_l461_461266

def is_coprime_to_15 (a : ℕ) : Prop := Nat.gcd a 15 = 1

def count_coprime_to_15 (n : ℕ) : ℕ :=
  (Finset.filter (λ a, is_coprime_to_15 a) (Finset.range n)).card

theorem count_coprime_to_15_eq_8 : count_coprime_to_15 15 = 8 := by
  sorry

end count_coprime_to_15_eq_8_l461_461266


namespace worker_time_per_toy_l461_461617

def hours_per_toy (total_hours : ℕ) (total_toys : ℕ) : ℕ :=
  total_hours / total_toys

theorem worker_time_per_toy :
  hours_per_toy 120 60 = 2 :=
by
  rw [hours_per_toy]
  norm_num
  sorry

end worker_time_per_toy_l461_461617


namespace solution_set_of_inequality_l461_461134

theorem solution_set_of_inequality : 
  { x : ℝ | (x + 2) * (1 - x) > 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l461_461134


namespace differentiable_at_eq_limit_l461_461055

open Real

variable {f : ℝ → ℝ}
variable {x₀ : ℝ}

theorem differentiable_at_eq_limit (h₁ : DifferentiableAt ℝ f x₀)
                                   (h₂ : tendsto (λ Δx : ℝ, (f (x₀ + 3 * Δx) - f x₀) / Δx) (𝓝 0) (𝓝 1)) :
  deriv f x₀ = 1 / 3 :=
by
  sorry

end differentiable_at_eq_limit_l461_461055


namespace solution_pairs_l461_461673

theorem solution_pairs (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) := 
by 
  sorry

end solution_pairs_l461_461673


namespace max_min_values_of_f_l461_461923

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values_of_f :
  let I := set.Icc (0 : ℝ) 3 in
  (∃ x ∈ I, ∀ y ∈ I, f y ≤ f x ∧ f x = 5) ∧
  (∃ x ∈ I, ∀ y ∈ I, f x ≤ f y ∧ f x = -15) :=
by
  sorry

end max_min_values_of_f_l461_461923


namespace part1_part2_l461_461775

-- Problem statement (1)
theorem part1 (a : ℝ) (h : a = -3) :
  (∀ x : ℝ, (x^2 + a * x + 2) ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2) →
  { x : ℝ // (x^2 + a * x + 2) ≥ 1 - x^2 } = { x : ℝ // x ≤ 1 / 2 ∨ x ≥ 1 } :=
sorry

-- Problem statement (2)
theorem part2 (a : ℝ) :
  (∀ x : ℝ, (x^2 + a * x + 2) + x^2 + 1 = 2 * x^2 + a * x + 3) →
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ (2 * x^2 + a * x + 3) = 0) →
  -5 < a ∧ a < -2 * Real.sqrt 6 :=
sorry

end part1_part2_l461_461775


namespace my_problem_l461_461092

theorem my_problem (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  a^2 * b + b^2 * c + c^2 * a < a^2 * c + b^2 * a + c^2 * b := 
sorry

end my_problem_l461_461092


namespace coefficient_x6_expansion_l461_461964

theorem coefficient_x6_expansion : 
  (∀ x : ℝ, coefficient (expand (1 - 3 * x ^ 3) 4) x 6 = 54) := 
sorry

end coefficient_x6_expansion_l461_461964


namespace geometric_sequence_sum_l461_461862

theorem geometric_sequence_sum (S : ℕ → ℤ) (n : ℕ) (a q : ℤ) 
  (S2_eq : S 2 = 3) 
  (S4_eq : S 4 = 15) :
  S 6 = 63 :=
begin
  sorry
end

end geometric_sequence_sum_l461_461862


namespace range_of_a_l461_461373

theorem range_of_a (a m n : ℝ) (P Q : ℝ → ℝ → Prop) (h1 : P 1 m) (h2 : Q (2 * a + 2) n) 
  (h3 : m > n) : 0 < a ∧ a < 1/2 ∨ a < -1/2 :=
by
  -- Definitions for quadratic function y = ax^2 - 4ax + 1
  let y := λ x : ℝ, a * x^2 - 4 * a * x + 1
  -- Point P is on the graph of y
  have hP : P 1 (y 1), from h1
  -- Point Q is on the graph of y
  have hQ : Q (2 * a + 2) (y (2 * a + 2)), from h2
  -- Given m > n
  exact sorry

end range_of_a_l461_461373


namespace seating_arrangements_l461_461413

theorem seating_arrangements : 
  let total := Nat.factorial 10
  let block := Nat.factorial 7 * Nat.factorial 4 
  total - block = 3507840 := 
by 
  let total := Nat.factorial 10
  let block := Nat.factorial 7 * Nat.factorial 4 
  sorry

end seating_arrangements_l461_461413


namespace dots_on_level_Z_l461_461421

theorem dots_on_level_Z : 
  ( ∀ n : ℕ, (n % 2 = 0 → dots (n + 1) = dots n)
    ∧ (n % 2 ≠ 0 → dots (n + 1) = 2 * dots n) )
  ∧ (dots 1 = 1)
  → dots 26 = 8192 :=
by
  sorry

end dots_on_level_Z_l461_461421


namespace cubic_roots_expression_l461_461058

noncomputable def root1 : ℂ := sorry
noncomputable def root2 : ℂ := sorry
noncomputable def root3 : ℂ := sorry

theorem cubic_roots_expression :
  (root1 + root2)^3 + (root2 + root3)^3 + (root3 + root1)^3 + 100 = 717 :=
by
  have h_roots : (Polynomial.C (6 : ℂ) * X ^ 3 + Polynomial.C (500 : ℂ) * X + Polynomial.C (1234 : ℂ)).roots = {root1, root2, root3} := sorry
  sorry

end cubic_roots_expression_l461_461058


namespace total_cans_in_tower_l461_461516

theorem total_cans_in_tower : 
  ∃ n : ℕ, (n = 11) ∧ (∀ (a_1 a_n : ℤ), a_1 = 30 → a_n = 1 → a_1 + (n-1)*d = a_n → d = -3 
    → (∑ i in (finset.range n), (a_1 + i * d)) = 170.5) := sorry

end total_cans_in_tower_l461_461516


namespace cos_pi_add_alpha_gt_zero_l461_461814

variable (α : ℝ) 

-- Angle α in the second quadrant implies sin(α) > 0 and cos(α) < 0
axiom angle_in_second_quadrant : 
(sin α > 0) ∧ (cos α < 0)

theorem cos_pi_add_alpha_gt_zero : 
  cos (π + α) > 0 :=
by 
  have h1 : cos (π + α) = -cos α := by sorry
  have h2 : cos α < 0 := angle_in_second_quadrant.right
  have h3: -cos α > 0 := by sorry
  exact h3

-- Ensure no compilation errors
#print axioms cos_pi_add_alpha_gt_zero

end cos_pi_add_alpha_gt_zero_l461_461814


namespace cost_to_marked_price_ratio_l461_461213

variable (p : ℝ)
def marked_price := p
def selling_price := (3 / 4) * p
def cost_price := (2 / 3) * selling_price

theorem cost_to_marked_price_ratio :
  cost_price / marked_price = 1 / 2 :=
by
  sorry

end cost_to_marked_price_ratio_l461_461213


namespace count_coprime_to_15_eq_8_l461_461264

def is_coprime_to_15 (a : ℕ) : Prop := Nat.gcd a 15 = 1

def count_coprime_to_15 (n : ℕ) : ℕ :=
  (Finset.filter (λ a, is_coprime_to_15 a) (Finset.range n)).card

theorem count_coprime_to_15_eq_8 : count_coprime_to_15 15 = 8 := by
  sorry

end count_coprime_to_15_eq_8_l461_461264


namespace sin_alpha_plus_7pi_over_6_l461_461724

variable (α : ℝ)

theorem sin_alpha_plus_7pi_over_6 (h : cos (α - π / 3) = 3 / 4) : sin (α + 7 * π / 6) = -3 / 4 := by
  sorry

end sin_alpha_plus_7pi_over_6_l461_461724


namespace radius_of_inscribed_circle_l461_461159

/-- Define the problem conditions -/
def d1 : ℝ := 14
def d2 : ℝ := 30
def r : ℝ := 105 / (2 * Real.sqrt 274)

/-- Define the side length of the rhombus in terms of the diagonals -/
def side_length : ℝ := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)

/-- Define the area of the rhombus in terms of the diagonals and in terms of the inscribed circle -/
def area_with_diagonals : ℝ := (d1 * d2) / 2
def area_with_side_and_radius : ℝ := 4 * side_length * r

/-- The statement to be proved: the radius of the inscribed circle in the rhombus with given diagonals -/
theorem radius_of_inscribed_circle : area_with_diagonals = area_with_side_and_radius := by
  sorry

end radius_of_inscribed_circle_l461_461159


namespace find_slope_of_linear_function_l461_461845

noncomputable def linear_function (m b x : ℝ) := m * x + b

theorem find_slope_of_linear_function (m b : ℝ) :
  (∫ x in 3..5, linear_function m b x = 0) →
  (∫ x in 5..7, linear_function m b x = 12) →
  m = 3 :=
by
  sorry

end find_slope_of_linear_function_l461_461845


namespace half_angle_tangent_sum_triangle_inequality_l461_461725

variables {A B C : ℝ}
variables {a b c R S : ℝ}
variables [triangle : triangle ABC a b c R S]

theorem half_angle_tangent_sum_triangle_inequality :
    tan (A / 2) + tan (B / 2) + tan (C / 2) ≤ 9 * R ^ 2 / (4 * S) := 
sorry

end half_angle_tangent_sum_triangle_inequality_l461_461725


namespace option_a_equals_sqrt2_over_2_option_c_equals_sqrt2_over_2_l461_461166

theorem option_a_equals_sqrt2_over_2 : 2 * sin (67.5) * cos (67.5) = real.sqrt 2 / 2 := by
  sorry

theorem option_c_equals_sqrt2_over_2 : 1 - 2 * sin (22.5)^2 = real.sqrt 2 / 2 := by
  sorry

end option_a_equals_sqrt2_over_2_option_c_equals_sqrt2_over_2_l461_461166


namespace sampling_correctness_probability_calculation_l461_461544
noncomputable theory

-- Define the total number of factories in districts A, B, and C.
def total_factories_A : ℕ := 18
def total_factories_B : ℕ := 27
def total_factories_C : ℕ := 18

-- Define the total number of factories.
def total_factories : ℕ := total_factories_A + total_factories_B + total_factories_C

-- Define the sample size.
def sample_size : ℕ := 7

-- Define the correct number of sampled factories in districts A, B, C.
def sampled_factories_A : ℕ := 2
def sampled_factories_B : ℕ := 3
def sampled_factories_C : ℕ := 2

-- First part (Ⅰ): Prove the number of sampled factories from each district
theorem sampling_correctness :
  sampled_factories_A + sampled_factories_B + sampled_factories_C = sample_size ∧
  sampled_factories_A * 9 = total_factories_A ∧
  sampled_factories_B * 9 = total_factories_B ∧
  sampled_factories_C * 9 = total_factories_C := 
by
  unfold sampled_factories_A sampled_factories_B sampled_factories_C sample_size
  unfold total_factories_A total_factories_B total_factories_C
  split
  . exact rfl
  . split
  . rfl
  . split
  . rfl
  . rfl

-- Second part (Ⅱ): Prove the probability that at least one of the two factories comes from district A is 11/21
theorem probability_calculation :
  2 / 7 := sorry


end sampling_correctness_probability_calculation_l461_461544


namespace floor_e_eq_2_l461_461697

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_eq_2_l461_461697


namespace population_approx_10000_2090_l461_461283

def population (initial_population : ℕ) (years : ℕ) : ℕ :=
  initial_population * 2 ^ (years / 20)

theorem population_approx_10000_2090 :
  ∃ y, y = 2090 ∧ population 500 (2090 - 2010) = 500 * 2 ^ (80 / 20) :=
by
  sorry

end population_approx_10000_2090_l461_461283


namespace find_hat_b_l461_461866

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem find_hat_b :
  let x := [1, 2, 3, 4, 5]
  let y := [0.5, 0.6, 1.0, 1.4, 1.5]
  let x̄ := mean x
  let ȳ := mean y
  ∃ (b : ℝ), ȳ = b * x̄ + 0.16 ∧ b = 0.28 :=
by
  let x := [1, 2, 3, 4, 5]
  let y := [0.5, 0.6, 1.0, 1.4, 1.5]
  let x̄ := mean x
  let ȳ := mean y
  have hx : x̄ = 3 := by sorry
  have hy : ȳ = 1 := by sorry
  use 0.28
  split
  · -- Proof that ȳ = 0.28 * 3 + 0.16
    rw [hx, hy]
    norm_num

  · -- Proof that b = 0.28
    norm_num

end find_hat_b_l461_461866


namespace max_value_f_on_interval_range_of_tangent_lines_l461_461362

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

theorem max_value_f_on_interval : ∃ x ∈ Icc (-2:ℝ) 1, f x = sqrt 2 :=
sorry

theorem range_of_tangent_lines (t : ℝ) (P : ℝ × ℝ) (hP : P = (1, t)) :
  (∃ x, 6 * x^2 - 3 = 0) → (∃ (x : ℝ), 2 * x^3 - 3 * x = t) → 
  -3 < t ∧ t < -1 :=
sorry

end max_value_f_on_interval_range_of_tangent_lines_l461_461362


namespace find_k_l461_461450

theorem find_k 
  (k : ℝ) 
  (hk : k > 1) 
  (h_sum : summable (λ n : ℕ, (6 * (n + 1) - 2) / k^(n + 1)) ∧ ∑' n, (6 * (n + 1) - 2) / k^(n + 1) = 5) : 
  k = 3 := 
sorry

end find_k_l461_461450


namespace max_full_box_cards_l461_461439

-- Given conditions
def total_cards : ℕ := 94
def unfilled_box_cards : ℕ := 6

-- Define the number of cards that are evenly distributed into full boxes
def evenly_distributed_cards : ℕ := total_cards - unfilled_box_cards

-- Prove that the maximum number of cards a full box can hold is 22
theorem max_full_box_cards (h : evenly_distributed_cards = 88) : ∃ x : ℕ, evenly_distributed_cards % x = 0 ∧ x = 22 :=
by 
  -- Proof goes here
  sorry

end max_full_box_cards_l461_461439


namespace part1_part2_l461_461352

-- Definition of the function f.
def f (x: ℝ) : ℝ := 2 * Real.log x + 1

-- Definition of the function g.
def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

-- Part 1: Prove that c ≥ -1 given f(x) ≤ 2x + c.
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
by
  -- Proof is omitted.
  sorry

-- Part 2: Prove that g(x) is monotonically decreasing on (0, a) and (a, +∞) given a > 0.
theorem part2 (a : ℝ) : a > 0 → (∀ x : ℝ, x > 0 → x ≠ a → 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo 0 a → x2 ∈ Ioo 0 a → x1 < x2 → g x2 a < g x1 a) ∧ 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x2 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x1 < x2 → g x2 a < g x1 a)) :=
by
  -- Proof is omitted.
  sorry

end part1_part2_l461_461352


namespace lines_from_equation_l461_461678

-- Definitions for the conditions
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Equivalent Lean statement to the proof problem
theorem lines_from_equation :
  (∀ x y : ℝ, satisfies_equation x y → (y = -x - 2) ∨ (y = -2 * x + 1)) :=
by
  intros x y h
  sorry

end lines_from_equation_l461_461678


namespace tan_A_over_tan_B_l461_461075

theorem tan_A_over_tan_B {A B C a b c : ℝ} 
(H1 : a = c * sin A / sin C)
(H2 : b = c * sin B / sin C)
(H3 : a * cos B - b * cos A = (3 / 5) * c) :
(tan A / tan B) = 4 :=
sorry

end tan_A_over_tan_B_l461_461075


namespace curve_representation_l461_461917

theorem curve_representation (ρ θ : ℝ) (x y : ℝ) 
  (h1 : x = ρ * cos θ)
  (h2 : y = ρ * sin θ)
  (h3 : ρ * cos θ = 2 * sin (2 * θ)) :
  (x = 0 ∨ x^2 + y^2 = 4 * y) :=
  sorry

end curve_representation_l461_461917


namespace minimum_passed_l461_461186

def total_participants : Nat := 100
def num_questions : Nat := 10
def correct_answers : List Nat := [93, 90, 86, 91, 80, 83, 72, 75, 78, 59]
def passing_criteria : Nat := 6

theorem minimum_passed (total_participants : ℕ) (num_questions : ℕ) (correct_answers : List ℕ) (passing_criteria : ℕ) :
  100 = total_participants → 10 = num_questions → correct_answers = [93, 90, 86, 91, 80, 83, 72, 75, 78, 59] →
  passing_criteria = 6 → 
  ∃ p : ℕ, p = 62 := 
by
  sorry

end minimum_passed_l461_461186


namespace rent_increase_percentage_l461_461207

theorem rent_increase_percentage (N : ℝ) (hN : N > 0) : 
  let original_tax_rate := 1 / 11
  let new_tax_rate := 1 / 10
  let x := 9.1 / 100 in
  (N - N * original_tax_rate) = ((N + N * x) - (N + N * x) * new_tax_rate) :=
by
  -- Define original net income
  let original_net_income := N * (1 - original_tax_rate)
  -- Define new total income
  let increased_income := N * (1 + x)
  -- Define new net income
  let new_net_income := increased_income * (1 - new_tax_rate)
  -- Prove the equality
  have : original_net_income = new_net_income :=
    sorry
  exact this

end rent_increase_percentage_l461_461207


namespace proof_problem_l461_461743

noncomputable def S_n (a_n : ℕ → ℕ) (n : ℕ) := ∑ i in finset.range (n+1), a_n i

def a_seq (n : ℕ) := 1 - 2 * n
def b_seq (n : ℕ) := 2 ^ n

def c_seq (n : ℕ) : ℚ :=
  if n % 2 = 0 then (2*n-1) / 2^(n-1) else 2

def T_n (n : ℕ) : ℚ :=
  if n % 2 = 0 then
    let k := n / 2 in
    2 * k + 26 / 9 - (12 * k + 13) / (9 * 2^(2*k-1))
  else 
    let k := (n + 1) / 2 in
    2 * k + 26 / 9 - (12 * k + 1) / (9 * 2^(2*k-3))

theorem proof_problem :
  (∀ n, S_n a_seq n = (n * (2 - 2 * n)) / 2) →
  (b_seq 1 = 2) →
  (b_seq 2 = 4) →
  (a_seq 3 + b_seq 2 = -1) →
  (S_n a_seq 3 + 2 * b_seq 3 = 7) →
  (∀ n, T_n n = 
    if n % 2 = 0 then
      let k := n / 2 in
      2 * k + 26 / 9 - (12 * k + 13) / (9 * 2^(2*k-1))
    else
      let k := (n + 1) / 2 in
      2 * k + 26 / 9 - (12 * k + 1) / (9 * 2^(2*k-3))) :=
begin
  sorry -- no proof needed, as specified
end

end proof_problem_l461_461743


namespace length_of_FD_l461_461424

theorem length_of_FD (a b c d f e : ℝ) (x : ℝ) :
  a = 0 ∧ b = 8 ∧ c = 8 ∧ d = 0 ∧ 
  e = 8 * (2 / 3) ∧ 
  (8 - x)^2 = x^2 + (8 / 3)^2 ∧ 
  a = d → c = b → 
  d = 8 → 
  x = 32 / 9 :=
by
  sorry

end length_of_FD_l461_461424


namespace sum_of_angles_l461_461762

-- Define the sides ratio of the triangle
def sides_ratio (a b c : ℝ) : Prop :=
  a / b = 5 / 7 ∧ b / c = 7 / 8

-- Define the cosine rule as a property
def cosine_rule (a b c : ℝ) (θ : ℝ) : Prop :=
  cos θ = (a^2 + b^2 - c^2) / (2 * a * b)

-- Proof statement
theorem sum_of_angles (a b c : ℝ) (θ : ℝ)
  (h_sides_ratio : sides_ratio a b c)
  (h_cosine_rule : cosine_rule 5 7 8 θ) : 
  180 - θ = 120 := 
  sorry

end sum_of_angles_l461_461762


namespace sufficient_but_not_necessary_condition_l461_461180

theorem sufficient_but_not_necessary_condition (x : ℝ) : 
    (1 / x ≥ 1 → 2 ^ (x - 1) ≤ 1) ∧ (¬(1 / x ≥ 1) → 2 ^ (x - 1) ≤ 1) := 
sorry

end sufficient_but_not_necessary_condition_l461_461180


namespace distance_from_P_to_directrix_l461_461749

-- Define an ellipse and necessary properties
def semiMajorAxis : ℝ := sqrt 3
def ellipse (x y : ℝ) : Prop := (x^2) / 3 + (y^2) / 2 = 1
def leftFocus (x y : ℝ) : Prop := x = -sqrt 3 ∧ y = 0
def distance (p1 p2 : ℝ × ℝ) : ℝ := sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def distanceToDirectrix (x0 : ℝ) (a : ℝ) : ℝ := 
  let c := sqrt (a^2 - (a / sqrt 3)^2) -- Using eccentricity e = sqrt(3)/3
  in a^2 / c - x0

-- Given conditions
def P (x0 y0 : ℝ) : Prop := ellipse x0 y0 ∧ distance (x0, y0) (-sqrt 3, 0) = sqrt 3 / 2

-- Prove the distance to the directrix is 9/2
theorem distance_from_P_to_directrix (x0 y0 : ℝ) (hP: P x0 y0) : 
  distanceToDirectrix (-3/2) semiMajorAxis = 9 / 2 :=
sorry

end distance_from_P_to_directrix_l461_461749


namespace min_value_expr_l461_461321

theorem min_value_expr (m n : ℝ) (h : m - n^2 = 8) : m^2 - 3 * n^2 + m - 14 ≥ 58 :=
sorry

end min_value_expr_l461_461321


namespace quadrilateral_area_eq_l461_461929

variables {A B C D K L M N P Q : Type} [points A B C D K L M N P Q]
variable {s : ℝ} -- area of parallelogram ABCD

-- Define the midpoints
def is_midpoint (p1 p2 p : Type) [points p1 p2 p] : Prop :=
  midpoint p1 p2 = p

-- Define the given conditions
axiom is_parallelogram : Parallelogram A B C D
axiom area_ABCD : area (Parallelogram A B C D) = s
axiom midpoint_K : is_midpoint A B K
axiom midpoint_L : is_midpoint B C L
axiom midpoint_M : is_midpoint C D M
axiom midpoint_N : is_midpoint A D N

-- Define the intersection points of the medians
axiom intersect_AL_AM_CK_CN (med1 med2 med3 med4 : Type) [lines med1 med2 med3 med4] :
  intersect (line A L) (line M A) (line C K) (line N C) = {P, Q}

-- The area of the quadrilateral formed by the intersection of the lines
theorem quadrilateral_area_eq :
  area (Quadrilateral P A Q C) = (1 / 3) * s :=
sorry

end quadrilateral_area_eq_l461_461929


namespace saree_blue_stripes_l461_461956

theorem saree_blue_stripes (brown_stripes gold_stripes blue_stripes : ℕ) 
    (h1 : brown_stripes = 4)
    (h2 : gold_stripes = 3 * brown_stripes)
    (h3 : blue_stripes = 5 * gold_stripes) : 
    blue_stripes = 60 := 
by
  sorry

end saree_blue_stripes_l461_461956


namespace rebecca_marbles_unknown_l461_461097

theorem rebecca_marbles_unknown (eggs marbles groups : ℕ) (h1 : eggs = 16) (h2 : ∀ x, x ∈ (range groups) → 2) (h3 : groups = 8) :
  ∃ marbles, True :=
by
  sorry

end rebecca_marbles_unknown_l461_461097


namespace count_coprime_to_15_l461_461261

def coprime_to_15 (a : ℕ) : Prop := Nat.gcd 15 a = 1

theorem count_coprime_to_15 : 
  (Finset.filter coprime_to_15 (Finset.range 15)).card = 8 := by
  sorry

end count_coprime_to_15_l461_461261


namespace min_symmetric_difference_l461_461047

variable (n : ℕ) (hn : n > 0)
def S := {x : ℕ | 1 ≤ x ∧ x ≤ n}
def SymDifference (X Y : Set ℕ) : Set ℕ := (X ∪ Y) \ (X ∩ Y)

theorem min_symmetric_difference (A B : Set ℕ) (hA : A.nonempty) (hB : B.nonempty)
  (C := {c | ∃ a b, a ∈ A ∧ b ∈ B ∧ c = a + b}) :
  |(SymDifference A S) ∪ (SymDifference B S) ∪ (SymDifference C S)| = n + 1 :=
sorry

end min_symmetric_difference_l461_461047


namespace range_of_a_l461_461813

open Real

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x ≤ y → f(x) ≥ f(y)) → a ≤ -1 :=
begin
  sorry
end

def f (x : ℝ) (a : ℝ) := x^2 + 2*(a - 1)*x + 2

end range_of_a_l461_461813


namespace problem_statement_l461_461329

theorem problem_statement
  (l : ∀ k : ℝ, ℝ × ℝ → Prop)
  (C : ℝ × ℝ → Prop)
  (fixed_pt : ℝ × ℝ)
  (center_C : ℝ × ℝ)
  (min_chord_AB : ℝ)
  (max_chord_AB : ℝ) :
  (∀ k : ℝ, l k (1, 1)) ∧
  center_C = (0, 2) ∧
  min_chord_AB = 2 * Real.sqrt 2 ∧
  max_chord_AB = 4 := by
  -- Definitions based on the conditions
  let l := λ k (p : ℝ × ℝ), k * p.1 - p.2 - k + 1 = 0
  let C := λ p : ℝ × ℝ, p.1^2 + p.2^2 - 4 * p.2 = 0
  let fixed_pt := (1, 1)
  let center_C := (0, 2)
  let min_chord_AB := 2 * Real.sqrt 2
  let max_chord_AB := 4

  sorry

end problem_statement_l461_461329


namespace line_parallel_perpendicular_l461_461587

variables (m n : line) (α : plane)

axiom parallel_lines : m.parallel n
axiom perpendicular_line_plane : m.perpendicular α

theorem line_parallel_perpendicular (m.parallel n) (m.perpendicular α) : n.perpendicular α :=
sorry

end line_parallel_perpendicular_l461_461587


namespace find_angle_QST_l461_461420

-- Conditions and definitions setup
variables {P Q R T S : Type}
variables [linear_order P] [linear_order Q] [linear_order R] [linear_order T] [linear_order S]
variables (angle_PQS angle_QRT angle_QTR angle_x : ℝ)

axiom h1 : angle_PQS = 110
axiom h2 : angle_QRT = 50
axiom h3 : angle_QTR = 40
axiom h4 : P + Q + R = 180 -- Line PQR is straight line
axiom h5 : T < PQR -- Point T is below PQR

-- Goal statement to prove that x = 70 given the above conditions
theorem find_angle_QST (angle_PQS angle_QRT angle_QTR : ℝ) (angle_x : ℝ) 
  (h1 : angle_PQS = 110) 
  (h2 : angle_QRT = 50) 
  (h3 : angle_QTR = 40) 
  (h4 : (P + Q + R) = 180) 
  (h5 : T < (PQR)) :
  angle_x = 70 :=
sorry

end find_angle_QST_l461_461420


namespace range_of_c_monotonicity_g_l461_461357

-- Define the given function f(x)
def f (x : ℝ) := 2 * real.log x + 1

-- Part 1: Define the hypothesis for the range of c
theorem range_of_c :
  ∀ x : ℝ, f(x) ≤ 2 * x + c ↔ c ∈ set.Ici (-1) :=
sorry

-- Part 2: Define the function g(x) and prove its monotonicity
def g (x a : ℝ) [ne_zero : a ≠ 0] := (f(x) - f(a)) / (x - a)

theorem monotonicity_g (a : ℝ) (h : 0 < a) : 
  ∀ x, (0 < x ∧ x < a) ∨ (x > a) → (g x a).deriv < 0 :=
sorry

end range_of_c_monotonicity_g_l461_461357


namespace part_b_part_c_l461_461179

-- Statement for part b: In how many ways can the figure be properly filled with the numbers from 1 to 5?
def proper_fill_count_1_to_5 : Nat :=
  8

-- Statement for part c: In how many ways can the figure be properly filled with the numbers from 1 to 7?
def proper_fill_count_1_to_7 : Nat :=
  48

theorem part_b :
  proper_fill_count_1_to_5 = 8 :=
sorry

theorem part_c :
  proper_fill_count_1_to_7 = 48 :=
sorry

end part_b_part_c_l461_461179


namespace ratio_of_smallest_to_middle_l461_461875

variable (RickAge OldestBrotherAge MiddleBrotherAge SmallestBrotherAge YoungestBrotherAge : ℕ)

def RickAge := 15
def OldestBrotherAge := 2 * RickAge
def MiddleBrotherAge := OldestBrotherAge / 3
def YoungestBrotherAge := 3
def SmallestBrotherAge := YoungestBrotherAge + 2

theorem ratio_of_smallest_to_middle :
  SmallestBrotherAge / MiddleBrotherAge = 1 / 2 :=
by
  rw [SmallestBrotherAge, MiddleBrotherAge, OldestBrotherAge, RickAge]
  sorry

end ratio_of_smallest_to_middle_l461_461875


namespace glasses_per_pitcher_l461_461943

theorem glasses_per_pitcher (t p g : ℕ) (ht : t = 54) (hp : p = 9) : g = t / p := by
  rw [ht, hp]
  norm_num
  sorry

end glasses_per_pitcher_l461_461943


namespace total_phones_in_Delaware_l461_461519

def population : ℕ := 974000
def phones_per_1000 : ℕ := 673

theorem total_phones_in_Delaware : (population / 1000) * phones_per_1000 = 655502 := by
  sorry

end total_phones_in_Delaware_l461_461519


namespace bugs_meeting_time_l461_461945

theorem bugs_meeting_time :
  let C1 := 14 * π
  let C2 := 6 * π
  let t1 := C1 / (4 * π)
  let t2 := C2 / (3.5 * π)
  LCM t1 t2 = 6 := 
by
  sorry

end bugs_meeting_time_l461_461945


namespace area_of_region_l461_461306

noncomputable def circle_radius : ℝ := 3

noncomputable def segment_length : ℝ := 4

theorem area_of_region : ∃ (area : ℝ), area = 4 * Real.pi :=
by
  sorry

end area_of_region_l461_461306


namespace permutation_divisible_by_37_l461_461608

/-- Theorem: Any permutation of a six-digit number divisible by 37 is also divisible by 37. -/
theorem permutation_divisible_by_37 (n m : ℕ) (hn : n % 37 = 0)
  (h_digits_n : ∀ i j : ℕ, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10)
  (h_perm : ∃ (a : list ℕ), (∀ (i : ℕ), a.nth i = some ((n / 10^i) % 10)) ∧ (m = a.foldr (λ d acc, acc * 10 + d) 0)) :
  m % 37 = 0 :=
sorry

end permutation_divisible_by_37_l461_461608


namespace option_A_option_C_option_D_l461_461834

variable {A B C a b c : ℝ}

theorem option_A 
  (h1: 3 * b * Real.cos C + 3 * c * Real.cos B = a^2)
  (h2: A + B + C = Real.pi) : 
  a = 3 := sorry

theorem option_C 
  (h1: B = Real.pi - A - C) 
  (h2: C = 2 * A) 
  (h3: 0 < A) (h4: A < Real.pi / 2) 
  (h5: 0 < B) (h6: B < Real.pi / 2)
  (h7: 0 < C) (h8: C < Real.pi / 2) :
  3 * Real.sqrt 2 < c ∧ c < 3 * Real.sqrt 3 :=
  sorry

theorem option_D 
  (h1: A = 2 * C) 
  (h2: Real.sin B = 2 * Real.sin C) 
  (h3: B = Real.pi - A - C) 
  (O : Type) 
  [is_incenter_triangle_O ABC] : 
  area (triangle AOB) = (3 * Real.sqrt 3 - 3) / 4 :=
  sorry

end option_A_option_C_option_D_l461_461834


namespace range_of_c_monotonicity_g_l461_461358

-- Define the given function f(x)
def f (x : ℝ) := 2 * real.log x + 1

-- Part 1: Define the hypothesis for the range of c
theorem range_of_c :
  ∀ x : ℝ, f(x) ≤ 2 * x + c ↔ c ∈ set.Ici (-1) :=
sorry

-- Part 2: Define the function g(x) and prove its monotonicity
def g (x a : ℝ) [ne_zero : a ≠ 0] := (f(x) - f(a)) / (x - a)

theorem monotonicity_g (a : ℝ) (h : 0 < a) : 
  ∀ x, (0 < x ∧ x < a) ∨ (x > a) → (g x a).deriv < 0 :=
sorry

end range_of_c_monotonicity_g_l461_461358


namespace two_zeros_implies_m_lt_neg_one_m_geq_min3_implies_ineq_holds_l461_461366

-- Definition of the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := Real.log x - x - m

-- Proof problem (1): If f(x) has two zeros, then m < -1
theorem two_zeros_implies_m_lt_neg_one (m : ℝ) (h_zeros : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 m = 0 ∧ f x2 m = 0) :
  m < -1 :=
  sorry

-- Proof problem (2): If m ≥ -3, the inequality f(x) + (x-2)e^x < 0 holds on [1/2, 1]
theorem m_geq_min3_implies_ineq_holds (m : ℝ) (h_m_geq : m ≥ -3)
  (x : ℝ) (h_x_interval : x ∈ Set.Icc (1 / 2) 1) :
  f x m + (x - 2) * Real.exp x < 0 :=
  sorry

end two_zeros_implies_m_lt_neg_one_m_geq_min3_implies_ineq_holds_l461_461366


namespace saree_blue_stripes_l461_461954

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    brown_stripes = 4 →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_gold h_blue h_brown
  sorry

end saree_blue_stripes_l461_461954


namespace geom_sequence_jump_condition_l461_461001

def is_jump_sequence {α : Type*} [LinearOrder α] (a : ℕ → α) : Prop :=
  ∀ i : ℕ, (a i - a (i + 2)) * (a (i + 2) - a (i + 1)) > 0

def is_geometric_sequence {α : Type*} [Mul α] [One α] (a : ℕ → α) (q : α) : Prop :=
  ∀ i : ℕ, a (i + 1) = a i * q

theorem geom_sequence_jump_condition {α : Type*} [LinearOrder α] [Mul α] [One α]
  {a : ℕ → α} {q : α} (h1 : is_geometric_sequence a q) (h2 : is_jump_sequence a) :
  q ∈ Set.Ioo (-1 : α) 0 :=
sorry

end geom_sequence_jump_condition_l461_461001


namespace frustum_lateral_edges_not_equal_l461_461167

noncomputable def frustum (P : Type) [polygon P] (b1 b2 : P) (lateral_faces : list (trapezoid P)) : Prop :=
∃ (pyramid : pyramid P) (cut_plane : plane), 
  pyramid.base = b1 ∧ 
  (∀ lateral_edge ∈ lateral_faces, ∃ plane_parallel : plane, is_parallelogram lateral_edge) ∧ 
  is_frustum cut_plane pyramid

theorem frustum_lateral_edges_not_equal (P : Type) [polygon P] (b1 b2 : P) (lateral_faces : list (trapezoid P)) :
  frustum P b1 b2 lateral_faces → ¬∀ (edge : lateral_faces), edge.length = all_equal ->
  false :=
begin
  intro h,
  cases h with pyramid cut_plane,
  sorry
end

end frustum_lateral_edges_not_equal_l461_461167


namespace tan_pi_over_four_minus_alpha_l461_461755

theorem tan_pi_over_four_minus_alpha (α : ℝ) 
  (h1 : α ∈ set.Ioo π (3 * π / 2)) 
  (h2 : cos α = -4 / 5) : 
  tan (π / 4 - α) = 1 / 7 := 
by 
  sorry

end tan_pi_over_four_minus_alpha_l461_461755


namespace find_S4_l461_461313

def sequence (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  (S 2 = 6) ∧ (∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * S n + 3)

theorem find_S4 :
  ∀ (a S : ℕ → ℕ),
    sequence a S →
    S 4 = 60 :=
by
  intros a S h_seq
  sorry

end find_S4_l461_461313


namespace two_marble_groups_count_l461_461145

def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

def total_combinations : ℕ :=
  let identical_yellow := 1
  let distinct_pairs := num_combinations 4 2
  identical_yellow + distinct_pairs

theorem two_marble_groups_count :
  total_combinations = 7 :=
by
  dsimp [total_combinations, num_combinations]
  rw [Nat.choose]
  norm_num
  sorry

end two_marble_groups_count_l461_461145


namespace coordinate_equation_solution_l461_461671

theorem coordinate_equation_solution (x y : ℝ) :
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 →
  (y = -x - 2) ∨ (y = -2 * x + 1) :=
by
  sorry

end coordinate_equation_solution_l461_461671


namespace find_n_values_l461_461763

theorem find_n_values (n : ℤ) (hn : ∃ x y : ℤ, x ≠ y ∧ x^2 - 6*x - 4*n^2 - 32*n = 0 ∧ y^2 - 6*y - 4*n^2 - 32*n = 0):
  n = 10 ∨ n = 0 ∨ n = -8 ∨ n = -18 := 
sorry

end find_n_values_l461_461763


namespace determine_m_l461_461310

noncomputable def power_function (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem determine_m (a : ℝ) (m : ℝ) :
  (power_function 2 a = (real.sqrt 2)/2) →
  (power_function m a = 2) →
  m = 1/4 :=
by
  sorry

end determine_m_l461_461310


namespace ages_of_Xs_sons_l461_461183

def ages_problem (x y : ℕ) : Prop :=
x ≠ y ∧ x ≤ 10 ∧ y ≤ 10 ∧
∀ u v : ℕ, u * v = x * y → u ≤ 10 ∧ v ≤ 10 → (u, v) = (x, y) ∨ (u, v) = (y, x) ∨
(∀ z w : ℕ, z / w = x / y → z = x ∧ w = y ∨ z = y ∧ w = x → u ≠ z ∧ v ≠ w) →
(∀ a b : ℕ, a - b = (x - y) ∨ b - a = (y - x) → (x, y) = (a, b) ∨ (x, y) = (b, a))

theorem ages_of_Xs_sons : ages_problem 8 2 := 
by {
  sorry
}


end ages_of_Xs_sons_l461_461183


namespace fib_100_mod_8_l461_461499

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 1     := 1
| 2     := 1
| (n+3) := (fib (n+1)) + (fib (n+2))

-- Define the problem statement
theorem fib_100_mod_8 : (fib 100) % 8 = 3 :=
by sorry

end fib_100_mod_8_l461_461499


namespace red_balls_count_l461_461517

theorem red_balls_count (w r : ℕ) (h1 : w = 16) (h2 : 4 * r = 3 * w) : r = 12 :=
by
  sorry

end red_balls_count_l461_461517


namespace circle_area_ratio_l461_461605

-- Setting up the definitions and conditions
def regular_hexagon (s : ℝ) : Prop := s = 2

def tangent_to_CD (circle_radius : ℝ) : Prop := 
  circle_radius = (2 * real.sqrt 3)/6

def area_of_circle (r : ℝ) : ℝ := 
  real.pi * r^2

-- Stating the theorem
theorem circle_area_ratio (r1 r2 : ℝ) (A1 A2 : ℝ) (h1 : regular_hexagon 2) (h2 : tangent_to_CD r1) 
  (h3 : tangent_to_CD r2) (h4 : A1 = area_of_circle r1) (h5 : A2 = area_of_circle r2) : 
  A1 / A2 = 1 :=
by
  sorry

end circle_area_ratio_l461_461605


namespace central_angle_of_dislike_is_correct_l461_461193

-- Define the ratios of the four satisfaction levels
def ratios : List ℕ := [6, 9, 2, 1]

-- The function to calculate the central angle for the dislike part
noncomputable def centralAngleDislikePart (r : List ℕ) : ℝ :=
  let total := r.foldl (· + ·) 0
  (1 / total.toReal) * 360

-- The theorem that states the central angle for the dislike part is 20 degrees
theorem central_angle_of_dislike_is_correct :
  centralAngleDislikePart ratios = 20 :=
by
  simp [centralAngleDislikePart, ratios]
  sorry

end central_angle_of_dislike_is_correct_l461_461193


namespace parallel_lines_slope_eq_l461_461922

theorem parallel_lines_slope_eq {a : ℝ} : (∀ x : ℝ, 2*x - 1 = a*x + 1) → a = 2 :=
by
  sorry

end parallel_lines_slope_eq_l461_461922


namespace first_year_exceeds_two_million_l461_461199

-- Definition of the initial R&D investment in 2015
def initial_investment : ℝ := 1.3

-- Definition of the annual growth rate
def growth_rate : ℝ := 1.12

-- Definition of the investment function for year n
def investment (n : ℕ) : ℝ := initial_investment * growth_rate ^ (n - 2015)

-- The problem statement to be proven
theorem first_year_exceeds_two_million : ∃ n : ℕ, n > 2015 ∧ investment n > 2 ∧ ∀ m : ℕ, (m < n ∧ m > 2015) → investment m ≤ 2 := by
  sorry

end first_year_exceeds_two_million_l461_461199


namespace planting_ways_l461_461204

def crop := {corn, wheat, soybeans, potatoes}

def is_valid_planting (planting: array (3 × 2) crop) : Prop :=
  -- Define the adjacency restrictions
  sorry

theorem planting_ways : 
  let ways := ∑ p in (finset.univ : finset (array (3 × 2) crop)), 
    if is_valid_planting p then 1 else 0 
  in ways = 148 :=
sorry

end planting_ways_l461_461204


namespace area_of_region_B_l461_461651

open Complex Real

def region_B : Set ℂ :=
  {z : ℂ | ∀ (C : ℂ), (abs (re (z / 30)) ≤ 1 ∧ abs (im (z / 30)) ≤ 1) ∧
                      (abs (re (30 / conj z)) ≤ 1 ∧ abs (im (30 / conj z)) ≤ 1)}

theorem area_of_region_B : measure_theory.measure_space.volume (region_B) = 675 - 112.5 * real.pi :=
  sorry

end area_of_region_B_l461_461651


namespace modulus_complex_number_l461_461129

variable (i : ℂ) (hz : i = Complex.i)

theorem modulus_complex_number (i := Complex.i) : 
  let z := (3 + 2 * i) * i in 
  Complex.abs z = Real.sqrt 13 := 
by
  sorry

end modulus_complex_number_l461_461129


namespace U_lies_on_g_l461_461048

-- Points A, B, C, D forming a convex cyclic quadrilateral with circumcenter U
variables {A B C D U : Type}
-- Conditions
variables [convex_cyclic_quadrilateral A B C D] [circumcenter U A B C D]
variables (is_perpendicular : ∀ {AC BD : Type}, AC.is_diagonal ∧ BD.is_diagonal ∧ AC ⊥ BD)
-- Reflection of diagonal AC across the angle bisector of angle BAD
noncomputable def g := reflect (diagonal AC) (angle_bisector (angle BAD))

-- Statement of the proof problem
theorem U_lies_on_g (h : point U ∈ line_of_reflection (diagonal AC) (angle_bisector (angle BAD))) : point U ∈ g :=
by
  sorry

end U_lies_on_g_l461_461048


namespace rosie_pies_l461_461890

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end rosie_pies_l461_461890


namespace Freddy_age_l461_461657

noncomputable def M : ℕ := 11
noncomputable def R : ℕ := M - 2
noncomputable def F : ℕ := M + 4

theorem Freddy_age : F = 15 :=
  by
    sorry

end Freddy_age_l461_461657


namespace triangle_geometry_problem_l461_461033

theorem triangle_geometry_problem
  (A B C D H E : Point)
  (h_angle_ACB : angle A C B = 45)
  (h_perpendicular_DH : is_perpendicular H D B C)
  (h_point_H : foot_of_perpendicular A B = H)
  (h_intersection_HD_AC : intersects_at HD AC E)
  (h_equal_AB_HD : dist A B = dist H D) :
  dist A E ^ 2 = 2 * dist D H ^ 2 + 2 * dist D E ^ 2 :=
sorry

end triangle_geometry_problem_l461_461033


namespace parabola_equation_with_left_focus_l461_461938

theorem parabola_equation_with_left_focus (x y : ℝ) :
  (∀ x y : ℝ, (x^2)/25 + (y^2)/9 = 1 → (y^2 = -16 * x)) :=
by
  sorry

end parabola_equation_with_left_focus_l461_461938


namespace solve_conjugate_l461_461732

noncomputable def problem_condition (z : ℂ) : Prop :=
  (2 * z + 3) * complex.I = 3 * z

theorem solve_conjugate {z : ℂ} (h : problem_condition z) : 
  complex.conj z = -6 / 13 - (9 / 13) * complex.I :=
sorry

end solve_conjugate_l461_461732


namespace isabella_babysits_afternoons_per_week_l461_461435

-- Defining the conditions of Isabella's babysitting job
def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 5
def days_per_week (weeks : ℕ) (total_earnings : ℕ) : ℕ := total_earnings / (weeks * (hourly_rate * hours_per_day))

-- Total earnings after 7 weeks
def total_earnings : ℕ := 1050
def weeks : ℕ := 7

-- State the theorem
theorem isabella_babysits_afternoons_per_week :
  days_per_week weeks total_earnings = 6 :=
by
  sorry

end isabella_babysits_afternoons_per_week_l461_461435


namespace probability_of_two_kings_or_atleast_two_aces_l461_461386

-- Definitions for the problem:
-- Standard deck has 52 cards
-- 4 aces and 4 kings in the deck

def probability_two_kings_or_atleast_two_aces : ℝ :=
  let ways_to_choose_3_cards := 52.choose 3
  let ways_to_choose_2_kings_1_non_king := (4.choose 2) * (50.choose 1)
  let ways_to_choose_2_aces_1_non_ace := (4.choose 2) * (48.choose 1)
  let ways_to_choose_3_aces := 4.choose 3
  let probability_2_kings := ways_to_choose_2_kings_1_non_king / ways_to_choose_3_cards
  let probability_2_aces := ways_to_choose_2_aces_1_non_ace / ways_to_choose_3_cards
  let probability_3_aces := ways_to_choose_3_aces / ways_to_choose_3_cards
  let probability_atleast_2_aces := probability_2_aces + probability_3_aces
  probability_2_kings + probability_atleast_2_aces

theorem probability_of_two_kings_or_atleast_two_aces :
  probability_two_kings_or_atleast_two_aces = 27/1007 :=
sorry

end probability_of_two_kings_or_atleast_two_aces_l461_461386


namespace num_odd_binomials_power_of_2_l461_461102

open Nat

theorem num_odd_binomials_power_of_2 (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, (number_of_odd_binomials n 0 n) = 2^k := by
  sorry

def number_of_odd_binomials (n : ℕ) (m_start m_end : ℕ) : ℕ :=
  ((m_start <= m_end) ∧ (m_start >= 0) ∧ (m_end <= n)) then count_odd ((range (m_start + 1)).map (λ m, choose n m)) else 0

def count_odd (lst : List ℕ) : ℕ :=
  lst.foldr (λ (x : ℕ) acc, if x % 2 = 1 then acc + 1 else acc) 0

end num_odd_binomials_power_of_2_l461_461102


namespace integral_split_integral_problem_l461_461236

theorem integral_split (a b : ℝ) (f g : ℝ → ℝ) (h₁ : ∫ x in a..b, f x = (π / 2)) (h₂ : ∫ x in a..b, g x = 0) :
  ∫ x in a..b, f x + g x = (π / 2) :=
by sorry

noncomputable def problem_statement : Prop :=
  ∀ (a b : ℝ) (f g : ℝ → ℝ),
  a = -1 ∧ b = 1 ∧
  (∀ x, f x = sqrt (1 - x^2)) ∧
  (∀ x, g x = sin x) →
  (∫ x in a..b, f x = (π / 2)) ∧
  (∫ x in a..b, g x = 0) →
  ∫ x in a..b, f x + g x = (π / 2)

theorem integral_problem : problem_statement :=
by sorry

end integral_split_integral_problem_l461_461236


namespace radius_Q_value_l461_461642

-- Define the problem conditions
def radius_P : ℝ := 4
def diameter_XY : ℝ := 2 * radius_P
def radius_R (r : ℝ) : Prop := 
  (4 - r)^2 = radius_P^2 - r^2 ∧ r ≠ 0

def radius_Q (q : ℝ) (r : ℝ) : Prop := 
  q = 4 * r

-- Define the main statement
theorem radius_Q_value : ∃ a b : ℕ, 
  ∃ r : ℝ, radius_R r ∧ radius_Q (sqrt (a : ℝ) - b) r ∧ a + b = 256 := 
  sorry

end radius_Q_value_l461_461642


namespace max_n_complex_numbers_l461_461907

theorem max_n_complex_numbers
  (n : ℕ)
  (z : ℕ → ℂ)
  (h : ∀ (j k : ℕ), j ≠ k → |z j - z k| ≥ max (finset.image (λ i, |z i|) (finset.range n))) :
  n ≤ 7 :=
sorry

end max_n_complex_numbers_l461_461907


namespace lambda_plus_mu_range_l461_461757

variables (A B C P G : Point)
variables (λ μ : ℝ)

-- Define conditions
def is_centroid (G A B C : Point) : Prop :=
  G = (1/3 : ℝ) • (A + B + C)

def is_interior_point_of_triangle (P G B C : Point) : Prop :=
  ∃ (α β γ : ℝ), 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = 1 ∧ P = α • G + β • B + γ • C

def equation_AP (A B C P : Point) (λ μ : ℝ) : Prop :=
  P = λ • (B - A) + μ • (C - A) + A

-- The main theorem
theorem lambda_plus_mu_range (A B C P G : Point) (λ μ : ℝ)
    (h_centroid : is_centroid G A B C)
    (h_interior : is_interior_point_of_triangle P G B C)
    (h_eq : equation_AP A B C P λ μ) :
    2/3 < λ + μ ∧ λ + μ < 1 :=
sorry

end lambda_plus_mu_range_l461_461757


namespace twice_midpoint_l461_461417

open Complex

def z1 : ℂ := -7 + 5 * I
def z2 : ℂ := 9 - 11 * I

theorem twice_midpoint : 2 * ((z1 + z2) / 2) = 2 - 6 * I := 
by
  -- Sorry is used to skip the proof
  sorry

end twice_midpoint_l461_461417


namespace pies_from_36_apples_l461_461880

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l461_461880


namespace inequality_properties_l461_461766

theorem inequality_properties (a b c : ℝ) (hac : a + b + c = 0) (hcb : c < b) (hba : b < a) :
  (cb^2 ≤ ab^2) ∧ (ab > ac) := sorry

end inequality_properties_l461_461766


namespace find_y_in_interval_l461_461286

theorem find_y_in_interval :
  { y : ℝ | y^2 + 7 * y < 12 } = { y : ℝ | -9 < y ∧ y < 2 } :=
sorry

end find_y_in_interval_l461_461286


namespace smallest_N_div_a3_possible_values_of_a3_l461_461452

-- Problem (a)
theorem smallest_N_div_a3 (a : Fin 10 → Nat) (h : StrictMono a) :
  Nat.lcm (a 0) (Nat.lcm (a 1) (Nat.lcm (a 2) (Nat.lcm (a 3) (Nat.lcm (a 4) (Nat.lcm (a 5) (Nat.lcm (a 6) (Nat.lcm (a 7) (Nat.lcm (a 8) (a 9))))))))) / (a 2) = 8 :=
sorry

-- Problem (b)
theorem possible_values_of_a3 (a : Nat) (h_a3_range : 1 ≤ a ∧ a ≤ 1000) :
  a = 315 ∨ a = 630 ∨ a = 945 :=
sorry

end smallest_N_div_a3_possible_values_of_a3_l461_461452


namespace first_digit_of_base16_representation_l461_461112

-- Firstly we define the base conversion from base 4 to base 10 and from base 10 to base 16.
-- For simplicity, we assume that the required functions exist and skip their implementations.

-- Assume base 4 to base 10 conversion function
def base4_to_base10 (n : String) : Nat :=
  sorry

-- Assume base 10 to base 16 conversion function that gives the first digit
def first_digit_base16 (n : Nat) : Nat :=
  sorry

-- Given the base 4 number as string
def y_base4 : String := "20313320132220312031"

-- Define the final statement
theorem first_digit_of_base16_representation :
  first_digit_base16 (base4_to_base10 y_base4) = 5 :=
by
  sorry

end first_digit_of_base16_representation_l461_461112


namespace count_coprime_to_15_eq_8_l461_461263

def is_coprime_to_15 (a : ℕ) : Prop := Nat.gcd a 15 = 1

def count_coprime_to_15 (n : ℕ) : ℕ :=
  (Finset.filter (λ a, is_coprime_to_15 a) (Finset.range n)).card

theorem count_coprime_to_15_eq_8 : count_coprime_to_15 15 = 8 := by
  sorry

end count_coprime_to_15_eq_8_l461_461263


namespace first_number_in_range_61_l461_461163

theorem first_number_in_range_61 :
  (∃ (n : ℕ), (n ≥ 1 ∧ n ≤ 70) ∧
  (∀ m ∈ 1..70, m % 10 = 1 ∨ m % 10 = 9 → n = 61)) :=
sorry

end first_number_in_range_61_l461_461163


namespace rosie_pies_proof_l461_461883

-- Define the given condition
def pies_per_apples (p: ℕ) (a: ℕ) : ℕ := a / p

-- Given that Rosie can make 3 pies from 12 apples
def given_condition : pies_per_apples 3 12 = 4 := rfl

-- The proof problem statement:
theorem rosie_pies_proof : ∀ (a: ℕ) (n: ℕ), pies_per_apples 3 12 = 4 → pies_per_apples n a = 4 → pies_per_apples n a = 9 :=
begin
  sorry
end

end rosie_pies_proof_l461_461883


namespace number_of_combinations_maximizing_sum_l461_461037

-- Define the conditions
def Box : Type := { n : Nat // n = 1 ∨ n = 2 }
def Draw : Type := (Box × Box × Box)

-- Prove the total number of combinations is 8
theorem number_of_combinations (d : Draw) : Finset.card (({1, 2} : Finset Nat).product (({1, 2} : Finset Nat).product {1, 2})) = 8 := by
  sorry
  
-- Prove that guessing 4 or 5 maximizes the chances of winning
theorem maximizing_sum (d : Draw) : 
  let sums := Finset.image (λ d : Draw, d.fst.1 + d.snd.fst.1 + d.snd.snd.1) (({1, 2} : Finset Nat).product (({1, 2} : Finset Nat).product {1, 2}))
  (4 ∈ sums ∧ 5 ∈ sums) := by
  sorry

end number_of_combinations_maximizing_sum_l461_461037


namespace election_vote_majority_l461_461409

noncomputable theory
def votes_polled : ℕ := 500
def winning_percentage : ℝ := 0.70
def losing_percentage : ℝ := 0.30

theorem election_vote_majority : 
  let winning_votes := winning_percentage * votes_polled
  let losing_votes := losing_percentage * votes_polled
  (winning_votes - losing_votes) = 200 :=
by
  sorry

end election_vote_majority_l461_461409


namespace complex_magnitude_l461_461733

-- We define the problem conditions and required proof as a theorem statement in Lean 4.
theorem complex_magnitude (z : ℂ) (h : z * (1 - 2 * complex.im) = 3 + 4 * complex.im) : complex.abs z = real.sqrt 5 :=
sorry

end complex_magnitude_l461_461733


namespace circles_tangent_ellipse_l461_461948

theorem circles_tangent_ellipse (r : ℝ) :
  (∃ x y : ℝ, (x - r)^2 + y^2 = r^2 ∧ x^2 + 4y^2 = 5) ∧ 
  (∃ x' y' : ℝ, (x' + r)^2 + y'^2 = r^2 ∧ x'^2 + 4y'^2 = 5) ∧ 
  (∃ cX cY : ℝ, cX = ±r ∧ cY = 0) ∧ 
  r > 0 ∧ 
  r = sqrt (15 / 16) :=
begin
  sorry,
end

end circles_tangent_ellipse_l461_461948


namespace floor_e_equals_2_l461_461693

theorem floor_e_equals_2 : Int.floor Real.exp = 2 := 
sorry

end floor_e_equals_2_l461_461693


namespace floor_e_eq_2_l461_461696

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_eq_2_l461_461696


namespace incorrect_expr_l461_461975

theorem incorrect_expr : 
  ∀ (a b c d : Prop) (h1 : a = 2) (h2 : b = 2) (h3 : c = 12) (h4 : d = -1), 
  c ≠ 12 :=
by sorry

end incorrect_expr_l461_461975


namespace coefficient_x6_expansion_l461_461963

theorem coefficient_x6_expansion : 
  (∀ x : ℝ, coefficient (expand (1 - 3 * x ^ 3) 4) x 6 = 54) := 
sorry

end coefficient_x6_expansion_l461_461963


namespace slope_angle_l461_461461

open Real

def is_on_parabola (x y : ℝ) : Prop := y^2 = 4 * x

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  ( (x2 - x1)^2 + (y2 - y1)^2 )^0.5

noncomputable def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  if x1 = x2 then 0 else (y2 - y1) / (x2 - x1)

theorem slope_angle (A F : ℝ × ℝ) (hA : ∃ t : ℝ, A = (t^2, 2 * t)) (hF : F = (1, 0)) 
  (hD : distance (fst A) (snd A) (fst F) (snd F) = 4) : 
  ∃ θ : ℝ, θ = π / 3 ∨ θ = 2 * π / 3 := 
sorry

end slope_angle_l461_461461


namespace problem1_problem2_has_solutions_l461_461181

-- First problem
theorem problem1 : (real.sqrt 2 - 1)^0 - (1/2)^(-1) + 2 * real.cos (real.pi / 3) = 0 :=
by
  sorry

-- Second problem
theorem problem2_has_solutions (x : ℝ) :
  (1/2) * x^2 + 3 * x - 1 = 0 ↔ x = -3 + real.sqrt 11 ∨ x = -3 - real.sqrt 11 :=
by
  sorry

end problem1_problem2_has_solutions_l461_461181


namespace max_value_of_b_l461_461711

noncomputable def max_b : ℕ :=
  let is_digit (n : ℕ) := n < 10 in
  let div_by_5 (n : ℕ) := (n % 5 = 0) in
  let div_by_11 (n : ℕ) := (n % 11 = 0) in
  ∃ a b c : ℕ,
  is_digit a ∧ is_digit b ∧ is_digit c ∧
  div_by_5 (a * 100000 + 2 * 10000 + b * 1000 + 3 * 100 + 4 * 10 + c) ∧
  div_by_11 (a * 100000 + 2 * 10000 + b * 1000 + 3 * 100 + 4 * 10 + c) ∧
  b = 7

theorem max_value_of_b : max_b = 7 := by
  sorry

end max_value_of_b_l461_461711


namespace area_ratio_GHI_JKL_l461_461152

-- Given conditions
def side_lengths_GHI : ℕ × ℕ × ℕ := (6, 8, 10)
def side_lengths_JKL : ℕ × ℕ × ℕ := (9, 12, 15)

-- Function to calculate the area of a right triangle given the lengths of the legs
def right_triangle_area (a b : ℕ) : ℕ :=
  (a * b) / 2

-- Function to determine if a triangle is a right triangle given its side lengths
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the main theorem
theorem area_ratio_GHI_JKL :
  let (a₁, b₁, c₁) := side_lengths_GHI
  let (a₂, b₂, c₂) := side_lengths_JKL
  is_right_triangle a₁ b₁ c₁ →
  is_right_triangle a₂ b₂ c₂ →
  right_triangle_area a₁ b₁ % right_triangle_area a₂ b₂ = 4 / 9 :=
by sorry

end area_ratio_GHI_JKL_l461_461152


namespace eccentricity_of_hyperbola_proof_l461_461368

noncomputable def hyperbola_eccentricity (a b c : ℝ) (A B F : ℝ × ℝ) :=
  (0 < a) → (0 < b) →
  (A = (c, b^2 / a)) →
  (B = (c, b * c / a)) →
  (F = (c, 0)) →
  (F.1 = c) →
  (F.2 = 0) →
  (2 * (A.2) = B.2) →
  (c^2 = 4 * b^2) →
  (c = 2 * sqrt 3 / 3 * a) →
  (e : ℝ) :=
  e = c / a

theorem eccentricity_of_hyperbola_proof (a b c : ℝ) (A B F : ℝ × ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : A = (c, b^2 / a)) 
  (h4 : B = (c, b * c / a)) 
  (h5 : F = (c, 0)) 
  (h6 : F.1 = c)
  (h7 : F.2 = 0)
  (h8 : 2 * A.2 = B.2)
  (h9 : c^2 = 4 * b^2)
  (h10 : c = 2 * sqrt 3 / 3 * a) : hyperbola_eccentricity a b c A B F (2 * sqrt 3 / 3) :=
sorry

end eccentricity_of_hyperbola_proof_l461_461368


namespace difference_even_odd_matchings_l461_461995

open Nat

def even_matchings (N : ℕ) (points: Finset (Fin (2 * N))) : Nat := sorry
def odd_matchings (N : ℕ) (points: Finset (Fin (2 * N))) : Nat := sorry

theorem difference_even_odd_matchings (N : ℕ) (points: Finset (Fin (2 * N))) :
  |even_matchings N points - odd_matchings N points| = 1 := sorry

end difference_even_odd_matchings_l461_461995


namespace saree_blue_stripes_l461_461957

theorem saree_blue_stripes (brown_stripes gold_stripes blue_stripes : ℕ) 
    (h1 : brown_stripes = 4)
    (h2 : gold_stripes = 3 * brown_stripes)
    (h3 : blue_stripes = 5 * gold_stripes) : 
    blue_stripes = 60 := 
by
  sorry

end saree_blue_stripes_l461_461957


namespace quadratic_nonnegative_quadratic_inv_nonnegative_l461_461721

-- Problem Definitions and Proof Statements

variables {R : Type*} [LinearOrderedField R]

def f (a b c x : R) : R := a * x^2 + 2 * b * x + c

theorem quadratic_nonnegative {a b c : R} (ha : a ≠ 0) (h : ∀ x : R, f a b c x ≥ 0) : 
  a ≥ 0 ∧ c ≥ 0 ∧ a * c - b^2 ≥ 0 :=
sorry

theorem quadratic_inv_nonnegative {a b c : R} (ha : a ≥ 0) (hc : c ≥ 0) (hac : a * c - b^2 ≥ 0) :
  ∀ x : R, f a b c x ≥ 0 :=
sorry

end quadratic_nonnegative_quadratic_inv_nonnegative_l461_461721


namespace line_parallel_to_plane_no_intersection_l461_461389

variables {α : Type*} [Plane α] {a : Line} 

def is_parallel (l : Line) (p : Plane) : Prop :=
  ∀ (x : Point), x ∈ p → x ∉ l

theorem line_parallel_to_plane_no_intersection (h : is_parallel a α) : 
  ∀ (b : Line), b ⊆ α → a ∩ b = ∅ := 
sorry

end line_parallel_to_plane_no_intersection_l461_461389


namespace range_of_c_monotonicity_of_g_l461_461350

noncomputable def f (x: ℝ) : ℝ := 2 * Real.log x + 1

theorem range_of_c (c: ℝ) : (∀ x > 0, f x ≤ 2 * x + c) → c ≥ -1 := by
  sorry

noncomputable def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

theorem monotonicity_of_g (a: ℝ) (ha: a > 0) : 
  (∀ x > 0, x ≠ a → ((x < a → g x a < g a a) ∧ (x > a → g x a < g a a))) := by
  sorry

end range_of_c_monotonicity_of_g_l461_461350


namespace determine_b_l461_461029

theorem determine_b (A B C : ℝ) (a b c : ℝ)
  (angle_C_eq_4A : C = 4 * A)
  (a_eq_30 : a = 30)
  (c_eq_48 : c = 48)
  (law_of_sines : ∀ x y, x / Real.sin A = y / Real.sin (4 * A))
  (cos_eq_solution : 4 * Real.cos A ^ 3 - 4 * Real.cos A = 8 / 5) :
  ∃ b : ℝ, b = 30 * (5 - 20 * (1 - Real.cos A ^ 2) + 16 * (1 - Real.cos A ^ 2) ^ 2) :=
by 
  sorry

end determine_b_l461_461029


namespace smallest_number_l461_461221

theorem smallest_number (a b c d : ℤ) (h1 : a = 1) (h2 : b = 0) (h3 : c = -1) (h4 : d = -3) :
  d = -3 ∧ d < c ∧ d < b ∧ d < a :=
by
  sorry

end smallest_number_l461_461221


namespace midpoint_of_line_segment_l461_461828

-- Definition of the endpoints of the line segment
def endpoint1 : ℂ := -5 + 7 * complex.I
def endpoint2 : ℂ := 9 - 3 * complex.I

-- Definition of the expected midpoint
def expected_midpoint : ℂ := 2 + 2 * complex.I

-- The theorem statement to prove 
theorem midpoint_of_line_segment : 
  (endpoint1 + endpoint2) / 2 = expected_midpoint :=
by
  sorry

end midpoint_of_line_segment_l461_461828


namespace values_of_a_and_b_l461_461391

theorem values_of_a_and_b (a b : ℕ) (h1 : a + b = 11) (h2 : 12 ∣ (1000 * a + 520 + b)) : (a = 7 ∧ b = 4) ∨ (a = 3 ∧ b = 8) :=
begin
  sorry -- proof goes here
end

end values_of_a_and_b_l461_461391


namespace reflect_E_length_l461_461151

def point := Prod ℤ ℤ

def reflect_y_axis (p : point) : point :=
  (-p.fst, p.snd)

def distance (p1 p2 : point) : ℤ :=
  Int.natAbs (p2.fst - p1.fst)

theorem reflect_E_length (E : point) : 
  E = (-2, -3) → distance E (reflect_y_axis E) = 4 :=
by
  intro hE
  rw [hE]
  rw [reflect_y_axis]
  dsimp
  norm_num

end reflect_E_length_l461_461151


namespace min_population_have_all_luxuries_l461_461578

theorem min_population_have_all_luxuries
  (total_population : ℕ) 
  (percent_refrigerators : ℝ)
  (percent_televisions : ℝ)
  (percent_computers : ℝ)
  (percent_air_conditioners : ℝ)
  (h_refrigerators : percent_refrigerators = 0.70)
  (h_televisions : percent_televisions = 0.75)
  (h_computers : percent_computers = 0.65)
  (h_air_conditioners : percent_air_conditioners = 0.95) :
  ∃ (min_people_all_luxuries : ℕ), min_people_all_luxuries ≥ 0.65 * total_population := 
sorry

end min_population_have_all_luxuries_l461_461578


namespace math_proof_equivalent_l461_461925

theorem math_proof_equivalent :
  (∀ x : ℝ, x^2 >= 0) ↔ (¬ ∃ x0 : ℝ, x0^2 < 0) ∧
  (∀ m : ℝ, (m > 1/2 → ∀ x : ℝ, (m*x^2 + 2*x + 2 ≠ 0 ∨ (-(4*m) < -1))) ∧ 
  (∀ x : ℝ, |x| ≠ 3 ↔ (x ≠ 3) ↔ (|x| ≠ 3 = false)) ∧
  (∀ A B : ℝ, (0 < A ∧ A < π / 2) → (0 < B ∧ B < π / 2) → (A + B > π / 2) → (π - (A + B) < A) → (cos B < sin A) → (sin A < tan A) → ((cos B < sin A ∧ sin A < tan A = true))

proof nat.succ 1 sorry

end math_proof_equivalent_l461_461925


namespace max_triangle_area_sum_l461_461489

noncomputable def segment_length := 2
def angle_value := 60
def max_area_sum := sqrt 3

theorem max_triangle_area_sum 
  (A A' B B' C C' O : Point) 
  (lengths_equal : segment_length = dist A A' ∧ segment_length = dist B B' ∧ segment_length = dist C C')
  (angles_equal : ∠ A O C' = angle_value ∧ ∠ B O A' = angle_value ∧ ∠ C O B' = angle_value)
  (intersect_at_O : Line.through A A' ∩ Line.through B B' ∩ Line.through C C' = {O}) :
  (area (triangle A O C') + area (triangle B O A') + area (triangle C O B')) ≤ max_area_sum :=
sorry

end max_triangle_area_sum_l461_461489


namespace rosie_can_make_nine_pies_l461_461894

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end rosie_can_make_nine_pies_l461_461894


namespace sum_of_lengths_of_intervals_l461_461370

noncomputable def f (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  (∑ i in finset.range n, a i / (x + a i))

theorem sum_of_lengths_of_intervals {n : ℕ} {a : ℕ → ℝ}
  (h_pos : ∀ i : ℕ, i < n → 0 < a i) 
  (h_increasing : ∀ i j : ℕ, i < j → i < n → j < n → a i < a j) : 
  let intervals := {x : ℝ | f a n x > 1}
  in intervals.to_finset.sum (λ x, x) = (∑ i in finset.range n, a i) :=
sorry

end sum_of_lengths_of_intervals_l461_461370


namespace trigonometric_identity_l461_461740

variable (α : Real)

noncomputable def point_on_terminal_side (α : Real) : Prop :=
  let x := -4
  let y := 3
  tan α = y / x

theorem trigonometric_identity 
  (h : point_on_terminal_side α) :
  (cos (π / 2 + α) * sin (-π - α)) /
  (cos (11 * π / 2 - α) * sin (9 * π / 2 + α)) = 
  -3 / 4 := 
  sorry

end trigonometric_identity_l461_461740


namespace ratio_A_B_share_l461_461215

-- Define the capital contributions and time in months
def A_capital : ℕ := 3500
def B_capital : ℕ := 15750
def A_months: ℕ := 12
def B_months: ℕ := 4

-- Effective capital contributions
def A_contribution : ℕ := A_capital * A_months
def B_contribution : ℕ := B_capital * B_months

-- Declare the theorem to prove the ratio 2:3
theorem ratio_A_B_share : A_contribution / 21000 = 2 ∧ B_contribution / 21000 = 3 :=
by
  -- Calculate and simplify the ratios
  have hA : A_contribution = 42000 := rfl
  have hB : B_contribution = 63000 := rfl
  have hGCD : Nat.gcd 42000 63000 = 21000 := rfl
  sorry

end ratio_A_B_share_l461_461215


namespace min_sum_PA_PB_l461_461750

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_sum_PA_PB :
  let A := (-1, 2), B := (2, 1) in
  ∃ P : ℝ × ℝ, P.1 = 1 ∧ P.2 = 0 ∧
    ∀ Q : ℝ × ℝ, Q.2 = 0 → 
      dist P A + dist P B ≤ dist Q A + dist Q B := 
by
  sorry

end min_sum_PA_PB_l461_461750


namespace train_length_is_correct_l461_461171

-- Definitions
def speed_kmh := 48.0 -- in km/hr
def time_sec := 9.0 -- in seconds

-- Conversion function
def convert_speed (s_kmh : Float) : Float :=
  s_kmh * 1000 / 3600

-- Function to calculate length of train
def length_of_train (speed_kmh : Float) (time_sec : Float) : Float :=
  let speed_ms := convert_speed speed_kmh
  speed_ms * time_sec

-- Proof problem: Given the speed of the train and the time it takes to cross a pole, prove the length of the train
theorem train_length_is_correct : length_of_train speed_kmh time_sec = 119.97 :=
by
  sorry

end train_length_is_correct_l461_461171


namespace area_of_square_is_324_l461_461825

-- Define the setup for the problem
variables (A B C D P Q R : Type) [MetricSpace A]
variables [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (square : IsSquare A B C D)
variables (P_on_AD : PointOnSegment P A D)
variables (Q_on_AB : PointOnSegment Q A B)
variables (intersect_at_right_angle : RightAngle (Segment B P) (Segment C Q) R)
variables (BR_length : length (Segment B R) = 9)
variables (PR_length : length (Segment P R) = 12)

-- The proof statement
theorem area_of_square_is_324 : 
  area_of_square A B C D = 324 :=
sorry

end area_of_square_is_324_l461_461825


namespace min_value_expr_l461_461661

-- Define the expression to be minimized
def expr (x : ℝ) : ℝ := (sin x)^8 + 16 * (cos x)^8 + 1
def denom (x : ℝ) : ℝ := (sin x)^6 + 4 * (cos x)^6 + 1

-- Define the function whose minimum we need to find
def func (x : ℝ) : ℝ := (expr x) / (denom x)

-- State the theorem in Lean 4
theorem min_value_expr : ∃ x : ℝ, func x = 4.7692 :=
by
  -- The proof goes here
  sorry

end min_value_expr_l461_461661


namespace max_sin_angle_APF_proof_l461_461747

noncomputable def max_sin_angle_APF (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (he : e = 1/5) : Prop :=
  let c := e * a in
  let A := (-a, 0) in
  let F := (-c, 0) in
  ∀ P : ℝ × ℝ, ∃ xq : ℝ × ℝ,
    P = (c, xq.snd) ∧
    (∀ θ : ℝ, θ = angle A P F →
      abs (sin θ) ≤ 1/2)

theorem max_sin_angle_APF_proof (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (he : e = 1/5) :
  max_sin_angle_APF a b h e he :=
sorry

end max_sin_angle_APF_proof_l461_461747


namespace equation_is_correct_l461_461087

-- Define the numbers
def n1 : ℕ := 2
def n2 : ℕ := 2
def n3 : ℕ := 11
def n4 : ℕ := 11

-- Define the mathematical expression and the target result
def expression : ℚ := (n1 + n2 / n3) * n4
def target_result : ℚ := 24

-- The proof statement
theorem equation_is_correct : expression = target_result := by
  sorry

end equation_is_correct_l461_461087


namespace avg_salary_all_workers_l461_461111

theorem avg_salary_all_workers :
  let A := 850 in
  let total_workers := 22 in
  let technicians := 7 in
  let non_technicians := total_workers - technicians in
  let avg_salary_technicians := 1000 in
  let avg_salary_non_technicians := 780 in
  (technicians * avg_salary_technicians + non_technicians * avg_salary_non_technicians) / total_workers = A :=
by
  sorry

end avg_salary_all_workers_l461_461111


namespace solve_trig_eq_l461_461982

theorem solve_trig_eq (x : ℝ) :
  (0.5 * (Real.cos (5 * x) + Real.cos (7 * x)) - Real.cos (2 * x) ^ 2 + Real.sin (3 * x) ^ 2 = 0) →
  (∃ k : ℤ, x = (Real.pi / 2) * (2 * k + 1) ∨ x = (2 * k * Real.pi / 11)) :=
sorry

end solve_trig_eq_l461_461982


namespace speed_of_faster_train_l461_461157

-- Constants given in the problem
def length_of_train : ℝ := 62.5
def speed_of_slower_train : ℝ := 36
def passing_time_in_seconds : ℝ := 45

-- Conditions translated to definitions
def relative_speed_m_per_s (v_f : ℝ) : ℝ := (v_f - speed_of_slower_train) * (5 / 18)
def total_distance : ℝ := 2 * length_of_train
def relative_speed_eq (v_f : ℝ) : Prop := relative_speed_m_per_s(v_f) = total_distance / passing_time_in_seconds

-- The statement we are proving
theorem speed_of_faster_train : ∃ v_f : ℝ, relative_speed_eq(v_f) ∧ v_f = 91.56 := 
sorry

end speed_of_faster_train_l461_461157


namespace trig_identity_l461_461384

theorem trig_identity (α : ℝ) (h : sin (π / 6 - α) = 1 / 3) : cos (π / 3 + α) = 1 / 3 :=
sorry

end trig_identity_l461_461384


namespace find_radius_of_S2_l461_461837

noncomputable def radius_of_S2 (r1 r3 : ℝ) : ℝ :=
  sqrt (r1 * r3)

theorem find_radius_of_S2 (r1 r3 : ℝ) (S2_radius : ℝ)
  (h1 : r1 = 1)
  (h2 : r3 = 9)
  (h3 : radius_of_S2 r1 r3 = S2_radius) :
  S2_radius = 3 := by
  sorry

end find_radius_of_S2_l461_461837


namespace min_objective_value_l461_461535

theorem min_objective_value (x y : ℝ) 
  (h1 : x + y ≥ 2) 
  (h2 : x - y ≤ 2) 
  (h3 : y ≥ 1) : ∃ (z : ℝ), z = x + 3 * y ∧ z = 4 :=
by
  -- Provided proof omitted
  sorry

end min_objective_value_l461_461535


namespace Paula_overall_score_l461_461086

theorem Paula_overall_score (s1 s2 s3 s4 q1 q2 q3 q4 : ℕ)
  (h1 : s1 = 9)    -- 75% of 12 questions
  (h2 : s2 = 17)   -- 85% of 20 questions
  (h3 : s3 = 24)   -- 80% of 30 questions
  (h4 : s4 = 6)    -- 60% of 10 questions
  (h5 : q1 = 12)
  (h6 : q2 = 20)
  (h7 : q3 = 30)
  (h8 : q4 = 10) :
  let total_correct := s1 + s2 + s3 + s4,
      total_questions := q1 + q2 + q3 + q4,
      overall_score := (total_correct : ℚ) / total_questions * 100 in
  overall_score.round = 78 := 
  by
  have total_correct : ℕ := s1 + s2 + s3 + s4,
  rw [h1, h2, h3, h4, Nat.add_assoc],
  norm_num at total_correct,
  have total_questions : ℕ := q1 + q2 + q3 + q4,
  rw [h5, h6, h7, h8, Nat.add_assoc],
  norm_num at total_questions,
  have overall_score : ℚ := (total_correct : ℚ) / total_questions * 100,
  suffices : overall_score = 77.7777, by
    rw [this],
    norm_num,
  have : (56 : ℚ) / 72 * 100 = 77.7777, by
    norm_num,
  exact this,
  sorry -- placeholder for missing proof steps

end Paula_overall_score_l461_461086


namespace numbers_in_ratio_l461_461540

theorem numbers_in_ratio (a b c : ℤ) :
  (∃ x : ℤ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x) ∧ (a * a + b * b + c * c = 725) →
  (a = 10 ∧ b = 15 ∧ c = 20 ∨ a = -10 ∧ b = -15 ∧ c = -20) :=
by
  sorry

end numbers_in_ratio_l461_461540


namespace lines_intersect_lines_perpendicular_lines_parallel_l461_461785

variables (l1 l2 : ℝ) (m : ℝ)

def intersect (m : ℝ) : Prop :=
  m ≠ -1 ∧ m ≠ 3

def perpendicular (m : ℝ) : Prop :=
  m = 1/2

def parallel (m : ℝ) : Prop :=
  m = -1

theorem lines_intersect (m : ℝ) : intersect m :=
by sorry

theorem lines_perpendicular (m : ℝ) : perpendicular m :=
by sorry

theorem lines_parallel (m : ℝ) : parallel m :=
by sorry

end lines_intersect_lines_perpendicular_lines_parallel_l461_461785


namespace matrix_inverse_is_zero_matrix_l461_461716

/- Define the matrix M -/
def M : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 10], ![-8, -20]]

/- Define the zero matrix -/
def zeroM : Matrix (Fin 2) (Fin 2) ℤ := 0

theorem matrix_inverse_is_zero_matrix :
  det M = 0 → (∀ N, inverse M = N → N = zeroM) :=
by
  intros h N inverse_def
  sorry

end matrix_inverse_is_zero_matrix_l461_461716


namespace floor_e_eq_two_l461_461706

theorem floor_e_eq_two
  (e_approx : Real ≈ 2.718) :
  ⌊e⌋ = 2 :=
sorry

end floor_e_eq_two_l461_461706


namespace equations_have_different_graphs_l461_461568

theorem equations_have_different_graphs :
  (∃ (x : ℝ), ∀ (y₁ y₂ y₃ : ℝ),
    (y₁ = x - 2) ∧
    (y₂ = (x^2 - 4) / (x + 2) ∧ x ≠ -2) ∧
    (y₃ = (x^2 - 4) / (x + 2) ∧ x ≠ -2 ∨ (x = -2 ∧ ∀ y₃ : ℝ, (x+2) * y₃ = x^2 - 4)))
  → (∃ y₁ y₂ y₃ : ℝ, y₁ ≠ y₂ ∨ y₁ ≠ y₃ ∨ y₂ ≠ y₃) := sorry

end equations_have_different_graphs_l461_461568


namespace epidemic_ends_l461_461400

noncomputable theory

def max_days := 101

structure Dwarf := 
(recovered : bool)
(infected : bool)
(immune : bool)

def all_dwarves_healthy (dwarves : List Dwarf) : Prop :=
  ∀ dwarf ∈ dwarves, ¬ dwarf.infected

def dwarves_transition (dwarves : List Dwarf) (day : ℕ) : List Dwarf :=
  -- Define how dwarves move from day to next day. This is just a placeholder.
  -- The real function would handle transitions based on the rules.
  dwarves -- This is just a placeholder!

theorem epidemic_ends
  (initial_dwarves : List Dwarf)
  (h_initial : length initial_dwarves = 100)
  (h_infection_rules : ∀ day < max_days, 
     let next_dwarves := dwarves_transition initial_dwarves day 
     in all_dwarves_healthy next_dwarves ∨ ¬ all_dwarves_healthy next_dwarves) :
  all_dwarves_healthy (dwarves_transition initial_dwarves max_days) :=
sorry

end epidemic_ends_l461_461400


namespace functional_equation_l461_461859

def f (x : ℝ) : ℝ := x + 1

theorem functional_equation (f : ℝ → ℝ) (h1 : f 0 = 1) (h2 : ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) :
  f = (λ x, x + 1) :=
by
  sorry

end functional_equation_l461_461859


namespace find_age_of_mother_l461_461382

def Grace_age := 60
def ratio_GM_Grace := 3 / 8
def ratio_GM_Mother := 2

theorem find_age_of_mother (G M GM : ℕ) (h1 : G = ratio_GM_Grace * GM) 
                           (h2 : GM = ratio_GM_Mother * M) (h3 : G = Grace_age) : 
  M = 80 :=
by
  sorry

end find_age_of_mother_l461_461382


namespace log_decreasing_interval_l461_461660

def quadratic_func (x : ℝ) : ℝ :=
  3 * x^2 - 7 * x + 2

def domain_cond1 : set ℝ :=
  { x | x < 1 / 3 }

def domain_cond2 : set ℝ :=
  { x | x > 2 }

theorem log_decreasing_interval :
  (∀ x ∈ domain_cond1, 3 * x^2 - 7 * x + 2 > 0) ∧
  (∀ x ∈ domain_cond2, 3 * x^2 - 7 * x + 2 > 0) →
  (∀ x ∈ domain_cond1, ∀ y ∈ domain_cond1, x < y → (log 2 (quadratic_func y)) < (log 2 (quadratic_func x))) :=
by
  sorry

end log_decreasing_interval_l461_461660


namespace price_difference_l461_461576

-- Define the conditions.
def initial_price (P : ℝ) := 1 / 0.85 * 68
def discounted_price := 68
def final_price := 68 * 1.25

-- Formulate the problem.
theorem price_difference :
  final_price - initial_price discounted_price = 5 :=
by
  calc
    final_price - initial_price discounted_price
        = 85 - 80 : by sorry
    ... = 5 : by sorry

end price_difference_l461_461576


namespace floor_e_eq_two_l461_461705

theorem floor_e_eq_two
  (e_approx : Real ≈ 2.718) :
  ⌊e⌋ = 2 :=
sorry

end floor_e_eq_two_l461_461705


namespace pyramid_volume_intersected_with_spheres_l461_461527

noncomputable def volume_of_solid (a : ℝ) : ℝ :=
  (81 - 4 * Real.pi) / 486 * a ^ 3

theorem pyramid_volume_intersected_with_spheres (a : ℝ) :
  volume_of_solid a = (81 - 4 * Real.pi) / 486 * a ^ 3 :=
by
  sorry

end pyramid_volume_intersected_with_spheres_l461_461527


namespace combined_gain_percent_l461_461100

/-- Given the purchase prices, repair costs, and selling prices of three scooters, 
    the combined gain percent is approximately 36.63%. -/
theorem combined_gain_percent :
  let scooterA_purchase := 900
  let scooterA_repairs := [150, 75, 225]
  let scooterA_selling := 1800

  let scooterB_purchase := 1200
  let scooterB_repairs := [200, 300]
  let scooterB_selling := 2400

  let scooterC_purchase := 1500
  let scooterC_repairs := [250, 100, 150]
  let scooterC_selling := 2700

  let total_cost := (scooterA_purchase + scooterA_repairs.sum) 
                 + (scooterB_purchase + scooterB_repairs.sum) 
                 + (scooterC_purchase + scooterC_repairs.sum)

  let total_selling := scooterA_selling + scooterB_selling + scooterC_selling

  let total_gain := total_selling - total_cost

  let gain_percent := (total_gain.toFloat / total_cost.toFloat) * 100

  abs (gain_percent - 36.63) < 0.01
:= by {
  sorry
}

end combined_gain_percent_l461_461100


namespace farmer_ear_count_l461_461203

theorem farmer_ear_count
    (seeds_per_ear : ℕ)
    (price_per_ear : ℝ)
    (cost_per_bag : ℝ)
    (seeds_per_bag : ℕ)
    (profit : ℝ)
    (target_profit : ℝ) :
  seeds_per_ear = 4 →
  price_per_ear = 0.1 →
  cost_per_bag = 0.5 →
  seeds_per_bag = 100 →
  target_profit = 40 →
  profit = price_per_ear - ((cost_per_bag / seeds_per_bag) * seeds_per_ear) →
  target_profit / profit = 500 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end farmer_ear_count_l461_461203


namespace floor_e_eq_2_l461_461701

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_eq_2_l461_461701


namespace positive_integers_between_300_and_1000_squared_l461_461796

theorem positive_integers_between_300_and_1000_squared :
  ∃ n : ℕ, 300 < n^2 ∧ n^2 < 1000 → ∃ m : ℕ, m = 14 := sorry

end positive_integers_between_300_and_1000_squared_l461_461796


namespace triangle_parallel_line_length_l461_461113

theorem triangle_parallel_line_length
  (base : ℝ)
  (area_ratios : list ℝ)
  (h_base : base = 18)
  (h_area_ratios : area_ratios = [1, 2, 1])
  (length_parallel_line : ℝ) :
  length_parallel_line = 9 * Real.sqrt 3 :=
sorry

end triangle_parallel_line_length_l461_461113


namespace area_B_correct_l461_461649

noncomputable def area_B : ℝ :=
  let circle_area_quarter := (π * (15 ^ 2)) / 4
  let square_area_quarter := 30 ^ 2 / 4
  2 * (square_area_quarter - circle_area_quarter) + square_area_quarter

theorem area_B_correct : area_B = 675 - 112.5 * π := by
  sorry

end area_B_correct_l461_461649


namespace s_eq_2CP_squared_l461_461462

-- Define the essential properties of the triangle and point conditions.
variables {a x : ℝ}

def hypotenuse : ℝ := 2 * a
def leg1 : ℝ := a
def leg2 : ℝ := a * sqrt 3
def CP_squared : ℝ := (3 * a^2) / 4

-- Define the function s based on point P's location.
def s (x : ℝ) (a : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ a then
    (x^2 + (a - x)^2)
  else
    ((x + a)^2 + x^2)

-- Define the condition that we need to prove.
theorem s_eq_2CP_squared :
  (∃ x, s x a = 2 * CP_squared) :=
sorry

end s_eq_2CP_squared_l461_461462


namespace bucket_problem_l461_461575

variable (A B C : ℝ)

theorem bucket_problem :
  (A - 6 = (1 / 3) * (B + 6)) →
  (B - 6 = (1 / 2) * (A + 6)) →
  (C - 8 = (1 / 2) * (A + 8)) →
  A = 13.2 :=
by
  sorry

end bucket_problem_l461_461575


namespace floor_e_eq_two_l461_461704

theorem floor_e_eq_two
  (e_approx : Real ≈ 2.718) :
  ⌊e⌋ = 2 :=
sorry

end floor_e_eq_two_l461_461704


namespace cost_of_bananas_l461_461229

theorem cost_of_bananas (A B : ℝ) (h1 : A + B = 5) (h2 : 2 * A + B = 7) : B = 3 :=
by
  sorry

end cost_of_bananas_l461_461229


namespace count_prime_sums_6_l461_461105

open BigOperators
open Nat

def is_prime_list (l : List ℕ) := l.filter Nat.Prime

noncomputable def generate_sequence : List ℕ :=
  let primes := nat.primes.map (λ x, Nat.succ x) -- to adjust for skipped primes
  (List.range 15).map (λ i, 
    let n := primes.take (i+1).sum -- take i+1 primes and compute the sum.
    if i % 2 = 1 then n - primes[i] else n 
  )

theorem count_prime_sums_6 : is_prime_list generate_sequence.length = 6 := by
  sorry

end count_prime_sums_6_l461_461105


namespace number_of_invertible_integers_mod_15_l461_461269

theorem number_of_invertible_integers_mod_15 :
  (finset.card {a ∈ finset.range 15 | Int.gcd a 15 = 1}) = 8 := by
  sorry

end number_of_invertible_integers_mod_15_l461_461269


namespace green_toads_per_acre_l461_461504

theorem green_toads_per_acre (brown_toads spotted_brown_toads green_toads : ℕ) 
  (h1 : ∀ g, 25 * g = brown_toads) 
  (h2 : spotted_brown_toads = brown_toads / 4) 
  (h3 : spotted_brown_toads = 50) : 
  green_toads = 8 :=
by
  sorry

end green_toads_per_acre_l461_461504


namespace number_of_people_l461_461870

/-- Prove that the number of people who went to the temple with Nathan is 3
given the cost per object and the total cost. -/
theorem number_of_people (cost_per_object : ℕ) (objects_per_person : ℕ) (total_cost : ℕ)
  (h1 : cost_per_object = 11)
  (h2 : objects_per_person = 5)
  (h3 : total_cost = 165) :
  ∃ P : ℕ, (objects_per_person * P * cost_per_object = total_cost) ∧ P = 3 :=
by
  use 3
  split
  · rw [h1, h2, h3]
    split
    done
  · sorry

end number_of_people_l461_461870


namespace area_of_rectangle_l461_461502

theorem area_of_rectangle (x y : ℝ)
    (h1 : x * y = (x + 3) * (y - 1))
    (h2 : x * y = (x - 4) * (y + 3 / 2)) :
    x * y = 108 := by
  sorry

end area_of_rectangle_l461_461502


namespace rosie_pies_proof_l461_461886

-- Define the given condition
def pies_per_apples (p: ℕ) (a: ℕ) : ℕ := a / p

-- Given that Rosie can make 3 pies from 12 apples
def given_condition : pies_per_apples 3 12 = 4 := rfl

-- The proof problem statement:
theorem rosie_pies_proof : ∀ (a: ℕ) (n: ℕ), pies_per_apples 3 12 = 4 → pies_per_apples n a = 4 → pies_per_apples n a = 9 :=
begin
  sorry
end

end rosie_pies_proof_l461_461886


namespace smallest_n_is_60_l461_461160

noncomputable def smallest_n : ℕ :=
  let n := 60
  in if n > 0 ∧ 24 ∣ n^2 ∧ 480 ∣ n^3 then n else 0

theorem smallest_n_is_60 :
  smallest_n = 60 := by
    sorry

end smallest_n_is_60_l461_461160


namespace even_decreasing_function_l461_461054

theorem even_decreasing_function (f : ℝ → ℝ) (x1 x2 : ℝ)
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y → x < 0 → y < 0 → f y < f x)
  (hx1_neg : x1 < 0)
  (hx1x2_pos : x1 + x2 > 0) :
  f x1 < f x2 :=
sorry

end even_decreasing_function_l461_461054


namespace domain_f_l461_461289

def h (x : ℝ) := arccos (x^2) + arctan x

theorem domain_f (x : ℝ) :
  -1 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ 1 ↔
  ∀ x, -1 ≤ x ∧ x ≤ 1 → h x ≠ π / 2 :=
sorry

end domain_f_l461_461289


namespace cost_of_douglas_fir_l461_461899

-- Given conditions
def total_trees : ℕ := 850
def douglas_fir_trees : ℕ := 350
def ponderosa_pine_trees : ℕ := total_trees - douglas_fir_trees
def cost_per_ponderosa_pine : ℕ := 225
def total_cost : ℕ := 217500

-- Definition of the problem parameter
def cost_per_douglas_fir (D : ℕ) : Prop :=
  350 * D + ponderosa_pine_trees * cost_per_ponderosa_pine = total_cost

-- Theorem statement
theorem cost_of_douglas_fir :
  ∃ D : ℕ, cost_per_douglas_fir D ∧ D = 300 :=
by
  use 300
  unfold cost_per_douglas_fir
  dsimp
  sorry

end cost_of_douglas_fir_l461_461899


namespace floor_e_eq_2_l461_461700

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_eq_2_l461_461700


namespace f_of_3_l461_461317

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 2 then log (x + 1) / log 2
else if h : x < 0 then -f (-x)
else -f (x - 4)

theorem f_of_3 (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f (x))
  (h_period : ∀ x, f (x - 4) = -f (x))
  (h_interval : ∀ x, 0 ≤ x ∧ x ≤ 2 → f (x) = log (x + 1) / log 2) :
  f 3 = 1 :=
by sorry

end f_of_3_l461_461317


namespace probability_accurate_forecast_l461_461949

theorem probability_accurate_forecast (p q : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : 0 ≤ q ∧ q ≤ 1) : 
  p * (1 - q) = p * (1 - q) :=
by {
  sorry
}

end probability_accurate_forecast_l461_461949


namespace exponent_of_5_in_30_factorial_l461_461430

theorem exponent_of_5_in_30_factorial : 
  ∑ (k : ℕ) in ({1, 2}).filter (λ (x : ℕ), x ≤ 2), (30 / 5^k).natAbs = 7 :=
by {
  sorry
}

end exponent_of_5_in_30_factorial_l461_461430


namespace bicycle_owners_no_car_l461_461010

-- Definitions based on the conditions in (a)
def total_adults : ℕ := 500
def bicycle_owners : ℕ := 450
def car_owners : ℕ := 120
def both_owners : ℕ := bicycle_owners + car_owners - total_adults

-- Proof problem statement
theorem bicycle_owners_no_car : (bicycle_owners - both_owners = 380) :=
by
  -- Placeholder proof
  sorry

end bicycle_owners_no_car_l461_461010


namespace legos_in_box_at_end_l461_461039

def initial_legos : ℕ := 500
def legos_used : ℕ := initial_legos / 2
def missing_legos : ℕ := 5
def remaining_legos := legos_used - missing_legos

theorem legos_in_box_at_end : remaining_legos = 245 := 
by
  sorry

end legos_in_box_at_end_l461_461039


namespace number_of_ordered_pairs_l461_461252

-- Definition of conditions in Lean
def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b
def equation (a b : ℕ) : Bool := 2 * a * b + 108 = 15 * lcm a b + 18 * Nat.gcd a b

-- The goal statement in Lean
theorem number_of_ordered_pairs : 
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ equation a b) → (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ equation a b ∧ ((a, b) = (36, 18) ∨ (a, b) = (18, 36))) := sorry

end number_of_ordered_pairs_l461_461252


namespace angle_C_exceeds_120_degrees_l461_461005

theorem angle_C_exceeds_120_degrees 
  (a b : ℝ) (h_a : a = Real.sqrt 3) (h_b : b = Real.sqrt 3) (c : ℝ) (h_c : c > 3) :
  ∀ (C : ℝ), C = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) 
             → C > 120 :=
by
  sorry

end angle_C_exceeds_120_degrees_l461_461005


namespace sum_of_reciprocals_of_factors_of_13_l461_461562

theorem sum_of_reciprocals_of_factors_of_13 : 
  (1 : ℚ) + (1 / 13) = 14 / 13 :=
by {
  sorry
}

end sum_of_reciprocals_of_factors_of_13_l461_461562


namespace price_increase_percentage_l461_461930

-- Define the initial parameters
variables (P Q : ℝ) -- Initial Price and Quantity

-- Conditions as given in the problem
variables (x : ℝ) -- Percentage increase in price
variables (R : ℝ) -- Initial Revenue = P * Q
variables (R' : ℝ) -- New Revenue = P * (1 + x / 100) * Q * 0.8

-- The effect on the revenue receipts is a 4% decrease
axiom revenue_decrease : R' = 0.96 * R

-- Original Revenue
def initial_revenue : ℝ := P * Q

-- New Revenue after price increase and quantity sold decrease
def new_revenue : ℝ := P * (1 + x / 100) * Q * 0.8

-- The Lean statement to prove that x = 20
theorem price_increase_percentage :
  ∀ (P Q : ℝ), R = initial_revenue P Q → R' = new_revenue P Q x → R' = 0.96 * R → x = 20 :=
by
  intros
  sorry

end price_increase_percentage_l461_461930


namespace rubies_in_chest_l461_461632

theorem rubies_in_chest (R : ℕ) (h₁ : 421 = R + 44) : R = 377 :=
by 
  sorry

end rubies_in_chest_l461_461632


namespace exist_N_for_prob_99_l461_461924

-- Define the main theorem
theorem exist_N_for_prob_99 : ∃ (N : ℕ), (let p : ℕ → ℕ → ℚ := 
                                          λ a N, if N = 0 then if a = 0 then 1 else 0 
                                                   else ((∑ i in finset.range (⌈2.01 * a⌉ + 1), p i (N - 1)) / (⌈2.01 * a⌉ + 1)))
                                              in p 2019 N ≥ 0.99 := by
    sorry

end exist_N_for_prob_99_l461_461924


namespace metal_detector_time_on_less_crowded_days_l461_461465

variable (find_parking_time walk_time crowded_metal_detector_time total_time_per_week : ℕ)
variable (week_days crowded_days less_crowded_days : ℕ)

theorem metal_detector_time_on_less_crowded_days
  (h1 : find_parking_time = 5)
  (h2 : walk_time = 3)
  (h3 : crowded_metal_detector_time = 30)
  (h4 : total_time_per_week = 130)
  (h5 : week_days = 5)
  (h6 : crowded_days = 2)
  (h7 : less_crowded_days = 3) :
  (total_time_per_week = (find_parking_time * week_days) + (walk_time * week_days) + (crowded_metal_detector_time * crowded_days) + (10 * less_crowded_days)) :=
sorry

end metal_detector_time_on_less_crowded_days_l461_461465


namespace other_number_is_31_l461_461078

namespace LucasProblem

-- Definitions of the integers a and b and the condition on their sum
variables (a b : ℤ)
axiom h_sum : 3 * a + 4 * b = 161
axiom h_one_is_17 : a = 17 ∨ b = 17

-- The theorem we need to prove
theorem other_number_is_31 (h_one_is_17 : a = 17 ∨ b = 17) : 
  (b = 17 → a = 31) ∧ (a = 17 → false) :=
by
  sorry

end LucasProblem

end other_number_is_31_l461_461078


namespace exists_polygon_with_n_triangulations_l461_461178

theorem exists_polygon_with_n_triangulations (n : ℕ) (h : n > 0) :
  ∃ (p : Type), is_polygon p ∧ triangulation_count p = n :=
sorry

end exists_polygon_with_n_triangulations_l461_461178


namespace find_angle_C_l461_461396

theorem find_angle_C (A B C : Type) [triangle A B C]
  (angle_B : B = 45) (AC_eq_sqrt2_BC : AC = sqrt 2 * BC) :
  C = 105 :=
sorry

end find_angle_C_l461_461396


namespace smallest_n_and_m_l461_461046

theorem smallest_n_and_m :
  ∃ n m : ℕ, 0 < n ∧ 
  m ∈ {i ∈ finset.range (n + 1) | i > 0} ∧ 
  (finset.sum (finset.filter (λ x, x ≠ m) (finset.range (n + 1))) * 13 = 439 * (n - 1)) ∧ 
  n = 68 ∧ 
  m = 45 := 
sorry

end smallest_n_and_m_l461_461046


namespace probability_two_males_chosen_l461_461403

open Nat

/-- 
In a competition, there are 7 finalists consisting of 4 females and 3 males.
If two finalists are chosen at random, what is the probability that both are male?
Express your answer as a common fraction.
-/
theorem probability_two_males_chosen (h : true) : 
  let total_finals := 7,
      total_females := 4,
      total_males := 3 in
  let ways_to_choose_two := choose 7 2,
      ways_to_choose_two_males := choose 3 2 in
  (ways_to_choose_two_males : ℚ) / (ways_to_choose_two : ℚ) = (1 / 7 : ℚ) := sorry

end probability_two_males_chosen_l461_461403


namespace num_elements_in_M_l461_461448

noncomputable def cos_nth_deriv (n : ℕ) : ℕ → ℝ → ℝ
| 0 := λ x, real.cos x
| 1 := λ x, -real.sin x
| 2 := λ x, -real.cos x
| 3 := λ x, real.sin x
| (n + 4) := cos_nth_deriv n

def M : set ℕ := { m | cos_nth_deriv m 1 = real.sin 1 ∧ m ≤ 2013 }

theorem num_elements_in_M : M.finite ∧ M.card = 503 :=
by
  sorry

end num_elements_in_M_l461_461448


namespace find_alpha_range_f_l461_461324

-- Condition 1: sin α tan α = 3/2
def cond1 (α : ℝ) : Prop := sin α * tan α = 3 / 2

-- Condition 2: 0 < α < π
def cond2 (α : ℝ) : Prop := 0 < α ∧ α < π

-- Definition of α
noncomputable def α : ℝ := π / 3

-- Function definition
def f (x α : ℝ) : ℝ := 4 * cos x * cos (x - α)

-- Lean 4 proof statement to find α
theorem find_alpha : ∃ α : ℝ, cond1 α ∧ cond2 α ∧ α = π / 3 := sorry

-- Lean 4 proof statement for range of function f
theorem range_f : ∃ S : Set ℝ, S = set.Icc 2 3 ∧ ∀ x, x ∈ set.Icc 0 (π / 4) → f x α ∈ S := sorry

end find_alpha_range_f_l461_461324


namespace T_n_lt_33_div_50_l461_461315

-- Define the sequence a_n
def a (n : ℕ) : ℚ :=
  if n = 1 then 3 / 2 else n

-- Define the b_n sequence using a_n
def b (n : ℕ) : ℚ :=
  1 / ((a n + 1) ^ 2)

-- Define the sum S_n for the sequence a_n
def S (n : ℕ) : ℚ :=
  (finset.range n).sum (λ i, a (i + 1))

-- Define the sum T_n for the sequence b_n
def T (n : ℕ) : ℚ :=
  (finset.range n).sum (λ i, b (i + 1))

-- Main theorem to prove
theorem T_n_lt_33_div_50 (n : ℕ) (hn : 0 < n) : 
  T n < 33 / 50 :=
sorry

end T_n_lt_33_div_50_l461_461315


namespace jelly_beans_in_jar_X_l461_461942

theorem jelly_beans_in_jar_X : 
  ∀ (X Y : ℕ), (X + Y = 1200) → (X = 3 * Y - 400) → X = 800 :=
by
  sorry

end jelly_beans_in_jar_X_l461_461942


namespace factors_and_pairs_of_ninety_l461_461555

open Finset

theorem factors_and_pairs_of_ninety :
  let n := 90 in
  let factors := {d ∈ Icc 1 n | n % d = 0} in
  let factors_lt_ten := {d ∈ factors | d < 10} in
  let num_pairs := {p ∈ factors.product factors | p.1 * p.2 = n}.card in
  (factors_lt_ten.card : ℚ) / (factors.card : ℚ) = 1 / 2 ∧ num_pairs = 6 :=
by
  sorry

end factors_and_pairs_of_ninety_l461_461555


namespace geometric_sequence_first_term_l461_461119

open Nat

theorem geometric_sequence_first_term : 
  ∃ (a r : ℝ), (a * r^3 = (6 : ℝ)!) ∧ (a * r^6 = (7 : ℝ)!) ∧ a = 720 / 7 :=
by
  sorry

end geometric_sequence_first_term_l461_461119


namespace real_solution_count_l461_461798

theorem real_solution_count (f : ℝ → ℝ) (h : ∀ x, f x = 4^(2*x+2) - 4^(x+3) - 4^x + 4) :
  (set.finite {x : ℝ | f x = 0} ∧ finset.card (set.to_finset {x : ℝ | f x = 0}) = 2) := 
sorry

end real_solution_count_l461_461798


namespace tan_alpha_l461_461729

theorem tan_alpha {α : ℝ} (h : Real.tan (α + π / 4) = 9) : Real.tan α = 4 / 5 :=
sorry

end tan_alpha_l461_461729


namespace dissimilar_terms_expansion_l461_461635

theorem dissimilar_terms_expansion (a b c d : ℤ) : 
  let n := 7
  let k := 4
  ∀ i j k l : ℕ, i + j + k + l = n →
  ∑ _ in (multiset.range (n + k - 1).pmap (λ x hx, x) (by exact set.to_finset (set.range (n + k - 1))), 1) = 120 :=
by 
  sorry

end dissimilar_terms_expansion_l461_461635


namespace problem_inequality_equality_case_l461_461480

theorem problem_inequality (n : ℕ) (h : 1 ≤ n) :
  \frac{1}{3} * (n:ℝ) ^ 2 + \frac{1}{2} * n + \frac{1}{6} ≥ (real.factorial n) ^ (2 / n) :=
sorry

theorem equality_case (n : ℕ) (h : 1 ≤ n) :
  (\frac{1}{3} * (n:ℝ) ^ 2 + \frac{1}{2} * n + \frac{1}{6} = (real.factorial n) ^ (2 / n)) ↔ (n = 1) :=
sorry

end problem_inequality_equality_case_l461_461480


namespace abs_m_plus_one_l461_461303

theorem abs_m_plus_one (m : ℝ) (h : |m| = m + 1) : (4 * m - 1) ^ 4 = 81 := by
  sorry

end abs_m_plus_one_l461_461303


namespace sequence_general_term_l461_461376

noncomputable def sequence (a b : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0
  else (an^2 + bn)

theorem sequence_general_term (a b : ℝ) (n : ℕ) (h : 0 < a ∧ 0 < b):
  sequence a b n = an^2 + bn := by sorry

end sequence_general_term_l461_461376


namespace mt_bisects_AMB_l461_461154

theorem mt_bisects_AMB (circle1 circle2 : Type)
  [metric_space circle1] [metric_space circle2]
  (M : circle1) (AB : set circle1) (T : set circle2) (A B: circle1) 
  (h1 : AB ⊆ circle1) (h2 : ∃ (U : circle2), M ∈ circle1 ∧ M ∈ circle2 ∧ AB ∈ larger_circle ∧ T ∈ smaller_circle) 
  (M_internal_touch : ∀ P : circle1, P ∈ (circle1 ∩ circle2) → P = M)
  (chord_touch : ∀ P : circle1, P ∈ AB → ∃ Q : circle2, Q = T)
  : ∃ MT : set circle1, is_angle_bisector (MT) (angle A M B) :=
by 
  sorry

end mt_bisects_AMB_l461_461154


namespace m_divisible_by_p_l461_461454

theorem m_divisible_by_p (p m n : ℕ) (hp : p > 2) (prime_p : Nat.Prime p) (hmn : m / n = 1 + (∑ i in Finset.range (p - 1), 1 / (i + 1))) : p ∣ m := 
by
  sorry

end m_divisible_by_p_l461_461454


namespace polynomial_identity_l461_461847

-- Define the polynomial expansion
def polynomialExpansion (x : ℝ) : ℝ :=
  (1 + 2 * x)^2 * (1 - x)^5

-- Conditions derived from the problem
def a : ℝ := polynomialExpansion 0 -- a = 1
def equation_at_neg1 : ℝ := polynomialExpansion (-1) -- a - a1 + a2 - a3 + a4 - a5 + a6 - a7 = 32

-- Translate the problem statement into Lean
theorem polynomial_identity :
  (let a1 := a1, a2 := a2, a3 := a3, a4 := a4, a5 := a5, a6 := a6, a7 := a7 in
  a1 - a2 + a3 - a4 + a5 - a6 + a7 = -31) :=
by
  -- Assuming the derivation steps can be skipped with a placeholder
  sorry

end polynomial_identity_l461_461847


namespace common_ratio_is_2_l461_461025

variable {a : ℕ → ℝ} 
variable {q : ℝ}

-- Conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n ∈ {1, 2, 3, 4, 5, 6}, a n > 0 ∧ a 4 = 4 ∧ a 6 = 16

-- Proving the common ratio q of the sequence a
theorem common_ratio_is_2 : geometric_sequence a q → q = 2 := by
  sorry

end common_ratio_is_2_l461_461025


namespace compute_series_l461_461643

theorem compute_series : ∀ (i : ℂ), i^2 = -1 → 2 * (∑ k in finset.range (604), i^k) = 2 := by
  intro i h
  sorry

end compute_series_l461_461643


namespace average_speed_trip_l461_461984

theorem average_speed_trip (d1 d2 : ℝ) (s1 s2 : ℝ) (h1 : d1 = 8) (h2 : s1 = 10) (h3 : d2 = 10) (h4 : s2 = 8) : 
  let t1 := d1 / s1 in
  let t2 := d2 / s2 in
  let total_distance := d1 + d2 in
  let total_time := t1 + t2 in
  (total_distance / total_time) ≈ 8.78 := 
by 
  sorry

end average_speed_trip_l461_461984


namespace find_angle_IJK_l461_461425

variables {α β : ℝ}

-- Conditions
def parallel_lines (AB CD : Prop) := AB → CD
def angle_CIJ := 10 * β
def angle_AGJ := 10 * α
def angle_CEJ := 6 * α
def angle_JFG := 6 * β

-- Question to prove
theorem find_angle_IJK (AB CD : Prop) (h1 : parallel_lines AB CD) (h2 : angle_CIJ = 10 * β) (h3 : angle_AGJ = 10 * α)
  (h4 : angle_CEJ = 6 * α) (h5 : angle_JFG = 6 * β) : 180 - (10 * (α + β)) = 120 :=
by
  sorry

end find_angle_IJK_l461_461425


namespace sequence_sum_correct_l461_461314

def a_n (n : ℕ) : ℤ := (-1)^(n+1) * (4*n - 3)

def S (n : ℕ) : ℤ := ∑ i in Finset.range (n + 1), a_n i

theorem sequence_sum_correct :
  S 15 + S 22 - S 31 = 48 := 
sorry

end sequence_sum_correct_l461_461314


namespace sqrt_three_irrational_among_l461_461225

theorem sqrt_three_irrational_among :
  (¬ ∃ a b : ℤ, b ≠ 0 ∧ sqrt 3 = a / b) ∧
  (∃ a b : ℤ, b ≠ 0 ∧ -1 = a / b) ∧
  (∃ a b : ℤ, b ≠ 0 ∧ 0 = a / b) ∧
  (∃ a b : ℤ, b ≠ 0 ∧ 1 / 2 = a / b) :=
by
  split
  · sorry -- Proof of irrationality of sqrt(3)
  split
  · use [-1, 1]
    split
    · exact ne_zero_of_pos one_pos
    · norm_num
  split
  · use [0, 1]
    split
    · exact one_ne_zero
    · norm_num
  · use [1, 2]
    split
    · exact two_ne_zero
    · norm_num

end sqrt_three_irrational_among_l461_461225


namespace lateral_surface_area_of_cube_l461_461988

-- Define the side length of the cube
def side_length : ℕ := 12

-- Define the area of one face of the cube
def area_of_one_face (s : ℕ) : ℕ := s * s

-- Define the lateral surface area of the cube
def lateral_surface_area (s : ℕ) : ℕ := 4 * (area_of_one_face s)

-- Prove the lateral surface area of a cube with side length 12 m is equal to 576 m²
theorem lateral_surface_area_of_cube : lateral_surface_area side_length = 576 := by
  sorry

end lateral_surface_area_of_cube_l461_461988


namespace village_school_absent_percentage_l461_461958

theorem village_school_absent_percentage 
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (absent_boys_fraction : ℚ)
  (absent_girls_fraction : ℚ)
  (absent_percentage : ℚ) :
  total_students = 150 → boys = 90 → girls = 60 → 
  absent_boys_fraction = 1 / 15 → absent_girls_fraction = 1 / 4 → 
  absent_percentage = 14 :=
by
  intros
  sorry

end village_school_absent_percentage_l461_461958


namespace tangent_line_at_neg1_l461_461116

-- Define the function given in the condition.
def f (x : ℝ) : ℝ := x^2 + 4 * x + 2

-- Define the point of tangency given in the condition.
def point_of_tangency : ℝ × ℝ := (-1, f (-1))

-- Define the derivative of the function.
def derivative_f (x : ℝ) : ℝ := 2 * x + 4

-- The proof statement: the equation of the tangent line at x = -1 is y = 2x + 1
theorem tangent_line_at_neg1 :
  ∃ (m b : ℝ), (∀ (x y : ℝ), y = f x → derivative_f (-1) = m ∧ point_of_tangency.fst = -1 ∧ y = m * (x + 1) + b) :=
sorry

end tangent_line_at_neg1_l461_461116


namespace solve_q_l461_461874

-- Definitions of conditions
variable (p q : ℝ)
variable (k : ℝ) 

-- Initial conditions
axiom h1 : p = 1500
axiom h2 : q = 0.5
axiom h3 : p * q = k
axiom h4 : k = 750

-- Goal
theorem solve_q (hp : p = 3000) : q = 0.250 :=
by
  -- The proof is omitted.
  sorry

end solve_q_l461_461874


namespace problem1_problem2_l461_461777

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x - 1

noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem problem1 (a : ℝ) (h1 : 2 / Real.exp 2 < a) (h2 : a < 1 / Real.exp 1) :
  ∃ (x1 x2 : ℝ), (0 < x1 ∧ x1 < 2) ∧ (0 < x2 ∧ x2 < 2) ∧ x1 ≠ x2 ∧ g x1 = a ∧ g x2 = a :=
sorry

theorem problem2 : ∀ x > 0, f x + 2 / (Real.exp 1 * g x) > 0 :=
sorry

end problem1_problem2_l461_461777


namespace rosie_pies_proof_l461_461885

-- Define the given condition
def pies_per_apples (p: ℕ) (a: ℕ) : ℕ := a / p

-- Given that Rosie can make 3 pies from 12 apples
def given_condition : pies_per_apples 3 12 = 4 := rfl

-- The proof problem statement:
theorem rosie_pies_proof : ∀ (a: ℕ) (n: ℕ), pies_per_apples 3 12 = 4 → pies_per_apples n a = 4 → pies_per_apples n a = 9 :=
begin
  sorry
end

end rosie_pies_proof_l461_461885


namespace rosie_pies_proof_l461_461882

-- Define the given condition
def pies_per_apples (p: ℕ) (a: ℕ) : ℕ := a / p

-- Given that Rosie can make 3 pies from 12 apples
def given_condition : pies_per_apples 3 12 = 4 := rfl

-- The proof problem statement:
theorem rosie_pies_proof : ∀ (a: ℕ) (n: ℕ), pies_per_apples 3 12 = 4 → pies_per_apples n a = 4 → pies_per_apples n a = 9 :=
begin
  sorry
end

end rosie_pies_proof_l461_461882


namespace sufficient_not_necessary_l461_461332

theorem sufficient_not_necessary (x : ℝ) : (x > 3 → x > 1) ∧ ¬ (x > 1 → x > 3) :=
by 
  sorry

end sufficient_not_necessary_l461_461332


namespace sqrt_expression_exists_l461_461585

noncomputable def verify_sqrt_expression (x y z : ℤ) : Prop :=
  x * x + y * y * z = 62 ∧ 2 * x * y * Int.sqrt z = 24 * Real.sqrt 11

theorem sqrt_expression_exists :
  ∃ (a b c : ℤ), verify_sqrt_expression a b c ∧ c % (n^2) ≠ 0 ∀ (n : ℕ), 1 < n ∧ n * n ∣ c ∧ (a + b + c = 19) :=
by
  sorry

end sqrt_expression_exists_l461_461585


namespace negation_of_existence_statement_l461_461782

open Real

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, exp x - x - 1 ≤ 0) ↔ ∀ x : ℝ, exp x - x - 1 > 0 :=
by
sory

end negation_of_existence_statement_l461_461782


namespace min_green_points_l461_461404

noncomputable def number_of_green_points (total_points : ℕ) (dist : ℝ) := 
∀ (total_points = 2020) (dist = 2020.0), ∃ g : ℕ, g = 45

theorem min_green_points :
  ∀ (total_points : ℕ) (dist : ℝ), 
  (∀ black_points green_points : ℕ, black_points + green_points = total_points ∧ 
  (∀ b, ∃ g1 g2: (ℕ × ℝ), dist = 2020.0)) → 
  ∃ g : ℕ, g = 45 :=
by
  intro _ _ h
  use 45
  finish

sorry

end min_green_points_l461_461404


namespace m_squared_n_minus_1_l461_461981

theorem m_squared_n_minus_1 (a b m n : ℝ)
  (h1 : a * m^2001 + b * n^2001 = 3)
  (h2 : a * m^2002 + b * n^2002 = 7)
  (h3 : a * m^2003 + b * n^2003 = 24)
  (h4 : a * m^2004 + b * n^2004 = 102) :
  m^2 * (n - 1) = 6 := by
  sorry

end m_squared_n_minus_1_l461_461981


namespace system1_solution_system2_solution_l461_461495

-- System 1
theorem system1_solution (x y : ℝ) 
  (h1 : y = 2 * x - 3)
  (h2 : 3 * x + 2 * y = 8) : 
  x = 2 ∧ y = 1 := 
by
  sorry

-- System 2
theorem system2_solution (x y : ℝ) 
  (h1 : x + 2 * y = 3)
  (h2 : 2 * x - 4 * y = -10) : 
  x = -1 ∧ y = 2 := 
by
  sorry

end system1_solution_system2_solution_l461_461495


namespace circumcircle_common_point_l461_461648

theorem circumcircle_common_point (p q x₁ x₂ : ℝ) (h₁ : x₁ * x₂ = q) : 
  ∃ D : ℝ × ℝ, D = (0, 1) ∧
  ∀ (A B C : ℝ × ℝ), (A = (x₁, 0)) → (B = (x₂, 0)) → (C = (0, q)) → 
    circle_through A B C D :=
by sorry

end circumcircle_common_point_l461_461648


namespace intersection_A_B_l461_461377

def A : set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
def B : set ℝ := { y | ∃ x, y = x^2 + 2 }

theorem intersection_A_B :
  A ∩ B = { x | 2 ≤ x ∧ x ≤ 3 } :=
sorry

end intersection_A_B_l461_461377


namespace pizza_slices_count_l461_461960

/-
  We ordered 21 pizzas. Each pizza has 8 slices. 
  Prove that the total number of slices of pizza is 168.
-/

theorem pizza_slices_count :
  (21 * 8) = 168 :=
by
  sorry

end pizza_slices_count_l461_461960


namespace geometric_sequence_first_term_l461_461118

noncomputable def first_term_of_geometric_sequence (a r : ℝ) : ℝ :=
  a

theorem geometric_sequence_first_term 
  (a r : ℝ)
  (h1 : a * r^3 = 720)   -- The fourth term is 6!
  (h2 : a * r^6 = 5040)  -- The seventh term is 7!
  : first_term_of_geometric_sequence a r = 720 / 7 :=
sorry

end geometric_sequence_first_term_l461_461118


namespace find_f_f_2_l461_461772

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then (1 / 2) ^ x else 1 / (x - 1)

theorem find_f_f_2 : f (f 2) = -4 / 3 :=
by
  -- Intended proof goes here
  sorry

end find_f_f_2_l461_461772


namespace laurent_series_expansion_l461_461708

noncomputable def f (z : ℂ) : ℂ := z^2 * (Real.cos (1 / z))

theorem laurent_series_expansion (z : ℂ) (hz : z ≠ 0) :
  f(z) = -1/2 + z^2 + 1/(4! * z^2) - 1/(6! * z^4) + ∑' (n : ℕ), if (n % 2 = 0) then (-1)^(n / 2) / (Real.fact (n) * z^n) else 0 :=
sorry

end laurent_series_expansion_l461_461708


namespace min_letters_l461_461125

/-- Defines a word in the language of the mumbo-jumbo tribe -/
structure Word :=
  (letters : List Char)
  (allowed_letters : ∀ c ∈ letters, c = 'A' ∨ c = 'O')

/-- The language consists of k different words -/
structure MumboJumboLanguage (k : ℕ) :=
  (words : Fin k → Word)
  (prefix_free : ∀ i j : Fin k, i ≠ j → ¬ (words i).letters.isPrefixOf (words j).letters)

/-- H is the length of the longest word in the language -/
def H : ℕ := \max i, ((MumboJumboLanguage.words i).letters.length)

/-- The minimum number of letters the complete vocabulary contains -/
theorem min_letters (k : ℕ) (L : MumboJumboLanguage k) :
  let H := \max (fun i, (L.words i).letters.length)
  in L.words.foldr (λ word sum, sum + word.letters.length) 0 = k * H + (2^H - k) :=
sorry

end min_letters_l461_461125


namespace sales_volume_A_correct_total_cost_B_correct_sales_volume_B_correct_max_total_profit_correct_possible_selling_prices_correct_l461_461436

section disinfectant_water

variables (x : ℕ) (x_greater_than_30 : x > 30)

-- Conditions
def cost_price_A := 20 -- yuan per bottle
def cost_price_B := 30 -- yuan per bottle
def total_cost := 2000 -- yuan

def initial_sales_A := 100 -- bottles at 30 yuan per bottle
def sell_decrease_A := 5 -- bottles per yuan increase
def sell_price_B := 60 -- yuan per bottle

-- Sales volume of type A disinfectant water
def sales_volume_A := 250 - 5 * x

-- Total cost price of type B disinfectant water
def total_cost_B := 2000 - 20 * (250 - 5 * x)

-- Sales volume of type B disinfectant water
def sales_volume_B := (total_cost_B x) / cost_price_B

-- Total profit function
def total_profit := (250 - 5 * x) * (x - 20) + ((total_cost_B x) / 30 - 100) * (60 - 30)

-- Maximum total profit
def max_total_profit := 2125

-- Possible selling prices for total profit >= 1945 yuan
def possible_selling_prices := {x : ℕ | 39 ≤ x ∧ x ≤ 50 ∧ (x % 3 = 0)}

-- Proofs
theorem sales_volume_A_correct : sales_volume_A x = 250 - 5 * x := by
  sorry

theorem total_cost_B_correct : total_cost_B x = 100 * x - 3000 := by
  sorry

theorem sales_volume_B_correct : sales_volume_B x = (100 * x - 3000) / 30 := by
  sorry

theorem max_total_profit_correct : total_profit x = -5*(x - 45) * (x - 45) + 2125 := by
  sorry

theorem possible_selling_prices_correct : ∀ x, 1945 ≤ total_profit x → x ∈ possible_selling_prices := by
  sorry

end disinfectant_water

end sales_volume_A_correct_total_cost_B_correct_sales_volume_B_correct_max_total_profit_correct_possible_selling_prices_correct_l461_461436


namespace traffic_light_probability_change_l461_461613

/-- 
A traffic light cycles in order: green for 45 seconds, yellow for 5 seconds, and red for 50 seconds.
Given that Cody picks a random five-second interval to observe the traffic light,
the probability that Cody sees a color change during his observation is \(3/20\).
-/
theorem traffic_light_probability_change :
  let total_time := 100 
  let green_time := 45 
  let yellow_time := 5 
  let red_time := 50 
  let observation_interval := 5 
  let change_interval := 15
  in (change_interval / total_time : ℚ) = 3 / 20 :=
by 
  let total_time := 100 
  let observation_interval := 5 
  let change_interval := 15 
  have h : (change_interval / total_time : ℚ) = 3 / 20, from sorry
  exact h

end traffic_light_probability_change_l461_461613


namespace pair_up_problem_l461_461139

noncomputable def pair_up_ways : ℕ := 18

theorem pair_up_problem (people : Fin 12 → Set (Fin 12))
  (condition1 : ∀ i : Fin 12, (# (people i) = 4))
  (condition2 : ∀ i : Fin 12, 1 ∈ people i ∧ 2 ∈ people i ∧ 7 ∈ people i ∧ 4 ∈ people i) :
  (∃ pairings : Set (Set (Fin 12)), (# pairings = 6 ∧ (∀ p ∈ pairings, (# p = 2 ∧ ∀ x ∈ p, people x ⊆ p)))) :=
by
  sorry

end pair_up_problem_l461_461139


namespace part1_part2_l461_461346

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.log x + 1

-- Part (1): Prove c ≥ -1 given f(x) ≤ 2x + c
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
  sorry

-- Define g with a > 0
def g (x a : ℝ) : ℝ := (f x - f a) / (x - a)

-- Part (2): Prove g is monotonically decreasing on (0, a) and (a, +∞)
theorem part2 (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, x > 0 → x ≠ a → g x a < g a a) :=
  sorry

end part1_part2_l461_461346


namespace complex_number_trace_ellipse_l461_461505

theorem complex_number_trace_ellipse (w : ℂ) (h : complex.abs w = 3) :
  ∃ a b : ℝ, ∀ z : ℂ, z = w + 1 / w → (real_part(z) / a)^2 + (imag_part(z) / b)^2 = 1 :=
sorry

end complex_number_trace_ellipse_l461_461505


namespace discount_percentage_l461_461607

noncomputable def cost_price : ℝ := 100
noncomputable def profit_with_discount : ℝ := 0.32 * cost_price
noncomputable def profit_without_discount : ℝ := 0.375 * cost_price

noncomputable def sp_with_discount : ℝ := cost_price + profit_with_discount
noncomputable def sp_without_discount : ℝ := cost_price + profit_without_discount

noncomputable def discount_amount : ℝ := sp_without_discount - sp_with_discount
noncomputable def percentage_discount : ℝ := (discount_amount / sp_without_discount) * 100

theorem discount_percentage : percentage_discount = 4 :=
by
  -- proof steps
  sorry

end discount_percentage_l461_461607


namespace sphere_surface_area_from_cube_l461_461810

theorem sphere_surface_area_from_cube (edge_length : ℝ) (h : edge_length = 2) : 
  let r := edge_length / 2
  in 4 * Real.pi * r^2 = 4 * Real.pi :=
by
  sorry

end sphere_surface_area_from_cube_l461_461810


namespace seating_arrangements_of_8_around_round_table_l461_461823

theorem seating_arrangements_of_8_around_round_table : 
  let n := 8 in (n.factorial / n) = 5040 :=
by
  sorry

end seating_arrangements_of_8_around_round_table_l461_461823


namespace cost_per_trip_l461_461475

theorem cost_per_trip (cost_per_pass : ℕ) (num_passes : ℕ) (trips_oldest : ℕ) (trips_youngest : ℕ) :
    cost_per_pass = 100 →
    num_passes = 2 →
    trips_oldest = 35 →
    trips_youngest = 15 →
    (cost_per_pass * num_passes) / (trips_oldest + trips_youngest) = 4 := by
  sorry

end cost_per_trip_l461_461475


namespace percentage_democrats_l461_461009

/-- In a certain city, some percent of the registered voters are Democrats and the rest are Republicans. In a mayoral race, 85 percent of the registered voters who are Democrats and 20 percent of the registered voters who are Republicans are expected to vote for candidate A. Candidate A is expected to get 59 percent of the registered voters' votes. Prove that 60 percent of the registered voters are Democrats. -/
theorem percentage_democrats (D R : ℝ) (h : D + R = 100) (h1 : 0.85 * D + 0.20 * R = 59) : 
  D = 60 :=
by
  sorry

end percentage_democrats_l461_461009


namespace no_int_solutions_5x2_minus_4y2_eq_2017_l461_461485

theorem no_int_solutions_5x2_minus_4y2_eq_2017 :
  ¬ ∃ x y : ℤ, 5 * x^2 - 4 * y^2 = 2017 :=
by
  -- The detailed proof goes here
  sorry

end no_int_solutions_5x2_minus_4y2_eq_2017_l461_461485


namespace odd_function_m_zero_l461_461805

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 + m

theorem odd_function_m_zero (m : ℝ) : (∀ x : ℝ, f (-x) m = -f x m) → m = 0 :=
by
  sorry

end odd_function_m_zero_l461_461805


namespace pies_from_36_apples_l461_461878

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l461_461878


namespace f_nonnegative_range_a_l461_461773

def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x - (1/2)

theorem f_nonnegative_range_a (a : ℝ) (h₀ : a ≠ 0) :
  (∀ x, 1 ≤ x → f x a ≥ 0) ↔ (a ∈ Set.Iio 0 ∪ Set.Ioc 0 1) :=
by
  sorry

end f_nonnegative_range_a_l461_461773


namespace simplify_fraction_l461_461170

variable (c : ℝ)

theorem simplify_fraction :
  (6 + 2 * c) / 7 + 3 = (27 + 2 * c) / 7 := 
by 
  sorry

end simplify_fraction_l461_461170


namespace least_number_of_faces_l461_461570

def faces_triangular_prism : ℕ := 5
def faces_quadrangular_prism : ℕ := 6
def faces_triangular_pyramid : ℕ := 4
def faces_quadrangular_pyramid : ℕ := 5
def faces_truncated_quadrangular_pyramid : ℕ := 6

theorem least_number_of_faces : faces_triangular_pyramid < faces_triangular_prism ∧
                                faces_triangular_pyramid < faces_quadrangular_prism ∧
                                faces_triangular_pyramid < faces_quadrangular_pyramid ∧
                                faces_triangular_pyramid < faces_truncated_quadrangular_pyramid 
                                :=
by {
  sorry
}

end least_number_of_faces_l461_461570


namespace product_of_solutions_abs_eq_l461_461423

theorem product_of_solutions_abs_eq (x : ℝ) :
  ( |x - 5| - 4 = -1 ) → ( x = 8 ∨ x = 2 ) ∧ ( 8 * 2 = 16 ) :=
by 
  intros,
  sorry

end product_of_solutions_abs_eq_l461_461423


namespace part_1_part_2_l461_461776

noncomputable
def f (x : ℝ) : ℝ := |2 * x - 1|

noncomputable
def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem part_1 : { x : ℝ | f x < 2 } = set.Ioo (-1 / 2) (3 / 2) :=
by sorry

theorem part_2 (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 2) : 
  (m^2 + 2) / m + (n^2 + 1) / n = (7 + 2 * real.sqrt 2) / 2 :=
by sorry

end part_1_part_2_l461_461776


namespace right_triangle_angles_l461_461028

/-- 
In a right triangle \(ABC\) with \(\angle ABC = \beta\),
Point \(E\) is the midpoint of \(BC\), 
A perpendicular \(EL\) is dropped from \(E\) to \(AB\). 
Given \(AE = \sqrt{10} \cdot EL\), \(BC > AC\):
Show that the angles of triangle \(ABC\) satisfy the given geometric constraints.
--/
theorem right_triangle_angles (A B C E L : ℝ) (β : ℝ) :
  (B ≠ 0) ∧ (C ≠ 0) ∧ (sin β ≠ 0) ∧ (cos β ≠ 0) ∧ 
  (E = (B + C) / 2) ∧ (AE = sqrt 10 * EL) ∧ (BC > AC) →
  ∃ α γ : ℝ, α + β + γ = π ∧ right_triangle A B C :=
sorry

end right_triangle_angles_l461_461028


namespace coefficient_of_x6_in_expansion_l461_461967

theorem coefficient_of_x6_in_expansion :
  let a := 1
  let b := -3 * (x : ℝ) ^ 3
  let n := 4
  let k := 2
  (1 - 3 * (x : ℝ) ^ 3) ^ 4 = ∑ k in finset.range (n + 1), 
    (nat.choose n k) * a ^ (n - k) * b ^ k →
  is_term_of_degree (1 - 3 * (x : ℝ) ^ 3) ^ 4 x 6 (54 * x ^ 6) :=
by
  sorry

end coefficient_of_x6_in_expansion_l461_461967


namespace allocation_schemes_l461_461076

theorem allocation_schemes :
  (choose 7 3) * (nat.factorial 3) + (choose 7 2) * (choose 3 2) * 2 = 336 :=
by
  -- Case 1: All three people in different labs
  have case1 : (choose 7 3) * (nat.factorial 3) = 210 := by
    rw [nat.choose_succ_succ, nat.factorial_succ, nat.factorial_succ, nat.factorial_one]
    simp
  -- Case 2: Two people in one lab, one in another
  have case2 : (choose 7 2) * (choose 3 2) * 2 = 126 := by
    rw [nat.choose_succ_succ, nat.choose_succ_succ, nat.factorial_succ, nat.factorial_one]
    simp
  -- Sum of the two cases
  calc
    (choose 7 3) * (nat.factorial 3) + (choose 7 2) * (choose 3 2) * 2
        = 210 + 126           : by rw [case1, case2]
    ... = 336                : by norm_num

end allocation_schemes_l461_461076


namespace min_value_f_exists_min_value_f_l461_461304

noncomputable def f (a b c : ℝ) := 1 / (b^2 + b * c) + 1 / (c^2 + c * a) + 1 / (a^2 + a * b)

theorem min_value_f (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) : f a b c ≥ 3 / 2 :=
  sorry

theorem exists_min_value_f : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧ f a b c = 3 / 2 :=
  sorry

end min_value_f_exists_min_value_f_l461_461304


namespace count_coprime_to_15_l461_461260

def coprime_to_15 (a : ℕ) : Prop := Nat.gcd 15 a = 1

theorem count_coprime_to_15 : 
  (Finset.filter coprime_to_15 (Finset.range 15)).card = 8 := by
  sorry

end count_coprime_to_15_l461_461260


namespace volume_ratio_l461_461014

-- Define a regular tetrahedron and necessary geometry
structure Tetrahedron (V : Type*) :=
  (vertices : fin 4 → V)
  (is_regular : ∀ i j k l : fin 4, i ≠ j → j ≠ k → k ≠ l → i ≠ l → 
    dist (vertices i) (vertices j) = 
    dist (vertices k) (vertices l))

noncomputable def center_of_face {V : Type*} [add_comm_group V] [vector_space ℝ V]
  (t : Tetrahedron V) (i : fin 4) : V :=
  (t.vertices ((i + 1) % 4) + t.vertices ((i + 2) % 4) + t.vertices ((i + 3) % 4)) / 3

noncomputable def smaller_tetrahedron {V : Type*} [add_comm_group V] [vector_space ℝ V]
  (t : Tetrahedron V) : Tetrahedron V :=
  { vertices := λ i, center_of_face t i,
    is_regular := sorry }

-- Given definitions
variable (r : ℝ)
variables (V : Type*) [inner_product_space ℝ V]
variable (large_tetrahedron : Tetrahedron V)
variable (hsphere : ∀ i, ∥large_tetrahedron.vertices i∥ = r)
variable (hregular : large_tetrahedron.is_regular)

-- Statement to prove
theorem volume_ratio (T' : Tetrahedron V) :
  T' = smaller_tetrahedron large_tetrahedron →
  volume T' / volume large_tetrahedron = 1 / 27 :=
sorry

end volume_ratio_l461_461014


namespace sin_C_in_right_triangle_l461_461824

theorem sin_C_in_right_triangle
  (A B C : ℝ)
  (sin_A : ℝ)
  (sin_B : ℝ)
  (B_right_angle : B = π / 2)
  (sin_A_value : sin_A = 3 / 5)
  (sin_B_value : sin_B = 1)
  (sin_of_C : ℝ)
  (tri_ABC : A + B + C = π ∧ A > 0 ∧ C > 0) :
    sin_of_C = 4 / 5 :=
by
  -- Skipping the proof
  sorry

end sin_C_in_right_triangle_l461_461824


namespace tan_half_sum_l461_461451

theorem tan_half_sum (p q : ℝ) (h1 : cos p + cos q = 1) (h2 : sin p + sin q = 1 / 2) : 
  tan ((p + q) / 2) = 1 / 2 :=
by
  sorry

end tan_half_sum_l461_461451


namespace central_angle_of_sector_l461_461330

theorem central_angle_of_sector (r : ℝ) (h : 3 * r = 2 * r + r) : 
  let l := r in
  let alpha := l / r in
  alpha = 1 :=
by sorry

end central_angle_of_sector_l461_461330


namespace imaginary_part_of_square_l461_461715

theorem imaginary_part_of_square (i : ℂ) (h : i^2 = -1) : 
  (complex.im ((1 - i)^2) = -2) :=
sorry

end imaginary_part_of_square_l461_461715


namespace Freddy_age_l461_461658

noncomputable def M : ℕ := 11
noncomputable def R : ℕ := M - 2
noncomputable def F : ℕ := M + 4

theorem Freddy_age : F = 15 :=
  by
    sorry

end Freddy_age_l461_461658


namespace number_of_elements_in_B_l461_461784

noncomputable def A : Set ℚ := {-1, -1/2, -1/3, 0, 2/3, 2, 3}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f_x (k : ℚ) : ℝ → ℝ := λ x, x^k

def B : Set (ℝ → ℝ) :=
  {f_x k | k ∈ A ∧ is_even_function (f_x k)}

theorem number_of_elements_in_B : B.card = 3 := sorry

end number_of_elements_in_B_l461_461784


namespace math_problem_l461_461138

theorem math_problem :
  (- (1 / 8)) ^ 2007 * (- 8) ^ 2008 = -8 :=
by
  sorry

end math_problem_l461_461138


namespace pens_sold_l461_461115

variable (C S : ℝ)
variable (n : ℕ)

-- Define conditions
def condition1 : Prop := 10 * C = n * S
def condition2 : Prop := S = 1.5 * C

-- Define the statement to be proved
theorem pens_sold (h1 : condition1 C S n) (h2 : condition2 C S) : n = 6 := by
  -- leave the proof steps to be filled in
  sorry

end pens_sold_l461_461115


namespace tips_earned_l461_461991

theorem tips_earned
  (total_customers : ℕ)
  (no_tip_customers : ℕ)
  (tip_amount : ℕ)
  (tip_customers := total_customers - no_tip_customers)
  (total_tips := tip_customers * tip_amount)
  (h1 : total_customers = 9)
  (h2 : no_tip_customers = 5)
  (h3 : tip_amount = 8) :
  total_tips = 32 := by
  -- Proof goes here
  sorry

end tips_earned_l461_461991


namespace find_y_l461_461534

theorem find_y (D : ℝ) (x y : ℝ) (h1 : x * y = D) (h2 : x + y = 30) (h3 : x = 7) :
  y = 200 / 7 :=
by
  have D_val : D = 200 := sorry
  have y_val : y = 200 / x := by sorry
  rw [h3] at y_val
  exact y_val
  sorry

end find_y_l461_461534


namespace minimum_roots_l461_461206

noncomputable def f : ℝ → ℝ := sorry

theorem minimum_roots (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (3 + x) = f (3 - x))
  (h2 : ∀ x : ℝ, f (8 + x) = f (8 - x))
  (h3 : f 0 = 0) :
  ∃ n ≥ 201, ∃ x : ℝ, (x ∈ Icc (-500) 500) ∧ f x = 0 :=
sorry

end minimum_roots_l461_461206


namespace total_white_papers_l461_461212

-- Define the given conditions
def papers_per_envelope : ℕ := 10
def number_of_envelopes : ℕ := 12

-- The theorem statement
theorem total_white_papers : (papers_per_envelope * number_of_envelopes) = 120 :=
by
  sorry

end total_white_papers_l461_461212


namespace geometric_sequence_common_ratio_l461_461026

theorem geometric_sequence_common_ratio (a : ℕ → ℕ) (q : ℕ) (h2 : a 2 = 8) (h5 : a 5 = 64)
  (h_geom : ∀ n, a (n+1) = a n * q) : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l461_461026


namespace ax_plus_by_equals_d_set_of_solutions_l461_461855

theorem ax_plus_by_equals_d (a b d : ℤ) (u v : ℤ) (h_d : d = a.gcd b) (h_uv : a * u + b * v = d) :
  ∀ (x y : ℤ), (a * x + b * y = d) ↔ ∃ k : ℤ, x = u + k * b ∧ y = v - k * a :=
by
  sorry

theorem set_of_solutions (a b d : ℤ) (u v : ℤ) (h_d : d = a.gcd b) (h_uv : a * u + b * v = d) :
  {p : ℤ × ℤ | a * p.1 + b * p.2 = d} = {p : ℤ × ℤ | ∃ k : ℤ, p = (u + k * b, v - k * a)} :=
by
  sorry

end ax_plus_by_equals_d_set_of_solutions_l461_461855


namespace triangle_area_correct_l461_461173

-- Definitions of sides a, b, c of the triangle
def a : ℝ := 26
def b : ℝ := 18
def c : ℝ := 10

-- The semi-perimeter s is defined as (a + b + c) / 2
def s : ℝ := (a + b + c) / 2

-- Heron's formula for the area of the triangle
def area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- The main theorem stating that the area equals √4131
theorem triangle_area_correct : area = Real.sqrt 4131 := 
by
  sorry

end triangle_area_correct_l461_461173


namespace optimal_approximation_value_l461_461681

variable {n : ℕ} (a : ℕ → ℝ )

theorem optimal_approximation_value (a_array : Fin n → ℝ) :
  let sum_squares (a : ℝ) := ∑ i, (a - a_array i) ^ 2
  in  ∃ (a : ℝ), (∀ a', sum_squares a ≤ sum_squares a') ∧ 
      a = (1 / n) * ∑ i, a_array i :=
sorry

end optimal_approximation_value_l461_461681


namespace petya_prevents_natural_sum_l461_461871

theorem petya_prevents_natural_sum (fracs : ℕ → ℕ → ℚ) (S : ℚ) 
  (h_fracs : ∀ n m, fracs n m = 1 / n) 
  (h_petya_first : True) 
  (h_petya_turn : ∀ t, Petya t ∈ ℕ → fracs (n+1) t ∈ ℚ) 
  (h_vasya_turn_inc : ∀ t, Vasya t ∈ ℕ → fracs (n + (t + 1)) t ∈ ℚ) :
  ∀ t, ∃ f, Petya t ∋ fracs f t → sum fracs t ∉ ℕ := 
sorry

end petya_prevents_natural_sum_l461_461871


namespace sum_of_remainders_l461_461565

theorem sum_of_remainders (n : ℤ) (h : n % 12 = 5) :
  (n % 4) + (n % 3) = 3 :=
by
  sorry

end sum_of_remainders_l461_461565


namespace problem_1_problem_2_l461_461006

noncomputable def given_problem (a b : ℝ) (C : ℝ) := 
  let cos_C := Real.cos (C * Real.pi / 180) in
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * cos_C) in
  let sin_C := Real.sin (C * Real.pi / 180) in
  let sin_B := b * sin_C / c in
  (c, sin_B) -- Return the values for c and sin_B

theorem problem_1 
  (h_a : ℝ := 3) 
  (h_b : ℝ := 4) 
  (h_C : ℝ := 60) :
  (given_problem h_a h_b h_C).fst = Real.sqrt 13 :=
sorry -- Proof skipped

theorem problem_2 
  (h_a : ℝ := 3) 
  (h_b : ℝ := 4) 
  (h_C : ℝ := 60) :
  (given_problem h_a h_b h_C).snd = 2 * Real.sqrt 39 / 13 :=
sorry -- Proof skipped

end problem_1_problem_2_l461_461006


namespace trig_expression_range_l461_461294

variable (C : ℝ)

-- Conditions
def is_internal_angle (C : ℝ) : Prop := C > 0 ∧ C < π
def sin_nonzero (C : ℝ) : Prop := sin C ≠ 0

-- Expression to find the range
def trig_expression (C : ℝ) : ℝ := (2 * cos (2 * C) / tan C + 1)

-- The proof problem statement
theorem trig_expression_range : (is_internal_angle C) → (sin_nonzero C) → -1 < trig_expression C ∧ trig_expression C ≤ sqrt 2 := by sorry

end trig_expression_range_l461_461294


namespace correlation_comparison_l461_461737

-- Definitions of the datasets
def data_XY : List (ℝ × ℝ) := [(10,1), (11.3,2), (11.8,3), (12.5,4), (13,5)]
def data_UV : List (ℝ × ℝ) := [(10,5), (11.3,4), (11.8,3), (12.5,2), (13,1)]

-- Definitions of the linear correlation coefficients
noncomputable def r1 : ℝ := sorry -- Calculation of correlation coefficient between X and Y
noncomputable def r2 : ℝ := sorry -- Calculation of correlation coefficient between U and V

-- The proof statement
theorem correlation_comparison :
  r2 < 0 ∧ 0 < r1 :=
sorry

end correlation_comparison_l461_461737


namespace closest_point_on_line_to_given_point_l461_461718

def vector := ℝ × ℝ × ℝ

def parametric_line (s : ℝ) : vector :=
(3 - s, -2 + 4 * s, 2 - 2 * s)

def distance_vector (p : vector) (q : vector) : vector :=
(p.1 - q.1, p.2 - q.2, p.3 - q.3)

def dot_product (v1 : vector) (v2 : vector) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def closest_point_to_line : vector :=
let s := 4 / 7 in
parametric_line s

theorem closest_point_on_line_to_given_point :
  closest_point_to_line = (17 / 7, 10 / 7, 6 / 7) :=
by
  sorry

end closest_point_on_line_to_given_point_l461_461718


namespace average_distinct_k_values_l461_461759

theorem average_distinct_k_values (k : ℕ) (h : ∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ r1 > 0 ∧ r2 > 0) : k = 15 :=
sorry

end average_distinct_k_values_l461_461759


namespace total_pets_count_l461_461498

/-- Taylor and his six friends have a total of 45 pets, given the specified conditions about the number of each type of pet they have. -/
theorem total_pets_count
  (Taylor_cats : ℕ := 4)
  (Friend1_pets : ℕ := 8 * 3)
  (Friend2_dogs : ℕ := 3)
  (Friend2_birds : ℕ := 1)
  (Friend3_dogs : ℕ := 5)
  (Friend3_cats : ℕ := 2)
  (Friend4_reptiles : ℕ := 2)
  (Friend4_birds : ℕ := 3)
  (Friend4_cats : ℕ := 1) :
  Taylor_cats + Friend1_pets + Friend2_dogs + Friend2_birds + Friend3_dogs + Friend3_cats + Friend4_reptiles + Friend4_birds + Friend4_cats = 45 :=
sorry

end total_pets_count_l461_461498


namespace quadrilateral_parallelogram_l461_461136

theorem quadrilateral_parallelogram (AB BC CD DA : ℝ) (A1 B1 C1 D1 : ℝ) (h1: A1 = B1) (h2: C1 = D1) (h3: A1 = C1 = D1) :
  (A1 + B1 + C1 + D1) = (AB + BC + CD + DA) / 2 → 
  parallelogram ABCD :=
  sorry

end quadrilateral_parallelogram_l461_461136


namespace max_sin_angle_APF_l461_461745

-- Definitions based on problem conditions
variables (a b c m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
variables (h4 : c = a / 5) (h5 : c = Real.sqrt (a^2 - b^2))

-- Coordinates of points
def A : ℝ × ℝ := (-a, 0)
def F : ℝ × ℝ := (-c, 0)
def P : ℝ × ℝ := (c, m)

-- Proof statement
theorem max_sin_angle_APF (a b c m : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : a > b) (h4 : c = a / 5) (h5 : c = Real.sqrt (a^2 - b^2)) :
    (∃ θ : ℝ, θ = Real.max (Real.sin θ) ∧ θ = Real.sin (Real.arctan (Real.sqrt 3 / 3))) :=
sorry

end max_sin_angle_APF_l461_461745


namespace cpu_transistors_in_2010_l461_461905

noncomputable def transistors_1990 : ℕ := 2000000
noncomputable def doubling_period : ℝ := 1.5
noncomputable def years_passed : ℕ := 20
noncomputable def number_of_periods := (years_passed : ℝ) / doubling_period

def transistors_2010 (initial_transistors : ℕ) (periods : ℝ) : ℕ :=
  initial_transistors * 2^ (periods.to_nat)

theorem cpu_transistors_in_2010 : transistors_2010 transistors_1990 number_of_periods = 16384000000 := by
  sorry

end cpu_transistors_in_2010_l461_461905


namespace max_profit_l461_461597

noncomputable def profit (x y : ℕ) : ℕ :=
  2100 * x + 900 * y

def constraint1 (x y : ℕ) : Prop :=
  1.5 * x + 0.5 * y ≤ 150

def constraint2 (x y : ℕ) : Prop :=
  1 * x + 0.3 * y ≤ 90

def constraint3 (x y : ℕ) : Prop :=
  5 * x + 3 * y ≤ 600

def nonnegativity (x y : ℕ) : Prop :=
  x ≥ 0 ∧ y ≥ 0

theorem max_profit : 
  constraint1 40 80 ∧ constraint2 40 80 ∧ constraint3 40 80 ∧ nonnegativity 40 80 → 
  profit 40 80 = 21600 := 
by 
  sorry

end max_profit_l461_461597


namespace correct_option_is_C_l461_461003

/- Definitions -/
def is_jump_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, (a i - a (i + 2)) * (a (i + 2) - a (i + 1)) > 0

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ i : ℕ, a (i + 1) = a i + d

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ i : ℕ, a (i + 1) = a i * q

def is_special_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, a (i + 1) = 2 * a i + 1

/- Theorem to be proved -/
theorem correct_option_is_C :
  (∃ (a : ℕ → ℝ) (d : ℝ), is_arithmetic_sequence a d ∧ is_jump_sequence a) ∨
  (∃ (a : ℕ → ℝ) (q > 0), is_geometric_sequence a q ∧ is_jump_sequence a) ∨
  (∀ (a : ℕ → ℝ) (q : ℝ), is_geometric_sequence a q → is_jump_sequence a → q ∈ Ioo (-1 : ℝ) 0) ∨
  (∀ (a : ℕ → ℝ), is_special_sequence a → is_jump_sequence a)
  → (∀ (a : ℕ → ℝ) (q : ℝ), is_geometric_sequence a q → is_jump_sequence a → q ∈ Ioo (-1 : ℝ) 0) :=
by
  sorry

end correct_option_is_C_l461_461003


namespace max_sin_angle_APF_l461_461746

-- Definitions based on problem conditions
variables (a b c m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
variables (h4 : c = a / 5) (h5 : c = Real.sqrt (a^2 - b^2))

-- Coordinates of points
def A : ℝ × ℝ := (-a, 0)
def F : ℝ × ℝ := (-c, 0)
def P : ℝ × ℝ := (c, m)

-- Proof statement
theorem max_sin_angle_APF (a b c m : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : a > b) (h4 : c = a / 5) (h5 : c = Real.sqrt (a^2 - b^2)) :
    (∃ θ : ℝ, θ = Real.max (Real.sin θ) ∧ θ = Real.sin (Real.arctan (Real.sqrt 3 / 3))) :=
sorry

end max_sin_angle_APF_l461_461746


namespace ball_placement_problem_l461_461244

-- Definitions for the problem conditions
def balls : List ℕ := [1, 2, 3, 4, 5]
def boxes : List ℕ := [1, 2, 3, 4, 5]

-- The definition of a derangement
def is_derangement {α : Type*} (σ : Equiv.Perm α) : Prop :=
  ∀ x, σ x ≠ x

-- Formalizing the problem statement
theorem ball_placement_problem :
  let valid_arrangements_count : ℕ := (Nat.choose 5 2) * 2 in
  valid_arrangements_count = 20 := by
  sorry

end ball_placement_problem_l461_461244


namespace integer_cube_less_than_triple_l461_461552

theorem integer_cube_less_than_triple (x : ℤ) : x^3 < 3 * x ↔ x = 0 :=
by 
  sorry

end integer_cube_less_than_triple_l461_461552


namespace total_people_in_line_l461_461572

theorem total_people_in_line (people_in_front : ℕ) (people_behind : ℕ) (yoon_jeong : ℕ) :
  people_in_front = 6 ∧ people_behind = 4 ∧ yoon_jeong = 1 → 
  people_in_front + yoon_jeong + people_behind = 11 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h2 h3
  rw [h1, h2, h3]
  norm_num

end total_people_in_line_l461_461572


namespace part1_c_range_part2_monotonicity_l461_461338

noncomputable def f (x : ℝ) := 2 * Real.log x + 1

theorem part1_c_range (c : ℝ) (x : ℝ) (h : a > 0) : f x ≤ 2 * x + c → c ≥ -1 :=
sorry

noncomputable def g (x a : ℝ) := (f x - f a) / (x - a)

theorem part2_monotonicity (a : ℝ) (h : a > 0) : monotone_decreasing_on g (0, a) ∧ monotone_decreasing_on g (a, +∞) :=
sorry

end part1_c_range_part2_monotonicity_l461_461338


namespace find_T_l461_461311

-- Definition of the sequence and related sum conditions
def is_valid_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) := ∀ n : ℕ, 0 < n → 2 * Real.sqrt (S n) = a n + 1 ∧ (S (n+1) = S n + a (n+1))

def sequence_a (n : ℕ) : ℕ := 2 * n - 1

def sequence_b (n : ℕ) : ℕ := (2 * n) * 4^(n-1)

def T_n (n : ℕ) : ℕ := 1 * 4 + 2 * 4^2 + ... -- Formula definition needs to be extended

theorem find_T (a : ℕ → ℕ) (S : ℕ → ℕ) (H : is_valid_sequence a S) : 
 (∀ n, 0 < n → a n = sequence_a (n)) → 
 (∀ n, 0 < n → T_n n = (4 / 9) + (3 * n - 1) / 9 * 4 ^ (n + 1)) := 
by sorry

end find_T_l461_461311


namespace total_length_of_joined_papers_l461_461153

theorem total_length_of_joined_papers :
  let length_each_sheet := 10 -- in cm
  let number_of_sheets := 20
  let overlap_length := 0.5 -- in cm
  let total_overlapping_connections := number_of_sheets - 1
  let total_length_without_overlap := length_each_sheet * number_of_sheets
  let total_overlap_length := overlap_length * total_overlapping_connections
  let total_length := total_length_without_overlap - total_overlap_length
  total_length = 190.5 :=
by {
    sorry
}

end total_length_of_joined_papers_l461_461153


namespace max_area_of_triangle_on_parabola_l461_461849

noncomputable def area_of_triangle_ABC (p : ℝ) : ℝ :=
  (1 / 2) * abs (3 * p^2 - 14 * p + 15)

theorem max_area_of_triangle_on_parabola :
  ∃ p : ℝ, 1 ≤ p ∧ p ≤ 3 ∧ area_of_triangle_ABC p = 2 := sorry

end max_area_of_triangle_on_parabola_l461_461849


namespace floor_e_eq_2_l461_461699

theorem floor_e_eq_2 : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_e_eq_2_l461_461699


namespace triangle_sides_l461_461015

noncomputable def sides (a b c : ℝ) : Prop :=
  (a = Real.sqrt (427 / 3)) ∧
  (b = Real.sqrt (427 / 3) + 3/2) ∧
  (c = Real.sqrt (427 / 3) - 3/2)

theorem triangle_sides (a b c : ℝ) (h1 : b - c = 3) (h2 : ∃ d : ℝ, d = 10)
  (h3 : ∃ BD CD : ℝ, CD - BD = 12 ∧ BD + CD = a ∧ 
    a = 2 * (BD + 12 / 2)) :
  sides a b c :=
  sorry

end triangle_sides_l461_461015


namespace num_rel_prime_to_15_l461_461253

theorem num_rel_prime_to_15 : 
  {a : ℕ | a < 15 ∧ Int.gcd 15 a = 1}.card = 8 := by 
  sorry

end num_rel_prime_to_15_l461_461253


namespace right_isosceles_areas_no_relations_l461_461098

theorem right_isosceles_areas_no_relations :
  let W := 1 / 2 * 5 * 5
  let X := 1 / 2 * 12 * 12
  let Y := 1 / 2 * 13 * 13
  ¬ (X + Y = 2 * W + X ∨ W + X = Y ∨ 2 * X = W + Y ∨ X + W = W ∨ W + Y = 2 * X) :=
by
  sorry

end right_isosceles_areas_no_relations_l461_461098


namespace ferris_wheel_small_seats_l461_461107

-- Given: There are 23 large seats on the Ferris wheel.
-- Each small seat can hold 14 people.
-- Each large seat can hold 54 people.
-- 28 people can ride the Ferris wheel on small seats.
-- Question: How many small seats are there?

theorem ferris_wheel_small_seats (large_seats : ℕ) (people_per_small_seat : ℕ) (people_per_large_seat : ℕ) 
  (people_on_small_seats : ℕ) (total_small_seats : ℕ) :
  23 = large_seats →
  14 = people_per_small_seat →
  54 = people_per_large_seat →
  28 = people_on_small_seats →
  total_small_seats = people_on_small_seats / people_per_small_seat →
  total_small_seats = 2 :=
by 
  intros h1 h2 h3 h4 h5
  rw [h2, h4]
  exact h5
  simp [Int.div_eq_of_eq_mul_right, h5]
  sorry

end ferris_wheel_small_seats_l461_461107


namespace union_of_A_and_B_l461_461460

def setA : Set ℝ := { x | -3 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 3 }
def setB : Set ℝ := { x | 1 < x }

theorem union_of_A_and_B :
  setA ∪ setB = { x | -1 ≤ x } := sorry

end union_of_A_and_B_l461_461460


namespace range_of_x_l461_461392

theorem range_of_x {a : ℝ} : 
  (∀ a : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (x = 0 ∨ x = -2) :=
by sorry

end range_of_x_l461_461392


namespace rat_value_l461_461406

def alphabet_value (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | 'c' => 3
  | 'd' => 4
  | 'e' => 5
  | 'f' => 6
  | 'g' => 7
  | 'h' => 8
  | 'i' => 9
  | 'j' => 10
  | 'k' => 11
  | 'l' => 12
  | 'm' => 13
  | 'n' => 14
  | 'o' => 15
  | 'p' => 16
  | 'q' => 17
  | 'r' => 18
  | 's' => 19
  | 't' => 20
  | 'u' => 21
  | 'v' => 22
  | 'w' => 23
  | 'x' => 24
  | 'y' => 25
  | 'z' => 26
  | _ => 0

def letter_value (c : Char) (N : ℕ) : ℕ :=
  (alphabet_value c) + N

def word_value (word : String) (N : ℕ) : ℕ :=
  let total_value := word.to_list.foldl (λ acc c => acc + letter_value c N) 0
  total_value * word.length

theorem rat_value (N : ℕ) : word_value "rat" N = 117 + 9 * N := by
  sorry

end rat_value_l461_461406


namespace even_terms_in_expansion_l461_461809

theorem even_terms_in_expansion (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  ∃ (count_even : ℕ), count_even = 5 ∧ 
    ∀ k < 9, 
      (binomial 8 k * m^(8-k) * n^k % 2 = 0) ↔
        (k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5 ∨ k = 6) := 
by
  sorry

end even_terms_in_expansion_l461_461809


namespace xiao_zhang_complete_task_l461_461571

open Nat

def xiaoZhangCharacters (n : ℕ) : ℕ :=
match n with
| 0 => 0
| (n+1) => 2 * (xiaoZhangCharacters n)

theorem xiao_zhang_complete_task :
  ∀ (total_chars : ℕ), (total_chars > 0) → 
  (xiaoZhangCharacters 5 = (total_chars / 3)) →
  (xiaoZhangCharacters 6 = total_chars) :=
by
  sorry

end xiao_zhang_complete_task_l461_461571


namespace lines_from_equation_l461_461677

-- Definitions for the conditions
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Equivalent Lean statement to the proof problem
theorem lines_from_equation :
  (∀ x y : ℝ, satisfies_equation x y → (y = -x - 2) ∨ (y = -2 * x + 1)) :=
by
  intros x y h
  sorry

end lines_from_equation_l461_461677


namespace degree_of_poly_l461_461971

-- Define the polynomials
def poly1 : Polynomial ℚ := (X^2 + 1) ^ 5
def poly2 : Polynomial ℚ := (X^3 + 1) ^ 4

-- Define the polynomial representing the product
def poly : Polynomial ℚ := poly1 * poly2

-- State the theorem to prove the degree
theorem degree_of_poly : (Polynomial.degree poly) = 22 := by 
  sorry

end degree_of_poly_l461_461971


namespace perimeter_divided_by_a_l461_461135

theorem perimeter_divided_by_a (a : ℝ) (h : a ≠ 0) :
  let P := dist (-a, -a/2) (a, -a) + dist (a, -a) (a/2, a) + dist (a/2, a) (-a, a) + dist (-a, a) (-a, -a/2) in
  P / a = (Real.sqrt 17 + 8) / 2 := 
by {
  let intersection1 := (-a, -a/2),
  let intersection2 := (a/2, a),
  let vertex1 := (a, -a),
  let vertex2 := (-a, a),

  have d1 : dist intersection1 vertex1 = (a * Real.sqrt (1+16))/2 := sorry,
  have d2 : dist vertex1 intersection2 = a * (Real.sqrt 1 +(1/4)) := sorry,
  have d3 : dist intersection2 vertex2 = 3 * a / 2 := sorry,
  have d4 : dist vertex2 intersection1 = 3 * a / 2 := sorry,
  have perimeter := d1 + d2 + d3 + d4,
  have emr := perimeter / a = (Real.sqrt 17 + 8) / 2 := by sorry,
  exact emr,
  
  sorry
}

end perimeter_divided_by_a_l461_461135


namespace burger_combinations_l461_461231

theorem burger_combinations (condiments : ℕ) (meat_patties : ℕ) (bun_types : ℕ) 
  (hcondiments : condiments = 10) (hmeat_patties : meat_patties = 3) 
  (hbun_types : bun_types = 2) : 
  2^condiments * meat_patties * bun_types = 6144 :=
by 
  -- initial computation
  have h1 : 2^condiments = 2^10 := by rw hcondiments;
  have h2 : 2^10 = 1024 := by norm_num;
  -- combining the results
  have h3 : 1024 * meat_patties * bun_types = 6144 := by rw [hmeat_patties, hbun_types]; norm_num;
  -- final assembly
  rw [h1, h2];
  exact h3

end burger_combinations_l461_461231


namespace part1_c_range_part2_monotonicity_l461_461340

noncomputable def f (x : ℝ) := 2 * Real.log x + 1

theorem part1_c_range (c : ℝ) (x : ℝ) (h : a > 0) : f x ≤ 2 * x + c → c ≥ -1 :=
sorry

noncomputable def g (x a : ℝ) := (f x - f a) / (x - a)

theorem part2_monotonicity (a : ℝ) (h : a > 0) : monotone_decreasing_on g (0, a) ∧ monotone_decreasing_on g (a, +∞) :=
sorry

end part1_c_range_part2_monotonicity_l461_461340


namespace entrance_fee_per_person_l461_461901

theorem entrance_fee_per_person :
  let ticket_price := 50.00
  let processing_fee_rate := 0.15
  let parking_fee := 10.00
  let total_cost := 135.00
  let known_cost := 2 * ticket_price + processing_fee_rate * (2 * ticket_price) + parking_fee
  ∃ entrance_fee_per_person, 2 * entrance_fee_per_person + known_cost = total_cost :=
by
  sorry

end entrance_fee_per_person_l461_461901


namespace number_of_invertible_integers_mod_15_l461_461270

theorem number_of_invertible_integers_mod_15 :
  (finset.card {a ∈ finset.range 15 | Int.gcd a 15 = 1}) = 8 := by
  sorry

end number_of_invertible_integers_mod_15_l461_461270


namespace function_has_three_zeros_l461_461130

def f (x : ℝ) : ℝ :=
  if x > 0 then 
    real.log x 
  else 
    -x * (x + 2)

theorem function_has_three_zeros :
  ∃ a b c : ℝ, (a < b) ∧ (b < c) ∧ f(a) = 0 ∧ f(b) = 0 ∧ f(c) = 0 ∧ ∀ x, f(x) = 0 → (x = a ∨ x = b ∨ x = c) := by
  sorry

end function_has_three_zeros_l461_461130


namespace valid_dozenal_numbers_l461_461218

/-- A two-digit number in the dozenal (base 12) system. -/
def dozenal_number (a b : ℕ) : ℕ := 12 * a + b

/-- Determine if a given dozenal number and its digits satisfy the condition. -/
def condition_satisfied (a b : ℕ) : Prop :=
  (12 * a + b - (a + b)) % 12 = 5

/-- There are 12 valid two-digit dozenal numbers satisfying the given condition. -/
theorem valid_dozenal_numbers : 
  (finset.univ.filter (λ x : ℕ × ℕ, condition_satisfied x.fst x.snd ∧ 1 ≤ x.fst ∧ x.fst ≤ 11 ∧ 0 ≤ x.snd ∧ x.snd ≤ 11)).card = 12 :=
by
  sorry

end valid_dozenal_numbers_l461_461218


namespace fermats_little_theorem_l461_461184

open Nat

theorem fermats_little_theorem (a n : ℕ) (h : gcd a n = 1) : a ^ euler_totient n ≡ 1 [MOD n] :=
by
  sorry

end fermats_little_theorem_l461_461184


namespace interior_point_is_centroid_l461_461093

-- Definitions of the vertices and lattice points
structure Point := (x y : Int)

def Point.isLatticePoint (p : Point) : Prop := true

-- Definition of a Triangle
structure Triangle :=
(O P Q : Point)

def Triangle.interior_point (𝛥 : Triangle) (C : Point) : Prop :=
Point.isLatticePoint C ∧ -- C is a lattice point
(0 < C.x ∧ C.x < 1) ∧ (0 < C.y ∧ C.y < 1) -- C is strictly inside the triangle

-- Predicate indicating that no additional lattice points lie on the boundaries
def Triangle.no_additional_boundary_points (𝛥 : Triangle) (C : Point) : Prop :=
∀(p : Point), Point.isLatticePoint p → -- p is a lattice point
(p ≠ 𝛥.O ∧ p ≠ 𝛥.P ∧ p ≠ 𝛥.Q) → -- not a vertex
(p ≠ C) → -- not the interior point
¬ ((p.x = 𝛥.O.x ∧ p.y = 𝛥.O.y) ∨ -- not on boundary OP
   (p.x = 𝛥.P.x ∧ p.y = 𝛥.P.y) ∨ -- not on boundary PQ
   (p.x = 𝛥.Q.x ∧ p.y = 𝛥.Q.y)) -- not on boundary QO

-- Definition of centroid
def Triangle.centroid (𝛥 : Triangle) : Point :=
{ x := (𝛥.O.x + 𝛥.P.x + 𝛥.Q.x) / 3,
  y := (𝛥.O.y + 𝛥.P.y + 𝛥.Q.y) / 3 }

-- Theorem: the only interior point is the centroid
theorem interior_point_is_centroid (𝛥 : Triangle) (C : Point)
  (h1 : Triangle.interior_point 𝛥 C)
  (h2 : Triangle.no_additional_boundary_points 𝛥 C) :
  C = Triangle.centroid 𝛥 :=
sorry

end interior_point_is_centroid_l461_461093


namespace relationship_among_abc_l461_461852

noncomputable def a : ℝ := Real.log 2 / Real.log 0.3
noncomputable def b : ℝ := 0.3 ^ 2
noncomputable def c : ℝ := 2 ^ 0.3

theorem relationship_among_abc : a < b ∧ b < c := by
  have ha : a < 0 := calc
    a = Real.log 2 / Real.log 0.3 : rfl
    ... < 0 : by sorry -- Use property of logarithm here
  have hb : 0 < b ∧ b < 1 := calc
    0 < 0.3 ^ 2 : by sorry -- Basic properties of exponentiation
    ... < 1 : by sorry -- Basic properties of exponentiation
  have hc : c > 1 := calc
    2 ^ 0.3 > 1 : by sorry -- Basic properties of exponentiation
  exact ⟨ha.left.trans hb.left, hb.right.trans hc⟩

end relationship_among_abc_l461_461852


namespace closest_to_210_l461_461994

theorem closest_to_210 : |2.43 * 8.2 * (5.15 + 4.88) - 210| < |2.43 * 8.2 * (5.15 + 4.88) - x| := by
  def product := 2.43 * 8.2 * (5.15 + 4.88)
  have h1 : abs (product - 210) < abs (product - 190) := sorry
  have h2 : abs (product - 210) < abs (product - 200) := sorry
  have h3 : abs (product - 210) < abs (product - 220) := sorry
  have h4 : abs (product - 210) < abs (product - 230) := sorry
  show abs (product - 210) < abs (product - x) from sorry

end closest_to_210_l461_461994


namespace usual_eggs_accepted_l461_461817

theorem usual_eggs_accepted (A R : ℝ) (h1 : A / R = 1 / 4) (h2 : (A + 12) / (R - 4) = 99 / 1) (h3 : A + R = 400) :
  A = 392 :=
by
  sorry

end usual_eggs_accepted_l461_461817


namespace total_pizzas_bought_l461_461609

theorem total_pizzas_bought (slices_small : ℕ) (slices_medium : ℕ) (slices_large : ℕ) 
                            (num_small : ℕ) (num_medium : ℕ) (total_slices : ℕ) :
  slices_small = 6 → 
  slices_medium = 8 → 
  slices_large = 12 → 
  num_small = 4 → 
  num_medium = 5 → 
  total_slices = 136 → 
  (total_slices = num_small * slices_small + num_medium * slices_medium + 72) →
  15 = num_small + num_medium + 6 :=
by
  intros
  sorry

end total_pizzas_bought_l461_461609


namespace geom_sequence_jump_condition_l461_461002

def is_jump_sequence {α : Type*} [LinearOrder α] (a : ℕ → α) : Prop :=
  ∀ i : ℕ, (a i - a (i + 2)) * (a (i + 2) - a (i + 1)) > 0

def is_geometric_sequence {α : Type*} [Mul α] [One α] (a : ℕ → α) (q : α) : Prop :=
  ∀ i : ℕ, a (i + 1) = a i * q

theorem geom_sequence_jump_condition {α : Type*} [LinearOrder α] [Mul α] [One α]
  {a : ℕ → α} {q : α} (h1 : is_geometric_sequence a q) (h2 : is_jump_sequence a) :
  q ∈ Set.Ioo (-1 : α) 0 :=
sorry

end geom_sequence_jump_condition_l461_461002


namespace license_plates_count_l461_461654

def letters : Finset Char := Finset.range 26  -- A-Z
def digits : Finset Char := Finset.range 10  -- 0-9

noncomputable def count_plates : Nat :=
  let same_letter_case := 26 * 10           -- first two the same, 26 options for the letter, 10 for the digit
  let different_letter_case := 26 * 25 * 10 -- first two different, 26 options for first letter, 25 for the second, 10 for the digit
  same_letter_case + different_letter_case  -- total plates

theorem license_plates_count : 
  count_plates = 6760 :=
by {
  sorry
}

end license_plates_count_l461_461654


namespace beads_problem_l461_461592

noncomputable def number_of_blue_beads (total_beads : ℕ) (beads_with_blue_neighbor : ℕ) (beads_with_green_neighbor : ℕ) : ℕ :=
  let beads_with_both_neighbors := beads_with_blue_neighbor + beads_with_green_neighbor - total_beads
  let beads_with_only_blue_neighbor := beads_with_blue_neighbor - beads_with_both_neighbors
  (2 * beads_with_only_blue_neighbor + beads_with_both_neighbors) / 2

theorem beads_problem : number_of_blue_beads 30 26 20 = 18 := by 
  -- ...
  sorry

end beads_problem_l461_461592


namespace rosie_pies_proof_l461_461884

-- Define the given condition
def pies_per_apples (p: ℕ) (a: ℕ) : ℕ := a / p

-- Given that Rosie can make 3 pies from 12 apples
def given_condition : pies_per_apples 3 12 = 4 := rfl

-- The proof problem statement:
theorem rosie_pies_proof : ∀ (a: ℕ) (n: ℕ), pies_per_apples 3 12 = 4 → pies_per_apples n a = 4 → pies_per_apples n a = 9 :=
begin
  sorry
end

end rosie_pies_proof_l461_461884


namespace range_of_a_l461_461336

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a - 1 / Real.exp x)

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ deriv (f a) x₁ = 0 ∧ deriv (f a) x₂ = 0) ↔ -1 / Real.exp 2 < a ∧ a < 0 := 
sorry

end range_of_a_l461_461336


namespace alpha_plus_beta_eq_neg_pi_over_4_l461_461804

theorem alpha_plus_beta_eq_neg_pi_over_4
  (α β : ℝ)
  (h1 : cos (2 * α) = - sqrt 10 / 10)
  (h2 : sin (α - β) = sqrt 5 / 5)
  (hα : α ∈ Set.Ioo (π / 4) (π / 2))
  (hβ : β ∈ Set.Ioo (-π) (-π / 2)) :
  α + β = -π / 4 :=
sorry

end alpha_plus_beta_eq_neg_pi_over_4_l461_461804


namespace tan_660_eq_neg_sqrt3_l461_461295

theorem tan_660_eq_neg_sqrt3 :
  let pi := Real.pi in
  tan (660 * pi / 180) = -Real.sqrt 3 :=
by
  sorry

end tan_660_eq_neg_sqrt3_l461_461295


namespace remainder_div_x_plus_1_l461_461563

theorem remainder_div_x_plus_1 (x : ℝ) : (x - 1) * (x + 1) + 1 = (x + 1) * (x - 1) * x = 0 := 
by sorry

end remainder_div_x_plus_1_l461_461563


namespace right_triangle_area_l461_461405

theorem right_triangle_area (leg1 leg2 : ℕ) (h1 : leg1 = 24) (h2 : leg2 = 30) : 
  let area := (1 / 2) * leg1 * leg2 in
  area = 360 := by
  sorry

end right_triangle_area_l461_461405


namespace pairs_of_powers_of_two_l461_461287

theorem pairs_of_powers_of_two (m n : ℕ) (h1 : m > 0) (h2 : n > 0)
  (h3 : ∃ a : ℕ, m + n = 2^a) (h4 : ∃ b : ℕ, mn + 1 = 2^b) :
  (∃ a : ℕ, m = 2^a - 1 ∧ n = 1) ∨ 
  (∃ a : ℕ, m = 2^(a-1) + 1 ∧ n = 2^(a-1) - 1) :=
sorry

end pairs_of_powers_of_two_l461_461287


namespace chord_length_l461_461127

-- Define the polar equations in Cartesian form
def circle := ∀ (x y : ℝ), x^2 + y^2 = 1
def line := ∀ (x y : ℝ), x = 1 / 2

-- State the theorem
theorem chord_length : ∀ (x y : ℝ), (circle x y ∧ line x y) → 2 * Real.sqrt(1 - (1/2)^2) = Real.sqrt 3 :=
by sorry

end chord_length_l461_461127


namespace sales_volume_A_correct_total_cost_B_correct_sales_volume_B_correct_max_total_profit_correct_possible_selling_prices_correct_l461_461437

section disinfectant_water

variables (x : ℕ) (x_greater_than_30 : x > 30)

-- Conditions
def cost_price_A := 20 -- yuan per bottle
def cost_price_B := 30 -- yuan per bottle
def total_cost := 2000 -- yuan

def initial_sales_A := 100 -- bottles at 30 yuan per bottle
def sell_decrease_A := 5 -- bottles per yuan increase
def sell_price_B := 60 -- yuan per bottle

-- Sales volume of type A disinfectant water
def sales_volume_A := 250 - 5 * x

-- Total cost price of type B disinfectant water
def total_cost_B := 2000 - 20 * (250 - 5 * x)

-- Sales volume of type B disinfectant water
def sales_volume_B := (total_cost_B x) / cost_price_B

-- Total profit function
def total_profit := (250 - 5 * x) * (x - 20) + ((total_cost_B x) / 30 - 100) * (60 - 30)

-- Maximum total profit
def max_total_profit := 2125

-- Possible selling prices for total profit >= 1945 yuan
def possible_selling_prices := {x : ℕ | 39 ≤ x ∧ x ≤ 50 ∧ (x % 3 = 0)}

-- Proofs
theorem sales_volume_A_correct : sales_volume_A x = 250 - 5 * x := by
  sorry

theorem total_cost_B_correct : total_cost_B x = 100 * x - 3000 := by
  sorry

theorem sales_volume_B_correct : sales_volume_B x = (100 * x - 3000) / 30 := by
  sorry

theorem max_total_profit_correct : total_profit x = -5*(x - 45) * (x - 45) + 2125 := by
  sorry

theorem possible_selling_prices_correct : ∀ x, 1945 ≤ total_profit x → x ∈ possible_selling_prices := by
  sorry

end disinfectant_water

end sales_volume_A_correct_total_cost_B_correct_sales_volume_B_correct_max_total_profit_correct_possible_selling_prices_correct_l461_461437


namespace final_price_of_jacket_l461_461909

noncomputable def original_price : ℝ := 240
noncomputable def initial_discount : ℝ := 0.6
noncomputable def additional_discount : ℝ := 0.25

theorem final_price_of_jacket :
  let price_after_initial_discount := original_price * (1 - initial_discount)
  let final_price := price_after_initial_discount * (1 - additional_discount)
  final_price = 72 := 
by
  sorry

end final_price_of_jacket_l461_461909


namespace ratio_of_radii_of_truncated_cone_l461_461610

theorem ratio_of_radii_of_truncated_cone (R r s : ℝ) (h_ratio : r = R / 2)
  (V_truncated_cone V_sphere : ℝ) (h_volume : V_truncated_cone = 3 * V_sphere)
  (h_sphere_vol : V_sphere = (4 / 3) * Real.pi * s^3)
  (h_truncated_cone_vol : V_truncated_cone = Real.pi * √(R*r) / (3 * s)):
  R / r = 2 := 
sorry

end ratio_of_radii_of_truncated_cone_l461_461610


namespace sample_mean_and_variance_l461_461594

variable (x : list ℝ)
variable (h_eq_sum_of_squares : (1 / 10 : ℝ) * (x.map (λ x_i, x_i^2)).sum ≈ 98.048)
variable (h_values : x = [9.6, 10.1, 9.7, 9.8, 10.0, 9.7, 10.0, 9.8, 10.1, 10.2])

noncomputable def sample_mean (x : list ℝ) : ℝ := (1 / x.length) * x.sum

noncomputable def sample_variance (x : list ℝ) (mean : ℝ) : ℝ := 
  (1 / x.length) * (x.map (λ x_i, x_i^2)).sum - mean^2

theorem sample_mean_and_variance :
    sample_mean x = 9.9 ∧
    sample_variance x (sample_mean x) = 0.038 ∧
   (sample_mean (x.map (+ 0.2)) = 10.1 ∧ sample_variance (x.map (+ 0.2)) (sample_mean (x.map (+ 0.2))) = sample_variance x (sample_mean x)) ∧
    (let new_samples := list.repeat (sample_mean (x.map (+ 0.2))) 10 in 
    (sample_mean new_samples ≈ 10.1 ∧ sample_mean new_samples < 10.0 → False)) :=
  by {
    sorry,
  }

end sample_mean_and_variance_l461_461594


namespace percentage_increase_of_x_l461_461524

theorem percentage_increase_of_x (C x y : ℝ) (P : ℝ) (h1 : x * y = C) (h2 : (x * (1 + P / 100)) * (y * (5 / 6)) = C) :
  P = 20 :=
by
  sorry

end percentage_increase_of_x_l461_461524


namespace identify_irrational_number_l461_461223

theorem identify_irrational_number :
  (∀ a b : ℤ, (-1 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (0 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (1 : ℚ) / (2 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  ¬(∃ a b : ℤ, (Real.sqrt 3) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) :=
sorry

end identify_irrational_number_l461_461223


namespace fib_999_1001_l461_461034

-- Definitions related to the Fibonacci sequence
def Fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => Fib (n+1) + Fib n

-- The matrix defined in the condition
def mat := λ n : ℕ,  (λ n : ℕ, (Finset.card n).choose n)

-- The given condition in the problem
axiom matrix_property (n : ℕ) (hn : 0 < n) :
  (Matrix.of ![![1, 1], ![1, 0]] ^ n) = 
  (Matrix.of ![(Fib (n + 1)), (Fib n)], ![(Fib n), (Fib (n - 1)]) )

-- Proof problem statement
theorem fib_999_1001 : Fib 999 * Fib 1001 - Fib 1000^2 = 1 :=
by sorry

end fib_999_1001_l461_461034


namespace shaded_area_correct_l461_461197

theorem shaded_area_correct :
  let O : Point := (0, 0)
  let radius : ℝ := 3
  let square_side : ℝ := 2
  let circle_center : Point := O
  let circle : Circle := Circle.mk O radius
  let square : Square := Square.mk OABC square_side
  let extended_AB_MEETS_circle : Point := D
  let extended_CB_MEETS_circle : Point := E
  let inscribed_square : Square := Square.mk PQRS s
  let vertices_P_and_Q_extends_AB_CB : Prop := vertices_P_and_Q_on_extensions PQRS AB CB
  let intersecting_points_F_and_G : Prop := intersections F_and_G PQ AB_and_CB

  ∃ (s : ℝ), 
    let side_length : ℝ := s 
    let angle_theta : ℝ := 45 
    let area_sector : ℝ := (angle_theta / 360) * π * radius^2
    let area_triangle : ℝ := (1/2) * radius^2 * sin(angle_theta * (π / 180))
    let shaded_area : ℝ := area_sector - area_triangle

    shaded_area = (9 * π / 8) - (9 * √2 / 4) := sorry

end shaded_area_correct_l461_461197


namespace equal_areas_of_trapezoid_and_triangle_l461_461507

theorem equal_areas_of_trapezoid_and_triangle
  (O : Point) (A B C D E F H G : Point)
  (circle_center : O)
  (diameters_perpendicular : O ∈ line(A, B) ∧ O ∈ line(C, D) ∧ line(A, B) ⊥ line(C, D))
  (chords_parallel : line(C, E) ∥ line(B, F))
  (reflection_E : reflection(CD, H) = E)
  (reflection_F : reflection(CD, G) = F)
  (trapezoid_ABFG : Trapezoid A B F G)
  (triangle_CEH : Triangle C E H):
  area(trapezoid_ABFG) = area(triangle_CEH) :=
sorry

end equal_areas_of_trapezoid_and_triangle_l461_461507


namespace shortest_path_cube_eight_vertices_shortest_path_cube_fourteen_vertices_l461_461985

-- Part (a) - Shortest path passing through all 8 vertices of a cube
theorem shortest_path_cube_eight_vertices :
  ∀ (vertices : Finset ℝ), (vertices.card = 8) →
  (∃ path : Finset (Finpair ℝ), path.card = 7 ∧ (∀ v ∈ vertices, ∃ u w, Finpair u w ∈ path)) :=
sorry

-- Part (b) - Shortest path passing through all 14 vertices of a cube with face centers included
theorem shortest_path_cube_fourteen_vertices :
  ∀ (vertices : Finset ℝ), (vertices.card = 14) →
  (∃ path : Finset (Finpair ℝ), path.card = 13 ∧ (∀ v ∈ vertices, ∃ u w, Finpair u w ∈ path) ∧ 
  Finset.card (Finset.filter (λ p, p.length = 6 * Real.sqrt 2 + 1) path) = 1) :=
sorry

end shortest_path_cube_eight_vertices_shortest_path_cube_fourteen_vertices_l461_461985


namespace f_even_f_increasing_f_m_range_l461_461774

def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

-- (1) Prove that f(x) is an even function
theorem f_even : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

-- (2) Prove that f(x) is an increasing function on (-∞, +∞)
theorem f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
by
  sorry

-- (3) Prove that for any x ∈ [1, 5], if f(x) > m, then m < 1/3
theorem f_m_range (m : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → f x > m) → m < 1 / 3 :=
by
  sorry

end f_even_f_increasing_f_m_range_l461_461774


namespace rubber_mass_calculation_l461_461591

noncomputable theory

def copper_wire_mass_kg := 100
def copper_wire_diameter_mm := 1.5
def rubber_insulation_thickness_mm := 0.5
def rubber_piece_length_m := 1
def rubber_piece_cross_sectional_area_mm2 := 3 * 3
def rubber_piece_mass_g := 10
def copper_specific_gravity := 8.95

def q := 22.07 -- the mass of rubber needed in kg

theorem rubber_mass_calculation :
  let mass_copper_wire_g := copper_wire_mass_kg * 1000 in
  let copper_wire_radius_cm := (copper_wire_diameter_mm / 2) * 0.1 in
  let volume_copper_wire_cm3 (h : ℝ) := Real.pi * (copper_wire_radius_cm ^ 2) * h in
  let density_copper := copper_specific_gravity in
  let equation_for_h := mass_copper_wire_g = volume_copper_wire_cm3 1 * density_copper in
  let h := mass_copper_wire_g / (Real.pi * density_copper * (copper_wire_radius_cm ^ 2)) in
  let rubber_piece_volume_cm3 := rubber_piece_length_m * 100 * (rubber_piece_cross_sectional_area_mm2 * 0.01 * 0.01) in
  let rubber_density_g_per_cm3 := rubber_piece_mass_g / rubber_piece_volume_cm3 in
  let outer_diameter_mm := copper_wire_diameter_mm + 2 * rubber_insulation_thickness_mm in
  let outer_radius_cm := (outer_diameter_mm / 2) * 0.1 in
  let delta_v_cm3 := Real.pi * ((outer_radius_cm ^ 2) - (copper_wire_radius_cm ^ 2)) * h in
  let rubber_mass_kg := delta_v_cm3 * rubber_density_g_per_cm3 / 1000 in
  q = 22.07 :=
sorry

end rubber_mass_calculation_l461_461591


namespace rosie_pies_l461_461889

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end rosie_pies_l461_461889


namespace sum_base4_equiv_l461_461124

theorem sum_base4_equiv (a b : ℕ) (h₁ : a = 172) (h₂ : b = 83) :
    let s := a + b in s = 255 ∧ (Nat.toDigits 4 s = [3, 3, 3, 3, 3]) :=
by
  sorry

end sum_base4_equiv_l461_461124


namespace euler_line_IKL_l461_461738

variables {A B C I D P Q K L : Type}
variables [Incenter I A B C] [IncircleTang ABC D] [OnSideBC P A B C D] [OnSideBC Q A B C D]
variables [AnglesEqual PAB BCA] [AnglesEqual QAC ABC]
variables [Incenter K A B P] [Incenter L A C Q]

theorem euler_line_IKL (ABC : Type) (A B C I D P Q K L : Type)
  [Incenter I A B C] [IncircleTang ABC D] [OnSideBC P A B C D] [OnSideBC Q A B C D]
  [AnglesEqual PAB BCA] [AnglesEqual QAC ABC]
  [Incenter K A B P] [Incenter L A C Q] :
  Euler_Line IKL AD :=
sorry

end euler_line_IKL_l461_461738


namespace find_not_right_triangle_l461_461227

noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

noncomputable def condition_a := is_right_triangle 3 4 5

noncomputable def condition_b := is_right_triangle 1 (sqrt 3) 2

noncomputable def condition_c := ¬ is_right_triangle (sqrt 11) 2 4

noncomputable def condition_d (a b c : ℝ) : Prop :=
  a^2 = (c + b) * (c - b) ∧ is_right_triangle a b c

theorem find_not_right_triangle :
  (∃ a b c, condition_d a b c) → 
  (¬condition_a ∨ ¬condition_b ∨ condition_c ∨ ¬condition_d 3 4 5) :=
sorry

end find_not_right_triangle_l461_461227


namespace prime_factor_of_difference_l461_461812

noncomputable def base20_number (A B : ℕ) (hA : A < 20) (hB : B < 20) : ℕ :=
  20 * A + B

theorem prime_factor_of_difference (A B : ℕ) (hA : A < 20) (hB : B < 20) (hAB : A ≠ B) :
  ∃ p : ℕ, Prime p ∧ p = 19 ∧ p ∣ abs (base20_number A B hA hB - base20_number B A hB hA) :=
by
  sorry

end prime_factor_of_difference_l461_461812


namespace largest_five_digit_integer_with_conditions_l461_461972

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digits_product (n : ℕ) : ℕ :=
  (n % 10) * ((n / 10) % 10) * ((n / 100) % 10) * ((n / 1000) % 10) * ((n / 10000) % 10)

def digits_sum (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10) + ((n / 10000) % 10)

theorem largest_five_digit_integer_with_conditions :
  ∃ n : ℕ, is_five_digit n ∧ digits_product n = 40320 ∧ digits_sum n < 35 ∧
  ∀ m : ℕ, is_five_digit m ∧ digits_product m = 40320 ∧ digits_sum m < 35 → n ≥ m :=
sorry

end largest_five_digit_integer_with_conditions_l461_461972


namespace divide_diagonal_into_three_equal_parts_l461_461084

theorem divide_diagonal_into_three_equal_parts (A C M1 M2 : Point) (rectangle : Rectangle) 
    (h1 : rectangle.width = 4) (h2 : rectangle.height = 1) 
    (h3 : diagonal A C rectangle) 
    (h4 : midpoint M1 M2) 
    (h5 : intersect_diagonal_in_three_equal_parts A C M1 M2) 
    : divides_diagonal_into_three_equal_segments A C :=
begin
  sorry
end

end divide_diagonal_into_three_equal_parts_l461_461084


namespace sale_price_correct_l461_461208

noncomputable def original_price : ℝ := 600.00
noncomputable def first_discount_factor : ℝ := 0.75
noncomputable def second_discount_factor : ℝ := 0.90
noncomputable def final_price : ℝ := original_price * first_discount_factor * second_discount_factor
noncomputable def expected_final_price : ℝ := 0.675 * original_price

theorem sale_price_correct : final_price = expected_final_price := sorry

end sale_price_correct_l461_461208


namespace coefficient_x_6_in_expansion_l461_461968

-- Define the variable expressions and constraints of the problem
def expansion_expr : ℕ → ℤ := λ k, Nat.choose 4 k * 1^(4 - k) * (-3)^(k)
def term_coefficient_of_x_pow_6 (k : ℕ) : ℕ := if (3 * k = 6) then Nat.choose 4 k * 9 else 0

-- Prove that the coefficient of x^6 in the expansion of (1-3x^3)^4 is 54
theorem coefficient_x_6_in_expansion : term_coefficient_of_x_pow_6 2 = 54 := by
  -- Simplify the expression for the term coefficient of x^6 when k = 2
  simp only [term_coefficient_of_x_pow_6]
  split_ifs
  simp [Nat.choose, Nat.factorial]
  sorry -- one could continue simplifying this manually or provide arithmetic through Lean library

end coefficient_x_6_in_expansion_l461_461968


namespace triangle_inequality_l461_461091

theorem triangle_inequality
  (la lb lc r p a b c S : ℝ)
  (h1 : la * lb * lc ≤ Real.sqrt (p^3 * (p - a) * (p - b) * (p - c)))
  (heron : S = Real.sqrt(p * (p - a) * (p - b) * (p - c)))
  (area_inradius : S = r * p) :
  la * lb * lc ≤ r * p^2 :=
by {
  sorry
}

end triangle_inequality_l461_461091


namespace compare_numbers_l461_461928

theorem compare_numbers (x y z : ℝ) (h1 : x = 0.3^2) (h2 : y = log 2 0.3) (h3 : z = 2^0.3) : 
  y < x ∧ x < z :=
by
  have h4 : 0 < 0.3, from sorry,
  have h5 : 0.3 < 1, from sorry,
  have h6 : log 2 1 = 0, from sorry,
  have h7 : log 2 0.3 < log 2 1, from sorry,
  have h8 : log 2 0.3 < 0, by {rw [h6] at h7, exact h7},
  have h9 : 1 = 2^0, from sorry,
  have h10 : 2^0 < 2^0.3, from sorry,
  exact ⟨h8, sorry, sorry⟩

end compare_numbers_l461_461928


namespace select_student_number_l461_461210

theorem select_student_number :
  ∀ (total_students interval interval_start lower_bound upper_bound : ℕ),
  (total_students = 800) →
  (interval = 16) →
  (interval_start = 7) →
  (lower_bound = 65) →
  (upper_bound = 80) →
  (lower_bound / interval).ceil + 1 = 5 →
  (interval_start + interval * 4 = 71) :=
by
  intros total_students interval interval_start lower_bound upper_bound
  intros h_total_students h_interval h_interval_start h_lower_bound h_upper_bound h_group_formula
  simp [h_total_students, h_interval, h_interval_start, h_lower_bound, h_upper_bound, h_group_formula]
  sorry

end select_student_number_l461_461210


namespace combination_mod_100_l461_461819

def totalDistinctHands : Nat := Nat.choose 60 12

def remainder (n : Nat) (m : Nat) : Nat := n % m

theorem combination_mod_100 :
  remainder totalDistinctHands 100 = R :=
sorry

end combination_mod_100_l461_461819


namespace angle_bisector_theorem_l461_461850

noncomputable def triangle_ABC (A B C : Point) : Prop :=
  ∃ (E G : Point),
  right_triangle A B C ∧
  (diameter (circle_through A C) A C) ∧ 
  (on_circle E (circle_through A C)) ∧
  collinear E A C ∧
  tangent_line_at_point_of_circle B E G ∧
  segment_eq E G G B

-- A definition stating the problem conditions
def right_triangle_proof : Prop :=
  ∀ (A B C : Point), (right_triangle A B C ∧ diameter (circle_through A C) A C ∧
    ∃ (E G : Point), 
    on_circle E (circle_through A C) ∧ collinear E A C ∧
    tangent_line_at_point_of_circle B E G) →
  segment_eq E G G B

theorem angle_bisector_theorem : right_triangle_proof :=
by
  sorry

end angle_bisector_theorem_l461_461850


namespace inequality_solution_set_l461_461935

theorem inequality_solution_set (x : ℝ) : (-2 < x ∧ x ≤ 3) ↔ (x - 3) / (x + 2) ≤ 0 := 
sorry

end inequality_solution_set_l461_461935


namespace coprime_integers_lt_15_l461_461274

theorem coprime_integers_lt_15 : ∃ (S : Finset ℕ), S.card = 8 ∧ (∀ a ∈ S, a < 15 ∧ Nat.gcd a 15 = 1) :=
by
  sorry

end coprime_integers_lt_15_l461_461274


namespace A_doubles_after_6_months_l461_461190

variable (x : ℕ)

def A_investment_share (x : ℕ) := (3000 * x) + (6000 * (12 - x))
def B_investment_share := 4500 * 12

theorem A_doubles_after_6_months (h : A_investment_share x = B_investment_share) : x = 6 :=
by
  sorry

end A_doubles_after_6_months_l461_461190


namespace smallest_integer_conditions_l461_461161

-- Definition of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Definition of having a prime factor less than a given number
def has_prime_factor_less_than (n k : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ p < k

-- Problem statement
theorem smallest_integer_conditions :
  ∃ n : ℕ, n > 0 ∧ ¬ is_prime n ∧ ¬ is_square n ∧ ¬ has_prime_factor_less_than n 60 ∧ ∀ m : ℕ, (m > 0 ∧ ¬ is_prime m ∧ ¬ is_square m ∧ ¬ has_prime_factor_less_than m 60) → n ≤ m :=
  sorry

end smallest_integer_conditions_l461_461161


namespace ratio_soda_water_l461_461238

variables (W S : ℕ) (k : ℕ)

-- Conditions of the problem
def condition1 : Prop := S = k * W - 6
def condition2 : Prop := W + S = 54
def positive_integer_k : Prop := k > 0

-- The theorem we want to prove
theorem ratio_soda_water (h1 : condition1 W S k) (h2 : condition2 W S) (h3 : positive_integer_k k) : S / gcd S W = 4 ∧ W / gcd S W = 5 :=
sorry

end ratio_soda_water_l461_461238


namespace area_of_triangle_AGH_l461_461032

theorem area_of_triangle_AGH (A B C G H : Point) (area_ABC : ℝ) (is_midpoint_G : midpoint G A B) (is_midpoint_H : midpoint H A C) (area_triangle_ABC : triangle_area A B C = 120) : 
  triangle_area A G H = 30 := sorry

end area_of_triangle_AGH_l461_461032


namespace extremum_at_x_equals_1_minimum_value_f_l461_461335

def f (a x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

theorem extremum_at_x_equals_1 (a : ℝ) (h : a > 0) (ha : (deriv (f a) 1 = 0)) : a = 1 :=
sorry

theorem minimum_value_f (a : ℝ) (h : a > 0) (hf_min : ∀ x, f a x ≥ 1) : 2 ≤ a :=
sorry

end extremum_at_x_equals_1_minimum_value_f_l461_461335


namespace perimeter_of_ABCDEFG_is_correct_l461_461463

-- Declare the lengths
def length_AB : ℝ := 6
def length_AD : ℝ := length_AB / 2  -- midpoint of AB
def length_BD : ℝ := length_AD
def length_DE : ℝ := length_AD  -- since DE is equal to BD
def length_DG : ℝ := length_DE / 2  -- midpoint of DE
def length_GE : ℝ := length_DG
def length_EA : ℝ := length_DE  -- since EA is equal to DE
def length_AG : ℝ := length_EA

-- Perimeter calculation
def perimeter_ABCDEFG : ℝ :=
  length_AB + length_BD + length_DE + length_DE + length_GE + length_AG

-- The theorem we need to prove
theorem perimeter_of_ABCDEFG_is_correct :
  perimeter_ABCDEFG = 19.5 :=
by
  -- skipping the proof
  sorry

end perimeter_of_ABCDEFG_is_correct_l461_461463


namespace min_books_borrowed_l461_461818

theorem min_books_borrowed
  (total_students : ℕ)
  (students_no_books : ℕ)
  (students_one_book : ℕ)
  (students_two_books : ℕ)
  (avg_books_per_student : ℝ)
  (total_students_eq : total_students = 40)
  (students_no_books_eq : students_no_books = 2)
  (students_one_book_eq : students_one_book = 12)
  (students_two_books_eq : students_two_books = 13)
  (avg_books_per_student_eq : avg_books_per_student = 2) :
  ∀ min_books_borrowed : ℕ, 
    (total_students * avg_books_per_student = 80) → 
    (students_one_book * 1 + students_two_books * 2 ≤ 38) → 
    (total_students - students_no_books - students_one_book - students_two_books = 13) →
    min_books_borrowed * 13 = 42 → 
    min_books_borrowed = 4 :=
by
  intros min_books_borrowed total_books_eq books_count_eq remaining_students_eq total_min_books_eq
  sorry

end min_books_borrowed_l461_461818


namespace eval_expr1_eval_expr2_l461_461284

-- 1. Prove that  2 sqrt(3) * 3rd-root(1.5) * 6th-root(12) = 6
theorem eval_expr1 : 2 * Real.sqrt(3) * Real.sqrt(1.5 ^ (1 / 3)) * Real.sqrt(12 ^ (1 / 6)) = 6 := sorry

-- 2. Prove that log8(27) * log3(4) + 3^(log3(2)) = 4
theorem eval_expr2 : Real.log 27 / Real.log 8 * Real.log 4 / Real.log 3 + 3 ^ (Real.log 2 / Real.log 3) = 4 := sorry

end eval_expr1_eval_expr2_l461_461284


namespace area_of_region_B_l461_461652

open Complex Real

def region_B : Set ℂ :=
  {z : ℂ | ∀ (C : ℂ), (abs (re (z / 30)) ≤ 1 ∧ abs (im (z / 30)) ≤ 1) ∧
                      (abs (re (30 / conj z)) ≤ 1 ∧ abs (im (30 / conj z)) ≤ 1)}

theorem area_of_region_B : measure_theory.measure_space.volume (region_B) = 675 - 112.5 * real.pi :=
  sorry

end area_of_region_B_l461_461652


namespace tiling_with_L_shaped_tiles_l461_461057

theorem tiling_with_L_shaped_tiles (n : ℕ) (h_pos : n > 0) (h_div : ¬(3 ∣ n)) (r c : ℕ) (h_rc : r < 2 * n ∧ c < 2 * n):
  ∃ tile : (ℕ × ℕ → bool), (∀ i j, i < 2 * n ∧ j < 2 * n → (i = r ∧ j = c) → tile (i, j) = false) ∧
  (∀ i j, i < 2 * n ∧ j < 2 * n → (i ≠ r ∨ j ≠ c) → (tile (i, j) = true ∨ tile (i, j) = false)) ∧
  (∀ s, (s.1 < 2 * n ∧ s.2 < 2 * n → tile s)) :=
sorry

end tiling_with_L_shaped_tiles_l461_461057


namespace sum_of_transformed_numbers_l461_461940

theorem sum_of_transformed_numbers (a b S : ℕ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l461_461940


namespace circle_equation_standard_l461_461937

open Real

noncomputable def equation_of_circle : Prop :=
  ∃ R : ℝ, R = sqrt 2 ∧ 
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 → x + y - 2 = 0 → 0 ≤ x ∧ x ≤ 2)

theorem circle_equation_standard :
    equation_of_circle := sorry

end circle_equation_standard_l461_461937


namespace fractions_product_l461_461637

theorem fractions_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by 
  sorry

end fractions_product_l461_461637


namespace perfect_square_example_l461_461541

theorem perfect_square_example (x : ℕ) : (x = 0 ∨ x = 9 ∨ x = 12) → (∃ (y : ℕ), 2^6 + 2^{10} + 2^x = y^2) :=
begin
  intro h,
  cases h,
  { use 2^5 + 1,
    sorry },
  cases h,
  { use 2^5 + 2^3,
    sorry },
  { use 2^6 + 2^3,
    sorry }
end

end perfect_square_example_l461_461541


namespace sum_of_distances_l461_461018

noncomputable def line_param_eq (t : ℝ) : ℝ × ℝ :=
  (1 - (1/2) * t, (real.sqrt 3 / 2) * t)

noncomputable def curve_polar_eq (θ : ℝ) : ℝ :=
  6 * real.cos θ

def point_p : ℝ × ℝ := (1, 0)

theorem sum_of_distances (t₁ t₂ : ℝ) (A B : ℝ × ℝ)
  (hA_line : A = line_param_eq t₁)
  (hB_line : B = line_param_eq t₂)
  (hA_curve : (A.1 - 3)^2 + A.2^2 = 9)
  (hB_curve : (B.1 - 3)^2 + B.2^2 = 9)
  (h_sol : (t₁ = -1 + real.sqrt 6 ∨ t₁ = -1 - real.sqrt 6) ∧
           (t₂ = -1 + real.sqrt 6 ∨ t₂ = -1 - real.sqrt 6))
  : real.abs (t₁) + real.abs (t₂) = 2 * real.sqrt 6 := 
sorry

end sum_of_distances_l461_461018


namespace number_of_integers_satisfying_l461_461791

theorem number_of_integers_satisfying (k1 k2 : ℕ) (hk1 : k1 = 300) (hk2 : k2 = 1000) :
  ∃ m : ℕ, m = 14 ∧ ∀ n : ℕ, 300 < n^2 → n^2 < 1000 → 18 ≤ n ∧ n ≤ 31 :=
by
  use 14
  sorry

end number_of_integers_satisfying_l461_461791


namespace vector_at_t4_is_correct_l461_461209

open Matrix

-- Define vectors at specific time points
def vec_t1 : ℝ × ℝ × ℝ := (2, 6, 16)
def vec_t5 : ℝ × ℝ × ℝ := (-1, -5, -10)

-- Define the parameterized line
def line (t : ℝ) (a : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.1 + t * d.1, a.2 + t * d.2, a.3 + t * d.3)

-- Calculate vector at t = 4
def vec_at_t4 := (line 4 (11/4, 35/4, 45/2) (-3/4, -11/4, -13/2))

theorem vector_at_t4_is_correct :
  vec_at_t4 = (-1, -9, -17) := by
  sorry

end vector_at_t4_is_correct_l461_461209


namespace trapezoid_divisible_into_equal_triangles_l461_461513

variable (m n : ℕ)
variable (hne : m ≠ n)

theorem trapezoid_divisible_into_equal_triangles : ∃ (k : ℕ), k > 0 ∧
  (∀ (tpezoid : Trapezoid), 
    tpezoid.base₁ = m ∧ tpezoid.base₂ = n → 
    tpezoid.divisible_into_equal_triangles k) := 
sorry

end trapezoid_divisible_into_equal_triangles_l461_461513


namespace Lenny_has_amount_left_l461_461843

def starting_amount := 500
def tech_store_total_cost_before_discount := 200 + 2 * 50 + 75
def tech_store_discount := 0.20 * tech_store_total_cost_before_discount
def tech_store_cost_after_discount := tech_store_total_cost_before_discount - tech_store_discount
def tech_store_tax := 0.10 * tech_store_cost_after_discount
def tech_store_final_cost := tech_store_cost_after_discount + tech_store_tax

def bookstore_total_books_cost := 25 + 30 + 15
def bookstore_promotion_cost := 25 + 30 -- Pay for two more expensive books
def bookstore_service_fee := 0.02 * bookstore_promotion_cost
def bookstore_final_cost := bookstore_promotion_cost + bookstore_service_fee

def total_spent := tech_store_final_cost + bookstore_final_cost
def amount_left := starting_amount - total_spent

theorem Lenny_has_amount_left : amount_left = 113.90 := by
  -- The proof logic goes here
  sorry

end Lenny_has_amount_left_l461_461843


namespace original_price_of_cycle_l461_461202

-- Define the gain percent and selling price
def gain_percent : ℝ := 0.20
def selling_price : ℝ := 1080

-- Define the original price and the equation based on conditions
def original_price (P : ℝ) := selling_price = P + gain_percent * P

-- State the theorem that proves the original price was Rs. 900
theorem original_price_of_cycle : ∃ P : ℝ, original_price P ∧ P = 900 :=
by {
  let P := 900,
  have h : original_price P,
  {
    simp [original_price, P, gain_percent, selling_price],
    norm_num,
  },
  exact ⟨P, h, rfl⟩,
}

end original_price_of_cycle_l461_461202


namespace max_xy_l461_461980

theorem max_xy (x y c : ℝ) (h : x + y = c - 195) :
  ∃ d, d = 4 ∧ (xy = (c - 195)^2 / 4 ≤ d) :=
by
  obtain ⟨d, hd⟩ := ⟨4, by
    sorry
  ⟩

end max_xy_l461_461980


namespace bobs_highest_success_ratio_l461_461621

def alice_first_day_success_ratio : ℚ := 220 / 400
def alice_second_day_success_ratio : ℚ := 180 / 200
def alice_two_day_total_attempt : ℚ := 600
def alice_two_day_success_ratio : ℚ := 2 / 3

theorem bobs_highest_success_ratio (x y z w : ℕ) 
  (h1 : 0 < x ∧ 0 < z) 
  (h2 : 0 < x / y ∧ x / y < alice_first_day_success_ratio) 
  (h3 : 0 < z / w ∧ z / w < alice_second_day_success_ratio)
  (h4 : y + w = 600) 
  : (x + z) / 600 ≤ 22 / 75 :=
by sorry

end bobs_highest_success_ratio_l461_461621


namespace cost_per_piece_l461_461079

variable (totalCost : ℝ) (numberOfPizzas : ℝ) (piecesPerPizza : ℝ)

theorem cost_per_piece (h1 : totalCost = 80) (h2 : numberOfPizzas = 4) (h3 : piecesPerPizza = 5) :
  totalCost / numberOfPizzas / piecesPerPizza = 4 := by
sorry

end cost_per_piece_l461_461079


namespace initial_ratio_of_partners_to_associates_l461_461205

theorem initial_ratio_of_partners_to_associates
  (P : ℕ) (A : ℕ)
  (hP : P = 18)
  (h_ratio_after_hiring : ∀ A, 45 + A = 18 * 34) :
  (P : ℤ) / (A : ℤ) = 2 / 63 := 
sorry

end initial_ratio_of_partners_to_associates_l461_461205


namespace find_multiplier_l461_461564

-- Define the variables x and y
variables (x y : ℕ)

-- Define the conditions
def condition1 := (x / 6) * y = 12
def condition2 := x = 6

-- State the theorem to prove
theorem find_multiplier (h1 : condition1 x y) (h2 : condition2 x) : y = 12 :=
sorry

end find_multiplier_l461_461564


namespace det_product_l461_461802

variable {n : Type} [DecidableEq n] [Fintype n]
variable {R : Type} [CommRing R]
variable (C D : Matrix n n R)

theorem det_product (hC : det C = 3) (hD : det D = 8) : det (C ⬝ D) = 24 :=
by
  sorry

end det_product_l461_461802


namespace ellipse_geometry_l461_461453

theorem ellipse_geometry
  (a b : ℝ)
  (h1 : 0 < b)
  (h2 : b < a)
  (P Q : ℝ × ℝ)
  (hP : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (hQ : Q.1^2 / a^2 + Q.2^2 / b^2 = 1)
  (O : ℝ × ℝ := (0, 0))
  (A : ℝ × ℝ := (-a, 0))
  (R : ℝ × ℝ)
  (hAQ_parallel_OP : (Q.2 - A.2) / (Q.1 - A.1) = P.2 / P.1)
  (hR_intersect_y_axis : R.1 = 0 ∧ (Q.2 - A.2) / (Q.1 - A.1) = (R.2 - A.2) / (R.1 - A.1)) :
  (sqrt ((Q.1 + a)^2 + (Q.2 - 0)^2) * sqrt ((0 + a)^2 + (R.2 - 0)^2)) / (sqrt (P.1^2 + P.2^2)^2) = 2 := by
  sorry

end ellipse_geometry_l461_461453


namespace evaluate_floor_ceiling_product_l461_461683

theorem evaluate_floor_ceiling_product :
  (Int.floor (-5 - 0.5) * Int.ceil (5 + 0.5) *
   Int.floor (-4 - 0.5) * Int.ceil (4 + 0.5) *
   Int.floor (-3 - 0.5) * Int.ceil (3 + 0.5) *
   Int.floor (-2 - 0.5) * Int.ceil (2 + 0.5) *
   Int.floor (-1 - 0.5) * Int.ceil (1 + 0.5)) = -518400 := by
  sorry

end evaluate_floor_ceiling_product_l461_461683


namespace length_of_each_song_l461_461233

-- Conditions
def first_side_songs : Nat := 6
def second_side_songs : Nat := 4
def total_length_of_tape : Nat := 40

-- Definition of length of each song
def total_songs := first_side_songs + second_side_songs

-- Question: Prove that each song is 4 minutes long
theorem length_of_each_song (h1 : first_side_songs = 6) 
                            (h2 : second_side_songs = 4) 
                            (h3 : total_length_of_tape = 40) 
                            (h4 : total_songs = first_side_songs + second_side_songs) : 
  total_length_of_tape / total_songs = 4 :=
by
  sorry

end length_of_each_song_l461_461233


namespace range_of_a_l461_461459

open Set

variable {α : Type*} [LinearOrderedField α]

def p (x : α) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q (x a : α) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (A B : Set α) (a : α) :
  (∀ x, p x → q x a) →
  A = { x | 1 / 2 ≤ x ∧ x ≤ 1 } →
  B = { x | a ≤ x ∧ x ≤ a + 1 } →
  p x → q x a :=
by sorry

end range_of_a_l461_461459


namespace number_of_invertible_integers_mod_15_l461_461272

theorem number_of_invertible_integers_mod_15 :
  (finset.card {a ∈ finset.range 15 | Int.gcd a 15 = 1}) = 8 := by
  sorry

end number_of_invertible_integers_mod_15_l461_461272


namespace circle_no_obtuse_prob_l461_461298

noncomputable def probability_no_obtuse_triangle (points : list Point) : ℝ :=
  -- Assumes points is a list of four randomly chosen points on the circle
  if points.length = 4 then
    1 / 108
  else
    0 

theorem circle_no_obtuse_prob :
  ∀ (points : list Point), points.length = 4 → 
    probability_no_obtuse_triangle points = 1 / 108 :=
by sorry

end circle_no_obtuse_prob_l461_461298


namespace smallest_five_digit_negative_int_mod_17_l461_461557

theorem smallest_five_digit_negative_int_mod_17 :
  ∃ k : ℤ, k ≡ 5 [ZMOD 17] ∧ -10000 ≤ k ∧ k < -9999 → k = -10013 :=
begin
  sorry
end

end smallest_five_digit_negative_int_mod_17_l461_461557


namespace markers_in_desk_l461_461987

theorem markers_in_desk (pens pencils markers : ℕ) 
  (h_ratio : pens = 2 * pencils ∧ pens = 2 * markers / 5) 
  (h_pens : pens = 10) : markers = 25 :=
by
  sorry

end markers_in_desk_l461_461987


namespace count_coprime_to_15_l461_461262

def coprime_to_15 (a : ℕ) : Prop := Nat.gcd 15 a = 1

theorem count_coprime_to_15 : 
  (Finset.filter coprime_to_15 (Finset.range 15)).card = 8 := by
  sorry

end count_coprime_to_15_l461_461262


namespace odd_expressions_l461_461066

-- Let's define p and q as positive odd integers
variables (p q : ℕ)

-- Assume that p and q are odd (using the fact from Nat.odd)
theorem odd_expressions (hp : p % 2 = 1) (hq : q % 2 = 1) :
  (pq + 2) % 2 = 1 ∧ (p^3 * q + q^2) % 2 = 1 :=
begin
  sorry
end

end odd_expressions_l461_461066


namespace system_of_inequalities_solution_set_quadratic_equation_when_m_is_2_l461_461143

theorem system_of_inequalities_solution_set : 
  (∀ x : ℝ, (2 * x - 1 < 7) → (x + 1 > 2) ↔ (1 < x ∧ x < 4)) := 
by 
  sorry

theorem quadratic_equation_when_m_is_2 : 
  (∀ x : ℝ, x^2 - 2 * x - 2 = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3)) := 
by 
  sorry

end system_of_inequalities_solution_set_quadratic_equation_when_m_is_2_l461_461143


namespace count_coprime_to_15_eq_8_l461_461265

def is_coprime_to_15 (a : ℕ) : Prop := Nat.gcd a 15 = 1

def count_coprime_to_15 (n : ℕ) : ℕ :=
  (Finset.filter (λ a, is_coprime_to_15 a) (Finset.range n)).card

theorem count_coprime_to_15_eq_8 : count_coprime_to_15 15 = 8 := by
  sorry

end count_coprime_to_15_eq_8_l461_461265


namespace find_ff_inv_9_l461_461363

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.logb 3 x else 2^x

theorem find_ff_inv_9 : f(f (1 / 9)) = 1 / 4 :=
by
  sorry

end find_ff_inv_9_l461_461363


namespace sum_first_n_even_numbers_sum_first_n_odd_numbers_l461_461973

theorem sum_first_n_even_numbers (n : ℕ) : 
  (Finset.range n).sum (λ k, 2 * (k + 1)) = n * (n + 1) := 
sorry

theorem sum_first_n_odd_numbers (n : ℕ) : 
  (Finset.range n).sum (λ k, 2 * k + 1) = n * n :=
sorry

end sum_first_n_even_numbers_sum_first_n_odd_numbers_l461_461973


namespace count_integers_satisfying_condition_l461_461792

theorem count_integers_satisfying_condition :
  ({n : ℕ | 300 < n^2 ∧ n^2 < 1000}.card = 14) :=
by
  sorry

end count_integers_satisfying_condition_l461_461792


namespace find_z_l461_461427

-- Define the given angles
def angle_ABC : ℝ := 95
def angle_BAC : ℝ := 65

-- Define the angle sum property for triangle ABC
def angle_sum_triangle_ABC (a b : ℝ) : ℝ := 180 - (a + b)

-- Define the angle DCE as equal to angle BCA
def angle_DCE : ℝ := angle_sum_triangle_ABC angle_ABC angle_BAC

-- Define the angle sum property for right triangle CDE
def z (dce : ℝ) : ℝ := 90 - dce

-- State the theorem to be proved
theorem find_z : z angle_DCE = 70 :=
by
  -- Statement for proof is provided
  sorry

end find_z_l461_461427


namespace probability_correct_statements_l461_461476

theorem probability_correct_statements :
  let P_A := 1/2
  let P_B := 1/3
  (P_A * P_B = 1/6) ∧ (1 - P_A * (1 - P_B) = 2/3) :=
by 
  have P_A := 1/2
  have P_B := 1/3
  split
  · exact by norm_num
  · exact by norm_num

end probability_correct_statements_l461_461476


namespace sum_diagonal_integers_l461_461932

-- Definition of the specific rectangular spiral movement rules as per the conditions
def spiral_coords (n : ℕ) : ℕ × ℕ :=
  match n with
  | 0     => (0, 0)
  | n + 1 =>
    let prev := spiral_coords n
    if n % 4 == 0 then (prev.1 + (n + 1) / 4, prev.2)
    else if n % 4 == 1 then (prev.1, prev.2 - (n + 1) / 4)
    else if n % 4 == 2 then (prev.1 - (n + 1) / 4, prev.2)
    else (prev.1, prev.2 + (n + 1) / 4) -- n % 4 == 3

-- Function to check if a coordinate (x, y) is on the line y = -x
def on_diagonal (coord : ℕ × ℕ) : Prop :=
  coord.2 = - coord.1

-- Function to find the integers situated on the line y = -x
def diagonal_values (up_to : ℕ) : List ℕ :=
  List.filterMap (λ n => let coord := spiral_coords (n - 1)
                         if on_diagonal coord then some n else none)
                 (List.range (up_to + 1))

-- Sum of integers situated on the line y = -x
def diagonal_sum (up_to : ℕ) : ℕ :=
  (diagonal_values up_to).sum

-- The final theorem statement
theorem sum_diagonal_integers : diagonal_sum 1000 = 10944 :=
by
  -- ... proof (excluded as per instruction)
  sorry

end sum_diagonal_integers_l461_461932


namespace part1_part2_l461_461353

-- Definition of the function f.
def f (x: ℝ) : ℝ := 2 * Real.log x + 1

-- Definition of the function g.
def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

-- Part 1: Prove that c ≥ -1 given f(x) ≤ 2x + c.
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
by
  -- Proof is omitted.
  sorry

-- Part 2: Prove that g(x) is monotonically decreasing on (0, a) and (a, +∞) given a > 0.
theorem part2 (a : ℝ) : a > 0 → (∀ x : ℝ, x > 0 → x ≠ a → 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo 0 a → x2 ∈ Ioo 0 a → x1 < x2 → g x2 a < g x1 a) ∧ 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x2 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x1 < x2 → g x2 a < g x1 a)) :=
by
  -- Proof is omitted.
  sorry

end part1_part2_l461_461353


namespace geometric_sequence_first_term_l461_461117

noncomputable def first_term_of_geometric_sequence (a r : ℝ) : ℝ :=
  a

theorem geometric_sequence_first_term 
  (a r : ℝ)
  (h1 : a * r^3 = 720)   -- The fourth term is 6!
  (h2 : a * r^6 = 5040)  -- The seventh term is 7!
  : first_term_of_geometric_sequence a r = 720 / 7 :=
sorry

end geometric_sequence_first_term_l461_461117


namespace leos_weight_l461_461172

variables (L K J : ℝ)

-- Conditions
def condition1 : Prop := L + 10 = 1.5 * K
def condition2 : Prop := J = 1.3 * (L + K)
def condition3 : Prop := L + K + J = 250

-- Theorem statement
theorem leos_weight (h1 : condition1 L K) (h2 : condition2 L K J) (h3 : condition3 L K J) : L ≈ 61.22 := 
sorry

end leos_weight_l461_461172


namespace initial_principal_amount_l461_461501

theorem initial_principal_amount
  (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (P : ℝ)
  (hA : A = 8400) 
  (hr : r = 0.05)
  (hn : n = 1) 
  (ht : t = 1) 
  (hformula : A = P * (1 + r / n) ^ (n * t)) : 
  P = 8000 :=
by
  rw [hA, hr, hn, ht] at hformula
  sorry

end initial_principal_amount_l461_461501


namespace point_on_ellipse_distances_l461_461446

noncomputable def P_on_ellipse := {P : ℝ × ℝ // (P.1^2 / 169) + (P.2^2 / 144) = 1}

def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  let c := real.sqrt (a^2 - b^2) in ((c, 0), (-c, 0))

theorem point_on_ellipse_distances (P : P_on_ellipse) (a b : ℝ) (h_a : a = 13) 
  (F1 F2 : ℝ × ℝ) (h_foci : foci a b = (F1, F2)) (h_ellipse : P.val.1^2 / 169 + P.val.2^2 / 144 = 1) 
  (h_4 : dist P.val F1 = 4) : dist P.val F2 = 22 :=
by {
  sorry
}

end point_on_ellipse_distances_l461_461446


namespace abs_d_eq_40_l461_461247

theorem abs_d_eq_40 
  (a b c d e : ℤ)
  (h : a * (3 + 1 * complex.i)^4 + b * (3 + 1 * complex.i)^3 + c * (3 + 1 * complex.i)^2 + d * (3 + 1 * complex.i) + e = 0)
  (coprime : is_coprime (a :: b :: c :: d :: e :: list.nil)) :
  |d| = 40 :=
sorry

end abs_d_eq_40_l461_461247


namespace minimum_value_of_abs_z_l461_461059

open Complex

noncomputable def distance (z w : ℂ) := complex.abs (z - w)

theorem minimum_value_of_abs_z (z : ℂ) 
  (h : distance z (2 * I) + distance z 5 = 7) : 
  complex.abs z = 10 / 7 :=
sorry

end minimum_value_of_abs_z_l461_461059


namespace floor_e_equals_2_l461_461690

theorem floor_e_equals_2 : Int.floor Real.exp = 2 := 
sorry

end floor_e_equals_2_l461_461690


namespace strictly_incr_seq_exists_l461_461584

theorem strictly_incr_seq_exists (M : ℝ) (hM : M > 2) :
  ∃ (a : ℕ → ℕ), 
    (∀ i : ℕ, i > 0 → a i > ⌊M^i⌋ ∧
    (∀ n : ℤ, n ≠ 0 → ∃ m : ℕ, ∃ b : ℕ → ℤ, 
      (∀ j, j > 0 → j ≤ m → b j = -1 ∨ b j = 1) ∧ 
      n = ∑ k in finset.range m, b k * a k)) :=
by sorry

end strictly_incr_seq_exists_l461_461584


namespace boy_can_ensure_last_two_candies_from_same_box_l461_461581

theorem boy_can_ensure_last_two_candies_from_same_box (n : ℕ) (h : n > 0) :
  ∃ (f : Fin 2n → Fin n), (∀ i : Fin 2n, (f i ≠ f (i + 1) % 2n) → i < 2n - 2) := sorry

end boy_can_ensure_last_two_candies_from_same_box_l461_461581


namespace simplify_expression_find_difference_l461_461491

-- Define the conditions for alpha and beta being acute angles and their sine values.
variables {α β : ℝ}
variable h1 : 0 < α ∧ α < π / 2
variable h2 : 0 < β ∧ β < π / 2
variable h3 : sin α = 2*sqrt 5 / 5
variable h4 : sin β = sqrt 10 / 10

-- Proof problem for the first question.
theorem simplify_expression (α : ℝ) (h1 : 0 < α ∧ α < π / 2) : 
  (2 * cos α ^ 2 - 1) / (2 * tan ((π / 4) - α) * sin ((π / 4) + α) ^ 2) = 1 :=
sorry

-- Proof problem for the second question.
theorem find_difference (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : sin α = 2*sqrt 5 / 5) 
  (h4 : sin β = sqrt 10 / 10) : 
  α - β = π / 4 :=
sorry

end simplify_expression_find_difference_l461_461491


namespace trajectory_of_Q_l461_461318

-- Define Circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define Line l
def lineL (x y : ℝ) : Prop := x + y = 2

-- Define Conditions based on polar definitions
def polarCircle (ρ θ : ℝ) : Prop := ρ = 2

def polarLine (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 2

-- Define points on ray OP
def pointP (ρ₁ θ : ℝ) : Prop := ρ₁ = 2 / (Real.cos θ + Real.sin θ)
def pointR (ρ₂ θ : ℝ) : Prop := ρ₂ = 2

-- Prove the trajectory of Q
theorem trajectory_of_Q (O P R Q : ℝ × ℝ)
  (ρ₁ θ ρ ρ₂ : ℝ)
  (h1: circleC O.1 O.2)
  (h2: lineL P.1 P.2)
  (h3: polarCircle ρ₂ θ)
  (h4: polarLine ρ₁ θ)
  (h5: ρ * ρ₁ = ρ₂^2) :
  ρ = 2 * (Real.cos θ + Real.sin θ) :=
by
  sorry

end trajectory_of_Q_l461_461318


namespace max_value_l461_461456

noncomputable def max_value_expr (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^3 * y + x^2 * y + x * y + x * y^2

theorem max_value (x y : ℝ) (h : x + y = 5) : 
  max_value_expr x y h ≤ 1175 / 16 :=
begin
  sorry  -- Proof will go here
end

end max_value_l461_461456


namespace num_games_round_robin_l461_461536

-- There are 10 classes in the second grade, each class forms one team.
def num_teams := 10

-- A round-robin format means each team plays against every other team once.
def num_games (n : Nat) := n * (n - 1) / 2

-- Proving the total number of games played with num_teams equals to 45
theorem num_games_round_robin : num_games num_teams = 45 := by
  sorry

end num_games_round_robin_l461_461536


namespace fraction_students_received_A_or_B_equal_97_over_130_l461_461007

theorem fraction_students_received_A_or_B_equal_97_over_130 :
  (let class1_students := 100,
       class1_As := 0.4,
       class1_Bs := 0.3,
       class2_students := 150,
       class2_As := 0.5,
       class2_Bs := 0.25,
       class3_students := 75,
       class3_As := 0.6,
       class3_Bs := 0.2,
       total_students := class1_students + class2_students + class3_students,
       students_received_As := (class1_students * class1_As) + (class2_students * class2_As) + (class3_students * class3_As),
       students_received_Bs := (class1_students * class1_Bs) + (class2_students * class2_Bs) + (class3_students * class3_Bs),
       students_received_A_or_B := students_received_As + students_received_Bs,
       fraction_A_or_B := students_received_A_or_B / total_students)
   in fraction_A_or_B = 97 / 130 :=
by
  sorry

end fraction_students_received_A_or_B_equal_97_over_130_l461_461007


namespace smallest_perfect_square_divisible_by_5_and_7_l461_461558

theorem smallest_perfect_square_divisible_by_5_and_7 
  (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = k^2)
  (h3 : 5 ∣ n)
  (h4 : 7 ∣ n) : 
  n = 1225 :=
sorry

end smallest_perfect_square_divisible_by_5_and_7_l461_461558


namespace minimum_value_of_f_minimum_value_at_7_l461_461302

-- Define the function
def f (x : ℝ) : ℝ := (x^2 - 4*x + 9) / (x - 4)

-- Define the condition
axiom hx : ∀ x, x ≥ 5

-- Prove the minimum value of f(x) is 10
theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5) : f x ≥ 10 :=
sorry

-- Prove that the minimum value is achieved at x = 7
theorem minimum_value_at_7 : f 7 = 10 :=
sorry

end minimum_value_of_f_minimum_value_at_7_l461_461302


namespace sum_first_n_terms_l461_461742

variable {a : ℕ → ℝ}
variable {n : ℕ}

-- Conditions of the problem
def common_difference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k, a (k + 1) = a k + d

def is_geometric_mean (x y z : ℝ) : Prop :=
  y * y = x * z

-- Hypotheses
def hypotheses (a : ℕ → ℝ) : Prop :=
  common_difference a (-2) ∧ is_geometric_mean (a 1) (a 3) (a 4)

-- Sum of the first n terms
def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a i

theorem sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) (h : hypotheses a) : 
  S_n a n = - (n : ℝ)^2 + 9 * (n : ℝ) :=
sorry

end sum_first_n_terms_l461_461742


namespace photo_arrangement_count_l461_461140
-- Import the necessary library

-- Define the conditions and the theorem statement
theorem photo_arrangement_count :
  let boys := 3
  let girls := 3
  let teacher := 1
  let total_positions := boys + girls + teacher
  let middle_position := teacher
  let adjacent_condition := "those adjacent to the teacher cannot be boys or girls at the same time"
  in
  (∃ count : ℕ, count = 432) := sorry

end photo_arrangement_count_l461_461140


namespace rectangle_perimeter_l461_461503

theorem rectangle_perimeter (A W : ℝ) (hA : A = 300) (hW : W = 15) : 
  (2 * ((A / W) + W)) = 70 := 
  sorry

end rectangle_perimeter_l461_461503


namespace determine_function_l461_461307

theorem determine_function (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2 - 1) : 
  ∀ x : ℝ, f x = x^2 - 2x :=
sorry

end determine_function_l461_461307


namespace logical_equivalence_l461_461977

variable (P Q R : Prop)

theorem logical_equivalence : (P ∧ R → ¬Q) ↔ (Q → ¬P ∨ ¬R) :=
begin
  sorry,
end

end logical_equivalence_l461_461977


namespace square_area_from_circle_l461_461596

-- Define the conditions for the circle's equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 8 * x - 8 * y + 28 

-- State the main theorem to prove the area of the square
theorem square_area_from_circle (x y : ℝ) (h : circle_equation x y) :
  ∃ s : ℝ, s^2 = 88 :=
sorry

end square_area_from_circle_l461_461596


namespace coordinate_equation_solution_l461_461669

theorem coordinate_equation_solution (x y : ℝ) :
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 →
  (y = -x - 2) ∨ (y = -2 * x + 1) :=
by
  sorry

end coordinate_equation_solution_l461_461669


namespace calculate_expression_l461_461237

theorem calculate_expression : (3/5)^5 * (2/3)^(-4) = 19683/50000 := by
  sorry

end calculate_expression_l461_461237


namespace count_integers_divisible_by_105_l461_461162

theorem count_integers_divisible_by_105 :
  let LCM := Nat.lcm (Nat.lcm 3 5) 7 in
  ∀ (a b : ℕ), a = 1 → b = 1000 → 
  Nat.card {x : ℕ | a ≤ x ∧ x ≤ b ∧ x % LCM = 0} = 9 :=
by
  intros LCM _ a b h_a h_b
  rw [h_a, h_b]
  have : LCM = 105 := by sorry
  sorry

end count_integers_divisible_by_105_l461_461162


namespace simplified_product_l461_461709

theorem simplified_product :
  ((1 - (1 / 2)) * (1 - (1 / 3)) * (1 - (1 / 4)) * ... * (1 - (1 / 150)))
  = (1 / 150) :=
sorry

end simplified_product_l461_461709


namespace correct_calculation_l461_461165

noncomputable def option_A : Prop := (Real.sqrt 3 + Real.sqrt 2) ≠ Real.sqrt 5
noncomputable def option_B : Prop := (Real.sqrt 3 * Real.sqrt 5) = Real.sqrt 15 ∧ Real.sqrt 15 ≠ 15
noncomputable def option_C : Prop := Real.sqrt (32 / 8) = 2 ∧ (Real.sqrt (32 / 8) ≠ -2)
noncomputable def option_D : Prop := (2 * Real.sqrt 3) - Real.sqrt 3 = Real.sqrt 3

theorem correct_calculation : option_D :=
by
  sorry

end correct_calculation_l461_461165


namespace arithmetic_sequence_sum_l461_461016

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
    (h_arith_seq : ∀ n, a (n + 1) = a n + d)
    (h_a5 : a 5 = 3)
    (h_a6 : a 6 = -2) :
  a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = -49 :=
by
  sorry

end arithmetic_sequence_sum_l461_461016


namespace rhombus_triangle_area_correct_l461_461156

noncomputable def rhombus_triangle_area (d1 d2 : ℝ) (x : ℝ) : ℝ :=
  (1/2) * (d1/2) * (d2/2) * sin x

theorem rhombus_triangle_area_correct (x : ℝ) : rhombus_triangle_area 15 20 x = 37.5 * sin x :=
by
  unfold rhombus_triangle_area
  ring
  norm_num
  sorry

end rhombus_triangle_area_correct_l461_461156


namespace sue_probability_l461_461906

theorem sue_probability :
  let total_shoes := 24;
  let blue_pairs := 7;
  let red_pairs := 3;
  let green_pairs := 2;
  let blue_shoes := 2 * blue_pairs;
  let red_shoes := 2 * red_pairs;
  let green_shoes := 2 * green_pairs;
  let p_blue := (blue_shoes / total_shoes) * ((blue_pairs) / (total_shoes - 1));
  let p_red := (red_shoes / total_shoes) * ((red_pairs) / (total_shoes - 1));
  let p_green := (green_shoes / total_shoes) * ((green_pairs) / (total_shoes - 1));
  p_blue + p_red + p_green = 31 / 138 :=
by
  let total_shoes := 24
  let blue_pairs := 7
  let red_pairs := 3
  let green_pairs := 2
  let blue_shoes := 2 * blue_pairs
  let red_shoes := 2 * red_pairs
  let green_shoes := 2 * green_pairs
  let p_blue := (blue_shoes.toRat / total_shoes.toRat) * ((blue_pairs.toRat) / (total_shoes.toRat - 1))
  let p_red := (red_shoes.toRat / total_shoes.toRat) * ((red_pairs.toRat) / (total_shoes.toRat - 1))
  let p_green := (green_shoes.toRat / total_shoes.toRat) * ((green_pairs.toRat) / (total_shoes.toRat - 1))
  have : p_blue + p_red + p_green = 31 / 138 := sorry
  exact this

end sue_probability_l461_461906


namespace problem_1_problem_2_problem_3_l461_461511

-- Define the odd function condition and prove that a = -1
def isOddFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_1 (g : ℝ → ℝ) (a : ℝ)
  (h : ∀ x, g x = log (1/2) (1 - a * x) / (x - 1)) :
  isOddFunction g → a = -1 :=
by
  sorry

-- Define the upper bounds condition and prove the set of all upper bounds
theorem problem_2 (g : ℝ → ℝ)
  (h : ∀ x, g x = log (1/2) ((1 + x) / (x - 1))) :
  set_of (λ M, ∀ x ∈ set.Icc (5 / 3) 3, |g x| ≤ M) = set.Ici 2 :=
by
  sorry

-- Define the conditions for f(x) being bounded and prove the range of a
theorem problem_3 (f : ℝ → ℝ) (a : ℝ)
  (h : ∀ x, f x = 1 + a * (1 / 2) ^ x + (1 / 4) ^ x)
  (h_bounded : ∀ x ∈ set.Ici 0, |f x| ≤ 3) :
  -5 ≤ a ∧ a ≤ 1 :=
by
  sorry

end problem_1_problem_2_problem_3_l461_461511


namespace modulo_calculation_l461_461853

theorem modulo_calculation (n : ℕ) (hn : 0 ≤ n ∧ n < 19) (hmod : 5 * n % 19 = 1) : 
  ((3^n)^2 - 3) % 19 = 3 := 
by 
  sorry

end modulo_calculation_l461_461853


namespace poly_divisible_by_30_poly_coeffs_range_l461_461663

def poly_f (x : ℤ) := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 7) * (x - 8) * (x - 9)

theorem poly_divisible_by_30 : ∀ n : ℕ, 30 ∣ poly_f n := 
sorry

theorem poly_coeffs_range : ∀ i, i ≤ 10 → (coeff poly_f i) ∈ {-1, 0, 1} := 
sorry

end poly_divisible_by_30_poly_coeffs_range_l461_461663


namespace arthur_first_day_spending_l461_461230

-- Define the costs of hamburgers and hot dogs.
variable (H D : ℝ)
-- Given conditions
axiom hot_dog_cost : D = 1
axiom second_day_purchase : 2 * H + 3 * D = 7

-- Goal: How much did Arthur spend on the first day?
-- We need to verify that 3H + 4D = 10
theorem arthur_first_day_spending : 3 * H + 4 * D = 10 :=
by
  -- Validating given conditions
  have h1 := hot_dog_cost
  have h2 := second_day_purchase
  -- Insert proof here
  sorry

end arthur_first_day_spending_l461_461230


namespace grain_demand_l461_461399

variable (F : ℝ)
def S0 : ℝ := 1800000 -- base supply value

theorem grain_demand : ∃ D : ℝ, S = 0.75 * D ∧ S = S0 * (1 + F) ∧ D = (1800000 * (1 + F) / 0.75) :=
by
  sorry

end grain_demand_l461_461399


namespace number_is_7625_l461_461083

-- We define x as a real number
variable (x : ℝ)

-- The condition given in the problem
def condition : Prop := x^2 + 95 = (x - 20)^2

-- The theorem we need to prove
theorem number_is_7625 (h : condition x) : x = 7.625 :=
by
  sorry

end number_is_7625_l461_461083


namespace coefficients_of_one_factor_less_than_a_l461_461444

variable {a : ℕ} (p : Polynomial ℤ)
variable (h1 : p.degree = 21)
variable (h2 : ∀ x ∈ p.root_set ℝ, abs x < 1 / 3)
variable (h3 : ∀ i, p.coeff i ∈ (Icc (-2019 * a : ℤ) (2019 * a : ℤ)))

theorem coefficients_of_one_factor_less_than_a (hp : irreducible p) : 
  ∃ q : Polynomial ℤ, q ∣ p ∧ ∀ i, abs (q.coeff i) < a := 
sorry

end coefficients_of_one_factor_less_than_a_l461_461444


namespace angle_PMN_is_60_l461_461419

-- Define given variables and their types
variable (P M N R Q : Prop)
variable (angle : Prop → Prop → Prop → ℝ)

-- Given conditions
variables (h1 : angle P Q R = 60)
variables (h2 : PM = MN)

-- The statement of what's to be proven
theorem angle_PMN_is_60 :
  angle P M N = 60 := sorry

end angle_PMN_is_60_l461_461419


namespace visible_point_exists_l461_461379

noncomputable def construct_visible_point (a b : line_segment) (α β : Real.Angle) : Set Point :=
  {P : Point | ∃ (L₁ L₂ : Set Point),
    (L₁ = locus_of_points_subtending_angle a α ∧
     L₂ = locus_of_points_subtending_angle b β ∧
     P ∈ L₁ ∧ P ∈ L₂)}

theorem visible_point_exists (a b : line_segment) (α β : Real.Angle) :
  ∃ (P : Point), P ∈ construct_visible_point a b α β :=
sorry

end visible_point_exists_l461_461379


namespace max_area_of_triangle_l461_461428

open Real

def point := ℝ × ℝ

def on_circle (C : point → Prop) (A : point) : Prop :=
  C A

def distance (P Q : point) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def line_eq_slope (P Q : point) (m : ℝ) : Prop :=
  (Q.2 - P.2) = m * (Q.1 - P.1)

noncomputable def area_of_triangle (P A B : point) : ℝ :=
  abs ((A.1 - P.1) * (B.2 - P.2) - (B.1 - P.1) * (A.2 - P.2)) / 2

def is_equidistant (P A B : point) : Prop :=
  distance P A = distance P B

theorem max_area_of_triangle
  (P A B : point)
  (C : point → Prop)
  (on_C_A : on_circle C A)
  (on_C_B : on_circle C B)
  (PA_eq_PB : is_equidistant P A B)
  (C_eq : ∀ (X : point), C X ↔ ((X.1)^2 + (X.2 - 1/2)^2 = 36)) :
  (∀ A B, (on_circle C A) ∧ (on_circle C B) → distance A B ≤ 12) →
  (distance A B = 12 → area_of_triangle P A B = 6) :=
by
  intros h1 h2
  sorry

end max_area_of_triangle_l461_461428


namespace construct_parabola_focus_l461_461250

theorem construct_parabola_focus 
  (D : Set Point) 
  (T1 T2 : Line) : 
  ∃ F : Point, is_focus_of_parabola_with_directrix_and_tangents F D T1 T2 := 
sorry

end construct_parabola_focus_l461_461250


namespace ratio_EG_to_FH_l461_461089

-- Define the points and distances given in our problem
variables (E F G H : Type) [linear_ordered_field H]
variables (EF FG EH : H)

-- Specify the conditions
def EF_len := 3
def FG_len := 6
def EH_len := 16

-- Define the equality for the required ratio
theorem ratio_EG_to_FH : (EG == EF + FG) ∧ (FH == EH - FG) ∧ (EG / FH == 9 / 10) :=
by
  -- Use the given conditions
  have h1 : EF = EF_len, from rfl
  have h2 : FG = FG_len, from rfl
  have h3 : EH = EH_len, from rfl
  sorry

end ratio_EG_to_FH_l461_461089


namespace choir_members_count_l461_461934

theorem choir_members_count : 
  ∃ n : ℕ, 120 ≤ n ∧ n ≤ 300 ∧
    n % 6 = 1 ∧
    n % 8 = 5 ∧
    n % 9 = 2 ∧
    n = 241 :=
by
  -- Proof will follow
  sorry

end choir_members_count_l461_461934


namespace product_of_solutions_abs_eq_l461_461422

theorem product_of_solutions_abs_eq (x : ℝ) :
  ( |x - 5| - 4 = -1 ) → ( x = 8 ∨ x = 2 ) ∧ ( 8 * 2 = 16 ) :=
by 
  intros,
  sorry

end product_of_solutions_abs_eq_l461_461422


namespace coefficient_x3y3_l461_461913

theorem coefficient_x3y3 : 
  let C (n k : ℕ) := Nat.choose n k
  let term_expansion (r : ℕ) := C 5 r * (2 ^ (5 - r)) * x ^ (5 - r) * y ^ r
  ∑ r in {0, 1, 2, 3, 4, 5}, term_expansion r
  let coefs := 4 * C 5 3 - 8 * C 5 2
  (∑ r in {0, 1, 2, 3, 4, 5}, (C 5 r * (2 ^ (5 - r)) * x ^ (5 - r) * y ^ r)) = -40 := by
  sorry

end coefficient_x3y3_l461_461913


namespace mary_sailboat_canvas_l461_461082

def rectangular_sail_area (length width : ℕ) : ℕ :=
  length * width

def triangular_sail_area (base height : ℕ) : ℕ :=
  (base * height) / 2

def total_canvas_area (length₁ width₁ base₁ height₁ base₂ height₂ : ℕ) : ℕ :=
  rectangular_sail_area length₁ width₁ +
  triangular_sail_area base₁ height₁ +
  triangular_sail_area base₂ height₂

theorem mary_sailboat_canvas :
  total_canvas_area 5 8 3 4 4 6 = 58 :=
by
  -- Begin proof (proof steps omitted, we just need the structure here)
  sorry -- end proof

end mary_sailboat_canvas_l461_461082


namespace find_p_l461_461455

theorem find_p (p : ℝ) (h_p_range : 0 < p ∧ p < 1) :
  (∀ t : ℕ, ∃ (a b : ℝ), 
    (∀ x : ℝ, 
      let f := (λ x, p * (3 * x + 1) + (1 - p) * (x / 2)) in 
      f^[t] x = a * t + b)) → 
  p = 1 / 5 := 
sorry

end find_p_l461_461455


namespace floor_e_eq_two_l461_461707

theorem floor_e_eq_two
  (e_approx : Real ≈ 2.718) :
  ⌊e⌋ = 2 :=
sorry

end floor_e_eq_two_l461_461707


namespace rosie_pies_l461_461888

theorem rosie_pies (apples_per_pie : ℕ) (apples_total : ℕ) (pies_initial : ℕ) 
  (h1 : 3 = pies_initial) (h2 : 12 = apples_total) : 
  (36 / (apples_total / pies_initial)) * pies_initial = 27 := 
by
  sorry

end rosie_pies_l461_461888


namespace nonneg_iff_pos_int_solutions_l461_461484

theorem nonneg_iff_pos_int_solutions {a n : ℕ} (x y : ℕ → ℕ) 
  (h1 : a > 0)
  (h2 : ∀ i, y i = x i - 1) :
  (¬ ∃ y_1 y_2 ... y_n : ℕ, (Σ i in finset.range n, (i + 1) * y i) = a - (n * (n + 1)) / 2) ↔ 
  (¬ ∃ x_1 x_2 ... x_n : ℕ, (Σ i in finset.range n, (i + 1) * x i) = a) :=
sorry

end nonneg_iff_pos_int_solutions_l461_461484


namespace unique_solution_real_l461_461490

theorem unique_solution_real {x y : ℝ} (h1 : x * (x + y)^2 = 9) (h2 : x * (y^3 - x^3) = 7) :
  x = 1 ∧ y = 2 :=
sorry

end unique_solution_real_l461_461490


namespace rainfall_second_day_l461_461944

theorem rainfall_second_day
  (x : ℝ)                                            -- the amount of rain on the second day
  (rain_first_day : ℝ) (rain_first_day = 26)          -- the amount of rain on the first day is 26 cm
  (rain_less_third_day : ℝ) (rain_less_third_day = 12) -- the third day has 12 cm less rain than the second day
  (average_rainfall : ℝ) (average_rainfall = 140)     -- average rainfall for the first three days in a normal year
  (rain_less_this_year : ℝ) (rain_less_this_year = 58): -- total rainfall this year is 58 cm less than average
  26 + x + (x - 12) = 140 - 58 → x = 34 :=
by
  sorry

end rainfall_second_day_l461_461944


namespace product_divisible_by_49_l461_461035

theorem product_divisible_by_49 (a b : ℕ) (h : (a^2 + b^2) % 7 = 0) : (a * b) % 49 = 0 :=
sorry

end product_divisible_by_49_l461_461035


namespace range_of_c_monotonicity_g_l461_461360

-- Define the given function f(x)
def f (x : ℝ) := 2 * real.log x + 1

-- Part 1: Define the hypothesis for the range of c
theorem range_of_c :
  ∀ x : ℝ, f(x) ≤ 2 * x + c ↔ c ∈ set.Ici (-1) :=
sorry

-- Part 2: Define the function g(x) and prove its monotonicity
def g (x a : ℝ) [ne_zero : a ≠ 0] := (f(x) - f(a)) / (x - a)

theorem monotonicity_g (a : ℝ) (h : 0 < a) : 
  ∀ x, (0 < x ∧ x < a) ∨ (x > a) → (g x a).deriv < 0 :=
sorry

end range_of_c_monotonicity_g_l461_461360


namespace legos_in_box_at_end_l461_461040

def initial_legos : ℕ := 500
def legos_used : ℕ := initial_legos / 2
def missing_legos : ℕ := 5
def remaining_legos := legos_used - missing_legos

theorem legos_in_box_at_end : remaining_legos = 245 := 
by
  sorry

end legos_in_box_at_end_l461_461040


namespace triangle_area_l461_461616

/-- 
Given a triangle with vertices at (3, 1), (8, -4), and (3, -4), 
prove that the area of the triangle is 12.5 square units.
-/
theorem triangle_area :
  let A := (3 : ℝ, 1 : ℝ)
  let B := (8 : ℝ, -4 : ℝ)
  let C := (3 : ℝ, -4 : ℝ)
  let base := real.dist (3, 1) (3, -4)
  let height := real.dist (8, -4) (3, -4)
  let area := 0.5 * base * height
  area = 12.5 :=
by
  sorry

end triangle_area_l461_461616


namespace sum_third_fifth_l461_461013

noncomputable def seq : ℕ → ℚ
| 1       := 1
| (n + 1) := (n + 1) ^ 3 / (n ^ 3) * seq n

theorem sum_third_fifth :
  seq 3 + seq 5 = 341 / 64 :=
by {
  sorry
}

end sum_third_fifth_l461_461013


namespace ball_bounce_height_l461_461998

theorem ball_bounce_height :
  ∃ k : ℕ, k = 4 ∧ 45 * (1 / 3 : ℝ) ^ k < 2 :=
by 
  use 4
  sorry

end ball_bounce_height_l461_461998


namespace floor_e_equals_two_l461_461685

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end floor_e_equals_two_l461_461685


namespace exists_2005_different_positive_square_numbers_with_square_sum_l461_461433

theorem exists_2005_different_positive_square_numbers_with_square_sum :
  ∃ (S : Finset ℕ), S.card = 2005 ∧ (∀ x ∈ S, ∃ k : ℕ, x = k^2) ∧ ∃ m : ℕ, (∑ x in S, x) = m^2 :=
by
  sorry

end exists_2005_different_positive_square_numbers_with_square_sum_l461_461433


namespace cos_angle_eq_one_third_l461_461245

theorem cos_angle_eq_one_third :
  ∃ P : ℝ × ℝ,
    (P.1^2 / 6 + P.2^2 / 2 = 1) ∧
    (P.1^2 / 3 - P.2 = 1) ∧
    let F₁ : ℝ × ℝ := (-2, 0),
        F₂ : ℝ × ℝ := (2, 0),
        PF₁ := (F₁.1 - P.1, F₁.2 - P.2),
        PF₂ := (F₂.1 - P.1, F₂.2 - P.2) in
    (PF₁.1 * PF₂.1 + PF₁.2 * PF₂.2) /
    (Real.sqrt (PF₁.1 * PF₁.1 + PF₁.2 * PF₁.2) * Real.sqrt (PF₂.1 * PF₂.1 + PF₂.2 * PF₂.2)) = 1/3 :=
by sorry

end cos_angle_eq_one_third_l461_461245


namespace rationalize_denominator_l461_461487

theorem rationalize_denominator : (7 / Real.sqrt 147) = (Real.sqrt 3 / 3) :=
by
  sorry

end rationalize_denominator_l461_461487


namespace triangle_is_isosceles_or_right_angled_l461_461815

-- Define the conditions
variables {A B C : ℝ}
variables {α β γ : ℝ}
variables (h₁ : A + B + C = π)
variables (h₂ : α = A ∨ α = B ∨ α = C)
variables (h₃ : β = A ∨ β = B ∨ β = C)
variables (h₄ : γ = A ∨ γ = B ∨ γ = C)
variables (h₅ : α ≠ β)
variables (h₆ : sin α * cos α = sin β * cos β)

-- Prove that the triangle is isosceles or right-angled
theorem triangle_is_isosceles_or_right_angled :
  (∀ A B C : ℝ, A + B + C = π → 
   (∃ α β γ, (α = A ∨ α = B ∨ α = C) ∧ (β = A ∨ β = B ∨ β = C) ∧ 
            (γ = A ∨ γ = B ∨ γ = C) ∧ α ≠ β ∧ sin α * cos α = sin β * cos β → 
   (∃ α β, α ≠ β ∧ (sin α * cos α = sin β * cos β)) → 
    (∃ A B, (A = 90 ∧ B + C = 90) ∨ (A = B))) 
  :
-- Insert proof here
sorry

end triangle_is_isosceles_or_right_angled_l461_461815


namespace last_five_digits_l461_461921

theorem last_five_digits : (99 * 10101 * 111 * 1001) % 100000 = 88889 :=
by
  sorry

end last_five_digits_l461_461921


namespace correct_statements_l461_461567

-- Definitions for statements A, B, C, and D
def statementA (x : ℝ) : Prop := |x| > 1 → x > 1
def statementB (A B C : ℝ) : Prop := (C > 90) ↔ (A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90))
def statementC (a b : ℝ) : Prop := (a * b ≠ 0) ↔ (a ≠ 0 ∧ b ≠ 0)
def statementD (a b : ℝ) : Prop := a > b → 1 / a < 1 / b

-- Proof problem stating which statements are correct
theorem correct_statements :
  (∀ x : ℝ, statementA x = false) ∧ 
  (∀ (A B C : ℝ), statementB A B C = false) ∧ 
  (∀ (a b : ℝ), statementC a b) ∧ 
  (∀ (a b : ℝ), statementD a b = false) :=
by
  sorry

end correct_statements_l461_461567


namespace indefinite_integral_example_l461_461989

theorem indefinite_integral_example :
  (∫ x in Real, (2 * x^4 + 2 * x^3 - 3 * x^2 + 2 * x - 9) / (x * (x - 1) * (x + 3))) = 
  (λ x: ℝ, x^2 - 2 * x + 3 * Real.log (abs x) - (3 / 2) * Real.log (abs (x - 1)) + (11 / 2) * Real.log (abs (x + 3)) + constant) :=
sorry

end indefinite_integral_example_l461_461989


namespace angle_between_vectors_l461_461319

variable {E : Type} [InnerProductSpace ℝ E] [Nontrivial E]

theorem angle_between_vectors (a b : E)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (h1 : ⟪a - (2 : ℝ) • b, a⟫ = 0)
  (h2 : ⟪b - (2 : ℝ) • a, b⟫ = 0) :
  real.angle.realθ a b = real.pi / (3 : ℝ) :=
sorry

end angle_between_vectors_l461_461319


namespace rosie_can_make_nine_pies_l461_461892

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end rosie_can_make_nine_pies_l461_461892


namespace sum_equivalence_mod_four_l461_461873

theorem sum_equivalence_mod_four (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a + (-1 : ℤ)^b * ∑ m in Finset.range (a + 1), (-1 : ℤ)^(m*nat.floor(b*m/a)))
  % 4 = 
  (b + (-1 : ℤ)^a * ∑ n in Finset.range (b + 1), (-1 : ℤ)^(n*nat.floor(a*n/b)))
  % 4 := sorry

end sum_equivalence_mod_four_l461_461873


namespace pentagon_area_correct_l461_461308

-- Define the side lengths of the pentagon
def side1 := 18
def side2 := 25
def side3 := 30
def side4 := 28
def side5 := 22

-- Define the specific angle between the sides of lengths 30 and 28
def angle := 110 -- degrees

-- Define the heights used for the trapezoids and triangle calculations
def height_trapezoid1 := 10
def height_trapezoid2 := 15
def height_triangle := 8

-- Function to calculate the area of a trapezoid
def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

-- Function to calculate the area of a triangle
def triangle_area (base height : ℕ) : ℕ :=
  base * height / 2

-- Calculation of individual areas
def area_trapezoid1 := trapezoid_area side1 side2 height_trapezoid1
def area_trapezoid2 := trapezoid_area side3 side4 height_trapezoid2
def area_triangle := triangle_area side5 height_triangle

-- Total area calculation
def total_area := area_trapezoid1 + area_trapezoid2 + area_triangle

-- Expected total area
def expected_area := 738

-- Lean statement to assert the total area equals the expected value
theorem pentagon_area_correct :
  total_area = expected_area :=
by sorry

end pentagon_area_correct_l461_461308


namespace find_P_l461_461841

def cube_side_length := 4
def cube_surface_area := 6 * (cube_side_length ^ 2)
def cylinder_surface_area (r : ℝ) := 2 * Real.pi * r^2 + 2 * Real.pi * r * 2 * r
def cylinder_volume (r : ℝ) := Real.pi * r^2 * (2 * r)

theorem find_P (P : ℝ) (r : ℝ) (h : 2 * Real.pi * r^2 + 2 * Real.pi * r * 2 * r = 96)
  (v : cylinder_volume r = P * Real.sqrt 2 / Real.pi) : P = 128 :=
by
  sorry

end find_P_l461_461841


namespace sum_of_coefficients_l461_461665

theorem sum_of_coefficients (a b c d : ℕ) (x : ℝ)
  (h1 : sin x ^ 2 + sin (2 * x) ^ 2 + sin (3 * x) ^ 2 + sin (4 * x) ^ 2 + cos (5 * x) ^ 2 = 3)
  (h2 : cos (a * x) * cos (b * x) * cos (c * x) * cos (d * x) = 0) :
  a + b + c + d = 12 :=
sorry

end sum_of_coefficients_l461_461665


namespace highest_attendance_day_l461_461282

def alice_unavailable (d : Day) : Prop :=
  d = Day.Monday ∨ d = Day.Thursday

def bob_unavailable (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Thursday ∨ d = Day.Friday

def clara_unavailable (d : Day) : Prop :=
  d = Day.Monday ∨ d = Day.Wednesday ∨ d = Day.Thursday

def david_unavailable (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Wednesday ∨ d = Day.Friday

def eve_unavailable (d : Day) : Prop :=
  d = Day.Monday ∨ d = Day.Tuesday

def max_attendees := 3

theorem highest_attendance_day :
  (∀ d : Day, (∀ (h : alice_unavailable d ∨ bob_unavailable d ∨ clara_unavailable d ∨ david_unavailable d ∨ eve_unavailable d), false) → ∃ a b, a ≠ b ∧ (a = Day.Wednesday ∨ a = Day.Thursday) ∧ (b = Day.Wednesday ∨ b = Day.Thursday) ∧ (a = d ∨ b = d)) ∧
  (∀ d : Day, ((alice_unavailable d ∨ bob_unavailable d ∨ clara_unavailable d ∨ david_unavailable d ∨ eve_unavailable d) → false) → 3 ≤ max_attendees) :=
  sorry

end highest_attendance_day_l461_461282


namespace identify_irrational_number_l461_461222

theorem identify_irrational_number :
  (∀ a b : ℤ, (-1 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (0 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  (∀ a b : ℤ, (1 : ℚ) / (2 : ℚ) = (a : ℚ) / (b : ℚ) → b ≠ 0) ∧
  ¬(∃ a b : ℤ, (Real.sqrt 3) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0) :=
sorry

end identify_irrational_number_l461_461222


namespace positive_integers_between_300_and_1000_squared_l461_461795

theorem positive_integers_between_300_and_1000_squared :
  ∃ n : ℕ, 300 < n^2 ∧ n^2 < 1000 → ∃ m : ℕ, m = 14 := sorry

end positive_integers_between_300_and_1000_squared_l461_461795


namespace joe_total_toy_cars_l461_461074

def initial_toy_cars : ℕ := 50
def uncle_additional_factor : ℝ := 1.5

theorem joe_total_toy_cars :
  (initial_toy_cars : ℝ) + uncle_additional_factor * initial_toy_cars = 125 := 
by
  sorry

end joe_total_toy_cars_l461_461074


namespace samia_walking_distance_l461_461488

noncomputable def total_distance (x : ℝ) : ℝ := 4 * x
noncomputable def biking_distance (x : ℝ) : ℝ := 3 * x
noncomputable def walking_distance (x : ℝ) : ℝ := x
noncomputable def biking_time (x : ℝ) : ℝ := biking_distance x / 12
noncomputable def walking_time (x : ℝ) : ℝ := walking_distance x / 4
noncomputable def total_time (x : ℝ) : ℝ := biking_time x + walking_time x

theorem samia_walking_distance : ∀ (x : ℝ), total_time x = 1 → walking_distance x = 2 :=
by
  sorry

end samia_walking_distance_l461_461488


namespace tina_wins_probability_l461_461946

theorem tina_wins_probability (P_lose : ℚ) (h1 : P_lose = 3 / 7) (h2 : ∀ P_win : ℚ, P_win + P_lose = 1): ∃ P_win : ℚ, P_win = 4 / 7 :=
by
  use 1 - P_lose
  rw [h1]
  norm_num
  sorry

end tina_wins_probability_l461_461946


namespace remainder_8x_mod_9_l461_461566

theorem remainder_8x_mod_9 (x : ℕ) (h : x % 9 = 5) : (8 * x) % 9 = 4 :=
by
  sorry

end remainder_8x_mod_9_l461_461566


namespace part1_part2_l461_461345

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.log x + 1

-- Part (1): Prove c ≥ -1 given f(x) ≤ 2x + c
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
  sorry

-- Define g with a > 0
def g (x a : ℝ) : ℝ := (f x - f a) / (x - a)

-- Part (2): Prove g is monotonically decreasing on (0, a) and (a, +∞)
theorem part2 (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, x > 0 → x ≠ a → g x a < g a a) :=
  sorry

end part1_part2_l461_461345


namespace range_of_a_l461_461781

theorem range_of_a (a : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < (π / 2) → a ≤ 1 / Real.sin θ + 1 / Real.cos θ) ↔ a ≤ 2 * Real.sqrt 2 :=
sorry

end range_of_a_l461_461781


namespace part1_c_range_part2_monotonicity_l461_461337

noncomputable def f (x : ℝ) := 2 * Real.log x + 1

theorem part1_c_range (c : ℝ) (x : ℝ) (h : a > 0) : f x ≤ 2 * x + c → c ≥ -1 :=
sorry

noncomputable def g (x a : ℝ) := (f x - f a) / (x - a)

theorem part2_monotonicity (a : ℝ) (h : a > 0) : monotone_decreasing_on g (0, a) ∧ monotone_decreasing_on g (a, +∞) :=
sorry

end part1_c_range_part2_monotonicity_l461_461337


namespace sphere_surface_area_l461_461751

/-- Given points A, B, and C are on the surface of a sphere.
AB = 6, BC = 8, AC = 10. 
The distance from the center O of the sphere to the plane ABC equals half of the radius of the sphere.
Prove that the surface area of the sphere is 400/3 * pi. -/
theorem sphere_surface_area (A B C O : ℝ^3) (R : ℝ)
  (AB : dist A B = 6)
  (BC : dist B C = 8)
  (AC : dist A C = 10)
  (h : (dist_plane_point (mk_plane A B C) O = R / 2)) :
  4 * π * R^2 = (400 / 3) * π := sorry

end sphere_surface_area_l461_461751


namespace coefficient_of_x6_in_expansion_l461_461966

theorem coefficient_of_x6_in_expansion :
  let a := 1
  let b := -3 * (x : ℝ) ^ 3
  let n := 4
  let k := 2
  (1 - 3 * (x : ℝ) ^ 3) ^ 4 = ∑ k in finset.range (n + 1), 
    (nat.choose n k) * a ^ (n - k) * b ^ k →
  is_term_of_degree (1 - 3 * (x : ℝ) ^ 3) ^ 4 x 6 (54 * x ^ 6) :=
by
  sorry

end coefficient_of_x6_in_expansion_l461_461966


namespace complement_union_l461_461322

open Set

def set_A : Set ℝ := {x | x ≤ 0}
def set_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem complement_union (A B : Set ℝ) (hA : A = set_A) (hB : B = set_B) :
  (univ \ (A ∪ B) = {x | 1 < x}) := by
  rw [hA, hB]
  sorry

end complement_union_l461_461322


namespace complex_number_solutions_l461_461383

theorem complex_number_solutions (z : ℂ) (hz : abs z < 10) (h : exp(2 * z) = (z - 1) / (z + 1)) : 
  ∃ n, n = 4 :=
sorry

end complex_number_solutions_l461_461383


namespace pies_from_36_apples_l461_461879

-- Definitions of conditions
def pies_from_apples (apples : Nat) : Nat :=
  apples / 4  -- because 12 apples = 3 pies implies 1 pie = 4 apples

-- Theorem to prove
theorem pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

end pies_from_36_apples_l461_461879


namespace f_is_even_l461_461856

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

variables (g : ℝ → ℝ)

-- Definition that g is even
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- Given assumptions
variables (h_even_g : is_even g)

-- Definition of f
def f (x : ℝ) : ℝ := abs (g (x^4))

-- Proof statement
theorem f_is_even : is_even_function f :=
  by
  sorry

end f_is_even_l461_461856


namespace rosie_can_make_nine_pies_l461_461893

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end rosie_can_make_nine_pies_l461_461893


namespace sum_a_n_l461_461720

def units_digit (n : ℕ) : ℕ := n % 10

def a_n (n : ℕ) : ℕ := units_digit (n ^ (n + 1) ^ (n - 2))

theorem sum_a_n (S : ℕ) : S = ∑ n in finset.range 2018, a_n (n + 1) → S = 5857 :=
begin
  intro h,
  sorry
end

end sum_a_n_l461_461720


namespace FindRadiusOfSphere_l461_461027

section Parallelepiped

variables (a : ℝ)

-- Define the conditions
def RectangularParallelepiped : Prop :=
  ∃ (AB BC AA1 : ℝ), AB = BC ∧ AA1 = a ∧ AB = 2 * a

def PlanePassesThroughPoints : Prop :=
  ∃ (B1 D : ℝ), B1 = D ∧ ∀ line, line // AC

-- Define the conclusion or the answer
def RadiusOfSphere (r : ℝ) :=
  r = (4 - 2 * Real.sqrt 2) / 3 * a

-- Statement combining conditions and the conclusion
theorem FindRadiusOfSphere 
  (h₁ : RectangularParallelepiped a)
  (h₂ : PlanePassesThroughPoints a) :
  ∃ r, RadiusOfSphere a r := 
  sorry

end Parallelepiped

end FindRadiusOfSphere_l461_461027


namespace sector_area_l461_461761

theorem sector_area (r : ℝ) (α : ℝ) (h_r : r = 2) (h_α : α = π / 4) :
  1/2 * r^2 * α = π / 2 :=
by
  subst h_r
  subst h_α
  sorry

end sector_area_l461_461761


namespace total_earnings_correct_l461_461550

noncomputable def total_earnings : ℝ :=
  let earnings1 := 12 * (2 + 15 / 60)
  let earnings2 := 15 * (1 + 40 / 60)
  let earnings3 := 10 * (3 + 10 / 60)
  earnings1 + earnings2 + earnings3

theorem total_earnings_correct : total_earnings = 83.75 := by
  sorry

end total_earnings_correct_l461_461550


namespace partI_l461_461858

noncomputable def f (x : ℝ) : ℝ := abs (1 - 1/x)

theorem partI (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) (h4 : f a = f b) :
  a * b > 1 :=
  sorry

end partI_l461_461858


namespace fractionA_is_simplest_form_l461_461976

variable (x a b y : ℝ)

def fractionA : ℝ := 3 * x / (3 * x - 2)
def fractionB : ℝ := 3 * a / (6 * a + 9 * b)
def fractionC : ℝ := (x - 4) / (16 - x^2)
def fractionD : ℝ := (x * y) / (x * y - x^2)

theorem fractionA_is_simplest_form (x : ℝ) : 
  (∃ u v : ℝ, u * v = 3 * x ∧ v * u = 3 * x - 2) → fractionA x = 3 * x / (3 * x - 2) :=
by 
  sorry

end fractionA_is_simplest_form_l461_461976


namespace range_of_possible_slopes_l461_461514

theorem range_of_possible_slopes (k : ℝ) :
  (∃ P : ℝ × ℝ, P = (- real.sqrt 3, -1) ∧ 
                 ∃ l : ℝ → ℝ, (l = λ x, k * (x + real.sqrt 3) - 1) ∧ 
                 ∃ x y : ℝ, (x, y) ∈ (λ (x y : ℝ), x^2 + y^2 = 1)) →
  0 ≤ k ∧ k ≤ real.sqrt 3 :=
by sorry

end range_of_possible_slopes_l461_461514


namespace find_a_l461_461767

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ :=
  (x + a) * Real.log x

noncomputable def curve_deriv (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x + (x + a) / x

theorem find_a (a : ℝ) (h : curve (x := 1) a = 2) : a = 1 :=
by
  have eq1 : curve 1 0 = (1 + a) * 0 := by sorry
  have eq2 : curve 1 1 = (1 + a) * Real.log 1 := by sorry
  have eq3 : curve_deriv a 1 = Real.log 1 + (1 + a) / 1 := by sorry
  have eq4 : 2 = 1 + a := by sorry
  sorry -- Complete proof would follow here

end find_a_l461_461767


namespace inequality_positive_l461_461508

theorem inequality_positive (x : ℝ) : (1 / 3) * x - x > 0 ↔ (-2 / 3) * x > 0 := 
  sorry

end inequality_positive_l461_461508


namespace sum_of_powers_inequality_l461_461722

theorem sum_of_powers_inequality
  (s : ℕ)
  (x : Fin s → ℝ)
  (h_pos : ∀ i, 0 < x i)
  (h_prod : ∏ i, x i = 1)
  (m n : ℕ)
  (h_mn : m ≥ n) :
  ∑ i, (x i) ^ m ≥ ∑ i, (x i) ^ n := 
sorry

end sum_of_powers_inequality_l461_461722


namespace maximum_value_g_on_interval_l461_461293

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem maximum_value_g_on_interval : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3 := by
  sorry

end maximum_value_g_on_interval_l461_461293


namespace vector_conditions_l461_461320

def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)
def projection (u v : ℝ × ℝ) : ℝ × ℝ := 
  let k := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  (k * v.1, k * v.2)

theorem vector_conditions :
  ∃ (a b : ℝ × ℝ) (λ : ℝ),
    (a = (1, λ)) ∧ (b = (-2, 1)) →
    (λ = -1/2 → parallel a b) ∧ 
    (λ = -1 → projection a b = (-3/5 * b.1, -3/5 * b.2)) :=
by
  sorry

end vector_conditions_l461_461320


namespace average_of_consecutive_integers_l461_461902

theorem average_of_consecutive_integers (n m : ℕ) 
  (h1 : m = (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7) : 
  (n + 6) = (m + (m+1) + (m+2) + (m+3) + (m+4) + (m+5) + (m+6)) / 7 :=
by
  sorry

end average_of_consecutive_integers_l461_461902


namespace number_of_possible_values_of_sum_of_90_element_subsets_l461_461071

theorem number_of_possible_values_of_sum_of_90_element_subsets :
  ∀ (A : finset ℕ), (∀ x ∈ A, x ∈ finset.range 101) → A.card = 90 →
  ∃ n : ℕ, n = 901 ∧
  ∀ S : ℕ, (S = A.sum id ↔ 4095 ≤ S ∧ S ≤ 4995) → n = 901 :=
by
  -- Definitions and conditions
  intro A hA hCard
  
  -- Given that \(S\) is the sum of the elements of \(A\)
  have hSumRange : ∀ S, (S = A.sum id ↔ 4095 ≤ S ∧ S ≤ 4995),
  sorry
  
  -- Prove the number of possible values of \(S\) is 901
  have n_possible_values : 901,
  sorry
  
  -- Combine the results
  use 901,
  split,
  refl,
  exact hSumRange

end number_of_possible_values_of_sum_of_90_element_subsets_l461_461071


namespace num_rel_prime_to_15_l461_461255

theorem num_rel_prime_to_15 : 
  {a : ℕ | a < 15 ∧ Int.gcd 15 a = 1}.card = 8 := by 
  sorry

end num_rel_prime_to_15_l461_461255


namespace knick_knack_weight_l461_461438

theorem knick_knack_weight :
  ∀ (w : ℕ), 
    (∃ (max_weight : ℕ) (hardcover_count : ℕ) (hardcover_weight : ℚ)
       (textbook_count : ℕ) (textbook_weight : ℚ) (knick_knack_count : ℕ)
       (over_limit_weight : ℕ) (total_books_weight : ℚ) (total_weight : ℚ),
       max_weight = 80 ∧
       hardcover_count = 70 ∧
       hardcover_weight = 0.5 ∧
       textbook_count = 30 ∧
       textbook_weight = 2 ∧
       knick_knack_count = 3 ∧
       over_limit_weight = 33 ∧
       total_books_weight = 
          hardcover_count * hardcover_weight + textbook_count * textbook_weight ∧
       total_weight = max_weight + over_limit_weight ∧
       total_weight - total_books_weight = knick_knack_count * w) → 
    w = 6
| _ := sorry

end knick_knack_weight_l461_461438


namespace lines_from_equation_l461_461679

-- Definitions for the conditions
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Equivalent Lean statement to the proof problem
theorem lines_from_equation :
  (∀ x y : ℝ, satisfies_equation x y → (y = -x - 2) ∨ (y = -2 * x + 1)) :=
by
  intros x y h
  sorry

end lines_from_equation_l461_461679


namespace legos_in_box_at_end_l461_461041

def initial_legos : ℕ := 500
def legos_used : ℕ := initial_legos / 2
def missing_legos : ℕ := 5
def remaining_legos := legos_used - missing_legos

theorem legos_in_box_at_end : remaining_legos = 245 := 
by
  sorry

end legos_in_box_at_end_l461_461041


namespace manuscript_pages_count_l461_461194

theorem manuscript_pages_count
  (P : ℕ)
  (cost_first_time : ℕ := 5 * P)
  (cost_once_revised : ℕ := 4 * 30)
  (cost_twice_revised : ℕ := 8 * 20)
  (total_cost : ℕ := 780)
  (h : cost_first_time + cost_once_revised + cost_twice_revised = total_cost) :
  P = 100 :=
sorry

end manuscript_pages_count_l461_461194


namespace sum_sequence_1024_terms_correct_l461_461526

def sequence : ℕ → ℕ
| 0 := 3
| (n + 1) := if (n % (nat.sqrt (n + 1))).toNat = 0 then 3 else 2

noncomputable def sum_first_1024_terms : ℕ :=
(0..1024).sum sequence

theorem sum_sequence_1024_terms_correct : sum_first_1024_terms = 4248 := by
    sorry

end sum_sequence_1024_terms_correct_l461_461526


namespace coat_price_reduction_l461_461175

theorem coat_price_reduction 
    (original_price : ℝ) 
    (reduction_amount : ℝ) 
    (h1 : original_price = 500) 
    (h2 : reduction_amount = 300) : 
    (reduction_amount / original_price) * 100 = 60 := 
by 
  sorry

end coat_price_reduction_l461_461175


namespace angle_between_diagonals_l461_461919

-- Definitions of the edges of the parallelepiped
variables (AB AD AA1 : ℝ)
-- Given parameters for edges
def AB := 2
def AD := 3
def AA1 := 4

-- Define the tangent of the angle
def tan_alpha : ℝ := (3 * real.sqrt 5) / 5

-- Define the angle
noncomputable def alpha : ℝ := real.arctan tan_alpha

-- Theorem to state the problem
theorem angle_between_diagonals : alpha = real.arctan ((3 * real.sqrt 5) / 5) :=
by 
  sorry

end angle_between_diagonals_l461_461919


namespace cupcakes_frosted_in_10_minutes_l461_461235

theorem cupcakes_frosted_in_10_minutes :
  let cagney_rate := 1 / 25 -- Cagney's rate in cupcakes per second
  let lacey_rate := 1 / 35 -- Lacey's rate in cupcakes per second
  let total_time := 600 -- Total time in seconds for 10 minutes
  let lacey_break := 60 -- Break duration in seconds
  let lacey_work_time := total_time - lacey_break
  let cupcakes_by_cagney := total_time / 25 
  let cupcakes_by_lacey := lacey_work_time / 35
  cupcakes_by_cagney + cupcakes_by_lacey = 39 := 
by {
  sorry
}

end cupcakes_frosted_in_10_minutes_l461_461235


namespace urn_contains_four_red_three_blue_prob_l461_461627

/-- Probability calculation for a specific urn problem. --/
theorem urn_contains_four_red_three_blue_prob :
  let initial_urn := ({red := 1, blue := 1} : Urn)
      operations := 5
      final_urn := ({red := 4, blue := 3} : UrnSet)
  exact probability_of_urn_state_after_operations initial_urn operations final_urn = 1/6 :=
sorry

end urn_contains_four_red_three_blue_prob_l461_461627


namespace d_coverable_condition_no_d_coverable_condition_l461_461101

noncomputable def smallest_d_coverable (n : ℕ) : Option ℕ :=
if n = 4 ∨ Prime n then some (n - 1)
else none

theorem d_coverable_condition (n : ℕ) (hn : n > 1) :
  (∀ S : Finset (Fin n), S.nonempty → ∃ P : Polynomial ℤ, P.natDegree ≤ n - 1 ∧ ∀ x : ℤ, x % n ∈ S ↔ (P.eval x % n) ∈ S) ↔
  smallest_d_coverable n = some (n - 1) :=
sorry

theorem no_d_coverable_condition (n : ℕ) (hn : n > 1) :
  (∀ d : ℕ, ¬ (∀ S : Finset (Fin n), S.nonempty → ∃ P : Polynomial ℤ, P.natDegree ≤ d ∧ ∀ x : ℤ, x % n ∈ S ↔ (P.eval x % n) ∈ S)) ↔
  smallest_d_coverable n = none :=
sorry

end d_coverable_condition_no_d_coverable_condition_l461_461101


namespace lap_distance_l461_461077

theorem lap_distance (total_distance : ℚ) (laps : ℕ) (distance_per_lap : ℚ) 
                     (h1 : total_distance = 13 / 4) 
                     (h2 : laps = 13) : 
  distance_per_lap = 1 / 4 :=
by
  -- Given conditions
  have td_eq : total_distance = 13 / 4 := h1
  have laps_eq : laps = 13 := h2
  
  -- Calculation
  let dpl := total_distance / laps
  have dpl_eq : dpl = 1 / 4 := sorry

  -- Conclusion
  exact dpl_eq

end lap_distance_l461_461077


namespace part1_part2_part3_l461_461182

section Part1
variables {a b : ℝ}

theorem part1 (h1 : a + b = 3) (h2 : a * b = 2) : a^2 + b^2 = 5 := 
sorry
end Part1

section Part2
variables {a b c : ℝ}

theorem part2 (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) : a^2 + b^2 + c^2 = 14 := 
sorry
end Part2

section Part3
variables {a b c : ℝ}

theorem part3 (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : a^4 + b^4 + c^4 = 18 :=
sorry
end Part3

end part1_part2_part3_l461_461182


namespace num_rel_prime_to_15_l461_461256

theorem num_rel_prime_to_15 : 
  {a : ℕ | a < 15 ∧ Int.gcd 15 a = 1}.card = 8 := by 
  sorry

end num_rel_prime_to_15_l461_461256


namespace no_real_roots_range_k_l461_461372

theorem no_real_roots_range_k (k : ℝ) : (x^2 - 2 * x - k = 0) ∧ (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) → k < -1 := 
by
  sorry

end no_real_roots_range_k_l461_461372


namespace two_marble_groups_count_l461_461146

def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

def total_combinations : ℕ :=
  let identical_yellow := 1
  let distinct_pairs := num_combinations 4 2
  identical_yellow + distinct_pairs

theorem two_marble_groups_count :
  total_combinations = 7 :=
by
  dsimp [total_combinations, num_combinations]
  rw [Nat.choose]
  norm_num
  sorry

end two_marble_groups_count_l461_461146


namespace instantaneous_speed_at_3_l461_461515

noncomputable def s (t : ℝ) : ℝ := 1 - t + 2 * t^2

theorem instantaneous_speed_at_3 : deriv s 3 = 11 :=
by
  sorry

end instantaneous_speed_at_3_l461_461515


namespace C_investment_l461_461618

theorem C_investment (A B total_profit A_share : ℝ) (x : ℝ) :
  A = 6300 → B = 4200 → total_profit = 12600 → A_share = 3780 →
  (A / (A + B + x) = A_share / total_profit) → x = 10500 :=
by
  intros hA hB h_total_profit h_A_share h_ratio
  sorry

end C_investment_l461_461618


namespace homework_problems_eq_l461_461786

variable (math_problems : ℕ) (science_problems : ℕ) (finished_problems : ℕ)
variable (total_problems : ℕ) (homework_problems : ℕ)

axiom total_problems_eq : total_problems = math_problems + science_problems
axiom finished_problems_eq : finished_problems = 24
axiom math_problems_eq : math_problems = 18
axiom science_problems_eq : science_problems = 11

theorem homework_problems_eq :
  homework_problems = total_problems - finished_problems :=
by
  rw [finished_problems_eq, math_problems_eq, science_problems_eq] at total_problems_eq
  have total_problems_def := total_problems_eq
  rw [total_problems_eq, finished_problems_eq]
  exact homework_problems = 5
-- sorry

end homework_problems_eq_l461_461786


namespace terrell_lift_equivalence_l461_461106

theorem terrell_lift_equivalence :
  ∃ n : ℕ, 80 * n = 750 ∧ n = 10 := 
begin
  use 10,
  split, 
  { norm_num, },
  { refl, }
end

end terrell_lift_equivalence_l461_461106


namespace smallest_scalene_prime_triangle_perimeter_l461_461606

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a scalene triangle with distinct side lengths
def is_scalene (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Define the triangle inequality
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define a valid scalene triangle with prime side lengths
def valid_scalene_triangle (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_scalene a b c ∧ triangle_inequality a b c

-- Proof statement
theorem smallest_scalene_prime_triangle_perimeter : ∃ (a b c : ℕ), 
  valid_scalene_triangle a b c ∧ a + b + c = 15 := 
sorry

end smallest_scalene_prime_triangle_perimeter_l461_461606


namespace smallest_number_l461_461369

theorem smallest_number (s : List ℝ) (h : s = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) :
  ∃ s' : List ℝ, 
    (∀ a b ∈ s, a = b → (a + b) / 2 ≤ (1 / 512)) ∧ 
    (1 / 512 ∈ s') := 
sorry

end smallest_number_l461_461369


namespace find_a_l461_461390

-- Definitions and theorem statement
def A (a : ℝ) : Set ℝ := {2, a^2 - a + 1}
def B (a : ℝ) : Set ℝ := {3, a + 3}
def C (a : ℝ) : Set ℝ := {3}

theorem find_a (a : ℝ) : A a ∩ B a = C a → a = 2 :=
by
  sorry

end find_a_l461_461390


namespace unique_solution_0_pi_J_in_terms_of_sin_alpha_J_gt_sqrt_2_l461_461442

noncomputable def f (x : ℝ) : ℝ := 1 - cos x - x * sin x

theorem unique_solution_0_pi : ∃! α ∈ Ioo 0 π, f α = 0 := sorry

noncomputable def J (α : ℝ) : ℝ := ∫ x in 0..π, abs (f x)

theorem J_in_terms_of_sin_alpha (α : ℝ) (hα : ∃ α ∈ Ioo 0 π, f α = 0) : J α = 2 * α * sin α := sorry

theorem J_gt_sqrt_2 (α : ℝ) (hα : ∃ α ∈ Ioo 0 π, f α = 0) : J α > sqrt 2 := sorry

end unique_solution_0_pi_J_in_terms_of_sin_alpha_J_gt_sqrt_2_l461_461442


namespace double_inputs_revenue_l461_461017

theorem double_inputs_revenue (A K L : ℝ) (α1 α2 : ℝ) (hα1 : α1 = 0.6) (hα2 : α2 = 0.5) (hα1_bound : 0 < α1 ∧ α1 < 1) (hα2_bound : 0 < α2 ∧ α2 < 1) :
  A * (2 * K) ^ α1 * (2 * L) ^ α2 > 2 * (A * K ^ α1 * L ^ α2) :=
by
  sorry

end double_inputs_revenue_l461_461017


namespace lines_from_equation_l461_461680

-- Definitions for the conditions
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Equivalent Lean statement to the proof problem
theorem lines_from_equation :
  (∀ x y : ℝ, satisfies_equation x y → (y = -x - 2) ∨ (y = -2 * x + 1)) :=
by
  intros x y h
  sorry

end lines_from_equation_l461_461680


namespace chocolate_bars_left_l461_461195

noncomputable def chocolateBarsCount : ℕ :=
  let initial_bars := 800
  let thomas_friends_bars := (3 * initial_bars) / 8
  let adjusted_thomas_friends_bars := thomas_friends_bars + 1  -- Adjust for the extra bar rounding issue
  let piper_bars_taken := initial_bars / 4
  let piper_bars_returned := 8
  let adjusted_piper_bars := piper_bars_taken - piper_bars_returned
  let paul_club_bars := 9
  let polly_club_bars := 7
  let catherine_bars_returned := 15
  
  initial_bars
  - adjusted_thomas_friends_bars
  - adjusted_piper_bars
  - paul_club_bars
  - polly_club_bars
  + catherine_bars_returned

theorem chocolate_bars_left : chocolateBarsCount = 308 := by
  sorry

end chocolate_bars_left_l461_461195


namespace part1_part2_l461_461770

-- Definition of function f.
def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- Condition provided in the problem.
axiom A1 : ∀ x : ℝ, f(x) = 2 * f(-x)

-- Lean proof for first question.
theorem part1 (x : ℝ) (h : f(x) = 2 * f(-x)) :
    (cos(x)^2 - sin(x) * cos(x)) / (1 + sin(x)^2) = 6/11 := 
by
  -- mathematical equivalent to solving given condition
  sorry

-- Definition and property of F
def F (x : ℝ) : ℝ := f(x) * f(-x) + f(x) ^ 2

-- Lean proof for second question.
theorem part2 (h1 : ∀ x : ℝ, 0 < x -> x < Real.pi/2) :
  (∀ x : ℝ, 0 < x ∧ x < Real.pi/2 → F(x) ∈ (0, Real.sqrt 2 + 1]) ∧
  (∀ x : ℝ, 0 < x ∧ x < Real.pi/8 → F(x + ε) > F(x) ∧ F(x) is increasing) :=
by
  -- analysis of range and interval of monotonic increase
  sorry

end part1_part2_l461_461770


namespace find_a_l461_461412

theorem find_a (a : ℝ) (h : a * (1 : ℝ)^2 - 6 * 1 + 3 = 0) : a = 3 :=
by
  sorry

end find_a_l461_461412


namespace find_p_l461_461024

-- Define the coordinates as given in the problem
def Q : ℝ × ℝ := (0, 15)
def A : ℝ × ℝ := (3, 15)
def B : ℝ × ℝ := (15, 0)
def O : ℝ × ℝ := (0, 0)
def C (p : ℝ) : ℝ × ℝ := (0, p)

-- Defining the function to calculate area of triangle given three points
def area_of_triangle (P1 P2 P3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (P1.fst * (P2.snd - P3.snd) + P2.fst * (P3.snd - P1.snd) + P3.fst * (P1.snd - P2.snd))

-- The statement we need to prove
theorem find_p :
  ∃ p : ℝ, area_of_triangle A B (C p) = 42 ∧ p = 11.75 :=
by
  sorry

end find_p_l461_461024


namespace range_of_a_if_sequence_increasing_l461_461374

noncomputable def sequence (a : ℝ) (n : ℕ) : ℝ :=
  if n ∈ {k | k > 0 ∧ k ≤ 7} then (3 - a) * ↑n - 3 else a^(n - 6)

theorem range_of_a_if_sequence_increasing (a : ℝ) (h : ∀ n m : ℕ, n < m → sequence a n < sequence a m) : 
  2 < a ∧ a < 3 :=
begin
  sorry
end

end range_of_a_if_sequence_increasing_l461_461374


namespace cost_per_piece_l461_461080

variable (totalCost : ℝ) (numberOfPizzas : ℝ) (piecesPerPizza : ℝ)

theorem cost_per_piece (h1 : totalCost = 80) (h2 : numberOfPizzas = 4) (h3 : piecesPerPizza = 5) :
  totalCost / numberOfPizzas / piecesPerPizza = 4 := by
sorry

end cost_per_piece_l461_461080


namespace smallest_even_number_tens_place_l461_461953

theorem smallest_even_number_tens_place :
  ∃ n : ℕ, even n ∧ digits n.card = 5 ∧ 
    (∀ m : ℕ, even m ∧ digits m.card = 5 ∧ 
      set_eq (digits n).μ {1, 2, 3, 5, 6, 8} → n ≤ m) ∧ 
    tens_digit n = 6 := sorry

end smallest_even_number_tens_place_l461_461953


namespace krista_total_amount_exceeds_2_dollars_l461_461842

noncomputable def total_amount_exceeds_2_dollars_day : String :=
  let deposits : List ℕ := List.range 14 |>.map (λ i, 2^i)
  let totalAmount := deposits.takeWhile (λ s, s.sum < 200).length
  let startDay : ℕ := 7 -- Sunday
  if totalAmount + startDay >= 14 then
    "Sunday"
  else
    let days := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    days[(totalAmount + startDay) % 7]

theorem krista_total_amount_exceeds_2_dollars : total_amount_exceeds_2_dollars_day = "Sunday" :=
  sorry

end krista_total_amount_exceeds_2_dollars_l461_461842


namespace proof_problem_l461_461625

variable {a_n S_n : ℕ → ℝ}
variable (b_n : ℕ → ℝ)

def geometric_sequence_increasing (a_n : ℕ → ℝ) : Prop := ∀ n, a_n < a_(n+1)

def sn_sum (a_n S_n : ℕ → ℝ) : Prop := ∀ n, S_n = ∑ i in range n, a_n i

def an_condition_1 (a_n : ℕ → ℝ) : Prop := a_n 5 ^ 2 = a_n 10

def an_condition_2 (a_n S_n : ℕ → ℝ) : Prop := ∀ n, 2 * a_n n + 5 * S_n n = 
5 * S_n (n + 1) - 2 * a_n (n + 2)

def bn_definition (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) : Prop := ∀ n, b_n n = 
a_n n * abs (cos (n * π / 2))

def tn_sum (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) : Prop := ∀ n, T_n = ∑ i in range n, b_n i

theorem proof_problem 
  (h1 : geometric_sequence_increasing a_n) 
  (h2 : sn_sum a_n S_n)
  (h3 : an_condition_1 a_n)
  (h4 : an_condition_2 a_n S_n)
  (h5 : bn_definition a_n b_n)
  (h6 : tn_sum b_n T_n) :
  (∃ q, (q = 2 → ∀ n, a_n n = 2 ^ n) 
    ∧ (q = 0.5 → ∀ n, a_n n = (1/2) ^ n)) 
    → ∃ n, (T_n n = 340) → (n = 8 ∨ n = 9) := sorry

end proof_problem_l461_461625


namespace color_plane_with_two_colors_l461_461599

/-- Given a finite set of circles that divides the plane into regions, we can color the plane such that no two adjacent regions have the same color. -/
theorem color_plane_with_two_colors (circles : Finset (Set ℝ)) :
  (∀ (r1 r2 : Set ℝ), (r1 ∩ r2).Nonempty → ∃ (coloring : Set ℝ → Bool), (coloring r1 ≠ coloring r2)) :=
  sorry

end color_plane_with_two_colors_l461_461599


namespace certain_number_is_negative_425_l461_461633

theorem certain_number_is_negative_425 (x : ℝ) :
  (3 - (1/5) * x = 88) ∧ (4 - (1/7) * 210 = -26) → x = -425 :=
by
  sorry

end certain_number_is_negative_425_l461_461633


namespace circle_tangent_to_line_l461_461061

-- Definitions of the relevant points and circles
variables {A B C D M N : Type}
variables [metric_space] [is_triangle ABC]
variables (AD : segment A D)
variables (l : line)
variables (circumcircle_ABD circumcircle_ACD : circle ABC)
variables (midpoint_BD midpoint_DC midpoint_MN : point)

-- Circumcircle tangency conditions
variables (tangent_ABD : tangent_line circumcircle_ABD l M)
variables (tangent_ACD : tangent_line circumcircle_ACD l N)

-- Given the statements from the condition
def is_angle_bisector (AD : segment A D) : Prop :=
  ∃ α : Angle, ∠BAD = α ∧ ∠CAD = α

def midpoints_and_circle (BD DC MN : segment) : circle_notation :=
  circle_passing_through (midpoint BD) (midpoint DC) (midpoint MN)

-- The theorem we need to prove
theorem circle_tangent_to_line 
  (h1 : is_angle_bisector AD ⁍)
  (h2 : tangent_circumcircle circumcircle_ABD l M ⁍)
  (h3 : tangent_circumcircle circumcircle_ACD l N ⁍):
  (tangent_line (midpoints_and_circle BD DC MN) l) := by
  sorry

end circle_tangent_to_line_l461_461061


namespace total_groups_of_two_marbles_l461_461149

-- Define the given conditions as constants
constant red_marble : Type
constant green_marble : Type
constant blue_marble : Type
constant purple_marble : Type
constant yellow_marble : Type

constant identical_yellow_marbles : ℕ

-- Prove the total number of different groups of two marbles
theorem total_groups_of_two_marbles :
  (identical_yellow_marbles = 4) →
  let marbles := [red_marble, green_marble, blue_marble, purple_marble, yellow_marble, yellow_marble, yellow_marble, yellow_marble] in
  (count_distinct_groups_of_two marbles = 11) :=
begin
  intros h,
  let marbles := [red_marble, green_marble, blue_marble, purple_marble, yellow_marble, yellow_marble, yellow_marble, yellow_marble],
  sorry -- proof omitted
end

end total_groups_of_two_marbles_l461_461149


namespace cheyenne_clay_pots_l461_461641

theorem cheyenne_clay_pots (P : ℕ) (cracked_ratio sold_ratio : ℝ) (total_revenue price_per_pot : ℝ) 
    (P_sold : ℕ) :
  cracked_ratio = (2 / 5) →
  sold_ratio = (3 / 5) →
  total_revenue = 1920 →
  price_per_pot = 40 →
  P_sold = 48 →
  (sold_ratio * P = P_sold) →
  P = 80 :=
by
  sorry

end cheyenne_clay_pots_l461_461641


namespace distance_between_A1C1_and_BD1_l461_461297

-- Defining the cube with edge length 1
variable {A B C D A1 B1 C1 D1 : Point}
variables (cube : Cube A B C D A1 B1 C1 D1)
variable (edge_length : ℝ)
hypothesis (h_edge_length : edge_length = 1)

-- Defining the lines A1C1 and BD1
variables (line_A1C1 : Line A1 C1)
variables (line_BD1 : Line B D1)

-- Definition of the distance calculation
def distance_between_lines (line1 line2 : Line Point) : ℝ := 
  sorry -- Placeholder for the distance function

-- Theorem to prove the distance between specified lines in the cube
theorem distance_between_A1C1_and_BD1 :
  distance_between_lines line_A1C1 line_BD1 = (Real.sqrt 6) / 6 :=
by
  -- Conditions of the problem: Cube with edge length 1
  let cube_edge_length := 1
  have h1 : edge_length = cube_edge_length := h_edge_length
  -- The distance to be proved
  show distance_between_lines line_A1C1 line_BD1 = (Real.sqrt 6) / 6
  sorry

end distance_between_A1C1_and_BD1_l461_461297


namespace inequality_proof_l461_461857

theorem inequality_proof 
  (a b c d : ℝ)
  (h₁ : a ≥ 0)
  (h₂ : b ≥ 0)
  (h₃ : c ≥ 0)
  (h₄ : d ≥ 0)
  (h₅ : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3) :=
begin
  sorry
end

end inequality_proof_l461_461857


namespace conditional_probability_l461_461500

open ProbabilityTheory

variables (A B : Event)
variables (P : Event → ℚ)

def P_A : ℚ := 4 / 15
def P_B : ℚ := 2 / 15
def P_AB : ℚ := 1 / 10

theorem conditional_probability :
  P(A|B) = 3 / 4 :=
by
  sorry

end conditional_probability_l461_461500


namespace least_positive_integer_n_least_positive_integer_l461_461291

theorem least_positive_integer_n :
  (finset.sum (finset.range 45) (λ k, (1 / (sin (real.to_radians (45 + k)) * sin (real.to_radians (46 + k))))))
  = 1 / sin (real.to_radians 1) :=
sorry

theorem least_positive_integer (n : ℕ) (hn : (finset.sum (finset.range 45) (λ k, (1 / (sin (real.to_radians (45 + k)) * sin (real.to_radians (46 + k))))))
  = 1 / sin (real.to_radians n)) : n = 1 :=
sorry

end least_positive_integer_n_least_positive_integer_l461_461291


namespace coefficient_x_6_in_expansion_l461_461969

-- Define the variable expressions and constraints of the problem
def expansion_expr : ℕ → ℤ := λ k, Nat.choose 4 k * 1^(4 - k) * (-3)^(k)
def term_coefficient_of_x_pow_6 (k : ℕ) : ℕ := if (3 * k = 6) then Nat.choose 4 k * 9 else 0

-- Prove that the coefficient of x^6 in the expansion of (1-3x^3)^4 is 54
theorem coefficient_x_6_in_expansion : term_coefficient_of_x_pow_6 2 = 54 := by
  -- Simplify the expression for the term coefficient of x^6 when k = 2
  simp only [term_coefficient_of_x_pow_6]
  split_ifs
  simp [Nat.choose, Nat.factorial]
  sorry -- one could continue simplifying this manually or provide arithmetic through Lean library

end coefficient_x_6_in_expansion_l461_461969


namespace geometric_series_common_ratio_l461_461647

-- Definitions
def first_term : ℚ := 7/8
def second_term : ℚ := -14/27
def third_term : ℚ := 28/81

-- Theorem statement
theorem geometric_series_common_ratio : 
  (second_term / first_term = third_term / second_term) →
  (second_term / first_term = -2/3) := 
begin
  -- Assume second_term / first_term = third_term / second_term
  intro h,
  -- Prove that second_term / first_term = -2/3
  sorry
end

end geometric_series_common_ratio_l461_461647


namespace march_volume_expression_l461_461192

variable (x : ℝ) (y : ℝ)

def initial_volume : ℝ := 500
def growth_rate_volumes (x : ℝ) (initial_volume : ℝ) : ℝ := initial_volume * (1 + x)
def calculate_march_volume (x : ℝ) (initial_volume : ℝ) : ℝ := initial_volume * (1 + x)^2

theorem march_volume_expression :
  y = calculate_march_volume x initial_volume :=
sorry

end march_volume_expression_l461_461192


namespace angle_bisector_theorem_l461_461088

variables {Point : Type*} [AffineSpace ℝ Point] 
variables (A B O M C : Point)

-- Conditions
-- Point M lies outside the angle AOB
-- OC is the bisector of the angle AOB

theorem angle_bisector_theorem (hM_outside : ¬(∠ A O M ≤ ∠ A O B ∧ ∠ B O M ≤ ∠ A O B))
  (hOC_bisector : angle_bisector A O B C) : 
  ∠ A O M + ∠ B O M = 2 * ∠ M O C := 
sorry

end angle_bisector_theorem_l461_461088


namespace product_fractions_is_one_iff_n_is_square_l461_461549

theorem product_fractions_is_one_iff_n_is_square (n : ℕ) :
  (∃ flips : Fin (n - 1) → Bool, ∏ i in Finset.range (n - 1), if flips i then (i+1)/n else n/(i+1) = 1) ↔ ∃ k : ℕ, n = k^2 :=
sorry

end product_fractions_is_one_iff_n_is_square_l461_461549


namespace range_of_a_l461_461768

noncomputable def f (x a : ℝ) : ℝ :=
if x > 0 then x^2 - 2 else -3 * |x + a| + a

theorem range_of_a : 
  (∃ x y : ℝ, f x a = y ∧ f (-x) a = y ∧ x ≠ 0 ∧ y ≠ f 0 a) → 
  a ∈ set.Ioo 1 (17 / 16) :=
sorry

end range_of_a_l461_461768


namespace nice_subsets_count_l461_461441

noncomputable def is_nice_subset (S : Finset ℝ) (A : Finset ℝ) (k l : ℕ) : Prop :=
  |A| = k ∧ (∃ (a b : ℝ), a = (∑ x in A, x) / k ∧ b = (∑ x in S \ A, x) / l ∧
  abs (a - b) ≤ (k + l) / (2 * k * l))

def number_of_nice_subsets (S : Finset ℝ) (k l : ℕ) : ℝ :=
  (S.powerset.filter (λ A, is_nice_subset S A k l)).card

theorem nice_subsets_count (S : Finset ℝ) (hS : ∀ x ∈ S, 0 ≤ x ∧ x ≤ 1) (k l : ℕ) (hk : 0 < k) (hl : 0 < l)
  (hkl : S.card = k + l) :
  number_of_nice_subsets S k l ≥ (2 / (k + l)) * Nat.choose (k + l) k :=
sorry

end nice_subsets_count_l461_461441


namespace range_of_m_l461_461764

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (1/3 < x ∧ x < 1/2) → (|x - m| < 1)) ↔ (-1/2 ≤ m ∧ m ≤ 4/3) :=
by
  split
  sorry -- no proof required as per instructions

end range_of_m_l461_461764


namespace floor_e_equals_2_l461_461695

theorem floor_e_equals_2 : Int.floor Real.exp = 2 := 
sorry

end floor_e_equals_2_l461_461695


namespace largest_prime_factor_of_4652_is_89_l461_461553

theorem largest_prime_factor_of_4652_is_89 : (∃ p : ℕ, prime p ∧ p ∣ 4652 ∧ (∀ q : ℕ, prime q ∧ q ∣ 4652 → q ≤ p)) ∧ p = 89 :=
by
  sorry

end largest_prime_factor_of_4652_is_89_l461_461553


namespace limit_seq_l461_461067

noncomputable def seq (a b : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a else if n = 1 then b else 0.5 * (seq a b (n - 1) + seq a b (n - 2))

theorem limit_seq (a b : ℝ) :
  filter.tendsto (λ n, seq a b n) filter.at_top (nhds (a + 2 * b) / 3) :=
sorry

end limit_seq_l461_461067


namespace perimeter_of_triangle_l461_461418

theorem perimeter_of_triangle (P Q R : Type) [MetricSpace P] 
  (a : Q ≠ R) (b : P ≠ R) (h1 : ∠PQR = ∠PRQ) (h2 : dist Q R = 5) (h3 : dist P R = 7) :
  let distPQ : ℝ := dist P Q
  is_isosceles_triangle (dist P Q) (dist P R) (dist Q R) :=
  dist P Q = dist P R :=
  ∃ (c : ℝ), distPQ + dist Q R + dist P R = 19 :=
begin
  sorry
end

end perimeter_of_triangle_l461_461418


namespace coordinate_equation_solution_l461_461670

theorem coordinate_equation_solution (x y : ℝ) :
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 →
  (y = -x - 2) ∨ (y = -2 * x + 1) :=
by
  sorry

end coordinate_equation_solution_l461_461670


namespace convex_polyhedron_two_faces_same_sides_l461_461095

variable {n : ℕ} (faces : Fin n → ℕ)
axiom face_sides : ∀ i : Fin n, 3 ≤ faces i ∧ faces i ≤ n - 1

theorem convex_polyhedron_two_faces_same_sides (n : ℕ) (faces : Fin n → ℕ) [H : ∀ i, 3 ≤ faces i ∧ faces i ≤ n - 1] :
    ∃ (i j : Fin n), i ≠ j ∧ faces i = faces j :=
by
  sorry

end convex_polyhedron_two_faces_same_sides_l461_461095


namespace quadratic_distinct_real_roots_l461_461000

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ^ 2 - 2 * x₁ + m = 0 ∧ x₂ ^ 2 - 2 * x₂ + m = 0) ↔ m < 1 :=
by sorry

end quadratic_distinct_real_roots_l461_461000


namespace saree_blue_stripes_l461_461955

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    brown_stripes = 4 →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_gold h_blue h_brown
  sorry

end saree_blue_stripes_l461_461955


namespace triangle_angle_bisectors_l461_461050

theorem triangle_angle_bisectors (A B C E E' : Point) 
  (H1 : Collinear A C E) 
  (H2 : Collinear A C E') 
  (H3 : AE = AB) 
  (H4 : AE' = AB) 
  (H5 : E' ≠ C) : 
  (Parallel (Line B E) (internalAngleBisector A)) ∧ 
  (Parallel (Line B E') (externalAngleBisector A)) ∧ 
  (RightAngle (Line E B) (Line E' B)) := 
sorry

end triangle_angle_bisectors_l461_461050


namespace range_PA_PB_PC_l461_461323

open Real EuclideanGeometry

variable {O A B C P : Point}

-- Define the circle with radius 1
def isCircle (O : Point) (r : ℝ) := ∀ P, dist O P = r

-- Assume the points A, B, and C lie on the circle with center O and radius 1
axiom points_on_circle : isCircle O 1

-- Assume AB is the diameter of the circle
axiom AB_diameter : dist A B = 2 * dist O A

-- Assume P is a point inside or on the circle
axiom P_inside_circle : dist O P ≤ 1

theorem range_PA_PB_PC :
  -4 / 3 ≤ (∥PA∥ * ∥PB∥ + ∥PB∥ * ∥PC∥ + ∥PC∥ * ∥PA∥) ∧
  (∥PA∥ * ∥PB∥ + ∥PB∥ * ∥PC∥ + ∥PC∥ * ∥PA∥) ≤ 4 :=
sorry

end range_PA_PB_PC_l461_461323


namespace problem_l461_461801

universes u
variables {U : Type u} [inhabited U]

def is_true_statement1 (A B : set U) : Prop :=
  A ∩ B = ∅ → (Aᶜ ∪ Bᶜ = set.univ)

def is_true_statement2 (A B : set U) : Prop :=
  A ∪ B = set.univ → (Aᶜ ∩ Bᶜ = ∅)

def is_true_statement3 (A B : set U) : Prop :=
  A ∪ B = ∅ → (A = ∅ ∧ B = ∅)

def count_true_statements (A B : set U) : Nat :=
  [is_true_statement1 A B, is_true_statement2 A B, is_true_statement3 A B].count (λ p, p)

theorem problem (A B : set U) :
  count_true_statements A B = 3 :=
sorry

end problem_l461_461801


namespace polynomial_multiplication_eq_difference_of_cubes_l461_461865

theorem polynomial_multiplication_eq_difference_of_cubes 
  (x y z : ℝ) :
  (3 * x^2 * z - 7 * y^3) * (9 * x^4 * z^2 + 21 * x^2 * y * z^3 + 49 * y^6) = 
  (27 * x^6 * z^3 - 343 * y^9) :=
by
  let a := 3 * x^2 * z
  let b := 7 * y^3
  have h : (a - b) * (a^2 + a * b + b^2) = a^3 - b^3,
  { sorry }
  rw [h],
  rw [←pow_succ, ←pow_succ, ←pow_succ, ←pow_succ],
  norm_num,
  field_simp,
  sorry

end polynomial_multiplication_eq_difference_of_cubes_l461_461865


namespace general_formula_a_n_sum_T_n_l461_461312

noncomputable def sequence_a (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 2
  | k+2 => (Finset.range (k + 1)).sum sequence_a + 2

def sequence_b (n : ℕ) : ℝ := Real.logb 2 (sequence_a n)

def term_inverse_product (n : ℕ) : ℝ := 1 / (sequence_b n * sequence_b (n + 1))

def T (n : ℕ) : ℝ := (Finset.range n).sum term_inverse_product

theorem general_formula_a_n (n : ℕ) (hn : n ≥ 1) :
  sequence_a n = 2^n :=
sorry

theorem sum_T_n (n : ℕ) :
  T n = n / (n + 1) :=
sorry

end general_formula_a_n_sum_T_n_l461_461312


namespace worker_A_time_l461_461169

def time_worker_B : ℝ := 10
def time_together : ℝ := 4.11764705882353
def work_rate_B : ℝ := 1 / time_worker_B
def combined_work_rate : ℝ := 1 / time_together

theorem worker_A_time :
  ∃ A : ℝ, (1 / A + work_rate_B = combined_work_rate) ∧ (A ≈ 2.92) :=
sorry

end worker_A_time_l461_461169


namespace probability_product_multiple_of_three_l461_461903
noncomputable theory
open_locale big_operators

def is_multiple_of_three (n : ℕ) : Prop :=
  n % 3 = 0

def all_pairs (n: ℕ) : list (ℕ × ℕ) :=
  (list.fin_range n).product (list.fin_range n)

def valid_pairs : list (ℕ × ℕ) :=
  (all_pairs 6).filter (λ p, is_multiple_of_three (p.1 * p.2))

theorem probability_product_multiple_of_three:
  (valid_pairs.length : ℚ) / (all_pairs 6).length = 5 / 9 :=
begin
  sorry
end

end probability_product_multiple_of_three_l461_461903


namespace prod_distances_on_line_circle_l461_461612

noncomputable def tan (α : ℝ) := 3 / 4

theorem prod_distances_on_line_circle (α : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (∃ t : ℝ, A = (1 + (4 / 5) * t, 1 + (3 / 5) * t)) ∧ 
    (∃ t : ℝ, B = (1 + (4 / 5) * t, 1 + (3 / 5) * t)) ∧ 
    (A.1 ^ 2 + A.2 ^ 2 = 4) ∧ 
    (B.1 ^ 2 + B.2 ^ 2 = 4) ∧ 
    (((1 - A.1)^2 + (1 - A.2)^2) * ((1 - B.1)^2 + (1 - B.2)^2) = 2)) :=
begin
  sorry
end

end prod_distances_on_line_circle_l461_461612


namespace find_divisor_l461_461717

-- Definitions from the condition
def original_number : ℕ := 724946
def least_number_subtracted : ℕ := 6
def remaining_number : ℕ := original_number - least_number_subtracted

theorem find_divisor (h1 : remaining_number % least_number_subtracted = 0) :
  Nat.gcd original_number least_number_subtracted = 2 :=
sorry

end find_divisor_l461_461717


namespace milk_percentage_after_adding_water_l461_461538

theorem milk_percentage_after_adding_water
  (initial_total_volume : ℚ) (initial_milk_percentage : ℚ)
  (additional_water_volume : ℚ) :
  initial_total_volume = 60 → initial_milk_percentage = 0.84 → additional_water_volume = 18.75 →
  (50.4 / (initial_total_volume + additional_water_volume) * 100 = 64) :=
by
  intros h1 h2 h3
  rw [h1, h3]
  simp
  sorry

end milk_percentage_after_adding_water_l461_461538


namespace find_a_values_l461_461445

noncomputable def M : Set ℝ := {x | x^2 + x - 6 = 0}
noncomputable def N (a : ℝ) : Set ℝ := {x | ax + 2 = 0}
def M_inter_N_eq_N (a : ℝ) : Prop := M ∩ (N a) = (N a)

theorem find_a_values (a : ℝ) (ha : M_inter_N_eq_N a) :
  a = 0 ∨ a = -1 ∨ a = (2/3) :=
sorry

end find_a_values_l461_461445


namespace range_of_c_monotonicity_of_g_l461_461349

noncomputable def f (x: ℝ) : ℝ := 2 * Real.log x + 1

theorem range_of_c (c: ℝ) : (∀ x > 0, f x ≤ 2 * x + c) → c ≥ -1 := by
  sorry

noncomputable def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

theorem monotonicity_of_g (a: ℝ) (ha: a > 0) : 
  (∀ x > 0, x ≠ a → ((x < a → g x a < g a a) ∧ (x > a → g x a < g a a))) := by
  sorry

end range_of_c_monotonicity_of_g_l461_461349


namespace event_B_is_certain_l461_461974

-- Define the event that the sum of two sides of a triangle is greater than the third side
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the term 'certain event'
def certain_event (E : Prop) : Prop := E

/-- Prove that the event "the sum of two sides of a triangle is greater than the third side" is a certain event -/
theorem event_B_is_certain (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  certain_event (triangle_inequality a b c) :=
sorry

end event_B_is_certain_l461_461974


namespace volume_of_sphere_given_surface_area_l461_461529

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Math.pi * r^3

theorem volume_of_sphere_given_surface_area :
  (∃ (r : ℝ), 4 * Math.pi * r^2 = 256 * Math.pi) →
  ∃ (V : ℝ), V = 2048 / 3 * Math.pi :=
by
  intro h
  cases h with r hr
  use volume_of_sphere r
  have : r = 8 := by
    have hr_squared : r ^ 2 = 64 := by 
      rw [mul_eq_mul_right_iff, mul_comm] at hr
      exact hr.elim_left
    exact by linarith using [real.sqrt_nonneg 64, real.sq_sqrt, hr_squared]
  rw [this]
  norm_num
  sorry

end volume_of_sphere_given_surface_area_l461_461529


namespace veronica_flashlight_distance_l461_461548

theorem veronica_flashlight_distance (V F Vel : ℕ) 
  (h1 : F = 3 * V)
  (h2 : Vel = 5 * F - 2000)
  (h3 : Vel = V + 12000) : 
  V = 1000 := 
by {
  sorry 
}

end veronica_flashlight_distance_l461_461548


namespace freddy_age_l461_461655

theorem freddy_age
  (mat_age : ℕ)  -- Matthew's age
  (reb_age : ℕ)  -- Rebecca's age
  (fre_age : ℕ)  -- Freddy's age
  (h1 : mat_age = reb_age + 2)
  (h2 : fre_age = mat_age + 4)
  (h3 : mat_age + reb_age + fre_age = 35) :
  fre_age = 15 :=
by sorry

end freddy_age_l461_461655


namespace root_in_interval_l461_461073

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem root_in_interval : 
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  intros h1 h2
  sorry

end root_in_interval_l461_461073


namespace solve_for_y_l461_461493

theorem solve_for_y : ∃ y : ℕ, 8^4 = 2^y ∧ y = 12 := by
  sorry

end solve_for_y_l461_461493


namespace ratio_BZ_ZC_l461_461816

/-- 
In triangle ABC, let AC > AB. 
Point P is the intersection of the perpendicular bisector of BC and the internal angle bisector of ∠A. 
PX ⊥ AB, intersecting the extension of AB at point X. 
PY ⊥ AC, intersecting AC at point Y. 
Z is the intersection of XY and BC. 
Prove that BZ / ZC = 1.
-/
theorem ratio_BZ_ZC {A B C P X Y Z : ℝ} 
  (hABC : A ≠ B ∧ A ≠ C ∧ B ≠ C) 
  (hAC_AB : AC > AB)
  (hP_bisector : is_angle_bisector ∠A P) 
  (hP_perp_bisector : is_perpendicular_bisector P BC)
  (hPX_perp_AB : PX ⊥ AB)
  (hPY_perp_AC : PY ⊥ AC)
  (hZ_intersection : Z ∈ line XY ∧ Z ∈ line BC) :
  BZ / ZC = 1 :=
by
  sorry

end ratio_BZ_ZC_l461_461816


namespace solve_inequality_l461_461494

theorem solve_inequality (a : ℝ) :
  let S := {x : ℝ | ax^2 - ax + x > 0} in
  (a = 0 → S = {x | x > 0}) ∧
  (a = 1 → S = {x | x ≠ 0}) ∧
  (a < 0 → S = {x | 0 < x ∧ x < 1 - 1/a}) ∧
  (a > 1 → S = {x | x < 0 ∨ x > 1 - 1/a}) ∧
  (0 < a ∧ a < 1 → S = {x | x < 1 - 1/a ∨ x > 0}) :=
by
  sorry

end solve_inequality_l461_461494


namespace tom_spent_on_videogames_l461_461150

theorem tom_spent_on_videogames (batman_game superman_game : ℝ) 
  (h1 : batman_game = 13.60) 
  (h2 : superman_game = 5.06) : 
  batman_game + superman_game = 18.66 :=
by 
  sorry

end tom_spent_on_videogames_l461_461150


namespace current_rate_l461_461936

theorem current_rate (c : ℝ) :
  30 + c = 37 ⇒ ∃ c : ℝ, (30 + c) * (3/5) = 22.2 :=
begin
  sorry
end

end current_rate_l461_461936


namespace max_S_value_min_S_value_l461_461309

noncomputable def S (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  let M := x.argmax
  n * (x M) ^ 2 + 2 * ∑ i j in Finset.univ.filter (λ x, i < j), x i * x j

theorem max_S_value (n : ℕ) (x : Fin n → ℝ)
  (hn : 2 ≤ n) (hx_nonneg : ∀ i, 0 ≤ x i) 
  (hx_sum : ∑ i, (x i) ^ 2 = 1):
  S n x ≤ n + Real.sqrt n - 1 := by
  sorry

theorem min_S_value (n : ℕ) (x : Fin n → ℝ)
  (hn : 2 ≤ n) (hx_nonneg : ∀ i, 0 ≤ x i) 
  (hx_sum : ∑ i, (x i) ^ 2 = 1):
  let t := if (Real.fract (Real.sqrt n) < 0.5) then Real.floor (Real.sqrt n) else Real.ceil (Real.sqrt n)
  in S n x ≥ n / t + t - 1 := by
  sorry

end max_S_value_min_S_value_l461_461309


namespace jane_doe_total_investment_mutual_funds_l461_461839

theorem jane_doe_total_investment_mutual_funds :
  ∀ (c m : ℝ) (total_investment : ℝ),
  total_investment = 250000 → m = 3 * c → c + m = total_investment → m = 187500 :=
by
  intros c m total_investment h_total h_relation h_sum
  sorry

end jane_doe_total_investment_mutual_funds_l461_461839


namespace coprime_divisibility_l461_461590

theorem coprime_divisibility (p q r P Q R : ℕ)
  (hpq : Nat.gcd p q = 1) (hpr : Nat.gcd p r = 1) (hqr : Nat.gcd q r = 1)
  (h : ∃ k : ℤ, (P:ℤ) * (q*r) + (Q:ℤ) * (p*r) + (R:ℤ) * (p*q) = k * (p*q * r)) :
  ∃ a b c : ℤ, (P:ℤ) = a * (p:ℤ) ∧ (Q:ℤ) = b * (q:ℤ) ∧ (R:ℤ) = c * (r:ℤ) :=
by
  sorry

end coprime_divisibility_l461_461590


namespace arithmetic_sequence_fifth_term_l461_461022

theorem arithmetic_sequence_fifth_term (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = 6) (h3 : a 3 = 2) (h_arith_seq : ∀ n, a (n + 1) = a n + d) : a 5 = -2 :=
sorry

end arithmetic_sequence_fifth_term_l461_461022


namespace amusement_park_trip_cost_l461_461473

def cost_per_trip (pass_cost : ℕ) (num_passes : ℕ) (oldest_trips : ℕ) (youngest_trips : ℕ) : ℕ :=
  let total_cost := num_passes * pass_cost
  let total_trips := oldest_trips + youngest_trips
  total_cost / total_trips

theorem amusement_park_trip_cost :
  ∀ (pass_cost num_passes oldest_trips youngest_trips : ℕ),
  pass_cost = 100 → num_passes = 2 → oldest_trips = 35 → youngest_trips = 15 →
  cost_per_trip pass_cost num_passes oldest_trips youngest_trips = 4 :=
by
  intros
  rw [H, H_1, H_2, H_3]
  sorry

end amusement_park_trip_cost_l461_461473


namespace range_of_a_and_t_minimum_of_y_l461_461993

noncomputable def minimum_value_y (a b : ℝ) (h : a + b = 1) : ℝ :=
(a + 1/a) * (b + 1/b)

theorem range_of_a_and_t (a b : ℝ) (h : a + b = 1) :
  0 < a ∧ a < 1 ∧ 0 < a * b ∧ a * b <= 1/4 :=
sorry

theorem minimum_of_y (a b : ℝ) (h : a + b = 1) :
  minimum_value_y a b h = 25/4 :=
sorry

end range_of_a_and_t_minimum_of_y_l461_461993


namespace range_of_c_monotonicity_g_l461_461359

-- Define the given function f(x)
def f (x : ℝ) := 2 * real.log x + 1

-- Part 1: Define the hypothesis for the range of c
theorem range_of_c :
  ∀ x : ℝ, f(x) ≤ 2 * x + c ↔ c ∈ set.Ici (-1) :=
sorry

-- Part 2: Define the function g(x) and prove its monotonicity
def g (x a : ℝ) [ne_zero : a ≠ 0] := (f(x) - f(a)) / (x - a)

theorem monotonicity_g (a : ℝ) (h : 0 < a) : 
  ∀ x, (0 < x ∧ x < a) ∨ (x > a) → (g x a).deriv < 0 :=
sorry

end range_of_c_monotonicity_g_l461_461359


namespace find_ff_inv_9_l461_461364

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.logb 3 x else 2^x

theorem find_ff_inv_9 : f(f (1 / 9)) = 1 / 4 :=
by
  sorry

end find_ff_inv_9_l461_461364


namespace ratio_of_height_and_radius_l461_461470

theorem ratio_of_height_and_radius 
  (h r : ℝ) 
  (V_X V_Y : ℝ)
  (hY rY : ℝ)
  (k : ℝ)
  (h_def : V_X = π * r^2 * h)
  (hY_def : hY = k * h)
  (rY_def : rY = k * r)
  (half_filled_VY : V_Y = 1/2 * π * rY^2 * hY)
  (V_X_value : V_X = 2)
  (V_Y_value : V_Y = 64):
  k = 4 :=
by
  sorry

end ratio_of_height_and_radius_l461_461470


namespace freddy_age_l461_461656

theorem freddy_age
  (mat_age : ℕ)  -- Matthew's age
  (reb_age : ℕ)  -- Rebecca's age
  (fre_age : ℕ)  -- Freddy's age
  (h1 : mat_age = reb_age + 2)
  (h2 : fre_age = mat_age + 4)
  (h3 : mat_age + reb_age + fre_age = 35) :
  fre_age = 15 :=
by sorry

end freddy_age_l461_461656


namespace solve_system_of_equations_l461_461904

theorem solve_system_of_equations (x y z : ℝ) :
  (2 * x^2 + 3 * y + 5 = 2 * real.sqrt (2 * z + 5)) ∧
  (2 * y^2 + 3 * z + 5 = 2 * real.sqrt (2 * x + 5)) ∧
  (2 * z^2 + 3 * x + 5 = 2 * real.sqrt (2 * y + 5))
  ↔ x = -1/2 ∧ y = -1/2 ∧ z = -1/2 :=
by sorry

end solve_system_of_equations_l461_461904


namespace area_triangle_ABG_l461_461831

theorem area_triangle_ABG (A B C D E G : Point) (h1 : dist A B = 5)
  (h2 : dist A C = 5) (h3 : angle A B C = 60) (h4 : dist A D = 1)
  (h5 : dist D C = 4) (h6 : centroid G A B C) (h7 : line_through E G D)
  (h8 : dist D E = 2): 
  area_of_triangle A B G = 6.25 :=
sorry

end area_triangle_ABG_l461_461831


namespace karan_borrowed_amount_l461_461469

noncomputable def Rs20500 : ℝ := 20500
noncomputable def interest_rate_1 : ℝ := 0.04
noncomputable def interest_rate_2 : ℝ := 0.07
noncomputable def duration_1 : ℝ := 5
noncomputable def duration_2 : ℝ := 5

theorem karan_borrowed_amount : ∃ P : ℝ, 
  P * (1 + interest_rate_1) ^ duration_1 * (1 + interest_rate_2) ^ duration_2 ≈ Rs20500 :=
by
    use 12016.77
    sorry

end karan_borrowed_amount_l461_461469


namespace fill_tank_time_l461_461580

theorem fill_tank_time :
  let length := 10
  let width := 6
  let depth := 5
  let rate := 5
  let volume := length * width * depth
  let time := volume / rate
  time = 60 :=
by
  let length := 10
  let width := 6
  let depth := 5
  let rate := 5
  let volume := length * width * depth
  let time := volume / rate
  have h_volume : volume = 300 := by norm_num
  have h_time : time = volume / rate := by rfl
  rwa [h_time, h_volume]
      at sorry

end fill_tank_time_l461_461580


namespace statistical_measures_unchanged_l461_461191

-- Define the given constants and assumptions
def total_members : ℕ := 30
def freq_ages (x : ℕ) : list (ℕ × ℕ) := [(13, 5), (14, 12), (15, x), (16, 11 - x), (17, 2)]

-- Define predicates for mode and median based on frequency table
def mode_unchanged (x : ℕ) : Prop :=
  ∀ x, (14, 12) ∈ freq_ages x ∧ (∀ a b, (a,b) ∈ freq_ages x → b ≤ 12)

def median_unchanged (x : ℕ) : Prop :=
  let sorted_ages := list.sort_by (λ y : ℕ × ℕ, y.snd) (freq_ages x) in
  sorted_ages.nth 14 = sorted_ages.nth 15

-- Define the main theorem based on the predicates
theorem statistical_measures_unchanged (x : ℕ) : mode_unchanged x ∧ median_unchanged x :=
by sorry

end statistical_measures_unchanged_l461_461191


namespace probability_calculation_correct_l461_461593

def total_balls : ℕ := 100
def white_balls : ℕ := 50
def green_balls : ℕ := 20
def yellow_balls : ℕ := 10
def red_balls : ℕ := 17
def purple_balls : ℕ := 3

def number_of_non_red_or_purple_balls : ℕ := total_balls - (red_balls + purple_balls)

def probability_of_non_red_or_purple : ℚ := number_of_non_red_or_purple_balls / total_balls

theorem probability_calculation_correct :
  probability_of_non_red_or_purple = 0.8 := 
  by 
    -- proof goes here
    sorry

end probability_calculation_correct_l461_461593


namespace unique_solution_l461_461062

variable {Z W λ : ℂ}

theorem unique_solution
  (hλ : |λ| ≠ 1)
  (hz : Z = (conj λ * W + conj W) / (1 - |λ|^2)) :
  (conj Z - λ * Z = W) :=
by
  sorry

end unique_solution_l461_461062


namespace count_integers_satisfying_condition_l461_461793

theorem count_integers_satisfying_condition :
  ({n : ℕ | 300 < n^2 ∧ n^2 < 1000}.card = 14) :=
by
  sorry

end count_integers_satisfying_condition_l461_461793


namespace point_P_satisfies_equation_l461_461996

def A := (-1, 0)
def B := (0, 1)
def C := (1, 0)
def P (x y : ℝ) := (x, y)

def line_eq_AB (x y : ℝ) : ℝ := x - y + 1
def line_eq_BC (x y : ℝ) : ℝ := x + y - 1
def line_eq_AC (y : ℝ) : ℝ := y

def d1 (x y : ℝ) := abs (line_eq_AB x y) / Real.sqrt 2
def d2 (x y : ℝ) := abs (line_eq_BC x y) / Real.sqrt 2
def d3 (y : ℝ) := abs y

theorem point_P_satisfies_equation (x y : ℝ) :
  (d1 x y) * (d2 x y) = (d3 y)^2 ↔ abs(x - y + 1) * abs(x + y - 1) = 2 * y^2 := 
by
  sorry

end point_P_satisfies_equation_l461_461996


namespace interest_rate_is_12_percent_l461_461602

-- Definitions
def SI : ℝ := 5400
def P : ℝ := 15000
def T : ℝ := 3

-- Theorem to prove the interest rate
theorem interest_rate_is_12_percent :
  SI = (P * 12 * T) / 100 :=
by
  sorry

end interest_rate_is_12_percent_l461_461602


namespace jello_mix_needed_per_pound_l461_461838

variable (bathtub_volume : ℝ) (gallons_per_cubic_foot : ℝ) 
          (pounds_per_gallon : ℝ) (cost_per_tablespoon : ℝ) 
          (total_cost : ℝ)

theorem jello_mix_needed_per_pound :
  bathtub_volume = 6 ∧
  gallons_per_cubic_foot = 7.5 ∧
  pounds_per_gallon = 8 ∧
  cost_per_tablespoon = 0.50 ∧
  total_cost = 270 →
  (total_cost / cost_per_tablespoon) / 
  (bathtub_volume * gallons_per_cubic_foot * pounds_per_gallon) = 1.5 :=
by
  sorry

end jello_mix_needed_per_pound_l461_461838


namespace maximum_area_of_triangle_l461_461848

noncomputable def area_of_triangle (A B C : Real × Real) : Real :=
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem maximum_area_of_triangle :
  let A := (0 : Real, 3 : Real)
  let B := (3 : Real, 0 : Real)
  ∃ p : Real, 0 ≤ p ∧ p ≤ 3 ∧
  let C := (p, p^2 - 4 * p + 3)
  area_of_triangle A B C = 22.5 := by
sorry

end maximum_area_of_triangle_l461_461848


namespace functional_equation_l461_461860

def f (x : ℝ) : ℝ := x + 1

theorem functional_equation (f : ℝ → ℝ) (h1 : f 0 = 1) (h2 : ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) :
  f = (λ x, x + 1) :=
by
  sorry

end functional_equation_l461_461860


namespace dodecagon_eq_triangle_area_l461_461947

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def dodecagon_side (a b c : ℝ) : ℝ :=
  let area := triangle_area a b c
  in Real.sqrt (area / (3 * (2 + Real.sqrt 3)))

theorem dodecagon_eq_triangle_area (a b c : ℝ) :
  let s := (a + b + c) / 2
      area_triangle := triangle_area a b c
      area_dodecagon := 3 * (2 + Real.sqrt 3) * (dodecagon_side a b c)^2
  in area_triangle = area_dodecagon :=
by sorry

end dodecagon_eq_triangle_area_l461_461947


namespace original_population_is_l461_461187

variable (P : ℕ)

def reduced_by_bombardment (P : ℕ) : ℝ :=
  0.85 * P

def reduced_by_fear (remaining_population : ℝ) : ℝ :=
  0.75 * remaining_population

def final_population (P : ℕ) : ℝ :=
  reduced_by_fear (reduced_by_bombardment P)

theorem original_population_is (h : final_population P = 4555) : P = 7143 :=
by
  sorry

end original_population_is_l461_461187


namespace question1_equivalent_question2_equivalent_l461_461734

theorem question1_equivalent (h1: /* Intersection condition */) (h2: /* Distance condition */) : 
  (∀ P : point, distance P (5, 0) 4 → (equation_line P = equation_line x = 2 ∨ equation_line P = equation_line 4x - 3y - 5 = 0))  := 
sorry

theorem question2_equivalent (h1: /* Intersection condition */) (h3: /* Perpendicular condition */) : 
  (∀ l : line, perpendicular_to_AB l → equation_line l = equation_line 3x - 4y - 2 = 0) := 
sorry

end question1_equivalent_question2_equivalent_l461_461734


namespace range_of_c_monotonicity_of_g_l461_461347

noncomputable def f (x: ℝ) : ℝ := 2 * Real.log x + 1

theorem range_of_c (c: ℝ) : (∀ x > 0, f x ≤ 2 * x + c) → c ≥ -1 := by
  sorry

noncomputable def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

theorem monotonicity_of_g (a: ℝ) (ha: a > 0) : 
  (∀ x > 0, x ≠ a → ((x < a → g x a < g a a) ∧ (x > a → g x a < g a a))) := by
  sorry

end range_of_c_monotonicity_of_g_l461_461347


namespace largest_prime_sum_of_two_primes_under_30_is_19_l461_461126

def is_prime (n : ℕ) : Prop := nat.prime n

def largest_prime_sum_of_two_primes_under_30 : Prop :=
  ∃ p q : ℕ, p < 30 ∧ is_prime p ∧ is_prime q ∧ p = 2 + q ∧ (∀ r, r < 30 ∧ is_prime r ∧ (∃ s, is_prime s ∧ r = 2 + s) → r ≤ p)

theorem largest_prime_sum_of_two_primes_under_30_is_19 : largest_prime_sum_of_two_primes_under_30 :=
by {
  sorry
}

end largest_prime_sum_of_two_primes_under_30_is_19_l461_461126


namespace jan_paid_amount_l461_461042

def number_of_roses (dozens : Nat) : Nat := dozens * 12

def total_cost (number_of_roses : Nat) (cost_per_rose : Nat) : Nat := number_of_roses * cost_per_rose

def discounted_price (total_cost : Nat) (discount_percentage : Nat) : Nat := total_cost * discount_percentage / 100

theorem jan_paid_amount :
  let dozens := 5
  let cost_per_rose := 6
  let discount_percentage := 80
  number_of_roses dozens = 60 →
  total_cost (number_of_roses dozens) cost_per_rose = 360 →
  discounted_price (total_cost (number_of_roses dozens) cost_per_rose) discount_percentage = 288 :=
by
  intros
  sorry

end jan_paid_amount_l461_461042


namespace quadratic_false_statements_l461_461735

theorem quadratic_false_statements 
  (a b c : ℝ) 
  (x1 x2 : ℂ) 
  (h_eq : a ≠ 0)
  (h_roots : a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) :
  (¬ (ax^2 + bx + c = 0 ∧ (x1 - x2 = 0 ∧ Im x1 ≠ 0))) ∧ (a * x1^2 + b * x1 + c = a * (x1-x1)(x1-x2)) :=
sorry

end quadratic_false_statements_l461_461735


namespace distinct_integer_solutions_le_8_l461_461065

noncomputable def f (x : ℤ) : Polynomial ℤ := sorry -- placeholder for the monic polynomial of degree 1991
noncomputable def g (x : ℤ) : Polynomial ℤ := f(x)^2 - 9

theorem distinct_integer_solutions_le_8 : 
  (∀ x : ℤ, g(x) = 0 → x = 0) ∧ (∃ n : ℕ, n ≤ 8) :=
sorry

end distinct_integer_solutions_le_8_l461_461065


namespace sum_primes_cong_mod_l461_461719

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define condition p satisfying $p \equiv 1 \pmod{6}$
def is_cong_1_mod_6 (p : ℕ) : Prop := p % 6 = 1

-- Define condition p satisfying $p \equiv 6 \pmod{7}$
def is_cong_6_mod_7 (p : ℕ) : Prop := p % 7 = 6

-- Define the main proposition for Lean
theorem sum_primes_cong_mod (s : ℕ) :
  s = ∑ p in Ico 1 201, (if is_prime p ∧ is_cong_1_mod_6 p ∧ is_cong_6_mod_7 p then p else 0) -> s = 430 := by
  sorry

end sum_primes_cong_mod_l461_461719


namespace kittens_percentage_is_80_l461_461466

-- Define the problem conditions
def number_of_matt_cats : ℕ := 12
def number_of_female_cats : ℕ := 7
def kittens_per_female_cat : ℕ := 9
def number_of_kittens_sold : ℕ := 15

-- Compute the derived values
def total_kittens : ℕ := number_of_female_cats * kittens_per_female_cat
def remaining_kittens : ℕ := total_kittens - number_of_kittens_sold
def initial_total_cats : ℕ := number_of_matt_cats + total_kittens
def remaining_total_cats : ℕ := initial_total_cats - number_of_kittens_sold

-- Prove that the percentage of remaining cats that are kittens is 80%
theorem kittens_percentage_is_80 : 
  (remaining_kittens * 100 / remaining_total_cats).nat = 80 :=
  sorry

end kittens_percentage_is_80_l461_461466


namespace distance_between_parallel_lines_l461_461918

theorem distance_between_parallel_lines : 
  let a := 2
  let b := 2
  let c1 := 4
  let c2 := -5
  let d := (abs (c2 - c1)) / (real.sqrt (a ^ 2 + b ^ 2))
  d = 9 * real.sqrt 2 / 4 :=
by
  sorry

end distance_between_parallel_lines_l461_461918


namespace Jennifer_has_24_dollars_left_l461_461043

def remaining_money (initial amount: ℕ) (spent_sandwich spent_museum_ticket spent_book: ℕ) : ℕ :=
  initial - (spent_sandwich + spent_museum_ticket + spent_book)

theorem Jennifer_has_24_dollars_left :
  remaining_money 180 (1/5*180) (1/6*180) (1/2*180) = 24 :=
by
  sorry

end Jennifer_has_24_dollars_left_l461_461043


namespace cos_555_equals_neg_sqrt6_plus_sqrt2_div4_l461_461941

theorem cos_555_equals_neg_sqrt6_plus_sqrt2_div4 :
  cos (555 * (Real.pi / 180)) = - ((Real.sqrt 6 + Real.sqrt 2) / 4) := by
  sorry

end cos_555_equals_neg_sqrt6_plus_sqrt2_div4_l461_461941


namespace area_B_correct_l461_461650

noncomputable def area_B : ℝ :=
  let circle_area_quarter := (π * (15 ^ 2)) / 4
  let square_area_quarter := 30 ^ 2 / 4
  2 * (square_area_quarter - circle_area_quarter) + square_area_quarter

theorem area_B_correct : area_B = 675 - 112.5 * π := by
  sorry

end area_B_correct_l461_461650


namespace money_last_duration_l461_461864

-- Defining the conditions
def money_from_mowing : ℕ := 14
def money_from_weed_eating : ℕ := 26
def money_spent_per_week : ℕ := 5

-- Theorem statement to prove Mike's money will last 8 weeks
theorem money_last_duration : (money_from_mowing + money_from_weed_eating) / money_spent_per_week = 8 := by
  sorry

end money_last_duration_l461_461864


namespace perpendicular_condition_line_through_point_l461_461779

-- Definitions for lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y = 6
def l2 (m : ℝ) (x y : ℝ) : Prop := m * x + y = 3

-- Part 1: Prove that l1 is perpendicular to l2 if and only if m = -3 or m = 0
theorem perpendicular_condition (m : ℝ) : 
  (∀ (x : ℝ), ∀ (y : ℝ), (l1 m x y ∧ l2 m x y) → (m = 0 ∨ m = -3)) :=
sorry

-- Part 2: Prove the equations of line l given the conditions
theorem line_through_point (m : ℝ) (l : ℝ → ℝ → Prop) : 
  (∀ (P : ℝ × ℝ), (P = (1, 2*m)) → (l2 m P.1 P.2) → 
  ((∀ (x y : ℝ), l x y → 2 * x - y = 0) ∨ (∀ (x y: ℝ), l x y → x + 2 * y - 5 = 0))) :=
sorry

end perpendicular_condition_line_through_point_l461_461779


namespace solution_l461_461142

def MishaKolyaVitya (K M B : ℕ) : Prop :=
  M = K + 943 ∧
  B = M + 127 ∧
  M + K = B + 479

theorem solution : ∃ (K M B : ℕ), MishaKolyaVitya K M B ∧ K = 606 ∧ M = 1549 ∧ B = 1676 :=
by
  use 606, 1549, 1676
  split; -- prove MishaKolyaVitya and equalities separately
  { split,
    { exact rfl, },
    split,
    { exact rfl, },
    { exact rfl, } },
  { split,
    { exact rfl, },
    split,
    { exact rfl, },
    { exact rfl } }

end solution_l461_461142


namespace first_digit_base9_of_21121212221122211121_3_l461_461912

noncomputable def base3_to_base10 (digits : List ℕ) : ℕ :=
  digits.reverse.enum.map (λ ⟨i, d⟩, d * 3^i).sum

noncomputable def base10_to_base9 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else let rec aux (n : ℕ) : List ℕ :=
    if n = 0 then [] else aux (n / 9) ++ [n % 9]
  aux n
  
def first_digit_base9 (n : ℕ) : ℕ :=
  (base10_to_base9 n).head

theorem first_digit_base9_of_21121212221122211121_3 :
  first_digit_base9 (base3_to_base10 [2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1]) = 5 :=
sorry

end first_digit_base9_of_21121212221122211121_3_l461_461912


namespace cube_root_simplify_l461_461104

theorem cube_root_simplify : Real.cbrt 2744000 = 140 := by
  sorry

end cube_root_simplify_l461_461104


namespace matrix_satisfies_cross_product_l461_461292

open Matrix

def cross_product (u v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => match i with
  | 0 => u 1 * v 2 - u 2 * v 1 
  | 1 => u 2 * v 0 - u 0 * v 2
  | 2 => u 0 * v 1 - u 1 * v 0

def M := fun (i j : Fin 3) => 
  if (i = 0 && j = 1) then -7 else
  if (i = 0 && j = 2) then -3 else
  if (i = 1 && j = 0) then 7 else
  if (i = 1 && j = 2) then -4 else
  if (i = 2 && j = 0) then 3 else
  if (i = 2 && j = 1) then 4 else 0

theorem matrix_satisfies_cross_product (v : Fin 3 → ℝ) :
  (mulVec (of fun (i j : Fin 3) => M i j) v) = (cross_product ![4, -3, 7] v) :=
sorry

end matrix_satisfies_cross_product_l461_461292


namespace solution_pairs_l461_461676

theorem solution_pairs (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) := 
by 
  sorry

end solution_pairs_l461_461676


namespace place_100_points_symmetric_and_non_collinear_l461_461434

theorem place_100_points_symmetric_and_non_collinear : 
  ∃ P : Fin 100 → ℝ × ℝ, 
    (∀ i j k : Fin 100, i ≠ j ∧ i ≠ k ∧ j ≠ k → ¬ collinear {P i, P j, P k}) ∧ 
    (∀ n : Fin (100 + 1), ∃ l : ℝ, ∃ m : ℝ, axis_of_symmetry (λ i, P (i : Fin 100)) n l m) :=
sorry

end place_100_points_symmetric_and_non_collinear_l461_461434


namespace find_N_with_12_divisors_l461_461211

-- Given conditions
variables {N d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 : ℕ}

-- Condition 1: N has exactly 12 positive divisors.
def has_exactly_12_divisors (N : ℕ) :=
  let divisors := (List.range (N + 1)).filter (N % · == 0)
  divisors.length = 12

-- Condition 2: d_4-1th divisor equals (d_1 + d_2 + d_4) * d_8
def specific_divisor_condition (N d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 : ℕ) :=
  let divisors := (List.range (N + 1)).filter (N % · == 0)
  divisors.nth (d4 - 1) = some ((d1 + d2 + d4) * d8)

-- Main proof statement
theorem find_N_with_12_divisors (h1 : has_exactly_12_divisors N)
  (h2 : specific_divisor_condition N d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12) :
  N = 1989 := 
  sorry

end find_N_with_12_divisors_l461_461211


namespace general_term_S1_general_term_S2_l461_461783

noncomputable def sequence_a1 (n : ℕ) (S : ℕ → ℕ) : ℕ :=
if h : n = 1 then S 1 else S n - S (n - 1)

noncomputable def S1 : ℕ → ℕ := λ n, n^2

noncomputable def S2 : ℕ → ℕ := λ n, n^2 + n + 1

theorem general_term_S1 (n : ℕ) : 
  sequence_a1 n S1 = 2 * n - 1 := 
sorry

theorem general_term_S2 (n : ℕ) : 
  sequence_a1 n S2 = if n = 1 then 3 else 2 * n := 
sorry

end general_term_S1_general_term_S2_l461_461783


namespace num_rel_prime_to_15_l461_461257

theorem num_rel_prime_to_15 : 
  {a : ℕ | a < 15 ∧ Int.gcd 15 a = 1}.card = 8 := by 
  sorry

end num_rel_prime_to_15_l461_461257


namespace problem1_problem2_l461_461334

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (a * Real.log x) / (x + 1) + b / x

theorem problem1 
  (h₁ : ∀ x, Deriv (f x a b) 1 = -1/2)
  (h₂ : f 1 a b = 1) : 
  (a = 1) ∧ (b = 1) :=
sorry

theorem problem2 (x : ℝ) (h₁ : 0 < x) (h₂ : x ≠ 1) :
    (\frac{Real.log x}{x + 1} + 1 / x > Real.log x / (x - 1)) := 
sorry

end problem1_problem2_l461_461334


namespace find_x_values_l461_461712

theorem find_x_values : ∀ (x : ℝ), 
  10 ≤ x →
  (8 / (real.sqrt (x - 10) - 10) + 
   2 / (real.sqrt (x - 10) - 5) + 
   9 / (real.sqrt (x - 10) + 5) + 
   15 / (real.sqrt (x - 10) + 10) = 0) →
  (x = 10 ∨ x ≈ 85.7353) :=
by
  sorry

end find_x_values_l461_461712


namespace percentage_loss_is_20_percent_l461_461604

-- Define the given conditions
def gain := 20 / 100
def selling_price_for_20_articles := 60
def selling_price_for_19_999997500000312_articles := 40
def num_articles := 20
def num_articles_approx := 19.999997500000312

-- Definition of cost price based on the first condition
def cost_price_for_20_articles := selling_price_for_20_articles / (1 + gain)
def cost_price_per_article := cost_price_for_20_articles / num_articles

-- Approximate cost price for nearly 20 articles
def cost_price_for_approx_articles := cost_price_per_article * num_articles_approx

-- Defining the loss and percentage loss
def loss := cost_price_for_approx_articles - selling_price_for_19_999997500000312_articles
def percentage_loss := (loss / cost_price_for_approx_articles) * 100

-- Theorem statement
theorem percentage_loss_is_20_percent : percentage_loss = 20 := by
  sorry

end percentage_loss_is_20_percent_l461_461604


namespace candy_probability_l461_461601

/-- 
A jar has 15 red candies, 15 blue candies, and 10 green candies. Terry picks three candies at random,
then Mary picks three of the remaining candies at random. Calculate the probability that they get 
the same color combination, irrespective of order, expressed as a fraction $m/n,$ where $m$ and $n$ 
are relatively prime positive integers. Find $m+n.$ -/
theorem candy_probability :
  let num_red := 15
  let num_blue := 15
  let num_green := 10
  let total_candies := num_red + num_blue + num_green
  let Terry_picks := 3
  let Mary_picks := 3
  let prob_equal_comb := (118545 : ℚ) / 2192991
  let m := 118545
  let n := 2192991
  m + n = 2310536 := sorry

end candy_probability_l461_461601


namespace compare_trigonometric_values_l461_461052

noncomputable def a : ℝ := (1 / 2) * real.cos (real.pi / 180 * 7) + (real.sqrt 3 / 2) * real.sin (real.pi / 180 * 7)
noncomputable def b : ℝ := (2 * real.tan (real.pi / 180 * 19)) / (1 - (real.tan (real.pi / 180 * 19))^2)
noncomputable def c : ℝ := real.sqrt ((1 - real.cos (real.pi / 180 * 72)) / 2)

theorem compare_trigonometric_values : b > a ∧ a > c := 
by 
  sorry

end compare_trigonometric_values_l461_461052


namespace smallest_number_among_list_l461_461226

theorem smallest_number_among_list : 
  let nums := [1, 0, -2, -6] in 
  ∃ n ∈ nums, ∀ m ∈ nums, n ≤ m ∧ n = -6 :=
by {
  sorry
}

end smallest_number_among_list_l461_461226


namespace quadratic_value_range_l461_461132

theorem quadratic_value_range (y : ℝ) (h : y^3 - 6 * y^2 + 11 * y - 6 < 0) : 
  1 ≤ y^2 - 4 * y + 5 ∧ y^2 - 4 * y + 5 ≤ 2 := 
sorry

end quadratic_value_range_l461_461132


namespace probability_pq_satisfies_equation_l461_461486

theorem probability_pq_satisfies_equation :
  let p_values := {p : ℕ | 1 ≤ p ∧ p ≤ 20 ∧ ∃ q : ℤ, p * q - 6 * p - 3 * q = 6} in
  (p_values.to_finset.card : ℚ) / 20 = 7 / 20 :=
by
  intro p_values
  have h : p_values = {4, 5, 6, 7, 9, 11, 15}, sorry
  rw [h, Finset.card_insert_of_not_mem, Finset.card_insert_of_not_mem, Finset.card_insert_of_not_mem,
    Finset.card_insert_of_not_mem, Finset.card_insert_of_not_mem, Finset.card_insert_of_not_mem,
    Finset.card_singleton, Finset.card_empty]
  norm_cast
  exact by norm_num

end probability_pq_satisfies_equation_l461_461486


namespace total_action_figures_l461_461220

def action_figures_per_shelf : ℕ := 11
def number_of_shelves : ℕ := 4

theorem total_action_figures : action_figures_per_shelf * number_of_shelves = 44 := 
by 
  have h1: action_figures_per_shelf = 11 := rfl
  have h2: number_of_shelves = 4 := rfl
  calc
    action_figures_per_shelf * number_of_shelves
      = 11 * 4 : by rw [h1, h2]
      ... = 44 : by norm_num

end total_action_figures_l461_461220


namespace cell_phones_in_Delaware_l461_461521

theorem cell_phones_in_Delaware (population : ℕ) (phones_per_1000_people : ℕ)
  (h_population : population = 974000)
  (h_phones_per_1000 : phones_per_1000_people = 673) :
  ∃ cell_phones : ℕ, cell_phones = population / 1000 * phones_per_1000_people ∧ cell_phones = 655502 :=
by {
  use 974 * 673,
  split,
  { rw [h_population, h_phones_per_1000], norm_num },
  { norm_num }
}

end cell_phones_in_Delaware_l461_461521


namespace fixed_point_min_value_l461_461122

theorem fixed_point_min_value {a m n : ℝ} (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hm_pos : 0 < m) (hn_pos : 0 < n)
  (h : 3 * m + n = 1) : (1 / m + 3 / n) = 12 := sorry

end fixed_point_min_value_l461_461122


namespace smallest_perfect_square_divisible_by_5_and_7_l461_461560

theorem smallest_perfect_square_divisible_by_5_and_7 
  (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = k^2)
  (h3 : 5 ∣ n)
  (h4 : 7 ∣ n) : 
  n = 1225 :=
sorry

end smallest_perfect_square_divisible_by_5_and_7_l461_461560


namespace speed_of_train_is_84_l461_461217

namespace TrainProblem

-- Define the conditions for the problem
def train_length : ℝ := 150 -- in meters
def man_speed_kmph : ℝ := 6 -- in kmph
def pass_time : ℝ := 6 -- in seconds

-- Define the speed conversion from kmph to m/s
def kmph_to_mps (kmph : ℝ) : ℝ := kmph * 1000 / 3600

-- Convert man's speed to m/s
def man_speed_mps := kmph_to_mps man_speed_kmph

-- Define the relative speed calculation who the train passes the man
def relative_speed := train_length / pass_time

-- Define the speed of the train in m/s
def train_speed_mps := relative_speed - man_speed_mps

-- Convert the train's speed back to kmph
def train_speed_kmph := train_speed_mps * 3600 / 1000

-- Theorem to prove
theorem speed_of_train_is_84 : train_speed_kmph = 84 := sorry

end TrainProblem

end speed_of_train_is_84_l461_461217


namespace find_f_of_2_l461_461367

def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem find_f_of_2 (a b : ℝ)
  (h1 : 3 + 2 * a + b = 0)
  (h2 : 1 + a + b + a^2 = 10)
  (ha : a = 4)
  (hb : b = -11) :
  f 2 a b = 18 := by {
  -- We assume the values of a and b provided by the user as the correct pair.
  sorry
}

end find_f_of_2_l461_461367


namespace ratio_of_times_l461_461530

theorem ratio_of_times (V_b V_s : ℕ) (hb : V_b = 63) (hs : V_s = 21) :
  let V_up := V_b - V_s,
      V_down := V_b + V_s
  in V_up ≠ 0 ∧ V_down ≠ 0 → (V_down / V_up = 2) :=
by
  intros _ _ hVb hVs
  simp [hVb, hVs]
  sorry

end ratio_of_times_l461_461530


namespace find_angle_A_find_b_plus_c_l461_461398

-- Define the conditions for the problem
variables (a b c S : ℝ)
variables (A B C : ℝ)
variables (triangle_ABC : ∃ (A B C : ℝ), A + B + C = π)
variables (side_condition : ∀ {A B C}, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
variables (area : S = (1/2) * b * c * sin A)
variables (relation : a * sin B = sqrt 3 * b * cos A)
variables (a_value : a = sqrt 3)
variables (S_value : S = sqrt 3 / 2)

-- Define the proof problem for question (1)
theorem find_angle_A : A = π / 3 :=
by sorry

-- Define the condition for bc = 2 derived from given S
variables (bc_condition : b * c = 2)

-- Define the proof problem for question (2)
theorem find_b_plus_c : b + c = 3 :=
by sorry

end find_angle_A_find_b_plus_c_l461_461398


namespace coprime_integers_lt_15_l461_461273

theorem coprime_integers_lt_15 : ∃ (S : Finset ℕ), S.card = 8 ∧ (∀ a ∈ S, a < 15 ∧ Nat.gcd a 15 = 1) :=
by
  sorry

end coprime_integers_lt_15_l461_461273


namespace solve_for_A_l461_461731

variable (a b : ℝ) 

theorem solve_for_A (A : ℝ) (h : (5 * a + 3 * b)^2 = (5 * a - 3 * b)^2 + A) : 
  A = 60 * a * b := by
  sorry

end solve_for_A_l461_461731


namespace fractions_comparison_l461_461241

theorem fractions_comparison : 
  (99 / 100 < 100 / 101) ∧ (100 / 101 > 199 / 201) ∧ (99 / 100 < 199 / 201) :=
by sorry

end fractions_comparison_l461_461241


namespace vertical_asymptote_l461_461296

theorem vertical_asymptote (x : ℝ) : (4 * x + 6 = 0) -> x = -3 / 2 :=
by
  sorry

end vertical_asymptote_l461_461296


namespace cost_per_trip_l461_461474

theorem cost_per_trip (cost_per_pass : ℕ) (num_passes : ℕ) (trips_oldest : ℕ) (trips_youngest : ℕ) :
    cost_per_pass = 100 →
    num_passes = 2 →
    trips_oldest = 35 →
    trips_youngest = 15 →
    (cost_per_pass * num_passes) / (trips_oldest + trips_youngest) = 4 := by
  sorry

end cost_per_trip_l461_461474


namespace max_gcd_seq_l461_461248

theorem max_gcd_seq (a : ℕ → ℕ) (d : ℕ → ℕ) :
  (∀ n : ℕ, a n = 121 + n^2) →
  (∀ n : ℕ, d n = Nat.gcd (a n) (a (n + 1))) →
  ∃ m : ℕ, ∀ n : ℕ, d n ≤ d m ∧ d m = 99 :=
by
  sorry

end max_gcd_seq_l461_461248


namespace percentage_of_number_l461_461185

theorem percentage_of_number (P : ℝ) (h : 0.10 * 3200 - 190 = P * 650) :
  P = 0.2 :=
sorry

end percentage_of_number_l461_461185


namespace fractions_product_l461_461636

theorem fractions_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by 
  sorry

end fractions_product_l461_461636


namespace range_of_c_monotonicity_of_g_l461_461348

noncomputable def f (x: ℝ) : ℝ := 2 * Real.log x + 1

theorem range_of_c (c: ℝ) : (∀ x > 0, f x ≤ 2 * x + c) → c ≥ -1 := by
  sorry

noncomputable def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

theorem monotonicity_of_g (a: ℝ) (ha: a > 0) : 
  (∀ x > 0, x ≠ a → ((x < a → g x a < g a a) ∧ (x > a → g x a < g a a))) := by
  sorry

end range_of_c_monotonicity_of_g_l461_461348


namespace average_speed_round_trip_l461_461099

theorem average_speed_round_trip (D : ℝ) (hD : D > 0) :
  let time_uphill := D / 5
  let time_downhill := D / 100
  let total_distance := 2 * D
  let total_time := time_uphill + time_downhill
  let average_speed := total_distance / total_time
  average_speed = 200 / 21 :=
by
  sorry

end average_speed_round_trip_l461_461099


namespace range_of_c_monotonicity_g_l461_461361

-- Define the given function f(x)
def f (x : ℝ) := 2 * real.log x + 1

-- Part 1: Define the hypothesis for the range of c
theorem range_of_c :
  ∀ x : ℝ, f(x) ≤ 2 * x + c ↔ c ∈ set.Ici (-1) :=
sorry

-- Part 2: Define the function g(x) and prove its monotonicity
def g (x a : ℝ) [ne_zero : a ≠ 0] := (f(x) - f(a)) / (x - a)

theorem monotonicity_g (a : ℝ) (h : 0 < a) : 
  ∀ x, (0 < x ∧ x < a) ∨ (x > a) → (g x a).deriv < 0 :=
sorry

end range_of_c_monotonicity_g_l461_461361


namespace alice_unanswered_questions_l461_461622

-- Declare variables for the proof
variables (c w u : ℕ)

-- State the problem in Lean
theorem alice_unanswered_questions :
  50 + 5 * c - 2 * w = 100 ∧
  40 + 7 * c - w - u = 120 ∧
  6 * c + 3 * u = 130 ∧
  c + w + u = 25 →
  u = 20 :=
by
  intros h
  sorry

end alice_unanswered_questions_l461_461622


namespace earning_hours_per_week_l461_461799

theorem earning_hours_per_week (totalEarnings : ℝ) (originalWeeks : ℝ) (missedWeeks : ℝ) 
  (originalHoursPerWeek : ℝ) : 
  missedWeeks = 3 → originalWeeks = 15 → originalHoursPerWeek = 25 → totalEarnings = 3750 → 
  (totalEarnings / ((totalEarnings / (originalWeeks * originalHoursPerWeek)) * (originalWeeks - missedWeeks))) = 31.25 :=
by
  intros
  sorry

end earning_hours_per_week_l461_461799


namespace team_order_l461_461407

-- Define the points of teams
variables (A B C D : ℕ)

-- State the conditions
def condition1 := A + C = B + D
def condition2 := B + A + 5 ≤ D + C
def condition3 := B + C ≥ A + D + 3

-- Statement of the theorem
theorem team_order (h1 : condition1 A B C D) (h2 : condition2 A B C D) (h3 : condition3 A B C D) :
  C > D ∧ D > B ∧ B > A :=
sorry

end team_order_l461_461407


namespace students_passed_both_tests_l461_461997

theorem students_passed_both_tests :
  ∀ (total students_passed_long_jump students_passed_shot_put students_failed_both x : ℕ),
    total = 50 →
    students_passed_long_jump = 40 →
    students_passed_shot_put = 31 →
    students_failed_both = 4 →
    (students_passed_long_jump - x) + (students_passed_shot_put - x) + x + students_failed_both = total →
    x = 25 :=
by intros total students_passed_long_jump students_passed_shot_put students_failed_both x
   intro total_eq students_passed_long_jump_eq students_passed_shot_put_eq students_failed_both_eq sum_eq
   sorry

end students_passed_both_tests_l461_461997


namespace number_of_real_solutions_to_system_l461_461645

theorem number_of_real_solutions_to_system :
  (∃ (x y z w : ℝ), x = z + w + zw - zw * x ∧ 
                    y = w + x + wx - wx * y ∧ 
                    z = x + y + xy - xy * z ∧ 
                    w = y + z + yz - yz * w) = 5 :=
by
  sorry

end number_of_real_solutions_to_system_l461_461645


namespace directly_proportional_l461_461623

-- Defining conditions
def A (x y : ℝ) : Prop := y = x + 8
def B (x y : ℝ) : Prop := (2 / (5 * y)) = x
def C (x y : ℝ) : Prop := (2 / 3) * x = y

-- Theorem stating that in the given equations, equation C shows direct proportionality
theorem directly_proportional (x y : ℝ) : C x y ↔ (∃ k : ℝ, k ≠ 0 ∧ y = k * x) :=
by
  sorry

end directly_proportional_l461_461623


namespace find_k_l461_461279

-- Given definition for a quadratic expression that we want to be a square of a binomial
def quadratic_expression (x k : ℝ) := x^2 - 20 * x + k

-- The binomial square matching.
def binomial_square (x b : ℝ) := (x + b)^2

-- Statement to prove that k = 100 makes the quadratic_expression to be a square of binomial
theorem find_k :
  (∃ k : ℝ, ∀ x : ℝ, quadratic_expression x k = binomial_square x (-10)) ↔ k = 100 :=
by
  sorry

end find_k_l461_461279


namespace fraction_product_simplification_l461_461639

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by
  sorry

end fraction_product_simplification_l461_461639


namespace two_marble_groups_count_l461_461144

def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

def total_combinations : ℕ :=
  let identical_yellow := 1
  let distinct_pairs := num_combinations 4 2
  identical_yellow + distinct_pairs

theorem two_marble_groups_count :
  total_combinations = 7 :=
by
  dsimp [total_combinations, num_combinations]
  rw [Nat.choose]
  norm_num
  sorry

end two_marble_groups_count_l461_461144


namespace area_shaded_region_l461_461506

theorem area_shaded_region (leg_large leg_small : ℝ) (h1 : leg_large = 5) (h2 : leg_small = 3) :
  let area (leg : ℝ) := (1 / 2) * leg * leg in
  area leg_large - area leg_small = 8 :=
by
  -- Definitions for areas of triangles using provided leg lengths
  let area := λ leg : ℝ, (1 / 2) * leg * leg
  have h_area_large : area leg_large = (1 / 2) * (5 : ℝ) * 5, from by simp [h1, area]
  have h_area_small : area leg_small = (1 / 2) * (3 : ℝ) * 3, from by simp [h2, area]
  -- Calculate and prove shaded area
  calc
    (1 / 2) * 5 * 5 - (1 / 2) * 3 * 3 = 12.5 - 4.5 := by simp
    ... = 8 := by norm_num

end area_shaded_region_l461_461506


namespace permutation_inequality_l461_461068

variables (p q : ℝ) (n : ℕ) (a b : Fin n.succ → ℝ)

theorem permutation_inequality (h₀ : 0 < p) 
  (ha : ∀ i, p ≤ a i ∧ a i ≤ q)
  (hb : ∃ σ : Equiv.Perm (Fin n.succ), ∀ i, b i = a (σ i)) :
  n + 1 ≤ (∑ i, a i / b i) ∧ (∑ i, a i / b i) ≤ n + 1 + ⌊(n + 1) / 2⌋ * (sqrt (p / q) - sqrt (q / p))^2 := 
sorry

end permutation_inequality_l461_461068


namespace triangle_ratio_l461_461397

theorem triangle_ratio
  (A B C D E P : Type)
  [plane_geometry : geometry Plane]
  (triangle_ABC : is_triangle A B C)
  (CE : line)
  (AD : line)
  (intersection_CE_AD : P = line_intersection CE AD)
  (CD_ratio : ratio (segment_length C D) (segment_length D B) = 2)
  (AE_ratio : ratio (segment_length A E) (segment_length E B) = 1)
  (angle_AEB_right : right_angle A E B) :
  (ratio (segment_length C P) (segment_length P E) = 5) :=
sorry

end triangle_ratio_l461_461397


namespace ratio_of_bases_l461_461611

noncomputable theory
open Real

-- Definitions based on the problem conditions
def R := ℝ  -- Radius of the larger base
def r := ℝ  -- Radius of the smaller base
def s := ℝ  -- Radius of the sphere

-- Geometric mean relationship
def geometric_mean (R r : ℝ) : ℝ := sqrt (R * r)

-- Volume of a sphere
def volume_sphere (s : ℝ) : ℝ := (4 / 3) * π * s^3

-- Volume of a truncated cone
def volume_truncated_cone (R r H : ℝ) : ℝ := (1 / 3) * π * (R^2 + R * r + r^2) * H

-- The given condition: volume of truncated cone is three times that of the sphere
def volume_relation (R r s H : ℝ) : Prop :=
  volume_truncated_cone R r 2 * s * (R / (R - r) + r / (R - r)) = 3 * volume_sphere s

-- Theorem to prove the ratio
theorem ratio_of_bases (R r s H : ℝ) (h : volume_relation R r s H)
  (h_geom : s = geometric_mean R r) : R / r = (4 + sqrt 13) / 3 :=
sorry

end ratio_of_bases_l461_461611


namespace part1_part2_l461_461356

-- Definition of the function f.
def f (x: ℝ) : ℝ := 2 * Real.log x + 1

-- Definition of the function g.
def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

-- Part 1: Prove that c ≥ -1 given f(x) ≤ 2x + c.
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
by
  -- Proof is omitted.
  sorry

-- Part 2: Prove that g(x) is monotonically decreasing on (0, a) and (a, +∞) given a > 0.
theorem part2 (a : ℝ) : a > 0 → (∀ x : ℝ, x > 0 → x ≠ a → 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo 0 a → x2 ∈ Ioo 0 a → x1 < x2 → g x2 a < g x1 a) ∧ 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x2 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x1 < x2 → g x2 a < g x1 a)) :=
by
  -- Proof is omitted.
  sorry

end part1_part2_l461_461356


namespace contrapositive_false_of_implication_false_l461_461760

variable (p q : Prop)

-- The statement we need to prove: If "if p then q" is false, 
-- then "if not q then not p" must be false.
theorem contrapositive_false_of_implication_false (h : ¬ (p → q)) : ¬ (¬ q → ¬ p) :=
by
sorry

end contrapositive_false_of_implication_false_l461_461760


namespace binomial_prob_l461_461736

open_locale classical

noncomputable def p_from_binom (n : ℕ) (p : ℚ) (ξ : ℕ → ℚ) : ℚ :=
if h : ξ ∼ Binomial(n, p) ∧ (E ξ = 300) ∧ (Var ξ = 200) then p else 0

theorem binomial_prob : ∀ (n : ℕ) (p : ℚ) (ξ : ℕ → ℚ),
  (ξ ∼ Binomial(n, p) ∧ (E ξ = 300) ∧ (Var ξ = 200)) → p = 1/3 :=
begin
  intros n p ξ h,
  sorry
end

end binomial_prob_l461_461736


namespace solve_for_x_l461_461174

theorem solve_for_x (y : ℝ) : 
  ∃ x : ℝ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 ∧ 
           x = -21 / 38 :=
by
  sorry

end solve_for_x_l461_461174


namespace domain_f_odd_function_a_eq_one_f_one_eq_three_a_eq_one_l461_461365

section

variable (a : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := 2 / (2 ^ x - 1) + a

-- Prove the domain of f is {x | x ≠ 0}
theorem domain_f : (f x) ∈ {y | x ≠ 0} := sorry

-- Prove that if f is an odd function then a = 1
theorem odd_function_a_eq_one (h : ∀ x, f (-x) = -f x) : a = 1 := sorry

-- Prove that if f(1) = 3 then a = 1
theorem f_one_eq_three_a_eq_one (h : f 1 = 3) : a = 1 := sorry

end

end domain_f_odd_function_a_eq_one_f_one_eq_three_a_eq_one_l461_461365


namespace labeling_possible_labeling_impossible_l461_461851

theorem labeling_possible (c : ℝ) (h : 0 < c ∧ c < sqrt 2) :
  ∃ (labeling : ℤ × ℤ → ℕ), 
    (finite (range labeling)) ∧ 
    (∀ i, ∀ p1 p2 : ℤ × ℤ, p1 ≠ p2 ∧ labeling p1 = i ∧ labeling p2 = i → dist p1 p2 ≥ c^i) :=
sorry

theorem labeling_impossible (c : ℝ) (h : c ≥ sqrt 2) :
  ¬ ∃ (labeling : ℤ × ℤ → ℕ), 
    (finite (range labeling)) ∧ 
    (∀ i, ∀ p1 p2 : ℤ × ℤ, p1 ≠ p2 ∧ labeling p1 = i ∧ labeling p2 = i → dist p1 p2 ≥ c^i) :=
sorry

end labeling_possible_labeling_impossible_l461_461851


namespace sum_of_digits_of_max_n_l461_461854

def is_single_digit_prime (p : ℕ) : Prop := p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7

def is_prime (p : ℕ) : Prop := Nat.Prime p

def single_digit_primes := {p : ℕ | is_single_digit_prime p}

noncomputable def max_n : ℕ :=
  let pairs := (single_digit_primes ×ˢ single_digit_primes).toFinset.filter (λ (ab : ℕ × ℕ), ab.fst ≠ ab.snd)
  let valid_products := pairs.filterMap (λ (ab : ℕ × ℕ), if is_prime (ab.fst + ab.snd)
                                        then some (ab.fst * ab.snd * (ab.fst + ab.snd))
                                        else none)
  valid_products.max' sorry

theorem sum_of_digits_of_max_n : (max_n.digits.sum = 7) :=
  sorry

end sum_of_digits_of_max_n_l461_461854


namespace power_zero_l461_461532

theorem power_zero (x : ℝ) (hx : x ≠ 0) : x^0 = 1 :=
by sorry

example : (2023 : ℝ)^0 = 1 :=
by apply power_zero
    linarith

end power_zero_l461_461532


namespace number_of_combinations_l461_461402

-- Define the finite set of test scores
def Scores := {70, 85, 88, 90, 98, 100}

-- Define the score function f(n) for n in {1, 2, 3, 4}
def f : Fin 4 → ℕ := sorry

-- Define the inequality condition
axiom h : ∀ (n : Fin 4), f 0 < f 1 ∧ f 1 ≤ f 2 ∧ f 2 < f 3

-- Prove that the number of valid combinations is 35
theorem number_of_combinations : (finset.univ.filter (λ v : Fin 4 → ℕ, v 0 < v 1 ∧ v 1 ≤ v 2 ∧ v 2 < v 3 ∧ ∀ (n : Fin 4), v n ∈ Scores)).card = 35 :=
by
  sorry

end number_of_combinations_l461_461402


namespace find_value_l461_461803

-- Conditions
variable {a b : ℝ}
hypothesis h₁ : (a + 2 * b) / a = 4

-- Theorem to prove
theorem find_value (h₁ : (a + 2 * b) / a = 4) : a / (b - a) = 2 := 
sorry

end find_value_l461_461803


namespace annual_interest_rate_l461_461551

-- Definitions for given conditions
def P : ℝ := 2000
def A : ℝ := 2800
def t : ℝ := 7
def compounded_annually : ℝ := 1

-- Prove the annual interest rate r
theorem annual_interest_rate (r : ℝ) 
  (h : A = P * (1 + r / compounded_annually) ^ (compounded_annually * t)) : 
  r ≈ 0.048113 := by
  sorry

end annual_interest_rate_l461_461551


namespace set_of_points_l461_461868

theorem set_of_points (x y : ℝ) (h : x^2 * y - y ≥ 0) :
  (y ≥ 0 ∧ |x| ≥ 1) ∨ (y ≤ 0 ∧ |x| ≤ 1) :=
sorry

end set_of_points_l461_461868


namespace rosie_can_make_nine_pies_l461_461896

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end rosie_can_make_nine_pies_l461_461896


namespace count_coprime_to_15_l461_461259

def coprime_to_15 (a : ℕ) : Prop := Nat.gcd 15 a = 1

theorem count_coprime_to_15 : 
  (Finset.filter coprime_to_15 (Finset.range 15)).card = 8 := by
  sorry

end count_coprime_to_15_l461_461259


namespace minimize_black_edges_l461_461682

noncomputable def smallest_black_edges {cube : Type} [fintype cube] (edges : cube → cube → Prop) (is_adjacent : cube → cube → Prop) : ℕ :=
Inf {n | ∃ (black_edges : finset (cube × cube)), black_edges.card = n ∧ 
         (∀ face, ∃ edge ∈ black_edges, edge ∈ face) ∧ 
         (∀ edge ∈ black_edges, ∀ edge' ∈ black_edges, (edge ≠ edge' → ¬ is_adjacent edge edge'))}

theorem minimize_black_edges (cube : Type) [fintype cube] (edges : cube → cube → Prop) (is_adjacent : cube → cube → Prop) :
  smallest_black_edges edges is_adjacent = 4 :=
sorry

end minimize_black_edges_l461_461682


namespace boris_daughter_candy_eaten_l461_461234

theorem boris_daughter_candy_eaten (initial_candy : ℕ) (bowls : ℕ) (taken_per_bowl : ℕ) (candy_per_bowl_after_taken : ℕ) (daughter_ate : ℕ) :
  initial_candy = 100 →
  bowls = 4 →
  taken_per_bowl = 3 →
  candy_per_bowl_after_taken = 20 →
  daughter_ate = initial_candy - (candy_per_bowl_after_taken + taken_per_bowl) * bowls :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  rw ←sub_eq_zero at *,
  sorry,
end


end boris_daughter_candy_eaten_l461_461234


namespace relationship_between_f_values_l461_461328

-- Conditions
variables {f : ℝ → ℝ}
variable (h_diff : ∀ x, Differentiable ℝ (f x))
variable (h_eq : ∀ x, f x = x^2 + 2 * x * (f' 2))

-- Proof statement
theorem relationship_between_f_values (h_diff : ∀ x, Differentiable ℝ (f x))
  (h_eq : ∀ x, f x = x^2 + 2 * x * (f' 2)) : f (-1) > f 1 := 
sorry

end relationship_between_f_values_l461_461328


namespace arithmetic_sequence_l461_461827

theorem arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (h₀ : a 1 = 2) (h₁ : a 2 + a 3 = 13)
    (h₂ : ∀ n, a n = a 1 + (n - 1) * d) : a 5 = 14 :=
by
  sorry

end arithmetic_sequence_l461_461827


namespace imaginary_part_is_1_l461_461326

noncomputable def imaginary_part_of_z (z : ℂ) : Prop :=
  1 + (0 : ℂ).im = z * (1 - (0 : ℂ).im)

theorem imaginary_part_is_1 : ∃ z : ℂ, imaginary_part_of_z z ∧ z.im = 1 :=
begin
  use i,
  split,
  { dsimp [imaginary_part_of_z],
    ring },
  { simp }
end

end imaginary_part_is_1_l461_461326


namespace price_reduction_is_50_rubles_l461_461523

theorem price_reduction_is_50_rubles :
  let P_Feb : ℕ := 300
  let P_Mar : ℕ := 250
  P_Feb - P_Mar = 50 :=
by
  let P_Feb : ℕ := 300
  let P_Mar : ℕ := 250
  sorry

end price_reduction_is_50_rubles_l461_461523


namespace currency_conversion_l461_461807

variable (a : ℚ)

theorem currency_conversion
  (h1 : (0.5 / 100) * a = 75 / 100) -- 0.5% of 'a' = 75 paise
  (rate_usd : ℚ := 0.012)          -- Conversion rate (USD/INR)
  (rate_eur : ℚ := 0.010)          -- Conversion rate (EUR/INR)
  (rate_gbp : ℚ := 0.009)          -- Conversion rate (GBP/INR)
  (paise_to_rupees : ℚ := 1 / 100) -- 1 Rupee = 100 paise
  : (a * paise_to_rupees * rate_usd = 1.8) ∧
    (a * paise_to_rupees * rate_eur = 1.5) ∧
    (a * paise_to_rupees * rate_gbp = 1.35) :=
by
  sorry

end currency_conversion_l461_461807


namespace slightly_used_crayons_l461_461537

theorem slightly_used_crayons (total_crayons : ℕ) (percent_new : ℚ) (percent_broken : ℚ) 
  (h1 : total_crayons = 250) (h2 : percent_new = 40/100) (h3 : percent_broken = 1/5) : 
  (total_crayons - percent_new * total_crayons - percent_broken * total_crayons) = 100 :=
by
  -- sorry here to indicate the proof is omitted
  sorry

end slightly_used_crayons_l461_461537


namespace num_rel_prime_to_15_l461_461254

theorem num_rel_prime_to_15 : 
  {a : ℕ | a < 15 ∧ Int.gcd 15 a = 1}.card = 8 := by 
  sorry

end num_rel_prime_to_15_l461_461254


namespace cubed_multiplication_identity_l461_461634

theorem cubed_multiplication_identity : 3^3 * 6^3 = 5832 := by
  sorry

end cubed_multiplication_identity_l461_461634


namespace sqrt_three_irrational_among_l461_461224

theorem sqrt_three_irrational_among :
  (¬ ∃ a b : ℤ, b ≠ 0 ∧ sqrt 3 = a / b) ∧
  (∃ a b : ℤ, b ≠ 0 ∧ -1 = a / b) ∧
  (∃ a b : ℤ, b ≠ 0 ∧ 0 = a / b) ∧
  (∃ a b : ℤ, b ≠ 0 ∧ 1 / 2 = a / b) :=
by
  split
  · sorry -- Proof of irrationality of sqrt(3)
  split
  · use [-1, 1]
    split
    · exact ne_zero_of_pos one_pos
    · norm_num
  split
  · use [0, 1]
    split
    · exact one_ne_zero
    · norm_num
  · use [1, 2]
    split
    · exact two_ne_zero
    · norm_num

end sqrt_three_irrational_among_l461_461224


namespace find_k_value_l461_461624

theorem find_k_value {A B C D O : Type*}
  (h1 : is_inscribed_in_circle A B C O)
  (h2 : is_acute_isosceles_triangle A B C)
  (h3 : are_tangents_meeting_at_point B C D)
  (h4 : ∠ABC = ∠ACB)
  (h5 : ∠ABC = 2 * ∠D)
  (h6 : ∠BAC = k * π) : k = 3 / 7 :=
sorry

end find_k_value_l461_461624


namespace prime_product_divisors_l461_461808

theorem prime_product_divisors (n : ℕ) (N : ℕ) 
  (h1 : ∀ (p : ℕ), p ∣ N → p > 3 ∧ Nat.Prime p)
  (h2 : ∀ (p q : ℕ), p ∣ N → q ∣ N → p ≠ q → Nat.Coprime p q)
  (h3 : Nat.Prime N) :
  Nat.NumDivisors (2 * N + 1) ≥ 4 * n := by
  sorry

end prime_product_divisors_l461_461808


namespace part1_part2_l461_461343

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.log x + 1

-- Part (1): Prove c ≥ -1 given f(x) ≤ 2x + c
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
  sorry

-- Define g with a > 0
def g (x a : ℝ) : ℝ := (f x - f a) / (x - a)

-- Part (2): Prove g is monotonically decreasing on (0, a) and (a, +∞)
theorem part2 (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, x > 0 → x ≠ a → g x a < g a a) :=
  sorry

end part1_part2_l461_461343


namespace problem1_problem2_l461_461586

-- Define the conditions for Problem 1
variable (a b : Type*) [InnerProductSpace ℝ a]

-- Given |a| = 2 and |b| = 1 and the angle between a and b is 60 degrees
axiom norm_a : ∥a∥ = 2
axiom norm_b : ∥b∥ = 1
axiom angle_ab : ⟪a, b⟫ = ∥a∥ * ∥b∥ * Real.cos (Real.pi / 3)

-- Prove that |2a - b| = √13
theorem problem1 : ∥2 • a - b∥ = Real.sqrt 13 := sorry

-- Define the conditions for Problem 2
variable (θ : ℝ)

-- Given tan θ = 3
axiom tan_theta : Real.tan θ = 3

-- Prove the expression
theorem problem2 : (5 * (Real.sin θ)^3 + Real.cos θ) / (2 * (Real.cos θ)^3 + (Real.sin θ)^2 * Real.cos θ) = 13 := sorry

end problem1_problem2_l461_461586


namespace principal_amount_l461_461394

theorem principal_amount (P : ℝ) : 
  let R := 7 in
  let T := 8 in
  let SI := (P * R * T) / 100 in
  SI = P - 5600 → 
  P ≈ 12727.27 :=
by
  intros
  dsimp only at *
  sorry

end principal_amount_l461_461394


namespace wax_blocks_needed_l461_461598

noncomputable def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def prism_volume (l w h : ℝ) : ℝ := l * w * h
def blocks_needed (cylinder_vol prism_vol : ℝ) : ℤ := (cylinder_vol / prism_vol).ceil

theorem wax_blocks_needed :
  blocks_needed 
    (cylinder_volume 2.5 10) 
    (prism_volume 8 3 2) = 5 :=
by
  sorry

end wax_blocks_needed_l461_461598


namespace p_minus_q_l461_461577

theorem p_minus_q (p q : ℚ) (h1 : 3 / p = 6) (h2 : 3 / q = 18) : p - q = 1 / 3 := by
  sorry

end p_minus_q_l461_461577


namespace part1_c_range_part2_monotonicity_l461_461339

noncomputable def f (x : ℝ) := 2 * Real.log x + 1

theorem part1_c_range (c : ℝ) (x : ℝ) (h : a > 0) : f x ≤ 2 * x + c → c ≥ -1 :=
sorry

noncomputable def g (x a : ℝ) := (f x - f a) / (x - a)

theorem part2_monotonicity (a : ℝ) (h : a > 0) : monotone_decreasing_on g (0, a) ∧ monotone_decreasing_on g (a, +∞) :=
sorry

end part1_c_range_part2_monotonicity_l461_461339


namespace position_of_B_total_fuel_consumption_l461_461109

-- Define the given distances
def distances : List Int := [18, -9, 7, -13, -6, 13, -6, -8]

-- Define the fuel consumption rate
def fuel_consumption_rate : Float := 0.1

-- Define the total distance traveled
def total_distance_traveled : Int := distances.sum

-- Define the total fuel consumed
def total_fuel_consumed : Float := fuel_consumption_rate * distances.map Int.toFloat |>.sum

-- Prove that the final position B is 4 kilometers south of A
theorem position_of_B : total_distance_traveled = -4 := by sorry

-- Prove that the total fuel consumption during the patrol is 8 liters
theorem total_fuel_consumption : total_fuel_consumed = 8 := by sorry

end position_of_B_total_fuel_consumption_l461_461109


namespace sin_value_length_of_AC_l461_461739

open Real

-- Define the conditions and questions
theorem sin_value (A B C S : ℝ) (h₁ : 0 < A < π) (h₂ : 3 * (sin B * sin C * cos A) = 2 * S) :
  sin A = (3 * sqrt 10) / 10 := sorry

theorem length_of_AC (A B C S : ℝ) (h₁ : C = π / 4) (h₂ : sin A = (3 * sqrt 10) / 10) 
  (h₃ : cos A = sqrt 10 / 10) (h₄ : sin B = (2 * sqrt 5) / 5) (h₅ : 3 * (sin B * cos (A + C)) = 16 * S) :
  AC = 8 := sorry

end sin_value_length_of_AC_l461_461739


namespace coefficient_x6_expansion_l461_461962

theorem coefficient_x6_expansion : 
  (∀ x : ℝ, coefficient (expand (1 - 3 * x ^ 3) 4) x 6 = 54) := 
sorry

end coefficient_x6_expansion_l461_461962


namespace part_a_part_b_part_c_l461_461583

-- Defining the problem's conditions
def athlete (n : ℕ) := Σ i, fin (2^n)
def event := fin (2^n)

structure competition (n : ℕ) :=
  (strength : athlete n → event → ℕ)

def advance (n : ℕ) (comp : competition n) (A : finset (athlete n)) (e : event) : finset (athlete n) :=
A.order_by (λ x, comp.strength x e).take (A.card / 2)

-- Defining potential winner
def is_potential_winner (n : ℕ) (comp : competition n) (a : athlete n) :=
∃ (e : fin n → event), last (iterate (advance n comp) (finset.univ) e) = a

-- Statements to prove
theorem part_a (n : ℕ) (comp : competition n) :
  2 ≤ n → ∃ S, S.card ≥ 2^(n-1) ∧ (∀ s ∈ S, is_potential_winner n comp s) :=
sorry

theorem part_b (n : ℕ) (comp : competition n) :
  2 ≤ n → (finset.univ.filter (is_potential_winner n comp)).card ≤ 2^n - n :=
sorry

theorem part_c (n : ℕ) (comp : competition n) :
  2 ≤ n → ∃ S, S.card = 2^n - n ∧ (∀ s ∈ S, is_potential_winner n comp s) :=
sorry

end part_a_part_b_part_c_l461_461583


namespace fraction_product_simplification_l461_461638

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by
  sorry

end fraction_product_simplification_l461_461638


namespace find_y_intercept_l461_461528

noncomputable def y_intercept (slope : ℚ) (x_intercept : ℚ × ℚ) : ℚ :=
  let (x₀, y₀) := x_intercept
  -slope * x₀ + y₀

theorem find_y_intercept : y_intercept (-3 / 2) (4, 0) = 6 :=
by
  dsimp [y_intercept]
  norm_num
  sorry

end find_y_intercept_l461_461528


namespace part1_part2_l461_461342

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.log x + 1

-- Part (1): Prove c ≥ -1 given f(x) ≤ 2x + c
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
  sorry

-- Define g with a > 0
def g (x a : ℝ) : ℝ := (f x - f a) / (x - a)

-- Part (2): Prove g is monotonically decreasing on (0, a) and (a, +∞)
theorem part2 (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, x > 0 → x ≠ a → g x a < g a a) :=
  sorry

end part1_part2_l461_461342


namespace curve_C_parametric_eq_common_points_count_l461_461588

-- Definition of the circle's equation and transformation
def circle_eq (x y : ℝ) := x^2 + y^2 = 1
def transform (x y : ℝ) := (x, 2*y)
def curve_C_eq (x y : ℝ) := x^2 + y^2 / 4 = 1

-- Theorem to prove the transformation leads to the equation of curve C
theorem curve_C_parametric_eq : 
  ∀ (x_1 y_1 : ℝ), circle_eq x_1 y_1 → (transform x_1 y_1 = (x, y)) → curve_C_eq x y := sorry

-- Definition of the line and the circle in Cartesian coordinates
def line_eq (x y : ℝ) := x + y = 2
def circle_cart_eq (x y : ℝ) := x^2 + y^2 = 4

-- Theorem to prove the number of common points
theorem common_points_count : 
  ∃! (A B : ℝ × ℝ), (line_eq A.1 A.2 ∧ circle_cart_eq A.1 A.2) ∧ (line_eq B.1 B.2 ∧ circle_cart_eq B.1 B.2) ∧ A ≠ B := sorry

end curve_C_parametric_eq_common_points_count_l461_461588


namespace areas_equal_l461_461820

-- Definitions of the points and shapes involved
variables {A B C D K M P : Type} [geometry A B C D] [geometry K C A B] [geometry M C A D] [geometry P K B D]

-- Define the quadrilateral with the specified angles and intersection points
noncomputable def quadrilateral (A B C D : Type) := sorry
def point_of_intersection (l1 l2 : Line) : Point := sorry
def line_parallel (l : Line) (p : Point) : Line := sorry
def area (quad : Quadrilateral) : ℝ := sorry

-- Conditions based on the problem statement
def conditions (A B C D K M P : Type) [quad : quadrilateral A B C D] : Prop :=
  (point_of_intersection (line_of_points A D) (line_parallel (line_of_points A B) C) = K) ∧
  (point_of_intersection (line_of_points A B) (line_parallel (line_of_points A D) C) = M) ∧
  (point_of_intersection (line_of_points B K) (line_of_points M D) = P)

-- Theorem stating the equality of areas of quadrilaterals AMPK and BCDP
theorem areas_equal (A B C D K M P : Type)
  [quad : quadrilateral A B C D]
  (h : conditions A B C D K M P) : 
  area (Quadrilateral.mk A M P K) = area (Quadrilateral.mk B C D P) :=
sorry

end areas_equal_l461_461820


namespace countInvalidNs_eq_30_l461_461449

def properDivisorsProduct (n : ℕ) : ℕ :=
  (List.foldr (*) 1 (List.filter (λ d, d > 0 ∧ d < n ∧ n % d = 0) (List.range (n + 1))))

noncomputable def countInvalidNs : ℕ :=
  List.length (List.filter (λ n, ¬ (n ∣ properDivisorsProduct n)) (List.range' 2 (100 - 2 + 1)))

theorem countInvalidNs_eq_30 : countInvalidNs = 30 := 
  sorry

end countInvalidNs_eq_30_l461_461449


namespace floor_e_equals_2_l461_461691

theorem floor_e_equals_2 : Int.floor Real.exp = 2 := 
sorry

end floor_e_equals_2_l461_461691


namespace smaller_sphere_radius_l461_461640

theorem smaller_sphere_radius (R x : ℝ) (h1 : (4/3) * Real.pi * R^3 = (4/3) * Real.pi * x^3 + (4/3) * Real.pi * (2 * x)^3) 
  (h2 : ∀ r₁ r₂ : ℝ, r₁ / r₂ = 1 / 2 → r₁ = x ∧ r₂ = 2 * x) : x = R / 3 :=
by 
  sorry

end smaller_sphere_radius_l461_461640


namespace painted_unit_cubes_count_l461_461251

def side_length_of_larger_cube_with_painted_faces (N : ℕ) : Prop :=
  ∃ n : ℕ, (n^3 = 24) ∧ (N = (n + 2)^3 - n^3)

theorem painted_unit_cubes_count 
  (N : ℕ)(H27 : N = 125)
  (H24 : ∃ n : ℕ, n^3 = 24) : side_length_of_larger_cube_with_painted_faces 101 :=
begin
  existsi 3,
  split,
  {
    -- proof that n^3 = 24 for n = 3
    have : (3 : ℕ)^3 = 27, { norm_num },
    ring,
  },
  {
    -- proof that painted faces with n=3 and hence N=125 gives 101 cubes.
    have total_small_cubes := (5 : ℕ)^3,
    norm_num at *,
    sorry, -- proof detail of side length larger proof
  }
end


end painted_unit_cubes_count_l461_461251


namespace six_points_in_square_l461_461628

noncomputable def smallest_possible_b (points : Fin 6 → ℝ × ℝ) : ℝ :=
  let distances := {d | ∃ (i j : Fin 6), i ≠ j ∧ d = dist (points i) (points j)};
  Inf distances

theorem six_points_in_square
  (side : ℝ)
  (points : Fin 6 → ℝ × ℝ)
  (h_square : ∀ i, 0 ≤ (points i).fst ∧ (points i).fst ≤ side ∧ 0 ≤ (points i).snd ∧ (points i).snd ≤ side)
  (h_side : side = 2) :
  smallest_possible_b points ≤ Real.sqrt 2 :=
sorry

end six_points_in_square_l461_461628


namespace trigonometric_identity_example_l461_461666

theorem trigonometric_identity_example :
  ∃ x y : ℝ, x = real.cos (20 * real.pi / 180) ∧ y = real.cos (10 * real.pi / 180) ∧
  (x * y - real.sin (20 * real.pi / 180) * real.sin (10 * real.pi / 180)) = real.cos (30 * real.pi / 180) := 
by
  have h : real.cos (30 * real.pi / 180) = sqrt 3 / 2 := sorry
  use real.cos (20 * real.pi / 180), real.cos (10 * real.pi / 180)
  repeat { split }
  sorry

end trigonometric_identity_example_l461_461666


namespace inclination_angle_of_line_l461_461512

theorem inclination_angle_of_line :
  ∀ (x y : ℝ), (√3 * x + y = 2) → (∃ α : ℝ, α = 120 ∧ tan α = -√3) := by
  sorry

end inclination_angle_of_line_l461_461512


namespace find_f_l461_461727

-- Define the function f such that f(x + 2) = x^2 - 4x
def f : ℝ → ℝ := sorry

-- Theorem statement: If the function f satisfies the condition f(x + 2) = x^2 - 4x for all x in ℝ,
-- then f(x) = x^2 - 8x + 12.
theorem find_f (h : ∀ x : ℝ, f(x + 2) = x^2 - 4x) : ∀ x : ℝ, f(x) = x^2 - 8x + 12 :=
by {
  sorry
}

end find_f_l461_461727


namespace tax_rate_first_20000_l461_461440

theorem tax_rate_first_20000  (income deductions total_tax : ℝ) (tax_rate_first_20000_rate : ℝ) :
  income = 100000 → deductions = 30000 → total_tax = 12000 → 
  (∃ r, 
    (let taxable_income := income - deductions in
     let tax_on_remaining := 0.2 * (taxable_income - 20000) in
     let tax_on_first_20000 := total_tax - tax_on_remaining in
     let tax_rate_first_20000 := (tax_on_first_20000 / 20000) * 100 in
     tax_rate_first_20000 = r)) → tax_rate_first_20000_rate = 10 := 
by
  intro h_income h_deductions h_total_tax h_exists_r
  rcases h_exists_r with ⟨r, hr⟩
  simp only [h_income, h_deductions, h_total_tax] at hr
  sorry

end tax_rate_first_20000_l461_461440


namespace cell_phones_in_Delaware_l461_461522

theorem cell_phones_in_Delaware (population : ℕ) (phones_per_1000_people : ℕ)
  (h_population : population = 974000)
  (h_phones_per_1000 : phones_per_1000_people = 673) :
  ∃ cell_phones : ℕ, cell_phones = population / 1000 * phones_per_1000_people ∧ cell_phones = 655502 :=
by {
  use 974 * 673,
  split,
  { rw [h_population, h_phones_per_1000], norm_num },
  { norm_num }
}

end cell_phones_in_Delaware_l461_461522


namespace integer_pairs_satisfying_equation_l461_461164

theorem integer_pairs_satisfying_equation :
  {p : ℤ × ℤ | (p.1)^3 + (p.2)^3 - 3*(p.1)^2 + 6*(p.2)^2 + 3*(p.1) + 12*(p.2) + 6 = 0}
  = {(1, -1), (2, -2)} := 
sorry

end integer_pairs_satisfying_equation_l461_461164


namespace odd_function_solution_l461_461927

variable {f : ℝ → ℝ}

theorem odd_function_solution
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 3 ≤ x → x < y → y ≤ 7 → f x < f y)
  (h_max : ∃ x ∈ Icc (3 : ℝ) 6, f x = 8)
  (h_min : ∃ x ∈ Icc (3 : ℝ) 6, f x = -1) :
  2 * f (-6) + f (-3) = -15 :=
by
  sorry

end odd_function_solution_l461_461927


namespace range_of_c_monotonicity_of_g_l461_461351

noncomputable def f (x: ℝ) : ℝ := 2 * Real.log x + 1

theorem range_of_c (c: ℝ) : (∀ x > 0, f x ≤ 2 * x + c) → c ≥ -1 := by
  sorry

noncomputable def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

theorem monotonicity_of_g (a: ℝ) (ha: a > 0) : 
  (∀ x > 0, x ≠ a → ((x < a → g x a < g a a) ∧ (x > a → g x a < g a a))) := by
  sorry

end range_of_c_monotonicity_of_g_l461_461351


namespace solve_for_x_l461_461931

theorem solve_for_x (x : ℝ) (h : (81 ^ (x - 2)) / (9 ^ (x - 2)) = 27 ^ (3 * x + 2)) : x = -10 / 7 :=
sorry

end solve_for_x_l461_461931


namespace floor_e_equals_2_l461_461694

theorem floor_e_equals_2 : Int.floor Real.exp = 2 := 
sorry

end floor_e_equals_2_l461_461694


namespace blue_paint_needed_l461_461085

/-- 
If the ratio of blue paint to green paint is \(4:1\), and Sarah wants to make 40 cans of the mixture,
prove that the number of cans of blue paint needed is 32.
-/
theorem blue_paint_needed (r: ℕ) (total_cans: ℕ) (h_ratio: r = 4) (h_total: total_cans = 40) : 
  ∃ b: ℕ, b = 4 / 5 * total_cans ∧ b = 32 :=
by
  sorry

end blue_paint_needed_l461_461085


namespace solve_system_of_equations_l461_461492

theorem solve_system_of_equations :
  let roots_eq1 := { x : ℝ | 5 * x^2 - 9 = 16 },
      roots_eq2 := { x : ℝ | |3 * x - 2| = 5 },
      roots_eq3 := { x : ℝ | x^3 - 4 * x = 0 } in
  ¬ (∀ x ∈ roots_eq1, (0 : ℝ) < x ∧ x ≤ 1 ∨
      ∀ x ∈ roots_eq1, x ∈ ℤ ∨
      ∀ x ∈ roots_eq1, x > 3 ∨
      ∃ x ∈ roots_eq1, x ≥ 0) ∧
  ¬ (∀ x ∈ roots_eq2, (0 : ℝ) < x ∧ x ≤ 1 ∨
      ∀ x ∈ roots_eq2, x ∈ ℤ ∨
      ∀ x ∈ roots_eq2, x > 3 ∨
      ∃ x ∈ roots_eq2, x ≥ 0) ∧
  ¬ (∀ x ∈ roots_eq3, (0 : ℝ) < x ∧ x ≤ 1 ∨
      ∀ x ∈ roots_eq3, x ∈ ℤ ∨
      ∀ x ∈ roots_eq3, x > 3 ∨
      ∃ x ∈ roots_eq3, x ≥ 0) :=
by
  -- Proof omitted
  sorry

end solve_system_of_equations_l461_461492


namespace c_should_pay_l461_461986

-- Define the grazing capacity equivalences
def horse_eq_oxen : ℝ := 2.0
def sheep_eq_oxen : ℝ := 0.5

-- Define the grazing capacities for each person in oxen-months
def a_grazing_oxen_months : ℝ := 10 * 7 + 4 * horse_eq_oxen * 3
def b_grazing_oxen_months : ℝ := 12 * 5
def c_grazing_oxen_months : ℝ := 15 * 3
def d_grazing_oxen_months : ℝ := 18 * 6 + 6 * sheep_eq_oxen * 8
def e_grazing_oxen_months : ℝ := 20 * 4
def f_grazing_oxen_months : ℝ := 5 * horse_eq_oxen * 2 + 10 * sheep_eq_oxen * 4

-- Define the total grazing capacity
def total_grazing_oxen_months : ℝ := a_grazing_oxen_months + b_grazing_oxen_months + c_grazing_oxen_months + d_grazing_oxen_months + e_grazing_oxen_months + f_grazing_oxen_months

-- Define the rent of the pasture
def total_rent : ℝ := 1200

-- Define the amount c should pay
def amount_c_should_pay : ℝ := (c_grazing_oxen_months / total_grazing_oxen_months) * total_rent

-- Prove that the amount c should pay is approximately Rs. 119.73
theorem c_should_pay (h_approx : | amount_c_should_pay - 119.73 | < 0.01) : true := by 
  -- Skip the proof for now
  sorry

end c_should_pay_l461_461986


namespace magic_square_div_by_3_l461_461603

def is_magic_square (M : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  let s := M 0 0 + M 0 1 + M 0 2 in
  (M 1 0 + M 1 1 + M 1 2 = s) ∧
  (M 2 0 + M 2 1 + M 2 2 = s) ∧
  (M 0 0 + M 1 0 + M 2 0 = s) ∧
  (M 0 1 + M1 1 + M 2 1 = s) ∧
  (M 0 2 + M 1 2 + M 2 2 = s) ∧
  (M 0 0 + M 1 1 + M 2 2 = s) ∧
  (M 0 2 + M 1 1 + M 2 0 = s)

theorem magic_square_div_by_3 (M : Matrix (Fin 3) (Fin 3) ℕ) (h : is_magic_square M) : 
  let s := M 0 0 + M 0 1 + M 0 2 in
  s % 3 = 0 :=
sorry

end magic_square_div_by_3_l461_461603


namespace dot_product_range_trajectory_of_P_l461_461861

-- Definition of the hyperbola and its foci
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 3 = 1

def focus₁ : ℝ × ℝ := (-real.sqrt 5, 0)
def focus₂ : ℝ × ℝ := (real.sqrt 5, 0)

-- Problem statement 1: Range of dot product for M on the right branch
theorem dot_product_range (x y : ℝ) (hx : x ≥ real.sqrt 2) (hM : hyperbola x y) :
  let π_MF₁ := x^2 + real.sqrt 5 * x - 3 + y^2
  π_MF₁ ∈ set.Ici (2 + real.sqrt 10) :=
sorry

-- Problem statement 2: Equation of trajectory of point P
theorem trajectory_of_P (a : ℝ) (P : ℝ × ℝ → Prop)
  (hsum_dist : ∀ (p : ℝ × ℝ), P p → dist p focus₁ + dist p focus₂ = 2 * a)
  (hmin_cos : ∀ (p : ℝ × ℝ), P p → cos_angle focus₁ p focus₂ > -1 / 9) :
  ∑ p : ℝ × ℝ, P p ↔ (p.1^2 / 9 + p.2^2 / 4 = 1) :=
sorry

end dot_product_range_trajectory_of_P_l461_461861


namespace zero_not_in_empty_set_l461_461710

theorem zero_not_in_empty_set : 0 ∉ ∅ :=
by
  sorry

end zero_not_in_empty_set_l461_461710


namespace sum_of_2n_terms_of_sequence_b_l461_461375

open Nat

/--
Given the sequence {aₙ} with the sum of its first n terms Sₙ = (n^2 + n)/2,
where n ∈ ℕ*.
(I) Prove that the general formula for the sequence {aₙ} is aₙ = n.
(II) Let bₙ = 2^aₙ + (-1)^n * aₙ.
Prove that the sum of the first 2n terms of the sequence {bₙ} is T_{2n} = 2^{2n+1} + n - 2.
-/

theorem sum_of_2n_terms_of_sequence_b (n : ℕ) (hn : 0 < n) :
  let S (n : ℕ) := (n^2 + n) / 2,
      a_1 := 1,
      a (n : ℕ) := if n = 1 then 1 else n,
      b (n : ℕ) := 2^(a n) + (-1)^n * n,
      T (2n : ℕ) := ∑ i in Finset.range (2*n), b (i+1)
  in T (2*n) = 2^(2*n+1) + n - 2 := 
  sorry

end sum_of_2n_terms_of_sequence_b_l461_461375


namespace total_pies_l461_461897

def apple_Pies (totalApples : ℕ) (applesPerPie : ℕ) (piesPerBatch : ℕ) : ℕ :=
  (totalApples / applesPerPie) * piesPerBatch

def pear_Pies (totalPears : ℕ) (pearsPerPie : ℕ) (piesPerBatch : ℕ) : ℕ :=
  (totalPears / pearsPerPie) * piesPerBatch

theorem total_pies :
  let apples : ℕ := 27
  let pears : ℕ := 30
  let applesPerPie : ℕ := 9
  let pearsPerPie : ℕ := 15
  let applePiesPerBatch : ℕ := 2
  let pearPiesPerBatch : ℕ := 3
  apple_Pies apples applesPerPie applePiesPerBatch + pear_Pies pears pearsPerPie pearPiesPerBatch = 12 :=
by
  sorry

end total_pies_l461_461897


namespace geometric_sequence_first_term_l461_461120

open Nat

theorem geometric_sequence_first_term : 
  ∃ (a r : ℝ), (a * r^3 = (6 : ℝ)!) ∧ (a * r^6 = (7 : ℝ)!) ∧ a = 720 / 7 :=
by
  sorry

end geometric_sequence_first_term_l461_461120


namespace floor_e_equals_two_l461_461689

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end floor_e_equals_two_l461_461689


namespace shorter_side_length_l461_461811

theorem shorter_side_length (L W : ℝ) (h₁ : L * W = 104) (h₂ : 2 * L + 2 * W = 42) : W = 8 :=
by
  have h₃ : L + W = 21 := by linarith
  have h₄ : W = 21 - L := by linarith
  have quad_eq : L^2 - 21*L + 104 = 0 := by linarith
  has_solution_L_one : L = 13 := sorry
  has_solution_L_two : L = 8 := sorry
  use W
  sorry

end shorter_side_length_l461_461811


namespace length_of_first_train_l461_461951

theorem length_of_first_train:
  ∀ (speed1 speed2 : ℝ) (length2 time: ℝ),
  speed1 = 80 ∧ speed2 = 65 ∧ length2 = 165 ∧ time = 8.093145651796132 →
  let total_speed := (speed1 + speed2) * 1000 / 3600 in
  let distance := total_speed * time in
  distance - length2 = 161.0864197530864 :=
by
  intros speed1 speed2 length2 time h
  obtain ⟨h_speed1, h_speed2, h_length2, h_time⟩ := h
  rw [h_speed1, h_speed2, h_length2, h_time]
  let total_speed := (80 + 65) * 1000 / 3600
  let distance := total_speed * 8.093145651796132
  have : distance = 326.0864197530864 := by sorry
  show 326.0864197530864 - 165 = 161.0864197530864
  sorry

end length_of_first_train_l461_461951


namespace cos_alpha_value_l461_461331

-- Define the point through which the terminal side of angle α passes
def point : ℝ × ℝ := (-4, 3)

-- Define the radius (distance from origin) term
def r : ℝ := real.sqrt (point.1^2 + point.2^2)

-- Define cos(α) in terms of x and r
def cos_alpha : ℝ := point.1 / r

-- State the theorem
theorem cos_alpha_value (h : point = (-4, 3)) : cos_alpha = -4 / 5 := 
by
  simp [cos_alpha, r, point]
  sorry

end cos_alpha_value_l461_461331


namespace cost_of_fencing_each_side_l461_461388

theorem cost_of_fencing_each_side (total_cost : ℕ) (n_sides : ℕ) (cost_per_side : ℕ) :
  total_cost = 172 ∧ n_sides = 4 → cost_per_side = total_cost / n_sides :=
begin
  intro h,
  cases h with htotal hn_sides,
  rw htotal,
  rw hn_sides,
  exact nat.div_eq_of_eq_mul (by norm_num : 172 = 4 * 43)
end

end cost_of_fencing_each_side_l461_461388


namespace perfect_square_k_l461_461176

theorem perfect_square_k (P X : Set ℕ) (f : ℕ → ℕ) (k : ℕ) 
  (hP : P = {p ∈ ℕ | ∃ i < 2010, Nat.prime p ∧ p = (Nat.primes.toFinset i).val})
  (hX : X = {n ∈ ℕ | ∀ p, Nat.prime p → ∃ k ∈ P, p^k ∣ n})
  (hf : ∀ m n ∈ X, f (m * f n) = f m * n^k) : ∃ l : ℕ, k = l^2 :=
sorry

end perfect_square_k_l461_461176


namespace p_q_r_s_sum_l461_461447

def Q (x : ℝ) := x^2 - 4 * x - 4

noncomputable def is_probability_satisfied (x : ℝ) : Prop :=
  3 <= x ∧ x <= 10 ∧ ⌊√(Q(x))⌋ = √(Q(⌊x⌋))

theorem p_q_r_s_sum : 
  let p := 17
  let q := 21
  let r := 18
  let s := 7 in p + q + r + s = 121
  := 
by {
  let p : ℤ := 17
  let q : ℤ := 21
  let r : ℤ := 18
  let s : ℤ := 7
  show p + q + r + s = 121
  sorry
}

end p_q_r_s_sum_l461_461447


namespace area_of_trapezoid_l461_461114

-- Definitions: sides of the triangle and the ratio of height division
def a := 28 -- side BC in cm
def b := 26 -- side AC in cm
def c := 30 -- side AB in cm
def height_ratio := (2 : ℕ, 3 : ℕ) -- ratio of height division

-- Statements for the problem conditions
theorem area_of_trapezoid (a b c : ℕ) (height_ratio : ℕ × ℕ) :
    a = 28 ∧ b = 26 ∧ c = 30 ∧ height_ratio = (2, 3) →
    let p := (a + b + c) / 2,
        S_abc := (p * (p - a) * (p - b) * (p - c)).sqrt,
        S_mnc_ratio := (height_ratio.1 / (height_ratio.1 + height_ratio.2))^2,
        S_mnc := S_mnc_ratio * S_abc,
        S_amnb := S_abc - S_mnc
    in S_amnb = 322.56 := 
sorry

end area_of_trapezoid_l461_461114


namespace PX_eq_CY_l461_461177

theorem PX_eq_CY 
  {A B C P Q X Y : Type} [IsoscelesTriangle A B C] 
  (hPQ_parallel : Parallel P Q B C)
  (hXY_parallel : Parallel X Y B C)
  (hP_on_AB : On P AB)
  (hQ_on_AC : On Q AC)
  (hX_angle_bisector : AngleBisector X B A C)
  (hY_angle_bisector : AngleBisector Y Q A P) 
  : Length P X = Length C Y := 
sorry

end PX_eq_CY_l461_461177


namespace SmartMart_science_kits_l461_461108

theorem SmartMart_science_kits (sc pz : ℕ) (h1 : pz = sc - 9) (h2 : pz = 36) : sc = 45 := by
  sorry

end SmartMart_science_kits_l461_461108


namespace simplify_G_l461_461051

def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

theorem simplify_G : ∀ x, F (2*x / (1+x^2)) = 2 * F x :=
by sorry

end simplify_G_l461_461051


namespace number_of_convex_numbers_l461_461216

-- Define the set S
def S : Finset ℕ := {1,2,3,4,5}

-- Define the convex condition
def is_convex (x y z : ℕ) : Prop :=
  x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ y > x ∧ y > z

-- The theorem to be proved
theorem number_of_convex_numbers : 
  (Finset.filter (λ (t : ℕ × ℕ × ℕ), is_convex t.1 t.2 t.3) (S.product (S.product S))).card = 20 := 
sorry

end number_of_convex_numbers_l461_461216


namespace percentage_scientists_born_in_june_l461_461123

theorem percentage_scientists_born_in_june :
  (18 / 200 * 100) = 9 :=
by sorry

end percentage_scientists_born_in_june_l461_461123


namespace total_groups_of_two_marbles_l461_461147

-- Define the given conditions as constants
constant red_marble : Type
constant green_marble : Type
constant blue_marble : Type
constant purple_marble : Type
constant yellow_marble : Type

constant identical_yellow_marbles : ℕ

-- Prove the total number of different groups of two marbles
theorem total_groups_of_two_marbles :
  (identical_yellow_marbles = 4) →
  let marbles := [red_marble, green_marble, blue_marble, purple_marble, yellow_marble, yellow_marble, yellow_marble, yellow_marble] in
  (count_distinct_groups_of_two marbles = 11) :=
begin
  intros h,
  let marbles := [red_marble, green_marble, blue_marble, purple_marble, yellow_marble, yellow_marble, yellow_marble, yellow_marble],
  sorry -- proof omitted
end

end total_groups_of_two_marbles_l461_461147


namespace complex_z_proof_l461_461060

def exists_complex_z_with_properties (A : Set ℕ) (n : ℕ) (h : n ≥ 2) : Prop :=
  ∃ (z : ℂ), abs z = 1 ∧ abs (∑ a in A, z ^ a) = Real.sqrt (n - 2)

theorem complex_z_proof (A : Set ℕ) (n : ℕ) (h : n ≥ 2) (hA : A.card = n) : 
  exists_complex_z_with_properties A n h := By sorry

end complex_z_proof_l461_461060


namespace fg_of_2_is_225_l461_461806

-- Conditions: Definitions of f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2*x^2 + 3*x + 1

-- Proof statement: Prove that f(g(2)) = 225
theorem fg_of_2_is_225 : f(g(2)) = 225 := by
  sorry -- Proof is not required, so we use sorry to complete the statement.

end fg_of_2_is_225_l461_461806


namespace Jack_hair_length_l461_461044

variable (Kate_hair: ℝ)
variable (Emily_hair: ℝ)
variable (Logan_hair: ℝ)
variable (Jack_hair: ℝ)

axiom Logan_hair_length : Logan_hair = 20
axiom Emily_hair_longer : Emily_hair = Logan_hair + 6
axiom Kate_hair_half : Kate_hair = Emily_hair / 2
axiom Jack_hair_times : Jack_hair = Kate_hair * 3

theorem Jack_hair_length : Jack_hair = 39 :=
by
  rw [Logan_hair_length, Emily_hair_longer, Kate_hair_half, Jack_hair_times]
  sorry

end Jack_hair_length_l461_461044


namespace hyperbola_specific_case_l461_461756

noncomputable def equation_of_hyperbola (a c : ℝ) : Prop :=
  let e := c / a in
  let b := Real.sqrt (c^2 - a^2) in
  e = 2 →
  a = 1 →
  (∀ x y, x^2 - (y^2 / b^2) = 1)

theorem hyperbola_specific_case :
  equation_of_hyperbola 1 2 := by
  intro h_e h_a
  have h_b : (b : ℝ) = Real.sqrt(3) := sorry
  exact sorry

end hyperbola_specific_case_l461_461756


namespace parabola_equation_l461_461121

theorem parabola_equation :
  ∃ a b c d e f : ℤ,
    c > 0 ∧ Int.gcd a b = 1 ∧ Int.gcd a c = 1 ∧ Int.gcd a d = 1 ∧ Int.gcd a e = 1 ∧ Int.gcd a f = 1 ∧
    Int.gcd b c = 1 ∧ Int.gcd b d = 1 ∧ Int.gcd b e = 1 ∧ Int.gcd b f = 1 ∧
    Int.gcd c d = 1 ∧ Int.gcd c e = 1 ∧ Int.gcd c f = 1 ∧
    Int.gcd d e = 1 ∧ Int.gcd d f = 1 ∧ Int.gcd e f = 1 ∧
    (a • (y^2) + b • (x ⋅ y) + c • (x^2) + d • y + e • x + f = 0) ∧
    (0 • (y^2) + 0 • (x ⋅ y) + 0 • (x^2) + 0 • y + 0 • x + 0 = 0) :=
by
  sorry

end parabola_equation_l461_461121


namespace cube_surface_area_l461_461131

theorem cube_surface_area (P : ℝ) (hP : P = 24) : 
  let edge_length := P / 4 in
  let surface_area := 6 * (edge_length ^ 2) in
  surface_area = 216 :=
by
  have edge_length_eq : edge_length = 6 :=
    by
      rw [hP, div_eq_mul_inv, mul_comm, ← mul_assoc, inv_mul_cancel (ne_of_gt (by norm_num : (4 : ℝ) > 0)), one_mul]
  have surface_area_eq : surface_area = 6 * (6 ^ 2) :=
    by
      rw [← edge_length_eq]
  calc
    surface_area = 6 * (6 ^ 2) : by rw [surface_area_eq]
    ... = 216     : by norm_num

end cube_surface_area_l461_461131


namespace max_distance_eq_line_l461_461416

open Real

-- Define the given conditions
variable (a b c x y : ℝ)
variable (M : ℝ × ℝ)

axiom midpoint_chord : ∃ x y : ℝ, M = (x, y) ∧ x^2 + y^2 = 16
axiom line_intercepted : a * x + b * y + c = 0
axiom condition : a + 2 * b - c = 0

-- Define the theorem
theorem max_distance_eq_line : 
  ∀ (M : ℝ × ℝ), M = (-1, -2) → 
  |sqrt ((-1:ℝ)^2 + (-2:ℝ)^2)| → 
  a * x + b * y + c = 0 → 
  x + 2 * y + 5 = 0 :=
by
  sorry

end max_distance_eq_line_l461_461416


namespace movement_of_hands_of_clock_involves_rotation_l461_461569

theorem movement_of_hands_of_clock_involves_rotation (A B C D : Prop) :
  (A ↔ (∃ p : ℝ, ∃ θ : ℝ, p ≠ θ)) → -- A condition: exists a fixed point and rotation around it
  (B ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- B condition: does not rotate around a fixed point
  (C ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- C condition: does not rotate around a fixed point
  (D ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- D condition: does not rotate around a fixed point
  A :=
by
  intros hA hB hC hD
  sorry

end movement_of_hands_of_clock_involves_rotation_l461_461569


namespace part1_part2_l461_461354

-- Definition of the function f.
def f (x: ℝ) : ℝ := 2 * Real.log x + 1

-- Definition of the function g.
def g (x a: ℝ) : ℝ := (f x - f a) / (x - a)

-- Part 1: Prove that c ≥ -1 given f(x) ≤ 2x + c.
theorem part1 (c : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) → c ≥ -1 :=
by
  -- Proof is omitted.
  sorry

-- Part 2: Prove that g(x) is monotonically decreasing on (0, a) and (a, +∞) given a > 0.
theorem part2 (a : ℝ) : a > 0 → (∀ x : ℝ, x > 0 → x ≠ a → 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo 0 a → x2 ∈ Ioo 0 a → x1 < x2 → g x2 a < g x1 a) ∧ 
  (∀ x1 x2 : ℝ, x1 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x2 ∈ Ioo a (Real.Inf.set (Set.Iio a)) → x1 < x2 → g x2 a < g x1 a)) :=
by
  -- Proof is omitted.
  sorry

end part1_part2_l461_461354


namespace partition_distance_l461_461481

theorem partition_distance (M : Set (ℝ → ℝ → Prop)) (M1 M2 M3 : Set ℝ) (a : ℝ) (h1 : M = {M1, M2, M3}) (h_partition : ∀ x, x ∈ M1 ∨ x ∈ M2 ∨ x ∈ M3) : 
  ∃ i ∈ {1, 2, 3}, ∀ a > 0, ∃ x y ∈ (if i = 1 then M1 else if i = 2 then M2 else M3), dist x y = a := 
by
  sorry

end partition_distance_l461_461481


namespace circle_symmetric_eq_l461_461290

theorem circle_symmetric_eq :
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 2 * y + 1 = 0) → (x - y + 3 = 0) → 
  (∃ (a b : ℝ), (a + 2)^2 + (b - 2)^2 = 1) :=
by
  intros x y hc hl
  sorry

end circle_symmetric_eq_l461_461290


namespace find_length_BE_l461_461829

-- We define the points and lengths based on the conditions
variables (A B C D M E : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M] [metric_space E]
variables (angle_BC_Angle : ∠ B = 60) (angle_CC_Angle : ∠ C = 60)
variables (length_BC : BC = 1)
variables (circle_diameter_CD_tangent_AB : M = tangent_point (circle (C,D) (diameter CD)) (AB))
variables (circle_intersect_BC : E = intersect_point (circle (C,D) (diameter CD)) (BC))

-- The main theorem to prove
theorem find_length_BE : length BE = 4 - 2 * sqrt 3 := by
  sorry

end find_length_BE_l461_461829


namespace neg_p_equiv_l461_461371

open Classical

theorem neg_p_equiv (x : ℝ) : (¬ ∃ x : ℝ, exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, exp x - x - 1 > 0) :=
by
  sorry

end neg_p_equiv_l461_461371


namespace solution_pairs_l461_461675

theorem solution_pairs (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) := 
by 
  sorry

end solution_pairs_l461_461675


namespace train_crossing_time_l461_461836

variable (train_length : ℕ) (train_speed_kmph : ℕ)

def speed_conversion (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600

def crossing_time (length : ℕ) (speed : ℕ) : ℕ := length / speed

theorem train_crossing_time
  (length_eq : train_length = 80) 
  (speed_kmph_eq : train_speed_kmph = 144) 
  (speed_eq : speed_conversion train_speed_kmph = 40) :
  crossing_time train_length (speed_conversion train_speed_kmph) = 2 := by
  rw [length_eq, speed_kmph_eq, speed_eq]
  simp
  sorry

end train_crossing_time_l461_461836


namespace candidate_A_valid_votes_l461_461410

theorem candidate_A_valid_votes :
  let total_votes := 1280000
  let invalid_vote_percentage := 0.25
  let valid_vote_percentage := 1 - invalid_vote_percentage
  let valid_votes := total_votes * valid_vote_percentage
  let a_vote_percentage := 0.6
  let a_votes := valid_votes * a_vote_percentage
  a_votes = 576000 :=
by
  let total_votes := 1280000
  let invalid_vote_percentage := 0.25
  let valid_vote_percentage := 1 - invalid_vote_percentage
  let valid_votes := total_votes * valid_vote_percentage
  let a_vote_percentage := 0.6
  let a_votes := valid_votes * a_vote_percentage
  have : a_votes = 576000 := by sorry
  exact this

end candidate_A_valid_votes_l461_461410


namespace amusement_park_trip_cost_l461_461472

def cost_per_trip (pass_cost : ℕ) (num_passes : ℕ) (oldest_trips : ℕ) (youngest_trips : ℕ) : ℕ :=
  let total_cost := num_passes * pass_cost
  let total_trips := oldest_trips + youngest_trips
  total_cost / total_trips

theorem amusement_park_trip_cost :
  ∀ (pass_cost num_passes oldest_trips youngest_trips : ℕ),
  pass_cost = 100 → num_passes = 2 → oldest_trips = 35 → youngest_trips = 15 →
  cost_per_trip pass_cost num_passes oldest_trips youngest_trips = 4 :=
by
  intros
  rw [H, H_1, H_2, H_3]
  sorry

end amusement_park_trip_cost_l461_461472


namespace sum_of_solutions_l461_461561

theorem sum_of_solutions :
  let solutions := {x ∈ Finset.Icc 1 30 | (17 * (5 * x - 3)) % 12 = 34 % 12}.sum id
  solutions = 39 :=
by
  sorry

end sum_of_solutions_l461_461561


namespace count_coprime_to_15_l461_461258

def coprime_to_15 (a : ℕ) : Prop := Nat.gcd 15 a = 1

theorem count_coprime_to_15 : 
  (Finset.filter coprime_to_15 (Finset.range 15)).card = 8 := by
  sorry

end count_coprime_to_15_l461_461258


namespace triangle_RS_length_l461_461030

theorem triangle_RS_length (PQ QR PS QS RS : ℝ)
  (h1 : PQ = 8) (h2 : QR = 8) (h3 : PS = 10) (h4 : QS = 5) :
  RS = 3.5 :=
by
  sorry

end triangle_RS_length_l461_461030


namespace trains_cross_time_l461_461952

noncomputable def time_to_cross_trains : ℝ :=
  200 / (89.992800575953935 * (1000 / 3600))

theorem trains_cross_time :
  abs (time_to_cross_trains - 8) < 1e-7 :=
by
  sorry

end trains_cross_time_l461_461952


namespace find_a1989_cos_x_l461_461305

def sequence_a (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then sin x
  else
    let a_n_minus_1 := (sequence_a (n - 1) x) in
    (-1)^(n / 2) * sqrt (1 - a_n_minus_1^2)

theorem find_a1989_cos_x (x : ℝ) : sequence_a 1989 x = cos x := sorry

end find_a1989_cos_x_l461_461305


namespace sum_of_lengths_not_equal_l461_461471

theorem sum_of_lengths_not_equal (n : ℕ) (hn : n = 2022) 
    (h_red_blue : ∀ i, i < n ↔ ((i < 1011) → red i) ∧ ((i ≥ 1011) → blue i)) 
    (dist_eq : ∀ i, (i < n - 1) → distance (i, i + 1) = distance (i + 1, i + 2)) :
  (∑ i j, (red i ∧ blue j) → length (i, j)) ≠ 
  (∑ i j, (blue i ∧ red j) → length (i, j)) :=
by
  sorry

end sum_of_lengths_not_equal_l461_461471


namespace susan_added_oranges_l461_461141

-- Conditions as definitions
def initial_oranges_in_box : ℝ := 55.0
def final_oranges_in_box : ℝ := 90.0

-- Define the quantity of oranges Susan put into the box
def susan_oranges := final_oranges_in_box - initial_oranges_in_box

-- Theorem statement to prove that the number of oranges Susan put into the box is 35.0
theorem susan_added_oranges : susan_oranges = 35.0 := by
  unfold susan_oranges
  sorry

end susan_added_oranges_l461_461141


namespace solution_pairs_l461_461674

theorem solution_pairs (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) := 
by 
  sorry

end solution_pairs_l461_461674


namespace correct_propositions_l461_461056

variables (m n : Line)
variables (α β γ : Plane)

-- Conditions:
def different_lines : Prop := m ≠ n
def different_planes : Prop := α ≠ β ∧ β ≠ γ ∧ γ ≠ α

-- Propositions:
def prop1 := m ⊆ β ∧ α ⊥ β → m ⊥ α
def prop2 := α ∥ β ∧ m ⊆ α → m ∥ β
def prop3 := n ⊥ α ∧ n ⊥ β ∧ m ⊥ α → m ⊥ β
def prop4 := α ⊥ γ ∧ β ⊥ γ → α ⊥ β

-- The proof problem:
theorem correct_propositions :
  different_lines m n ∧ different_planes α β γ →
  (prop2 α β m ∧ prop3 α β γ m n) :=
by
  sorry

end correct_propositions_l461_461056


namespace max_groups_l461_461158

theorem max_groups (cards : ℕ) (sum_group : ℕ) (c5 c2 c1 : ℕ) (cond1 : cards = 600) (cond2 : c5 = 200)
  (cond3 : c2 = 200) (cond4 : c1 = 200) (cond5 : sum_group = 9) :
  ∃ max_g : ℕ, max_g = 100 :=
by
  sorry

end max_groups_l461_461158


namespace equal_perpendiculars_l461_461432

theorem equal_perpendiculars
  (A B C D E F : Type)
  [linear_ordered_field A]
  [inner_product_space A B]
  (hBC : line B C)
  (hD_mid : midpoint D B C)
  (hE_foot : foot_perpendicular_of B D)
  (hF_foot : foot_perpendicular_of C D) :
  distance B E = distance C F :=
by
  sorry

end equal_perpendiculars_l461_461432


namespace zero_point_interval_l461_461713

open Real

noncomputable def f (x : ℝ) : ℝ :=
  log (x + 2) / log 2 - 3 / x

theorem zero_point_interval :
  (∃ c, 1 < c ∧ c < 2 ∧ f c = 0) :=
by
  have h1 : f 1 < 0 :=
    by
      simp [f]
      -- We use approximate log calculation here for demonstration:
      have : log 3 / log 2 - 3 < 0 := by sorry
      exact this
  have h2 : f 2 > 0 := 
    by
      simp [f]
      -- We use direct log calculation here:
      have : 2 - 3 / 2 > 0 := by linarith
      exact this
  have h_intermediate : ∃ c ∈ Ioo (1 : ℝ) 2, f c = 0 :=
    IntermediateValueTheorem.exists_zero h1 h2 _
  exact h_intermediate

end zero_point_interval_l461_461713


namespace required_moles_h2so4_l461_461788

-- Defining chemical equation conditions
def balanced_reaction (nacl h2so4 hcl nahso4 : ℕ) : Prop :=
  nacl = h2so4 ∧ hcl = nacl ∧ nahso4 = nacl

-- Theorem statement
theorem required_moles_h2so4 (nacl_needed moles_h2so4 : ℕ) (hcl_produced nahso4_produced : ℕ)
  (h : nacl_needed = 2 ∧ balanced_reaction nacl_needed moles_h2so4 hcl_produced nahso4_produced) :
  moles_h2so4 = 2 :=
  sorry

end required_moles_h2so4_l461_461788


namespace jack_birth_year_l461_461920

theorem jack_birth_year 
  (first_amc8_year : ℕ) 
  (amc8_annual : ℕ → ℕ → ℕ) 
  (jack_age_ninth_amc8 : ℕ) 
  (ninth_amc8_year : amc8_annual first_amc8_year 9 = 1998) 
  (jack_age_in_ninth_amc8 : jack_age_ninth_amc8 = 15)
  : (1998 - jack_age_ninth_amc8 = 1983) := by
  sorry

end jack_birth_year_l461_461920


namespace max_sum_of_cubes_tennis_tournament_l461_461990

theorem max_sum_of_cubes_tennis_tournament :
  ∃ (B : Fin 12 → ℕ), (∑ i, B i = 66) ∧ (∑ i, (B i)^3 = 4356) :=
by
  sorry

end max_sum_of_cubes_tennis_tournament_l461_461990


namespace smallest_positive_angle_l461_461395

def sin_of_2pi_over_3 : Real := (Real.sin (2 * Real.pi / 3))
def cos_of_2pi_over_3 : Real := (Real.cos (2 * Real.pi / 3))

def point_P : Real × Real := (sin_of_2pi_over_3, cos_of_2pi_over_3)

theorem smallest_positive_angle (α : Real) (h1 : sin_of_2pi_over_3 = √3 / 2)
  (h2 : cos_of_2pi_over_3 = -1 / 2) (h3 : point_P = (√3 / 2, -1 / 2)) :
  α = 11 * Real.pi / 6 :=
sorry

end smallest_positive_angle_l461_461395


namespace negation_p_negation_q_l461_461979

variables {a b c : ℝ}

def p : Prop :=
  ∃ x, f(x) = 0 ∧ ∀ y ≠ x, f(y) ≠ 0
  where f(x : ℝ) := a * x ^ 2 + b * x + c

def q : Prop :=
  (x = 3 ∨ x = 4) → (x ^ 2 - 7 * x + 12 = 0)
  where x : ℝ

theorem negation_p :
  ¬p ↔
  (∀ x, f(x) ≠ 0) ∨
  (∃ x y, x ≠ y ∧ f(x) = 0 ∧ f(y) = 0)
  where f(x : ℝ) := a * x ^ 2 + b * x + c :=
sorry

theorem negation_q :
  ¬q ↔
  (∃ x, (x = 3 ∨ x = 4) ∧ (x ^ 2 - 7 * x + 12 ≠ 0))
  where x : ℝ :=
sorry

end negation_p_negation_q_l461_461979


namespace coprime_integers_lt_15_l461_461277

theorem coprime_integers_lt_15 : ∃ (S : Finset ℕ), S.card = 8 ∧ (∀ a ∈ S, a < 15 ∧ Nat.gcd a 15 = 1) :=
by
  sorry

end coprime_integers_lt_15_l461_461277


namespace smallest_perfect_square_divisible_by_5_and_7_l461_461559

theorem smallest_perfect_square_divisible_by_5_and_7 
  (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = k^2)
  (h3 : 5 ∣ n)
  (h4 : 7 ∣ n) : 
  n = 1225 :=
sorry

end smallest_perfect_square_divisible_by_5_and_7_l461_461559


namespace train_pass_time_is_correct_l461_461614

-- Definitions of given conditions
def train_length : ℝ := 110 -- in meters
def train_speed : ℝ := 65 -- in km/hr
def man_speed : ℝ := 7 -- in km/hr

-- Conversion factor from km/hr to m/s
def km_hr_to_m_s (speed : ℝ) : ℝ := speed * (5.0 / 18.0)

-- Relative speed in m/s
def relative_speed : ℝ := km_hr_to_m_s (train_speed + man_speed)

-- Calculate the time it takes for the train to pass the man
def pass_time : ℝ := train_length / relative_speed

-- The proof statement
theorem train_pass_time_is_correct : pass_time = 5.5 :=
by
  sorry

end train_pass_time_is_correct_l461_461614


namespace complex_subtraction_l461_461961

-- Define the given conditions
def a : ℂ := 5 - 3 * complex.I
def b : ℂ := 2 + complex.I

-- State the theorem
theorem complex_subtraction : a - 3 * b = -1 - 6 * complex.I := by
  sorry

end complex_subtraction_l461_461961


namespace values_of_fractions_l461_461667

theorem values_of_fractions (A B : ℝ) :
  (∀ x : ℝ, 3 * x ^ 2 + 2 * x - 8 ≠ 0) →
  (∀ x : ℝ, (6 * x - 7) / (3 * x ^ 2 + 2 * x - 8) = A / (x - 2) + B / (3 * x + 4)) →
  A = 1 / 2 ∧ B = 4.5 :=
by
  intros h1 h2
  sorry

end values_of_fractions_l461_461667


namespace triangle_has_side_property_l461_461832

theorem triangle_has_side_property (a b c : ℝ) (A B C : ℝ) 
  (h₀ : 3 * b * Real.cos C + 3 * c * Real.cos B = a^2)
  (h₁ : A + B + C = Real.pi)
  (h₂ : a = 3) :
  a = 3 := 
sorry

end triangle_has_side_property_l461_461832


namespace correct_option_is_C_l461_461004

/- Definitions -/
def is_jump_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, (a i - a (i + 2)) * (a (i + 2) - a (i + 1)) > 0

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ i : ℕ, a (i + 1) = a i + d

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ i : ℕ, a (i + 1) = a i * q

def is_special_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, a (i + 1) = 2 * a i + 1

/- Theorem to be proved -/
theorem correct_option_is_C :
  (∃ (a : ℕ → ℝ) (d : ℝ), is_arithmetic_sequence a d ∧ is_jump_sequence a) ∨
  (∃ (a : ℕ → ℝ) (q > 0), is_geometric_sequence a q ∧ is_jump_sequence a) ∨
  (∀ (a : ℕ → ℝ) (q : ℝ), is_geometric_sequence a q → is_jump_sequence a → q ∈ Ioo (-1 : ℝ) 0) ∨
  (∀ (a : ℕ → ℝ), is_special_sequence a → is_jump_sequence a)
  → (∀ (a : ℕ → ℝ) (q : ℝ), is_geometric_sequence a q → is_jump_sequence a → q ∈ Ioo (-1 : ℝ) 0) :=
by
  sorry

end correct_option_is_C_l461_461004


namespace sample_mean_estimates_population_mean_l461_461393

variable (x_bar : ℝ) (mu : ℝ) -- Define variables for sample mean and population mean

def is_estimate_of (x_bar mu : ℝ) : Prop :=
  ∃ (data : List ℝ), x_bar = data.sum / data.length ∧ mu = stats.mean data

theorem sample_mean_estimates_population_mean
  (x_bar : ℝ) (mu : ℝ) (data : List ℝ) (h₁ : x_bar = data.sum / data.length)
  (h₂ : mu = stats.mean data) : is_estimate_of x_bar mu :=
by
  sorry

end sample_mean_estimates_population_mean_l461_461393


namespace magnitude_of_conjugate_l461_461200

-- Define the given complex number z and the imaginary unit i
def complex_z : ℂ := 1 / (1 - complex.i)

-- Define the conjugate of z
def conj_z := complex.conj complex_z

-- Define the magnitude of the conjugate of z
def mag_conj_z := complex.abs conj_z

-- State the theorem to be proved
theorem magnitude_of_conjugate :
  mag_conj_z = real.sqrt(2) / 2 :=
sorry

end magnitude_of_conjugate_l461_461200


namespace distance_correct_l461_461239

noncomputable def distance_between_A_and_B 
  (Va_initial Vb : ℕ) 
  (Va_return_factor : ℕ) 
  (second_meeting_distance : ℕ) : ℚ :=
  let D := second_meeting_distance * 20 / 7 in D

theorem distance_correct (second_meeting_distance : ℕ) : 
  distance_between_A_and_B (40 : ℕ) (60 : ℕ) (48 : ℕ) (50 : ℕ) =
  1000 / 7 :=
by
  -- The proof is omitted
  sorry

end distance_correct_l461_461239


namespace number_of_ordered_9_tuples_l461_461069

theorem number_of_ordered_9_tuples :
  let a_1, a_2, ..., a_9 : ℕ := sorry,
  ∀ i j k : ℕ, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ 9 → 
    ∃ l : ℕ, 1 ≤ l ∧ l ≤ 9 ∧ l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ (a_i + a_j + a_k + a_l = 100) →
  (the number of such ordered 9-tuples (a_1, a_2, ..., a_9) is 2017) :=
begin
  sorry
end

end number_of_ordered_9_tuples_l461_461069


namespace find_smallest_polynomial_l461_461064

noncomputable def f (x : ℝ) (b c : ℝ) := x^2 + b * x + c

def P (i : ℕ) (b c : ℝ) : ℝ × ℝ := (i, f i b c)

def tangent_line_at (i : ℕ) (b c : ℝ) : ℝ → ℝ :=
  λ x, (2 * i + b) * (x - i) + f i b c

def A (i : ℕ) (b c : ℝ) : ℝ × ℝ :=
  let x_int := i + 1 / 2 in
  (x_int, f x_int b c)

theorem find_smallest_polynomial :
  ∀ (b c : ℝ), ∃ p : Polynomial ℝ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 9 → p.eval (i + 1 / 2) = (A i b c).snd) ∧ p.degree = 2 := 
begin
  sorry
end

end find_smallest_polynomial_l461_461064


namespace typist_current_salary_l461_461933

-- Define the initial conditions as given in the problem
def initial_salary : ℝ := 6000
def raise_percentage : ℝ := 0.10
def reduction_percentage : ℝ := 0.05

-- Define the calculations for raised and reduced salaries
def raised_salary := initial_salary * (1 + raise_percentage)
def current_salary := raised_salary * (1 - reduction_percentage)

-- State the theorem to prove the current salary
theorem typist_current_salary : current_salary = 6270 := 
by
  -- Sorry is used to skip proof, overriding with the statement to ensure code builds successfully
  sorry

end typist_current_salary_l461_461933


namespace polynomial_value_at_two_l461_461547

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem polynomial_value_at_two : f 2 = 243 := by
  -- Proof steps go here
  sorry

end polynomial_value_at_two_l461_461547


namespace employed_females_percentage_l461_461579

theorem employed_females_percentage (total_employed_percentage employed_males_percentage employed_females_percentage : ℝ) 
    (h1 : total_employed_percentage = 64) 
    (h2 : employed_males_percentage = 48) 
    (h3 : employed_females_percentage = total_employed_percentage - employed_males_percentage) :
    (employed_females_percentage / total_employed_percentage * 100) = 25 :=
by
  sorry

end employed_females_percentage_l461_461579


namespace range_of_a_l461_461769

noncomputable def f (x a : ℝ) := Real.logBase a (x^2 - 6 * x + 5)

theorem range_of_a (a : ℝ) :
  (∀ x > a, ∀ y > x, f x a ≥ f y a) → (a ≥ 5) :=
by
  sorry

end range_of_a_l461_461769


namespace floor_e_equals_two_l461_461688

/-- Prove that the floor of Euler's number is 2. -/
theorem floor_e_equals_two : (⌊Real.exp 1⌋ = 2) :=
sorry

end floor_e_equals_two_l461_461688


namespace contrapositive_iff_l461_461916

theorem contrapositive_iff (a b : ℝ) :
  (a^2 < b → - real.sqrt b < a ∧ a < real.sqrt b) ↔
  (a ≥ real.sqrt b ∨ a ≤ - real.sqrt b → a^2 ≥ b) := 
sorry

end contrapositive_iff_l461_461916


namespace main_theorem_l461_461300
-- Import necessary library

-- Define the core statements as conditions:
variables {R r A B C AB_BC_ne} (h1 : AB ≠ BC)

-- First part of the problem
lemma part1 : IB = 2 * sqrt(R * r * cot (B / 2) / (cot (A / 2) + cot (C / 2))) :=
sorry

-- Second part of the problem
lemma part2 : (∠BOI - π / 2) * (cot (A / 2) * cot (C / 2) - (R + r) / (R - r)) < 0 ↔ ∠BOI ≠ π / 2 :=
sorry

-- Combine both parts into a main theorem
theorem main_theorem (h1: AB ≠ BC) :
  (IB = 2 * sqrt (R * r * cot (B / 2) / (cot (A / 2) + cot (C / 2)))) ∧
  ((∠BOI - π / 2) * (cot (A / 2) * cot (C / 2) - (R + r) / (R - r)) < 0 ↔ ∠BOI ≠ π / 2) :=
by sorry

end main_theorem_l461_461300


namespace min_value_expression_l461_461554

theorem min_value_expression (x y : ℝ) : 
  ∃ (z : ℝ), z ≤ (xy-1)^3 + (x+y)^3 ∧ ∀ (a b : ℝ), ((a*b-1)^3 + (a+b)^3) ≥ z := 
begin
  use -1,
  sorry
end

end min_value_expression_l461_461554


namespace min_value_y_minus_one_over_x_l461_461754

variable {x y : ℝ}

-- Condition 1: x is the median of the dataset
def is_median (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 5

-- Condition 2: The average of the dataset is 1
def average_is_one (x y : ℝ) : Prop := 1 + 2 + x^2 - y = 4

-- The statement to be proved
theorem min_value_y_minus_one_over_x :
  ∀ (x y : ℝ), is_median x → average_is_one x y → y = x^2 - 1 → (y - 1/x) ≥ 23/3 :=
by 
  -- This is a placeholder for the actual proof
  sorry

end min_value_y_minus_one_over_x_l461_461754


namespace union_of_sets_l461_461299

theorem union_of_sets (x y : ℕ) (A B : Set ℕ) (h1 : A = {x, y}) (h2 : B = {x + 1, 5}) (h3 : A ∩ B = {2}) : A ∪ B = {1, 2, 5} :=
sorry

end union_of_sets_l461_461299


namespace fill_tank_with_bowl_l461_461539

-- Define the dimensions of the tank
def tank_length : ℝ := 30
def tank_width : ℝ := 20
def tank_height : ℝ := 5

-- Define the dimensions of the bowl
def bowl_length : ℝ := 6
def bowl_width : ℝ := 4
def bowl_height : ℝ := 1

-- Calculate the volume of the tank
def V_tank : ℝ := tank_length * tank_width * tank_height

-- Calculate the volume of the bowl
def V_bowl : ℝ := bowl_length * bowl_width * bowl_height

-- Define the expected number of times to fill the tank
def num_times_to_fill_tank : ℝ := V_tank / V_bowl

-- State the theorem to prove
theorem fill_tank_with_bowl : num_times_to_fill_tank = 125 :=
by
  sorry

end fill_tank_with_bowl_l461_461539


namespace winning_candidate_votes_percentage_l461_461411

theorem winning_candidate_votes_percentage (majority : ℕ) (total_votes : ℕ) (winning_percentage : ℚ) :
  majority = 174 ∧ total_votes = 435 ∧ winning_percentage = 70 → 
  ∃ P : ℚ, (P / 100) * total_votes - ((100 - P) / 100) * total_votes = majority ∧ P = 70 :=
by
  sorry

end winning_candidate_votes_percentage_l461_461411


namespace election_total_votes_l461_461822

-- Define the parameters
variables (V : ℝ) -- Total number of votes
variables (p1 p2 : ℝ) -- Percentage of votes the candidates have

-- Define the necessary conditions
def candidate_percentage1 := 0.60
def candidate_percentage2 := 0.40
def majority_votes := 1040

-- Define the proof problem
theorem election_total_votes (h1 : p1 = candidate_percentage1) (h2 : p2 = candidate_percentage2) (h3 : p1 * V - p2 * V = majority_votes) : V = 5200 :=
sorry

end election_total_votes_l461_461822


namespace volume_of_tetrahedron_l461_461214

variables {A B C D : Type} [euclidean_geometry Tetrahedron]

-- Condition definitions
def sphere_circumscribed_around_ABCD (A B C D : P) : Prop :=
  ∃ (S : Sphere), (A ∈ S.pts) ∧ (B ∈ S.pts) ∧ (C ∈ S.pts) ∧ (D ∈ S.pts)

def sphere_tangent_to_plane_ABC_at_D (A B C D : P) (r : ℝ) : Prop :=
  ∃ (S : Sphere), S.radius = r ∧ (D ∈ S.pts) ∧ (Plane.contains (plane A B C) S)

-- Given conditions
def conditions (A B C D : P) : Prop :=
  (AD = 3) ∧
  (cos (angle B A C) = 4 / 5) ∧
  (cos (angle B A D) = 1 / sqrt 2) ∧
  (cos (angle C A D) = 1 / sqrt 2) ∧
  sphere_circumscribed_around_ABCD A B C D ∧
  sphere_tangent_to_plane_ABC_at_D A B C D 1

-- Statement to prove the volume of tetrahedron ABCD under given conditions is 18/5
theorem volume_of_tetrahedron (A B C D : P) (h : conditions A B C D) :
  volume (tetrahedron A B C D) = 18 / 5 :=
sorry

end volume_of_tetrahedron_l461_461214


namespace five_x_minus_two_l461_461800

theorem five_x_minus_two (x : ℚ) (h : 4 * x - 8 = 13 * x + 3) : 5 * (x - 2) = -145 / 9 := by
  sorry

end five_x_minus_two_l461_461800


namespace trajectory_is_straight_line_l461_461531

theorem trajectory_is_straight_line (x y : ℝ) (h : x + y = 0) : ∃ m b : ℝ, y = m * x + b :=
by
  use -1
  use 0
  sorry

end trajectory_is_straight_line_l461_461531


namespace percentage_increase_indeterminate_l461_461464

def Machine_A_production_rate : ℝ := 4
def sprockets_produced : ℕ := 440
def time_machine_Q (T : ℝ) : Prop := T > 0
def time_difference (T : ℝ) : ℝ := T + 10
def production_rate_machine_Q (T : ℝ) : ℝ := sprockets_produced / T
def production_rate_machine_P (T : ℝ) : ℝ := sprockets_produced / time_difference(T)

theorem percentage_increase_indeterminate (T : ℝ) (H : time_machine_Q T) :
  ∃ Q_rate : ℝ, ∃ P_rate : ℝ, P_rate = production_rate_machine_P T ∧ Q_rate = production_rate_machine_Q T ∧
  (∃ percentage_increase : ℝ, percentage_increase = ((Q_rate - Machine_A_production_rate) / Machine_A_production_rate) * 100)
  ↔ false := by
  sorry

end percentage_increase_indeterminate_l461_461464
