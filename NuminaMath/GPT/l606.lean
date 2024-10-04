import Mathlib

namespace cory_can_eat_fruits_in_105_ways_l606_606048

-- Define the number of apples, oranges, and bananas Cory has
def apples := 4
def oranges := 1
def bananas := 2

-- Define the total number of fruits Cory has
def total_fruits := apples + oranges + bananas

-- Calculate the number of distinct orders in which Cory can eat the fruits
theorem cory_can_eat_fruits_in_105_ways :
  (Nat.factorial total_fruits) / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas) = 105 :=
by
  -- Provide a sorry to skip the proof
  sorry

end cory_can_eat_fruits_in_105_ways_l606_606048


namespace distance_between_points_l606_606254

variables {x1 x2 y1 y2 m n p : ℝ}

-- Define the condition that (x1, y1) and (x2, y2) lie on the parabola
def on_parabola (x1 y1 x2 y2 m n p : ℝ) : Prop :=
  y1 = m * x1^2 + n * x1 + p ∧ y2 = m * x2^2 + n * x2 + p

-- The final target is proving the distance in terms of x1, x2, m, n
theorem distance_between_points:
  on_parabola x1 y1 x2 y2 m n p →
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = |x2 - x1| * real.sqrt (1 + m^2 * (x2 + x1)^2 + n^2) :=
sorry

end distance_between_points_l606_606254


namespace locus_area_l606_606030

theorem locus_area (R : ℝ) (r : ℝ) (hR : R = 6 * Real.sqrt 7) (hr : r = Real.sqrt 7) :
    ∃ (L : ℝ), (L = 2 * Real.sqrt 42 ∧ L^2 * Real.pi = 168 * Real.pi) :=
by
  sorry

end locus_area_l606_606030


namespace proportion_fourth_number_l606_606713

theorem proportion_fourth_number (x y : ℝ) (h₀ : 0.75 * y = 5 * x) (h₁ : x = 1.65) : y = 11 :=
by
  sorry

end proportion_fourth_number_l606_606713


namespace product_price_reduction_l606_606400

theorem product_price_reduction (z : ℝ) (x : ℝ) (hp1 : z > 0) (hp2 : 0.85 * 0.85 * z = z * (1 - x / 100)) : x = 27.75 := by
  sorry

end product_price_reduction_l606_606400


namespace tv_sale_increase_l606_606262

theorem tv_sale_increase (P Q : ℝ) :
  let new_price := 0.9 * P
  let original_sale_value := P * Q
  let increased_percentage := 1.665
  ∃ x : ℝ, (new_price * (1 + x / 100) * Q = increased_percentage * original_sale_value) → x = 85 :=
by
  sorry

end tv_sale_increase_l606_606262


namespace prism_volume_l606_606820

noncomputable def volume_prism (x y z : ℝ) : ℝ := x * y * z

theorem prism_volume (x y z : ℝ) (h1 : x * y = 12) (h2 : y * z = 8) (h3 : z * x = 6) :
  volume_prism x y z = 24 :=
by
  sorry

end prism_volume_l606_606820


namespace carson_air_per_pump_l606_606027

-- Define the conditions
def total_air_needed : ℝ := 2 * 500 + 0.6 * 500 + 0.3 * 500

def total_pumps : ℕ := 29

-- Proof problem statement
theorem carson_air_per_pump : total_air_needed / total_pumps = 50 := by
  sorry

end carson_air_per_pump_l606_606027


namespace projection_is_negative_three_l606_606174

variables {R : Type*} [InnerProductSpace ℝ R]

def projection_of_b_on_a (a b : R) (h₁ : ∥a∥ = 3) (h₂ : ∥b∥ = 2 * Real.sqrt 3) (h₃ : InnerProductSpace.is_orthogonal a (a + b)) : ℝ :=
  (inner a b) / ∥a∥

theorem projection_is_negative_three (a b : R) (h₁ : ∥a∥ = 3) (h₂ : ∥b∥ = 2 * Real.sqrt 3) (h₃ : InnerProductSpace.is_orthogonal a (a + b)) : projection_of_b_on_a a b h₁ h₂ h₃ = -3 :=
  by sorry

end projection_is_negative_three_l606_606174


namespace inequality_proof_l606_606532

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606532


namespace part1_part2_l606_606676

-- Step to define the required constants and ellipse properties.
def c : ℝ := 1
def a : ℝ := 2
def b : ℝ := sqrt 3

-- The standard equation of ellipse E
def standard_ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

-- Definitions for coordinates and perpendicular vectors
variables {x₀ y₀ t : ℝ}

-- Condition for point on ellipse
def point_on_ellipse (x₀ y₀ : ℝ) : Prop :=
  standard_ellipse_eq x₀ y₀

-- Perpendicular condition
def perp_vectors (x₀ y₀ t : ℝ) : Prop :=
  (t - x₀) * (2 - x₀) + y₀^2 = 0

-- Lean theorem statements for both parts.
theorem part1 : standard_ellipse_eq x y := sorry

theorem part2 (h1 : ∀ M, point_on_ellipse x₀ y₀ → perp_vectors x₀ y₀ t) : t ∈ set.Ioo (-2:ℝ) (-1:ℝ) := sorry

end part1_part2_l606_606676


namespace mutually_exclusive_A_C_independent_A_B_independent_B_C_l606_606276

/-- Definition of the events A, B, and C --/
def events : set ℕ → Prop := λ n, n ∈ {1, 2, 3, 4, 5, 6, 7, 8}

def is_red (n : ℕ) : Prop := n = 1 ∨ n = 2
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

/-- Probabilities for the events A, B, and C --/
def P (s : set ℕ) : ℚ := (s.card : ℚ) / 8

def A := {1, 2}
def B := {2, 4, 6}
def C := {3, 6}

theorem mutually_exclusive_A_C : ∀ x, is_red x → ¬ is_multiple_of_3 x := sorry

theorem independent_A_B : P (A ∩ B) = P A * P B := sorry

theorem independent_B_C : P (B ∩ C) = P B * P C := sorry

end mutually_exclusive_A_C_independent_A_B_independent_B_C_l606_606276


namespace train_crossing_l606_606891

theorem train_crossing (train_length : ℕ) (train_speed_kmh man_speed_kmh : ℕ) 
    (h_train_length : train_length = 400)
    (h_train_speed : train_speed_kmh = 46) 
    (h_man_speed : man_speed_kmh = 6) : 
    let relative_speed_kmh := train_speed_kmh - man_speed_kmh in
    let relative_speed_ms := (relative_speed_kmh * 1000) / 3600 in
    let crossing_time := train_length / relative_speed_ms in
    crossing_time = 36 :=
by
  -- Definitions and conditions
  have h_relative_speed_kmh : relative_speed_kmh = 40 := by
    simp [relative_speed_kmh, h_train_speed, h_man_speed]

  have h_relative_speed_ms : relative_speed_ms = 100 / 9 := by
    simp [relative_speed_ms, h_relative_speed_kmh]
    norm_num

  have h_crossing_time : crossing_time = train_length * 9 / 100 := by
    simp [crossing_time, h_relative_speed_ms]
    field_simp
    norm_num

  rw [h_train_length, h_crossing_time]
  norm_num
  simp
  sorry

end train_crossing_l606_606891


namespace students_exceed_rabbits_l606_606057

theorem students_exceed_rabbits (students_per_classroom rabbits_per_classroom number_of_classrooms : ℕ) 
  (h_students : students_per_classroom = 18)
  (h_rabbits : rabbits_per_classroom = 2)
  (h_classrooms : number_of_classrooms = 4) : 
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 64 :=
by {
  sorry
}

end students_exceed_rabbits_l606_606057


namespace system_solution_l606_606265

theorem system_solution (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) : 
  c / d = -2 / 5 :=
by
  sorry

end system_solution_l606_606265


namespace least_int_gt_sqrt_450_l606_606430

theorem least_int_gt_sqrt_450 : ∃ n : ℕ, n > nat.sqrt 450 ∧ (∀ m : ℕ, m > nat.sqrt 450 → n ≤ m) :=
by
  have h₁ : 20^2 = 400 := by norm_num
  have h₂ : 21^2 = 441 := by norm_num
  have h₃ : 22^2 = 484 := by norm_num
  have sqrt_bounds : 21 < real.sqrt 450 ∧ real.sqrt 450 < 22 :=
    by sorry
  use 22
  split
  · sorry
  · sorry

end least_int_gt_sqrt_450_l606_606430


namespace james_total_fish_correct_l606_606760

variables (trout salmon tuna total_fish : ℕ)

def conditions (trout salmon tuna : ℕ) : Prop :=
  trout = 200 ∧
  salmon = 3 * (trout / 2) ∧
  tuna = 2 * trout

theorem james_total_fish_correct (h : conditions trout salmon tuna) :
  total_fish = trout + salmon + tuna → total_fish = 900 :=
by {
  intro h1,
  rcases h with ⟨ht, hs, hue⟩,
  rw [ht, hs, hue] at h1,
  sorry
}

end james_total_fish_correct_l606_606760


namespace repeating_decimal_sum_l606_606067

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem repeating_decimal_sum : repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l606_606067


namespace ap_bisects_cd_l606_606807

open EuclideanGeometry

-- Given a convex pentagon ABCDE
variable (A B C D E P : Point)
variable [convex_pentagon ABCDE: ConvexHull A B C D E]

-- with conditions of angle equalities
variable (h1: Angle BAC = Angle CAD)
variable (h2: Angle CAD = Angle DAE)
variable (h3: Angle ABC = Angle ACD)
variable (h4: Angle ACD = Angle ADE)

-- and diagonals BD and CE intersect at point P
variable (h5: Intersect BD CE P)

-- Prove that AP bisects CD
theorem ap_bisects_cd : SegmentBisector A P C D :=
by
  sorry

end ap_bisects_cd_l606_606807


namespace a_sequence_formula_T_n_sum_l606_606291

noncomputable def a_sequence (n : ℕ) : ℕ := 3 * n - 1

def b_sequence (n : ℕ) : ℚ := 1 / (a_sequence n * a_sequence (n + 1))

def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_sequence i 

theorem a_sequence_formula (n : ℕ) : a_sequence n = 3 * n - 1 :=
sorry

theorem T_n_sum (n : ℕ) : T_n n = n / (2 * (3 * n + 2)) :=
sorry

end a_sequence_formula_T_n_sum_l606_606291


namespace solve_for_z_l606_606239

variable {z : ℂ}

theorem solve_for_z (h : complex.I * z = 4 + 3*complex.I) : z = 3 - 4*complex.I :=
sorry

end solve_for_z_l606_606239


namespace initial_pocket_money_l606_606765

variable (P : ℝ)

-- Conditions
axiom chocolates_expenditure : P * (1/9) ≥ 0
axiom fruits_expenditure : P * (2/5) ≥ 0
axiom remaining_money : P * (22/45) = 220

-- Theorem statement
theorem initial_pocket_money : P = 450 :=
by
  have h₁ : P * (1/9) + P * (2/5) = P * (23/45) := by sorry
  have h₂ : P * (1 - 23/45) = P * (22/45) := by sorry
  have h₃ : P = 220 / (22/45) := by sorry
  have h₄ : P = 220 * (45/22) := by sorry
  have h₅ : P = 450 := by sorry
  exact h₅

end initial_pocket_money_l606_606765


namespace angle_DCE_l606_606745

theorem angle_DCE {A B C D E : Type}
  (h1 : Measure.Angle E C D = 58)
  (h2 : Measure.Angle C D C = 90) :
  Measure.Angle D C E = 32 :=
sorry

end angle_DCE_l606_606745


namespace number_of_triangles_with_perimeter_eight_l606_606180

theorem number_of_triangles_with_perimeter_eight :
  { (a, b, c) : ℕ × ℕ × ℕ // a + b + c = 8 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b }.card = 5 :=
by
  -- Declaration of the proof structure
  sorry

end number_of_triangles_with_perimeter_eight_l606_606180


namespace largest_prime_factor_of_divisors_sum_180_l606_606789

def divisors_sum (n : ℕ) : ℕ :=
  (divisors n).sum

def largest_prime_factor (n : ℕ) : ℕ :=
  (factors n).filter prime ∣> ∧ .toList.maximum' sorry -- assume there is a maximum

theorem largest_prime_factor_of_divisors_sum_180 :
  ∃ N, N = divisors_sum 180 ∧ largest_prime_factor N = 13 := by
  sorry

end largest_prime_factor_of_divisors_sum_180_l606_606789


namespace count_perfect_squares_17th_digit_7_greater_than_8_l606_606756

theorem count_perfect_squares_17th_digit_7_greater_than_8 :
  let perfect_squares := { n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 10^20 },
      count_digit_7 := { n ∈ perfect_squares | ((n / 10^16) % 10 = 7) },
      count_digit_8 := { n ∈ perfect_squares | ((n / 10^16) % 10 = 8) }
  in count_digit_7.card > count_digit_8.card := 
by
  sorry

end count_perfect_squares_17th_digit_7_greater_than_8_l606_606756


namespace surface_area_change_is_6_square_feet_more_l606_606928

-- Defining the dimensions of the rectangular solid
def length : ℝ := 4
def width : ℝ := 3
def height : ℝ := 2

-- Defining the dimensions of the cube to be removed
def cube_side : ℝ := 1

-- Original surface area calculation
def original_surface_area : ℝ :=
  2 * (length * width + length * height + width * height)

-- Change in surface area due to removal of the cube
def change_in_surface_area : ℝ :=
  6 * (cube_side * cube_side)

-- Prove that the change in the total surface area is 6 square feet more
theorem surface_area_change_is_6_square_feet_more :
  change_in_surface_area = 6 := 
by
  sorry

end surface_area_change_is_6_square_feet_more_l606_606928


namespace given_statements_l606_606037

def addition_is_associative (x y z : ℝ) : Prop := (x + y) + z = x + (y + z)

def averaging_is_commutative (x y : ℝ) : Prop := (x + y) / 2 = (y + x) / 2

def addition_distributes_over_averaging (x y z : ℝ) : Prop := 
  x + (y + z) / 2 = (x + y + x + z) / 2

def averaging_distributes_over_addition (x y z : ℝ) : Prop := 
  (x + (y + z)) / 2 = ((x + y) / 2) + ((x + z) / 2)

def averaging_has_identity_element (x e : ℝ) : Prop := 
  (x + e) / 2 = x

theorem given_statements (x y z e : ℝ) :
  addition_is_associative x y z ∧ 
  averaging_is_commutative x y ∧ 
  addition_distributes_over_averaging x y z ∧ 
  ¬averaging_distributes_over_addition x y z ∧ 
  ¬∃ e, averaging_has_identity_element x e :=
by
  sorry

end given_statements_l606_606037


namespace find_k_l606_606970

noncomputable def linearly_dependent {k : ℝ} : Prop :=
  ∃ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ (a * 2 + b * 4 = 0) ∧ (a * 5 + b * k = 0) 

theorem find_k : linearly_dependent {k} → k = 10 := by
  sorry

end find_k_l606_606970


namespace ellipse_focus_xaxis_l606_606155

theorem ellipse_focus_xaxis (k : ℝ) (h : 1 - k > 2 + k ∧ 2 + k > 0) : -2 < k ∧ k < -1/2 :=
by sorry

end ellipse_focus_xaxis_l606_606155


namespace triangle_is_isosceles_l606_606347

theorem triangle_is_isosceles 
  (A B C M K A_1 B_1 : Point)
  (h1 : is_median C M A B)
  (h2 : lies_on K C M)
  (h3 : line_intersects AK BC A_1)
  (h4 : line_intersects BK AC B_1)
  (h5 : is_cyclic_quadrilateral A B_1 A_1 B) 
  : is_isosceles_triangle A B C :=
sorry

end triangle_is_isosceles_l606_606347


namespace triangle_abc_is_isosceles_l606_606355

variable (A B C M K A1 B1 : Point)

variables (C_M_median : is_median C M A B)
variables (K_on_CM : on_line_segment C M K)
variables (A1_on_BC : on_intersect AK BC A1)
variables (B1_on_AC : on_intersect BK AC B1)
variables (AB1A1B_inscribed : is_inscribed_quadrilateral A B1 A1 B)

theorem triangle_abc_is_isosceles : AB = AC :=
by
  sorry

end triangle_abc_is_isosceles_l606_606355


namespace correct_statements_for_curve_l606_606658

def symmetric_about_x_axis (P : ℝ × ℝ) := ∃ (y : ℝ), P = (y, -y)
def symmetric_about_y_axis (P : ℝ × ℝ) := ∃ (y : ℝ), P = (-y, y)
def symmetric_about_origin (P : ℝ × ℝ) := ∃ (y : ℝ), P = (-y, -y)
def closed_curve {x y : ℝ} (h : x^4 + y^2 = 1) := ∀ P, P = (x, y) → (∃ Q, Q = P)
def center_of_symmetry (P : ℝ × ℝ) := ∀ Q, Q = (-P.1, -P.2)

theorem correct_statements_for_curve : 
  (∀ (x y : ℝ), x^4 + y^2 = 1 → symmetric_about_x_axis (x, y)) ∧ 
  (∀ (x y : ℝ), x^4 + y^2 = 1 → symmetric_about_y_axis (x, y)) ∧ 
  (∀ (x y : ℝ), x^4 + y^2 = 1 → symmetric_about_origin (x, y)) ∧ 
  (∃ (P : ℝ × ℝ), center_of_symmetry P ∧ (∀ Q, Q = P → ¬∃ R, R = (-P.1, -P.2))) ∧ 
  (∀ (x y : ℝ), x^4 + y^2 = 1 → closed_curve (x^4 + y^2 = 1)) ∧ 
  (∀ (x y : ℝ), x^4 + y^2 = 1 → ∀ P, P = (x, y) → (x^2 + y^2 ≥ 1) → (area (x^4 + y^2 = 1) > π)) :=
sorry

end correct_statements_for_curve_l606_606658


namespace largest_prime_accidentally_crossed_out_l606_606013

theorem largest_prime_accidentally_crossed_out : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ p < q ∧ q ≤ 1000 / 3 ∧ q^2 < p * q ∧ p * q ≤ 1000 ∧ 331 = q :=
sorry

end largest_prime_accidentally_crossed_out_l606_606013


namespace largest_prime_factor_sum_divisors_180_l606_606795

theorem largest_prime_factor_sum_divisors_180 :
  let N := ∑ d in (Finset.divisors 180), d in
  Nat.greatest_prime_factor N = 13 :=
by
  sorry

end largest_prime_factor_sum_divisors_180_l606_606795


namespace problem_equation_false_l606_606759

theorem problem_equation_false (K T U Ch O H : ℕ) 
  (h1 : K ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h2 : T ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h3 : U ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h4 : Ch ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h5 : O ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h6 : H ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (distinct : ∀ x ∈ {K, T, U, Ch, O, H}, ∀ y ∈ {K, T, U, Ch, O, H}, x ≠ y → x ≠ y) :
  (K * 0 * T = U * Ch * O * H * H * U) → False :=
by
  sorry

end problem_equation_false_l606_606759


namespace charity_total_cost_l606_606568

theorem charity_total_cost
  (plates : ℕ)
  (rice_cost_per_plate chicken_cost_per_plate : ℕ)
  (h1 : plates = 100)
  (h2 : rice_cost_per_plate = 10)
  (h3 : chicken_cost_per_plate = 40) :
  plates * (rice_cost_per_plate + chicken_cost_per_plate) / 100 = 50 := 
by
  sorry

end charity_total_cost_l606_606568


namespace problem_statement_l606_606798

open Probability

noncomputable def binomialProbability : ℕ → ℕ → ℚ → ℚ := sorry

theorem problem_statement (X : ℕ → ℕ → ℚ -> ℕ → ℚ) (n : ℕ) (p : ℚ) :
  (X n p).pdf (λ x => x = 2) = 80 / 243 :=
by sorry

end problem_statement_l606_606798


namespace complex_solution_l606_606203

theorem complex_solution (z : ℂ) (h : ((0 : ℝ) + 1 * z = 4 + 3 * (complex.I))) : 
  z = 3 - 4 * (complex.I) :=
sorry

end complex_solution_l606_606203


namespace solve_for_z_l606_606219

variable (z : ℂ)

theorem solve_for_z (h : (complex.I * z = 4 + 3 * complex.I)) : z = 3 - 4 * complex.I := 
sorry

end solve_for_z_l606_606219


namespace addition_schemes_count_l606_606913

variable (Peilan Bingpian Dingxiang Shichangpu : Type)

-- Definition of the set of Chinese medicines
noncomputable def ChineseMedicines := {Peilan, Bingpian, Dingxiang, Shichangpu}

-- Total number of different possible addition schemes where at least one medicine is added
theorem addition_schemes_count : ChineseMedicines.to_finset.powerset.card - 1 = 15 := 
by sorry

end addition_schemes_count_l606_606913


namespace area_of_path_approx_l606_606504

noncomputable def A_path_approx : ℝ :=
  let π := Real.pi
  let r_inner := 2
  let r_outer := r_inner + 0.25
  let A_inner := π * r_inner^2
  let A_outer := π * r_outer^2
  let A_path := A_outer - A_inner
  A_path

theorem area_of_path_approx : A_path_approx ≈ 3.34 :=
by
  let π := Real.pi
  let r_inner := 2
  let r_outer := r_inner + 0.25
  let A_inner := π * r_inner^2
  let A_outer := π * r_outer^2
  let A_path := A_outer - A_inner
  calc
    A_path = π * (r_outer^2 - r_inner^2) := by sorry
    ... = π * (2.25^2 - 2^2) := by sorry
    ... ≈ 3.34 := by
      have h1 : π ≈ 3.14 := by sorry
      show 1.0625 * 3.14 ≈ 3.34 by linarith

end area_of_path_approx_l606_606504


namespace rational_numbers_in_set_l606_606753

noncomputable def sqrt_3 (x : ℝ) := (x ^ (1/3))
noncomputable def sqrt_2 (x : ℝ) := (x ^ (1/2))

theorem rational_numbers_in_set :
  card {x : ℝ // x = sqrt_3 8 ∨ x = π/2 ∨ x = sqrt_2 12 ∨ x = 7/3 ∧ is_rational x} = 2 :=
by sorry

end rational_numbers_in_set_l606_606753


namespace area_of_triangle_PQR_l606_606297

variables {P Q R S T : Type} 

-- Let's assume PQR is an isosceles triangle.
variables (PQ QR : ℝ) (QS : ℝ)
variable [IsoscelesTriangle : PQ = QR]
-- Let's define QT = 8
def QT : ℝ := 8

-- Assuming the tangents of the specified angles form a geometric progression.
def tan_RQT := tan (angle R Q T)
def tan_SQT := tan (angle S Q T)
def tan_PQT := tan (angle P Q T)
def tan_geometric_progression := tan_RQT * tan_PQT = tan_SQT^2

-- Assuming the cotangents of the specified angles form an arithmetic progression.
def cot_SQT := cot (angle S Q T)
def cot_RQT := cot (angle R Q T)
def cot_SQP := cot (angle S Q P)
def cot_arithmetic_progression := 2 * cot_SQT = cot_RQT + cot_SQP

-- Main statement to prove the area of triangle PQR
theorem area_of_triangle_PQR 
  (tan_geometric_prog : tan_geometric_progression)
  (cot_arith_prog : cot_arithmetic_progression)
  (isosceles : PQ = QR)
  (QT_len : QT = 8) 
  : 
  (1/2 * PQ * QS = 32/3) :=
sorry

end area_of_triangle_PQR_l606_606297


namespace graph_symmetry_l606_606696

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

theorem graph_symmetry :
  (∀ x, f(x) = f(11*Real.pi/12 - x + 11*Real.pi/12)) ∧
  (f(2*Real.pi/3) = 0) :=
by
  sorry

end graph_symmetry_l606_606696


namespace solve_y_in_terms_of_b_l606_606838

theorem solve_y_in_terms_of_b (b y : ℝ) (h : b ≠ 0)
  (det_eq_zero : det ![
    ![y + b, -y, y],
    ![-y, y + b, -y],
    ![y, -y, y + b]
  ] = 0) : y = -b / 3 :=
by {
  sorry
}

end solve_y_in_terms_of_b_l606_606838


namespace perimeter_8_triangles_count_l606_606183

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def distinct_triangles_count (n : ℕ) : ℕ := 
  (Finset.unorderedPairs (Finset.range (n - 2 + 1))).filter (λ ⟨a, bc⟩, ∃ b c, bc = b + c ∧ is_triangle a b c ∧ a + b + c = n).card

theorem perimeter_8_triangles_count : distinct_triangles_count 8 = 2 := 
by
  sorry

end perimeter_8_triangles_count_l606_606183


namespace max_inequality_l606_606777

theorem max_inequality (n : ℕ) (a b : ℕ → ℝ)
  (h1 : 0 ≤ ∀ i, a i)
  (h2 : 0 ≤ ∀ i, b i)
  (h3 : ∑ i in finset.range (n + 1), (a i + b i) = 1)
  (h4 : ∑ i in finset.range (n + 1), i * (a i - b i) = 0)
  (h5 : ∑ i in finset.range (n + 1), i^2 * (a i + b i) = 10) :
  ∀ k, 1 ≤ k → k ≤ n → max (a k) (b k) ≤ 10 / (10 + k^2) :=
by
  sorry

end max_inequality_l606_606777


namespace fraction_muscle_to_fat_is_quarter_l606_606761

def initial_weight := 120
def final_weight := 150
def muscle_gain_fraction := 0.20

noncomputable def muscle_gain := muscle_gain_fraction * initial_weight
noncomputable def total_weight_gain := final_weight - initial_weight
noncomputable def fat_gain := total_weight_gain - muscle_gain
noncomputable def fat_fraction := fat_gain / muscle_gain

theorem fraction_muscle_to_fat_is_quarter : fat_fraction = 1 / 4 := 
  sorry

end fraction_muscle_to_fat_is_quarter_l606_606761


namespace largest_prime_factor_divisors_sum_l606_606785

def prime_factors (n : ℕ) : List ℕ := sorry -- Dummy placeholder for prime factorization

theorem largest_prime_factor_divisors_sum :
  let N := (1 + 2 + 2^2) * (1 + 3 + 3^2) * (1 + 5)
  in List.maximum (prime_factors N) = 13 :=
by
  let N := (1 + 2 + 2^2) * (1 + 3 + 3^2) * (1 + 5)
  have h : prime_factors N = [2, 3, 7, 13] := sorry -- Placeholder
  exact List.maximum_eq_some.mp ⟨13, List.mem_cons_self 13 _, sorry⟩

end largest_prime_factor_divisors_sum_l606_606785


namespace claire_photos_l606_606816

variable (C : ℕ) -- Claire's photos
variable (L : ℕ) -- Lisa's photos
variable (R : ℕ) -- Robert's photos

-- Conditions
axiom Lisa_photos : L = 3 * C
axiom Robert_photos : R = C + 16
axiom Lisa_Robert_same : L = R

-- Proof Goal
theorem claire_photos : C = 8 :=
by
  -- Sorry skips the proof and allows the theorem to compile
  sorry

end claire_photos_l606_606816


namespace mod_multiplication_l606_606842

theorem mod_multiplication :
  (176 * 929) % 50 = 4 :=
by
  sorry

end mod_multiplication_l606_606842


namespace seq_sum_2018_l606_606168

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 2008 ∧ a 2 = 2009 ∧ a 3 = 1 ∧ ∀ n, a (n+1) = a n + a (n+2)

theorem seq_sum_2018 (a : ℕ → ℤ) (h : seq a) : (∑ i in Finset.range 2018, a i.succ) = 4017 :=
by sorry

end seq_sum_2018_l606_606168


namespace exists_polynomial_p_l606_606997

theorem exists_polynomial_p : ∃ p : ℚ[x], p.comp(p) = (λ x, x * p + x^2) :=
by
  let p := -X + 1
  use p
  sorry

end exists_polynomial_p_l606_606997


namespace least_integer_greater_than_sqrt_450_l606_606470

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606470


namespace simplify_and_evaluate_l606_606378

theorem simplify_and_evaluate (a : ℤ) (h : a = -4) :
  (4 * a ^ 2 - 3 * a) - (2 * a ^ 2 + a - 1) + (2 - a ^ 2 + 4 * a) = 19 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l606_606378


namespace age_of_15th_student_is_15_7_l606_606729

noncomputable def total_age_all_students : ℝ := 15 * 15.2
noncomputable def total_age_first_group : ℝ := 5 * 14
noncomputable def total_age_third_group : ℝ := 3 * 16.6
noncomputable def total_age_remaining_boys : ℝ := 3 * 15.4

noncomputable def age_15th_student (G : ℝ) : ℝ := 
  let total_age_second_group := G in
  total_age_second_group / 4

theorem age_of_15th_student_is_15_7 (G : ℝ) (h : G = 62.8) :
  age_15th_student G = 15.7 := by 
  rw [h]
  simp
  norm_num

end age_of_15th_student_is_15_7_l606_606729


namespace least_integer_gt_sqrt_450_l606_606466

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l606_606466


namespace charity_dinner_cost_l606_606565

def cost_of_rice_per_plate : ℝ := 0.10
def cost_of_chicken_per_plate : ℝ := 0.40
def number_of_plates : ℕ := 100

theorem charity_dinner_cost : 
  cost_of_rice_per_plate + cost_of_chicken_per_plate * number_of_plates = 50 :=
by
  sorry

end charity_dinner_cost_l606_606565


namespace smallest_sum_S5_l606_606743

noncomputable theory

variables {a_n : ℕ → ℝ} {S_n : ℕ → ℝ}

-- Assumption: Arithmetic sequence with a_n as general term
def arithmetic_seq (a_n : ℕ → ℝ) : Prop :=
  ∃ d a, ∀ n, a_n n = a + n * d

-- Given conditions
axiom a3_a8_gt_zero : a_n 3 + a_n 8 > 0
axiom S9_lt_zero : S_n 9 < 0

-- S_n is the sum of first n terms of arithmetic sequence
def sum_arithmetic_seq (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2

-- question to prove: The smallest among S_1, S_2, ..., S_9 is S_5
theorem smallest_sum_S5 (h_arith_seq : arithmetic_seq a_n) (h_sum_seq : sum_arithmetic_seq S_n a_n) : 
  ∃ i, i ∈ finset.range 1 10 ∧ S_n i = S_n 5 ∧ (∀ j ∈ finset.range 1 10, S_n j ≥ S_n 5) :=
sorry

end smallest_sum_S5_l606_606743


namespace find_abc_l606_606330

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.cos x + 3 * Real.sin x

theorem find_abc (a b c : ℝ) : 
  (∀ x : ℝ, a * f x + b * f (x - c) = 1) →
  (∃ n : ℤ, a = 1 / 2 ∧ b = 1 / 2 ∧ c = (2 * n + 1) * Real.pi) :=
by
  sorry

end find_abc_l606_606330


namespace find_set_of_x_l606_606339

def f (x : ℝ) : ℝ := ite (x = abs x) (2^x - 4) (2^(abs x) - 4)

lemma even_fn (x : ℝ) : f(x) = f(-x) := 
by sorry

theorem find_set_of_x : {x : ℝ | f(x-2) > 0} = { x : ℝ | x < 0 ∨ x > 4} :=
by sorry

end find_set_of_x_l606_606339


namespace solution_set_of_inequality_l606_606967

variable {f : ℝ → ℝ}

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem solution_set_of_inequality (h_odd : odd_function f) (h_f_neg1 : f (-1) = -2)
    (h_f'_pos : ∀ x > 0, f' x > 2) :
  {x : ℝ | f x > 2 * x} = {x | x ∈ Ioo (-1 : ℝ) 0 ∪ Ioi 1} :=
sorry

end solution_set_of_inequality_l606_606967


namespace k_plus_m_eq_27_l606_606322

theorem k_plus_m_eq_27 (k m : ℝ) (a b c : ℝ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a > 0) (h5 : b > 0) (h6 : c > 0)
  (h7 : a + b + c = 8) 
  (h8 : k = a * b + a * c + b * c) 
  (h9 : m = a * b * c) :
  k + m = 27 :=
by
  sorry

end k_plus_m_eq_27_l606_606322


namespace angle_maximization_l606_606500

noncomputable def maximize_angle_OPX (O : Point) (r : ℝ) (X : Point) (A B : Point) (C : Circle) := 
  (Circle.center C = O) ∧
  (A ∈ C.points) ∧
  (B ∈ C.points) ∧
  (X ≠ O) ∧
  (X ∈ C.interior) ∧
  (∃ ABchord : Line, (X ∈ ABchord) ∧ (O ∈ ABchord.perpendicular) ∧ (A, B ∈ ABchord)) ∧
  (∀ P : Point, P ∈ C.points → (∠ O P X ≤ ∠ O A X))

theorem angle_maximization (O : Point) (r : ℝ) (X : Point) (A B : Point) (C : Circle) : 
  maximize_angle_OPX O r X A B C :=
by {
  sorry
}

end angle_maximization_l606_606500


namespace tangent_circles_r_value_l606_606717

-- Definitions of the circles and center distances
def circle1 (x y : ℝ) := x^2 + y^2 = 1
def circle2 (x y : ℝ) (r : ℝ) := (x-3)^2 + y^2 = r^2

theorem tangent_circles_r_value (r : ℝ) (h : r > 0) :
  (∀ x y, circle1 x y → circle2 x y r → (∃ c1, ∃ c2, dist c1 c2 = r - 1)) → 
  r = 4 :=
by
  sorry -- The actual proof steps are omitted.

end tangent_circles_r_value_l606_606717


namespace largest_number_is_B_l606_606497

def option_A := 8.25678
def option_B := 8.256777777...
def option_C := 8.2567676767...
def option_D := 8.2567567567...
def option_E := 8.2567256725...

theorem largest_number_is_B :
  option_B > option_A ∧
  option_B > option_C ∧
  option_B > option_D ∧
  option_B > option_E :=
sorry

end largest_number_is_B_l606_606497


namespace probability_of_total_greater_than_7_l606_606279

-- Definitions for conditions
def total_outcomes : ℕ := 36
def favorable_outcome_count : ℕ := 15

-- Probability Calculation
def calc_probability (total : ℕ) (favorable : ℕ) : ℚ := favorable / total 

-- The theorem statement
theorem probability_of_total_greater_than_7 :
  calc_probability total_outcomes favorable_outcome_count = 5 / 12 :=
sorry

end probability_of_total_greater_than_7_l606_606279


namespace perpendicular_lines_l606_606691

variables (m n l : Line) (α β : Plane)
variables (h1 : Parallel α β) (h2 : Perpendicular l α) (h3 : SubsetLine n β)

theorem perpendicular_lines (hα_parallel_β : Parallel α β) (hl_perp_α : Perpendicular l α) (hn_sub_β : SubsetLine n β) : Perpendicular l n :=
sorry

end perpendicular_lines_l606_606691


namespace circle_center_coordinates_l606_606872

theorem circle_center_coordinates (x y : ℝ) (h : y = 4 * sin (atan2 y x)) :
  (0, 2) = (0, 2) :=
by { sorry }

end circle_center_coordinates_l606_606872


namespace geometric_sequence_common_ratio_l606_606129

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} {S : ℕ → ℝ} (q : ℝ) 
  (h_pos : ∀ n, 0 < a n)
  (h_geo : ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q))  
  (h_condition : ∀ n : ℕ+, S (2 * n) / S n < 5) :
  0 < q ∧ q ≤ 1 :=
sorry

end geometric_sequence_common_ratio_l606_606129


namespace repeating_decimal_sum_l606_606080

theorem repeating_decimal_sum : (0.\overline{6} + 0.\overline{2} - 0.\overline{4} : ℚ) = (4 / 9 : ℚ) :=
by
  sorry

end repeating_decimal_sum_l606_606080


namespace inequality_proof_l606_606537

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606537


namespace polynomial_g_zero_l606_606320

/-- Let g be a non-constant polynomial such that 
  ∀ (x : ℝ), x ≠ 0 → g(x - 1) + g(x) + g(x + 1) = (g(x) ^ 2) / (4030 * x).
  Then g(0) = 0. -/
theorem polynomial_g_zero (g : ℝ → ℝ) (h_poly : ∀ x, Polynomial g x) (h_nonconst : ¬(∀ x, g x = g 0)) :
  (∀ x : ℝ, x ≠ 0 → g(x - 1) + g(x) + g(x + 1) = (g(x) ^ 2) / (4030 * x)) → g 0 = 0 :=
sorry

end polynomial_g_zero_l606_606320


namespace ABCD_cyclic_l606_606916

-- Definitions for points and circles used in the problem context.
variables {A B C D O : Type} [Point A] [Point B] [Point C] [Point D] [Point O]
          {circle_A: Circle A} {circle_B: Circle B} {circle_C: Circle C} {circle_D: Circle D} {circle_O: Circle O}

-- Define the conditions.
def passes_through : Circle D → Point A → Point B → Point O → Prop := sorry
def incenter : Triangle ABC → Point O → Prop := sorry
def tangent : Side BC → Circle O → Prop := sorry
def extensions_tangent : Side AB → Side AC → Circle O → Prop := sorry
def cyclic : Point A → Point B → Point C → Point D → Prop := sorry

-- The theorem statement.
theorem ABCD_cyclic (h1 : passes_through circle_D A B O)
                    (h2 : incenter (triangle ABC) O) 
                    (h3 : tangent (side BC) circle_O)
                    (h4 : extensions_tangent (side AB) (side AC) circle_O) :
                    cyclic A B C D :=
sorry

end ABCD_cyclic_l606_606916


namespace hyperbola_equation_l606_606166

theorem hyperbola_equation
  (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
  (asymptote : ∀ x : ℝ, b = sqrt 3 * a)
  (focus : a^2 + b^2 = 16) :
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1) :=
begin
  sorry
end

end hyperbola_equation_l606_606166


namespace find_a_for_square_binomial_l606_606103

theorem find_a_for_square_binomial (a r s : ℝ) 
  (h1 : ax^2 + 18 * x + 9 = (r * x + s)^2)
  (h2 : a = r^2)
  (h3 : 2 * r * s = 18)
  (h4 : s^2 = 9) : 
  a = 9 := 
by sorry

end find_a_for_square_binomial_l606_606103


namespace problem_inequality_l606_606669

theorem problem_inequality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    (y + z) / (2 * x) + (z + x) / (2 * y) + (x + y) / (2 * z) ≥
    2 * x / (y + z) + 2 * y / (z + x) + 2 * z / (x + y) :=
by
  sorry

end problem_inequality_l606_606669


namespace part_I_part_II_l606_606270

variables {A B C a b c : ℝ}
variables {α β γ : ℝ} -- angles in the triangle

-- Conditions
def condition_1 (A B : ℝ) (a b c : ℝ) : Prop := (2 * c - b) * (Real.cos A) - a * (Real.cos B) = 0
def condition_2 : Prop := a = 4
def condition_3 : Prop := A = Real.pi / 3

-- Prove that A = π/3
theorem part_I (a b c : ℝ) (A B : ℝ) (h : condition_1 A B a b c) (h_nonzero : Real.sin (Real.pi - A - B) ≠ 0) : 
  A = Real.pi / 3 :=
sorry

-- Prove that the maximum area is 4√3
theorem part_II (a b c : ℝ) (A : ℝ) (h1 : condition_2) (h2 : condition_3) : 
  (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 3 :=
sorry

end part_I_part_II_l606_606270


namespace probability_same_gender_probability_same_school_l606_606832

-- Definitions for the problem setup
def school_a_teachers := ["A", "B", "C"] -- A and B are male, C is female
def school_b_teachers := ["D", "E", "F"] -- D is male, E and F are female
def all_teachers := school_a_teachers ++ school_b_teachers

-- The number of male teachers in schools A and B
def school_a_male_teachers := ["A", "B"]
def school_b_male_teachers := ["D"]

-- The number of female teachers in schools A and B
def school_a_female_teachers := ["C"]
def school_b_female_teachers := ["E", "F"]

-- Number of ways to choose a male teacher from school A and B respectively
def male_pairs := [(school_a_male_teachers.product school_b_male_teachers).length]

-- Number of ways to choose a female teacher from school A and B respectively
def female_pairs := [(school_a_female_teachers.product school_b_female_teachers).length]

-- Number of total pairs of teachers from school A and B
def total_pairs := (school_a_teachers.product school_b_teachers).length

-- Probability for Question I
theorem probability_same_gender : (male_pairs.sum + female_pairs.sum).toRational / total_pairs.toRational = 4 / 9 :=
by
  sorry

-- Number of ways to choose 2 teachers from the same school
def same_school_pairs := [(school_a_teachers.choose 2).length + (school_b_teachers.choose 2).length]

-- Number of ways to choose any 2 teachers from all teachers
def total_pairs_any := (all_teachers.choose 2).length

-- Probability for Question II
theorem probability_same_school : (same_school_pairs.sum).toRational / total_pairs_any.toRational = 2 / 5 :=
by
  sorry

end probability_same_gender_probability_same_school_l606_606832


namespace coffee_decaf_percentage_l606_606887

variable (initial_stock : ℝ) (initial_decaf_percent : ℝ)
variable (new_stock : ℝ) (new_decaf_percent : ℝ)

noncomputable def decaf_coffee_percentage : ℝ :=
  let initial_decaf : ℝ := initial_stock * (initial_decaf_percent / 100)
  let new_decaf : ℝ := new_stock * (new_decaf_percent / 100)
  let total_decaf : ℝ := initial_decaf + new_decaf
  let total_stock : ℝ := initial_stock + new_stock
  (total_decaf / total_stock) * 100

theorem coffee_decaf_percentage :
  initial_stock = 400 →
  initial_decaf_percent = 20 →
  new_stock = 100 →
  new_decaf_percent = 50 →
  decaf_coffee_percentage initial_stock initial_decaf_percent new_stock new_decaf_percent = 26 :=
by
  intros
  sorry

end coffee_decaf_percentage_l606_606887


namespace moles_of_CH4_l606_606647

theorem moles_of_CH4 (moles_Be2C moles_H2O : ℕ) (balanced_equation : 1 * Be2C + 4 * H2O = 2 * CH4 + 2 * BeOH2) 
  (h_Be2C : moles_Be2C = 3) (h_H2O : moles_H2O = 12) : 
  6 = 2 * moles_Be2C :=
by
  sorry

end moles_of_CH4_l606_606647


namespace negation_of_p_l606_606697

theorem negation_of_p (p : Prop) :
  (¬ (∀ (a : ℝ), a ≥ 0 → a^4 + a^2 ≥ 0)) ↔ (∃ (a : ℝ), a ≥ 0 ∧ a^4 + a^2 < 0) := 
by
  sorry

end negation_of_p_l606_606697


namespace correct_statements_BC_l606_606499

theorem correct_statements_BC : 
  (¬ (∃ x : ℝ, x ≥ 1 ∧ x^2 > 1) = ∀ x : ℝ, x ≥ 1 → x^2 ≤ 1) ∧ 
  ((∃ x : ℝ, x = 1 ∧ x^2 + 2*x - 3 = 0) ∧ 
  (¬ ∀ x : ℝ, x ≠ 1 → (x^2 + 2*x - 3 = 0)) ) ∧ 
  ((∀ p q s : Prop, (p → q) ∧ (q → s) → (p → s)) ∧ 
  ¬ ((∀ x : ℝ, mx^2 + mx + 1 ≥ 0) → (0 < m < 4))) :=
by 
  sorry

end correct_statements_BC_l606_606499


namespace largest_prime_factor_divisors_sum_l606_606786

def prime_factors (n : ℕ) : List ℕ := sorry -- Dummy placeholder for prime factorization

theorem largest_prime_factor_divisors_sum :
  let N := (1 + 2 + 2^2) * (1 + 3 + 3^2) * (1 + 5)
  in List.maximum (prime_factors N) = 13 :=
by
  let N := (1 + 2 + 2^2) * (1 + 3 + 3^2) * (1 + 5)
  have h : prime_factors N = [2, 3, 7, 13] := sorry -- Placeholder
  exact List.maximum_eq_some.mp ⟨13, List.mem_cons_self 13 _, sorry⟩

end largest_prime_factor_divisors_sum_l606_606786


namespace count_3digit_numbers_divisible_by_11_l606_606188

theorem count_3digit_numbers_divisible_by_11 :
  let smallest := 100 in
  let largest := 999 in
  -- Determine the smallest and largest 3-digit numbers divisible by 11
  let smallest_divisible_by_11 := ((smallest + 10) / 11) * 11 in
  let largest_divisible_by_11 := (largest / 11) * 11 in
  -- Number of positive 3-digit numbers divisible by 11
  (largest_divisible_by_11 / 11) - (smallest_divisible_by_11 / 11) + 1 = 81 :=
by
  sorry

end count_3digit_numbers_divisible_by_11_l606_606188


namespace correct_propositions_count_l606_606945

def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def f1 (x : ℝ) := 1
def g1 (x : ℝ) := x ^ 3
def H1 (f g : ℝ → ℝ) (x : ℝ) := f x * g x

theorem correct_propositions_count : 
  let p1 := is_even f1,
      p2 := ¬ is_odd g1,
      p3 := ∀ (f g : ℝ → ℝ), is_odd f → is_even g → is_odd (H1 f g),
      p4 := is_even (λ x, f1 (|x|))
  in
  [p1, ¬ p2, p3 f1 g1, p4].count true = 3 :=
by 
  intros;
  sorry

end correct_propositions_count_l606_606945


namespace inequality_proof_l606_606541

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l606_606541


namespace least_integer_greater_than_sqrt_450_l606_606455

-- Define that 21^2 is less than 450
def square_of_21_less_than_450 : Prop := 21^2 < 450

-- Define that 450 is less than 22^2
def square_of_22_greater_than_450 : Prop := 450 < 22^2

-- State the theorem that 22 is the smallest integer greater than sqrt(450)
theorem least_integer_greater_than_sqrt_450 
  (h21 : square_of_21_less_than_450) 
  (h22 : square_of_22_greater_than_450) : 
    ∃ n : ℕ, n = 22 ∧ (√450 : ℝ) < n := 
sorry

end least_integer_greater_than_sqrt_450_l606_606455


namespace find_f_2_l606_606387

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x + y) = f x + f y
axiom f_8 : f 8 = 3

theorem find_f_2 : f 2 = 3 / 4 := 
by sorry

end find_f_2_l606_606387


namespace sum_of_third_sequence_l606_606130

-- The statement of the problem
theorem sum_of_third_sequence :
  ∃ (a q d : ℕ), 
  (a = 1) ∧
  (q + d = 1) ∧
  (q^2 + 2 * d = 2) ∧
  (∑ i in (range 10), a * q^i + i * d = 978) :=
sorry

end sum_of_third_sequence_l606_606130


namespace repetend_five_seventeen_l606_606652

noncomputable def repetend_of_fraction (n : ℕ) (d : ℕ) : ℕ := sorry

theorem repetend_five_seventeen : repetend_of_fraction 5 17 = 294117647058823529 := sorry

end repetend_five_seventeen_l606_606652


namespace num_integral_triangles_perimeter_8_l606_606177

/-- Proving the number of different triangles with integral sides and a perimeter of 8 -/
theorem num_integral_triangles_perimeter_8 : 
  let triangles := {(a, b, c) | a + b + c = 8 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a} in
  finset.card triangles = 8 := 
sorry

end num_integral_triangles_perimeter_8_l606_606177


namespace medicine_dosage_l606_606576

theorem medicine_dosage (weight_kg dose_per_kg parts : ℕ) (h_weight : weight_kg = 30) (h_dose_per_kg : dose_per_kg = 5) (h_parts : parts = 3) :
  ((weight_kg * dose_per_kg) / parts) = 50 :=
by sorry

end medicine_dosage_l606_606576


namespace domain_of_g_l606_606971

noncomputable def g (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_g : ∀ x : ℝ, x > 6^625 → ∃ y : ℝ, g x = y := by
  intros x hx
  have h1 : log 6 x > 1 := sorry
  have h2 : log 5 (log 6 x) > 1 := sorry
  have h3 : log 4 (log 5 (log 6 x)) > 1 := sorry
  have h4 : log 3 (log 4 (log 5 (log 6 x))) > 0 := sorry
  use g x
  exact sorry

end domain_of_g_l606_606971


namespace inequality_condition_l606_606844

theorem inequality_condition {a b c : ℝ} :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (Real.sqrt (a^2 + b^2) < c) :=
by
  sorry

end inequality_condition_l606_606844


namespace repeating_decimals_sum_l606_606071

theorem repeating_decimals_sum :
  let x := 0.6666666 -- 0.\overline{6}
  let y := 0.2222222 -- 0.\overline{2}
  let z := 0.4444444 -- 0.\overline{4}
  (x + y - z) = 4 / 9 := 
by
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  calc
    -- Calculate (x + y - z)
    (x + y - z) = (2 / 3 + 2 / 9 - 4 / 9) : by sorry
                ... = 4 / 9 : by sorry


end repeating_decimals_sum_l606_606071


namespace cube_probability_l606_606573

def prob_same_color_vertical_faces : ℕ := sorry

theorem cube_probability :
  prob_same_color_vertical_faces = 1 / 27 := 
sorry

end cube_probability_l606_606573


namespace differential_equation_solution_l606_606376

theorem differential_equation_solution (x y : ℝ) (hy : y = (1 + x) / (1 - x)) :
  has_deriv_at (λ x, (1 + x) / (1 - x)) ((1 + y^2) / (1 + x^2)) x :=
sorry

end differential_equation_solution_l606_606376


namespace ferry_time_difference_l606_606890

def ferry_p_speed : ℝ := 6
def ferry_p_time : ℝ := 3
def ferry_q_time (ferry_q_speed : ℝ) : ℝ := (3 * ferry_p_speed * ferry_p_time) / ferry_q_speed
def ferry_r_speed (ferry_q_speed : ℝ) : ℝ := ferry_q_speed / 2
def ferry_r_time : ℝ := 5

theorem ferry_time_difference :
  ∃ (ferry_q_speed : ℝ), ferry_q_speed = ferry_p_speed + 3 ∧
  (ferry_r_time - ferry_p_time = 2) :=
begin
  sorry
end

end ferry_time_difference_l606_606890


namespace additional_people_needed_to_mow_lawn_l606_606628

noncomputable def people_needed (initial_people : ℕ) (initial_hours : ℕ) (target_hours : ℕ) : ℕ :=
  let k := initial_people * initial_hours
  let target_people := (k + target_hours - 1) / target_hours -- Round up by adding (target_hours - 1) before integer division
  target_people - initial_people

theorem additional_people_needed_to_mow_lawn 
  (initial_people : ℕ := 8) (initial_hours : ℕ := 5) (target_hours : ℕ := 3) : people_needed initial_people initial_hours target_hours = 6 :=
by
  rfl

end additional_people_needed_to_mow_lawn_l606_606628


namespace find_z_l606_606246

theorem find_z (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : z = 3 - 4 * complex.I :=
by {
  sorry
}

end find_z_l606_606246


namespace percentage_of_error_in_area_l606_606009

-- Define the conditions
variable (x : ℝ)
def measured_side := 1.12 * x
def actual_area := x^2
def erroneous_area := (measured_side x)^2
def error_area := erroneous_area x - actual_area x
def percentage_error := (error_area x / actual_area x) * 100

-- The theorem to be proven
theorem percentage_of_error_in_area : percentage_error x = 25.44 := by
  sorry

end percentage_of_error_in_area_l606_606009


namespace common_tangent_line_unique_m_l606_606165

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * (Real.log (x + 1))

def g (x : ℝ) : ℝ := x / (x + 1)

noncomputable def F (m : ℝ) (x : ℝ) : ℝ := f m x - g x

theorem common_tangent_line_unique_m :
  ∃ (m : ℝ), m > 0 ∧
    (∀ (a b : ℝ), a > -1 ∧ b > -1 → (m = 1) ∧
      (f m a = g b) ∧
      (differentiable_at ℝ (f m) a) ∧ 
      (differentiable_at ℝ g b) ∧
      (deriv (f m) a = deriv g b)) :=
begin
  existsi (1 : ℝ),
  split,
  { linarith },
  intros a b hab,
  split,
  { linarith },
  { sorry },
end

end common_tangent_line_unique_m_l606_606165


namespace complex_point_coordinates_l606_606744

theorem complex_point_coordinates : 
  (⟦(1 - 2 * Complex.i) / (2 + Complex.i) + 2 * Complex.i⟧.re, 
   ⟦(1 - 2 * Complex.i) / (2 + Complex.i) + 2 * Complex.i⟧.im) = (2 / 5, 1) :=
by sorry

end complex_point_coordinates_l606_606744


namespace find_tangent_values_l606_606994

theorem find_tangent_values (n : ℝ) : 
  (Real.tan (n * real.pi / 180) = Real.tan (675 * real.pi / 180) ∧ -180 < n ∧ n < 180) ↔ (n = 135 ∨ n = -45) := 
sorry

end find_tangent_values_l606_606994


namespace sales_tax_percentage_l606_606918

theorem sales_tax_percentage (total_amount : ℝ) (tip_percentage : ℝ) (food_price : ℝ) (tax_percentage : ℝ) : 
  total_amount = 158.40 ∧ tip_percentage = 0.20 ∧ food_price = 120 → tax_percentage = 0.10 :=
by
  intros h
  sorry

end sales_tax_percentage_l606_606918


namespace area_of_triangle_ABC_l606_606805

-- Declaring the points A, B, C, D, E, F, and P in a 2D plane
variables (A B C D E F P : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited P]

-- Lengths of line segments
variables (BP PD PC PE AF : ℝ)

-- Consider the given conditions
axiom BP_len : BP = 6
axiom PD_len : PD = 6
axiom PC_len : PC = 9
axiom PE_len : PE = 3
axiom AF_len : AF = 20

-- Concurrent lines at point P
axiom AF_BD_CE_concur : ∃ (P : Type), P ∈ (line_through A F) ∧ P ∈ (line_through B D) ∧ P ∈ (line_through C E)

-- Definition of area (could use Lean definition or third-party library)
variable (S : ℝ) -- This represents the area of the triangle ABC

-- This theorem states that given the conditions the area of the triangle ABC is 108.
theorem area_of_triangle_ABC : S = 108 :=
sorry

end area_of_triangle_ABC_l606_606805


namespace inequality_for_n_greater_than_1_l606_606835

noncomputable def inequality (n : ℕ) : Prop :=
  (3 * n + 1) / (2 * n + 2) < 1 / n * (Finset.sum (Finset.range n) (λ r, (r / n : ℝ) ^ n)) ∧ 
  1 / n * (Finset.sum (Finset.range n) (λ r, (r / n : ℝ) ^ n)) < 2

theorem inequality_for_n_greater_than_1 (n : ℕ) (h : 1 < n) : inequality n :=
by 
  sorry

end inequality_for_n_greater_than_1_l606_606835


namespace least_integer_greater_than_sqrt_450_l606_606452

-- Define that 21^2 is less than 450
def square_of_21_less_than_450 : Prop := 21^2 < 450

-- Define that 450 is less than 22^2
def square_of_22_greater_than_450 : Prop := 450 < 22^2

-- State the theorem that 22 is the smallest integer greater than sqrt(450)
theorem least_integer_greater_than_sqrt_450 
  (h21 : square_of_21_less_than_450) 
  (h22 : square_of_22_greater_than_450) : 
    ∃ n : ℕ, n = 22 ∧ (√450 : ℝ) < n := 
sorry

end least_integer_greater_than_sqrt_450_l606_606452


namespace solve_complex_equation_l606_606213

theorem solve_complex_equation (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : 
  z = 3 - 4 * complex.I :=
sorry

end solve_complex_equation_l606_606213


namespace average_students_difference_l606_606596

-- Definition of the problem conditions
def total_students : ℕ := 120
def total_teachers : ℕ := 4
def class_sizes : List ℕ := [60, 30, 20, 10]

-- Definition of t and s
def t : ℚ := (class_sizes.sum * (1 / total_teachers))
def s : ℚ := class_sizes.map (λ n => n * (n / total_students)).sum

-- Statement to prove
theorem average_students_difference :
  t - s = -11.66 := by
  sorry

end average_students_difference_l606_606596


namespace repeating_decimal_sum_l606_606084

theorem repeating_decimal_sum : (0.\overline{6} + 0.\overline{2} - 0.\overline{4} : ℚ) = (4 / 9 : ℚ) :=
by
  sorry

end repeating_decimal_sum_l606_606084


namespace maximum_abs_value_l606_606779

-- Definitions of variables and the problem conditions
variables (x y z : ℝ)

-- Define the inequalities as conditions
def conditions : Prop :=
  | x + 2 * y - 3 * z | ≤ 6 ∧
  | x - 2 * y + 3 * z | ≤ 6 ∧
  | x - 2 * y - 3 * z | ≤ 6 ∧
  | x + 2 * y + 3 * z | ≤ 6

-- Assert that the maximum value of |x| + |y| + |z| is 6
theorem maximum_abs_value (h : conditions x y z) :
  |x| + |y| + |z| ≤ 6 :=
sorry

end maximum_abs_value_l606_606779


namespace expected_value_variance_of_X_l606_606859

open Probability.Theory

noncomputable def expected_value (X : ℝ → ℝ) (p : ℝ) : ℝ :=
  0 * (1 - p) + 1 * p

noncomputable def variance (X : ℝ → ℝ) (p : ℝ) : ℝ :=
  (0^2 * (1 - p) + 1^2 * p) - (expected_value X p)^2

-- Lean statement for the problem.
theorem expected_value_variance_of_X (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  expected_value (λ x => x) p = p ∧ variance (λ x => x) p = p * (1 - p) :=
by
  sorry

end expected_value_variance_of_X_l606_606859


namespace solve_for_z_l606_606220

variable (z : ℂ)

theorem solve_for_z (h : (complex.I * z = 4 + 3 * complex.I)) : z = 3 - 4 * complex.I := 
sorry

end solve_for_z_l606_606220


namespace intersection_of_M_and_N_l606_606171

def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | x < 1}

theorem intersection_of_M_and_N : (M ∩ N = {x : ℝ | -1 < x ∧ x < 1}) :=
by
  sorry

end intersection_of_M_and_N_l606_606171


namespace transistors_in_2010_l606_606942

-- Define the conditions
def initial_transistors : ℕ := 2500000
def doubling_period_months : ℕ := 18
def doublings (months : ℕ) : ℕ := months / doubling_period_months

-- Define the main theorem
theorem transistors_in_2010 : 
  let start_year := 1995 in
  let end_year := 2010 in
  let months_in_year := 12 in
  initial_transistors * 2 ^ (doublings ((end_year - start_year) * months_in_year)) = 2560000000 :=
by
  -- proof would go here
  sorry

end transistors_in_2010_l606_606942


namespace complex_number_identity_l606_606666

theorem complex_number_identity {a b : ℝ} 
  (h : (1 + complex.i) + (2 - 3 * complex.i) = a + b * complex.i) : 
  a = 3 ∧ b = -2 :=
by
  sorry

end complex_number_identity_l606_606666


namespace cannot_all_numbers_in_geometric_progressions_l606_606026

theorem cannot_all_numbers_in_geometric_progressions 
: ¬ (∃ (G : fin 12 → set ℕ), ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 100 → ∃ (g : fin 12), n ∈ G g ∧ ∀ (g : fin 12), ∃ (primes : list ℕ), prime_sublist primes ∧ 2 ≤ t ∧ list.tiny allprimes 100 → listing g allnat)
:= sorry

end cannot_all_numbers_in_geometric_progressions_l606_606026


namespace Zn_is_one_l606_606332

noncomputable def Zn (n : ℕ) (z : ℂ) : ℂ :=
  ∏ k in finset.range n, (z ^ (2^k) + (1 / z ^ (2^k)) - 1)

theorem Zn_is_one {n : ℕ} (hn : n % 2 = 1) {z : ℂ}
  (hz : z^(2^n - 1) = 1) : Zn n z = 1 := by
  sorry

end Zn_is_one_l606_606332


namespace minimum_h_17_l606_606034

noncomputable def h : ℕ → ℝ := sorry  -- Define the function h

theorem minimum_h_17 :
  (∀ x y : ℕ, (x > 0 ∧ y > 0) → h(x) + h(y) > y^2) ∧
  (∀ n : ℕ, ∑ i in (finset.range 26) \ {0}, h i = n → n ≥ 4250) →
  h 17 = 205 :=
sorry

end minimum_h_17_l606_606034


namespace adults_riding_bicycles_l606_606603

theorem adults_riding_bicycles (A : ℕ) (H1 : 15 * 3 + 2 * A = 57) : A = 6 :=
by
  sorry

end adults_riding_bicycles_l606_606603


namespace plane_equation_parallel_to_Oz_l606_606116

theorem plane_equation_parallel_to_Oz (A B D : ℝ)
  (h1 : A * 1 + B * 0 + D = 0)
  (h2 : A * (-2) + B * 1 + D = 0)
  (h3 : ∀ z : ℝ, exists c : ℝ, A * z + B * c + D = 0):
  A = 1 ∧ B = 3 ∧ D = -1 :=
  by
  sorry

end plane_equation_parallel_to_Oz_l606_606116


namespace distance_from_origin_to_line_l606_606850

theorem distance_from_origin_to_line : 
  let a := 1
  let b := 2
  let c := -5
  let x0 := 0
  let y0 := 0
  let distance := (|a * x0 + b * y0 + c|) / (Real.sqrt (a^2 + b^2))
  distance = Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_line_l606_606850


namespace triangle_area_l606_606424

theorem triangle_area (A B C : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (9, 2)) (hC : C = (5, 10)) :
  let base := |B.1 - A.1|,
      height := |C.2 - A.2|,
      area := (1 / 2) * base * height
  in area = 28 :=
by
  have base_eq : |B.1 - A.1| = 7 := by linarith [hA, hB]
  have height_eq : |C.2 - A.2| = 8 := by linarith [hA, hC]
  have area_eq : (1 / 2) * base_eq * height_eq = 28 := by norm_num
  exact area_eq

end triangle_area_l606_606424


namespace perpendicular_collinear_l606_606015

theorem perpendicular_collinear {O O1 O2 : Point} {M N S T : Point}
  (h1 : Circle O1 intersects Circle O at M and N)
  (h2 : Circle O1 tangent to Circle O at S)
  (h3 : Circle O2 tangent to Circle O at T):
  (OM ⊥ MN) ↔ collinear {S, N, T} :=
sorry

end perpendicular_collinear_l606_606015


namespace equilateral_triangle_volume_ratio_l606_606007

theorem equilateral_triangle_volume_ratio {R : ℝ} (hR : 0 < R) :
  let V := (4 / 3) * Real.pi * R^3,
      MA := (R * Real.sqrt 3) / 2,
      BM := (3 / 2) * R,
      v := (1 / 3) * Real.pi * (MA ^ 2) * BM
  in (v / V) ≈ 0.28 :=
by
  assume V := (4 / 3) * Real.pi * R^3
  assume MA := (R * Real.sqrt 3) / 2
  assume BM := (3 / 2) * R
  assume v := (1 / 3) * Real.pi * (MA ^ 2) * BM
  sorry

end equilateral_triangle_volume_ratio_l606_606007


namespace regular_polygon_perimeter_l606_606929

theorem regular_polygon_perimeter (side_length : ℝ) (interior_angle : ℝ) 
  (h1 : side_length = 2) (h2 : interior_angle = 135) : 
  (P : ℝ) := 
  side_length * 8 = 16 :=
sorry

end regular_polygon_perimeter_l606_606929


namespace least_integer_greater_than_sqrt_450_l606_606454

-- Define that 21^2 is less than 450
def square_of_21_less_than_450 : Prop := 21^2 < 450

-- Define that 450 is less than 22^2
def square_of_22_greater_than_450 : Prop := 450 < 22^2

-- State the theorem that 22 is the smallest integer greater than sqrt(450)
theorem least_integer_greater_than_sqrt_450 
  (h21 : square_of_21_less_than_450) 
  (h22 : square_of_22_greater_than_450) : 
    ∃ n : ℕ, n = 22 ∧ (√450 : ℝ) < n := 
sorry

end least_integer_greater_than_sqrt_450_l606_606454


namespace Janka_bottle_caps_l606_606763

theorem Janka_bottle_caps (n : ℕ) :
  (∃ k1 : ℕ, n = 3 * k1) ∧ (∃ k2 : ℕ, n = 4 * k2) ↔ n = 12 ∨ n = 24 :=
by
  sorry

end Janka_bottle_caps_l606_606763


namespace sum_f_eq_neg_one_l606_606147

noncomputable def f : ℝ → ℝ := λ x, sorry

axiom even_f : ∀ x : ℝ, f(x) = f(-x)
axiom odd_f_shift_2 : ∀ x : ℝ, f(x + 2) = -f(-x + 2)
axiom f_zero : f(0) = 1

theorem sum_f_eq_neg_one : (∑ i in (finset.range 2024).filter (λ i, i > 0), f i) = -1 :=
by
  sorry

end sum_f_eq_neg_one_l606_606147


namespace inequality_proof_l606_606547

variable {x y z : ℝ}

theorem inequality_proof (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * real.sqrt(z * x)) + 
  y^3 / (y^3 + 2 * z^2 * real.sqrt(x * y)) + 
  z^3 / (z^3 + 2 * x^2 * real.sqrt(y * z)) >= 1) :=
sorry

end inequality_proof_l606_606547


namespace max_sum_nonnegative_largest_k_such_that_inequality_holds_l606_606557

theorem max_sum_nonnegative (a b c d : ℝ) (h : a + b + c + d = 0) : 
  ∑ (pair : (ℝ × ℝ)), max pair.1 pair.2 ∈ [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)] ≥ 0 :=
by
  sorry

theorem largest_k_such_that_inequality_holds (a b c d : ℝ) (h: a + b + c + d = 0) : 
  ∃ k : ℕ, ∀ idxs : set (ℕ × (ℝ × ℝ)), idxs.card ≤ k → 
    ∑ (pair : ℝ × ℝ), if pair ∈ idxs then min pair.1 pair.2 else max pair.1 pair.2 ≥ 0 ∧ k = 2 :=
by
  sorry

end max_sum_nonnegative_largest_k_such_that_inequality_holds_l606_606557


namespace total_turnips_l606_606307

theorem total_turnips (keith_turnips : ℕ) (alyssa_turnips : ℕ) 
  (h_keith : keith_turnips = 6) (h_alyssa : alyssa_turnips = 9) : 
  keith_turnips + alyssa_turnips = 15 :=
by
  rw [h_keith, h_alyssa]
  exact rfl

end total_turnips_l606_606307


namespace number_of_jerseys_bought_l606_606764

-- Define the given constants
def initial_money : ℕ := 50
def cost_per_jersey : ℕ := 2
def cost_basketball : ℕ := 18
def cost_shorts : ℕ := 8
def money_left : ℕ := 14

-- Define the theorem to prove the number of jerseys Jeremy bought.
theorem number_of_jerseys_bought :
  (initial_money - money_left) = (cost_basketball + cost_shorts + 5 * cost_per_jersey) :=
by
  sorry

end number_of_jerseys_bought_l606_606764


namespace repeating_decimals_sum_l606_606073

theorem repeating_decimals_sum :
  let x := 0.6666666 -- 0.\overline{6}
  let y := 0.2222222 -- 0.\overline{2}
  let z := 0.4444444 -- 0.\overline{4}
  (x + y - z) = 4 / 9 := 
by
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  calc
    -- Calculate (x + y - z)
    (x + y - z) = (2 / 3 + 2 / 9 - 4 / 9) : by sorry
                ... = 4 / 9 : by sorry


end repeating_decimals_sum_l606_606073


namespace cost_price_of_computer_table_l606_606398

theorem cost_price_of_computer_table
  (C : ℝ) 
  (S : ℝ := 1.20 * C)
  (S_eq : S = 8600) : 
  C = 7166.67 :=
by
  sorry

end cost_price_of_computer_table_l606_606398


namespace solve_system_l606_606665

theorem solve_system (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = 7) : x + y = 5 :=
by
  sorry

end solve_system_l606_606665


namespace range_of_m_l606_606261

theorem range_of_m (m : ℝ) :
  (∃ x y : ℤ, (x ≠ y) ∧ (x ≥ m ∧ y ≥ m) ∧ (3 - 2 * x ≥ 0) ∧ (3 - 2 * y ≥ 0)) ↔ (-1 < m ∧ m ≤ 0) :=
by
  sorry

end range_of_m_l606_606261


namespace complete_square_form_l606_606724

theorem complete_square_form {a h k : ℝ} :
  ∀ x, (x^2 - 5 * x) = a * (x - h)^2 + k → k = -25 / 4 :=
by
  intro x
  intro h_eq
  sorry

end complete_square_form_l606_606724


namespace num_rectangles_in_grid_l606_606271

theorem num_rectangles_in_grid : 
  let width := 35
  let height := 44
  ∃ n, n = 87 ∧ 
  ∀ x y, (1 ≤ x ∧ x ≤ width) ∧ (1 ≤ y ∧ y ≤ height) → 
    n = (x * (x + 1) / 2) * (y * (y + 1) / 2) := 
by
  sorry

end num_rectangles_in_grid_l606_606271


namespace inequality_proof_l606_606511

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l606_606511


namespace problem1_task_one_1_task_one_2_task_one_3_task_two_l606_606607

noncomputable def expr : ℝ :=
  -202 * (3:ℝ) ^ 0 + ((-1/2:ℝ) ^ (-2)) * abs (-1 + 3) + real.sqrt 12

theorem problem1 : expr = -194 + 2 * real.sqrt 3 :=
  sorry

theorem task_one_1 : transformation_is_factorization :=
  sorry

theorem task_one_2 : (step_finding_common_denominator_based_on_basic_properties = 3) :=
  sorry

theorem task_one_3 : (error_step_is_4 ∧ reason_is_sign_change_error) :=
  sorry

theorem task_two : ( (x - 1)/(x + 1) - (x^2 - 1)/(x^2 - 2*x + 1) ) / (x/(x - 1)) = -(4/(x + 1)) :=
  sorry

end problem1_task_one_1_task_one_2_task_one_3_task_two_l606_606607


namespace axes_positioning_l606_606592

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem axes_positioning (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c < 0) :
  ∃ x_vertex y_intercept, x_vertex < 0 ∧ y_intercept < 0 ∧ (∀ x, f a b c x > f a b c x) :=
by
  sorry

end axes_positioning_l606_606592


namespace least_integer_greater_than_sqrt_450_l606_606472

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606472


namespace least_integer_greater_than_sqrt_450_l606_606469

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606469


namespace digit_b_divisible_by_5_l606_606423

theorem digit_b_divisible_by_5 (B : ℕ) (h : B = 0 ∨ B = 5) : 
  (∃ n : ℕ, (947 * 10 + B) = 5 * n) ↔ (B = 0 ∨ B = 5) :=
by {
  sorry
}

end digit_b_divisible_by_5_l606_606423


namespace inequality_xyz_l606_606523

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x))) +
  (y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y))) +
  (z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 :=
by sorry

end inequality_xyz_l606_606523


namespace repetend_of_decimal_expansion_l606_606650

theorem repetend_of_decimal_expansion :
  ∃ (r : ℕ), decimal_repetend (5 / 17) = some r ∧ r = 294117647058823529 :=
by
  sorry

end repetend_of_decimal_expansion_l606_606650


namespace arithmetic_sequence_S10_l606_606316

theorem arithmetic_sequence_S10
  (a : ℕ → ℝ)
  (d : ℝ)
  (h₀ : d = -2)
  (h₁ : a 4 + 2 * a 6 = 4)
  (h₂ : ∀ n : ℕ, a n = a 1 + (n - 1) * d) :
  ∑ i in finset.range 10, a i = 10 := by
  sorry

end arithmetic_sequence_S10_l606_606316


namespace cube_color_probability_l606_606979

-- Definitions related to the problem conditions
def face_color := {red, blue, green} -- Set of colors

-- Definition of a cube where each face is painted
structure Cube :=
(front : face_color)
(back : face_color)
(left : face_color)
(right : face_color)
(top : face_color)
(bottom : face_color)

-- Definition of the probability calculation
def cube_probability (c : Cube) : ℚ :=
  if (c.front = c.back) ∧ (c.front = c.left) ∧ (c.front = c.right) then
    if (c.top = c.bottom) then
      1 / 3 ^ 6
    else
      1 / 3 ^ 5
  else
    0

-- Main theorem to prove
theorem cube_color_probability : (75 / 729 : ℚ) = (25 / 243 : ℚ) :=
by sorry

end cube_color_probability_l606_606979


namespace smallest_number_conditions_l606_606880

theorem smallest_number_conditions :
  ∃ m : ℕ, (∀ k ∈ [3, 4, 5, 6, 7], m % k = 2) ∧ (m % 8 = 0) ∧ ( ∀ n : ℕ, (∀ k ∈ [3, 4, 5, 6, 7], n % k = 2) ∧ (n % 8 = 0) → m ≤ n ) :=
sorry

end smallest_number_conditions_l606_606880


namespace least_integer_greater_than_sqrt_450_l606_606435

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606435


namespace solve_complex_equation_l606_606216

theorem solve_complex_equation (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : 
  z = 3 - 4 * complex.I :=
sorry

end solve_complex_equation_l606_606216


namespace processing_plant_growth_eq_l606_606914

-- Definition of the conditions given in the problem
def initial_amount : ℝ := 10
def november_amount : ℝ := 13
def growth_rate (x : ℝ) : ℝ := initial_amount * (1 + x)^2

-- Lean theorem statement to prove the equation
theorem processing_plant_growth_eq (x : ℝ) : 
  growth_rate x = november_amount ↔ initial_amount * (1 + x)^2 = 13 := 
by
  sorry

end processing_plant_growth_eq_l606_606914


namespace inequality_abc_l606_606326

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 :=
by
  sorry

end inequality_abc_l606_606326


namespace defective_units_shipped_percentage_l606_606749

theorem defective_units_shipped_percentage :
  let units_produced := 100
  let typeA_defective := 0.07 * units_produced
  let typeB_defective := 0.08 * units_produced
  let typeA_shipped := 0.03 * typeA_defective
  let typeB_shipped := 0.06 * typeB_defective
  let total_shipped := typeA_shipped + typeB_shipped
  let percentage_shipped := total_shipped / units_produced * 100
  percentage_shipped = 1 :=
by
  sorry

end defective_units_shipped_percentage_l606_606749


namespace inequality_proof_l606_606546

variable {x y z : ℝ}

theorem inequality_proof (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * real.sqrt(z * x)) + 
  y^3 / (y^3 + 2 * z^2 * real.sqrt(x * y)) + 
  z^3 / (z^3 + 2 * x^2 * real.sqrt(y * z)) >= 1) :=
sorry

end inequality_proof_l606_606546


namespace inequality_proof_l606_606526

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606526


namespace correct_operation_l606_606883

theorem correct_operation :
(∃ (x y : Real), sqrt 4 + sqrt 9 = sqrt 13 ∨ 5 * x^3 * 3 * x^5 = 15 * x^15 ∨ 3 / sqrt 6 = sqrt 6 / 2 ∨ 3 * y^2 * x + 2 * y * x^2 = 5 * y^2 * x^2) ↔
  (∃ (x y : Real), False ∨ False ∨ (3 / sqrt 6 = sqrt 6 / 2) ∨ False) :=
by
  sorry

end correct_operation_l606_606883


namespace solve_for_z_l606_606243

variable {z : ℂ}

theorem solve_for_z (h : complex.I * z = 4 + 3*complex.I) : z = 3 - 4*complex.I :=
sorry

end solve_for_z_l606_606243


namespace seed_grow_prob_l606_606392

theorem seed_grow_prob (P_G P_S_given_G : ℝ) (hP_G : P_G = 0.9) (hP_S_given_G : P_S_given_G = 0.8) :
  P_G * P_S_given_G = 0.72 :=
by
  rw [hP_G, hP_S_given_G]
  norm_num

end seed_grow_prob_l606_606392


namespace person_reaches_before_bus_l606_606593

theorem person_reaches_before_bus (dist : ℝ) (speed1 speed2 : ℝ) (miss_time_minutes : ℝ) :
  dist = 2.2 → speed1 = 3 → speed2 = 6 → miss_time_minutes = 12 →
  ((60 : ℝ) * (dist/speed1) - miss_time_minutes) - ((60 : ℝ) * (dist/speed2)) = 10 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end person_reaches_before_bus_l606_606593


namespace solve_equation_l606_606989

theorem solve_equation (x : ℝ) :
  (8 / (real.sqrt (x - 12) - 10) +
   2 / (real.sqrt (x - 12) - 5) +
   10 / (real.sqrt (x - 12) + 5) +
   16 / (real.sqrt (x - 12) + 10) = 0)
  ↔ (x = 208 / 9 ∨ x = 62) :=
sorry

end solve_equation_l606_606989


namespace defective_units_shipped_percentage_l606_606750

theorem defective_units_shipped_percentage :
  let units_produced := 100
  let typeA_defective := 0.07 * units_produced
  let typeB_defective := 0.08 * units_produced
  let typeA_shipped := 0.03 * typeA_defective
  let typeB_shipped := 0.06 * typeB_defective
  let total_shipped := typeA_shipped + typeB_shipped
  let percentage_shipped := total_shipped / units_produced * 100
  percentage_shipped = 1 :=
by
  sorry

end defective_units_shipped_percentage_l606_606750


namespace find_quadratic_function_l606_606131

noncomputable def quadratic_function (a : ℝ) := a * (x : ℝ) * (x + 2)

theorem find_quadratic_function : 
  (∃ a : ℝ, ∀ x : ℝ, quadratic_function a x = a * x * (x + 2) ∧ 
                      quadratic_function a 0 = 0 ∧ 
                      quadratic_function a (-2) = 0 ∧ 
                      quadratic_function a (-1) = -1) →
  quadratic_function 1 = x^2 + 2 * x :=
sorry

end find_quadratic_function_l606_606131


namespace least_int_gt_sqrt_450_l606_606429

theorem least_int_gt_sqrt_450 : ∃ n : ℕ, n > nat.sqrt 450 ∧ (∀ m : ℕ, m > nat.sqrt 450 → n ≤ m) :=
by
  have h₁ : 20^2 = 400 := by norm_num
  have h₂ : 21^2 = 441 := by norm_num
  have h₃ : 22^2 = 484 := by norm_num
  have sqrt_bounds : 21 < real.sqrt 450 ∧ real.sqrt 450 < 22 :=
    by sorry
  use 22
  split
  · sorry
  · sorry

end least_int_gt_sqrt_450_l606_606429


namespace find_z_l606_606251

theorem find_z (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : z = 3 - 4 * complex.I :=
by {
  sorry
}

end find_z_l606_606251


namespace find_f_of_3_l606_606143

def f (x : ℝ) : ℝ := 
  if x < 0 then x * (1 - x) 
  else -f (-x)

/-- Given the conditions that f(x) is odd and for x in (-∞, 0), f(x) = x * (1-x), 
    prove that f(3) = 12. -/
theorem find_f_of_3 : f 3 = 12 := 
  sorry

end find_f_of_3_l606_606143


namespace hiking_supplies_l606_606583

theorem hiking_supplies (hours_per_day : ℕ) (days : ℕ) (rate_mph : ℝ) 
    (supply_per_mile : ℝ) (resupply_rate : ℝ)
    (initial_pack_weight : ℝ) : 
    hours_per_day = 8 → days = 5 → rate_mph = 2.5 → 
    supply_per_mile = 0.5 → resupply_rate = 0.25 → 
    initial_pack_weight = (40 : ℝ) :=
by
  intros hpd hd rm spm rr
  sorry

end hiking_supplies_l606_606583


namespace union_of_sets_l606_606685

theorem union_of_sets (A : Set ℤ) (B : Set ℤ) (hA : A = {0, 1, 2}) (hB : B = {-1, 0}) :
  A ∪ B = {-1, 0, 1, 2} :=
by
  rw [hA, hB]
  apply Set.ext
  intro x
  simp
  tauto

end union_of_sets_l606_606685


namespace odd_and_decreasing_l606_606946

-- Define the function
def f (x : ℝ) : ℝ := real.log ((1 - x) / (1 + x))

-- State the properties we need to prove
theorem odd_and_decreasing : ∀ x : ℝ, -1 < x ∧ x < 1 → 
  (f(-x) = -f(x)) ∧ (∀ y z : ℝ, -1 < y ∧ y < 1 ∧ -1 < z ∧ z < 1 ∧ y < z → f(y) > f(z)) :=
by
  sorry

end odd_and_decreasing_l606_606946


namespace medicine_dose_per_part_l606_606575

-- Define the given conditions
def kg_weight : ℕ := 30
def ml_per_kg : ℕ := 5
def parts : ℕ := 3

-- The theorem statement
theorem medicine_dose_per_part : 
  (kg_weight * ml_per_kg) / parts = 50 :=
by
  sorry

end medicine_dose_per_part_l606_606575


namespace limit_of_f_at_neg1_l606_606899

noncomputable def f (x : ℝ) : ℝ := (x^3 - 2*x - 1) / (x^4 + 2*x + 1)

theorem limit_of_f_at_neg1 : 
  tendsto f (𝓝 (-1)) (𝓝 (-1/2)) :=
by
  sorry

end limit_of_f_at_neg1_l606_606899


namespace sum_of_decimals_l606_606606

theorem sum_of_decimals : 5.47 + 2.58 + 1.95 = 10.00 := by
  sorry

end sum_of_decimals_l606_606606


namespace repeating_decimal_sum_l606_606066

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem repeating_decimal_sum : repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l606_606066


namespace midpoints_sum_of_integers_l606_606553

theorem midpoints_sum_of_integers {A B C D E F : Point} 
  (convex_quad : convex_quadrilateral ABCD)
  (not_parallel_AB_CD : ¬parallel AB CD)
  (midpoint_E : E = midpoint A D)
  (midpoint_F : F = midpoint B C)
  (length_CD : |CD| = 12)
  (length_AB : |AB| = 22) :
  (∑ x in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, x) = 121 :=
sorry

end midpoints_sum_of_integers_l606_606553


namespace ways_to_replace_asterisks_l606_606295

theorem ways_to_replace_asterisks :
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8}
  let n := 10 
  let base_digits := [2, 0, 1, 6, 0]
  ∃ d1 d2 d3 d4 d5 : ℕ, d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ d5 ∈ digits ∧
    let full_number := [2, d1, 0, d2, 1, d3, 6, d4, 0, d5]
    (d5 % 2 = 0) ∧ ((base_digits.sum + d1 + d2 + d3 + d4 + d5) % 9 = 0) ∧
    (5 * 9 * 9 * 9 * 1 = 3645) :=
  begin
    sorry
  end

end ways_to_replace_asterisks_l606_606295


namespace least_integer_greater_than_sqrt_450_l606_606480

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, (n > real.sqrt 450) ∧ ∀ m : ℤ, (m > real.sqrt  450) → n ≤ m :=
by
  apply exists.intro 22
  sorry

end least_integer_greater_than_sqrt_450_l606_606480


namespace sum_cot_identity_l606_606370

open Real

theorem sum_cot_identity (n : ℕ) (x : ℝ)
  (h1 : 0 < n)
  (h2 : ∀ k : ℕ, k ≤ n → x ≠ (k * π) / 2^k) :
  ∑ i in Finset.range n, 1 / sin (2^(i + 1) * x) = cot x - cot (2^n * x) :=
by
  sorry

end sum_cot_identity_l606_606370


namespace least_integer_greater_than_sqrt_450_l606_606457

-- Define that 21^2 is less than 450
def square_of_21_less_than_450 : Prop := 21^2 < 450

-- Define that 450 is less than 22^2
def square_of_22_greater_than_450 : Prop := 450 < 22^2

-- State the theorem that 22 is the smallest integer greater than sqrt(450)
theorem least_integer_greater_than_sqrt_450 
  (h21 : square_of_21_less_than_450) 
  (h22 : square_of_22_greater_than_450) : 
    ∃ n : ℕ, n = 22 ∧ (√450 : ℝ) < n := 
sorry

end least_integer_greater_than_sqrt_450_l606_606457


namespace eval_expr_eq_zero_l606_606986

theorem eval_expr_eq_zero :
  (\lceil (7 : ℚ) / 3 \rceil + ⌊ - (7 : ℚ) / 3 ⌋) ^ 2 = 0 := by
  sorry

end eval_expr_eq_zero_l606_606986


namespace cos_seven_pi_over_four_l606_606641

theorem cos_seven_pi_over_four :
  Real.cos (7 * Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_seven_pi_over_four_l606_606641


namespace simplest_square_root_l606_606950

theorem simplest_square_root : 
  let options := [Real.sqrt 20, Real.sqrt 2, Real.sqrt (1 / 2), Real.sqrt 0.2] in
  ∃ x ∈ options, x = Real.sqrt 2 ∧ (∀ y ∈ options, y ≠ Real.sqrt 2 → ¬(Real.sqrt y).simpler_than (Real.sqrt 2)) :=
by
  let options := [Real.sqrt 20, Real.sqrt 2, Real.sqrt (1 / 2), Real.sqrt 0.2]
  have h_sqrt_2_in_options : Real.sqrt 2 ∈ options := by simp [options]
  use Real.sqrt 2
  constructor
  . exact h_sqrt_2_in_options
  . intro y hy_ne_sqrt_2 hy_options
    sorry

end simplest_square_root_l606_606950


namespace find_S_coordinates_l606_606408

variables (P Q R S : ℝ × ℝ × ℝ)

def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

def length (A B : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2)

theorem find_S_coordinates (hP : P = (2, 0, 1))
                           (hQ : Q = (0, 3, -2))
                           (hR : R = (-2, 2, 1))
                           (h_diagonals_eq : length P R = length Q S)
                           (h_midpoints_eq : midpoint P R = midpoint Q S) :
  S = (0, -1, 4) :=
sorry

end find_S_coordinates_l606_606408


namespace part1_part2_l606_606698

-- Define the quadratic function f
def f (x : ℝ) (a b : ℝ) := x^2 + a * x + b + 1

-- Define the given condition
def condition1 (b : ℝ) (x : ℝ) := f x (-2) b - (2 * b - 1) * x + b^2 < 1

-- Define the solution interval condition
def condition2 (b : ℝ) := ∀ x : ℝ, x ∈ (b, b + 1) ↔ condition1 b x

-- Formalize the proof problem as Lean theorems
theorem part1 (b : ℝ) (hb : b ≠ 0) :  ∃ a : ℝ, condition2 b → a = -2 :=
sorry

noncomputable def g (x : ℝ) (b : ℝ) := (f x (-2) b) / (x - 1)

noncomputable def phi (x : ℝ) (k b : ℝ) := g x b - k * Real.log (x - 1)

theorem part2 (b k : ℝ) (hb : b ≠ 0) :
  (∃ x : ℝ, x > 1 ∧ ∃ x0 : ℝ, x = x0 ∧ 
    (phi x0 k b - sorry = 0)) → -- This should be more specific on the derivative and extreme point conditions
    ((b > 0 → true) ∧ (b < 0 → k > 2 * Real.sqrt (-b))) :=
sorry

end part1_part2_l606_606698


namespace TowerOfHanoi_optimal_moves_8_TowerOfHanoi_no_direct_moves_8_TowerOfHanoi_no_smallest_disk_on_peg_2_8_l606_606901

-- Part (a): Prove that the puzzle has a solution and find the optimal number of moves.
theorem TowerOfHanoi_optimal_moves_8 : 
  ∀ (n : ℕ), (K n = 2^n - 1) → (K 8 = 255) :=
begin
  intros,
  sorry
end

-- Part (b): How many moves are needed if direct moves from peg 1 to peg 3 are prohibited?
theorem TowerOfHanoi_no_direct_moves_8 :
  ∀ (n : ℕ), (K n = 3^n - 1) → (K 8 = 6560) :=
begin
  intros,
  sorry
end

-- Part (c): How many moves are needed if the smallest disk cannot be placed on peg 2?
theorem TowerOfHanoi_no_smallest_disk_on_peg_2_8 :
  ∀ (n : ℕ), (K n = 2 * 3^(n - 1) - 1) → (K 8 = 4373) :=
begin
  intros,
  sorry
end

end TowerOfHanoi_optimal_moves_8_TowerOfHanoi_no_direct_moves_8_TowerOfHanoi_no_smallest_disk_on_peg_2_8_l606_606901


namespace max_shortest_side_of_triangle_on_cube_faces_l606_606050

theorem max_shortest_side_of_triangle_on_cube_faces (a b c : ℝ → ℝ → ℝ → Prop) (side_length : ℝ) (cube : ℝ → ℝ → ℝ → Prop) :
  (∀ x, side_length = 1) →
  (∀ x, cube (f x)) →
  (∃ y z w : ℝ, a y z w ∧ b y z w ∧ c y z w ∧ a = b = c = λ x y z, x^2 + y^2 + z^2 = 2) → 
  (min (dist y z) (min (dist y w) (dist z w)) = sqrt 2) :=
by
  intro h_side_length h_cube h_triangle
  sorry

end max_shortest_side_of_triangle_on_cube_faces_l606_606050


namespace inequality_proof_l606_606512

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l606_606512


namespace ce_over_de_l606_606267

theorem ce_over_de {A B C D E T : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ (A →ₗ[ℝ] B)]
  {AT DT BT ET CE DE : ℝ}
  (h1 : AT / DT = 2)
  (h2 : BT / ET = 3) :
  CE / DE = 1 / 2 := 
sorry

end ce_over_de_l606_606267


namespace inequality_proof_l606_606535

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606535


namespace solution_l606_606232

noncomputable def z : ℂ := 3 - 4i

theorem solution (z : ℂ) (h : i * z = 4 + 3 * i) : z = 3 - 4 * i :=
by sorry

end solution_l606_606232


namespace isosceles_triangle_l606_606352

-- Definitions
variables {A B C M K A1 B1 : Type*}
variables (point : A) (point B) (point C) (point M) (point K) (point A1) (point B1)
variable (triangle : A → B → C → Type*)
variable (median : triangle.point → point → Type*)

-- Conditions
variable (is_median_CM : median C M)
variable (is_on_CM : point K ∈ median C M)
variable (intersects_AK_BC : point A1 = AK ∩ BC)
variable (intersects_BK_AC : point B1 = BK ∩ AC)
variable (inscribed_AB1A1B : circle AB1A1B)

-- Proof
theorem isosceles_triangle (h1 : is_median_CM) (h2 : is_on_CM)
  (h3 : intersects_AK_BC) (h4 : intersects_BK_AC)
  (h5 : inscribed_AB1A1B) :
  AB = AC :=
by
  sorry

end isosceles_triangle_l606_606352


namespace range_of_values_for_k_l606_606156

theorem range_of_values_for_k (k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 2*k + 3 = 0 → ((-2)^2 + 0^2 - 4*(2*k + 3) > 0)) ↔ (k < -1) :=
by
  intro h
  sorry

end range_of_values_for_k_l606_606156


namespace shaded_area_of_intersecting_semcircles_is_pi_l606_606751

def midpoint (P Q R: Point) := -- Definition for midpoint (assuming a structure Point)
  Q = ((P.1 + R.1)/2, (P.2 + R.2)/2)

theorem shaded_area_of_intersecting_semcircles_is_pi (P Q R S T M N O: Point): 
  (arc_semicircle_radius P Q R 2 ∧ arc_semicircle_radius R S T 2 ∧ arc_semicircle_radius M N O 2 ∧  
   midpoint P M R ∧ midpoint R N T ∧ midpoint M O N ∧ 
   arc_semicircle M N O) →
  (shaded_area_of_intersections_is_pi (P Q R S T M N O)) :=
begin
  sorry
end

end shaded_area_of_intersecting_semcircles_is_pi_l606_606751


namespace hiking_supplies_l606_606582

theorem hiking_supplies (hours_per_day : ℕ) (days : ℕ) (rate_mph : ℝ) 
    (supply_per_mile : ℝ) (resupply_rate : ℝ)
    (initial_pack_weight : ℝ) : 
    hours_per_day = 8 → days = 5 → rate_mph = 2.5 → 
    supply_per_mile = 0.5 → resupply_rate = 0.25 → 
    initial_pack_weight = (40 : ℝ) :=
by
  intros hpd hd rm spm rr
  sorry

end hiking_supplies_l606_606582


namespace find_angle_B_find_sin_angle_BAC_l606_606269

variables (a b c : ℝ) (A B C : ℝ)
variables (AD BD : ℝ) (α : ℝ)

-- Define the conditions
def triangle_condition : Prop :=
  a^2 + c^2 = b^2 - a*c

def angle_bisector_condition : Prop :=
  AD = 2 * real.sqrt 3 ∧ BD = 1

-- Prove the measure of angle B under the given condition
theorem find_angle_B (h : triangle_condition a b c) : B = 2/3 * real.pi :=
sorry

-- Prove the value of sin(angle BAC) under the given condition
theorem find_sin_angle_BAC (h1 : angle_bisector_condition AD BD) (h2 : α = A) :
  real.sin α = real.sqrt 15 / 8 :=
sorry

end find_angle_B_find_sin_angle_BAC_l606_606269


namespace probability_painted_face_l606_606361

theorem probability_painted_face 
  (cube_volume : ℝ) 
  (small_cube_volume : ℝ) 
  (n_small_cubes : ℕ) 
  (painted_cubes : ℕ)
  (cube_with_painted_face: ℕ): 
  cube_volume = 27 ∧ 
  small_cube_volume = 1 ∧ 
  n_small_cubes = 27 ∧ 
  painted_cubes = 26 →
  cube_with_painted_face / n_small_cubes = 26 / 27 :=
by
  intros h,
  sorry

end probability_painted_face_l606_606361


namespace recurring_decimal_sum_l606_606075

theorem recurring_decimal_sum :
  (0.666666...:ℚ) + (0.222222...:ℚ) - (0.444444...:ℚ) = 4 / 9 :=
begin
  sorry
end

end recurring_decimal_sum_l606_606075


namespace solution_l606_606231

noncomputable def z : ℂ := 3 - 4i

theorem solution (z : ℂ) (h : i * z = 4 + 3 * i) : z = 3 - 4 * i :=
by sorry

end solution_l606_606231


namespace least_integer_greater_than_sqrt_450_l606_606456

-- Define that 21^2 is less than 450
def square_of_21_less_than_450 : Prop := 21^2 < 450

-- Define that 450 is less than 22^2
def square_of_22_greater_than_450 : Prop := 450 < 22^2

-- State the theorem that 22 is the smallest integer greater than sqrt(450)
theorem least_integer_greater_than_sqrt_450 
  (h21 : square_of_21_less_than_450) 
  (h22 : square_of_22_greater_than_450) : 
    ∃ n : ℕ, n = 22 ∧ (√450 : ℝ) < n := 
sorry

end least_integer_greater_than_sqrt_450_l606_606456


namespace candies_leftover_l606_606714

theorem candies_leftover (n : ℕ) : 31254389 % 6 = 5 :=
by {
  sorry
}

end candies_leftover_l606_606714


namespace sum_of_constants_l606_606040

theorem sum_of_constants (c d : ℝ) (h₁ : 16 = 2 * 4 + c) (h₂ : 16 = 4 * 4 + d) : c + d = 8 := by
  sorry

end sum_of_constants_l606_606040


namespace isosceles_triangle_base_angle_l606_606282

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle_isosceles : a = b ∨ b = c ∨ c = a)
  (h_angle_sum : a + b + c = 180) (h_one_angle : a = 50 ∨ b = 50 ∨ c = 50) :
  a = 50 ∨ b = 50 ∨ c = 50 ∨ a = 65 ∨ b = 65 ∨ c = 65 :=
by
  sorry

end isosceles_triangle_base_angle_l606_606282


namespace folded_polygon_perimeter_less_l606_606594

structure Polygon where
  vertices: List (ℝ × ℝ)
  non_empty : vertices ≠ []

def length_segment (A B : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))

def perimeter (P : Polygon) : ℝ :=
  List.sum (List.zipWith length_segment P.vertices (P.vertices.tail ++ [P.vertices.head]))

theorem folded_polygon_perimeter_less (P : Polygon) (A B : ℝ × ℝ)
  (hA : A ∈ P.vertices) (hB : B ∈ P.vertices) :
  let P_folded := (* define the folded polygon appropriately *)
  perimeter P_folded < perimeter P :=
by 
  sorry

end folded_polygon_perimeter_less_l606_606594


namespace inverse_function_of_f_l606_606667

noncomputable def f (x : ℝ) : ℝ := real.sqrt (25 - x^2)

theorem inverse_function_of_f :
  ∀ x, 3 ≤ x ∧ x ≤ 5 → ∃ y, 0 ≤ y ∧ y ≤ 4 ∧ f y = x :=
sorry

end inverse_function_of_f_l606_606667


namespace distinct_pairs_at_round_table_six_people_l606_606379

noncomputable def number_of_distinct_pairs (n : ℕ) : ℕ := 
  if h : n = 6 then 6 else sorry

theorem distinct_pairs_at_round_table_six_people : number_of_distinct_pairs 6 = 6 :=
by {
  unfold number_of_distinct_pairs,
  rw if_pos,
  exact rfl,
  exact rfl
}

end distinct_pairs_at_round_table_six_people_l606_606379


namespace math_problem_l606_606878

theorem math_problem : 
  (Real.sqrt 4) * (4 ^ (1 / 2: ℝ)) + (16 / 4) * 2 - (8 ^ (1 / 2: ℝ)) = 12 - 2 * Real.sqrt 2 :=
by
  sorry

end math_problem_l606_606878


namespace least_integer_greater_than_sqrt_450_l606_606436

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606436


namespace tiling_exists_l606_606049

def valid_tiling (grid : Finset (Fin 6 × Fin 6)) (tiles : Finset (Fin 6 × Fin 6 × Fin 6 × Fin 6 × Fin 6 × Fin 6)) : Prop :=
  (∀ tile ∈ tiles, ∃ a b c : (Fin 6 × Fin 6), 
      ({a, b, c} ⊆ grid ∧
       (abs (a.1 - b.1) + abs (a.2 - b.2) = 1 ∨ abs (a.1 - c.1) + abs (a.2 - c.2) = 1) ∧
       (abs (b.1 - c.1) + abs (b.2 - c.2) = 1 ∨ abs (b.1 - a.1) + abs (b.2 - a.2) = 1) ∧
       (abs (c.1 - a.1) + abs (c.2 - a.2) = 1 ∨ abs (c.1 - b.1) + abs (c.2 - b.2) = 1))) ∧
  (∀ tile1 tile2 ∈ tiles, tile1 ≠ tile2 →
      tile1 ∩ tile2 = ∅ ∧ 
      (∃ a b φ1 φ2 : (Fin 6 × Fin 6), (a ∈ tile1 ∧ b ∈ tile2) → abs (a.1 - b.1) + abs (a.2 - b.2) = 1 → false)) ∧
  (tiles.card = 12)

theorem tiling_exists : ∃ (tiles : Finset (Fin 6 × Fin 6 × Fin 6 × Fin 6 × Fin 6 × Fin 6)),
  valid_tiling (Finset.univ : Finset (Fin 6 × Fin 6)) tiles :=
by
  sorry

end tiling_exists_l606_606049


namespace handshake_problem_proof_l606_606019

noncomputable def total_handshakes
  (groupA groupB groupC : Finset ℕ)
  (groupA_card groupB_card groupC_card : ℕ)
  (groupA_knows_each_other : ∀ a ∈ groupA, ∀ b ∈ groupA, a ≠ b → a ∉ groupA → b ∉ groupA)
  (groupB_knows_noone : ∀ b ∈ groupB, ∀ a ∈ groupA, a ∉ groupB → b ∉ groupB)
  (groupC_knows_15_in_groupA : ∀ c ∈ groupC, (groupA.filter (λ a, a ∈ groupA)).card = 15)
  (groupA_card : groupA.card = 25)
  (groupB_card : groupB.card = 10)
  (groupC_card : groupC.card = 5) : ℕ :=
  let handshakes_A := (groupA.card * (groupA.card - 1)) / 2 in
  let handshakes_B := (groupB.card * (groupB.card - 1)) / 2 in
  let handshakes_C := groupC.card * 15 in
  let total := handshakes_A + handshakes_B + handshakes_C in
  total

theorem handshake_problem_proof
  (groupA groupB groupC : Finset ℕ)
  (groupA_card groupB_card groupC_card : ℕ)
  (groupA_knows_each_other : ∀ a ∈ groupA, ∀ b ∈ groupA, a ≠ b → a ∉ groupA → b ∉ groupA)
  (groupB_knows_noone : ∀ b ∈ groupB, ∀ a ∈ groupA, a ∉ groupB → b ∉ groupB)
  (groupC_knows_15_in_groupA : ∀ c ∈ groupC, (groupA.filter (λ a, a ∈ groupA)).card = 15)
  (groupA_card : groupA.card = 25)
  (groupB_card : groupB.card = 10)
  (groupC_card : groupC.card = 5) :
  total_handshakes groupA groupB groupC groupA_card groupB_card groupC_card groupA_knows_each_other groupB_knows_noone groupC_knows_15_in_groupA groupA_card groupB_card groupC_card = 420 := 
  by sorry

end handshake_problem_proof_l606_606019


namespace least_integer_greater_than_sqrt_450_l606_606441

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606441


namespace least_integer_gt_sqrt_450_l606_606464

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l606_606464


namespace nails_needed_for_house_wall_l606_606115

theorem nails_needed_for_house_wall :
  ∀ (large_planks : ℕ) (nails_per_large_plank : ℕ) (additional_nails : ℕ), 
    large_planks = 13 →
    nails_per_large_plank = 17 →
    additional_nails = 8 →
    (large_planks * nails_per_large_plank + additional_nails = 229) :=
by
  intros large_planks nails_per_large_plank additional_nails
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end nails_needed_for_house_wall_l606_606115


namespace largest_prime_factor_divisors_sum_l606_606787

def prime_factors (n : ℕ) : List ℕ := sorry -- Dummy placeholder for prime factorization

theorem largest_prime_factor_divisors_sum :
  let N := (1 + 2 + 2^2) * (1 + 3 + 3^2) * (1 + 5)
  in List.maximum (prime_factors N) = 13 :=
by
  let N := (1 + 2 + 2^2) * (1 + 3 + 3^2) * (1 + 5)
  have h : prime_factors N = [2, 3, 7, 13] := sorry -- Placeholder
  exact List.maximum_eq_some.mp ⟨13, List.mem_cons_self 13 _, sorry⟩

end largest_prime_factor_divisors_sum_l606_606787


namespace number_of_points_C_l606_606274

theorem number_of_points_C (A B : ℝ × ℝ) (hAB : dist A B = 12) :
  ∃ C : ℝ × ℝ, 4 = card {C : ℝ × ℝ | let area_ABC := dist A (0, 0) * dist B (0, 0) / 2 in
                                      let perimeter_ABC := dist A (0, 0) + dist B (0, 0) + dist C A in
                                      perimeter_ABC = 60 ∧ area_ABC = 120} :=
sorry

end number_of_points_C_l606_606274


namespace union_sets_l606_606172

theorem union_sets (M N : Set ℝ) (hM : M = {x | -3 < x ∧ x < 1}) (hN : N = {x | x ≤ -3}) :
  M ∪ N = {x | x < 1} :=
sorry

end union_sets_l606_606172


namespace triangle_inequality_l606_606133

theorem triangle_inequality (K L M S N : Type)
  [MetricSpace K] [MetricSpace L] [MetricSpace M]
  [AffineSpace ℝ K] [AffineSpace ℝ L] [AffineSpace ℝ M]
  (angle_KML : angle K M L = 121 * (π / 180))
  (K_S_eq_S_N : distance K S = distance S N)
  (S_N_eq_N_L : distance S N = distance N L)
  (M_N_gt_KS : distance M N > distance K S) :
  distance M S < distance N L := by
  sorry

end triangle_inequality_l606_606133


namespace subset_bound_l606_606132

theorem subset_bound (A : list (set ℕ)) (k n t : ℕ) 
  (h1 : ∀ A_i A_j ∈ A, A_i ≠ A_j → (A_i ∆ A_j).card = k)
  (h2 : k = (n + 1) / 2 ∨ (k ≠ (n + 1) / 2)) : 
  (k = (n + 1) / 2 → t ≤ n + 1) ∧ (k ≠ (n + 1) / 2 → t ≤ n) :=
by {
  sorry
}

end subset_bound_l606_606132


namespace repeating_decimal_sum_l606_606063

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem repeating_decimal_sum : repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l606_606063


namespace inequality_proof_l606_606551

variable {x y z : ℝ}

theorem inequality_proof (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * real.sqrt(z * x)) + 
  y^3 / (y^3 + 2 * z^2 * real.sqrt(x * y)) + 
  z^3 / (z^3 + 2 * x^2 * real.sqrt(y * z)) >= 1) :=
sorry

end inequality_proof_l606_606551


namespace square_room_tiles_and_triangles_l606_606937

theorem square_room_tiles_and_triangles:
  ∀ n : ℕ, (2 * n - 1 = 57) → (n^2 = 841 ∧ 4 triangles):
  sorry

end square_room_tiles_and_triangles_l606_606937


namespace solve_for_z_l606_606218

variable (z : ℂ)

theorem solve_for_z (h : (complex.I * z = 4 + 3 * complex.I)) : z = 3 - 4 * complex.I := 
sorry

end solve_for_z_l606_606218


namespace find_angle_between_gradients_at_M0_l606_606643

noncomputable def u (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)
noncomputable def v (x y : ℝ) : ℝ := x + y + 2 * Real.sqrt (x * y)

def gradient (f : ℝ → ℝ → ℝ) (x y : ℝ) : ℝ × ℝ :=
  (Real.deriv (λ x, f x y) x, Real.deriv (λ y, f x y) y)

def dot_product (A B : ℝ × ℝ) : ℝ := A.1 * B.1 + A.2 * B.2
def magnitude (A : ℝ × ℝ) : ℝ := Real.sqrt (A.1^2 + A.2^2)

def cos_theta (A B : ℝ × ℝ) : ℝ :=
  dot_product A B / (magnitude A * magnitude B)

theorem find_angle_between_gradients_at_M0 :
  cos_theta (gradient u 1 1) (gradient v 1 1) = 1 :=
sorry

end find_angle_between_gradients_at_M0_l606_606643


namespace common_fraction_l606_606091

noncomputable def x : ℚ := 0.6666666 -- represents 0.\overline{6}
noncomputable def y : ℚ := 0.2222222 -- represents 0.\overline{2}
noncomputable def z : ℚ := 0.4444444 -- represents 0.\overline{4}

theorem common_fraction :
  x + y - z = 4 / 9 :=
by
  -- Provide proofs here
  sorry

end common_fraction_l606_606091


namespace systematic_sampling_second_invoice_l606_606020

theorem systematic_sampling_second_invoice 
  (N : ℕ) 
  (valid_invoice : N ≥ 10)
  (first_invoice : Fin 10) :
  ¬ (∃ k : ℕ, k ≥ 1 ∧ first_invoice.1 + k * 10 = 23) := 
by 
  -- Proof omitted
  sorry

end systematic_sampling_second_invoice_l606_606020


namespace inequality_proof_l606_606544

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l606_606544


namespace spinner_probability_l606_606936

-- Define the geometric properties of the square game board
def eight_triangular_regions (square : Type) := 
  ∃ (triangle : Type), 
    (square → Prop) ∧ 
    (triangle → Prop) ∧ 
    -- The square can be divided into eight triangular regions
    (∀ s : square, ∃ t : triangle, 8 * t = s) 

-- Define the shaded regions
def shaded_regions (triangle : Type) := 
  ∃ (shaded : triangle → Prop), 
    -- Four triangles are shaded
    (∀ t : triangle, ∃ st : shaded t, 4 * st = t)

-- Probability function
def probability (shaded total : ℕ) : ℚ := shaded / total

-- Proof problem statement
theorem spinner_probability :
  ∀ (square : Type) (triangle : Type) (s : square) (t : triangle),
  eight_triangular_regions square →
  shaded_regions triangle →
  probability 4 8 = 1 / 2 := 
by
  intro square triangle s t
  intros h1 h2
  -- Proof omitted
  sorry

end spinner_probability_l606_606936


namespace limit_of_f_at_neg1_l606_606898

noncomputable def f (x : ℝ) : ℝ := (x^3 - 2*x - 1) / (x^4 + 2*x + 1)

theorem limit_of_f_at_neg1 : 
  tendsto f (𝓝 (-1)) (𝓝 (-1/2)) :=
by
  sorry

end limit_of_f_at_neg1_l606_606898


namespace divide_money_according_to_ratio_l606_606833

theorem divide_money_according_to_ratio :
  let total_amount := 12000
  let ratio := (2, 4, 6, 3, 5)
  let total_parts := ratio.1 + ratio.2 + ratio.3 + ratio.4 + ratio.5
  let part_value := total_amount / total_parts
  let john := ratio.1 * part_value
  let jose := ratio.2 * part_value
  let binoy := ratio.3 * part_value
  let sofia := ratio.4 * part_value
  let ravi := ratio.5 * part_value
  john = 1200 ∧ jose = 2400 ∧ binoy = 3600 ∧ sofia = 1800 ∧ ravi = 3000 :=
by
  sorry

end divide_money_according_to_ratio_l606_606833


namespace lines_not_intersecting_may_be_parallel_or_skew_l606_606257

theorem lines_not_intersecting_may_be_parallel_or_skew (a b : ℝ × ℝ → Prop) 
  (h : ∀ x, ¬ (a x ∧ b x)) : 
  (∃ c d : ℝ × ℝ → Prop, a = c ∧ b = d) := 
sorry

end lines_not_intersecting_may_be_parallel_or_skew_l606_606257


namespace find_pairs_l606_606778

noncomputable def p (x : ℂ) : ℂ := x^5 + x

noncomputable def q (x : ℂ) : ℂ := x^5 + x^2

theorem find_pairs (w z : ℂ) (h1 : w ≠ z) :
  p w = p z ∧ q w = q z ↔ 
  ((w = complex.exp(2 * π * complex.I / 3) ∧ z = 1 - complex.exp(2 * π * complex.I / 3)) ∨
  (w = complex.exp(4 * π * complex.I / 3) ∧ z = 1 - complex.exp(4 * π * complex.I / 3)) ∨
  (w = (1 + complex.I * complex.sqrt 3) / 2 ∧ z = (1 - complex.I * complex.sqrt 3) / 2) ∨
  (w = (1 - complex.I * complex.sqrt 3) / 2 ∧ z = (1 + complex.I * complex.sqrt 3) / 2)) :=
sorry

end find_pairs_l606_606778


namespace seating_arrangements_7_people_l606_606285

-- Define a function to calculate the number of ways the 7 people can be arranged
-- under the given conditions
def seatingArrangements (totalPeople : Nat) (specificPair : Nat) : Nat := 
  let units := totalPeople - 1
  factorial units * specificPair / units

-- The main theorem stating the number of arrangements is 240
theorem seating_arrangements_7_people :
  seatingArrangements 7 2 = 240 := by
  sorry

end seating_arrangements_7_people_l606_606285


namespace perimeter_of_triangle_APR_is_50_l606_606418

noncomputable def perimeter_triangle_APR : ℝ :=
  let AB := 25
  let PQ := 2.5
  let QR := 2.5
  let BP := PQ
  let CR := QR
  let AP := AB - BP
  let AR := AB - CR
  let PR := PQ + QR
  AP + PR + AR

theorem perimeter_of_triangle_APR_is_50 : perimeter_triangle_APR = 50 :=
  by
    let AB := 25
    let PQ := 2.5
    let QR := 2.5
    have BP := PQ
    have CR := QR
    have AP := AB - BP
    have AR := AB - CR
    have PR := PQ + QR
    calc
      AP + PR + AR = 22.5 + 5 + 22.5 := by sorry -- Calculation step, can be elaborated if needed
              ... = 50 := by rfl

end perimeter_of_triangle_APR_is_50_l606_606418


namespace initial_lives_l606_606871

theorem initial_lives (L : ℕ) (h1 : L - 6 + 37 = 41) : L = 10 :=
by
  sorry

end initial_lives_l606_606871


namespace largest_prime_factor_of_sum_of_divisors_of_180_l606_606784

-- Define the function to compute the sum of divisors
noncomputable def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ k in (Finset.range (n + 1)).filter (λ k, n % k = 0), k

-- Define a function to find the largest prime factor of a number
noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  Finset.max' (Finset.filter Nat.prime (Finset.range (n + 1))) sorry

-- Define the problem conditions
def N : ℕ := sum_of_divisors 180

-- State the main theorem to be proved
theorem largest_prime_factor_of_sum_of_divisors_of_180 : largest_prime_factor N = 13 :=
by sorry

end largest_prime_factor_of_sum_of_divisors_of_180_l606_606784


namespace intersection_A_B_l606_606719

def is_natural (x : ℕ) := true

def A : set ℕ := {x ∈ (λ x, is_natural x) | 5 + 4 * x - x^2 > 0}
def B : set ℕ := {x ∈ (λ x, is_natural x) | x < 3}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} :=
by {
  sorry
}

end intersection_A_B_l606_606719


namespace find_cx_squared_l606_606808

def unit_square (A B C D : ℝ × ℝ) := 
  A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

def equidistant_to_diagonals (X : ℝ × ℝ) :=
  dist_to_AC X = dist_to_BD X

def dist_to_AC (X : ℝ × ℝ) (A C : ℝ × ℝ) : ℝ := sorry
def dist_to_BD (X : ℝ × ℝ) (B D : ℝ × ℝ) : ℝ := sorry

noncomputable def point_X (X : ℝ × ℝ) (A C : ℝ × ℝ) :=
  (X.1 - A.1)^2 + (X.2 - A.2)^2 = (sqrt 2 / 2)^2

theorem find_cx_squared :
  ∀ (A B C D X : ℝ × ℝ),
    unit_square A B C D →
    X.1 > 1 ∨ X.2 > 1 ∨ X.1 < 0 ∨ X.2 < 0 → 
    dist_to_AC X A C = dist_to_BD X B D →
    point_X X A →
    (C.1 - X.1)^2 + (C.2 - X.2)^2 = 5 / 2 := 
by
  intros A B C D X h_sq h_outside h_dist h_AX
  sorry

end find_cx_squared_l606_606808


namespace problem1_problem2_l606_606025

-- Problem 1
theorem problem1 (m n : ℚ) (h : m ≠ n) : 
  (m / (m - n)) + (n / (n - m)) = 1 := 
by
  -- Proof steps would go here
  sorry

-- Problem 2
theorem problem2 (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) : 
  (2 / (x^2 - 1)) / (1 / (x + 1)) = 2 / (x - 1) := 
by
  -- Proof steps would go here
  sorry

end problem1_problem2_l606_606025


namespace price_of_eraser_l606_606501

variables (x y : ℝ)

theorem price_of_eraser : 
  (3 * x + 5 * y = 10.6) ∧ (4 * x + 4 * y = 12) → x = 2.2 :=
by
  sorry

end price_of_eraser_l606_606501


namespace number_of_distinct_denominators_l606_606843

def is_digit (n : ℕ) : Prop :=
  n < 10

def valid_combination (a b : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ ¬(a = 0 ∧ b = 0)

def count_possible_denominators : ℕ :=
  7

theorem number_of_distinct_denominators (a b : ℕ) (h : valid_combination a b) : 
  ∃ d, d ∈ {3, 9, 27, 37, 111, 333, 999} :=
sorry

end number_of_distinct_denominators_l606_606843


namespace min_area_triangle_l606_606664

theorem min_area_triangle (m n : ℝ) (h : m^2 + n^2 = 1/3) : ∃ S, S = 3 :=
by
  sorry

end min_area_triangle_l606_606664


namespace Q_investment_l606_606893

-- Given conditions
variables (P Q : Nat) (P_investment : P = 30000) (profit_ratio : 2 / 3 = P / Q)

-- Target statement
theorem Q_investment : Q = 45000 :=
by 
  sorry

end Q_investment_l606_606893


namespace solution_l606_606233

noncomputable def z : ℂ := 3 - 4i

theorem solution (z : ℂ) (h : i * z = 4 + 3 * i) : z = 3 - 4 * i :=
by sorry

end solution_l606_606233


namespace range_of_a_l606_606260

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2*x + a * real.log x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Ioo 0 1, deriv (f a) x ≤ 0) ↔ a ≤ -4 :=
sorry

end range_of_a_l606_606260


namespace find_a_l606_606196

theorem find_a (a b c : ℤ) (h : (∀ x : ℝ, (x - a) * (x - 5) + 4 = (x + b) * (x + c))) :
  a = 0 ∨ a = 1 :=
sorry

end find_a_l606_606196


namespace part_1_part_2_l606_606123

variable (m : ℝ)

-- Define the propositions p and q using Lean constructs
def p : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → m ≤ x^2 - 2*x
def q : Prop := ∃ x : ℝ, 0 ≤ x ∧ 2^x + 3 = m

-- Statement for Part 1
theorem part_1 (hp : p m) : m ∈ set.Iic (-1) := sorry

-- Statement for Part 2
theorem part_2 (hpq_true : p m ∨ q m) (hpq_false : ¬(p m ∧ q m)) :
  m ∈ set.Iic (-1) ∪ set.Ici 4 := sorry

end part_1_part_2_l606_606123


namespace curve_is_line_l606_606053

theorem curve_is_line :
  ∀ (r θ : ℝ), r = 1 / (2 * sin θ - cos θ) →
  ∃ (x y : ℝ), x = r * cos θ ∧ y = r * sin θ ∧ 2 * x * y - 3 * x - 1 = 0 :=
by
  intro r θ hr
  use [r * cos θ, r * sin θ]
  sorry

end curve_is_line_l606_606053


namespace border_material_correct_l606_606848

noncomputable def pi_approx := (22 : ℚ) / 7

def circle_radius (area : ℚ) (pi_value : ℚ) : ℚ :=
  (area * (7 / 22)).sqrt

def circumference (radius : ℚ) (pi_value : ℚ) : ℚ :=
  2 * pi_value * radius

def total_border_material (area : ℚ) (pi_value : ℚ) (extra : ℚ) : ℚ :=
  circumference (circle_radius area pi_value) pi_value + extra

theorem border_material_correct :
  total_border_material 616 pi_approx 3 = 91 :=
by
  sorry

end border_material_correct_l606_606848


namespace length_EF_l606_606569

theorem length_EF (a b c : ℝ) (C : a ≤ b) (h_incirc : ∃ r, r * (a + b + c) = 2 * triangle_area(a, b, c)) 
    (E F : triangle.pts_rest (isosceles_triangle ABC a >= AB) 
    (BE_PQ : ∃ P Q : R2, P ∈ circle ∧ Q ∈ circle ∧ segment(BE).contains P ∧ segment(BE).contains Q) 
    (BP_PQ : BP = 1 ∧ PQ = 8) : EF = 6 := 
begin 
    sorry 
end

end length_EF_l606_606569


namespace total_bathing_suits_l606_606910

theorem total_bathing_suits 
  (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ)
  (ha : a = 8500) (hb : b = 12750) (hc : c = 5900) (hd : d = 7250) (he : e = 1100) :
  a + b + c + d + e = 35500 :=
by
  sorry

end total_bathing_suits_l606_606910


namespace inequality_proof_l606_606540

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l606_606540


namespace percentage_error_in_area_l606_606010

theorem percentage_error_in_area {x : ℝ} (hx : x > 0) :
    let measured_side := 1.12 * x
    let actual_area := x^2
    let erroneous_area := (1.12 * x)^2
    let percentage_error := ((erroneous_area - actual_area) / actual_area) * 100
  in percentage_error = 25.44 :=
by
  sorry

end percentage_error_in_area_l606_606010


namespace count_non_adjacent_arrangements_l606_606836

-- Define the number of people and the specific individuals A and B
def num_people : ℕ := 6
def individuals : set ℕ := {1, 2, 3, 4, 5, 6}
def a_person : ℕ := 1
def b_person : ℕ := 2

-- Define the constraint that A and B should not be adjacent
def not_adjacent (arrangement : list ℕ) : Prop :=
  ∀ (i : ℕ), i < arrangement.length - 1 →
    (arrangement.nth i ≠ some a_person ∨ arrangement.nth (i + 1) ≠ some b_person) ∧
    (arrangement.nth i ≠ some b_person ∨ arrangement.nth (i + 1) ≠ some a_person)

-- Define the main theorem statement
theorem count_non_adjacent_arrangements :
  ∃ (num_arrangements : ℕ), num_arrangements = 480 :=
sorry

end count_non_adjacent_arrangements_l606_606836


namespace cos_seven_pi_over_four_l606_606640

theorem cos_seven_pi_over_four :
  Real.cos (7 * Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_seven_pi_over_four_l606_606640


namespace largest_odd_digit_multiple_of_5_lt_10000_l606_606876

def is_odd_digit (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), is_odd_digit d

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem largest_odd_digit_multiple_of_5_lt_10000 :
  ∃ n, n < 10000 ∧ all_odd_digits n ∧ is_multiple_of_5 n ∧
        ∀ m, m < 10000 → all_odd_digits m → is_multiple_of_5 m → m ≤ n :=
  sorry

end largest_odd_digit_multiple_of_5_lt_10000_l606_606876


namespace repeating_decimals_sum_l606_606094

theorem repeating_decimals_sum : (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ) = 4 / 9 :=
 by 
  have h₁ : (0.666666... : ℚ) = 2 / 3,
    -- Since x = 0.6666..., then 10x = 6.6666...,
    -- so 10x - x = 6, then x = 6 / 9, hence 2 / 3
    sorry,
  have h₂ : (0.222222... : ℚ) = 2 / 9,
    -- Since x = 0.2222..., then 10x = 2.2222...,
    -- so 10x - x = 2, then x = 2 / 9
    sorry,
  have h₃ : (0.444444... : ℚ) = 4 / 9,
    -- Since x = 0.4444..., then 10x = 4.4444...,
    -- so 10x - x = 4, then x = 4 / 9
    sorry,
  calc
    (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ)
        = (2 / 3) + (2 / 9) - (4 / 9) : by rw [h₁, h₂, h₃]
    ... = (6 / 9) + (2 / 9) - (4 / 9) : by norm_num
    ... = 4 / 9 : by ring

end repeating_decimals_sum_l606_606094


namespace cookie_distribution_l606_606380

theorem cookie_distribution:
  ∀ (initial_boxes brother_cookies sister_cookies leftover after_siblings leftover_sonny : ℕ),
    initial_boxes = 45 →
    brother_cookies = 12 →
    sister_cookies = 9 →
    after_siblings = initial_boxes - brother_cookies - sister_cookies →
    leftover_sonny = 17 →
    leftover = after_siblings - leftover_sonny →
    leftover = 7 :=
by
  intros initial_boxes brother_cookies sister_cookies leftover after_siblings leftover_sonny
  intros h1 h2 h3 h4 h5 h6
  sorry

end cookie_distribution_l606_606380


namespace minimum_n_value_l606_606421

def is_three_element_subset (s : Finset ℕ) : Prop :=
  s.card = 3

def has_at_most_one_common_element (S : Finset (Finset ℕ)) : Prop :=
  ∀ s1 s2 ∈ S, s1 ≠ s2 → (s1 ∩ s2).card ≤ 1

def is_nice_subset (S : Finset (Finset ℕ)) (N : Finset ℕ) : Prop :=
  ∀ s ∈ S, ¬ (s ⊆ N)

theorem minimum_n_value :
  ∀ (S : Finset (Finset ℕ)),
  (∀ s ∈ S, is_three_element_subset s) →
  has_at_most_one_common_element S →
  (∀ N : Finset ℕ, N.card = 29 → (∃ x ∉ N, is_nice_subset S (insert x N))) →
  ∃ n ≥ 436, ∀ N : Finset ℕ, N.card = 29 → (∃ x ∉ N, is_nice_subset S (insert x N))
:= sorry

end minimum_n_value_l606_606421


namespace angle_between_vectors_l606_606173

noncomputable def vector (n : Type*) := ℝ^n

variables (a b : vector 2) (theta : ℝ)

-- Conditions
axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0
axiom perp_condition1 : (a - (2 : ℝ) • b) ⬝ a = 0
axiom perp_condition2 : (b - (2 : ℝ) • a) ⬝ b = 0

-- Statement
theorem angle_between_vectors : theta = π / 3 :=
sorry

end angle_between_vectors_l606_606173


namespace sin_magnitude_comparison_l606_606031

theorem sin_magnitude_comparison (x : ℝ) (n : ℕ) (hn : 0 < n) : 
  n * (Real.sin x)^2 ≥ (Real.sin x) * (Real.sin(n * x)) :=
sorry

end sin_magnitude_comparison_l606_606031


namespace dihedral_angle_l606_606826

-- Define the pyramid PQRS with given conditions
variables (PQ RS PR PS PT : ℝ)
variables (angle_PQT : ℝ)
variables (cos_phi : ℝ)
variables (a b : ℝ)

-- Conditions based on problem statement
def pyramid_conditions : Prop :=
  PQ = PR ∧ PQ = PS ∧ PQ = PT ∧ RS = QR
  ∧ angle_PQT = 60

-- Equivalent proof problem stating the relationship between cos_phi, a, and b
theorem dihedral_angle :
  pyramid_conditions PQ RS PR PS PT angle_PQT
  → cos_phi = a - real.sqrt b
  → a + b = 4 :=
by
  sorry

end dihedral_angle_l606_606826


namespace repeating_decimals_sum_l606_606092

theorem repeating_decimals_sum : (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ) = 4 / 9 :=
 by 
  have h₁ : (0.666666... : ℚ) = 2 / 3,
    -- Since x = 0.6666..., then 10x = 6.6666...,
    -- so 10x - x = 6, then x = 6 / 9, hence 2 / 3
    sorry,
  have h₂ : (0.222222... : ℚ) = 2 / 9,
    -- Since x = 0.2222..., then 10x = 2.2222...,
    -- so 10x - x = 2, then x = 2 / 9
    sorry,
  have h₃ : (0.444444... : ℚ) = 4 / 9,
    -- Since x = 0.4444..., then 10x = 4.4444...,
    -- so 10x - x = 4, then x = 4 / 9
    sorry,
  calc
    (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ)
        = (2 / 3) + (2 / 9) - (4 / 9) : by rw [h₁, h₂, h₃]
    ... = (6 / 9) + (2 / 9) - (4 / 9) : by norm_num
    ... = 4 / 9 : by ring

end repeating_decimals_sum_l606_606092


namespace four_dimensional_measure_of_hypersphere_l606_606736

theorem four_dimensional_measure_of_hypersphere (r V : ℝ) (hV : V = 12 * π * r^3) :
  ∃ (W : ℝ), W = 3 * π * r^4 :=
by
  have h := ∫ (12 * π * r^3) dx -- Assuming an appropriate integral method
  sorry

end four_dimensional_measure_of_hypersphere_l606_606736


namespace condition_check_l606_606258

variables {α : Type*} (M N : Set α) (a : α)

theorem condition_check (h₁ : M ⊂ N) (h₂ : M.nonempty) :
  (a ∈ M ∩ N) ↔
    (a ∈ M ∧ a ∈ N) := sorry

end condition_check_l606_606258


namespace inequality_proof_l606_606548

variable {x y z : ℝ}

theorem inequality_proof (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * real.sqrt(z * x)) + 
  y^3 / (y^3 + 2 * z^2 * real.sqrt(x * y)) + 
  z^3 / (z^3 + 2 * x^2 * real.sqrt(y * z)) >= 1) :=
sorry

end inequality_proof_l606_606548


namespace students_more_than_rabbits_by_64_l606_606055

-- Define the conditions as constants
def number_of_classrooms : ℕ := 4
def students_per_classroom : ℕ := 18
def rabbits_per_classroom : ℕ := 2

-- Define the quantities that need calculations
def total_students : ℕ := number_of_classrooms * students_per_classroom
def total_rabbits : ℕ := number_of_classrooms * rabbits_per_classroom
def difference_students_rabbits : ℕ := total_students - total_rabbits

-- State the theorem to be proven
theorem students_more_than_rabbits_by_64 :
  difference_students_rabbits = 64 := by
  sorry

end students_more_than_rabbits_by_64_l606_606055


namespace min_sum_ab_l606_606136

def point : Type := (ℝ × ℝ)

noncomputable def A : point := (1, -1)
noncomputable def B : point := (4, 0)
noncomputable def C : point := (2, 2)

-- Define the vector operation and the area condition
def vector_sub (p q : point) : point := (p.1 - q.1, p.2 - q.2)
def scalar_mul (λ : ℝ) (v : point) : point := (λ * v.1, λ * v.2)
def vector_add (v1 v2 : point) : point := (v1.1 + v2.1, v1.2 + v2.2)

-- Defining the conditions
def condition (P : point) (a b : ℝ): Prop :=
  ∃ λ μ : ℝ, 1 < λ ∧ λ ≤ a ∧ 1 < μ ∧ μ ≤ b ∧ 
  P = vector_add (scalar_mul λ (vector_sub B A)) (scalar_mul μ (vector_sub C A))

noncomputable def region (a b : ℝ) : set point :=
  { P : point | condition P a b }

-- Assert the area condition
axiom area_condition (a b : ℝ) : area (region a b) = 8

-- Prove that a + b = 4
theorem min_sum_ab (a b : ℝ): a + b = 4 :=
by
  -- Skip proof
  sorry

end min_sum_ab_l606_606136


namespace inequality_xyz_l606_606524

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x))) +
  (y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y))) +
  (z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 :=
by sorry

end inequality_xyz_l606_606524


namespace probability_correct_digit_in_two_attempts_l606_606399

theorem probability_correct_digit_in_two_attempts :
  let total_digits := 10
  let probability_first_correct := 1 / total_digits
  let probability_first_incorrect := 9 / total_digits
  let probability_second_correct_if_first_incorrect := 1 / (total_digits - 1)
  (probability_first_correct + probability_first_incorrect * probability_second_correct_if_first_incorrect) = 1 / 5 := 
sorry

end probability_correct_digit_in_two_attempts_l606_606399


namespace inradius_of_isosceles_triangle_l606_606810

theorem inradius_of_isosceles_triangle 
  (A B C I : Point)
  (h_isosceles : AB = AC) 
  (BC_eq_24 : dist B C = 24)
  (IC_eq_20 : dist I C = 20)
  (h_incenter : is_incenter I A B C) :
  inradius I A B C = 16 := 
by 
  sorry

end inradius_of_isosceles_triangle_l606_606810


namespace triangle_angle_bisectors_l606_606142

theorem triangle_angle_bisectors {a b c : ℝ} (ht : (a = 2 ∧ b = 3 ∧ c < 5)) : 
  (∃ h_a h_b h_c : ℝ, h_a + h_b > h_c ∧ h_a + h_c > h_b ∧ h_b + h_c > h_a) →
  ¬ (∃ ell_a ell_b ell_c : ℝ, ell_a + ell_b > ell_c ∧ ell_a + ell_c > ell_b ∧ ell_b + ell_c > ell_a) :=
by
  sorry

end triangle_angle_bisectors_l606_606142


namespace intersection_range_of_k_l606_606495

theorem intersection_range_of_k :
  ∀ k : ℝ,
  (∃ x y : ℝ, y = 1 + real.sqrt(1 - x^2) ∧ y = k * (x - 3) + 3) ↔
  k ∈ set.Ioc ((3 - real.sqrt 3) / 4) (1 / 2) :=
sorry

end intersection_range_of_k_l606_606495


namespace sqrt_inequality_l606_606032

theorem sqrt_inequality : sqrt 3 + sqrt 7 < 2 * sqrt 5 :=
by
  sorry

end sqrt_inequality_l606_606032


namespace find_ab_l606_606704

noncomputable def perpendicular_condition (a b : ℝ) :=
  a * (a - 1) - b = 0

noncomputable def point_on_l1_condition (a b : ℝ) :=
  -3 * a + b + 4 = 0

noncomputable def parallel_condition (a b : ℝ) :=
  a + b * (a - 1) = 0

noncomputable def distance_condition (a : ℝ) :=
  4 = abs ((-a) / (a - 1))

theorem find_ab (a b : ℝ) :
  (perpendicular_condition a b ∧ point_on_l1_condition a b ∧
   parallel_condition a b ∧ distance_condition a) →
  ((a = 2 ∧ b = 2) ∨ (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = 2)) :=
by
  sorry

end find_ab_l606_606704


namespace adam_final_score_l606_606554

theorem adam_final_score :
  ∀ (q1 q2 points_per_question : ℕ), 
  q1 = 5 → q2 = 5 → points_per_question = 5 →
  (q1 + q2) * points_per_question = 50 :=
by 
  intros q1 q2 points_per_question hq1 hq2 hpoints_per_question
  rw [hq1, hq2, hpoints_per_question]
  sorry

end adam_final_score_l606_606554


namespace count_correct_statements_l606_606600

theorem count_correct_statements :
  (altitude_is_line_segment : ∀ (A B C : Type) (T : Triangle A B C), is_line_segment (altitude T))
  → (median_is_line_segment : ∀ (A B C : Type) (T : Triangle A B C), is_line_segment (median T))
  → (angle_bisector_is_line_segment : ∀ (A B C : Type) (T : Triangle A B C), is_line_segment (angle_bisector T))
  → (exterior_angle_property : ∀ (A B C : Type) (T : Triangle A B C) (ext_angle : Angle T), ext_angle > interior_angle1 T ∨ ext_angle > interior_angle2 T)
  → (angle_condition : ∀ (A B C : ℝ), ∀ T : Triangle A B C, (angle A = 2 * angle B = 3 * angle C) → ¬(is_right_triangle T))
  → (triangle_condition : ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b > c → (is_triangle (a, b, c) ↔ (a + b > c ∧ a + c > b ∧ b + c > a)))
  → 1 = 1 := 
by
  intros
  /- proof goes here, but is omitted -/
  sorry

end count_correct_statements_l606_606600


namespace part1_part2_part3_l606_606338

def z (m : ℝ) : ℂ :=
  (1 + complex.i) * m^2 + (5 - 2 * complex.i) * m + (6 - 15 * complex.i)

theorem part1 : ∀ (m : ℝ), z m ∈ ℝ ↔ (m = 5 ∨ m = -3) :=
by
  intro m
  split
  sorry, sorry

theorem part2 : ∀ (m : ℝ), z m ∈ ℂ.im ↔ m = -2 :=
by
  intro m
  split
  sorry, sorry

theorem part3 : ∀ (m : ℝ), let z_m := z m in 
  z_m.re + z_m.im + 7 = 0 ↔ (m = 1/2 ∨ m = -2) :=
by
  intro m
  split
  sorry, sorry

end part1_part2_part3_l606_606338


namespace relationship_among_abc_l606_606624

-- Definitions for the given conditions
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

noncomputable def a : ℝ := 1 -- a is found to be 1
noncomputable def b : ℝ := ∫ x in 0..1, x
noncomputable def c : ℝ := ∫ x in 0..1, real.sqrt (1 - x^2)

-- The theorem to prove the relationship among a, b, and c
theorem relationship_among_abc 
  (h_pure_imaginary : is_purely_imaginary (⟨a, 1⟩ * ⟨1, 1⟩ / 2))
: b < c < a :=
  sorry

end relationship_among_abc_l606_606624


namespace students_exceed_rabbits_l606_606058

theorem students_exceed_rabbits (students_per_classroom rabbits_per_classroom number_of_classrooms : ℕ) 
  (h_students : students_per_classroom = 18)
  (h_rabbits : rabbits_per_classroom = 2)
  (h_classrooms : number_of_classrooms = 4) : 
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 64 :=
by {
  sorry
}

end students_exceed_rabbits_l606_606058


namespace extreme_values_a_eq_3_monotonicity_a_gt_1_l606_606162

section
variable {x : ℝ}

def f (x : ℝ) (a : ℝ) : ℝ := (1 - a) / 2 * x^2 + a * x - log x

theorem extreme_values_a_eq_3 :
  (∀ x > 0, f x 3 ≤ 2) ∧ (∃ y > 0, f y 3 = 2) ∧ (∃ z > 0, f z 3 = (5 / 4) + log 2) :=
sorry

theorem monotonicity_a_gt_1 (a : ℝ) (h : a > 1) :
  (1 < a ∧ a < 2 → (∀ x, 0 < x ∧ x < 1 → f' x a < 0) ∧ 
                     (∀ x, 1 < x ∧ x < 1 / (a - 1) → f' x a > 0) ∧
                     (∀ x, 1 / (a - 1) < x → f' x a < 0)) ∧
  (a = 2 → ∀ x, 0 < x → f' x 2 ≤ 0) ∧
  (a > 2 → (∀ x, 0 < x ∧ x < 1 / (a - 1) → f' x a < 0) ∧
                   (∀ x, 1 / (a - 1) < x ∧ x < 1 → f' x a > 0) ∧
                   (∀ x, 1 < x → f' x a < 0)) :=
sorry
end

end extreme_values_a_eq_3_monotonicity_a_gt_1_l606_606162


namespace find_tan_theta_l606_606119

open Real

theorem find_tan_theta (θ : ℝ) (h1 : sin θ + cos θ = 7 / 13) (h2 : 0 < θ ∧ θ < π) :
  tan θ = -12 / 5 :=
sorry

end find_tan_theta_l606_606119


namespace parallelogram_altitudes_l606_606740

theorem parallelogram_altitudes
  (DC : ℝ) (EB : ℝ) (DE : ℝ) (AB : ℝ) (BC : ℝ) (area : ℝ)
  (h1 : DC = 15)
  (h2 : EB = 3)
  (h3 : DE = 5)
  (h4 : AB = 15)
  (h5 : BC = 15)
  (h6 : area = 75) :
  ∃ DF : ℝ, DF = 5 :=
by
  use 5
  sorry

end parallelogram_altitudes_l606_606740


namespace min_ineq_l606_606646

theorem min_ineq (x : ℝ) (hx : x > 0) : 3*x + 1/x^2 ≥ 4 :=
sorry

end min_ineq_l606_606646


namespace problem_condition_l606_606599

noncomputable def fA (x : ℝ) : ℝ := x + 4 / x
noncomputable def fB (x : ℝ) : ℝ := Real.sin x + 4 / Real.sin x
noncomputable def fC (x : ℝ) : ℝ := 4 * log 3 x + log x 3
noncomputable def fD (x : ℝ) : ℝ := 4 * Real.exp x + Real.exp (-x)

theorem problem_condition :
  (∀ x : ℝ, fA x ≠ 4) ∧
  (∀ x : ℝ, 0 < x ∧ x < π → fB x ≠ 4) ∧
  (∀ x : ℝ, fC x ≠ 4) ∧
  (∃ x : ℝ, fD x = 4) :=
by
  -- Proof omitted for brevity
  sorry

end problem_condition_l606_606599


namespace count_3digit_numbers_divisible_by_11_l606_606186

theorem count_3digit_numbers_divisible_by_11 :
  let smallest := 100 in
  let largest := 999 in
  -- Determine the smallest and largest 3-digit numbers divisible by 11
  let smallest_divisible_by_11 := ((smallest + 10) / 11) * 11 in
  let largest_divisible_by_11 := (largest / 11) * 11 in
  -- Number of positive 3-digit numbers divisible by 11
  (largest_divisible_by_11 / 11) - (smallest_divisible_by_11 / 11) + 1 = 81 :=
by
  sorry

end count_3digit_numbers_divisible_by_11_l606_606186


namespace at_least_one_wins_l606_606417

def probability_A := 1 / 2
def probability_B := 1 / 4

def probability_at_least_one (pA pB : ℚ) : ℚ := 
  1 - ((1 - pA) * (1 - pB))

theorem at_least_one_wins :
  probability_at_least_one probability_A probability_B = 5 / 8 := 
by
  sorry

end at_least_one_wins_l606_606417


namespace equivalent_terminal_side_l606_606644

theorem equivalent_terminal_side (k : ℤ) : 
    (∃ k : ℤ, (5 * π / 3 = -π / 3 + 2 * π * k)) :=
sorry

end equivalent_terminal_side_l606_606644


namespace count_valid_four_digit_numbers_l606_606184

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 5

def is_valid_four_digit_number (n : ℕ) : Prop := 
  n >= 1000 ∧ n < 10000 ∧
  (let d1 := n / 1000 % 10 in is_valid_digit d1) ∧
  (let d2 := n / 100 % 10 in is_valid_digit d2) ∧
  (let d3 := n / 10 % 10 in is_valid_digit d3) ∧
  (let d4 := n % 10 in is_valid_digit d4)

theorem count_valid_four_digit_numbers : 
  {n : ℕ | is_valid_four_digit_number n}.to_finset.card = 16 := 
by 
  sorry

end count_valid_four_digit_numbers_l606_606184


namespace arc_length_of_polar_curve_l606_606959

theorem arc_length_of_polar_curve :
  let ρ (φ : ℝ) := 4 * real.exp (4 * φ / 3)
  in (∫ φ in 0..(real.pi / 3), real.sqrt ((ρ φ)^2 + (deriv ρ φ)^2)) = (5 / 3) * (real.exp (4 * real.pi / 9) - 1) :=
by
  -- Definitions of functions and variables involved in the conditions
  let ρ := λ φ : ℝ, 4 * real.exp (4 * φ / 3)
  let ρ' := λ φ : ℝ, deriv ρ φ
  have hρ' : ∀ φ, ρ' φ = deriv ρ φ := by simp [ρ]
  -- Integral evaluation related theorem
  sorry

end arc_length_of_polar_curve_l606_606959


namespace domain_of_function_l606_606052

theorem domain_of_function :
  {x : ℝ | 2 - x > 0 ∧ 1 + x > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end domain_of_function_l606_606052


namespace incorrect_major_premise_extremum_l606_606038

theorem incorrect_major_premise_extremum :
  ¬ (∀ (f : ℝ → ℝ) (x_0 : ℝ), (Differentiable ℝ f) → (deriv f x_0 = 0 → ∃ (c : ℝ), IsLocalExtremum f x_0)) := 
by 
  sorry

end incorrect_major_premise_extremum_l606_606038


namespace polygon_vertices_distribution_l606_606341

theorem polygon_vertices_distribution :
  ∀ (n : ℕ) (h : n = 2018) (a b c : ℕ) (d1 : a = 18) (d2 : b = 1018) (d3 : c = 2000),
    a < b ∧ b < c ∧ c ≤ n →
    let polygon1_vertices := b - a + 1,
        polygon2_vertices := c - b + 1,
        polygon3_vertices := (n - c + 1) + a - 1 + 1
    in polygon1_vertices = 1001 ∧ polygon2_vertices = 983 ∧ polygon3_vertices = 38 := 
by
  intros n h a b c d1 d2 d3 hlt,
  rw [hlt.1, hlt.2.1, hlt.2.2] at *,
  unfold polygon1_vertices polygon2_vertices polygon3_vertices,
  calc
    polygon1_vertices = 1001 := by sorry
    polygon2_vertices = 983 := by sorry
    polygon3_vertices = 38 := by sorry

end polygon_vertices_distribution_l606_606341


namespace rose_sweets_problem_l606_606830

theorem rose_sweets_problem :
  ∃ s : ℕ, s ≡ 5 [MOD 6] ∧ s ≡ 3 [MOD 8] ∧ s ≡ 6 [MOD 9] ∧ s ≡ 10 [MOD 11] ∧ s = 2095 :=
by
  use 2095
  split;
  {
    norm_num,
    sorry -- Alternatively norm_num can be used to simplify some common modular arithmetic cases.
  }

end rose_sweets_problem_l606_606830


namespace inequality_proof_l606_606514

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l606_606514


namespace triangle_is_isosceles_l606_606345

theorem triangle_is_isosceles 
  (A B C M K A_1 B_1 : Point)
  (h1 : is_median C M A B)
  (h2 : lies_on K C M)
  (h3 : line_intersects AK BC A_1)
  (h4 : line_intersects BK AC B_1)
  (h5 : is_cyclic_quadrilateral A B_1 A_1 B) 
  : is_isosceles_triangle A B C :=
sorry

end triangle_is_isosceles_l606_606345


namespace rectangle_width_l606_606503

theorem rectangle_width (L W : ℝ) (h1 : 2 * (L + W) = 16) (h2 : W = L + 2) : W = 5 :=
by
  sorry

end rectangle_width_l606_606503


namespace smallest_term_index_l606_606391

theorem smallest_term_index (a_n : ℕ → ℤ) (h : ∀ n, a_n n = 3 * n^2 - 38 * n + 12) : ∃ n, a_n n = a_n 6 ∧ ∀ m, a_n m ≥ a_n 6 :=
by
  sorry

end smallest_term_index_l606_606391


namespace least_integer_gt_sqrt_450_l606_606447

theorem least_integer_gt_sqrt_450 : 
  let x := 450 in
  (21 * 21 = 441) → 
  (22 * 22 = 484) →
  (441 < x ∧ x < 484) →
  (∃ n : ℤ, n = 22 ∧ n > int.sqrt x) :=
by
  intros h1 h2 h3
  use 22
  split
  · refl
  · sorry

end least_integer_gt_sqrt_450_l606_606447


namespace least_integer_gt_sqrt_450_l606_606461

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l606_606461


namespace monotonic_power_function_l606_606149

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := (a^2 - 2 * a - 2) * x^a

theorem monotonic_power_function (a : ℝ) (h1 : ∀ x : ℝ, ( ∀ x1 x2 : ℝ, x1 < x2 → power_function a x1 < power_function a x2 ) )
  (h2 : a^2 - 2 * a - 2 = 1) (h3 : a > 0) : a = 3 :=
by
  sorry

end monotonic_power_function_l606_606149


namespace cone_volume_divided_by_pi_l606_606907

noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * 2 * Real.pi * r

noncomputable def sector_to_cone_radius (arc_len : ℝ) : ℝ := arc_len / (2 * Real.pi)

noncomputable def cone_height (r_base : ℝ) (slant_height : ℝ) : ℝ :=
  Real.sqrt (slant_height ^ 2 - r_base ^ 2)

noncomputable def cone_volume (r_base : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r_base ^ 2 * height

theorem cone_volume_divided_by_pi (r slant_height θ : ℝ) (h : slant_height = 15 ∧ θ = 270):
  cone_volume (sector_to_cone_radius (arc_length r θ)) (cone_height (sector_to_cone_radius (arc_length r θ)) slant_height) / Real.pi = (453.515625 * Real.sqrt 10.9375) :=
by
  sorry

end cone_volume_divided_by_pi_l606_606907


namespace k_m_sum_l606_606323

theorem k_m_sum (k m : ℝ) (h : ∀ {x : ℝ}, x^3 - 8 * x^2 + k * x - m = 0 → x ∈ {1, 2, 5} ∨ x ∈ {1, 3, 4}) :
  k + m = 27 ∨ k + m = 31 :=
by
  sorry

end k_m_sum_l606_606323


namespace solution_l606_606234

noncomputable def z : ℂ := 3 - 4i

theorem solution (z : ℂ) (h : i * z = 4 + 3 * i) : z = 3 - 4 * i :=
by sorry

end solution_l606_606234


namespace gigi_remaining_batches_l606_606117

variable (f b1 tf remaining_batches : ℕ)
variable (f_pos : 0 < f)
variable (batches_nonneg : 0 ≤ b1)
variable (t_f_pos : 0 < tf)
variable (h_f : f = 2)
variable (h_b1 : b1 = 3)
variable (h_tf : tf = 20)

theorem gigi_remaining_batches (h : remaining_batches = (tf - (f * b1)) / f) : remaining_batches = 7 := by
  sorry

end gigi_remaining_batches_l606_606117


namespace turnips_total_l606_606304

theorem turnips_total (Keith_turnips Alyssa_turnips : ℕ) (hK : Keith_turnips = 6) (hA : Alyssa_turnips = 9) : Keith_turnips + Alyssa_turnips = 15 :=
by
  rw [hK, hA]
  exact rfl

end turnips_total_l606_606304


namespace find_increase_l606_606415

-- The cylinders' initial radius and height
def initial_radius : ℝ := 8
def initial_height : ℝ := 3

-- Volume formula for a cylinder
def cylinder_volume (r h : ℝ) : ℝ := real.pi * r^2 * h

-- Increased dimensions
def increased_radius (x : ℝ) : ℝ := initial_radius + x
def increased_height (x : ℝ) : ℝ := initial_height + x

-- Volumes with increased dimensions
def volume_increased_radius (x : ℝ) : ℝ := cylinder_volume (increased_radius x) initial_height
def volume_increased_height (x : ℝ) : ℝ := cylinder_volume initial_radius (increased_height x)

-- The main theorem stating the problem
theorem find_increase (x : ℝ) :
  volume_increased_radius x = volume_increased_height x → x = 16 / 3 := by
  sorry

end find_increase_l606_606415


namespace positive_integers_satisfying_conditions_l606_606106

theorem positive_integers_satisfying_conditions (n : ℕ) (hpos : 1 < n) (hneq3 : n ≠ 3) : 
  ∃ (x : Fin n → ℤ) (y : ℤ), 
    (∑ i, x i = 0) ∧ 
    (∑ i, (x i) ^ 2 = n * y^2) :=
by
  sorry

end positive_integers_satisfying_conditions_l606_606106


namespace chessboard_cut_correct_l606_606834

namespace ChessboardCut

-- Define the type for color
inductive Color
| White
| Black

-- Define the position on the chessboard
structure Position :=
(x : Nat) (y : Nat)

-- Predicate to identify the color of a square at a given position on an 8x8 chessboard
def color_at (p : Position) : Color :=
  if (p.x + p.y) % 2 = 0 then Color.White else Color.Black

-- Predicate to describe the cutting process of the chessboard
def is_cut_correct (cut_positions : List (Position × Position)) : Prop :=
  ∀ p : Position, p.x < 8 ∧ p.y < 8 → 
  match color_at p with
  | Color.White => ∃ q : Position, cut_positions.any (fun r => r.1 = p ∧ r.2 = q)
  | Color.Black => ∃ q1 q2 : Position, q1 ≠ q2 ∧ cut_positions.any (fun r => r.1 = p ∧ r.2 = q1) ∧ cut_positions.any (fun r => r.1 = p ∧ r.2 = q2)

theorem chessboard_cut_correct : 
  ∃ cut_positions : List (Position × Position), is_cut_correct cut_positions :=
sorry

end ChessboardCut

end chessboard_cut_correct_l606_606834


namespace isosceles_triangle_from_median_and_cyclic_quadrilateral_l606_606359

theorem isosceles_triangle_from_median_and_cyclic_quadrilateral
    {A B C M K A1 B1 : Type*}
    (hCM_med : IsMedian A B C M)
    (hK_on_CM : OnLine K CM)
    (hAK_int_BC : IsIntersection AK BC A1)
    (hBK_int_AC : IsIntersection BK AC B1)
    (quad_cyclic : CyclicQuadrilateral A B1 A1 B) 
    : IsIsoscelesTriangle A B C :=
by
  sorry

end isosceles_triangle_from_median_and_cyclic_quadrilateral_l606_606359


namespace sum_of_smallest_integer_values_for_intervals_l606_606990

theorem sum_of_smallest_integer_values_for_intervals :
  (∑ i in {2, 3, 4}, i) = 9 :=
by sorry

end sum_of_smallest_integer_values_for_intervals_l606_606990


namespace num_integral_triangles_perimeter_8_l606_606175

/-- Proving the number of different triangles with integral sides and a perimeter of 8 -/
theorem num_integral_triangles_perimeter_8 : 
  let triangles := {(a, b, c) | a + b + c = 8 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a} in
  finset.card triangles = 8 := 
sorry

end num_integral_triangles_perimeter_8_l606_606175


namespace unique_flavors_l606_606111

noncomputable def distinctFlavors : Nat :=
  let redCandies := 5
  let greenCandies := 4
  let blueCandies := 2
  (90 - 15 - 18 - 30 + 3 + 5 + 6) / 3  -- Adjustments and consideration for equivalent ratios.
  
theorem unique_flavors :
  distinctFlavors = 11 :=
  by
    sorry

end unique_flavors_l606_606111


namespace inequality_xyz_l606_606521

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x))) +
  (y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y))) +
  (z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 :=
by sorry

end inequality_xyz_l606_606521


namespace regular_tetrahedron_has_no_diagonals_l606_606930

/--
In a regular tetrahedron with four vertices and six edges, a segment joining two vertices not joined by an edge is a diagonal. 
Prove that the number of diagonals in a regular tetrahedron is zero.
-/
theorem regular_tetrahedron_has_no_diagonals 
  (V : Finset ℝ) (E : Finset (ℝ × ℝ))
  (hv : V.card = 4) (he : E.card = 6) (hcomplete : ∀ a b ∈ V, a ≠ b → (a, b) ∈ E) :
  ∀ d ∈ V.pairs, ¬(d ∈ E) → d = 0 :=
by
  sorry

end regular_tetrahedron_has_no_diagonals_l606_606930


namespace repeating_decimal_sum_l606_606082

theorem repeating_decimal_sum : (0.\overline{6} + 0.\overline{2} - 0.\overline{4} : ℚ) = (4 / 9 : ℚ) :=
by
  sorry

end repeating_decimal_sum_l606_606082


namespace inequality_xyz_l606_606520

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x))) +
  (y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y))) +
  (z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 :=
by sorry

end inequality_xyz_l606_606520


namespace simplest_sqrt_l606_606498

noncomputable def sqrt32 := real.sqrt 32
noncomputable def sqrt5 := real.sqrt 5
noncomputable def sqrt43 := real.sqrt (4 / 3)
noncomputable def sqrt15 := real.sqrt 1.5

theorem simplest_sqrt : sqrt5 = real.sqrt 5 :=
by sorry

end simplest_sqrt_l606_606498


namespace straight_line_equation_exists_l606_606369

theorem straight_line_equation_exists (x y : ℝ) (l : set (ℝ × ℝ)) :
  (∃ (a b c : ℝ), (∀ p ∈ l, a * p.1 + b * p.2 + c = 0) ∧ (a ≠ 0 ∨ b ≠ 0)) :=
sorry

end straight_line_equation_exists_l606_606369


namespace cinderella_prevents_overflow_l606_606509

-- Definitions and conditions
def bucket_capacity : ℝ := 2  -- 2 liters capacity per bucket
def initial_buckets : ℕ → ℝ := λ i, 0  -- All buckets start empty

-- Stepmother's move: adds 1 liter of water arbitrarily distributed
def stepmother_move (buckets : ℕ → ℝ) : ℕ → ℝ := sorry 

-- Cinderella's move: chooses a pair of neighboring buckets to empty
def cinderella_move (buckets : ℕ → ℝ) : ℕ → ℝ := sorry 

-- Goal: Prove that Cinderella can always prevent overflow
theorem cinderella_prevents_overflow :
  ∀ (rounds : ℕ), 
    let moves := λ buckets, (stepmother_move ∘ cinderella_move)^[rounds] buckets in
    ∀ (i : ℕ), moves initial_buckets i < bucket_capacity := sorry

end cinderella_prevents_overflow_l606_606509


namespace num_integral_triangles_perimeter_8_l606_606176

/-- Proving the number of different triangles with integral sides and a perimeter of 8 -/
theorem num_integral_triangles_perimeter_8 : 
  let triangles := {(a, b, c) | a + b + c = 8 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a} in
  finset.card triangles = 8 := 
sorry

end num_integral_triangles_perimeter_8_l606_606176


namespace fifth_term_seq_l606_606169

-- Define the initial conditions and recursive relation for the sequence
def seq : ℕ → ℤ
| 1 := 3
| 2 := 6
| (n + 2) := seq (n + 1) - seq n

-- The theorem to prove that the fifth term of the sequence is -6
theorem fifth_term_seq : seq 5 = -6 :=
sorry

end fifth_term_seq_l606_606169


namespace number_of_elements_after_14_transformations_l606_606922

def lattice_point := (ℤ × ℤ)

def transform (S : set lattice_point) : set lattice_point :=
{p : lattice_point | ∃ (dx dy : ℤ), (dx = 0 ∧ dy = 0 ∨ dx = -1 ∨ dx = 1 ∨ dy = -1 ∨ dy = 1) ∧ 
  (p.1 - dx, p.2 - dy) ∈ S }

def apply_transform_n_times (S : set lattice_point) (n : ℕ) : set lattice_point :=
nat.rec_on n S (λ _ S', transform S')

theorem number_of_elements_after_14_transformations :
  ∀ (S : set lattice_point), S = {(0, 0)} → set.card (apply_transform_n_times S 14) = 421 :=
by {
  intros,
  admit, -- proof here
}

end number_of_elements_after_14_transformations_l606_606922


namespace problem_statement_l606_606264

noncomputable def sum_of_solutions_with_negative_imaginary_part : ℂ :=
  let x3 := 2 * complex.of_real (real.cos (195 * real.pi / 180)) + 2 * complex.I * complex.of_real (real.sin (195 * real.pi / 180)),
      x4 := 2 * complex.of_real (real.cos (255 * real.pi / 180)) + 2 * complex.I * complex.of_real (real.sin (255 * real.pi / 180)),
      x5 := 2 * complex.of_real (real.cos (315 * real.pi / 180)) + 2 * complex.I * complex.of_real (real.sin (315 * real.pi / 180))
  in x3 + x4 + x5

theorem problem_statement : sum_of_solutions_with_negative_imaginary_part =
  2 * (complex.of_real (real.cos (195 * real.pi / 180)) + complex.of_real (real.cos (255 * real.pi / 180)) + complex.of_real (real.cos (315 * real.pi / 180))) +
  2 * complex.I * (complex.of_real (real.sin (195 * real.pi / 180)) + complex.of_real (real.sin (255 * real.pi / 180)) + complex.of_real (real.sin (315 * real.pi / 180))) :=
sorry

end problem_statement_l606_606264


namespace isosceles_triangle_l606_606349

-- Definitions
variables {A B C M K A1 B1 : Type*}
variables (point : A) (point B) (point C) (point M) (point K) (point A1) (point B1)
variable (triangle : A → B → C → Type*)
variable (median : triangle.point → point → Type*)

-- Conditions
variable (is_median_CM : median C M)
variable (is_on_CM : point K ∈ median C M)
variable (intersects_AK_BC : point A1 = AK ∩ BC)
variable (intersects_BK_AC : point B1 = BK ∩ AC)
variable (inscribed_AB1A1B : circle AB1A1B)

-- Proof
theorem isosceles_triangle (h1 : is_median_CM) (h2 : is_on_CM)
  (h3 : intersects_AK_BC) (h4 : intersects_BK_AC)
  (h5 : inscribed_AB1A1B) :
  AB = AC :=
by
  sorry

end isosceles_triangle_l606_606349


namespace watermelon_does_not_necessarily_split_l606_606001

theorem watermelon_does_not_necessarily_split (R : ℝ) (h : ℝ) (d : ℝ) :
    R = 10 → (h = 17 ∨ h = 18) → ¬ ∀ (cuts : ℕ), cuts = 3 → 
          (∀ (planar_cut : ℝ → ℝ), planar_cut = h → (exists segment_height: ℝ, segment_height = h)) → false :=
by
  intro radius_eq heights_eq cuts_eq all_cuts
  -- proof will go here
  sorry

end watermelon_does_not_necessarily_split_l606_606001


namespace least_int_gt_sqrt_450_l606_606431

theorem least_int_gt_sqrt_450 : ∃ n : ℕ, n > nat.sqrt 450 ∧ (∀ m : ℕ, m > nat.sqrt 450 → n ≤ m) :=
by
  have h₁ : 20^2 = 400 := by norm_num
  have h₂ : 21^2 = 441 := by norm_num
  have h₃ : 22^2 = 484 := by norm_num
  have sqrt_bounds : 21 < real.sqrt 450 ∧ real.sqrt 450 < 22 :=
    by sorry
  use 22
  split
  · sorry
  · sorry

end least_int_gt_sqrt_450_l606_606431


namespace find_face_value_l606_606895

-- Define the conditions
def cash_realized (FV : ℝ) : ℝ := FV - (FV * 1 / 400)

-- Statement of the proof problem
theorem find_face_value:
  (∀ FV : ℝ, cash_realized FV = 108.25 → FV ≈ 108.52) :=
sorry

end find_face_value_l606_606895


namespace parabola_tangent_AB_length_l606_606167

noncomputable def length_of_tangent_segment 
    (p : ℝ) (hp : p > 0) : ℝ :=
if h : p > 0 then 2 * p else 0

theorem parabola_tangent_AB_length 
    (p : ℝ) (hp : p > 0)
    (A B : ℝ × ℝ)
    (parabola : ℝ × ℝ → Prop := 
       λ ⟨x, y⟩, x^2 = 2 * p * y)
    (M : ℝ × ℝ := ⟨0, -p / 2⟩)
    (is_tangent : (ℝ × ℝ) → Prop := 
       λ ⟨x, y⟩, (y = x - p / 2) ∨ (y = -x - p / 2)) :
    dist A B = 2 * p := sorry

end parabola_tangent_AB_length_l606_606167


namespace percentage_error_in_area_l606_606011

theorem percentage_error_in_area {x : ℝ} (hx : x > 0) :
    let measured_side := 1.12 * x
    let actual_area := x^2
    let erroneous_area := (1.12 * x)^2
    let percentage_error := ((erroneous_area - actual_area) / actual_area) * 100
  in percentage_error = 25.44 :=
by
  sorry

end percentage_error_in_area_l606_606011


namespace day_crew_fraction_loaded_l606_606889

-- Let D be the number of boxes loaded by each worker on the day crew
-- Let W_d be the number of workers on the day crew
-- Let W_n be the number of workers on the night crew
-- Let B_d be the total number of boxes loaded by the day crew
-- Let B_n be the total number of boxes loaded by the night crew

variable (D W_d : ℕ) 
variable (B_d := D * W_d)
variable (W_n := (4 / 9 : ℚ) * W_d)
variable (B_n := (3 / 4 : ℚ) * D * W_n)
variable (total_boxes := B_d + B_n)

theorem day_crew_fraction_loaded : 
  (D * W_d) / (D * W_d + (3 / 4 : ℚ) * D * ((4 / 9 : ℚ) * W_d)) = (3 / 4 : ℚ) := sorry

end day_crew_fraction_loaded_l606_606889


namespace find_k_a_l606_606974

-- Define the conditions
def polynomial := (x : ℝ) → x^4 - 5*x^3 + 13*x^2 - 19*x + 8
def divisor (k : ℝ) := (x : ℝ) → x^2 - 3*x + k
def remainder (a : ℝ) := (x : ℝ) → 2*x + a

-- State the mathematically equivalent proof problem
theorem find_k_a (k a : ℝ) :
  ∀ (x : ℝ), polynomial(x) = (divisor k)(x) * quotient(x) + (remainder a)(x) →
  k = 5 / 2 ∧ a = -13 / 4 :=
sorry -- Proof to be provided

end find_k_a_l606_606974


namespace find_interest_rate_l606_606107

noncomputable def interest_rate (A P T : ℚ) : ℚ := (A - P) / (P * T) * 100

theorem find_interest_rate :
  let A := 1120
  let P := 921.0526315789474
  let T := 2.4
  interest_rate A P T = 9 := 
by
  sorry

end find_interest_rate_l606_606107


namespace sum_product_leq_a_squared_div_four_l606_606318

theorem sum_product_leq_a_squared_div_four {n : ℕ} (a : ℝ) (a_i : ℕ → ℝ)
  (h_nonneg : ∀ i, 1 ≤ i → i ≤ n → 0 ≤ a_i i)
  (h_sum : (∑ i in Finset.range n, a_i i) = a) :
  (∑ i in Finset.range (n-1), a_i i * a_i (i+1)) ≤ a^2 / 4 := 
by
  sorry

end sum_product_leq_a_squared_div_four_l606_606318


namespace inequality_xyz_l606_606518

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x))) +
  (y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y))) +
  (z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 :=
by sorry

end inequality_xyz_l606_606518


namespace arithmetic_sequence_part_a_arithmetic_sequence_part_b_l606_606002

theorem arithmetic_sequence_part_a (e u k : ℕ) (n : ℕ) 
  (h1 : e = 1) 
  (h2 : u = 1000) 
  (h3 : k = 343) 
  (h4 : n = 100) : ¬ (∃ d m, e + (m - 1) * d = k ∧ u = e + (n - 1) * d ∧ 1 < m ∧ m < n) :=
by sorry

theorem arithmetic_sequence_part_b (e u k : ℝ) (n : ℕ) 
  (h1 : e = 81 * Real.sqrt 2 - 64 * Real.sqrt 3) 
  (h2 : u = 54 * Real.sqrt 2 - 28 * Real.sqrt 3)
  (h3 : k = 69 * Real.sqrt 2 - 48 * Real.sqrt 3)
  (h4 : n = 100) : (∃ d m, e + (m - 1) * d = k ∧ u = e + (n - 1) * d ∧ 1 < m ∧ m < n) :=
by sorry

end arithmetic_sequence_part_a_arithmetic_sequence_part_b_l606_606002


namespace count_3_digit_numbers_divisible_by_11_l606_606192

/-- 
  Define the mathematical conditions.
-/
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def is_divisible_by (n m : ℕ) : Prop := n % m = 0

/--
  Define the problem statement to prove that the count of 3-digit numbers divisible by 11 is 82.
-/
theorem count_3_digit_numbers_divisible_by_11 : 
  let nums := { n | smallest_3_digit ≤ n ∧ n ≤ largest_3_digit ∧ is_divisible_by n 11},
      count := nums.card
  in count = 82 := by
{
  sorry
}

end count_3_digit_numbers_divisible_by_11_l606_606192


namespace imaginary_part_of_fraction_l606_606395

open Complex

theorem imaginary_part_of_fraction : 
  let z := (2 + 4 * I) / (1 + I) in
  z.im = 1 :=
by
  sorry

end imaginary_part_of_fraction_l606_606395


namespace math_problem_l606_606630

-- Define the parametric equations of circle C
def parametric_eq_circle (φ : ℝ) : ℝ × ℝ := 
  (1 + Real.cos φ, Real.sin φ)

-- Define the polar equation of line l
def polar_eq_line (ρ θ : ℝ) : Prop := 
  2 * ρ * Real.sin (θ + π / 3) = 3 * Real.sqrt 3

-- Define the polar equation of the circle we need to prove
def polar_eq_circle (ρ θ : ℝ) : Prop := 
  ρ = 2 * Real.cos θ

-- Define the length of PQ based on given conditions
def length_PQ : ℝ :=
  let ρ1 := 1 -- From θ1= π / 3 and ρ1 = 2 * cos(θ1)
  let ρ2 := 3 -- From θ2= π / 3 and solving polar_eq_line
  abs(ρ1 - ρ2)

-- Main theorem statement combining both parts
theorem math_problem :
  (∀ ρ θ, polar_eq_circle ρ θ -> ρ = 2 * Real.cos θ) ∧ 
  (length_PQ = 2) :=
  by
    sorry

end math_problem_l606_606630


namespace recurring_decimal_sum_l606_606074

theorem recurring_decimal_sum :
  (0.666666...:ℚ) + (0.222222...:ℚ) - (0.444444...:ℚ) = 4 / 9 :=
begin
  sorry
end

end recurring_decimal_sum_l606_606074


namespace transform_to_all_ones_l606_606036

theorem transform_to_all_ones :
  ∀ (A : ℕ → ℕ → ℤ),
    (∀ i j, 1 ≤ i ∧ i ≤ 8 ∧ 1 ≤ j ∧ j ≤ 8 → A i j = 1 ∨ A i j = -1) →
    ∃ M : ℕ, ∃ moves : Fin M → (ℕ × ℕ),
      (∀ i j, 1 ≤ i ∧ i ≤ 8 ∧ 1 ≤ j ∧ j ≤ 8 → 
        (iter_moves A moves i j = 1)) :=
by
  sorry

-- Helper function representing the effect of a sequence of moves
noncomputable def iter_moves (A : ℕ → ℕ → ℤ) (moves : Fin M → (ℕ × ℕ)) : ℕ → ℕ → ℤ :=
  sorry

end transform_to_all_ones_l606_606036


namespace PS_length_l606_606758

-- Definitions related to the problem
structure Triangle (P Q R : Type) :=
  (PQ QR PR : ℝ)
  (angleQ : ℝ)
  (right_angle : angleQ = π / 2)
  (PQ_val : PQ = 3)
  (QR_val : QR = 3 * Real.sqrt 3)

-- Definitions related to the bisector
def angle_bisector (P Q R S : Type) [Triangle P Q R] := sorry

-- The statement we need to prove
theorem PS_length (P Q R S : Type) [Triangle P Q R] :
  Triangle.PS = 6 * Real.sqrt 3 - 12 :=
sorry

end PS_length_l606_606758


namespace inequality_example_l606_606668

theorem inequality_example (a b c : ℝ) (hac : a ≠ 0) (hbc : b ≠ 0) (hcc : c ≠ 0) :
  (a^4) / (4 * a^4 + b^4 + c^4) + (b^4) / (a^4 + 4 * b^4 + c^4) + (c^4) / (a^4 + b^4 + 4 * c^4) ≤ 1 / 2 :=
sorry

end inequality_example_l606_606668


namespace c_value_for_infinite_solutions_l606_606626

theorem c_value_for_infinite_solutions :
  ∀ (c : ℝ), (∀ (x : ℝ), 3 * (5 + c * x) = 15 * x + 15) ↔ c = 5 :=
by
  -- Proof
  sorry

end c_value_for_infinite_solutions_l606_606626


namespace sum_expression_eq_62_l606_606681

noncomputable def a : ℕ → ℝ
| 0       := b 2015 
| (n + 1) := (1 / 65) * Real.sqrt (2 * (n + 1) + 2) + a n

noncomputable def b : ℕ → ℝ
| 0       := a 2015
| (n + 1) := (1 / 1009) * Real.sqrt (2 * (n + 1) + 2) - b n

theorem sum_expression_eq_62 : 
  (∑ k in Finset.range 2015 + 1, a (k + 1) * b k - a k * b (k + 1)) = 62 :=
sorry

end sum_expression_eq_62_l606_606681


namespace shaded_region_area_l606_606414

theorem shaded_region_area
  (O : Point)
  (r₁ r₂ : ℝ)
  (h1 : r₁ = 40)
  (h2 : r₂ = sqrt (1600 + 3600))
  (AB : LineSegment)
  (AB_length : ℝ)
  (h3 : AB_length = 120)
  (tangent : Tangent AB O r₁) :
  π * (r₂^2 - r₁^2) = 3600 * π := by
  sorry

end shaded_region_area_l606_606414


namespace largest_x_undefined_l606_606426

theorem largest_x_undefined :
  ∃ x, (8 * x^2 - 65 * x + 8 = 0) ∧ (∀ y, (8 * y^2 - 65 * y + 8 = 0) → y ≤ x) :=
begin
  use 8,
  split,
  { -- Proof that 8 is a root of the equation
    have h : (8 * 8^2 - 65 * 8 + 8) = 0,
    { norm_num },
    exact h,
  },
  { -- Proof that 8 is the largest value
    intros y h_zero,
    apply le_of_not_gt,
    intro h,
    cases h : 8 * y^2 - 65 * y + 8,
    { norm_num at h_zero },
    { linarith }
  }
end

end largest_x_undefined_l606_606426


namespace least_integer_gt_sqrt_450_l606_606459

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l606_606459


namespace problem1_problem2_l606_606153

-- Define the conditions and the corresponding proofs

theorem problem1 (m n : ℝ) :
  (∀ (x : ℝ), (|2 * x - 3| < x) ↔ (x^2 - m * x + n < 0)) → m - n = 1 := by
  sorry

theorem problem2 (a b c : ℝ) (m n : ℝ) :
  (ab + bc + ac = m - n) → (m - n = 1) → a ∈ (0, 1) → b ∈ (0, 1) → c ∈ (0, 1) → a + b + c ≥ sqrt 3 := by
  sorry

end problem1_problem2_l606_606153


namespace least_integer_greater_than_sqrt_450_l606_606475

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, (n > real.sqrt 450) ∧ ∀ m : ℤ, (m > real.sqrt  450) → n ≤ m :=
by
  apply exists.intro 22
  sorry

end least_integer_greater_than_sqrt_450_l606_606475


namespace circle_areas_equal_volume_ratio_condition_l606_606935
noncomputable

variables (R m n : ℝ)

-- Problem 1: Proving the equality of circle areas for given distances
theorem circle_areas_equal (m : ℝ) (R : ℝ) (h : m = 2*R/5 ∨ m = 2*R) :
    let r_sphere := sqrt (2*R*m - m^2)
    let r_cone := R - m/2
    r_sphere^2 = r_cone^2 := by
  sorry

-- Problem 2: Proving volume condition for values of n
theorem volume_ratio_condition (R m n : ℝ) (h : 2*R / 5 ≤ m ∧ m ≤ 2*R) :
    ∀ n, (n >= 1/2) → 
      let vol_truncated_cone := π*m*(R^2 + R*(R - m/2) + (R - m/2)^2) / 3
      let vol_spherical_segment := π*m^2*(3*R - m) / 3
      vol_truncated_cone = n * vol_spherical_segment := by
  sorry

end circle_areas_equal_volume_ratio_condition_l606_606935


namespace solve_for_z_l606_606241

variable {z : ℂ}

theorem solve_for_z (h : complex.I * z = 4 + 3*complex.I) : z = 3 - 4*complex.I :=
sorry

end solve_for_z_l606_606241


namespace quadrilateral_circle_condition_l606_606014

variables {A B C D G H E F : Point}
variable (ABCD : convexQuadrilateral A B C D)
variable (circle : Circle)
variable (tangentG : circle.tangentAt G AB)
variable (tangentH : circle.tangentAt H BC)
variable (intersectE : circle.intersectsAt E AC)
variable (intersectF : circle.intersectsAt F AC)
variable (extensionDA : LineExtension DA)
variable (extensionDC : LineExtension DC)

theorem quadrilateral_circle_condition :
  (AB + AD = BC + CD) ↔ (∃ anotherCircle : Circle, anotherCircle.passesThrough E ∧ anotherCircle.passesThrough F ∧ anotherCircle.tangentAtExtension DA ∧ anotherCircle.tangentAtExtension DC) :=
sorry  -- Proof goes here.

end quadrilateral_circle_condition_l606_606014


namespace value_of_f_range_and_symmetry_l606_606695

-- Definition of the function f
noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (x + Real.pi / 6) - 2 * Real.cos x

open Real

-- Define condition variables
variables (x : ℝ) (h₁ : sin x = 4 / 5) (h₂ : x ∈ Icc (pi / 2) pi)

-- Statement for part 1: value of f(x)
theorem value_of_f (x : ℝ) (h₁ : sin x = 4 / 5) (h₂ : x ∈ Icc (pi / 2) pi) :
  f x = (4 * sqrt 3 + 3) / 5 :=
sorry

-- Statement for part 2: range and axis of symmetry of f(x)
theorem range_and_symmetry (x : ℝ) :
  (∀ y, y ∈ Set.range f ↔ y ∈ Set.Icc (1 : ℝ) (2 : ℝ)) ∧
  (x = 2 * pi / 3) :=
sorry

end value_of_f_range_and_symmetry_l606_606695


namespace least_integer_greater_than_sqrt_450_l606_606471

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606471


namespace medicine_dose_per_part_l606_606574

-- Define the given conditions
def kg_weight : ℕ := 30
def ml_per_kg : ℕ := 5
def parts : ℕ := 3

-- The theorem statement
theorem medicine_dose_per_part : 
  (kg_weight * ml_per_kg) / parts = 50 :=
by
  sorry

end medicine_dose_per_part_l606_606574


namespace odd_function_f_l606_606164

variable {R : Type} [OrderedRing R]

noncomputable def f (x : R) : R :=
  if x > 0 then x + 2 * x^2
  else -f (-x)

theorem odd_function_f (x : R) (h : x < 0) : f x = x - 2 * x^2 := by
  sorry

end odd_function_f_l606_606164


namespace isosceles_triangle_from_median_and_cyclic_quadrilateral_l606_606358

theorem isosceles_triangle_from_median_and_cyclic_quadrilateral
    {A B C M K A1 B1 : Type*}
    (hCM_med : IsMedian A B C M)
    (hK_on_CM : OnLine K CM)
    (hAK_int_BC : IsIntersection AK BC A1)
    (hBK_int_AC : IsIntersection BK AC B1)
    (quad_cyclic : CyclicQuadrilateral A B1 A1 B) 
    : IsIsoscelesTriangle A B C :=
by
  sorry

end isosceles_triangle_from_median_and_cyclic_quadrilateral_l606_606358


namespace sluice_fill_time_l606_606953

noncomputable def sluice_open_equal_time (x y : ℝ) (m : ℝ) : ℝ :=
  -- Define time (t) required for both sluice gates to be open equally to fill the lake
  m / 11

theorem sluice_fill_time :
  ∀ (x y : ℝ),
    (10 * x + 14 * y = 9900) →
    (18 * x + 12 * y = 9900) →
    sluice_open_equal_time x y 9900 = 900 := sorry

end sluice_fill_time_l606_606953


namespace least_integer_gt_sqrt_450_l606_606465

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l606_606465


namespace problem_solution_l606_606290

open Real

-- Definitions based on conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

def line_eq (t : ℝ) : ℝ × ℝ := (-(sqrt 3 / 2) * t, 2 + t / 2)

-- Statement to prove
theorem problem_solution :
  (exists θ, (2, pi/2) = (ρ, θ) ∧ ρ = 4*sin θ) ∧
  let t1 := 2 in let t2 := -2 in let t0 := -4 in
  abs (t1 - t0) + abs (t2 - t0) = 8 :=
by
  sorry

end problem_solution_l606_606290


namespace recurring_decimal_sum_l606_606076

theorem recurring_decimal_sum :
  (0.666666...:ℚ) + (0.222222...:ℚ) - (0.444444...:ℚ) = 4 / 9 :=
begin
  sorry
end

end recurring_decimal_sum_l606_606076


namespace solve_for_z_l606_606237

variable {z : ℂ}

theorem solve_for_z (h : complex.I * z = 4 + 3*complex.I) : z = 3 - 4*complex.I :=
sorry

end solve_for_z_l606_606237


namespace least_integer_gt_sqrt_450_l606_606449

theorem least_integer_gt_sqrt_450 : 
  let x := 450 in
  (21 * 21 = 441) → 
  (22 * 22 = 484) →
  (441 < x ∧ x < 484) →
  (∃ n : ℤ, n = 22 ∧ n > int.sqrt x) :=
by
  intros h1 h2 h3
  use 22
  split
  · refl
  · sorry

end least_integer_gt_sqrt_450_l606_606449


namespace math_problem_l606_606962

theorem math_problem (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 30) :
  (∃ (count : ℕ), (∀ k, 1 ≤ k ∧ k ≤ 30 →
  (nat.factorial (k^3 - 1) / (nat.factorial k)^(k^2)) ∈ ℤ ↔ k = 1) ∧ count = 1) :=
sorry

end math_problem_l606_606962


namespace evaluate_expression_l606_606985

theorem evaluate_expression :
  let x := 1.93
  let y := 51.3
  let z := 0.47
  Float.round (x * (y + z)) = 100 := by
sorry

end evaluate_expression_l606_606985


namespace boxed_boxed_15_l606_606113

def sigma (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ i => n % i = 0).sum id

theorem boxed_boxed_15 : sigma (sigma 15) = 60 := by
  sorry

end boxed_boxed_15_l606_606113


namespace scientific_notation_of_150000000000_l606_606860

theorem scientific_notation_of_150000000000 :
  150000000000 = 1.5 * 10^11 :=
sorry

end scientific_notation_of_150000000000_l606_606860


namespace product_approximation_l606_606605

theorem product_approximation :
  (3.05 * 7.95 * (6.05 + 3.95)) = 240 := by
  sorry

end product_approximation_l606_606605


namespace hiking_packing_weight_l606_606590

theorem hiking_packing_weight
  (miles_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days : ℝ)
  (supply_per_mile : ℝ)
  (resupply_fraction : ℝ)
  (expected_first_pack_weight : ℝ)
  (hiking_hours : ℝ := hours_per_day * days)
  (total_miles : ℝ := miles_per_hour * hiking_hours)
  (total_weight_needed : ℝ := supply_per_mile * total_miles)
  (resupply_weight : ℝ := resupply_fraction * total_weight_needed)
  (first_pack_weight : ℝ := total_weight_needed - resupply_weight) :
  first_pack_weight = 37.5 :=
by
  -- The proof goes here, but is omitted since the proof is not required.
  sorry

end hiking_packing_weight_l606_606590


namespace find_z_l606_606245

theorem find_z (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : z = 3 - 4 * complex.I :=
by {
  sorry
}

end find_z_l606_606245


namespace expand_expression_l606_606061

theorem expand_expression : 
  ∀ (x : ℝ), (7 * x^3 - 5 * x + 2) * 4 * x^2 = 28 * x^5 - 20 * x^3 + 8 * x^2 :=
by
  intros x
  sorry

end expand_expression_l606_606061


namespace derivative_at_zero_l606_606122

def f (f' : ℝ → ℝ) (x : ℝ) : ℝ := x^2 + 2*x * f' 1

theorem derivative_at_zero (f' : ℝ → ℝ) (h : ∀ x, HasDerivAt (λ x, x^2 + 2*x * f' 1) (2*x + 2 * f'(1)) x) :
  f' 0 = -4 :=
sorry

end derivative_at_zero_l606_606122


namespace people_got_off_l606_606407

theorem people_got_off (initial left got_off : ℕ) (h_initial : initial = 48) (h_left : left = 31) : got_off = 17 :=
by
  have h_got_off : got_off = initial - left := by
    sorry
  rw [h_initial, h_left] at h_got_off
  exact h_got_off
  sorry

end people_got_off_l606_606407


namespace cond_seq_a_S_geom_seq_cond_prove_a_n_prove_b_n_prove_sum_T_prove_sum_c_k_c_k1_l606_606170

-- Definitions and conditions from part a)
def seq_a (n : ℕ+) : ℕ := 3 ^ n
def seq_b (n : ℕ+) : ℕ := 2 * n - 1
def sum_S (S : ℕ+ → ℕ) (n : ℕ+) : ℕ := S n

-- Known relationships
theorem cond_seq_a_S (n : ℕ+) (S : ℕ+ → ℕ) : 3 * seq_a n = 2 * sum_S S n + 3 :=
sorry

theorem geom_seq_cond (b1 b2 b4 b6 : ℕ) : 
  b2 + 5 = b1 → b4 + 1 = b2 → b6 - 3 = b4 → 
  (b4 + 1) ^ 2 = (b2 + 5) * (b6 - 3) :=
sorry

-- The proof problem asked in part c)
theorem prove_a_n (n : ℕ+) : seq_a n = 3 ^ n :=
sorry

theorem prove_b_n (n : ℕ+) : seq_b n = 2 * n - 1 :=
sorry

noncomputable def dn (n : ℕ+) : ℚ := 
  (seq_b (n + 2) - 1 : ℚ) / ((seq_b n : ℚ) * (seq_b (n + 1) : ℚ) * (seq_a n : ℚ))

theorem prove_sum_T (n : ℕ+) : 
  (finset.range n).sum (λ k, dn (k + 1)) = 1 / 2 - 1 / (2 * (2 * n + 1) * 3 ^ n) :=
sorry

def cn (n : ℕ+) : ℕ := 
  if n % 2 = 1 then seq_a n else seq_b (n / 2)

theorem prove_sum_c_k_c_k1 (n : ℕ+) : 
  (finset.range (2 * n)).sum (λ k, cn (k + 1) * cn (k + 2)) = 
  (75 / 16 : ℚ) + (40 * n - 25) / 48 * (9 ^ (n + 1)) :=
sorry

end cond_seq_a_S_geom_seq_cond_prove_a_n_prove_b_n_prove_sum_T_prove_sum_c_k_c_k1_l606_606170


namespace amy_total_score_correct_l606_606006

def amyTotalScore (points_per_treasure : ℕ) (treasures_first_level : ℕ) (treasures_second_level : ℕ) : ℕ :=
  (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level)

theorem amy_total_score_correct:
  amyTotalScore 4 6 2 = 32 :=
by
  -- Proof goes here
  sorry

end amy_total_score_correct_l606_606006


namespace AC_value_l606_606727

variable (ABC : Type) [triangle ABC]
variable (BC : ℝ) (B : ℝ) (area_ABC : ℝ)

theorem AC_value 
  (h1 : BC = 2)
  (h2 : B = Real.pi / 3)
  (h3 : area_ABC = Real.sqrt 3 / 2) :
  AC = Real.sqrt 3 := 
  sorry

end AC_value_l606_606727


namespace counterfeit_coin_is_C_l606_606005

-- Given coins A, B, C, D, E, and F
variables (A B C D E F : ℝ)

-- There are 5 genuine coins and 1 counterfeit coin among A, B, C, D, E, and F.
def is_genuine (x : ℝ) : Prop := x ≠ C
def is_counterfeit (x : ℝ) : Prop := x = C

-- The combined weight of coins A and B is 10 grams
axiom weight_AB : A + B = 10

-- The combined weight of coins C and D is 11 grams
axiom weight_CD : C + D = 11

-- The combined weight of coins A, C, and E is 16 grams
axiom weight_ACE : A + C + E = 16

-- The 5 genuine coins have the same weight
constant w : ℝ
-- weight for genuine coins
axiom genuine_weight_A : is_genuine A → A = w
axiom genuine_weight_B : is_genuine B → B = w
axiom genuine_weight_E : is_genuine E → E = w

-- The counterfeit coin has a different weight
axiom counterfeit_weight : ∀ x, is_counterfeit x → x ≠ w

-- Proof that C is the counterfeit coin
theorem counterfeit_coin_is_C : is_counterfeit C :=
sorry

end counterfeit_coin_is_C_l606_606005


namespace largest_prime_factor_of_sum_of_divisors_of_180_l606_606781

-- Define the function to compute the sum of divisors
noncomputable def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ k in (Finset.range (n + 1)).filter (λ k, n % k = 0), k

-- Define a function to find the largest prime factor of a number
noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  Finset.max' (Finset.filter Nat.prime (Finset.range (n + 1))) sorry

-- Define the problem conditions
def N : ℕ := sum_of_divisors 180

-- State the main theorem to be proved
theorem largest_prime_factor_of_sum_of_divisors_of_180 : largest_prime_factor N = 13 :=
by sorry

end largest_prime_factor_of_sum_of_divisors_of_180_l606_606781


namespace common_fraction_l606_606087

noncomputable def x : ℚ := 0.6666666 -- represents 0.\overline{6}
noncomputable def y : ℚ := 0.2222222 -- represents 0.\overline{2}
noncomputable def z : ℚ := 0.4444444 -- represents 0.\overline{4}

theorem common_fraction :
  x + y - z = 4 / 9 :=
by
  -- Provide proofs here
  sorry

end common_fraction_l606_606087


namespace tensor_calculation_jiaqi_statement_l606_606966

def my_tensor (a b : ℝ) : ℝ := a * (1 - b)

theorem tensor_calculation :
  my_tensor (1 + Real.sqrt 2) (Real.sqrt 2) = -1 := 
by
  sorry

theorem jiaqi_statement (a b : ℝ) (h : a + b = 0) :
  my_tensor a a + my_tensor b b = 2 * a * b := 
by
  sorry

end tensor_calculation_jiaqi_statement_l606_606966


namespace gallop_waddle_difference_l606_606822

theorem gallop_waddle_difference :
  let gaps := 30,
      total_distance := 3720,
      percy_waddles := 36 * gaps,
      zelda_gallops := 15 * gaps,
      percy_waddle_length := total_distance / percy_waddles,
      zelda_gallop_length := total_distance / zelda_gallops
  in zelda_gallop_length - percy_waddle_length = (31 / 15) := 
by
  sorry

end gallop_waddle_difference_l606_606822


namespace recurring_decimal_sum_l606_606078

theorem recurring_decimal_sum :
  (0.666666...:ℚ) + (0.222222...:ℚ) - (0.444444...:ℚ) = 4 / 9 :=
begin
  sorry
end

end recurring_decimal_sum_l606_606078


namespace total_oranges_picked_l606_606817

theorem total_oranges_picked (mary_oranges : Nat) (jason_oranges : Nat) (hmary : mary_oranges = 122) (hjason : jason_oranges = 105) : mary_oranges + jason_oranges = 227 := by
  sorry

end total_oranges_picked_l606_606817


namespace unique_functions_satisfy_l606_606105

noncomputable def satisfying_functions (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x) * f(y) + f(x + y) = x * y

theorem unique_functions_satisfy :
  ∀ f : ℝ → ℝ, satisfying_functions f ↔ (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = x + 1) :=
by
  sorry

end unique_functions_satisfy_l606_606105


namespace non_periodic_length_l606_606655

noncomputable def vp(n : ℕ, p : ℕ) : ℕ :=
  if h : p > 1 then
    @nat.find (λ m, ¬ p ^ (m + 1) ∣ n) (nat.exists_not_dvd_of_dvd h)
  else 0

theorem non_periodic_length {n : ℕ} (h : n > 1) :
  ∃ l, l = max (vp n 2) (vp n 5) :=
by
  sorry

end non_periodic_length_l606_606655


namespace modulus_of_complex_l606_606154

-- Define the conditions
def complex_number (z : ℂ) : Prop :=
  z * (1 - complex.i) ^ 2 = 3 - 4 * complex.i

-- Lean 4 statement to express the proof problem
theorem modulus_of_complex (z : ℂ) (h : complex_number z) : complex.abs z = 5 / 2 := 
by sorry

end modulus_of_complex_l606_606154


namespace area_of_picture_l606_606195

theorem area_of_picture {x y : ℕ} (hx : x > 1) (hy : y > 1) 
  (h : (2 * x + 3) * (y + 2) - x * y = 34) : x * y = 8 := 
by
  sorry

end area_of_picture_l606_606195


namespace least_integer_gt_sqrt_450_l606_606448

theorem least_integer_gt_sqrt_450 : 
  let x := 450 in
  (21 * 21 = 441) → 
  (22 * 22 = 484) →
  (441 < x ∧ x < 484) →
  (∃ n : ℤ, n = 22 ∧ n > int.sqrt x) :=
by
  intros h1 h2 h3
  use 22
  split
  · refl
  · sorry

end least_integer_gt_sqrt_450_l606_606448


namespace train_speed_l606_606939

noncomputable def original_speed_of_train (v d : ℝ) : Prop :=
  (120 ≤ v / (5/7)) ∧
  (2 * d) / (5 * v) = 65 / 60 ∧
  (2 * (d - 42)) / (5 * v) = 45 / 60

theorem train_speed (v d : ℝ) (h : original_speed_of_train v d) : v = 50.4 :=
by sorry

end train_speed_l606_606939


namespace variance_of_white_balls_l606_606272

section
variable (n : ℕ := 7) 
variable (p : ℚ := 3/7)

def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_white_balls : binomial_variance n p = 12/7 :=
by
  sorry
end

end variance_of_white_balls_l606_606272


namespace find_n_plus_m_l606_606694

noncomputable def f (x : ℝ) := abs (Real.log x / Real.log 2)

theorem find_n_plus_m (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n)
    (h4 : f m = f n) (h5 : ∀ x, m^2 ≤ x ∧ x ≤ n → f x ≤ 2) :
    n + m = 5 / 2 := sorry

end find_n_plus_m_l606_606694


namespace person_not_wet_l606_606560

theorem person_not_wet (n : ℕ) (h : n = 25) (closet : set (fin n × fin n)) :
  (∀ i : fin n, ∃ j : fin n, j ≠ i ∧ (i, j) ∈ closet ∧ ∀ k : fin n, k ≠ j → (i, j) ∈ closet) →
  ∃ i : fin n, ∀ j : fin n, (∃ k : fin n, k ≠ j ∧ (j, k) ∈ closet) → (∃ l : fin n, l ≠ j → (i, j) ∉ closet) :=
begin
  sorry
end

end person_not_wet_l606_606560


namespace largest_x_value_l606_606381

theorem largest_x_value 
  (x a b c d : ℝ)
  (h1 : 7 * x / 4 + 2 = 8 / x) 
  (h2 : x = (a + b * real.sqrt c) / d) 
  (ha : a = -4) 
  (hb : b = 8) 
  (hc : c = 15) 
  (hd : d = 7) :
  (a * c * d) / b = -52.5 :=
sorry

end largest_x_value_l606_606381


namespace charity_dinner_cost_l606_606566

def cost_of_rice_per_plate : ℝ := 0.10
def cost_of_chicken_per_plate : ℝ := 0.40
def number_of_plates : ℕ := 100

theorem charity_dinner_cost : 
  cost_of_rice_per_plate + cost_of_chicken_per_plate * number_of_plates = 50 :=
by
  sorry

end charity_dinner_cost_l606_606566


namespace least_int_gt_sqrt_450_l606_606433

theorem least_int_gt_sqrt_450 : ∃ n : ℕ, n > nat.sqrt 450 ∧ (∀ m : ℕ, m > nat.sqrt 450 → n ≤ m) :=
by
  have h₁ : 20^2 = 400 := by norm_num
  have h₂ : 21^2 = 441 := by norm_num
  have h₃ : 22^2 = 484 := by norm_num
  have sqrt_bounds : 21 < real.sqrt 450 ∧ real.sqrt 450 < 22 :=
    by sorry
  use 22
  split
  · sorry
  · sorry

end least_int_gt_sqrt_450_l606_606433


namespace logarithmic_inequality_solution_l606_606841

theorem logarithmic_inequality_solution {x : ℝ} :
  log (1/2) (2 * x + 1) ≥ log (1/2) 3 ↔ - (1/2) < x ∧ x ≤ 1 :=
begin
  sorry
end

end logarithmic_inequality_solution_l606_606841


namespace alcohol_formula_l606_606494

noncomputable def molecularFormula (mass : ℝ) (volume : ℝ) (Mgas : ℝ) (Vgas : ℝ) : ℕ :=
  let moles_gas := volume / Vgas
  let moles_alcohol := moles_gas
  let M_alcohol := mass / moles_alcohol
  let n := (M_alcohol - 18) / 14
  n.toNat

theorem alcohol_formula :
  molecularFormula 28.8 4.48 2 22.4 = 9 :=
  sorry

end alcohol_formula_l606_606494


namespace three_digit_numbers_divisible_by_11_l606_606191

theorem three_digit_numbers_divisible_by_11 : 
  let lower_bound := 100;
      upper_bound := 999 in
  let start := Nat.ceil (lower_bound / 11) in
  let end := Nat.floor (upper_bound / 11) in
  end - start + 1 = 81 := 
by
  let lower_bound := 100;
      upper_bound := 999 in
  let start := Nat.ceil (lower_bound / 11) in
  let end := Nat.floor (upper_bound / 11) in
  exact (end - start + 1 = 81)

end three_digit_numbers_divisible_by_11_l606_606191


namespace usual_time_to_school_l606_606420

theorem usual_time_to_school (S T t : ℝ) (h : 1.2 * S * (T - t) = S * T) : T = 6 * t :=
by
  sorry

end usual_time_to_school_l606_606420


namespace intervals_of_monotonicity_minimum_value_f_zeros_of_f_l606_606337

noncomputable def f (a x : ℝ) : ℝ := a^x + x^2 - x * real.log a - a

theorem intervals_of_monotonicity (x : ℝ) : (∀ x, f real.exp x > 0) ∧ (∀ x, f real.exp x < 0) := 
sorry

theorem minimum_value_f (a : ℝ) (ha_pos : 0 < a) (ha_neq_one : a ≠ 1) : 
  ∃ x, f a x = 1 - a := 
sorry

theorem zeros_of_f (a : ℝ) (ha_pos : 0 < a) (ha_neq_one : a ≠ 1) : 
  (0 < a ∧ a < 1 → ∀ x, f a x ≠ 0) ∧
  (a > 1 → ∃ x1 x2, f a x1 = 0 ∧ f a x2 = 0 ∧ x1 ≠ x2) := 
sorry

end intervals_of_monotonicity_minimum_value_f_zeros_of_f_l606_606337


namespace least_integer_greater_than_sqrt_450_l606_606476

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, (n > real.sqrt 450) ∧ ∀ m : ℤ, (m > real.sqrt  450) → n ≤ m :=
by
  apply exists.intro 22
  sorry

end least_integer_greater_than_sqrt_450_l606_606476


namespace inequality_sine_cosine_l606_606823

theorem inequality_sine_cosine (t : ℝ) (ht : t > 0) : 3 * Real.sin t < 2 * t + t * Real.cos t := 
sorry

end inequality_sine_cosine_l606_606823


namespace total_turnips_l606_606306

theorem total_turnips (keith_turnips : ℕ) (alyssa_turnips : ℕ) 
  (h_keith : keith_turnips = 6) (h_alyssa : alyssa_turnips = 9) : 
  keith_turnips + alyssa_turnips = 15 :=
by
  rw [h_keith, h_alyssa]
  exact rfl

end total_turnips_l606_606306


namespace least_integer_greater_than_sqrt_450_l606_606473

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606473


namespace largest_odd_digit_multiple_of_5_is_9955_l606_606875

def is_odd_digit (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (nat.digits 10 n), is_odd_digit d

def largest_odd_digit_multiple_of_5 (n : ℕ) : Prop :=
  n < 10000 ∧ n % 5 = 0 ∧ all_odd_digits n

theorem largest_odd_digit_multiple_of_5_is_9955 :
  ∃ n, largest_odd_digit_multiple_of_5 n ∧ n = 9955 :=
begin
  sorry
end

end largest_odd_digit_multiple_of_5_is_9955_l606_606875


namespace solution_set_l606_606802

variables {f : ℝ → ℝ}

-- Assume f is an odd function
axiom odd_function (f : ℝ → ℝ) : ∀ x, f(-x) = -f(x)

-- Given condition f(2) = 0
axiom f_at_2_zero : f 2 = 0

-- Given condition for x > 0, the inequality holds
axiom inequality_holds (x : ℝ) (hx : x > 0) : (x * (deriv f x) - f x) / (x^2) < 0

-- Define the statement to be proven
theorem solution_set :
  { x : ℝ | x^2 * f x > 0 } = {x : ℝ | (x < -2) ∨ (0 < x ∧ x < 2)} :=
by
  sorry

end solution_set_l606_606802


namespace problem_solution_l606_606310

noncomputable def p (x : ℝ) : ℝ := 
  (x - (Real.sin 1)^2) * (x - (Real.sin 3)^2) * (x - (Real.sin 9)^2)

theorem problem_solution : ∃ a b n : ℕ, 
  p (1 / 4) = Real.sin (a * Real.pi / 180) / (n * Real.sin (b * Real.pi / 180)) ∧
  a > 0 ∧ b > 0 ∧ a ≤ 90 ∧ b ≤ 90 ∧ a + b + n = 216 :=
sorry

end problem_solution_l606_606310


namespace inequality_proof_l606_606533

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606533


namespace collinear_vector_lambda_value_l606_606706

theorem collinear_vector_lambda_value :
  let a := (-1, 2)
  let b := (2, -3)
  let c := (-4, 7)
  ∃ λ : ℝ, λ * a + b = λ * c → λ = -2 :=
by
  -- Proof goes here. The statement above is sufficient for the starting point.
  sorry

end collinear_vector_lambda_value_l606_606706


namespace right_triangle_parity_l606_606718

theorem right_triangle_parity {a b c m n : ℤ} 
  (h1 : a = m^2 - n^2) 
  (h2 : b = 2 * m * n) 
  (h3 : c = m^2 + n^2) 
  (h4 : m = n + 1 ∨ m = n - 1) 
  (h5 : ¬(even a ∧ even b ∧ even c)):
  (odd c) ∧ (even b) ∧ (odd a) :=
by
  sorry

end right_triangle_parity_l606_606718


namespace solve_for_z_l606_606217

variable (z : ℂ)

theorem solve_for_z (h : (complex.I * z = 4 + 3 * complex.I)) : z = 3 - 4 * complex.I := 
sorry

end solve_for_z_l606_606217


namespace ab_bc_ca_leq_zero_l606_606662

theorem ab_bc_ca_leq_zero (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end ab_bc_ca_leq_zero_l606_606662


namespace complex_quadratic_solution_l606_606017

theorem complex_quadratic_solution (c d : ℤ) (h1 : 0 < c) (h2 : 0 < d) (h3 : (c + d * Complex.I) ^ 2 = 7 + 24 * Complex.I) :
  c + d * Complex.I = 4 + 3 * Complex.I :=
sorry

end complex_quadratic_solution_l606_606017


namespace locus_of_O_l606_606613

-- Define the setup and the given conditions
variable (ω Ω : Circle) (X Y T : Point)
variable (h_tangent : ω.isTangentToInterior Ω at T)
variable (P : Point) (h_P_on_Ω : P ∈ Ω)
variable (S : Point) (h_S_on_ω : S ∈ ω)
variable (h_PS_tangent_ω : Line(P, S).isTangentTo ω at S)

-- Define the object O as the circumcenter of triangle PST
def O := circumcenter (triangle P S T)

-- The required result: the locus of O
theorem locus_of_O :
  locus O = Circle(Y, sqrt (distance Y X * distance Y T)) :=
by
  sorry

end locus_of_O_l606_606613


namespace tangent_lengths_sum_l606_606029

noncomputable def circle_equation_with_tangents (O A : Point) (r OA BC : ℝ) :=
  (dist O A = OA) ∧ (dist O A > r) ∧ (r = 6) ∧ (OA = 15) ∧ (BC = 14)

theorem tangent_lengths_sum (O A B C T_1 T_2 T_3 : Point) (r OA BC : ℝ)
  (h₁ : circle_equation_with_tangents O A r OA BC)
  (h₂ : tangent_point O A B T_1)
  (h₃ : tangent_point O A C T_2)
  (h₄ : tangent_point_circle O B C T_3 BC) :
  dist A B + dist A C = 6 * real.sqrt 21 + 14 :=
by
  sorry

end tangent_lengths_sum_l606_606029


namespace largest_prime_factor_of_divisors_sum_180_l606_606792

def divisors_sum (n : ℕ) : ℕ :=
  (divisors n).sum

def largest_prime_factor (n : ℕ) : ℕ :=
  (factors n).filter prime ∣> ∧ .toList.maximum' sorry -- assume there is a maximum

theorem largest_prime_factor_of_divisors_sum_180 :
  ∃ N, N = divisors_sum 180 ∧ largest_prime_factor N = 13 := by
  sorry

end largest_prime_factor_of_divisors_sum_180_l606_606792


namespace max_min_plane_difference_l606_606976

-- Define the structure of the tetrahedron and relevant segments
structure Tetrahedron where
  V : Type
  faces : set (set V)
  edges : set (set V)

-- Define a regular tetrahedron
def regular_tetrahedron : Tetrahedron := sorry -- assumed to be defined correctly with all conditions met

-- Define the planes and the intersection conditions
def planes_intersecting (T : Tetrahedron) (k : ℕ) : Prop :=
  ∃ (p : fin k → set (set T.V)), 
    (∀ i, p i ≠ ∅) ∧
    let S := (T.faces) in
    let P := ⋃ j, p j in
    ∀ f ∈ S, ∃ s ∈ P, s ⊆ f

-- State the theorem
theorem max_min_plane_difference (T : Tetrahedron) :
  planes_intersecting T 8 ∧ planes_intersecting T 4 →
  (8 - 4) = 4 :=
by
  sorry

end max_min_plane_difference_l606_606976


namespace initial_percentage_of_milk_l606_606406

theorem initial_percentage_of_milk (M : ℝ) (H1 : M / 100 * 60 = 0.58 * 86.9) : M = 83.99 :=
by
  sorry

end initial_percentage_of_milk_l606_606406


namespace plumber_max_earnings_l606_606925

theorem plumber_max_earnings : 
  let charge_sink := 30
      let charge_shower := 40
      let charge_toilet := 50
      let job1 := 3 * charge_toilet + 3 * charge_sink
      let job2 := 2 * charge_toilet + 5 * charge_sink
      let job3 := 1 * charge_toilet + 2 * charge_shower + 3 * charge_sink
  in max job1 (max job2 job3) = 250 := 
sorry

end plumber_max_earnings_l606_606925


namespace inequality_proof_l606_606516

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l606_606516


namespace product_of_real_values_r_l606_606648

theorem product_of_real_values_r (x r : ℝ) (h : x ≠ 0) :
  (∀ x, 2 * x * (r - x) = 7 → discriminant (2 * x^2 - 2 * r * x + 7) = 0) 
  → (Real.sqrt 14) * (-Real.sqrt 14) = -14 :=
sorry

end product_of_real_values_r_l606_606648


namespace least_integer_greater_than_sqrt_450_l606_606481

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, (n > real.sqrt 450) ∧ ∀ m : ℤ, (m > real.sqrt  450) → n ≤ m :=
by
  apply exists.intro 22
  sorry

end least_integer_greater_than_sqrt_450_l606_606481


namespace license_plate_combinations_l606_606604

theorem license_plate_combinations :
  let letters := 26
  let two_other_letters := Nat.choose 25 2
  let repeated_positions := Nat.choose 4 2
  let arrange_two_letters := 2
  let first_digit_choices := 10
  let second_digit_choices := 9
  letters * two_other_letters * repeated_positions * arrange_two_letters * first_digit_choices * second_digit_choices = 8424000 :=
  sorry

end license_plate_combinations_l606_606604


namespace evaluate_sum_base_neg4_digits_l606_606329

def base_neg4_digits (n : ℕ) : ℕ :=
  if n < 4 then 1
  else if n < 52 then 3
  else if n < 820 then 5
  else 7

theorem evaluate_sum_base_neg4_digits :
  (∑ i in finset.range 2013, base_neg4_digits (i + 1)) = 12345 := sorry

end evaluate_sum_base_neg4_digits_l606_606329


namespace first_discount_correct_l606_606861

noncomputable def first_discount (x : ℝ) : Prop :=
  let initial_price := 600
  let first_discounted_price := initial_price * (1 - x / 100)
  let final_price := first_discounted_price * (1 - 0.05)
  final_price = 456

theorem first_discount_correct : ∃ x : ℝ, first_discount x ∧ abs (x - 57.29) < 0.01 :=
by
  sorry

end first_discount_correct_l606_606861


namespace betty_reaches_abel_in_17_5_minutes_l606_606941

theorem betty_reaches_abel_in_17_5_minutes (
  (initial_distance : ℝ) (initial_distance = 25) 
  (decrease_rate : ℝ) (decrease_rate = 2) 
  (time_abel_stops : ℝ) (time_abel_stops = 10) 
  (abel_speed_to_betty_speed_ratio : ℝ) (abel_speed_to_betty_speed_ratio = 2)
) : 
Betty_reaches_Abel_in (17.5) := by
  sorry

end betty_reaches_abel_in_17_5_minutes_l606_606941


namespace largest_odd_digit_multiple_of_5_lt_10000_l606_606877

def is_odd_digit (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), is_odd_digit d

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem largest_odd_digit_multiple_of_5_lt_10000 :
  ∃ n, n < 10000 ∧ all_odd_digits n ∧ is_multiple_of_5 n ∧
        ∀ m, m < 10000 → all_odd_digits m → is_multiple_of_5 m → m ≤ n :=
  sorry

end largest_odd_digit_multiple_of_5_lt_10000_l606_606877


namespace triangles_same_perimeter_l606_606902

theorem triangles_same_perimeter
    (S₁ S₂ : Circle)
    (P Q : Point)
    (h_intersect : intersects_disjoint S₁ S₂ P Q)
    (ℓ₁ ℓ₂ : Line)
    (h_parallel : parallel ℓ₁ ℓ₂)
    (A₁ A₂ B₁ B₂ : Point)
    (h_ℓ₁_through_P : passes_through ℓ₁ P)
    (h_ℓ₁_inter_S₁ : intersects_at ℓ₁ S₁ A₁)
    (h_ℓ₁_inter_S₂ : intersects_at ℓ₁ S₂ A₂)
    (h_A₁_ne_P : A₁ ≠ P)
    (h_A₂_ne_P : A₂ ≠ P)
    (h_ℓ₂_through_Q : passes_through ℓ₂ Q)
    (h_ℓ₂_inter_S₁ : intersects_at ℓ₂ S₁ B₁)
    (h_ℓ₂_inter_S₂ : intersects_at ℓ₂ S₂ B₂)
    (h_B₁_ne_Q : B₁ ≠ Q)
    (h_B₂_ne_Q : B₂ ≠ Q) :
    perimeter (Triangle.mk A₁ Q A₂) = perimeter (Triangle.mk B₁ P B₂) := 
sorry

end triangles_same_perimeter_l606_606902


namespace determine_a_l606_606905

theorem determine_a : 
  (∃ (a : ℝ), ∀ (x y : ℝ), (x, y) = (-1, 2) → 3 * x + y + a = 0) → ∃ (a : ℝ), a = 1 :=
by
  sorry

end determine_a_l606_606905


namespace charlie_delta_purchase_ways_is_correct_l606_606570

noncomputable def numOfWays_ToPurchase4Items : ℕ :=
  let totalCookies := 7
  let totalProducts := totalCookies + 4
  let charliePurchases (n : ℕ) := (totalProducts.choose n)
  let deltaPurchases (n : ℕ) := totalCookies.multichoose n
  (charliePurchases 4) +
  (charliePurchases 3) * (deltaPurchases 1) +
  (charliePurchases 2) * (deltaPurchases 2) +
  (charliePurchases 1) * (deltaPurchases 3) +
  deltaPurchases 4

theorem charlie_delta_purchase_ways_is_correct :
  numOfWays_ToPurchase4Items = 4054 := by
  -- use sorry to observe non proved areas
  sorry

end charlie_delta_purchase_ways_is_correct_l606_606570


namespace trapezoid_area_l606_606746

theorem trapezoid_area 
  (ABC : Triangle)
  (AB_eq_AC : ABC.AB = ABC.AC)
  (area_ABC : ABC.area = 80)
  (small_triangles_area : ∀ t, t ∈ (small_triangles ABC) → t.area = 2)
  (small_triangles_count : cardinality (small_triangles ABC) = 10)
  : ∃ DB CE, DBCE.area = 70 := 
by
  sorry

end trapezoid_area_l606_606746


namespace shaded_area_equivalent_l606_606016

theorem shaded_area_equivalent :
  ∃ (A_triangle A_hexagon A_shaded : ℝ),
    A_triangle = 960 ∧
    A_hexagon = 840 ∧
    A_shaded = 735 :=
by
  use 960
  use 840
  use 735
  split
  { refl }
  split
  { refl }
  { refl }

end shaded_area_equivalent_l606_606016


namespace number_of_children_l606_606018

theorem number_of_children 
  (num_adults : ℕ) 
  (meal_cost : ℕ) 
  (total_bill : ℕ) 
  (num_adults = 2) 
  (meal_cost = 3) 
  (total_bill = 21) 
  : ∃ (C : ℕ), C = (total_bill - num_adults * meal_cost) / meal_cost ∧ C = 5 := 
by 
  sorry

end number_of_children_l606_606018


namespace magnitude_z_conjugate_z_in_third_quadrant_l606_606126

noncomputable def z : ℂ := -3 + 4 * I

theorem magnitude_z : complex.abs z = 5 := 
by 
  -- the proof would go here 
  sorry

theorem conjugate_z_in_third_quadrant : 
  (complex.conj z).re < 0 ∧ (complex.conj z).im < 0 :=
by
  -- the proof would go here
  sorry

end magnitude_z_conjugate_z_in_third_quadrant_l606_606126


namespace first_pack_weight_l606_606586

-- Define the conditions
def miles_per_hour := 2.5
def hours_per_day := 8
def days := 5
def supply_per_mile := 0.5
def resupply_percentage := 0.25
def total_hiking_time := hours_per_day * days
def total_miles_hiked := total_hiking_time * miles_per_hour
def total_supplies_needed := total_miles_hiked * supply_per_mile
def resupply_factor := 1 + resupply_percentage

-- Define the theorem
theorem first_pack_weight :
  (total_supplies_needed / resupply_factor) = 40 :=
by
  sorry

end first_pack_weight_l606_606586


namespace john_finishes_third_task_at_1220_l606_606766

-- Define the starting time for the first task and the completion time for the second task
def start_time : Nat := 540 -- 9:00 AM in minutes
def end_time : Nat := 690 -- 11:30 AM in minutes

-- Define the variables for the duration of tasks
variable (t_first t_second t_third : Nat)

-- Specify the conditions
axiom task_conditions : 
  t_second = 2 * t_first ∧ 
  t_third = t_first ∧
  start_time + t_first + t_second = end_time

-- Prove that John finishes the third task at 12:20 PM
theorem john_finishes_third_task_at_1220 :
  start_time + t_first + t_second + t_third = 740 := -- 12:20 PM in minutes
by 
  apply task_conditions
  sorry

end john_finishes_third_task_at_1220_l606_606766


namespace sum_even_odd_difference_l606_606873

theorem sum_even_odd_difference :
  (finset.sum (finset.range 1000) (λ n : ℕ, 2 * n - 1)) - 
  (finset.sum (finset.range 1000) (λ n : ℕ, 2 * n)) = 1000 :=
begin
  sorry
end

end sum_even_odd_difference_l606_606873


namespace triangle_abc_is_isosceles_l606_606356

variable (A B C M K A1 B1 : Point)

variables (C_M_median : is_median C M A B)
variables (K_on_CM : on_line_segment C M K)
variables (A1_on_BC : on_intersect AK BC A1)
variables (B1_on_AC : on_intersect BK AC B1)
variables (AB1A1B_inscribed : is_inscribed_quadrilateral A B1 A1 B)

theorem triangle_abc_is_isosceles : AB = AC :=
by
  sorry

end triangle_abc_is_isosceles_l606_606356


namespace inequality_solution_l606_606968

-- Condition definitions in lean
def numerator (x : ℝ) : ℝ := (x^5 - 13 * x^3 + 36 * x) * (x^4 - 17 * x^2 + 16)
def denominator (y : ℝ) : ℝ := (y^5 - 13 * y^3 + 36 * y) * (y^4 - 17 * y^2 + 16)

-- Given the critical conditions
def is_zero_or_pm1_pm2_pm3_pm4 (y : ℝ) : Prop := 
  y = 0 ∨ y = 1 ∨ y = -1 ∨ y = 2 ∨ y = -2 ∨ y = 3 ∨ y = -3 ∨ y = 4 ∨ y = -4

-- The theorem statement
theorem inequality_solution (x y : ℝ) : 
  (numerator x / denominator y) ≥ 0 ↔ ¬ (is_zero_or_pm1_pm2_pm3_pm4 y) :=
sorry -- proof to be filled in later

end inequality_solution_l606_606968


namespace union_of_sets_l606_606682

variable (A : Set ℤ) (B : Set ℤ)

theorem union_of_sets (hA : A = {0, 1, 2}) (hB : B = {-1, 0}) : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_sets_l606_606682


namespace count_regular_40_gon_choices_l606_606562

theorem count_regular_40_gon_choices (n : ℕ) (m : ℕ) (r : ℕ)
  (h1 : n = 3600)
  (h2 : m = 40)
  (h3 : r = 72) :
  let d := n / r in
  let red_vertices := {i | ∃ k, i = k * d ∧ 1 ≤ i ∧ i ≤ n} in
  let valid_indices := {a | 1 ≤ a ∧ a ≤ n / m ∧ ∀ k, (a + k * (n / m)) % (n / r) ≠ 0} in
  valid_indices.card = 81 :=
sorry

end count_regular_40_gon_choices_l606_606562


namespace find_p_plus_q_l606_606700

noncomputable def sequence_a : ℕ → ℚ
| 1     := 1
| 2     := 3/7
| (n+3) := (sequence_a (n+1) * sequence_a (n+2)) / (2 * sequence_a (n+1) - sequence_a (n+2))

theorem find_p_plus_q :
  ∃ (p q : ℕ), 
  sequence_a 2019 = p / q ∧ Nat.coprime p q ∧ p + q = 8078 :=
sorry

end find_p_plus_q_l606_606700


namespace dihedral_angle_is_30_degrees_l606_606112

def isosceles_right_triangle (ABC : Triangle) : Prop :=
  ∃ A B C : Point, is_isosceles_right_triangle A B C

def height_from_hypotenuse (ABC : Triangle) (AD : Line) : Prop :=
  ∃ A B C D : Point, height_from_hypotenuse A B C D

def folded_equilateral_triangle (ABC : Triangle) (AD : Line) (angle : ℝ) : Prop :=
  fold_triangle_along_height A B C D angle = true → 
  is_equilateral_triangle A B C 

theorem dihedral_angle_is_30_degrees (ABC : Triangle) (AD : Line) :
  isosceles_right_triangle ABC →
  height_from_hypotenuse ABC AD →
  folded_equilateral_triangle ABC AD 30 :=
by
  sorry

end dihedral_angle_is_30_degrees_l606_606112


namespace height_of_parallelogram_l606_606735

-- Define the parallelogram and its properties.
variables (A B C D : Type) [point : Point A] [point : Point B] [point : Point C] [point : Point D]
variables (area : ℝ) (base : ℝ) (height : ℝ)

-- Given conditions as definitions.
def parallelogram_area (base height : ℝ) : ℝ := base * height
def given_area := 72
def given_base := 12
def find_height := 6

-- Main theorem to prove.
theorem height_of_parallelogram :
  parallelogram_area given_base find_height = given_area := by
    unfold parallelogram_area given_base find_height given_area
    rw [mul_comm]
    norm_num
    sorry

end height_of_parallelogram_l606_606735


namespace min_sum_of_arithmetic_sequence_l606_606854

theorem min_sum_of_arithmetic_sequence :
  let a_n := λ n : ℕ, 2 * (n : ℤ) - 49
  let S_n := λ n : ℕ, (n * (2 * n - 99) : ℤ) / 2
  ∃ n : ℕ, (∀ m : ℕ, S_n n ≤ S_n m) ∧ n = 24 :=
sorry

end min_sum_of_arithmetic_sequence_l606_606854


namespace cylinder_plane_intersection_curve_l606_606128

variable (r h : ℝ)
variable (α : ℝ)
variable (x : ℝ)

def valid_angle (a : ℝ) : Prop := 0 < a ∧ a < 90

def unfolded_curve (r h α : ℝ) (x : ℝ) : ℝ :=
  r * tan α * sin (x / r - π / 2)

theorem cylinder_plane_intersection_curve :
  valid_angle α → (0 < x ∧ x < 2 * π * r) → 
  (∃ y, y = r * tan α * sin (x / r - π / 2)) :=
by
  intros hα hx
  use unfolded_curve r h α x
  exact ⟨hα, hx⟩
  sorry

end cylinder_plane_intersection_curve_l606_606128


namespace find_z_l606_606250

theorem find_z (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : z = 3 - 4 * complex.I :=
by {
  sorry
}

end find_z_l606_606250


namespace cylindrical_to_rectangular_l606_606046

structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

structure RectangularCoord where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def convertCylindricalToRectangular (c : CylindricalCoord) : RectangularCoord :=
  { x := c.r * Real.cos c.θ,
    y := c.r * Real.sin c.θ,
    z := c.z }

theorem cylindrical_to_rectangular :
  convertCylindricalToRectangular ⟨7, Real.pi / 3, -3⟩ = ⟨3.5, 7 * Real.sqrt 3 / 2, -3⟩ :=
by sorry

end cylindrical_to_rectangular_l606_606046


namespace solution_l606_606227

noncomputable def z : ℂ := 3 - 4i

theorem solution (z : ℂ) (h : i * z = 4 + 3 * i) : z = 3 - 4 * i :=
by sorry

end solution_l606_606227


namespace choose_president_and_committee_l606_606286

-- Define the condition of the problem
def total_people := 10
def committee_size := 3

-- Define the function to calculate the number of combinations
def comb (n k : ℕ) : ℕ := Nat.choose n k

-- Proving the number of ways to choose the president and the committee
theorem choose_president_and_committee :
  (total_people * comb (total_people - 1) committee_size) = 840 :=
by
  sorry

end choose_president_and_committee_l606_606286


namespace painted_cube_probability_l606_606981

/-- Define the probability of specific event that painted cube can be placed on a horizontal surface
    so that the four vertical faces are all the same color.-/
theorem painted_cube_probability :
  let total_combinations := (3: ℕ) ^ 6,
      same_color_faces := (3: ℕ) * nat.choose 6 6 + (3: ℕ) * nat.choose 6 5 + (3: ℕ) * 2 * 1 * 3 in
  (same_color_faces: ℚ) / (total_combinations: ℚ) = 111 / 729 :=
by
  sorry

end painted_cube_probability_l606_606981


namespace least_integer_greater_than_sqrt_450_l606_606478

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, (n > real.sqrt 450) ∧ ∀ m : ℤ, (m > real.sqrt  450) → n ≤ m :=
by
  apply exists.intro 22
  sorry

end least_integer_greater_than_sqrt_450_l606_606478


namespace exercise_felt_weight_l606_606410

variable (n w : ℕ)
variable (p : ℝ)

def total_weight (n : ℕ) (w : ℕ) : ℕ := n * w

def felt_weight (total_weight : ℕ) (p : ℝ) : ℝ := total_weight * (1 + p)

theorem exercise_felt_weight (h1 : n = 10) (h2 : w = 30) (h3 : p = 0.20) : 
  felt_weight (total_weight n w) p = 360 :=
by 
  sorry

end exercise_felt_weight_l606_606410


namespace product_of_roots_l606_606709

variables {α β p q r s : ℝ}
variables (tgα tgβ : ℝ) (ctgα ctgβ : ℝ)

def problem_conditions :=
  (tgα = Math.atan α ∧ tgβ = Math.atan β) ∧ 
  (ctgα = Math.acot α ∧ ctgβ = Math.acot β) ∧
  (p = tgα + tgβ) ∧
  (q = tgα * tgβ) ∧
  (r = ctgα + ctgβ) ∧
  (s = ctgα * ctgβ)

theorem product_of_roots (h : problem_conditions α β p q r s tgα tgβ ctgα ctgβ) :
  r * s = p / q^2 :=
by sorry

end product_of_roots_l606_606709


namespace find_x_plus_y_over_2_l606_606712

noncomputable def log_base (a b : ℝ) := log b / log a

theorem find_x_plus_y_over_2 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : log_base y x + log_base x y = 6) (h4 : x * y = 128) (h5 : x = 2 * y^2) :
  (x + y) / 2 = 18 :=
by
  sorry

end find_x_plus_y_over_2_l606_606712


namespace round_neg_8_47_to_nearest_tenth_l606_606373

theorem round_neg_8_47_to_nearest_tenth : Float.round (-8.47) (10^-1) = -8.5 := 
by 
  -- providing a placeholder as the proof is not required
  sorry

end round_neg_8_47_to_nearest_tenth_l606_606373


namespace side_length_of_square_in_right_triangle_l606_606412

theorem side_length_of_square_in_right_triangle
  (PQ PR: ℝ) (h1: PQ = 9) (h2: PR = 12) :
  ∃ s: ℝ, s = 15 / 2 ∧ ∃ (QR: ℝ), QR = (PQ^2 + PR^2) ^ (1 / 2) :=
by
  use 15 / 2
  split
  sorry

end side_length_of_square_in_right_triangle_l606_606412


namespace value_of_u4_v4_w4_l606_606617

-- Defining the fourth roots as constants
def delta : ℝ := real.root 4 17
def epsilon : ℝ := real.root 4 37
def zeta : ℝ := real.root 4 57

-- Definition of the polynomial whose roots are u, v, w
def polynomial (x : ℝ) : ℝ :=
  (x - delta) * (x - epsilon) * (x - zeta) - (1 / 4)

-- Stating the goal in Lean
theorem value_of_u4_v4_w4 {u v w : ℝ} (hu : polynomial u = 0) (hv : polynomial v = 0) (hw : polynomial w = 0) (h_uniq : u ≠ v ∧ v ≠ w ∧ u ≠ w) :
  u^4 + v^4 + w^4 = 112 := sorry

end value_of_u4_v4_w4_l606_606617


namespace b_alone_days_l606_606505

-- Definitions from the conditions
def work_rate_b (W_b : ℝ) : ℝ := W_b
def work_rate_a (W_b : ℝ) : ℝ := 2 * W_b
def work_rate_c (W_b : ℝ) : ℝ := 6 * W_b
def combined_work_rate (W_b : ℝ) : ℝ := work_rate_a W_b + work_rate_b W_b + work_rate_c W_b
def total_days_together : ℝ := 10
def total_work (W_b : ℝ) : ℝ := combined_work_rate W_b * total_days_together

-- The proof problem
theorem b_alone_days (W_b : ℝ) : 90 = total_work W_b / work_rate_b W_b :=
by
  sorry

end b_alone_days_l606_606505


namespace minimum_amount_spent_l606_606837

def players : ℕ := 17
def juice_boxes_per_pack : ℕ := 3
def price_per_juice_pack : ℤ := 200 -- in cents
def apples_per_bag : ℕ := 5
def price_per_apple_bag : ℤ := 400 -- in cents

theorem minimum_amount_spent : 
  (let p := (players + juice_boxes_per_pack - 1) / juice_boxes_per_pack in
  let b := (players + apples_per_bag - 1) / apples_per_bag in 
  p * price_per_juice_pack + b * price_per_apple_bag = 2800) := sorry

end minimum_amount_spent_l606_606837


namespace nested_function_value_l606_606336

def g (x : ℝ) : ℝ :=
  if x < 8 then x^2 + 2 else x - 15

theorem nested_function_value :
  g (g (g 18)) = -4 :=
by
  -- place proof here
  sorry

end nested_function_value_l606_606336


namespace rational_roots_of_cubic_l606_606845

theorem rational_roots_of_cubic (a b c : ℚ) :
  (∀ x : ℚ, x^3 + a * x^2 + b * x + c = 0 → x ∈ ℤ) →
  (a, b, c) = (1, -1, -1) ∨ (a, b, c) = (0, 0, 0) ∨ (a, b, c) = (1, -2, 0) :=
sorry

end rational_roots_of_cubic_l606_606845


namespace area_proof_l606_606299

def point := (Real × Real)
def vertex := (Real × Real)

def rectangle_vertices : List vertex := [(0, 0), (8, 0), (8, 7), (0, 7)]

noncomputable def area_of_shaded_region (P : point) (vertices : List vertex) : Real :=
  let (x, y) := P
  let areas := vertices.map (λ (v₁, v₂), 0.5 * (abs ((0 * (0 - y) + 8 * (y - 0) + x * (0 - 0)))))
  areas.sum

theorem area_proof :
  area_of_shaded_region (sqrt 2, sqrt 7) rectangle_vertices = 28 :=
by
  -- Area calculation steps go here
  sorry

end area_proof_l606_606299


namespace seven_remaining_weights_can_be_determined_l606_606867

theorem seven_remaining_weights_can_be_determined
  (masses : List ℕ)
  (consec_weights : List ℕ)
  (orig_weights : masses = List.range' 1 13)
  (seven_consec : consec_weights ⊆ masses ∧ consec_weights.length = 7) :
  ∃ weigh1 weigh2 : ℕ → ℕ → bool, ( -- defining two weighings as λ functions returning boolean (balance)
    ∀ i j (hij : i ≠ j),
      let (sum1_1, sum1_2) := weigh1 i j in
      let (sum2_1, sum2_2) := weigh2 i j in
      (sum1_1 + sum1_2 = sum2_1 + sum2_2) ∨ (sum1_1 + sum1_2 < sum2_1 + sum2_2) ∨ (sum1_1 + sum1_2 > sum2_1 + sum2_2)
  )
  :=
sorry

end seven_remaining_weights_can_be_determined_l606_606867


namespace same_speed_l606_606021

theorem same_speed (x : ℝ) (h : (2 * x^2 + 2 * x - 24) = ((2 * x^2 - 4 * x - 120) / (2 * x - 4))) :
  (2 * x^2 + 2 * x - 24) = 25.5 :=
begin
  sorry
end

end same_speed_l606_606021


namespace coin_toss_fairness_l606_606693

-- Statement of the problem as a Lean theorem.
theorem coin_toss_fairness (P_Heads P_Tails : ℝ) (h1 : P_Heads = 0.5) (h2 : P_Tails = 0.5) : 
  P_Heads = P_Tails ∧ P_Heads = 0.5 := 
sorry

end coin_toss_fairness_l606_606693


namespace inequality_proof_l606_606550

variable {x y z : ℝ}

theorem inequality_proof (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * real.sqrt(z * x)) + 
  y^3 / (y^3 + 2 * z^2 * real.sqrt(x * y)) + 
  z^3 / (z^3 + 2 * x^2 * real.sqrt(y * z)) >= 1) :=
sorry

end inequality_proof_l606_606550


namespace problem_part_I_problem_part_II_l606_606725

variables {a b c m : ℝ}
variables {A B C : ℝ}

theorem problem_part_I (h1 : a = 2) (h2 : m = 5 / 4) (h3 : sin B + sin C = m * sin A) (h4 : a^2 - 4 * b * c = 0) :
  (b = 2 ∧ c = 1 / 2) ∨ (b = 1 / 2 ∧ c = 2) :=
sorry

theorem problem_part_II (hA_acute : A < π / 2) (h3 : sin B + sin C = m * sin A) (h4 : a^2 - 4 * b * c = 0) :
  sqrt 6 / 2 < m ∧ m < sqrt 2 :=
sorry

end problem_part_I_problem_part_II_l606_606725


namespace ceil_evaluation_l606_606060

theorem ceil_evaluation : 
  (Int.ceil (((-7 : ℚ) / 4) ^ 2 - (1 / 8)) = 3) :=
sorry

end ceil_evaluation_l606_606060


namespace cos_seven_pi_over_four_l606_606639

theorem cos_seven_pi_over_four :
  Real.cos (7 * Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_seven_pi_over_four_l606_606639


namespace minimum_value_of_omega_l606_606161

theorem minimum_value_of_omega
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h₁ : ∀ x, f x = 3 * sin (ω * x + π / 6) - 2)
  (h₂ : ω > 0)
  (h₃ : ∃ n ∈ ℤ, ω = 3 * n) :
  ω = 3 :=
sorry

end minimum_value_of_omega_l606_606161


namespace inscribed_triangle_equality_angles_l606_606294

open EuclideanGeometry

theorem inscribed_triangle_equality_angles
  {A B C D E F G O : Point} :
  is_inscribed_triangle A B C O ∧
  midpoint D B C ∧
  line_intersects_circle (line_through A D) O E ∧
  parallel_lines_through E (line_through E F) (line_through B C) ∧
  perpendicular (line_through C G) (line_through A C) ∧
  line_intersects E A G
→ ∠ A G C = ∠ F G C :=
by
  sorry

end inscribed_triangle_equality_angles_l606_606294


namespace union_of_sets_l606_606684

theorem union_of_sets (A : Set ℤ) (B : Set ℤ) (hA : A = {0, 1, 2}) (hB : B = {-1, 0}) :
  A ∪ B = {-1, 0, 1, 2} :=
by
  rw [hA, hB]
  apply Set.ext
  intro x
  simp
  tauto

end union_of_sets_l606_606684


namespace mn_product_l606_606197

-- Define the imaginary unit
def i : ℂ := complex.I

-- Define the property that we need to prove
theorem mn_product (m n : ℝ) (h : (1 + m * i) / i = 1 + n * i) : m * n = -1 :=
by
  sorry

end mn_product_l606_606197


namespace inequality_proof_l606_606525

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606525


namespace part_i_part_ii_l606_606160

noncomputable def f (ω φ x : Real) : Real := sqrt 3 * sin (ω * x + φ)

theorem part_i (ω : Real) (φ : Real) (hω_pos : ω > 0)
  (h_sym : ∀ x : Real, f ω φ x = f ω φ (π / 3 - x))
  (h_dist : ∀ b : Real, f ω φ (b + π) = f ω φ b) :
  ω = 2 ∧ φ = -π / 6 :=
sorry

theorem part_ii (α : Real) (h_f : f 2 (-π / 6) (α / 2) = sqrt 3 / 4) 
  (h_α_bounds : π / 6 < α ∧ α < 2 * π / 3) :
  cos (α + π / 3) = (sqrt 3 + sqrt 15) / 8 :=
sorry

end part_i_part_ii_l606_606160


namespace chord_difference_l606_606702
variable (a c b d : ℝ)

theorem chord_difference :
  EF = 2 * (a * d - b * c) :=
sorry

end chord_difference_l606_606702


namespace ratio_cost_to_posted_price_l606_606933

variable (p : ℝ)

theorem ratio_cost_to_posted_price (h1 : ∃ p : ℝ, p > 0)
    (h2 : ∀ p > 0, let sp := (3 / 4) * p in let cp := (5 / 6) * sp in (cp / p) = (5 / 8)) :
  true :=
by
  sorry

end ratio_cost_to_posted_price_l606_606933


namespace brittany_average_correct_l606_606958

def brittany_first_score : ℤ :=
78

def brittany_second_score : ℤ :=
84

def brittany_average_after_second_test (score1 score2 : ℤ) : ℤ :=
(score1 + score2) / 2

theorem brittany_average_correct : 
  brittany_average_after_second_test brittany_first_score brittany_second_score = 81 := 
by
  sorry

end brittany_average_correct_l606_606958


namespace rational_number_relation_l606_606926

variable (A B : ℚ)

/-- Given a rational number x exceeds half its value by 1/5 of the difference between three-fourths 
and two-thirds of the total of A and B, prove that the rational number is 1/30 of the total of A and B. -/
theorem rational_number_relation : ∃ (x : ℚ), 
  x = (1 / 2) * x + (1 / 5) * ((3 / 4) * (A + B) - (2 / 3) * (A + B)) → 
  x = (1 / 30) * (A + B) :=
by
  sorry 

end rational_number_relation_l606_606926


namespace aunt_wang_bought_n_lilies_l606_606957

theorem aunt_wang_bought_n_lilies 
  (cost_rose : ℕ) 
  (cost_lily : ℕ) 
  (total_spent : ℕ) 
  (num_roses : ℕ) 
  (num_lilies : ℕ) 
  (roses_cost : num_roses * cost_rose = 10) 
  (total_spent_cond : total_spent = 55) 
  (cost_conditions : cost_rose = 5 ∧ cost_lily = 9) 
  (spending_eq : total_spent = num_roses * cost_rose + num_lilies * cost_lily) : 
  num_lilies = 5 :=
by 
  sorry

end aunt_wang_bought_n_lilies_l606_606957


namespace inequality_xyz_l606_606519

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x))) +
  (y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y))) +
  (z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 :=
by sorry

end inequality_xyz_l606_606519


namespace final_score_l606_606977

-- Definitions based on the conditions
def bullseye_points : ℕ := 50
def miss_points : ℕ := 0
def half_bullseye_points : ℕ := bullseye_points / 2

-- Statement to prove
theorem final_score : bullseye_points + miss_points + half_bullseye_points = 75 :=
by
  sorry

end final_score_l606_606977


namespace complex_solution_l606_606200

theorem complex_solution (z : ℂ) (h : ((0 : ℝ) + 1 * z = 4 + 3 * (complex.I))) : 
  z = 3 - 4 * (complex.I) :=
sorry

end complex_solution_l606_606200


namespace yi_original_speed_l606_606896

-- Define the given constants
def d_ab : ℝ := 300
def d_jia1 : ℝ := 140
def d_yi1 : ℝ := 160
def d_jia2 : ℝ := 120
def d_yi2 : ℝ := 180

-- Define the equation for the speed ratio
def speed_ratio_first : ℝ := d_jia1 / d_yi1
def speed_ratio_second (v_yi : ℝ) : ℝ := d_jia2 / (v_yi + 1)

-- The theorem to prove Yi's original speed
theorem yi_original_speed (v_yi : ℝ) : 
  speed_ratio_first = 7/8 ∧ 
  speed_ratio_second v_yi = 2/3 → 
  v_yi = 3.2 :=
by
  sorry

end yi_original_speed_l606_606896


namespace sum_of_squares_of_coeffs_l606_606157

-- Define the expression
def expr := 5 * (x ^ 3 - 3 * x ^ 2 + 4) - 8 * (2 * x ^ 3 - x ^ 2 - 2)

-- Define the fully simplified expression
def simplified_expr := -11 * x ^ 3 - 7 * x ^ 2 + 36

-- Calculate and prove the sum of the squares of the coefficients
theorem sum_of_squares_of_coeffs : 
  let a := -11
  let b := -7
  let c := 36
  (a ^ 2 + b ^ 2 + c ^ 2) = 1466 := 
by 
  let a := -11
  let b := -7
  let c := 36
  show (a ^ 2 + b ^ 2 + c ^ 2) = 1466
  sorry

end sum_of_squares_of_coeffs_l606_606157


namespace find_z_l606_606252

theorem find_z (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : z = 3 - 4 * complex.I :=
by {
  sorry
}

end find_z_l606_606252


namespace base_angle_of_isosceles_triangle_l606_606283

theorem base_angle_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : a = 50) (h₂ : a + b + c = 180) (h₃ : a = b ∨ b = c ∨ c = a) : 
  b = 50 ∨ b = 65 :=
by sorry

end base_angle_of_isosceles_triangle_l606_606283


namespace platform_length_correct_l606_606919

/-- Assume the length of the train is known. --/
def train_length : ℝ := 240.0416

/-- Assume the speed of the train in meters per second. --/
def train_speed : ℝ := 72 * 1000 / 3600

/-- Assume the time taken for the train to cross the platform. --/
def crossing_time : ℝ := 26

/-- The distance covered by the train while crossing the platform. --/
def distance_covered : ℝ := train_speed * crossing_time

/-- The length of the platform. --/
def platform_length : ℝ := distance_covered - train_length

theorem platform_length_correct : 
  platform_length = 279.9584 := 
by 
  unfold platform_length distance_covered 
  unfold train_speed crossing_time train_length 
  sorry

end platform_length_correct_l606_606919


namespace isosceles_triangle_l606_606350

-- Definitions
variables {A B C M K A1 B1 : Type*}
variables (point : A) (point B) (point C) (point M) (point K) (point A1) (point B1)
variable (triangle : A → B → C → Type*)
variable (median : triangle.point → point → Type*)

-- Conditions
variable (is_median_CM : median C M)
variable (is_on_CM : point K ∈ median C M)
variable (intersects_AK_BC : point A1 = AK ∩ BC)
variable (intersects_BK_AC : point B1 = BK ∩ AC)
variable (inscribed_AB1A1B : circle AB1A1B)

-- Proof
theorem isosceles_triangle (h1 : is_median_CM) (h2 : is_on_CM)
  (h3 : intersects_AK_BC) (h4 : intersects_BK_AC)
  (h5 : inscribed_AB1A1B) :
  AB = AC :=
by
  sorry

end isosceles_triangle_l606_606350


namespace seq_properties_l606_606673

theorem seq_properties (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) 
  (h1 : ∀ n, S n = n * (n + 1) / 2) (h2 : ∀ n, b n = 1 / (S n))
  (h3 : a 3 * b 3 = 1 / 2) (h4 : S 5 + S 3 = 21) : 
  (∀ n, S n = n * (n + 1) / 2) ∧ (∀ n, (∑ i in finset.range n, b i) = 2 * n / (n + 1)) :=
  sorry

end seq_properties_l606_606673


namespace sequence_a_n_a_99_value_l606_606754

theorem sequence_a_n_a_99_value :
  ∃ (a : ℕ → ℝ), a 1 = 3 ∧ (∀ n, 2 * (a (n + 1)) - 2 * (a n) = 1) ∧ a 99 = 52 :=
by {
  sorry
}

end sequence_a_n_a_99_value_l606_606754


namespace decreases_with_increasing_n_l606_606039

theorem decreases_with_increasing_n
  (e : ℝ) (R : ℝ) (r : ℝ) (n : ℝ) :
  (0 < e) →
  (0 < R) →
  (0 < r) →
  (0 < n) →
  let C := λ n, e * real.sqrt n / (R + n * r^2) in
  ∀ n1 n2 : ℝ, (0 < n1) → (0 < n2) → (n1 < n2) → (C n1 > C n2) :=
by
  sorry

end decreases_with_increasing_n_l606_606039


namespace least_integer_greater_than_sqrt_450_l606_606442

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606442


namespace common_fraction_l606_606090

noncomputable def x : ℚ := 0.6666666 -- represents 0.\overline{6}
noncomputable def y : ℚ := 0.2222222 -- represents 0.\overline{2}
noncomputable def z : ℚ := 0.4444444 -- represents 0.\overline{4}

theorem common_fraction :
  x + y - z = 4 / 9 :=
by
  -- Provide proofs here
  sorry

end common_fraction_l606_606090


namespace least_integer_greater_than_sqrt_450_l606_606487

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, n > real.sqrt 450 ∧ ∀ m : ℕ, m > real.sqrt 450 → n ≤ m :=
begin
  use 22,
  split,
  { sorry },
  { sorry }
end

end least_integer_greater_than_sqrt_450_l606_606487


namespace repeating_decimals_sum_l606_606069

theorem repeating_decimals_sum :
  let x := 0.6666666 -- 0.\overline{6}
  let y := 0.2222222 -- 0.\overline{2}
  let z := 0.4444444 -- 0.\overline{4}
  (x + y - z) = 4 / 9 := 
by
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  calc
    -- Calculate (x + y - z)
    (x + y - z) = (2 / 3 + 2 / 9 - 4 / 9) : by sorry
                ... = 4 / 9 : by sorry


end repeating_decimals_sum_l606_606069


namespace terminating_decimal_expansion_l606_606998

theorem terminating_decimal_expansion (a b : ℕ) (h : 1600 = 2^6 * 5^2) :
  (13 : ℚ) / 1600 = 65 / 1000 :=
by
  sorry

end terminating_decimal_expansion_l606_606998


namespace polygon_sides_l606_606256

-- Given conditions
def is_interior_angle (angle : ℝ) : Prop :=
  angle = 150

-- The theorem to prove the number of sides
theorem polygon_sides (h : is_interior_angle 150) : ∃ n : ℕ, n = 12 :=
  sorry

end polygon_sides_l606_606256


namespace dave_walks_400_feet_or_less_l606_606964

theorem dave_walks_400_feet_or_less :
  let gates := 12
  let distance_between_gates := 100
  let total_combinations := gates * (gates - 1)
  let valid_combinations :=
    (2 * (4 + 5 + 6 + 7)) + (4 * 8)
  let probability := valid_combinations.to_rational / total_combinations.to_rational
  let fraction := rat.mk 19 33
  let m := 19
  let n := 33
  m + n = 52 :=
by
  -- Proof omitted
  sorry

end dave_walks_400_feet_or_less_l606_606964


namespace least_integer_greater_than_sqrt_450_l606_606439

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606439


namespace practice_problems_total_l606_606855

theorem practice_problems_total :
  let marvin_yesterday := 40
  let marvin_today := 3 * marvin_yesterday
  let arvin_yesterday := 2 * marvin_yesterday
  let arvin_today := 2 * marvin_today
  let kevin_yesterday := 30
  let kevin_today := kevin_yesterday + 10
  let total_problems := (marvin_yesterday + marvin_today) + (arvin_yesterday + arvin_today) + (kevin_yesterday + kevin_today)
  total_problems = 550 :=
by
  sorry

end practice_problems_total_l606_606855


namespace angle_between_vectors_acute_l606_606680

theorem angle_between_vectors_acute (y : ℝ):
  let A := (-1, 1) in
  let B := (3, y) in
  let a := (1, 2) in
  let AB := (B.1 - A.1, B.2 - A.2) in
  (AB.1 * a.1 + AB.2 * a.2 > 0) → y ∈ Set.Ioo (-1 : ℝ) 9 ∪ Set.Ioi 9 :=
by
  intro h
  sorry

end angle_between_vectors_acute_l606_606680


namespace largest_prime_factor_sum_divisors_180_l606_606796

theorem largest_prime_factor_sum_divisors_180 :
  let N := ∑ d in (Finset.divisors 180), d in
  Nat.greatest_prime_factor N = 13 :=
by
  sorry

end largest_prime_factor_sum_divisors_180_l606_606796


namespace solution_l606_606230

noncomputable def z : ℂ := 3 - 4i

theorem solution (z : ℂ) (h : i * z = 4 + 3 * i) : z = 3 - 4 * i :=
by sorry

end solution_l606_606230


namespace complex_solution_l606_606201

theorem complex_solution (z : ℂ) (h : ((0 : ℝ) + 1 * z = 4 + 3 * (complex.I))) : 
  z = 3 - 4 * (complex.I) :=
sorry

end complex_solution_l606_606201


namespace triangle_perpendicular_l606_606821

/-- Lean 4 statement for: Given a triangle ABC with circumcircle ω, a point D such that DB and DC 
are tangent to ω, B' as the reflection of B over AC, C' as the reflection of C over AB, and O 
as the circumcenter of triangle DB'C', prove that AO is perpendicular to BC. -/
theorem triangle_perpendicular (ABC : Triangle) (ω : Circle) (D : Point) (B C B' C' O : Point) :
  Tangent (ω) D B → Tangent (ω) D C → 
  Reflect (B, AC) B' → Reflect (C, AB) C' →
  Circumcenter (Triangle.mk D B' C') O →
  Perpendicular (Line.mk A O) (Line.mk B C) := by
  sorry

end triangle_perpendicular_l606_606821


namespace largest_prime_factor_divisors_sum_l606_606788

def prime_factors (n : ℕ) : List ℕ := sorry -- Dummy placeholder for prime factorization

theorem largest_prime_factor_divisors_sum :
  let N := (1 + 2 + 2^2) * (1 + 3 + 3^2) * (1 + 5)
  in List.maximum (prime_factors N) = 13 :=
by
  let N := (1 + 2 + 2^2) * (1 + 3 + 3^2) * (1 + 5)
  have h : prime_factors N = [2, 3, 7, 13] := sorry -- Placeholder
  exact List.maximum_eq_some.mp ⟨13, List.mem_cons_self 13 _, sorry⟩

end largest_prime_factor_divisors_sum_l606_606788


namespace sum_of_first_4_terms_is_minus_20_l606_606139

-- Define the geometric sequence and sum conditions
def a_n_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  finset.sum (finset.range n) a

def S_n (a : ℕ → ℝ) (n : ℕ) (q : ℝ) : ℝ :=
  if q = 1 then n else (1 - q ^ n) / (1 - q)

-- Given conditions
def given_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a_n_geometric a q ∧ 28 * S_n a 3 q = S_n a 6 q

-- The Lean statement to prove
theorem sum_of_first_4_terms_is_minus_20 (a : ℕ → ℝ) (q : ℝ) 
  (h : given_conditions a q) : sum_first_n_terms a 4 = -20 :=
sorry

end sum_of_first_4_terms_is_minus_20_l606_606139


namespace four_people_seven_chairs_l606_606287

def num_arrangements (total_chairs : ℕ) (num_reserved : ℕ) (num_people : ℕ) : ℕ :=
  (total_chairs - num_reserved).choose num_people * (num_people.factorial)

theorem four_people_seven_chairs (total_chairs : ℕ) (chairs_occupied : ℕ) (num_people : ℕ): 
    total_chairs = 7 → chairs_occupied = 2 → num_people = 4 →
    num_arrangements total_chairs chairs_occupied num_people = 120 :=
by
  intros
  unfold num_arrangements
  sorry

end four_people_seven_chairs_l606_606287


namespace ea_fc_bd_concurrent_l606_606775

variable {Point : Type} [AffinePlane Point]

theorem ea_fc_bd_concurrent
  (A B C D E F : Point)
  (l : Line)
  (h_parallelogram : Parallelogram A B C D)
  (h_line_parallel : l ∥ LineAC)
  (h_pass : D ∈ l)
  (h_equidistant1 : dist D E = dist D B)
  (h_equidistant2 : dist D F = dist D B) :
  Concurrent (Line E A) (Line F C) (Line B D) := 
sorry

end ea_fc_bd_concurrent_l606_606775


namespace factor_2w4_minus_50_l606_606098

noncomputable def factor_expr (w : Polynomial ℝ) : Polynomial ℝ :=
  2 * (w^2 - 5) * (w^2 + 5)

theorem factor_2w4_minus_50 (w : Polynomial ℝ) :
  2 * w^4 - 50 = factor_expr w :=
by {
  sorry
}

end factor_2w4_minus_50_l606_606098


namespace tangent_line_to_circle_l606_606851

theorem tangent_line_to_circle :
  ∀ (x y : ℝ), (x^2 + y^2 - 4*x = 0) ∧ (x = 1 ∧ y = sqrt 3) → (x - sqrt 3 * y + 2 = 0) :=
by
  intros x y h
  cases h with h_circle h_point
  cases h_point with hx hy
  sorry

end tangent_line_to_circle_l606_606851


namespace b_seq_formula_l606_606699

noncomputable def b_seq : ℕ → ℚ
| 0     := 1/2
| (n+1) := 2 - 1 / b_seq n

theorem b_seq_formula : ∀ n : ℕ, b_seq n = n / (n + 1) :=
by
  intro n
  induction n with k ih
  · rw [b_seq, zero_div, add_zero, one_div, div_one]
    norm_num
  · rw [b_seq, ih, add_one_div, div_add]
    norm_num
    sorry -- Proof goes here

lemma compare_x_x_y_y (n : ℕ) (hn : 0 < n) :
  let x := (n / (n + 1 : ℝ)) ^ n
  let y := (n / (n + 1 : ℝ)) ^ (n + 1)
  in x ^ x = y ^ y :=
by
  intros
  let x := (n / (n + 1 : ℝ)) ^ n
  let y := (n / (n + 1 : ℝ)) ^ (n + 1)
  have hxy : x = y := by
    rw [pow_add_one_div, mul_div, div_self, pow]
    norm_num
    sorry -- Proof goes here
  sorry -- Proof goes here

end b_seq_formula_l606_606699


namespace integral_sin_over_two_plus_sin_l606_606993

-- Define the indefinite integral using Lean's integral notation
def indefinite_integral (f : ℝ → ℝ) : ℝ → ℝ := λ x, ∫ t in 0..x, f t

-- State the problem in Lean 4
theorem integral_sin_over_two_plus_sin :
  indefinite_integral (λ x, (sin x) / (2 + sin x)) = 
  λ x, x - (4 / Real.sqrt 3) * Real.arctan ((2 * Real.tan (x / 2) + 1) / Real.sqrt 3) + arbitrary_constant :=
by
  -- The proof will be provided here
  sorry

end integral_sin_over_two_plus_sin_l606_606993


namespace water_consumed_is_correct_l606_606597

def water_consumed (traveler_ounces : ℕ) (camel_multiplier : ℕ) (ounces_per_gallon : ℕ) : ℕ :=
  let camel_ounces := traveler_ounces * camel_multiplier
  let total_ounces := traveler_ounces + camel_ounces
  total_ounces / ounces_per_gallon

theorem water_consumed_is_correct :
  water_consumed 32 7 128 = 2 :=
by
  -- add proof here
  sorry

end water_consumed_is_correct_l606_606597


namespace minimize_material_use_l606_606000

theorem minimize_material_use 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y)
  (total_area : x * y + (x^2 / 4) = 8) :
  (abs (x - 2.343) ≤ 0.001) ∧ (abs (y - 2.828) ≤ 0.001) :=
sorry

end minimize_material_use_l606_606000


namespace cube_face_perimeter_l606_606857

theorem cube_face_perimeter (V : ℝ) (hV : V = 1000) : ∃ P : ℝ, P = 40 :=
by
  let s := real.cbrt V
  have hs : s = 10 := by sorry
  let P := 4 * s
  have hP : P = 40 := by sorry
  exact ⟨P, hP⟩

end cube_face_perimeter_l606_606857


namespace area_of_enclosed_region_l606_606023

noncomputable def curve1 (x : ℝ) : ℝ := (1/3) * x^2
noncomputable def curve2 (x : ℝ) : ℝ := x

theorem area_of_enclosed_region :
  let S := ∫ x in 0..3, (curve2 x - curve1 x)
  S = 1 :=
by
  sorry

end area_of_enclosed_region_l606_606023


namespace sum_f_from_1_to_2023_l606_606145

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom odd_f_plus_2 : ∀ x : ℝ, f (x + 2) = -f (-x + 2)
axiom f_zero : f 0 = 1

theorem sum_f_from_1_to_2023 : (∑ i in finset.range 2023, f (i + 1)) = -1 := 
by
  sorry

end sum_f_from_1_to_2023_l606_606145


namespace solve_for_z_l606_606238

variable {z : ℂ}

theorem solve_for_z (h : complex.I * z = 4 + 3*complex.I) : z = 3 - 4*complex.I :=
sorry

end solve_for_z_l606_606238


namespace two_good_collections_count_l606_606293

-- Each card is represented as a tuple of 4 attributes. Each attribute can take one of 3 values (0, 1, or 2).
-- Define a set deck as the set of all 81 possible tuples.
def set_deck := {c : (ℕ × ℕ × ℕ × ℕ) | ∀ i ∈ [c.1, c.2, c.3, c.4], i < 3}

-- Define a good attribute for 3 cards:
-- An attribute is good if all three cards either all take on the same value or take on all three different values.
def is_good_attr (cards : List (ℕ × ℕ × ℕ × ℕ)) (attr_index : Fin 4) : Prop :=
  (cards.map (λ card, card.nth_le attr_index)).nodup ∨
  ((cards.map (λ card, card.nth_le attr_index)).length = 1 ∨ 
   (cards.map (λ card, card.nth_le attr_index)).length = 3)

-- Define a two-good collection of 3 cards:
def is_two_good (cards : List (ℕ × ℕ × ℕ × ℕ)) : Prop :=
  cards.length = 3 ∧
  (Finset.range 4).filter (is_good_attr cards).card = 2

-- The problem is to find the number of two-good collections of 3 cards
theorem two_good_collections_count : 
  (Finset.powersetLen 3 (Finset.univ : Finset set_deck)).filter is_two_good).card = 25272 :=
sorry

end two_good_collections_count_l606_606293


namespace complex_solution_l606_606207

theorem complex_solution (z : ℂ) (h : ((0 : ℝ) + 1 * z = 4 + 3 * (complex.I))) : 
  z = 3 - 4 * (complex.I) :=
sorry

end complex_solution_l606_606207


namespace least_integer_greater_than_sqrt_450_l606_606488

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, n > real.sqrt 450 ∧ ∀ m : ℕ, m > real.sqrt 450 → n ≤ m :=
begin
  use 22,
  split,
  { sorry },
  { sorry }
end

end least_integer_greater_than_sqrt_450_l606_606488


namespace repeating_decimal_sum_l606_606064

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem repeating_decimal_sum : repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l606_606064


namespace product_and_quotient_l606_606608

theorem product_and_quotient : (16 * 0.0625 / 4 * 0.5 * 2) = (1 / 4) :=
by
  -- The proof steps would go here
  sorry

end product_and_quotient_l606_606608


namespace find_even_integer_l606_606259

theorem find_even_integer (x y z : ℤ) (h₁ : Even x) (h₂ : Odd y) (h₃ : Odd z)
  (h₄ : x < y) (h₅ : y < z) (h₆ : y - x > 5) (h₇ : z - x = 9) : x = 2 := 
by 
  sorry

end find_even_integer_l606_606259


namespace grape_juice_problem_l606_606912

noncomputable def grape_juice_amount (initial_mixture_volume : ℕ) (initial_concentration : ℝ) (final_concentration : ℝ) : ℝ :=
  let initial_grape_juice := initial_mixture_volume * initial_concentration
  let total_volume := initial_mixture_volume + final_concentration * (final_concentration - initial_grape_juice) / (1 - final_concentration) -- Total volume after adding x gallons
  let added_grape_juice := total_volume - initial_mixture_volume -- x gallons added
  added_grape_juice

theorem grape_juice_problem :
  grape_juice_amount 40 0.20 0.36 = 10 := 
by
  sorry

end grape_juice_problem_l606_606912


namespace least_integer_greater_than_sqrt_450_l606_606479

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, (n > real.sqrt 450) ∧ ∀ m : ℤ, (m > real.sqrt  450) → n ≤ m :=
by
  apply exists.intro 22
  sorry

end least_integer_greater_than_sqrt_450_l606_606479


namespace knights_and_liars_l606_606278

-- Variables and definitions
def senator := ℕ -- assuming each senator can be represented by a natural number

-- Conditions
def num_senators : ℕ := 100
def is_knight (s : senator) : Prop
def is_liar (s : senator) : Prop := ¬ is_knight s

-- At least one knight
axiom at_least_one_knight : ∃ s : senator, is_knight s

-- Out of any two randomly chosen senators, at least one is a liar
axiom at_least_one_liar_in_pair : ∀ s1 s2 : senator, s1 ≠ s2 → is_liar s1 ∨ is_liar s2

-- Proof goal
theorem knights_and_liars : 
  ∃ num_knights num_liars : ℕ,
    num_knights + num_liars = num_senators ∧
    num_knights = 1 ∧
    num_liars = num_senators - 1 :=
begin
  sorry -- Proof to be provided
end

end knights_and_liars_l606_606278


namespace painted_cube_probability_l606_606980

/-- Define the probability of specific event that painted cube can be placed on a horizontal surface
    so that the four vertical faces are all the same color.-/
theorem painted_cube_probability :
  let total_combinations := (3: ℕ) ^ 6,
      same_color_faces := (3: ℕ) * nat.choose 6 6 + (3: ℕ) * nat.choose 6 5 + (3: ℕ) * 2 * 1 * 3 in
  (same_color_faces: ℚ) / (total_combinations: ℚ) = 111 / 729 :=
by
  sorry

end painted_cube_probability_l606_606980


namespace circle_equation_proof_l606_606862

noncomputable def equation_of_circle : Prop :=
  ∃ (a b r : ℝ), 
  (a + b = 0) ∧
  ((0 - a)^2 + (2 - b)^2 = r^2) ∧
  ((-4 - a)^2 + (0 - b)^2 = r^2) ∧
  ((x - a)^2 + (y - b)^2 = r^2) =
  ((x + 3)^2 + (y - 3)^2 = 10)

theorem circle_equation_proof : equation_of_circle :=
sorry

end circle_equation_proof_l606_606862


namespace Elijah_total_cards_l606_606631

theorem Elijah_total_cards :
  let playing_cards := 6 * 52
  let pinochle_cards := 4 * 48
  let tarot_cards := 2 * 78
  let uno_cards := 3 * 108
  playing_cards + pinochle_cards + tarot_cards + uno_cards = 984 :=
by
  let playing_cards := 6 * 52
  let pinochle_cards := 4 * 48
  let tarot_cards := 2 * 78
  let uno_cards := 3 * 108
  have h1 : playing_cards = 312 := by rfl
  have h2 : pinochle_cards = 192 := by rfl
  have h3 : tarot_cards = 156 := by rfl
  have h4 : uno_cards = 324 := by rfl
  have h_total : 312 + 192 + 156 + 324 = 984 := by norm_num
  rwa [h1, h2, h3, h4]

end Elijah_total_cards_l606_606631


namespace find_angle_A_find_a_l606_606737

noncomputable theory
open_locale classical

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π ∧
  a = 2 * sin B ∧
  (b + c = 4 * sqrt 2) ∧
  (1 / 2) * b * c * sin A = 2

theorem find_angle_A (a b c A B C : ℝ) (h : triangle_ABC a b c A B C) :
  A = π / 6 :=
begin
  sorry,
end

theorem find_a (b c a A B C : ℝ) (h : triangle_ABC a b c A B C) :
  a = 2 * (sqrt 3 - 1) :=
begin
  sorry,
end

end find_angle_A_find_a_l606_606737


namespace necessary_but_not_sufficient_condition_l606_606555

variable (a : ℝ)

theorem necessary_but_not_sufficient_condition (h : 0 ≤ a ∧ a ≤ 4) :
  (∀ x : ℝ, x^2 + a * x + a > 0) → (0 ≤ a ∧ a ≤ 4 ∧ ¬ (∀ x : ℝ, x^2 + a * x + a > 0)) :=
sorry

end necessary_but_not_sufficient_condition_l606_606555


namespace solve_for_z_l606_606224

variable (z : ℂ)

theorem solve_for_z (h : (complex.I * z = 4 + 3 * complex.I)) : z = 3 - 4 * complex.I := 
sorry

end solve_for_z_l606_606224


namespace length_major_axis_of_ellipse_l606_606601

noncomputable def length_major_axis_ellipse : ℝ :=
    let F1 : ℝ × ℝ := (5, 10)
    let F2 : ℝ × ℝ := (35, 40)
    let tangent_line : ℝ → ℝ := λ y, -5
    let reflect_F1 := (5, -20)
    let distance_F1pF2 := Real.sqrt ((35 - 5)^2 + (40 - (-20))^2)
    2 * distance_F1pF2

theorem length_major_axis_of_ellipse : length_major_axis_ellipse = 30 * Real.sqrt 5 := sorry

end length_major_axis_of_ellipse_l606_606601


namespace triangle_angle_tangent_l606_606268

theorem triangle_angle_tangent (a b c : ℝ) (A B C : ℝ) (h1 : A < π / 2) 
  (h2 : 0 < b) (h3 : 0 < c) (h4 : b = c / (1/2 + sqrt 3)) (h5 : 1/2 * b * c * sin A = sqrt 3 / 4 * b * c) : 
  A = π / 3 ∧ tan B = 1 / 2 :=
sorry

end triangle_angle_tangent_l606_606268


namespace octahedron_minimal_distance_l606_606627

theorem octahedron_minimal_distance (a : ℝ) :
  ∃ (M : ℝ) (N : ℝ), M ≠ N ∧ -- Ensure M and N are distinct
  (∃ (P : ℝ), is_octahedron P ∧ -- P denotes a point validating octahedron structure
  min_distance_on_skew_edges a = (a * Real.sqrt 6) / 3) := 
  sorry

-- Define what an octahedron is, its properties can be defined using
-- the existing Mathlib structures defined for geometric computations.

def is_octahedron (x : ℝ) : Prop := 
  -- properties of a regular octahedron with edge length x should be defined here
  sorry

def min_distance_on_skew_edges (x: ℝ) : ℝ := 
  -- computation of the minimum distance as per characteristics of regular octahedron
  sorry

end octahedron_minimal_distance_l606_606627


namespace students_more_than_rabbits_by_64_l606_606056

-- Define the conditions as constants
def number_of_classrooms : ℕ := 4
def students_per_classroom : ℕ := 18
def rabbits_per_classroom : ℕ := 2

-- Define the quantities that need calculations
def total_students : ℕ := number_of_classrooms * students_per_classroom
def total_rabbits : ℕ := number_of_classrooms * rabbits_per_classroom
def difference_students_rabbits : ℕ := total_students - total_rabbits

-- State the theorem to be proven
theorem students_more_than_rabbits_by_64 :
  difference_students_rabbits = 64 := by
  sorry

end students_more_than_rabbits_by_64_l606_606056


namespace football_tournament_l606_606733

theorem football_tournament (points: List ℕ) (n: ℕ) 
    (h₀ : points = [16, 14, 10, 10, 8, 6, 5, 3])
    (h₁ : (n - 1) * n = 72) 
    (h₂ : points.length = n - 1) 
    (h₃ : ∀x ∈ points, x ≤ 16) : 
    n = 9 ∧ 
    ((16 - points.head!) + 
     (16 - List.nthLe points 1 (by linarith)) + 
     (16 - List.nthLe points 2 (by linarith)) + 
     (16 - List.nthLe points 3 (by linarith))) = 14 :=
by
  sorry

end football_tournament_l606_606733


namespace Q_E_F_collinear_l606_606780

noncomputable def P : ℝ := sorry    -- Declare P as a point
noncomputable def Q : ℝ := sorry    -- Declare Q as a point

variables {A B C D E F : ℝ}  -- Declare variables as points on the real number line

-- Definitions of points and harmonic mean property into conditions.
def cyclic_quadrilateral (A B C D : ℝ) : Prop := sorry -- Definition of cyclic quadrilateral
def intersection_at_P (A B C D P : ℝ) : Prop := sorry -- Definition when AB and CD intersect at P
def intersection_at_Q (B C D A Q : ℝ) : Prop := sorry -- Definition when BC and DA intersect at Q
def harmonic_mean_PE (P A B E : ℝ) : Prop := sorry -- Definition of harmonic mean for PE
def harmonic_mean_PF (P C D F : ℝ) : Prop := sorry -- Definition of harmonic mean for PF

-- The main theorem we need to prove
theorem Q_E_F_collinear
  (h_cyclic: cyclic_quadrilateral A B C D)
  (h_inter_P: intersection_at_P A B C D P)
  (h_inter_Q: intersection_at_Q B C D A Q)
  (h_harmonic_PE: harmonic_mean_PE P A B E)
  (h_harmonic_PF: harmonic_mean_PF P C D F) :
  collinear Q E F :=
sorry

end Q_E_F_collinear_l606_606780


namespace semicircle_circumference_is_correct_l606_606856

-- Let p be π, and dtype be the type for dimensions
-- We'll use real numbers in this proof
open Real

-- We'll define the data and parameters
def length := 20
def breadth := 14
def rectangle_perimeter := 2 * (length + breadth)

def side_of_square (x : ℝ) := 4 * x = rectangle_perimeter

def diameter_of_semicircle := 68 / 4 -- since the perimeter of the square is same as the rectangle

def circumference_of_semicircle (d : ℝ) := (pi * d) / 2 + d

-- Now we state the problem to be proven in Lean 4
theorem semicircle_circumference_is_correct :
  let s := (68 / 4 : ℝ) in
  let C_semi := (π * s) / 2 + s in
  x = 43.69 := 
by
  sorry

end semicircle_circumference_is_correct_l606_606856


namespace frac_eq_l606_606620

def my_at (a b : ℕ) := a * b + b^2
def my_hash (a b : ℕ) := a^2 + b + a * b^2

theorem frac_eq : my_at 4 3 / my_hash 4 3 = 21 / 55 :=
by
  sorry

end frac_eq_l606_606620


namespace sergey_records_same_distinct_counts_as_alyosha_l606_606943

theorem sergey_records_same_distinct_counts_as_alyosha
  (boxes : List ℕ)
  (k : ℕ)
  (h_distinct_counts : (boxes.eraseDuplicates.length = k)) :
  let count_distinct_trays := List.range (List.maximum boxes).getD 0 + 1
    .map (λ n => boxes.map (λ b => if b - n > 0 then 1 else 0).sum)
  in count_distinct_trays.eraseDuplicates.length = k :=
by
  let trays := List.range (List.maximum boxes).getD 0 + 1
    .map (λ n => boxes.map (λ b => if b - n > 0 then 1 else 0).sum)
  sorry

end sergey_records_same_distinct_counts_as_alyosha_l606_606943


namespace shortest_distance_evil_league_l606_606846

/-- The Evil League of Evil plans to set out from their headquarters at (5,1) to poison two pipes:
    one along the line y = x and the other along the line x = 7. They wish to determine the shortest
    distance they can travel to visit both pipes and then return to their headquarters. -/
theorem shortest_distance_evil_league :
  let headquarters := (5, 1)
  let pipe1 := λ x y : ℝ, y = x
  let pipe2 := λ x y : ℝ, x = 7
  shortest_distance headquarters pipe1 pipe2 = 4 * Real.sqrt 5 := sorry

end shortest_distance_evil_league_l606_606846


namespace repeating_decimal_sum_l606_606083

theorem repeating_decimal_sum : (0.\overline{6} + 0.\overline{2} - 0.\overline{4} : ℚ) = (4 / 9 : ℚ) :=
by
  sorry

end repeating_decimal_sum_l606_606083


namespace complex_solution_l606_606202

theorem complex_solution (z : ℂ) (h : ((0 : ℝ) + 1 * z = 4 + 3 * (complex.I))) : 
  z = 3 - 4 * (complex.I) :=
sorry

end complex_solution_l606_606202


namespace sin_double_angle_of_fourth_quadrant_l606_606687

theorem sin_double_angle_of_fourth_quadrant (θ : ℝ) (h1 : 0 < θ ∧ θ < π/2 ∨ 3*π/2 < θ ∧ θ < 2*π) (h2 : cos θ = 4/5) :
  sin (2 * θ) = -24/25 :=
by
  sorry

end sin_double_angle_of_fourth_quadrant_l606_606687


namespace dishes_per_day_l606_606767

def signature_dish_crab_meat : ℝ := 1.5
def crab_meat_price_per_pound : ℝ := 8
def weekly_expenditure : ℝ := 1920
def closed_days_per_week : ℕ := 3
def open_days_per_week : ℕ := 7 - closed_days_per_week

theorem dishes_per_day :
  let daily_expenditure := weekly_expenditure / open_days_per_week in
  let cost_per_dish := signature_dish_crab_meat * crab_meat_price_per_pound in
  daily_expenditure / cost_per_dish = 40 := 
by
  sorry

end dishes_per_day_l606_606767


namespace sixth_equation_proof_nth_equation_proof_calculate_sum_l606_606610

-- Define the nth equation
def nth_equation (n : ℕ) : Prop :=
  (n+1)^2 - 1 = n * (n + 2)

-- Define the sum to be calculated over odd indices from 1 to 2023
def partial_sum (sum : ℝ) : Prop :=
  sum = 0.5 - 1 / 2025

theorem sixth_equation_proof : nth_equation 6 :=
  by sorry

theorem nth_equation_proof (n : ℕ) : n ≥ 1 → nth_equation n :=
  by sorry

theorem calculate_sum : partial_sum (∑ k in finset.range 1012, 1 / (2 * k + 1) * (2 * k + 1 + 2)) :=
  by sorry

end sixth_equation_proof_nth_equation_proof_calculate_sum_l606_606610


namespace least_integer_gt_sqrt_450_l606_606446

theorem least_integer_gt_sqrt_450 : 
  let x := 450 in
  (21 * 21 = 441) → 
  (22 * 22 = 484) →
  (441 < x ∧ x < 484) →
  (∃ n : ℤ, n = 22 ∧ n > int.sqrt x) :=
by
  intros h1 h2 h3
  use 22
  split
  · refl
  · sorry

end least_integer_gt_sqrt_450_l606_606446


namespace box_height_is_sqrt_65_l606_606906

noncomputable def box_height : ℝ :=
  let r_large := 3
  let r_small := 2
  let box_width := 6
  let h := (6 : ℝ) -- Initially, the height of the box is not known
  h

theorem box_height_is_sqrt_65 :
  let r_large := 3
  let r_small := 2
  let box_width := 6
  let h := sqrt 65
  r_large + r_small + (h - 2) = sqrt 65 :=
by 
  sorry

end box_height_is_sqrt_65_l606_606906


namespace solve_complex_equation_l606_606210

theorem solve_complex_equation (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : 
  z = 3 - 4 * complex.I :=
sorry

end solve_complex_equation_l606_606210


namespace inequality_proof_l606_606543

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l606_606543


namespace find_sum_l606_606012

variable {f : ℝ → ℝ}

-- Conditions of the problem
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def condition_2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
def condition_3 (f : ℝ → ℝ) : Prop := f 1 = 9

theorem find_sum (h_odd : odd_function f) (h_cond2 : condition_2 f) (h_cond3 : condition_3 f) :
  f 2010 + f 2011 + f 2012 = -9 :=
sorry

end find_sum_l606_606012


namespace terry_mary_same_color_combination_l606_606920

theorem terry_mary_same_color_combination :
  let red_candies := 10
  let blue_candies := 10
  let total_candies := red_candies + blue_candies
  let terry_prob := (red_candies * (red_candies - 1) + blue_candies * (blue_candies - 1)) / (total_candies * (total_candies - 1))
  let mary_given_terry_red_prob := ((blue_candies * (blue_candies - 1)) / ((total_candies - 2) * (total_candies - 3))) + 
                                   ((red_candies - 2) * (red_candies - 3) / ((total_candies - 2) * (total_candies - 3)))
  let mary_given_terry_blue_prob := ((blue_candies - 2) * (blue_candies - 3) / ((total_candies - 2) * (total_candies - 3))) + 
                                    ((red_candies) * (red_candies - 1) / ((total_candies - 2) * (total_candies - 3)))
  let combined_prob := 2 * ((terry_prob * mary_given_terry_red_prob) + (terry_prob * mary_given_terry_blue_prob))
  combined_prob = 73 / 323 :=
sorry

end terry_mary_same_color_combination_l606_606920


namespace expectation_neg_xi_l606_606730

noncomputable def xi (ω : Ω) : ℕ := sorry  -- Placeholder for the random variable
variable (ω : Ω)

def binomial_xi := ∀ ω, IsBinomial (5 : ℕ) (1/4 : ℝ) (xi ω)

theorem expectation_neg_xi (h : binomial_xi ω) : E (- xi ω) = - (5 : ℝ) / 4 :=
sorry

end expectation_neg_xi_l606_606730


namespace part1_part2_part3_l606_606828

-- Part 1
theorem part1 (a b : ℝ) : 
    3 * (a - b) ^ 2 - 6 * (a - b) ^ 2 + 2 * (a - b) ^ 2 = - (a - b) ^ 2 := 
    sorry

-- Part 2
theorem part2 (x y : ℝ) (h : x ^ 2 - 2 * y = 4) : 
    3 * x ^ 2 - 6 * y - 21 = -9 := 
    sorry

-- Part 3
theorem part3 (a b c d : ℝ) (h1 : a - 5 * b = 3) (h2 : 5 * b - 3 * c = -5) (h3 : 3 * c - d = 10) : 
    (a - 3 * c) + (5 * b - d) - (5 * b - 3 * c) = 8 := 
    sorry

end part1_part2_part3_l606_606828


namespace factorize_expression_l606_606638

variable (x y : ℝ)

theorem factorize_expression : (x - y) ^ 2 + 2 * y * (x - y) = (x - y) * (x + y) := by
  sorry

end factorize_expression_l606_606638


namespace change_one_road_l606_606739

theorem change_one_road (n : ℕ) (h: n = 100)
  (G : SimpleGraph (Fin n))
  (H : ∀ (i j : Fin n), i ≠ j → G.Adj i j ∨ G.Adj j i) :
  ∃ (G' : SimpleGraph (Fin n)),
  (∀ i j : Fin n, i ≠ j → G' ↦r i j) ∧
  ∃ e : Sym2 (Fin n), G.EdgeSet e ↔ G'.EdgeSet e :=
sorry

end change_one_road_l606_606739


namespace h_x_expression_range_of_k_interval_condition_l606_606678

-- Problem (1)
theorem h_x_expression :
  ∀ (x : ℝ), (∀ (x : ℝ), f x = x^2 + 2x) ∧ (∀ (x : ℝ), g x = -x^2 + 2x) ∧ (∀ (x : ℝ), h x = 2x) ∧ 
             (∀ (x : ℝ), (f x) ≥ (h x) ∧ (h x) ≥ (g x)) → 
             (∀ (x : ℝ), h x = 2x) :=
by sorry

-- Problem (2)
theorem range_of_k :
  ∀ (x : ℝ), (∀ (x : ℝ), f x = x^2 - x + 1) ∧ (∀ (x : ℝ), g x = k * real.log x) ∧ (∀ (x : ℝ), h x = k * x - k) ∧ 
             (∀ (x : ℝ), 0 < x) ∧ (k ∈ set.Icc (0:ℝ) 3) →
             (∀ (x : ℝ), (f x) ≥ (h x) ∧ (h x) ≥ (g x)) :=
by sorry

-- Problem (3)
theorem interval_condition:
  ∀ (x : ℝ) (t : ℝ), (∀ (x : ℝ), f x = x^4 - 2x^2) ∧ (∀ (x : ℝ), g x = 4x^2 - 8) ∧ 
             (∀ (x : ℝ), h x = 4 * (t^3 - t) * x - 3 * t^4 + 2 * t^2) ∧ (0 < |t| ∧ |t| ≤ sqrt 2) ∧ 
             (∀ (x : ℝ), m ≤ x ∧ x ≤ n ∧ - sqrt 2 ≤ x ∧ x ≤ sqrt 2) →
             n - m ≤ sqrt 7 :=
by sorry

end h_x_expression_range_of_k_interval_condition_l606_606678


namespace probability_sum_l606_606273

noncomputable def ballsInBox := 9
noncomputable def blackBalls := 5
noncomputable def whiteBalls := 4

def P_A := (blackBalls : ℚ) / (ballsInBox : ℚ)
def P_B_given_A := (whiteBalls : ℚ) / ((ballsInBox - 1) : ℚ)
def P_AB := P_A * P_B_given_A
def P_B := P_B_given_A

theorem probability_sum :
  P_AB + P_B = 7 / 9 :=
by
  sorry

end probability_sum_l606_606273


namespace fraction_of_females_this_year_l606_606363

theorem fraction_of_females_this_year 
  (participation_increase : ℝ := 0.15)
  (males_increase : ℝ := 0.10)
  (females_increase : ℝ := 0.25)
  (males_last_year : ℤ := 30) :
  let males_this_year := males_last_year * (1 + males_increase)
  let total_participants_last_year := males_last_year + y
  let y : ℝ := (total_participants_increase - males_this_year - males_last_year * females_increase) / females_increase
  let females_this_year := y * (1 + females_increase)
  let total_participants_this_year := total_participants_last_year * (1 + participation_increase)
  in total_participants_this_year = males_this_year + females_this_year →
  females_this_year / total_participants_this_year = 19 / 52 :=
by
  sorry

end fraction_of_females_this_year_l606_606363


namespace trisect_angle_with_compass_ruler_iff_polynomial_reducible_l606_606508

open Polynomial

theorem trisect_angle_with_compass_ruler_iff_polynomial_reducible (θ : ℝ) :
  (∃ α, 3 * α = θ ∧ constructible α) ↔
  ∃ (f : Polynomial ℚ), f = Polynomial.C (4 : ℚ) * X^3 - Polynomial.C (3 : ℚ) * X - (algebra_map ℝ ℚ (cos θ)) ∧ f.is_reducible :=
sorry

end trisect_angle_with_compass_ruler_iff_polynomial_reducible_l606_606508


namespace cone_volume_l606_606558

theorem cone_volume (r l : ℝ) (h : ℝ) (H1 : r = 3) (H2 : l = 5) (H3 : h = real.sqrt (l^2 - r^2)) : 
  1/3 * real.pi * r^2 * h = 12 * real.pi := 
by 
  rw [H1, H2, H3]
  norm_num
  sorry

end cone_volume_l606_606558


namespace correct_quadratic_graph_l606_606394

theorem correct_quadratic_graph (a b c : ℝ) (ha : a > 0) (hb : b < 0) (hc : c < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (-b / (2 * a) > 0) ∧ (c < 0) :=
by
  sorry

end correct_quadratic_graph_l606_606394


namespace chickpea_flour_amount_l606_606374

def total_flour (rye_flour whole_wheat_bread_flour whole_wheat_pastry_flour chickpea_flour : ℕ) : ℕ :=
  rye_flour + whole_wheat_bread_flour + whole_wheat_pastry_flour + chickpea_flour

theorem chickpea_flour_amount 
  (rye_flour wh_bread_flour wh_pastry_flour total_flour : ℕ)
  (h1 : rye_flour = 5)
  (h2 : wh_bread_flour = 10)
  (h3 : wh_pastry_flour = 2)
  (h4 : total_flour = 20) :
  ∃ (chickpea_flour : ℕ), chickpea_flour = 3 :=
by
  let total_before = rye_flour + wh_bread_flour + wh_pastry_flour
  have hb : total_before = 17 := by rw [h1, h2, h3]; norm_num
  have hc : 20 - total_before = 3 := by rw [←h4, hb]; norm_num
  use (20 - total_before)
  rw hc
  done

end chickpea_flour_amount_l606_606374


namespace primes_if_and_only_if_factorial_condition_l606_606654

theorem primes_if_and_only_if_factorial_condition (n : ℕ) (h1 : n > 1) 
  (h2 : odd n) : 
  (nat.prime n ∧ nat.prime (n + 2)) ↔ ((n - 1)! ∣ n = false ∧ (n - 1)! ∣ (n + 2) = false) :=
sorry

end primes_if_and_only_if_factorial_condition_l606_606654


namespace inequality_proof_l606_606545

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l606_606545


namespace train_speed_kmph_l606_606921

noncomputable def jogger_speed_kmph : ℕ := 9
noncomputable def jogger_ahead_meters : ℕ := 240
noncomputable def train_length_meters : ℕ := 120
noncomputable def time_to_pass_seconds : ℕ := 36

theorem train_speed_kmph :
  let total_distance_meters := jogger_ahead_meters + train_length_meters in
  let relative_speed_mps := total_distance_meters / time_to_pass_seconds in
  let relative_speed_kmph := relative_speed_mps * 3.6 in
  let train_speed_kmph := relative_speed_kmph + jogger_speed_kmph in
  train_speed_kmph = 45 := by
  sorry

end train_speed_kmph_l606_606921


namespace median_of_sequence_l606_606616

theorem median_of_sequence : 
  let seq := List.join (List.map (fun n => List.replicate n n) (List.range' 1 151))
  let N := seq.length
  let median_pos := (N + 1) / 2
  (List.nth_le seq (median_pos - 1) (by { simp [seq] })) = 106 := 
by
  sorry

end median_of_sequence_l606_606616


namespace max_value_of_distances_l606_606571

open Real

noncomputable def curve_M_parametric (φ : ℝ) : ℝ × ℝ :=
  (1 + cos φ, 1 + sin φ)

noncomputable def polar_equation_M (ρ θ : ℝ) : ℝ :=
  ρ^2 - 2 * ρ *(cos θ + sin θ) + 1

noncomputable def l1_polar_equation (θ α : ℝ) (ρ : ℝ) : Prop :=
  θ = α

theorem max_value_of_distances (α : ℝ) 
  (h1 : α ∈ Ioc 0 (π / 6))
  (h2 : ∀ ρ θ, polar_equation_M ρ θ = 0) :
  max (2 + 2 * sqrt 3) = 2 + 2 * sqrt 3 :=
by
  sorry

end max_value_of_distances_l606_606571


namespace trace_of_perpendicular_planes_l606_606041

theorem trace_of_perpendicular_planes
    (Π1 Π2 Π3 : Plane)
    (h1 : Π1 ⊥ Π2)
    (h2 : Π2 ⊥ Π3)
    (h3 : Π1 ⊥ Π3)
    (A : Point) (B : Point) (C : Point)
    (hA : A ∈ Π1)
    (hB : B ∈ Π2)
    (hC : C ∈ Π3) :
    ∃(traces : Set Plane), 
    (∀ t ∈ traces, t ∩ Π1 = {A}) ∧ 
    (∀ t ∈ traces, t ∩ Π2 = {B}) ∧ 
    (∀ t ∈ traces, t ∩ Π3 = {C}) := sorry

end trace_of_perpendicular_planes_l606_606041


namespace largest_prime_factor_of_divisors_sum_180_l606_606791

def divisors_sum (n : ℕ) : ℕ :=
  (divisors n).sum

def largest_prime_factor (n : ℕ) : ℕ :=
  (factors n).filter prime ∣> ∧ .toList.maximum' sorry -- assume there is a maximum

theorem largest_prime_factor_of_divisors_sum_180 :
  ∃ N, N = divisors_sum 180 ∧ largest_prime_factor N = 13 := by
  sorry

end largest_prime_factor_of_divisors_sum_180_l606_606791


namespace hiking_packing_weight_l606_606588

theorem hiking_packing_weight
  (miles_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days : ℝ)
  (supply_per_mile : ℝ)
  (resupply_fraction : ℝ)
  (expected_first_pack_weight : ℝ)
  (hiking_hours : ℝ := hours_per_day * days)
  (total_miles : ℝ := miles_per_hour * hiking_hours)
  (total_weight_needed : ℝ := supply_per_mile * total_miles)
  (resupply_weight : ℝ := resupply_fraction * total_weight_needed)
  (first_pack_weight : ℝ := total_weight_needed - resupply_weight) :
  first_pack_weight = 37.5 :=
by
  -- The proof goes here, but is omitted since the proof is not required.
  sorry

end hiking_packing_weight_l606_606588


namespace length_ST_l606_606411

-- Definition of the given constants and triangle sides
def PQ := 5
def QR := 7
def PR := 8

-- Conditions on points S and T based on the problem
def PS := PQ + 1 -- Example value satisfying PQ < PS
def PT := PS + 1 -- Example value satisfying PS < PT

-- Intersection point U with given distances from S and T
def SU := 3
def TU := 8

-- The main theorem to prove
theorem length_ST : ST = 4 * Real.sqrt (71/20) :=
by
  let cos_Q := (PQ^2 + PR^2 - QR^2) / (2 * PQ * PR)
  let ST := Real.sqrt (SU^2 + TU^2 - 2 * SU * TU * cos_Q)
  -- Target length
  have target_ST := 4 * Real.sqrt (71/20)
  -- Prove the main equality
  sorry

end length_ST_l606_606411


namespace lines_concurrent_l606_606033

theorem lines_concurrent
  (n : ℕ)
  (lines : fin n → (ℂ → Prop))
  (color : fin n → Prop)
  (h1 : ∀ i j, i ≠ j → ¬ ∀ x, lines i x ↔ lines j x)
  (h2 : ∀ (i j : fin n) (x : ℂ), i ≠ j → lines i x → lines j x → ∃ k, k ≠ i ∧ k ≠ j ∧ ¬ lines k x) :
  ∃ P : ℂ, ∀ i, lines i P :=
by
  sorry

end lines_concurrent_l606_606033


namespace verify_triangle_inequality_l606_606367

-- Conditions of the problem
variables (L : ℕ → ℕ)
-- The rods lengths are arranged in increasing order
axiom rods_in_order : ∀ i : ℕ, L i ≤ L (i + 1)

-- Define the critical check
def critical_check : Prop :=
  L 98 + L 99 > L 100

-- Prove that verifying the critical_check is sufficient
theorem verify_triangle_inequality (h : critical_check L) :
  ∀ i j k : ℕ, 1 ≤ i → i < j → j < k → k ≤ 100 → L i + L j > L k :=
by
  sorry

end verify_triangle_inequality_l606_606367


namespace turnips_total_l606_606305

theorem turnips_total (Keith_turnips Alyssa_turnips : ℕ) (hK : Keith_turnips = 6) (hA : Alyssa_turnips = 9) : Keith_turnips + Alyssa_turnips = 15 :=
by
  rw [hK, hA]
  exact rfl

end turnips_total_l606_606305


namespace tens_digit_2031_pow_2024_minus_2033_l606_606492

theorem tens_digit_2031_pow_2024_minus_2033 :
  (let tens_digit n := (n / 10) % 10 in
   tens_digit (2031 ^ 2024 - 2033) = 8) :=
by
  sorry

end tens_digit_2031_pow_2024_minus_2033_l606_606492


namespace last_two_digits_of_1032_pow_1032_l606_606995

noncomputable def last_two_digits (n : ℕ) : ℕ :=
  n % 100

theorem last_two_digits_of_1032_pow_1032 : last_two_digits (1032^1032) = 76 := by
  sorry

end last_two_digits_of_1032_pow_1032_l606_606995


namespace prove_equal_ratio_l606_606726

variables {D E F N : Type} [T : triangle D E F]
noncomputable def DN := (14 : ℕ)
noncomputable def EF := (15 : ℕ)
noncomputable def DF := (21 : ℕ)

def on_segment (A B P : Point) : Prop :=
  P ∈ segment A B

def equal_incircles (T1 T2 : triangle) : Prop :=
  inradius T1 = inradius T2

variable (N : Point)
variable (DN : ℕ)
variable (NF : ℕ)
variable (p q : ℕ)

axiom point_on_segment : on_segment D F N
axiom equal_incircle_radii : equal_incircles (triangle D E N) (triangle E F N)

theorem prove_equal_ratio : p + q = sorry :=
sorry

end prove_equal_ratio_l606_606726


namespace minor_arc_length_eq_twenty_two_over_three_pi_l606_606312

-- Given points P, Q, and R on a circle of radius 12
-- and an angle ∠PRQ = 110 degrees, prove the circumference 
-- of the minor arc PQ equals to (22 / 3) * pi

theorem minor_arc_length_eq_twenty_two_over_three_pi
  (P Q R : Type)
  (radius : ℝ)
  (angle_PRQ : ℝ)
  (circle : ∀ (X : Type), X ∈ set_of (c : X → ℝ) | c = 12)
  (h_radius : radius = 12)
  (h_angle_PRQ : angle_PRQ = 110 * (π / 180)) :
  let circumference := 2 * π * radius in
  let arc_length := (110 / 360) * circumference in
  arc_length = (22 / 3) * π :=
by
  sorry

end minor_arc_length_eq_twenty_two_over_three_pi_l606_606312


namespace union_of_sets_l606_606683

variable (A : Set ℤ) (B : Set ℤ)

theorem union_of_sets (hA : A = {0, 1, 2}) (hB : B = {-1, 0}) : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_sets_l606_606683


namespace hiking_packing_weight_l606_606589

theorem hiking_packing_weight
  (miles_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days : ℝ)
  (supply_per_mile : ℝ)
  (resupply_fraction : ℝ)
  (expected_first_pack_weight : ℝ)
  (hiking_hours : ℝ := hours_per_day * days)
  (total_miles : ℝ := miles_per_hour * hiking_hours)
  (total_weight_needed : ℝ := supply_per_mile * total_miles)
  (resupply_weight : ℝ := resupply_fraction * total_weight_needed)
  (first_pack_weight : ℝ := total_weight_needed - resupply_weight) :
  first_pack_weight = 37.5 :=
by
  -- The proof goes here, but is omitted since the proof is not required.
  sorry

end hiking_packing_weight_l606_606589


namespace number_of_small_balls_l606_606708

theorem number_of_small_balls (R r : ℝ) (V : ℝ → ℝ) (π : ℝ) (h : r = R / 4) :
  V (R : ℝ) = (4 / 3) * π * R^3 →
  V (r : ℝ) = (4 / 3) * π * r^3 →
  (V (R) / V (r) = 64) :=
by 
  -- Definitions
  have V_R := V R,
  have V_r := V r,
  -- Volume of large iron ball
  have V_R_eq : V_R = (4 / 3) * π * R ^ 3 := by assumption,
  -- Volume of small iron ball
  have V_r_eq : V_r = (4 / 3) * π * (R / 4) ^ 3 := by assumption,
  have V_r_eq' : V_r = (4 / 3) * π * (R ^ 3 / 64) := by
    rw [div_pow, mul_pow, h],
  rw [V_r_eq, V_r_eq'],
  simp,
  sorry

end number_of_small_balls_l606_606708


namespace scientific_notation_eq_l606_606637

theorem scientific_notation_eq (n : ℕ) (h : n = 32000000) : 
    let s := 3.2
    let t := 10^7
    n = s * t := 
by
  have n_eq : n = 32000000 := h
  let s := 3.2
  let t := 10^7
  suffices : 32000000 = 3.2 * 10^7 by
    rw n_eq
    assumption
  sorry

end scientific_notation_eq_l606_606637


namespace find_sin_theta_l606_606314

noncomputable def condition1 (a b c d : ℝ^3) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :=
  ¬∃ k : ℝ, k ≠ 0 ∧ (b = k • d ∨ d = k • b)

noncomputable def condition2 (a b c d : ℝ^3) :=
  (a × b) × (c × d) = (1 / 4) * ‖b‖ * ‖d‖ * a

theorem find_sin_theta {a b c d : ℝ^3} (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0)
  (h_parallel : condition1 a b c d h_a h_b h_c h_d) (h_eq : condition2 a b c d) :
  ∃ θ : ℝ, sin θ = (√15) / 4 :=
sorry

end find_sin_theta_l606_606314


namespace max_days_parliament_l606_606629

def set_of_politicians : Set (Set ℕ) := {s | s ≠ ∅ ∧ ∀ t ∈ set_of_politicians, s ≠ t ∧ s ∩ t ≠ ∅}

theorem max_days_parliament (U : Finset ℕ) (hU : U.card = 8) : 
  ∃ n ≤ 128, ∀ (days : Fin n (Set ℕ)), 
  (∀ i, days i ≠ ∅) ∧ 
  (∀ i j, i ≠ j → days i ≠ days j) ∧ 
  (∀ i < n, ∀ k < i, ∃ x ∈ days i, x ∈ days k) := 
sorry

end max_days_parliament_l606_606629


namespace gardener_payment_l606_606868

theorem gardener_payment (total_cost : ℕ) (rect_area : ℕ) (rect_side1 : ℕ) (rect_side2 : ℕ)
                         (square1_area : ℕ) (square2_area : ℕ) (cost_per_are : ℕ) :
  total_cost = 570 →
  rect_area = 600 → rect_side1 = 20 → rect_side2 = 30 →
  square1_area = 400 → square2_area = 900 →
  cost_per_are * (rect_area + square1_area + square2_area) / 100 = total_cost →
  cost_per_are = 30 →
  ∃ (rect_payment : ℕ) (square1_payment : ℕ) (square2_payment : ℕ),
    rect_payment = 6 * cost_per_are ∧
    square1_payment = 4 * cost_per_are ∧
    square2_payment = 9 * cost_per_are ∧
    rect_payment + square1_payment + square2_payment = total_cost :=
by
  intros
  sorry

end gardener_payment_l606_606868


namespace largest_prime_factor_sum_divisors_180_l606_606793

theorem largest_prime_factor_sum_divisors_180 :
  let N := ∑ d in (Finset.divisors 180), d in
  Nat.greatest_prime_factor N = 13 :=
by
  sorry

end largest_prime_factor_sum_divisors_180_l606_606793


namespace Carmen_average_speed_l606_606611

/-- Carmen participates in a two-part cycling race. In the first part, she covers 24 miles in 3 hours.
    In the second part, due to fatigue, her speed decreases, and she takes 4 hours to cover 16 miles.
    Calculate Carmen's average speed for the entire race. -/
theorem Carmen_average_speed :
  let distance1 := 24 -- miles in the first part
  let time1 := 3 -- hours in the first part
  let distance2 := 16 -- miles in the second part
  let time2 := 4 -- hours in the second part
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 40 / 7 :=
by
  sorry

end Carmen_average_speed_l606_606611


namespace solve_for_z_l606_606221

variable (z : ℂ)

theorem solve_for_z (h : (complex.I * z = 4 + 3 * complex.I)) : z = 3 - 4 * complex.I := 
sorry

end solve_for_z_l606_606221


namespace repeating_decimals_sum_l606_606097

theorem repeating_decimals_sum : (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ) = 4 / 9 :=
 by 
  have h₁ : (0.666666... : ℚ) = 2 / 3,
    -- Since x = 0.6666..., then 10x = 6.6666...,
    -- so 10x - x = 6, then x = 6 / 9, hence 2 / 3
    sorry,
  have h₂ : (0.222222... : ℚ) = 2 / 9,
    -- Since x = 0.2222..., then 10x = 2.2222...,
    -- so 10x - x = 2, then x = 2 / 9
    sorry,
  have h₃ : (0.444444... : ℚ) = 4 / 9,
    -- Since x = 0.4444..., then 10x = 4.4444...,
    -- so 10x - x = 4, then x = 4 / 9
    sorry,
  calc
    (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ)
        = (2 / 3) + (2 / 9) - (4 / 9) : by rw [h₁, h₂, h₃]
    ... = (6 / 9) + (2 / 9) - (4 / 9) : by norm_num
    ... = 4 / 9 : by ring

end repeating_decimals_sum_l606_606097


namespace simplest_square_root_l606_606949

theorem simplest_square_root : 
  let options := [Real.sqrt 20, Real.sqrt 2, Real.sqrt (1 / 2), Real.sqrt 0.2] in
  ∃ x ∈ options, x = Real.sqrt 2 ∧ (∀ y ∈ options, y ≠ Real.sqrt 2 → ¬(Real.sqrt y).simpler_than (Real.sqrt 2)) :=
by
  let options := [Real.sqrt 20, Real.sqrt 2, Real.sqrt (1 / 2), Real.sqrt 0.2]
  have h_sqrt_2_in_options : Real.sqrt 2 ∈ options := by simp [options]
  use Real.sqrt 2
  constructor
  . exact h_sqrt_2_in_options
  . intro y hy_ne_sqrt_2 hy_options
    sorry

end simplest_square_root_l606_606949


namespace sum_f_eq_neg_one_l606_606148

noncomputable def f : ℝ → ℝ := λ x, sorry

axiom even_f : ∀ x : ℝ, f(x) = f(-x)
axiom odd_f_shift_2 : ∀ x : ℝ, f(x + 2) = -f(-x + 2)
axiom f_zero : f(0) = 1

theorem sum_f_eq_neg_one : (∑ i in (finset.range 2024).filter (λ i, i > 0), f i) = -1 :=
by
  sorry

end sum_f_eq_neg_one_l606_606148


namespace marble_count_l606_606340

theorem marble_count (x : ℕ) 
  (h1 : ∀ (Liam Mia Noah Olivia: ℕ), Mia = 3 * Liam ∧ Noah = 4 * Mia ∧ Olivia = 2 * Noah)
  (h2 : Liam + Mia + Noah + Olivia = 156)
  : x = 4 :=
by sorry

end marble_count_l606_606340


namespace arithmetic_geometric_sum_l606_606738

noncomputable def a_n (n : ℕ) := 3 * n - 2
noncomputable def b_n (n : ℕ) := 4 ^ (n - 1)

theorem arithmetic_geometric_sum (n : ℕ) :
    a_n 1 = 1 ∧ a_n 2 = b_n 2 ∧ a_n 6 = b_n 3 ∧ S_n = 1 + (n - 1) * 4 ^ n :=
by sorry

end arithmetic_geometric_sum_l606_606738


namespace solve_complex_equation_l606_606212

theorem solve_complex_equation (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : 
  z = 3 - 4 * complex.I :=
sorry

end solve_complex_equation_l606_606212


namespace ramesh_discount_l606_606827

-- Define the context for the problem
def labelled_price : ℝ := 21000
def purchase_price : ℝ := 16500
def discount_amount : ℝ := labelled_price - purchase_price
def discount_percentage : ℝ := (discount_amount / labelled_price) * 100

-- State the theorem
theorem ramesh_discount :
  discount_percentage ≈ 21.43 := sorry

end ramesh_discount_l606_606827


namespace problem_1992_AHSME_43_l606_606716

theorem problem_1992_AHSME_43 (a b c : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : Odd a) (h2 : Odd b) : Odd (3^a + (b-1)^2 * c) :=
sorry

end problem_1992_AHSME_43_l606_606716


namespace evaluate_expression_l606_606493

theorem evaluate_expression : 
  101^3 + 3 * (101^2) * 2 + 3 * 101 * (2^2) + 2^3 = 1092727 := 
by 
  sorry

end evaluate_expression_l606_606493


namespace initial_ratio_of_stamps_l606_606298

variable (K A : ℕ)

theorem initial_ratio_of_stamps (h1 : (K - 12) * 3 = (A + 12) * 4) (h2 : K - 12 = A + 44) : K/A = 5/3 :=
sorry

end initial_ratio_of_stamps_l606_606298


namespace repeating_decimals_sum_l606_606093

theorem repeating_decimals_sum : (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ) = 4 / 9 :=
 by 
  have h₁ : (0.666666... : ℚ) = 2 / 3,
    -- Since x = 0.6666..., then 10x = 6.6666...,
    -- so 10x - x = 6, then x = 6 / 9, hence 2 / 3
    sorry,
  have h₂ : (0.222222... : ℚ) = 2 / 9,
    -- Since x = 0.2222..., then 10x = 2.2222...,
    -- so 10x - x = 2, then x = 2 / 9
    sorry,
  have h₃ : (0.444444... : ℚ) = 4 / 9,
    -- Since x = 0.4444..., then 10x = 4.4444...,
    -- so 10x - x = 4, then x = 4 / 9
    sorry,
  calc
    (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ)
        = (2 / 3) + (2 / 9) - (4 / 9) : by rw [h₁, h₂, h₃]
    ... = (6 / 9) + (2 / 9) - (4 / 9) : by norm_num
    ... = 4 / 9 : by ring

end repeating_decimals_sum_l606_606093


namespace part1_part2_l606_606556

-- Definitions and conditions
variable (x y : ℝ)

def pointA := (2, 1) : ℝ × ℝ

def medianLineCM := ∀ (x y : ℝ), 2*x + y - 1 = 0
def altitudeLineBH := ∀ (x y : ℝ), x - y = 0

-- Part 1: Prove the equation of the line passing through point A with equal intercepts
theorem part1 : 
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → 
  (( ∃ k:ℝ, 0.k * 1 = (1 - 0) / (2 - 0) ) ∨
  (∃ a:ℝ, ∀ (x y:ℝ), x + y - a = 0) = 3)) → 
  (x + y - 3 = 0) := sorry

-- Part 2: Prove the equation of the line BC
theorem part2 : 
  (∀ (A B C : ℝ × ℝ), 
    A = (2,1) → 
    (∃ (m:ℝ), B = (m, m)) ∧  
    (length (A, B) = 1)
    (∃ median: ℝ, 2(A+B)*(1 median) = 0) ∧ 
    (∃ (H:ℝ), C != b h) cm(2, 1, x*y-xy+0) ∧
    (C != b c 6x+y + 7 = 0) := 
    (( ∃ k: (1-(1-0)) /(2+0).k) 
    ∨ (2x + y -1)=0 ∨ (6 * (-1-2) + 7 = ))) := sorry

end part1_part2_l606_606556


namespace lucas_initial_money_l606_606342

theorem lucas_initial_money : (3 * 2 + 14 = 20) := by sorry

end lucas_initial_money_l606_606342


namespace no_positive_integer_satisfies_conditions_l606_606657

theorem no_positive_integer_satisfies_conditions : 
  ¬ ∃ (n : ℕ), (100 ≤ n / 4) ∧ (n / 4 ≤ 999) ∧ (100 ≤ 4 * n) ∧ (4 * n ≤ 999) :=
by
  -- Proof will go here.
  sorry

end no_positive_integer_satisfies_conditions_l606_606657


namespace hyperbola_distance_between_foci_l606_606383

noncomputable def hyperbola_foci_distance {ℝ : Type*} [linear_ordered_field ℝ] : ℝ :=
  let a := real.sqrt (45 / 4) in
  2 * real.sqrt (2 * a^2)

-- The given conditions:
def asymptote1 (x y : ℝ) : Prop := y = 2 * x - 1
def asymptote2 (x y : ℝ) : Prop := y = 1 - 2 * x
def passes_through (x y : ℝ) : Prop := x = 4 ∧ y = 1

-- The question rewritten as a statement to be proved:
theorem hyperbola_distance_between_foci :
  ∀ (x y : ℝ), asymptote1 x y ∨ asymptote2 x y ∧ passes_through x y →
  2 * real.sqrt (2 * (real.sqrt(45 / 4))^2) = 3 * real.sqrt 10 :=
by simp [asymptote1, asymptote2, passes_through, hyperbola_foci_distance]; sorry

end hyperbola_distance_between_foci_l606_606383


namespace divisor_count_congruence_l606_606711

theorem divisor_count_congruence (n : ℕ) (h : 0 < n) :
  let d := ∏ i in (finset.range (nat.prime_factors n).length), (18 * (nat.prime_factors n).nat_degree + 1) in
  d % 3 = 1 := sorry

end divisor_count_congruence_l606_606711


namespace find_f_5_l606_606663

noncomputable def f (x a b : ℝ) : ℝ := x^5 - a*x^3 + b*sin x + 2

theorem find_f_5 (a b : ℝ) (h1 : f (-5) a b = 17) : f 5 a b = -13 :=
by
  sorry

end find_f_5_l606_606663


namespace floor_fraction_factorial_sum_l606_606999

-- Define the floor function
def floor (x : ℝ) : ℤ := int.floor x

-- Define the series in the denominator
def factorial_sum (n : ℕ) : ℕ := (finset.range n).sum (λ i, nat.factorial (i + 1))

-- State the theorem to be proven
theorem floor_fraction_factorial_sum :
  floor ((2002 * nat.factorial 2001 : ℝ) / (factorial_sum 2001 : ℝ)) = 2000 :=
sorry

end floor_fraction_factorial_sum_l606_606999


namespace least_integer_greater_than_sqrt_450_l606_606440

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606440


namespace N_minus_one_is_square_of_integer_l606_606404

noncomputable def proof_problem (N : ℕ) : Prop :=
  (N > 0) ∧ (
  ∃ G : Type, 
  ∃ [Fintype G],
  ∀ (a b : G), (a ≠ b) → 
  ((∃ (p : Finset (Finset G)), 
  ∀ s ∈ p, Finset.card s = 2 ∧ 
  ∀ e ∈ s, true  -- Placeholder for exactly one route between a and b using at most two flights.
  ) ∧ (∀ x : G, ∃ y : G, x ≠ y))) →
  ∃ k : ℕ, N - 1 = k^2

-- Proof placeholder
theorem N_minus_one_is_square_of_integer (N : ℕ) (hN0 : N > 0)
    (hcond: ∃ (G : Type) [Fintype G], 
      ∀ (a b : G), a ≠ b → (∃ (p : Finset (Finset G)),
      ∀ s ∈ p, Finset.card s = 2 ∧ 
      ∀ e ∈ s, true  -- Here we assume that in conditions there's a route using at most two flights 
      ) ∧ (∀ x : G, ∃ y : G, x ≠ y)
    ): proof_problem N :=
sorry

end N_minus_one_is_square_of_integer_l606_606404


namespace pyramid_volume_l606_606402

-- Definition of the variables and conditions
variables (a b : ℝ)

-- Volume of the regular triangular pyramid
theorem pyramid_volume (h : b > 0) (h1 : 3 * (3 * a^2) > 4 * b^2) : 
  let V := (a^3 * b) / (12 * sqrt(3 * a^2 - 4 * b^2)) in
  V = (a^3 * b) / (12 * sqrt(3 * a^2 - 4 * b^2)) :=
by
  sorry

end pyramid_volume_l606_606402


namespace orthocenter_condition_l606_606280

variables {A B C D H F : Type*}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] 
          [InnerProductSpace ℝ D] [InnerProductSpace ℝ H] [InnerProductSpace ℝ F]

-- Definitions of the points and properties
def is_acute_triangle (A B C : Type*) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] : Prop :=
  sorry

def is_obtuse_angle (A D B : Type*) [InnerProductSpace ℝ A] [InnerProductSpace ℝ D] [InnerProductSpace ℝ B] : Prop :=
  sorry

def is_orthocenter (H : Type*) (A B D : Type*) [InnerProductSpace ℝ H] [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ D] : Prop :=
  sorry

def on_circumcircle (F : Type*) (A B D : Type*) [InnerProductSpace ℝ F] [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ D] : Prop :=
  sorry

def parallel_lines (HD : Type*) (CF : Type*) [InnerProductSpace ℝ HD] [InnerProductSpace ℝ CF] : Prop :=
  sorry

def on_circumcircle_ABC (H : Type*) (A B C : Type*) [InnerProductSpace ℝ H] [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] : Prop :=
  sorry

-- The main theorem
theorem orthocenter_condition 
  (h1 : is_acute_triangle A B C)
  (h2 : is_obtuse_angle A D B)
  (h3 : is_orthocenter H A B D)
  (h4 : on_circumcircle F A B D) :
  (is_orthocenter F A B C ↔ parallel_lines HD CF ∧ on_circumcircle_ABC H A B C) :=
begin
  sorry
end

end orthocenter_condition_l606_606280


namespace max_label_correct_l606_606897

noncomputable def max_label (n : ℕ) (h : n > 4) : ℕ :=
  if odd n then 2 * n - 3 else 2 * n - 4

theorem max_label_correct (n : ℕ) (h : n > 4) :
  ∃ k : ℕ, (odd n → k = 2 * n - 3) ∧ (¬ odd n → k = 2 * n - 4) :=
begin
  -- Sorry - this is a placeholder.
  sorry,
end

end max_label_correct_l606_606897


namespace decreasing_interval_of_function_l606_606849

theorem decreasing_interval_of_function :
  ∀ x : ℝ, (2 * x^2 - 3 * x + 1) ∈ [3/4, +∞) ↔ (λ x, (1/2)^(2 * x^2 - 3 * x + 1)) is_decreasing_on [3/4, +∞] := by
  sorry

end decreasing_interval_of_function_l606_606849


namespace least_integer_greater_than_sqrt_450_l606_606468

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606468


namespace complex_solution_l606_606205

theorem complex_solution (z : ℂ) (h : ((0 : ℝ) + 1 * z = 4 + 3 * (complex.I))) : 
  z = 3 - 4 * (complex.I) :=
sorry

end complex_solution_l606_606205


namespace integer_solutions_count_l606_606185

theorem integer_solutions_count : 
  {x : ℤ // (x - 3)^((30 - x^2)) = 1 ∧ x + 2 > 0}.size = 2 :=
by
  sorry

end integer_solutions_count_l606_606185


namespace common_fraction_l606_606086

noncomputable def x : ℚ := 0.6666666 -- represents 0.\overline{6}
noncomputable def y : ℚ := 0.2222222 -- represents 0.\overline{2}
noncomputable def z : ℚ := 0.4444444 -- represents 0.\overline{4}

theorem common_fraction :
  x + y - z = 4 / 9 :=
by
  -- Provide proofs here
  sorry

end common_fraction_l606_606086


namespace find_z_l606_606247

theorem find_z (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : z = 3 - 4 * complex.I :=
by {
  sorry
}

end find_z_l606_606247


namespace solve_complex_equation_l606_606215

theorem solve_complex_equation (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : 
  z = 3 - 4 * complex.I :=
sorry

end solve_complex_equation_l606_606215


namespace sum_inequality_l606_606308

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (a 2 = 2) ∧ (∀ n : ℕ, ∀ h : 1 < n, a (n + 1)^4 / a n^3 = 2 * a (n + 2) - a (n + 1))

theorem sum_inequality (a : ℕ → ℝ) (h_seq : sequence a) (N : ℕ) (hN : N > 1) :
  ∑ k in Finset.range(N + 1), a k^2 / a (k + 1) < 3 := 
sorry

end sum_inequality_l606_606308


namespace a_squared_gt_b_squared_l606_606135

theorem a_squared_gt_b_squared {a b : ℝ} (h : a ≠ 0) (hb : b ≠ 0) (hb_domain : b > -1 ∧ b < 1) (h_eq : a = Real.log (1 + b) - Real.log (1 - b)) :
  a^2 > b^2 := 
sorry

end a_squared_gt_b_squared_l606_606135


namespace expected_value_is_10_l606_606382

noncomputable def expected_value_adjacent_pairs (boys girls : ℕ) (total_people : ℕ) : ℕ :=
  if total_people = 20 ∧ boys = 8 ∧ girls = 12 then 10 else sorry

theorem expected_value_is_10 : expected_value_adjacent_pairs 8 12 20 = 10 :=
by
  -- Intuition and all necessary calculations (proof steps) have already been explained.
  -- Here we are directly stating the conclusion based on given problem conditions.
  trivial

end expected_value_is_10_l606_606382


namespace beths_total_crayons_l606_606022

def packs : ℕ := 4
def crayons_per_pack : ℕ := 10
def extra_crayons : ℕ := 6

theorem beths_total_crayons : packs * crayons_per_pack + extra_crayons = 46 := by
  sorry

end beths_total_crayons_l606_606022


namespace concyclic_quadrilateral_l606_606703

theorem concyclic_quadrilateral
  (Γ1 Γ2 : Circle)
  {A B : Point}
  {P : Point}
  {Q : Point}
  {T : Point}
  [Γ1_intersects_Γ2 : Intersects Γ1 Γ2 A B]
  (P_on_Γ1 : OnCircle P Γ1)
  (Q_on_Γ2 : OnCircle Q Γ2)
  (P_B_Q_collinear : Collinear [P, B, Q])
  (T_is_tangent_intersection : IsTangentIntersection T P Q Γ2) :
  CyclicQuadrilateral A Q T P :=
sorry

end concyclic_quadrilateral_l606_606703


namespace original_profit_margin_l606_606915

theorem original_profit_margin 
  (a : ℝ) -- original purchase price
  (x : ℝ) -- original profit margin (fraction form)
  (h_price_decrease : a * (1 - 0.08)) -- decrease in purchase price by 8%
  (h_profit_increase : x + 0.10) -- increase in profit margin by 10%
  (h_eq : (a * (1 + x) - a * (1 - 0.08)) / (a * (1 - 0.08)) = x + 0.10) :
  x = 0.15 := -- 15% as fraction
sorry

end original_profit_margin_l606_606915


namespace number_of_triangles_with_perimeter_eight_l606_606179

theorem number_of_triangles_with_perimeter_eight :
  { (a, b, c) : ℕ × ℕ × ℕ // a + b + c = 8 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b }.card = 5 :=
by
  -- Declaration of the proof structure
  sorry

end number_of_triangles_with_perimeter_eight_l606_606179


namespace generalized_binomial_coefficients_integer_l606_606811

-- Given conditions
variables (a : ℕ → ℕ) (h_gcd : ∀ (m n : ℕ), a (Nat.gcd m n) = Nat.gcd (a m) (a n))

-- Integer generalized binomial coefficients
theorem generalized_binomial_coefficients_integer (n k : ℕ) (hk : k ≤ n):
  (a n * a (n-1) * ... * a (n-k+1)) / (a k * a (k-1) * ... * a 1) ∈ ℤ :=
sorry

end generalized_binomial_coefficients_integer_l606_606811


namespace contrapositive_of_equality_square_l606_606829

theorem contrapositive_of_equality_square (a b : ℝ) (h : a^2 ≠ b^2) : a ≠ b := 
by 
  sorry

end contrapositive_of_equality_square_l606_606829


namespace repeating_decimal_sum_l606_606081

theorem repeating_decimal_sum : (0.\overline{6} + 0.\overline{2} - 0.\overline{4} : ℚ) = (4 / 9 : ℚ) :=
by
  sorry

end repeating_decimal_sum_l606_606081


namespace complex_solution_l606_606199

theorem complex_solution (z : ℂ) (h : ((0 : ℝ) + 1 * z = 4 + 3 * (complex.I))) : 
  z = 3 - 4 * (complex.I) :=
sorry

end complex_solution_l606_606199


namespace problem_eq_l606_606144

theorem problem_eq : 
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → y = x / (x + 1) → (x - y + 4 * x * y) / (x * y) = 5 :=
by
  intros x y hx hnz hyxy
  sorry

end problem_eq_l606_606144


namespace solve_for_z_l606_606242

variable {z : ℂ}

theorem solve_for_z (h : complex.I * z = 4 + 3*complex.I) : z = 3 - 4*complex.I :=
sorry

end solve_for_z_l606_606242


namespace simplest_square_root_l606_606947

theorem simplest_square_root : 
  let a1 := Real.sqrt 20
  let a2 := Real.sqrt 2
  let a3 := Real.sqrt (1 / 2)
  let a4 := Real.sqrt 0.2
  a2 = Real.sqrt 2 ∧
  (a1 ≠ Real.sqrt 2 ∧ a3 ≠ Real.sqrt 2 ∧ a4 ≠ Real.sqrt 2) :=
by {
  -- Here, we fill in the necessary proof steps, but it's omitted for now.
  sorry
}

end simplest_square_root_l606_606947


namespace kolya_number_less_than_100_diff_polycarp_and_kolya_number_l606_606707

def is_unique_digits (n : ℕ) : Prop :=
  let digits := [2, 0, 4, 6, 8]
  List.nodup digits ∧ List.sorted digits

def smallest_5_digit_number_with_unique_even_digits : ℕ :=
  20468

def kolya_number_permutation (n : ℕ) (k : ℕ) : Prop :=
  let polycarps_number := 20468
  List.Perm (nat.digits 10 n) (nat.digits 10 k)

theorem kolya_number_less_than_100_diff (k : ℕ) : Prop :=
  abs (k - 20468) < 100

theorem polycarp_and_kolya_number :
  ∃ k, is_unique_digits 20468 ∧ 
        smallest_5_digit_number_with_unique_even_digits = 20468 ∧ 
        kolya_number_permutation 20468 k ∧
        kolya_number_less_than_100_diff k :=
sorry

end kolya_number_less_than_100_diff_polycarp_and_kolya_number_l606_606707


namespace three_times_x_not_much_different_from_two_l606_606636

theorem three_times_x_not_much_different_from_two (x : ℝ) :
  3 * x - 2 ≤ -1 := 
sorry

end three_times_x_not_much_different_from_two_l606_606636


namespace num_valid_sequences_l606_606368

def isValidSeq (seq : List ℕ) : Prop :=
  seq.length = 100 ∧
  (∀ i < seq.length - 1, abs (seq.get (i + 1) - seq.get i) ≤ 2) ∧
  (4 ∈ seq ∨ 5 ∈ seq)

theorem num_valid_sequences : 
  ∃ (s : Finset (List ℕ)), (∀ seq ∈ s, isValidSeq seq) ∧ s.card = 5^100 - 3^100 :=
sorry

end num_valid_sequences_l606_606368


namespace angle_sum_bounds_l606_606125

theorem angle_sum_bounds (α β γ : ℝ) 
  (hα : 0 < α ∧ α < π / 4)
  (hβ : 0 < β ∧ β < π / 4)
  (hγ : 0 < γ ∧ γ < π / 4)
  (h : sin(α)^2 + sin(β)^2 + sin(γ)^2 = 1) :
  π / 2 < α + β + γ ∧ α + β + γ ≤ 3 * arcsin (sqrt 3 / 3) :=
sorry

end angle_sum_bounds_l606_606125


namespace sum_of_leading_digits_l606_606311

def M : ℕ := 8 * 10 ^ 499 + 8 * 10 ^ 498 + ... + 8 * 10 ^ 0 -- or use a sum to represent this explicitly (details omitted here)

def leading_digit_of_root (n k : ℕ) : ℕ :=
  let root := real.cbrt (M ^ (1 / (k : ℝ))) in -- Place holder for actual computation
  root.to_digits.head -- Assumption: existing functions to extract leading digit

def g (k : ℕ) : ℕ :=
  leading_digit_of_root M k

theorem sum_of_leading_digits :
  g 2 + g 3 + g 4 + g 5 + g 6 + g 7 = 11 :=
by
  sorry

end sum_of_leading_digits_l606_606311


namespace line_BC_eq_circumscribed_circle_eq_l606_606677

noncomputable def A : ℝ × ℝ := (3, 0)
noncomputable def B : ℝ × ℝ := (0, -1)
noncomputable def altitude_line (x y : ℝ) : Prop := x + y + 1 = 0
noncomputable def equation_line_BC (x y : ℝ) : Prop := 3 * x - y - 1 = 0
noncomputable def circumscribed_circle (x y : ℝ) : Prop := (x - 5 / 2)^2 + (y + 7 / 2)^2 = 50 / 4

theorem line_BC_eq :
  ∃ x y : ℝ, altitude_line x y →
             B = (x, y) →
             equation_line_BC x y :=
by sorry

theorem circumscribed_circle_eq :
  ∃ x y : ℝ, altitude_line x y →
             (x - 3)^2 + y^2 = (5 / 2)^2 →
             circumscribed_circle x y :=
by sorry

end line_BC_eq_circumscribed_circle_eq_l606_606677


namespace perimeter_8_triangles_count_l606_606182

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def distinct_triangles_count (n : ℕ) : ℕ := 
  (Finset.unorderedPairs (Finset.range (n - 2 + 1))).filter (λ ⟨a, bc⟩, ∃ b c, bc = b + c ∧ is_triangle a b c ∧ a + b + c = n).card

theorem perimeter_8_triangles_count : distinct_triangles_count 8 = 2 := 
by
  sorry

end perimeter_8_triangles_count_l606_606182


namespace solve_sin_x_l606_606138

theorem solve_sin_x (x : ℝ) (h : real.sec x + real.tan x = 5 / 2) : real.sin x = 21 / 29 :=
sorry

end solve_sin_x_l606_606138


namespace find_a6_l606_606621

-- Define the sequence recursively
def a : ℕ → ℚ
| 1       := 4
| 2       := 8/3
| (n+3) := (a (n+2) * a (n+1)) / (3 * a (n+1) - 2 * a (n+2))

-- Define the goal to prove
theorem find_a6 : a 6 = 4/9 ∧ (4 + 9) = 13 :=
by {
  -- Now proceed to prove this theorem
  sorry
}

end find_a6_l606_606621


namespace sum_x_coords_above_line_zero_l606_606975

def lies_above_line (p : ℝ × ℝ) : Prop :=
  p.2 > 3 * p.1 + 4

def points : List (ℝ × ℝ) := [(2, 9), (8, 25), (10, 30), (15, 45), (25, 60)]

/-- The sum of the $x$-coordinates of points among (2, 9), (8, 25), (10, 30), (15, 45), and (25, 60) 
    that are located above the line $y = 3x + 4$ in the coordinate plane is 0. -/
theorem sum_x_coords_above_line_zero :
  (points.filter lies_above_line).sum (λ p, p.1) = 0 := 
sorry

end sum_x_coords_above_line_zero_l606_606975


namespace determine_b_c_d_l606_606963

noncomputable def a_n (n : ℕ) : ℕ :=
let seq := (λ k, if (k % 2 = 1) then mk_seq k (k+2) else 0) in 
seq n

axiom mk_seq : ℕ → ℕ → ℕ

theorem determine_b_c_d:
  (∀ n : ℕ, ∃ b c d, a_n n = b * (Int.floor (Real.sqrt (n + c)))^2 + d) ∧ 
  (b = 1) ∧ (c = -1) ∧ (d = 1) :=
sorry

end determine_b_c_d_l606_606963


namespace charity_total_cost_l606_606567

theorem charity_total_cost
  (plates : ℕ)
  (rice_cost_per_plate chicken_cost_per_plate : ℕ)
  (h1 : plates = 100)
  (h2 : rice_cost_per_plate = 10)
  (h3 : chicken_cost_per_plate = 40) :
  plates * (rice_cost_per_plate + chicken_cost_per_plate) / 100 = 50 := 
by
  sorry

end charity_total_cost_l606_606567


namespace min_magnitude_perpendicular_l606_606137

open Real

variables {a b : ℝ^n} [non_zero_vector_a : a ≠ 0] [non_zero_vector_b : b ≠ 0] {λ : ℝ}

theorem min_magnitude_perpendicular (a b : ℝ^n) (non_zero_a : a ≠ 0) (non_zero_b : b ≠ 0) :
  ∃ λ : ℝ, (b ⬝ (a + λ • b) = 0) → ∀ μ : ℝ, ∥a + λ • b∥ ≤ ∥a + μ • b∥ := by
  sorry

end min_magnitude_perpendicular_l606_606137


namespace train_speed_l606_606888

theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) 
  (length_eq : length = 480) 
  (time_eq : time = 16) 
  (speed_eq : speed = 30) : 
  speed = length / time := 
by 
  rw [length_eq, time_eq, speed_eq]
  norm_num
  sorry

end train_speed_l606_606888


namespace modulus_first_expression_correct_modulus_second_expression_correct_l606_606984

noncomputable def modulus_first_expression (z : ℂ) : ℝ :=
  complex.abs (z^2 + 4 * z + 40)

theorem modulus_first_expression_correct (z : ℂ) (hz : z = 5 + 3 * complex.I) :
  modulus_first_expression z = real.sqrt 7540 := by
  rw [hz]
  sorry

noncomputable def modulus_second_expression (z : ℂ) : ℝ :=
  complex.abs (2 * z^2 + 5 * z + 3)

theorem modulus_second_expression_correct (z : ℂ) (hz : z = 5 + 3 * complex.I) :
  modulus_second_expression z = real.sqrt 9225 := by
  rw [hz]
  sorry

end modulus_first_expression_correct_modulus_second_expression_correct_l606_606984


namespace relationship_between_y_and_x_selling_price_for_profit_8000_l606_606578

noncomputable def sales_volume (x : ℝ) := -10 * x + 800

theorem relationship_between_y_and_x :
  (∀ x, sales_volume x = -10 * x + 800) ↔ 
  (sales_volume 30 = 500 ∧ sales_volume 40 = 400) :=
begin
  split,
  { intros h, 
    split,
    { exact h 30 }, 
    { exact h 40 } },
  { intros h x,
    cases h with h1 h2,
    have k_eq : -10 = -10 := rfl,
    have b_eq : 800 = 800 := rfl,
    exact rfl }
end 

theorem selling_price_for_profit_8000 :
  ∃ x : ℝ, x ≤ 45 ∧ (x - 20) * (-10 * x + 800) = 8000 :=
begin
  use 40,
  split,
  { linarith },
  { norm_num },
  sorry 
end

end relationship_between_y_and_x_selling_price_for_profit_8000_l606_606578


namespace range_of_a_l606_606853

-- Define the function f
def f (a x : ℝ) : ℝ := a * x ^ 2 + a * x - 1

-- State the problem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x < 0) ↔ -4 < a ∧ a ≤ 0 :=
by {
  sorry
}

end range_of_a_l606_606853


namespace find_k_value_l606_606109

variable (x y z k : ℝ)

theorem find_k_value (h : 7 / (x + y) = k / (x + z) ∧ k / (x + z) = 11 / (z - y)) :
  k = 18 :=
sorry

end find_k_value_l606_606109


namespace colored_chessboard_limit_l606_606672

theorem colored_chessboard_limit (l : ℕ → ℕ) (h : ∀ n, ∀ k, n ≥ 1 → k ≥ 1 → k ≤ n → 
  (∃ (vertices_colored: ℕ → ℕ → Prop), 
  (∀ i j, i ≤ n ∧ j ≤ n ∧ vertices_colored i j -> ∃ i' j', i' ≤ k ∧ j' ≤ k ∧ 
  (vertices_colored i' j' ∧ (i' = i + 1 ∨ j' = j + 1))))) :
  (lim (λ n, (l n : ℝ)/ (n^2 : ℝ)) = (2/7)) := 
begin
  sorry
end

end colored_chessboard_limit_l606_606672


namespace meet_at_start_l606_606886

-- Definitions to capture the initial conditions
def trackLength : ℕ := 200
def speedA_kmph : ℕ := 36
def speedB_kmph : ℕ := 72

-- Convert speeds from kmph to m/s
def speedA_mps : ℚ := (speedA_kmph * 1000) / 3600
def speedB_mps : ℚ := (speedB_kmph * 1000) / 3600

-- Calculate the time to complete one lap for A and B
def timeA : ℚ := trackLength / speedA_mps
def timeB : ℚ := trackLength / speedB_mps

-- The least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- The theorem stating that they meet after 20 seconds
theorem meet_at_start : lcm timeA.natAbs timeB.natAbs = 20 := 
by
  sorry

end meet_at_start_l606_606886


namespace percentage_of_error_in_area_l606_606008

-- Define the conditions
variable (x : ℝ)
def measured_side := 1.12 * x
def actual_area := x^2
def erroneous_area := (measured_side x)^2
def error_area := erroneous_area x - actual_area x
def percentage_error := (error_area x / actual_area x) * 100

-- The theorem to be proven
theorem percentage_of_error_in_area : percentage_error x = 25.44 := by
  sorry

end percentage_of_error_in_area_l606_606008


namespace find_z_l606_606249

theorem find_z (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : z = 3 - 4 * complex.I :=
by {
  sorry
}

end find_z_l606_606249


namespace inequality_proof_l606_606534

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606534


namespace least_integer_greater_than_sqrt_450_l606_606482

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, (n > real.sqrt 450) ∧ ∀ m : ℤ, (m > real.sqrt  450) → n ≤ m :=
by
  apply exists.intro 22
  sorry

end least_integer_greater_than_sqrt_450_l606_606482


namespace quadrilateral_pyramid_proof_l606_606752

noncomputable def quadrilateral_pyramid 
    (A B C D E : Type) 
    [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] 
    (plane_ACE: Type) 
    [metric_space plane_ACE]
    (angle_A_ACE : real) (angle_B_ACE : real) (angle_C_ACE : real) (angle_D_ACE : real)
    (angle_ABCD_ACE : real) (angle_ABE_ACE : real) (angle_BCE_ACE : real) 
    (angle_CDE_ACE : real) (angle_DAE_ACE : real) :=
    (angle_ABCD_ACE = 45) ∧
    (angle_ABE_ACE = 45) ∧
    (angle_BCE_ACE = 45) ∧
    (angle_CDE_ACE = 45) ∧
    (angle_DAE_ACE = 45)

theorem quadrilateral_pyramid_proof 
    (A B C D E : Type) 
    [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
    (plane_ACE: Type) 
    [metric_space plane_ACE]
    (angle_A_ACE : real) (angle_B_ACE : real) (angle_C_ACE : real) (angle_D_ACE : real)
    (angle_ABCD_ACE : real) (angle_ABE_ACE : real) (angle_BCE_ACE : real)
    (angle_CDE_ACE : real) (angle_DAE_ACE : real)
    (h : quadrilateral_pyramid A B C D E plane_ACE angle_A_ACE angle_B_ACE 
        angle_C_ACE angle_D_ACE angle_ABCD_ACE angle_ABE_ACE angle_BCE_ACE 
        angle_CDE_ACE angle_DAE_ACE) :
    (dist A B)^2 + (dist A D)^2 = (dist B C)^2 + (dist C D)^2 :=
by
  sorry

end quadrilateral_pyramid_proof_l606_606752


namespace sequence_le_zero_l606_606328

noncomputable def sequence_property (N : ℕ) (a : ℕ → ℝ) : Prop :=
  (a 0 = 0) ∧ (a N = 0) ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ N - 1 → a (i + 1) - 2 * a i + a (i - 1) = a i ^ 2)

theorem sequence_le_zero {N : ℕ} (a : ℕ → ℝ) (h : sequence_property N a) : 
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ N - 1 → a i ≤ 0 :=
sorry

end sequence_le_zero_l606_606328


namespace num_pairs_45_degree_angle_l606_606908

theorem num_pairs_45_degree_angle : 
  let block := {v : ℕ × ℕ | v.1 < 3 ∧ v.2 < 3} in
  let pairs := {(a, b) : (ℕ × ℕ) × (ℕ × ℕ) | a ∈ block ∧ b ∈ block ∧ a ≠ b ∧ (a.1 - b.1).natAbs = (a.2 - b.2).natAbs} in
  pairs.to_finset.card = 18 :=
by
  sorry

end num_pairs_45_degree_angle_l606_606908


namespace midpoint_constant_of_moving_c_l606_606335

theorem midpoint_constant_of_moving_c (a b : ℂ) :
  ∀ (c : ℂ), midpoint (a + (c - a) * complex.I) (b + (b - c) * complex.I) = (a * (1 - complex.I) + b * (1 + complex.I)) / 2 := 
by
  intro c
  sorry

end midpoint_constant_of_moving_c_l606_606335


namespace total_students_in_class_l606_606384

theorem total_students_in_class 
    (avg_age_n_minus_1 : ℕ → ℝ)
    (avg_age_4 : ℝ)
    (avg_age_9 : ℝ)
    (age_15th_student : ℝ) :
    avg_age_n_minus_1 14 = 15 →
    avg_age_4 = 14 →
    avg_age_9 = 16 →
    age_15th_student = 25 →
    let N := 15 in N = 15 :=
by 
    intros h1 h2 h3 h4
    let total_age_n_minus_1 := 15 * 14
    let total_age_4 := 14 * 4
    let total_age_9 := 16 * 9
    have total_age_N_1_equation : total_age_n_minus_1 + 25 = total_age_4 + total_age_9 + 25,
    {
        rw [h1, h2, h3, h4],
        linarith,
    }
    sorry

end total_students_in_class_l606_606384


namespace intersection_distance_l606_606741

noncomputable def line_l_eqs (t : ℝ) : ℝ × ℝ :=
  let x := (1/2) * t
  let y := (sqrt 2 / 2) + (sqrt 3 / 2) * t
  (x, y)

def curve_C_eq (θ : ℝ) : ℝ :=
  2 * cos (θ - π / 4)

theorem intersection_distance :
  (∀ t θ, let (x, y) := line_l_eqs t in 
          (∀ ρ, ρ = curve_C_eq θ → x = ρ * cos θ ∧ y = ρ * sin θ)
          → |sqrt 10 / 2| = |sqrt 10 / 2| := λ t θ (h : curve_C_eq θ = ρ) =>
  sorry

end intersection_distance_l606_606741


namespace cleaning_times_l606_606301

theorem cleaning_times (A B C : ℕ) (hA : A = 40) (hB : B = A / 4) (hC : C = 2 * B) : 
  B = 10 ∧ C = 20 := by
  sorry

end cleaning_times_l606_606301


namespace arithmetic_sequence_sum_n_squared_l606_606661

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_mean (x y z : ℝ) : Prop :=
(y * y = x * z)

def is_strictly_increasing (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) > a n

theorem arithmetic_sequence_sum_n_squared
  (a : ℕ → ℝ)
  (h₁ : is_arithmetic_sequence a)
  (h₂ : a 1 = 1)
  (h₃ : is_geometric_mean (a 1) (a 2) (a 5))
  (h₄ : is_strictly_increasing a) :
  ∃ S : ℕ → ℝ, ∀ n : ℕ, S n = n ^ 2 :=
sorry

end arithmetic_sequence_sum_n_squared_l606_606661


namespace first_pack_weight_l606_606585

-- Define the conditions
def miles_per_hour := 2.5
def hours_per_day := 8
def days := 5
def supply_per_mile := 0.5
def resupply_percentage := 0.25
def total_hiking_time := hours_per_day * days
def total_miles_hiked := total_hiking_time * miles_per_hour
def total_supplies_needed := total_miles_hiked * supply_per_mile
def resupply_factor := 1 + resupply_percentage

-- Define the theorem
theorem first_pack_weight :
  (total_supplies_needed / resupply_factor) = 40 :=
by
  sorry

end first_pack_weight_l606_606585


namespace isosceles_triangle_l606_606351

-- Definitions
variables {A B C M K A1 B1 : Type*}
variables (point : A) (point B) (point C) (point M) (point K) (point A1) (point B1)
variable (triangle : A → B → C → Type*)
variable (median : triangle.point → point → Type*)

-- Conditions
variable (is_median_CM : median C M)
variable (is_on_CM : point K ∈ median C M)
variable (intersects_AK_BC : point A1 = AK ∩ BC)
variable (intersects_BK_AC : point B1 = BK ∩ AC)
variable (inscribed_AB1A1B : circle AB1A1B)

-- Proof
theorem isosceles_triangle (h1 : is_median_CM) (h2 : is_on_CM)
  (h3 : intersects_AK_BC) (h4 : intersects_BK_AC)
  (h5 : inscribed_AB1A1B) :
  AB = AC :=
by
  sorry

end isosceles_triangle_l606_606351


namespace train_crossing_time_l606_606938

theorem train_crossing_time
    (length_of_train : ℕ)
    (speed_of_train_kmph : ℕ)
    (length_of_bridge : ℕ)
    (h_train_length : length_of_train = 160)
    (h_speed_kmph : speed_of_train_kmph = 45)
    (h_bridge_length : length_of_bridge = 215)
  : length_of_train + length_of_bridge / ((speed_of_train_kmph * 1000) / 3600) = 30 :=
by
  rw [h_train_length, h_speed_kmph, h_bridge_length]
  norm_num
  sorry

end train_crossing_time_l606_606938


namespace least_integer_greater_than_sqrt_450_l606_606490

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, n > real.sqrt 450 ∧ ∀ m : ℕ, m > real.sqrt 450 → n ≤ m :=
begin
  use 22,
  split,
  { sorry },
  { sorry }
end

end least_integer_greater_than_sqrt_450_l606_606490


namespace trig_solution_l606_606840

open Real

noncomputable def solve_trig_eq (x y z : ℝ) (n k m : ℤ) : Prop :=
  (sin x ≠ 0) ∧ (sin y ≠ 0) ∧ 
  ((sin^2 x + (1/(sin^2 x)))^3 + (sin^2 y + (1/(sin^2 y)))^3 = 16 * cos z) →
  (∃ n k m : ℤ, x = π/2 + π * n ∧ y = π/2 + π * k ∧ z = 2 * π * m)

theorem trig_solution (x y z : ℝ) (n k m : ℤ) :
  solve_trig_eq x y z n k m :=
sorry

end trig_solution_l606_606840


namespace numbers_not_divisible_by_4_or_6_l606_606944

theorem numbers_not_divisible_by_4_or_6 (n : ℕ) (h : n = 1000) : 
  ∃ k : ℕ, k = 667 ∧ (∀ m : ℕ, 1 ≤ m ∧ m ≤ n → (m % 4 ≠ 0 ∧ m % 6 ≠ 0) ↔ m ∉ (filter (λ x, x % 4 = 0 ∨ x % 6 = 0) (range (n + 1)))).length = k := 
by
  sorry

end numbers_not_divisible_by_4_or_6_l606_606944


namespace ball_bounce_height_l606_606909

theorem ball_bounce_height :
  ∃ k : ℕ, 2000 * (2 / 3 : ℝ) ^ k < 2 ∧ ∀ j : ℕ, j < k → 2000 * (2 / 3 : ℝ) ^ j ≥ 2 :=
by {
  sorry
}

end ball_bounce_height_l606_606909


namespace range_of_a_l606_606253

theorem range_of_a (x y a : ℝ) (h1 : x < y) (h2 : (a - 3) * x > (a - 3) * y) : a < 3 :=
sorry

end range_of_a_l606_606253


namespace find_KH_l606_606818

variable {Triangle : Type}
variables {K P H M : Triangle} {a b : ℝ}

-- Assume conditions
axiom midpoint_K : midpoint K
axiom foot_P : angle_bisector P
axiom tangency_H : incircle_tangency H
axiom altitude_M : altitude_foot M
axiom KP_eq_a : distance K P = a
axiom KM_eq_b : distance K M = b

theorem find_KH (midpoint_K : midpoint K) (foot_P : angle_bisector P) (tangency_H : incircle_tangency H)
  (altitude_M : altitude_foot M) (KP_eq_a : distance K P = a) (KM_eq_b : distance K M = b) : 
  distance K H = sqrt (a * b) :=
by
  sorry

end find_KH_l606_606818


namespace line_through_points_l606_606720

theorem line_through_points (a b : ℝ)
  (h1 : 2 = a * 1 + b)
  (h2 : 14 = a * 5 + b) :
  a - b = 4 := 
  sorry

end line_through_points_l606_606720


namespace person_speed_l606_606924

theorem person_speed (distance_m : ℝ) (time_min : ℝ)
  (h₀ : distance_m = 1080) (h₁ : time_min = 14) : 
  (distance_m / 1000) / (time_min / 60) ≈ 4.63 :=
by
  sorry

end person_speed_l606_606924


namespace line_passes_through_fixed_point_l606_606710

theorem line_passes_through_fixed_point (k b : ℝ) (h : k + b = -2) :
    ∃ (x y : ℝ), y = k * x + b ∧ x = 1 ∧ y = -2 :=
    by
    use 1
    use -2
    simp [h, add_eq_zero_iff_eq_neg]
    sorry

end line_passes_through_fixed_point_l606_606710


namespace repeating_decimals_sum_l606_606095

theorem repeating_decimals_sum : (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ) = 4 / 9 :=
 by 
  have h₁ : (0.666666... : ℚ) = 2 / 3,
    -- Since x = 0.6666..., then 10x = 6.6666...,
    -- so 10x - x = 6, then x = 6 / 9, hence 2 / 3
    sorry,
  have h₂ : (0.222222... : ℚ) = 2 / 9,
    -- Since x = 0.2222..., then 10x = 2.2222...,
    -- so 10x - x = 2, then x = 2 / 9
    sorry,
  have h₃ : (0.444444... : ℚ) = 4 / 9,
    -- Since x = 0.4444..., then 10x = 4.4444...,
    -- so 10x - x = 4, then x = 4 / 9
    sorry,
  calc
    (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ)
        = (2 / 3) + (2 / 9) - (4 / 9) : by rw [h₁, h₂, h₃]
    ... = (6 / 9) + (2 / 9) - (4 / 9) : by norm_num
    ... = 4 / 9 : by ring

end repeating_decimals_sum_l606_606095


namespace recurring_decimal_sum_l606_606079

theorem recurring_decimal_sum :
  (0.666666...:ℚ) + (0.222222...:ℚ) - (0.444444...:ℚ) = 4 / 9 :=
begin
  sorry
end

end recurring_decimal_sum_l606_606079


namespace articles_profit_l606_606386

variable {C S : ℝ}

theorem articles_profit (h1 : 20 * C = x * S) (h2 : S = 1.25 * C) : x = 16 :=
by
  sorry

end articles_profit_l606_606386


namespace shaded_area_equals_l606_606292

noncomputable def shaded_area (AB BC CD DE EF FG : ℝ) : ℝ :=
  let small_semicircle_area (d : ℝ) := (π * (d / 2)^2) / 2
  let large_semicircle_area (d : ℝ) := (π * (d / 2)^2) / 2
  small_semicircle_area AB + small_semicircle_area BC + small_semicircle_area CD +
  small_semicircle_area DE + small_semicircle_area EF + small_semicircle_area FG

theorem shaded_area_equals (h : AB = 4) (h1 : BC = 4) (h2 : CD = 4) (h3 : DE = 4) (h4 : EF = 4) (h5 : FG = 4) :
  shaded_area AB BC CD DE EF FG = 72 * π := by
  sorry

end shaded_area_equals_l606_606292


namespace largest_solution_validate_largest_solution_l606_606108

open Real

theorem largest_solution (x : ℝ) (hx : floor(x) = 8 + 50 * fract(x)) : x ≤ 57.98 :=
by
  -- Proof steps here
  sorry

theorem validate_largest_solution : ∃ x, floor(x) = 8 + 50 * fract(x) ∧ x = 57.98 :=
by 
  -- Proof steps here
  sorry

end largest_solution_validate_largest_solution_l606_606108


namespace least_integer_gt_sqrt_450_l606_606460

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l606_606460


namespace find_negative_integer_l606_606951

theorem find_negative_integer (numbers : set ℝ) (neg_int : ℤ) (h1 : numbers = {-2.4, 0, -2, 2})
  (h2 : neg_int = -2) : 
  neg_int ∈ numbers ∧ neg_int < 0 :=
by 
  sorry

end find_negative_integer_l606_606951


namespace repeating_decimals_sum_l606_606072

theorem repeating_decimals_sum :
  let x := 0.6666666 -- 0.\overline{6}
  let y := 0.2222222 -- 0.\overline{2}
  let z := 0.4444444 -- 0.\overline{4}
  (x + y - z) = 4 / 9 := 
by
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  calc
    -- Calculate (x + y - z)
    (x + y - z) = (2 / 3 + 2 / 9 - 4 / 9) : by sorry
                ... = 4 / 9 : by sorry


end repeating_decimals_sum_l606_606072


namespace regular_price_of_ticket_l606_606343

theorem regular_price_of_ticket (P : Real) (discount_paid : Real) (discount_rate : Real) (paid : Real)
  (h_discount_rate : discount_rate = 0.40)
  (h_paid : paid = 9)
  (h_discount_paid : discount_paid = P * (1 - discount_rate))
  (h_paid_eq_discount_paid : paid = discount_paid) :
  P = 15 := 
by
  sorry

end regular_price_of_ticket_l606_606343


namespace num_mappings_satisfying_condition_l606_606118

open Finset

def A := {0, 1}
def B := {-1, 0, 1}

def mappings_satisfying_condition := 
  {f : A → B | f 0 > f 1}

#eval mappings_satisfying_condition.card  -- This should return 3 in the proof

theorem num_mappings_satisfying_condition : 
  mappings_satisfying_condition.card = 3 := 
by sorry

end num_mappings_satisfying_condition_l606_606118


namespace factor_expression_l606_606100

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) :=
by
  sorry

end factor_expression_l606_606100


namespace ratio_F₁F₂_V₁V₂_l606_606315

noncomputable def V₁ : (ℝ × ℝ) := (0, 0)

noncomputable def F₁ : (ℝ × ℝ) := (0, 1 / 16)

-- Parabola Q represented as y = 8x^2 + 1/2
noncomputable def V₂ : (ℝ × ℝ) := (0, 1 / 2)

noncomputable def F₂ : (ℝ × ℝ) := (0, 17 / 32)

def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ratio_F₁F₂_V₁V₂ : 
  (distance F₁ F₂) / (distance V₁ V₂) = 15 / 16 := 
by
  sorry

end ratio_F₁F₂_V₁V₂_l606_606315


namespace points_concyclic_l606_606732

variable {Point : Type}
variable {ABCD : Type} [cyclic_quadrilateral ABCD]
variable {O K L M P Q : Point}
variable {S1 S2 : circle}

-- Definitions from conditions
def diagonals_intersect_at (O : Point) (ABCD : Type) : Prop := 
  exists A B C D : Point, diagonals A B C D meet at O

def circumcircle_of_ABO (S1 : circle) (A B O : Point) : Prop := 
  circumcircle A B O = S1

def circumcircle_of_CDO (S2 : circle) (C D O : Point) : Prop := 
  circumcircle C D O = S2

def intersect_at_O_K (S1 S2 : circle) (O K : Point) : Prop := 
  intersect S1 S2 = {O, K}

def line_through_parallel_to (O : Point) (A B : Point) (L M : Point)
  (c1 c2 : circle) : Prop := (line_through O L) ∥ (line_through A B) ∧ 
                             on_circle L c1 ∧ (line_through O M) ∥ (line_through A B) ∧ 
                             on_circle M c2

def ratio_OP_PL_MQ_QO (O P L M Q : Point) : Prop := 
  OP / PL = MQ / QO

-- The main theorem statement
theorem points_concyclic 
  (ABCD : Type) [cyclic_quadrilateral ABCD] (O K : Point)
  (S1 S2 : circle) (L M P Q : Point)
  (h1 : diagonals_intersect_at O ABCD)
  (h2 : circumcircle_of_ABO S1 A B O)
  (h3 : circumcircle_of_CDO S2 C D O)
  (h4 : intersect_at_O_K S1 S2 O K)
  (h5 : line_through_parallel_to O A B L M S1 S2)
  (h6 : ratio_OP_PL_MQ_QO O P L M Q) : 
  concyclic {O, K, P, Q} := 
sorry

end points_concyclic_l606_606732


namespace shaded_area_proof_l606_606814

-- Define the conditions: right isosceles triangle, hypotenuse 10 cm, and associated circle
def right_isosceles_triangle (h : ℝ) : Prop :=
  h = 10

def circle_radius (r : ℝ) : Prop :=
  r = 5

def sector_area (a : ℝ) : Prop :=
  a = (25 * Real.pi) / 8

def triangle_area (t : ℝ) : Prop :=
  t = 25 / 2

def shaded_area_expression (a b c : ℝ) : Prop :=
  a = 25 ∧ b = 50 ∧ c = 1 ∧ (a + b + c = 76)

-- The theorem stating the problem
theorem shaded_area_proof :
  ∃ (a b c : ℝ), right_isosceles_triangle 10 ∧ circle_radius 5 ∧ sector_area ((25 * Real.pi) / 8) ∧ triangle_area (25 / 2) ∧ shaded_area_expression a b c :=
by
  apply Exists.intro 25
  apply Exists.intro 50
  apply Exists.intro 1
  split
  { -- Right isosceles triangle condition
    exact rfl }
  split
  { -- Circle radius condition
    exact rfl }
  split
  { -- Sector area condition
    exact rfl }
  split
  { -- Triangle area condition
    exact rfl }
  { -- Shaded area expression condition
    simp [shaded_area_expression, right_isosceles_triangle, circle_radius, sector_area, triangle_area]
    sorry
  }

end shaded_area_proof_l606_606814


namespace min_trips_l606_606510

def masses : List ℕ := [130, 60, 61, 65, 68, 70, 79, 81, 83, 87, 90, 91, 95]

def capacity : ℕ := 175

theorem min_trips (h : (∀ x ∈ masses, x ≤ 130 ∧ x ≥ 60) ∧ (capacity = 175)) :
  ∃ n, minNumTrips masses capacity = 7 :=
by
  sorry

end min_trips_l606_606510


namespace new_bag_marbles_l606_606059

open Nat

theorem new_bag_marbles 
  (start_marbles : ℕ)
  (lost_marbles : ℕ)
  (given_marbles : ℕ)
  (received_back_marbles : ℕ)
  (end_marbles : ℕ)
  (h_start : start_marbles = 40)
  (h_lost : lost_marbles = 3)
  (h_given : given_marbles = 5)
  (h_received_back : received_back_marbles = 2 * given_marbles)
  (h_end : end_marbles = 54) :
  (end_marbles = (start_marbles - lost_marbles - given_marbles + received_back_marbles + new_bag) ∧ new_bag = 12) :=
by
  sorry

end new_bag_marbles_l606_606059


namespace percentage_of_hindu_boys_l606_606277

theorem percentage_of_hindu_boys (total_boys : ℕ) (muslim_percentage : ℝ) (sikh_percentage : ℝ) (other_community_boys : ℕ) :
  total_boys = 300 → muslim_percentage = 0.44 → sikh_percentage = 0.1 → other_community_boys = 54 →
  ((total_boys - (muslim_percentage * total_boys).to_nat - (sikh_percentage * total_boys).to_nat - other_community_boys) / total_boys.to_rat) * 100 = 28 :=
by
  intros h1 h2 h3 h4
  sorry

end percentage_of_hindu_boys_l606_606277


namespace largest_prime_divisor_S_l606_606982

-- Define the product of non-zero digits function
def p (n : ℕ) : ℕ :=
  (n.digits 10).filter (≠ 0).prod

-- Define S as the sum of p_i for i from 1 to 999
def S : ℕ :=
  (list.range 1 1000).map p).sum

-- Statement of the problem
theorem largest_prime_divisor_S : ∃ (p : ℕ), p.prime ∧ p.factors.max' = 103 :=
sorry

end largest_prime_divisor_S_l606_606982


namespace area_transformed_graph_l606_606401

noncomputable def f : ℝ → ℝ := sorry
axiom area_f : ∫ x in -∞..∞, f x = 12

theorem area_transformed_graph : ∫ x in -∞..∞, 4 * f (x + 3) = 48 :=
by
  -- Proof would go here
  sorry

end area_transformed_graph_l606_606401


namespace minimal_red_cubes_l606_606595

-- Define a positive integer n
variables (n : ℕ) (n_pos : 0 < n)

-- Define the problem of counting the red cubes needed
theorem minimal_red_cubes (n_pos : 0 < n) :
  ∃ (k : ℕ), k = (n + 1) ^ 3 ∧
  (∀ i j k, 0 ≤ i < 3 * n → 0 ≤ j < 3 * n → 0 ≤ k < 3 * n → (∃ a b c, abs (i - a) ≤ 1 ∧ abs (j - b) ≤ 1 ∧ abs (k - c) ≤ 1 ∧ (a % 3 == 0) ∧ (b % 3 == 0) ∧ (c % 3 == 0))) :=
  sorry

end minimal_red_cubes_l606_606595


namespace repeating_decimal_sum_l606_606619
open Nat

/-- Convert the repeating decimal 0.\overline{27} into its simplest fractional form.
Then, prove that the sum of the numerator and denominator of this fraction is 14. -/
theorem repeating_decimal_sum :
  let x := 0.2727272727 -- repeating part
  let frac := 27 / 99   -- conversion and simplification steps
  let gcd := gcd 27 99
  let simplified := (27 / gcd, 99 / gcd)
  let numerator := fst simplified
  let denominator := snd simplified
  numerator + denominator = 14 :=
by {
  sorry
}

end repeating_decimal_sum_l606_606619


namespace inequality_proof_l606_606517

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l606_606517


namespace count_3digit_numbers_divisible_by_11_l606_606187

theorem count_3digit_numbers_divisible_by_11 :
  let smallest := 100 in
  let largest := 999 in
  -- Determine the smallest and largest 3-digit numbers divisible by 11
  let smallest_divisible_by_11 := ((smallest + 10) / 11) * 11 in
  let largest_divisible_by_11 := (largest / 11) * 11 in
  -- Number of positive 3-digit numbers divisible by 11
  (largest_divisible_by_11 / 11) - (smallest_divisible_by_11 / 11) + 1 = 81 :=
by
  sorry

end count_3digit_numbers_divisible_by_11_l606_606187


namespace hyperbola_perimeter_l606_606679

theorem hyperbola_perimeter
  (P F1 F2 : ℝ × ℝ)
  (hP_on_hyperbola : (P.1)^2 - (P.2)^2 = 1)
  (h_F1F2 : ∥F1 - F2∥ = 2 * Real.sqrt 2)
  (h_PF1_PF2_ratio : ∥P - F1∥ = 3 * ∥P - F2∥) :
  ∥P - F1∥ + ∥P - F2∥ + ∥F1 - F2∥ = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end hyperbola_perimeter_l606_606679


namespace part1_part2_l606_606797

-- Define the sequence and its sum
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ (a_1 d : ℝ), ∀ n : ℕ, a_n (n+1) = a_1 + n * d

def sequence_sum (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S_n (n+1) = (n+1) * a_n 0 + (n * (n+1) / 2) * (a_n 1 - a_n 0)

def b_n (S_n : ℕ → ℝ) (n : ℕ) : ℝ := S_n n / n

-- Problem statements
theorem part1 (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : arithmetic_sequence a_n)
  (h2 : sequence_sum S_n a_n)
  (h3 : ∀ n, b n = b_n S_n n) :
  ∃ a1 d, ∀ n, b (n+1) = a1 + n * (d / 2) := sorry

theorem part2 (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
  (h1 : arithmetic_sequence a_n)
  (h2 : sequence_sum S_n a_n)
  (h3 : S_n 7 = 7)
  (h4 : S_n 15 = 75) :
  ∃ a1 d, ∀ n, b_n S_n (n+1) = (n - 4) / 2 := sorry

end part1_part2_l606_606797


namespace color_of_85th_bead_l606_606303

def bead_pattern : List String := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "blue", "blue"]

def bead_color (n : ℕ) : String :=
  bead_pattern.get! (n % bead_pattern.length)

theorem color_of_85th_bead : bead_color 84 = "yellow" := 
by
  sorry

end color_of_85th_bead_l606_606303


namespace butter_needed_for_original_recipe_l606_606983

-- Define the conditions
def butter_to_flour_ratio : ℚ := 12 / 56

def flour_for_original_recipe : ℚ := 14

def butter_for_original_recipe (ratio : ℚ) (flour : ℚ) : ℚ :=
  ratio * flour

-- State the theorem
theorem butter_needed_for_original_recipe :
  butter_for_original_recipe butter_to_flour_ratio flour_for_original_recipe = 3 := 
sorry

end butter_needed_for_original_recipe_l606_606983


namespace product_of_all_real_values_s_l606_606649

noncomputable def product_of_solutions : ℝ :=
  let s_values := {s : ℝ | ∃ x : ℝ, x ≠ 0 ∧ (1 / (3 * x) = (s - x) / 10) ∧ (9 * s^2 - 120 = 0)} in
  ∏ s in s_values, s

theorem product_of_all_real_values_s : product_of_solutions = -40 / 3 := sorry

end product_of_all_real_values_s_l606_606649


namespace angle_equality_l606_606288

-- Setup the geometric environment
variables {A B C D E F G : Type} [point A] [point B] [point C] [point D] [point E] [point F] [point G]
variables (bad_angle_bisector : bisects_diagonal A C B D) -- AC bisects angle BAD
          (E_on_CD : on_line_segment E C D) -- E is on CD
          (F_intersection : intersection_point F B E A C) -- BE intersects AC at F
          (G_intersection : extended_intersection_point G D F C B) -- DF extended intersects BC at G

-- The theorem to prove
theorem angle_equality : ∠G A C = ∠E A C :=
by
  sorry

end angle_equality_l606_606288


namespace min_marked_squares_l606_606035

-- Define the problem with necessary conditions and the main theorem
variable (n : ℕ) (h_even: n % 2 = 0 ∧ n > 0)

-- Define the theorem statement with given conditions and the correct answer
theorem min_marked_squares (n : ℕ) (h_even: n % 2 = 0 ∧ n > 0) : 
  ∃ N, N = (n^2 + 2 * n) / 4 ∧ 
  ∀ i j, i < n ∧ j < n -> (∃ x y, x < n ∧ y < n ∧ abs (i - x) + abs (j - y) = 1 ∧ marked x y) :=
sorry

-- Definition for marked squares based on the problem statement.
definition marked : ℕ -> ℕ -> Prop := sorry

end min_marked_squares_l606_606035


namespace inequality_proof_l606_606542

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l606_606542


namespace initial_order_cogs_l606_606954

theorem initial_order_cogs (x : ℕ) (h : (x + 60 : ℚ) / (x / 36 + 1) = 45) : x = 60 := 
sorry

end initial_order_cogs_l606_606954


namespace sunscreen_discount_l606_606768

theorem sunscreen_discount
  (months : ℕ)
  (cost_per_bottle : ℕ)
  (discounted_total_cost : ℤ)
  (months = 12)
  (cost_per_bottle = 30)
  (discounted_total_cost = 252) :
  let original_total_cost := months * cost_per_bottle,
      discount_amount     := original_total_cost - discounted_total_cost,
      discount_percentage := (↑discount_amount / original_total_cost) * 100 in
  discount_percentage = 30 := 
sorry

end sunscreen_discount_l606_606768


namespace fir_tree_probability_l606_606579

-- Define the number of trees by type.
def pine_trees : ℕ := 5
def cedar_trees : ℕ := 6
def fir_trees : ℕ := 7

-- Define total non-fir trees.
def non_fir_trees : ℕ := pine_trees + cedar_trees

-- Define the number of slots available for placing fir trees.
def slots : ℕ := non_fir_trees + 1

-- Total number of arrangements of trees.
def total_arrangements : ℕ := (finset.card (finset.range (slots.choose fir_trees))).to_nat

-- The number of different ways to place fir trees so that no two fir trees are adjacent.
def valid_fir_placements : ℕ := finset.card (finset.range (slots).powerset.filter (λ s, finset.card s = fir_trees)).to_nat

-- Probability calculation.
def probability := valid_fir_placements.to_rat / total_arrangements.to_rat

-- Prove that m + n = 41, where the fraction m/n represents the probability in simplest terms.
theorem fir_tree_probability : let m := 1, n := 40 in (probability = m / n) → (m + n = 41) :=
by
  sorry

end fir_tree_probability_l606_606579


namespace min_points_necessary_l606_606491

noncomputable def min_points_on_circle (circumference : ℝ) (dist1 dist2 : ℝ) : ℕ :=
  1304

theorem min_points_necessary :
  ∀ (circumference : ℝ) (dist1 dist2 : ℝ),
  circumference = 1956 →
  dist1 = 1 →
  dist2 = 2 →
  (min_points_on_circle circumference dist1 dist2) = 1304 :=
sorry

end min_points_necessary_l606_606491


namespace least_integer_greater_than_sqrt_450_l606_606486

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, n > real.sqrt 450 ∧ ∀ m : ℕ, m > real.sqrt 450 → n ≤ m :=
begin
  use 22,
  split,
  { sorry },
  { sorry }
end

end least_integer_greater_than_sqrt_450_l606_606486


namespace find_n_l606_606645

theorem find_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 19 ∧ n ≡ -5678 [MOD 20] ∧ n = 2 :=
by
  sorry

end find_n_l606_606645


namespace charlie_father_rope_usage_l606_606028

theorem charlie_father_rope_usage :
  let lengths := [24, 20, 14, 12, 18, 22]
  in list.sum lengths = 110 :=
by
  let lengths := [24, 20, 14, 12, 18, 22]
  show list.sum lengths = 110
  sorry

end charlie_father_rope_usage_l606_606028


namespace cube_color_probability_l606_606978

-- Definitions related to the problem conditions
def face_color := {red, blue, green} -- Set of colors

-- Definition of a cube where each face is painted
structure Cube :=
(front : face_color)
(back : face_color)
(left : face_color)
(right : face_color)
(top : face_color)
(bottom : face_color)

-- Definition of the probability calculation
def cube_probability (c : Cube) : ℚ :=
  if (c.front = c.back) ∧ (c.front = c.left) ∧ (c.front = c.right) then
    if (c.top = c.bottom) then
      1 / 3 ^ 6
    else
      1 / 3 ^ 5
  else
    0

-- Main theorem to prove
theorem cube_color_probability : (75 / 729 : ℚ) = (25 / 243 : ℚ) :=
by sorry

end cube_color_probability_l606_606978


namespace provisions_initial_days_l606_606405

theorem provisions_initial_days (D : ℕ) (P : ℕ) (Q : ℕ) (X : ℕ) (Y : ℕ)
  (h1 : P = 300) 
  (h2 : X = 30) 
  (h3 : Y = 90) 
  (h4 : Q = 200) 
  (h5 : P * D = P * X + Q * Y) : D + X = 120 :=
by
  -- We need to prove that the initial number of days the provisions were meant to last is 120.
  sorry

end provisions_initial_days_l606_606405


namespace johnny_school_distance_l606_606892

theorem johnny_school_distance :
  ∃ d : ℝ, (d / 5 + d / 30 = 1) ∧ d = 30 / 7 :=
begin
  sorry -- Proof omitted
end

end johnny_school_distance_l606_606892


namespace sum_gcd_inequality_l606_606334

open Nat

def gcd (a b : Nat) : Nat := 
if a = 0 then b else 
if b = 0 then a else 
if a = b then a else 
if a > b then gcd (a - b) b else gcd a (b - a)

theorem sum_gcd_inequality (N : Nat) (hN : N > 0) : 
  ∃ N, ∀ n > N, 
    (∑ i in range (n + 1), ∑ j in range (n + 1), gcd i j) > 4 * n^2 := 
by {
  sorry
}

end sum_gcd_inequality_l606_606334


namespace point_on_inverse_proportion_l606_606884

noncomputable def inverse_proportion (x : ℝ) := 4 / x

def P1 : ℝ × ℝ := (1, -4)
def P2 : ℝ × ℝ := (4, -1)
def P3 : ℝ × ℝ := (2, 4)
def P4 : ℝ × ℝ := (2 * Real.sqrt 2, Real.sqrt 2)

theorem point_on_inverse_proportion : P4.snd = inverse_proportion P4.fst := by
  sorry

end point_on_inverse_proportion_l606_606884


namespace trig_solution_l606_606839

open Real

noncomputable def solve_trig_eq (x y z : ℝ) (n k m : ℤ) : Prop :=
  (sin x ≠ 0) ∧ (sin y ≠ 0) ∧ 
  ((sin^2 x + (1/(sin^2 x)))^3 + (sin^2 y + (1/(sin^2 y)))^3 = 16 * cos z) →
  (∃ n k m : ℤ, x = π/2 + π * n ∧ y = π/2 + π * k ∧ z = 2 * π * m)

theorem trig_solution (x y z : ℝ) (n k m : ℤ) :
  solve_trig_eq x y z n k m :=
sorry

end trig_solution_l606_606839


namespace total_cents_l606_606769

/-
Given:
1. Lance has 70 cents.
2. Margaret has three-fourths of a dollar.
3. Guy has two quarters and a dime.
4. Bill has six dimes.

Prove:
The combined total amount of money they have is 265 cents.
-/
theorem total_cents (lance margaret guy bill : ℕ) 
  (hl : lance = 70)
  (hm : margaret = 3 * 100 / 4) -- Margaret's cents
  (hg : guy = 2 * 25 + 10)      -- Guy's cents
  (hb : bill = 6 * 10)          -- Bill's cents
  : lance + margaret + guy + bill = 265 :=
by
  rw [hl, hm, hg, hb]
  norm_num
  sorry

end total_cents_l606_606769


namespace find_y_l606_606104

theorem find_y (h c d : ℕ) (m : ℕ) (h_pos : 1 ≤ h) (h_lt_12 : h < 12)
  (hc_pos : 1 ≤ c) (hc_lt_10 : c < 10)
  (hd_pos : 0 ≤ d) (hd_lt_10 : d < 10) :
  m = 10 * c + d ∧ m = h * (c + d) ∧ 10 * d + c = (12 - h) * (c + d) → 12 - h = (12 - h) :=
by
  intros,
  sorry

end find_y_l606_606104


namespace volume_of_water_in_prism_l606_606927

-- Define the given dimensions and conditions
def length_x := 20 -- cm
def length_y := 30 -- cm
def length_z := 40 -- cm
def angle := 30 -- degrees
def total_volume := 24 -- liters

-- The wet fraction of the upper surface
def wet_fraction := 1 / 4

-- Correct answer to be proven
def volume_water := 18.8 -- liters

theorem volume_of_water_in_prism :
  -- Given the conditions
  (length_x = 20) ∧ (length_y = 30) ∧ (length_z = 40) ∧ (angle = 30) ∧ (wet_fraction = 1 / 4) ∧ (total_volume = 24) →
  -- Prove that the volume of water is as calculated
  volume_water = 18.8 :=
sorry

end volume_of_water_in_prism_l606_606927


namespace least_integer_greater_than_sqrt_450_l606_606453

-- Define that 21^2 is less than 450
def square_of_21_less_than_450 : Prop := 21^2 < 450

-- Define that 450 is less than 22^2
def square_of_22_greater_than_450 : Prop := 450 < 22^2

-- State the theorem that 22 is the smallest integer greater than sqrt(450)
theorem least_integer_greater_than_sqrt_450 
  (h21 : square_of_21_less_than_450) 
  (h22 : square_of_22_greater_than_450) : 
    ∃ n : ℕ, n = 22 ∧ (√450 : ℝ) < n := 
sorry

end least_integer_greater_than_sqrt_450_l606_606453


namespace solution_l606_606228

noncomputable def z : ℂ := 3 - 4i

theorem solution (z : ℂ) (h : i * z = 4 + 3 * i) : z = 3 - 4 * i :=
by sorry

end solution_l606_606228


namespace reflected_arcs_intersect_l606_606774

variable {A B C : Type} [euclidean_geometry] {P Q : point}

/-- Let ABC be an acute triangle. The arcs AB and AC of the circumcircle
    of the triangle are reflected over the lines AB and AC, respectively.
    Prove that the two arcs obtained intersect in another point besides A. -/
theorem reflected_arcs_intersect (h_acute_triangle : triangle_is_acute A B C)
  (circumcircle : ∀ (p : point), (p ∈ circle_through_points A B C) ↔ ∃ θ, p = rotate_about_center A B θ)
  (reflect_arc_ab_over_ab : ∀ (P : point), P ∈ arc A B → reflect_over_line P (line_through_points A B) ∈ arc A B)
  (reflect_arc_ac_over_ac : ∀ (P : point), P ∈ arc A C → reflect_over_line P (line_through_points A C) ∈ arc A C) :
  ∃ H, H ≠ A ∧ intersection_point (reflect_over_line (arc A B) (line_through_points A B)) 
                            (reflect_over_line (arc A C) (line_through_points A C)) = H :=
sorry

end reflected_arcs_intersect_l606_606774


namespace inequality_proof_l606_606539

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l606_606539


namespace simplest_square_root_l606_606948

theorem simplest_square_root : 
  let a1 := Real.sqrt 20
  let a2 := Real.sqrt 2
  let a3 := Real.sqrt (1 / 2)
  let a4 := Real.sqrt 0.2
  a2 = Real.sqrt 2 ∧
  (a1 ≠ Real.sqrt 2 ∧ a3 ≠ Real.sqrt 2 ∧ a4 ≠ Real.sqrt 2) :=
by {
  -- Here, we fill in the necessary proof steps, but it's omitted for now.
  sorry
}

end simplest_square_root_l606_606948


namespace triangle_abc_is_isosceles_l606_606354

variable (A B C M K A1 B1 : Point)

variables (C_M_median : is_median C M A B)
variables (K_on_CM : on_line_segment C M K)
variables (A1_on_BC : on_intersect AK BC A1)
variables (B1_on_AC : on_intersect BK AC B1)
variables (AB1A1B_inscribed : is_inscribed_quadrilateral A B1 A1 B)

theorem triangle_abc_is_isosceles : AB = AC :=
by
  sorry

end triangle_abc_is_isosceles_l606_606354


namespace repeating_decimals_sum_l606_606096

theorem repeating_decimals_sum : (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ) = 4 / 9 :=
 by 
  have h₁ : (0.666666... : ℚ) = 2 / 3,
    -- Since x = 0.6666..., then 10x = 6.6666...,
    -- so 10x - x = 6, then x = 6 / 9, hence 2 / 3
    sorry,
  have h₂ : (0.222222... : ℚ) = 2 / 9,
    -- Since x = 0.2222..., then 10x = 2.2222...,
    -- so 10x - x = 2, then x = 2 / 9
    sorry,
  have h₃ : (0.444444... : ℚ) = 4 / 9,
    -- Since x = 0.4444..., then 10x = 4.4444...,
    -- so 10x - x = 4, then x = 4 / 9
    sorry,
  calc
    (0.666666... : ℚ) + (0.222222... : ℚ) - (0.444444... : ℚ)
        = (2 / 3) + (2 / 9) - (4 / 9) : by rw [h₁, h₂, h₃]
    ... = (6 / 9) + (2 / 9) - (4 / 9) : by norm_num
    ... = 4 / 9 : by ring

end repeating_decimals_sum_l606_606096


namespace min_vertical_segment_length_l606_606625

noncomputable def f₁ (x : ℝ) : ℝ := |x|
noncomputable def f₂ (x : ℝ) : ℝ := -x^2 - 4 * x - 3

theorem min_vertical_segment_length :
  ∃ m : ℝ, m = 3 ∧
            ∀ x : ℝ, abs (f₁ x - f₂ x) ≥ m :=
sorry

end min_vertical_segment_length_l606_606625


namespace part_a_part_b_l606_606728

theorem part_a (n k : ℕ) (points : Fin n → ℝ → Prop) : 
  (∀ i, ∑ p in (points i), p ≤ k) → (∃ i, ∑ p in (points i), p / 2 ≤ 2 * k) :=
sorry

theorem part_b (n k : ℕ) (points : Fin n → ℝ → Prop) : 
  (∀ i, ∑ p in (points i), p ≤ k) → (∃ (rooms : Fin n → Fin (2 * k + 1)), ∀ i j, rooms i = rooms j → i = j → ¬ (played_against i j)) :=
sorry

end part_a_part_b_l606_606728


namespace polygon_centrally_symmetric_and_center_of_symmetry_l606_606127

open Set

variables {α : Type*} [OrderedCommGroup α] (O : Point α) (P : ConvexPolygon α)

theorem polygon_centrally_symmetric_and_center_of_symmetry (h1 : ∀ l : Line α, l ∋ O → divides_area_in_half l P) : 
  is_centrally_symmetric P O :=
sorry

end polygon_centrally_symmetric_and_center_of_symmetry_l606_606127


namespace option_B_is_linear_inequality_with_one_var_l606_606496

noncomputable def is_linear_inequality_with_one_var (in_eq : String) : Prop :=
  match in_eq with
  | "3x^2 > 45 - 9x" => false
  | "3x - 2 < 4" => true
  | "1 / x < 2" => false
  | "4x - 3 < 2y - 7" => false
  | _ => false

theorem option_B_is_linear_inequality_with_one_var :
  is_linear_inequality_with_one_var "3x - 2 < 4" = true :=
by
  -- Add proof steps here
  sorry

end option_B_is_linear_inequality_with_one_var_l606_606496


namespace problem_l606_606723

theorem problem (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ q) : ¬ p ∧ q :=
by
  -- proof goes here
  sorry

end problem_l606_606723


namespace solution_l606_606226

noncomputable def z : ℂ := 3 - 4i

theorem solution (z : ℂ) (h : i * z = 4 + 3 * i) : z = 3 - 4 * i :=
by sorry

end solution_l606_606226


namespace sum_f_from_1_to_2023_l606_606146

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom odd_f_plus_2 : ∀ x : ℝ, f (x + 2) = -f (-x + 2)
axiom f_zero : f 0 = 1

theorem sum_f_from_1_to_2023 : (∑ i in finset.range 2023, f (i + 1)) = -1 := 
by
  sorry

end sum_f_from_1_to_2023_l606_606146


namespace ship_speed_upstream_l606_606932

theorem ship_speed_upstream (v : ℝ) (h1 : 0 ≤ v) : 
  let Speed_downstream := 26 in
  let Speed_ship := Speed_downstream - v in
  Speed_ship - v = 26 - 2 * v :=
by
  unfold Speed_downstream Speed_ship
  sorry

end ship_speed_upstream_l606_606932


namespace proof_problem_l606_606317

theorem proof_problem 
  (a b c : ℝ) 
  (h1 : ∀ x, (x < -4 ∨ (23 ≤ x ∧ x ≤ 27)) ↔ ((x - a) * (x - b) / (x - c) ≤ 0))
  (h2 : a < b) : 
  a + 2 * b + 3 * c = 65 :=
sorry

end proof_problem_l606_606317


namespace largest_odd_digit_multiple_of_5_is_9955_l606_606874

def is_odd_digit (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (nat.digits 10 n), is_odd_digit d

def largest_odd_digit_multiple_of_5 (n : ℕ) : Prop :=
  n < 10000 ∧ n % 5 = 0 ∧ all_odd_digits n

theorem largest_odd_digit_multiple_of_5_is_9955 :
  ∃ n, largest_odd_digit_multiple_of_5 n ∧ n = 9955 :=
begin
  sorry
end

end largest_odd_digit_multiple_of_5_is_9955_l606_606874


namespace smallest_b_l606_606800

theorem smallest_b {a b c d : ℕ} (r : ℕ) 
  (h1 : a = b - r) (h2 : c = b + r) (h3 : d = b + 2 * r) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h5 : a * b * c * d = 256) : b = 4 :=
by
  sorry

end smallest_b_l606_606800


namespace range_of_a_l606_606705

-- Defining the propositions P and Q 
def P (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + a*x + 1 > 0
def Q (a : ℝ) : Prop := ∃ x1 x2 : ℝ, x1^2 - x1 + a = 0 ∧ x2^2 - x2 + a = 0

-- Stating the theorem
theorem range_of_a (a : ℝ) :
  (P a ∧ ¬Q a) ∨ (¬P a ∧ Q a) ↔ a ∈ Set.Ioo (1/4 : ℝ) 4 ∪ Set.Iio 0 :=
sorry

end range_of_a_l606_606705


namespace angle_equal_proof_l606_606688

theorem angle_equal_proof
  (A B C D K T M : Point)
  (h_isosceles : is_isosceles_triangle A B C)
  (h_D_on_AC : lies_on D (segment A C))
  (h_K_minor_arc : lies_on_minor_arc K (circumcircle B C D))
  (h_T_on_parallel : lies_on T (line_through_parallel_point A B C K))
  (h_M_midpoint : is_midpoint M D T):
  angle A K T = angle C A M := by
  sorry

end angle_equal_proof_l606_606688


namespace least_int_gt_sqrt_450_l606_606432

theorem least_int_gt_sqrt_450 : ∃ n : ℕ, n > nat.sqrt 450 ∧ (∀ m : ℕ, m > nat.sqrt 450 → n ≤ m) :=
by
  have h₁ : 20^2 = 400 := by norm_num
  have h₂ : 21^2 = 441 := by norm_num
  have h₃ : 22^2 = 484 := by norm_num
  have sqrt_bounds : 21 < real.sqrt 450 ∧ real.sqrt 450 < 22 :=
    by sorry
  use 22
  split
  · sorry
  · sorry

end least_int_gt_sqrt_450_l606_606432


namespace product_floor_ceil_l606_606633

theorem product_floor_ceil :
  let product := ∏ n in Finset.range 5, (Int.floor (n + 0.5) * Int.ceil (-n - 0.5))
  in product = 0 :=
by
  sorry

end product_floor_ceil_l606_606633


namespace solve_complex_equation_l606_606211

theorem solve_complex_equation (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : 
  z = 3 - 4 * complex.I :=
sorry

end solve_complex_equation_l606_606211


namespace find_angle_C_l606_606266

-- Defining the given conditions
def a : ℝ := 2
def c : ℝ := real.sqrt 6
def A : ℝ := real.pi / 4 -- converting 45 degrees to radians

-- Math proof problem to be proved
theorem find_angle_C (C : ℝ) : 
  sin C = (c * sin A) / a → 
  a < c →
  (C = real.pi / 3 ∨ C = 2 * real.pi / 3) := 
sorry

end find_angle_C_l606_606266


namespace derivative_of_f_l606_606991

noncomputable def f (x : ℝ) : ℝ := x + x⁻¹

theorem derivative_of_f (x : ℝ) (h : x ≠ 0) : deriv f x = 1 - x^(-2) :=
by
  sorry

end derivative_of_f_l606_606991


namespace image_of_A_under_f_l606_606701

-- Define the set A
def A := { p : ℝ × ℝ | p.1 + p.2 = 1 }

-- Define the function f mapping elements of A to B
def f (p : A) : ℝ × ℝ := (2^p.1, 2^p.2)

-- Define the set B
def B := { q : ℝ × ℝ | q.1 * q.2 = 2 ∧ q.1 > 0 ∧ q.2 > 0 }

-- The theorem we need to prove
theorem image_of_A_under_f :
  ∀ q : ℝ × ℝ, q ∈ set.image f A ↔ q ∈ B :=
by
  sorry

end image_of_A_under_f_l606_606701


namespace solve_for_z_l606_606225

variable (z : ℂ)

theorem solve_for_z (h : (complex.I * z = 4 + 3 * complex.I)) : z = 3 - 4 * complex.I := 
sorry

end solve_for_z_l606_606225


namespace ravi_multiple_of_average_jump_l606_606371

def sum_heights := 23 + 27 + 28
def average_jump := sum_heights / 3
def ravi_jump := 39

theorem ravi_multiple_of_average_jump : ravi_jump / average_jump = 1.5 :=
by
  -- This is where the proof would go.
  sorry

end ravi_multiple_of_average_jump_l606_606371


namespace cos_double_angle_given_tan_l606_606120

theorem cos_double_angle_given_tan (x : ℝ) (h : Real.tan x = 2) : Real.cos (2 * x) = -3 / 5 :=
by sorry

end cos_double_angle_given_tan_l606_606120


namespace ratio_of_ages_l606_606831

variables (R J K : ℕ)

axiom h1 : R = J + 8
axiom h2 : R + 4 = 2 * (J + 4)
axiom h3 : (R + 4) * (K + 4) = 192

theorem ratio_of_ages : (R - J) / (R - K) = 2 :=
by sorry

end ratio_of_ages_l606_606831


namespace repetend_five_seventeen_l606_606653

noncomputable def repetend_of_fraction (n : ℕ) (d : ℕ) : ℕ := sorry

theorem repetend_five_seventeen : repetend_of_fraction 5 17 = 294117647058823529 := sorry

end repetend_five_seventeen_l606_606653


namespace d_n_is_geom_seq_l606_606151

-- Define the geometric sequence {c_n} with positive terms
def geom_seq (c : ℕ → ℝ) := ∃ r > 0, ∀ n, c (n + 1) = c n * r

-- Define the sequence d_n as the geometric mean of the first n terms of c_n
def d_n (c : ℕ → ℝ) (n : ℕ) := (∏ i in finset.range n, c (i + 1))^(1 / n.to_real)

-- Prove that {d_n} is also a geometric sequence
theorem d_n_is_geom_seq (c : ℕ → ℝ) (hc : ∀ n, c n > 0) (h_geom : geom_seq c) :
  geom_seq (d_n c) :=
sorry

end d_n_is_geom_seq_l606_606151


namespace circles_intersect_l606_606150

theorem circles_intersect
  (r : ℝ) (R : ℝ) (d : ℝ)
  (hr : r = 4)
  (hR : R = 5)
  (hd : d = 6) :
  1 < d ∧ d < r + R :=
by
  sorry

end circles_intersect_l606_606150


namespace cone_lateral_surface_area_l606_606385

theorem cone_lateral_surface_area (r l : ℝ) (hr : r = 2) (hl : l = 6) : real.pi * r * l = 12 * real.pi :=
by
  -- Use the given conditions
  rw [hr, hl]
  -- Simplify the left-hand side
  calc
    real.pi * 2 * 6 = 12 * real.pi :
      by ring

end cone_lateral_surface_area_l606_606385


namespace least_integer_greater_than_sqrt_450_l606_606451

-- Define that 21^2 is less than 450
def square_of_21_less_than_450 : Prop := 21^2 < 450

-- Define that 450 is less than 22^2
def square_of_22_greater_than_450 : Prop := 450 < 22^2

-- State the theorem that 22 is the smallest integer greater than sqrt(450)
theorem least_integer_greater_than_sqrt_450 
  (h21 : square_of_21_less_than_450) 
  (h22 : square_of_22_greater_than_450) : 
    ∃ n : ℕ, n = 22 ∧ (√450 : ℝ) < n := 
sorry

end least_integer_greater_than_sqrt_450_l606_606451


namespace repetend_of_decimal_expansion_l606_606651

theorem repetend_of_decimal_expansion :
  ∃ (r : ℕ), decimal_repetend (5 / 17) = some r ∧ r = 294117647058823529 :=
by
  sorry

end repetend_of_decimal_expansion_l606_606651


namespace min_n_minus_m_l606_606158

noncomputable def f : ℝ → ℝ :=
λ x, if x > 1 then log x else (1/2) * x + (1/2)

theorem min_n_minus_m (m n : ℝ) (hmn : m < n) (hfn : f m = f n) : n - m = 3 - 2 * log 2 :=
sorry

end min_n_minus_m_l606_606158


namespace inequality_proof_l606_606515

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l606_606515


namespace solve_complex_equation_l606_606208

theorem solve_complex_equation (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : 
  z = 3 - 4 * complex.I :=
sorry

end solve_complex_equation_l606_606208


namespace volume_of_inscribed_sphere_l606_606934

theorem volume_of_inscribed_sphere (cube_edge_length : ℝ) (h : cube_edge_length = 10) : 
  ∃ (V : ℝ), V = (4/3) * π * (5 ^ 3) ∧ V = (500/3) * π :=
by
  use (4/3) * π * (5 ^ 3)
  split
  sorry
  sorry

end volume_of_inscribed_sphere_l606_606934


namespace inequality_xyz_l606_606522

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x))) +
  (y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y))) +
  (z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 :=
by sorry

end inequality_xyz_l606_606522


namespace hiking_supplies_l606_606584

theorem hiking_supplies (hours_per_day : ℕ) (days : ℕ) (rate_mph : ℝ) 
    (supply_per_mile : ℝ) (resupply_rate : ℝ)
    (initial_pack_weight : ℝ) : 
    hours_per_day = 8 → days = 5 → rate_mph = 2.5 → 
    supply_per_mile = 0.5 → resupply_rate = 0.25 → 
    initial_pack_weight = (40 : ℝ) :=
by
  intros hpd hd rm spm rr
  sorry

end hiking_supplies_l606_606584


namespace chessboard_can_be_divided_l606_606344

-- Definitions to capture the chessboard and marked squares
structure Chessboard :=
(width : ℕ)
(height : ℕ)
(squares : fin(width * height) → Prop)

structure Mark :=
(row : fin 8)
(col : fin 8)

structure Problem :=
(chessboard : Chessboard)
(mark1 : Mark)
(mark2 : Mark)

-- The main theorem statement
theorem chessboard_can_be_divided (cb : Chessboard)
(mark1 mark2 : Mark) (h : cb.width = 8 ∧ cb.height = 8)
: (∃ horiz_cut vert_cut : ℕ,
    horiz_cut < cb.width ∧ vert_cut < cb.height ∧
    partitioned_halves_identical cb mark1 mark2 horiz_cut vert_cut) :=
sorry

end chessboard_can_be_divided_l606_606344


namespace fraction_of_females_this_year_l606_606362

theorem fraction_of_females_this_year 
  (participation_increase : ℝ := 0.15)
  (males_increase : ℝ := 0.10)
  (females_increase : ℝ := 0.25)
  (males_last_year : ℤ := 30) :
  let males_this_year := males_last_year * (1 + males_increase)
  let total_participants_last_year := males_last_year + y
  let y : ℝ := (total_participants_increase - males_this_year - males_last_year * females_increase) / females_increase
  let females_this_year := y * (1 + females_increase)
  let total_participants_this_year := total_participants_last_year * (1 + participation_increase)
  in total_participants_this_year = males_this_year + females_this_year →
  females_this_year / total_participants_this_year = 19 / 52 :=
by
  sorry

end fraction_of_females_this_year_l606_606362


namespace fraction_females_in_league_l606_606364

theorem fraction_females_in_league 
  (participation_increase : ℝ)
  (male_increase : ℝ)
  (female_increase : ℝ)
  (males_last_year : ℕ)
  (total_increase : ℝ := 1.15) -- given as 15% higher
  (males_incr : ℝ := 1.10)    -- given as 10% higher
  (females_incr : ℝ := 1.25)  -- given as 25% higher
  (males_last_year_val : ℕ := 30) -- given as 30 males last year
  (total_participants : ℝ) :
  total_increase * (males_last_year + total_participants) = 
  males_incr * males_last_year + females_incr * total_participants →
  (females_incr * total_participants) / (males_incr * males_last_year + females_incr * total_participants) = 25/69 := 
by
  intro h,
  sorry

end fraction_females_in_league_l606_606364


namespace polynomial_factorization_l606_606772

theorem polynomial_factorization :
  ∃ p q : Polynomial ℤ,
  monic p ∧ monic q ∧
  p.natDegree > 0 ∧ q.natDegree > 0 ∧
  (X ^ 8 - 50 * X ^ 4 + 9) = p * q ∧
  p.eval 1 + q.eval 1 = -48 := 
sorry

end polynomial_factorization_l606_606772


namespace monotonic_intervals_range_of_a_harmonic_sum_less_than_log_l606_606319

-- Definition of the function f
def f (x a : ℝ) : ℝ := x * real.exp x - 2 * a * real.exp x

-- Definition of the function g
def g (x a : ℝ) : ℝ := -2 - a * x

-- Proof in Lean
theorem monotonic_intervals (a : ℝ) : 
  (a ≤ 1/2 → ∀ x, 0 ≤ x → f x a ≥ 0) ∧
  (a > 1/2 → (∀ x, 2 * a - 1 < x → f x a ≥ 0) ∧
             (∀ x, 0 ≤ x →  x < 2 * a - 1 → f x a ≤ 0)) :=
sorry

theorem range_of_a (a : ℝ) (x : ℝ) (hx : x ≥ 0) :
  f x a > g x a → a ∈ set.Iic (1 : ℝ) :=
sorry

theorem harmonic_sum_less_than_log (n : ℕ) (hn : n > 0) :
  (∑ i in finset.range (n + 1), 1 / i.succ) < real.log (2 * n + 1) :=
sorry

end monotonic_intervals_range_of_a_harmonic_sum_less_than_log_l606_606319


namespace three_digit_numbers_divisible_by_11_l606_606189

theorem three_digit_numbers_divisible_by_11 : 
  let lower_bound := 100;
      upper_bound := 999 in
  let start := Nat.ceil (lower_bound / 11) in
  let end := Nat.floor (upper_bound / 11) in
  end - start + 1 = 81 := 
by
  let lower_bound := 100;
      upper_bound := 999 in
  let start := Nat.ceil (lower_bound / 11) in
  let end := Nat.floor (upper_bound / 11) in
  exact (end - start + 1 = 81)

end three_digit_numbers_divisible_by_11_l606_606189


namespace intersection_points_of_absolute_value_graphs_l606_606973

theorem intersection_points_of_absolute_value_graphs :
  (∀ x : ℝ, |3 * x + 4| ≠ -|4 * x - 1|) :=
begin
  intros x,
  sorry
end

end intersection_points_of_absolute_value_graphs_l606_606973


namespace interesting_removal_l606_606614

-- Definitions used in Lean 4 statement should only directly appear in the conditions
-- Each condition in a) should be used as a definition in Lean 4

structure Polygon (n : ℕ) :=
(vertices : Fin n → ℝ)
(angle_sum : ℝ)

def is_interesting (n : ℕ) (P : Polygon n) (coloring : Fin n → Bool) : Prop :=
  let blue_sum := ∑ i in Finset.univ.filter (λ i => coloring i), P.vertices i
  let red_sum := ∑ i in Finset.univ.filter (λ i => ¬ coloring i), P.vertices i
  blue_sum = red_sum

variable {n : ℕ}

theorem interesting_removal (P : Polygon (n + 1)) (marked : Fin (n + 1))
  (h : ∀ i, i ≠ marked → is_interesting n (Polygon.mk (λ j => if j.1 < i.1 then P.vertices ⟨j.1, Nat.lt_of_lt_pred j.2⟩ else P.vertices ⟨j.1 + 1, Nat.lt_succ_of_lt j.2⟩) P.angle_sum)) :
  is_interesting n (Polygon.mk (λ i => if i.1 < marked.1 then P.vertices ⟨i.1, Nat.lt_of_lt_pred i.2⟩ else P.vertices ⟨i.1 + 1, Nat.lt_succ_of_lt i.2⟩) P.angle_sum) :=
by
  sorry -- Proof omitted

end interesting_removal_l606_606614


namespace player_A_has_winning_strategy_l606_606366

def game_grid := (19, 94)
def valid_k (k : ℕ) : Prop := 1 ≤ k ∧ k ≤ 19
def can_blackout (grid : ℕ × ℕ) (k : ℕ) : Prop :=
  grid.1 ≥ k ∧ grid.2 ≥ k

theorem player_A_has_winning_strategy :
  ∀ (grid : ℕ × ℕ), grid = game_grid → 
  (∀ (k : ℕ), valid_k k → can_blackout grid k) →
  ∃ (strategy : ℕ × ℕ → ℕ × ℕ → Prop), 
  (optimal_play strategy → player_A_wins strategy) :=
by
  intro grid h_grid k h_k blackout
  sorry

end player_A_has_winning_strategy_l606_606366


namespace g_neither_even_nor_odd_l606_606300

def g (x : ℝ) : ℝ := 3^(x^2 + 3*x + 2) - |x| + 3*x

theorem g_neither_even_nor_odd : ¬ (∀ x, g (-x) = g x) ∧ ¬ (∀ x, g (-x) = -g x) :=
by
  intros
  sorry

end g_neither_even_nor_odd_l606_606300


namespace dealer_is_cheating_l606_606931

variable (w a : ℝ)
noncomputable def measured_weight (w : ℝ) (a : ℝ) : ℝ :=
  (a * w + w / a) / 2

theorem dealer_is_cheating (h : a > 0) : measured_weight w a ≥ w :=
by
  sorry

end dealer_is_cheating_l606_606931


namespace probability_alternating_draws_equals_one_div_462_l606_606563

-- Define the problem and conditions
def box_contains_6_white_balls_and_6_black_balls : Prop := 
  ∃ (black white : ℕ), black = 6 ∧ white = 6

-- Define the event of drawing balls in an alternating color pattern starting with a black ball
def alternating_draws : Prop := 
  ∀ (draws : list ℕ), length draws = 12 ∧
  (∀ i, i < 12 → (draws.nth i = some 0 ↔ i % 2 = 0) ∧ (draws.nth i = some 1 ↔ i % 2 = 1))

-- Define the probability calculation
def probability_of_alternating_draws :=
  (1 / (6.choose 6)) * (1 / 2)^(length (list.range 12))

-- State the theorem
theorem probability_alternating_draws_equals_one_div_462 :
  box_contains_6_white_balls_and_6_black_balls →
  alternating_draws →
  probability_of_alternating_draws = (1 / 462) :=
by sorry

end probability_alternating_draws_equals_one_div_462_l606_606563


namespace village_population_rate_l606_606870

theorem village_population_rate (r : ℕ) :
  let PX := 72000
  let PY := 42000
  let decrease_rate_X := 1200
  let years := 15
  let population_X_after_years := PX - decrease_rate_X * years
  let population_Y_after_years := PY + r * years
  population_X_after_years = population_Y_after_years → r = 800 :=
by
  sorry

end village_population_rate_l606_606870


namespace smallest_p_l606_606815

def between_parabolas (u v : ℝ) : Prop :=
  u^2 ≤ v ∧ v ≤ u^2 + 1

theorem smallest_p (p : ℝ) (line_segment : ℝ × ℝ → ℝ) :
  (∀ (x : ℝ), between_parabolas x (line_segment (-1, 0)) ∧
               between_parabolas x (line_segment (0, 0)) ∧
               between_parabolas x (line_segment (1, 0))) →
               p = 9 / 8 → 
               (∀ (x : ℝ), between_parabolas x (line_segment x, 0) → x^2 ≤ line_segment x + p) :=
sorry

end smallest_p_l606_606815


namespace probability_at_most_one_red_ball_l606_606911

noncomputable def combin : ℕ → ℕ → ℕ := λ n k, Nat.choose n k

def totalBalls : ℕ := 36
def whiteBalls : ℕ := 4
def otherBalls : ℕ := totalBalls - whiteBalls
def totalBallsChoose4 : ℕ := combin totalBalls 4
def probAtMostOneRedBall : ℚ := (combin otherBalls 1 * combin whiteBalls 3 + combin whiteBalls 2) / totalBallsChoose4

theorem probability_at_most_one_red_ball :
    probAtMostOneRedBall = (combin 32 1 * combin 4 3 + combin 4 2) / combin 36 4 := by
    sorry

end probability_at_most_one_red_ball_l606_606911


namespace least_integer_greater_than_sqrt_450_l606_606477

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, (n > real.sqrt 450) ∧ ∀ m : ℤ, (m > real.sqrt  450) → n ≤ m :=
by
  apply exists.intro 22
  sorry

end least_integer_greater_than_sqrt_450_l606_606477


namespace pentagon_inequality_l606_606776

-- Given angles
variables (ABCDE : Type) [Pentagon ABCDE] 

axiom angle_A : ∠ A = 120
axiom angle_B : ∠ B = 120
axiom angle_C : ∠ C = 120
axiom angle_D : ∠ D = 120
axiom angle_AED : ∠ AED = 60

-- To prove the following inequality
theorem pentagon_inequality (AC BD AE ED : ℝ) :
  4 * AC * BD ≥ 3 * AE * ED := sorry

end pentagon_inequality_l606_606776


namespace min_value_expression_l606_606327

theorem min_value_expression (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a + b + c = 1) :
  9 ≤ (1 / (a^2 + 2 * b^2)) + (1 / (b^2 + 2 * c^2)) + (1 / (c^2 + 2 * a^2)) :=
by
  sorry

end min_value_expression_l606_606327


namespace inequality_proof_l606_606513

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l606_606513


namespace y_intercepts_count_l606_606972

noncomputable def discriminant (a b c : ℝ) : ℝ :=
    b^2 - 4 * a * c

theorem y_intercepts_count :
    (∀ x y : ℝ, x = 3 * y^2 - 5 * y + 1 → x = 0 → let Δ := discriminant 3 (-5) 1 in Δ = 13) → 2 = 2 :=
by
    intros x y h_eq_x h_eq_zero Δ h_discriminant
    sorry

end y_intercepts_count_l606_606972


namespace polynomial_mod_p_zero_l606_606806

def is_zero_mod_p (p : ℕ) [Fact (Nat.Prime p)] (f : (List ℕ → ℤ)) : Prop :=
  ∀ (x : List ℕ), f x % p = 0

theorem polynomial_mod_p_zero
  (p : ℕ) [Fact (Nat.Prime p)]
  (n : ℕ) 
  (f : (List ℕ → ℤ)) 
  (h : ∀ (x : List ℕ), f x % p = 0) 
  (g : (List ℕ → ℤ)) :
  (∀ (x : List ℕ), g x % p = 0) := sorry

end polynomial_mod_p_zero_l606_606806


namespace min_colors_needed_l606_606615

-- Define the concept of a three-element subset
def three_element_subsets (S : Finset ℕ) : Finset (Finset ℕ) :=
  S.powerset.filter (λ t, t.card = 3)

-- Define the graph based on the given conditions
def vertex_set (k : ℕ) : Type :=
  { t : Finset ℕ // t ∈ three_element_subsets (Finset.range (2^k + 1)) }

def edge_set (k : ℕ) (v1 v2 : vertex_set k) : Prop :=
  (v1.1 ∩ v2.1).card = 1

-- The chromatic number of such a graph
def chromatic_number (k : ℕ) : ℕ :=
  sorry  -- We state it explicitly in the theorem below

theorem min_colors_needed (k : ℕ) :
  chromatic_number k = (1 / 6 : ℚ) * (2^k - 1) * (2^k - 2) :=
sorry

end min_colors_needed_l606_606615


namespace solve_complex_equation_l606_606209

theorem solve_complex_equation (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : 
  z = 3 - 4 * complex.I :=
sorry

end solve_complex_equation_l606_606209


namespace binary_to_decimal_and_octal_l606_606042

theorem binary_to_decimal_and_octal (b : String) (d : ℕ) (o : ℕ) :
  b = "101101" ∧ decimal_val b = 45 ∧ octal_val 45 = 55 :=
by
  have h_bin : b = "101101" := rfl
  have h_dec : decimal_val b = 45 := sorry
  have h_oct : octal_val 45 = 55 := sorry
  exact ⟨h_bin, h_dec, h_oct⟩

end binary_to_decimal_and_octal_l606_606042


namespace relatively_prime_consecutive_squares_l606_606824

variable (b : ℕ)
def a := b - 1
def c := b + 1

theorem relatively_prime_consecutive_squares :
  Nat.gcd (b^2 - a^2) (c^2 - b^2) = 1 :=
by
  sorry

end relatively_prime_consecutive_squares_l606_606824


namespace greatest_divisor_l606_606992

theorem greatest_divisor (n : ℕ) (h1 : 1657 % n = 6) (h2 : 2037 % n = 5) : n = 127 :=
by
  sorry

end greatest_divisor_l606_606992


namespace find_g9_l606_606390

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g3_value : g 3 = 4

theorem find_g9 : g 9 = 64 := sorry

end find_g9_l606_606390


namespace temperature_problem_product_of_possible_N_l606_606602

theorem temperature_problem (M L : ℤ) (N : ℤ) :
  (M = L + N) →
  (M - 8 = L + N - 8) →
  (L + 4 = L + 4) →
  (|((L + N - 8) - (L + 4))| = 3) →
  N = 15 ∨ N = 9 :=
by sorry

theorem product_of_possible_N :
  (∀ M L : ℤ, ∀ N : ℤ,
    (M = L + N) →
    (M - 8 = L + N - 8) →
    (L + 4 = L + 4) →
    (|((L + N - 8) - (L + 4))| = 3) →
    N = 15 ∨ N = 9) →
    15 * 9 = 135 :=
by sorry

end temperature_problem_product_of_possible_N_l606_606602


namespace coefficient_x2_sum_binomials_l606_606747

/-- Prove that the coefficient of x^2 in the expansion of the sum of binomials 
    from (1 + x) to (1 + x)^9 is 120. -/
theorem coefficient_x2_sum_binomials : (∑ n in finset.range 8, nat.choose (n + 2) 2) = 120 := 
by sorry

end coefficient_x2_sum_binomials_l606_606747


namespace least_integer_greater_than_sqrt_450_l606_606483

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, n > real.sqrt 450 ∧ ∀ m : ℕ, m > real.sqrt 450 → n ≤ m :=
begin
  use 22,
  split,
  { sorry },
  { sorry }
end

end least_integer_greater_than_sqrt_450_l606_606483


namespace tangent_line_eq_at_point_l606_606852

theorem tangent_line_eq_at_point :
  let f : ℝ → ℝ := fun x => x^3 - 2 * x^2
  tangent_eq_at_point f 1 (-1) (fun x => -x)
:= sorry

end tangent_line_eq_at_point_l606_606852


namespace find_value_of_m_l606_606670

variable (m : ℝ)

-- The two points A and B
def A : ℝ × ℝ := (-2, m)
def B : ℝ × ℝ := (m, 4)

-- Equation of the line 2x + y + 1 = 0
def line1 (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Line passing through A and B
def line_derivative (A B : ℝ × ℝ) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

-- The slope of the line defined by 2x + y + 1 = 0
def slope_line1 : ℝ := -2

-- The proof statement
theorem find_value_of_m
  (h_slope: line_derivative A B = slope_line1) : m = 8 := sorry

end find_value_of_m_l606_606670


namespace least_integer_gt_sqrt_450_l606_606462

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l606_606462


namespace divides_343_l606_606773

theorem divides_343 
  (x y z : ℕ) 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h : 7 ∣ (x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y)) :
  343 ∣ (x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y) :=
by sorry

end divides_343_l606_606773


namespace maximize_f_l606_606333

theorem maximize_f (x y z u v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0) (hv : v > 0) :
  let f := (x * y + y * z + z * u + u * v) / (2 * x^2 + y^2 + 2 * z^2 + u^2 + 2 * v^2) in
  f ≤ sqrt 6 / 4 :=
sorry

end maximize_f_l606_606333


namespace sum_of_three_consecutive_even_l606_606881

theorem sum_of_three_consecutive_even (a1 a2 a3 : ℤ) (h1 : a1 % 2 = 0) (h2 : a2 = a1 + 2) (h3 : a3 = a1 + 4) (h4 : a1 + a3 = 128) : a1 + a2 + a3 = 192 :=
sorry

end sum_of_three_consecutive_even_l606_606881


namespace cost_of_letter_is_0_37_l606_606612

-- Definitions based on the conditions
def total_cost : ℝ := 4.49
def package_cost : ℝ := 0.88
def num_letters : ℕ := 5
def num_packages : ℕ := 3
def letter_cost (L : ℝ) : ℝ := 5 * L
def package_total_cost : ℝ := num_packages * package_cost

-- Theorem that encapsulates the mathematical proof problem
theorem cost_of_letter_is_0_37 (L : ℝ) (h : letter_cost L + package_total_cost = total_cost) : L = 0.37 :=
by sorry

end cost_of_letter_is_0_37_l606_606612


namespace man_l606_606923

/-- A man can row downstream at the rate of 45 kmph.
    A man can row upstream at the rate of 23 kmph.
    The rate of current is 11 kmph.
    The man's rate in still water is 34 kmph. -/
theorem man's_rate_in_still_water
  (v c : ℕ)
  (h1 : v + c = 45)
  (h2 : v - c = 23)
  (h3 : c = 11) : v = 34 := by
  sorry

end man_l606_606923


namespace least_integer_greater_than_sqrt_450_l606_606467

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606467


namespace sin_BAD_div_sin_CAD_l606_606296

theorem sin_BAD_div_sin_CAD
  (B C : Type*) [EuclideanSpace B] [EuclideanSpace C]
  (BD CD : ℝ)
  (AngleB AngleC : ℝ)
  (hB : AngleB = π / 3)
  (hC : AngleC = π / 4)
  (h_ratio : BD / CD = 1 / 3) :
  (sin (angle BAD)) / (sin (angle CAD)) = sqrt 6 / 6 :=
by
  sorry

end sin_BAD_div_sin_CAD_l606_606296


namespace sachin_age_38_5_l606_606894

-- Definitions
def Sachin_age (S R : ℝ) := S = R + 7
def age_ratio (S R : ℕ) := 9 * S = 11 * R

-- Theorem statement
theorem sachin_age_38_5 (R S : ℝ) 
  (h1 : Sachin_age S R)
  (h2 : age_ratio S R) : S = 38.5 :=
by 
  -- Proof is omitted as per instructions
  sorry

end sachin_age_38_5_l606_606894


namespace complex_solution_l606_606206

theorem complex_solution (z : ℂ) (h : ((0 : ℝ) + 1 * z = 4 + 3 * (complex.I))) : 
  z = 3 - 4 * (complex.I) :=
sorry

end complex_solution_l606_606206


namespace math_equivalent_problem_l606_606660

/-- Given conditions -/
structure ProblemData where
  F : ℝ × ℝ := (1, 0)
  l_x : ℝ := -1
  M : ℝ × ℝ := (-1, 0)
  eq_condition : ∀ (P : ℝ × ℝ), let Q := (l_x, P.2) in 
                                 (Q.1 - P.1) * (Q.2 - P.2) + (Q.1 - F.1) * (Q.2 - F.2) = (F.1 - P.1) * (F.2 - P.2) * (F.1 - Q.1)

/-- Define the trajectory equation G of the moving point P -/
def trajectory (P : ℝ × ℝ) : Prop :=
  P.2^2 = 4 * P.1

/-- Define slopes for proving complementary angles -/
def slope (P1 P2 : ℝ × ℝ) : ℝ :=
  (P2.2 - P1.2) / (P2.1 - P1.1)

/-- Proof problem -/
theorem math_equivalent_problem (data : ProblemData) :
  (∀ P, data.eq_condition P ↔ trajectory P) ∧ 
  (∀ (A B C D : ℝ × ℝ),
    slope (data.F) A + slope (data.F) B = 0 → 
    slope A C + slope B D = 0) ∧ 
  (∀ (C D : ℝ × ℝ), 
    C.1 = 1 ∧ C.2 = 0 → 
    (∃! (P : ℝ × ℝ), slope C D = slope C (1, 0) ∧ P = (1, 0))) := by
sory

end math_equivalent_problem_l606_606660


namespace least_integer_greater_than_sqrt_450_l606_606484

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, n > real.sqrt 450 ∧ ∀ m : ℕ, m > real.sqrt 450 → n ≤ m :=
begin
  use 22,
  split,
  { sorry },
  { sorry }
end

end least_integer_greater_than_sqrt_450_l606_606484


namespace sample_size_is_15_l606_606940

-- Define the given conditions as constants and assumptions within the Lean environment.
def total_employees := 750
def young_workers := 350
def middle_aged_workers := 250
def elderly_workers := 150
def sample_young_workers := 7

-- Define the proposition that given these conditions, the sample size is 15.
theorem sample_size_is_15 : ∃ n : ℕ, (7 / n = 350 / 750) ∧ n = 15 := by
  sorry

end sample_size_is_15_l606_606940


namespace greatest_difference_units_digit_l606_606865

theorem greatest_difference_units_digit :
  ∀ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 5 = 0 ∧ (n / 10) = 74 → (n % 10 = 0 ∨ n % 10 = 5) ∧ (max (n % 10) (if n % 10 = 0 then 5 else 0) - min (n % 10) (if n % 10 = 0 then 0 else 0) = 5) :=
by
  -- Specify that n must be a three-digit integer and a multiple of 5
  intros n h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  cases h6 with h7 h8

  -- Prove that the units digit must be either 0 or 5
  have h9 : n % 10 = 0 ∨ n % 10 = 5,
    from
      by
        cases h7 with h10 h11
        case or.inl =>
          -- if n % 10 is 0
          exact h10
        case or.inr =>
          -- if n % 10 is 5
          exact h11

  -- Calculate the greatest possible difference between these units digits
  have h10 : max (n % 10) (if n % 10 = 0 then 5 else 0) - min (n % 10) (if n % 10 = 0 then 0 else 0) = 5,
    from
      by
        cases h9,
        case or.inl =>
          -- if n % 10 is 0
          exact rfl
        case or.inr =>
          -- if n % 10 is 5
          rw [nat.max_eq_left],
          exact nat.sub_eq_zero_of_le,
          exact rfl
          exact zero_le

  -- Return the result as a combination of the units digit and the greatest possible difference
  exact ⟨h9, h10⟩
end

end greatest_difference_units_digit_l606_606865


namespace quadratic_fixed_points_l606_606591

theorem quadratic_fixed_points (m : ℝ) (x : ℝ) (y : ℝ) (h₀ : m ≠ 0) :
  (∀ (m : ℝ), y = m * x^2 - 2 * m * x + 3 → (x = 0 ∧ y = 3) ∨ (x = 2 ∧ y = 3)) :=
by
  intros m h₀ hx
  sorry

end quadratic_fixed_points_l606_606591


namespace angle_BDC_in_quadrilateral_l606_606734

theorem angle_BDC_in_quadrilateral (BCDE : Type) [quadrilateral BCDE]
  (A E C B D : BCDE)
  (angle_A : angle A = 45)
  (angle_E : angle E = 25)
  (angle_C : angle C = 20)
  : angle BDC = 45 :=
sorry

end angle_BDC_in_quadrilateral_l606_606734


namespace least_integer_gt_sqrt_450_l606_606443

theorem least_integer_gt_sqrt_450 : 
  let x := 450 in
  (21 * 21 = 441) → 
  (22 * 22 = 484) →
  (441 < x ∧ x < 484) →
  (∃ n : ℤ, n = 22 ∧ n > int.sqrt x) :=
by
  intros h1 h2 h3
  use 22
  split
  · refl
  · sorry

end least_integer_gt_sqrt_450_l606_606443


namespace rainfall_ratio_l606_606054

noncomputable def total_rainfall := 35
noncomputable def rainfall_second_week := 21

theorem rainfall_ratio 
  (R1 R2 : ℝ)
  (hR2 : R2 = rainfall_second_week)
  (hTotal : R1 + R2 = total_rainfall) :
  R2 / R1 = 3 / 2 := 
by 
  sorry

end rainfall_ratio_l606_606054


namespace inequality_proof_l606_606528

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606528


namespace line_AB_parallel_to_plane_l606_606692

-- Define the vectors and the point on the plane
def vec_n : Vector ℝ := ⟨[2, -2, 4]⟩
def vec_AB : Vector ℝ := ⟨[-3, 1, 2]⟩

-- Define the condition that point A is not on the plane
axiom point_A_not_on_plane : True

-- Define the property to prove
theorem line_AB_parallel_to_plane :
  vec_n ⬝ vec_AB = 0 → (∀ A point, point_A_not_on_plane) → AB ∥ α := 
by
  sorry

end line_AB_parallel_to_plane_l606_606692


namespace total_cents_l606_606770

/-
Given:
1. Lance has 70 cents.
2. Margaret has three-fourths of a dollar.
3. Guy has two quarters and a dime.
4. Bill has six dimes.

Prove:
The combined total amount of money they have is 265 cents.
-/
theorem total_cents (lance margaret guy bill : ℕ) 
  (hl : lance = 70)
  (hm : margaret = 3 * 100 / 4) -- Margaret's cents
  (hg : guy = 2 * 25 + 10)      -- Guy's cents
  (hb : bill = 6 * 10)          -- Bill's cents
  : lance + margaret + guy + bill = 265 :=
by
  rw [hl, hm, hg, hb]
  norm_num
  sorry

end total_cents_l606_606770


namespace least_squares_minimization_l606_606671

variables {ι : Type*} (n : ℕ)
variables (a b : ℝ) (x y : ι → ℝ)

noncomputable def least_squares_objective (a b : ℝ) (x y : ι → ℝ) : ℝ :=
∑ i in finset.range n, (y i - (a + b * x i))^2

theorem least_squares_minimization (x y : ι → ℝ) (a b : ℝ) :
  ∃ a b, is_least (λ ab : ℝ × ℝ, least_squares_objective n ab.1 ab.2 x y) (a, b) :=
sorry

end least_squares_minimization_l606_606671


namespace count_triangles_not_collinear_l606_606289

def in_bounds (x y : ℕ) : Prop := 1 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 4

def points : List (ℕ × ℕ) :=
  [ (x, y) | x ← [1, 2, 3, 4], y ← [1, 2, 3, 4] ]

theorem count_triangles_not_collinear :
  let total_points := 16
  let combinations := Nat.choose total_points 3
  let collinear_points := 44
  combinations - collinear_points = 516 := by
    sorry

end count_triangles_not_collinear_l606_606289


namespace base_angle_of_isosceles_triangle_l606_606284

theorem base_angle_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : a = 50) (h₂ : a + b + c = 180) (h₃ : a = b ∨ b = c ∨ c = a) : 
  b = 50 ∨ b = 65 :=
by sorry

end base_angle_of_isosceles_triangle_l606_606284


namespace least_integer_gt_sqrt_450_l606_606445

theorem least_integer_gt_sqrt_450 : 
  let x := 450 in
  (21 * 21 = 441) → 
  (22 * 22 = 484) →
  (441 < x ∧ x < 484) →
  (∃ n : ℤ, n = 22 ∧ n > int.sqrt x) :=
by
  intros h1 h2 h3
  use 22
  split
  · refl
  · sorry

end least_integer_gt_sqrt_450_l606_606445


namespace inequality_proof_l606_606529

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606529


namespace solve_for_z_l606_606236

variable {z : ℂ}

theorem solve_for_z (h : complex.I * z = 4 + 3*complex.I) : z = 3 - 4*complex.I :=
sorry

end solve_for_z_l606_606236


namespace cylindrical_to_rectangular_l606_606044

open real

theorem cylindrical_to_rectangular (r θ z x y : ℝ) (h₀ : r = 7) (h₁ : θ = π / 3) (h₂ : z = -3) 
(h₃ : x = r * cos θ) (h₄ : y = r * sin θ) :
  (x, y, z) = (3.5, (7 * sqrt 3) / 2, -3) :=
by
  rw [h₀, h₁, h₂, h₃, h₄]
  sorry

end cylindrical_to_rectangular_l606_606044


namespace cylindrical_to_rectangular_l606_606045

structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

structure RectangularCoord where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def convertCylindricalToRectangular (c : CylindricalCoord) : RectangularCoord :=
  { x := c.r * Real.cos c.θ,
    y := c.r * Real.sin c.θ,
    z := c.z }

theorem cylindrical_to_rectangular :
  convertCylindricalToRectangular ⟨7, Real.pi / 3, -3⟩ = ⟨3.5, 7 * Real.sqrt 3 / 2, -3⟩ :=
by sorry

end cylindrical_to_rectangular_l606_606045


namespace triangle_abc_is_isosceles_l606_606353

variable (A B C M K A1 B1 : Point)

variables (C_M_median : is_median C M A B)
variables (K_on_CM : on_line_segment C M K)
variables (A1_on_BC : on_intersect AK BC A1)
variables (B1_on_AC : on_intersect BK AC B1)
variables (AB1A1B_inscribed : is_inscribed_quadrilateral A B1 A1 B)

theorem triangle_abc_is_isosceles : AB = AC :=
by
  sorry

end triangle_abc_is_isosceles_l606_606353


namespace binomial_fourth_term_l606_606690

theorem binomial_fourth_term :
  ∀ (a : ℝ) (x : ℝ),
    a = ∫ t in 1..2, (3 * t^2 - 2 * t) →
    (binomial.expansion ((a * x^2) - (1 / x)) 6).nth 4 = -1280 * x^3 :=
by
  sorry

end binomial_fourth_term_l606_606690


namespace inequality_proof_l606_606552

variable {x y z : ℝ}

theorem inequality_proof (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * real.sqrt(z * x)) + 
  y^3 / (y^3 + 2 * z^2 * real.sqrt(x * y)) + 
  z^3 / (z^3 + 2 * x^2 * real.sqrt(y * z)) >= 1) :=
sorry

end inequality_proof_l606_606552


namespace first_pack_weight_l606_606587

-- Define the conditions
def miles_per_hour := 2.5
def hours_per_day := 8
def days := 5
def supply_per_mile := 0.5
def resupply_percentage := 0.25
def total_hiking_time := hours_per_day * days
def total_miles_hiked := total_hiking_time * miles_per_hour
def total_supplies_needed := total_miles_hiked * supply_per_mile
def resupply_factor := 1 + resupply_percentage

-- Define the theorem
theorem first_pack_weight :
  (total_supplies_needed / resupply_factor) = 40 :=
by
  sorry

end first_pack_weight_l606_606587


namespace sin_cos_identity_l606_606325

theorem sin_cos_identity {x : Real} 
    (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11 / 36) : 
    Real.sin x ^ 12 + Real.cos x ^ 12 = 5 / 18 :=
sorry

end sin_cos_identity_l606_606325


namespace polynomial_remainder_l606_606331

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

-- Statement of the problem: the remainder when h(x^10) is divided by h(x) is 8
theorem polynomial_remainder : (∃ q r : ℝ, r < h(x) ∧ h(x ^ 10) = h(x) * q + r) ∧ r = 8 := sorry

end polynomial_remainder_l606_606331


namespace min_inverse_ab_l606_606812

theorem min_inverse_ab (a b : ℝ) (h1 : a + a * b + 2 * b = 30) (h2 : a > 0) (h3 : b > 0) :
  ∃ m : ℝ, m = 1 / 18 ∧ (∀ x y : ℝ, (x + x * y + 2 * y = 30) → (x > 0) → (y > 0) → 1 / (x * y) ≥ m) :=
sorry

end min_inverse_ab_l606_606812


namespace solve_for_z_l606_606222

variable (z : ℂ)

theorem solve_for_z (h : (complex.I * z = 4 + 3 * complex.I)) : z = 3 - 4 * complex.I := 
sorry

end solve_for_z_l606_606222


namespace no_three_partition_exists_l606_606960

/-- Define the partitioning property for three subsets -/
def partitions (A B C : Set ℤ) : Prop :=
  ∀ n : ℤ, (n ∈ A ∨ n ∈ B ∨ n ∈ C) ∧ (n ∈ A ↔ n-50 ∈ B ∧ n+1987 ∈ C) ∧ (n-50 ∈ A ∨ n-50 ∈ B ∨ n-50 ∈ C) ∧ (n-50 ∈ B ↔ n-50-50 ∈ A ∧ n-50+1987 ∈ C) ∧ (n+1987 ∈ A ∨ n+1987 ∈ B ∨ n+1987 ∈ C) ∧ (n+1987 ∈ C ↔ n+1987-50 ∈ A ∧ n+1987+1987 ∈ B)

/-- The main theorem stating that no such partition is possible -/
theorem no_three_partition_exists :
  ¬∃ A B C : Set ℤ, partitions A B C :=
sorry

end no_three_partition_exists_l606_606960


namespace number_of_cars_l606_606656

theorem number_of_cars (C : ℕ) : 
  let bicycles := 3
  let pickup_trucks := 8
  let tricycles := 1
  let car_tires := 4
  let bicycle_tires := 2
  let pickup_truck_tires := 4
  let tricycle_tires := 3
  let total_tires := 101
  (4 * C + 3 * bicycle_tires + 8 * pickup_truck_tires + 1 * tricycle_tires = total_tires) → C = 15 := by
  intros h
  sorry

end number_of_cars_l606_606656


namespace k_m_sum_l606_606324

theorem k_m_sum (k m : ℝ) (h : ∀ {x : ℝ}, x^3 - 8 * x^2 + k * x - m = 0 → x ∈ {1, 2, 5} ∨ x ∈ {1, 3, 4}) :
  k + m = 27 ∨ k + m = 31 :=
by
  sorry

end k_m_sum_l606_606324


namespace find_common_ratio_l606_606674

-- Defining the conditions in Lean
variables (a : ℕ → ℝ) (d q : ℝ)

-- The arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) - a n = d

-- The geometric sequence condition
def is_geometric_sequence (a1 a2 a4 q : ℝ) : Prop :=
a2 ^ 2 = a1 * a4

-- Proving the main theorem
theorem find_common_ratio (a : ℕ → ℝ) (d q : ℝ) (h_arith : is_arithmetic_sequence a d) (d_ne_zero : d ≠ 0) 
(h_geom : is_geometric_sequence (a 1) (a 2) (a 4) q) : q = 2 :=
by
  sorry

end find_common_ratio_l606_606674


namespace Robie_l606_606372

def initial_bags (X : ℕ) := (X - 2) + 3 = 4

theorem Robie's_initial_bags (X : ℕ) (h : initial_bags X) : X = 3 :=
by
  unfold initial_bags at h
  sorry

end Robie_l606_606372


namespace mu_value_l606_606675

noncomputable def Sn (n : ℕ) : ℝ := sorry
noncomputable def Sm (m : ℕ) : ℝ := sorry
noncomputable def Sk (k : ℕ) : ℝ := sorry

theorem mu_value (n m k : ℕ) (hn : n > 0) (hm : m > 0) (hk : k > 0) :
  let OP := (n, Sn n / n)
  let OP1 := (m, Sm m / m)
  let OP2 := (k, Sk k / k)
  let λ := sorry -- λ can be any arbitrary real number
  let μ := (n - m : ℝ) / (k - m) in
  OP = (λ * OP1.1 + μ * OP2.1, λ * OP1.2 + μ * OP2.2) :=
begin
  sorry
end

end mu_value_l606_606675


namespace find_z_l606_606248

theorem find_z (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : z = 3 - 4 * complex.I :=
by {
  sorry
}

end find_z_l606_606248


namespace bisectors_intersect_on_AB_l606_606731
-- Import the necessary mathematics library

-- Define the problem statement in Lean
theorem bisectors_intersect_on_AB 
  (A B C D M : Type) 
  [convex_quadrilateral A B C D]
  (H1: ∃ (M : Type), angle_bisector_intersects_cd (angle_bisector_of_angle_cad A C D) (angle_bisector_of_angle_cbd B C D) M (is_point_on_CD M C D)):
  ∃ (N : Type), angle_bisector_intersect_ab (angle_bisector_of_angle_adb A D B) (angle_bisector_of_angle_acb A C B) N (is_point_on_AB N A B) :=
sorry

end bisectors_intersect_on_AB_l606_606731


namespace ellipse_standard_equation_and_no_right_angle_l606_606134

theorem ellipse_standard_equation_and_no_right_angle (a b : ℝ) (h1 : a > b ∧ b > 0) 
  (h2 : ∀ (x y : ℝ), (x, y) ∈ set_of (λ p : ℝ × ℝ, ((p.1^2) / (a^2) + (p.2 ^ 2) / (b^2) = 1)) → ((0, 1) = (x, y))) 
  (h3 : eccentricity(a, b) = (√3) / 2) :
  (∀ (x y : ℝ), ((x^2) / 4 + y^2 = 1)) ∧ 
  (∀ (l : line) (A B C D : point), 
    (A = (1, 0)) ∧ (B = (2, 0)) ∧ ((C, D) ∈ intersections(l, ellipse(a, b))) → ¬ angle CBD = π / 2) :=
sorry

end ellipse_standard_equation_and_no_right_angle_l606_606134


namespace oil_price_l606_606559

theorem oil_price (P : ℝ) 
  (price_first_oil : 10 * 40)
  (volume_second_oil : 5)
  (total_mixture_volume : 15)
  (price_mixture : 48.67) :
  400 + 5 * P = 15 * 48.67 -> P = 66.01 :=
by
  intros h
  sorry

end oil_price_l606_606559


namespace angle_MF1F2_eq_30_l606_606686

noncomputable def hyperbola : Type :=
{ x : ℝ // x^2 - y^2 / 2 = 1 }

variables (F1 F2 M : ℝ) (x y : ℝ)

def conditions : Prop :=
x^2 - y^2 / 2 = 1 ∧ ∃ (F1 F2 : ℝ), |F1 - F2| = 2 * sqrt 3 ∧ 
(M ∈ right_branch_hyperbola ∧ |M - F1| + |M - F2| = 6)

theorem angle_MF1F2_eq_30 (h : conditions F1 F2 M x y) : 
  ∠ MF1 F2 = 30 := 
sorry

end angle_MF1F2_eq_30_l606_606686


namespace fraction_females_in_league_l606_606365

theorem fraction_females_in_league 
  (participation_increase : ℝ)
  (male_increase : ℝ)
  (female_increase : ℝ)
  (males_last_year : ℕ)
  (total_increase : ℝ := 1.15) -- given as 15% higher
  (males_incr : ℝ := 1.10)    -- given as 10% higher
  (females_incr : ℝ := 1.25)  -- given as 25% higher
  (males_last_year_val : ℕ := 30) -- given as 30 males last year
  (total_participants : ℝ) :
  total_increase * (males_last_year + total_participants) = 
  males_incr * males_last_year + females_incr * total_participants →
  (females_incr * total_participants) / (males_incr * males_last_year + females_incr * total_participants) = 25/69 := 
by
  intro h,
  sorry

end fraction_females_in_league_l606_606365


namespace smallest_number_diminished_by_10_divisible_l606_606879

theorem smallest_number_diminished_by_10_divisible :
  ∃ (x : ℕ), (x - 10) % 24 = 0 ∧ x = 34 :=
by
  sorry

end smallest_number_diminished_by_10_divisible_l606_606879


namespace factor_expression_l606_606101

theorem factor_expression (y : ℝ) : 
  5 * y * (y + 2) + 8 * (y + 2) + 15 = (5 * y + 8) * (y + 2) + 15 := 
by
  sorry

end factor_expression_l606_606101


namespace medicine_dosage_l606_606577

theorem medicine_dosage (weight_kg dose_per_kg parts : ℕ) (h_weight : weight_kg = 30) (h_dose_per_kg : dose_per_kg = 5) (h_parts : parts = 3) :
  ((weight_kg * dose_per_kg) / parts) = 50 :=
by sorry

end medicine_dosage_l606_606577


namespace three_digit_numbers_divisible_by_11_l606_606190

theorem three_digit_numbers_divisible_by_11 : 
  let lower_bound := 100;
      upper_bound := 999 in
  let start := Nat.ceil (lower_bound / 11) in
  let end := Nat.floor (upper_bound / 11) in
  end - start + 1 = 81 := 
by
  let lower_bound := 100;
      upper_bound := 999 in
  let start := Nat.ceil (lower_bound / 11) in
  let end := Nat.floor (upper_bound / 11) in
  exact (end - start + 1 = 81)

end three_digit_numbers_divisible_by_11_l606_606190


namespace isosceles_triangle_from_median_and_cyclic_quadrilateral_l606_606357

theorem isosceles_triangle_from_median_and_cyclic_quadrilateral
    {A B C M K A1 B1 : Type*}
    (hCM_med : IsMedian A B C M)
    (hK_on_CM : OnLine K CM)
    (hAK_int_BC : IsIntersection AK BC A1)
    (hBK_int_AC : IsIntersection BK AC B1)
    (quad_cyclic : CyclicQuadrilateral A B1 A1 B) 
    : IsIsoscelesTriangle A B C :=
by
  sorry

end isosceles_triangle_from_median_and_cyclic_quadrilateral_l606_606357


namespace min_value_f_xyz_inequality_l606_606159

-- Define the function f
def f (x : ℝ) : ℝ := 2 * |x + 1| + |x - 2|

-- Question 1: Prove that the minimum value of f(x) is 3
theorem min_value_f : ∃ x : ℝ, f(x) = 3 := by
  sorry

-- Definitions and assumptions for Question 2
variables (a b c : ℝ)
-- Assumption: a, b, and c are positive real numbers
variable (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
-- Assumption: a + b + c = 3
variable (h2 : a + b + c = 3)

-- Question 2: Prove the inequality
theorem xyz_inequality : h1 ∧ h2 → ( b^2 / a + c^2 / b + a^2 / c) ≥ 3 := by
  intro h1 h2
  sorry

end min_value_f_xyz_inequality_l606_606159


namespace inequality_proof_l606_606527

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606527


namespace earrings_cost_l606_606375

theorem earrings_cost (initial_savings necklace_cost remaining_savings : ℕ) 
  (h_initial : initial_savings = 80) 
  (h_necklace : necklace_cost = 48) 
  (h_remaining : remaining_savings = 9) : 
  initial_savings - remaining_savings - necklace_cost = 23 := 
by {
  -- insert proof steps here -- 
  sorry
}

end earrings_cost_l606_606375


namespace ratio_M_N_l606_606715

theorem ratio_M_N (M Q P R N : ℝ) 
(h1 : M = 0.40 * Q) 
(h2 : Q = 0.25 * P) 
(h3 : R = 0.60 * P) 
(h4 : N = 0.75 * R) : 
  M / N = 2 / 9 := 
by
  sorry

end ratio_M_N_l606_606715


namespace triangle_is_isosceles_l606_606348

theorem triangle_is_isosceles 
  (A B C M K A_1 B_1 : Point)
  (h1 : is_median C M A B)
  (h2 : lies_on K C M)
  (h3 : line_intersects AK BC A_1)
  (h4 : line_intersects BK AC B_1)
  (h5 : is_cyclic_quadrilateral A B_1 A_1 B) 
  : is_isosceles_triangle A B C :=
sorry

end triangle_is_isosceles_l606_606348


namespace unique_triple_solution_l606_606642

theorem unique_triple_solution :
  ∀ (m : ℕ) (p q : ℕ), m > 0 → Prime p → Prime q → 2^m * p^2 + 1 = q^5 → (m, p, q) = (1, 11, 3) :=
begin
  assume m p q h1 h2 h3 h4,
  -- solution steps would go here
  sorry
end

end unique_triple_solution_l606_606642


namespace perimeter_8_triangles_count_l606_606181

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def distinct_triangles_count (n : ℕ) : ℕ := 
  (Finset.unorderedPairs (Finset.range (n - 2 + 1))).filter (λ ⟨a, bc⟩, ∃ b c, bc = b + c ∧ is_triangle a b c ∧ a + b + c = n).card

theorem perimeter_8_triangles_count : distinct_triangles_count 8 = 2 := 
by
  sorry

end perimeter_8_triangles_count_l606_606181


namespace jane_wins_probability_l606_606762

-- Define the spinner sectors and the win/loss condition for Jane and her brother
def spinner_sectors : Finset ℕ := Finset.range 6 \ {0}

def win_condition (a b : ℕ) : Bool :=
  | (a, b) => Nat.abs (a - b) < 4

-- Prove the probability that Jane wins
theorem jane_wins_probability : 
  (∑ x in spinner_sectors, ∑ y in spinner_sectors, if win_condition x y then 1 else 0) / (spinner_sectors.card * spinner_sectors.card) = 5 / 6 :=
sorry

end jane_wins_probability_l606_606762


namespace repeating_decimal_sum_l606_606085

theorem repeating_decimal_sum : (0.\overline{6} + 0.\overline{2} - 0.\overline{4} : ℚ) = (4 / 9 : ℚ) :=
by
  sorry

end repeating_decimal_sum_l606_606085


namespace sum_GCF_LCM_l606_606799

-- Definitions of GCD and LCM for the numbers 18, 27, and 36
def GCF : ℕ := Nat.gcd (Nat.gcd 18 27) 36
def LCM : ℕ := Nat.lcm (Nat.lcm 18 27) 36

-- Theorem statement proof
theorem sum_GCF_LCM : GCF + LCM = 117 := by
  sorry

end sum_GCF_LCM_l606_606799


namespace correct_propositions_l606_606140

variables {a b m n : Set ℝ} -- Assuming lines are represented as sets in ℝ
variables {α β : Set (Set ℝ)} -- Assuming planes are sets of sets in ℝ

-- Definitions of parallelism and subsets for lines and planes
def parallel (X Y : Set ℝ) : Prop := ∀ x ∈ X, ∀ y ∈ Y, x - y ≠ 0 → x - y //| x - y ∈ Y
def subset (X Y : Set ℝ) : Prop := ∀ x ∈ X, x ∈ Y
def perpendicular (X Y : Set ℝ) : Prop := ∀ x ∈ X, ∀ y ∈ Y, x • y = 0 -- Assuming simple dot product for simplicity

theorem correct_propositions :
    (parallel (α) (β) ∧ subset (a) (α) → parallel (a) (β))
    ∧ (parallel (α) (β) ∧ subset (m) (α) ∧ subset (n) (β) → parallel (m) (n) ∨ ∃ v ∈ m, ∃ w ∈ n, v ≠ w)
    ∧ (perpendicular (a) (b) ∧ perpendicular (a) (α) → parallel (b) (α) ∨ subset (b) (α))
    ∧ (perpendicular (a) (α) ∧ parallel (α) (β) ∧ parallel (b) (β) → perpendicular (a) (b))
    → true := by sorry

end correct_propositions_l606_606140


namespace unique_solution_exists_l606_606969

theorem unique_solution_exists (n m k : ℕ) :
  n = m^3 ∧ n = 1000 * m + k ∧ 0 ≤ k ∧ k < 1000 ∧ (1000 * m ≤ m^3 ∧ m^3 < 1000 * (m + 1)) → n = 32768 :=
by
  sorry

end unique_solution_exists_l606_606969


namespace inequality_proof_l606_606538

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606538


namespace truck_tank_capacity_l606_606422

-- Definitions based on conditions
def truck_tank (T : ℝ) : Prop := true
def car_tank : Prop := true
def truck_half_full (T : ℝ) : Prop := true
def car_third_full : Prop := true
def add_fuel (T : ℝ) : Prop := T / 2 + 8 = 18

-- Theorem statement
theorem truck_tank_capacity (T : ℝ) (ht : truck_tank T) (hc : car_tank) 
  (ht_half : truck_half_full T) (hc_third : car_third_full) (hf_add : add_fuel T) : T = 20 :=
  sorry

end truck_tank_capacity_l606_606422


namespace scrambles_equal_two_twos_l606_606900

/-- Define a scramble: a rearrangement of a sequence such that no number 
    is in its original position. -/
def is_scramble (orig_seq new_seq : List ℕ) : Prop :=
  (orig_seq.length = new_seq.length) ∧
  ∀ i, i < orig_seq.length → orig_seq.get i ≠ new_seq.get i

/-- Define a two-two: a rearrangement of a sequence where exactly two numbers
    in the new sequence are exactly two more than the numbers that originally
    occupied those positions. -/
def is_two_two (orig_seq new_seq : List ℕ) : Prop :=
  (orig_seq.length = new_seq.length) ∧
  (∃ (i j : ℕ), i < orig_seq.length ∧ j < orig_seq.length ∧ i ≠ j ∧ 
    new_seq.get i = orig_seq.get i + 2 ∧ new_seq.get j = orig_seq.get j + 2 ∧
    ∀ k, k ≠ i ∧ k ≠ j → new_seq.get k = orig_seq.get k)

/-- The first sequence contains the numbers {1,1,2,3,...,n-1,n}. -/
def sequenceA (n : ℕ) : List ℕ :=
  1 :: 1 :: List.range' 2 n

/-- The second sequence contains the numbers {1,2,3,...,n+1}. -/
def sequenceB (n : ℕ) : List ℕ :=
  List.range' 1 (n + 1)

theorem scrambles_equal_two_twos (n : ℕ) (hn : n ≥ 2) :
  ∃ (f : List ℕ → List ℕ), 
    ∀ seqA seqB, 
      seqA = sequenceA n → seqB = sequenceB n →
        (is_scramble seqA (f seqA) ↔ is_two_two seqB (f seqB)) :=
sorry

end scrambles_equal_two_twos_l606_606900


namespace median_and_std_dev_of_transformed_data_l606_606721

-- Assume data set and its properties
variables {n : ℕ} (x : Fin n → ℝ) (a s : ℝ)

-- Definitions for median and variance
def is_median (m : ℝ) : Prop :=
  ∃ (sorted_x : Fin n → ℝ), 
  (sorted_x = sort x) ∧ 
  ((n % 2 = 1 → m = sorted_x (n / 2)) ∧ (n % 2 = 0 → m = ((sorted_x (n / 2 - 1) + sorted_x (n / 2)) / 2)))

def variance (v : ℝ) : Prop :=
  (1 / n • ∑ i, (x i - (1 / n • ∑ j, x j)) ^ 2) = v

-- Transformed data set
noncomputable def y (i : Fin n) : ℝ := 3 * x i + 5

-- Theorem statement
theorem median_and_std_dev_of_transformed_data 
  (h_median : is_median x a)
  (h_variance : variance x (s^2)) :
  is_median y (3 * a + 5) ∧ sqrt (variance y) = 3 * s :=
sorry

end median_and_std_dev_of_transformed_data_l606_606721


namespace k_plus_m_eq_27_l606_606321

theorem k_plus_m_eq_27 (k m : ℝ) (a b c : ℝ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a > 0) (h5 : b > 0) (h6 : c > 0)
  (h7 : a + b + c = 8) 
  (h8 : k = a * b + a * c + b * c) 
  (h9 : m = a * b * c) :
  k + m = 27 :=
by
  sorry

end k_plus_m_eq_27_l606_606321


namespace watch_cost_price_l606_606507

theorem watch_cost_price (CP : ℝ) (H1 : 0.90 * CP = CP - 0.10 * CP)
(H2 : 1.04 * CP = CP + 0.04 * CP)
(H3 : 1.04 * CP - 0.90 * CP = 168) : CP = 1200 := by
sorry

end watch_cost_price_l606_606507


namespace largest_prime_factor_sum_divisors_180_l606_606794

theorem largest_prime_factor_sum_divisors_180 :
  let N := ∑ d in (Finset.divisors 180), d in
  Nat.greatest_prime_factor N = 13 :=
by
  sorry

end largest_prime_factor_sum_divisors_180_l606_606794


namespace find_integer_pairs_l606_606988

theorem find_integer_pairs (a b : ℕ) :
  (∃ (p : ℕ) (k : ℕ), prime p ∧ a^2 + b + 1 = p^k) ∧
  (a^2 + b + 1 ∣ b^2 - a^3 - 1) ∧
  ¬ (a^2 + b + 1 ∣ (a + b - 1)^2)
  → ∃ (x : ℕ), x ≥ 2 ∧ a = 2^x ∧ b = 2^(2*x) - 1 :=
sorry

end find_integer_pairs_l606_606988


namespace largest_prime_factor_of_sum_of_divisors_of_180_l606_606783

-- Define the function to compute the sum of divisors
noncomputable def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ k in (Finset.range (n + 1)).filter (λ k, n % k = 0), k

-- Define a function to find the largest prime factor of a number
noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  Finset.max' (Finset.filter Nat.prime (Finset.range (n + 1))) sorry

-- Define the problem conditions
def N : ℕ := sum_of_divisors 180

-- State the main theorem to be proved
theorem largest_prime_factor_of_sum_of_divisors_of_180 : largest_prime_factor N = 13 :=
by sorry

end largest_prime_factor_of_sum_of_divisors_of_180_l606_606783


namespace inradius_of_triangle_l606_606255

noncomputable def area_triangle (a b c : ℝ) (s : ℝ) (r : ℝ) : Prop :=
  s = (a + b + c) / 2 ∧
  2 * s * r = a * b * c

theorem inradius_of_triangle 
  (a b c : ℝ) (area : ℝ) (radius : ℝ)
  (h1 : a = 30) (h2 : b = 21) (h3 : c = 15) (h4 : area = 77)
  (s : ℝ) (r : ℝ) :
  area_triangle a b c s r → 
  r = 77 / 33 := 
by {
  -- Given conditions
  assume area_triangle,
  
  -- Sorry to skip the proof.
  sorry
}

end inradius_of_triangle_l606_606255


namespace least_integer_greater_than_sqrt_450_l606_606474

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606474


namespace least_integer_greater_than_sqrt_450_l606_606458

-- Define that 21^2 is less than 450
def square_of_21_less_than_450 : Prop := 21^2 < 450

-- Define that 450 is less than 22^2
def square_of_22_greater_than_450 : Prop := 450 < 22^2

-- State the theorem that 22 is the smallest integer greater than sqrt(450)
theorem least_integer_greater_than_sqrt_450 
  (h21 : square_of_21_less_than_450) 
  (h22 : square_of_22_greater_than_450) : 
    ∃ n : ℕ, n = 22 ∧ (√450 : ℝ) < n := 
sorry

end least_integer_greater_than_sqrt_450_l606_606458


namespace transformation_of_95_squared_l606_606882

theorem transformation_of_95_squared :
  (9.5 : ℝ) ^ 2 = (10 : ℝ) ^ 2 - 2 * (10 : ℝ) * (0.5 : ℝ) + (0.5 : ℝ) ^ 2 :=
by
  sorry

end transformation_of_95_squared_l606_606882


namespace midpoint_of_AB_l606_606955

-- Define the conditions of the problem
variables (A B C D E P : Type)
variables [Geometry A B C D E P]

-- Define the given conditions in the math problem
variables (AD DC DP EP BE EC : ℝ)
variables (h1 : AD = DC)
variables (h2 : DP = EP)
variables (h3 : BE = EC)
variables (h4 : ∠ADC = 90)
variables (h5 : ∠DPE = 90)
variables (h6 : ∠BEC = 90)

-- State what needs to be proven
theorem midpoint_of_AB (A B P : Point) : midpoint A B P :=
sorry

end midpoint_of_AB_l606_606955


namespace isosceles_triangle_base_angle_l606_606281

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle_isosceles : a = b ∨ b = c ∨ c = a)
  (h_angle_sum : a + b + c = 180) (h_one_angle : a = 50 ∨ b = 50 ∨ c = 50) :
  a = 50 ∨ b = 50 ∨ c = 50 ∨ a = 65 ∨ b = 65 ∨ c = 65 :=
by
  sorry

end isosceles_triangle_base_angle_l606_606281


namespace product_geometric_seq_l606_606813

   variable (a : ℕ → ℝ) (q : ℝ)
   variable (pos_terms : ∀ n, a n > 0)
   variable (sum_condition : a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 1)
   variable (reciprocal_sum_condition : (1 / a 0) + (1 / a 1) + (1 / a 2) + (1 / a 3) + (1 / a 4) + (1 / a 5) = 10)
   variable (geometric_seq : ∀ n, a (n + 1) = a n * q)

   theorem product_geometric_seq : a 0 * a 1 * a 2 * a 3 * a 4 * a 5 = 10⁻³ :=
   by
      sorry
   
end product_geometric_seq_l606_606813


namespace least_integer_gt_sqrt_450_l606_606463

theorem least_integer_gt_sqrt_450 : 
  ∀ n : ℕ, (∃ m : ℕ, m * m ≤ 450 ∧ 450 < (m + 1) * (m + 1)) → n = 22 :=
by
  sorry

end least_integer_gt_sqrt_450_l606_606463


namespace inequality_proof_l606_606141

theorem inequality_proof (n : ℕ) (a : fin n → ℝ) (h : ∀ i, 0 < a i) :
  (∑ i, a i) * (∑ i, 1 / (a i)) ≥ n ^ 2 := 
sorry

end inequality_proof_l606_606141


namespace geometric_arithmetic_sequences_sum_l606_606748

theorem geometric_arithmetic_sequences_sum (a b : ℕ → ℝ) (S_n : ℕ → ℝ) 
  (q d : ℝ) (h1 : 0 < q) 
  (h2 : a 1 = 1) (h3 : b 1 = 1) 
  (h4 : a 5 + b 3 = 21) 
  (h5 : a 3 + b 5 = 13) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, b n = 2*n - 1) ∧ (∀ n, S_n n = 3 - (2*n + 3)/(2^n)) := 
sorry

end geometric_arithmetic_sequences_sum_l606_606748


namespace boat_meeting_point_l606_606003

/-- Alex launches his boat from point 0, and Alice launches her boat from point 8 miles upstream. 
    Both boats move at 6 miles per hour in still water; the river flows downstream at 2.3 miles per hour. 
    Given these conditions, Alex and Alice meet at a distance of 37/15 miles from Alex's starting point. 
    The task is to prove that the sum of the numerator and denominator of this fraction is 52. -/
theorem boat_meeting_point :
  let v_river := 23 / 10
  let v_Alex := 6
  let v_Alice := 6
  let v_Alex_effective := v_Alex - v_river
  let v_Alice_effective := v_Alice + v_river
  let d := 37 / 15
  let m := 37
  let n := 15
  in m + n = 52 := 
by {
  sorry
}

end boat_meeting_point_l606_606003


namespace median_duration_proof_l606_606863

noncomputable def median_duration : ℕ :=
let data := [28, 28, 50, 58, 100, 102, 115, 122, 140, 145, 155, 163, 165, 170, 180, 180, 180, 205, 210, 216, 240, 255] in
data.nth 9 == 145

theorem median_duration_proof : median_duration = 145 := by
  sorry

end median_duration_proof_l606_606863


namespace m_value_l606_606263

noncomputable def solve_m : ℝ :=
  let z := (1 - (m : ℂ) * complex.I) / (1 - 2 * complex.I) in
  if (z.re + z.im = 0) then m else 0

theorem m_value (m : ℝ) : 
  let z := (1 - (m : ℂ) * complex.I) / (1 - 2 * complex.I) in
  (z.re + z.im = 0) → m = -3 :=
begin
  intro h,
  sorry
end

end m_value_l606_606263


namespace least_integer_greater_than_sqrt_450_l606_606485

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, n > real.sqrt 450 ∧ ∀ m : ℕ, m > real.sqrt 450 → n ≤ m :=
begin
  use 22,
  split,
  { sorry },
  { sorry }
end

end least_integer_greater_than_sqrt_450_l606_606485


namespace either_framed_by_jack_or_taken_by_octavia_or_sam_l606_606275

noncomputable def total_photos_framed_by_jack (jack_octavia : ℕ) (jack_sam : ℕ) (jack_alice : ℕ) :=
  jack_octavia + jack_sam + jack_alice

noncomputable def total_photos_taken (octavia : ℕ) (sam : ℕ) :=
  octavia + sam

noncomputable def photos_framed_by_jack_and_taken_by_octavia_or_sam (jack_octavia : ℕ) (jack_sam : ℕ) :=
  jack_octavia + jack_sam

theorem either_framed_by_jack_or_taken_by_octavia_or_sam
  (jack_octavia : ℕ) (jack_sam : ℕ) (jack_alice : ℕ)
  (octavia : ℕ) (sam : ℕ) :
  jack_octavia = 24 ∧ jack_sam = 12 ∧ jack_alice = 8 ∧ octavia = 36 ∧ sam = 20 →
  let framed_by_jack := total_photos_framed_by_jack jack_octavia jack_sam jack_alice,
      taken_by_octavia_or_sam := total_photos_taken octavia sam,
      by_both := photos_framed_by_jack_and_taken_by_octavia_or_sam jack_octavia jack_sam
  in framed_by_jack + taken_by_octavia_or_sam - by_both = 100 :=
by intros;
   rw [h.1, h.2, h.3, h.4, h.5];
   simp [←total_photos_framed_by_jack, ←total_photos_taken, ←photos_framed_by_jack_and_taken_by_octavia_or_sam];
   exact dec_trivial

end either_framed_by_jack_or_taken_by_octavia_or_sam_l606_606275


namespace sum_powers_zero_for_all_k_l606_606124

variable (n : ℕ) (C : ℕ → ℂ)
variable (H : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ∑ i in Finset.range n, (C i)^k = 0)

theorem sum_powers_zero_for_all_k :
  ∀ k : ℕ, 0 < k → ∑ i in Finset.range n, (C i)^k = 0 :=
by
  assume k : ℕ,
  assume hk : 0 < k,
  sorry

end sum_powers_zero_for_all_k_l606_606124


namespace always_passes_through_fixed_point_l606_606869

noncomputable def fixed_point_of_bisector (O1 O2 : Point) (r : ℝ) : Point := sorry

theorem always_passes_through_fixed_point
  (C1 C2 : Circle) (O1 O2 : Point) (r : ℝ)
  (hC1 : C1.center = O1 ∧ C1.radius = r)
  (hC2 : C2.center = O2 ∧ C2.radius = r)
  (Δ : Triangle)
  (A B C : Point)
  (hGrayTouch : ∃ X ∈ C1, A ≠ X ∧ OnSegment A X B)
  (hBlackTouch : ∃ Y ∈ C2, B ≠ Y ∧ OnSegment B Y C)
  (hColoring : gray_side Δ A B ∧ black_side Δ B C) :
  ∃ K : Point, ∀ Δ' A' B' C',
  gray_side Δ' A' B' ∧ black_side Δ' B' C' →
  ∃ O1' O2' r',
  C1'.center = O1' ∧ C1'.radius = r' ∧
  C2'.center = O2' ∧ C2'.radius = r' ∧
  touches_circle Δ' A' O1' ∧ touches_circle Δ' B' O2' →
  (bisector_line A' B' C').contains K := sorry

end always_passes_through_fixed_point_l606_606869


namespace sum_of_possible_b_l606_606866

theorem sum_of_possible_b (b : ℤ) (h : ∃ r s : ℤ, g(x) = x^2 - b * x + 3 * b ∧ r ≠ s ∧ r * s = 3 * b ∧ r + s = b)
    (Hroots : ∀ r s : ℤ, r ≠ s ∧ r * s = 3 * b ∧ r + s = b →
      ∃ (k : ℤ), b^2 - 12 * b = k^2 := by sorry
    ) : 
    let possible_b := { b | ∃ (r s: ℤ), r ≠ s ∧ r * s = 3 * b ∧ r + s = b } in
      (∑ bₖ in possible_b.to_finset, bₖ) = 28 :=
by sorry

end sum_of_possible_b_l606_606866


namespace largest_prime_factor_of_divisors_sum_180_l606_606790

def divisors_sum (n : ℕ) : ℕ :=
  (divisors n).sum

def largest_prime_factor (n : ℕ) : ℕ :=
  (factors n).filter prime ∣> ∧ .toList.maximum' sorry -- assume there is a maximum

theorem largest_prime_factor_of_divisors_sum_180 :
  ∃ N, N = divisors_sum 180 ∧ largest_prime_factor N = 13 := by
  sorry

end largest_prime_factor_of_divisors_sum_180_l606_606790


namespace main_seat_ticket_cost_l606_606917

/-- A concert sells out a 20,000 seat arena where back seat tickets cost $45 and 14,500 such tickets are sold. The concert made $955,000 in total. Prove that the main seat tickets cost $55 each. -/

theorem main_seat_ticket_cost :
  ∀ (total_seats back_seat_cost total_revenue back_seat_tickets_sold main_seat_tickets_sold main_seat_tickets_revenue main_seat_ticket_cost : ℝ),
  total_seats = 20000 →
  back_seat_cost = 45 →
  total_revenue = 955000 →
  back_seat_tickets_sold = 14500 →
  main_seat_tickets_sold = total_seats - back_seat_tickets_sold →
  main_seat_tickets_revenue = total_revenue - (back_seat_tickets_sold * back_seat_cost) →
  main_seat_ticket_cost = main_seat_tickets_revenue / main_seat_tickets_sold →
  main_seat_ticket_cost = 55 :=
by
  intros total_seats back_seat_cost total_revenue back_seat_tickets_sold main_seat_tickets_sold main_seat_tickets_revenue main_seat_ticket_cost
  assume h1 h2 h3 h4 h5 h6 h7
  sorry

end main_seat_ticket_cost_l606_606917


namespace problem_integer_B_values_count_l606_606114

def B (n : ℕ) : ℝ :=
  ∫ (x : ℝ) in (2 : ℝ)..(n : ℝ), (x * ⌈real.sqrt x⌉₊)

lemma integer_B_values_count : n ≥ 2 ∧ n ≤ 2000 ∧ integer_B n :=
  sorry

def integer_B (n : ℕ) : Prop :=
  B n ∈ ℚ

theorem problem_integer_B_values_count : 
  ∃ (count : ℕ), count = 850 ∧ ∀ n, (2 ≤ n ∧ n ≤ 2000) → integer_B n :=
begin
  use 850,
  intros n hn,
  sorry
end

end problem_integer_B_values_count_l606_606114


namespace inequality_proof_l606_606530

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606530


namespace recurring_decimal_sum_l606_606077

theorem recurring_decimal_sum :
  (0.666666...:ℚ) + (0.222222...:ℚ) - (0.444444...:ℚ) = 4 / 9 :=
begin
  sorry
end

end recurring_decimal_sum_l606_606077


namespace binomial_coeff_sum_l606_606864

-- Define the problem: compute the numerical sum of the binomial coefficients
theorem binomial_coeff_sum (a b : ℕ) (h_a1 : a = 1) (h_b1 : b = 1) : 
  (a + b) ^ 8 = 256 :=
by
  -- Therefore, the sum must be 256
  sorry

end binomial_coeff_sum_l606_606864


namespace prime_factorization_sum_l606_606804

theorem prime_factorization_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : 13 * x^7 = 17 * y^11) : 
  a * e + b * f = 18 :=
by
  -- Let a and b be prime factors of x
  let a : ℕ := 17 -- prime factor found in the solution
  let e : ℕ := 1 -- exponent found for 17
  let b : ℕ := 0 -- no second prime factor
  let f : ℕ := 0 -- corresponding exponent

  sorry

end prime_factorization_sum_l606_606804


namespace complex_solution_l606_606204

theorem complex_solution (z : ℂ) (h : ((0 : ℝ) + 1 * z = 4 + 3 * (complex.I))) : 
  z = 3 - 4 * (complex.I) :=
sorry

end complex_solution_l606_606204


namespace total_dividend_is_840_l606_606506

-- Define the initial investment
def investment : ℝ := 14400

-- Define the nominal value of each share
def nominal_share_value : ℝ := 100

-- Define the premium on each share in percentage
def premium : ℝ := 20

-- Define the dividend rate in percentage
def dividend_rate : ℝ := 7

-- Compute the cost per share considering the premium
def cost_per_share : ℝ := nominal_share_value + (premium / 100) * nominal_share_value

-- Compute the number of shares bought
def number_of_shares : ℝ := investment / cost_per_share

-- Compute the dividend per share
def dividend_per_share : ℝ := (dividend_rate / 100) * nominal_share_value

-- Compute the total dividend received
def total_dividend : ℝ := number_of_shares * dividend_per_share

-- Theorem statement: The total dividend received is Rs. 840
theorem total_dividend_is_840 : total_dividend = 840 := by
  sorry

end total_dividend_is_840_l606_606506


namespace solve_for_z_l606_606235

variable {z : ℂ}

theorem solve_for_z (h : complex.I * z = 4 + 3*complex.I) : z = 3 - 4*complex.I :=
sorry

end solve_for_z_l606_606235


namespace repeating_decimals_sum_l606_606070

theorem repeating_decimals_sum :
  let x := 0.6666666 -- 0.\overline{6}
  let y := 0.2222222 -- 0.\overline{2}
  let z := 0.4444444 -- 0.\overline{4}
  (x + y - z) = 4 / 9 := 
by
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  calc
    -- Calculate (x + y - z)
    (x + y - z) = (2 / 3 + 2 / 9 - 4 / 9) : by sorry
                ... = 4 / 9 : by sorry


end repeating_decimals_sum_l606_606070


namespace cakes_and_bread_weight_l606_606403

theorem cakes_and_bread_weight 
  (B : ℕ)
  (cake_weight : ℕ := B + 100)
  (h1 : 4 * cake_weight = 800)
  : 3 * cake_weight + 5 * B = 1100 := by
  sorry

end cakes_and_bread_weight_l606_606403


namespace common_fraction_l606_606089

noncomputable def x : ℚ := 0.6666666 -- represents 0.\overline{6}
noncomputable def y : ℚ := 0.2222222 -- represents 0.\overline{2}
noncomputable def z : ℚ := 0.4444444 -- represents 0.\overline{4}

theorem common_fraction :
  x + y - z = 4 / 9 :=
by
  -- Provide proofs here
  sorry

end common_fraction_l606_606089


namespace b_mod_9_l606_606803

-- Define conditions
variable (n : ℕ) (n_pos : 0 < n)
def b : ℤ := (3^(2*n + 1) + 5)⁻¹ % 9

-- Proof statement
theorem b_mod_9 : b n n_pos ≡ 2 [MOD 9] :=
sorry

end b_mod_9_l606_606803


namespace maximum_ab_value_l606_606689

noncomputable def maximum_ab (a b : ℝ) :=
  if a > 0 ∧ b > 0 ∧ a + 2 * b = 2 then ab else 0

theorem maximum_ab_value : ∀ (a b : ℝ), 
  (a > 0 ∧ b > 0 ∧ a + 2 * b = 2) → ab = 1/2 :=
begin
  sorry
end

end maximum_ab_value_l606_606689


namespace matrix_power_three_scaling_condition_l606_606809

variables {B : Matrix (Fin 2) (Fin 2) ℝ}
variables {v : Vector ℝ 2} {v' : Vector ℝ 2}
variables (three : ℝ := 3)

theorem matrix_power_three_scaling_condition (h : B • ![4, -3] = ![12, -9])
: (B ^ 3) • ![4, -3] = ![108, -81] :=
sorry

end matrix_power_three_scaling_condition_l606_606809


namespace polynomial_quotient_l606_606110

-- Definitions of the polynomials
def numerator := (5 * X^4) - (3 * X^3) + (4 * X^2) - (8 * X) + 3
def denominator := (X^2) + (2 * X) + 1
def expected_quotient := (5 * X^2) - (13 * X) + 21

-- The theorem statement
theorem polynomial_quotient :
  ∀ (z : ℝ), (numerator) / (denominator) = expected_quotient := 
begin
  sorry
end

end polynomial_quotient_l606_606110


namespace find_x_to_make_whole_number_l606_606659

noncomputable def n : ℕ := 216

theorem find_x_to_make_whole_number (x : ℝ) (h_whole : (Real.log n / Real.log 3) + (Real.log n / Real.log x) ∈ ℤ) : x = 6 :=
by
  sorry

end find_x_to_make_whole_number_l606_606659


namespace probability_YD_gt_6sqrt2_l606_606413

noncomputable def right_triangle (XY : ℝ) (angle_YXZ : ℝ) (angle_XYZ : ℝ) := 
  angle_XYZ = 90 ∧ angle_YXZ = 60 ∧ XY = 12

def calculate_probability (XY : ℝ) (YZ : ℝ) (XZ : ℝ) (YD : ℝ) : ℝ :=
  if YD > 6 * Real.sqrt 2 then 
    (XZ - 6) / XZ 
  else 
    0

theorem probability_YD_gt_6sqrt2 (XY YZ XZ : ℝ) : 
  right_triangle XY (60) (90) → 
  calculate_probability XY 6 (6 * Real.sqrt 3) (6 * Real.sqrt 2) = (3 - Real.sqrt 3) / 3 :=
by
  intros h
  sorry

end probability_YD_gt_6sqrt2_l606_606413


namespace cylindrical_to_rectangular_l606_606043

open real

theorem cylindrical_to_rectangular (r θ z x y : ℝ) (h₀ : r = 7) (h₁ : θ = π / 3) (h₂ : z = -3) 
(h₃ : x = r * cos θ) (h₄ : y = r * sin θ) :
  (x, y, z) = (3.5, (7 * sqrt 3) / 2, -3) :=
by
  rw [h₀, h₁, h₂, h₃, h₄]
  sorry

end cylindrical_to_rectangular_l606_606043


namespace find_m_l606_606618

def f (x : ℝ) := 6 * x^3 - (3 / x) + 7
def g (x : ℝ) (m : ℝ) := 3 * x^2 - 2 * x - m

theorem find_m : (f 3 - g 3 m = 5) → m = -142 :=
by
  intro h
  sorry

end find_m_l606_606618


namespace find_z_l606_606244

theorem find_z (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : z = 3 - 4 * complex.I :=
by {
  sorry
}

end find_z_l606_606244


namespace solve_for_z_l606_606240

variable {z : ℂ}

theorem solve_for_z (h : complex.I * z = 4 + 3*complex.I) : z = 3 - 4*complex.I :=
sorry

end solve_for_z_l606_606240


namespace cone_lateral_surface_area_l606_606847

theorem cone_lateral_surface_area
  (r : ℝ) (h : ℝ) (π : ℝ) [Real_radical_comm_ring π]
  (radius_cond : r = 3)
  (height_cond : h = 4)
  (pi_cond : π = real.pi) :
  let slant_height := sqrt (r^2 + h^2) in
  let circumference := 2 * real.pi * r in
  let lateral_surface_area := 0.5 * circumference * slant_height in
  lateral_surface_area = 15 * real.pi := 
 by
  {
    -- Here we add the proof steps if necessary
    sorry
  }

end cone_lateral_surface_area_l606_606847


namespace sum_of_perpendiculars_constant_l606_606956

theorem sum_of_perpendiculars_constant (ABC : Triangle) (h k l : ℝ) 
  (H K L : Point) (P : Point) (s : ℝ)
  (hABC : equilateral ABC) (hP_inside : inside_triangle P ABC)
  (hH : foot_of_perpendicular P ABC.side_ab H) 
  (hK : foot_of_perpendicular P ABC.side_bc K) 
  (hL : foot_of_perpendicular P ABC.side_ca L) 
  (h_equation : area ABC = (√3 / 4) * s^2) 
  (h_area : area(ABC) = (1 / 2) * s * (h + k + l)) :
  h + k + l = (2 * area(ABC)) / s :=
sorry

end sum_of_perpendiculars_constant_l606_606956


namespace move_points_arbitrarily_far_l606_606885

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem move_points_arbitrarily_far 
  (k : ℝ) (N : ℕ) (hN : N > 1) (h_pos : 0 < k) 
  (initial_points : Fin N → ℝ) 
  (h_distinct : ∃ (i j : Fin N), i ≠ j ∧ initial_points i ≠ initial_points j) :
  (∃ f : ℕ → (Fin N → ℝ), 
    (∀ n, ∃ i j, i ≠ j ∧ f n (i) < f n (j) ∧ 
                f (n + 1) = λ x, if x = j then f n (i) + k * (f n (i) - f n (j)) else f n x) ∧
    (∀ m : ℕ, ∀ i : Fin N, f m i ≤ f (m + 1) i) ∧
    (∀ M : ℝ, ∃ m : ℕ, ∀ i : Fin N, f m i > M)) ↔ k ≥ 1 / (N - 1) :=
sorry

end move_points_arbitrarily_far_l606_606885


namespace monotonic_intervals_k_range_l606_606163

variable {x : ℝ}
variable {k : ℝ}

def f (x : ℝ) : ℝ := x^2 / Real.exp x

theorem monotonic_intervals :
  (∀ x, 0 < x ∧ x < 2 → (f x) > (f (x - ε))) ∧ 
  (∀ x, x ≤ 0 ∨ x ≥ 2 → (f x) < (f (x + ε))) :=
sorry

theorem k_range (x : ℝ) (h : 1 < x) :
  f x + k * (1 + Real.log x) ≤ 0 → k ≤ -1 / Real.exp 1 :=
sorry

end monotonic_intervals_k_range_l606_606163


namespace repeating_decimal_sum_l606_606062

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem repeating_decimal_sum : repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l606_606062


namespace f_f_f_three_l606_606771

def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 1

theorem f_f_f_three : f (f (f 3)) = 43 :=
by
  -- Introduction of definitions and further necessary steps here are skipped
  sorry

end f_f_f_three_l606_606771


namespace inequality_proof_l606_606549

variable {x y z : ℝ}

theorem inequality_proof (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * real.sqrt(z * x)) + 
  y^3 / (y^3 + 2 * z^2 * real.sqrt(x * y)) + 
  z^3 / (z^3 + 2 * x^2 * real.sqrt(y * z)) >= 1) :=
sorry

end inequality_proof_l606_606549


namespace incorrect_statement_D_l606_606393

noncomputable theory

def f (x : ℝ) := 2 * Real.sin (3 * x + Real.pi / 6)

theorem incorrect_statement_D :
  (∃ x : ℝ, x ∈ Set.Icc (-(Real.pi / 9)) (Real.pi / 9) ∧ ∀ y : ℝ, y ∈ Set.Icc (-(Real.pi / 9)) (Real.pi / 9) → f y ≥ f x) = false :=
by
  sorry

end incorrect_statement_D_l606_606393


namespace centers_of_internal_squares_are_midpoints_of_ABC_l606_606819

open EuclideanGeometry

/-- Definition of the input triangle and centers of squares constructed on its sides --/
variables {A B C P Q R : Point}

/-- Given conditions: construction of triangles and squares as described. --/
axiom construction_of_squares_on_ABC : externally_constructed_square_on_side A B P ∧ externally_constructed_square_on_side B C Q ∧ externally_constructed_square_on_side C A R
axiom internal_squares_on_PQR : internally_constructed_square_on_side P Q ∧ internally_constructed_square_on_side Q R ∧ internally_constructed_square_on_side R P

theorem centers_of_internal_squares_are_midpoints_of_ABC (A B C P Q R : Point)
  (h1 : externally_constructed_square_on_side A B P)
  (h2 : externally_constructed_square_on_side B C Q)
  (h3 : externally_constructed_square_on_side C A R)
  (h4 : internally_constructed_square_on_side P Q)
  (h5 : internally_constructed_square_on_side Q R)
  (h6 : internally_constructed_square_on_side R P) :
  (midpoint P R = midpoint B C) ∧ (midpoint Q R = midpoint C A) ∧ (midpoint P Q = midpoint A B) :=
begin
  -- Proof would follow here
  sorry,
end

end centers_of_internal_squares_are_midpoints_of_ABC_l606_606819


namespace sequence_value_of_m_l606_606755

theorem sequence_value_of_m (a : ℕ → ℝ) (m : ℕ) (h1 : a 1 = 1)
                            (h2 : ∀ n : ℕ, n > 0 → a n - a (n + 1) = a (n + 1) * a n)
                            (h3 : 8 * a m = 1) :
                            m = 8 := by
  sorry

end sequence_value_of_m_l606_606755


namespace least_integer_greater_than_sqrt_450_l606_606489

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℕ, n > real.sqrt 450 ∧ ∀ m : ℕ, m > real.sqrt 450 → n ≤ m :=
begin
  use 22,
  split,
  { sorry },
  { sorry }
end

end least_integer_greater_than_sqrt_450_l606_606489


namespace bridge_length_is_235_l606_606397

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * time_sec
  let bridge_length := total_distance - train_length
  bridge_length

theorem bridge_length_is_235 :
  length_of_bridge 140 45 30 = 235 :=
by 
  sorry

end bridge_length_is_235_l606_606397


namespace even_function_xf_l606_606198

-- Define odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

-- Main theorem statement
theorem even_function_xf (f : ℝ → ℝ) (h : is_odd_function f) : 
∀ x : ℝ, (x * f x) = (x * f x) :=
begin
  sorry,
end

end even_function_xf_l606_606198


namespace parametric_curve_length_theorem_l606_606996

noncomputable def parametric_curve_length : ℝ :=
  ∫ t in 0..2*Real.pi, Real.sqrt (9 * (Real.sin t)^2 + 4 * (Real.cos t)^2)

theorem parametric_curve_length_theorem :
  parametric_curve_length = ∫ t in 0..2*Real.pi, Real.sqrt (9 * (Real.sin t)^2 + 4 * (Real.cos t)^2) :=
by
  sorry

end parametric_curve_length_theorem_l606_606996


namespace least_integer_gt_sqrt_450_l606_606444

theorem least_integer_gt_sqrt_450 : 
  let x := 450 in
  (21 * 21 = 441) → 
  (22 * 22 = 484) →
  (441 < x ∧ x < 484) →
  (∃ n : ℤ, n = 22 ∧ n > int.sqrt x) :=
by
  intros h1 h2 h3
  use 22
  split
  · refl
  · sorry

end least_integer_gt_sqrt_450_l606_606444


namespace exists_k_for_composite_expression_l606_606377

theorem exists_k_for_composite_expression :
  ∃ k : ℕ, (∀ n : ℕ, 0 < n -> Nat.composite (k * 2^n + 1)) := sorry

end exists_k_for_composite_expression_l606_606377


namespace max_sides_regular_prism_l606_606396

-- Definitions
def regular_prism (n : ℕ) (lateral_edge_length base_side_length : ℝ) : Prop := 
  -- regular prism condition: equal lateral edge and base side length
  lateral_edge_length = base_side_length ∧ n > 2

-- Property to be proven
theorem max_sides_regular_prism (n : ℕ) (lateral_edge_length base_side_length : ℝ) :
  regular_prism n lateral_edge_length base_side_length → n ≤ 5 := 
by
  -- Proof placeholder
  intros h,
  sorry


end max_sides_regular_prism_l606_606396


namespace greatest_c_not_in_range_l606_606425

theorem greatest_c_not_in_range {c : ℤ}
  (h : ∀ x : ℝ, x^2 + c * x + 20 ≠ -9) :
  c ≤ 10 :=
begin
  sorry
end

end greatest_c_not_in_range_l606_606425


namespace wire_length_l606_606598

noncomputable def max_area : ℝ := 154.0619849129547

theorem wire_length (C : ℝ) (h : 2 * (Real.sqrt (max_area * π)) = C) : C ≈ 44 :=
by sorry

end wire_length_l606_606598


namespace count_3_digit_numbers_divisible_by_11_l606_606194

/-- 
  Define the mathematical conditions.
-/
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def is_divisible_by (n m : ℕ) : Prop := n % m = 0

/--
  Define the problem statement to prove that the count of 3-digit numbers divisible by 11 is 82.
-/
theorem count_3_digit_numbers_divisible_by_11 : 
  let nums := { n | smallest_3_digit ≤ n ∧ n ≤ largest_3_digit ∧ is_divisible_by n 11},
      count := nums.card
  in count = 82 := by
{
  sorry
}

end count_3_digit_numbers_divisible_by_11_l606_606194


namespace isosceles_triangle_from_median_and_cyclic_quadrilateral_l606_606360

theorem isosceles_triangle_from_median_and_cyclic_quadrilateral
    {A B C M K A1 B1 : Type*}
    (hCM_med : IsMedian A B C M)
    (hK_on_CM : OnLine K CM)
    (hAK_int_BC : IsIntersection AK BC A1)
    (hBK_int_AC : IsIntersection BK AC B1)
    (quad_cyclic : CyclicQuadrilateral A B1 A1 B) 
    : IsIsoscelesTriangle A B C :=
by
  sorry

end isosceles_triangle_from_median_and_cyclic_quadrilateral_l606_606360


namespace even_extremeña_faces_l606_606004

def vertex_color := {c // c = "green" ∨ c = "white" ∨ c = "black"}

structure face :=
(vertices : fin 3 → vertex_color)

structure polyhedron :=
(faces : fin n → face)
(h_triangular_faces : ∀ (f : fin n), ∃ (v₁ v₂ v₃ : vertex_color), faces f = ⟨[v₁, v₂, v₃]⟩)

def is_extremeña (f : face) : Prop :=
∃ (v₁ v₂ v₃ : vertex_color),
  v₁ ≠ v₂ ∧ v₂ ≠ v₃ ∧ v₁ ≠ v₃ ∧ 
  (v₁ = "green" ∨ v₁ = "white" ∨ v₁ = "black") ∧
  (v₂ = "green" ∨ v₂ = "white" ∨ v₂ = "black") ∧
  (v₃ = "green" ∨ v₃ = "white" ∨ v₃ = "black")

theorem even_extremeña_faces (poly : polyhedron) : 
  (∃ (n : ℕ), ∀ (i : fin n), is_extremeña (poly.faces i)) → sorry :=
begin
  -- proof here
end

end even_extremeña_faces_l606_606004


namespace least_integer_greater_than_sqrt_450_l606_606437

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606437


namespace area_of_triangle_l606_606416

theorem area_of_triangle : 
  let A := (1, 3)
  let B := (9, -1 / 2)
  let C := (3, 5)
  let area := (1 / 2 : ℚ) * ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))).nat_abs
  area = 14 :=
by
  -- Definitions for the points of intersection extracted from the conditions and the calculated points.
  -- Direct usage of conditions without assuming solution steps ensures equivalence.
  let A := (1 : ℚ, 3 : ℚ)
  let B := (9 : ℚ, -1 / 2)
  let C := (3 : ℚ, 5 : ℚ)
  -- Using Lean's nat_abs for absolute value since we're dealing with simulation of integer operations
  let area := (1 / 2 : ℚ) * ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))).nat_abs
  -- Insert placeholder for proof
  sorry

end area_of_triangle_l606_606416


namespace average_speed_round_trip_l606_606302

def time_to_walk_uphill := 30 -- in minutes
def time_to_walk_downhill := 10 -- in minutes
def distance_one_way := 1 -- in km

theorem average_speed_round_trip :
  (2 * distance_one_way) / ((time_to_walk_uphill + time_to_walk_downhill) / 60) = 3 := by
  sorry

end average_speed_round_trip_l606_606302


namespace det_matrix_A_l606_606961

noncomputable def matrix_A (n : ℕ) : Matrix (Fin n) (Fin n) ℝ :=
  λ i j, if i = j then 2 else (-1) ^ (| ↑i - ↑j |)

theorem det_matrix_A (n : ℕ) :
  Matrix.det (matrix_A n) = n + 1 :=
sorry

end det_matrix_A_l606_606961


namespace arithmetic_sequence_fifth_term_l606_606389

-- Definitions and conditions are based on part a).
def term1 (x y : ℕ) : ℕ := x - y
def term2 (x y : ℕ) : ℕ := x
def term3 (x y : ℕ) : ℕ := x + y
def term4 (x y : ℕ) : ℕ := x + 2y

-- The statement to prove that the fifth term is x + 3y.
theorem arithmetic_sequence_fifth_term (x y : ℕ) : 
  let d := term2 x y - term1 x y in
  let a5 := term4 x y + d in
  a5 = x + 3y := 
by
  sorry

end arithmetic_sequence_fifth_term_l606_606389


namespace evaluate_expression_l606_606635

theorem evaluate_expression : 3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 :=
by
  intro x y h1 h2
  sorry

end evaluate_expression_l606_606635


namespace limit_cos_cot2x_sin3x_eq_l606_606024

noncomputable def limit_expression (x : ℝ) : ℝ := (Real.cos x) ^ ((Real.cot (2 * x)) / (Real.sin (3 * x)))

theorem limit_cos_cot2x_sin3x_eq : 
  tendsto (λ x : ℝ, limit_expression x) (𝓝[≠] (2 * Real.pi)) (𝓝 (Real.exp (-1 / 12))) := 
sorry

end limit_cos_cot2x_sin3x_eq_l606_606024


namespace B_finishes_in_10_days_l606_606564

noncomputable def B_remaining_work_days (A_work_days : ℕ := 15) (A_initial_days_worked : ℕ := 5) (B_work_days : ℝ := 14.999999999999996) : ℝ :=
  let A_rate := 1 / A_work_days
  let B_rate := 1 / B_work_days
  let remaining_work := 1 - (A_rate * A_initial_days_worked)
  let days_for_B := remaining_work / B_rate
  days_for_B

theorem B_finishes_in_10_days :
  B_remaining_work_days 15 5 14.999999999999996 = 10 :=
by
  sorry

end B_finishes_in_10_days_l606_606564


namespace solution_l606_606229

noncomputable def z : ℂ := 3 - 4i

theorem solution (z : ℂ) (h : i * z = 4 + 3 * i) : z = 3 - 4 * i :=
by sorry

end solution_l606_606229


namespace inequality_solution_equality_condition_l606_606309

theorem inequality_solution (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) (h3 : b < -1 ∨ b > 0) :
  (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b :=
sorry

theorem equality_condition (a b : ℝ) :
  (1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b :=
sorry

end inequality_solution_equality_condition_l606_606309


namespace triangle_is_isosceles_l606_606346

theorem triangle_is_isosceles 
  (A B C M K A_1 B_1 : Point)
  (h1 : is_median C M A B)
  (h2 : lies_on K C M)
  (h3 : line_intersects AK BC A_1)
  (h4 : line_intersects BK AC B_1)
  (h5 : is_cyclic_quadrilateral A B_1 A_1 B) 
  : is_isosceles_triangle A B C :=
sorry

end triangle_is_isosceles_l606_606346


namespace tenth_term_in_sequence_l606_606580

def seq (n : ℕ) : ℚ :=
  (-1) ^ (n + 1) * ((2 * n - 1) / (n ^ 2 + 1))

theorem tenth_term_in_sequence :
  seq 10 = -19 / 101 :=
by
  -- Proof omitted
  sorry

end tenth_term_in_sequence_l606_606580


namespace domain_of_g_l606_606623

theorem domain_of_g :
  (forall x : ℝ, g(x) = sqrt(2 - sqrt(5 - sqrt(4 - 2 * x))) -> x <= 3 / 2) :=
begin
  sorry
end

noncomputable def g (x : ℝ) : ℝ := sqrt(2 - sqrt(5 - sqrt(4 - 2 * x)))

end domain_of_g_l606_606623


namespace least_integer_gt_sqrt_450_l606_606450

theorem least_integer_gt_sqrt_450 : 
  let x := 450 in
  (21 * 21 = 441) → 
  (22 * 22 = 484) →
  (441 < x ∧ x < 484) →
  (∃ n : ℤ, n = 22 ∧ n > int.sqrt x) :=
by
  intros h1 h2 h3
  use 22
  split
  · refl
  · sorry

end least_integer_gt_sqrt_450_l606_606450


namespace smallest_absolute_value_36k_5l_l606_606952

theorem smallest_absolute_value_36k_5l : 
  ∃ k l : ℕ, abs (36^k - 5^l) = 11 ∧ 
  ∀ k' l' : ℕ, abs (36^k' - 5^l') ≥ 11 :=
sorry

end smallest_absolute_value_36k_5l_l606_606952


namespace largest_prime_factor_of_sum_of_divisors_of_180_l606_606782

-- Define the function to compute the sum of divisors
noncomputable def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ k in (Finset.range (n + 1)).filter (λ k, n % k = 0), k

-- Define a function to find the largest prime factor of a number
noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  Finset.max' (Finset.filter Nat.prime (Finset.range (n + 1))) sorry

-- Define the problem conditions
def N : ℕ := sum_of_divisors 180

-- State the main theorem to be proved
theorem largest_prime_factor_of_sum_of_divisors_of_180 : largest_prime_factor N = 13 :=
by sorry

end largest_prime_factor_of_sum_of_divisors_of_180_l606_606782


namespace floor_ceiling_sum_l606_606634

theorem floor_ceiling_sum : 
    Int.floor (0.998 : ℝ) + Int.ceil (2.002 : ℝ) = 3 := by
  sorry

end floor_ceiling_sum_l606_606634


namespace correct_front_view_l606_606102

def first_row : List ℕ := [1, 4, 3]
def second_row : List ℕ := [2, 1, 2]

def front_view (r1 r2 : List ℕ) : List ℕ :=
  List.map₂ max r1 r2

theorem correct_front_view :
  front_view first_row second_row = [2, 4, 3] :=
by 
  -- the proof goes here
  sorry

end correct_front_view_l606_606102


namespace solution_set_l606_606121

def f (x : ℝ) : ℝ :=
if x ≥ 0 then 1 else 0

theorem solution_set (x : ℝ) : xf(x) + x ≤ 2 ↔ x ≤ 1 :=
by sorry

end solution_set_l606_606121


namespace solutions_exist_l606_606987

theorem solutions_exist (a b n : ℤ) (ha : a > 1) (hb : b > 1) (hn : n > 1) :
  (a^3 + b^3)^n = 4 * (a * b)^1995 ↔
  (a, b, n) = (1, 1, 2) ∨ (a, b, n) = (2, 2, 998) ∨
  (a, b, n) = (32, 32, 1247) ∨ 
  (a, b, n) = (2^55, 2^55, 1322) ∨ 
  (a, b, n) = (2^221, 2^221, 1328) := 
sorry

end solutions_exist_l606_606987


namespace parabola_expression_l606_606722

theorem parabola_expression (a c : ℝ) (h1 : a = 1/4 ∨ a = -1/4) (h2 : ∀ x : ℝ, x = 1 → (a * x^2 + c = 0)) :
  (a = 1/4 ∧ c = -1/4) ∨ (a = -1/4 ∧ c = 1/4) :=
by {
  sorry
}

end parabola_expression_l606_606722


namespace find_area_of_third_polygon_l606_606409

noncomputable def area_of_third_polygon (S1 S2 : ℝ) : ℝ :=
  sqrt (2 * S2^3 / (S1 + S2))

theorem find_area_of_third_polygon (S1 S2 : ℝ) (hS1 : S1 > 0) (hS2 : S2 > 0) :
  ∃ S, S = sqrt (2 * S2^3 / (S1 + S2)) ∧ S = area_of_third_polygon S1 S2 :=
by
  use area_of_third_polygon S1 S2
  split
  . rfl
  . rfl

end find_area_of_third_polygon_l606_606409


namespace least_integer_greater_than_sqrt_450_l606_606438

theorem least_integer_greater_than_sqrt_450 : ∃ (n : ℤ), n = 22 ∧ (n > Real.sqrt 450) ∧ (∀ m : ℤ, m > Real.sqrt 450 → m ≥ n) :=
by
  sorry

end least_integer_greater_than_sqrt_450_l606_606438


namespace symmetric_points_ab_value_l606_606742

theorem symmetric_points_ab_value
  (a b : ℤ)
  (h₁ : a + 2 = -4)
  (h₂ : 2 = b) :
  a * b = -12 :=
by
  sorry

end symmetric_points_ab_value_l606_606742


namespace tangent_expression_l606_606152

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Conditions
axiom h1 : ∀ n, a n = a 0 + n * d  -- Arithmetic sequence with some common difference d
axiom h2 : ∀ n, b n = b 0 * (r ^ n)  -- Geometric sequence with some ratio r
axiom h3 : a 1000 + a 1018 = 2 * Real.pi
axiom h4 : b 6 * b 2012 = 2

-- Theorem to prove
theorem tangent_expression : tan ((a 2 + a 2016) / (1 + b 3 * b 2015)) = -Real.sqrt 3 :=
by
  sorry

end tangent_expression_l606_606152


namespace f_value_at_5pi_over_3_l606_606965

noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the even property of f
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Define the periodic property of f with period π
def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x : ℝ, f x = f (x + p)

-- Conditions
axiom f_even : even_function f
axiom f_periodic : periodic f π
axiom f_def : ∀ x ∈ Icc 0 (π / 2), f x = sin (x / 2)

-- The theorem to be proven
theorem f_value_at_5pi_over_3 : f (5 * π / 3) = 1 / 2 := sorry

end f_value_at_5pi_over_3_l606_606965


namespace stream_speed_l606_606581

theorem stream_speed (v : ℝ) (t : ℝ) (h1 : t > 0)
  (h2 : ∃ k : ℝ, k = 2 * t)
  (h3 : (9 + v) * t = (9 - v) * (2 * t)) :
  v = 3 := 
sorry

end stream_speed_l606_606581


namespace solve_for_z_l606_606223

variable (z : ℂ)

theorem solve_for_z (h : (complex.I * z = 4 + 3 * complex.I)) : z = 3 - 4 * complex.I := 
sorry

end solve_for_z_l606_606223


namespace common_fraction_l606_606088

noncomputable def x : ℚ := 0.6666666 -- represents 0.\overline{6}
noncomputable def y : ℚ := 0.2222222 -- represents 0.\overline{2}
noncomputable def z : ℚ := 0.4444444 -- represents 0.\overline{4}

theorem common_fraction :
  x + y - z = 4 / 9 :=
by
  -- Provide proofs here
  sorry

end common_fraction_l606_606088


namespace calculate_expression_l606_606609

theorem calculate_expression : (632^2 - 568^2 + 100) = 76900 :=
by sorry

end calculate_expression_l606_606609


namespace least_int_gt_sqrt_450_l606_606434

theorem least_int_gt_sqrt_450 : ∃ n : ℕ, n > nat.sqrt 450 ∧ (∀ m : ℕ, m > nat.sqrt 450 → n ≤ m) :=
by
  have h₁ : 20^2 = 400 := by norm_num
  have h₂ : 21^2 = 441 := by norm_num
  have h₃ : 22^2 = 484 := by norm_num
  have sqrt_bounds : 21 < real.sqrt 450 ∧ real.sqrt 450 < 22 :=
    by sorry
  use 22
  split
  · sorry
  · sorry

end least_int_gt_sqrt_450_l606_606434


namespace factor_expression_l606_606099

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) :=
by
  sorry

end factor_expression_l606_606099


namespace inequality_proof_l606_606536

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606536


namespace sum_of_g_36_l606_606801

def f (x : ℝ) : ℝ := 4*x^2 - 4

def g (y : ℝ) : ℝ := 
  let x := if y = 36 then sqrt 10 else -sqrt 10 in
  x^2 - x + 2

theorem sum_of_g_36 : g(36) = 24 :=
  sorry

end sum_of_g_36_l606_606801


namespace number_of_triangles_with_perimeter_eight_l606_606178

theorem number_of_triangles_with_perimeter_eight :
  { (a, b, c) : ℕ × ℕ × ℕ // a + b + c = 8 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b }.card = 5 :=
by
  -- Declaration of the proof structure
  sorry

end number_of_triangles_with_perimeter_eight_l606_606178


namespace number_of_women_l606_606561

-- Definitions for the given conditions
variables (m w : ℝ)
variable (x : ℝ)

-- Conditions
def cond1 : Prop := 3 * m + 8 * w = 6 * m + 2 * w
def cond2 : Prop := 4 * m + x * w = 0.9285714285714286 * (3 * m + 8 * w)

-- Theorem to prove the number of women in the third group (x)
theorem number_of_women (h1 : cond1 m w) (h2 : cond2 m w x) : x = 5 :=
sorry

end number_of_women_l606_606561


namespace triangle_barycenter_area_l606_606757

noncomputable def triangle_area (A B C : Point) : ℝ := sorry
noncomputable def area_of_N1N2N3 (ABC_area : ℝ) : ℝ := ABC_area / 64

theorem triangle_barycenter_area 
  (A B C D E F N1 N2 N3 : Point)
  (hD : on_segment D B C)
  (hE : on_segment E A C)
  (hF : on_segment F A B)
  (hCD : dist C D = (1/4) * dist B C)
  (hAE : dist A E = (1/4) * dist A C)
  (hBF : dist B F = (1/4) * dist A B)
  (hN1 : intersects (line_through A D) (line_through B C) N1)
  (hN2 : intersects (line_through B E) (line_through C A) N2)
  (hN3 : intersects (line_through C F) (line_through A B) N3) :
  triangle_area N1 N2 N3 = area_of_N1N2N3 (triangle_area A B C) := 
sorry

end triangle_barycenter_area_l606_606757


namespace least_int_gt_sqrt_450_l606_606427

theorem least_int_gt_sqrt_450 : ∃ n : ℕ, n > nat.sqrt 450 ∧ (∀ m : ℕ, m > nat.sqrt 450 → n ≤ m) :=
by
  have h₁ : 20^2 = 400 := by norm_num
  have h₂ : 21^2 = 441 := by norm_num
  have h₃ : 22^2 = 484 := by norm_num
  have sqrt_bounds : 21 < real.sqrt 450 ∧ real.sqrt 450 < 22 :=
    by sorry
  use 22
  split
  · sorry
  · sorry

end least_int_gt_sqrt_450_l606_606427


namespace inequality_proof_l606_606531

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 := 
by
  sorry

end inequality_proof_l606_606531


namespace average_square_feet_per_person_approx_l606_606858

def population : ℕ := 39512223
def area : ℕ := 163696
def square_feet_per_square_mile : ℕ := 5280 * 5280
def total_square_feet : ℕ := area * square_feet_per_square_mile
def average_square_feet_per_person : ℕ := total_square_feet / population

theorem average_square_feet_per_person_approx :
  average_square_feet_per_person ≈ 115491 :=
by sorry

end average_square_feet_per_person_approx_l606_606858


namespace least_int_gt_sqrt_450_l606_606428

theorem least_int_gt_sqrt_450 : ∃ n : ℕ, n > nat.sqrt 450 ∧ (∀ m : ℕ, m > nat.sqrt 450 → n ≤ m) :=
by
  have h₁ : 20^2 = 400 := by norm_num
  have h₂ : 21^2 = 441 := by norm_num
  have h₃ : 22^2 = 484 := by norm_num
  have sqrt_bounds : 21 < real.sqrt 450 ∧ real.sqrt 450 < 22 :=
    by sorry
  use 22
  split
  · sorry
  · sorry

end least_int_gt_sqrt_450_l606_606428


namespace find_perpendicular_line_through_P_l606_606388

def point := (ℝ × ℝ)
def line (a b c : ℝ) := ∀ (x y : ℝ), a * x + b * y + c = 0

-- Given conditions
def P : point := (1, -1)
def l (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- The statement to prove
theorem find_perpendicular_line_through_P : 
  ∃ (a b c : ℝ), line a b c ∧ (a ≠ 0 ∨ b ≠ 0) ∧ (∀ x y, l x y → a * x + b * y + c = 0) ∧ 
  ∀ (x y : ℝ), line 2 1 (-1) x y :=
by
  sorry

end find_perpendicular_line_through_P_l606_606388


namespace solve_complex_equation_l606_606214

theorem solve_complex_equation (z : ℂ) (h : (complex.I * z) = 4 + 3 * complex.I) : 
  z = 3 - 4 * complex.I :=
sorry

end solve_complex_equation_l606_606214


namespace sec_225_eq_neg_sqrt_2_l606_606051

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_225_eq_neg_sqrt_2 :
  sec (225 * Real.pi / 180) = -Real.sqrt 2 :=
by
  have cos_45 : Real.cos (45 * Real.pi / 180) = 1 / Real.sqrt 2 :=
    by sorry
  have cos_225 : Real.cos (225 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
    by
      rw [Real.cos_add_pi, cos_45]
      sorry
  show sec (225 * Real.pi / 180) = -Real.sqrt 2,
    by
      unfold sec
      rw [cos_225]
      sorry

end sec_225_eq_neg_sqrt_2_l606_606051


namespace cube_dot_path_length_l606_606572

/-- A cube with edges of length 2 cm has a dot marked in the center of the top face.
    The cube sits on a flat table and is rolled in one direction without lifting or slipping,
    making a few rotations by pivoting around its edges. 
    The total length of the path followed by the dot is dπ, where d is a constant.
    Prove that d = 1 given the following conditions:
    - Initial rotation around an edge joining the top and bottom faces,
    - Subsequent rotation around an edge on the side face until the dot returns to the top face.
-/ 
theorem cube_dot_path_length : 
  let side_length : ℝ := 2
  let radius : ℝ := side_length / 2
  let quarter_turn_length : ℝ := (1 / 4) * 2 * Real.pi * radius
  (2 * quarter_turn_length = Real.pi) → 
  d = 1 :=
by
  sorry

end cube_dot_path_length_l606_606572


namespace value_of_sequence_l606_606622

def f : ℝ → ℝ := sorry

axiom even_function (f : ℝ → ℝ) : ∀ x : ℝ, f (-x) = f x

axiom condition1 (f : ℝ → ℝ) : ∀ x : ℝ, f (3/2 + x) = f (3/2 - x)
axiom condition2 (f : ℝ → ℝ) : f (-1) = 1
axiom condition3 (f : ℝ → ℝ) : f 0 = -2

theorem value_of_sequence (f : ℝ → ℝ) [even_function f] [condition1 f] [condition2 f] [condition3 f] :
    (∑ i in (finset.range 2014).map (λ i, i + 1), f i) = 1 := sorry

end value_of_sequence_l606_606622


namespace population_of_village_Y_l606_606419

theorem population_of_village_Y (P_Y : ℕ) :
  let P_X := 72000
  let rate_X := -1200
  let rate_Y := 800
  let years := 15
  (P_X + rate_X * years = P_Y + rate_Y * years) →
  P_Y = 42000 :=
by
  intros P_X rate_X rate_Y years
  intro h
  exact sorry

end population_of_village_Y_l606_606419


namespace sum_of_base3_elements_converted_l606_606313

noncomputable def sum_of_four_digit_base3_elements := 
  let S : Finset ℕ := (Finset.range 81).filter (λ x, 27 ≤ x ∧ x ≤ 80)
  ∑ i in S, i

theorem sum_of_base3_elements_converted :
  Nat.toDigits 3 sum_of_four_digit_base3_elements = [1, 1, 2, 1, 2, 0, 0] :=
by
  sorry

end sum_of_base3_elements_converted_l606_606313


namespace count_3_digit_numbers_divisible_by_11_l606_606193

/-- 
  Define the mathematical conditions.
-/
def smallest_3_digit : ℕ := 100
def largest_3_digit : ℕ := 999
def is_divisible_by (n m : ℕ) : Prop := n % m = 0

/--
  Define the problem statement to prove that the count of 3-digit numbers divisible by 11 is 82.
-/
theorem count_3_digit_numbers_divisible_by_11 : 
  let nums := { n | smallest_3_digit ≤ n ∧ n ≤ largest_3_digit ∧ is_divisible_by n 11},
      count := nums.card
  in count = 82 := by
{
  sorry
}

end count_3_digit_numbers_divisible_by_11_l606_606193


namespace square_of_real_is_positive_or_zero_l606_606903

def p (x : ℝ) : Prop := x^2 > 0
def q (x : ℝ) : Prop := x^2 = 0

theorem square_of_real_is_positive_or_zero (x : ℝ) : (p x ∨ q x) :=
by
  sorry

end square_of_real_is_positive_or_zero_l606_606903


namespace calc_market_values_l606_606502

def market_value (initial_investment annual_income brokerage_fee : ℝ) : ℝ :=
  initial_investment + annual_income - (initial_investment * (brokerage_fee / 100))

theorem calc_market_values :
  market_value 6500 756 0.25 = 7223.75 ∧
  market_value 5500 935 0.33 = 6398.85 ∧
  market_value 4000 1225 0.50 = 5205 := 
by
  -- Proof of each part can be done here
  sorry

end calc_market_values_l606_606502


namespace sum_of_squares_CE_k_eq_16_l606_606632

def side_length : ℕ := 7
def BD1_length : ℕ := 2

-- Define equilateral triangle
structure EquilateralTriangle :=
(side_len : ℕ)

-- Define congruent triangles structure
structure CongruentTriangle extends EquilateralTriangle :=
()

-- Define vertices
structure Vertices :=
(D1 D2 E1 E2 E3 E4 C : ℕ)

noncomputable def sum_of_squares (s : ℕ) (r : ℕ) : ℕ :=
  let s_sq := s * s in
  let term1 := 8 * s_sq in
  let cos_theta := (s_sq - r*r) / (2 * s_sq) in
  let term2 := 8 * s_sq * cos_theta in
  term1 - term2

theorem sum_of_squares_CE_k_eq_16 (s : ℕ) (r : ℕ) (eq_tri : EquilateralTriangle) (congr_tris : List CongruentTriangle) : 
  (s = side_length) → (r = BD1_length) → (congr_tris.length = 4) →
  (sum_of_squares s r) = 16 :=
by
  intros
  sorry

end sum_of_squares_CE_k_eq_16_l606_606632


namespace exists_isosceles_triangle_l606_606825

noncomputable def right_triangle_area {α : ℝ} (h : 0 < α ∧ α < π / 2) :=
  let b := cos α
  let h := sin α
  in (b * h) / 2

noncomputable def isosceles_triangle_area {α : ℝ} (h : 0 < α ∧ α < π / 2) :=
  let b := cos α
  let h := sin α
  in (b^2 * sin α) / 2

theorem exists_isosceles_triangle (α : ℝ) (hα : 0 < α ∧ α < π / 2) : 
  ∃ (t₁ : ℝ), t₁ = isosceles_triangle_area hα ∧ t₁ ≥ right_triangle_area hα / real.cbrt 2 := 
sorry

end exists_isosceles_triangle_l606_606825


namespace convert_speed_approx_l606_606047

theorem convert_speed_approx :
  let speed_in_mps : ℚ := 18 / 42
  let conversion_factor : ℚ := 3.6
  speed_in_mps * conversion_factor ≈ 1.542857 := 
sorry

end convert_speed_approx_l606_606047


namespace correct_blanks_l606_606904

def fill_in_blanks (category : String) (plural_noun : String) : String :=
  "For many, winning remains " ++ category ++ " dream, but they continue trying their luck as there're always " ++ plural_noun ++ " chances that they might succeed."

theorem correct_blanks :
  fill_in_blanks "a" "" = "For many, winning remains a dream, but they continue trying their luck as there're always chances that they might succeed." :=
sorry

end correct_blanks_l606_606904


namespace repeating_decimals_sum_l606_606068

theorem repeating_decimals_sum :
  let x := 0.6666666 -- 0.\overline{6}
  let y := 0.2222222 -- 0.\overline{2}
  let z := 0.4444444 -- 0.\overline{4}
  (x + y - z) = 4 / 9 := 
by
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  calc
    -- Calculate (x + y - z)
    (x + y - z) = (2 / 3 + 2 / 9 - 4 / 9) : by sorry
                ... = 4 / 9 : by sorry


end repeating_decimals_sum_l606_606068


namespace repeating_decimal_sum_l606_606065

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem repeating_decimal_sum : repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l606_606065
