import Mathlib

namespace NUMINAMATH_GPT_sin_18_eq_sin_18_sin_54_eq_sin_36_sin_72_eq_l2085_208553

-- Part 1: Prove that sin 18° = ( √5 - 1 ) / 4
theorem sin_18_eq : Real.sin (Real.pi / 10) = (Real.sqrt 5 - 1) / 4 := sorry

-- Part 2: Given sin 18° = ( √5 - 1 ) / 4, prove that sin 18° * sin 54° = 1 / 4
theorem sin_18_sin_54_eq :
  Real.sin (Real.pi / 10) = (Real.sqrt 5 - 1) / 4 → 
  Real.sin (Real.pi / 10) * Real.sin (3 * Real.pi / 10) = 1 / 4 := sorry

-- Part 3: Given sin 18° = ( √5 - 1 ) / 4, prove that sin 36° * sin 72° = √5 / 4
theorem sin_36_sin_72_eq :
  Real.sin (Real.pi / 10) = (Real.sqrt 5 - 1) / 4 → 
  Real.sin (Real.pi / 5) * Real.sin (2 * Real.pi / 5) = Real.sqrt 5 / 4 := sorry

end NUMINAMATH_GPT_sin_18_eq_sin_18_sin_54_eq_sin_36_sin_72_eq_l2085_208553


namespace NUMINAMATH_GPT_range_of_a_l2085_208560

theorem range_of_a (a : ℝ) : ({x : ℝ | a - 4 < x ∧ x < a + 4} ⊆ {x : ℝ | 1 < x ∧ x < 3}) → (-1 ≤ a ∧ a ≤ 5) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2085_208560


namespace NUMINAMATH_GPT_sqrt_square_eq_self_sqrt_784_square_l2085_208522

theorem sqrt_square_eq_self (n : ℕ) (h : n ≥ 0) : (Real.sqrt n) ^ 2 = n :=
by
  sorry

theorem sqrt_784_square : (Real.sqrt 784) ^ 2 = 784 :=
by
  exact sqrt_square_eq_self 784 (Nat.zero_le 784)

end NUMINAMATH_GPT_sqrt_square_eq_self_sqrt_784_square_l2085_208522


namespace NUMINAMATH_GPT_range_of_a_l2085_208592

theorem range_of_a (a x : ℝ) (h : x - a = 1 - 2*x) (non_neg_x : x ≥ 0) : a ≥ -1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2085_208592


namespace NUMINAMATH_GPT_decagon_diagonals_l2085_208555

-- Definition of the number of diagonals in a polygon with n sides.
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- The proof problem statement
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end NUMINAMATH_GPT_decagon_diagonals_l2085_208555


namespace NUMINAMATH_GPT_poly_ineq_solution_l2085_208540

-- Define the inequality conversion
def poly_ineq (x : ℝ) : Prop :=
  x^2 + 2 * x ≤ -1

-- Formalize the set notation for the solution
def solution_set : Set ℝ :=
  { x | x = -1 }

-- State the theorem
theorem poly_ineq_solution : {x : ℝ | poly_ineq x} = solution_set :=
by
  sorry

end NUMINAMATH_GPT_poly_ineq_solution_l2085_208540


namespace NUMINAMATH_GPT_no_real_roots_x2_bx_8_eq_0_l2085_208534

theorem no_real_roots_x2_bx_8_eq_0 (b : ℝ) :
  (∀ x : ℝ, x^2 + b * x + 5 ≠ -3) ↔ (-4 * Real.sqrt 2 < b ∧ b < 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_GPT_no_real_roots_x2_bx_8_eq_0_l2085_208534


namespace NUMINAMATH_GPT_tom_spent_video_games_l2085_208527

def cost_football := 14.02
def cost_strategy := 9.46
def cost_batman := 12.04
def total_spent := cost_football + cost_strategy + cost_batman

theorem tom_spent_video_games : total_spent = 35.52 :=
by
  sorry

end NUMINAMATH_GPT_tom_spent_video_games_l2085_208527


namespace NUMINAMATH_GPT_bumper_cars_line_l2085_208541

theorem bumper_cars_line (initial in_line_leaving newcomers : ℕ) 
  (h_initial : initial = 9)
  (h_leaving : in_line_leaving = 6)
  (h_newcomers : newcomers = 3) :
  initial - in_line_leaving + newcomers = 6 :=
by
  sorry

end NUMINAMATH_GPT_bumper_cars_line_l2085_208541


namespace NUMINAMATH_GPT_equation_has_real_solution_l2085_208538

theorem equation_has_real_solution (m : ℝ) : ∃ x : ℝ, x^2 - m * x + m - 1 = 0 :=
by
  -- provide the hint that the discriminant (Δ) is (m - 2)^2
  have h : (m - 2)^2 ≥ 0 := by apply pow_two_nonneg
  sorry

end NUMINAMATH_GPT_equation_has_real_solution_l2085_208538


namespace NUMINAMATH_GPT_supplement_of_supplement_l2085_208542

def supplement (angle : ℝ) : ℝ :=
  180 - angle

theorem supplement_of_supplement (θ : ℝ) (h : θ = 35) : supplement (supplement θ) = 35 := by
  -- It is enough to state the theorem; the proof is not required as per the instruction.
  sorry

end NUMINAMATH_GPT_supplement_of_supplement_l2085_208542


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l2085_208552

-- Given conditions
variables {p q a b c : ℝ}
variables (hp_pos : 0 < p) (hq_pos : 0 < q) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
variables (hp_neq_q : p ≠ q)

-- p, a, q form a geometric sequence
variables (h_geo : a^2 = p * q)

-- p, b, c, q form an arithmetic sequence
variables (h_arith1 : 2 * b = p + c)
variables (h_arith2 : 2 * c = b + q)

-- Proof statement
theorem quadratic_no_real_roots (hp_pos hq_pos ha_pos hb_pos hc_pos hp_neq_q h_geo h_arith1 h_arith2 : ℝ) :
    (b * (x : ℝ)^2 - 2 * a * x + c = 0) → false :=
sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l2085_208552


namespace NUMINAMATH_GPT_remainder_4059_div_32_l2085_208513

theorem remainder_4059_div_32 : 4059 % 32 = 27 := by
  sorry

end NUMINAMATH_GPT_remainder_4059_div_32_l2085_208513


namespace NUMINAMATH_GPT_similar_triangle_legs_l2085_208565

theorem similar_triangle_legs (y : ℝ) 
  (h1 : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 15 ∧ b = 12)
  (h2 : ∃ u v w : ℝ, u^2 + v^2 = w^2 ∧ u = y ∧ v = 9) 
  (h3 : ∀ (a b c u v w : ℝ), (a^2 + b^2 = c^2 ∧ u^2 + v^2 = w^2 ∧ a/u = b/v) → (a = b → u = v)) 
  : y = 11.25 := 
  by 
    sorry

end NUMINAMATH_GPT_similar_triangle_legs_l2085_208565


namespace NUMINAMATH_GPT_arith_seq_sum_l2085_208575

theorem arith_seq_sum (a₃ a₄ a₅ : ℤ) (h₁ : a₃ = 7) (h₂ : a₄ = 11) (h₃ : a₅ = 15) :
  let d := a₄ - a₃;
  let a := a₄ - 3 * d;
  (6 / 2 * (2 * a + 5 * d)) = 54 :=
by
  sorry

end NUMINAMATH_GPT_arith_seq_sum_l2085_208575


namespace NUMINAMATH_GPT_find_sample_size_l2085_208549

theorem find_sample_size (f r : ℝ) (h1 : f = 20) (h2 : r = 0.125) (h3 : r = f / n) : n = 160 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_sample_size_l2085_208549


namespace NUMINAMATH_GPT_geometric_sequence_result_l2085_208533

-- Definitions representing the conditions
variables {a : ℕ → ℝ}

-- Conditions
axiom cond1 : a 7 * a 11 = 6
axiom cond2 : a 4 + a 14 = 5

theorem geometric_sequence_result :
  ∃ x, x = a 20 / a 10 ∧ (x = 2 / 3 ∨ x = 3 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_result_l2085_208533


namespace NUMINAMATH_GPT_smallest_number_divisible_by_20_and_36_l2085_208562

-- Define the conditions that x must be divisible by both 20 and 36
def divisible_by (x n : ℕ) : Prop := ∃ m : ℕ, x = n * m

-- Define the problem statement
theorem smallest_number_divisible_by_20_and_36 : 
  ∃ x : ℕ, divisible_by x 20 ∧ divisible_by x 36 ∧ 
  (∀ y : ℕ, (divisible_by y 20 ∧ divisible_by y 36) → y ≥ x) ∧ x = 180 := 
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_20_and_36_l2085_208562


namespace NUMINAMATH_GPT_convert_89_to_binary_l2085_208505

def divide_by_2_remainders (n : Nat) : List Nat :=
  if n = 0 then [] else (n % 2) :: divide_by_2_remainders (n / 2)

def binary_rep (n : Nat) : List Nat :=
  (divide_by_2_remainders n).reverse

theorem convert_89_to_binary :
  binary_rep 89 = [1, 0, 1, 1, 0, 0, 1] := sorry

end NUMINAMATH_GPT_convert_89_to_binary_l2085_208505


namespace NUMINAMATH_GPT_simplify_expression_l2085_208594

theorem simplify_expression (x : ℝ) :
  (3 * x)^3 - (4 * x^2) * (2 * x^3) = 27 * x^3 - 8 * x^5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2085_208594


namespace NUMINAMATH_GPT_max_value_quadratic_l2085_208511

theorem max_value_quadratic : ∀ s : ℝ, ∃ M : ℝ, (∀ s : ℝ, -3 * s^2 + 54 * s - 27 ≤ M) ∧ M = 216 :=
by
  sorry

end NUMINAMATH_GPT_max_value_quadratic_l2085_208511


namespace NUMINAMATH_GPT_conic_section_is_ellipse_l2085_208583

open Real

def is_conic_section_ellipse (x y : ℝ) (k : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  sqrt ((x - p1.1) ^ 2 + (y - p1.2) ^ 2) + sqrt ((x - p2.1) ^ 2 + (y - p2.2) ^ 2) = k

theorem conic_section_is_ellipse :
  is_conic_section_ellipse 2 (-2) 12 (2, -2) (-3, 5) :=
by
  sorry

end NUMINAMATH_GPT_conic_section_is_ellipse_l2085_208583


namespace NUMINAMATH_GPT_pepper_remaining_l2085_208507

/-- Brennan initially had 0.25 grams of pepper. He used 0.16 grams for scrambling eggs. 
His friend added x grams of pepper to another dish. Given y grams are remaining, 
prove that y = 0.09 + x . --/
theorem pepper_remaining (x y : ℝ) (h1 : 0.25 - 0.16 = 0.09) (h2 : y = 0.09 + x) : y = 0.09 + x := 
by
  sorry

end NUMINAMATH_GPT_pepper_remaining_l2085_208507


namespace NUMINAMATH_GPT_expected_balls_in_original_positions_after_transpositions_l2085_208579

theorem expected_balls_in_original_positions_after_transpositions :
  let num_balls := 7
  let first_swap_probability := 2 / 7
  let second_swap_probability := 1 / 7
  let third_swap_probability := 1 / 7
  let original_position_probability := (2 / 343) + (125 / 343)
  let expected_balls := num_balls * original_position_probability
  expected_balls = 889 / 343 := 
sorry

end NUMINAMATH_GPT_expected_balls_in_original_positions_after_transpositions_l2085_208579


namespace NUMINAMATH_GPT_area_inside_circle_outside_square_is_zero_l2085_208570

theorem area_inside_circle_outside_square_is_zero 
  (side_length : ℝ) (circle_radius : ℝ)
  (h_square_side : side_length = 2) (h_circle_radius : circle_radius = 1) : 
  (π * circle_radius^2) - (side_length^2) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_area_inside_circle_outside_square_is_zero_l2085_208570


namespace NUMINAMATH_GPT_find_coordinates_of_P_l2085_208500

theorem find_coordinates_of_P (P : ℝ × ℝ) (hx : abs P.2 = 5) (hy : abs P.1 = 3) (hq : P.1 < 0 ∧ P.2 > 0) : 
  P = (-3, 5) := 
  sorry

end NUMINAMATH_GPT_find_coordinates_of_P_l2085_208500


namespace NUMINAMATH_GPT_isosceles_triangle_base_angles_l2085_208580

theorem isosceles_triangle_base_angles (a b : ℝ) (h1 : a + b + b = 180)
  (h2 : a = 110) : b = 35 :=
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_angles_l2085_208580


namespace NUMINAMATH_GPT_ratio_of_sides_l2085_208556

theorem ratio_of_sides (a b : ℝ) (h1 : a + b = 3 * a) (h2 : a + b - Real.sqrt (a^2 + b^2) = (1 / 3) * b) : a / b = 1 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_sides_l2085_208556


namespace NUMINAMATH_GPT_kylie_first_hour_apples_l2085_208509

variable (A : ℕ) -- The number of apples picked in the first hour

-- Definitions based on the given conditions
def applesInFirstHour := A
def applesInSecondHour := 2 * A
def applesInThirdHour := A / 3

-- Total number of apples picked in all three hours
def totalApplesPicked := applesInFirstHour + applesInSecondHour + applesInThirdHour

-- The given condition that the total number of apples picked is 220
axiom total_is_220 : totalApplesPicked = 220

-- Proving that the number of apples picked in the first hour is 66
theorem kylie_first_hour_apples : A = 66 := by
  sorry

end NUMINAMATH_GPT_kylie_first_hour_apples_l2085_208509


namespace NUMINAMATH_GPT_sum_of_fractions_l2085_208572

theorem sum_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (7 : ℚ) / 9
  a + b = 83 / 72 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l2085_208572


namespace NUMINAMATH_GPT_number_of_people_per_cubic_yard_l2085_208588

-- Lean 4 statement

variable (P : ℕ) -- Number of people per cubic yard

def city_population_9000 := 9000 * P
def city_population_6400 := 6400 * P

theorem number_of_people_per_cubic_yard :
  city_population_9000 - city_population_6400 = 208000 →
  P = 80 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_per_cubic_yard_l2085_208588


namespace NUMINAMATH_GPT_least_positive_x_l2085_208585

theorem least_positive_x (x : ℕ) (h : (2 * x + 45)^2 % 43 = 0) : x = 42 :=
  sorry

end NUMINAMATH_GPT_least_positive_x_l2085_208585


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_pentagon_l2085_208578

theorem sum_of_interior_angles_of_pentagon :
  let n := 5
  let angleSum := 180 * (n - 2)
  angleSum = 540 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_pentagon_l2085_208578


namespace NUMINAMATH_GPT_symmetric_colors_different_at_8281_div_2_l2085_208558

def is_red (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ n = 81 * x + 100 * y

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n

theorem symmetric_colors_different_at_8281_div_2 :
  ∃ n : ℕ, (is_red n ∧ is_blue (8281 - n)) ∨ (is_blue n ∧ is_red (8281 - n)) ∧ 2 * n = 8281 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_colors_different_at_8281_div_2_l2085_208558


namespace NUMINAMATH_GPT_divisibility_by_11_l2085_208559

theorem divisibility_by_11 (m n : ℤ) (h : (5 * m + 3 * n) % 11 = 0) : (9 * m + n) % 11 = 0 := by
  sorry

end NUMINAMATH_GPT_divisibility_by_11_l2085_208559


namespace NUMINAMATH_GPT_car_returns_to_start_after_5_operations_l2085_208550

theorem car_returns_to_start_after_5_operations (α : ℝ) (h1 : 0 < α) (h2 : α < 180) : α = 72 ∨ α = 144 :=
sorry

end NUMINAMATH_GPT_car_returns_to_start_after_5_operations_l2085_208550


namespace NUMINAMATH_GPT_max_area_triangle_ABO1_l2085_208519

-- Definitions of the problem conditions
def l1 := {p : ℝ × ℝ | 2 * p.1 + 5 * p.2 = 1}

def C := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 + 4 * p.2 = 4}

def parallel (l1 l2 : ℝ × ℝ → Prop) := 
  ∃ m c1 c2, (∀ p, l1 p ↔ (p.2 = m * p.1 + c1)) ∧ (∀ p, l2 p ↔ (p.2 = m * p.1 + c2))

def intersects (l : ℝ × ℝ → Prop) (C: ℝ × ℝ → Prop) : Prop :=
  ∃ A B, (l A ∧ C A ∧ l B ∧ C B ∧ A ≠ B)

noncomputable def area (A B O : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((A.1 * (B.2 - O.2)) + (B.1 * (O.2 - A.2)) + (O.1 * (A.2 - B.2)))

-- Main statement to prove
theorem max_area_triangle_ABO1 :
  ∀ l2, parallel l1 l2 →
  intersects l2 C →
  ∃ A B, area A B (1, -2) ≤ 9 / 2 := 
sorry

end NUMINAMATH_GPT_max_area_triangle_ABO1_l2085_208519


namespace NUMINAMATH_GPT_find_total_children_l2085_208566

-- Define conditions as a Lean structure
structure SchoolDistribution where
  B : ℕ     -- Total number of bananas
  C : ℕ     -- Total number of children
  absent : ℕ := 160      -- Number of absent children (constant)
  bananas_per_child : ℕ := 2 -- Bananas per child originally (constant)
  bananas_extra : ℕ := 2      -- Extra bananas given to present children (constant)

-- Define the theorem we want to prove
theorem find_total_children (dist : SchoolDistribution) 
  (h1 : dist.B = 2 * dist.C) 
  (h2 : dist.B = 4 * (dist.C - dist.absent)) :
  dist.C = 320 := by
  sorry

end NUMINAMATH_GPT_find_total_children_l2085_208566


namespace NUMINAMATH_GPT_vector_rotation_correct_l2085_208597

def vector_rotate_z_90 (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := v
  ( -y, x, z )

theorem vector_rotation_correct :
  vector_rotate_z_90 (3, -1, 4) = (-3, 0, 4) := 
by 
  sorry

end NUMINAMATH_GPT_vector_rotation_correct_l2085_208597


namespace NUMINAMATH_GPT_minimize_cost_at_4_l2085_208520

-- Given definitions and conditions
def surface_area : ℝ := 12
def max_side_length : ℝ := 5
def front_face_cost_per_sqm : ℝ := 400
def sides_cost_per_sqm : ℝ := 150
def roof_ground_cost : ℝ := 5800
def wall_height : ℝ := 3

-- Definition of the total cost function
noncomputable def total_cost (x : ℝ) : ℝ :=
  900 * (x + 16 / x) + 5800

-- The main theorem to be proven
theorem minimize_cost_at_4 (h : 0 < x ∧ x ≤ max_side_length) : 
  (∀ x, total_cost x ≥ total_cost 4) ∧ total_cost 4 = 13000 :=
sorry

end NUMINAMATH_GPT_minimize_cost_at_4_l2085_208520


namespace NUMINAMATH_GPT_monotonic_function_a_ge_one_l2085_208564

theorem monotonic_function_a_ge_one (a : ℝ) :
  (∀ x : ℝ, (x^2 + 2 * x + a) ≥ 0) → a ≥ 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_monotonic_function_a_ge_one_l2085_208564


namespace NUMINAMATH_GPT_rectangle_area_l2085_208532

theorem rectangle_area (AB AC : ℝ) (AB_eq : AB = 15) (AC_eq : AC = 17) : 
  ∃ (BC : ℝ), (BC^2 = AC^2 - AB^2) ∧ (AB * BC = 120) := 
by
  -- Assuming necessary geometry axioms, such as the definition of a rectangle and Pythagorean theorem.
  sorry

end NUMINAMATH_GPT_rectangle_area_l2085_208532


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_l2085_208563

theorem arithmetic_geometric_mean (a b : ℝ) 
  (h1 : (a + b) / 2 = 20) 
  (h2 : Real.sqrt (a * b) = Real.sqrt 135) : 
  a^2 + b^2 = 1330 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_l2085_208563


namespace NUMINAMATH_GPT_prime_gt3_43_divides_expression_l2085_208569

theorem prime_gt3_43_divides_expression {p : ℕ} (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  (7^p - 6^p - 1) % 43 = 0 := 
  sorry

end NUMINAMATH_GPT_prime_gt3_43_divides_expression_l2085_208569


namespace NUMINAMATH_GPT_cost_of_fencing_l2085_208518

theorem cost_of_fencing
  (length width : ℕ)
  (ratio : 3 * width = 2 * length ∧ length * width = 5766)
  (cost_per_meter_in_paise : ℕ := 50)
  : (cost_per_meter_in_paise / 100 : ℝ) * 2 * (length + width) = 155 := 
by
  -- definitions
  sorry

end NUMINAMATH_GPT_cost_of_fencing_l2085_208518


namespace NUMINAMATH_GPT_arithmetic_seq_solution_l2085_208582

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

-- Definition of arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of first n terms of arithmetic sequence
def sum_arithmetic_seq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n + 1) / 2 * (a 0 + a n)

-- Given conditions
def given_conditions (a : ℕ → ℝ) : Prop :=
  a 0 + a 4 + a 8 = 27

-- Main theorem to be proved
theorem arithmetic_seq_solution (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (ha : arithmetic_seq a d)
  (hs : sum_arithmetic_seq S a)
  (h_given : given_conditions a) :
  a 4 = 9 ∧ S 8 = 81 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_solution_l2085_208582


namespace NUMINAMATH_GPT_carolyn_sum_of_removed_numbers_eq_31_l2085_208526

theorem carolyn_sum_of_removed_numbers_eq_31 :
  let initial_list := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let carolyn_first_turn := 4
  let carolyn_numbers_removed := [4, 9, 10, 8]
  let sum := carolyn_numbers_removed.sum
  sum = 31 :=
by
  sorry

end NUMINAMATH_GPT_carolyn_sum_of_removed_numbers_eq_31_l2085_208526


namespace NUMINAMATH_GPT_distance_between_lines_l2085_208503

-- Define lines l1 and l2
def line_l1 (x y : ℝ) := x + y + 1 = 0
def line_l2 (x y : ℝ) := 2 * x + 2 * y + 3 = 0

-- Proof statement for the distance between parallel lines
theorem distance_between_lines :
  let a := 1
  let b := 1
  let c1 := 1
  let c2 := 3 / 2
  let distance := |c2 - c1| / (Real.sqrt (a^2 + b^2))
  distance = Real.sqrt 2 / 4 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_lines_l2085_208503


namespace NUMINAMATH_GPT_find_t_l2085_208544

-- Define the roots and basic properties
variables (a b c : ℝ)
variables (r s t : ℝ)

-- Define conditions from the first cubic equation
def first_eq_roots : Prop :=
  a + b + c = -5 ∧ a * b * c = 13

-- Define conditions from the second cubic equation with shifted roots
def second_eq_roots : Prop :=
  t = -(a * b * c + a * b + a * c + b * c + a + b + c + 1)

-- The theorem stating the value of t
theorem find_t (h₁ : first_eq_roots a b c) (h₂ : second_eq_roots a b c t) : t = -15 :=
sorry

end NUMINAMATH_GPT_find_t_l2085_208544


namespace NUMINAMATH_GPT_fraction_of_lollipops_given_to_emily_is_2_3_l2085_208554

-- Given conditions as definitions
def initial_lollipops := 42
def kept_lollipops := 4
def lou_received := 10

-- The fraction of lollipops given to Emily
def fraction_given_to_emily : ℚ :=
  have emily_received : ℚ := initial_lollipops - (kept_lollipops + lou_received)
  have total_lollipops : ℚ := initial_lollipops
  emily_received / total_lollipops

-- The proof statement assert that fraction_given_to_emily is equal to 2/3
theorem fraction_of_lollipops_given_to_emily_is_2_3 : fraction_given_to_emily = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_of_lollipops_given_to_emily_is_2_3_l2085_208554


namespace NUMINAMATH_GPT_compositeShapeSum_is_42_l2085_208543

-- Define the pentagonal prism's properties
structure PentagonalPrism where
  faces : ℕ := 7
  edges : ℕ := 15
  vertices : ℕ := 10

-- Define the pyramid addition effect
structure PyramidAddition where
  additional_faces : ℕ := 5
  additional_edges : ℕ := 5
  additional_vertices : ℕ := 1
  covered_faces : ℕ := 1

-- Definition of composite shape properties
def compositeShapeSum (prism : PentagonalPrism) (pyramid : PyramidAddition) : ℕ :=
  (prism.faces - pyramid.covered_faces + pyramid.additional_faces) +
  (prism.edges + pyramid.additional_edges) +
  (prism.vertices + pyramid.additional_vertices)

-- The theorem to be proved: that the total sum is 42
theorem compositeShapeSum_is_42 : compositeShapeSum ⟨7, 15, 10⟩ ⟨5, 5, 1, 1⟩ = 42 := by
  sorry

end NUMINAMATH_GPT_compositeShapeSum_is_42_l2085_208543


namespace NUMINAMATH_GPT_boundary_length_is_25_point_7_l2085_208504

-- Define the side length derived from the given area.
noncomputable def sideLength (area : ℝ) : ℝ :=
  Real.sqrt area

-- Define the length of each segment when the square's side is divided into four equal parts.
noncomputable def segmentLength (side : ℝ) : ℝ :=
  side / 4

-- Define the total boundary length, which includes the circumference of the quarter-circle arcs and the straight segments.
noncomputable def totalBoundaryLength (area : ℝ) : ℝ :=
  let side := sideLength area
  let segment := segmentLength side
  let arcsLength := 2 * Real.pi * segment  -- the full circle's circumference
  let straightLength := 4 * segment
  arcsLength + straightLength

-- State the theorem that the total boundary length is approximately 25.7 units.
theorem boundary_length_is_25_point_7 :
  totalBoundaryLength 100 = 5 * Real.pi + 10 :=
by sorry

end NUMINAMATH_GPT_boundary_length_is_25_point_7_l2085_208504


namespace NUMINAMATH_GPT_prudence_sleep_4_weeks_equals_200_l2085_208524

-- Conditions
def sunday_to_thursday_sleep := 6 
def friday_saturday_sleep := 9 
def nap := 1 

-- Number of days in the mentioned periods per week
def sunday_to_thursday_days := 5
def friday_saturday_days := 2
def nap_days := 2

-- Calculate total sleep per week
def total_sleep_per_week : Nat :=
  (sunday_to_thursday_days * sunday_to_thursday_sleep) +
  (friday_saturday_days * friday_saturday_sleep) +
  (nap_days * nap)

-- Calculate total sleep in 4 weeks
def total_sleep_in_4_weeks : Nat :=
  4 * total_sleep_per_week

theorem prudence_sleep_4_weeks_equals_200 : total_sleep_in_4_weeks = 200 := by
  sorry

end NUMINAMATH_GPT_prudence_sleep_4_weeks_equals_200_l2085_208524


namespace NUMINAMATH_GPT_part1_part2_find_min_value_l2085_208545

open Real

-- Proof of Part 1
theorem part1 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : a^2 / b + b^2 / a ≥ a + b :=
by sorry

-- Proof of Part 2
theorem part2 (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : (1 - x)^2 / x + x^2 / (1 - x) ≥ 1 :=
by sorry

-- Corollary to find the minimum value
theorem find_min_value (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : (1 - x)^2 / x + x^2 / (1 - x) = 1 ↔ x = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_find_min_value_l2085_208545


namespace NUMINAMATH_GPT_scientific_notation_28400_is_correct_l2085_208561

theorem scientific_notation_28400_is_correct : (28400 : ℝ) = 2.84 * 10^4 := 
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_28400_is_correct_l2085_208561


namespace NUMINAMATH_GPT_ticket_door_price_l2085_208508

theorem ticket_door_price
  (total_attendance : ℕ)
  (tickets_before : ℕ)
  (price_before : ℚ)
  (total_receipts : ℚ)
  (tickets_bought_before : ℕ)
  (price_door : ℚ)
  (h_attendance : total_attendance = 750)
  (h_price_before : price_before = 2)
  (h_receipts : total_receipts = 1706.25)
  (h_tickets_before : tickets_bought_before = 475)
  (h_total_receipts : (tickets_bought_before * price_before) + (((total_attendance - tickets_bought_before) : ℕ) * price_door) = total_receipts) :
  price_door = 2.75 :=
by
  sorry

end NUMINAMATH_GPT_ticket_door_price_l2085_208508


namespace NUMINAMATH_GPT_B_works_alone_in_24_days_l2085_208531

noncomputable def B_completion_days (A B : ℝ) (h1 : A = B) (h2 : (A + B) / 12 = 1) : ℝ :=
24

theorem B_works_alone_in_24_days (A B : ℝ) (h1 : A = B) (h2 : (A + B) / 12 = 1) : 
  B_completion_days A B h1 h2 = 24 :=
sorry

end NUMINAMATH_GPT_B_works_alone_in_24_days_l2085_208531


namespace NUMINAMATH_GPT_Amelia_sell_JetBars_l2085_208539

theorem Amelia_sell_JetBars (M : ℕ) (h : 2 * M - 16 = 74) : M = 45 := by
  sorry

end NUMINAMATH_GPT_Amelia_sell_JetBars_l2085_208539


namespace NUMINAMATH_GPT_total_pictures_l2085_208512

theorem total_pictures :
  let Randy_pictures := 5
  let Peter_pictures := Randy_pictures + 3
  let Quincy_pictures := Peter_pictures + 20
  let Susan_pictures := 2 * Quincy_pictures - 7
  let Thomas_pictures := Randy_pictures ^ 3
  Randy_pictures + Peter_pictures + Quincy_pictures + Susan_pictures + Thomas_pictures = 215 := by 
    let Randy_pictures := 5
    let Peter_pictures := Randy_pictures + 3
    let Quincy_pictures := Peter_pictures + 20
    let Susan_pictures := 2 * Quincy_pictures - 7
    let Thomas_pictures := Randy_pictures ^ 3
    sorry

end NUMINAMATH_GPT_total_pictures_l2085_208512


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l2085_208521

theorem arithmetic_sequence_general_term (a_n S_n : ℕ → ℕ) (d : ℕ) (a1 S1 S5 S7 : ℕ)
  (h1: a_n 3 = 5)
  (h2: ∀ n, S_n n = (n * (a1 * 2 + (n - 1) * d)) / 2)
  (h3: S1 = S_n 1)
  (h4: S5 = S_n 5)
  (h5: S7 = S_n 7)
  (h6: S1 + S7 = 2 * S5):
  ∀ n, a_n n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l2085_208521


namespace NUMINAMATH_GPT_max_principals_in_8_years_l2085_208574

theorem max_principals_in_8_years 
  (years_in_term : ℕ)
  (terms_in_given_period : ℕ)
  (term_length : ℕ)
  (term_length_eq : term_length = 4)
  (given_period : ℕ)
  (given_period_eq : given_period = 8) :
  terms_in_given_period = given_period / term_length :=
by
  rw [term_length_eq, given_period_eq]
  sorry

end NUMINAMATH_GPT_max_principals_in_8_years_l2085_208574


namespace NUMINAMATH_GPT_mary_max_weekly_earnings_l2085_208535

noncomputable def mary_weekly_earnings (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℕ) (overtime_rate_factor : ℕ) : ℕ :=
  let overtime_hours := max_hours - regular_hours
  let overtime_rate := regular_rate + regular_rate * (overtime_rate_factor / 100)
  (regular_hours * regular_rate) + (overtime_hours * overtime_rate)

theorem mary_max_weekly_earnings : mary_weekly_earnings 60 30 12 50 = 900 :=
by
  sorry

end NUMINAMATH_GPT_mary_max_weekly_earnings_l2085_208535


namespace NUMINAMATH_GPT_magician_inequality_l2085_208576

theorem magician_inequality (N : ℕ) : 
  (N - 1) * 10^(N - 2) ≥ 10^N → N ≥ 101 :=
by
  sorry

end NUMINAMATH_GPT_magician_inequality_l2085_208576


namespace NUMINAMATH_GPT_find_C_l2085_208587

variable (A B C : ℚ)

def condition1 := A + B + C = 350
def condition2 := A + C = 200
def condition3 := B + C = 350

theorem find_C : condition1 A B C → condition2 A C → condition3 B C → C = 200 :=
by
  sorry

end NUMINAMATH_GPT_find_C_l2085_208587


namespace NUMINAMATH_GPT_greatest_possible_grapes_thrown_out_l2085_208567

theorem greatest_possible_grapes_thrown_out (n : ℕ) : 
  n % 7 ≤ 6 := by 
  sorry

end NUMINAMATH_GPT_greatest_possible_grapes_thrown_out_l2085_208567


namespace NUMINAMATH_GPT_square_combinations_l2085_208523

theorem square_combinations (n : ℕ) (h : n * (n - 1) = 30) : n * (n - 1) = 30 :=
by sorry

end NUMINAMATH_GPT_square_combinations_l2085_208523


namespace NUMINAMATH_GPT_students_take_neither_l2085_208528

variable (Total Mathematic Physics Both MathPhysics ChemistryNeither Neither : ℕ)

axiom Total_students : Total = 80
axiom students_mathematics : Mathematic = 50
axiom students_physics : Physics = 40
axiom students_both : Both = 25
axiom students_chemistry_neither : ChemistryNeither = 10

theorem students_take_neither :
  Neither = Total - (Mathematic - Both + Physics - Both + Both + ChemistryNeither) :=
  by
  have Total_students := Total_students
  have students_mathematics := students_mathematics
  have students_physics := students_physics
  have students_both := students_both
  have students_chemistry_neither := students_chemistry_neither
  sorry

end NUMINAMATH_GPT_students_take_neither_l2085_208528


namespace NUMINAMATH_GPT_find_positive_integers_with_divisors_and_sum_l2085_208506

theorem find_positive_integers_with_divisors_and_sum (n : ℕ) :
  (∃ d1 d2 d3 d4 d5 d6 : ℕ,
    (n ≠ 0) ∧ (n ≠ 1) ∧ 
    n = d1 * d2 * d3 * d4 * d5 * d6 ∧
    d1 ≠ 1 ∧ d2 ≠ 1 ∧ d3 ≠ 1 ∧ d4 ≠ 1 ∧ d5 ≠ 1 ∧ d6 ≠ 1 ∧
    (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d1 ≠ d5) ∧ (d1 ≠ d6) ∧
    (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d2 ≠ d5) ∧ (d2 ≠ d6) ∧
    (d3 ≠ d4) ∧ (d3 ≠ d5) ∧ (d3 ≠ d6) ∧
    (d4 ≠ d5) ∧ (d4 ≠ d6) ∧
    (d5 ≠ d6) ∧
    d1 + d2 + d3 + d4 + d5 + d6 = 14133
  ) -> 
  (n = 16136 ∨ n = 26666) :=
sorry

end NUMINAMATH_GPT_find_positive_integers_with_divisors_and_sum_l2085_208506


namespace NUMINAMATH_GPT_vote_count_l2085_208537

theorem vote_count 
(h_total: 200 = h_votes + l_votes + y_votes)
(h_hl: 3 * l_votes = 2 * h_votes)
(l_ly: 6 * y_votes = 5 * l_votes):
h_votes = 90 ∧ l_votes = 60 ∧ y_votes = 50 := by 
sorry

end NUMINAMATH_GPT_vote_count_l2085_208537


namespace NUMINAMATH_GPT_fourth_guard_distance_l2085_208546

theorem fourth_guard_distance 
  (length : ℝ) (width : ℝ)
  (total_distance_three_guards: ℝ)
  (P : ℝ := 2 * (length + width)) 
  (total_distance_four_guards : ℝ := P)
  (total_three : total_distance_three_guards = 850)
  (length_value : length = 300)
  (width_value : width = 200) :
  ∃ distance_fourth_guard : ℝ, distance_fourth_guard = 150 :=
by 
  sorry

end NUMINAMATH_GPT_fourth_guard_distance_l2085_208546


namespace NUMINAMATH_GPT_find_common_difference_l2085_208595

variable {aₙ : ℕ → ℝ}
variable {Sₙ : ℕ → ℝ}

-- Condition that the sum of the first n terms of the arithmetic sequence is S_n
def is_arith_seq (aₙ : ℕ → ℝ) (Sₙ : ℕ → ℝ) : Prop :=
  ∀ n, Sₙ n = (n * (aₙ 0 + (aₙ (n - 1))) / 2)

-- Condition given in the problem
def problem_condition (Sₙ : ℕ → ℝ) : Prop :=
  2 * Sₙ 3 - 3 * Sₙ 2 = 12

theorem find_common_difference (h₀ : is_arith_seq aₙ Sₙ) (h₁ : problem_condition Sₙ) : 
  ∃ d : ℝ, d = 4 := 
sorry

end NUMINAMATH_GPT_find_common_difference_l2085_208595


namespace NUMINAMATH_GPT_diamond_more_olivine_l2085_208568

theorem diamond_more_olivine :
  ∃ A O D : ℕ, A = 30 ∧ O = A + 5 ∧ A + O + D = 111 ∧ D - O = 11 :=
by
  sorry

end NUMINAMATH_GPT_diamond_more_olivine_l2085_208568


namespace NUMINAMATH_GPT_distance_X_X_l2085_208571

/-
  Define the vertices of the triangle XYZ
-/
def X : ℝ × ℝ := (2, -4)
def Y : ℝ × ℝ := (-1, 2)
def Z : ℝ × ℝ := (5, 1)

/-
  Define the reflection of point X over the y-axis
-/
def X' : ℝ × ℝ := (-2, -4)

/-
  Prove that the distance between X and X' is 4 units.
-/
theorem distance_X_X' : (Real.sqrt (((-2) - 2) ^ 2 + ((-4) - (-4)) ^ 2)) = 4 := by
  sorry

end NUMINAMATH_GPT_distance_X_X_l2085_208571


namespace NUMINAMATH_GPT_totalCandy_l2085_208548

-- Define the number of pieces of candy each person had
def TaquonCandy : ℕ := 171
def MackCandy : ℕ := 171
def JafariCandy : ℕ := 76

-- Prove that the total number of pieces of candy they had together is 418
theorem totalCandy : TaquonCandy + MackCandy + JafariCandy = 418 := by
  sorry

end NUMINAMATH_GPT_totalCandy_l2085_208548


namespace NUMINAMATH_GPT_fraction_simplification_l2085_208598

theorem fraction_simplification (x : ℝ) (h : x = Real.sqrt 2) : 
  ( (x^2 - 1) / (x^2 - x) - 1) = Real.sqrt 2 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_simplification_l2085_208598


namespace NUMINAMATH_GPT_total_rankings_l2085_208577

-- Defines the set of players
inductive Player
| P : Player
| Q : Player
| R : Player
| S : Player

-- Defines a function to count the total number of ranking sequences
def total_possible_rankings (p : Player → Player → Prop) : Nat := 
  4 * 2 * 2

-- Problem statement
theorem total_rankings : ∃ t : Player → Player → Prop, total_possible_rankings t = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_rankings_l2085_208577


namespace NUMINAMATH_GPT_find_x_l2085_208529

variable (x : ℝ)
variable (s : ℝ)

-- Conditions as hypothesis
def square_perimeter_60 (s : ℝ) : Prop := 4 * s = 60
def triangle_area_150 (x s : ℝ) : Prop := (1 / 2) * x * s = 150
def height_equals_side (s : ℝ) : Prop := true

-- Proof problem statement
theorem find_x 
  (h1 : square_perimeter_60 s)
  (h2 : triangle_area_150 x s)
  (h3 : height_equals_side s) : 
  x = 20 := 
sorry

end NUMINAMATH_GPT_find_x_l2085_208529


namespace NUMINAMATH_GPT_cooper_remaining_pies_l2085_208515

def total_pies (pies_per_day : ℕ) (days : ℕ) : ℕ := pies_per_day * days

def remaining_pies (total : ℕ) (eaten : ℕ) : ℕ := total - eaten

theorem cooper_remaining_pies :
  remaining_pies (total_pies 7 12) 50 = 34 :=
by sorry

end NUMINAMATH_GPT_cooper_remaining_pies_l2085_208515


namespace NUMINAMATH_GPT_full_price_ticket_revenue_l2085_208586

theorem full_price_ticket_revenue 
  (f h p : ℕ)
  (h1 : f + h = 160)
  (h2 : f * p + h * (p / 3) = 2400) :
  f * p = 400 := 
sorry

end NUMINAMATH_GPT_full_price_ticket_revenue_l2085_208586


namespace NUMINAMATH_GPT_probability_quadratic_real_roots_l2085_208573

noncomputable def probability_real_roots : ℝ := 3 / 4

theorem probability_quadratic_real_roots :
  (∀ a b : ℝ, -π ≤ a ∧ a ≤ π ∧ -π ≤ b ∧ b ≤ π →
  (∃ x : ℝ, x^2 + 2*a*x - b^2 + π = 0) ↔ a^2 + b^2 ≥ π) →
  (probability_real_roots = 3 / 4) :=
sorry

end NUMINAMATH_GPT_probability_quadratic_real_roots_l2085_208573


namespace NUMINAMATH_GPT_y_exceeds_x_by_100_percent_l2085_208530

theorem y_exceeds_x_by_100_percent (x y : ℝ) (h : x = 0.5 * y) : (y - x) / x = 1 := by
sorry

end NUMINAMATH_GPT_y_exceeds_x_by_100_percent_l2085_208530


namespace NUMINAMATH_GPT_twigs_per_branch_l2085_208596

/-- Definitions -/
def total_branches : ℕ := 30
def total_leaves : ℕ := 12690
def percentage_4_leaves : ℝ := 0.30
def leaves_per_twig_4_leaves : ℕ := 4
def percentage_5_leaves : ℝ := 0.70
def leaves_per_twig_5_leaves : ℕ := 5

/-- Given conditions translated to Lean -/
def hypothesis (T : ℕ) : Prop :=
  (percentage_4_leaves * T * leaves_per_twig_4_leaves) +
  (percentage_5_leaves * T * leaves_per_twig_5_leaves) = total_leaves

/-- The main theorem to prove -/
theorem twigs_per_branch
  (T : ℕ)
  (h : hypothesis T) :
  (T / total_branches) = 90 :=
sorry

end NUMINAMATH_GPT_twigs_per_branch_l2085_208596


namespace NUMINAMATH_GPT_roots_eqn_values_l2085_208547

theorem roots_eqn_values : 
  ∀ (x1 x2 : ℝ), (x1^2 + x1 - 4 = 0) ∧ (x2^2 + x2 - 4 = 0) ∧ (x1 + x2 = -1)
  → (x1^3 - 5 * x2^2 + 10 = -19) := 
by
  intros x1 x2
  intros h
  sorry

end NUMINAMATH_GPT_roots_eqn_values_l2085_208547


namespace NUMINAMATH_GPT_tennis_balls_ordered_l2085_208590

variables (W Y : ℕ)
def original_eq (W Y : ℕ) := W = Y
def ratio_condition (W Y : ℕ) := W / (Y + 90) = 8 / 13
def total_tennis_balls (W Y : ℕ) := W + Y = 288

theorem tennis_balls_ordered (W Y : ℕ) (h1 : original_eq W Y) (h2 : ratio_condition W Y) : total_tennis_balls W Y :=
sorry

end NUMINAMATH_GPT_tennis_balls_ordered_l2085_208590


namespace NUMINAMATH_GPT_fraction_to_decimal_l2085_208516

theorem fraction_to_decimal : (7 : Rat) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l2085_208516


namespace NUMINAMATH_GPT_blankets_warmth_increase_l2085_208501

-- Conditions
def blankets_in_closet : ℕ := 14
def blankets_used : ℕ := blankets_in_closet / 2
def degree_per_blanket : ℕ := 3

-- Goal: Prove that the total temperature increase is 21 degrees.
theorem blankets_warmth_increase : blankets_used * degree_per_blanket = 21 :=
by
  sorry

end NUMINAMATH_GPT_blankets_warmth_increase_l2085_208501


namespace NUMINAMATH_GPT_simplify_expansion_l2085_208510

-- Define the variables and expressions
variable (x : ℝ)

-- The main statement
theorem simplify_expansion : (x + 5) * (4 * x - 12) = 4 * x^2 + 8 * x - 60 :=
by sorry

end NUMINAMATH_GPT_simplify_expansion_l2085_208510


namespace NUMINAMATH_GPT_infinite_squares_form_l2085_208581

theorem infinite_squares_form (k : ℕ) (hk : 0 < k) : ∃ f : ℕ → ℕ, ∀ n, ∃ a, a^2 = f n * 2^k - 7 :=
by
  sorry

end NUMINAMATH_GPT_infinite_squares_form_l2085_208581


namespace NUMINAMATH_GPT_product_inequality_l2085_208551

variable (x1 x2 x3 x4 y1 y2 : ℝ)

theorem product_inequality (h1 : y2 ≥ y1) 
                          (h2 : y1 ≥ x1)
                          (h3 : x1 ≥ x3)
                          (h4 : x3 ≥ x2)
                          (h5 : x2 ≥ x1)
                          (h6 : x1 ≥ 2)
                          (h7 : x1 + x2 + x3 + x4 ≥ y1 + y2) : 
                          x1 * x2 * x3 * x4 ≥ y1 * y2 :=
  sorry

end NUMINAMATH_GPT_product_inequality_l2085_208551


namespace NUMINAMATH_GPT_sqrt_mul_simplify_l2085_208514

theorem sqrt_mul_simplify : Real.sqrt 18 * Real.sqrt 32 = 24 := 
sorry

end NUMINAMATH_GPT_sqrt_mul_simplify_l2085_208514


namespace NUMINAMATH_GPT_y_relationship_l2085_208593

variable (a c : ℝ) (h_a : a < 0)

def f (x : ℝ) : ℝ := a * (x - 3) ^ 2 + c

theorem y_relationship (y1 y2 y3 : ℝ)
  (h1 : y1 = f a c (Real.sqrt 5))
  (h2 : y2 = f a c 0)
  (h3 : y3 = f a c 4) :
  y2 < y3 ∧ y3 < y1 :=
  sorry

end NUMINAMATH_GPT_y_relationship_l2085_208593


namespace NUMINAMATH_GPT_one_twenty_percent_of_number_l2085_208557

theorem one_twenty_percent_of_number (x : ℝ) (h : 0.20 * x = 300) : 1.20 * x = 1800 :=
by 
sorry

end NUMINAMATH_GPT_one_twenty_percent_of_number_l2085_208557


namespace NUMINAMATH_GPT_tank_capacity_l2085_208591

-- Define the initial fullness of the tank and the total capacity
def initial_fullness (w c : ℝ) : Prop :=
  w = c / 5

-- Define the fullness of the tank after adding 5 liters
def fullness_after_adding (w c : ℝ) : Prop :=
  (w + 5) / c = 2 / 7

-- The main theorem: if both conditions hold, c must equal to 35/3
theorem tank_capacity (w c : ℝ) (h1 : initial_fullness w c) (h2 : fullness_after_adding w c) : 
  c = 35 / 3 :=
sorry

end NUMINAMATH_GPT_tank_capacity_l2085_208591


namespace NUMINAMATH_GPT_music_tool_cost_l2085_208589

namespace BandCost

def trumpet_cost : ℝ := 149.16
def song_book_cost : ℝ := 4.14
def total_spent : ℝ := 163.28

theorem music_tool_cost : (total_spent - (trumpet_cost + song_book_cost)) = 9.98 :=
by
  sorry

end NUMINAMATH_GPT_music_tool_cost_l2085_208589


namespace NUMINAMATH_GPT_number_of_men_in_company_l2085_208517

noncomputable def total_workers : ℝ := 2752.8
noncomputable def women_in_company : ℝ := 91.76
noncomputable def workers_without_retirement_plan : ℝ := (1 / 3) * total_workers
noncomputable def percent_women_without_retirement_plan : ℝ := 0.10
noncomputable def percent_men_with_retirement_plan : ℝ := 0.40
noncomputable def workers_with_retirement_plan : ℝ := (2 / 3) * total_workers
noncomputable def men_with_retirement_plan : ℝ := percent_men_with_retirement_plan * workers_with_retirement_plan

theorem number_of_men_in_company : (total_workers - women_in_company) = 2661.04 := by
  -- Insert the exact calculations and algebraic manipulations
  sorry

end NUMINAMATH_GPT_number_of_men_in_company_l2085_208517


namespace NUMINAMATH_GPT_work_completion_days_l2085_208502

theorem work_completion_days (A B C : ℕ) 
  (hA : A = 4) (hB : B = 8) (hC : C = 8) : 
  2 = 1 / (1 / A + 1 / B + 1 / C) :=
by
  -- skip the proof for now
  sorry

end NUMINAMATH_GPT_work_completion_days_l2085_208502


namespace NUMINAMATH_GPT_product_form_l2085_208584

theorem product_form (a b c d : ℤ) :
    (a^2 - 7*b^2) * (c^2 - 7*d^2) = (a*c + 7*b*d)^2 - 7*(a*d + b*c)^2 :=
by sorry

end NUMINAMATH_GPT_product_form_l2085_208584


namespace NUMINAMATH_GPT_committee_selections_with_at_least_one_prev_served_l2085_208536

-- Define the conditions
def total_candidates := 20
def previously_served := 8
def committee_size := 4
def never_served := total_candidates - previously_served

-- The proof problem statement
theorem committee_selections_with_at_least_one_prev_served : 
  (Nat.choose total_candidates committee_size - Nat.choose never_served committee_size) = 4350 :=
by
  sorry

end NUMINAMATH_GPT_committee_selections_with_at_least_one_prev_served_l2085_208536


namespace NUMINAMATH_GPT_max_sum_is_38_l2085_208599

-- Definition of the problem variables and conditions
def number_set : Set ℤ := {2, 3, 8, 9, 14, 15}
variable (a b c d e : ℤ)

-- Conditions translated to Lean
def condition1 : Prop := b = c
def condition2 : Prop := a = d

-- Sum condition to find maximum sum
def max_combined_sum : ℤ := a + b + e

theorem max_sum_is_38 : 
  ∃ a b c d e, 
    {a, b, c, d, e} ⊆ number_set ∧
    b = c ∧ 
    a = d ∧ 
    a + b + e = 38 :=
sorry

end NUMINAMATH_GPT_max_sum_is_38_l2085_208599


namespace NUMINAMATH_GPT_common_points_line_circle_l2085_208525

theorem common_points_line_circle (a : ℝ) : 
  (∀ x y: ℝ, (x - 2*y + a = 0) → ((x - 2)^2 + y^2 = 1)) ↔ (-2 - Real.sqrt 5 ≤ a ∧ a ≤ -2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_GPT_common_points_line_circle_l2085_208525
