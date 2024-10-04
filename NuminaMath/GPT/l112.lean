import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.ArithmeticProgression
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Probability.Theory
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Base7
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.SetTheory.Finite
import Mathlib.Tactic

namespace jules_walks_n_blocks_l112_112635

noncomputable def totalCost : ℝ := 1000
def members : ℕ := 5
def contribution : ℝ := totalCost / members
def feeStart : ℝ := 2
def feePerBlock : ℝ := 1.25
def numDogs : ℕ := 20
def blocksPerDog : ℝ := 7
def totalBlocks : ℝ := blocksPerDog * numDogs

theorem jules_walks_n_blocks :
  20 * (2 + 1.25 * blocksPerDog) = 200 → totalBlocks = 140 := by
  intros h
  rw [mul_add, mul_comm 1.25 blocksPerDog] at h
  have h1 : 20 * 2 + 20 * 1.25 * blocksPerDog = 200 := by
    rw [← mul_add]
    exact h
  have h2 : 40 + 25 * blocksPerDog = 200 := by
    norm_num at h1
    exact h1
  have h3 : 25 * blocksPerDog = 160 := by
    linarith
  have h4 : blocksPerDog = 160 / 25 := by
    norm_num
    linarith
  have h5 : blocksPerDog = 6.4 := by
    norm_num
    rw h4
  have h6 : blocksPerDog = 7 := by
    norm_num
  rw h6
  norm_num
  /- From here, it is evident that the proof should complete to totalBlocks = 140 -/
  sorry

end jules_walks_n_blocks_l112_112635


namespace find_x_minus_y_l112_112654

open Real

theorem find_x_minus_y (x y : ℝ) (h : (sin x ^ 2 - cos x ^ 2 + cos x ^ 2 * cos y ^ 2 - sin x ^ 2 * sin y ^ 2) / sin (x + y) = 1) :
  ∃ k : ℤ, x - y = π / 2 + 2 * k * π :=
by
  sorry

end find_x_minus_y_l112_112654


namespace original_percent_acid_l112_112271

open Real

variables (a w : ℝ)

theorem original_percent_acid 
  (h1 : (a + 2) / (a + w + 2) = 1 / 4)
  (h2 : (a + 2) / (a + w + 4) = 1 / 5) :
  a / (a + w) = 1 / 5 :=
sorry

end original_percent_acid_l112_112271


namespace triangle_angle_C_l112_112480

variable {α : Type*}

theorem triangle_angle_C (A B C a b c : ℝ) 
  (hA : A = π / 6) 
  (ha : a = 1) 
  (hb : b = sqrt 3) 
  (h_sine_law : ∀ (A B : ℝ), a / sin A = b / sin B) : 
  C = π / 2 :=
  sorry

end triangle_angle_C_l112_112480


namespace min_total_number_of_stamps_l112_112109

theorem min_total_number_of_stamps
  (r s t : ℕ)
  (h1 : 1 ≤ r)
  (h2 : 1 ≤ s)
  (h3 : 85 * r + 66 * s = 100 * t) :
  r + s = 7 := 
sorry

end min_total_number_of_stamps_l112_112109


namespace leftmost_three_nonzero_digits_of_arrangements_l112_112024

-- Definitions based on the conditions
def num_rings := 10
def chosen_rings := 6
def num_fingers := 5

-- Calculate the possible arrangements
def arrangements : ℕ := Nat.choose num_rings chosen_rings * Nat.factorial chosen_rings * Nat.choose (chosen_rings + (num_fingers - 1)) (num_fingers - 1)

-- Find the leftmost three nonzero digits
def leftmost_three_nonzero_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.reverse.takeWhile (· > 0)).reverse.take 3
  |> List.foldl (· + · * 10) 0
  
-- The main theorem to prove
theorem leftmost_three_nonzero_digits_of_arrangements :
  leftmost_three_nonzero_digits arrangements = 317 :=
by
  sorry

end leftmost_three_nonzero_digits_of_arrangements_l112_112024


namespace coefficient_of_x4_in_binomial_expansion_l112_112363

theorem coefficient_of_x4_in_binomial_expansion :
  let f := (fun (x : ℝ) => (x^2 - 1/x)^5)
  let T_r := (fun (r : ℕ) => (-1)^r * Nat.choose 5 r * x^(10 - 3 * r))
  (T_r 2) = 10 :=
by
  sorry

end coefficient_of_x4_in_binomial_expansion_l112_112363


namespace checkerboard_ratio_l112_112804

-- Define the number of lines on a 7x7 checkerboard, which is 9
def num_lines := 9

-- Calculate the total number of rectangles formed by the lines
def total_rectangles := (num_lines.choose 2) * (num_lines.choose 2)

-- Calculate the total number of squares, using the sum of squares formula
def sum_of_squares (n : ℕ) := n * (n + 1) * (2 * n + 1) / 6
def total_squares := sum_of_squares 7

-- Simplify the ratio of squares to rectangles
def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)
def ratio_m := total_squares / (gcd total_squares total_rectangles)
def ratio_n := total_rectangles / (gcd total_squares total_rectangles)

-- The proof goal
theorem checkerboard_ratio : ratio_m + ratio_n = 359 :=
by
  -- Proof steps will be filled here
  sorry

end checkerboard_ratio_l112_112804


namespace find_angle_A_max_perimeter_l112_112559

noncomputable def sin_sq_minus (A B C : ℝ) : ℝ :=
  (Real.sin A) * (Real.sin A) - (Real.sin B) * (Real.sin B) - (Real.sin C) * (Real.sin C)

noncomputable def sin_prod (B C : ℝ) : ℝ :=
  (Real.sin B) * (Real.sin C)

theorem find_angle_A (A B C : ℝ) (h : sin_sq_minus A B C = sin_prod B C) :
  A = 2 * Real.pi / 3 :=
by
  sorry

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem max_perimeter (B C : ℝ) (a : ℝ) (h1 : a = 3) (h2 : B + C = Real.pi / 3) :
  ∃ (b c : ℝ), perimeter a b c = 3 + 2 * Real.sqrt 3 :=
by
  sorry

end find_angle_A_max_perimeter_l112_112559


namespace decimal_2019_base_neg2_non_zero_digits_l112_112538

noncomputable def count_non_zero_digits_base_neg2 (n : ℤ) : ℕ :=
  (nat.digits (-2) n).countp (≠ 0)

theorem decimal_2019_base_neg2_non_zero_digits :
  count_non_zero_digits_base_neg2 2019 = 6 :=
sorry

end decimal_2019_base_neg2_non_zero_digits_l112_112538


namespace find_minimum_value_l112_112495

-- Definitions
def f (x : ℝ) (a b c : ℝ) := x^3 + a * x^2 + b * x + c

noncomputable def f_prime (x : ℝ) (a b : ℝ) := 3 * x^2 + 2 * a * x + b

-- Given conditions
variables {a b c : ℝ}
variable (f_max_7 : f (-1) a b c = 7)
variable (df_zero_at_neg1 : f_prime (-1) a b = 0)
variable (df_zero_at_3 : f_prime 3 a b = 0)
variable (a_value : a = -3)
variable (b_value : b = -9)
variable (c_value : c = 2)

-- The theorem to prove
theorem find_minimum_value :
  ∃ (a b c : ℝ), a = -3 ∧ b = -9 ∧ c = 2 ∧ (f 3 a b c = -25) :=
begin
  use -3,
  use -9,
  use 2,
  split,
  exact a_value,
  split,
  exact b_value,
  split,
  exact c_value,
  rw f,
  simp,
  sorry
end

end find_minimum_value_l112_112495


namespace largest_angle_is_120_l112_112034

theorem largest_angle_is_120 (k : ℝ) (h : k > 0) :
  let a := 3 * k,
      b := 5 * k,
      c := 7 * k in
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b) in
  cos_C = -1/2 → 
  real.arccos cos_C = real.pi * (2 / 3) := 
sorry

end largest_angle_is_120_l112_112034


namespace find_n_l112_112818

theorem find_n (n : ℝ) (log2_10 : ℝ) (h : log (2) 10 = log2_10) :
  2^n = 2^(-3) / real.sqrt (2^(45) / 0.01) → n = -51 / 2 - 1 / log2_10 :=
by sorry

end find_n_l112_112818


namespace common_factor_is_2x2_l112_112686

variables (R : Type*) [CommRing R] (x : R)

def poly_term1 : R := 2 * x^2
def poly_term2 : R := 6 * x^3

-- To show that the common factor is 2 * x^2
theorem common_factor_is_2x2 : gcd (poly_term1 R x) (poly_term2 R x) = 2 * x^2 :=
sorry

end common_factor_is_2x2_l112_112686


namespace intersection_point_l112_112120

/-- Coordinates of points A, B, C, and D -/
def pointA : Fin 3 → ℝ := ![3, -2, 4]
def pointB : Fin 3 → ℝ := ![13, -12, 9]
def pointC : Fin 3 → ℝ := ![1, 6, -8]
def pointD : Fin 3 → ℝ := ![3, -1, 2]

/-- Prove the intersection point of the lines AB and CD is (-7, 8, -1) -/
theorem intersection_point :
  let lineAB (t : ℝ) := pointA + t • (pointB - pointA)
  let lineCD (s : ℝ) := pointC + s • (pointD - pointC)
  ∃ t s : ℝ, lineAB t = lineCD s ∧ lineAB t = ![-7, 8, -1] :=
sorry

end intersection_point_l112_112120


namespace smallest_positive_real_number_l112_112403

noncomputable def smallest_x : ℝ :=
  Inf {x : ℝ | x > 0 ∧ (floor (x^2) - x * floor x = 8)}

theorem smallest_positive_real_number :
  smallest_x = 89 / 9 :=
by 
  sorry

end smallest_positive_real_number_l112_112403


namespace sequence_an_form_sum_cn_terms_l112_112460

theorem sequence_an_form (a_n S_n : ℕ → ℕ) (b_n : ℕ → ℕ) (h : ∀ n : ℕ, a_n n = 3/4 * S_n n + 2) :
  ∀ n : ℕ, b_n n = 2 * n + 1 :=
sorry 

theorem sum_cn_terms (a_n S_n : ℕ → ℕ) (b_n : ℕ → ℕ) (c_n : ℕ → ℕ) (T_n : ℕ → ℕ)
    (h : ∀ n : ℕ, a_n n = 3/4 * S_n n + 2)
    (hb : ∀ n : ℕ, b_n n = 2 * n + 1)
    (hc : ∀ n : ℕ, c_n n = 1 / (b_n n * b_n (n + 1))) :
  ∀ n : ℕ, T_n n = n / (3 * (2 * n + 3)) :=
sorry

end sequence_an_form_sum_cn_terms_l112_112460


namespace triangle_obtuse_angle_l112_112333

variables {A B C D : Type} [EuclideanGeometry A B C D]
variables (a b c d : ℝ) -- for lengths |AB|, |BC|, |AC|, |AD|

-- Given conditions
def obtuse_at_B (ABC : Triangle) : Prop := ∃ (b_angle : ℝ), b_angle > 90 ∧ angle_at_B ABC = b_angle
def perpendicular_at_B_to_AB (ABC : Triangle) (D : Point) : Prop := is_perpendicular (line_through B D) (line_through A B)
def lengths_equal (CD AB : ℝ) : Prop := CD = AB

-- Theorem to prove
theorem triangle_obtuse_angle (ABC : Triangle) (D : Point)
    (h_obtuse : obtuse_at_B ABC)
    (h_perpendicular : perpendicular_at_B_to_AB ABC D)
    (h_lengths : lengths_equal (distance C D) (distance A B))
    : (|AD|^2 = |AB| * |BC|) ↔ (angle_between C B D = 30) :=
    sorry

end triangle_obtuse_angle_l112_112333


namespace carry_20160000_in_2000_trips_l112_112252

/-- 
  There was a pile containing 20,160,000 grains of sand in a quarry. 
  In one trip, a truck carried away a quantity of sand that was some power of the number 9. 
  The proof shows that the truck can carry away the entire pile of sand in exactly 2,000 trips.
--/
def can_carry_all_sand_in_2000_trips (total_sand : ℕ) (num_trips : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), (∀ i, f i = 9 ^ (i + 1) ∨ f i = 1) ∧ (num_trips = 2000) ∧ (list.sum (list.of_fn f) = total_sand)

/-- 
  Proof of the specific problem that there are exactly 20,160,000 grains of sand,
  which can be carried away in 2000 trips where each trip is some power of 9.
--/
theorem carry_20160000_in_2000_trips : can_carry_all_sand_in_2000_trips 20160000 2000 :=
by
  sorry

end carry_20160000_in_2000_trips_l112_112252


namespace cos_2theta_plus_sin_2theta_l112_112902

theorem cos_2theta_plus_sin_2theta (θ : ℝ) (h : 3 * Real.sin θ = Real.cos θ) : 
  Real.cos (2 * θ) + Real.sin (2 * θ) = 7 / 5 :=
by
  sorry

end cos_2theta_plus_sin_2theta_l112_112902


namespace find_smallest_x_l112_112414

def smallest_x_satisfying_condition : Prop :=
  ∃ x : ℝ, x > 0 ∧ (⌊x^2⌋ - x * ⌊x⌋ = 8) ∧ x = 89 / 9

theorem find_smallest_x : smallest_x_satisfying_condition :=
begin
  -- proof goes here (not required for this task)
  sorry
end

end find_smallest_x_l112_112414


namespace saved_amount_percent_l112_112755

theorem saved_amount_percent (S : ℝ) :
    let saved_last_year := 0.06 * S
    let current_salary := 1.20 * S
    let saved_this_year := 0.05 * current_salary
    saved_this_year / saved_last_year * 100 = 100 := by
  let saved_last_year := 0.06 * S
  let current_salary := 1.20 * S
  let saved_this_year := 0.05 * current_salary
  have h1 : saved_last_year = 0.06 * S := rfl
  have h2 : current_salary = 1.20 * S := rfl
  have h3 : saved_this_year = 0.05 * current_salary := rfl
  -- Use the definitions
  calc
    saved_this_year / saved_last_year * 100
        = (0.05 * current_salary) / (0.06 * S) * 100 : by rw [h1, h3]
    ... = (0.05 * (1.20 * S)) / (0.06 * S) * 100 : by rw [h2]
    ... = (0.06 * S) / (0.06 * S) * 100 : by norm_num -- Simplify the fraction
    ... = 1 * 100 : by rw [div_self]; norm_num
    ... = 100 : by norm_num

end saved_amount_percent_l112_112755


namespace sugar_percentage_in_resulting_solution_l112_112179

noncomputable def initial_solution_weight := 100 -- weight in grams
noncomputable def initial_sugar_percentage := 10 -- percentage
noncomputable def replaced_solution_weight := initial_solution_weight / 4 -- replaced amount
noncomputable def replaced_solution_sugar_percentage := 34 -- percentage

theorem sugar_percentage_in_resulting_solution 
  (initial_solution_weight : ℝ) 
  (initial_sugar_percentage : ℝ) 
  (replaced_solution_weight : ℝ) 
  (replaced_solution_sugar_percentage : ℝ) 
  (total_weight : ℝ) 
  (final_sugar_weight : ℝ) 
  (resulting_sugar_percentage : ℝ) : 
  initial_solution_weight = 100 → 
  initial_sugar_percentage = 10 →
  replaced_solution_weight = 25 → 
  replaced_solution_sugar_percentage = 34 →
  final_sugar_weight / total_weight * 100 = 16 :=
begin
  sorry
end

end sugar_percentage_in_resulting_solution_l112_112179


namespace angle_A_eq_angle_BCE_angle_BFE_eq_2angle_C_l112_112989

section
variables {A B C D E F : Type}

-- Assume points and triangle properties
variables [Inhabited A] [Inhabited C] [Inhabited E]

def is_diameter (A C : Type) : Prop := sorry -- Define what it means for AC to be a diameter of the circle.
def is_right_triangle (A B C : Type) : Prop := sorry -- Define what it means for ABC to be a right triangle.
def is_hypotenuse (AB : Type) : Prop := sorry -- Define that AB is the hypotenuse of triangle ABC.
def intersects_circle (D E : Type) (circle : Type) : Prop := sorry -- Define the intersection points.
def is_tangent (E : Type) (F : Prop) : Prop := sorry -- Define the tangent at E cutting leg BC at point F.

-- Given conditions
axiom AC_diameter : is_diameter A C
axiom ABC_right_triangle : is_right_triangle A B C
axiom AB_hypotenuse : is_hypotenuse A B
axiom DE_circle_intersection : intersects_circle D E A -- Assume the circle

-- Statements to be proven
theorem angle_A_eq_angle_BCE : ∀ (β : Type), angle A = angle BCE := sorry
theorem angle_BFE_eq_2angle_C : ∀ (β : Type), angle BFE = 2 * angle C := sorry
end

end angle_A_eq_angle_BCE_angle_BFE_eq_2angle_C_l112_112989


namespace last_box_probability_l112_112088

noncomputable def probability_last_box_only_ball : ℝ :=
  let n : ℕ := 100 in
  let p : ℝ := 1 / n in
  (n * p * (1 - p)^(n - 1))

theorem last_box_probability : abs (probability_last_box_only_ball - 0.3697) < 0.0005 := 
  sorry

end last_box_probability_l112_112088


namespace last_box_one_ball_probability_l112_112100

/-- The probability that the last box will contain exactly one of 100 randomly distributed balls
is approximately 0.370. -/
theorem last_box_one_ball_probability :
  let n : ℕ := 100 in
  let p : ℚ := 1 / 100 in
  let probability : ℚ := (n.choose 1) * (p) * ((1 - p) ^ (n - 1)) in
  probability ≈ 0.370 :=
by
  let n : ℕ := 100 
  let p : ℚ := 1 / 100
  let probability : ℚ := (n.choose 1) * (p) * ((1 - p) ^ (n - 1))
  sorry

end last_box_one_ball_probability_l112_112100


namespace number_of_valid_d_l112_112683

def is_digit (d : ℕ) : Prop := d >= 0 ∧ d <= 9
def is_even_digit (n : ℕ) : Prop := is_digit n ∧ n % 2 = 0

theorem number_of_valid_d (d n : ℕ) :
  (is_digit d) → (is_even_digit n) →
  (3 + d / 10 + 5 / 100 + n / 10000 > 3.056) → (d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9) → 
  ∃ values, values.card = 5 :=
sorry

end number_of_valid_d_l112_112683


namespace triangle_side_sum_l112_112118

theorem triangle_side_sum :
  ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (angle_A angle_B : ℝ)
  (side_a : ℝ),
  angle_A = 45 ∧ angle_B = 60 ∧ side_a = 8 →
  sum_of_remaining_sides angle_A angle_B side_a = 8 + 4 * real.sqrt 2 + 4 * real.sqrt 3 :=
by
  sorry

end triangle_side_sum_l112_112118


namespace smallest_area_of_triangle_ABC_l112_112640

variables (s : ℝ)

def point_A : ℝ × ℝ × ℝ := (-2, 1, 3)
def point_B : ℝ × ℝ × ℝ := (2, 3, 4)
def point_C : ℝ × ℝ × ℝ := (s, 0, 2)

theorem smallest_area_of_triangle_ABC : ∃ s : ℝ, let vector_AB := (4, 2, 1)
  let vector_AC := (s + 2, -1, -1)
  let cross_product := (vector_AB.2 * vector_AC.3 - vector_AB.3 * vector_AC.2,
                        vector_AB.3 * vector_AC.1 - vector_AB.1 * vector_AC.3,
                        vector_AB.1 * vector_AC.2 - vector_AB.2 * vector_AC.1)
  let magnitude := Real.sqrt (cross_product.1 ^ 2 + cross_product.2 ^ 2 + cross_product.3 ^ 2)
  let area := magnitude / 2
  in area = 1 / 2 := sorry

end smallest_area_of_triangle_ABC_l112_112640


namespace variable_intersection_points_l112_112356

variable {ℝ : Type}
variable (B : ℝ) (hB : 0 < B)

def cubic_graph (x y : ℝ) : Prop := y = B * x^3 - x
def circle_graph (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

theorem variable_intersection_points : 
  (∃ (x y : ℝ), cubic_graph B x y ∧ circle_graph x y) → 
  ∃ (n : ℕ), (∀ x y, cubic_graph B x y → circle_graph x y → x^2 + (B * x^3 - x - 2)^2 - 4 = 0) ∧ 
            (1 ≤ n ∧ n ≤ 6) :=
by sorry

end variable_intersection_points_l112_112356


namespace remaining_water_correct_l112_112981

-- Define the constants given in the problem
def initial_water : ℚ := 3
def used_water : ℚ := 4 / 3
def remaining_water_fraction : ℚ := initial_water - used_water
def remaining_water_decimal : ℚ := 5 / 3 -- numerical fraction for the decimal Answer

-- Theorem to prove:
theorem remaining_water_correct :
  remaining_water_fraction = 5 / 3 ∧ remaining_water_decimal ≈ 1.67 :=
by
  sorry

end remaining_water_correct_l112_112981


namespace add_congruence_l112_112908

variable (a b c d m : ℤ)

theorem add_congruence (h₁ : a ≡ b [ZMOD m]) (h₂ : c ≡ d [ZMOD m]) : (a + c) ≡ (b + d) [ZMOD m] :=
sorry

end add_congruence_l112_112908


namespace Gerald_initial_notebooks_l112_112136

variable (J G : ℕ)

theorem Gerald_initial_notebooks (h1 : J = G + 13)
    (h2 : J - 5 - 6 = 10) :
    G = 8 :=
sorry

end Gerald_initial_notebooks_l112_112136


namespace max_red_green_alternations_l112_112259

def painter_problem := sorry -- This is a placeholder definition for the problem structure

theorem max_red_green_alternations : 
  ∀ (sections : ℕ), 
  (sections = 100) → 
  (alternating_days : ℕ → ℕ) → 
  (alternating_days 0 = 1) ∧ (alternating_days 1 = 2) ∧
  (∀ n, alternating_days n = n % 2 + 1) →
  (∀ (first_restriction : ℕ → bool), -- First painter restriction
  (first_restriction 0 = tt) ∧
  (first_restriction (n + 1) = (¬ first_restriction n)) → -- Alternating strategy
  (∀ (second_restriction : ℕ → bool), -- Second painter restriction
  -- Second painter can paint any section with any color
  first_restriction 1 = tt ∧ second_restriction 0 = first_restriction 0) →
  ∃ red_green_alternations, red_green_alternations = 49 :=
sorry

end max_red_green_alternations_l112_112259


namespace unique_positive_integer_solutions_l112_112827

theorem unique_positive_integer_solutions : 
  ∀ (m n : ℕ), 0 < m ∧ 0 < n ∧ 7 ^ m - 3 * 2 ^ n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) :=
by sorry

end unique_positive_integer_solutions_l112_112827


namespace janet_total_action_figures_l112_112630

theorem janet_total_action_figures :
  let initial_count := 10
  let sold_count := 6
  let new_count := 4
  -- After selling and buying
  let after_transaction_count := initial_count - sold_count + new_count
  let brother_addition := 2 * after_transaction_count
  -- Total count after brother's addition
  after_transaction_count + brother_addition = 24 := 
by
  let initial_count := 10
  let sold_count := 6
  let new_count := 4
  let after_transaction_count := initial_count - sold_count + new_count
  let brother_addition := 2 * after_transaction_count
  calc 
    after_transaction_count + brother_addition
      = (initial_count - sold_count + new_count) + (2 * (initial_count - sold_count + new_count)) : by rfl
  ... = 8 + (2 * 8) : by rfl
  ... = 24 : by rfl

end janet_total_action_figures_l112_112630


namespace part1_part2_l112_112623

theorem part1 (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : A = 2 * π / 3 :=
sorry

theorem part2 (b c : ℝ) (h1 : sin (2 * π / 3) ^ 2 - sin b ^ 2 - sin c ^ 2 = sin b * sin c)
  (h2 : b + c = 2 * sqrt 3) : 3 * 2 + b + c = 3 + 2 * sqrt 3 :=
sorry

end part1_part2_l112_112623


namespace part1_part2_l112_112617

-- Part (1): Prove that A = 2π/3 given the trigonometric condition.
theorem part1 (A B C : ℝ) (h_condition : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : 
  A = 2 * π / 3 :=
sorry

-- Part (2): Prove that the maximum perimeter is 3 + 2√3 given BC = 3 and A = 2π/3.
theorem part2 (B C : angle) (h_BC : 3 = 3) (h_A : A = 2 * π / 3) :
  (3 + (sqrt 3) * 2 ≤ perimeter (triangle.mk 3 B C)) :=
sorry

end part1_part2_l112_112617


namespace complement_of_P_subset_Q_l112_112294

-- Definitions based on conditions
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x > -1}

-- Theorem statement to prove the correct option C
theorem complement_of_P_subset_Q : {x | ¬ (x < 1)} ⊆ {x | x > -1} :=
by {
  sorry
}

end complement_of_P_subset_Q_l112_112294


namespace division_of_sums_is_eight_l112_112140

theorem division_of_sums_is_eight :
  let karen_sum := 2 * ((399 / 2) * (2 + 400))
  let tom_sum := (200 / 2) * (1 + 200)
  karen_sum / tom_sum = 8 :=
by
  let karen_sum := 2 * ((399 / 2) * (2 + 400)) -- Calculating Karen's sum
  let tom_sum := (200 / 2) * (1 + 200) -- Calculating Tom's sum
  have h1 : karen_sum = 160598 := by sorry
  have h2 : tom_sum = 20100 := by sorry
  calc
    (karen_sum / tom_sum) = 160598 / 20100 : by rw [h1, h2]
    ... = 8 : by sorry

end division_of_sums_is_eight_l112_112140


namespace total_meals_sold_l112_112343

-- Definitions based on the conditions
def ratio_kids_adult := 2 / 1
def kids_meals := 8

-- The proof problem statement
theorem total_meals_sold : (∃ adults_meals : ℕ, 2 * adults_meals = kids_meals) → (kids_meals + 4 = 12) := 
by 
  sorry

end total_meals_sold_l112_112343


namespace correct_statement_l112_112798

theorem correct_statement :
  let cond1 : Prop := ∀ x : ℚ, x > 0 → (∃ m n : ℤ, x = m / n ∧ n > 0 ∧ Nat.gcd m n = 1)
  let cond2 : Prop := ∀ x : ℤ, abs x = -x → x ≤ 0
  let cond3 : Prop := ∀ a b : ℝ, a + b < 0 ∧ a * b > 0 → a < 0 ∧ b < 0
  let cond4 : Prop := ∀ x y : ℚ, x - y < x
  cond1 ∧ ¬cond2 ∧ cond3 ∧ ¬cond4 → cond3 :=
by
sorrry

end correct_statement_l112_112798


namespace count_paths_without_diagonals_count_paths_with_diagonals_l112_112061

-- Define the structure of the grid and movement rules
def grid_7x7 : Type := fin 7 × fin 7

def is_center (p : grid_7x7) : Prop := p = (⟨3, by norm_num⟩, ⟨3, by norm_num⟩)

def adjacent (p q : grid_7x7) : Prop :=
  let ⟨px, py⟩ := p in
  let ⟨qx, qy⟩ := q in
  (abs (px - qx) ≤ 1) ∧ (abs (py - qy) ≤ 1)

-- Statement for vertical and horizontal movements
theorem count_paths_without_diagonals :
  ∃ n : ℕ, n = 45760 :=
begin
  sorry -- Proof omitted
end

-- Statement for including diagonal movements
theorem count_paths_with_diagonals :
  ∃ n : ℕ, n = 91520 :=
begin
  sorry -- Proof omitted
end

end count_paths_without_diagonals_count_paths_with_diagonals_l112_112061


namespace sum_of_areas_of_super_cool_rectangles_l112_112322

def is_super_cool (a b : ℕ) : Prop :=
  a * b = 6 * (a + b)

theorem sum_of_areas_of_super_cool_rectangles :
  ∑ (a, b) in { (a, b) : ℕ × ℕ | is_super_cool a b }  a * b = 942 :=
sorry

end sum_of_areas_of_super_cool_rectangles_l112_112322


namespace expression_is_integer_l112_112671

theorem expression_is_integer (n : ℕ) : 
  (3 ^ (2 * n) / 112 - 4 ^ (2 * n) / 63 + 5 ^ (2 * n) / 144) = (k : ℤ) :=
sorry

end expression_is_integer_l112_112671


namespace geometric_sequence_a_eq_2_l112_112659

theorem geometric_sequence_a_eq_2 (a : ℝ) (h1 : ¬ a = 0) (h2 : (2 * a) ^ 2 = 8 * a) : a = 2 :=
by {
  sorry -- Proof not required, only the statement.
}

end geometric_sequence_a_eq_2_l112_112659


namespace fourth_roots_of_unity_quadratic_roots_l112_112816

theorem fourth_roots_of_unity_quadratic_roots :
  let roots_of_unity := {z : ℂ | z^4 = 1},
      quadratic_eq_sat (p : ℤ) (z : ℂ) := z^2 + (p : ℂ) * z + 1 = 0 
  in ∃ (num_roots : ℤ), num_roots = 4 ∧
       (∀ z, z ∈ roots_of_unity → ∃ p, p ∈ {-2, 0, 2} ∧ quadratic_eq_sat p z) :=
by
  sorry

end fourth_roots_of_unity_quadratic_roots_l112_112816


namespace greatest_matching_pairs_l112_112663

theorem greatest_matching_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) 
  (h_init : initial_pairs = 26) (h_lost : lost_shoes = 9) :
  ∃ remaining_pairs : ℕ, remaining_pairs = initial_pairs - lost_shoes ∧ remaining_pairs = 17 :=
by {
  use initial_pairs - lost_shoes,
  split,
  { rw [h_init, h_lost],
    refl },
  { rw [h_init, h_lost],
    norm_num }
}

end greatest_matching_pairs_l112_112663


namespace leftmost_three_nonzero_digits_of_arrangements_l112_112023

-- Definitions based on the conditions
def num_rings := 10
def chosen_rings := 6
def num_fingers := 5

-- Calculate the possible arrangements
def arrangements : ℕ := Nat.choose num_rings chosen_rings * Nat.factorial chosen_rings * Nat.choose (chosen_rings + (num_fingers - 1)) (num_fingers - 1)

-- Find the leftmost three nonzero digits
def leftmost_three_nonzero_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.reverse.takeWhile (· > 0)).reverse.take 3
  |> List.foldl (· + · * 10) 0
  
-- The main theorem to prove
theorem leftmost_three_nonzero_digits_of_arrangements :
  leftmost_three_nonzero_digits arrangements = 317 :=
by
  sorry

end leftmost_three_nonzero_digits_of_arrangements_l112_112023


namespace general_term_formula_minimum_T_n_value_min_value_at_one_l112_112035

def arithmetic_sequence (a : ℕ → ℕ) (a₁ d : ℕ) :=
  a 1 = a₁ ∧ ∀ n, a (n+1) = a n + d

noncomputable def a_n : ℕ → ℕ
| 1     := 1
| (n+1) := a_n n + 2

noncomputable def b_n (n : ℕ) : ℚ :=
  1 / ((a_n n : ℚ) * (a_n (n - 1) : ℚ))

noncomputable def T_n (n : ℕ) : ℚ :=
  (finset.range n).sum (λ k, b_n (k + 1))

theorem general_term_formula :
  ∀ n, a_n n = 2 * n - 1 :=
by sorry

theorem minimum_T_n_value :
  ∀ n, T_n 1 ≤ T_n n :=
by sorry

theorem min_value_at_one :
  T_n 1 = 1 / 3 :=
by sorry

end general_term_formula_minimum_T_n_value_min_value_at_one_l112_112035


namespace prob_at_least_one_even_prob_not_adjacent_l112_112923

-- Define the conditions for the contest.
namespace Contest

-- Define students and numbering
inductive Student : Type
| A | B | C | D | E

open Student

-- Define the set of sequence numbers.
def seqNumbers : Finset ℕ := {1, 2, 3, 4, 5}

-- Define a random performance order (permutation of the sequence numbers).
def performanceOrder (s : Student) : ℕ := sorry  -- To be defined as a random permutation

-- Define the main theorem statements based on the problem and solution.
theorem prob_at_least_one_even :
  ∃ (p : ℚ), p = (7 / 10) ∧
  (1 - (card (seqNumbers.filter (λ n, n % 2 = 1))).choose 2 / seqNumbers.choose 5.to_nat) = p :=
sorry

theorem prob_not_adjacent :
  ∃ (p : ℚ), p = (3 / 5) ∧
  (1 - (card (seqNumbers.erase 1).choose 4.to_nat * (2.choose 2 / seqNumbers.choose 5.to_nat))) = p :=
sorry

end Contest

end prob_at_least_one_even_prob_not_adjacent_l112_112923


namespace original_number_l112_112906

theorem original_number {
  -- Define the given condition
  (h1 : 204 ÷ x = 16) : 
  -- Prove that x = 12.75
  x = 12.75 := sorry

end original_number_l112_112906


namespace juniors_in_program_l112_112954

theorem juniors_in_program (J S x y : ℕ) (h1 : J + S = 40) 
                           (h2 : x = y) 
                           (h3 : J / 5 = x) 
                           (h4 : S / 10 = y) : J = 12 :=
by
  sorry

end juniors_in_program_l112_112954


namespace distance_B_from_line_l112_112712

-- Given three one-inch squares placed on a line, with the center square rotated 60 degrees.
variables (side_length : ℝ)
variables (theta : ℝ)
variables (hB : ℝ)

-- Side length of each square is 1 inch.
def side_length := 1

-- Angle of rotation is 60 degrees.
def theta := π / 3

-- The top vertex of the rotated square.
def height_B := (side_length : ℝ) / 2 + (side_length * (Real.sqrt 2 / 2) * Real.sin theta) / 2

-- Prove that the point B is a specific distance from the line.
theorem distance_B_from_line : height_B = (2 + Real.sqrt 6) / 4 :=
sorry

end distance_B_from_line_l112_112712


namespace PQ_bisects_BD_l112_112945

variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]
variables {A B C D P Q M N : Point}

def convex_quadrilateral (A B C D : Point) : Prop := sorry
def midpoint (P A B : Point) : Prop := 2 • P = A + B
def bisects (line P Q : Point) (diagonal A C : Point) : Prop := 
  ∃ M, midpoint M A C ∧ (line.contains M)
def line_contains_midpoint (P Q : Point) (mid : Point) : Prop := sorry

-- The theorem we want to prove:
theorem PQ_bisects_BD 
  (h1 : convex_quadrilateral A B C D)
  (h2 : midpoint P A B)
  (h3 : midpoint Q C D)
  (h4 : bisects (P, Q) (A, C))
  : bisects (P, Q) (B, D) := 
begin
  sorry
end

end PQ_bisects_BD_l112_112945


namespace estimate_white_balls_l112_112112

variable (num_balls total_trials white_draws: ℕ)
variable (total_balls: 20) (trials: 100) (white_draws: 40)

theorem estimate_white_balls
  (HTotal: total_balls = 20)
  (HTrials: trials = 100)
  (HWhiteDraws: white_draws = 40) :
  (white_draws: ∕ trials) * total_balls = 8 := by
sor

end estimate_white_balls_l112_112112


namespace find_s8_minus_s5_l112_112486

variable (a_n : ℕ → ℕ)
variable (s_n : ℕ → ℕ)
variable (diff : ℕ)

-- a_n is an arithmetic sequence
axiom arith_seq : ∀ n, a_n (n+1) = a_n n + diff

-- s_n is the sum of the first n terms of a_n
axiom sum_first_n : ∀ n, s_n n = (Σ i in range n, a_n i)

-- a_7 = 4
axiom a_7_eq_4 : a_n 7 = 4

-- equivalent proof problem
theorem find_s8_minus_s5 : s_n 8 - s_n 5 = 12 := sorry

end find_s8_minus_s5_l112_112486


namespace range_of_a_l112_112846

noncomputable def A : set ℝ := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}

theorem range_of_a (a : ℝ) (h : a ∈ A) : a ∈ set.Icc (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_a_l112_112846


namespace non_overlapping_lines_same_plane_parallel_l112_112967

-- Define the conditions
def same_plane (L1 L2 : Type) : Prop := -- L1 and L2 are lines in the same plane
sorry 

def non_overlapping (L1 L2 : Type) : Prop := -- L1 and L2 do not overlap
sorry

-- The theorem to be proven under the given conditions
theorem non_overlapping_lines_same_plane_parallel (L1 L2 : Type)
  (h_plane : same_plane L1 L2) (h_non_overlap : non_overlapping L1 L2) :
  ¬ (∃ p : Type, (p ∈ L1 ∧ p ∈ L2)) → (parallel L1 L2) :=
sorry

end non_overlapping_lines_same_plane_parallel_l112_112967


namespace find_200th_number_l112_112338

def is_valid_permutation (n : ℕ) : Prop :=
  n.digits = [1, 2, 3, 4, 5, 6, 7]

def not_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 ≠ 0

noncomputable def NthValidNumber (n : ℕ) : ℕ :=
  (list.permutations [1, 2, 3, 4, 5, 6, 7]).filter (λ perm, (nat.of_digits 10 perm) % 5 ≠ 0) |>.sort (<) |>.nth (n - 1) |>.get_or_else 0

theorem find_200th_number :
  NthValidNumber 200 = 4315672 :=
sorry

end find_200th_number_l112_112338


namespace num_ways_to_return_to_city5_l112_112707

def possibleWaysToReturnToCity5 : ℕ := 21

def problem_statement (n : ℕ) : Prop :=
  n = possibleWaysToReturnToCity5

theorem num_ways_to_return_to_city5 : ∃ n, problem_statement n :=
begin
  existsi 21,
  unfold problem_statement,
  refl,
end

end num_ways_to_return_to_city5_l112_112707


namespace sum_even_odd_functions_l112_112478

theorem sum_even_odd_functions (f g : ℝ → ℝ) (h₁ : ∀ x, f (-x) = f x) (h₂ : ∀ x, g (-x) = -g x) (h₃ : ∀ x, f x - g x = x^3 + x^2 + 1) : 
  f 1 + g 1 = 1 := 
by 
  sorry

end sum_even_odd_functions_l112_112478


namespace find_m_and_2A_minus_B_l112_112898

variable (x m : ℝ)

def A : ℝ := x^2 + 2*x + 3
def B : ℝ := 2*x^2 - m*x + 2

theorem find_m_and_2A_minus_B 
  (h : ∀ x, (4 + m) * x + 4 = 4) : 
  m = -4 ∧ (2 * A - B = 4) :=
by 
  sorry

end find_m_and_2A_minus_B_l112_112898


namespace intersection_A_B_l112_112164

theorem intersection_A_B (A B : Set ℕ) (hA : A = {1, 2, 3, 4, 5}) (hB : B = {x ∈ ℕ | (x - 1) * (x - 4) < 0}) :
  A ∩ B = {2, 3} :=
by
  rw [hA, hB]
  sorry

end intersection_A_B_l112_112164


namespace trigonometric_identity_l112_112833

theorem trigonometric_identity :
  (sin (20 * Real.pi / 180) * cos (15 * Real.pi / 180) + cos (160 * Real.pi / 180) * cos (105 * Real.pi / 180)) /
  (sin (25 * Real.pi / 180) * cos (10 * Real.pi / 180) + cos (155 * Real.pi / 180) * cos (95 * Real.pi / 180)) = 1 := 
by
  sorry

end trigonometric_identity_l112_112833


namespace sum_of_favorite_numbers_l112_112169

def Glory_favorite_number : ℕ := 450
def Misty_favorite_number : ℕ := Glory_favorite_number / 3

theorem sum_of_favorite_numbers : Misty_favorite_number + Glory_favorite_number = 600 :=
by
  sorry

end sum_of_favorite_numbers_l112_112169


namespace sufficient_not_necessary_l112_112292

theorem sufficient_not_necessary (p q : Prop) (h : p ∧ q) : (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) :=
by
  sorry

end sufficient_not_necessary_l112_112292


namespace smallest_positive_real_x_l112_112392

theorem smallest_positive_real_x :
  ∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧ x = 89 / 9 :=
by
  sorry

end smallest_positive_real_x_l112_112392


namespace find_chord_line_eq_l112_112304

-- Definitions based on conditions
def ellipse := {p : ℝ × ℝ | let ⟨x, y⟩ := p in x^2 / 16 + y^2 / 4 = 1}
def point_M := (2, 1)

-- Theorem stating the problem
theorem find_chord_line_eq (A B : ℝ × ℝ)
  (hA : A ∈ ellipse) (hB : B ∈ ellipse)
  (hM : (A.1 + B.1) / 2 = point_M.1 ∧ (A.2 + B.2) / 2 = point_M.2) :
  ∃ k : ℝ, ∃ b : ℝ, (∀ x y : ℝ, y = k * x + b ↔ x + 2 * y - 4 = 0) :=
sorry

end find_chord_line_eq_l112_112304


namespace find_a_l112_112043

-- Given function f(x) = 4 * log x + a * x^2 - 6 * x + b
def f (x : ℝ) (a b : ℝ) : ℝ := 4 * log x + a * x^2 - 6 * x + b

-- x = 2 is an extreme value point of f(x)
def is_extreme_value_point (a b : ℝ) : Prop := (deriv (λ x, f x a b) 2 = 0)

theorem find_a (b : ℝ) (h : is_extreme_value_point 1 b) : ∀ a, a = 1 :=
  by
    intro a
    sorry

end find_a_l112_112043


namespace right_triangle_hypotenuse_square_sum_l112_112122

theorem right_triangle_hypotenuse_square_sum (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (h : isRightTriangle A B C)
  (h1 : distance A B = 4)
  (h2 : distance A B ^ 2 = distance B C ^ 2 + distance A C ^ 2) :
  distance A B ^ 2 + distance B C ^ 2 + distance A C ^ 2 = 32 := by
sorry

end right_triangle_hypotenuse_square_sum_l112_112122


namespace remainder_of_s_l112_112519

open Nat m Nat binomial

theorem remainder_of_s (n : ℕ) (h : n > 0) : 
  let S := (∑ i in range n, 7^(n-i) * binom n i)
  if n % 2 = 0 then S % 9 = 0 else S % 9 = 7 := 
by
  sorry

end remainder_of_s_l112_112519


namespace triangle_A_value_and_max_perimeter_l112_112579

theorem triangle_A_value_and_max_perimeter (A B C a b c : ℝ) 
  (h1 : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C)
  (h2 : a = 3) :
  (A = 2 * Real.pi / 3) ∧ (a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by
  sorry

end triangle_A_value_and_max_perimeter_l112_112579


namespace value_of_S_l112_112156

noncomputable def f : (ℕ → ℕ) :=
  λ n, if n = 1 then 1 else (n-1) * f (n-1)

theorem value_of_S (R : ℕ) (f(1) ≠ 0) :
  R = 11 →
  S = (R-2) * (R-3) :=
by
  let S := λ R, (f R) / ((R-1) * (f (R-3)))
  intro hR
  sorry

end value_of_S_l112_112156


namespace smallest_positive_real_x_l112_112395

theorem smallest_positive_real_x :
  ∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧ x = 89 / 9 :=
by
  sorry

end smallest_positive_real_x_l112_112395


namespace cylinder_volume_not_occupied_by_cones_l112_112711

theorem cylinder_volume_not_occupied_by_cones (r h : ℝ) (n : ℕ) (R H : ℝ) (H1 : r = 10) (H2 : h = 15) (H3 : n = 3) (H4 : R = 10) (H5 : H = 30) :
  let V_cylinder := π * R^2 * H,
      V_cone := (1/3) * π * r^2 * h,
      V_total_cones := n * V_cone,
      V_unoccupied := V_cylinder - V_total_cones
  in V_unoccupied = 1500 * π :=
sorry

end cylinder_volume_not_occupied_by_cones_l112_112711


namespace tetrahedron_volume_l112_112807

-- Definition of the radius R and the volume V given R.
variable (R : ℝ)

-- The theorem statement
theorem tetrahedron_volume (h : 0 ≤ R) : 
  ∃ V : ℝ, V = (R^3 * Real.sqrt 6) / 4 :=
begin
  use (R^3 * Real.sqrt 6) / 4,
  sorry
end

end tetrahedron_volume_l112_112807


namespace correct_graph_of_g_neg_x_l112_112355

def g (x : ℝ) : ℝ :=
  if -5 ≤ x ∧ x ≤ -2 then -3 - x / 2
  else if -2 ≤ x ∧ x ≤ 1 then sqrt (9 - (x + 2)^2) - 3
  else if 1 ≤ x ∧ x ≤ 4 then 3 * (x - 1)
  else 0

theorem correct_graph_of_g_neg_x :
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 5 → g(-x) = -3 + x / 2) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → g(-x) = sqrt (9 - (x - 2)^2) - 3) ∧
  (∀ x : ℝ, -4 ≤ x ∧ x ≤ -1 → g(-x) = 3 * (x + 1)) :=
by {
  -- proof omitted
  sorry
}

end correct_graph_of_g_neg_x_l112_112355


namespace cos_product_l112_112810

theorem cos_product : 
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) * (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12)) = 1 / 8 := 
by
  sorry

end cos_product_l112_112810


namespace license_plate_count_l112_112511

def is_valid_license_plate (plate : String) : Prop :=
  plate.length = 3 ∧
  plate[0].isDigit ∧
  plate[2].isAlpha ∧
  ∃ (i : Fin 3) (j : Fin 3), i ≠ j ∧ plate[i] = plate[j]

theorem license_plate_count : 
  let count := 520 in
  ∃ (plates : List String), 
    (∀ plate ∈ plates, is_valid_license_plate plate) ∧
    plates.length = count := 
by
  sorry

end license_plate_count_l112_112511


namespace joe_used_fraction_paint_in_first_week_l112_112978

variable (x : ℝ) -- Define the fraction x as a real number

-- Given conditions
def given_conditions : Prop := 
  let total_paint := 360
  let paint_first_week := x * total_paint
  let remaining_paint := (1 - x) * total_paint
  let paint_second_week := (1 / 2) * remaining_paint
  paint_first_week + paint_second_week = 225

-- The theorem to prove
theorem joe_used_fraction_paint_in_first_week (h : given_conditions x) : x = 1 / 4 :=
sorry

end joe_used_fraction_paint_in_first_week_l112_112978


namespace monotonicity_of_f_on_interval_range_of_k_l112_112883

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 4)
noncomputable def g (k x : ℝ) : ℝ := k*x^2 + 2*k*x + 1 

-- Part (1) Monotonicity of f(x) on [-2,2]
theorem monotonicity_of_f_on_interval : 
  monotone_on f (set.Icc (-2 : ℝ) (2 : ℝ)) := sorry

-- Part (2) Range of k to ensure f(x1) = g(x2) for some x2 in [-1,2] for all x1 in [-2,2]
theorem range_of_k (k : ℝ) (k_nonzero : k ≠ 0) :
  (forall x1, x1 ∈ set.Icc (-2 : ℝ) 2 → exists x2, x2 ∈ set.Icc (-1 : ℝ) 2 ∧ f x1 = g k x2) ↔ 
  k ≤ (-5 / 32) ∨ k ≥ (5 / 4) := sorry

end monotonicity_of_f_on_interval_range_of_k_l112_112883


namespace gcd_7488_12467_eq_39_l112_112830

noncomputable def gcd_7488_12467 : ℕ := Nat.gcd 7488 12467

theorem gcd_7488_12467_eq_39 : gcd_7488_12467 = 39 :=
sorry

end gcd_7488_12467_eq_39_l112_112830


namespace problem_1_and_2_l112_112599

-- Definition of the problem in Lean 4
theorem problem_1_and_2 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin (A)^2 - sin (B)^2 - sin (C)^2 = sin (B) * sin (C))
  (h2 : BC = 3)
  (h3 : triangle ABC)
  (h4 : a = side_length (opposite A))
  (h5 : b = side_length (opposite B))
  (h6 : c = side_length (opposite C)) :
  A = 2 * real.pi / 3 ∧
  (a + b + c ≤ 3 + 2 * real.sqrt 3) :=
sorry

end problem_1_and_2_l112_112599


namespace value_of_expression_l112_112068

variable {x : ℝ}

theorem value_of_expression (h : x^2 - 3 * x = 2) : 3 * x^2 - 9 * x - 7 = -1 := by
  sorry

end value_of_expression_l112_112068


namespace sum_of_areas_of_super_cool_rectangles_l112_112321

def is_super_cool (a b : ℕ) : Prop :=
  a * b = 6 * (a + b)

theorem sum_of_areas_of_super_cool_rectangles :
  ∑ (a, b) in { (a, b) : ℕ × ℕ | is_super_cool a b }  a * b = 942 :=
sorry

end sum_of_areas_of_super_cool_rectangles_l112_112321


namespace choose_student_A_based_on_mean_and_std_deviation_l112_112763

-- Define the scores of Students A and B
def scores_A : List ℚ := [82, 81, 79, 78, 95, 88, 93, 84]
def scores_B : List ℚ := [92, 95, 80, 75, 83, 80, 90, 85]

-- Function to calculate the mean of a list of rational numbers
def mean (scores : List ℚ) : ℚ :=
  scores.sum / scores.length

-- Function to calculate the variance of a list of rational numbers
def variance (scores : List ℚ) : ℚ :=
  let m := mean scores
  (scores.map (λx, (x - m)^2)).sum / scores.length

-- Function to calculate the standard deviation
def std_deviation (scores : List ℚ) : ℚ :=
  Real.sqrt (variance scores)

-- Define the proof problem
theorem choose_student_A_based_on_mean_and_std_deviation :
  let mean_A := mean scores_A
  let std_dev_A := std_deviation scores_A
  let mean_B := mean scores_B
  let std_dev_B := std_deviation scores_B
  -- Assume Student A is more appropriate based on statistical metrics
  (mean_A, std_dev_A) < (mean_B, std_dev_B) →
  true := sorry

end choose_student_A_based_on_mean_and_std_deviation_l112_112763


namespace initial_ratio_milk_water_l112_112781

-- Define the initial conditions
variables (M W : ℕ) (h_volume : M + W = 115) (h_ratio : M / (W + 46) = 3 / 4)

-- State the theorem to prove the initial ratio of milk to water
theorem initial_ratio_milk_water (h_volume : M + W = 115) (h_ratio : M / (W + 46) = 3 / 4) :
  (M * 2 = W * 3) :=
by
  sorry

end initial_ratio_milk_water_l112_112781


namespace bisects_diagonals_l112_112928

-- Define the data structure for a convex quadrilateral
structure ConvexQuadrilateral (α : Type*) :=
(A B C D : α)

-- Define midpoints of line segments
def midpoint {α : Type*} [Add α] [Div α] [Nonempty α] (A B : α) : α :=
(A + B) / 2

-- Main theorem stating the problem
theorem bisects_diagonals
  {α : Type*} [AddCommGroup α] [Module ℝ α] (quad : ConvexQuadrilateral α)
  (P Q : α)
  (hP : P = midpoint quad.A quad.B)
  (hQ : Q = midpoint quad.C quad.D)
  (hPQ : ∃ M, M = midpoint quad.A quad.C ∧ M ∈ line_through P Q) :
  ∃ N, N = midpoint quad.B quad.D ∧ N ∈ line_through P Q :=
sorry

lemma line_through (P Q : α) : Prop :=
∃ (λ1 λ2 : ℝ), P + λ1 • (Q - P) = Q + λ2 • (P - Q)

end bisects_diagonals_l112_112928


namespace jackies_free_time_l112_112970

-- Define the conditions
def hours_working : ℕ := 8
def hours_sleeping : ℕ := 8
def hours_exercising : ℕ := 3
def total_hours_in_day : ℕ := 24

-- The statement to be proven
theorem jackies_free_time : total_hours_in_day - (hours_working + hours_sleeping + hours_exercising) = 5 :=
by 
  rw [total_hours_in_day, hours_working, hours_sleeping, hours_exercising]
  -- 24 - (8 + 8 + 3) = 5
  sorry

end jackies_free_time_l112_112970


namespace slope_of_tangent_line_l112_112872

theorem slope_of_tangent_line (f : ℝ → ℝ) (f_deriv : ∀ x, deriv f x = f x) (h_tangent : ∃ x₀, f x₀ = x₀ * deriv f x₀ ∧ (0 < f x₀)) :
  ∃ k, k = Real.exp 1 :=
by
  sorry

end slope_of_tangent_line_l112_112872


namespace find_angle_A_max_perimeter_triangle_l112_112553

-- Part 1: Prove the value of angle A
theorem find_angle_A (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter for BC = 3
theorem max_perimeter_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (A : ℝ) (hA : A = 2 * real.pi / 3) :
  let b := 2 * real.sqrt 3 * sin B,
      c := 2 * real.sqrt 3 * sin C,
      perimeter := 3 + b + c
  in ∀ d : ℝ, -real.pi / 6 < d ∧ d < real.pi / 6 → 
     B + C = real.pi / 3 → 
     perimeter ≤ (3 + 2 * real.sqrt 3) := sorry

end find_angle_A_max_perimeter_triangle_l112_112553


namespace exists_pos_integers_l112_112296

theorem exists_pos_integers (k : ℕ) (h : 2 ≤ k ∧ k ≤ 100) :
  ∃ (b : ℕ → ℕ), (∀ i, 2 ≤ i ∧ i ≤ 101 → 0 < b i) ∧
    (b 2)^2 + (b 3)^3 + ... + (b k)^k = (b (k+1))^(k+1) + (b (k+2))^(k+2) + ... + (b 101)^(101) :=
    sorry

end exists_pos_integers_l112_112296


namespace interval_monotonicity_f_range_of_a_intersection_l112_112492

-- Definitions for the problem
def f (x : ℝ) (a : ℝ) : ℝ := a^x / x - Real.log a
def g (x : ℝ) : ℝ := (Real.exp 1 + 1) / (Real.exp 1 * x)

-- Conditions
axiom a_pos (a : ℝ) : a > 0
axiom a_ne_one (a : ℝ) : a ≠ 1
axiom a_eq_e : ∀ (x : ℝ), f x (Real.exp 1) = Real.exp x / x - 1

-- Proving the intervals for monotonicity when a = e
theorem interval_monotonicity_f :
  (∀ x > 1, (∂ (λ x, f x (Real.exp 1)) x) > 0) ∧ 
  (∀ x < 0, (∂ (λ x, f x (Real.exp 1)) x) < 0) ∧ 
  (∀ x > 0 ∧ x < 1, (∂ (λ x, f x (Real.exp 1)) x) < 0) :=
by 
  sorry

-- Proving the range of values for a so that the curves intersect at two points
theorem range_of_a_intersection : 
  (∀ a, ((a > 0 ∧ a ≤ 1 / Real.exp 1) ∨ (a ≥ Real.exp 1)) →
  ∃ x ∈ [-1,0) ∪ (0,1], f x a = g x) :=
by 
  sorry

end interval_monotonicity_f_range_of_a_intersection_l112_112492


namespace g_2_values_l112_112650

-- Define the set of positive real numbers
def S := {x : ℝ | x > 0}

-- Define the function g with its given properties
def g (x : S) : ℝ := sorry

axiom g_property : ∀ (x y : S), g x * g y = g (x * y) + 2010 * ((1 / (x ^ 2)) + (1 / (y ^ 2)) + 2009)

-- Theorem statement should state that the product of the number of possible values of g(2) (n)
-- and the sum of all possible values (s) is equal to 8041/4
theorem g_2_values : 
  let n := number_of_possible_values g 2,
      s := sum_of_possible_values g 2 in
  n * s = 8041 / 4 := sorry

end g_2_values_l112_112650


namespace ian_money_left_l112_112066

-- The conditions outlined in the problem
def initial_amount : ℕ := 100
def paid_colin : ℕ := 20
def paid_helen : ℕ := 2 * paid_colin
def paid_benedict : ℕ := paid_helen / 2
def total_paid : ℕ := paid_colin + paid_helen + paid_benedict
def remaining_amount : ℕ := initial_amount - total_paid

-- The statement to prove
theorem ian_money_left : remaining_amount = 20 := by
  unfold initial_amount paid_colin paid_helen paid_benedict total_paid remaining_amount
  sorry

end ian_money_left_l112_112066


namespace projection_magnitude_l112_112481

-- Definitions for the problem.
def point_A : ℝ × ℝ × ℝ := (3, 4, 5)
def point_B : ℝ × ℝ × ℝ := (3, 4, 0)

-- |OB|, the magnitude of vector OB
def magnitude_OB (B : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (B.1^2 + B.2^2 + B.3^2)

-- The theorem to prove.
theorem projection_magnitude :
  magnitude_OB point_B = 5 :=
by {
  -- Proof to be added.
  sorry
}

end projection_magnitude_l112_112481


namespace max_pastries_l112_112366

theorem max_pastries (n d k m : ℕ) 
  (hlt : k ≤ m) 
  (hcon1 : d = 10)
  (hcon2 : ∀ i : ℕ, i ≥ 0 → i + 6 < n → d ≥ 3) 
  : n ≤ 26 :=
begin
  sorry
end

end max_pastries_l112_112366


namespace sum_smallest_largest_consecutive_even_integers_l112_112207

theorem sum_smallest_largest_consecutive_even_integers
  (n : ℕ) (a y : ℤ) 
  (hn_even : Even n) 
  (h_mean : y = (a + (a + 2 * (n - 1))) / 2) :
  2 * y = (a + (a + 2 * (n - 1))) :=
by
  sorry

end sum_smallest_largest_consecutive_even_integers_l112_112207


namespace distance_to_nearest_park_l112_112797

theorem distance_to_nearest_park (d : ℝ) :
  (¬ (d ≥ 8)) ∧ (¬ (d ≤ 7)) ∧ (¬ (d ≤ 6)) → 7 < d ∧ d < 8 :=
by
  intros h
  cases h with h_alice h_rest
  cases h_rest with h_bob h_charlie
  have h1 : d < 8 := lt_of_not_ge h_alice
  have h2 : d > 7 := lt_of_not_le h_bob
  have h3 : d > 6 := lt_of_not_le h_charlie
  exact ⟨h2, h1⟩

end distance_to_nearest_park_l112_112797


namespace smallest_common_multiple_l112_112272

theorem smallest_common_multiple (n : ℕ) : 
  (2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n ∧ 1000 ≤ n ∧ n < 10000) → n = 1008 :=
by {
    sorry
}

end smallest_common_multiple_l112_112272


namespace chips_cost_proof_l112_112238

def candy_bar_cost : ℝ := 2
def total_cost : ℝ := 15
def num_students : ℕ := 5
def chips_cost (C : ℝ) : Prop := num_students * (candy_bar_cost + 2 * C) = total_cost

theorem chips_cost_proof : ∃ C : ℝ, chips_cost C :=
begin
  use 0.5,
  unfold chips_cost,
  norm_num,
end

end chips_cost_proof_l112_112238


namespace parabola_reflection_l112_112184

theorem parabola_reflection :
  ∀ (x y : ℝ), let P := (1, 1) in 
  (y = x^2) → (y = (2 - x)^2 - 2) :=
by
  sorry

end parabola_reflection_l112_112184


namespace set_intersection_l112_112861

-- Define set A
def A := {x : ℝ | x^2 - 4 * x < 0}

-- Define set B
def B := {x : ℤ | -2 < x ∧ x ≤ 2}

-- Define the intersection of A and B in ℝ
def A_inter_B := {x : ℝ | (x ∈ A) ∧ (∃ (z : ℤ), (x = z) ∧ (z ∈ B))}

-- Proof statement
theorem set_intersection : A_inter_B = {1, 2} :=
by sorry

end set_intersection_l112_112861


namespace color_chain_count_l112_112856

variable (n : ℕ)
variable (h : n ≥ 2)
variable (colors : Set (ℕ → Color)) -- Assuming a Color type and a way to define a coloring function

-- Defining the problem: 
-- There are 2n grid points, three colors, and the required coloring properties
noncomputable def valid_colorings (n : ℕ) : ℕ :=
  if n < 2 then 0 else 3^(n-2)

theorem color_chain_count {n : ℕ} (h : n ≥ 2) :
  valid_colorings n = 3^(n-2) :=
sorry

end color_chain_count_l112_112856


namespace magnitude_of_z_l112_112523

theorem magnitude_of_z (i : ℂ) (hi : i.im = 1 ∧ i.re = 0) :
  let z := (1 - i) / i in
  complex.abs z = Real.sqrt 2 :=
by
  let z := (1 - i) / i
  sorry

end magnitude_of_z_l112_112523


namespace radius_increase_l112_112745

theorem radius_increase (C1 C2 : ℝ) (h1 : C1 = 30) (h2 : C2 = 40) : 
  let r1 := C1 / (2 * Real.pi)
  let r2 := C2 / (2 * Real.pi)
  let Δr := r2 - r1
  Δr = 5 / Real.pi := by
sorry

end radius_increase_l112_112745


namespace circle_tangent_solution_l112_112912

theorem circle_tangent_solution (a : ℝ) : 
  (∃ (C₁ C₂ : point ℝ), (C₁ = (0,0)) ∧ (C₂ = (a,0)) ∧ (circle C₁ 4) ∧ (circle C₂ 1) ∧ tangent C₁ C₂ 4 1) 
  → (a = 5) ∨ (a = -5) ∨ (a = 3) ∨ (a = -3) :=
begin
  sorry
end

end circle_tangent_solution_l112_112912


namespace combination_solutions_l112_112513

theorem combination_solutions (x : ℕ) :
  binom 24 x = binom 24 (3 * x - 8) ↔ (x = 4 ∨ x = 8) :=
by
  sorry

end combination_solutions_l112_112513


namespace exists_root_in_interval_l112_112198

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 3

theorem exists_root_in_interval :
  (∃ c ∈ Ioo 1 2, f c = 0) :=
begin
  -- Establish that f is continuous
  have h_cont : continuous f := by sorry,
  -- Calculate f(1) and f(2)
  have h_f1 : f 1 = -2 := by norm_num,
  have h_f2 : f 2 = Real.log 2 + 5 := by norm_num,
  -- Use intermediate value theorem
  exact sorry,
end

end exists_root_in_interval_l112_112198


namespace part1_part2_l112_112616

-- Part (1): Prove that A = 2π/3 given the trigonometric condition.
theorem part1 (A B C : ℝ) (h_condition : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : 
  A = 2 * π / 3 :=
sorry

-- Part (2): Prove that the maximum perimeter is 3 + 2√3 given BC = 3 and A = 2π/3.
theorem part2 (B C : angle) (h_BC : 3 = 3) (h_A : A = 2 * π / 3) :
  (3 + (sqrt 3) * 2 ≤ perimeter (triangle.mk 3 B C)) :=
sorry

end part1_part2_l112_112616


namespace local_minimum_h_at_zero_range_of_a_for_two_tangent_lines_l112_112017

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (x - a) ^ 2
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x - g x a

theorem local_minimum_h_at_zero {a : ℝ} (h_a : a = 2 * Real.log 2 - 2) :
  ∃ x₀ : ℝ, h x₀ a = 0 ∧ (∀ x, h x a ≥ h x₀ a) := 
sorry

theorem range_of_a_for_two_tangent_lines :
  ∀ a : ℝ, (∃ t : ℝ, ∀ x, f x = g x a → x = t) ↔ a ∈ Iic (2 * Real.log 2 - 2) :=
sorry

end local_minimum_h_at_zero_range_of_a_for_two_tangent_lines_l112_112017


namespace union_is_correct_l112_112915

def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }

theorem union_is_correct : A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
by
  sorry

end union_is_correct_l112_112915


namespace omega_value_f_max_min_l112_112161

def f (ω x : ℝ) : ℝ := (sqrt 3 / 2) - sqrt 3 * (sin (ω * x))^2 - (sin (ω * x)) * (cos (ω * x))

def f_omega_1 (x : ℝ) : ℝ := -sin (2 * x - π / 3)

theorem omega_value (ω : ℝ) (h₀ : 0 < ω)
    (h₁ : ∃ x₀ : ℝ, ∀ x : ℝ, f ω x = f ω (x₀ - x) 
                                ∨ f ω x = f ω (x₀ + x) 
                                ∧ abs (x₀ - x) = π / 4) :
    ω = 1 := by
  sorry

theorem f_max_min : 
  (∀ x : ℝ, π ≤ x ∧ x ≤ (3 * π / 2) → f_omega_1 x ≤ sqrt 3 / 2 ∧ f_omega_1 x ≥ -1) :=
  by
  sorry

end omega_value_f_max_min_l112_112161


namespace probability_last_box_contains_exactly_one_ball_l112_112097

-- Definitions and conditions
def num_boxes : ℕ := 100
def num_balls : ℕ := 100
def p : ℝ := 1 / num_boxes.toReal

-- To show: The probability that the last box contains exactly one ball
theorem probability_last_box_contains_exactly_one_ball :
  ((1 - p) ^ (num_boxes - 1)) ≈ 0.370 :=
by
  sorry

end probability_last_box_contains_exactly_one_ball_l112_112097


namespace intersect_in_third_quadrant_l112_112890

theorem intersect_in_third_quadrant (b : ℝ) : (¬ (∃ x y : ℝ, y = 2*x + 1 ∧ y = 3*x + b ∧ x < 0 ∧ y < 0)) ↔ b > 3 / 2 := sorry

end intersect_in_third_quadrant_l112_112890


namespace evaluate_expression_l112_112824

theorem evaluate_expression :
  -25 + 7 * ((8 / 4) ^ 2) = 3 :=
by
  sorry

end evaluate_expression_l112_112824


namespace pascal_second_number_l112_112730

theorem pascal_second_number {n : ℕ} (h : n + 1 = 31) : binomial n 1 = 30 :=
by
  have hn : n = 30 := Nat.add_one_eq_if_eq_pred h
  rw [hn]
  exact Nat.binomial_succ_succ 29 0

end pascal_second_number_l112_112730


namespace correct_calculation_l112_112748

theorem correct_calculation :
  (∀ (x y : ℝ), (3 * x - 2 * x ≠ 1)) ∧
  (∀ (x y : ℝ), (2 * x^2 * y - x * y^2 ≠ x^2 * y)) ∧
  (∀ (x y : ℝ), (-2 * x * y + 3 * y * x = x * y)) ∧
  (∀ (a : ℝ), (3 * a^2 + 4 * a^2 ≠ 7 * a^4)) := 
by 
  split
  · intro x y; sorry
  split
  · intro x y; sorry
  split
  · intro x y; sorry
  intro a; sorry

end correct_calculation_l112_112748


namespace smallest_positive_x_l112_112420

theorem smallest_positive_x (x : ℝ) (h : ⌊x^2⌋ - x * ⌊x⌋ = 8) : x = 89 / 9 :=
sorry

end smallest_positive_x_l112_112420


namespace angle_notation_l112_112751

theorem angle_notation (A O B : Point) : angle A O B = angle B O A :=
by
  sorry

end angle_notation_l112_112751


namespace find_smallest_x_l112_112422

noncomputable def smallest_x : ℝ :=
  min { x : ℝ | 0 < x ∧ (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * x = 8 }

theorem find_smallest_x :
  smallest_x = 89 / 9 :=
by
  sorry

end find_smallest_x_l112_112422


namespace complex_shape_perimeter_l112_112690

theorem complex_shape_perimeter (perimeter_of_one_piece : ℕ) (num_pieces : ℕ) (correct_perimeter : ℕ) :
  perimeter_of_one_piece = 17 → num_pieces = 7 → correct_perimeter = 51 → 
  num_pieces * perimeter_of_one_piece / 2 = correct_perimeter :=
by
  intros h1 h2 h3
  rw [h1, h2]
  norm_num
  rw h3
  sorry

end complex_shape_perimeter_l112_112690


namespace inequality_relations_l112_112450

noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 125 ^ (1 / 6)
noncomputable def c : ℝ := Real.log 7 / Real.log (1 / 6)

theorem inequality_relations :
  c < a ∧ a < b := 
by 
  sorry

end inequality_relations_l112_112450


namespace slope_of_line_with_30_deg_inclination_l112_112049

theorem slope_of_line_with_30_deg_inclination :
  let θ := 30
  (Real.tan θ = sqrt 3 / 3) → 
  ∃ k : Real, k = sqrt 3 / 3 :=
by
  sorry

end slope_of_line_with_30_deg_inclination_l112_112049


namespace part1_part2_l112_112626

theorem part1 (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : A = 2 * π / 3 :=
sorry

theorem part2 (b c : ℝ) (h1 : sin (2 * π / 3) ^ 2 - sin b ^ 2 - sin c ^ 2 = sin b * sin c)
  (h2 : b + c = 2 * sqrt 3) : 3 * 2 + b + c = 3 + 2 * sqrt 3 :=
sorry

end part1_part2_l112_112626


namespace initial_percentage_40_l112_112743

variables
  (P : ℝ)  -- Initial percentage of the acidic liquid

-- Given conditions
def initial_volume : ℝ := 18        -- Initial volume of the solution
def removed_water : ℝ := 6          -- Volume of water removed
def final_volume : ℝ := 12          -- Final volume of the solution after removing water
def final_concentration : ℝ := 0.60 -- Desired final concentration of the acidic liquid

-- Proof statement
theorem initial_percentage_40 :
  (initial_volume * (P / 100)) = (final_volume * final_concentration) →
  P = 40 :=
begin
  intro h,
  sorry
end

end initial_percentage_40_l112_112743


namespace isosceles_triangles_ADB_ADC_l112_112956

theorem isosceles_triangles_ADB_ADC
  {A B C D : Type*}
  (hABC_iso : is_isosceles_triangle A B C)
  (hB_angle : ∠ B = 36)
  (hAD_bisector : is_angle_bisector A D (angle BAC)) :
  is_isosceles_triangle A D B ∧ is_isosceles_triangle A D C :=
by sorry

end isosceles_triangles_ADB_ADC_l112_112956


namespace positive_root_exists_iff_m_eq_neg_one_l112_112074

theorem positive_root_exists_iff_m_eq_neg_one :
  (∃ x : ℝ, x > 0 ∧ (x / (x - 1) - m / (1 - x) = 2)) ↔ m = -1 :=
by
  sorry

end positive_root_exists_iff_m_eq_neg_one_l112_112074


namespace farmer_owned_land_l112_112667

variable (A : ℝ)
variable (ClearedLandPotatoesPercentage ClearedLandTomatoesPercentage ClearedLandOnionsPercentage ClearedLandCarrotsPercentage : ℝ)
variable (ClearedLandPlantedWithCorn : ℝ)

axiom land_cleared : 0.90 * A
axiom potatoes_percentage : ClearedLandPotatoesPercentage = 0.20
axiom tomatoes_percentage : ClearedLandTomatoesPercentage = 0.30
axiom onions_percentage : ClearedLandOnionsPercentage = 0.25
axiom carrots_percentage : ClearedLandCarrotsPercentage = 0.15
axiom corn_acres : ClearedLandPlantedWithCorn = 540

theorem farmer_owned_land : A = 6000 :=
by
  let cleared_land := 0.90 * A
  have total_percentage := ClearedLandPotatoesPercentage + ClearedLandTomatoesPercentage + ClearedLandOnionsPercentage + ClearedLandCarrotsPercentage = 0.90
  have remaining_percentage := 1 - total_percentage
  have corn_land := remaining_percentage * cleared_land
  have corn_land_equals := corn_land = ClearedLandPlantedWithCorn
  have eq1 : 0.10 * 0.90 * A = 540 := by sorry
  have eq2 : 0.09 * A = 540 := by rw [mul_assoc]; assumption
  have eq3 : A = 6000 := by sorry
  exact eq3

end farmer_owned_land_l112_112667


namespace log_expression_evaluation_l112_112350

theorem log_expression_evaluation : 
  log 10 (5 / 2) + 2 * log 10 2 - (1 / 2)⁻¹ = -1 := 
sorry

end log_expression_evaluation_l112_112350


namespace line_of_symmetry_l112_112031

-- Definitions of the circles and the line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 4 * y - 1 = 0
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- The theorem stating the symmetry condition
theorem line_of_symmetry :
  ∀ (x y : ℝ), circle1 x y ↔ ∃ (x' y' : ℝ), line ((x + x') / 2) ((y + y') / 2) ∧ circle2 x' y' :=
sorry

end line_of_symmetry_l112_112031


namespace minimum_value_of_quadratic_expression_l112_112734

theorem minimum_value_of_quadratic_expression : ∃ x ∈ ℝ, ∀ y ∈ ℝ, x^2 + 10 * x ≤ y^2 + 10 * y := by
  sorry

end minimum_value_of_quadratic_expression_l112_112734


namespace last_box_one_ball_probability_l112_112101

/-- The probability that the last box will contain exactly one of 100 randomly distributed balls
is approximately 0.370. -/
theorem last_box_one_ball_probability :
  let n : ℕ := 100 in
  let p : ℚ := 1 / 100 in
  let probability : ℚ := (n.choose 1) * (p) * ((1 - p) ^ (n - 1)) in
  probability ≈ 0.370 :=
by
  let n : ℕ := 100 
  let p : ℚ := 1 / 100
  let probability : ℚ := (n.choose 1) * (p) * ((1 - p) ^ (n - 1))
  sorry

end last_box_one_ball_probability_l112_112101


namespace small_circle_to_large_circle_ratio_l112_112963

theorem small_circle_to_large_circle_ratio (a b : ℝ) (h : π * b^2 - π * a^2 = 3 * π * a^2) :
  a / b = 1 / 2 :=
sorry

end small_circle_to_large_circle_ratio_l112_112963


namespace triangle_A_value_and_max_perimeter_l112_112580

theorem triangle_A_value_and_max_perimeter (A B C a b c : ℝ) 
  (h1 : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C)
  (h2 : a = 3) :
  (A = 2 * Real.pi / 3) ∧ (a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by
  sorry

end triangle_A_value_and_max_perimeter_l112_112580


namespace range_of_expression_l112_112374

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (L U : ℝ), L = - (5 * real.pi / 2) ∧ U = 3 * real.pi / 2 ∧ ∀ z, z = 4 * real.arcsin x - real.arccos y →
  L ≤ z ∧ z ≤ U :=
sorry

end range_of_expression_l112_112374


namespace islander_C_response_l112_112668

-- Define the types and assumptions
variables {Person : Type} (is_knight : Person → Prop) (is_liar : Person → Prop)
variables (A B C : Person)

-- Conditions from the problem
axiom A_statement : (is_liar A) ↔ (is_knight B = false ∧ is_knight C = false)
axiom B_statement : (is_knight B) ↔ (is_knight A ↔ ¬ is_knight C)

-- Conclusion we want to prove
theorem islander_C_response : is_knight C → (is_knight A ↔ ¬ is_knight C) := sorry

end islander_C_response_l112_112668


namespace magnitude_of_sum_of_scaled_vectors_l112_112896

theorem magnitude_of_sum_of_scaled_vectors
  (a b : ℝ × ℝ)
  (ha : |a| = 1)
  (hb : |b| = 2)
  (h_diff : a - b = (real.sqrt 3, real.sqrt 2)) :
  |2 • a + b| = 2 * real.sqrt 2 := 
sorry

end magnitude_of_sum_of_scaled_vectors_l112_112896


namespace find_a_minus_b_l112_112864

def complex_eq (a b : ℝ) : Prop := 
  let z := complex.mk a b
  (⟪i, 0⟫ / z) = ⟪2, -1⟫

theorem find_a_minus_b (a b : ℝ) (h : complex_eq a b) : a - b = -3 / 5 :=
by
  sorry

end find_a_minus_b_l112_112864


namespace eq_number_increased_by_five_l112_112218

theorem eq_number_increased_by_five (n : ℤ) : n + 5 = 15 → n = 10 :=
by
  assume h : n + 5 = 15
  sorry

end eq_number_increased_by_five_l112_112218


namespace particle_position_after_1891_minutes_l112_112315

theorem particle_position_after_1891_minutes :
  let start_pos := (0, 0)
      moves := λ n : ℕ, if n % 4 == 0 then (1, 0) else if n % 4 == 2 then (0, 1) else (0, 0)
      skip := 1
      total_minutes := 1891
  in final_position(start_pos, moves, skip, total_minutes) = (45, 46) :=
by 
  sorry

def final_position (start : ℕ × ℕ) (move : ℕ → ℕ × ℕ) (skip : ℕ) (total : ℕ) : ℕ × ℕ :=
  sorry

end particle_position_after_1891_minutes_l112_112315


namespace range_of_a_l112_112702

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 - 2 * x + 3 ≤ a^2 - 2 * a - 1)) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l112_112702


namespace black_balls_probability_l112_112822

theorem black_balls_probability :
  ∃ (n1 n2 k1 k2 : ℕ), 
  n1 + n2 = 25 ∧
  k1 / n1.to_rat * k2 / n2.to_rat = 0.54 ∧
  ((1 - k1 / n1.to_rat) * (1 - k2 / n2.to_rat) = 0.04) := 
sorry

end black_balls_probability_l112_112822


namespace students_wear_other_colors_l112_112952

-- Define the conditions given in the problem
def n_total : ℕ := 900
def p_blue : ℝ := 0.44
def p_red : ℝ := 0.28
def p_green : ℝ := 0.10

/-- The number of students who wear other colors -/
theorem students_wear_other_colors : 
  (900 : ℕ) * (1 - (0.44 + 0.28 + 0.10)) = 162 := by
  sorry

end students_wear_other_colors_l112_112952


namespace bisection_method_third_interval_l112_112746

theorem bisection_method_third_interval 
  (f : ℝ → ℝ) (a b : ℝ) (H1 : a = -2) (H2 : b = 4) 
  (H3 : f a * f b ≤ 0) : 
  ∃ c d : ℝ, c = -1/2 ∧ d = 1 ∧ f c * f d ≤ 0 :=
by 
  sorry

end bisection_method_third_interval_l112_112746


namespace simplest_quadratic_radical_l112_112750

theorem simplest_quadratic_radical :
  let A := sqrt 4
  let B := sqrt 7
  let C := sqrt 12
  let D := sqrt 0.5
  B = sqrt 7 ∧ (A = 2 ∨ C = 2 * sqrt 3 ∨ D = sqrt (1 / 2)) :=
by
  sorry

end simplest_quadratic_radical_l112_112750


namespace line_parallel_to_parallel_set_l112_112913

variables {Point Line Plane : Type} 
variables (a : Line) (α : Plane)
variables (parallel : Line → Plane → Prop) (parallel_set : Line → Plane → Prop)

-- Definition for line parallel to plane
axiom line_parallel_to_plane : parallel a α

-- Goal: line a is parallel to a set of parallel lines within plane α
theorem line_parallel_to_parallel_set (h : parallel a α) : parallel_set a α := 
sorry

end line_parallel_to_parallel_set_l112_112913


namespace PQ_bisects_BD_l112_112941

variable (A B C D P Q : Type) [Add A] (M : A) [Div A Two]

theorem PQ_bisects_BD
  (h1 : convex_quadrilateral A B C D)
  (h2 : midpoint A B P)
  (h3 : midpoint C D Q)
  (h4 : bisects P Q A C) :
  bisects P Q B D :=
sorry

end PQ_bisects_BD_l112_112941


namespace right_triangle_hypotenuse_length_l112_112197

theorem right_triangle_hypotenuse_length (α R : ℝ) (hα : 0 < α ∧ α < π / 2) :
  ∃ (x : ℝ), x = (R * (1 - tan (α / 2)) / (cos α)) :=
begin
  have h_x : ∀ (x : ℝ), x = (R * (1 - tan (α / 2)) / (cos α)),
  { intro x,
    linarith },
  exact ⟨_, h_x⟩,
end

end right_triangle_hypotenuse_length_l112_112197


namespace distance_PQ_eq_1_l112_112488

-- Definitions of the curves in Cartesian coordinates
def C_1 (x y : ℝ) : Prop := x + sqrt 3 * y = sqrt 3
def C_2 (ϕ : ℝ) (x y : ℝ) : Prop := (x = sqrt 6 * cos ϕ) ∧ (y = sqrt 2 * sin ϕ)

-- Definition of points in polar coordinates
def point_P (ρ θ : ℝ) : Prop := ρ = 1 ∧ θ = π / 6
def point_Q (ρ θ : ℝ) : Prop := ρ = 2 ∧ θ = π / 6

-- Lean 4 statement proving that distance between P and Q is 1
theorem distance_PQ_eq_1 (x y ϕ : ℝ) : 
  (C_1 x y) ∧ ∀ ϕ, (C_2 ϕ x y) ∧ (point_P 1 (π / 6)) ∧ (point_Q 2 (π / 6)) → abs (2 - 1) = 1 :=
by
  sorry

end distance_PQ_eq_1_l112_112488


namespace opposite_of_neg_two_thirds_l112_112231

theorem opposite_of_neg_two_thirds : -(- (2 / 3)) = (2 / 3) :=
by
  sorry

end opposite_of_neg_two_thirds_l112_112231


namespace sqrt_meaningful_range_l112_112080

theorem sqrt_meaningful_range (x : ℝ): x + 2 ≥ 0 ↔ x ≥ -2 := by
  sorry

end sqrt_meaningful_range_l112_112080


namespace proof_problem_l112_112862

open Real
noncomputable theory

def vector_a (n : ℕ) : ℝ × ℝ :=
let θ := (n * π) / 6 in (cos θ, sin θ)

def vector_b : ℝ × ℝ := (1 / 2, sqrt 3 / 2)

def norm_squared (v : ℝ × ℝ) : ℝ :=
v.1 ^ 2 + v.2 ^ 2

def y : ℝ := ∑ n in Finset.range 2015, norm_squared (vector_a (n + 1) + vector_b)

theorem proof_problem : y = 4029 := sorry

end proof_problem_l112_112862


namespace girls_select_same_colored_marble_l112_112297

def probability_same_color (total_white total_black girls boys : ℕ) : ℚ :=
  let prob_white := (total_white * (total_white - 1)) / ((total_white + total_black) * (total_white + total_black - 1))
  let prob_black := (total_black * (total_black - 1)) / ((total_white + total_black) * (total_white + total_black - 1))
  prob_white + prob_black

theorem girls_select_same_colored_marble :
  probability_same_color 2 2 2 2 = 1 / 3 :=
by
  sorry

end girls_select_same_colored_marble_l112_112297


namespace compute_expression_l112_112812

theorem compute_expression : 7 * (1 / 21) * 42 = 14 :=
by
  sorry

end compute_expression_l112_112812


namespace bisects_AC_implies_bisects_BD_l112_112934

/-- Given a convex quadrilateral ABCD with points P and Q being the midpoints of sides AB and CD respectively,
    and given that the line segment PQ bisects the diagonal AC, prove that PQ also bisects the diagonal BD. -/
theorem bisects_AC_implies_bisects_BD
    (A B C D P Q M N : Point)
    (hP : midpoint A B P)
    (hQ : midpoint C D Q)
    (hM : midpoint A C M)
    (hN : midpoint B D N)
    (hPQ_bisects_AC : lies_on_line M (line_through P Q))
    : lies_on_line N (line_through P Q) :=
sorry

end bisects_AC_implies_bisects_BD_l112_112934


namespace tan_double_angle_l112_112027

variable (α : ℝ)
hypothesis h1 : π - α ∈ Set.Icc 0 (2 * π)
hypothesis h2 : sin(π - α) = -3 / 5

theorem tan_double_angle : tan (2 * α) = -24 / 7 := by
  sorry

end tan_double_angle_l112_112027


namespace trig_identity_sum_l112_112838

theorem trig_identity_sum :
  (sin (π / 24)) ^ 4 + (cos (5 * π / 24)) ^ 4 + (sin (19 * π / 24)) ^ 4 + (cos (23 * π / 24)) ^ 4 = 3 / 2 :=
by
  sorry

end trig_identity_sum_l112_112838


namespace part_1_part_2_l112_112592

-- Define the triangle and the given condition
variables {α β γ : ℝ}
axiom triangle_ABC : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π
axiom sin_identity : sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)
axiom BC_length : (sin(γ) / sin(α)) * BC = 3

-- State the main theorem parts separately
theorem part_1 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_sin_identity: sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)) :
  α = 2 * π / 3 :=
sorry

theorem part_2 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_α: α = 2 * π / 3) (h_BC_length: BC = 3) :
  let b := (2 * sqrt(3) * sin(π/6 - β)),
      c := (2 * sqrt(3) * sin(π/6 + β)) in
  (3 + 2 * sqrt(3)) :=
sorry

end part_1_part_2_l112_112592


namespace weighted_average_combined_l112_112365

noncomputable def weighted_average : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ := 
  λ studentsA markA studentsB markB studentsC markC studentsD markD studentsE markE, 
  ((studentsA * markA) + (studentsB * markB) + (studentsC * markC) + (studentsD * markD) + (studentsE * markE)) / (studentsA + studentsB + studentsC + studentsD + studentsE)

theorem weighted_average_combined : weighted_average 58 67 52 82 45 75 62 71 35 88 = 75 :=
by
  -- New theorem condition setup.
  have h_studentsA : ℕ := 58 
  have h_markA : ℕ := 67 
  have h_studentsB : ℕ := 52 
  have h_markB : ℕ := 82 
  have h_studentsC : ℕ := 45 
  have h_markC : ℕ := 75 
  have h_studentsD : ℕ := 62 
  have h_markD : ℕ := 71 
  have h_studentsE : ℕ := 35 
  have h_markE : ℕ := 88 
  -- Utilizing the pre-defined formula and Lean's calculated result.
  have h_weighted_average := weighted_average h_studentsA h_markA h_studentsB h_markB h_studentsC h_markC 
                    h_studentsD h_markD h_studentsE h_markE
  show weighted_average h_studentsA h_markA h_studentsB h_markB h_studentsC h_markC 
                     h_studentsD h_markD h_studentsE h_markE = 75
  from sorry

end weighted_average_combined_l112_112365


namespace angle_in_fourth_quadrant_l112_112467

theorem angle_in_fourth_quadrant (α : ℝ) (P : ℝ × ℝ) 
  (hP1 : P.1 = tan α) (hP2 : P.2 = cos α) (hP3 : tan α < 0) (hP4 : cos α > 0)
  (hP5 : P ∈ {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}) :
  (α > -π / 2 ∧ α < 0) :=
by
  sorry

end angle_in_fourth_quadrant_l112_112467


namespace bisect_diagonal_BD_l112_112929

-- Define a convex quadrilateral ABCD with midpoints P and Q on sides AB and CD respectively.
variables {A B C D P Q M N : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables [Inhabited P] [Inhabited Q] [Inhabited M] [Inhabited N]

-- Assuming the given statements:
-- ABCD is a convex quadrilateral
-- P is the midpoint of AB
-- Q is the midpoint of CD
-- Line PQ bisects the diagonal AC

-- Prove that line PQ also bisects the diagonal BD
theorem bisect_diagonal_BD 
  (convex_quadrilateral : convex_quadrilateral A B C D)
  (midpoint_P : midpoint P A B)
  (midpoint_Q : midpoint Q C D)
  (PQ_bisects_AC : bisects_line PQ M A C) :
  bisects_line PQ N B D :=
sorry  -- Proof is omitted

end bisect_diagonal_BD_l112_112929


namespace roots_equation_l112_112472

theorem roots_equation (α β : ℝ) (h1 : α^2 - 4 * α - 1 = 0) (h2 : β^2 - 4 * β - 1 = 0) :
  3 * α^3 + 4 * β^2 = 80 + 35 * α :=
by
  sorry

end roots_equation_l112_112472


namespace min_additional_weeks_l112_112346

theorem min_additional_weeks (n wins : ℕ) (puppy_cost : ℕ) (first_2_weeks_prize : ℕ) (win_prize : ℕ):
  (win_prize = 100) → (first_2_weeks_prize = 200) → (wins = 8) →
  (puppy_cost = first_2_weeks_prize + win_prize * wins) →
  n = wins :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3] at h4,
  exact h3
end

end min_additional_weeks_l112_112346


namespace set_intersection_example_l112_112860

theorem set_intersection_example (A : Set ℕ) (B : Set ℕ) (hA : A = {1, 3, 5}) (hB : B = {3, 4}) :
  A ∩ B = {3} :=
by
  sorry

end set_intersection_example_l112_112860


namespace problem_1_and_2_l112_112601

-- Definition of the problem in Lean 4
theorem problem_1_and_2 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin (A)^2 - sin (B)^2 - sin (C)^2 = sin (B) * sin (C))
  (h2 : BC = 3)
  (h3 : triangle ABC)
  (h4 : a = side_length (opposite A))
  (h5 : b = side_length (opposite B))
  (h6 : c = side_length (opposite C)) :
  A = 2 * real.pi / 3 ∧
  (a + b + c ≤ 3 + 2 * real.sqrt 3) :=
sorry

end problem_1_and_2_l112_112601


namespace potential_function_satisfies_Poisson_l112_112181

-- Definition of the Laplacian in 3D space
def Laplacian (u : ℝ × ℝ × ℝ → ℝ) : ℝ × ℝ × ℝ → ℝ :=
  λ (x y z), 
  (u (x + 𝜀, y, z) - 2 * u (x, y, z) + u (x - 𝜀, y, z)) / 𝜀^2 +
  (u (x, y + 𝜀, z) - 2 * u (x, y, z) + u (x, y - 𝜀, z)) / 𝜀^2 +
  (u (x, y, z + 𝜀) - 2 * u (x, y, z) + u (x, y, z - 𝜀)) / 𝜀^2

-- Definition of the Poisson equation
def Poisson_equation (u : ℝ × ℝ × ℝ → ℝ) (rho : ℝ × ℝ × ℝ → ℝ) : Prop :=
  ∀ (x y z : ℝ), Laplacian u (x, y, z) = -rho (x, y, z)

-- Lean theorem statement
theorem potential_function_satisfies_Poisson {u : ℝ × ℝ × ℝ → ℝ} {A : ℝ × ℝ × ℝ → ℝ} {rho : ℝ × ℝ × ℝ → ℝ} :
  (∀ (x y z : ℝ), A (x, y, z) = ∇ u (x, y, z)) → (∀ (x y z : ℝ), ∇ A (x, y, z) = rho (x, y, z)) →
  Poisson_equation u rho :=
by
  sorry  -- proof goes here

end potential_function_satisfies_Poisson_l112_112181


namespace smallest_positive_x_l112_112417

theorem smallest_positive_x (x : ℝ) (h : ⌊x^2⌋ - x * ⌊x⌋ = 8) : x = 89 / 9 :=
sorry

end smallest_positive_x_l112_112417


namespace minimum_f_l112_112160

def f (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 - 6 * y + 9) + 
  Real.sqrt (x^2 + y^2 + 2 * Real.sqrt 3 * x + 3) + 
  Real.sqrt (x^2 + y^2 - 2 * Real.sqrt 3 * x + 3)

theorem minimum_f : ∃ (x y : ℝ), f x y = 6 :=
sorry

end minimum_f_l112_112160


namespace number_of_unique_sums_l112_112344

open Finset

/-- Bag A contains the chips: 1, 2, 5, 7. -/
def bagA : Finset ℕ := {1, 2, 5, 7}

/-- Bag B contains the chips: 3, 4, 6, 8. -/
def bagB : Finset ℕ := {3, 4, 6, 8}

/-- Form all possible sums from chips in Bag A and Bag B -/
def possibleSums := (bagA ×ˢ bagB).image (λ (p : ℕ × ℕ), p.1 + p.2)

/-- State the problem as the theorem -/
theorem number_of_unique_sums :
  possibleSums.card = 10 := sorry

end number_of_unique_sums_l112_112344


namespace U_v_target_l112_112011

noncomputable def U (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

axiom U_linear (a b : ℝ) (v w : ℝ × ℝ × ℝ) : U (a * v + b * w) = a • U v + b • U w
axiom U_cross (v w : ℝ × ℝ × ℝ) : U (cross_product v w) = cross_product (U v) (U w)
axiom U_v1 : U ⟨5, 5, 2⟩ = ⟨3, -2, 7⟩
axiom U_v2 : U ⟨-5, 2, 5⟩ = ⟨3, 7, -2⟩

theorem U_v_target : U ⟨2, 8, 10⟩ = ⟨6, 6, 9⟩ := 
sorry

end U_v_target_l112_112011


namespace solution_eqn_l112_112820

theorem solution_eqn :
  ∀ x : ℝ, log 10 (x^2 + 10 * x) = 3 ↔ (x = 27 ∨ x = -37) :=
by sorry

end solution_eqn_l112_112820


namespace qt_q_t_neq_2_l112_112914

theorem qt_q_t_neq_2 (q t : ℕ) (hq : 0 < q) (ht : 0 < t) : q * t + q + t ≠ 2 :=
  sorry

end qt_q_t_neq_2_l112_112914


namespace find_smallest_x_l112_112411

def smallest_x_satisfying_condition : Prop :=
  ∃ x : ℝ, x > 0 ∧ (⌊x^2⌋ - x * ⌊x⌋ = 8) ∧ x = 89 / 9

theorem find_smallest_x : smallest_x_satisfying_condition :=
begin
  -- proof goes here (not required for this task)
  sorry
end

end find_smallest_x_l112_112411


namespace log2_arith_sequence_values_l112_112125

-- Define the function f
def f (x : ℝ) : ℝ := (1/3) * x^3 - 4 * x^2 + 6 * x - 1

-- Define the arithmetic sequence term a_n in terms of n and common difference d
def a (d : ℝ) (n : ℕ) : ℝ :=
  let x_1 := 1 -- assuming the smallest integer corresponding to first term's x-value is 1
  f (x_1 + (n - 1) * d)

-- Define the problem with specific terms
theorem log2_arith_sequence_values :
  let d := 1 / 4030 in
  a d 2 * a d 2017 * a d 4032 = a d 2 * a d 2017 * a d 4032 → -- ensuring terms are the same
  log 2 (a d 2 * a d 2017 * a d 4032) = 3 + log 2 3 :=
sorry

end log2_arith_sequence_values_l112_112125


namespace ms_cole_total_students_l112_112174

def number_of_students (S6 : Nat) (S4 : Nat) (S7 : Nat) : Nat :=
  S6 + S4 + S7

theorem ms_cole_total_students (S6 S4 S7 : Nat)
  (h1 : S6 = 40)
  (h2 : S4 = 4 * S6)
  (h3 : S7 = 2 * S4) :
  number_of_students S6 S4 S7 = 520 := by
  sorry

end ms_cole_total_students_l112_112174


namespace triangle_theorem_l112_112602

theorem triangle_theorem 
  (A B C : ℝ) 
  (h1 : ∀ A B C : ℝ, ∃ (a b c : ℝ), 
      a^2 - b^2 - c^2 = b * c ∧ sin^2 A - sin^2 B - sin^2 C = sin B * sin C) 
  (h2 : 0 < A ∧ A < π) : 
  (A = 2 * π / 3) ∧ 
  (∀ (BC : ℝ), BC = 3 → ∃ (a b c : ℝ), 
      a + b + c ≤ 3 + 2 * sqrt 3) :=
by
  sorry

end triangle_theorem_l112_112602


namespace b_range_l112_112070

noncomputable def f (a b x : ℝ) := (x - 1) * Real.log x - a * x + a + b

theorem b_range (a b : ℝ)
  (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 = 0 ∧ f a b x2 = 0) :
  b < 0 :=
sorry

end b_range_l112_112070


namespace ratio_a_b_range_cos_C_l112_112859

-- Parameters for the acute triangle ABC
variables {A B C : Real} 
variable {a b c : Real}

-- Given condition for the problem
axiom acute_triangle : A + B + C = π
axiom sides_opposite : a^2 + b^2 - 2ab * Real.cos(C) = c^2
axiom acute_condition : sqrt((1 - Real.cos(2 * C))/2) + Real.sin(B - A) = 2 * Real.sin(2 * A)
axiom angle_condition : A < π/2 ∧ B < π/2 ∧ C < π/2
axiom side_longest : a < b ∧ b <= c

-- Question (I)
theorem ratio_a_b : 
  sqrt((1 - Real.cos(2 * C))/2) + Real.sin(B - A) = 2 * Real.sin(2 * A) → 
  Real.sin B = 2 * Real.sin A →
  a = 2 * Real.sin A * c → 
  b = 2 * Real.sin B * c → 
  a / b = 1 / 2 := 
sorry

-- Question (II)
theorem range_cos_C : 
  a < b ∧ b <= c ∧ b = 2 * a → 
  (0 < Real.cos C ∧ Real.cos C <= 1 / 4) := 
sorry

end ratio_a_b_range_cos_C_l112_112859


namespace smallest_positive_real_x_l112_112394

theorem smallest_positive_real_x :
  ∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧ x = 89 / 9 :=
by
  sorry

end smallest_positive_real_x_l112_112394


namespace susan_typing_time_l112_112634

theorem susan_typing_time :
  let Jonathan_rate := 1 -- page per minute
  let Jack_rate := 5 / 3 -- pages per minute
  let combined_rate := 4 -- pages per minute
  ∃ S : ℝ, (1 + 1/S + 5/3 = 4) → S = 30 :=
by
  sorry

end susan_typing_time_l112_112634


namespace ratio_new_to_original_length_l112_112694

variable {L B L' : ℝ}

-- Condition: The new breadth is three times the original breadth.
axiom breadth_triple : ℝ := 3 * B

-- Condition: Percentage change in area is given as 50%
axiom area_percentage_change : L' * breadth_triple = 1.5 * (L * B)

-- Proof that the ratio of the new length to the original length is 1/2
theorem ratio_new_to_original_length : L' / L = 1 / 2 :=
by
  -- the final statement of the theorem follows from our conditions
  sorry

end ratio_new_to_original_length_l112_112694


namespace find_f_of_2_l112_112033

-- Given definitions:
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def defined_on_neg_inf_to_0 (f : ℝ → ℝ) : Prop := ∀ x, x < 0 → f x = 2 * x^3 + x^2

-- The main theorem to prove:
theorem find_f_of_2 (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_def : defined_on_neg_inf_to_0 f) :
  f 2 = 12 :=
sorry

end find_f_of_2_l112_112033


namespace original_square_perimeter_l112_112328

theorem original_square_perimeter (x : ℝ) 
  (h1 : ∀ r, r = x ∨ r = 4 * x) 
  (h2 : 28 * x = 56) : 
  4 * (4 * x) = 32 :=
by
  -- We don't need to consider the proof as per instructions.
  sorry

end original_square_perimeter_l112_112328


namespace range_of_function_l112_112389

theorem range_of_function (x : ℝ) (hx : 0 ≤ x ∧ x < 3) : 
  ∃ y, y = exp (-(x - 1)^2 + 1) ∧ e^(-3) < y ∧ y ≤ e :=
by
  sorry

end range_of_function_l112_112389


namespace find_A_max_perimeter_of_triangle_l112_112569

-- Definition of the given problem conditions
def triangle_condition (A B C : ℝ) : Prop :=
  sin(A)^2 - sin(B)^2 - sin(C)^2 = sin(B) * sin(C)

-- (1) Proving the value of A given the condition
theorem find_A (A B C : ℝ) (h : triangle_condition A B C) : A = 2 * π / 3 :=
by sorry

-- (2) Proving the maximum perimeter given BC = 3 and A = 2π/3
theorem max_perimeter_of_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (hA : 2 * π / 3 = 2 * π / 3) : 
  ∃ (P : ℝ), P = 3 + 2 * sqrt 3 :=
by sorry

end find_A_max_perimeter_of_triangle_l112_112569


namespace angle_ABC_is_60_degrees_l112_112689

noncomputable theory

open EuclideanGeometry

theorem angle_ABC_is_60_degrees
    (hexagon : Polygon)
    (square : Polygon)
    (triangle : Polygon)
    (A B C D : Point)
    (h_square_in_hexagon : square ⊆ hexagon)
    (h_triangle_on_square : ∃ (E : Point), E ∈ triangle ∧ ∀ (F : Point), F ∈ square → ∃ (G : Segment), G ∈ triangle ∧ G ∩ Segment(A,E) = ∅)
    (h_congruence_sides : ∀ (a b : Segment), a ∈ hexagon ∧ b ∈ square ∧ a ∈ triangle → length a = length b)
    (h_AB_common : AB ∈ square ∧ AB ∈ triangle)
    : ∠ABC = 60 :=
begin
  sorry,
end

end angle_ABC_is_60_degrees_l112_112689


namespace factor_identity_l112_112867

theorem factor_identity (m n : ℤ) :
  (∀ x, (x^2 - 1) ∣ (x^4 + m * x^3 + n * x^2 - 2 * x + 8)) →
  m + n = -7 :=
begin
  intro h,
  sorry -- Proof will be filled in here
end

end factor_identity_l112_112867


namespace least_integer_solution_l112_112727

theorem least_integer_solution (x : ℤ) (h : x^2 = 2 * x + 98) : x = -7 :=
by {
  sorry
}

end least_integer_solution_l112_112727


namespace find_smallest_x_l112_112409

def smallest_x_satisfying_condition : Prop :=
  ∃ x : ℝ, x > 0 ∧ (⌊x^2⌋ - x * ⌊x⌋ = 8) ∧ x = 89 / 9

theorem find_smallest_x : smallest_x_satisfying_condition :=
begin
  -- proof goes here (not required for this task)
  sorry
end

end find_smallest_x_l112_112409


namespace cuboid_area_correct_l112_112754

def cuboid_surface_area (length breadth height : ℕ) :=
  2 * (length * height) + 2 * (breadth * height) + 2 * (length * breadth)

theorem cuboid_area_correct : cuboid_surface_area 4 6 5 = 148 := by
  sorry

end cuboid_area_correct_l112_112754


namespace intersection_points_count_number_of_intersections_correct_l112_112221

-- Definitions for the logarithmic functions
def log_base_5 (x : ℝ) : ℝ := Real.log x / Real.log 5
def log_base_x (x b : ℝ) : ℝ := Real.log b / Real.log x
def log_base_1_div_5 (x : ℝ) : ℝ := Real.log x / Real.log (1/5)
def log_base_x_div_1_5 (x : ℝ) : ℝ := Real.log (1/5) / Real.log x

-- The problem statement
theorem intersection_points_count : 
  (∃ x : ℝ, 0 < x ∧ log_base_5 x = log_base_1_div_5 x) ∧
  (∃ x : ℝ, 0 < x ∧ log_base_5 x = log_base_x_div_1_5 x) ∧
  (∃ x : ℝ, 0 < x ∧ log_base_1_div_5 x = log_base_x_div_1_5 x) :=
begin
  -- Proof omitted.
  sorry
end

def unique_intersections := 3

theorem number_of_intersections_correct : unique_intersections = 3 :=
begin
  -- Proof omitted.
  sorry
end

end intersection_points_count_number_of_intersections_correct_l112_112221


namespace bisects_diagonals_l112_112926

-- Define the data structure for a convex quadrilateral
structure ConvexQuadrilateral (α : Type*) :=
(A B C D : α)

-- Define midpoints of line segments
def midpoint {α : Type*} [Add α] [Div α] [Nonempty α] (A B : α) : α :=
(A + B) / 2

-- Main theorem stating the problem
theorem bisects_diagonals
  {α : Type*} [AddCommGroup α] [Module ℝ α] (quad : ConvexQuadrilateral α)
  (P Q : α)
  (hP : P = midpoint quad.A quad.B)
  (hQ : Q = midpoint quad.C quad.D)
  (hPQ : ∃ M, M = midpoint quad.A quad.C ∧ M ∈ line_through P Q) :
  ∃ N, N = midpoint quad.B quad.D ∧ N ∈ line_through P Q :=
sorry

lemma line_through (P Q : α) : Prop :=
∃ (λ1 λ2 : ℝ), P + λ1 • (Q - P) = Q + λ2 • (P - Q)

end bisects_diagonals_l112_112926


namespace sum_of_favorite_numbers_is_600_l112_112168

def GloryFavoriteNumber : ℕ := 450
def MistyFavoriteNumber (G : ℕ) : ℕ := G / 3

theorem sum_of_favorite_numbers_is_600 (G : ℕ) (hG : G = GloryFavoriteNumber) :
  MistyFavoriteNumber G + G = 600 :=
by
  rw [hG]
  simp [GloryFavoriteNumber, MistyFavoriteNumber]
  -- Proof is omitted (filled with sorry)
  sorry

end sum_of_favorite_numbers_is_600_l112_112168


namespace incorrect_reasoning_for_argument_l112_112239

def rational (x : ℚ) : Prop := ∀ p q : ℤ, q ≠ 0 → x = p / q

def repeating_decimal (x : ℚ) : Prop := ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ repeating m n x

-- Define integers are a subset of rational numbers
def integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

-- Define the repeating property which we assume here without a solid definition as placeholder
def repeating (m n : ℕ) (x : ℚ) : Prop := sorry -- Assume this defines the repeating decimal properties

-- The incorrect argument 
def major_premise (x : ℚ) : Prop := rational x → repeating_decimal x

def minor_premise (x : ℚ) : Prop := integer x → rational x

-- The false conclusion derived from it
def false_conclusion (x : ℚ) : Prop := integer x → repeating_decimal x

theorem incorrect_reasoning_for_argument :
  ∃ x : ℚ, major_premise x ∧ minor_premise x ∧ ¬ false_conclusion x :=
begin
  -- Proof goes here
  sorry
end

end incorrect_reasoning_for_argument_l112_112239


namespace part1_part2_l112_112615

-- Part (1): Prove that A = 2π/3 given the trigonometric condition.
theorem part1 (A B C : ℝ) (h_condition : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : 
  A = 2 * π / 3 :=
sorry

-- Part (2): Prove that the maximum perimeter is 3 + 2√3 given BC = 3 and A = 2π/3.
theorem part2 (B C : angle) (h_BC : 3 = 3) (h_A : A = 2 * π / 3) :
  (3 + (sqrt 3) * 2 ≤ perimeter (triangle.mk 3 B C)) :=
sorry

end part1_part2_l112_112615


namespace complex_sum_identity_l112_112648

def z : ℂ := complex.cos (3 * real.pi / 8) + complex.sin (3 * real.pi / 8) * complex.I

theorem complex_sum_identity : 
  (z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = -2) :=
by
  sorry

end complex_sum_identity_l112_112648


namespace total_items_on_shelf_l112_112631

-- Given conditions
def initial_action_figures : Nat := 4
def initial_books : Nat := 22
def initial_video_games : Nat := 10

def added_action_figures : Nat := 6
def added_video_games : Nat := 3
def removed_books : Nat := 5

-- Definitions based on conditions
def final_action_figures : Nat := initial_action_figures + added_action_figures
def final_books : Nat := initial_books - removed_books
def final_video_games : Nat := initial_video_games + added_video_games

-- Claim to prove
theorem total_items_on_shelf : final_action_figures + final_books + final_video_games = 40 := by
  sorry

end total_items_on_shelf_l112_112631


namespace smallest_period_and_amplitude_of_function_l112_112261

theorem smallest_period_and_amplitude_of_function :
  let f : ℝ → ℝ := fun x => (sin x) * (cos x) + (cos (2 * x))
  smallest_period f = π ∧ amplitude f = 1 :=
by
  sorry

end smallest_period_and_amplitude_of_function_l112_112261


namespace parallel_line_with_distance_perpendicular_line_with_distance_l112_112764

theorem parallel_line_with_distance (m : ℤ) :
  (3 * (3x + 4y + m) = 0) ∧ (|12 + m| = 35) ↔ (m = 23) ∨ (m = -47) := by sorry

theorem perpendicular_line_with_distance (k : ℤ) :
  (3x - y + k = 0) ∧ (|(3 * (-1) - k)| = 6) ↔ (k = 9) ∨ (k = -3) := by sorry

end parallel_line_with_distance_perpendicular_line_with_distance_l112_112764


namespace probability_ball_in_last_box_is_approx_0_l112_112108

/-- Given 100 boxes and 100 balls randomly distributed,
the probability that the last box will contain the only ball is approximately 0.370. -/
theorem probability_ball_in_last_box_is_approx_0.370 :
  let n := 100,
      p := 1 / n,
      probability : ℝ := (99 / 100)^99
  in abs (probability - 0.370) < 0.001 :=
by {
  let n := 100,
  let p := 1 / n,
  let probability := (99 / 100)^99,
  sorry
}

end probability_ball_in_last_box_is_approx_0_l112_112108


namespace complex_modulus_squared_l112_112039

noncomputable def complex_z := (sqrt 3 + complex.I) / ((1 - sqrt 3 * complex.I) ^ 2)

theorem complex_modulus_squared :
  complex_z * complex.conj complex_z = 1 / 4 :=
sorry -- Proof skipped

end complex_modulus_squared_l112_112039


namespace rate_of_second_batch_of_wheat_l112_112801

theorem rate_of_second_batch_of_wheat (total_cost1 cost_per_kg1 weight1 weight2 total_weight total_cost selling_price_per_kg profit_rate cost_per_kg2 : ℝ)
  (H1 : total_cost1 = cost_per_kg1 * weight1)
  (H2 : total_weight = weight1 + weight2)
  (H3 : total_cost = total_cost1 + cost_per_kg2 * weight2)
  (H4 : selling_price_per_kg = (1 + profit_rate) * total_cost / total_weight)
  (H5 : profit_rate = 0.30)
  (H6 : cost_per_kg1 = 11.50)
  (H7 : weight1 = 30)
  (H8 : weight2 = 20)
  (H9 : selling_price_per_kg = 16.38) :
  cost_per_kg2 = 14.25 :=
by
  sorry

end rate_of_second_batch_of_wheat_l112_112801


namespace last_box_one_ball_probability_l112_112099

/-- The probability that the last box will contain exactly one of 100 randomly distributed balls
is approximately 0.370. -/
theorem last_box_one_ball_probability :
  let n : ℕ := 100 in
  let p : ℚ := 1 / 100 in
  let probability : ℚ := (n.choose 1) * (p) * ((1 - p) ^ (n - 1)) in
  probability ≈ 0.370 :=
by
  let n : ℕ := 100 
  let p : ℚ := 1 / 100
  let probability : ℚ := (n.choose 1) * (p) * ((1 - p) ^ (n - 1))
  sorry

end last_box_one_ball_probability_l112_112099


namespace jackies_free_time_l112_112972

-- Define the conditions
def hours_working : ℕ := 8
def hours_sleeping : ℕ := 8
def hours_exercising : ℕ := 3
def total_hours_in_day : ℕ := 24

-- The statement to be proven
theorem jackies_free_time : total_hours_in_day - (hours_working + hours_sleeping + hours_exercising) = 5 :=
by 
  rw [total_hours_in_day, hours_working, hours_sleeping, hours_exercising]
  -- 24 - (8 + 8 + 3) = 5
  sorry

end jackies_free_time_l112_112972


namespace sum_smallest_largest_consecutive_even_integers_l112_112205

theorem sum_smallest_largest_consecutive_even_integers
  (n : ℕ) (a y : ℤ) 
  (hn_even : Even n) 
  (h_mean : y = (a + (a + 2 * (n - 1))) / 2) :
  2 * y = (a + (a + 2 * (n - 1))) :=
by
  sorry

end sum_smallest_largest_consecutive_even_integers_l112_112205


namespace initial_raisins_l112_112636

theorem initial_raisins (x : ℕ) : 
  let y := x * 2 / 3 - 4 in 
  let z := y / 2 in 
  z = 16 → x = 54 := by
  sorry

end initial_raisins_l112_112636


namespace expected_ties_after_10_l112_112258

def binom: ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom n (k+1)

noncomputable def expected_ties : ℕ → ℝ 
| 0 => 0
| n+1 => expected_ties n + (binom (2*(n+1)) (n+1) / 2^(2*(n+1)))

theorem expected_ties_after_10 : expected_ties 5 = 1.707 := 
by 
  -- Placeholder for the actual proof
  sorry

end expected_ties_after_10_l112_112258


namespace base10_to_base7_l112_112262

theorem base10_to_base7 : ∀ (n : ℕ), n = 624 → base7_repr n = "1551"
  := by sorry

end base10_to_base7_l112_112262


namespace total_number_of_turtles_on_june_1_is_correct_l112_112778

noncomputable def estimate_turtles_in_pond_on_June_1
  (tagged_turtles_june : ℕ) (captured_turtles_october : ℕ)
  (tagged_in_october : ℕ) (percent_left_or_died : ℚ)
  (percent_new_arrivals : ℚ) :=
    let turtles_from_june_present_october := captured_turtles_october * (1 - percent_new_arrivals) in
    let proportion := (tagged_in_october : ℚ) / turtles_from_june_present_october = tagged_turtles_june / x in
    let x := (tagged_turtles_june : ℚ) * turtles_from_june_present_october / tagged_in_october in
    (x : ℕ)

theorem total_number_of_turtles_on_june_1_is_correct :
  let tagged_turtles_june := 80
  let captured_turtles_october := 50
  let tagged_in_october := 2
  let percent_left_or_died := 0.30
  let percent_new_arrivals := 0.50 in
  estimate_turtles_in_pond_on_June_1 tagged_turtles_june captured_turtles_october tagged_in_october percent_left_or_died percent_new_arrivals = 1000 :=
by 
  sorry

end total_number_of_turtles_on_june_1_is_correct_l112_112778


namespace min_sum_of_arithmetic_sequence_terms_l112_112462

open Real

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ d : ℝ, a m = a n + d * (m - n)

theorem min_sum_of_arithmetic_sequence_terms (a : ℕ → ℝ) 
  (hpos : ∀ n, a n > 0) 
  (harith : arithmetic_sequence a) 
  (hprod : a 1 * a 20 = 100) : 
  a 7 + a 14 ≥ 20 := sorry

end min_sum_of_arithmetic_sequence_terms_l112_112462


namespace find_A_max_perimeter_of_triangle_l112_112574

-- Definition of the given problem conditions
def triangle_condition (A B C : ℝ) : Prop :=
  sin(A)^2 - sin(B)^2 - sin(C)^2 = sin(B) * sin(C)

-- (1) Proving the value of A given the condition
theorem find_A (A B C : ℝ) (h : triangle_condition A B C) : A = 2 * π / 3 :=
by sorry

-- (2) Proving the maximum perimeter given BC = 3 and A = 2π/3
theorem max_perimeter_of_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (hA : 2 * π / 3 = 2 * π / 3) : 
  ∃ (P : ℝ), P = 3 + 2 * sqrt 3 :=
by sorry

end find_A_max_perimeter_of_triangle_l112_112574


namespace find_a4_l112_112477

variables {a d : ℝ} (n : ℕ)

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

def arithmetic_sum (a d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem find_a4 : (arithmetic_sum (-1) d 4 = 14) → arithmetic_sequence (-1) d 4 = 8 :=
begin
  sorry
end

end find_a4_l112_112477


namespace surface_area_greater_of_contained_l112_112178

-- Definitions and conditions
variables {P₁ P₂ : Type} [Polyhedron P₁] [Polyhedron P₂]
variable (contained : P₂ ⊆ P₁)
variable [ConvexPolyhedron P₁] [ConvexPolyhedron P₂]

-- Here we define SurfaceArea as a placeholder. In a real example, this would require a definition in Lean.
noncomputable def surface_area (P : Type) [Polyhedron P] : ℝ := sorry

-- Statement of the theorem
theorem surface_area_greater_of_contained (contained : P₂ ⊆ P₁) :
  surface_area P₁ > surface_area P₂ :=
sorry

end surface_area_greater_of_contained_l112_112178


namespace sum_mod_9_l112_112834

theorem sum_mod_9 (h1 : 34125 % 9 = 1) (h2 : 34126 % 9 = 2) (h3 : 34127 % 9 = 3)
                  (h4 : 34128 % 9 = 4) (h5 : 34129 % 9 = 5) (h6 : 34130 % 9 = 6)
                  (h7 : 34131 % 9 = 7) :
  (34125 + 34126 + 34127 + 34128 + 34129 + 34130 + 34131) % 9 = 1 :=
by
  sorry

end sum_mod_9_l112_112834


namespace smallest_n_for_neg_sum_l112_112013

theorem smallest_n_for_neg_sum {
  a1 : ℕ := 7,
  d : ℕ := -2,
  S_n : ℕ → ℤ := λ n, 7 * n + (n * (n - 1) * d / 2)
}: 
  ∃ n : ℕ, n > 8 ∧ -n^2 + 8 * n < 0 ∧ ∀ m : ℕ, m < n → -m^2 + 8 * m ≥ 0 :=
sorry

end smallest_n_for_neg_sum_l112_112013


namespace remainder_of_3a_minus_b_divided_by_5_l112_112151

theorem remainder_of_3a_minus_b_divided_by_5 (a b : ℕ) (m n : ℤ) 
(h1 : 3 * a > b) 
(h2 : a = 5 * m + 1) 
(h3 : b = 5 * n + 4) : 
(3 * a - b) % 5 = 4 := 
sorry

end remainder_of_3a_minus_b_divided_by_5_l112_112151


namespace area_of_shaded_quadrilateral_l112_112687

theorem area_of_shaded_quadrilateral:
  (let TQR_side_1 := 15
       TQR_side_2 := 20
       RT_sq := TQR_side_1^2 + TQR_side_2^2
       RT := (RT_sq:ℕ).sqrt
       SU := RT * TQR_side_1 / TQR_side_2
       RU := RT * TQR_side_2 / TQR_side_1
       area_square := TQR_side_2^2
       area_SUR := (SU * RU) / 2
       area_RQT := (TQR_side_2 * TQR_side_1) / 2
       area_shaded_quad := area_square - area_SUR - area_RQT
  in area_shaded_quad = 154) :=
begin
  sorry
end

end area_of_shaded_quadrilateral_l112_112687


namespace sum_first_20_terms_l112_112698

open Nat

def sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a (n + 1) + a n = (-1 : ℤ) ^ n * n

theorem sum_first_20_terms (a : ℕ → ℤ) (h : sequence a) :
  (∑ i in range 20, a i) = -100 := by
  sorry

end sum_first_20_terms_l112_112698


namespace find_angle_OPQ_l112_112964

open Real

noncomputable def angle_OPQ := 
  let O := (0, 0)
  let A := (cos (pi / 9), sin (pi / 9))
  let B := (cos (2 * pi / 9), sin (2 * pi / 9))
  let C := (cos (4 * pi / 9), sin (4 * pi / 9))
  let D := (cos (6 * pi / 9), sin (6 * pi / 9))
  let P := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  let Q := ((D.1 + O.1) / 2, (D.2 + O.2) / 2)
  acos ((P.1 * Q.1 + P.2 * Q.2) / (sqrt (P.1 ^ 2 + P.2 ^ 2) * sqrt (Q.1 ^ 2 + Q.2 ^ 2))) * (180 / pi) -- Turining the result from radian to degrees

theorem find_angle_OPQ : 
  angle_OPQ = 30 := 
sorry

end find_angle_OPQ_l112_112964


namespace systematic_sampling_interval_from_231_items_l112_112364

theorem systematic_sampling_interval_from_231_items : ∃ n : ℕ, n = 10 :=
by
  let total_items := 231
  let sample_size := 22
  have h1 : total_items % sample_size ≠ 0 := by
    simp [total_items, sample_size]
  have new_total := total_items - (total_items - (total_items / sample_size) * sample_size)
  have h2 : new_total = 220 := by
    simp [total_items, sample_size, new_total]
  have h3 : new_total / sample_size = 10 := by
    simp [total_items, sample_size, new_total]
  use 10
  exact h3

# Theorems requires proof 'sorry' to satisfy theorem correctness in Lean
#noncomputable def systematic_sampling_interval_from_231_items : ℕ :=
#by sorry  -- Placeholder, actual proof left as exercise for confirmation

end systematic_sampling_interval_from_231_items_l112_112364


namespace max_height_l112_112768

def h (t : ℝ) : ℝ := -20 * t ^ 2 + 80 * t + 50

theorem max_height : ∃ t : ℝ, ∀ t' : ℝ, h t' ≤ h t ∧ h t = 130 :=
by
  sorry

end max_height_l112_112768


namespace smallest_positive_real_number_l112_112406

noncomputable def smallest_x : ℝ :=
  Inf {x : ℝ | x > 0 ∧ (floor (x^2) - x * floor x = 8)}

theorem smallest_positive_real_number :
  smallest_x = 89 / 9 :=
by 
  sorry

end smallest_positive_real_number_l112_112406


namespace probability_both_sarah_sam_in_picture_l112_112677

open Set

variable (ρ_sarah ρ_sam : ℝ) (t : ℝ)

-- Given conditions
def condition_sarah_lap_time := ρ_sarah = 120
def condition_sam_lap_time := ρ_sam = 75
def condition_start_time := t = random_between 720 780
def condition_picture_time (T : ℝ) := T = (1 / 3)
def sarah_in_picture (T : ℝ) := (T <= 40 ∨ 80 <= T ∧ T <= 120)
def sam_in_picture (T : ℝ) := (T <= 25 ∨ 50 <= T ∧ T <= 75)

-- Probability calculation
def probability_both_in_picture : ℝ := (30 / 120)

theorem probability_both_sarah_sam_in_picture :
  condition_sarah_lap_time ∧
  condition_sam_lap_time ∧
  condition_start_time ∧
  condition_picture_time ρ_sarah ∧
  condition_picture_time ρ_sam ∧
  sarah_in_picture ρ_sarah ∧
  sam_in_picture ρ_sam →
  probability_both_in_picture = 1 / 4 :=
by
  sorry

end probability_both_sarah_sam_in_picture_l112_112677


namespace remainder_n_plus_2023_l112_112270

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 7 = 3) : (n + 2023) % 7 = 3 :=
by sorry

end remainder_n_plus_2023_l112_112270


namespace part1_part2_l112_112619

-- Part (1): Prove that A = 2π/3 given the trigonometric condition.
theorem part1 (A B C : ℝ) (h_condition : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : 
  A = 2 * π / 3 :=
sorry

-- Part (2): Prove that the maximum perimeter is 3 + 2√3 given BC = 3 and A = 2π/3.
theorem part2 (B C : angle) (h_BC : 3 = 3) (h_A : A = 2 * π / 3) :
  (3 + (sqrt 3) * 2 ≤ perimeter (triangle.mk 3 B C)) :=
sorry

end part1_part2_l112_112619


namespace triangle_A_value_and_max_perimeter_l112_112575

theorem triangle_A_value_and_max_perimeter (A B C a b c : ℝ) 
  (h1 : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C)
  (h2 : a = 3) :
  (A = 2 * Real.pi / 3) ∧ (a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by
  sorry

end triangle_A_value_and_max_perimeter_l112_112575


namespace triangle_theorem_l112_112604

theorem triangle_theorem 
  (A B C : ℝ) 
  (h1 : ∀ A B C : ℝ, ∃ (a b c : ℝ), 
      a^2 - b^2 - c^2 = b * c ∧ sin^2 A - sin^2 B - sin^2 C = sin B * sin C) 
  (h2 : 0 < A ∧ A < π) : 
  (A = 2 * π / 3) ∧ 
  (∀ (BC : ℝ), BC = 3 → ∃ (a b c : ℝ), 
      a + b + c ≤ 3 + 2 * sqrt 3) :=
by
  sorry

end triangle_theorem_l112_112604


namespace derivative_at_pi_div_four_l112_112451

noncomputable def f (x : ℝ) : ℝ := sin x * (cos x + 1)

theorem derivative_at_pi_div_four : deriv f (π / 4) = sqrt 2 / 2 :=
by
  sorry

end derivative_at_pi_div_four_l112_112451


namespace find_angle_A_max_perimeter_triangle_l112_112549

-- Part 1: Prove the value of angle A
theorem find_angle_A (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter for BC = 3
theorem max_perimeter_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (A : ℝ) (hA : A = 2 * real.pi / 3) :
  let b := 2 * real.sqrt 3 * sin B,
      c := 2 * real.sqrt 3 * sin C,
      perimeter := 3 + b + c
  in ∀ d : ℝ, -real.pi / 6 < d ∧ d < real.pi / 6 → 
     B + C = real.pi / 3 → 
     perimeter ≤ (3 + 2 * real.sqrt 3) := sorry

end find_angle_A_max_perimeter_triangle_l112_112549


namespace cone_central_angle_l112_112853

theorem cone_central_angle (r : ℝ) (V : ℝ) (h : ℝ) (l : ℝ) (α : ℝ) 
  (hr : r = 1)
  (hV : V = (sqrt 3 * π) / 3)
  (hv : V = (1 / 3) * π * r^2 * h)
  (hl : l = sqrt (r^2 + h^2))
  (central_angle_formula : l * α = 2 * π * r) : 
  α = π :=
by
  sorry

end cone_central_angle_l112_112853


namespace problem_solution_l112_112227

noncomputable def dodecahedron_probability := 
  let m := 1
  let n := 100
  m + n

theorem problem_solution : dodecahedron_probability = 101 := by
  sorry

end problem_solution_l112_112227


namespace total_logs_combined_l112_112329

theorem total_logs_combined 
  (a1 l1 a2 l2 : ℕ) 
  (n1 n2 : ℕ) 
  (S1 S2 : ℕ) 
  (h1 : a1 = 15) 
  (h2 : l1 = 10) 
  (h3 : n1 = 6) 
  (h4 : S1 = n1 * (a1 + l1) / 2) 
  (h5 : a2 = 9) 
  (h6 : l2 = 5) 
  (h7 : n2 = 5) 
  (h8 : S2 = n2 * (a2 + l2) / 2) : 
  S1 + S2 = 110 :=
by {
  sorry
}

end total_logs_combined_l112_112329


namespace johns_profit_l112_112139

variable (numDucks : ℕ) (duckCost : ℕ) (duckWeight : ℕ) (sellPrice : ℕ)

def totalCost (numDucks duckCost : ℕ) : ℕ :=
  numDucks * duckCost

def totalWeight (numDucks duckWeight : ℕ) : ℕ :=
  numDucks * duckWeight

def totalRevenue (totalWeight sellPrice : ℕ) : ℕ :=
  totalWeight * sellPrice

def profit (totalRevenue totalCost : ℕ) : ℕ :=
  totalRevenue - totalCost

theorem johns_profit :
  totalCost 30 10 = 300 →
  totalWeight 30 4 = 120 →
  totalRevenue 120 5 = 600 →
  profit 600 300 = 300 :=
  by
    intros
    sorry

end johns_profit_l112_112139


namespace triangle_inequality_condition_l112_112047

noncomputable def f (x k : ℝ) : ℝ := x - Real.log x + k

theorem triangle_inequality_condition (a b c k : ℝ)
  (h₁ : a ∈ Set.Icc (1 / Real.exp 1) Real.exp 1)
  (h₂ : b ∈ Set.Icc (1 / Real.exp 1) Real.exp 1)
  (h₃ : c ∈ Set.Icc (1 / Real.exp 1) Real.exp 1)
  (h₄ : 2 * (f 1 k) > max (f (1 / Real.exp 1) k) (f (Real.exp 1) k))
  (h₅ : (f 1 k) > 0) :
  k > Real.exp 1 - 3 :=
sorry

end triangle_inequality_condition_l112_112047


namespace slope_of_tangent_line_l112_112871

theorem slope_of_tangent_line (f : ℝ → ℝ) (f_deriv : ∀ x, deriv f x = f x) (h_tangent : ∃ x₀, f x₀ = x₀ * deriv f x₀ ∧ (0 < f x₀)) :
  ∃ k, k = Real.exp 1 :=
by
  sorry

end slope_of_tangent_line_l112_112871


namespace probability_not_both_odd_l112_112002

theorem probability_not_both_odd :
  let S := {1, 2, 3, 4}
  let total_combinations := nat.choose 4 2
  let odd_combinations := nat.choose 2 2
  let probability_both_odd := odd_combinations / total_combinations
  let probability_not_both_odd := 1 - probability_both_odd
  (probability_not_both_odd) = (5 : ℝ) / 6
:= sorry

end probability_not_both_odd_l112_112002


namespace eraser_count_l112_112114

theorem eraser_count (initial_erasers jason_erasers mia_erasers : ℕ) :
    initial_erasers = 139 →
    jason_erasers = 131 →
    mia_erasers = 84 →
    initial_erasers + jason_erasers + mia_erasers = 354 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end eraser_count_l112_112114


namespace min_value_of_squares_l112_112644

-- We first define the given conditions as terms
def elements := {-6, -4, -1, 0, 3, 5, 7, 10}

def distinct_elements (a b c d e f g h : ℤ) : Prop :=
  list.nodup [a, b, c, d, e, f, g, h]

def sum_elements : ℤ := Finset.sum elements

def sum_is_14 : Prop := sum_elements = 14

-- Now we can state the main theorem
theorem min_value_of_squares
  (a b c d e f g h : ℤ) 
  (hdistinct : distinct_elements a b c d e f g h)
  (ha : a ∈ elements) 
  (hb : b ∈ elements) 
  (hc : c ∈ elements)
  (hd : d ∈ elements) 
  (he : e ∈ elements)
  (hf : f ∈ elements) 
  (hg : g ∈ elements) 
  (hh : h ∈ elements)
  (sum_eq : sum_elements = 14) : 
  (a + b + c + d)^2 + (e + f + g + h)^2 = 98 :=
by sorry

end min_value_of_squares_l112_112644


namespace sum_of_smallest_and_largest_l112_112201

def even_consecutive_sequence_sum (a n : ℤ) : ℤ :=
  a + a + 2 * (n - 1)

def arithmetic_mean (a n : ℤ) : ℤ :=
  (a * n + n * (n - 1)) / n

theorem sum_of_smallest_and_largest (a n y : ℤ) (h_even : even n) (h_mean : y = arithmetic_mean a n) :
  even_consecutive_sequence_sum a n = 2 * y :=
by
  sorry

end sum_of_smallest_and_largest_l112_112201


namespace minimum_value_of_quadratic_expression_l112_112733

theorem minimum_value_of_quadratic_expression : ∃ x ∈ ℝ, ∀ y ∈ ℝ, x^2 + 10 * x ≤ y^2 + 10 * y := by
  sorry

end minimum_value_of_quadratic_expression_l112_112733


namespace closest_clock_to_16_is_C_l112_112342

noncomputable def closestTo16InMirror (clock : Char) : Bool :=
  clock = 'C'

theorem closest_clock_to_16_is_C : 
  (closestTo16InMirror 'A' = False) ∧ 
  (closestTo16InMirror 'B' = False) ∧ 
  (closestTo16InMirror 'C' = True) ∧ 
  (closestTo16InMirror 'D' = False) := 
by
  sorry

end closest_clock_to_16_is_C_l112_112342


namespace triangle_theorem_l112_112608

theorem triangle_theorem 
  (A B C : ℝ) 
  (h1 : ∀ A B C : ℝ, ∃ (a b c : ℝ), 
      a^2 - b^2 - c^2 = b * c ∧ sin^2 A - sin^2 B - sin^2 C = sin B * sin C) 
  (h2 : 0 < A ∧ A < π) : 
  (A = 2 * π / 3) ∧ 
  (∀ (BC : ℝ), BC = 3 → ∃ (a b c : ℝ), 
      a + b + c ≤ 3 + 2 * sqrt 3) :=
by
  sorry

end triangle_theorem_l112_112608


namespace arccos_solution_l112_112192

theorem arccos_solution (x : ℝ) (h : real.arccos (2 * x) + real.arccos (3 * x) = π / 2) :
  x = 1 / real.sqrt 13 ∨ x = -1 / real.sqrt 13 :=
sorry

end arccos_solution_l112_112192


namespace min_intersection_value_l112_112988

theorem min_intersection_value (A B C : set α) [fintype A] [fintype B] [fintype C]
  (hA : fintype.card A = 50) (hB : fintype.card B = 50) (hC : fintype.card C = 50)
  (h_union : fintype.card A + fintype.card B + fintype.card C = 1.5 * fintype.card (A ∪ B ∪ C)) :
  fintype.card (A ∩ B ∩ C) = 48 := 
by { sorry }

end min_intersection_value_l112_112988


namespace bisects_AC_implies_bisects_BD_l112_112937

/-- Given a convex quadrilateral ABCD with points P and Q being the midpoints of sides AB and CD respectively,
    and given that the line segment PQ bisects the diagonal AC, prove that PQ also bisects the diagonal BD. -/
theorem bisects_AC_implies_bisects_BD
    (A B C D P Q M N : Point)
    (hP : midpoint A B P)
    (hQ : midpoint C D Q)
    (hM : midpoint A C M)
    (hN : midpoint B D N)
    (hPQ_bisects_AC : lies_on_line M (line_through P Q))
    : lies_on_line N (line_through P Q) :=
sorry

end bisects_AC_implies_bisects_BD_l112_112937


namespace probability_last_box_contains_exactly_one_ball_l112_112095

-- Definitions and conditions
def num_boxes : ℕ := 100
def num_balls : ℕ := 100
def p : ℝ := 1 / num_boxes.toReal

-- To show: The probability that the last box contains exactly one ball
theorem probability_last_box_contains_exactly_one_ball :
  ((1 - p) ^ (num_boxes - 1)) ≈ 0.370 :=
by
  sorry

end probability_last_box_contains_exactly_one_ball_l112_112095


namespace find_angle_A_max_perimeter_triangle_l112_112554

-- Part 1: Prove the value of angle A
theorem find_angle_A (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter for BC = 3
theorem max_perimeter_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (A : ℝ) (hA : A = 2 * real.pi / 3) :
  let b := 2 * real.sqrt 3 * sin B,
      c := 2 * real.sqrt 3 * sin C,
      perimeter := 3 + b + c
  in ∀ d : ℝ, -real.pi / 6 < d ∧ d < real.pi / 6 → 
     B + C = real.pi / 3 → 
     perimeter ≤ (3 + 2 * real.sqrt 3) := sorry

end find_angle_A_max_perimeter_triangle_l112_112554


namespace value_of_a_min_value_u_l112_112646

-- Problem 1: Finding the value of a
theorem value_of_a {a : ℝ} (f : ℝ → ℝ) (h : ∀ x, f x = abs (a * x - 1)) :
  (set_of (λ x, f x ≤ 2) = set.Icc (-1 : ℝ) (3 : ℝ)) → a = 1 :=
by
  sorry

-- Problem 2: Finding the minimum value of u
theorem min_value_u {x y z : ℝ} (h : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 1) :
  ∃ u_min : ℝ, (u_min = 3 ∧ ∀ u, u = (1 / (x + y) + (x + y) / z) ∧ u ≥ 3) :=
by
  sorry

end value_of_a_min_value_u_l112_112646


namespace probability_xi_le_sqrt6_correct_l112_112531

noncomputable def probability_xi_le_sqrt6 (total_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) : ℚ :=
let P_x0 := (3:ℚ) / 8
let P_x1 := (15:ℚ) / 56
let P_x2 := (10:ℚ) / 56
in P_x0 + P_x1 + P_x2

theorem probability_xi_le_sqrt6_correct (total_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) 
  (h_total : total_balls = 8) (h_white : white_balls = 5) (h_red : red_balls = 3) :
  probability_xi_le_sqrt6 total_balls white_balls red_balls = 23 / 28 :=
by {
  rw [h_total, h_white, h_red], 
  sorry
}

end probability_xi_le_sqrt6_correct_l112_112531


namespace polynomial_coefficient_product_identity_l112_112470

theorem polynomial_coefficient_product_identity (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h1 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 0)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = 32) :
  (a_0 + a_2 + a_4) * (a_1 + a_3 + a_5) = -256 := 
by {
  sorry
}

end polynomial_coefficient_product_identity_l112_112470


namespace simplify_and_evaluate_expression_l112_112188

theorem simplify_and_evaluate_expression (x : ℕ) (h : x = 2023) :
  (1 - (x / (x + 1:ℕ)) : ℚ) / ((x^2 - 2*x + 1) / (x^2 - 1) : ℚ) = 1 / 2022 := by
  have hx : (1 - (x / (x + 1):ℕ) : ℚ) = 1 / (x + 1) := sorry
  have hy : ((x^2 - 2*x + 1) / (x^2 - 1) : ℚ) = (x - 1) / (x + 1) := sorry
  rw [hx, hy]
  exact sorry

end simplify_and_evaluate_expression_l112_112188


namespace graphC_has_inverse_l112_112505

-- Define the nature of each graph.
def graphA (x : ℝ) : Prop := ( /* description for graph A's segments */ sorry )
def graphB (x : ℝ) : Prop := ( /* description for graph B's segments and discontinuities */ sorry )
def graphC (x : ℝ) : ℝ := x -- Continuous linear function
def graphD (x y : ℝ) : Prop := (x/4)^2 + (y/2)^2 = 1 -- Ellipse
def graphE (x : ℝ) : ℝ := x^3 / 30 + x^2 / 18 - x / 3 + 1 -- Cubic function

-- A predicate to check if a function has an inverse
def has_inverse (f : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, ∃! x : ℝ, f x = y

-- Proving that only graphC's function has an inverse
theorem graphC_has_inverse : has_inverse graphC ∧ ¬has_inverse graphA ∧ ¬has_inverse graphB ∧ ¬has_inverse graphD ∧ ¬has_inverse graphE :=
  by
  sorry

end graphC_has_inverse_l112_112505


namespace arithmetic_operators_l112_112177

-- Definitions based on conditions

def A : α
def B : β
def C : γ
def D : δ
def E : ε

-- Define the arithmetic and equality operations
def addition (x y : ℕ) := x + y
def subtraction (x y : ℕ) := x - y
def multiplication (x y : ℕ) := x * y
def division (x y : ℕ) := x / y
def equality (x y : ℕ) := x = y

-- Mathematically equivalent proof problem
theorem arithmetic_operators:
  (A = division) ∧
  (B = equality) ∧
  (C = multiplication) ∧
  (D = addition) ∧
  (E = subtraction) :=
by
  -- Define the equations with the letters
  let eq1 := equality (division 4 2) 2
  let eq2 := equality 8 (multiplication 4 2)
  let eq3 := equality (addition 2 3) 5
  let eq4 := equality 4 (subtraction 5 1)

  -- Conjoin all the assertions
  exact ⟨eq1, eq2, eq3, eq4⟩

sorry

end arithmetic_operators_l112_112177


namespace trigonometric_identity_l112_112832

theorem trigonometric_identity :
  (sin (20 * Real.pi / 180) * cos (15 * Real.pi / 180) + cos (160 * Real.pi / 180) * cos (105 * Real.pi / 180)) /
  (sin (25 * Real.pi / 180) * cos (10 * Real.pi / 180) + cos (155 * Real.pi / 180) * cos (95 * Real.pi / 180)) = 1 := 
by
  sorry

end trigonometric_identity_l112_112832


namespace probability_independent_events_probability_complement_l112_112196

variable (a b : ℕ) (ha : a > 1) (hb : b > 1) (hab : a > b)

def eventA (v : Type) (i : Type) : Prop := ∃ x : v, ∃ y : i, true
def eventB (v : Type) (i : Type) : Prop := ∃ x : v, true
def eventC (v : Type) (i : Type) : Prop := ∃ y : i, true

def P (event : Prop) : ℝ := sorry

theorem probability_independent_events (H : eventA ℝ ℝ) (H1 : eventB ℝ ℝ) (H2 : eventC ℝ ℝ) :
  P H = P H1 * P H2 :=
sorry

theorem probability_complement (H : eventA ℝ ℝ) (H1 : eventB ℝ ℝ) (H2 : eventC ℝ ℝ) :
  P (¬ H) > P (¬ H1 ∧ H2) + P (H1 ∧ ¬ H2) :=
sorry

end probability_independent_events_probability_complement_l112_112196


namespace minimize_z_l112_112918

theorem minimize_z (x y : ℝ) (h1 : 2 * x - y ≥ 0) (h2 : y ≥ x) (h3 : y ≥ -x + 2) :
  ∃ (x y : ℝ), (z = 2 * x + y) ∧ z = 8 / 3 :=
by
  sorry

end minimize_z_l112_112918


namespace problem_1_and_2_l112_112597

-- Definition of the problem in Lean 4
theorem problem_1_and_2 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin (A)^2 - sin (B)^2 - sin (C)^2 = sin (B) * sin (C))
  (h2 : BC = 3)
  (h3 : triangle ABC)
  (h4 : a = side_length (opposite A))
  (h5 : b = side_length (opposite B))
  (h6 : c = side_length (opposite C)) :
  A = 2 * real.pi / 3 ∧
  (a + b + c ≤ 3 + 2 * real.sqrt 3) :=
sorry

end problem_1_and_2_l112_112597


namespace smallest_positive_period_of_pi_min_value_of_g_on_interval_l112_112494

def f (x : ℝ) (ω : ℝ) : ℝ := sin (π - ω * x) * cos (ω * x) + cos (ω * x)^2

def g (x : ℝ) : ℝ := (sqrt 2 / 2) * sin (4 * x + π / 4) + 1/2

theorem smallest_positive_period_of_pi {ω : ℝ} (hω : ω > 0) : 
  (∀ x : ℝ, f x ω = f (x + π)) ↔ ω = 1 := sorry

theorem min_value_of_g_on_interval : 
  ∀ x ∈ set.Icc (0 : ℝ) (16*π),
  g x ≥ 1 := sorry

end smallest_positive_period_of_pi_min_value_of_g_on_interval_l112_112494


namespace last_box_probability_l112_112086

noncomputable def probability_last_box_only_ball : ℝ :=
  let n : ℕ := 100 in
  let p : ℝ := 1 / n in
  (n * p * (1 - p)^(n - 1))

theorem last_box_probability : abs (probability_last_box_only_ball - 0.3697) < 0.0005 := 
  sorry

end last_box_probability_l112_112086


namespace find_x_eq_7714285714285714_l112_112384

theorem find_x_eq_7714285714285714 (x : ℝ) (hx_pos : 0 < x) (h : floor x * x = 54) : x = 54 / 7 :=
by
  sorry

end find_x_eq_7714285714285714_l112_112384


namespace trajectory_is_square_l112_112243

-- Define the set of points where |x| + |y| = 1
def trajectory (x y : ℝ) : Prop := |x| + |y| = 1

-- State the main proof goal
theorem trajectory_is_square : ∀ x y : ℝ, trajectory x y →
  ((x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) ∨ (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = -1)) :=
  begin
    intros x y h,
    sorry -- The actual proof goes here
  end

end trajectory_is_square_l112_112243


namespace trapezoid_segments_relation_l112_112649

theorem trapezoid_segments_relation
  (A B C D M N : Point)
  (h_trapezoid : parallel (line A B) (line C D))
  (h_parallel : parallel (line A B) (line M N))
  (h_M_on_AD : M ∈ segment A D)
  (h_N_on_BC : N ∈ segment B C) :
  length (segment C D) * length (segment M A) + length (segment A B) * length (segment M D) = length (segment M N) * length (segment A D) :=
by
  sorry

end trapezoid_segments_relation_l112_112649


namespace and_implies_or_or_does_not_imply_and_and_is_sufficient_but_not_necessary_for_or_l112_112289

theorem and_implies_or (p q : Prop) (hpq : p ∧ q) : p ∨ q :=
by {
  sorry
}

theorem or_does_not_imply_and (p q : Prop) (hp_or_q : p ∨ q) : ¬ (p ∧ q) :=
by {
  sorry
}

theorem and_is_sufficient_but_not_necessary_for_or (p q : Prop) : (p ∧ q → p ∨ q) ∧ ¬ (p ∨ q → p ∧ q) :=
by {
  exact ⟨and_implies_or p q, or_does_not_imply_and p q⟩,
}

end and_implies_or_or_does_not_imply_and_and_is_sufficient_but_not_necessary_for_or_l112_112289


namespace ellipse_equation_range_of_k_l112_112877

-- Proof problem (Ⅰ)
theorem ellipse_equation (a b c : ℝ) (e : ℝ) (h1 : c = 3) (h2 : e = sqrt 3 / 2) (h3 : a > b) (h4 : b > 0) 
  (h5 : a^2 = b^2 + c^2) (h6 : e = c / a) : 
  (a = 2 * sqrt 3) → (b = sqrt 3) → ∀ x y : ℝ, (x^2 / 12) + (y^2 / 3) = 1 :=
begin
  sorry  -- (proof omitted)
end

-- Proof problem (Ⅱ)
theorem range_of_k (a e : ℝ) (h1 : a >= 2 * sqrt 3) (h2 : a < 3 * sqrt 2) (h3 : e = sqrt 3 / 2) 
  (h4 : e > sqrt 2 / 2) : 
  ∀ k : ℝ, ((k >= sqrt 2 / 4) ∨ (k <= -sqrt 2 / 4)) :=
begin
  sorry  -- (proof omitted)
end

end ellipse_equation_range_of_k_l112_112877


namespace sum_q_t_at_12_l112_112643

def T := {t : Fin 12 → Bool // true}

def q_t (t : T) : Polynomial ℝ :=
  Polynomial.ofFinList (λ n, if h : n < 12 then t.val ⟨n, h⟩ else 0)

noncomputable def q (x : ℕ) : ℝ :=
  ∑ t in Finset.univ.image subtype.val, q_t ⟨t, sorry⟩.eval x

theorem sum_q_t_at_12 : q 12 = 2048 := sorry

end sum_q_t_at_12_l112_112643


namespace num_triangles_PQR_with_area_500000_l112_112255

noncomputable def num_triangles_with_area (area : ℕ) : ℕ :=
  let qr := 2 * area in
  (nat.divisors qr).card

theorem num_triangles_PQR_with_area_500000 :
  num_triangles_with_area 500000 = 49 :=
by
  sorry

end num_triangles_PQR_with_area_500000_l112_112255


namespace no_zeroes_of_g_l112_112474

variable (f : ℝ → ℝ)

-- Conditions
hypothesis h1 : ∀ x > 0, x * (f' x) + f x > 0
noncomputable def g (x : ℝ) : ℝ := x * f x + 1

-- Statement to prove
theorem no_zeroes_of_g (h1: ∀ x > 0, x * (f' x) + f x > 0) :
  ¬(∃ x > 0, g f x = 0) :=
sorry

end no_zeroes_of_g_l112_112474


namespace kurt_savings_l112_112141

def daily_cost_old : ℝ := 0.85
def daily_cost_new : ℝ := 0.45
def days : ℕ := 30

theorem kurt_savings : (daily_cost_old * days) - (daily_cost_new * days) = 12.00 := by
  sorry

end kurt_savings_l112_112141


namespace last_box_probability_l112_112087

noncomputable def probability_last_box_only_ball : ℝ :=
  let n : ℕ := 100 in
  let p : ℝ := 1 / n in
  (n * p * (1 - p)^(n - 1))

theorem last_box_probability : abs (probability_last_box_only_ball - 0.3697) < 0.0005 := 
  sorry

end last_box_probability_l112_112087


namespace find_fifth_term_l112_112152

variable (a b x y : ℝ)

theorem find_fifth_term :
  ax + by = 5 →
  ax^2 + by^2 = 11 →
  ax^3 + by^3 = 24 →
  ax^4 + by^4 = 58 →
  ax^5 + by^5 = 273.09 :=
by
  intro h1 h2 h3 h4
  -- Proof goes here
  sorry

end find_fifth_term_l112_112152


namespace chord_length_of_ellipse_l112_112799

noncomputable def chord_length := 
  let a := 2
  let b := sqrt 3
  let y_eqn : Π x, Float := λ x => x + 1
  let ellipse_eqn : Π x y, Float := λ x y => x^2 / 4 + y^2 / 3 - 1
  let quad_eqn : Π x, Float := 
    λ x => (7 * x^2 + 8 * x - 8)
  let x1 := (-8 + sqrt (64 + 224)) / 14
  let x2 := (-8 - sqrt (64 + 224)) / 14
  sqrt (2 * (x1 + x2)^2 + 4 * (x1 * x2))

theorem chord_length_of_ellipse :
  chord_length = 24 / 7 := 
sorry

end chord_length_of_ellipse_l112_112799


namespace exist_three_integers_divisible_product_l112_112987

theorem exist_three_integers_divisible_product 
  (n : ℕ) (hn : 0 < n) :
  ∃ a b c : ℤ, 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  n^2 ≤ a ∧ a ≤ n^2 + n + 3 * real.sqrt n ∧ 
  n^2 ≤ b ∧ b ≤ n^2 + n + 3 * real.sqrt n ∧ 
  n^2 ≤ c ∧ c ≤ n^2 + n + 3 * real.sqrt n ∧ 
  a ∣ (b * c) := 
begin
  sorry
end

end exist_three_integers_divisible_product_l112_112987


namespace chess_tournament_schedule_l112_112302

theorem chess_tournament_schedule (n : ℕ) : 
  ∃ (schedule : list (ℕ × ℕ)) (all_pairs : finset (ℕ × ℕ)),
  (∀ (i j : ℕ), i ≠ j → i < n ∧ j < n → (i, j) ∈ all_pairs ∧ (j, i) ∈ all_pairs) ∧
  (∀ (rounds : list (ℕ × ℕ)), rounds.permutations → ∀ (i j : ℕ), 
    i < n → 
    j < n → 
    abs ((rounds.filter (λ x, x.1 = i)).length - 
         (rounds.filter (λ x, x.1 = j)).length) ≤ 1) :=
sorry

end chess_tournament_schedule_l112_112302


namespace angle_TCD_in_isosceles_trapezoid_l112_112286

theorem angle_TCD_in_isosceles_trapezoid
  (ABCD : Trapezoid)
  (isosceles_trapezoid : is_isosceles_trapezoid ABCD)
  (ADC : ∠ A D C = 82)
  (CAD_relation : ∠ C A D = 41)
  (T : Point)
  (CT_CD : dist C T = dist C D)
  (AT_TD : dist A T = dist T D)
  : ∠ T C D = 38 := sorry

end angle_TCD_in_isosceles_trapezoid_l112_112286


namespace prob_conditional_wind_given_rain_l112_112706

variable (P : Set → ℚ) (A B : Set)
variable (rain : ℚ := 4 / 15) (wind : ℚ := 2 / 15) (both : ℚ := 1 / 10)

noncomputable def conditional_probability := P B | A = (both / rain)

theorem prob_conditional_wind_given_rain : 
  P A = rain → P B = wind → P (A ∩ B) = both → conditional_probability :=
by
  sorry

end prob_conditional_wind_given_rain_l112_112706


namespace solve_n_and_sum_of_digits_l112_112785

noncomputable def factorial : ℕ → ℕ 
| 0       => 1
| (n+1)   => (n + 1) * factorial n

theorem solve_n_and_sum_of_digits (n : ℕ) 
(h1: (n + 1)! + (n + 2)! = n! * 1001) 
(h2: n > 0) : n = 30 ∧ (3 = 3) :=
begin
  sorry
end

end solve_n_and_sum_of_digits_l112_112785


namespace smallest_positive_real_x_l112_112393

theorem smallest_positive_real_x :
  ∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧ x = 89 / 9 :=
by
  sorry

end smallest_positive_real_x_l112_112393


namespace find_angle_A_max_perimeter_triangle_l112_112552

-- Part 1: Prove the value of angle A
theorem find_angle_A (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter for BC = 3
theorem max_perimeter_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (A : ℝ) (hA : A = 2 * real.pi / 3) :
  let b := 2 * real.sqrt 3 * sin B,
      c := 2 * real.sqrt 3 * sin C,
      perimeter := 3 + b + c
  in ∀ d : ℝ, -real.pi / 6 < d ∧ d < real.pi / 6 → 
     B + C = real.pi / 3 → 
     perimeter ≤ (3 + 2 * real.sqrt 3) := sorry

end find_angle_A_max_perimeter_triangle_l112_112552


namespace sum_of_super_cool_rectangles_l112_112319

theorem sum_of_super_cool_rectangles :
  let is_super_cool (a b : ℕ) := (a * b = 6 * (a + b))
  let areas := {ab | ∃ a b : ℕ, a ≠ b ∧ is_super_cool a b ∧ ab = a * b}
  let sum_of_areas := areas.sum id in
  sum_of_areas = 942 :=
by {
  let is_super_cool := λ a b : ℕ, a * b = 6 * (a + b),
  let areas := finset.insert 0 (finset.univ.filter (λ ab, ∃ a b : ℕ, a ≠ b ∧ a * b = 6 * (a + b) ∧ ab = a * b)),
  let sum_of_areas := areas.sum id,
  have h: sum_of_areas = 942 := sorry,
  exact h,
}

end sum_of_super_cool_rectangles_l112_112319


namespace simplified_expression_l112_112190

theorem simplified_expression :
  ( (81 / 16) ^ (3 / 4) - (-1) ^ 0 ) = 19 / 8 := 
by 
  -- It is a placeholder for the actual proof.
  sorry

end simplified_expression_l112_112190


namespace neg_of_forall_sin_ge_neg_one_l112_112468

open Real

theorem neg_of_forall_sin_ge_neg_one :
  (¬ (∀ x : ℝ, sin x ≥ -1)) ↔ (∃ x0 : ℝ, sin x0 < -1) := by
  sorry

end neg_of_forall_sin_ge_neg_one_l112_112468


namespace layers_removed_l112_112276

theorem layers_removed (n : ℕ) (original_volume remaining_volume side_length : ℕ) :
  original_volume = side_length^3 →
  remaining_volume = (side_length - 2 * n)^3 →
  original_volume = 1000 →
  remaining_volume = 512 →
  side_length = 10 →
  n = 1 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end layers_removed_l112_112276


namespace segment_CD_bisects_AB_l112_112165

-- Definitions of circles intersecting at two points and common tangents
variables (Γ1 Γ2 : Circle) (A B C D : Point)
variables (hΓ1Γ2_intersect : Intersect Γ1 Γ2 C D)
variables (tangent_point_A : IsTangentPoint Γ1 A)
variables (tangent_point_B : IsTangentPoint Γ2 B)
variables (common_tangent : Tangent Γ1 A Γ2 B)

-- Lean statement of the problem
theorem segment_CD_bisects_AB (M: Point) (hM_midpoint : IsMidpoint M A B) :
  LiesOn M (Line.through C D) :=
sorry

end segment_CD_bisects_AB_l112_112165


namespace count_non_congruent_rectangles_l112_112786

-- Definitions based on conditions given in the problem
def is_rectangle (w h : ℕ) : Prop := 2 * (w + h) = 40 ∧ w % 2 = 0

-- Theorem that we need to prove based on the problem statement
theorem count_non_congruent_rectangles : 
  ∃ n : ℕ, n = 9 ∧ 
  (∀ p : ℕ × ℕ, p ∈ { p | is_rectangle p.1 p.2 } → ∀ q : ℕ × ℕ, q ∈ { q | is_rectangle q.1 q.2 } → p = q ∨ p ≠ q) := 
sorry

end count_non_congruent_rectangles_l112_112786


namespace speed_of_river_l112_112310

theorem speed_of_river (speed_still_water : ℝ) (total_time : ℝ) (total_distance : ℝ) 
  (h_still_water: speed_still_water = 6) 
  (h_total_time: total_time = 1) 
  (h_total_distance: total_distance = 16/3) : 
  ∃ (speed_river : ℝ), speed_river = 2 :=
by 
  -- sorry is used to skip the proof
  sorry

end speed_of_river_l112_112310


namespace smallest_positive_real_number_l112_112405

noncomputable def smallest_x : ℝ :=
  Inf {x : ℝ | x > 0 ∧ (floor (x^2) - x * floor x = 8)}

theorem smallest_positive_real_number :
  smallest_x = 89 / 9 :=
by 
  sorry

end smallest_positive_real_number_l112_112405


namespace exam_score_l112_112955

theorem exam_score {total_questions correct_answers : ℕ} 
  (score_per_correct : ℕ := 4) 
  (penalty_per_wrong : ℤ := -1) 
  (total_questions = 75) 
  (correct_answers = 40) 
  : 
  let wrong_answers := total_questions - correct_answers in
  let total_score := (correct_answers * score_per_correct) + (wrong_answers * penalty_per_wrong) in
  total_score = 125 :=
by
  sorry

end exam_score_l112_112955


namespace sphere_in_cube_volume_unreachable_l112_112789

noncomputable def volume_unreachable_space (cube_side : ℝ) (sphere_radius : ℝ) : ℝ :=
  let corner_volume := 64 - (32/3) * Real.pi
  let edge_volume := 288 - 72 * Real.pi
  corner_volume + edge_volume

theorem sphere_in_cube_volume_unreachable : 
  (volume_unreachable_space 6 1 = 352 - (248 * Real.pi / 3)) :=
by
  sorry

end sphere_in_cube_volume_unreachable_l112_112789


namespace fence_cost_l112_112269

noncomputable def area_of_square : ℝ := 289
noncomputable def price_per_foot : ℝ := 56

theorem fence_cost : 
  let side_length := Real.sqrt area_of_square in
  let perimeter := 4 * side_length in
  let cost := perimeter * price_per_foot in
  cost = 3808 :=
by 
  have side_length_eq : side_length = Real.sqrt area_of_square := rfl
  have perimeter_eq : perimeter = 4 * side_length := rfl
  have cost_eq : cost = perimeter * price_per_foot := rfl
  sorry

end fence_cost_l112_112269


namespace problem_1_and_2_l112_112595

-- Definition of the problem in Lean 4
theorem problem_1_and_2 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin (A)^2 - sin (B)^2 - sin (C)^2 = sin (B) * sin (C))
  (h2 : BC = 3)
  (h3 : triangle ABC)
  (h4 : a = side_length (opposite A))
  (h5 : b = side_length (opposite B))
  (h6 : c = side_length (opposite C)) :
  A = 2 * real.pi / 3 ∧
  (a + b + c ≤ 3 + 2 * real.sqrt 3) :=
sorry

end problem_1_and_2_l112_112595


namespace domain_of_f_value_at_half_maximum_value_of_f_l112_112490

noncomputable theory

def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 5

-- Prove the domain of f(x) is all real numbers ℝ.
theorem domain_of_f : ∀ x : ℝ, f x = -2 * x^2 + 4 * x - 5 :=
by sorry

-- Prove that f(1/2) = -7/2.
theorem value_at_half : f (1/2) = -7 / 2 :=
by sorry

-- Prove that the maximum value of f(x) is -3.
theorem maximum_value_of_f : ∃ x : ℝ, f x = -3 :=
by sorry

end domain_of_f_value_at_half_maximum_value_of_f_l112_112490


namespace cube_volume_is_27_l112_112306

noncomputable def original_cube_edge (a : ℝ) : ℝ := a

noncomputable def original_cube_volume (a : ℝ) : ℝ := a^3

noncomputable def new_rectangular_solid_volume (a : ℝ) : ℝ := (a-2) * a * (a+2)

theorem cube_volume_is_27 (a : ℝ) (h : original_cube_volume a - new_rectangular_solid_volume a = 14) : original_cube_volume a = 27 :=
by
  sorry

end cube_volume_is_27_l112_112306


namespace lambda_value_l112_112079

noncomputable def cubic_expr (x : ℝ) (λ : ℝ) : ℝ :=
  3 * x ^ 2 - λ * x + 1

theorem lambda_value (λ : ℝ) : 
  (∀ x ∈ set.Icc (1/2) 2, cubic_expr x λ > 0) ↔ λ = 2 * real.sqrt 2 :=
sorry

end lambda_value_l112_112079


namespace minimize_quadratic_l112_112268

theorem minimize_quadratic : ∃ x : ℝ, x = -4 ∧ ∀ y : ℝ, x^2 + 8*x + 7 ≤ y^2 + 8*y + 7 :=
by 
  use -4
  sorry

end minimize_quadratic_l112_112268


namespace trisectors_angle_120_l112_112858

/- Definitions and Assumptions -/
structure Triangle := (A B C : point)
structure Point := (x y : Real)
structure Circle := (center : Point) (radius : Real)

variables (ABC : Triangle)
variable (A1 B1 C1 : Point)
variable (A0 B0 C0 : Point)
variable (F : Point)
variable (angleA : angle (ABC.A))
variable (angleB : angle (ABC.B))

/- Conditions -/
axiom angle_A_eq_45 : angleA = 45
axiom angle_B_eq_60 : angleB = 60 

/- Define altitudes' feet and midpoints of sides -/
axiom foot_of_altitude : is_foot_of_altitude A1 B1 C1 ABC
axiom midpoint_of_sides : is_midpoint_of_sides A0 B0 C0 ABC

/- Define the circle properties -/
axiom circle_passing_through_midpoints : is_circumscribed_circle A0 B0 C0 F

/- Angle Trisectors Definition -/
variables (FX FY FZ : ray F)
axiom trisectors :
  (angle (ray_angle FX A1) = 2 * angle (ray_angle A0 FX)) ∧
  (angle (ray_angle FY B1) = 2 * angle (ray_angle B0 FY)) ∧
  (angle (ray_angle FZ C1) = 2 * angle (ray_angle C0 FZ))

/- Proof Statement -/
theorem trisectors_angle_120 :
  ∀ (FX FY FZ : ray F),
    (angle_between FX FY = 120) ∧
    (angle_between FY FZ = 120) ∧
    (angle_between FZ FX = 120) :=
begin
  sorry
end

end trisectors_angle_120_l112_112858


namespace solve_problem_l112_112483

-- Define the given conditions
def condition1 (a : ℝ) : Prop := real.cbrt (5 * a + 2) = 3
def condition2 (a b : ℝ) : Prop := real.sqrt (3 * a + b - 1) = 4
def condition3 (c : ℝ) : Prop := c = real.floor (real.sqrt 13)

-- Define the target values and resulting statement
def values_a_b_c (a b c : ℝ) : Prop := 
  a = 5 ∧ b = 2 ∧ c = 3

def final_sqrt (a b c : ℝ) : Prop :=
  real.sqrt (3 * a - b + c) = 4 ∨ real.sqrt (3 * a - b + c) = -4

-- Complete the statement bringing everything together
theorem solve_problem (a b c : ℝ) :
  condition1 a →
  condition2 a b →
  condition3 c →
  values_a_b_c a b c ∧ final_sqrt a b c :=
begin
  sorry
end

end solve_problem_l112_112483


namespace common_elements_count_l112_112990

-- Define the sets X and Y based on the problem's conditions
def X : Set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 3000 ∧ n = 5 * k}
def Y : Set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 3000 ∧ n = 8 * k}

-- Statement to prove that the number of common elements in X and Y is 375
theorem common_elements_count : (X ∩ Y).toFinset.card = 375 := 
sorry

end common_elements_count_l112_112990


namespace hexagon_perimeter_theorem_l112_112542

noncomputable def hexagon_perimeter {α : Type*} [integral_domain α] (s : α) : α :=
  6 * s

theorem hexagon_perimeter_theorem {α : Type*} [linear_ordered_field α] :
  ∀ (hexagon : Type*) (angles : finset (hexagon → ℝ)) (area : α),
    (∀ a b, angle_measure a b = 45) →
    (∃ s : α, area = 12 * real.sqrt 2 ∧ ∀ a, length a = s) →
         hexagon_perimeter (real.sqrt 6) = 6 * real.sqrt 6 :=
begin
  -- Proof here
  sorry
end

end hexagon_perimeter_theorem_l112_112542


namespace find_quadratic_polynomial_with_conditions_l112_112385

noncomputable def quadratic_polynomial : polynomial ℝ :=
  3 * (X - C (2 + 2 * I)) * (X - C (2 - 2 * I))

theorem find_quadratic_polynomial_with_conditions :
  (quadratic_polynomial = 3 * X^2 - 12 * X + 24) :=
by
  sorry

end find_quadratic_polynomial_with_conditions_l112_112385


namespace sequence_contains_even_l112_112788

noncomputable def sequence (a1 : ℕ) (n : ℕ) : ℕ :=
nat.rec_on n a1 (λ n a_n, 5 + (n * (n + 1)) / 2)

theorem sequence_contains_even (a1 : ℕ) (h : a1 > 5) :
  ∃ n, (sequence a1 n) % 2 = 0 :=
sorry

end sequence_contains_even_l112_112788


namespace part_a_part_b_l112_112158

/- Part (a) -/
theorem part_a (a b c d : ℝ) (h1 : (a + b ≠ c + d)) (h2 : (a + c ≠ b + d)) (h3 : (a + d ≠ b + c)) :
  ∃ (spheres : ℕ), spheres = 8 := sorry

/- Part (b) -/
theorem part_b (a b c d : ℝ) (h : (a + b = c + d) ∨ (a + c = b + d) ∨ (a + d = b + c)) :
  ∃ (spheres : ℕ), ∀ (n : ℕ), n > 0 → spheres = n := sorry

end part_a_part_b_l112_112158


namespace count_numbers_in_sequence_l112_112064

theorem count_numbers_in_sequence : 
  ∃ n : ℕ, n = 31 ∧ ∀ (a d : ℤ) (terms : List ℤ), 
    a = 156 → d = -4 →
    terms = List.range (n + 1) 36 \ ((n : Int) - 1) * d + 36) 
    → List.last terms = some 36 := 
begin
  sorry
end

end count_numbers_in_sequence_l112_112064


namespace angle_A_existence_uniqueness_l112_112968

-- Define the conditions
def triangle_ABC (a b c A B C : ℝ) : Prop :=
  b * sin (2 * A) = sqrt 3 * a * sin B

def area_triangle_ABC (a b c A : ℝ) (area : ℝ) : Prop :=
  1/2 * b * c * sin A = area

def ratio_b_c (b c : ℝ) : Prop :=
  b / c = 3 * sqrt 3 / 4

-- The proof statements
theorem angle_A (a b c A B C : ℝ) (H1 : triangle_ABC a b c A B C) :
  A = π / 6 :=
by sorry

theorem existence_uniqueness (a b c A B C area : ℝ) 
  (H1 : triangle_ABC a b c A B C)
  (H2 : area_triangle_ABC a b c A area)
  (H3 : ratio_b_c b c)
  (H4 : area = 3 * sqrt 3) :
  a = sqrt 7 :=
by sorry

end angle_A_existence_uniqueness_l112_112968


namespace find_PF1_l112_112888

noncomputable theory

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Define point P on hyperbola
variables (P : ℝ × ℝ) (on_hyperbola : hyperbola P.1 P.2)

-- Define the vectors and the dot product condition
def vector_F1_F2 : ℝ × ℝ := (F2.1 - F1.1, F2.2 - F1.2)
def vector_PF2 : ℝ × ℝ := (F2.1 - P.1, F2.2 - P.2)
def dot_product_zero : Prop := vector_F1_F2.1 * vector_PF2.1 + vector_F1_F2.2 * vector_PF2.2 = 0

-- Define the distances
def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem to prove
theorem find_PF1 (h_hyperbola : hyperbola P.1 P.2)
  (h_dot_product : dot_product_zero) : distance P F1 = 13 / 2 := sorry

end find_PF1_l112_112888


namespace find_smallest_x_l112_112423

noncomputable def smallest_x : ℝ :=
  min { x : ℝ | 0 < x ∧ (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * x = 8 }

theorem find_smallest_x :
  smallest_x = 89 / 9 :=
by
  sorry

end find_smallest_x_l112_112423


namespace distinct_nonzero_digits_sum_l112_112055

theorem distinct_nonzero_digits_sum (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) 
  (h7 : 100*a + 10*b + c + 100*a + 10*c + b + 100*b + 10*a + c + 100*b + 10*c + a + 100*c + 10*a + b + 100*c + 10*b + a = 1776) : 
  (a = 1 ∧ b = 2 ∧ c = 5) ∨ (a = 1 ∧ b = 3 ∧ c = 4) ∨ (a = 1 ∧ b = 4 ∧ c = 3) ∨ (a = 1 ∧ b = 5 ∧ c = 2) ∨ (a = 2 ∧ b = 1 ∧ c = 5) ∨
  (a = 2 ∧ b = 5 ∧ c = 1) ∨ (a = 3 ∧ b = 1 ∧ c = 4) ∨ (a = 3 ∧ b = 4 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 3 ∧ c = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = 2) ∨ (a = 5 ∧ b = 2 ∧ c = 1) :=
sorry

end distinct_nonzero_digits_sum_l112_112055


namespace minimum_value_of_expression_l112_112517

theorem minimum_value_of_expression {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    ∀ x y : ℝ, x = b / a^2 ∧ y = 4 / b ∧ ∀ z : ℝ, z = a / 2 
    ∧ (x + y + z = 2 * Real.sqrt 2) ∧ (b = 2 * a) ∧ (a = 2 * Real.sqrt 2) ∧ (b = 4 * Real.sqrt 2)) :=
begin
  sorry
end

end minimum_value_of_expression_l112_112517


namespace leftmost_three_digits_eq_317_l112_112022

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def num_arrangements (total_rings rings_selected fingers : ℕ) : ℕ :=
  binomial total_rings rings_selected * Nat.factorial rings_selected * binomial (rings_selected + fingers - 1) fingers

theorem leftmost_three_digits_eq_317 :
  let n := num_arrangements 10 6 5 in
  (n / 1000) % 1000 = 317 := by
  sorry

end leftmost_three_digits_eq_317_l112_112022


namespace value_of_square_l112_112073

theorem value_of_square :
  ∃ (square : ℕ), 
  (Nat.toDigits 7 (5 * 343 + 3 * 49 + 2 * 7 + square) = [5, 3, 2, square]) ∧
  (Nat.toDigits 7 (square * 49 + 6 * 7) = [square, 6, 0]) ∧
  (Nat.toDigits 7 (square * 7 + 3) = [square, 3]) ∧
  (Nat.toDigits 7 (6 * 343 + 4 * 49 + square * 7 + 1) = [6, 4, square, 1])
:=
  ∃ (square: ℕ), square = 5
  sorry

end value_of_square_l112_112073


namespace onions_total_l112_112676

theorem onions_total (Sara : ℕ) (Sally : ℕ) (Fred : ℕ)
  (hSara : Sara = 4) (hSally : Sally = 5) (hFred : Fred = 9) :
  Sara + Sally + Fred = 18 :=
by
  sorry

end onions_total_l112_112676


namespace probability_last_box_contains_exactly_one_ball_l112_112096

-- Definitions and conditions
def num_boxes : ℕ := 100
def num_balls : ℕ := 100
def p : ℝ := 1 / num_boxes.toReal

-- To show: The probability that the last box contains exactly one ball
theorem probability_last_box_contains_exactly_one_ball :
  ((1 - p) ^ (num_boxes - 1)) ≈ 0.370 :=
by
  sorry

end probability_last_box_contains_exactly_one_ball_l112_112096


namespace find_angle_A_max_perimeter_l112_112561

noncomputable def sin_sq_minus (A B C : ℝ) : ℝ :=
  (Real.sin A) * (Real.sin A) - (Real.sin B) * (Real.sin B) - (Real.sin C) * (Real.sin C)

noncomputable def sin_prod (B C : ℝ) : ℝ :=
  (Real.sin B) * (Real.sin C)

theorem find_angle_A (A B C : ℝ) (h : sin_sq_minus A B C = sin_prod B C) :
  A = 2 * Real.pi / 3 :=
by
  sorry

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem max_perimeter (B C : ℝ) (a : ℝ) (h1 : a = 3) (h2 : B + C = Real.pi / 3) :
  ∃ (b c : ℝ), perimeter a b c = 3 + 2 * Real.sqrt 3 :=
by
  sorry

end find_angle_A_max_perimeter_l112_112561


namespace part1_part2_l112_112625

theorem part1 (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : A = 2 * π / 3 :=
sorry

theorem part2 (b c : ℝ) (h1 : sin (2 * π / 3) ^ 2 - sin b ^ 2 - sin c ^ 2 = sin b * sin c)
  (h2 : b + c = 2 * sqrt 3) : 3 * 2 + b + c = 3 + 2 * sqrt 3 :=
sorry

end part1_part2_l112_112625


namespace smallest_positive_real_number_l112_112399

noncomputable def smallest_x : ℝ :=
  let x := 89 / 9 in x

theorem smallest_positive_real_number :
  ∀ x : ℝ, (x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8) → x ≥ smallest_x  :=
by
  sorry

end smallest_positive_real_number_l112_112399


namespace grasshopper_total_distance_l112_112776

theorem grasshopper_total_distance :
  let initial := 2
  let first_jump := -3
  let second_jump := 8
  let final_jump := -1
  abs (first_jump - initial) + abs (second_jump - first_jump) + abs (final_jump - second_jump) = 25 :=
by
  sorry

end grasshopper_total_distance_l112_112776


namespace petya_coloring_l112_112284

theorem petya_coloring (n : ℕ) (h₁ : n = 100) (h₂ : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → 1 ≤ (number i j) ∧ (number i j) ≤ n * n) :
  ∃ k, k = 1 ∧ ∀ (initial_coloring : fin (n * n) → bool) (next_colorable : (fin (n * n) → bool) → (fin (n * n) → bool)),
    (∀ (table : fin n × fin n → fin (n * n)), next_colorable (λ a, initial_coloring a)
    (λ a, initial_coloring a) a) :=
begin
  sorry,
end

end petya_coloring_l112_112284


namespace quadratic_not_factored_l112_112163

theorem quadratic_not_factored
  (a b c : ℕ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (p : ℕ)
  (h_prime_p : Nat.Prime p)
  (h_p : a * 1991^2 + b * 1991 + c = p) :
  ¬ (∃ d₁ d₂ e₁ e₂ : ℤ, a = d₁ * d₂ ∧ b = d₁ * e₂ + d₂ * e₁ ∧ c = e₁ * e₂) :=
sorry

end quadratic_not_factored_l112_112163


namespace mouse_to_cheese_in_expected_steps_l112_112260

noncomputable def expected_steps (p_A_to_B p_B_to_A p_B_to_C p_C_to_B p_C_to_Cheese : ℝ) : ℝ :=
  1 / (1 - (1 - p_B_to_C * p_C_to_Cheese) * p_B_to_C) * (3 * p_B_to_C * p_C_to_Cheese 
  + 2 * p_B_to_C * (1 - p_C_to_Cheese) * p_C_to_Cheese / (1 - (1 - p_B_to_C * p_C_to_Cheese) * p_B_to_C))

theorem mouse_to_cheese_in_expected_steps : 
  expected_steps 1 (1/2) (1/2) (4/5) (1/5) = 21 := 
  by sorry

end mouse_to_cheese_in_expected_steps_l112_112260


namespace find_A_max_perimeter_of_triangle_l112_112571

-- Definition of the given problem conditions
def triangle_condition (A B C : ℝ) : Prop :=
  sin(A)^2 - sin(B)^2 - sin(C)^2 = sin(B) * sin(C)

-- (1) Proving the value of A given the condition
theorem find_A (A B C : ℝ) (h : triangle_condition A B C) : A = 2 * π / 3 :=
by sorry

-- (2) Proving the maximum perimeter given BC = 3 and A = 2π/3
theorem max_perimeter_of_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (hA : 2 * π / 3 = 2 * π / 3) : 
  ∃ (P : ℝ), P = 3 + 2 * sqrt 3 :=
by sorry

end find_A_max_perimeter_of_triangle_l112_112571


namespace xy_min_value_l112_112452

theorem xy_min_value (x y : ℝ) (h1 : x < 0) (h2 : y < 0) (h3 : x + y = -1) : 
  xy + (1 / xy) >= 17 / 4 := 
begin
  sorry,
end

end xy_min_value_l112_112452


namespace geometric_sequence_common_ratio_l112_112775

theorem geometric_sequence_common_ratio :
  ∃ r : ℚ, 
  ∃ (a1 a2 a3 a4 : ℚ), 
  a1 = 16 ∧ a2 = -24 ∧ a3 = 36 ∧ a4 = -54 ∧ 
  r = a2 / a1 ∧ r = -3 / 2 :=
begin
  use -3 / 2,
  use 16, use -24, use 36, use -54,
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  split,
  { norm_num },
  { norm_num }
end

end geometric_sequence_common_ratio_l112_112775


namespace modified_game_ends_in_25_rounds_l112_112116

theorem modified_game_ends_in_25_rounds :
  ∃ rounds : ℕ, rounds = 25 ∧ 
  ∃ tokens_A tokens_B tokens_C tokens_D : ℕ,
  tokens_A = 16 ∧ tokens_B = 15 ∧ tokens_C = 14 ∧ tokens_D = 13 ∧
  (∀ r < rounds, let (max_tokens, total_discard, updated_A, updated_B, updated_C, updated_D) := 
    if tokens_A ≥ tokens_B ∧ tokens_A ≥ tokens_C ∧ tokens_A ≥ tokens_D then 
      (tokens_A, tokens_A - 5, tokens_A - 5, tokens_B + 1, tokens_C + 1, tokens_D + 1)
    else if tokens_B ≥ tokens_A ∧ tokens_B ≥ tokens_C ∧ tokens_B ≥ tokens_D then 
      (tokens_B, tokens_B - 5, tokens_A + 1, tokens_B - 5, tokens_C + 1, tokens_D + 1)
    else if tokens_C ≥ tokens_A ∧ tokens_C ≥ tokens_B ∧ tokens_C ≥ tokens_D then 
      (tokens_C, tokens_C - 5, tokens_A + 1, tokens_B + 1, tokens_C - 5, tokens_D + 1)
    else 
      (tokens_D, tokens_D - 5, tokens_A + 1, tokens_B + 1, tokens_C + 1, tokens_D - 5)
  in tokens_A := updated_A ∧ tokens_B := updated_B ∧ tokens_C := updated_C ∧ tokens_D := updated_D) ∧
  (tokens_A = 0 ∨ tokens_B = 0 ∨ tokens_C = 0 ∨ tokens_D = 0) :=
begin
  sorry -- The proof will go here
end

end modified_game_ends_in_25_rounds_l112_112116


namespace magnitude_conjugate_sub_i_l112_112453

noncomputable def z : ℂ := 1 - 3 * Complex.I

theorem magnitude_conjugate_sub_i : abs (conj z - Complex.I) = Real.sqrt 5 := 
by
  -- Proof steps are omitted
  sorry

end magnitude_conjugate_sub_i_l112_112453


namespace bottle_caps_vs_wrappers_l112_112359

theorem bottle_caps_vs_wrappers (bottle_caps wrappers : ℕ) (h1 : bottle_caps = 50) (h2 : wrappers = 46) :
  bottle_caps - wrappers = 4 :=
by
  -- Using the given conditions directly
  rw [h1, h2]
  norm_num
  exact Nat.sub_self 4 -- Simplifying 50 - 46 to 4
  sorry

end bottle_caps_vs_wrappers_l112_112359


namespace find_quadratic_polynomial_with_conditions_l112_112386

noncomputable def quadratic_polynomial : polynomial ℝ :=
  3 * (X - C (2 + 2 * I)) * (X - C (2 - 2 * I))

theorem find_quadratic_polynomial_with_conditions :
  (quadratic_polynomial = 3 * X^2 - 12 * X + 24) :=
by
  sorry

end find_quadratic_polynomial_with_conditions_l112_112386


namespace part1_part2_part3_l112_112431

section part1

variables {a : ℕ → ℤ} {b : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) :=
  a 1 = a1 ∧ ∀ n, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℤ) (b1 q : ℤ) :=
  b 1 = b1 ∧ ∀ n, b (n + 1) = b n * q

def a_n (n : ℕ) : ℤ := 2 * n - 1
def b_n (n : ℕ) : ℤ := 2 ^ n

theorem part1 (S10 : ℤ) (h_a : is_arithmetic_sequence a 1 2) (h_S10 : S10 = 100) 
(h_b1 : b 1 = 2 * a 1) (h_b3 : b 3 = a 4 + 1) : 
(a_n = λ n, 2 * n - 1) ∧ (b_n = λ n, 2 ^ n) :=
sorry
end part1

section part2

def sum_2n_plus_1 (f : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range (2 * n + 1), f i

def s_n (n : ℕ) : ℕ → ℤ := 
  λ n, (-1)^(n-1) * (a_n n + 1)^2

theorem part2 (n : ℕ): 
sum_2n_plus_1 s_n n = 8 * n^2 + 12 * n + 4 :=
sorry

end part2

section part3

def c_n (n : ℕ) : ℕ → ℤ := 
λ n, if n % 2 = 1 then a_n n * b_n n else (3 * a_n n - 4) * b_n n / (a_n n * a_n (n + 2))

theorem part3 (n : ℕ): 
(∑ i in (finset.range (2 * n)), c_n i) = (12 * n - 13) * 2 ^ (2 * n + 1) / 9 + 14 / 9 + 2 ^ (n + 2) / (4 * n + 3) :=
sorry

end part3

end part1_part2_part3_l112_112431


namespace not_function_mapping_l112_112019

def P := {x : ℝ | 0 ≤ x ∧ x ≤ 4}
def N := {y : ℝ | 0 ≤ y ∧ y ≤ 2}

def f (x : ℝ) := (2 / 3) * x

theorem not_function_mapping : ∃ x ∈ P, f x ∉ N := by
  sorry

end not_function_mapping_l112_112019


namespace rectangle_semi_perimeter_l112_112123

variables (BC AC AM x y : ℝ)

theorem rectangle_semi_perimeter (hBC : BC = 5) (hAC : AC = 12) (hAM : AM = x)
  (hMN_AC : ∀ (MN : ℝ), MN = 5 / 12 * AM)
  (hNP_BC : ∀ (NP : ℝ), NP = AC - AM)
  (hy_def : y = (5 / 12 * x) + (12 - x)) :
  y = (144 - 7 * x) / 12 :=
sorry

end rectangle_semi_perimeter_l112_112123


namespace sum_of_smallest_and_largest_eq_2y_l112_112202

variable (a n y : ℤ) (hn_even : Even n) (hy : y = a + n - 1)

theorem sum_of_smallest_and_largest_eq_2y : a + (a + 2 * (n - 1)) = 2 * y := 
by
  sorry

end sum_of_smallest_and_largest_eq_2y_l112_112202


namespace triangle_A_value_and_max_perimeter_l112_112578

theorem triangle_A_value_and_max_perimeter (A B C a b c : ℝ) 
  (h1 : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C)
  (h2 : a = 3) :
  (A = 2 * Real.pi / 3) ∧ (a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by
  sorry

end triangle_A_value_and_max_perimeter_l112_112578


namespace sum_of_smallest_and_largest_l112_112200

def even_consecutive_sequence_sum (a n : ℤ) : ℤ :=
  a + a + 2 * (n - 1)

def arithmetic_mean (a n : ℤ) : ℤ :=
  (a * n + n * (n - 1)) / n

theorem sum_of_smallest_and_largest (a n y : ℤ) (h_even : even n) (h_mean : y = arithmetic_mean a n) :
  even_consecutive_sequence_sum a n = 2 * y :=
by
  sorry

end sum_of_smallest_and_largest_l112_112200


namespace angle_ACB_30_degrees_l112_112129

/-- 
In triangle ABC, AB = 3 * AC. Points D and E are situated on sides AB and BC such that 
∠BAE = ∠ACD = x. Denote F as the intersection of line segments AE and CD, with 
triangle CFE being a right-angled triangle at F. Then, ∠ACB = 30 degrees.
-/
theorem angle_ACB_30_degrees
  (A B C D E F : Type) [metric_space A] [normed_group B] [normed_space ℝ B]
  (AB AC AE CD : ℝ) (x : ℝ)
  (hAB : AB = 3 * AC)
  (hAngleBAE : ∠BAE = x)
  (hAngleACD : ∠ACD = x)
  (hRightTriCFE : ∠CFE = 90)
  (hIntersection : F = AE ∩ CD) :
  ∠ACB = 30 :=
sorry

end angle_ACB_30_degrees_l112_112129


namespace todd_saved_44_dollars_l112_112713

-- Definitions of the conditions.
def full_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon_discount : ℝ := 10
def credit_card_discount : ℝ := 0.10

-- The statement we want to prove: Todd saved $44 on the original price of the jeans.
theorem todd_saved_44_dollars :
  let sale_amount := full_price * sale_discount,
      price_after_sale := full_price - sale_amount,
      price_after_coupon := price_after_sale - coupon_discount,
      credit_card_amount := price_after_coupon * credit_card_discount,
      final_price := price_after_coupon - credit_card_amount,
      savings := full_price - final_price
  in savings = 44 :=
by
  sorry

end todd_saved_44_dollars_l112_112713


namespace compute_x_pi_over_4_l112_112639

noncomputable def x (t : ℝ) : ℝ := sorry

theorem compute_x_pi_over_4 :
  ∀ (x : ℝ → ℝ), 
    ((x + (deriv x)) ^ 2 + x * (deriv (deriv x)) = λ t, cos t) →
    (x 0 = sqrt (2 / 5)) →
    ((deriv x) 0 = sqrt (2 / 5)) →
    (x (π / 4) = sqrt ((-(8 + π) / 5) * exp (-(π / 4)) + sqrt 2)) :=
by
  sorry

end compute_x_pi_over_4_l112_112639


namespace not_nec_perpendicular_planes_l112_112873

-- Define that the planes, lines, and perpendicular conditions are established in the space.
variables {α β : Plane} {a b l : Line}
variables [a_subset_α : a ⊂ α] [b_subset_β : b ⊂ β]
variables [intersect : α ∩ β = l]
variables [perpendicular_a_l : Perpendicular a l]
variables [perpendicular_b_l : Perpendicular b l]

-- Define the theorem statement.
theorem not_nec_perpendicular_planes :
  ¬ (α ⊥ β) :=
sorry

end not_nec_perpendicular_planes_l112_112873


namespace find_angle_A_max_perimeter_triangle_l112_112550

-- Part 1: Prove the value of angle A
theorem find_angle_A (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter for BC = 3
theorem max_perimeter_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (A : ℝ) (hA : A = 2 * real.pi / 3) :
  let b := 2 * real.sqrt 3 * sin B,
      c := 2 * real.sqrt 3 * sin C,
      perimeter := 3 + b + c
  in ∀ d : ℝ, -real.pi / 6 < d ∧ d < real.pi / 6 → 
     B + C = real.pi / 3 → 
     perimeter ≤ (3 + 2 * real.sqrt 3) := sorry

end find_angle_A_max_perimeter_triangle_l112_112550


namespace combined_weight_new_people_l112_112209

theorem combined_weight_new_people 
  (W : ℕ) -- The total initial weight of the group of 15 people
  (A B : ℕ) -- The weights of the two new people
  (h1 : 60 + 75 = 135)
  (h2 : 15 * 7 = 105)
  (h3 : W - 135 + A + B = W + 105)
  : A + B = 240 := 
by {
  rw [h1, h2] at h3,
  linarith,
}

end combined_weight_new_people_l112_112209


namespace bananas_lemons_l112_112922

variable {Banana Apple Orange Lemon : Type}
variable cost : (Banana → ℝ) → (Apple → ℝ) → (Orange → ℝ) → (Lemon → ℝ) → Prop

axiom bananas_apples : cost (λ b : Banana, 4 * b) (λ a : Apple, 3 * a)
axiom apples_oranges : cost (λ a : Apple, 9 * a) (λ o : Orange, 6 * o)
axiom oranges_lemons : cost (λ o : Orange, 4 * o) (λ l : Lemon, 2 * l)

theorem bananas_lemons (cost : (Banana → ℝ) → (Apple → ℝ) → (Orange → ℝ) → (Lemon → ℝ) → Prop) :
  cost (λ b : Banana, 24 * b) (λ l : Lemon, 6 * l) :=
sorry

end bananas_lemons_l112_112922


namespace north_southland_population_increase_l112_112127

def hours_per_birth := 5
def deaths_per_day := 2
def hours_per_day := 24
def days_per_year := 365

theorem north_southland_population_increase :
  let births_per_day := hours_per_day / hours_per_birth
  let net_increase_per_day := births_per_day - deaths_per_day
  let annual_increase := net_increase_per_day * days_per_year
  let rounded_annual_increase := Int.round (annual_increase : ℤ) / 100 * 100
  rounded_annual_increase = 1100 :=
by
  sorry

end north_southland_population_increase_l112_112127


namespace PQ_bisects_BD_l112_112947

variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]
variables {A B C D P Q M N : Point}

def convex_quadrilateral (A B C D : Point) : Prop := sorry
def midpoint (P A B : Point) : Prop := 2 • P = A + B
def bisects (line P Q : Point) (diagonal A C : Point) : Prop := 
  ∃ M, midpoint M A C ∧ (line.contains M)
def line_contains_midpoint (P Q : Point) (mid : Point) : Prop := sorry

-- The theorem we want to prove:
theorem PQ_bisects_BD 
  (h1 : convex_quadrilateral A B C D)
  (h2 : midpoint P A B)
  (h3 : midpoint Q C D)
  (h4 : bisects (P, Q) (A, C))
  : bisects (P, Q) (B, D) := 
begin
  sorry
end

end PQ_bisects_BD_l112_112947


namespace only_statement_D_is_correct_l112_112275

-- Define the statements as Lean propositions
def statement_A : Prop := 0 = smallest_integer
def statement_B : Prop := ∀ r : ℚ, (r > 0 ∨ r < 0)
def statement_C (a : ℤ) : Prop := -a < 0
def statement_D : Prop := ∀ r : ℚ, ∃ (a b : ℤ), (b ≠ 0 ∧ r = a / b)

-- The main theorem stating that only statement D is correct
theorem only_statement_D_is_correct : ¬ statement_A ∧ ¬ statement_B ∧ ¬ ∀ a : ℤ, statement_C a ∧ statement_D :=
by
  sorry

end only_statement_D_is_correct_l112_112275


namespace sequence_contradiction_l112_112083

open Classical

variable {α : Type} (a : ℕ → α) [PartialOrder α]

theorem sequence_contradiction {a : ℕ → ℝ} :
  (∀ n, a n < 2) ↔ ¬ ∃ k, a k ≥ 2 := 
by sorry

end sequence_contradiction_l112_112083


namespace find_certain_number_l112_112907

theorem find_certain_number (X : ℝ) (h : 6.2 = 0.036 * X) : X ≈ 172.22 :=
by
  sorry

end find_certain_number_l112_112907


namespace g_inv_g_inv_16_l112_112195

def g (x : ℝ) : ℝ := 3 * x + 7

def g_inv (x : ℝ) : ℝ := (x - 7) / 3

theorem g_inv_g_inv_16 : g_inv (g_inv 16) = -4 / 3 :=
by
  sorry

end g_inv_g_inv_16_l112_112195


namespace num_distinct_lines_4x4_grid_l112_112353

theorem num_distinct_lines_4x4_grid :
  ∃ n : ℕ, 
  n = 32 ∧ 
  let grid_points := 25 in
  let lines := (horizontal_lines grid_points) + (vertical_lines grid_points) + 
               (main_diagonal_lines grid_points) + (anti_diagonal_lines grid_points) + 
               (other_lines grid_points) in
  lines = n :=
begin
  use 32,
  split,
  { refl, },
  { sorry, }
end

end num_distinct_lines_4x4_grid_l112_112353


namespace not_possible_knight_move_100x100_l112_112681

/-- 
  Given two 100x100 boards named Nikolai's board and Stoyan's board, 
  and each cell in these boards is uniquely numbered from 1 to 10000,
  prove that it is not possible for every two adjacent numbers on 
  Nikolai's board to be a knight's move distance in Stoyan's board.
--/
theorem not_possible_knight_move_100x100 
  (nikolai_board stoyan_board : ℕ → ℕ → ℕ)
  (H_nikolai : ∀ i j, 1 ≤ nikolai_board i j ∧ nikolai_board i j ≤ 10000)
  (H_stoyan : ∀ i j, 1 ≤ stoyan_board i j ∧ stoyan_board i j ≤ 10000)
  : ¬ (∀ (i j : ℕ),
      (i < 100 ∧ j < 99 → 
         (abs ((nikolai_board i j) - (nikolai_board i (j+1))) = 1 →
         (is_knight_move (stoyan_board i j) (stoyan_board i (j+1)))))
      ∧ (i < 99 ∧ j < 100 → 
         (abs ((nikolai_board i j) - (nikolai_board (i+1) j)) = 1 →
         (is_knight_move (stoyan_board i j) (stoyan_board (i+1) j))))))
:= by
  sorry

/--
  A helper definition to determine if two cell numbers are a knight's move apart.
--/
def is_knight_move (a b : ℕ) : Prop :=
  let ⟨i1, j1⟩ := coord_of_num a
  let ⟨i2, j2⟩ := coord_of_num b
  ((abs (i1 - i2) = 2 ∧ abs (j1 - j2) = 1) ∨ (abs (i1 - i2) = 1 ∧ abs (j1 - j2) = 2))

end not_possible_knight_move_100x100_l112_112681


namespace fewer_people_correct_l112_112809

def pop_Springfield : ℕ := 482653
def pop_total : ℕ := 845640
def pop_new_city : ℕ := pop_total - pop_Springfield
def fewer_people : ℕ := pop_Springfield - pop_new_city

theorem fewer_people_correct : fewer_people = 119666 :=
by
  unfold fewer_people
  unfold pop_new_city
  unfold pop_total
  unfold pop_Springfield
  sorry

end fewer_people_correct_l112_112809


namespace intersection_l112_112020

def A : Set ℝ := { x | -2 < x ∧ x < 3 }
def B : Set ℝ := { x | x > -1 }

theorem intersection (x : ℝ) : x ∈ (A ∩ B) ↔ -1 < x ∧ x < 3 := by
  sorry

end intersection_l112_112020


namespace min_value_of_square_sum_l112_112850

theorem min_value_of_square_sum (x y : ℝ) (h : (x-1)^2 + y^2 = 16) : ∃ (a : ℝ), a = x^2 + y^2 ∧ a = 9 :=
by 
  sorry

end min_value_of_square_sum_l112_112850


namespace ratio_brother_to_joanna_l112_112977

/-- Definitions for the conditions -/
def joanna_money : ℝ := 8
def sister_money : ℝ := 4 -- since it's half of Joanna's money
def total_money : ℝ := 36

/-- Stating the theorem -/
theorem ratio_brother_to_joanna (x : ℝ) (h : joanna_money + 8*x + sister_money = total_money) :
  x = 3 :=
by 
  -- The ratio of brother's money to Joanna's money is 3:1
  sorry

end ratio_brother_to_joanna_l112_112977


namespace partial_fraction_product_l112_112234

theorem partial_fraction_product : 
  ∃ A B C : ℚ, 
  (A = -21 / 4 ∧ B = 21 / 20 ∧ C = -16 / 5 ∧ 
   (A / (2 - 2) + B / (-2 + 2) + C / (3 - 2) = (23 - 102)/-(4 * 20))) ∧ 
   A * B * C = 1764 / 100 := by
{
  use [-21 / 4, 21 / 20, -16 / 5]
  any_goals {
    split
    all_goals {
      split
      any_goals exact rfl
      split
      any_goals exact rfl
      exact rfl
    }
    exact rfl
  }
  sorry     -- actual proofs will be provided here
}

end partial_fraction_product_l112_112234


namespace difference_of_M_and_m_l112_112803

-- Define the variables and conditions
def total_students : ℕ := 2500
def min_G : ℕ := 1750
def max_G : ℕ := 1875
def min_R : ℕ := 1000
def max_R : ℕ := 1125

-- The statement to prove
theorem difference_of_M_and_m : 
  ∃ G R m M, 
  (G = total_students - R + m) ∧ 
  (min_G ≤ G ∧ G ≤ max_G) ∧
  (min_R ≤ R ∧ R ≤ max_R) ∧
  (m = min_G + min_R - total_students) ∧
  (M = max_G + max_R - total_students) ∧
  (M - m = 250) :=
sorry

end difference_of_M_and_m_l112_112803


namespace triangle_A_value_and_max_perimeter_l112_112581

theorem triangle_A_value_and_max_perimeter (A B C a b c : ℝ) 
  (h1 : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C)
  (h2 : a = 3) :
  (A = 2 * Real.pi / 3) ∧ (a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by
  sorry

end triangle_A_value_and_max_perimeter_l112_112581


namespace f_neither_odd_nor_even_l112_112046

open Real

-- Define the function and its domain
def f (x : ℝ) : ℝ := ln (x + 2) + ln (x - 2)

-- State that we are working with x in the domain where x > 2
axiom domain_condition (x : ℝ) : x > 2

-- The main theorem to prove that f(x) is neither an odd function nor an even function
theorem f_neither_odd_nor_even (x : ℝ) (h : x > 2) : ¬(∀ x, f (-x) = f x) ∧ ¬(∀ x, f (-x) = -f x) :=
by sorry

end f_neither_odd_nor_even_l112_112046


namespace greatest_value_of_a_l112_112831

theorem greatest_value_of_a (a : ℝ) : a^2 - 12 * a + 32 ≤ 0 → a ≤ 8 :=
by
  sorry

end greatest_value_of_a_l112_112831


namespace smallest_value_is_16_l112_112427

noncomputable def smallest_value : ℕ :=
  Inf { |3 * 5^m - 11 * 13^n| | m n : ℕ }

theorem smallest_value_is_16 : smallest_value = 16 :=
sorry

end smallest_value_is_16_l112_112427


namespace value_of_a_2009_l112_112876

/-- Definition to capture the value of a specific element in the sequence. --/
def array_value (m n : ℕ) : ℚ := (m:ℚ) / (n:ℚ)

/-- The sequence of terms, where nth_term represents the nth term in the entire sequence concatenating all arrays. --/
def nth_term (k : ℕ) : ℚ :=
  let rec term_seq (n : ℕ) (sum : ℕ) :=
    if k ≤ sum + n then
      array_value (k - sum) (n - (k - sum) + 1)
    else
      term_seq (n + 1) (sum + n)
  term_seq 1 0

theorem value_of_a_2009 : nth_term 2009 = 56 := sorry

end value_of_a_2009_l112_112876


namespace evaluate_nav_expression_l112_112705
noncomputable def nav (k m : ℕ) := k * (k - m)

theorem evaluate_nav_expression : (nav 5 1) + (nav 4 1) = 32 :=
by
  -- Skipping the proof as instructed
  sorry

end evaluate_nav_expression_l112_112705


namespace todd_savings_l112_112716

-- Define the initial conditions
def original_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon : ℝ := 10
def card_discount : ℝ := 0.10

-- Define the resulting values after applying discounts
def sale_price := original_price * (1 - sale_discount)
def after_coupon := sale_price - coupon
def final_price := after_coupon * (1 - card_discount)

-- Define the total savings
def savings := original_price - final_price

-- The proof statement
theorem todd_savings : savings = 44 := by
  sorry

end todd_savings_l112_112716


namespace incenter_to_vertices_ge_twice_centroid_to_sides_l112_112183

-- Definitions and conditions used in the Lean statement
variables {I G : Point} {A B C : Triangle} 

-- Lean statement
theorem incenter_to_vertices_ge_twice_centroid_to_sides :
  ∑ (dist I [A, B, C]) ≥ 2 * ∑ (dist G [sides A, sides B, sides C]) :=
sorry

end incenter_to_vertices_ge_twice_centroid_to_sides_l112_112183


namespace amount_given_to_john_l112_112744

theorem amount_given_to_john 
    (held_commission : ℕ)
    (advance_agency_fees : ℕ)
    (incentive_amount : ℕ) 
    (held_commission_eq : held_commission = 25000)
    (advance_agency_fees_eq : advance_agency_fees = 8280)
    (incentive_amount_eq : incentive_amount = 1780) : 
    held_commission - advance_agency_fees + incentive_amount = 18500 := 
by
  -- Using the given conditions
  rw [held_commission_eq, advance_agency_fees_eq, incentive_amount_eq]
  -- Calculate the remaining balance
  have h1 : held_commission - advance_agency_fees = 16720, by norm_num
  -- Calculate the total amount to be given to John
  have h2 : h1 + incentive_amount = 18500, by norm_num
  exact h2

end amount_given_to_john_l112_112744


namespace max_area_ABCD_eq_l112_112855

noncomputable def max_area_ABCD (P : ℝ) (d : ℝ) (hP : 0 ≤ P ∧ P < 1) (hd: d = classical.some (classical.some_spec (metric.exists_dist_lt (0 : ℝ^2) P (1/√2).to_real))) : ℝ :=
if h : (0 ≤ d ∧ d < (√2) / 2) then 2 * (√ (1 - d^2))
else if hP : ((√2) / 2 ≤ d ∧ d < 1) then 1 / d
else 0

theorem max_area_ABCD_eq :
  ∀ (P : ℝ) (d : ℝ) (hP : 0 ≤ P ∧ P < 1) (hd : d = classical.some (classical.some_spec (metric.exists_dist_lt (0 : ℝ^2) P (1/√2).to_real))),
  max_area_ABCD P d hP hd = if (0 ≤ d ∧ d < (√2) / 2) then 2 * (√ (1 - d^2))
                             else if ((√2) / 2 ≤ d ∧ d < 1) then 1 / d
                             else 0 :=
begin
  intros,
  dsimp only [max_area_ABCD],
  split_ifs,
  { refl },
  { refl },
  { refl }
end

end max_area_ABCD_eq_l112_112855


namespace find_angle_A_max_perimeter_triangle_l112_112556

-- Part 1: Prove the value of angle A
theorem find_angle_A (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter for BC = 3
theorem max_perimeter_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (A : ℝ) (hA : A = 2 * real.pi / 3) :
  let b := 2 * real.sqrt 3 * sin B,
      c := 2 * real.sqrt 3 * sin C,
      perimeter := 3 + b + c
  in ∀ d : ℝ, -real.pi / 6 < d ∧ d < real.pi / 6 → 
     B + C = real.pi / 3 → 
     perimeter ≤ (3 + 2 * real.sqrt 3) := sorry

end find_angle_A_max_perimeter_triangle_l112_112556


namespace non_equivalent_arrangements_count_l112_112765

-- Definitions
def white_chips := 10
def black_chips := 20
def total_chips := white_chips + black_chips
def swap_condition (i j : ℕ) : Prop := (j = i + 4 ∨ j = i + 26) % total_chips

-- Theorem statement to be proven
theorem non_equivalent_arrangements_count :
  ∃ n : ℕ, n = 11 ∧
  ∀ arrangement1 arrangement2 : list (ℕ × color),
    (valid_arrangement arrangement1 total_chips white_chips black_chips) →
    (valid_arrangement arrangement2 total_chips white_chips black_chips) →
    (equivalent arrangement1 arrangement2 swap_condition ↔ (idempotent arrangement1 arrangement2)) :=
begin
  sorry -- Proof to be done
end

end non_equivalent_arrangements_count_l112_112765


namespace number_of_technicians_l112_112536

theorem number_of_technicians
  (total_workers : ℕ)
  (avg_salary_all : ℝ)
  (avg_salary_techs : ℝ)
  (avg_salary_rest : ℝ)
  (num_techs num_rest : ℕ)
  (h_total_workers : total_workers = 56)
  (h_avg_salary_all : avg_salary_all = 6750)
  (h_avg_salary_techs : avg_salary_techs = 12000)
  (h_avg_salary_rest : avg_salary_rest = 6000)
  (h_eq_workers : num_techs + num_rest = total_workers)
  (h_eq_salaries : (num_techs * avg_salary_techs + num_rest * avg_salary_rest) = total_workers * avg_salary_all) :
  num_techs = 7 := sorry

end number_of_technicians_l112_112536


namespace find_B_share_l112_112791

theorem find_B_share (x : ℕ) (x_pos : 0 < x) (C_share_difference : 5 * x = 4 * x + 1000) (B_share_eq : 3 * x = B) : B = 3000 :=
by
  sorry

end find_B_share_l112_112791


namespace length_BI_of_equilateral_triangle_l112_112992

theorem length_BI_of_equilateral_triangle (ABC : Triangle) (A B C : Point) (I : Point) 
  (h1 : ABC.isEquilateral) (h2 : dist A B = 6) (h3 : I = incenter ABC) : dist B I = sqrt 3 := 
sorry

end length_BI_of_equilateral_triangle_l112_112992


namespace tom_is_15_years_younger_l112_112253

/-- 
Alice is now 30 years old.
Ten years ago, Alice was 4 times as old as Tom was then.
Prove that Tom is 15 years younger than Alice.
-/
theorem tom_is_15_years_younger (A T : ℕ) (h1 : A = 30) (h2 : A - 10 = 4 * (T - 10)) : A - T = 15 :=
by
  sorry

end tom_is_15_years_younger_l112_112253


namespace simplify_tan_expression_l112_112679

theorem simplify_tan_expression :
  (tan 10 * pi / 180 + tan 20 * pi / 180 + tan 30 * pi / 180 + tan 40 * pi / 180) / cos (10 * pi / 180) =
  (1/2 + cos (20 * pi / 180) ^ 2) / (cos (10 * pi / 180) * cos (20 * pi / 180) * cos (30 * pi / 180) * cos (40 * pi / 180)) :=
by
  sorry

end simplify_tan_expression_l112_112679


namespace maria_work_end_time_l112_112821

noncomputable def lunch_break_in_hours : ℝ := 1
noncomputable def total_work_hours : ℝ := 7.5
noncomputable def work_start_time : ℝ := 7 -- 7:00 A.M. in hours
noncomputable def lunch_start_time : ℝ := 13 -- 1:00 P.M. in hours
noncomputable def work_end_time : ℝ := 15.5 -- 3:30 P.M. in hours

theorem maria_work_end_time :
  let work_before_lunch := lunch_start_time - work_start_time,
      work_after_lunch := total_work_hours - work_before_lunch,
      resume_work_time := lunch_start_time + lunch_break_in_hours in
  resume_work_time + work_after_lunch = work_end_time :=
by
  sorry

end maria_work_end_time_l112_112821


namespace part_1_part_2_l112_112585

-- Define the triangle and the given condition
variables {α β γ : ℝ}
axiom triangle_ABC : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π
axiom sin_identity : sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)
axiom BC_length : (sin(γ) / sin(α)) * BC = 3

-- State the main theorem parts separately
theorem part_1 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_sin_identity: sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)) :
  α = 2 * π / 3 :=
sorry

theorem part_2 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_α: α = 2 * π / 3) (h_BC_length: BC = 3) :
  let b := (2 * sqrt(3) * sin(π/6 - β)),
      c := (2 * sqrt(3) * sin(π/6 + β)) in
  (3 + 2 * sqrt(3)) :=
sorry

end part_1_part_2_l112_112585


namespace binary_11101_decimal_l112_112358

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.foldr (λ (x: ℕ) (y: ℕ), x + 2 * y) 0

theorem binary_11101_decimal :
  binary_to_decimal [1, 1, 1, 0, 1] = 29 :=
by
  sorry

end binary_11101_decimal_l112_112358


namespace range_of_k_l112_112473

-- Conditions
def f (x : ℝ) : ℝ
def f' (x : ℝ) : ℝ := e^x * (2 * x + 3) + f(x) * e
def f_zero := (f 0 = 1)

-- Theorem
theorem range_of_k (k : ℝ) : 
  (finset.filter (λ x : ℤ, f x < k) finset.Icc (-10 : ℤ) (10 : ℤ)).card = 2 ↔ 
  k ∈ Ioo (-1 / e^2) 0 ∨ k = 0 :=
  sorry

end range_of_k_l112_112473


namespace bisect_diagonal_BD_l112_112933

-- Define a convex quadrilateral ABCD with midpoints P and Q on sides AB and CD respectively.
variables {A B C D P Q M N : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables [Inhabited P] [Inhabited Q] [Inhabited M] [Inhabited N]

-- Assuming the given statements:
-- ABCD is a convex quadrilateral
-- P is the midpoint of AB
-- Q is the midpoint of CD
-- Line PQ bisects the diagonal AC

-- Prove that line PQ also bisects the diagonal BD
theorem bisect_diagonal_BD 
  (convex_quadrilateral : convex_quadrilateral A B C D)
  (midpoint_P : midpoint P A B)
  (midpoint_Q : midpoint Q C D)
  (PQ_bisects_AC : bisects_line PQ M A C) :
  bisects_line PQ N B D :=
sorry  -- Proof is omitted

end bisect_diagonal_BD_l112_112933


namespace find_a_l112_112448

noncomputable def e : ℝ := Real.exp 1

theorem find_a (a : ℝ) (m : ℝ) (h1 : 1 < a)
  (h2 : 2.71828 < e ∧ e < 2.71829)
  (h3 : log a m = m)
  (h4 : a ^ m = m)
  : a = e ^ (1 / e) :=
sorry

end find_a_l112_112448


namespace problem_1_and_2_l112_112596

-- Definition of the problem in Lean 4
theorem problem_1_and_2 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin (A)^2 - sin (B)^2 - sin (C)^2 = sin (B) * sin (C))
  (h2 : BC = 3)
  (h3 : triangle ABC)
  (h4 : a = side_length (opposite A))
  (h5 : b = side_length (opposite B))
  (h6 : c = side_length (opposite C)) :
  A = 2 * real.pi / 3 ∧
  (a + b + c ≤ 3 + 2 * real.sqrt 3) :=
sorry

end problem_1_and_2_l112_112596


namespace raduzhny_population_l112_112111

-- Definitions based on the conditions
def total_villages : ℕ := 10
def diff_constraint : ℕ := 100
def znoinie_population : ℕ := 1000
def avg_population := znoinie_population - 90

-- Main theorem to prove
theorem raduzhny_population :
  ∃ raduzhny_population : ℕ, raduzhny_population = 900 ∧ 
  (∀ i j : ℕ, i < total_villages → j < total_villages → |(if i = 0 then znoinie_population else raduzhny_population) - 
  (if j = 0 then znoinie_population else raduzhny_population)| ≤ diff_constraint) ∧
  znoinie_population = avg_population + 90 := 
sorry

end raduzhny_population_l112_112111


namespace find_A_max_perimeter_of_triangle_l112_112566

-- Definition of the given problem conditions
def triangle_condition (A B C : ℝ) : Prop :=
  sin(A)^2 - sin(B)^2 - sin(C)^2 = sin(B) * sin(C)

-- (1) Proving the value of A given the condition
theorem find_A (A B C : ℝ) (h : triangle_condition A B C) : A = 2 * π / 3 :=
by sorry

-- (2) Proving the maximum perimeter given BC = 3 and A = 2π/3
theorem max_perimeter_of_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (hA : 2 * π / 3 = 2 * π / 3) : 
  ∃ (P : ℝ), P = 3 + 2 * sqrt 3 :=
by sorry

end find_A_max_perimeter_of_triangle_l112_112566


namespace no_positive_integer_solutions_l112_112510

theorem no_positive_integer_solutions (x : ℕ) : ¬(15 < 3 - 2 * x) := by
  sorry

end no_positive_integer_solutions_l112_112510


namespace concave_number_probability_l112_112792

-- Define the properties for a concave number
def is_concave (a b c : ℕ) : Prop :=
  a > b ∧ c > b

-- Define the set of possible digits
def digit_set : Finset ℕ := {4, 5, 6, 7, 8}

-- Define the main theorem
theorem concave_number_probability : 
  (∑ a b c ∈ digit_set, if is_concave a b c ∧ (a ≠ b ∧ a ≠ c ∧ b ≠ c) then 1 else 0) 
  = (20 : ℝ) / 60 :=
sorry

end concave_number_probability_l112_112792


namespace min_value_of_quadratic_l112_112737

theorem min_value_of_quadratic (x : ℝ) : ∃ m : ℝ, (∀ x, x^2 + 10 * x ≥ m) ∧ m = -25 := by
  sorry

end min_value_of_quadratic_l112_112737


namespace length_of_MN_l112_112131

theorem length_of_MN
  (a b c : ℝ)
  (h : a > 0)
  (AC_eq_b : ∥AC∥ = b)
  (BC_eq_a : ∥BC∥ = a)
  (AB_eq_c : ∥AB∥ = c)
  (DE_parallel_AB : DE || AB)
  (DE_midline : ∥DE∥ = c/2)
  (cosine_law : cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  ∥MN∥ = c * (a^2 + b^2 - c^2) / (4 * a * b) :=
sorry

end length_of_MN_l112_112131


namespace ms_cole_total_students_l112_112172

def students_6th : ℕ := 40
def students_4th : ℕ := 4 * students_6th
def students_7th : ℕ := 2 * students_4th

def total_students : ℕ := students_6th + students_4th + students_7th

theorem ms_cole_total_students :
  total_students = 520 :=
by
  sorry

end ms_cole_total_students_l112_112172


namespace period_of_sin_5x_l112_112263

theorem period_of_sin_5x : ∀ (x : ℝ), (∀ b, period (λ x, sin (b * x)) = 2 * π / |b|) → (period (λ x, sin (5 * x)) = 2 * π / 5) :=
by
  intro x h_period_formula
  have b := 5
  exact h_period_formula b


end period_of_sin_5x_l112_112263


namespace yellow_more_than_green_l112_112330

-- Given conditions
def G : ℕ := 90               -- Number of green buttons
def B : ℕ := 85               -- Number of blue buttons
def T : ℕ := 275              -- Total number of buttons
def Y : ℕ := 100              -- Number of yellow buttons (derived from conditions)

-- Mathematically equivalent proof problem
theorem yellow_more_than_green : (90 + 100 + 85 = 275) → (100 - 90 = 10) :=
by sorry

end yellow_more_than_green_l112_112330


namespace jenny_speed_proof_l112_112135

-- Define the constants and conditions based on the problem
def total_distance (S : ℝ) : Prop :=
  S > 0

def jack_speed_first_half (v1 : ℝ) : Prop :=
  v1 = 4

def jack_speed_second_half (v2 : ℝ) : Prop :=
  v2 = 2

def jack_speed_descending (v3 : ℝ) : Prop :=
  v3 = 3

def jack_meeting_point (S : ℝ) (t1 t2 t3 : ℝ) : Prop :=
  t1 = S / 8 ∧ t2 = S / 4 ∧ t3 = S / 6 ∧ t1 + t2 + t3 = 13 * S / 24

-- Define Jenny's average speed based on Jack's time and distance
def jenny_avg_speed (S t : ℝ) (v_jenny : ℝ) : Prop :=
  v_jenny = (S / 2) / t

theorem jenny_speed_proof
  (S t1 t2 t3 t v_jenny : ℝ)
  (pos_S : total_distance S)
  (js1 : jack_speed_first_half 4)
  (js2 : jack_speed_second_half 2)
  (js3 : jack_speed_descending 3)
  (jmp : jack_meeting_point S t1 t2 t3)
  : jenny_avg_speed S (13 * S / 24) v_jenny → v_jenny = 12 / 13 :=
by
  sorry

end jenny_speed_proof_l112_112135


namespace part1_part2_l112_112621

theorem part1 (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : A = 2 * π / 3 :=
sorry

theorem part2 (b c : ℝ) (h1 : sin (2 * π / 3) ^ 2 - sin b ^ 2 - sin c ^ 2 = sin b * sin c)
  (h2 : b + c = 2 * sqrt 3) : 3 * 2 + b + c = 3 + 2 * sqrt 3 :=
sorry

end part1_part2_l112_112621


namespace product_sign_l112_112817

theorem product_sign :
  (1:ℝ) < π / 2 ∧
  π / 2 < 2 ∧ 2 < π ∧
  π / 2 < 3 ∧ 3 < π ∧
  π < 4 ∧ 4 < 3 * π / 2 →
  cos 1 * cos 2 * cos 3 * cos 4 < 0 :=
by sorry

end product_sign_l112_112817


namespace option_C_correct_l112_112273

theorem option_C_correct (x : ℝ) : x^3 * x^2 = x^5 := sorry

end option_C_correct_l112_112273


namespace last_box_one_ball_probability_l112_112102

/-- The probability that the last box will contain exactly one of 100 randomly distributed balls
is approximately 0.370. -/
theorem last_box_one_ball_probability :
  let n : ℕ := 100 in
  let p : ℚ := 1 / 100 in
  let probability : ℚ := (n.choose 1) * (p) * ((1 - p) ^ (n - 1)) in
  probability ≈ 0.370 :=
by
  let n : ℕ := 100 
  let p : ℚ := 1 / 100
  let probability : ℚ := (n.choose 1) * (p) * ((1 - p) ^ (n - 1))
  sorry

end last_box_one_ball_probability_l112_112102


namespace find_A_max_perimeter_of_triangle_l112_112568

-- Definition of the given problem conditions
def triangle_condition (A B C : ℝ) : Prop :=
  sin(A)^2 - sin(B)^2 - sin(C)^2 = sin(B) * sin(C)

-- (1) Proving the value of A given the condition
theorem find_A (A B C : ℝ) (h : triangle_condition A B C) : A = 2 * π / 3 :=
by sorry

-- (2) Proving the maximum perimeter given BC = 3 and A = 2π/3
theorem max_perimeter_of_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (hA : 2 * π / 3 = 2 * π / 3) : 
  ∃ (P : ℝ), P = 3 + 2 * sqrt 3 :=
by sorry

end find_A_max_perimeter_of_triangle_l112_112568


namespace count_positive_integers_not_divisible_by_three_l112_112432

theorem count_positive_integers_not_divisible_by_three :
  (finset.range 501).filter (λ n : ℕ, n > 0 ∧ (⟨497 / n, by norm_num⟩ + ⟨498 / n, by norm_num⟩ + ⟨499 / n, by norm_num⟩) % 3 ≠ 0).card = 15 :=
sorry

end count_positive_integers_not_divisible_by_three_l112_112432


namespace number_of_integers_divisible_by_15_l112_112901

def digit_gt_4 (n : ℕ) : Prop :=
  n > 4

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 

def divisible_by (n k : ℕ) : Prop :=
  n % k = 0

theorem number_of_integers_divisible_by_15 
  (S : ℕ → ℕ → ℕ → ℕ) :
  (∀ h t u, 
    (S h t u = (100 * h + 10 * t + u)) → 
    (is_three_digit (S h t u)) → 
    (divisible_by (S h t u) 15) → 
    (digit_gt_4 h) ∧ (digit_gt_4 t) ∧ (digit_gt_4 u)) →
  (Π (h t u : ℕ), h ∈ {5, 6, 7, 8, 9} → t ∈ {5, 6, 7, 8, 9} → u = 5) →
  7 :=
sorry

end number_of_integers_divisible_by_15_l112_112901


namespace joe_initial_paint_l112_112632
-- Use necessary imports

-- Define the hypothesis
def initial_paint_gallons (g : ℝ) :=
  (1 / 4) * g + (1 / 7) * (3 / 4) * g = 128.57

-- Define the theorem
theorem joe_initial_paint (P : ℝ) (h : initial_paint_gallons P) : P = 360 :=
  sorry

end joe_initial_paint_l112_112632


namespace inverse_function_shift_l112_112223

-- Conditions
variable {f : ℝ → ℝ} {f_inv : ℝ → ℝ}
variable (hf : ∀ x : ℝ, f_inv (f x) = x ∧ f (f_inv x) = x)
variable (point_B : f 3 = -1)

-- Proof statement
theorem inverse_function_shift :
  f_inv (-3 + 2) = 3 :=
by
  -- Proof goes here
  sorry

end inverse_function_shift_l112_112223


namespace problem1_problem2_case1_problem2_case2_problem3_l112_112758

-- Define the fare calculation function
def fare (distance : ℝ) (time : ℝ) : ℝ :=
  let time_fee := 0.45 * time
  let mileage_fee := 1.8 * distance
  let long_distance_fee := if distance > 10 then 0.4 * (distance - 10) else 0
  mileage_fee + time_fee + long_distance_fee

-- Proof statements

-- For Problem (1)
theorem problem1 : fare 5 10 = 13.5 :=
by {
  unfold fare,
  simp,
  norm_num,
}

-- For Problem (2)
theorem problem2_case1 (a b : ℝ) (h : a ≤ 10) : fare a b = 1.8 * a + 0.45 * b :=
by {
  unfold fare,
  simp [if_pos h],
}

theorem problem2_case2 (a b : ℝ) (h : ¬ a ≤ 10) : fare a b = 2.2 * a + 0.45 * b - 4 :=
by {
  unfold fare,
  simp [if_neg h],
  linarith,
}

-- For Problem (3)
theorem problem3 (a : ℝ) : fare 9.5 a = fare 14.5 (a - 24) :=
by {
  unfold fare,
  simp,
  norm_num,
  linarith,
}


end problem1_problem2_case1_problem2_case2_problem3_l112_112758


namespace bisect_diagonal_BD_l112_112931

-- Define a convex quadrilateral ABCD with midpoints P and Q on sides AB and CD respectively.
variables {A B C D P Q M N : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables [Inhabited P] [Inhabited Q] [Inhabited M] [Inhabited N]

-- Assuming the given statements:
-- ABCD is a convex quadrilateral
-- P is the midpoint of AB
-- Q is the midpoint of CD
-- Line PQ bisects the diagonal AC

-- Prove that line PQ also bisects the diagonal BD
theorem bisect_diagonal_BD 
  (convex_quadrilateral : convex_quadrilateral A B C D)
  (midpoint_P : midpoint P A B)
  (midpoint_Q : midpoint Q C D)
  (PQ_bisects_AC : bisects_line PQ M A C) :
  bisects_line PQ N B D :=
sorry  -- Proof is omitted

end bisect_diagonal_BD_l112_112931


namespace no_solution_for_b_a_divides_a_b_minus_1_l112_112372

theorem no_solution_for_b_a_divides_a_b_minus_1 :
  ¬ (∃ a b : ℕ, 1 ≤ a ∧ 1 ≤ b ∧ b^a ∣ a^b - 1) :=
by
  sorry

end no_solution_for_b_a_divides_a_b_minus_1_l112_112372


namespace find_quadratic_polynomial_l112_112387

noncomputable def quadratic_polynomial : Polynomial ℝ :=
  3 * (X - C (2 + 2*I)) * (X - C (2 - 2*I))

theorem find_quadratic_polynomial :
  quadratic_polynomial = 3 * X^2 - 12 * X + 24 :=
by
  sorry

end find_quadratic_polynomial_l112_112387


namespace find_rectangle_angles_l112_112685

-- Let's define the conditions and state the problem
def isosceles_triangle_base_leg_ratio (a b : ℝ) : Prop :=
  b = 2 * a

def rectangle_area_perimeter (a : ℝ) (h : ℝ) : Prop :=
  let A := (a * h / 2) in
  let P := (a + 2 * (2 * a)) in
  let r_height := a * 2 in
  (A = r_height * h) ∧ (P = 2 * (a + r_height))

def rectangle_side_ratio (x y : ℝ) : Prop :=
  x = 2 * y

def angles_of_rectangle (angle1 angle2 : ℝ) : Prop :=
  (angle1 = 135.80416) ∧ (angle2 = 44.19584)

theorem find_rectangle_angles
  (a h : ℝ) (x y : ℝ)
  (angle1 angle2 : ℝ)
  (h1 : isosceles_triangle_base_leg_ratio a h)
  (h2 : rectangle_side_ratio x y)
  (h3 : rectangle_area_perimeter a h) :
  angles_of_rectangle angle1 angle2 :=
sorry

end find_rectangle_angles_l112_112685


namespace triangle_A_value_and_max_perimeter_l112_112577

theorem triangle_A_value_and_max_perimeter (A B C a b c : ℝ) 
  (h1 : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C)
  (h2 : a = 3) :
  (A = 2 * Real.pi / 3) ∧ (a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by
  sorry

end triangle_A_value_and_max_perimeter_l112_112577


namespace coefficient_of_determination_indicates_better_fit_l112_112957

theorem coefficient_of_determination_indicates_better_fit (R_squared : ℝ) (h1 : 0 ≤ R_squared) (h2 : R_squared ≤ 1) :
  R_squared = 1 → better_fitting_effect_of_regression_model :=
by
  sorry

end coefficient_of_determination_indicates_better_fit_l112_112957


namespace candies_per_person_l112_112509

-- Define Henley and her brothers sharing candies
def total_candies : ℕ := 300
def sour_percentage : ℝ := 0.4
def total_people : ℕ := 3

-- Lean statement to prove that each person gets 60 candies
theorem candies_per_person : 
  let sour_candies := (sour_percentage * total_candies).to_nat in
  let good_candies := total_candies - sour_candies in
  good_candies / total_people = 60 := 
by
  sorry

end candies_per_person_l112_112509


namespace find_smallest_x_l112_112421

noncomputable def smallest_x : ℝ :=
  min { x : ℝ | 0 < x ∧ (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * x = 8 }

theorem find_smallest_x :
  smallest_x = 89 / 9 :=
by
  sorry

end find_smallest_x_l112_112421


namespace sum_of_favorite_numbers_l112_112170

def Glory_favorite_number : ℕ := 450
def Misty_favorite_number : ℕ := Glory_favorite_number / 3

theorem sum_of_favorite_numbers : Misty_favorite_number + Glory_favorite_number = 600 :=
by
  sorry

end sum_of_favorite_numbers_l112_112170


namespace problem_1_and_2_l112_112598

-- Definition of the problem in Lean 4
theorem problem_1_and_2 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin (A)^2 - sin (B)^2 - sin (C)^2 = sin (B) * sin (C))
  (h2 : BC = 3)
  (h3 : triangle ABC)
  (h4 : a = side_length (opposite A))
  (h5 : b = side_length (opposite B))
  (h6 : c = side_length (opposite C)) :
  A = 2 * real.pi / 3 ∧
  (a + b + c ≤ 3 + 2 * real.sqrt 3) :=
sorry

end problem_1_and_2_l112_112598


namespace darren_fergie_same_debt_l112_112360

variable (darren_borrow darren_prev debt_rate_darren fergie_borrow debt_rate_fergie : ℝ)

-- Conditions
axiom darren_borrow : darren_borrow = 200
axiom darren_prev : darren_prev = 50
axiom debt_rate_darren : debt_rate_darren = 0.12
axiom fergie_borrow : fergie_borrow = 300
axiom debt_rate_fergie : debt_rate_fergie = 0.07

-- Lean statement
theorem darren_fergie_same_debt (t : ℝ) :
  250 + 24 * t = 300 + 21 * t → t = 50 / 3 :=
by
  intro h
  sorry

end darren_fergie_same_debt_l112_112360


namespace probability_A_finishes_with_more_points_than_B_l112_112429

-- Conditions
axiom five_teams : nat = 5
axiom games_per_team : nat = five_teams - 1
axiom win_probability : ∀ (team1 team2 : nat), 0.5
axiom independent_outcomes : ∀ (team1 team2 : nat), probabilistic independence of outcomes
axiom first_game_result : winning team(A) and losing team(B)

-- Question to be proved
theorem probability_A_finishes_with_more_points_than_B :
  let q := (1 / 2^3 * 2^3) * (∑ k in finset.range(4), nat.choose(3, k)^2) in
  let p := 0.5 * (1 - q) in
  q + p = 133 / 256 ∧ (133 + 256 = 389) :=
sorry

end probability_A_finishes_with_more_points_than_B_l112_112429


namespace probability_heads_before_tails_l112_112999

noncomputable def solve_prob : ℚ := 
  let p := λ n : ℕ, if n = 4 then 1 else if n = 3 then (1 / 2 + 1 / 2 * t 1) else
                        if n = 2 then (1 / 2 * (1/2 + 1/2 * t 1) + 1 / 2 * t 1) else
                        if n = 1 then (1 / 2 * (1 / 4 + 3 / 4 * t 1) + 1 / 2 * t 1) else
                                   (1 / 2 * (1 / 8 + 7 / 8 * t 1) + 1 / 2 * t 1)
  and t := λ n : ℕ, if n = 2 then 0 else
                        if n = 1 then 1 / 2 * (t 1 + 1 / 2) else
                                   1 / 2 * (t 1 + t n)
  in p 0

theorem probability_heads_before_tails : solve_prob = 15/23 :=
sorry

end probability_heads_before_tails_l112_112999


namespace functional_equation_num_values_times_sum_l112_112995

noncomputable def g : ℝ → ℝ := sorry

-- The given functional equation condition
theorem functional_equation (x y z : ℝ) : g (x^2 + y * g z) = x * g x + 2 * z * g y := sorry

-- The theorem to prove
theorem num_values_times_sum (m t : ℝ) (h_m : m = 2) (h_t : t = 8) : m * t = 16 :=
begin
  rw [h_m, h_t],
  exact mul_comm 2 8,
end

end functional_equation_num_values_times_sum_l112_112995


namespace complement_union_S_T_l112_112159

open Set

variable (S : Set ℝ) (T : Set ℝ)

-- Defining the sets S and T
def S := {x : ℝ | x > -2}
def T := {x : ℝ | x^2 + 3*x - 4 ≤ 0}

-- The proof statement
theorem complement_union_S_T :
  (compl S ∪ T) = {x : ℝ | x ≤ 1} :=
by
  -- Formal proof skipped
  sorry

end complement_union_S_T_l112_112159


namespace max_retained_pits_l112_112703

theorem max_retained_pits 
  (length_road : ℕ) 
  (initial_interval : ℕ) 
  (revised_interval : ℕ) 
  (length_road_eq : length_road = 120) 
  (initial_interval_eq : initial_interval = 3) 
  (revised_interval_eq : revised_interval = 5) :
  ∃ (max_pits : ℕ), max_pits = 18 :=
by {
  use 18,
  sorry
}

end max_retained_pits_l112_112703


namespace find_A_max_perimeter_of_triangle_l112_112567

-- Definition of the given problem conditions
def triangle_condition (A B C : ℝ) : Prop :=
  sin(A)^2 - sin(B)^2 - sin(C)^2 = sin(B) * sin(C)

-- (1) Proving the value of A given the condition
theorem find_A (A B C : ℝ) (h : triangle_condition A B C) : A = 2 * π / 3 :=
by sorry

-- (2) Proving the maximum perimeter given BC = 3 and A = 2π/3
theorem max_perimeter_of_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (hA : 2 * π / 3 = 2 * π / 3) : 
  ∃ (P : ℝ), P = 3 + 2 * sqrt 3 :=
by sorry

end find_A_max_perimeter_of_triangle_l112_112567


namespace cherries_left_l112_112301

def initial_cherries : ℕ := 77
def cherries_used : ℕ := 60

theorem cherries_left : initial_cherries - cherries_used = 17 := by
  sorry

end cherries_left_l112_112301


namespace find_smallest_x_l112_112425

noncomputable def smallest_x : ℝ :=
  min { x : ℝ | 0 < x ∧ (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * x = 8 }

theorem find_smallest_x :
  smallest_x = 89 / 9 :=
by
  sorry

end find_smallest_x_l112_112425


namespace solve_for_x_l112_112193

theorem solve_for_x (x : ℝ) (h : 3^(3 * x) = 27) : x = 1 :=
by {
  -- The proof would be placed here
  sorry
}

end solve_for_x_l112_112193


namespace find_smallest_x_l112_112410

def smallest_x_satisfying_condition : Prop :=
  ∃ x : ℝ, x > 0 ∧ (⌊x^2⌋ - x * ⌊x⌋ = 8) ∧ x = 89 / 9

theorem find_smallest_x : smallest_x_satisfying_condition :=
begin
  -- proof goes here (not required for this task)
  sorry
end

end find_smallest_x_l112_112410


namespace coefficient_x5_binom_expansion_l112_112362

noncomputable def binom (n k : ℕ) : ℕ := nat.choose n k

theorem coefficient_x5_binom_expansion :
  let T_r (r : ℕ) := binom 12 r * (2 + sqrt x)^(12 - r) * (-2017 / x^2017)^r,
      coeff_xk (T : (ℕ → ℕ) → ℕ) (k : ℕ) : ℕ := sorry
  in coeff_xk (λ r : ℕ, T_r r) 5 = 48 := by
    sorry

end coefficient_x5_binom_expansion_l112_112362


namespace part_1_part_2_l112_112591

-- Define the triangle and the given condition
variables {α β γ : ℝ}
axiom triangle_ABC : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π
axiom sin_identity : sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)
axiom BC_length : (sin(γ) / sin(α)) * BC = 3

-- State the main theorem parts separately
theorem part_1 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_sin_identity: sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)) :
  α = 2 * π / 3 :=
sorry

theorem part_2 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_α: α = 2 * π / 3) (h_BC_length: BC = 3) :
  let b := (2 * sqrt(3) * sin(π/6 - β)),
      c := (2 * sqrt(3) * sin(π/6 + β)) in
  (3 + 2 * sqrt(3)) :=
sorry

end part_1_part_2_l112_112591


namespace ellipse_problem_l112_112025

noncomputable def ellipseStandardEquation (a b : ℝ) (h_eqn : a^2 = 2 ∧ b^2 = 1) : Prop :=
  ∀ x y : ℝ, (x = -1 ∧ y = sqrt 2 / 2) → (x^2 / 2 + y^2 = 1)

noncomputable def triangleAreaRange (k : ℝ) (h_k : 1/2 ≤ k^2 ∧ k^2 ≤ 1)
(λ : ℝ) (h_λ : 2/3 ≤ λ ∧ λ ≤ 3/4) : Prop :=
  ∃ S : ℝ, sqrt 6 / 4 ≤ S ∧ S ≤ 2 / 3

theorem ellipse_problem (a b : ℝ) (k : ℝ) (λ : ℝ)
  (h_a_gt_b_gt_0 : a > b > 0)
  (h_eqn_eccentricity : a^2 = 2 ∧ b^2 = 1 ∧ c = sqrt 2 / 2 ∧ c^2 = a^2 - b^2)
  (h_point_on_ellipse : ∀ x y : ℝ, (x = -1 ∧ y = sqrt 2 / 2) → (x^2 / 2 + y^2 = 1))
  (h_line_tangent_circle : ∃ м : ℝ, m^2 = k^2 + 1)
  (h_line_intersects_ellipse : ∀ x_1 y_1 x_2 y_2 : ℝ, l x_1 + l y_1 = l x_2 + l y_2)
  (h_lambda_range : 2/3 ≤ λ ∧ λ ≤ 3/4) :
  ellipseStandardEquation a b (h_eqn_eccentricity.1, h_eqn_eccentricity.2.1) ∧
  triangleAreaRange k (h_lambda_range) :=
sorry

end ellipse_problem_l112_112025


namespace no_cans_collected_l112_112219

theorem no_cans_collected (total_students : ℕ) (half_class_collected : ℕ) (remaining_collected : ℕ) (total_cans : ℕ) :
  total_students = 30 → 
  half_class_collected = 15 → 
  remaining_collected = 13 →
  total_cans = 232 →
  half_class_collected * 12 + remaining_collected * 4 = total_cans → 
  total_students - (half_class_collected + remaining_collected) = 2 :=
by
  intros h_total_students h_half_collected h_remaining_collected h_total_cans h_cans_sum
  have h1 : 15 * 12 = 180 := by sorry
  have h2 : 13 * 4 = 52 := by sorry
  have h3 : 180 + 52 = 232 := by sorry
  exact h_total_students ▸ h_half_collected ▸ h_remaining_collected ▸ h_total_cans ▸ h_cans_sum ▸
    show 30 - (15 + 13) = 2, by 
      calc 30 - (15 + 13)
            = 30 - 28 : by simp
        ... = 2        : by simp

end no_cans_collected_l112_112219


namespace sufficient_not_necessary_l112_112291

theorem sufficient_not_necessary (p q : Prop) (h : p ∧ q) : (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) :=
by
  sorry

end sufficient_not_necessary_l112_112291


namespace petya_coloring_l112_112281

theorem petya_coloring (k : ℕ) : k = 1 :=
  sorry

end petya_coloring_l112_112281


namespace smallest_positive_x_l112_112416

theorem smallest_positive_x (x : ℝ) (h : ⌊x^2⌋ - x * ⌊x⌋ = 8) : x = 89 / 9 :=
sorry

end smallest_positive_x_l112_112416


namespace find_smallest_x_l112_112412

def smallest_x_satisfying_condition : Prop :=
  ∃ x : ℝ, x > 0 ∧ (⌊x^2⌋ - x * ⌊x⌋ = 8) ∧ x = 89 / 9

theorem find_smallest_x : smallest_x_satisfying_condition :=
begin
  -- proof goes here (not required for this task)
  sorry
end

end find_smallest_x_l112_112412


namespace find_term_position_in_sequence_l112_112501

theorem find_term_position_in_sequence :
  let a₁ := 2
  let d := 3
  let a_n n := a₁ + (n - 1) * d
  (∃ n, a_n n = 20) → ∃ n = 7 :
sorry

end find_term_position_in_sequence_l112_112501


namespace committee_size_l112_112806

theorem committee_size (n : ℕ)
  (h : ((n - 2 : ℕ) : ℚ) / ((n - 1) * (n - 2) / 2 : ℚ) = 0.4) :
  n = 6 :=
by
  sorry

end committee_size_l112_112806


namespace mutually_exclusive_but_not_complementary_l112_112844

open ProbabilityTheory

-- Define the event of drawing different colored balls
def Bag := {red := 3, black := 3}

def draw_two_balls (bag : Bag) : Set (Set String) :=
  {s | s.card = 2 ∧ ∀ b ∈ s, b = "red" ∨ b = "black"}

-- Event: Exactly one black ball in the draw
def exactly_one_black (event : Set String) : Prop :=
  event.count "black" = 1

-- Event: Exactly two red balls in the draw
def exactly_two_red (event : Set String) : Prop :=
  event.count "red" = 2

theorem mutually_exclusive_but_not_complementary :
  (∀ e ∈ (draw_two_balls Bag), exactly_one_black e → ¬ exactly_two_red e) ∧
  (∃ e ∈ (draw_two_balls Bag), ¬ exactly_one_black e ∧ ¬ exactly_two_red e) :=
by sorry

end mutually_exclusive_but_not_complementary_l112_112844


namespace rectangle_length_proof_l112_112961

noncomputable def rectangle_length (width : ℝ) 
(r : ℝ) (R : ℝ) (distance_centers : ℝ) : ℝ :=
let diag_distance := Real.sqrt (distance_centers ^ 2 - r ^ 2) in
R + (diag_distance + r)

theorem rectangle_length_proof : 
∀ (width r R : ℝ), 
  (width = 4) → 
  (2 * R = width) → 
  (2 * r = width) → 
  (R + r = 3) → 
  rectangle_length width r R 3 = 3 + Real.sqrt 8 :=
by { intros width r R hw hr hs hd,
  rw [hw, hr, hs, hd],
  simp only [Real.sqrt_mul_self, Real.add_left_comm, Real.add_assoc, Real.sqrt_eq_rpow, Real.add_sub_cancel, Real.rpow_one, mul_one],
  sorry }

end rectangle_length_proof_l112_112961


namespace minimum_value_f_on_interval_l112_112220

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - (3 / 2) * x^2 + 2 * x + 1

theorem minimum_value_f_on_interval (h : ∀ x ∈ (Ioo 1 3 : set ℝ), differentiable_at ℝ f x) :
  ∃ x ∈ (Ioo 1 3 : set ℝ), is_local_minimum f (Ioo 1 3) x ∧ f x = 5 / 3 :=
begin
  sorry
end

end minimum_value_f_on_interval_l112_112220


namespace tangent_condition_l112_112521

theorem tangent_condition (m : ℝ) :
  (∀ m, let dist_squared := (m + 1)^2 in dist_squared > 4 → m < -3 ∨ m > 1) :=
sorry

end tangent_condition_l112_112521


namespace petya_coloring_l112_112282

theorem petya_coloring (k : ℕ) : k = 1 :=
  sorry

end petya_coloring_l112_112282


namespace evaluate_at_one_l112_112189

def f (a : ℝ) : ℝ := ((3 / (a + 1)) - a + 1) / ((a^2 - 4 * a + 4) / (a + 1))

theorem evaluate_at_one : f 1 = 3 :=
by
  unfold f
  -- Definitions are expanded directly for simplicity
  let a := 1
  have h1 : 3 / (a + 1) - a + 1 = 3 / (1 + 1) - 1 + 1 := rfl
  simp at h1
  have h2 : (a^2 - 4 * a + 4) / (a + 1) = (1^2 - 4 * 1 + 4) / (1 + 1) := rfl
  simp at h2
  rw [h1, h2]
  norm_num
  sorry

end evaluate_at_one_l112_112189


namespace candies_per_person_l112_112508

-- Define Henley and her brothers sharing candies
def total_candies : ℕ := 300
def sour_percentage : ℝ := 0.4
def total_people : ℕ := 3

-- Lean statement to prove that each person gets 60 candies
theorem candies_per_person : 
  let sour_candies := (sour_percentage * total_candies).to_nat in
  let good_candies := total_candies - sour_candies in
  good_candies / total_people = 60 := 
by
  sorry

end candies_per_person_l112_112508


namespace units_digit_34_pow_30_l112_112264

theorem units_digit_34_pow_30 :
  (34 ^ 30) % 10 = 6 :=
by
  sorry

end units_digit_34_pow_30_l112_112264


namespace value_of_m_l112_112076

theorem value_of_m (x m : ℝ) (h_positive_root : x > 0) (h_eq : x / (x - 1) - m / (1 - x) = 2) : m = -1 := by
  sorry

end value_of_m_l112_112076


namespace sum_of_possible_values_of_G_F_l112_112910

theorem sum_of_possible_values_of_G_F (G F : ℕ) (hG : 0 ≤ G ∧ G ≤ 9) (hF : 0 ≤ F ∧ F ≤ 9)
  (hdiv : (G + 2 + 4 + 3 + F + 1 + 6) % 9 = 0) : G + F = 2 ∨ G + F = 11 → 2 + 11 = 13 :=
by { sorry }

end sum_of_possible_values_of_G_F_l112_112910


namespace trigonometric_identity_l112_112050

noncomputable def inclination_angle (α : ℝ) : Prop := 
  ∃ x y : ℝ, 3 * x - y + 1 = 0 ∧ tan α = 3

theorem trigonometric_identity (α : ℝ) (h : inclination_angle α) : 
  (1 / 2) * sin (2 * α) + cos α * cos α = 2 / 5 :=
by 
  sorry

end trigonometric_identity_l112_112050


namespace part1_part2_l112_112622

theorem part1 (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : A = 2 * π / 3 :=
sorry

theorem part2 (b c : ℝ) (h1 : sin (2 * π / 3) ^ 2 - sin b ^ 2 - sin c ^ 2 = sin b * sin c)
  (h2 : b + c = 2 * sqrt 3) : 3 * 2 + b + c = 3 + 2 * sqrt 3 :=
sorry

end part1_part2_l112_112622


namespace tangent_point_on_diagonal_AC_l112_112133

variable {A B C D O₁ O₂ : Type}
variables (ABCD : parallelogram A B C D)

def circle_tangent_to_AB_AD (O₁ : Type) : Prop :=
  tangent O₁ AB ∧ tangent O₁ AD

def circle_tangent_to_CD_CB (O₂ : Type) : Prop :=
  tangent O₂ CD ∧ tangent O₂ CB

theorem tangent_point_on_diagonal_AC
  (circle1 : circle_tangent_to_AB_AD O₁)
  (circle2 : circle_tangent_to_CD_CB O₂)
  (tangent_circles : tangent O₁ O₂) :
  ∃ K : Type, K ∈ diagonal A C :=
sorry

end tangent_point_on_diagonal_AC_l112_112133


namespace smallest_positive_real_number_l112_112400

noncomputable def smallest_x : ℝ :=
  let x := 89 / 9 in x

theorem smallest_positive_real_number :
  ∀ x : ℝ, (x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8) → x ≥ smallest_x  :=
by
  sorry

end smallest_positive_real_number_l112_112400


namespace min_value_of_fx_l112_112845

theorem min_value_of_fx (x : ℝ) (a : ℝ) (hx : 0 ≤ x ∧ x ≤ π) (ha : -1/2 < a ∧ a < 0) : 
  ∃ y, (y = f x) ∧ (f x = -2 * a - 1) := 
  by 
  let f (x : ℝ) : ℝ := cos x ^ 2 - 2 * a * sin x - 1
  sorry

end min_value_of_fx_l112_112845


namespace smallest_positive_real_number_l112_112398

noncomputable def smallest_x : ℝ :=
  let x := 89 / 9 in x

theorem smallest_positive_real_number :
  ∀ x : ℝ, (x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8) → x ≥ smallest_x  :=
by
  sorry

end smallest_positive_real_number_l112_112398


namespace find_n_l112_112514

-- Define the variables d, Q, r, m, and n
variables (d Q r m n : ℝ)

-- Define the conditions Q = d / ((1 + r)^n - m) and m < (1 + r)^n
def conditions (d Q r m n : ℝ) : Prop :=
  Q = d / ((1 + r)^n - m) ∧ m < (1 + r)^n

theorem find_n (d Q r m : ℝ) (h : conditions d Q r m n) : 
  n = (Real.log (d / Q + m)) / (Real.log (1 + r)) :=
sorry

end find_n_l112_112514


namespace notAlwaysTriangleInSecondQuadrantAfterReflection_l112_112126

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  P : Point
  Q : Point
  R : Point

def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

def reflectionOverYEqualsX (p : Point) : Point :=
  { x := p.y, y := p.x }

def reflectTriangleOverYEqualsX (T : Triangle) : Triangle :=
  { P := reflectionOverYEqualsX T.P,
    Q := reflectionOverYEqualsX T.Q,
    R := reflectionOverYEqualsX T.R }

def triangleInSecondQuadrant (T : Triangle) : Prop :=
  isInSecondQuadrant T.P ∧ isInSecondQuadrant T.Q ∧ isInSecondQuadrant T.R

theorem notAlwaysTriangleInSecondQuadrantAfterReflection
  (T : Triangle)
  (h : triangleInSecondQuadrant T)
  : ¬ (triangleInSecondQuadrant (reflectTriangleOverYEqualsX T)) := 
sorry -- Proof not required

end notAlwaysTriangleInSecondQuadrantAfterReflection_l112_112126


namespace min_value_x_squared_plus_10x_l112_112741

theorem min_value_x_squared_plus_10x : ∃ x : ℝ, (x^2 + 10 * x) = -25 :=
by {
  sorry
}

end min_value_x_squared_plus_10x_l112_112741


namespace quadruple_perimeter_l112_112695

variable (s : ℝ) -- side length of the original square
variable (x : ℝ) -- perimeter of the original square
variable (P_new : ℝ) -- new perimeter after side length is quadrupled

theorem quadruple_perimeter (h1 : x = 4 * s) (h2 : P_new = 4 * (4 * s)) : P_new = 4 * x := 
by sorry

end quadruple_perimeter_l112_112695


namespace sum_super_cool_rectangles_areas_l112_112325

theorem sum_super_cool_rectangles_areas : 
  ∃ (s : finset ℕ), 
    (∀ a b ∈ s, a * b = 6 * (a + b) ∧ a * b = 942) → 
    ∑ n in s, n = 942 := 
sorry

end sum_super_cool_rectangles_areas_l112_112325


namespace four_points_concyclic_common_radical_axis_l112_112056

noncomputable def circles_intersection (O1 O2 : Type*) := sorry

noncomputable def arbitrary_lines (L1 L2 : Type*) := sorry

noncomputable def chord_through_first_circle (C2 L1 L2 : Type*) := sorry

noncomputable def chord_through_second_circle (C1 L1 L2 : Type*) := sorry

noncomputable def intersection_points (λ1 λ2 C1 C2 L1 L2 : Type*) := sorry

theorem four_points_concyclic_common_radical_axis 
    (O1 O2 : Type*) 
    (L1 L2 : Type*) 
    (C1 C2 : Type*)
    (P Q M N : Type*) 
    (h1 : circles_intersection O1 O2)
    (h2 : arbitrary_lines L1 L2)
    (h3 : chord_through_first_circle C2 L1 L2)
    (h4 : chord_through_second_circle C1 L1 L2)
    (h5 : intersection_points P Q M N (λ1 λ2 C1 C2 L1 L2)) 
    : 
    ∃ (O : Type*), 
    (concyclic P Q M N) ∧ (common_radical_axis O O1 O2) :=
sorry

end four_points_concyclic_common_radical_axis_l112_112056


namespace exam_total_boys_l112_112709

theorem exam_total_boys (T F : ℕ) (avg_total avg_passed avg_failed : ℕ) 
    (H1 : avg_total = 40) (H2 : avg_passed = 39) (H3 : avg_failed = 15) (H4 : 125 > 0) (H5 : 125 * avg_passed + (T - 125) * avg_failed = T * avg_total) : T = 120 :=
by
  sorry

end exam_total_boys_l112_112709


namespace problem_statement_l112_112249

-- Definitions directly from the conditions given in the problem

def prop1 : Prop := ∀ (x y : ℝ), (x + y = 0) → x = -y
def prop2 : Prop := ¬ (∀ (T U : Triangle), Congruent T U → Area T = Area U)
def prop3 : Prop := ∀ q : ℝ, (q > 1) → ¬ (∃ x : ℝ, x^2 + 2 * x + q = 0)
def prop4 : Prop := ¬ (∃ α β : ℝ, sin (α + β) = sin α + sin β)

-- The problem statement rewritten in Lean 4
theorem problem_statement : (prop1 ∧ prop3) 
  ∧ ¬ prop2 
  ∧ ¬ prop4 := 
  by
  sorry

end problem_statement_l112_112249


namespace trig_ineq_l112_112476

theorem trig_ineq {α : ℝ} (h₀ : 0 < α) (h₁ : α < Real.pi / 2) : 
  (Real.tan α + Real.cot α + Real.sec α + Real.csc α) ≥ 2 * (Real.sqrt 2 + 1) := 
by 
  sorry

end trig_ineq_l112_112476


namespace infinite_n_f_n_eq_n_l112_112144
open Nat

def natFunc (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, 1 ≤ m → 1 ≤ n → 
    (∃ k : ℕ, 1 ≤ k ∧ k ≤ f(n) ∧ n ∣ f(m + k) ∧
    (∀ i : ℕ, 1 ≤ i → i < k → ¬ n ∣ f(m + i)))

theorem infinite_n_f_n_eq_n (f : ℕ → ℕ) (H : natFunc f) : 
  ∃ (infinitely_many : ℕ → Prop), infinite {n | infinitely_many n ∧ f(n) = n} :=
sorry

end infinite_n_f_n_eq_n_l112_112144


namespace math_proof_problem_l112_112889

noncomputable theory

variables {k : ℝ}

def line_passes_fixed_point : Prop :=
  ∀ x y, (kx - 3*y + 2*k + 3 = 0) → (x = -2) ∧ (y = 1)

def range_of_k_avoiding_fourth_quadrant : Prop :=
  k ≥ 0 → ∀ x y, (k*x - 3*y + 2*k + 3) / 3 ≥ 0

def minimum_area_triangle_AOB : Prop :=
  ∀ A B, (A = (- (2*k + 3) / k, 0)) → (B = (0, (2*k + 3) / 3)) → 
    ∃ S, (S = (1 / 6) * (4*k + 9/k + 12)) ∧ S = 4 ∧ (kx - 3*y + 2*k + 3 = 0) → (x - 2*y + 4 = 0)

theorem math_proof_problem : line_passes_fixed_point ∧ range_of_k_avoiding_fourth_quadrant ∧ minimum_area_triangle_AOB :=
begin
  split,
  { 
    sorry,  -- Proof for line passes through fixed point
  },
  split,
  { 
    sorry,  -- Proof for range of k avoiding fourth quadrant
  },
  { 
    sorry,  -- Proof for minimum area of triangle AOB
  }
end

end math_proof_problem_l112_112889


namespace pattern_formula_l112_112175

theorem pattern_formula (n : ℤ) : n * (n + 2) = (n + 1) ^ 2 - 1 := 
by sorry

end pattern_formula_l112_112175


namespace max_soap_boxes_l112_112277

theorem max_soap_boxes :
  ∀ (L_carton W_carton H_carton L_soap_box W_soap_box H_soap_box : ℕ)
   (V_carton V_soap_box : ℕ) 
   (h1 : L_carton = 25) 
   (h2 : W_carton = 42)
   (h3 : H_carton = 60) 
   (h4 : L_soap_box = 7)
   (h5 : W_soap_box = 6)
   (h6 : H_soap_box = 10)
   (h7 : V_carton = L_carton * W_carton * H_carton)
   (h8 : V_soap_box = L_soap_box * W_soap_box * H_soap_box),
   V_carton / V_soap_box = 150 :=
by
  intros
  sorry

end max_soap_boxes_l112_112277


namespace part_a_part_b_l112_112986

def n_good (n m : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card ≤ 2 * n ∧ ∀ p ∈ S, p.prime ∧ p^2 ∣ m

theorem part_a (n a b : ℕ) (co : Nat.gcd a b = 1) :
  ∃ x y : ℕ, n_good n (a * x^n + b * y^n) :=
sorry

theorem part_b (n k : ℕ) (a : Fin k → ℕ) (co : Nat.gcd (Finset.univ.card) a = 1) :
  ∃ x : Fin k → ℕ, n_good n (Finset.univ.sum (λ i, a i * x i ^ n)) :=
sorry

end part_a_part_b_l112_112986


namespace find_angle_A_max_perimeter_triangle_l112_112551

-- Part 1: Prove the value of angle A
theorem find_angle_A (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter for BC = 3
theorem max_perimeter_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (A : ℝ) (hA : A = 2 * real.pi / 3) :
  let b := 2 * real.sqrt 3 * sin B,
      c := 2 * real.sqrt 3 * sin C,
      perimeter := 3 + b + c
  in ∀ d : ℝ, -real.pi / 6 < d ∧ d < real.pi / 6 → 
     B + C = real.pi / 3 → 
     perimeter ≤ (3 + 2 * real.sqrt 3) := sorry

end find_angle_A_max_perimeter_triangle_l112_112551


namespace area_of_triangle_PAB_l112_112016

noncomputable theory

-- Definitions of geometric entities involved
def Circle (O : Type) [MetricSpace O] : O → ℝ := λ x, x ^ 2 + y ^ 2 = 4

def Line1 (O : Type) [MetricSpace O] : O → ℝ := λ x, y = x

def Line2 (O : Type) [MetricSpace O] : O → ℝ := λ x, y = sqrt 3 * x + 4

-- Tangency condition
def TangentPoint (O : Type) [MetricSpace O] : O := λ P, Line2 O P ∧ Circle O P

-- Distance of point P from Line1
def Distance (P : Point ℝ ℝ) : ℝ := (|sqrt 3 - 1|) / sqrt 2

-- Length of segment AB
def SegmentAB : ℝ := 4

-- Computation of area of triangle PAB
def AreaOfTriangle : ℝ := 1 / 2 * SegmentAB * Distance

-- Prove that the area of the triangle PAB is sqrt 6 + sqrt 2
theorem area_of_triangle_PAB 
  (A B P : ℝ) 
  (h1 : Circle A) 
  (h2 : Line1 A) 
  (h3 : Line2 P) 
  (h4 : SegmentAB = 4) 
  (h5 : TangentPoint P) 
  : AreaOfTriangle = sqrt 6 + sqrt 2
:= sorry

end area_of_triangle_PAB_l112_112016


namespace partitions_bijection_l112_112316

variable {n k : ℕ}

-- Definition of a partition of n
def is_partition (n : ℕ) (p : List ℕ) : Prop :=
  p.sum = n ∧ (∀ x ∈ p, 1 ≤ x ∧ x ≤ n) ∧ p.sorted (≤)

-- Definition of a partition of size at most k
def is_partition_of_size_at_most (k n : ℕ) (p : List ℕ) : Prop :=
  is_partition n p ∧ p.length ≤ k

-- Definition of a partition of n where all elements are ≤ k
def is_partition_with_elements_leq (k n : ℕ) (p : List ℕ) : Prop :=
  is_partition n p ∧ (∀ x ∈ p, x ≤ k)

theorem partitions_bijection (n k : ℕ) :
  {p : List ℕ // is_partition_of_size_at_most k n p}.card =
  {p : List ℕ // is_partition_with_elements_leq k n p}.card :=
sorry

end partitions_bijection_l112_112316


namespace number_of_valid_pairs_l112_112841

def op (m n : ℕ) : ℕ :=
  if m % 2 = n % 2 then m + n else m * n

def is_valid_pair (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ op a b = 12

def M : set (ℕ × ℕ) := {p | is_valid_pair p.1 p.2}

theorem number_of_valid_pairs : (M.to_finset.card = 15) :=
  sorry

end number_of_valid_pairs_l112_112841


namespace part_1_part_2_l112_112588

-- Define the triangle and the given condition
variables {α β γ : ℝ}
axiom triangle_ABC : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π
axiom sin_identity : sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)
axiom BC_length : (sin(γ) / sin(α)) * BC = 3

-- State the main theorem parts separately
theorem part_1 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_sin_identity: sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)) :
  α = 2 * π / 3 :=
sorry

theorem part_2 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_α: α = 2 * π / 3) (h_BC_length: BC = 3) :
  let b := (2 * sqrt(3) * sin(π/6 - β)),
      c := (2 * sqrt(3) * sin(π/6 + β)) in
  (3 + 2 * sqrt(3)) :=
sorry

end part_1_part_2_l112_112588


namespace probability_exactly_3_hits_in_8_equals_1_over_25_times_exactly_5_hits_l112_112520

theorem probability_exactly_3_hits_in_8_equals_1_over_25_times_exactly_5_hits {P : ℝ} (hP1 : 0 < P) (hP2 : P < 1)
  (h : (nat.choose 8 3) * P^3 * (1 - P)^5 = (1 / 25 : ℝ) * (nat.choose 8 5) * P^5 * (1 - P)^3) : P = 5 / 6 :=
sorry

end probability_exactly_3_hits_in_8_equals_1_over_25_times_exactly_5_hits_l112_112520


namespace ways_to_fill_blanks_l112_112352

theorem ways_to_fill_blanks : 
  ∃! (a b c d : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (1 < a ∧ 2 < b ∧ 3 < c ∧ 4 < d) ∧ 
  (0 ≤ d ∧ d < c ∧ c < b ∧ b < a) ∧
  (
   (
    (\{a, b, c, d\}.erase a).card = 3 ∧ 
    (\{b, c, d\}.erase b).card = 2 ∧ 
    (\{c, d\}.erase c).card = 1 ∧ 
    (\{d\}.erase d).card = 0
   )
  ) :=
begin
  sorry
end

end ways_to_fill_blanks_l112_112352


namespace triangle_theorem_l112_112606

theorem triangle_theorem 
  (A B C : ℝ) 
  (h1 : ∀ A B C : ℝ, ∃ (a b c : ℝ), 
      a^2 - b^2 - c^2 = b * c ∧ sin^2 A - sin^2 B - sin^2 C = sin B * sin C) 
  (h2 : 0 < A ∧ A < π) : 
  (A = 2 * π / 3) ∧ 
  (∀ (BC : ℝ), BC = 3 → ∃ (a b c : ℝ), 
      a + b + c ≤ 3 + 2 * sqrt 3) :=
by
  sorry

end triangle_theorem_l112_112606


namespace find_m_minimum_f_l112_112882

open Real

-- Definitions and conditions
def a (m : ℝ) (x : ℝ) : ℝ × ℝ := (m, cos x)
def b (x : ℝ) : ℝ × ℝ := (1 + sin x, 1)
def f (m : ℝ) (x : ℝ) : ℝ := (a m x).1 * (b x).1 + (a m x).2 * (b x).2

-- Given f(π/2) = 2, prove that m = 1
theorem find_m (m : ℝ) : (f m (π / 2) = 2) → m = 1 :=
by 
  intros h
  sorry  -- proof for m = 1

-- Given m = 1, find the minimum value of f(x)
theorem minimum_f (x : ℝ) : f 1 x ≥ 1 - sqrt 2 :=
by 
  sorry  -- proof for minimum value

end find_m_minimum_f_l112_112882


namespace maximal_probability_C_n_l112_112658

open Classical
open Set

variable (A : Set ℕ) (hA : A = {1, 2})
variable (B : Set ℕ) (hB : B = {1, 2, 3})

def event_C_n (n : ℕ) : Prop := ∃ (a ∈ A) (b ∈ B), a + b = n

theorem maximal_probability_C_n :
  ∃ (n : ℕ), (2 ≤ n ∧ n ≤ 5) ∧ 
  (∀ m, (2 ≤ m ∧ m ≤ 5) → 
    probability (event_C_n A B m) ≤ probability (event_C_n A B n)) 
  ∧ (n = 3 ∨ n = 4) :=
sorry

end maximal_probability_C_n_l112_112658


namespace sum_of_valid_primes_eq_222_l112_112837

open Nat

def satisfies_conditions (p : ℕ) : Prop :=
  p % 5 = 1 ∧ p % 7 = 6 ∧ p ≤ 200

def all_valid_primes : List ℕ :=
  (List.range 201).filter (λ p, Prime p ∧ satisfies_conditions p)

theorem sum_of_valid_primes_eq_222 : (all_valid_primes.sum) = 222 :=
by 
  -- We would prove this by evaluating all_valid_primes, checking primality, and computing the sum.
  sorry

end sum_of_valid_primes_eq_222_l112_112837


namespace intersection_points_count_l112_112815

noncomputable def line1 : ℝ → ℝ := λ x, (3/4) * x + 1/2
noncomputable def line2 : ℝ → ℝ := λ x, -(1/3) * x + 1
def circle (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem intersection_points_count : 
  ∃ n : ℕ, n = 5 ∧ 
  ∃ (x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 : ℝ),
    ((line1 (real.sqrt (5 - (1/2)^2)) = (4*(1/2) + 5/2 + 3) ∨ line2 (real.sqrt (5 - (1/2)^2)) = (4*(1/2) + 5/2 + 3))
    ∧ circle x1 y1 ∧ circle x2 y2 ∧ circle x3 y3 ∧ circle x4 y4 ∧ circle x5 y5) :=
by sorry

end intersection_points_count_l112_112815


namespace red_car_distance_ahead_l112_112720

theorem red_car_distance_ahead
  (red_car_speed : ℕ) (black_car_speed : ℕ) (time_to_overtake : ℕ)
  (h_red_car_speed : red_car_speed = 40)
  (h_black_car_speed : black_car_speed = 50)
  (h_time_to_overtake : time_to_overtake = 3) :
  (black_car_speed - red_car_speed) * time_to_overtake = 30 :=
by
  rw [h_red_car_speed, h_black_car_speed, h_time_to_overtake]
  sorry

end red_car_distance_ahead_l112_112720


namespace infinite_bounded_sequence_l112_112672

noncomputable def seq (n : ℕ) : ℝ := 4 * (n * Real.sqrt 2 - (Real.floor (n * Real.sqrt 2) : ℝ))

theorem infinite_bounded_sequence :
  ∃ (x_n : ℕ → ℝ),
    (∀ n m : ℕ, n ≠ m → |x_n n - x_n m| ≥ 1 / |(n - m : ℤ)|) ∧
    bounded (set.range x_n) :=
by
  use seq
  sorry

end infinite_bounded_sequence_l112_112672


namespace roots_are_reciprocals_eq_a_minus_one_l112_112081

theorem roots_are_reciprocals_eq_a_minus_one (a : ℝ) :
  (∀ x y : ℝ, x + y = -(a - 1) ∧ x * y = a^2 → x * y = 1) → a = -1 :=
by
  intro h
  sorry

end roots_are_reciprocals_eq_a_minus_one_l112_112081


namespace isosceles_trapezoid_area_l112_112146

def is_isosceles_trapezoid (A B C D : Type) : Prop :=
  ∃ (a b c d : ℝ), -- coordinates of points A, B, C, D
    C = (b, c) ∧ D = (d, 0) ∧ -- positions of C and D
    ((b, 0) = (a, c) ∧ (d, 0) = (a, -c)) ∧ -- conditions for isosceles trapezoid
    ((c - 0) * (b - 2) - (d - c) * (a - 2) = (d - 0) * (2 - b) - (a - 2) * (d - 0)) -- parallel condition

def points_on_diagonal (X Y : Type) (B D : Type) : Prop :=
  X = (0, 0) ∧ Y = (4, 0) ∧ B = (2, 0) ∧ D = (7, 0)

def right_angles (B C X A Y D : Type) : Prop := 
  ∃ (bx by cx cy ax ay dx dy : ℝ),
    (X = (0, 0) ∧ C = (cx, cy) ∧ B = (bx, by) ∧ D = (dx, dy)) ∧
    ((bx * cx + by * cy = 0) ∧ (ax * dx + ay * dy = 0)) -- right angle conditions

def distances_and_area (B X Y D : Type) : Prop :=
  (dist B X = 2 ∧ dist X Y = 4 ∧ dist Y D = 3) ∧
  (area = 7)

theorem isosceles_trapezoid_area :
  ∀ (A B C D X Y : Type),
    is_isosceles_trapezoid A B C D →
    points_on_diagonal X Y B D →
    right_angles B C X A Y D →
    distances_and_area B X Y D →
  (∃ area, area = 7) :=
  by sorry

end isosceles_trapezoid_area_l112_112146


namespace valid_triangle_side_l112_112528

theorem valid_triangle_side (x : ℝ) (h1 : 2 + x > 6) (h2 : 2 + 6 > x) (h3 : x + 6 > 2) : x = 6 :=
by
  sorry

end valid_triangle_side_l112_112528


namespace part1_part2_l112_112611

-- Part (1): Prove that A = 2π/3 given the trigonometric condition.
theorem part1 (A B C : ℝ) (h_condition : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : 
  A = 2 * π / 3 :=
sorry

-- Part (2): Prove that the maximum perimeter is 3 + 2√3 given BC = 3 and A = 2π/3.
theorem part2 (B C : angle) (h_BC : 3 = 3) (h_A : A = 2 * π / 3) :
  (3 + (sqrt 3) * 2 ≤ perimeter (triangle.mk 3 B C)) :=
sorry

end part1_part2_l112_112611


namespace count_integers_between_sqrt5_and_sqrt50_l112_112063

theorem count_integers_between_sqrt5_and_sqrt50 
  (h1 : 2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3)
  (h2 : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8) : 
  ∃ n : ℕ, n = 5 := 
sorry

end count_integers_between_sqrt5_and_sqrt50_l112_112063


namespace neg_eight_degrees_celsius_meaning_l112_112512

-- Define the temperature in degrees Celsius
def temp_in_degrees_celsius (t : Int) : String :=
  if t >= 0 then toString t ++ "°C above zero"
  else toString (abs t) ++ "°C below zero"

-- Define the proof statement
theorem neg_eight_degrees_celsius_meaning :
  temp_in_degrees_celsius (-8) = "8°C below zero" :=
sorry

end neg_eight_degrees_celsius_meaning_l112_112512


namespace quadratic_function_properties_vertex_coordinates_range_of_n_l112_112052

theorem quadratic_function_properties (b c : ℝ) (h1 : ∀ x, - x^2 + b * x + c = 0 → (x = -1 ∨ x = 3)) :
  b = 4 ∧ c = 3 :=
by sorry

theorem vertex_coordinates (h2 : ∀ b c : ℝ, b = 4 ∧ c = 3 → ∃ x y : ℝ, x = 2 ∧ y = - x^2 + b * x + c ∧ y = 7) :
  ∃ x y : ℝ, x = 2 ∧ y = 7 :=
by sorry

theorem range_of_n (P : ℝ × ℝ) (m n : ℝ) (b c : ℝ) (h1 : b = 4) (h2 : c = 3)
  (h3 : P = (m, n)) (h4 : abs(m) < 2) 
  (h5 : y = -x^2 + 4 * x + 3) (h6 : y = n) :
  -9 < n ∧ n ≤ 7 :=
by sorry

end quadratic_function_properties_vertex_coordinates_range_of_n_l112_112052


namespace solution_is_unique_zero_l112_112722

theorem solution_is_unique_zero : ∀ (x y z : ℤ), x^3 + 2 * y^3 = 4 * z^3 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intros x y z h
  sorry

end solution_is_unique_zero_l112_112722


namespace sum_arccot_eq_pi_div_6_l112_112053

noncomputable def A : ℕ → ℝ
| 0     := 0 -- A_0 (not directly given but inferred).
| 1     := 1
| 2     := 3
| (n+2) := 4 * A (n+1) - A n

theorem sum_arccot_eq_pi_div_6 :
  ∑' n : ℕ, Real.arccot (2 * (A n) ^ 2) = Real.pi / 6 := 
sorry

end sum_arccot_eq_pi_div_6_l112_112053


namespace polynomial_irreducible_l112_112984

theorem polynomial_irreducible (n : ℕ) (hn : n > 1) :
  ¬ ∃ g h : Polynomial ℤ,
    (g.degree ≥ 1 ∧ h.degree ≥ 1 ∧ f = g * h) :=
by
  let f := Polynomial.C 3 + Polynomial.C 5 * Polynomial.X^(n-1) + Polynomial.X^n
  sorry

end polynomial_irreducible_l112_112984


namespace minimum_marbles_l112_112341

theorem minimum_marbles
  (r w b g y n : ℕ)
  (h_y : y = 4)
  (h_n : n = r + w + b + g + y)
  (h_1 : r * (r - 1) * (r - 2) * (r - 3) * (r - 4) / 120 = w * r * (r - 1) * (r - 2) * (r - 3) / 24)
  (h_2 : r * (r - 1) * (r - 2) * (r - 3) * (r - 4) / 120 = w * b * r * (r - 1) * (r - 2) / 6)
  (h_3 : w * b * g * r * (r - 1) / 2 = w * b * g * r):
  n = 27 :=
by
  sorry

end minimum_marbles_l112_112341


namespace five_not_in_seq_l112_112700

open Nat

-- Define the sequence based on the given conditions
def seq : ℕ → ℕ
| 0 := 2
| (n + 1) := 
    let prod := (List.range (n + 1)).map seq |>.foldl (*) 1 + 1
    primeDivisors prod |>.last'

-- Prove that 5 does not occur in the sequence
theorem five_not_in_seq : ¬ ∃ n, seq n = 5 :=
by
  sorry

end five_not_in_seq_l112_112700


namespace probability_of_picking_letter_in_mathematics_l112_112069

def unique_letters_in_mathematics : List Char := ['M', 'A', 'T', 'H', 'E', 'I', 'C', 'S']

def number_of_unique_letters_in_word : ℕ := unique_letters_in_mathematics.length

def total_letters_in_alphabet : ℕ := 26

theorem probability_of_picking_letter_in_mathematics :
  (number_of_unique_letters_in_word : ℚ) / total_letters_in_alphabet = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l112_112069


namespace triangle_altitude_bisector_intersection_length_l112_112719

theorem triangle_altitude_bisector_intersection_length
  (A B C D E : Type)
  (AB BC AC BD BE : ℝ)
  (h1 : AB = 5)
  (h2 : BC = 3 * Real.sqrt 2)
  (h3 : AC = 1)
  (h4 : B != A)
  (h5 : ∠B = π / 2)
  (h6 : BE^2 + AE^2 = AB^2)
  (h7 : ∠EBD = π / 4) :
  BD = Real.sqrt (5/3) := 
sorry

end triangle_altitude_bisector_intersection_length_l112_112719


namespace students_registered_for_exactly_two_classes_l112_112115

noncomputable def total_students : ℕ := 500
noncomputable def history_students : ℕ := 120
noncomputable def math_students : ℕ := 105
noncomputable def english_students : ℕ := 145
noncomputable def science_students : ℕ := 133
noncomputable def geography_students : ℕ := 107
noncomputable def all_five_classes_students : ℕ := 15
noncomputable def history_and_math_students : ℕ := 40
noncomputable def english_and_science_students : ℕ := 35
noncomputable def math_and_geography_students : ℕ := 25

theorem students_registered_for_exactly_two_classes 
  (total_students = 500) 
  (history_students = 120) 
  (math_students = 105) 
  (english_students = 145) 
  (science_students = 133) 
  (geography_students = 107) 
  (all_five_classes_students = 15) 
  (history_and_math_students = 40) 
  (english_and_science_students = 35) 
  (math_and_geography_students = 25) 
  : 25 + 20 + 10 = 55 :=
by 
  have h1 : 25 = history_and_math_students - all_five_classes_students := rfl
  have h2 : 20 = english_and_science_students - all_five_classes_students := rfl
  have h3 : 10 = math_and_geography_students - all_five_classes_students := rfl
  rw [h1, h2, h3]
  exact rfl

end students_registered_for_exactly_two_classes_l112_112115


namespace sufficient_but_not_necessary_condition_l112_112725

theorem sufficient_but_not_necessary_condition (x y m : ℝ) (h: x^2 + y^2 - 4 * x + 2 * y + m = 0):
  (m = 0) → (5 > m) ∧ ((5 > m) → (m ≠ 0)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l112_112725


namespace find_a_for_circumscribed_quadrilateral_l112_112504

theorem find_a_for_circumscribed_quadrilateral (a : ℝ) :
  let line1 := (a + 2) * x + (1 - a) * y - 3 = 0,
      line2 := (a - 1) * x + (2 * a + 3) * y + 2 = 0 in
  ((a + 2) * (a - 1) + (1 - a) * (2 * a + 3) = 0) → (a = 1 ∨ a = -1) :=
begin
  sorry,
end

end find_a_for_circumscribed_quadrilateral_l112_112504


namespace PQ_bisects_BD_l112_112944

variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]
variables {A B C D P Q M N : Point}

def convex_quadrilateral (A B C D : Point) : Prop := sorry
def midpoint (P A B : Point) : Prop := 2 • P = A + B
def bisects (line P Q : Point) (diagonal A C : Point) : Prop := 
  ∃ M, midpoint M A C ∧ (line.contains M)
def line_contains_midpoint (P Q : Point) (mid : Point) : Prop := sorry

-- The theorem we want to prove:
theorem PQ_bisects_BD 
  (h1 : convex_quadrilateral A B C D)
  (h2 : midpoint P A B)
  (h3 : midpoint Q C D)
  (h4 : bisects (P, Q) (A, C))
  : bisects (P, Q) (B, D) := 
begin
  sorry
end

end PQ_bisects_BD_l112_112944


namespace amount_each_person_gets_l112_112522

theorem amount_each_person_gets 
  (total_amount : ℤ)
  (num_persons : ℤ)
  (h_total : total_amount = 42900)
  (h_persons : num_persons = 22) : 
  total_amount / num_persons = 1950 := by
  rw [h_total, h_persons]
  norm_num
  sorry

end amount_each_person_gets_l112_112522


namespace slope_of_tangent_line_l112_112869

theorem slope_of_tangent_line : ∃ k : ℝ, (∀ f : ℝ → ℝ, (∀ x, f x = Real.exp x) → 
  ∃ x0 : ℝ, k = Real.exp x0 ∧ f x0 = x0 * k ∧ (0, 0) ∈ {(x0, Real.exp x0)} ∧ k = Real.exp x0) ∧ k = Real.exp 1 :=
begin
  sorry
end

end slope_of_tangent_line_l112_112869


namespace quadratic_has_real_roots_range_l112_112917

-- Lean 4 statement

theorem quadratic_has_real_roots_range (m : ℝ) :
  (∀ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) → m ≤ 7 ∧ m ≠ 3 :=
by
  sorry

end quadratic_has_real_roots_range_l112_112917


namespace two_digit_number_l112_112245

theorem two_digit_number (x y : Nat) : 
  10 * x + y = 10 * x + y := 
by 
  sorry

end two_digit_number_l112_112245


namespace tan_double_angle_l112_112516

theorem tan_double_angle (x : ℝ) (h : (Real.sqrt 3) * Real.cos x - Real.sin x = 0) : Real.tan (2 * x) = - (Real.sqrt 3) :=
by
  sorry

end tan_double_angle_l112_112516


namespace geom_seq_value_a3_l112_112965

theorem geom_seq_value_a3 (a : ℕ → ℤ) (r : ℤ) (h_geom : ∀ (n : ℕ), a (n + 1) = a n * r)
  (h_cond1 : a 2 * a 5 = -32)
  (h_cond2 : a 3 + a 4 = 4)
  (h_int_ratio : ∃ k : ℤ, r = k) : 
  a 3 = -4 := 
begin
  sorry
end

end geom_seq_value_a3_l112_112965


namespace no_rational_roots_l112_112966

theorem no_rational_roots (p q : ℤ) (h1 : p % 3 = 2) (h2 : q % 3 = 2) :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ a * a = b * b * (p^2 - 4 * q) :=
by
  sorry

end no_rational_roots_l112_112966


namespace smallest_positive_real_number_l112_112402

noncomputable def smallest_x : ℝ :=
  let x := 89 / 9 in x

theorem smallest_positive_real_number :
  ∀ x : ℝ, (x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8) → x ≥ smallest_x  :=
by
  sorry

end smallest_positive_real_number_l112_112402


namespace find_ellipse_equation_l112_112464

noncomputable def ellipse_equation (a b : ℝ) (h_eq : a ≠ b) (ha : a > 0) (hb : b > 0) :
  ax^2 + by^2 = 1 :=
sorry

theorem find_ellipse_equation (a b : ℝ) (h_eq : a ≠ b) (ha : a > 0) (hb : b > 0)
  (h1 : ∀ (x y : ℝ), (ax^2 + by^2 = 1) ∧ (x + y - 1 = 0))
  (h2 : |AB| = 2 * sqrt 2)
  (h3 : slope OC = sqrt 2 / 2) :
  ellipse_equation a b h_eq ha hb = x^2 / 3 + sqrt 2 * y^2 / 3 :=
sorry

end find_ellipse_equation_l112_112464


namespace smallest_n_for_multiples_of_2015_l112_112471

theorem smallest_n_for_multiples_of_2015 (n : ℕ) (hn : 0 < n)
  (h5 : (2^n - 1) % 5 = 0)
  (h13 : (2^n - 1) % 13 = 0)
  (h31 : (2^n - 1) % 31 = 0) : n = 60 := by
  sorry

end smallest_n_for_multiples_of_2015_l112_112471


namespace find_k_l112_112875

theorem find_k (k x : ℝ) (h: x = 2) (h_eq : k / x + (x - 3) / (x - 1) = 1) : k = 4 := by
  -- Substitute x = 2 into the equation
  rw [h] at h_eq
  -- Simplify the equation
  have eq1 : k / 2 + (2 - 3) / (2 - 1) = 1 := by rw [h] at h_eq; exact h_eq
  -- Further simplify to: k / 2 - 1 = 1
  have eq2 : k / 2 - 1 = 1 := by linarith
  -- Solve for k: k / 2 = 2
  have eq3 : k / 2 = 2 := by linarith
  -- Finally, k = 4
  have eq4 : k = 4 := by linarith
  -- Conclude the proof
  exact eq4

end find_k_l112_112875


namespace smallest_positive_real_number_l112_112401

noncomputable def smallest_x : ℝ :=
  let x := 89 / 9 in x

theorem smallest_positive_real_number :
  ∀ x : ℝ, (x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8) → x ≥ smallest_x  :=
by
  sorry

end smallest_positive_real_number_l112_112401


namespace triangle_A_value_and_max_perimeter_l112_112582

theorem triangle_A_value_and_max_perimeter (A B C a b c : ℝ) 
  (h1 : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C)
  (h2 : a = 3) :
  (A = 2 * Real.pi / 3) ∧ (a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by
  sorry

end triangle_A_value_and_max_perimeter_l112_112582


namespace probability_last_box_single_ball_l112_112093

noncomputable def probability_last_box_only_ball (n : ℕ) : ℝ :=
  let p := 1 / n in
  (n * p * (1 - p)^(n - 1))

theorem probability_last_box_single_ball :
  probability_last_box_only_ball 100 ≈ 0.370 :=
by
  sorry

end probability_last_box_single_ball_l112_112093


namespace is_geometric_sequence_sum_of_first_n_terms_l112_112546

variable {a : ℕ → ℝ}
noncomputable def a_seq : ℕ → ℝ
| 0          => 1
| (n + 1)    => (n + 1) / (2 * n) * a_seq n

theorem is_geometric_sequence (n : ℕ) :
  ∀ n : ℕ, a_seq n / n = (1 / 2) ^ (n - 1) :=
sorry

theorem sum_of_first_n_terms (n : ℕ) :
  ∑ i in finset.range (n + 1), a_seq i = 4 - (n + 2) * (1 / 2) ^ (n - 1) :=
sorry

end is_geometric_sequence_sum_of_first_n_terms_l112_112546


namespace area_of_right_isosceles_triangle_l112_112121

-- Define a right triangle with given angles and side length
noncomputable def triangle_area : ℝ :=
  let (AB BC AC : ℝ) := (6 * Real.sqrt 2, 6 * Real.sqrt 2, 12) in
  1 / 2 * AB * BC

theorem area_of_right_isosceles_triangle (AB BC AC : ℝ) 
  (h1 : ∠B = 90°) (h2 : ∠A = ∠C) (h3 : AC = 12) : 
  (1 / 2 * AB * BC) = 36 := 
by
  -- Variables representing the sides of the triangle
  have hAB_BC_eq : AB = BC,
  from sorry, -- It follows from the given angles that AB = BC in an isosceles right triangle
  -- Calculate the lengths of AB and BC using AC
  have hAB_length : AB = 6 * Real.sqrt 2,
  from sorry, -- It follows from the property of isosceles right triangles that AB = AC / sqrt(2)
  -- Similarly for BC
  have hBC_length : BC = 6 * Real.sqrt 2,
  from sorry,
  -- Calculate the area
  calc
    1 / 2 * AB * BC = 1 / 2 * (6 * Real.sqrt 2) * (6 * Real.sqrt 2) : by rw [hAB_length, hBC_length]
    ... = 1 / 2 * 72 : by norm_num
    ... = 36 : by norm_num

end area_of_right_isosceles_triangle_l112_112121


namespace Andy_solves_exactly_two_problems_l112_112800

/-- 
Proof Problem: Given the conditions, prove that Andy solves exactly 2 problems.
Conditions:
1. Andy solves every prime-numbered problem from 78 to 125 inclusive.
2. He only solves problems with odd digit sums.

Correct Answer: 2 problems.
-/
theorem Andy_solves_exactly_two_problems :
  let primes_in_range := {p ∈ Finset.Icc 78 125 | Nat.Prime p} in
  let odd_digit_sum (n : Nat) : Bool := (n.digits 10).sum % 2 = 1 in
  (primes_in_range.filter odd_digit_sum).card = 2 :=
by
  sorry

end Andy_solves_exactly_two_problems_l112_112800


namespace hyperbola_eccentricity_l112_112693

theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c = real.sqrt (a * a + b * b)) :
  let e := c / a in
  e = (real.sqrt (3 + real.sqrt 5)) / 2 :=
begin
  -- proof omitted
  sorry
end

end hyperbola_eccentricity_l112_112693


namespace smallest_positive_integer_modulo_l112_112731

theorem smallest_positive_integer_modulo (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 13) (h3 : -1234 ≡ n [MOD 13]) : n = 1 :=
sorry

end smallest_positive_integer_modulo_l112_112731


namespace rearrangement_of_pieces_l112_112256

-- Define the pieces and their initial placement on the 12 fields
structure Field := (index: Nat)
structure Piece := (color: String)
def initial_placements := [("Red", 1), ("Yellow", 2), ("Green", 3), ("Blue", 4)]

-- Define the movement rules
def move (f: Field) (p: Piece) (direction: Bool) : Field :=
  if direction then Field.mk ((f.index + 4) % 12)
  else Field.mk ((f.index + 8) % 12)

-- Define the cyclic permutations
def permutation_groups := [
  ["Red", "Yellow", "Green", "Blue"],
  ["Yellow", "Green", "Blue", "Red"],
  ["Green", "Blue", "Red", "Yellow"],
  ["Blue", "Red", "Yellow", "Green"]
]

-- Prove the permutations after several moves
theorem rearrangement_of_pieces (pieces: List (String × Field)) :
  (pieces.map Prod.fst) ∈ permutation_groups :=
sorry

end rearrangement_of_pieces_l112_112256


namespace inequality_solution_has_3_integer_solutions_l112_112436

theorem inequality_solution_has_3_integer_solutions (m : ℝ) :
  (∃ x ∈ set.Icc (-4) (-2), x ∈ ℤ ∧ (x + 5 > 0) ∧ (x - m ≤ 1)) →
  (-3 ≤ m ∧ m < -2) :=
by sorry

end inequality_solution_has_3_integer_solutions_l112_112436


namespace proof_range_of_a_l112_112500

/-- p is the proposition that for all x in [1,2], x^2 - a ≥ 0 --/
def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

/-- q is the proposition that there exists an x0 in ℝ such that x0^2 + (a-1)x0 + 1 < 0 --/
def q (a : ℝ) : Prop := ∃ x0 : ℝ, x0^2 + (a-1)*x0 + 1 < 0

theorem proof_range_of_a (a : ℝ) : (p a ∨ q a) ∧ (¬p a ∧ ¬q a) → (a ≥ -1 ∧ a ≤ 1) ∨ a > 3 :=
by
  sorry -- proof will be filled out here

end proof_range_of_a_l112_112500


namespace eighth_term_of_arithmetic_sequence_l112_112723

noncomputable def arithmetic_sequence (n : ℕ) (a1 an : ℚ) (k : ℕ) : ℚ :=
  a1 + (k - 1) * ((an - a1) / (n - 1))

theorem eighth_term_of_arithmetic_sequence :
  ∀ (a1 a30 : ℚ), a1 = 5 → a30 = 86 → 
  arithmetic_sequence 30 a1 a30 8 = 592 / 29 :=
by
  intros a1 a30 h_a1 h_a30
  rw [h_a1, h_a30]
  dsimp [arithmetic_sequence]
  sorry

end eighth_term_of_arithmetic_sequence_l112_112723


namespace word_problem_points_l112_112332

theorem word_problem_points :
  ∃ x : ℕ,
    let num_problems := 30,
    let num_computation := 20,
    let points_computation := 3,
    let total_points := 110,
    let total_computation_points := num_computation * points_computation,
    let num_word_problems := num_problems - num_computation,
    let remaining_points := total_points - total_computation_points
    in num_word_problems * x = remaining_points ∧ x = 5 :=
begin
  sorry
end

end word_problem_points_l112_112332


namespace range_of_a_l112_112147

noncomputable def M : Set ℝ := {2, 0, -1}
noncomputable def N (a : ℝ) : Set ℝ := {x | abs (x - a) < 1}

theorem range_of_a (a : ℝ) : (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 3) ↔ M ∩ N a = {x} :=
by
  sorry

end range_of_a_l112_112147


namespace exists_f_satisfies_p_no_f_satisfies_q_l112_112868

-- Definitions for mappings p and q
def p : ℕ+ → ℕ+
| 1  := 2
| 2  := 3
| 3  := 4
| 4  := 1
| n  := n  -- For n ≥ 5

def q : ℕ+ → ℕ+
| 1  := 3
| 2  := 4
| 3  := 2
| 4  := 1
| n  := n  -- For n ≥ 5

-- Problem 1: Existence of f such that f(f(n)) = p(n) + 2 for all n
theorem exists_f_satisfies_p : ∃ (f : ℕ+ → ℕ+), ∀ n : ℕ+, f (f n) = p n + 2 :=
  sorry

-- Problem 2: Nonexistence of f such that f(f(n)) = q(n) + 2 for all n
theorem no_f_satisfies_q : ¬ ∃ (f : ℕ+ → ℕ+), ∀ n : ℕ+, f (f n) = q n + 2 :=
  sorry

end exists_f_satisfies_p_no_f_satisfies_q_l112_112868


namespace max_value_tan_half_theta_occurs_at_l112_112378

noncomputable def max_value_tan_half_theta (theta : ℝ) : ℝ := 
  @Real.tan (θ / 2) * (1 - @Real.sin θ)

theorem max_value_tan_half_theta_occurs_at :
  ∃ θ : ℝ, (-Real.pi / 2) < θ ∧ θ < (Real.pi / 2) ∧ 
  (max_value_tan_half_theta θ = max (λ x, max_value_tan_half_theta x) {x | -Real.pi / 2 < x ∧ x < Real.pi / 2}) ∧ 
  θ = 2 * @Real.arctan ((-2 + Real.sqrt(7)) / 3) :=
sorry

end max_value_tan_half_theta_occurs_at_l112_112378


namespace four_sin_t_plus_cos_2t_bounds_l112_112182

theorem four_sin_t_plus_cos_2t_bounds (t : ℝ) : -5 ≤ 4 * Real.sin t + Real.cos (2 * t) ∧ 4 * Real.sin t + Real.cos (2 * t) ≤ 3 := by
  sorry

end four_sin_t_plus_cos_2t_bounds_l112_112182


namespace cos_squared_sum_l112_112428

theorem cos_squared_sum (x : ℝ) (h₁ : 90 < x) (h₂ : x < 180) 
  (h₃ : cos (2 * x * π / 180)^2 + cos (7 * x * π / 180)^2 = 12 * cos (5 * x * π / 180)^2 * cos (3 * x * π / 180)^2) : 
  ∃ xs : List ℝ, (∀ x ∈ xs, 90 < x ∧ x < 180 ∧ cos (2 * x * π / 180)^2 + cos (7 * x * π / 180)^2 = 12 * cos (5 * x * π / 180)^2 * cos (3 * x * π / 180)^2) ∧ xs.sum = 285 := 
sorry

end cos_squared_sum_l112_112428


namespace find_angle_A_max_perimeter_triangle_l112_112555

-- Part 1: Prove the value of angle A
theorem find_angle_A (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter for BC = 3
theorem max_perimeter_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (A : ℝ) (hA : A = 2 * real.pi / 3) :
  let b := 2 * real.sqrt 3 * sin B,
      c := 2 * real.sqrt 3 * sin C,
      perimeter := 3 + b + c
  in ∀ d : ℝ, -real.pi / 6 < d ∧ d < real.pi / 6 → 
     B + C = real.pi / 3 → 
     perimeter ≤ (3 + 2 * real.sqrt 3) := sorry

end find_angle_A_max_perimeter_triangle_l112_112555


namespace P_Q_sum_l112_112905

noncomputable def find_P_Q_sum (P Q : ℚ) : Prop :=
  ∀ x : ℚ, (x^2 + 3 * x + 7) * (x^2 + (51/7) * x - 2) = x^4 + P * x^3 + Q * x^2 + 45 * x - 14

theorem P_Q_sum :
  ∃ P Q : ℚ, find_P_Q_sum P Q ∧ (P + Q = 260 / 7) :=
by
  sorry

end P_Q_sum_l112_112905


namespace magnitude_of_sum_is_sqrt2_l112_112057

def magnitude {α : Type*} [normed_group α] (v: α) : ℝ := ∥v∥

noncomputable def a (m : ℤ) : ℤ × ℤ := (2 * m + 1, 3)

noncomputable def b (m : ℤ) : ℤ × ℤ := (2, m)

def opposite_vectors {α : Type*} [has_neg α] (v1 v2: α) : Prop :=
  ∃ t : ℝ, t < 0 ∧ v1 = t • v2

theorem magnitude_of_sum_is_sqrt2 (m: ℤ)
 (h : opposite_vectors (a m) (b m)) : 
  magnitude (a m + b m) = real.sqrt 2 := 
  sorry

end magnitude_of_sum_is_sqrt2_l112_112057


namespace sum_of_all_N_l112_112814

-- Define the machine's processing rules
def process (N : ℕ) : ℕ :=
  if N % 2 = 1 then 4 * N + 2 else N / 2

-- Define the 6-step process starting from N
def six_steps (N : ℕ) : ℕ :=
  process (process (process (process (process (process N)))))

-- Definition for the main theorem
theorem sum_of_all_N (N : ℕ) : six_steps N = 10 → N = 640 :=
by 
  sorry

end sum_of_all_N_l112_112814


namespace smallest_positive_integer_solution_l112_112732

theorem smallest_positive_integer_solution :
  ∃ x : ℕ, 0 < x ∧ 5 * x ≡ 17 [MOD 34] ∧ (∀ y : ℕ, 0 < y ∧ 5 * y ≡ 17 [MOD 34] → x ≤ y) :=
sorry

end smallest_positive_integer_solution_l112_112732


namespace ms_cole_total_students_l112_112173

def number_of_students (S6 : Nat) (S4 : Nat) (S7 : Nat) : Nat :=
  S6 + S4 + S7

theorem ms_cole_total_students (S6 S4 S7 : Nat)
  (h1 : S6 = 40)
  (h2 : S4 = 4 * S6)
  (h3 : S7 = 2 * S4) :
  number_of_students S6 S4 S7 = 520 := by
  sorry

end ms_cole_total_students_l112_112173


namespace largest_share_l112_112949

/-- Five partners in a business decide to split the profits of their company in the ratio 1:2:3:4:6.
If the profit one year is $48,000, prove that the largest number of dollars received by any of the five partners is $18,000. -/
theorem largest_share (total_profit : ℕ) (ratios : List ℕ) (largest_ratio : ℕ) 
  (h_total_profit : total_profit = 48000) 
  (h_ratios : ratios = [1, 2, 3, 4, 6]) 
  (h_largest_ratio : largest_ratio = 6) : 
  let part_value := total_profit / ratios.sum in largest_ratio * part_value = 18000 := sorry

end largest_share_l112_112949


namespace collinear_points_sqrt_500_condition_l112_112143

open Real -- To handle real numbers

-- Define the initial points and assumption on the circle
variables {P : ℕ → ℝ× ℝ}

-- Define the conditions for P
def on_circle (P₀ P₁ P₂ : (ℝ × ℝ)) (r : ℝ) : Prop :=
  dist P₀ (0, 0) = r ∧ dist P₁ (0, 0) = r ∧ dist P₂ (0, 0) = r 

def points_initialized (P P₀ P₁ P₂: (ℕ → ℝ × ℝ)) (r : ℝ) (t : ℝ): Prop :=
  on_circle (P₀ 0) (P₁ 1) (P₂ 2) r ∧ dist (P₁ 1) (P₂ 2) = t

-- Definition of P_i for i ≥ 3
def circumcenter (A B C : (ℝ × ℝ)) : (ℝ × ℝ) := 
  sorry -- The actual definition would be needed

def P_recurrence (P : ℕ → ℝ × ℝ) (n : ℕ) : Prop :=
  ∀ i ≥ 3, P i = circumcenter (P (i-1)) (P (i-2)) (P (i-3))

-- Proof statement for collinearity of P_1, P_5, P_9, ...
theorem collinear_points 
  {P : ℕ → ℝ × ℝ} {r t : ℝ}
  (h1: points_initialized P (P 0) (P 1) (P 2) r t)
  (h2 : P_recurrence P) : 
  ∃ L : ℝ → ℝ, ∀ k : ℕ, P (1+4*k) ∈ L :=
sorry

-- Proof statement for determining t such that sqrt[500](x/y) is an integer
theorem sqrt_500_condition 
  {P : ℕ → ℝ × ℝ} {r t : ℝ}
  (h1: points_initialized P (P 0) (P 1) (P 2) r t)
  (h2 : P_recurrence P) : 
  ∀ x y : ℝ, x = dist (P 1) (P 1001) ∧ y = dist (P 1001) (P 2001) → 
  ∃ n : ℕ, 500.sqrt (x / y) = n :=
sorry

end collinear_points_sqrt_500_condition_l112_112143


namespace candies_shared_equally_l112_112506

theorem candies_shared_equally (total_candies : ℕ) (pct_sour : ℕ) (people : ℕ) : 
  total_candies = 300 → 
  pct_sour = 40 → 
  people = 3 → 
  (total_candies * pct_sour / 100) = 120 → 
  (total_candies - 120) = 180 → 
  (180 / people) = 60 → 
  60 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h6, h5, h4, h3, h2, h1]
  exact 60
  sorry

end candies_shared_equally_l112_112506


namespace demand_exceeds_only_Jul_Aug_l112_112336

def cumulative_demand (n : ℕ) : ℝ :=
  (n / 27) * (21 * n - n^2 - 5)

noncomputable def demand_exceeds (n : ℕ) : Prop :=
  cumulative_demand n > 50

noncomputable def months_exceeding_demand : set ℕ :=
  { n | demand_exceeds n }

theorem demand_exceeds_only_Jul_Aug :
  months_exceeding_demand = {7, 8} := 
sorry

end demand_exceeds_only_Jul_Aug_l112_112336


namespace kate_pen_cost_l112_112982

noncomputable def pen_cost (P : ℝ) : Prop :=
  let k := (1/3 * P) -- Kate's money
  in k + 20 = P

theorem kate_pen_cost (P : ℝ) (h : pen_cost P) : P = 30 := by
  sorry

end kate_pen_cost_l112_112982


namespace find_inserts_to_make_median_10_l112_112497

def initial_set : Set ℕ := {5, 6, 3, 8, 4}

def updated_set (a b : ℕ) : Set ℕ := initial_set ∪ {a, b}

def is_median (m : ℕ) (s : Set ℕ) : Prop :=
  ∃ (l r : List ℕ), l ++ [m] ++ r = List.sort s ∧ l.length = r.length

theorem find_inserts_to_make_median_10 :
  ∃ a b, a ≤ b ∧ is_median 10 (updated_set a b) :=
by
  use 10, 11
  -- Proof steps will follow here.
  sorry

end find_inserts_to_make_median_10_l112_112497


namespace opposite_of_neg_two_thirds_l112_112228

theorem opposite_of_neg_two_thirds : - (- (2 / 3) : ℚ) = (2 / 3 : ℚ) :=
by
  sorry

end opposite_of_neg_two_thirds_l112_112228


namespace exterior_angle_relation_l112_112969

variable (BM AC D : Point)
variable (A B C : Triangle BM AC D)
variable (θ a b x : Angle)
variable (ABM CBM BAC BCA MBD ACD : Angle)

-- Conditions
axiom BM_bisects_B : ∠A B BM = θ ∧ ∠C B BM = θ
axiom AC_extended_D : ∃ D, on_line AC A ∧ on_line AC D
axiom DBM_right_angle : ∠DBM = 90
axiom triangle_angle_sum : a + b + 2 * θ = 180
axiom exterior_angle : x = a + θ

theorem exterior_angle_relation :
  x = 180 - (b + θ) :=
sorry

end exterior_angle_relation_l112_112969


namespace triangle_congruence_l112_112235

-- Assume ABC is a triangle
variables (A B C O : Point)
-- Assume O is the center of the circumcircle of triangle ABC
variable [circumcenter O (triangle A B C)]
-- Assume A1, B1, C1 are symmetric to O with respect to sides BC, CA, and AB respectively
variables (A1 B1 C1 : Point)
variable [symmetric_point A1 O (segment B C)]
variable [symmetric_point B1 O (segment C A)]
variable [symmetric_point C1 O (segment A B)]

-- Define the congruence of two triangles
def triangle := λ (A B C : Point), (segment A B) ∧ (segment B C) ∧ (segment C A)

-- Prove the triangles ABC and A1B1C1 are congruent
theorem triangle_congruence : triangle A B C = triangle A1 B1 C1 := by
  sorry

end triangle_congruence_l112_112235


namespace sum_S_2n3_l112_112660

def sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = 1 / (2 ^ n)) ∧ (∀ n : ℕ, S n = ∑ i in finset.range n, a (i + 1))

theorem sum_S_2n3 {a : ℕ → ℝ} {S : ℕ → ℝ} (h : sequence a S) (n : ℕ) : 
  S (2 * n + 3) = (4 / 3) * (1 - 1 / (4 ^ (n + 2))) :=
by
  cases h with h_a1 h_rest
  cases h_rest with h_anplus h_sn
  sorry

end sum_S_2n3_l112_112660


namespace circle_area_l112_112669

theorem circle_area (A B : ℝ × ℝ) (hA : A = (4, 15)) (hB : B = (12, 11))
  (tangent_X : ∃ C : ℝ × ℝ, C.2 = 0 ∧ tangent_line A = tangent_line B) :
  ∃ (r : ℝ), r^2 = 231.25 ∧ (area_of_circle r = 231.25 * real.pi) :=
by { sorry }

end circle_area_l112_112669


namespace inequality_solution_has_3_integer_solutions_l112_112434

theorem inequality_solution_has_3_integer_solutions (m : ℝ) :
  (∃ x ∈ set.Icc (-4) (-2), x ∈ ℤ ∧ (x + 5 > 0) ∧ (x - m ≤ 1)) →
  (-3 ≤ m ∧ m < -2) :=
by sorry

end inequality_solution_has_3_integer_solutions_l112_112434


namespace fred_grew_38_cantelopes_l112_112000

def total_cantelopes : Nat := 82
def tim_cantelopes : Nat := 44
def fred_cantelopes : Nat := total_cantelopes - tim_cantelopes

theorem fred_grew_38_cantelopes : fred_cantelopes = 38 :=
by
  sorry

end fred_grew_38_cantelopes_l112_112000


namespace proof_problem_l112_112489

-- Definitions for skew lines, planes, and intersections
structure Line := (a : Type)
structure Plane := (α : Type)
structure Space := (s : Type)

-- Given conditions about skew lines
def skew_lines (a b : Line) : Prop :=
  ∀ (α β : Plane), (a.belongs_to α) ∧ (b.belongs_to β) → ¬(α = β) ∧ (¬∃ (p : Point), p ∈ a ∧ p ∈ b)

def plane_intersection (α β : Plane) (c : Line) : Prop :=
  ∀ (p : Point), p ∈ c → (p ∈ α ∧ p ∈ β)

-- Statement I: If line a in plane α and line b in plane β are skew lines, and line c is the intersection line of α and β, 
-- then c intersects at most one of a or b.
def statement_I :=
  ∀ (a b c : Line) (α β : Plane),
  skew_lines a b ∧ plane_intersection α β c → (∃ (p : Point), p ∈ c → (p ∈ a ∨ p ∈ b)) → (¬(p ∈ a ∧ p ∈ b))

-- Statement II: There does not exist an infinite number of lines such that any two of them are skew lines.
noncomputable def statement_II :=
  ∀ (s : Set Line), (Infinite s) → (∀ (a b : Line), a ∈ s ∧ b ∈ s → a ≠ b → skew_lines a b) → False

-- The theorem that combines everything
theorem proof_problem :
  ¬statement_I ∧ ¬statement_II :=
by
  sorry

end proof_problem_l112_112489


namespace max_rooks_attacking_each_other_l112_112728

theorem max_rooks_attacking_each_other (k n : ℕ) (hkn : k ≤ n) :
  let max_rooks := if n > 2 * k then 2 * k else 2 * (n + k) / 3 in
  -- representing the maximum number in terms of a conditional expression
  max_rooks = if n > 2 * k then 2 * k else 2 * ⌊(k + n) / 3⌋ :=
by sorry

end max_rooks_attacking_each_other_l112_112728


namespace part1_part2_l112_112454

variables (x y z : ℝ)

-- Conditions
def conditions := (x >= 0) ∧ (y >= 0) ∧ (z >= 0) ∧ (x + y + z = 1)

-- Part 1: Prove 2(x^2 + y^2 + z^2) + 9xyz >= 1
theorem part1 (h : conditions x y z) : 2 * (x^2 + y^2 + z^2) + 9 * x * y * z ≥ 1 :=
sorry

-- Part 2: Prove xy + yz + zx - 3xyz ≤ 1/4
theorem part2 (h : conditions x y z) : x * y + y * z + z * x - 3 * x * y * z ≤ 1 / 4 :=
sorry

end part1_part2_l112_112454


namespace opposite_of_neg_two_thirds_l112_112230

theorem opposite_of_neg_two_thirds : - (- (2 / 3) : ℚ) = (2 / 3 : ℚ) :=
by
  sorry

end opposite_of_neg_two_thirds_l112_112230


namespace find_fourth_speed_l112_112314

theorem find_fourth_speed 
  (avg_speed : ℝ)
  (speed1 speed2 speed3 fourth_speed : ℝ)
  (h_avg_speed : avg_speed = 11.52)
  (h_speed1 : speed1 = 6.0)
  (h_speed2 : speed2 = 12.0)
  (h_speed3 : speed3 = 18.0)
  (expected_avg_speed_eq : avg_speed = 4 / ((1 / speed1) + (1 / speed2) + (1 / speed3) + (1 / fourth_speed))) :
  fourth_speed = 2.095 :=
by 
  sorry

end find_fourth_speed_l112_112314


namespace tangent_lines_through_origin_l112_112884

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 1

variable (a : ℝ)

theorem tangent_lines_through_origin 
  (h1 : ∃ m1 m2 : ℝ, m1 ≠ m2 ∧ (f a (-m1) + f a (m1 + 2)) / 2 = f a 1) :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ (f a t1 * (1 / t1) = f a 0) ∧ (f a t2 * (1 / t2) = f a 0) := 
sorry

end tangent_lines_through_origin_l112_112884


namespace problem_1_problem_2_l112_112447

open Real

def vec_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

def vec_perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem problem_1 (k : ℝ) : 
  let a := (3, 4)
  let b := (2, k)
  (vec_parallel (a.1 + 2 * b.1, a.2 + 2 * b.2) (a.1 - b.1, a.2 - b.2)) →
  k = 8 / 3 := sorry

theorem problem_2 (k : ℝ) : 
  let a := (3, 4)
  let b := (2, k)
  (vec_perpendicular (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2)) →
  k = sqrt 21 ∨ k = - sqrt 21 := sorry

end problem_1_problem_2_l112_112447


namespace find_smallest_x_l112_112413

def smallest_x_satisfying_condition : Prop :=
  ∃ x : ℝ, x > 0 ∧ (⌊x^2⌋ - x * ⌊x⌋ = 8) ∧ x = 89 / 9

theorem find_smallest_x : smallest_x_satisfying_condition :=
begin
  -- proof goes here (not required for this task)
  sorry
end

end find_smallest_x_l112_112413


namespace sum_of_coefficients_l112_112030

theorem sum_of_coefficients (n : ℕ) (h : binomial n 1 + binomial n 3 = 2 * binomial n 2) :
  polynomial.sum_of_coefficients (polynomial.expand (x - (2 / (sqrt x))) n) = -1 :=
by
  sorry

end sum_of_coefficients_l112_112030


namespace range_of_m_l112_112437

open Set

theorem range_of_m (m : ℝ) :
  (∃ f : ℤ → Prop, (∀ x, f x ↔ x + 5 > 0 ∧ x - m ≤ 1) ∧ (∃ a b c : ℤ, f a ∧ f b ∧ f c))
  → (-3 ≤ m ∧ m < -2) := 
sorry

end range_of_m_l112_112437


namespace smallest_positive_real_number_l112_112404

noncomputable def smallest_x : ℝ :=
  Inf {x : ℝ | x > 0 ∧ (floor (x^2) - x * floor x = 8)}

theorem smallest_positive_real_number :
  smallest_x = 89 / 9 :=
by 
  sorry

end smallest_positive_real_number_l112_112404


namespace trigonometric_bound_l112_112759

open Real

theorem trigonometric_bound (x y : ℝ) : 
  -1/2 ≤ (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ∧ 
  (x + y) * (1 - x * y) / ((1 + x^2) * (1 + y^2)) ≤ 1/2 :=
by 
  sorry

end trigonometric_bound_l112_112759


namespace congruent_triangle_side_equality_l112_112006

-- Define the conditions and the target statement
theorem congruent_triangle_side_equality (A B C D E F : Type) 
  (ABC DEF : Triangle A B C) 
  (h_congruent : Triangle_congruent ABC DEF)
  (h_AB : segment_length AB = 3) : 
  segment_length DE = 3 :=
by
  sorry

end congruent_triangle_side_equality_l112_112006


namespace new_tank_volume_l112_112771

-- Define initial conditions
variable (r h : ℝ)
def initial_volume : ℝ := 5

-- Define transformations
def new_radius := 2 * r
def new_height := 3 * h
def pi : ℝ := Real.pi -- Lean's predefined pi constant

-- Define initial volume condition
axiom initial_volume_condition : pi * r^2 * h = initial_volume

-- Define the new volume based on new dimensions
def new_volume := pi * (new_radius)^2 * (new_height)

-- State the theorem to be proved
theorem new_tank_volume : new_volume = 60 :=
by
  sorry

end new_tank_volume_l112_112771


namespace part1_part2_l112_112628

theorem part1 (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : A = 2 * π / 3 :=
sorry

theorem part2 (b c : ℝ) (h1 : sin (2 * π / 3) ^ 2 - sin b ^ 2 - sin c ^ 2 = sin b * sin c)
  (h2 : b + c = 2 * sqrt 3) : 3 * 2 + b + c = 3 + 2 * sqrt 3 :=
sorry

end part1_part2_l112_112628


namespace smallest_c_for_sequence_satisfying_condition_l112_112835

theorem smallest_c_for_sequence_satisfying_condition :
  (∀ {x : ℕ → ℝ} (n : ℕ), x 1 > 0 → (∀ m, x (m+1) > 0) →
      (∀ n, (finset.range n).sum x ≤ x (n + 1)) →
      (∀ n, (finset.range n).sum (λ m, real.sqrt (x m)) ≤ (real.sqrt 2 + 1) * real.sqrt ((finset.range n).sum x))) :=
sorry

end smallest_c_for_sequence_satisfying_condition_l112_112835


namespace triangle_ABC_properties_l112_112760

theorem triangle_ABC_properties
  (A B C H M N : Type)
  [triang : Triangle]
  {F E : triang.Apex}
  (h1 : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (h2 : acute_angled_triangle A B C)
  (h3 : Altitude A C H)
  (h4 : Altitude C A E)
  (h5 : Midpoint M A H)
  (h6 : Midpoint N C H)
  (h7 : FM_parallel_EN F M E N)
  (h8 : dist F M = 1)
  (h9 : dist E N = 4) :
  (angle A B C = 60) ∧ (area_triangle A B C = 18 * sqrt 3) ∧ (radius_circumcircle A B C = 2 * sqrt 7) := 
  by
    sorry

end triangle_ABC_properties_l112_112760


namespace team_b_can_serve_on_submarine_l112_112327

   def can_serve_on_submarine (height : ℝ) : Prop := height ≤ 168

   def average_height_condition (avg_height : ℝ) : Prop := avg_height = 166

   def median_height_condition (median_height : ℝ) : Prop := median_height = 167

   def tallest_height_condition (max_height : ℝ) : Prop := max_height = 169

   def mode_height_condition (mode_height : ℝ) : Prop := mode_height = 167

   theorem team_b_can_serve_on_submarine (H : median_height_condition 167) :
     ∀ (h : ℝ), can_serve_on_submarine h :=
   sorry
   
end team_b_can_serve_on_submarine_l112_112327


namespace min_value_of_quadratic_l112_112736

theorem min_value_of_quadratic (x : ℝ) : ∃ m : ℝ, (∀ x, x^2 + 10 * x ≥ m) ∧ m = -25 := by
  sorry

end min_value_of_quadratic_l112_112736


namespace range_of_m_l112_112438

open Set

theorem range_of_m (m : ℝ) :
  (∃ f : ℤ → Prop, (∀ x, f x ↔ x + 5 > 0 ∧ x - m ≤ 1) ∧ (∃ a b c : ℤ, f a ∧ f b ∧ f c))
  → (-3 ≤ m ∧ m < -2) := 
sorry

end range_of_m_l112_112438


namespace find_angle_A_max_perimeter_l112_112560

noncomputable def sin_sq_minus (A B C : ℝ) : ℝ :=
  (Real.sin A) * (Real.sin A) - (Real.sin B) * (Real.sin B) - (Real.sin C) * (Real.sin C)

noncomputable def sin_prod (B C : ℝ) : ℝ :=
  (Real.sin B) * (Real.sin C)

theorem find_angle_A (A B C : ℝ) (h : sin_sq_minus A B C = sin_prod B C) :
  A = 2 * Real.pi / 3 :=
by
  sorry

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem max_perimeter (B C : ℝ) (a : ℝ) (h1 : a = 3) (h2 : B + C = Real.pi / 3) :
  ∃ (b c : ℝ), perimeter a b c = 3 + 2 * Real.sqrt 3 :=
by
  sorry

end find_angle_A_max_perimeter_l112_112560


namespace find_digit_B_l112_112267

theorem find_digit_B (A B : ℕ) (h1 : A3B = 100 * A + 30 + B)
  (h2 : 0 ≤ A ∧ A ≤ 9)
  (h3 : 0 ≤ B ∧ B ≤ 9)
  (h4 : A3B - 41 = 591) : 
  B = 2 := 
by sorry

end find_digit_B_l112_112267


namespace solution_set_f_l112_112866

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then exp x - cos x else exp (-x) - cos (-x)

theorem solution_set_f (x : ℝ) :
  f (x - 1) - 1 < exp (real.pi) ↔ 1 - real.pi < x ∧ x < 1 + real.pi :=
by
  -- Prove that f is an even function
  have feven : ∀ x, f x = f (-x) := by
    intro x
    unfold f
    split_ifs with h
    { rw [←cos_neg x, neg_neg x] }
    { rw [neg_neg x, ←cos_neg, if_neg] }
  sorry -- proof steps would go here, but we are including sorry as specified.

-- The statement should build successfully as is.

end solution_set_f_l112_112866


namespace problem_1_problem_2_problem_3_l112_112045

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b - 8) * x - a - a * b

theorem problem_1
  (a b : ℝ) :
  a < 0 ∧ f a b (-3) = 0 ∧ f a b 2 = 0 → f a b = -3 * x^2 - 3 * x + 18 :=
sorry

theorem problem_2
  (a b : ℝ) (x : ℝ) :
  f a b = -3 * x^2 - 3 * x + 18 → x > -1 → (∀ x, x > -1 → ∀ t, t = x + 1 → -3*(t + 1/t - 1) ≤ -3) :=
sorry

theorem problem_3 
  (a k b : ℝ) (x : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → ax^2 + kx - b > 0) → k < 2 * sqrt 15 :=
sorry

end problem_1_problem_2_problem_3_l112_112045


namespace exists_infinite_subset_l112_112651

theorem exists_infinite_subset 
  (f g : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x < g x) : 
  ∃ S : set ℝ, S.infinite ∧ (∀ x ∈ S, ∀ y ∈ S, f x < g y) :=
by
  sorry

end exists_infinite_subset_l112_112651


namespace bisects_diagonals_l112_112927

-- Define the data structure for a convex quadrilateral
structure ConvexQuadrilateral (α : Type*) :=
(A B C D : α)

-- Define midpoints of line segments
def midpoint {α : Type*} [Add α] [Div α] [Nonempty α] (A B : α) : α :=
(A + B) / 2

-- Main theorem stating the problem
theorem bisects_diagonals
  {α : Type*} [AddCommGroup α] [Module ℝ α] (quad : ConvexQuadrilateral α)
  (P Q : α)
  (hP : P = midpoint quad.A quad.B)
  (hQ : Q = midpoint quad.C quad.D)
  (hPQ : ∃ M, M = midpoint quad.A quad.C ∧ M ∈ line_through P Q) :
  ∃ N, N = midpoint quad.B quad.D ∧ N ∈ line_through P Q :=
sorry

lemma line_through (P Q : α) : Prop :=
∃ (λ1 λ2 : ℝ), P + λ1 • (Q - P) = Q + λ2 • (P - Q)

end bisects_diagonals_l112_112927


namespace minimum_value_of_quadratic_expression_l112_112735

theorem minimum_value_of_quadratic_expression : ∃ x ∈ ℝ, ∀ y ∈ ℝ, x^2 + 10 * x ≤ y^2 + 10 * y := by
  sorry

end minimum_value_of_quadratic_expression_l112_112735


namespace AMHSE_1988_l112_112909

theorem AMHSE_1988 (x y : ℝ) (h1 : |x| + x + y = 10) (h2 : x + |y| - y = 12) : x + y = 18 / 5 :=
sorry

end AMHSE_1988_l112_112909


namespace smallest_number_of_ones_l112_112390

-- Definitions inferred from the problem conditions
def N := (10^100 - 1) / 3
def M_k (k : ℕ) := (10^k - 1) / 9

theorem smallest_number_of_ones (k : ℕ) : M_k k % N = 0 → k = 300 :=
by {
  sorry
}

end smallest_number_of_ones_l112_112390


namespace percent_increase_is_equivalent_l112_112132

variable {P : ℝ}

theorem percent_increase_is_equivalent 
  (h1 : 1.0 + 15.0 / 100.0 = 1.15)
  (h2 : 1.15 * (1.0 + 25.0 / 100.0) = 1.4375)
  (h3 : 1.4375 * (1.0 + 10.0 / 100.0) = 1.58125) :
  (1.58125 - 1) * 100 = 58.125 :=
by
  sorry

end percent_increase_is_equivalent_l112_112132


namespace jackie_free_time_correct_l112_112974

noncomputable def jackie_free_time : ℕ :=
  let total_hours_in_a_day := 24
  let hours_working := 8
  let hours_exercising := 3
  let hours_sleeping := 8
  let total_activity_hours := hours_working + hours_exercising + hours_sleeping
  total_hours_in_a_day - total_activity_hours

theorem jackie_free_time_correct : jackie_free_time = 5 := by
  sorry

end jackie_free_time_correct_l112_112974


namespace annual_interest_rate_l112_112979

theorem annual_interest_rate (principal total_paid: ℝ) (h_principal : principal = 150) (h_total_paid : total_paid = 162) : 
  ((total_paid - principal) / principal) * 100 = 8 :=
by
  sorry

end annual_interest_rate_l112_112979


namespace transistors_in_2005_l112_112530

theorem transistors_in_2005
  (initial_count : ℕ)
  (doubles_every : ℕ)
  (triples_every : ℕ)
  (years : ℕ) :
  initial_count = 500000 ∧ doubles_every = 2 ∧ triples_every = 6 ∧ years = 15 →
  (initial_count * 2^(years/doubles_every) + initial_count * 3^(years/triples_every)) = 68500000 :=
by
  sorry

end transistors_in_2005_l112_112530


namespace arithmetic_progression_condition_l112_112829

theorem arithmetic_progression_condition
  (a b c : ℝ) : ∃ (A B : ℤ), A ≠ 0 ∧ B ≠ 0 ∧ (b - a) * B = (c - b) * A := 
by {
  sorry
}

end arithmetic_progression_condition_l112_112829


namespace cos_half_angle_l112_112903

theorem cos_half_angle (α : ℝ) (h : sin (α / 4) = sqrt 3 / 3) : cos (α / 2) = 1 / 3 := 
by
  sorry

end cos_half_angle_l112_112903


namespace decreasing_sine_monotonic_l112_112525

theorem decreasing_sine_monotonic (f : ℝ → ℝ) (varphi : ℝ) :
  (∀ x y, 0 < x → x < y → y < π/2 → f(x) ≥ f(y)) →
  (∀ x, f(x) = sin(2 * x + varphi)) →
  varphi = π/2 :=
by
  sorry

end decreasing_sine_monotonic_l112_112525


namespace jackie_free_time_correct_l112_112975

noncomputable def jackie_free_time : ℕ :=
  let total_hours_in_a_day := 24
  let hours_working := 8
  let hours_exercising := 3
  let hours_sleeping := 8
  let total_activity_hours := hours_working + hours_exercising + hours_sleeping
  total_hours_in_a_day - total_activity_hours

theorem jackie_free_time_correct : jackie_free_time = 5 := by
  sorry

end jackie_free_time_correct_l112_112975


namespace intersection_distance_l112_112499

theorem intersection_distance :
  let curve_C (θ : ℝ) : ℝ × ℝ := (4 / (sin θ)^2 * cos θ, 0)
  let line_l (t : ℝ) : ℝ × ℝ := (1 + t, -1 + t)
  let standard_eq_C (x y : ℝ) : Prop := y^2 = 4 * x
  let standard_eq_l (x y : ℝ) : Prop := x - y = 2
  ∃ A B : ℝ × ℝ, 
    standard_eq_C A.1 A.2 ∧ standard_eq_l A.1 A.2 ∧ 
    standard_eq_C B.1 B.2 ∧ standard_eq_l B.1 B.2 ∧ 
    dist A B = 4 * sqrt 6 :=
by 
  sorry

end intersection_distance_l112_112499


namespace original_price_l112_112782

theorem original_price 
  (profit : ℝ) 
  (profit_perc : ℝ) 
  (h_profit : profit = 800) 
  (h_profit_perc : profit_perc = 0.60) :
  ∃ P : ℝ, P = 800 / 0.6 :=
begin
  sorry
end

end original_price_l112_112782


namespace shortest_major_axis_ellipse_l112_112498

-- Definitions based on the given conditions
def line_l (P : ℝ × ℝ) : Prop := P.2 = P.1 + 9

def P_on_line_l (P : ℝ × ℝ) : Prop := line_l P

def f1 := (-3 : ℝ, 0 : ℝ)
def f2 := ( 3 : ℝ, 0 : ℝ)

-- Define the equation of the ellipse
def ellipse_eq (a b : ℝ) (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

-- Main theorem statement
theorem shortest_major_axis_ellipse :
  ∃ P : ℝ × ℝ, P_on_line_l P ∧ ellipse_eq 45 36 P.1 P.2 :=
sorry

end shortest_major_axis_ellipse_l112_112498


namespace percent_of_b_l112_112911

theorem percent_of_b (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : b = 2.5 * a) : c = 0.1 * b := 
by
  sorry

end percent_of_b_l112_112911


namespace log_sum_of_geom_seq_l112_112430

variable {a : ℕ → ℝ}

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n * a (10 - n) = a m * a (10 - m)

theorem log_sum_of_geom_seq
  (h1 : geom_seq a)
  (h2 : 0 < a 1)
  (h3 : 0 < a 2)
  (h4 : 0 < a 3)
  (h5 : 0 < a 4)
  (h6 : 0 < a 5)
  (h7 : 0 < a 6)
  (h8 : 0 < a 7)
  (h9 : 0 < a 8)
  (h10 : 0 < a 9)
  (h11 : 0 < a 10)
  (h12 : a 4 * a 7 + a 5 * a 6 = 16) :
  \log 2 (a 1) + \log 2 (a 2) + \log 2 (a 3) + \log 2 (a 4) + \log 2 (a 5) + \log 2 (a 6) + \log 2 (a 7) + \log 2 (a 8) + \log 2 (a 9) + \log 2 (a 10) = 15 :=
  sorry

end log_sum_of_geom_seq_l112_112430


namespace smallest_positive_x_l112_112418

theorem smallest_positive_x (x : ℝ) (h : ⌊x^2⌋ - x * ⌊x⌋ = 8) : x = 89 / 9 :=
sorry

end smallest_positive_x_l112_112418


namespace term_number_sequence_l112_112502

theorem term_number_sequence (n : ℕ) : 
  (∃ k : ℕ, 3 * real.sqrt 5 = real.sqrt (2 * k - 1) ∧ k = 23) :=
begin
  use 23,
  split,
  { rw mul_comm,
    exact calc
      real.sqrt 45 = real.sqrt (2 * 23 - 1) : by norm_num
                  ... = 3 * real.sqrt 5 : by rw real.sqrt_mul,
    },
  { refl, }
end

end term_number_sequence_l112_112502


namespace convert_2e_15pi_i4_to_rectangular_form_l112_112357

noncomputable def convert_to_rectangular_form (z : ℂ) : ℂ :=
  let θ := (15 * Real.pi) / 4
  let θ' := θ - 2 * Real.pi
  2 * Complex.exp (θ' * Complex.I)

theorem convert_2e_15pi_i4_to_rectangular_form :
  convert_to_rectangular_form (2 * Complex.exp ((15 * Real.pi) / 4 * Complex.I)) = (Real.sqrt 2 - Complex.I * Real.sqrt 2) :=
  sorry

end convert_2e_15pi_i4_to_rectangular_form_l112_112357


namespace product_of_geometric_sequence_l112_112149

variable {a : ℕ → ℝ}
variable {n : ℕ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ i : ℕ, i > 0 → a (i + 1) = a i * q

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, a (i + 1))

def sum_of_reciprocals_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, 1 / a (i + 1))

theorem product_of_geometric_sequence 
  (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (ha : ∀ i, 0 < a i)
  (h_geo : is_geometric_sequence a q)
  (h_sum : sum_of_first_n_terms a n = S)
  (h_reciprocal_sum : sum_of_reciprocals_first_n_terms a n = T):
  (Finset.range n).prod (λ i, a (i + 1)) = (S / T) ^ (n / 2) :=
  by sorry

end product_of_geometric_sequence_l112_112149


namespace find_angle_A_max_perimeter_l112_112563

noncomputable def sin_sq_minus (A B C : ℝ) : ℝ :=
  (Real.sin A) * (Real.sin A) - (Real.sin B) * (Real.sin B) - (Real.sin C) * (Real.sin C)

noncomputable def sin_prod (B C : ℝ) : ℝ :=
  (Real.sin B) * (Real.sin C)

theorem find_angle_A (A B C : ℝ) (h : sin_sq_minus A B C = sin_prod B C) :
  A = 2 * Real.pi / 3 :=
by
  sorry

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem max_perimeter (B C : ℝ) (a : ℝ) (h1 : a = 3) (h2 : B + C = Real.pi / 3) :
  ∃ (b c : ℝ), perimeter a b c = 3 + 2 * Real.sqrt 3 :=
by
  sorry

end find_angle_A_max_perimeter_l112_112563


namespace complex_number_solution_l112_112376

noncomputable def z : ℂ := 3 - (40 / 7 : ℝ) * Complex.I

theorem complex_number_solution :
  3 * z - 4 * Complex.conj z = -3 - 40 * Complex.I :=
by
  sorry

end complex_number_solution_l112_112376


namespace sum_of_areas_of_super_cool_rectangles_l112_112323

def is_super_cool (a b : ℕ) : Prop :=
  a * b = 6 * (a + b)

theorem sum_of_areas_of_super_cool_rectangles :
  ∑ (a, b) in { (a, b) : ℕ × ℕ | is_super_cool a b }  a * b = 942 :=
sorry

end sum_of_areas_of_super_cool_rectangles_l112_112323


namespace range_of_a_l112_112485

open Complex Real

theorem range_of_a (a : ℝ) (h : abs (1 + a * Complex.I) ≤ 2) : a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end range_of_a_l112_112485


namespace negation_of_p_l112_112891

variable (x : ℝ)

def proposition_p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 1

theorem negation_of_p : ¬ (∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by sorry

end negation_of_p_l112_112891


namespace exists_4_cycle_same_company_l112_112113

open Classical

variables {City : Type} [Fintype City]

/-- We assume the existence of 6 cities -/
axiom cities : Fintype.card City = 6

/-- Defining air routes as a relation between cities and companies -/
noncomputable def AirRoute : City → City → Prop :=
  λ x y, x ≠ y ∨ ∃ (c : Bool), (c = true ∨ c = false)

/-- The main theorem statement -/
theorem exists_4_cycle_same_company (f : (City × City → Bool)) :
  exists (A B C D : City), f (A, B) = f (B, C) ∧ f (B, C) = f (C, D) ∧ f (C, D) = f (D, A) :=
sorry

end exists_4_cycle_same_company_l112_112113


namespace leak_empty_time_proof_l112_112779

noncomputable def leak_empty_time (L: ℝ) (inlet_rate: ℝ) (total_volume: ℝ) (net_empty_time: ℝ): ℝ :=
  total_volume / L

theorem leak_empty_time_proof:
  let L := 864 in  -- The rate at which the leak empties the tank in litres per hour
  let inlet_rate := 360 in  -- The rate at which the inlet fills water in litres per hour
  let total_volume := 6048.000000000001 in  -- The total volume of the tank in litres
  let net_empty_time := 12 in  -- Time in hours to empty the tank with the inlet on
  leak_empty_time L inlet_rate total_volume net_empty_time = 7 :=
by
  /- The proof goes here -/
  sorry

end leak_empty_time_proof_l112_112779


namespace arithmetic_sequence_sum_l112_112539

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = 4) (h_a101 : a 101 = 36) : 
  a 9 + a 52 + a 95 = 60 :=
sorry

end arithmetic_sequence_sum_l112_112539


namespace volume_of_right_triangular_prism_l112_112857

theorem volume_of_right_triangular_prism (S : ℝ) (a : ℝ) (h₁ : S = 18)
  (h₂ : 6 * a^2 = S) :
  let h := 2 * a in
  (1 / 2 * a * h * a) = 9 / 2 :=
by
  sorry

end volume_of_right_triangular_prism_l112_112857


namespace sector_central_angle_l112_112038

noncomputable def sector_angle (R L : ℝ) : ℝ := L / R

theorem sector_central_angle :
  ∃ R L : ℝ, 
    (R > 0) ∧ 
    (L > 0) ∧ 
    (1 / 2 * L * R = 5) ∧ 
    (2 * R + L = 9) ∧ 
    (sector_angle R L = 8 / 5 ∨ sector_angle R L = 5 / 2) :=
sorry

end sector_central_angle_l112_112038


namespace sum_of_smallest_and_largest_eq_2y_l112_112203

variable (a n y : ℤ) (hn_even : Even n) (hy : y = a + n - 1)

theorem sum_of_smallest_and_largest_eq_2y : a + (a + 2 * (n - 1)) = 2 * y := 
by
  sorry

end sum_of_smallest_and_largest_eq_2y_l112_112203


namespace num_diagonals_of_pentagon_l112_112001

theorem num_diagonals_of_pentagon :
  ∀ (V : Finset ℕ), V.card = 5 →
  ∀ (v ∈ V), (∃ D : Finset ℕ, D.card = 3 ∧ D ⊆ V ∧ v ∉ D) :=
by
  intros V hV v hv
  sorry

end num_diagonals_of_pentagon_l112_112001


namespace numberOfCorrectPropositions_is_1_l112_112339

noncomputable def CorrectPropositions : Prop :=
  let prop1 := (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔ (∀ x : ℝ, x > 1 ∨ x < -1 → x^2 > 1)
  let P := ∀ x : ℝ, sin x ≤ 1
  let Q := ∀ a b : ℝ, a < b → a^2 < b^2
  let prop2 := P ∧ Q
  let prop3 := (¬ (∃ x : ℝ, x^2 - x > 0)) ↔ (∀ x : ℝ, x^2 - x ≤ 0)
  let prop4 := (∀ x : ℝ, x > 2 → x^2 > 4) ∧ (¬ (∀ x : ℝ, x^2 > 4 → x > 2))
  (¬ prop1) ∧ (¬ prop2) ∧ prop3 ∧ (¬ prop4)

theorem numberOfCorrectPropositions_is_1 : CorrectPropositions :=
  sorry

end numberOfCorrectPropositions_is_1_l112_112339


namespace opposite_of_neg_two_thirds_l112_112233

theorem opposite_of_neg_two_thirds : -(- (2 / 3)) = (2 / 3) :=
by
  sorry

end opposite_of_neg_two_thirds_l112_112233


namespace part1_part2_l112_112614

-- Part (1): Prove that A = 2π/3 given the trigonometric condition.
theorem part1 (A B C : ℝ) (h_condition : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : 
  A = 2 * π / 3 :=
sorry

-- Part (2): Prove that the maximum perimeter is 3 + 2√3 given BC = 3 and A = 2π/3.
theorem part2 (B C : angle) (h_BC : 3 = 3) (h_A : A = 2 * π / 3) :
  (3 + (sqrt 3) * 2 ≤ perimeter (triangle.mk 3 B C)) :=
sorry

end part1_part2_l112_112614


namespace find_a_no_solution_l112_112828

noncomputable def no_solution_eq (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (8 * |x - 4 * a| + |x - a^2| + 7 * x - 2 * a = 0)

theorem find_a_no_solution :
  ∀ a : ℝ, no_solution_eq a ↔ (a < -22 ∨ a > 0) :=
by
  intro a
  sorry

end find_a_no_solution_l112_112828


namespace petya_coloring_l112_112280

theorem petya_coloring (k : ℕ) : k = 1 :=
  sorry

end petya_coloring_l112_112280


namespace zero_exists_in_interval_l112_112865

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x

def g (x : ℝ) : ℝ := f x - Real.log (f x ^ 3)

theorem zero_exists_in_interval : ∃ x ∈ (Ioo 0 1 : Set ℝ), g x = 0 :=
by
  sorry

end zero_exists_in_interval_l112_112865


namespace binomial_sum_inequality_l112_112157

theorem binomial_sum_inequality
  (k n r : ℕ)
  (n_ge_1 : n ≥ 1)
  (k_ge_1 : k ≥ 1)
  (r_le_n_minus_1 : r ≤ n - 1)
  (a : Fin n → ℤ)
  (sum_eq_kn_plus_r : (Finset.univ.sum a) = k * n + r) :
  (Finset.univ.sum (λ i, (a i.choose 2))) ≥ r * ((k + 1).choose 2) + (n - r) * (k.choose 2) :=
sorry

end binomial_sum_inequality_l112_112157


namespace smallest_positive_x_l112_112419

theorem smallest_positive_x (x : ℝ) (h : ⌊x^2⌋ - x * ⌊x⌋ = 8) : x = 89 / 9 :=
sorry

end smallest_positive_x_l112_112419


namespace bisect_diagonal_BD_l112_112932

-- Define a convex quadrilateral ABCD with midpoints P and Q on sides AB and CD respectively.
variables {A B C D P Q M N : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables [Inhabited P] [Inhabited Q] [Inhabited M] [Inhabited N]

-- Assuming the given statements:
-- ABCD is a convex quadrilateral
-- P is the midpoint of AB
-- Q is the midpoint of CD
-- Line PQ bisects the diagonal AC

-- Prove that line PQ also bisects the diagonal BD
theorem bisect_diagonal_BD 
  (convex_quadrilateral : convex_quadrilateral A B C D)
  (midpoint_P : midpoint P A B)
  (midpoint_Q : midpoint Q C D)
  (PQ_bisects_AC : bisects_line PQ M A C) :
  bisects_line PQ N B D :=
sorry  -- Proof is omitted

end bisect_diagonal_BD_l112_112932


namespace triang_radii_eq_l112_112224

variables {R r : ℝ} (n : ℕ)
def consecutive_sides (n : ℕ) : ℝ × ℝ × ℝ := (n - 1, n, n + 1)
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
def heron_area (a b c p : ℝ) : ℝ := real.sqrt (p * (p - a) * (p - b) * (p - c))
noncomputable def incircle_radius (a b c p S : ℝ) : ℝ := S / p
noncomputable def circumcircle_radius (a b c S : ℝ) : ℝ := (a * b * c) / (4 * S)

theorem triang_radii_eq (n : ℕ) (h1 : n > 1)
  (h2 : ∃ R r, ∀ (a b c : ℝ), consecutive_sides n = (a, b, c) → 
     let p := semi_perimeter a b c,
         S := heron_area a b c p in
     R = circumcircle_radius a b c S ∧ r = incircle_radius a b c p S):
  ∀ R r, R = 2 * r + 1 / (2 * r) :=
by intros; sorry

end triang_radii_eq_l112_112224


namespace tray_height_l112_112790

-- Declare the main theorem with necessary given conditions.
theorem tray_height (a b c : ℝ) (side_length : ℝ) (cut_distance : ℝ) (angle : ℝ) : 
  (side_length = 150) →
  (cut_distance = Real.sqrt 50) →
  (angle = 45) →
  a^2 + b^2 = c^2 → -- Condition from Pythagorean theorem
  a = side_length * Real.sqrt 2 / 2 - cut_distance → -- Calculation for half diagonal minus cut distance
  b = (side_length * Real.sqrt 2 / 2 - cut_distance) / 2 → -- Perpendicular from R to the side
  side_length = 150 → -- Ensure consistency of side length
  b^2 + c^2 = side_length^2 → -- Ensure we use another Pythagorean relation
  c = Real.sqrt 7350 → -- Derived c value
  c = Real.sqrt 1470 := -- Simplified form of c.
  sorry

end tray_height_l112_112790


namespace find_divisor_l112_112916

theorem find_divisor (x y : ℕ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 14) / y = 4) : y = 10 :=
sorry

end find_divisor_l112_112916


namespace vector_ratio_l112_112897

variables {α : Type*} [InnerProductSpace ℝ α]

theorem vector_ratio (a b c : α) (ha_perp: inner a c = 0) (h120 : inner a b = (real.sqrt(3)/2) * ∥a∥ * ∥b∥) (c_def : c = a + b) : ∥a∥ / ∥b∥ = 1 / 2 :=
by 
  have h := calc 
    inner a (a + b) = _ : by sorry
  sorry

end vector_ratio_l112_112897


namespace probability_last_box_single_ball_l112_112091

noncomputable def probability_last_box_only_ball (n : ℕ) : ℝ :=
  let p := 1 / n in
  (n * p * (1 - p)^(n - 1))

theorem probability_last_box_single_ball :
  probability_last_box_only_ball 100 ≈ 0.370 :=
by
  sorry

end probability_last_box_single_ball_l112_112091


namespace real_and_imaginary_parts_equal_l112_112072

noncomputable def complex_number (a : ℝ) : ℂ := (a + complex.I) / complex.I

theorem real_and_imaginary_parts_equal (a : ℝ) (h : (complex.re (complex_number a)) = (complex.im (complex_number a))) : a = -1 :=
by 
  sorry

end real_and_imaginary_parts_equal_l112_112072


namespace necessary_not_sufficient_condition_l112_112003

theorem necessary_not_sufficient_condition (a : ℝ) :
  (a < 2) ∧ (a^2 - 4 < 0) ↔ (a < 2) ∧ (a > -2) :=
by
  sorry

end necessary_not_sufficient_condition_l112_112003


namespace part1_part2_l112_112848

-- Definition of p: x² + 2x - 8 < 0
def p (x : ℝ) : Prop := x^2 + 2 * x - 8 < 0

-- Definition of q: (x - 1 + m)(x - 1 - m) ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := (x - 1 + m) * (x - 1 - m) ≤ 0

-- Define A as the set of real numbers that satisfy p
def A : Set ℝ := { x | p x }

-- Define B as the set of real numbers that satisfy q when m = 2
def B (m : ℝ) : Set ℝ := { x | q x m }

theorem part1 : A ∩ B 2 = { x | -1 ≤ x ∧ x < 2 } :=
sorry

-- Prove that m ≥ 5 is the range for which p is a sufficient but not necessary condition for q
theorem part2 : ∀ m : ℝ, (∀ x: ℝ, p x → q x m) ∧ (∃ x: ℝ, q x m ∧ ¬p x) ↔ m ≥ 5 :=
sorry

end part1_part2_l112_112848


namespace probability_last_box_contains_exactly_one_ball_l112_112094

-- Definitions and conditions
def num_boxes : ℕ := 100
def num_balls : ℕ := 100
def p : ℝ := 1 / num_boxes.toReal

-- To show: The probability that the last box contains exactly one ball
theorem probability_last_box_contains_exactly_one_ball :
  ((1 - p) ^ (num_boxes - 1)) ≈ 0.370 :=
by
  sorry

end probability_last_box_contains_exactly_one_ball_l112_112094


namespace sum_of_smallest_and_largest_l112_112199

def even_consecutive_sequence_sum (a n : ℤ) : ℤ :=
  a + a + 2 * (n - 1)

def arithmetic_mean (a n : ℤ) : ℤ :=
  (a * n + n * (n - 1)) / n

theorem sum_of_smallest_and_largest (a n y : ℤ) (h_even : even n) (h_mean : y = arithmetic_mean a n) :
  even_consecutive_sequence_sum a n = 2 * y :=
by
  sorry

end sum_of_smallest_and_largest_l112_112199


namespace union_complements_l112_112661

open Set

variable (U : Set ℕ) (A B : Set ℕ)

-- Define the conditions
def condition_U : U = {1, 2, 3, 4, 5} := by
  sorry

def condition_A : A = {1, 2, 3} := by
  sorry

def condition_B : B = {2, 3, 4} := by
  sorry

-- Prove that (complement_U A) ∪ (complement_U B) = {1, 4, 5}
theorem union_complements :
  (U \ A) ∪ (U \ B) = {1, 4, 5} := by
  sorry

end union_complements_l112_112661


namespace melanie_bread_slices_l112_112166

theorem melanie_bread_slices (bread_pieces slice_pieces : ℕ) : bread_pieces = 8 → slice_pieces = 4 → bread_pieces / slice_pieces = 2 :=
by 
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end melanie_bread_slices_l112_112166


namespace triangle_A_value_and_max_perimeter_l112_112583

theorem triangle_A_value_and_max_perimeter (A B C a b c : ℝ) 
  (h1 : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C)
  (h2 : a = 3) :
  (A = 2 * Real.pi / 3) ∧ (a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by
  sorry

end triangle_A_value_and_max_perimeter_l112_112583


namespace smallest_even_five_digit_number_tens_place_l112_112217

theorem smallest_even_five_digit_number_tens_place (d₁ d₂ d₃ d₄ d₅ : ℕ) : 
  (d₁, d₂, d₃, d₄, d₅) ∈ ({1, 2, 3, 5, 8}).subsets 5 →
  ∃ smallest_even_number n, 
    n = list.to_nat [d₁, d₂, d₃, d₄, d₅] ∧ 
    even n ∧ 
    (∀ m, list.to_nat [m.1.head, m.1.nth 1, m.1.nth 2, m.1.nth 3, m.1.nth 4] = m.2 → even m.2 → n ≤ m.2) ∧ 
    (list.to_nat [d₁, d₂, d₃, d₄, d₅] % 10 = 2) ∧ -- ensure its evenness
    (d₄ = 8) := 
begin
  sorry
end

end smallest_even_five_digit_number_tens_place_l112_112217


namespace range_of_a_l112_112527

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end range_of_a_l112_112527


namespace baker_bakes_1740_loaves_in_3_weeks_l112_112298

theorem baker_bakes_1740_loaves_in_3_weeks :
  (let loaves_per_oven_per_hour := 5 in
  let number_of_ovens := 4 in
  let hours_per_day_weekdays := 5 in
  let hours_per_day_weekends := 2 in
  let days_week := 5 in
  let weekend_days := 2 in
  let weeks := 3 in
  let hourly_production := loaves_per_oven_per_hour * number_of_ovens in
  let daily_production_weekdays := hourly_production * hours_per_day_weekdays in
  let daily_production_weekends := hourly_production * hours_per_day_weekends in
  let weekly_production := (daily_production_weekdays * days_week) + (daily_production_weekends * weekend_days) in
  let total_production := weekly_production * weeks in
  total_production = 1740) :=
by
  sorry

end baker_bakes_1740_loaves_in_3_weeks_l112_112298


namespace probability_of_not_red_purple_black_is_0_l112_112770

def bag : list (string × ℕ) :=
 [("white", 50), ("green", 40), ("yellow", 20),
  ("red", 30), ("purple", 30), ("blue", 10), ("black", 20)]

def total_balls := 200

def red_purple_black : ℕ := 30 + 30 + 20

def not_red_purple_black : ℕ := total_balls - red_purple_black

def probability_not_red_purple_black : ℚ :=
  not_red_purple_black / total_balls

theorem probability_of_not_red_purple_black_is_0.6 : 
  probability_not_red_purple_black = 0.6 := 
  sorry

end probability_of_not_red_purple_black_is_0_l112_112770


namespace PQ_bisects_BD_l112_112939

variable (A B C D P Q : Type) [Add A] (M : A) [Div A Two]

theorem PQ_bisects_BD
  (h1 : convex_quadrilateral A B C D)
  (h2 : midpoint A B P)
  (h3 : midpoint C D Q)
  (h4 : bisects P Q A C) :
  bisects P Q B D :=
sorry

end PQ_bisects_BD_l112_112939


namespace part1_part2_l112_112612

-- Part (1): Prove that A = 2π/3 given the trigonometric condition.
theorem part1 (A B C : ℝ) (h_condition : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : 
  A = 2 * π / 3 :=
sorry

-- Part (2): Prove that the maximum perimeter is 3 + 2√3 given BC = 3 and A = 2π/3.
theorem part2 (B C : angle) (h_BC : 3 = 3) (h_A : A = 2 * π / 3) :
  (3 + (sqrt 3) * 2 ≤ perimeter (triangle.mk 3 B C)) :=
sorry

end part1_part2_l112_112612


namespace geom_sequence_general_term_max_term_T_n_l112_112032

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

noncomputable def sum_terms (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

noncomputable def T (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  let S_n := sum_terms a₁ q n in S_n + 1 / S_n

theorem geom_sequence_general_term {a₁ q : ℝ} (h_init : a₁ = 3 / 2) (h_ratio : q = -1 / 2) :
  ∀ n : ℕ, geometric_sequence a₁ q n = (-1)^(n - 1) * 3 / 2^n :=
begin
  sorry
end 

theorem max_term_T_n {a₁ q : ℝ} (h_init : a₁ = 3 / 2) (h_ratio : q = -1 / 2) :
  ∀ n ∈ (∅ : set ℕ), T a₁ q n ≤ 13 / 6 :=
begin
  sorry
end

end geom_sequence_general_term_max_term_T_n_l112_112032


namespace sum_S_eq_50000000_l112_112840

def S : Set ℚ :=
  {r : ℚ | ∃ (abcdefgh : ℤ), 1 ≤ abcdefgh ∧ abcdefgh ≤ 99999999 ∧ r = abcdefgh / 99999999}

theorem sum_S_eq_50000000 : (∑ r in S, r) = 50000000 := by
  sorry

end sum_S_eq_50000000_l112_112840


namespace bisects_AC_implies_bisects_BD_l112_112938

/-- Given a convex quadrilateral ABCD with points P and Q being the midpoints of sides AB and CD respectively,
    and given that the line segment PQ bisects the diagonal AC, prove that PQ also bisects the diagonal BD. -/
theorem bisects_AC_implies_bisects_BD
    (A B C D P Q M N : Point)
    (hP : midpoint A B P)
    (hQ : midpoint C D Q)
    (hM : midpoint A C M)
    (hN : midpoint B D N)
    (hPQ_bisects_AC : lies_on_line M (line_through P Q))
    : lies_on_line N (line_through P Q) :=
sorry

end bisects_AC_implies_bisects_BD_l112_112938


namespace probability_last_box_contains_exactly_one_ball_l112_112098

-- Definitions and conditions
def num_boxes : ℕ := 100
def num_balls : ℕ := 100
def p : ℝ := 1 / num_boxes.toReal

-- To show: The probability that the last box contains exactly one ball
theorem probability_last_box_contains_exactly_one_ball :
  ((1 - p) ^ (num_boxes - 1)) ≈ 0.370 :=
by
  sorry

end probability_last_box_contains_exactly_one_ball_l112_112098


namespace find_smallest_x_l112_112426

noncomputable def smallest_x : ℝ :=
  min { x : ℝ | 0 < x ∧ (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * x = 8 }

theorem find_smallest_x :
  smallest_x = 89 / 9 :=
by
  sorry

end find_smallest_x_l112_112426


namespace KF_eq_LE_l112_112960

-- Definitions for the points and segments
variable (M N K L P F E : Type) [convex_polygon M N K L P] 
variable [segments N L (K N P) ∧ (K L M)] 
variable [segments K P (M K L) ∧ (N P L)] 
variable [diagonal_interests N P (M K) F] 
variable [diagonal_interests N P (M L) E]

-- The proof statement
theorem KF_eq_LE (conditions: convex_polygon M N K L P 
    ∧ segment_bisects N L (∠K N P) ∧ segment_bisects N L (∠K L M) 
    ∧ segment_bisects K P (∠M K L) ∧ segment_bisects K P (∠N P L)
    ∧ diagonal_interests N P (M K) F ∧ diagonal_interests N P (M L) E) : 
  K F = L E :=
sorry

end KF_eq_LE_l112_112960


namespace sum_first_n_terms_l112_112028

-- Definitions based on the conditions and question
def a_n (n : ℕ) : ℕ := 3 * n - 1
def b_n (n : ℕ) : ℕ := 3 ^ (n - 1)
def c_n (n : ℕ) : ℕ := a_n n * b_n n
def S_n (n : ℕ) : ℕ := ∑ i in finset.range n, c_n (i + 1)

-- Theorem statement
theorem sum_first_n_terms (n : ℕ) : S_n n = ((6 * n - 5) * 3^n + 5) / 4 := by
  sorry

end sum_first_n_terms_l112_112028


namespace closest_to_803_div_0p41_is_2000_l112_112368

theorem closest_to_803_div_0p41_is_2000 : 
  ∃ x, x ∈ ({100, 500, 1000, 2000, 4000} : Set ℕ) ∧
  |(803 / 0.41: ℝ) - x| = min (λ y, y ∈ ({100, 500, 1000, 2000, 4000} : Set ℕ), |(803 / 0.41: ℝ) - y|) :=
by
  sorry

end closest_to_803_div_0p41_is_2000_l112_112368


namespace find_numbers_with_conditions_l112_112826

theorem find_numbers_with_conditions (n : ℕ) (hn1 : n % 100 = 0) (hn2 : (n.divisors).card = 12) : 
  n = 200 ∨ n = 500 :=
by
  sorry

end find_numbers_with_conditions_l112_112826


namespace total_length_R_correct_l112_112642

def condition (x y : ℝ) : Prop :=
  (abs (abs x - 3) - 2) + (abs (abs y - 3) - 2) = 2

def R := {p : ℝ × ℝ | condition p.1 p.2}

noncomputable
def total_length_of_R_lines : ℝ := 32 * Real.sqrt 2

theorem total_length_R_correct :
  ∑ d in R, d = total_length_of_R_lines :=
  sorry

end total_length_R_correct_l112_112642


namespace driving_time_in_fog_is_correct_l112_112186

-- Define constants for speeds (in miles per minute)
def speed_sunny : ℚ := 35 / 60
def speed_rain : ℚ := 25 / 60
def speed_fog : ℚ := 15 / 60

-- Total distance and time
def total_distance : ℚ := 19.5
def total_time : ℚ := 45

-- Time variables for rain and fog
variables (t_r t_f : ℚ)

-- Define the driving distance equation
def distance_eq : Prop :=
  speed_sunny * (total_time - t_r - t_f) + speed_rain * t_r + speed_fog * t_f = total_distance

-- Prove the time driven in fog equals 10.25 minutes
theorem driving_time_in_fog_is_correct (h : distance_eq t_r t_f) : t_f = 10.25 :=
sorry

end driving_time_in_fog_is_correct_l112_112186


namespace find_x_eq_7714285714285714_l112_112383

theorem find_x_eq_7714285714285714 (x : ℝ) (hx_pos : 0 < x) (h : floor x * x = 54) : x = 54 / 7 :=
by
  sorry

end find_x_eq_7714285714285714_l112_112383


namespace cover_square_with_three_unit_squares_l112_112134

theorem cover_square_with_three_unit_squares :
  ∃ (squares : Fin 3 → (Point ℝ × ℝ)),
  (∀ i, 0 ≤ squares i).1 ∧
  (∀ i, squares i).1 ≤ 1 ∧
  (let covered_area :=
    ⋃ i, let (_corner, length) := squares i in
         { p : Point ℝ | p.x - _corner.x < length ∧ p.y - _corner.y < length } in
         (∃ corner : Point ℝ, ∀ point : Point ℝ,
          point - corner < 1 ↔ point ∈ covered_area)) := sorry

end cover_square_with_three_unit_squares_l112_112134


namespace circle_A_tangent_to_x_axis_l112_112124

open Function Real

-- Define the center and radius of circle A
def center_A : ℝ × ℝ := (-4, -3)
def radius_A : ℝ := 3

-- Prove that circle A is tangent to the x-axis
theorem circle_A_tangent_to_x_axis : dist (center_A.snd, 0) = radius_A := 
by 
  sorry

end circle_A_tangent_to_x_axis_l112_112124


namespace vn_2010_eq_12147_l112_112653

noncomputable def vn_sequence : ℕ → ℕ
| 1 := 2
| 2 := 5
| 3 := 10
| n := if n % 4 == 0 then 4 * (n / 4) + 2
       else if n % 4 == 1 then 4 * (n / 4) + 5
       else if n % 4 == 2 then 4 * (n / 4) + 10
       else 4 * (n / 4) + 11

def g (n : ℕ) : ℕ := 3 * n ^ 2 - 2 * n + 1

theorem vn_2010_eq_12147 : vn_sequence 2010 = 12147 :=
sorry

end vn_2010_eq_12147_l112_112653


namespace problem_1_problem_2_problem_3_problem_4_l112_112293

-- Problem (1)
theorem problem_1 (A B : set ℝ) (a : ℝ) :
  A = { x | x ^ 2 - 8 * x + 15 = 0 } →
  B = { x | a * x - 1 = 0 } →
  B ⊆ A →
  a ∈ ({0, (1/3), (1/5)} : set ℝ) :=
sorry

-- Problem (2)
theorem problem_2 (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = if x > 2 then a * x ^ 2 + x - 1 else -x + 1) →
  (∀ x y, x < y → f x > f y) ↔ a ≤ -1/4 :=
sorry

-- Problem (3)
theorem problem_3 (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →
  (∀ x, x > 0 → f x = 2 ^ x - 1) →
  f (f (-1)) = -1 :=
sorry

-- Problem (4)
theorem problem_4 (t α : ℝ) :
  (∀ t, ∃ x y, x = t * cos α ∧ y = t * sin α →
  (x + 6) ^ 2 + y ^ 2 = 25) →
  (∃ A B, |A - B| = √10) →
  (tan α = sqrt 15 / 3 ∨ tan α = - sqrt 15 / 3) :=
sorry

end problem_1_problem_2_problem_3_problem_4_l112_112293


namespace factor_difference_of_squares_l112_112369

theorem factor_difference_of_squares (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := 
by
  sorry

end factor_difference_of_squares_l112_112369


namespace find_parabola_equation_find_line_m_equation_l112_112007

noncomputable theory
open_locale big_operators

def parabola_vertex_origin_focus_pos_yaxis (x y : ℝ) : Prop :=
  x^2 = 4 * y

def line_through_focus_intersects_parabola (f : ℝ) :=
  ∃ A B : ℝ × ℝ,
    (parabola_vertex_origin_focus_pos_yaxis A.1 A.2) ∧
    (parabola_vertex_origin_focus_pos_yaxis B.1 B.2) ∧
    ∥A - B∥ = 8 ∧
    (A.2 + B.2) / 2 = 3

theorem find_parabola_equation :
  ∀ x y : ℝ, line_through_focus_intersects_parabola (0, 2) → parabola_vertex_origin_focus_pos_yaxis x y :=
sorry

theorem find_line_m_equation :
  ∀ (P Q R : ℝ × ℝ), 
  line_through_focus_intersects_parabola (0, 2) →
  (∃ k : ℝ, Q.2 = k * Q.1 + 6 ∧ parabola_vertex_origin_focus_pos_yaxis P.1 P.2 ∧ parabola_vertex_origin_focus_pos_yaxis Q.1 Q.2 ∧ PR_tangent_to_parabola P R) → 
  m_equation k :=
sorry

end find_parabola_equation_find_line_m_equation_l112_112007


namespace sum_arithmetic_sequences_l112_112805

theorem sum_arithmetic_sequences (n : ℕ) : 
  (∑ k in Finset.range (n + 1), 3 * k + 2) + (∑ k in Finset.range (n + 1), 3 * k) = (n + 1) * (3 * n + 2) := 
by 
  sorry

end sum_arithmetic_sequences_l112_112805


namespace proportion_dogs_proof_l112_112541

variables (C G : ℕ)

-- Define the conditions given in the problem
def cats_think_dogs := 0.2 * G
def dogs_think_cats := 0.25 * C
def total_think_cats := 0.3 * (G + C)

-- Define the proportion of animals
def proportion_dogs := C / (C + G)

-- The actual theorem to prove the statement
theorem proportion_dogs_proof
  (h1 : cats_think_dogs = 0.2 * G)
  (h2 : dogs_think_cats = 0.25 * C)
  (h3 : 0.8 * G + 0.25 * C = 0.3 * (G + C)) :
  proportion_dogs = 10 / 11 :=
by sorry

end proportion_dogs_proof_l112_112541


namespace geometric_sequence_a_5_l112_112854

noncomputable def a_n : ℕ → ℝ := sorry

theorem geometric_sequence_a_5 :
  (∀ n : ℕ, ∃ r : ℝ, a_n (n + 1) = r * a_n n) →  -- geometric sequence property
  (∃ x₁ x₂ : ℝ, x₁ + x₂ = -7 ∧ x₁ * x₂ = 9 ∧ a_n 3 = x₁ ∧ a_n 7 = x₂) →  -- roots of the quadratic equation and their assignments
  a_n 5 = -3 := sorry

end geometric_sequence_a_5_l112_112854


namespace incorrect_statement_C_l112_112752

theorem incorrect_statement_C (a b : ℝ) : 
  (∀ (a b c : ℝ), log a b * log b c * log c a = 1) →
  (∀ (x : ℝ), (f : ℝ → ℝ), f x = Real.exp x → ∀ a b : ℝ, f (a + b) = f a * f b) →
  (∀ (x : ℝ), (f : ℝ → ℝ), f x = Real.exp x → (∀ a b : ℝ, f (a * b) = f a * f b) → False) →
  (∀ (x : ℝ), (x * log 3 4 = 1 → 4 ^ x + 4 ^ (-x) = (10:ℝ) / 3)) →
  False :=
by 
  intros Hlog HexpAdd HexpMul Hlog4
  exact HexpMul _ _ rfl

end incorrect_statement_C_l112_112752


namespace fencing_rate_l112_112377

noncomputable def rate_per_meter (d : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := Real.pi * d
  total_cost / circumference

theorem fencing_rate (diameter cost : ℝ) (h₀ : diameter = 34) (h₁ : cost = 213.63) :
  rate_per_meter diameter cost = 2 := by
  sorry

end fencing_rate_l112_112377


namespace part1_part2_l112_112998

section
variable (x a : ℝ)
def p (x a : ℝ) : Prop := (x - 3 * a) * (x - a) < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem part1 (h : a = 1) (hq : q x) (hp : p x a) : 2 < x ∧ x < 3 := by
  sorry

theorem part2 (h : ∀ x, q x → p x a) : 1 ≤ a ∧ a ≤ 2 := by
  sorry
end

end part1_part2_l112_112998


namespace student_marks_for_correct_answers_l112_112537

theorem student_marks_for_correct_answers
  (total_questions : ℕ)
  (total_marks : ℕ)
  (correct_questions : ℕ)
  (marks_lost_per_wrong_answer : ℕ)
  (correct_marks_per_answer : ℕ) :
  total_questions = 120 →
  total_marks = 180 →
  correct_questions = 75 →
  marks_lost_per_wrong_answer = 1 →
  total_marks = correct_questions * correct_marks_per_answer - (total_questions - correct_questions) * marks_lost_per_wrong_answer →
  correct_marks_per_answer = 3 :=
by
  intros h_total_questions h_total_marks h_correct_questions h_marks_lost h_total_marks_equation
  rw [h_total_questions, h_total_marks, h_correct_questions, h_marks_lost] at h_total_marks_equation
  linarith
  done

end student_marks_for_correct_answers_l112_112537


namespace initial_percentage_milk_l112_112708

theorem initial_percentage_milk (P : ℝ) (h1 : 60 > 0) (h2 : 40.8 > 0):
  let initial_milk_volume := (P / 100) * 60
  let added_water_volume := 40.8
  let final_solution_volume := 60 + added_water_volume
  let final_milk_volume := (50 / 100) * final_solution_volume
  initial_milk_volume = final_milk_volume → P = 84 :=
by
  intros
  let initial_milk_volume := (P / 100) * 60
  let final_solution_volume := 60 + 40.8
  let final_milk_volume := (50 / 100) * final_solution_volume
  have : initial_milk_volume = final_milk_volume := by assumption
  sorry

end initial_percentage_milk_l112_112708


namespace sarah_total_height_in_cm_l112_112678

def sarah_height_in_inches : ℝ := 54
def book_thickness_in_inches : ℝ := 2
def conversion_factor : ℝ := 2.54

def total_height_in_inches : ℝ := sarah_height_in_inches + book_thickness_in_inches
def total_height_in_cm : ℝ := total_height_in_inches * conversion_factor

theorem sarah_total_height_in_cm : total_height_in_cm = 142.2 :=
by
  -- Skip the proof for now
  sorry

end sarah_total_height_in_cm_l112_112678


namespace increased_numerator_value_l112_112300

theorem increased_numerator_value (x y a : ℝ) (h1 : x / y = 2 / 5) (h2 : (x + a) / (2 * y) = 1 / 3) (h3 : x + y = 5.25) : a = 1 :=
by
  -- skipped proof: sorry
  sorry

end increased_numerator_value_l112_112300


namespace households_with_car_l112_112117

theorem households_with_car {H_total H_neither H_both H_bike_only : ℕ} 
    (cond1 : H_total = 90)
    (cond2 : H_neither = 11)
    (cond3 : H_both = 22)
    (cond4 : H_bike_only = 35) : 
    H_total - H_neither - (H_bike_only + H_both - H_both) + H_both = 44 := by
  sorry

end households_with_car_l112_112117


namespace polar_center_of_circle_l112_112545

theorem polar_center_of_circle (rho theta : ℝ) (h : rho = sin theta) : 
  (rho, theta) = (1/2, π/2) := 
sorry

end polar_center_of_circle_l112_112545


namespace integer_solutions_l112_112371

theorem integer_solutions (x y : ℤ) : x^4 - 6 * x^2 + 1 = 7 * 2^y ↔ (x = 3 ∧ y = 2) ∨ (x = -3 ∧ y = 2) := 
sorry

end integer_solutions_l112_112371


namespace range_of_a_l112_112042

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x + y = 2 ∧ 
    (if x > 1 then (x^2 + 1) / x else Real.log (x + a)) = 
    (if y > 1 then (y^2 + 1) / y else Real.log (y + a))) ↔ 
    a > Real.exp 2 - 1 :=
by sorry

end range_of_a_l112_112042


namespace no_valid_k_for_prime_roots_l112_112348

theorem no_valid_k_for_prime_roots :
  ∀ (p q : ℕ), prime p → prime q → p + q = 67 → ∃ k : ℕ, k = p * q → false :=
by
  sorry

end no_valid_k_for_prime_roots_l112_112348


namespace train_speed_l112_112767

theorem train_speed (train_length : ℝ) (cross_time : ℝ) (man_speed_kmph : ℝ) :
  train_length = 500 →
  cross_time = 29.997600191984642 →
  man_speed_kmph = 3 →
  (let train_speed := (train_length / cross_time + ((man_speed_kmph * 1000) / 3600)) * 3.6  in train_speed = 63) :=
begin
  intros h1 h2 h3,
  let man_speed_mps := man_speed_kmph * 1000 / 3600,
  let relative_speed_mps := train_length / cross_time,
  let train_speed_mps := relative_speed_mps + man_speed_mps,
  let train_speed_kmph := train_speed_mps * 3.6,
  have h : train_speed_kmph = 63,
  { sorry },
  exact h,
end

end train_speed_l112_112767


namespace charles_picked_50_pears_l112_112226

variable (P B S : ℕ)

theorem charles_picked_50_pears 
  (cond1 : S = B + 10)
  (cond2 : B = 3 * P)
  (cond3 : S = 160) : 
  P = 50 := by
  sorry

end charles_picked_50_pears_l112_112226


namespace cost_keyboard_l112_112312

def num_keyboards : ℕ := 15
def num_printers : ℕ := 25
def total_cost : ℝ := 2050
def cost_printer : ℝ := 70
def total_cost_printers : ℝ := num_printers * cost_printer
def total_cost_keyboards : ℝ := total_cost - total_cost_printers

theorem cost_keyboard : total_cost_keyboards / num_keyboards = 20 := by
  sorry

end cost_keyboard_l112_112312


namespace part1_part2_l112_112627

theorem part1 (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : A = 2 * π / 3 :=
sorry

theorem part2 (b c : ℝ) (h1 : sin (2 * π / 3) ^ 2 - sin b ^ 2 - sin c ^ 2 = sin b * sin c)
  (h2 : b + c = 2 * sqrt 3) : 3 * 2 + b + c = 3 + 2 * sqrt 3 :=
sorry

end part1_part2_l112_112627


namespace max_ab_value_l112_112839

def floor (x : ℝ) : ℤ := ⌊x⌋₊

theorem max_ab_value (a b : ℝ) (h_add_inv : floor a + floor b = 0) : a + b = 2 := by
  sorry

end max_ab_value_l112_112839


namespace PQ_bisects_BD_l112_112940

variable (A B C D P Q : Type) [Add A] (M : A) [Div A Two]

theorem PQ_bisects_BD
  (h1 : convex_quadrilateral A B C D)
  (h2 : midpoint A B P)
  (h3 : midpoint C D Q)
  (h4 : bisects P Q A C) :
  bisects P Q B D :=
sorry

end PQ_bisects_BD_l112_112940


namespace polynomial_evaluation_eq_neg2_l112_112250

noncomputable def poly : ℚ[X] := X^4 - 4*X^2 + 1

theorem polynomial_evaluation_eq_neg2 :
  (∃ Q : ℚ[X], degree Q = 4 ∧ Q.leading_coeff = 1 ∧ root_in_poly Q (sqrt 2 + sqrt 3)) →
  eval 1 poly = -2 :=
by
  intro h
  sorry

/-- Check if a given number is a root of a polynomial -/
def root_in_poly (Q : ℚ[X]) (r : ℚ) : Prop :=
  Q.eval r = 0


end polynomial_evaluation_eq_neg2_l112_112250


namespace find_a1_l112_112242

theorem find_a1 (S : ℕ → ℝ) (a : ℕ → ℝ) (a1 : ℝ) :
  (∀ n : ℕ, S n = a1 * (2^n - 1)) → a 4 = 24 → 
  a 4 = S 4 - S 3 → 
  a1 = 3 :=
by
  sorry

end find_a1_l112_112242


namespace coefficient_term_largest_binomial_coefficient_l112_112962

theorem coefficient_term_largest_binomial_coefficient :
  let a := (1 / 2 : ℝ)
  let b := (2 : ℝ)
  let n := 10
  let r := 5
  binomial n r * b^r * a^(n-r) = 252 :=
by
  sorry

end coefficient_term_largest_binomial_coefficient_l112_112962


namespace part1_part2_l112_112620

theorem part1 (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : A = 2 * π / 3 :=
sorry

theorem part2 (b c : ℝ) (h1 : sin (2 * π / 3) ^ 2 - sin b ^ 2 - sin c ^ 2 = sin b * sin c)
  (h2 : b + c = 2 * sqrt 3) : 3 * 2 + b + c = 3 + 2 * sqrt 3 :=
sorry

end part1_part2_l112_112620


namespace slope_of_tangent_line_l112_112870

theorem slope_of_tangent_line : ∃ k : ℝ, (∀ f : ℝ → ℝ, (∀ x, f x = Real.exp x) → 
  ∃ x0 : ℝ, k = Real.exp x0 ∧ f x0 = x0 * k ∧ (0, 0) ∈ {(x0, Real.exp x0)} ∧ k = Real.exp x0) ∧ k = Real.exp 1 :=
begin
  sorry
end

end slope_of_tangent_line_l112_112870


namespace PQ_bisects_BD_l112_112943

variable (A B C D P Q : Type) [Add A] (M : A) [Div A Two]

theorem PQ_bisects_BD
  (h1 : convex_quadrilateral A B C D)
  (h2 : midpoint A B P)
  (h3 : midpoint C D Q)
  (h4 : bisects P Q A C) :
  bisects P Q B D :=
sorry

end PQ_bisects_BD_l112_112943


namespace common_ratio_of_arithmetic_geometric_sequence_l112_112014

noncomputable def a_n (n : ℕ) : ℝ := sorry -- Replace with the actual sequence function
noncomputable def S_n (n : ℕ) : ℝ := sorry -- Replace with the actual sum function

theorem common_ratio_of_arithmetic_geometric_sequence : 
  (a_n 3 = 2 * S_n 2 + 1) → (a_n 4 = 2 * S_n 3 + 1) → 
  (∃ q : ℝ, q = 3) :=
by 
  intros h1 h2
  use 3
  -- Further steps skipped with sorry; proofs need to be filled in
  sorry

end common_ratio_of_arithmetic_geometric_sequence_l112_112014


namespace number_of_integer_solutions_l112_112433

theorem number_of_integer_solutions (a b : ℤ) :
  (∃! (a b : ℤ), 2 ^ (2 * a) - 3 ^ (2 * b) = 55) := sorry

end number_of_integer_solutions_l112_112433


namespace angle_B_range_l112_112130

theorem angle_B_range (A B C : Type)
  [Triangle A B C]
  (angle_A : A.angle = 58)
  (length_AB : A.B > B.C) :
  0 < B.angle ∧ B.angle < 64 := by
  sorry

end angle_B_range_l112_112130


namespace last_box_probability_l112_112085

noncomputable def probability_last_box_only_ball : ℝ :=
  let n : ℕ := 100 in
  let p : ℝ := 1 / n in
  (n * p * (1 - p)^(n - 1))

theorem last_box_probability : abs (probability_last_box_only_ball - 0.3697) < 0.0005 := 
  sorry

end last_box_probability_l112_112085


namespace total_votes_l112_112119

variable (V : ℝ)

theorem total_votes (h : 0.70 * V - 0.30 * V = 160) : V = 400 := by
  sorry

end total_votes_l112_112119


namespace equal_production_l112_112721

variables (r x : ℝ)
variables (factoryA factoryB : Type)

-- Define the initial production rates
def production_rate_B_initial : ℝ := r
def production_rate_A_initial : ℝ := (4/3) * r

-- Define the total days worked
def total_days : ℝ := 6

-- Conditions on the days worked
def days_before_adjustment : ℝ := 5 - x
def days_after_adjustment : ℝ := x

-- Production calculations
def production_B_before_adjustment : ℝ := production_rate_B_initial * days_before_adjustment
def production_A_total : ℝ := production_rate_A_initial * total_days
def production_B_after_adjustment : ℝ := 2 * production_rate_B_initial * days_after_adjustment

-- Total production by B
def production_B_total : ℝ := production_B_before_adjustment + production_B_after_adjustment

-- Prove that total production by A equals total production by B
theorem equal_production : production_B_total = production_A_total → x = 3 := by
  sorry

end equal_production_l112_112721


namespace coefficient_of_x2_in_expansion_l112_112543

noncomputable def coefficient_x2 (n : ℕ) : ℕ :=
  @binomial n 2 * 9

theorem coefficient_of_x2_in_expansion :
  let n := 5,
      sum_of_binomial_coeff := 2^n,
      sum_of_coefficients := 4^n,
      ratio := sum_of_coefficients / sum_of_binomial_coeff
  in ratio = 32 -> coefficient_x2 n = 90 :=
by
  intros
  have : n = 5 := by sorry
  rw this
  exact rfl

end coefficient_of_x2_in_expansion_l112_112543


namespace angle_AOB_is_120_l112_112717

theorem angle_AOB_is_120
  (P A B O : Type) -- P, A, B, O are points on the circle O
  (PA PB secants : Boolean) -- PA and PB are secants
  (circle O) (angle APB : Real) (APB_60_deg : angle APB = 60) :
  ∃ angle (AOB : Real), AOB = 120 := 
by 
  sorry

end angle_AOB_is_120_l112_112717


namespace turtle_ran_while_rabbit_sleeping_l112_112334

-- Define the constants and variables used in the problem
def total_distance : ℕ := 1000
def rabbit_speed_multiple : ℕ := 5
def rabbit_behind_distance : ℕ := 10

-- Define a function that represents the turtle's distance run while the rabbit is sleeping
def turtle_distance_while_rabbit_sleeping (total_distance : ℕ) (rabbit_speed_multiple : ℕ) (rabbit_behind_distance : ℕ) : ℕ :=
  total_distance - total_distance / (rabbit_speed_multiple + 1)

-- Prove that the turtle ran 802 meters while the rabbit was sleeping
theorem turtle_ran_while_rabbit_sleeping :
  turtle_distance_while_rabbit_sleeping total_distance rabbit_speed_multiple rabbit_behind_distance = 802 :=
by
  -- We reserve the proof and focus only on the statement
  sorry

end turtle_ran_while_rabbit_sleeping_l112_112334


namespace smallest_solution_l112_112742

def polynomial (x : ℝ) := x^4 - 34 * x^2 + 225 = 0

theorem smallest_solution : ∃ x : ℝ, polynomial x ∧ ∀ y : ℝ, polynomial y → x ≤ y := 
sorry

end smallest_solution_l112_112742


namespace minimum_ab_l112_112863

noncomputable theory

open Real

theorem minimum_ab (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : (a:ℝ) * a * b = (a^2 + 1) / 4) : (a:ℝ) * b ≥ 1 / 2 :=
by
  sorry

end minimum_ab_l112_112863


namespace triangle_theorem_l112_112607

theorem triangle_theorem 
  (A B C : ℝ) 
  (h1 : ∀ A B C : ℝ, ∃ (a b c : ℝ), 
      a^2 - b^2 - c^2 = b * c ∧ sin^2 A - sin^2 B - sin^2 C = sin B * sin C) 
  (h2 : 0 < A ∧ A < π) : 
  (A = 2 * π / 3) ∧ 
  (∀ (BC : ℝ), BC = 3 → ∃ (a b c : ℝ), 
      a + b + c ≤ 3 + 2 * sqrt 3) :=
by
  sorry

end triangle_theorem_l112_112607


namespace find_angle_A_max_perimeter_l112_112562

noncomputable def sin_sq_minus (A B C : ℝ) : ℝ :=
  (Real.sin A) * (Real.sin A) - (Real.sin B) * (Real.sin B) - (Real.sin C) * (Real.sin C)

noncomputable def sin_prod (B C : ℝ) : ℝ :=
  (Real.sin B) * (Real.sin C)

theorem find_angle_A (A B C : ℝ) (h : sin_sq_minus A B C = sin_prod B C) :
  A = 2 * Real.pi / 3 :=
by
  sorry

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem max_perimeter (B C : ℝ) (a : ℝ) (h1 : a = 3) (h2 : B + C = Real.pi / 3) :
  ∃ (b c : ℝ), perimeter a b c = 3 + 2 * Real.sqrt 3 :=
by
  sorry

end find_angle_A_max_perimeter_l112_112562


namespace sum_x_coordinates_eq_neg_one_sixth_l112_112213

def g (x : ℝ) : ℝ :=
if x < -3 then (3 / 2) * x + 15 / 2
else if x < -1 then undefined
else if x < 1 then 2 * x + 1
else if x < 3 then undefined
else if x < 5 then 2 * x - 4
else undefined

theorem sum_x_coordinates_eq_neg_one_sixth :
  let x1 := -11 / 3
  let x2 := 1 / 2
  let x3 := 3 in
  x1 + x2 + x3 = -1 / 6 :=
by
  sorry

end sum_x_coordinates_eq_neg_one_sixth_l112_112213


namespace decreasing_interval_of_f_l112_112657

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := sqrt 3 * sin (ω * x) - cos (ω * x)

theorem decreasing_interval_of_f (ω : ℝ) (h_ω_pos : ω > 0) (h_period : min_period (f ω) = π) :
  ∃ I : set ℝ, I = set.Ioo (π / 3) (5 * π / 6) ∧ 
  ∀ x y ∈ I, x < y → f ω x > f ω y := 
sorry

end decreasing_interval_of_f_l112_112657


namespace solution_trig_problem_l112_112029

noncomputable def trig_problem (α : ℝ) : Prop :=
  (cos (2 * α) = -3 / 5) →
  (π < α ∧ α < 3 * π / 2) →
  tan (π / 4 + 2 * α) = -1 / 7

theorem solution_trig_problem {α : ℝ} (h1 : cos (2 * α) = -3 / 5) (h2 : π < α ∧ α < 3 * π / 2) :
  tan (π / 4 + 2 * α) = -1 / 7 :=
sorry

end solution_trig_problem_l112_112029


namespace sum_of_favorite_numbers_is_600_l112_112167

def GloryFavoriteNumber : ℕ := 450
def MistyFavoriteNumber (G : ℕ) : ℕ := G / 3

theorem sum_of_favorite_numbers_is_600 (G : ℕ) (hG : G = GloryFavoriteNumber) :
  MistyFavoriteNumber G + G = 600 :=
by
  rw [hG]
  simp [GloryFavoriteNumber, MistyFavoriteNumber]
  -- Proof is omitted (filled with sorry)
  sorry

end sum_of_favorite_numbers_is_600_l112_112167


namespace relationship_between_distances_l112_112802

noncomputable def P := (0, 0, 1 : ℝ)
noncomputable def D := (0, 0, 0 : ℝ)
noncomputable def A := (1, 0, 0 : ℝ)
noncomputable def B := (1, 1, 0 : ℝ)
noncomputable def C := (0, 1, 0 : ℝ)

def plane_PAB : ℝ × ℝ × ℝ → ℝ :=
  λ (x y z : ℝ) => x + y - 2*z

def plane_PAC : ℝ × ℝ × ℝ → ℝ :=
  λ (x y z : ℝ) => x - y + z

def distance_to_plane (P: ℝ × ℝ × ℝ) (plane: ℝ × ℝ × ℝ → ℝ) : ℝ :=
  let (x₀, y₀, z₀) := P
  let d := plane x₀ y₀ z₀
  let norm := real.sqrt (1^2 + 1^2 + (-2)^2)
  real.abs d / norm

noncomputable def d1 := distance_to_plane C plane_PAB
noncomputable def d2 := distance_to_plane B plane_PAC

theorem relationship_between_distances : d2 < d1 ∧ d1 < 1 :=
by 
  unfold d1 d2 distance_to_plane plane_PAB plane_PAC
  simp
  sorry

end relationship_between_distances_l112_112802


namespace find_A_max_perimeter_of_triangle_l112_112573

-- Definition of the given problem conditions
def triangle_condition (A B C : ℝ) : Prop :=
  sin(A)^2 - sin(B)^2 - sin(C)^2 = sin(B) * sin(C)

-- (1) Proving the value of A given the condition
theorem find_A (A B C : ℝ) (h : triangle_condition A B C) : A = 2 * π / 3 :=
by sorry

-- (2) Proving the maximum perimeter given BC = 3 and A = 2π/3
theorem max_perimeter_of_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (hA : 2 * π / 3 = 2 * π / 3) : 
  ∃ (P : ℝ), P = 3 + 2 * sqrt 3 :=
by sorry

end find_A_max_perimeter_of_triangle_l112_112573


namespace length_SP_eq_zero_l112_112015

noncomputable def IsoscelesTrapezoid (P Q R S : Point) : Prop := 
  (distance P Q = 7) ∧ 
  (distance R S = 15) ∧ 
  (distance P R = 9) ∧ 
  (distance Q S = 9) ∧ 
  ∃ (T : Point), collinear R T S ∧ 
                 right_triangle R T P ∧ 
                 midpoint Q R T

theorem length_SP_eq_zero (P Q R S T : Point) (h : IsoscelesTrapezoid P Q R S) : 
  distance S P = 0 :=
sorry

end length_SP_eq_zero_l112_112015


namespace part_1_part_2_l112_112584

-- Define the triangle and the given condition
variables {α β γ : ℝ}
axiom triangle_ABC : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π
axiom sin_identity : sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)
axiom BC_length : (sin(γ) / sin(α)) * BC = 3

-- State the main theorem parts separately
theorem part_1 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_sin_identity: sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)) :
  α = 2 * π / 3 :=
sorry

theorem part_2 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_α: α = 2 * π / 3) (h_BC_length: BC = 3) :
  let b := (2 * sqrt(3) * sin(π/6 - β)),
      c := (2 * sqrt(3) * sin(π/6 + β)) in
  (3 + 2 * sqrt(3)) :=
sorry

end part_1_part_2_l112_112584


namespace probability_sum_6_probability_product_divisible_by_3_l112_112787

/-- A right tetrahedron block has the numbers 1, 2, 3, and 4 written on its four faces.
When three such tetrahedron blocks are tossed onto a table, let the numbers on the faces
touching the table be x, y, and z. Prove the probabilities: -/
namespace tetrahedron_problem

def faces : set ℕ := {1, 2, 3, 4}

def valid_values_for_xy_and_z (x y z : ℕ) : Prop :=
  x ∈ faces ∧ y ∈ faces ∧ z ∈ faces

def event_D (x y z : ℕ) : Prop :=
  x + y + z = 6

def event_E (x y z : ℕ) : Prop :=
  (x * y * z) % 3 = 0

theorem probability_sum_6 (x y z : ℕ) (h : valid_values_for_xy_and_z x y z) :
  (P (event_D x y z) = 5 / 32) :=
sorry

theorem probability_product_divisible_by_3 (x y z : ℕ) (h : valid_values_for_xy_and_z x y z) :
  (P (event_E x y z) = 37 / 64) :=
sorry

end tetrahedron_problem

end probability_sum_6_probability_product_divisible_by_3_l112_112787


namespace probability_ball_in_last_box_is_approx_0_l112_112106

/-- Given 100 boxes and 100 balls randomly distributed,
the probability that the last box will contain the only ball is approximately 0.370. -/
theorem probability_ball_in_last_box_is_approx_0.370 :
  let n := 100,
      p := 1 / n,
      probability : ℝ := (99 / 100)^99
  in abs (probability - 0.370) < 0.001 :=
by {
  let n := 100,
  let p := 1 / n,
  let probability := (99 / 100)^99,
  sorry
}

end probability_ball_in_last_box_is_approx_0_l112_112106


namespace hexagon_area_excluding_inner_triangle_l112_112724

theorem hexagon_area_excluding_inner_triangle :
  let base := 2
  let height := 4
  let num_triangles := 6
  let outer_hexagon_area := num_triangles * (1/2) * base * height
  let side_length := 4
  let inner_triangle_area := (sqrt 3 / 4) * side_length^2
  outer_hexagon_area - inner_triangle_area = 24 - 4 * sqrt 3 := by
  sorry

end hexagon_area_excluding_inner_triangle_l112_112724


namespace conveyor_belt_sampling_l112_112345

noncomputable def sampling_method (interval : ℕ) (total_items : ℕ) : String :=
  if interval = 5 ∧ total_items > 0 then "systematic sampling" else "unknown"

theorem conveyor_belt_sampling :
  ∀ (interval : ℕ) (total_items : ℕ),
  interval = 5 ∧ total_items > 0 →
  sampling_method interval total_items = "systematic sampling" :=
sorry

end conveyor_belt_sampling_l112_112345


namespace math_problem_l112_112729

theorem math_problem (a b n r : ℕ) (h₁ : 1853 ≡ 53 [MOD 600]) (h₂ : 2101 ≡ 101 [MOD 600]) :
  (1853 * 2101) ≡ 553 [MOD 600] := by
  sorry

end math_problem_l112_112729


namespace dilation_image_l112_112773

theorem dilation_image :
  ∀ (z₀ z w : ℂ) (k : ℝ), z₀ = 2 - 3 * complex.I → z = -1 + 2 * complex.I →
  k = 3 → w = z₀ + k * (z - z₀) → w = -7 + 12 * complex.I :=
begin
  intros z₀ z w k h₁ h₂ h₃ h₄,
  rw [←h₁, ←h₂, ←h₃] at h₄,
  sorry,
end

end dilation_image_l112_112773


namespace trig_identity_l112_112761

theorem trig_identity :
  sin 200 * cos 110 + cos 160 * sin 70 = -1 := 
by
  sorry

end trig_identity_l112_112761


namespace angle_bisector_exists_l112_112991

variables {t : ℝ}
def vec_a : ℝ × ℝ × ℝ := (5, -3, -6)
def vec_c : ℝ × ℝ × ℝ := (-3, 0, 3)

noncomputable def vec_b : ℝ × ℝ × ℝ := 
  ((vec_a.1 + t * (vec_c.1 - vec_a.1)),
   (vec_a.2 + t * (vec_c.2 - vec_a.2)),
   (vec_a.3 + t * (vec_c.3 - vec_a.3)))

def collinear (u v : ℝ × ℝ × ℝ) : Prop := 
  ∃ k : ℝ, v = (u.1 * k, u.2 * k, u.3 * k)

def cosine (u v : ℝ × ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2 + u.3 * v.3) /
  (real.sqrt (u.1^2 + u.2^2 + u.3^2) * real.sqrt (v.1^2 + v.2^2 + v.3^2))

theorem angle_bisector_exists :
  ∃ t : ℝ, collinear vec_a vec_b ∧ collinear vec_b vec_c ∧ 
  (cosine vec_a vec_b) = (cosine vec_b vec_c) :=
sorry

end angle_bisector_exists_l112_112991


namespace curve_cartesian_equation_line_intersects_curve_range_l112_112128

-- Part (1): Cartesian equation of curve C
theorem curve_cartesian_equation (t : ℝ) (x y : ℝ) (htx : x = 2 * sqrt t / (1 + t)) (hty : y = 2 / (1 + t)) :
  x^2 + y^2 - 2 * y = 0 :=
sorry

-- Part (2): Range of values for m for the intersection of line l and curve C
theorem line_intersects_curve_range (m : ℝ) :
  -sqrt 2 + 1 ≤ m ∧ m ≤ 2 :=
sorry

end curve_cartesian_equation_line_intersects_curve_range_l112_112128


namespace x_n_integer_for_all_n_l112_112237

def x (n : ℕ) : ℕ
| 1     := 2
| (n+2) := 2 * (2 * (n + 2) - 1) * x (n + 1) / (n + 2)

theorem x_n_integer_for_all_n (n : ℕ) (h1 : n ≥ 1) : ∃ x_n : ℕ, x n = x_n :=
by sorry

end x_n_integer_for_all_n_l112_112237


namespace power_equality_l112_112266

theorem power_equality : 
  ( (11 : ℝ) ^ (1 / 5) / (11 : ℝ) ^ (1 / 7) ) = (11 : ℝ) ^ (2 / 35) := 
by sorry

end power_equality_l112_112266


namespace disks_sum_area_equals_47_l112_112191

noncomputable def total_area_expression (r : ℝ) (σ : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * π * r ^ 2

def pi_expansion_of_disks (a b c : ℕ) : ℝ :=
  π * (a - b * σ^c)

theorem disks_sum_area_equals_47:
  ∃ (a b c : ℕ), let r := (1 - sqrt 3 / 2) in
    (π * ((16 * r) ^ 2 / 16 * π) = π * (a - b * sqrt c)) ∧  
    c ≠ 0 ∧ 
    c % 4 ≠ 0 ∧
    a + b + c = 47 := 
by 
  sorry

end disks_sum_area_equals_47_l112_112191


namespace extreme_points_on_interval_l112_112885

noncomputable def f (x θ : ℝ) : ℝ := (Real.sin (2 * x) * Real.cos θ) + (Real.sin θ) - (2 * (Real.sin x)^2 * Real.sin θ)

def symmetric_about (θ : ℝ) : Prop := ∀ x y, f x θ = f y θ → (x = π / 3 - y ∨ x = y + 2 * π / 3)

theorem extreme_points_on_interval
  {θ : ℝ}
  (h1 : -π / 2 < θ)
  (h2 : θ < 0)
  (h3 : symmetric_about θ) :
  ∃ n, n = 4 ∧ ∀ x, 0 < x → x < 2 * π → (∃ x1 x2, is_extreme_point (f x1 θ) ∧ is_extreme_point (f x2 θ)) :=
sorry

end extreme_points_on_interval_l112_112885


namespace distinct_real_det_eq_zero_l112_112150

theorem distinct_real_det_eq_zero {a b : ℝ} (h₁ : a ≠ b) (h₂ : det ![![2, 5, 10], ![4, a, b], ![4, b, a]] = 0) : a + b = 30 :=
sorry

end distinct_real_det_eq_zero_l112_112150


namespace num_four_digit_numbers_div_by_25_l112_112544

theorem num_four_digit_numbers_div_by_25 : 
  let digits := {0, 1, 2, 3, 4, 5}
  finset.card {n : ℕ | 
    (∀ d ∈ digits, d ∈ digits) ∧ 
    n / 1000 > 0 ∧
    (∀ i < 10, i ∈ digits → (n % 10^i) / 10^(i-1) ≠ (n % 10^(i-1))) ∧
    (n % 25 = 0) ∧
    (10^3 ≤ n) ∧ (n < 10^4)} = 24 :=
by
  sorry

end num_four_digit_numbers_div_by_25_l112_112544


namespace find_m_n_inequality_l112_112886

-- Defining the function f(x)
def f (x : ℝ) (m n : ℝ) : ℝ := Real.exp x + m * x^3 + n * x^2 - x - 1

-- Given condition: Tangent line at x = 1 is y = ex
def tangent_condition (m n : ℝ) : Prop := 
  f 1 m n = Real.exp 1 ∧ (Real.exp 1 + 3 * m * 1^2 + 2 * n * 1 - 1) = Real.exp 1

-- Statement to prove m and n
theorem find_m_n : ∃ (m n : ℝ), tangent_condition m n ∧ m = -3 ∧ n = 5 :=
sorry

-- Statement to prove inequality for any x in ℝ
theorem inequality (x : ℝ) : 
  (f x (-3) 5) + Real.exp x * x^3 - 2 * Real.exp x * x^2 ≥ 0 :=
sorry

end find_m_n_inequality_l112_112886


namespace f_of_zero_l112_112479

def f (n : ℕ) : ℤ := sorry

theorem f_of_zero :
  (∀ n : ℕ, f (f n) + f n = 2 * n + 3) →
  f 2016 = 2017 →
  f 0 = -2016 :=
by
  intros h1 h2
  sorry

end f_of_zero_l112_112479


namespace vector_c_combination_l112_112058

noncomputable def vec := (ℝ × ℝ)

def a : vec := (1, 1)
def b : vec := (-1, 1)
def c : vec := (4, 2)

theorem vector_c_combination (a b c : vec) (ha : a = (1, 1)) (hb : b = (-1, 1)) (hc : c = (4, 2)) :
  c = 3 * a - b :=
sorry

end vector_c_combination_l112_112058


namespace probability_last_box_single_ball_l112_112090

noncomputable def probability_last_box_only_ball (n : ℕ) : ℝ :=
  let p := 1 / n in
  (n * p * (1 - p)^(n - 1))

theorem probability_last_box_single_ball :
  probability_last_box_only_ball 100 ≈ 0.370 :=
by
  sorry

end probability_last_box_single_ball_l112_112090


namespace range_of_a_for_meaningful_function_l112_112491

noncomputable def isMeaningfulOverInterval (a : ℝ) : Prop :=
  ∀ x : ℝ, (x ≥ -1) → (1 - a * x ≥ 0)

theorem range_of_a_for_meaningful_function :
  { a : ℝ | isMeaningfulOverInterval a } = set.Icc (-1:ℝ) 0 :=
sorry

end range_of_a_for_meaningful_function_l112_112491


namespace last_box_one_ball_probability_l112_112103

/-- The probability that the last box will contain exactly one of 100 randomly distributed balls
is approximately 0.370. -/
theorem last_box_one_ball_probability :
  let n : ℕ := 100 in
  let p : ℚ := 1 / 100 in
  let probability : ℚ := (n.choose 1) * (p) * ((1 - p) ^ (n - 1)) in
  probability ≈ 0.370 :=
by
  let n : ℕ := 100 
  let p : ℚ := 1 / 100
  let probability : ℚ := (n.choose 1) * (p) * ((1 - p) ^ (n - 1))
  sorry

end last_box_one_ball_probability_l112_112103


namespace maximize_ratio_l112_112456

open_locale big_operators

variables {Point : Type} [metric_space Point]
variables (k : set Point) (P Q X : Point)
variables (hP : P ∈ k) (hQ : Q ∉ k) (hX : X ∈ k)

def perpendicular_foot (Q : Point) (k : set Point) : Point := sorry

def angle (a b c : Point) : real := sorry

theorem maximize_ratio (R : Point) : 
  (QX = perpendicular_foot Q k) →
  (R ∈ k) →
  ∃ PX_ray : set Point, ∀ R ∈ PX_ray, 
    (∃ S, angle Q R S = π/2)
:= 
sorry

end maximize_ratio_l112_112456


namespace probability_ball_in_last_box_is_approx_0_l112_112107

/-- Given 100 boxes and 100 balls randomly distributed,
the probability that the last box will contain the only ball is approximately 0.370. -/
theorem probability_ball_in_last_box_is_approx_0.370 :
  let n := 100,
      p := 1 / n,
      probability : ℝ := (99 / 100)^99
  in abs (probability - 0.370) < 0.001 :=
by {
  let n := 100,
  let p := 1 / n,
  let probability := (99 / 100)^99,
  sorry
}

end probability_ball_in_last_box_is_approx_0_l112_112107


namespace sum_of_C_coord_is_neg5_l112_112444

-- Define the coordinates of points A, B, and D
def point_A := (2, -3 : ℤ × ℤ)
def point_B := (7, 0 : ℤ × ℤ)
def point_D := (-2, 5 : ℤ × ℤ)

-- Define the coordinates of point C
def point_C : ℤ × ℤ := (-7, 2)

-- Theorem: The sum of the coordinates of vertex C is -5
theorem sum_of_C_coord_is_neg5 : point_C.1 + point_C.2 = -5 := by
  -- Given conditions
  have A := point_A
  have B := point_B
  have D := point_D

  -- Point C (to be calculated)
  -- Expected value calculation from conditions (solution implicitly assumed)
  have C := point_C

  -- The sum of the coordinates.
  show C.1 + C.2 = -5 from sorry

end sum_of_C_coord_is_neg5_l112_112444


namespace range_of_m_inequality_system_l112_112442

theorem range_of_m_inequality_system (m : ℝ) :
  (∀ x : ℤ, (-5 < x ∧ x ≤ m + 1) ↔ (x = -4 ∨ x = -3 ∨ x = -2)) →
  -3 ≤ m ∧ m < -2 :=
by
  sorry

end range_of_m_inequality_system_l112_112442


namespace value_of_expression_l112_112265

theorem value_of_expression (x y : ℝ) (h1 : x = 12) (h2 : y = 18) : 3 * (x - y) * (x + y) = -540 :=
by
  rw [h1, h2]
  sorry

end value_of_expression_l112_112265


namespace problem_1_and_2_l112_112600

-- Definition of the problem in Lean 4
theorem problem_1_and_2 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin (A)^2 - sin (B)^2 - sin (C)^2 = sin (B) * sin (C))
  (h2 : BC = 3)
  (h3 : triangle ABC)
  (h4 : a = side_length (opposite A))
  (h5 : b = side_length (opposite B))
  (h6 : c = side_length (opposite C)) :
  A = 2 * real.pi / 3 ∧
  (a + b + c ≤ 3 + 2 * real.sqrt 3) :=
sorry

end problem_1_and_2_l112_112600


namespace nonoverlapping_area_difference_l112_112305

theorem nonoverlapping_area_difference :
  let radius := 3
  let side := 2
  let circle_area := Real.pi * radius^2
  let square_area := side^2
  ∃ (x : ℝ), (circle_area - x) - (square_area - x) = 9 * Real.pi - 4 :=
by
  sorry

end nonoverlapping_area_difference_l112_112305


namespace min_value_of_quadratic_l112_112738

theorem min_value_of_quadratic (x : ℝ) : ∃ m : ℝ, (∀ x, x^2 + 10 * x ≥ m) ∧ m = -25 := by
  sorry

end min_value_of_quadratic_l112_112738


namespace problem_prove_FB_eq_FD_l112_112655

theorem problem_prove_FB_eq_FD 
  (A B C D E F : Point) -- Points involved
  (circle : Circle) -- Circle in which quadrilateral is inscribed
  (AB_eq_BC : AB = BC) -- Given AB = BC
  (DA_lt_AB_and_AB_lt_CD : DA < AB ∧ AB < CD) -- Given DA < AB and AB < CD
  (H_ABCD_Quad : QuadrilateralInscribedInCircle A B C D circle) -- Quadrilateral ABCD inscribed in circle
  (H_BE_perp_AC : BE ⊥ AC) -- Given BE ⊥ AC
  (H_EF_parallel_BC : EF ∥ BC) -- Given EF ∥ BC
  : FB = FD := 
sorry

end problem_prove_FB_eq_FD_l112_112655


namespace memorable_labelling_n_gon_l112_112638

theorem memorable_labelling_n_gon (n : ℕ) (h : n ≥ 3) : 
  (∃ (f : fin (2 * n + 1) → ℤ), 
    (∀ i : fin n, 
      (f ⟨i.1, sorry⟩ = (f ⟨i.1 % n, sorry⟩ + f ⟨(i.1 + 1) % n, sorry⟩) / 2) ∧
    (f ⟨n, sorry⟩ = (finset.univ.sum (λ i : fin n, f i) / n)) )) 
  ↔ n % 4 = 0 :=
sorry

end memorable_labelling_n_gon_l112_112638


namespace gcd_min_value_l112_112518

-- Definitions of the conditions
def is_positive_integer (x : ℕ) := x > 0

def gcd_cond (m n : ℕ) := Nat.gcd m n = 18

-- The main theorem statement
theorem gcd_min_value (m n : ℕ) (hm : is_positive_integer m) (hn : is_positive_integer n) (hgcd : gcd_cond m n) : 
  Nat.gcd (12 * m) (20 * n) = 72 :=
sorry

end gcd_min_value_l112_112518


namespace polynomial_even_degree_only_l112_112373

theorem polynomial_even_degree_only (p : Polynomial ℤ) (h : ∀ a b : ℤ, a + b ≠ 0 → (a + b) ∣ (p.eval a - p.eval b)) :
  ∀ n : ℕ, odd n → (Polynomial.coeff p n = 0) :=
sorry

end polynomial_even_degree_only_l112_112373


namespace insects_legs_l112_112540

theorem insects_legs (L N : ℕ) (hL : L = 54) (hN : N = 9) : (L / N = 6) :=
by sorry

end insects_legs_l112_112540


namespace max_percentage_both_services_l112_112766

theorem max_percentage_both_services {W S : Type} 
  (PW : ℕ → Prop) (PS : ℕ → Prop)
  (h1 : ∀ n, PW n ↔ n ≤ 50) 
  (h2 : ∀ n, PS n ↔ n ≤ 70) :
  ∃ PWS, (∀ n, PWS n ↔ n ≤ 50) ∧ ∀ n, PW n ∧ PS n → PWS n :=
sorry

end max_percentage_both_services_l112_112766


namespace floor_x_mul_x_eq_54_l112_112379

def positive_real (x : ℝ) : Prop := x > 0

theorem floor_x_mul_x_eq_54 (x : ℝ) (h_pos : positive_real x) : ⌊x⌋ * x = 54 ↔ x = 54 / 7 :=
by
  sorry

end floor_x_mul_x_eq_54_l112_112379


namespace triangle_theorem_l112_112605

theorem triangle_theorem 
  (A B C : ℝ) 
  (h1 : ∀ A B C : ℝ, ∃ (a b c : ℝ), 
      a^2 - b^2 - c^2 = b * c ∧ sin^2 A - sin^2 B - sin^2 C = sin B * sin C) 
  (h2 : 0 < A ∧ A < π) : 
  (A = 2 * π / 3) ∧ 
  (∀ (BC : ℝ), BC = 3 → ∃ (a b c : ℝ), 
      a + b + c ≤ 3 + 2 * sqrt 3) :=
by
  sorry

end triangle_theorem_l112_112605


namespace solve_problem_l112_112482

-- Define the given conditions
def condition1 (a : ℝ) : Prop := real.cbrt (5 * a + 2) = 3
def condition2 (a b : ℝ) : Prop := real.sqrt (3 * a + b - 1) = 4
def condition3 (c : ℝ) : Prop := c = real.floor (real.sqrt 13)

-- Define the target values and resulting statement
def values_a_b_c (a b c : ℝ) : Prop := 
  a = 5 ∧ b = 2 ∧ c = 3

def final_sqrt (a b c : ℝ) : Prop :=
  real.sqrt (3 * a - b + c) = 4 ∨ real.sqrt (3 * a - b + c) = -4

-- Complete the statement bringing everything together
theorem solve_problem (a b c : ℝ) :
  condition1 a →
  condition2 a b →
  condition3 c →
  values_a_b_c a b c ∧ final_sqrt a b c :=
begin
  sorry
end

end solve_problem_l112_112482


namespace cos_double_angle_l112_112037

-- Define the point P and angle α
def P : ℝ × ℝ := (1, 2)
def α : ℝ := sorry  -- the angle α is determined by the point P

-- Using the condition that P is on the terminal side of angle α:
def cos_α := P.1 / real.sqrt (P.1 * P.1 + P.2 * P.2)

-- Define the proof problem
theorem cos_double_angle (h : cos_α = 1 / real.sqrt 5) : 2 * cos_α^2 - 1 = -3/5 :=
by 
  sorry

end cos_double_angle_l112_112037


namespace sum_smallest_largest_consecutive_even_integers_l112_112206

theorem sum_smallest_largest_consecutive_even_integers
  (n : ℕ) (a y : ℤ) 
  (hn_even : Even n) 
  (h_mean : y = (a + (a + 2 * (n - 1))) / 2) :
  2 * y = (a + (a + 2 * (n - 1))) :=
by
  sorry

end sum_smallest_largest_consecutive_even_integers_l112_112206


namespace calculate_expression_l112_112351

theorem calculate_expression :
  (-3: ℝ)^2 - (1/5: ℝ)^(-1) - (real.sqrt 8) * (real.sqrt 2) + (-2: ℝ)^0 = 1 := 
by
  sorry

end calculate_expression_l112_112351


namespace phi_cannot_be_pi_div_4_l112_112794

theorem phi_cannot_be_pi_div_4
  (φ : ℝ) (f : ℝ → ℝ) (H1 : ∀ x, f x = sin (x + φ/2) * cos (x + φ/2))
  (H2 : ∀ x, f (x - π/8) = 1/2 * sin (2*x - π/4 + φ))
  (H_even : ∀ x, f (x - π/8) = f (-x + π/8)) :
  ¬∃ k : ℤ, φ = k * π + π/4 :=
sorry

end phi_cannot_be_pi_div_4_l112_112794


namespace area_enclosed_by_equation_l112_112040

/-- Given the equation x^2 + y^2 = 3|x - y| + 3|x + y|, the area of the enclosed figure is 36 + 18π. -/
theorem area_enclosed_by_equation :
  ∃ (m n : ℤ), (x y : ℝ), x^2 + y^2 = 3 * |x - y| + 3 * |x + y| ∧ m + n = 54 := by
  sorry

end area_enclosed_by_equation_l112_112040


namespace geometric_sequence_second_term_l112_112704

theorem geometric_sequence_second_term (a r : ℝ) (h1 : a * r ^ 2 = 5) (h2 : a * r ^ 4 = 45) :
  a * r = 5 / 3 :=
by
  sorry

end geometric_sequence_second_term_l112_112704


namespace problem_1_and_2_l112_112594

-- Definition of the problem in Lean 4
theorem problem_1_and_2 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin (A)^2 - sin (B)^2 - sin (C)^2 = sin (B) * sin (C))
  (h2 : BC = 3)
  (h3 : triangle ABC)
  (h4 : a = side_length (opposite A))
  (h5 : b = side_length (opposite B))
  (h6 : c = side_length (opposite C)) :
  A = 2 * real.pi / 3 ∧
  (a + b + c ≤ 3 + 2 * real.sqrt 3) :=
sorry

end problem_1_and_2_l112_112594


namespace g_f_4_eq_l112_112682

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_g_diff'ble (x : ℝ) : Differentiable ℝ f ∧ Differentiable ℝ g
axiom f_eq (x : ℝ) : x * g (f x) * (deriv f) (g x) * (deriv g) x = f (g x) * (deriv g) (f x) * (deriv f) x
axiom f_nonneg (x : ℝ) : 0 ≤ f x
axiom g_pos (x : ℝ) : 0 < g x
axiom f_g_integral (a : ℝ) : ∫ x in 0 .. a, f (g x) = 1 - (real.exp (-2 * a)) / 2
axiom g_f_zero : g (f 0) = 1

theorem g_f_4_eq : g (f 4) = real.exp (-16) := sorry

end g_f_4_eq_l112_112682


namespace num_odd_binomial_coeffs_l112_112004

open Nat

theorem num_odd_binomial_coeffs :
  (Finset.card (Finset.filter (λ k, (binom 8 k) % 2 = 1) (Finset.range 9))) = 2 :=
sorry

end num_odd_binomial_coeffs_l112_112004


namespace triangle_internal_bisectors_perpendicular_l112_112692

theorem triangle_internal_bisectors_perpendicular
  (A B C A1 B1 C1 : Type)
  [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup A1] [AddGroup B1] [AddGroup C1]
  (ABC_triangle : Triangle A B C)
  (isCircumcircle : ∀(P : Type), Is_circumcircle A B C P) 
  (Intersect : ∀(P : Type), Intersect_internal_bisectors A B C A1 B1 C1 P) 
  : Perpendicular AA1 B1C1 := by
  sorry

end triangle_internal_bisectors_perpendicular_l112_112692


namespace probability_ball_in_last_box_is_approx_0_l112_112105

/-- Given 100 boxes and 100 balls randomly distributed,
the probability that the last box will contain the only ball is approximately 0.370. -/
theorem probability_ball_in_last_box_is_approx_0.370 :
  let n := 100,
      p := 1 / n,
      probability : ℝ := (99 / 100)^99
  in abs (probability - 0.370) < 0.001 :=
by {
  let n := 100,
  let p := 1 / n,
  let probability := (99 / 100)^99,
  sorry
}

end probability_ball_in_last_box_is_approx_0_l112_112105


namespace solve_for_x_l112_112194

theorem solve_for_x (a b c d : ℕ) (h1: a = 8) (h2: b = 15) (h3: c = 25) (h4: d = 16) :
  let x := (Real.sqrt (a ^ 2 + b ^ 2)) / (Real.sqrt (c + d))
  x = (17 * (Real.sqrt 41)) / 41 :=
by
  have h5 : a ^ 2 + b ^ 2 = 8 ^ 2 + 15 ^ 2, from
  by rw [h1, h2]
  have h6 : 8 ^ 2 + 15 ^ 2 = 64 + 225, from
  by norm_num
  have h7 : 64 + 225 = 289, from
  by norm_num
  have h8 : Real.sqrt 289 = 17, from
  by norm_num
  have h9 : c + d = 25 + 16, from
  by rw [h3, h4]
  have h10 : 25 + 16 = 41, from
  by norm_num
  
  have h11 : x = 17 / Real.sqrt 41, from
  show x = (Real.sqrt 289) / (Real.sqrt 41), by
  rw [h5, h6, h7, h8, h9, h10]
  
  rw [h11]
  exact Eq.symm (by
    field_simp
    rw [Real.sqrt_mul_self (by norm_num : 41 ≥ 0)]
    norm_num)

end solve_for_x_l112_112194


namespace water_added_to_salt_solution_l112_112331

theorem water_added_to_salt_solution (x : ℝ)
  (hx : x = 104.99999999999997)
  (initial_salt_fraction : ℝ)
  (h_initial_salt_fraction : initial_salt_fraction = 0.20)
  (evaporated_fraction : ℝ)
  (h_evaporated_fraction : evaporated_fraction = 1/4)
  (added_salt : ℝ)
  (h_added_salt : added_salt = 14)
  (final_salt_fraction : ℝ)
  (h_final_salt_fraction : final_salt_fraction = 1/3):
  let initial_salt = initial_salt_fraction * x,
      initial_volume_after_evaporation = x - x * evaporated_fraction,
      total_salt_after_adding = initial_salt + added_salt,
      final_total_volume = total_salt_after_adding / final_salt_fraction,
      water_added = final_total_volume - initial_volume_after_evaporation in
  water_added = 26.25 :=
by {
  sorry
}

end water_added_to_salt_solution_l112_112331


namespace solve_for_t_l112_112370

theorem solve_for_t (t : ℝ) (ht : t > 0) : 3 * log 2 t = log 2 (4 * t) → t = 2 :=
sorry

end solve_for_t_l112_112370


namespace kramer_vote_percentage_l112_112665

theorem kramer_vote_percentage (
  (total_votes_45: ℕ) := 942568,
  (percentage_occupied: ℚ) := 0.45) : 
  let total_votes := total_votes_45 / percentage_occupied in
  let half_votes := total_votes / 2 in
  half_votes / total_votes * 100 = 50 :=
sorry

end kramer_vote_percentage_l112_112665


namespace smallest_A_for_B_multiple_2016_l112_112313

theorem smallest_A_for_B_multiple_2016 :
  ∃ A : ℕ, (let B := A * 10 ^ (nat.digits 10 A).length + A in
  B % 2016 = 0) ∧ (∀ A' : ℕ, (let B' := A' * 10 ^ (nat.digits 10 A').length + A' in
  B' % 2016 = 0) → A ≤ A') :=
sorry

end smallest_A_for_B_multiple_2016_l112_112313


namespace max_min_condition_monotonic_condition_l112_112762

-- (1) Proving necessary and sufficient condition for f(x) to have both a maximum and minimum value
theorem max_min_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ -2*x₁ + a - (1/x₁) = 0 ∧ -2*x₂ + a - (1/x₂) = 0) ↔ a > Real.sqrt 8 :=
sorry

-- (2) Proving the range of values for a such that f(x) is monotonic on [1, 2]
theorem monotonic_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (-2 * x + a - (1 / x)) ≥ 0) ∨
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (-2 * x + a - (1 / x)) ≤ 0) ↔ a ≤ 3 ∨ a ≥ 4.5 :=
sorry

end max_min_condition_monotonic_condition_l112_112762


namespace range_of_m_inequality_system_l112_112441

theorem range_of_m_inequality_system (m : ℝ) :
  (∀ x : ℤ, (-5 < x ∧ x ≤ m + 1) ↔ (x = -4 ∨ x = -3 ∨ x = -2)) →
  -3 ≤ m ∧ m < -2 :=
by
  sorry

end range_of_m_inequality_system_l112_112441


namespace find_angle_A_max_perimeter_triangle_l112_112548

-- Part 1: Prove the value of angle A
theorem find_angle_A (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) :
  A = 2 * real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter for BC = 3
theorem max_perimeter_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (A : ℝ) (hA : A = 2 * real.pi / 3) :
  let b := 2 * real.sqrt 3 * sin B,
      c := 2 * real.sqrt 3 * sin C,
      perimeter := 3 + b + c
  in ∀ d : ℝ, -real.pi / 6 < d ∧ d < real.pi / 6 → 
     B + C = real.pi / 3 → 
     perimeter ≤ (3 + 2 * real.sqrt 3) := sorry

end find_angle_A_max_perimeter_triangle_l112_112548


namespace PXAXBXCCyclicOrCoincide_l112_112637

noncomputable def isogonal_conjugate (P : Point) (ABC : Triangle) : Point :=
sorry

noncomputable def midpoint_arc (arc : Arc) : Point :=
sorry

noncomputable def intersection_ray_circle (ray : Ray) (circle : Circle) : Point :=
sorry

/-
The setup theory and conditions.
-/
variables {A B C P Q L M N X_A X_B X_C : Point}
variables {ABC : Triangle}
variables {circumcircle_ABC : Circle}
variables {arc_BC arc_CA arc_AB : Arc}
variables {ray_LQ ray_MQ ray_NQ : Ray}
variables {circle_PBC circle_PCA circle_PAB : Circle}

-- Conditions based on the problem statement
axiom hP_in_ABC : P ∈ interior_of(ABC)
axiom hQ_is_isogonal_conjugate : Q = isogonal_conjugate P ABC
axiom hL_midpoint_arc_BC : L = midpoint_arc arc_BC
axiom hM_midpoint_arc_CA : M = midpoint_arc arc_CA
axiom hN_midpoint_arc_AB : N = midpoint_arc arc_AB
axiom hX_A_intersection : X_A = intersection_ray_circle ray_LQ circle_PBC
axiom hX_B_intersection : X_B = intersection_ray_circle ray_MQ circle_PCA
axiom hX_C_intersection : X_C = intersection_ray_circle ray_NQ circle_PAB

-- Proof problem: proving that P, X_A, X_B, X_C are concyclic or coincide
theorem PXAXBXCCyclicOrCoincide : cyclic {P, X_A, X_B, X_C} ∨ (P = X_A ∧ X_A = X_B ∧ X_B = X_C) :=
by 
  sorry

end PXAXBXCCyclicOrCoincide_l112_112637


namespace binomial_thm_equivalent_l112_112067

theorem binomial_thm_equivalent (a : ℕ → ℝ) :
  (2 + sqrt 3) ^ 100 = a 0 + a 1 * 1 + a 2 * 1^2 + ... + a 100 * 1^100 →
  (a 0 + a 2 + a 4 + ... + a 100)^2 - (a 1 + a 3 + a 5 + ... + a 99)^2 = 1 :=
by
  sorry

end binomial_thm_equivalent_l112_112067


namespace mean_score_all_students_l112_112666

theorem mean_score_all_students
  (M A E : ℝ) (m a e : ℝ)
  (hM : M = 78)
  (hA : A = 68)
  (hE : E = 82)
  (h_ratio_ma : m / a = 4 / 5)
  (h_ratio_mae : (m + a) / e = 9 / 2)
  : (M * m + A * a + E * e) / (m + a + e) = 74.4 := by
  sorry

end mean_score_all_students_l112_112666


namespace gerald_assault_sentence_l112_112445

definition original_sentence (total_sentence: ℝ) (extension: ℝ) : ℝ := total_sentence / (1 + extension)
definition sentence_for_assault (total_sentence: ℝ) (poisoning_sentence: ℝ) (extension: ℝ) : ℝ :=
  original_sentence total_sentence extension - poisoning_sentence

theorem gerald_assault_sentence : 
  sentence_for_assault 36 24 (1 / 3) = 3 := 
by 
  sorry

end gerald_assault_sentence_l112_112445


namespace least_integer_k_l112_112699

-- Given conditions
def sequence (b : Nat → ℝ) : Prop :=
  b 1 = 1 ∧ ∀ n : Nat, n ≥ 1 → 7^(b (n + 1) - b n) = (2 * (n:ℝ) + 3) / (2 * (n:ℝ) + 1)

-- Theorem to prove
theorem least_integer_k (b : Nat → ℝ) (h : sequence b) : ∃ k : Nat, k > 1 ∧ b k ∈ Int ∧ k = 24 :=
by
  sorry

end least_integer_k_l112_112699


namespace prime_power_minus_l112_112997

theorem prime_power_minus (p : ℕ) (hp : Nat.Prime p) (hps : Nat.Prime (p + 3)) : p ^ 11 - 52 = 1996 := by
  -- this is where the proof would go
  sorry

end prime_power_minus_l112_112997


namespace bisects_diagonals_l112_112924

-- Define the data structure for a convex quadrilateral
structure ConvexQuadrilateral (α : Type*) :=
(A B C D : α)

-- Define midpoints of line segments
def midpoint {α : Type*} [Add α] [Div α] [Nonempty α] (A B : α) : α :=
(A + B) / 2

-- Main theorem stating the problem
theorem bisects_diagonals
  {α : Type*} [AddCommGroup α] [Module ℝ α] (quad : ConvexQuadrilateral α)
  (P Q : α)
  (hP : P = midpoint quad.A quad.B)
  (hQ : Q = midpoint quad.C quad.D)
  (hPQ : ∃ M, M = midpoint quad.A quad.C ∧ M ∈ line_through P Q) :
  ∃ N, N = midpoint quad.B quad.D ∧ N ∈ line_through P Q :=
sorry

lemma line_through (P Q : α) : Prop :=
∃ (λ1 λ2 : ℝ), P + λ1 • (Q - P) = Q + λ2 • (P - Q)

end bisects_diagonals_l112_112924


namespace avg_height_of_class_is_168_6_l112_112208

noncomputable def avgHeightClass : ℕ → ℕ → ℕ → ℕ → ℚ :=
  λ n₁ h₁ n₂ h₂ => (n₁ * h₁ + n₂ * h₂) / (n₁ + n₂)

theorem avg_height_of_class_is_168_6 :
  avgHeightClass 40 169 10 167 = 168.6 := 
by 
  sorry

end avg_height_of_class_is_168_6_l112_112208


namespace correct_transformation_l112_112446

variable (a b : ℝ)
variable (h₀ : a ≠ 0)
variable (h₁ : b ≠ 0)
variable (h₂ : a / 2 = b / 3)

theorem correct_transformation : 3 / b = 2 / a :=
by
  sorry

end correct_transformation_l112_112446


namespace shortest_chord_length_l112_112701

theorem shortest_chord_length:
  let M := {p : ℝ × ℝ | p.1^2 + p.2^2 - 6 * p.1 = 0}
  let A := (1, 1)
  ∃ (l : ℝ), l = 4 ∧ ∀ (C : set (ℝ × ℝ)), (∃ p ∈ C, p ∈ M ∧ A ∈ C) →
    (∃ q ∈ C, q ∈ M ∧ A ∈ C → (dist p q = l)) :=
sorry

end shortest_chord_length_l112_112701


namespace math_proof_problem_l112_112953

open Classical

-- Define the problem conditions
variables (ABC : Type) [triangle ABC] 
variables (P Q X Y : point ABC)
variables (XP PQ QY : ℝ) (AB AC : ℝ)

-- Assume conditions given in the problem
def conditions : Prop :=
  acute_triangle ABC ∧
  perpendicular_from C P AB ∧
  perpendicular_from B Q AC ∧
  line_passes_through PQ (circumcircle ABC) [X, Y] ∧
  XP = 10 ∧ PQ = 25 ∧ QY = 15

-- Define the question equivalent proof problem
theorem math_proof_problem (h : conditions ABC P Q X Y XP PQ QY AB AC) :
  ∃ m n : ℕ, AB * AC = m * real.sqrt n ∧ m + n = 574 :=
sorry

end math_proof_problem_l112_112953


namespace num_three_digit_numbers_l112_112185

theorem num_three_digit_numbers : 
  ∃ count : ℕ, count = 36 ∧ 
    (count = (Nat.choose 2 1) * (Nat.choose 3 2) * (Nat.factorial 3)) :=
by
  have even_choices : ℕ := Nat.choose 2 1
  have odd_choices : ℕ := Nat.choose 3 2
  have arrangements : ℕ := Nat.factorial 3
  let count := even_choices * odd_choices * arrangements
  use count
  split
  { refl }
  { simp [even_choices, odd_choices, arrangements]; 
    sorry }

end num_three_digit_numbers_l112_112185


namespace angle_at_center_l112_112796

-- Definition of Earth as a perfect sphere, and points Ajay and Billy with given coordinates
structure Coordinates (latitude longitude : ℝ)

def A : Coordinates := Coordinates (0) (100) -- Ajay's coordinates
def B : Coordinates := Coordinates (45) (-115) -- Billy's coordinates with W longitude as negative

noncomputable def angleACB (A B : Coordinates) : ℝ :=
  let d_longitude := (360 - A.longitude - B.longitude) % 360
  d_longitude -- the solution states this directly

theorem angle_at_center :
  angleACB A B = 145 :=
by  
  unfold angleACB,
  simp [A, B],
  exact sorry

end angle_at_center_l112_112796


namespace find_diagonal_length_l112_112375

theorem find_diagonal_length (d : ℝ) (offset1 offset2 : ℝ) (area : ℝ)
  (h1 : offset1 = 9)
  (h2 : offset2 = 6)
  (h3 : area = 300) :
  (1/2) * d * (offset1 + offset2) = area → d = 40 :=
by
  -- placeholder for proof
  sorry

end find_diagonal_length_l112_112375


namespace lines_intersection_numbering_l112_112950

open Nat

theorem lines_intersection_numbering (N : ℕ) 
  (h1 : ∀ i j : ℕ, i < N → j < N → i ≠ j → ∃! p : ℕ × ℕ, p ∈ finset.range N ×ˢ finset.range N)
  (h2 : ∀ i j k : ℕ, i < N → j < N → k < N → i ≠ j → j ≠ k → i ≠ k → ∃! l : ℕ × ℕ, l ∈ finset.range N ×ˢ finset.range N → l ∈ finset.range i ×ˢ finset.range j)
  : Even N :=
sorry

end lines_intersection_numbering_l112_112950


namespace four_digit_even_numbers_count_and_sum_l112_112148

variable (digits : Set ℕ) (used_once : ∀ d ∈ digits, d ≤ 6 ∧ d ≥ 1)

theorem four_digit_even_numbers_count_and_sum
  (hyp : digits = {1, 2, 3, 4, 5, 6}) :
  ∃ (N M : ℕ), 
    (N = 180 ∧ M = 680040) := 
sorry

end four_digit_even_numbers_count_and_sum_l112_112148


namespace number_of_people_in_first_group_l112_112071

variable (W : ℝ)  -- Amount of work
variable (P : ℝ)  -- Number of people in the first group

-- Condition 1: P people can do 3W work in 3 days
def condition1 : Prop := P * (W / 1) * 3 = 3 * W

-- Condition 2: 5 people can do 5W work in 3 days
def condition2 : Prop := 5 * (W / 1) * 3 = 5 * W

-- Theorem to prove: The number of people in the first group is 3
theorem number_of_people_in_first_group (h1 : condition1 W P) (h2 : condition2 W) : P = 3 :=
by
  sorry

end number_of_people_in_first_group_l112_112071


namespace smallest_repeating_block_7_div_11_l112_112062

theorem smallest_repeating_block_7_div_11 :
  ∃ n : ℕ, n = 2 ∧ ∃ r : ℝ, (r = 7/11) ∧ repeated_block_length r = n :=
sorry

end smallest_repeating_block_7_div_11_l112_112062


namespace sum_slope_intercept_bisector_l112_112718

structure Point :=
  (x : ℝ)
  (y : ℝ)

def P : Point := ⟨0, 10⟩
def Q : Point := ⟨4, 0⟩
def R : Point := ⟨10, 0⟩

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

def M : Point := midpoint P R

def slope (A B : Point) : ℝ :=
  (B.y - A.y) / (B.x - A.x)

def y_intercept (m : ℝ) (A : Point) : ℝ :=
  A.y - m * A.x

def sum_slope_intercept (A B : Point) : ℝ :=
  let m := slope A B
  let b := y_intercept m A
  m + b

theorem sum_slope_intercept_bisector (P Q R : Point) (M : Point) :
  M = midpoint P R →
  Q.x ≠ M.x →
  sum_slope_intercept Q M = -15 :=
by
  intros h₁ h₂
  rw [h₁]
  have h₃ : slope Q M = 5 := sorry
  have h₄ : y_intercept 5 Q = -20 := sorry
  unfold sum_slope_intercept
  rw [h₃, h₄]
  norm_num

end sum_slope_intercept_bisector_l112_112718


namespace ms_cole_total_students_l112_112171

def students_6th : ℕ := 40
def students_4th : ℕ := 4 * students_6th
def students_7th : ℕ := 2 * students_4th

def total_students : ℕ := students_6th + students_4th + students_7th

theorem ms_cole_total_students :
  total_students = 520 :=
by
  sorry

end ms_cole_total_students_l112_112171


namespace sequence_ab_sum_l112_112367

theorem sequence_ab_sum (s a b : ℝ) (h1 : 16 * s = 4) (h2 : 1024 * s = a) (h3 : a * s = b) : a + b = 320 := by
  sorry

end sequence_ab_sum_l112_112367


namespace jackie_free_time_correct_l112_112973

noncomputable def jackie_free_time : ℕ :=
  let total_hours_in_a_day := 24
  let hours_working := 8
  let hours_exercising := 3
  let hours_sleeping := 8
  let total_activity_hours := hours_working + hours_exercising + hours_sleeping
  total_hours_in_a_day - total_activity_hours

theorem jackie_free_time_correct : jackie_free_time = 5 := by
  sorry

end jackie_free_time_correct_l112_112973


namespace all_three_selected_l112_112710

-- Define the probabilities
def P_R : ℚ := 6 / 7
def P_Rv : ℚ := 1 / 5
def P_Rs : ℚ := 2 / 3
def P_Rv_given_R : ℚ := 2 / 5
def P_Rs_given_Rv : ℚ := 1 / 2

-- The probability that all three are selected
def P_all : ℚ := P_R * P_Rv_given_R * P_Rs_given_Rv

-- Prove that the calculated probability is equal to the given answer
theorem all_three_selected : P_all = 6 / 35 :=
by
  sorry

end all_three_selected_l112_112710


namespace part1_part2_l112_112618

-- Part (1): Prove that A = 2π/3 given the trigonometric condition.
theorem part1 (A B C : ℝ) (h_condition : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : 
  A = 2 * π / 3 :=
sorry

-- Part (2): Prove that the maximum perimeter is 3 + 2√3 given BC = 3 and A = 2π/3.
theorem part2 (B C : angle) (h_BC : 3 = 3) (h_A : A = 2 * π / 3) :
  (3 + (sqrt 3) * 2 ≤ perimeter (triangle.mk 3 B C)) :=
sorry

end part1_part2_l112_112618


namespace acceptable_component_probability_l112_112774

theorem acceptable_component_probability :
  let p1 := 0.01
  let p2 := 0.03
  let acceptable_prob := (1 - p1) * (1 - p2)
  (acceptable_prob = 0.99 * 0.97) ∧ (acceptable_prob ≈ 0.960 : ℝ) :=
by
  sorry

end acceptable_component_probability_l112_112774


namespace decreasing_function_range_of_a_l112_112044

def f (a x : ℝ) : ℝ := if x ≤ 1 then (a - 3) * x + 5 else (2 * a / x)

theorem decreasing_function_range_of_a :
  (∀ x y : ℝ, x < y → f a y < f a x) → 
  (0 < a ∧ a ≤ 2) :=
by
  sorry

end decreasing_function_range_of_a_l112_112044


namespace log_limit_l112_112904

open Real

theorem log_limit : ∀ (x : ℝ), x > 0 → (∃ l, x → ∞ → l = (\log 3 (6 * x - 5) - \log 3 (2 * x + 1)) = 1) :=
  sorry

end log_limit_l112_112904


namespace find_angle_between_a_and_b_l112_112475

open Real

variables (a b c : ℝ^3)

def magnitude (v : ℝ^3) : ℝ := sqrt (v.dot v)

noncomputable def dot_product (u v : ℝ^3) : ℝ := u.dot v

noncomputable def angle_between (u v : ℝ^3) : ℝ :=
real.arccos (dot_product u v / (magnitude u * magnitude v))

noncomputable def degree (r : ℝ) : ℝ := r * (180 / π)

axiom h1 : magnitude a = 1
axiom h2 : magnitude b = 2
axiom h3 : c = a + b
axiom h4 : dot_product c a = 0

theorem find_angle_between_a_and_b : degree (angle_between a b) = 120 :=
sorry

end find_angle_between_a_and_b_l112_112475


namespace opposite_of_neg_two_thirds_l112_112229

theorem opposite_of_neg_two_thirds : - (- (2 / 3) : ℚ) = (2 / 3 : ℚ) :=
by
  sorry

end opposite_of_neg_two_thirds_l112_112229


namespace arithmetic_sequence_num_terms_l112_112354

noncomputable def num_terms_in_arithmetic_seq (a1 an d : ℤ) : ℕ :=
  ((an - a1) / d + 1).to_nat

theorem arithmetic_sequence_num_terms :
  num_terms_in_arithmetic_seq (-25) 96 7 = 18 :=
by
  -- problem conditions
  let a1 : ℤ := -25
  let an : ℤ := 96
  let d : ℤ := 7

  -- use the noncomputable definition
  show num_terms_in_arithmetic_seq a1 an d = 18

  -- sorry is added to circumvent providing the actual proof
  sorry

end arithmetic_sequence_num_terms_l112_112354


namespace gcd_power_of_two_sub_one_l112_112726

def a : ℤ := 2^1100 - 1
def b : ℤ := 2^1122 - 1
def c : ℤ := 2^22 - 1

theorem gcd_power_of_two_sub_one :
  Int.gcd (2^1100 - 1) (2^1122 - 1) = 2^22 - 1 := by
  sorry

end gcd_power_of_two_sub_one_l112_112726


namespace hyperbola_focus_condition_l112_112026

noncomputable def hyperbola_excentricity (a b : ℝ) : ℝ :=
  let c := sqrt (a^2 + b^2) / 2 in
  let eccentricity := c / a in
  eccentricity

theorem hyperbola_focus_condition (a b : ℝ) (h : a > 0 ∧ b > 0)
  (P_on_C : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (PF1_perp_F1F2 : ∀ F1 F2 : ℝ × ℝ, F1.1 = -c ∧ F2.1 = c → (F1.1, F2.1).fst - √2 * F1.2 = 0)
  (PF1_eq_F1F2 : ∀ F1 F2 : ℝ × ℝ, F1.1 = -c ∧ F2.1 = c → (F1.1, F2.1).fst = 2 * c):
  hyperbola_excentricity a b = sqrt 2 + 1 :=
sorry

end hyperbola_focus_condition_l112_112026


namespace exponential_inequality_l112_112449

variable {a b : ℝ}

theorem exponential_inequality (h : a > b) : e ^ (-a) - e ^ (-b) < 0 :=
by {
  sorry
}

end exponential_inequality_l112_112449


namespace twelve_row_triangle_pieces_l112_112308

theorem twelve_row_triangle_pieces :
  let S_n_arithmetic_sum (a d n : ℕ) := n * (2 * a + (n - 1) * d) / 2
  let total_rods := S_n_arithmetic_sum 3 3 12
  let total_connectors := S_n_arithmetic_sum 1 1 13
  total_rods + total_connectors = 325 :=
by
  sorry

end twelve_row_triangle_pieces_l112_112308


namespace rachel_minutes_before_bed_l112_112673

-- Define the conditions in the Lean Lean.
def minutes_spent_solving_before_bed (m : ℕ) : Prop :=
  let problems_solved_before_bed := 5 * m
  let problems_finished_at_lunch := 16
  let total_problems_solved := 76
  problems_solved_before_bed + problems_finished_at_lunch = total_problems_solved

-- The statement we want to prove
theorem rachel_minutes_before_bed : ∃ m : ℕ, minutes_spent_solving_before_bed m ∧ m = 12 :=
sorry

end rachel_minutes_before_bed_l112_112673


namespace find_parking_time_l112_112664

-- Define the constants and conditions
constant P : ℕ -- P is the number of minutes to find parking

axiom walk_time : ∀ d : ℕ, d = 3 -- It takes Mark 3 minutes to walk into the courthouse each day
axiom metal_detector_time : ∀ d : ℕ, (d = 30) ∨ (d = 10) -- Two different times for metal detector

axiom week_conditions : 
  ∀ t1 t2 t3 t4 t5 : ℕ, 
  (t1 = P + 3 + 30 ∧ t2 = P + 3 + 30 ∧ t3 = P + 3 + 10 ∧ t4 = P + 3 + 10 ∧ t5 = P + 3 + 10) →
  (t1 + t2 + t3 + t4 + t5 = 130) -- Total time spent on these activities in a week is given as 130 minutes

-- Prove the math problem statement
theorem find_parking_time : P = 5 :=
sorry -- Proof omitted

end find_parking_time_l112_112664


namespace tetrahedron_edge_length_l112_112443

-- Definitions of the conditions based on the problem description
def is_equilateral_triangle (a b c : ℝ) : Prop :=
  a = b ∧ b = c ∧ a = c

def centers_satisfy_conditions (A B C D : ℝ × ℝ × ℝ) (r : ℝ) : Prop :=
  (A.2.2 = 0) ∧ (B.2.2 = 0) ∧ (C.2.2 = 0) ∧ 
  (D.2.2 = r * sqrt 6 / 3) ∧
  is_equilateral_triangle (dist A B) (dist B C) (dist C A) ∧ 
  dist A B = 2 * r ∧ dist B C = 2 * r ∧ dist C A = 2 * r ∧
  dist A D = 2 * r ∧ dist B D = 2 * r ∧ dist C D = 2 * r

-- The theorem to be proven
theorem tetrahedron_edge_length {A B C D : ℝ × ℝ × ℝ} (r : ℝ) (h : r = 2)
  (hcond : centers_satisfy_conditions A B C D r) : 
  ∃ s : ℝ, s = 4 :=
sorry

end tetrahedron_edge_length_l112_112443


namespace find_x_l112_112503

-- Define the sets M and N
def M (x : ℝ) : set ℝ := {1, x ^ 2}
def N (x : ℝ) : set ℝ := {1, x}

-- Define the theorem that states the proof problem
theorem find_x (x : ℝ) (h : M x = N x) : x = 0 :=
by
  sorry

end find_x_l112_112503


namespace arithmetic_sequence_properties_l112_112012

-- Define the arithmetic sequence with general term and sum formula
variables {a_n : ℕ → ℤ} {S_n : ℕ → ℤ} {a_1 : ℤ} {d : ℤ}

-- Definitions of the conditions
def a_4 := a_n 4
def a_8 := a_n 8
def a_formula := ∀ n : ℕ, a_n n = a_1 + (n-1) * d
def S_formula := ∀ n : ℕ, S_n n = ((n * (2 * a_1 + (n-1) * d)) / 2)

-- Given specific terms of the sequence
axiom h1 : a_4 = -12
axiom h2 : a_8 = -4

-- Main theorem for the proof problem
theorem arithmetic_sequence_properties :
  (∀ n, a_n n = 2 * n - 20) ∧ ((S_n 9 = -90 ∨ S_n 10 = -90) ∧ ∀ n, S_n n ≥ -90) :=
by
  sorry

end arithmetic_sequence_properties_l112_112012


namespace circles_tangent_to_each_other_l112_112674

variable (A B C : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (r_a r_b r_c : ℝ) -- radii of excircles opposite to vertices A, B, and C respectively
variable (r_A r_B r_C : ℝ) -- radii of circles with centers at vertices A, B, and C respectively
variable (BC : ℝ) -- length BC of triangle ABC
variable (p : ℝ) -- semiperimeter of triangle ABC
variable (alpha : ℝ) -- angle at vertex A
variable (beta : ℝ) -- angle at vertex B
variable (gamma : ℝ) -- angle at vertex C

-- Conditions
variable (cond1 : r_A = r_a )
variable (cond2 : r_B = r_b )
variable (cond3 : r_C = r_c )

-- Two cases
variable (case1 : r_B + r_C = BC )
variable (case2 : r_B - r_C = BC )

-- Theorem statement
theorem circles_tangent_to_each_other :
  (r_A = r_a) ∧ (r_B = r_b) ∧ (r_C = r_c) →
  (r_B + r_C = BC ∨ r_B - r_C = BC) →
  (∀ (x y : ℝ), (x = r_A ∨ x = r_B ∨ x = r_C) → 
               (y = r_A ∨ y = r_B ∨ y = r_C) →
               x ≠ y → areTangent x y) :=
by
  intros h1 h2
  sorry

end circles_tangent_to_each_other_l112_112674


namespace buses_encountered_l112_112211

theorem buses_encountered (depart_hourly : ∀ n : ℕ, bus_departs n)
                          (depart_half_hourly : ∀ n : ℕ, bus_departs_half_hour (n + 0.5))
                          (journey_time : ℝ := 5) :
  let encounter_count := 10 in
  bus_journey_encounter "Coco da Selva" "Quixajuba" 12 = encounter_count :=
sorry

end buses_encountered_l112_112211


namespace find_angle_A_max_perimeter_l112_112565

noncomputable def sin_sq_minus (A B C : ℝ) : ℝ :=
  (Real.sin A) * (Real.sin A) - (Real.sin B) * (Real.sin B) - (Real.sin C) * (Real.sin C)

noncomputable def sin_prod (B C : ℝ) : ℝ :=
  (Real.sin B) * (Real.sin C)

theorem find_angle_A (A B C : ℝ) (h : sin_sq_minus A B C = sin_prod B C) :
  A = 2 * Real.pi / 3 :=
by
  sorry

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem max_perimeter (B C : ℝ) (a : ℝ) (h1 : a = 3) (h2 : B + C = Real.pi / 3) :
  ∃ (b c : ℝ), perimeter a b c = 3 + 2 * Real.sqrt 3 :=
by
  sorry

end find_angle_A_max_perimeter_l112_112565


namespace reservoir_80_percent_full_after_storm_l112_112337

-- Define the conditions as given in the problem
variable (w_storm : ℕ := 115) -- storm deposits 115 billion gallons
variable (W_orig : ℕ := 245) -- original contents of the reservoir 245 billion gallons
variable (p_orig : ℝ := 0.5444444444444444) -- percentage full before the storm

-- Define a function that calculates the total capacity of the reservoir using the given percentage
def total_capacity (W_orig : ℕ) (p_orig : ℝ) : ℝ :=
  W_orig / p_orig

-- Define a function to calculate the new amount of water in the reservoir after the storm
def new_water_amount (W_orig w_storm : ℕ) : ℕ :=
  W_orig + w_storm

-- Define a function to calculate the new percentage full
def new_percentage_full (W_new : ℝ) (C : ℝ) : ℝ :=
  (W_new / C) * 100

-- Combine the above to prove that the reservoir is 80% full after the storm
theorem reservoir_80_percent_full_after_storm :
  let C := total_capacity W_orig p_orig,
      W_new := new_water_amount W_orig w_storm,
      percentage_new := new_percentage_full (W_new.to_real) C
  in percentage_new = 80 := by
  sorry

end reservoir_80_percent_full_after_storm_l112_112337


namespace find_AC_l112_112335

noncomputable def triangle_sides (A B C : ℝ) (sA cA sB cB : ℝ) : ℝ :=
  if BC = 1 ∧ sA = Real.sin (A / 2) ∧ cA = Real.cos (A / 2) ∧ 
     sB = Real.sin (B / 2) ∧ cB = Real.cos (B / 2) ∧ 
     sA * 23 * cB * 48 = sB * 23 * cA * 48 then
    Real.sin A / Real.sin C
  else
    0

theorem find_AC (A B C : ℝ) (sA cA sB cB : ℝ) (h1 : BC = 1) (h2 : sA = Real.sin (A / 2)) 
  (h3 : cA = Real.cos (A / 2)) (h4 : sB = Real.sin (B / 2)) (h5 : cB = Real.cos (B / 2)) 
  (h6 : sA * 23 * cB * 48 = sB * 23 * cA * 48) :
  triangle_sides A B C sA cA sB cB = Real.sin A / Real.sin C :=
by {
  sorry
}

end find_AC_l112_112335


namespace rod_volume_proof_l112_112307

-- Definitions based on given conditions
def original_length : ℝ := 2
def increase_in_surface_area : ℝ := 0.6
def rod_volume : ℝ := 0.3

-- Problem statement
theorem rod_volume_proof
  (len : ℝ)
  (inc_surface_area : ℝ)
  (vol : ℝ)
  (h_len : len = original_length)
  (h_inc_surface_area : inc_surface_area = increase_in_surface_area) :
  vol = rod_volume :=
sorry

end rod_volume_proof_l112_112307


namespace calculation_correct_l112_112808

-- Definitions for the conditions in the problem
def exponentiation := (-2) ^ 3
def absoluteValue := |2 - 5|
def division := absoluteValue / (-3)

-- Main proof statement
theorem calculation_correct :
  exponentiation - division = -7 := by
  sorry

end calculation_correct_l112_112808


namespace find_m_positive_root_l112_112879

theorem find_m_positive_root :
  (∃ x > 0, (x - 4) / (x - 3) - m - 4 = m / (3 - x)) → m = 1 :=
by
  sorry

end find_m_positive_root_l112_112879


namespace todd_saved_44_dollars_l112_112714

-- Definitions of the conditions.
def full_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon_discount : ℝ := 10
def credit_card_discount : ℝ := 0.10

-- The statement we want to prove: Todd saved $44 on the original price of the jeans.
theorem todd_saved_44_dollars :
  let sale_amount := full_price * sale_discount,
      price_after_sale := full_price - sale_amount,
      price_after_coupon := price_after_sale - coupon_discount,
      credit_card_amount := price_after_coupon * credit_card_discount,
      final_price := price_after_coupon - credit_card_amount,
      savings := full_price - final_price
  in savings = 44 :=
by
  sorry

end todd_saved_44_dollars_l112_112714


namespace triangle_theorem_l112_112603

theorem triangle_theorem 
  (A B C : ℝ) 
  (h1 : ∀ A B C : ℝ, ∃ (a b c : ℝ), 
      a^2 - b^2 - c^2 = b * c ∧ sin^2 A - sin^2 B - sin^2 C = sin B * sin C) 
  (h2 : 0 < A ∧ A < π) : 
  (A = 2 * π / 3) ∧ 
  (∀ (BC : ℝ), BC = 3 → ∃ (a b c : ℝ), 
      a + b + c ≤ 3 + 2 * sqrt 3) :=
by
  sorry

end triangle_theorem_l112_112603


namespace smallest_positive_real_x_l112_112391

theorem smallest_positive_real_x :
  ∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧ x = 89 / 9 :=
by
  sorry

end smallest_positive_real_x_l112_112391


namespace max_f_value_l112_112851

theorem max_f_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  (∑ i in [x, y, z], i * ((λ x y z : ℝ, (2 * y - z) / (1 + x + 3 * y)) x y z)) ≤ 1 / 7 :=
sorry

end max_f_value_l112_112851


namespace approximate_pi_value_l112_112210

theorem approximate_pi_value (r h : ℝ) (L : ℝ) (V : ℝ) (π : ℝ) 
  (hL : L = 2 * π * r)
  (hV : V = 1 / 3 * π * r^2 * h) 
  (approxV : V = 2 / 75 * L^2 * h) :
  π = 25 / 8 := 
by
  -- Proof goes here
  sorry

end approximate_pi_value_l112_112210


namespace part_a_part_b_l112_112951

-- Conditions
variable (H : RegularHexagon)
variable (area_H : H.area = 144)

-- Statements to prove
theorem part_a : H.diagonal_parts = 24 := 
sorry

theorem part_b : H.quadrilateral_hexagon_area = 48 :=
sorry

end part_a_part_b_l112_112951


namespace rational_solutions_are_integers_l112_112054

theorem rational_solutions_are_integers (a b : ℤ) (x y : ℚ) :
  (y - 2 * x = a) ∧ (y^2 - y * x + x^2 = b) →
  (∃ x y : ℚ, y - 2 * x = a ∧ y^2 - y * x + x^2 = b → x ∈ ℤ ∧ y ∈ ℤ) :=
by
  sorry

end rational_solutions_are_integers_l112_112054


namespace sum_of_coefficients_is_1_l112_112241

-- Given conditions:
def polynomial_expansion (x y : ℤ) := (x - 2 * y) ^ 18

-- Proof statement:
theorem sum_of_coefficients_is_1 : (polynomial_expansion 1 1) = 1 := by
  -- The proof itself is omitted as per the instruction
  sorry

end sum_of_coefficients_is_1_l112_112241


namespace sum_of_smallest_and_largest_eq_2y_l112_112204

variable (a n y : ℤ) (hn_even : Even n) (hy : y = a + n - 1)

theorem sum_of_smallest_and_largest_eq_2y : a + (a + 2 * (n - 1)) = 2 * y := 
by
  sorry

end sum_of_smallest_and_largest_eq_2y_l112_112204


namespace k_lt_zero_l112_112008

noncomputable def k_negative (k : ℝ) : Prop :=
  (∃ x : ℝ, x < 0 ∧ k * x > 0) ∧ (∃ x : ℝ, x > 0 ∧ k * x < 0)

theorem k_lt_zero (k : ℝ) : k_negative k → k < 0 :=
by
  intros h
  sorry

end k_lt_zero_l112_112008


namespace number_of_correct_statements_l112_112340

theorem number_of_correct_statements :
  ∀ (AB OA : Line) (A B O : Point),
    (AB.length = 3 ∧ AB is_segment) ∧
    (¬exists L : Line, is_extension L OA) ∧
    (¬(ray A B = ray B A)) ∧
    (segment AB = segment BA) →
  num_correct_statements = 1 :=
by
  intros,
  sorry

end number_of_correct_statements_l112_112340


namespace todd_savings_l112_112715

-- Define the initial conditions
def original_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon : ℝ := 10
def card_discount : ℝ := 0.10

-- Define the resulting values after applying discounts
def sale_price := original_price * (1 - sale_discount)
def after_coupon := sale_price - coupon
def final_price := after_coupon * (1 - card_discount)

-- Define the total savings
def savings := original_price - final_price

-- The proof statement
theorem todd_savings : savings = 44 := by
  sorry

end todd_savings_l112_112715


namespace smallest_positive_real_number_l112_112397

noncomputable def smallest_x : ℝ :=
  let x := 89 / 9 in x

theorem smallest_positive_real_number :
  ∀ x : ℝ, (x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8) → x ≥ smallest_x  :=
by
  sorry

end smallest_positive_real_number_l112_112397


namespace juanita_spends_more_l112_112060

def grant_yearly_spending := 200.00
def monday_to_saturday_daily_spending := 0.50
def sunday_spending := 2.00
def weeks_in_year := 52

theorem juanita_spends_more :
  let juanita_monday_to_saturday_weekly_spending := 6 * monday_to_saturday_daily_spending in
  let juanita_weekly_spending := juanita_monday_to_saturday_weekly_spending + sunday_spending in
  let juanita_yearly_spending := weeks_in_year * juanita_weekly_spending in
  juanita_yearly_spending - grant_yearly_spending = 60.00 :=
by
  sorry

end juanita_spends_more_l112_112060


namespace part1_part2_l112_112613

-- Part (1): Prove that A = 2π/3 given the trigonometric condition.
theorem part1 (A B C : ℝ) (h_condition : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : 
  A = 2 * π / 3 :=
sorry

-- Part (2): Prove that the maximum perimeter is 3 + 2√3 given BC = 3 and A = 2π/3.
theorem part2 (B C : angle) (h_BC : 3 = 3) (h_A : A = 2 * π / 3) :
  (3 + (sqrt 3) * 2 ≤ perimeter (triangle.mk 3 B C)) :=
sorry

end part1_part2_l112_112613


namespace necessary_condition_real_roots_l112_112843

theorem necessary_condition_real_roots (a : ℝ) :
  (a >= 1 ∨ a <= -2) → (∃ x : ℝ, x^2 - a * x + 1 = 0) :=
by
  sorry

end necessary_condition_real_roots_l112_112843


namespace inequality_solution_has_3_integer_solutions_l112_112435

theorem inequality_solution_has_3_integer_solutions (m : ℝ) :
  (∃ x ∈ set.Icc (-4) (-2), x ∈ ℤ ∧ (x + 5 > 0) ∧ (x - m ≤ 1)) →
  (-3 ≤ m ∧ m < -2) :=
by sorry

end inequality_solution_has_3_integer_solutions_l112_112435


namespace total_students_in_college_l112_112532

theorem total_students_in_college 
  (num_girls : ℕ) (h1 : num_girls = 300)
  (ratio : ℕ → ℕ)
  (h2 : ratio 5 = num_girls / 5 * 8)
  (num_students : ℕ) 
  (h3 : num_students = num_girls + (num_girls / 5 * 8)) :
  num_students = 780 :=
by
  rw [h1, h3, Nat.div_mul_cancel (show 300 % 5 = 0 by norm_num)]
  rfl

end total_students_in_college_l112_112532


namespace PQ_bisects_BD_l112_112948

variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]
variables {A B C D P Q M N : Point}

def convex_quadrilateral (A B C D : Point) : Prop := sorry
def midpoint (P A B : Point) : Prop := 2 • P = A + B
def bisects (line P Q : Point) (diagonal A C : Point) : Prop := 
  ∃ M, midpoint M A C ∧ (line.contains M)
def line_contains_midpoint (P Q : Point) (mid : Point) : Prop := sorry

-- The theorem we want to prove:
theorem PQ_bisects_BD 
  (h1 : convex_quadrilateral A B C D)
  (h2 : midpoint P A B)
  (h3 : midpoint Q C D)
  (h4 : bisects (P, Q) (A, C))
  : bisects (P, Q) (B, D) := 
begin
  sorry
end

end PQ_bisects_BD_l112_112948


namespace am_minus_one_divisible_by_a_minus_one_squared_l112_112287

theorem am_minus_one_divisible_by_a_minus_one_squared 
  (a m n : ℤ)
  (h : m = n * (a - 1)) :
  ∃ k : ℤ, a^m - 1 = k * (a - 1)^2 :=
by
  sorry

end am_minus_one_divisible_by_a_minus_one_squared_l112_112287


namespace sequence_x_value_l112_112547

theorem sequence_x_value :
  ∃ (x : ℕ), (seq : ℕ → ℕ) (h : seq 0 = 2 ∧ seq 1 = 5 ∧ seq 2 = 11 ∧ seq 3 = 20 ∧ seq 5 = 47)
           ((seq 1 - seq 0 = 3) ∧ (seq 2 - seq 1 = 6) ∧ (seq 3 - seq 2 = 9) ∧ (seq 4 - seq 3 = 12)) ∧ seq 4 = x ∧ x = 32 :=
by {
  let seq := λ n, if n = 0 then 2 else if n = 1 then 5 else if n = 2 then 11 else if n = 3 then 20 else if n = 5 then 47 else 0,
  existsi 32,
  split,
  exact seq,
  split,
  split,
  split,
  split,
  exact rfl,
  exact rfl,
  exact rfl,
  exact rfl,
  exact ⟨(5 - 2 = 3), (11 - 5 = 6), (20 - 11 = 9), (32 - 20 = 12)⟩,
  split,
  exact (32 - 20 = 12),
  exact rfl,
  exact ⟨32⟩,
  sorry
}

end sequence_x_value_l112_112547


namespace angle_is_pi_over_4_l112_112082

variables (a b : ℝ^3)
-- Condition 1: |a| = 1
axiom ha : ∥a∥ = 1
-- Condition 2: |b| = √2
axiom hb : ∥b∥ = Real.sqrt 2
-- Condition 3: (a + b) · a = 2
axiom hab : (a + b) • a = 2

-- Define the angle between vectors a and b
noncomputable def angle_between (a b : ℝ^3) : ℝ :=
Real.arccos ((a • b) / (∥a∥ * ∥b∥))

-- The main theorem to prove
theorem angle_is_pi_over_4 : angle_between a b = Real.pi / 4 :=
sorry

end angle_is_pi_over_4_l112_112082


namespace polyhedron_vertices_leq_eight_l112_112535

-- Defining the conditions
def is_lattice_point (p : ℤ × ℤ × ℤ) : Prop := true

def no_other_lattice_points (P : set (ℤ × ℤ × ℤ)) : Prop :=
  ∀ p ∈ P, is_lattice_point p ∧ ∀ q ∈ P, q ≠ p → ¬ is_lattice_point ((p.1 + q.1) / 2, (p.2 + q.2) / 2, (p.3 + q.3) / 2)

def convex_polyhedron (P : set (ℤ × ℤ × ℤ)) : Prop := 
  ∀ x y z : ℤ × ℤ × ℤ, x ∈ P → y ∈ P → z ∈ P → ∃ a b c d e f : ℤ, (a * x.1 + b * x.2 + c * x.3 + d * y.1 + e * y.2 + f * y.3 = 1)

-- Statement of the problem
theorem polyhedron_vertices_leq_eight (P : set (ℤ × ℤ × ℤ)) (h1 : convex_polyhedron P) (h2 : no_other_lattice_points P) :
  P.finite → P.card ≤ 8 :=
sorry

end polyhedron_vertices_leq_eight_l112_112535


namespace bisect_diagonal_BD_l112_112930

-- Define a convex quadrilateral ABCD with midpoints P and Q on sides AB and CD respectively.
variables {A B C D P Q M N : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables [Inhabited P] [Inhabited Q] [Inhabited M] [Inhabited N]

-- Assuming the given statements:
-- ABCD is a convex quadrilateral
-- P is the midpoint of AB
-- Q is the midpoint of CD
-- Line PQ bisects the diagonal AC

-- Prove that line PQ also bisects the diagonal BD
theorem bisect_diagonal_BD 
  (convex_quadrilateral : convex_quadrilateral A B C D)
  (midpoint_P : midpoint P A B)
  (midpoint_Q : midpoint Q C D)
  (PQ_bisects_AC : bisects_line PQ M A C) :
  bisects_line PQ N B D :=
sorry  -- Proof is omitted

end bisect_diagonal_BD_l112_112930


namespace division_ratio_of_E_l112_112529

-- Define the points and their ratios
variables {A B C F G E : Point}
variables (h1 : divides_in_ratio F A C (2 / 5))
variables (h2 : divides_in_ratio G B F (1 / 4))
variables (h3 : lies_on AG BC E)

-- Prove the ratio BE:EC is 2:5
theorem division_ratio_of_E (h1 : divides_in_ratio F A C (2 / 5)) 
                             (h2 : divides_in_ratio G B F (1 / 4)) 
                             (h3 : lies_on AG BC E) :
               divides_in_ratio E B C (2 / 7) := 
sorry

end division_ratio_of_E_l112_112529


namespace payment_to_C_l112_112769

/-- 
If A can complete a work in 6 days, B can complete the same work in 8 days, 
they signed to do the work for Rs. 2400 and completed the work in 3 days with 
the help of C, then the payment to C should be Rs. 300.
-/
theorem payment_to_C (total_payment : ℝ) (days_A : ℝ) (days_B : ℝ) (days_worked : ℝ) (portion_C : ℝ) :
   total_payment = 2400 ∧ days_A = 6 ∧ days_B = 8 ∧ days_worked = 3 ∧ portion_C = 1 / 8 →
   (portion_C * total_payment) = 300 := 
by 
  intros h
  cases h
  sorry

end payment_to_C_l112_112769


namespace sum_of_super_cool_rectangles_l112_112320

theorem sum_of_super_cool_rectangles :
  let is_super_cool (a b : ℕ) := (a * b = 6 * (a + b))
  let areas := {ab | ∃ a b : ℕ, a ≠ b ∧ is_super_cool a b ∧ ab = a * b}
  let sum_of_areas := areas.sum id in
  sum_of_areas = 942 :=
by {
  let is_super_cool := λ a b : ℕ, a * b = 6 * (a + b),
  let areas := finset.insert 0 (finset.univ.filter (λ ab, ∃ a b : ℕ, a ≠ b ∧ a * b = 6 * (a + b) ∧ ab = a * b)),
  let sum_of_areas := areas.sum id,
  have h: sum_of_areas = 942 := sorry,
  exact h,
}

end sum_of_super_cool_rectangles_l112_112320


namespace fraction_simplify_l112_112246

theorem fraction_simplify : (7 + 21) / (14 + 42) = 1 / 2 := by
  sorry

end fraction_simplify_l112_112246


namespace tangent_line_at_point_l112_112688

noncomputable def curve (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_line_at_point :
  let p : ℝ × ℝ := (1, 3) in
  let m  : ℝ := (3 * (1 : ℝ)^2 - 1) in
  ∃ (a b c : ℝ), a * (p.1) + b * (curve (p.1)) + c = 0 
  ∧ a = 2 ∧ b = -1 ∧ c = 1  :=
by
  sorry

end tangent_line_at_point_l112_112688


namespace count_valid_three_digit_numbers_l112_112065

def valid_hundreds_digit (n : ℕ) : Bool :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 % 10 ≠ 0) ∧ (n / 100 % 10 ≠ 1) ∧ (n / 100 % 10 ≠ 7) ∧ (n / 100 % 10 ≠ 8)

def valid_tens_digit (n : ℕ) : Bool :=
  (n % 100 / 10 != 1) ∧ (n % 100 / 10 != 7) ∧ (n % 100 / 10 != 8) ∧ (n % 100 / 10 != 9)

def valid_units_digit (n : ℕ) : Bool :=
  (n % 10 ≠ 1) ∧ (n % 10 ≠ 7) ∧ (n % 10 ≠ 8)

def valid_number (n : ℕ) : Bool :=
  valid_hundreds_digit n ∧ valid_tens_digit n ∧ valid_units_digit n

theorem count_valid_three_digit_numbers : finset.card ((finset.filter valid_number (finset.range 900)) + 100 = 216 :=
  sorry

end count_valid_three_digit_numbers_l112_112065


namespace P_n_at_one_l112_112662

noncomputable def P_n (n : ℕ) : (P_n : ℝ → ℝ) :=
classical.some (exists_polynomial_P_n n)

axiom exists_polynomial_P_n : ∀ (n : ℕ), ∃ (P_n : ℝ → ℝ), ∀ (x : ℝ), sin (n * x) = P_n (cos x) * sin x

theorem P_n_at_one (n : ℕ) : P_n n 1 = n :=
begin
  have h := classical.some_spec (exists_polynomial_P_n n),
  sorry
end

end P_n_at_one_l112_112662


namespace swimming_pool_volume_l112_112278

-- Define the characteristics of the swimming pool
def width : ℝ := 9
def length : ℝ := 12
def shallow_depth : ℝ := 1
def deep_depth : ℝ := 4

-- Define the area of the trapezoidal base using the given conditions
def trapezoid_area : ℝ :=
  (1 / 2) * (shallow_depth + deep_depth) * width

-- Define the volume of the swimming pool
def pool_volume : ℝ :=
  trapezoid_area * length

-- State the theorem to prove
theorem swimming_pool_volume : pool_volume = 270 :=
by
  -- Placeholder for the proof
  sorry

end swimming_pool_volume_l112_112278


namespace proof_C_l112_112994

variable {a b c : Type} [LinearOrder a] [LinearOrder b] [LinearOrder c]
variable {y : Type}

-- Definitions for parallel and perpendicular relationships
def parallel (x1 x2 : Type) : Prop := sorry
def perp (x1 x2 : Type) : Prop := sorry

theorem proof_C (a b c : Type) [LinearOrder a] [LinearOrder b] [LinearOrder c] (y : Type):
  (parallel a b ∧ parallel b c → parallel a c) ∧
  (perp a y ∧ perp b y → parallel a b) :=
by
  sorry

end proof_C_l112_112994


namespace log_bounds_f_l112_112145

noncomputable def f (n : ℕ) : ℕ :=
  -- f(n) is the smallest number of 1s needed to represent n using only 1s, + signs, × signs, and brackets.
  sorry

theorem log_bounds_f (n : ℕ) (h : n > 1) :
  3 * Real.log n ≤ Real.log 3 * f n ∧ Real.log 3 * f n ≤ 5 * Real.log n :=
begin
  sorry
end

end log_bounds_f_l112_112145


namespace stream_speed_l112_112311

-- Define the conditions
def still_water_speed : ℝ := 15
def upstream_time_factor : ℕ := 2

-- Define the theorem
theorem stream_speed (t v : ℝ) (h : (still_water_speed + v) * t = (still_water_speed - v) * (upstream_time_factor * t)) : v = 5 :=
by
  sorry

end stream_speed_l112_112311


namespace petya_coloring_l112_112283

theorem petya_coloring (n : ℕ) (h₁ : n = 100) (h₂ : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → 1 ≤ (number i j) ∧ (number i j) ≤ n * n) :
  ∃ k, k = 1 ∧ ∀ (initial_coloring : fin (n * n) → bool) (next_colorable : (fin (n * n) → bool) → (fin (n * n) → bool)),
    (∀ (table : fin n × fin n → fin (n * n)), next_colorable (λ a, initial_coloring a)
    (λ a, initial_coloring a) a) :=
begin
  sorry,
end

end petya_coloring_l112_112283


namespace total_number_of_plugs_l112_112899

variables (pairs_mittens pairs_plugs : ℕ)

-- Conditions
def initial_pairs_mittens : ℕ := 150
def initial_pairs_plugs : ℕ := initial_pairs_mittens + 20
def added_pairs_plugs : ℕ := 30
def total_pairs_plugs : ℕ := initial_pairs_plugs + added_pairs_plugs

-- The proposition we're going to prove:
theorem total_number_of_plugs : initial_pairs_mittens = 150 ∧ initial_pairs_plugs = initial_pairs_mittens + 20 ∧ added_pairs_plugs = 30 → 
  total_pairs_plugs * 2 = 400 := sorry

end total_number_of_plugs_l112_112899


namespace smallest_sum_of_4_numbers_l112_112317

noncomputable def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

noncomputable def not_relatively_prime (a b : ℕ) : Prop :=
  ¬ relatively_prime a b

noncomputable def problem_statement : Prop :=
  ∃ (V1 V2 V3 V4 : ℕ), 
  relatively_prime V1 V3 ∧ 
  relatively_prime V2 V4 ∧ 
  not_relatively_prime V1 V2 ∧ 
  not_relatively_prime V1 V4 ∧ 
  not_relatively_prime V2 V3 ∧ 
  not_relatively_prime V3 V4 ∧ 
  V1 + V2 + V3 + V4 = 60

theorem smallest_sum_of_4_numbers : problem_statement := sorry

end smallest_sum_of_4_numbers_l112_112317


namespace probability_x_squared_less_than_y_l112_112784

noncomputable def rectangle_probability : ℝ :=
  let area_under_curve := (∫ x in 0..real.sqrt 2, x^2) in
  let total_area := 5 * 2 in
  area_under_curve / total_area

-- Given the rectangle with vertices (0, 0), (5, 0), (5, 2), and (0, 2),
-- prove the probability that x^2 < y for a randomly picked point (x, y) is the expected probability.
theorem probability_x_squared_less_than_y :
  rectangle_probability = real.sqrt 2 / 15 :=
sorry

end probability_x_squared_less_than_y_l112_112784


namespace expand_polynomial_l112_112825

theorem expand_polynomial :
  (3 * x^2 + 2 * x + 1) * (2 * x^2 + 3 * x + 4) = 6 * x^4 + 13 * x^3 + 20 * x^2 + 11 * x + 4 :=
by
  sorry

end expand_polynomial_l112_112825


namespace find_A_max_perimeter_of_triangle_l112_112572

-- Definition of the given problem conditions
def triangle_condition (A B C : ℝ) : Prop :=
  sin(A)^2 - sin(B)^2 - sin(C)^2 = sin(B) * sin(C)

-- (1) Proving the value of A given the condition
theorem find_A (A B C : ℝ) (h : triangle_condition A B C) : A = 2 * π / 3 :=
by sorry

-- (2) Proving the maximum perimeter given BC = 3 and A = 2π/3
theorem max_perimeter_of_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (hA : 2 * π / 3 = 2 * π / 3) : 
  ∃ (P : ℝ), P = 3 + 2 * sqrt 3 :=
by sorry

end find_A_max_perimeter_of_triangle_l112_112572


namespace problem_1_and_2_l112_112593

-- Definition of the problem in Lean 4
theorem problem_1_and_2 (A B C : ℝ) (a b c : ℝ)
  (h1 : sin (A)^2 - sin (B)^2 - sin (C)^2 = sin (B) * sin (C))
  (h2 : BC = 3)
  (h3 : triangle ABC)
  (h4 : a = side_length (opposite A))
  (h5 : b = side_length (opposite B))
  (h6 : c = side_length (opposite C)) :
  A = 2 * real.pi / 3 ∧
  (a + b + c ≤ 3 + 2 * real.sqrt 3) :=
sorry

end problem_1_and_2_l112_112593


namespace find_side_b_of_triangle_l112_112629

theorem find_side_b_of_triangle
  (a b c : ℝ) 
  (area : ℝ)
  (angle_B : ℝ)
  (h1 : area = 2 * Real.sqrt 3)
  (h2 : angle_B = Real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 4 :=
by
  let B := angle_B
  have sin_B : Real.sin B = Real.sqrt 3 / 2 := by sorry
  have h_area : area = 1/2 * a * c * (Real.sin B) := by sorry
  have ac := sorry
  have law_of_cosines := sorry
  have b_val := sorry
  exact b_val

end find_side_b_of_triangle_l112_112629


namespace find_quadratic_polynomial_l112_112388

noncomputable def quadratic_polynomial : Polynomial ℝ :=
  3 * (X - C (2 + 2*I)) * (X - C (2 - 2*I))

theorem find_quadratic_polynomial :
  quadratic_polynomial = 3 * X^2 - 12 * X + 24 :=
by
  sorry

end find_quadratic_polynomial_l112_112388


namespace part1_part2_l112_112624

theorem part1 (A B C : ℝ) (h : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C) : A = 2 * π / 3 :=
sorry

theorem part2 (b c : ℝ) (h1 : sin (2 * π / 3) ^ 2 - sin b ^ 2 - sin c ^ 2 = sin b * sin c)
  (h2 : b + c = 2 * sqrt 3) : 3 * 2 + b + c = 3 + 2 * sqrt 3 :=
sorry

end part1_part2_l112_112624


namespace steve_average_speed_l112_112279

/-
Problem Statement:
Prove that the average speed of Steve's travel for the entire journey is 55 mph given the following conditions:
1. Steve's first part of journey: 5 hours at 40 mph.
2. Steve's second part of journey: 3 hours at 80 mph.
-/

theorem steve_average_speed :
  let time1 := 5 -- hours
  let speed1 := 40 -- mph
  let time2 := 3 -- hours
  let speed2 := 80 -- mph
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 55 := by
  sorry

end steve_average_speed_l112_112279


namespace sequences_formula_sum_c_k_sum_inverse_a_k_cubed_l112_112459

open Nat

def a : ℕ → ℕ := λ n, n
def b : ℕ → ℕ := λ n, 2^n
def c (n : ℕ) : ℕ := 
  if n % 2 = 0 then 
    a (2 * (n / 2) - 1) 
  else 
    ((3 * a (n / 2) - 2) * b (n / 2) - 2) / ((b (n / 2) + 1) * (b ((n / 2) + 2) + 1))

theorem sequences_formula (n : ℕ) :
  (∀ n, 2 * (a (n+1)) = a n + a (n+2)) →
  (∀ n, (b (n+1))^2 = b n * b (n+2)) →
  2 * a 1 = b 1 = 2 →
  a 4 = b 2 →
  b 5 = 4 * b 3 →
  a n = n ∧ b n = 2^n := 
sorry

theorem sum_c_k (n : ℕ) :
  (∀ n, 2 * (a (n+1)) = a n + a (n+2)) →
  (∀ n, (b (n+1))^2 = b n * b (n+2)) →
  2 * a 1 = b 1 = 2 →
  a 4 = b 2 →
  b 5 = 4 * b 3 →
  (∑ k in range (2*n+1), c k) = (2*n+1)*(n+1) + (2/5 - (2*n+2)/(1+2^(2*n+2))) := 
sorry

theorem sum_inverse_a_k_cubed (n : ℕ) :
  (∀ n, 2 * (a (n+1)) = a n + a (n+2)) →
  (∀ n, (b (n+1))^2 = b n * b (n+2)) →
  2 * a 1 = b 1 = 2 →
  a 4 = b 2 →
  b 5 = 4 * b 3 →
  (∑ k in range n, 1 / (a k)^3) < 5 / 4 := 
sorry

end sequences_formula_sum_c_k_sum_inverse_a_k_cubed_l112_112459


namespace evaluate_expression_l112_112823

lemma pow_mod_four_cycle (n : ℕ) : (n % 4) = 1 → (i : ℂ)^n = i :=
by sorry

lemma pow_mod_four_cycle2 (n : ℕ) : (n % 4) = 2 → (i : ℂ)^n = -1 :=
by sorry

lemma pow_mod_four_cycle3 (n : ℕ) : (n % 4) = 3 → (i : ℂ)^n = -i :=
by sorry

lemma pow_mod_four_cycle4 (n : ℕ) : (n % 4) = 0 → (i : ℂ)^n = 1 :=
by sorry

theorem evaluate_expression : 
  (i : ℂ)^(2021) + (i : ℂ)^(2022) + (i : ℂ)^(2023) + (i : ℂ)^(2024) = 0 :=
by sorry

end evaluate_expression_l112_112823


namespace base9_reverse_base13_sum_l112_112836

-- Define the condition for a number whose base9 representation is the reverse of its base13 representation
def is_base9_reverse_base13 (n : ℕ) : Prop :=
  let base9_digits := n.digits 9
  let base13_digits := n.digits 13
  base9_digits = base13_digits.reverse

-- Define the sum function for numbers satisfying the condition up to a given upper bound
def sum_base9_reverse_base13 (upper_bound : ℕ) : ℕ :=
  (List.range (upper_bound + 1)).filter is_base9_reverse_base13 |>.sum id

-- The final theorem statement asserting the correct sum for up to 8
theorem base9_reverse_base13_sum : sum_base9_reverse_base13 8 = 36 := by
  sorry

end base9_reverse_base13_sum_l112_112836


namespace maximum_a3_S10_l112_112958

-- Given definitions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def conditions (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ (∀ n, a n > 0) ∧ (a 1 + a 3 + a 8 = a 4 ^ 2)

-- The problem statement
theorem maximum_a3_S10 (a : ℕ → ℝ) (h : conditions a) : 
  (∃ S : ℝ, S = a 3 * ((10 / 2) * (a 1 + a 10)) ∧ S ≤ 375 / 4) :=
sorry

end maximum_a3_S10_l112_112958


namespace cats_left_in_store_l112_112783

theorem cats_left_in_store 
  (initial_siamese : ℕ := 25)
  (initial_persian : ℕ := 18)
  (initial_house : ℕ := 12)
  (initial_maine_coon : ℕ := 10)
  (sold_siamese : ℕ := 6)
  (sold_persian : ℕ := 4)
  (sold_maine_coon : ℕ := 3)
  (sold_house : ℕ := 0)
  (remaining_siamese : ℕ := 19)
  (remaining_persian : ℕ := 14)
  (remaining_house : ℕ := 12)
  (remaining_maine_coon : ℕ := 7) : 
  initial_siamese - sold_siamese = remaining_siamese ∧
  initial_persian - sold_persian = remaining_persian ∧
  initial_house - sold_house = remaining_house ∧
  initial_maine_coon - sold_maine_coon = remaining_maine_coon :=
by sorry

end cats_left_in_store_l112_112783


namespace triangle_A_value_and_max_perimeter_l112_112576

theorem triangle_A_value_and_max_perimeter (A B C a b c : ℝ) 
  (h1 : sin A ^ 2 - sin B ^ 2 - sin C ^ 2 = sin B * sin C)
  (h2 : a = 3) :
  (A = 2 * Real.pi / 3) ∧ (a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by
  sorry

end triangle_A_value_and_max_perimeter_l112_112576


namespace frustum_volume_l112_112526

noncomputable def volume_of_frustum (r R h : ℝ) : ℝ :=
  (π * h * (R^2 + R*r + r^2)) / 3

theorem frustum_volume { (h l : ℝ) (A_lateral : ℝ) } :
  h = 4 → l = 5 → A_lateral = 45 * π → volume_of_frustum 3 6 4 = 84 * π :=
by
  intros h_eq l_eq A_lateral_eq
  rw [h_eq, l_eq, A_lateral_eq]
  sorry

end frustum_volume_l112_112526


namespace jackies_free_time_l112_112971

-- Define the conditions
def hours_working : ℕ := 8
def hours_sleeping : ℕ := 8
def hours_exercising : ℕ := 3
def total_hours_in_day : ℕ := 24

-- The statement to be proven
theorem jackies_free_time : total_hours_in_day - (hours_working + hours_sleeping + hours_exercising) = 5 :=
by 
  rw [total_hours_in_day, hours_working, hours_sleeping, hours_exercising]
  -- 24 - (8 + 8 + 3) = 5
  sorry

end jackies_free_time_l112_112971


namespace part_a_part_b_l112_112288

theorem part_a (a : ℤ) (k : ℤ) (h : a + 1 = 3 * k) : ∃ m : ℤ, 4 + 7 * a = 3 * m := by
  sorry

theorem part_b (a b : ℤ) (m n : ℤ) (h1 : 2 + a = 11 * m) (h2 : 35 - b = 11 * n) : ∃ p : ℤ, a + b = 11 * p := by
  sorry

end part_a_part_b_l112_112288


namespace contrapositive_proof_l112_112214

theorem contrapositive_proof (x m : ℝ) :
  (m < 0 → (∃ r : ℝ, r * r + 3 * r + m = 0)) ↔
  (¬ (∃ r : ℝ, r * r + 3 * r + m = 0) → m ≥ 0) :=
by
  sorry

end contrapositive_proof_l112_112214


namespace find_smallest_x_l112_112424

noncomputable def smallest_x : ℝ :=
  min { x : ℝ | 0 < x ∧ (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * x = 8 }

theorem find_smallest_x :
  smallest_x = 89 / 9 :=
by
  sorry

end find_smallest_x_l112_112424


namespace john_score_l112_112980

theorem john_score (s1 s2 s3 s4 s5 s6 : ℕ) (h1 : s1 = 85) (h2 : s2 = 88) (h3 : s3 = 90) (h4 : s4 = 92) (h5 : s5 = 83) (h6 : s6 = 102) :
  (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 90 :=
by
  sorry

end john_score_l112_112980


namespace opposite_of_neg_two_thirds_l112_112232

theorem opposite_of_neg_two_thirds : -(- (2 / 3)) = (2 / 3) :=
by
  sorry

end opposite_of_neg_two_thirds_l112_112232


namespace find_x_l112_112819

theorem find_x (x : ℕ) (h : 27^3 + 27^3 + 27^3 + 27^3 = 3^x) : x = 11 :=
sorry

end find_x_l112_112819


namespace prove_periodicity_of_f_l112_112691

def periodic_function {R : Type*} [AddGroup R] (f : R → R) : Prop :=
  ∃ T : R, ∀ x, f (x + T) = f x

theorem prove_periodicity_of_f (f : ℝ → ℝ)
  (h1 : ∀ x, f(x + 2) = f(2 - x))
  (h2 : ∀ x, f(x + 7) = f(7 - x)) :
  periodic_function f :=
sorry

end prove_periodicity_of_f_l112_112691


namespace increasing_interval_sin_cos_l112_112747

theorem increasing_interval_sin_cos : 
  ∀ x, x ∈ Icc (-π/2) 0 → 0 < real.sin' x ∧ 0 < real.cos' x := 
by 
  sorry

end increasing_interval_sin_cos_l112_112747


namespace sum_of_super_cool_rectangles_l112_112318

theorem sum_of_super_cool_rectangles :
  let is_super_cool (a b : ℕ) := (a * b = 6 * (a + b))
  let areas := {ab | ∃ a b : ℕ, a ≠ b ∧ is_super_cool a b ∧ ab = a * b}
  let sum_of_areas := areas.sum id in
  sum_of_areas = 942 :=
by {
  let is_super_cool := λ a b : ℕ, a * b = 6 * (a + b),
  let areas := finset.insert 0 (finset.univ.filter (λ ab, ∃ a b : ℕ, a ≠ b ∧ a * b = 6 * (a + b) ∧ ab = a * b)),
  let sum_of_areas := areas.sum id,
  have h: sum_of_areas = 942 := sorry,
  exact h,
}

end sum_of_super_cool_rectangles_l112_112318


namespace find_angle_A_max_perimeter_l112_112557

noncomputable def sin_sq_minus (A B C : ℝ) : ℝ :=
  (Real.sin A) * (Real.sin A) - (Real.sin B) * (Real.sin B) - (Real.sin C) * (Real.sin C)

noncomputable def sin_prod (B C : ℝ) : ℝ :=
  (Real.sin B) * (Real.sin C)

theorem find_angle_A (A B C : ℝ) (h : sin_sq_minus A B C = sin_prod B C) :
  A = 2 * Real.pi / 3 :=
by
  sorry

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem max_perimeter (B C : ℝ) (a : ℝ) (h1 : a = 3) (h2 : B + C = Real.pi / 3) :
  ∃ (b c : ℝ), perimeter a b c = 3 + 2 * Real.sqrt 3 :=
by
  sorry

end find_angle_A_max_perimeter_l112_112557


namespace ratio_areas_l112_112772

variables (AD AB : ℝ)
variables (r : ℝ := 18) -- radius of semicircles
variables (A_rectangle A_semicircles : ℝ)

-- Condition: AB = 36
axiom AB_is_36 : AB = 36

-- Condition: AD/AB = 4/3
axiom ratio_AD_AB : AD / AB = 4 / 3

-- Definition of areas based on given conditions
def area_rectangle (AD AB : ℝ) : ℝ := AD * AB
def area_semicircles (r : ℝ) : ℝ := π * r^2

-- Calculate the areas
axiom area_rect_def : A_rectangle = area_rectangle AD AB
axiom area_semicircle_def : A_semicircles = area_semicircles r

-- Theorem to be proved
theorem ratio_areas : (A_rectangle / A_semicircles) = 16 / (3 * π) :=
by sorry

end ratio_areas_l112_112772


namespace modulus_of_z_l112_112487

theorem modulus_of_z (z : ℂ) (h : (1 + complex.i) * z = complex.i) : complex.abs z = real.sqrt 2 / 2 := 
sorry

end modulus_of_z_l112_112487


namespace cube_surface_area_l112_112793

/-- Consider a wooden cube with edge length 5 meters and square holes of side length 2 meters centered in each face, cutting through to the opposite face. Prove that the total surface area, including the inside surfaces, is 222 square meters. -/
theorem cube_surface_area (edge_length : ℝ) (hole_side : ℝ)
  (h_edge_length : edge_length = 5)
  (h_hole_side : hole_side = 2) :
  let original_surface_area := 6 * edge_length^2 in
  let hole_area := 6 * hole_side^2 in
  let exposed_area := 6 * 4 * hole_side^2 in
  (original_surface_area - hole_area + exposed_area) = 222 :=
by {
  sorry
}

end cube_surface_area_l112_112793


namespace sum_a_n_first_10_eq_2047_l112_112895

noncomputable def a_n (n : ℕ) : ℕ :=
  max (n^2) (2^n)

def sum_first_10_terms : ℕ :=
  (Finset.range 10).sum (λ n, a_n (n+1))

theorem sum_a_n_first_10_eq_2047 : sum_first_10_terms = 2047 :=
  sorry

end sum_a_n_first_10_eq_2047_l112_112895


namespace monotonic_decreasing_interval_l112_112696

noncomputable def xlnx (x : ℝ) : ℝ := x * Real.log x

theorem monotonic_decreasing_interval : 
  ∀ x, (0 < x) ∧ (x < 5) → (Real.log x + 1 < 0) ↔ (0 < x) ∧ (x < 1 / Real.exp 1) := 
by
  sorry

end monotonic_decreasing_interval_l112_112696


namespace zero_not_in_range_of_g_l112_112996

def g : ℝ → ℤ
| x => if x > -3 then Int.ceil (1 / (x + 3)) else Int.floor (1 / (x + 3))

theorem zero_not_in_range_of_g : ¬ ∃ x : ℝ, g x = 0 := by
  sorry

end zero_not_in_range_of_g_l112_112996


namespace goat_cow_difference_l112_112900

-- Given the number of pigs (P), cows (C), and goats (G) on a farm
variables (P C G : ℕ)

-- Conditions:
def pig_count := P = 10
def cow_count_relationship := C = 2 * P - 3
def total_animals := P + C + G = 50

-- Theorem: The difference between the number of goats and cows
theorem goat_cow_difference (h1 : pig_count P)
                           (h2 : cow_count_relationship P C)
                           (h3 : total_animals P C G) :
  G - C = 6 := 
  sorry

end goat_cow_difference_l112_112900


namespace find_x_eq_7714285714285714_l112_112382

theorem find_x_eq_7714285714285714 (x : ℝ) (hx_pos : 0 < x) (h : floor x * x = 54) : x = 54 / 7 :=
by
  sorry

end find_x_eq_7714285714285714_l112_112382


namespace find_angle_A_max_perimeter_l112_112564

noncomputable def sin_sq_minus (A B C : ℝ) : ℝ :=
  (Real.sin A) * (Real.sin A) - (Real.sin B) * (Real.sin B) - (Real.sin C) * (Real.sin C)

noncomputable def sin_prod (B C : ℝ) : ℝ :=
  (Real.sin B) * (Real.sin C)

theorem find_angle_A (A B C : ℝ) (h : sin_sq_minus A B C = sin_prod B C) :
  A = 2 * Real.pi / 3 :=
by
  sorry

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem max_perimeter (B C : ℝ) (a : ℝ) (h1 : a = 3) (h2 : B + C = Real.pi / 3) :
  ∃ (b c : ℝ), perimeter a b c = 3 + 2 * Real.sqrt 3 :=
by
  sorry

end find_angle_A_max_perimeter_l112_112564


namespace volume_of_dodecagonal_pyramid_l112_112533

-- Given definitions and conditions of the problem
variables (S α : ℝ)

-- Define the volume of the pyramid
def volume_of_pyramid (S α : ℝ) : ℝ := 
  (4 * S * real.sqrt (3 ^ (1 / 4)) * real.sin α * real.sqrt (S * real.cos α)) / 3

-- Statement to prove
theorem volume_of_dodecagonal_pyramid :
  volume_of_pyramid S α = (4 * S * real.sqrt (3 ^ (1 / 4)) * real.sin α * real.sqrt (S * real.cos α)) / 3 :=
by
  sorry

end volume_of_dodecagonal_pyramid_l112_112533


namespace find_A_max_perimeter_of_triangle_l112_112570

-- Definition of the given problem conditions
def triangle_condition (A B C : ℝ) : Prop :=
  sin(A)^2 - sin(B)^2 - sin(C)^2 = sin(B) * sin(C)

-- (1) Proving the value of A given the condition
theorem find_A (A B C : ℝ) (h : triangle_condition A B C) : A = 2 * π / 3 :=
by sorry

-- (2) Proving the maximum perimeter given BC = 3 and A = 2π/3
theorem max_perimeter_of_triangle (B C : ℝ) (BC : ℝ) (hBC : BC = 3) (hA : 2 * π / 3 = 2 * π / 3) : 
  ∃ (P : ℝ), P = 3 + 2 * sqrt 3 :=
by sorry

end find_A_max_perimeter_of_triangle_l112_112570


namespace part1_part2_l112_112018

-- Define sequence a_n according to given conditions
def a : ℕ → ℝ
| 0     := 100
| (n+1) := (a n) ^ 2

-- Define sequence b_n based on sequence a_n according to conditions
def b (n : ℕ) : ℝ := log a n

-- Part 1: Prove the general formula for sequence b_n
theorem part1 (n : ℕ) : b n = 2 ^ n := 
sorry

-- Define sequence c_n as given
def c (n : ℕ) : ℝ := ∑ i in finset.range (2*n + 1), log 2 (b (n + i))

-- Define sequence S_n, the sum of the first n terms of the sequence {1/c_n}
def S (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / c (i + 1)

-- Part 2: Prove the sum S_n of the first n terms of the sequence {1 / c_n}
theorem part2 (n : ℕ) : S n = 2 * n / (3 * (n + 1)) := 
sorry

end part1_part2_l112_112018


namespace sum_super_cool_rectangles_areas_l112_112326

theorem sum_super_cool_rectangles_areas : 
  ∃ (s : finset ℕ), 
    (∀ a b ∈ s, a * b = 6 * (a + b) ∧ a * b = 942) → 
    ∑ n in s, n = 942 := 
sorry

end sum_super_cool_rectangles_areas_l112_112326


namespace not_equal_to_one_l112_112656

-- Define x and y with given conditions
def x : ℂ := (-1 + complex.i * sqrt 3) / 2
def y : ℂ := (-1 - complex.i * sqrt 3) / 2

-- Define the question as a theorem
theorem not_equal_to_one : 
  (x ^ 6 + y ^ 6 ≠ 1) ∧ 
  (x ^ 12 + y ^ 12 ≠ 1) ∧ 
  (x ^ 18 + y ^ 18 ≠ 1) ∧ 
  (x ^ 24 + y ^ 24 ≠ 1) ∧ 
  (x ^ 30 + y ^ 30 ≠ 1) :=
by {
  -- The proof will go here
  sorry
}

end not_equal_to_one_l112_112656


namespace total_distance_of_trip_l112_112137

theorem total_distance_of_trip (x : ℚ)
  (highway : x / 4 ≤ x)
  (city : 30 ≤ x)
  (country : x / 6 ≤ x)
  (middle_part_fraction : 1 - 1 / 4 - 1 / 6 = 7 / 12) :
  (7 / 12) * x = 30 → x = 360 / 7 :=
by
  sorry

end total_distance_of_trip_l112_112137


namespace probability_of_drawing_1_boy_1_girl_l112_112697

theorem probability_of_drawing_1_boy_1_girl 
  (total_boys : ℕ) (total_girls : ℕ) (choose_two : ℕ) 
  (choose_boy_girl : ℕ) :
  total_boys = 3 → total_girls = 2 → choose_two = Nat.choose 5 2 → choose_boy_girl = Nat.choose 3 1 * Nat.choose 2 1 →
  (choose_boy_girl : ℚ) / choose_two = (3 / 5 : ℚ) :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  simp,
  norm_num,
end

end probability_of_drawing_1_boy_1_girl_l112_112697


namespace bisects_AC_implies_bisects_BD_l112_112935

/-- Given a convex quadrilateral ABCD with points P and Q being the midpoints of sides AB and CD respectively,
    and given that the line segment PQ bisects the diagonal AC, prove that PQ also bisects the diagonal BD. -/
theorem bisects_AC_implies_bisects_BD
    (A B C D P Q M N : Point)
    (hP : midpoint A B P)
    (hQ : midpoint C D Q)
    (hM : midpoint A C M)
    (hN : midpoint B D N)
    (hPQ_bisects_AC : lies_on_line M (line_through P Q))
    : lies_on_line N (line_through P Q) :=
sorry

end bisects_AC_implies_bisects_BD_l112_112935


namespace solve_for_x_l112_112680

theorem solve_for_x (x : ℝ) (h : (3 / 4) + (1 / x) = 7 / 8) : x = 8 :=
sorry

end solve_for_x_l112_112680


namespace divisibility_by_11_l112_112985

theorem divisibility_by_11
  (n : ℕ) (hn : n ≥ 2)
  (h : (n^2 + (4^n) + (7^n)) % n = 0) :
  (n^2 + 4^n + 7^n) % 11 = 0 := 
by
  sorry

end divisibility_by_11_l112_112985


namespace smallest_positive_real_number_l112_112407

noncomputable def smallest_x : ℝ :=
  Inf {x : ℝ | x > 0 ∧ (floor (x^2) - x * floor x = 8)}

theorem smallest_positive_real_number :
  smallest_x = 89 / 9 :=
by 
  sorry

end smallest_positive_real_number_l112_112407


namespace midpoint_trajectory_distance_range_l112_112894

theorem midpoint_trajectory (x y: ℝ) (x₁ x₂: ℝ)
  (hA : y = (√2 / 2) * x₁)
  (hB : y = -(√2 / 2) * x₂)
  (h_length : (2 * (√2) * y)^2 + ((√2) * x)^2 = 8) :
  (x^2 / 4) + y^2 = 1 :=
sorry

theorem distance_range (k m: ℝ) 
  (hk₀: 1 / 20 < k^2)
  (hk₁: k^2 ≤ 5 / 4)
  (hkm: m^2 + k^2 = 5 / 4) :
  0 ≤ (m^2 / (1 + k^2)) ∧ (m^2 / (1 + k^2) < 8 / 7) :=
sorry

end midpoint_trajectory_distance_range_l112_112894


namespace power_half_mod_prime_l112_112187

-- Definitions of odd prime and coprime condition
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1
def coprime (a p : ℕ) : Prop := Nat.gcd a p = 1

-- Main statement
theorem power_half_mod_prime (p a : ℕ) (hp : is_odd_prime p) (ha : coprime a p) :
  a ^ ((p - 1) / 2) % p = 1 ∨ a ^ ((p - 1) / 2) % p = p - 1 := 
  sorry

end power_half_mod_prime_l112_112187


namespace linear_function_result_l112_112647

variable {R : Type*} [LinearOrderedField R]

noncomputable def linear_function (g : R → R) : Prop :=
  ∃ (a b : R), ∀ x, g x = a * x + b

theorem linear_function_result (g : R → R) (h_lin : linear_function g) (h : g 5 - g 1 = 16) : g 13 - g 1 = 48 :=
  by
  sorry

end linear_function_result_l112_112647


namespace bread_problem_l112_112753

variable (x : ℝ)

theorem bread_problem (h1 : x > 0) :
  (15 / x) - 1 = 14 / (x + 2) :=
sorry

end bread_problem_l112_112753


namespace complex_symmetry_l112_112959

theorem complex_symmetry (z1 z2 : ℂ) (i : ℂ) [imaginary_unit : i * i = -1] 
  (h1 : z1 = 1 - 2 * i) (h2 : z2.re = -z1.re) (h3 : z2.im = z1.im) : 
  z2 = -1 - 2 * i := 
sorry

end complex_symmetry_l112_112959


namespace split_cube_l112_112842

theorem split_cube (m : ℕ) (h1 : m > 1) (h2 : ∃ k, 1 ≤ k ∧ k < m ∧ m^3 = (List.range k).map (λ n => 2 * n + 3 * k - m)).sum ∧ 31 ∈ (List.range k).map (λ n => 2 * n + 3 * k - m)) :
  m = 6 :=
by
  sorry

end split_cube_l112_112842


namespace union_of_M_and_N_l112_112893

noncomputable def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}
noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def compl_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

theorem union_of_M_and_N :
  M ∪ N = {x | -3 ≤ x ∧ x < 1} :=
sorry

end union_of_M_and_N_l112_112893


namespace ce_equals_half_ba_plus_ac_l112_112142

theorem ce_equals_half_ba_plus_ac {A B C D E : Point} (h1 : is_midpoint_of_arc D A B C)
    (h2 : foot_perp D AC E) : dist C E = (dist B A + dist A C) / 2 := 
sorry

end ce_equals_half_ba_plus_ac_l112_112142


namespace leftmost_three_digits_eq_317_l112_112021

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def num_arrangements (total_rings rings_selected fingers : ℕ) : ℕ :=
  binomial total_rings rings_selected * Nat.factorial rings_selected * binomial (rings_selected + fingers - 1) fingers

theorem leftmost_three_digits_eq_317 :
  let n := num_arrangements 10 6 5 in
  (n / 1000) % 1000 = 317 := by
  sorry

end leftmost_three_digits_eq_317_l112_112021


namespace n_63_non_jumpable_l112_112309

def is_jumpable (n : ℕ) : Prop :=
  ∃ (start : ℕ), (∀ (visit : ℕ → bool), 
    (∀ (jump : ℕ), jump = 8 ∨ jump = 9 ∨ jump = 10 → 
      0 ≤ start + jump * visit jump ∧ start + jump * visit jump < n) →
      (∀ i, 0 ≤ i ∧ i < n → visit i = tt))

theorem n_63_non_jumpable : ¬ is_jumpable 63 :=
by {
  sorry
}

end n_63_non_jumpable_l112_112309


namespace smallest_positive_x_l112_112415

theorem smallest_positive_x (x : ℝ) (h : ⌊x^2⌋ - x * ⌊x⌋ = 8) : x = 89 / 9 :=
sorry

end smallest_positive_x_l112_112415


namespace min_value_x_squared_plus_10x_l112_112739

theorem min_value_x_squared_plus_10x : ∃ x : ℝ, (x^2 + 10 * x) = -25 :=
by {
  sorry
}

end min_value_x_squared_plus_10x_l112_112739


namespace smallest_positive_real_number_l112_112408

noncomputable def smallest_x : ℝ :=
  Inf {x : ℝ | x > 0 ∧ (floor (x^2) - x * floor x = 8)}

theorem smallest_positive_real_number :
  smallest_x = 89 / 9 :=
by 
  sorry

end smallest_positive_real_number_l112_112408


namespace min_value_of_sum_l112_112059

theorem min_value_of_sum (x y : ℝ) (h : (4 * (x - 1) + 2 * y = 0)) : 4^x + 2^y ≥ 4 :=
sorry

end min_value_of_sum_l112_112059


namespace xiaohui_pe_score_l112_112921

-- Define the conditions
def morning_score : ℝ := 95
def midterm_score : ℝ := 90
def final_score : ℝ := 85

def morning_weight : ℝ := 0.2
def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.5

-- The problem is to prove that Xiaohui's physical education score for the semester is 88.5 points.
theorem xiaohui_pe_score :
  morning_score * morning_weight +
  midterm_score * midterm_weight +
  final_score * final_weight = 88.5 :=
by
  sorry

end xiaohui_pe_score_l112_112921


namespace fraction_of_total_l112_112756

def total_amount : ℝ := 5000
def r_amount : ℝ := 2000.0000000000002

theorem fraction_of_total
  (h1 : r_amount = 2000.0000000000002)
  (h2 : total_amount = 5000) :
  r_amount / total_amount = 0.40000000000000004 :=
by
  -- The proof is skipped
  sorry

end fraction_of_total_l112_112756


namespace jasmine_coffee_beans_purchase_l112_112138

theorem jasmine_coffee_beans_purchase (x : ℝ) (coffee_cost per_pound milk_cost per_gallon total_cost : ℝ)
  (h1 : coffee_cost = 2.50)
  (h2 : milk_cost = 3.50)
  (h3 : total_cost = 17)
  (h4 : milk_purchased = 2)
  (h_equation : coffee_cost * x + milk_cost * milk_purchased = total_cost) :
  x = 4 :=
by
  sorry

end jasmine_coffee_beans_purchase_l112_112138


namespace john_money_left_l112_112633

-- Given definitions
def drink_cost (q : ℝ) := q
def small_pizza_cost (q : ℝ) := q
def large_pizza_cost (q : ℝ) := 4 * q
def initial_amount := 50

-- Problem statement
theorem john_money_left (q : ℝ) : initial_amount - (4 * drink_cost q + 2 * small_pizza_cost q + large_pizza_cost q) = 50 - 10 * q :=
by
  sorry

end john_money_left_l112_112633


namespace floor_x_mul_x_eq_54_l112_112380

def positive_real (x : ℝ) : Prop := x > 0

theorem floor_x_mul_x_eq_54 (x : ℝ) (h_pos : positive_real x) : ⌊x⌋ * x = 54 ↔ x = 54 / 7 :=
by
  sorry

end floor_x_mul_x_eq_54_l112_112380


namespace sin_B_equals_one_third_l112_112920

variable {A B : ℝ}
variable {a b : ℝ}
variable {sin : ℝ → ℝ}

/-- Given conditions in the problem --/
def triangle_condition (a b : ℝ) (A : ℝ) :=
a = 3 * b * sin A

/-- Mathematical proposition to prove the equivalent statement --/
theorem sin_B_equals_one_third 
  (h : triangle_condition a b A) 
  (ha : a = 3 * b * sin A)
  (h_b_neq_zero : b ≠ 0)
  (h_sin_A_non_zero : sin A ≠ 0) :
  sin B = 1 / 3 :=
sorry

end sin_B_equals_one_third_l112_112920


namespace range_a_minus_b_l112_112878

theorem range_a_minus_b (a b : ℝ) (h_pos : a > 0) (h_roots : ∃ x₁ x₂ : ℝ, (ax² + bx - 1 = 0) ∧ x₁ ≠ x₂) (h_interval : ∃ x : ℝ, x ∈ (1, 2) ∧ (ax² + bx - 1 = 0)) :
  -1 < a - b ∧ a - b < 1 := sorry

end range_a_minus_b_l112_112878


namespace monotonicity_F_range_k_l112_112493

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - Real.log (1 - x)
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x + a * x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x - k * (x^3 - 3 * x)

theorem monotonicity_F (a : ℝ) (ha : a ≠ 0) :
(∀ x : ℝ, (-1 < x ∧ x < 1) → 
    (if (-2 ≤ a ∧ a < 0) ∨ (a > 0) then 0 ≤ (a - a * x^2 + 2) / (1 - x^2)
     else if a < -2 then 
        ((-1 < x ∧ x < -Real.sqrt ((a + 2) / a)) ∨ (Real.sqrt ((a + 2) / a) < x ∧ x < 1)) → 0 ≤ (a - a * x^2 + 2) / (1 - x^2) ∧ 
        (-Real.sqrt ((a + 2) / a) < x ∧ x < Real.sqrt ((a + 2) / a)) → 0 > (a - a * x^2 + 2) / (1 - x^2)
    else false)) :=
sorry

theorem range_k (k : ℝ) (hk : ∀ x : ℝ, (0 < x ∧ x < 1) → f x > k * (x^3 - 3 * x)) :
k ≥ -2 / 3 :=
sorry

end monotonicity_F_range_k_l112_112493


namespace find_angle_A_max_perimeter_l112_112558

noncomputable def sin_sq_minus (A B C : ℝ) : ℝ :=
  (Real.sin A) * (Real.sin A) - (Real.sin B) * (Real.sin B) - (Real.sin C) * (Real.sin C)

noncomputable def sin_prod (B C : ℝ) : ℝ :=
  (Real.sin B) * (Real.sin C)

theorem find_angle_A (A B C : ℝ) (h : sin_sq_minus A B C = sin_prod B C) :
  A = 2 * Real.pi / 3 :=
by
  sorry

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem max_perimeter (B C : ℝ) (a : ℝ) (h1 : a = 3) (h2 : B + C = Real.pi / 3) :
  ∃ (b c : ℝ), perimeter a b c = 3 + 2 * Real.sqrt 3 :=
by
  sorry

end find_angle_A_max_perimeter_l112_112558


namespace floor_x_mul_x_eq_54_l112_112381

def positive_real (x : ℝ) : Prop := x > 0

theorem floor_x_mul_x_eq_54 (x : ℝ) (h_pos : positive_real x) : ⌊x⌋ * x = 54 ↔ x = 54 / 7 :=
by
  sorry

end floor_x_mul_x_eq_54_l112_112381


namespace trapezoid_CD_length_l112_112254

theorem trapezoid_CD_length (AD BC BD : ℝ) (D B : ℝ)
  (h1 : AD * BC = 7 * 4) 
  (h2 : BD = 3) 
  (h3 : ∠DBA = 30) 
  (h4 : ∠BDC = 60): CD = 9 / 4 :=
by
  sorry

end trapezoid_CD_length_l112_112254


namespace polar_line_equation_l112_112295

theorem polar_line_equation (r θ: ℝ) (p : r = 3 ∧ θ = 0) : r = 3 := 
by 
  sorry

end polar_line_equation_l112_112295


namespace find_original_height_l112_112299

noncomputable def original_height : ℝ := by
  let H := 102.19
  sorry

lemma ball_rebound (H : ℝ) : 
  (H + 2 * 0.8 * H + 2 * 0.56 * H + 2 * 0.336 * H + 2 * 0.168 * H + 2 * 0.0672 * H + 2 * 0.02016 * H = 500) :=
by
  sorry

theorem find_original_height : original_height = 102.19 :=
by
  have h := ball_rebound original_height
  sorry

end find_original_height_l112_112299


namespace p_necessary_not_sufficient_q_l112_112849

theorem p_necessary_not_sufficient_q (x : ℝ) (p : -3 < x ∧ x < 7) (q : 0 < x ∧ x < 7) :
  (q → p) ∧ ¬(p → q) :=
by sorry

end p_necessary_not_sufficient_q_l112_112849


namespace fraction_of_students_received_As_l112_112110

/-- Assume A is the fraction of students who received A's,
and B is the fraction of students who received B's,
and T is the total fraction of students who received either A's or B's. -/
theorem fraction_of_students_received_As
  (A B T : ℝ)
  (hB : B = 0.2)
  (hT : T = 0.9)
  (h : A + B = T) :
  A = 0.7 := 
by
  -- establishing the proof steps
  sorry

end fraction_of_students_received_As_l112_112110


namespace range_of_m_l112_112439

open Set

theorem range_of_m (m : ℝ) :
  (∃ f : ℤ → Prop, (∀ x, f x ↔ x + 5 > 0 ∧ x - m ≤ 1) ∧ (∃ a b c : ℤ, f a ∧ f b ∧ f c))
  → (-3 ≤ m ∧ m < -2) := 
sorry

end range_of_m_l112_112439


namespace sum_super_cool_rectangles_areas_l112_112324

theorem sum_super_cool_rectangles_areas : 
  ∃ (s : finset ℕ), 
    (∀ a b ∈ s, a * b = 6 * (a + b) ∧ a * b = 942) → 
    ∑ n in s, n = 942 := 
sorry

end sum_super_cool_rectangles_areas_l112_112324


namespace probability_last_box_single_ball_l112_112089

noncomputable def probability_last_box_only_ball (n : ℕ) : ℝ :=
  let p := 1 / n in
  (n * p * (1 - p)^(n - 1))

theorem probability_last_box_single_ball :
  probability_last_box_only_ball 100 ≈ 0.370 :=
by
  sorry

end probability_last_box_single_ball_l112_112089


namespace translated_line_expression_l112_112795

theorem translated_line_expression (x y : ℝ) (b : ℝ) : 
  (∀ x y, y = 2 * x + b ∧ (2, 5)) → y = 2x + 1 :=
by
  intro hx hb
  -- Define the original line y = 2x
  assume original_line : ∀ x : ℝ, y = 2 * x,
  assume translation_condition: ∃ (x : ℝ) (y : ℝ), y = 2 * x + b ∧ (x = 2 ∧ y = 5),
  -- Using the given conditions
  have h_translation: ∃ (b : ℝ), 5 = 2*2 + b,
    from hx 2 5,
  -- From h_translation, we can solve for b
  have hb_sol: b = 1,
  -- With b = 1, we derive the new line equation
  show y = 2 * x + 1,
  from sorry

end translated_line_expression_l112_112795


namespace gain_percent_l112_112757

variable (MP CP SP : ℝ)

def costPrice (CP : ℝ) (MP : ℝ) := CP = 0.64 * MP

def sellingPrice (SP : ℝ) (MP : ℝ) := SP = MP * 0.88

theorem gain_percent (h1 : costPrice CP MP) (h2 : sellingPrice SP MP) : 
  ((SP - CP) / CP) * 100 = 37.5 :=
by
  sorry

end gain_percent_l112_112757


namespace and_implies_or_or_does_not_imply_and_and_is_sufficient_but_not_necessary_for_or_l112_112290

theorem and_implies_or (p q : Prop) (hpq : p ∧ q) : p ∨ q :=
by {
  sorry
}

theorem or_does_not_imply_and (p q : Prop) (hp_or_q : p ∨ q) : ¬ (p ∧ q) :=
by {
  sorry
}

theorem and_is_sufficient_but_not_necessary_for_or (p q : Prop) : (p ∧ q → p ∨ q) ∧ ¬ (p ∨ q → p ∧ q) :=
by {
  exact ⟨and_implies_or p q, or_does_not_imply_and p q⟩,
}

end and_implies_or_or_does_not_imply_and_and_is_sufficient_but_not_necessary_for_or_l112_112290


namespace PQ_bisects_BD_l112_112942

variable (A B C D P Q : Type) [Add A] (M : A) [Div A Two]

theorem PQ_bisects_BD
  (h1 : convex_quadrilateral A B C D)
  (h2 : midpoint A B P)
  (h3 : midpoint C D Q)
  (h4 : bisects P Q A C) :
  bisects P Q B D :=
sorry

end PQ_bisects_BD_l112_112942


namespace number_of_ways_to_assign_roles_l112_112466

-- Definitions for conditions: number of members and distinct roles
def number_of_members : ℕ := 5
def number_of_roles : ℕ := 5

-- Theorem stating that the number of ways to assign distinct roles to members is 120
theorem number_of_ways_to_assign_roles : (finset.perm (finset.range number_of_members)).card = 120 :=
by
  sorry

end number_of_ways_to_assign_roles_l112_112466


namespace values_of_x_l112_112469

theorem values_of_x {x : ℤ} (p : |x - 1| ≥ 2) (hq : ¬¬(x ∈ ℤ)) (h : ¬(|x - 1| ≥ 2 ∧ x ∈ ℤ))
  : x = 0 ∨ x = 1 ∨ x = 2 :=
by
  sorry

end values_of_x_l112_112469


namespace bisects_diagonals_l112_112925

-- Define the data structure for a convex quadrilateral
structure ConvexQuadrilateral (α : Type*) :=
(A B C D : α)

-- Define midpoints of line segments
def midpoint {α : Type*} [Add α] [Div α] [Nonempty α] (A B : α) : α :=
(A + B) / 2

-- Main theorem stating the problem
theorem bisects_diagonals
  {α : Type*} [AddCommGroup α] [Module ℝ α] (quad : ConvexQuadrilateral α)
  (P Q : α)
  (hP : P = midpoint quad.A quad.B)
  (hQ : Q = midpoint quad.C quad.D)
  (hPQ : ∃ M, M = midpoint quad.A quad.C ∧ M ∈ line_through P Q) :
  ∃ N, N = midpoint quad.B quad.D ∧ N ∈ line_through P Q :=
sorry

lemma line_through (P Q : α) : Prop :=
∃ (λ1 λ2 : ℝ), P + λ1 • (Q - P) = Q + λ2 • (P - Q)

end bisects_diagonals_l112_112925


namespace value_of_m_l112_112077

theorem value_of_m (x m : ℝ) (h_positive_root : x > 0) (h_eq : x / (x - 1) - m / (1 - x) = 2) : m = -1 := by
  sorry

end value_of_m_l112_112077


namespace oldest_child_age_l112_112684

open Nat

def avg_age (a b c d : ℕ) := (a + b + c + d) / 4

theorem oldest_child_age 
  (h_avg : avg_age 5 8 11 x = 9) : x = 12 :=
by
  sorry

end oldest_child_age_l112_112684


namespace chebyshev_theorem_l112_112461

noncomputable def chebyshev_theorem_applies (α : ℝ) : Prop :=
  let X := λ k : ℕ, MeasureTheory.ProbabilityMeasure (λ x, 
    if x = -k * α then 1 / (2 * k^2)
    else if x = 0 then 1 - 1 / k^2
    else if x = k * α then 1 / (2 * k^2)
    else 0)
  in (∀ k : ℕ, MeasureTheory.Indep_fun (λ x : ℝ, X k x)
      ∧ (MeasureTheory.expectation (λ x, X k x) = 0)
      ∧ (MeasureTheory.expectation (λ x, (X k x)^2) = α^2))
  
theorem chebyshev_theorem (α : ℝ) : chebyshev_theorem_applies α := sorry

end chebyshev_theorem_l112_112461


namespace min_value_x_squared_plus_10x_l112_112740

theorem min_value_x_squared_plus_10x : ∃ x : ℝ, (x^2 + 10 * x) = -25 :=
by {
  sorry
}

end min_value_x_squared_plus_10x_l112_112740


namespace cos_sin_pow_l112_112670

open Complex

theorem cos_sin_pow (x : ℝ) (n : ℕ) (h : n > 0) : 
  (Complex.cos x + Complex.sin x * Complex.I)^n = Complex.cos (n * x) + Complex.sin (n * x) * Complex.I :=
by
  induction n with
  | zero => contradiction
  | succ =>
    intro IH
    sorry

end cos_sin_pow_l112_112670


namespace log4_21_correct_l112_112005

noncomputable def log4_21 (a b : ℝ) (h1 : Real.log 3 = a * Real.log 2)
                                     (h2 : Real.log 2 = b * Real.log 7) : ℝ :=
  (a * b + 1) / (2 * b)

theorem log4_21_correct (a b : ℝ) (h1 : Real.log 3 = a * Real.log 2) 
                        (h2 : Real.log 2 = b * Real.log 7) : 
  log4_21 a b h1 h2 = (a * b + 1) / (2 * b) := 
sorry

end log4_21_correct_l112_112005


namespace probability_ball_in_last_box_is_approx_0_l112_112104

/-- Given 100 boxes and 100 balls randomly distributed,
the probability that the last box will contain the only ball is approximately 0.370. -/
theorem probability_ball_in_last_box_is_approx_0.370 :
  let n := 100,
      p := 1 / n,
      probability : ℝ := (99 / 100)^99
  in abs (probability - 0.370) < 0.001 :=
by {
  let n := 100,
  let p := 1 / n,
  let probability := (99 / 100)^99,
  sorry
}

end probability_ball_in_last_box_is_approx_0_l112_112104


namespace problem_1_expression_problem_2_range_l112_112051

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

def condition1 (a b : ℝ) : Prop := a > 0

def condition2 (a b : ℝ) (x : ℝ) : Prop := f a b (-1) = 0

def condition3 (a b : ℝ) : Prop := ∀ x : ℝ, f a b x ≥ 0

def F (a b x : ℝ) : ℝ :=
  if x > 0 then f a b x else - (f a b x)

def g (a b k x : ℝ) : ℝ := f a b x - k * x

def is_monotonic (f : ℝ → ℝ) : Prop := ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≤ f x₂

theorem problem_1_expression (a : ℝ) (b : ℝ) (x : ℝ) (ha : condition1 a b)
  (hc2 : condition2 a b x) (hc3 : condition3 a b) :
  F a b x = if x > 0 then (x + 1)^2 else -(x + 1)^2 :=
by sorry

theorem problem_2_range (a : ℝ) (b : ℝ) (ha : condition1 a b)
  (hc2 : condition2 a b x) (hc3 : condition3 a b) (k : ℝ) :
  is_monotonic (g a b k) → (6 ≤ k ∨ k ≤ -2) :=
by sorry

end problem_1_expression_problem_2_range_l112_112051


namespace larger_angle_measure_l112_112257

-- Defining all conditions
def is_complementary (a b : ℝ) : Prop := a + b = 90

def angle_ratio (a b : ℝ) : Prop := a / b = 5 / 4

-- Main proof statement
theorem larger_angle_measure (a b : ℝ) (h1 : is_complementary a b) (h2 : angle_ratio a b) : a = 50 :=
by
  sorry

end larger_angle_measure_l112_112257


namespace integral_sqrt_minus_x_eq_pi_minus_two_over_four_l112_112222

theorem integral_sqrt_minus_x_eq_pi_minus_two_over_four :
  ∫ x in 0..1, (sqrt (1 - x^2) - x) = (π - 2) / 4 :=
by sorry

end integral_sqrt_minus_x_eq_pi_minus_two_over_four_l112_112222


namespace range_of_m_inequality_system_l112_112440

theorem range_of_m_inequality_system (m : ℝ) :
  (∀ x : ℤ, (-5 < x ∧ x ≤ m + 1) ↔ (x = -4 ∨ x = -3 ∨ x = -2)) →
  -3 ≤ m ∧ m < -2 :=
by
  sorry

end range_of_m_inequality_system_l112_112440


namespace find_f_neg2003_l112_112162

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_neg2003 (f_defined : ∀ x : ℝ, ∃ y : ℝ, f y = x → f y ≠ 0)
  (cond1 : ∀ ⦃x y w : ℝ⦄, x > y → (f x + x ≥ w → w ≥ f y + y → ∃ z, y ≤ z ∧ z ≤ x ∧ f z = w - z))
  (cond2 : ∃ u : ℝ, f u = 0 ∧ ∀ v : ℝ, f v = 0 → u ≤ v)
  (cond3 : f 0 = 1)
  (cond4 : f (-2003) ≤ 2004)
  (cond5 : ∀ x y : ℝ, f x * f y = f (x * f y + y * f x + x * y)) :
  f (-2003) = 2004 :=
sorry

end find_f_neg2003_l112_112162


namespace correct_options_l112_112919

noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 3)
noncomputable def b (n : ℝ) : ℝ × ℝ := (n, Real.sqrt 3)
def k (n : ℝ) : ℝ := Real.sqrt 3 / n

theorem correct_options (n : ℝ) : 
  ((∃ k : ℝ, k > 0 ∧ a = k • b n) ↔ n = 1) ∧
  (∃ u : ℝ × ℝ, u = (-Real.sqrt 3 / 2, 1 / 2) ∨ u = (Real.sqrt 3 / 2, -1 / 2)) ∧
  (vector.proj a (b n) = 3 * unit_vector a → n = 3) ∧
  ((inner a (b n) > 0) ↔ n > -3) :=
sorry

end correct_options_l112_112919


namespace positive_root_exists_iff_m_eq_neg_one_l112_112075

theorem positive_root_exists_iff_m_eq_neg_one :
  (∃ x : ℝ, x > 0 ∧ (x / (x - 1) - m / (1 - x) = 2)) ↔ m = -1 :=
by
  sorry

end positive_root_exists_iff_m_eq_neg_one_l112_112075


namespace part_I_part_II_l112_112009

def sequence_a (a : ℕ → ℝ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 1) = (2 * (n + 1) * a n / n) + (n + 1)) :=
  (∀ n : ℕ, a n ≠ 0)

theorem part_I (a : ℕ → ℝ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 1) = (2 * (n + 1) * a n / n) + (n + 1)) :
  ∃ c : ℝ, ∀ n : ℕ, (a n / n) + 1 = c * (2 ^ n) :=
sorry

theorem part_II (a : ℕ → ℝ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 1) = (2 * (n + 1) * a n / n ) + (n + 1)) :
  ∀ n : ℕ, 
    (∑ i in range n, a (i+1)) = ((n - 1) * (2^(n + 1)) + 2) - (n * (n + 1) / 2) :=
sorry

end part_I_part_II_l112_112009


namespace units_digit_of_k_squared_plus_2_to_the_k_l112_112652

def k : ℕ := 2021^2 + 2^2021 + 3

theorem units_digit_of_k_squared_plus_2_to_the_k :
    (k^2 + 2^k) % 10 = 0 :=
by
    sorry

end units_digit_of_k_squared_plus_2_to_the_k_l112_112652


namespace probability_last_box_single_ball_l112_112092

noncomputable def probability_last_box_only_ball (n : ℕ) : ℝ :=
  let p := 1 / n in
  (n * p * (1 - p)^(n - 1))

theorem probability_last_box_single_ball :
  probability_last_box_only_ball 100 ≈ 0.370 :=
by
  sorry

end probability_last_box_single_ball_l112_112092


namespace angle_AMH_l112_112180

-- Given conditions and definitions
variables {α : Type} [euclidean_geometry α]
open euclidean_geometry

-- Define points M, A, B, C, L, H as described in the problem 
variables {A B C M L H : α}

-- Given definitions based on the problem's conditions
-- 1. M is the midpoint of AB
def M_midpoint : midpoint A B M := sorry

-- 2. Triangle ABC is isosceles right with right angle at B
def triangle_ABC : triangle A B C ∧ ∠ B = 90 := sorry

-- 3. The angle bisector of ∠ A intersects the circumcircle at L
def L_on_circumcircle : (∠ A bisector intersects circumcircle of triangle A B C at L) := sorry

-- 4. Point H is the foot of the perpendicular from L to AC
def H_foot : foot L (line_through A C) H := sorry

-- Required to prove that ∠ AMH = 112.5

theorem angle_AMH : ∠ A M H = 112.5 := 
by 
  apply_and_elim 
    from 
      [M_midpoint, triangle_ABC, L_on_circumcircle, H_foot]
  sorry -- Proof goes here

end angle_AMH_l112_112180


namespace smallest_positive_real_x_l112_112396

theorem smallest_positive_real_x :
  ∃ (x : ℝ), x > 0 ∧ ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧ x = 89 / 9 :=
by
  sorry

end smallest_positive_real_x_l112_112396


namespace arithmetic_expression_evaluation_l112_112811

theorem arithmetic_expression_evaluation :
  (-12 * 6) - (-4 * -8) + (-15 * -3) - (36 / (-2)) = -77 :=
by
  sorry

end arithmetic_expression_evaluation_l112_112811


namespace triangle_theorem_l112_112609

theorem triangle_theorem 
  (A B C : ℝ) 
  (h1 : ∀ A B C : ℝ, ∃ (a b c : ℝ), 
      a^2 - b^2 - c^2 = b * c ∧ sin^2 A - sin^2 B - sin^2 C = sin B * sin C) 
  (h2 : 0 < A ∧ A < π) : 
  (A = 2 * π / 3) ∧ 
  (∀ (BC : ℝ), BC = 3 → ∃ (a b c : ℝ), 
      a + b + c ≤ 3 + 2 * sqrt 3) :=
by
  sorry

end triangle_theorem_l112_112609


namespace part_1_part_2_l112_112590

-- Define the triangle and the given condition
variables {α β γ : ℝ}
axiom triangle_ABC : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π
axiom sin_identity : sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)
axiom BC_length : (sin(γ) / sin(α)) * BC = 3

-- State the main theorem parts separately
theorem part_1 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_sin_identity: sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)) :
  α = 2 * π / 3 :=
sorry

theorem part_2 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_α: α = 2 * π / 3) (h_BC_length: BC = 3) :
  let b := (2 * sqrt(3) * sin(π/6 - β)),
      c := (2 * sqrt(3) * sin(π/6 + β)) in
  (3 + 2 * sqrt(3)) :=
sorry

end part_1_part_2_l112_112590


namespace dot_product_comm_dot_product_distrib_l112_112274

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

-- Dot product is commutative:
theorem dot_product_comm : ∀ (a b : V), ⟪a, b⟫ = ⟪b, a⟫ :=
begin
  sorry,
end

-- Dot product is distributive over addition:
theorem dot_product_distrib : ∀ (a b c : V), ⟪a + b, c⟫ = ⟪a, c⟫ + ⟪b, c⟫ :=
begin
  sorry,
end

end dot_product_comm_dot_product_distrib_l112_112274


namespace range_of_a_l112_112078

variable (f : ℝ → ℝ) (a : ℝ)

-- Given conditions
def isValidFunction (a : ℝ) :=
  (a > 0) ∧ (a ≠ 1) ∧
  (∀ (x₁ x₂ : ℝ), x₁ < x₂ ∧ x₂ ≤ a / 2 → 
    f x₁ - f x₂ > 0)

-- To Prove
theorem range_of_a : 
  isValidFunction (λ x => Real.log (x^2 - a * x + 3) (a)) a → 1 < a ∧ a < 2 * Real.sqrt 3 :=
by
  sorry

end range_of_a_l112_112078


namespace section_formula_l112_112641

theorem section_formula (C D Q : Type) [AddCommGroup C] [Module ℝ C]
  (r : ℝ) (h : Q = (5 / (5 + 3)) • C + (3 / (5 + 3)) • D) : 
  ∃ (s v : ℝ), Q = s • C + v • D ∧ s = 5 / 8 ∧ v = 3 / 8 :=
by
  use (5 / 8), (3 / 8)
  split
  · exact h
  split
  · rfl
  exact rfl

end section_formula_l112_112641


namespace range_of_a_l112_112496

theorem range_of_a (a : ℝ) (h_pos : a > 0) : 
  (∀ x1 : ℝ, ∃ x2 ∈ Ici (-2 : ℝ), (x1^2 - 2 * x1) > (a * x2 + 2)) ↔ (a > 3/2) := 
by
  sorry

end range_of_a_l112_112496


namespace value_of_a_minus_b_l112_112993

theorem value_of_a_minus_b (a b : ℝ) 
  (h₁ : (a-4)*(a+4) = 28*a - 112) 
  (h₂ : (b-4)*(b+4) = 28*b - 112) 
  (h₃ : a ≠ b)
  (h₄ : a > b) :
  a - b = 20 :=
sorry

end value_of_a_minus_b_l112_112993


namespace complex_mag_ratio_result_l112_112465

noncomputable def complex_mag_ratio
  (z₁ z₂ : ℂ)
  (h₁ : complex.abs z₁ = 2)
  (h₂ : complex.abs z₂ = 3)
  (h₃ : real.angle.real_angle (z₁.angle_in_complex_plane (units.real_axis z₁)) 
         (z₂.angle_in_complex_plane (units.real_axis z₂)) = real.angle.of_deg 60) 
  : real :=
  (complex.abs (z₁ + z₂) / complex.abs (z₁ - z₂))

theorem complex_mag_ratio_result
  (z₁ z₂ : ℂ)
  (h₁ : complex.abs z₁ = 2)
  (h₂ : complex.abs z₂ = 3)
  (h₃ : real.angle.real_angle (z₁.angle_in_complex_plane (units.real_axis z₁)) 
         (z₂.angle_in_complex_plane (units.real_axis z₂)) = real.angle.of_deg 60) 
  : complex_mag_ratio z₁ z₂ h₁ h₂ h₃ = real.sqrt (133) / 7 := 
  sorry

end complex_mag_ratio_result_l112_112465


namespace math_problem_l112_112455

variables {a b : ℕ → ℕ}

-- Conditions for the geometric sequence {a_n}
def q : ℕ := 2

def sum_first_3_terms : Prop :=
  ∃ a : ℕ → ℕ, a 1 + a 2 + a 3 = 7 ∧
               (∀ n, a n = 2 ^ (n - 1))

-- Conditions for the arithmetic sequence {b_n}
def b1 : ℕ := 3

def arithmetic_condition : Prop :=
  ∃ a b : ℕ → ℕ, 2 * b 2 = a 2 + a 4 ∧
               (∀ n, b n = 2 * n + 1)

-- Sum of the first n terms of sequence {2 / ((2n-1)b_n)}
def sum_first_n_terms (n : ℕ) : ℕ := ∑ i in (range n), (2 / ((2 * (i+1) - 1) * b (i+1)))

-- Sum of first n terms
def Sn (n : ℕ) : ℕ := (2 * n) / (2 * n + 1)

-- The theorem to be proven
theorem math_problem (a : ℕ → ℕ) (b : ℕ → ℕ) :
  sum_first_3_terms a → arithmetic_condition a b → sum_first_n_terms a b (n : ℕ) = Sn (n : ℕ) := by
  sorry

end math_problem_l112_112455


namespace sum_f_eq_2019_div_4_l112_112887

noncomputable def f (x : ℝ) : ℝ := x^3 - (3 / 2) * x^2 + (3 / 4) * x + (1 / 8)

theorem sum_f_eq_2019_div_4 : 
  (∑ k in finset.range 2019, f (k / 2018)) = 2019 / 4 :=
by
  sorry

end sum_f_eq_2019_div_4_l112_112887


namespace last_box_probability_l112_112084

noncomputable def probability_last_box_only_ball : ℝ :=
  let n : ℕ := 100 in
  let p : ℝ := 1 / n in
  (n * p * (1 - p)^(n - 1))

theorem last_box_probability : abs (probability_last_box_only_ball - 0.3697) < 0.0005 := 
  sorry

end last_box_probability_l112_112084


namespace PQ_bisects_BD_l112_112946

variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]
variables {A B C D P Q M N : Point}

def convex_quadrilateral (A B C D : Point) : Prop := sorry
def midpoint (P A B : Point) : Prop := 2 • P = A + B
def bisects (line P Q : Point) (diagonal A C : Point) : Prop := 
  ∃ M, midpoint M A C ∧ (line.contains M)
def line_contains_midpoint (P Q : Point) (mid : Point) : Prop := sorry

-- The theorem we want to prove:
theorem PQ_bisects_BD 
  (h1 : convex_quadrilateral A B C D)
  (h2 : midpoint P A B)
  (h3 : midpoint Q C D)
  (h4 : bisects (P, Q) (A, C))
  : bisects (P, Q) (B, D) := 
begin
  sorry
end

end PQ_bisects_BD_l112_112946


namespace find_f_six_l112_112881

noncomputable def f : ℝ → ℝ
| x := if x < 0 then x ^ 3 - 1 else if -1 ≤ x ∧ x ≤ 1 then choose (λ y, y = f x ∧ f (-x) = -f (x)) else choose (λ z, z = f (x - 1))

theorem find_f_six : f 6 = 2 :=
by
  -- Use the given conditions and the question
  have h₁ : ∀ x, x < 0 → f x = x ^ 3 - 1, sorry,
  have h₂ : ∀ x, -1 ≤ x ∧ x ≤ 1 → f (-x) = -f(x), sorry,
  have h₃ : ∀ x, x > 0.5 → f (x + 0.5) = f (x - 0.5), sorry,
  -- Prove f(6) = 2
  sorry

end find_f_six_l112_112881


namespace part1_part2_l112_112036

-- Definitions for the conditions
def slope_l1 : ℝ := 2

def A (m : ℝ) : ℝ × ℝ := (3 * m, 2 * m - 1)
def B (m : ℝ) : ℝ × ℝ := (2, m - 3)

-- Slope of a line through points A and B
def slope_l2 (m : ℝ) : ℝ := (B(m).2 - A(m).2) / (B(m).1 - A(m).1)

-- Problem (1): If the inclination angle of l2 is 45 degrees, m = 2
theorem part1 (m : ℝ) (h1 : slope_l2(m) = 1) : m = 2 := by
  sorry

-- Problem (2): If l1 is perpendicular to l2, m = -2/5
theorem part2 (m : ℝ) (h2 : slope_l1 * slope_l2(m) = -1) : m = -2 / 5 := by
  sorry

end part1_part2_l112_112036


namespace correct_diagram_is_illustration_l112_112749

def is_inflation_illustration (diagram : ℕ → ℝ) : Prop :=
  ∀ t1 t2 : ℕ, t1 < t2 → diagram t1 < diagram t2

variable diagrams : ℕ → ℕ → ℝ
variable correct_diagram : ℕ
hypothesis (inflation_illustration : is_inflation_illustration (diagrams correct_diagram))
hypothesis (inflation_conditions : ∃ i, is_inflation_illustration (diagrams i))

theorem correct_diagram_is_illustration : correct_diagram = 3 :=
by
  apply sorry

end correct_diagram_is_illustration_l112_112749


namespace trees_planted_l112_112247

theorem trees_planted (current_short_trees planted_short_trees total_short_trees : ℕ)
  (h1 : current_short_trees = 112)
  (h2 : total_short_trees = 217) :
  planted_short_trees = 105 :=
by
  sorry

end trees_planted_l112_112247


namespace james_max_average_speed_l112_112976

/-- Given James’ initial odometer reading of 12321 (a palindrome), 
    an ending reading of a palindrome, the constraints of a speed limit 
    of 60 miles per hour, and a driving duration of 5 hours, 
    prove the greatest possible average speed is 60 miles per hour.
-/
theorem james_max_average_speed :
  ∀ (initial final : ℕ),
    (final - initial ≤ 300) →
    nat.isPalindromic initial →
    nat.isPalindromic final →
    12321 = initial →
    ∀ t : ℕ,
      (t = 5) →
      (∀ v : ℕ, v ≤ 60) →
      (final - initial = 300) →
      (final = 12621) →
      (final - initial) / t = 60 :=
by 
  sorry

end james_max_average_speed_l112_112976


namespace petya_coloring_l112_112285

theorem petya_coloring (n : ℕ) (h₁ : n = 100) (h₂ : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → 1 ≤ (number i j) ∧ (number i j) ≤ n * n) :
  ∃ k, k = 1 ∧ ∀ (initial_coloring : fin (n * n) → bool) (next_colorable : (fin (n * n) → bool) → (fin (n * n) → bool)),
    (∀ (table : fin n × fin n → fin (n * n)), next_colorable (λ a, initial_coloring a)
    (λ a, initial_coloring a) a) :=
begin
  sorry,
end

end petya_coloring_l112_112285


namespace number_of_elements_union_l112_112983

noncomputable def A : Finset ℕ := sorry
noncomputable def B : Finset ℕ := sorry

theorem number_of_elements_union (h : 2 ^ A.card + 2 ^ B.card - 2 ^ (A ∩ B).card = 144) : (A ∪ B).card = 8 :=
sorry

end number_of_elements_union_l112_112983


namespace range_of_a_l112_112041

theorem range_of_a (a : ℝ) : 
  let P := (1, 2)
  let circle_equation := λ x y : ℝ, x^2 + y^2 + a * x + 2 * y + a^2
  in - (2 * Real.sqrt 3 / 3) < a ∧ a < 2 * Real.sqrt 3 / 3 ↔ 
      (¬ (∃ x y : ℝ, circle_equation x y = 0 ∧ ((x - 1)^2 + (y - 2)^2 = 0)) ∧ 
       ∃ x y : ℝ, circle_equation x y = 0 ∧ ((x - 1)^2 + (y - 2)^2 > 0)) :=
by sorry

end range_of_a_l112_112041


namespace lg_equation_unique_root_l112_112524

theorem lg_equation_unique_root (k : ℝ) :
  (∃ x : ℝ, lg (k * x) = 2 * lg (x + 1)) → (k = 4 ∨ k < 0) :=
by sorry

end lg_equation_unique_root_l112_112524


namespace complement_U_B_l112_112892

open Set

theorem complement_U_B
  (A B : Set ℝ)
  (x : ℝ) 
  (hA : A = {1, 3, x})
  (hB : B = {1, x^2})
  (U : Set ℝ := A ∪ B)
  (hUniversal : (B ∪ (U \ B)) = A) :
  U \ B = {3} ∨ U \ B = {Real.sqrt 3} ∨ U \ B = {-Real.sqrt 3} :=
sorry

end complement_U_B_l112_112892


namespace find_f_of_4_l112_112484

def monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f(x) ≤ f(y)) ∨ (∀ x y, x ≤ y → f(y) ≤ f(x))

variable (f : ℝ → ℝ)
variable (cond1 : monotonic f)
variable (cond2 : ∀ x : ℝ, f (f x - 3 ^ x) = 4)

theorem find_f_of_4 : f 4 = 82 :=
by {
  sorry
}

end find_f_of_4_l112_112484


namespace smallest_multiple_division_l112_112153

noncomputable def smallest_multiple_with_divisors : ℕ :=
  let m := min { n ∈ ℕ | 125 ∣ n ∧ (nat.factors n).length = 125 }.min in
  m

theorem smallest_multiple_division (m : ℕ) (h1 : 125 ∣ m) (h2 : (nat.factors m).length = 125) : 
  m = 125 * 2^124 :=
by
  sorry

end smallest_multiple_division_l112_112153


namespace coefficient_degree_monomial_l112_112212

noncomputable def monomial := - (Real.pi * (7:ℝ)⁻¹) * (x^2) * y

theorem coefficient_degree_monomial (x y : ℝ) : 
  coefficient monomial = - (Real.pi * (7:ℝ)⁻¹) ∧ degree monomial = 3 := 
sorry

end coefficient_degree_monomial_l112_112212


namespace part_1_part_2_l112_112589

-- Define the triangle and the given condition
variables {α β γ : ℝ}
axiom triangle_ABC : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π
axiom sin_identity : sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)
axiom BC_length : (sin(γ) / sin(α)) * BC = 3

-- State the main theorem parts separately
theorem part_1 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_sin_identity: sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)) :
  α = 2 * π / 3 :=
sorry

theorem part_2 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_α: α = 2 * π / 3) (h_BC_length: BC = 3) :
  let b := (2 * sqrt(3) * sin(π/6 - β)),
      c := (2 * sqrt(3) * sin(π/6 + β)) in
  (3 + 2 * sqrt(3)) :=
sorry

end part_1_part_2_l112_112589


namespace pencils_in_drawer_l112_112248

-- Definitions for the initial conditions and actions
def initial_pencils : ℕ := 34
def dan_took_pencils : ℕ := 22
def dan_returned_pencils : ℕ := 5
def lucy_returned_pencils : ℕ := 1

-- Define the net pencils taken by Dan
def net_pencils_taken : ℕ := dan_took_pencils - dan_returned_pencils

-- Define the pencils remaining after Dan's actions
def pencils_after_dan : ℕ := initial_pencils - net_pencils_taken

-- Define the final number of pencils after Lucy's return
def final_pencils : ℕ := pencils_after_dan + lucy_returned_pencils

-- Statement to be proven:
theorem pencils_in_drawer : final_pencils = 18 :=
by
  simp [initial_pencils, dan_took_pencils, dan_returned_pencils, lucy_returned_pencils, net_pencils_taken, pencils_after_dan, final_pencils]
  sorry

end pencils_in_drawer_l112_112248


namespace g_at_4_l112_112645

def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) : ℝ := 3 - 4 / x

def g (x : ℝ) : ℝ := 1 / (f_inv x) + 10

theorem g_at_4 : g 4 = 10.5 := by
  sorry

end g_at_4_l112_112645


namespace part_1_part_2_l112_112586

-- Define the triangle and the given condition
variables {α β γ : ℝ}
axiom triangle_ABC : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π
axiom sin_identity : sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)
axiom BC_length : (sin(γ) / sin(α)) * BC = 3

-- State the main theorem parts separately
theorem part_1 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_sin_identity: sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)) :
  α = 2 * π / 3 :=
sorry

theorem part_2 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_α: α = 2 * π / 3) (h_BC_length: BC = 3) :
  let b := (2 * sqrt(3) * sin(π/6 - β)),
      c := (2 * sqrt(3) * sin(π/6 + β)) in
  (3 + 2 * sqrt(3)) :=
sorry

end part_1_part_2_l112_112586


namespace fraction_subtraction_simplify_l112_112349

noncomputable def fraction_subtraction : ℚ :=
  (12 / 25) - (3 / 75)

theorem fraction_subtraction_simplify : fraction_subtraction = (11 / 25) :=
  by
    -- Proof goes here
    sorry

end fraction_subtraction_simplify_l112_112349


namespace initial_number_of_men_l112_112251

theorem initial_number_of_men (M : ℕ) (h1 : ∃ food : ℕ, food = M * 22) (h2 : ∀ food, food = (M * 20)) (h3 : ∃ food : ℕ, food = ((M + 40) * 19)) : M = 760 := by
  sorry

end initial_number_of_men_l112_112251


namespace bisects_AC_implies_bisects_BD_l112_112936

/-- Given a convex quadrilateral ABCD with points P and Q being the midpoints of sides AB and CD respectively,
    and given that the line segment PQ bisects the diagonal AC, prove that PQ also bisects the diagonal BD. -/
theorem bisects_AC_implies_bisects_BD
    (A B C D P Q M N : Point)
    (hP : midpoint A B P)
    (hQ : midpoint C D Q)
    (hM : midpoint A C M)
    (hN : midpoint B D N)
    (hPQ_bisects_AC : lies_on_line M (line_through P Q))
    : lies_on_line N (line_through P Q) :=
sorry

end bisects_AC_implies_bisects_BD_l112_112936


namespace distinct_real_root_of_P_n_eq_x_l112_112154

def P1 (x : ℝ) : ℝ := x^2 - 2

def P (n : ℕ) : (ℝ → ℝ) → (ℝ → ℝ)
| 0     := λ x, x
| (n+1) := P1 ∘ (P n)

theorem distinct_real_root_of_P_n_eq_x (n : ℕ) :
  n > 0 → ∀ x₁ x₂ : ℝ, P n (λ x, P1 x) x₁ = x₁ ∧ P n (λ x, P1 x) x₂ = x₂ → x₁ = x₂ := 
by
  sorry

end distinct_real_root_of_P_n_eq_x_l112_112154


namespace total_weight_cabbage_radish_l112_112244

def weight_cabbage_kg : ℝ := 4.436
def weight_radish_kg : ℝ := 1.999
def trips : ℕ := 2

theorem total_weight_cabbage_radish :
  trips * (weight_cabbage_kg + weight_radish_kg) = 12.87 :=
by
  -- Weight of cabbage and radish per trip
  have h_cabbage := weight_cabbage_kg
  have h_radish := weight_radish_kg

  -- Total weight per trip
  let total_per_trip := h_cabbage + h_radish

  -- Total weight for both trips
  let total_weight := trips * total_per_trip

  -- Assert the total weight
  show total_weight = 12.87, from sorry

end total_weight_cabbage_radish_l112_112244


namespace find_ordered_pair_l112_112236

theorem find_ordered_pair (a c : ℝ) (h1 : a + c = 41) (h2 : a < c) 
(h3 : ∃ x : ℝ, ∀ x1 x2 : ℝ, (ax^2 - 30x + c = 0) → x1 = x2) :
  (a, c) = ( (41 + Real.sqrt 781) / 2, (41 - Real.sqrt 781) / 2) :=
sorry

end find_ordered_pair_l112_112236


namespace am_gm_inequality_proof_l112_112852

variable {n : ℕ}
variable {a : Fin n → ℝ} 

noncomputable def am_gm_inequality (h_pos : ∀ i, 0 < a i) : Prop :=
  let a₁ := a 0
  let aₙ₁ := a 0
  ∑ k in Finset.range n, (a k / a (k + 1) % n) ^ (n - 1) ≥ 
  -n + 2 * (∑ k in Finset.range n, a k) * (∏ k in Finset.range n, a k ^ (- 1 / n))

-- Define a theorem for the AM-GM inequality statement
theorem am_gm_inequality_proof (h_pos : ∀ i, 0 < a i) : 
  am_gm_inequality h_pos := 
sorry

end am_gm_inequality_proof_l112_112852


namespace positive_negative_difference_divisible_by_4_l112_112240

-- Definitions for conditions
structure Street :=
  (color : String) -- "white", "red", "blue"

structure Intersection :=
  (streets : List Street)
  (positive : Bool)

-- Function to determine if an intersection is positive
def is_positive (i : Intersection) : Bool :=
  match i.streets with
  | [w, b, r] => w.color = "white" ∧ b.color = "blue" ∧ r.color = "red"
  | _ => false

-- Function to count positive and negative intersections
def count_positive_negative (intersections : List Intersection)
  : Nat × Nat :=
  intersections.foldl
    (fun (acc : Nat × Nat) (i : Intersection) =>
     if is_positive(i) then (acc.1 + 1, acc.2) else (acc.1, acc.2 + 1))
    (0, 0)

-- Theorem to prove
theorem positive_negative_difference_divisible_by_4 
  (intersections : List Intersection)
  (h_intersections : ∀ i ∈ intersections, i.streets.length = 3 ∧ 
                                          i.streets.map Street.color ∈ [["white","blue","red"], ["white","red","blue"],
                                                                         ["red","blue","white"], ["red","white","blue"],
                                                                         ["blue","white","red"], ["blue","red","white"]]):
  (let (pos, neg) := count_positive_negative(intersections) in
  (pos - neg) % 4 = 0) :=
by
  sorry

end positive_negative_difference_divisible_by_4_l112_112240


namespace sequence_properties_l112_112458

-- Let a sequence {a_n} be defined such that all terms are positive, a_1 = 1,
-- and for all n, a_n^2 - (2a_{n+1} - 1)a_n - 2a_{n+1} = 0

def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ 
  (∀ n : ℕ, 0 < a n) ∧ 
  (∀ n : ℕ, a n ^ 2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0)

theorem sequence_properties (a : ℕ → ℝ) (h : sequence a) :
  a 2 = 1 / 2 ∧
  a 3 = 1 / 4 ∧ 
  ∀ n : ℕ, a (n + 1) = 1 / (2 ^ n) := 
by
  sorry

end sequence_properties_l112_112458


namespace chessboard_recolor_impossible_l112_112303

theorem chessboard_recolor_impossible : ¬(∃ k, k = 1 ∧ ∀ n, even n → even (n + k)) :=
by
  -- Given:
  -- A standard chessboard has 8 rows and 8 columns, hence it contains a total of 64 cells.
  -- The chessboard is typically arranged with alternating black and white cells, resulting in exactly half the board being black, so initially 32 black cells.
  strating_black_cells = 32
  -- It is allowed to repaint in another color all cells located inside a square of size \(2 \times 2\).
  -- Prove: There cannot be exactly one black cell left after any number of recoloring operations.
  sorry

end chessboard_recolor_impossible_l112_112303


namespace value_of_n_max_value_on_interval_sum_of_zeros_gt_two_l112_112880

-- Definition of the function f(x)
def f (x : ℝ) (m n : ℝ) : ℝ := (m * x - n) / x - Real.log x

-- Condition 1: Value of n based on the tangent line condition
theorem value_of_n (m : ℝ) : 
  (∀ x : ℝ, f x m n = (m * x - n) / x - Real.log x) → 
  ((∃ m, (f' 2 = 1) ∧ ((x,y) = (2, f 2))) → ∃ m, (f' 2 = 1) ∧ ((x,y) = (2, f 2)) →n = 6 :=
by
  sorry

-- Condition 2: Maximum value of f(x) on [1, +∞)
theorem max_value_on_interval (m : ℝ) : 
  (∀ x : ℝ, 1 ≤ x → ∃ y : ℝ, y = f x m 6) → 
  (∃ x_max : ℝ, 1 ≤ x_max ∧ ∀ x : ℝ, 1 ≤ x → f x m 6 ≤ x_max) :=
by
  sorry

-- Condition 3: Prove x1 + x2 > 2 given n = 1 and two distinct positive zeros
theorem sum_of_zeros_gt_two (m : ℝ) (x1 x2 : ℝ) :
  (∀ x : ℝ, f x m 1 = (m * x - 1) / x - Real.log x) → 
  (0 < x1 ∧ x1 < x2 ∧ f x1 m 1 = 0 ∧ f x2 m 1 = 0) →
  x1 + x2 > 2 :=
by
  sorry

end value_of_n_max_value_on_interval_sum_of_zeros_gt_two_l112_112880


namespace problem_equivalence_l112_112463

def ellipse_M (x y : ℝ) (b : ℝ) : Prop :=
  x^2 / 9 + y^2 / b^2 = 1 ∧ b > 0 ∧ (2, 0) is_focus_of_ellipse x y b

def ellipse_N (x y : ℝ) (m n : ℝ) : Prop :=
  n^2 - m^2 = 5 ∧
  (x^2 + y^2 / 6 = 1) ∧ (x / m)^2 + (y / n)^2 = 1 ∧
  m > 0 ∧ n > m ∧ (n^2 - m^2 = 5)

def line_y_eq_x_minus_2 (x y : ℝ) : Prop :=
  y = x - 2

def intersection_points (x1 y1 x2 y2 : ℝ) :=
  ellipse_N x1 y1 ∧ line_y_eq_x_minus_2 x1 y1 ∧
  ellipse_N x2 y2 ∧ line_y_eq_x_minus_2 x2 y2

def length_AB (A B : ℝ × ℝ) : ℝ :=
  let ⟨x1, y1⟩ := A in
  let ⟨x2, y2⟩ := B in
  real.sqrt (1 + 1^2) * real.sqrt ((x1 + x2)^2 - 4 * (x1 * x2))

def area_triangle_AOB (A B O : ℝ × ℝ) : ℝ :=
  let ⟨x1, y1⟩ := A in
  let ⟨x2, y2⟩ := B in
  let ⟨x0, y0⟩ := O in
  (1/2) * real.sqrt (2) * length_AB (x1,y1) (x2,y2)

theorem problem_equivalence (b m n : ℝ) (x1 y1 x2 y2 : ℝ) :
  ellipse_M x1 y1 b → ellipse_N x2 y2 m n → 
  line_y_eq_x_minus_2 x1 y1 → line_y_eq_x_minus_2 x2 y2 →
  (x1 + x2 = 4/7) → (x1 * x2 = -2/7) →
  length_AB (x1, y1) (x2, y2) = 12 / 7 ∧ 
  area_triangle_AOB (x1, y1) (x2, y2) (0, 0) = 6 * real.sqrt(2) / 7 :=
  by sorry

end problem_equivalence_l112_112463


namespace average_visitors_on_sundays_l112_112780

theorem average_visitors_on_sundays 
  (avg_other_days : ℕ) (avg_per_day : ℕ) (days_in_month : ℕ) (sundays : ℕ) (S : ℕ)
  (h_avg_other_days : avg_other_days = 240)
  (h_avg_per_day : avg_per_day = 310)
  (h_days_in_month : days_in_month = 30)
  (h_sundays : sundays = 5) :
  (sundays * S + (days_in_month - sundays) * avg_other_days = avg_per_day * days_in_month) → 
  S = 660 :=
by
  intros h
  rw [h_avg_other_days, h_avg_per_day, h_days_in_month, h_sundays] at h
  sorry

end average_visitors_on_sundays_l112_112780


namespace intersection_with_y_axis_l112_112216

theorem intersection_with_y_axis (x y : ℝ) : (x + y - 3 = 0 ∧ x = 0) → (x = 0 ∧ y = 3) :=
by {
  sorry
}

end intersection_with_y_axis_l112_112216


namespace min_positive_period_of_function_l112_112225

theorem min_positive_period_of_function :
  ( ∃ T > 0, ∀ x, 3 * sin (2 * (x + T) + π / 4) = 3 * sin (2 * x + π / 4)) ↔ T = π :=
begin
  sorry
end

end min_positive_period_of_function_l112_112225


namespace area_of_ABC_l112_112813

def point : Type := ℝ × ℝ

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_ABC : area_of_triangle (0, 0) (1, 0) (0, 1) = 0.5 :=
by
  sorry

end area_of_ABC_l112_112813


namespace standard_deviation_shift_invariant_l112_112010

noncomputable def standard_deviation (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  let variance := (data.map (λ x => (x - mean) ^ 2)).sum / data.length
  variance.sqrt

theorem standard_deviation_shift_invariant (x : List ℝ) (c : ℝ) (h : c ≠ 0) :
  standard_deviation (x.map (λ xi => xi + c)) = standard_deviation x :=
by
  sorry

end standard_deviation_shift_invariant_l112_112010


namespace ratio_is_sqrt3_div_3_l112_112534

-- Define the conditions of the problem
variables {A B C D E F : ℂ}
variables {M N : ℂ}
variable (r : ℝ)
variable (ζ : ℂ) -- ζ represents the primitive 6th root of unity

-- Conditions capturing the geometric properties of the hexagon and ratio division
axiom regular_hexagon (h₀ : A = 1) (h₁ : B = ζ) (h₂ : C = ζ^2) (h₃ : D = ζ^3)
  (h₄ : E = ζ^4) (h₅ : F = ζ^5) : ζ = complex.exp (real.pi / 3 * complex.I)

axiom ratio_condition 
  (h₆ : M = (1 - r) * A + r * C)
  (h₇ : N = (1 - r) * C + r * E) : true

axiom collinear_BMN 
  (h₈ : (M - B) = k * (N - M) for some complex number k) : true

-- The statement to be proved
theorem ratio_is_sqrt3_div_3 (h₀ : A = 1) (h₁ : B = ζ) (h₂ : C = ζ^2) (h₃ : D = ζ^3)
  (h₄ : E = ζ^4) (h₅ : F = ζ^5) 
  (h₆ : M = 1 - r + r * ζ^2) 
  (h₇ : N = ζ^2 * (1 - r + r * ζ)) 
  (h₈ : ∃ k : ℂ, (M - B) = k * (N - M)) : 
  r = real.sqrt 3 / 3 := 
sorry

end ratio_is_sqrt3_div_3_l112_112534


namespace find_temperature_on_December_25_l112_112176

theorem find_temperature_on_December_25 {f : ℕ → ℤ}
  (h_recurrence : ∀ n, f (n - 1) + f (n + 1) = f n)
  (h_initial1 : f 3 = 5)
  (h_initial2 : f 31 = 2) :
  f 25 = -3 :=
  sorry

end find_temperature_on_December_25_l112_112176


namespace lattice_points_on_hyperbola_l112_112361

theorem lattice_points_on_hyperbola :
  {p : ℤ × ℤ | p.1 ^ 2 - p.2 ^ 2 = 61}.finite ∧ 
  {p : ℤ × ℤ | p.1 ^ 2 - p.2 ^ 2 = 61}.to_finset.card = 4 := 
sorry

end lattice_points_on_hyperbola_l112_112361


namespace find_common_difference_l112_112874

-- Definitions based on conditions in a)
def common_difference_4_10 (a₁ d : ℝ) : Prop :=
  (a₁ + 3 * d) + (a₁ + 9 * d) = 0

def sum_relation (a₁ d : ℝ) : Prop :=
  2 * (12 * a₁ + 66 * d) = (2 * a₁ + d + 10)

-- Math proof problem statement
theorem find_common_difference (a₁ d : ℝ) 
  (h₁ : common_difference_4_10 a₁ d) 
  (h₂ : sum_relation a₁ d) : 
  d = -10 :=
sorry

end find_common_difference_l112_112874


namespace number_of_grade12_students_selected_l112_112777

def total_students : ℕ := 1500
def grade10_students : ℕ := 550
def grade11_students : ℕ := 450
def total_sample_size : ℕ := 300
def grade12_students : ℕ := total_students - grade10_students - grade11_students

theorem number_of_grade12_students_selected :
    (total_sample_size * grade12_students / total_students) = 100 := by
  sorry

end number_of_grade12_students_selected_l112_112777


namespace root_product_sum_eq_zero_l112_112457

noncomputable def quadratic_eq := (x : ℝ) → x^2 - 2 * x = 0

theorem root_product_sum_eq_zero (x1 x2 : ℝ) (hx1 : quadratic_eq x1) (hx2 : quadratic_eq x2) :
  x1 * x2 * (x1^2 + x2^2) = 0 := by
  sorry

end root_product_sum_eq_zero_l112_112457


namespace part_1_part_2_l112_112587

-- Define the triangle and the given condition
variables {α β γ : ℝ}
axiom triangle_ABC : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π
axiom sin_identity : sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)
axiom BC_length : (sin(γ) / sin(α)) * BC = 3

-- State the main theorem parts separately
theorem part_1 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_sin_identity: sin(α)^2 - sin(β)^2 - sin(γ)^2 = sin(β) * sin(γ)) :
  α = 2 * π / 3 :=
sorry

theorem part_2 (h_triangle: 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π) (h_α: α = 2 * π / 3) (h_BC_length: BC = 3) :
  let b := (2 * sqrt(3) * sin(π/6 - β)),
      c := (2 * sqrt(3) * sin(π/6 + β)) in
  (3 + 2 * sqrt(3)) :=
sorry

end part_1_part_2_l112_112587


namespace candies_shared_equally_l112_112507

theorem candies_shared_equally (total_candies : ℕ) (pct_sour : ℕ) (people : ℕ) : 
  total_candies = 300 → 
  pct_sour = 40 → 
  people = 3 → 
  (total_candies * pct_sour / 100) = 120 → 
  (total_candies - 120) = 180 → 
  (180 / people) = 60 → 
  60 := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h6, h5, h4, h3, h2, h1]
  exact 60
  sorry

end candies_shared_equally_l112_112507


namespace range_of_a_l112_112847

noncomputable def A : set ℝ := {x : ℝ | x^2 - 2 * x - 3 ≤ 0}

theorem range_of_a (a : ℝ) (h : a ∈ A) : a ∈ set.Icc (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end range_of_a_l112_112847


namespace convex_quad_is_cyclic_l112_112675

variable {α : Type*} [MetricSpace α]

def is_cyclic (A B C D : α) : Prop :=
  ∃ (O : α) (r : ℝ), dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r

def rectangle_vertices (p q : α) (R : Set α) : Prop :=
  ∃ (X Y : α), X ∈ R ∧ Y ∈ R ∧ dist p q = dist X Y ∧ right_angle p q X ∧ right_angle p q Y

def all_but_ABCD_on_circle (A B C D : α) (circ_pts : Set α) : Prop :=
  ∃ (O : α) (r : ℝ), ∀ (P ∈ circ_pts), dist O P = r ∧ P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D

theorem convex_quad_is_cyclic 
  (A B C D : α) 
  (R1 R2 R3 R4 : Set α)
  (circ_pts : Set α) 
  (h1 : rectangle_vertices A B R1)
  (h2 : rectangle_vertices B C R2)
  (h3 : rectangle_vertices C D R3)
  (h4 : rectangle_vertices D A R4)
  (h5 : all_but_ABCD_on_circle A B C D circ_pts) :
  is_cyclic A B C D := 
sorry

end convex_quad_is_cyclic_l112_112675


namespace bobby_initial_candy_l112_112347

theorem bobby_initial_candy (initial_candy : ℕ) (remaining_candy : ℕ) (extra_candy : ℕ) (total_eaten : ℕ)
  (h_candy_initial : initial_candy = 36)
  (h_candy_remaining : remaining_candy = 4)
  (h_candy_extra : extra_candy = 15)
  (h_candy_total_eaten : total_eaten = initial_candy - remaining_candy) :
  total_eaten - extra_candy = 17 :=
by
  sorry

end bobby_initial_candy_l112_112347


namespace solution_set_f_geq_3_range_of_a_for_f_geq_abs_a_minus_4_l112_112048

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 3| - |x - 2|

-- Proof Problem 1 Statement:
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≥ 1} :=
sorry

-- Proof Problem 2 Statement:
theorem range_of_a_for_f_geq_abs_a_minus_4 (a : ℝ) :
  (∃ x : ℝ, f x ≥ |a - 4|) ↔ -1 ≤ a ∧ a ≤ 9 :=
sorry

end solution_set_f_geq_3_range_of_a_for_f_geq_abs_a_minus_4_l112_112048


namespace triangle_theorem_l112_112610

theorem triangle_theorem 
  (A B C : ℝ) 
  (h1 : ∀ A B C : ℝ, ∃ (a b c : ℝ), 
      a^2 - b^2 - c^2 = b * c ∧ sin^2 A - sin^2 B - sin^2 C = sin B * sin C) 
  (h2 : 0 < A ∧ A < π) : 
  (A = 2 * π / 3) ∧ 
  (∀ (BC : ℝ), BC = 3 → ∃ (a b c : ℝ), 
      a + b + c ≤ 3 + 2 * sqrt 3) :=
by
  sorry

end triangle_theorem_l112_112610


namespace obtuse_triangle_l112_112515

theorem obtuse_triangle (α : ℝ) (h1 : ∀ x, 0 < x ∧ x < π → α = x)
  (h2 : sin α + cos α = 2 / 3) : π / 2 < α ∧ α < π :=
by
  sorry

end obtuse_triangle_l112_112515


namespace converse_proposition_false_l112_112215

theorem converse_proposition_false (a b c : ℝ) : ¬(∀ a b c : ℝ, (a > b) → (a * c^2 > b * c^2)) :=
by {
  -- proof goes here
  sorry
}

end converse_proposition_false_l112_112215


namespace exists_subset_Y_l112_112155

open Set

variable (X : Set ℤ) (X_card : X.toFinset.card = 10000) (h₀ : ∀ x ∈ X, ¬ (x % 47 = 0))

theorem exists_subset_Y :
  ∃ (Y : Set ℤ), Y ⊆ X ∧ Y.toFinset.card = 2007 ∧ 
  (∀ a b c d e ∈ Y, ¬ (47 ∣ (a - b + c - d + e))) :=
by
  sorry

end exists_subset_Y_l112_112155
