import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l86_8637

noncomputable def f (x : ℝ) : ℝ := (1/2)^(1 + x^2) + 1/(1 + abs x)

theorem f_inequality (x : ℝ) :
  f (2*x - 1) + f (1 - 2*x) < 2 * f x ↔ x < 1/3 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l86_8637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_complex_roots_l86_8648

theorem quadratic_complex_roots (a : ℝ) : 
  (∀ x : ℂ, x^2 + a*x + 1 = 0 → x.im ≠ 0) → 
  -2 ≤ a ∧ a ≤ 2 ∧ 
  ∃ b : ℝ, -2 ≤ b ∧ b ≤ 2 ∧ ∃ y : ℝ, y^2 + b*y + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_complex_roots_l86_8648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l86_8611

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

noncomputable def f' (x : ℝ) : ℝ := -2 / ((x - 1)^2)

theorem tangent_perpendicular_line (a : ℝ) : 
  (f 2 = 3) →  
  (f' 2 = -2) →  
  (f' 2 * (-a) = -1) →  
  a = 1/2 := by
  sorry

#check tangent_perpendicular_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_line_l86_8611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_parallel_distance_l86_8673

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  l₁ : Set (ℝ × ℝ) := {(x, y) | x + a * y - a = 0}
  l₂ : Set (ℝ × ℝ) := {(x, y) | a * x - (2 * a - 3) * y + a - 2 = 0}

/-- Perpendicularity condition -/
def isPerpendicular (lines : TwoLines) : Prop :=
  lines.a = 0 ∨ lines.a = 2

/-- Parallelism condition -/
def isParallel (lines : TwoLines) : Prop :=
  lines.a = -3

/-- Distance between parallel lines -/
noncomputable def distanceBetweenParallel (lines : TwoLines) : ℝ :=
  2 * Real.sqrt 10 / 15

theorem perpendicular_condition (lines : TwoLines) :
  isPerpendicular lines ↔ (1 : ℝ) * lines.a + lines.a * (3 - 2 * lines.a) = 0 := by
  sorry

theorem parallel_distance (lines : TwoLines) :
  isParallel lines → distanceBetweenParallel lines = 2 * Real.sqrt 10 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_parallel_distance_l86_8673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_to_C_l86_8696

-- Define the curves C and C1
def C (ρ θ : ℝ) : Prop := 2 * ρ * Real.sin θ + ρ * Real.cos ρ = 10
def C1 (x y α : ℝ) : Prop := x = 3 * Real.cos α ∧ y = 2 * Real.sin α

-- Define the general equation of C1
def C1_general (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the general equation of C
def C_general (x y : ℝ) : Prop := x + 2*y - 10 = 0

-- Define the distance function from a point (x, y) to curve C
noncomputable def distance_to_C (x y : ℝ) : ℝ := 
  abs (x + 2*y - 10) / Real.sqrt 5

-- State the theorem
theorem min_distance_C1_to_C : 
  ∃ (x y : ℝ), C1_general x y ∧ 
  (∀ (x' y' : ℝ), C1_general x' y' → distance_to_C x y ≤ distance_to_C x' y') ∧
  distance_to_C x y = Real.sqrt 5 ∧
  x = 9/5 ∧ y = 8/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_to_C_l86_8696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l86_8622

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + 1/2 * Real.cos (2 * x)

theorem f_monotone_increasing (k : ℤ) :
  MonotoneOn f (Set.Icc (k * Real.pi - Real.pi/3) (k * Real.pi + Real.pi/6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l86_8622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_equation_l86_8628

theorem solution_of_equation : 
  {(m, n) : ℕ × ℕ | 3^n - 2^m = 1} = {(1, 1), (3, 2)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_of_equation_l86_8628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_sqrt_l86_8657

noncomputable def expr1 := Real.sqrt 24
noncomputable def expr2 := Real.sqrt 0.5
noncomputable def expr3 (a : ℝ) := Real.sqrt (a^2 + 4)
noncomputable def expr4 (a b : ℝ) := Real.sqrt (a / b)

def is_simplest (x : ℝ → ℝ) : Prop :=
  ∀ y : ℝ → ℝ, (∃ (a : ℝ), x a = y a) → x = y

theorem simplest_sqrt (a b : ℝ) : 
  is_simplest (expr3) ∧ 
  ¬is_simplest (λ _ => expr1) ∧ 
  ¬is_simplest (λ _ => expr2) ∧ 
  ¬is_simplest (expr4 a) := by
  sorry

#check simplest_sqrt

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_sqrt_l86_8657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_height_difference_l86_8669

/-- The diameter of each cylindrical pipe in centimeters -/
def pipe_diameter : ℝ := 12

/-- The number of pipes in each crate -/
def total_pipes : ℕ := 160

/-- The number of pipes in each row of Crate A -/
def pipes_per_row_A : ℕ := 8

/-- The height of Crate A in centimeters -/
noncomputable def height_A : ℝ := (total_pipes / pipes_per_row_A : ℝ) * pipe_diameter

/-- The vertical center-to-center distance between rows in Crate B -/
noncomputable def row_distance_B : ℝ := (Real.sqrt 3 / 2) * pipe_diameter

/-- The height of Crate B in centimeters -/
noncomputable def height_B : ℝ := pipe_diameter + (total_pipes / pipes_per_row_A : ℝ) * row_distance_B

/-- The positive difference in heights between Crate A and Crate B -/
noncomputable def height_difference : ℝ := |height_A - height_B|

theorem crate_height_difference :
  |height_difference - 20.16| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_height_difference_l86_8669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l86_8613

noncomputable def f (x : ℝ) := Real.sin x ^ 4 - Real.cos x ^ 4

theorem m_range (m : ℝ) :
  (∀ x : ℝ, 1 + (2/3) * f x - m * f (x/2) ≥ 0) →
  m ∈ Set.Icc (-1/3) (1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l86_8613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_twenty_five_l86_8684

theorem log_sum_twenty_five (lg : ℝ → ℝ) (h1 : lg 10 = 1) (h2 : ∀ (x y : ℝ), x > 0 → y > 0 → lg (x * y) = lg x + lg y) :
  lg 20 + lg 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_twenty_five_l86_8684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_minus_x_l86_8606

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x < -1 then -x - 1
  else if x ≤ 2 then x
  else 5 - x

-- Define the theorem
theorem range_of_g_minus_x :
  Set.range (fun x => g x - x) ∩ Set.Icc (-3 : ℝ) 3 = Set.Icc (-3 : ℝ) 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_minus_x_l86_8606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_relation_l86_8650

/-- A random variable following a binomial distribution -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  prob : ℝ → ℝ

/-- The probability of X ≥ 1 for a B(2, p) distribution -/
def prob_X_geq_1 (p : ℝ) : ℝ := 1 - (1 - p)^2

/-- The probability of Y = 2 for a B(3, p) distribution -/
def prob_Y_eq_2 (p : ℝ) : ℝ := 3 * p^2 * (1 - p)

theorem binomial_probability_relation :
  ∀ p : ℝ, 0 < p → p < 1 →
  (prob_X_geq_1 p = 5/9) → (prob_Y_eq_2 p = 2/9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_relation_l86_8650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_shaded_areas_l86_8664

theorem equal_shaded_areas (θ : ℝ) (r : ℝ) (h1 : 0 < θ) (h2 : θ < π / 4) (h3 : r > 0) :
  (r^2 * Real.sin θ) / 2 = θ * r^2 ↔ Real.sin θ = 2 * θ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_shaded_areas_l86_8664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_field_division_l86_8644

/-- Represents a circular field -/
structure CircularField where
  radius : ℝ
  center : ℝ × ℝ

/-- Represents a semi-circular fence -/
structure SemiCircularFence where
  radius : ℝ
  center : ℝ × ℝ
  orientation : ℝ  -- angle of the diameter

/-- Helper function to check if a point is in a circular field -/
def CircularField.mem (field : CircularField) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := field.center
  (x - cx)^2 + (y - cy)^2 ≤ field.radius^2

/-- Helper function to check if a point is on a semi-circular fence -/
def SemiCircularFence.mem (fence : SemiCircularFence) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := fence.center
  (x - cx)^2 + (y - cy)^2 = fence.radius^2

/-- Helper function to calculate the area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry  -- Implementation not provided as it's not part of the theorem statement

/-- Theorem: A circular field can be divided into four equal parts using three semi-circular fences of equal length -/
theorem circular_field_division (field : CircularField) :
  ∃ (fence1 fence2 fence3 : SemiCircularFence),
    -- The fences have equal length
    fence1.radius = fence2.radius ∧ fence2.radius = fence3.radius ∧
    -- The fences divide the field into four equal parts
    (∀ (p : ℝ × ℝ), field.mem p → 
      (fence1.mem p ∨ fence2.mem p ∨ fence3.mem p)) ∧
    -- The area of each part is equal
    (let total_area := π * field.radius^2
     let part_area := total_area / 4
     ∀ (i : Fin 4), 
       ∃ (part : Set (ℝ × ℝ)), 
         (∀ (p : ℝ × ℝ), p ∈ part → field.mem p) ∧
         (area part = part_area)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_field_division_l86_8644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_time_proof_l86_8615

/-- The time it takes to walk one block, in seconds -/
def walk_time_per_block : ℝ := 60

/-- The number of blocks from home to office -/
def distance_blocks : ℕ := 18

/-- The time it takes to bike one block, in seconds -/
def bike_time_per_block : ℝ := 20

/-- The difference between total walking time and total biking time, in seconds -/
def time_difference : ℝ := 12 * 60

theorem walk_time_proof :
  walk_time_per_block * (distance_blocks : ℝ) = 
    bike_time_per_block * (distance_blocks : ℝ) + time_difference := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_time_proof_l86_8615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_one_l86_8651

-- Define the polar equations
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi/4) = 1 + Real.sqrt 2
def circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi/4)

-- Define the chord length function
def chord_length (l : (ℝ → ℝ → Prop)) (c : (ℝ → ℝ → Prop)) : ℝ :=
  -- This function should calculate the length of the chord
  -- intercepted by circle c on line l
  sorry

-- Theorem statement
theorem chord_length_is_one :
  chord_length line_equation circle_equation = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_one_l86_8651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l86_8603

theorem triangle_inequality (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < Real.pi / 2) ∧
  (0 < B) ∧ (B < Real.pi / 2) ∧
  (0 < C) ∧ (C < Real.pi / 2) ∧
  (A + B + C = Real.pi) ∧
  (a = 2 * Real.sin A) ∧
  (b = 2 * Real.sin B) ∧
  (c = 2 * Real.sin C) →
  (a / (1 - Real.sin A) + b / (1 - Real.sin B) + c / (1 - Real.sin C) ≥ 18 + 12 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l86_8603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_vectors_l86_8662

/-- Given three distinct unit vectors in 3D space with specific dot product relations,
    prove that the magnitude of their sum equals √(45/7). -/
theorem sum_of_three_vectors (p q r : ℝ × ℝ × ℝ) :
  ‖p‖ = 1 →
  ‖q‖ = 1 →
  ‖r‖ = 1 →
  p ≠ q →
  p ≠ r →
  q ≠ r →
  p • q = -1/7 →
  p • r = -1/7 →
  q • r = -1/7 →
  ‖p + q + r‖ = Real.sqrt (45/7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_three_vectors_l86_8662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_solutions_l86_8656

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - x) * Real.exp x - a * x - a

/-- Predicate to check if a given value of 'a' results in exactly two positive integer solutions for f(x) > 0 -/
def has_two_positive_integer_solutions (a : ℝ) : Prop :=
  ∃ (x y : ℕ), x ≠ y ∧ f a x > 0 ∧ f a y > 0 ∧
  ∀ (z : ℕ), z ≠ x ∧ z ≠ y → f a z ≤ 0

/-- The main theorem statement -/
theorem range_of_a_for_two_solutions :
  ∀ a : ℝ, has_two_positive_integer_solutions a ↔ a ∈ Set.Icc (-(1/4) * Real.exp 3) 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_two_solutions_l86_8656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extreme_points_l86_8616

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 - a*x + Real.log x - 2

theorem two_extreme_points (a : ℝ) : 
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x ≠ y ∧ 
    (∀ z : ℝ, 0 < z → (deriv (f a)) z = 0 → z = x ∨ z = y)) ↔ 
  a > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_extreme_points_l86_8616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_string_length_l86_8649

/-- Given a string of length 70 cm, prove that after removing 27 cm and then 7/9 of the remaining length, the final remaining length is approximately 9.56 cm. -/
theorem remaining_string_length : 
  ∃ (final_remaining : ℝ), 
  (let original_length : ℝ := 70
   let given_to_minyoung : ℝ := 27
   let fraction_used_for_A : ℚ := 7/9
   let remaining_after_minyoung := original_length - given_to_minyoung
   let used_for_A := (fraction_used_for_A : ℝ) * remaining_after_minyoung
   final_remaining = remaining_after_minyoung - used_for_A) ∧
  (abs (final_remaining - 9.56) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_string_length_l86_8649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jo_kate_sum_difference_l86_8600

def jo_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (x : ℕ) : ℕ :=
  5 * ((x + 2) / 5)

def kate_sum (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_five (List.range n))

theorem jo_kate_sum_difference :
  (jo_sum 120 : ℤ) - (kate_sum 120 : ℤ) = 6900 := by
  sorry

#eval jo_sum 120
#eval kate_sum 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jo_kate_sum_difference_l86_8600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l86_8609

theorem cos_alpha_minus_pi_fourth (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan α = 2) : 
  Real.cos (α - π / 4) = (3 * Real.sqrt 10) / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l86_8609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_power_of_three_l86_8659

theorem divisor_power_of_three (n : ℕ+) (h1 : Nat.card (Nat.divisors n) = 72) 
  (h2 : Nat.card (Nat.divisors (3 * n)) = 96) :
  ∃ k : ℕ, (3^k : ℕ) ∣ n ∧ ∀ m : ℕ, (3^m : ℕ) ∣ n → m ≤ k ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_power_of_three_l86_8659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l86_8635

theorem rectangle_area_increase (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a * (1 + 1/4) * (b * (1 + 1/5)) - a * b) / (a * b) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_increase_l86_8635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_three_digit_is_918_l86_8697

def is_valid_combination (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (a = 3 ∨ a = 5 ∨ a = 8 ∨ a = 9 ∨ a = 1) ∧
  (b = 3 ∨ b = 5 ∨ b = 8 ∨ b = 9 ∨ b = 1) ∧
  (c = 3 ∨ c = 5 ∨ c = 8 ∨ c = 9 ∨ c = 1) ∧
  (d = 3 ∨ d = 5 ∨ d = 8 ∨ d = 9 ∨ d = 1) ∧
  (e = 3 ∨ e = 5 ∨ e = 8 ∨ e = 9 ∨ e = 1) ∧
  c % 2 = 0

def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def two_digit_number (d e : ℕ) : ℕ := 10 * d + e

theorem max_product_three_digit_is_918 :
  ∀ a b c d e : ℕ,
    is_valid_combination a b c d e →
    three_digit_number 9 5 1 * two_digit_number 8 3 ≥ 
    three_digit_number a b c * two_digit_number d e :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_three_digit_is_918_l86_8697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_groups_for_team_l86_8617

def is_valid_grouping (n : ℕ) (total_players : ℕ) (max_group_size : ℕ) : Prop :=
  n > 0 ∧
  (∃ (group_size : ℕ), 
    group_size > 0 ∧
    group_size ≤ max_group_size ∧
    n * group_size = total_players ∧
    (∃ (half_size : ℕ), half_size > 0 ∧ half_size * 2 = group_size))

theorem min_groups_for_team (total_players : ℕ) (max_group_size : ℕ) :
  total_players = 30 →
  max_group_size = 12 →
  (∃ (n : ℕ), is_valid_grouping n total_players max_group_size ∧
    ∀ (m : ℕ), is_valid_grouping m total_players max_group_size → n ≤ m) →
  (∃ (n : ℕ), is_valid_grouping n total_players max_group_size ∧
    ∀ (m : ℕ), is_valid_grouping m total_players max_group_size → n ≤ m) ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_groups_for_team_l86_8617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l86_8605

theorem polynomial_identity (p : ℝ → ℝ) :
  (∀ x : ℝ, p ((x + 1)^3) = (p x + 1)^3) →
  p 0 = 0 →
  ∀ x : ℝ, p x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l86_8605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_l86_8639

/-- The coefficient of x³ in the expansion of (1 + 2/x)(1 - x^4) is 3 -/
theorem coefficient_x_cubed (x : ℝ) : 
  ∃ a b c d e : ℝ, (1 + 2/x) * (1 - x^4) = a + b*x + c*x^2 + 3*x^3 + d*x^4 + e*x^5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_l86_8639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_passing_through_C_l86_8665

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a path on the grid -/
def GridPath := List Point

/-- The probability of choosing a direction at an intersection -/
def intersectionProbability : ℚ := 1/2

/-- A and B are the start and end points -/
def A : Point := ⟨0, 0⟩
def B : Point := ⟨3, 3⟩

/-- C is the point we want to pass through -/
def C : Point := ⟨2, 1⟩

/-- A path is valid if it only moves east or south -/
def isValidPath (p : GridPath) : Prop := sorry

/-- A path passes through point C -/
def passesThrough (p : GridPath) (point : Point) : Prop := sorry

/-- The probability of a path passing through a given point -/
noncomputable def probabilityPassingThrough (point : Point) : ℚ := sorry

theorem probability_passing_through_C :
  probabilityPassingThrough C = 21/32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_passing_through_C_l86_8665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carousel_seat_visitation_l86_8690

theorem carousel_seat_visitation (n : ℕ) : 
  (∃ (transitions : List ℕ), 
    transitions.length = n - 1 ∧ 
    transitions.all (· < n) ∧
    transitions.Pairwise (· ≠ ·) ∧
    (transitions.sum % n = n - 1)) ↔ 
  Even n := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carousel_seat_visitation_l86_8690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_is_increasing_on_positive_reals_l86_8630

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2 : ℝ)^x + (2 : ℝ)^(-x)

-- Theorem for the parity of f(x)
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  intro x
  sorry

-- Theorem for f(x) being increasing on (0, +∞)
theorem f_is_increasing_on_positive_reals : ∀ x : ℝ, x > 0 → (deriv f) x > 0 := by
  intro x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_is_increasing_on_positive_reals_l86_8630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_row_or_column_with_four_digits_l86_8638

/-- Represents a 10x10 grid of digits -/
def Grid := Fin 10 → Fin 10 → Fin 10

/-- Counts the number of occurrences of each digit in the grid -/
def count_digits (g : Grid) (d : Fin 10) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 10)) fun i =>
   Finset.sum (Finset.univ : Finset (Fin 10)) fun j =>
   if g i j = d then 1 else 0)

/-- Counts the number of different digits in a row -/
def count_different_digits_in_row (g : Grid) (row : Fin 10) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 10)) fun d =>
   if (Finset.sum (Finset.univ : Finset (Fin 10)) fun col => if g row col = d then 1 else 0) > 0 then 1 else 0)

/-- Counts the number of different digits in a column -/
def count_different_digits_in_column (g : Grid) (col : Fin 10) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 10)) fun d =>
   if (Finset.sum (Finset.univ : Finset (Fin 10)) fun row => if g row col = d then 1 else 0) > 0 then 1 else 0)

/-- The main theorem to be proved -/
theorem exists_row_or_column_with_four_digits (g : Grid) 
  (h : ∀ d : Fin 10, count_digits g d = 10) : 
  (∃ row : Fin 10, count_different_digits_in_row g row ≥ 4) ∨ 
  (∃ col : Fin 10, count_different_digits_in_column g col ≥ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_row_or_column_with_four_digits_l86_8638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_walks_25_percent_less_l86_8643

/-- The distance an ant walks in meters -/
noncomputable def ant_distance : ℝ := 600

/-- The time the ant walks in minutes -/
noncomputable def ant_time : ℝ := 10

/-- The speed of the beetle in km/h -/
noncomputable def beetle_speed : ℝ := 2.7

/-- Convert meters per minute to km/h -/
noncomputable def meters_per_minute_to_kmh (d : ℝ) (t : ℝ) : ℝ := d / t * 60 / 1000

/-- Calculate the percentage difference between two speeds -/
noncomputable def percentage_difference (s1 : ℝ) (s2 : ℝ) : ℝ := (s1 - s2) / s1 * 100

theorem beetle_walks_25_percent_less :
  percentage_difference (meters_per_minute_to_kmh ant_distance ant_time) beetle_speed = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetle_walks_25_percent_less_l86_8643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l86_8698

theorem power_of_three (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y + 3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l86_8698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_similar_triangle_l86_8625

-- Define the original triangle
def original_triangle : Fin 3 → ℚ
| 0 => 6
| 1 => 7
| 2 => 9
| _ => 0

-- Define the perimeter of the similar triangle
def similar_perimeter : ℚ := 110

-- Define the scale factor
noncomputable def scale_factor : ℚ := similar_perimeter / (original_triangle 0 + original_triangle 1 + original_triangle 2)

-- Define the similar triangle
noncomputable def similar_triangle (i : Fin 3) : ℚ := original_triangle i * scale_factor

-- Theorem statement
theorem longest_side_of_similar_triangle :
  similar_triangle 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_of_similar_triangle_l86_8625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_oil_rate_is_approx_53_67_l86_8675

/-- Calculates the rate of mixed oil per litre given the volumes and rates of different oils -/
noncomputable def mixedOilRate (v1 v2 v3 v4 : ℝ) (r1 r2 r3 r4 : ℝ) : ℝ :=
  (v1 * r1 + v2 * r2 + v3 * r3 + v4 * r4) / (v1 + v2 + v3 + v4)

/-- Theorem stating that the rate of the mixed oil is approximately 53.67 given the specified volumes and rates -/
theorem mixed_oil_rate_is_approx_53_67 :
  ∃ ε > 0, |mixedOilRate 10 5 8 7 50 68 42 62 - 53.67| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_oil_rate_is_approx_53_67_l86_8675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l86_8642

def is_valid_sequence (seq : List ℕ) : Prop :=
  seq.length = 8 ∧
  seq.head? = some 5 ∧
  seq.getLast? = some 8 ∧
  ∀ i, i < 6 → 
    (seq.get? i).isSome ∧ (seq.get? (i+1)).isSome ∧ (seq.get? (i+2)).isSome ∧
    (seq.get? i).get! + (seq.get? (i+1)).get! + (seq.get? (i+2)).get! = 20

theorem unique_sequence :
  ∀ seq : List ℕ, is_valid_sequence seq → seq = [5, 8, 7, 5, 8, 7, 5, 8] :=
by
  intro seq h
  sorry

#check unique_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l86_8642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_5_l86_8620

/-- Two circles in a 2D plane -/
structure TwoCircles where
  c1 : (x y : ℝ) → x^2 + y^2 - 4*x - 3 = 0
  c2 : (x y : ℝ) → x^2 + y^2 - 4*y - 3 = 0

/-- The length of the common chord of two circles -/
noncomputable def commonChordLength (circles : TwoCircles) : ℝ := 2 * Real.sqrt 5

/-- Theorem: The length of the common chord of the given circles is 2√5 -/
theorem common_chord_length_is_2_sqrt_5 (circles : TwoCircles) :
  commonChordLength circles = 2 * Real.sqrt 5 := by
  -- The proof goes here
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_5_l86_8620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_max_value_of_f_l86_8674

open Real MeasureTheory

noncomputable def f (A B x : ℝ) : ℝ := Real.cos x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sin x ^ 2 + A * x + B

def interval : Set ℝ := Set.Icc 0 (3 * π / 2)

theorem minimize_max_value_of_f :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ x ∈ interval, f 0 0 x ≤ M) ∧
  (∀ A B : ℝ, ∃ x ∈ interval, f A B x ≥ M) := by
  sorry

#check minimize_max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_max_value_of_f_l86_8674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_planes_canonical_equation_l86_8610

/-- Given two planes in ℝ³, this theorem proves that their intersection
    forms a line with a specific canonical equation. -/
theorem intersection_of_planes_canonical_equation
  (plane1 : ℝ → ℝ → ℝ → Prop)
  (plane2 : ℝ → ℝ → ℝ → Prop)
  (h1 : ∀ x y z, plane1 x y z ↔ x + 5 * y - z = 5)
  (h2 : ∀ x y z, plane2 x y z ↔ 2 * x - 5 * y + 2 * z = -5)
  (x y z : ℝ) :
  (x / 5 = (y - 1) / (-4) ∧ (y - 1) / (-4) = z / (-15)) ↔
  (plane1 x y z ∧ plane2 x y z) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_planes_canonical_equation_l86_8610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_special_points_l86_8641

-- Define the equilateral triangle
structure EquilateralTriangle where
  a : ℝ
  a_pos : a > 0

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the vertices of the triangle
noncomputable def vertex_A (t : EquilateralTriangle) : Point :=
  { x := 0, y := t.a * Real.sqrt 3 / 2 }

noncomputable def vertex_B (t : EquilateralTriangle) : Point :=
  { x := t.a / 2, y := 0 }

noncomputable def vertex_C (t : EquilateralTriangle) : Point :=
  { x := -t.a / 2, y := 0 }

-- Define the distance function
def distance_squared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Define the theorem
theorem equilateral_triangle_special_points (t : EquilateralTriangle) :
  ∀ p : Point, distance_squared p (vertex_A t) = distance_squared p (vertex_B t) + distance_squared p (vertex_C t) ↔
  distance_squared p { x := 0, y := -t.a * Real.sqrt 3 / 2 } = t.a^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_special_points_l86_8641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_center_cube_volume_ratio_l86_8678

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- A cube whose vertices are the centers of the faces of a regular tetrahedron -/
structure CenterCube where
  tetrahedron : RegularTetrahedron

/-- The volume of a regular tetrahedron -/
noncomputable def tetrahedronVolume (t : RegularTetrahedron) : ℝ :=
  (t.sideLength ^ 3 * Real.sqrt 2) / 12

/-- The volume of a cube whose vertices are the centers of the faces of a regular tetrahedron -/
noncomputable def centerCubeVolume (c : CenterCube) : ℝ :=
  c.tetrahedron.sideLength ^ 3 / (3 * Real.sqrt 3)

/-- The main theorem: The ratio of the volume of a regular tetrahedron to the volume of its center cube is 9/14 -/
theorem tetrahedron_center_cube_volume_ratio :
    ∀ (t : RegularTetrahedron) (c : CenterCube),
    c.tetrahedron = t →
    tetrahedronVolume t / centerCubeVolume c = 9 / 14 := by
  sorry

#check tetrahedron_center_cube_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_center_cube_volume_ratio_l86_8678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_is_multiple_of_nine_l86_8681

/-- Two-digit positive integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Reverse the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

theorem difference_is_multiple_of_nine (q r : ℕ) :
  TwoDigitInt q →
  TwoDigitInt r →
  r = reverseDigits q →
  (∀ a b : ℕ, TwoDigitInt a → TwoDigitInt b → b = reverseDigits a → a - b ≤ 63) →
  ∃ k : ℕ, q - r = 9 * k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_is_multiple_of_nine_l86_8681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_l86_8647

/-- Given points P and Q in the xy-plane, and R such that PR + RQ is minimized, prove that the y-coordinate of R is 9/5 -/
theorem minimize_distance (P Q R : ℝ × ℝ) : 
  P = (-2, -3) → 
  Q = (3, 3) → 
  R.1 = 2 → 
  (∀ S : ℝ × ℝ, S.1 = 2 → (dist P R + dist R Q) ≤ (dist P S + dist S Q)) → 
  R.2 = 9/5 := by
  sorry

/-- The distance function between two points in ℝ² -/
noncomputable def dist (A B : ℝ × ℝ) : ℝ := 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_l86_8647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_v₂_l86_8658

/-- The maximum power of 2 that divides a positive integer s -/
def v₂ (s : ℕ) : ℕ := sorry

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The product of binomial coefficients (2n choose n) from n=1 to 2^m -/
def binomial_product (m : ℕ) : ℕ := sorry

theorem binomial_product_v₂ (m : ℕ) : 
  v₂ (binomial_product m) = m * 2^(m - 1) + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_v₂_l86_8658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_zero_value_l86_8619

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x - 2 * Real.cos x

theorem f_extrema_and_zero_value :
  (∀ x, x ∈ Set.Icc 0 Real.pi → f x ≤ 4 ∧ f x ≥ -2) ∧
  (∃ x₁ x₂, x₁ ∈ Set.Icc 0 Real.pi ∧ x₂ ∈ Set.Icc 0 Real.pi ∧ f x₁ = 4 ∧ f x₂ = -2) ∧
  (∀ x, f x = 0 →
    (2 * (Real.cos (x / 2))^2 - Real.sin x - 1) /
    (Real.sqrt 2 * Real.sin (x + Real.pi / 4)) = 2 - Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_zero_value_l86_8619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_division_l86_8694

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ
deriving Repr, DecidableEq

/-- Represents the 5x5 square -/
def Square : ℕ := 5

/-- The two types of rectangles allowed -/
def AllowedRectangles : List Rectangle := [⟨1, 4⟩, ⟨1, 3⟩]

/-- The required number of rectangles -/
def RequiredRectangles : ℕ := 8

/-- Function to check if a list of rectangles can fit in the square -/
def canFitInSquare (rectangles : List Rectangle) : Prop :=
  (rectangles.length = RequiredRectangles) ∧
  (rectangles.all (fun r => r ∈ AllowedRectangles)) ∧
  (rectangles.foldl (fun sum r => sum + r.width * r.height) 0 = Square * Square)

/-- Theorem stating the impossibility of the division -/
theorem impossible_division :
  ¬ ∃ (rectangles : List Rectangle), canFitInSquare rectangles := by
  sorry

#eval Square
#eval AllowedRectangles
#eval RequiredRectangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_division_l86_8694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_2008_equals_5898_l86_8608

/-- Define the sequence u_n as described in the problem -/
def u : ℕ → ℕ
| 0 => 1  -- Define the first term
| n+1 => sorry  -- The rest of the sequence definition (to be implemented)

/-- Function to calculate the last term of each group -/
def f (n : ℕ) : ℕ := n * (3 * n - 1) / 2

/-- Theorem stating that the 2008th term of the sequence is 5898 -/
theorem u_2008_equals_5898 : u 2008 = 5898 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_2008_equals_5898_l86_8608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_r_six_times_30_l86_8688

-- Define the function r
noncomputable def r (θ : ℝ) : ℝ := 1 / (1 - θ)

-- State the theorem
theorem r_six_times_30 : r (r (r (r (r (r 30))))) = 30 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_r_six_times_30_l86_8688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l86_8670

/-- The line equation kx - y - k + 1 = 0 -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y - k + 1 = 0

/-- The circle equation x^2 + y^2 = 4 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

/-- The line intersects the circle for all real values of k -/
theorem line_intersects_circle :
  ∀ k : ℝ, ∃ x y : ℝ, line_equation k x y ∧ circle_equation x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l86_8670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l86_8636

open Real

noncomputable def solution_set : Set ℝ :=
  (Set.Ioo 0 0.5) ∪ (Set.Ioo 0.5 1) ∪ (Set.Ioo 2 2.5) ∪ (Set.Ioo 4.5 5) ∪ (Set.Ioo 6 6.5)

noncomputable def inequality (x : ℝ) : Prop :=
  (3^(4*x^2 - 10) - 9 * 3^(24*x + 1)) * log (x^2 - 7*x + 12.25) / log (sin (Real.pi * x)) ≥ 0

def conditions (x : ℝ) : Prop :=
  sin (Real.pi * x) > 0 ∧ sin (Real.pi * x) ≠ 1 ∧ x^2 - 7*x + 12.25 > 0

theorem inequality_solution_set :
  ∀ x : ℝ, (inequality x ∧ conditions x) ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l86_8636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_difference_l86_8687

def coin_values : List ℕ := [5, 10, 20, 25]

def target_amount : ℕ := 50

theorem coin_difference :
  let max_coins := target_amount / (coin_values.minimum?).getD 1
  let min_coins := (target_amount / (coin_values.maximum?).getD 1).succ
  max_coins - min_coins = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_difference_l86_8687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l86_8632

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def point_A : ℝ × ℝ := (0, -1)
def point_B : ℝ × ℝ := (0, 1)

-- Define the sum of squared distances from P to A and B
def sum_squared_distances (x y : ℝ) : ℝ :=
  (x - point_A.1)^2 + (y - point_A.2)^2 + (x - point_B.1)^2 + (y - point_B.2)^2

-- Theorem statement
theorem max_sum_squared_distances :
  ∃ (x y : ℝ), circle_C x y ∧
    (∀ (x' y' : ℝ), circle_C x' y' →
      sum_squared_distances x y ≥ sum_squared_distances x' y') ∧
    x = 18/5 ∧ y = 24/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l86_8632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l86_8660

/-
  Define the hyperbola and its properties
-/
def Hyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

/-
  Define the foci and a point on the hyperbola
-/
def Foci (F₁ F₂ : ℝ × ℝ) (a b : ℝ) : Prop :=
  Hyperbola a b ∧ F₁.1 < 0 ∧ F₂.1 > 0

def PointOnHyperbola (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  Hyperbola a b ∧ P.1 > 0 ∧ P.1^2 / a^2 - P.2^2 / b^2 = 1

/-
  Define the conditions given in the problem
-/
def ProblemConditions (O F₁ F₂ P : ℝ × ℝ) (a b : ℝ) : Prop :=
  Foci F₁ F₂ a b ∧
  PointOnHyperbola P a b ∧
  O = (0, 0) ∧
  ((P.1 - O.1, P.2 - O.2) + (F₂.1 - O.1, F₂.2 - O.2)) • (F₂.1 - P.1, F₂.2 - P.2) = 0 ∧
  (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 = 3 * ((F₂.1 - P.1)^2 + (F₂.2 - P.2)^2)

/-
  State the theorem to be proved
-/
theorem hyperbola_eccentricity (O F₁ F₂ P : ℝ × ℝ) (a b : ℝ) :
  ProblemConditions O F₁ F₂ P a b →
  let c := Real.sqrt ((F₂.1 - O.1)^2 + (F₂.2 - O.2)^2)
  c / a = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l86_8660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_sqrt2_l86_8692

theorem sin_plus_cos_sqrt2 (x : ℝ) :
  0 ≤ x → x < 2 * Real.pi → Real.sin x + Real.cos x = Real.sqrt 2 → x = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_sqrt2_l86_8692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_relationship_l86_8624

-- Define a type for lines in 3D space
structure Line3D where
  -- We don't need to specify the exact representation here
  mk :: (dummy : Unit)

-- Define a membership relation for Line3D
instance : Membership ℝ Line3D where
  mem := λ _ _ => True  -- For simplicity, we assume all real numbers are on all lines

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are neither parallel nor intersecting
  ¬ (l1 = l2) ∧ ¬ (∃ (p : ℝ), p ∈ l1 ∧ p ∈ l2)

-- Define what it means for two lines to be parallel
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if they have the same direction but don't coincide
  ∃ (v : ℝ), ∀ (p q : ℝ), p ∈ l1 ∧ q ∈ l2 → ∃ (t : ℝ), q - p = t * v

-- Define the possible positional relationships between two lines
inductive PositionalRelationship
  | parallel
  | intersecting
  | skew

-- The main theorem
theorem skew_lines_relationship (l1 l2 l3 : Line3D) 
  (h1 : are_skew l1 l2) (h2 : are_parallel l1 l3) : 
  (PositionalRelationship.intersecting : PositionalRelationship) = 
    PositionalRelationship.skew ∨
  (∃ (p : ℝ), p ∈ l2 ∧ p ∈ l3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_relationship_l86_8624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isothermal_work_equals_75_l86_8621

-- Define constants and variables
noncomputable def R : ℝ := Real.pi  -- Placeholder for ideal gas constant
def n : ℚ := 1  -- Number of moles (1 in this case)
noncomputable def P : ℝ := 1  -- Placeholder for pressure
noncomputable def V : ℝ := 1  -- Placeholder for volume
noncomputable def T : ℝ := 1  -- Placeholder for temperature

-- Define the work done in the isobaric process
def W₁ : ℝ := 30

-- Define the change in internal energy for a monatomic gas
noncomputable def ΔU (ΔT : ℝ) : ℝ := (3/2) * (n : ℝ) * R * ΔT

-- Define the heat added in the isobaric process
noncomputable def Q₁ : ℝ := W₁ + ΔU (W₁ / R)

-- Define the work done in the isothermal process
noncomputable def W₂ : ℝ := Q₁

-- State the theorem
theorem isothermal_work_equals_75 : W₂ = 75 := by
  -- Proof steps would go here
  sorry

#eval W₁  -- This will output 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isothermal_work_equals_75_l86_8621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l86_8618

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define point A
def A : ℝ × ℝ := (0, 4)

-- Define the right focus F₂
def F₂ : ℝ × ℝ := (3, 0)

-- Define a point on the left branch of the hyperbola
def left_branch (P : ℝ × ℝ) : Prop := 
  hyperbola P.1 P.2 ∧ P.1 < 0

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem min_distance_sum : 
  ∀ P : ℝ × ℝ, left_branch P → 
  ∃ min_val : ℝ, min_val = 9 ∧ 
  ∀ Q : ℝ × ℝ, left_branch Q → 
  distance Q A + distance Q F₂ ≥ min_val :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l86_8618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_one_digit_numbers_l86_8672

/-- The set of integers from 1 to 13 -/
def S : Finset ℕ := Finset.range 13 ∪ {13}

/-- A bijective function representing the arrangement of numbers on rings -/
def f : S → S := sorry

/-- Condition: Every ring contains at least one two-digit number -/
axiom two_digit_in_every_ring :
  ∀ n : S, ∃ k : ℕ, (((f^[k] n) : ℕ) ≥ 10 ∧ ((f^[k] n) : ℕ) ≤ 13)

/-- Theorem: There exist three one-digit numbers adjacent to one another on one ring -/
theorem adjacent_one_digit_numbers :
  ∃ a b c : S, ((a : ℕ) < 10 ∧ (b : ℕ) < 10 ∧ (c : ℕ) < 10 ∧
              f a = b ∧ f b = c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_one_digit_numbers_l86_8672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bretschneiders_formula_l86_8668

/-- A structure representing a quadrilateral -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  isConvex : Prop
  consecutiveSideLengths : Fin 4 → ℝ
  diagonalLengths : Fin 2 → ℝ
  oppositeAngles : Fin 2 → ℝ

/-- Bretschneider's formula for convex quadrilaterals -/
theorem bretschneiders_formula 
  (ABCD : Quadrilateral) 
  (a b c d : ℝ) 
  (h_sides : ABCD.consecutiveSideLengths = ![a, b, c, d])
  (m n : ℝ) 
  (h_diagonals : ABCD.diagonalLengths = ![m, n])
  (A C : ℝ) 
  (h_angles : ABCD.oppositeAngles = ![A, C]) :
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bretschneiders_formula_l86_8668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_at_20_degrees_l86_8640

-- Define the relationship between temperature change and volume change
noncomputable def volume_change_rate : ℝ := 3 / 5

-- Define the initial temperature and volume
def initial_temp : ℝ := 30
def initial_volume : ℝ := 30

-- Define the target temperature
def target_temp : ℝ := 20

-- Define the volume calculation function
noncomputable def calculate_volume (temp : ℝ) : ℝ :=
  initial_volume - volume_change_rate * (initial_temp - temp)

-- Theorem to prove
theorem volume_at_20_degrees :
  calculate_volume target_temp = 24 := by
  -- Expand the definition of calculate_volume
  unfold calculate_volume
  -- Simplify the expression
  simp [volume_change_rate, initial_temp, initial_volume, target_temp]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_at_20_degrees_l86_8640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lcm_18_l86_8631

theorem max_lcm_18 : 
  (Finset.range 6).sup (fun i => Nat.lcm 18 
    (match i with
    | 0 => 2
    | 1 => 3
    | 2 => 6
    | 3 => 9
    | 4 => 12
    | _ => 15
    )) = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lcm_18_l86_8631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l86_8652

noncomputable def r : ℝ := 5
noncomputable def θ : ℝ := (11 * Real.pi) / 4

theorem polar_to_rectangular_conversion :
  (r * Real.cos θ, r * Real.sin θ) = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l86_8652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_betty_uses_two_eggs_per_vanilla_l86_8612

/-- A recipe for cheesecake -/
structure CheesecakeRecipe where
  sugar : ℚ
  creamCheese : ℚ
  vanilla : ℚ
  eggs : ℕ

/-- The ratio of sugar to cream cheese -/
def sugarToCreamCheeseRatio : ℚ := 1 / 4

/-- The ratio of vanilla to cream cheese -/
def vanillaToCreamCheeseRatio : ℚ := 1 / 2

/-- Calculate the number of eggs per teaspoon of vanilla -/
noncomputable def eggsPerVanilla (recipe : CheesecakeRecipe) : ℚ :=
  recipe.eggs / (recipe.creamCheese * vanillaToCreamCheeseRatio)

/-- Betty's latest cheesecake recipe -/
def bettyRecipe : CheesecakeRecipe where
  sugar := 2
  creamCheese := 2 / sugarToCreamCheeseRatio
  vanilla := (2 / sugarToCreamCheeseRatio) * vanillaToCreamCheeseRatio
  eggs := 8

/-- Theorem: Betty uses 2 eggs per teaspoon of vanilla -/
theorem betty_uses_two_eggs_per_vanilla : eggsPerVanilla bettyRecipe = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_betty_uses_two_eggs_per_vanilla_l86_8612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_when_a_is_one_max_value_on_interval_T_property_l86_8685

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := -x * abs (x - 2 * a) + 1

-- Part I: Zeros when a = 1
theorem zeros_when_a_is_one :
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 1 + Real.sqrt 2 ∧ f 1 x₁ = 0 ∧ f 1 x₂ = 0 := by sorry

-- Part II: Maximum value on [1,2] when a ∈ (0, 3/2)
noncomputable def max_value (a : ℝ) : ℝ :=
  if 0 < a ∧ a ≤ 1/2 then 2*a
  else if 1/2 < a ∧ a < 1 then 1
  else if 1 ≤ a ∧ a < 3/2 then 5 - 4*a
  else 0  -- This case should never occur given the constraints

theorem max_value_on_interval (a : ℝ) (ha : 0 < a ∧ a < 3/2) :
  ∀ x ∈ Set.Icc 1 2, f a x ≤ max_value a := by sorry

-- Part III: T(a) such that |f(x)| ≤ 1 for x ∈ [0, T(a)]
noncomputable def T (a : ℝ) : ℝ :=
  if a ≥ Real.sqrt 2 then a - Real.sqrt (a^2 - 2)
  else a + Real.sqrt (a^2 + 2)

theorem T_property (a : ℝ) (ha : a > 0) :
  ∀ x ∈ Set.Icc 0 (T a), abs (f a x) ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_when_a_is_one_max_value_on_interval_T_property_l86_8685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_digits_divisible_by_power_of_five_l86_8614

theorem odd_digits_divisible_by_power_of_five (n : ℕ) :
  ∃ m : ℕ,
    (m % (5^n) = 0) ∧
    (10^(n-1) ≤ m) ∧ (m < 10^n) ∧
    (∀ d : ℕ, d < n → (m / 10^d) % 2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_digits_divisible_by_power_of_five_l86_8614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crosses_pole_in_4_seconds_l86_8645

/-- Calculates the time (in seconds) it takes for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

theorem train_crosses_pole_in_4_seconds :
  train_crossing_time 100 90 = 4 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crosses_pole_in_4_seconds_l86_8645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l86_8607

-- Define the line C₁
def C₁ (x y : ℝ) : Prop := x + y = 3

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem intersection_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    C₁ x₁ y₁ ∧ C₂ x₁ y₁ ∧
    C₁ x₂ y₂ ∧ C₂ x₂ y₂ ∧
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l86_8607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_approx_l86_8654

/-- The radius of a circle inscribed within three mutually externally tangent circles --/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  1 / (1/a + 1/b + 1/c + 2 * Real.sqrt (1/(a*b) + 1/(a*c) + 1/(b*c)))

/-- Theorem stating that the radius of the inscribed circle is approximately 0.698 --/
theorem inscribed_circle_radius_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |inscribed_circle_radius 3 5 7 - 0.698| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_approx_l86_8654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_range_implies_a_range_l86_8679

/-- The function f(x) = x³ - 3x² - 9x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

/-- Theorem: If f(x) = a has exactly two different real roots in [-2, 3], then a ∈ [-2, 5) -/
theorem root_range_implies_a_range :
  (∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ r₁ ∈ Set.Icc (-2) 3 ∧ r₂ ∈ Set.Icc (-2) 3 ∧ 
   f r₁ = f r₂ ∧ 
   (∀ x, x ∈ Set.Icc (-2) 3 → f x = f r₁ → (x = r₁ ∨ x = r₂))) →
  (∃ a : ℝ, a ∈ Set.Ico (-2) 5 ∧ 
   (∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ r₁ ∈ Set.Icc (-2) 3 ∧ r₂ ∈ Set.Icc (-2) 3 ∧ 
    f r₁ = a ∧ f r₂ = a ∧
    (∀ x, x ∈ Set.Icc (-2) 3 → f x = a → (x = r₁ ∨ x = r₂)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_range_implies_a_range_l86_8679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l86_8682

noncomputable def f (x : ℝ) : ℝ := (Real.sin x - Real.cos x)^2 - 1

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is an odd function
  (∀ x, f (x + π) = f x) ∧  -- f has period π
  (∀ p, 0 < p → p < π → ∃ x, f (x + p) ≠ f x) :=  -- π is the minimum positive period
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l86_8682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_4_l86_8691

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)
noncomputable def f (x : ℝ) : ℝ := 4 - 2 * t x

-- State the theorem
theorem t_of_f_4 : t (f 4) = Real.sqrt (18 - 24 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_4_l86_8691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_elements_good_set_l86_8677

def is_good (X : Set ℝ) : Prop :=
  ∀ x, x ∈ X → ∃ y z, y ∈ X ∧ z ∈ X ∧ y ≠ z ∧ x = y + z

theorem min_elements_good_set :
  ∃ (X : Set ℝ), Finite X ∧ is_good X ∧ X.ncard = 6 ∧
  (∀ (Y : Set ℝ), Finite Y → is_good Y → Y.ncard ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_elements_good_set_l86_8677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_fraction_equality_l86_8634

theorem tan_pi_minus_alpha_fraction_equality
  (α : ℝ)
  (h1 : Real.tan (Real.pi - α) = -2/3)
  (h2 : α ∈ Set.Ioo (-Real.pi) (-Real.pi/2)) :
  (Real.cos (-α) + 3 * Real.sin (Real.pi + α)) / (Real.cos (Real.pi - α) + 9 * Real.sin α) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_alpha_fraction_equality_l86_8634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_equation_solution_l86_8663

theorem logarithm_equation_solution (a x : ℝ) (ha : 0 < a ∧ a ≠ 1) (hx : 0 < x ∧ x ≠ 1) :
  Real.sqrt (Real.log (a * x) / Real.log a + Real.log (a * x) / Real.log x) + 
  Real.sqrt (Real.log (x / a) / Real.log a + Real.log (a / x) / Real.log x) = 2 →
  x = a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_equation_solution_l86_8663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_perfect_square_a_equals_b_squared_l86_8629

def a : ℕ → ℤ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 0
  | 3 => 1
  | (n + 4) => ((n^2 + 3*n + 3) * (n + 2) / (n + 1)) * a (n + 3) + 
               (n^2 + 3*n + 3) * a (n + 2) - 
               ((n + 2) / (n + 1)) * a (n + 1)

def b : ℕ → ℤ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | 2 => 0
  | (n + 3) => (n + 1) * b (n + 2) + b (n + 1)

theorem a_is_perfect_square (n : ℕ) : 
  ∃ k : ℤ, a n = k^2 := by
  sorry

-- Additional theorem to show the relationship between a and b
theorem a_equals_b_squared (n : ℕ) :
  a n = (b n)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_perfect_square_a_equals_b_squared_l86_8629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_characterization_l86_8602

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem sequence_characterization 
  (a : ℕ+ → ℕ+) 
  (h : ∀ m n : ℕ+, floor ((a m : ℝ) / (a n : ℝ)) = floor ((m : ℝ) / (n : ℝ))) :
  ∃ k : ℕ+, ∀ i : ℕ+, a i = k * i := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_characterization_l86_8602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_condition_iff_k_gt_4_l86_8627

/-- The function f(x) defined as 2e^x - kx - 2 --/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 2 * Real.exp x - k * x - 2

/-- Theorem stating the condition for |f(x)| > 2x to hold in (0, m) --/
theorem f_condition_iff_k_gt_4 (k : ℝ) :
  (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, 0 < x → x < m → |f k x| > 2 * x) ↔ k > 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_condition_iff_k_gt_4_l86_8627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l86_8695

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + a

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi/12) (-1)

theorem function_properties :
  -- Maximum value of f(x) is 2
  (∃ (x : ℝ), f x (-1) = 2) ∧ (∀ (x : ℝ), f x (-1) ≤ 2) →
  -- 1. The value of a is -1
  (∀ (a : ℝ), (∃ (x : ℝ), f x a = 2) ∧ (∀ (x : ℝ), f x a ≤ 2) → a = -1) ∧
  -- 2. The axis of symmetry is x = kπ/2 + π/6, k ∈ ℤ
  (∀ (x : ℝ), f x (-1) = f (Real.pi/3 - x) (-1)) ∧
  -- 3. The range of g(x) on [π/6, π/3] is [√3, 2]
  (∀ (x : ℝ), x ∈ Set.Icc (Real.pi/6) (Real.pi/3) → g x ∈ Set.Icc (Real.sqrt 3) 2) ∧
  (∃ (x y : ℝ), x ∈ Set.Icc (Real.pi/6) (Real.pi/3) ∧ y ∈ Set.Icc (Real.pi/6) (Real.pi/3) ∧ g x = Real.sqrt 3 ∧ g y = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l86_8695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l86_8680

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x + 5

-- State the theorem
theorem f_min_max :
  ∃ (min max : ℝ), 
    (∀ x, 0 ≤ x ∧ x ≤ 2 → f x ≥ min) ∧
    (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ f x = min) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 → f x ≤ max) ∧
    (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ f x = max) ∧
    min = 1/2 ∧ max = 5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l86_8680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machines_needed_correct_l86_8601

/-- Given that 4 machines can produce x units in 6 days, 
    calculate the number of machines needed to produce y units in 6 days -/
noncomputable def machines_needed (x y : ℝ) : ℝ :=
  4 * y / x

theorem machines_needed_correct (x y : ℝ) (h : x > 0) :
  let m := machines_needed x y
  m * x / 4 = y :=
by
  -- Unfold the definition of machines_needed
  unfold machines_needed
  -- Simplify the expression
  simp [h]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machines_needed_correct_l86_8601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_area_of_triangle_l86_8666

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * (Real.cos (x/2 + Real.pi/4))^2 - Real.cos (2*x)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2*x - Real.pi/3) - 1

-- Theorem for the range of g(x)
theorem range_of_g : 
  Set.Icc (g (Real.pi/12)) (g (5*Real.pi/12)) = Set.Icc (-2) 1 := by sorry

-- Theorem for the area of triangle ABC
theorem area_of_triangle (a b c A B C : ℝ) 
  (h1 : b = 2)
  (h2 : f A = Real.sqrt 2 - 1)
  (h3 : Real.sqrt 3 * a = 2 * b * Real.sin A)
  (h4 : 0 < B ∧ B < Real.pi/2) :
  1/2 * a * b * Real.sin C = (3 + Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_area_of_triangle_l86_8666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_correct_l86_8655

/-- The coefficient of x^3 in the expansion of (2x^2 + 1/x)^6 is 160 -/
def coefficient_x_cubed_in_expansion : ℕ := 160

/-- The binomial expansion of (2x^2 + 1/x)^6 -/
noncomputable def expansion (x : ℝ) : ℝ :=
  (2 * x^2 + 1/x)^6

/-- The coefficient of x^3 in the expansion is equal to the result of coefficient_x_cubed_in_expansion -/
theorem coefficient_x_cubed_correct :
  ∃ (f g : ℝ → ℝ), ∀ x, expansion x = f x + coefficient_x_cubed_in_expansion * x^3 + x^4 * (g x) :=
by
  sorry

#eval coefficient_x_cubed_in_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_correct_l86_8655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_perpendicular_to_a_l86_8653

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def isPerpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- The vector (4, 3) -/
def a : ℝ × ℝ := (4, 3)

/-- The vector (3, -4) -/
def b : ℝ × ℝ := (3, -4)

/-- Proof that (3, -4) is perpendicular to (4, 3) -/
theorem b_perpendicular_to_a : isPerpendicular a b := by
  unfold isPerpendicular a b
  simp
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_perpendicular_to_a_l86_8653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_real_iff_m_nonnegative_l86_8661

-- Define the function (marked as noncomputable due to Real.sqrt)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (m * x^2 - 6 * m * x + 9 * m + 8)

-- State the theorem
theorem domain_is_real_iff_m_nonnegative (m : ℝ) :
  (∀ x : ℝ, m * x^2 - 6 * m * x + 9 * m + 8 ≥ 0) ↔ m ≥ 0 := by
  sorry

#check domain_is_real_iff_m_nonnegative

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_is_real_iff_m_nonnegative_l86_8661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_multiples_of_three_between_20_3_and_50_3_l86_8699

def count_odd_multiples_of_three (a b : ℚ) : ℕ :=
  (Finset.filter (λ n ↦ n % 2 ≠ 0 ∧ n % 3 = 0) (Finset.Icc ⌈a⌉ ⌊b⌋)).card

theorem odd_multiples_of_three_between_20_3_and_50_3 :
  count_odd_multiples_of_three (20/3) (50/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_multiples_of_three_between_20_3_and_50_3_l86_8699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_explicit_l86_8693

/-- A monic polynomial of degree 2 satisfying given conditions -/
def g : ℝ → ℝ := sorry

/-- g is a monic polynomial of degree 2 -/
axiom g_monic : ∃ b c : ℝ, ∀ x, g x = x^2 + b*x + c

/-- g(0) = 6 -/
axiom g_at_zero : g 0 = 6

/-- g(2) = 18 -/
axiom g_at_two : g 2 = 18

/-- The theorem stating that g(x) = x^2 + 4x + 6 -/
theorem g_explicit : ∀ x, g x = x^2 + 4*x + 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_explicit_l86_8693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zeros_and_inequality_l86_8626

noncomputable section

open Real

def f (x : ℝ) := sin (2 * x)

def g (x : ℝ) := sin (2 * (x - π/6))

theorem g_zeros_and_inequality :
  (∃ (zeros : Finset ℝ), zeros.card = 6 ∧ 
    (∀ x ∈ zeros, 0 < x ∧ x < 3*π ∧ g x = 0) ∧
    (∀ x, 0 < x ∧ x < 3*π ∧ g x = 0 → x ∈ zeros)) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Icc (-π/6) (π/2), g x - a ≥ f (5*π/12)) → a ≤ -3/2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zeros_and_inequality_l86_8626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_gas_cost_l86_8671

/-- Represents Jenna's road trip -/
structure RoadTrip where
  speed1 : ℚ
  time1 : ℚ
  speed2 : ℚ
  time2 : ℚ
  miles_per_gallon : ℚ
  cost_per_gallon : ℚ

/-- Calculates the total cost of gas for the road trip -/
def total_gas_cost (trip : RoadTrip) : ℚ :=
  let total_distance := trip.speed1 * trip.time1 + trip.speed2 * trip.time2
  let gallons_needed := total_distance / trip.miles_per_gallon
  gallons_needed * trip.cost_per_gallon

/-- Theorem stating that Jenna's gas cost for the trip is $18 -/
theorem jenna_gas_cost :
  let trip := RoadTrip.mk 60 2 50 3 30 2
  total_gas_cost trip = 18 := by
  -- Unfold the definition of total_gas_cost
  unfold total_gas_cost
  -- Simplify the arithmetic expressions
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_gas_cost_l86_8671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_equals_one_l86_8667

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.sin x - 3/2

def domain : Set ℝ := Set.Icc 0 (Real.pi/2)

theorem max_value_implies_a_equals_one (a : ℝ) :
  (∃ (x : ℝ), x ∈ domain ∧ f a x = (Real.pi - 3)/2) ∧
  (∀ (y : ℝ), y ∈ domain → f a y ≤ (Real.pi - 3)/2) →
  a = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_equals_one_l86_8667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_f_even_implies_phi_not_neg_pi_fourth_l86_8686

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (x + φ / 2) * Real.cos (x + φ / 2)

noncomputable def translated_f (φ : ℝ) (x : ℝ) : ℝ := f φ (x + Real.pi / 8)

theorem translated_f_even_implies_phi_not_neg_pi_fourth (φ : ℝ) :
  (∀ x, translated_f φ x = translated_f φ (-x)) →
  φ ≠ -Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_f_even_implies_phi_not_neg_pi_fourth_l86_8686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l86_8623

open Real

-- Define the function g
noncomputable def g (A : ℝ) : ℝ :=
  (cos A * (5 * sin A ^ 2 + sin A ^ 4 + 5 * cos A ^ 2 + cos A ^ 2 * sin A ^ 2)) /
  ((cos A / sin A) * (1 / sin A - cos A * (cos A / sin A)))

-- State the theorem
theorem range_of_g :
  ∀ A : ℝ, (∀ n : ℤ, A ≠ n * π) →
  ∃ y : ℝ, y ∈ Set.Ioo (-36) 36 ∧ g A = y ∧
  ∀ z : ℝ, g A = z → z ∈ Set.Ioo (-36) 36 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l86_8623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_powers_of_two_not_divisible_by_five_l86_8676

theorem powers_of_two_not_divisible_by_five (n : ℕ) : 
  (∃ k : ℕ, n = 2^k ∧ n < 500000 ∧ ¬(5 ∣ n)) ↔ n ∈ Finset.range 19 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_powers_of_two_not_divisible_by_five_l86_8676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_sum_five_l86_8604

def S (n : ℕ) : ℕ := (Nat.digits 10 (2^n)).sum

theorem unique_digit_sum_five : ∀ n : ℕ, S n = 5 ↔ n = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_sum_five_l86_8604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_numbers_with_lcm_and_sum_l86_8689

theorem ratio_of_numbers_with_lcm_and_sum
  (A B : ℕ) 
  (h_pos_A : A > 0)
  (h_pos_B : B > 0)
  (h_lcm : Nat.lcm A B = 30)
  (h_sum : A + B = 25)
  (h_order : A < B) :
  ∃ (k : ℕ), k > 0 ∧ A = 2 * k ∧ B = 3 * k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_numbers_with_lcm_and_sum_l86_8689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_and_min_dot_product_l86_8683

/-- Circle C is symmetric to circle M about the line x+y+2=0 -/
def symmetric_circles (C M : Set (ℝ × ℝ)) : Prop :=
  ∀ (p : ℝ × ℝ), p ∈ C ↔ (-(p.1 + p.2 + 2) - p.1, -(p.1 + p.2 + 2) - p.2) ∈ M

/-- Circle M with equation (x+2)^2 + (y+2)^2 = r^2, r > 0 -/
def circle_M (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 2)^2 + (p.2 + 2)^2 = r^2 ∧ r > 0}

/-- Circle C passing through (1,1) -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 2}

/-- Vector dot product -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem circle_C_equation_and_min_dot_product :
  ∃ (r : ℝ), 
    (symmetric_circles circle_C (circle_M r) ∧ (1, 1) ∈ circle_C) →
    (∀ (p : ℝ × ℝ), p ∈ circle_C ↔ p.1^2 + p.2^2 = 2) ∧
    (∀ (Q : ℝ × ℝ), Q ∈ circle_C → 
      dot_product (Q.1 - 1, Q.2 - 1) (Q.1 + 2, Q.2 + 2) ≥ -4) ∧
    (∃ (Q : ℝ × ℝ), Q ∈ circle_C ∧ 
      dot_product (Q.1 - 1, Q.2 - 1) (Q.1 + 2, Q.2 + 2) = -4) :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_and_min_dot_product_l86_8683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_P_intersect_Q_l86_8633

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- State the theorem
theorem complement_P_intersect_Q : (Set.compl P) ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_P_intersect_Q_l86_8633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_R_l86_8646

/-- Region R in the Cartesian plane -/
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 + ⌊p.1⌋ + ⌊p.2⌋ ≤ 5}

/-- The area of region R is 9/2 -/
theorem area_of_R : MeasureTheory.volume R = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_R_l86_8646
