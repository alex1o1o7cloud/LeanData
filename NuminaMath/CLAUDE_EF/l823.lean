import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l823_82306

-- Define the parabola equation as noncomputable
noncomputable def parabola_equation (x : ℝ) : ℝ := -1/3 * x^2 + 2*x + 2

-- State the theorem
theorem parabola_properties :
  -- The vertex is (3,5)
  (parabola_equation 3 = 5) ∧
  -- The axis of symmetry is vertical (x = 3)
  (∀ (h : ℝ), parabola_equation (3 - h) = parabola_equation (3 + h)) ∧
  -- The point (0,2) is on the parabola
  (parabola_equation 0 = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l823_82306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_is_four_l823_82313

-- Define the function f
noncomputable def f (x : ℝ) := 1 + (2^(x+1))/(2^x+1) + Real.sin x

-- Define the theorem
theorem sum_of_max_min_is_four (k : ℝ) (h : k > 0) :
  ∃ (m n : ℝ), (∀ x ∈ Set.Icc (-k) k, m ≤ f x ∧ f x ≤ n) ∧ m + n = 4 := by
  sorry

#check sum_of_max_min_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_is_four_l823_82313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_of_nine_seven_games_probability_l823_82359

/-- The probability of a best-of-9 series lasting exactly 7 games -/
theorem best_of_nine_seven_games_probability (p : ℝ) (hp : p = 2/3) :
  let q := 1 - p
  (Nat.choose 6 2 : ℝ) * p^4 * q^2 * p + (Nat.choose 6 2 : ℝ) * q^4 * p^2 * q = 20/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_of_nine_seven_games_probability_l823_82359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_of_glued_cubes_l823_82384

/-- Given two cubes where the larger cube's edge length is 6 times that of the smaller cube,
    and they are glued together on one face, the ratio of the exposed red surface area
    (from the larger cube) to the exposed yellow surface area (from the smaller cube) is 43. -/
theorem surface_area_ratio_of_glued_cubes :
  ∀ (a : ℝ), a > 0 →
  (let small_cube_edge := a
   let large_cube_edge := 6 * a
   let small_cube_surface_area := 6 * a^2
   let large_cube_surface_area := 6 * (6*a)^2
   let exposed_small_cube_area := small_cube_surface_area - a^2
   let exposed_large_cube_area := large_cube_surface_area - a^2
   (exposed_large_cube_area / exposed_small_cube_area : ℝ) = 43) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_of_glued_cubes_l823_82384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_super_double_probability_l823_82339

/-- A super domino is a pair of integers from 0 to 12 -/
def SuperDomino := Fin 13 × Fin 13

/-- A super double is a super domino with the same integer on both squares -/
def SuperDouble (d : SuperDomino) : Prop := d.1 = d.2

/-- The set of all possible super dominoes -/
def AllSuperDominoes : Finset SuperDomino :=
  Finset.product (Finset.univ : Finset (Fin 13)) (Finset.univ : Finset (Fin 13))

/-- The set of all super doubles -/
def AllSuperDoubles : Finset SuperDomino :=
  Finset.filter (fun d => d.1 = d.2) AllSuperDominoes

theorem super_double_probability :
  (Finset.card AllSuperDoubles : ℚ) / (Finset.card AllSuperDominoes : ℚ) = 13 / 169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_super_double_probability_l823_82339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_calculation_l823_82383

theorem absolute_value_calculation : |(-2 : ℝ)| - (-Real.sqrt 2)^(0 : ℝ) + (1/3 : ℝ)^(-1 : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_calculation_l823_82383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_theorem_l823_82335

-- Define the circle
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 - 2*a*x + 2*y - 1 = 0

-- Define the point P
def point_P (a : ℝ) : ℝ × ℝ := (-5, a)

-- Define the tangent condition
def tangent_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) + (x₁ + x₂ - 2) / (y₁ + y₂) = 0

-- State the theorem
theorem circle_tangent_theorem (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_eq (-5) a a ∧
    circle_eq x₁ y₁ a ∧
    circle_eq x₂ y₂ a ∧
    tangent_condition x₁ y₁ x₂ y₂) →
  a = 3 ∨ a = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_theorem_l823_82335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l823_82368

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x + Real.pi / 3)

theorem f_properties (ω : ℝ) (h_ω_pos : ω > 0) (h_period : ∀ x, f ω (x + Real.pi / ω) = f ω x) :
  (ω = 1) ∧
  (∀ m : ℝ, (∃ x y : ℝ, x ≠ y ∧ x ∈ Set.Icc 0 (Real.pi / 2) ∧ y ∈ Set.Icc 0 (Real.pi / 2) ∧ 
                        f ω x = m ∧ f ω y = m) ↔ 
             m ∈ Set.Icc 0 (1 - Real.sqrt 3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l823_82368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_products_l823_82376

-- Define the type for permutations of {1, 2, ..., 6}
def Perm6 : Type := Fin 6 → Fin 6

-- Define the property that a function is a permutation
def isPerm (f : Perm6) : Prop :=
  Function.Bijective f

-- Define the sum of products
def sumProducts (a b c : Perm6) : ℕ :=
  Finset.sum Finset.univ (fun i => (a i).val.succ * (b i).val.succ * (c i).val.succ)

-- State the theorem
theorem min_sum_products (a b c : Perm6) (ha : isPerm a) (hb : isPerm b) (hc : isPerm c) :
  sumProducts a b c ≥ 162 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_products_l823_82376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_0_466_to_hundredth_l823_82321

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- Theorem stating that rounding 0.466 to the nearest hundredth results in 0.47 -/
theorem round_0_466_to_hundredth :
  round_to_hundredth 0.466 = 0.47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_0_466_to_hundredth_l823_82321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_fractional_part_l823_82318

theorem irrational_fractional_part (x : ℝ) (h_irr : Irrational x) (h_pos : x > 0) :
  ∀ m : ℕ, ∃ n > m, ∀ k : ℕ, 1 ≤ k → k ≤ n → (k * x - ⌊k * x⌋) ≥ 1 / (n + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_fractional_part_l823_82318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_value_l823_82356

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def y (x u : ℝ) : ℝ := 4 * ((x + floor u) / 4 - floor ((x + floor u) / 4))

theorem u_value (x : ℤ) :
  (∀ x ∈ ({1, 8, 11, 14} : Set ℤ), y x ((x - 1) / 4) = 1) ∧
  (∀ x ∈ ({2, 5, 12, 15} : Set ℤ), y x ((x - 1) / 4) = 2) ∧
  (∀ x ∈ ({3, 6, 9, 16} : Set ℤ), y x ((x - 1) / 4) = 3) ∧
  (∀ x ∈ ({4, 7, 10, 13} : Set ℤ), y x ((x - 1) / 4) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_value_l823_82356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_coefficient_l823_82366

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the point on the curve
def point : ℝ × ℝ := (2, 8)

-- Define the slope of the tangent line
def tangent_slope : ℝ := 12

-- Define the equation of the tangent line
def tangent_line (a : ℝ) (x y : ℝ) : Prop :=
  12 * x - a * y - 16 = 0

-- Theorem statement
theorem tangent_line_coefficient (a : ℝ) :
  curve point.fst = point.snd →
  (∀ x y : ℝ, tangent_line a x y ↔ y - point.snd = tangent_slope * (x - point.fst)) →
  a = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_coefficient_l823_82366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_154_l823_82382

theorem greatest_prime_factor_of_154 : 
  (Nat.factors 154).maximum? = some 11 := by
  -- The proof goes here
  sorry

#eval Nat.factors 154  -- This will show us the prime factors of 154

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_154_l823_82382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_representation_of_21_over_22_l823_82302

noncomputable def decimal_digit_at (q : ℚ) (n : ℕ) : ℕ := 
  (Int.floor (q * 10^n) - 10 * Int.floor (q * 10^(n-1))).toNat % 10

theorem decimal_representation_of_21_over_22 :
  ∀ (n : ℕ), (21 : ℚ) / 22 = (95 : ℚ) / 100 + (45 : ℚ) / (100 * 100^n) → 
  ∀ (i : ℕ), i > 1 → 
    (decimal_digit_at ((21 : ℚ) / 22) i = 4 → decimal_digit_at ((21 : ℚ) / 22) (i + 1) = 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_representation_of_21_over_22_l823_82302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_problem_l823_82389

/-- The amount of alloy B mixed with alloy A -/
def alloy_B_amount : ℝ → Prop := λ x => x = 140

/-- The ratio of lead to tin in alloy A -/
def alloy_A_ratio : ℚ := 3 / 4

/-- The ratio of tin to copper in alloy B -/
def alloy_B_ratio : ℚ := 2 / 5

/-- The amount of alloy A in kg -/
def alloy_A_amount : ℝ := 90

/-- The amount of tin in the new alloy in kg -/
def tin_in_new_alloy : ℝ := 91.42857142857143

theorem alloy_mixture_problem :
  alloy_B_amount 140 :=
by
  unfold alloy_B_amount
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_problem_l823_82389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2023_of_7_div_22_l823_82303

/-- The decimal expansion of 7/22 -/
def decimal_expansion : ℚ := 7/22

/-- The repeating pattern in the decimal expansion -/
def repeating_pattern : List ℕ := [3, 1, 8, 1, 8, 1]

/-- The length of the repeating pattern -/
def pattern_length : ℕ := 6

/-- The position we're interested in -/
def target_position : ℕ := 2023

/-- The theorem stating that the 2023rd digit past the decimal point in 7/22 is 8 -/
theorem digit_2023_of_7_div_22 : 
  (repeating_pattern.get! ((target_position - 1) % pattern_length)) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2023_of_7_div_22_l823_82303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_for_intersection_points_l823_82330

/-- A type representing a line in a plane -/
structure Line where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A type representing a point in a plane -/
structure Point where
  -- Add necessary fields here
  mk :: -- Constructor

/-- The set of intersection points of the lines -/
def intersectionPoints (lines : List Line) : Set Point :=
  sorry

/-- Predicate to check if two points are adjacent on a line -/
def areAdjacent (p1 p2 : Point) (line : Line) : Prop :=
  sorry

/-- A coloring of points -/
def Coloring := Point → Nat

/-- Predicate to check if a coloring is valid -/
def isValidColoring (c : Coloring) (lines : List Line) : Prop :=
  ∀ line, line ∈ lines → ∀ p1 p2, p1 ∈ intersectionPoints lines → p2 ∈ intersectionPoints lines →
    areAdjacent p1 p2 line → c p1 ≠ c p2

/-- Predicate to check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Predicate to check if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

/-- The main theorem -/
theorem min_colors_for_intersection_points :
  ∀ (lines : List Line),
    lines.length = 2015 →
    (∀ l1 l2, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → ¬ are_parallel l1 l2) →
    (∀ l1 l2 l3, l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 → ¬ are_concurrent l1 l2 l3) →
    (∃ (c : Coloring), isValidColoring c lines ∧ (∀ p, c p < 3)) ∧
    (∀ (c : Coloring), isValidColoring c lines → ∃ p, c p ≥ 2) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_for_intersection_points_l823_82330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_term_prime_l823_82323

def sequenceNum (n : ℕ) : ℕ := 
  (17 : ℕ) * ((10^(2*n) - 1) / 99)

def is_prime_in_sequence (n : ℕ) : Prop :=
  Nat.Prime (sequenceNum n)

theorem only_first_term_prime : 
  (∃! k : ℕ, k < 10 ∧ is_prime_in_sequence k) ∧ 
  (∀ k : ℕ, k < 10 ∧ is_prime_in_sequence k → k = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_term_prime_l823_82323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_saturday_rate_l823_82355

/-- Represents Jason's work schedule and earnings --/
structure JasonWork where
  afterSchoolRate : ℚ
  totalHours : ℚ
  totalEarnings : ℚ
  saturdayHours : ℚ

/-- Calculates Jason's Saturday hourly rate --/
def saturdayRate (w : JasonWork) : ℚ :=
  (w.totalEarnings - w.afterSchoolRate * (w.totalHours - w.saturdayHours)) / w.saturdayHours

/-- Theorem: Jason's Saturday hourly rate is $6.00 --/
theorem jason_saturday_rate :
  let w : JasonWork := {
    afterSchoolRate := 4,
    totalHours := 18,
    totalEarnings := 88,
    saturdayHours := 8
  }
  saturdayRate w = 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jason_saturday_rate_l823_82355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_bookshop_inventory_l823_82345

theorem johns_bookshop_inventory (monday tuesday wednesday thursday friday : ℕ)
  (h1 : monday = 62)
  (h2 : tuesday = 62)
  (h3 : wednesday = 60)
  (h4 : thursday = 48)
  (h5 : friday = 40)
  (h6 : (66 : ℚ) / 100 * (monday + tuesday + wednesday + thursday + friday : ℚ) = 
        ((monday + tuesday + wednesday + thursday + friday : ℚ) * (34 : ℚ) / 100)) :
  (monday + tuesday + wednesday + thursday + friday : ℚ) / ((34 : ℚ) / 100) = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_bookshop_inventory_l823_82345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l823_82342

/-- The eccentricity of a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 is sqrt(1 + b^2/a^2) -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- The hyperbola equation x^2/16 - y^2/9 = 1 has eccentricity 5/4 -/
theorem hyperbola_eccentricity : eccentricity 4 3 = 5/4 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l823_82342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_negative_power_inequality_l823_82353

/-- Power function -/
noncomputable def power_function (k : ℚ) (x : ℝ) : ℝ := x ^ (k : ℝ)

/-- Power function is increasing -/
theorem power_function_increasing (k : ℚ) (hk : k > 0) :
  StrictMono (power_function k) := by sorry

/-- Inequality for negative powers -/
theorem negative_power_inequality {a b c : ℝ} (hab : a > b) (hb : b > 0) (hc : c > 0) :
  a^(-c) < b^(-c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_negative_power_inequality_l823_82353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_return_calculation_l823_82361

/-- The Fisher equation for calculating real rate of return -/
noncomputable def fisher_equation (n : ℝ) (i : ℝ) : ℝ := (1 + n) / (1 + i) - 1

/-- Pavel's nominal rate of return -/
def nominal_rate : ℝ := 0.21

/-- The inflation rate -/
def inflation_rate : ℝ := 0.10

/-- The real rate of return -/
noncomputable def real_rate : ℝ := fisher_equation nominal_rate inflation_rate

theorem real_return_calculation :
  real_rate = 0.10 := by
  -- Unfold definitions
  unfold real_rate
  unfold fisher_equation
  unfold nominal_rate
  unfold inflation_rate
  -- Simplify the expression
  simp
  -- The proof itself is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_return_calculation_l823_82361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l823_82371

/-- Triangle ABC with vertices A, B, and C, where C is on the line x + y = 10 -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A = (6, 0)
  h_B : B = (0, 6)
  h_C : C.1 + C.2 = 10

/-- The area of triangle ABC is 12 -/
theorem triangle_area (t : Triangle) : abs ((t.A.1 - t.C.1) * (t.B.2 - t.C.2) - (t.B.1 - t.C.1) * (t.A.2 - t.C.2)) / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l823_82371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetric_axis_specific_parabola_symmetric_axis_l823_82352

/-- The symmetric axis of a parabola y = a(x-h)^2 + k is the line x = h -/
theorem parabola_symmetric_axis (a h k : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * (x - h)^2 + k
  ∃! l : ℝ, ∀ x y : ℝ, f x = f y → (x + y) / 2 = l := by
  sorry

/-- The symmetric axis of the parabola y = 2(x-3)^2 + 5 is the line x = 3 -/
theorem specific_parabola_symmetric_axis :
  let f : ℝ → ℝ := λ x => 2 * (x - 3)^2 + 5
  ∃! l : ℝ, (∀ x y : ℝ, f x = f y → (x + y) / 2 = l) ∧ l = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_symmetric_axis_specific_parabola_symmetric_axis_l823_82352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_has_repeated_digits_l823_82346

def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def remove_first_digit_and_add (n : Nat) : Nat :=
  let first_digit := n / (10^(Nat.log n 10))
  let remaining := n % (10^(Nat.log n 10))
  remaining + first_digit

def iterate_operation (n : Nat) : Nat :=
  if n < 10^10 then n else iterate_operation (remove_first_digit_and_add n)
termination_by iterate_operation n => n

theorem final_number_has_repeated_digits : 
  ∃ (i j : Fin 10), i ≠ j ∧ 
    (iterate_operation (2^1970)).toString.get ⟨i.val, by sorry⟩ = 
    (iterate_operation (2^1970)).toString.get ⟨j.val, by sorry⟩ := by
  sorry

#eval iterate_operation (2^1970)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_has_repeated_digits_l823_82346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_first_3002_even_integers_l823_82322

def count_digits (n : ℕ) : ℕ := 
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

def sum_digits_even (n : ℕ) : ℕ :=
  (List.range n).filter (λ x => x % 2 = 0 ∧ x > 0)
    |>.map count_digits
    |>.sum

theorem sum_digits_first_3002_even_integers :
  sum_digits_even 6005 = 11456 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_digits_first_3002_even_integers_l823_82322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_values_l823_82351

theorem integer_fraction_values (a : ℤ) : 
  (∃ k : ℤ, (a + 9 : ℚ) / (a + 6 : ℚ) = k) ↔ a ∈ ({-5, -7, -3, -9} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_values_l823_82351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l823_82365

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.log (x + Real.sqrt (4 + x^2))

-- Theorem stating that g is neither even nor odd
theorem g_neither_even_nor_odd :
  (∃ x : ℝ, g (-x) ≠ g x) ∧ (∃ x : ℝ, g (-x) ≠ -g x) :=
by
  -- We'll use x = 1 as our example
  let x : ℝ := 1
  
  have h1 : g (-x) ≠ g x := by
    -- Proof that g is not even
    sorry
  
  have h2 : g (-x) ≠ -g x := by
    -- Proof that g is not odd
    sorry
  
  -- Combine the two results
  exact ⟨⟨x, h1⟩, ⟨x, h2⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l823_82365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_2_sqrt_41_l823_82320

-- Define the points
def start : ℝ × ℝ := (-3, 6)
def stop : ℝ × ℝ := (1, 1)
def finish : ℝ × ℝ := (6, -3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem total_distance_is_2_sqrt_41 :
  distance start stop + distance stop finish = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_2_sqrt_41_l823_82320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_uniqueness_l823_82379

-- Define f as a quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 12

-- Define g in terms of f
noncomputable def g (x : ℝ) : ℝ := (2:ℝ)^x * f x

-- Theorem statement
theorem quadratic_functions_uniqueness :
  (f 0 = 12) ∧
  (∀ x : ℝ, g (x + 1) - g x ≥ (2:ℝ)^(x + 1) * x^2) ∧
  (∀ x : ℝ, f x = 2 * x^2 - 8 * x + 12) ∧
  (∀ x : ℝ, g x = (2 * x^2 - 8 * x + 12) * (2:ℝ)^x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_uniqueness_l823_82379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_armans_hourly_rate_l823_82333

/-- Arman's hourly rate last week -/
def last_week_rate : ℝ := sorry

/-- Arman's hours worked last week -/
def last_week_hours : ℝ := 35

/-- Arman's hours worked this week -/
def this_week_hours : ℝ := 40

/-- Hourly rate increase this week -/
def rate_increase : ℝ := 0.5

/-- Total amount earned over two weeks -/
def total_earned : ℝ := 770

theorem armans_hourly_rate :
  last_week_rate * last_week_hours + 
  (last_week_rate + rate_increase) * this_week_hours = total_earned →
  last_week_rate = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_armans_hourly_rate_l823_82333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l823_82336

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates a quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Calculates the vertex of a quadratic function -/
noncomputable def QuadraticFunction.vertex (f : QuadraticFunction) : ℝ × ℝ :=
  let x := -f.b / (2 * f.a)
  (x, f.evaluate x)

theorem parabola_properties (f : QuadraticFunction) : 
  f.a = -3 ∧ f.b = 18 ∧ f.c = -22 →
  f.vertex = (3, 5) ∧
  f.evaluate 2 = 2 := by
  sorry

#check parabola_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l823_82336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l823_82338

-- Define the function f
noncomputable def f (x α : ℝ) : ℝ := Real.sin (x - α) + 2 * Real.cos x

-- Define the maximum value function h
noncomputable def h (α : ℝ) : ℝ := Real.sqrt (5 - 4 * Real.sin α)

theorem f_properties :
  -- The maximum value of f(x) is h(α), and the maximum value of h(α) is 3
  (∀ α x, f x α ≤ h α) ∧ (∃ α, h α = 3) ∧
  -- There exists an α such that f(x) is an even function
  (∃ α, ∀ x, f x α = f (-x) α) ∧
  -- When α = π/6, (-π/3, 0) is a symmetry center of f(x)
  (∀ x, f (x + (-Real.pi/3)) (Real.pi/6) = -f (-x + (-Real.pi/3)) (Real.pi/6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l823_82338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solids_l823_82311

-- Define the surfaces for each solid
def surface1 (x y z : ℝ) : Prop :=
  x + y + z = 4 ∨ x = 3 ∨ y = 2 ∨ x = 0 ∨ y = 0 ∨ z = 0

def surface2 (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 = 2*z ∧ x^2 + y^2 = z^2

def surface3 (x y z : ℝ) : Prop :=
  2*z = x^2 + y^2 ∧ y + z = 4

-- Define the volumes of the solids
noncomputable def volume1 : ℝ := 17/6
noncomputable def volume2 : ℝ := Real.pi/3
noncomputable def volume3 : ℝ := 81*Real.pi/4

-- Hypothetical function for volume calculation
noncomputable def volume_of_solid (surface : ℝ → ℝ → ℝ → Prop) : ℝ := sorry

-- State the theorem
theorem volume_of_solids :
  (∃ V1, volume_of_solid surface1 = V1 ∧ V1 = volume1) ∧
  (∃ V2, volume_of_solid surface2 = V2 ∧ V2 = volume2) ∧
  (∃ V3, volume_of_solid surface3 = V3 ∧ V3 = volume3) := by
  sorry

-- Note: volume_of_solid is a hypothetical function that calculates the volume of a solid
-- bounded by a given surface. Its implementation is not provided in this statement.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_solids_l823_82311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trigonometric_expression_min_value_achieved_l823_82369

open Real

theorem min_value_trigonometric_expression (θ : ℝ) (h : 0 < θ ∧ θ < π/2) :
  3 * sin θ + 2 / cos θ + 2 * Real.sqrt 3 * (cos θ / sin θ) ≥ 6 * Real.sqrt 3 :=
sorry

theorem min_value_achieved (θ : ℝ) (h : 0 < θ ∧ θ < π/2) :
  ∃ θ₀, 0 < θ₀ ∧ θ₀ < π/2 ∧
  3 * sin θ₀ + 2 / cos θ₀ + 2 * Real.sqrt 3 * (cos θ₀ / sin θ₀) = 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trigonometric_expression_min_value_achieved_l823_82369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_project_initial_days_l823_82324

/-- Represents a road construction project -/
structure RoadProject where
  totalLength : ℚ
  initialMen : ℚ
  daysElapsed : ℚ
  completedLength : ℚ
  extraMen : ℚ

/-- Calculates the total days initially planned for a road project -/
noncomputable def calculateInitialDays (project : RoadProject) : ℚ :=
  let totalMen := project.initialMen + project.extraMen
  let remainingLength := project.totalLength - project.completedLength
  project.daysElapsed + (remainingLength * totalMen * project.daysElapsed) / (project.completedLength * project.initialMen)

/-- Theorem stating that for the given road project, the initially planned days is 300 -/
theorem road_project_initial_days :
  let project : RoadProject := {
    totalLength := 15
    initialMen := 35
    daysElapsed := 100
    completedLength := 5/2
    extraMen := 105/2
  }
  calculateInitialDays project = 300 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_project_initial_days_l823_82324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_is_two_l823_82331

/-- A regular k-gon inscribed in a unit circle -/
structure RegularKGon (k : ℕ) where
  vertices : Fin k → ℂ
  is_regular : ∀ i : Fin k, Complex.abs (vertices i) = 1
  is_kth_roots : ∀ i : Fin k, (vertices i) ^ k = 1

/-- The product of distances from a point to all vertices of a k-gon -/
noncomputable def distance_product (k : ℕ) (kg : RegularKGon k) (p : ℂ) : ℝ :=
  Finset.prod (Finset.univ : Finset (Fin k)) (fun i => Complex.abs (p - kg.vertices i))

/-- The maximum value of the distance product is 2 -/
theorem max_distance_product_is_two (k : ℕ) (k_pos : k > 0) :
  ∀ kg : RegularKGon k, ∀ p : ℂ, Complex.abs p ≤ 1 →
  distance_product k kg p ≤ 2 ∧ 
  ∃ p₀ : ℂ, Complex.abs p₀ ≤ 1 ∧ distance_product k kg p₀ = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_product_is_two_l823_82331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_swim_distance_l823_82390

/-- The intended number of lengths Rebecca wants to swim -/
def D : ℕ := 120

/-- The number of additional lengths Rebecca swims -/
def additional_lengths : ℕ := 6

/-- The fraction of the intended distance Rebecca swims initially -/
def initial_fraction : ℚ := 1 / 5

/-- The fraction of the intended distance Rebecca swims after additional lengths -/
def final_fraction : ℚ := 1 / 4

theorem rebecca_swim_distance :
  (final_fraction - initial_fraction) * D = additional_lengths ∧ D = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebecca_swim_distance_l823_82390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_third_l823_82387

theorem sin_double_angle_plus_pi_third (α : ℝ) :
  0 < α → α < π / 2 →
  Real.cos (α + π / 6) = 4 / 5 →
  Real.sin (2 * α + π / 3) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_third_l823_82387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l823_82392

/-- The equation of a parabola -/
noncomputable def parabola_equation (x : ℝ) : ℝ := (x^2 - 4*x + 7) / 8

/-- The equation of the directrix -/
noncomputable def directrix_equation : ℝ := -19/8

/-- Theorem stating that the given equation is the directrix of the parabola -/
theorem parabola_directrix : 
  ∀ x y : ℝ, y = parabola_equation x → 
  ∃ p : ℝ × ℝ, 
    (p.1 - x)^2 + (p.2 - y)^2 = (y - directrix_equation)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l823_82392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_class_probability_l823_82341

-- Define the set of classes
inductive ClassType
| A
| B
| C

-- Define a function to represent the assignment of a student to a class
def Assignment := ClassType

-- Define the probability space
def Ω := Assignment × Assignment

-- Define the probability measure (using noncomputable)
noncomputable def P : Set Ω → ℝ := sorry

-- State the theorem
theorem same_class_probability :
  let same_class := {ω : Ω | ω.1 = ω.2}
  P same_class = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_class_probability_l823_82341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_pinedale_mall_l823_82307

/-- The distance to Pinedale mall -/
noncomputable def distance_to_mall (avg_speed : ℝ) (time_between_stops : ℝ) (num_stops : ℕ) : ℝ :=
  avg_speed * (time_between_stops * (num_stops : ℝ) / 60)

/-- Theorem: The distance to Pinedale mall is 40 km -/
theorem distance_to_pinedale_mall :
  distance_to_mall 60 5 8 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_pinedale_mall_l823_82307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l823_82325

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The asymptote equation -/
def asymptote (x y : ℝ) : Prop := 3 * x - 4 * y = 0

/-- The distance formula from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

theorem distance_to_asymptote :
  distance_point_to_line 3 0 3 (-4) 0 = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l823_82325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_expenses_representation_l823_82310

/-- Represents a monetary value in yuan -/
structure Yuan where
  value : Int

/-- Represents income in yuan -/
def income (amount : Yuan) : Yuan := amount

/-- Represents expenses in yuan -/
def expenses (amount : Yuan) : Yuan := ⟨-amount.value⟩

/-- The correct representation of income and expenses -/
theorem income_expenses_representation (amount : Yuan) :
  income amount = amount ∧ expenses amount = ⟨-amount.value⟩ := by
  constructor
  · rfl
  · rfl

/-- Demonstration of the theorem -/
example : expenses ⟨5⟩ = ⟨-5⟩ := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_expenses_representation_l823_82310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_lower_bound_l823_82358

theorem distinct_sums_lower_bound (n k : ℕ) (S : Finset ℝ) 
  (h1 : 0 < k) (h2 : k ≤ n) (h3 : Finset.card S = n) :
  let T := (S.powersetCard k).image (fun s ↦ s.sum id)
  Finset.card T ≥ k * (n - k) + 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_lower_bound_l823_82358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_cocaptains_selected_l823_82370

/-- Represents a math team with its size and number of co-captains -/
structure MathTeam where
  size : ℕ
  cocaptains : ℕ

/-- Calculates the probability of selecting two co-captains from a given team -/
def probBothCocaptains (team : MathTeam) : ℚ :=
  (team.cocaptains.choose 2 : ℚ) / team.size.choose 2

/-- The set of math teams in the area -/
def mathTeams : List MathTeam := [
  { size := 6, cocaptains := 2 },
  { size := 9, cocaptains := 2 },
  { size := 10, cocaptains := 3 }
]

theorem prob_both_cocaptains_selected :
  (mathTeams.map probBothCocaptains).sum / mathTeams.length = 29 / 540 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_cocaptains_selected_l823_82370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l823_82374

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 1) :
  ∃ m : ℝ, m = (1 / (a * (1 - a))) ∧ 
  ∀ x : ℝ, x > 2 → (3*a*x/b) + (x/(a*b)) + (6/(x-2)) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l823_82374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_l823_82347

noncomputable def f (x m : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) - Real.cos x ^ 2 - m

theorem max_value_implies_m (m : ℝ) : 
  (∀ x ∈ Set.Icc (5 * Real.pi / 24) (3 * Real.pi / 4), f x m ≤ 0) ∧
  (∃ x ∈ Set.Icc (5 * Real.pi / 24) (3 * Real.pi / 4), f x m = 0) →
  m = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_m_l823_82347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_distance_relation_l823_82344

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem collinear_points_distance_relation (x y : ℝ) :
  collinear x 5 (-2) y 1 1 →
  distance (-2) y 1 1 = 2 * distance x 5 1 1 →
  x + y = -9/2 ∨ x + y = 17/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_distance_relation_l823_82344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l823_82349

open Set
open Real

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * sin (ω * x + 5 * π / 6)

-- Define the domain D
def D : Set ℝ := Icc 0 π

-- State the theorem
theorem omega_range (ω : ℝ) :
  ω > 0 ∧ 
  (∀ x ∈ D, f ω x ∈ Icc 0 2) ∧
  (f ω '' (f ω ⁻¹' (Icc 0 2) ∩ D) = Icc 0 2) →
  ω ≥ 5/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l823_82349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l823_82375

-- Define the length of the train in meters
noncomputable def train_length : ℝ := 400

-- Define the speed of the train in km/h
noncomputable def train_speed_kmh : ℝ := 144

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Calculate the speed of the train in m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmh * kmh_to_ms

-- Theorem: The time taken for the train to cross the man is 10 seconds
theorem train_crossing_time : 
  train_length / train_speed_ms = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l823_82375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_a_range_l823_82328

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≥ 0 then a * x^2 + 1 else (a + 3) * Real.exp (a * x)

-- Define monotonicity
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Theorem statement
theorem f_monotonic_a_range (a : ℝ) :
  monotonic (f a) → a ∈ Set.Icc (-2) 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_a_range_l823_82328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_equations_l823_82378

noncomputable section

-- Define the triangle ABC
def A : ℝ × ℝ := (-3, 0)
noncomputable def B : ℝ × ℝ := (1, Real.sqrt 3)
def C : ℝ × ℝ := (4, 0)

-- Define the equations
def line_BC (x y : ℝ) : Prop := Real.sqrt 3 * x + 4 * y - 4 * Real.sqrt 3 = 0
def median_equation (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- State the theorem
theorem right_triangle_equations :
  -- ABC is a right triangle with right angle at B
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (B.1 - C.1)^2 + (B.2 - C.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 →
  -- B is the right angle
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →
  -- C is on the x-axis
  C.2 = 0 →
  -- Equation of line BC
  (∀ x y, line_BC x y ↔ (x - B.1) * (C.2 - B.2) = (y - B.2) * (C.1 - B.1)) ∧
  -- Equation of median to hypotenuse
  (∀ x y, median_equation x y ↔ (x - ((A.1 + C.1) / 2)) * (B.2 - 0) = (y - 0) * (B.1 - ((A.1 + C.1) / 2))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_equations_l823_82378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l823_82305

def point : ℝ × ℝ × ℝ := (3, -2, 1)

def line1_direction : ℝ × ℝ × ℝ := (2, 3, -1)
def line2_direction : ℝ × ℝ × ℝ := (4, -1, 3)

def perpendicular_line (p : ℝ × ℝ × ℝ) (d1 d2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ → Prop :=
  λ x => ∃ t : ℝ, x = (p.fst + 4*t, p.snd.fst - 5*t, p.snd.snd - 7*t) ∧
         (4 * d1.fst - 5 * d1.snd.fst - 7 * d1.snd.snd = 0) ∧
         (4 * d2.fst - 5 * d2.snd.fst - 7 * d2.snd.snd = 0)

theorem perpendicular_line_equation :
  perpendicular_line point line1_direction line2_direction =
  λ (x : ℝ × ℝ × ℝ) => (x.fst - 3) / 4 = (x.snd.fst + 2) / (-5) ∧ (x.snd.fst + 2) / (-5) = (x.snd.snd - 1) / (-7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l823_82305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l823_82316

-- Define the given points
def p1 : ℝ × ℝ := (1, 5)
def p2 : ℝ × ℝ := (4, -3)
def p3 : ℝ × ℝ := (9, 5)

-- Define the ellipse center
noncomputable def center : ℝ × ℝ := ((p1.1 + p3.1) / 2, (p1.2 + p3.2) / 2)

-- Define the semi-major and semi-minor axes
noncomputable def a : ℝ := Real.sqrt ((p2.1 - center.1)^2 + (p2.2 - center.2)^2)
noncomputable def b : ℝ := (p3.1 - p1.1) / 2

-- Theorem statement
theorem ellipse_foci_distance : 
  2 * Real.sqrt (a^2 - b^2) = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l823_82316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_one_l823_82326

/-- The infinite repeating decimal 0.999... -/
noncomputable def repeating_decimal : ℝ := ∑' n, 9 * (1/10)^(n+1)

/-- Theorem stating that the infinite repeating decimal 0.999... is equal to 1 -/
theorem repeating_decimal_equals_one : repeating_decimal = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_one_l823_82326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l823_82393

theorem beta_value (α β : Real) 
  (h1 : Real.sin α = Real.sqrt 5 / 5)
  (h2 : Real.sin (α - β) = -(Real.sqrt 10 / 10))
  (h3 : 0 < α ∧ α < Real.pi/2)
  (h4 : 0 < β ∧ β < Real.pi/2) :
  β = Real.pi/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l823_82393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l823_82377

/-- The line equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A triangle bounded by the x-axis, y-axis, and a line -/
structure AxisAlignedTriangle where
  boundingLine : Line

/-- Calculate the area of an axis-aligned triangle -/
noncomputable def areaOfAxisAlignedTriangle (t : AxisAlignedTriangle) : ℝ :=
  let xIntercept := t.boundingLine.c / t.boundingLine.a
  let yIntercept := t.boundingLine.c / t.boundingLine.b
  (xIntercept * yIntercept) / 2

theorem area_of_specific_triangle :
  let t := AxisAlignedTriangle.mk (Line.mk 3 1 9)
  areaOfAxisAlignedTriangle t = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l823_82377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l823_82304

/-- The cubic polynomial q(x) that satisfies the given conditions -/
noncomputable def q (x : ℝ) : ℝ := (8/3) * x^3 - (52/3) * x^2 + (34/3) * x - 2

/-- Theorem stating that q(x) satisfies all the given conditions -/
theorem q_satisfies_conditions :
  q 0 = 2 ∧
  q 1 = -8 ∧
  q 2 = -18 ∧
  q 3 = -20 ∧
  q 1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l823_82304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_nuts_count_l823_82394

theorem initial_nuts_count (eaten_fraction : Rat) (remaining_nuts : Nat) : Nat :=
  let initial_nuts := 30
  by
    have h1 : eaten_fraction = 5/6 := by sorry
    have h2 : remaining_nuts = 5 := by sorry
    have h3 : (1 - eaten_fraction) * initial_nuts = remaining_nuts := by sorry
    sorry

#check initial_nuts_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_nuts_count_l823_82394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_cubic_yards_approx_cubic_meters_l823_82367

/-- Conversion factor from yards to meters -/
def yard_to_meter : ℝ := 0.9144

/-- Volume in cubic meters of 5 cubic yards -/
def five_cubic_yards_in_cubic_meters : ℝ := 5 * (yard_to_meter ^ 3)

/-- Theorem stating that 5 cubic yards is approximately 3.82277 cubic meters -/
theorem five_cubic_yards_approx_cubic_meters :
  abs (five_cubic_yards_in_cubic_meters - 3.82277) < 0.00001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_cubic_yards_approx_cubic_meters_l823_82367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marks_chessboard_l823_82385

/-- Represents the chessboard marking operation -/
def can_mark (board : Fin n → Fin n → Bool) (r1 r2 c1 c2 : Fin n) : Bool :=
  (board r1 c1 && board r1 c2 && board r2 c1) ||
  (board r1 c1 && board r1 c2 && board r2 c2) ||
  (board r1 c1 && board r2 c1 && board r2 c2) ||
  (board r1 c2 && board r2 c1 && board r2 c2)

/-- Checks if all squares can be marked given an initial marking -/
def can_mark_all (n : ℕ) (initial_marks : Fin n → Fin n → Bool) : Prop :=
  ∃ (final_marks : Fin n → Fin n → Bool),
    (∀ i j, initial_marks i j → final_marks i j) ∧
    (∀ i j, final_marks i j) ∧
    (∀ i j, ¬initial_marks i j → 
      ∃ (r1 r2 c1 c2 : Fin n), can_mark final_marks r1 r2 c1 c2 ∧ 
        ((r1 = i ∧ c1 = j) ∨ (r1 = i ∧ c2 = j) ∨ (r2 = i ∧ c1 = j) ∨ (r2 = i ∧ c2 = j)))

/-- The main theorem: (n-1)² + 1 is the smallest number of initial marks needed -/
theorem min_marks_chessboard (n : ℕ) (h : n > 1) :
  ∀ m : ℕ, (∀ initial_marks : Fin n → Fin n → Bool, 
    (Finset.sum (Finset.univ : Finset (Fin n)) (λ i ↦ 
      Finset.sum (Finset.univ : Finset (Fin n)) (λ j ↦ 
        if initial_marks i j then 1 else 0)) = m) →
    can_mark_all n initial_marks) ↔ m ≥ (n - 1)^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marks_chessboard_l823_82385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_octagon_HX_extended_octagon_HX_proof_l823_82348

/-- Regular octagon with side length 3 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Point X on the extension of side AB such that AX = 4AB -/
def extension_point (octagon : RegularOctagon) : ℝ :=
  4 * octagon.side_length

/-- Length of segment HX in the extended octagon -/
noncomputable def segment_HX (octagon : RegularOctagon) : ℝ :=
  9 + 6 * Real.sqrt 2

/-- Theorem stating the length of HX in the extended regular octagon -/
theorem extended_octagon_HX (octagon : RegularOctagon) :
  segment_HX octagon = 9 + 6 * Real.sqrt 2 :=
by
  -- Unfold the definition of segment_HX
  unfold segment_HX
  -- The equality is now trivial
  rfl

/-- Proof that the calculated length is correct -/
theorem extended_octagon_HX_proof (octagon : RegularOctagon) :
  (segment_HX octagon) ^ 2 = 81 + 36 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_octagon_HX_extended_octagon_HX_proof_l823_82348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_cone_volume_proof_l823_82395

/-- The volume of a cone with slant height 15 cm and height 9 cm is 432π cubic centimeters. -/
theorem cone_volume (π : ℝ) : ℝ :=
  let slant_height : ℝ := 15
  let height : ℝ := 9
  let radius : ℝ := Real.sqrt (slant_height^2 - height^2)
  (1/3) * π * radius^2 * height

/-- Proof that the volume of the cone is indeed 432π cubic centimeters. -/
theorem cone_volume_proof : cone_volume Real.pi = 432 * Real.pi := by
  -- Unfold the definition of cone_volume
  unfold cone_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_cone_volume_proof_l823_82395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l823_82327

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the Euclidean distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The shortest path problem for the cowboy -/
theorem cowboy_shortest_path (cowboy stream cabin : Point)
  (h1 : cowboy.x = 0 ∧ cowboy.y = -6)
  (h2 : stream.y = 0)
  (h3 : cabin.x = cowboy.x + 10 ∧ cabin.y = cowboy.y - 5) :
  distance cowboy { x := cowboy.x, y := stream.y } + distance { x := cowboy.x, y := stream.y } cabin = Real.sqrt 389 := by
  sorry

#check cowboy_shortest_path

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cowboy_shortest_path_l823_82327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l823_82315

noncomputable def A : Set ℝ := {1, 2, 4}

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def B : Set ℝ := f '' A

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l823_82315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_implies_m_eq_neg_one_perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two_l823_82357

def OA : ℝ × ℝ := (-1, 3)
def OB : ℝ × ℝ := (3, -1)
def OC (m : ℝ) : ℝ × ℝ := (m, 1)

def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)
def AC (m : ℝ) : ℝ × ℝ := ((OC m).1 - OA.1, (OC m).2 - OA.2)
def BC (m : ℝ) : ℝ × ℝ := ((OC m).1 - OB.1, (OC m).2 - OB.2)

def isParallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def isPerpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem parallel_implies_m_eq_neg_one (m : ℝ) :
  isParallel AB (OC m) → m = -1 := by sorry

theorem perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two (m : ℝ) :
  isPerpendicular (AC m) (BC m) → m = 1 + 2 * Real.sqrt 2 ∨ m = 1 - 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_implies_m_eq_neg_one_perpendicular_implies_m_eq_one_plus_minus_two_sqrt_two_l823_82357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_identity_l823_82396

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1/2, -Real.sqrt 3/2; Real.sqrt 3/2, 1/2]

theorem smallest_power_identity : 
  (∀ k : ℕ, 0 < k → k < 12 → A^k ≠ 1) ∧ A^12 = 1 := by
  sorry

#check smallest_power_identity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_identity_l823_82396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_negative_six_delta_satisfies_limit_condition_l823_82314

-- Define the function f(x) = (3x^2 + 17x - 6) / (x + 6)
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 17 * x - 6) / (x + 6)

-- State the limit theorem
theorem limit_of_f_at_negative_six :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ -6 → 0 < |x + 6| → |x + 6| < δ → |f x + 19| < ε := by
  sorry

-- Define delta as a function of epsilon
noncomputable def delta (ε : ℝ) : ℝ := ε / 3

-- Prove that delta satisfies the limit condition
theorem delta_satisfies_limit_condition :
  ∀ ε > 0, ∀ x : ℝ, x ≠ -6 → 0 < |x + 6| → |x + 6| < delta ε → |f x + 19| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_f_at_negative_six_delta_satisfies_limit_condition_l823_82314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_inclination_angle_sqrt_three_line_inclination_l823_82388

/-- The inclination angle of a line -/
noncomputable def inclination_angle (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- The inclination angle of a vertical line is 90 degrees -/
theorem vertical_line_inclination_angle (a : ℝ) : 
  let l := {p : ℝ × ℝ | p.1 = a}
  inclination_angle l = 90 := by
  sorry

/-- The line x = √3 has an inclination angle of 90 degrees -/
theorem sqrt_three_line_inclination : 
  let l := {p : ℝ × ℝ | p.1 = Real.sqrt 3}
  inclination_angle l = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_inclination_angle_sqrt_three_line_inclination_l823_82388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_rounded_l823_82373

-- Define the actual side length of the square
def actual_side_length : ℝ → ℝ := λ x ↦ x

-- Define the area of the square
def square_area (x : ℝ) : ℝ := (actual_side_length x) ^ 2

-- Define the rounding function to the nearest integer
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 0.5 then ⌊x⌋ else ⌈x⌉

-- Theorem statement
theorem square_side_length_rounded (min_area : ℝ) (h : min_area = 42.25) :
  ∀ x : ℝ, square_area x ≥ min_area → round_to_nearest (actual_side_length x) = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_rounded_l823_82373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l823_82301

-- Define the power function as noncomputable
noncomputable def powerFunction (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- State the theorem
theorem power_function_through_point :
  ∃ a : ℝ, (powerFunction a (1/2) = Real.sqrt 2/2) ∧
  (∀ x : ℝ, powerFunction a x = x^(1/2)) :=
by
  -- Prove the theorem
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l823_82301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_equalize_l823_82332

/-- Represents a circular arrangement of boxes with coins -/
def BoxArrangement := Vector ℕ 7

/-- The initial arrangement of coins in the boxes -/
def initial_arrangement : BoxArrangement := ⟨[9, 13, 10, 20, 5, 17, 18], by simp⟩

/-- A move transfers one coin between adjacent boxes -/
def is_valid_move (before after : BoxArrangement) : Prop :=
  ∃ (i j : Fin 7), 
    (j = i.succ ∨ (i = 6 ∧ j = 0)) ∧
    after.get i + 1 = before.get i ∧
    after.get j = before.get j + 1 ∧
    (∀ (k : Fin 7), k ≠ i ∧ k ≠ j → after.get k = before.get k)

/-- The arrangement is equalized when all boxes have the same number of coins -/
def is_equalized (arr : BoxArrangement) : Prop :=
  ∀ (i j : Fin 7), arr.get i = arr.get j

/-- The main theorem to be proved -/
theorem min_moves_to_equalize :
  ∃ (n : ℕ) (sequence : Vector BoxArrangement (n + 1)),
    sequence.get 0 = initial_arrangement ∧
    is_equalized (sequence.get n) ∧
    (∀ (i : Fin n), is_valid_move (sequence.get i.val) (sequence.get i.val.succ)) ∧
    n = 22 ∧
    (∀ (m : ℕ) (other_sequence : Vector BoxArrangement (m + 1)),
      (other_sequence.get 0 = initial_arrangement ∧
       is_equalized (other_sequence.get m) ∧
       (∀ (i : Fin m), is_valid_move (other_sequence.get i.val) (other_sequence.get i.val.succ)))
      → m ≥ 22) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_equalize_l823_82332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l823_82399

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi

def is_acute_triangle (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

def satisfies_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C = t.b + t.c

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : is_acute_triangle t)
  (h3 : satisfies_condition t)
  (h4 : t.b = 2) :
  t.A = Real.pi/3 ∧ 
  Real.sqrt 3/2 < (1/2 * t.a * t.b * Real.sin t.C) ∧ 
  (1/2 * t.a * t.b * Real.sin t.C) < 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l823_82399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l823_82317

noncomputable def series : ℕ → ℝ × ℝ
| 0 => (0, 0)  -- Arbitrary initial value
| n + 1 => let (a, b) := series n
           (2 * Real.sqrt 3 * a + b, Real.sqrt 3 * b - 2 * a)

axiom series_150 : series 150 = (-1, 3)

theorem series_sum : (series 1).1 + (series 1).2 = -1 / 4^149 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l823_82317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_difference_product_condition_l823_82350

/-- Two integers are coprime if their greatest common divisor is 1 -/
def are_coprime (a b : ℤ) : Prop := Int.gcd a b = 1

/-- A fraction is irreducible if its numerator and denominator are coprime -/
def is_irreducible_fraction (a b : ℤ) : Prop := are_coprime a b ∧ b ≠ 0

/-- The condition for two irreducible fractions where their difference equals their product -/
theorem fraction_difference_product_condition
  (a b c d : ℤ) (hb : b ≠ 0) (hd : d ≠ 0)
  (hab : is_irreducible_fraction a b) (hcd : is_irreducible_fraction c d)
  (h : (a : ℚ) / b - c / d = (a : ℚ) / b * c / d) :
  c = a ∧ d = a + b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_difference_product_condition_l823_82350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_wire_for_specific_poles_l823_82329

noncomputable section

/-- The length of the shortest wire around two cylindrical poles -/
def shortest_wire_length (d1 d2 : ℝ) : ℝ :=
  let r1 := d1 / 2
  let r2 := d2 / 2
  let center_distance := r1 + r2
  let radius_diff := r2 - r1
  let straight_section := 2 * Real.sqrt (center_distance^2 - radius_diff^2)
  let small_arc := 2 * Real.pi * r1 * (1/6)
  let large_arc := 2 * Real.pi * r2 * (1/3)
  straight_section + small_arc + large_arc

/-- Theorem stating the length of the shortest wire around two specific poles -/
theorem shortest_wire_for_specific_poles :
  shortest_wire_length 8 24 = 16 * Real.sqrt 3 + 28 * Real.pi / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_wire_for_specific_poles_l823_82329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l823_82364

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance between two parallel lines -/
noncomputable def distance (l₁ l₂ : Line) : ℝ := 
  abs (l₁.c - l₂.c) / Real.sqrt (l₁.a^2 + l₁.b^2)

theorem parallel_lines_distance : 
  ∃ (m : ℝ), 
    let l₁ : Line := ⟨3, m, -3⟩
    let l₂ : Line := ⟨6, 4, 1⟩
    l₁.a / l₁.b = l₂.a / l₂.b → 
    distance l₁ l₂ = 7 * Real.sqrt 13 / 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l823_82364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_overs_is_twelve_l823_82397

/-- Represents the cricket game scenario -/
structure CricketGame where
  target_score : ℚ
  initial_run_rate : ℚ
  required_run_rate : ℚ
  remaining_overs : ℚ

/-- Calculates the number of overs played initially -/
def initial_overs (game : CricketGame) : ℚ :=
  (game.target_score - game.required_run_rate * game.remaining_overs) / game.initial_run_rate

/-- Theorem stating that the number of overs played initially is 12 -/
theorem initial_overs_is_twelve (game : CricketGame) 
  (h1 : game.target_score = 360)
  (h2 : game.initial_run_rate = 9/2)
  (h3 : game.required_run_rate = 306/38)
  (h4 : game.remaining_overs = 38) :
  initial_overs game = 12 := by
  sorry

#eval initial_overs { target_score := 360, initial_run_rate := 9/2, required_run_rate := 306/38, remaining_overs := 38 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_overs_is_twelve_l823_82397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_pipe_fills_in_nine_hours_l823_82309

/-- Represents the time it takes for the slower pipe to fill the pool -/
noncomputable def slower_pipe_time (faster_pipe_rate : ℝ) (combined_time : ℝ) : ℝ :=
  let slower_pipe_rate := faster_pipe_rate / 1.25
  let combined_rate := faster_pipe_rate + slower_pipe_rate
  1 / slower_pipe_rate

/-- 
Given two pipes where one is 1.25 times faster than the other,
and together they fill a pool in 4 hours, prove that the slower
pipe alone will take 9 hours to fill the pool.
-/
theorem slower_pipe_fills_in_nine_hours :
  ∃ (faster_pipe_rate : ℝ),
    faster_pipe_rate > 0 ∧
    (faster_pipe_rate + faster_pipe_rate / 1.25) * 4 = 1 ∧
    slower_pipe_time faster_pipe_rate 4 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_pipe_fills_in_nine_hours_l823_82309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_theorem_l823_82340

/-- Parabola struct representing y² = 2px -/
structure Parabola where
  p : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_area_theorem (para : Parabola) (A F K : Point) :
  para.p = 4 →
  F.x = 2 ∧ F.y = 0 →
  K.x = 6 ∧ K.y = 0 →
  A.x = 8 ∧ A.y = 4 →
  A.y^2 = 2 * para.p * A.x →
  distance A K = Real.sqrt 2 * distance A F →
  (1/2) * distance A K * distance K F * Real.sqrt (1/2) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_theorem_l823_82340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l823_82381

open Real

variable (θ : ℝ)
variable (a : ℕ+ → ℝ)

axiom theta_bounds : 0 < θ ∧ θ < π / 2

axiom a_1 : a 1 = 2 * cos θ

axiom a_rec : ∀ n : ℕ+, a (n + 1) = Real.sqrt (2 + a n)

theorem a_formula : ∀ n : ℕ+, a n = 2 * cos (θ / (2 ^ (n - 1 : ℕ))) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l823_82381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_sufficient_not_necessary_l823_82386

/-- Two planes are different if they are not equal -/
def different_planes (α β : Set (Fin 3 → ℝ)) : Prop := α ≠ β

/-- Two lines are different if they are not equal -/
def different_lines (m n : Set (Fin 3 → ℝ)) : Prop := m ≠ n

/-- The intersection of two planes is a line -/
def planes_intersection (α β m : Set (Fin 3 → ℝ)) : Prop := α ∩ β = m

/-- A line is not contained in a plane -/
def line_not_in_plane (n α : Set (Fin 3 → ℝ)) : Prop := ¬(n ⊆ α)

/-- A line is parallel to a plane -/
noncomputable def line_parallel_plane (n α : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A line is parallel to another line -/
noncomputable def line_parallel_line (n m : Set (Fin 3 → ℝ)) : Prop := sorry

theorem parallel_line_sufficient_not_necessary
  (α β : Set (Fin 3 → ℝ)) (m n : Set (Fin 3 → ℝ))
  (h1 : different_planes α β)
  (h2 : different_lines m n)
  (h3 : planes_intersection α β m)
  (h4 : line_not_in_plane n α)
  (h5 : line_not_in_plane n β) :
  (line_parallel_line n m → line_parallel_plane n α ∧ line_parallel_plane n β) ∧
  ¬(line_parallel_plane n α ∧ line_parallel_plane n β → line_parallel_line n m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_sufficient_not_necessary_l823_82386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_first_quadrant_l823_82362

/-- A complex number z = 1/(1+ai) is in the first quadrant iff a < 0 -/
theorem complex_first_quadrant (a : ℝ) :
  (0 < ((1 : ℂ) / (1 + a * I)).re ∧ 0 < ((1 : ℂ) / (1 + a * I)).im) ↔ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_first_quadrant_l823_82362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_identification_implies_half_zombie_no_english_l823_82354

/-- Represents the possible types of natives on the island --/
inductive NativeType
  | Human
  | Zombie
  | HalfZombie

/-- Represents the possible languages a native can use to answer --/
inductive AnswerLanguage
  | English
  | Native

/-- Represents the possible answers a native can give --/
inductive Answer
  | Yes
  | No
  | Bal
  | Da

/-- Represents a native's response to the question --/
structure Response where
  answer : Answer
  language : AnswerLanguage

/-- Function to determine if a response allows unique identification of native type --/
def allowsUniqueIdentification (r : Response) : Prop := sorry

/-- The main theorem to be proved --/
theorem unique_identification_implies_half_zombie_no_english 
  (r : Response) (t : NativeType) :
  allowsUniqueIdentification r → 
  (r.answer = Answer.No ∧ r.language = AnswerLanguage.English ∧ t = NativeType.HalfZombie) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_identification_implies_half_zombie_no_english_l823_82354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_half_plus_theta_l823_82391

theorem cos_three_pi_half_plus_theta (θ : Real) 
  (h1 : Real.sin (θ - Real.pi/6) = 1/4) 
  (h2 : θ ∈ Set.Ioo (Real.pi/6) (2*Real.pi/3)) : 
  Real.cos (3*Real.pi/2 + θ) = (Real.sqrt 15 + Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_three_pi_half_plus_theta_l823_82391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_simplification_l823_82319

theorem absolute_value_simplification (x : ℝ) :
  (abs (3*x - 2) + abs (2*x + 3) = 
    if x < -3/2 then -5*x - 1
    else if x < 2/3 then -x + 5
    else 5*x + 1) ∧
  (abs (abs (x - 1) - 3) + abs (3*x + 1) = 
    if x < -2 then -4*x - 3
    else if x < -1/3 then -2*x + 1
    else if x < 1 then 4*x + 3
    else if x < 4 then 2*x + 5
    else 4*x - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_simplification_l823_82319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_fold_l823_82343

-- Define the page as a rectangle in 2D space
structure Page where
  width : ℝ
  height : ℝ

-- Define points on the page
structure Point where
  x : ℝ
  y : ℝ

-- Define a line segment
structure LineSegment where
  start : Point
  end_ : Point

-- Define the fold
def Fold (p : Page) (start : Point) (end_ : Point) : LineSegment :=
  { start := start, end_ := end_ }

-- Function to calculate the length of a line segment
noncomputable def length (l : LineSegment) : ℝ :=
  Real.sqrt ((l.end_.x - l.start.x)^2 + (l.end_.y - l.start.y)^2)

-- Define the geometric construction
def geometricConstruction (p : Page) (A B C D E F H : Point) : Prop :=
  -- A is the lower outer corner
  A.x = 0 ∧ A.y = 0 ∧
  -- B is on the inner edge
  B.x = p.width ∧
  -- H is directly above B on the top edge
  H.x = B.x ∧ H.y = p.height ∧
  -- C is the midpoint of AB
  C.x = (A.x + B.x) / 2 ∧ C.y = (A.y + B.y) / 2 ∧
  -- D is the midpoint of AC
  D.x = (A.x + C.x) / 2 ∧ D.y = (A.y + C.y) / 2 ∧
  -- E is on the semicircle with diameter DB
  (E.x - D.x)^2 + (E.y - D.y)^2 = (B.x - D.x)^2 + (B.y - D.y)^2 ∧
  -- E is also on the line CG parallel to BH
  (E.y - C.y) / (E.x - C.x) = (H.y - B.y) / (H.x - B.x) ∧
  -- F is on the inner edge
  F.x = p.width ∧
  -- DEF is a straight line
  (F.y - E.y) / (F.x - E.x) = (E.y - D.y) / (E.x - D.x)

-- Theorem statement
theorem shortest_fold (p : Page) (A B C D E F H : Point) :
  geometricConstruction p A B C D E F H →
  ∀ (G : Point), G.x = p.width → length (Fold p A G) ≥ length (Fold p A F) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_fold_l823_82343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_positive_integers_l823_82398

def M : Set ℕ := sorry

axiom M_contains_2018 : (2018 : ℕ) ∈ M

axiom M_contains_divisors :
  ∀ m : ℕ, m ∈ M → ∀ d : ℕ, d > 0 ∧ d ∣ m → d ∈ M

axiom M_closed_under_operation :
  ∀ k m : ℕ, k ∈ M → m ∈ M → 1 < k → k < m → k * m + 1 ∈ M

theorem M_equals_positive_integers : M = {n : ℕ | n > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_positive_integers_l823_82398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_completion_time_l823_82372

/-- The time taken by x to complete the work alone -/
noncomputable def time_x : ℝ := sorry

/-- The time taken by y to complete the work alone -/
noncomputable def time_y : ℝ := sorry

/-- The portion of work completed in one day -/
noncomputable def work_per_day (time : ℝ) : ℝ := 1 / time

/-- The total work is 1 unit -/
def total_work : ℝ := 1

theorem x_completion_time :
  time_y = 35 →
  work_per_day time_x * 8 + work_per_day time_y * 28 = total_work →
  time_x = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_completion_time_l823_82372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_sum_diff_collinear_l823_82380

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def isCollinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem collinear_iff_sum_diff_collinear (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0)) :
  isCollinear a b ↔ isCollinear (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_sum_diff_collinear_l823_82380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_digit_divisible_number_l823_82334

/-- Represents an n-digit number as a list of its digits -/
def DigitList := List Nat

/-- Checks if all digits in the list are non-zero -/
def allNonZero (digits : DigitList) : Prop :=
  digits.all (· ≠ 0)

/-- Converts a list of digits to a natural number -/
def listToNat (digits : DigitList) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

/-- Checks if a number represented by digits is divisible by another -/
def isDivisibleBy (dividend : DigitList) (divisor : DigitList) : Prop :=
  (listToNat dividend) % (listToNat divisor) = 0

/-- Main theorem statement -/
theorem largest_n_digit_divisible_number :
  (∀ n : Nat, n > 6 →
    ¬∃ (digits : DigitList),
      digits.length = n ∧
      allNonZero digits ∧
      digits.headI ≠ 0 ∧
      (digits.get? 1).isSome ∧
      digits.getLast? ≠ none ∧
      isDivisibleBy digits (digits.tail)) ∧
  (∃ (digits : DigitList),
    digits.length = 6 ∧
    allNonZero digits ∧
    digits.headI ≠ 0 ∧
    (digits.get? 1).isSome ∧
    digits.getLast? ≠ none ∧
    isDivisibleBy digits (digits.tail)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_digit_divisible_number_l823_82334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l823_82363

structure RightTriangle where
  DE : ℝ
  DF : ℝ
  EF : ℝ
  right_angle : DE^2 = DF^2 + EF^2

theorem right_triangle_side_length 
  (DEF : RightTriangle) 
  (cos_F : Real.cos (Real.arccos (DEF.DF / DEF.DE)) = (5 * Real.sqrt 221) / 221)
  (hypotenuse : DEF.DE = Real.sqrt 221) :
  DEF.DF = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l823_82363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sqrt_count_l823_82308

theorem integer_sqrt_count : 
  ∃! (S : Finset ℝ), 
    (∀ x ∈ S, ∃ k : ℕ, (123 - Real.sqrt x).sqrt = k) ∧ 
    (∀ x : ℝ, (∃ k : ℕ, (123 - Real.sqrt x).sqrt = k) → x ∈ S) ∧
    Finset.card S = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_sqrt_count_l823_82308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_regular_octagon_approx_l823_82312

/-- The length of a diagonal in a regular octagon with side length 12 -/
noncomputable def diagonal_length_regular_octagon : ℝ := 24 * Real.cos (67.5 * Real.pi / 180)

/-- Theorem: The length of a diagonal in a regular octagon with side length 12 is approximately 9.192 -/
theorem diagonal_length_regular_octagon_approx :
  ∃ ε > 0, |diagonal_length_regular_octagon - 9.192| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_regular_octagon_approx_l823_82312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_32_5_l823_82300

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a trapezoid given its vertices -/
noncomputable def trapezoidArea (e f g h : Point) : ℝ :=
  let height := |g.x - e.x|
  let base1 := |f.y - e.y|
  let base2 := |g.y - h.y|
  (base1 + base2) * height / 2

/-- Theorem: The area of trapezoid EFGH with given vertices is 32.5 square units -/
theorem trapezoid_area_is_32_5 :
  let e := Point.mk 2 (-3)
  let f := Point.mk 2 2
  let g := Point.mk 7 8
  let h := Point.mk 7 0
  trapezoidArea e f g h = 32.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_32_5_l823_82300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roll_theorem_l823_82337

/-- The length of the path followed by the center point of a face of a unit cube when rolled -/
noncomputable def cube_roll_path_length : ℝ := ((1 + Real.sqrt 5) / 2) * Real.pi

/-- Theorem stating the length of the path followed by the center point of a face of a unit cube -/
theorem cube_roll_theorem :
  let cube_edge_length : ℝ := 1
  let always_two_vertices_touching : Prop := True
  let dot_on_top_face_center : Prop := True
  let roll_without_lifting_or_slipping : Prop := True
  let roll_until_dot_on_top : Prop := True
  cube_roll_path_length = ((1 + Real.sqrt 5) / 2) * Real.pi :=
by
  sorry

#check cube_roll_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roll_theorem_l823_82337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_part_sum_l823_82360

theorem max_real_part_sum (z : Fin 12 → ℂ) (w : Fin 12 → ℂ) : 
  (∀ j : Fin 12, z j ^ 12 = (2 : ℂ) ^ 36) →
  (∀ j : Fin 12, w j = z j ∨ w j = Complex.I * z j) →
  (∃ m n : ℕ+, ∃ x : ℝ, x = Finset.sum Finset.univ (λ j => (w j).re) ∧ x = m + Real.sqrt n) →
  ∃ m n : ℕ+, m + n = 784 ∧ ∃ x : ℝ, x = Finset.sum Finset.univ (λ j => (w j).re) ∧ x = m + Real.sqrt n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_part_sum_l823_82360
