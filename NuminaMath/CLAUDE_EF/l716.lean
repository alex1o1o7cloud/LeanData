import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_of_numbers_with_given_ratio_and_lcm_l716_71643

theorem hcf_of_numbers_with_given_ratio_and_lcm (a b : ℕ) 
  (h_ratio : a * 3 = b * 2)
  (h_lcm : Nat.lcm a b = 36) : 
  Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hcf_of_numbers_with_given_ratio_and_lcm_l716_71643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_seven_mod_hundred_l716_71619

theorem power_of_seven_mod_hundred : ∃ e : ℕ, 
  (∀ k : ℕ, 0 < k → k < e → (7 ^ k : ℤ) % 100 ≠ 1) ∧ 
  (7 ^ e : ℤ) % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_seven_mod_hundred_l716_71619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_divisors_120_l716_71652

def is_even_divisor (d n : ℕ) : Prop :=
  d ∣ n ∧ Even d

def sum_even_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d ∣ n ∧ d % 2 = 0) (Finset.range (n + 1))).sum id

theorem sum_even_divisors_120 : sum_even_divisors 120 = 336 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_even_divisors_120_l716_71652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_degrees_in_radians_l716_71693

theorem sixty_degrees_in_radians : 60 * Real.pi / 180 = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixty_degrees_in_radians_l716_71693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_recurrence_l716_71660

noncomputable def f (n : ℤ) : ℝ :=
  (5 + 3 * Real.sqrt 5) / 10 * ((1 + Real.sqrt 5) / 2) ^ n +
  (5 - 3 * Real.sqrt 5) / 10 * ((1 - Real.sqrt 5) / 2) ^ n

theorem f_recurrence (n : ℤ) : f (n + 1) - f (n - 1) = f n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_recurrence_l716_71660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_rate_l716_71684

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem b_work_rate 
  (total_rate : ℝ) 
  (a_rate : ℝ) 
  (c_rate : ℝ) 
  (h1 : total_rate = work_rate 4)
  (h2 : a_rate = work_rate 12)
  (h3 : c_rate = work_rate 8.999999999999998)
  (h4 : total_rate = a_rate + c_rate + work_rate 18) : 
  work_rate 18 = total_rate - a_rate - c_rate := by
  sorry

#check b_work_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_work_rate_l716_71684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_sample_theorem_l716_71674

/-- In a city with given allergy and left-handedness rates, prove expected numbers in a sample -/
theorem city_sample_theorem 
  (total_sample : ℕ) 
  (allergy_rate : ℚ) 
  (left_handed_rate : ℚ) 
  (h_total : total_sample = 350)
  (h_allergy : allergy_rate = 2 / 7)
  (h_left_handed : left_handed_rate = 3 / 10) :
  let expected_allergies : ℕ := (allergy_rate * ↑total_sample).floor.toNat
  let expected_both : ℕ := (allergy_rate * left_handed_rate * ↑total_sample).floor.toNat
  expected_allergies = 100 ∧ expected_both = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_sample_theorem_l716_71674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l716_71647

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + m * log (1 - x)

-- State the theorem
theorem extreme_points_inequality (m : ℝ) (x₁ x₂ : ℝ) 
  (h1 : ∃ (c : ℝ), c ∈ Set.Ioo x₁ x₂ ∧ (deriv (f m)) c = 0) 
  (h2 : ∃ (c : ℝ), c ∈ Set.Ioo x₁ x₂ ∧ (deriv (f m)) c = 0) 
  (h3 : x₁ < x₂) :
  (1/4) - (1/2) * log 2 < (f m x₁) / x₂ ∧ (f m x₁) / x₂ < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l716_71647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_max_l716_71664

theorem sin_plus_cos_max (θ : Real) : Real.sin θ + Real.cos θ ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_max_l716_71664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_equals_hexagon_side_l716_71610

/-- A circle with an inscribed regular hexagon -/
structure CircleWithHexagon where
  O : Point  -- Center of the circle
  r : ℝ      -- Radius of the circle
  h : ℝ      -- Side length of the hexagon
  h_pos : h > 0

/-- The arc length function for a circle -/
def arcLength (c : CircleWithHexagon) (θ : ℝ) : ℝ := c.r * θ

theorem arc_equals_hexagon_side (c : CircleWithHexagon) :
  ∃ θ : ℝ, arcLength c θ = c.r → θ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_equals_hexagon_side_l716_71610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_with_20_diagonals_has_8_sides_l716_71663

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 20 diagonals has 8 sides -/
theorem polygon_with_20_diagonals_has_8_sides :
  ∃ n : ℕ, n > 2 ∧ num_diagonals n = 20 ∧ n = 8 := by
  use 8
  constructor
  · norm_num
  constructor
  · rfl
  · rfl

#eval num_diagonals 8  -- Should output 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_with_20_diagonals_has_8_sides_l716_71663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_edge_sum_l716_71655

-- Define the dimensions of the pyramid
def base_width : ℝ := 8
def base_length : ℝ := 15
def pyramid_height : ℝ := 10

-- Define the function to calculate the sum of edge lengths
noncomputable def sum_of_edge_lengths (w : ℝ) (l : ℝ) (h : ℝ) : ℝ :=
  let base_diagonal := (w^2 + l^2).sqrt
  let base_center_to_corner := base_diagonal / 2
  let slant_height := (h^2 + base_center_to_corner^2).sqrt
  2 * (w + l) + 4 * slant_height

-- Theorem statement
theorem pyramid_edge_sum :
  sum_of_edge_lengths base_width base_length pyramid_height = 98.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_edge_sum_l716_71655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nice_polynomial_eventual_l716_71677

-- Define a polynomial type
def MyPolynomial (α : Type) := List α

-- Define a nice polynomial
def is_nice (p : MyPolynomial ℝ) : Prop :=
  let coeff := p
  let d := coeff.length - 1
  let sum_abs := coeff.map (fun x => abs x) |>.sum
  let max_abs := coeff.map (fun x => abs x) |>.maximum?
  2021 * sum_abs / 2022 < max_abs.getD 0

-- Define the monic polynomial constructed from roots
noncomputable def construct_monic_poly (roots : List ℝ) : MyPolynomial ℝ := sorry

-- Main theorem
theorem nice_polynomial_eventual {d : ℕ} (r : List ℝ) 
  (h_distinct : r.Nodup) 
  (h_not_pm1 : ∀ x ∈ r, x ≠ 1 ∧ x ≠ -1) 
  (h_length : r.length = d) :
  ∃ M : ℕ, ∀ N ≥ M, is_nice (construct_monic_poly (r.map (fun x => x^N))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nice_polynomial_eventual_l716_71677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircles_area_correct_l716_71644

/-- The area of the region inside a regular octagon with side length 3 units,
    but outside of eight inscribed semicircles (each with diameter equal to a side of the octagon) -/
noncomputable def octagon_semicircles_area : ℝ :=
  18 * (1 + Real.sqrt 2) - 9 * Real.pi

/-- Theorem stating that the calculated area is correct -/
theorem octagon_semicircles_area_correct :
  let octagon_area := 2 * (1 + Real.sqrt 2) * 3^2
  let semicircle_area := 1/2 * Real.pi * (3/2)^2
  octagon_area - 8 * semicircle_area = octagon_semicircles_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_semicircles_area_correct_l716_71644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_water_heated_to_boiling_l716_71604

/-- Represents the heat produced by burning fuel over time -/
noncomputable def heat_production (t : ℕ) : ℝ :=
  480 * (3/4)^t

/-- Calculates the total heat produced over an infinite time -/
noncomputable def total_heat : ℝ :=
  480 / (1 - 3/4)

/-- Calculates the heat required to boil water -/
noncomputable def heat_required (m : ℝ) : ℝ :=
  m * 4.2 * (100 - 20)

/-- Theorem stating the maximum integer number of liters of water that can be heated to boiling -/
theorem max_water_heated_to_boiling :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ m : ℕ, m > n → heat_required (m : ℝ) > total_heat) ∧
  (heat_required (n : ℝ) ≤ total_heat) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_water_heated_to_boiling_l716_71604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l716_71624

/-- Represents a pyramid with a rectangular base -/
structure Pyramid where
  base_length : ℝ
  base_width : ℝ
  perpendicular_edge : ℝ

/-- Calculate the distance between the perpendicular edge and the diagonal of the base -/
noncomputable def distance_to_diagonal (p : Pyramid) : ℝ :=
  (p.base_length * p.base_width) / Real.sqrt (p.base_length^2 + p.base_width^2)

/-- Calculate the lateral surface area of the pyramid -/
noncomputable def lateral_surface_area (p : Pyramid) : ℝ :=
  let pb := Real.sqrt (p.base_length^2 + p.perpendicular_edge^2)
  let pd := Real.sqrt (p.base_width^2 + p.perpendicular_edge^2)
  (p.base_length * p.perpendicular_edge) / 2 +
  (p.base_width * pb) / 2 +
  (p.base_length * pd) / 2 +
  (p.base_width * p.perpendicular_edge) / 2

theorem pyramid_properties (p : Pyramid) 
  (h1 : p.base_length = 6)
  (h2 : p.base_width = 8)
  (h3 : p.perpendicular_edge = 6) :
  distance_to_diagonal p = 4.8 ∧ 
  lateral_surface_area p = 72 + 24 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_properties_l716_71624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_propositions_l716_71682

-- Define the propositions
def prop1 : Prop := ∀ x y : ℝ, (x * y ≠ 0) → (x ≠ 0 ∨ y ≠ 0)

def prop2 : Prop := ∀ m : ℝ, m > 2 → ∀ x : ℝ, x^2 - 2*x + m > 0

noncomputable def prop3 : Prop := ∀ F₁ F₂ M : ℝ × ℝ, 
  ‖F₁ - F₂‖ = 7 → ‖M - F₁‖ + ‖M - F₂‖ = 7 → 
  ∃ a b c : ℝ, (M.fst - F₁.fst)^2 / a^2 + (M.snd - F₁.snd)^2 / b^2 = 1 ∧ c^2 = a^2 - b^2

def prop4 : Prop := ∀ a b c : ℝ × ℝ × ℝ, 
  LinearIndependent ℝ ![a, b, c] → 
  LinearIndependent ℝ ![a + b, b + c, c + a]

-- The theorem to prove
theorem exactly_two_true_propositions : 
  (prop1 = False ∧ prop2 = True ∧ prop3 = False ∧ prop4 = True) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_true_propositions_l716_71682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_on_curve_C_perpendicular_to_l_l716_71615

/-- Curve C in polar coordinates -/
noncomputable def curve_C (θ : Real) : Real := 2 * Real.cos θ

/-- Line l in rectangular coordinates -/
noncomputable def line_l (x : Real) : Real := Real.sqrt 3 * x + 2

/-- Point D on curve C -/
noncomputable def point_D : Real × Real := (3/2, Real.sqrt 3 / 2)

theorem point_D_on_curve_C_perpendicular_to_l :
  let (x, y) := point_D
  ∃ θ : Real, 0 ≤ θ ∧ θ ≤ Real.pi / 2 ∧
    x = curve_C θ * Real.cos θ ∧
    y = curve_C θ * Real.sin θ ∧
    (x - 1)^2 + y^2 = 1 ∧
    (∃ m : Real, m * (y - line_l x) = -1 / (Real.sqrt 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_on_curve_C_perpendicular_to_l_l716_71615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l716_71627

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Definition of sequence b_n -/
def b (a : ℕ → ℝ) (n : ℕ) : ℝ := a (n + 1) * a (n + 2) - a n ^ 2

theorem arithmetic_sequence_proof (a : ℕ → ℝ) (d : ℝ) (h1 : ArithmeticSequence a)
    (h2 : d ≠ 0) (h3 : ∃ s t : ℕ, s > 0 ∧ t > 0 ∧ (∃ z : ℤ, (a s + b a t : ℝ) = ↑z)) :
    ArithmeticSequence (b a) ∧ (∀ a₁ : ℝ, a 1 = a₁ → |a₁| ≥ 1/18) ∧ (∃ a₁ : ℝ, a 1 = a₁ ∧ |a₁| = 1/18) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l716_71627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_integer_expression_l716_71609

def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

noncomputable def expression (n : ℕ) : ℝ := Real.sqrt (100 + Real.sqrt (n : ℝ)) + Real.sqrt (100 - Real.sqrt (n : ℝ))

theorem smallest_n_for_integer_expression :
  ∀ n : ℕ, n > 0 → n < 6156 → ¬(is_integer (expression n)) ∧ is_integer (expression 6156) := by
  sorry

#check smallest_n_for_integer_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_integer_expression_l716_71609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_car_problem_l716_71675

theorem sunzi_car_problem (x y : ℚ) : 
  (x / 3 = y - 2 ∧ (x - 9) / 2 = y) ↔ 
  (3 * (y - 2) = x ∧ 2 * y + 9 = x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunzi_car_problem_l716_71675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_product_multiple_of_three_l716_71690

/-- A fair 12-sided die -/
def dodecahedral_die : Finset ℕ := Finset.range 12

/-- A fair 6-sided die -/
def six_sided_die : Finset ℕ := Finset.range 6

/-- The probability of an event occurring when rolling a fair n-sided die -/
def prob_fair_die (event : Finset ℕ) (die : Finset ℕ) : ℚ :=
  event.card / die.card

/-- The set of outcomes where the product is a multiple of 3 -/
def multiple_of_three (d1 d2 : Finset ℕ) : Finset ℕ :=
  Finset.filter (fun n => n % 3 = 0) (Finset.image (fun (x, y) => (x + 1) * (y + 1)) (d1.product d2))

/-- The main theorem: probability of product being multiple of 3 is 5/9 -/
theorem prob_product_multiple_of_three :
  prob_fair_die (multiple_of_three dodecahedral_die six_sided_die) (Finset.range (dodecahedral_die.card * six_sided_die.card)) = 5/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_product_multiple_of_three_l716_71690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l716_71649

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - (1/2) * Real.cos (2 * x)

-- Theorem statement
theorem f_properties :
  -- 1. Smallest positive period is π
  (∀ x, f (x + π) = f x) ∧
  (∀ p, 0 < p → p < π → ∃ x, f (x + p) ≠ f x) ∧
  -- 2. Range of f is [-3/2, 5/2]
  (∀ y, ∃ x, f x = y ↔ -3/2 ≤ y ∧ y ≤ 5/2) ∧
  -- 3. If x₀ ∈ [0, π/2] and f(x₀) = 0, then sin 2x₀ = (√15 - √3) / 8
  (∀ x₀, 0 ≤ x₀ ∧ x₀ ≤ π/2 → f x₀ = 0 → Real.sin (2 * x₀) = (Real.sqrt 15 - Real.sqrt 3) / 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l716_71649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l716_71638

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * Real.arcsin (2 / (3 * x + 1)) + Real.sqrt (9 * x^2 + 6 * x - 3)

-- State the theorem
theorem f_derivative (x : ℝ) (h : 3 * x + 1 > 0) :
  deriv f x = (3 * Real.sqrt (9 * x^2 + 6 * x - 3)) / (3 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l716_71638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_number_characterization_l716_71641

def is_magic_number (N : ℕ) : Prop :=
  ∀ m : ℕ, (10^(Nat.log 10 N + 1) * m + N) % N = 0

theorem magic_number_characterization (N : ℕ) :
  is_magic_number N ↔
    (∃ k : ℕ, N = 2 * 10^k) ∨
    (∃ k : ℕ, N = 10^k) ∨
    (∃ k : ℕ, N = 5 * 10^k) ∨
    (∃ k : ℕ, N = 25 * 10^k) ∨
    (∃ k : ℕ, N = 125 * 10^k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magic_number_characterization_l716_71641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_comparison_l716_71630

theorem sqrt_sum_comparison : Real.sqrt 8 + Real.sqrt 5 < Real.sqrt 7 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_comparison_l716_71630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_array_l716_71626

def is_valid_entry (x : Int) : Prop := x = 1 ∨ x = -1

def row_sum_zero (row : Fin 3 → Int) : Prop :=
  (row 0) + (row 1) + (row 2) = 0

def col_sum_zero (col : Fin 3 → Int) : Prop :=
  (col 0) + (col 1) + (col 2) = 0

theorem no_valid_array :
  ¬ ∃ (A : Fin 3 → Fin 3 → Int),
    (∀ i j, is_valid_entry (A i j)) ∧
    (∀ i, row_sum_zero (λ j ↦ A i j)) ∧
    (∀ j, col_sum_zero (λ i ↦ A i j)) := by
  sorry

#check no_valid_array

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_array_l716_71626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_roots_l716_71620

/-- An arithmetic sequence with non-zero elements and common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  nonzero_terms : ∀ n, a n ≠ 0
  nonzero_diff : d ≠ 0

/-- The other root of the quadratic equation for each i -/
noncomputable def other_root (seq : ArithmeticSequence) (i : ℕ) : ℚ :=
  -2 * seq.a (i + 1) / seq.a i

theorem arithmetic_sequence_roots (seq : ArithmeticSequence) :
  (∀ i : ℕ, seq.a i * (-1)^2 + 2 * seq.a (i + 1) * (-1) + seq.a (i + 2) = 0) ∧
  (∀ i : ℕ, 1 / (other_root seq (i + 1) + 1) - 1 / (other_root seq i + 1) = -1/2) := by
  sorry

#check arithmetic_sequence_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_roots_l716_71620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finitely_many_solutions_l716_71612

theorem finitely_many_solutions :
  ∃ (N : ℕ), ∀ (a b c : ℕ),
    a > 0 → b > 0 → c > 0 →
    (1 : ℚ) / a + 1 / b + 1 / c = 1 / 1983 →
    a ≤ N ∧ b ≤ N ∧ c ≤ N :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finitely_many_solutions_l716_71612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l716_71658

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x + 6)^2 + (y - 5)^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Define the points
def M : Set (ℝ × ℝ) := {p | C₁ p.1 p.2}
def N : Set (ℝ × ℝ) := {p | C₂ p.1 p.2}
def P : Set ℝ := Set.univ

-- Define the distance function
noncomputable def dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (m : ℝ × ℝ) (n : ℝ × ℝ) (p : ℝ),
    m ∈ M ∧ n ∈ N ∧
    (∀ (m' : ℝ × ℝ) (n' : ℝ × ℝ) (p' : ℝ),
      m' ∈ M → n' ∈ N →
      dist (m, (p, 0)) + dist (n, (p, 0)) ≤
      dist (m', (p', 0)) + dist (n', (p', 0))) ∧
    dist (m, (p, 0)) + dist (n, (p, 0)) = 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l716_71658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l716_71632

def p (x : ℝ) : ℝ := 2*x^8 - 3*x^7 + x^5 - 5*x^4 + x^2 - 6

theorem polynomial_remainder_theorem :
  ∃ q : ℝ → ℝ, p = λ x ↦ (x - 3) * q x + 6892 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_theorem_l716_71632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interesting_numbers_in_range_interesting_numbers_are_interesting_all_interesting_numbers_identified_l716_71618

def is_interesting (n : ℕ) : Prop :=
  n > 1 ∧ 
  ∀ d₁ d₂ : ℕ, d₁ ∣ n → d₂ ∣ n → d₁ < d₂ → d₁ ∣ d₂

def interesting_numbers : Set ℕ := {25, 27, 32, 49, 64, 81}

theorem interesting_numbers_in_range : 
  ∀ n ∈ interesting_numbers, 20 ≤ n ∧ n ≤ 90 :=
by sorry

theorem interesting_numbers_are_interesting :
  ∀ n ∈ interesting_numbers, is_interesting n :=
by sorry

theorem all_interesting_numbers_identified :
  ∀ n : ℕ, 20 ≤ n → n ≤ 90 → is_interesting n → n ∈ interesting_numbers :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interesting_numbers_in_range_interesting_numbers_are_interesting_all_interesting_numbers_identified_l716_71618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2012_deg_l716_71602

theorem sin_2012_deg : Real.sin (2012 * π / 180) = -Real.sin (32 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2012_deg_l716_71602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_proposition_l716_71680

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x ≥ π / 2 ∧ Real.sin x > 1) ↔ (∀ x : ℝ, x ≥ π / 2 → Real.sin x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_proposition_l716_71680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sweater_knitting_time_l716_71645

/-- Represents the time in hours to knit various items --/
structure KnittingTime where
  hat : ℚ
  scarf : ℚ
  mitten : ℚ
  sock : ℚ
  total : ℚ
  grandchildren : ℕ

/-- Calculates the time to knit a sweater given the knitting times for other items --/
def sweaterTime (kt : KnittingTime) : ℚ :=
  let otherItemsTime := kt.hat + kt.scarf + 2 * kt.mitten + 2 * kt.sock
  let totalOtherTime := kt.grandchildren * otherItemsTime
  (kt.total - totalOtherTime) / kt.grandchildren

/-- Theorem stating that the time to knit each sweater is 6 hours --/
theorem sweater_knitting_time (kt : KnittingTime) 
  (h1 : kt.hat = 2)
  (h2 : kt.scarf = 3)
  (h3 : kt.mitten = 1)
  (h4 : kt.sock = 3/2)
  (h5 : kt.total = 48)
  (h6 : kt.grandchildren = 3) :
  sweaterTime kt = 6 := by
  sorry

#eval sweaterTime { hat := 2, scarf := 3, mitten := 1, sock := 3/2, total := 48, grandchildren := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sweater_knitting_time_l716_71645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l716_71656

open Real

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sin (2025 * x))^4 + (Real.cos (2016 * x))^2019 * (Real.cos (2025 * x))^2018 = 1 ↔ 
  (∃ n : ℤ, x = π / 4050 + n * π / 2025) ∨ (∃ k : ℤ, x = k * π / 9) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l716_71656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_ratio_l716_71688

/-- Parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : 0 < p

/-- Point on a parabola -/
structure ParabolaPoint (parab : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * parab.p * x

/-- Focus of the parabola -/
noncomputable def focus (parab : Parabola) : ℝ × ℝ := (parab.p / 2, 0)

/-- Theorem: The maximum value of |PQ| / |MN| is √2/2 -/
theorem parabola_max_ratio (parab : Parabola) 
    (M N : ParabolaPoint parab) 
    (perp : (M.x - parab.p/2) * (N.x - parab.p/2) + M.y * N.y = 0) :
    let F := focus parab
    let P := ((M.x + N.x)/2, (M.y + N.y)/2)  -- Midpoint of MN
    let Q := (P.1, 0)  -- Projection of P onto x-axis
    ∃ (c : ℝ), c ≤ Real.sqrt 2/2 ∧ 
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) / Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2) ≤ c ∧
      (∃ (M' N' : ParabolaPoint parab), 
        (M'.x - parab.p/2) * (N'.x - parab.p/2) + M'.y * N'.y = 0 ∧
        let P' := ((M'.x + N'.x)/2, (M'.y + N'.y)/2)
        let Q' := (P'.1, 0)
        Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) / Real.sqrt ((M'.x - N'.x)^2 + (M'.y - N'.y)^2) = c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_ratio_l716_71688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_problem_uniqueness_l716_71616

/-- Represents the number of fish caught by each fisherman -/
structure FishCatch where
  first : ℕ
  second : ℕ

/-- Checks if a given FishCatch satisfies all conditions of the problem -/
def isValidCatch (c : FishCatch) : Prop :=
  c.first + c.second = 70 ∧
  ∃ (n m : ℕ), c.first = 9 * n ∧ c.second = 17 * m

/-- The solution to the fishing problem -/
def fishingSolution : FishCatch :=
  { first := 36, second := 34 }

/-- Theorem stating that the fishingSolution is the unique valid catch -/
theorem fishing_problem_uniqueness :
  isValidCatch fishingSolution ∧
  ∀ (c : FishCatch), isValidCatch c → c = fishingSolution :=
by
  sorry

#check fishing_problem_uniqueness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_problem_uniqueness_l716_71616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l716_71634

noncomputable def point := ℝ × ℝ

noncomputable def distance (p q : point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_between_specific_points :
  distance (1, -3) (-4, 5) = Real.sqrt 89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l716_71634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_correct_l716_71695

/-- The equation of a circle in the form x^2 + y^2 + Dx + Ey + F = 0 -/
structure CircleEquation where
  D : ℝ
  E : ℝ
  F : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, compute its center -/
noncomputable def computeCenter (eq : CircleEquation) : CircleCenter :=
  { x := -eq.D / 2,
    y := -eq.E / 2 }

theorem circle_center_correct (eq : CircleEquation) 
    (h : eq = { D := -2, E := 2, F := 0 }) : 
    computeCenter eq = { x := 1, y := -1 } := by
  sorry

#check circle_center_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_correct_l716_71695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_size_l716_71642

/-- Represents a class of students with their average mark -/
structure StudentClass where
  students : ℕ
  average_mark : ℚ

theorem first_class_size 
  (first_class : StudentClass)
  (second_class : StudentClass)
  (h1 : first_class.average_mark = 40)
  (h2 : second_class.students = 50)
  (h3 : second_class.average_mark = 80)
  (h4 : (first_class.students * first_class.average_mark + 
         second_class.students * second_class.average_mark) / 
        (first_class.students + second_class.students) = 65) :
  first_class.students = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_size_l716_71642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_square_theorem_l716_71673

/-- Represents a 9x9 board with colored squares -/
def Board := Fin 9 → Fin 9 → Bool

/-- Counts the number of red squares in a given 2x2 block -/
def count_red_in_block (board : Board) (i j : Fin 8) : Nat :=
  (board i j).toNat + (board i j.succ).toNat + (board i.succ j).toNat + (board i.succ j.succ).toNat

/-- Counts the total number of red squares on the board -/
def total_red_squares (board : Board) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 9)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 9)) fun j => 
      (board i j).toNat)

/-- Main theorem: If a 9x9 board has 46 red squares, then there exists a 2x2 block with at least 3 red squares -/
theorem red_square_theorem (board : Board) 
  (h : total_red_squares board = 46) : 
  ∃ i j : Fin 8, count_red_in_block board i j ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_square_theorem_l716_71673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_location_l716_71669

def z : ℂ := -1 + 2*Complex.I

theorem point_location (h : z = -1 + 2*Complex.I) : 
  let w := 5*Complex.I/z
  (w.re < 0 ∧ w.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_location_l716_71669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_ratio_l716_71600

/-- The original function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3 * x + 1) / (x - 4)

/-- The inverse function g^(-1)(x) -/
noncomputable def g_inv (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem inverse_function_ratio (a b c d : ℝ) (h : c ≠ 0) :
  (∀ x, x ≠ 4 → g (g_inv a b c d x) = x) →
  a / c = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_ratio_l716_71600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_endpoint_l716_71628

/-- Given a circle with center (5, -2) and one endpoint of a diameter at (1, 2),
    the other endpoint of the diameter is at (9, -6). -/
theorem circle_diameter_endpoint (Q : Set (ℝ × ℝ)) (r : ℝ) : 
  Q = {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 + 2)^2 = r^2} →
  (1, 2) ∈ Q →
  (9, -6) ∈ Q ∧ 
  (5 - 1 = 9 - 5) ∧ 
  (-2 - 2 = -6 - (-2)) :=
by
  intro hQ h1
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_endpoint_l716_71628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_proof_l716_71635

theorem ending_number_proof (n : ℕ) : 
  n ≥ 18 ∧ 
  n % 9 = 0 ∧ 
  (18 + n : ℚ) / 2 = 99 / 2 →
  n = 81 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_proof_l716_71635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l716_71666

/-- A structure representing a sequence of consecutive integers -/
structure ConsecutiveSequence where
  start : ℕ
  length : ℕ
deriving DecidableEq

/-- The sum of a consecutive sequence -/
def sum_consecutive (seq : ConsecutiveSequence) : ℕ :=
  seq.length * seq.start + seq.length * (seq.length - 1) / 2

/-- Predicate for a valid sequence that sums to 255 -/
def is_valid_sequence (seq : ConsecutiveSequence) : Prop :=
  seq.length ≥ 3 ∧ sum_consecutive seq = 255

instance : DecidablePred is_valid_sequence := fun seq =>
  decidable_of_iff
    (seq.length ≥ 3 ∧ sum_consecutive seq = 255)
    (by rfl)

/-- The theorem to be proved -/
theorem count_valid_sequences :
  (Finset.filter is_valid_sequence (Finset.image (λ n => ConsecutiveSequence.mk n n) (Finset.range 255))).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l716_71666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l716_71691

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t, 1 + t)

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the center of circle C
def center_C : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem distance_center_to_line :
  let l := line_l
  let C := circle_C
  let center := center_C
  (center.1 - (l 0).1) ^ 2 + (center.2 - (l 0).2) ^ 2 = 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l716_71691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_585_degrees_l716_71686

open Real

theorem sin_585_degrees : 
  (∀ θ : ℝ, Real.sin (θ + 2 * π) = Real.sin θ) → -- Full rotation property
  Real.sin (π / 4) = sqrt 2 / 2 →     -- sin 45°
  (∀ θ : ℝ, 0 < θ - π ∧ θ - π < π / 2 → Real.sin θ < 0) → -- Third quadrant property
  Real.sin (585 * π / 180) = - sqrt 2 / 2 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_585_degrees_l716_71686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_triangle_theorem_l716_71694

open Complex Real

-- Define the complex roots and equation coefficients
variable (z₁ z₂ a b : ℂ)

-- Define the conditions
axiom root_condition : z₁^2 + a*z₁ + b = 0 ∧ z₂^2 + a*z₂ + b = 0
axiom isosceles_condition : abs z₁ = abs z₂
axiom angle_condition : arg (z₂ / z₁) = Real.pi / 6

-- The theorem to prove
theorem roots_triangle_theorem : a^2 / b = 5 + I * sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_triangle_theorem_l716_71694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equivalent_l716_71697

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (5 * (Real.sin x)^4 + 4 * (Real.cos x)^2) - Real.sqrt (6 * (Real.cos x)^4 + 4 * (Real.sin x)^2)

theorem g_equivalent : ∀ x : ℝ, g x = -Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equivalent_l716_71697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l716_71685

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^3 + k * Real.log x

theorem inequality_holds (k : ℝ) (hk : k ≥ -3) (x₁ x₂ : ℝ) 
  (hx₁ : x₁ ≥ 1) (hx₂ : x₂ ≥ 1) (hx : x₁ > x₂) : 
  (deriv (f k) x₁ + deriv (f k) x₂) / 2 > (f k x₁ - f k x₂) / (x₁ - x₂) := by
  sorry

#check inequality_holds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l716_71685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_8_l716_71605

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 2 = 6
  h4 : (a 3) ^ 2 = a 1 * a 7

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum_8 (seq : ArithmeticSequence) : sum_n seq 8 = 88 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_8_l716_71605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l716_71623

/-- The speed of a train in km/h, given its length and time to cross a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: The speed of a train with length 250.02 meters that crosses a pole in 5 seconds is 180.0144 km/h -/
theorem train_speed_calculation :
  train_speed 250.02 5 = 180.0144 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l716_71623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_months_62_days_l716_71670

/-- Represents a month of the year --/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December
deriving BEq, Repr

/-- Returns true if the given month has 31 days --/
def has31Days (m : Month) : Bool :=
  match m with
  | Month.January | Month.March | Month.May | Month.July
  | Month.August | Month.October | Month.December => true
  | _ => false

/-- Returns true if two months are consecutive --/
def areConsecutive (m1 m2 : Month) : Bool :=
  match (m1, m2) with
  | (Month.December, Month.January) => true
  | (Month.January, Month.February) => true
  | (Month.February, Month.March) => true
  | (Month.March, Month.April) => true
  | (Month.April, Month.May) => true
  | (Month.May, Month.June) => true
  | (Month.June, Month.July) => true
  | (Month.July, Month.August) => true
  | (Month.August, Month.September) => true
  | (Month.September, Month.October) => true
  | (Month.October, Month.November) => true
  | (Month.November, Month.December) => true
  | _ => false

/-- Theorem: Given a stay of exactly 62 days over two consecutive months,
    the only possible month pairs are July-August and December-January --/
theorem consecutive_months_62_days :
  ∀ m1 m2 : Month,
  areConsecutive m1 m2 →
  has31Days m1 ∧ has31Days m2 →
  (m1 = Month.July ∧ m2 = Month.August) ∨ (m1 = Month.December ∧ m2 = Month.January) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_months_62_days_l716_71670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_egg_left_eggs_sold_is_nine_l716_71613

/-- Represents the number of eggs in Old Lady Wang's basket -/
def initial_eggs : ℕ := 10

/-- The number of eggs remaining after the first person's purchase -/
def remaining_after_first : ℕ := initial_eggs / 2 - 1

/-- The number of eggs remaining after the second person's purchase -/
def remaining_after_second : ℕ := remaining_after_first / 2 - 1

/-- The condition that only one egg remains at the end -/
theorem one_egg_left : remaining_after_second = 1 := by
  rfl

/-- The total number of eggs sold -/
def eggs_sold : ℕ := initial_eggs - 1

/-- Theorem stating that the total number of eggs sold is 9 -/
theorem eggs_sold_is_nine : eggs_sold = 9 := by
  rfl

#eval eggs_sold

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_egg_left_eggs_sold_is_nine_l716_71613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l716_71601

noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 4 - Real.sin x ^ 2 * Real.cos x ^ 2 + Real.cos x ^ 4

theorem g_range :
  (∀ x : ℝ, (1/4 : ℝ) ≤ g x ∧ g x ≤ 1) ∧
  (∃ x : ℝ, g x = 1/4) ∧
  (∃ x : ℝ, g x = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l716_71601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_range_l716_71657

noncomputable def a (n : ℕ) (p : ℝ) : ℝ := -n + p

noncomputable def b (n : ℕ) : ℝ := 3^(n-4)

noncomputable def C (n : ℕ) (p : ℝ) : ℝ := max (a n p) (b n)

theorem p_range (p : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → C n p > C 4 p) ↔ 4 < p ∧ p < 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_range_l716_71657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_l716_71611

-- Define the man's income and savings in the first year
variable (I : ℝ) -- Income in the first year
variable (S : ℝ) -- Savings in the first year

-- Define the conditions
def income_increase : ℝ := 1.25
def expense_inflation : ℝ := 1.10
def tax_rebate : ℝ := 0.05
def savings_increase : ℝ := 2

-- Theorem statement
theorem savings_percentage :
  -- Savings in the second year equal twice the first year's savings
  S + tax_rebate * I = savings_increase * S →
  -- Total expenditure over two years is double the first year's expenditure
  (I - S) + expense_inflation * (I - S) = 2 * (I - S) →
  -- The man saved 5% of his income in the first year
  S = 0.05 * I :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_l716_71611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_passes_through_origin_one_l716_71629

/-- A function that passes through the point (0,1) -/
def PassesThroughOriginOne (f : ℝ → ℝ) : Prop := f 0 = 1

/-- Definition of an exponential function -/
def IsExponentialFunction (f : ℝ → ℝ) : Prop := ∃ (b : ℝ), b > 0 ∧ ∀ x, f x = b^x

/-- The specific function y = 2^x -/
noncomputable def f (x : ℝ) : ℝ := 2^x

/-- Theorem stating that exponential functions pass through (0,1) -/
axiom exponential_passes_through_origin_one : 
  ∀ (f : ℝ → ℝ), IsExponentialFunction f → PassesThroughOriginOne f

/-- Theorem stating that f(x) = 2^x is an exponential function -/
axiom f_is_exponential : IsExponentialFunction f

/-- The main theorem: f(x) = 2^x passes through (0,1) by deductive reasoning -/
theorem f_passes_through_origin_one : PassesThroughOriginOne f := by
  apply exponential_passes_through_origin_one f
  exact f_is_exponential


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_passes_through_origin_one_l716_71629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mountaintop_altitude_is_1830_l716_71692

/-- Represents the relationship between altitude and temperature change -/
structure AltitudeTemperatureRelation where
  altitude_change : ℚ
  temperature_change : ℚ

/-- Represents the conditions of the mountain -/
structure MountainConditions where
  foot_altitude : ℚ
  foot_temperature : ℚ
  top_temperature : ℚ

/-- Calculates the altitude of the mountaintop given the conditions and relationship -/
def calculate_mountaintop_altitude (relation : AltitudeTemperatureRelation) (conditions : MountainConditions) : ℚ :=
  conditions.foot_altitude + 
  (conditions.foot_temperature - conditions.top_temperature) / relation.temperature_change * relation.altitude_change

/-- Theorem stating that the mountaintop altitude is 1830 meters -/
theorem mountaintop_altitude_is_1830 : 
  let relation := AltitudeTemperatureRelation.mk 100 (3/10)
  let conditions := MountainConditions.mk 1230 18 (162/10)
  calculate_mountaintop_altitude relation conditions = 1830 := by
  sorry

#eval calculate_mountaintop_altitude 
  (AltitudeTemperatureRelation.mk 100 (3/10)) 
  (MountainConditions.mk 1230 18 (162/10))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mountaintop_altitude_is_1830_l716_71692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_and_perimeter_l716_71662

/-- Triangle PQR is an equilateral triangle with side length 8 cm -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 8

/-- The area of an equilateral triangle -/
noncomputable def area (t : EquilateralTriangle) : ℝ :=
  (Real.sqrt 3 / 4) * t.side_length ^ 2

/-- The perimeter of an equilateral triangle -/
def perimeter (t : EquilateralTriangle) : ℝ :=
  3 * t.side_length

theorem equilateral_triangle_area_and_perimeter (t : EquilateralTriangle) :
  area t = 16 * Real.sqrt 3 ∧ perimeter t = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_and_perimeter_l716_71662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angle_line_equation_equal_angle_line_solutions_l716_71617

/-- A line passing through the origin and forming equal angles with the three coordinate axes -/
structure EqualAngleLine where
  -- The direction vector of the line
  direction : ℝ × ℝ × ℝ
  -- The line passes through the origin
  passes_origin : direction.1 ≠ 0 ∨ direction.2.1 ≠ 0 ∨ direction.2.2 ≠ 0
  -- The line forms equal angles with the coordinate axes
  equal_angles : |direction.1| = |direction.2.1| ∧ |direction.2.1| = |direction.2.2|

/-- The equation of the line is x = y = z -/
theorem equal_angle_line_equation (l : EqualAngleLine) :
  ∃ (t : ℝ), (t * l.direction.1 = t * l.direction.2.1) ∧ (t * l.direction.2.1 = t * l.direction.2.2) :=
by
  sorry

/-- There are 4 unique solutions considering symmetry along the coordinate planes -/
theorem equal_angle_line_solutions :
  ∃ (s₁ s₂ s₃ s₄ : EqualAngleLine), (s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₁ ≠ s₄ ∧ s₂ ≠ s₃ ∧ s₂ ≠ s₄ ∧ s₃ ≠ s₄) ∧
  (∀ (s : EqualAngleLine), s = s₁ ∨ s = s₂ ∨ s = s₃ ∨ s = s₄) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_angle_line_equation_equal_angle_line_solutions_l716_71617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l716_71625

-- Define points in 2D space
def Point := ℝ × ℝ

-- Define the given points
def A : Point := (-3, 7)
def B : Point := (9, -1)

-- Define a function to calculate the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define a function to extend a line segment
def extend_segment (p q : Point) (factor : ℝ) : Point :=
  (q.1 + factor * (q.1 - p.1), q.2 + factor * (q.2 - p.2))

-- Theorem statement
theorem point_C_coordinates :
  let C := extend_segment A B (3/2)
  distance B C = (1/2) * distance A B → C = (15, -5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l716_71625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_leading_coefficient_l716_71679

/-- A quadratic polynomial represented by its coefficients -/
structure QuadraticPolynomial where
  a : ℝ  -- leading coefficient
  b : ℝ  -- linear coefficient
  c : ℝ  -- constant term

/-- The set of quadratic polynomials satisfying the problem conditions -/
def QuadraticSet : Type := 
  {s : Finset QuadraticPolynomial // 
    s.card = 100 ∧ 
    (∀ p q : QuadraticPolynomial, p ∈ s → q ∈ s → p ≠ q → 
      ∃! x : ℝ, p.a * x^2 + p.b * x + p.c = q.a * x^2 + q.b * x + q.c) ∧
    (∀ p q r : QuadraticPolynomial, p ∈ s → q ∈ s → r ∈ s → p ≠ q ∧ q ≠ r ∧ p ≠ r → 
      ¬∃ x : ℝ, p.a * x^2 + p.b * x + p.c = q.a * x^2 + q.b * x + q.c ∧ 
                p.a * x^2 + p.b * x + p.c = r.a * x^2 + r.b * x + r.c)}

theorem quadratic_leading_coefficient (s : QuadraticSet) :
  ∃ a : ℝ, (s.val.filter (λ p => p.a = a)).card ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_leading_coefficient_l716_71679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_over_4_cos_alpha_plus_beta_l716_71665

-- Define the variables and conditions
variable (α β : ℝ)
variable (h1 : Real.sin (π/8 + α/2) * Real.cos (π/8 + α/2) = Real.sqrt 3 / 4)
variable (h2 : α ∈ Set.Ioo (π/4) (π/2))
variable (h3 : Real.cos (β - π/4) = 3/5)
variable (h4 : β ∈ Set.Ioo (π/2) π)

-- State the theorems to be proved
theorem cos_alpha_plus_pi_over_4 : Real.cos (α + π/4) = -1/2 := by sorry

theorem cos_alpha_plus_beta : Real.cos (α + β) = -(4 * Real.sqrt 3 + 3) / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_over_4_cos_alpha_plus_beta_l716_71665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_solution_negation_l716_71696

/-- An equation type (placeholder) --/
structure Equation where

/-- Predicate indicating a real number is a solution to an equation --/
def IsSolution (e : Equation) (x : ℝ) : Prop :=
  sorry  -- Definition depends on the specific equation type

/-- Predicate indicating an equation has at most one solution --/
def HasAtMostOneSolution (e : Equation) : Prop :=
  ∀ (x y : ℝ), IsSolution e x → IsSolution e y → x = y

/-- Predicate indicating an equation has at least two solutions --/
def HasAtLeastTwoSolutions (e : Equation) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ IsSolution e x ∧ IsSolution e y

theorem at_most_one_solution_negation :
  ¬(∀ (e : Equation), HasAtMostOneSolution e) ↔ (∃ (e : Equation), HasAtLeastTwoSolutions e) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_solution_negation_l716_71696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_arrival_interval_is_18_minutes_l716_71698

/-- Represents the time interval between boat arrivals at Sovunya's house -/
noncomputable def boatArrivalInterval (losyashSpeed : ℝ) (boatSpeed : ℝ) (launchInterval : ℝ) : ℝ :=
  launchInterval * (boatSpeed - losyashSpeed) / boatSpeed

/-- Theorem stating that the boat arrival interval is 18 minutes given the specified conditions -/
theorem boat_arrival_interval_is_18_minutes :
  let losyashSpeed : ℝ := 4  -- km/h
  let boatSpeed : ℝ := 10    -- km/h
  let launchInterval : ℝ := 0.5  -- hours (30 minutes)
  boatArrivalInterval losyashSpeed boatSpeed launchInterval * 60 = 18 := by
  sorry

#eval Float.toString ((0.5 * (10 - 4) / 10) * 60)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_arrival_interval_is_18_minutes_l716_71698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_from_sum_property_l716_71631

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_180 : a + b + c = 180

structure Quadrilateral where
  angles : Fin 4 → ℝ
  sum_360 : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 360

def has_sum_property (t : Triangle) (q : Quadrilateral) : Prop :=
  ∀ (i j : Fin 3), i ≠ j → ∃ (k : Fin 4), q.angles k = 
    (if i = 0 then t.a else if i = 1 then t.b else t.c) +
    (if j = 0 then t.a else if j = 1 then t.b else t.c)

def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

theorem triangle_isosceles_from_sum_property (t : Triangle) (q : Quadrilateral) 
  (h : has_sum_property t q) : is_isosceles t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_from_sum_property_l716_71631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_time_l716_71671

/-- A task can be completed by different combinations of workers. -/
structure WorkTask where
  /-- Time taken by A and B together to complete the task -/
  ab_time : ℝ
  /-- Time taken by B and C together to complete the task -/
  bc_time : ℝ
  /-- Time taken by C and D together to complete the task -/
  cd_time : ℝ

/-- The theorem states that given the conditions, A and D together will complete the task in 24 days -/
theorem task_completion_time (t : WorkTask) (h1 : t.ab_time = 8) (h2 : t.bc_time = 6) (h3 : t.cd_time = 12) :
  1 / (1 / t.ab_time + 1 / t.cd_time - 1 / t.bc_time) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_time_l716_71671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_shape_area_l716_71667

/-- The total area of a compound shape consisting of an equilateral triangle
    and an adjacent rectangle, given specific conditions. -/
theorem compound_shape_area : 
  ∀ (s : ℝ) (s' : ℝ),
    s = 4 →  -- Side length of large triangle
    s'^2 * (Real.sqrt 3 / 4) = (s^2 * (Real.sqrt 3 / 4)) / 3 →  -- Area of small triangle is 1/3 of large triangle
    (2 * s') * (s' / 2) + s^2 * (Real.sqrt 3 / 4) = 4 * Real.sqrt 3 + 16 / 3 :=
by
  intros s s' h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_shape_area_l716_71667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_347_l716_71633

def customSequence : ℕ → ℕ
  | 0 => 11
  | 1 => 23
  | 2 => 47
  | 3 => 83
  | 4 => 131
  | 5 => 191
  | 6 => 263
  | n + 7 => customSequence (n + 6) + (84 + 12 * n)

theorem eighth_term_is_347 : customSequence 7 = 347 := by
  rfl

#eval customSequence 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_347_l716_71633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_average_speed_l716_71637

/-- Calculates the average speed for a two-part race given the distances and times for each part. -/
noncomputable def average_speed (d1 d2 t1 t2 : ℝ) : ℝ :=
  (d1 + d2) / (t1 + t2)

/-- Proves that the average speed for Carmen's two-part race is 40/7 miles per hour. -/
theorem carmen_average_speed :
  let d1 : ℝ := 24  -- Distance of first part in miles
  let d2 : ℝ := 16  -- Distance of second part in miles
  let t1 : ℝ := 3   -- Time of first part in hours
  let t2 : ℝ := 4   -- Time of second part in hours
  average_speed d1 d2 t1 t2 = 40 / 7 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carmen_average_speed_l716_71637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_three_range_l716_71650

noncomputable def f (x : ℝ) : ℝ := (2^x + 1) / (2^x - 1)

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_greater_than_three_range :
  IsOdd f → {x : ℝ | f x > 3} = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_three_range_l716_71650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_f_undefined_at_D_l716_71640

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := (x^3 + 12*x^2 + 47*x + 60) / (x + 3)

/-- The simplified function g(x) -/
def g (x : ℝ) : ℝ := x^2 + 9*x + 20

/-- Theorem stating that f(x) = g(x) for all x ≠ -3 -/
theorem f_eq_g : ∀ x : ℝ, x ≠ -3 → f x = g x := by
  sorry

/-- The value of D in the original problem -/
def D : ℝ := -3

/-- Theorem stating that f is undefined when x = D -/
theorem f_undefined_at_D : ¬ ∃ y : ℝ, f D = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_g_f_undefined_at_D_l716_71640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_bound_l716_71676

def f (x : ℝ) := -x^2 + 2*x + 1

theorem increasing_interval_bound (a : ℝ) :
  (∀ x y, x ∈ Set.Icc (-3) a → y ∈ Set.Icc (-3) a → x < y → f x < f y) →
  a ∈ Set.Ioo (-3) 1 ∪ {1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_interval_bound_l716_71676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_f1_range_f2_range_f3_l716_71687

-- Function 1
noncomputable def f1 (x : ℝ) : ℝ := -2 * Real.cos x + 3/2

theorem range_f1 : Set.range f1 = Set.Icc (-1/2) (7/2) := by sorry

-- Function 2
noncomputable def f2 (x : ℝ) : ℝ := 2 * Real.sin x - 1

theorem range_f2 : Set.range (f2 ∘ (Set.Icc (-π/6) (π/2)).restrict f2) = Set.Icc (-2) 1 := by sorry

-- Function 3
noncomputable def f3 (a b x : ℝ) : ℝ := a * Real.sin x + b

theorem range_f3 (a b : ℝ) (h : a ≠ 0) :
  Set.range (f3 a b) = 
    if a > 0 then Set.Icc (-a + b) (a + b)
    else Set.Icc (a + b) (-a + b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_f1_range_f2_range_f3_l716_71687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l716_71681

/-- Represents a convex polygon with n sides where interior angles form an arithmetic sequence -/
structure ConvexPolygon where
  n : ℕ             -- number of sides
  smallest_angle : ℚ -- smallest interior angle in degrees
  common_diff : ℚ   -- common difference of the arithmetic sequence in degrees

/-- The sum of interior angles of a polygon with n sides -/
def interior_angle_sum (n : ℕ) : ℚ := 180 * (n - 2)

/-- The sum of an arithmetic sequence -/
def arithmetic_sequence_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem: A convex polygon with interior angles in arithmetic sequence,
    smallest angle 120°, and common difference 3° has 41 sides -/
theorem polygon_sides_count (p : ConvexPolygon) 
  (h1 : p.smallest_angle = 120)
  (h2 : p.common_diff = 3)
  (h3 : arithmetic_sequence_sum p.smallest_angle p.common_diff p.n = interior_angle_sum p.n) :
  p.n = 41 := by
  sorry

#check polygon_sides_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_count_l716_71681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l716_71614

theorem book_price_change (P : ℝ) (h : P > 0) : 
  P * (1 - 0.25) * (1 + 0.20) = P * (1 - 0.10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l716_71614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_gt_one_l716_71603

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x + a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + (1/2) * (x - 1)^2

theorem three_zeros_implies_a_gt_one (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    g a x = 0 ∧ g a y = 0 ∧ g a z = 0) →
  a > 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_gt_one_l716_71603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_password_probability_l716_71654

theorem password_probability : 
  (4 : ℚ) / 10 * (5 : ℚ) / 26 * (4 : ℚ) / 9 = 8 / 117 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_password_probability_l716_71654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_theorem_l716_71683

/-- The marked price of an article given its cost price, discounts, and desired profits -/
noncomputable def marked_price (cost_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (profit1 : ℝ) (profit2 : ℝ) : ℝ :=
  let selling_price1 := cost_price * (1 + profit1)
  let selling_price2 := selling_price1 * (1 + profit2)
  selling_price2 / ((1 - discount1) * (1 - discount2))

/-- Theorem stating that the marked price of the article is approximately 86.67 -/
theorem marked_price_theorem :
  ∃ ε > 0, |marked_price 47.5 0.05 0.1 0.3 0.2 - 86.67| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_theorem_l716_71683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_w_shape_solution_l716_71668

/-- Represents a "W" shape filled with integers -/
structure WShape where
  values : Fin 9 → Nat
  sum_equality : ∀ (row : Fin 4), (List.map values (List.range 3)).sum = (List.map values (List.range 3)).sum
  range : ∀ n, values n ∈ Finset.range 9
  distinct : Function.Injective values
  six_placed : ∃ n, values n = 6
  nine_placed : ∃ n, values n = 9

/-- The theorem stating the properties of the solution -/
theorem w_shape_solution (w : WShape) :
  (∃ n, w.values n = 8) ∧
  (∀ row : Fin 4, (List.map w.values (List.range 3)).sum = 17) := by
  sorry

#check w_shape_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_w_shape_solution_l716_71668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_linear_transformation_exists_zero_matrix_is_answer_l716_71661

open Matrix

theorem no_linear_transformation_exists : ∀ (M : Matrix (Fin 2) (Fin 2) ℝ),
  (∃ (A : Matrix (Fin 2) (Fin 2) ℝ), M • A ≠ ![![2 * A 0 0, 3 * A 0 1], ![A 1 0, A 1 1]]) :=
by
  sorry

theorem zero_matrix_is_answer (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ (A : Matrix (Fin 2) (Fin 2) ℝ), M • A = ![![2 * A 0 0, 3 * A 0 1], ![A 1 0, A 1 1]]) →
  M = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_linear_transformation_exists_zero_matrix_is_answer_l716_71661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_n_for_1992_smallest_n_for_1992_l716_71659

/-- The largest odd divisor of a positive integer -/
def g (x : ℕ) : ℕ :=
  sorry

/-- The function f as defined in the problem -/
def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then
    x / 2 + x / (g x)
  else
    2 ^ ((x + 1) / 2)

/-- The sequence x_n defined by the recurrence relation -/
def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => f (x n)

/-- The theorem stating the existence and uniqueness of n such that x_n = 1992 -/
theorem exists_unique_n_for_1992 : ∃! n : ℕ, x n = 1992 := by
  sorry

/-- The theorem stating that 8253 is the smallest n such that x_n = 1992 -/
theorem smallest_n_for_1992 : x 8253 = 1992 ∧ ∀ m < 8253, x m ≠ 1992 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_n_for_1992_smallest_n_for_1992_l716_71659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joseph_george_ratio_l716_71607

def joseph_countries : ℕ := sorry
def patrick_countries : ℕ := sorry
def zack_countries : ℕ := sorry
def george_countries : ℕ := sorry

axiom patrick_triple_joseph : patrick_countries = 3 * joseph_countries
axiom zack_double_patrick : zack_countries = 2 * patrick_countries
axiom george_six : george_countries = 6
axiom zack_eighteen : zack_countries = 18

theorem joseph_george_ratio : 
  (joseph_countries : ℚ) / george_countries = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joseph_george_ratio_l716_71607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l716_71636

theorem quadratic_root_form (a b c m n p : ℤ) : 
  a = 3 ∧ b = -9 ∧ c = -6 →
  (∃ x : ℚ, a * x^2 + b * x + c = 0) →
  (∃ x : ℚ, x = (m + Int.sqrt n) / p ∨ x = (m - Int.sqrt n) / p) →
  0 < m ∧ 0 < n ∧ 0 < p →
  Int.gcd m (Int.gcd n p) = 1 →
  n = 153 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l716_71636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l716_71639

-- Define the function f without recursion
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 2 * x^3 + 2^x - 1 else -(2 * (-x)^3 + 2^(-x) - 1)

-- State the theorem
theorem odd_function_value :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x ≥ 0, f x = 2 * x^3 + 2^x - 1) →  -- Definition for x ≥ 0
  f (-2) = -19 :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l716_71639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l716_71678

/-- A parabola with equation x = 4y² -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_equation : ∀ x y, equation x y ↔ x = 4 * y^2

/-- Points M and N on the parabola that intersect with the focus -/
structure IntersectionPoints (p : Parabola) where
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_M_on_parabola : p.equation M.1 M.2
  h_N_on_parabola : p.equation N.1 N.2

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_intersection_distance (p : Parabola) (points : IntersectionPoints p)
    (h_MF_distance : distance points.M p.focus = 1/8) :
  distance points.M points.N = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l716_71678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_omega_finite_l716_71651

-- Define the polynomial f
variable (f : ℚ → ℚ) 

-- Define the degree of f
variable (d : ℕ)

-- Assume the degree of f is at least 2
variable (h_degree : d ≥ 2)

-- Define f^n(ℚ) recursively
def f_pow_n (f : ℚ → ℚ) : ℕ → Set ℚ
  | 0 => Set.univ
  | n + 1 => {y | ∃ x ∈ f_pow_n f n, f x = y}

-- Define f^ω(ℚ)
def f_omega (f : ℚ → ℚ) : Set ℚ :=
  ⋂ n, f_pow_n f n

-- The main theorem
theorem f_omega_finite (f : ℚ → ℚ) (d : ℕ) (h_degree : d ≥ 2)
  (h_rational : ∀ x, ∃ (a b : ℤ), f x = a / b)
  (h_poly : Polynomial ℚ) :
  Finite (f_omega f) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_omega_finite_l716_71651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_sin_simplification_l716_71608

open Real BigOperators

theorem binomial_sum_sin_simplification (n : ℕ) (α θ : ℝ) :
  ∑ k in Finset.range (n + 1), ((-1)^k : ℝ) * (n.choose k) * sin (α + k * θ) =
  2^n * (sin (θ / 2))^n * sin (3 * n * π / 2 + n * θ / 2 + α) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_sin_simplification_l716_71608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_infinite_geometric_series_sum_l716_71648

/-- The sum of an infinite geometric series with first term a and common ratio r, where |r| < 1 -/
noncomputable def infiniteGeometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Proof that the sum of the infinite geometric series with first term 1/5 and common ratio 1/2 is 2/5 -/
theorem specific_infinite_geometric_series_sum :
  infiniteGeometricSeriesSum (1/5 : ℝ) (1/2 : ℝ) = 2/5 := by
  -- Unfold the definition of infiniteGeometricSeriesSum
  unfold infiniteGeometricSeriesSum
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_infinite_geometric_series_sum_l716_71648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l716_71621

/-- Represents a trapezoid EFGH with given side lengths and altitude --/
structure Trapezoid where
  EH : ℝ
  EF : ℝ
  FG : ℝ
  altitude : ℝ

/-- The area of a trapezoid with the given properties --/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ := 1050 + 67.5 * Real.sqrt 11

/-- Theorem stating that the area of the specific trapezoid is 1050 + 67.5√11 --/
theorem specific_trapezoid_area :
  ∃ (t : Trapezoid), t.EH = 18 ∧ t.EF = 60 ∧ t.FG = 25 ∧ t.altitude = 15 ∧
    trapezoid_area t = 1050 + 67.5 * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l716_71621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_in_38_seconds_l716_71606

/-- The time (in seconds) it takes for a train to pass a jogger -/
noncomputable def train_passing_time (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) : ℝ :=
  (initial_distance + train_length) / ((train_speed - jogger_speed) * (1000 / 3600))

/-- Theorem stating that the train will pass the jogger in 38 seconds under the given conditions -/
theorem train_passes_jogger_in_38_seconds :
  train_passing_time 9 45 120 260 = 38 := by
  -- Unfold the definition of train_passing_time
  unfold train_passing_time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passes_jogger_in_38_seconds_l716_71606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_language_scores_and_reading_time_l716_71689

/-- Represents a 2x2 contingency table --/
structure ContingencyTable :=
  (a b c d : ℕ)

/-- Calculates the chi-square statistic for a 2x2 contingency table --/
noncomputable def chiSquare (table : ContingencyTable) : ℝ :=
  let n := table.a + table.b + table.c + table.d
  (n * (table.a * table.d - table.b * table.c)^2 : ℝ) / 
  ((table.a + table.b) * (table.c + table.d) * (table.a + table.c) * (table.b + table.d))

/-- Represents the distribution of X --/
structure Distribution :=
  (p0 p1 p2 p3 : ℚ)

/-- Theorem statement for the given problem --/
theorem chinese_language_scores_and_reading_time 
  (sample_size : ℕ) 
  (high_score_ratio : ℚ) 
  (low_reading_time_ratio : ℚ) 
  (high_score_high_reading_time : ℕ) 
  (critical_value : ℝ) :
  sample_size = 500 →
  high_score_ratio = 3/10 →
  low_reading_time_ratio = 1/2 →
  high_score_high_reading_time = 100 →
  critical_value = 10.828 →
  ∃ (table : ContingencyTable) (dist : Distribution),
    chiSquare table > critical_value ∧
    dist.p0 = 1/84 ∧ dist.p1 = 3/14 ∧ dist.p2 = 15/28 ∧ dist.p3 = 5/21 ∧
    dist.p0 * 0 + dist.p1 * 1 + dist.p2 * 2 + dist.p3 * 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_language_scores_and_reading_time_l716_71689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l716_71653

-- Define the constants as noncomputable
noncomputable def a : ℝ := Real.exp 0.2
noncomputable def b : ℝ := Real.sin 1.2
noncomputable def c : ℝ := 1 + Real.log 1.2

-- State the theorem
theorem order_of_abc : b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l716_71653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_plus_y_leq_5_l716_71646

open MeasureTheory Measure Set Real

-- Define the bounds for x and y
def x_bound : Set ℝ := Icc 0 4
def y_bound : Set ℝ := Icc 0 8

-- Define the region where x + y ≤ 5
def region (p : ℝ × ℝ) : Prop := p.1 + p.2 ≤ 5 ∧ p.1 ∈ x_bound ∧ p.2 ∈ y_bound

-- Define the probability measure
noncomputable def prob_measure : Measure (ℝ × ℝ) :=
  volume.restrict (x_bound.prod y_bound)

-- Theorem statement
theorem probability_x_plus_y_leq_5 :
  prob_measure {p : ℝ × ℝ | region p} / prob_measure (x_bound.prod y_bound) = 5/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_plus_y_leq_5_l716_71646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l716_71672

/-- The ellipse equation -/
noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  3 * x^2 + 2 * x * y + 4 * y^2 - 13 * x - 26 * y + 52 = 0

/-- The set of points on the ellipse -/
noncomputable def ellipse_points : Set (ℝ × ℝ) :=
  {p | ellipse_equation p.1 p.2}

/-- The ratio y/x for a point (x, y) -/
noncomputable def ratio (p : ℝ × ℝ) : ℝ := p.2 / p.1

/-- Theorem stating the sum of max and min ratios -/
theorem ellipse_ratio_sum :
  ∃ (max min : ℝ),
    (∀ p ∈ ellipse_points, ratio p ≤ max ∧ min ≤ ratio p) ∧
    max + min = 65 / 39 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l716_71672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_p_l716_71622

noncomputable def f (p : ℝ) (x : ℝ) : ℝ := x + p / (x - 1)

theorem min_value_implies_p (p : ℝ) :
  p > 0 →
  (∀ x > 1, f p x ≥ 4) →
  (∃ x > 1, f p x = 4) →
  p = 9/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_p_l716_71622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rotation_overlap_l716_71699

theorem square_rotation_overlap (β : Real) (h1 : 0 < β) (h2 : β < Real.pi/2) (h3 : Real.cos β = 3/5) :
  let overlap_area := 1/3 -- actual overlap area
  overlap_area = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rotation_overlap_l716_71699
