import Mathlib

namespace NUMINAMATH_CALUDE_parabola_vertex_l2636_263642

/-- The parabola defined by y = (x-1)^2 + 2 -/
def parabola (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = (x-1)^2 + 2 has coordinates (1, 2) -/
theorem parabola_vertex : 
  ∀ (x : ℝ), parabola x ≥ parabola (vertex.1) ∧ parabola (vertex.1) = vertex.2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2636_263642


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l2636_263634

theorem complex_modulus_sqrt_5 (z : ℂ) (x y : ℝ) :
  z = x + y * I →
  x / (1 - I) = 1 + y * I →
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_5_l2636_263634


namespace NUMINAMATH_CALUDE_average_temperature_of_three_cities_l2636_263620

/-- The average temperature of three cities given specific temperature relationships --/
theorem average_temperature_of_three_cities 
  (temp_new_york : ℝ)
  (temp_diff_miami_new_york : ℝ)
  (temp_diff_san_diego_miami : ℝ)
  (h1 : temp_new_york = 80)
  (h2 : temp_diff_miami_new_york = 10)
  (h3 : temp_diff_san_diego_miami = 25) :
  (temp_new_york + (temp_new_york + temp_diff_miami_new_york) + 
   (temp_new_york + temp_diff_miami_new_york + temp_diff_san_diego_miami)) / 3 = 95 := by
  sorry

#check average_temperature_of_three_cities

end NUMINAMATH_CALUDE_average_temperature_of_three_cities_l2636_263620


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2636_263669

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2636_263669


namespace NUMINAMATH_CALUDE_janet_sock_purchase_l2636_263673

theorem janet_sock_purchase : 
  ∀ (x y z : ℕ),
  -- Total number of pairs
  x + y + z = 18 →
  -- Total cost
  2*x + 5*y + 7*z = 60 →
  -- Exactly 3 pairs of $7 socks
  z = 3 →
  -- x represents the number of $2 socks
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_janet_sock_purchase_l2636_263673


namespace NUMINAMATH_CALUDE_find_k_l2636_263687

theorem find_k : ∃ k : ℝ, (64 / k = 4) ∧ (k = 16) := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2636_263687


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2636_263644

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 4 = 12 → a 1 + a 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2636_263644


namespace NUMINAMATH_CALUDE_specific_quilt_shaded_fraction_l2636_263630

/-- Represents a square quilt composed of unit squares -/
structure Quilt :=
  (total_squares : ℕ)
  (whole_squares : ℕ)
  (half_shaded_squares : ℕ)
  (fully_half_shaded_squares : ℕ)

/-- Calculates the shaded fraction of a quilt -/
def shaded_fraction (q : Quilt) : ℚ :=
  let shaded_area := (q.half_shaded_squares : ℚ) / 2 + (q.fully_half_shaded_squares : ℚ) / 2
  shaded_area / q.total_squares

/-- Theorem stating that for a specific quilt configuration, the shaded fraction is 2.5/9 -/
theorem specific_quilt_shaded_fraction :
  let q := Quilt.mk 9 4 1 4
  shaded_fraction q = 5/18 := by sorry

end NUMINAMATH_CALUDE_specific_quilt_shaded_fraction_l2636_263630


namespace NUMINAMATH_CALUDE_line_symmetry_l2636_263624

/-- The original line -/
def original_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- The line of symmetry -/
def symmetry_line (x : ℝ) : Prop := x = 1

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to the symmetry_line -/
theorem line_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  original_line x₁ y₁ →
  symmetric_line x₂ y₂ →
  ∃ (x_sym : ℝ),
    symmetry_line x_sym ∧
    x_sym - x₁ = x₂ - x_sym ∧
    y₁ = y₂ :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l2636_263624


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2636_263682

theorem complex_expression_simplification :
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2636_263682


namespace NUMINAMATH_CALUDE_six_grades_assignments_l2636_263655

/-- The number of ways to assign n grades, where grades are 2, 3, or 4, and no two consecutive 2s are allowed. -/
def gradeAssignments (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | n + 2 => 2 * gradeAssignments (n + 1) + 2 * gradeAssignments n

/-- The theorem stating that there are 448 ways to assign 6 grades under the given conditions. -/
theorem six_grades_assignments : gradeAssignments 6 = 448 := by
  sorry

end NUMINAMATH_CALUDE_six_grades_assignments_l2636_263655


namespace NUMINAMATH_CALUDE_young_bonnets_theorem_l2636_263621

/-- Calculates the number of bonnets Mrs. Young sends to each orphanage --/
def bonnets_per_orphanage : ℕ :=
  let monday := 10
  let tuesday_wednesday := 2 * monday
  let thursday := monday + 5
  let friday := thursday - 5
  let saturday := friday - 8
  let sunday := 3 * saturday
  let total := monday + tuesday_wednesday + thursday + friday + saturday + sunday
  total / 10

/-- Theorem stating that Mrs. Young sends 6 bonnets to each orphanage --/
theorem young_bonnets_theorem : bonnets_per_orphanage = 6 := by
  sorry

end NUMINAMATH_CALUDE_young_bonnets_theorem_l2636_263621


namespace NUMINAMATH_CALUDE_tangent_line_circle_l2636_263696

-- Define the set of real numbers m+n should belong to
def tangent_range : Set ℝ :=
  {x | x ≤ 2 - 2 * Real.sqrt 2 ∨ x ≥ 2 + 2 * Real.sqrt 2}

-- Define the condition for the line to be tangent to the circle
def is_tangent (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), ((m + 1) * x + (n + 1) * y - 2 = 0) ∧
                ((x - 1)^2 + (y - 1)^2 = 1)

-- Theorem statement
theorem tangent_line_circle (m n : ℝ) :
  is_tangent m n → (m + n) ∈ tangent_range := by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l2636_263696


namespace NUMINAMATH_CALUDE_race_outcomes_l2636_263662

/-- The number of participants in the race -/
def num_participants : Nat := 6

/-- The number of podium positions (1st, 2nd, 3rd) -/
def num_podium_positions : Nat := 3

/-- Represents whether a participant can finish in a specific position -/
def can_finish (participant : Nat) (position : Nat) : Prop :=
  ¬(participant = num_participants ∧ position = num_podium_positions)

/-- The number of valid race outcomes -/
def num_valid_outcomes : Nat := 120

theorem race_outcomes :
  (∀ (p₁ p₂ p₃ : Nat), p₁ ≤ num_participants → p₂ ≤ num_participants → p₃ ≤ num_participants →
    p₁ ≠ p₂ → p₁ ≠ p₃ → p₂ ≠ p₃ →
    can_finish p₁ 1 → can_finish p₂ 2 → can_finish p₃ 3 →
    ∃! (outcome : Nat), outcome = num_valid_outcomes) :=
by sorry

end NUMINAMATH_CALUDE_race_outcomes_l2636_263662


namespace NUMINAMATH_CALUDE_wangwa_smallest_growth_rate_l2636_263611

structure BreedingBase where
  name : String
  growthRate : Float

def liwa : BreedingBase := { name := "Liwa", growthRate := 3.25 }
def wangwa : BreedingBase := { name := "Wangwa", growthRate := -2.75 }
def jiazhuang : BreedingBase := { name := "Jiazhuang", growthRate := 4.6 }
def wuzhuang : BreedingBase := { name := "Wuzhuang", growthRate := -1.76 }

def breedingBases : List BreedingBase := [liwa, wangwa, jiazhuang, wuzhuang]

theorem wangwa_smallest_growth_rate :
  ∀ b ∈ breedingBases, wangwa.growthRate ≤ b.growthRate :=
by sorry

end NUMINAMATH_CALUDE_wangwa_smallest_growth_rate_l2636_263611


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l2636_263617

theorem smallest_whole_number_above_sum : 
  ⌈(3 + 1/7 : ℚ) + (4 + 1/8 : ℚ) + (5 + 1/9 : ℚ) + (6 + 1/10 : ℚ)⌉ = 19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l2636_263617


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2636_263606

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + 2*x + 3 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2636_263606


namespace NUMINAMATH_CALUDE_factorization_x_squared_plus_5x_l2636_263680

theorem factorization_x_squared_plus_5x (x : ℝ) : x^2 + 5*x = x*(x+5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_plus_5x_l2636_263680


namespace NUMINAMATH_CALUDE_ExistsSpecialSequence_l2636_263676

-- Define the sequence type
def InfiniteSequence := ℕ → ℕ

-- Define the properties of the sequence
def NoDivisibility (seq : InfiniteSequence) :=
  ∀ i j, i ≠ j → ¬(seq i ∣ seq j)

def CommonDivisorGreaterThanOne (seq : InfiniteSequence) :=
  ∀ i j, i ≠ j → ∃ k, k > 1 ∧ k ∣ seq i ∧ k ∣ seq j

def NoCommonDivisorGreaterThanOne (seq : InfiniteSequence) :=
  ¬∃ k, k > 1 ∧ (∀ i, k ∣ seq i)

-- Main theorem
theorem ExistsSpecialSequence :
  ∃ seq : InfiniteSequence,
    NoDivisibility seq ∧
    CommonDivisorGreaterThanOne seq ∧
    NoCommonDivisorGreaterThanOne seq :=
by sorry


end NUMINAMATH_CALUDE_ExistsSpecialSequence_l2636_263676


namespace NUMINAMATH_CALUDE_only_D_satisfies_all_preferences_l2636_263661

-- Define the set of movies
inductive Movie : Type
  | A | B | C | D | E

-- Define the preferences of each person
def xiao_zhao_preference (m : Movie) : Prop := m ≠ Movie.B
def xiao_zhang_preference (m : Movie) : Prop := m = Movie.B ∨ m = Movie.C ∨ m = Movie.D ∨ m = Movie.E
def xiao_li_preference (m : Movie) : Prop := m ≠ Movie.C
def xiao_liu_preference (m : Movie) : Prop := m ≠ Movie.E

-- Define a function that checks if a movie satisfies all preferences
def satisfies_all_preferences (m : Movie) : Prop :=
  xiao_zhao_preference m ∧
  xiao_zhang_preference m ∧
  xiao_li_preference m ∧
  xiao_liu_preference m

-- Theorem: D is the only movie that satisfies all preferences
theorem only_D_satisfies_all_preferences :
  ∀ m : Movie, satisfies_all_preferences m ↔ m = Movie.D :=
by sorry


end NUMINAMATH_CALUDE_only_D_satisfies_all_preferences_l2636_263661


namespace NUMINAMATH_CALUDE_ellipse_axis_sum_l2636_263692

/-- Proves that for an ellipse with given conditions, a + b = 40 -/
theorem ellipse_axis_sum (M N a b : ℝ) : 
  M > 0 → 
  N > 0 → 
  M = π * a * b → 
  N = π * (a + b) → 
  M / N = 10 → 
  a = b → 
  a + b = 40 := by
sorry

end NUMINAMATH_CALUDE_ellipse_axis_sum_l2636_263692


namespace NUMINAMATH_CALUDE_gcd_of_abcd_plus_dcba_l2636_263651

theorem gcd_of_abcd_plus_dcba :
  ∃ (g : ℕ), g > 1 ∧ 
  (∀ (a : ℕ), 0 ≤ a → a ≤ 3 → 
    g ∣ (1000 * a + 100 * (a + 2) + 10 * (a + 4) + (a + 6)) + 
        (1000 * (a + 6) + 100 * (a + 4) + 10 * (a + 2) + a)) ∧
  (∀ (d : ℕ), d > g → 
    ∃ (a : ℕ), 0 ≤ a ∧ a ≤ 3 ∧ 
      ¬(d ∣ (1000 * a + 100 * (a + 2) + 10 * (a + 4) + (a + 6)) + 
           (1000 * (a + 6) + 100 * (a + 4) + 10 * (a + 2) + a))) ∧
  g = 2 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_abcd_plus_dcba_l2636_263651


namespace NUMINAMATH_CALUDE_total_maggots_is_twenty_l2636_263656

/-- The number of maggots served in the first attempt -/
def first_attempt : ℕ := 10

/-- The number of maggots served in the second attempt -/
def second_attempt : ℕ := 10

/-- The total number of maggots served -/
def total_maggots : ℕ := first_attempt + second_attempt

theorem total_maggots_is_twenty : total_maggots = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_maggots_is_twenty_l2636_263656


namespace NUMINAMATH_CALUDE_polynomial_property_l2636_263639

def P (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem polynomial_property (a b c : ℝ) :
  P a b c 0 = 5 →
  (-a/3 : ℝ) = -c →
  (-a/3 : ℝ) = 1 + a + b + c →
  b = -26 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l2636_263639


namespace NUMINAMATH_CALUDE_profit_after_reduction_profit_for_target_l2636_263613

/-- Represents the daily sales and profit calculations for a product. -/
structure ProductSales where
  basePrice : ℝ
  baseSales : ℝ
  profitPerItem : ℝ
  salesIncreasePerYuan : ℝ

/-- Calculates the daily profit given a price reduction. -/
def dailyProfit (p : ProductSales) (priceReduction : ℝ) : ℝ :=
  (p.profitPerItem - priceReduction) * (p.baseSales + p.salesIncreasePerYuan * priceReduction)

/-- Theorem stating that a 3 yuan price reduction results in 1692 yuan daily profit. -/
theorem profit_after_reduction (p : ProductSales) 
  (h1 : p.basePrice = 50)
  (h2 : p.baseSales = 30)
  (h3 : p.profitPerItem = 50)
  (h4 : p.salesIncreasePerYuan = 2) :
  dailyProfit p 3 = 1692 := by sorry

/-- Theorem stating that a 25 yuan price reduction results in 2000 yuan daily profit. -/
theorem profit_for_target (p : ProductSales)
  (h1 : p.basePrice = 50)
  (h2 : p.baseSales = 30)
  (h3 : p.profitPerItem = 50)
  (h4 : p.salesIncreasePerYuan = 2) :
  dailyProfit p 25 = 2000 := by sorry

end NUMINAMATH_CALUDE_profit_after_reduction_profit_for_target_l2636_263613


namespace NUMINAMATH_CALUDE_scientific_notation_of_7413000000_l2636_263618

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_7413000000 :
  toScientificNotation 7413000000 = ScientificNotation.mk 7.413 9 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_7413000000_l2636_263618


namespace NUMINAMATH_CALUDE_ratio_chain_l2636_263654

theorem ratio_chain (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 2)
  : e / a = 1 / 17.5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_chain_l2636_263654


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l2636_263679

theorem fraction_sum_simplification :
  150 / 225 + 90 / 135 = 4 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l2636_263679


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l2636_263664

/-- Given a circle with equation x^2 + y^2 - 12x + 4y + 36 = 0, 
    the sum of the coordinates of its center is 4. -/
theorem circle_center_coordinate_sum : 
  ∀ (x y : ℝ), x^2 + y^2 - 12*x + 4*y + 36 = 0 → 
  ∃ (h k : ℝ), (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 36) ∧ h + k = 4) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l2636_263664


namespace NUMINAMATH_CALUDE_amy_homework_rate_l2636_263623

/-- Given a total number of problems and the time taken to complete them,
    calculate the number of problems completed per hour. -/
def problems_per_hour (total_problems : ℕ) (total_hours : ℕ) : ℚ :=
  total_problems / total_hours

/-- Theorem stating that with 24 problems completed in 6 hours,
    the number of problems completed per hour is 4. -/
theorem amy_homework_rate :
  problems_per_hour 24 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_amy_homework_rate_l2636_263623


namespace NUMINAMATH_CALUDE_unknown_number_problem_l2636_263653

theorem unknown_number_problem (x : ℝ) : 
  (0.1 * 30 + 0.15 * x = 10.5) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_problem_l2636_263653


namespace NUMINAMATH_CALUDE_fraction_simplification_l2636_263677

theorem fraction_simplification (x y z : ℝ) 
  (h1 : x > z) (h2 : z > y) (h3 : y > 0) :
  (x^z * z^y * y^x) / (z^z * y^y * x^x) = x^(z-x) * z^(y-z) * y^(x-y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2636_263677


namespace NUMINAMATH_CALUDE_gold_silver_alloy_composition_l2636_263638

/-- Prove the composition of a gold-silver alloy given its properties -/
theorem gold_silver_alloy_composition
  (total_mass : ℝ)
  (total_volume : ℝ)
  (density_gold : ℝ)
  (density_silver : ℝ)
  (h_total_mass : total_mass = 13.85)
  (h_total_volume : total_volume = 0.9)
  (h_density_gold : density_gold = 19.3)
  (h_density_silver : density_silver = 10.5) :
  ∃ (mass_gold mass_silver : ℝ),
    mass_gold + mass_silver = total_mass ∧
    mass_gold / density_gold + mass_silver / density_silver = total_volume ∧
    mass_gold = 9.65 ∧
    mass_silver = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_gold_silver_alloy_composition_l2636_263638


namespace NUMINAMATH_CALUDE_alex_zhu_same_section_probability_l2636_263694

def total_students : ℕ := 100
def selected_students : ℕ := 60
def num_sections : ℕ := 3
def students_per_section : ℕ := 20

theorem alex_zhu_same_section_probability :
  (3 : ℚ) * (Nat.choose 58 18) / (Nat.choose 60 20) = 19 / 165 := by
  sorry

end NUMINAMATH_CALUDE_alex_zhu_same_section_probability_l2636_263694


namespace NUMINAMATH_CALUDE_triangle_right_angled_l2636_263693

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if b*cos(B) + c*cos(C) = a*cos(A), then the triangle is right-angled. -/
theorem triangle_right_angled (A B C a b c : ℝ) : 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  b * Real.cos B + c * Real.cos C = a * Real.cos A →
  B = π/2 ∨ C = π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l2636_263693


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2636_263640

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 5*x + 6 = -4) → (∃ y : ℝ, y^2 - 5*y + 6 = -4 ∧ x + y = 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2636_263640


namespace NUMINAMATH_CALUDE_cubic_polynomials_with_specific_roots_and_difference_l2636_263659

/-- Two monic cubic polynomials with specific roots and a constant difference -/
theorem cubic_polynomials_with_specific_roots_and_difference (f g : ℝ → ℝ) (r : ℝ) :
  (∀ x, f x = (x - (r + 1)) * (x - (r + 7)) * (x - (3 * r + 8))) →  -- f is monic cubic with roots r+1, r+7, and 3r+8
  (∀ x, g x = (x - (r + 3)) * (x - (r + 9)) * (x - (3 * r + 12))) →  -- g is monic cubic with roots r+3, r+9, and 3r+12
  (∀ x, f x - g x = r) →  -- constant difference between f and g
  r = 32 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_with_specific_roots_and_difference_l2636_263659


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_hyperbola_l2636_263604

/-- Given an ellipse and a parabola that intersect, prove that the radius of the circumcircle
    of the triangle formed by their intersection points and the origin, along with the parameter
    of the parabola, satisfy a hyperbolic equation. -/
theorem ellipse_parabola_intersection_hyperbola (p r : ℝ) (hp : p > 0) (hr : r > 0) :
  (∃ x y : ℝ, x^2/4 + y^2/2 = 1 ∧ y^2 = 2*p*x) →
  (∃ x₀ y₀ : ℝ, x₀^2/4 + y₀^2/2 = 1 ∧ y₀^2 = 2*p*x₀ ∧ x₀^2 + y₀^2 = r^2) →
  r^2 - p^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_hyperbola_l2636_263604


namespace NUMINAMATH_CALUDE_tickets_spent_on_glow_bracelets_l2636_263672

/-- Given Connie's ticket redemption scenario, prove the number of tickets spent on glow bracelets. -/
theorem tickets_spent_on_glow_bracelets 
  (total_tickets : ℕ) 
  (koala_tickets : ℕ) 
  (earbud_tickets : ℕ) : 
  total_tickets = 50 → 
  koala_tickets = total_tickets / 2 → 
  earbud_tickets = 10 → 
  total_tickets - (koala_tickets + earbud_tickets) = 15 := by
  sorry


end NUMINAMATH_CALUDE_tickets_spent_on_glow_bracelets_l2636_263672


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2636_263619

theorem z_in_first_quadrant :
  ∀ z : ℂ, (1 + Complex.I)^2 * z = -1 + Complex.I →
  (z.re > 0 ∧ z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2636_263619


namespace NUMINAMATH_CALUDE_scale_E_accurate_l2636_263645

/-- Represents the weight measured by a scale -/
structure Scale where
  weight : ℝ

/-- Represents a set of five scales used in a health check center -/
structure HealthCheckScales where
  A : Scale
  B : Scale
  C : Scale
  D : Scale
  E : Scale

/-- The conditions of the health check scales problem -/
def ScaleConditions (s : HealthCheckScales) : Prop :=
  s.C.weight = s.B.weight - 0.3 ∧
  s.D.weight = s.C.weight - 0.1 ∧
  s.E.weight = s.A.weight - 0.1 ∧
  s.C.weight = s.E.weight - 0.1

/-- The average weight of all scales is accurate -/
def AverageWeightAccurate (s : HealthCheckScales) (actualWeight : ℝ) : Prop :=
  (s.A.weight + s.B.weight + s.C.weight + s.D.weight + s.E.weight) / 5 = actualWeight

/-- Theorem stating that scale E is accurate given the conditions -/
theorem scale_E_accurate (s : HealthCheckScales) (actualWeight : ℝ)
  (h1 : ScaleConditions s)
  (h2 : AverageWeightAccurate s actualWeight) :
  s.E.weight = actualWeight :=
sorry

end NUMINAMATH_CALUDE_scale_E_accurate_l2636_263645


namespace NUMINAMATH_CALUDE_segment_construction_l2636_263628

theorem segment_construction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := a + b
  4 * (c * (a * c).sqrt).sqrt * x = 4 * (c * (a * c).sqrt).sqrt * a + 4 * (c * (a * c).sqrt).sqrt * b :=
by sorry

end NUMINAMATH_CALUDE_segment_construction_l2636_263628


namespace NUMINAMATH_CALUDE_marble_ratio_l2636_263690

theorem marble_ratio (total : ℕ) (blue green yellow : ℕ) 
  (h_total : total = 164)
  (h_blue : blue = total / 2)
  (h_green : green = 27)
  (h_yellow : yellow = 14) :
  (total - (blue + green + yellow)) * 4 = total := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l2636_263690


namespace NUMINAMATH_CALUDE_four_propositions_l2636_263622

-- Definition of opposite numbers
def are_opposite (x y : ℝ) : Prop := x = -y

-- Definition of congruent triangles
def congruent_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry

-- Definition of triangle area
def triangle_area (t : Set ℝ × Set ℝ) : ℝ := sorry

-- Statement of the theorem
theorem four_propositions :
  (∀ x y : ℝ, are_opposite x y → x + y = 0) ∧
  (∀ q : ℝ, (∃ x : ℝ, x^2 + 2*x + q = 0) → q ≤ 1) ∧
  (∃ t1 t2 : Set ℝ × Set ℝ, ¬(congruent_triangles t1 t2) ∧ triangle_area t1 = triangle_area t2) ∧
  (∃ a b c : ℝ, a > b ∧ ¬(a * c^2 > b * c^2)) :=
by sorry

end NUMINAMATH_CALUDE_four_propositions_l2636_263622


namespace NUMINAMATH_CALUDE_boat_current_rate_l2636_263674

/-- Proves that the rate of the current is 5 km/hr given the conditions of the boat problem -/
theorem boat_current_rate 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 20) 
  (h2 : distance = 11.25) 
  (h3 : time_minutes = 27) : 
  ∃ current_rate : ℝ, 
    current_rate = 5 ∧ 
    distance = (boat_speed + current_rate) * (time_minutes / 60) :=
by
  sorry


end NUMINAMATH_CALUDE_boat_current_rate_l2636_263674


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2636_263668

theorem quadratic_equation_solution :
  let x₁ : ℝ := 1 + Real.sqrt 2 / 2
  let x₂ : ℝ := 1 - Real.sqrt 2 / 2
  (2 * x₁^2 - 4 * x₁ + 1 = 0) ∧ (2 * x₂^2 - 4 * x₂ + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2636_263668


namespace NUMINAMATH_CALUDE_average_equals_x_l2636_263665

theorem average_equals_x (x : ℝ) : 
  (2 + 5 + x + 14 + 15) / 5 = x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_equals_x_l2636_263665


namespace NUMINAMATH_CALUDE_range_of_f_l2636_263652

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2636_263652


namespace NUMINAMATH_CALUDE_chess_draw_probability_l2636_263691

theorem chess_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.4) 
  (h_not_lose : p_not_lose = 0.9) : 
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l2636_263691


namespace NUMINAMATH_CALUDE_munchausen_palindrome_exists_l2636_263698

/-- A type representing a multi-digit number --/
def MultiDigitNumber := List Nat

/-- Check if a number is a palindrome --/
def isPalindrome (n : MultiDigitNumber) : Prop :=
  n = n.reverse

/-- Check if a list of numbers contains all numbers from 1 to N exactly once --/
def containsOneToN (l : List Nat) (N : Nat) : Prop :=
  l.toFinset = Finset.range N

/-- A function that represents cutting a number between digits --/
def cutBetweenDigits (n : MultiDigitNumber) : List Nat := sorry

/-- The main theorem --/
theorem munchausen_palindrome_exists :
  ∃ (n : MultiDigitNumber),
    isPalindrome n ∧
    containsOneToN (cutBetweenDigits n) 19 := by
  sorry

end NUMINAMATH_CALUDE_munchausen_palindrome_exists_l2636_263698


namespace NUMINAMATH_CALUDE_relationship_abc_l2636_263684

theorem relationship_abc : ∃ (a b c : ℝ), 
  a = 2^(2/5) ∧ b = 9^(1/5) ∧ c = 3^(3/4) ∧ a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2636_263684


namespace NUMINAMATH_CALUDE_modular_inverse_15_mod_16_l2636_263660

theorem modular_inverse_15_mod_16 : ∃ x : ℤ, (15 * x) % 16 = 1 :=
by
  use 15
  sorry

end NUMINAMATH_CALUDE_modular_inverse_15_mod_16_l2636_263660


namespace NUMINAMATH_CALUDE_mateo_grape_bottles_l2636_263671

/-- Represents the number of bottles of a specific soda type. -/
structure SodaCount where
  orange : ℕ
  grape : ℕ

/-- Represents a person's soda inventory. -/
structure SodaInventory where
  count : SodaCount
  litersPerBottle : ℕ

def julio : SodaInventory :=
  { count := { orange := 4, grape := 7 },
    litersPerBottle := 2 }

def mateo (grapeBottles : ℕ) : SodaInventory :=
  { count := { orange := 1, grape := grapeBottles },
    litersPerBottle := 2 }

def totalLiters (inventory : SodaInventory) : ℕ :=
  (inventory.count.orange + inventory.count.grape) * inventory.litersPerBottle

theorem mateo_grape_bottles :
  ∃ g : ℕ, totalLiters julio = totalLiters (mateo g) + 14 ∧ g = 3 := by
  sorry

end NUMINAMATH_CALUDE_mateo_grape_bottles_l2636_263671


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l2636_263686

/-- The circle equation passing through points A(4, 1), B(6, -3), and C(-3, 0) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y - 15 = 0

/-- Point A coordinates -/
def point_A : ℝ × ℝ := (4, 1)

/-- Point B coordinates -/
def point_B : ℝ × ℝ := (6, -3)

/-- Point C coordinates -/
def point_C : ℝ × ℝ := (-3, 0)

theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l2636_263686


namespace NUMINAMATH_CALUDE_roots_of_equation_l2636_263658

theorem roots_of_equation (x : ℝ) : 
  (x = 0 ∨ x = -3) ↔ -x * (x + 3) = x * (x + 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2636_263658


namespace NUMINAMATH_CALUDE_cookies_removed_theorem_l2636_263632

/-- Calculates the number of cookies removed in 4 days given initial and final cookie counts over a week -/
def cookies_removed_in_four_days (initial_cookies : ℕ) (remaining_cookies : ℕ) : ℕ :=
  let total_removed : ℕ := initial_cookies - remaining_cookies
  let daily_removal : ℕ := total_removed / 7
  4 * daily_removal

/-- Theorem stating that given 70 initial cookies and 28 remaining after a week, 24 cookies are removed in 4 days -/
theorem cookies_removed_theorem :
  cookies_removed_in_four_days 70 28 = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookies_removed_theorem_l2636_263632


namespace NUMINAMATH_CALUDE_exists_center_nail_pierces_one_cardboard_l2636_263643

/-- A cardboard figure -/
structure Cardboard where
  shape : Set (ℝ × ℝ)

/-- A rectangular box bottom -/
structure Box where
  width : ℝ
  height : ℝ

/-- A configuration of two cardboard pieces on a box bottom -/
structure Configuration where
  box : Box
  piece1 : Cardboard
  piece2 : Cardboard
  position1 : ℝ × ℝ
  position2 : ℝ × ℝ

/-- Predicate to check if a point is covered by a cardboard piece at a given position -/
def covers (c : Cardboard) (pos : ℝ × ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - pos.1, point.2 - pos.2) ∈ c.shape

/-- Predicate to check if a configuration completely covers the box bottom -/
def completelyCovers (config : Configuration) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ config.box.width → 0 ≤ y ∧ y ≤ config.box.height →
    covers config.piece1 config.position1 (x, y) ∨ covers config.piece2 config.position2 (x, y)

/-- Theorem stating that there exists a configuration where the center nail pierces only one cardboard -/
theorem exists_center_nail_pierces_one_cardboard :
  ∃ (config : Configuration), completelyCovers config ∧
    (covers config.piece1 config.position1 (config.box.width / 2, config.box.height / 2) ≠
     covers config.piece2 config.position2 (config.box.width / 2, config.box.height / 2)) :=
by sorry

end NUMINAMATH_CALUDE_exists_center_nail_pierces_one_cardboard_l2636_263643


namespace NUMINAMATH_CALUDE_existence_of_prime_and_power_l2636_263688

/-- The distance from a real number to its nearest integer -/
noncomputable def dist_to_nearest_int (x : ℝ) : ℝ :=
  |x - round x|

/-- The statement of the theorem -/
theorem existence_of_prime_and_power (a b : ℕ+) :
  ∃ (p : ℕ) (k : ℕ), Prime p ∧ p % 2 = 1 ∧
    dist_to_nearest_int (a / p^k : ℝ) +
    dist_to_nearest_int (b / p^k : ℝ) +
    dist_to_nearest_int ((a + b) / p^k : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_and_power_l2636_263688


namespace NUMINAMATH_CALUDE_division_problem_l2636_263601

theorem division_problem (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) : 
  dividend = 64 → quotient = 8 → dividend = divisor * quotient → divisor = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2636_263601


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l2636_263608

/-- The probability of getting exactly 3 tails in 8 flips of an unfair coin -/
theorem unfair_coin_probability (n : ℕ) (k : ℕ) (p : ℚ) :
  n = 8 →
  k = 3 →
  p = 2/3 →
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k) = 448/6561 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l2636_263608


namespace NUMINAMATH_CALUDE_hyperbola_parabola_focus_coincidence_l2636_263650

/-- The value of p for which the right focus of the hyperbola x^2 - y^2/3 = 1 
    coincides with the focus of the parabola y^2 = 2px -/
theorem hyperbola_parabola_focus_coincidence (p : ℝ) : 
  (∃ (x y : ℝ), x^2 - y^2/3 = 1 ∧ y^2 = 2*p*x ∧ 
   (x, y) = (2, 0) ∧ (x, y) = (p/2, 0)) → 
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_focus_coincidence_l2636_263650


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l2636_263626

theorem fixed_point_of_linear_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a * x + 2
  f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l2636_263626


namespace NUMINAMATH_CALUDE_root_sum_squares_l2636_263685

theorem root_sum_squares (a b c d : ℂ) : 
  (a^4 - 24*a^3 + 50*a^2 - 35*a + 7 = 0) →
  (b^4 - 24*b^3 + 50*b^2 - 35*b + 7 = 0) →
  (c^4 - 24*c^3 + 50*c^2 - 35*c + 7 = 0) →
  (d^4 - 24*d^3 + 50*d^2 - 35*d + 7 = 0) →
  (a+b+c)^2 + (b+c+d)^2 + (c+d+a)^2 + (d+a+b)^2 = 2104 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_squares_l2636_263685


namespace NUMINAMATH_CALUDE_expression_factorization_l2636_263627

theorem expression_factorization (x : ℝ) : 
  (12 * x^5 + 33 * x^3 + 10) - (3 * x^5 - 4 * x^3 - 1) = x^3 * (9 * x^2 + 37) + 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2636_263627


namespace NUMINAMATH_CALUDE_fixed_point_of_function_l2636_263657

theorem fixed_point_of_function (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (a - 1) * 2^x - 2*a
  f 1 = -2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_function_l2636_263657


namespace NUMINAMATH_CALUDE_integer_fraction_pairs_l2636_263602

theorem integer_fraction_pairs : 
  {p : ℕ × ℕ | p.1 > 0 ∧ p.2 > 0 ∧ (∃ k : ℤ, (p.2^3 + 1 : ℤ) = k * (p.1 * p.2 - 1))} = 
  {(2,1), (3,1), (2,2), (5,2), (5,3), (2,5), (3,5)} := by
sorry

end NUMINAMATH_CALUDE_integer_fraction_pairs_l2636_263602


namespace NUMINAMATH_CALUDE_polynomial_roots_l2636_263663

/-- The polynomial function we're investigating -/
def f (x : ℝ) : ℝ := x^4 - 3*x^3 + 3*x^2 - x - 6

/-- The set of real roots of the polynomial -/
def roots : Set ℝ := {-1, 1, 2}

/-- Theorem stating that the roots of the polynomial are correct -/
theorem polynomial_roots : 
  ∀ x : ℝ, f x = 0 ↔ x ∈ roots ∨ (x = 1 ∧ (deriv f) x = 0) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2636_263663


namespace NUMINAMATH_CALUDE_complement_of_A_l2636_263635

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem complement_of_A (A B : Set ℕ) 
  (h1 : A ∪ B = {1, 2, 3, 4, 5})
  (h2 : A ∩ B = {3, 4, 5}) :
  Aᶜ = {6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l2636_263635


namespace NUMINAMATH_CALUDE_f_equals_F_l2636_263633

/-- The function f(x) = 3x^4 - x^3 -/
def f (x : ℝ) : ℝ := 3 * x^4 - x^3

/-- The function F(x) = x(3x^3 - 1) -/
def F (x : ℝ) : ℝ := x * (3 * x^3 - 1)

/-- Theorem stating that f and F are the same function -/
theorem f_equals_F : f = F := by sorry

end NUMINAMATH_CALUDE_f_equals_F_l2636_263633


namespace NUMINAMATH_CALUDE_trips_per_month_l2636_263610

/-- Given a person who spends 72 hours driving in a year, with each round trip
    taking 3 hours, prove that the number of trips per month is 2. -/
theorem trips_per_month (hours_per_year : ℕ) (hours_per_trip : ℕ) 
    (months_per_year : ℕ) : ℕ :=
  by
  have h1 : hours_per_year = 72 := by sorry
  have h2 : hours_per_trip = 3 := by sorry
  have h3 : months_per_year = 12 := by sorry
  
  let trips_per_year : ℕ := hours_per_year / hours_per_trip
  
  have h4 : trips_per_year = 24 := by sorry
  
  exact trips_per_year / months_per_year

end NUMINAMATH_CALUDE_trips_per_month_l2636_263610


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2636_263683

theorem polynomial_divisibility (p q : ℚ) : 
  (∀ x : ℚ, (x + 1) * (x - 2) ∣ (x^5 - x^4 + 2*x^3 - p*x^2 + q*x - 5)) → 
  p = 3/2 ∧ q = -21/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2636_263683


namespace NUMINAMATH_CALUDE_library_shelves_l2636_263689

theorem library_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 14240) (h2 : books_per_shelf = 8) :
  total_books / books_per_shelf = 1780 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_l2636_263689


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_function_ordering_l2636_263648

/-- The quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution_implies_function_ordering
  (a b c : ℝ)
  (h : ∀ x, a * x^2 + b * x + c > 0 ↔ -2 < x ∧ x < 4) :
  f a b c 5 < f a b c (-1) ∧ f a b c (-1) < f a b c 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_function_ordering_l2636_263648


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l2636_263607

theorem quadratic_form_minimum (x y : ℝ) : 
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 7 ≥ 28 ∧ 
  ∃ x₀ y₀ : ℝ, 3 * x₀^2 + 4 * x₀ * y₀ + 2 * y₀^2 - 6 * x₀ + 4 * y₀ + 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l2636_263607


namespace NUMINAMATH_CALUDE_laptop_sticker_price_l2636_263678

/-- The sticker price of a laptop satisfying certain discount conditions -/
theorem laptop_sticker_price : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (0.7 * x - (0.8 * x - 120) = 30) ∧ 
  (x = 900) := by
  sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_l2636_263678


namespace NUMINAMATH_CALUDE_tv_price_change_l2636_263631

theorem tv_price_change (x : ℝ) : 
  (1 - x / 100) * (1 + 40 / 100) = 1 + 12 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l2636_263631


namespace NUMINAMATH_CALUDE_eighteen_percent_of_500_is_90_l2636_263675

theorem eighteen_percent_of_500_is_90 : 
  (18 : ℚ) / 100 * 500 = 90 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_percent_of_500_is_90_l2636_263675


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_four_is_solution_four_is_smallest_l2636_263695

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 3 → (∃ n : ℕ, 3 * b + 4 = n^2) → b ≥ 4 :=
by sorry

theorem four_is_solution : 
  ∃ n : ℕ, 3 * 4 + 4 = n^2 :=
by sorry

theorem four_is_smallest : 
  ∀ b : ℕ, b > 3 ∧ (∃ n : ℕ, 3 * b + 4 = n^2) → b = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_four_is_solution_four_is_smallest_l2636_263695


namespace NUMINAMATH_CALUDE_twelfth_term_is_12_l2636_263667

/-- An arithmetic sequence with a₂ = -8 and common difference d = 2 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  let a₁ := -10  -- Derived from a₂ = -8 and d = 2
  a₁ + (n - 1) * 2

/-- The 12th term of the arithmetic sequence is 12 -/
theorem twelfth_term_is_12 : arithmetic_sequence 12 = 12 := by
  sorry

#eval arithmetic_sequence 12  -- For verification

end NUMINAMATH_CALUDE_twelfth_term_is_12_l2636_263667


namespace NUMINAMATH_CALUDE_average_tickets_sold_l2636_263670

/-- Proves that the average number of tickets sold per day is 80 -/
theorem average_tickets_sold (total_days : ℕ) (total_worth : ℕ) (ticket_cost : ℕ) :
  total_days = 3 →
  total_worth = 960 →
  ticket_cost = 4 →
  (total_worth / ticket_cost) / total_days = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_tickets_sold_l2636_263670


namespace NUMINAMATH_CALUDE_boys_in_jakes_class_l2636_263615

/-- Calculates the number of boys in a class given the ratio of girls to boys and the total number of students -/
def number_of_boys (girls_ratio : ℕ) (boys_ratio : ℕ) (total_students : ℕ) : ℕ :=
  (boys_ratio * total_students) / (girls_ratio + boys_ratio)

/-- Proves that in a class with a 3:4 ratio of girls to boys and 35 total students, there are 20 boys -/
theorem boys_in_jakes_class :
  number_of_boys 3 4 35 = 20 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_jakes_class_l2636_263615


namespace NUMINAMATH_CALUDE_area_of_smaller_triangle_l2636_263625

/-- Given an outer equilateral triangle with area 36 square units and an inner equilateral triangle
    with area 4 square units, if the space between these triangles is divided into four congruent
    triangles, then the area of each of these smaller triangles is 8 square units. -/
theorem area_of_smaller_triangle (outer_area inner_area : ℝ) (h1 : outer_area = 36)
    (h2 : inner_area = 4) (h3 : outer_area > inner_area) :
  (outer_area - inner_area) / 4 = 8 :=
by sorry

end NUMINAMATH_CALUDE_area_of_smaller_triangle_l2636_263625


namespace NUMINAMATH_CALUDE_solve_for_q_l2636_263600

theorem solve_for_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_solve_for_q_l2636_263600


namespace NUMINAMATH_CALUDE_triangle_side_length_l2636_263636

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (A + B + C = π) →
  -- Condition: √3 sin A + cos A = 2
  (Real.sqrt 3 * Real.sin A + Real.cos A = 2) →
  -- Condition: a = 3
  (a = 3) →
  -- Condition: C = 5π/12
  (C = 5 * π / 12) →
  -- Conclusion: b = √6
  (b = Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2636_263636


namespace NUMINAMATH_CALUDE_rectangular_prism_face_fits_in_rectangle_l2636_263646

/-- Represents a rectangular prism with dimensions a ≤ b ≤ c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : 0 < a
  h2 : a ≤ b
  h3 : b ≤ c

/-- Represents a rectangle with dimensions d₁ ≤ d₂ -/
structure Rectangle where
  d1 : ℝ
  d2 : ℝ
  h : d1 ≤ d2

/-- Theorem: Given a rectangular prism and a rectangle that can contain
    the prism's hexagonal cross-section, prove that the rectangle can
    contain one face of the prism -/
theorem rectangular_prism_face_fits_in_rectangle
  (prism : RectangularPrism) (rect : Rectangle)
  (hex_fits : ∃ (h : ℝ), h > 0 ∧ h^2 + rect.d1^2 ≥ prism.b^2 + prism.c^2) :
  min rect.d1 rect.d2 ≥ prism.a ∧ max rect.d1 rect.d2 ≥ prism.b :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_face_fits_in_rectangle_l2636_263646


namespace NUMINAMATH_CALUDE_correct_bucket_size_l2636_263641

/-- The size of the bucket needed to collect leaking fluid -/
def bucket_size (leak_rate : ℝ) (max_time : ℝ) : ℝ :=
  2 * leak_rate * max_time

/-- Theorem stating the correct bucket size for the given conditions -/
theorem correct_bucket_size :
  bucket_size 1.5 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_bucket_size_l2636_263641


namespace NUMINAMATH_CALUDE_largest_exponent_sum_l2636_263629

/-- The set of integers from 1 to 1997 -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 1997}

/-- The condition for the intersection of A and B with {1,2} -/
def intersection_condition (A B : Set ℕ) : Prop :=
  (A ∩ B) ∩ {1, 2} = {1}

/-- The number of ordered pairs (A, B) satisfying the conditions -/
def N : ℕ := sorry

/-- The theorem stating the largest possible value of m + k -/
theorem largest_exponent_sum : 
  ∀ m k : ℕ, (3^m * 5^k : ℕ) ∣ N → m + k ≤ 1004 :=
by sorry

end NUMINAMATH_CALUDE_largest_exponent_sum_l2636_263629


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2636_263647

/-- Given a quadratic function f(x) = ax^2 + bx + c, 
    if f(x₁) = f(x₂) and x₁ ≠ x₂, then f(x₁ + x₂) = c -/
theorem quadratic_function_property (a b c x₁ x₂ : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  x₁ ≠ x₂ → f x₁ = f x₂ → f (x₁ + x₂) = c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2636_263647


namespace NUMINAMATH_CALUDE_circle_passes_through_points_circle_uniqueness_l2636_263699

/-- A circle is defined by the equation x^2 + y^2 + Dx + Ey + F = 0 --/
def Circle (D E F : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 + y^2 + D*x + E*y + F = 0

/-- The specific circle we're interested in --/
def SpecificCircle : ℝ × ℝ → Prop :=
  Circle (-4) (-6) 0

theorem circle_passes_through_points :
  SpecificCircle (0, 0) ∧
  SpecificCircle (4, 0) ∧
  SpecificCircle (-1, 1) :=
by sorry

/-- Uniqueness of the circle --/
theorem circle_uniqueness (D E F : ℝ) :
  Circle D E F (0, 0) →
  Circle D E F (4, 0) →
  Circle D E F (-1, 1) →
  ∀ (x y : ℝ), Circle D E F (x, y) ↔ SpecificCircle (x, y) :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_circle_uniqueness_l2636_263699


namespace NUMINAMATH_CALUDE_chord_length_l2636_263616

/-- The circle passing through the intersection points of y = x, y = 2x, and y = 15 - 0.5x -/
def special_circle : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ (x - 5)^2 + (y - 5)^2 = 50}

/-- The line x + y = 16 -/
def line : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ x + y = 16}

/-- The chord formed by the intersection of the special circle and the line -/
def chord : Set (ℝ × ℝ) :=
  special_circle ∩ line

theorem chord_length : 
  ∃ (p q : ℝ × ℝ), p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l2636_263616


namespace NUMINAMATH_CALUDE_composite_divisibility_l2636_263605

theorem composite_divisibility (k : ℕ) (p_k : ℕ) (n : ℕ) 
  (h1 : k ≥ 14)
  (h2 : Nat.Prime p_k)
  (h3 : p_k < k)
  (h4 : ∀ p, Nat.Prime p → p < k → p ≤ p_k)
  (h5 : p_k ≥ 3 * k / 4)
  (h6 : ¬ Nat.Prime n) :
  (n = 2 * p_k → ¬ (n ∣ Nat.factorial (n - k))) ∧
  (n > 2 * p_k → n ∣ Nat.factorial (n - k)) := by
  sorry

end NUMINAMATH_CALUDE_composite_divisibility_l2636_263605


namespace NUMINAMATH_CALUDE_jellybean_problem_l2636_263697

theorem jellybean_problem (initial : ℕ) (first_removal : ℕ) (second_removal : ℕ) (final : ℕ) 
  (h1 : initial = 37)
  (h2 : first_removal = 15)
  (h3 : second_removal = 4)
  (h4 : final = 23) :
  ∃ (added_back : ℕ), initial - first_removal + added_back - second_removal = final ∧ added_back = 5 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l2636_263697


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l2636_263666

/-- Given a circle with center (4, -2) and one endpoint of a diameter at (1, 5),
    the other endpoint of the diameter is at (7, -9). -/
theorem circle_diameter_endpoint :
  let center : ℝ × ℝ := (4, -2)
  let endpoint1 : ℝ × ℝ := (1, 5)
  let endpoint2 : ℝ × ℝ := (7, -9)
  (endpoint1.1 - center.1 = center.1 - endpoint2.1) ∧
  (endpoint1.2 - center.2 = center.2 - endpoint2.2) :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l2636_263666


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_l2636_263637

/-- The number of digits in the given number -/
def total_digits : ℕ := 4

/-- The number of distinct digits in the given number -/
def distinct_digits : ℕ := 2

/-- The number of occurrences of the first digit (3) -/
def count_first_digit : ℕ := 2

/-- The number of occurrences of the second digit (0) -/
def count_second_digit : ℕ := 2

/-- The function to calculate the number of permutations -/
def permutations (n : ℕ) (n1 : ℕ) (n2 : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial n1 * Nat.factorial n2)

/-- The theorem stating the number of different four-digit numbers -/
theorem four_digit_numbers_count : 
  permutations (total_digits - 1) (count_first_digit - 1) count_second_digit = 3 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_l2636_263637


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2636_263614

theorem quadratic_equations_solutions :
  (∀ x, 4 * x^2 = 12 * x ↔ x = 0 ∨ x = 3) ∧
  (∀ x, 3/4 * x^2 - 2*x - 1/2 = 0 ↔ x = (4 + Real.sqrt 22) / 3 ∨ x = (4 - Real.sqrt 22) / 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2636_263614


namespace NUMINAMATH_CALUDE_inequality_solution_l2636_263603

theorem inequality_solution (x : ℝ) :
  x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 →
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔
  (x < 1 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2636_263603


namespace NUMINAMATH_CALUDE_added_value_to_doubled_number_l2636_263681

theorem added_value_to_doubled_number (n : ℝ) (x : ℝ) : 
  n = 4 → (2 * n + x = 8 * n - 4) → x = 20 := by
sorry

end NUMINAMATH_CALUDE_added_value_to_doubled_number_l2636_263681


namespace NUMINAMATH_CALUDE_yellow_crayon_count_prove_yellow_crayons_l2636_263612

/-- Proves that the number of yellow crayons is 32 given the conditions of the problem. -/
theorem yellow_crayon_count : ℕ → ℕ → ℕ → Prop :=
  fun red blue yellow =>
    (red = 14) →
    (blue = red + 5) →
    (yellow = 2 * blue - 6) →
    (yellow = 32)

/-- The main theorem that proves the number of yellow crayons. -/
theorem prove_yellow_crayons :
  ∃ (red blue yellow : ℕ),
    yellow_crayon_count red blue yellow :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_crayon_count_prove_yellow_crayons_l2636_263612


namespace NUMINAMATH_CALUDE_bicycle_rental_theorem_l2636_263649

/-- Represents the rental time for a bicycle. -/
inductive RentalTime
  | LessThanTwo
  | TwoToThree
  | ThreeToFour

/-- Calculates the rental fee based on the rental time. -/
def rentalFee (time : RentalTime) : ℕ :=
  match time with
  | RentalTime.LessThanTwo => 0
  | RentalTime.TwoToThree => 2
  | RentalTime.ThreeToFour => 4

/-- Represents the probabilities for each rental time for a person. -/
structure RentalProbabilities where
  lessThanTwo : ℚ
  twoToThree : ℚ
  threeToFour : ℚ

/-- The rental probabilities for person A. -/
def probA : RentalProbabilities :=
  { lessThanTwo := 1/4, twoToThree := 1/2, threeToFour := 1/4 }

/-- The rental probabilities for person B. -/
def probB : RentalProbabilities :=
  { lessThanTwo := 1/2, twoToThree := 1/4, threeToFour := 1/4 }

/-- Calculates the probability that two people pay the same fee. -/
def probSameFee (pA pB : RentalProbabilities) : ℚ :=
  pA.lessThanTwo * pB.lessThanTwo +
  pA.twoToThree * pB.twoToThree +
  pA.threeToFour * pB.threeToFour

/-- Calculates the expected value of the sum of fees for two people. -/
def expectedSumFees (pA pB : RentalProbabilities) : ℚ :=
  0 * (pA.lessThanTwo * pB.lessThanTwo) +
  2 * (pA.lessThanTwo * pB.twoToThree + pA.twoToThree * pB.lessThanTwo) +
  4 * (pA.lessThanTwo * pB.threeToFour + pA.twoToThree * pB.twoToThree + pA.threeToFour * pB.lessThanTwo) +
  6 * (pA.twoToThree * pB.threeToFour + pA.threeToFour * pB.twoToThree) +
  8 * (pA.threeToFour * pB.threeToFour)

theorem bicycle_rental_theorem :
  probSameFee probA probB = 5/16 ∧
  expectedSumFees probA probB = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_rental_theorem_l2636_263649


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_a_equals_one_l2636_263609

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def PurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number a. -/
def z (a : ℝ) : ℂ :=
  ⟨a^2 + 2*a - 3, a + 3⟩

theorem purely_imaginary_implies_a_equals_one :
  ∀ a : ℝ, PurelyImaginary (z a) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_a_equals_one_l2636_263609
