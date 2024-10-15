import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_property_l1761_176194

theorem divisibility_property (y : ℕ) (h : y ≠ 0) :
  (y - 1) ∣ (y^(y^2) - 2*y^(y+1) + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l1761_176194


namespace NUMINAMATH_CALUDE_right_triangle_legs_l1761_176197

theorem right_triangle_legs (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a^2 + b^2 = 37^2 → 
  a * b = (a + 7) * (b - 2) →
  (a = 35 ∧ b = 12) ∨ (a = 12 ∧ b = 35) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l1761_176197


namespace NUMINAMATH_CALUDE_problem_solution_l1761_176143

theorem problem_solution (x : ℝ) (h : x + 1/x = 5) :
  (x - 3)^2 + 36/((x - 3)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1761_176143


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1761_176178

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set M
def M : Set ℕ := {x ∈ U | x^2 - 6*x + 5 ≤ 0}

-- State the theorem
theorem complement_of_M_in_U :
  (U \ M) = {6, 7} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1761_176178


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l1761_176180

theorem complex_number_imaginary_part (i : ℂ) (a : ℝ) :
  i * i = -1 →
  let z := (1 - a * i) / (1 + i)
  Complex.im z = -3 →
  a = 5 := by sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l1761_176180


namespace NUMINAMATH_CALUDE_min_value_expression_l1761_176198

theorem min_value_expression (a b c : ℝ) 
  (sum_condition : a + b + c = -1) 
  (product_condition : a * b * c ≤ -3) :
  (a * b + 1) / (a + b) + (b * c + 1) / (b + c) + (c * a + 1) / (c + a) ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1761_176198


namespace NUMINAMATH_CALUDE_sqrt_400_div_2_l1761_176138

theorem sqrt_400_div_2 : Real.sqrt 400 / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_400_div_2_l1761_176138


namespace NUMINAMATH_CALUDE_remainder_nineteen_power_plus_nineteen_mod_twenty_l1761_176147

theorem remainder_nineteen_power_plus_nineteen_mod_twenty : (19^19 + 19) % 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_nineteen_power_plus_nineteen_mod_twenty_l1761_176147


namespace NUMINAMATH_CALUDE_atlantis_population_growth_l1761_176183

def initial_year : Nat := 2000
def initial_population : Nat := 400
def island_capacity : Nat := 15000
def years_to_check : Nat := 200

def population_after_n_cycles (n : Nat) : Nat :=
  initial_population * 2^n

theorem atlantis_population_growth :
  ∃ (y : Nat), y ≤ years_to_check ∧ 
  population_after_n_cycles (y / 40) ≥ island_capacity :=
sorry

end NUMINAMATH_CALUDE_atlantis_population_growth_l1761_176183


namespace NUMINAMATH_CALUDE_trig_identity_l1761_176195

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1761_176195


namespace NUMINAMATH_CALUDE_inequality_proof_l1761_176101

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1761_176101


namespace NUMINAMATH_CALUDE_not_perfect_square_l1761_176120

theorem not_perfect_square (n : ℤ) : ¬ ∃ m : ℤ, m^2 = 4*n + 3 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1761_176120


namespace NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l1761_176187

theorem sum_of_four_primes_divisible_by_60 (p q r s : ℕ) :
  Prime p → Prime q → Prime r → Prime s →
  5 < p → p < q → q < r → r < s → s < p + 10 →
  60 ∣ (p + q + r + s) := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l1761_176187


namespace NUMINAMATH_CALUDE_rectangle_arrangement_l1761_176115

/-- Given 110 identical rectangular sheets where each sheet's length is 10 cm longer than its width,
    and when arranged as in Figure 1 they form a rectangle of length 2750 cm,
    prove that the length of the rectangle formed when arranged as in Figure 2 is 1650 cm. -/
theorem rectangle_arrangement (n : ℕ) (sheet_length sheet_width : ℝ) 
  (h1 : n = 110)
  (h2 : sheet_length = sheet_width + 10)
  (h3 : n * sheet_length = 2750) :
  n * sheet_width = 1650 := by
  sorry

#check rectangle_arrangement

end NUMINAMATH_CALUDE_rectangle_arrangement_l1761_176115


namespace NUMINAMATH_CALUDE_sequence_problem_l1761_176109

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

/-- The main theorem -/
theorem sequence_problem (a b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_geom : geometric_sequence b)
    (h_a_sum : a 1 + a 5 + a 9 = 9)
    (h_b_prod : b 2 * b 5 * b 8 = 3 * Real.sqrt 3) :
    (a 2 + a 8) / (1 + b 2 * b 8) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1761_176109


namespace NUMINAMATH_CALUDE_cone_sphere_radius_theorem_l1761_176121

/-- Represents a right cone with a sphere inscribed within it. -/
structure ConeWithSphere where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ

/-- Checks if the sphere radius can be expressed in the form b√d - b. -/
def has_valid_sphere_radius (cone : ConeWithSphere) (b d : ℕ) : Prop :=
  cone.sphere_radius = b * Real.sqrt d - b

/-- The main theorem stating the relationship between b and d for the given cone. -/
theorem cone_sphere_radius_theorem (cone : ConeWithSphere) (b d : ℕ) :
  cone.base_radius = 15 →
  cone.height = 20 →
  has_valid_sphere_radius cone b d →
  b + d = 17 := by
  sorry

#check cone_sphere_radius_theorem

end NUMINAMATH_CALUDE_cone_sphere_radius_theorem_l1761_176121


namespace NUMINAMATH_CALUDE_frog_jump_distance_l1761_176152

/-- The jumping contest problem -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (frog_extra_jump : ℕ) :
  grasshopper_jump = 36 →
  frog_extra_jump = 17 →
  grasshopper_jump + frog_extra_jump = 53 :=
by
  sorry

#check frog_jump_distance

end NUMINAMATH_CALUDE_frog_jump_distance_l1761_176152


namespace NUMINAMATH_CALUDE_jacob_michael_age_difference_l1761_176191

theorem jacob_michael_age_difference :
  ∀ (jacob_age michael_age : ℕ),
    (jacob_age + 4 = 11) →
    (michael_age + 5 = 2 * (jacob_age + 5)) →
    (michael_age - jacob_age = 12) :=
by sorry

end NUMINAMATH_CALUDE_jacob_michael_age_difference_l1761_176191


namespace NUMINAMATH_CALUDE_rational_criterion_l1761_176141

/-- The number of different digit sequences of length n in the decimal expansion of a real number -/
def num_digit_sequences (a : ℝ) (n : ℕ) : ℕ := sorry

/-- A real number is rational if there exists a natural number n such that 
    the number of different digit sequences of length n in its decimal expansion 
    is less than or equal to n + 8 -/
theorem rational_criterion (a : ℝ) : 
  (∃ n : ℕ, num_digit_sequences a n ≤ n + 8) → ∃ q : ℚ, a = ↑q := by sorry

end NUMINAMATH_CALUDE_rational_criterion_l1761_176141


namespace NUMINAMATH_CALUDE_acute_triangle_trig_ranges_l1761_176155

variable (B C : Real)

theorem acute_triangle_trig_ranges 
  (acute : 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (angle_sum : B + C = π/3) :
  let A : Real := π/3
  (((3 + Real.sqrt 3) / 2 < Real.sin A + Real.sin B + Real.sin C) ∧ 
   (Real.sin A + Real.sin B + Real.sin C ≤ (6 + Real.sqrt 3) / 2)) ∧
  ((0 < Real.sin A * Real.sin B * Real.sin C) ∧ 
   (Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8)) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_trig_ranges_l1761_176155


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l1761_176119

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_f_at_one :
  deriv f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l1761_176119


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l1761_176124

theorem opposite_of_fraction : 
  -(11 / 2022 : ℚ) = -11 / 2022 := by sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l1761_176124


namespace NUMINAMATH_CALUDE_dog_weight_is_ten_l1761_176160

/-- Represents the weights of a kitten, rabbit, and dog satisfying certain conditions -/
structure AnimalWeights where
  kitten : ℝ
  rabbit : ℝ
  dog : ℝ
  total_weight : kitten + rabbit + dog = 30
  kitten_rabbit_twice_dog : kitten + rabbit = 2 * dog
  kitten_dog_equals_rabbit : kitten + dog = rabbit

/-- The weight of the dog in the AnimalWeights structure is 10 pounds -/
theorem dog_weight_is_ten (w : AnimalWeights) : w.dog = 10 := by
  sorry

end NUMINAMATH_CALUDE_dog_weight_is_ten_l1761_176160


namespace NUMINAMATH_CALUDE_multiples_of_five_l1761_176149

/-- The largest number n such that there are 999 positive integers 
    between 5 and n (inclusive) that are multiples of 5 is 4995. -/
theorem multiples_of_five (n : ℕ) : 
  (∃ (k : ℕ), k = 999 ∧ 
    (∀ m : ℕ, 5 ≤ m ∧ m ≤ n ∧ m % 5 = 0 ↔ m ∈ Finset.range k)) →
  n = 4995 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_five_l1761_176149


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_sum_30_satisfies_condition_greatest_sum_is_30_l1761_176172

theorem greatest_sum_consecutive_integers (n : ℤ) : 
  (n - 1) * n * (n + 1) < 1000 → (n - 1) + n + (n + 1) ≤ 30 := by
  sorry

theorem sum_30_satisfies_condition : 
  (9 : ℤ) * 10 * 11 < 1000 ∧ 9 + 10 + 11 = 30 := by
  sorry

theorem greatest_sum_is_30 : 
  ∃ (n : ℤ), (n - 1) * n * (n + 1) < 1000 ∧ 
             (n - 1) + n + (n + 1) = 30 ∧ 
             ∀ (m : ℤ), (m - 1) * m * (m + 1) < 1000 → 
                        (m - 1) + m + (m + 1) ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_sum_30_satisfies_condition_greatest_sum_is_30_l1761_176172


namespace NUMINAMATH_CALUDE_tetrahedrons_from_cube_l1761_176130

/-- A cube has 8 vertices -/
def cube_vertices : ℕ := 8

/-- The number of tetrahedrons that can be formed using the vertices of a cube -/
def num_tetrahedrons : ℕ := 58

/-- Theorem: The number of tetrahedrons that can be formed using the vertices of a cube is 58 -/
theorem tetrahedrons_from_cube : num_tetrahedrons = 58 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedrons_from_cube_l1761_176130


namespace NUMINAMATH_CALUDE_particular_solutions_l1761_176111

/-- The differential equation -/
def diff_eq (x y y' : ℝ) : Prop :=
  x * y'^2 - 2 * y * y' + 4 * x = 0

/-- The general integral -/
def general_integral (x y C : ℝ) : Prop :=
  x^2 = C * (y - C)

/-- Theorem stating that y = 2x and y = -2x are particular solutions -/
theorem particular_solutions (x : ℝ) (hx : x > 0) :
  (diff_eq x (2*x) 2 ∧ diff_eq x (-2*x) (-2)) ∧
  (∃ C, general_integral x (2*x) C) ∧
  (∃ C, general_integral x (-2*x) C) :=
sorry

end NUMINAMATH_CALUDE_particular_solutions_l1761_176111


namespace NUMINAMATH_CALUDE_chess_club_team_probability_l1761_176159

def total_members : ℕ := 20
def num_boys : ℕ := 12
def num_girls : ℕ := 8
def team_size : ℕ := 4

theorem chess_club_team_probability :
  let total_combinations := Nat.choose total_members team_size
  let valid_combinations := 
    Nat.choose num_boys 2 * Nat.choose num_girls 2 + 
    Nat.choose num_boys 3 * Nat.choose num_girls 1 + 
    Nat.choose num_boys 4 * Nat.choose num_girls 0
  (valid_combinations : ℚ) / total_combinations = 4103 / 4845 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_team_probability_l1761_176159


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_coefficient_x_squared_is_three_l1761_176181

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 4).sum (λ k => (Nat.choose 3 k) * x^k) = 1 + 3*x + 3*x^2 + x^3 := by
  sorry

theorem coefficient_x_squared_is_three : 
  (Finset.range 4).sum (λ k => (Nat.choose 3 k) * (if k = 2 then 1 else 0)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_coefficient_x_squared_is_three_l1761_176181


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1761_176145

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/6) = 1/3) :
  Real.sin α = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1761_176145


namespace NUMINAMATH_CALUDE_stick_cutting_l1761_176108

theorem stick_cutting (n : ℕ) (a₁ a₂ a₃ : ℕ) 
  (h_pos : n > 0)
  (h_min : a₁ ≥ n ∧ a₂ ≥ n ∧ a₃ ≥ n)
  (h_sum : a₁ + a₂ + a₃ = n * (n + 1) / 2) :
  ∃ (segments : List ℕ), 
    segments.length = n ∧ 
    segments.sum = a₁ + a₂ + a₃ ∧
    (∀ i ∈ Finset.range n, (i + 1) ∈ segments) :=
by sorry

end NUMINAMATH_CALUDE_stick_cutting_l1761_176108


namespace NUMINAMATH_CALUDE_subtraction_proof_l1761_176122

theorem subtraction_proof : 6236 - 797 = 5439 := by sorry

end NUMINAMATH_CALUDE_subtraction_proof_l1761_176122


namespace NUMINAMATH_CALUDE_g_of_6_l1761_176188

def g (x : ℝ) : ℝ := 2*x^4 - 13*x^3 + 28*x^2 - 32*x - 48

theorem g_of_6 : g 6 = 552 := by
  sorry

end NUMINAMATH_CALUDE_g_of_6_l1761_176188


namespace NUMINAMATH_CALUDE_arithmetic_progression_condition_l1761_176150

theorem arithmetic_progression_condition (a b c : ℝ) : 
  (∃ d : ℝ, ∃ n k p : ℤ, b = a + d * (k - n) ∧ c = a + d * (p - n)) →
  (∃ A B : ℤ, (b - a) / (c - b) = A / B) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_condition_l1761_176150


namespace NUMINAMATH_CALUDE_erased_digit_is_four_l1761_176129

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the property that a number is divisible by 9
def divisibleBy9 (n : ℕ) : Prop := n % 9 = 0

-- Main theorem
theorem erased_digit_is_four (N : ℕ) (D : ℕ) (x : ℕ) :
  D = N - sumOfDigits N →
  divisibleBy9 D →
  sumOfDigits D = 131 + x →
  x = 4 := by sorry

end NUMINAMATH_CALUDE_erased_digit_is_four_l1761_176129


namespace NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_five_l1761_176144

/-- The number of five-digit odd numbers -/
def A : ℕ := 45000

/-- The number of five-digit multiples of 5 -/
def B : ℕ := 18000

/-- The sum of five-digit odd numbers and five-digit multiples of 5 is 63000 -/
theorem sum_of_odd_and_multiples_of_five : A + B = 63000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_five_l1761_176144


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l1761_176127

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z t : ℝ, (f x + f z) * (f y + f t) = f (x*y - z*t) + f (x*t + y*z)

/-- The main theorem stating the possible solutions. -/
theorem functional_equation_solutions (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  (∀ x, f x = 0) ∨ (∀ x, f x = 1/2) ∨ (∀ x, f x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l1761_176127


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l1761_176186

theorem inverse_proportion_ordering (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : y₃ = 2 / x₃) 
  (h4 : x₁ < x₂) 
  (h5 : x₂ < 0) 
  (h6 : 0 < x₃) : 
  y₂ < y₁ ∧ y₁ < y₃ := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l1761_176186


namespace NUMINAMATH_CALUDE_coefficient_of_x_is_negative_one_l1761_176182

-- Define the expression as a polynomial
def expression (x : ℝ) : ℝ := 5 * (x - 6) + 3 * (8 - 3 * x^2 + 7 * x) - 9 * (3 * x - 2)

-- Theorem stating that the coefficient of x in the expression is -1
theorem coefficient_of_x_is_negative_one :
  ∃ (a b c : ℝ), expression = fun x => a * x^2 + (-1) * x + c :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_is_negative_one_l1761_176182


namespace NUMINAMATH_CALUDE_abs_value_of_z_l1761_176117

/-- Given a complex number z = ((1+i)/(1-i))^2, prove that its absolute value |z| is equal to 1. -/
theorem abs_value_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.abs ((((1:ℂ) + i) / ((1:ℂ) - i))^2) = 1 := by sorry

end NUMINAMATH_CALUDE_abs_value_of_z_l1761_176117


namespace NUMINAMATH_CALUDE_x_range_l1761_176134

theorem x_range (x : ℝ) (h1 : 2 ≤ |x - 5| ∧ |x - 5| ≤ 10) (h2 : x > 0) :
  (0 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 15) := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1761_176134


namespace NUMINAMATH_CALUDE_shopkeeper_card_decks_l1761_176184

theorem shopkeeper_card_decks 
  (total_cards : ℕ) 
  (additional_cards : ℕ) 
  (cards_per_deck : ℕ) 
  (h1 : total_cards = 160)
  (h2 : additional_cards = 4)
  (h3 : cards_per_deck = 52) :
  (total_cards - additional_cards) / cards_per_deck = 3 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_card_decks_l1761_176184


namespace NUMINAMATH_CALUDE_seating_arrangements_l1761_176142

/-- Represents a row of seats -/
structure Row :=
  (total : ℕ)
  (available : ℕ)

/-- Calculates the number of seating arrangements for two people in a single row -/
def arrangementsInRow (row : Row) : ℕ :=
  row.available * (row.available - 1)

/-- Calculates the number of seating arrangements for two people in different rows -/
def arrangementsAcrossRows (row1 row2 : Row) : ℕ :=
  row1.available * row2.available * 2

/-- The main theorem stating the total number of seating arrangements -/
theorem seating_arrangements :
  let frontRow : Row := ⟨11, 8⟩
  let backRow : Row := ⟨12, 12⟩
  arrangementsInRow frontRow + arrangementsInRow backRow + arrangementsAcrossRows frontRow backRow = 334 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1761_176142


namespace NUMINAMATH_CALUDE_tangent_lines_range_l1761_176132

/-- The range of k values for which two tangent lines can be drawn from (1, 2) to the circle x^2 + y^2 + kx + 2y + k^2 - 15 = 0 -/
theorem tangent_lines_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + k*x + 2*y + k^2 - 15 = 0 ∧ 
   ∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧ 
     (∀ (x' y' : ℝ), (y' - 2 = m₁ * (x' - 1) ∨ y' - 2 = m₂ * (x' - 1)) →
       (x'^2 + y'^2 + k*x' + 2*y' + k^2 - 15 = 0 → x' = 1 ∧ y' = 2))) ↔ 
  (k ∈ Set.Ioo (-8 * Real.sqrt 3 / 3) (-3) ∪ Set.Ioo 2 (8 * Real.sqrt 3 / 3)) :=
by sorry


end NUMINAMATH_CALUDE_tangent_lines_range_l1761_176132


namespace NUMINAMATH_CALUDE_money_difference_l1761_176107

theorem money_difference (eric ben jack : ℕ) 
  (h1 : eric = ben - 10)
  (h2 : ben < 26)
  (h3 : eric + ben + 26 = 50)
  : jack - ben = 9 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l1761_176107


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1761_176161

def total_outcomes : ℕ := 6^6

def ways_to_choose_numbers : ℕ := Nat.choose 6 2

def ways_to_arrange_dice : ℕ := Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 3 * Nat.factorial 1)

def successful_outcomes : ℕ := ways_to_choose_numbers * ways_to_arrange_dice

theorem dice_roll_probability :
  (successful_outcomes : ℚ) / total_outcomes = 25 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1761_176161


namespace NUMINAMATH_CALUDE_arithmetic_segments_form_quadrilateral_l1761_176158

/-- Four segments in an arithmetic sequence with total length 3 can form a quadrilateral -/
theorem arithmetic_segments_form_quadrilateral :
  ∀ (a d : ℝ),
  a > 0 ∧ d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) = 3 →
  (a + (a + d) + (a + 2*d) > a + 3*d) ∧
  (a + (a + d) + (a + 3*d) > a + 2*d) ∧
  (a + (a + 2*d) + (a + 3*d) > a + d) ∧
  ((a + d) + (a + 2*d) + (a + 3*d) > a) :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_segments_form_quadrilateral_l1761_176158


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l1761_176112

/-- The perimeter of a semi-circle with radius r is πr + 2r -/
theorem semicircle_perimeter (r : ℝ) (hr : r > 0) : 
  let P := π * r + 2 * r
  P = π * r + 2 * r := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l1761_176112


namespace NUMINAMATH_CALUDE_office_employees_l1761_176171

theorem office_employees (total : ℝ) 
  (h1 : 0.65 * total = total * (1 - 0.35))  -- 65% of total are males
  (h2 : 0.25 * (0.65 * total) = (0.65 * total) * (1 - 0.75))  -- 25% of males are at least 50
  (h3 : 0.75 * (0.65 * total) = 3120)  -- number of males below 50
  : total = 6400 := by
sorry

end NUMINAMATH_CALUDE_office_employees_l1761_176171


namespace NUMINAMATH_CALUDE_percentage_difference_l1761_176106

theorem percentage_difference : (62 / 100 * 150) - (20 / 100 * 250) = 43 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1761_176106


namespace NUMINAMATH_CALUDE_nested_arithmetic_expression_l1761_176196

theorem nested_arithmetic_expression : 1 - (2 - (3 - 4 - (5 - 6))) = -1 := by
  sorry

end NUMINAMATH_CALUDE_nested_arithmetic_expression_l1761_176196


namespace NUMINAMATH_CALUDE_lisa_works_32_hours_l1761_176177

/-- Given Greta's work hours, Greta's hourly wage, and Lisa's hourly wage,
    calculate the number of hours Lisa needs to work to equal Greta's earnings. -/
def lisa_equal_hours (greta_hours : ℕ) (greta_wage : ℚ) (lisa_wage : ℚ) : ℚ :=
  (greta_hours : ℚ) * greta_wage / lisa_wage

/-- Prove that Lisa needs to work 32 hours to equal Greta's earnings. -/
theorem lisa_works_32_hours :
  lisa_equal_hours 40 12 15 = 32 := by
  sorry

end NUMINAMATH_CALUDE_lisa_works_32_hours_l1761_176177


namespace NUMINAMATH_CALUDE_largest_last_digit_is_two_l1761_176110

/-- A function that checks if a two-digit number is divisible by 17 or 23 -/
def isDivisibleBy17Or23 (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n % 17 = 0 ∨ n % 23 = 0)

/-- A function that represents a valid digit string according to the problem conditions -/
def isValidDigitString (s : List Nat) : Prop :=
  s.length = 1001 ∧
  s.head? = some 2 ∧
  ∀ i, i < 1000 → isDivisibleBy17Or23 (s[i]! * 10 + s[i+1]!)

/-- The theorem stating that the largest possible last digit is 2 -/
theorem largest_last_digit_is_two (s : List Nat) (h : isValidDigitString s) :
  s[1000]! ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_largest_last_digit_is_two_l1761_176110


namespace NUMINAMATH_CALUDE_truck_distance_problem_l1761_176137

/-- Proves that the initial distance between two trucks is 1025 km given the problem conditions --/
theorem truck_distance_problem (speed_A speed_B : ℝ) (extra_distance : ℝ) :
  speed_A = 90 →
  speed_B = 80 →
  extra_distance = 145 →
  ∃ (time : ℝ), 
    time > 0 ∧
    speed_A * (time + 1) = speed_B * time + extra_distance ∧
    speed_A * (time + 1) + speed_B * time = 1025 :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_problem_l1761_176137


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1761_176174

theorem arithmetic_computation : 12 + 4 * (5 - 2 * 3)^2 = 16 := by sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1761_176174


namespace NUMINAMATH_CALUDE_difference_squared_equals_negative_sixteen_l1761_176126

-- Define the given conditions
variable (a b : ℝ)
variable (h1 : a^2 + 8 > 0)  -- Ensure a^2 + 8 is positive to avoid division by zero
variable (h2 : a * b = 12)

-- State the theorem
theorem difference_squared_equals_negative_sixteen : (a - b)^2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_difference_squared_equals_negative_sixteen_l1761_176126


namespace NUMINAMATH_CALUDE_right_triangle_cos_c_l1761_176163

theorem right_triangle_cos_c (A B C : Real) (sinB : Real) :
  -- Triangle ABC exists
  -- Angle A is a right angle (90 degrees)
  A + B + C = Real.pi →
  A = Real.pi / 2 →
  -- sin B is given as 3/5
  sinB = 3 / 5 →
  -- Prove that cos C = 3/5
  Real.cos C = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_c_l1761_176163


namespace NUMINAMATH_CALUDE_gcd_490_910_l1761_176102

theorem gcd_490_910 : Nat.gcd 490 910 = 70 := by
  sorry

end NUMINAMATH_CALUDE_gcd_490_910_l1761_176102


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_of_primes_mod_240_l1761_176175

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the fourth powers of the first n prime numbers -/
def sumOfFourthPowersOfPrimes (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => (nthPrime (i + 1)) ^ 4)

/-- The main theorem -/
theorem sum_of_fourth_powers_of_primes_mod_240 :
  sumOfFourthPowersOfPrimes 2014 % 240 = 168 := by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_of_primes_mod_240_l1761_176175


namespace NUMINAMATH_CALUDE_smallest_number_with_digit_product_10_factorial_l1761_176192

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def is_valid_number (n : ℕ) : Prop :=
  digit_product n = factorial 10

theorem smallest_number_with_digit_product_10_factorial :
  ∀ n : ℕ, n < 45578899 → ¬(is_valid_number n) ∧ is_valid_number 45578899 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_digit_product_10_factorial_l1761_176192


namespace NUMINAMATH_CALUDE_mary_money_left_l1761_176146

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_spent := 2 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Theorem stating that Mary has 50 - 8q dollars left after her purchases -/
theorem mary_money_left (q : ℝ) : money_left q = 50 - 8 * q := by
  sorry

end NUMINAMATH_CALUDE_mary_money_left_l1761_176146


namespace NUMINAMATH_CALUDE_problem_statement_l1761_176113

theorem problem_statement : (3.14 - Real.pi) ^ 0 - 2 ^ (-1 : ℤ) = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1761_176113


namespace NUMINAMATH_CALUDE_exists_finite_harmonic_progression_no_infinite_harmonic_progression_l1761_176173

-- Define a harmonic progression
def IsHarmonicProgression (a : ℕ → ℕ) : Prop :=
  ∃ d : ℚ, ∀ k : ℕ, k > 0 → (1 : ℚ) / a (k + 1) - (1 : ℚ) / a k = d

-- Part (a)
theorem exists_finite_harmonic_progression (N : ℕ) :
  ∃ a : ℕ → ℕ, (∀ k : ℕ, k < N → a k < a (k + 1)) ∧ IsHarmonicProgression a :=
sorry

-- Part (b)
theorem no_infinite_harmonic_progression :
  ¬ ∃ a : ℕ → ℕ, (∀ k : ℕ, a k < a (k + 1)) ∧ IsHarmonicProgression a :=
sorry

end NUMINAMATH_CALUDE_exists_finite_harmonic_progression_no_infinite_harmonic_progression_l1761_176173


namespace NUMINAMATH_CALUDE_total_pears_picked_l1761_176136

theorem total_pears_picked (jason_pears keith_pears mike_pears sarah_pears : ℝ)
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_pears = 12)
  (h4 : sarah_pears = 32.5)
  (emma_pears : ℝ)
  (h5 : emma_pears = 2 / 3 * mike_pears)
  (james_pears : ℝ)
  (h6 : james_pears = 2 * sarah_pears - 3) :
  jason_pears + keith_pears + mike_pears + sarah_pears + emma_pears + james_pears = 207.5 := by
  sorry

#check total_pears_picked

end NUMINAMATH_CALUDE_total_pears_picked_l1761_176136


namespace NUMINAMATH_CALUDE_fold_symmetry_l1761_176169

/-- A fold on a graph paper is represented by its axis of symmetry -/
structure Fold :=
  (axis : ℝ)

/-- A point on a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Determine if two points coincide after a fold -/
def coincide (p1 p2 : Point) (f : Fold) : Prop :=
  (p1.x + p2.x) / 2 = f.axis ∧ p1.y = p2.y

/-- Find the symmetric point of a given point with respect to a fold -/
def symmetric_point (p : Point) (f : Fold) : Point :=
  { x := 2 * f.axis - p.x, y := p.y }

theorem fold_symmetry (f : Fold) (p1 p2 p3 : Point) :
  coincide p1 p2 f →
  f.axis = 3 →
  p3 = { x := -4, y := 1 } →
  symmetric_point p3 f = { x := 10, y := 1 } := by
  sorry

#check fold_symmetry

end NUMINAMATH_CALUDE_fold_symmetry_l1761_176169


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1761_176125

theorem quadratic_function_property (a m : ℝ) (h1 : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - x + a
  (f m < 0) → (f (m - 1) > 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1761_176125


namespace NUMINAMATH_CALUDE_azalea_sheep_count_l1761_176162

/-- The number of sheep Azalea sheared -/
def num_sheep : ℕ := 200

/-- The amount paid to the shearer -/
def shearer_payment : ℕ := 2000

/-- The amount of wool produced by each sheep in pounds -/
def wool_per_sheep : ℕ := 10

/-- The price of wool per pound -/
def wool_price : ℕ := 20

/-- The profit made by Azalea -/
def profit : ℕ := 38000

/-- Theorem stating that the number of sheep Azalea sheared is 200 -/
theorem azalea_sheep_count :
  num_sheep = (profit + shearer_payment) / (wool_per_sheep * wool_price) :=
by sorry

end NUMINAMATH_CALUDE_azalea_sheep_count_l1761_176162


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1761_176135

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 8, 5, -4; 3, 3, 6]
  Matrix.det A = 48 := by sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1761_176135


namespace NUMINAMATH_CALUDE_area_enclosed_is_nine_halves_l1761_176139

-- Define the constant term a
def a : ℝ := 3

-- Define the functions for the line and curve
def f (x : ℝ) : ℝ := a * x
def g (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem area_enclosed_is_nine_halves :
  ∫ x in (0)..(3), (f x - g x) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_area_enclosed_is_nine_halves_l1761_176139


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_9_l1761_176151

theorem smallest_common_multiple_of_6_and_9 : 
  ∃ n : ℕ+, (∀ m : ℕ+, 6 ∣ m ∧ 9 ∣ m → n ≤ m) ∧ 6 ∣ n ∧ 9 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_9_l1761_176151


namespace NUMINAMATH_CALUDE_f_of_x_plus_one_l1761_176176

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 4*x - 3

-- State the theorem
theorem f_of_x_plus_one (x : ℝ) : f (x + 1) = x^2 + 6*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_x_plus_one_l1761_176176


namespace NUMINAMATH_CALUDE_A_intersect_Z_l1761_176157

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

theorem A_intersect_Z : A ∩ (Set.range (Int.cast : ℤ → ℝ)) = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_Z_l1761_176157


namespace NUMINAMATH_CALUDE_tree_cutting_and_planting_l1761_176168

theorem tree_cutting_and_planting (initial_trees : ℕ) : 
  (initial_trees : ℝ) - 0.2 * initial_trees + 5 * (0.2 * initial_trees) = 720 →
  initial_trees = 400 := by
sorry

end NUMINAMATH_CALUDE_tree_cutting_and_planting_l1761_176168


namespace NUMINAMATH_CALUDE_jack_morning_emails_l1761_176193

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The difference between morning and afternoon emails -/
def email_difference : ℕ := 7

theorem jack_morning_emails :
  morning_emails = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l1761_176193


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1761_176185

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence with S_2 = 4 and S_4 = 20, the common difference is 3 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h2 : seq.S 2 = 4)
  (h4 : seq.S 4 = 20) :
  seq.d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1761_176185


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l1761_176140

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) :
  total_children = 740 →
  absent_children = 370 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ),
    initial_bananas * total_children = (initial_bananas + extra_bananas) * (total_children - absent_children) ∧
    initial_bananas = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l1761_176140


namespace NUMINAMATH_CALUDE_triangle_inequality_l1761_176116

theorem triangle_inequality (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_a_geq_b : a ≥ b)
  (h_a_geq_c : a ≥ c)
  (h_sum1 : a + b - c > 0)
  (h_sum2 : b + c - a > 0) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1761_176116


namespace NUMINAMATH_CALUDE_simplify_expression_l1761_176170

theorem simplify_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = 3*(x + y)) : 
  x/y + y/x - 3/(x*y) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1761_176170


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1761_176156

/-- Given that B's current age is 42 years and A is 12 years older than B,
    prove that the ratio of A's age in 10 years to B's age 10 years ago is 2:1 -/
theorem age_ratio_proof (B_current : ℕ) (A_current : ℕ) : 
  B_current = 42 →
  A_current = B_current + 12 →
  (A_current + 10) / (B_current - 10) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1761_176156


namespace NUMINAMATH_CALUDE_jack_emails_l1761_176103

theorem jack_emails (morning_emails : ℕ) (afternoon_emails : ℕ) 
  (h1 : morning_emails = 6)
  (h2 : afternoon_emails = morning_emails + 2) : 
  afternoon_emails = 8 := by
sorry

end NUMINAMATH_CALUDE_jack_emails_l1761_176103


namespace NUMINAMATH_CALUDE_tim_nickels_count_l1761_176123

/-- The number of nickels Tim had initially -/
def initial_nickels : ℕ := 9

/-- The number of nickels Tim received from his dad -/
def received_nickels : ℕ := 3

/-- The total number of nickels Tim has after receiving coins from his dad -/
def total_nickels : ℕ := initial_nickels + received_nickels

theorem tim_nickels_count : total_nickels = 12 := by
  sorry

end NUMINAMATH_CALUDE_tim_nickels_count_l1761_176123


namespace NUMINAMATH_CALUDE_exponents_in_30_factorial_l1761_176190

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def exponent_in_factorial (p n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1) / p) 0

theorem exponents_in_30_factorial :
  exponent_in_factorial 2 30 = 26 ∧ exponent_in_factorial 5 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_exponents_in_30_factorial_l1761_176190


namespace NUMINAMATH_CALUDE_correct_calculation_l1761_176128

theorem correct_calculation : 
  (2 * Real.sqrt 5 + 3 * Real.sqrt 5 = 5 * Real.sqrt 5) ∧ 
  (Real.sqrt 8 ≠ 2) ∧ 
  (Real.sqrt ((-3)^2) ≠ -3) ∧ 
  ((Real.sqrt 2 + 1)^2 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l1761_176128


namespace NUMINAMATH_CALUDE_rocky_knockout_percentage_l1761_176167

/-- Proves that the percentage of Rocky's knockouts that were in the first round is 20% -/
theorem rocky_knockout_percentage : 
  ∀ (total_fights : ℕ) 
    (knockout_percentage : ℚ) 
    (first_round_knockouts : ℕ),
  total_fights = 190 →
  knockout_percentage = 1/2 →
  first_round_knockouts = 19 →
  (first_round_knockouts : ℚ) / (knockout_percentage * total_fights) = 1/5 := by
sorry

end NUMINAMATH_CALUDE_rocky_knockout_percentage_l1761_176167


namespace NUMINAMATH_CALUDE_concrete_mixture_problem_l1761_176153

theorem concrete_mixture_problem (total_weight : Real) (final_cement_percentage : Real) 
  (amount_each_type : Real) (h1 : total_weight = 4500) 
  (h2 : final_cement_percentage = 10.8) (h3 : amount_each_type = 1125) :
  ∃ (first_type_percentage : Real),
    first_type_percentage = 2 ∧
    (amount_each_type * first_type_percentage / 100 + 
     amount_each_type * (2 * final_cement_percentage - first_type_percentage) / 100) = 
    (total_weight * final_cement_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_concrete_mixture_problem_l1761_176153


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1761_176133

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {x | Real.log (x + 1) > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1761_176133


namespace NUMINAMATH_CALUDE_negative_two_times_inequality_l1761_176114

theorem negative_two_times_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_times_inequality_l1761_176114


namespace NUMINAMATH_CALUDE_right_triangle_arc_segment_l1761_176166

theorem right_triangle_arc_segment (AC CB : ℝ) (h_AC : AC = 15) (h_CB : CB = 8) :
  let AB := Real.sqrt (AC^2 + CB^2)
  let CP := (AC * CB) / AB
  let PB := Real.sqrt (CB^2 - CP^2)
  let BD := 2 * PB
  BD = 128 / 17 := by sorry

end NUMINAMATH_CALUDE_right_triangle_arc_segment_l1761_176166


namespace NUMINAMATH_CALUDE_monotonicity_for_a_2_non_monotonicity_condition_l1761_176148

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1/2 * x^2 - (1 + a) * x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := a / x + x - (1 + a)

theorem monotonicity_for_a_2 :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f 2 x₁ < f 2 x₂ ∧
  ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f 2 x₁ > f 2 x₂ ∧
  ∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f 2 x₁ < f 2 x₂ :=
sorry

theorem non_monotonicity_condition :
  ∀ a, (∃ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ f a x₁ < f a x₂ ∧
       ∃ y₁ y₂, 1 < y₁ ∧ y₁ < y₂ ∧ y₂ < 2 ∧ f a y₁ > f a y₂) ↔
  1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_for_a_2_non_monotonicity_condition_l1761_176148


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1761_176105

theorem expand_and_simplify (x : ℝ) : (x + 2) * (x - 2) - x * (x + 1) = -x - 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1761_176105


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l1761_176179

/-- Define ω as a complex number satisfying the properties of a cube root of unity -/
def ω : ℂ := sorry

/-- ω is a cube root of unity -/
axiom ω_cubed : ω^3 = 1

/-- ω satisfies the equation ω^2 + ω + 1 = 0 -/
axiom ω_sum : ω^2 + ω + 1 = 0

/-- Theorem: ω^9 + (ω^2)^9 = 2 -/
theorem cube_root_unity_sum : ω^9 + (ω^2)^9 = 2 := by sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l1761_176179


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1761_176104

theorem quadratic_inequality_range (m : ℝ) : 
  (¬ ∃ x : ℝ, (m + 1) * x^2 - (m + 1) * x + 1 ≤ 0) ↔ 
  (m ≥ -1 ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1761_176104


namespace NUMINAMATH_CALUDE_triangle_sine_product_inequality_l1761_176131

theorem triangle_sine_product_inequality (A B C : ℝ) (h_triangle : A + B + C = π) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_product_inequality_l1761_176131


namespace NUMINAMATH_CALUDE_inverse_congruence_solution_l1761_176164

theorem inverse_congruence_solution (p : ℕ) (a : ℤ) (hp : Nat.Prime p) :
  (∃ x : ℤ, (a * x) % p = 1 ∧ x % p = a % p) ↔ (a % p = 1 ∨ a % p = p - 1) := by
  sorry

end NUMINAMATH_CALUDE_inverse_congruence_solution_l1761_176164


namespace NUMINAMATH_CALUDE_skittles_transfer_l1761_176199

theorem skittles_transfer (bridget_initial : ℕ) (henry_initial : ℕ) : 
  bridget_initial = 4 → henry_initial = 4 → bridget_initial + henry_initial = 8 := by
sorry

end NUMINAMATH_CALUDE_skittles_transfer_l1761_176199


namespace NUMINAMATH_CALUDE_gold_cube_value_scaling_l1761_176154

/-- Represents the properties of a gold cube -/
structure GoldCube where
  side_length : ℝ
  value : ℝ

/-- Theorem stating the relationship between two gold cubes of different sizes -/
theorem gold_cube_value_scaling (small_cube large_cube : GoldCube) :
  small_cube.side_length = 4 →
  small_cube.value = 800 →
  large_cube.side_length = 6 →
  large_cube.value = 2700 := by
  sorry

#check gold_cube_value_scaling

end NUMINAMATH_CALUDE_gold_cube_value_scaling_l1761_176154


namespace NUMINAMATH_CALUDE_kozlov_inequality_l1761_176118

theorem kozlov_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : 
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 
  2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_kozlov_inequality_l1761_176118


namespace NUMINAMATH_CALUDE_solid_with_isosceles_triangle_views_is_tetrahedron_l1761_176189

/-- A solid object in 3D space -/
structure Solid where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a view (projection) of a solid -/
inductive View
  | Front
  | Top
  | Side

/-- Represents the shape of a view -/
inductive Shape
  | IsoscelesTriangle
  | Other

/-- Function to get the shape of a view for a given solid -/
def viewShape (s : Solid) (v : View) : Shape :=
  sorry -- Implementation details

/-- Predicate to check if a solid is a tetrahedron -/
def isTetrahedron (s : Solid) : Prop :=
  sorry -- Definition of a tetrahedron

/-- Theorem: If all three views of a solid are isosceles triangles, then it's a tetrahedron -/
theorem solid_with_isosceles_triangle_views_is_tetrahedron (s : Solid) :
  (∀ v : View, viewShape s v = Shape.IsoscelesTriangle) →
  isTetrahedron s :=
sorry

end NUMINAMATH_CALUDE_solid_with_isosceles_triangle_views_is_tetrahedron_l1761_176189


namespace NUMINAMATH_CALUDE_fourth_coaster_speed_l1761_176165

/-- Given 5 rollercoasters with known speeds for 4 of them and a known average speed for all 5,
    prove that the speed of the unknown coaster is equal to the total speed (based on the average)
    minus the sum of the known speeds. -/
theorem fourth_coaster_speed
  (speed1 speed2 speed3 speed5 : ℝ)
  (average_speed : ℝ)
  (h1 : speed1 = 50)
  (h2 : speed2 = 62)
  (h3 : speed3 = 73)
  (h5 : speed5 = 40)
  (h_avg : average_speed = 59)
  : ∃ speed4 : ℝ,
    speed4 = 5 * average_speed - (speed1 + speed2 + speed3 + speed5) :=
by sorry

end NUMINAMATH_CALUDE_fourth_coaster_speed_l1761_176165


namespace NUMINAMATH_CALUDE_willow_peach_tree_count_l1761_176100

/-- Represents the dimensions of a rectangular playground -/
structure Playground where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangular playground -/
def perimeter (p : Playground) : ℕ := 2 * (p.length + p.width)

/-- Represents the spacing between trees -/
def treeSpacing : ℕ := 10

/-- Calculates the total number of tree positions along the perimeter -/
def totalTreePositions (p : Playground) : ℕ := perimeter p / treeSpacing

/-- Theorem: The number of willow trees (or peach trees) is half of the total tree positions -/
theorem willow_peach_tree_count (p : Playground) (h1 : p.length = 150) (h2 : p.width = 60) :
  totalTreePositions p / 2 = 21 := by
  sorry

#check willow_peach_tree_count

end NUMINAMATH_CALUDE_willow_peach_tree_count_l1761_176100
