import Mathlib

namespace NUMINAMATH_CALUDE_success_arrangements_eq_420_l861_86115

/-- The number of letters in the word "SUCCESS" -/
def word_length : ℕ := 7

/-- The number of occurrences of the letter 'S' in "SUCCESS" -/
def s_count : ℕ := 3

/-- The number of occurrences of the letter 'C' in "SUCCESS" -/
def c_count : ℕ := 2

/-- The number of unique arrangements of the letters in "SUCCESS" -/
def success_arrangements : ℕ := word_length.factorial / (s_count.factorial * c_count.factorial)

theorem success_arrangements_eq_420 : success_arrangements = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_arrangements_eq_420_l861_86115


namespace NUMINAMATH_CALUDE_collinearity_necessary_not_sufficient_l861_86157

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

theorem collinearity_necessary_not_sufficient :
  ∃ a : ℝ, collinear (a, a^2) (1, 2) ∧ a ≠ 2 ∧
  (∀ a : ℝ, a = 2 → collinear (a, a^2) (1, 2)) :=
sorry

end NUMINAMATH_CALUDE_collinearity_necessary_not_sufficient_l861_86157


namespace NUMINAMATH_CALUDE_proof_arrangements_l861_86106

/-- The number of letters in the word PROOF -/
def word_length : ℕ := 5

/-- The number of times the letter 'O' appears in PROOF -/
def o_count : ℕ := 2

/-- Formula for calculating the number of arrangements -/
def arrangements (n : ℕ) (k : ℕ) : ℕ := n.factorial / k.factorial

/-- Theorem stating that the number of unique arrangements of PROOF is 60 -/
theorem proof_arrangements : arrangements word_length o_count = 60 := by
  sorry

end NUMINAMATH_CALUDE_proof_arrangements_l861_86106


namespace NUMINAMATH_CALUDE_total_football_games_l861_86159

/-- Calculates the total number of football games in a season -/
theorem total_football_games 
  (games_per_month : ℝ) 
  (season_duration : ℝ) 
  (h1 : games_per_month = 323.0)
  (h2 : season_duration = 17.0) :
  games_per_month * season_duration = 5491.0 := by
  sorry

end NUMINAMATH_CALUDE_total_football_games_l861_86159


namespace NUMINAMATH_CALUDE_coterminal_angle_proof_l861_86167

/-- The angle (in degrees) that is coterminal with -415° and lies between 0° and 360° -/
def coterminal_angle : ℝ := 305

theorem coterminal_angle_proof : 
  0 ≤ coterminal_angle ∧ 
  coterminal_angle < 360 ∧ 
  ∃ k : ℤ, coterminal_angle = k * 360 - 415 := by
  sorry

#check coterminal_angle_proof

end NUMINAMATH_CALUDE_coterminal_angle_proof_l861_86167


namespace NUMINAMATH_CALUDE_transformed_quadratic_roots_l861_86100

theorem transformed_quadratic_roots (α β : ℂ) : 
  (3 * α^2 + 2 * α + 1 = 0) → 
  (3 * β^2 + 2 * β + 1 = 0) → 
  ((3 * α + 2)^2 + 4 = 0) ∧ ((3 * β + 2)^2 + 4 = 0) := by
sorry

end NUMINAMATH_CALUDE_transformed_quadratic_roots_l861_86100


namespace NUMINAMATH_CALUDE_product_1_to_30_trailing_zeros_l861_86123

/-- The number of trailing zeros in the product of integers from 1 to n -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem stating that the product of integers from 1 to 30 has 7 trailing zeros -/
theorem product_1_to_30_trailing_zeros :
  trailingZeros 30 = 7 := by sorry

end NUMINAMATH_CALUDE_product_1_to_30_trailing_zeros_l861_86123


namespace NUMINAMATH_CALUDE_divisibility_and_sum_of_primes_l861_86199

theorem divisibility_and_sum_of_primes :
  ∃ (p₁ p₂ p₃ : ℕ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    (p₁ ∣ (2^10 - 1)) ∧ (p₂ ∣ (2^10 - 1)) ∧ (p₃ ∣ (2^10 - 1)) ∧
    (∀ q : ℕ, Prime q → (q ∣ (2^10 - 1)) → (q = p₁ ∨ q = p₂ ∨ q = p₃)) ∧
    p₁ + p₂ + p₃ = 45 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_and_sum_of_primes_l861_86199


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l861_86104

noncomputable def x (θ : Real) : Real := |Real.sin (θ / 2) + Real.cos (θ / 2)|
noncomputable def y (θ : Real) : Real := 1 + Real.sin θ

theorem parametric_to_ordinary_equation :
  ∀ θ : Real, 0 ≤ θ ∧ θ < 2 * Real.pi →
  ∃ x_val y_val : Real,
    x θ = x_val ∧
    y θ = y_val ∧
    x_val ^ 2 = y_val ∧
    0 ≤ x_val ∧ x_val ≤ Real.sqrt 2 ∧
    0 ≤ y_val ∧ y_val ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l861_86104


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l861_86108

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l861_86108


namespace NUMINAMATH_CALUDE_rationalize_denominator_l861_86129

theorem rationalize_denominator :
  Real.sqrt (5 / (2 + Real.sqrt 2)) = Real.sqrt 5 - Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l861_86129


namespace NUMINAMATH_CALUDE_fraction_sum_product_equality_l861_86177

theorem fraction_sum_product_equality (x y : ℤ) :
  (19 : ℚ) / x + (96 : ℚ) / y = ((19 : ℚ) / x) * ((96 : ℚ) / y) →
  ∃ m : ℤ, x = 19 * m ∧ y = 96 - 96 * m :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_product_equality_l861_86177


namespace NUMINAMATH_CALUDE_equilateral_triangle_exists_l861_86146

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  vertex : Point
  base1 : Point
  base2 : Point

-- Define a function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to check if a triangle is equilateral
def isEquilateral (t : EquilateralTriangle) : Prop :=
  let d1 := ((t.vertex.x - t.base1.x)^2 + (t.vertex.y - t.base1.y)^2)
  let d2 := ((t.vertex.x - t.base2.x)^2 + (t.vertex.y - t.base2.y)^2)
  let d3 := ((t.base1.x - t.base2.x)^2 + (t.base1.y - t.base2.y)^2)
  d1 = d2 ∧ d2 = d3

-- Theorem statement
theorem equilateral_triangle_exists (P : Point) (l : Line) :
  ∃ (t : EquilateralTriangle), t.vertex = P ∧ 
    isPointOnLine t.base1 l ∧ isPointOnLine t.base2 l ∧ 
    isEquilateral t :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_exists_l861_86146


namespace NUMINAMATH_CALUDE_coordinates_uniquely_determine_position_l861_86125

-- Define a structure for geographical coordinates
structure GeoCoord where
  longitude : Real
  latitude : Real

-- Define a type for position descriptors
inductive PositionDescriptor
  | Distance (d : Real) (reference : String)
  | RoadName (name : String)
  | Coordinates (coord : GeoCoord)
  | Direction (angle : Real) (reference : String)

-- Function to check if a descriptor uniquely determines a position
def uniquelyDeterminesPosition (descriptor : PositionDescriptor) : Prop :=
  match descriptor with
  | PositionDescriptor.Coordinates _ => True
  | _ => False

-- Theorem stating that only coordinates uniquely determine a position
theorem coordinates_uniquely_determine_position
  (descriptor : PositionDescriptor) :
  uniquelyDeterminesPosition descriptor ↔
  ∃ (coord : GeoCoord), descriptor = PositionDescriptor.Coordinates coord :=
sorry

#check coordinates_uniquely_determine_position

end NUMINAMATH_CALUDE_coordinates_uniquely_determine_position_l861_86125


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_negative_a_l861_86107

theorem tangent_perpendicular_implies_negative_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 * a * x^2 + 1 / x = 0)) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_negative_a_l861_86107


namespace NUMINAMATH_CALUDE_min_value_of_expression_l861_86122

theorem min_value_of_expression (x y : ℝ) : (x + y - 1)^2 + (x*y)^2 ≥ 0 ∧ ∃ a b : ℝ, (a + b - 1)^2 + (a*b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l861_86122


namespace NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l861_86135

/-- Represents a five-digit number in the form AB,CBA -/
def abcba (a b c : Nat) : Nat := 10000 * a + 1000 * b + 100 * c + 10 * b + a

/-- Check if three digits are distinct -/
def distinct_digits (a b c : Nat) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem greatest_abcba_divisible_by_13 :
  ∀ a b c : Nat,
  a < 10 → b < 10 → c < 10 →
  distinct_digits a b c →
  abcba a b c ≤ 99999 →
  abcba a b c ≡ 0 [MOD 13] →
  abcba a b c ≤ 95159 :=
sorry

end NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l861_86135


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_sum_l861_86140

/-- Given two parallel vectors a and b, prove that tan(α + π/4) = 7 --/
theorem parallel_vectors_tan_sum (α : ℝ) : 
  let a : ℝ × ℝ := (3, 4)
  let b : ℝ × ℝ := (Real.sin α, Real.cos α)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) →  -- Parallel vectors condition
  Real.tan (α + π/4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_sum_l861_86140


namespace NUMINAMATH_CALUDE_laundry_loads_count_l861_86165

theorem laundry_loads_count :
  let wash_time : ℚ := 45 / 60  -- wash time in hours
  let dry_time : ℚ := 1  -- dry time in hours
  let total_time : ℚ := 14  -- total time in hours
  let load_time : ℚ := wash_time + dry_time  -- time per load in hours
  ∃ (loads : ℕ), (loads : ℚ) * load_time = total_time ∧ loads = 8
  := by sorry

end NUMINAMATH_CALUDE_laundry_loads_count_l861_86165


namespace NUMINAMATH_CALUDE_subset_implies_a_value_l861_86198

def A : Set ℤ := {0, 1}
def B (a : ℤ) : Set ℤ := {-1, 0, a+3}

theorem subset_implies_a_value (h : A ⊆ B a) : a = -2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_value_l861_86198


namespace NUMINAMATH_CALUDE_perfect_square_condition_l861_86102

theorem perfect_square_condition (Z K : ℤ) : 
  (1000 < Z) → (Z < 5000) → (K > 1) → (Z = K * K^2) → 
  (∃ (n : ℤ), Z = n^2) → (K = 16) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l861_86102


namespace NUMINAMATH_CALUDE_remainder_of_product_divided_by_11_l861_86156

theorem remainder_of_product_divided_by_11 : (108 * 110) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_divided_by_11_l861_86156


namespace NUMINAMATH_CALUDE_range_of_a_l861_86174

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then 3^x + 4*a else 2*x + a^2

theorem range_of_a (a : ℝ) (h₁ : a > 0) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l861_86174


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l861_86126

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_second_term
  (a : ℕ → ℚ)
  (h_geometric : GeometricSequence a)
  (h_fifth : a 5 = 48)
  (h_sixth : a 6 = 72) :
  a 2 = 128 / 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l861_86126


namespace NUMINAMATH_CALUDE_bacteria_at_8_20_am_l861_86169

/-- Calculates the bacterial population after a given time period -/
def bacterial_population (initial_population : ℕ) (doubling_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  initial_population * (2 ^ (elapsed_time / doubling_time))

/-- Theorem stating the bacterial population at 8:20 AM -/
theorem bacteria_at_8_20_am : 
  let initial_population : ℕ := 30
  let doubling_time : ℕ := 4  -- in minutes
  let elapsed_time : ℕ := 20  -- in minutes
  bacterial_population initial_population doubling_time elapsed_time = 960 :=
by
  sorry


end NUMINAMATH_CALUDE_bacteria_at_8_20_am_l861_86169


namespace NUMINAMATH_CALUDE_binary_representation_of_37_l861_86105

/-- Converts a natural number to its binary representation as a list of booleans -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The binary representation of 37 -/
def binary37 : List Bool := [true, false, true, false, false, true]

/-- Theorem stating that the binary representation of 37 is [true, false, true, false, false, true] -/
theorem binary_representation_of_37 : toBinary 37 = binary37 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_37_l861_86105


namespace NUMINAMATH_CALUDE_sqrt_2_plus_x_real_range_l861_86131

theorem sqrt_2_plus_x_real_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 + x) ↔ x ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_plus_x_real_range_l861_86131


namespace NUMINAMATH_CALUDE_mixture_salt_concentration_l861_86134

/-- Represents the concentration of a solution as a real number between 0 and 1 -/
def Concentration := { c : ℝ // 0 ≤ c ∧ c ≤ 1 }

/-- Calculates the concentration of salt in a mixture of pure water and salt solution -/
def mixtureSaltConcentration (pureWaterVolume : ℝ) (saltSolutionVolume : ℝ) (saltSolutionConcentration : Concentration) : Concentration :=
  sorry

/-- Theorem: The concentration of salt in a mixture of 1 liter of pure water and 0.2 liters of 60% salt solution is 10% -/
theorem mixture_salt_concentration :
  let pureWaterVolume : ℝ := 1
  let saltSolutionVolume : ℝ := 0.2
  let saltSolutionConcentration : Concentration := ⟨0.6, by sorry⟩
  let resultingConcentration : Concentration := mixtureSaltConcentration pureWaterVolume saltSolutionVolume saltSolutionConcentration
  resultingConcentration.val = 0.1 := by sorry

end NUMINAMATH_CALUDE_mixture_salt_concentration_l861_86134


namespace NUMINAMATH_CALUDE_smartphone_cost_l861_86172

theorem smartphone_cost (selling_price : ℝ) (loss_percentage : ℝ) (initial_cost : ℝ) : 
  selling_price = 255 ∧ 
  loss_percentage = 15 ∧ 
  selling_price = initial_cost * (1 - loss_percentage / 100) →
  initial_cost = 300 :=
by sorry

end NUMINAMATH_CALUDE_smartphone_cost_l861_86172


namespace NUMINAMATH_CALUDE_angle_triple_complement_l861_86191

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l861_86191


namespace NUMINAMATH_CALUDE_quadrilateral_area_l861_86109

/-- Quadrilateral PQRS with given side lengths -/
structure Quadrilateral :=
  (PS : ℝ)
  (SR : ℝ)
  (PQ : ℝ)
  (RQ : ℝ)

/-- The area of the quadrilateral PQRS is 36 -/
theorem quadrilateral_area (q : Quadrilateral) 
  (h1 : q.PS = 3)
  (h2 : q.SR = 4)
  (h3 : q.PQ = 13)
  (h4 : q.RQ = 12) : 
  ∃ (area : ℝ), area = 36 := by
  sorry

#check quadrilateral_area

end NUMINAMATH_CALUDE_quadrilateral_area_l861_86109


namespace NUMINAMATH_CALUDE_office_employees_count_l861_86136

theorem office_employees_count (men women : ℕ) : 
  men = women →
  6 = women / 5 →
  men + women = 60 :=
by sorry

end NUMINAMATH_CALUDE_office_employees_count_l861_86136


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l861_86195

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 8) :
  ∀ z : ℝ, z = x + 2*y → z ≥ 4 ∧ ∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ x' + 2*y' + 2*x'*y' = 8 ∧ x' + 2*y' = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l861_86195


namespace NUMINAMATH_CALUDE_shortest_side_of_triangle_l861_86113

theorem shortest_side_of_triangle (a b c : ℕ) (area : ℕ) : 
  a = 18 ∧ 
  a + b + c = 42 ∧ 
  (a ≤ b ∧ a ≤ c) ∧
  area * area = (21 * (21 - a) * (21 - b) * (21 - c)) →
  a = 5 :=
by sorry

end NUMINAMATH_CALUDE_shortest_side_of_triangle_l861_86113


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1980_l861_86141

theorem largest_perfect_square_factor_of_1980 : 
  ∃ (n : ℕ), n^2 = 36 ∧ n^2 ∣ 1980 ∧ ∀ (m : ℕ), m^2 ∣ 1980 → m^2 ≤ 36 := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1980_l861_86141


namespace NUMINAMATH_CALUDE_max_b_in_box_l861_86183

/-- Given a rectangular box with volume 360 cubic units and integer dimensions a, b, and c 
    where a > b > c > 2, the maximum value of b is 10. -/
theorem max_b_in_box (a b c : ℕ) : 
  a * b * c = 360 →
  a > b →
  b > c →
  c > 2 →
  b ≤ 10 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ a' > b' ∧ b' > c' ∧ c' > 2 ∧ b' = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_b_in_box_l861_86183


namespace NUMINAMATH_CALUDE_quadratic_always_has_two_roots_find_m_value_l861_86179

/-- Given quadratic equation x^2 - (2m+1)x + m - 2 = 0 -/
def quadratic_equation (m x : ℝ) : Prop :=
  x^2 - (2*m+1)*x + m - 2 = 0

theorem quadratic_always_has_two_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ :=
sorry

theorem find_m_value :
  ∃ m : ℝ, m = 6/5 ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic_equation m x₁ ∧ 
    quadratic_equation m x₂ ∧
    x₁ + x₂ + 3*x₁*x₂ = 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_has_two_roots_find_m_value_l861_86179


namespace NUMINAMATH_CALUDE_josie_money_left_l861_86197

/-- Calculates the amount of money Josie has left after grocery shopping --/
def money_left_after_shopping (initial_amount : ℚ) (milk_price : ℚ) (bread_price : ℚ) 
  (detergent_price : ℚ) (banana_price_per_pound : ℚ) (banana_pounds : ℚ) 
  (milk_discount : ℚ) (detergent_discount : ℚ) : ℚ :=
  let milk_cost := milk_price * (1 - milk_discount)
  let detergent_cost := detergent_price - detergent_discount
  let banana_cost := banana_price_per_pound * banana_pounds
  let total_cost := milk_cost + bread_price + detergent_cost + banana_cost
  initial_amount - total_cost

/-- Theorem stating that Josie has $4.00 left after shopping --/
theorem josie_money_left : 
  money_left_after_shopping 20 4 3.5 10.25 0.75 2 0.5 1.25 = 4 := by
  sorry

end NUMINAMATH_CALUDE_josie_money_left_l861_86197


namespace NUMINAMATH_CALUDE_vet_donation_calculation_l861_86139

/-- Represents the vet fees for different animals --/
structure VetFees where
  dog : ℝ
  cat : ℝ
  rabbit : ℝ
  parrot : ℝ

/-- Represents the number of adoptions for each animal type --/
structure Adoptions where
  dogs : ℕ
  cats : ℕ
  rabbits : ℕ
  parrots : ℕ

/-- Calculates the total vet fees with discounts applied --/
def calculateTotalFees (fees : VetFees) (adoptions : Adoptions) (multiAdoptDiscount : ℝ) 
    (dogCatAdoptions : ℕ) (parrotRabbitAdoptions : ℕ) : ℝ := sorry

/-- Calculates the vet's donation based on the total fees --/
def calculateDonation (totalFees : ℝ) (donationRate : ℝ) : ℝ := sorry

theorem vet_donation_calculation (fees : VetFees) (adoptions : Adoptions) 
    (multiAdoptDiscount : ℝ) (dogCatAdoptions : ℕ) (parrotRabbitAdoptions : ℕ) 
    (donationRate : ℝ) :
  fees.dog = 15 ∧ fees.cat = 13 ∧ fees.rabbit = 10 ∧ fees.parrot = 12 ∧
  adoptions.dogs = 8 ∧ adoptions.cats = 3 ∧ adoptions.rabbits = 5 ∧ adoptions.parrots = 2 ∧
  multiAdoptDiscount = 0.1 ∧ dogCatAdoptions = 2 ∧ parrotRabbitAdoptions = 1 ∧
  donationRate = 1/3 →
  calculateDonation (calculateTotalFees fees adoptions multiAdoptDiscount dogCatAdoptions parrotRabbitAdoptions) donationRate = 54.27 := by
  sorry

end NUMINAMATH_CALUDE_vet_donation_calculation_l861_86139


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l861_86188

/-- A round-robin tournament where each player plays every other player exactly once. -/
structure RoundRobinTournament where
  num_players : ℕ
  num_players_pos : 0 < num_players

/-- The number of matches in a round-robin tournament. -/
def num_matches (t : RoundRobinTournament) : ℕ := t.num_players.choose 2

theorem ten_player_tournament_matches :
  ∀ t : RoundRobinTournament, t.num_players = 10 → num_matches t = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_matches_l861_86188


namespace NUMINAMATH_CALUDE_factorization_problems_l861_86120

theorem factorization_problems (m a x : ℝ) : 
  (9 * m^2 - 4 = (3 * m + 2) * (3 * m - 2)) ∧ 
  (2 * a * x^2 + 12 * a * x + 18 * a = 2 * a * (x + 3)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l861_86120


namespace NUMINAMATH_CALUDE_ellipse_constraint_l861_86150

/-- An ellipse passing through (2,1) with |y| > 1 -/
def EllipseWithConstraint (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (4 / a^2 + 1 / b^2 = 1)

theorem ellipse_constraint (a b : ℝ) (h : EllipseWithConstraint a b) :
  {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 ∧ |p.2| > 1} =
  {p : ℝ × ℝ | p.1^2 + p.2^2 < 5 ∧ |p.2| > 1} := by
  sorry

end NUMINAMATH_CALUDE_ellipse_constraint_l861_86150


namespace NUMINAMATH_CALUDE_function_inequality_condition_l861_86152

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = 4 * x + 3) →
  a > 0 →
  b > 0 →
  (∀ x : ℝ, |x + 3| < b → |f x + 5| < a) ↔
  b ≤ a / 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l861_86152


namespace NUMINAMATH_CALUDE_football_games_this_year_l861_86116

theorem football_games_this_year 
  (total_games : ℕ) 
  (last_year_games : ℕ) 
  (h1 : total_games = 9)
  (h2 : last_year_games = 5) :
  total_games - last_year_games = 4 :=
by sorry

end NUMINAMATH_CALUDE_football_games_this_year_l861_86116


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l861_86185

theorem other_solution_quadratic (x : ℚ) : 
  (48 * (3/4)^2 + 25 = 77 * (3/4) + 4) → 
  (48 * x^2 + 25 = 77 * x + 4) → 
  x = 3/4 ∨ x = 7/12 := by
sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l861_86185


namespace NUMINAMATH_CALUDE_ab_value_l861_86127

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 11) : a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l861_86127


namespace NUMINAMATH_CALUDE_trigonometric_identity_l861_86149

theorem trigonometric_identity : 
  Real.sin (37 * π / 180) * Real.cos (34 * π / 180)^2 + 
  2 * Real.sin (34 * π / 180) * Real.cos (37 * π / 180) * Real.cos (34 * π / 180) - 
  Real.sin (37 * π / 180) * Real.sin (34 * π / 180)^2 = 
  (Real.sqrt 6 + Real.sqrt 2) / 4 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l861_86149


namespace NUMINAMATH_CALUDE_ball_bearing_sale_price_l861_86182

/-- The sale price of ball bearings that satisfies the given conditions -/
def sale_price : ℝ := 0.75

theorem ball_bearing_sale_price :
  let num_machines : ℕ := 10
  let bearings_per_machine : ℕ := 30
  let normal_price : ℝ := 1
  let bulk_discount : ℝ := 0.2
  let total_savings : ℝ := 120
  
  let total_bearings : ℕ := num_machines * bearings_per_machine
  let normal_total_cost : ℝ := total_bearings * normal_price
  let sale_total_cost : ℝ := total_bearings * sale_price * (1 - bulk_discount)
  
  normal_total_cost - sale_total_cost = total_savings :=
by sorry

end NUMINAMATH_CALUDE_ball_bearing_sale_price_l861_86182


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l861_86178

/-- The vertex coordinates of a parabola in the form y = -(x + h)^2 + k are (h, k) -/
theorem parabola_vertex_coordinates (h k : ℝ) :
  let f : ℝ → ℝ := λ x => -(x + h)^2 + k
  (∀ x, f x = -(x + h)^2 + k) →
  (h, k) = Prod.mk (- h) k :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l861_86178


namespace NUMINAMATH_CALUDE_triangle_problem_l861_86189

theorem triangle_problem (a b c : ℝ) 
  (h : |a - Real.sqrt 7| + Real.sqrt (b - 5) + (c - 4 * Real.sqrt 2)^2 = 0) :
  a = Real.sqrt 7 ∧ b = 5 ∧ c = 4 * Real.sqrt 2 ∧
  ∃ (x y z : ℝ), x^2 + y^2 = z^2 ∧ 
  Set.toFinset {x, y, z} = Set.toFinset {a, b, c} :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l861_86189


namespace NUMINAMATH_CALUDE_pet_store_cages_l861_86171

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 102)
  (h2 : sold_puppies = 21)
  (h3 : puppies_per_cage = 9) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 9 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l861_86171


namespace NUMINAMATH_CALUDE_lunch_costs_more_than_breakfast_l861_86124

/-- Represents the cost of Anna's meals -/
structure MealCosts where
  bagel : ℝ
  orange_juice : ℝ
  sandwich : ℝ
  milk : ℝ

/-- Calculates the difference between lunch and breakfast costs -/
def lunch_breakfast_difference (costs : MealCosts) : ℝ :=
  (costs.sandwich + costs.milk) - (costs.bagel + costs.orange_juice)

/-- Theorem stating the difference between lunch and breakfast costs -/
theorem lunch_costs_more_than_breakfast (costs : MealCosts) 
  (h1 : costs.bagel = 0.95)
  (h2 : costs.orange_juice = 0.85)
  (h3 : costs.sandwich = 4.65)
  (h4 : costs.milk = 1.15) :
  lunch_breakfast_difference costs = 4.00 := by
  sorry

end NUMINAMATH_CALUDE_lunch_costs_more_than_breakfast_l861_86124


namespace NUMINAMATH_CALUDE_jons_number_l861_86190

theorem jons_number : ∃ (x : ℝ), 5 * (3 * x + 6) - 8 = 142 ∧ x = 8 := by sorry

end NUMINAMATH_CALUDE_jons_number_l861_86190


namespace NUMINAMATH_CALUDE_students_under_three_l861_86130

/-- Represents the number of students in different age groups in a nursery school -/
structure NurserySchool where
  total : ℕ
  fourAndOlder : ℕ
  underThree : ℕ
  notBetweenThreeAndFour : ℕ

/-- Theorem stating the number of students under three years old in the nursery school -/
theorem students_under_three (school : NurserySchool) 
  (h1 : school.total = 300)
  (h2 : school.fourAndOlder = school.total / 10)
  (h3 : school.notBetweenThreeAndFour = 50)
  (h4 : school.notBetweenThreeAndFour = school.fourAndOlder + school.underThree) :
  school.underThree = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_under_three_l861_86130


namespace NUMINAMATH_CALUDE_inequality_solution_l861_86147

theorem inequality_solution (x : ℝ) : 
  -1 < (x^2 - 14*x + 11) / (x^2 - 2*x + 3) ∧ 
  (x^2 - 14*x + 11) / (x^2 - 2*x + 3) < 1 ↔ 
  (2/3 < x ∧ x < 1) ∨ (7 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l861_86147


namespace NUMINAMATH_CALUDE_binomial_22_12_l861_86168

theorem binomial_22_12 (h1 : Nat.choose 20 10 = 184756)
                       (h2 : Nat.choose 20 11 = 167960)
                       (h3 : Nat.choose 20 12 = 125970) :
  Nat.choose 22 12 = 646646 := by
  sorry

end NUMINAMATH_CALUDE_binomial_22_12_l861_86168


namespace NUMINAMATH_CALUDE_quadratic_roots_impossibility_l861_86187

theorem quadratic_roots_impossibility (n : ℕ) (a b : Fin n → ℝ) 
  (h_n : n ≥ 2)
  (h_distinct : ∀ i j : Fin n, (i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j)) :
  ¬(∀ i : Fin n, ∃ j : Fin n, (a i)^2 - a j * (a i) + b j = 0 ∨ (b i)^2 - a j * (b i) + b j = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_impossibility_l861_86187


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l861_86173

/-- The volume of a rectangular box with face areas 24, 16, and 6 square inches is 48 cubic inches -/
theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 24)
  (area2 : w * h = 16)
  (area3 : l * h = 6) :
  l * w * h = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l861_86173


namespace NUMINAMATH_CALUDE_square_root_of_nine_l861_86101

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l861_86101


namespace NUMINAMATH_CALUDE_rosie_pies_theorem_l861_86164

/-- Represents the number of pies Rosie can make given a certain number of apples -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples * 3) / 12

theorem rosie_pies_theorem :
  pies_from_apples 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_theorem_l861_86164


namespace NUMINAMATH_CALUDE_point_on_line_l861_86181

/-- Given a line equation and two points on the line, prove the value of some_value -/
theorem point_on_line (m n some_value : ℝ) : 
  (m = n / 6 - 2 / 5) →  -- First point (m, n) satisfies the line equation
  (m + 3 = (n + some_value) / 6 - 2 / 5) →  -- Second point (m + 3, n + some_value) satisfies the line equation
  some_value = -12 / 5 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l861_86181


namespace NUMINAMATH_CALUDE_cupcake_distribution_l861_86138

/-- Given initial cupcakes, eaten cupcakes, and number of packages, 
    calculate the number of cupcakes in each package. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Theorem stating that with 18 initial cupcakes, 8 eaten cupcakes, 
    and 5 packages, there are 2 cupcakes in each package. -/
theorem cupcake_distribution : cupcakes_per_package 18 8 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_distribution_l861_86138


namespace NUMINAMATH_CALUDE_vanaspati_percentage_in_original_mixture_l861_86145

/-- Represents the composition of a ghee mixture -/
structure GheeMixture where
  total : ℝ
  pure_percentage : ℝ

/-- Calculates the percentage of vanaspati in a ghee mixture -/
def vanaspati_percentage (mixture : GheeMixture) : ℝ :=
  100 - mixture.pure_percentage

theorem vanaspati_percentage_in_original_mixture 
  (original : GheeMixture)
  (h_original_total : original.total = 10)
  (h_original_pure : original.pure_percentage = 60)
  (h_after_addition : 
    let new_total := original.total + 10
    let new_pure := original.total * (original.pure_percentage / 100) + 10
    (100 - (new_pure / new_total * 100)) = 20) :
  vanaspati_percentage original = 40 := by
  sorry

#eval vanaspati_percentage { total := 10, pure_percentage := 60 }

end NUMINAMATH_CALUDE_vanaspati_percentage_in_original_mixture_l861_86145


namespace NUMINAMATH_CALUDE_probability_no_defective_bulbs_l861_86143

def total_bulbs : ℕ := 10
def defective_bulbs : ℕ := 4
def selected_bulbs : ℕ := 4

theorem probability_no_defective_bulbs :
  (Nat.choose (total_bulbs - defective_bulbs) selected_bulbs) /
  (Nat.choose total_bulbs selected_bulbs) = 1 / 14 :=
by sorry

end NUMINAMATH_CALUDE_probability_no_defective_bulbs_l861_86143


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_eighth_l861_86117

theorem fraction_sum_equals_one_eighth :
  (1 : ℚ) / 6 - 5 / 12 + 3 / 8 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_eighth_l861_86117


namespace NUMINAMATH_CALUDE_olly_shoes_count_l861_86111

/-- The number of shoes needed for Olly's pets -/
def shoes_needed (num_dogs num_cats num_ferrets : ℕ) : ℕ :=
  4 * (num_dogs + num_cats + num_ferrets)

/-- Theorem: Olly needs 24 shoes for his pets -/
theorem olly_shoes_count : shoes_needed 3 2 1 = 24 := by
  sorry

end NUMINAMATH_CALUDE_olly_shoes_count_l861_86111


namespace NUMINAMATH_CALUDE_no_matching_roots_l861_86163

theorem no_matching_roots : ∀ x : ℝ,
  (x^2 - 4*x + 3 = 0) → 
  ¬(∃ y : ℝ, (y = x - 1 ∧ y = x - 3)) :=
by sorry

end NUMINAMATH_CALUDE_no_matching_roots_l861_86163


namespace NUMINAMATH_CALUDE_complex_powers_sum_l861_86121

theorem complex_powers_sum : 
  (((Complex.I * Real.sqrt 3 - 1) / 2) ^ 6 + ((Complex.I * Real.sqrt 3 + 1) / (-2)) ^ 6 = 2) ∧
  (∀ n : ℕ, Odd n → ((Complex.I + 1) / Real.sqrt 2) ^ (4 * n) + ((1 - Complex.I) / Real.sqrt 2) ^ (4 * n) = -2) :=
by sorry

end NUMINAMATH_CALUDE_complex_powers_sum_l861_86121


namespace NUMINAMATH_CALUDE_triangle_number_puzzle_l861_86155

theorem triangle_number_puzzle :
  ∀ (A B C D E F : ℕ),
    A ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    B ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    C ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    D ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    F ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F →
    D + E + B = 14 →
    A + C + F = 6 →
    A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_number_puzzle_l861_86155


namespace NUMINAMATH_CALUDE_larger_number_with_given_hcf_and_lcm_factors_l861_86128

theorem larger_number_with_given_hcf_and_lcm_factors : 
  ∀ (a b : ℕ+), 
    (Nat.gcd a b = 47) → 
    (∃ (k : ℕ+), Nat.lcm a b = k * 47 * 7^2 * 11 * 13 * 17^3) →
    (a ≥ b) →
    a = 123800939 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_with_given_hcf_and_lcm_factors_l861_86128


namespace NUMINAMATH_CALUDE_factors_with_more_than_three_factors_l861_86175

def number_to_factor := 2550

-- Function to count factors of a number
def count_factors (n : ℕ) : ℕ := sorry

-- Function to count numbers with more than 3 factors
def count_numbers_with_more_than_three_factors (n : ℕ) : ℕ := sorry

theorem factors_with_more_than_three_factors :
  count_numbers_with_more_than_three_factors number_to_factor = 9 := by sorry

end NUMINAMATH_CALUDE_factors_with_more_than_three_factors_l861_86175


namespace NUMINAMATH_CALUDE_egg_tray_problem_l861_86161

theorem egg_tray_problem (eggs_per_tray : ℕ) (total_eggs : ℕ) : 
  eggs_per_tray = 10 → total_eggs = 70 → total_eggs / eggs_per_tray = 7 := by
  sorry

end NUMINAMATH_CALUDE_egg_tray_problem_l861_86161


namespace NUMINAMATH_CALUDE_circle_intersection_perpendicular_l861_86180

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the intersect relation between circles
variable (intersect : Circle → Circle → Prop)

-- Define the on_circle relation between points and circles
variable (on_circle : Point → Circle → Prop)

-- Define the distance function between points
variable (dist : Point → Point → ℝ)

-- Define the intersect_line_circle relation
variable (intersect_line_circle : Point → Point → Circle → Point → Prop)

-- Define the center_of_arc relation
variable (center_of_arc : Point → Point → Circle → Point → Prop)

-- Define the intersection_of_lines relation
variable (intersection_of_lines : Point → Point → Point → Point → Point → Prop)

-- Define the perpendicular relation
variable (perpendicular : Point → Point → Point → Point → Prop)

-- State the theorem
theorem circle_intersection_perpendicular 
  (C₁ C₂ : Circle) 
  (A B P Q M N C D E : Point) :
  intersect C₁ C₂ →
  on_circle P C₁ →
  on_circle Q C₂ →
  dist A P = dist A Q →
  intersect_line_circle P Q C₁ M →
  intersect_line_circle P Q C₂ N →
  center_of_arc B P C₁ C →
  center_of_arc B Q C₂ D →
  intersection_of_lines C M D N E →
  perpendicular A E C D :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_perpendicular_l861_86180


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l861_86133

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_8th_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_4th : a 4 = 23)
  (h_6th : a 6 = 47) :
  a 8 = 71 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l861_86133


namespace NUMINAMATH_CALUDE_beta_conditions_l861_86186

theorem beta_conditions (β : ℂ) (h1 : β ≠ -1) 
  (h2 : Complex.abs (β^3 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^6 - 1) = 6 * Complex.abs (β - 1)) :
  Complex.abs (β^3 + 1) = 3 ∧ Complex.abs (β^6 + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_beta_conditions_l861_86186


namespace NUMINAMATH_CALUDE_roots_product_minus_one_l861_86114

theorem roots_product_minus_one (d e : ℝ) : 
  (3 * d^2 + 5 * d - 2 = 0) → 
  (3 * e^2 + 5 * e - 2 = 0) → 
  (d-1)*(e-1) = 2 := by
sorry

end NUMINAMATH_CALUDE_roots_product_minus_one_l861_86114


namespace NUMINAMATH_CALUDE_exam_survey_analysis_l861_86110

structure SurveyData where
  total_candidates : Nat
  sample_size : Nat

def sampling_survey_method (data : SurveyData) : Prop :=
  data.sample_size < data.total_candidates

def is_population (data : SurveyData) (n : Nat) : Prop :=
  n = data.total_candidates

def is_sample (data : SurveyData) (n : Nat) : Prop :=
  n = data.sample_size

theorem exam_survey_analysis (data : SurveyData)
  (h1 : data.total_candidates = 60000)
  (h2 : data.sample_size = 1000) :
  ∃ (correct_statements : Finset (Fin 4)),
    correct_statements.card = 2 ∧
    (1 ∈ correct_statements ↔ sampling_survey_method data) ∧
    (2 ∈ correct_statements ↔ is_population data data.total_candidates) ∧
    (3 ∈ correct_statements ↔ is_sample data data.sample_size) ∧
    (4 ∈ correct_statements ↔ data.sample_size = 1000) :=
sorry

end NUMINAMATH_CALUDE_exam_survey_analysis_l861_86110


namespace NUMINAMATH_CALUDE_small_jars_count_l861_86162

/-- Proves that the number of small jars is 62 given the conditions of the problem -/
theorem small_jars_count :
  ∀ (small_jars large_jars : ℕ),
    small_jars + large_jars = 100 →
    3 * small_jars + 5 * large_jars = 376 →
    small_jars = 62 := by
  sorry

end NUMINAMATH_CALUDE_small_jars_count_l861_86162


namespace NUMINAMATH_CALUDE_largest_changeable_digit_is_nine_l861_86132

/-- The original incorrect sum --/
def original_sum : ℕ := 2436

/-- The correct sum of the addends --/
def correct_sum : ℕ := 731 + 962 + 843

/-- The difference between the correct sum and the original sum --/
def difference : ℕ := correct_sum - original_sum

/-- The largest digit in the hundreds place of the addends --/
def largest_hundreds_digit : ℕ := max (731 / 100) (max (962 / 100) (843 / 100))

theorem largest_changeable_digit_is_nine :
  largest_hundreds_digit = 9 ∧ difference = 100 :=
sorry

end NUMINAMATH_CALUDE_largest_changeable_digit_is_nine_l861_86132


namespace NUMINAMATH_CALUDE_right_triangle_area_thrice_hypotenuse_l861_86176

theorem right_triangle_area_thrice_hypotenuse : ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive sides
  c^2 = a^2 + b^2 ∧        -- Pythagorean theorem
  (1/2) * a * b = 3 * c    -- Area equals thrice the hypotenuse
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_thrice_hypotenuse_l861_86176


namespace NUMINAMATH_CALUDE_enrique_commission_l861_86148

/-- Calculates the commission for a given item --/
def calculate_commission (price : ℝ) (quantity : ℕ) (commission_rate : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate) * quantity * commission_rate

/-- Calculates the total commission for all items sold --/
def total_commission (suit_price suit_quantity : ℕ) (shirt_price shirt_quantity : ℕ) 
                     (loafer_price loafer_quantity : ℕ) (tie_price tie_quantity : ℕ) 
                     (sock_price sock_quantity : ℕ) : ℝ :=
  let suit_commission := calculate_commission (suit_price : ℝ) suit_quantity 0.15 0.1 0
  let shirt_commission := calculate_commission (shirt_price : ℝ) shirt_quantity 0.15 0 0.05
  let loafer_commission := calculate_commission (loafer_price : ℝ) loafer_quantity 0.1 0 0.05
  let tie_commission := calculate_commission (tie_price : ℝ) tie_quantity 0.1 0 0.05
  let sock_commission := calculate_commission (sock_price : ℝ) sock_quantity 0.1 0 0.05
  suit_commission + shirt_commission + loafer_commission + tie_commission + sock_commission

theorem enrique_commission : 
  total_commission 700 2 50 6 150 2 30 4 10 5 = 285.60 := by
  sorry

end NUMINAMATH_CALUDE_enrique_commission_l861_86148


namespace NUMINAMATH_CALUDE_largest_angle_in_789_ratio_triangle_l861_86142

/-- Given a triangle with interior angles in a 7:8:9 ratio, 
    the largest interior angle measures 67.5 degrees. -/
theorem largest_angle_in_789_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    a + b + c = 180 →
    b = (8/7) * a →
    c = (9/7) * a →
    max a (max b c) = 67.5 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_789_ratio_triangle_l861_86142


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l861_86192

theorem largest_gcd_of_sum_1008 :
  ∃ (max_gcd : ℕ), ∀ (a b : ℕ), 
    a > 0 → b > 0 → a + b = 1008 →
    gcd a b ≤ max_gcd ∧
    ∃ (a' b' : ℕ), a' > 0 ∧ b' > 0 ∧ a' + b' = 1008 ∧ gcd a' b' = max_gcd :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1008_l861_86192


namespace NUMINAMATH_CALUDE_inverse_proportion_m_value_l861_86170

theorem inverse_proportion_m_value : 
  ∃! m : ℝ, m^2 - 5 = -1 ∧ m + 2 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_m_value_l861_86170


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l861_86184

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) : a > 0 → b > 0 →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ x y : ℝ, (x - c)^2 + y^2 = 4 * a^2) →  -- Circle equation
  (∃ x y : ℝ, (x - c)^2 + y^2 = 4 * a^2 ∧ b * x + a * y = 0 ∧ y^2 = b^2) →  -- Chord condition
  c^2 = a^2 * (1 + (c^2 / a^2 - 1)) →  -- Semi-latus rectum condition
  Real.sqrt ((c^2 / a^2) - 1) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l861_86184


namespace NUMINAMATH_CALUDE_first_alloy_weight_in_mixture_l861_86158

/-- Represents the composition of an alloy mixture -/
structure AlloyMixture where
  first_alloy_weight : ℝ
  second_alloy_weight : ℝ
  first_alloy_chromium_percent : ℝ
  second_alloy_chromium_percent : ℝ
  resulting_chromium_percent : ℝ

/-- Theorem stating the correct weight of the first alloy in the mixture -/
theorem first_alloy_weight_in_mixture 
  (mixture : AlloyMixture)
  (h1 : mixture.first_alloy_chromium_percent = 12)
  (h2 : mixture.second_alloy_chromium_percent = 8)
  (h3 : mixture.second_alloy_weight = 35)
  (h4 : mixture.resulting_chromium_percent = 9.2) :
  mixture.first_alloy_weight = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_alloy_weight_in_mixture_l861_86158


namespace NUMINAMATH_CALUDE_expressions_correctness_l861_86166

theorem expressions_correctness (a b : ℝ) (h1 : a * b > 0) (h2 : a + b < 0) :
  (∃ x : ℝ, x * x = a / b) ∧ 
  (∃ y : ℝ, y * y = b / a) ∧
  (∃ z : ℝ, z * z = a * b) ∧
  (∃ w : ℝ, w * w = a / b) ∧
  (Real.sqrt (a / b) * Real.sqrt (b / a) = 1) ∧
  (Real.sqrt (a * b) / Real.sqrt (a / b) = -b) := by
  sorry

end NUMINAMATH_CALUDE_expressions_correctness_l861_86166


namespace NUMINAMATH_CALUDE_sequence_property_main_theorem_l861_86112

def sequence_a (n : ℕ+) : ℝ :=
  sorry

theorem sequence_property (n : ℕ+) :
  (Finset.range n).sum (λ i => sequence_a ⟨i + 1, Nat.succ_pos i⟩) = n - sequence_a n :=
sorry

def sequence_b (n : ℕ+) : ℝ :=
  (2 - n) * (sequence_a n - 1)

theorem main_theorem :
  (∃ r : ℝ, ∀ n : ℕ+, sequence_a (n + 1) - 1 = r * (sequence_a n - 1)) ∧
  (∀ t : ℝ, (∀ n : ℕ+, sequence_b n + (1/4) * t ≤ t^2) ↔ t ≤ -1/4 ∨ t ≥ 1/2) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_main_theorem_l861_86112


namespace NUMINAMATH_CALUDE_inequality_proof_l861_86194

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + y^2)^2 ≥ (x+y+z)*(x-y+z)*(x+y-z)*(y+z-x) ∧
  ((x^2 + y^2)^2 = (x+y+z)*(x-y+z)*(x+y-z)*(y+z-x) ↔ x = y ∧ z = x*Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l861_86194


namespace NUMINAMATH_CALUDE_volume_conversion_l861_86154

/-- Proves that a volume of 108 cubic feet is equal to 4 cubic yards -/
theorem volume_conversion (box_volume_cubic_feet : ℝ) (cubic_feet_per_cubic_yard : ℝ) :
  box_volume_cubic_feet = 108 →
  cubic_feet_per_cubic_yard = 27 →
  box_volume_cubic_feet / cubic_feet_per_cubic_yard = 4 := by
sorry

end NUMINAMATH_CALUDE_volume_conversion_l861_86154


namespace NUMINAMATH_CALUDE_divisibility_of_factorial_plus_one_l861_86103

theorem divisibility_of_factorial_plus_one (p : ℕ) : 
  (Nat.Prime p → p ∣ (Nat.factorial (p - 1) + 1)) ∧
  (¬Nat.Prime p → ¬(p ∣ (Nat.factorial (p - 1) + 1))) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_factorial_plus_one_l861_86103


namespace NUMINAMATH_CALUDE_cryptic_message_solution_l861_86137

/-- Represents a digit in the cryptic message --/
structure Digit (d : ℕ) where
  value : ℕ
  is_valid : value < d

/-- Represents the cryptic message as an addition problem --/
def is_valid_solution (d : ℕ) (D E P O N : Digit d) : Prop :=
  let deep := D.value * d^3 + E.value * d^2 + E.value * d + P.value
  let pond := P.value * d^3 + O.value * d^2 + N.value * d + D.value
  let done := D.value * d^3 + O.value * d^2 + N.value * d + E.value
  (deep + pond + deep) % d^4 = done

/-- The main theorem stating the existence of a solution --/
theorem cryptic_message_solution :
  ∃ (d : ℕ) (D E P O N : Digit d),
    d = 10 ∧
    is_valid_solution d D E P O N ∧
    D.value ≠ E.value ∧ D.value ≠ P.value ∧ D.value ≠ O.value ∧ D.value ≠ N.value ∧
    E.value ≠ P.value ∧ E.value ≠ O.value ∧ E.value ≠ N.value ∧
    P.value ≠ O.value ∧ P.value ≠ N.value ∧
    O.value ≠ N.value ∧
    D.value = 3 ∧ E.value = 2 ∧ P.value = 3 ∧ O.value = 6 ∧ N.value = 2 :=
sorry

end NUMINAMATH_CALUDE_cryptic_message_solution_l861_86137


namespace NUMINAMATH_CALUDE_sugar_ratio_l861_86144

theorem sugar_ratio (a₁ a₂ a₃ a₄ : ℝ) (h₁ : a₁ = 24) (h₂ : a₄ = 3)
  (h_geom : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) :
  a₂ / a₁ = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sugar_ratio_l861_86144


namespace NUMINAMATH_CALUDE_petya_winning_strategy_l861_86160

/-- Represents the state of cups on a 2n-gon -/
def CupState (n : ℕ) := Fin (2 * n) → Bool

/-- Checks if two positions are adjacent on a 2n-gon -/
def adjacent (n : ℕ) (i j : Fin (2 * n)) : Prop :=
  (i.val + 1) % (2 * n) = j.val ∨ (j.val + 1) % (2 * n) = i.val

/-- Checks if two positions are symmetric with respect to the center of a 2n-gon -/
def symmetric (n : ℕ) (i j : Fin (2 * n)) : Prop :=
  (i.val + n) % (2 * n) = j.val

/-- Checks if a move is valid in the tea-pouring game -/
def valid_move (n : ℕ) (state : CupState n) (i j : Fin (2 * n)) : Prop :=
  ¬state i ∧ ¬state j ∧ (adjacent n i j ∨ symmetric n i j)

/-- Represents a winning strategy for Petya in the tea-pouring game -/
def petya_wins (n : ℕ) : Prop :=
  ∀ (state : CupState n),
    (∃ (i j : Fin (2 * n)), valid_move n state i j) →
    ∃ (i j : Fin (2 * n)), valid_move n state i j ∧
      ¬(∃ (k l : Fin (2 * n)), valid_move n (Function.update (Function.update state i true) j true) k l)

/-- The main theorem: Petya has a winning strategy if and only if n is odd -/
theorem petya_winning_strategy (n : ℕ) : petya_wins n ↔ Odd n := by
  sorry

end NUMINAMATH_CALUDE_petya_winning_strategy_l861_86160


namespace NUMINAMATH_CALUDE_three_same_one_different_probability_l861_86119

/-- The probability of a child being born a boy or a girl -/
def child_probability : ℚ := 1 / 2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The number of possible combinations for having three children of one sex and one of the opposite sex -/
def num_combinations : ℕ := 8

/-- The probability of having three children of one sex and one of the opposite sex in a family of four children -/
theorem three_same_one_different_probability :
  (child_probability ^ num_children) * num_combinations = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_three_same_one_different_probability_l861_86119


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l861_86196

/-- Given a parabola with equation y = 8x^2, its latus rectum has equation y = 1/32 -/
theorem latus_rectum_of_parabola (x y : ℝ) :
  y = 8 * x^2 → (∃ (x₀ : ℝ), y = 1/32 ∧ x₀ ≠ 0 ∧ y = 8 * x₀^2) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l861_86196


namespace NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l861_86151

theorem sphere_radius_from_surface_area (A : ℝ) (r : ℝ) (h : A = 64 * Real.pi) :
  A = 4 * Real.pi * r^2 → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l861_86151


namespace NUMINAMATH_CALUDE_root_sum_squares_l861_86118

theorem root_sum_squares (r s : ℝ) (α β γ δ : ℂ) : 
  (α^2 - r*α - 2 = 0) → 
  (β^2 - r*β - 2 = 0) → 
  (γ^2 + s*γ - 2 = 0) → 
  (δ^2 + s*δ - 2 = 0) → 
  (α - γ)^2 + (β - γ)^2 + (α + δ)^2 + (β + δ)^2 = 4*s*(r - s) + 8 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_squares_l861_86118


namespace NUMINAMATH_CALUDE_movie_ticket_distribution_l861_86153

/-- The number of ways to distribute distinct objects to distinct recipients --/
def distribute_distinct (n_objects : ℕ) (n_recipients : ℕ) : ℕ :=
  (n_recipients - n_objects + 1).factorial / (n_recipients - n_objects).factorial

/-- The number of ways to distribute 3 different movie tickets among 10 people --/
theorem movie_ticket_distribution :
  distribute_distinct 3 10 = 720 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_distribution_l861_86153


namespace NUMINAMATH_CALUDE_additive_inverse_problem_l861_86193

theorem additive_inverse_problem (m : ℤ) : (m + 1) + (-2) = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverse_problem_l861_86193
