import Mathlib

namespace NUMINAMATH_CALUDE_three_cubes_exposed_faces_sixty_cubes_exposed_faces_l1654_165462

/-- The number of exposed faces for n cubes in a row on a table -/
def exposed_faces (n : ℕ) : ℕ := 3 * n + 2

/-- Theorem stating that for 3 cubes, there are 11 exposed faces -/
theorem three_cubes_exposed_faces : exposed_faces 3 = 11 := by sorry

/-- Theorem to prove the number of exposed faces for 60 cubes -/
theorem sixty_cubes_exposed_faces : exposed_faces 60 = 182 := by sorry

end NUMINAMATH_CALUDE_three_cubes_exposed_faces_sixty_cubes_exposed_faces_l1654_165462


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l1654_165412

theorem subtraction_preserves_inequality (a b : ℝ) : a > b → a - 1 > b - 1 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l1654_165412


namespace NUMINAMATH_CALUDE_pyramid_volume_l1654_165464

/-- Given a pyramid with a rhombus base (diagonals d₁ and d₂, where d₁ > d₂) and height passing
    through the vertex of the acute angle of the rhombus, if the area of the diagonal cross-section
    made through the smaller diagonal is Q, then the volume of the pyramid is
    (d₁/12) * √(16Q² - d₁²d₂²). -/
theorem pyramid_volume (d₁ d₂ Q : ℝ) (h₁ : d₁ > d₂) (h₂ : d₂ > 0) (h₃ : Q > 0) :
  let V := d₁ / 12 * Real.sqrt (16 * Q^2 - d₁^2 * d₂^2)
  ∃ (height : ℝ), height > 0 ∧ 
    (V = (1/3) * (1/2 * d₁ * d₂) * height) ∧
    (Q = (1/2) * d₂ * (2 * Q / d₂)) ∧
    (height = Real.sqrt ((2 * Q / d₂)^2 - (d₁ / 2)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1654_165464


namespace NUMINAMATH_CALUDE_unique_prime_p_l1654_165477

theorem unique_prime_p (p : ℕ) : 
  Prime p ∧ 
  Prime (8 * p^4 - 3003) ∧ 
  (8 * p^4 - 3003 > 0) ↔ 
  p = 5 := by sorry

end NUMINAMATH_CALUDE_unique_prime_p_l1654_165477


namespace NUMINAMATH_CALUDE_solution_set_has_three_elements_l1654_165400

/-- A pair of positive integers representing the sides of a rectangle. -/
structure RectangleSides where
  a : ℕ+
  b : ℕ+

/-- The condition that the perimeter of a rectangle equals its area. -/
def perimeterEqualsArea (sides : RectangleSides) : Prop :=
  2 * (sides.a.val + sides.b.val) = sides.a.val * sides.b.val

/-- The set of all rectangle sides satisfying the perimeter-area equality. -/
def solutionSet : Set RectangleSides :=
  {sides | perimeterEqualsArea sides}

/-- The theorem stating that the solution set contains exactly three elements. -/
theorem solution_set_has_three_elements :
    solutionSet = {⟨3, 6⟩, ⟨6, 3⟩, ⟨4, 4⟩} := by sorry

end NUMINAMATH_CALUDE_solution_set_has_three_elements_l1654_165400


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l1654_165441

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors created by combining 5 scoops from 3 basic flavors -/
def ice_cream_flavors : ℕ := distribute 5 3

/-- Theorem: The number of ice cream flavors is 21 -/
theorem ice_cream_flavors_count : ice_cream_flavors = 21 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l1654_165441


namespace NUMINAMATH_CALUDE_field_division_l1654_165453

theorem field_division (total_area smaller_area larger_area : ℝ) : 
  total_area = 900 ∧
  smaller_area + larger_area = total_area ∧
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 405 := by
  sorry

end NUMINAMATH_CALUDE_field_division_l1654_165453


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l1654_165461

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Theorem: The result of (101110₂) × (110100₂) ÷ (110₂) is 101011100₂ -/
theorem binary_multiplication_division :
  let a := binaryToNat [true, false, true, true, true, false]  -- 101110₂
  let b := binaryToNat [true, true, false, true, false, false] -- 110100₂
  let c := binaryToNat [true, true, false]                     -- 110₂
  let result := binaryToNat [true, false, true, false, true, true, true, false, false] -- 101011100₂
  a * b / c = result := by
  sorry


end NUMINAMATH_CALUDE_binary_multiplication_division_l1654_165461


namespace NUMINAMATH_CALUDE_f_two_roots_iff_m_range_f_min_value_on_interval_l1654_165448

/-- The function f(x) = x^2 - 4mx + 6m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*m*x + 6*m

theorem f_two_roots_iff_m_range (m : ℝ) :
  (∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ m < 0 ∨ m > 3/2 := by sorry

theorem f_min_value_on_interval (m : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f m x ≥ (
    if m ≤ 0 then 6*m
    else if m < 3/2 then -4*m^2 + 6*m
    else 9 - 6*m
  )) ∧
  (∃ x ∈ Set.Icc 0 3, f m x = (
    if m ≤ 0 then 6*m
    else if m < 3/2 then -4*m^2 + 6*m
    else 9 - 6*m
  )) := by sorry

end NUMINAMATH_CALUDE_f_two_roots_iff_m_range_f_min_value_on_interval_l1654_165448


namespace NUMINAMATH_CALUDE_descending_order_proof_l1654_165495

theorem descending_order_proof :
  (1909 > 1100 ∧ 1100 > 1090 ∧ 1090 > 1009) ∧
  (10000 > 9999 ∧ 9999 > 9990 ∧ 9990 > 8909 ∧ 8909 > 8900) := by
  sorry

end NUMINAMATH_CALUDE_descending_order_proof_l1654_165495


namespace NUMINAMATH_CALUDE_towel_area_decrease_l1654_165456

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let new_length := 0.7 * L
  let new_breadth := 0.85 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.405 := by
sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l1654_165456


namespace NUMINAMATH_CALUDE_second_sample_not_23_l1654_165460

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total : ℕ  -- Total number of items
  sample_size : ℕ  -- Number of items to be sampled
  first_sample : ℕ  -- The first sampled item

/-- The second sample in a systematic sampling scheme -/
def second_sample (s : SystematicSampling) : ℕ :=
  s.first_sample + (s.total / s.sample_size)

/-- Theorem: The second sample cannot be 23 in the given systematic sampling scheme -/
theorem second_sample_not_23 (s : SystematicSampling) 
  (h1 : s.total > 0)
  (h2 : s.sample_size > 0)
  (h3 : s.sample_size ≤ s.total)
  (h4 : s.first_sample ≤ 10)
  (h5 : s.first_sample > 0)
  (h6 : s.sample_size = s.total / 10) :
  second_sample s ≠ 23 := by
  sorry

end NUMINAMATH_CALUDE_second_sample_not_23_l1654_165460


namespace NUMINAMATH_CALUDE_sequence_properties_l1654_165471

/-- A sequence where the sum of the first n terms is S_n = 2n^2 + 3n -/
def S (n : ℕ) : ℕ := 2 * n^2 + 3 * n

/-- The nth term of the sequence -/
def a (n : ℕ) : ℕ := 4 * n + 1

theorem sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = a (n + 1)) ∧
  (∀ n : ℕ, a (n + 1) - a n = 4) ∧
  (a 10 = 41) := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1654_165471


namespace NUMINAMATH_CALUDE_f_at_two_l1654_165499

noncomputable section

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_monotonic : Monotone f
axiom f_condition : ∀ x : ℝ, f (f x - exp x) = exp 1 + 1

-- Theorem to prove
theorem f_at_two : f 2 = exp 2 + 1 := by sorry

end NUMINAMATH_CALUDE_f_at_two_l1654_165499


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1654_165420

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals : 
  num_diagonals dodecagon_sides = 54 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1654_165420


namespace NUMINAMATH_CALUDE_sports_league_games_l1654_165478

/-- Represents a sports league with the given conditions -/
structure SportsLeague where
  total_teams : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculate the total number of games in the sports league -/
def total_games (league : SportsLeague) : Nat :=
  let games_per_team := (league.teams_per_division - 1) * league.intra_division_games +
                        league.teams_per_division * league.inter_division_games
  (games_per_team * league.total_teams) / 2

/-- Theorem stating the total number of games in the given sports league configuration -/
theorem sports_league_games :
  let league := SportsLeague.mk 16 8 3 2
  total_games league = 296 := by
  sorry

end NUMINAMATH_CALUDE_sports_league_games_l1654_165478


namespace NUMINAMATH_CALUDE_f_difference_l1654_165465

/-- The function f(x) = 3x^2 + 5x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

/-- Theorem stating that f(x+h) - f(x) = h(6x + 3h + 5) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h + 5) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l1654_165465


namespace NUMINAMATH_CALUDE_inequality_solution_set_f_less_than_one_l1654_165450

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Theorem 1: Solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 2 - |x + 1|} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2/3} := by sorry

-- Theorem 2: Proof that f(x) < 1 under given conditions
theorem f_less_than_one (x y : ℝ) (h1 : |x - y - 1| ≤ 1/3) (h2 : |2*y + 1| ≤ 1/6) :
  f x < 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_f_less_than_one_l1654_165450


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l1654_165434

theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let plain_pizza_cost : ℚ := 12
  let mushroom_cost : ℚ := 3
  let bob_mushroom_slices : ℕ := total_slices / 3
  let bob_plain_slices : ℕ := 3
  let alice_slices : ℕ := total_slices - (bob_mushroom_slices + bob_plain_slices)
  let total_cost : ℚ := plain_pizza_cost + mushroom_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let bob_payment : ℚ := (bob_mushroom_slices + bob_plain_slices) * cost_per_slice
  let alice_payment : ℚ := alice_slices * (plain_pizza_cost / total_slices)
  bob_payment - alice_payment = 3.75 := by sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l1654_165434


namespace NUMINAMATH_CALUDE_reverse_digit_integers_l1654_165402

theorem reverse_digit_integers (q r : ℕ) : 
  (q ≥ 10 ∧ q < 100) →  -- q is a two-digit number
  (r ≥ 10 ∧ r < 100) →  -- r is a two-digit number
  (∃ (a b : ℕ), q = 10 * a + b ∧ r = 10 * b + a) →  -- q and r have reversed digits
  (q > r → q - r < 30) →  -- positive difference less than 30
  (∀ (q' r' : ℕ), (q' ≥ 10 ∧ q' < 100) → (r' ≥ 10 ∧ r' < 100) → 
    (∃ (a' b' : ℕ), q' = 10 * a' + b' ∧ r' = 10 * b' + a') → 
    (q' > r' → q' - r' ≤ q - r)) →  -- q - r is the greatest possible difference
  (q - r = 27) →  -- greatest difference is 27
  (∃ (a b : ℕ), q = 10 * a + b ∧ r = 10 * b + a ∧ a - b = 3 ∧ a = 9 ∧ b = 6) :=
by sorry

end NUMINAMATH_CALUDE_reverse_digit_integers_l1654_165402


namespace NUMINAMATH_CALUDE_no_ten_digit_divisor_with_different_digits_l1654_165488

/-- The number consisting of 1000 ones -/
def number_of_ones : ℕ := 10^1000 - 1

/-- A function to check if a natural number has exactly 10 digits -/
def has_ten_digits (n : ℕ) : Prop := 10^9 ≤ n ∧ n < 10^10

/-- A function to check if all digits in a number are different -/
def all_digits_different (n : ℕ) : Prop :=
  ∀ d₁ d₂, 0 ≤ d₁ ∧ d₁ < 10 ∧ 0 ≤ d₂ ∧ d₂ < 10 → d₁ ≠ d₂ → (n / 10^d₁ % 10) ≠ (n / 10^d₂ % 10)

/-- The main theorem stating that the number of 1000 ones has no ten-digit divisor with all different digits -/
theorem no_ten_digit_divisor_with_different_digits : 
  ¬ ∃ (d : ℕ), d ∣ number_of_ones ∧ has_ten_digits d ∧ all_digits_different d := by
  sorry

end NUMINAMATH_CALUDE_no_ten_digit_divisor_with_different_digits_l1654_165488


namespace NUMINAMATH_CALUDE_polynomial_value_l1654_165489

theorem polynomial_value (a : ℝ) (h : a^2 + 2*a = 1) : 
  2*a^5 + 7*a^4 + 5*a^3 + 2*a^2 + 5*a + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l1654_165489


namespace NUMINAMATH_CALUDE_mistaken_calculation_system_l1654_165421

theorem mistaken_calculation_system (x y : ℝ) : 
  (5/4 * x = 4/5 * x + 36) ∧ 
  (7/3 * y = 3/7 * y + 28) → 
  x = 80 ∧ y = 14.7 := by
sorry

end NUMINAMATH_CALUDE_mistaken_calculation_system_l1654_165421


namespace NUMINAMATH_CALUDE_cube_greater_iff_l1654_165408

theorem cube_greater_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_greater_iff_l1654_165408


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1654_165474

theorem complex_number_quadrant : ∃ (z : ℂ), z = (4 * Complex.I) / (1 + Complex.I) ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1654_165474


namespace NUMINAMATH_CALUDE_airline_capacity_proof_l1654_165475

/-- Calculates the number of passengers an airline can accommodate daily -/
def airline_capacity (num_airplanes : ℕ) (rows_per_airplane : ℕ) (seats_per_row : ℕ) (flights_per_day : ℕ) : ℕ :=
  num_airplanes * rows_per_airplane * seats_per_row * flights_per_day

/-- Proves that the airline company can accommodate 1400 passengers daily -/
theorem airline_capacity_proof :
  airline_capacity 5 20 7 2 = 1400 := by
  sorry

#eval airline_capacity 5 20 7 2

end NUMINAMATH_CALUDE_airline_capacity_proof_l1654_165475


namespace NUMINAMATH_CALUDE_cristinas_leftover_croissants_l1654_165443

/-- Represents the types of croissants --/
inductive CroissantType
  | Chocolate
  | Plain

/-- Represents a guest's dietary restriction --/
inductive DietaryRestriction
  | Vegan
  | ChocolateAllergy
  | NoRestriction

/-- Represents the croissant distribution problem --/
structure CroissantDistribution where
  total_croissants : ℕ
  chocolate_croissants : ℕ
  plain_croissants : ℕ
  guests : List DietaryRestriction
  more_chocolate : chocolate_croissants > plain_croissants

/-- The specific instance of the problem --/
def cristinas_distribution : CroissantDistribution := {
  total_croissants := 17,
  chocolate_croissants := 12,
  plain_croissants := 5,
  guests := [DietaryRestriction.Vegan, DietaryRestriction.Vegan, DietaryRestriction.Vegan,
             DietaryRestriction.ChocolateAllergy, DietaryRestriction.ChocolateAllergy,
             DietaryRestriction.NoRestriction, DietaryRestriction.NoRestriction],
  more_chocolate := by sorry
}

/-- Function to calculate the number of leftover croissants --/
def leftover_croissants (d : CroissantDistribution) : ℕ := 
  d.total_croissants - d.guests.length

/-- Theorem stating that the number of leftover croissants in Cristina's distribution is 3 --/
theorem cristinas_leftover_croissants :
  leftover_croissants cristinas_distribution = 3 := by sorry

end NUMINAMATH_CALUDE_cristinas_leftover_croissants_l1654_165443


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1654_165487

theorem largest_angle_in_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines holds
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given conditions
  Real.cos A = 3/4 →
  C = 2 * A →
  -- Conclusion
  C > A ∧ C > B :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1654_165487


namespace NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l1654_165446

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt x + 1 / x ≥ 4 ∧ ∃ y > 0, 3 * Real.sqrt y + 1 / y = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l1654_165446


namespace NUMINAMATH_CALUDE_symmetry_of_even_functions_l1654_165481

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define symmetry about a point
def IsSymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

theorem symmetry_of_even_functions :
  (∀ f : ℝ → ℝ, IsEven f → IsSymmetricAbout (fun x ↦ f (x + 2)) (-2)) ∧
  (∀ f : ℝ → ℝ, IsEven (fun x ↦ f (x + 2)) → IsSymmetricAbout f 2) := by
  sorry


end NUMINAMATH_CALUDE_symmetry_of_even_functions_l1654_165481


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1654_165476

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) / (a^3 + b^3 + c^3 - 2*a*b*c)
  A ≤ 6 ∧ (A = 6 ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1654_165476


namespace NUMINAMATH_CALUDE_total_cats_l1654_165492

/-- The number of cats owned by Mr. Thompson -/
def thompson_cats : ℕ := 15

/-- The number of cats owned by Mrs. Sheridan -/
def sheridan_cats : ℕ := 11

/-- The number of cats owned by Mrs. Garrett -/
def garrett_cats : ℕ := 24

/-- The number of cats owned by Mr. Ravi -/
def ravi_cats : ℕ := 18

/-- The theorem stating that the total number of cats is 68 -/
theorem total_cats : thompson_cats + sheridan_cats + garrett_cats + ravi_cats = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_l1654_165492


namespace NUMINAMATH_CALUDE_preserve_inequality_arithmetic_harmonic_mean_max_product_fixed_sum_inequality_squares_not_always_max_sum_fixed_product_l1654_165463

-- Statement A
theorem preserve_inequality (a b c : ℝ) (h : a < b) (k : ℝ) (hk : k > 0) :
  k * a < k * b ∧ a / k < b / k := by sorry

-- Statement B
theorem arithmetic_harmonic_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  (a + b) / 2 > 2 * a * b / (a + b) := by sorry

-- Statement C
theorem max_product_fixed_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (s : ℝ) (hs : s = a + b) :
  a * b ≤ (s / 2) * (s / 2) := by sorry

-- Statement D
theorem inequality_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  (1 / 3) * (a^2 + b^2) > ((1 / 3) * (a + b))^2 := by sorry

-- Statement E (incorrect)
theorem not_always_max_sum_fixed_product (P : ℝ → ℝ → Prop) :
  (∃ a b k, a > 0 ∧ b > 0 ∧ a * b = k ∧ a + b > 2 * Real.sqrt k) → 
  ¬(∀ x y, x > 0 → y > 0 → x * y = k → x + y ≤ 2 * Real.sqrt k) := by sorry

end NUMINAMATH_CALUDE_preserve_inequality_arithmetic_harmonic_mean_max_product_fixed_sum_inequality_squares_not_always_max_sum_fixed_product_l1654_165463


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1654_165444

/-- The y-intercept of the line 3x - 4y = 12 is -3 -/
theorem y_intercept_of_line (x y : ℝ) : 3 * x - 4 * y = 12 → x = 0 → y = -3 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1654_165444


namespace NUMINAMATH_CALUDE_min_vertical_distance_is_zero_l1654_165494

-- Define the functions
def f (x : ℝ) := abs x
def g (x : ℝ) := -x^2 - 5*x - 4

-- Define the vertical distance function
def verticalDistance (x : ℝ) := f x - g x

-- Theorem statement
theorem min_vertical_distance_is_zero :
  ∃ x : ℝ, verticalDistance x = 0 ∧ ∀ y : ℝ, verticalDistance y ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_is_zero_l1654_165494


namespace NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l1654_165407

theorem abs_m_minus_n_equals_five (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : 
  |m - n| = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_equals_five_l1654_165407


namespace NUMINAMATH_CALUDE_trick_or_treat_distribution_l1654_165406

/-- The number of blocks in the village -/
def num_blocks : ℕ := 9

/-- The total number of children going trick or treating -/
def total_children : ℕ := 54

/-- There are some children on each block -/
axiom children_on_each_block : ∀ b : ℕ, b < num_blocks → ∃ c : ℕ, c > 0

/-- The number of children on each block -/
def children_per_block : ℕ := total_children / num_blocks

theorem trick_or_treat_distribution :
  children_per_block = 6 :=
sorry

end NUMINAMATH_CALUDE_trick_or_treat_distribution_l1654_165406


namespace NUMINAMATH_CALUDE_shop_prices_l1654_165449

theorem shop_prices (x y : ℝ) 
  (sum_condition : x + y = 5)
  (retail_condition : 3 * (x + 1) + 2 * (2 * y - 1) = 19) :
  x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_shop_prices_l1654_165449


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1654_165454

/-- Represents the sampling methods --/
inductive SamplingMethod
  | StratifiedSampling
  | LotteryMethod
  | SystematicSampling
  | RandomNumberTableMethod

/-- Represents a grade with classes and students --/
structure Grade where
  num_classes : Nat
  students_per_class : Nat
  selected_number : Nat

/-- Determines the sampling method based on the grade structure --/
def determineSamplingMethod (g : Grade) : SamplingMethod :=
  if g.num_classes > 0 ∧ 
     g.students_per_class > 0 ∧ 
     g.selected_number > 0 ∧ 
     g.selected_number ≤ g.students_per_class
  then SamplingMethod.SystematicSampling
  else SamplingMethod.StratifiedSampling  -- Default case, not actually used in this problem

theorem systematic_sampling_theorem (g : Grade) :
  g.num_classes = 12 ∧ 
  g.students_per_class = 50 ∧ 
  g.selected_number = 14 →
  determineSamplingMethod g = SamplingMethod.SystematicSampling :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1654_165454


namespace NUMINAMATH_CALUDE_robin_gum_count_l1654_165496

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 9

/-- The number of gum pieces in each package -/
def pieces_per_package : ℕ := 15

/-- The total number of gum pieces Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_count : total_pieces = 135 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_count_l1654_165496


namespace NUMINAMATH_CALUDE_sum_in_base7_l1654_165466

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- The sum of 666₇, 66₇, and 6₇ in base 7 is 1400₇ -/
theorem sum_in_base7 : 
  base10ToBase7 (base7ToBase10 666 + base7ToBase10 66 + base7ToBase10 6) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base7_l1654_165466


namespace NUMINAMATH_CALUDE_terrell_total_hike_distance_l1654_165484

/-- Represents a hike with distance, duration, and calorie expenditure -/
structure Hike where
  distance : ℝ
  duration : ℝ
  calories : ℝ

/-- Calculates the total distance of two hikes -/
def total_distance (h1 h2 : Hike) : ℝ :=
  h1.distance + h2.distance

theorem terrell_total_hike_distance :
  let saturday_hike : Hike := { distance := 8.2, duration := 5, calories := 4000 }
  let sunday_hike : Hike := { distance := 1.6, duration := 2, calories := 1500 }
  total_distance saturday_hike sunday_hike = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_terrell_total_hike_distance_l1654_165484


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l1654_165413

def A (c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; c, d]

theorem matrix_is_own_inverse (c d : ℝ) :
  A c d * A c d = 1 ↔ c = 3/2 ∧ d = -2 := by sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l1654_165413


namespace NUMINAMATH_CALUDE_percentage_relationship_l1654_165457

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 0.4444444444444444)) :
  y = x * 1.8 := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l1654_165457


namespace NUMINAMATH_CALUDE_sqrt_tan_domain_l1654_165436

theorem sqrt_tan_domain (x : ℝ) :
  ∃ (y : ℝ), y = Real.sqrt (Real.tan x) ↔ ∃ (k : ℤ), k * Real.pi ≤ x ∧ x < k * Real.pi + Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_tan_domain_l1654_165436


namespace NUMINAMATH_CALUDE_triangle_side_length_20_l1654_165424

theorem triangle_side_length_20 :
  ∃ (T S : ℕ), 
    T = 20 ∧ 
    3 * T = 4 * S :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_20_l1654_165424


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1654_165498

theorem quadratic_equation_solution :
  let x₁ : ℝ := (2 + Real.sqrt 3) / 2
  let x₂ : ℝ := (2 - Real.sqrt 3) / 2
  4 * x₁^2 - 8 * x₁ + 1 = 0 ∧ 4 * x₂^2 - 8 * x₂ + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1654_165498


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1654_165430

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, n < 100 → n % 17 = 0 → n ≤ 85 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1654_165430


namespace NUMINAMATH_CALUDE_f_at_two_equals_three_l1654_165452

def f (x : ℝ) : ℝ := 5 * x - 7

theorem f_at_two_equals_three : f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_at_two_equals_three_l1654_165452


namespace NUMINAMATH_CALUDE_king_probability_l1654_165431

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (kings : Nat)
  (unique_combinations : Bool)

/-- Probability of an event -/
def probability (favorable_outcomes : Nat) (total_outcomes : Nat) : ℚ :=
  favorable_outcomes / total_outcomes

theorem king_probability (d : Deck) (h1 : d.cards = 52) (h2 : d.ranks = 13) 
  (h3 : d.suits = 4) (h4 : d.kings = 4) (h5 : d.unique_combinations = true) :
  probability d.kings d.cards = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_king_probability_l1654_165431


namespace NUMINAMATH_CALUDE_problem_trip_mpg_l1654_165459

/-- Represents a car trip with odometer readings and gas fill amounts -/
structure CarTrip where
  initial_odometer : ℕ
  final_odometer : ℕ
  gas_fills : List ℕ

/-- Calculates the average miles per gallon for a car trip -/
def averageMPG (trip : CarTrip) : ℚ :=
  let total_distance := trip.final_odometer - trip.initial_odometer
  let total_gas := trip.gas_fills.sum
  (total_distance : ℚ) / total_gas

/-- The specific car trip from the problem -/
def problemTrip : CarTrip := {
  initial_odometer := 68300
  final_odometer := 69600
  gas_fills := [15, 25]
}

/-- Theorem stating that the average MPG for the problem trip is 32.5 -/
theorem problem_trip_mpg : averageMPG problemTrip = 32.5 := by
  sorry


end NUMINAMATH_CALUDE_problem_trip_mpg_l1654_165459


namespace NUMINAMATH_CALUDE_right_triangle_rational_sides_equiv_arithmetic_progression_l1654_165493

theorem right_triangle_rational_sides_equiv_arithmetic_progression (d : ℤ) :
  (∃ (a b c : ℚ), a^2 + b^2 = c^2 ∧ (1/2) * a * b = d) ↔
  (∃ (x y z : ℚ), 2 * y^2 = x^2 + z^2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_rational_sides_equiv_arithmetic_progression_l1654_165493


namespace NUMINAMATH_CALUDE_distance_between_squares_l1654_165435

theorem distance_between_squares (small_perimeter : ℝ) (large_area : ℝ) : 
  small_perimeter = 8 →
  large_area = 25 →
  let small_side := small_perimeter / 4
  let large_side := Real.sqrt large_area
  let horizontal := small_side + large_side
  let vertical := large_side - small_side
  Real.sqrt (horizontal ^ 2 + vertical ^ 2) = Real.sqrt 58 := by
  sorry

#check distance_between_squares

end NUMINAMATH_CALUDE_distance_between_squares_l1654_165435


namespace NUMINAMATH_CALUDE_parallel_lines_equal_angles_plane_l1654_165422

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the relation for a line forming equal angles with a plane
variable (forms_equal_angles : Line → Plane → Prop)

-- Theorem statement
theorem parallel_lines_equal_angles_plane (a b : Line) (M : Plane) :
  (parallel a b → forms_equal_angles b M) ∧
  ¬(forms_equal_angles b M → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_equal_angles_plane_l1654_165422


namespace NUMINAMATH_CALUDE_oplus_two_one_l1654_165401

def oplus (x y : ℝ) : ℝ := x^3 - 3*x*y^2 + y^3

theorem oplus_two_one : oplus 2 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_oplus_two_one_l1654_165401


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1654_165469

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 13*x^3) 
  (h3 : a - b = 2*x) : 
  (a = x + (Real.sqrt 66 * x) / 6) ∨ (a = x - (Real.sqrt 66 * x) / 6) :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1654_165469


namespace NUMINAMATH_CALUDE_stamp_ratio_problem_l1654_165410

theorem stamp_ratio_problem (x : ℕ) 
  (h1 : x > 0)
  (h2 : 7 * x - 8 = (4 * x + 8) + 8) :
  (7 * x - 8) / (4 * x + 8) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_stamp_ratio_problem_l1654_165410


namespace NUMINAMATH_CALUDE_all_propositions_false_l1654_165442

-- Define the basic types
variable (Line Plane : Type)

-- Define the parallel relation
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)

-- Define the "passes through" relation for planes and lines
variable (passes_through : Plane → Line → Prop)

-- Define the "within" relation for lines and planes
variable (within : Line → Plane → Prop)

-- Define the "has common points" relation
variable (has_common_points : Line → Line → Prop)

-- Define a proposition for "countless lines within a plane"
variable (countless_parallel_lines : Line → Plane → Prop)

-- State the theorem
theorem all_propositions_false :
  -- Proposition 1
  (∀ l1 l2 : Line, ∀ p : Plane, 
    parallel l1 l2 → passes_through p l2 → parallelLP l1 p) ∧
  -- Proposition 2
  (∀ l : Line, ∀ p : Plane,
    parallelLP l p → 
    (∀ l2 : Line, within l2 p → ¬(has_common_points l l2)) ∧
    (∀ l2 : Line, within l2 p → parallel l l2)) ∧
  -- Proposition 3
  (∀ l : Line, ∀ p : Plane,
    ¬(parallelLP l p) → ∀ l2 : Line, within l2 p → ¬(parallel l l2)) ∧
  -- Proposition 4
  (∀ l : Line, ∀ p : Plane,
    countless_parallel_lines l p → parallelLP l p)
  → False := by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l1654_165442


namespace NUMINAMATH_CALUDE_south_american_stamps_cost_l1654_165497

structure Country where
  name : String
  continent : String
  price : Rat
  stamps_50s : Nat
  stamps_60s : Nat

def brazil : Country := {
  name := "Brazil"
  continent := "South America"
  price := 6/100
  stamps_50s := 4
  stamps_60s := 7
}

def peru : Country := {
  name := "Peru"
  continent := "South America"
  price := 4/100
  stamps_50s := 6
  stamps_60s := 4
}

def france : Country := {
  name := "France"
  continent := "Europe"
  price := 6/100
  stamps_50s := 8
  stamps_60s := 4
}

def spain : Country := {
  name := "Spain"
  continent := "Europe"
  price := 5/100
  stamps_50s := 3
  stamps_60s := 9
}

def south_american_countries : List Country := [brazil, peru]

def total_cost (countries : List Country) : Rat :=
  countries.foldl (fun acc country => 
    acc + (country.price * (country.stamps_50s + country.stamps_60s : Rat))) 0

theorem south_american_stamps_cost :
  total_cost south_american_countries = 106/100 := by
  sorry

#eval total_cost south_american_countries

end NUMINAMATH_CALUDE_south_american_stamps_cost_l1654_165497


namespace NUMINAMATH_CALUDE_square_to_parallelogram_l1654_165409

/-- Represents a plane figure --/
structure PlaneFigure where
  -- Add necessary fields

/-- Represents the oblique side drawing method --/
def obliqueSideDrawing (figure : PlaneFigure) : PlaneFigure :=
  sorry

/-- Predicate to check if a figure is a square --/
def isSquare (figure : PlaneFigure) : Prop :=
  sorry

/-- Predicate to check if a figure is a parallelogram --/
def isParallelogram (figure : PlaneFigure) : Prop :=
  sorry

/-- Theorem: The intuitive diagram of a square using oblique side drawing is a parallelogram --/
theorem square_to_parallelogram (figure : PlaneFigure) :
  isSquare figure → isParallelogram (obliqueSideDrawing figure) :=
by sorry

end NUMINAMATH_CALUDE_square_to_parallelogram_l1654_165409


namespace NUMINAMATH_CALUDE_all_T_divisible_by_4_l1654_165472

/-- The set of all numbers which are the sum of the squares of four consecutive integers
    added to the sum of the integers themselves. -/
def T : Set ℤ :=
  {x | ∃ n : ℤ, x = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 + (n-1) + n + (n+1) + (n+2)}

/-- All members of set T are divisible by 4. -/
theorem all_T_divisible_by_4 : ∀ x ∈ T, 4 ∣ x := by sorry

end NUMINAMATH_CALUDE_all_T_divisible_by_4_l1654_165472


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l1654_165486

theorem system_of_equations_solutions :
  -- System (1)
  (∃ x y : ℝ, 3 * y - 4 * x = 0 ∧ 4 * x + y = 8 ∧ x = 3/2 ∧ y = 2) ∧
  -- System (2)
  (∃ x y : ℝ, x + y = 3 ∧ (x - 1)/4 + y/2 = 3/4 ∧ x = 2 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l1654_165486


namespace NUMINAMATH_CALUDE_total_earnings_l1654_165425

-- Define pizza types
inductive PizzaType
| Margherita
| Pepperoni
| VeggieSupreme
| MeatLovers
| Hawaiian

-- Define pizza prices
def slicePrice (p : PizzaType) : ℚ :=
  match p with
  | .Margherita => 3
  | .Pepperoni => 4
  | .VeggieSupreme => 5
  | .MeatLovers => 6
  | .Hawaiian => 4.5

def wholePizzaPrice (p : PizzaType) : ℚ :=
  match p with
  | .Margherita => 15
  | .Pepperoni => 18
  | .VeggieSupreme => 22
  | .MeatLovers => 25
  | .Hawaiian => 20

-- Define discount and promotion rules
def wholeDiscountRate : ℚ := 0.1
def wholeDiscountThreshold : ℕ := 3
def regularToppingPrice : ℚ := 2
def weekendToppingPrice : ℚ := 1
def happyHourPrice : ℚ := 3

-- Define sales data
structure SalesData where
  margheritaSlices : ℕ
  margheritaHappyHour : ℕ
  pepperoniSlices : ℕ
  pepperoniHappyHour : ℕ
  pepperoniToppings : ℕ
  veggieSupremeWhole : ℕ
  veggieSupremeToppings : ℕ
  margheritaWholePackage : ℕ
  meatLoversSlices : ℕ
  meatLoversHappyHour : ℕ
  hawaiianSlices : ℕ
  hawaiianToppings : ℕ
  pepperoniWholeWeekend : ℕ
  pepperoniWholeToppings : ℕ

def salesData : SalesData := {
  margheritaSlices := 24,
  margheritaHappyHour := 12,
  pepperoniSlices := 16,
  pepperoniHappyHour := 8,
  pepperoniToppings := 6,
  veggieSupremeWhole := 4,
  veggieSupremeToppings := 8,
  margheritaWholePackage := 3,
  meatLoversSlices := 20,
  meatLoversHappyHour := 10,
  hawaiianSlices := 12,
  hawaiianToppings := 4,
  pepperoniWholeWeekend := 1,
  pepperoniWholeToppings := 3
}

-- Theorem statement
theorem total_earnings (data : SalesData) :
  let earnings := 
    (data.margheritaSlices - data.margheritaHappyHour) * slicePrice PizzaType.Margherita +
    data.margheritaHappyHour * happyHourPrice +
    (data.pepperoniSlices - data.pepperoniHappyHour) * slicePrice PizzaType.Pepperoni +
    data.pepperoniHappyHour * happyHourPrice +
    data.pepperoniToppings * weekendToppingPrice +
    data.veggieSupremeWhole * wholePizzaPrice PizzaType.VeggieSupreme +
    data.veggieSupremeToppings * weekendToppingPrice +
    (data.margheritaWholePackage * wholePizzaPrice PizzaType.Margherita) * (1 - wholeDiscountRate) +
    (data.meatLoversSlices - data.meatLoversHappyHour) * slicePrice PizzaType.MeatLovers +
    data.meatLoversHappyHour * happyHourPrice +
    data.hawaiianSlices * slicePrice PizzaType.Hawaiian +
    data.hawaiianToppings * weekendToppingPrice +
    data.pepperoniWholeWeekend * wholePizzaPrice PizzaType.Pepperoni +
    data.pepperoniWholeToppings * weekendToppingPrice
  earnings = 439.5 := by sorry


end NUMINAMATH_CALUDE_total_earnings_l1654_165425


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l1654_165479

theorem gcd_factorial_eight_ten : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l1654_165479


namespace NUMINAMATH_CALUDE_cylinder_sphere_intersection_ellipse_sum_l1654_165418

/-- Configuration of cylinder and spheres --/
structure Configuration where
  cylinder_radius : ℝ
  sphere_radius : ℝ
  sphere_center_distance : ℝ

/-- Ellipse formed by intersection of plane with cylinder --/
structure IntersectionEllipse where
  major_axis : ℝ
  minor_axis : ℝ

/-- Theorem statement --/
theorem cylinder_sphere_intersection_ellipse_sum
  (config : Configuration)
  (ellipse : IntersectionEllipse)
  (h_cylinder : config.cylinder_radius = 6)
  (h_sphere : config.sphere_radius = 6)
  (h_distance : config.sphere_center_distance = 13)
  (h_tangent : True)  -- Represents the plane being tangent to both spheres
  (h_intersect : True)  -- Represents the plane intersecting the cylinder surface
  : ellipse.major_axis + ellipse.minor_axis = 25 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_sphere_intersection_ellipse_sum_l1654_165418


namespace NUMINAMATH_CALUDE_probability_even_and_less_equal_three_l1654_165411

def dice_sides : ℕ := 6

def prob_even_first_die : ℚ := 1 / 2

def prob_less_equal_three_second_die : ℚ := 1 / 2

theorem probability_even_and_less_equal_three (independence : True) :
  prob_even_first_die * prob_less_equal_three_second_die = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_and_less_equal_three_l1654_165411


namespace NUMINAMATH_CALUDE_smallest_c_for_max_sine_l1654_165432

theorem smallest_c_for_max_sine (c : ℝ) : c > 0 → (
  c = π ↔ 
  ∀ x : ℝ, x = -π/4 → 3 * Real.sin (2*x + c) = 3 * Real.sin (π/2) ∧
  ∀ d : ℝ, d > 0 ∧ d < c → ∃ y : ℝ, y = -π/4 ∧ 3 * Real.sin (2*y + d) < 3 * Real.sin (π/2)
) := by sorry

end NUMINAMATH_CALUDE_smallest_c_for_max_sine_l1654_165432


namespace NUMINAMATH_CALUDE_multiple_of_ab_l1654_165438

theorem multiple_of_ab (a b : ℕ+) : 
  (∃ k : ℕ, a.val ^ 2017 + b.val = k * a.val * b.val) ↔ 
  ((a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 2^2017)) := by
sorry

end NUMINAMATH_CALUDE_multiple_of_ab_l1654_165438


namespace NUMINAMATH_CALUDE_no_diametrical_opposition_possible_l1654_165473

/-- Represents a circular arrangement of numbers from 1 to 2014 -/
def CircularArrangement := Fin 2014 → Fin 2014

/-- Checks if a swap between two adjacent positions is valid -/
def validSwap (arr : CircularArrangement) (pos : Fin 2014) : Prop :=
  arr pos + arr ((pos + 1) % 2014) ≠ 2015

/-- Represents a sequence of swaps -/
def SwapSequence := List (Fin 2014)

/-- Applies a sequence of swaps to an arrangement -/
def applySwaps (initial : CircularArrangement) (swaps : SwapSequence) : CircularArrangement :=
  sorry

/-- Checks if a number is diametrically opposite its initial position -/
def isDiametricallyOpposite (initial final : CircularArrangement) (pos : Fin 2014) : Prop :=
  final pos = initial ((pos + 1007) % 2014)

/-- The main theorem stating that it's impossible to achieve diametrical opposition for all numbers -/
theorem no_diametrical_opposition_possible :
  ∀ (initial : CircularArrangement) (swaps : SwapSequence),
    (∀ (pos : Fin 2014), validSwap (applySwaps initial swaps) pos) →
    ¬(∀ (pos : Fin 2014), isDiametricallyOpposite initial (applySwaps initial swaps) pos) :=
  sorry

end NUMINAMATH_CALUDE_no_diametrical_opposition_possible_l1654_165473


namespace NUMINAMATH_CALUDE_cheerful_not_green_l1654_165491

-- Define the universe of snakes
variable (Snake : Type)

-- Define properties of snakes
variable (green : Snake → Prop)
variable (cheerful : Snake → Prop)
variable (can_sing : Snake → Prop)
variable (can_multiply : Snake → Prop)

-- Define the conditions
axiom all_cheerful_can_sing : ∀ s : Snake, cheerful s → can_sing s
axiom no_green_can_multiply : ∀ s : Snake, green s → ¬can_multiply s
axiom cannot_multiply_cannot_sing : ∀ s : Snake, ¬can_multiply s → ¬can_sing s

-- Theorem to prove
theorem cheerful_not_green : ∀ s : Snake, cheerful s → ¬green s := by
  sorry

end NUMINAMATH_CALUDE_cheerful_not_green_l1654_165491


namespace NUMINAMATH_CALUDE_circle_line_distance_l1654_165415

theorem circle_line_distance (m : ℝ) : 
  (∃ (A B C : ℝ × ℝ), 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧ (C.1^2 + C.2^2 = 4) ∧
    (|A.2 + A.1 - m| / Real.sqrt 2 = 1) ∧
    (|B.2 + B.1 - m| / Real.sqrt 2 = 1) ∧
    (|C.2 + C.1 - m| / Real.sqrt 2 = 1)) →
  -Real.sqrt 2 ≤ m ∧ m ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_l1654_165415


namespace NUMINAMATH_CALUDE_reach_destination_in_time_l1654_165437

/-- The distance to the destination in kilometers -/
def destination_distance : ℝ := 62

/-- The walking speed in km/hr -/
def walking_speed : ℝ := 5

/-- The car speed in km/hr -/
def car_speed : ℝ := 50

/-- The maximum time allowed to reach the destination in hours -/
def max_time : ℝ := 3

/-- A strategy represents a plan for A, B, and C to reach the destination -/
structure Strategy where
  -- Add necessary fields to represent the strategy
  dummy : Unit

/-- Calculates the time taken to execute a given strategy -/
def time_taken (s : Strategy) : ℝ :=
  -- Implement the calculation of time taken for the strategy
  sorry

/-- Theorem stating that there exists a strategy to reach the destination in less than the maximum allowed time -/
theorem reach_destination_in_time :
  ∃ (s : Strategy), time_taken s < max_time :=
sorry

end NUMINAMATH_CALUDE_reach_destination_in_time_l1654_165437


namespace NUMINAMATH_CALUDE_circular_paper_pieces_for_square_border_l1654_165439

theorem circular_paper_pieces_for_square_border (side_length : ℝ) (pieces_per_circle : ℕ) : 
  side_length = 10 → pieces_per_circle = 20 → (4 * side_length) / (2 * π) * pieces_per_circle = 40 := by
  sorry

#check circular_paper_pieces_for_square_border

end NUMINAMATH_CALUDE_circular_paper_pieces_for_square_border_l1654_165439


namespace NUMINAMATH_CALUDE_part_one_part_two_l1654_165480

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x| ≥ 2}
def B (a : ℝ) : Set ℝ := {x | (x - 2*a)*(x + 3) < 0}

-- Part I
theorem part_one : 
  A 3 ∩ B 3 = {x | -3 < x ∧ x ≤ -2 ∨ 2 ≤ x ∧ x < 6} := by sorry

-- Part II
theorem part_two (a : ℝ) (h : a > 0) :
  A a ∪ B a = Set.univ → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1654_165480


namespace NUMINAMATH_CALUDE_regular_24gon_symmetry_sum_l1654_165433

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Additional properties of regular polygons can be added here if needed

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def smallestRotationAngle (p : RegularPolygon n) : ℝ := sorry

/-- Theorem: For a regular 24-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is 39 -/
theorem regular_24gon_symmetry_sum :
  ∀ (p : RegularPolygon 24),
    (linesOfSymmetry p : ℝ) + smallestRotationAngle p = 39 := by sorry

end NUMINAMATH_CALUDE_regular_24gon_symmetry_sum_l1654_165433


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1654_165403

theorem simple_interest_problem (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2000 → P = 2500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1654_165403


namespace NUMINAMATH_CALUDE_rectangle_area_l1654_165447

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 6 →
  ratio = 3 →
  (2 * r) * (ratio * 2 * r) = 432 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1654_165447


namespace NUMINAMATH_CALUDE_soccer_game_total_goals_l1654_165482

theorem soccer_game_total_goals :
  let team_a_first_half : ℕ := 8
  let team_b_first_half : ℕ := team_a_first_half / 2
  let team_b_second_half : ℕ := team_a_first_half
  let team_a_second_half : ℕ := team_b_second_half - 2
  team_a_first_half + team_b_first_half + team_a_second_half + team_b_second_half = 26 :=
by sorry

end NUMINAMATH_CALUDE_soccer_game_total_goals_l1654_165482


namespace NUMINAMATH_CALUDE_power_multiplication_l1654_165470

theorem power_multiplication (a : ℝ) : a^6 * a^2 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1654_165470


namespace NUMINAMATH_CALUDE_union_equals_N_implies_a_in_range_l1654_165483

/-- Given sets M and N, if their union equals N, then a is in the interval [-2, 2] -/
theorem union_equals_N_implies_a_in_range (a : ℝ) :
  let M := {x : ℝ | x * (x - a - 1) < 0}
  let N := {x : ℝ | x^2 - 2*x - 3 ≤ 0}
  (M ∪ N = N) → a ∈ Set.Icc (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_N_implies_a_in_range_l1654_165483


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1654_165423

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 1 ∧ 
  (5026 - x) % 5 = 0 ∧ 
  ∀ (y : ℕ), y < x → (5026 - y) % 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1654_165423


namespace NUMINAMATH_CALUDE_store_pricing_strategy_l1654_165426

/-- Calculates the sale price given the cost price and profit percentage -/
def calculateSalePrice (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  costPrice * (1 + profitPercentage / 100)

/-- Represents the store's pricing strategy -/
theorem store_pricing_strategy :
  let costA : ℚ := 320
  let costB : ℚ := 480
  let costC : ℚ := 600
  let profitA : ℚ := 50
  let profitB : ℚ := 70
  let profitC : ℚ := 40
  (calculateSalePrice costA profitA = 480) ∧
  (calculateSalePrice costB profitB = 816) ∧
  (calculateSalePrice costC profitC = 840) := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_strategy_l1654_165426


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1654_165468

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, a * (3 : ℝ)^x + (3 : ℝ)^(-x) = 3) ↔ 
  a ∈ Set.Iic (0 : ℝ) ∪ {9/4} :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1654_165468


namespace NUMINAMATH_CALUDE_ones_digit_of_13_power_power_cycle_of_3_main_theorem_l1654_165416

theorem ones_digit_of_13_power (n : ℕ) : n > 0 → (13^n) % 10 = (3^n) % 10 := by sorry

theorem power_cycle_of_3 (n : ℕ) : (3^n) % 10 = (3^(n % 4)) % 10 := by sorry

theorem main_theorem : (13^(13 * (12^12))) % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_13_power_power_cycle_of_3_main_theorem_l1654_165416


namespace NUMINAMATH_CALUDE_spinach_quiche_egg_volume_l1654_165419

/-- Represents the ingredients and their quantities in a spinach quiche recipe. -/
structure SpinachQuiche where
  raw_spinach : ℝ
  cooked_spinach_ratio : ℝ
  cream_cheese : ℝ
  total_volume : ℝ

/-- Calculates the volume of eggs used in the spinach quiche recipe. -/
def egg_volume (quiche : SpinachQuiche) : ℝ :=
  quiche.total_volume - (quiche.raw_spinach * quiche.cooked_spinach_ratio + quiche.cream_cheese)

/-- Theorem stating that the volume of eggs in the given spinach quiche recipe is 4 ounces. -/
theorem spinach_quiche_egg_volume :
  let quiche : SpinachQuiche := {
    raw_spinach := 40,
    cooked_spinach_ratio := 0.2,
    cream_cheese := 6,
    total_volume := 18
  }
  egg_volume quiche = 4 := by sorry

end NUMINAMATH_CALUDE_spinach_quiche_egg_volume_l1654_165419


namespace NUMINAMATH_CALUDE_bob_puppy_savings_l1654_165451

/-- The minimum number of additional weeks Bob must win first place to buy a puppy -/
def minimum_additional_weeks (initial_weeks : ℕ) (prize_per_week : ℕ) (puppy_cost : ℕ) : ℕ :=
  let initial_earnings := initial_weeks * prize_per_week
  let remaining_cost := puppy_cost - initial_earnings
  (remaining_cost + prize_per_week - 1) / prize_per_week

theorem bob_puppy_savings : minimum_additional_weeks 2 100 1000 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bob_puppy_savings_l1654_165451


namespace NUMINAMATH_CALUDE_x_range_l1654_165427

theorem x_range (x : ℝ) : (Real.sqrt ((5 - x)^2) = x - 5) → x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1654_165427


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1654_165404

-- Define the repeating decimals
def repeating_decimal_2 : ℚ := 2 / 9
def repeating_decimal_02 : ℚ := 2 / 99

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_2 + repeating_decimal_02 = 8 / 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1654_165404


namespace NUMINAMATH_CALUDE_square_side_length_l1654_165445

/-- Given a rectangle with sides 9 cm and 16 cm and a square with the same area,
    prove that the side length of the square is 12 cm. -/
theorem square_side_length (rectangle_width : ℝ) (rectangle_length : ℝ) (square_side : ℝ) :
  rectangle_width = 9 →
  rectangle_length = 16 →
  rectangle_width * rectangle_length = square_side * square_side →
  square_side = 12 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l1654_165445


namespace NUMINAMATH_CALUDE_log_ratio_squared_l1654_165405

theorem log_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * (Real.log a)^2 - 4 * (Real.log a) + 1 = 0) →
  (2 * (Real.log b)^2 - 4 * (Real.log b) + 1 = 0) →
  ((Real.log (a / b))^2 = 2) := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l1654_165405


namespace NUMINAMATH_CALUDE_total_blue_balloons_l1654_165414

theorem total_blue_balloons (alyssa_balloons sandy_balloons sally_balloons : ℕ)
  (h1 : alyssa_balloons = 37)
  (h2 : sandy_balloons = 28)
  (h3 : sally_balloons = 39) :
  alyssa_balloons + sandy_balloons + sally_balloons = 104 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l1654_165414


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1654_165429

theorem polynomial_factorization (x y z : ℝ) :
  2 * x^3 - x^2 * z - 4 * x^2 * y + 2 * x * y * z + 2 * x * y^2 - y^2 * z = (2 * x - z) * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1654_165429


namespace NUMINAMATH_CALUDE_johns_hat_cost_l1654_165428

/-- The total cost of John's hats -/
def total_cost (weeks : ℕ) (odd_cost even_cost : ℕ) : ℕ :=
  let total_days := weeks * 7
  let odd_days := total_days / 2
  let even_days := total_days / 2
  odd_days * odd_cost + even_days * even_cost

/-- Theorem stating that the total cost of John's hats is $7350 -/
theorem johns_hat_cost :
  total_cost 20 45 60 = 7350 := by
  sorry

end NUMINAMATH_CALUDE_johns_hat_cost_l1654_165428


namespace NUMINAMATH_CALUDE_min_lateral_perimeter_is_six_l1654_165458

/-- Represents a rectangular parallelepiped with a square base -/
structure Parallelepiped where
  base_side : ℝ
  height : ℝ

/-- The volume of a parallelepiped -/
def volume (p : Parallelepiped) : ℝ :=
  p.base_side^2 * p.height

/-- The perimeter of a lateral face of a parallelepiped -/
def lateral_perimeter (p : Parallelepiped) : ℝ :=
  2 * p.base_side + 2 * p.height

/-- Theorem: The minimum perimeter of a lateral face among all rectangular
    parallelepipeds with volume 4 and square bases is 6 -/
theorem min_lateral_perimeter_is_six :
  ∀ p : Parallelepiped, volume p = 4 → lateral_perimeter p ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_min_lateral_perimeter_is_six_l1654_165458


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1654_165455

/-- The quadratic function y = x^2 - ax + a + 3 -/
def f (a x : ℝ) : ℝ := x^2 - a*x + a + 3

theorem quadratic_function_properties (a : ℝ) :
  (∃ x, f a x = 0) ↔ (a ≤ -2 ∨ a ≥ 6) ∧
  (∀ x, f a x ≥ 4 ↔ 
    (a > 2 ∧ (x ≤ 1 ∨ x ≥ a - 1)) ∨
    (a = 2 ∧ true) ∨
    (a < 2 ∧ (x ≤ a - 1 ∨ x ≥ 1))) ∧
  ((∃ x ∈ Set.Icc 2 4, f a x = 0) → a ∈ Set.Icc 6 7) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l1654_165455


namespace NUMINAMATH_CALUDE_telephone_number_increase_l1654_165485

/-- The number of possible n-digit telephone numbers with a non-zero first digit -/
def telephone_numbers (n : ℕ) : ℕ := 9 * 10^(n - 1)

/-- The increase in telephone numbers when moving from 6 to 7 digits -/
def increase_in_numbers : ℕ := telephone_numbers 7 - telephone_numbers 6

theorem telephone_number_increase :
  increase_in_numbers = 81 * 10^5 := by
  sorry

end NUMINAMATH_CALUDE_telephone_number_increase_l1654_165485


namespace NUMINAMATH_CALUDE_octagon_area_l1654_165467

/-- The area of a regular octagon inscribed in a circle with area 256π -/
theorem octagon_area (circle_area : ℝ) (h : circle_area = 256 * Real.pi) :
  ∃ (octagon_area : ℝ), octagon_area = 1024 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l1654_165467


namespace NUMINAMATH_CALUDE_prob_at_least_one_consonant_l1654_165490

def word : String := "barkhint"

def is_consonant (c : Char) : Bool :=
  c ∈ ['b', 'r', 'k', 'h', 'n', 't']

def num_letters : Nat := word.length

def num_vowels : Nat := word.toList.filter (fun c => !is_consonant c) |>.length

def num_ways_to_select_two : Nat := num_letters * (num_letters - 1) / 2

def num_ways_to_select_two_vowels : Nat := num_vowels * (num_vowels - 1) / 2

theorem prob_at_least_one_consonant :
  (1 : ℚ) - (num_ways_to_select_two_vowels : ℚ) / num_ways_to_select_two = 27 / 28 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_consonant_l1654_165490


namespace NUMINAMATH_CALUDE_three_roots_implies_a_range_l1654_165417

theorem three_roots_implies_a_range (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ x : ℝ, x^2 = a * Real.exp x ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  0 < a ∧ a < 4 / Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_three_roots_implies_a_range_l1654_165417


namespace NUMINAMATH_CALUDE_problem_statement_l1654_165440

theorem problem_statement (x y : ℤ) (a b : ℤ) : 
  (∃ k₁ k₂ : ℤ, x - 5 = 7 * k₁ ∧ y + 7 = 7 * k₂) →
  (∃ k₃ : ℤ, x^2 + y^3 = 11 * k₃) →
  x = 7 * a + 5 →
  y = 7 * b - 7 →
  (y - x) / 13 = (7 * (b - a) - 12) / 13 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1654_165440
