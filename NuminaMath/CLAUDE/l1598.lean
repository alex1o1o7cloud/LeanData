import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solution_l1598_159830

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation that z satisfies
def satisfies_equation (z : ℂ) : Prop := (2 * i) / z = 1 - i

-- Theorem statement
theorem complex_equation_solution :
  ∀ z : ℂ, satisfies_equation z → z = -1 + i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1598_159830


namespace NUMINAMATH_CALUDE_matches_played_calculation_l1598_159856

/-- A football competition with a specific scoring system and number of matches --/
structure FootballCompetition where
  totalMatches : ℕ
  pointsForWin : ℕ
  pointsForDraw : ℕ
  pointsForLoss : ℕ

/-- A team's current state in the competition --/
structure TeamState where
  pointsScored : ℕ
  matchesPlayed : ℕ

/-- Theorem stating the number of matches played by the team --/
theorem matches_played_calculation (comp : FootballCompetition)
    (state : TeamState) (minWinsNeeded : ℕ) (targetPoints : ℕ) :
    comp.totalMatches = 20 ∧
    comp.pointsForWin = 3 ∧
    comp.pointsForDraw = 1 ∧
    comp.pointsForLoss = 0 ∧
    state.pointsScored = 14 ∧
    minWinsNeeded = 6 ∧
    targetPoints = 40 →
    state.matchesPlayed = 14 := by
  sorry

end NUMINAMATH_CALUDE_matches_played_calculation_l1598_159856


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l1598_159846

-- Define the cryptarithm equation
def cryptarithm (A B C : ℕ) : Prop :=
  A < 10 ∧ B < 10 ∧ C < 10 ∧ 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  100 * C + 10 * B + A + 100 * A + 10 * A + A = 10 * B + A

-- Theorem statement
theorem cryptarithm_solution :
  ∃! (A B C : ℕ), cryptarithm A B C ∧ A = 5 ∧ B = 9 ∧ C = 3 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l1598_159846


namespace NUMINAMATH_CALUDE_larger_integer_value_l1598_159819

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) : 
  max a b = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l1598_159819


namespace NUMINAMATH_CALUDE_sum_of_squares_l1598_159802

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 22) (h2 : x * y = 12) : x^2 + y^2 = 460 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1598_159802


namespace NUMINAMATH_CALUDE_f_4_1981_l1598_159817

/-- Definition of the function f satisfying the given conditions -/
def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| x + 1, 0 => f x 1
| x + 1, y + 1 => f x (f (x + 1) y)

/-- Theorem stating that f(4, 1981) equals 2^1984 - 3 -/
theorem f_4_1981 : f 4 1981 = 2^1984 - 3 := by
  sorry

end NUMINAMATH_CALUDE_f_4_1981_l1598_159817


namespace NUMINAMATH_CALUDE_gary_money_after_sale_l1598_159877

theorem gary_money_after_sale (initial_amount selling_price : ℝ) 
  (h1 : initial_amount = 73.0) 
  (h2 : selling_price = 55.0) : 
  initial_amount + selling_price = 128.0 :=
by sorry

end NUMINAMATH_CALUDE_gary_money_after_sale_l1598_159877


namespace NUMINAMATH_CALUDE_sum_of_tenth_powers_l1598_159883

/-- Given a sequence of sums of powers of a and b, prove that a^10 + b^10 = 123 -/
theorem sum_of_tenth_powers (a b : ℝ) 
  (sum1 : a + b = 1)
  (sum2 : a^2 + b^2 = 3)
  (sum3 : a^3 + b^3 = 4)
  (sum4 : a^4 + b^4 = 7)
  (sum5 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tenth_powers_l1598_159883


namespace NUMINAMATH_CALUDE_price_reduction_equation_l1598_159868

/-- Represents the price reduction percentage -/
def x : ℝ := sorry

/-- The original price of the medicine -/
def original_price : ℝ := 25

/-- The final price of the medicine after two reductions -/
def final_price : ℝ := 16

/-- Theorem stating the relationship between the original price, 
    final price, and the reduction percentage -/
theorem price_reduction_equation : 
  original_price * (1 - x)^2 = final_price := by sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l1598_159868


namespace NUMINAMATH_CALUDE_fraction_addition_l1598_159806

theorem fraction_addition : (3 : ℚ) / 8 + (9 : ℚ) / 12 = (9 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1598_159806


namespace NUMINAMATH_CALUDE_total_interest_calculation_l1598_159869

/-- Calculates the total interest for a loan split into two parts with different interest rates -/
theorem total_interest_calculation 
  (A B : ℝ) 
  (h1 : A > 0) 
  (h2 : B > 0) 
  (h3 : A + B = 10000) : 
  ∃ I : ℝ, I = 0.08 * A + 0.1 * B := by
  sorry

#check total_interest_calculation

end NUMINAMATH_CALUDE_total_interest_calculation_l1598_159869


namespace NUMINAMATH_CALUDE_tigrasha_first_snezhok_last_l1598_159843

-- Define the kittens
inductive Kitten : Type
| Chernysh : Kitten
| Tigrasha : Kitten
| Snezhok : Kitten
| Pushok : Kitten

-- Define the eating speed for each kitten
def eating_speed (k : Kitten) : ℕ :=
  match k with
  | Kitten.Chernysh => 2
  | Kitten.Tigrasha => 5
  | Kitten.Snezhok => 3
  | Kitten.Pushok => 4

-- Define the initial number of sausages (same for all kittens)
def initial_sausages : ℕ := 7

-- Define the time to finish eating for each kitten
def time_to_finish (k : Kitten) : ℚ :=
  (initial_sausages : ℚ) / (eating_speed k : ℚ)

-- Theorem statement
theorem tigrasha_first_snezhok_last :
  (∀ k : Kitten, k ≠ Kitten.Tigrasha → time_to_finish Kitten.Tigrasha ≤ time_to_finish k) ∧
  (∀ k : Kitten, k ≠ Kitten.Snezhok → time_to_finish k ≤ time_to_finish Kitten.Snezhok) :=
sorry

end NUMINAMATH_CALUDE_tigrasha_first_snezhok_last_l1598_159843


namespace NUMINAMATH_CALUDE_inequality_proof_l1598_159826

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1598_159826


namespace NUMINAMATH_CALUDE_prime_power_divisibility_l1598_159848

theorem prime_power_divisibility (p : ℕ) (x : ℕ) (h_prime : Nat.Prime p) :
  1 ≤ x ∧ x ≤ 2 * p ∧ (x ^ (p - 1) ∣ (p - 1) ^ x + 1) →
  (p = 2 ∧ (x = 1 ∨ x = 2)) ∨ (p = 3 ∧ (x = 1 ∨ x = 3)) ∨ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_prime_power_divisibility_l1598_159848


namespace NUMINAMATH_CALUDE_find_d_l1598_159866

theorem find_d : ∃ d : ℝ, 
  (∃ x : ℤ, x^2 + 5*x - 36 = 0 ∧ x = ⌊d⌋) ∧ 
  (∃ y : ℝ, 3*y^2 - 11*y + 2 = 0 ∧ y = d - ⌊d⌋) ∧
  d = 13/3 := by
sorry

end NUMINAMATH_CALUDE_find_d_l1598_159866


namespace NUMINAMATH_CALUDE_prime_divides_abc_l1598_159884

theorem prime_divides_abc (p a b c : ℤ) (hp : Prime p)
  (h1 : (6 : ℤ) ∣ p + 1)
  (h2 : p ∣ a + b + c)
  (h3 : p ∣ a^4 + b^4 + c^4) :
  p ∣ a ∧ p ∣ b ∧ p ∣ c := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_abc_l1598_159884


namespace NUMINAMATH_CALUDE_line_slope_l1598_159898

theorem line_slope (x y : ℝ) (h : 4 * x + 7 * y = 28) : 
  (y - 4) / x = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l1598_159898


namespace NUMINAMATH_CALUDE_pink_cubes_count_l1598_159862

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Represents a colored cube with a given side length and number of colored faces -/
structure ColoredCube extends Cube where
  coloredFaces : ℕ

/-- Calculates the number of smaller cubes with color when a large cube is cut -/
def coloredCubesCount (largeCube : Cube) (coloredFaces : ℕ) : ℕ :=
  sorry

theorem pink_cubes_count :
  let largeCube : Cube := ⟨125⟩
  let coloredFaces : ℕ := 2
  coloredCubesCount largeCube coloredFaces = 46 := by
  sorry

end NUMINAMATH_CALUDE_pink_cubes_count_l1598_159862


namespace NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l1598_159861

/-- If x varies as the fourth power of y, and y varies as the cube root of z,
    then x varies as the (4/3)th power of z. -/
theorem x_varies_as_four_thirds_power_of_z 
  (x y z : ℝ) 
  (hxy : ∃ (a : ℝ), x = a * y^4) 
  (hyz : ∃ (b : ℝ), y = b * z^(1/3)) :
  ∃ (c : ℝ), x = c * z^(4/3) := by
sorry

end NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l1598_159861


namespace NUMINAMATH_CALUDE_product_real_condition_l1598_159872

theorem product_real_condition (a b c d : ℝ) :
  (∃ (x : ℝ), (a + b * Complex.I) * (c + d * Complex.I) = x) ↔ a * d + b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_real_condition_l1598_159872


namespace NUMINAMATH_CALUDE_total_checks_is_30_l1598_159887

/-- The number of $50 checks -/
def F : ℕ := sorry

/-- The number of $100 checks -/
def H : ℕ := sorry

/-- The total worth of all checks is $1800 -/
axiom total_worth : 50 * F + 100 * H = 1800

/-- The average of remaining checks after removing 18 $50 checks is $75 -/
axiom remaining_average : (1800 - 18 * 50) / (F + H - 18) = 75

/-- The total number of travelers checks -/
def total_checks : ℕ := F + H

/-- Theorem: The total number of travelers checks is 30 -/
theorem total_checks_is_30 : total_checks = 30 := by sorry

end NUMINAMATH_CALUDE_total_checks_is_30_l1598_159887


namespace NUMINAMATH_CALUDE_joan_marbles_l1598_159890

theorem joan_marbles (mary_marbles : ℕ) (total_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : total_marbles = 12) :
  total_marbles - mary_marbles = 3 :=
by sorry

end NUMINAMATH_CALUDE_joan_marbles_l1598_159890


namespace NUMINAMATH_CALUDE_percent_of_percent_l1598_159881

theorem percent_of_percent (y : ℝ) (h : y ≠ 0) :
  (0.6 * (0.3 * y)) / y * 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1598_159881


namespace NUMINAMATH_CALUDE_shoe_discount_ratio_l1598_159822

/-- Proves the ratio of extra discount to total amount before discount is 1:4 --/
theorem shoe_discount_ratio :
  let first_pair_price : ℚ := 40
  let second_pair_price : ℚ := 60
  let discount_rate : ℚ := 1/2
  let total_paid : ℚ := 60
  let cheaper_pair_price := min first_pair_price second_pair_price
  let discount_amount := discount_rate * cheaper_pair_price
  let total_before_extra_discount := first_pair_price + second_pair_price - discount_amount
  let extra_discount := total_before_extra_discount - total_paid
  extra_discount / total_before_extra_discount = 1/4 := by
sorry

end NUMINAMATH_CALUDE_shoe_discount_ratio_l1598_159822


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l1598_159810

theorem middle_part_of_proportional_division (total : ℚ) (a b c : ℚ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 ∧ a = 2 ∧ b = 3/2 ∧ c = 1/2 →
  (b * total) / (a + b + c) = 39 := by
sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l1598_159810


namespace NUMINAMATH_CALUDE_cardboard_box_square_cutout_l1598_159853

theorem cardboard_box_square_cutout (length width area : ℝ) 
  (h1 : length = 80)
  (h2 : width = 60)
  (h3 : area = 1500) :
  ∃ (x : ℝ), x > 0 ∧ x < 30 ∧ (length - 2*x) * (width - 2*x) = area ∧ x = 15 :=
sorry

end NUMINAMATH_CALUDE_cardboard_box_square_cutout_l1598_159853


namespace NUMINAMATH_CALUDE_area_of_connected_paper_l1598_159899

/-- The area of connected colored paper sheets -/
theorem area_of_connected_paper (n : ℕ) (side_length overlap : ℝ) :
  n > 0 →
  side_length > 0 →
  overlap ≥ 0 →
  overlap < side_length →
  let total_length := side_length + (n - 1 : ℝ) * (side_length - overlap)
  let area := total_length * side_length
  n = 6 ∧ side_length = 30 ∧ overlap = 7 →
  area = 4350 := by
  sorry

end NUMINAMATH_CALUDE_area_of_connected_paper_l1598_159899


namespace NUMINAMATH_CALUDE_sin_cos_equality_solution_l1598_159828

theorem sin_cos_equality_solution :
  ∃ x : ℝ, x * (180 / π) = 9 ∧ Real.sin (4 * x) * Real.sin (6 * x) = Real.cos (4 * x) * Real.cos (6 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equality_solution_l1598_159828


namespace NUMINAMATH_CALUDE_inequality_solution_l1598_159858

theorem inequality_solution (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 3)) < 1 / 4) ↔
  (x < -3 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1598_159858


namespace NUMINAMATH_CALUDE_impossible_distance_l1598_159864

/-- Two circles with no common points -/
structure DisjointCircles where
  r₁ : ℝ
  r₂ : ℝ
  d : ℝ
  h₁ : r₁ = 2
  h₂ : r₂ = 5
  h₃ : d < r₂ - r₁ ∨ d > r₂ + r₁

theorem impossible_distance (c : DisjointCircles) : c.d ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_impossible_distance_l1598_159864


namespace NUMINAMATH_CALUDE_square_difference_equals_six_l1598_159804

theorem square_difference_equals_six (a b : ℝ) 
  (sum_eq : a + b = 2) 
  (diff_eq : a - b = 3) : 
  a^2 - b^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_difference_equals_six_l1598_159804


namespace NUMINAMATH_CALUDE_stephanie_distance_l1598_159813

/-- Calculates the distance traveled given time and speed -/
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Proves that running for 3 hours at 5 miles per hour results in a distance of 15 miles -/
theorem stephanie_distance :
  let time : ℝ := 3
  let speed : ℝ := 5
  distance time speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_distance_l1598_159813


namespace NUMINAMATH_CALUDE_xyz_inequality_l1598_159870

theorem xyz_inequality (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) 
  (h_eq : x^2 + y^2 + z^2 + x*y*z = 4) : 
  x*y*z ≤ x*y + y*z + z*x ∧ x*y + y*z + z*x ≤ x*y*z + 2 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l1598_159870


namespace NUMINAMATH_CALUDE_absent_children_count_l1598_159800

/-- Proves that the number of absent children is 32 given the conditions of the sweet distribution problem --/
theorem absent_children_count (total_children : ℕ) (original_sweets_per_child : ℕ) (extra_sweets : ℕ) : 
  total_children = 112 →
  original_sweets_per_child = 15 →
  extra_sweets = 6 →
  (total_children - (total_children - 32)) * (original_sweets_per_child + extra_sweets) = total_children * original_sweets_per_child :=
by sorry

end NUMINAMATH_CALUDE_absent_children_count_l1598_159800


namespace NUMINAMATH_CALUDE_archery_competition_theorem_l1598_159888

/-- Represents the point system for the archery competition -/
def PointSystem : Fin 4 → ℕ
  | 0 => 11  -- 1st place
  | 1 => 7   -- 2nd place
  | 2 => 5   -- 3rd place
  | 3 => 2   -- 4th place

/-- Represents the participation counts for each place -/
structure Participation where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the product of points based on participation -/
def pointProduct (p : Participation) : ℕ :=
  (PointSystem 0) ^ p.first * 
  (PointSystem 1) ^ p.second * 
  (PointSystem 2) ^ p.third * 
  (PointSystem 3) ^ p.fourth

/-- Calculates the total number of participations -/
def totalParticipations (p : Participation) : ℕ :=
  p.first + p.second + p.third + p.fourth

/-- Theorem: If the product of points is 38500, then the total participations is 7 -/
theorem archery_competition_theorem (p : Participation) :
  pointProduct p = 38500 → totalParticipations p = 7 := by
  sorry


end NUMINAMATH_CALUDE_archery_competition_theorem_l1598_159888


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_y_l1598_159809

theorem tan_theta_in_terms_of_x_y (θ x y : ℝ) (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : Real.sin (θ/2) = Real.sqrt ((y - x)/(y + x))) : 
  Real.tan θ = (2 * Real.sqrt (x * y)) / (3 * x - y) := by
sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_y_l1598_159809


namespace NUMINAMATH_CALUDE_derivative_at_pi_over_four_l1598_159833

open Real

theorem derivative_at_pi_over_four :
  let f (x : ℝ) := cos x * (sin x - cos x)
  let f' := deriv f
  f' (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_over_four_l1598_159833


namespace NUMINAMATH_CALUDE_line_slope_theorem_l1598_159845

/-- Given a line with equation x = 5y + 5 passing through points (m, n) and (m + a, n + p),
    where p = 0.4, prove that a = 2. -/
theorem line_slope_theorem (m n a p : ℝ) : 
  p = 0.4 →
  m = 5 * n + 5 →
  (m + a) = 5 * (n + p) + 5 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_theorem_l1598_159845


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1598_159814

/-- The imaginary part of the complex number z = (1-i)/(2i) is equal to -1/2 -/
theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - i) / (2 * i)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1598_159814


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l1598_159837

/-- Given a bus that stops for 24 minutes per hour and has a speed of 45 kmph including stoppages,
    its speed excluding stoppages is 75 kmph. -/
theorem bus_speed_excluding_stoppages 
  (stop_time : ℝ) 
  (speed_with_stops : ℝ) 
  (h1 : stop_time = 24)
  (h2 : speed_with_stops = 45) : 
  speed_with_stops * (60 / (60 - stop_time)) = 75 := by
  sorry


end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l1598_159837


namespace NUMINAMATH_CALUDE_determinant_in_terms_of_r_s_t_l1598_159835

theorem determinant_in_terms_of_r_s_t (r s t : ℝ) (a b c : ℝ) : 
  (a^3 - r*a^2 + s*a - t = 0) →
  (b^3 - r*b^2 + s*b - t = 0) →
  (c^3 - r*c^2 + s*c - t = 0) →
  (a + b + c = r) →
  (a*b + a*c + b*c = s) →
  (a*b*c = t) →
  Matrix.det !![2+a, 2, 2; 2, 2+b, 2; 2, 2, 2+c] = t - 2*s := by
sorry

end NUMINAMATH_CALUDE_determinant_in_terms_of_r_s_t_l1598_159835


namespace NUMINAMATH_CALUDE_min_sum_of_dimensions_l1598_159860

theorem min_sum_of_dimensions (l w h : ℕ) : 
  l > 0 → w > 0 → h > 0 → l * w * h = 2310 → 
  ∀ a b c : ℕ, a > 0 → b > 0 → c > 0 → a * b * c = 2310 → 
  l + w + h ≤ a + b + c → l + w + h = 42 := by sorry

end NUMINAMATH_CALUDE_min_sum_of_dimensions_l1598_159860


namespace NUMINAMATH_CALUDE_point_A_in_transformed_plane_l1598_159894

/-- The similarity transformation coefficient -/
def k : ℝ := -2

/-- The original plane equation -/
def plane_a (x y z : ℝ) : Prop := x - 2*y + z + 1 = 0

/-- The transformed plane equation -/
def plane_a' (x y z : ℝ) : Prop := x - 2*y + z - 2 = 0

/-- Point A -/
def point_A : ℝ × ℝ × ℝ := (2, 1, 2)

/-- Theorem stating that point A belongs to the image of plane a -/
theorem point_A_in_transformed_plane : 
  let (x, y, z) := point_A
  plane_a' x y z := by sorry

end NUMINAMATH_CALUDE_point_A_in_transformed_plane_l1598_159894


namespace NUMINAMATH_CALUDE_segment_length_l1598_159863

/-- Given a line segment CD with points R and S on it, prove that CD has length 22.5 -/
theorem segment_length (C D R S : ℝ) : 
  R > C → -- R is to the right of C
  S > R → -- S is to the right of R
  D > S → -- D is to the right of S
  (R - C) / (D - R) = 1 / 4 → -- R divides CD in ratio 1:4
  (S - C) / (D - S) = 1 / 2 → -- S divides CD in ratio 1:2
  S - R = 3 → -- Length of RS is 3
  D - C = 22.5 := by -- Length of CD is 22.5
sorry


end NUMINAMATH_CALUDE_segment_length_l1598_159863


namespace NUMINAMATH_CALUDE_least_five_digit_prime_congruent_to_7_mod_20_l1598_159825

theorem least_five_digit_prime_congruent_to_7_mod_20 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (n % 20 = 7) ∧              -- congruent to 7 (mod 20)
  Nat.Prime n ∧               -- prime number
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) → (m % 20 = 7) → Nat.Prime m → m ≥ n) ∧
  n = 10127 := by
sorry

end NUMINAMATH_CALUDE_least_five_digit_prime_congruent_to_7_mod_20_l1598_159825


namespace NUMINAMATH_CALUDE_tenth_term_is_24_l1598_159824

/-- The sum of the first n terms of an arithmetic sequence -/
def sequence_sum (n : ℕ) : ℕ := n^2 + 5*n

/-- The nth term of the arithmetic sequence -/
def nth_term (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n-1)

theorem tenth_term_is_24 : nth_term 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_24_l1598_159824


namespace NUMINAMATH_CALUDE_girls_count_l1598_159818

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- Theorem stating that given the conditions, the number of girls in the college is 160 -/
theorem girls_count (c : College) 
  (ratio : c.boys * 5 = c.girls * 8) 
  (total : c.boys + c.girls = 416) : 
  c.girls = 160 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_l1598_159818


namespace NUMINAMATH_CALUDE_integral_f_equals_four_thirds_l1598_159827

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x^2
  else if 1 < x ∧ x < Real.exp 1 then 1/x
  else 0  -- undefined elsewhere

-- State the theorem
theorem integral_f_equals_four_thirds :
  ∫ x in (0)..(Real.exp 1), f x = 4/3 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_four_thirds_l1598_159827


namespace NUMINAMATH_CALUDE_marble_probability_l1598_159874

theorem marble_probability (total_marbles : ℕ) (p_white p_green p_yellow : ℚ) :
  total_marbles = 250 →
  p_white = 2 / 5 →
  p_green = 1 / 4 →
  p_yellow = 1 / 10 →
  1 - (p_white + p_green + p_yellow) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l1598_159874


namespace NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l1598_159865

/-- Simple interest calculation -/
def simple_interest (principal time rate : ℝ) : ℝ :=
  principal * time * rate

/-- Given conditions -/
def principal : ℝ := 2500
def time : ℝ := 4
def interest : ℝ := 1000

/-- Theorem to prove -/
theorem interest_rate_is_ten_percent :
  ∃ (rate : ℝ), simple_interest principal time rate = interest ∧ rate = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_ten_percent_l1598_159865


namespace NUMINAMATH_CALUDE_nut_distribution_l1598_159805

def distribute_nuts (total : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ := sorry

theorem nut_distribution (total : ℕ) :
  let (tamas, erzsi, bela, juliska, remaining) := distribute_nuts total
  (tamas + bela) - (erzsi + juliska) = 100 →
  total = 1021 ∧ remaining = 321 := by sorry

end NUMINAMATH_CALUDE_nut_distribution_l1598_159805


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1598_159847

-- Define variables
variable (x y a b : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : 3 * (4 * x - 2 * y) - 3 * (-y + 8 * x) = -12 * x - 3 * y := by
  sorry

-- Theorem for the second expression
theorem simplify_expression_2 : 3 * a^2 - 2 * (2 * a^2 - (2 * a * b - a^2) + 4 * a * b) = -3 * a^2 - 4 * a * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1598_159847


namespace NUMINAMATH_CALUDE_prime_consecutive_property_l1598_159816

theorem prime_consecutive_property (p : ℕ) (hp : Prime p) (hp2 : Prime (p + 2)) :
  p = 3 ∨ 6 ∣ (p + 1) :=
sorry

end NUMINAMATH_CALUDE_prime_consecutive_property_l1598_159816


namespace NUMINAMATH_CALUDE_product_equals_sum_solutions_l1598_159801

theorem product_equals_sum_solutions (a b c d e f g : ℕ+) :
  a * b * c * d * e * f * g = a + b + c + d + e + f + g →
  ((a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 2 ∧ g = 7) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 3 ∧ g = 4) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 7 ∧ g = 2) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 4 ∧ g = 3) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 2 ∧ f = 1 ∧ g = 7) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 3 ∧ f = 1 ∧ g = 4) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 2 ∧ e = 1 ∧ f = 1 ∧ g = 7) ∨
   (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 3 ∧ e = 1 ∧ f = 1 ∧ g = 4) ∨
   (a = 1 ∧ b = 1 ∧ c = 2 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 7) ∨
   (a = 1 ∧ b = 1 ∧ c = 3 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 4) ∨
   (a = 1 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 7) ∨
   (a = 1 ∧ b = 3 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 4) ∨
   (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 7) ∨
   (a = 3 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 4) ∨
   (a = 4 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 3) ∨
   (a = 7 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1 ∧ g = 2)) := by
  sorry

end NUMINAMATH_CALUDE_product_equals_sum_solutions_l1598_159801


namespace NUMINAMATH_CALUDE_factors_of_48_l1598_159875

def number_of_factors (n : Nat) : Nat :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_48 : number_of_factors 48 = 10 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_48_l1598_159875


namespace NUMINAMATH_CALUDE_points_collinear_implies_a_equals_4_l1598_159844

-- Define the points
def A : ℝ × ℝ := (4, 3)
def B (a : ℝ) : ℝ × ℝ := (5, a)
def C : ℝ × ℝ := (6, 5)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - q.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - q.1)

-- Theorem statement
theorem points_collinear_implies_a_equals_4 (a : ℝ) :
  collinear A (B a) C → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_points_collinear_implies_a_equals_4_l1598_159844


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l1598_159842

/-- Given a canoe with a speed in still water and a downstream speed, calculate its upstream speed -/
theorem canoe_upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 12.5)
  (h2 : speed_downstream = 16) :
  speed_still - (speed_downstream - speed_still) = 9 := by
  sorry

#check canoe_upstream_speed

end NUMINAMATH_CALUDE_canoe_upstream_speed_l1598_159842


namespace NUMINAMATH_CALUDE_starting_lineup_theorem_l1598_159871

/-- The number of ways to choose a starting lineup from a basketball team. -/
def starting_lineup_choices (team_size : ℕ) (lineup_size : ℕ) (point_guard_count : ℕ) : ℕ :=
  team_size * Nat.choose (team_size - 1) (lineup_size - 1)

/-- Theorem: The number of ways to choose a starting lineup of 5 players
    from a team of 12, where one player must be the point guard and
    the other four positions are interchangeable, is equal to 3960. -/
theorem starting_lineup_theorem :
  starting_lineup_choices 12 5 1 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_theorem_l1598_159871


namespace NUMINAMATH_CALUDE_greatest_integer_jo_l1598_159808

theorem greatest_integer_jo (n : ℕ) : n < 150 → 
  (∃ k : ℤ, n = 9 * k - 1) → 
  (∃ l : ℤ, n = 6 * l - 5) → 
  n ≤ 125 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_jo_l1598_159808


namespace NUMINAMATH_CALUDE_glasses_displayed_is_70_l1598_159811

/-- Represents the cupboard system with given capacities and a broken shelf --/
structure CupboardSystem where
  tall_capacity : ℕ
  wide_capacity : ℕ
  narrow_capacity : ℕ
  narrow_shelves : ℕ
  broken_shelves : ℕ

/-- Calculates the total number of glasses displayed in the cupboard system --/
def total_glasses_displayed (cs : CupboardSystem) : ℕ :=
  cs.tall_capacity + cs.wide_capacity + 
  (cs.narrow_capacity / cs.narrow_shelves) * (cs.narrow_shelves - cs.broken_shelves)

/-- Theorem stating that the total number of glasses displayed is 70 --/
theorem glasses_displayed_is_70 : ∃ (cs : CupboardSystem), 
  cs.tall_capacity = 20 ∧
  cs.wide_capacity = 2 * cs.tall_capacity ∧
  cs.narrow_capacity = 15 ∧
  cs.narrow_shelves = 3 ∧
  cs.broken_shelves = 1 ∧
  total_glasses_displayed cs = 70 := by
  sorry

end NUMINAMATH_CALUDE_glasses_displayed_is_70_l1598_159811


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1598_159891

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -3 ∧ 
  k = 0 ∧ 
  c = Real.sqrt 50 ∧ 
  a = 5 ∧ 
  c^2 = a^2 + b^2 →
  h + k + a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1598_159891


namespace NUMINAMATH_CALUDE_compute_expression_l1598_159849

theorem compute_expression : 9 * (2 / 7 : ℚ)^4 = 144 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1598_159849


namespace NUMINAMATH_CALUDE_age_sum_problem_l1598_159821

theorem age_sum_problem (a b c : ℕ+) : 
  b = c →                   -- The twins have the same age
  a < b →                   -- Kiana is younger than her brothers
  a * b * c = 256 →         -- The product of their ages is 256
  a + b + c = 20 :=         -- The sum of their ages is 20
by sorry

end NUMINAMATH_CALUDE_age_sum_problem_l1598_159821


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l1598_159831

/-- Given a sequence of positive real numbers {aₙ} where the sum of the first n terms
    Sₙ satisfies Sₙ = (1/2)(aₙ + 1/aₙ), prove that aₙ = √n - √(n-1) for all positive integers n. -/
theorem sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) 
  (h_pos : ∀ k, k > 0 → a k > 0)
  (h_sum : ∀ k, k > 0 → S k = (1/2) * (a k + 1 / a k)) :
  a n = Real.sqrt n - Real.sqrt (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l1598_159831


namespace NUMINAMATH_CALUDE_third_term_base_l1598_159857

theorem third_term_base (h a b c : ℕ+) (base : ℕ+) : 
  (225 ∣ h) → 
  (216 ∣ h) → 
  h = 2^(a.val) * 3^(b.val) * base^(c.val) →
  a.val + b.val + c.val = 8 →
  base = 5 := by sorry

end NUMINAMATH_CALUDE_third_term_base_l1598_159857


namespace NUMINAMATH_CALUDE_average_salary_increase_proof_l1598_159885

def average_salary_increase 
  (initial_employees : ℕ) 
  (initial_average_salary : ℚ) 
  (manager_salary : ℚ) : ℚ :=
  let total_initial_salary := initial_employees * initial_average_salary
  let new_total_salary := total_initial_salary + manager_salary
  let new_average_salary := new_total_salary / (initial_employees + 1)
  new_average_salary - initial_average_salary

theorem average_salary_increase_proof :
  average_salary_increase 24 1500 11500 = 400 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_increase_proof_l1598_159885


namespace NUMINAMATH_CALUDE_prime_remainder_mod_30_l1598_159892

theorem prime_remainder_mod_30 (p : ℕ) (hp : Prime p) : 
  ∃ (r : ℕ), p % 30 = r ∧ (r = 1 ∨ (Prime r ∧ r < 30)) := by
  sorry

end NUMINAMATH_CALUDE_prime_remainder_mod_30_l1598_159892


namespace NUMINAMATH_CALUDE_stock_price_calculation_l1598_159886

theorem stock_price_calculation (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : 
  initial_price = 100 ∧ 
  first_year_increase = 1.5 ∧ 
  second_year_decrease = 0.4 → 
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease) = 150 := by
sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l1598_159886


namespace NUMINAMATH_CALUDE_piece_in_313th_row_l1598_159855

/-- Represents a chessboard with pieces -/
structure Chessboard :=
  (size : ℕ)
  (pieces : ℕ)
  (symmetrical : Bool)

/-- Checks if a row contains a piece -/
def has_piece_in_row (board : Chessboard) (row : ℕ) : Prop :=
  sorry

theorem piece_in_313th_row (board : Chessboard) 
  (h1 : board.size = 625)
  (h2 : board.pieces = 1977)
  (h3 : board.symmetrical = true) :
  has_piece_in_row board 313 :=
sorry

end NUMINAMATH_CALUDE_piece_in_313th_row_l1598_159855


namespace NUMINAMATH_CALUDE_geometric_proportion_proof_l1598_159895

theorem geometric_proportion_proof :
  let a : ℝ := 21
  let b : ℝ := 7
  let c : ℝ := 9
  let d : ℝ := 3
  (a / b = c / d) ∧
  (a + d = 24) ∧
  (b + c = 16) ∧
  (a^2 + b^2 + c^2 + d^2 = 580) := by
  sorry

end NUMINAMATH_CALUDE_geometric_proportion_proof_l1598_159895


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l1598_159896

theorem two_digit_number_problem (x y : ℕ) :
  x < 10 ∧ y < 10 ∧ 
  (10 * x + y) - (10 * y + x) = 36 ∧
  x + y = 8 →
  10 * x + y = 62 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l1598_159896


namespace NUMINAMATH_CALUDE_xy_sum_l1598_159841

theorem xy_sum (x y : ℕ+) (h : (2 * x - 5) * (2 * y - 5) = 25) :
  x + y = 10 ∨ x + y = 18 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_l1598_159841


namespace NUMINAMATH_CALUDE_unique_c_l1598_159851

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := -x^2 + c*x - 12

-- Define the condition for the inequality
def condition (c : ℝ) : Prop :=
  ∀ x : ℝ, f c x < 0 ↔ (x < 2 ∨ x > 7)

-- Theorem statement
theorem unique_c : ∃! c : ℝ, condition c :=
  sorry

end NUMINAMATH_CALUDE_unique_c_l1598_159851


namespace NUMINAMATH_CALUDE_inverse_sum_equals_target_l1598_159838

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then x + 3 else x^2 - 4*x + 5

noncomputable def g_inverse (y : ℝ) : ℝ :=
  if y ≤ 5 then y - 3 else 2 + Real.sqrt (y - 1)

theorem inverse_sum_equals_target : g_inverse 1 + g_inverse 6 + g_inverse 11 = 2 + Real.sqrt 5 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_target_l1598_159838


namespace NUMINAMATH_CALUDE_faye_score_l1598_159873

/-- Given a baseball team with the following properties:
  * The team has 5 players
  * The team scored a total of 68 points
  * 4 players scored 8 points each
  Prove that the remaining player (Faye) scored 36 points. -/
theorem faye_score (total_score : ℕ) (team_size : ℕ) (other_player_score : ℕ) :
  total_score = 68 →
  team_size = 5 →
  other_player_score = 8 →
  ∃ (faye_score : ℕ), faye_score = total_score - (team_size - 1) * other_player_score ∧
                      faye_score = 36 :=
by sorry

end NUMINAMATH_CALUDE_faye_score_l1598_159873


namespace NUMINAMATH_CALUDE_arrangement_counts_l1598_159829

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

def girls_not_adjacent : ℕ := sorry

def boys_adjacent : ℕ := sorry

def girl_A_not_left_B_not_right : ℕ := sorry

def girls_ABC_height_order : ℕ := sorry

theorem arrangement_counts :
  girls_not_adjacent = 1440 ∧
  boys_adjacent = 576 ∧
  girl_A_not_left_B_not_right = 3720 ∧
  girls_ABC_height_order = 840 := by sorry

end NUMINAMATH_CALUDE_arrangement_counts_l1598_159829


namespace NUMINAMATH_CALUDE_train_length_l1598_159882

/-- Given a train crossing a bridge, calculate its length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 235 →
  (train_speed * crossing_time) - bridge_length = 140 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1598_159882


namespace NUMINAMATH_CALUDE_fraction_value_l1598_159893

theorem fraction_value : 
  (10 + (-9) + 8 + (-7) + 6 + (-5) + 4 + (-3) + 2 + (-1)) / 
  (2 - 4 + 6 - 8 + 10 - 12 + 14 - 16 + 18) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1598_159893


namespace NUMINAMATH_CALUDE_sqrt_expressions_l1598_159820

-- Define the theorem
theorem sqrt_expressions :
  -- Part 1
  (∀ (a b m n : ℤ), a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 → 
    a = m^2 + 3*n^2 ∧ b = 2*m*n) ∧
  -- Part 2
  (∀ (a m n : ℕ+), a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 → 
    a = 13 ∨ a = 7) ∧
  -- Part 3
  Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 := by
sorry


end NUMINAMATH_CALUDE_sqrt_expressions_l1598_159820


namespace NUMINAMATH_CALUDE_average_of_multiples_l1598_159836

theorem average_of_multiples (x : ℝ) : 
  let terms := [0, 2*x, 4*x, 8*x, 16*x]
  let multiplied_terms := List.map (· * 3) terms
  List.sum multiplied_terms / 5 = 18 * x := by
sorry

end NUMINAMATH_CALUDE_average_of_multiples_l1598_159836


namespace NUMINAMATH_CALUDE_sara_movie_expenses_l1598_159832

/-- The total amount Sara spent on movies -/
def total_spent (ticket_price : ℚ) (num_tickets : ℕ) (rental_price : ℚ) (purchase_price : ℚ) : ℚ :=
  ticket_price * num_tickets + rental_price + purchase_price

/-- Theorem stating the total amount Sara spent on movies -/
theorem sara_movie_expenses :
  let ticket_price : ℚ := 10.62
  let num_tickets : ℕ := 2
  let rental_price : ℚ := 1.59
  let purchase_price : ℚ := 13.95
  total_spent ticket_price num_tickets rental_price purchase_price = 36.78 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_expenses_l1598_159832


namespace NUMINAMATH_CALUDE_quadratic_function_zeros_l1598_159815

theorem quadratic_function_zeros (a : ℝ) :
  (∃ x y : ℝ, x > 2 ∧ y < -1 ∧
   -x^2 + a*x + 4 = 0 ∧
   -y^2 + a*y + 4 = 0) →
  0 < a ∧ a < 3 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_zeros_l1598_159815


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1598_159854

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 - 3*(a 3) + 1 = 0 →
  (a 7)^2 - 3*(a 7) + 1 = 0 →
  a 4 + a 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1598_159854


namespace NUMINAMATH_CALUDE_dinner_bill_proof_l1598_159852

theorem dinner_bill_proof (n : ℕ) (extra : ℝ) (total : ℝ) : 
  n = 10 →
  extra = 3 →
  (n - 1) * (total / n + extra) = total →
  total = 270 := by
sorry

end NUMINAMATH_CALUDE_dinner_bill_proof_l1598_159852


namespace NUMINAMATH_CALUDE_rectangular_plot_dimensions_l1598_159867

theorem rectangular_plot_dimensions (area : ℝ) (fence_length : ℝ) :
  area = 800 ∧ fence_length = 100 →
  ∃ (length width : ℝ),
    (length * width = area ∧
     2 * length + width = fence_length) ∧
    ((length = 40 ∧ width = 20) ∨ (length = 10 ∧ width = 80)) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_dimensions_l1598_159867


namespace NUMINAMATH_CALUDE_snow_probability_l1598_159839

theorem snow_probability (p : ℝ) (h : p = 3/4) : 
  1 - (1 - p)^4 = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l1598_159839


namespace NUMINAMATH_CALUDE_number_pairing_l1598_159859

theorem number_pairing (numbers : List ℕ) (h1 : numbers = [41, 35, 19, 9, 26, 45, 13, 28]) :
  let total_sum := numbers.sum
  let pair_sum := total_sum / 4
  ∃ (pairs : List (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 + p.2 = pair_sum) ∧ 
    (∀ n ∈ numbers, ∃ p ∈ pairs, n = p.1 ∨ n = p.2) ∧
    (∃ p ∈ pairs, p = (13, 41) ∨ p = (41, 13)) :=
by sorry

end NUMINAMATH_CALUDE_number_pairing_l1598_159859


namespace NUMINAMATH_CALUDE_no_factors_l1598_159850

def main_polynomial (z : ℂ) : ℂ := z^6 + 3*z^3 + 18

def option1 (z : ℂ) : ℂ := z^3 + 6
def option2 (z : ℂ) : ℂ := z - 2
def option3 (z : ℂ) : ℂ := z^3 - 6
def option4 (z : ℂ) : ℂ := z^3 - 3*z - 9

theorem no_factors :
  (∀ z, main_polynomial z ≠ 0 → option1 z ≠ 0) ∧
  (∀ z, main_polynomial z ≠ 0 → option2 z ≠ 0) ∧
  (∀ z, main_polynomial z ≠ 0 → option3 z ≠ 0) ∧
  (∀ z, main_polynomial z ≠ 0 → option4 z ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_factors_l1598_159850


namespace NUMINAMATH_CALUDE_periodic_sine_condition_l1598_159876

/-- Given a function f(x) = 2sin(ωx - π/3), prove that
    "∀x∈ℝ, f(x+π)=f(x)" is a necessary but not sufficient condition for ω = 2 -/
theorem periodic_sine_condition (ω : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x - π / 3)
  (∀ x, f (x + π) = f x) → ω = 2 ∧
  ∃ ω', ω' ≠ 2 ∧ (∀ x, 2 * Real.sin (ω' * x - π / 3) = 2 * Real.sin (ω' * (x + π) - π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_periodic_sine_condition_l1598_159876


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1598_159879

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1598_159879


namespace NUMINAMATH_CALUDE_base7_246_to_base10_l1598_159807

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (d2 d1 d0 : ℕ) : ℕ :=
  d2 * 7^2 + d1 * 7^1 + d0 * 7^0

/-- The base 10 representation of 246 in base 7 is 132 -/
theorem base7_246_to_base10 : base7ToBase10 2 4 6 = 132 := by
  sorry

end NUMINAMATH_CALUDE_base7_246_to_base10_l1598_159807


namespace NUMINAMATH_CALUDE_no_cube_in_sequence_l1598_159834

theorem no_cube_in_sequence : ∀ (n : ℕ), ¬ ∃ (k : ℤ), 2^(2^n) + 1 = k^3 := by sorry

end NUMINAMATH_CALUDE_no_cube_in_sequence_l1598_159834


namespace NUMINAMATH_CALUDE_group_element_identity_l1598_159803

theorem group_element_identity (G : Type) [Group G] (a b : G) 
  (h1 : a * b^2 = b^3 * a) (h2 : b * a^2 = a^3 * b) : a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_group_element_identity_l1598_159803


namespace NUMINAMATH_CALUDE_investment_income_is_660_l1598_159878

/-- Calculates the total annual income from an investment split between a savings account and bonds. -/
def totalAnnualIncome (totalInvestment : ℝ) (savingsAmount : ℝ) (savingsRate : ℝ) (bondRate : ℝ) : ℝ :=
  let bondAmount := totalInvestment - savingsAmount
  savingsAmount * savingsRate + bondAmount * bondRate

/-- Proves that the total annual income from the given investment scenario is $660. -/
theorem investment_income_is_660 :
  totalAnnualIncome 10000 6000 0.05 0.09 = 660 := by
  sorry

#eval totalAnnualIncome 10000 6000 0.05 0.09

end NUMINAMATH_CALUDE_investment_income_is_660_l1598_159878


namespace NUMINAMATH_CALUDE_survey_mn_value_l1598_159840

/-- Proves that mn = 2.5 given the survey conditions --/
theorem survey_mn_value (total : ℕ) (table_tennis basketball soccer : ℕ) 
  (h1 : total = 100)
  (h2 : table_tennis = 40)
  (h3 : (table_tennis : ℚ) / total = 2/5)
  (h4 : (basketball : ℚ) / total = 1/4)
  (h5 : soccer = total - (table_tennis + basketball))
  (h6 : (soccer : ℚ) / total = (soccer : ℚ) / 100) :
  (basketball : ℚ) * ((soccer : ℚ) / 100) = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_survey_mn_value_l1598_159840


namespace NUMINAMATH_CALUDE_exactly_one_defective_two_genuine_mutually_exclusive_not_contradictory_l1598_159823

/-- Represents the outcome of selecting two products -/
inductive SelectionOutcome
  | TwoGenuine
  | OneGenuineOneDefective
  | TwoDefective

/-- Represents the total number of products -/
def totalProducts : Nat := 5

/-- Represents the number of genuine products -/
def genuineProducts : Nat := 3

/-- Represents the number of defective products -/
def defectiveProducts : Nat := 2

/-- Checks if two events are mutually exclusive -/
def mutuallyExclusive (e1 e2 : Set SelectionOutcome) : Prop :=
  e1 ∩ e2 = ∅

/-- Checks if two events are not contradictory -/
def notContradictory (e1 e2 : Set SelectionOutcome) : Prop :=
  e1 ∪ e2 ≠ Set.univ

/-- The event of selecting exactly one defective product -/
def exactlyOneDefective : Set SelectionOutcome :=
  {SelectionOutcome.OneGenuineOneDefective}

/-- The event of selecting exactly two genuine products -/
def exactlyTwoGenuine : Set SelectionOutcome :=
  {SelectionOutcome.TwoGenuine}

/-- Theorem stating that exactly one defective and exactly two genuine are mutually exclusive but not contradictory -/
theorem exactly_one_defective_two_genuine_mutually_exclusive_not_contradictory :
  mutuallyExclusive exactlyOneDefective exactlyTwoGenuine ∧
  notContradictory exactlyOneDefective exactlyTwoGenuine :=
sorry

end NUMINAMATH_CALUDE_exactly_one_defective_two_genuine_mutually_exclusive_not_contradictory_l1598_159823


namespace NUMINAMATH_CALUDE_square_root_sum_of_squares_l1598_159812

theorem square_root_sum_of_squares (x y : ℝ) : 
  (∃ (s : ℝ), s^2 = x - 2 ∧ (s = 2 ∨ s = -2)) →
  (2*x + y + 7)^(1/3) = 3 →
  ∃ (t : ℝ), t^2 = x^2 + y^2 ∧ (t = 10 ∨ t = -10) := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_of_squares_l1598_159812


namespace NUMINAMATH_CALUDE_delta_problem_l1598_159889

-- Define the delta operation
def delta (a b : ℕ) : ℕ := a^2 - b

-- State the theorem
theorem delta_problem : delta (5^(delta 7 2)) (4^(delta 3 8)) = 5^94 - 4 := by
  sorry

end NUMINAMATH_CALUDE_delta_problem_l1598_159889


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l1598_159897

theorem sqrt_difference_inequality (m : ℝ) (h : m > 1) :
  Real.sqrt (m + 1) - Real.sqrt m < Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l1598_159897


namespace NUMINAMATH_CALUDE_log_sequence_a_is_geometric_l1598_159880

def sequence_a : ℕ → ℝ
  | 0 => 2
  | n + 1 => (sequence_a n) ^ 2

theorem log_sequence_a_is_geometric :
  ∃ r : ℝ, ∀ n : ℕ, n > 0 → Real.log (sequence_a (n + 1)) = r * Real.log (sequence_a n) := by
  sorry

end NUMINAMATH_CALUDE_log_sequence_a_is_geometric_l1598_159880
