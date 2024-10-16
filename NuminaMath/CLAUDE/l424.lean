import Mathlib

namespace NUMINAMATH_CALUDE_monomial_sum_l424_42432

/-- Given that mxy³ + x^(n+2)y³ = 5xy³, prove that m - n = 5 -/
theorem monomial_sum (x y m n : ℝ) (h : ∀ x y, m * x * y^3 + x^(n+2) * y^3 = 5 * x * y^3) : 
  m - n = 5 := by
sorry

end NUMINAMATH_CALUDE_monomial_sum_l424_42432


namespace NUMINAMATH_CALUDE_coin_tosses_properties_l424_42408

/-- Two independent coin tosses where A is "first coin is heads" and B is "second coin is tails" -/
structure CoinTosses where
  /-- Probability of event A (first coin is heads) -/
  prob_A : ℝ
  /-- Probability of event B (second coin is tails) -/
  prob_B : ℝ
  /-- A and B are independent events -/
  independent : Prop
  /-- Both coins are fair -/
  fair_coins : prob_A = 1/2 ∧ prob_B = 1/2

/-- Properties of the coin tosses -/
theorem coin_tosses_properties (ct : CoinTosses) :
  ct.independent ∧ 
  (1 - (1 - ct.prob_A) * (1 - ct.prob_B) = 3/4) ∧
  ct.prob_A = ct.prob_B :=
sorry

end NUMINAMATH_CALUDE_coin_tosses_properties_l424_42408


namespace NUMINAMATH_CALUDE_sequence_has_unique_occurrence_l424_42441

def is_unique_occurrence (s : ℕ → ℝ) (x : ℝ) : Prop :=
  ∃! n : ℕ, s n = x

theorem sequence_has_unique_occurrence
  (a : ℕ → ℝ)
  (h_inc : ∀ i j : ℕ, i < j → a i < a j)
  (h_bound : ∀ i : ℕ, 0 < a i ∧ a i < 1) :
  ∃ x : ℝ, is_unique_occurrence (λ i => a i / i) x :=
sorry

end NUMINAMATH_CALUDE_sequence_has_unique_occurrence_l424_42441


namespace NUMINAMATH_CALUDE_choir_group_division_l424_42436

theorem choir_group_division (sopranos altos tenors basses : ℕ) 
  (h_sopranos : sopranos = 10)
  (h_altos : altos = 15)
  (h_tenors : tenors = 12)
  (h_basses : basses = 18) :
  ∃ (n : ℕ), n = 3 ∧ 
  n > 0 ∧
  sopranos % n = 0 ∧ 
  altos % n = 0 ∧ 
  tenors % n = 0 ∧ 
  basses % n = 0 ∧
  sopranos / n < (altos + tenors + basses) / n ∧
  ∀ m : ℕ, m > n → 
    (sopranos % m ≠ 0 ∨ 
     altos % m ≠ 0 ∨ 
     tenors % m ≠ 0 ∨ 
     basses % m ≠ 0 ∨
     sopranos / m ≥ (altos + tenors + basses) / m) :=
by sorry

end NUMINAMATH_CALUDE_choir_group_division_l424_42436


namespace NUMINAMATH_CALUDE_polynomial_sum_l424_42497

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₄ - a₃ + a₂ - a₁ + a₀ = 16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l424_42497


namespace NUMINAMATH_CALUDE_largest_certain_divisor_l424_42412

/-- An eight-sided die with numbers 1 through 8 -/
def Die : Finset ℕ := Finset.range 8 

/-- The product of 7 visible numbers on the die -/
def Q (visible : Finset ℕ) : ℕ := 
  Finset.prod visible id

/-- The theorem stating that 192 is the largest number that always divides Q -/
theorem largest_certain_divisor : 
  ∀ visible : Finset ℕ, visible ⊆ Die → visible.card = 7 → 
    (∀ n : ℕ, n > 192 → ∃ visible : Finset ℕ, visible ⊆ Die ∧ visible.card = 7 ∧ ¬(n ∣ Q visible)) ∧
    (∀ visible : Finset ℕ, visible ⊆ Die → visible.card = 7 → 192 ∣ Q visible) :=
by sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_l424_42412


namespace NUMINAMATH_CALUDE_function_equality_condition_l424_42437

theorem function_equality_condition (m n p q : ℝ) : 
  let f := λ x : ℝ => m * x^2 + n
  let g := λ x : ℝ => p * x + q
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p^2) = q * (1 - m) := by sorry

end NUMINAMATH_CALUDE_function_equality_condition_l424_42437


namespace NUMINAMATH_CALUDE_systematic_sampling_l424_42494

/-- Systematic sampling problem -/
theorem systematic_sampling
  (population_size : ℕ)
  (sample_size : ℕ)
  (last_sampled : ℕ)
  (h1 : population_size = 8000)
  (h2 : sample_size = 50)
  (h3 : last_sampled = 7894)
  (h4 : last_sampled < population_size) :
  let segment_size := population_size / sample_size
  let first_sampled := last_sampled - (segment_size - 1)
  first_sampled = 735 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l424_42494


namespace NUMINAMATH_CALUDE_total_players_l424_42416

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) : 
  kabadi = 10 → kho_kho_only = 25 → both = 5 → 
  kabadi + kho_kho_only - both = 30 := by
sorry

end NUMINAMATH_CALUDE_total_players_l424_42416


namespace NUMINAMATH_CALUDE_bottom_is_red_l424_42431

/-- Represents the colors of the squares -/
inductive Color
  | R | B | O | Y | G | W | P

/-- Represents a face of the cube -/
structure Face where
  color : Color

/-- Represents the cube configuration -/
structure Cube where
  top : Face
  bottom : Face
  sides : List Face
  outward : Face

/-- Theorem: Given the cube configuration, the bottom face is Red -/
theorem bottom_is_red (cube : Cube)
  (h1 : cube.top.color = Color.W)
  (h2 : cube.outward.color = Color.P)
  (h3 : cube.sides.length = 4)
  (h4 : ∀ c : Color, c ≠ Color.P → c ∈ (cube.top :: cube.bottom :: cube.sides).map Face.color) :
  cube.bottom.color = Color.R :=
sorry

end NUMINAMATH_CALUDE_bottom_is_red_l424_42431


namespace NUMINAMATH_CALUDE_system_solution_l424_42449

def solution_set : Set (ℝ × ℝ) :=
  {(-1/Real.sqrt 10, 3/Real.sqrt 10), (-1/Real.sqrt 10, -3/Real.sqrt 10),
   (1/Real.sqrt 10, 3/Real.sqrt 10), (1/Real.sqrt 10, -3/Real.sqrt 10)}

def satisfies_system (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 ≤ 1 ∧
  x^4 - 18*x^2*y^2 + 81*y^4 - 20*x^2 - 180*y^2 + 100 = 0

theorem system_solution :
  {p : ℝ × ℝ | satisfies_system p} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l424_42449


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l424_42450

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.mk (a^2 - 2*a - 3) (a + 1)).im ≠ 0 ∧ 
  (Complex.mk (a^2 - 2*a - 3) (a + 1)).re = 0 → 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l424_42450


namespace NUMINAMATH_CALUDE_trig_expression_equals_sqrt_two_l424_42442

/-- Proves that the given trigonometric expression equals √2 --/
theorem trig_expression_equals_sqrt_two :
  (Real.cos (10 * π / 180) - Real.sqrt 3 * Real.cos (-100 * π / 180)) /
  Real.sqrt (1 - Real.sin (10 * π / 180)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_sqrt_two_l424_42442


namespace NUMINAMATH_CALUDE_unique_positive_root_implies_a_less_than_neg_one_l424_42470

/-- Given two functions f and g, if their difference has a unique positive root,
    then the parameter a in f must be less than -1 -/
theorem unique_positive_root_implies_a_less_than_neg_one
  (f g : ℝ → ℝ)
  (h : ∀ x : ℝ, f x = 2 * a * x^3 + 3)
  (k : ∀ x : ℝ, g x = 3 * x^2 + 2)
  (unique_root : ∃! x₀ : ℝ, x₀ > 0 ∧ f x₀ = g x₀) :
  a < -1 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_root_implies_a_less_than_neg_one_l424_42470


namespace NUMINAMATH_CALUDE_square_brush_ratio_l424_42476

theorem square_brush_ratio (s w : ℝ) (h : s > 0) (h' : w > 0) : 
  w^2 + ((s - w)^2) / 2 = s^2 / 3 → s / w = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_brush_ratio_l424_42476


namespace NUMINAMATH_CALUDE_distance_to_place_l424_42439

/-- Proves that the distance to a place is 144 km given the rowing speed, current speed, and total round trip time. -/
theorem distance_to_place (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) :
  rowing_speed = 10 →
  current_speed = 2 →
  total_time = 30 →
  (total_time * (rowing_speed + current_speed) * (rowing_speed - current_speed)) / (2 * rowing_speed) = 144 := by
sorry

end NUMINAMATH_CALUDE_distance_to_place_l424_42439


namespace NUMINAMATH_CALUDE_sequence_property_l424_42471

theorem sequence_property (a : ℕ → ℕ) 
    (h_nondecreasing : ∀ n m : ℕ, n ≤ m → a n ≤ a m)
    (h_nonconstant : ∃ n m : ℕ, a n ≠ a m)
    (h_divides : ∀ n : ℕ, a n ∣ n^2) :
  (∃ n₁ : ℕ, ∀ n ≥ n₁, a n = n) ∨ 
  (∃ n₂ : ℕ, ∀ n ≥ n₂, a n = n^2) := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l424_42471


namespace NUMINAMATH_CALUDE_log_comparison_l424_42406

theorem log_comparison : 
  (Real.log 4 / Real.log 3) > (0.3 ^ 4) ∧ (0.3 ^ 4) > (Real.log 0.9 / Real.log 1.1) := by
sorry

end NUMINAMATH_CALUDE_log_comparison_l424_42406


namespace NUMINAMATH_CALUDE_largest_root_bound_l424_42445

theorem largest_root_bound (b₂ b₁ b₀ : ℤ) (h₂ : |b₂| ≤ 3) (h₁ : |b₁| ≤ 3) (h₀ : |b₀| ≤ 3) :
  ∃ r : ℝ, 3.5 < r ∧ r < 4 ∧
  (∀ x : ℝ, x > r → x^3 + (b₂ : ℝ) * x^2 + (b₁ : ℝ) * x + (b₀ : ℝ) ≠ 0) ∧
  (∃ x : ℝ, x ≤ r ∧ x^3 + (b₂ : ℝ) * x^2 + (b₁ : ℝ) * x + (b₀ : ℝ) = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_root_bound_l424_42445


namespace NUMINAMATH_CALUDE_number_value_l424_42418

theorem number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 75 := by
  sorry

end NUMINAMATH_CALUDE_number_value_l424_42418


namespace NUMINAMATH_CALUDE_balloon_count_theorem_l424_42414

/-- Represents the number of balloons each person has --/
structure BalloonCount where
  allan : ℕ
  jake : ℕ
  sarah : ℕ

/-- Initial balloon count --/
def initial : BalloonCount :=
  { allan := 5, jake := 4, sarah := 0 }

/-- Sarah buys balloons at the park --/
def sarah_buys (bc : BalloonCount) (n : ℕ) : BalloonCount :=
  { bc with sarah := bc.sarah + n }

/-- Allan buys balloons at the park --/
def allan_buys (bc : BalloonCount) (n : ℕ) : BalloonCount :=
  { bc with allan := bc.allan + n }

/-- Allan gives balloons to Jake --/
def allan_gives_to_jake (bc : BalloonCount) (n : ℕ) : BalloonCount :=
  { bc with allan := bc.allan - n, jake := bc.jake + n }

/-- The final balloon count after all actions --/
def final : BalloonCount :=
  allan_gives_to_jake (allan_buys (sarah_buys initial 7) 3) 2

theorem balloon_count_theorem :
  final = { allan := 6, jake := 6, sarah := 7 } := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_theorem_l424_42414


namespace NUMINAMATH_CALUDE_students_per_bus_l424_42420

/-- Given a field trip scenario with buses and students, calculate the number of students per bus. -/
theorem students_per_bus (total_seats : ℕ) (num_buses : ℚ) : 
  total_seats = 28 → num_buses = 2 → (total_seats : ℚ) / num_buses = 14 := by
  sorry

end NUMINAMATH_CALUDE_students_per_bus_l424_42420


namespace NUMINAMATH_CALUDE_greatest_b_for_quadratic_range_l424_42479

theorem greatest_b_for_quadratic_range : ∃ (b : ℤ), 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -4) ∧ 
  (∀ c : ℤ, c > b → ∃ x : ℝ, x^2 + c*x + 20 = -4) ∧
  b = 9 := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_for_quadratic_range_l424_42479


namespace NUMINAMATH_CALUDE_tommy_nickels_l424_42467

/-- Represents Tommy's coin collection --/
structure CoinCollection where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Conditions for Tommy's coin collection --/
def tommy_collection (c : CoinCollection) : Prop :=
  c.quarters = 4 ∧
  c.dimes = c.pennies + 10 ∧
  c.nickels = 2 * c.dimes ∧
  c.pennies = 10 * c.quarters

theorem tommy_nickels (c : CoinCollection) : 
  tommy_collection c → c.nickels = 100 := by
  sorry

end NUMINAMATH_CALUDE_tommy_nickels_l424_42467


namespace NUMINAMATH_CALUDE_marble_ratio_l424_42425

theorem marble_ratio (blue red : ℕ) (h1 : blue = red + 24) (h2 : red = 6) :
  blue / red = 5 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l424_42425


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l424_42475

def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem simple_interest_calculation :
  let principal : ℚ := 80325
  let rate : ℚ := 1
  let time : ℚ := 5
  simple_interest principal rate time = 4016.25 := by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l424_42475


namespace NUMINAMATH_CALUDE_mean_calculation_l424_42477

theorem mean_calculation (x : ℝ) : 
  (28 + x + 50 + 78 + 104) / 5 = 62 → 
  (48 + 62 + 98 + 124 + x) / 5 = 76.4 := by
sorry

end NUMINAMATH_CALUDE_mean_calculation_l424_42477


namespace NUMINAMATH_CALUDE_circle_to_ellipse_l424_42464

/-- If z is a complex number tracing a circle centered at the origin with radius 3,
    then z + 1/z traces an ellipse. -/
theorem circle_to_ellipse (z : ℂ) (h : ∀ θ : ℝ, z = 3 * Complex.exp (Complex.I * θ)) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  ∀ θ : ℝ, ∃ x y : ℝ, 
    z + 1/z = Complex.mk x y ∧ 
    (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_to_ellipse_l424_42464


namespace NUMINAMATH_CALUDE_midpoint_of_specific_segment_l424_42458

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The midpoint of two polar points -/
def polarMidpoint (p1 p2 : PolarPoint) : PolarPoint :=
  sorry

theorem midpoint_of_specific_segment :
  let p1 : PolarPoint := ⟨6, π/6⟩
  let p2 : PolarPoint := ⟨2, -π/6⟩
  let m := polarMidpoint p1 p2
  0 ≤ m.θ ∧ m.θ < 2*π ∧ m.r > 0 ∧ m = ⟨Real.sqrt 13, π/6⟩ := by
  sorry

end NUMINAMATH_CALUDE_midpoint_of_specific_segment_l424_42458


namespace NUMINAMATH_CALUDE_problem_solution_l424_42413

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + 2 * |x - 1|

-- Define the function g
def g (x m : ℝ) : ℝ := |x + 1 + m| + 2 * |x|

theorem problem_solution :
  (∀ x : ℝ, m > 0 → 
    (m = 1 → (f x m ≤ 10 ↔ -3 ≤ x ∧ x ≤ 11/3))) ∧
  (∀ m : ℝ, (∀ x : ℝ, g x m ≥ 3) ↔ m ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l424_42413


namespace NUMINAMATH_CALUDE_orthogonal_to_pencil_l424_42481

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A pencil of circles -/
structure PencilOfCircles where
  circles : Set Circle

/-- Two circles are orthogonal if the square of the distance between their centers
    is equal to the sum of the squares of their radii -/
def orthogonal (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = c1.radius^2 + c2.radius^2

theorem orthogonal_to_pencil
  (S : Circle) (P : PencilOfCircles) (S1 S2 : Circle)
  (h1 : S1 ∈ P.circles) (h2 : S2 ∈ P.circles) (h3 : S1 ≠ S2)
  (orth1 : orthogonal S S1) (orth2 : orthogonal S S2) :
  ∀ C ∈ P.circles, orthogonal S C :=
sorry

end NUMINAMATH_CALUDE_orthogonal_to_pencil_l424_42481


namespace NUMINAMATH_CALUDE_book_pages_theorem_l424_42461

/-- Represents the number of pages read each night --/
structure ReadingPattern :=
  (night1 : ℕ)
  (night2 : ℕ)
  (night3 : ℕ)
  (night4 : ℕ)

/-- Calculates the total number of pages in the book --/
def totalPages (rp : ReadingPattern) : ℕ :=
  rp.night1 + rp.night2 + rp.night3 + rp.night4

/-- Theorem: The book has 100 pages in total --/
theorem book_pages_theorem (rp : ReadingPattern) 
  (h1 : rp.night1 = 15)
  (h2 : rp.night2 = 2 * rp.night1)
  (h3 : rp.night3 = rp.night2 + 5)
  (h4 : rp.night4 = 20) : 
  totalPages rp = 100 := by
  sorry


end NUMINAMATH_CALUDE_book_pages_theorem_l424_42461


namespace NUMINAMATH_CALUDE_base4_division_theorem_l424_42451

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent. -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

/-- Represents a number in base 4. -/
structure Base4 where
  digits : List Nat
  valid : ∀ d ∈ digits.toFinset, d < 4

/-- The dividend in base 4. -/
def dividend : Base4 := {
  digits := [0, 2, 3, 2, 1]
  valid := by sorry
}

/-- The divisor in base 4. -/
def divisor : Base4 := {
  digits := [2, 1]
  valid := by sorry
}

/-- The quotient in base 4. -/
def quotient : Base4 := {
  digits := [1, 2, 1, 1]
  valid := by sorry
}

/-- Theorem stating that the division of the dividend by the divisor equals the quotient in base 4. -/
theorem base4_division_theorem :
  (base4ToDecimal dividend.digits) / (base4ToDecimal divisor.digits) = base4ToDecimal quotient.digits := by
  sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l424_42451


namespace NUMINAMATH_CALUDE_total_distinct_plants_l424_42447

def X : ℕ := 600
def Y : ℕ := 500
def Z : ℕ := 400
def XY : ℕ := 70
def XZ : ℕ := 80
def YZ : ℕ := 60
def XYZ : ℕ := 30

theorem total_distinct_plants : X + Y + Z - XY - XZ - YZ + XYZ = 1320 := by
  sorry

end NUMINAMATH_CALUDE_total_distinct_plants_l424_42447


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l424_42433

theorem sphere_surface_area_with_inscribed_cube (cube_surface_area : ℝ) 
  (h : cube_surface_area = 54) : 
  ∃ (sphere_surface_area : ℝ), sphere_surface_area = 27 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l424_42433


namespace NUMINAMATH_CALUDE_current_speed_l424_42493

/-- The speed of the current given a woman's swimming times and distances -/
theorem current_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 125) (h2 : upstream_distance = 60) 
  (h3 : time = 10) : ∃ (v_w v_c : ℝ), 
  downstream_distance = (v_w + v_c) * time ∧ 
  upstream_distance = (v_w - v_c) * time ∧ 
  v_c = 3.25 :=
by sorry

end NUMINAMATH_CALUDE_current_speed_l424_42493


namespace NUMINAMATH_CALUDE_triangle_side_length_l424_42460

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = π/3 →
  Real.cos B = (2 * Real.sqrt 7) / 7 →
  b = 3 →
  a = (3 * Real.sqrt 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l424_42460


namespace NUMINAMATH_CALUDE_textile_firm_profit_decrease_l424_42444

/-- Represents the decrease in profit due to loom breakdowns -/
def decrease_in_profit (
  total_looms : ℕ)
  (monthly_sales : ℝ)
  (monthly_manufacturing_expenses : ℝ)
  (monthly_establishment_charges : ℝ)
  (breakdown_days : List ℕ)
  (repair_cost_per_loom : ℝ)
  : ℝ :=
  sorry

/-- Theorem stating the decrease in profit for the given scenario -/
theorem textile_firm_profit_decrease :
  decrease_in_profit 70 1000000 150000 75000 [10, 5, 15] 2000 = 20285.70 :=
sorry

end NUMINAMATH_CALUDE_textile_firm_profit_decrease_l424_42444


namespace NUMINAMATH_CALUDE_interior_angles_sum_plus_three_l424_42409

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: Given a convex polygon with n sides whose interior angles sum to 2340 degrees,
    the sum of interior angles of a convex polygon with n + 3 sides is 2880 degrees. -/
theorem interior_angles_sum_plus_three (n : ℕ) 
  (h : sum_interior_angles n = 2340) : 
  sum_interior_angles (n + 3) = 2880 := by
  sorry


end NUMINAMATH_CALUDE_interior_angles_sum_plus_three_l424_42409


namespace NUMINAMATH_CALUDE_bill_score_l424_42498

theorem bill_score (john sue bill : ℕ) 
  (score_diff : bill = john + 20)
  (bill_half_sue : bill * 2 = sue)
  (total_score : john + bill + sue = 160) :
  bill = 45 := by
sorry

end NUMINAMATH_CALUDE_bill_score_l424_42498


namespace NUMINAMATH_CALUDE_circle_tangent_properties_l424_42434

-- Define the circle C
def circle_C (a r : ℝ) := {(x, y) : ℝ × ℝ | (x - 2)^2 + (y - a)^2 = r^2}

-- Define the tangent line
def tangent_line := {(x, y) : ℝ × ℝ | x + 2*y - 7 = 0}

-- Define the condition that the line is tangent to the circle at (3, 2)
def is_tangent (a r : ℝ) : Prop :=
  (3, 2) ∈ circle_C a r ∧ (3, 2) ∈ tangent_line ∧
  ∀ (x y : ℝ), (x, y) ∈ circle_C a r ∩ tangent_line → (x, y) = (3, 2)

-- Theorem statement
theorem circle_tangent_properties (a r : ℝ) (h : is_tangent a r) :
  a = 0 ∧ (-1, -1) ∉ circle_C a r :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_properties_l424_42434


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_l424_42411

theorem sqrt_2_irrational : Irrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_l424_42411


namespace NUMINAMATH_CALUDE_fraction_equality_l424_42430

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 1 / 3) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l424_42430


namespace NUMINAMATH_CALUDE_college_students_count_l424_42448

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 175) :
  boys + girls = 455 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l424_42448


namespace NUMINAMATH_CALUDE_octal_245_equals_decimal_165_l424_42472

/-- Converts an octal number to decimal --/
def octal_to_decimal (a b c : ℕ) : ℕ := c * 8^2 + b * 8^1 + a * 8^0

/-- Proves that 245 in octal is equal to 165 in decimal --/
theorem octal_245_equals_decimal_165 : octal_to_decimal 5 4 2 = 165 := by
  sorry

end NUMINAMATH_CALUDE_octal_245_equals_decimal_165_l424_42472


namespace NUMINAMATH_CALUDE_product_of_cosines_l424_42499

theorem product_of_cosines (π : Real) : 
  (1 + Real.cos (π / 9)) * (1 + Real.cos (2 * π / 9)) * 
  (1 + Real.cos (4 * π / 9)) * (1 + Real.cos (5 * π / 9)) = 
  (1 / 2) * (Real.sin (π / 9))^4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l424_42499


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l424_42424

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l424_42424


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l424_42453

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l424_42453


namespace NUMINAMATH_CALUDE_least_number_of_beads_beads_divisibility_least_beads_l424_42405

theorem least_number_of_beads (n : ℕ) : n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n → n ≥ 840 := by
  sorry

theorem beads_divisibility : 2 ∣ 840 ∧ 3 ∣ 840 ∧ 5 ∣ 840 ∧ 7 ∣ 840 ∧ 8 ∣ 840 := by
  sorry

theorem least_beads : ∃ (n : ℕ), n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n ∧ n = 840 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_beads_beads_divisibility_least_beads_l424_42405


namespace NUMINAMATH_CALUDE_no_three_consecutive_squares_l424_42480

/-- An arithmetic progression of natural numbers -/
structure ArithmeticProgression where
  terms : ℕ → ℕ
  common_difference : ℕ
  increasing : ∀ n, terms n < terms (n + 1)
  difference_property : ∀ n, terms (n + 1) - terms n = common_difference
  difference_ends_2019 : common_difference % 10000 = 2019

/-- Three consecutive squares in an arithmetic progression -/
def ThreeConsecutiveSquares (ap : ArithmeticProgression) (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    ap.terms n = a^2 ∧ 
    ap.terms (n + 1) = b^2 ∧ 
    ap.terms (n + 2) = c^2

theorem no_three_consecutive_squares (ap : ArithmeticProgression) :
  ¬ ∃ n, ThreeConsecutiveSquares ap n :=
sorry

end NUMINAMATH_CALUDE_no_three_consecutive_squares_l424_42480


namespace NUMINAMATH_CALUDE_gift_original_price_gift_price_calculation_l424_42419

/-- The original price of a gift, given certain conditions --/
theorem gift_original_price (half_cost : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let full_cost := 2 * half_cost
  let discounted_price := (1 - discount_rate) * full_cost / ((1 - discount_rate) * (1 + tax_rate))
  discounted_price

/-- The original price of the gift is approximately $30.50 --/
theorem gift_price_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |gift_original_price 14 0.15 0.08 - 30.50| < ε :=
sorry

end NUMINAMATH_CALUDE_gift_original_price_gift_price_calculation_l424_42419


namespace NUMINAMATH_CALUDE_point_coordinates_l424_42478

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: If a point P is in the second quadrant, its distance to the x-axis is 3,
    and its distance to the y-axis is 10, then its coordinates are (-10, 3) -/
theorem point_coordinates (P : Point)
  (h1 : SecondQuadrant P)
  (h2 : DistanceToXAxis P = 3)
  (h3 : DistanceToYAxis P = 10) :
  P.x = -10 ∧ P.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l424_42478


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l424_42427

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (sum_prod : x + y = 6 * x * y) (double : y = 2 * x) :
  1 / x + 1 / y = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l424_42427


namespace NUMINAMATH_CALUDE_geometric_sum_problem_l424_42440

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem : 
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_problem_l424_42440


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l424_42423

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l424_42423


namespace NUMINAMATH_CALUDE_no_integer_square_root_l424_42486

theorem no_integer_square_root : 
  ¬ ∃ (x : ℤ), ∃ (y : ℤ), x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_root_l424_42486


namespace NUMINAMATH_CALUDE_complement_of_union_l424_42483

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x ∈ U | x^2 - 4*x + 3 = 0}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l424_42483


namespace NUMINAMATH_CALUDE_banana_preference_percentage_l424_42482

/-- Represents the preference count for each fruit in the survey. -/
structure FruitPreferences where
  apple : ℕ
  banana : ℕ
  cherry : ℕ
  dragonfruit : ℕ

/-- Calculates the percentage of people who preferred a specific fruit. -/
def fruitPercentage (prefs : FruitPreferences) (fruitCount : ℕ) : ℚ :=
  (fruitCount : ℚ) / (prefs.apple + prefs.banana + prefs.cherry + prefs.dragonfruit : ℚ) * 100

/-- Theorem stating that the percentage of people who preferred Banana is 37.5%. -/
theorem banana_preference_percentage
  (prefs : FruitPreferences)
  (h1 : prefs.apple = 45)
  (h2 : prefs.banana = 75)
  (h3 : prefs.cherry = 30)
  (h4 : prefs.dragonfruit = 50) :
  fruitPercentage prefs prefs.banana = 37.5 := by
  sorry

#eval fruitPercentage ⟨45, 75, 30, 50⟩ 75

end NUMINAMATH_CALUDE_banana_preference_percentage_l424_42482


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_252_l424_42454

theorem distinct_prime_factors_of_252 : Nat.card (Nat.factors 252).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_252_l424_42454


namespace NUMINAMATH_CALUDE_f_is_odd_iff_l424_42456

-- Define the function f
def f (a b x : ℝ) : ℝ := x * abs (x + a) + b

-- State the theorem
theorem f_is_odd_iff (a b : ℝ) :
  (∀ x, f a b (-x) = -f a b x) ↔ a^2 + b^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_is_odd_iff_l424_42456


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l424_42469

theorem sqrt_equation_solutions :
  ∀ x : ℚ, (Real.sqrt (9 * x - 4) + 16 / Real.sqrt (9 * x - 4) = 9) ↔ (x = 68/9 ∨ x = 5/9) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l424_42469


namespace NUMINAMATH_CALUDE_domino_rearrangement_l424_42462

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_covered : Bool)
  (empty_corner : Nat × Nat)

/-- Represents a domino -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Checks if a given position is a corner of the chessboard -/
def is_corner (board : Chessboard) (pos : Nat × Nat) : Prop :=
  (pos.1 = 1 ∨ pos.1 = board.size) ∧ (pos.2 = 1 ∨ pos.2 = board.size)

/-- Main theorem statement -/
theorem domino_rearrangement 
  (board : Chessboard) 
  (domino : Domino) 
  (h1 : board.size = 9)
  (h2 : domino.length = 1 ∧ domino.width = 2)
  (h3 : board.is_covered = true)
  (h4 : is_corner board board.empty_corner) :
  ∀ (corner : Nat × Nat), is_corner board corner → 
  ∃ (new_board : Chessboard), 
    new_board.size = board.size ∧ 
    new_board.is_covered = true ∧ 
    new_board.empty_corner = corner :=
sorry

end NUMINAMATH_CALUDE_domino_rearrangement_l424_42462


namespace NUMINAMATH_CALUDE_f_properties_l424_42488

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x - 2 * (Real.cos x)^2 + 1

theorem f_properties :
  (∃ (x : ℝ), f x = Real.sqrt 2 ∧ ∀ (y : ℝ), f y ≤ Real.sqrt 2) ∧
  (∀ (θ : ℝ), f θ = 3/5 → Real.cos (2 * (π/4 - 2*θ)) = 16/25) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l424_42488


namespace NUMINAMATH_CALUDE_factor_problem_l424_42457

theorem factor_problem (x : ℝ) (f : ℝ) : 
  x = 6 → (2 * x + 9) * f = 63 → f = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_problem_l424_42457


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l424_42438

/-- The repeating decimal 0.37268̄ expressed as a fraction -/
def repeating_decimal : ℚ := 371896 / 99900

/-- The decimal representation of 0.37268̄ -/
def decimal_representation : ℚ := 37 / 100 + 268 / 99900

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = decimal_representation := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l424_42438


namespace NUMINAMATH_CALUDE_floor_greater_than_x_minus_one_l424_42463

theorem floor_greater_than_x_minus_one (x : ℝ) : ⌊x⌋ > x - 1 := by sorry

end NUMINAMATH_CALUDE_floor_greater_than_x_minus_one_l424_42463


namespace NUMINAMATH_CALUDE_fermat_number_units_digit_F5_l424_42428

theorem fermat_number_units_digit_F5 :
  (2^(2^5) + 1) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_units_digit_F5_l424_42428


namespace NUMINAMATH_CALUDE_percentage_to_fraction_l424_42452

theorem percentage_to_fraction (p : ℚ) : p = 166 / 1000 → p = 83 / 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_fraction_l424_42452


namespace NUMINAMATH_CALUDE_expression_equality_l424_42402

theorem expression_equality : (-1)^2023 - Real.sqrt 9 + |1 - Real.sqrt 2| - ((-8) ^ (1/3 : ℝ)) = Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l424_42402


namespace NUMINAMATH_CALUDE_complex_modulus_l424_42459

theorem complex_modulus (z : ℂ) (h : z = Complex.I / (1 + Complex.I)) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l424_42459


namespace NUMINAMATH_CALUDE_lana_total_pages_l424_42490

/-- Calculate the total number of pages Lana will have after receiving pages from Duane and Alexa -/
theorem lana_total_pages
  (lana_initial : ℕ)
  (duane_pages : ℕ)
  (duane_percentage : ℚ)
  (alexa_pages : ℕ)
  (alexa_percentage : ℚ)
  (h1 : lana_initial = 8)
  (h2 : duane_pages = 42)
  (h3 : duane_percentage = 70 / 100)
  (h4 : alexa_pages = 48)
  (h5 : alexa_percentage = 25 / 100)
  : ℕ := by
  sorry

#check lana_total_pages

end NUMINAMATH_CALUDE_lana_total_pages_l424_42490


namespace NUMINAMATH_CALUDE_bicycle_trip_time_l424_42415

theorem bicycle_trip_time (distance : Real) (outbound_speed return_speed : Real) :
  distance = 28.8 ∧ outbound_speed = 16 ∧ return_speed = 24 →
  distance / outbound_speed + distance / return_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_trip_time_l424_42415


namespace NUMINAMATH_CALUDE_triangle_equilateral_iff_sum_squares_eq_sum_products_l424_42485

/-- A triangle with sides a, b, and c is equilateral if and only if a² + b² + c² = ab + bc + ca -/
theorem triangle_equilateral_iff_sum_squares_eq_sum_products {a b c : ℝ} (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (a = b ∧ b = c) ↔ a^2 + b^2 + c^2 = a*b + b*c + c*a := by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_iff_sum_squares_eq_sum_products_l424_42485


namespace NUMINAMATH_CALUDE_inequality_solution_l424_42489

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | 1 < x }
  else if 0 < a ∧ a < 2 then { x | 1 < x ∧ x < 2/a }
  else if a = 2 then ∅
  else if a > 2 then { x | 2/a < x ∧ x < 1 }
  else { x | x < 2/a ∨ 1 < x }

theorem inequality_solution (a : ℝ) :
  { x : ℝ | a * x^2 - (a + 2) * x + 2 < 0 } = solution_set a := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l424_42489


namespace NUMINAMATH_CALUDE_monotone_cubic_function_condition_l424_42421

/-- Given a function f(x) = -x^3 + bx that is monotonically increasing on (0, 1),
    prove that b ≥ 3 -/
theorem monotone_cubic_function_condition (b : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, Monotone (fun x => -x^3 + b*x)) →
  b ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_cubic_function_condition_l424_42421


namespace NUMINAMATH_CALUDE_tenth_pattern_stones_l424_42487

def stone_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => stone_sequence n + 3 * (n + 2) - 2

theorem tenth_pattern_stones : stone_sequence 9 = 145 := by
  sorry

end NUMINAMATH_CALUDE_tenth_pattern_stones_l424_42487


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l424_42401

theorem complex_sum_theorem :
  let A : ℂ := 3 + 2*I
  let O : ℂ := -1 - 2*I
  let P : ℂ := 2*I
  let S : ℂ := 1 + 3*I
  A - O + P + S = 5 + 9*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l424_42401


namespace NUMINAMATH_CALUDE_cars_distance_theorem_l424_42407

/-- The distance between two cars after their movements on a main road -/
def final_distance (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - (car1_distance + car2_distance)

/-- Theorem stating the final distance between two cars -/
theorem cars_distance_theorem (initial_distance car1_distance car2_distance : ℝ) 
  (h1 : initial_distance = 113)
  (h2 : car1_distance = 50)
  (h3 : car2_distance = 35) :
  final_distance initial_distance car1_distance car2_distance = 28 := by
  sorry

#eval final_distance 113 50 35

end NUMINAMATH_CALUDE_cars_distance_theorem_l424_42407


namespace NUMINAMATH_CALUDE_main_theorem_l424_42426

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define a point P on line l
structure Point_P where
  x : ℝ
  y : ℝ
  on_line : line_l x y

-- Define the tangent length
def tangent_length (p : Point_P) : ℝ := sorry

-- Define the circumcircle N
def circle_N (p : Point_P) (x y : ℝ) : Prop := sorry

-- Define the chord length AB
def chord_length (p : Point_P) : ℝ := sorry

theorem main_theorem :
  (∃ p1 p2 : Point_P, tangent_length p1 = 2*Real.sqrt 3 ∧ tangent_length p2 = 2*Real.sqrt 3 ∧
    ((p1.x = 0 ∧ p1.y = 0) ∨ (p1.x = 16/5 ∧ p1.y = 8/5)) ∧
    ((p2.x = 0 ∧ p2.y = 0) ∨ (p2.x = 16/5 ∧ p2.y = 8/5))) ∧
  (∀ p : Point_P, circle_N p 0 4 ∧ circle_N p (8/5) (4/5)) ∧
  (∃ p_min : Point_P, ∀ p : Point_P, chord_length p_min ≤ chord_length p ∧ chord_length p_min = Real.sqrt 11) :=
sorry

end NUMINAMATH_CALUDE_main_theorem_l424_42426


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l424_42466

theorem complex_number_in_second_quadrant :
  let z : ℂ := (3 + 4 * Complex.I) * Complex.I
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l424_42466


namespace NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l424_42400

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of prime divisors of 50! is equal to the number of prime numbers less than or equal to 50. -/
theorem prime_divisors_of_50_factorial (p : ℕ → Prop) :
  (∃ (n : ℕ), p n ∧ n ∣ factorial 50) ↔ (∃ (n : ℕ), p n ∧ n ≤ 50) :=
sorry

end NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l424_42400


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l424_42468

/-- Given two hyperbolas x²/9 - y²/16 = 1 and y²/25 - x²/M = 1,
    prove that M = 225/16 for the hyperbolas to have the same asymptotes -/
theorem hyperbolas_same_asymptotes :
  ∀ M : ℝ,
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) →
  (∀ k : ℝ, (∃ x y : ℝ, y = k * x ∧ x^2 / 9 - y^2 / 16 = 1) ↔
            (∃ x y : ℝ, y = k * x ∧ y^2 / 25 - x^2 / M = 1)) →
  M = 225 / 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l424_42468


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l424_42495

theorem largest_divisor_of_difference_of_squares (m n : ℤ) : 
  Odd m → Odd n → n < m → 
  (∃ k : ℤ, m ^ 2 - n ^ 2 = 8 * k) ∧ 
  (∀ d : ℤ, d > 8 → ∃ m' n' : ℤ, Odd m' ∧ Odd n' ∧ n' < m' ∧ ¬(d ∣ (m' ^ 2 - n' ^ 2))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l424_42495


namespace NUMINAMATH_CALUDE_floor_length_percentage_l424_42435

/-- Proves that for a rectangular floor with given length and area, 
    the percentage by which the length is more than the breadth is 200% -/
theorem floor_length_percentage (length : ℝ) (area : ℝ) :
  length = 19.595917942265423 →
  area = 128 →
  let breadth := area / length
  ((length - breadth) / breadth) * 100 = 200 := by sorry

end NUMINAMATH_CALUDE_floor_length_percentage_l424_42435


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l424_42465

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 1) * x + 3

-- Part 1
theorem part_one (a b : ℝ) (ha : a ≠ 0) 
  (h_solution_set : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) :
  2 * a + b = -3 := by sorry

-- Part 2
theorem part_two (a b : ℝ) (ha : a ≠ 0) (hf1 : f a b 1 = 5) (hb : b > -1) :
  (∀ a' b', a' ≠ 0 → b' > -1 → f a' b' 1 = 5 → 
    1 / |a| + 4 * |a| / (b + 1) ≤ 1 / |a'| + 4 * |a'| / (b' + 1)) ∧
  1 / |a| + 4 * |a| / (b + 1) = 2 := by sorry

-- Part 3
theorem part_three (a : ℝ) (ha : a ≠ 0) :
  let b := -a - 3
  let solution_set := {x : ℝ | f a b x < -2 * x + 1}
  (a < 0 → solution_set = {x | x < 2/a ∨ x > 1}) ∧
  (0 < a ∧ a < 2 → solution_set = {x | 1 < x ∧ x < 2/a}) ∧
  (a = 2 → solution_set = ∅) ∧
  (a > 2 → solution_set = {x | 2/a < x ∧ x < 1}) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l424_42465


namespace NUMINAMATH_CALUDE_cubic_sum_powers_l424_42492

theorem cubic_sum_powers (a : ℝ) (h : a^3 + 3*a^2 + 3*a + 2 = 0) :
  (a + 1)^2008 + (a + 1)^2009 + (a + 1)^2010 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_powers_l424_42492


namespace NUMINAMATH_CALUDE_exist_integers_with_gcd_property_l424_42403

theorem exist_integers_with_gcd_property :
  ∃ (a : Fin 2011 → ℕ+), (∀ i j, i < j → a i < a j) ∧
    (∀ i j, i < j → Nat.gcd (a i) (a j) = (a j) - (a i)) := by
  sorry

end NUMINAMATH_CALUDE_exist_integers_with_gcd_property_l424_42403


namespace NUMINAMATH_CALUDE_cubic_function_unique_negative_zero_l424_42410

/-- Given a cubic function f(x) = ax³ - 3x² + 1 with a unique zero point x₀ < 0, prove that a > 2 -/
theorem cubic_function_unique_negative_zero (a : ℝ) :
  (∃! x₀ : ℝ, a * x₀^3 - 3 * x₀^2 + 1 = 0) →
  (∀ x₀ : ℝ, a * x₀^3 - 3 * x₀^2 + 1 = 0 → x₀ < 0) →
  a > 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_unique_negative_zero_l424_42410


namespace NUMINAMATH_CALUDE_coefficients_of_given_equation_l424_42491

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given quadratic equation x^2 - x + 3 = 0 -/
def givenEquation : QuadraticEquation :=
  { a := 1, b := -1, c := 3 }

theorem coefficients_of_given_equation :
  givenEquation.a = 1 ∧ givenEquation.b = -1 ∧ givenEquation.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_given_equation_l424_42491


namespace NUMINAMATH_CALUDE_girl_multiplication_problem_l424_42484

theorem girl_multiplication_problem (mistake_factor : ℕ) (difference : ℕ) (base : ℕ) (correct_factor : ℕ) : 
  mistake_factor = 34 →
  difference = 1233 →
  base = 137 →
  base * correct_factor = base * mistake_factor + difference →
  correct_factor = 43 := by
sorry

end NUMINAMATH_CALUDE_girl_multiplication_problem_l424_42484


namespace NUMINAMATH_CALUDE_face_value_of_shares_l424_42473

/-- Calculates the face value of shares given investment details -/
theorem face_value_of_shares
  (investment : ℝ)
  (quoted_price : ℝ)
  (dividend_rate : ℝ)
  (annual_income : ℝ)
  (h1 : investment = 4940)
  (h2 : quoted_price = 9.5)
  (h3 : dividend_rate = 0.14)
  (h4 : annual_income = 728)
  : ∃ (face_value : ℝ),
    face_value = 10 ∧
    annual_income = (investment / quoted_price) * (dividend_rate * face_value) :=
by sorry

end NUMINAMATH_CALUDE_face_value_of_shares_l424_42473


namespace NUMINAMATH_CALUDE_square_diff_div_four_xy_eq_one_l424_42446

theorem square_diff_div_four_xy_eq_one (x y : ℝ) (h : x * y ≠ 0) :
  ((x + y)^2 - (x - y)^2) / (4 * x * y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_div_four_xy_eq_one_l424_42446


namespace NUMINAMATH_CALUDE_pentagon_rectangle_apothem_ratio_l424_42429

theorem pentagon_rectangle_apothem_ratio :
  let pentagon_side := (40 : ℝ) / (1 + Real.sqrt 5)
  let pentagon_apothem := pentagon_side * ((1 + Real.sqrt 5) / 4)
  let rectangle_width := (3 : ℝ) / 2
  let rectangle_apothem := rectangle_width / 2
  pentagon_apothem / rectangle_apothem = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_apothem_ratio_l424_42429


namespace NUMINAMATH_CALUDE_count_integers_with_at_most_three_divisors_cubic_plus_eight_l424_42443

def has_at_most_three_divisors (x : ℤ) : Prop :=
  (∃ p : ℕ, Prime p ∧ x = p^2) ∨ (∃ p : ℕ, Prime p ∧ x = p) ∨ x = 1

theorem count_integers_with_at_most_three_divisors_cubic_plus_eight :
  ∃! (S : Finset ℤ), ∀ n : ℤ, n ∈ S ↔ has_at_most_three_divisors (n^3 + 8) ∧ Finset.card S = 2 :=
sorry

end NUMINAMATH_CALUDE_count_integers_with_at_most_three_divisors_cubic_plus_eight_l424_42443


namespace NUMINAMATH_CALUDE_x_range_for_f_l424_42455

-- Define the function f
def f (x : ℝ) := x^3 + 3*x

-- State the theorem
theorem x_range_for_f (x : ℝ) :
  (∀ m ∈ Set.Icc (-2 : ℝ) 2, f (m*x - 2) + f x < 0) →
  x ∈ Set.Ioo (-2 : ℝ) (2/3) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_f_l424_42455


namespace NUMINAMATH_CALUDE_bus_stop_problem_l424_42496

/-- The number of children who got on the bus at the bus stop -/
def children_got_on : ℕ := sorry

/-- The initial number of children on the bus -/
def initial_children : ℕ := 22

/-- The number of children who got off the bus at the bus stop -/
def children_got_off : ℕ := 60

/-- The final number of children on the bus after the bus stop -/
def final_children : ℕ := 2

theorem bus_stop_problem :
  initial_children - children_got_off + children_got_on = final_children ∧
  children_got_on = 40 := by sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l424_42496


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l424_42404

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.82 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 82 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l424_42404


namespace NUMINAMATH_CALUDE_power_product_equality_l424_42417

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 11 = 39600 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l424_42417


namespace NUMINAMATH_CALUDE_complex_power_modulus_l424_42422

theorem complex_power_modulus : Complex.abs ((2 + 2*Complex.I)^6) = 512 := by sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l424_42422


namespace NUMINAMATH_CALUDE_range_of_b_for_two_intersection_points_l424_42474

/-- The range of b for which there are exactly two points P satisfying the given conditions -/
theorem range_of_b_for_two_intersection_points (b : ℝ) : 
  (∃! (P₁ P₂ : ℝ × ℝ), 
    P₁ ≠ P₂ ∧ 
    (P₁.1 + Real.sqrt 3 * P₁.2 = b) ∧ 
    (P₂.1 + Real.sqrt 3 * P₂.2 = b) ∧ 
    ((P₁.1 - 4)^2 + P₁.2^2 = 4 * (P₁.1^2 + P₁.2^2)) ∧
    ((P₂.1 - 4)^2 + P₂.2^2 = 4 * (P₂.1^2 + P₂.2^2))) ↔ 
  (-20/3 < b ∧ b < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_for_two_intersection_points_l424_42474
