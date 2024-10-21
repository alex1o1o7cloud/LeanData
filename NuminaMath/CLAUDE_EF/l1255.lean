import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibboican_numerator_power_of_two_l1255_125529

def fibboican : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | k+2 => if k % 2 = 0 then 
             fibboican (k+1) + fibboican k
           else 
             1 / (1 / fibboican (k+1) + 1 / fibboican k)

theorem fibboican_numerator_power_of_two (m : ℕ) (hm : m ≥ 1) :
  ∃ (p : ℕ) (q : ℕ), Odd q ∧ fibboican m = (2^p : ℚ) / q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibboican_numerator_power_of_two_l1255_125529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_l1255_125573

/-- Calculates the time (in seconds) for a train to cross a platform -/
noncomputable def train_crossing_time (train_speed_kmh : ℝ) (train_length_m : ℝ) (platform_length_m : ℝ) : ℝ :=
  let total_distance_m := train_length_m + platform_length_m
  let train_speed_ms := train_speed_kmh * (5/18)
  total_distance_m / train_speed_ms

/-- Theorem: A train with speed 72 km/h and length 240 m takes 26 seconds to cross a 280 m platform -/
theorem train_crossing_platform :
  train_crossing_time 72 240 280 = 26 := by
  unfold train_crossing_time
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_platform_l1255_125573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_3_equals_9_l1255_125547

-- Define the inverse function f⁻¹
noncomputable def f_inv (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem f_3_equals_9 : 
  (∃ f : ℝ → ℝ, (∀ x, f_inv (f x) = x) ∧ (∀ y, f (f_inv y) = y)) → 
  (∃ f : ℝ → ℝ, f 3 = 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_3_equals_9_l1255_125547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_spheres_volume_l1255_125516

/-- Regular quadrilateral prism -/
structure RegularQuadPrism where
  height : ℝ
  angle : ℝ

/-- Sequence of inscribed spheres in the prism -/
noncomputable def InscribedSpheres (prism : RegularQuadPrism) : ℕ → ℝ
  | 0 => prism.height / 3
  | n + 1 => (1 / 3) * InscribedSpheres prism n

/-- Total volume of all inscribed spheres -/
noncomputable def TotalVolume (prism : RegularQuadPrism) : ℝ :=
  (4 * Real.pi / 3) * (27 / 26) * (InscribedSpheres prism 0) ^ 3

theorem inscribed_spheres_volume (prism : RegularQuadPrism) 
    (h : prism.height = 3) 
    (θ : prism.angle = Real.pi / 3) : 
    TotalVolume prism = 18 * Real.pi / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_spheres_volume_l1255_125516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_area_l1255_125594

/-- Define a square -/
def is_square (S : Set (ℝ × ℝ)) : Prop := sorry

/-- Define the vertices of a square -/
def vertices (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Define the extended edges of a square -/
def extended_edges (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Define the side length of a square -/
noncomputable def side_length (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Define the distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Define point B of square ABCD -/
noncomputable def point_B (S : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Define point E of square EFGH -/
noncomputable def point_E (S : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Define the area of a square -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Given two squares ABCD and EFGH where EFGH is inside ABCD, 
    EFGH can be extended to pass through a vertex of ABCD,
    the side length of ABCD is 10, and BE = 2,
    prove that the area of EFGH is 100 - 4√96 -/
theorem inner_square_area (ABCD EFGH : Set (ℝ × ℝ)) :
  is_square ABCD → 
  is_square EFGH →
  EFGH ⊆ ABCD →
  (∃ (v : ℝ × ℝ), v ∈ vertices ABCD ∧ v ∈ extended_edges EFGH) →
  side_length ABCD = 10 →
  distance (point_B ABCD) (point_E EFGH) = 2 →
  area EFGH = 100 - 4 * Real.sqrt 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_area_l1255_125594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_tax_is_four_percent_l1255_125543

/-- Represents the tax rates and spending percentages for Jill's shopping trip -/
structure ShoppingTaxes where
  clothing_spending : ℚ
  food_spending : ℚ
  other_spending : ℚ
  other_tax_rate : ℚ
  total_tax_rate : ℚ

/-- Calculates the tax rate on clothing given the shopping tax structure -/
def clothing_tax_rate (s : ShoppingTaxes) : ℚ :=
  (s.total_tax_rate - s.other_tax_rate * s.other_spending) / s.clothing_spending

/-- Theorem stating that the clothing tax rate is 4% given the specific conditions -/
theorem clothing_tax_is_four_percent : 
  let s : ShoppingTaxes := {
    clothing_spending := 1/2,
    food_spending := 1/5,
    other_spending := 3/10,
    other_tax_rate := 2/25,
    total_tax_rate := 11/250
  }
  clothing_tax_rate s = 1/25 := by
  sorry

#eval clothing_tax_rate {
  clothing_spending := 1/2,
  food_spending := 1/5,
  other_spending := 3/10,
  other_tax_rate := 2/25,
  total_tax_rate := 11/250
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_tax_is_four_percent_l1255_125543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2018_equals_2_l1255_125514

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => -1
  | n + 1 => 1 - 1 / sequenceA n

theorem sequence_2018_equals_2 : sequenceA 2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2018_equals_2_l1255_125514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottom_distance_is_53_l1255_125532

/-- A circular plate on a square table -/
structure PlateOnTable where
  /-- The distance from the left edge of the plate to the left edge of the table -/
  left_distance : ℝ
  /-- The distance from the right edge of the plate to the right edge of the table -/
  right_distance : ℝ
  /-- The distance from the top edge of the plate to the top edge of the table -/
  top_distance : ℝ
  /-- The distance from the bottom edge of the plate to the bottom edge of the table -/
  bottom_distance : ℝ
  /-- The table is square -/
  square_table : left_distance + right_distance = top_distance + bottom_distance
  /-- The left distance is 10 cm -/
  left_is_10 : left_distance = 10
  /-- The right distance is 63 cm -/
  right_is_63 : right_distance = 63
  /-- The top distance is 20 cm -/
  top_is_20 : top_distance = 20

/-- The distance from the bottom edge of the plate to the bottom edge of the table is 53 cm -/
theorem bottom_distance_is_53 (pot : PlateOnTable) : pot.bottom_distance = 53 := by
  have h1 : pot.left_distance + pot.right_distance = 73 := by
    rw [pot.left_is_10, pot.right_is_63]
    norm_num
  
  have h2 : pot.top_distance + pot.bottom_distance = 73 := by
    rw [pot.square_table] at h1
    exact h1

  have h3 : pot.bottom_distance = 73 - pot.top_distance := by
    linarith

  rw [pot.top_is_20] at h3
  norm_num at h3
  exact h3


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottom_distance_is_53_l1255_125532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_with_conditions_l1255_125582

theorem least_sum_with_conditions (m n : ℕ) 
  (hm : m > 0)
  (hn : n > 0)
  (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m^m = k * n^n)
  (h3 : ¬∃ k : ℕ, m = k * n) :
  m + n ≥ 119 ∧ ∃ (m₀ n₀ : ℕ), 
    m₀ > 0 ∧ n₀ > 0 ∧
    m₀ + n₀ = 119 ∧ 
    Nat.gcd (m₀ + n₀) 330 = 1 ∧ 
    (∃ k : ℕ, m₀^m₀ = k * n₀^n₀) ∧ 
    ¬∃ k : ℕ, m₀ = k * n₀ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_with_conditions_l1255_125582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_theorem_l1255_125597

/-- The distance of a point from the other side of an angle -/
noncomputable def distance_to_other_side (α : ℝ) (a b : ℝ) : ℝ :=
  Real.sin α * Real.sqrt (a^2 - b^2) - b * Real.cos α

/-- Theorem stating the distance of a point from the other side of an angle -/
theorem distance_theorem (α : ℝ) (a b : ℝ) 
  (h1 : 0 < α) (h2 : α < Real.pi) (h3 : 0 < a) (h4 : 0 < b) (h5 : b < a) :
  ∃ (x : ℝ), x = distance_to_other_side α a b ∧ 
  x = Real.sin α * Real.sqrt (a^2 - b^2) - b * Real.cos α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_theorem_l1255_125597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raisin_water_percentage_is_16_l1255_125593

/-- The percentage of water in grapes -/
def grape_water_percentage : ℚ := 93

/-- The weight of the raisins in kilograms -/
def raisin_weight : ℚ := 8

/-- The weight of the grapes before drying in kilograms -/
def grape_weight : ℚ := 96

/-- Calculate the percentage of water in raisins -/
noncomputable def raisin_water_percentage : ℚ := 
  let original_water_weight := grape_weight * (grape_water_percentage / 100)
  let water_weight_in_raisins := original_water_weight - (grape_weight - raisin_weight)
  (water_weight_in_raisins / raisin_weight) * 100

theorem raisin_water_percentage_is_16 : 
  raisin_water_percentage = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raisin_water_percentage_is_16_l1255_125593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l1255_125574

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function y = ln(√(1 + ax²) - x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (Real.sqrt (1 + a * x^2) - x)

/-- If f(a) is an odd function, then a = 1 -/
theorem odd_function_implies_a_equals_one :
    ∀ a : ℝ, IsOdd (f a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l1255_125574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_evenness_oddness_and_range_l1255_125544

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * 2^x - 1) / (2^x + 1)

-- Define evenness and oddness
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem f_evenness_oddness_and_range (a : ℝ) :
  (is_even (f a) ↔ a = -1) ∧
  (is_odd (f a) ↔ a = 1) ∧
  ((∀ x ≥ 1, 1 ≤ f a x ∧ f a x ≤ 3) ↔ 2 ≤ a ∧ a ≤ 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_evenness_oddness_and_range_l1255_125544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_rotated_arc_l1255_125590

-- Define the circle and rotation parameters
variable (R a y₁ y₂ : ℝ)

-- Define the conditions
variable (h₁ : 0 < y₁)
variable (h₂ : y₁ ≤ y₂)
variable (h₃ : y₂ < R)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - a)^2 = R^2

-- Define the surface area function
noncomputable def surface_area : ℝ := 2 * Real.pi * R * (y₂ - y₁)

-- State the theorem
theorem surface_area_of_rotated_arc :
  surface_area R y₁ y₂ = 2 * Real.pi * R * (y₂ - y₁) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_rotated_arc_l1255_125590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1255_125542

def a : ℕ → ℕ
  | 0 => 5  -- Added case for 0
  | 1 => 5
  | n + 1 => 2 * a n + 1

def S (n : ℕ) : ℕ := (List.range n).map a |>.sum

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → (a (n + 1) + 1) = 2 * (a n + 1)) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 6 * 2^(n - 1) - 1) ∧
  (∀ n : ℕ, n ≥ 1 → S n = 6 * 2^n - n - 6) := by
  sorry

#eval a 5  -- Added to test the function
#eval S 5  -- Added to test the sum function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1255_125542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_caramel_price_is_three_l1255_125524

/-- The price of a single caramel in dollars -/
noncomputable def caramel_price : ℝ := sorry

/-- The price of a single candy bar in dollars -/
noncomputable def candy_bar_price : ℝ := 2 * caramel_price

/-- The price of a single cotton candy in dollars -/
noncomputable def cotton_candy_price : ℝ := (1/2) * 4 * candy_bar_price

/-- The total cost of 6 candy bars, 3 caramels, and 1 cotton candy in dollars -/
noncomputable def total_cost : ℝ := 6 * candy_bar_price + 3 * caramel_price + cotton_candy_price

theorem caramel_price_is_three :
  total_cost = 57 → caramel_price = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_caramel_price_is_three_l1255_125524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_marble_ratio_l1255_125549

/-- Represents the number of marbles of each color in a jar --/
structure MarbleJar where
  total : ℕ
  red : ℕ
  orange : ℕ
  blue : ℕ

/-- The ratio of blue marbles to total marbles --/
def blueRatio (jar : MarbleJar) : ℚ :=
  ↑jar.blue / ↑jar.total

theorem blue_marble_ratio (jar : MarbleJar) 
  (h_total : jar.total = 24)
  (h_red : jar.red = 6)
  (h_orange : jar.orange = 6)
  (h_sum : jar.red + jar.orange + jar.blue = jar.total) :
  blueRatio jar = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_marble_ratio_l1255_125549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l1255_125563

-- Define set A
def A : Set ℕ := {x : ℕ | x ≤ 2}

-- Define set B
def B : Set ℤ := {-1, 1}

-- Helper function to convert ℕ to ℤ
def natToInt (n : ℕ) : ℤ := Int.ofNat n

-- Theorem statement
theorem union_of_A_and_B :
  (A.image natToInt) ∪ B = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l1255_125563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023rd_term_l1255_125599

def digit_sum_of_squares (n : Nat) : Nat :=
  (n.digits 10).map (λ d => d * d) |>.sum

def sequence_term (n : Nat) : Nat :=
  match n with
  | 0 => 2023
  | n + 1 => digit_sum_of_squares (sequence_term n)

theorem sequence_2023rd_term : sequence_term 2022 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023rd_term_l1255_125599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_l1255_125553

/-- The number of ways to color a 1 × n chessboard with m colors, where adjacent squares have different colors. -/
def colorings (m n : ℕ) : ℕ :=
  m * (m - 1) ^ (n - 1)

/-- A function representing the actual number of valid colorings. -/
noncomputable def number_of_valid_colorings : ℕ → ℕ → ℕ :=
  fun m n => m * (m - 1) ^ (n - 1)

/-- Theorem stating the number of valid colorings for a 1 × n chessboard with m colors. -/
theorem valid_colorings (m n : ℕ) (h1 : m ≥ 2) (h2 : n > 0) :
  colorings m n = number_of_valid_colorings m n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_colorings_l1255_125553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_boys_l1255_125521

/-- Given a school with 200 total students, where 120 students represent x percent of the boys,
    prove that the percentage y of boys in the total school population is given by y = 6000 / x. -/
theorem percentage_of_boys (x : ℝ) (h : x ≠ 0) :
  let total_students : ℝ := 200
  let boys_subset : ℝ := 120
  let y := (boys_subset / (x / 100)) / total_students * 100
  y = 6000 / x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_boys_l1255_125521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l1255_125554

theorem vector_decomposition (x p q r : ℝ × ℝ × ℝ) : 
  x = (-1, 7, -4) →
  p = (-1, 2, 1) →
  q = (2, 0, 3) →
  r = (1, 1, -1) →
  x = 2 • p - 1 • q + 3 • r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l1255_125554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_shifted_polynomial_l1255_125556

theorem roots_shifted_polynomial (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 6*x^2 + 11*x - 6 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (∀ x : ℝ, x^3 + 3*x^2 + 2*x = 0 ↔ (x = a - 3 ∨ x = b - 3 ∨ x = c - 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_shifted_polynomial_l1255_125556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1255_125578

/-- The fixed point A -/
def A : ℝ × ℝ := (2, 0)

/-- The x-coordinate of the fixed line l -/
def l : ℝ := -1

/-- The distance from a point P(x, y) to the fixed point A -/
noncomputable def dist_to_A (P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

/-- The distance from a point P(x, y) to the fixed line l -/
def dist_to_l (P : ℝ × ℝ) : ℝ :=
  |P.1 - l|

/-- The condition that the distance from P to A is greater by 1 than its distance to l -/
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  dist_to_A P = dist_to_l P + 1

theorem trajectory_equation (P : ℝ × ℝ) (h : satisfies_condition P) :
  P.2^2 = 8 * P.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1255_125578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_science_only_students_l1255_125586

theorem science_only_students (total science tech : ℕ)
  (h1 : total = 150)
  (h2 : science = 110)
  (h3 : tech = 97)
  (h4 : ∀ s, s ≤ total → s ≤ science ∨ s ≤ tech) :
  science - (science + tech - total) = 53 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_science_only_students_l1255_125586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_to_line_l1255_125557

/-- The distance formula from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- Theorem stating that for a point A(a, 2) at a distance of √2 from the line x - y + 3 = 0,
    the value of a is either 1 or -3 -/
theorem point_distance_to_line (a : ℝ) :
  distance_point_to_line a 2 1 (-1) 3 = Real.sqrt 2 ↔ a = 1 ∨ a = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_to_line_l1255_125557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l1255_125525

-- Define the triangle and points
variable (A B C : EuclideanSpace ℝ (Fin 2))
variable (Bₐ Bₒ Aₒ Aᵦ Cₐ Cᵦ : EuclideanSpace ℝ (Fin 2))
variable (P : EuclideanSpace ℝ (Fin 2))

-- Define the squares outside the triangle
noncomputable def square_ABcBaC : Set (EuclideanSpace ℝ (Fin 2)) := sorry
noncomputable def square_CAbAcB : Set (EuclideanSpace ℝ (Fin 2)) := sorry
noncomputable def square_BCaCbA : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the square outside AB₍c₎B₍a₎C
noncomputable def square_BcBcBaBa : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the lines
noncomputable def line_BP : Set (EuclideanSpace ℝ (Fin 2)) := sorry
noncomputable def line_CaBa : Set (EuclideanSpace ℝ (Fin 2)) := sorry
noncomputable def line_AcBc : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- The theorem to prove
theorem lines_concurrent :
  ∃ (X : EuclideanSpace ℝ (Fin 2)), X ∈ line_BP ∧ X ∈ line_CaBa ∧ X ∈ line_AcBc :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l1255_125525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crosses_configuration_exists_l1255_125592

/-- A configuration of crosses in a 4x4 grid -/
def Configuration := Fin 4 → Fin 4 → Bool

/-- The number of crosses in a given configuration -/
def num_crosses (c : Configuration) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 4)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 4)) (λ j =>
      if c i j then 1 else 0))

/-- The number of crosses in a specific row -/
def crosses_in_row (c : Configuration) (row : Fin 4) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 4)) (λ col =>
    if c row col then 1 else 0)

/-- The number of crosses in a specific column -/
def crosses_in_column (c : Configuration) (col : Fin 4) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 4)) (λ row =>
    if c row col then 1 else 0)

/-- There exists a configuration with 8 crosses where no row or column has exactly 2 crosses -/
theorem crosses_configuration_exists : ∃ (c : Configuration),
  num_crosses c = 8 ∧
  (∀ row, crosses_in_row c row ≠ 2) ∧
  (∀ col, crosses_in_column c col ≠ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crosses_configuration_exists_l1255_125592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l1255_125598

theorem diophantine_equation_solution (x y z v lambda : ℕ+) :
  (x^2 + y^2 + z^2 + v^2 = lambda * x * y * z * v) ↔ (lambda = 1 ∨ lambda = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l1255_125598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_side_length_proof_l1255_125572

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi

theorem triangle_proof (t : Triangle) (h : TriangleConditions t) :
  (Real.sqrt 3 * t.c * Real.cos t.A + t.a * Real.sin t.C = Real.sqrt 3 * t.c) →
  t.A = Real.pi / 3 :=
by sorry

theorem side_length_proof (t : Triangle) (h : TriangleConditions t) :
  (t.b + t.c = 5) →
  (1 / 2 * t.b * t.c * Real.sin t.A = Real.sqrt 3) →
  (t.A = Real.pi / 3) →
  t.a = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_side_length_proof_l1255_125572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_real_l1255_125564

theorem complex_product_real (a b c d : ℝ) :
  (∃ r : ℝ, (Complex.mk a b) * (Complex.mk c d) = r) ↔ a * d + b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_real_l1255_125564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_midpoint_triangle_l1255_125562

/-- A right prism with regular pentagonal bases -/
structure RightPentagonalPrism where
  height : ℝ
  base_side_length : ℝ

/-- Points on the edges of the prism -/
structure PrismPoint where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The perimeter of a triangle given its three vertices -/
noncomputable def triangle_perimeter (A B C : PrismPoint) : ℝ :=
  ((A.x - B.x)^2 + (A.y - B.y)^2 + (A.z - B.z)^2).sqrt +
  ((B.x - C.x)^2 + (B.y - C.y)^2 + (B.z - C.z)^2).sqrt +
  ((C.x - A.x)^2 + (C.y - A.y)^2 + (C.z - A.z)^2).sqrt

/-- The main theorem -/
theorem perimeter_of_midpoint_triangle (prism : RightPentagonalPrism) 
  (X Y Z : PrismPoint) : 
  prism.height = 20 →
  prism.base_side_length = 10 →
  -- X is midpoint of AB
  -- Y is midpoint of BC
  -- Z is midpoint of CD
  triangle_perimeter X Y Z = 4.57 + 10 * (5 : ℝ).sqrt := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_midpoint_triangle_l1255_125562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_impossibility_l1255_125575

theorem test_score_impossibility : ∀ (correct incorrect unanswered : ℕ),
  correct + incorrect + unanswered = 25 →
  (4 : ℤ) * correct - incorrect + unanswered ≠ 85 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_impossibility_l1255_125575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_on_positive_reals_l1255_125534

open Real

noncomputable def f (x : ℝ) : ℝ := 1/x - x

theorem decreasing_f_on_positive_reals :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
  (x₁ - x₂) * (f x₁ - f x₂) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_on_positive_reals_l1255_125534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_mapping_l1255_125558

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {-1, 0, 1}

def f (x : ℤ) : ℤ := x^2

theorem is_mapping : 
  (∀ x, x ∈ A → f x ∈ B) ∧ 
  (∀ x, x ∈ A → ∃! y, y ∈ B ∧ f x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_mapping_l1255_125558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_axis_length_of_hyperbola_l1255_125526

/-- The length of the conjugate axis of a hyperbola with equation y²/4 - x² = 1 is 2 -/
theorem conjugate_axis_length_of_hyperbola : 
  ∃ (f : ℝ → ℝ → ℝ), 
    (∀ x y, f x y = y^2/4 - x^2) ∧ 
    (∀ x y, f x y = 1 → 2 = (let b := 1; 2*b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_axis_length_of_hyperbola_l1255_125526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_for_six_cookies_l1255_125517

-- Define the relationship between cookies and milk
noncomputable def cookies_to_quarts (cookies : ℚ) : ℚ := (5 / 24) * cookies

-- Define the conversion from quarts to pints
def quarts_to_pints (quarts : ℚ) : ℚ := 2 * quarts

-- Theorem statement
theorem milk_for_six_cookies :
  quarts_to_pints (cookies_to_quarts 6) = 5/2 := by
  -- Unfold definitions
  unfold quarts_to_pints cookies_to_quarts
  -- Simplify the expression
  simp [mul_assoc, mul_comm, mul_div_assoc]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_for_six_cookies_l1255_125517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_ab_l1255_125539

-- Define the rates of water flow for each valve
variable (a b c : ℝ)

-- Define the time it takes to fill the tank with different valve combinations
noncomputable def time_abc : ℝ := 1/2
noncomputable def time_ac : ℝ := 1
noncomputable def time_bc : ℝ := 2

-- Define the relationships between rates and times
axiom fill_abc : a + b + c = 1 / time_abc
axiom fill_ac : a + c = 1 / time_ac
axiom fill_bc : b + c = 1 / time_bc

-- Define the time it takes to fill the tank with valves A and B
noncomputable def time_ab : ℝ := 1 / (a + b)

-- Theorem to prove
theorem fill_time_ab : time_ab = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_ab_l1255_125539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_tangent_circle_l1255_125500

/-- Given a circle and a hyperbola where the circle is tangent to one of the asymptotes of the hyperbola,
    the eccentricity of the hyperbola is 2√3/3 -/
theorem hyperbola_eccentricity_with_tangent_circle 
  (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) : 
  let circle := fun (x y : ℝ) => (x - Real.sqrt 3)^2 + (y - 1)^2 = 3
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let asymptote := fun (x y : ℝ) => b * x - a * y = 0 ∨ b * x + a * y = 0
  let is_tangent := ∃ (x y : ℝ), circle x y ∧ asymptote x y
  let e := Real.sqrt (a^2 + b^2) / a  -- eccentricity formula
  is_tangent → e = 2 * Real.sqrt 3 / 3 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_tangent_circle_l1255_125500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_point_iff_catcher_l1255_125552

-- Define a sequence of real numbers
def Sequence : ℕ → ℝ := sorry

-- Define what it means for a real number to be a limit point of a sequence
def IsLimitPoint (a : ℝ) (x : ℕ → ℝ) : Prop :=
  ∀ (ε : ℝ) (k : ℕ), ε > 0 → ∃ (n : ℕ), n > k ∧ |x n - a| < ε

-- Define what it means for an interval to be a "catcher" for a sequence
def IsCatcher (a : ℝ) (x : ℕ → ℝ) : Prop :=
  ∀ (ε : ℝ), ε > 0 → ∀ (N : ℕ), ∃ (n : ℕ), n ≥ N ∧ |x n - a| < ε

-- State the theorem
theorem limit_point_iff_catcher (a : ℝ) (x : ℕ → ℝ) :
  IsLimitPoint a x ↔ IsCatcher a x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_point_iff_catcher_l1255_125552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1255_125533

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ

def Parabola.equation (para : Parabola) (point : Point) : Prop :=
  point.y^2 = 2 * para.p * point.x

def Line.equation (line : Line) (point : Point) : Prop :=
  point.y = line.slope * point.x + line.intercept

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_intersection_theorem (para : Parabola) (line : Line) 
    (A B : Point) (h_para : para.p > 0) (h_slope : line.slope = 2 * Real.sqrt 2)
    (h_int_A : Parabola.equation para A ∧ Line.equation line A)
    (h_int_B : Parabola.equation para B ∧ Line.equation line B)
    (h_order : A.x < B.x) (h_distance : distance A B = 9/2) :
  para.p = 2 ∧
  ∀ (P Q : Point) (M : Point := ⟨1, 2⟩),
    Parabola.equation para P ∧ Parabola.equation para Q ∧
    Parabola.equation para M ∧
    (P.x - M.x) * (Q.x - M.x) + (P.y - M.y) * (Q.y - M.y) = 0 →
    ∃ (PQ : Line),
      Line.equation PQ P ∧ Line.equation PQ Q ∧
      Line.equation PQ ⟨5, -2⟩ ∧
      (∀ (R : Point), Line.equation PQ R → distance ⟨0, 0⟩ R ≤ Real.sqrt 29) ∧
      (∃ (S : Point), Line.equation PQ S ∧ distance ⟨0, 0⟩ S = Real.sqrt 29) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l1255_125533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_minus_sin_squared_transformation_l1255_125509

theorem cos_squared_minus_sin_squared_transformation (x : ℝ) :
  (Real.cos x)^2 - (Real.sin x)^2 = Real.sin (2*x + Real.pi/2) ∧
  ∃ (k : ℝ), k = Real.pi/4 ∧ (Real.cos x)^2 - (Real.sin x)^2 = Real.sin (2*(x + k)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_minus_sin_squared_transformation_l1255_125509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1255_125585

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1) - x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := 2 / (2 * x - 1) - 1

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -1
  let m : ℝ := f_derivative x₀
  (∀ x y, y - y₀ = m * (x - x₀) ↔ x - y - 2 = 0) ∧
  f x₀ = y₀ ∧
  f_derivative x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1255_125585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l1255_125545

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola) : ℝ × ℝ :=
  (h.a * eccentricity h, 0)

/-- A point on the asymptote of the hyperbola -/
noncomputable def asymptote_point (h : Hyperbola) : ℝ × ℝ :=
  (h.a * eccentricity h / 2, -h.b * eccentricity h / 2)

/-- Theorem: For a hyperbola, if a line passing through its right focus and parallel to one asymptote
    intersects the other asymptote at point M such that ∠F₁MF₂ is acute, then the eccentricity e > 2 -/
theorem hyperbola_eccentricity_bound (h : Hyperbola) : 
  let M := asymptote_point h
  let F₂ := right_focus h
  let F₁ := (-F₂.1, F₂.2)
  (M.1 - F₁.1) * (F₂.1 - M.1) + (M.2 - F₁.2) * (F₂.2 - M.2) > 0 →
  eccentricity h > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bound_l1255_125545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_area_ratio_l1255_125591

/-- A rectangular piece of paper with length twice its width -/
structure Paper where
  width : ℝ
  length : ℝ
  length_eq_twice_width : length = 2 * width

/-- The area of the paper -/
noncomputable def area (p : Paper) : ℝ := p.width * p.length

/-- The area of the paper after folding along the diagonal -/
noncomputable def folded_area (p : Paper) : ℝ := (1 / 2) * p.width * p.length

/-- The ratio of the folded area to the original area -/
noncomputable def area_ratio (p : Paper) : ℝ := folded_area p / area p

/-- Theorem stating that the ratio of folded area to original area is 1/2 -/
theorem folded_area_ratio (p : Paper) : area_ratio p = 1 / 2 := by
  sorry

#check folded_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_area_ratio_l1255_125591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1255_125520

def f (a b : ℝ) : ℝ → ℝ := λ x => a * x^2 + (b - 2) * x + 3

theorem min_value_theorem (a b : ℝ) (h1 : a ≠ 0) (h2 : f a b 1 = 2) (h3 : a > 0) (h4 : b > 0) :
  ∃ m : ℝ, m = 9 ∧ ∀ a' b' : ℝ, a' > 0 → b' > 0 → 1 / a' + 4 / b' ≥ m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1255_125520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relation_l1255_125504

theorem condition_relation : 
  (∀ x : ℝ, abs (x - 2) < 1 → x^2 + x - 2 > 0) ∧ 
  (∃ x : ℝ, x^2 + x - 2 > 0 ∧ abs (x - 2) ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relation_l1255_125504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l1255_125584

/-- Calculates the average speed of a car journey given the speeds for each hour -/
theorem average_speed_calculation (speeds : Fin 5 → ℝ) 
  (h1 : speeds 0 = 70)
  (h2 : speeds 1 = 90)
  (h3 : speeds 2 = 80)
  (h4 : speeds 3 = 100)
  (h5 : speeds 4 = 60) :
  (speeds 0 + speeds 1 + speeds 2 + speeds 3 + speeds 4) / 5 = 80 := by
  sorry

#check average_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_calculation_l1255_125584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_f_1200_l1255_125527

/-- A function satisfying the given property for all positive real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y

/-- The main theorem statement -/
theorem find_f_1200 (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 800 = 3) :
  f 1200 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_f_1200_l1255_125527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_ratio_l1255_125519

theorem rectangle_triangle_area_ratio :
  ∀ (L W : ℝ), L > 0 → W > 0 →
  (L * W) / ((1 / 2) * L * W) = 2 := by
  intro L W hL hW
  field_simp
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_ratio_l1255_125519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_branch_trajectory_l1255_125503

/-- The definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The definition of one branch of a hyperbola -/
def is_hyperbola_branch (S : Set Point) (F1 F2 : Point) (a : ℝ) : Prop :=
  ∀ P ∈ S, distance P F1 - distance P F2 = 2*a ∧ distance F1 F2 > 2*a

/-- The statement to be proved -/
theorem hyperbola_branch_trajectory :
  let F1 : Point := ⟨-3, 0⟩
  let F2 : Point := ⟨3, 0⟩
  let S : Set Point := {P | distance P F1 - distance P F2 = 4}
  is_hyperbola_branch S F1 F2 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_branch_trajectory_l1255_125503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_conditions_l1255_125548

noncomputable def f (x : ℝ) := Real.sin (4 * x + 5 * Real.pi / 6)

theorem function_satisfies_conditions :
  -- Point A(π/24, 0) lies on the graph of f(x)
  f (Real.pi / 24) = 0 ∧
  -- ω > 0 (implicitly satisfied by 4 > 0)
  -- 0 < φ < π (implicitly satisfied by 5π/6)
  -- The line x = π/6 is a symmetry axis of the graph of f(x)
  (∀ x : ℝ, f (Real.pi / 3 - x) = f x) ∧
  -- f(x) is monotonically increasing on the interval (π/6, π/3)
  (∀ x y : ℝ, Real.pi / 6 < x ∧ x < y ∧ y < Real.pi / 3 → f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_conditions_l1255_125548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_g_l1255_125540

/-- Determinant of a 2x2 matrix -/
def det2 (a1 a2 a3 a4 : ℝ) : ℝ := a1 * a4 - a2 * a3

/-- Original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := det2 (Real.sin x) (Real.cos x) 1 (Real.sqrt 3)

/-- Translated function g(x) -/
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x - φ)

/-- A function is odd if f(-x) = -f(x) for all x -/
def isOdd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem min_phi_for_odd_g :
  ∃ φ : ℝ, φ > 0 ∧ isOdd (g φ) ∧ ∀ ψ, (0 < ψ ∧ ψ < φ) → ¬isOdd (g ψ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_odd_g_l1255_125540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1255_125541

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/6) (Real.pi/3) → f x ∈ Set.Icc 1 4) ∧
  (∀ (y : ℝ), y ∈ Set.Icc 1 4 → ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/6) (Real.pi/3) ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1255_125541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1255_125510

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := x^2 - x + Real.sqrt (2*x^4 - 6*x^2 + 8*x + 16)

/-- Theorem stating that the minimum value of f(x) is 4 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1255_125510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_one_after_ten_operations_l1255_125528

/-- The operation performed on a positive integer -/
def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 1

/-- The number of times the operation is applied -/
def num_operations : ℕ := 10

/-- The sequence of numbers that reach 1 after exactly n operations -/
def reach_one_sequence : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => reach_one_sequence n + reach_one_sequence (n + 1)

theorem reach_one_after_ten_operations :
  reach_one_sequence num_operations = 55 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reach_one_after_ten_operations_l1255_125528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l1255_125581

theorem min_distance_to_line (x y : ℝ) (h : 5 * x + 12 * y = 60) :
  ∃ (min_val : ℝ), min_val = 60 / 13 ∧ Real.sqrt (x^2 + y^2) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l1255_125581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_congruent_triangles_l1255_125538

/-- Given an equilateral triangle PQR with side length √123 and four congruent triangles
    PS₁T₁, PS₁T₂, PS₂T₃, and PS₂T₄ where QS₁ = QS₂ = √13, prove that the sum of the
    squares of PT₁, PT₂, PT₃, and PT₄ is equal to 178. -/
theorem sum_of_squares_congruent_triangles :
  ∀ (P Q R S₁ S₂ T₁ T₂ T₃ T₄ : ℝ × ℝ),
  let d := λ (a b : ℝ × ℝ) => Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  -- PQR is an equilateral triangle with side length √123
  d P Q = Real.sqrt 123 ∧ d Q R = Real.sqrt 123 ∧ d R P = Real.sqrt 123 →
  -- PS₁T₁, PS₁T₂, PS₂T₃, and PS₂T₄ are congruent to PQR
  d P S₁ = d P Q ∧ d S₁ T₁ = d P Q ∧ d P T₁ = d P Q ∧
  d P S₁ = d P Q ∧ d S₁ T₂ = d P Q ∧ d P T₂ = d P Q ∧
  d P S₂ = d P Q ∧ d S₂ T₃ = d P Q ∧ d P T₃ = d P Q ∧
  d P S₂ = d P Q ∧ d S₂ T₄ = d P Q ∧ d P T₄ = d P Q →
  -- QS₁ = QS₂ = √13
  d Q S₁ = Real.sqrt 13 ∧ d Q S₂ = Real.sqrt 13 →
  -- The sum of squares of PT₁, PT₂, PT₃, and PT₄ is 178
  (d P T₁)^2 + (d P T₂)^2 + (d P T₃)^2 + (d P T₄)^2 = 178 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_congruent_triangles_l1255_125538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1255_125530

/-- A parabola with focus F -/
structure Parabola where
  F : ℝ × ℝ

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  onParabola : True  -- We assume this point satisfies the parabola equation

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that for a point P(3, m) on the parabola, |PF| = 4 -/
theorem parabola_focus_distance (p : Parabola) (P : PointOnParabola p) 
    (h : P.x = 3) : distance (P.x, P.y) p.F = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1255_125530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_increase_equivalence_l1255_125561

noncomputable def price_increase (original_price : ℝ) (percentage : ℝ) : ℝ :=
  original_price * (1 + percentage / 100)

theorem triple_increase_equivalence (original_price : ℝ) (x : ℝ) :
  let first_increase := price_increase original_price 10
  let second_increase := price_increase first_increase 10
  let final_increase := price_increase second_increase x
  final_increase = original_price * (1 + (21 + 1.21 * x) / 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_increase_equivalence_l1255_125561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_ploughing_problem_l1255_125588

/-- The number of hectares the farmer must plough per day to finish on time -/
noncomputable def hectares_per_day (total_area planned_days actual_days hectares_per_day_actual hectares_left : ℝ) : ℝ :=
  total_area / (planned_days + (actual_days - planned_days))

theorem farmer_ploughing_problem (total_area planned_days actual_days hectares_per_day_actual hectares_left : ℝ) :
  total_area = 1400 ∧ 
  hectares_per_day_actual = 85 ∧ 
  actual_days = planned_days + 2 ∧ 
  hectares_left = 40 →
  hectares_per_day total_area planned_days actual_days hectares_per_day_actual hectares_left = 100 := by
  sorry

-- Remove the #eval line as it's causing issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_ploughing_problem_l1255_125588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_transformation_l1255_125583

theorem cubic_root_transformation (p1 p2 p3 : ℂ) : 
  (p1^3 - 4*p1^2 + 5*p1 - 3 = 0) ∧ 
  (p2^3 - 4*p2^2 + 5*p2 - 3 = 0) ∧ 
  (p3^3 - 4*p3^2 + 5*p3 - 3 = 0) → 
  ((3*p1)^3 - 12*(3*p1)^2 + 45*(3*p1) - 81 = 0) ∧
  ((3*p2)^3 - 12*(3*p2)^2 + 45*(3*p2) - 81 = 0) ∧
  ((3*p3)^3 - 12*(3*p3)^2 + 45*(3*p3) - 81 = 0) := by
  intro h
  sorry

#check cubic_root_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_transformation_l1255_125583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_unplayed_value_l1255_125569

/-- Represents a team in the knockout match -/
structure Team :=
  (members : Nat)

/-- Represents the state of the match -/
structure MatchState :=
  (teamA : Team)
  (teamB : Team)
  (unplayed_A : Nat)

/-- Probability of a team winning a single match -/
noncomputable def win_prob : ℝ := 1 / 2

/-- Expected number of unplayed members from Team A -/
noncomputable def expected_unplayed (initial_state : MatchState) : ℝ :=
  sorry

/-- Main theorem: Expected number of unplayed Team A members is 187/256 -/
theorem expected_unplayed_value :
  expected_unplayed { teamA := { members := 5 },
                      teamB := { members := 5 },
                      unplayed_A := 5 } = 187 / 256 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_unplayed_value_l1255_125569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fractional_energy_transfer_l1255_125555

/-- Represents the fractional energy transfer in a perfectly elastic collision -/
noncomputable def fractionalEnergyTransfer (m M v₀ : ℝ) : ℝ :=
  (4 * m * M) / ((m + M)^2)

/-- Theorem stating that the fractional energy transfer is maximum when m = M -/
theorem max_fractional_energy_transfer
  (m M v₀ : ℝ)
  (hm : m > 0)
  (hM : M > 0)
  (hv₀ : v₀ ≠ 0) :
  ∀ x > 0, fractionalEnergyTransfer m M v₀ ≥ fractionalEnergyTransfer x M v₀ :=
by
  sorry

#check max_fractional_energy_transfer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fractional_energy_transfer_l1255_125555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_points_inside_unit_circle_l1255_125566

theorem integer_points_inside_unit_circle (θ : ℝ) : 
  (∃! p : ℤ × ℤ, (p.fst : ℝ)^2 + (p.snd : ℝ)^2 < 1) ∧ 
  (∀ x y : ℝ, x * Real.cos θ + y * Real.sin θ = 1 ↔ x^2 + y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_points_inside_unit_circle_l1255_125566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_is_prime_l1255_125579

def first_ten_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

def sum_is_prime (p q : Nat) : Bool :=
  is_prime (p + q)

def valid_pairs : List (Nat × Nat) :=
  List.filter (fun pair => sum_is_prime pair.1 pair.2)
    (List.filter (fun pair => pair.1 < pair.2)
      (List.product first_ten_primes first_ten_primes))

theorem probability_sum_is_prime :
  (List.length valid_pairs : Rat) / (Nat.choose 10 2 : Rat) = 1 / 9 := by
  sorry

#eval valid_pairs
#eval List.length valid_pairs
#eval Nat.choose 10 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_is_prime_l1255_125579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1255_125505

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1)^2 * (Real.exp x - a)

-- State the theorem
theorem f_properties :
  (∀ x y : ℝ, x < y → f (Real.exp 1) x < f (Real.exp 1) y) ∧
  (∃ x : ℝ, f (1/2) x = 1/2 ∧ 
    (∀ y : ℝ, f (1/2) y ≤ f (1/2) x ∨ f (1/2) y ≥ f (1/2) x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1255_125505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_cube_root_l1255_125568

theorem rationalize_denominator_cube_root : (7 : ℝ) / (343 : ℝ)^(1/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_cube_root_l1255_125568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_subsequence_l1255_125565

def seq (n : ℕ) : ℤ := 2^n - 3

theorem infinite_coprime_subsequence :
  ∃ (f : ℕ → ℕ), (∀ i j, i < j → f i < f j) ∧
    (∀ i j, i ≠ j → Int.gcd (seq (f i)) (seq (f j)) = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_subsequence_l1255_125565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crude_oil_mixture_theorem_l1255_125515

/-- Represents the composition of a crude oil mixture -/
structure CrudeOilMixture where
  total : ℝ
  hydrocarbon_percent : ℝ

/-- Calculates the amount of hydrocarbons in a crude oil mixture -/
noncomputable def hydrocarbons (m : CrudeOilMixture) : ℝ :=
  m.total * m.hydrocarbon_percent / 100

theorem crude_oil_mixture_theorem
  (source1 : CrudeOilMixture)
  (source2 : CrudeOilMixture)
  (target : CrudeOilMixture)
  (h1 : source1.hydrocarbon_percent = 25)
  (h2 : source2.hydrocarbon_percent = 75)
  (h3 : target.total = 50)
  (h4 : target.hydrocarbon_percent = 55)
  (h5 : source1.total + source2.total = target.total)
  (h6 : hydrocarbons source1 + hydrocarbons source2 = hydrocarbons target) :
  source2.total = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crude_oil_mixture_theorem_l1255_125515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_water_timing_l1255_125571

noncomputable def hot_faucet_time : ℝ := 23
noncomputable def cold_faucet_time : ℝ := 17

noncomputable def hot_faucet_rate : ℝ := 1 / hot_faucet_time
noncomputable def cold_faucet_rate : ℝ := 1 / cold_faucet_time

noncomputable def half_tub : ℝ := 1 / 2

theorem equal_water_timing :
  let hot_time := half_tub / hot_faucet_rate
  let cold_time := half_tub / cold_faucet_rate
  hot_time - cold_time = 3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_water_timing_l1255_125571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventuallyStable_l1255_125587

/-- Represents the state of a person's vote -/
inductive Vote
  | Yes
  | No
deriving Repr, DecidableEq

/-- Represents the circular arrangement of people and their votes -/
def Circle := Fin 25 → Vote

/-- Updates a single person's vote based on their neighbors -/
def updateVote (circle : Circle) (i : Fin 25) : Vote :=
  let left := circle ((i - 1) % 25)
  let right := circle ((i + 1) % 25)
  let current := circle i
  if current = left ∨ current = right then current else
    match current with
    | Vote.Yes => Vote.No
    | Vote.No => Vote.Yes

/-- Updates the entire circle's votes -/
def updateCircle (circle : Circle) : Circle :=
  λ i => updateVote circle i

/-- Checks if the circle has reached a stable state -/
def isStable (circle : Circle) : Prop :=
  ∀ i, updateVote circle i = circle i

/-- The main theorem stating that the system will eventually stabilize -/
theorem eventuallyStable (initial : Circle) : ∃ n : Nat, isStable (Nat.iterate updateCircle n initial) := by
  sorry

#check eventuallyStable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventuallyStable_l1255_125587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_funding_adjustment_verify_total_earnings_l1255_125576

/-- Calculates the required weekly hours given the original plan and missed weeks --/
noncomputable def calculate_new_weekly_hours (original_hours : ℝ) (original_weeks : ℕ) 
  (total_earnings : ℝ) (missed_weeks : ℕ) : ℝ :=
  let remaining_weeks := original_weeks - missed_weeks
  let hourly_rate := total_earnings / (original_hours * original_weeks)
  total_earnings / (hourly_rate * remaining_weeks)

/-- Theorem stating the required weekly hours for the given scenario --/
theorem project_funding_adjustment :
  let original_hours : ℝ := 25
  let original_weeks : ℕ := 15
  let total_earnings : ℝ := 3750
  let missed_weeks : ℕ := 3
  calculate_new_weekly_hours original_hours original_weeks total_earnings missed_weeks = 31.25 := by
  sorry

/-- Verifies that the calculated hours yield the required total earnings --/
theorem verify_total_earnings :
  let original_hours : ℝ := 25
  let original_weeks : ℕ := 15
  let total_earnings : ℝ := 3750
  let missed_weeks : ℕ := 3
  let new_hours := calculate_new_weekly_hours original_hours original_weeks total_earnings missed_weeks
  new_hours * (original_weeks - missed_weeks) * (total_earnings / (original_hours * original_weeks)) = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_funding_adjustment_verify_total_earnings_l1255_125576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_gnomes_on_pentagonal_pyramid_l1255_125535

/-- A pentagonal pyramid -/
structure PentagonalPyramid where
  faces : Finset (Fin 6)
  gnomes : Finset ℕ
  gnome_location : ℕ → Fin 6

/-- The number of gnomes on a face -/
def gnomes_on_face (p : PentagonalPyramid) (f : Fin 6) : ℕ :=
  (p.gnomes.filter (λ g => p.gnome_location g = f)).card

/-- Theorem: The minimum number of gnomes on a pentagonal pyramid is 6 -/
theorem min_gnomes_on_pentagonal_pyramid (p : PentagonalPyramid) :
  (p.faces = Finset.univ) →
  (∀ g ∈ p.gnomes, p.gnome_location g ∈ p.faces) →
  (∀ f₁ f₂ : Fin 6, f₁ ∈ p.faces → f₂ ∈ p.faces → f₁ ≠ f₂ → gnomes_on_face p f₁ ≠ gnomes_on_face p f₂) →
  p.gnomes.card ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_gnomes_on_pentagonal_pyramid_l1255_125535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_base_radii_l1255_125531

theorem truncated_cone_base_radii (α l : ℝ) (h_α : 0 < α ∧ α < π) (h_l : l > 0) :
  ∃ (R r : ℝ),
    R = l * (Real.cos (α / 2))^3 / Real.sin (α / 2) ∧
    r = l * (Real.sin (α / 2))^3 / Real.cos (α / 2) ∧
    (∃ (x y : ℝ),
      x > 0 ∧ y > 0 ∧ x + y = l ∧ x - y = l * Real.cos α ∧
      R = x / Real.tan (α / 2) ∧ r = y * Real.tan (α / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_base_radii_l1255_125531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1255_125513

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) + 2

theorem f_properties :
  let period : ℝ := Real.pi
  let max_value : ℝ := 2 + Real.sqrt 2
  let min_value : ℝ := 1
  let interval : Set ℝ := Set.Icc 0 (Real.pi / 2)
  (∀ x : ℝ, f (x + period) = f x) ∧
  (∃ x ∈ interval, f x = max_value) ∧
  (∃ x ∈ interval, f x = min_value) ∧
  (∀ x ∈ interval, min_value ≤ f x ∧ f x ≤ max_value) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1255_125513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_a_leave_time_is_correct_l1255_125502

/-- The time Person A leaves, in minutes after 8 AM -/
noncomputable def person_a_leave_time : ℝ :=
  let total_time : ℝ := 4 * 60 -- 4 hours in minutes
  let work_rate_a : ℝ := 1 / 6
  let work_rate_b : ℝ := 1 / 8
  let work_rate_c : ℝ := 1 / 10
  let work_rate_bc : ℝ := work_rate_b + work_rate_c
  let work_done_bc : ℝ := work_rate_bc * total_time
  let work_done_a : ℝ := 1 - work_done_bc
  work_done_a / work_rate_a

theorem person_a_leave_time_is_correct :
  person_a_leave_time = 36 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_a_leave_time_is_correct_l1255_125502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_relationship_in_triangle_l1255_125570

theorem cosine_sine_relationship_in_triangle (A B C : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) : 
  (Real.cos A < Real.cos B) ↔ (Real.sin A > Real.sin B) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_relationship_in_triangle_l1255_125570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1255_125511

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi/6) * Real.sin (x + Real.pi/3)

/-- The statement that π is the smallest positive period of f(x) -/
theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1255_125511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_240924_l1255_125551

theorem sum_of_distinct_prime_factors_240924 :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (240924 + 1)))
    (fun p => if p ∣ 240924 then p else 0)) = 34 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_prime_factors_240924_l1255_125551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equality_characterization_l1255_125537

noncomputable def M (k : ℕ) : ℕ := Finset.range k.succ |>.toList.foldl Nat.lcm 1

theorem M_equality_characterization (n : ℕ) (h : n > 0) :
  M (n - 1) = M n ↔ ¬∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = p ^ k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equality_characterization_l1255_125537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_center_line_l1255_125522

/-- Two circles with centers O₁ and O₂ -/
structure TwoCircles where
  O₁ : EuclideanSpace ℝ (Fin 2)
  O₂ : EuclideanSpace ℝ (Fin 2)
  r₁ : ℝ
  r₂ : ℝ
  outside : ‖O₁ - O₂‖ > r₁ + r₂

/-- Four points on the circles forming tangent lines -/
structure TangentPoints (circles : TwoCircles) where
  K : EuclideanSpace ℝ (Fin 2)
  L : EuclideanSpace ℝ (Fin 2)
  M : EuclideanSpace ℝ (Fin 2)
  N : EuclideanSpace ℝ (Fin 2)
  on_circle₁ : (‖circles.O₁ - K‖ = circles.r₁) ∧ (‖circles.O₁ - M‖ = circles.r₁)
  on_circle₂ : (‖circles.O₂ - L‖ = circles.r₂) ∧ (‖circles.O₂ - N‖ = circles.r₂)
  external_tangent : sorry -- Placeholder for external tangent condition
  internal_tangent : sorry -- Placeholder for internal tangent condition

/-- The main theorem -/
theorem intersection_on_center_line (circles : TwoCircles) (points : TangentPoints circles) :
  ∃ P : EuclideanSpace ℝ (Fin 2), 
    (∃ t : ℝ, P = (1 - t) • points.K + t • points.M) ∧ 
    (∃ s : ℝ, P = (1 - s) • points.L + s • points.N) ∧
    (∃ r : ℝ, P = (1 - r) • circles.O₁ + r • circles.O₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_center_line_l1255_125522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_equals_neg_one_l1255_125508

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define m as a real number
def m : ℝ := sorry

-- Axioms representing the conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom period_property : ∀ x, f (x + 1) = f (1 - x)
axiom interval_definition : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2^x - m

-- Theorem to prove
theorem f_2019_equals_neg_one : f 2019 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_equals_neg_one_l1255_125508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_fixed_point_value_l1255_125596

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := (2 * (x + 3)) / 3 + 9

-- Theorem statement
theorem unique_fixed_point_of_h :
  ∃! x : ℝ, h x = x :=
by
  -- The proof goes here
  sorry

-- Theorem for the specific value of the fixed point
theorem fixed_point_value :
  ∀ x : ℝ, h x = x → x = 33 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_of_h_fixed_point_value_l1255_125596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1255_125567

/-- Represents a point in 2D space -/
structure Point where
  x : Real
  y : Real

/-- Represents a curvilinear triangle formed by a quarter circle and two straight lines -/
structure CurvilinearTriangle where
  center : Point
  radius : Real

/-- Represents an arc -/
structure Arc where
  center : Point
  passThrough : Point

/-- Calculates the area of a curvilinear triangle -/
noncomputable def areaOfCurvilinearTriangle (ct : CurvilinearTriangle) : Real :=
  (Real.pi / 4) - (1 / 2)

/-- Checks if an arc divides a curvilinear triangle into two equal areas -/
def dividesEquallyByArc (ct : CurvilinearTriangle) (arc : Arc) : Prop :=
  ∃ (area1 area2 : Real),
    area1 = area2 ∧
    area1 + area2 = areaOfCurvilinearTriangle ct

/-- Theorem: An arc centered at one marked point and passing through the other
    divides the curvilinear triangle into two parts of equal area -/
theorem equal_area_division (ct : CurvilinearTriangle) (p1 p2 : Point) :
  (p1 = ct.center ∨ p2 = ct.center) →
  dividesEquallyByArc ct (Arc.mk p1 p2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1255_125567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_num_small_chips_l1255_125546

/-- The total number of chips in the box -/
def total_chips : ℕ := 100

/-- Represents the number of small chips -/
def num_small_chips : ℕ → Prop := λ _ => True

/-- Represents the number of large chips -/
def num_large_chips : ℕ → Prop := λ _ => True

/-- The difference between small and large chips is prime -/
def diff_is_prime (s l : ℕ) : Prop := Nat.Prime (s - l)

/-- The theorem stating the smallest possible number of small chips -/
theorem smallest_num_small_chips :
  ∃ (s l : ℕ),
    s + l = total_chips ∧
    num_small_chips s ∧
    num_large_chips l ∧
    diff_is_prime s l ∧
    s = 51 ∧
    ∀ (s' l' : ℕ),
      s' + l' = total_chips →
      num_small_chips s' →
      num_large_chips l' →
      diff_is_prime s' l' →
      s' ≥ s :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_num_small_chips_l1255_125546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l1255_125523

/-- Given that the solution set of x² + ax + b > 0 is (-∞, -2) ∪ (-1/2, +∞),
    prove that the solution set of bx² + ax + 1 < 0 is (-2, -1/2) -/
theorem quadratic_inequality_solution_sets
  (a b : ℝ)
  (h : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ x < -2 ∨ x > -1/2) :
  ∀ x : ℝ, b * x^2 + a * x + 1 < 0 ↔ -2 < x ∧ x < -1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_sets_l1255_125523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_2002_l1255_125550

/-- A function satisfying the given recursive property -/
def special_function (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 1 →
    ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

theorem special_function_2002 (f : ℕ → ℝ) 
  (h : special_function f) (h2001 : f 2001 = 1) : f 2002 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_2002_l1255_125550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_not_two_l1255_125560

theorem sin_plus_cos_not_two : ¬ ∃ x : ℝ, Real.sin x + Real.cos x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_not_two_l1255_125560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_points_eq_target_segment_l1255_125506

-- Define the circles and points
def Circle (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ) := 
  {x : EuclideanSpace ℝ (Fin 2) | ‖x - center‖ = radius}

variable (P B : EuclideanSpace ℝ (Fin 2))
variable (r : ℝ)

def Γ := Circle P r
def Δ := Circle B (r/2)

-- Define the set of points A satisfying the condition
def SatisfyingPoints := 
  {A : EuclideanSpace ℝ (Fin 2) | ∀ X, X ∈ Γ P r ∨ X ∈ Δ B r → ‖A - B‖ ≤ ‖A - X‖}

-- Define the segment from P to the midpoint of PB
def TargetSegment := 
  {A : EuclideanSpace ℝ (Fin 2) | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1/2 ∧ A = P + t • (B - P)}

-- State the theorem
theorem satisfying_points_eq_target_segment : 
  SatisfyingPoints P B r = TargetSegment P B := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_points_eq_target_segment_l1255_125506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_l1255_125512

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) * Real.cos x

-- Define the derivative of the function
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 
  Real.exp (a * x) * (a * Real.cos x - Real.sin x)

-- Theorem statement
theorem tangent_perpendicular (a : ℝ) : 
  (f_derivative a 0 * (-1/2) = -1) ↔ (a = 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_l1255_125512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_quotient_implies_all_l1255_125536

/-- If (x^n - y^n) / (x - y) is an integer for four consecutive positive integers n,
    then it is an integer for all positive integers n. -/
theorem consecutive_integer_quotient_implies_all (x y : ℝ) :
  (∃ k : ℕ, ∀ n : ℕ, n ∈ Finset.range 4 → ∃ m : ℤ, (x^(k+n) - y^(k+n)) / (x - y) = m) →
  (∀ n : ℕ, ∃ m : ℤ, (x^n - y^n) / (x - y) = m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integer_quotient_implies_all_l1255_125536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l1255_125577

noncomputable section

/-- Two-dimensional vector -/
def Vector2D := ℝ × ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Magnitude (length) of a 2D vector -/
noncomputable def magnitude (v : Vector2D) : ℝ := Real.sqrt (dot_product v v)

/-- The angle between two vectors in radians -/
noncomputable def angle (v w : Vector2D) : ℝ := Real.arccos ((dot_product v w) / (magnitude v * magnitude w))

theorem vector_difference_magnitude 
  (a b : Vector2D) 
  (h1 : a = (1, 0)) 
  (h2 : magnitude b = Real.sqrt 3) 
  (h3 : angle a b = π / 6) : 
  magnitude (a.1 - b.1, a.2 - b.2) = 1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_difference_magnitude_l1255_125577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_max_value_l1255_125559

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

-- Define the interval
def interval : Set ℝ := Set.Icc (-3/4) (1/4)

-- Theorem for the minimum value
theorem f_min_value :
  ∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = Real.log 2 + 1/4 := by sorry

-- Theorem for the maximum value
theorem f_max_value :
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 1/16 + Real.log (7/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_max_value_l1255_125559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_sum_l1255_125580

theorem recurring_decimal_sum : 
  (∃ (x y : ℚ), x = 2/9 ∧ y = 1/33) → 
  2/9 + 1/33 = 25/99 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_sum_l1255_125580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1255_125518

noncomputable def g (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 4 * Real.sin x ^ 2 + 2 * Real.sin x + 3 * Real.cos x ^ 2 - 9) / (Real.sin x - 2)

theorem g_range : 
  ∀ y : ℝ, (∃ x : ℝ, g x = y) ↔ 0 ≤ y ∧ y ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1255_125518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roaming_area_difference_l1255_125501

/-- Represents the dimensions of the barn and the rope length -/
structure BarnSetup where
  length : ℝ
  width : ℝ
  rope_length : ℝ

/-- Calculates the roaming area for Case 1 -/
noncomputable def case1_area (setup : BarnSetup) : ℝ :=
  0.5 * Real.pi * setup.rope_length^2

/-- Calculates the roaming area for Case 2 -/
noncomputable def case2_area (setup : BarnSetup) : ℝ :=
  0.75 * Real.pi * setup.rope_length^2 + 0.25 * Real.pi * 3^2

/-- Theorem stating the difference in roaming area between Case 2 and Case 1 -/
theorem roaming_area_difference (setup : BarnSetup) 
  (h1 : setup.length = 20)
  (h2 : setup.width = 10)
  (h3 : setup.rope_length = 10) : 
  case2_area setup - case1_area setup = 27.25 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roaming_area_difference_l1255_125501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_perfect_squares_div_by_five_l1255_125595

theorem three_digit_perfect_squares_div_by_five :
  ∃! (n : ℕ), n = 5 ∧
    (∀ x : ℕ, (100 ≤ x ∧ x ≤ 999) ∧  -- Three-digit number
               (∃ y : ℕ, x = y * y) ∧  -- Perfect square
               (x % 5 = 0) →           -- Divisible by 5
     x ∈ ({100, 225, 400, 625, 900} : Set ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_perfect_squares_div_by_five_l1255_125595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_is_floor_function_l1255_125589

noncomputable def floor_function (x : ℝ) : ℤ := Int.floor x

theorem function_is_floor_function
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f x + f y + 1 ≥ f (x + y) ∧ f (x + y) ≥ f x + f y)
  (h2 : ∀ x : ℝ, 0 ≤ x → x < 1 → f 0 ≥ f x)
  (h3 : -f (-1) = f 1 ∧ f 1 = 1) :
  ∀ x : ℝ, f x = ↑(floor_function x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_is_floor_function_l1255_125589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_range_specific_angles_l1255_125507

structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

def altitude_tangent (t : Triangle) (vertex : Nat) : Prop :=
  match vertex with
  | 0 => t.B + 2 * t.C = 90
  | 1 => t.C + 2 * t.A = 90
  | _ => t.A + 2 * t.B = 90

theorem largest_angle_range (t : Triangle) (h : altitude_tangent t 1) :
  90 ≤ max t.A (max t.B t.C) ∧ max t.A (max t.B t.C) < 135 := by
  sorry

theorem specific_angles (t : Triangle) (h1 : altitude_tangent t 1) (h2 : altitude_tangent t 2) :
  t.A = 120 ∧ t.B = 30 ∧ t.C = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_range_specific_angles_l1255_125507
