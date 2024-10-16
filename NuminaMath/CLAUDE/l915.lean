import Mathlib

namespace NUMINAMATH_CALUDE_nathaniel_win_probability_l915_91583

-- Define the game state
structure GameState where
  tally : ℕ
  currentPlayer : Bool  -- True for Nathaniel, False for Obediah

-- Define the probability of winning for a given game state
def winProbability (state : GameState) : ℚ :=
  sorry

-- Define the theorem
theorem nathaniel_win_probability :
  winProbability ⟨0, true⟩ = 5/11 := by sorry

end NUMINAMATH_CALUDE_nathaniel_win_probability_l915_91583


namespace NUMINAMATH_CALUDE_comparison_arithmetic_geometric_mean_l915_91571

theorem comparison_arithmetic_geometric_mean (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ¬(∀ a b c, (a + b + c) / 3 ≥ (a^2 * b * b * c * c * a)^(1/3)) ∧ 
  ¬(∀ a b c, (a + b + c) / 3 ≤ (a^2 * b * b * c * c * a)^(1/3)) ∧ 
  ¬(∀ a b c, (a + b + c) / 3 = (a^2 * b * b * c * c * a)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_comparison_arithmetic_geometric_mean_l915_91571


namespace NUMINAMATH_CALUDE_abc_sum_root_l915_91515

theorem abc_sum_root (a b c : ℝ) 
  (h1 : b + c = 7) 
  (h2 : c + a = 8) 
  (h3 : a + b = 9) : 
  Real.sqrt (a * b * c * (a + b + c)) = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_root_l915_91515


namespace NUMINAMATH_CALUDE_f_even_implies_constant_or_increasing_l915_91542

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m^2 - 1) * x + 1

theorem f_even_implies_constant_or_increasing (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  (∀ x ≤ 0, f m x = 1 ∨ (∀ y z, y < z ∧ z ≤ 0 → f m y < f m z)) :=
by sorry

end NUMINAMATH_CALUDE_f_even_implies_constant_or_increasing_l915_91542


namespace NUMINAMATH_CALUDE_train_length_l915_91579

/-- Calculates the length of a train given its speed and the time and distance it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (5/18) → 
  platform_length = 250 → 
  crossing_time = 30 → 
  (train_speed * crossing_time) - platform_length = 350 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l915_91579


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l915_91512

theorem sum_of_fractions_equals_two_ninths :
  let sum := (1 : ℚ) / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)
  sum = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l915_91512


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l915_91597

theorem geometric_sequence_fourth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 1024) 
  (h2 : a * r^5 = 32) : 
  a * r^3 = 128 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l915_91597


namespace NUMINAMATH_CALUDE_weight_of_b_l915_91546

/-- Given three weights a, b, and c, prove that b equals 31 when:
    1. The average of a, b, and c is 45.
    2. The average of a and b is 40.
    3. The average of b and c is 43. -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) : 
  b = 31 := by
  sorry


end NUMINAMATH_CALUDE_weight_of_b_l915_91546


namespace NUMINAMATH_CALUDE_polygon_diagonals_sides_l915_91510

theorem polygon_diagonals_sides (n : ℕ) (h : n = 8) : (n * (n - 3)) / 2 = 2 * n + 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_sides_l915_91510


namespace NUMINAMATH_CALUDE_one_third_of_six_to_thirty_l915_91544

theorem one_third_of_six_to_thirty (x : ℚ) :
  x = (1 / 3) * (6 ^ 30) → x = 2 * (6 ^ 29) := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_six_to_thirty_l915_91544


namespace NUMINAMATH_CALUDE_largest_prime_factor_l915_91539

theorem largest_prime_factor (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (17^4 + 2 * 17^2 + 1 - 16^4) ∧
    ∀ q : ℕ, Nat.Prime q → q ∣ (17^4 + 2 * 17^2 + 1 - 16^4) → q ≤ p) →
  (∃ p : ℕ, p = 17 ∧ Nat.Prime p ∧ p ∣ (17^4 + 2 * 17^2 + 1 - 16^4) ∧
    ∀ q : ℕ, Nat.Prime q → q ∣ (17^4 + 2 * 17^2 + 1 - 16^4) → q ≤ p) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l915_91539


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_l915_91500

theorem triangle_isosceles_or_right 
  (A B C : ℝ) 
  (triangle_angles : A + B + C = π) 
  (angle_condition : Real.sin (A + B - C) = Real.sin (A - B + C)) : 
  (B = C) ∨ (B + C = π / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_l915_91500


namespace NUMINAMATH_CALUDE_largest_x_floor_div_l915_91549

theorem largest_x_floor_div : ∃ (x : ℝ), x = 63/8 ∧ 
  (∀ (y : ℝ), y > x → ⌊y⌋/y ≠ 8/9) ∧ ⌊x⌋/x = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_div_l915_91549


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l915_91527

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = -4/5 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l915_91527


namespace NUMINAMATH_CALUDE_seven_by_seven_checkerboard_shading_l915_91506

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a checkerboard shading pattern on a grid -/
def checkerboard_shading (g : Grid) : ℕ :=
  (g.size * g.size) / 2

/-- Calculates the percentage of shaded squares in a grid -/
def shaded_percentage (g : Grid) : ℚ :=
  (checkerboard_shading g : ℚ) / (g.size * g.size : ℚ) * 100

/-- Theorem: The percentage of shaded squares in a 7x7 checkerboard is 2400/49 -/
theorem seven_by_seven_checkerboard_shading :
  shaded_percentage { size := 7 } = 2400 / 49 := by
  sorry

end NUMINAMATH_CALUDE_seven_by_seven_checkerboard_shading_l915_91506


namespace NUMINAMATH_CALUDE_find_divisor_l915_91566

theorem find_divisor (N : ℝ) (D : ℝ) (h1 : (N - 6) / D = 2) (h2 : N = 32) : D = 13 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l915_91566


namespace NUMINAMATH_CALUDE_order_of_f_values_l915_91557

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

theorem order_of_f_values :
  let a := f (Real.sqrt 2 / 2)
  let b := f (Real.sqrt 3 / 2)
  let c := f (Real.sqrt 6 / 2)
  b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_order_of_f_values_l915_91557


namespace NUMINAMATH_CALUDE_subset_proof_l915_91520

def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {0, 1, 2}

theorem subset_proof : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_subset_proof_l915_91520


namespace NUMINAMATH_CALUDE_clearance_sale_prices_l915_91590

/-- Calculates the final price after applying two successive discounts --/
def finalPrice (initialPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initialPrice * (1 - discount1) * (1 - discount2)

/-- Proves that the final prices of the hat and gloves are correct --/
theorem clearance_sale_prices 
  (hatInitialPrice : ℝ) 
  (hatDiscount1 : ℝ) 
  (hatDiscount2 : ℝ)
  (glovesInitialPrice : ℝ) 
  (glovesDiscount1 : ℝ) 
  (glovesDiscount2 : ℝ)
  (hatInitialPrice_eq : hatInitialPrice = 15)
  (hatDiscount1_eq : hatDiscount1 = 0.20)
  (hatDiscount2_eq : hatDiscount2 = 0.40)
  (glovesInitialPrice_eq : glovesInitialPrice = 8)
  (glovesDiscount1_eq : glovesDiscount1 = 0.25)
  (glovesDiscount2_eq : glovesDiscount2 = 0.30) :
  finalPrice hatInitialPrice hatDiscount1 hatDiscount2 = 7.20 ∧
  finalPrice glovesInitialPrice glovesDiscount1 glovesDiscount2 = 4.20 := by
  sorry

#check clearance_sale_prices

end NUMINAMATH_CALUDE_clearance_sale_prices_l915_91590


namespace NUMINAMATH_CALUDE_current_speed_l915_91521

/-- Given a boat's upstream and downstream speeds, calculate the speed of the current. -/
theorem current_speed (upstream_time : ℝ) (downstream_time : ℝ) :
  upstream_time = 20 →
  downstream_time = 9 →
  let upstream_speed := 60 / upstream_time
  let downstream_speed := 60 / downstream_time
  abs ((downstream_speed - upstream_speed) / 2 - 1.835) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l915_91521


namespace NUMINAMATH_CALUDE_circle_with_AB_diameter_l915_91530

-- Define the points A and B
def A : ℝ × ℝ := (-3, -5)
def B : ℝ × ℝ := (5, 1)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 25

-- Theorem statement
theorem circle_with_AB_diameter :
  ∀ x y : ℝ,
  circle_equation x y ↔ 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    ((x = (1 - t) * A.1 + t * B.1) ∧
     (y = (1 - t) * A.2 + t * B.2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_with_AB_diameter_l915_91530


namespace NUMINAMATH_CALUDE_cone_base_circumference_l915_91594

/-- The circumference of the base of a right circular cone formed from a circular piece of paper 
    with radius 6 inches, after removing a 180° sector, is equal to 6π inches. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) : 
  (2 * π * r) * (1/2) = 6 * π := by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l915_91594


namespace NUMINAMATH_CALUDE_equation_solutions_l915_91564

theorem equation_solutions : 
  ∀ x : ℝ, x * (2 * x - 4) = 3 * (2 * x - 4) ↔ x = 3 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l915_91564


namespace NUMINAMATH_CALUDE_book_pages_theorem_l915_91529

theorem book_pages_theorem (total_pages : ℕ) : 
  (total_pages / 5 : ℚ) + 24 + (3/2 : ℚ) * ((total_pages / 5 : ℚ) + 24) = (3/4 : ℚ) * total_pages →
  total_pages = 240 := by
sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l915_91529


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_ABD_l915_91592

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a plane -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Represents a quadrilateral -/
structure Quadrilateral where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D

/-- Function to calculate the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Main theorem statement -/
theorem area_of_quadrilateral_ABD (cube : Cube) (plane : Plane) (quadABD : Quadrilateral) :
  cube.sideLength = 2 →
  -- A is a vertex of the cube
  -- B and D are midpoints of edges adjacent to A
  -- C' is the midpoint of a face diagonal not including A
  -- Plane passes through A, B, D, and C'
  -- quadABD lies in the plane
  areaQuadrilateral quadABD = 2 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_ABD_l915_91592


namespace NUMINAMATH_CALUDE_ladder_matches_l915_91537

/-- Represents the number of matches needed for a ladder with a given number of steps. -/
def matches_for_ladder (steps : ℕ) : ℕ :=
  6 * steps

theorem ladder_matches :
  matches_for_ladder 3 = 18 →
  matches_for_ladder 25 = 150 :=
by sorry

end NUMINAMATH_CALUDE_ladder_matches_l915_91537


namespace NUMINAMATH_CALUDE_john_average_score_l915_91541

def john_scores : List ℝ := [95, 88, 91, 87, 92, 90]

theorem john_average_score :
  (john_scores.sum / john_scores.length : ℝ) = 90.5 := by
  sorry

end NUMINAMATH_CALUDE_john_average_score_l915_91541


namespace NUMINAMATH_CALUDE_star_perimeter_calculation_l915_91550

/-- The perimeter of the original star can be calculated from its components. -/
theorem star_perimeter_calculation (n m : ℕ) (P : ℝ) : 
  n = 5 → 
  m = 5 → 
  m * 18 + 3 = n * 7 + P → 
  P = 58 := by
  sorry

end NUMINAMATH_CALUDE_star_perimeter_calculation_l915_91550


namespace NUMINAMATH_CALUDE_seashell_collection_l915_91593

theorem seashell_collection (x y : ℝ) : 
  let initial := x
  let additional := y
  let total := initial + additional
  let after_jessica := (2/3) * total
  let after_henry := (3/4) * after_jessica
  after_henry = (1/2) * total
  := by sorry

end NUMINAMATH_CALUDE_seashell_collection_l915_91593


namespace NUMINAMATH_CALUDE_robertos_salary_proof_l915_91536

theorem robertos_salary_proof (current_salary : ℝ) : 
  current_salary = 134400 →
  ∃ (starting_salary : ℝ),
    starting_salary = 80000 ∧
    current_salary = 1.2 * (1.4 * starting_salary) :=
by
  sorry

end NUMINAMATH_CALUDE_robertos_salary_proof_l915_91536


namespace NUMINAMATH_CALUDE_alexander_pencil_difference_alexander_pencil_difference_proof_l915_91576

/-- Proves that Alexander has 60 more pencils than Asaf given the problem conditions -/
theorem alexander_pencil_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun asaf_age alexander_age asaf_pencils alexander_pencils =>
    asaf_age = 50 ∧
    asaf_age + alexander_age = 140 ∧
    alexander_age - asaf_age = asaf_pencils / 2 ∧
    asaf_pencils + alexander_pencils = 220 →
    alexander_pencils - asaf_pencils = 60

/-- Proof of the theorem -/
theorem alexander_pencil_difference_proof :
  ∃ (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ),
    alexander_pencil_difference asaf_age alexander_age asaf_pencils alexander_pencils :=
by
  sorry

end NUMINAMATH_CALUDE_alexander_pencil_difference_alexander_pencil_difference_proof_l915_91576


namespace NUMINAMATH_CALUDE_construction_materials_cost_l915_91548

/-- The total cost of construction materials for Mr. Zander -/
def total_cost (cement_bags : ℕ) (cement_price : ℕ) (sand_lorries : ℕ) (sand_tons_per_lorry : ℕ) (sand_price : ℕ) : ℕ :=
  cement_bags * cement_price + sand_lorries * sand_tons_per_lorry * sand_price

/-- Theorem stating that the total cost of construction materials for Mr. Zander is $13,000 -/
theorem construction_materials_cost :
  total_cost 500 10 20 10 40 = 13000 := by
  sorry

end NUMINAMATH_CALUDE_construction_materials_cost_l915_91548


namespace NUMINAMATH_CALUDE_fraction_simplification_l915_91582

theorem fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - c * b) / (a * b - a^2) = (a^2 + a * b + b^2 + a * c) / (a * b) :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l915_91582


namespace NUMINAMATH_CALUDE_rectangle_rotation_path_length_l915_91501

/-- The length of the path traveled by point P of a rectangle PQRS after two 90° rotations -/
theorem rectangle_rotation_path_length (P Q R S : ℝ × ℝ) : 
  let pq : ℝ := 2
  let rs : ℝ := 2
  let qr : ℝ := 6
  let sp : ℝ := 6
  let first_rotation_radius : ℝ := Real.sqrt (pq^2 + qr^2)
  let first_rotation_arc_length : ℝ := (π / 2) * first_rotation_radius
  let second_rotation_radius : ℝ := sp
  let second_rotation_arc_length : ℝ := (π / 2) * second_rotation_radius
  let total_path_length : ℝ := first_rotation_arc_length + second_rotation_arc_length
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = pq^2 →
  (R.1 - S.1)^2 + (R.2 - S.2)^2 = rs^2 →
  (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = qr^2 →
  (S.1 - P.1)^2 + (S.2 - P.2)^2 = sp^2 →
  total_path_length = (3 + Real.sqrt 10) * π := by
sorry

end NUMINAMATH_CALUDE_rectangle_rotation_path_length_l915_91501


namespace NUMINAMATH_CALUDE_valid_numbers_l915_91540

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_valid_sequence (a bc de fg : ℕ) : Prop :=
  2 ∣ a ∧
  is_prime bc ∧
  5 ∣ de ∧
  3 ∣ fg ∧
  fg - de = de - bc ∧
  de - bc = bc - a

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a bc de fg : ℕ),
    is_valid_sequence a bc de fg ∧
    n = de * 100 + bc

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n = 2013 ∨ n = 4023 := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l915_91540


namespace NUMINAMATH_CALUDE_thomas_run_conversion_l915_91524

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 7^3 + d₂ * 7^2 + d₁ * 7^1 + d₀ * 7^0

/-- Thomas's run in base 7 -/
def thomasRunBase7 : ℕ × ℕ × ℕ × ℕ := (4, 2, 1, 3)

theorem thomas_run_conversion :
  let (d₃, d₂, d₁, d₀) := thomasRunBase7
  base7ToBase10 d₃ d₂ d₁ d₀ = 1480 := by
sorry

end NUMINAMATH_CALUDE_thomas_run_conversion_l915_91524


namespace NUMINAMATH_CALUDE_integral_proof_l915_91569

theorem integral_proof (x C : ℝ) : 
  (deriv (λ x => 1 / (2 * (2 * Real.sin x - 3 * Real.cos x)^2) + C)) x = 
  (2 * Real.cos x + 3 * Real.sin x) / (2 * Real.sin x - 3 * Real.cos x)^3 := by
  sorry

end NUMINAMATH_CALUDE_integral_proof_l915_91569


namespace NUMINAMATH_CALUDE_count_divisible_by_11_between_100_and_500_l915_91505

def count_divisible (lower upper divisor : ℕ) : ℕ :=
  (upper / divisor - (lower - 1) / divisor)

theorem count_divisible_by_11_between_100_and_500 :
  count_divisible 100 500 11 = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_11_between_100_and_500_l915_91505


namespace NUMINAMATH_CALUDE_motorcycles_parked_count_l915_91577

/-- The number of motorcycles parked between cars on a road -/
def motorcycles_parked (foreign_cars : ℕ) (domestic_cars_between : ℕ) : ℕ :=
  let total_cars := foreign_cars + (foreign_cars - 1) * domestic_cars_between
  total_cars - 1

/-- Theorem stating that given 5 foreign cars and 2 domestic cars between each pair,
    the number of motorcycles parked between all adjacent cars is 12 -/
theorem motorcycles_parked_count :
  motorcycles_parked 5 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_motorcycles_parked_count_l915_91577


namespace NUMINAMATH_CALUDE_inequality_proof_l915_91543

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c + c / b ≥ 4 * a / (a + b)) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l915_91543


namespace NUMINAMATH_CALUDE_zero_in_P_l915_91528

def P : Set ℝ := {x | x > -1}

theorem zero_in_P : (0 : ℝ) ∈ P := by sorry

end NUMINAMATH_CALUDE_zero_in_P_l915_91528


namespace NUMINAMATH_CALUDE_chicken_duck_count_l915_91558

theorem chicken_duck_count : ∃ (chickens ducks : ℕ),
  chickens + ducks = 239 ∧
  ducks = 3 * chickens + 15 ∧
  chickens = 56 ∧
  ducks = 183 := by
  sorry

end NUMINAMATH_CALUDE_chicken_duck_count_l915_91558


namespace NUMINAMATH_CALUDE_divisible_by_30_l915_91596

theorem divisible_by_30 (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_30_l915_91596


namespace NUMINAMATH_CALUDE_income_redistribution_l915_91575

/-- Represents the income distribution in a city --/
structure CityIncome where
  x : ℝ
  poor_income : ℝ
  middle_income : ℝ
  rich_income : ℝ
  tax_rate : ℝ

/-- Theorem stating the income redistribution after tax --/
theorem income_redistribution (c : CityIncome) 
  (h1 : c.poor_income = c.x)
  (h2 : c.middle_income = 3 * c.x)
  (h3 : c.rich_income = 6 * c.x)
  (h4 : c.poor_income + c.middle_income + c.rich_income = 100)
  (h5 : c.tax_rate = c.x^2 / 5 + c.x)
  (h6 : c.x = 10) :
  let tax_amount := c.rich_income * c.tax_rate / 100
  let poor_new := c.poor_income + 2 * tax_amount / 3
  let middle_new := c.middle_income + tax_amount / 3
  let rich_new := c.rich_income - tax_amount
  (poor_new = 22 ∧ middle_new = 36 ∧ rich_new = 42) := by
  sorry


end NUMINAMATH_CALUDE_income_redistribution_l915_91575


namespace NUMINAMATH_CALUDE_largest_value_l915_91503

theorem largest_value (a b c d e : ℝ) 
  (ha : a = 6 * (6 ^ (1 / 6)))
  (hb : b = 6 ^ (1 / 3))
  (hc : c = 6 ^ (1 / 4))
  (hd : d = 2 * (6 ^ (1 / 3)))
  (he : e = 3 * (4 ^ (1 / 3))) :
  d ≥ a ∧ d ≥ b ∧ d ≥ c ∧ d ≥ e :=
sorry

end NUMINAMATH_CALUDE_largest_value_l915_91503


namespace NUMINAMATH_CALUDE_min_value_of_f_l915_91552

theorem min_value_of_f (x y : ℝ) : 
  let f := fun (x y : ℝ) => Real.sqrt (x^2 + y^2) + Real.sqrt ((x-1)^2 + (y-1)^2) + Real.sqrt ((x+2)^2 + (y+2)^2)
  f x y ≥ 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l915_91552


namespace NUMINAMATH_CALUDE_circle_advance_theorem_l915_91532

/-- The number of ways to advance around a circle with n points exactly twice, 
    beginning and ending at a fixed point, without repeating a move. -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem stating that a(n-1) + a(n) = 2^n for all n ≥ 4 -/
theorem circle_advance_theorem (n : ℕ) (h : n ≥ 4) : 
  a (n - 1) + a n = 2^n := by sorry

end NUMINAMATH_CALUDE_circle_advance_theorem_l915_91532


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l915_91584

/-- Arithmetic sequence sum -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n - 1) * d) / 2

/-- Geometric sequence term -/
def geometric_term (a₁ : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  a₁ * q ^ (n - 1)

theorem arithmetic_and_geometric_sequences :
  (arithmetic_sum (-2) 4 8 = 96) ∧
  (geometric_term 1 3 7 = 729) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequences_l915_91584


namespace NUMINAMATH_CALUDE_distinct_prime_factors_30_factorial_l915_91547

/-- The number of distinct prime factors of 30! -/
def num_distinct_prime_factors_30_factorial : ℕ := 10

/-- Theorem stating that the number of distinct prime factors of 30! is 10 -/
theorem distinct_prime_factors_30_factorial :
  num_distinct_prime_factors_30_factorial = 10 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_30_factorial_l915_91547


namespace NUMINAMATH_CALUDE_bathtub_fill_time_l915_91514

/-- Proves that a bathtub with given capacity filled by a tap with given flow rate takes the calculated time to fill -/
theorem bathtub_fill_time (bathtub_capacity : ℝ) (tap_volume : ℝ) (tap_time : ℝ) (fill_time : ℝ) 
    (h1 : bathtub_capacity = 140)
    (h2 : tap_volume = 15)
    (h3 : tap_time = 3)
    (h4 : fill_time = bathtub_capacity / (tap_volume / tap_time)) :
  fill_time = 28 := by
  sorry

end NUMINAMATH_CALUDE_bathtub_fill_time_l915_91514


namespace NUMINAMATH_CALUDE_p_necessary_but_not_sufficient_for_q_l915_91519

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := 5*x - 6 > x^2

-- Define what it means for p to be necessary but not sufficient for q
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x)

-- Theorem statement
theorem p_necessary_but_not_sufficient_for_q :
  necessary_but_not_sufficient p q :=
sorry

end NUMINAMATH_CALUDE_p_necessary_but_not_sufficient_for_q_l915_91519


namespace NUMINAMATH_CALUDE_triangle_side_relation_l915_91507

theorem triangle_side_relation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a^2 - 16*b^2 - c^2 + 6*a*b + 10*b*c = 0) :
  a + c = 2*b := by sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l915_91507


namespace NUMINAMATH_CALUDE_inequality_range_proof_l915_91538

theorem inequality_range_proof (a : ℝ) : 
  (∀ x : ℝ, x > -2/a → a * Real.exp (a * x) - Real.log (x + 2/a) - 2 ≥ 0) ↔ 
  (a ≥ Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_proof_l915_91538


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l915_91517

theorem nested_fraction_evaluation : 
  2 + (1 / (2 + (1 / (2 + 2)))) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l915_91517


namespace NUMINAMATH_CALUDE_second_number_value_l915_91511

theorem second_number_value (A B C D : ℝ) : 
  C = 4.5 * B →
  B = 2.5 * A →
  D = 0.5 * (A + B) →
  (A + B + C + D) / 4 = 165 →
  B = 100 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l915_91511


namespace NUMINAMATH_CALUDE_cos_pi_4_minus_alpha_l915_91508

theorem cos_pi_4_minus_alpha (α : ℝ) (h : Real.sin (π / 4 + α) = 2 / 3) :
  Real.cos (π / 4 - α) = -Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_4_minus_alpha_l915_91508


namespace NUMINAMATH_CALUDE_triangle_with_equal_angles_isosceles_l915_91598

/-- A triangle is isosceles if it has at least two equal angles. -/
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

/-- Given a triangle ABC where ∠A = ∠B = 2∠C, prove that the triangle is isosceles. -/
theorem triangle_with_equal_angles_isosceles (a b c : ℝ) 
  (h1 : a + b + c = 180) -- Sum of angles in a triangle is 180°
  (h2 : a = b) -- ∠A = ∠B
  (h3 : a = 2 * c) -- ∠A = 2∠C
  : IsIsosceles a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_with_equal_angles_isosceles_l915_91598


namespace NUMINAMATH_CALUDE_f_has_zero_in_interval_l915_91533

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the function f in terms of g
def f (x : ℝ) : ℝ := (x^2 - 3*x + 2) * g x + 3*x - 4

-- State the theorem
theorem f_has_zero_in_interval (hg : Continuous g) :
  ∃ c ∈ Set.Ioo 1 2, f g c = 0 := by
  sorry


end NUMINAMATH_CALUDE_f_has_zero_in_interval_l915_91533


namespace NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l915_91580

/-- The price of a single window -/
def window_price : ℕ := 100

/-- The number of windows needed to get one free -/
def windows_for_free : ℕ := 3

/-- The number of windows Dave needs -/
def dave_windows : ℕ := 11

/-- The number of windows Doug needs -/
def doug_windows : ℕ := 9

/-- Calculate the cost of windows given the number needed -/
def calculate_cost (windows_needed : ℕ) : ℕ :=
  let free_windows := windows_needed / windows_for_free
  let paid_windows := windows_needed - free_windows
  paid_windows * window_price

/-- The theorem stating that there's no savings when purchasing together -/
theorem no_savings_on_joint_purchase :
  calculate_cost dave_windows + calculate_cost doug_windows =
  calculate_cost (dave_windows + doug_windows) :=
sorry

end NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l915_91580


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l915_91545

theorem angle_sum_theorem (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  α + 2*β = 2*π/3 →
  Real.tan (α/2) * Real.tan β = 2 - Real.sqrt 3 →
  α + β = 5*π/12 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l915_91545


namespace NUMINAMATH_CALUDE_total_spent_on_toys_l915_91531

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

theorem total_spent_on_toys : football_cost + marbles_cost = 12.30 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_toys_l915_91531


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l915_91589

theorem no_simultaneous_squares : ¬∃ (x y : ℕ), ∃ (a b : ℕ), 
  (x^2 + y = a^2) ∧ (y^2 + x = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l915_91589


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l915_91585

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -1) (h2 : b = 1/2) :
  ((2*a + b)^2 - (2*a + b)*(2*a - b)) / (2*b) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l915_91585


namespace NUMINAMATH_CALUDE_intersection_area_is_five_sevenths_l915_91551

/-- Represents a polygon in the sequence of cuts -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Represents the cutting process -/
def cut (p : Polygon) : Polygon := sorry

/-- The area of a polygon -/
def area (p : Polygon) : ℝ := sorry

/-- The initial unit square -/
def initialSquare : Polygon :=
  { vertices := [(0, 0), (1, 0), (1, 1), (0, 1)] }

/-- The sequence of polygons created by the cutting process -/
def polygonSequence : ℕ → Polygon
  | 0 => initialSquare
  | n + 1 => cut (polygonSequence n)

/-- The area of the intersection of all polygons in the sequence -/
def intersectionArea : ℝ := sorry

theorem intersection_area_is_five_sevenths :
  intersectionArea = 5/7 := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_five_sevenths_l915_91551


namespace NUMINAMATH_CALUDE_sin_symmetry_l915_91516

theorem sin_symmetry (t : ℝ) : 
  Real.sin ((π / 6 + t) + π / 3) = Real.sin ((π / 6 - t) + π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_symmetry_l915_91516


namespace NUMINAMATH_CALUDE_complex_equation_solution_l915_91525

theorem complex_equation_solution (a b : ℝ) : 
  (1 - Complex.I) * (a + 2 * Complex.I) = b * Complex.I → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l915_91525


namespace NUMINAMATH_CALUDE_original_class_size_l915_91565

theorem original_class_size (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) : 
  original_avg = 50 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ (original_size : ℕ), 
    (original_size : ℝ) * original_avg + (new_students : ℝ) * new_avg = 
    (original_size + new_students : ℝ) * (original_avg - avg_decrease) ∧
    original_size = 42 :=
by sorry


end NUMINAMATH_CALUDE_original_class_size_l915_91565


namespace NUMINAMATH_CALUDE_odd_function_domain_l915_91560

-- Define the function f
def f (a : ℝ) : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_domain (a : ℝ) : 
  (∀ x, f a x ≠ 0 → x ∈ Set.Ioo (3 - 2*a) (a + 1)) →  -- Domain condition
  (∀ x, f a (x + 1) = -f a (-x - 1)) →                -- Odd function condition
  a = 2 := by sorry

end NUMINAMATH_CALUDE_odd_function_domain_l915_91560


namespace NUMINAMATH_CALUDE_prob_at_least_two_long_specific_l915_91572

/-- Represents the probability of a road being at least 5 miles long -/
structure RoadProbability where
  ab : ℚ  -- Probability for road A to B
  bc : ℚ  -- Probability for road B to C
  cd : ℚ  -- Probability for road C to D

/-- Calculates the probability of selecting at least two roads that are at least 5 miles long -/
def prob_at_least_two_long (p : RoadProbability) : ℚ :=
  p.ab * p.bc * (1 - p.cd) +  -- A to B and B to C are long, C to D is not
  p.ab * (1 - p.bc) * p.cd +  -- A to B and C to D are long, B to C is not
  (1 - p.ab) * p.bc * p.cd +  -- B to C and C to D are long, A to B is not
  p.ab * p.bc * p.cd          -- All three roads are long

theorem prob_at_least_two_long_specific : 
  let p : RoadProbability := { ab := 3/4, bc := 2/3, cd := 1/2 }
  prob_at_least_two_long p = 11/24 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_long_specific_l915_91572


namespace NUMINAMATH_CALUDE_sqrt_three_custom_op_equals_twelve_l915_91586

-- Define the custom operation
def custom_op (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- Theorem statement
theorem sqrt_three_custom_op_equals_twelve :
  custom_op (Real.sqrt 3) (Real.sqrt 3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_custom_op_equals_twelve_l915_91586


namespace NUMINAMATH_CALUDE_spurs_total_basketballs_l915_91535

/-- Represents a basketball team -/
structure BasketballTeam where
  num_players : ℕ
  balls_per_player : ℕ

/-- Calculates the total number of basketballs for a team -/
def total_basketballs (team : BasketballTeam) : ℕ :=
  team.num_players * team.balls_per_player

/-- The Spurs basketball team -/
def spurs : BasketballTeam :=
  { num_players := 35
    balls_per_player := 15 }

/-- Theorem: The Spurs basketball team has 525 basketballs in total -/
theorem spurs_total_basketballs :
  total_basketballs spurs = 525 := by
  sorry

end NUMINAMATH_CALUDE_spurs_total_basketballs_l915_91535


namespace NUMINAMATH_CALUDE_anitas_strawberries_l915_91522

theorem anitas_strawberries (total_cartons : ℕ) (blueberry_cartons : ℕ) (cartons_to_buy : ℕ) 
  (h1 : total_cartons = 26)
  (h2 : blueberry_cartons = 9)
  (h3 : cartons_to_buy = 7) :
  total_cartons - (blueberry_cartons + cartons_to_buy) = 10 := by
  sorry

end NUMINAMATH_CALUDE_anitas_strawberries_l915_91522


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l915_91574

theorem rectangular_box_volume 
  (a b c : ℝ) 
  (edge_sum : 4 * a + 4 * b + 4 * c = 180) 
  (diagonal : a^2 + b^2 + c^2 = 25^2) : 
  a * b * c = 32125 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l915_91574


namespace NUMINAMATH_CALUDE_product_sum_relation_l915_91526

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 10 → b = 9 → b - a = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l915_91526


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l915_91555

-- Define the type of real-valued functions
def RealFunction := ℝ → ℝ

-- Define the properties of f
def IsStrictlyIncreasing (f : RealFunction) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

def HasInverse (f g : RealFunction) : Prop :=
  (∀ x : ℝ, f (g x) = x) ∧ (∀ x : ℝ, g (f x) = x)

def SatisfiesEquation (f g : RealFunction) : Prop :=
  ∀ x : ℝ, f x + g x = 2 * x

-- Main theorem
theorem unique_function_satisfying_conditions :
  ∃! f : RealFunction,
    IsStrictlyIncreasing f ∧
    (∃ g : RealFunction, HasInverse f g ∧ SatisfiesEquation f g) ∧
    (∀ x : ℝ, f x = x) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l915_91555


namespace NUMINAMATH_CALUDE_probability_of_speaking_hindi_l915_91568

/-- The probability of speaking Hindi in a village -/
theorem probability_of_speaking_hindi 
  (total_population : ℕ) 
  (tamil_speakers : ℕ) 
  (english_speakers : ℕ) 
  (both_speakers : ℕ) 
  (h_total : total_population = 1024)
  (h_tamil : tamil_speakers = 720)
  (h_english : english_speakers = 562)
  (h_both : both_speakers = 346)
  (h_non_negative : total_population ≥ tamil_speakers + english_speakers - both_speakers) :
  (total_population - (tamil_speakers + english_speakers - both_speakers)) / total_population = 
  (1024 - (720 + 562 - 346)) / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_speaking_hindi_l915_91568


namespace NUMINAMATH_CALUDE_name_length_difference_l915_91587

/-- Given that Elida has 5 letters in her name and the total of 10 times the average number
    of letters in both names is 65, prove that Adrianna's name has 3 more letters than Elida's name. -/
theorem name_length_difference (elida_length : ℕ) (adrianna_length : ℕ) : 
  elida_length = 5 →
  10 * ((elida_length + adrianna_length) / 2) = 65 →
  adrianna_length = elida_length + 3 := by
sorry


end NUMINAMATH_CALUDE_name_length_difference_l915_91587


namespace NUMINAMATH_CALUDE_sum_of_all_coeff_sum_of_even_coeff_sum_of_coeff_except_a₀_S_mod_9_l915_91556

-- Define the polynomial coefficients
def a₀ : ℝ := sorry
def a₁ : ℝ := sorry
def a₂ : ℝ := sorry
def a₃ : ℝ := sorry
def a₄ : ℝ := sorry

-- Define the polynomial equation
axiom polynomial_eq : ∀ x : ℝ, (3*x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4

-- Define S
def S : ℕ := (Finset.range 27).sum (fun k => Nat.choose 27 (k + 1))

-- Theorem statements
theorem sum_of_all_coeff : a₀ + a₁ + a₂ + a₃ + a₄ = 16 := by sorry

theorem sum_of_even_coeff : a₀ + a₂ + a₄ = 136 := by sorry

theorem sum_of_coeff_except_a₀ : a₁ + a₂ + a₃ + a₄ = 15 := by sorry

theorem S_mod_9 : S % 9 = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_all_coeff_sum_of_even_coeff_sum_of_coeff_except_a₀_S_mod_9_l915_91556


namespace NUMINAMATH_CALUDE_equal_earnings_l915_91559

theorem equal_earnings (t : ℝ) : 
  (t - 6) * (2 * t - 5) = (2 * t - 8) * (t - 5) → t = 10 := by
  sorry

end NUMINAMATH_CALUDE_equal_earnings_l915_91559


namespace NUMINAMATH_CALUDE_segment_length_on_number_line_l915_91534

theorem segment_length_on_number_line : 
  let a : ℝ := -3
  let b : ℝ := 5
  |b - a| = 8 := by sorry

end NUMINAMATH_CALUDE_segment_length_on_number_line_l915_91534


namespace NUMINAMATH_CALUDE_vector_operation_result_l915_91513

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

variable (O A B C E : E)

theorem vector_operation_result :
  (A - B) - (C - B) + (O - E) - (O - C) = A - E := by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l915_91513


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l915_91578

/-- A particle moving in a straight line with distance-time relationship s(t) = 4t^2 - 3 -/
def s (t : ℝ) : ℝ := 4 * t^2 - 3

/-- The instantaneous velocity function v(t) -/
def v (t : ℝ) : ℝ := 8 * t

theorem instantaneous_velocity_at_5 : v 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l915_91578


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l915_91502

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_faces : Fin 6 → Bool

/-- Counts the number of unit cubes with at least two painted faces in a painted cube -/
def count_multi_painted_cubes (c : Cube 4) : ℕ :=
  sorry

/-- The theorem stating that a 4x4x4 painted cube has 56 unit cubes with at least two painted faces -/
theorem painted_cube_theorem (c : Cube 4) 
  (h : ∀ (f : Fin 6), c.painted_faces f = true) : 
  count_multi_painted_cubes c = 56 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l915_91502


namespace NUMINAMATH_CALUDE_negative_reciprocal_inequality_l915_91591

theorem negative_reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : -1/a < -1/b := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_inequality_l915_91591


namespace NUMINAMATH_CALUDE_f_always_negative_l915_91553

/-- The function f(x) = ax^2 + ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 1

/-- Theorem stating that f(x) < 0 for all x ∈ ℝ if and only if -4 < a ≤ 0 -/
theorem f_always_negative (a : ℝ) : 
  (∀ x : ℝ, f a x < 0) ↔ (-4 < a ∧ a ≤ 0) := by sorry

end NUMINAMATH_CALUDE_f_always_negative_l915_91553


namespace NUMINAMATH_CALUDE_unique_multiplication_problem_l915_91567

theorem unique_multiplication_problem :
  ∃! (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    100 ≤ a * b ∧ a * b < 1000 ∧
    (a * b) % 100 / 10 = 1 ∧
    (a * b) % 10 = 2 ∧
    (b * (a % 10)) % 100 = 0 ∧
    (a % 10 + b % 10) = 6 ∧
    a * b = 612 :=
by sorry

end NUMINAMATH_CALUDE_unique_multiplication_problem_l915_91567


namespace NUMINAMATH_CALUDE_ellipse_foci_y_axis_l915_91518

theorem ellipse_foci_y_axis (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (9 - m) + y^2 / (m - 5) = 1 → 
    ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1^2 + p.2^2 = c^2 → 
      (0, c) ∈ {f : ℝ × ℝ | (f.1 - x)^2 + (f.2 - y)^2 + (f.1 + x)^2 + (f.2 - y)^2 = 4 * c^2}) →
  7 < m ∧ m < 9 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_y_axis_l915_91518


namespace NUMINAMATH_CALUDE_four_digit_integers_with_five_or_seven_l915_91561

theorem four_digit_integers_with_five_or_seven (total_four_digit : Nat) 
  (four_digit_without_five_or_seven : Nat) :
  total_four_digit = 9000 →
  four_digit_without_five_or_seven = 3584 →
  total_four_digit - four_digit_without_five_or_seven = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_integers_with_five_or_seven_l915_91561


namespace NUMINAMATH_CALUDE_correct_solution_l915_91509

theorem correct_solution (a b : ℚ) : 
  (∀ x y, x = 13 ∧ y = 7 → b * x - 7 * y = 16) →
  (∀ x y, x = 9 ∧ y = 4 → 2 * x + a * y = 6) →
  2 * 6 + a * 2 = 6 ∧ b * 6 - 7 * 2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_correct_solution_l915_91509


namespace NUMINAMATH_CALUDE_sins_match_prayers_l915_91573

structure Sin :=
  (teDeum : ℕ)
  (paterNoster : ℕ)
  (credo : ℕ)

def pride : Sin := ⟨1, 2, 0⟩
def slander : Sin := ⟨0, 2, 7⟩
def sloth : Sin := ⟨2, 0, 0⟩
def adultery : Sin := ⟨10, 10, 10⟩
def gluttony : Sin := ⟨1, 0, 0⟩
def selfishness : Sin := ⟨0, 3, 1⟩
def jealousy : Sin := ⟨0, 3, 0⟩
def evilSpeaking : Sin := ⟨0, 7, 2⟩

def totalPrayers (sins : List Sin) : Sin :=
  sins.foldl (λ acc sin => ⟨acc.teDeum + sin.teDeum, acc.paterNoster + sin.paterNoster, acc.credo + sin.credo⟩) ⟨0, 0, 0⟩

theorem sins_match_prayers :
  let sins := [slander] ++ List.replicate 2 evilSpeaking ++ [selfishness] ++ List.replicate 9 gluttony
  totalPrayers sins = ⟨9, 12, 10⟩ := by sorry

end NUMINAMATH_CALUDE_sins_match_prayers_l915_91573


namespace NUMINAMATH_CALUDE_equal_roots_real_roots_l915_91562

/-- The quadratic equation given in the problem -/
def quadratic_equation (m x : ℝ) : Prop :=
  2 * (m + 1) * x^2 + 4 * m * x + 3 * m = 2

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  -8 * m^2 - 8 * m + 16

theorem equal_roots (m : ℝ) : 
  (∃ x : ℝ, quadratic_equation m x ∧ 
    ∀ y : ℝ, quadratic_equation m y → y = x) ↔ 
  (m = -2 ∨ m = 1) :=
sorry

theorem real_roots (m : ℝ) :
  (m = -1 → ∃ x : ℝ, quadratic_equation m x ∧ x = -5/4) ∧
  (m ≠ -1 → ∃ x : ℝ, quadratic_equation m x ∧ 
    ∃ s : ℝ, s^2 = -2*m^2 - 2*m + 4 ∧ 
      (x = (-2*m + s) / (2*(m+1)) ∨ x = (-2*m - s) / (2*(m+1)))) :=
sorry

end NUMINAMATH_CALUDE_equal_roots_real_roots_l915_91562


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_inequality_l915_91595

theorem integer_pairs_satisfying_inequality :
  ∀ a b : ℕ+, 
    (11 * a * b ≤ a^3 - b^3 ∧ a^3 - b^3 ≤ 12 * a * b) ↔ 
    ((a = 30 ∧ b = 25) ∨ (a = 8 ∧ b = 4)) := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_inequality_l915_91595


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_l915_91599

theorem cos_alpha_minus_pi_sixth (α : ℝ) 
  (h : Real.sin (α + π / 6) + Real.cos α = 4 * Real.sqrt 3 / 5) : 
  Real.cos (α - π / 6) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_l915_91599


namespace NUMINAMATH_CALUDE_quadratic_inequality_max_value_l915_91570

theorem quadratic_inequality_max_value (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = 2 * Real.sqrt 2 - 2 ∧
    ∀ k : ℝ, (b^2) / (a^2 + c^2) ≤ k → k ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_max_value_l915_91570


namespace NUMINAMATH_CALUDE_fraction_multiplication_l915_91588

theorem fraction_multiplication : (2 : ℚ) / 5 * (7 : ℚ) / 10 = (7 : ℚ) / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l915_91588


namespace NUMINAMATH_CALUDE_three_weighings_sufficient_and_necessary_l915_91581

/-- Represents the outcome of a weighing on a balance scale -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- A type representing a weighing strategy -/
def WeighStrategy := List (List Nat × List Nat)

/-- Represents the state of knowledge about which coin might be fake -/
structure FakeCoinInfo where
  possibleFakes : List Nat
  isHeavy : Option Bool

/-- The total number of coins -/
def totalCoins : Nat := 13

/-- A theorem stating that 3 weighings are sufficient and necessary to identify the fake coin -/
theorem three_weighings_sufficient_and_necessary :
  ∃ (strategy : WeighStrategy),
    (strategy.length ≤ 3) ∧
    (∀ (fakeCoin : Nat) (isHeavy : Bool),
      fakeCoin < totalCoins →
      ∃ (finalInfo : FakeCoinInfo),
        finalInfo.possibleFakes = [fakeCoin] ∧
        finalInfo.isHeavy = some isHeavy) ∧
    (∀ (strategy' : WeighStrategy),
      strategy'.length < 3 →
      ∃ (fakeCoin1 fakeCoin2 : Nat) (isHeavy1 isHeavy2 : Bool),
        fakeCoin1 ≠ fakeCoin2 ∧
        fakeCoin1 < totalCoins ∧
        fakeCoin2 < totalCoins ∧
        ¬∃ (finalInfo : FakeCoinInfo),
          (finalInfo.possibleFakes = [fakeCoin1] ∧ finalInfo.isHeavy = some isHeavy1) ∨
          (finalInfo.possibleFakes = [fakeCoin2] ∧ finalInfo.isHeavy = some isHeavy2)) :=
by sorry

end NUMINAMATH_CALUDE_three_weighings_sufficient_and_necessary_l915_91581


namespace NUMINAMATH_CALUDE_set_disjoint_iff_m_range_l915_91504

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m < x ∧ x < m+1}

theorem set_disjoint_iff_m_range (m : ℝ) : 
  (∀ x ∈ A, x ∉ B m) ↔ m ∈ Set.Iic (-2) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_set_disjoint_iff_m_range_l915_91504


namespace NUMINAMATH_CALUDE_student_distribution_l915_91563

theorem student_distribution (total : ℕ) (h_total : total > 0) :
  let third_year := (30 : ℕ) * total / 100
  let not_second_year := (90 : ℕ) * total / 100
  let second_year := total - not_second_year
  let not_third_year := total - third_year
  (second_year : ℚ) / not_third_year = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_student_distribution_l915_91563


namespace NUMINAMATH_CALUDE_logarithm_inequality_l915_91523

theorem logarithm_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log (1 + Real.sqrt (a * b)) ≤ (Real.log (1 + a) + Real.log (1 + b)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l915_91523


namespace NUMINAMATH_CALUDE_aunt_uncle_gift_amount_l915_91554

/-- The amount of money Chris had before his birthday -/
def initial_amount : ℕ := 159

/-- The amount Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- The amount Chris received from his parents -/
def parents_gift : ℕ := 75

/-- The total amount Chris has after his birthday -/
def final_amount : ℕ := 279

/-- The amount Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := final_amount - initial_amount - grandmother_gift - parents_gift

theorem aunt_uncle_gift_amount : aunt_uncle_gift = 20 := by
  sorry

end NUMINAMATH_CALUDE_aunt_uncle_gift_amount_l915_91554
