import Mathlib

namespace NUMINAMATH_CALUDE_xiaoGang_weight_not_80_grams_l3948_394855

-- Define a person
structure Person where
  name : String
  weight : Float  -- weight in kilograms

-- Define Xiao Gang
def xiaoGang : Person := { name := "Xiao Gang", weight := 80 }

-- Theorem to prove
theorem xiaoGang_weight_not_80_grams : 
  xiaoGang.weight ≠ 0.08 := by sorry

end NUMINAMATH_CALUDE_xiaoGang_weight_not_80_grams_l3948_394855


namespace NUMINAMATH_CALUDE_max_distance_to_origin_l3948_394834

theorem max_distance_to_origin : 
  let curve := {(x, y) : ℝ × ℝ | ∃ θ : ℝ, x = Real.sqrt 3 + Real.cos θ ∧ y = 1 + Real.sin θ}
  ∀ p ∈ curve, ∃ q ∈ curve, ∀ r ∈ curve, Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) ≥ Real.sqrt ((r.1 - 0)^2 + (r.2 - 0)^2) ∧
  Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) = 3 := by
sorry


end NUMINAMATH_CALUDE_max_distance_to_origin_l3948_394834


namespace NUMINAMATH_CALUDE_anya_lost_games_l3948_394858

/-- Represents a girl playing table tennis -/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- Represents the state of a girl (playing or resting) -/
inductive State
| Playing
| Resting

/-- The number of games each girl played -/
def games_played (g : Girl) : Nat :=
  match g with
  | Girl.Anya => 4
  | Girl.Bella => 6
  | Girl.Valya => 7
  | Girl.Galya => 10
  | Girl.Dasha => 11

/-- The total number of games played -/
def total_games : Nat := 19

/-- Predicate to check if a girl lost a specific game -/
def lost_game (g : Girl) (game_number : Nat) : Prop := sorry

/-- Theorem stating that Anya lost in games 4, 8, 12, and 16 -/
theorem anya_lost_games :
  lost_game Girl.Anya 4 ∧
  lost_game Girl.Anya 8 ∧
  lost_game Girl.Anya 12 ∧
  lost_game Girl.Anya 16 :=
by sorry

end NUMINAMATH_CALUDE_anya_lost_games_l3948_394858


namespace NUMINAMATH_CALUDE_chicken_ratio_problem_l3948_394814

/-- Given the following conditions:
    - Wendi initially has 4 chickens
    - She increases the number of chickens by a ratio r
    - One chicken is eaten by a neighbor's dog
    - Wendi finds and brings home 6 more chickens
    - The final number of chickens is 13
    Prove that the ratio r is equal to 2 -/
theorem chicken_ratio_problem (r : ℚ) : 
  (4 * r - 1 + 6 : ℚ) = 13 → r = 2 := by sorry

end NUMINAMATH_CALUDE_chicken_ratio_problem_l3948_394814


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3948_394897

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_second : a 2 = 4)
  (h_sixth : a 6 = 64) :
  a 4 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3948_394897


namespace NUMINAMATH_CALUDE_chocolates_for_charlie_l3948_394891

/-- Represents the number of Saturdays in a month -/
def saturdays_in_month : ℕ := 4

/-- Represents the number of chocolates Kantana buys for herself each Saturday -/
def chocolates_for_self : ℕ := 2

/-- Represents the number of chocolates Kantana buys for her sister each Saturday -/
def chocolates_for_sister : ℕ := 1

/-- Represents the total number of chocolates Kantana bought for the month -/
def total_chocolates : ℕ := 22

/-- Theorem stating that Kantana bought 10 chocolates for Charlie's birthday gift -/
theorem chocolates_for_charlie : 
  total_chocolates - (saturdays_in_month * (chocolates_for_self + chocolates_for_sister)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_for_charlie_l3948_394891


namespace NUMINAMATH_CALUDE_probability_point_in_circle_l3948_394850

/-- The probability that a randomly selected point from a square with side length 6
    is within a circle of radius 2 centered at the origin is π/9 -/
theorem probability_point_in_circle (s : ℝ) (r : ℝ) : 
  s = 6 → r = 2 → (π * r^2) / (s^2) = π / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l3948_394850


namespace NUMINAMATH_CALUDE_simplify_fraction_l3948_394867

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3948_394867


namespace NUMINAMATH_CALUDE_eggs_sold_equals_540_l3948_394892

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 36

/-- The initial number of trays collected -/
def initial_trays : ℕ := 10

/-- The number of trays dropped accidentally -/
def dropped_trays : ℕ := 2

/-- The number of additional trays added -/
def additional_trays : ℕ := 7

/-- The total number of eggs sold -/
def total_eggs_sold : ℕ := eggs_per_tray * (initial_trays - dropped_trays + additional_trays)

theorem eggs_sold_equals_540 : total_eggs_sold = 540 := by
  sorry

end NUMINAMATH_CALUDE_eggs_sold_equals_540_l3948_394892


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_largest_n_is_49999_l3948_394809

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 50000 →
  (3 * (n - 3)^2 - 4 * n + 28) % 7 = 0 →
  n ≤ 49999 :=
by sorry

theorem largest_n_is_49999 : 
  (3 * (49999 - 3)^2 - 4 * 49999 + 28) % 7 = 0 ∧
  ∀ m : ℕ, m > 49999 → m < 50000 → (3 * (m - 3)^2 - 4 * m + 28) % 7 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_largest_n_is_49999_l3948_394809


namespace NUMINAMATH_CALUDE_fraction_domain_l3948_394857

theorem fraction_domain (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_fraction_domain_l3948_394857


namespace NUMINAMATH_CALUDE_lcm_problem_l3948_394894

theorem lcm_problem (n : ℕ) (h1 : n > 0) (h2 : Nat.lcm 30 n = 90) (h3 : Nat.lcm n 45 = 180) : n = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3948_394894


namespace NUMINAMATH_CALUDE_rectangle_arrangement_possible_l3948_394830

/-- Represents a small 1×2 rectangle with 2 stars -/
structure SmallRectangle :=
  (width : Nat) (height : Nat) (stars : Nat)

/-- Represents the large 5×200 rectangle -/
structure LargeRectangle :=
  (width : Nat) (height : Nat)
  (smallRectangles : List SmallRectangle)

/-- Checks if a number is even -/
def isEven (n : Nat) : Prop := ∃ k, n = 2 * k

/-- Calculates the total number of stars in a list of small rectangles -/
def totalStars (rectangles : List SmallRectangle) : Nat :=
  rectangles.foldl (fun acc rect => acc + rect.stars) 0

/-- Theorem: It's possible to arrange 500 1×2 rectangles into a 5×200 rectangle
    with an even number of stars in each row and column -/
theorem rectangle_arrangement_possible :
  ∃ (largeRect : LargeRectangle),
    largeRect.width = 200 ∧
    largeRect.height = 5 ∧
    largeRect.smallRectangles.length = 500 ∧
    (∀ smallRect ∈ largeRect.smallRectangles, smallRect.width = 1 ∧ smallRect.height = 2 ∧ smallRect.stars = 2) ∧
    (∀ row ∈ List.range 5, isEven (totalStars (largeRect.smallRectangles.filter (fun _ => true)))) ∧
    (∀ col ∈ List.range 200, isEven (totalStars (largeRect.smallRectangles.filter (fun _ => true)))) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_possible_l3948_394830


namespace NUMINAMATH_CALUDE_inequality_preservation_l3948_394843

theorem inequality_preservation (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3948_394843


namespace NUMINAMATH_CALUDE_smart_number_characterization_smart_number_2015_l3948_394865

/-- A positive integer is a smart number if it can be expressed as the difference of squares of two positive integers. -/
def is_smart_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > b ∧ n = a ^ 2 - b ^ 2

/-- Theorem stating the characterization of smart numbers -/
theorem smart_number_characterization (n : ℕ) :
  is_smart_number n ↔ (n > 1 ∧ n % 2 = 1) ∨ (n ≥ 8 ∧ n % 4 = 0) :=
sorry

/-- Function to get the nth smart number -/
def nth_smart_number (n : ℕ) : ℕ :=
sorry

/-- Theorem stating that the 2015th smart number is 2689 -/
theorem smart_number_2015 : nth_smart_number 2015 = 2689 :=
sorry

end NUMINAMATH_CALUDE_smart_number_characterization_smart_number_2015_l3948_394865


namespace NUMINAMATH_CALUDE_vector_decomposition_l3948_394820

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![2, -1, 11]
def p : Fin 3 → ℝ := ![1, 1, 0]
def q : Fin 3 → ℝ := ![0, 1, -2]
def r : Fin 3 → ℝ := ![1, 0, 3]

/-- Theorem stating that x can be expressed as a linear combination of p, q, and r -/
theorem vector_decomposition :
  x = (-3 : ℝ) • p + 2 • q + 5 • r := by
  sorry

end NUMINAMATH_CALUDE_vector_decomposition_l3948_394820


namespace NUMINAMATH_CALUDE_average_temperature_l3948_394872

def temperatures : List ℚ := [73, 76, 75, 78, 74]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℚ) = 75.2 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l3948_394872


namespace NUMINAMATH_CALUDE_largest_solution_and_ratio_l3948_394890

theorem largest_solution_and_ratio : ∃ (a b c d : ℤ),
  let x : ℝ := (a + b * Real.sqrt c) / d
  (7 * x) / 9 + 2 = 4 / x ∧
  (∀ (a' b' c' d' : ℤ), 
    let x' : ℝ := (a' + b' * Real.sqrt c') / d'
    (7 * x') / 9 + 2 = 4 / x' → x' ≤ x) ∧
  x = (-9 + 3 * Real.sqrt 111) / 7 ∧
  a * c * d / b = -2313 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_and_ratio_l3948_394890


namespace NUMINAMATH_CALUDE_base_height_example_l3948_394812

/-- Given a sculpture height in feet and inches, and a total height of sculpture and base,
    calculate the height of the base in feet. -/
def base_height (sculpture_feet : ℕ) (sculpture_inches : ℕ) (total_height : ℚ) : ℚ :=
  total_height - (sculpture_feet : ℚ) - ((sculpture_inches : ℚ) / 12)

/-- Theorem stating that for a sculpture of 2 feet 10 inches and a total height of 3.5 feet,
    the base height is 2/3 feet. -/
theorem base_height_example : base_height 2 10 (7/2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_base_height_example_l3948_394812


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3948_394838

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 1 - Complex.I) : 
  z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3948_394838


namespace NUMINAMATH_CALUDE_inequality_of_distinct_positive_numbers_l3948_394885

theorem inequality_of_distinct_positive_numbers (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hcd : c ≠ d) (hda : d ≠ a) :
  a^2 / b + b^2 / c + c^2 / d + d^2 / a > a + b + c + d :=
sorry

end NUMINAMATH_CALUDE_inequality_of_distinct_positive_numbers_l3948_394885


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l3948_394832

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  h_a : a < 10
  h_b : b < 10
  h_c : c < 10
  h_a_pos : 0 < a

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : ℕ :=
  100 * n.a + 10 * n.b + n.c

def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : ℕ :=
  100 * n.c + 10 * n.b + n.a

def ThreeDigitNumber.sumDigits (n : ThreeDigitNumber) : ℕ :=
  n.a + n.b + n.c

theorem three_digit_number_theorem (n : ThreeDigitNumber) :
  (n.toNat / n.reverse = 3 ∧ n.toNat % n.reverse = n.sumDigits) →
  (n.toNat = 441 ∨ n.toNat = 882) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l3948_394832


namespace NUMINAMATH_CALUDE_saline_solution_concentration_l3948_394841

/-- Proves that mixing 100 kg of 30% saline solution with 200 kg of pure water
    results in a final saline solution with a concentration of 10%. -/
theorem saline_solution_concentration
  (initial_solution_weight : ℝ)
  (initial_concentration : ℝ)
  (pure_water_weight : ℝ)
  (h1 : initial_solution_weight = 100)
  (h2 : initial_concentration = 0.3)
  (h3 : pure_water_weight = 200) :
  let salt_weight := initial_solution_weight * initial_concentration
  let total_weight := initial_solution_weight + pure_water_weight
  let final_concentration := salt_weight / total_weight
  final_concentration = 0.1 := by
sorry

end NUMINAMATH_CALUDE_saline_solution_concentration_l3948_394841


namespace NUMINAMATH_CALUDE_intersection_range_l3948_394893

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_O₂ (x y r : ℝ) : Prop := (x - 7)^2 + y^2 = r^2

-- Define the condition for r
def r_positive (r : ℝ) : Prop := r > 0

-- Define the intersection condition
def circles_intersect (r : ℝ) : Prop :=
  ∃ x y, circle_O₁ x y ∧ circle_O₂ x y r

-- Main theorem
theorem intersection_range :
  ∀ r, r_positive r → (circles_intersect r ↔ 2 < r ∧ r < 12) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l3948_394893


namespace NUMINAMATH_CALUDE_m_range_l3948_394896

theorem m_range (m : ℝ) : 
  (∀ θ : ℝ, m^2 + (Real.cos θ)^2 * m - 5*m + 4*(Real.sin θ)^2 ≥ 0) → 
  (m ≥ 4 ∨ m ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_m_range_l3948_394896


namespace NUMINAMATH_CALUDE_smallest_possible_a_l3948_394870

theorem smallest_possible_a (a b c : ℚ) :
  a > 0 ∧
  (∃ n : ℚ, a + b + c = n) ∧
  (∀ x y : ℚ, y = a * x^2 + b * x + c ↔ y + 2/3 = a * (x - 1/3)^2) →
  ∀ a' : ℚ, (a' > 0 ∧
    (∃ b' c' : ℚ, (∃ n : ℚ, a' + b' + c' = n) ∧
    (∀ x y : ℚ, y = a' * x^2 + b' * x + c' ↔ y + 2/3 = a' * (x - 1/3)^2))) →
  a ≤ a' ∧ a = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l3948_394870


namespace NUMINAMATH_CALUDE_nectarines_per_box_l3948_394874

theorem nectarines_per_box (num_crates : ℕ) (oranges_per_crate : ℕ) (num_boxes : ℕ) (total_fruits : ℕ) :
  num_crates = 12 →
  oranges_per_crate = 150 →
  num_boxes = 16 →
  total_fruits = 2280 →
  (total_fruits - num_crates * oranges_per_crate) / num_boxes = 30 :=
by sorry

end NUMINAMATH_CALUDE_nectarines_per_box_l3948_394874


namespace NUMINAMATH_CALUDE_girls_count_l3948_394878

/-- The number of boys in the school -/
def num_boys : ℕ := 841

/-- The difference between the number of boys and girls -/
def boy_girl_diff : ℕ := 807

/-- The number of girls in the school -/
def num_girls : ℕ := num_boys - boy_girl_diff

theorem girls_count : num_girls = 34 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_l3948_394878


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3948_394828

/-- Given a cylinder whose lateral surface unfolds into a rectangle with sides of length 6π and 4π,
    its total surface area is either 24π² + 18π or 24π² + 8π. -/
theorem cylinder_surface_area (h : ℝ) (r : ℝ) :
  (h = 6 * Real.pi ∧ 2 * Real.pi * r = 4 * Real.pi) ∨ 
  (h = 4 * Real.pi ∧ 2 * Real.pi * r = 6 * Real.pi) →
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 18 * Real.pi ∨
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3948_394828


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l3948_394823

theorem intersection_points_theorem :
  let roots : Set ℝ := {1, 2}
  let eq_A : ℝ → ℝ → Prop := λ x y ↦ (y = x^2 ∨ y = 3*x)
  let eq_B : ℝ → ℝ → Prop := λ x y ↦ (y = x^2 - 3*x + 2 ∨ y = 2)
  let eq_C : ℝ → ℝ → Prop := λ x y ↦ (y = x ∨ y = x - 2)
  let eq_D : ℝ → ℝ → Prop := λ x y ↦ (y = x^2 - 3*x + 3 ∨ y = 3)
  (∀ x y, eq_A x y → x ∉ roots) ∧
  (∀ x y, eq_B x y → x ∉ roots) ∧
  (¬∃ x y, eq_C x y) ∧
  (∀ x y, eq_D x y → x ∉ roots) := by
  sorry


end NUMINAMATH_CALUDE_intersection_points_theorem_l3948_394823


namespace NUMINAMATH_CALUDE_angle_between_planes_l3948_394836

def plane1 : ℝ → ℝ → ℝ → ℝ := fun x y z ↦ 3 * x - 4 * y + z - 8
def plane2 : ℝ → ℝ → ℝ → ℝ := fun x y z ↦ 9 * x - 12 * y - 4 * z + 6

def normal1 : Fin 3 → ℝ := fun i ↦ match i with
  | 0 => 3
  | 1 => -4
  | 2 => 1

def normal2 : Fin 3 → ℝ := fun i ↦ match i with
  | 0 => 9
  | 1 => -12
  | 2 => -4

theorem angle_between_planes :
  let dot_product := (normal1 0 * normal2 0 + normal1 1 * normal2 1 + normal1 2 * normal2 2)
  let magnitude1 := Real.sqrt (normal1 0 ^ 2 + normal1 1 ^ 2 + normal1 2 ^ 2)
  let magnitude2 := Real.sqrt (normal2 0 ^ 2 + normal2 1 ^ 2 + normal2 2 ^ 2)
  dot_product / (magnitude1 * magnitude2) = 71 / (Real.sqrt 26 * Real.sqrt 241) := by
sorry

end NUMINAMATH_CALUDE_angle_between_planes_l3948_394836


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l3948_394853

/-- Represents a rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  total_area : ℝ
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  sum_of_areas : total_area = area1 + area2 + area3 + area4

/-- Theorem: Given a divided rectangle with specific areas, the fourth area is 28 -/
theorem fourth_rectangle_area
  (rect : DividedRectangle)
  (h1 : rect.total_area = 100)
  (h2 : rect.area1 = 24)
  (h3 : rect.area2 = 30)
  (h4 : rect.area3 = 18) :
  rect.area4 = 28 := by
  sorry


end NUMINAMATH_CALUDE_fourth_rectangle_area_l3948_394853


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_3_and_4_l3948_394889

theorem largest_five_digit_divisible_by_3_and_4 : ∃ n : ℕ, 
  (n ≤ 99999) ∧ 
  (n ≥ 10000) ∧ 
  (n % 3 = 0) ∧ 
  (n % 4 = 0) ∧ 
  (∀ m : ℕ, m ≤ 99999 ∧ m ≥ 10000 ∧ m % 3 = 0 ∧ m % 4 = 0 → m ≤ n) ∧
  n = 99996 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_3_and_4_l3948_394889


namespace NUMINAMATH_CALUDE_quadratic_equations_integer_solutions_l3948_394856

theorem quadratic_equations_integer_solutions 
  (b c : ℤ) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h1 : ∃ x y : ℤ, x ≠ y ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0)
  (h2 : ∃ u v : ℤ, u ≠ v ∧ u^2 + b*u - c = 0 ∧ v^2 + b*v - c = 0) :
  (∃ p q : ℕ+, p ≠ q ∧ 2*b^2 = p^2 + q^2) ∧
  (∃ r s : ℕ+, r ≠ s ∧ b^2 = r^2 + s^2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_integer_solutions_l3948_394856


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3948_394884

/-- 
Given a geometric series with first term a and common ratio r,
prove that if the sum of the series is 24 and the sum of terms
with odd powers of r is 10, then r = 5/7.
-/
theorem geometric_series_ratio (a r : ℝ) : 
  (∑' n, a * r^n) = 24 →
  (∑' n, a * r^(2*n+1)) = 10 →
  r = 5/7 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3948_394884


namespace NUMINAMATH_CALUDE_largest_c_for_function_range_l3948_394849

theorem largest_c_for_function_range (f : ℝ → ℝ) (c : ℝ) :
  (∀ x, f x = x^2 - 6*x + c) →
  (∃ x, f x = 4) →
  c ≤ 13 ∧ 
  (∀ d > 13, ¬∃ x, x^2 - 6*x + d = 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_function_range_l3948_394849


namespace NUMINAMATH_CALUDE_function_inequality_range_l3948_394803

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem function_inequality_range (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Set.Ioo (-1) 1, f x ∈ Set.Ioo (-1) 1)
  (h2 : is_odd f)
  (h3 : is_decreasing_on f (-1) 1) :
  {a : ℝ | f (1 - a) + f (1 - a^2) < 0} = Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_range_l3948_394803


namespace NUMINAMATH_CALUDE_intersection_point_pq_rs_l3948_394800

/-- The intersection point of two lines in 3D space --/
def intersection_point (p q r s : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Given points in 3D space --/
def P : ℝ × ℝ × ℝ := (4, -3, 6)
def Q : ℝ × ℝ × ℝ := (20, -23, 14)
def R : ℝ × ℝ × ℝ := (-2, 7, -10)
def S : ℝ × ℝ × ℝ := (6, -11, 16)

/-- Theorem stating that the intersection point of lines PQ and RS is (180/19, -283/19, 202/19) --/
theorem intersection_point_pq_rs :
  intersection_point P Q R S = (180/19, -283/19, 202/19) := by sorry

end NUMINAMATH_CALUDE_intersection_point_pq_rs_l3948_394800


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_is_three_l3948_394813

theorem min_value_of_function (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

theorem min_value_is_three : ∃ (x : ℝ), x > 1 ∧ x + 1 / (x - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_is_three_l3948_394813


namespace NUMINAMATH_CALUDE_power_of_81_equals_9_l3948_394826

theorem power_of_81_equals_9 : (81 : ℝ) ^ (0.25 : ℝ) * (81 : ℝ) ^ (0.20 : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_equals_9_l3948_394826


namespace NUMINAMATH_CALUDE_partnership_investment_timing_l3948_394854

/-- A partnership problem with three investors --/
theorem partnership_investment_timing
  (x : ℝ)  -- A's investment amount
  (m : ℝ)  -- Months after which B invests
  (total_gain : ℝ)  -- Total annual gain
  (a_share : ℝ)  -- A's share of the gain
  (h1 : total_gain = 21000)  -- Given total gain
  (h2 : a_share = 7000)  -- Given A's share
  (h3 : a_share / total_gain = (x * 12) / (x * 12 + 2 * x * (12 - m) + 3 * x * 4))  -- Profit ratio equation
  : m = 6 :=
sorry

end NUMINAMATH_CALUDE_partnership_investment_timing_l3948_394854


namespace NUMINAMATH_CALUDE_sarah_apple_slices_l3948_394887

/-- Given a number of boxes of apples, apples per box, and slices per apple,
    calculate the total number of apple slices -/
def total_apple_slices (boxes : ℕ) (apples_per_box : ℕ) (slices_per_apple : ℕ) : ℕ :=
  boxes * apples_per_box * slices_per_apple

/-- Theorem: Sarah has 392 apple slices -/
theorem sarah_apple_slices :
  total_apple_slices 7 7 8 = 392 := by
  sorry

end NUMINAMATH_CALUDE_sarah_apple_slices_l3948_394887


namespace NUMINAMATH_CALUDE_inequality_proof_l3948_394899

theorem inequality_proof (a b c d : ℝ) : 
  (a + c)^2 * (b + d)^2 - 2 * (a * b^2 * c + b * c^2 * d + c * d^2 * a + d * a^2 * b + 4 * a * b * c * d) ≥ 0 ∧ 
  (a + c)^2 * (b + d)^2 - 4 * b * c * (c * d + d * a + a * b) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3948_394899


namespace NUMINAMATH_CALUDE_r_th_term_of_sequence_l3948_394863

/-- Given a sequence where the sum of the first n terms is Sn = 3n + 4n^2,
    prove that the r-th term of the sequence is 8r - 1 -/
theorem r_th_term_of_sequence (n r : ℕ) (Sn : ℕ → ℤ) 
  (h : ∀ n, Sn n = 3*n + 4*n^2) :
  Sn r - Sn (r-1) = 8*r - 1 := by
  sorry

end NUMINAMATH_CALUDE_r_th_term_of_sequence_l3948_394863


namespace NUMINAMATH_CALUDE_perimeter_quarter_circle_square_l3948_394837

/-- The perimeter of a region bounded by quarter circular arcs constructed on each side of a square with side length 4/π is equal to 8. -/
theorem perimeter_quarter_circle_square : 
  let side_length : ℝ := 4 / Real.pi
  let quarter_circle_arc_length : ℝ := (1/4) * (2 * Real.pi * side_length)
  let num_arcs : ℕ := 4
  let perimeter : ℝ := num_arcs * quarter_circle_arc_length
  perimeter = 8 := by sorry

end NUMINAMATH_CALUDE_perimeter_quarter_circle_square_l3948_394837


namespace NUMINAMATH_CALUDE_new_tax_rate_calculation_l3948_394833

theorem new_tax_rate_calculation (original_rate : ℝ) (income : ℝ) (savings : ℝ) : 
  original_rate = 0.46 → 
  income = 36000 → 
  savings = 5040 → 
  (income * original_rate - savings) / income = 0.32 := by
  sorry

end NUMINAMATH_CALUDE_new_tax_rate_calculation_l3948_394833


namespace NUMINAMATH_CALUDE_function_difference_l3948_394805

theorem function_difference (f : ℝ → ℝ) (h : ∀ x, f x = 8^x) :
  ∀ x, f (x + 1) - f x = 7 * f x := by
sorry

end NUMINAMATH_CALUDE_function_difference_l3948_394805


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l3948_394815

theorem least_number_for_divisibility (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ (1101 + k) % 24 = 0) → 
  (∃ m : ℕ, m ≥ 0 ∧ (1101 + m) % 24 = 0 ∧ ∀ l : ℕ, l < m → (1101 + l) % 24 ≠ 0) →
  n = 3 := by
sorry

#eval (1101 + 3) % 24  -- This should evaluate to 0

end NUMINAMATH_CALUDE_least_number_for_divisibility_l3948_394815


namespace NUMINAMATH_CALUDE_inequality_proof_l3948_394871

theorem inequality_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  (1.7 : ℝ)^(0.3 : ℝ) > (0.9 : ℝ)^(3.1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3948_394871


namespace NUMINAMATH_CALUDE_max_value_of_a_l3948_394810

theorem max_value_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 4 → 2 * x^2 - 2 * a * x * y + y^2 ≥ 0) →
  a ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3948_394810


namespace NUMINAMATH_CALUDE_octal_131_equals_binary_1011001_l3948_394876

-- Define octal_to_decimal function
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

-- Define decimal_to_binary function
def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

-- Theorem statement
theorem octal_131_equals_binary_1011001 :
  decimal_to_binary (octal_to_decimal 131) = [1, 0, 1, 1, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_octal_131_equals_binary_1011001_l3948_394876


namespace NUMINAMATH_CALUDE_possible_lists_count_l3948_394824

/-- The number of balls in the bin -/
def num_balls : ℕ := 15

/-- The number of draws Joe makes -/
def num_draws : ℕ := 4

/-- The number of possible lists when drawing with replacement -/
def num_possible_lists : ℕ := num_balls ^ num_draws

/-- Theorem: The number of possible lists is 50625 -/
theorem possible_lists_count : num_possible_lists = 50625 := by
  sorry

end NUMINAMATH_CALUDE_possible_lists_count_l3948_394824


namespace NUMINAMATH_CALUDE_william_napkins_before_l3948_394822

/-- The number of napkins William had before receiving napkins from Olivia and Amelia. -/
def napkins_before : ℕ := sorry

/-- The number of napkins Olivia gave to William. -/
def olivia_napkins : ℕ := 10

/-- The number of napkins Amelia gave to William. -/
def amelia_napkins : ℕ := 2 * olivia_napkins

/-- The total number of napkins William has now. -/
def total_napkins : ℕ := 45

theorem william_napkins_before :
  napkins_before = total_napkins - (olivia_napkins + amelia_napkins) :=
by sorry

end NUMINAMATH_CALUDE_william_napkins_before_l3948_394822


namespace NUMINAMATH_CALUDE_unique_divisible_by_101_l3948_394883

theorem unique_divisible_by_101 : ∃! n : ℕ, 
  201300 ≤ n ∧ n < 201400 ∧ n % 101 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_101_l3948_394883


namespace NUMINAMATH_CALUDE_pool_area_is_30_l3948_394848

/-- The surface area of a rectangular pool -/
def pool_surface_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The surface area of a rectangular pool with width 3 meters and length 10 meters is 30 square meters -/
theorem pool_area_is_30 : pool_surface_area 3 10 = 30 := by
  sorry

end NUMINAMATH_CALUDE_pool_area_is_30_l3948_394848


namespace NUMINAMATH_CALUDE_specific_trapezoid_height_l3948_394804

/-- A trapezoid with given dimensions -/
structure Trapezoid where
  leg1 : ℝ
  leg2 : ℝ
  base1 : ℝ
  base2 : ℝ

/-- The height of a trapezoid -/
def trapezoidHeight (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating the height of the specific trapezoid -/
theorem specific_trapezoid_height :
  let t : Trapezoid := { leg1 := 6, leg2 := 8, base1 := 4, base2 := 14 }
  trapezoidHeight t = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_height_l3948_394804


namespace NUMINAMATH_CALUDE_monday_temp_value_l3948_394821

/-- The average temperature for a week -/
def average_temp : ℝ := 99

/-- The number of days in a week -/
def num_days : ℕ := 7

/-- The temperatures for 6 days of the week -/
def known_temps : List ℝ := [99.1, 98.7, 99.3, 99.8, 99, 98.9]

/-- The temperature on Monday -/
def monday_temp : ℝ := num_days * average_temp - known_temps.sum

theorem monday_temp_value : monday_temp = 98.2 := by sorry

end NUMINAMATH_CALUDE_monday_temp_value_l3948_394821


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l3948_394881

theorem rectangular_field_dimensions (m : ℝ) : 
  (2 * m + 9) * (m - 4) = 88 → m = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l3948_394881


namespace NUMINAMATH_CALUDE_video_game_rounds_l3948_394829

/-- The number of points earned per win in the video game competition. -/
def points_per_win : ℕ := 5

/-- The number of points Vlad scored. -/
def vlad_points : ℕ := 64

/-- The total points scored by both players. -/
def total_points : ℕ := 150

/-- Taro's points in terms of the total points. -/
def taro_points (P : ℕ) : ℤ := (3 * P) / 5 - 4

theorem video_game_rounds :
  (total_points = taro_points total_points + vlad_points) →
  (total_points / points_per_win = 30) := by
sorry

end NUMINAMATH_CALUDE_video_game_rounds_l3948_394829


namespace NUMINAMATH_CALUDE_movie_of_the_year_fraction_l3948_394898

/-- The required fraction for a film to be considered for "movie of the year" -/
def required_fraction (total_members : ℕ) (min_lists : ℚ) : ℚ :=
  min_lists / total_members

/-- Theorem stating the required fraction for the Cinematic Academy's "movie of the year" consideration -/
theorem movie_of_the_year_fraction :
  required_fraction 765 (191.25 : ℚ) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_movie_of_the_year_fraction_l3948_394898


namespace NUMINAMATH_CALUDE_toris_initial_height_l3948_394808

/-- Given Tori's growth and current height, prove her initial height --/
theorem toris_initial_height (growth : ℝ) (current_height : ℝ) 
  (h1 : growth = 2.86)
  (h2 : current_height = 7.26) :
  current_height - growth = 4.40 := by
  sorry

end NUMINAMATH_CALUDE_toris_initial_height_l3948_394808


namespace NUMINAMATH_CALUDE_cubic_equation_root_sum_l3948_394879

/-- Given a cubic equation with roots a, b, c and parameter k, prove that k = 5 -/
theorem cubic_equation_root_sum (k : ℝ) (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - (k+1)*x^2 + k*x + 12 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  (a - 2)^3 + (b - 2)^3 + (c - 2)^3 = -18 →
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_sum_l3948_394879


namespace NUMINAMATH_CALUDE_count_valid_numbers_l3948_394844

/-- A function that generates all valid numbers from digits 1, 2, and 3 without repetition -/
def validNumbers : List ℕ :=
  [1, 2, 3, 12, 13, 21, 23, 31, 32, 123, 132, 213, 231, 312, 321]

/-- The count of natural numbers composed of digits 1, 2, and 3 without repetition -/
theorem count_valid_numbers : validNumbers.length = 15 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l3948_394844


namespace NUMINAMATH_CALUDE_part_one_part_two_l3948_394802

/-- Given expressions for A and B -/
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b - 1

/-- Theorem for part 1 -/
theorem part_one (a b : ℝ) :
  4 * A a b - (3 * A a b - 2 * B a b) = 5 * a * b - 2 * a - 3 := by sorry

/-- Theorem for part 2 -/
theorem part_two (b : ℝ) :
  (∀ a : ℝ, A a b + 2 * B a b = A 0 b + 2 * B 0 b) → b = 2/5 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3948_394802


namespace NUMINAMATH_CALUDE_therapy_hours_is_five_l3948_394831

/-- Represents the cost structure and billing for a psychologist's therapy sessions. -/
structure TherapyCost where
  firstHourCost : ℕ
  additionalHourCost : ℕ
  firstHourPremium : ℕ
  twoHourTotal : ℕ
  someHoursTotal : ℕ

/-- Calculates the number of therapy hours given the cost structure and total charge. -/
def calculateTherapyHours (cost : TherapyCost) : ℕ :=
  sorry

/-- Theorem stating that given the specific cost structure, the calculated therapy hours is 5. -/
theorem therapy_hours_is_five (cost : TherapyCost)
  (h1 : cost.firstHourCost = cost.additionalHourCost + cost.firstHourPremium)
  (h2 : cost.firstHourPremium = 25)
  (h3 : cost.twoHourTotal = 115)
  (h4 : cost.someHoursTotal = 250) :
  calculateTherapyHours cost = 5 :=
sorry

end NUMINAMATH_CALUDE_therapy_hours_is_five_l3948_394831


namespace NUMINAMATH_CALUDE_calculation_difference_l3948_394861

/-- The correct calculation of 12 - (3 × 2) + 4 -/
def H : Int := 12 - (3 * 2) + 4

/-- The incorrect calculation of 12 - 3 × 2 + 4 (ignoring parentheses) -/
def P : Int := 12 - 3 * 2 + 4

/-- The difference between the correct and incorrect calculations -/
def difference : Int := H - P

/-- Theorem stating that the difference between the correct and incorrect calculations is -12 -/
theorem calculation_difference : difference = -12 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l3948_394861


namespace NUMINAMATH_CALUDE_intersection_distance_l3948_394895

theorem intersection_distance (a b : ℤ) (k : ℝ) : 
  k = a + Real.sqrt b →
  (k + 4) / k = Real.sqrt 5 →
  a + b = 6 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_l3948_394895


namespace NUMINAMATH_CALUDE_remainder_of_power_minus_seven_l3948_394839

theorem remainder_of_power_minus_seven (n : Nat) : (10^23 - 7) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_minus_seven_l3948_394839


namespace NUMINAMATH_CALUDE_divisors_of_86400000_l3948_394886

/-- The number of divisors of 86,400,000 -/
def num_divisors : ℕ := 264

/-- The sum of all divisors of 86,400,000 -/
def sum_divisors : ℕ := 319823280

/-- The prime factorization of 86,400,000 -/
def n : ℕ := 2^10 * 3^3 * 5^5

theorem divisors_of_86400000 :
  (∃ (d : Finset ℕ), d.card = num_divisors ∧ 
    (∀ x : ℕ, x ∈ d ↔ x ∣ n) ∧
    d.sum id = sum_divisors) :=
sorry

end NUMINAMATH_CALUDE_divisors_of_86400000_l3948_394886


namespace NUMINAMATH_CALUDE_composition_equation_solution_l3948_394860

/-- Given functions f and g, prove that if f(g(a)) = 4, then a = 3/4 -/
theorem composition_equation_solution (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = (2*x - 1) / 3 + 2)
  (hg : ∀ x, g x = 5 - 2*x)
  (h : f (g a) = 4) : 
  a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l3948_394860


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l3948_394817

theorem half_abs_diff_squares_20_15 : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l3948_394817


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3948_394852

-- Problem 1
theorem problem_1 : (1) - 1^2 + 16 / (-4)^2 * (-3 - 1) = -5 := by sorry

-- Problem 2
theorem problem_2 : 5 * (5/8) - 2 * (-5/8) - 7 * (5/8) = 0 := by sorry

-- Problem 3
theorem problem_3 (x y : ℝ) : x - 3*y - (-3*x + 4*y) = 4*x - 7*y := by sorry

-- Problem 4
theorem problem_4 (a b : ℝ) : 3*a - 4*(a - 3/2*b) - 2*(4*b + 5*a) = -11*a - 2*b := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3948_394852


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3948_394846

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2 * Real.sqrt 5 - 1
  (1 / (x^2 + 2*x + 1)) * (1 + 3 / (x - 1)) / ((x + 2) / (x^2 - 1)) = Real.sqrt 5 / 10 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3948_394846


namespace NUMINAMATH_CALUDE_contrapositive_theorem_negation_theorem_l3948_394869

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem contrapositive_theorem (a b : ℝ) :
  (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M) :=
sorry

theorem negation_theorem :
  (∃ x : ℝ, x^2 - x - 1 > 0) ↔ ¬(∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_theorem_negation_theorem_l3948_394869


namespace NUMINAMATH_CALUDE_average_age_of_four_students_l3948_394864

theorem average_age_of_four_students
  (total_students : ℕ)
  (average_age_all : ℝ)
  (num_group1 : ℕ)
  (average_age_group1 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 9)
  (h4 : average_age_group1 = 16)
  (h5 : age_last_student = 25)
  : (total_students * average_age_all - num_group1 * average_age_group1 - age_last_student) / (total_students - num_group1 - 1) = 14 :=
by
  sorry

#check average_age_of_four_students

end NUMINAMATH_CALUDE_average_age_of_four_students_l3948_394864


namespace NUMINAMATH_CALUDE_no_solution_equations_l3948_394840

theorem no_solution_equations :
  (∀ x : ℝ, (x - 5)^2 ≠ -1) ∧
  (∀ x : ℝ, |2*x| + 3 ≠ 0) ∧
  (∃ x : ℝ, Real.sqrt (x + 3) - 1 = 0) ∧
  (∃ x : ℝ, Real.sqrt (4 - x) - 3 = 0) ∧
  (∃ x : ℝ, |2*x| - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_equations_l3948_394840


namespace NUMINAMATH_CALUDE_solution_line_correct_l3948_394862

/-- Given two lines in the plane -/
def line1 : ℝ → ℝ → Prop := λ x y => 4*x + 3*y - 1 = 0
def line2 : ℝ → ℝ → Prop := λ x y => x + 2*y + 1 = 0

/-- The line to which our solution should be perpendicular -/
def perp_line : ℝ → ℝ → Prop := λ x y => x - 2*y - 1 = 0

/-- The proposed solution line -/
def solution_line : ℝ → ℝ → Prop := λ x y => 2*x + y - 1 = 0

/-- The intersection point of line1 and line2 -/
def intersection_point : ℝ × ℝ := (1, -1)

theorem solution_line_correct :
  (∀ x y, line1 x y ∧ line2 x y → solution_line x y) ∧
  (∀ m₁ m₂, (∀ x y, perp_line x y ↔ y = m₁ * x + m₂) →
            (∀ x y, solution_line x y ↔ y = (-1/m₁) * x + m₂) →
            m₁ * (-1/m₁) = -1) :=
sorry

end NUMINAMATH_CALUDE_solution_line_correct_l3948_394862


namespace NUMINAMATH_CALUDE_dumbbell_distribution_impossible_l3948_394807

def dumbbell_weights : List ℕ := [4, 5, 6, 9, 10, 11, 14, 19, 23, 24]

theorem dumbbell_distribution_impossible :
  ¬ ∃ (rack1 rack2 rack3 : List ℕ),
    (rack1 ++ rack2 ++ rack3).toFinset = dumbbell_weights.toFinset ∧
    (rack1.sum : ℚ) * 2 = rack2.sum ∧
    (rack2.sum : ℚ) * 2 = rack3.sum :=
by sorry

end NUMINAMATH_CALUDE_dumbbell_distribution_impossible_l3948_394807


namespace NUMINAMATH_CALUDE_icosahedral_die_expected_digits_l3948_394873

/-- The expected number of digits when rolling a fair icosahedral die -/
def expected_digits : ℝ := 1.55

/-- The number of faces on an icosahedral die -/
def num_faces : ℕ := 20

/-- The number of one-digit faces on the die -/
def one_digit_faces : ℕ := 9

/-- The number of two-digit faces on the die -/
def two_digit_faces : ℕ := 11

theorem icosahedral_die_expected_digits :
  expected_digits = (one_digit_faces : ℝ) / num_faces + 2 * (two_digit_faces : ℝ) / num_faces :=
sorry

end NUMINAMATH_CALUDE_icosahedral_die_expected_digits_l3948_394873


namespace NUMINAMATH_CALUDE_even_function_sum_a_b_l3948_394882

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 3) * x + 3

-- Define the property of being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Main theorem
theorem even_function_sum_a_b :
  ∀ a b : ℝ,
  (∀ x, x ∈ Set.Icc (2 * a - 3) (4 - a) → f a b x = f a b (-x)) →
  a + b = 2 :=
by sorry

end NUMINAMATH_CALUDE_even_function_sum_a_b_l3948_394882


namespace NUMINAMATH_CALUDE_correct_time_exists_l3948_394875

/-- Represents the position of a watch hand on the face of the watch -/
def HandPosition := ℝ

/-- Represents the angle of rotation for the watch dial -/
def DialRotation := ℝ

/-- Represents a point in time within a 24-hour period -/
def TimePoint := ℝ

/-- A watch with fixed hour and minute hands -/
structure Watch where
  hourHand : HandPosition
  minuteHand : HandPosition

/-- Calculates the correct angle between hour and minute hands for a given time -/
noncomputable def correctAngle (t : TimePoint) : ℝ :=
  sorry

/-- Calculates the actual angle between hour and minute hands for a given watch and dial rotation -/
noncomputable def actualAngle (w : Watch) (r : DialRotation) : ℝ :=
  sorry

/-- States that for any watch with fixed hands, there exists a dial rotation
    such that the watch shows the correct time at least once in a 24-hour period -/
theorem correct_time_exists (w : Watch) :
  ∃ r : DialRotation, ∃ t : TimePoint, actualAngle w r = correctAngle t :=
sorry

end NUMINAMATH_CALUDE_correct_time_exists_l3948_394875


namespace NUMINAMATH_CALUDE_sqrt_expressions_l3948_394816

theorem sqrt_expressions (a b : ℝ) (h1 : a = Real.sqrt 5 + Real.sqrt 3) (h2 : b = Real.sqrt 5 - Real.sqrt 3) :
  (a + b = 2 * Real.sqrt 5) ∧ (a * b = 2) ∧ (a^2 + a*b + b^2 = 18) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_l3948_394816


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3948_394825

theorem decimal_to_fraction : (3.75 : ℚ) = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3948_394825


namespace NUMINAMATH_CALUDE_consecutive_primes_sum_composite_l3948_394835

theorem consecutive_primes_sum_composite (p₁ p₂ q : ℕ) : 
  Nat.Prime p₁ → Nat.Prime p₂ → 
  Odd p₁ → Odd p₂ → 
  p₁ < p₂ → 
  ¬∃k, Nat.Prime k ∧ p₁ < k ∧ k < p₂ →
  p₁ + p₂ = 2 * q → 
  ¬(Nat.Prime q) := by
sorry

end NUMINAMATH_CALUDE_consecutive_primes_sum_composite_l3948_394835


namespace NUMINAMATH_CALUDE_chair_arrangement_l3948_394819

theorem chair_arrangement (total_chairs : Nat) (h1 : total_chairs = 49) :
  (∃! (rows columns : Nat), rows ≥ 2 ∧ columns ≥ 2 ∧ rows * columns = total_chairs) :=
by sorry

end NUMINAMATH_CALUDE_chair_arrangement_l3948_394819


namespace NUMINAMATH_CALUDE_child_ticket_price_l3948_394845

/-- Given the following information about a movie theater's ticket sales:
  - Total tickets sold is 900
  - Total revenue is $5,100
  - Adult ticket price is $7
  - Number of adult tickets sold is 500
  Prove that the price of a child's ticket is $4. -/
theorem child_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (adult_price : ℕ) 
  (adult_tickets : ℕ) 
  (h1 : total_tickets = 900) 
  (h2 : total_revenue = 5100) 
  (h3 : adult_price = 7) 
  (h4 : adult_tickets = 500) : 
  (total_revenue - adult_price * adult_tickets) / (total_tickets - adult_tickets) = 4 := by
sorry


end NUMINAMATH_CALUDE_child_ticket_price_l3948_394845


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3948_394859

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def units_digit (n : ℕ) : ℕ := n % 10

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem unique_three_digit_number :
  ∃! n : ℕ, is_three_digit n ∧
            units_digit n = 4 ∧
            hundreds_digit n = 5 ∧
            tens_digit n % 2 = 0 ∧
            n % 8 = 0 ∧
            n = 544 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3948_394859


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l3948_394868

/-- The volume of a tetrahedron with vertices on coordinate axes -/
theorem tetrahedron_volume (d e f : ℝ) : 
  d > 0 → e > 0 → f > 0 →  -- Positive coordinates
  d^2 + e^2 = 49 →         -- DE = 7
  e^2 + f^2 = 64 →         -- EF = 8
  f^2 + d^2 = 81 →         -- FD = 9
  (1/6 : ℝ) * d * e * f = 4 * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_volume_l3948_394868


namespace NUMINAMATH_CALUDE_function_property_l3948_394818

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x + 3 else a * x + b

theorem function_property (a b : ℝ) :
  (∀ x₁ : ℝ, x₁ ≠ 0 → ∃! x₂ : ℝ, x₁ ≠ x₂ ∧ f a b x₁ = f a b x₂) →
  f a b (2 * a) = f a b (3 * b) →
  a + b = -Real.sqrt 6 / 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3948_394818


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3948_394827

/-- A quadratic radical is considered simpler if it cannot be further simplified by factoring out perfect squares or simplifying fractions. -/
def is_simplest_quadratic_radical (x : ℝ) (options : List ℝ) : Prop :=
  x ∈ options ∧ 
  (∀ y ∈ options, x ≠ y → ∃ (n : ℕ) (m : ℚ), n > 1 ∧ y = n • (Real.sqrt m) ∨ ∃ (a b : ℚ), b ≠ 1 ∧ y = (Real.sqrt a) / b)

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 7) [Real.sqrt 12, Real.sqrt 7, Real.sqrt (2/3), Real.sqrt 0.2] :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3948_394827


namespace NUMINAMATH_CALUDE_leading_zeros_in_decimal_representation_l3948_394801

theorem leading_zeros_in_decimal_representation (n : ℕ) (m : ℕ) :
  (∃ k : ℕ, (1 : ℚ) / (2^7 * 5^3) = (k : ℚ) / 10^n ∧ 
   k ≠ 0 ∧ k < 10^m) → n - m = 5 := by
  sorry

end NUMINAMATH_CALUDE_leading_zeros_in_decimal_representation_l3948_394801


namespace NUMINAMATH_CALUDE_equivalent_statements_l3948_394880

theorem equivalent_statements :
  (∀ x : ℝ, x ≥ 0 → x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0 → x < 0) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_statements_l3948_394880


namespace NUMINAMATH_CALUDE_triangle_abc_is_right_triangle_l3948_394847

/-- Given a triangle ABC where:
    - The sides opposite to angles A, B, C are a, b, c respectively
    - A = π/3
    - a = √3
    - b = 1
    Prove that C = π/2, i.e., the triangle is a right triangle -/
theorem triangle_abc_is_right_triangle (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b = 1 →
  A = π / 3 →
  a / Real.sin A = b / Real.sin B →
  A + B + C = π →
  C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_is_right_triangle_l3948_394847


namespace NUMINAMATH_CALUDE_johnson_work_completion_l3948_394806

/-- Johnson and Vincent's Work Completion Problem -/
theorem johnson_work_completion (vincent_days : ℕ) (together_days : ℕ) (johnson_days : ℕ) : 
  vincent_days = 40 → together_days = 8 → johnson_days = 10 →
  (1 : ℚ) / johnson_days + (1 : ℚ) / vincent_days = (1 : ℚ) / together_days := by
  sorry

#check johnson_work_completion

end NUMINAMATH_CALUDE_johnson_work_completion_l3948_394806


namespace NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l3948_394866

theorem power_three_nineteen_mod_ten : 3^19 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_three_nineteen_mod_ten_l3948_394866


namespace NUMINAMATH_CALUDE_catch_up_equation_l3948_394888

/-- The number of days it takes for a good horse to catch up with a slow horse -/
def catch_up_days (good_horse_speed slow_horse_speed : ℕ) (head_start : ℕ) : ℕ → Prop :=
  λ x => good_horse_speed * x = slow_horse_speed * x + slow_horse_speed * head_start

/-- Theorem stating the equation for the number of days it takes for the good horse to catch up -/
theorem catch_up_equation :
  let good_horse_speed := 240
  let slow_horse_speed := 150
  let head_start := 12
  ∃ x : ℕ, catch_up_days good_horse_speed slow_horse_speed head_start x :=
by
  sorry

end NUMINAMATH_CALUDE_catch_up_equation_l3948_394888


namespace NUMINAMATH_CALUDE_coins_taken_out_l3948_394851

/-- The number of coins Tina put in during the first hour -/
def first_hour_coins : ℕ := 20

/-- The number of coins Tina put in during each of the second and third hours -/
def second_third_hour_coins : ℕ := 30

/-- The number of coins Tina put in during the fourth hour -/
def fourth_hour_coins : ℕ := 40

/-- The number of coins left in the jar after the fifth hour -/
def coins_left : ℕ := 100

/-- The total number of coins Tina put in the jar -/
def total_coins_in : ℕ := first_hour_coins + 2 * second_third_hour_coins + fourth_hour_coins

/-- Theorem: The number of coins Tina's mother took out is equal to the total number of coins Tina put in minus the number of coins left in the jar after the fifth hour -/
theorem coins_taken_out : total_coins_in - coins_left = 20 := by
  sorry

end NUMINAMATH_CALUDE_coins_taken_out_l3948_394851


namespace NUMINAMATH_CALUDE_sin_negative_1740_degrees_l3948_394842

theorem sin_negative_1740_degrees : Real.sin ((-1740 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1740_degrees_l3948_394842


namespace NUMINAMATH_CALUDE_can_tower_sum_l3948_394877

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => a₁ + d * i)

theorem can_tower_sum :
  let a₁ : ℤ := 34
  let aₙ : ℤ := 4
  let d : ℤ := -6
  let n : ℕ := 6
  let sequence := arithmetic_sequence a₁ d n
  (sequence.length = n) ∧
  (sequence.head? = some a₁) ∧
  (sequence.getLast? = some aₙ) ∧
  (∀ i, 0 < i → i < n - 1 → sequence[i+1]! - sequence[i]! = d) →
  sequence.sum = 114 := by
  sorry

end NUMINAMATH_CALUDE_can_tower_sum_l3948_394877


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l3948_394811

/- Given a point B with three angles around it -/
def point_B (angle_ABC angle_ABD angle_CBD : ℝ) : Prop :=
  /- ∠CBD is a right angle -/
  angle_CBD = 90 ∧
  /- The sum of angles around point B is 200° -/
  angle_ABC + angle_ABD + angle_CBD = 200 ∧
  /- The measure of ∠ABD is 70° -/
  angle_ABD = 70

/- Theorem statement -/
theorem angle_ABC_measure :
  ∀ (angle_ABC angle_ABD angle_CBD : ℝ),
  point_B angle_ABC angle_ABD angle_CBD →
  angle_ABC = 40 := by
sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l3948_394811
