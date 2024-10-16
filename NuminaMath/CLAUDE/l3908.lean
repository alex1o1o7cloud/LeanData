import Mathlib

namespace NUMINAMATH_CALUDE_triangle_properties_l3908_390856

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.cos t.C = (2 * t.a + t.c) * Real.cos (π - t.B))
  (h2 : t.b = Real.sqrt 13)
  (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 4) :
  t.B = 2 * π / 3 ∧ t.a + t.c = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3908_390856


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l3908_390887

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 6)
  (h₃ : contact_distance = 20) :
  Real.sqrt ((r₁ - r₂)^2 + contact_distance^2) = 2 * Real.sqrt 104 :=
by sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l3908_390887


namespace NUMINAMATH_CALUDE_dog_roaming_area_l3908_390885

/-- The area a dog can roam when tied to the corner of a rectangular shed --/
theorem dog_roaming_area (shed_length shed_width leash_length : ℝ) 
  (h1 : shed_length = 4)
  (h2 : shed_width = 3)
  (h3 : leash_length = 4) : 
  let area := (3/4) * Real.pi * leash_length^2 + (1/4) * Real.pi * 1^2
  area = 12.25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_dog_roaming_area_l3908_390885


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3908_390864

/-- The interest rate at which B lent to C, given the conditions of the problem -/
def interest_rate_B_to_C (principal : ℚ) (rate_A_to_B : ℚ) (years : ℚ) (gain_B : ℚ) : ℚ :=
  let interest_A_to_B := principal * rate_A_to_B * years
  let total_interest_B_from_C := interest_A_to_B + gain_B
  (total_interest_B_from_C * 100) / (principal * years)

theorem interest_rate_calculation (principal : ℚ) (rate_A_to_B : ℚ) (years : ℚ) (gain_B : ℚ) :
  principal = 2000 →
  rate_A_to_B = 15 / 100 →
  years = 4 →
  gain_B = 160 →
  interest_rate_B_to_C principal rate_A_to_B years gain_B = 17 / 100 := by
  sorry

#eval interest_rate_B_to_C 2000 (15/100) 4 160

end NUMINAMATH_CALUDE_interest_rate_calculation_l3908_390864


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3908_390863

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop := sorry

theorem lines_perpendicular_to_plane_are_parallel 
  (l1 l2 : Line3D) (p : Plane3D) :
  perpendicular l1 p → perpendicular l2 p → parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l3908_390863


namespace NUMINAMATH_CALUDE_license_plate_count_l3908_390866

/-- The number of consonants in the alphabet. -/
def num_consonants : ℕ := 20

/-- The number of vowels in the alphabet (including Y). -/
def num_vowels : ℕ := 6

/-- The number of digits (0-9). -/
def num_digits : ℕ := 10

/-- The total number of letters in the alphabet. -/
def num_letters : ℕ := 26

/-- The number of unique five-character license plates with the sequence:
    consonant, vowel, consonant, digit, any letter. -/
def num_license_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits * num_letters

theorem license_plate_count :
  num_license_plates = 624000 :=
sorry

end NUMINAMATH_CALUDE_license_plate_count_l3908_390866


namespace NUMINAMATH_CALUDE_C_excircle_touches_circumcircle_l3908_390809

-- Define the basic geometric structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

structure Circle where
  center : Point
  radius : ℝ

-- Define the semiperimeter of a triangle
def semiperimeter (t : Triangle) : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define the C-excircle of a triangle
def C_excircle (t : Triangle) : Circle := sorry

-- Define the circumcircle of a triangle
def circumcircle (t : Triangle) : Circle := sorry

-- Define tangency between two circles
def are_tangent (c1 c2 : Circle) : Prop := sorry

-- Theorem statement
theorem C_excircle_touches_circumcircle 
  (ABC : Triangle) 
  (p : ℝ) 
  (E F : Point) :
  semiperimeter ABC = p →
  E.x ≤ F.x →
  distance ABC.A E + distance E F + distance F ABC.B = distance ABC.A ABC.B →
  distance ABC.C E = p →
  distance ABC.C F = p →
  are_tangent (C_excircle ABC) (circumcircle (Triangle.mk E F ABC.C)) :=
sorry

end NUMINAMATH_CALUDE_C_excircle_touches_circumcircle_l3908_390809


namespace NUMINAMATH_CALUDE_division_remainder_l3908_390837

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 127 →
  divisor = 25 →
  quotient = 5 →
  dividend = divisor * quotient + remainder →
  remainder = 2 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3908_390837


namespace NUMINAMATH_CALUDE_problem_statement_l3908_390854

theorem problem_statement (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : b^3 + b ≤ a - a^3) :
  (b < a ∧ a < 1) ∧ a^2 + b^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3908_390854


namespace NUMINAMATH_CALUDE_sqrt_x_plus_2_meaningful_l3908_390838

theorem sqrt_x_plus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 2) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_2_meaningful_l3908_390838


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l3908_390800

theorem triangle_perimeter_range (a b c : ℝ) : 
  a = 2 → b = 7 → (a + b > c ∧ b + c > a ∧ c + a > b) → 
  14 < a + b + c ∧ a + b + c < 18 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l3908_390800


namespace NUMINAMATH_CALUDE_perimeter_of_square_figure_l3908_390890

/-- A figure composed of four identical squares -/
structure SquareFigure where
  -- Side length of each square
  side_length : ℝ
  -- Total area of the figure
  total_area : ℝ
  -- Number of vertical segments
  vertical_segments : ℕ
  -- Number of horizontal segments
  horizontal_segments : ℕ
  -- Condition: Total area is the area of four squares
  area_condition : total_area = 4 * side_length ^ 2

/-- The perimeter of the square figure -/
def perimeter (f : SquareFigure) : ℝ :=
  (f.vertical_segments + f.horizontal_segments) * f.side_length

/-- Theorem: If the total area is 144 cm² and the figure has 4 vertical and 6 horizontal segments,
    then the perimeter is 60 cm -/
theorem perimeter_of_square_figure (f : SquareFigure) 
    (h_area : f.total_area = 144) 
    (h_vertical : f.vertical_segments = 4) 
    (h_horizontal : f.horizontal_segments = 6) : 
    perimeter f = 60 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_square_figure_l3908_390890


namespace NUMINAMATH_CALUDE_range_of_c_l3908_390891

theorem range_of_c (a b c : ℝ) :
  (∀ x : ℝ, (Real.sqrt (2 * x^2 + a * x + b) > x - c) ↔ (x ≤ 0 ∨ x > 1)) →
  c ∈ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_c_l3908_390891


namespace NUMINAMATH_CALUDE_pirate_treasure_l3908_390893

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_l3908_390893


namespace NUMINAMATH_CALUDE_dice_edge_length_l3908_390831

theorem dice_edge_length (volume : ℝ) (edge_length : ℝ) :
  volume = 8 →
  edge_length^3 = volume →
  edge_length * 10 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_dice_edge_length_l3908_390831


namespace NUMINAMATH_CALUDE_expression_evaluation_l3908_390874

theorem expression_evaluation : (2000^2 : ℝ) / (402^2 - 398^2) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3908_390874


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3908_390818

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define what it means for a quadratic radical to be in its simplest form
def isSimplestQuadraticRadical (n : ℚ) : Prop :=
  ∃ (a : ℕ), isPrime a ∧ n = (a : ℚ).sqrt

-- Theorem statement
theorem simplest_quadratic_radical :
  let options : List ℚ := [9, 7, 20, (1/3 : ℚ)]
  ∃ (x : ℚ), x ∈ options ∧ isSimplestQuadraticRadical x ∧
    ∀ (y : ℚ), y ∈ options → y ≠ x → ¬(isSimplestQuadraticRadical y) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3908_390818


namespace NUMINAMATH_CALUDE_hardcover_count_l3908_390873

/-- Represents the purchase of a book series -/
structure BookPurchase where
  total_volumes : ℕ
  paperback_price : ℕ
  hardcover_price : ℕ
  total_cost : ℕ

/-- Theorem stating that under given conditions, the number of hardcover books is 6 -/
theorem hardcover_count (purchase : BookPurchase)
  (h_total : purchase.total_volumes = 8)
  (h_paperback : purchase.paperback_price = 10)
  (h_hardcover : purchase.hardcover_price = 20)
  (h_cost : purchase.total_cost = 140) :
  ∃ (h : ℕ), h = 6 ∧ 
    h * purchase.hardcover_price + (purchase.total_volumes - h) * purchase.paperback_price = purchase.total_cost :=
by sorry

end NUMINAMATH_CALUDE_hardcover_count_l3908_390873


namespace NUMINAMATH_CALUDE_petrol_price_equation_l3908_390861

/-- The original price of petrol per gallon -/
def P : ℝ := sorry

/-- The reduced price is 90% of the original price -/
def reduced_price : ℝ := 0.9 * P

/-- The equation representing the relationship between the original and reduced prices -/
theorem petrol_price_equation : 250 / reduced_price = 250 / P + 5 := by sorry

end NUMINAMATH_CALUDE_petrol_price_equation_l3908_390861


namespace NUMINAMATH_CALUDE_water_storage_problem_l3908_390879

/-- Calculates the total gallons of water stored given the conditions --/
def total_water_stored (total_jars : ℕ) (jar_sizes : ℕ) : ℚ :=
  let jars_per_size := total_jars / jar_sizes
  let quart_gallons := jars_per_size * (1 / 4 : ℚ)
  let half_gallons := jars_per_size * (1 / 2 : ℚ)
  let full_gallons := jars_per_size * 1
  quart_gallons + half_gallons + full_gallons

/-- Theorem stating that under the given conditions, the total water stored is 42 gallons --/
theorem water_storage_problem :
  total_water_stored 72 3 = 42 := by
  sorry


end NUMINAMATH_CALUDE_water_storage_problem_l3908_390879


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3908_390819

theorem inequality_solution_set (x : ℝ) :
  (|2*x + 1| - 2*|x - 1| > 0) ↔ (x > 0) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3908_390819


namespace NUMINAMATH_CALUDE_park_trees_after_planting_l3908_390867

/-- The total number of dogwood trees after planting -/
def total_trees (current : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  current + planted_today + planted_tomorrow

/-- Theorem: The park will have 100 dogwood trees when the workers are finished -/
theorem park_trees_after_planting :
  total_trees 39 41 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_park_trees_after_planting_l3908_390867


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3908_390812

theorem fourteenth_root_of_unity : 
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (2 * n * π * Complex.I / 14) :=
by sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3908_390812


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l3908_390870

theorem cubic_polynomial_roots (a b c : ℤ) (r₁ r₂ r₃ : ℤ) : 
  (∀ x : ℤ, x^3 + a*x^2 + b*x + c = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (r₁ > 2 ∧ r₂ > 2 ∧ r₃ > 2) →
  a + b + c + 1 = -2009 →
  (r₁ - 1) * (r₂ - 1) * (r₃ - 1) = 2009 →
  a = -58 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l3908_390870


namespace NUMINAMATH_CALUDE_brother_contribution_l3908_390859

/-- The number of wood pieces Alvin needs in total -/
def total_needed : ℕ := 376

/-- The number of wood pieces Alvin's friend gave him -/
def friend_gave : ℕ := 123

/-- The number of wood pieces Alvin still needs to gather -/
def still_needed : ℕ := 117

/-- The number of wood pieces Alvin's brother gave him -/
def brother_gave : ℕ := total_needed - friend_gave - still_needed

theorem brother_contribution : brother_gave = 136 := by
  sorry

end NUMINAMATH_CALUDE_brother_contribution_l3908_390859


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3908_390868

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 < x ∧ x < 9 }

theorem union_of_A_and_B : A ∪ B = { x | -1 ≤ x ∧ x < 9 } := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3908_390868


namespace NUMINAMATH_CALUDE_miles_walked_approx_2250_l3908_390842

/-- Represents a pedometer with a maximum reading before reset --/
structure Pedometer where
  max_reading : ℕ
  steps_per_mile : ℕ

/-- Represents the pedometer readings over a year --/
structure YearlyReading where
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total miles walked based on pedometer data --/
def total_miles_walked (p : Pedometer) (yr : YearlyReading) : ℚ :=
  let total_steps : ℕ := p.max_reading * yr.resets + yr.final_reading + 1
  (total_steps : ℚ) / p.steps_per_mile

/-- Theorem stating that the total miles walked is approximately 2250 --/
theorem miles_walked_approx_2250 (p : Pedometer) (yr : YearlyReading) :
  p.max_reading = 99999 →
  p.steps_per_mile = 1600 →
  yr.resets = 36 →
  yr.final_reading = 25000 →
  2249 < total_miles_walked p yr ∧ total_miles_walked p yr < 2251 :=
sorry

end NUMINAMATH_CALUDE_miles_walked_approx_2250_l3908_390842


namespace NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l3908_390839

theorem intersection_perpendicular_tangents (a : ℝ) (h : a > 0) : 
  ∃ x : ℝ, x ∈ Set.Ioo 0 (Real.pi / 2) ∧ 
  (2 * Real.sin x = a * Real.cos x) ∧
  (2 * Real.cos x) * (-a * Real.sin x) = -1 
  → a = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l3908_390839


namespace NUMINAMATH_CALUDE_average_glasses_per_box_l3908_390860

/-- Proves that given the specified conditions, the average number of glasses per box is 15 -/
theorem average_glasses_per_box : 
  ∀ (small_boxes large_boxes : ℕ),
  small_boxes > 0 →
  large_boxes = small_boxes + 16 →
  12 * small_boxes + 16 * large_boxes = 480 →
  (480 : ℚ) / (small_boxes + large_boxes) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_average_glasses_per_box_l3908_390860


namespace NUMINAMATH_CALUDE_nelly_winning_bid_l3908_390807

-- Define Joe's bid
def joes_bid : ℕ := 160000

-- Define Nelly's bid calculation
def nellys_bid : ℕ := 3 * joes_bid + 2000

-- Theorem to prove
theorem nelly_winning_bid : nellys_bid = 482000 := by
  sorry

end NUMINAMATH_CALUDE_nelly_winning_bid_l3908_390807


namespace NUMINAMATH_CALUDE_cone_volume_gravel_pile_l3908_390852

/-- The volume of a cone with base diameter 10 feet and height 80% of its diameter is 200π/3 cubic feet. -/
theorem cone_volume_gravel_pile :
  let base_diameter : ℝ := 10
  let height_ratio : ℝ := 0.8
  let height : ℝ := height_ratio * base_diameter
  let radius : ℝ := base_diameter / 2
  let volume : ℝ := (1 / 3) * π * radius^2 * height
  volume = 200 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_gravel_pile_l3908_390852


namespace NUMINAMATH_CALUDE_football_game_score_l3908_390877

theorem football_game_score (total_points winning_margin : ℕ) 
  (h1 : total_points = 34) 
  (h2 : winning_margin = 14) : 
  ∃ (panthers_score cougars_score : ℕ), 
    panthers_score + cougars_score = total_points ∧ 
    cougars_score = panthers_score + winning_margin ∧ 
    panthers_score = 10 := by
  sorry

end NUMINAMATH_CALUDE_football_game_score_l3908_390877


namespace NUMINAMATH_CALUDE_first_perfect_square_all_remainders_l3908_390834

theorem first_perfect_square_all_remainders : 
  ∀ n : ℕ, n ≤ 20 → 
    (∃ k ≤ n, k^2 % 10 = 0) ∧ 
    (∃ k ≤ n, k^2 % 10 = 1) ∧ 
    (∃ k ≤ n, k^2 % 10 = 2) ∧ 
    (∃ k ≤ n, k^2 % 10 = 3) ∧ 
    (∃ k ≤ n, k^2 % 10 = 4) ∧ 
    (∃ k ≤ n, k^2 % 10 = 5) ∧ 
    (∃ k ≤ n, k^2 % 10 = 6) ∧ 
    (∃ k ≤ n, k^2 % 10 = 7) ∧ 
    (∃ k ≤ n, k^2 % 10 = 8) ∧ 
    (∃ k ≤ n, k^2 % 10 = 9) ↔ 
    n = 20 :=
by sorry

end NUMINAMATH_CALUDE_first_perfect_square_all_remainders_l3908_390834


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3908_390808

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*m*x + m^2 - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ * (x₂ + x₁) + x₂^2 = 5*m →
  m = (5 - Real.sqrt 13) / 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3908_390808


namespace NUMINAMATH_CALUDE_fraction_problem_l3908_390847

theorem fraction_problem (f : ℚ) : 3 + (1/2) * f * (1/5) * 90 = (1/15) * 90 ↔ f = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3908_390847


namespace NUMINAMATH_CALUDE_quadratic_radical_equivalence_l3908_390897

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem quadratic_radical_equivalence (m : ℕ) :
  (is_prime 2 ∧ is_prime (2023 - m)) → m = 2021 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equivalence_l3908_390897


namespace NUMINAMATH_CALUDE_data_set_property_l3908_390845

theorem data_set_property (m n : ℝ) : 
  (m + n + 9 + 8 + 10) / 5 = 9 →
  ((m^2 + n^2 + 9^2 + 8^2 + 10^2) / 5) - 9^2 = 2 →
  |m - n| = 4 :=
by sorry

end NUMINAMATH_CALUDE_data_set_property_l3908_390845


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l3908_390898

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 15 → 
    (isPalindrome n 2 ∧ isPalindrome n 4) → 
    n ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l3908_390898


namespace NUMINAMATH_CALUDE_train_length_l3908_390828

/-- Calculates the length of a train given its speed, bridge length, and time to cross the bridge. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 90 * (1000 / 3600) → 
  bridge_length = 275 → 
  crossing_time = 30 → 
  train_speed * crossing_time - bridge_length = 475 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3908_390828


namespace NUMINAMATH_CALUDE_two_consecutive_late_charges_l3908_390840

theorem two_consecutive_late_charges (original_bill : ℝ) (late_charge_rate : ℝ) : 
  original_bill = 400 → 
  late_charge_rate = 0.01 → 
  original_bill * (1 + late_charge_rate)^2 = 408.04 := by
sorry


end NUMINAMATH_CALUDE_two_consecutive_late_charges_l3908_390840


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3908_390802

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|) ∧ 
  ¬(|x + 1| + |x - 1| = 2 * |x| → x ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3908_390802


namespace NUMINAMATH_CALUDE_smallest_n_candies_l3908_390869

theorem smallest_n_candies : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n + 6) % 7 = 0 ∧ 
  (n - 9) % 4 = 0 ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m + 6) % 7 = 0 ∧ (m - 9) % 4 = 0 → n ≤ m) ∧
  n = 113 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_candies_l3908_390869


namespace NUMINAMATH_CALUDE_marathon_practice_distance_l3908_390846

/-- Calculates the total distance run given the number of days and miles per day -/
def total_distance (days : ℕ) (miles_per_day : ℕ) : ℕ :=
  days * miles_per_day

/-- Proves that running 8 miles for 9 days results in a total of 72 miles -/
theorem marathon_practice_distance :
  total_distance 9 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_marathon_practice_distance_l3908_390846


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3908_390876

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 74 → a = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3908_390876


namespace NUMINAMATH_CALUDE_employed_female_percentage_l3908_390888

/-- Represents the percentage of a population --/
def Percentage := Finset (Fin 100)

theorem employed_female_percentage
  (total_employed : Percentage)
  (employed_males : Percentage)
  (h1 : total_employed.card = 60)
  (h2 : employed_males.card = 48) :
  (total_employed.card - employed_males.card : ℚ) / total_employed.card * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_employed_female_percentage_l3908_390888


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3908_390815

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x < 1) ∧ 
  (∃ y : ℝ, y < 1 ∧ ¬(y > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3908_390815


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_z_purely_imaginary_iff_l3908_390835

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := m * (3 + Complex.I) - (2 + Complex.I)

-- Theorem 1: z is in the fourth quadrant when 2/3 < m < 1
theorem z_in_fourth_quadrant (m : ℝ) (h1 : 2/3 < m) (h2 : m < 1) :
  (z m).re > 0 ∧ (z m).im < 0 :=
sorry

-- Theorem 2: z is purely imaginary iff m = 2/3
theorem z_purely_imaginary_iff (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2/3 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_z_purely_imaginary_iff_l3908_390835


namespace NUMINAMATH_CALUDE_expression_value_at_three_l3908_390850

theorem expression_value_at_three :
  let f (x : ℝ) := (x^2 - 5*x + 4) / (x - 4)
  f 3 = 2 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l3908_390850


namespace NUMINAMATH_CALUDE_product_inequality_solve_for_a_l3908_390894

-- Part I
theorem product_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 + 1/a) * (1 + 1/b) ≥ 9 := by sorry

-- Part II
theorem solve_for_a (a : ℝ) (h : ∀ x, |x + 3| - |x - a| ≥ 2 ↔ x ≥ 1) :
  a = 2 := by sorry

end NUMINAMATH_CALUDE_product_inequality_solve_for_a_l3908_390894


namespace NUMINAMATH_CALUDE_farmer_pomelo_shipment_l3908_390855

/-- Calculates the total number of dozens of pomelos shipped given the number of boxes and pomelos from last week and the number of boxes shipped this week. -/
def totalDozensShipped (lastWeekBoxes : ℕ) (lastWeekPomelos : ℕ) (thisWeekBoxes : ℕ) : ℕ :=
  let pomelosPerBox := lastWeekPomelos / lastWeekBoxes
  let totalPomelos := lastWeekPomelos + thisWeekBoxes * pomelosPerBox
  totalPomelos / 12

/-- Proves that given 10 boxes containing 240 pomelos in total last week, and 20 boxes shipped this week, the total number of dozens of pomelos shipped is 40. -/
theorem farmer_pomelo_shipment :
  totalDozensShipped 10 240 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_farmer_pomelo_shipment_l3908_390855


namespace NUMINAMATH_CALUDE_train_crossing_time_l3908_390895

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 45 →
  train_speed_kmh = 108 →
  crossing_time = train_length / (train_speed_kmh * (1000 / 3600)) →
  crossing_time = 1.5 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3908_390895


namespace NUMINAMATH_CALUDE_sum_two_smallest_trite_numbers_l3908_390816

def is_trite (n : ℕ+) : Prop :=
  ∃ (d : Fin 12 → ℕ+),
    (∀ i j, i < j → d i < d j) ∧
    d 0 = 1 ∧
    d 11 = n ∧
    (∀ k, k ∣ n ↔ ∃ i, d i = k) ∧
    5 + (d 5) * (d 5 + d 3) = (d 6) * (d 3)

theorem sum_two_smallest_trite_numbers : 
  ∃ (a b : ℕ+), is_trite a ∧ is_trite b ∧ 
  (∀ n : ℕ+, is_trite n → a ≤ n) ∧
  (∀ n : ℕ+, is_trite n ∧ n ≠ a → b ≤ n) ∧
  a + b = 151127 :=
sorry

end NUMINAMATH_CALUDE_sum_two_smallest_trite_numbers_l3908_390816


namespace NUMINAMATH_CALUDE_inequality_proof_l3908_390823

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3908_390823


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_number_l3908_390841

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ        -- Total number of items
  sampleSize : ℕ   -- Size of the sample
  step : ℕ         -- Step size for systematic sampling
  first : ℕ        -- First sample number

/-- Generates the nth sample number in a systematic sample -/
def nthSample (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first + (n - 1) * s.step

/-- Checks if a number is in the sample -/
def isInSample (s : SystematicSample) (num : ℕ) : Prop :=
  ∃ n : ℕ, n ≤ s.sampleSize ∧ nthSample s n = num

theorem systematic_sample_fourth_number
  (s : SystematicSample)
  (h_total : s.total = 52)
  (h_size : s.sampleSize = 4)
  (h_7 : isInSample s 7)
  (h_33 : isInSample s 33)
  (h_46 : isInSample s 46) :
  isInSample s 20 :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_number_l3908_390841


namespace NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l3908_390829

def repeating_decimal_one_third : ℚ := 1/3

theorem one_minus_repeating_third_equals_two_thirds :
  1 - repeating_decimal_one_third = 2/3 := by sorry

end NUMINAMATH_CALUDE_one_minus_repeating_third_equals_two_thirds_l3908_390829


namespace NUMINAMATH_CALUDE_ladder_height_correct_l3908_390892

/-- The height of the ceiling in centimeters -/
def ceiling_height : ℝ := 300

/-- The distance of the light fixture below the ceiling in centimeters -/
def fixture_below_ceiling : ℝ := 15

/-- Bob's height in centimeters -/
def bob_height : ℝ := 170

/-- The distance Bob can reach above his head in centimeters -/
def bob_reach : ℝ := 52

/-- The height of the ladder in centimeters -/
def ladder_height : ℝ := 63

theorem ladder_height_correct :
  ceiling_height - fixture_below_ceiling = bob_height + bob_reach + ladder_height := by
  sorry

end NUMINAMATH_CALUDE_ladder_height_correct_l3908_390892


namespace NUMINAMATH_CALUDE_correct_fraction_l3908_390820

theorem correct_fraction (number : ℚ) (incorrect_fraction : ℚ) (difference : ℚ) :
  number = 96 →
  incorrect_fraction = 5 / 6 →
  incorrect_fraction * number = number * x + difference →
  difference = 50 →
  x = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_fraction_l3908_390820


namespace NUMINAMATH_CALUDE_ammonia_molecular_weight_l3908_390836

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The number of nitrogen atoms in an ammonia molecule -/
def nitrogen_count : ℕ := 1

/-- The number of hydrogen atoms in an ammonia molecule -/
def hydrogen_count : ℕ := 3

/-- The molecular weight of ammonia in atomic mass units (amu) -/
def ammonia_weight : ℝ := nitrogen_weight * nitrogen_count + hydrogen_weight * hydrogen_count

theorem ammonia_molecular_weight :
  ammonia_weight = 17.034 := by sorry

end NUMINAMATH_CALUDE_ammonia_molecular_weight_l3908_390836


namespace NUMINAMATH_CALUDE_perpendicular_planes_perpendicular_lines_parallel_l3908_390889

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (planePerpendicular : Plane → Plane → Prop)

-- Theorem 1: If a line is perpendicular to a plane and contained in another plane,
-- then the two planes are perpendicular
theorem perpendicular_planes
  (a : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : contains β a) :
  planePerpendicular α β :=
sorry

-- Theorem 2: If two lines are perpendicular to the same plane,
-- then the lines are parallel
theorem perpendicular_lines_parallel
  (a b : Line) (α : Plane)
  (h1 : perpendicular a α)
  (h2 : perpendicular b α) :
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_perpendicular_lines_parallel_l3908_390889


namespace NUMINAMATH_CALUDE_mean_median_difference_l3908_390851

def is_valid_set (x d : ℕ) : Prop :=
  x > 0 ∧ x + 2 > 0 ∧ x + 4 > 0 ∧ x + 7 > 0 ∧ x + d > 0

def median (x : ℕ) : ℕ := x + 4

def mean (x d : ℕ) : ℚ :=
  (x + (x + 2) + (x + 4) + (x + 7) + (x + d)) / 5

theorem mean_median_difference (x d : ℕ) :
  is_valid_set x d →
  mean x d = (median x : ℚ) + 5 →
  d = 32 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l3908_390851


namespace NUMINAMATH_CALUDE_rationalize_sqrt_5_12_l3908_390805

theorem rationalize_sqrt_5_12 : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_5_12_l3908_390805


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l3908_390826

theorem coconut_grove_problem (x : ℝ) 
  (yield_1 : (x + 2) * 40 = (x + 2) * 40)
  (yield_2 : x * 120 = x * 120)
  (yield_3 : (x - 2) * 180 = (x - 2) * 180)
  (average_yield : ((x + 2) * 40 + x * 120 + (x - 2) * 180) / (3 * x) = 100) :
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l3908_390826


namespace NUMINAMATH_CALUDE_megan_homework_time_l3908_390886

/-- The time it takes to complete all problems given the number of math problems,
    spelling problems, and problems that can be finished per hour. -/
def time_to_complete (math_problems : ℕ) (spelling_problems : ℕ) (problems_per_hour : ℕ) : ℕ :=
  (math_problems + spelling_problems) / problems_per_hour

/-- Theorem stating that with 36 math problems, 28 spelling problems,
    and the ability to finish 8 problems per hour, it takes 8 hours to complete all problems. -/
theorem megan_homework_time :
  time_to_complete 36 28 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_megan_homework_time_l3908_390886


namespace NUMINAMATH_CALUDE_store_employees_l3908_390853

/-- The number of employees in Sergio's store -/
def num_employees : ℕ := 20

/-- The initial average number of items sold per employee -/
def initial_average : ℚ := 75

/-- The new average number of items sold per employee -/
def new_average : ℚ := 783/10

/-- The number of items sold by the top three performers on the next day -/
def top_three_sales : ℕ := 6 + 5 + 4

theorem store_employees :
  (initial_average * num_employees + top_three_sales + 3 * (num_employees - 3)) / num_employees = new_average :=
sorry

end NUMINAMATH_CALUDE_store_employees_l3908_390853


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3908_390804

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, n^2 < 2^n) ↔ (∃ n₀ : ℕ, n₀^2 ≥ 2^n₀) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3908_390804


namespace NUMINAMATH_CALUDE_bus_stop_time_l3908_390832

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 50) 
  (h2 : speed_with_stops = 42) : 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_time_l3908_390832


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l3908_390881

/-- Represents a cube with painted strips on its faces -/
structure PaintedCube where
  size : Nat
  totalUnitCubes : Nat
  stripsPerFace : Nat
  stripWidth : Nat
  stripLength : Nat

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpaintedCubes (cube : PaintedCube) : Nat :=
  cube.totalUnitCubes - paintedCubes cube
where
  /-- Helper function to calculate the number of painted unit cubes -/
  paintedCubes (cube : PaintedCube) : Nat :=
    let totalPainted := 6 * cube.stripsPerFace * cube.stripLength
    let edgeOverlaps := 12 * cube.stripWidth / 2
    let cornerOverlaps := 8
    totalPainted - edgeOverlaps - cornerOverlaps

/-- Theorem stating that a 6x6x6 cube with specific painted strips has 170 unpainted unit cubes -/
theorem unpainted_cubes_count :
  let cube : PaintedCube := {
    size := 6,
    totalUnitCubes := 216,
    stripsPerFace := 2,
    stripWidth := 1,
    stripLength := 6
  }
  unpaintedCubes cube = 170 := by
  sorry


end NUMINAMATH_CALUDE_unpainted_cubes_count_l3908_390881


namespace NUMINAMATH_CALUDE_linear_equation_result_l3908_390813

theorem linear_equation_result (x m : ℝ) : 
  (∃ a b : ℝ, x^(2*m-3) + 6 = a*x + b) → (x + 3)^2010 = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_result_l3908_390813


namespace NUMINAMATH_CALUDE_compass_leg_swap_impossible_l3908_390875

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents the state of the compass -/
structure CompassState where
  leg1 : GridPoint
  leg2 : GridPoint

/-- The squared distance between two grid points -/
def squaredDistance (p1 p2 : GridPoint) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- A valid move of the compass -/
def isValidMove (start finish : CompassState) : Prop :=
  (start.leg1 = finish.leg1 ∧ squaredDistance start.leg1 start.leg2 = squaredDistance start.leg1 finish.leg2) ∨
  (start.leg2 = finish.leg2 ∧ squaredDistance start.leg1 start.leg2 = squaredDistance finish.leg1 start.leg2)

/-- A sequence of valid moves -/
def isValidMoveSequence : List CompassState → Prop
  | [] => True
  | [_] => True
  | s1 :: s2 :: rest => isValidMove s1 s2 ∧ isValidMoveSequence (s2 :: rest)

/-- The main theorem stating it's impossible to swap compass legs -/
theorem compass_leg_swap_impossible (start finish : CompassState) (moves : List CompassState) :
  isValidMoveSequence (start :: moves ++ [finish]) →
  squaredDistance start.leg1 start.leg2 = squaredDistance finish.leg1 finish.leg2 →
  ¬(start.leg1 = finish.leg2 ∧ start.leg2 = finish.leg1) :=
sorry

end NUMINAMATH_CALUDE_compass_leg_swap_impossible_l3908_390875


namespace NUMINAMATH_CALUDE_fountain_area_l3908_390883

theorem fountain_area (diameter : Real) (radius : Real) :
  diameter = 20 →
  radius * 2 = diameter →
  radius ^ 2 = 244 →
  π * radius ^ 2 = 244 * π :=
by sorry

end NUMINAMATH_CALUDE_fountain_area_l3908_390883


namespace NUMINAMATH_CALUDE_painting_cost_tripled_l3908_390896

/-- Cost of painting a room's walls -/
structure PaintingCost where
  length : ℝ
  breadth : ℝ
  height : ℝ
  cost : ℝ

/-- Theorem: Cost of painting a room 3 times larger -/
theorem painting_cost_tripled (room : PaintingCost) (h : room.cost = 350) :
  let tripled_room := PaintingCost.mk (3 * room.length) (3 * room.breadth) (3 * room.height) 0
  tripled_room.cost = 6300 := by
  sorry


end NUMINAMATH_CALUDE_painting_cost_tripled_l3908_390896


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3908_390880

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3908_390880


namespace NUMINAMATH_CALUDE_smallest_class_size_l3908_390806

theorem smallest_class_size : 
  (∃ n : ℕ, n > 30 ∧ 
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 
      n = 3 * x + y ∧ 
      y = x + 1) ∧
    (∀ m : ℕ, m > 30 → 
      (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 
        m = 3 * a + b ∧ 
        b = a + 1) → 
      m ≥ n)) →
  (∃ n : ℕ, n = 33 ∧ n > 30 ∧ 
    (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 
      n = 3 * x + y ∧ 
      y = x + 1) ∧
    (∀ m : ℕ, m > 30 → 
      (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 
        m = 3 * a + b ∧ 
        b = a + 1) → 
      m ≥ n)) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l3908_390806


namespace NUMINAMATH_CALUDE_walking_speed_problem_l3908_390833

/-- Proves that given the conditions of the walking problem, Deepak's speed is 4.5 km/hr -/
theorem walking_speed_problem (track_circumference : ℝ) (wife_speed : ℝ) (meeting_time : ℝ) :
  track_circumference = 528 →
  wife_speed = 3.75 →
  meeting_time = 3.84 →
  ∃ (deepak_speed : ℝ),
    deepak_speed = 4.5 ∧
    (wife_speed * 1000 / 60) * meeting_time + deepak_speed * 1000 / 60 * meeting_time = track_circumference :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l3908_390833


namespace NUMINAMATH_CALUDE_solve_system_l3908_390824

theorem solve_system (x y : ℚ) (h1 : x - y = 12) (h2 : 2 * x + y = 10) : y = -14/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3908_390824


namespace NUMINAMATH_CALUDE_jake_peaches_l3908_390848

/-- The number of peaches Jill has -/
def jill_peaches : ℕ := 5

/-- The number of peaches Steven has more than Jill -/
def steven_more_than_jill : ℕ := 18

/-- The number of peaches Jake has fewer than Steven -/
def jake_fewer_than_steven : ℕ := 6

/-- Theorem: Jake has 17 peaches -/
theorem jake_peaches : 
  jill_peaches + steven_more_than_jill - jake_fewer_than_steven = 17 := by
  sorry

end NUMINAMATH_CALUDE_jake_peaches_l3908_390848


namespace NUMINAMATH_CALUDE_clock_hand_alignments_in_day_l3908_390821

/-- Represents a traditional 12-hour analog clock -/
structure AnalogClock where
  hourHand : ℝ
  minuteHand : ℝ
  secondHand : ℝ

/-- The number of times the clock hands align in a 12-hour period -/
def alignmentsIn12Hours : ℕ := 1

/-- The number of 12-hour periods in a day -/
def periodsInDay : ℕ := 2

/-- Theorem: The number of times all three hands align in a 24-hour period is 2 -/
theorem clock_hand_alignments_in_day :
  alignmentsIn12Hours * periodsInDay = 2 := by sorry

end NUMINAMATH_CALUDE_clock_hand_alignments_in_day_l3908_390821


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l3908_390811

-- Define the function f(x) = x³ - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- State the theorem
theorem tangent_line_triangle_area :
  let tangent_slope : ℝ := f' 0
  let tangent_y_intercept : ℝ := 1
  let tangent_x_intercept : ℝ := 1
  (1 / 2) * tangent_x_intercept * tangent_y_intercept = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l3908_390811


namespace NUMINAMATH_CALUDE_large_pizza_cost_is_10_l3908_390843

/-- Represents the cost of a pizza topping --/
structure ToppingCost where
  first : ℝ
  next_two : ℝ
  rest : ℝ

/-- Calculates the total cost of toppings --/
def total_topping_cost (tc : ToppingCost) (num_toppings : ℕ) : ℝ :=
  tc.first + 
  (if num_toppings > 1 then min (num_toppings - 1) 2 * tc.next_two else 0) +
  (if num_toppings > 3 then (num_toppings - 3) * tc.rest else 0)

/-- The cost of a large pizza without toppings --/
def large_pizza_cost (slices : ℕ) (cost_per_slice : ℝ) (num_toppings : ℕ) (tc : ToppingCost) : ℝ :=
  slices * cost_per_slice - total_topping_cost tc num_toppings

/-- Theorem: The cost of a large pizza without toppings is $10.00 --/
theorem large_pizza_cost_is_10 : 
  large_pizza_cost 8 2 7 ⟨2, 1, 0.5⟩ = 10 := by
  sorry

end NUMINAMATH_CALUDE_large_pizza_cost_is_10_l3908_390843


namespace NUMINAMATH_CALUDE_max_xy_in_H_inf_l3908_390827

-- Define the set H_n
def H (n : ℕ) : Set (ℝ × ℝ) :=
  match n with
  | 0 => {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}
  | n+1 => 
    let prev := H n
    let divide (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
      if n % 2 = 0 then
        {p : ℝ × ℝ | ∃q ∈ s, (p.1 = q.1 + 1/2^(n+2) ∧ p.2 = q.2 + 1/2^(n+2)) ∨
                             (p.1 = q.1 - 1/2^(n+2) ∧ p.2 = q.2 - 1/2^(n+2))}
      else
        {p : ℝ × ℝ | ∃q ∈ s, (p.1 = q.1 + 1/2^(n+2) ∧ p.2 = q.2 - 1/2^(n+2)) ∨
                             (p.1 = q.1 - 1/2^(n+2) ∧ p.2 = q.2 + 1/2^(n+2))}
    divide prev

-- Define the intersection of all H_n
def H_inf : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∀n : ℕ, p ∈ H n}

-- Theorem statement
theorem max_xy_in_H_inf :
  ∃p ∈ H_inf, ∀q ∈ H_inf, p.1 * p.2 ≥ q.1 * q.2 ∧ p.1 * p.2 = 11/16 :=
sorry

end NUMINAMATH_CALUDE_max_xy_in_H_inf_l3908_390827


namespace NUMINAMATH_CALUDE_jorge_goals_this_season_l3908_390899

/-- Given that Jorge scored 156 goals last season and the total number of goals he scored is 343,
    prove that the number of goals he scored this season is 187. -/
theorem jorge_goals_this_season (goals_last_season goals_total : ℕ)
    (h1 : goals_last_season = 156)
    (h2 : goals_total = 343) :
    goals_total - goals_last_season = 187 := by
  sorry

end NUMINAMATH_CALUDE_jorge_goals_this_season_l3908_390899


namespace NUMINAMATH_CALUDE_annas_gold_amount_annas_gold_theorem_l3908_390810

theorem annas_gold_amount (gary_gold : ℕ) (gary_cost_per_gram : ℕ) (anna_cost_per_gram : ℕ) (total_cost : ℕ) : ℕ :=
  let gary_total_cost := gary_gold * gary_cost_per_gram
  let anna_total_cost := total_cost - gary_total_cost
  anna_total_cost / anna_cost_per_gram

theorem annas_gold_theorem :
  annas_gold_amount 30 15 20 1450 = 50 := by
  sorry

end NUMINAMATH_CALUDE_annas_gold_amount_annas_gold_theorem_l3908_390810


namespace NUMINAMATH_CALUDE_ab_max_and_inverse_sum_min_l3908_390830

theorem ab_max_and_inverse_sum_min (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + 4*b = 4) : 
  (∀ x y, x > 0 → y > 0 → x + 4*y = 4 → a*b ≥ x*y) ∧ 
  (∀ x y, x > 0 → y > 0 → x + 4*y = 4 → 1/a + 4/b ≤ 1/x + 4/y) ∧
  (a*b = 1) ∧ (1/a + 4/b = 25/4) :=
sorry

end NUMINAMATH_CALUDE_ab_max_and_inverse_sum_min_l3908_390830


namespace NUMINAMATH_CALUDE_profit_percentage_is_30_percent_l3908_390844

def cost_per_dog : ℝ := 1000
def selling_price_two_dogs : ℝ := 2600

theorem profit_percentage_is_30_percent :
  let cost_two_dogs := 2 * cost_per_dog
  let profit := selling_price_two_dogs - cost_two_dogs
  let profit_percentage := (profit / cost_two_dogs) * 100
  profit_percentage = 30 := by sorry

end NUMINAMATH_CALUDE_profit_percentage_is_30_percent_l3908_390844


namespace NUMINAMATH_CALUDE_polynomial_property_l3908_390878

def P (a b c : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + c

theorem polynomial_property (a b c : ℝ) :
  (∃ x y z : ℝ, x * y * z = -c / 2 ∧ 
                x^2 + y^2 + z^2 = -c / 2 ∧ 
                2 + a + b + c = -c / 2) →
  P a b c 0 = 12 →
  b = -56 := by sorry

end NUMINAMATH_CALUDE_polynomial_property_l3908_390878


namespace NUMINAMATH_CALUDE_complex_modulus_seven_l3908_390882

theorem complex_modulus_seven (x : ℝ) : 
  x > 0 → (Complex.abs (3 + x * Complex.I) = 7 ↔ x = 2 * Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_seven_l3908_390882


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l3908_390825

theorem no_integer_solutions_for_equation : ∀ x y : ℤ, 19 * x^3 - 17 * y^3 ≠ 50 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l3908_390825


namespace NUMINAMATH_CALUDE_expedition_cans_required_l3908_390858

/-- The number of days between neighboring camps -/
def days_between_camps : ℕ := 1

/-- The number of days from base camp to destination camp -/
def days_to_destination : ℕ := 5

/-- The maximum number of cans a member can carry -/
def max_cans_per_member : ℕ := 3

/-- The number of cans consumed by a member per day -/
def cans_consumed_per_day : ℕ := 1

/-- Function to calculate the minimum number of cans required -/
def min_cans_required (n : ℕ) : ℕ := max_cans_per_member ^ n

/-- Theorem stating the minimum number of cans required for the expedition -/
theorem expedition_cans_required :
  min_cans_required days_to_destination = 243 :=
sorry

end NUMINAMATH_CALUDE_expedition_cans_required_l3908_390858


namespace NUMINAMATH_CALUDE_bill_per_person_l3908_390884

def total_bill : ℚ := 139
def num_people : ℕ := 8
def tip_percentage : ℚ := 1 / 10

theorem bill_per_person : 
  ∃ (bill_share : ℚ), 
    (bill_share * num_people).ceil = 
      ((total_bill * (1 + tip_percentage)).ceil) ∧ 
    bill_share = 1911 / 100 := by
  sorry

end NUMINAMATH_CALUDE_bill_per_person_l3908_390884


namespace NUMINAMATH_CALUDE_race_length_proof_l3908_390862

/-- The length of the race in metres -/
def race_length : ℝ := 200

/-- The fraction of the race completed -/
def fraction_completed : ℝ := 0.25

/-- The distance run so far in metres -/
def distance_run : ℝ := 50

theorem race_length_proof : 
  fraction_completed * race_length = distance_run :=
by sorry

end NUMINAMATH_CALUDE_race_length_proof_l3908_390862


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l3908_390814

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_pos : ∀ x > 0, f x = x * (1 - x)) :
  ∀ x < 0, f x = x * (1 + x) := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l3908_390814


namespace NUMINAMATH_CALUDE_carries_work_hours_l3908_390801

/-- Proves that Carrie works 35 hours per week given the problem conditions -/
theorem carries_work_hours :
  let hourly_rate : ℕ := 8
  let weeks_worked : ℕ := 4
  let bike_cost : ℕ := 400
  let money_left : ℕ := 720
  let total_earned : ℕ := bike_cost + money_left
  ∃ (hours_per_week : ℕ),
    hours_per_week * hourly_rate * weeks_worked = total_earned ∧
    hours_per_week = 35
  := by sorry

end NUMINAMATH_CALUDE_carries_work_hours_l3908_390801


namespace NUMINAMATH_CALUDE_square_construction_with_compass_l3908_390857

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a compass operation
def compassIntersection (c1 c2 : Circle) : Set Point :=
  { p : Point | (p.x - c1.center.x)^2 + (p.y - c1.center.y)^2 = c1.radius^2 ∧
                (p.x - c2.center.x)^2 + (p.y - c2.center.y)^2 = c2.radius^2 }

-- Define a square
structure Square where
  vertices : Fin 4 → Point

-- Theorem statement
theorem square_construction_with_compass :
  ∃ (s : Square), 
    (∀ i j : Fin 4, i ≠ j → 
      (s.vertices i).x^2 + (s.vertices i).y^2 = 
      (s.vertices j).x^2 + (s.vertices j).y^2) ∧
    (∀ i : Fin 4, 
      (s.vertices i).x^2 + (s.vertices i).y^2 = 
      ((s.vertices (i + 1)).x - (s.vertices i).x)^2 + 
      ((s.vertices (i + 1)).y - (s.vertices i).y)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_square_construction_with_compass_l3908_390857


namespace NUMINAMATH_CALUDE_equation_solution_l3908_390865

theorem equation_solution : ∃ x : ℚ, (4 / 7) * (1 / 5) * x + 2 = 8 ∧ x = 105 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3908_390865


namespace NUMINAMATH_CALUDE_product_positive_l3908_390872

theorem product_positive (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x^2 - x > y^2) (h2 : y^2 - y > x^2) : x * y > 0 := by
  sorry

end NUMINAMATH_CALUDE_product_positive_l3908_390872


namespace NUMINAMATH_CALUDE_rectangle_DC_length_l3908_390817

/-- Represents a rectangle ABCF with points E and D on FC -/
structure Rectangle :=
  (AB : ℝ)
  (AF : ℝ)
  (FE : ℝ)
  (area_ABDE : ℝ)

/-- The length of DC in the rectangle -/
def length_DC (r : Rectangle) : ℝ :=
  -- Definition of DC length
  sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem rectangle_DC_length (r : Rectangle) 
  (h1 : r.AB = 30)
  (h2 : r.AF = 14)
  (h3 : r.FE = 5)
  (h4 : r.area_ABDE = 266) :
  length_DC r = 17 :=
sorry

end NUMINAMATH_CALUDE_rectangle_DC_length_l3908_390817


namespace NUMINAMATH_CALUDE_all_roots_of_polynomial_l3908_390871

/-- The polynomial function we're considering -/
def f (x : ℝ) : ℝ := x^3 - x^2 - 4*x + 4

/-- The set of roots we claim are correct -/
def roots : Set ℝ := {-2, 1, 2}

/-- Theorem stating that the given set contains all roots of the polynomial -/
theorem all_roots_of_polynomial :
  ∀ x : ℝ, f x = 0 ↔ x ∈ roots := by sorry

end NUMINAMATH_CALUDE_all_roots_of_polynomial_l3908_390871


namespace NUMINAMATH_CALUDE_inequality_of_powers_l3908_390849

theorem inequality_of_powers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^(2*a) * b^(2*b) * c^(2*c) ≥ a^(b+c) * b^(c+a) * c^(a+b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l3908_390849


namespace NUMINAMATH_CALUDE_tank_filling_l3908_390822

/-- Proves that adding 4 gallons to a 32-gallon tank that is 3/4 full results in the tank being 7/8 full -/
theorem tank_filling (tank_capacity : ℚ) (initial_fraction : ℚ) (added_amount : ℚ) : 
  tank_capacity = 32 →
  initial_fraction = 3 / 4 →
  added_amount = 4 →
  (initial_fraction * tank_capacity + added_amount) / tank_capacity = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_l3908_390822


namespace NUMINAMATH_CALUDE_marked_hexagon_properties_l3908_390803

/-- A regular hexagon with diagonals marked -/
structure MarkedHexagon where
  /-- The area of the hexagon in square centimeters -/
  area : ℝ
  /-- The hexagon is regular -/
  regular : Bool
  /-- All diagonals are marked -/
  diagonals_marked : Bool

/-- The number of parts the hexagon is divided into by its diagonals -/
def num_parts (h : MarkedHexagon) : ℕ := sorry

/-- The area of the smaller hexagon formed by quadrilateral parts -/
def smaller_hexagon_area (h : MarkedHexagon) : ℝ := sorry

/-- Theorem about the properties of a marked regular hexagon -/
theorem marked_hexagon_properties (h : MarkedHexagon) 
  (h_area : h.area = 144)
  (h_regular : h.regular = true)
  (h_marked : h.diagonals_marked = true) :
  num_parts h = 24 ∧ smaller_hexagon_area h = 48 := by sorry

end NUMINAMATH_CALUDE_marked_hexagon_properties_l3908_390803
