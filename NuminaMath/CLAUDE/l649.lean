import Mathlib

namespace NUMINAMATH_CALUDE_probability_one_unit_apart_l649_64928

/-- The number of points around the square -/
def num_points : ℕ := 12

/-- The number of pairs of points that are one unit apart -/
def favorable_pairs : ℕ := 12

/-- The total number of ways to choose two points from num_points -/
def total_pairs : ℕ := num_points.choose 2

/-- The probability of choosing two points one unit apart -/
def probability : ℚ := favorable_pairs / total_pairs

theorem probability_one_unit_apart : probability = 2 / 11 := by sorry

end NUMINAMATH_CALUDE_probability_one_unit_apart_l649_64928


namespace NUMINAMATH_CALUDE_seventh_diagram_shaded_fraction_l649_64906

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Total number of triangles in the nth diagram -/
def total_triangles (n : ℕ) : ℕ := Nat.factorial n

/-- Fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ :=
  (fib n : ℚ) / (total_triangles n : ℚ)

/-- The main theorem -/
theorem seventh_diagram_shaded_fraction :
  shaded_fraction 7 = 13 / 5040 := by
  sorry

end NUMINAMATH_CALUDE_seventh_diagram_shaded_fraction_l649_64906


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_is_17_l649_64912

structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

structure DistancesToFaces where
  ABC : ℝ
  ABD : ℝ
  ACD : ℝ
  BCD : ℝ

def ABCD : Tetrahedron := sorry

def X : Point := sorry
def Y : Point := sorry

def distances_X : DistancesToFaces := {
  ABC := 14,
  ABD := 11,
  ACD := 29,
  BCD := 8
}

def distances_Y : DistancesToFaces := {
  ABC := 15,
  ABD := 13,
  ACD := 25,
  BCD := 11
}

def inscribed_sphere_radius (t : Tetrahedron) : ℝ := sorry

theorem inscribed_sphere_radius_is_17 :
  inscribed_sphere_radius ABCD = 17 := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_is_17_l649_64912


namespace NUMINAMATH_CALUDE_tess_decoration_l649_64951

/-- The number of heart stickers Tess has -/
def heart_stickers : ℕ := 120

/-- The number of star stickers Tess has -/
def star_stickers : ℕ := 81

/-- The number of smiley stickers Tess has -/
def smiley_stickers : ℕ := 45

/-- The greatest number of pages Tess can decorate -/
def max_pages : ℕ := Nat.gcd (Nat.gcd heart_stickers star_stickers) smiley_stickers

theorem tess_decoration :
  max_pages = 3 ∧
  heart_stickers % max_pages = 0 ∧
  star_stickers % max_pages = 0 ∧
  smiley_stickers % max_pages = 0 ∧
  ∀ n : ℕ, n > max_pages →
    (heart_stickers % n = 0 ∧ star_stickers % n = 0 ∧ smiley_stickers % n = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_tess_decoration_l649_64951


namespace NUMINAMATH_CALUDE_carrot_to_lettuce_ratio_l649_64971

def lettuce_calories : ℕ := 50
def dressing_calories : ℕ := 210
def pizza_crust_calories : ℕ := 600
def pizza_cheese_calories : ℕ := 400
def total_consumed_calories : ℕ := 330

def pizza_total_calories : ℕ := pizza_crust_calories + (pizza_crust_calories / 3) + pizza_cheese_calories

def salad_calories (carrot_calories : ℕ) : ℕ := lettuce_calories + carrot_calories + dressing_calories

theorem carrot_to_lettuce_ratio :
  ∃ (carrot_calories : ℕ),
    (salad_calories carrot_calories / 4 + pizza_total_calories / 5 = total_consumed_calories) ∧
    (carrot_calories / lettuce_calories = 2) := by
  sorry

end NUMINAMATH_CALUDE_carrot_to_lettuce_ratio_l649_64971


namespace NUMINAMATH_CALUDE_class_visual_conditions_most_comprehensive_l649_64990

/-- Represents a survey option -/
inductive SurveyOption
| LightTubes
| ClassVisualConditions
| NationwideExerciseTime
| FoodPigmentContent

/-- Defines characteristics of a survey -/
structure SurveyCharacteristics where
  population_size : ℕ
  geographical_spread : Bool
  data_collection_feasibility : Bool

/-- Defines what makes a survey comprehensive -/
def is_comprehensive (s : SurveyCharacteristics) : Prop :=
  s.population_size ≤ 100 ∧ ¬s.geographical_spread ∧ s.data_collection_feasibility

/-- Assigns characteristics to each survey option -/
def survey_characteristics : SurveyOption → SurveyCharacteristics
| SurveyOption.LightTubes => ⟨50, false, false⟩
| SurveyOption.ClassVisualConditions => ⟨30, false, true⟩
| SurveyOption.NationwideExerciseTime => ⟨1000000, true, false⟩
| SurveyOption.FoodPigmentContent => ⟨500, true, false⟩

/-- Theorem stating that investigating visual conditions of a class is the most suitable for a comprehensive survey -/
theorem class_visual_conditions_most_comprehensive :
  ∀ (s : SurveyOption), s ≠ SurveyOption.ClassVisualConditions →
  is_comprehensive (survey_characteristics SurveyOption.ClassVisualConditions) ∧
  ¬(is_comprehensive (survey_characteristics s)) :=
by sorry

end NUMINAMATH_CALUDE_class_visual_conditions_most_comprehensive_l649_64990


namespace NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l649_64905

theorem power_of_product_equals_product_of_powers (a : ℝ) : (2 * a^3)^3 = 8 * a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l649_64905


namespace NUMINAMATH_CALUDE_min_value_theorem_l649_64949

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^4 + 16*x + 256/x^6 ≥ 56 ∧
  (x^4 + 16*x + 256/x^6 = 56 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l649_64949


namespace NUMINAMATH_CALUDE_polynomial_simplification_l649_64973

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 6*x^3 = 
  6*x^3 - x^2 + 23*x - 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l649_64973


namespace NUMINAMATH_CALUDE_stone_price_calculation_l649_64994

/-- The price per stone when selling a collection of precious stones -/
def price_per_stone (total_amount : ℕ) (num_stones : ℕ) : ℚ :=
  (total_amount : ℚ) / (num_stones : ℚ)

/-- Theorem stating that the price per stone is $1785 when 8 stones are sold for $14280 -/
theorem stone_price_calculation :
  price_per_stone 14280 8 = 1785 := by
  sorry

end NUMINAMATH_CALUDE_stone_price_calculation_l649_64994


namespace NUMINAMATH_CALUDE_zoo_sandwiches_l649_64938

theorem zoo_sandwiches (people : ℝ) (sandwiches_per_person : ℝ) :
  people = 219.0 →
  sandwiches_per_person = 3.0 →
  people * sandwiches_per_person = 657.0 := by
  sorry

end NUMINAMATH_CALUDE_zoo_sandwiches_l649_64938


namespace NUMINAMATH_CALUDE_gcd_180_270_l649_64974

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_270_l649_64974


namespace NUMINAMATH_CALUDE_special_trapezoid_area_ratios_l649_64956

/-- A trapezoid with a diagonal forming a 45° angle with the base, 
    and both inscribed and circumscribed circles -/
structure SpecialTrapezoid where
  -- Base lengths
  a : ℝ
  b : ℝ
  -- Height
  h : ℝ
  -- Diagonal forms 45° angle with base
  diagonal_angle : Real.cos (45 * π / 180) = h / (a - b)
  -- Inscribed circle exists
  inscribed_circle_exists : ∃ r : ℝ, r > 0 ∧ r = h / 2
  -- Circumscribed circle exists
  circumscribed_circle_exists : ∃ R : ℝ, R > 0 ∧ R = h / Real.sqrt 2

/-- The main theorem about the area ratios -/
theorem special_trapezoid_area_ratios (t : SpecialTrapezoid) : 
  (t.a + t.b) * t.h / (π * (t.h / 2)^2) = 4 / π ∧
  (t.a + t.b) * t.h / (π * (t.h / Real.sqrt 2)^2) = 2 / π := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_area_ratios_l649_64956


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l649_64940

/-- Given a right triangle with sides of length 6 and 8, 
    the length of the third side is 2√7 -/
theorem shortest_side_of_right_triangle : ∃ (a b c : ℝ), 
  a = 6 ∧ b = 8 ∧ c = 2 * Real.sqrt 7 ∧ 
  a^2 + c^2 = b^2 ∧ 
  ∀ (x : ℝ), (x^2 + a^2 = b^2 → x ≥ c) :=
by sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l649_64940


namespace NUMINAMATH_CALUDE_circle_radius_l649_64942

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 = 0

-- State the theorem
theorem circle_radius : ∃ (h k r : ℝ), r = 2 ∧
  ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l649_64942


namespace NUMINAMATH_CALUDE_min_value_theorem_l649_64981

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := (x-1)^2 + (y-3)^2 = 4

-- Define the distance function
def dist_squared (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₁ - x₂)^2 + (y₁ - y₂)^2

-- Define the condition |PC₁| = |PC₂|
def point_condition (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
  dist_squared a b x₁ y₁ = dist_squared a b x₂ y₂

-- Define the expression to be minimized
def expr_to_minimize (a b : ℝ) : ℝ := a^2 + b^2 - 6*a - 4*b + 13

-- State the theorem
theorem min_value_theorem :
  ∃ (min : ℝ), min = 8/5 ∧
  ∀ (a b : ℝ), point_condition a b → expr_to_minimize a b ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l649_64981


namespace NUMINAMATH_CALUDE_perfect_square_values_l649_64958

theorem perfect_square_values (x : ℕ) : 
  (x = 0 ∨ x = 9 ∨ x = 12) → 
  ∃ y : ℕ, 2^6 + 2^10 + 2^x = y^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_values_l649_64958


namespace NUMINAMATH_CALUDE_fencing_required_l649_64904

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 680 ∧ uncovered_side = 20 → 
  2 * (area / uncovered_side) + uncovered_side = 88 := by
  sorry

#check fencing_required

end NUMINAMATH_CALUDE_fencing_required_l649_64904


namespace NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l649_64932

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l649_64932


namespace NUMINAMATH_CALUDE_coronavirus_size_scientific_notation_l649_64954

/-- The size of a novel coronavirus in meters -/
def coronavirus_size : ℝ := 0.000000125

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Convert a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation := sorry

theorem coronavirus_size_scientific_notation :
  to_scientific_notation coronavirus_size = ScientificNotation.mk 1.25 (-7) := by sorry

end NUMINAMATH_CALUDE_coronavirus_size_scientific_notation_l649_64954


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l649_64963

def volleyball_team_size : ℕ := 16
def num_twins : ℕ := 2
def num_starters : ℕ := 8

theorem volleyball_lineup_count :
  (Nat.choose (volleyball_team_size - num_twins) num_starters) +
  (num_twins * Nat.choose (volleyball_team_size - num_twins) (num_starters - 1)) = 9867 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l649_64963


namespace NUMINAMATH_CALUDE_prism_volume_l649_64926

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h_side : side_area = 18)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ x y z : ℝ, 
    x * y = side_area ∧ 
    y * z = front_area ∧ 
    x * z = bottom_area ∧ 
    x * y * z = 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l649_64926


namespace NUMINAMATH_CALUDE_jamal_has_one_black_marble_l649_64966

/-- Represents the bag of marbles with different colors. -/
structure MarbleBag where
  yellow : ℕ
  blue : ℕ
  green : ℕ
  black : ℕ

/-- The probability of drawing a black marble. -/
def blackProbability : ℚ := 1 / 28

/-- Jamal's bag of marbles. -/
def jamalsBag : MarbleBag := {
  yellow := 12,
  blue := 10,
  green := 5,
  black := 1  -- We'll prove this is correct
}

/-- The total number of marbles in the bag. -/
def totalMarbles (bag : MarbleBag) : ℕ :=
  bag.yellow + bag.blue + bag.green + bag.black

/-- Theorem stating that Jamal's bag contains exactly one black marble. -/
theorem jamal_has_one_black_marble :
  jamalsBag.black = 1 ∧
  (jamalsBag.black : ℚ) / (totalMarbles jamalsBag : ℚ) = blackProbability :=
by sorry

end NUMINAMATH_CALUDE_jamal_has_one_black_marble_l649_64966


namespace NUMINAMATH_CALUDE_road_signs_ratio_l649_64985

/-- Represents the number of road signs at each intersection -/
structure RoadSigns where
  s1 : ℕ  -- First intersection
  s2 : ℕ  -- Second intersection
  s3 : ℕ  -- Third intersection
  s4 : ℕ  -- Fourth intersection

/-- Theorem stating the ratio of road signs at the third to second intersection -/
theorem road_signs_ratio 
  (signs : RoadSigns) 
  (h1 : signs.s1 = 40)
  (h2 : signs.s2 = signs.s1 + signs.s1 / 4)
  (h3 : signs.s4 = signs.s3 - 20)
  (h4 : signs.s1 + signs.s2 + signs.s3 + signs.s4 = 270) :
  signs.s3 / signs.s2 = 2 := by
  sorry

#eval (100 : ℚ) / 50  -- Expected output: 2

end NUMINAMATH_CALUDE_road_signs_ratio_l649_64985


namespace NUMINAMATH_CALUDE_amusement_park_initial_cost_l649_64976

/-- The initial cost to open an amusement park, given the conditions described in the problem. -/
def initial_cost : ℝ → Prop := λ C =>
  let daily_running_cost := 0.01 * C
  let daily_revenue := 1500
  let days_to_breakeven := 200
  C = days_to_breakeven * (daily_revenue - daily_running_cost)

/-- Theorem stating that the initial cost to open the amusement park is $100,000. -/
theorem amusement_park_initial_cost :
  ∃ C : ℝ, initial_cost C ∧ C = 100000 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_initial_cost_l649_64976


namespace NUMINAMATH_CALUDE_total_points_target_l649_64987

def average_points_after_two_games : ℝ := 61.5
def points_in_game_three : ℕ := 47
def additional_points_needed : ℕ := 330

theorem total_points_target :
  (2 * average_points_after_two_games + points_in_game_three + additional_points_needed : ℝ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_points_target_l649_64987


namespace NUMINAMATH_CALUDE_treasure_burial_year_l649_64933

def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8^i)) 0

theorem treasure_burial_year : 
  octal_to_decimal [1, 7, 6, 2] = 1465 := by
  sorry

end NUMINAMATH_CALUDE_treasure_burial_year_l649_64933


namespace NUMINAMATH_CALUDE_y_to_x_equals_25_l649_64950

theorem y_to_x_equals_25 (x y : ℝ) (h : |x - 2| + (y + 5)^2 = 0) : y^x = 25 := by
  sorry

end NUMINAMATH_CALUDE_y_to_x_equals_25_l649_64950


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l649_64968

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (a + I) * (1 - 2*I) = b*I) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l649_64968


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_of_250_l649_64924

theorem closest_integer_to_cube_root_of_250 : 
  ∃ (n : ℤ), n = 6 ∧ ∀ (m : ℤ), |m ^ 3 - 250| ≥ |n ^ 3 - 250| :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_of_250_l649_64924


namespace NUMINAMATH_CALUDE_triangle_area_l649_64934

theorem triangle_area (x : ℝ) (α : ℝ) : 
  let BC := 4*x
  let CD := x
  let AC := 8*x*(Real.sqrt 2/Real.sqrt 3)
  let AD := (3/4 : ℝ)
  let cos_α := Real.sqrt 2/Real.sqrt 3
  let sin_α := 1/Real.sqrt 3
  (AD^2 = 33*x^2) →
  (1/2 * AC * BC * sin_α = Real.sqrt 2/11) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l649_64934


namespace NUMINAMATH_CALUDE_supplement_bottles_sum_l649_64910

/-- Given 5 supplement bottles, where 2 bottles have 30 pills each, and after using 70 pills,
    350 pills remain, prove that the sum of pills in the other 3 bottles is 360. -/
theorem supplement_bottles_sum (total_bottles : Nat) (small_bottles : Nat) (pills_per_small_bottle : Nat)
  (pills_used : Nat) (pills_remaining : Nat) :
  total_bottles = 5 →
  small_bottles = 2 →
  pills_per_small_bottle = 30 →
  pills_used = 70 →
  pills_remaining = 350 →
  ∃ (a b c : Nat), a + b + c = 360 :=
by sorry

end NUMINAMATH_CALUDE_supplement_bottles_sum_l649_64910


namespace NUMINAMATH_CALUDE_tan_105_degrees_l649_64992

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l649_64992


namespace NUMINAMATH_CALUDE_intersection_and_complement_when_m_is_3_intersection_equals_B_iff_m_in_range_l649_64923

-- Define sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

-- Theorem 1
theorem intersection_and_complement_when_m_is_3 :
  (A ∩ B 3 = {x | 3 ≤ x ∧ x ≤ 4}) ∧
  (A ∩ (Set.univ \ B 3) = {x | 1 ≤ x ∧ x < 3}) := by sorry

-- Theorem 2
theorem intersection_equals_B_iff_m_in_range :
  ∀ m : ℝ, A ∩ B m = B m ↔ 1 ≤ m ∧ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_and_complement_when_m_is_3_intersection_equals_B_iff_m_in_range_l649_64923


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l649_64922

theorem weight_of_replaced_person
  (n : ℕ)
  (average_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : n = 10)
  (h2 : average_increase = 2.5)
  (h3 : new_person_weight = 90)
  : ∃ (replaced_weight : ℝ),
    replaced_weight = new_person_weight - n * average_increase :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l649_64922


namespace NUMINAMATH_CALUDE_correct_calculation_l649_64903

theorem correct_calculation (x : ℕ) (h : x - 6 = 51) : x * 6 = 342 := by
  sorry

#check correct_calculation

end NUMINAMATH_CALUDE_correct_calculation_l649_64903


namespace NUMINAMATH_CALUDE_rainfall_difference_l649_64916

def camping_days : ℕ := 14
def rainy_days : ℕ := 7
def friend_rainfall : ℕ := 65

def greg_rainfall : List ℕ := [3, 6, 5, 7, 4, 8, 9]

theorem rainfall_difference :
  friend_rainfall - (greg_rainfall.sum) = 23 :=
by sorry

end NUMINAMATH_CALUDE_rainfall_difference_l649_64916


namespace NUMINAMATH_CALUDE_solve_for_k_l649_64902

theorem solve_for_k : ∀ k : ℤ, 
  (∀ x : ℤ, 2*x - 3 = 3*x - 2 + k ↔ x = 2) → 
  k = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l649_64902


namespace NUMINAMATH_CALUDE_ratio_of_terms_l649_64975

/-- Two arithmetic sequences and their properties -/
structure ArithmeticSequences where
  a : ℕ → ℚ  -- First sequence
  b : ℕ → ℚ  -- Second sequence
  S : ℕ → ℚ  -- Sum of first n terms of a
  T : ℕ → ℚ  -- Sum of first n terms of b
  h_arithmetic_a : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  h_arithmetic_b : ∀ n : ℕ, b (n + 2) - b (n + 1) = b (n + 1) - b n
  h_sum_a : ∀ n : ℕ, S n = (n : ℚ) * (a 1 + a n) / 2
  h_sum_b : ∀ n : ℕ, T n = (n : ℚ) * (b 1 + b n) / 2
  h_ratio : ∀ n : ℕ, S n / T n = (2 * n : ℚ) / (3 * n + 1)

/-- Main theorem: If the ratio of sums is given, then a_5 / b_6 = 9 / 17 -/
theorem ratio_of_terms (seq : ArithmeticSequences) : seq.a 5 / seq.b 6 = 9 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_terms_l649_64975


namespace NUMINAMATH_CALUDE_total_spots_granger_and_cisco_l649_64964

/-- The number of spots Rover has -/
def rover_spots : ℕ := 46

/-- The number of spots Cisco has -/
def cisco_spots : ℕ := rover_spots / 2 - 5

/-- The number of spots Granger has -/
def granger_spots : ℕ := 5 * cisco_spots

/-- Theorem stating the total number of spots Granger and Cisco have combined -/
theorem total_spots_granger_and_cisco : 
  granger_spots + cisco_spots = 108 := by sorry

end NUMINAMATH_CALUDE_total_spots_granger_and_cisco_l649_64964


namespace NUMINAMATH_CALUDE_calculation_proof_l649_64952

theorem calculation_proof : 5 * 7 * 11 + 21 / 7 - 3 = 385 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l649_64952


namespace NUMINAMATH_CALUDE_grid_paths_l649_64972

theorem grid_paths (total_steps : ℕ) (right_steps : ℕ) (up_steps : ℕ) 
  (h1 : total_steps = right_steps + up_steps)
  (h2 : total_steps = 10)
  (h3 : right_steps = 6)
  (h4 : up_steps = 4) :
  Nat.choose total_steps up_steps = 210 := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_l649_64972


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solve_y_l649_64944

/-- Given that 1/3, y-2, and 4y are consecutive terms of an arithmetic sequence, prove that y = -13/6 -/
theorem arithmetic_sequence_solve_y (y : ℚ) : 
  (y - 2 - (1/3 : ℚ) = 4*y - (y - 2)) → y = -13/6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solve_y_l649_64944


namespace NUMINAMATH_CALUDE_speed_adjustment_l649_64978

/-- Given a constant distance traveled at 10 km/h in 6 minutes,
    the speed required to travel the same distance in 8 minutes is 7.5 km/h. -/
theorem speed_adjustment (initial_speed initial_time new_time : ℝ) :
  initial_speed = 10 →
  initial_time = 6 / 60 →
  new_time = 8 / 60 →
  let distance := initial_speed * initial_time
  let new_speed := distance / new_time
  new_speed = 7.5 := by
sorry

end NUMINAMATH_CALUDE_speed_adjustment_l649_64978


namespace NUMINAMATH_CALUDE_spotted_fluffy_cats_l649_64957

def village_cats : ℕ := 120

def spotted_fraction : ℚ := 1/3

def fluffy_spotted_fraction : ℚ := 1/4

theorem spotted_fluffy_cats :
  (village_cats : ℚ) * spotted_fraction * fluffy_spotted_fraction = 10 := by
  sorry

end NUMINAMATH_CALUDE_spotted_fluffy_cats_l649_64957


namespace NUMINAMATH_CALUDE_complex_division_simplification_l649_64935

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  (8 - i) / (2 + i) = 3 - 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l649_64935


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l649_64901

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 2 then 2 / (2 - x) else 0

theorem unique_function_satisfying_conditions :
  (∀ x y, x ≥ 0 → y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x, 0 ≤ x → x < 2 → f x ≠ 0) ∧
  (∀ g : ℝ → ℝ, (∀ x y, x ≥ 0 → y ≥ 0 → g (x * g y) * g y = g (x + y)) →
    (g 2 = 0) →
    (∀ x, 0 ≤ x → x < 2 → g x ≠ 0) →
    (∀ x, x ≥ 0 → g x = f x)) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l649_64901


namespace NUMINAMATH_CALUDE_units_digit_is_nine_l649_64939

/-- The product of digits of a two-digit number -/
def P (n : ℕ) : ℕ := (n / 10) * (n % 10)

/-- The sum of digits of a two-digit number -/
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

/-- A two-digit number is between 10 and 99, inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem units_digit_is_nine (N : ℕ) (h1 : is_two_digit N) (h2 : N = P N + S N) :
  N % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_is_nine_l649_64939


namespace NUMINAMATH_CALUDE_principal_calculation_l649_64927

/-- Given a principal amount P at simple interest for 3 years, 
    if increasing the interest rate by 1% results in Rs. 72 more interest, 
    then P = 2400. -/
theorem principal_calculation (P : ℝ) (R : ℝ) : 
  (P * (R + 1) * 3) / 100 - (P * R * 3) / 100 = 72 → P = 2400 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l649_64927


namespace NUMINAMATH_CALUDE_school_pupils_count_l649_64999

theorem school_pupils_count (girls boys : ℕ) (h1 : girls = 542) (h2 : boys = 387) :
  girls + boys = 929 := by
  sorry

end NUMINAMATH_CALUDE_school_pupils_count_l649_64999


namespace NUMINAMATH_CALUDE_roots_sum_power_l649_64967

theorem roots_sum_power (c d : ℝ) : 
  c^2 - 5*c + 6 = 0 → d^2 - 5*d + 6 = 0 → c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_power_l649_64967


namespace NUMINAMATH_CALUDE_triangle_reconstruction_possible_l649_64980

-- Define the basic types and structures
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the given points
variable (X Y Z : Point)

-- Define the properties of the given points
def is_circumcenter (X : Point) (t : Triangle) : Prop := sorry

def is_midpoint (Y : Point) (B C : Point) : Prop := sorry

def is_altitude_foot (Z : Point) (B A C : Point) : Prop := sorry

-- State the theorem
theorem triangle_reconstruction_possible 
  (h_circumcenter : ∃ t : Triangle, is_circumcenter X t)
  (h_midpoint : ∃ B C : Point, is_midpoint Y B C)
  (h_altitude_foot : ∃ A B C : Point, is_altitude_foot Z B A C) :
  ∃! t : Triangle, 
    is_circumcenter X t ∧ 
    is_midpoint Y t.B t.C ∧ 
    is_altitude_foot Z t.B t.A t.C :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_possible_l649_64980


namespace NUMINAMATH_CALUDE_stratified_sample_size_l649_64931

/-- Represents the ratio of students in each grade -/
structure GradeRatio where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the sample size and number of third grade students in the sample -/
structure Sample where
  size : ℕ
  thirdGrade : ℕ

/-- Theorem stating the sample size given the conditions -/
theorem stratified_sample_size 
  (ratio : GradeRatio) 
  (sample : Sample) 
  (h1 : ratio.first = 4)
  (h2 : ratio.second = 3)
  (h3 : ratio.third = 2)
  (h4 : sample.thirdGrade = 10) :
  (ratio.third : ℚ) / (ratio.first + ratio.second + ratio.third : ℚ) = 
  (sample.thirdGrade : ℚ) / (sample.size : ℚ) → 
  sample.size = 45 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l649_64931


namespace NUMINAMATH_CALUDE_product_of_cosines_l649_64988

theorem product_of_cosines : 
  (1 + Real.cos (π / 9)) * (1 + Real.cos (2 * π / 9)) * 
  (1 + Real.cos (8 * π / 9)) * (1 + Real.cos (7 * π / 9)) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l649_64988


namespace NUMINAMATH_CALUDE_reading_homework_pages_l649_64936

theorem reading_homework_pages
  (math_pages : ℕ)
  (problems_per_page : ℕ)
  (total_problems : ℕ)
  (h1 : math_pages = 6)
  (h2 : problems_per_page = 3)
  (h3 : total_problems = 30) :
  (total_problems - math_pages * problems_per_page) / problems_per_page = 4 :=
by sorry

end NUMINAMATH_CALUDE_reading_homework_pages_l649_64936


namespace NUMINAMATH_CALUDE_sequence_non_positive_l649_64953

theorem sequence_non_positive (N : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hN : a N = 0) 
  (h_rec : ∀ i ∈ Finset.range (N - 1), 
    a (i + 2) - 2 * a (i + 1) + a i = (a (i + 1))^2) :
  ∀ i ∈ Finset.range (N - 1), a (i + 1) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l649_64953


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l649_64948

/-- Given two arithmetic sequences {a_n} and {b_n} with sums A_n and B_n,
    if A_n / B_n = (7n + 45) / (n + 3) for all n, then a_5 / b_5 = 9 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (A B : ℕ → ℚ) :
  (∀ n, A n = (n / 2) * (a 1 + a n)) →
  (∀ n, B n = (n / 2) * (b 1 + b n)) →
  (∀ n, A n / B n = (7 * n + 45) / (n + 3)) →
  a 5 / b 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l649_64948


namespace NUMINAMATH_CALUDE_original_length_is_one_meter_l649_64909

/-- The length of the line after erasing part of it, in centimeters -/
def remaining_length : ℝ := 76

/-- The length that was erased from the line, in centimeters -/
def erased_length : ℝ := 24

/-- The number of centimeters in one meter -/
def cm_per_meter : ℝ := 100

/-- The theorem stating that the original length of the line was 1 meter -/
theorem original_length_is_one_meter : 
  (remaining_length + erased_length) / cm_per_meter = 1 := by sorry

end NUMINAMATH_CALUDE_original_length_is_one_meter_l649_64909


namespace NUMINAMATH_CALUDE_new_average_salary_l649_64984

/-- Calculates the new average monthly salary after a change in supervisor --/
theorem new_average_salary
  (num_people : ℕ)
  (num_workers : ℕ)
  (old_average : ℚ)
  (old_supervisor_salary : ℚ)
  (new_supervisor_salary : ℚ)
  (h_num_people : num_people = 9)
  (h_num_workers : num_workers = 8)
  (h_old_average : old_average = 430)
  (h_old_supervisor : old_supervisor_salary = 870)
  (h_new_supervisor : new_supervisor_salary = 960) :
  (num_people * old_average - old_supervisor_salary + new_supervisor_salary) / num_people = 440 :=
sorry

end NUMINAMATH_CALUDE_new_average_salary_l649_64984


namespace NUMINAMATH_CALUDE_triangle_side_angle_equivalence_l649_64921

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_side_angle_equivalence (t : Triangle) :
  (t.a / Real.cos t.A = t.b / Real.cos t.B) ↔ (t.a = t.b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_angle_equivalence_l649_64921


namespace NUMINAMATH_CALUDE_toy_cost_l649_64917

theorem toy_cost (price_A price_B price_C : ℝ) 
  (h1 : 2 * price_A + price_B + 3 * price_C = 24)
  (h2 : 3 * price_A + 4 * price_B + 2 * price_C = 36) :
  price_A + price_B + price_C = 12 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_l649_64917


namespace NUMINAMATH_CALUDE_a_plus_b_value_l649_64943

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (A ∪ B a b = Set.univ) →
  (A ∩ B a b = Set.Ioc 3 4) →
  a + b = -7 := by
  sorry


end NUMINAMATH_CALUDE_a_plus_b_value_l649_64943


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l649_64907

theorem solve_exponential_equation (y : ℝ) : (1000 : ℝ)^4 = 10^y → y = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l649_64907


namespace NUMINAMATH_CALUDE_three_boys_three_girls_arrangements_l649_64960

/-- The number of possible arrangements for 3 boys and 3 girls in an alternating pattern -/
def alternating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  2 * (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating that the number of arrangements for 3 boys and 3 girls is 72 -/
theorem three_boys_three_girls_arrangements :
  alternating_arrangements 3 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_three_boys_three_girls_arrangements_l649_64960


namespace NUMINAMATH_CALUDE_bird_nest_theorem_l649_64945

/-- Represents a bird's trip information -/
structure BirdTrip where
  trips_to_x : ℕ
  trips_to_y : ℕ
  trips_to_z : ℕ
  distance_to_x : ℕ
  distance_to_y : ℕ
  distance_to_z : ℕ
  time_to_x : ℕ
  time_to_y : ℕ
  time_to_z : ℕ

def bird_a : BirdTrip :=
  { trips_to_x := 15
  , trips_to_y := 0
  , trips_to_z := 10
  , distance_to_x := 300
  , distance_to_y := 0
  , distance_to_z := 400
  , time_to_x := 30
  , time_to_y := 0
  , time_to_z := 40 }

def bird_b : BirdTrip :=
  { trips_to_x := 0
  , trips_to_y := 20
  , trips_to_z := 5
  , distance_to_x := 0
  , distance_to_y := 500
  , distance_to_z := 600
  , time_to_x := 0
  , time_to_y := 60
  , time_to_z := 50 }

def total_distance (bird : BirdTrip) : ℕ :=
  2 * (bird.trips_to_x * bird.distance_to_x +
       bird.trips_to_y * bird.distance_to_y +
       bird.trips_to_z * bird.distance_to_z)

def total_time (bird : BirdTrip) : ℕ :=
  bird.trips_to_x * bird.time_to_x +
  bird.trips_to_y * bird.time_to_y +
  bird.trips_to_z * bird.time_to_z

theorem bird_nest_theorem :
  total_distance bird_a + total_distance bird_b = 43000 ∧
  total_time bird_a + total_time bird_b = 2300 := by
  sorry

end NUMINAMATH_CALUDE_bird_nest_theorem_l649_64945


namespace NUMINAMATH_CALUDE_balls_to_boxes_count_l649_64979

/-- The number of ways to distribute n indistinguishable objects into k distinguishable groups -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 7 indistinguishable balls into 3 distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls_to_boxes : ℕ := distribute 4 3

theorem balls_to_boxes_count :
  distribute_balls_to_boxes = 15 := by sorry

end NUMINAMATH_CALUDE_balls_to_boxes_count_l649_64979


namespace NUMINAMATH_CALUDE_arithmetic_operations_l649_64955

theorem arithmetic_operations :
  ((-20) - (-14) + (-18) - 13 = -37) ∧
  (((-3/4) + (1/6) - (5/8)) / (-1/24) = 29) ∧
  ((-3^2) + (-3)^2 + 3*2 + |(-4)| = 10) ∧
  (16 / (-2)^3 - (-1/6) * (-4) + (-1)^2024 = -5/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l649_64955


namespace NUMINAMATH_CALUDE_max_mondays_in_45_days_l649_64925

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days we're considering -/
def days_considered : ℕ := 45

/-- The maximum number of Mondays in the first 45 days of a year -/
def max_mondays : ℕ := 7

/-- Theorem: The maximum number of Mondays in the first 45 days of a year is 7 -/
theorem max_mondays_in_45_days : 
  ∀ (start_day : ℕ), start_day < days_in_week →
  (∃ (monday_count : ℕ), 
    monday_count ≤ max_mondays ∧
    monday_count = (days_considered / days_in_week) + 
      (if start_day = 0 then 1 else 0)) :=
sorry

end NUMINAMATH_CALUDE_max_mondays_in_45_days_l649_64925


namespace NUMINAMATH_CALUDE_max_value_sqrt_product_max_value_achieved_l649_64930

theorem max_value_sqrt_product (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  (Real.sqrt (a * b * c * d) + Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d))) ≤ 1 :=
by sorry

theorem max_value_achieved (a b c d : Real) :
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) →
  Real.sqrt (a * b * c * d) + Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_product_max_value_achieved_l649_64930


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l649_64908

theorem factorization_of_difference_of_squares (a : ℝ) : 1 - a^2 = (1 + a) * (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l649_64908


namespace NUMINAMATH_CALUDE_power_sum_sequence_l649_64961

theorem power_sum_sequence (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h_rec : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^10 + b^10 = 123 := by
sorry

end NUMINAMATH_CALUDE_power_sum_sequence_l649_64961


namespace NUMINAMATH_CALUDE_original_savings_calculation_l649_64989

theorem original_savings_calculation (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) :
  furniture_fraction = 3 / 4 →
  tv_cost = 200 →
  (1 - furniture_fraction) * savings = tv_cost →
  savings = 800 := by
sorry

end NUMINAMATH_CALUDE_original_savings_calculation_l649_64989


namespace NUMINAMATH_CALUDE_sin_330_degrees_l649_64962

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l649_64962


namespace NUMINAMATH_CALUDE_ducks_in_lake_l649_64996

theorem ducks_in_lake (initial_ducks joining_ducks : ℕ) 
  (h1 : initial_ducks = 13) 
  (h2 : joining_ducks = 20) : 
  initial_ducks + joining_ducks = 33 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l649_64996


namespace NUMINAMATH_CALUDE_gcd_45_75_l649_64900

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_75_l649_64900


namespace NUMINAMATH_CALUDE_range_of_f_l649_64918

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | 0 ≤ y ∧ y ≤ 4 } := by sorry

end NUMINAMATH_CALUDE_range_of_f_l649_64918


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l649_64993

theorem min_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^2 - 2*z + 3) ≥ 2 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l649_64993


namespace NUMINAMATH_CALUDE_tent_donation_problem_l649_64929

theorem tent_donation_problem (total_tents : ℕ) (total_value : ℕ) 
  (cost_A : ℕ) (cost_B : ℕ) :
  total_tents = 300 →
  total_value = 260000 →
  cost_A = 800 →
  cost_B = 1000 →
  ∃ (num_A num_B : ℕ),
    num_A + num_B = total_tents ∧
    num_A * cost_A + num_B * cost_B = total_value ∧
    num_A = 200 ∧
    num_B = 100 :=
by sorry

end NUMINAMATH_CALUDE_tent_donation_problem_l649_64929


namespace NUMINAMATH_CALUDE_square_congruent_one_count_l649_64941

/-- For n ≥ 2, the number of integers x with 0 ≤ x < n such that x² ≡ 1 (mod n) 
    is equal to 2 times the number of pairs (a, b) such that ab = n and gcd(a, b) = 1 -/
theorem square_congruent_one_count (n : ℕ) (h : n ≥ 2) :
  (Finset.filter (fun x => x^2 % n = 1) (Finset.range n)).card =
  2 * (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = n ∧ Nat.gcd p.1 p.2 = 1) 
    (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card := by
  sorry

end NUMINAMATH_CALUDE_square_congruent_one_count_l649_64941


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l649_64998

theorem opposite_of_negative_2023 :
  ∃ x : ℤ, x + (-2023) = 0 ∧ x = 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l649_64998


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l649_64920

/-- Given two vectors a and b in R², prove that if they are perpendicular and have specific coordinates, then the x-coordinate of a is -2. -/
theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) :
  a.1 = x ∧ a.2 = 1 ∧ b = (3, 6) ∧ a.1 * b.1 + a.2 * b.2 = 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l649_64920


namespace NUMINAMATH_CALUDE_max_value_2x_minus_y_l649_64914

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0)
  (h2 : y + 1 ≥ 0)
  (h3 : x + y + 1 ≤ 0) :
  ∃ (max : ℝ), max = 1 ∧ ∀ x' y' : ℝ, 
    x' - y' + 1 ≥ 0 → y' + 1 ≥ 0 → x' + y' + 1 ≤ 0 → 
    2 * x' - y' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_minus_y_l649_64914


namespace NUMINAMATH_CALUDE_right_triangles_with_sqrt1001_leg_l649_64946

theorem right_triangles_with_sqrt1001_leg :
  ∃! (n : ℕ), n > 0 ∧ n = (Finset.filter 
    (fun t : ℕ × ℕ × ℕ => 
      t.1 * t.1 + 1001 = t.2.2 * t.2.2 ∧ 
      t.2.1 * t.2.1 = 1001 ∧
      t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0)
    (Finset.product (Finset.range 1000) (Finset.product (Finset.range 1000) (Finset.range 1000)))).card ∧
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_right_triangles_with_sqrt1001_leg_l649_64946


namespace NUMINAMATH_CALUDE_star_commutative_l649_64911

variable {M : Type*} [Nonempty M]
variable (star : M → M → M)

axiom left_inverse : ∀ a b : M, star (star a b) b = a
axiom right_inverse : ∀ a b : M, star a (star a b) = b

theorem star_commutative : ∀ a b : M, star a b = star b a := by sorry

end NUMINAMATH_CALUDE_star_commutative_l649_64911


namespace NUMINAMATH_CALUDE_spelling_bee_points_l649_64969

theorem spelling_bee_points : 
  let max_points : ℝ := 7
  let dulce_points : ℝ := 5
  let val_points : ℝ := 4 * (max_points + dulce_points)
  let sarah_points : ℝ := 2 * dulce_points
  let steve_points : ℝ := 2.5 * (max_points + val_points)
  let team_points : ℝ := max_points + dulce_points + val_points + sarah_points + steve_points
  let opponents_points : ℝ := 200
  team_points - opponents_points = 7.5 := by sorry

end NUMINAMATH_CALUDE_spelling_bee_points_l649_64969


namespace NUMINAMATH_CALUDE_second_number_value_l649_64937

theorem second_number_value (x y : ℚ) 
  (h1 : (1 : ℚ) / 5 * x = (5 : ℚ) / 8 * y) 
  (h2 : x + 35 = 4 * y) : 
  y = 40 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l649_64937


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l649_64995

theorem chicken_rabbit_problem :
  ∀ (chickens rabbits : ℕ),
    chickens + rabbits = 15 →
    2 * chickens + 4 * rabbits = 40 →
    chickens = 10 ∧ rabbits = 5 := by
  sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l649_64995


namespace NUMINAMATH_CALUDE_susan_coins_value_l649_64991

theorem susan_coins_value :
  ∀ (n d : ℕ),
  n + d = 30 →
  5 * n + 10 * d + 90 = 10 * n + 5 * d →
  5 * n + 10 * d = 180 := by
sorry

end NUMINAMATH_CALUDE_susan_coins_value_l649_64991


namespace NUMINAMATH_CALUDE_tangent_line_equation_l649_64997

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - 2*y + 2*Real.log 2 - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l649_64997


namespace NUMINAMATH_CALUDE_fraction_modification_result_l649_64915

theorem fraction_modification_result (a b : ℤ) (h1 : a.gcd b = 1) 
  (h2 : (a - 1) / (b - 2) = (a + 1) / b) : (a - 1) / (b - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_modification_result_l649_64915


namespace NUMINAMATH_CALUDE_greatest_sum_of_two_integers_l649_64913

theorem greatest_sum_of_two_integers (n : ℤ) : 
  (∀ m : ℤ, m * (m + 2) < 500 → m ≤ n) →
  n * (n + 2) < 500 →
  n + (n + 2) = 44 := by
sorry

end NUMINAMATH_CALUDE_greatest_sum_of_two_integers_l649_64913


namespace NUMINAMATH_CALUDE_original_mixture_volume_l649_64982

/-- Proves that given a mixture with 20% alcohol, if adding 2 litres of water
    results in a new mixture with 17.647058823529413% alcohol,
    then the original mixture volume was 15 litres. -/
theorem original_mixture_volume
  (original_alcohol_percentage : Real)
  (added_water : Real)
  (new_alcohol_percentage : Real)
  (h1 : original_alcohol_percentage = 0.20)
  (h2 : added_water = 2)
  (h3 : new_alcohol_percentage = 0.17647058823529413)
  : ∃ (original_volume : Real),
    original_volume * original_alcohol_percentage /
    (original_volume + added_water) = new_alcohol_percentage ∧
    original_volume = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_mixture_volume_l649_64982


namespace NUMINAMATH_CALUDE_expression_evaluation_l649_64977

theorem expression_evaluation (c : ℕ) (h : c = 4) : 
  (c^c - c * (c - 1)^(c - 1))^c = 148^4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l649_64977


namespace NUMINAMATH_CALUDE_tenth_term_geometric_sequence_l649_64959

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r ^ (n - 1)

theorem tenth_term_geometric_sequence :
  geometric_sequence 4 (5/3) 10 = 7812500/19683 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_geometric_sequence_l649_64959


namespace NUMINAMATH_CALUDE_value_of_a_l649_64965

theorem value_of_a (a b c : ℝ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 6) 
  (eq3 : c = 4) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l649_64965


namespace NUMINAMATH_CALUDE_no_finite_k_with_zero_difference_l649_64947

def u (n : ℕ) : ℕ := n^4 + n^2

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
  | f => fun n => f (n + 1) - f n

def iteratedΔ : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0 => id
  | k + 1 => Δ ∘ iteratedΔ k

theorem no_finite_k_with_zero_difference :
  ∀ k : ℕ, ∃ n : ℕ, (iteratedΔ k u) n ≠ 0 := by sorry

end NUMINAMATH_CALUDE_no_finite_k_with_zero_difference_l649_64947


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_l649_64970

/-- Represents different sampling methods -/
inductive SamplingMethod
| Random
| Systematic
| Stratified

/-- Represents income levels -/
inductive IncomeLevel
| High
| Middle
| Low

/-- Represents a community with different income levels -/
structure Community where
  total_households : ℕ
  high_income : ℕ
  middle_income : ℕ
  low_income : ℕ

/-- Represents a sampling scenario -/
structure SamplingScenario where
  population_size : ℕ
  sample_size : ℕ
  has_distinct_strata : Bool

/-- Determines the optimal sampling method for a given scenario -/
def optimal_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- The community described in the problem -/
def problem_community : Community :=
  { total_households := 1000
  , high_income := 250
  , middle_income := 560
  , low_income := 190 }

/-- The household study sampling scenario -/
def household_study : SamplingScenario :=
  { population_size := 1000
  , sample_size := 200
  , has_distinct_strata := true }

/-- The discussion forum sampling scenario -/
def discussion_forum : SamplingScenario :=
  { population_size := 20
  , sample_size := 6
  , has_distinct_strata := false }

theorem optimal_sampling_methods :
  optimal_sampling_method household_study = SamplingMethod.Stratified ∧
  optimal_sampling_method discussion_forum = SamplingMethod.Random :=
sorry

end NUMINAMATH_CALUDE_optimal_sampling_methods_l649_64970


namespace NUMINAMATH_CALUDE_card_trick_strategy_exists_l649_64983

/-- Represents a card in the set of 29 cards -/
def Card := Fin 29

/-- Represents the strategy for selecting two cards to show -/
def Strategy := (Card × Card) → (Card × Card)

/-- Checks if two cards are adjacent in the circular arrangement -/
def adjacent (a b : Card) : Prop :=
  b.val = (a.val % 29 + 1) ∨ a.val = (b.val % 29 + 1)

/-- Determines if a strategy is valid for guessing hidden cards -/
def valid_strategy (s : Strategy) : Prop :=
  ∀ (hidden : Card × Card),
    let shown := s hidden
    ∃! (guessed : Card × Card),
      (guessed = hidden ∧ ¬adjacent guessed.1 guessed.2) ∨
      (guessed = hidden ∧ adjacent guessed.1 guessed.2)

/-- Theorem stating that there exists a valid strategy for the card trick -/
theorem card_trick_strategy_exists : ∃ (s : Strategy), valid_strategy s := by
  sorry

end NUMINAMATH_CALUDE_card_trick_strategy_exists_l649_64983


namespace NUMINAMATH_CALUDE_range_of_m_l649_64986

theorem range_of_m : ∃ (a b : ℝ), a = 1 ∧ b = 3 ∧
  ∀ m : ℝ, (∀ x : ℝ, |m - x| < 2 → -1 < x ∧ x < 5) →
  a ≤ m ∧ m ≤ b :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l649_64986


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l649_64919

/-- An arithmetic sequence is a sequence where the difference between 
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℚ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third_term : a 3 = 5)
  (h_seventh_term : a 7 = 13) :
  a 10 = 19 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l649_64919
