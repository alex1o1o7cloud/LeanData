import Mathlib

namespace NUMINAMATH_CALUDE_rhinoceros_population_increase_l2939_293943

/-- Calculates the percentage increase in rhinoceros population given initial conditions --/
theorem rhinoceros_population_increase 
  (initial_rhinos : ℕ)
  (watering_area : ℕ)
  (grazing_per_rhino : ℕ)
  (total_expanded_area : ℕ)
  (h1 : initial_rhinos = 8000)
  (h2 : watering_area = 10000)
  (h3 : grazing_per_rhino = 100)
  (h4 : total_expanded_area = 890000) :
  (((total_expanded_area - (initial_rhinos * grazing_per_rhino + watering_area)) / grazing_per_rhino) / initial_rhinos : ℚ) = 1/10 :=
by sorry

end NUMINAMATH_CALUDE_rhinoceros_population_increase_l2939_293943


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l2939_293938

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) (M : ℝ × ℝ) :
  (2 * b - Real.sqrt 3 * c) * Real.cos A = Real.sqrt 3 * a * Real.cos C →
  B = π / 6 →
  Real.sqrt ((M.1 - (b + c) / 2)^2 + (M.2)^2) = Real.sqrt 7 →
  A = π / 6 ∧
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l2939_293938


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2939_293999

/-- 
Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
if the distance between one of its foci and an asymptote is one-fourth of its focal distance,
then the eccentricity of the hyperbola is 2√3/3.
-/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let focal_distance := 2 * c
  let asymptote_distance := b
  focal_distance = 4 * asymptote_distance →
  c / a = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2939_293999


namespace NUMINAMATH_CALUDE_min_cost_1001_grid_square_l2939_293908

/-- Represents a grid square with side length n -/
def GridSquare (n : ℕ) := {m : ℕ × ℕ | m.1 ≤ n ∧ m.2 ≤ n}

/-- The cost of coloring a single cell -/
def colorCost : ℕ := 1

/-- The minimum number of cells that need to be colored to create a complete grid square -/
def minColoredCells (n : ℕ) : ℕ := n * n + 2 * n * (n - 1)

theorem min_cost_1001_grid_square :
  minColoredCells 1001 * colorCost = 503000 :=
sorry

end NUMINAMATH_CALUDE_min_cost_1001_grid_square_l2939_293908


namespace NUMINAMATH_CALUDE_egg_distribution_l2939_293904

theorem egg_distribution (num_boxes : ℝ) (eggs_per_box : ℝ) (h1 : num_boxes = 2.0) (h2 : eggs_per_box = 1.5) :
  num_boxes * eggs_per_box = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_egg_distribution_l2939_293904


namespace NUMINAMATH_CALUDE_original_number_proof_l2939_293995

theorem original_number_proof (h1 : 213 * 16 = 3408) :
  ∃ x : ℝ, 1.6 * x = 34.080000000000005 ∧ x = 21.3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2939_293995


namespace NUMINAMATH_CALUDE_peach_expense_l2939_293932

theorem peach_expense (total berries apples : ℝ) 
  (h_total : total = 34.72)
  (h_berries : berries = 11.08)
  (h_apples : apples = 14.33) :
  total - (berries + apples) = 9.31 := by sorry

end NUMINAMATH_CALUDE_peach_expense_l2939_293932


namespace NUMINAMATH_CALUDE_pizzeria_sales_l2939_293939

/-- Calculates the total sales of a pizzeria given the prices and quantities of small and large pizzas sold. -/
theorem pizzeria_sales
  (small_price : ℕ)
  (large_price : ℕ)
  (small_quantity : ℕ)
  (large_quantity : ℕ)
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : small_quantity = 8)
  (h4 : large_quantity = 3) :
  small_price * small_quantity + large_price * large_quantity = 40 :=
by sorry

#check pizzeria_sales

end NUMINAMATH_CALUDE_pizzeria_sales_l2939_293939


namespace NUMINAMATH_CALUDE_circle_tangent_line_l2939_293953

theorem circle_tangent_line (a : ℝ) : 
  (∃ (x y : ℝ), x - y + 1 = 0 ∧ x^2 + y^2 - 2*x + 1 - a = 0 ∧ 
  ∀ (x' y' : ℝ), x' - y' + 1 = 0 → x'^2 + y'^2 - 2*x' + 1 - a ≥ 0) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_line_l2939_293953


namespace NUMINAMATH_CALUDE_complementary_fraction_irreducible_l2939_293970

theorem complementary_fraction_irreducible (a b : ℤ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : Nat.gcd a.natAbs b.natAbs = 1) : 
  Nat.gcd (b - a).natAbs b.natAbs = 1 := by
sorry

end NUMINAMATH_CALUDE_complementary_fraction_irreducible_l2939_293970


namespace NUMINAMATH_CALUDE_buffy_whiskers_l2939_293981

/-- The number of whiskers for each cat -/
structure CatWhiskers where
  juniper : ℕ
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ

/-- The conditions for the cat whiskers problem -/
def catWhiskersConditions (c : CatWhiskers) : Prop :=
  c.juniper = 12 ∧
  c.puffy = 3 * c.juniper ∧
  c.scruffy = 2 * c.puffy ∧
  c.buffy = (c.juniper + c.puffy + c.scruffy) / 3

/-- Theorem stating that given the conditions, Buffy has 40 whiskers -/
theorem buffy_whiskers (c : CatWhiskers) (h : catWhiskersConditions c) : c.buffy = 40 := by
  sorry

end NUMINAMATH_CALUDE_buffy_whiskers_l2939_293981


namespace NUMINAMATH_CALUDE_base8_subtraction_l2939_293927

/-- Converts a base-8 number represented as a list of digits to a natural number. -/
def base8ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits. -/
def natToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The subtraction operation in base 8. -/
def base8Sub (a b : List Nat) : List Nat :=
  natToBase8 (base8ToNat a - base8ToNat b)

theorem base8_subtraction :
  base8Sub [4, 5, 3] [3, 2, 6] = [1, 2, 5] := by sorry

end NUMINAMATH_CALUDE_base8_subtraction_l2939_293927


namespace NUMINAMATH_CALUDE_obtuse_triangle_x_range_l2939_293958

/-- Represents the side lengths of a triangle --/
structure TriangleSides where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is obtuse --/
def isObtuse (t : TriangleSides) : Prop :=
  (t.a ^ 2 + t.b ^ 2 < t.c ^ 2) ∨ (t.a ^ 2 + t.c ^ 2 < t.b ^ 2) ∨ (t.b ^ 2 + t.c ^ 2 < t.a ^ 2)

/-- The theorem stating the range of x for the given obtuse triangle --/
theorem obtuse_triangle_x_range :
  ∀ x : ℝ,
  let t := TriangleSides.mk 3 4 x
  isObtuse t →
  (1 < x ∧ x < Real.sqrt 7) ∨ (5 < x ∧ x < 7) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_x_range_l2939_293958


namespace NUMINAMATH_CALUDE_tim_stored_bales_l2939_293907

theorem tim_stored_bales (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 28) 
  (h2 : final_bales = 54) : 
  final_bales - initial_bales = 26 := by
  sorry

end NUMINAMATH_CALUDE_tim_stored_bales_l2939_293907


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l2939_293972

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial : 
  (∃ k : ℕ, k = 30 ∧ 
   (∀ m : ℕ, 2010^m ∣ factorial 2010 → m ≤ k) ∧
   2010^k ∣ factorial 2010) ∧
  2010 = 2 * 3 * 5 * 67 := by
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l2939_293972


namespace NUMINAMATH_CALUDE_runs_scored_for_new_average_l2939_293940

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  matches_played : ℕ
  total_runs : ℕ

/-- Calculate the batting average of a player -/
def batting_average (player : CricketPlayer) : ℚ :=
  player.total_runs / player.matches_played

/-- Calculate the total runs after a new match -/
def total_runs_after_match (player : CricketPlayer) (new_runs : ℕ) : ℕ :=
  player.total_runs + new_runs

/-- Calculate the new batting average after a match -/
def new_batting_average (player : CricketPlayer) (new_runs : ℕ) : ℚ :=
  (total_runs_after_match player new_runs) / (player.matches_played + 1)

theorem runs_scored_for_new_average 
  (player : CricketPlayer) 
  (new_runs : ℕ) :
  player.matches_played = 5 ∧ 
  batting_average player = 51 ∧
  new_batting_average player new_runs = 54 →
  new_runs = 69 := by
sorry

end NUMINAMATH_CALUDE_runs_scored_for_new_average_l2939_293940


namespace NUMINAMATH_CALUDE_suzanne_reading_l2939_293971

theorem suzanne_reading (total_pages : ℕ) (extra_pages : ℕ) (pages_left : ℕ) 
  (h1 : total_pages = 64)
  (h2 : extra_pages = 16)
  (h3 : pages_left = 18) :
  ∃ (monday_pages : ℕ), 
    monday_pages + (monday_pages + extra_pages) = total_pages - pages_left ∧ 
    monday_pages = 15 := by
  sorry

end NUMINAMATH_CALUDE_suzanne_reading_l2939_293971


namespace NUMINAMATH_CALUDE_pattern_and_application_l2939_293962

theorem pattern_and_application (n : ℕ) (a b : ℝ) :
  n > 1 →
  (n : ℝ) * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) ∧
  (a * Real.sqrt (7 / b) = Real.sqrt (a + 7 / b) → a + b = 55) :=
by sorry

end NUMINAMATH_CALUDE_pattern_and_application_l2939_293962


namespace NUMINAMATH_CALUDE_line_point_k_value_l2939_293950

/-- Given a line containing points (2, 9), (10, k), and (25, 4), prove that k = 167/23 -/
theorem line_point_k_value (k : ℚ) : 
  (∃ (m b : ℚ), 9 = m * 2 + b ∧ k = m * 10 + b ∧ 4 = m * 25 + b) → 
  k = 167 / 23 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l2939_293950


namespace NUMINAMATH_CALUDE_triangle_existence_theorem_l2939_293913

/-- The sum of angles in a triangle is 180 degrees -/
axiom triangle_angle_sum : ℝ → ℝ → ℝ → Prop

/-- A right angle is 90 degrees -/
def is_right_angle (angle : ℝ) : Prop := angle = 90

/-- An acute angle is less than 90 degrees -/
def is_acute_angle (angle : ℝ) : Prop := angle < 90

/-- An equilateral triangle has three equal angles -/
def is_equilateral_triangle (a b c : ℝ) : Prop := a = b ∧ b = c

theorem triangle_existence_theorem :
  (∀ a b c : ℝ, triangle_angle_sum a b c → is_right_angle a → b = 60 → c = 60 → False) ∧
  (∃ a b c : ℝ, triangle_angle_sum a b c ∧ is_equilateral_triangle a b c ∧ a = 60) ∧
  (∀ a b c : ℝ, triangle_angle_sum a b c → is_right_angle a → is_right_angle b → is_right_angle c → False) ∧
  (∃ a b c : ℝ, triangle_angle_sum a b c ∧ is_equilateral_triangle a b c ∧ is_acute_angle a) ∧
  (∀ a b c : ℝ, triangle_angle_sum a b c → is_right_angle a → b = 45 → c = 15 → False) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_theorem_l2939_293913


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_zero_one_four_l2939_293929

-- Define set A
def A : Set ℤ := {x | x^2 - 4*x ≤ 0}

-- Define set B
def B : Set ℤ := {y | ∃ m ∈ A, y = m^2}

-- Theorem statement
theorem A_intersect_B_equals_zero_one_four : A ∩ B = {0, 1, 4} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_zero_one_four_l2939_293929


namespace NUMINAMATH_CALUDE_no_simultaneous_cubes_l2939_293954

theorem no_simultaneous_cubes (n : ℕ) : 
  ¬(∃ (a b : ℤ), (2^(n+1) - 1 = a^3) ∧ (2^(n-1) * (2^n - 1) = b^3)) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_cubes_l2939_293954


namespace NUMINAMATH_CALUDE_square_of_binomial_l2939_293948

theorem square_of_binomial (k : ℚ) : 
  (∃ t u : ℚ, ∀ x, k * x^2 + 28 * x + 9 = (t * x + u)^2) → k = 196 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2939_293948


namespace NUMINAMATH_CALUDE_binomial_coefficient_10_3_l2939_293979

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_10_3_l2939_293979


namespace NUMINAMATH_CALUDE_final_price_calculation_l2939_293928

/-- Calculates the final price of an item after applying discounts and tax -/
theorem final_price_calculation (original_price : ℝ) 
  (first_discount_rate : ℝ) (second_discount_rate : ℝ) (tax_rate : ℝ) : 
  original_price = 200 ∧ 
  first_discount_rate = 0.5 ∧ 
  second_discount_rate = 0.25 ∧ 
  tax_rate = 0.1 → 
  original_price * (1 - first_discount_rate) * (1 - second_discount_rate) * (1 + tax_rate) = 82.5 := by
  sorry

#check final_price_calculation

end NUMINAMATH_CALUDE_final_price_calculation_l2939_293928


namespace NUMINAMATH_CALUDE_problem_statement_l2939_293969

theorem problem_statement (x y z : ℝ) (hx : x = 3) (hy : y = 4) (hz : z = 2) :
  (x^5 + 3*y^3 + z^2) / 12 = 439/12 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2939_293969


namespace NUMINAMATH_CALUDE_group_trip_cost_l2939_293986

/-- The total cost for a group trip given the number of people and cost per person -/
def total_cost (num_people : ℕ) (cost_per_person : ℕ) : ℕ :=
  num_people * cost_per_person

/-- Proof that the total cost for 11 people at $1100 each is $12100 -/
theorem group_trip_cost : total_cost 11 1100 = 12100 := by
  sorry

end NUMINAMATH_CALUDE_group_trip_cost_l2939_293986


namespace NUMINAMATH_CALUDE_remainder_2357912_div_8_l2939_293900

theorem remainder_2357912_div_8 : 2357912 % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2357912_div_8_l2939_293900


namespace NUMINAMATH_CALUDE_harry_routine_duration_is_90_l2939_293956

/-- Harry's morning routine duration --/
def harry_routine_duration : ℕ :=
  let coffee_bagel_time := 15
  let dog_walking_time := 20
  let exercise_time := 25
  let reading_eating_time := 2 * coffee_bagel_time
  coffee_bagel_time + dog_walking_time + exercise_time + reading_eating_time

/-- Theorem stating that Harry's morning routine takes 90 minutes --/
theorem harry_routine_duration_is_90 : harry_routine_duration = 90 := by
  sorry

end NUMINAMATH_CALUDE_harry_routine_duration_is_90_l2939_293956


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l2939_293987

theorem part_to_whole_ratio 
  (N P : ℚ) 
  (h1 : (1/4) * (2/5) * P = 15) 
  (h2 : (40/100) * N = 180) : 
  P/N = 1/6 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l2939_293987


namespace NUMINAMATH_CALUDE_modular_exponentiation_l2939_293988

theorem modular_exponentiation (n : ℕ) :
  (47^2051 - 25^2051) % 5 = 3 := by sorry

end NUMINAMATH_CALUDE_modular_exponentiation_l2939_293988


namespace NUMINAMATH_CALUDE_only_triangle_combines_l2939_293991

/-- Represents a regular polygon --/
structure RegularPolygon where
  interior_angle : ℝ

/-- Checks if two regular polygons can be combined to form a 360° vertex --/
def can_combine (p1 p2 : RegularPolygon) : Prop :=
  ∃ (n m : ℕ), n * p1.interior_angle + m * p2.interior_angle = 360

/-- The given regular polygon with 150° interior angle --/
def given_polygon : RegularPolygon :=
  { interior_angle := 150 }

/-- Regular quadrilateral --/
def quadrilateral : RegularPolygon :=
  { interior_angle := 90 }

/-- Regular hexagon --/
def hexagon : RegularPolygon :=
  { interior_angle := 120 }

/-- Regular octagon --/
def octagon : RegularPolygon :=
  { interior_angle := 135 }

/-- Equilateral triangle --/
def equilateral_triangle : RegularPolygon :=
  { interior_angle := 60 }

/-- Theorem stating that only the equilateral triangle can be combined with the given polygon --/
theorem only_triangle_combines :
  ¬(can_combine given_polygon quadrilateral) ∧
  ¬(can_combine given_polygon hexagon) ∧
  ¬(can_combine given_polygon octagon) ∧
  (can_combine given_polygon equilateral_triangle) :=
sorry

end NUMINAMATH_CALUDE_only_triangle_combines_l2939_293991


namespace NUMINAMATH_CALUDE_product_b3_b17_l2939_293917

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The main theorem -/
theorem product_b3_b17 (a b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_cond : 3 * a 1 - (a 8)^2 + 3 * a 15 = 0)
  (h_eq : a 8 = b 10) :
  b 3 * b 17 = 36 := by
  sorry

end NUMINAMATH_CALUDE_product_b3_b17_l2939_293917


namespace NUMINAMATH_CALUDE_combinations_equal_thirty_l2939_293924

/-- The number of color options available -/
def num_colors : ℕ := 5

/-- The number of painting method options available -/
def num_methods : ℕ := 3

/-- The number of finish type options available -/
def num_finishes : ℕ := 2

/-- The total number of combinations of color, painting method, and finish type -/
def total_combinations : ℕ := num_colors * num_methods * num_finishes

/-- Theorem stating that the total number of combinations is 30 -/
theorem combinations_equal_thirty : total_combinations = 30 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_thirty_l2939_293924


namespace NUMINAMATH_CALUDE_function_properties_l2939_293926

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x^2 + (1 - 2*m) * x + 1

theorem function_properties (m : ℝ) :
  (f m 1 = -1) →
  (∃ (x_max : ℝ), ∀ (x : ℝ), x > 0 → f m x ≤ f m x_max) →
  (∀ (x : ℝ), x > 0 → f m x ≤ 0) →
  (m = 1 ∧ 
   (∃ (x_max : ℝ), f m x_max = 1/4 - Real.log 2) ∧
   (∀ (ε : ℝ), ε > 0 → ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ f m x₁ < f m x₂) ∧
   (∀ (k : ℤ), k < 1 → ∃ (x : ℝ), x > 0 ∧ f k x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2939_293926


namespace NUMINAMATH_CALUDE_money_distribution_l2939_293901

/-- Given three people A, B, and C with some amount of money, prove that B and C together have 340 rupees. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 500 →  -- Total money between A, B, and C
  A + C = 200 →      -- Money A and C have together
  C = 40 →           -- Money C has
  B + C = 340 :=     -- Prove that B and C have 340 together
by sorry

end NUMINAMATH_CALUDE_money_distribution_l2939_293901


namespace NUMINAMATH_CALUDE_value_of_x_l2939_293903

theorem value_of_x : ∀ (w y z x : ℤ), 
  w = 65 → 
  z = w + 25 → 
  y = z + 15 → 
  x = y + 7 → 
  x = 112 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l2939_293903


namespace NUMINAMATH_CALUDE_bobby_has_more_books_l2939_293925

/-- Given that Bobby has 142 books and Kristi has 78 books, 
    prove that Bobby has 64 more books than Kristi. -/
theorem bobby_has_more_books : 
  let bobby_books : ℕ := 142
  let kristi_books : ℕ := 78
  bobby_books - kristi_books = 64 := by sorry

end NUMINAMATH_CALUDE_bobby_has_more_books_l2939_293925


namespace NUMINAMATH_CALUDE_percentage_same_grade_l2939_293905

/-- Represents the grade a student can receive -/
inductive Grade
| A
| B
| C
| D
| E

/-- Represents the grade distribution for a single test -/
structure GradeDistribution :=
  (A : Nat)
  (B : Nat)
  (C : Nat)
  (D : Nat)
  (E : Nat)

/-- The total number of students in the class -/
def totalStudents : Nat := 50

/-- The grade distribution for the first test -/
def firstTestDistribution : GradeDistribution := {
  A := 7,
  B := 12,
  C := 19,
  D := 8,
  E := 4
}

/-- The grade distribution for the second test -/
def secondTestDistribution : GradeDistribution := {
  A := 8,
  B := 16,
  C := 14,
  D := 7,
  E := 5
}

/-- The number of students who received the same grade on both tests -/
def sameGradeCount : Nat := 20

/-- Theorem: The percentage of students who received the same grade on both tests is 40% -/
theorem percentage_same_grade :
  (sameGradeCount : ℚ) / (totalStudents : ℚ) * 100 = 40 := by sorry

end NUMINAMATH_CALUDE_percentage_same_grade_l2939_293905


namespace NUMINAMATH_CALUDE_eggs_remaining_l2939_293963

theorem eggs_remaining (initial_eggs : ℕ) (eggs_taken : ℕ) (eggs_left : ℕ) : 
  initial_eggs = 47 → eggs_taken = 5 → eggs_left = initial_eggs - eggs_taken → eggs_left = 42 :=
by sorry

end NUMINAMATH_CALUDE_eggs_remaining_l2939_293963


namespace NUMINAMATH_CALUDE_max_M_value_l2939_293937

theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (eq1 : x - 2*y = z - 2*u) (eq2 : 2*y*z = u*x) (h_zy : z ≥ y) :
  ∃ M : ℝ, M > 0 ∧ M ≤ z/y ∧ ∀ N : ℝ, (N > 0 ∧ N ≤ z/y → N ≤ M) ∧ M = 6 + 4*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_M_value_l2939_293937


namespace NUMINAMATH_CALUDE_sum_of_coefficients_plus_a_l2939_293959

theorem sum_of_coefficients_plus_a (a : ℝ) (as : Fin 2007 → ℝ) :
  (∀ x : ℝ, (1 - 2 * x)^2006 = a + (Finset.sum (Finset.range 2007) (λ i => as i * x^i))) →
  Finset.sum (Finset.range 2007) (λ i => a + as i) = 2006 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_plus_a_l2939_293959


namespace NUMINAMATH_CALUDE_science_fiction_readers_l2939_293994

theorem science_fiction_readers
  (total : ℕ)
  (literary : ℕ)
  (both : ℕ)
  (h1 : total = 150)
  (h2 : literary = 90)
  (h3 : both = 60) :
  total = literary + (total - literary - both) - both :=
by sorry

end NUMINAMATH_CALUDE_science_fiction_readers_l2939_293994


namespace NUMINAMATH_CALUDE_polynomial_sum_squares_l2939_293957

theorem polynomial_sum_squares (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (x - 2)^8 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                        a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8) →
  (a₂ + a₄ + a₆ + a₈)^2 - (a₁ + a₃ + a₅ + a₇)^2 = -255 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_squares_l2939_293957


namespace NUMINAMATH_CALUDE_smaller_angle_at_4_oclock_l2939_293946

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The hour we're considering -/
def target_hour : ℕ := 4

/-- Calculates the angle between clock hands at a given hour -/
def clock_angle (hour : ℕ) : ℕ := 
  (hour * full_circle_degrees) / clock_hours

theorem smaller_angle_at_4_oclock : 
  min (clock_angle target_hour) (full_circle_degrees - clock_angle target_hour) = 120 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_at_4_oclock_l2939_293946


namespace NUMINAMATH_CALUDE_complex_product_simplification_l2939_293964

theorem complex_product_simplification (a b x y : ℝ) : 
  (a * x + Complex.I * b * y) * (a * x - Complex.I * b * y) = a^2 * x^2 - b^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_simplification_l2939_293964


namespace NUMINAMATH_CALUDE_penelope_food_amount_l2939_293990

/-- Amount of food animals eat per day -/
structure AnimalFood where
  greta : ℝ
  penelope : ℝ
  milton : ℝ
  elmer : ℝ

/-- Conditions for animal food consumption -/
def valid_food_amounts (food : AnimalFood) : Prop :=
  food.penelope = 10 * food.greta ∧
  food.milton = food.greta / 100 ∧
  food.elmer = 4000 * food.milton ∧
  food.elmer = food.penelope + 60

theorem penelope_food_amount (food : AnimalFood) 
  (h : valid_food_amounts food) : food.penelope = 20 := by
  sorry

#check penelope_food_amount

end NUMINAMATH_CALUDE_penelope_food_amount_l2939_293990


namespace NUMINAMATH_CALUDE_yogurt_cost_yogurt_cost_is_one_l2939_293935

/-- The cost of yogurt given Seth's purchase information -/
theorem yogurt_cost (ice_cream_quantity : ℕ) (yogurt_quantity : ℕ) 
  (ice_cream_cost : ℕ) (extra_spent : ℕ) : ℕ :=
  let total_ice_cream_cost := ice_cream_quantity * ice_cream_cost
  let yogurt_cost := (total_ice_cream_cost - extra_spent) / yogurt_quantity
  yogurt_cost

/-- Proof that the cost of each carton of yogurt is $1 -/
theorem yogurt_cost_is_one :
  yogurt_cost 20 2 6 118 = 1 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_cost_yogurt_cost_is_one_l2939_293935


namespace NUMINAMATH_CALUDE_min_square_side_for_given_dimensions_l2939_293930

/-- Represents the dimensions of a table and cube -/
structure TableDimensions where
  length : ℕ
  breadth : ℕ
  cube_side : ℕ

/-- Calculates the minimum side length of a square formed by arranging tables -/
def min_square_side (td : TableDimensions) : ℕ :=
  2 * td.length + 2 * td.breadth

/-- Theorem stating the minimum side length of the square formed by tables -/
theorem min_square_side_for_given_dimensions :
  ∀ (td : TableDimensions),
    td.length = 12 →
    td.breadth = 16 →
    td.cube_side = 4 →
    min_square_side td = 56 :=
by
  sorry

#eval min_square_side ⟨12, 16, 4⟩

end NUMINAMATH_CALUDE_min_square_side_for_given_dimensions_l2939_293930


namespace NUMINAMATH_CALUDE_pencils_per_row_l2939_293921

/-- Given a total of 154 pencils arranged in 14 rows with an equal number of pencils in each row,
    prove that there are 11 pencils in each row. -/
theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 154)
  (h2 : num_rows = 14)
  (h3 : total_pencils = num_rows * pencils_per_row) :
  pencils_per_row = 11 := by
  sorry

#check pencils_per_row

end NUMINAMATH_CALUDE_pencils_per_row_l2939_293921


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2939_293936

theorem tan_alpha_value (α β : Real) 
  (h1 : Real.tan (α + β) = 3/5) 
  (h2 : Real.tan β = 1/3) : 
  Real.tan α = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2939_293936


namespace NUMINAMATH_CALUDE_min_abs_z_l2939_293911

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 2*I) + Complex.abs (z - 5) = 7) :
  Complex.abs z ≥ 10 / Real.sqrt 29 ∧ ∃ w : ℂ, Complex.abs (w - 2*I) + Complex.abs (w - 5) = 7 ∧ Complex.abs w = 10 / Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_l2939_293911


namespace NUMINAMATH_CALUDE_special_triangle_DE_length_l2939_293922

/-- Triangle ABC with given side lengths and DE parallel to BC containing the incenter -/
structure SpecialTriangle where
  -- Side lengths
  AB : ℝ
  AC : ℝ
  BC : ℝ
  -- Points D and E
  D : ℝ × ℝ
  E : ℝ × ℝ
  -- Condition that DE is parallel to BC
  DE_parallel_BC : Bool
  -- Condition that DE contains the incenter
  DE_contains_incenter : Bool

/-- The length of DE in the special triangle -/
def length_DE (t : SpecialTriangle) : ℝ := sorry

/-- Theorem stating the length of DE in the special case -/
theorem special_triangle_DE_length :
  ∀ t : SpecialTriangle,
  t.AB = 28 ∧ t.AC = 29 ∧ t.BC = 26 ∧ t.DE_parallel_BC ∧ t.DE_contains_incenter →
  length_DE t = 806 / 57 := by sorry

end NUMINAMATH_CALUDE_special_triangle_DE_length_l2939_293922


namespace NUMINAMATH_CALUDE_certain_number_problem_l2939_293974

theorem certain_number_problem (x : ℝ) : 
  ((x + 20) * 2) / 2 - 2 = 88 / 2 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2939_293974


namespace NUMINAMATH_CALUDE_blue_notebook_cost_l2939_293977

/-- The cost of each blue notebook given Mike's purchase details -/
theorem blue_notebook_cost 
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (red_price : ℕ)
  (green_notebooks : ℕ)
  (green_price : ℕ)
  (h1 : total_spent = 37)
  (h2 : total_notebooks = 12)
  (h3 : red_notebooks = 3)
  (h4 : red_price = 4)
  (h5 : green_notebooks = 2)
  (h6 : green_price = 2)
  (h7 : total_notebooks = red_notebooks + green_notebooks + (total_notebooks - red_notebooks - green_notebooks))
  : (total_spent - red_notebooks * red_price - green_notebooks * green_price) / (total_notebooks - red_notebooks - green_notebooks) = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_notebook_cost_l2939_293977


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2939_293923

-- Problem 1
theorem problem_1 : 23 + (-16) - (-7) = 14 := by sorry

-- Problem 2
theorem problem_2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by sorry

-- Problem 3
theorem problem_3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -10/3 := by sorry

-- Problem 4
theorem problem_4 : -1^4 - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2939_293923


namespace NUMINAMATH_CALUDE_area_ratio_is_one_fourth_l2939_293965

/-- A square with vertices A, B, C, D -/
structure Square where
  side_length : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- A particle moving along the edges of a square -/
structure Particle where
  position : ℝ → ℝ × ℝ  -- position as a function of time
  speed : ℝ

/-- The region enclosed by the path of the midpoint -/
def enclosed_region (p1 p2 : Particle) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating the ratio of areas -/
theorem area_ratio_is_one_fourth (sq : Square) (p1 p2 : Particle) :
  sq.A = (0, 0) ∧ 
  sq.B = (sq.side_length, 0) ∧ 
  sq.C = (sq.side_length, sq.side_length) ∧ 
  sq.D = (0, sq.side_length) ∧
  p1.position 0 = sq.A ∧
  p2.position 0 = ((sq.C.1 + sq.D.1) / 2, sq.C.2) ∧
  p1.speed = p2.speed →
  area (enclosed_region p1 p2) / area {p | p.1 ∈ Set.Icc 0 sq.side_length ∧ p.2 ∈ Set.Icc 0 sq.side_length} = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_one_fourth_l2939_293965


namespace NUMINAMATH_CALUDE_largest_n_for_product_2010_l2939_293914

def is_arithmetic_sequence (s : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, s (n + 1) - s n = d

theorem largest_n_for_product_2010 (a b : ℕ → ℤ) 
  (ha : is_arithmetic_sequence a) 
  (hb : is_arithmetic_sequence b)
  (h1 : a 1 = 1 ∧ b 1 = 1)
  (h2 : a 2 ≤ b 2)
  (h3 : ∃ n : ℕ, a n * b n = 2010)
  : (∃ n : ℕ, a n * b n = 2010 ∧ ∀ m : ℕ, a m * b m = 2010 → m ≤ n) ∧
    (∀ n : ℕ, a n * b n = 2010 → n ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_2010_l2939_293914


namespace NUMINAMATH_CALUDE_inequality_proof_l2939_293912

theorem inequality_proof (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_condition : x^2 + y^2 + z^2 = x + y + z) :
  (x + 1) / Real.sqrt (x^5 + x + 1) + (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) ≥ 3 ∧
  ((x + 1) / Real.sqrt (x^5 + x + 1) + (y + 1) / Real.sqrt (y^5 + y + 1) + 
   (z + 1) / Real.sqrt (z^5 + z + 1) = 3 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2939_293912


namespace NUMINAMATH_CALUDE_negation_equivalence_l2939_293989

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2939_293989


namespace NUMINAMATH_CALUDE_math_expressions_equality_l2939_293909

theorem math_expressions_equality : 
  (∃ (a b c d : ℝ), 
    a = (Real.sqrt 5 - (Real.sqrt 3 + Real.sqrt 15) / (Real.sqrt 6 * Real.sqrt 2)) ∧
    b = ((Real.sqrt 48 - 4 * Real.sqrt (1/8)) - (3 * Real.sqrt (1/3) - 2 * Real.sqrt 0.5)) ∧
    c = ((3 + Real.sqrt 5) * (3 - Real.sqrt 5) - (Real.sqrt 3 - 1)^2) ∧
    d = ((- Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) - Real.sqrt ((-3)^2) + 1 / (2 - Real.sqrt 5)) ∧
    a = -1 ∧
    b = 3 * Real.sqrt 3 ∧
    c = 2 * Real.sqrt 3 ∧
    d = -3 - Real.sqrt 5) := by
  sorry

#check math_expressions_equality

end NUMINAMATH_CALUDE_math_expressions_equality_l2939_293909


namespace NUMINAMATH_CALUDE_quartic_roots_arithmetic_sequence_l2939_293945

theorem quartic_roots_arithmetic_sequence (m n : ℚ) : 
  (∃ a b c d : ℚ, 
    (a^2 - 2*a + m) * (a^2 - 2*a + n) = 0 ∧
    (b^2 - 2*b + m) * (b^2 - 2*b + n) = 0 ∧
    (c^2 - 2*c + m) * (c^2 - 2*c + n) = 0 ∧
    (d^2 - 2*d + m) * (d^2 - 2*d + n) = 0 ∧
    a = 1/4 ∧
    b - a = c - b ∧
    c - b = d - c) →
  |m - n| = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quartic_roots_arithmetic_sequence_l2939_293945


namespace NUMINAMATH_CALUDE_line_slope_range_l2939_293975

/-- Given two points A and B, and a line l that intersects line segment AB,
    prove that the range of possible slopes for line l is (-∞,-4] ∪ [1/2,+∞). -/
theorem line_slope_range (a : ℝ) :
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (4, 0)
  let l := {(x, y) : ℝ × ℝ | a * x + y - 2 * a + 1 = 0}
  let intersects := ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    (∃ (x y : ℝ), (x, y) = (1 - t) • A + t • B ∧ (x, y) ∈ l)
  intersects →
  (a ≤ -4 ∨ a ≥ 1/2) :=
sorry

end NUMINAMATH_CALUDE_line_slope_range_l2939_293975


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l2939_293910

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

theorem largest_digit_divisible_by_6 :
  ∀ N : ℕ, N ≤ 9 →
    (is_divisible_by_6 (2345 * 10 + N) → N ≤ 4) ∧
    (is_divisible_by_6 (2345 * 10 + 4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l2939_293910


namespace NUMINAMATH_CALUDE_givenSampleIsValidSystematic_l2939_293961

/-- Checks if a list of integers represents a valid systematic sample -/
def isValidSystematicSample (sample : List Nat) (populationSize : Nat) : Prop :=
  let n := sample.length
  ∃ k : Nat,
    k > 0 ∧
    (∀ i : Fin n, sample[i] = k * (i + 1)) ∧
    sample.all (· ≤ populationSize)

/-- The given sample -/
def givenSample : List Nat := [3, 13, 23, 33, 43]

/-- The theorem stating that the given sample is a valid systematic sample -/
theorem givenSampleIsValidSystematic :
  isValidSystematicSample givenSample 50 := by
  sorry


end NUMINAMATH_CALUDE_givenSampleIsValidSystematic_l2939_293961


namespace NUMINAMATH_CALUDE_min_prime_sum_l2939_293997

theorem min_prime_sum (m n p : ℕ) : 
  m.Prime ∧ n.Prime ∧ p.Prime →
  ∃ k : ℕ, k = 47 + m ∧ k = 53 + n ∧ k = 71 + p →
  m + n + p ≥ 57 :=
by sorry

end NUMINAMATH_CALUDE_min_prime_sum_l2939_293997


namespace NUMINAMATH_CALUDE_custom_operation_equation_solution_l2939_293960

-- Define the custom operation
def star (a b : ℝ) : ℝ := 4 * a * b

-- Theorem statement
theorem custom_operation_equation_solution :
  ∀ x : ℝ, star x x + 2 * (star 1 x) - star 2 2 = 0 → x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equation_solution_l2939_293960


namespace NUMINAMATH_CALUDE_water_formed_moles_l2939_293918

-- Define the chemical species
inductive ChemicalSpecies
| NaOH
| HCl
| H2O
| NaCl

-- Define a function to represent the stoichiometric coefficient in the balanced equation
def stoichiometric_coefficient (reactant product : ChemicalSpecies) : ℕ :=
  match reactant, product with
  | ChemicalSpecies.NaOH, ChemicalSpecies.H2O => 1
  | ChemicalSpecies.HCl, ChemicalSpecies.H2O => 1
  | _, _ => 0

-- Define the given amounts of reactants
def initial_NaOH : ℕ := 2
def initial_HCl : ℕ := 2

-- State the theorem
theorem water_formed_moles :
  min initial_NaOH initial_HCl = 
  stoichiometric_coefficient ChemicalSpecies.NaOH ChemicalSpecies.H2O * 
  stoichiometric_coefficient ChemicalSpecies.HCl ChemicalSpecies.H2O * 2 :=
by sorry

end NUMINAMATH_CALUDE_water_formed_moles_l2939_293918


namespace NUMINAMATH_CALUDE_line_plane_relationship_l2939_293944

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Defines when a line is contained in a plane -/
def Line3D.containedIn (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Defines when a line is parallel to a plane -/
def Line3D.parallelTo (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Defines when a line intersects a plane -/
def Line3D.intersects (l : Line3D) (p : Plane3D) : Prop := sorry

/-- The main theorem -/
theorem line_plane_relationship (a : Line3D) (α : Plane3D) :
  ¬(a.containedIn α) → (a.parallelTo α ∨ a.intersects α) := by
  sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l2939_293944


namespace NUMINAMATH_CALUDE_square_comparison_l2939_293968

theorem square_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * b * Real.sqrt (a^2 + b^2)) / (a^2 + a * b + b^2) < (a * b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_square_comparison_l2939_293968


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l2939_293949

/-- Amanda's ticket sales problem -/
theorem amanda_ticket_sales
  (total_tickets : ℕ)
  (first_day_sales : ℕ)
  (third_day_sales : ℕ)
  (h1 : total_tickets = 80)
  (h2 : first_day_sales = 20)
  (h3 : third_day_sales = 28) :
  total_tickets - first_day_sales - third_day_sales = 32 := by
  sorry

#check amanda_ticket_sales

end NUMINAMATH_CALUDE_amanda_ticket_sales_l2939_293949


namespace NUMINAMATH_CALUDE_represent_383_l2939_293992

/-- Given a number of hundreds, tens, and ones, calculate the represented number. -/
def representedNumber (hundreds tens ones : ℕ) : ℕ :=
  100 * hundreds + 10 * tens + ones

/-- Prove that 3 hundreds, 8 tens, and 3 ones represent the number 383. -/
theorem represent_383 : representedNumber 3 8 3 = 383 := by
  sorry

end NUMINAMATH_CALUDE_represent_383_l2939_293992


namespace NUMINAMATH_CALUDE_triangle_area_l2939_293942

/-- The area of a triangle is half the product of two adjacent sides and the sine of the angle between them. -/
theorem triangle_area (a b : ℝ) (γ : ℝ) (ha : a > 0) (hb : b > 0) (hγ : 0 < γ ∧ γ < π) :
  ∃ (S : ℝ), S = (1 / 2) * a * b * Real.sin γ ∧ S > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2939_293942


namespace NUMINAMATH_CALUDE_mars_bars_count_l2939_293984

theorem mars_bars_count (total : ℕ) (snickers : ℕ) (butterfingers : ℕ) 
  (h1 : total = 12)
  (h2 : snickers = 3)
  (h3 : butterfingers = 7) :
  total - snickers - butterfingers = 2 := by
  sorry

end NUMINAMATH_CALUDE_mars_bars_count_l2939_293984


namespace NUMINAMATH_CALUDE_certain_number_equation_l2939_293952

theorem certain_number_equation (x : ℝ) : x = 25 ↔ 0.8 * 45 = 4/5 * x + 16 := by sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2939_293952


namespace NUMINAMATH_CALUDE_part_one_part_two_l2939_293919

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem part_one : 
  ∀ x : ℝ, (p x 1 ∧ q x) → (2 < x ∧ x < 3) :=
sorry

-- Part 2
theorem part_two :
  (∀ x a : ℝ, q x → p x a) ∧ 
  (∃ x a : ℝ, p x a ∧ ¬(q x)) →
  (∀ a : ℝ, 1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2939_293919


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l2939_293982

theorem triangle_abc_proof (a b c A B C S_ΔABC : Real) 
  (h1 : a = Real.sqrt 3)
  (h2 : b = Real.sqrt 2)
  (h3 : A = π / 3)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π)
  (h8 : Real.sin A / a = Real.sin B / b)
  (h9 : S_ΔABC = (1 / 2) * a * b * Real.sin C) :
  B = π / 4 ∧ S_ΔABC = (3 + Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l2939_293982


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l2939_293951

/-- The minimum number of additional coins needed to distribute distinct, positive numbers of coins to a given number of friends, starting with a given number of coins. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let required_coins := (num_friends * (num_friends + 1)) / 2
  if required_coins > initial_coins then
    required_coins - initial_coins
  else
    0

/-- The problem statement as a theorem -/
theorem alex_coin_distribution :
  min_additional_coins 15 97 = 23 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l2939_293951


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2939_293966

theorem no_positive_integer_solutions :
  ¬ ∃ (x₁ x₂ : ℕ), 903 * x₁ + 731 * x₂ = 1106 := by
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2939_293966


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l2939_293902

theorem cos_2alpha_value (α : Real) (h : Real.tan (α - π/4) = -1/3) : 
  Real.cos (2 * α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l2939_293902


namespace NUMINAMATH_CALUDE_students_playing_neither_l2939_293998

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 35 →
  football = 26 →
  tennis = 20 →
  both = 17 →
  total - (football + tennis - both) = 6 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_neither_l2939_293998


namespace NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l2939_293978

theorem square_of_negative_sqrt_two : (-Real.sqrt 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sqrt_two_l2939_293978


namespace NUMINAMATH_CALUDE_orchid_rose_difference_is_nine_l2939_293967

/-- Flower quantities and ratios in a vase --/
structure FlowerVase where
  initial_roses : ℕ
  initial_orchids : ℕ
  initial_tulips : ℕ
  final_roses : ℕ
  final_orchids : ℕ
  final_tulips : ℕ
  rose_orchid_ratio : ℚ
  rose_tulip_ratio : ℚ

/-- The difference between orchids and roses after adding new flowers --/
def orchid_rose_difference (v : FlowerVase) : ℕ :=
  v.final_orchids - v.final_roses

/-- Theorem stating the difference between orchids and roses is 9 --/
theorem orchid_rose_difference_is_nine (v : FlowerVase)
  (h1 : v.initial_roses = 7)
  (h2 : v.initial_orchids = 12)
  (h3 : v.initial_tulips = 5)
  (h4 : v.final_roses = 11)
  (h5 : v.final_orchids = 20)
  (h6 : v.final_tulips = 10)
  (h7 : v.rose_orchid_ratio = 2/5)
  (h8 : v.rose_tulip_ratio = 3/5) :
  orchid_rose_difference v = 9 := by
  sorry

#eval orchid_rose_difference {
  initial_roses := 7,
  initial_orchids := 12,
  initial_tulips := 5,
  final_roses := 11,
  final_orchids := 20,
  final_tulips := 10,
  rose_orchid_ratio := 2/5,
  rose_tulip_ratio := 3/5
}

end NUMINAMATH_CALUDE_orchid_rose_difference_is_nine_l2939_293967


namespace NUMINAMATH_CALUDE_not_all_prime_l2939_293931

theorem not_all_prime (a₁ a₂ a₃ : ℕ) : 
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃ →
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 →
  a₁ ∣ (a₂ + a₃ + a₂ * a₃) →
  a₂ ∣ (a₃ + a₁ + a₃ * a₁) →
  a₃ ∣ (a₁ + a₂ + a₁ * a₂) →
  ¬(Prime a₁ ∧ Prime a₂ ∧ Prime a₃) :=
by sorry

end NUMINAMATH_CALUDE_not_all_prime_l2939_293931


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2939_293916

/-- Given a rhombus with diagonals of 14 inches and 48 inches, its perimeter is 100 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 48) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2939_293916


namespace NUMINAMATH_CALUDE_equation_solution_l2939_293955

theorem equation_solution (x : ℝ) (h : x ≠ -1) :
  (x^2 + 2*x + 3) / (x + 1) = x + 3 ↔ x = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2939_293955


namespace NUMINAMATH_CALUDE_pies_sold_in_week_l2939_293996

/-- Calculates the total number of pies sold in a week -/
def total_pies_sold (pies_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  pies_per_day * days_in_week

/-- Proves that the total number of pies sold in a week is 56 -/
theorem pies_sold_in_week : total_pies_sold 8 7 = 56 := by
  sorry

end NUMINAMATH_CALUDE_pies_sold_in_week_l2939_293996


namespace NUMINAMATH_CALUDE_parallel_lines_shapes_l2939_293973

/-- Two parallel lines with marked points -/
structure ParallelLines :=
  (line1 : Finset ℕ)
  (line2 : Finset ℕ)
  (h1 : line1.card = 10)
  (h2 : line2.card = 11)

/-- The number of triangles formed by the points on parallel lines -/
def num_triangles (pl : ParallelLines) : ℕ :=
  pl.line1.card * Nat.choose pl.line2.card 2 + pl.line2.card * Nat.choose pl.line1.card 2

/-- The number of quadrilaterals formed by the points on parallel lines -/
def num_quadrilaterals (pl : ParallelLines) : ℕ :=
  Nat.choose pl.line1.card 2 * Nat.choose pl.line2.card 2

theorem parallel_lines_shapes (pl : ParallelLines) :
  num_triangles pl = 1045 ∧ num_quadrilaterals pl = 2475 := by
  sorry

#eval num_triangles ⟨Finset.range 10, Finset.range 11, rfl, rfl⟩
#eval num_quadrilaterals ⟨Finset.range 10, Finset.range 11, rfl, rfl⟩

end NUMINAMATH_CALUDE_parallel_lines_shapes_l2939_293973


namespace NUMINAMATH_CALUDE_committee_count_l2939_293941

theorem committee_count (n m k : ℕ) (h1 : n = 8) (h2 : m = 2) (h3 : k = 5) :
  (Nat.choose n k) - (Nat.choose (n - m) (k - m)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_l2939_293941


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2939_293934

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 150 →
  crossing_time = 12.499 →
  ∃ (speed : ℝ), abs (speed - 72) < 0.1 ∧ speed = (train_length + bridge_length) / crossing_time * 3.6 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2939_293934


namespace NUMINAMATH_CALUDE_car_lot_power_windows_l2939_293915

theorem car_lot_power_windows 
  (total : ℕ) 
  (air_bags : ℕ) 
  (both : ℕ) 
  (neither : ℕ) 
  (h1 : total = 65)
  (h2 : air_bags = 45)
  (h3 : both = 12)
  (h4 : neither = 2) :
  ∃ power_windows : ℕ, power_windows = 30 ∧ 
    total = air_bags + power_windows - both + neither :=
by sorry

end NUMINAMATH_CALUDE_car_lot_power_windows_l2939_293915


namespace NUMINAMATH_CALUDE_range_of_H_l2939_293920

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ y ∈ Set.Icc (-4) 4 := by sorry

end NUMINAMATH_CALUDE_range_of_H_l2939_293920


namespace NUMINAMATH_CALUDE_distance_of_problem_lines_l2939_293947

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  direction : ℝ × ℝ

/-- The distance between two parallel lines -/
def distance_between_parallel_lines (lines : ParallelLines) : ℝ :=
  sorry

/-- The specific parallel lines from the problem -/
def problem_lines : ParallelLines :=
  { point1 := (3, -4)
  , point2 := (-1, 1)
  , direction := (2, -5) }

theorem distance_of_problem_lines :
  distance_between_parallel_lines problem_lines = (150 * Real.sqrt 2) / 29 :=
sorry

end NUMINAMATH_CALUDE_distance_of_problem_lines_l2939_293947


namespace NUMINAMATH_CALUDE_remainder_of_Q_l2939_293933

-- Define the polynomial Q
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom Q_div_21 : ∃ P₁ : ℝ → ℝ, ∀ x, Q x = (x - 21) * (P₁ x) + 105
axiom Q_div_105 : ∃ P₂ : ℝ → ℝ, ∀ x, Q x = (x - 105) * (P₂ x) + 21

-- Theorem statement
theorem remainder_of_Q : 
  ∃ P : ℝ → ℝ, ∀ x, Q x = (x - 21) * (x - 105) * (P x) + (-x + 126) := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_Q_l2939_293933


namespace NUMINAMATH_CALUDE_hexagon_with_90_degree_angle_l2939_293985

/-- A hexagon with angles in geometric progression has an angle of 90 degrees. -/
theorem hexagon_with_90_degree_angle :
  ∃ (a r : ℝ), 
    a > 0 ∧ r > 0 ∧
    a + a*r + a*r^2 + a*r^3 + a*r^4 + a*r^5 = 720 ∧
    (a = 90 ∨ a*r = 90 ∨ a*r^2 = 90 ∨ a*r^3 = 90 ∨ a*r^4 = 90 ∨ a*r^5 = 90) :=
by sorry

end NUMINAMATH_CALUDE_hexagon_with_90_degree_angle_l2939_293985


namespace NUMINAMATH_CALUDE_min_savings_theorem_l2939_293980

/-- Represents Kathleen's savings and spending -/
structure KathleenFinances where
  june_savings : ℕ
  july_savings : ℕ
  august_savings : ℕ
  school_supplies_cost : ℕ
  clothes_cost : ℕ
  amount_left : ℕ

/-- The minimum amount Kathleen needs to save to get $25 from her aunt -/
def min_savings_for_bonus (k : KathleenFinances) : ℕ :=
  k.amount_left

theorem min_savings_theorem (k : KathleenFinances) 
  (h1 : k.june_savings = 21)
  (h2 : k.july_savings = 46)
  (h3 : k.august_savings = 45)
  (h4 : k.school_supplies_cost = 12)
  (h5 : k.clothes_cost = 54)
  (h6 : k.amount_left = 46)
  (h7 : k.june_savings + k.july_savings + k.august_savings - k.school_supplies_cost - k.clothes_cost = k.amount_left) :
  min_savings_for_bonus k = k.amount_left :=
by sorry

end NUMINAMATH_CALUDE_min_savings_theorem_l2939_293980


namespace NUMINAMATH_CALUDE_f_has_minimum_at_one_point_five_l2939_293906

def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

theorem f_has_minimum_at_one_point_five :
  ∃ (y : ℝ), ∀ (x : ℝ), f x ≥ f (3/2) := by
  sorry

end NUMINAMATH_CALUDE_f_has_minimum_at_one_point_five_l2939_293906


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2939_293976

-- Problem 1
theorem problem_1 : -23 + 58 - (-5) = 40 := by sorry

-- Problem 2
theorem problem_2 : (5/8 + 1/6 - 3/4) * 24 = 1 := by sorry

-- Problem 3
theorem problem_3 : -3^2 - (-5 - 0.2 / (4/5) * (-2)^2) = -3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2939_293976


namespace NUMINAMATH_CALUDE_linear_function_passes_through_points_l2939_293993

/-- A linear function passing through (-1, 4) also passes through (1, 0) -/
theorem linear_function_passes_through_points :
  ∀ k : ℝ, (4 = k * (-1) - k) → (0 = k * 1 - k) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_points_l2939_293993


namespace NUMINAMATH_CALUDE_number_equals_five_l2939_293983

theorem number_equals_five (N x : ℝ) (h1 : N / (4 + 1/x) = 1) (h2 : x = 1) : N = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_five_l2939_293983
