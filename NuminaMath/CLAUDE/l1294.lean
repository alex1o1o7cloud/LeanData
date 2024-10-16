import Mathlib

namespace NUMINAMATH_CALUDE_point_coordinates_l1294_129455

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (p : CartesianPoint) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : CartesianPoint) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : CartesianPoint) : ℝ :=
  |p.x|

theorem point_coordinates
  (p : CartesianPoint)
  (h1 : in_third_quadrant p)
  (h2 : distance_to_x_axis p = 5)
  (h3 : distance_to_y_axis p = 6) :
  p.x = -6 ∧ p.y = -5 :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l1294_129455


namespace NUMINAMATH_CALUDE_max_min_sum_of_quadratic_expression_l1294_129415

theorem max_min_sum_of_quadratic_expression (a b : ℝ) 
  (h : a^2 + a*b + b^2 = 3) : 
  let f := fun (x y : ℝ) => x^2 - x*y + y^2
  ∃ (M m : ℝ), (∀ x y, f x y ≤ M ∧ m ≤ f x y) ∧ M + m = 10 := by
sorry

end NUMINAMATH_CALUDE_max_min_sum_of_quadratic_expression_l1294_129415


namespace NUMINAMATH_CALUDE_right_triangle_k_values_l1294_129434

/-- A right triangle ABC with vectors AB and AC -/
structure RightTriangle where
  AB : ℝ × ℝ
  AC : ℝ × ℝ
  is_right : Bool

/-- The possible k values for a right triangle with AB = (2, 3) and AC = (1, k) -/
def possible_k_values : Set ℝ :=
  {-2/3, 11/3, (3 + Real.sqrt 13)/2, (3 - Real.sqrt 13)/2}

/-- Theorem stating that k must be one of the possible values -/
theorem right_triangle_k_values (triangle : RightTriangle) 
  (h1 : triangle.AB = (2, 3)) 
  (h2 : triangle.AC = (1, triangle.AC.snd)) 
  (h3 : triangle.is_right = true) : 
  triangle.AC.snd ∈ possible_k_values := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_k_values_l1294_129434


namespace NUMINAMATH_CALUDE_square_coverage_l1294_129467

theorem square_coverage : 
  let large_square_area := (5/4)^2
  let unit_square_area := 1
  let num_unit_squares := 3
  num_unit_squares * unit_square_area ≥ large_square_area := by
sorry

end NUMINAMATH_CALUDE_square_coverage_l1294_129467


namespace NUMINAMATH_CALUDE_circle_equation_l1294_129435

/-- Given a circle with center (2, -3) intercepted by the line 2x + 3y - 8 = 0
    with a chord length of 4√3, prove that its standard equation is (x-2)² + (y+3)² = 25 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -3)
  let line (x y : ℝ) : ℝ := 2*x + 3*y - 8
  let chord_length : ℝ := 4 * Real.sqrt 3
  ∃ (r : ℝ), r > 0 ∧ 
    (∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 ↔ 
      ((p.1 - 2)^2 + (p.2 + 3)^2 = 25 ∧ 
       ∃ (q : ℝ × ℝ), line q.1 q.2 = 0 ∧ 
         (q.1 - p.1)^2 + (q.2 - p.2)^2 ≤ chord_length^2)) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1294_129435


namespace NUMINAMATH_CALUDE_brownies_calculation_l1294_129464

/-- The number of brownies in each batch -/
def brownies_per_batch : ℕ := 200

/-- The number of batches baked -/
def num_batches : ℕ := 10

/-- The fraction of brownies set aside for the bake sale -/
def bake_sale_fraction : ℚ := 3/4

/-- The fraction of remaining brownies put in a container -/
def container_fraction : ℚ := 3/5

/-- The number of brownies given out -/
def brownies_given_out : ℕ := 20

theorem brownies_calculation (b : ℕ) (h : b = brownies_per_batch) : 
  (1 - bake_sale_fraction) * (1 - container_fraction) * (b * num_batches) = brownies_given_out := by
  sorry

#check brownies_calculation

end NUMINAMATH_CALUDE_brownies_calculation_l1294_129464


namespace NUMINAMATH_CALUDE_twenty_percent_of_twenty_l1294_129420

theorem twenty_percent_of_twenty : (20 : ℝ) / 100 * 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_of_twenty_l1294_129420


namespace NUMINAMATH_CALUDE_not_divisible_5n_minus_1_by_4n_minus_1_l1294_129423

theorem not_divisible_5n_minus_1_by_4n_minus_1 (n : ℕ) :
  ¬ (5^n - 1 ∣ 4^n - 1) := by sorry

end NUMINAMATH_CALUDE_not_divisible_5n_minus_1_by_4n_minus_1_l1294_129423


namespace NUMINAMATH_CALUDE_hour_hand_rotation_3_to_6_l1294_129403

/-- The number of segments in a clock face. -/
def clock_segments : ℕ := 12

/-- The number of degrees in a full rotation. -/
def full_rotation : ℕ := 360

/-- The number of hours between 3 o'clock and 6 o'clock. -/
def hours_passed : ℕ := 3

/-- The degree measure of the rotation of the hour hand from 3 o'clock to 6 o'clock. -/
def hour_hand_rotation : ℕ := (full_rotation / clock_segments) * hours_passed

theorem hour_hand_rotation_3_to_6 :
  hour_hand_rotation = 90 := by sorry

end NUMINAMATH_CALUDE_hour_hand_rotation_3_to_6_l1294_129403


namespace NUMINAMATH_CALUDE_intersection_in_second_quadrant_l1294_129407

theorem intersection_in_second_quadrant (k : ℝ) :
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y - x = 2 * k) →
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y - x = 2 * k ∧ x < 0 ∧ y > 0) →
  0 < k ∧ k < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_second_quadrant_l1294_129407


namespace NUMINAMATH_CALUDE_pages_per_day_l1294_129408

theorem pages_per_day (chapters : ℕ) (total_pages : ℕ) (days : ℕ) 
  (h1 : chapters = 41) 
  (h2 : total_pages = 450) 
  (h3 : days = 30) :
  total_pages / days = 15 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l1294_129408


namespace NUMINAMATH_CALUDE_cubic_function_property_l1294_129437

theorem cubic_function_property (p q r s : ℝ) :
  (∀ x : ℝ, p * x^3 + q * x^2 + r * x + s = x * (x - 1) * (x + 2) / 6) →
  5 * p - 3 * q + 2 * r - s = 5 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1294_129437


namespace NUMINAMATH_CALUDE_gcd_max_value_l1294_129431

theorem gcd_max_value (m : ℕ+) : 
  Nat.gcd (14 * m.val + 4) (9 * m.val + 2) ≤ 8 ∧ 
  ∃ n : ℕ+, Nat.gcd (14 * n.val + 4) (9 * n.val + 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_max_value_l1294_129431


namespace NUMINAMATH_CALUDE_coin_division_sum_25_l1294_129426

/-- Represents the sum of products for coin divisions -/
def sum_of_products (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

/-- Theorem: The sum of products for 25 coins is 300 -/
theorem coin_division_sum_25 :
  sum_of_products 25 = 300 := by
  sorry

#eval sum_of_products 25  -- Should output 300

end NUMINAMATH_CALUDE_coin_division_sum_25_l1294_129426


namespace NUMINAMATH_CALUDE_eight_percent_problem_l1294_129460

theorem eight_percent_problem (x : ℝ) : (8 / 100) * x = 64 → x = 800 := by
  sorry

end NUMINAMATH_CALUDE_eight_percent_problem_l1294_129460


namespace NUMINAMATH_CALUDE_part_one_part_two_l1294_129476

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m-9)*x + m^2 - 9*m ≥ 0}

-- Define the complement of B in ℝ
def C_R_B (m : ℝ) : Set ℝ := (Set.univ : Set ℝ) \ B m

-- Part 1: If A ∩ B = [-3, 3], then m = 12
theorem part_one (m : ℝ) : A ∩ B m = Set.Icc (-3) 3 → m = 12 := by sorry

-- Part 2: If A ⊆ C_ℝB, then 5 < m < 6
theorem part_two (m : ℝ) : A ⊆ C_R_B m → 5 < m ∧ m < 6 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1294_129476


namespace NUMINAMATH_CALUDE_average_weight_increase_l1294_129446

theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 50 + 70
  let new_average := new_total / 8
  new_average - initial_average = 2.5 := by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l1294_129446


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l1294_129478

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Murtha's pebble collection over 20 days -/
theorem murtha_pebble_collection : sum_of_first_n 20 = 210 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebble_collection_l1294_129478


namespace NUMINAMATH_CALUDE_oxen_count_l1294_129468

/-- Represents the number of oxen put by each person -/
structure Oxen where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the number of months the oxen graze -/
structure Months where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total cost based on oxen and months -/
def totalCost (o : Oxen) (m : Months) : ℚ :=
  o.a * m.a + o.b * m.b + o.c * m.c

/-- The main theorem to prove -/
theorem oxen_count (total_rent : ℚ) (c_rent : ℚ) (o : Oxen) (m : Months) :
  total_rent = 245 ∧
  o.b = 12 ∧
  o.c = 15 ∧
  m.a = 7 ∧
  m.b = 5 ∧
  m.c = 3 ∧
  c_rent = 62.99999999999999 ∧
  c_rent = (o.c * m.c / totalCost o m) * total_rent →
  o.a = 17 := by
  sorry


end NUMINAMATH_CALUDE_oxen_count_l1294_129468


namespace NUMINAMATH_CALUDE_stock_price_decrease_l1294_129444

theorem stock_price_decrease (a : ℝ) (n : ℕ) (h1 : a > 0) : a * (0.99 ^ n) < a := by
  sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l1294_129444


namespace NUMINAMATH_CALUDE_unique_intersection_l1294_129424

/-- The value of k for which |z - 4| = 3|z + 2| intersects |z| = k at exactly one point -/
def k : ℝ := 5.5

/-- The set of complex numbers z satisfying |z - 4| = 3|z + 2| -/
def S : Set ℂ := {z : ℂ | Complex.abs (z - 4) = 3 * Complex.abs (z + 2)}

theorem unique_intersection :
  ∀ z : ℂ, z ∈ S ↔ Complex.abs z = k ∧ ∀ w : ℂ, w ∈ S → Complex.abs w = k → w = z :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l1294_129424


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1294_129439

theorem lcm_from_product_and_hcf (a b : ℕ+) (h1 : a * b = 17820) (h2 : Nat.gcd a b = 12) :
  Nat.lcm a b = 1485 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1294_129439


namespace NUMINAMATH_CALUDE_consecutive_primes_as_greatest_divisors_l1294_129456

theorem consecutive_primes_as_greatest_divisors (p q : ℕ) 
  (hp : Prime p) (hq : Prime q) (hpq : p < q) (hqp : q < 2 * p) :
  ∃ n : ℕ, 
    (∃ k : ℕ+, n = k * p ∧ ∀ m : ℕ, m > p → m.Prime → ¬(m ∣ n)) ∧
    (∃ l : ℕ+, n + 1 = l * q ∧ ∀ m : ℕ, m > q → m.Prime → ¬(m ∣ (n + 1))) ∨
    (∃ k : ℕ+, n = k * q ∧ ∀ m : ℕ, m > q → m.Prime → ¬(m ∣ n)) ∧
    (∃ l : ℕ+, n + 1 = l * p ∧ ∀ m : ℕ, m > p → m.Prime → ¬(m ∣ (n + 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_primes_as_greatest_divisors_l1294_129456


namespace NUMINAMATH_CALUDE_square_difference_601_599_l1294_129458

theorem square_difference_601_599 : (601 : ℤ)^2 - (599 : ℤ)^2 = 2400 := by sorry

end NUMINAMATH_CALUDE_square_difference_601_599_l1294_129458


namespace NUMINAMATH_CALUDE_distribution_theorem_l1294_129461

/-- Represents the number of communities --/
def n : ℕ := 5

/-- Represents the number of fitness equipment --/
def k : ℕ := 7

/-- Represents the number of communities that must receive at least 2 items --/
def m : ℕ := 2

/-- Represents the minimum number of items each of the m communities must receive --/
def min_items : ℕ := 2

/-- The number of ways to distribute k identical items among n recipients,
    where m specific recipients must receive at least min_items each --/
def distribution_schemes (n k m min_items : ℕ) : ℕ := sorry

theorem distribution_theorem : distribution_schemes n k m min_items = 35 := by sorry

end NUMINAMATH_CALUDE_distribution_theorem_l1294_129461


namespace NUMINAMATH_CALUDE_base8_addition_l1294_129443

/-- Addition in base 8 -/
def base8_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 8 -/
def to_base8 (n : ℕ) : ℕ := sorry

/-- Conversion from base 8 to base 10 -/
def from_base8 (n : ℕ) : ℕ := sorry

theorem base8_addition : base8_add (from_base8 12) (from_base8 157) = from_base8 171 := by sorry

end NUMINAMATH_CALUDE_base8_addition_l1294_129443


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l1294_129471

/-- The radius of the inscribed circle in a right-angled triangle -/
theorem inscribed_circle_radius_right_triangle (a b c r : ℝ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  r = (a * b) / (a + b + c) :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l1294_129471


namespace NUMINAMATH_CALUDE_g_inverse_property_l1294_129495

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 2 * x

theorem g_inverse_property (c d : ℝ) :
  (∀ x, g c d (g c d x) = x) → c + d = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_g_inverse_property_l1294_129495


namespace NUMINAMATH_CALUDE_bird_nest_problem_l1294_129406

theorem bird_nest_problem (birds : ℕ) (difference : ℕ) (nests : ℕ) : 
  birds = 6 → difference = 3 → birds - nests = difference → nests = 3 := by
  sorry

end NUMINAMATH_CALUDE_bird_nest_problem_l1294_129406


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1294_129481

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ+), x^2 + 2*y^2 = 2*x^3 - x := by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1294_129481


namespace NUMINAMATH_CALUDE_total_individual_packs_l1294_129430

def cookies_packs : ℕ := 3
def cookies_per_pack : ℕ := 4
def noodles_packs : ℕ := 4
def noodles_per_pack : ℕ := 8
def juice_packs : ℕ := 5
def juice_per_pack : ℕ := 6
def snacks_packs : ℕ := 2
def snacks_per_pack : ℕ := 10

theorem total_individual_packs :
  cookies_packs * cookies_per_pack +
  noodles_packs * noodles_per_pack +
  juice_packs * juice_per_pack +
  snacks_packs * snacks_per_pack = 94 := by
  sorry

end NUMINAMATH_CALUDE_total_individual_packs_l1294_129430


namespace NUMINAMATH_CALUDE_expression_simplification_l1294_129447

theorem expression_simplification (x y : ℚ) (hx : x = 1/8) (hy : y = -4) :
  ((x * y - 2) * (x * y + 2) - 2 * x^2 * y^2 + 4) / (-x * y) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1294_129447


namespace NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l1294_129466

theorem max_value_of_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 15 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_x_plus_inverse_l1294_129466


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1294_129483

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  (1 - 1 / (a + 1)) * ((a^2 + 2*a + 1) / a) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1294_129483


namespace NUMINAMATH_CALUDE_missing_carton_dimension_l1294_129427

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dims : BoxDimensions) : ℝ :=
  dims.length * dims.width * dims.height

/-- Theorem: Given the dimensions of a carton and soap box, if 300 soap boxes fit exactly in the carton,
    then the missing dimension of the carton is 25 inches -/
theorem missing_carton_dimension
  (x : ℝ)
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h1 : carton = { length := x, width := 48, height := 60 })
  (h2 : soap = { length := 8, width := 6, height := 5 })
  (h3 : (boxVolume carton) / (boxVolume soap) = 300)
  : x = 25 := by
  sorry

#check missing_carton_dimension

end NUMINAMATH_CALUDE_missing_carton_dimension_l1294_129427


namespace NUMINAMATH_CALUDE_brothers_snowballs_l1294_129451

theorem brothers_snowballs (janet_snowballs : ℕ) (janet_percentage : ℚ) : 
  janet_snowballs = 50 → janet_percentage = 1/4 → 
  (1 - janet_percentage) * (janet_snowballs / janet_percentage) = 150 := by
  sorry

end NUMINAMATH_CALUDE_brothers_snowballs_l1294_129451


namespace NUMINAMATH_CALUDE_group_ratio_theorem_l1294_129477

/-- Represents the group composition and average ages -/
structure GroupComposition where
  avg_age : ℝ
  doc_age : ℝ
  law_age : ℝ
  eng_age : ℝ
  doc_count : ℝ
  law_count : ℝ
  eng_count : ℝ

/-- Theorem stating the ratios of group members based on given average ages -/
theorem group_ratio_theorem (g : GroupComposition) 
  (h1 : g.avg_age = 45)
  (h2 : g.doc_age = 40)
  (h3 : g.law_age = 55)
  (h4 : g.eng_age = 35)
  (h5 : g.avg_age * (g.doc_count + g.law_count + g.eng_count) = 
        g.doc_age * g.doc_count + g.law_age * g.law_count + g.eng_age * g.eng_count) :
  g.doc_count / g.law_count = 1 ∧ g.eng_count / g.law_count = 2 := by
  sorry

#check group_ratio_theorem

end NUMINAMATH_CALUDE_group_ratio_theorem_l1294_129477


namespace NUMINAMATH_CALUDE_right_handed_players_count_l1294_129492

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) : 
  total_players = 70 →
  throwers = 46 →
  (total_players - throwers) / 3 * 2 + throwers = 62 :=
by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l1294_129492


namespace NUMINAMATH_CALUDE_min_value_abc_l1294_129485

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  ∃ (min : ℝ), min = 1/2916 ∧ ∀ x, x = a^3 * b^2 * c → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_abc_l1294_129485


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1294_129400

theorem no_solution_for_equation : ¬∃ x : ℝ, 1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1294_129400


namespace NUMINAMATH_CALUDE_fraction_reduction_l1294_129432

theorem fraction_reduction (b y : ℝ) (h : 4 * b^2 + y^4 ≠ 0) :
  ((Real.sqrt (4 * b^2 + y^4) - (y^4 - 4 * b^2) / Real.sqrt (4 * b^2 + y^4)) / (4 * b^2 + y^4)) ^ (2/3) = 
  (8 * b^2) / (4 * b^2 + y^4) :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l1294_129432


namespace NUMINAMATH_CALUDE_sum_of_bases_equals_1500_l1294_129402

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (13 ^ i)) 0

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (14 ^ i)) 0

/-- The main theorem to prove -/
theorem sum_of_bases_equals_1500 :
  let num1 := base13ToBase10 [6, 2, 3]
  let num2 := base14ToBase10 [9, 12, 4]
  num1 + num2 = 1500 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equals_1500_l1294_129402


namespace NUMINAMATH_CALUDE_quadratic_roots_special_case_l1294_129421

theorem quadratic_roots_special_case (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : d^2 + c*d + d = 0) : c = 1 ∧ d = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_special_case_l1294_129421


namespace NUMINAMATH_CALUDE_swimming_frequency_difference_l1294_129484

def camden_total : ℕ := 16
def susannah_total : ℕ := 24
def weeks_in_month : ℕ := 4

theorem swimming_frequency_difference :
  (susannah_total / weeks_in_month) - (camden_total / weeks_in_month) = 2 :=
by sorry

end NUMINAMATH_CALUDE_swimming_frequency_difference_l1294_129484


namespace NUMINAMATH_CALUDE_zero_in_interval_l1294_129416

-- Define the function f(x) = x³ - 2x - 1
def f (x : ℝ) : ℝ := x^3 - 2*x - 1

-- State the theorem
theorem zero_in_interval :
  (f 1.5 < 0) → (f 2 > 0) → ∃ x, x ∈ Set.Ioo 1.5 2 ∧ f x = 0 := by
  sorry

-- Note: Set.Ioo represents an open interval (1.5, 2)

end NUMINAMATH_CALUDE_zero_in_interval_l1294_129416


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l1294_129404

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_abc_theorem (t : Triangle) 
  (h1 : t.b * Real.sin t.C = Real.sqrt 3)
  (h2 : t.B = π / 4)
  (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = 9 / 2) :
  t.c = Real.sqrt 6 ∧ t.b = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l1294_129404


namespace NUMINAMATH_CALUDE_cube_cutting_l1294_129429

theorem cube_cutting (n : ℕ) : 
  (∃ s : ℕ, n > s ∧ n^3 - s^3 = 152) → n = 6 := by
sorry

end NUMINAMATH_CALUDE_cube_cutting_l1294_129429


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l1294_129498

theorem sugar_solution_percentage (initial_sugar_percentage : ℝ) 
  (replaced_fraction : ℝ) (final_sugar_percentage : ℝ) :
  initial_sugar_percentage = 22 →
  replaced_fraction = 1/4 →
  final_sugar_percentage = 35 →
  let remaining_fraction := 1 - replaced_fraction
  let initial_sugar := initial_sugar_percentage * remaining_fraction
  let added_sugar := final_sugar_percentage - initial_sugar
  added_sugar / replaced_fraction = 74 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l1294_129498


namespace NUMINAMATH_CALUDE_function_six_monotonic_intervals_l1294_129453

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * |x^3| - (a/2) * x^2 + (3-a) * |x| + b

theorem function_six_monotonic_intervals (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-x)) →
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (∀ x : ℝ, x > 0 → (x^2 - a*x + (3-a) = 0 ↔ (x = x₁ ∨ x = x₂)))) →
  2 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_function_six_monotonic_intervals_l1294_129453


namespace NUMINAMATH_CALUDE_floor_sum_example_l1294_129454

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by sorry

end NUMINAMATH_CALUDE_floor_sum_example_l1294_129454


namespace NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_palindromes_l1294_129491

def three_digit_palindrome (a b : ℕ) : ℕ := 100 * a + 10 * b + a

def is_valid_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ n = three_digit_palindrome a b

theorem greatest_common_factor_of_three_digit_palindromes :
  ∃ g : ℕ, g > 0 ∧
  (∀ n : ℕ, is_valid_palindrome n → g ∣ n) ∧
  (∀ d : ℕ, d > 0 → (∀ n : ℕ, is_valid_palindrome n → d ∣ n) → d ≤ g) ∧
  g = 1 := by sorry

end NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_palindromes_l1294_129491


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1294_129479

theorem abs_sum_inequality (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + 6| > a) → a < 5 := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1294_129479


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1294_129418

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 10
  let θ : ℝ := 4 * π / 3
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x = -5 * Real.sqrt 3) ∧ (y = -15 / 2) ∧ (z = 5) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1294_129418


namespace NUMINAMATH_CALUDE_inequality_proof_l1294_129457

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1/2) :
  1/(1-a) + 1/(1-b) ≥ 4 ∧ (1/(1-a) + 1/(1-b) = 4 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1294_129457


namespace NUMINAMATH_CALUDE_stating_isosceles_triangle_with_special_bisectors_l1294_129496

/-- Represents an isosceles triangle with angle bisectors -/
structure IsoscelesTriangle where
  -- Base angle of the isosceles triangle
  β : Real
  -- Ratio of the lengths of two angle bisectors
  bisector_ratio : Real

/-- 
  Theorem stating the approximate angles of an isosceles triangle 
  where one angle bisector is twice the length of another
-/
theorem isosceles_triangle_with_special_bisectors 
  (triangle : IsoscelesTriangle) 
  (h1 : triangle.bisector_ratio = 2) 
  (h2 : 76.9 ≤ triangle.β ∧ triangle.β ≤ 77.1) : 
  25.9 ≤ 180 - 2 * triangle.β ∧ 180 - 2 * triangle.β ≤ 26.1 := by
  sorry

#check isosceles_triangle_with_special_bisectors

end NUMINAMATH_CALUDE_stating_isosceles_triangle_with_special_bisectors_l1294_129496


namespace NUMINAMATH_CALUDE_train_length_train_length_is_120_l1294_129438

/-- The length of a train that overtakes a motorbike -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) 
  (h1 : train_speed = 100) 
  (h2 : motorbike_speed = 64) 
  (h3 : overtake_time = 12) : ℝ :=
let train_speed_ms := train_speed * 1000 / 3600
let motorbike_speed_ms := motorbike_speed * 1000 / 3600
let relative_speed := train_speed_ms - motorbike_speed_ms
120

/-- The length of the train is 120 meters -/
theorem train_length_is_120 (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) 
  (h1 : train_speed = 100) 
  (h2 : motorbike_speed = 64) 
  (h3 : overtake_time = 12) : 
  train_length train_speed motorbike_speed overtake_time h1 h2 h3 = 120 := by
sorry

end NUMINAMATH_CALUDE_train_length_train_length_is_120_l1294_129438


namespace NUMINAMATH_CALUDE_gretchen_walking_time_l1294_129411

/-- The number of minutes Gretchen should walk for every 90 minutes of sitting -/
def walking_time_per_90_min : ℕ := 10

/-- The number of minutes in 90 minutes -/
def sitting_time_per_break : ℕ := 90

/-- The number of hours Gretchen spends working at her desk -/
def work_hours : ℕ := 6

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculates the total walking time for Gretchen based on her work hours -/
def total_walking_time : ℕ :=
  (work_hours * minutes_per_hour / sitting_time_per_break) * walking_time_per_90_min

theorem gretchen_walking_time :
  total_walking_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_walking_time_l1294_129411


namespace NUMINAMATH_CALUDE_cindy_marbles_l1294_129413

/-- Proves that Cindy initially had 500 marbles given the conditions -/
theorem cindy_marbles : 
  ∀ (initial_marbles : ℕ),
  (initial_marbles - 4 * 80 > 0) →
  (4 * (initial_marbles - 4 * 80) = 720) →
  initial_marbles = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_cindy_marbles_l1294_129413


namespace NUMINAMATH_CALUDE_ratio_to_percent_l1294_129401

theorem ratio_to_percent (a b : ℕ) (h : a = 15 ∧ b = 25) : 
  (a : ℝ) / b * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percent_l1294_129401


namespace NUMINAMATH_CALUDE_intersection_singleton_l1294_129452

/-- The set A defined by the equation y = ax + 1 -/
def A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a * p.1 + 1}

/-- The set B defined by the equation y = |x| -/
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = |p.1|}

/-- Theorem stating the condition for A ∩ B to be a singleton set -/
theorem intersection_singleton (a : ℝ) : (A a ∩ B).Finite ∧ (A a ∩ B).Nonempty ↔ a ≥ 1 ∨ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_singleton_l1294_129452


namespace NUMINAMATH_CALUDE_triangle_30_60_90_divisible_l1294_129480

/-- A triangle with angles 30°, 60°, and 90° -/
structure Triangle30_60_90 where
  -- We define the triangle using its angles
  angle1 : Real
  angle2 : Real
  angle3 : Real
  angle1_eq : angle1 = 30
  angle2_eq : angle2 = 60
  angle3_eq : angle3 = 90
  sum_angles : angle1 + angle2 + angle3 = 180

/-- A representation of three equal triangles -/
structure ThreeEqualTriangles where
  -- We define three triangles and their equality
  triangle1 : Triangle30_60_90
  triangle2 : Triangle30_60_90
  triangle3 : Triangle30_60_90
  equality12 : triangle1 = triangle2
  equality23 : triangle2 = triangle3

/-- Theorem stating that a 30-60-90 triangle can be divided into three equal triangles -/
theorem triangle_30_60_90_divisible (t : Triangle30_60_90) : 
  ∃ (et : ThreeEqualTriangles), True :=
sorry

end NUMINAMATH_CALUDE_triangle_30_60_90_divisible_l1294_129480


namespace NUMINAMATH_CALUDE_total_seashells_eq_sum_l1294_129493

/-- The number of seashells Dan found on the beach -/
def total_seashells : ℕ := 56

/-- The number of seashells Dan gave to Jessica -/
def seashells_given : ℕ := 34

/-- The number of seashells Dan has left -/
def seashells_left : ℕ := 22

/-- Theorem stating that the total number of seashells is equal to
    the sum of seashells given away and seashells left -/
theorem total_seashells_eq_sum :
  total_seashells = seashells_given + seashells_left := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_eq_sum_l1294_129493


namespace NUMINAMATH_CALUDE_complement_A_intersection_nonempty_union_equals_B_l1294_129494

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem for the complement of A
theorem complement_A : (Set.univ \ A) = {x : ℝ | x ≤ -1 ∨ x > 2} := by sorry

-- Theorem for the range of a when A ∩ B ≠ ∅
theorem intersection_nonempty (a : ℝ) : (A ∩ B a).Nonempty → a > -1 := by sorry

-- Theorem for the range of a when A ∪ B = B
theorem union_equals_B (a : ℝ) : A ∪ B a = B a → a > 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersection_nonempty_union_equals_B_l1294_129494


namespace NUMINAMATH_CALUDE_max_points_tournament_l1294_129445

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (games_per_pair : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculate the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  t.num_teams.choose 2 * t.games_per_pair

/-- Calculate the maximum points achievable by the top three teams -/
def max_points_top_three (t : Tournament) : ℕ :=
  let games_against_lower := (t.num_teams - 3) * t.games_per_pair
  let points_from_lower := games_against_lower * t.points_for_win
  let games_among_top := 2 * t.games_per_pair
  let points_among_top := games_among_top * t.points_for_win / 2
  points_from_lower + points_among_top

/-- The main theorem stating the maximum points for top three teams -/
theorem max_points_tournament :
  ∀ t : Tournament,
    t.num_teams = 8 →
    t.games_per_pair = 2 →
    t.points_for_win = 3 →
    t.points_for_draw = 1 →
    t.points_for_loss = 0 →
    max_points_top_three t = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_points_tournament_l1294_129445


namespace NUMINAMATH_CALUDE_prism_with_18_edges_has_8_faces_l1294_129469

/-- A prism is a polyhedron with two congruent parallel faces (bases) and whose other faces (lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ :=
  let L := p.edges / 3
  2 + L

theorem prism_with_18_edges_has_8_faces (p : Prism) (h : p.edges = 18) :
  num_faces p = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_18_edges_has_8_faces_l1294_129469


namespace NUMINAMATH_CALUDE_inequality_proof_l1294_129473

theorem inequality_proof (a b c : ℝ) 
  (ha : -1 < a ∧ a < -2/3) 
  (hb : -1/3 < b ∧ b < 0) 
  (hc : c > 1) : 
  1/c < 1/(b-a) ∧ 1/(b-a) < 1/(a*b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1294_129473


namespace NUMINAMATH_CALUDE_factorization_problems_l1294_129442

theorem factorization_problems (x y : ℝ) : 
  ((x^2 + y^2)^2 - 4*x^2*y^2 = (x + y)^2 * (x - y)^2) ∧ 
  (3*x^3 - 12*x^2*y + 12*x*y^2 = 3*x*(x - 2*y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1294_129442


namespace NUMINAMATH_CALUDE_frog_eyes_in_pond_l1294_129440

/-- The number of eyes a frog has -/
def eyes_per_frog : ℕ := 2

/-- The number of frogs in the pond -/
def frogs_in_pond : ℕ := 4

/-- The total number of frog eyes in the pond -/
def total_frog_eyes : ℕ := frogs_in_pond * eyes_per_frog

theorem frog_eyes_in_pond : total_frog_eyes = 8 := by
  sorry

end NUMINAMATH_CALUDE_frog_eyes_in_pond_l1294_129440


namespace NUMINAMATH_CALUDE_irrational_sqrt_three_others_rational_l1294_129482

theorem irrational_sqrt_three_others_rational :
  ¬ (∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 3 = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (-1 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (1/2 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (3.14 : ℝ) = a / b) :=
by sorry

end NUMINAMATH_CALUDE_irrational_sqrt_three_others_rational_l1294_129482


namespace NUMINAMATH_CALUDE_root_values_l1294_129475

-- Define the polynomial
def polynomial (a b c : ℝ) (x : ℂ) : ℂ := x^3 + a*x^2 + b*x - c

-- State the theorem
theorem root_values (a b c : ℝ) :
  (polynomial a b c (1 - 2*I) = 0) →
  (polynomial a b c (2 - I) = 0) →
  (a, b, c) = (-6, 21, -30) := by
  sorry

end NUMINAMATH_CALUDE_root_values_l1294_129475


namespace NUMINAMATH_CALUDE_inequality_proof_l1294_129462

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b > 0) :
  a / b^2 + b / a^2 > 1 / a + 1 / b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1294_129462


namespace NUMINAMATH_CALUDE_children_attendance_l1294_129405

/-- Proves the number of children attending a concert given ticket prices and total revenue -/
theorem children_attendance (adult_price : ℕ) (adult_count : ℕ) (total_revenue : ℕ) : 
  adult_price = 26 →
  adult_count = 183 →
  total_revenue = 5122 →
  ∃ (child_count : ℕ), 
    adult_price * adult_count + (adult_price / 2) * child_count = total_revenue ∧
    child_count = 28 := by
  sorry

end NUMINAMATH_CALUDE_children_attendance_l1294_129405


namespace NUMINAMATH_CALUDE_probability_of_winning_pair_l1294_129410

/-- Represents the color of a card -/
inductive Color
| Red
| Green

/-- Represents the label of a card -/
inductive Label
| A | B | C | D | E

/-- Represents a card in the deck -/
structure Card where
  color : Color
  label : Label

/-- The deck of cards -/
def deck : Finset Card := sorry

/-- Predicate for a winning pair of cards -/
def is_winning_pair (c1 c2 : Card) : Prop :=
  c1.color = c2.color ∨ c1.label = c2.label

/-- The number of cards in the deck -/
def deck_size : ℕ := sorry

/-- The number of winning pairs -/
def winning_pairs : ℕ := sorry

theorem probability_of_winning_pair :
  (winning_pairs : ℚ) / (deck_size.choose 2 : ℚ) = 51 / 91 := by sorry

end NUMINAMATH_CALUDE_probability_of_winning_pair_l1294_129410


namespace NUMINAMATH_CALUDE_bookcase_weight_excess_l1294_129459

theorem bookcase_weight_excess :
  let bookcase_limit : ℝ := 80
  let hardcover_count : ℕ := 70
  let hardcover_weight : ℝ := 0.5
  let textbook_count : ℕ := 30
  let textbook_weight : ℝ := 2
  let knickknack_count : ℕ := 3
  let knickknack_weight : ℝ := 6
  let total_weight := hardcover_count * hardcover_weight +
                      textbook_count * textbook_weight +
                      knickknack_count * knickknack_weight
  total_weight - bookcase_limit = 33 := by
sorry

end NUMINAMATH_CALUDE_bookcase_weight_excess_l1294_129459


namespace NUMINAMATH_CALUDE_lottery_solution_l1294_129414

def lottery_numbers (A B C D E : ℕ) : Prop :=
  -- Define the five numbers
  let AB := 10 * A + B
  let BC := 10 * B + C
  let CA := 10 * C + A
  let CB := 10 * C + B
  let CD := 10 * C + D
  -- Conditions
  (1 ≤ A) ∧ (A < B) ∧ (B < C) ∧ (C < 9) ∧ (B < D) ∧ (D ≤ 9) ∧
  (AB < BC) ∧ (BC < CA) ∧ (CA < CB) ∧ (CB < CD) ∧
  (AB + BC + CA + CB + CD = 100 * B + 10 * C + C) ∧
  (CA * BC = 1000 * B + 100 * B + 10 * E + C) ∧
  (CA * CD = 1000 * E + 100 * C + 10 * C + D)

theorem lottery_solution :
  ∃! (A B C D E : ℕ), lottery_numbers A B C D E ∧ A = 1 ∧ B = 2 ∧ C = 8 ∧ D = 5 ∧ E = 6 := by
  sorry

end NUMINAMATH_CALUDE_lottery_solution_l1294_129414


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1294_129412

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 + r2

/-- The theorem stating that the two given circles are externally tangent -/
theorem circles_externally_tangent :
  let c1 : ℝ × ℝ := (0, 8)
  let c2 : ℝ × ℝ := (-6, 0)
  let r1 : ℝ := 6
  let r2 : ℝ := 2
  externally_tangent c1 c2 r1 r2 := by
  sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1294_129412


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_a_l1294_129499

def i : ℂ := Complex.I

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_a (a : ℝ) :
  is_pure_imaginary ((2 + a * i) / (2 - i)) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_a_l1294_129499


namespace NUMINAMATH_CALUDE_AE_length_l1294_129486

-- Define the points A, B, C, D, E, and M on a line
variable (A B C D E M : ℝ)

-- Define the conditions
axiom divide_four_equal : B - A = C - B ∧ C - B = D - C ∧ D - C = E - D
axiom M_midpoint : M - A = E - M
axiom MC_length : M - C = 12

-- Theorem to prove
theorem AE_length : E - A = 48 := by
  sorry

end NUMINAMATH_CALUDE_AE_length_l1294_129486


namespace NUMINAMATH_CALUDE_power_equality_l1294_129425

theorem power_equality (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 5) : a^(3*m + 2*n) = 200 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1294_129425


namespace NUMINAMATH_CALUDE_jeans_discount_impossibility_total_price_calculation_l1294_129433

/-- Represents the prices and discount rates for jeans --/
structure JeansSale where
  fox_price : ℝ
  pony_price : ℝ
  fox_quantity : ℕ
  pony_quantity : ℕ
  total_discount_rate : ℝ
  pony_discount_rate : ℝ

/-- Theorem stating the impossibility of the given discount rates --/
theorem jeans_discount_impossibility (sale : JeansSale)
  (h1 : sale.fox_price = 15)
  (h2 : sale.pony_price = 18)
  (h3 : sale.fox_quantity = 3)
  (h4 : sale.pony_quantity = 2)
  (h5 : sale.total_discount_rate = 0.18)
  (h6 : sale.pony_discount_rate = 0.5667) :
  False := by
  sorry

/-- Function to calculate the total regular price --/
def total_regular_price (sale : JeansSale) : ℝ :=
  sale.fox_price * sale.fox_quantity + sale.pony_price * sale.pony_quantity

/-- Theorem stating the total regular price for the given quantities --/
theorem total_price_calculation (sale : JeansSale)
  (h1 : sale.fox_price = 15)
  (h2 : sale.pony_price = 18)
  (h3 : sale.fox_quantity = 3)
  (h4 : sale.pony_quantity = 2) :
  total_regular_price sale = 81 := by
  sorry

end NUMINAMATH_CALUDE_jeans_discount_impossibility_total_price_calculation_l1294_129433


namespace NUMINAMATH_CALUDE_function_value_at_three_l1294_129487

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f(3) = 11 -/
theorem function_value_at_three (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : f 1 = 5)
    (h2 : ∀ x, f x = a * x + b * x + 2) : 
  f 3 = 11 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_three_l1294_129487


namespace NUMINAMATH_CALUDE_order_of_6_undefined_l1294_129449

def f (x : ℤ) : ℤ := x^2 % 13

def f_iter (n : ℕ) (x : ℤ) : ℤ := 
  match n with
  | 0 => x
  | n+1 => f (f_iter n x)

theorem order_of_6_undefined : ¬ ∃ m : ℕ, m > 0 ∧ f_iter m 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_order_of_6_undefined_l1294_129449


namespace NUMINAMATH_CALUDE_ten_people_round_table_arrangements_l1294_129441

/-- The number of unique seating arrangements for n people around a round table,
    considering rotations as identical. -/
def uniqueRoundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem stating that the number of unique seating arrangements for 10 people
    around a round table, considering rotations as identical, is 362,880. -/
theorem ten_people_round_table_arrangements :
  uniqueRoundTableArrangements 10 = 362880 := by sorry

end NUMINAMATH_CALUDE_ten_people_round_table_arrangements_l1294_129441


namespace NUMINAMATH_CALUDE_sector_angle_measure_l1294_129422

/-- Given a sector with radius 2 cm and area 4 cm², 
    the radian measure of its central angle is 2. -/
theorem sector_angle_measure (r : ℝ) (S : ℝ) (θ : ℝ) : 
  r = 2 →  -- radius is 2 cm
  S = 4 →  -- area is 4 cm²
  S = 1/2 * r^2 * θ →  -- formula for sector area
  θ = 2 :=  -- central angle is 2 radians
by sorry

end NUMINAMATH_CALUDE_sector_angle_measure_l1294_129422


namespace NUMINAMATH_CALUDE_min_sum_squares_roots_l1294_129450

theorem min_sum_squares_roots (m : ℝ) (α β : ℝ) : 
  (∀ x : ℝ, x^2 - 2*m*x + 2 - m^2 = 0 ↔ x = α ∨ x = β) → 
  ∃ (k : ℝ), ∀ m : ℝ, α^2 + β^2 ≥ k ∧ ∃ m : ℝ, α^2 + β^2 = k :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_roots_l1294_129450


namespace NUMINAMATH_CALUDE_apple_cost_l1294_129470

/-- The cost of apples under specific pricing conditions -/
theorem apple_cost (l q : ℚ) : 
  (30 * l + 3 * q = 333) →  -- Price for 33 kg
  (30 * l + 6 * q = 366) →  -- Price for 36 kg
  (15 * l = 150)            -- Price for 15 kg
:= by sorry

end NUMINAMATH_CALUDE_apple_cost_l1294_129470


namespace NUMINAMATH_CALUDE_line_circle_intersection_l1294_129489

/-- The circle equation x^2 + y^2 - 4x - 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The line equation ax + y - 5 = 0 -/
def line_equation (a x y : ℝ) : Prop :=
  a*x + y - 5 = 0

/-- The chord length of the intersection is 4 -/
def chord_length_is_4 (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4^2

theorem line_circle_intersection (a : ℝ) :
  chord_length_is_4 a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l1294_129489


namespace NUMINAMATH_CALUDE_mia_wins_two_l1294_129436

/-- Represents a player in the chess tournament -/
inductive Player : Type
  | Sarah : Player
  | Ryan : Player
  | Mia : Player

/-- Represents the number of games won by a player -/
def wins : Player → ℕ
  | Player.Sarah => 5
  | Player.Ryan => 2
  | Player.Mia => 2  -- This is what we want to prove

/-- Represents the number of games lost by a player -/
def losses : Player → ℕ
  | Player.Sarah => 1
  | Player.Ryan => 4
  | Player.Mia => 4

/-- The total number of games played in the tournament -/
def total_games : ℕ := 6

theorem mia_wins_two : wins Player.Mia = 2 := by
  sorry

#check mia_wins_two

end NUMINAMATH_CALUDE_mia_wins_two_l1294_129436


namespace NUMINAMATH_CALUDE_percentage_of_employees_6_years_or_more_l1294_129409

/-- Represents the distribution of employees' duration of service at the Fermat Company -/
structure EmployeeDistribution :=
  (less_than_1_year : ℕ)
  (from_1_to_1_5_years : ℕ)
  (from_1_5_to_2_5_years : ℕ)
  (from_2_5_to_3_5_years : ℕ)
  (from_3_5_to_4_5_years : ℕ)
  (from_4_5_to_5_5_years : ℕ)
  (from_5_5_to_6_5_years : ℕ)
  (from_6_5_to_7_5_years : ℕ)
  (from_7_5_to_8_5_years : ℕ)
  (from_8_5_to_10_years : ℕ)

/-- Calculates the total number of employees -/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1_year + d.from_1_to_1_5_years + d.from_1_5_to_2_5_years +
  d.from_2_5_to_3_5_years + d.from_3_5_to_4_5_years + d.from_4_5_to_5_5_years +
  d.from_5_5_to_6_5_years + d.from_6_5_to_7_5_years + d.from_7_5_to_8_5_years +
  d.from_8_5_to_10_years

/-- Calculates the number of employees who have worked for 6 years or more -/
def employees_6_years_or_more (d : EmployeeDistribution) : ℕ :=
  d.from_5_5_to_6_5_years + d.from_6_5_to_7_5_years + d.from_7_5_to_8_5_years +
  d.from_8_5_to_10_years

/-- The theorem to be proved -/
theorem percentage_of_employees_6_years_or_more
  (d : EmployeeDistribution)
  (h1 : d.less_than_1_year = 4)
  (h2 : d.from_1_to_1_5_years = 6)
  (h3 : d.from_1_5_to_2_5_years = 7)
  (h4 : d.from_2_5_to_3_5_years = 4)
  (h5 : d.from_3_5_to_4_5_years = 3)
  (h6 : d.from_4_5_to_5_5_years = 3)
  (h7 : d.from_5_5_to_6_5_years = 2)
  (h8 : d.from_6_5_to_7_5_years = 1)
  (h9 : d.from_7_5_to_8_5_years = 1)
  (h10 : d.from_8_5_to_10_years = 1) :
  (employees_6_years_or_more d : ℚ) / (total_employees d : ℚ) = 5 / 32 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_employees_6_years_or_more_l1294_129409


namespace NUMINAMATH_CALUDE_papa_worms_correct_l1294_129472

/-- The number of worms Papa bird caught -/
def papa_worms (babies : ℕ) (worms_per_baby_per_day : ℕ) (days : ℕ) 
  (mama_caught : ℕ) (stolen : ℕ) (mama_needs : ℕ) : ℕ :=
  babies * worms_per_baby_per_day * days - ((mama_caught - stolen) + mama_needs)

theorem papa_worms_correct : 
  papa_worms 6 3 3 13 2 34 = 9 := by
  sorry

end NUMINAMATH_CALUDE_papa_worms_correct_l1294_129472


namespace NUMINAMATH_CALUDE_outfit_count_l1294_129490

def red_shirts : ℕ := 7
def green_shirts : ℕ := 7
def pants : ℕ := 8
def green_hats : ℕ := 10
def red_hats : ℕ := 10
def blue_hats : ℕ := 5

def total_outfits : ℕ := red_shirts * pants * (green_hats + blue_hats) + 
                          green_shirts * pants * (red_hats + blue_hats)

theorem outfit_count : total_outfits = 1680 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l1294_129490


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1294_129463

theorem polynomial_factorization (a b c : ℝ) :
  (a - 2*b) * (a - 2*b - 4) + 4 - c^2 = ((a - 2*b) - 2 + c) * ((a - 2*b) - 2 - c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1294_129463


namespace NUMINAMATH_CALUDE_definite_integral_x_x_squared_sin_x_l1294_129419

theorem definite_integral_x_x_squared_sin_x : 
  ∫ x in (-1)..1, (x + x^2 + Real.sin x) = 2/3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_x_x_squared_sin_x_l1294_129419


namespace NUMINAMATH_CALUDE_jenny_easter_eggs_problem_l1294_129497

theorem jenny_easter_eggs_problem :
  let total_red : ℕ := 30
  let total_blue : ℕ := 42
  let min_eggs_per_basket : ℕ := 5
  ∃ (eggs_per_basket : ℕ),
    eggs_per_basket ≥ min_eggs_per_basket ∧
    eggs_per_basket ∣ total_red ∧
    eggs_per_basket ∣ total_blue ∧
    ∀ (n : ℕ), n > eggs_per_basket →
      ¬(n ∣ total_red ∧ n ∣ total_blue) →
    eggs_per_basket = 6 :=
by sorry

end NUMINAMATH_CALUDE_jenny_easter_eggs_problem_l1294_129497


namespace NUMINAMATH_CALUDE_managers_salary_l1294_129474

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 600 →
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - num_employees * avg_salary) = 14100 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l1294_129474


namespace NUMINAMATH_CALUDE_not_parallel_implies_m_eq_one_perpendicular_implies_m_eq_neg_five_thirds_l1294_129465

/-- Two lines l₁ and l₂ in the plane -/
structure TwoLines (m : ℝ) where
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → Prop
  l₁_eq : ∀ x y, l₁ x y ↔ (3 + m) * x + 4 * y = 5 - 3 * m
  l₂_eq : ∀ x y, l₂ x y ↔ 2 * x + (m + 1) * y = -20

/-- Condition for two lines to be not parallel -/
def NotParallel (m : ℝ) (lines : TwoLines m) : Prop :=
  (3 + m) * (1 + m) - 4 * 2 ≠ 0

/-- Condition for two lines to be perpendicular -/
def Perpendicular (m : ℝ) (lines : TwoLines m) : Prop :=
  2 * (3 + m) + 4 * (1 + m) = 0

/-- Theorem: If the lines are not parallel, then m = 1 -/
theorem not_parallel_implies_m_eq_one (m : ℝ) (lines : TwoLines m) :
  NotParallel m lines → m = 1 := by sorry

/-- Theorem: If the lines are perpendicular, then m = -5/3 -/
theorem perpendicular_implies_m_eq_neg_five_thirds (m : ℝ) (lines : TwoLines m) :
  Perpendicular m lines → m = -5/3 := by sorry

end NUMINAMATH_CALUDE_not_parallel_implies_m_eq_one_perpendicular_implies_m_eq_neg_five_thirds_l1294_129465


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1294_129448

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0
  asymptote_eq : ∀ (x y : ℝ), x = 2 * y ∨ x = -2 * y
  point_on_curve : (4 : ℝ)^2 / a^2 - 1^2 / b^2 = 1

/-- The specific equation of the hyperbola -/
def specific_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 12 - y^2 / 3 = 1

/-- Theorem stating that the specific equation holds for the given hyperbola -/
theorem hyperbola_equation (h : Hyperbola) :
  ∀ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1 ↔ specific_equation h x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1294_129448


namespace NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l1294_129428

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A four-digit number formed from the given digits -/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  distinct : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

theorem sum_of_four_digit_numbers :
  (allFourDigitNumbers.sum FourDigitNumber.value) = 399960 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l1294_129428


namespace NUMINAMATH_CALUDE_quadratic_solution_existence_l1294_129417

theorem quadratic_solution_existence (a b c : ℝ) (f : ℝ → ℝ) 
  (hf : f = fun x ↦ a * x^2 + b * x + c)
  (h1 : f 3.11 < 0)
  (h2 : f 3.12 > 0) :
  ∃ x : ℝ, f x = 0 ∧ 3.11 < x ∧ x < 3.12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_existence_l1294_129417


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1294_129488

/-- Given two arithmetic sequences, prove the ratio of their 4th terms -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) : 
  (∀ n, S n / T n = (7 * n + 2) / (n + 3)) →  -- Given condition
  (∀ n, S n = (a 1 + a n) * n / 2) →  -- Definition of S_n for arithmetic sequence
  (∀ n, T n = (b 1 + b n) * n / 2) →  -- Definition of T_n for arithmetic sequence
  a 4 / b 4 = 51 / 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1294_129488
