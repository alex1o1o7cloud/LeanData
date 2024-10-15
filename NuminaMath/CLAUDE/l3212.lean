import Mathlib

namespace NUMINAMATH_CALUDE_flower_count_proof_l3212_321253

/-- The number of daisy seeds planted -/
def daisy_seeds : ℕ := 25

/-- The number of sunflower seeds planted -/
def sunflower_seeds : ℕ := 25

/-- The percentage of daisy seeds that germinate -/
def daisy_germination_rate : ℚ := 60 / 100

/-- The percentage of sunflower seeds that germinate -/
def sunflower_germination_rate : ℚ := 80 / 100

/-- The percentage of germinated plants that produce flowers -/
def flower_production_rate : ℚ := 80 / 100

/-- The total number of plants that produce flowers -/
def plants_with_flowers : ℕ := 28

theorem flower_count_proof :
  (daisy_seeds * daisy_germination_rate * flower_production_rate +
   sunflower_seeds * sunflower_germination_rate * flower_production_rate).floor = plants_with_flowers := by
  sorry

end NUMINAMATH_CALUDE_flower_count_proof_l3212_321253


namespace NUMINAMATH_CALUDE_max_sides_convex_polygon_is_maximum_max_sides_convex_polygon_is_convex_l3212_321296

/-- The maximum number of sides for a convex polygon with interior angles
    forming an arithmetic sequence with a common difference of 1°. -/
def max_sides_convex_polygon : ℕ := 27

/-- The common difference of the arithmetic sequence formed by the interior angles. -/
def common_difference : ℝ := 1

/-- Predicate to check if a polygon is convex based on its number of sides. -/
def is_convex (n : ℕ) : Prop :=
  let α : ℝ := (n - 2) * 180 / n - (n - 1) / 2
  α > 0 ∧ α + (n - 1) * common_difference < 180

/-- Theorem stating that max_sides_convex_polygon is the maximum number of sides
    for a convex polygon with interior angles forming an arithmetic sequence
    with a common difference of 1°. -/
theorem max_sides_convex_polygon_is_maximum :
  ∀ n : ℕ, n > max_sides_convex_polygon → ¬(is_convex n) :=
sorry

/-- Theorem stating that max_sides_convex_polygon satisfies the convexity condition. -/
theorem max_sides_convex_polygon_is_convex :
  is_convex max_sides_convex_polygon :=
sorry

end NUMINAMATH_CALUDE_max_sides_convex_polygon_is_maximum_max_sides_convex_polygon_is_convex_l3212_321296


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l3212_321297

theorem least_three_digit_multiple_of_nine : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 9 ∣ n → n ≥ 108 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l3212_321297


namespace NUMINAMATH_CALUDE_triangle_side_sum_l3212_321243

theorem triangle_side_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 60) (h3 : b = 30) (h4 : c = 90) 
  (side_opposite_30 : ℝ) (h5 : side_opposite_30 = 8 * Real.sqrt 3) :
  ∃ (other_sides_sum : ℝ), other_sides_sum = 12 + 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l3212_321243


namespace NUMINAMATH_CALUDE_martin_initial_hens_correct_martin_initial_hens_unique_l3212_321274

/-- Represents the farm's egg production scenario -/
structure FarmScenario where
  initial_hens : ℕ
  initial_eggs : ℕ
  initial_days : ℕ
  added_hens : ℕ
  final_eggs : ℕ
  final_days : ℕ

/-- The specific scenario from the problem -/
def martin_farm : FarmScenario :=
  { initial_hens := 25,  -- This is what we want to prove
    initial_eggs := 80,
    initial_days := 10,
    added_hens := 15,
    final_eggs := 300,
    final_days := 15 }

/-- Theorem stating that Martin's initial number of hens is correct -/
theorem martin_initial_hens_correct :
  martin_farm.initial_hens * martin_farm.final_days * martin_farm.initial_eggs =
  martin_farm.initial_days * martin_farm.final_eggs * martin_farm.initial_hens +
  martin_farm.initial_days * martin_farm.final_eggs * martin_farm.added_hens :=
by sorry

/-- Theorem proving that 25 is the only solution -/
theorem martin_initial_hens_unique (h : ℕ) :
  h * martin_farm.final_days * martin_farm.initial_eggs =
  martin_farm.initial_days * martin_farm.final_eggs * h +
  martin_farm.initial_days * martin_farm.final_eggs * martin_farm.added_hens →
  h = martin_farm.initial_hens :=
by sorry

end NUMINAMATH_CALUDE_martin_initial_hens_correct_martin_initial_hens_unique_l3212_321274


namespace NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l3212_321217

theorem gcd_of_lcm_and_ratio (X Y : ℕ) : 
  X ≠ 0 → Y ≠ 0 → 
  lcm X Y = 180 → 
  ∃ (k : ℕ), X = 2 * k ∧ Y = 5 * k → 
  gcd X Y = 18 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_lcm_and_ratio_l3212_321217


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3212_321259

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1
  roots_equation : a 1 ^ 2 - 10 * a 1 + 16 = 0 ∧ a 99 ^ 2 - 10 * a 99 + 16 = 0

/-- The main theorem -/
theorem geometric_sequence_product (seq : GeometricSequence) :
  seq.a 20 * seq.a 50 * seq.a 80 = 64 ∨ seq.a 20 * seq.a 50 * seq.a 80 = -64 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l3212_321259


namespace NUMINAMATH_CALUDE_store_discount_percentage_l3212_321271

theorem store_discount_percentage (C : ℝ) (C_pos : C > 0) : 
  let initial_price := 1.20 * C
  let new_year_price := 1.25 * initial_price
  let february_price := 1.20 * C
  let discount := new_year_price - february_price
  discount / new_year_price = 0.20 := by
sorry

end NUMINAMATH_CALUDE_store_discount_percentage_l3212_321271


namespace NUMINAMATH_CALUDE_h_eq_f_reflected_and_shifted_l3212_321291

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define the function h obtained from f by reflection and shift
def h (x : ℝ) : ℝ := f (6 - x)

-- Theorem stating the relationship between h and f
theorem h_eq_f_reflected_and_shifted :
  ∀ x : ℝ, h f x = f (6 - x) := by sorry

end NUMINAMATH_CALUDE_h_eq_f_reflected_and_shifted_l3212_321291


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l3212_321235

/-- A "T" shaped figure composed of unit squares -/
structure TShape :=
  (top_row : Fin 3 → Unit)
  (bottom_column : Fin 2 → Unit)

/-- The perimeter of a TShape -/
def perimeter (t : TShape) : ℕ :=
  14

theorem t_shape_perimeter :
  ∀ (t : TShape), perimeter t = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l3212_321235


namespace NUMINAMATH_CALUDE_chips_cost_split_l3212_321264

theorem chips_cost_split (num_friends : ℕ) (num_bags : ℕ) (cost_per_bag : ℕ) :
  num_friends = 3 →
  num_bags = 5 →
  cost_per_bag = 3 →
  (num_bags * cost_per_bag) / num_friends = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_chips_cost_split_l3212_321264


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3212_321284

theorem complex_number_in_first_quadrant :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3212_321284


namespace NUMINAMATH_CALUDE_machinery_expenditure_l3212_321210

theorem machinery_expenditure (total : ℝ) (raw_materials : ℝ) (cash_percentage : ℝ) :
  total = 137500 →
  raw_materials = 80000 →
  cash_percentage = 0.20 →
  ∃ machinery : ℝ,
    machinery = 30000 ∧
    raw_materials + machinery + (cash_percentage * total) = total :=
by sorry

end NUMINAMATH_CALUDE_machinery_expenditure_l3212_321210


namespace NUMINAMATH_CALUDE_star_operation_associative_l3212_321289

-- Define the curve y = x^3
def cubic_curve (x : ℝ) : ℝ := x^3

-- Define a point on the curve
structure CurvePoint where
  x : ℝ
  y : ℝ
  on_curve : y = cubic_curve x

-- Define the * operation
def star_operation (A B : CurvePoint) : CurvePoint :=
  sorry

-- Theorem statement
theorem star_operation_associative :
  ∀ (A B C : CurvePoint),
    star_operation (star_operation A B) C = star_operation A (star_operation B C) := by
  sorry

end NUMINAMATH_CALUDE_star_operation_associative_l3212_321289


namespace NUMINAMATH_CALUDE_sandy_average_book_price_l3212_321221

/-- The average price Sandy paid per book given her purchases from two shops -/
theorem sandy_average_book_price (books1 : ℕ) (price1 : ℚ) (books2 : ℕ) (price2 : ℚ) 
  (h1 : books1 = 65)
  (h2 : price1 = 1380)
  (h3 : books2 = 55)
  (h4 : price2 = 900) :
  (price1 + price2) / (books1 + books2 : ℚ) = 19 := by
  sorry

end NUMINAMATH_CALUDE_sandy_average_book_price_l3212_321221


namespace NUMINAMATH_CALUDE_book_length_is_300_l3212_321275

/-- The length of a book in pages -/
def book_length : ℕ := 300

/-- The fraction of the book Soja has finished reading -/
def finished_fraction : ℚ := 2/3

/-- The difference between pages read and pages left to read -/
def pages_difference : ℕ := 100

/-- Theorem stating that the book length is 300 pages -/
theorem book_length_is_300 : 
  book_length = 300 ∧ 
  finished_fraction * book_length - (1 - finished_fraction) * book_length = pages_difference := by
  sorry

end NUMINAMATH_CALUDE_book_length_is_300_l3212_321275


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3212_321202

theorem inequality_solution_set (x : ℝ) : (3 * x + 1) / (1 - 2 * x) ≥ 0 ↔ -1/3 ≤ x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3212_321202


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l3212_321219

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- The function f(x) = x^4 + (m - 1)x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^4 + (m - 1) * x + 1

/-- If f(x) = x^4 + (m - 1)x + 1 is an even function, then m = 1 -/
theorem even_function_implies_m_equals_one (m : ℝ) : IsEven (f m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l3212_321219


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l3212_321220

/-- Represents a student with their skills -/
structure Student where
  hasExcellentEnglish : Bool
  hasStrongComputer : Bool

/-- The total number of students -/
def totalStudents : Nat := 8

/-- The number of students with excellent English scores -/
def excellentEnglishCount : Nat := 2

/-- The number of students with strong computer skills -/
def strongComputerCount : Nat := 3

/-- The number of students to be allocated to each company -/
def studentsPerCompany : Nat := 4

/-- Calculates the number of valid allocation schemes -/
def countAllocationSchemes (students : List Student) : Nat :=
  sorry

/-- Theorem stating the number of valid allocation schemes -/
theorem allocation_schemes_count :
  ∀ (students : List Student),
    students.length = totalStudents →
    (students.filter (·.hasExcellentEnglish)).length = excellentEnglishCount →
    (students.filter (·.hasStrongComputer)).length = strongComputerCount →
    countAllocationSchemes students = 36 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l3212_321220


namespace NUMINAMATH_CALUDE_work_efficiency_ratio_l3212_321273

theorem work_efficiency_ratio (a b : ℝ) : 
  a + b = 1 / 26 → b = 1 / 39 → a / b = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_work_efficiency_ratio_l3212_321273


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_less_than_700_l3212_321226

theorem greatest_multiple_of_5_and_7_less_than_700 :
  (∃ n : ℕ, n * 5 * 7 < 700 ∧ 
    ∀ m : ℕ, m * 5 * 7 < 700 → m * 5 * 7 ≤ n * 5 * 7) →
  (∃ n : ℕ, n * 5 * 7 = 695) :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_less_than_700_l3212_321226


namespace NUMINAMATH_CALUDE_linear_system_fraction_sum_l3212_321267

theorem linear_system_fraction_sum (a b c x y z : ℝ) 
  (eq1 : 11 * x + b * y + c * z = 0)
  (eq2 : a * x + 24 * y + c * z = 0)
  (eq3 : a * x + b * y + 41 * z = 0)
  (ha : a ≠ 11)
  (hx : x ≠ 0) :
  a / (a - 11) + b / (b - 24) + c / (c - 41) = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_fraction_sum_l3212_321267


namespace NUMINAMATH_CALUDE_f_composition_value_l3212_321212

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) else Real.tan x

theorem f_composition_value : f (f (3 * Real.pi / 4)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3212_321212


namespace NUMINAMATH_CALUDE_meaningful_expression_l3212_321256

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 2 / Real.sqrt (x - 1)) ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3212_321256


namespace NUMINAMATH_CALUDE_parallel_vectors_angle_l3212_321246

theorem parallel_vectors_angle (α : Real) : 
  α > 0 → 
  α < π / 2 → 
  let a : Fin 2 → Real := ![3/4, Real.sin α]
  let b : Fin 2 → Real := ![Real.cos α, 1/3]
  (∃ (k : Real), k ≠ 0 ∧ a = k • b) → 
  α = π / 12 ∨ α = 5 * π / 12 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_angle_l3212_321246


namespace NUMINAMATH_CALUDE_man_in_dark_probability_l3212_321213

/-- The number of revolutions per minute made by the searchlight -/
def revolutions_per_minute : ℝ := 3

/-- The probability that a man will stay in the dark for at least some seconds -/
def probability_in_dark : ℝ := 0.25

/-- Theorem stating the probability of a man staying in the dark -/
theorem man_in_dark_probability :
  probability_in_dark = 0.25 := by sorry

end NUMINAMATH_CALUDE_man_in_dark_probability_l3212_321213


namespace NUMINAMATH_CALUDE_photo_selection_choices_l3212_321232

-- Define the number of items to choose from
def n : ℕ := 10

-- Define the possible numbers of items to be chosen
def k₁ : ℕ := 5
def k₂ : ℕ := 6

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem photo_selection_choices : 
  combination n k₁ + combination n k₂ = 462 := by sorry

end NUMINAMATH_CALUDE_photo_selection_choices_l3212_321232


namespace NUMINAMATH_CALUDE_distance_sum_equals_radii_sum_l3212_321231

/-- An acute-angled triangle with its circumscribed and inscribed circles -/
structure AcuteTriangle where
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The distance from the circumcenter to side a -/
  da : ℝ
  /-- The distance from the circumcenter to side b -/
  db : ℝ
  /-- The distance from the circumcenter to side c -/
  dc : ℝ
  /-- The triangle is acute-angled -/
  acute : R > 0
  /-- The radii and distances are positive -/
  positive : r > 0 ∧ da > 0 ∧ db > 0 ∧ dc > 0

/-- The sum of distances from the circumcenter to the sides equals the sum of circumradius and inradius -/
theorem distance_sum_equals_radii_sum (t : AcuteTriangle) : t.da + t.db + t.dc = t.R + t.r := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_equals_radii_sum_l3212_321231


namespace NUMINAMATH_CALUDE_arithmetic_sum_specific_l3212_321207

/-- Sum of arithmetic sequence with given parameters -/
def arithmetic_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic sequence with first term -45, last term 0, and common difference 3 is -360 -/
theorem arithmetic_sum_specific : arithmetic_sum (-45) 0 3 = -360 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_specific_l3212_321207


namespace NUMINAMATH_CALUDE_octagon_cannot_tile_l3212_321223

/-- A regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The interior angle of a regular polygon with n sides --/
def interiorAngle (n : ℕ) (p : RegularPolygon n) : ℚ :=
  180 - (360 / n)

/-- A regular polygon can tile the plane if its interior angle divides 360° evenly --/
def canTilePlane (n : ℕ) (p : RegularPolygon n) : Prop :=
  ∃ k : ℕ, k * interiorAngle n p = 360

/-- The set of regular polygons we're considering --/
def consideredPolygons : Set (Σ n, RegularPolygon n) :=
  {⟨3, ⟨by norm_num⟩⟩, ⟨4, ⟨by norm_num⟩⟩, ⟨6, ⟨by norm_num⟩⟩, ⟨8, ⟨by norm_num⟩⟩}

theorem octagon_cannot_tile :
  ∀ p ∈ consideredPolygons, ¬(canTilePlane p.1 p.2) ↔ p.1 = 8 := by
  sorry

#check octagon_cannot_tile

end NUMINAMATH_CALUDE_octagon_cannot_tile_l3212_321223


namespace NUMINAMATH_CALUDE_base16_to_base4_C2A_l3212_321237

/-- Represents a digit in base 16 --/
inductive Base16Digit
| C | Two | A

/-- Represents a number in base 16 --/
def Base16Number := List Base16Digit

/-- Represents a digit in base 4 --/
inductive Base4Digit
| Zero | One | Two | Three

/-- Represents a number in base 4 --/
def Base4Number := List Base4Digit

/-- Converts a Base16Number to a Base4Number --/
def convertBase16ToBase4 (n : Base16Number) : Base4Number := sorry

/-- The main theorem --/
theorem base16_to_base4_C2A :
  convertBase16ToBase4 [Base16Digit.C, Base16Digit.Two, Base16Digit.A] =
  [Base4Digit.Three, Base4Digit.Zero, Base4Digit.Zero,
   Base4Digit.Two, Base4Digit.Two, Base4Digit.Two] :=
by sorry

end NUMINAMATH_CALUDE_base16_to_base4_C2A_l3212_321237


namespace NUMINAMATH_CALUDE_river_current_speed_l3212_321234

/-- The speed of a river's current given a swimmer's performance -/
theorem river_current_speed 
  (swimmer_still_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : swimmer_still_speed = 3)
  (h2 : distance = 8)
  (h3 : time = 5) :
  swimmer_still_speed - (distance / time) = (1.4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_river_current_speed_l3212_321234


namespace NUMINAMATH_CALUDE_jogger_train_distance_l3212_321240

/-- Calculates the distance between a jogger and a train engine given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_train_distance
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (passing_time : ℝ)
  (h1 : jogger_speed = 9 * 1000 / 3600)  -- 9 km/hr in m/s
  (h2 : train_speed = 45 * 1000 / 3600)  -- 45 km/hr in m/s
  (h3 : train_length = 120)              -- 120 meters
  (h4 : passing_time = 31)               -- 31 seconds
  : ∃ (distance : ℝ), distance = 190 ∧ distance = (train_speed - jogger_speed) * passing_time - train_length :=
by
  sorry


end NUMINAMATH_CALUDE_jogger_train_distance_l3212_321240


namespace NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l3212_321241

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720 degrees -/
theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l3212_321241


namespace NUMINAMATH_CALUDE_two_segment_trip_avg_speed_l3212_321222

/-- Calculates the average speed for a two-segment trip -/
theorem two_segment_trip_avg_speed 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 40) 
  (h2 : speed1 = 30) 
  (h3 : distance2 = 40) 
  (h4 : speed2 = 15) : 
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 20 := by
  sorry

#check two_segment_trip_avg_speed

end NUMINAMATH_CALUDE_two_segment_trip_avg_speed_l3212_321222


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l3212_321298

theorem least_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 56 = 3 ∧
  n % 78 = 3 ∧
  n % 9 = 0 ∧
  ∀ m : ℕ, m > 0 ∧ m % 56 = 3 ∧ m % 78 = 3 ∧ m % 9 = 0 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l3212_321298


namespace NUMINAMATH_CALUDE_cupcake_price_l3212_321254

theorem cupcake_price (cupcake_count : ℕ) (cookie_count : ℕ) (cookie_price : ℚ)
  (basketball_count : ℕ) (basketball_price : ℚ) (drink_count : ℕ) (drink_price : ℚ) :
  cupcake_count = 50 →
  cookie_count = 40 →
  cookie_price = 1/2 →
  basketball_count = 2 →
  basketball_price = 40 →
  drink_count = 20 →
  drink_price = 2 →
  ∃ (cupcake_price : ℚ),
    cupcake_count * cupcake_price + cookie_count * cookie_price =
    basketball_count * basketball_price + drink_count * drink_price ∧
    cupcake_price = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_cupcake_price_l3212_321254


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3212_321279

/-- Given a geometric series with first term a and common ratio r,
    if the sum of the series is 20 and the sum of terms involving odd powers of r is 8,
    then r = 1/4 -/
theorem geometric_series_ratio (a r : ℝ) 
  (h1 : a / (1 - r) = 20)
  (h2 : a * r / (1 - r^2) = 8) :
  r = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3212_321279


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3212_321216

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (m = f' 1) ∧
    (f 1 = m * 1 + b) ∧
    (m * 1 - f 1 + b = 0) ∧
    (m = 2 ∧ b = -1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3212_321216


namespace NUMINAMATH_CALUDE_least_positive_integer_with_congruences_l3212_321276

theorem least_positive_integer_with_congruences : ∃ (b : ℕ), 
  b > 0 ∧ 
  b % 3 = 2 ∧ 
  b % 5 = 4 ∧ 
  b % 6 = 5 ∧ 
  b % 7 = 6 ∧ 
  (∀ (x : ℕ), x > 0 ∧ x % 3 = 2 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ x % 7 = 6 → x ≥ b) ∧
  b = 209 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_congruences_l3212_321276


namespace NUMINAMATH_CALUDE_acute_angle_probability_l3212_321249

/-- Represents a clock with hour and minute hands. -/
structure Clock :=
  (hour_hand : ℝ)
  (minute_hand : ℝ)

/-- The angle between the hour and minute hands is acute. -/
def is_acute_angle (c : Clock) : Prop :=
  let angle := (c.minute_hand - c.hour_hand + 12) % 12
  angle < 3 ∨ angle > 9

/-- A random clock stop event. -/
def random_clock_stop : Clock → Prop :=
  sorry

/-- The probability of an event occurring. -/
def probability (event : Clock → Prop) : ℝ :=
  sorry

/-- The main theorem: The probability of an acute angle between clock hands is 1/2. -/
theorem acute_angle_probability :
  probability is_acute_angle = 1/2 :=
sorry

end NUMINAMATH_CALUDE_acute_angle_probability_l3212_321249


namespace NUMINAMATH_CALUDE_complement_of_M_union_N_in_U_l3212_321239

open Set

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2}

-- Define set N
def N : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_of_M_union_N_in_U :
  (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_union_N_in_U_l3212_321239


namespace NUMINAMATH_CALUDE_circle_intersection_problem_l3212_321294

/-- Given two circles C₁ and C₂ with equations as defined below, prove that the value of a is 4 -/
theorem circle_intersection_problem (a b x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 + y₁^2 - 2*x₁ + 4*y₁ - b^2 + 5 = 0) →  -- C₁ equation for point A
  (x₂^2 + y₂^2 - 2*x₂ + 4*y₂ - b^2 + 5 = 0) →  -- C₁ equation for point B
  (x₁^2 + y₁^2 - 2*(a-6)*x₁ - 2*a*y₁ + 2*a^2 - 12*a + 27 = 0) →  -- C₂ equation for point A
  (x₂^2 + y₂^2 - 2*(a-6)*x₂ - 2*a*y₂ + 2*a^2 - 12*a + 27 = 0) →  -- C₂ equation for point B
  ((y₁ + y₂)/(x₁ + x₂) + (x₁ - x₂)/(y₁ - y₂) = 0) →  -- Given condition
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →  -- Distinct points condition
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_problem_l3212_321294


namespace NUMINAMATH_CALUDE_pentagon_area_l3212_321208

/-- The area of a specific pentagon -/
theorem pentagon_area : 
  ∀ (s₁ s₂ s₃ s₄ s₅ : ℝ) (θ : ℝ),
  s₁ = 18 → s₂ = 20 → s₃ = 27 → s₄ = 24 → s₅ = 20 →
  θ = Real.pi / 2 →
  ∃ (A : ℝ),
  A = (1/2 * s₁ * s₂) + (1/2 * (s₃ + s₄) * s₅) ∧
  A = 690 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_area_l3212_321208


namespace NUMINAMATH_CALUDE_min_value_expression_l3212_321292

theorem min_value_expression (a b : ℤ) (h : a > b) :
  (2 : ℝ) ≤ ((2*a + 3*b : ℝ) / (a - 2*b : ℝ)) + ((a - 2*b : ℝ) / (2*a + 3*b : ℝ)) ∧
  ∃ (a' b' : ℤ), a' > b' ∧ ((2*a' + 3*b' : ℝ) / (a' - 2*b' : ℝ)) + ((a' - 2*b' : ℝ) / (2*a' + 3*b' : ℝ)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3212_321292


namespace NUMINAMATH_CALUDE_skateboard_cost_l3212_321285

theorem skateboard_cost (toy_cars_cost toy_trucks_cost total_toys_cost : ℚ)
  (h1 : toy_cars_cost = 14.88)
  (h2 : toy_trucks_cost = 5.86)
  (h3 : total_toys_cost = 25.62) :
  total_toys_cost - (toy_cars_cost + toy_trucks_cost) = 4.88 := by
sorry

end NUMINAMATH_CALUDE_skateboard_cost_l3212_321285


namespace NUMINAMATH_CALUDE_water_left_proof_l3212_321205

-- Define the initial amount of water
def initial_water : ℚ := 7/2

-- Define the amount of water used
def water_used : ℚ := 9/4

-- Theorem statement
theorem water_left_proof : initial_water - water_used = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_water_left_proof_l3212_321205


namespace NUMINAMATH_CALUDE_cubic_quadratic_equation_solution_l3212_321281

theorem cubic_quadratic_equation_solution :
  ∃! (y : ℝ), y ≠ 0 ∧ (8 * y)^3 = (16 * y)^2 ∧ y = 1/2 := by sorry

end NUMINAMATH_CALUDE_cubic_quadratic_equation_solution_l3212_321281


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3212_321229

theorem quadratic_roots_product (p q P Q : ℝ) (α β γ δ : ℝ) 
  (h1 : α^2 + p*α + q = 0)
  (h2 : β^2 + p*β + q = 0)
  (h3 : γ^2 + P*γ + Q = 0)
  (h4 : δ^2 + P*δ + Q = 0) :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = P^2 * q - P * p * Q + Q^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3212_321229


namespace NUMINAMATH_CALUDE_expression_simplification_l3212_321250

theorem expression_simplification (x y : ℚ) 
  (hx : x = -1/3) (hy : y = -2) : 
  2 * (x^2 - 2*x^2*y) - (3*(x^2 - x*y^2) - (x^2*y - 2*x*y^2 + x^2)) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3212_321250


namespace NUMINAMATH_CALUDE_three_four_five_pythagorean_one_two_five_not_pythagorean_two_three_four_not_pythagorean_four_five_six_not_pythagorean_only_three_four_five_pythagorean_l3212_321251

/-- A function that checks if three numbers form a Pythagorean triple --/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- Theorem stating that (3, 4, 5) is a Pythagorean triple --/
theorem three_four_five_pythagorean : isPythagoreanTriple 3 4 5 := by
  sorry

/-- Theorem stating that (1, 2, 5) is not a Pythagorean triple --/
theorem one_two_five_not_pythagorean : ¬ isPythagoreanTriple 1 2 5 := by
  sorry

/-- Theorem stating that (2, 3, 4) is not a Pythagorean triple --/
theorem two_three_four_not_pythagorean : ¬ isPythagoreanTriple 2 3 4 := by
  sorry

/-- Theorem stating that (4, 5, 6) is not a Pythagorean triple --/
theorem four_five_six_not_pythagorean : ¬ isPythagoreanTriple 4 5 6 := by
  sorry

/-- Main theorem stating that among the given sets, only (3, 4, 5) is a Pythagorean triple --/
theorem only_three_four_five_pythagorean :
  (isPythagoreanTriple 3 4 5) ∧
  (¬ isPythagoreanTriple 1 2 5) ∧
  (¬ isPythagoreanTriple 2 3 4) ∧
  (¬ isPythagoreanTriple 4 5 6) := by
  sorry

end NUMINAMATH_CALUDE_three_four_five_pythagorean_one_two_five_not_pythagorean_two_three_four_not_pythagorean_four_five_six_not_pythagorean_only_three_four_five_pythagorean_l3212_321251


namespace NUMINAMATH_CALUDE_lucas_avocados_l3212_321242

/-- Calculates the number of avocados bought given initial money, cost per avocado, and change --/
def avocados_bought (initial_money change cost_per_avocado : ℚ) : ℚ :=
  (initial_money - change) / cost_per_avocado

/-- Proves that Lucas bought 3 avocados --/
theorem lucas_avocados :
  let initial_money : ℚ := 20
  let change : ℚ := 14
  let cost_per_avocado : ℚ := 2
  avocados_bought initial_money change cost_per_avocado = 3 := by
  sorry

end NUMINAMATH_CALUDE_lucas_avocados_l3212_321242


namespace NUMINAMATH_CALUDE_monomial_sum_l3212_321258

theorem monomial_sum (m n : ℤ) (a b : ℝ) : 
  (∀ a b : ℝ, -2 * a^2 * b^(m+1) + n * a^2 * b^4 = 0) → m + n = 5 := by
sorry

end NUMINAMATH_CALUDE_monomial_sum_l3212_321258


namespace NUMINAMATH_CALUDE_cosine_value_proof_l3212_321260

theorem cosine_value_proof (α : Real) 
    (h : Real.sin (α - π/3) = 1/3) : 
    Real.cos (π/6 + α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_proof_l3212_321260


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_no_solutions_for_2891_l3212_321252

def cubic_equation (x y n : ℤ) : Prop :=
  x^3 - 3*x*y^2 + y^3 = n

theorem cubic_equation_solutions (n : ℤ) (hn : n > 0) :
  (∃ x y : ℤ, cubic_equation x y n) →
  (∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℤ, 
    cubic_equation x₁ y₁ n ∧ 
    cubic_equation x₂ y₂ n ∧ 
    cubic_equation x₃ y₃ n ∧ 
    (x₁, y₁) ≠ (x₂, y₂) ∧ 
    (x₁, y₁) ≠ (x₃, y₃) ∧ 
    (x₂, y₂) ≠ (x₃, y₃)) :=
sorry

theorem no_solutions_for_2891 :
  ¬ ∃ x y : ℤ, cubic_equation x y 2891 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_no_solutions_for_2891_l3212_321252


namespace NUMINAMATH_CALUDE_diamond_weight_calculation_l3212_321272

/-- The weight of a single diamond in grams -/
def diamond_weight : ℝ := sorry

/-- The weight of a single jade in grams -/
def jade_weight : ℝ := sorry

/-- The total weight of 5 diamonds in grams -/
def five_diamonds_weight : ℝ := 5 * diamond_weight

theorem diamond_weight_calculation :
  (4 * diamond_weight + 2 * jade_weight = 140) →
  (jade_weight = diamond_weight + 10) →
  five_diamonds_weight = 100 := by sorry

end NUMINAMATH_CALUDE_diamond_weight_calculation_l3212_321272


namespace NUMINAMATH_CALUDE_solution_set_f_less_g_range_of_a_l3212_321247

-- Define the functions f and g
def f (x : ℝ) := abs (x - 4)
def g (x : ℝ) := abs (2 * x + 1)

-- Statement 1
theorem solution_set_f_less_g :
  {x : ℝ | f x < g x} = {x : ℝ | x < -5 ∨ x > 1} := by sorry

-- Statement 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * f x + g x > a * x) ↔ a ∈ Set.Icc (-4) (9/4) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_g_range_of_a_l3212_321247


namespace NUMINAMATH_CALUDE_faster_speed_calculation_l3212_321269

theorem faster_speed_calculation (total_distance : ℝ) (initial_speed : ℝ) (initial_time : ℝ) 
  (second_time : ℝ) (remaining_distance : ℝ) 
  (h1 : total_distance = 600)
  (h2 : initial_speed = 50)
  (h3 : initial_time = 3)
  (h4 : second_time = 4)
  (h5 : remaining_distance = 130) : 
  ∃ faster_speed : ℝ, faster_speed = 80 ∧ 
  total_distance = initial_speed * initial_time + faster_speed * second_time + remaining_distance :=
by
  sorry


end NUMINAMATH_CALUDE_faster_speed_calculation_l3212_321269


namespace NUMINAMATH_CALUDE_songs_per_album_l3212_321225

theorem songs_per_album (total_albums : ℕ) (total_songs : ℕ) 
  (h1 : total_albums = 3 + 5) 
  (h2 : total_songs = 24) 
  (h3 : ∀ (x : ℕ), x * total_albums = total_songs → x = 3) :
  ∃ (songs_per_album : ℕ), songs_per_album * total_albums = total_songs ∧ songs_per_album = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_songs_per_album_l3212_321225


namespace NUMINAMATH_CALUDE_number_calculation_l3212_321233

theorem number_calculation (x : ℚ) : (30 / 100 * x = 25 / 100 * 50) → x = 125 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l3212_321233


namespace NUMINAMATH_CALUDE_acetone_nine_moles_weight_l3212_321248

/-- The molecular weight of a single molecule of Acetone in g/mol -/
def acetone_molecular_weight : ℝ :=
  3 * 12.01 + 6 * 1.008 + 1 * 16.00

/-- The molecular weight of n moles of Acetone in grams -/
def acetone_weight (n : ℝ) : ℝ :=
  n * acetone_molecular_weight

/-- Theorem: The molecular weight of 9 moles of Acetone is 522.702 grams -/
theorem acetone_nine_moles_weight :
  acetone_weight 9 = 522.702 := by
  sorry

end NUMINAMATH_CALUDE_acetone_nine_moles_weight_l3212_321248


namespace NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l3212_321290

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 30th term of the specified arithmetic sequence is 264. -/
theorem arithmetic_sequence_30th_term :
  ∀ a : ℕ → ℝ, is_arithmetic_sequence a →
  a 1 = 3 → a 2 = 12 → a 3 = 21 →
  a 30 = 264 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l3212_321290


namespace NUMINAMATH_CALUDE_first_month_sale_is_7435_l3212_321299

/-- Calculates the sale in the first month given the sales for months 2-6 and the average sale --/
def first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale2 + sale3 + sale4 + sale5 + sale6)

/-- Proves that the first month's sale is 7435 given the problem conditions --/
theorem first_month_sale_is_7435 :
  first_month_sale 7920 7855 8230 7560 6000 7500 = 7435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_is_7435_l3212_321299


namespace NUMINAMATH_CALUDE_max_profit_theorem_profit_range_theorem_l3212_321206

/-- The daily sales volume as a function of the selling price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1000

/-- The daily profit as a function of the selling price -/
def profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The theorem stating the maximum profit and the price at which it occurs -/
theorem max_profit_theorem :
  ∃ (max_profit : ℝ) (optimal_price : ℝ),
    (∀ x : ℝ, 50 ≤ x ∧ x ≤ 65 → profit x ≤ max_profit) ∧
    profit optimal_price = max_profit ∧
    optimal_price = 65 ∧
    max_profit = 8750 :=
sorry

/-- The theorem stating the range of prices for which the profit is at least 8000 -/
theorem profit_range_theorem :
  ∀ x : ℝ, (60 ≤ x ∧ x ≤ 65) ↔ profit x ≥ 8000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_theorem_profit_range_theorem_l3212_321206


namespace NUMINAMATH_CALUDE_square_area_ratio_when_tripled_l3212_321244

theorem square_area_ratio_when_tripled (s : ℝ) (h : s > 0) :
  (3 * s)^2 / s^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_when_tripled_l3212_321244


namespace NUMINAMATH_CALUDE_all_four_digit_numbers_generated_l3212_321245

/-- Represents the operations that can be performed on a number -/
inductive Operation
  | mul2sub2 : Operation
  | mul3add4 : Operation
  | add7 : Operation

/-- Applies an operation to a number -/
def applyOperation (op : Operation) (x : ℕ) : ℕ :=
  match op with
  | Operation.mul2sub2 => 2 * x - 2
  | Operation.mul3add4 => 3 * x + 4
  | Operation.add7 => x + 7

/-- Returns true if the number is four digits -/
def isFourDigits (n : ℕ) : Bool :=
  1000 ≤ n ∧ n ≤ 9999

/-- The set of all four-digit numbers -/
def fourDigitNumbers : Set ℕ :=
  {n : ℕ | isFourDigits n}

/-- The set of numbers that can be generated from 1 using the given operations -/
def generatedNumbers : Set ℕ :=
  {n : ℕ | ∃ (ops : List Operation), n = ops.foldl (fun acc op => applyOperation op acc) 1}

/-- Theorem stating that all four-digit numbers can be generated -/
theorem all_four_digit_numbers_generated :
  fourDigitNumbers ⊆ generatedNumbers :=
sorry

end NUMINAMATH_CALUDE_all_four_digit_numbers_generated_l3212_321245


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3212_321293

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 28 x = Nat.choose 28 (3 * x - 8)) → (x = 4 ∨ x = 9) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3212_321293


namespace NUMINAMATH_CALUDE_cube_property_l3212_321266

-- Define a cube type
structure Cube where
  side : ℝ
  volume_eq : volume = 8 * x
  area_eq : surfaceArea = x / 2

-- Define volume and surface area functions
def volume (c : Cube) : ℝ := c.side ^ 3
def surfaceArea (c : Cube) : ℝ := 6 * c.side ^ 2

-- State the theorem
theorem cube_property (x : ℝ) (c : Cube) : x = 110592 := by
  sorry

end NUMINAMATH_CALUDE_cube_property_l3212_321266


namespace NUMINAMATH_CALUDE_f_maximized_at_three_tenths_l3212_321228

/-- The probability that exactly k out of n items are defective, given probability p for each item -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability that exactly 3 out of 10 items are defective -/
def f (p : ℝ) : ℝ := binomial_probability 10 3 p

/-- The theorem stating that f(p) is maximized when p = 3/10 -/
theorem f_maximized_at_three_tenths (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  ∃ (max_p : ℝ), max_p = 3/10 ∧ ∀ q, 0 < q → q < 1 → f q ≤ f max_p :=
sorry

end NUMINAMATH_CALUDE_f_maximized_at_three_tenths_l3212_321228


namespace NUMINAMATH_CALUDE_total_memory_space_l3212_321286

def morning_songs : ℕ := 10
def afternoon_songs : ℕ := 15
def night_songs : ℕ := 3
def song_size : ℕ := 5

theorem total_memory_space : 
  (morning_songs + afternoon_songs + night_songs) * song_size = 140 :=
by sorry

end NUMINAMATH_CALUDE_total_memory_space_l3212_321286


namespace NUMINAMATH_CALUDE_right_triangle_leg_lengths_l3212_321280

theorem right_triangle_leg_lengths 
  (c : ℝ) 
  (α β : ℝ) 
  (h_right : α + β = π / 2) 
  (h_tan : 6 * Real.tan β = Real.tan α + 1) :
  ∃ (a b : ℝ), 
    a^2 + b^2 = c^2 ∧ 
    a = (2 * c * Real.sqrt 5) / 5 ∧ 
    b = (c * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_lengths_l3212_321280


namespace NUMINAMATH_CALUDE_pentagon_area_relationship_l3212_321204

/-- Represents the areas of different parts of a pentagon -/
structure PentagonAreas where
  x : ℝ  -- Area of the smaller similar pentagon
  y : ℝ  -- Area of one type of surrounding region
  z : ℝ  -- Area of another type of surrounding region
  total : ℝ  -- Total area of the larger pentagon

/-- Theorem about the relationship between areas in a specially divided pentagon -/
theorem pentagon_area_relationship (p : PentagonAreas) 
  (h_positive : p.x > 0 ∧ p.y > 0 ∧ p.z > 0 ∧ p.total > 0)
  (h_similar : ∃ (k : ℝ), k > 0 ∧ p.x = k^2 * p.total)
  (h_total : p.total = p.x + 5*p.y + 5*p.z) :
  p.y = p.z ∧ 
  p.y = (p.total - p.x) / 10 ∧
  p.total = p.x + 10*p.y := by
  sorry


end NUMINAMATH_CALUDE_pentagon_area_relationship_l3212_321204


namespace NUMINAMATH_CALUDE_counterexample_exists_l3212_321209

theorem counterexample_exists : ∃ (a b c : ℝ), 0 < a ∧ a < b ∧ b < c ∧ a ≥ b * c := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3212_321209


namespace NUMINAMATH_CALUDE_initial_investment_calculation_l3212_321268

def initial_rate : ℚ := 5 / 100
def additional_rate : ℚ := 8 / 100
def total_rate : ℚ := 6 / 100
def additional_investment : ℚ := 4000

theorem initial_investment_calculation (x : ℚ) :
  initial_rate * x + additional_rate * additional_investment = 
  total_rate * (x + additional_investment) →
  x = 8000 := by
sorry

end NUMINAMATH_CALUDE_initial_investment_calculation_l3212_321268


namespace NUMINAMATH_CALUDE_balls_after_5000_steps_l3212_321265

/-- Converts a natural number to its base-6 representation --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Sums the digits in a list of natural numbers --/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (· + ·) 0

/-- Represents the ball placement process --/
def ballPlacement (steps : ℕ) : ℕ :=
  sumDigits (toBase6 steps)

/-- The main theorem stating that after 5000 steps, there are 13 balls in the boxes --/
theorem balls_after_5000_steps :
  ballPlacement 5000 = 13 := by
  sorry

end NUMINAMATH_CALUDE_balls_after_5000_steps_l3212_321265


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l3212_321238

theorem floor_ceiling_sum : ⌊(3.999 : ℝ)⌋ + ⌈(4.001 : ℝ)⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l3212_321238


namespace NUMINAMATH_CALUDE_expression_value_l3212_321214

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3212_321214


namespace NUMINAMATH_CALUDE_hoseok_number_subtraction_l3212_321255

theorem hoseok_number_subtraction (n : ℕ) : n / 10 = 6 → n - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_number_subtraction_l3212_321255


namespace NUMINAMATH_CALUDE_uncovered_area_square_circle_l3212_321200

/-- The area of a square that cannot be covered by a moving circle -/
theorem uncovered_area_square_circle (square_side : ℝ) (circle_diameter : ℝ) 
  (h_square : square_side = 4)
  (h_circle : circle_diameter = 1) :
  (square_side - circle_diameter) ^ 2 + π * (circle_diameter / 2) ^ 2 = 4 + π / 4 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_square_circle_l3212_321200


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l3212_321278

-- Define the number of houses on the block
def num_houses : ℕ := 6

-- Define the total number of junk mail pieces
def total_junk_mail : ℕ := 24

-- Define the function to calculate junk mail per house
def junk_mail_per_house (houses : ℕ) (total_mail : ℕ) : ℕ :=
  total_mail / houses

-- Theorem statement
theorem junk_mail_distribution :
  junk_mail_per_house num_houses total_junk_mail = 4 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l3212_321278


namespace NUMINAMATH_CALUDE_corner_value_theorem_l3212_321277

/-- Represents a 3x3 grid with the given corner values -/
structure Grid :=
  (top_left : ℤ)
  (top_right : ℤ)
  (bottom_left : ℤ)
  (bottom_right : ℤ)
  (top_middle : ℤ)
  (left_middle : ℤ)
  (right_middle : ℤ)
  (bottom_middle : ℤ)
  (center : ℤ)

/-- Checks if all 2x2 subgrids have the same sum -/
def equal_subgrid_sums (g : Grid) : Prop :=
  g.top_left + g.top_middle + g.left_middle + g.center =
  g.top_middle + g.top_right + g.center + g.right_middle ∧
  g.left_middle + g.center + g.bottom_left + g.bottom_middle =
  g.center + g.right_middle + g.bottom_middle + g.bottom_right

/-- The main theorem -/
theorem corner_value_theorem (g : Grid) 
  (h1 : g.top_left = 2)
  (h2 : g.top_right = 4)
  (h3 : g.bottom_right = 3)
  (h4 : equal_subgrid_sums g) :
  g.bottom_left = 1 := by
  sorry

end NUMINAMATH_CALUDE_corner_value_theorem_l3212_321277


namespace NUMINAMATH_CALUDE_set_equality_condition_l3212_321227

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 3}

-- State the theorem
theorem set_equality_condition (a : ℝ) : 
  A ∪ B a = A ↔ a ≤ -2 ∨ a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_set_equality_condition_l3212_321227


namespace NUMINAMATH_CALUDE_unique_solution_values_l3212_321201

/-- The function representing the quadratic expression inside the absolute value -/
def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 3*a

/-- The inequality condition -/
def inequality_condition (a x : ℝ) : Prop := |f a x| ≤ 2

/-- The property of having exactly one solution -/
def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃! x, inequality_condition a x

/-- The main theorem stating that a = 1 and a = 2 are the only values satisfying the condition -/
theorem unique_solution_values :
  ∀ a : ℝ, has_exactly_one_solution a ↔ (a = 1 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_values_l3212_321201


namespace NUMINAMATH_CALUDE_simplification_and_exponent_sum_l3212_321203

-- Define the original expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^9 * z^14)^(1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z^3 * (5 * x^2 * z^5)^(1/3)

-- Define the sum of exponents outside the radical
def sum_of_exponents : ℕ := 1 + 1 + 3

-- Theorem statement
theorem simplification_and_exponent_sum (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) : 
  original_expression x y z = simplified_expression x y z ∧ 
  sum_of_exponents = 5 := by
  sorry

end NUMINAMATH_CALUDE_simplification_and_exponent_sum_l3212_321203


namespace NUMINAMATH_CALUDE_square_digit_sum_100_bound_l3212_321283

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem square_digit_sum_100_bound (n : ℕ) :
  sum_of_digits (n^2) = 100 → n ≤ 100 := by sorry

end NUMINAMATH_CALUDE_square_digit_sum_100_bound_l3212_321283


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3212_321211

/-- 
Given an ellipse with equation x²/25 + y²/16 = 1, 
if the distance from a point P on the ellipse to one focus is 3, 
then the distance from P to the other focus is 7.
-/
theorem ellipse_foci_distance (x y : ℝ) (P : ℝ × ℝ) :
  x^2 / 25 + y^2 / 16 = 1 →  -- Ellipse equation
  P.1^2 / 25 + P.2^2 / 16 = 1 →  -- Point P is on the ellipse
  ∃ (F₁ F₂ : ℝ × ℝ), -- There exist two foci F₁ and F₂
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 3 ∨
     Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 3) →
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 7 ∨
     Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 7) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3212_321211


namespace NUMINAMATH_CALUDE_tan_660_degrees_l3212_321218

theorem tan_660_degrees : Real.tan (660 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_660_degrees_l3212_321218


namespace NUMINAMATH_CALUDE_intersection_line_slope_l3212_321224

theorem intersection_line_slope (u : ℝ) :
  let line1 := {(x, y) : ℝ × ℝ | 2 * x + 3 * y = 8 * u + 4}
  let line2 := {(x, y) : ℝ × ℝ | 3 * x + 2 * y = 9 * u + 1}
  let intersection := {(x, y) : ℝ × ℝ | (x, y) ∈ line1 ∩ line2}
  ∃ (m b : ℝ), m = 6 / 47 ∧ ∀ (x y : ℝ), (x, y) ∈ intersection → y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l3212_321224


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3212_321270

theorem other_root_of_quadratic (a : ℝ) : 
  (2^2 + 3*2 + a = 0) → (-5^2 + 3*(-5) + a = 0) := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3212_321270


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3212_321261

theorem min_value_of_expression :
  (∀ x y : ℝ, (2*x*y - 3)^2 + (x + y)^2 ≥ 1) ∧
  (∃ x y : ℝ, (2*x*y - 3)^2 + (x + y)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3212_321261


namespace NUMINAMATH_CALUDE_f_properties_l3212_321262

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + Real.sin (x + Real.pi / 2)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∃ x, f x = 55 / 8) ∧
  (∃ x, f x = -9 / 8) ∧
  (∀ x, f x ≤ 55 / 8) ∧
  (∀ x, f x ≥ -9 / 8) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3212_321262


namespace NUMINAMATH_CALUDE_monthly_salary_is_6250_l3212_321287

/-- Calculates the monthly salary given savings rate, expense increase, and new savings amount -/
def calculate_salary (savings_rate : ℚ) (expense_increase : ℚ) (new_savings : ℚ) : ℚ :=
  new_savings / (savings_rate - (1 - savings_rate) * expense_increase)

/-- Theorem stating that under the given conditions, the monthly salary is 6250 -/
theorem monthly_salary_is_6250 :
  let savings_rate : ℚ := 1/5
  let expense_increase : ℚ := 1/5
  let new_savings : ℚ := 250
  calculate_salary savings_rate expense_increase new_savings = 6250 := by
sorry

#eval calculate_salary (1/5) (1/5) 250

end NUMINAMATH_CALUDE_monthly_salary_is_6250_l3212_321287


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3212_321263

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, -1/2 < x ∧ x ≤ 1 ∧ 2^x - a > Real.arccos x) ↔ 
  a < Real.sqrt 2 / 2 - 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3212_321263


namespace NUMINAMATH_CALUDE_quadrilateral_area_inequality_l3212_321295

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  convex : Bool
  area : ℝ

-- State the theorem
theorem quadrilateral_area_inequality (q : ConvexQuadrilateral) (h : q.convex = true) :
  q.area ≤ (q.a^2 + q.b^2 + q.c^2 + q.d^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_inequality_l3212_321295


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l3212_321215

/-- Proves that three successive discounts are equivalent to a single discount --/
theorem successive_discounts_equivalence : 
  let original_price : ℝ := 800
  let discount1 : ℝ := 0.15
  let discount2 : ℝ := 0.10
  let discount3 : ℝ := 0.05
  let final_price : ℝ := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let single_discount : ℝ := 0.27325
  final_price = original_price * (1 - single_discount) := by
  sorry

#check successive_discounts_equivalence

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l3212_321215


namespace NUMINAMATH_CALUDE_cake_sugar_amount_l3212_321230

theorem cake_sugar_amount (total_sugar frosting_sugar : ℚ)
  (h1 : total_sugar = 0.8)
  (h2 : frosting_sugar = 0.6) :
  total_sugar - frosting_sugar = 0.2 := by
sorry

end NUMINAMATH_CALUDE_cake_sugar_amount_l3212_321230


namespace NUMINAMATH_CALUDE_train_length_calculation_l3212_321257

-- Define the given constants
def bridge_crossing_time : Real := 30  -- seconds
def train_speed : Real := 45  -- km/hr
def bridge_length : Real := 230  -- meters

-- Define the theorem
theorem train_length_calculation :
  let speed_in_meters_per_second : Real := train_speed * 1000 / 3600
  let total_distance : Real := speed_in_meters_per_second * bridge_crossing_time
  let train_length : Real := total_distance - bridge_length
  train_length = 145 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3212_321257


namespace NUMINAMATH_CALUDE_bus_speed_relation_l3212_321236

/-- Represents the speed and stoppage characteristics of a bus -/
structure Bus where
  speed_with_stops : ℝ
  stop_time : ℝ
  speed_without_stops : ℝ

/-- Theorem stating the relationship between bus speeds and stop time -/
theorem bus_speed_relation (b : Bus) 
  (h1 : b.speed_with_stops = 12)
  (h2 : b.stop_time = 45)
  : b.speed_without_stops = 48 := by
  sorry

#check bus_speed_relation

end NUMINAMATH_CALUDE_bus_speed_relation_l3212_321236


namespace NUMINAMATH_CALUDE_power_zero_eq_one_three_power_zero_l3212_321282

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem three_power_zero : (3 : ℝ)^0 = 1 := by sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_three_power_zero_l3212_321282


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_100_l3212_321288

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_non_factor_product_of_100 :
  ∃ (a b : ℕ),
    a ≠ b ∧
    a > 0 ∧
    b > 0 ∧
    is_factor a 100 ∧
    is_factor b 100 ∧
    ¬(is_factor (a * b) 100) ∧
    a * b = 8 ∧
    ∀ (c d : ℕ),
      c ≠ d →
      c > 0 →
      d > 0 →
      is_factor c 100 →
      is_factor d 100 →
      ¬(is_factor (c * d) 100) →
      c * d ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_100_l3212_321288
