import Mathlib

namespace NUMINAMATH_CALUDE_speed_conversion_proof_l883_88368

/-- Converts a speed from meters per second to kilometers per hour. -/
def convert_mps_to_kmh (speed_mps : ℚ) : ℚ :=
  speed_mps * 3.6

/-- Proves that converting 17/36 m/s to km/h results in 1.7 km/h. -/
theorem speed_conversion_proof :
  convert_mps_to_kmh (17/36) = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_proof_l883_88368


namespace NUMINAMATH_CALUDE_collinear_points_sum_l883_88303

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p q r : ℝ × ℝ × ℝ) : Prop := sorry

/-- The theorem states that if the given points are collinear, then a + b = 6. -/
theorem collinear_points_sum (a b : ℝ) : 
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l883_88303


namespace NUMINAMATH_CALUDE_root_sum_arctan_l883_88373

theorem root_sum_arctan (x₁ x₂ : ℝ) (α β : ℝ) : 
  x₁^2 + 3 * Real.sqrt 3 * x₁ + 4 = 0 →
  x₂^2 + 3 * Real.sqrt 3 * x₂ + 4 = 0 →
  α = Real.arctan x₁ →
  β = Real.arctan x₂ →
  α + β = π / 3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_arctan_l883_88373


namespace NUMINAMATH_CALUDE_comics_in_box_l883_88323

theorem comics_in_box (pages_per_comic : ℕ) (found_pages : ℕ) (untorn_comics : ℕ) : 
  pages_per_comic = 25 →
  found_pages = 150 →
  untorn_comics = 5 →
  (found_pages / pages_per_comic + untorn_comics : ℕ) = 11 :=
by sorry

end NUMINAMATH_CALUDE_comics_in_box_l883_88323


namespace NUMINAMATH_CALUDE_min_buses_is_eleven_l883_88347

/-- The maximum number of students a bus can hold -/
def bus_capacity : ℕ := 38

/-- The total number of students to be transported -/
def total_students : ℕ := 411

/-- The minimum number of buses needed is the ceiling of the division of total students by bus capacity -/
def min_buses : ℕ := (total_students + bus_capacity - 1) / bus_capacity

/-- Theorem stating that the minimum number of buses needed is 11 -/
theorem min_buses_is_eleven : min_buses = 11 := by sorry

end NUMINAMATH_CALUDE_min_buses_is_eleven_l883_88347


namespace NUMINAMATH_CALUDE_inequality_proof_l883_88370

theorem inequality_proof (a b : ℝ) (h : a + b > 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l883_88370


namespace NUMINAMATH_CALUDE_first_concert_attendance_l883_88321

theorem first_concert_attendance (second_concert : ℕ) (difference : ℕ) : 
  second_concert = 66018 → difference = 119 → second_concert - difference = 65899 := by
  sorry

end NUMINAMATH_CALUDE_first_concert_attendance_l883_88321


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l883_88393

theorem imaginary_part_of_z (z : ℂ) : z = (1 - I) / (1 + 3*I) → z.im = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l883_88393


namespace NUMINAMATH_CALUDE_lowry_big_bonsai_sold_l883_88354

/-- Represents the sale of bonsai trees -/
structure BonsaiSale where
  small_price : ℕ  -- Price of a small bonsai
  big_price : ℕ    -- Price of a big bonsai
  small_sold : ℕ   -- Number of small bonsai sold
  total_earnings : ℕ -- Total earnings from the sale

/-- Calculates the number of big bonsai sold -/
def big_bonsai_sold (sale : BonsaiSale) : ℕ :=
  (sale.total_earnings - sale.small_price * sale.small_sold) / sale.big_price

/-- Theorem stating the number of big bonsai sold in Lowry's sale -/
theorem lowry_big_bonsai_sold :
  let sale := BonsaiSale.mk 30 20 3 190
  big_bonsai_sold sale = 5 := by
  sorry

end NUMINAMATH_CALUDE_lowry_big_bonsai_sold_l883_88354


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l883_88355

theorem cubic_polynomials_common_roots (c d : ℝ) :
  c = -5 ∧ d = -6 →
  ∃ (r s : ℝ), r ≠ s ∧
    (r^3 + c*r^2 + 12*r + 7 = 0) ∧ 
    (r^3 + d*r^2 + 15*r + 9 = 0) ∧
    (s^3 + c*s^2 + 12*s + 7 = 0) ∧ 
    (s^3 + d*s^2 + 15*s + 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l883_88355


namespace NUMINAMATH_CALUDE_expression_simplification_l883_88381

theorem expression_simplification (w : ℝ) : 2*w + 4*w + 6*w + 8*w + 10*w + 12 = 30*w + 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l883_88381


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l883_88339

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n_terms a n / sum_n_terms b n = (2 * n + 2 : ℚ) / (n + 3)) →
  a.a 10 / b.a 10 = 20 / 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l883_88339


namespace NUMINAMATH_CALUDE_butterfly_collection_l883_88365

theorem butterfly_collection (total : ℕ) (black : ℕ) :
  total = 11 →
  black = 5 →
  ∃ (blue yellow : ℕ),
    blue = 2 * yellow ∧
    blue + yellow + black = total ∧
    blue = 4 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_collection_l883_88365


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l883_88391

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ - 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l883_88391


namespace NUMINAMATH_CALUDE_all_graphs_different_l883_88319

-- Define the three equations
def eq1 (x y : ℝ) : Prop := y = x - 1
def eq2 (x y : ℝ) : Prop := y = (x^2 - 1) / (x + 1)
def eq3 (x y : ℝ) : Prop := (x + 1) * y = x^2 - 1

-- Define what it means for two equations to have the same graph
def same_graph (eq_a eq_b : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq_a x y ↔ eq_b x y

-- Theorem stating that all equations have different graphs
theorem all_graphs_different :
  ¬(same_graph eq1 eq2) ∧ ¬(same_graph eq1 eq3) ∧ ¬(same_graph eq2 eq3) :=
sorry

end NUMINAMATH_CALUDE_all_graphs_different_l883_88319


namespace NUMINAMATH_CALUDE_period_of_inverse_a_l883_88390

/-- Represents a 100-digit number with 1 at the start, 6 at the end, and 98 sevens in between -/
def a : ℕ := 1777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777776

/-- The period of the decimal representation of 1/n -/
def decimal_period (n : ℕ) : ℕ := sorry

theorem period_of_inverse_a : decimal_period a = 99 := by sorry

end NUMINAMATH_CALUDE_period_of_inverse_a_l883_88390


namespace NUMINAMATH_CALUDE_gas_tank_fill_level_l883_88306

theorem gas_tank_fill_level (tank_capacity : ℚ) (initial_fill_fraction : ℚ) (added_amount : ℚ) : 
  tank_capacity = 42 → 
  initial_fill_fraction = 3/4 → 
  added_amount = 7 → 
  (initial_fill_fraction * tank_capacity + added_amount) / tank_capacity = 833/909 := by
  sorry

end NUMINAMATH_CALUDE_gas_tank_fill_level_l883_88306


namespace NUMINAMATH_CALUDE_gcd_count_for_product_252_l883_88388

theorem gcd_count_for_product_252 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 252) :
  ∃! (s : Finset ℕ+), s.card = 8 ∧ ∀ d, d ∈ s ↔ ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 252 :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_252_l883_88388


namespace NUMINAMATH_CALUDE_knicks_to_knocks_equivalence_l883_88305

/-- Represents the number of units of a given type -/
structure UnitCount (α : Type) where
  count : ℚ

/-- Conversion rate between two types of units -/
def ConversionRate (α β : Type) : Type :=
  UnitCount α → UnitCount β

/-- Given conversion rates, prove that 40 knicks are equivalent to 36 knocks -/
theorem knicks_to_knocks_equivalence 
  (knick knack knock : Type)
  (knicks_to_knacks : ConversionRate knick knack)
  (knacks_to_knocks : ConversionRate knack knock)
  (h1 : knicks_to_knacks ⟨5⟩ = ⟨3⟩)
  (h2 : knacks_to_knocks ⟨4⟩ = ⟨6⟩)
  : ∃ (f : ConversionRate knick knock), f ⟨40⟩ = ⟨36⟩ := by
  sorry

end NUMINAMATH_CALUDE_knicks_to_knocks_equivalence_l883_88305


namespace NUMINAMATH_CALUDE_not_all_probabilities_equal_l883_88360

/-- Represents a student in the sampling process -/
structure Student :=
  (id : Nat)

/-- Represents the sampling process -/
structure SamplingProcess :=
  (totalStudents : Nat)
  (selectedStudents : Nat)
  (excludedStudents : Nat)

/-- Represents the probability of a student being selected -/
def selectionProbability (student : Student) (process : SamplingProcess) : ℝ :=
  sorry

/-- The main theorem stating that not all probabilities are equal -/
theorem not_all_probabilities_equal
  (process : SamplingProcess)
  (h1 : process.totalStudents = 2010)
  (h2 : process.selectedStudents = 50)
  (h3 : process.excludedStudents = 10) :
  ∃ (s1 s2 : Student), selectionProbability s1 process ≠ selectionProbability s2 process :=
sorry

end NUMINAMATH_CALUDE_not_all_probabilities_equal_l883_88360


namespace NUMINAMATH_CALUDE_sixteen_eggs_in_groups_of_two_l883_88361

/-- The number of groups formed when splitting a given number of eggs into groups of a specified size -/
def number_of_groups (total_eggs : ℕ) (group_size : ℕ) : ℕ :=
  total_eggs / group_size

/-- Theorem: Splitting 16 eggs into groups of 2 results in 8 groups -/
theorem sixteen_eggs_in_groups_of_two :
  number_of_groups 16 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_eggs_in_groups_of_two_l883_88361


namespace NUMINAMATH_CALUDE_perpendicular_line_angle_l883_88331

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x - Real.sqrt 3 * y + 2 = 0

-- Define the angle of inclination of a line perpendicular to l
def perpendicular_angle (θ : ℝ) : Prop :=
  Real.tan θ = -(Real.sqrt 3 / 3)

-- Theorem statement
theorem perpendicular_line_angle :
  ∃ θ, perpendicular_angle θ ∧ θ = 150 * (π / 180) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_angle_l883_88331


namespace NUMINAMATH_CALUDE_total_revenue_is_146475_l883_88314

/-- The number of cookies baked by Clementine -/
def C : ℕ := 72

/-- The number of cookies baked by Jake -/
def J : ℕ := (5 * C) / 2

/-- The number of cookies baked by Tory -/
def T : ℕ := (J + C) / 2

/-- The number of cookies baked by Spencer -/
def S : ℕ := (3 * (J + T)) / 2

/-- The price of each cookie in cents -/
def price_per_cookie : ℕ := 175

/-- The total revenue in cents -/
def total_revenue : ℕ := (C + J + T + S) * price_per_cookie

theorem total_revenue_is_146475 : total_revenue = 146475 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_146475_l883_88314


namespace NUMINAMATH_CALUDE_inner_shape_area_ratio_l883_88385

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- Points that trisect the sides of a hexagon -/
def trisection_points (h : RegularHexagon) : Fin 6 → ℝ × ℝ :=
  sorry

/-- The shape formed by joining the trisection points -/
def inner_shape (h : RegularHexagon) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating that the area of the inner shape is 2/3 of the original hexagon -/
theorem inner_shape_area_ratio (h : RegularHexagon) :
    area (inner_shape h) = (2 / 3) * area (Set.range h.vertices) := by
  sorry

end NUMINAMATH_CALUDE_inner_shape_area_ratio_l883_88385


namespace NUMINAMATH_CALUDE_pet_store_cages_l883_88337

theorem pet_store_cages (birds_per_cage : ℕ) (total_birds : ℕ) (num_cages : ℕ) : 
  birds_per_cage = 8 → 
  total_birds = 48 → 
  num_cages * birds_per_cage = total_birds → 
  num_cages = 6 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cages_l883_88337


namespace NUMINAMATH_CALUDE_parabola_intersection_sum_of_squares_l883_88357

theorem parabola_intersection_sum_of_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + k*x + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 > 8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_sum_of_squares_l883_88357


namespace NUMINAMATH_CALUDE_faye_earnings_proof_l883_88341

/-- The number of bead necklaces Faye sold -/
def bead_necklaces : ℕ := 3

/-- The number of gem stone necklaces Faye sold -/
def gem_necklaces : ℕ := 7

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 7

/-- Faye's total earnings from selling necklaces -/
def faye_earnings : ℕ := (bead_necklaces + gem_necklaces) * necklace_cost

theorem faye_earnings_proof : faye_earnings = 70 := by
  sorry

end NUMINAMATH_CALUDE_faye_earnings_proof_l883_88341


namespace NUMINAMATH_CALUDE_hot_water_bottle_price_is_six_l883_88335

/-- The price of a hot-water bottle given the conditions of the problem -/
def hot_water_bottle_price (thermometer_price : ℚ) (total_sales : ℚ) 
  (thermometer_to_bottle_ratio : ℕ) (bottles_sold : ℕ) : ℚ :=
  (total_sales - thermometer_price * (thermometer_to_bottle_ratio * bottles_sold)) / bottles_sold

theorem hot_water_bottle_price_is_six :
  hot_water_bottle_price 2 1200 7 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hot_water_bottle_price_is_six_l883_88335


namespace NUMINAMATH_CALUDE_division_problem_l883_88320

theorem division_problem (x y : ℕ+) 
  (h1 : (x : ℝ) / (y : ℝ) = 96.12)
  (h2 : (x : ℝ) % (y : ℝ) = 11.52) : 
  y = 96 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l883_88320


namespace NUMINAMATH_CALUDE_function_symmetry_and_periodicity_l883_88327

/-- A function f: ℝ → ℝ is symmetric about the line x = a if f(2a - x) = f(x) for all x ∈ ℝ -/
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

/-- A function f: ℝ → ℝ is periodic with period p if f(x + p) = f(x) for all x ∈ ℝ -/
def Periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

/-- A function f: ℝ → ℝ is symmetric about the point (a, b) if f(2a - x) = 2b - f(x) for all x ∈ ℝ -/
def SymmetricAboutPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = 2 * b - f x

theorem function_symmetry_and_periodicity (f : ℝ → ℝ) :
  (∀ x, f (2 - x) = f x) → SymmetricAboutLine f 1 ∧
  (∀ x, f (x - 1) = f (x + 1)) → Periodic f 2 ∧
  (∀ x, f (2 - x) = -f x) → SymmetricAboutPoint f 1 0 := by
  sorry


end NUMINAMATH_CALUDE_function_symmetry_and_periodicity_l883_88327


namespace NUMINAMATH_CALUDE_max_segments_theorem_l883_88396

/-- A configuration of points on a plane. -/
structure PointConfiguration where
  n : ℕ  -- number of points
  m : ℕ  -- number of points on the convex hull
  no_collinear_triple : Bool  -- no three points are collinear
  m_le_n : m ≤ n  -- number of points on convex hull cannot exceed total points

/-- The maximum number of non-intersecting line segments for a given point configuration. -/
def max_segments (config : PointConfiguration) : ℕ :=
  3 * config.n - config.m - 3

/-- Theorem stating the maximum number of non-intersecting line segments. -/
theorem max_segments_theorem (config : PointConfiguration) :
  config.no_collinear_triple →
  max_segments config = 3 * config.n - config.m - 3 :=
sorry

end NUMINAMATH_CALUDE_max_segments_theorem_l883_88396


namespace NUMINAMATH_CALUDE_pirate_rick_sand_ratio_l883_88380

/-- Pirate Rick's treasure digging problem -/
theorem pirate_rick_sand_ratio :
  let initial_sand : ℝ := 8
  let initial_time : ℝ := 4
  let tsunami_sand : ℝ := 2
  let final_time : ℝ := 3
  let digging_rate : ℝ := initial_sand / initial_time
  let final_sand : ℝ := final_time * digging_rate
  let storm_sand : ℝ := initial_sand + tsunami_sand - final_sand
  storm_sand / initial_sand = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_pirate_rick_sand_ratio_l883_88380


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l883_88397

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) (h4 : w = 5) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (rectangle_length + w) = 22 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l883_88397


namespace NUMINAMATH_CALUDE_fruit_purchase_total_l883_88333

/-- Calculates the total amount paid for fruits after discounts --/
def total_amount_paid (peach_price apple_price orange_price : ℚ)
                      (peach_count apple_count orange_count : ℕ)
                      (peach_discount apple_discount orange_discount : ℚ)
                      (peach_discount_threshold apple_discount_threshold orange_discount_threshold : ℚ) : ℚ :=
  let peach_total := peach_price * peach_count
  let apple_total := apple_price * apple_count
  let orange_total := orange_price * orange_count
  let peach_discount_applied := (peach_total / peach_discount_threshold).floor * peach_discount
  let apple_discount_applied := (apple_total / apple_discount_threshold).floor * apple_discount
  let orange_discount_applied := (orange_total / orange_discount_threshold).floor * orange_discount
  peach_total + apple_total + orange_total - peach_discount_applied - apple_discount_applied - orange_discount_applied

/-- Theorem stating the total amount paid for fruits after discounts --/
theorem fruit_purchase_total :
  total_amount_paid (40/100) (60/100) (50/100) 400 150 200 2 3 (3/2) 10 15 7 = 279 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_total_l883_88333


namespace NUMINAMATH_CALUDE_golden_rectangle_perimeter_l883_88345

/-- A golden rectangle is a rectangle where the ratio of its width to its length is (√5 - 1) / 2 -/
def is_golden_rectangle (width length : ℝ) : Prop :=
  width / length = (Real.sqrt 5 - 1) / 2

/-- The perimeter of a rectangle given its width and length -/
def rectangle_perimeter (width length : ℝ) : ℝ :=
  2 * (width + length)

theorem golden_rectangle_perimeter :
  ∀ width length : ℝ,
  is_golden_rectangle width length →
  (width = Real.sqrt 5 - 1 ∨ length = Real.sqrt 5 - 1) →
  rectangle_perimeter width length = 4 ∨ rectangle_perimeter width length = 2 * Real.sqrt 5 + 2 :=
by sorry

end NUMINAMATH_CALUDE_golden_rectangle_perimeter_l883_88345


namespace NUMINAMATH_CALUDE_algebraic_expression_symmetry_l883_88338

/-- Given an algebraic expression ax^5 + bx^3 + cx - 8, if its value is 6 when x = 5,
    then its value is -22 when x = -5 -/
theorem algebraic_expression_symmetry (a b c : ℝ) :
  (5^5 * a + 5^3 * b + 5 * c - 8 = 6) →
  ((-5)^5 * a + (-5)^3 * b + (-5) * c - 8 = -22) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_symmetry_l883_88338


namespace NUMINAMATH_CALUDE_lloyds_work_hours_l883_88358

/-- Calculates the total hours worked given the conditions of Lloyd's work and pay --/
theorem lloyds_work_hours
  (regular_hours : ℝ)
  (regular_rate : ℝ)
  (overtime_multiplier : ℝ)
  (total_pay : ℝ)
  (h1 : regular_hours = 7.5)
  (h2 : regular_rate = 4)
  (h3 : overtime_multiplier = 1.5)
  (h4 : total_pay = 48) :
  ∃ (total_hours : ℝ), total_hours = 10.5 ∧
    total_pay = regular_hours * regular_rate +
                (total_hours - regular_hours) * (regular_rate * overtime_multiplier) :=
by sorry

end NUMINAMATH_CALUDE_lloyds_work_hours_l883_88358


namespace NUMINAMATH_CALUDE_sports_club_membership_l883_88313

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 42 →
  badminton = 20 →
  tennis = 23 →
  both = 7 →
  total - (badminton + tennis - both) = 6 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_membership_l883_88313


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l883_88318

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The given number in base 7 --/
def base7Number : List Nat := [4, 3, 6, 2, 5]

/-- Theorem: The base 10 equivalent of 52634₇ is 13010 --/
theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 13010 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l883_88318


namespace NUMINAMATH_CALUDE_count_satisfying_integers_l883_88374

def is_geometric_mean_integer (n : ℕ+) : Prop :=
  ∃ k : ℕ+, n = 2015 * k^2

def is_harmonic_mean_integer (n : ℕ+) : Prop :=
  ∃ m : ℕ+, 2 * 2015 * n = m * (2015 + n)

def satisfies_conditions (n : ℕ+) : Prop :=
  is_geometric_mean_integer n ∧ is_harmonic_mean_integer n

theorem count_satisfying_integers :
  (∃! (s : Finset ℕ+), s.card = 5 ∧ ∀ n, n ∈ s ↔ satisfies_conditions n) ∧
  2015 = 5 * 13 * 31 := by
  sorry

end NUMINAMATH_CALUDE_count_satisfying_integers_l883_88374


namespace NUMINAMATH_CALUDE_cube_mean_inequality_l883_88328

theorem cube_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 + b^3) / 2 ≥ ((a + b) / 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_mean_inequality_l883_88328


namespace NUMINAMATH_CALUDE_chocolate_mixture_percentage_l883_88382

theorem chocolate_mixture_percentage (initial_amount : ℝ) (initial_percentage : ℝ) 
  (added_amount : ℝ) (desired_percentage : ℝ) : 
  initial_amount = 220 →
  initial_percentage = 0.5 →
  added_amount = 220 →
  desired_percentage = 0.75 →
  (initial_amount * initial_percentage + added_amount) / (initial_amount + added_amount) = desired_percentage :=
by sorry

end NUMINAMATH_CALUDE_chocolate_mixture_percentage_l883_88382


namespace NUMINAMATH_CALUDE_medical_team_arrangements_l883_88330

/-- The number of male doctors --/
def num_male_doctors : ℕ := 6

/-- The number of female nurses --/
def num_female_nurses : ℕ := 3

/-- The number of medical teams --/
def num_teams : ℕ := 3

/-- The number of male doctors per team --/
def doctors_per_team : ℕ := 2

/-- The number of female nurses per team --/
def nurses_per_team : ℕ := 1

/-- The number of distinct locations --/
def num_locations : ℕ := 3

/-- The total number of arrangements --/
def total_arrangements : ℕ := 540

theorem medical_team_arrangements :
  (num_male_doctors.choose doctors_per_team *
   (num_male_doctors - doctors_per_team).choose doctors_per_team *
   (num_male_doctors - 2 * doctors_per_team).choose doctors_per_team) /
  num_teams.factorial *
  num_teams.factorial *
  num_teams.factorial = total_arrangements :=
sorry

end NUMINAMATH_CALUDE_medical_team_arrangements_l883_88330


namespace NUMINAMATH_CALUDE_final_piggy_bank_amount_l883_88395

def piggy_bank_savings (initial_amount : ℝ) (weekly_allowance : ℝ) (savings_fraction : ℝ) (weeks : ℕ) : ℝ :=
  initial_amount + (weekly_allowance * savings_fraction * weeks)

theorem final_piggy_bank_amount :
  piggy_bank_savings 43 10 0.5 8 = 83 := by
  sorry

end NUMINAMATH_CALUDE_final_piggy_bank_amount_l883_88395


namespace NUMINAMATH_CALUDE_flat_terrain_distance_l883_88364

theorem flat_terrain_distance (total_time : ℚ) (total_distance : ℕ) 
  (speed_uphill speed_flat speed_downhill : ℚ) 
  (h_total_time : total_time = 29 / 15)
  (h_total_distance : total_distance = 9)
  (h_speed_uphill : speed_uphill = 4)
  (h_speed_flat : speed_flat = 5)
  (h_speed_downhill : speed_downhill = 6) :
  ∃ (x y : ℕ), 
    x + y ≤ total_distance ∧
    x / speed_uphill + y / speed_flat + (total_distance - x - y) / speed_downhill = total_time ∧
    y = 3 := by
  sorry

end NUMINAMATH_CALUDE_flat_terrain_distance_l883_88364


namespace NUMINAMATH_CALUDE_sean_charles_whistle_difference_l883_88398

theorem sean_charles_whistle_difference : 
  ∀ (sean_whistles charles_whistles : ℕ),
    sean_whistles = 223 →
    charles_whistles = 128 →
    sean_whistles - charles_whistles = 95 := by
  sorry

end NUMINAMATH_CALUDE_sean_charles_whistle_difference_l883_88398


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l883_88362

/-- The vertex of the parabola y = 1/2 * (x + 2)^2 + 1 has coordinates (-2, 1) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x ↦ (1/2) * (x + 2)^2 + 1
  ∃! (h k : ℝ), (∀ x, f x = (1/2) * (x - h)^2 + k) ∧ h = -2 ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l883_88362


namespace NUMINAMATH_CALUDE_max_M_value_l883_88342

/-- Given a system of equations and conditions, prove the maximum value of M --/
theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (heq1 : x - 2*y = z - 2*u) (heq2 : 2*y*z = u*x) (hyz : z ≥ y) :
  ∃ (M : ℝ), M > 0 ∧ M ≤ z/y ∧ ∀ (N : ℝ), (N > 0 ∧ N ≤ z/y → N ≤ 6 + 4*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_M_value_l883_88342


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l883_88315

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 2/3) (h₃ : a₃ = 5/6) :
  arithmetic_sequence a₁ (a₂ - a₁) 10 = 2 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l883_88315


namespace NUMINAMATH_CALUDE_new_person_weight_l883_88399

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 98.6 kg -/
theorem new_person_weight :
  weight_of_new_person 8 4.2 65 = 98.6 := by
  sorry

#eval weight_of_new_person 8 4.2 65

end NUMINAMATH_CALUDE_new_person_weight_l883_88399


namespace NUMINAMATH_CALUDE_natural_number_solution_system_l883_88367

theorem natural_number_solution_system (x y z t a b : ℕ) : 
  x^2 + y^2 = a ∧ 
  z^2 + t^2 = b ∧ 
  (x^2 + t^2) * (z^2 + y^2) = 50 →
  ((x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨
   (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨
   (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1)) :=
by sorry

#check natural_number_solution_system

end NUMINAMATH_CALUDE_natural_number_solution_system_l883_88367


namespace NUMINAMATH_CALUDE_rod_mass_is_one_fourth_l883_88392

/-- The linear density function of the rod -/
def ρ : ℝ → ℝ := fun x ↦ x^3

/-- The length of the rod -/
def rod_length : ℝ := 1

/-- The mass of the rod -/
noncomputable def rod_mass : ℝ := ∫ x in (0)..(rod_length), ρ x

/-- Theorem: The mass of the rod is equal to 1/4 -/
theorem rod_mass_is_one_fourth : rod_mass = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_rod_mass_is_one_fourth_l883_88392


namespace NUMINAMATH_CALUDE_equation_solution_l883_88375

theorem equation_solution (a b x : ℝ) (h : b ≠ 0) :
  a * (Real.cos (x / 2))^2 - (a + 2 * b) * (Real.sin (x / 2))^2 = a * Real.cos x - b * Real.sin x ↔
  (∃ n : ℤ, x = 2 * n * Real.pi) ∨ (∃ k : ℤ, x = Real.pi / 2 * (4 * k + 1)) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l883_88375


namespace NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l883_88349

/-- The square with vertices at (0,0), (0,3), (3,3), and (3,0) -/
def square : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

/-- The region where x + y < 4 -/
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 < 4}

/-- The area of the square -/
def square_area : ℝ := 9

/-- The area of the region inside the square where x + y < 4 -/
def region_area : ℝ := 7

theorem probability_x_plus_y_less_than_4 :
  (region_area / square_area : ℝ) = 7 / 9 := by
  sorry


end NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l883_88349


namespace NUMINAMATH_CALUDE_max_value_implies_a_values_l883_88384

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

-- State the theorem
theorem max_value_implies_a_values (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = 2) →
  a = 2 ∨ a = -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_values_l883_88384


namespace NUMINAMATH_CALUDE_smallest_m_is_251_l883_88300

/-- Represents a circular arrangement of grids with real numbers -/
def CircularGrids (n : ℕ) := Fin n → ℝ

/-- Checks if the difference condition is satisfied for a given grid and step -/
def satisfiesDifferenceCondition (grids : CircularGrids 999) (a k : Fin 999) : Prop :=
  (grids a - grids ((a + k) % 999) = k) ∨ (grids a - grids ((999 + a - k) % 999) = k)

/-- Checks if the consecutive condition is satisfied for a given starting grid -/
def satisfiesConsecutiveCondition (grids : CircularGrids 999) (s : Fin 999) : Prop :=
  (∀ k : Fin 998, grids ((s + k) % 999) = grids s + k) ∨
  (∀ k : Fin 998, grids ((999 + s - k) % 999) = grids s + k)

/-- The main theorem stating that 251 is the smallest positive integer satisfying the conditions -/
theorem smallest_m_is_251 : 
  ∀ m : ℕ+, 
    (m = 251 ↔ 
      (∀ grids : CircularGrids 999, 
        (∀ a : Fin 999, ∀ k : Fin m, satisfiesDifferenceCondition grids a k) →
        (∃ s : Fin 999, satisfiesConsecutiveCondition grids s)) ∧
      (∀ m' : ℕ+, m' < m →
        ∃ grids : CircularGrids 999, 
          (∀ a : Fin 999, ∀ k : Fin m', satisfiesDifferenceCondition grids a k) ∧
          (∀ s : Fin 999, ¬satisfiesConsecutiveCondition grids s))) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_251_l883_88300


namespace NUMINAMATH_CALUDE_school_supplies_purchase_l883_88356

-- Define the cost of one unit of type A and one unit of type B
def cost_A : ℝ := 15
def cost_B : ℝ := 25

-- Define the total number of units to be purchased
def total_units : ℕ := 100

-- Define the maximum total cost
def max_total_cost : ℝ := 2000

-- Theorem to prove
theorem school_supplies_purchase :
  -- Condition 1: The sum of costs of one unit of each type is $40
  cost_A + cost_B = 40 →
  -- Condition 2: The number of units of type A that can be purchased with $90 
  -- is the same as the number of units of type B that can be purchased with $150
  90 / cost_A = 150 / cost_B →
  -- Condition 3: The total cost should not exceed $2000
  ∀ y : ℕ, y ≤ total_units → cost_A * y + cost_B * (total_units - y) ≤ max_total_cost →
  -- Conclusion: The minimum number of units of type A to be purchased is 50
  (∀ z : ℕ, z < 50 → cost_A * z + cost_B * (total_units - z) > max_total_cost) ∧
  cost_A * 50 + cost_B * (total_units - 50) ≤ max_total_cost :=
by sorry


end NUMINAMATH_CALUDE_school_supplies_purchase_l883_88356


namespace NUMINAMATH_CALUDE_computer_time_theorem_l883_88329

/-- Calculates the average time per person on a computer given the number of people, 
    number of computers, and working day duration. -/
def averageComputerTime (people : ℕ) (computers : ℕ) (workingHours : ℕ) (workingMinutes : ℕ) : ℕ :=
  let totalMinutes : ℕ := workingHours * 60 + workingMinutes
  let totalComputerTime : ℕ := totalMinutes * computers
  totalComputerTime / people

/-- Theorem stating that given 8 people, 5 computers, and a working day of 2 hours and 32 minutes, 
    the average time each person spends on a computer is 95 minutes. -/
theorem computer_time_theorem :
  averageComputerTime 8 5 2 32 = 95 := by
  sorry

end NUMINAMATH_CALUDE_computer_time_theorem_l883_88329


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l883_88336

theorem quadratic_rewrite (b n : ℝ) : 
  (∀ x, x^2 + b*x + 72 = (x + n)^2 + 20) → 
  n > 0 → 
  b = 4 * Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l883_88336


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l883_88377

/-- The number of people in the circular arrangement -/
def total_people : ℕ := 8

/-- The number of friends excluding Cara and Mark -/
def remaining_friends : ℕ := total_people - 2

/-- The number of different pairs Cara could be sitting between -/
def possible_pairs : ℕ := remaining_friends

theorem cara_seating_arrangements :
  possible_pairs = 6 :=
sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l883_88377


namespace NUMINAMATH_CALUDE_prob_four_suits_in_five_draws_l883_88383

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Type := Unit

/-- Represents the number of suits in a standard deck -/
def numSuits : ℕ := 4

/-- Represents the number of cards drawn -/
def numDraws : ℕ := 5

/-- Represents the probability of drawing a card from a particular suit -/
def probSuitDraw : ℚ := 1 / 4

/-- The probability of drawing 4 cards representing each of the 4 suits 
    when drawing 5 cards with replacement from a standard 52-card deck -/
theorem prob_four_suits_in_five_draws (deck : StandardDeck) : 
  (3 : ℚ) / 32 = probSuitDraw^3 * (1 - probSuitDraw) * (2 - probSuitDraw) * (3 - probSuitDraw) / 6 :=
sorry

end NUMINAMATH_CALUDE_prob_four_suits_in_five_draws_l883_88383


namespace NUMINAMATH_CALUDE_positive_solutions_conditions_l883_88322

theorem positive_solutions_conditions (a m x y z : ℝ) : 
  (x + y - z = 2 * a) →
  (x^2 + y^2 = z^2) →
  (m * (x + y) = x * y) →
  (x > 0 ∧ y > 0 ∧ z > 0) ↔ 
  (a / 2 * (2 + Real.sqrt 2) ≤ m ∧ m ≤ 2 * a ∧ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_positive_solutions_conditions_l883_88322


namespace NUMINAMATH_CALUDE_sum_18_probability_l883_88376

/-- The number of ways to distribute 10 points among 8 dice -/
def ways_to_distribute : ℕ := 19448

/-- The total number of possible outcomes when throwing 8 dice -/
def total_outcomes : ℕ := 6^8

/-- The probability of obtaining a sum of 18 when throwing 8 fair 6-sided dice -/
def probability_sum_18 : ℚ := ways_to_distribute / total_outcomes

theorem sum_18_probability :
  probability_sum_18 = 19448 / 6^8 :=
sorry

end NUMINAMATH_CALUDE_sum_18_probability_l883_88376


namespace NUMINAMATH_CALUDE_diagonals_concurrent_l883_88343

-- Define a regular 12-gon
def Regular12gon (P : Fin 12 → ℝ × ℝ) : Prop :=
  ∀ i j : Fin 12, dist (P i) (P ((i + 1) % 12)) = dist (P j) (P ((j + 1) % 12))

-- Define a diagonal in the 12-gon
def Diagonal (P : Fin 12 → ℝ × ℝ) (i j : Fin 12) : Set (ℝ × ℝ) :=
  {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • (P i) + t • (P j)}

-- Define concurrency of three lines
def Concurrent (L₁ L₂ L₃ : Set (ℝ × ℝ)) : Prop :=
  ∃ x : ℝ × ℝ, x ∈ L₁ ∧ x ∈ L₂ ∧ x ∈ L₃

-- Theorem statement
theorem diagonals_concurrent (P : Fin 12 → ℝ × ℝ) (h : Regular12gon P) :
  Concurrent (Diagonal P 0 8) (Diagonal P 11 3) (Diagonal P 1 10) :=
sorry

end NUMINAMATH_CALUDE_diagonals_concurrent_l883_88343


namespace NUMINAMATH_CALUDE_cyclist_round_time_l883_88301

/-- Given a rectangular park with length L and breadth B, prove that a cyclist
    traveling at 12 km/hr along the park's boundary will complete one round in 8 minutes
    when the length to breadth ratio is 1:3 and the area is 120,000 sq. m. -/
theorem cyclist_round_time (L B : ℝ) (h_ratio : B = 3 * L) (h_area : L * B = 120000) :
  (2 * L + 2 * B) / (12000 / 60) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_round_time_l883_88301


namespace NUMINAMATH_CALUDE_total_water_consumed_l883_88363

-- Define the conversion rate from quarts to ounces
def quart_to_ounce : ℚ := 32

-- Define the amount of water in the bottle (in quarts)
def bottle_water : ℚ := 3/2

-- Define the amount of water in the can (in ounces)
def can_water : ℚ := 12

-- Theorem statement
theorem total_water_consumed :
  bottle_water * quart_to_ounce + can_water = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_water_consumed_l883_88363


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l883_88369

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 1}

theorem union_of_A_and_B : A ∪ B = {x | -3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l883_88369


namespace NUMINAMATH_CALUDE_yellow_raisins_cups_l883_88379

theorem yellow_raisins_cups (total_raisins : Real) (black_raisins : Real) (yellow_raisins : Real) :
  total_raisins = 0.7 →
  black_raisins = 0.4 →
  total_raisins = yellow_raisins + black_raisins →
  yellow_raisins = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_raisins_cups_l883_88379


namespace NUMINAMATH_CALUDE_smallest_n_for_special_function_l883_88371

theorem smallest_n_for_special_function : ∃ (n : ℕ) (f : ℤ → Fin n),
  (∀ (A B : ℤ), |A - B| ∈ ({5, 7, 12} : Set ℤ) → f A ≠ f B) ∧
  (∀ (m : ℕ), m < n → ¬∃ (g : ℤ → Fin m), ∀ (A B : ℤ), |A - B| ∈ ({5, 7, 12} : Set ℤ) → g A ≠ g B) ∧
  n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_special_function_l883_88371


namespace NUMINAMATH_CALUDE_profit_to_cost_ratio_l883_88386

/-- Given an article with a sale price and cost price, this theorem proves
    that if the ratio of sale price to cost price is 6:2,
    then the ratio of profit to cost price is 2:1. -/
theorem profit_to_cost_ratio
  (sale_price cost_price : ℚ)
  (h : sale_price / cost_price = 6 / 2) :
  (sale_price - cost_price) / cost_price = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_profit_to_cost_ratio_l883_88386


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l883_88352

theorem inequality_solution_sets : ∃ (x : ℝ), 
  (x > 15 ∧ ¬(x - 7 < 2*x + 8)) ∨ (x - 7 < 2*x + 8 ∧ ¬(x > 15)) ∧
  (∀ y : ℝ, (5*y > 10 ↔ 3*y > 6)) ∧
  (∀ z : ℝ, (6*z - 9 < 3*z + 6 ↔ z < 5)) ∧
  (∀ w : ℝ, (w < -2 ↔ -14*w > 28)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l883_88352


namespace NUMINAMATH_CALUDE_supermarket_spending_l883_88309

theorem supermarket_spending (total : ℚ) : 
  (1/2 : ℚ) * total + (1/3 : ℚ) * total + (1/10 : ℚ) * total + 5 = total →
  total = 75 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l883_88309


namespace NUMINAMATH_CALUDE_movie_marathon_duration_is_9_hours_l883_88372

/-- The duration of Tim's movie marathon given the specified movie lengths --/
def movie_marathon_duration (first_movie : ℝ) (second_movie_percentage : ℝ) (third_movie_offset : ℝ) : ℝ :=
  let second_movie := first_movie * (1 + second_movie_percentage)
  let first_two := first_movie + second_movie
  let third_movie := first_two - third_movie_offset
  first_movie + second_movie + third_movie

/-- Theorem stating that Tim's movie marathon lasts 9 hours --/
theorem movie_marathon_duration_is_9_hours :
  movie_marathon_duration 2 0.5 1 = 9 :=
sorry

end NUMINAMATH_CALUDE_movie_marathon_duration_is_9_hours_l883_88372


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l883_88332

theorem trigonometric_expression_equality : 
  4 * Real.sin (60 * π / 180) - Real.sqrt 12 + (-3)^2 - (1 / (2 - Real.sqrt 3)) = 7 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l883_88332


namespace NUMINAMATH_CALUDE_divisible_by_2_3_5_less_than_300_l883_88394

theorem divisible_by_2_3_5_less_than_300 : 
  (Finset.filter (fun n : ℕ => n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) (Finset.range 300)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_2_3_5_less_than_300_l883_88394


namespace NUMINAMATH_CALUDE_min_value_problem_l883_88308

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z * (x + y + z) = 1) : 
  (x + y) * (y + z) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l883_88308


namespace NUMINAMATH_CALUDE_speed_difference_20_l883_88344

/-- The speed equation for a subway train -/
def speed (s : ℝ) : ℝ := s^2 + 2*s

/-- Theorem stating that the speed difference between 5 and 3 seconds is 20 km/h -/
theorem speed_difference_20 : speed 5 - speed 3 = 20 := by sorry

end NUMINAMATH_CALUDE_speed_difference_20_l883_88344


namespace NUMINAMATH_CALUDE_find_M_l883_88302

theorem find_M : ∃ M : ℕ, 995 + 997 + 999 + 1001 + 1003 = 5100 - M ∧ M = 104 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l883_88302


namespace NUMINAMATH_CALUDE_min_sum_of_six_l883_88334

def consecutive_numbers (start : ℕ) : List ℕ :=
  List.range 11 |>.map (λ i => start + i)

theorem min_sum_of_six (start : ℕ) :
  (consecutive_numbers start).length = 11 →
  start + (start + 10) = 90 →
  ∃ (subset : List ℕ), subset.length = 6 ∧ 
    subset.all (λ x => x ∈ consecutive_numbers start) ∧
    subset.sum = 90 ∧
    ∀ (other_subset : List ℕ), other_subset.length = 6 →
      other_subset.all (λ x => x ∈ consecutive_numbers start) →
      other_subset.sum ≥ 90 :=
by
  sorry

#check min_sum_of_six

end NUMINAMATH_CALUDE_min_sum_of_six_l883_88334


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l883_88389

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The number of diagonals in a hexagon is 9 -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l883_88389


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l883_88387

theorem no_positive_integer_solution :
  ¬ ∃ (a b c d : ℕ+), a + b + c + d - 3 = a * b + c * d := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l883_88387


namespace NUMINAMATH_CALUDE_notP_set_equals_interval_l883_88311

-- Define the proposition P
def P (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0

-- Define the set of x satisfying ¬P
def notP_set : Set ℝ := {x : ℝ | ¬(P x)}

-- Theorem stating that notP_set is equal to the closed interval [-1, 2]
theorem notP_set_equals_interval :
  notP_set = Set.Icc (-1 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_notP_set_equals_interval_l883_88311


namespace NUMINAMATH_CALUDE_hoseok_english_score_l883_88317

theorem hoseok_english_score 
  (korean_math_avg : ℝ) 
  (all_subjects_avg : ℝ) 
  (h1 : korean_math_avg = 88)
  (h2 : all_subjects_avg = 90) :
  ∃ (korean math english : ℝ),
    (korean + math) / 2 = korean_math_avg ∧
    (korean + math + english) / 3 = all_subjects_avg ∧
    english = 94 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_english_score_l883_88317


namespace NUMINAMATH_CALUDE_x_minus_y_value_l883_88366

theorem x_minus_y_value (x y : ℝ) (h1 : 3 = 0.2 * x) (h2 : 3 = 0.4 * y) : x - y = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l883_88366


namespace NUMINAMATH_CALUDE_base_twelve_equality_l883_88351

/-- Given a base b, this function converts a number in base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The proposition states that in base b, 35₍ᵦ₎² equals 1331₍ᵦ₎, and b equals 12 --/
theorem base_twelve_equality : ∃ b : Nat, 
  b > 1 ∧ 
  (toDecimal [3, 5] b)^2 = toDecimal [1, 3, 3, 1] b ∧ 
  b = 12 := by
  sorry

end NUMINAMATH_CALUDE_base_twelve_equality_l883_88351


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l883_88312

theorem simplify_trig_expression (x : ℝ) :
  (1 + Real.sin x + Real.cos x) / (1 - Real.sin x + Real.cos x) = Real.tan (π / 4 + x / 2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l883_88312


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l883_88348

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def binomialProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

/-- The statement to prove -/
theorem fair_coin_probability_difference :
  (binomialProbability 3 2) - (binomialProbability 3 3) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l883_88348


namespace NUMINAMATH_CALUDE_tree_heights_theorem_l883_88359

/-- Represents the heights of 5 trees -/
structure TreeHeights where
  h1 : ℤ
  h2 : ℤ
  h3 : ℤ
  h4 : ℤ
  h5 : ℤ

/-- The condition that each tree is either twice as tall or half as tall as the one to its right -/
def validHeights (h : TreeHeights) : Prop :=
  (h.h1 = 2 * h.h2 ∨ 2 * h.h1 = h.h2) ∧
  (h.h2 = 2 * h.h3 ∨ 2 * h.h2 = h.h3) ∧
  (h.h3 = 2 * h.h4 ∨ 2 * h.h3 = h.h4) ∧
  (h.h4 = 2 * h.h5 ∨ 2 * h.h4 = h.h5)

/-- The average height of the trees -/
def averageHeight (h : TreeHeights) : ℚ :=
  (h.h1 + h.h2 + h.h3 + h.h4 + h.h5) / 5

/-- The main theorem -/
theorem tree_heights_theorem (h : TreeHeights) 
  (h_valid : validHeights h) 
  (h_second : h.h2 = 11) : 
  averageHeight h = 121 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tree_heights_theorem_l883_88359


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_sum_power_l883_88378

theorem sqrt_abs_sum_zero_implies_sum_power (a b : ℝ) : 
  (Real.sqrt (a + 2) + |b - 1| = 0) → ((a + b)^2023 = -1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_sum_power_l883_88378


namespace NUMINAMATH_CALUDE_intersection_points_sum_l883_88346

/-- The quadratic function f(x) = (x+2)(x-4) -/
def f (x : ℝ) : ℝ := (x + 2) * (x - 4)

/-- The function g(x) = -f(x) -/
def g (x : ℝ) : ℝ := -f x

/-- The function h(x) = f(-x) -/
def h (x : ℝ) : ℝ := f (-x)

/-- The number of intersection points between y=f(x) and y=g(x) -/
def a : ℕ := 2

/-- The number of intersection points between y=f(x) and y=h(x) -/
def b : ℕ := 1

theorem intersection_points_sum : 10 * a + b = 21 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_sum_l883_88346


namespace NUMINAMATH_CALUDE_framed_photo_area_l883_88304

/-- The area of a framed rectangular photo -/
theorem framed_photo_area 
  (paper_width : ℝ) 
  (paper_length : ℝ) 
  (frame_width : ℝ) 
  (h1 : paper_width = 8) 
  (h2 : paper_length = 12) 
  (h3 : frame_width = 2) : 
  (paper_width + 2 * frame_width) * (paper_length + 2 * frame_width) = 192 :=
by sorry

end NUMINAMATH_CALUDE_framed_photo_area_l883_88304


namespace NUMINAMATH_CALUDE_onion_shelf_problem_l883_88316

/-- Given the initial conditions of onions on a shelf, prove the final number of onions. -/
theorem onion_shelf_problem (initial : ℕ) (sold : ℕ) (added : ℕ) (given_away : ℕ) : 
  initial = 98 → sold = 65 → added = 20 → given_away = 10 → 
  initial - sold + added - given_away = 43 := by
sorry

end NUMINAMATH_CALUDE_onion_shelf_problem_l883_88316


namespace NUMINAMATH_CALUDE_inequality_proof_l883_88353

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / ((a * b * c) ^ (1/3) * (1 + (a * b * c) ^ (1/3))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l883_88353


namespace NUMINAMATH_CALUDE_piggy_bank_pennies_l883_88325

theorem piggy_bank_pennies (initial_pennies : ℕ) : 
  (12 * (initial_pennies + 6) = 96) → initial_pennies = 2 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_pennies_l883_88325


namespace NUMINAMATH_CALUDE_pond_eyes_count_l883_88307

/-- The number of eyes for each frog -/
def frog_eyes : ℕ := 2

/-- The number of eyes for each crocodile -/
def crocodile_eyes : ℕ := 2

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- The number of crocodiles in the pond -/
def num_crocodiles : ℕ := 10

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := num_frogs * frog_eyes + num_crocodiles * crocodile_eyes

theorem pond_eyes_count : total_eyes = 60 := by
  sorry

end NUMINAMATH_CALUDE_pond_eyes_count_l883_88307


namespace NUMINAMATH_CALUDE_cos_sum_eleventh_l883_88326

theorem cos_sum_eleventh : 
  Real.cos (2 * Real.pi / 11) + Real.cos (6 * Real.pi / 11) + Real.cos (8 * Real.pi / 11) = 
    (-1 + Real.sqrt (-11)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_eleventh_l883_88326


namespace NUMINAMATH_CALUDE_income_calculation_l883_88324

-- Define the total income
def total_income : ℝ := sorry

-- Define the percentage given to children
def children_percentage : ℝ := 0.2 * 3

-- Define the percentage given to wife
def wife_percentage : ℝ := 0.3

-- Define the remaining percentage after giving to children and wife
def remaining_percentage : ℝ := 1 - children_percentage - wife_percentage

-- Define the percentage donated to orphan house
def orphan_house_percentage : ℝ := 0.05

-- Define the final amount left
def final_amount : ℝ := 40000

-- Theorem to prove
theorem income_calculation : 
  ∃ (total_income : ℝ),
    total_income > 0 ∧
    final_amount = total_income * remaining_percentage * (1 - orphan_house_percentage) ∧
    (total_income ≥ 421052) ∧ (total_income ≤ 421053) :=
by sorry

end NUMINAMATH_CALUDE_income_calculation_l883_88324


namespace NUMINAMATH_CALUDE_english_only_enrollment_l883_88340

/-- The number of students enrolled only in English -/
def students_only_english (total : ℕ) (both_eng_ger : ℕ) (german : ℕ) (spanish : ℕ) : ℕ :=
  total - (german + spanish - both_eng_ger)

theorem english_only_enrollment :
  let total := 75
  let both_eng_ger := 18
  let german := 32
  let spanish := 25
  students_only_english total both_eng_ger german spanish = 18 := by
  sorry

#eval students_only_english 75 18 32 25

end NUMINAMATH_CALUDE_english_only_enrollment_l883_88340


namespace NUMINAMATH_CALUDE_three_digit_subtraction_l883_88350

/-- Represents a three-digit number with digits a, b, c -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h_a : a < 10
  h_b : b < 10
  h_c : c < 10

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

theorem three_digit_subtraction
  (n₁ n₂ : ThreeDigitNumber)
  (h_reverse : n₂.a = n₁.c ∧ n₂.b = n₁.b ∧ n₂.c = n₁.a)
  (h_result_units : (n₁.toNat - n₂.toNat) % 10 = 2)
  (h_result_tens : ((n₁.toNat - n₂.toNat) / 10) % 10 = 9)
  (h_borrow : n₁.c < n₂.c) :
  n₁.a = 9 ∧ n₁.b = 9 ∧ n₁.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_l883_88350


namespace NUMINAMATH_CALUDE_drums_per_day_l883_88310

/-- Given that 2916 drums of grapes are filled in 9 days, 
    prove that 324 drums of grapes are filled per day. -/
theorem drums_per_day (total_drums : ℕ) (total_days : ℕ) 
  (h1 : total_drums = 2916) (h2 : total_days = 9) :
  total_drums / total_days = 324 := by
  sorry

end NUMINAMATH_CALUDE_drums_per_day_l883_88310
