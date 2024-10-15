import Mathlib

namespace NUMINAMATH_CALUDE_system_solvability_l2498_249884

-- Define the system of equations
def system (x y p : ℝ) : Prop :=
  (x - p)^2 = 16 * (y - 3 + p) ∧
  y^2 + ((x - 3) / (|x| - 3))^2 = 1 ∧
  |x| ≠ 3

-- Define the set of valid p values
def valid_p_set : Set ℝ :=
  {p | (3 < p ∧ p ≤ 4) ∨ (12 ≤ p ∧ p < 19) ∨ (p > 19)}

-- Theorem statement
theorem system_solvability (p : ℝ) :
  (∃ x y, system x y p) ↔ p ∈ valid_p_set :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l2498_249884


namespace NUMINAMATH_CALUDE_first_day_next_year_monday_l2498_249879

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : Nat
  is_leap : Bool

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Theorem: If a non-leap year has 53 Sundays, then the first day of the following year is a Monday -/
theorem first_day_next_year_monday 
  (year : Year) 
  (h1 : year.is_leap = false) 
  (h2 : ∃ (sundays : Nat), sundays = 53) : 
  nextDay DayOfWeek.Sunday = DayOfWeek.Monday := by
  sorry

#check first_day_next_year_monday

end NUMINAMATH_CALUDE_first_day_next_year_monday_l2498_249879


namespace NUMINAMATH_CALUDE_circular_competition_rounds_l2498_249873

theorem circular_competition_rounds (m : ℕ) (h : m ≥ 17) :
  ∃ (n : ℕ), n = m - 1 ∧
  (∀ (schedule : ℕ → Fin (2*m) → Fin (2*m) → Prop),
    (∀ (i : Fin (2*m)), ∀ (j : Fin (2*m)), i ≠ j → ∃ (k : Fin (2*m - 1)), schedule k i j) →
    (∀ (k : Fin (2*m - 1)), ∀ (i : Fin (2*m)), ∃! (j : Fin (2*m)), i ≠ j ∧ schedule k i j) →
    (∀ (a b c d : Fin (2*m)), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d →
      (∀ (k : Fin n), ¬(schedule k a b ∨ schedule k a c ∨ schedule k a d ∨ schedule k b c ∨ schedule k b d ∨ schedule k c d)) ∨
      (∃ (k₁ k₂ : Fin n), k₁ ≠ k₂ ∧
        ((schedule k₁ a b ∧ schedule k₂ c d) ∨
         (schedule k₁ a c ∧ schedule k₂ b d) ∨
         (schedule k₁ a d ∧ schedule k₂ b c))))) ∧
  (∀ (n' : ℕ), n' < n →
    ∃ (schedule : ℕ → Fin (2*m) → Fin (2*m) → Prop),
      (∀ (i : Fin (2*m)), ∀ (j : Fin (2*m)), i ≠ j → ∃ (k : Fin (2*m - 1)), schedule k i j) ∧
      (∀ (k : Fin (2*m - 1)), ∀ (i : Fin (2*m)), ∃! (j : Fin (2*m)), i ≠ j ∧ schedule k i j) ∧
      (∃ (a b c d : Fin (2*m)), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
        (∀ (k : Fin n'), ¬(schedule k a b ∨ schedule k a c ∨ schedule k a d ∨ schedule k b c ∨ schedule k b d ∨ schedule k c d)) ∧
        ¬(∃ (k₁ k₂ : Fin n'), k₁ ≠ k₂ ∧
          ((schedule k₁ a b ∧ schedule k₂ c d) ∨
           (schedule k₁ a c ∧ schedule k₂ b d) ∨
           (schedule k₁ a d ∧ schedule k₂ b c))))) :=
by
  sorry


end NUMINAMATH_CALUDE_circular_competition_rounds_l2498_249873


namespace NUMINAMATH_CALUDE_max_ac_without_racing_stripes_l2498_249878

/-- Represents the properties of a car group -/
structure CarGroup where
  total : ℕ
  without_ac : ℕ
  with_racing_stripes : ℕ
  (total_valid : total = 100)
  (without_ac_valid : without_ac = 37)
  (racing_stripes_valid : with_racing_stripes ≥ 41)

/-- Theorem: The greatest number of cars that could have air conditioning but not racing stripes -/
theorem max_ac_without_racing_stripes (group : CarGroup) : 
  (group.total - group.without_ac) - group.with_racing_stripes ≤ 22 :=
sorry

end NUMINAMATH_CALUDE_max_ac_without_racing_stripes_l2498_249878


namespace NUMINAMATH_CALUDE_bear_cubs_count_l2498_249828

/-- Represents the bear's hunting scenario -/
structure BearHunt where
  totalMeat : ℕ  -- Total meat needed per week
  cubMeat : ℕ    -- Meat needed per cub per week
  rabbitWeight : ℕ -- Weight of each rabbit
  dailyCatch : ℕ  -- Number of rabbits caught daily

/-- Calculates the number of cubs based on the hunting scenario -/
def numCubs (hunt : BearHunt) : ℕ :=
  let weeklyHunt := hunt.dailyCatch * hunt.rabbitWeight * 7
  (weeklyHunt - hunt.totalMeat) / hunt.cubMeat

/-- Theorem stating that the number of cubs is 4 given the specific hunting scenario -/
theorem bear_cubs_count (hunt : BearHunt) 
  (h1 : hunt.totalMeat = 210)
  (h2 : hunt.cubMeat = 35)
  (h3 : hunt.rabbitWeight = 5)
  (h4 : hunt.dailyCatch = 10) :
  numCubs hunt = 4 := by
  sorry

end NUMINAMATH_CALUDE_bear_cubs_count_l2498_249828


namespace NUMINAMATH_CALUDE_two_integers_sum_l2498_249857

theorem two_integers_sum (a b : ℕ) : 
  a > 0 → b > 0 → 
  a * b + a + b = 255 → 
  (Odd a ∨ Odd b) → 
  a < 30 → b < 30 → 
  a + b = 30 := by sorry

end NUMINAMATH_CALUDE_two_integers_sum_l2498_249857


namespace NUMINAMATH_CALUDE_factorization_ax2_minus_4ay2_l2498_249834

theorem factorization_ax2_minus_4ay2 (a x y : ℝ) :
  a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax2_minus_4ay2_l2498_249834


namespace NUMINAMATH_CALUDE_noah_age_in_ten_years_l2498_249885

/-- Calculates Noah's age after a given number of years -/
def noah_age_after (joe_age : ℕ) (years_passed : ℕ) : ℕ :=
  2 * joe_age + years_passed

/-- Proves that Noah will be 22 years old after 10 years, given the initial conditions -/
theorem noah_age_in_ten_years (joe_age : ℕ) (h : joe_age = 6) :
  noah_age_after joe_age 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_noah_age_in_ten_years_l2498_249885


namespace NUMINAMATH_CALUDE_cats_after_sale_l2498_249855

/-- The number of cats left after a sale in a pet store -/
theorem cats_after_sale (siamese_cats house_cats sold_cats : ℕ) :
  siamese_cats = 12 →
  house_cats = 20 →
  sold_cats = 20 →
  siamese_cats + house_cats - sold_cats = 12 := by
  sorry

end NUMINAMATH_CALUDE_cats_after_sale_l2498_249855


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l2498_249888

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 3 = 1

-- Define the focus of the hyperbola
def focus (F : ℝ × ℝ) : Prop := 
  F.1^2 - F.2^2 = 6 ∧ F.2 = 0

-- Define an asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = x ∨ y = -x

-- Theorem statement
theorem distance_focus_to_asymptote :
  ∀ (F : ℝ × ℝ) (x y : ℝ),
  focus F → hyperbola x y → asymptote x y →
  ∃ (d : ℝ), d = Real.sqrt 3 ∧ 
  d = Real.sqrt ((F.1 - x)^2 + (F.2 - y)^2) := by sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l2498_249888


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l2498_249829

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l2498_249829


namespace NUMINAMATH_CALUDE_polygon_perimeter_l2498_249815

/-- The perimeter of a polygon formed by cutting a right triangle from a rectangle --/
theorem polygon_perimeter (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c < 10) : 
  2 * (10 + (10 - b)) - a = 29 :=
by sorry

end NUMINAMATH_CALUDE_polygon_perimeter_l2498_249815


namespace NUMINAMATH_CALUDE_min_regions_for_12_intersections_l2498_249808

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A set of circles in a plane --/
def CircleSet := Set Circle

/-- The number of intersection points between circles in a set --/
def intersectionPoints (s : CircleSet) : ℕ := sorry

/-- The number of regions into which a set of circles divides the plane --/
def regions (s : CircleSet) : ℕ := sorry

/-- The theorem stating the minimum number of regions --/
theorem min_regions_for_12_intersections (s : CircleSet) :
  intersectionPoints s = 12 → regions s ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_min_regions_for_12_intersections_l2498_249808


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2498_249820

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop :=
  x^2 + 2*x*y + y^2 + 3*x + y = 0

-- Define the axis of symmetry
def axis_of_symmetry (x y : ℝ) : Prop :=
  x + y + 1 = 0

-- Theorem statement
theorem parabola_axis_of_symmetry :
  ∀ (x y : ℝ), parabola_eq x y → axis_of_symmetry x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2498_249820


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l2498_249832

theorem arithmetic_square_root_of_16 : ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l2498_249832


namespace NUMINAMATH_CALUDE_pages_left_to_write_l2498_249801

-- Define the total number of pages for the book
def total_pages : ℕ := 500

-- Define the number of pages written on each day
def day1_pages : ℕ := 25
def day2_pages : ℕ := 2 * day1_pages
def day3_pages : ℕ := 2 * day2_pages
def day4_pages : ℕ := 10

-- Define the total number of pages written so far
def pages_written : ℕ := day1_pages + day2_pages + day3_pages + day4_pages

-- Define the number of pages left to write
def pages_left : ℕ := total_pages - pages_written

-- Theorem stating that the number of pages left to write is 315
theorem pages_left_to_write : pages_left = 315 := by sorry

end NUMINAMATH_CALUDE_pages_left_to_write_l2498_249801


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_8_l2498_249807

def binary_number : ℕ := 110110111010

-- Define a function to get the last three bits of a binary number
def last_three_bits (n : ℕ) : ℕ := n % 8

-- Theorem statement
theorem remainder_of_binary_div_8 :
  binary_number % 8 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_8_l2498_249807


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l2498_249893

/-- A function f(x) = kx - ln x is monotonically increasing on (1/2, +∞) if and only if k ≥ 2 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > (1/2 : ℝ), Monotone (λ x => k * x - Real.log x)) ↔ k ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l2498_249893


namespace NUMINAMATH_CALUDE_symmetry_probability_one_third_l2498_249851

/-- A square grid with n^2 points -/
structure SquareGrid (n : ℕ) where
  points : Fin n → Fin n → Bool

/-- The center point of a square grid -/
def centerPoint (n : ℕ) : Fin n × Fin n :=
  (⟨n / 2, sorry⟩, ⟨n / 2, sorry⟩)

/-- A line of symmetry for a square grid -/
def isSymmetryLine (n : ℕ) (grid : SquareGrid n) (p q : Fin n × Fin n) : Prop :=
  sorry

/-- The number of symmetry lines through the center point -/
def numSymmetryLines (n : ℕ) (grid : SquareGrid n) : ℕ :=
  sorry

theorem symmetry_probability_one_third (grid : SquareGrid 11) :
  let center := centerPoint 11
  let totalPoints := 121
  let nonCenterPoints := totalPoints - 1
  let symmetryLines := numSymmetryLines 11 grid
  (symmetryLines : ℚ) / nonCenterPoints = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_symmetry_probability_one_third_l2498_249851


namespace NUMINAMATH_CALUDE_phantom_needs_126_more_l2498_249846

/-- The amount of additional money Phantom needs to buy printer inks -/
def additional_money_needed (initial_money : ℕ) 
  (black_price red_price yellow_price blue_price : ℕ) 
  (black_quantity red_quantity yellow_quantity blue_quantity : ℕ) : ℕ :=
  let total_cost := black_price * black_quantity + 
                    red_price * red_quantity + 
                    yellow_price * yellow_quantity + 
                    blue_price * blue_quantity
  total_cost - initial_money

/-- Theorem stating that Phantom needs $126 more to buy the printer inks -/
theorem phantom_needs_126_more : 
  additional_money_needed 50 12 16 14 17 3 4 3 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_phantom_needs_126_more_l2498_249846


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2498_249842

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    (n ≥ 100 ∧ n < 1000) ∧ 
    (70 * n) % 350 = 210 ∧
    (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (70 * m) % 350 = 210 → n ≤ m) ∧
    n = 103 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2498_249842


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_40_60_1_7_l2498_249882

/-- Sum of an arithmetic series -/
def arithmetic_series_sum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_40_60_1_7 :
  arithmetic_series_sum 40 60 (1/7) = 7050 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_40_60_1_7_l2498_249882


namespace NUMINAMATH_CALUDE_solution_set_correct_l2498_249861

/-- The set of all solutions to the equation x² + y² = 3 · 2016ᶻ + 77 where x, y, and z are natural numbers -/
def SolutionSet : Set (ℕ × ℕ × ℕ) :=
  {(77, 14, 1), (14, 77, 1), (70, 35, 1), (35, 70, 1), (8, 4, 0), (4, 8, 0)}

/-- Predicate that checks if a triplet (x, y, z) satisfies the equation -/
def SatisfiesEquation (t : ℕ × ℕ × ℕ) : Prop :=
  let (x, y, z) := t
  x^2 + y^2 = 3 * 2016^z + 77

theorem solution_set_correct :
  ∀ t : ℕ × ℕ × ℕ, SatisfiesEquation t ↔ t ∈ SolutionSet := by
  sorry

end NUMINAMATH_CALUDE_solution_set_correct_l2498_249861


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l2498_249833

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 1 →
  (∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) →
  a 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l2498_249833


namespace NUMINAMATH_CALUDE_no_integers_product_zeros_l2498_249803

theorem no_integers_product_zeros : 
  ¬∃ (x y : ℤ), 
    (x % 10 ≠ 0) ∧ 
    (y % 10 ≠ 0) ∧ 
    (x * y = 100000) := by
  sorry

end NUMINAMATH_CALUDE_no_integers_product_zeros_l2498_249803


namespace NUMINAMATH_CALUDE_selection_schemes_count_l2498_249804

def num_students : ℕ := 6
def num_tasks : ℕ := 4
def num_restricted_students : ℕ := 2

theorem selection_schemes_count :
  (num_students.factorial / (num_students - num_tasks).factorial) -
  (num_restricted_students * (num_students - 1).factorial / (num_students - num_tasks).factorial) = 240 :=
by sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l2498_249804


namespace NUMINAMATH_CALUDE_factorization_equality_l2498_249875

theorem factorization_equality (a b : ℝ) : a * b^2 - 4 * a = a * (b + 2) * (b - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2498_249875


namespace NUMINAMATH_CALUDE_min_value_xyz_product_min_value_achieved_l2498_249824

theorem min_value_xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x + 2 * y) * (y + 2 * z) * (x * z + 1) ≥ 16 :=
sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 2 * y₀) * (y₀ + 2 * z₀) * (x₀ * z₀ + 1) = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_xyz_product_min_value_achieved_l2498_249824


namespace NUMINAMATH_CALUDE_two_digit_divisor_with_remainder_l2498_249830

theorem two_digit_divisor_with_remainder (x y : ℕ) : ∃! n : ℕ, 
  (0 < x ∧ x ≤ 9) ∧ 
  (0 ≤ y ∧ y ≤ 9) ∧
  (n = 10 * x + y) ∧
  (∃ q : ℕ, 491 = n * q + 59) ∧
  (n = 72) := by
sorry

end NUMINAMATH_CALUDE_two_digit_divisor_with_remainder_l2498_249830


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2498_249870

theorem hyperbola_equation (a c : ℝ) (h1 : a = 5) (h2 : c = 7) :
  ∃ (x y : ℝ), (x^2 / 25 - y^2 / 24 = 1) ∨ (y^2 / 25 - x^2 / 24 = 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2498_249870


namespace NUMINAMATH_CALUDE_pyramid_section_is_trapezoid_l2498_249866

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  d : ℝ

/-- Represents a pyramid -/
structure Pyramid where
  apex : Point3D
  base : List Point3D

/-- Represents a parallelogram -/
structure Parallelogram where
  vertices : List Point3D

/-- Represents a trapezoid -/
structure Trapezoid where
  vertices : List Point3D

def is_parallelogram (p : Parallelogram) : Prop := sorry

def is_point_on_edge (p : Point3D) (e1 e2 : Point3D) : Prop := sorry

def intersection_is_trapezoid (plane : Plane) (pyr : Pyramid) : Prop := sorry

theorem pyramid_section_is_trapezoid 
  (S A B C D M : Point3D) 
  (base : Parallelogram) 
  (pyr : Pyramid) 
  (plane : Plane) :
  is_parallelogram base →
  pyr.apex = S →
  pyr.base = base.vertices →
  is_point_on_edge M S C →
  plane.normal = sorry → -- Define the normal vector of plane ABM
  plane.d = sorry → -- Define the d value for plane ABM
  intersection_is_trapezoid plane pyr := by
  sorry

#check pyramid_section_is_trapezoid

end NUMINAMATH_CALUDE_pyramid_section_is_trapezoid_l2498_249866


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2498_249876

/-- Given a geometric sequence {a_n} with sum of first n terms S_n, 
    if S_1, S_3, and S_2 form an arithmetic sequence, and a_1 - a_3 = 3, 
    then the common ratio q = -1/2 and a_1 = 4 -/
theorem geometric_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * (a 2 / a 1))
  (h_sum : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1)))
  (h_arith : S 3 - S 2 = S 2 - S 1)
  (h_diff : a 1 - a 3 = 3) :
  a 2 / a 1 = -1/2 ∧ a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2498_249876


namespace NUMINAMATH_CALUDE_triangle_properties_l2498_249854

open Real

theorem triangle_properties 
  (a b c A B C : ℝ) 
  (h1 : sin A / (sin B + sin C) = 1 - (a - b) / (a - c))
  (h2 : b = Real.sqrt 3)
  (h3 : 0 < A ∧ A < 2 * π / 3) :
  ∃ (area : ℝ) (range_lower range_upper : ℝ),
    (∀ perimeter, perimeter ≤ 3 * Real.sqrt 3 → 
      area * 2 ≤ perimeter * Real.sqrt (perimeter * (perimeter - 2*a) * (perimeter - 2*b) * (perimeter - 2*c)) / 4) ∧
    (area = 3 * Real.sqrt 3 / 4) ∧
    (∀ m_dot_n : ℝ, 
      (∃ A', 0 < A' ∧ A' < 2 * π / 3 ∧ 
        m_dot_n = 6 * sin A' * cos B + cos (2 * A')) → 
      (range_lower < m_dot_n ∧ m_dot_n ≤ range_upper)) ∧
    (range_lower = 1 ∧ range_upper = 17/8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2498_249854


namespace NUMINAMATH_CALUDE_members_playing_both_l2498_249831

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  totalMembers : ℕ
  badmintonPlayers : ℕ
  tennisPlayers : ℕ
  neitherPlayers : ℕ

/-- Calculates the number of members playing both badminton and tennis -/
def playBoth (club : SportsClub) : ℕ :=
  club.badmintonPlayers + club.tennisPlayers - (club.totalMembers - club.neitherPlayers)

/-- Theorem stating the number of members playing both sports in the given scenario -/
theorem members_playing_both (club : SportsClub)
    (h1 : club.totalMembers = 50)
    (h2 : club.badmintonPlayers = 25)
    (h3 : club.tennisPlayers = 32)
    (h4 : club.neitherPlayers = 5) :
    playBoth club = 12 := by
  sorry


end NUMINAMATH_CALUDE_members_playing_both_l2498_249831


namespace NUMINAMATH_CALUDE_large_bus_most_cost_effective_l2498_249822

/-- Represents the transportation options for the field trip --/
inductive TransportOption
  | Van
  | Minibus
  | LargeBus

/-- Calculates the number of vehicles needed for a given option --/
def vehiclesNeeded (option : TransportOption) : ℕ :=
  match option with
  | .Van => 6
  | .Minibus => 3
  | .LargeBus => 1

/-- Calculates the total cost for a given option --/
def totalCost (option : TransportOption) : ℕ :=
  match option with
  | .Van => 50 * vehiclesNeeded .Van
  | .Minibus => 100 * vehiclesNeeded .Minibus
  | .LargeBus => 250

/-- States that the large bus is the most cost-effective option --/
theorem large_bus_most_cost_effective :
  ∀ option : TransportOption, totalCost .LargeBus ≤ totalCost option :=
by sorry

end NUMINAMATH_CALUDE_large_bus_most_cost_effective_l2498_249822


namespace NUMINAMATH_CALUDE_non_congruent_triangles_count_l2498_249812

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a set of points -/
def PointSet : Type := List Point

/-- The set of nine points as described in the problem -/
def ninePoints : PointSet :=
  [
    {x := 0, y := 0}, {x := 1, y := 0}, {x := 2, y := 0},
    {x := 0, y := 0}, {x := 1, y := 1}, {x := 2, y := 2},
    {x := 0, y := 0}, {x := 0.5, y := 1}, {x := 1, y := 2}
  ]

/-- Checks if three points form a non-congruent triangle with respect to a set of triangles -/
def isNonCongruentTriangle (p1 p2 p3 : Point) (triangles : List (Point × Point × Point)) : Bool :=
  sorry

/-- Counts the number of non-congruent triangles that can be formed from a set of points -/
def countNonCongruentTriangles (points : PointSet) : Nat :=
  sorry

/-- The main theorem stating that the number of non-congruent triangles is 5 -/
theorem non_congruent_triangles_count :
  countNonCongruentTriangles ninePoints = 5 :=
sorry

end NUMINAMATH_CALUDE_non_congruent_triangles_count_l2498_249812


namespace NUMINAMATH_CALUDE_combination_equality_l2498_249868

theorem combination_equality (n : ℕ) : 
  (Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l2498_249868


namespace NUMINAMATH_CALUDE_perfect_square_count_l2498_249826

theorem perfect_square_count : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ n ≤ 2000 ∧ ∃ k : ℕ, 10 * n = k^2) ∧ 
  (∀ n : ℕ, n > 0 ∧ n ≤ 2000 ∧ (∃ k : ℕ, 10 * n = k^2) → n ∈ S) ∧
  Finset.card S = 14 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_count_l2498_249826


namespace NUMINAMATH_CALUDE_gcd_360_504_l2498_249865

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_504_l2498_249865


namespace NUMINAMATH_CALUDE_flagstaff_shadow_length_l2498_249864

/-- Given a flagstaff and a building casting shadows under similar conditions,
    prove that the length of the shadow cast by the flagstaff is 40.1 m. -/
theorem flagstaff_shadow_length 
  (flagstaff_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagstaff_height = 17.5)
  (h2 : building_height = 12.5)
  (h3 : building_shadow = 28.75) :
  flagstaff_height / (flagstaff_height * building_shadow / building_height) = 17.5 / 40.1 :=
by sorry

end NUMINAMATH_CALUDE_flagstaff_shadow_length_l2498_249864


namespace NUMINAMATH_CALUDE_prob_of_three_l2498_249894

/-- The decimal representation of 8/13 -/
def decimal_rep : ℚ := 8 / 13

/-- The length of the repeating block in the decimal representation -/
def block_length : ℕ := 6

/-- The count of digit 3 in one repeating block -/
def count_of_threes : ℕ := 1

/-- The probability of randomly selecting the digit 3 from the decimal representation of 8/13 -/
theorem prob_of_three (decimal_rep : ℚ) (block_length : ℕ) (count_of_threes : ℕ) :
  decimal_rep = 8 / 13 →
  block_length = 6 →
  count_of_threes = 1 →
  (count_of_threes : ℚ) / (block_length : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_of_three_l2498_249894


namespace NUMINAMATH_CALUDE_total_spent_is_135_l2498_249823

/-- The amount Jen spent on pastries -/
def jen_spent : ℝ := sorry

/-- The amount Lisa spent on pastries -/
def lisa_spent : ℝ := sorry

/-- For every dollar Jen spent, Lisa spent 20 cents less -/
axiom lisa_spent_relation : lisa_spent = 0.8 * jen_spent

/-- Jen spent $15 more than Lisa -/
axiom jen_spent_more : jen_spent = lisa_spent + 15

/-- The total amount spent on pastries -/
def total_spent : ℝ := jen_spent + lisa_spent

/-- Theorem stating that the total amount spent is $135 -/
theorem total_spent_is_135 : total_spent = 135 := by sorry

end NUMINAMATH_CALUDE_total_spent_is_135_l2498_249823


namespace NUMINAMATH_CALUDE_cookie_days_count_l2498_249850

-- Define the total number of school days
def total_days : ℕ := 5

-- Define the number of days with peanut butter sandwiches
def peanut_butter_days : ℕ := 2

-- Define the number of days with ham sandwiches
def ham_days : ℕ := 3

-- Define the number of days with cake
def cake_days : ℕ := 1

-- Define the probability of ham sandwich and cake on the same day
def ham_cake_prob : ℚ := 12 / 100

-- Theorem to prove
theorem cookie_days_count : 
  total_days - cake_days - peanut_butter_days = 2 :=
sorry

end NUMINAMATH_CALUDE_cookie_days_count_l2498_249850


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l2498_249859

/-- The number of diagonals in a polygon with N sides -/
def num_diagonals (N : ℕ) : ℕ := N * (N - 3) / 2

/-- A regular hexagon has 9 diagonals -/
theorem hexagon_diagonals :
  num_diagonals 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l2498_249859


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l2498_249848

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_3rd : a 3 = 10)
  (h_6th : a 6 = 20) :
  a 12 = 40 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l2498_249848


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2498_249852

theorem sufficient_not_necessary (p q : Prop) : 
  (¬(p ∨ q) → ¬p) ∧ ¬(∀ p q, ¬p → ¬(p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2498_249852


namespace NUMINAMATH_CALUDE_triangle_side_equations_l2498_249877

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The equation of a line given two points -/
def lineEquation (p1 p2 : Point) : Line :=
  let a := p2.y - p1.y
  let b := p1.x - p2.x
  let c := p2.x * p1.y - p1.x * p2.y
  { a := a, b := b, c := c }

theorem triangle_side_equations (A B C : Point)
  (hA : A = { x := -5, y := 0 })
  (hB : B = { x := 3, y := -3 })
  (hC : C = { x := 0, y := 2 }) :
  let AB := lineEquation A B
  let AC := lineEquation A C
  let BC := lineEquation B C
  AB = { a := 3, b := 8, c := 15 } ∧
  AC = { a := 2, b := -5, c := 10 } ∧
  BC = { a := 5, b := 3, c := -6 } :=
sorry

end NUMINAMATH_CALUDE_triangle_side_equations_l2498_249877


namespace NUMINAMATH_CALUDE_dot_product_AO_AB_l2498_249891

/-- The circle O with equation x^2 + y^2 = 4 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- The theorem statement -/
theorem dot_product_AO_AB (O A B : ℝ × ℝ) :
  A ∈ Circle → B ∈ Circle →
  ‖(A - O) + (B - O)‖ = ‖(A - O) - (B - O)‖ →
  (A - O) • (A - B) = 4 := by
sorry

end NUMINAMATH_CALUDE_dot_product_AO_AB_l2498_249891


namespace NUMINAMATH_CALUDE_distance_A_P_main_theorem_l2498_249853

/-- A rectangle with two equilateral triangles positioned on its sides -/
structure TrianglesOnRectangle where
  /-- The length of side YC of rectangle YQZC -/
  yc : ℝ
  /-- The length of side CZ of rectangle YQZC -/
  cz : ℝ
  /-- The side length of equilateral triangles ABC and PQR -/
  triangle_side : ℝ
  /-- Assumption that YC = 8 -/
  yc_eq : yc = 8
  /-- Assumption that CZ = 15 -/
  cz_eq : cz = 15
  /-- Assumption that the side length of triangles is 9 -/
  triangle_side_eq : triangle_side = 9

/-- The distance between points A and P is 10 -/
theorem distance_A_P (t : TrianglesOnRectangle) : ℝ :=
  10

#check distance_A_P

/-- The main theorem stating that the distance between A and P is 10 -/
theorem main_theorem (t : TrianglesOnRectangle) : distance_A_P t = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_P_main_theorem_l2498_249853


namespace NUMINAMATH_CALUDE_green_peaches_count_l2498_249862

/-- Given a basket of peaches, prove that the number of green peaches is 3 -/
theorem green_peaches_count (total : ℕ) (red : ℕ) (h1 : total = 16) (h2 : red = 13) :
  total - red = 3 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2498_249862


namespace NUMINAMATH_CALUDE_even_quadratic_implies_m_zero_l2498_249849

/-- A quadratic function f(x) = (m-1)x^2 - 2mx + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

/-- Definition of an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_quadratic_implies_m_zero (m : ℝ) :
  is_even (f m) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_implies_m_zero_l2498_249849


namespace NUMINAMATH_CALUDE_correct_division_l2498_249819

theorem correct_division (dividend : ℕ) (wrong_divisor correct_divisor wrong_quotient : ℕ) 
  (h1 : wrong_divisor = 87)
  (h2 : correct_divisor = 36)
  (h3 : wrong_quotient = 24)
  (h4 : dividend = wrong_divisor * wrong_quotient) :
  dividend / correct_divisor = 58 := by
sorry

end NUMINAMATH_CALUDE_correct_division_l2498_249819


namespace NUMINAMATH_CALUDE_unique_solution_l2498_249818

theorem unique_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * Real.sqrt b - c = a) ∧ 
  (b * Real.sqrt c - a = b) ∧ 
  (c * Real.sqrt a - b = c) →
  a = 4 ∧ b = 4 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2498_249818


namespace NUMINAMATH_CALUDE_solve_for_q_l2498_249895

theorem solve_for_q (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l2498_249895


namespace NUMINAMATH_CALUDE_floor_power_equality_l2498_249841

theorem floor_power_equality (a b : ℝ) (h : a > 0) (h' : b > 0)
  (h_infinite : ∃ᶠ k : ℕ in atTop, ⌊a^k⌋ + ⌊b^k⌋ = ⌊a⌋^k + ⌊b⌋^k) :
  ⌊a^2014⌋ + ⌊b^2014⌋ = ⌊a⌋^2014 + ⌊b⌋^2014 := by
sorry

end NUMINAMATH_CALUDE_floor_power_equality_l2498_249841


namespace NUMINAMATH_CALUDE_cranberry_juice_unit_cost_l2498_249887

/-- The unit cost of cranberry juice in cents per ounce -/
def unit_cost (total_cost : ℚ) (volume : ℚ) : ℚ :=
  total_cost / volume

/-- Theorem stating that the unit cost of cranberry juice is 7 cents per ounce -/
theorem cranberry_juice_unit_cost :
  let total_cost : ℚ := 84
  let volume : ℚ := 12
  unit_cost total_cost volume = 7 := by sorry

end NUMINAMATH_CALUDE_cranberry_juice_unit_cost_l2498_249887


namespace NUMINAMATH_CALUDE_price_increase_percentage_l2498_249886

theorem price_increase_percentage (old_price new_price : ℝ) 
  (h1 : old_price = 300)
  (h2 : new_price = 360) :
  (new_price - old_price) / old_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l2498_249886


namespace NUMINAMATH_CALUDE_rectangle_breadth_unchanged_l2498_249844

theorem rectangle_breadth_unchanged 
  (L B : ℝ) 
  (h1 : L > 0) 
  (h2 : B > 0) 
  (new_L : ℝ) 
  (h3 : new_L = L / 2) 
  (new_A : ℝ) 
  (h4 : new_A = L * B / 2) :
  ∃ (new_B : ℝ), new_A = new_L * new_B ∧ new_B = B := by
sorry

end NUMINAMATH_CALUDE_rectangle_breadth_unchanged_l2498_249844


namespace NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l2498_249825

-- Define the function f(x) = 3x - x^3
def f (x : ℝ) : ℝ := 3*x - x^3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 - 3*x^2

-- Theorem for the tangent line at x = 2
theorem tangent_line_at_2 :
  ∃ (m b : ℝ), m = -9 ∧ b = 18 ∧
  ∀ x, f x = m * (x - 2) + f 2 := by sorry

-- Theorem for monotonicity intervals
theorem monotonicity_intervals :
  (∀ x < -1, (f' x < 0)) ∧
  (∀ x ∈ Set.Ioo (-1) 1, (f' x > 0)) ∧
  (∀ x > 1, (f' x < 0)) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l2498_249825


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2498_249856

/-- The parabola y² = 4x in the cartesian plane -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- A line in the cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The intersection points of a line with the parabola -/
def intersection (l : Line) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ Parabola ∧ p.1 = l.slope * p.2 + l.intercept}

/-- The dot product of two points in ℝ² -/
def dot_product (p q : ℝ × ℝ) : ℝ :=
  p.1 * q.1 + p.2 * q.2

theorem line_passes_through_fixed_point (l : Line) 
    (h_distinct : ∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ intersection l ∧ B ∈ intersection l)
    (h_dot_product : ∃ A B : ℝ × ℝ, A ∈ intersection l ∧ B ∈ intersection l ∧ 
                     dot_product A B = -4) :
    (2, 0) ∈ {p : ℝ × ℝ | p.1 = l.slope * p.2 + l.intercept} :=
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2498_249856


namespace NUMINAMATH_CALUDE_jenny_lasagna_sales_l2498_249814

/-- The number of pans of lasagna Jenny makes and sells -/
def num_pans : ℕ := 20

/-- The cost to make each pan of lasagna -/
def cost_per_pan : ℚ := 10

/-- The selling price of each pan of lasagna -/
def price_per_pan : ℚ := 25

/-- The profit after expenses -/
def profit : ℚ := 300

/-- Theorem stating that the number of pans sold is correct given the conditions -/
theorem jenny_lasagna_sales : 
  (price_per_pan - cost_per_pan) * num_pans = profit := by sorry

end NUMINAMATH_CALUDE_jenny_lasagna_sales_l2498_249814


namespace NUMINAMATH_CALUDE_second_test_score_proof_l2498_249836

def first_test_score : ℝ := 78
def new_average : ℝ := 81

theorem second_test_score_proof :
  ∃ (second_score : ℝ), (first_test_score + second_score) / 2 = new_average ∧ second_score = 84 :=
by sorry

end NUMINAMATH_CALUDE_second_test_score_proof_l2498_249836


namespace NUMINAMATH_CALUDE_fraction_equality_l2498_249871

theorem fraction_equality : 
  (14 : ℚ) / 12 = 7 / 6 ∧
  (1 : ℚ) + 1 / 6 = 7 / 6 ∧
  (21 : ℚ) / 18 = 7 / 6 ∧
  (1 : ℚ) + 2 / 12 = 7 / 6 ∧
  (1 : ℚ) + 1 / 3 ≠ 7 / 6 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2498_249871


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l2498_249835

/-- The volume of a cylinder minus the volumes of two cones -/
theorem cylinder_minus_cones_volume (r h₁ h₂ h : ℝ) (hr : r = 10) (hh₁ : h₁ = 10) (hh₂ : h₂ = 16) (hh : h = 26) :
  π * r^2 * h - (1/3 * π * r^2 * h₁ + 1/3 * π * r^2 * h₂) = 2600/3 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l2498_249835


namespace NUMINAMATH_CALUDE_pradeep_failed_marks_l2498_249811

def total_marks : ℕ := 550
def passing_percentage : ℚ := 40 / 100
def pradeep_marks : ℕ := 200

theorem pradeep_failed_marks : 
  (total_marks * passing_percentage).floor - pradeep_marks = 20 := by
  sorry

end NUMINAMATH_CALUDE_pradeep_failed_marks_l2498_249811


namespace NUMINAMATH_CALUDE_problem_statement_l2498_249847

theorem problem_statement (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2498_249847


namespace NUMINAMATH_CALUDE_one_can_per_person_day1_l2498_249816

/-- Represents the food bank scenario --/
structure FoodBank where
  initialStock : ℕ
  day1People : ℕ
  day1Restock : ℕ
  day2People : ℕ
  day2CansPerPerson : ℕ
  day2Restock : ℕ
  totalGivenAway : ℕ

/-- Calculates the number of cans each person took on the first day --/
def cansPerPersonDay1 (fb : FoodBank) : ℕ :=
  (fb.totalGivenAway - fb.day2People * fb.day2CansPerPerson) / fb.day1People

/-- Theorem stating that each person took 1 can on the first day --/
theorem one_can_per_person_day1 (fb : FoodBank)
    (h1 : fb.initialStock = 2000)
    (h2 : fb.day1People = 500)
    (h3 : fb.day1Restock = 1500)
    (h4 : fb.day2People = 1000)
    (h5 : fb.day2CansPerPerson = 2)
    (h6 : fb.day2Restock = 3000)
    (h7 : fb.totalGivenAway = 2500) :
    cansPerPersonDay1 fb = 1 := by
  sorry

#eval cansPerPersonDay1 {
  initialStock := 2000,
  day1People := 500,
  day1Restock := 1500,
  day2People := 1000,
  day2CansPerPerson := 2,
  day2Restock := 3000,
  totalGivenAway := 2500
}

end NUMINAMATH_CALUDE_one_can_per_person_day1_l2498_249816


namespace NUMINAMATH_CALUDE_k_value_l2498_249813

def length (k : ℕ) : ℕ := sorry

theorem k_value (k : ℕ) (h1 : k > 1) (h2 : length k = 4) (h3 : k = 2 * 2 * 2 * 3) : k = 24 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l2498_249813


namespace NUMINAMATH_CALUDE_line_counting_theorem_l2498_249892

theorem line_counting_theorem (n : ℕ) : 
  n > 0 → 
  n % 4 = 3 → 
  (∀ k : ℕ, k ≤ n → k % 4 = (if k % 4 = 0 then 4 else k % 4)) → 
  n = 47 := by
sorry

end NUMINAMATH_CALUDE_line_counting_theorem_l2498_249892


namespace NUMINAMATH_CALUDE_oranges_in_box_l2498_249897

/-- The number of oranges Jonathan takes from the box -/
def oranges_taken : ℕ := 45

/-- The number of oranges left in the box after Jonathan takes some -/
def oranges_left : ℕ := 51

/-- The initial number of oranges in the box -/
def initial_oranges : ℕ := oranges_taken + oranges_left

theorem oranges_in_box : initial_oranges = 96 := by
  sorry

end NUMINAMATH_CALUDE_oranges_in_box_l2498_249897


namespace NUMINAMATH_CALUDE_factorial_2007_properties_l2498_249805

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def trailingZeros (n : ℕ) : ℕ :=
  (List.range 4).foldl (λ acc i => acc + n / (5 ^ (i + 1))) 0

def lastNonZeroDigit (n : ℕ) : ℕ := n % 10

theorem factorial_2007_properties :
  trailingZeros (factorial 2007) = 500 ∧
  lastNonZeroDigit (factorial 2007 / (10 ^ trailingZeros (factorial 2007))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_2007_properties_l2498_249805


namespace NUMINAMATH_CALUDE_abs_is_even_and_increasing_l2498_249874

def f (x : ℝ) := abs x

theorem abs_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_abs_is_even_and_increasing_l2498_249874


namespace NUMINAMATH_CALUDE_four_digit_numbers_divisible_by_13_l2498_249863

theorem four_digit_numbers_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card + 1 = 693 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_divisible_by_13_l2498_249863


namespace NUMINAMATH_CALUDE_marbles_left_l2498_249839

theorem marbles_left (red : ℕ) (blue : ℕ) (broken : ℕ) 
  (h1 : red = 156) 
  (h2 : blue = 267) 
  (h3 : broken = 115) : 
  red + blue - broken = 308 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_l2498_249839


namespace NUMINAMATH_CALUDE_total_daily_allowance_l2498_249889

theorem total_daily_allowance (total_students : ℕ) 
  (high_allowance : ℚ) (low_allowance : ℚ) :
  total_students = 60 →
  high_allowance = 6 →
  low_allowance = 4 →
  (2 : ℚ) / 3 * total_students * high_allowance + 
  (1 : ℚ) / 3 * total_students * low_allowance = 320 := by
sorry

end NUMINAMATH_CALUDE_total_daily_allowance_l2498_249889


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_l2498_249890

theorem fraction_inequality_solution (y : ℝ) : 
  1 / (y * (y + 2)) - 1 / ((y + 2) * (y + 4)) < 1 / 4 ↔ 
  y < -4 ∨ (-2 < y ∧ y < 0) ∨ y > 2 :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_l2498_249890


namespace NUMINAMATH_CALUDE_base_irrelevant_l2498_249858

theorem base_irrelevant (b : ℝ) : 
  ∃ (x y : ℝ), 3^x * b^y = 19683 ∧ x - y = 9 ∧ x = 9 → 3^9 * b^0 = 19683 := by
  sorry

end NUMINAMATH_CALUDE_base_irrelevant_l2498_249858


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_and_triangle_l2498_249860

theorem quadratic_equation_roots_and_triangle (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - (m+3)*x₁ + m + 1 = 0 ∧ 
    x₂^2 - (m+3)*x₂ + m + 1 = 0) ∧ 
  (∃ x : ℝ, x^2 - (m+3)*x + m + 1 = 0 ∧ x = 4 → 
    ∃ y : ℝ, y^2 - (m+3)*y + m + 1 = 0 ∧ y ≠ 4 ∧ 
    4 + 4 + y = 26/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_and_triangle_l2498_249860


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l2498_249896

theorem algebraic_expression_equality (x : ℝ) (h : x * (x + 2) = 2023) :
  2 * (x + 3) * (x - 1) - 2018 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l2498_249896


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_1024_for_25_divisibility_l2498_249806

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem least_addition_to_1024_for_25_divisibility :
  ∃ (x : ℕ), x < 25 ∧ (1024 + x) % 25 = 0 ∧ ∀ (y : ℕ), y < x → (1024 + y) % 25 ≠ 0 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_to_1024_for_25_divisibility_l2498_249806


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2498_249827

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) 
  (h2 : a 1 - a 2 = 3) 
  (h3 : a 1 - a 3 = 2) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = -1/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2498_249827


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l2498_249872

theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℚ) : 
  num_shelves = 520 → books_per_shelf = 37.5 → num_shelves * books_per_shelf = 19500 := by
  sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l2498_249872


namespace NUMINAMATH_CALUDE_math_team_selection_ways_l2498_249883

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem math_team_selection_ways :
  let boys := 7
  let girls := 9
  let team_boys := 3
  let team_girls := 3
  (choose boys team_boys) * (choose girls team_girls) = 2940 := by
sorry

end NUMINAMATH_CALUDE_math_team_selection_ways_l2498_249883


namespace NUMINAMATH_CALUDE_apple_ratio_l2498_249810

theorem apple_ratio (blue_apples : ℕ) (yellow_apples : ℕ) : 
  blue_apples = 5 →
  yellow_apples + blue_apples - (yellow_apples + blue_apples) / 5 = 12 →
  yellow_apples / blue_apples = 2 :=
by sorry

end NUMINAMATH_CALUDE_apple_ratio_l2498_249810


namespace NUMINAMATH_CALUDE_max_sum_cubes_constraint_max_sum_cubes_constraint_achievable_l2498_249838

theorem max_sum_cubes_constraint (p q r s t : ℝ) 
  (h : p^2 + q^2 + r^2 + s^2 + t^2 = 5) : 
  p^3 + q^3 + r^3 + s^3 + t^3 ≤ 5 * Real.sqrt 5 := by
  sorry

theorem max_sum_cubes_constraint_achievable : 
  ∃ (p q r s t : ℝ), p^2 + q^2 + r^2 + s^2 + t^2 = 5 ∧ 
  p^3 + q^3 + r^3 + s^3 + t^3 = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_cubes_constraint_max_sum_cubes_constraint_achievable_l2498_249838


namespace NUMINAMATH_CALUDE_garden_plant_count_l2498_249802

/-- The number of plants in a garden with given rows and columns -/
def garden_plants (rows : ℕ) (columns : ℕ) : ℕ := rows * columns

/-- Theorem: A garden with 52 rows and 15 columns has 780 plants -/
theorem garden_plant_count : garden_plants 52 15 = 780 := by
  sorry

end NUMINAMATH_CALUDE_garden_plant_count_l2498_249802


namespace NUMINAMATH_CALUDE_min_m_value_l2498_249899

theorem min_m_value (m : ℝ) (h_m : m > 0) :
  (∀ x : ℝ, x ∈ Set.Ioc 0 1 → |m * x^3 - Real.log x| ≥ 1) →
  m ≥ (1/3) * Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_min_m_value_l2498_249899


namespace NUMINAMATH_CALUDE_total_balloons_proof_l2498_249880

def sam_initial_balloons : ℝ := 6.0
def sam_given_balloons : ℝ := 5.0
def mary_balloons : ℝ := 7.0

theorem total_balloons_proof :
  sam_initial_balloons - sam_given_balloons + mary_balloons = 8 :=
by sorry

end NUMINAMATH_CALUDE_total_balloons_proof_l2498_249880


namespace NUMINAMATH_CALUDE_negation_is_false_l2498_249867

theorem negation_is_false : 
  ¬(∀ x y : ℝ, (x > 2 ∧ y > 3) → x + y > 5) = False :=
sorry

end NUMINAMATH_CALUDE_negation_is_false_l2498_249867


namespace NUMINAMATH_CALUDE_polar_equation_circle_and_ray_l2498_249817

/-- The polar equation (ρ - 1)(θ - π) = 0 with ρ ≥ 0 represents the union of a circle and a ray -/
theorem polar_equation_circle_and_ray (ρ θ : ℝ) :
  ρ ≥ 0 → (ρ - 1) * (θ - Real.pi) = 0 → 
  (∃ (x y : ℝ), x^2 + y^2 = 1) ∨ 
  (∃ (t : ℝ), t ≥ 0 → ∃ (x y : ℝ), x = -t ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_polar_equation_circle_and_ray_l2498_249817


namespace NUMINAMATH_CALUDE_banana_cost_18lbs_l2498_249881

/-- The cost of bananas given a rate, weight, and discount condition -/
def banana_cost (rate : ℚ) (rate_weight : ℚ) (weight : ℚ) (discount_threshold : ℚ) (discount_rate : ℚ) : ℚ :=
  let base_cost := (weight / rate_weight) * rate
  if weight ≥ discount_threshold then
    base_cost * (1 - discount_rate)
  else
    base_cost

/-- Theorem stating the cost of 18 pounds of bananas given the specified conditions -/
theorem banana_cost_18lbs : 
  banana_cost 3 3 18 15 (1/10) = 162/10 := by
  sorry

end NUMINAMATH_CALUDE_banana_cost_18lbs_l2498_249881


namespace NUMINAMATH_CALUDE_f_negative_one_value_l2498_249821

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The definition of f for positive x -/
def f_pos (x : ℝ) : ℝ :=
  2 * x^2 - 1

theorem f_negative_one_value
    (f : ℝ → ℝ)
    (h_odd : IsOdd f)
    (h_pos : ∀ x > 0, f x = f_pos x) :
    f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_one_value_l2498_249821


namespace NUMINAMATH_CALUDE_average_daily_sales_l2498_249869

/-- Theorem: Average daily sales of cups over a 12-day period -/
theorem average_daily_sales (day_one_sales : ℕ) (other_days_sales : ℕ) (total_days : ℕ) :
  day_one_sales = 86 →
  other_days_sales = 50 →
  total_days = 12 →
  (day_one_sales + (total_days - 1) * other_days_sales) / total_days = 53 :=
by sorry

end NUMINAMATH_CALUDE_average_daily_sales_l2498_249869


namespace NUMINAMATH_CALUDE_marks_fruit_consumption_l2498_249845

/-- Given the conditions of Mark's fruit consumption, prove that he ate 5 pieces in the first four days --/
theorem marks_fruit_consumption
  (total : ℕ)
  (kept_for_next_week : ℕ)
  (brought_on_friday : ℕ)
  (h1 : total = 10)
  (h2 : kept_for_next_week = 2)
  (h3 : brought_on_friday = 3) :
  total - kept_for_next_week - brought_on_friday = 5 := by
  sorry

end NUMINAMATH_CALUDE_marks_fruit_consumption_l2498_249845


namespace NUMINAMATH_CALUDE_total_dolls_count_l2498_249843

theorem total_dolls_count (big_box_capacity : ℕ) (small_box_capacity : ℕ) 
                          (big_box_count : ℕ) (small_box_count : ℕ) 
                          (h1 : big_box_capacity = 7)
                          (h2 : small_box_capacity = 4)
                          (h3 : big_box_count = 5)
                          (h4 : small_box_count = 9) :
  big_box_capacity * big_box_count + small_box_capacity * small_box_count = 71 :=
by sorry

end NUMINAMATH_CALUDE_total_dolls_count_l2498_249843


namespace NUMINAMATH_CALUDE_least_positive_four_digit_solution_l2498_249800

theorem least_positive_four_digit_solution (x : ℕ) : 
  (10 * x ≡ 30 [ZMOD 20]) ∧ 
  (2 * x + 10 ≡ 19 [ZMOD 9]) ∧ 
  (-3 * x + 1 ≡ x [ZMOD 19]) ∧ 
  (x ≥ 1000) ∧ (x < 10000) →
  x ≥ 1296 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_four_digit_solution_l2498_249800


namespace NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l2498_249840

theorem circle_radius_from_longest_chord (c : ℝ) (h : c > 0) :
  (∃ (r : ℝ), r > 0 ∧ c = 24 ∧ c = 2 * r) → c / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l2498_249840


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2498_249837

theorem simplify_trig_expression :
  1 / Real.sin (15 * π / 180) - 1 / Real.cos (15 * π / 180) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2498_249837


namespace NUMINAMATH_CALUDE_star_difference_equals_45_l2498_249809

/-- The star operation defined as x ★ y = x^2y - 3x -/
def star (x y : ℝ) : ℝ := x^2 * y - 3 * x

/-- Theorem stating that (6 ★ 3) - (3 ★ 6) = 45 -/
theorem star_difference_equals_45 : star 6 3 - star 3 6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_equals_45_l2498_249809


namespace NUMINAMATH_CALUDE_rosalina_total_gifts_l2498_249898

/-- The number of gifts Rosalina received from Emilio -/
def gifts_from_emilio : ℕ := 11

/-- The number of gifts Rosalina received from Jorge -/
def gifts_from_jorge : ℕ := 6

/-- The number of gifts Rosalina received from Pedro -/
def gifts_from_pedro : ℕ := 4

/-- The total number of gifts Rosalina received -/
def total_gifts : ℕ := gifts_from_emilio + gifts_from_jorge + gifts_from_pedro

theorem rosalina_total_gifts : total_gifts = 21 := by
  sorry

end NUMINAMATH_CALUDE_rosalina_total_gifts_l2498_249898
