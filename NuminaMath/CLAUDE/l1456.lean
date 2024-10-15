import Mathlib

namespace NUMINAMATH_CALUDE_sugar_in_recipe_l1456_145665

/-- Given a cake recipe and Mary's baking progress, calculate the amount of sugar required. -/
theorem sugar_in_recipe (total_flour sugar remaining_flour : ℕ) : 
  total_flour = 10 →
  remaining_flour = total_flour - 7 →
  remaining_flour = sugar + 1 →
  sugar = 2 := by sorry

end NUMINAMATH_CALUDE_sugar_in_recipe_l1456_145665


namespace NUMINAMATH_CALUDE_second_concert_attendance_l1456_145608

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (attendance_increase : ℕ) 
  (h1 : first_concert = 65899)
  (h2 : attendance_increase = 119) :
  first_concert + attendance_increase = 66018 :=
by sorry

end NUMINAMATH_CALUDE_second_concert_attendance_l1456_145608


namespace NUMINAMATH_CALUDE_field_ratio_proof_l1456_145691

theorem field_ratio_proof (length width : ℝ) : 
  length = 24 → 
  width = 13.5 → 
  (2 * width) / length = 9 / 8 := by
sorry

end NUMINAMATH_CALUDE_field_ratio_proof_l1456_145691


namespace NUMINAMATH_CALUDE_complement_of_angle_A_l1456_145618

-- Define the angle A
def angle_A : ℝ := 36

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- Theorem statement
theorem complement_of_angle_A :
  complement angle_A = 54 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle_A_l1456_145618


namespace NUMINAMATH_CALUDE_complex_sum_problem_l1456_145620

theorem complex_sum_problem (a b c d e f : ℝ) : 
  b = 4 →
  e = 2 * (-a - c) →
  Complex.mk (a + c + e) (b + d + f) = Complex.I * 6 →
  d + f = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l1456_145620


namespace NUMINAMATH_CALUDE_odd_function_properties_and_inequality_l1456_145692

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x - a * 2^(-x)

theorem odd_function_properties_and_inequality (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) →
  (a = 1 ∧ ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (∀ t : ℝ, (∀ x : ℝ, f a (x - t) + f a (x^2 - t^2) ≥ 0) → t = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_and_inequality_l1456_145692


namespace NUMINAMATH_CALUDE_fraction_pure_fuji_l1456_145657

-- Define the total number of trees
def total_trees : ℕ := 180

-- Define the number of pure Fuji trees
def pure_fuji : ℕ := 135

-- Define the number of pure Gala trees
def pure_gala : ℕ := 27

-- Define the number of cross-pollinated trees
def cross_pollinated : ℕ := 18

-- Define the cross-pollination rate
def cross_pollination_rate : ℚ := 1/10

-- Theorem stating the fraction of pure Fuji trees
theorem fraction_pure_fuji :
  (pure_fuji : ℚ) / total_trees = 3/4 :=
by
  sorry

-- Conditions from the problem
axiom condition1 : pure_fuji + cross_pollinated = 153
axiom condition2 : (cross_pollinated : ℚ) / total_trees = cross_pollination_rate
axiom condition3 : total_trees = pure_fuji + pure_gala + cross_pollinated

end NUMINAMATH_CALUDE_fraction_pure_fuji_l1456_145657


namespace NUMINAMATH_CALUDE_product_of_polynomials_l1456_145658

theorem product_of_polynomials (p q : ℚ) : 
  (∀ d : ℚ, (8 * d^2 - 4 * d + p) * (4 * d^2 + q * d - 9) = 32 * d^4 - 68 * d^3 + 5 * d^2 + 23 * d - 36) →
  p + q = 3/4 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l1456_145658


namespace NUMINAMATH_CALUDE_triangular_weight_is_60_l1456_145688

/-- The weight of a rectangular weight in grams -/
def rectangular_weight : ℝ := 90

/-- The weight of a round weight in grams -/
def round_weight : ℝ := 30

/-- The weight of a triangular weight in grams -/
def triangular_weight : ℝ := 60

/-- Theorem stating that the weight of a triangular weight is 60 grams -/
theorem triangular_weight_is_60 :
  (1 * round_weight + 1 * triangular_weight = 3 * round_weight) ∧
  (4 * round_weight + 1 * triangular_weight = 1 * triangular_weight + 1 * round_weight + rectangular_weight) →
  triangular_weight = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangular_weight_is_60_l1456_145688


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_7_l1456_145622

/-- A bag containing cards numbered from 0 to 5 -/
def Bag : Finset ℕ := {0, 1, 2, 3, 4, 5}

/-- The sample space of drawing one card from each bag -/
def SampleSpace : Finset (ℕ × ℕ) := Bag.product Bag

/-- The event where the sum of two drawn cards is greater than 7 -/
def EventSumGreaterThan7 : Finset (ℕ × ℕ) :=
  SampleSpace.filter (fun p => p.1 + p.2 > 7)

/-- The probability of the event -/
def ProbabilityEventSumGreaterThan7 : ℚ :=
  EventSumGreaterThan7.card / SampleSpace.card

theorem probability_sum_greater_than_7 :
  ProbabilityEventSumGreaterThan7 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_7_l1456_145622


namespace NUMINAMATH_CALUDE_prob_queens_or_aces_l1456_145606

def standard_deck : ℕ := 52
def num_aces : ℕ := 4
def num_queens : ℕ := 4

def prob_two_queens : ℚ := (num_queens * (num_queens - 1)) / (standard_deck * (standard_deck - 1))
def prob_one_ace : ℚ := 2 * (num_aces * (standard_deck - num_aces)) / (standard_deck * (standard_deck - 1))
def prob_two_aces : ℚ := (num_aces * (num_aces - 1)) / (standard_deck * (standard_deck - 1))

theorem prob_queens_or_aces :
  prob_two_queens + prob_one_ace + prob_two_aces = 2 / 13 :=
sorry

end NUMINAMATH_CALUDE_prob_queens_or_aces_l1456_145606


namespace NUMINAMATH_CALUDE_solve_y_l1456_145617

theorem solve_y (x y : ℝ) (h1 : x^2 = y - 3) (h2 : x = 7) : 
  y = 52 ∧ y ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_solve_y_l1456_145617


namespace NUMINAMATH_CALUDE_sphere_diameter_sum_l1456_145630

theorem sphere_diameter_sum (r : ℝ) (d : ℝ) (a b : ℕ) : 
  r = 6 →
  d = 2 * (3 * (4 / 3 * π * r^3))^(1/3) →
  d = a * (b : ℝ)^(1/3) →
  b > 0 →
  ∀ k : ℕ, k > 1 → k^3 ∣ b → k = 1 →
  a + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_sphere_diameter_sum_l1456_145630


namespace NUMINAMATH_CALUDE_correct_units_l1456_145670

-- Define the volume units
inductive VolumeUnit
| Milliliter
| Liter

-- Define the containers
structure Container where
  name : String
  volume : ℕ
  unit : VolumeUnit

-- Define the given containers
def orangeJuiceCup : Container :=
  { name := "Cup of orange juice", volume := 500, unit := VolumeUnit.Milliliter }

def waterBottle : Container :=
  { name := "Water bottle", volume := 3, unit := VolumeUnit.Liter }

-- Theorem to prove
theorem correct_units :
  (orangeJuiceCup.unit = VolumeUnit.Milliliter) ∧
  (waterBottle.unit = VolumeUnit.Liter) :=
by sorry

end NUMINAMATH_CALUDE_correct_units_l1456_145670


namespace NUMINAMATH_CALUDE_right_triangle_sin_c_l1456_145661

theorem right_triangle_sin_c (A B C : ℝ) (h_right : A = 90) (h_sin_b : Real.sin B = 3/5) :
  Real.sin C = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_c_l1456_145661


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1456_145654

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

-- Define the universal set U
def U : Set ℕ := A ∪ B

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1456_145654


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1456_145672

/-- The area of the union of a square and a circle, where the square has side length 8 and the circle has radius 12 and is centered at one of the square's vertices. -/
theorem square_circle_union_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 64 + 108 * π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1456_145672


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1456_145623

theorem sufficient_not_necessary (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, HasDerivAt (fun x => a^x) (a^x * Real.log a) x ∧ a^x * Real.log a < 0) →
  (∀ x : ℝ, HasDerivAt (fun x => (2-a)*x^3) (3*(2-a)*x^2) x ∧ 3*(2-a)*x^2 > 0) ∧
  ¬((∀ x : ℝ, HasDerivAt (fun x => (2-a)*x^3) (3*(2-a)*x^2) x ∧ 3*(2-a)*x^2 > 0) →
    (∀ x : ℝ, HasDerivAt (fun x => a^x) (a^x * Real.log a) x ∧ a^x * Real.log a < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1456_145623


namespace NUMINAMATH_CALUDE_cost_of_seven_sandwiches_five_sodas_l1456_145632

def sandwich_cost : ℝ := 4
def soda_cost : ℝ := 3
def discount_threshold : ℕ := 10
def discount_rate : ℝ := 0.1

def total_cost (num_sandwiches num_sodas : ℕ) : ℝ :=
  let total_items := num_sandwiches + num_sodas
  let subtotal := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  if total_items > discount_threshold then
    subtotal * (1 - discount_rate)
  else
    subtotal

theorem cost_of_seven_sandwiches_five_sodas :
  total_cost 7 5 = 38.7 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_seven_sandwiches_five_sodas_l1456_145632


namespace NUMINAMATH_CALUDE_expression_value_l1456_145695

theorem expression_value (x y : ℝ) (h : x - 3*y = 4) : 15*y - 5*x + 6 = -14 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1456_145695


namespace NUMINAMATH_CALUDE_steven_peach_count_l1456_145687

/-- 
Given that:
- Jake has 84 more apples than Steven
- Jake has 10 fewer peaches than Steven
- Steven has 52 apples
- Jake has 3 peaches

Prove that Steven has 13 peaches.
-/
theorem steven_peach_count (jake_apple_diff : ℕ) (jake_peach_diff : ℕ) 
  (steven_apples : ℕ) (jake_peaches : ℕ) 
  (h1 : jake_apple_diff = 84)
  (h2 : jake_peach_diff = 10)
  (h3 : steven_apples = 52)
  (h4 : jake_peaches = 3) : 
  jake_peaches + jake_peach_diff = 13 := by
  sorry

end NUMINAMATH_CALUDE_steven_peach_count_l1456_145687


namespace NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l1456_145604

theorem tan_fifteen_ratio_equals_sqrt_three :
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l1456_145604


namespace NUMINAMATH_CALUDE_age_difference_l1456_145641

theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 5 →
  sachin_age * 12 = rahul_age * 5 →
  rahul_age - sachin_age = 7 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1456_145641


namespace NUMINAMATH_CALUDE_min_value_3x_plus_y_l1456_145667

theorem min_value_3x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 2 / (x + 4) + 1 / (y + 3) = 1 / 4) :
  3 * x + y ≥ -8 + 20 * Real.sqrt 2 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧
    2 / (x₀ + 4) + 1 / (y₀ + 3) = 1 / 4 ∧
    3 * x₀ + y₀ = -8 + 20 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_y_l1456_145667


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_28_remainder_4_mod_15_l1456_145650

theorem smallest_number_divisible_by_28_remainder_4_mod_15 :
  ∃ n : ℕ, (n % 28 = 0) ∧ (n % 15 = 4) ∧ 
  (∀ m : ℕ, m < n → (m % 28 ≠ 0 ∨ m % 15 ≠ 4)) ∧ 
  n = 364 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_28_remainder_4_mod_15_l1456_145650


namespace NUMINAMATH_CALUDE_max_sum_of_goods_l1456_145655

theorem max_sum_of_goods (a b : ℕ) : 
  a > 0 → b > 0 → 5 * a + 19 * b = 213 → (∀ x y : ℕ, x > 0 → y > 0 → 5 * x + 19 * y = 213 → a + b ≥ x + y) → a + b = 37 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_goods_l1456_145655


namespace NUMINAMATH_CALUDE_min_gennadies_for_festival_l1456_145690

/-- Represents the number of people with a specific name -/
structure NameCount where
  alexanders : Nat
  borises : Nat
  vasilies : Nat

/-- Calculates the minimum number of Gennadies required -/
def minGennadies (counts : NameCount) : Nat :=
  counts.borises - 1 - (counts.alexanders + counts.vasilies)

/-- Theorem stating the minimum number of Gennadies required for the given counts -/
theorem min_gennadies_for_festival (counts : NameCount) 
  (h_alex : counts.alexanders = 45)
  (h_boris : counts.borises = 122)
  (h_vasily : counts.vasilies = 27) :
  minGennadies counts = 49 := by
  sorry

#eval minGennadies { alexanders := 45, borises := 122, vasilies := 27 }

end NUMINAMATH_CALUDE_min_gennadies_for_festival_l1456_145690


namespace NUMINAMATH_CALUDE_tony_books_count_l1456_145675

/-- The number of books Tony read -/
def tony_books : ℕ := 23

/-- The number of books Dean read -/
def dean_books : ℕ := 12

/-- The number of books Breanna read -/
def breanna_books : ℕ := 17

/-- The number of books Tony and Dean both read -/
def tony_dean_overlap : ℕ := 3

/-- The number of books all three read -/
def all_overlap : ℕ := 1

/-- The total number of different books read by all three -/
def total_different_books : ℕ := 47

theorem tony_books_count :
  tony_books + dean_books - tony_dean_overlap + breanna_books - all_overlap = total_different_books :=
by sorry

end NUMINAMATH_CALUDE_tony_books_count_l1456_145675


namespace NUMINAMATH_CALUDE_hit_at_least_once_complement_of_miss_all_l1456_145669

-- Define the sample space
def Ω : Type := Fin 3 → Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ∃ i, ω i = true

-- Define the event of not hitting the target at all
def miss_all (ω : Ω) : Prop :=
  ∀ i, ω i = false

-- Theorem statement
theorem hit_at_least_once_complement_of_miss_all :
  ∀ ω : Ω, hit_at_least_once ω ↔ ¬(miss_all ω) :=
sorry

end NUMINAMATH_CALUDE_hit_at_least_once_complement_of_miss_all_l1456_145669


namespace NUMINAMATH_CALUDE_perpendicular_lines_intersection_l1456_145686

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℚ) : Prop := m₁ * m₂ = -1

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def point_on_line (x y a b c : ℚ) : Prop := a * x + b * y + c = 0

/-- Given two perpendicular lines and their intersection point, prove p - m - n = 4 -/
theorem perpendicular_lines_intersection (m n p : ℚ) : 
  perpendicular (-2/m) (3/2) →
  point_on_line 2 p 2 m (-1) →
  point_on_line 2 p 3 (-2) n →
  p - m - n = 4 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_intersection_l1456_145686


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l1456_145621

theorem opposite_sides_line_range (a : ℝ) : 
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l1456_145621


namespace NUMINAMATH_CALUDE_circle_reflection_y_axis_l1456_145619

/-- Given a circle with equation (x+2)^2 + y^2 = 5, 
    its reflection about the y-axis has the equation (x-2)^2 + y^2 = 5 -/
theorem circle_reflection_y_axis (x y : ℝ) :
  ((x + 2)^2 + y^2 = 5) → 
  ∃ (x' y' : ℝ), ((x' - 2)^2 + y'^2 = 5 ∧ x' = -x ∧ y' = y) :=
sorry

end NUMINAMATH_CALUDE_circle_reflection_y_axis_l1456_145619


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1456_145698

theorem sufficient_not_necessary (p q : Prop) : 
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ 
  ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1456_145698


namespace NUMINAMATH_CALUDE_passengers_off_north_carolina_l1456_145638

/-- Represents the number of passengers at different stages of the flight --/
structure FlightPassengers where
  initial : ℕ
  offTexas : ℕ
  onTexas : ℕ
  onNorthCarolina : ℕ
  crew : ℕ
  landedVirginia : ℕ

/-- Calculates the number of passengers who got off in North Carolina --/
def passengersOffNorthCarolina (fp : FlightPassengers) : ℕ :=
  fp.initial - fp.offTexas + fp.onTexas - (fp.landedVirginia - fp.crew - fp.onNorthCarolina)

/-- Theorem stating that 47 passengers got off in North Carolina --/
theorem passengers_off_north_carolina :
  let fp : FlightPassengers := {
    initial := 124,
    offTexas := 58,
    onTexas := 24,
    onNorthCarolina := 14,
    crew := 10,
    landedVirginia := 67
  }
  passengersOffNorthCarolina fp = 47 := by
  sorry


end NUMINAMATH_CALUDE_passengers_off_north_carolina_l1456_145638


namespace NUMINAMATH_CALUDE_four_m_squared_minus_n_squared_l1456_145603

theorem four_m_squared_minus_n_squared (m n : ℝ) 
  (h1 : 2*m + n = 3) (h2 : 2*m - n = 1) : 4*m^2 - n^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_four_m_squared_minus_n_squared_l1456_145603


namespace NUMINAMATH_CALUDE_right_triangle_legs_l1456_145647

theorem right_triangle_legs (a Δ : ℝ) (ha : a > 0) (hΔ : Δ > 0) :
  ∃ x y : ℝ,
    x > 0 ∧ y > 0 ∧
    x^2 + y^2 = a^2 ∧
    x * y / 2 = Δ ∧
    x = (Real.sqrt (a^2 + 4*Δ) + Real.sqrt (a^2 - 4*Δ)) / 2 ∧
    y = (Real.sqrt (a^2 + 4*Δ) - Real.sqrt (a^2 - 4*Δ)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l1456_145647


namespace NUMINAMATH_CALUDE_package_contains_100_masks_l1456_145697

/-- The number of masks in a package used by a family -/
def number_of_masks (family_members : ℕ) (days_per_mask : ℕ) (total_days : ℕ) : ℕ :=
  family_members * (total_days / days_per_mask)

/-- Theorem: The package contains 100 masks -/
theorem package_contains_100_masks :
  number_of_masks 5 4 80 = 100 := by
  sorry

end NUMINAMATH_CALUDE_package_contains_100_masks_l1456_145697


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l1456_145631

theorem value_of_a_minus_b (a b : ℤ) 
  (eq1 : 2020 * a + 2024 * b = 2040)
  (eq2 : 2022 * a + 2026 * b = 2044) : 
  a - b = 1002 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l1456_145631


namespace NUMINAMATH_CALUDE_pizza_meat_distribution_l1456_145673

/-- Pizza meat distribution problem -/
theorem pizza_meat_distribution 
  (pepperoni : ℕ) 
  (ham : ℕ) 
  (sausage : ℕ) 
  (slices : ℕ) 
  (h1 : pepperoni = 30)
  (h2 : ham = 2 * pepperoni)
  (h3 : sausage = pepperoni + 12)
  (h4 : slices = 6)
  : (pepperoni + ham + sausage) / slices = 22 := by
  sorry

end NUMINAMATH_CALUDE_pizza_meat_distribution_l1456_145673


namespace NUMINAMATH_CALUDE_leap_day_2024_is_thursday_l1456_145679

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Function to determine if a year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Function to calculate the number of days between two dates -/
def daysBetween (date1 date2 : Date) : ℕ :=
  sorry

/-- Function to determine the day of the week given a starting day and number of days passed -/
def getDayOfWeek (startDay : DayOfWeek) (daysPassed : ℕ) : DayOfWeek :=
  sorry

theorem leap_day_2024_is_thursday :
  let leap_day_1996 : Date := ⟨1996, 2, 29⟩
  let leap_day_2024 : Date := ⟨2024, 2, 29⟩
  let days_between := daysBetween leap_day_1996 leap_day_2024
  getDayOfWeek DayOfWeek.Thursday days_between = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_leap_day_2024_is_thursday_l1456_145679


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l1456_145678

theorem sphere_radius_ratio (V_large V_small : ℝ) (h1 : V_large = 450 * Real.pi) 
  (h2 : V_small = 0.25 * V_large) : 
  (V_small / V_large) ^ (1/3 : ℝ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l1456_145678


namespace NUMINAMATH_CALUDE_f_2x_eq_3_l1456_145633

/-- A function that is constant 3 for all real inputs -/
def f : ℝ → ℝ := fun x ↦ 3

/-- Theorem: f(2x) = 3 given that f(x) = 3 for all real x -/
theorem f_2x_eq_3 : ∀ x : ℝ, f (2 * x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_2x_eq_3_l1456_145633


namespace NUMINAMATH_CALUDE_number_2008_row_l1456_145659

theorem number_2008_row : ∃ (n : ℕ), n = 45 ∧ 
  (n - 1)^2 < 2008 ∧ 2008 ≤ n^2 ∧ 
  (∀ (k : ℕ), k < n → k^2 < 2008) :=
by sorry

end NUMINAMATH_CALUDE_number_2008_row_l1456_145659


namespace NUMINAMATH_CALUDE_fib_105_mod_7_l1456_145624

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- The period of Fibonacci sequence modulo 7 -/
def fib_mod7_period : ℕ := 16

theorem fib_105_mod_7 : fib 104 % 7 = 2 := by
  sorry

#eval fib 104 % 7

end NUMINAMATH_CALUDE_fib_105_mod_7_l1456_145624


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1456_145652

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 5| + 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1456_145652


namespace NUMINAMATH_CALUDE_marble_difference_l1456_145668

theorem marble_difference (drew_initial marcus_initial : ℕ) : 
  drew_initial - marcus_initial = 24 ∧ 
  drew_initial - 12 = 25 ∧ 
  marcus_initial + 12 = 25 :=
by
  sorry

#check marble_difference

end NUMINAMATH_CALUDE_marble_difference_l1456_145668


namespace NUMINAMATH_CALUDE_no_valid_m_exists_l1456_145651

theorem no_valid_m_exists : ¬ ∃ (m : ℝ),
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 1 2 →
    x₁ + m > x₂^2 - m*x₂ + m^2/2 + 2*m - 3) ∧
  (Set.Ioo 1 2 = {x | x^2 - m*x + m^2/2 + 2*m - 3 < m^2/2 + 1}) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_m_exists_l1456_145651


namespace NUMINAMATH_CALUDE_field_trip_adults_l1456_145649

/-- Field trip problem -/
theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) :
  van_capacity = 5 →
  num_students = 12 →
  num_vans = 3 →
  (num_vans * van_capacity - num_students : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_adults_l1456_145649


namespace NUMINAMATH_CALUDE_orange_juice_ratio_l1456_145677

-- Define the given quantities
def servings : Nat := 280
def serving_size : Nat := 6  -- in ounces
def concentrate_cans : Nat := 35
def concentrate_can_size : Nat := 12  -- in ounces

-- Define the theorem
theorem orange_juice_ratio :
  let total_juice := servings * serving_size
  let total_concentrate := concentrate_cans * concentrate_can_size
  let water_needed := total_juice - total_concentrate
  let water_cans := water_needed / concentrate_can_size
  (water_cans : Int) / (concentrate_cans : Int) = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_ratio_l1456_145677


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_equals_one_two_l1456_145648

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set P
def P : Finset Nat := {1, 2, 3, 4}

-- Define set Q
def Q : Finset Nat := {3, 4, 5}

-- Theorem statement
theorem intersection_P_complement_Q_equals_one_two :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_equals_one_two_l1456_145648


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l1456_145628

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The 20th term of the arithmetic sequence with first term -6 and common difference 5 -/
theorem twentieth_term_of_sequence :
  arithmeticSequence (-6) 5 20 = 89 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l1456_145628


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l1456_145653

/-- Simplification of a trigonometric expression -/
theorem trig_expression_simplification :
  let expr := (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
               Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / 
              Real.cos (10 * π / 180)
  ∃ (k : ℝ), expr = (2 * Real.cos (40 * π / 180)) / Real.cos (10 * π / 180) * k :=
by
  sorry


end NUMINAMATH_CALUDE_trig_expression_simplification_l1456_145653


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l1456_145676

-- First expression
theorem factorization_1 (x y : ℝ) : -x^2 + 12*x*y - 36*y^2 = -(x - 6*y)^2 := by
  sorry

-- Second expression
theorem factorization_2 (x : ℝ) : x^4 - 9*x^2 = x^2 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l1456_145676


namespace NUMINAMATH_CALUDE_max_perimeter_constrained_quadrilateral_l1456_145662

/-- A convex quadrilateral with specific side and diagonal constraints -/
structure ConstrainedQuadrilateral where
  -- Two sides are equal to 1
  side1 : ℝ
  side2 : ℝ
  side1_eq_one : side1 = 1
  side2_eq_one : side2 = 1
  -- Other sides and diagonals are not greater than 1
  side3 : ℝ
  side4 : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  side3_le_one : side3 ≤ 1
  side4_le_one : side4 ≤ 1
  diagonal1_le_one : diagonal1 ≤ 1
  diagonal2_le_one : diagonal2 ≤ 1
  -- Convexity condition (simplified for this problem)
  is_convex : diagonal1 + diagonal2 > side1 + side3

/-- The maximum perimeter of a constrained quadrilateral -/
theorem max_perimeter_constrained_quadrilateral (q : ConstrainedQuadrilateral) :
  q.side1 + q.side2 + q.side3 + q.side4 ≤ 2 + 4 * Real.sin (15 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_constrained_quadrilateral_l1456_145662


namespace NUMINAMATH_CALUDE_fraction_of_120_l1456_145694

theorem fraction_of_120 : (1 / 3 : ℚ) * (1 / 4 : ℚ) * (1 / 6 : ℚ) * 120 = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_120_l1456_145694


namespace NUMINAMATH_CALUDE_book_price_increase_l1456_145637

theorem book_price_increase (original_price : ℝ) : 
  original_price * (1 + 0.6) = 480 → original_price = 300 := by
  sorry

end NUMINAMATH_CALUDE_book_price_increase_l1456_145637


namespace NUMINAMATH_CALUDE_hamburger_cost_l1456_145646

/-- Proves that the cost of each hamburger is $4 given the initial amount,
    the number of items purchased, the cost of milkshakes, and the remaining amount. -/
theorem hamburger_cost (initial_amount : ℕ) (num_hamburgers : ℕ) (num_milkshakes : ℕ)
                        (milkshake_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 120 →
  num_hamburgers = 8 →
  num_milkshakes = 6 →
  milkshake_cost = 3 →
  remaining_amount = 70 →
  ∃ (hamburger_cost : ℕ),
    initial_amount = num_hamburgers * hamburger_cost + num_milkshakes * milkshake_cost + remaining_amount ∧
    hamburger_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_hamburger_cost_l1456_145646


namespace NUMINAMATH_CALUDE_stone_123_is_12_l1456_145636

/-- Represents the counting pattern on a circle of stones -/
def stone_count (n : ℕ) : ℕ := 
  let cycle := 28
  n % cycle

/-- The original position of a stone given its count number -/
def original_position (count : ℕ) : ℕ :=
  if count ≤ 15 then count
  else 16 - (count - 15)

theorem stone_123_is_12 : original_position (stone_count 123) = 12 := by sorry

end NUMINAMATH_CALUDE_stone_123_is_12_l1456_145636


namespace NUMINAMATH_CALUDE_A_B_symmetric_about_x_axis_l1456_145613

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite. -/
def symmetric_about_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Point A in the coordinate plane -/
def A : ℝ × ℝ := (-1, 3)

/-- Point B in the coordinate plane -/
def B : ℝ × ℝ := (-1, -3)

/-- Theorem stating that points A and B are symmetric about the x-axis -/
theorem A_B_symmetric_about_x_axis :
  symmetric_about_x_axis A B := by sorry

end NUMINAMATH_CALUDE_A_B_symmetric_about_x_axis_l1456_145613


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1456_145635

theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_rest : ℕ) : 
  total_workers = 12 → 
  num_technicians = 7 → 
  avg_salary_technicians = 12000 → 
  avg_salary_rest = 6000 → 
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_rest) / total_workers = 9500 := by
sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l1456_145635


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l1456_145609

theorem largest_integer_inequality (x : ℤ) : x ≤ 4 ↔ x / 3 + 3 / 4 < 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l1456_145609


namespace NUMINAMATH_CALUDE_rod_system_equilibrium_l1456_145684

/-- Represents the equilibrium state of a rod system -/
structure RodSystem where
  l : Real          -- Length of the rod in meters
  m₂ : Real         -- Mass of the rod in kg
  s : Real          -- Distance of left thread attachment from right end in meters
  m₁ : Real         -- Mass of the load in kg

/-- Checks if the rod system is in equilibrium -/
def is_equilibrium (sys : RodSystem) : Prop :=
  sys.m₁ * sys.s = sys.m₂ * (sys.l / 2)

/-- Theorem stating the equilibrium condition for the given rod system -/
theorem rod_system_equilibrium :
  ∀ (sys : RodSystem),
    sys.l = 0.5 ∧ 
    sys.m₂ = 2 ∧ 
    sys.s = 0.1 ∧ 
    sys.m₁ = 5 →
    is_equilibrium sys := by
  sorry

end NUMINAMATH_CALUDE_rod_system_equilibrium_l1456_145684


namespace NUMINAMATH_CALUDE_mark_paid_54_l1456_145607

/-- The total amount Mark paid for hiring a singer -/
def total_paid (hours : ℕ) (rate : ℚ) (tip_percentage : ℚ) : ℚ :=
  let base_cost := hours * rate
  let tip := base_cost * tip_percentage
  base_cost + tip

/-- Theorem stating that Mark paid $54 for hiring the singer -/
theorem mark_paid_54 :
  total_paid 3 15 (20 / 100) = 54 := by
  sorry

end NUMINAMATH_CALUDE_mark_paid_54_l1456_145607


namespace NUMINAMATH_CALUDE_infinitely_many_n_exist_l1456_145627

-- Define the s operation on sets of integers
def s (F : Set ℤ) : Set ℤ :=
  {a : ℤ | (a ∈ F ∧ a - 1 ∉ F) ∨ (a ∉ F ∧ a - 1 ∈ F)}

-- Define the n-fold application of s
def s_power (F : Set ℤ) : ℕ → Set ℤ
  | 0 => F
  | n + 1 => s (s_power F n)

theorem infinitely_many_n_exist (F : Set ℤ) (h_finite : Set.Finite F) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, s_power F n = F ∪ {a + n | a ∈ F} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_n_exist_l1456_145627


namespace NUMINAMATH_CALUDE_expression_equals_forty_l1456_145639

theorem expression_equals_forty : (20 - (2010 - 201)) + (2010 - (201 - 20)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_forty_l1456_145639


namespace NUMINAMATH_CALUDE_initial_men_count_l1456_145626

/-- Given provisions that last 15 days for an initial group of men and 12.5 days when 200 more men join,
    prove that the initial number of men is 1000. -/
theorem initial_men_count (M : ℕ) (P : ℝ) : 
  (P / (15 * M) = P / (12.5 * (M + 200))) → M = 1000 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l1456_145626


namespace NUMINAMATH_CALUDE_triangle_existence_l1456_145683

/-- Given two angles and a perimeter, prove the existence of a triangle with these properties -/
theorem triangle_existence (A B P : ℝ) (h_angle_sum : 0 < A + B ∧ A + B < 180) (h_perimeter : P > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
    a + b > c ∧ b + c > a ∧ a + c > b ∧  -- Triangle inequality
    a + b + c = P ∧  -- Perimeter condition
    ∃ (C : ℝ), C = 180 - (A + B) ∧  -- Third angle
    Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) ∧  -- Cosine law for angle A
    Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) ∧  -- Cosine law for angle B
    Real.cos C = (a^2 + b^2 - c^2) / (2*a*b) :=  -- Cosine law for angle C
by sorry


end NUMINAMATH_CALUDE_triangle_existence_l1456_145683


namespace NUMINAMATH_CALUDE_locus_is_parabola_l1456_145644

-- Define the fixed point M
def M : ℝ × ℝ := (1, 0)

-- Define the fixed line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the locus of points P
def locus : Set (ℝ × ℝ) := {P : ℝ × ℝ | ∃ B ∈ l, dist P M = dist P B}

-- Theorem statement
theorem locus_is_parabola : 
  ∃ a b c : ℝ, locus = {P : ℝ × ℝ | P.2 = a * P.1^2 + b * P.1 + c} := by
  sorry

end NUMINAMATH_CALUDE_locus_is_parabola_l1456_145644


namespace NUMINAMATH_CALUDE_even_function_range_theorem_l1456_145645

-- Define an even function f on ℝ
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_range_theorem (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h_even : isEvenFunction f) 
  (h_deriv : ∀ x, HasDerivAt f (f' x) x)
  (h_ineq : ∀ x, 2 * f x + x * f' x < 2) :
  {x : ℝ | x^2 * f x - 4 * f 2 < x^2 - 4} = {x : ℝ | x < -2 ∨ x > 2} :=
by sorry

end NUMINAMATH_CALUDE_even_function_range_theorem_l1456_145645


namespace NUMINAMATH_CALUDE_harry_travel_time_ratio_l1456_145642

/-- Given Harry's travel times, prove the ratio of walking time to bus journey time -/
theorem harry_travel_time_ratio :
  let total_time : ℕ := 60
  let bus_time_elapsed : ℕ := 15
  let bus_time_remaining : ℕ := 25
  let bus_time_total : ℕ := bus_time_elapsed + bus_time_remaining
  let walking_time : ℕ := total_time - bus_time_total
  walking_time / bus_time_total = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_harry_travel_time_ratio_l1456_145642


namespace NUMINAMATH_CALUDE_larger_number_with_given_hcf_and_lcm_factors_l1456_145685

theorem larger_number_with_given_hcf_and_lcm_factors : 
  ∀ (a b : ℕ+), 
    (Nat.gcd a b = 47) → 
    (∃ (k : ℕ+), Nat.lcm a b = k * 47 * 7^2 * 11 * 13 * 17^3) →
    (a ≥ b) →
    a = 123800939 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_with_given_hcf_and_lcm_factors_l1456_145685


namespace NUMINAMATH_CALUDE_j_travel_time_l1456_145656

/-- Given two travelers J and L, where:
  * J takes 45 minutes less time than L to travel 45 miles
  * J travels 1/2 mile per hour faster than L
  * y is J's rate of speed in miles per hour
Prove that J's time to travel 45 miles is equal to 45/y -/
theorem j_travel_time (y : ℝ) (h1 : y > 0) : ∃ (t_j t_l : ℝ),
  t_j = 45 / y ∧
  t_l = 45 / (y - 1/2) ∧
  t_l - t_j = 3/4 :=
sorry

end NUMINAMATH_CALUDE_j_travel_time_l1456_145656


namespace NUMINAMATH_CALUDE_max_value_theorem_l1456_145666

structure Point where
  x : ℝ
  y : ℝ

def ellipse (p : Point) : Prop :=
  p.x^2 / 4 + 9 * p.y^2 / 4 = 1

def condition (p q : Point) : Prop :=
  p.x * q.x + 9 * p.y * q.y = -2

def expression (p q : Point) : ℝ :=
  |2 * p.x + 3 * p.y - 3| + |2 * q.x + 3 * q.y - 3|

theorem max_value_theorem (p q : Point) 
  (h1 : p ≠ q) 
  (h2 : ellipse p) 
  (h3 : ellipse q) 
  (h4 : condition p q) : 
  ∃ (max : ℝ), max = 6 + 2 * Real.sqrt 5 ∧ 
    ∀ (p' q' : Point), 
      p' ≠ q' → 
      ellipse p' → 
      ellipse q' → 
      condition p' q' → 
      expression p' q' ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1456_145666


namespace NUMINAMATH_CALUDE_function_sum_negative_l1456_145629

theorem function_sum_negative
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (x + 2) = -f (-x + 2))
  (h_increasing : ∀ x y, x > 2 → y > 2 → x < y → f x < f y)
  (x₁ x₂ : ℝ)
  (h_sum : x₁ + x₂ > 4)
  (h_product : (x₁ - 2) * (x₂ - 2) < 0) :
  f x₁ + f x₂ < 0 := by
sorry

end NUMINAMATH_CALUDE_function_sum_negative_l1456_145629


namespace NUMINAMATH_CALUDE_family_ages_solution_l1456_145699

def family_ages (w h s d : ℕ) : Prop :=
  -- Woman's age reversed equals husband's age
  w = 10 * (h % 10) + (h / 10) ∧
  -- Husband is older than woman
  h > w ∧
  -- Difference between ages is one-eleventh of their sum
  h - w = (h + w) / 11 ∧
  -- Son's age is difference between parents' ages
  s = h - w ∧
  -- Daughter's age is average of all ages
  d = (w + h + s) / 3 ∧
  -- Sum of digits of each age is the same
  (w % 10 + w / 10) = (h % 10 + h / 10) ∧
  (w % 10 + w / 10) = s ∧
  (w % 10 + w / 10) = (d % 10 + d / 10)

theorem family_ages_solution :
  ∃ (w h s d : ℕ), family_ages w h s d ∧ w = 45 ∧ h = 54 ∧ s = 9 ∧ d = 36 :=
by sorry

end NUMINAMATH_CALUDE_family_ages_solution_l1456_145699


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_specific_sqrt_equality_l1456_145689

theorem sqrt_fraction_equality (n : ℕ) (hn : n > 0) :
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = 1 + 1 / (n * (n+1) : ℝ) := by sorry

theorem specific_sqrt_equality :
  Real.sqrt (101/100 + 1/121) = 1 + 1/110 := by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_specific_sqrt_equality_l1456_145689


namespace NUMINAMATH_CALUDE_base9_arithmetic_l1456_145640

/-- Represents a number in base 9 --/
def Base9 : Type := ℕ

/-- Addition in base 9 --/
def add_base9 (a b : Base9) : Base9 := sorry

/-- Subtraction in base 9 --/
def sub_base9 (a b : Base9) : Base9 := sorry

/-- Conversion from decimal to base 9 --/
def to_base9 (n : ℕ) : Base9 := sorry

theorem base9_arithmetic :
  sub_base9 (add_base9 (to_base9 374) (to_base9 625)) (to_base9 261) = to_base9 738 := by sorry

end NUMINAMATH_CALUDE_base9_arithmetic_l1456_145640


namespace NUMINAMATH_CALUDE_function_properties_l1456_145643

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : isEven (fun x ↦ f (x - 3)))
  (h2 : isOdd (fun x ↦ f (2 * x - 1))) :
  (f (-1) = 0) ∧ 
  (∀ x, f x = f (-x - 6)) ∧ 
  (f 7 = 0) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l1456_145643


namespace NUMINAMATH_CALUDE_collinear_points_sum_l1456_145625

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t1 t2 : ℝ), p2 = (t1 • p1 + (1 - t1) • p3) ∧ p3 = (t2 • p1 + (1 - t2) • p2)

/-- If the points (1, a, b), (a, 2, b), and (a, b, 3) are collinear in 3-space, then a + b = 4. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (1, a, b) (a, 2, b) (a, b, 3) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l1456_145625


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l1456_145614

theorem sin_cos_sum_equals_one : 
  Real.sin (65 * π / 180) * Real.sin (115 * π / 180) + 
  Real.cos (65 * π / 180) * Real.sin (25 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l1456_145614


namespace NUMINAMATH_CALUDE_ellipse_sum_bound_l1456_145674

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 144 + y^2 / 25 = 1

-- Theorem statement
theorem ellipse_sum_bound :
  ∀ x y : ℝ, is_on_ellipse x y → -13 ≤ x + y ∧ x + y ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_bound_l1456_145674


namespace NUMINAMATH_CALUDE_expression_simplification_l1456_145616

theorem expression_simplification (x y : ℚ) 
  (hx : x = -3/8) (hy : y = 4) : 
  (x - 2*y)^2 + (x - 2*y)*(x + 2*y) - 2*x*(x - y) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1456_145616


namespace NUMINAMATH_CALUDE_candy_price_increase_l1456_145664

theorem candy_price_increase (W : ℝ) (P : ℝ) (h1 : W > 0) (h2 : P > 0) :
  let new_weight := 0.6 * W
  let old_price_per_unit := P / W
  let new_price_per_unit := P / new_weight
  (new_price_per_unit - old_price_per_unit) / old_price_per_unit * 100 = (5/3 - 1) * 100 :=
by sorry

end NUMINAMATH_CALUDE_candy_price_increase_l1456_145664


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l1456_145696

theorem product_remainder_mod_five : (14452 * 15652 * 16781) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l1456_145696


namespace NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l1456_145671

def vec_a : ℝ × ℝ := (2, 3)
def vec_b : ℝ × ℝ := (-1, 5)

theorem vector_addition_scalar_multiplication :
  vec_a + 3 • vec_b = (-1, 18) := by sorry

end NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l1456_145671


namespace NUMINAMATH_CALUDE_ham_to_pepperoni_ratio_l1456_145680

/-- Represents the number of pieces of each type of meat on a pizza -/
structure PizzaToppings where
  pepperoni : ℕ
  ham : ℕ
  sausage : ℕ

/-- Represents the properties of the pizza -/
structure Pizza where
  toppings : PizzaToppings
  slices : ℕ
  meat_per_slice : ℕ

/-- The ratio of ham to pepperoni is 2:1 given the specified conditions -/
theorem ham_to_pepperoni_ratio (pizza : Pizza) : 
  pizza.toppings.pepperoni = 30 ∧ 
  pizza.toppings.sausage = pizza.toppings.pepperoni + 12 ∧
  pizza.slices = 6 ∧
  pizza.meat_per_slice = 22 →
  pizza.toppings.ham = 2 * pizza.toppings.pepperoni := by
  sorry

#check ham_to_pepperoni_ratio

end NUMINAMATH_CALUDE_ham_to_pepperoni_ratio_l1456_145680


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l1456_145615

theorem largest_five_digit_divisible_by_8 : 
  ∀ n : ℕ, n ≤ 99999 ∧ n ≥ 10000 ∧ n % 8 = 0 → n ≤ 99992 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_8_l1456_145615


namespace NUMINAMATH_CALUDE_a_positive_if_f_decreasing_l1456_145681

/-- A function that represents a(x³ - x) --/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x^3 - x)

/-- The theorem stating that if f is decreasing on (-√3/3, √3/3), then a > 0 --/
theorem a_positive_if_f_decreasing (a : ℝ) :
  (∀ x₁ x₂ : ℝ, -Real.sqrt 3 / 3 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.sqrt 3 / 3 → f a x₁ > f a x₂) →
  a > 0 := by
  sorry


end NUMINAMATH_CALUDE_a_positive_if_f_decreasing_l1456_145681


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1456_145610

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * ((a - Complex.I) * (1 + Complex.I))).re = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1456_145610


namespace NUMINAMATH_CALUDE_focus_of_parabola_l1456_145612

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- State the theorem
theorem focus_of_parabola :
  ∃ (f : ℝ × ℝ), f = (1, 0) ∧ 
  (∀ (x y : ℝ), parabola x y → 
    (x - f.1)^2 + y^2 = (x - f.1 + f.1)^2) :=
sorry

end NUMINAMATH_CALUDE_focus_of_parabola_l1456_145612


namespace NUMINAMATH_CALUDE_no_rotation_matrix_exists_zero_matrix_is_answer_l1456_145602

theorem no_rotation_matrix_exists : ¬∃ (M : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), M * A = ![![A 1 0, A 0 0], ![A 1 1, A 0 1]] := by
  sorry

theorem zero_matrix_is_answer : 
  (∀ (A : Matrix (Fin 2) (Fin 2) ℝ), (0 : Matrix (Fin 2) (Fin 2) ℝ) * A ≠ ![![A 1 0, A 0 0], ![A 1 1, A 0 1]]) ∧
  (¬∃ (M : Matrix (Fin 2) (Fin 2) ℝ), ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), M * A = ![![A 1 0, A 0 0], ![A 1 1, A 0 1]]) :=
by
  sorry

end NUMINAMATH_CALUDE_no_rotation_matrix_exists_zero_matrix_is_answer_l1456_145602


namespace NUMINAMATH_CALUDE_highest_page_number_l1456_145693

/-- Represents the count of available digits --/
def DigitCount := Fin 10 → ℕ

/-- The set of digits where all digits except 5 are unlimited --/
def unlimitedExceptFive (d : DigitCount) : Prop :=
  ∀ i : Fin 10, i.val ≠ 5 → d i = 0 ∧ d 5 = 18

/-- Counts the occurrences of a digit in a natural number --/
def countDigit (digit : Fin 10) (n : ℕ) : ℕ :=
  sorry

/-- Counts the total occurrences of a digit in all numbers up to n --/
def totalDigitCount (digit : Fin 10) (n : ℕ) : ℕ :=
  sorry

/-- The main theorem --/
theorem highest_page_number (d : DigitCount) (h : unlimitedExceptFive d) :
  ∀ n : ℕ, n > 99 → totalDigitCount 5 n > 18 :=
sorry

end NUMINAMATH_CALUDE_highest_page_number_l1456_145693


namespace NUMINAMATH_CALUDE_train_length_l1456_145611

/-- The length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 65 →
  man_speed = 7 →
  passing_time = 5.4995600351971845 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.5 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l1456_145611


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1456_145663

def C : Set Nat := {47, 49, 51, 53, 55}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧
  (∀ (m : Nat), m ∈ C → (Nat.minFac n ≤ Nat.minFac m)) ∧
  n = 51 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1456_145663


namespace NUMINAMATH_CALUDE_probability_of_same_number_l1456_145660

/-- The upper bound for the selected numbers -/
def upper_bound : ℕ := 500

/-- Billy's number is a multiple of this value -/
def billy_multiple : ℕ := 20

/-- Bobbi's number is a multiple of this value -/
def bobbi_multiple : ℕ := 30

/-- The probability of Billy and Bobbi selecting the same number -/
def same_number_probability : ℚ := 1 / 50

/-- Theorem stating the probability of Billy and Bobbi selecting the same number -/
theorem probability_of_same_number :
  (∃ (b₁ b₂ : ℕ), b₁ > 0 ∧ b₂ > 0 ∧ b₁ < upper_bound ∧ b₂ < upper_bound ∧
   b₁ % billy_multiple = 0 ∧ b₂ % bobbi_multiple = 0) →
  same_number_probability = 1 / 50 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_same_number_l1456_145660


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1456_145600

theorem min_value_of_expression (a b : ℤ) (h1 : a > b) (h2 : a ≠ b) :
  (((a^2 + b^2) / (a^2 - b^2)) + ((a^2 - b^2) / (a^2 + b^2)) : ℚ) ≥ 2 ∧
  ∃ (a b : ℤ), a > b ∧ a ≠ b ∧ (((a^2 + b^2) / (a^2 - b^2)) + ((a^2 - b^2) / (a^2 + b^2)) : ℚ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1456_145600


namespace NUMINAMATH_CALUDE_modulo_evaluation_l1456_145605

theorem modulo_evaluation : (203 * 19 - 22 * 8 + 6) % 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_modulo_evaluation_l1456_145605


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1456_145634

theorem arithmetic_progression_first_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) = a n + 5) →  -- Common difference is 5
    a 21 = 103 →                 -- 21st term is 103
    a 1 = 3 :=                   -- First term is 3
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1456_145634


namespace NUMINAMATH_CALUDE_berts_money_l1456_145682

-- Define Bert's initial amount of money
variable (n : ℚ)

-- Define the remaining money after each step
def remaining_after_hardware (n : ℚ) : ℚ := (3/4) * n
def remaining_after_dry_cleaners (n : ℚ) : ℚ := remaining_after_hardware n - 9
def remaining_after_grocery (n : ℚ) : ℚ := (1/2) * remaining_after_dry_cleaners n
def remaining_after_books (n : ℚ) : ℚ := (2/3) * remaining_after_grocery n
def final_remaining (n : ℚ) : ℚ := (4/5) * remaining_after_books n

-- Theorem stating the relationship between n and the final amount
theorem berts_money (n : ℚ) : final_remaining n = 27 ↔ n = 72 := by sorry

end NUMINAMATH_CALUDE_berts_money_l1456_145682


namespace NUMINAMATH_CALUDE_solution_set_min_value_l1456_145601

-- Define the function f
def f (x : ℝ) := |2*x + 1| - |x - 4|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set (x : ℝ) : f x ≥ 2 ↔ x ≤ -7 ∨ x ≥ 5/3 :=
sorry

-- Theorem for the minimum value of f(x)
theorem min_value : ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = -9/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_min_value_l1456_145601
