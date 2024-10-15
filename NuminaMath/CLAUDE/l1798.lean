import Mathlib

namespace NUMINAMATH_CALUDE_max_residents_in_block_l1798_179875

/-- Represents a block of flats -/
structure BlockOfFlats where
  totalFloors : ℕ
  apartmentsPerFloorType1 : ℕ
  apartmentsPerFloorType2 : ℕ
  maxResidentsPerApartment : ℕ

/-- Calculates the maximum number of residents in a block of flats -/
def maxResidents (block : BlockOfFlats) : ℕ :=
  let floorsType1 := block.totalFloors / 2
  let floorsType2 := block.totalFloors - floorsType1
  let totalApartments := floorsType1 * block.apartmentsPerFloorType1 + floorsType2 * block.apartmentsPerFloorType2
  totalApartments * block.maxResidentsPerApartment

/-- Theorem stating the maximum number of residents in the given block of flats -/
theorem max_residents_in_block :
  let block : BlockOfFlats := {
    totalFloors := 12,
    apartmentsPerFloorType1 := 6,
    apartmentsPerFloorType2 := 5,
    maxResidentsPerApartment := 4
  }
  maxResidents block = 264 := by
  sorry

end NUMINAMATH_CALUDE_max_residents_in_block_l1798_179875


namespace NUMINAMATH_CALUDE_least_possible_difference_l1798_179821

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → 
  Odd y → Odd z → 
  (∀ w, w = z - x → w ≥ 9) ∧ (∃ w, w = z - x ∧ w = 9) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_difference_l1798_179821


namespace NUMINAMATH_CALUDE_family_theorem_l1798_179877

structure Family where
  teresa_age : ℕ
  morio_age : ℕ
  morio_age_at_michiko_birth : ℕ
  kenji_michiko_age_diff : ℕ
  yuki_kenji_age_diff : ℕ
  years_after_yuki_adoption_to_anniversary : ℕ
  anniversary_years : ℕ

def years_into_marriage_at_michiko_birth (f : Family) : ℕ := sorry

def teresa_age_at_michiko_birth (f : Family) : ℕ := sorry

theorem family_theorem (f : Family)
  (h1 : f.teresa_age = 59)
  (h2 : f.morio_age = 71)
  (h3 : f.morio_age_at_michiko_birth = 38)
  (h4 : f.kenji_michiko_age_diff = 4)
  (h5 : f.yuki_kenji_age_diff = 3)
  (h6 : f.years_after_yuki_adoption_to_anniversary = 3)
  (h7 : f.anniversary_years = 25) :
  years_into_marriage_at_michiko_birth f = 8 ∧ teresa_age_at_michiko_birth f = 26 := by
  sorry


end NUMINAMATH_CALUDE_family_theorem_l1798_179877


namespace NUMINAMATH_CALUDE_completing_square_solution_l1798_179896

theorem completing_square_solution (x : ℝ) :
  (x^2 - 4*x + 3 = 0) ↔ ((x - 2)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_solution_l1798_179896


namespace NUMINAMATH_CALUDE_equidistant_points_l1798_179804

/-- Two points are equidistant if the larger of their distances to the x and y axes are equal -/
def equidistant (p q : ℝ × ℝ) : Prop :=
  max (|p.1|) (|p.2|) = max (|q.1|) (|q.2|)

theorem equidistant_points :
  (equidistant (-3, 7) (3, -7) ∧ equidistant (-3, 7) (7, 4)) ∧
  (equidistant (-4, 2) (-4, -3) ∧ equidistant (-4, 2) (3, 4)) ∧
  (equidistant (3, 4 + 2) (2 * 2 - 5, 6) ∧ equidistant (3, 4 + 9) (2 * 9 - 5, 6)) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_points_l1798_179804


namespace NUMINAMATH_CALUDE_f_two_plus_f_five_l1798_179800

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_one : f 1 = 4

axiom f_z (z : ℝ) : z ≠ 1 → f z = 3 * z + 6

axiom f_sum (x y : ℝ) : ∃ (a b : ℝ), f (x + y) = f x + f y + a * x * y + b

theorem f_two_plus_f_five : f 2 + f 5 = 33 := by sorry

end NUMINAMATH_CALUDE_f_two_plus_f_five_l1798_179800


namespace NUMINAMATH_CALUDE_largest_two_digit_number_l1798_179897

def digits : Finset Nat := {1, 2, 4, 6}

def valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ n = 10 * a + b

theorem largest_two_digit_number :
  ∀ n, valid_number n → n ≤ 64 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_number_l1798_179897


namespace NUMINAMATH_CALUDE_equality_of_sqrt_five_terms_l1798_179824

theorem equality_of_sqrt_five_terms 
  (a b c d : ℚ) 
  (h : a + b * Real.sqrt 5 = c + d * Real.sqrt 5) : 
  a = c ∧ b = d := by
sorry

end NUMINAMATH_CALUDE_equality_of_sqrt_five_terms_l1798_179824


namespace NUMINAMATH_CALUDE_nina_running_distance_l1798_179895

theorem nina_running_distance :
  let d1 : ℚ := 0.08333333333333333
  let d2 : ℚ := 0.08333333333333333
  let d3 : ℚ := 0.6666666666666666
  d1 + d2 + d3 = 0.8333333333333333 := by sorry

end NUMINAMATH_CALUDE_nina_running_distance_l1798_179895


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1798_179842

/-- The area of the quadrilateral formed by three coplanar squares -/
theorem quadrilateral_area (s₁ s₂ s₃ : ℝ) (hs₁ : s₁ = 3) (hs₂ : s₂ = 5) (hs₃ : s₃ = 7) : 
  let h₁ := s₁ * (s₃ / (s₁ + s₂ + s₃))
  let h₂ := (s₁ + s₂) * (s₃ / (s₁ + s₂ + s₃))
  (h₁ + h₂) * s₂ / 2 = 12.825 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1798_179842


namespace NUMINAMATH_CALUDE_cindy_pen_addition_l1798_179837

theorem cindy_pen_addition (initial_pens : ℕ) (mike_pens : ℕ) (sharon_pens : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 20)
  (h2 : mike_pens = 22)
  (h3 : sharon_pens = 19)
  (h4 : final_pens = 65) :
  final_pens - (initial_pens + mike_pens - sharon_pens) = 42 :=
by sorry

end NUMINAMATH_CALUDE_cindy_pen_addition_l1798_179837


namespace NUMINAMATH_CALUDE_pool_time_ratio_l1798_179808

/-- The ratio of George's time to Elaine's time in the pool --/
def time_ratio (jerry_time elaine_time george_time : ℚ) : ℚ × ℚ :=
  (george_time, elaine_time)

theorem pool_time_ratio :
  ∀ (jerry_time elaine_time george_time total_time : ℚ),
    jerry_time = 3 →
    elaine_time = 2 * jerry_time →
    total_time = 11 →
    total_time = jerry_time + elaine_time + george_time →
    time_ratio jerry_time elaine_time george_time = (1, 3) := by
  sorry

#check pool_time_ratio

end NUMINAMATH_CALUDE_pool_time_ratio_l1798_179808


namespace NUMINAMATH_CALUDE_greatest_k_value_l1798_179893

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 72) →
  k ≤ 2 * Real.sqrt 26 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l1798_179893


namespace NUMINAMATH_CALUDE_quadratic_roots_l1798_179879

theorem quadratic_roots (m : ℝ) : 
  (∃! x : ℝ, (m + 2) * x^2 - 2 * (m + 1) * x + m = 0) →
  (∃ x : ℝ, (m + 1) * x^2 - 2 * m * x + (m - 2) = 0 ∧
             ∀ y : ℝ, (m + 1) * y^2 - 2 * m * y + (m - 2) = 0 → y = x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1798_179879


namespace NUMINAMATH_CALUDE_students_liking_both_soda_and_coke_l1798_179834

/-- Given a school with the following conditions:
  - Total number of students: 500
  - Students who like soda: 337
  - Students who like coke: 289
  - Students who neither like soda nor coke: 56
  Prove that the number of students who like both soda and coke is 182. -/
theorem students_liking_both_soda_and_coke 
  (total : ℕ) (soda : ℕ) (coke : ℕ) (neither : ℕ) 
  (h_total : total = 500)
  (h_soda : soda = 337)
  (h_coke : coke = 289)
  (h_neither : neither = 56) :
  soda + coke - total + neither = 182 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_soda_and_coke_l1798_179834


namespace NUMINAMATH_CALUDE_kylies_coins_l1798_179839

theorem kylies_coins (piggy_bank : ℕ) (brother : ℕ) (father : ℕ) (gave_away : ℕ) (left : ℕ) : 
  piggy_bank = 15 → 
  brother = 13 → 
  gave_away = 21 → 
  left = 15 → 
  piggy_bank + brother + father - gave_away = left → 
  father = 8 := by
sorry

end NUMINAMATH_CALUDE_kylies_coins_l1798_179839


namespace NUMINAMATH_CALUDE_variance_best_for_stability_l1798_179803

-- Define a type for math test scores
def MathScore := ℝ

-- Define a type for a set of consecutive math test scores
def ConsecutiveScores := List MathScore

-- Define a function to calculate variance
noncomputable def variance (scores : ConsecutiveScores) : ℝ := sorry

-- Define a function to calculate other statistical measures
noncomputable def otherMeasure (scores : ConsecutiveScores) : ℝ := sorry

-- Define a function to measure stability
noncomputable def stability (scores : ConsecutiveScores) : ℝ := sorry

-- Theorem stating that variance is the most appropriate measure for stability
theorem variance_best_for_stability (scores : ConsecutiveScores) :
  ∀ (other : ConsecutiveScores → ℝ), other ≠ variance →
  |stability scores - variance scores| < |stability scores - other scores| :=
sorry

end NUMINAMATH_CALUDE_variance_best_for_stability_l1798_179803


namespace NUMINAMATH_CALUDE_box_2_neg2_3_l1798_179866

def box (a b c : ℤ) : ℚ := (a ^ b) - (b ^ c) + (c ^ a)

theorem box_2_neg2_3 : box 2 (-2) 3 = 5 / 4 := by sorry

end NUMINAMATH_CALUDE_box_2_neg2_3_l1798_179866


namespace NUMINAMATH_CALUDE_probability_three_fourths_radius_l1798_179884

/-- A circle concentric with and outside a square --/
structure ConcentricCircleSquare where
  squareSideLength : ℝ
  circleRadius : ℝ
  squareSideLength_pos : 0 < squareSideLength
  circleRadius_gt_squareSideLength : squareSideLength < circleRadius

/-- The probability of seeing two sides of the square from a random point on the circle --/
def probabilityTwoSides (c : ConcentricCircleSquare) : ℝ := sorry

theorem probability_three_fourths_radius (c : ConcentricCircleSquare) 
  (h : c.squareSideLength = 4) 
  (prob : probabilityTwoSides c = 3/4) : 
  c.circleRadius = 8 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_three_fourths_radius_l1798_179884


namespace NUMINAMATH_CALUDE_rational_roots_quadratic_l1798_179861

theorem rational_roots_quadratic (r : ℚ) : 
  (∃ (n : ℤ), 2 * r = n) →
  (∃ (x : ℚ), (r^2 + r) * x^2 + 4 - r^2 = 0) →
  r = 2 ∨ r = -2 ∨ r = -4 := by
sorry

end NUMINAMATH_CALUDE_rational_roots_quadratic_l1798_179861


namespace NUMINAMATH_CALUDE_solve_equation_l1798_179827

theorem solve_equation : 42 / (7 - 3/7) = 147/23 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1798_179827


namespace NUMINAMATH_CALUDE_power_division_equality_l1798_179888

theorem power_division_equality : 8^15 / 64^6 = 512 :=
by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l1798_179888


namespace NUMINAMATH_CALUDE_smallest_multiple_l1798_179883

theorem smallest_multiple (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k) ∧ 
  n % 101 = 3 ∧ 
  (∀ m : ℕ, m < n → ¬((∃ k : ℕ, m = 17 * k) ∧ m % 101 = 3)) → 
  n = 306 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1798_179883


namespace NUMINAMATH_CALUDE_school_arrival_time_l1798_179805

/-- Calculates how early (in minutes) a boy arrives at school on the second day given the following conditions:
  * The distance between home and school is 2.5 km
  * On the first day, he travels at 5 km/hr and arrives 5 minutes late
  * On the second day, he travels at 10 km/hr and arrives early
-/
theorem school_arrival_time (distance : ℝ) (speed1 speed2 : ℝ) (late_time : ℝ) : 
  distance = 2.5 ∧ 
  speed1 = 5 ∧ 
  speed2 = 10 ∧ 
  late_time = 5 → 
  (distance / speed1 * 60 - late_time) - (distance / speed2 * 60) = 10 := by
  sorry

#check school_arrival_time

end NUMINAMATH_CALUDE_school_arrival_time_l1798_179805


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l1798_179871

theorem factor_implies_d_value (d : ℚ) : 
  (∀ x : ℚ, (x - 5) ∣ (d*x^4 + 19*x^3 - 10*d*x^2 + 45*x - 90)) → 
  d = -502/75 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l1798_179871


namespace NUMINAMATH_CALUDE_marble_217_is_red_l1798_179817

/-- Represents the color of a marble -/
inductive MarbleColor
| Red
| Blue
| Green

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  let cycleLength := 15
  let position := n % cycleLength
  if position ≤ 6 then MarbleColor.Red
  else if position ≤ 11 then MarbleColor.Blue
  else MarbleColor.Green

/-- Theorem stating that the 217th marble is red -/
theorem marble_217_is_red : marbleColor 217 = MarbleColor.Red := by
  sorry


end NUMINAMATH_CALUDE_marble_217_is_red_l1798_179817


namespace NUMINAMATH_CALUDE_solve_system_for_x_l1798_179816

theorem solve_system_for_x (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 8) 
  (eq2 : 2 * x + 3 * y = 1) : 
  x = 28 / 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_for_x_l1798_179816


namespace NUMINAMATH_CALUDE_intersection_line_equation_l1798_179830

/-- Definition of line l1 -/
def l1 (x y : ℝ) : Prop := x - y + 3 = 0

/-- Definition of line l2 -/
def l2 (x y : ℝ) : Prop := 2*x + y = 0

/-- Definition of the intersection point of l1 and l2 -/
def intersection_point (x y : ℝ) : Prop := l1 x y ∧ l2 x y

/-- Definition of a line with inclination angle π/3 passing through a point -/
def line_with_inclination (x₀ y₀ x y : ℝ) : Prop :=
  y - y₀ = Real.sqrt 3 * (x - x₀)

/-- The main theorem -/
theorem intersection_line_equation :
  ∃ x₀ y₀ : ℝ, intersection_point x₀ y₀ ∧
  ∀ x y : ℝ, line_with_inclination x₀ y₀ x y ↔ Real.sqrt 3 * x - y + Real.sqrt 3 + 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l1798_179830


namespace NUMINAMATH_CALUDE_towns_distance_l1798_179868

/-- Given a map distance and a scale, calculate the actual distance between two towns. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem stating that for a map distance of 20 inches and a scale of 1 inch = 10 miles,
    the actual distance between the towns is 200 miles. -/
theorem towns_distance :
  let map_distance : ℝ := 20
  let scale : ℝ := 10
  actual_distance map_distance scale = 200 := by
sorry

end NUMINAMATH_CALUDE_towns_distance_l1798_179868


namespace NUMINAMATH_CALUDE_shaded_area_formula_l1798_179831

/-- An equilateral triangle inscribed in a circle -/
structure InscribedTriangle where
  /-- Side length of the equilateral triangle -/
  side_length : ℝ
  /-- The triangle is inscribed in a circle -/
  inscribed : Bool
  /-- Two vertices of the triangle are endpoints of a circle diameter -/
  diameter_endpoints : Bool

/-- The shaded area outside the triangle but inside the circle -/
def shaded_area (t : InscribedTriangle) : ℝ := sorry

/-- Theorem stating the shaded area for a specific inscribed triangle -/
theorem shaded_area_formula (t : InscribedTriangle) 
  (h1 : t.side_length = 10)
  (h2 : t.inscribed = true)
  (h3 : t.diameter_endpoints = true) :
  shaded_area t = (50 * Real.pi / 3) - 25 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_shaded_area_formula_l1798_179831


namespace NUMINAMATH_CALUDE_farm_animal_count_l1798_179859

/-- Represents the count of animals on a farm -/
structure FarmCount where
  chickens : ℕ
  ducks : ℕ
  geese : ℕ
  quails : ℕ
  turkeys : ℕ
  cow_sheds : ℕ
  cows_per_shed : ℕ
  pigs : ℕ

/-- Calculates the total number of animals on the farm -/
def total_animals (farm : FarmCount) : ℕ :=
  farm.chickens + farm.ducks + farm.geese + farm.quails + farm.turkeys +
  (farm.cow_sheds * farm.cows_per_shed) + farm.pigs

/-- Theorem stating that the total number of animals on the given farm is 219 -/
theorem farm_animal_count :
  let farm := FarmCount.mk 60 40 20 50 10 3 8 15
  total_animals farm = 219 := by
  sorry

#eval total_animals (FarmCount.mk 60 40 20 50 10 3 8 15)

end NUMINAMATH_CALUDE_farm_animal_count_l1798_179859


namespace NUMINAMATH_CALUDE_maria_candy_l1798_179872

/-- Calculates the remaining candy pieces after eating some. -/
def remaining_candy (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem: Maria has 3 pieces of candy left -/
theorem maria_candy : remaining_candy 67 64 = 3 := by
  sorry

end NUMINAMATH_CALUDE_maria_candy_l1798_179872


namespace NUMINAMATH_CALUDE_amount_fraction_is_one_third_l1798_179851

/-- Represents the amounts received by A, B, and C in dollars -/
structure Amounts where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (x : Amounts) (total : ℝ) (fraction : ℝ) : Prop :=
  x.a + x.b + x.c = total ∧
  x.a = fraction * (x.b + x.c) ∧
  x.b = (2 / 7) * (x.a + x.c) ∧
  x.a = x.b + 10

theorem amount_fraction_is_one_third :
  ∃ (x : Amounts) (fraction : ℝ),
    satisfies_conditions x 360 fraction ∧ fraction = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_amount_fraction_is_one_third_l1798_179851


namespace NUMINAMATH_CALUDE_circle_graph_proportions_l1798_179891

theorem circle_graph_proportions :
  ∀ (total : ℝ) (blue : ℝ),
    blue > 0 →
    total = blue + 3 * blue + 0.5 * blue →
    (3 * blue / total = 2 / 3) ∧
    (blue / total = 1 / 4.5) ∧
    (0.5 * blue / total = 1 / 9) := by
  sorry

end NUMINAMATH_CALUDE_circle_graph_proportions_l1798_179891


namespace NUMINAMATH_CALUDE_wanda_walking_distance_l1798_179807

/-- The distance in miles Wanda walks to school one way -/
def distance_to_school : ℝ := 0.5

/-- The number of round trips Wanda makes per day -/
def round_trips_per_day : ℕ := 2

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 4

/-- The total distance Wanda walks after the given number of weeks -/
def total_distance : ℝ :=
  distance_to_school * 2 * round_trips_per_day * school_days_per_week * num_weeks

theorem wanda_walking_distance :
  total_distance = 40 := by sorry

end NUMINAMATH_CALUDE_wanda_walking_distance_l1798_179807


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1798_179844

/-- A quadratic equation with coefficients m and n has exactly one real root if and only if m > 0 and n = 9m^2 -/
theorem quadratic_one_root (m n : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + n = 0) ∧ (m > 0) ∧ (n > 0) ↔ (m > 0) ∧ (n = 9*m^2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1798_179844


namespace NUMINAMATH_CALUDE_min_value_theorem_l1798_179802

theorem min_value_theorem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 8) 
  (h2 : e * f * g * h = 16) : 
  (2 * a * e)^2 + (2 * b * f)^2 + (2 * c * g)^2 + (2 * d * h)^2 ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1798_179802


namespace NUMINAMATH_CALUDE_face_mask_profit_l1798_179860

/-- Calculates the total profit from selling face masks --/
def calculate_profit (num_boxes : ℕ) (price_per_mask : ℚ) (masks_per_box : ℕ) (total_cost : ℚ) : ℚ :=
  num_boxes * price_per_mask * masks_per_box - total_cost

/-- Proves that the total profit is $15 given the specified conditions --/
theorem face_mask_profit :
  let num_boxes : ℕ := 3
  let price_per_mask : ℚ := 1/2
  let masks_per_box : ℕ := 20
  let total_cost : ℚ := 15
  calculate_profit num_boxes price_per_mask masks_per_box total_cost = 15 := by
  sorry

#eval calculate_profit 3 (1/2) 20 15

end NUMINAMATH_CALUDE_face_mask_profit_l1798_179860


namespace NUMINAMATH_CALUDE_postage_for_5_25_ounces_l1798_179801

/-- Calculates the postage cost for a letter given its weight and postage rates. -/
def calculate_postage (weight : ℚ) (base_rate : ℕ) (additional_rate : ℕ) : ℚ :=
  let additional_weight := max (weight - 1) 0
  let additional_charges := ⌈additional_weight⌉
  (base_rate + additional_charges * additional_rate) / 100

/-- Theorem stating that the postage for a 5.25 ounce letter is $1.60 under the given rates. -/
theorem postage_for_5_25_ounces :
  calculate_postage (5.25 : ℚ) 35 25 = (1.60 : ℚ) := by
  sorry

#eval calculate_postage (5.25 : ℚ) 35 25

end NUMINAMATH_CALUDE_postage_for_5_25_ounces_l1798_179801


namespace NUMINAMATH_CALUDE_sum_of_digits_l1798_179865

/-- Given two single-digit numbers a and b, if ab + ba = 202, then a + b = 12 -/
theorem sum_of_digits (a b : ℕ) : 
  a < 10 → b < 10 → (10 * a + b) + (10 * b + a) = 202 → a + b = 12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_l1798_179865


namespace NUMINAMATH_CALUDE_badminton_medals_count_l1798_179829

theorem badminton_medals_count (total_medals : ℕ) (track_medals : ℕ) : 
  total_medals = 20 →
  track_medals = 5 →
  total_medals = track_medals + 2 * track_medals + (total_medals - track_medals - 2 * track_medals) →
  (total_medals - track_medals - 2 * track_medals) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_badminton_medals_count_l1798_179829


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l1798_179812

/-- Calculates the monthly income given the percentage saved and the amount saved -/
def calculate_income (percent_saved : ℚ) (amount_saved : ℚ) : ℚ :=
  amount_saved / percent_saved

/-- The percentage of income spent on various categories -/
def total_expenses : ℚ := 35 + 18 + 6 + 11 + 12 + 5 + 7

/-- The percentage of income saved -/
def percent_saved : ℚ := 100 - total_expenses

/-- The amount saved in Rupees -/
def amount_saved : ℚ := 12500

theorem monthly_income_calculation :
  calculate_income percent_saved amount_saved = 208333.33 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l1798_179812


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1798_179881

/-- A function that checks if a natural number is a palindrome in a given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- A function that converts a natural number from one base to another. -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- A function that returns the number of digits of a natural number in a given base. -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ n : ℕ,
  (isPalindrome n 2 ∧ numDigits n 2 = 5) →
  (∃ b : ℕ, b > 2 ∧ isPalindrome (baseConvert n 2 b) b ∧ numDigits (baseConvert n 2 b) b = 3) →
  n ≥ 17 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1798_179881


namespace NUMINAMATH_CALUDE_field_division_l1798_179823

theorem field_division (total_area smaller_area : ℝ) (h1 : total_area = 700) (h2 : smaller_area = 315) :
  ∃ (larger_area X : ℝ),
    larger_area + smaller_area = total_area ∧
    larger_area - smaller_area = (1 / 5) * X ∧
    X = 350 := by
  sorry

end NUMINAMATH_CALUDE_field_division_l1798_179823


namespace NUMINAMATH_CALUDE_distance_between_cities_l1798_179869

/-- The distance between two cities given the speeds and times of two cars traveling between them -/
theorem distance_between_cities (meeting_time : ℝ) (car_b_speed : ℝ) (car_a_remaining_time : ℝ) :
  let car_a_speed := car_b_speed * meeting_time / car_a_remaining_time
  let total_distance := (car_a_speed + car_b_speed) * meeting_time
  meeting_time = 6 ∧ car_b_speed = 69 ∧ car_a_remaining_time = 4 →
  total_distance = (69 * 6 / 4 + 69) * 6 := by
sorry

end NUMINAMATH_CALUDE_distance_between_cities_l1798_179869


namespace NUMINAMATH_CALUDE_difference_ones_zeros_is_six_l1798_179846

-- Define the number in base 10
def base_10_num : Nat := 253

-- Define a function to convert a number to its binary representation
def to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

-- Define functions to count zeros and ones in a binary representation
def count_zeros (binary : List Bool) : Nat :=
  binary.filter (· = false) |>.length

def count_ones (binary : List Bool) : Nat :=
  binary.filter (· = true) |>.length

-- Theorem statement
theorem difference_ones_zeros_is_six :
  let binary := to_binary base_10_num
  let y := count_ones binary
  let x := count_zeros binary
  y - x = 6 := by sorry

end NUMINAMATH_CALUDE_difference_ones_zeros_is_six_l1798_179846


namespace NUMINAMATH_CALUDE_C_nec_not_suff_A_l1798_179870

-- Define the propositions
variable (A B C : Prop)

-- Define the relationships between A, B, and C
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_nec_and_suff_C : (B ↔ C)

-- State the theorem to be proved
theorem C_nec_not_suff_A : (C → A) ∧ ¬(A → C) := by sorry

end NUMINAMATH_CALUDE_C_nec_not_suff_A_l1798_179870


namespace NUMINAMATH_CALUDE_bike_five_times_a_week_l1798_179856

/-- Given Onur's daily biking distance, Hanil's additional distance, and their total weekly distance,
    calculate the number of days they bike per week. -/
def biking_days_per_week (onur_daily : ℕ) (hanil_additional : ℕ) (total_weekly : ℕ) : ℕ :=
  total_weekly / (onur_daily + (onur_daily + hanil_additional))

/-- Theorem stating that under the given conditions, Onur and Hanil bike 5 times a week. -/
theorem bike_five_times_a_week :
  biking_days_per_week 250 40 2700 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bike_five_times_a_week_l1798_179856


namespace NUMINAMATH_CALUDE_missing_number_implies_next_prime_l1798_179826

theorem missing_number_implies_next_prime (n : ℕ) : n > 3 →
  (∀ r s : ℕ, r ≥ 3 ∧ s ≥ 3 → n ≠ r * s - (r + s)) →
  Nat.Prime (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_implies_next_prime_l1798_179826


namespace NUMINAMATH_CALUDE_expression_evaluation_l1798_179864

theorem expression_evaluation :
  let x : ℝ := 16
  let expr := (2 + x * (2 + Real.sqrt x) - 4^2) / (Real.sqrt x - 4 + x^2)
  expr = 41 / 128 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1798_179864


namespace NUMINAMATH_CALUDE_sum_largest_triangles_geq_twice_area_l1798_179843

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and properties to define a convex polygon
  -- This is a simplified representation
  vertices : Set ℝ × ℝ
  is_convex : sorry

/-- The area of a polygon -/
def area (P : ConvexPolygon) : ℝ := sorry

/-- The largest triangle area for a given side of the polygon -/
def largest_triangle_area (P : ConvexPolygon) (side : ℝ × ℝ × ℝ × ℝ) : ℝ := sorry

/-- The sum of largest triangle areas for all sides of the polygon -/
def sum_largest_triangle_areas (P : ConvexPolygon) : ℝ := sorry

/-- Theorem: The sum of the areas of the largest triangles formed within P, 
    each having one side coinciding with a side of P, 
    is at least twice the area of P -/
theorem sum_largest_triangles_geq_twice_area (P : ConvexPolygon) :
  sum_largest_triangle_areas P ≥ 2 * area P := by sorry

end NUMINAMATH_CALUDE_sum_largest_triangles_geq_twice_area_l1798_179843


namespace NUMINAMATH_CALUDE_parabola_point_shift_l1798_179848

/-- Given a point P(m,n) on the parabola y = ax^2 (a ≠ 0), 
    prove that (m-1, n) lies on y = a(x+1)^2 -/
theorem parabola_point_shift (a m n : ℝ) (h1 : a ≠ 0) (h2 : n = a * m^2) :
  n = a * ((m - 1) + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_shift_l1798_179848


namespace NUMINAMATH_CALUDE_adjusted_work_schedule_earnings_l1798_179818

/-- Proves that the adjusted work schedule results in the same total earnings --/
theorem adjusted_work_schedule_earnings (initial_hours_per_week : ℝ) 
  (initial_weeks : ℕ) (missed_weeks : ℕ) (total_earnings : ℝ) 
  (adjusted_hours_per_week : ℝ) :
  initial_hours_per_week = 25 →
  initial_weeks = 15 →
  missed_weeks = 3 →
  total_earnings = 3750 →
  adjusted_hours_per_week = 31.25 →
  (initial_weeks - missed_weeks : ℝ) * adjusted_hours_per_week = initial_weeks * initial_hours_per_week :=
by sorry

end NUMINAMATH_CALUDE_adjusted_work_schedule_earnings_l1798_179818


namespace NUMINAMATH_CALUDE_test_probabilities_l1798_179841

theorem test_probabilities (p_first : ℝ) (p_second : ℝ) (p_both : ℝ) 
  (h1 : p_first = 0.7)
  (h2 : p_second = 0.55)
  (h3 : p_both = 0.45) :
  1 - (p_first + p_second - p_both) = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_test_probabilities_l1798_179841


namespace NUMINAMATH_CALUDE_line_points_relation_l1798_179882

/-- Given a line x = 5y + 5 passing through points (a, n) and (a + 2, n + 0.4),
    prove that a = 5n + 5 -/
theorem line_points_relation (a n : ℝ) : 
  (a = 5 * n + 5 ∧ (a + 2) = 5 * (n + 0.4) + 5) → a = 5 * n + 5 := by
  sorry

end NUMINAMATH_CALUDE_line_points_relation_l1798_179882


namespace NUMINAMATH_CALUDE_abab_baba_divisible_by_three_l1798_179820

theorem abab_baba_divisible_by_three (A B : ℕ) :
  A ≠ B →
  A ∈ Finset.range 10 →
  B ∈ Finset.range 10 →
  A ≠ 0 →
  B ≠ 0 →
  ∃ k : ℤ, (1010 * A + 101 * B) - (101 * A + 1010 * B) = 3 * k :=
by sorry

end NUMINAMATH_CALUDE_abab_baba_divisible_by_three_l1798_179820


namespace NUMINAMATH_CALUDE_puppies_weight_difference_l1798_179889

/-- The weight difference between two dogs after a year, given their initial weights and weight gain percentage -/
def weight_difference (labrador_initial : ℝ) (dachshund_initial : ℝ) (weight_gain_percentage : ℝ) : ℝ :=
  (labrador_initial * (1 + weight_gain_percentage)) - (dachshund_initial * (1 + weight_gain_percentage))

/-- Theorem stating that the weight difference between the labrador and dachshund puppies after a year is 35 pounds -/
theorem puppies_weight_difference :
  weight_difference 40 12 0.25 = 35 := by
  sorry

end NUMINAMATH_CALUDE_puppies_weight_difference_l1798_179889


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1798_179894

open Set

theorem complement_intersection_theorem (M N : Set ℝ) :
  M = {x | x > 1} →
  N = {x | |x| ≤ 2} →
  (𝓤 \ M) ∩ N = Icc (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1798_179894


namespace NUMINAMATH_CALUDE_power_multiplication_l1798_179852

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1798_179852


namespace NUMINAMATH_CALUDE_smallest_area_special_square_l1798_179867

/-- A square with vertices on a line and parabola -/
structure SpecialSquare where
  -- One pair of opposite vertices lie on this line
  line : Real → Real
  -- The other pair of opposite vertices lie on this parabola
  parabola : Real → Real
  -- The line is y = -2x + 17
  line_eq : line = fun x => -2 * x + 17
  -- The parabola is y = x^2 - 2
  parabola_eq : parabola = fun x => x^2 - 2

/-- The smallest possible area of a SpecialSquare is 160 -/
theorem smallest_area_special_square (s : SpecialSquare) :
  ∃ (area : Real), area = 160 ∧ 
  (∀ (other_area : Real), other_area ≥ area) :=
by sorry

end NUMINAMATH_CALUDE_smallest_area_special_square_l1798_179867


namespace NUMINAMATH_CALUDE_salesman_commission_problem_l1798_179810

/-- A problem about a salesman's commission schemes -/
theorem salesman_commission_problem 
  (old_commission_rate : ℝ)
  (fixed_salary : ℝ)
  (sales_threshold : ℝ)
  (total_sales : ℝ)
  (remuneration_difference : ℝ)
  (h1 : old_commission_rate = 0.05)
  (h2 : fixed_salary = 1000)
  (h3 : sales_threshold = 4000)
  (h4 : total_sales = 12000)
  (h5 : remuneration_difference = 600) :
  ∃ new_commission_rate : ℝ,
    new_commission_rate * (total_sales - sales_threshold) + fixed_salary = 
    old_commission_rate * total_sales + remuneration_difference ∧
    new_commission_rate = 0.025 := by
  sorry

end NUMINAMATH_CALUDE_salesman_commission_problem_l1798_179810


namespace NUMINAMATH_CALUDE_parallelogram_angles_l1798_179863

/-- A parallelogram with angles measured in degrees -/
structure Parallelogram where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_360 : A + B + C + D = 360
  opposite_equal_AC : A = C
  opposite_equal_BD : B = D

/-- Theorem: In a parallelogram ABCD where angle A measures 125°, 
    the measures of angles B, C, and D are 55°, 125°, and 55° respectively. -/
theorem parallelogram_angles (p : Parallelogram) (h : p.A = 125) : 
  p.B = 55 ∧ p.C = 125 ∧ p.D = 55 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_angles_l1798_179863


namespace NUMINAMATH_CALUDE_bread_cost_is_30_cents_l1798_179822

/-- The cost of a sandwich in dollars -/
def sandwich_price : ℚ := 1.5

/-- The cost of a slice of ham in dollars -/
def ham_cost : ℚ := 0.25

/-- The cost of a slice of cheese in dollars -/
def cheese_cost : ℚ := 0.35

/-- The total cost to make a sandwich in dollars -/
def total_cost : ℚ := 0.9

/-- The number of slices of bread in a sandwich -/
def bread_slices : ℕ := 2

/-- Theorem: The cost of a slice of bread is $0.30 -/
theorem bread_cost_is_30_cents :
  (total_cost - ham_cost - cheese_cost) / bread_slices = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_is_30_cents_l1798_179822


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l1798_179876

/-- The number of team members in the relay race -/
def team_size : ℕ := 5

/-- The position in which Jordan (the fastest runner) must run -/
def jordan_position : ℕ := 5

/-- The number of runners that need to be arranged -/
def runners_to_arrange : ℕ := team_size - 1

/-- Calculates the number of permutations of n elements -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

theorem relay_race_arrangements :
  permutations runners_to_arrange = 24 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l1798_179876


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1798_179890

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = -2 + Real.sqrt 2 ∧ x₂ = -2 - Real.sqrt 2) ∧ 
  (∀ x : ℝ, x^2 + 4*x + 2 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1798_179890


namespace NUMINAMATH_CALUDE_reciprocal_location_l1798_179857

theorem reciprocal_location (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a^2 + b^2 < 1) :
  let F := Complex.mk a b
  let recip := F⁻¹
  (Complex.re recip > 0) ∧ (Complex.im recip > 0) ∧ (Complex.abs recip > 1) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_location_l1798_179857


namespace NUMINAMATH_CALUDE_third_number_in_proportion_l1798_179886

theorem third_number_in_proportion (x : ℝ) (h : x = 3) : 
  ∃ y : ℝ, (x + 1) / (x + 5) = (x + 5) / (x + y) → y = 13 := by
  sorry

end NUMINAMATH_CALUDE_third_number_in_proportion_l1798_179886


namespace NUMINAMATH_CALUDE_correct_popularity_order_l1798_179847

/-- Represents the activities available for the sports day --/
inductive Activity
| dodgeball
| chessTournament
| track
| swimming

/-- Returns the fraction of students preferring a given activity --/
def preference (a : Activity) : Rat :=
  match a with
  | Activity.dodgeball => 3/8
  | Activity.chessTournament => 9/24
  | Activity.track => 5/16
  | Activity.swimming => 1/3

/-- Compares two activities based on their popularity --/
def morePopularThan (a b : Activity) : Prop :=
  preference a > preference b

/-- States that the given order of activities is correct based on popularity --/
theorem correct_popularity_order :
  morePopularThan Activity.swimming Activity.dodgeball ∧
  morePopularThan Activity.dodgeball Activity.chessTournament ∧
  morePopularThan Activity.chessTournament Activity.track :=
by sorry

end NUMINAMATH_CALUDE_correct_popularity_order_l1798_179847


namespace NUMINAMATH_CALUDE_bananas_needed_l1798_179853

def yogurt_count : ℕ := 5
def slices_per_yogurt : ℕ := 8
def slices_per_banana : ℕ := 10

theorem bananas_needed : 
  (yogurt_count * slices_per_yogurt + slices_per_banana - 1) / slices_per_banana = 4 := by
  sorry

end NUMINAMATH_CALUDE_bananas_needed_l1798_179853


namespace NUMINAMATH_CALUDE_alan_tickets_l1798_179833

theorem alan_tickets (total : ℕ) (alan : ℕ) (marcy : ℕ) 
  (h1 : total = 150)
  (h2 : alan + marcy = total)
  (h3 : marcy = 5 * alan - 6) :
  alan = 26 := by
sorry

end NUMINAMATH_CALUDE_alan_tickets_l1798_179833


namespace NUMINAMATH_CALUDE_sin_three_pi_halves_l1798_179809

theorem sin_three_pi_halves : Real.sin (3 * π / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_halves_l1798_179809


namespace NUMINAMATH_CALUDE_thirteen_travel_methods_l1798_179811

/-- The number of different methods to travel from Place A to Place B -/
def travel_methods (bus_services train_services ship_services : ℕ) : ℕ :=
  bus_services + train_services + ship_services

/-- Theorem: There are 13 different methods to travel from Place A to Place B -/
theorem thirteen_travel_methods :
  travel_methods 8 3 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_travel_methods_l1798_179811


namespace NUMINAMATH_CALUDE_min_distinct_sums_products_l1798_179885

theorem min_distinct_sums_products (a b c d : ℤ) (h1 : a < b) (h2 : b < c) (h3 : c < d) :
  let sums := {a + b, a + c, a + d, b + c, b + d, c + d}
  let products := {a * b, a * c, a * d, b * c, b * d, c * d}
  Finset.card (sums ∪ products) ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_distinct_sums_products_l1798_179885


namespace NUMINAMATH_CALUDE_line_through_focus_line_intersects_ellipse_l1798_179898

/-- The equation of the line l -/
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 4

/-- The equation of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

/-- The x-coordinate of the left focus of the ellipse -/
def left_focus : ℝ := -2

/-- Theorem: When the line passes through the left focus of the ellipse, k = 2 -/
theorem line_through_focus (k : ℝ) : 
  line k left_focus = 0 → k = 2 :=
sorry

/-- Theorem: The line intersects the ellipse if and only if k is in the specified range -/
theorem line_intersects_ellipse (k : ℝ) : 
  (∃ x y, ellipse x y ∧ y = line k x) ↔ k ≤ -Real.sqrt 3 ∨ k ≥ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_line_through_focus_line_intersects_ellipse_l1798_179898


namespace NUMINAMATH_CALUDE_max_abs_z5_l1798_179858

open Complex

theorem max_abs_z5 (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h1 : abs z₁ ≤ 1) (h2 : abs z₂ ≤ 1)
  (h3 : abs (2 * z₃ - (z₁ + z₂)) ≤ abs (z₁ - z₂))
  (h4 : abs (2 * z₄ - (z₁ + z₂)) ≤ abs (z₁ - z₂))
  (h5 : abs (2 * z₅ - (z₃ + z₄)) ≤ abs (z₃ - z₄)) :
  abs z₅ ≤ Real.sqrt 3 ∧ ∃ z₁ z₂ z₃ z₄ z₅ : ℂ, abs z₅ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z5_l1798_179858


namespace NUMINAMATH_CALUDE_least_common_duration_l1798_179892

/-- Represents a business partner -/
structure Partner where
  investment : ℚ
  duration : ℕ

/-- Represents the business venture -/
structure BusinessVenture where
  p : Partner
  q : Partner
  r : Partner
  investmentRatio : Fin 3 → ℚ
  profitRatio : Fin 3 → ℚ

/-- The profit is proportional to the product of investment and duration -/
def profitProportional (bv : BusinessVenture) : Prop :=
  ∃ (k : ℚ), k > 0 ∧
    bv.profitRatio 0 = k * bv.p.investment * bv.p.duration ∧
    bv.profitRatio 1 = k * bv.q.investment * bv.q.duration ∧
    bv.profitRatio 2 = k * bv.r.investment * bv.r.duration

/-- The main theorem -/
theorem least_common_duration (bv : BusinessVenture) 
    (h1 : bv.investmentRatio = ![7, 5, 3])
    (h2 : bv.profitRatio = ![7, 10, 6])
    (h3 : bv.p.duration = 8)
    (h4 : bv.q.duration = 6)
    (h5 : profitProportional bv) :
    bv.r.duration = 6 := by
  sorry

end NUMINAMATH_CALUDE_least_common_duration_l1798_179892


namespace NUMINAMATH_CALUDE_apple_packing_difference_is_500_l1798_179862

/-- Represents the apple packing scenario over two weeks -/
structure ApplePacking where
  apples_per_box : ℕ
  boxes_per_day : ℕ
  days_per_week : ℕ
  total_apples_two_weeks : ℕ

/-- Calculates the difference in daily apple packing between the first and second week -/
def daily_packing_difference (ap : ApplePacking) : ℕ :=
  let normal_daily_packing := ap.apples_per_box * ap.boxes_per_day
  let first_week_total := normal_daily_packing * ap.days_per_week
  let second_week_total := ap.total_apples_two_weeks - first_week_total
  let second_week_daily_average := second_week_total / ap.days_per_week
  normal_daily_packing - second_week_daily_average

/-- Theorem stating the difference in daily apple packing is 500 -/
theorem apple_packing_difference_is_500 :
  ∀ (ap : ApplePacking),
    ap.apples_per_box = 40 ∧
    ap.boxes_per_day = 50 ∧
    ap.days_per_week = 7 ∧
    ap.total_apples_two_weeks = 24500 →
    daily_packing_difference ap = 500 := by
  sorry

end NUMINAMATH_CALUDE_apple_packing_difference_is_500_l1798_179862


namespace NUMINAMATH_CALUDE_multiplication_digits_sum_l1798_179828

theorem multiplication_digits_sum (x y : Nat) : 
  x < 10 → y < 10 → (30 + x) * (10 * y + 4) = 136 → x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_digits_sum_l1798_179828


namespace NUMINAMATH_CALUDE_white_washing_cost_calculation_l1798_179838

/-- Calculate the cost of white washing a room's walls --/
def white_washing_cost (room_length room_width room_height : ℝ)
                       (door_height door_width : ℝ)
                       (window_height window_width : ℝ)
                       (num_windows : ℕ)
                       (cost_per_sqft : ℝ) : ℝ :=
  let wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_height * door_width
  let window_area := num_windows * (window_height * window_width)
  let area_to_wash := wall_area - (door_area + window_area)
  area_to_wash * cost_per_sqft

/-- Theorem stating the cost of white washing the room --/
theorem white_washing_cost_calculation :
  white_washing_cost 25 15 12 6 3 4 3 3 4 = 3624 := by
  sorry

end NUMINAMATH_CALUDE_white_washing_cost_calculation_l1798_179838


namespace NUMINAMATH_CALUDE_pinky_pies_l1798_179836

theorem pinky_pies (helen_pies total_pies : ℕ) 
  (helen_made : helen_pies = 56)
  (total : total_pies = 203) :
  total_pies - helen_pies = 147 := by
  sorry

end NUMINAMATH_CALUDE_pinky_pies_l1798_179836


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l1798_179878

theorem shaded_area_percentage (total_squares : Nat) (shaded_squares : Nat) :
  total_squares = 36 →
  shaded_squares = 16 →
  (shaded_squares : ℚ) / total_squares * 100 = 44.44 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l1798_179878


namespace NUMINAMATH_CALUDE_sphere_ratios_l1798_179819

/-- Given two spheres with radii in the ratio 2:3, prove that the ratio of their surface areas is 4:9 and the ratio of their volumes is 8:27 -/
theorem sphere_ratios (r₁ r₂ : ℝ) (h : r₁ / r₂ = 2 / 3) :
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4 / 9 ∧
  ((4 / 3) * π * r₁^3) / ((4 / 3) * π * r₂^3) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_ratios_l1798_179819


namespace NUMINAMATH_CALUDE_distinct_roots_range_reciprocal_roots_sum_squares_l1798_179854

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - 3*x + m - 3

-- Theorem for the range of m
theorem distinct_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic x₁ m = 0 ∧ quadratic x₂ m = 0) ↔ m < 21/4 :=
sorry

-- Theorem for the sum of squares of reciprocal roots
theorem reciprocal_roots_sum_squares (m : ℝ) (x₁ x₂ : ℝ) :
  quadratic x₁ m = 0 ∧ quadratic x₂ m = 0 ∧ x₁ * x₂ = 1 →
  x₁^2 + x₂^2 = 7 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_range_reciprocal_roots_sum_squares_l1798_179854


namespace NUMINAMATH_CALUDE_max_value_p_l1798_179850

theorem max_value_p (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b * c + a + c = b) : 
  let p := 2 / (1 + a^2) - 2 / (1 + b^2) + 3 / (1 + c^2)
  ∃ (max_p : ℝ), max_p = 10/3 ∧ p ≤ max_p := by
  sorry

end NUMINAMATH_CALUDE_max_value_p_l1798_179850


namespace NUMINAMATH_CALUDE_smallest_with_eight_factors_l1798_179873

def factorCount (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_with_eight_factors :
  ∀ n : ℕ, n > 0 → factorCount n = 8 → n ≥ 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_with_eight_factors_l1798_179873


namespace NUMINAMATH_CALUDE_system_solution_unique_l1798_179815

theorem system_solution_unique :
  ∃! (x y z : ℝ), 
    x^2 - y*z = 1 ∧
    y^2 - x*z = 2 ∧
    z^2 - x*y = 3 ∧
    (x = 5*Real.sqrt 2/6 ∨ x = -5*Real.sqrt 2/6) ∧
    (y = -Real.sqrt 2/6 ∨ y = Real.sqrt 2/6) ∧
    (z = -7*Real.sqrt 2/6 ∨ z = 7*Real.sqrt 2/6) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1798_179815


namespace NUMINAMATH_CALUDE_g_value_at_4_l1798_179849

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧  -- g is a cubic polynomial
  (g 0 = -2) ∧  -- g(0) = -2
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2)  -- roots of g are squares of roots of f

-- Theorem statement
theorem g_value_at_4 (g : ℝ → ℝ) (h : g_properties g) : g 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_4_l1798_179849


namespace NUMINAMATH_CALUDE_simplify_expression_l1798_179840

theorem simplify_expression : 18 * (8 / 16) * (3 / 27) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1798_179840


namespace NUMINAMATH_CALUDE_missy_capacity_l1798_179814

/-- The number of claims each agent can handle -/
structure AgentCapacity where
  jan : ℕ
  john : ℕ
  missy : ℕ

/-- Calculate the capacity of insurance agents based on given conditions -/
def calculate_capacity : AgentCapacity :=
  let jan_capacity := 20
  let john_capacity := jan_capacity + (jan_capacity * 30 / 100)
  let missy_capacity := john_capacity + 15
  { jan := jan_capacity,
    john := john_capacity,
    missy := missy_capacity }

/-- Theorem stating that Missy can handle 41 claims -/
theorem missy_capacity : (calculate_capacity).missy = 41 := by
  sorry

end NUMINAMATH_CALUDE_missy_capacity_l1798_179814


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1798_179880

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 4) ↔ (a * x^2 + b * x - 2 > 0)) → 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1798_179880


namespace NUMINAMATH_CALUDE_line_relationships_l1798_179806

-- Define a type for lines in a plane
structure Line2D where
  -- You might represent a line by its slope and y-intercept, or by two points, etc.
  -- For this abstract representation, we'll leave the internal structure unspecified

-- Define a type for planes
structure Plane where
  -- Again, we'll leave the internal structure unspecified for this abstract representation

-- Define what it means for two lines to be non-overlapping
def non_overlapping (l1 l2 : Line2D) : Prop :=
  l1 ≠ l2

-- Define what it means for two lines to be in the same plane
def same_plane (p : Plane) (l1 l2 : Line2D) : Prop :=
  -- This would typically involve some geometric condition
  True  -- placeholder

-- Define parallel relationship
def parallel (l1 l2 : Line2D) : Prop :=
  -- This would typically involve some geometric condition
  sorry

-- Define intersecting relationship
def intersecting (l1 l2 : Line2D) : Prop :=
  -- This would typically involve some geometric condition
  sorry

-- The main theorem
theorem line_relationships (p : Plane) (l1 l2 : Line2D) 
  (h1 : non_overlapping l1 l2) (h2 : same_plane p l1 l2) :
  parallel l1 l2 ∨ intersecting l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_line_relationships_l1798_179806


namespace NUMINAMATH_CALUDE_sin_neg_360_degrees_l1798_179813

theorem sin_neg_360_degrees : Real.sin (-(360 * π / 180)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_360_degrees_l1798_179813


namespace NUMINAMATH_CALUDE_carl_removed_heads_probability_l1798_179855

/-- Represents the state of a coin (Heads or Tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents the configuration of three coins on the table -/
def CoinConfiguration := (CoinState × CoinState × CoinState)

/-- The initial configuration with Alice's coin -/
def initialConfig : CoinConfiguration := (CoinState.Heads, CoinState.Heads, CoinState.Heads)

/-- The set of all possible configurations after Bill flips two coins -/
def allConfigurations : Set CoinConfiguration := {
  (CoinState.Heads, CoinState.Heads, CoinState.Heads),
  (CoinState.Heads, CoinState.Heads, CoinState.Tails),
  (CoinState.Heads, CoinState.Tails, CoinState.Heads),
  (CoinState.Heads, CoinState.Tails, CoinState.Tails)
}

/-- The set of configurations that result in two heads showing after Carl removes a coin -/
def twoHeadsConfigurations : Set CoinConfiguration := {
  (CoinState.Heads, CoinState.Heads, CoinState.Heads),
  (CoinState.Heads, CoinState.Heads, CoinState.Tails),
  (CoinState.Heads, CoinState.Tails, CoinState.Heads)
}

/-- The probability of Carl removing a heads coin given that two heads are showing -/
def probHeadsRemoved : ℚ := 3 / 5

theorem carl_removed_heads_probability :
  probHeadsRemoved = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_carl_removed_heads_probability_l1798_179855


namespace NUMINAMATH_CALUDE_reposition_convergence_l1798_179825

/-- Reposition transformation function -/
def reposition (n : ℕ) : ℕ :=
  sorry

/-- Theorem: Repeated reposition of a 4-digit number always results in 312 -/
theorem reposition_convergence (n : ℕ) (h : 1000 ≤ n ∧ n ≤ 9999) :
  ∃ k : ℕ, ∀ m : ℕ, m ≥ k → (reposition^[m] n) = 312 :=
sorry

end NUMINAMATH_CALUDE_reposition_convergence_l1798_179825


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1798_179845

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a-1)*x + 1 < 0) → (a > 3 ∨ a < -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1798_179845


namespace NUMINAMATH_CALUDE_paint_together_l1798_179835

/-- The amount of wall Heidi and Tom can paint together in a given time -/
def wall_painted (heidi_time tom_time paint_time : ℚ) : ℚ :=
  paint_time * (1 / heidi_time + 1 / tom_time)

/-- Theorem: Heidi and Tom can paint 5/12 of the wall in 15 minutes -/
theorem paint_together : wall_painted 60 90 15 = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_paint_together_l1798_179835


namespace NUMINAMATH_CALUDE_bucket_problem_l1798_179832

/-- Represents the state of the two buckets -/
structure BucketState :=
  (large : ℕ)  -- Amount in 7-liter bucket
  (small : ℕ)  -- Amount in 3-liter bucket

/-- Represents a single operation on the buckets -/
inductive BucketOperation
  | FillLarge
  | FillSmall
  | EmptyLarge
  | EmptySmall
  | PourLargeToSmall
  | PourSmallToLarge

/-- Applies a single operation to a bucket state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  match op with
  | BucketOperation.FillLarge => { large := 7, small := state.small }
  | BucketOperation.FillSmall => { large := state.large, small := 3 }
  | BucketOperation.EmptyLarge => { large := 0, small := state.small }
  | BucketOperation.EmptySmall => { large := state.large, small := 0 }
  | BucketOperation.PourLargeToSmall =>
      let amount := min state.large (3 - state.small)
      { large := state.large - amount, small := state.small + amount }
  | BucketOperation.PourSmallToLarge =>
      let amount := min state.small (7 - state.large)
      { large := state.large + amount, small := state.small - amount }

/-- Applies a sequence of operations to an initial state -/
def applyOperations (initial : BucketState) (ops : List BucketOperation) : BucketState :=
  ops.foldl applyOperation initial

/-- Checks if a specific amount can be measured using a sequence of operations -/
def canMeasure (amount : ℕ) : Prop :=
  ∃ (ops : List BucketOperation),
    (applyOperations { large := 0, small := 0 } ops).large = amount ∨
    (applyOperations { large := 0, small := 0 } ops).small = amount

theorem bucket_problem :
  canMeasure 1 ∧ canMeasure 2 ∧ canMeasure 4 ∧ canMeasure 5 ∧ canMeasure 6 :=
sorry

end NUMINAMATH_CALUDE_bucket_problem_l1798_179832


namespace NUMINAMATH_CALUDE_system_solution_l1798_179874

theorem system_solution :
  ∀ (x y a : ℝ),
  (2 * x + y = a) →
  (x + y = 3) →
  (x = 2) →
  (a = 5 ∧ y = 1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1798_179874


namespace NUMINAMATH_CALUDE_other_communities_count_l1798_179899

theorem other_communities_count (total_boys : ℕ) 
  (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 700 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  (total_boys : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 126 :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l1798_179899


namespace NUMINAMATH_CALUDE_specific_prism_surface_area_l1798_179887

/-- A right prism with an isosceles trapezoid base -/
structure RightPrism where
  AB : ℝ
  BC : ℝ
  AD : ℝ
  diagonal_cross_section_area : ℝ

/-- The total surface area of the right prism -/
def total_surface_area (p : RightPrism) : ℝ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the total surface area of the specific prism -/
theorem specific_prism_surface_area :
  ∃ (p : RightPrism),
    p.AB = 13 ∧
    p.BC = 11 ∧
    p.AD = 21 ∧
    p.diagonal_cross_section_area = 180 ∧
    total_surface_area p = 906 := by
  sorry

end NUMINAMATH_CALUDE_specific_prism_surface_area_l1798_179887
