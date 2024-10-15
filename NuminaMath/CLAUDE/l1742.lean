import Mathlib

namespace NUMINAMATH_CALUDE_multiplication_simplification_l1742_174274

theorem multiplication_simplification : 9 * (1 / 13) * 26 = 18 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_simplification_l1742_174274


namespace NUMINAMATH_CALUDE_one_third_of_5_4_l1742_174212

theorem one_third_of_5_4 : (5.4 / 3 : ℚ) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_5_4_l1742_174212


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l1742_174216

theorem cos_alpha_plus_pi_sixth (α : ℝ) (h : Real.sin (α - π/3) = 1/3) :
  Real.cos (α + π/6) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l1742_174216


namespace NUMINAMATH_CALUDE_sum_of_squares_l1742_174297

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -2) : a^2 + b^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1742_174297


namespace NUMINAMATH_CALUDE_double_dimensions_cylinder_l1742_174299

/-- A cylindrical container with original volume and new volume after doubling dimensions -/
structure Container where
  originalVolume : ℝ
  newVolume : ℝ

/-- The volume of a cylinder doubles when its radius is doubled -/
def volumeDoubledRadius (v : ℝ) : ℝ := 4 * v

/-- The volume of a cylinder doubles when its height is doubled -/
def volumeDoubledHeight (v : ℝ) : ℝ := 2 * v

/-- Theorem: Doubling all dimensions of a 5-gallon cylindrical container results in a 40-gallon container -/
theorem double_dimensions_cylinder (c : Container) 
  (h₁ : c.originalVolume = 5)
  (h₂ : c.newVolume = volumeDoubledHeight (volumeDoubledRadius c.originalVolume)) :
  c.newVolume = 40 := by
  sorry

#check double_dimensions_cylinder

end NUMINAMATH_CALUDE_double_dimensions_cylinder_l1742_174299


namespace NUMINAMATH_CALUDE_inequality_solution_l1742_174260

theorem inequality_solution (x : ℝ) : (x - 2) * (6 + 2*x) > 0 ↔ x > 2 ∨ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1742_174260


namespace NUMINAMATH_CALUDE_defective_units_percentage_l1742_174284

theorem defective_units_percentage 
  (shipped_defective_ratio : Real) 
  (total_shipped_defective_ratio : Real) 
  (h1 : shipped_defective_ratio = 0.04)
  (h2 : total_shipped_defective_ratio = 0.0016) : 
  total_shipped_defective_ratio / shipped_defective_ratio = 0.04 := by
sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l1742_174284


namespace NUMINAMATH_CALUDE_real_part_divisible_by_p_l1742_174220

/-- A Gaussian integer is a complex number with integer real and imaginary parts. -/
structure GaussianInteger where
  re : ℤ
  im : ℤ

/-- The real part of a complex number z^p - z is divisible by p for any Gaussian integer z and odd prime p. -/
theorem real_part_divisible_by_p (z : GaussianInteger) (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (k : ℤ), (z.re^p - z.re : ℤ) = p * k := by
  sorry

end NUMINAMATH_CALUDE_real_part_divisible_by_p_l1742_174220


namespace NUMINAMATH_CALUDE_number_equation_solution_l1742_174210

theorem number_equation_solution : 
  ∃ x : ℝ, x - (1002 / 20.04) = 3500 ∧ x = 3550 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1742_174210


namespace NUMINAMATH_CALUDE_fathers_age_l1742_174265

/-- Represents the ages of family members and proves the father's age -/
theorem fathers_age (total_age sister_age kaydence_age : ℕ) 
  (h1 : total_age = 200)
  (h2 : sister_age = 40)
  (h3 : kaydence_age = 12) :
  ∃ (father_age : ℕ),
    father_age = 60 ∧
    ∃ (mother_age brother_age : ℕ),
      mother_age = father_age - 2 ∧
      brother_age = father_age / 2 ∧
      father_age + mother_age + brother_age + sister_age + kaydence_age = total_age :=
by
  sorry


end NUMINAMATH_CALUDE_fathers_age_l1742_174265


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1742_174200

def f (a x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = 2) →
  a = -1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1742_174200


namespace NUMINAMATH_CALUDE_remainder_of_1021_pow_1022_mod_1023_l1742_174280

theorem remainder_of_1021_pow_1022_mod_1023 : 
  1021^1022 % 1023 = 16 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1021_pow_1022_mod_1023_l1742_174280


namespace NUMINAMATH_CALUDE_furniture_fraction_l1742_174204

def original_savings : ℚ := 960
def tv_cost : ℚ := 240

theorem furniture_fraction : 
  (original_savings - tv_cost) / original_savings = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_furniture_fraction_l1742_174204


namespace NUMINAMATH_CALUDE_jaewoong_ran_most_l1742_174201

-- Define the athletes and their distances
def jaewoong_distance : ℕ := 20  -- in kilometers
def seongmin_distance : ℕ := 2600  -- in meters
def eunseong_distance : ℕ := 5000  -- in meters

-- Define the conversion factor from kilometers to meters
def km_to_m : ℕ := 1000

-- Theorem to prove Jaewoong ran the most
theorem jaewoong_ran_most :
  (jaewoong_distance * km_to_m > seongmin_distance) ∧
  (jaewoong_distance * km_to_m > eunseong_distance) :=
by
  sorry

#check jaewoong_ran_most

end NUMINAMATH_CALUDE_jaewoong_ran_most_l1742_174201


namespace NUMINAMATH_CALUDE_line_equation_through_two_points_l1742_174259

/-- The equation of a line passing through two points is x + y = 1 -/
theorem line_equation_through_two_points :
  ∀ (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ),
  A = (1, -2) →
  B = (-3, 2) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ ((x - 1) * (2 - (-2)) = (y - (-2)) * ((-3) - 1))) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x + y = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_two_points_l1742_174259


namespace NUMINAMATH_CALUDE_tank_capacity_l1742_174281

/-- Represents a cylindrical water tank --/
structure WaterTank where
  capacity : ℝ
  currentPercentage : ℝ
  currentVolume : ℝ

/-- Theorem: A cylindrical tank that is 25% full with 60 liters has a total capacity of 240 liters --/
theorem tank_capacity (tank : WaterTank) 
  (h1 : tank.currentPercentage = 0.25)
  (h2 : tank.currentVolume = 60) : 
  tank.capacity = 240 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l1742_174281


namespace NUMINAMATH_CALUDE_cube_volume_l1742_174228

/-- Represents a rectangular box with given dimensions -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ :=
  b.length * b.width * b.height

/-- Represents the problem setup -/
def problemSetup : Box × ℕ :=
  (⟨7, 18, 3⟩, 42)

/-- Theorem stating the volume of each cube -/
theorem cube_volume (box : Box) (num_cubes : ℕ) 
  (h1 : box = problemSetup.1) 
  (h2 : num_cubes = problemSetup.2) : 
  (boxVolume box) / num_cubes = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l1742_174228


namespace NUMINAMATH_CALUDE_OPQRS_shape_l1742_174245

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The figure formed by connecting points O, P, Q, R, and S -/
inductive Figure
  | Parallelepiped
  | Plane
  | StraightLine
  | General3D

/-- The theorem stating that OPQRS can only be a parallelepiped or a plane -/
theorem OPQRS_shape (O P Q R S : Point3D)
  (hO : O = ⟨0, 0, 0⟩)
  (hR : R = ⟨P.x + Q.x, P.y + Q.y, P.z + Q.z⟩)
  (hDistinct : O ≠ P ∧ O ≠ Q ∧ O ≠ R ∧ O ≠ S ∧ P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S) :
  (∃ f : Figure, f = Figure.Parallelepiped ∨ f = Figure.Plane) ∧
  ¬(∃ f : Figure, f = Figure.StraightLine ∨ f = Figure.General3D) :=
sorry

end NUMINAMATH_CALUDE_OPQRS_shape_l1742_174245


namespace NUMINAMATH_CALUDE_number_of_children_l1742_174242

/-- Given a group of children born at 2-year intervals with the youngest being 6 years old,
    and the sum of their ages being 50 years, prove that there are 5 children. -/
theorem number_of_children (sum_of_ages : ℕ) (age_difference : ℕ) (youngest_age : ℕ) 
  (h1 : sum_of_ages = 50)
  (h2 : age_difference = 2)
  (h3 : youngest_age = 6) :
  ∃ (n : ℕ), n = 5 ∧ 
  sum_of_ages = n * (youngest_age + (n - 1) * age_difference / 2) := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l1742_174242


namespace NUMINAMATH_CALUDE_fliers_remaining_l1742_174253

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ)
  (h_total : total = 2000)
  (h_morning : morning_fraction = 1 / 10)
  (h_afternoon : afternoon_fraction = 1 / 4) :
  total - (total * morning_fraction).floor - ((total - (total * morning_fraction).floor) * afternoon_fraction).floor = 1350 :=
by sorry

end NUMINAMATH_CALUDE_fliers_remaining_l1742_174253


namespace NUMINAMATH_CALUDE_arithmetic_progression_properties_l1742_174290

-- Define the arithmetic progression
def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

-- Define the geometric progression condition
def geometric_progression_condition (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, a 4 = a 2 * r ∧ a 8 = a 4 * r

-- Main theorem
theorem arithmetic_progression_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_progression a)
  (h_a1 : a 1 = 1)
  (h_geom : geometric_progression_condition a) :
  (∀ n : ℕ, a n = n) ∧
  (∀ n : ℕ, n ≤ 98 ↔ 100 * (1 - 1 / (n + 1)) < 99) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_properties_l1742_174290


namespace NUMINAMATH_CALUDE_eighth_grade_students_l1742_174236

theorem eighth_grade_students (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 68 → 
  girls = 28 → 
  boys < 2 * girls →
  boys = total - girls →
  2 * girls - boys = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_eighth_grade_students_l1742_174236


namespace NUMINAMATH_CALUDE_goose_egg_problem_l1742_174222

-- Define the total number of goose eggs laid
variable (E : ℕ)

-- Define the conditions
axiom hatch_ratio : (1 : ℚ) / 4 * E = (E / 4 : ℕ)
axiom first_month_survival : (4 : ℚ) / 5 * (E / 4 : ℕ) = (4 * E / 20 : ℕ)
axiom first_year_survival : (4 * E / 20 : ℕ) = 120

-- Define the theorem
theorem goose_egg_problem :
  E = 2400 ∧ ((4 * E / 20 : ℕ) - 120 : ℚ) / (4 * E / 20 : ℕ) = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_goose_egg_problem_l1742_174222


namespace NUMINAMATH_CALUDE_symmetric_line_passes_through_fixed_point_l1742_174269

/-- A line in 2D space represented by its slope and a point it passes through -/
structure Line2D where
  slope : ℝ
  point : ℝ × ℝ

/-- The symmetric point of a given point with respect to a center point -/
def symmetricPoint (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line2D) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * (p.1 - l.point.1) + l.point.2

/-- Two lines are symmetric about a point if the reflection of any point on one line
    through the center point lies on the other line -/
def symmetricLines (l1 l2 : Line2D) (center : ℝ × ℝ) : Prop :=
  ∀ p : ℝ × ℝ, pointOnLine l1 p → pointOnLine l2 (symmetricPoint p center)

theorem symmetric_line_passes_through_fixed_point :
  ∀ (k : ℝ) (l1 l2 : Line2D),
    l1.slope = k ∧
    l1.point = (4, 0) ∧
    symmetricLines l1 l2 (2, 1) →
    pointOnLine l2 (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_passes_through_fixed_point_l1742_174269


namespace NUMINAMATH_CALUDE_unique_rope_triangle_l1742_174213

/-- An isosceles triangle formed from a rope --/
structure RopeTriangle where
  total_length : ℝ
  base_length : ℝ
  side_length : ℝ
  is_isosceles : side_length = (total_length - base_length) / 2
  is_triangle : base_length + 2 * side_length = total_length

/-- The specific rope triangle from the problem --/
def problem_triangle : RopeTriangle where
  total_length := 24
  base_length := 6
  side_length := 9
  is_isosceles := by sorry
  is_triangle := by sorry

/-- Theorem stating that the problem_triangle is the unique solution --/
theorem unique_rope_triangle :
  ∀ (t : RopeTriangle), t.total_length = 24 ∧ t.base_length = 6 → t = problem_triangle :=
by sorry

end NUMINAMATH_CALUDE_unique_rope_triangle_l1742_174213


namespace NUMINAMATH_CALUDE_recruit_line_unique_solution_l1742_174271

/-- Represents the position of a person in the line of recruits -/
structure Position :=
  (front : ℕ)  -- number of people in front
  (behind : ℕ) -- number of people behind

/-- The line of recruits -/
structure RecruitLine :=
  (total : ℕ)
  (peter : Position)
  (nikolai : Position)
  (denis : Position)

/-- Conditions of the problem -/
def problem_conditions (line : RecruitLine) : Prop :=
  line.peter.front = 50 ∧
  line.nikolai.front = 100 ∧
  line.denis.front = 170 ∧
  (line.peter.behind = 4 * line.denis.behind ∨
   line.nikolai.behind = 4 * line.denis.behind ∨
   line.peter.behind = 4 * line.nikolai.behind) ∧
  line.total = line.denis.front + 1 + line.denis.behind

/-- The theorem to be proved -/
theorem recruit_line_unique_solution :
  ∃! line : RecruitLine, problem_conditions line ∧ line.total = 301 :=
sorry

end NUMINAMATH_CALUDE_recruit_line_unique_solution_l1742_174271


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1742_174227

/-- A hyperbola with center at the origin, axes of symmetry being coordinate axes,
    one focus coinciding with the focus of y^2 = 8x, and one asymptote being x + y = 0 -/
structure Hyperbola where
  /-- The focus of the parabola y^2 = 8x is (2, 0) -/
  focus : ℝ × ℝ
  /-- One asymptote of the hyperbola is x + y = 0 -/
  asymptote : ℝ → ℝ
  /-- The hyperbola's equation is in the form (x^2 / a^2) - (y^2 / b^2) = 1 -/
  a : ℝ
  b : ℝ
  focus_eq : focus = (2, 0)
  asymptote_eq : asymptote = fun x => -x
  ab_relation : b / a = 1

/-- The equation of the hyperbola is x^2/2 - y^2/2 = 1 -/
theorem hyperbola_equation (C : Hyperbola) : 
  ∀ x y : ℝ, (x^2 / 2) - (y^2 / 2) = 1 ↔ 
    (x^2 / C.a^2) - (y^2 / C.b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1742_174227


namespace NUMINAMATH_CALUDE_max_value_implies_a_l1742_174298

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + a

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 4, f a x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 4, f a x = 3) →
  a = 3 := by
  sorry


end NUMINAMATH_CALUDE_max_value_implies_a_l1742_174298


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_forty_l1742_174255

theorem fraction_of_fraction_of_forty : (2/3 : ℚ) * ((3/4 : ℚ) * 40) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_forty_l1742_174255


namespace NUMINAMATH_CALUDE_cattle_count_farm_cattle_count_l1742_174291

theorem cattle_count (cow_ratio : ℕ) (bull_ratio : ℕ) (bull_count : ℕ) : ℕ :=
  let total_ratio := cow_ratio + bull_ratio
  let parts := bull_count / bull_ratio
  let total_cattle := parts * total_ratio
  total_cattle

/-- Given a ratio of cows to bulls of 10:27 and 405 bulls, the total number of cattle is 675. -/
theorem farm_cattle_count : cattle_count 10 27 405 = 675 := by
  sorry

end NUMINAMATH_CALUDE_cattle_count_farm_cattle_count_l1742_174291


namespace NUMINAMATH_CALUDE_common_difference_from_terms_l1742_174247

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ :=
  seq.a 1 - seq.a 0

theorem common_difference_from_terms
  (seq : ArithmeticSequence)
  (h5 : seq.a 5 = 10)
  (h12 : seq.a 12 = 31) :
  commonDifference seq = 3 := by
  sorry


end NUMINAMATH_CALUDE_common_difference_from_terms_l1742_174247


namespace NUMINAMATH_CALUDE_two_digit_powers_of_three_l1742_174252

theorem two_digit_powers_of_three :
  ∃! (s : Finset ℕ), (∀ n ∈ s, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_powers_of_three_l1742_174252


namespace NUMINAMATH_CALUDE_least_n_squared_minus_n_divisibility_l1742_174288

theorem least_n_squared_minus_n_divisibility : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k = 0) ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k ≠ 0) ∧
    (∀ (m : ℕ), m > 0 ∧ m < n → 
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k = 0) ∨
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k ≠ 0))) ∧
  (∀ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k = 0) ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k ≠ 0) ∧
    (∀ (m : ℕ), m > 0 ∧ m < n → 
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k = 0) ∨
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k ≠ 0)) →
    n ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_least_n_squared_minus_n_divisibility_l1742_174288


namespace NUMINAMATH_CALUDE_dog_food_duration_aunt_gemma_dog_food_duration_l1742_174268

/-- Calculates the number of days dog food will last given the number of dogs, 
    feeding frequency, food consumption per meal, number of sacks, and weight of each sack. -/
theorem dog_food_duration (num_dogs : ℕ) (feedings_per_day : ℕ) (food_per_meal : ℕ)
                          (num_sacks : ℕ) (sack_weight_kg : ℕ) : ℕ :=
  let total_food_grams : ℕ := num_sacks * sack_weight_kg * 1000
  let daily_consumption : ℕ := num_dogs * food_per_meal * feedings_per_day
  total_food_grams / daily_consumption

/-- Proves that given Aunt Gemma's specific conditions, the dog food will last for 50 days. -/
theorem aunt_gemma_dog_food_duration : 
  dog_food_duration 4 2 250 2 50 = 50 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_duration_aunt_gemma_dog_food_duration_l1742_174268


namespace NUMINAMATH_CALUDE_min_cost_for_20_oranges_l1742_174263

/-- Represents a discount scheme for oranges -/
structure DiscountScheme where
  quantity : ℕ
  price : ℕ

/-- Calculates the cost of oranges given a discount scheme and number of groups -/
def calculateCost (scheme : DiscountScheme) (groups : ℕ) : ℕ :=
  scheme.price * groups

/-- Finds the minimum cost for a given number of oranges using available discount schemes -/
def minCostForOranges (schemes : List DiscountScheme) (targetOranges : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem min_cost_for_20_oranges :
  let schemes := [
    DiscountScheme.mk 4 12,
    DiscountScheme.mk 7 21
  ]
  minCostForOranges schemes 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_20_oranges_l1742_174263


namespace NUMINAMATH_CALUDE_borgnine_chimps_count_l1742_174238

/-- The number of chimps Borgnine has seen at the zoo -/
def num_chimps : ℕ := 25

/-- The total number of legs Borgnine wants to see at the zoo -/
def total_legs : ℕ := 1100

/-- The number of lions Borgnine has seen -/
def num_lions : ℕ := 8

/-- The number of lizards Borgnine has seen -/
def num_lizards : ℕ := 5

/-- The number of tarantulas Borgnine needs to see -/
def num_tarantulas : ℕ := 125

/-- The number of legs a lion, lizard, or chimp has -/
def legs_per_mammal_or_reptile : ℕ := 4

/-- The number of legs a tarantula has -/
def legs_per_tarantula : ℕ := 8

theorem borgnine_chimps_count :
  num_chimps * legs_per_mammal_or_reptile +
  num_lions * legs_per_mammal_or_reptile +
  num_lizards * legs_per_mammal_or_reptile +
  num_tarantulas * legs_per_tarantula = total_legs :=
by sorry

end NUMINAMATH_CALUDE_borgnine_chimps_count_l1742_174238


namespace NUMINAMATH_CALUDE_trig_identity_l1742_174223

theorem trig_identity (α : Real) (h : Real.tan α = 1/2) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1742_174223


namespace NUMINAMATH_CALUDE_platform_length_l1742_174221

/-- The length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmph = 72 →
  crossing_time = 20 →
  (train_speed_kmph * (1000 / 3600) * crossing_time) - train_length = 150 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l1742_174221


namespace NUMINAMATH_CALUDE_snow_total_l1742_174218

theorem snow_total (monday_snow tuesday_snow : Real) 
  (h1 : monday_snow = 0.32)
  (h2 : tuesday_snow = 0.21) : 
  monday_snow + tuesday_snow = 0.53 := by
sorry

end NUMINAMATH_CALUDE_snow_total_l1742_174218


namespace NUMINAMATH_CALUDE_intersection_M_N_l1742_174257

-- Define set M
def M : Set ℕ := {y | y < 6}

-- Define set N
def N : Set ℕ := {2, 3, 6}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1742_174257


namespace NUMINAMATH_CALUDE_class_size_proof_l1742_174246

/-- Represents the number of students who like art -/
def art_students : ℕ := 35

/-- Represents the number of students who like music -/
def music_students : ℕ := 32

/-- Represents the number of students who like both art and music -/
def both_students : ℕ := 19

/-- Represents the total number of students in the class -/
def total_students : ℕ := art_students + music_students - both_students

theorem class_size_proof :
  total_students = 48 :=
sorry

end NUMINAMATH_CALUDE_class_size_proof_l1742_174246


namespace NUMINAMATH_CALUDE_problem_solution_l1742_174261

theorem problem_solution : (π - 3.14)^0 + Real.sqrt ((Real.sqrt 2 - 1)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1742_174261


namespace NUMINAMATH_CALUDE_line_segment_param_sum_l1742_174243

/-- Given a line segment connecting (1, -3) and (-4, 5), parameterized by x = pt + q and y = rt + s
    where 0 ≤ t ≤ 2 and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 32.25 -/
theorem line_segment_param_sum (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 2 → p * t + q = 1 - 5 * t / 2 ∧ r * t + s = -3 + 4 * t) →
  p^2 + q^2 + r^2 + s^2 = 129/4 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_l1742_174243


namespace NUMINAMATH_CALUDE_sqrt2_not_in_rational_intervals_l1742_174234

theorem sqrt2_not_in_rational_intervals (p q : ℕ) (h_coprime : Nat.Coprime p q) 
  (h_p_lt_q : p < q) (h_q_ne_0 : q ≠ 0) : 
  |Real.sqrt 2 / 2 - p / q| > 1 / (4 * q^2) :=
sorry

end NUMINAMATH_CALUDE_sqrt2_not_in_rational_intervals_l1742_174234


namespace NUMINAMATH_CALUDE_gain_amount_calculation_l1742_174282

/-- Calculates the amount given the gain and gain percent -/
def calculateAmount (gain : ℚ) (gainPercent : ℚ) : ℚ :=
  gain / (gainPercent / 100)

/-- Theorem: Given a gain of 0.70 rupees and a gain percent of 1%, 
    the amount on which the gain is made is 70 rupees -/
theorem gain_amount_calculation (gain : ℚ) (gainPercent : ℚ) 
  (h1 : gain = 70/100) (h2 : gainPercent = 1) : 
  calculateAmount gain gainPercent = 70 := by
  sorry

#eval calculateAmount (70/100) 1

end NUMINAMATH_CALUDE_gain_amount_calculation_l1742_174282


namespace NUMINAMATH_CALUDE_min_value_cube_sum_plus_inverse_cube_equality_condition_l1742_174254

theorem min_value_cube_sum_plus_inverse_cube (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + 1 / (a + b)^3 ≥ 4^(1/4) :=
sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 + 1 / (a + b)^3 = 4^(1/4) ↔ a = b ∧ a = (4^(1/4) / 2)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_cube_sum_plus_inverse_cube_equality_condition_l1742_174254


namespace NUMINAMATH_CALUDE_math_team_combinations_l1742_174209

theorem math_team_combinations (girls : ℕ) (boys : ℕ) : 
  girls = 5 → boys = 8 → (girls.choose 1) * ((girls - 1).choose 2) * (boys.choose 2) = 840 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l1742_174209


namespace NUMINAMATH_CALUDE_triangle_area_l1742_174292

/-- Given a triangle with side lengths a, b, c where:
  - a = 13
  - The angle opposite side a is 60°
  - b : c = 4 : 3
  Prove that the area of the triangle is 39√3 -/
theorem triangle_area (a b c : ℝ) (A : ℝ) (h1 : a = 13) (h2 : A = π / 3)
    (h3 : ∃ (k : ℝ), b = 4 * k ∧ c = 3 * k) :
    (1 / 2) * b * c * Real.sin A = 39 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1742_174292


namespace NUMINAMATH_CALUDE_problem_solution_l1742_174224

theorem problem_solution :
  let a : ℚ := -1/2
  let x : ℤ := 8
  let y : ℤ := 5
  (a * (a^4 - a + 1) * (a - 2) = 125/64) ∧
  ((x + 2*y) * (x - y) - (2*x - y) * (-x - y) = 87) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1742_174224


namespace NUMINAMATH_CALUDE_only_taller_students_not_set_l1742_174277

-- Define the options
inductive SetOption
  | PrimesUpTo20
  | RootsOfEquation
  | TallerStudents
  | AllSquares

-- Define a predicate for well-defined sets
def is_well_defined_set (option : SetOption) : Prop :=
  match option with
  | SetOption.PrimesUpTo20 => true
  | SetOption.RootsOfEquation => true
  | SetOption.TallerStudents => false
  | SetOption.AllSquares => true

-- Theorem statement
theorem only_taller_students_not_set :
  ∀ (option : SetOption),
    ¬(is_well_defined_set option) ↔ option = SetOption.TallerStudents :=
by sorry

end NUMINAMATH_CALUDE_only_taller_students_not_set_l1742_174277


namespace NUMINAMATH_CALUDE_impossibility_of_2005_vectors_l1742_174248

/-- A type representing a vector in a plane -/
def PlaneVector : Type := ℝ × ℝ

/-- A function to check if a vector is non-zero -/
def is_nonzero (v : PlaneVector) : Prop := v ≠ (0, 0)

/-- A function to calculate the sum of three vectors -/
def sum_three (v1 v2 v3 : PlaneVector) : PlaneVector :=
  (v1.1 + v2.1 + v3.1, v1.2 + v2.2 + v3.2)

/-- The main theorem -/
theorem impossibility_of_2005_vectors :
  ¬ ∃ (vectors : Fin 2005 → PlaneVector),
    (∀ i, is_nonzero (vectors i)) ∧
    (∀ (subset : Fin 10 → Fin 2005),
      ∃ (i j k : Fin 10), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
        sum_three (vectors (subset i)) (vectors (subset j)) (vectors (subset k)) = (0, 0)) :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_2005_vectors_l1742_174248


namespace NUMINAMATH_CALUDE_computer_science_marks_l1742_174215

theorem computer_science_marks 
  (geography : ℕ) 
  (history_government : ℕ) 
  (art : ℕ) 
  (modern_literature : ℕ) 
  (average : ℚ) 
  (h1 : geography = 56)
  (h2 : history_government = 60)
  (h3 : art = 72)
  (h4 : modern_literature = 80)
  (h5 : average = 70.6)
  : ∃ (computer_science : ℕ),
    (geography + history_government + art + computer_science + modern_literature) / 5 = average ∧ 
    computer_science = 85 := by
sorry

end NUMINAMATH_CALUDE_computer_science_marks_l1742_174215


namespace NUMINAMATH_CALUDE_car_part_cost_l1742_174202

/-- Calculates the cost of a car part given the total repair cost, labor time, and hourly rate. -/
theorem car_part_cost (total_cost labor_time hourly_rate : ℝ) : 
  total_cost = 300 ∧ labor_time = 2 ∧ hourly_rate = 75 → 
  total_cost - (labor_time * hourly_rate) = 150 := by
sorry

end NUMINAMATH_CALUDE_car_part_cost_l1742_174202


namespace NUMINAMATH_CALUDE_ellipse_parabola_problem_l1742_174278

/-- Given an ellipse and a parabola with specific properties, prove the equation of the ellipse,
    the coordinates of a point, and the range of a certain expression. -/
theorem ellipse_parabola_problem (a b p : ℝ) (F : ℝ × ℝ) :
  a > b ∧ b > 0 ∧ p > 0 ∧  -- Conditions on a, b, and p
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ y^2 = 2*p*x) ∧  -- C₁ and C₂ have a common point
  (F.1 - F.2 + 1)^2 / 2 = 2 ∧  -- Distance from F to x - y + 1 = 0 is √2
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ y^2 = 2*p*x ∧ (x - 3/2)^2 + y^2 = 6) →  -- Common chord length is 2√6
  (a^2 = 9 ∧ b^2 = 8 ∧ F = (1, 0)) ∧  -- Equation of C₁ and coordinates of F
  (∀ k : ℝ, k ≠ 0 → 
    1/6 < (21*k^2 + 8)/(48*(k^2 + 1)) ∧ (21*k^2 + 8)/(48*(k^2 + 1)) ≤ 7/16) -- Range of 1/|AB| + 1/|CD|
  := by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_problem_l1742_174278


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l1742_174211

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a trapezoid formed after removing segments from a rectangle -/
structure Trapezoid where
  longBase : ℝ
  shortBase : ℝ
  height : ℝ

/-- Calculates the total length of segments in the trapezoid -/
def totalLength (t : Trapezoid) : ℝ :=
  t.longBase + t.shortBase + t.height

/-- Theorem stating that the total length of segments in the resulting trapezoid is 19 units -/
theorem trapezoid_segment_length 
  (r : Rectangle)
  (t : Trapezoid)
  (h1 : r.length = 11)
  (h2 : r.width = 3)
  (h3 : r.height = 12)
  (h4 : t.longBase = 8)
  (h5 : t.shortBase = r.width)
  (h6 : t.height = r.height - 4) :
  totalLength t = 19 := by
    sorry


end NUMINAMATH_CALUDE_trapezoid_segment_length_l1742_174211


namespace NUMINAMATH_CALUDE_fifth_employee_speed_is_140_l1742_174237

/-- Calculates the typing speed of the fifth employee given the team size, average speed, and speeds of four employees --/
def fifth_employee_speed (team_size : ℕ) (average_speed : ℕ) (speed1 speed2 speed3 speed4 : ℕ) : ℕ :=
  team_size * average_speed - speed1 - speed2 - speed3 - speed4

/-- Proves that the fifth employee's typing speed is 140 words per minute --/
theorem fifth_employee_speed_is_140 :
  fifth_employee_speed 5 80 64 76 91 89 = 140 := by
  sorry

end NUMINAMATH_CALUDE_fifth_employee_speed_is_140_l1742_174237


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l1742_174296

def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_sum_of_coefficients 
  (a b c d : ℝ) 
  (h1 : g a b c d (3*Complex.I) = 0)
  (h2 : g a b c d (3 + Complex.I) = 0) :
  a + b + c + d = 49 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l1742_174296


namespace NUMINAMATH_CALUDE_complex_sequence_sum_l1742_174205

theorem complex_sequence_sum (a b : ℕ → ℝ) :
  (∀ n : ℕ, (Complex.I + 2) ^ n = Complex.mk (a n) (b n)) →
  (∑' n, (a n * b n) / (7 : ℝ) ^ n) = 7 / 16 := by
sorry

end NUMINAMATH_CALUDE_complex_sequence_sum_l1742_174205


namespace NUMINAMATH_CALUDE_number_added_to_23_l1742_174293

theorem number_added_to_23 : ∃! x : ℝ, 23 + x = 34 := by
  sorry

end NUMINAMATH_CALUDE_number_added_to_23_l1742_174293


namespace NUMINAMATH_CALUDE_aubrey_garden_yield_l1742_174286

/-- Represents Aubrey's garden layout and plant yields --/
structure Garden where
  total_rows : Nat
  tomato_plants_per_row : Nat
  cucumber_plants_per_row : Nat
  bell_pepper_plants_per_row : Nat
  tomato_yield_first_last : Nat
  tomato_yield_middle : Nat
  cucumber_yield_a : Nat
  cucumber_yield_b : Nat
  bell_pepper_yield : Nat

/-- Calculates the total yield of vegetables in Aubrey's garden --/
def calculate_yield (g : Garden) : Nat × Nat × Nat :=
  let pattern_rows := 4
  let patterns := g.total_rows / pattern_rows
  let tomato_rows := patterns
  let cucumber_rows := 2 * patterns
  let bell_pepper_rows := patterns

  let tomatoes_per_row := 2 * g.tomato_yield_first_last + (g.tomato_plants_per_row - 2) * g.tomato_yield_middle
  let cucumbers_per_row := (g.cucumber_plants_per_row / 2) * (g.cucumber_yield_a + g.cucumber_yield_b)
  let bell_peppers_per_row := g.bell_pepper_plants_per_row * g.bell_pepper_yield

  let total_tomatoes := tomato_rows * tomatoes_per_row
  let total_cucumbers := cucumber_rows * cucumbers_per_row
  let total_bell_peppers := bell_pepper_rows * bell_peppers_per_row

  (total_tomatoes, total_cucumbers, total_bell_peppers)

/-- Theorem stating the total yield of Aubrey's garden --/
theorem aubrey_garden_yield (g : Garden)
  (h1 : g.total_rows = 20)
  (h2 : g.tomato_plants_per_row = 8)
  (h3 : g.cucumber_plants_per_row = 6)
  (h4 : g.bell_pepper_plants_per_row = 12)
  (h5 : g.tomato_yield_first_last = 6)
  (h6 : g.tomato_yield_middle = 4)
  (h7 : g.cucumber_yield_a = 4)
  (h8 : g.cucumber_yield_b = 5)
  (h9 : g.bell_pepper_yield = 2) :
  calculate_yield g = (180, 270, 120) := by
  sorry

#eval calculate_yield {
  total_rows := 20,
  tomato_plants_per_row := 8,
  cucumber_plants_per_row := 6,
  bell_pepper_plants_per_row := 12,
  tomato_yield_first_last := 6,
  tomato_yield_middle := 4,
  cucumber_yield_a := 4,
  cucumber_yield_b := 5,
  bell_pepper_yield := 2
}

end NUMINAMATH_CALUDE_aubrey_garden_yield_l1742_174286


namespace NUMINAMATH_CALUDE_power_zero_eq_one_iff_nonzero_l1742_174272

theorem power_zero_eq_one_iff_nonzero (a : ℝ) : a ^ 0 = 1 ↔ a ≠ 0 := by sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_iff_nonzero_l1742_174272


namespace NUMINAMATH_CALUDE_track_length_l1742_174232

/-- Represents a circular running track with two runners -/
structure CircularTrack where
  length : ℝ
  initial_distance : ℝ
  first_meeting_distance : ℝ
  second_meeting_additional_distance : ℝ

/-- The track satisfies the given conditions -/
def satisfies_conditions (track : CircularTrack) : Prop :=
  track.initial_distance = 120 ∧
  track.first_meeting_distance = 150 ∧
  track.second_meeting_additional_distance = 200

/-- The theorem stating the length of the track -/
theorem track_length (track : CircularTrack) 
  (h : satisfies_conditions track) : track.length = 450 := by
  sorry

end NUMINAMATH_CALUDE_track_length_l1742_174232


namespace NUMINAMATH_CALUDE_unique_solution_l1742_174240

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^2*y + x*y^2 - 2*x - 2*y + 10 = 0
def equation2 (x y : ℝ) : Prop := x^3*y - x*y^3 - 2*x^2 + 2*y^2 - 30 = 0

-- State the theorem
theorem unique_solution :
  ∃! p : ℝ × ℝ, equation1 p.1 p.2 ∧ equation2 p.1 p.2 ∧ p = (-4, -1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1742_174240


namespace NUMINAMATH_CALUDE_line_parameterization_l1742_174241

/-- Given a line y = 2x - 17 parameterized by (x,y) = (f(t), 20t - 12), prove that f(t) = 10t + 5/2 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 20*t - 12 = 2*(f t) - 17) → 
  (∀ t : ℝ, f t = 10*t + 5/2) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l1742_174241


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_two_three_l1742_174279

theorem chinese_remainder_theorem_two_three :
  (∀ (a b : ℤ), ∃ (x : ℤ), x ≡ a [ZMOD 5] ∧ x ≡ b [ZMOD 6] ∧ x = 6*a + 25*b) ∧
  (∀ (a b c : ℤ), ∃ (y : ℤ), y ≡ a [ZMOD 5] ∧ y ≡ b [ZMOD 6] ∧ y ≡ c [ZMOD 7] ∧ y = 126*a + 175*b + 120*c) :=
by sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_two_three_l1742_174279


namespace NUMINAMATH_CALUDE_b_rent_exceeds_total_cost_l1742_174262

/-- Represents the rent rates for different animals -/
structure RentRates where
  horse : ℕ
  cow : ℕ
  sheep : ℕ
  goat : ℕ

/-- Represents the animals and duration for a renter -/
structure RenterAnimals where
  horses : ℕ
  horseDuration : ℕ
  sheep : ℕ
  sheepDuration : ℕ
  goats : ℕ
  goatDuration : ℕ

/-- Calculates the total rent for a renter given their animals and rent rates -/
def calculateRent (animals : RenterAnimals) (rates : RentRates) : ℕ :=
  animals.horses * animals.horseDuration * rates.horse +
  animals.sheep * animals.sheepDuration * rates.sheep +
  animals.goats * animals.goatDuration * rates.goat

/-- The total cost of the pasture -/
def totalPastureCost : ℕ := 5820

/-- The rent rates for different animals -/
def givenRates : RentRates :=
  { horse := 30
    cow := 40
    sheep := 20
    goat := 25 }

/-- B's animals and their durations -/
def bAnimals : RenterAnimals :=
  { horses := 16
    horseDuration := 9
    sheep := 18
    sheepDuration := 7
    goats := 4
    goatDuration := 6 }

theorem b_rent_exceeds_total_cost :
  calculateRent bAnimals givenRates > totalPastureCost := by
  sorry

end NUMINAMATH_CALUDE_b_rent_exceeds_total_cost_l1742_174262


namespace NUMINAMATH_CALUDE_part1_part2_l1742_174276

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 ≤ 0}

def p (x : ℝ) : Prop := x ∈ A
def q (m : ℝ) (x : ℝ) : Prop := x ∈ B m

theorem part1 (m : ℝ) (h : ∀ x, q m x → p x) (h' : ∃ x, p x ∧ ¬q m x) :
  0 ≤ m ∧ m ≤ 1 := by sorry

theorem part2 (m : ℝ) (h : ∀ x ∈ A, x^2 + m ≥ 4 + 3*x) :
  m ≥ 25/4 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1742_174276


namespace NUMINAMATH_CALUDE_internship_arrangement_l1742_174267

theorem internship_arrangement (n : Nat) (k : Nat) (m : Nat) : 
  n = 5 → k = 4 → m = 2 →
  (Nat.choose k m / 2) * (Nat.factorial n / (Nat.factorial (n - m))) = 60 := by
  sorry

end NUMINAMATH_CALUDE_internship_arrangement_l1742_174267


namespace NUMINAMATH_CALUDE_range_of_f_l1742_174289

def f (x : ℝ) := |x + 5| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Iic 14 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1742_174289


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_inequality_l1742_174219

theorem arithmetic_geometric_harmonic_inequality {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > 2 * a * b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_inequality_l1742_174219


namespace NUMINAMATH_CALUDE_sin_difference_simplification_l1742_174233

theorem sin_difference_simplification (x y : ℝ) : 
  Real.sin (x + y) * Real.cos y - Real.cos (x + y) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_simplification_l1742_174233


namespace NUMINAMATH_CALUDE_luke_laundry_problem_l1742_174251

/-- Given a total number of clothing pieces, the number of pieces in the first load,
    and the number of remaining loads, calculate the number of pieces in each small load. -/
def pieces_per_small_load (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) : ℕ :=
  (total - first_load) / num_small_loads

/-- Theorem stating that given the specific conditions of the problem,
    the number of pieces in each small load is 10. -/
theorem luke_laundry_problem :
  pieces_per_small_load 105 34 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_luke_laundry_problem_l1742_174251


namespace NUMINAMATH_CALUDE_yoga_to_exercise_ratio_l1742_174235

/-- Proves that the ratio of yoga time to total exercise time is 1:1 -/
theorem yoga_to_exercise_ratio : 
  ∀ (gym_time bicycle_time yoga_time : ℝ),
  gym_time / bicycle_time = 2 / 3 →
  bicycle_time = 12 →
  yoga_time = 20 →
  yoga_time / (gym_time + bicycle_time) = 1 := by
  sorry

end NUMINAMATH_CALUDE_yoga_to_exercise_ratio_l1742_174235


namespace NUMINAMATH_CALUDE_sum_of_square_perimeters_l1742_174208

/-- The sum of the perimeters of an infinite sequence of squares, where each subsequent square
    is formed by connecting the midpoints of the sides of the previous square, given that the
    initial square has a side length of s. -/
theorem sum_of_square_perimeters (s : ℝ) (h : s > 0) :
  (∑' n, 4 * s / (2 ^ n)) = 8 * s := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_perimeters_l1742_174208


namespace NUMINAMATH_CALUDE_pentagon_square_side_ratio_l1742_174275

theorem pentagon_square_side_ratio :
  let pentagon_perimeter : ℝ := 100
  let square_perimeter : ℝ := 100
  let pentagon_side : ℝ := pentagon_perimeter / 5
  let square_side : ℝ := square_perimeter / 4
  pentagon_side / square_side = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_square_side_ratio_l1742_174275


namespace NUMINAMATH_CALUDE_median_equation_l1742_174217

/-- The equation of median BD in triangle ABC -/
theorem median_equation (A B C D : ℝ × ℝ) : 
  A = (4, 1) → B = (0, 3) → C = (2, 4) → D = ((A.1 + C.1)/2, (A.2 + C.2)/2) →
  (fun (x y : ℝ) => x + 6*y - 18) = (fun (x y : ℝ) => 0) := by sorry

end NUMINAMATH_CALUDE_median_equation_l1742_174217


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1742_174256

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1742_174256


namespace NUMINAMATH_CALUDE_larger_sample_more_accurate_l1742_174230

-- Define a sampling survey
structure SamplingSurvey where
  population : Set ℝ
  sample : Set ℝ
  sample_size : ℕ

-- Define estimation accuracy
def estimation_accuracy (survey : SamplingSurvey) : ℝ := sorry

-- Theorem stating that larger sample size leads to more accurate estimation
theorem larger_sample_more_accurate (survey1 survey2 : SamplingSurvey) 
  (h : survey1.population = survey2.population) 
  (h_size : survey1.sample_size < survey2.sample_size) : 
  estimation_accuracy survey1 < estimation_accuracy survey2 := by
  sorry

end NUMINAMATH_CALUDE_larger_sample_more_accurate_l1742_174230


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_range_l1742_174249

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a ≥ 0}

theorem intersection_nonempty_implies_a_range :
  (∃ a : ℝ, (A ∩ B a).Nonempty) ↔ {a : ℝ | a > -8} = Set.Ioi (-8) := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_range_l1742_174249


namespace NUMINAMATH_CALUDE_inequality_proof_l1742_174226

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  a^2 + b^2 + c^2 + 3 ≥ 1/a + 1/b + 1/c + a + b + c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1742_174226


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l1742_174231

theorem repeating_decimal_division (A B C D : Nat) : 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) →
  (100 * A + 10 * B + C) / (1000 * B + 100 * B + 10 * B + B) = 
    (1000 * B + 100 * C + 10 * D + B) / 9999 →
  A = 2 ∧ B = 1 ∧ C = 9 ∧ D = 7 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l1742_174231


namespace NUMINAMATH_CALUDE_sequence_term_from_sum_l1742_174206

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℕ := n^2 + 3*n

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℕ := 2*n + 2

theorem sequence_term_from_sum (n : ℕ) : 
  n > 0 → S n - S (n-1) = a n :=
by sorry

end NUMINAMATH_CALUDE_sequence_term_from_sum_l1742_174206


namespace NUMINAMATH_CALUDE_expression_simplification_and_ratio_l1742_174287

theorem expression_simplification_and_ratio :
  let expr := (6 * m + 4 * n + 12) / 4
  let a := 3/2
  let b := 1
  let c := 3
  expr = a * m + b * n + c ∧ (a + b + c) / c = 11/6 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_ratio_l1742_174287


namespace NUMINAMATH_CALUDE_parents_present_l1742_174229

theorem parents_present (total_people pupils teachers : ℕ) 
  (h1 : total_people = 1541)
  (h2 : pupils = 724)
  (h3 : teachers = 744) :
  total_people - (pupils + teachers) = 73 := by
  sorry

end NUMINAMATH_CALUDE_parents_present_l1742_174229


namespace NUMINAMATH_CALUDE_smallest_possible_value_l1742_174225

theorem smallest_possible_value (m n x : ℕ+) : 
  m = 60 →
  Nat.gcd m.val n.val = x.val + 5 →
  Nat.lcm m.val n.val = 2 * x.val * (x.val + 5) →
  (∀ n' : ℕ+, n'.val < n.val → 
    (Nat.gcd m.val n'.val ≠ x.val + 5 ∨ 
     Nat.lcm m.val n'.val ≠ 2 * x.val * (x.val + 5))) →
  n.val = 75 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_value_l1742_174225


namespace NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l1742_174214

theorem lattice_points_on_hyperbola : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1^2 - p.2^2 = 65) ∧ 
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l1742_174214


namespace NUMINAMATH_CALUDE_min_ellipse_area_l1742_174266

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b

/-- A circle with center (h, 0) and radius 1 -/
structure Circle where
  h : ℝ

/-- The ellipse is tangent to the circle -/
def is_tangent (e : Ellipse) (c : Circle) : Prop :=
  ∃ x y : ℝ, (x^2 / e.a^2) + (y^2 / e.b^2) = 1 ∧ (x - c.h)^2 + y^2 = 1

/-- The theorem stating the minimum area of the ellipse -/
theorem min_ellipse_area (e : Ellipse) (c1 c2 : Circle) 
  (h1 : is_tangent e c1) (h2 : is_tangent e c2) (h3 : c1.h = 2) (h4 : c2.h = -2) :
  e.a * e.b * π ≥ (10 * Real.sqrt 15 / 3) * π :=
sorry

end NUMINAMATH_CALUDE_min_ellipse_area_l1742_174266


namespace NUMINAMATH_CALUDE_only_equilateral_forms_triangle_l1742_174244

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 8), (5, 6, 11), (4, 4, 8), (8, 8, 8)]

/-- Theorem stating that only (8, 8, 8) can form a triangle among the given sets -/
theorem only_equilateral_forms_triangle :
  ∃! set : ℝ × ℝ × ℝ, set ∈ segment_sets ∧ can_form_triangle set.1 set.2.1 set.2.2 :=
by sorry

end NUMINAMATH_CALUDE_only_equilateral_forms_triangle_l1742_174244


namespace NUMINAMATH_CALUDE_ducks_remaining_theorem_l1742_174295

def ducks_remaining (initial : ℕ) : ℕ :=
  let after_first := initial - (initial / 4)
  let after_second := after_first - (after_first / 6)
  after_second - (after_second * 3 / 10)

theorem ducks_remaining_theorem :
  ducks_remaining 320 = 140 := by
  sorry

end NUMINAMATH_CALUDE_ducks_remaining_theorem_l1742_174295


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1742_174283

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 2) (h2 : x - y = 4) :
  x^2 - y^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1742_174283


namespace NUMINAMATH_CALUDE_not_repeating_decimal_l1742_174203

/-- Definition of the number we're considering -/
def x : ℚ := 3.66666

/-- Definition of a repeating decimal -/
def is_repeating_decimal (q : ℚ) : Prop :=
  ∃ (a b : ℕ) (c : ℤ), q = (c : ℚ) + (a : ℚ) / (10^b - 1)

/-- Theorem stating that 3.66666 is not a repeating decimal -/
theorem not_repeating_decimal : ¬ is_repeating_decimal x := by
  sorry

end NUMINAMATH_CALUDE_not_repeating_decimal_l1742_174203


namespace NUMINAMATH_CALUDE_max_consecutive_integers_l1742_174250

def consecutive_sum (start : ℕ) (n : ℕ) : ℕ :=
  n * (2 * start + n - 1) / 2

def is_valid_sequence (start : ℕ) (n : ℕ) : Prop :=
  consecutive_sum start n = 2014 ∧ start > 0

theorem max_consecutive_integers :
  (∃ (start : ℕ), is_valid_sequence start 53) ∧
  (∀ (m : ℕ) (start : ℕ), m > 53 → ¬ is_valid_sequence start m) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_l1742_174250


namespace NUMINAMATH_CALUDE_bicycle_selling_prices_l1742_174264

def calculate_selling_price (purchase_price : ℕ) (loss_percentage : ℕ) : ℕ :=
  purchase_price - (purchase_price * loss_percentage / 100)

def bicycle1_purchase_price : ℕ := 1800
def bicycle1_loss_percentage : ℕ := 25

def bicycle2_purchase_price : ℕ := 2700
def bicycle2_loss_percentage : ℕ := 15

def bicycle3_purchase_price : ℕ := 2200
def bicycle3_loss_percentage : ℕ := 20

theorem bicycle_selling_prices :
  (calculate_selling_price bicycle1_purchase_price bicycle1_loss_percentage = 1350) ∧
  (calculate_selling_price bicycle2_purchase_price bicycle2_loss_percentage = 2295) ∧
  (calculate_selling_price bicycle3_purchase_price bicycle3_loss_percentage = 1760) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_selling_prices_l1742_174264


namespace NUMINAMATH_CALUDE_translation_of_parabola_l1742_174258

theorem translation_of_parabola (t m : ℝ) : 
  (∀ x : ℝ, (x - 3)^2 = (t - 3)^2 → x = t) →  -- P is on y=(x-3)^2
  (t - m)^2 = (t - 3)^2 →                     -- Q is on y=x^2
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_translation_of_parabola_l1742_174258


namespace NUMINAMATH_CALUDE_large_square_perimeter_l1742_174294

-- Define the original square's perimeter
def original_perimeter : ℝ := 56

-- Define the number of parts the original square is divided into
def division_parts : ℕ := 4

-- Define the number of small squares used to form the large square
def small_squares : ℕ := 441

-- Theorem statement
theorem large_square_perimeter (original_perimeter : ℝ) (division_parts : ℕ) (small_squares : ℕ) :
  original_perimeter = 56 ∧ 
  division_parts = 4 ∧ 
  small_squares = 441 →
  (original_perimeter / (4 * Real.sqrt (small_squares : ℝ))) * 
  (4 * Real.sqrt (small_squares : ℝ)) = 588 := by
  sorry


end NUMINAMATH_CALUDE_large_square_perimeter_l1742_174294


namespace NUMINAMATH_CALUDE_circle_equation_is_correct_l1742_174239

/-- A circle C with center (1,2) that is tangent to the line x+2y=0 -/
def CircleC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 5}

/-- The line x+2y=0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 2*p.2 = 0}

theorem circle_equation_is_correct :
  (∀ p ∈ CircleC, (p.1 - 1)^2 + (p.2 - 2)^2 = 5) ∧
  (∃ q ∈ CircleC ∩ TangentLine, q = (1, 2)) ∧
  (∀ r ∈ CircleC, r ≠ (1, 2) → r ∉ TangentLine) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_is_correct_l1742_174239


namespace NUMINAMATH_CALUDE_optimal_racket_purchase_l1742_174273

/-- The optimal purchasing plan for badminton rackets -/
theorem optimal_racket_purchase 
  (total_cost : ℕ) 
  (num_pairs : ℕ) 
  (price_diff : ℕ) 
  (discount_a : ℚ) 
  (discount_b : ℕ) 
  (max_cost : ℕ) 
  (min_a : ℕ) :
  total_cost = num_pairs * (price_a + price_b) ∧
  price_b = price_a - price_diff ∧
  new_price_a = price_a * discount_a ∧
  new_price_b = price_b - discount_b ∧
  (∀ m : ℕ, m ≥ min_a → m ≤ 50 → 
    new_price_a * m + new_price_b * (50 - m) ≤ max_cost) →
  optimal_m = 38 ∧ 
  optimal_cost = new_price_a * optimal_m + new_price_b * (50 - optimal_m) ∧
  (∀ m : ℕ, m ≥ min_a → m ≤ 50 → 
    new_price_a * m + new_price_b * (50 - m) ≥ optimal_cost) :=
by
  sorry

#check optimal_racket_purchase 1300 20 15 (4/5) 4 1500 38

end NUMINAMATH_CALUDE_optimal_racket_purchase_l1742_174273


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1742_174285

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 3 ∧ x₂ = -5) ∧ 
  (x₁^2 + 2*x₁ - 15 = 0) ∧ 
  (x₂^2 + 2*x₂ - 15 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1742_174285


namespace NUMINAMATH_CALUDE_inequalities_proof_l1742_174270

theorem inequalities_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : a < b) (h3 : b < c) : 
  (a * b < b * c) ∧ 
  (a * c < b * c) ∧ 
  (a * b < a * c) ∧ 
  (a + b < b + c) := by
sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1742_174270


namespace NUMINAMATH_CALUDE_product_equals_zero_l1742_174207

theorem product_equals_zero (b : ℤ) (h : b = 5) : 
  ((b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) * 
   (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b) = 0 := by
sorry

end NUMINAMATH_CALUDE_product_equals_zero_l1742_174207
