import Mathlib

namespace NUMINAMATH_CALUDE_solve_system_of_equations_l351_35155

theorem solve_system_of_equations (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20)
  (eq2 : 6 * p + 5 * q = 27) :
  p = 62 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l351_35155


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_angles_pi_half_l351_35184

/-- Two vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_sum_angles_pi_half
  (α β : ℝ)
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h_a : a = (Real.sin α, Real.cos β))
  (h_b : b = (Real.cos α, Real.sin β))
  (h_parallel : parallel a b) :
  α + β = π / 2 := by
sorry


end NUMINAMATH_CALUDE_parallel_vectors_sum_angles_pi_half_l351_35184


namespace NUMINAMATH_CALUDE_complement_A_in_U_l351_35181

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l351_35181


namespace NUMINAMATH_CALUDE_tutorial_time_multiplier_l351_35127

/-- Represents the time spent on various activities before playing a game --/
structure GamePreparationTime where
  download : ℝ
  install : ℝ
  tutorial : ℝ
  total : ℝ

/-- Theorem: Given the conditions, the tutorial time multiplier is 3 --/
theorem tutorial_time_multiplier (t : GamePreparationTime) : 
  t.download = 10 ∧ 
  t.install = t.download / 2 ∧ 
  t.total = 60 ∧ 
  t.total = t.download + t.install + t.tutorial → 
  t.tutorial = 3 * (t.download + t.install) :=
by sorry

end NUMINAMATH_CALUDE_tutorial_time_multiplier_l351_35127


namespace NUMINAMATH_CALUDE_angle_with_complement_33_percent_of_supplement_is_45_degrees_l351_35194

theorem angle_with_complement_33_percent_of_supplement_is_45_degrees (x : ℝ) :
  (90 - x = (1 / 3) * (180 - x)) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_33_percent_of_supplement_is_45_degrees_l351_35194


namespace NUMINAMATH_CALUDE_plane_equation_correct_l351_35102

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
structure ParametricPlane where
  origin : Point3D
  direction1 : Point3D
  direction2 : Point3D

/-- Represents the equation of a plane in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point satisfies a plane equation -/
def satisfiesPlaneEquation (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- The given parametric equation of the plane -/
def givenPlane : ParametricPlane :=
  { origin := { x := 2, y := 4, z := 1 }
  , direction1 := { x := 2, y := 1, z := -3 }
  , direction2 := { x := -3, y := 0, z := 1 }
  }

/-- The equation of the plane to be proven -/
def planeEquation : PlaneEquation :=
  { A := 1, B := 8, C := 3, D := -37 }

theorem plane_equation_correct :
  (∀ s t : ℝ, satisfiesPlaneEquation
    { x := 2 + 2*s - 3*t
    , y := 4 + s
    , z := 1 - 3*s + t
    } planeEquation) ∧
  planeEquation.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs planeEquation.A) (Int.natAbs planeEquation.B))
          (Nat.gcd (Int.natAbs planeEquation.C) (Int.natAbs planeEquation.D)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l351_35102


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l351_35169

theorem prime_square_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_three : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l351_35169


namespace NUMINAMATH_CALUDE_special_arithmetic_sequence_sum_l351_35196

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence (m n : ℕ) :=
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : S n = m)
  (h2 : S m = n)
  (h3 : m ≠ n)

/-- The sum of the first (m+n) terms of the special arithmetic sequence -/
def sumMPlusN (seq : ArithmeticSequence m n) : ℝ :=
  seq.S (m + n)

theorem special_arithmetic_sequence_sum (m n : ℕ) (seq : ArithmeticSequence m n) :
  sumMPlusN seq = -(m + n) := by
  sorry

#check special_arithmetic_sequence_sum

end NUMINAMATH_CALUDE_special_arithmetic_sequence_sum_l351_35196


namespace NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l351_35160

/-- Given vectors a and b, prove that c = 2a + 5b has the specified coordinates -/
theorem vector_addition_and_scalar_multiplication (a b : ℝ × ℝ × ℝ) 
  (h1 : a = (3, -4, 5)) 
  (h2 : b = (-1, 0, -2)) : 
  (2 : ℝ) • a + (5 : ℝ) • b = (1, -8, 0) := by
  sorry

#check vector_addition_and_scalar_multiplication

end NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l351_35160


namespace NUMINAMATH_CALUDE_valera_coin_count_l351_35195

/-- Represents the number of coins of each denomination -/
structure CoinCount where
  fifteenKopecks : Nat
  twentyKopecks : Nat

/-- Calculates the total value in kopecks given a CoinCount -/
def totalValue (coins : CoinCount) : Nat :=
  15 * coins.fifteenKopecks + 20 * coins.twentyKopecks

/-- Represents the conditions of the problem -/
structure ProblemConditions where
  initialCoins : CoinCount
  movieTicketCoins : Nat
  movieTicketValue : Nat
  lunchCoins : Nat

/-- The main theorem to prove -/
theorem valera_coin_count (conditions : ProblemConditions) : 
  (conditions.initialCoins.twentyKopecks > conditions.initialCoins.fifteenKopecks) →
  (conditions.movieTicketCoins = 2) →
  (conditions.lunchCoins = 3) →
  (totalValue conditions.initialCoins / 5 = conditions.movieTicketValue) →
  (((totalValue conditions.initialCoins - conditions.movieTicketValue) / 2) % (totalValue conditions.initialCoins - conditions.movieTicketValue) = 0) →
  (conditions.initialCoins = CoinCount.mk 2 6) :=
by sorry

#check valera_coin_count

end NUMINAMATH_CALUDE_valera_coin_count_l351_35195


namespace NUMINAMATH_CALUDE_product_of_fractions_l351_35109

theorem product_of_fractions :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l351_35109


namespace NUMINAMATH_CALUDE_square_floor_tiles_l351_35138

theorem square_floor_tiles (s : ℕ) (h_odd : Odd s) (h_middle : (s + 1) / 2 = 49) :
  s * s = 9409 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l351_35138


namespace NUMINAMATH_CALUDE_teresas_siblings_teresa_has_three_siblings_l351_35174

/-- Given Teresa's pencil collection and distribution rules, calculate the number of her siblings --/
theorem teresas_siblings (colored_pencils : ℕ) (black_pencils : ℕ) (kept_pencils : ℕ) (pencils_per_sibling : ℕ) : ℕ :=
  let total_pencils := colored_pencils + black_pencils
  let shared_pencils := total_pencils - kept_pencils
  shared_pencils / pencils_per_sibling

/-- Prove that Teresa has 3 siblings given the problem conditions --/
theorem teresa_has_three_siblings :
  teresas_siblings 14 35 10 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_teresas_siblings_teresa_has_three_siblings_l351_35174


namespace NUMINAMATH_CALUDE_unique_solution_for_a_l351_35183

theorem unique_solution_for_a (a b c : ℕ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a < b ∧ b < c)
  (h_sum_reciprocals : 1 / a + 1 / b + 1 / c = 1)
  (h_sum : a + b + c = 11) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_a_l351_35183


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l351_35198

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 47 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 47 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l351_35198


namespace NUMINAMATH_CALUDE_bananas_to_kiwis_ratio_l351_35119

/-- Represents the cost of a dozen apples in dollars -/
def dozen_apples_cost : ℚ := 14

/-- Represents the amount Brian spent on kiwis in dollars -/
def kiwis_cost : ℚ := 10

/-- Represents the maximum number of apples Brian can buy -/
def max_apples : ℕ := 24

/-- Represents the amount Brian left his house with in dollars -/
def initial_amount : ℚ := 50

/-- Represents the subway fare in dollars -/
def subway_fare : ℚ := 3.5

/-- Calculates the amount spent on bananas -/
def bananas_cost : ℚ := initial_amount - 2 * subway_fare - kiwis_cost - (max_apples / 12) * dozen_apples_cost

/-- Theorem stating that the ratio of bananas cost to kiwis cost is 1:2 -/
theorem bananas_to_kiwis_ratio : bananas_cost / kiwis_cost = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bananas_to_kiwis_ratio_l351_35119


namespace NUMINAMATH_CALUDE_difference_from_averages_l351_35103

theorem difference_from_averages (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) : 
  c - a = 90 := by
sorry

end NUMINAMATH_CALUDE_difference_from_averages_l351_35103


namespace NUMINAMATH_CALUDE_equation_solution_l351_35162

theorem equation_solution (z : ℚ) : 
  Real.sqrt (5 - 4 * z) = 10 → z = -95/4 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l351_35162


namespace NUMINAMATH_CALUDE_first_thrilling_thursday_is_correct_l351_35142

/-- Represents a date with a day and a month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns true if the given date is a Thursday -/
def isThursday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Returns true if the given date is a Thrilling Thursday -/
def isThrillingThursday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- The date of school start -/
def schoolStartDate : Date :=
  { day := 12, month := 9 }

/-- The day of the week when school starts -/
def schoolStartDay : DayOfWeek :=
  DayOfWeek.Tuesday

/-- The date of the first Thrilling Thursday -/
def firstThrillingThursday : Date :=
  { day := 26, month := 10 }

theorem first_thrilling_thursday_is_correct :
  isThrillingThursday firstThrillingThursday schoolStartDate schoolStartDay ∧
  ∀ d, d.month ≥ schoolStartDate.month ∧ 
       (d.month > schoolStartDate.month ∨ (d.month = schoolStartDate.month ∧ d.day ≥ schoolStartDate.day)) ∧
       isThrillingThursday d schoolStartDate schoolStartDay →
       (d.month > firstThrillingThursday.month ∨ 
        (d.month = firstThrillingThursday.month ∧ d.day ≥ firstThrillingThursday.day)) :=
  sorry

end NUMINAMATH_CALUDE_first_thrilling_thursday_is_correct_l351_35142


namespace NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_neg_five_halves_l351_35123

/-- The dividend polynomial -/
def dividend (b x : ℝ) : ℝ := 12 * x^4 - 5 * x^3 + b * x^2 - 4 * x + 8

/-- The divisor polynomial -/
def divisor (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

/-- Theorem stating that the remainder is constant iff b = -5/2 -/
theorem constant_remainder_iff_b_eq_neg_five_halves :
  ∃ (q : ℝ → ℝ) (r : ℝ), ∀ x, dividend (-5/2) x = q x * divisor x + r ↔ 
  ∀ b, (∃ (q : ℝ → ℝ) (r : ℝ), ∀ x, dividend b x = q x * divisor x + r) → b = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_neg_five_halves_l351_35123


namespace NUMINAMATH_CALUDE_max_rectangular_pen_area_l351_35178

theorem max_rectangular_pen_area (perimeter : ℝ) (h : perimeter = 60) : 
  ∃ (width height : ℝ), 
    width > 0 ∧ height > 0 ∧
    2 * (width + height) = perimeter ∧
    ∀ (w h : ℝ), w > 0 → h > 0 → 2 * (w + h) = perimeter → w * h ≤ width * height ∧
    width * height = 225 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangular_pen_area_l351_35178


namespace NUMINAMATH_CALUDE_mask_package_duration_l351_35116

/-- Calculates the number of days a package of masks will last for a family -/
def mask_duration (total_masks : ℕ) (family_size : ℕ) (days_per_mask : ℕ) : ℕ :=
  (total_masks / family_size) * days_per_mask

/-- Theorem: A package of 100 masks lasts 80 days for a family of 5, changing masks every 4 days -/
theorem mask_package_duration :
  mask_duration 100 5 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_mask_package_duration_l351_35116


namespace NUMINAMATH_CALUDE_barb_dress_fraction_l351_35114

theorem barb_dress_fraction (original_price savings paid : ℝ) (f : ℝ) :
  original_price = 180 →
  savings = 80 →
  paid = original_price - savings →
  paid = f * original_price - 10 →
  f = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_barb_dress_fraction_l351_35114


namespace NUMINAMATH_CALUDE_f_properties_l351_35197

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2

theorem f_properties :
  let T := Real.pi
  let monotonic_interval (k : ℤ) := Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)
  ∀ x : ℝ,
  (∀ y : ℝ, f (x + T) = f x) ∧ 
  (∀ k : ℤ, StrictMono (f ∘ (fun t => t + k * Real.pi - Real.pi / 3) : monotonic_interval k → ℝ)) ∧
  (x ∈ Set.Icc 0 (Real.pi / 4) → 
    f x ∈ Set.Icc 0 1 ∧
    (f x = 0 ↔ x = 0) ∧
    (f x = 1 ↔ x = Real.pi / 6)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l351_35197


namespace NUMINAMATH_CALUDE_planes_parallel_or_intersect_l351_35164

-- Define a type for 3D space
structure Space3D where
  -- Add necessary fields here
  
-- Define a type for planes in 3D space
structure Plane where
  -- Add necessary fields here

-- Define a type for lines in 3D space
structure Line where
  -- Add necessary fields here

-- Define what it means for a line to be parallel to a plane
def Line.parallelTo (l : Line) (p : Plane) : Prop :=
  sorry

-- Define what it means for a plane to contain a line
def Plane.contains (p : Plane) (l : Line) : Prop :=
  sorry

-- Define what it means for two planes to be parallel
def Plane.parallel (p1 p2 : Plane) : Prop :=
  sorry

-- Define what it means for two planes to intersect
def Plane.intersect (p1 p2 : Plane) : Prop :=
  sorry

-- The main theorem
theorem planes_parallel_or_intersect (p1 p2 : Plane) :
  (∃ (S : Set Line), Set.Infinite S ∧ (∀ l ∈ S, p1.contains l ∧ l.parallelTo p2)) →
  (p1.parallel p2 ∨ p1.intersect p2) :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_or_intersect_l351_35164


namespace NUMINAMATH_CALUDE_sphere_volume_containing_cube_l351_35140

theorem sphere_volume_containing_cube (edge_length : ℝ) (h : edge_length = 2) :
  let diagonal := edge_length * Real.sqrt 3
  let radius := diagonal / 2
  let volume := (4 / 3) * Real.pi * radius ^ 3
  volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_containing_cube_l351_35140


namespace NUMINAMATH_CALUDE_mean_median_difference_l351_35149

/-- Represents the frequency distribution of days missed by students -/
def frequency_distribution : List (Nat × Nat) := [
  (0, 4),  -- 4 students missed 0 days
  (1, 2),  -- 2 students missed 1 day
  (2, 5),  -- 5 students missed 2 days
  (3, 2),  -- 2 students missed 3 days
  (4, 1),  -- 1 student missed 4 days
  (5, 3),  -- 3 students missed 5 days
  (6, 1)   -- 1 student missed 6 days
]

/-- Calculate the median of the distribution -/
def median (dist : List (Nat × Nat)) : Nat :=
  sorry

/-- Calculate the mean of the distribution -/
def mean (dist : List (Nat × Nat)) : Rat :=
  sorry

/-- The total number of students -/
def total_students : Nat := frequency_distribution.map (·.2) |>.sum

theorem mean_median_difference :
  mean frequency_distribution - median frequency_distribution = 0 ∧ total_students = 18 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l351_35149


namespace NUMINAMATH_CALUDE_garden_width_l351_35111

theorem garden_width (garden_perimeter : ℝ) (playground_length playground_width : ℝ) 
  (h1 : garden_perimeter = 64)
  (h2 : playground_length = 16)
  (h3 : playground_width = 12) :
  ∃ (garden_width : ℝ),
    garden_width > 0 ∧
    garden_width < garden_perimeter / 2 ∧
    (garden_perimeter / 2 - garden_width) * garden_width = playground_length * playground_width ∧
    garden_width = 12 := by
  sorry

#check garden_width

end NUMINAMATH_CALUDE_garden_width_l351_35111


namespace NUMINAMATH_CALUDE_sum_of_integers_l351_35177

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 325) (h2 : x * y = 120) :
  (x : ℝ) + y = Real.sqrt 565 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l351_35177


namespace NUMINAMATH_CALUDE_f_value_at_5_l351_35148

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * x + 2

theorem f_value_at_5 (a b : ℝ) :
  f a b (-5) = 17 → f a b 5 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_5_l351_35148


namespace NUMINAMATH_CALUDE_range_of_a_l351_35192

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 2 4, x^2 - a*x - 8 > 0) ∧ 
  (∃ θ : ℝ, a - 1 ≤ Real.sin θ - 2) → 
  a < -2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l351_35192


namespace NUMINAMATH_CALUDE_points_lost_l351_35151

theorem points_lost (first_round : ℕ) (second_round : ℕ) (final_score : ℕ) :
  first_round = 40 →
  second_round = 50 →
  final_score = 86 →
  (first_round + second_round) - final_score = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_points_lost_l351_35151


namespace NUMINAMATH_CALUDE_sequence_sum_l351_35145

theorem sequence_sum (a : ℕ → ℚ) (x y : ℚ) :
  (∀ n, a (n + 1) = a n * (1 / 4)) →
  a 0 = 256 ∧ a 1 = x ∧ a 2 = y ∧ a 3 = 4 →
  x + y = 80 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l351_35145


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l351_35128

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  aluminium : ℕ
  oxygen : ℕ
  hydrogen : ℕ

/-- Calculates the molecular weight of a compound given the number of atoms -/
def molecularWeight (c : Compound) : ℕ :=
  27 * c.aluminium + 16 * c.oxygen + c.hydrogen

/-- Theorem stating that the compound with 3 oxygen atoms satisfies the given conditions -/
theorem compound_oxygen_atoms : 
  ∃ (c : Compound), c.aluminium = 1 ∧ c.hydrogen = 3 ∧ molecularWeight c = 78 ∧ c.oxygen = 3 := by
  sorry

#check compound_oxygen_atoms

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l351_35128


namespace NUMINAMATH_CALUDE_symmetric_points_implies_power_of_negative_two_l351_35199

/-- If points M(3a+b, 8) and N(9, 2a+3b) are symmetric about the x-axis, then (-2)^(2a+b) = 16 -/
theorem symmetric_points_implies_power_of_negative_two (a b : ℝ) : 
  (3 * a + b = 9 ∧ 2 * a + 3 * b = -8) → (-2 : ℝ) ^ (2 * a + b) = 16 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_implies_power_of_negative_two_l351_35199


namespace NUMINAMATH_CALUDE_consecutive_numbers_problem_l351_35190

theorem consecutive_numbers_problem (x y z w : ℤ) : 
  x > y → y > z →  -- x, y, and z are consecutive with x > y > z
  w > x →  -- w is greater than x
  5 * x = 3 * w →  -- ratio of x to w is 3:5
  2 * x + 3 * y + 3 * z = 5 * y + 11 →  -- given equation
  x - y = y - z →  -- consecutive numbers condition
  z = 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_problem_l351_35190


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l351_35122

-- Define an isosceles triangle with a vertex angle of 40°
structure IsoscelesTriangle where
  vertex_angle : ℝ
  is_isosceles : Bool
  vertex_angle_value : vertex_angle = 40

-- Define the property we want to prove
def base_angle_is_70 (triangle : IsoscelesTriangle) : Prop :=
  ∃ (base_angle : ℝ), base_angle = 70 ∧ 
    triangle.vertex_angle + 2 * base_angle = 180

-- State the theorem
theorem isosceles_triangle_base_angle 
  (triangle : IsoscelesTriangle) : 
  base_angle_is_70 triangle :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l351_35122


namespace NUMINAMATH_CALUDE_pigs_joined_l351_35108

theorem pigs_joined (initial_pigs final_pigs : ℕ) (h : initial_pigs ≤ final_pigs) :
  final_pigs - initial_pigs = final_pigs - initial_pigs :=
by sorry

end NUMINAMATH_CALUDE_pigs_joined_l351_35108


namespace NUMINAMATH_CALUDE_rectangle_area_error_percentage_l351_35136

/-- Theorem: Error percentage in rectangle area calculation with measurement errors -/
theorem rectangle_area_error_percentage
  (L W : ℝ)  -- Actual length and width of the rectangle
  (h_L_pos : L > 0)
  (h_W_pos : W > 0)
  : let L_measured := L * (1 + 0.12)
    let W_measured := W * (1 - 0.05)
    let area_actual := L * W
    let area_calculated := L_measured * W_measured
    let error_percentage := (area_calculated - area_actual) / area_actual * 100
    error_percentage = 6.4 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percentage_l351_35136


namespace NUMINAMATH_CALUDE_longest_line_segment_in_quarter_pie_l351_35187

theorem longest_line_segment_in_quarter_pie (d : ℝ) (h : d = 16) :
  let r := d / 2
  let θ := π / 2
  let chord_length := 2 * r * Real.sin (θ / 2)
  chord_length ^ 2 = 128 :=
by sorry

end NUMINAMATH_CALUDE_longest_line_segment_in_quarter_pie_l351_35187


namespace NUMINAMATH_CALUDE_special_triangle_side_lengths_l351_35144

/-- Triangle with consecutive integer side lengths and perpendicular median and angle bisector -/
structure SpecialTriangle where
  -- Side lengths
  a : ℕ
  b : ℕ
  c : ℕ
  -- Consecutive integer side lengths
  consecutive_sides : c = b + 1 ∧ b = a + 1
  -- Median from A
  median_a : ℝ × ℝ
  -- Angle bisector from B
  bisector_b : ℝ × ℝ
  -- Perpendicularity condition
  perpendicular : median_a.1 * bisector_b.1 + median_a.2 * bisector_b.2 = 0

/-- The side lengths of a special triangle are 2, 3, and 4 -/
theorem special_triangle_side_lengths (t : SpecialTriangle) : t.a = 2 ∧ t.b = 3 ∧ t.c = 4 :=
sorry

end NUMINAMATH_CALUDE_special_triangle_side_lengths_l351_35144


namespace NUMINAMATH_CALUDE_bird_count_l351_35163

/-- Represents the count of animals in a nature reserve --/
structure AnimalCount where
  birds : ℕ
  mythical : ℕ
  mammals : ℕ

/-- Theorem stating the number of two-legged birds in the nature reserve --/
theorem bird_count (ac : AnimalCount) : 
  ac.birds + ac.mythical + ac.mammals = 300 →
  2 * ac.birds + 3 * ac.mythical + 4 * ac.mammals = 708 →
  ac.birds = 192 := by
  sorry


end NUMINAMATH_CALUDE_bird_count_l351_35163


namespace NUMINAMATH_CALUDE_marked_elements_not_distinct_l351_35125

theorem marked_elements_not_distinct (marked : Fin 10 → Fin 10) : 
  (∀ i j, i ≠ j → marked i ≠ marked j) → False :=
by
  intro h
  -- The proof goes here
  sorry

#check marked_elements_not_distinct

end NUMINAMATH_CALUDE_marked_elements_not_distinct_l351_35125


namespace NUMINAMATH_CALUDE_product_three_consecutive_divisible_by_six_l351_35186

theorem product_three_consecutive_divisible_by_six (n : ℕ) : 
  6 ∣ (n * (n + 1) * (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_product_three_consecutive_divisible_by_six_l351_35186


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l351_35106

/-- For a positive arithmetic sequence with a_3 = 2, the common difference d is in the range [0, 1). -/
theorem arithmetic_sequence_common_difference_range 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_positive : ∀ n, a n > 0)
  (h_a3 : a 3 = 2) :
  ∃ d, (∀ n, a (n + 1) = a n + d) ∧ 0 ≤ d ∧ d < 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l351_35106


namespace NUMINAMATH_CALUDE_width_of_right_triangle_in_square_l351_35173

/-- A right triangle that fits inside a square -/
structure RightTriangleInSquare where
  height : ℝ
  width : ℝ
  square_side : ℝ
  is_right_triangle : True
  fits_in_square : height ≤ square_side ∧ width ≤ square_side

/-- Theorem: The width of a right triangle with height 2 that fits in a 2x2 square is 2 -/
theorem width_of_right_triangle_in_square
  (triangle : RightTriangleInSquare)
  (h_height : triangle.height = 2)
  (h_square : triangle.square_side = 2) :
  triangle.width = 2 :=
sorry

end NUMINAMATH_CALUDE_width_of_right_triangle_in_square_l351_35173


namespace NUMINAMATH_CALUDE_anns_age_l351_35133

theorem anns_age (a b : ℕ) : 
  a + b = 72 → 
  b = (a / 3 : ℚ) + 2 * (a - b) → 
  a = 46 :=
by sorry

end NUMINAMATH_CALUDE_anns_age_l351_35133


namespace NUMINAMATH_CALUDE_expansion_coefficients_properties_l351_35189

theorem expansion_coefficients_properties :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ),
  (∀ x : ℚ, (2*x + 1)^6 = a₀*x^6 + a₁*x^5 + a₂*x^4 + a₃*x^3 + a₄*x^2 + a₅*x + a₆) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 729) ∧
  (a₁ + a₃ + a₅ = 364) ∧
  (a₂ + a₄ = 300) := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficients_properties_l351_35189


namespace NUMINAMATH_CALUDE_average_study_time_difference_l351_35185

def average_difference (differences : List Int) : ℚ :=
  (differences.sum : ℚ) / differences.length

theorem average_study_time_difference 
  (differences : List Int) 
  (h1 : differences.length = 5) :
  average_difference differences = 
    (differences.sum : ℚ) / 5 := by sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l351_35185


namespace NUMINAMATH_CALUDE_bucket_capacity_proof_l351_35124

theorem bucket_capacity_proof (capacity : ℝ) : 
  (12 * capacity = 108 * 9) → capacity = 81 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_proof_l351_35124


namespace NUMINAMATH_CALUDE_odd_divisor_of_power_plus_one_l351_35167

theorem odd_divisor_of_power_plus_one (n : ℕ) :
  n > 0 ∧ Odd n ∧ n ∣ (3^n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisor_of_power_plus_one_l351_35167


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l351_35105

theorem number_exceeding_fraction : 
  ∀ x : ℚ, x = (3 / 8) * x + 15 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l351_35105


namespace NUMINAMATH_CALUDE_cosine_equation_roots_l351_35100

theorem cosine_equation_roots (θ : Real) :
  (0 ≤ θ) ∧ (θ < 360) →
  (3 * Real.cos θ + 1 / Real.cos θ = 4) →
  ∃ p : Nat, p = 3 := by sorry

end NUMINAMATH_CALUDE_cosine_equation_roots_l351_35100


namespace NUMINAMATH_CALUDE_sqrt_difference_square_l351_35104

theorem sqrt_difference_square : (Real.sqrt 7 + Real.sqrt 6) * (Real.sqrt 7 - Real.sqrt 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_square_l351_35104


namespace NUMINAMATH_CALUDE_custom_mul_one_one_eq_neg_eleven_l351_35193

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 1

/-- Theorem: Given the conditions, 1 * 1 = -11 -/
theorem custom_mul_one_one_eq_neg_eleven 
  (a b : ℝ) 
  (h1 : custom_mul a b 3 5 = 15) 
  (h2 : custom_mul a b 4 7 = 28) : 
  custom_mul a b 1 1 = -11 :=
by sorry

end NUMINAMATH_CALUDE_custom_mul_one_one_eq_neg_eleven_l351_35193


namespace NUMINAMATH_CALUDE_rachel_budget_value_l351_35154

/-- The cost of Sara's shoes -/
def sara_shoes : ℕ := 50

/-- The cost of Sara's dress -/
def sara_dress : ℕ := 200

/-- The cost of Tina's shoes -/
def tina_shoes : ℕ := 70

/-- The cost of Tina's dress -/
def tina_dress : ℕ := 150

/-- Rachel's budget is twice the sum of Sara's and Tina's expenses -/
def rachel_budget : ℕ := 2 * (sara_shoes + sara_dress + tina_shoes + tina_dress)

theorem rachel_budget_value : rachel_budget = 940 := by
  sorry

end NUMINAMATH_CALUDE_rachel_budget_value_l351_35154


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l351_35165

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + (a-1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x : ℝ | x^2 - m*x + 2 = 0}

theorem set_equality_implies_values (a m : ℝ) 
  (h1 : A ∪ B a = A) 
  (h2 : A ∩ C m = C m) : 
  (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l351_35165


namespace NUMINAMATH_CALUDE_square_area_with_inscribed_circle_l351_35112

theorem square_area_with_inscribed_circle (r : ℝ) (h1 : r > 0) 
  (h2 : (r - 1)^2 + (r - 2)^2 = r^2) : (2*r)^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_inscribed_circle_l351_35112


namespace NUMINAMATH_CALUDE_complement_of_complement_is_A_l351_35180

-- Define the universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- Define the complement of A in U
def C_UA : Set ℕ := {5, 7}

-- Define set A
def A : Set ℕ := {1, 3, 9}

-- Theorem statement
theorem complement_of_complement_is_A :
  A = U \ C_UA :=
by sorry

end NUMINAMATH_CALUDE_complement_of_complement_is_A_l351_35180


namespace NUMINAMATH_CALUDE_minimum_at_two_implies_m_geq_five_range_of_m_l351_35188

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - 1| + m * |x - 2| + 6 * |x - 3|

/-- The theorem stating that if f attains its minimum at x = 2, then m ≥ 5 -/
theorem minimum_at_two_implies_m_geq_five (m : ℝ) :
  (∀ x : ℝ, f m x ≥ f m 2) → m ≥ 5 := by
  sorry

/-- The main theorem describing the range of m -/
theorem range_of_m :
  {m : ℝ | ∀ x : ℝ, f m x ≥ f m 2} = {m : ℝ | m ≥ 5} := by
  sorry

end NUMINAMATH_CALUDE_minimum_at_two_implies_m_geq_five_range_of_m_l351_35188


namespace NUMINAMATH_CALUDE_largest_increase_2006_2007_l351_35126

def students : Fin 6 → ℕ
  | 0 => 50  -- 2003
  | 1 => 55  -- 2004
  | 2 => 60  -- 2005
  | 3 => 65  -- 2006
  | 4 => 75  -- 2007
  | 5 => 80  -- 2008

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreasePair : Fin 5 := sorry

theorem largest_increase_2006_2007 :
  largestIncreasePair = 3 ∧
  ∀ i : Fin 5, percentageIncrease (students i) (students (i + 1)) ≤
    percentageIncrease (students 3) (students 4) :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_2006_2007_l351_35126


namespace NUMINAMATH_CALUDE_divisible_by_three_exists_l351_35117

/-- A type representing the arrangement of natural numbers in a circle. -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- Predicate to check if two numbers differ by 1, 2, or by a factor of two. -/
def ValidDifference (a b : ℕ) : Prop :=
  (a = b + 1) ∨ (b = a + 1) ∨ (a = b + 2) ∨ (b = a + 2) ∨ (a = 2 * b) ∨ (b = 2 * a)

/-- Theorem stating that in any arrangement of 99 natural numbers in a circle
    where any two neighboring numbers differ either by 1, or by 2, or by a factor of two,
    at least one of these numbers is divisible by 3. -/
theorem divisible_by_three_exists (arr : CircularArrangement 99)
  (h : ∀ i : Fin 99, ValidDifference (arr i) (arr (i + 1))) :
  ∃ i : Fin 99, 3 ∣ arr i := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_exists_l351_35117


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l351_35141

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / a + 1 / b ≥ 2 ∧ (1 / a + 1 / b = 2 ↔ a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l351_35141


namespace NUMINAMATH_CALUDE_pyramid_cross_section_theorem_l351_35170

/-- Represents a regular pyramid -/
structure RegularPyramid where
  lateralEdgeLength : ℝ

/-- Represents a cross-section of a pyramid -/
structure CrossSection where
  areaRatio : ℝ  -- ratio of cross-section area to base area

/-- 
Given a regular pyramid with lateral edge length 3 cm, if a plane parallel to the base
creates a cross-section with an area 1/9 of the base area, then the lateral edge length
of the smaller pyramid removed is 1 cm.
-/
theorem pyramid_cross_section_theorem (p : RegularPyramid) (cs : CrossSection) :
  p.lateralEdgeLength = 3 → cs.areaRatio = 1/9 → 
  ∃ (smallerPyramid : RegularPyramid), smallerPyramid.lateralEdgeLength = 1 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_cross_section_theorem_l351_35170


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l351_35132

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 20) →
  p + q = 87 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l351_35132


namespace NUMINAMATH_CALUDE_chain_store_sales_theorem_l351_35157

-- Define the basic parameters
def cost_price : ℝ := 60
def initial_selling_price : ℝ := 80
def new_selling_price : ℝ := 100

-- Define the sales functions
def y₁ (x : ℝ) : ℝ := x^2 - 8*x + 56
def y₂ (x : ℝ) : ℝ := 2*x + 8

-- Define the gross profit function for sales > 60
def W (x : ℝ) : ℝ := 8*x^2 - 96*x - 512

-- Theorem statement
theorem chain_store_sales_theorem :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 10 → y₁ x ≥ 0) ∧
  (y₁ 4 = 40) ∧
  (y₁ 6 = 44) ∧
  (∀ x : ℝ, 11 ≤ x ∧ x ≤ 31 → y₂ x ≥ 0) ∧
  ((initial_selling_price - cost_price) * (y₁ 8) = 1120) ∧
  (∀ x : ℝ, 26 < x ∧ x ≤ 31 → W x = (new_selling_price - (cost_price - 2*(y₂ x - 60))) * y₂ x) := by
  sorry


end NUMINAMATH_CALUDE_chain_store_sales_theorem_l351_35157


namespace NUMINAMATH_CALUDE_c_share_calculation_l351_35130

theorem c_share_calculation (total : ℝ) (a b c d : ℝ) : 
  total = 392 →
  a = b / 2 →
  b = c / 2 →
  d = total / 4 →
  a + b + c + d = total →
  c = 168 := by
sorry

end NUMINAMATH_CALUDE_c_share_calculation_l351_35130


namespace NUMINAMATH_CALUDE_cos_300_deg_l351_35113

theorem cos_300_deg : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_deg_l351_35113


namespace NUMINAMATH_CALUDE_base_4_7_digit_difference_l351_35152

def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

theorem base_4_7_digit_difference : 
  num_digits 4563 4 - num_digits 4563 7 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_base_4_7_digit_difference_l351_35152


namespace NUMINAMATH_CALUDE_salary_difference_l351_35175

theorem salary_difference (ram_salary raja_salary : ℝ) 
  (h : raja_salary = 0.8 * ram_salary) : 
  (ram_salary - raja_salary) / raja_salary = 0.25 := by
sorry

end NUMINAMATH_CALUDE_salary_difference_l351_35175


namespace NUMINAMATH_CALUDE_quadratic_always_positive_range_l351_35137

theorem quadratic_always_positive_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_range_l351_35137


namespace NUMINAMATH_CALUDE_hundredth_training_day_l351_35134

def training_program (start_day : Nat) (n : Nat) : Nat :=
  (start_day + (n - 1) * 8 + (n - 1) % 6) % 7

theorem hundredth_training_day :
  training_program 1 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_training_day_l351_35134


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l351_35129

/-- The complex number z = (2+3i)/(1-i) lies in the second quadrant of the complex plane. -/
theorem complex_in_second_quadrant : 
  let z : ℂ := (2 + 3*I) / (1 - I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l351_35129


namespace NUMINAMATH_CALUDE_exists_special_function_l351_35120

/-- Number of divisors function -/
def num_divisors (m : ℕ) : ℕ := sorry

/-- The function we want to prove exists -/
noncomputable def f : ℕ → ℕ := sorry

theorem exists_special_function :
  ∃ (f : ℕ → ℕ),
    (∃ (n : ℕ), f n ≠ n) ∧
    (∀ (m n : ℕ), (num_divisors m = f n) ↔ (num_divisors (f m) = n)) := by
  sorry

end NUMINAMATH_CALUDE_exists_special_function_l351_35120


namespace NUMINAMATH_CALUDE_bc_length_l351_35172

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

-- Define the conditions of the problem
def problem_conditions (t : Triangle) : Prop :=
  t.AB = 5 ∧ t.AC = 6 ∧ Real.sin t.A = 3/5

-- Theorem statement
theorem bc_length (t : Triangle) 
  (h_acute : t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2)
  (h_cond : problem_conditions t) : 
  t.BC = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_bc_length_l351_35172


namespace NUMINAMATH_CALUDE_chess_tournament_games_l351_35146

/-- The number of games played in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 9 players, where each player plays every
    other player exactly once, the total number of games played is 36. -/
theorem chess_tournament_games :
  num_games 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l351_35146


namespace NUMINAMATH_CALUDE_puzzle_solution_l351_35101

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- The problem statement -/
theorem puzzle_solution 
  (EH OY AY OH : TwoDigitNumber)
  (h1 : EH.val = 4 * OY.val)
  (h2 : AY.val = 4 * OH.val) :
  EH.val + OY.val + AY.val + OH.val = 150 :=
sorry

end NUMINAMATH_CALUDE_puzzle_solution_l351_35101


namespace NUMINAMATH_CALUDE_output_value_S_l351_35171

theorem output_value_S : ∃ S : ℕ, S = 1 * 3^1 + 2 * 3^2 + 3 * 3^3 ∧ S = 102 := by
  sorry

end NUMINAMATH_CALUDE_output_value_S_l351_35171


namespace NUMINAMATH_CALUDE_one_root_cubic_equation_a_range_l351_35147

theorem one_root_cubic_equation_a_range (a : ℝ) : 
  (∃! x : ℝ, x^3 + (1-3*a)*x^2 + 2*a^2*x - 2*a*x + x + a^2 - a = 0) → 
  (-Real.sqrt 3 / 2 < a ∧ a < Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_one_root_cubic_equation_a_range_l351_35147


namespace NUMINAMATH_CALUDE_fixed_point_exists_l351_35191

/-- For any a > 0 and a ≠ 1, the function f(x) = ax - 5 has a fixed point at x = 2 -/
theorem fixed_point_exists (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x : ℝ, a * x - 5 = x ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exists_l351_35191


namespace NUMINAMATH_CALUDE_square_roots_of_nine_l351_35159

theorem square_roots_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_square_roots_of_nine_l351_35159


namespace NUMINAMATH_CALUDE_production_days_calculation_l351_35153

theorem production_days_calculation (n : ℕ) : 
  (∀ (P : ℝ), P / n = 60 → 
    (P + 90) / (n + 1) = 62) → 
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_production_days_calculation_l351_35153


namespace NUMINAMATH_CALUDE_real_part_of_z_l351_35131

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l351_35131


namespace NUMINAMATH_CALUDE_selection_and_assignment_problem_l351_35176

def number_of_ways (male_students female_students total_selected num_tasks : ℕ) : ℕ :=
  sorry

theorem selection_and_assignment_problem :
  let male_students := 4
  let female_students := 3
  let total_selected := 4
  let num_tasks := 3
  number_of_ways male_students female_students total_selected num_tasks = 792 := by
  sorry

end NUMINAMATH_CALUDE_selection_and_assignment_problem_l351_35176


namespace NUMINAMATH_CALUDE_total_triangles_is_68_l351_35158

/-- Represents a rectangle divided into triangles as described in the problem -/
structure DividedRectangle where
  -- The rectangle is divided into 4 quarters
  quarters : Nat
  -- Number of smallest triangles in each quarter
  smallestTrianglesPerQuarter : Nat
  -- Number of half-inner rectangles from smaller triangles
  halfInnerRectangles : Nat
  -- Number of central and side isosceles triangles
  isoscelesTriangles : Nat
  -- Number of large right triangles covering half the rectangle
  largeRightTriangles : Nat
  -- Number of largest isosceles triangles
  largestIsoscelesTriangles : Nat

/-- Calculates the total number of triangles in the divided rectangle -/
def totalTriangles (r : DividedRectangle) : Nat :=
  r.smallestTrianglesPerQuarter * r.quarters +
  r.halfInnerRectangles +
  r.isoscelesTriangles +
  r.largeRightTriangles +
  r.largestIsoscelesTriangles

/-- The specific rectangle configuration from the problem -/
def problemRectangle : DividedRectangle where
  quarters := 4
  smallestTrianglesPerQuarter := 8
  halfInnerRectangles := 16
  isoscelesTriangles := 8
  largeRightTriangles := 8
  largestIsoscelesTriangles := 4

theorem total_triangles_is_68 : totalTriangles problemRectangle = 68 := by
  sorry


end NUMINAMATH_CALUDE_total_triangles_is_68_l351_35158


namespace NUMINAMATH_CALUDE_new_person_weight_l351_35115

/-- Given a group of 8 people, if replacing one person weighing 65 kg
    with a new person increases the average weight by 3.5 kg,
    then the weight of the new person is 93 kg. -/
theorem new_person_weight
  (initial_count : Nat)
  (weight_increase : ℝ)
  (replaced_weight : ℝ)
  (h1 : initial_count = 8)
  (h2 : weight_increase = 3.5)
  (h3 : replaced_weight = 65)
  : ℝ :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l351_35115


namespace NUMINAMATH_CALUDE_min_angle_between_planes_l351_35135

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Represents a line in 3D space -/
structure Line where
  direction : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Given two planes, compute the angle between them -/
def angle_between_planes (p1 p2 : Plane) : ℝ := sorry

/-- Check if a plane is perpendicular to a line -/
def plane_perpendicular_to_line (p : Plane) (l : Line) : Prop := sorry

/-- Check if a plane is parallel to a line -/
def plane_parallel_to_line (p : Plane) (l : Line) : Prop := sorry

/-- Extract the line A₁C₁ from a cube -/
def line_A₁C₁ (c : Cube) : Line := sorry

/-- Extract the line CD₁ from a cube -/
def line_CD₁ (c : Cube) : Line := sorry

theorem min_angle_between_planes (c : Cube) (α β : Plane) 
  (h1 : plane_perpendicular_to_line α (line_A₁C₁ c))
  (h2 : plane_parallel_to_line β (line_CD₁ c)) :
  ∃ (θ : ℝ), (∀ (α' β' : Plane), 
    plane_perpendicular_to_line α' (line_A₁C₁ c) →
    plane_parallel_to_line β' (line_CD₁ c) →
    angle_between_planes α' β' ≥ θ) ∧
  θ = π / 6 := by sorry

end NUMINAMATH_CALUDE_min_angle_between_planes_l351_35135


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l351_35156

theorem pencil_pen_cost (pencil_cost pen_cost : ℝ) 
  (h1 : 5 * pencil_cost + pen_cost = 2.50)
  (h2 : pencil_cost + 2 * pen_cost = 1.85) :
  2 * pencil_cost + pen_cost = 1.45 := by
sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l351_35156


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l351_35166

theorem partial_fraction_decomposition :
  ∃ (A B C D : ℚ),
    (A = 1/15) ∧ (B = 5/2) ∧ (C = -59/6) ∧ (D = 42/5) ∧
    (∀ x : ℚ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 →
      (x^3 - 7) / ((x - 2) * (x - 3) * (x - 5) * (x - 7)) =
      A / (x - 2) + B / (x - 3) + C / (x - 5) + D / (x - 7)) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l351_35166


namespace NUMINAMATH_CALUDE_exists_non_regular_triangle_with_similar_median_triangle_l351_35110

/-- Represents a triangle with sides a, b, c and medians s_a, s_b, s_c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  s_a : ℝ
  s_b : ℝ
  s_c : ℝ
  h_order : a ≤ b ∧ b ≤ c
  h_median_a : 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2
  h_median_b : 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2
  h_median_c : 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2

/-- Two triangles are similar if the ratios of their corresponding sides are equal -/
def similar (t1 t2 : Triangle) : Prop :=
  (t1.a / t2.a)^2 = (t1.b / t2.b)^2 ∧ (t1.b / t2.b)^2 = (t1.c / t2.c)^2

/-- A triangle is regular if all its sides are equal -/
def regular (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem exists_non_regular_triangle_with_similar_median_triangle :
  ∃ t : Triangle, ¬regular t ∧ similar t ⟨t.s_a, t.s_b, t.s_c, 0, 0, 0, sorry, sorry, sorry, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_exists_non_regular_triangle_with_similar_median_triangle_l351_35110


namespace NUMINAMATH_CALUDE_distance_between_points_l351_35121

/-- The distance between two points (5, -3) and (9, 6) in a 2D plane is √97 units. -/
theorem distance_between_points : Real.sqrt 97 = Real.sqrt ((9 - 5)^2 + (6 - (-3))^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l351_35121


namespace NUMINAMATH_CALUDE_tyler_meal_combinations_l351_35182

/-- The number of types of meat available -/
def num_meats : ℕ := 4

/-- The number of types of vegetables available -/
def num_vegetables : ℕ := 5

/-- The number of types of desserts available -/
def num_desserts : ℕ := 5

/-- The number of types of drinks available -/
def num_drinks : ℕ := 4

/-- The number of vegetables Tyler must choose -/
def vegetables_to_choose : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The theorem stating the number of different meal combinations Tyler can choose -/
theorem tyler_meal_combinations : 
  num_meats * choose num_vegetables vegetables_to_choose * num_desserts * num_drinks = 800 := by
  sorry

end NUMINAMATH_CALUDE_tyler_meal_combinations_l351_35182


namespace NUMINAMATH_CALUDE_art_of_passing_through_walls_l351_35161

theorem art_of_passing_through_walls (n : ℝ) :
  (8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n))) ↔ n = 63 := by
  sorry

end NUMINAMATH_CALUDE_art_of_passing_through_walls_l351_35161


namespace NUMINAMATH_CALUDE_changed_number_proof_l351_35143

theorem changed_number_proof (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 8 →
  (8 + b + c + d + e) / 5 = 9 →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_changed_number_proof_l351_35143


namespace NUMINAMATH_CALUDE_proper_subsets_count_l351_35179

def S : Finset ℕ := {0, 3, 4}

theorem proper_subsets_count : (Finset.powerset S).card - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_proper_subsets_count_l351_35179


namespace NUMINAMATH_CALUDE_x_range_for_positive_f_l351_35139

/-- The function f(x) for a given a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

/-- The theorem stating the range of x given the conditions -/
theorem x_range_for_positive_f :
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, ∀ x, f a x > 0) →
  (∀ x, x < 1 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_positive_f_l351_35139


namespace NUMINAMATH_CALUDE_number_of_boys_l351_35107

def total_students : ℕ := 1150

def is_valid_distribution (boys : ℕ) : Prop :=
  let girls := (boys * 100) / total_students
  boys + girls = total_students

theorem number_of_boys : ∃ (boys : ℕ), boys = 1058 ∧ is_valid_distribution boys := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l351_35107


namespace NUMINAMATH_CALUDE_incenter_coeff_sum_specific_triangle_incenter_l351_35150

/-- Given a triangle XYZ with sides x, y, z, the position vector of its incenter J
    can be expressed as J⃗ = p X⃗ + q Y⃗ + r Z⃗, where p, q, r are constants. -/
def incenter_position_vector (x y z : ℝ) (p q r : ℝ) : Prop :=
  p = x / (x + y + z) ∧ q = y / (x + y + z) ∧ r = z / (x + y + z)

/-- The sum of coefficients p, q, r in the incenter position vector equation is 1. -/
theorem incenter_coeff_sum (x y z : ℝ) (p q r : ℝ) 
  (h : incenter_position_vector x y z p q r) : p + q + r = 1 := by sorry

/-- For a triangle with sides 8, 11, and 5, the position vector of its incenter
    is given by (1/3, 11/24, 5/24). -/
theorem specific_triangle_incenter : 
  incenter_position_vector 8 11 5 (1/3) (11/24) (5/24) := by sorry

end NUMINAMATH_CALUDE_incenter_coeff_sum_specific_triangle_incenter_l351_35150


namespace NUMINAMATH_CALUDE_no_geometric_progression_2_3_5_l351_35168

theorem no_geometric_progression_2_3_5 : 
  ¬ (∃ (a r : ℝ) (k n : ℕ), 
    a > 0 ∧ r > 0 ∧ 
    a * r^0 = 2 ∧
    a * r^k = 3 ∧
    a * r^n = 5 ∧
    0 < k ∧ k < n) :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_progression_2_3_5_l351_35168


namespace NUMINAMATH_CALUDE_janes_change_calculation_l351_35118

-- Define the prices and quantities
def skirt_price : ℝ := 65
def skirt_quantity : ℕ := 2
def blouse_price : ℝ := 30
def blouse_quantity : ℕ := 3
def shoes_price : ℝ := 125
def handbag_price : ℝ := 175

-- Define the discounts and taxes
def handbag_discount : ℝ := 0.10
def total_discount : ℝ := 0.05
def coupon_discount : ℝ := 20
def sales_tax : ℝ := 0.08

-- Define the exchange rate and amount paid
def exchange_rate : ℝ := 0.8
def amount_paid : ℝ := 600

-- Theorem to prove
theorem janes_change_calculation :
  let initial_total := skirt_price * skirt_quantity + blouse_price * blouse_quantity + shoes_price + handbag_price
  let handbag_discounted := initial_total - handbag_discount * handbag_price
  let total_discounted := handbag_discounted * (1 - total_discount)
  let coupon_applied := total_discounted - coupon_discount
  let taxed_total := coupon_applied * (1 + sales_tax)
  let home_currency_total := taxed_total * exchange_rate
  amount_paid - home_currency_total = 204.828 := by sorry

end NUMINAMATH_CALUDE_janes_change_calculation_l351_35118
