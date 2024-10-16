import Mathlib

namespace NUMINAMATH_CALUDE_T_not_subset_S_l879_87937

def S : Set ℤ := {x | ∃ n : ℤ, x = 2 * n + 1}
def T : Set ℤ := {y | ∃ k : ℤ, y = 4 * k + 1}

theorem T_not_subset_S : ¬(T ⊆ S) := by
  sorry

end NUMINAMATH_CALUDE_T_not_subset_S_l879_87937


namespace NUMINAMATH_CALUDE_decimal_93_to_binary_binary_to_decimal_93_l879_87965

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec toBinaryAux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: toBinaryAux (m / 2)
    toBinaryAux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_93_to_binary :
  toBinary 93 = [true, false, true, true, true, false, true] := by
  sorry

theorem binary_to_decimal_93 :
  fromBinary [true, false, true, true, true, false, true] = 93 := by
  sorry

end NUMINAMATH_CALUDE_decimal_93_to_binary_binary_to_decimal_93_l879_87965


namespace NUMINAMATH_CALUDE_smallest_perimeter_circle_circle_center_on_L_l879_87958

-- Define the points A and B
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (-1, 4)

-- Define the line L
def L (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Define a general circle equation
def isCircle (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

-- Define a circle passing through two points
def circlePassingThrough (h k r : ℝ) : Prop :=
  isCircle h k r A.1 A.2 ∧ isCircle h k r B.1 B.2

-- Theorem for the smallest perimeter circle
theorem smallest_perimeter_circle :
  ∃ (h k r : ℝ), circlePassingThrough h k r ∧
  isCircle h k r = fun x y => x^2 + (y - 1)^2 = 10 :=
sorry

-- Theorem for the circle with center on line L
theorem circle_center_on_L :
  ∃ (h k r : ℝ), circlePassingThrough h k r ∧
  L h k ∧
  isCircle h k r = fun x y => (x - 3)^2 + (y - 2)^2 = 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_circle_circle_center_on_L_l879_87958


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l879_87955

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(2*x - 1) + 1
  f (1/2) = 2 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l879_87955


namespace NUMINAMATH_CALUDE_incoming_scholars_count_l879_87907

theorem incoming_scholars_count :
  ∃! n : ℕ, n < 600 ∧ n % 15 = 14 ∧ n % 19 = 13 ∧ n = 509 := by
  sorry

end NUMINAMATH_CALUDE_incoming_scholars_count_l879_87907


namespace NUMINAMATH_CALUDE_sum_of_cubes_l879_87991

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a * b + a * c + b * c = -4) 
  (h3 : a * b * c = -6) : 
  a^3 + b^3 + c^3 = -5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l879_87991


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l879_87992

/-- The diagonal of a rectangular prism with dimensions 15, 25, and 15 is 5√43 -/
theorem rectangular_prism_diagonal : 
  ∀ (a b c d : ℝ), 
    a = 15 → 
    b = 25 → 
    c = 15 → 
    d ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 → 
    d = 5 * Real.sqrt 43 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l879_87992


namespace NUMINAMATH_CALUDE_percentage_of_x_minus_y_l879_87930

theorem percentage_of_x_minus_y (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (30 / 100) * (x + y) →
  y = (25 / 100) * x →
  P = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_x_minus_y_l879_87930


namespace NUMINAMATH_CALUDE_additional_friends_average_weight_l879_87989

theorem additional_friends_average_weight
  (initial_count : ℕ)
  (additional_count : ℕ)
  (average_increase : ℝ)
  (final_average : ℝ)
  (h1 : initial_count = 50)
  (h2 : additional_count = 40)
  (h3 : average_increase = 12)
  (h4 : final_average = 46) :
  let total_count := initial_count + additional_count
  let initial_average := final_average - average_increase
  let initial_total_weight := initial_count * initial_average
  let final_total_weight := total_count * final_average
  let additional_total_weight := final_total_weight - initial_total_weight
  let additional_average := additional_total_weight / additional_count
  additional_average = 61 := by
sorry

end NUMINAMATH_CALUDE_additional_friends_average_weight_l879_87989


namespace NUMINAMATH_CALUDE_inequality_equivalence_l879_87932

theorem inequality_equivalence (x : ℝ) : 
  (12 * x^2 + 24 * x - 75) / ((3 * x - 5) * (x + 5)) < 4 ↔ -5 < x ∧ x < 5/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l879_87932


namespace NUMINAMATH_CALUDE_trailing_zeros_for_specific_fraction_l879_87936

/-- The number of trailing zeros in the decimal representation of a rational number -/
def trailingZeros (n d : ℕ) : ℕ :=
  sorry

/-- The main theorem: number of trailing zeros for 1 / (2^3 * 5^7) -/
theorem trailing_zeros_for_specific_fraction :
  trailingZeros 1 (2^3 * 5^7) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_for_specific_fraction_l879_87936


namespace NUMINAMATH_CALUDE_longest_side_is_ten_l879_87917

/-- A rectangular solid with given face areas and volume -/
structure RectangularSolid where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  volume : ℝ

/-- The longest side of a rectangular solid -/
def longest_side (r : RectangularSolid) : ℝ := sorry

/-- Theorem stating the longest side of the given rectangular solid is 10 -/
theorem longest_side_is_ten (r : RectangularSolid) 
  (h1 : r.area1 = 20) 
  (h2 : r.area2 = 15) 
  (h3 : r.area3 = 12) 
  (h4 : r.volume = 60) : 
  longest_side r = 10 := by sorry

end NUMINAMATH_CALUDE_longest_side_is_ten_l879_87917


namespace NUMINAMATH_CALUDE_absolute_value_problem_l879_87903

theorem absolute_value_problem (m n : ℤ) 
  (hm : |m| = 4) (hn : |n| = 3) : 
  ((m * n > 0 → |m - n| = 1) ∧ 
   (m * n < 0 → |m + n| = 1)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l879_87903


namespace NUMINAMATH_CALUDE_min_rectangles_to_cover_square_l879_87961

-- Define the dimensions of the rectangle
def rectangle_width : ℕ := 3
def rectangle_height : ℕ := 4

-- Define the area of the rectangle
def rectangle_area : ℕ := rectangle_width * rectangle_height

-- Define the function to calculate the number of rectangles needed
def rectangles_needed (square_side : ℕ) : ℕ :=
  (square_side * square_side) / rectangle_area

-- Theorem statement
theorem min_rectangles_to_cover_square :
  ∃ (n : ℕ), 
    n > 0 ∧
    rectangles_needed n = 12 ∧
    ∀ (m : ℕ), m > 0 → rectangles_needed m ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_rectangles_to_cover_square_l879_87961


namespace NUMINAMATH_CALUDE_binomial_coefficient_17_16_l879_87972

theorem binomial_coefficient_17_16 : Nat.choose 17 16 = 17 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_17_16_l879_87972


namespace NUMINAMATH_CALUDE_square_area_problem_l879_87924

theorem square_area_problem (small_square_area : ℝ) (triangle_area : ℝ) :
  small_square_area = 16 →
  triangle_area = 1 →
  ∃ (large_square_area : ℝ),
    large_square_area = 18 ∧
    ∃ (small_side large_side triangle_side : ℝ),
      small_side ^ 2 = small_square_area ∧
      triangle_side ^ 2 = 2 ∧
      large_side ^ 2 = large_square_area ∧
      large_side ^ 2 = small_side ^ 2 + triangle_side ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_square_area_problem_l879_87924


namespace NUMINAMATH_CALUDE_tetrahedron_relationships_l879_87984

/-- Properties of a tetrahedron with inscribed and face-touching spheres -/
structure Tetrahedron where
  ρ : ℝ  -- radius of inscribed sphere
  ρ₁ : ℝ  -- radius of sphere touching face opposite to A
  ρ₂ : ℝ  -- radius of sphere touching face opposite to B
  ρ₃ : ℝ  -- radius of sphere touching face opposite to C
  ρ₄ : ℝ  -- radius of sphere touching face opposite to D
  m₁ : ℝ  -- length of altitude from A to opposite face
  m₂ : ℝ  -- length of altitude from B to opposite face
  m₃ : ℝ  -- length of altitude from C to opposite face
  m₄ : ℝ  -- length of altitude from D to opposite face
  ρ_pos : 0 < ρ
  ρ₁_pos : 0 < ρ₁
  ρ₂_pos : 0 < ρ₂
  ρ₃_pos : 0 < ρ₃
  ρ₄_pos : 0 < ρ₄
  m₁_pos : 0 < m₁
  m₂_pos : 0 < m₂
  m₃_pos : 0 < m₃
  m₄_pos : 0 < m₄

/-- Theorem about relationships in a tetrahedron -/
theorem tetrahedron_relationships (t : Tetrahedron) :
  (2 / t.ρ = 1 / t.ρ₁ + 1 / t.ρ₂ + 1 / t.ρ₃ + 1 / t.ρ₄) ∧
  (1 / t.ρ = 1 / t.m₁ + 1 / t.m₂ + 1 / t.m₃ + 1 / t.m₄) ∧
  (1 / t.ρ₁ = -1 / t.m₁ + 1 / t.m₂ + 1 / t.m₃ + 1 / t.m₄) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_relationships_l879_87984


namespace NUMINAMATH_CALUDE_display_window_problem_l879_87906

/-- The number of configurations for two display windows --/
def total_configurations : ℕ := 36

/-- The number of non-fiction books in the right window --/
def non_fiction_books : ℕ := 3

/-- The number of fiction books in the left window --/
def fiction_books : ℕ := 3

theorem display_window_problem :
  fiction_books.factorial * non_fiction_books.factorial = total_configurations :=
sorry

end NUMINAMATH_CALUDE_display_window_problem_l879_87906


namespace NUMINAMATH_CALUDE_fraction_equality_l879_87963

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : a^2 + b^2 ≠ 0) (h4 : a^4 - 2*b^4 ≠ 0) 
  (h5 : a^2 * b^2 / (a^4 - 2*b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l879_87963


namespace NUMINAMATH_CALUDE_hash_three_six_l879_87976

/-- Custom operation # defined for any two real numbers -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem stating that 3 # 6 = 48 -/
theorem hash_three_six : hash 3 6 = 48 := by sorry

end NUMINAMATH_CALUDE_hash_three_six_l879_87976


namespace NUMINAMATH_CALUDE_max_value_of_c_max_value_of_c_achieved_l879_87901

theorem max_value_of_c (x : ℝ) (c : ℝ) (h1 : x > 1) (h2 : c = 2 - x + 2 * Real.sqrt (x - 1)) :
  c ≤ 2 :=
by sorry

theorem max_value_of_c_achieved (x : ℝ) :
  ∃ c, x > 1 ∧ c = 2 - x + 2 * Real.sqrt (x - 1) ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_c_max_value_of_c_achieved_l879_87901


namespace NUMINAMATH_CALUDE_remaining_episodes_l879_87954

theorem remaining_episodes (total_seasons : ℕ) (episodes_per_season : ℕ) 
  (watched_fraction : ℚ) : 
  total_seasons = 12 → 
  episodes_per_season = 20 → 
  watched_fraction = 1/3 →
  total_seasons * episodes_per_season - (watched_fraction * (total_seasons * episodes_per_season)).num = 160 := by
  sorry

end NUMINAMATH_CALUDE_remaining_episodes_l879_87954


namespace NUMINAMATH_CALUDE_inequality_proof_l879_87904

theorem inequality_proof (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧ 
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l879_87904


namespace NUMINAMATH_CALUDE_problem_solution_l879_87945

-- Define the variables
def x : ℝ := 12 * (1 + 0.2)
def y : ℝ := 0.75 * x^2
def z : ℝ := 3 * y + 16
def w : ℝ := 2 * z - y
def v : ℝ := z^3 - 0.5 * y

-- State the theorem
theorem problem_solution :
  v = 112394885.1456 ∧ w = 809.6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l879_87945


namespace NUMINAMATH_CALUDE_tank_filling_time_l879_87957

/-- The time required to fill a tank with different valve combinations -/
theorem tank_filling_time 
  (fill_time_xyz : Real) 
  (fill_time_xz : Real) 
  (fill_time_yz : Real) 
  (h1 : fill_time_xyz = 2)
  (h2 : fill_time_xz = 4)
  (h3 : fill_time_yz = 3) :
  let rate_x := 1 / fill_time_xz - 1 / fill_time_xyz
  let rate_y := 1 / fill_time_yz - 1 / fill_time_xyz
  1 / (rate_x + rate_y) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l879_87957


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l879_87982

/-- The complex number z = i² + i³ corresponds to a point in the third quadrant of the complex plane -/
theorem complex_number_in_third_quadrant :
  let z : ℂ := Complex.I^2 + Complex.I^3
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l879_87982


namespace NUMINAMATH_CALUDE_remaining_calculation_l879_87997

def calculate_remaining (income : ℝ) : ℝ :=
  let after_rent := income * (1 - 0.15)
  let after_education := after_rent * (1 - 0.15)
  let after_misc := after_education * (1 - 0.10)
  let after_medical := after_misc * (1 - 0.15)
  after_medical

theorem remaining_calculation (income : ℝ) :
  income = 10037.77 →
  calculate_remaining income = 5547.999951125 := by
  sorry

end NUMINAMATH_CALUDE_remaining_calculation_l879_87997


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l879_87910

def maya_age : ℕ := 15
def drew_age : ℕ := maya_age + 5
def peter_age : ℕ := drew_age + 4
def john_age : ℕ := 30
def jacob_age : ℕ := 11

theorem age_ratio_in_two_years :
  (jacob_age + 2) / (peter_age + 2) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l879_87910


namespace NUMINAMATH_CALUDE_min_xy_min_x_plus_y_l879_87933

-- Define the conditions
def condition (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ 2 * x + 8 * y - x * y = 0

-- Theorem for the minimum value of xy
theorem min_xy (x y : ℝ) (h : condition x y) :
  x * y ≥ 64 ∧ ∃ x y, condition x y ∧ x * y = 64 :=
sorry

-- Theorem for the minimum value of x + y
theorem min_x_plus_y (x y : ℝ) (h : condition x y) :
  x + y ≥ 18 ∧ ∃ x y, condition x y ∧ x + y = 18 :=
sorry

end NUMINAMATH_CALUDE_min_xy_min_x_plus_y_l879_87933


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l879_87926

theorem students_in_both_clubs
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (either_club : ℕ)
  (h1 : total_students = 400)
  (h2 : drama_club = 180)
  (h3 : science_club = 230)
  (h4 : either_club = 350) :
  drama_club + science_club - either_club = 60 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l879_87926


namespace NUMINAMATH_CALUDE_strawberry_weight_sum_l879_87900

theorem strawberry_weight_sum (marco_weight dad_weight : ℕ) 
  (h1 : marco_weight = 15) 
  (h2 : dad_weight = 22) : 
  marco_weight + dad_weight = 37 := by
sorry

end NUMINAMATH_CALUDE_strawberry_weight_sum_l879_87900


namespace NUMINAMATH_CALUDE_seventh_group_draw_l879_87966

/-- Represents the systematic sampling method for a population -/
structure SystematicSampling where
  population_size : Nat
  num_groups : Nat
  sample_size : Nat
  m : Nat

/-- Calculates the number drawn from a specific group -/
def number_drawn (ss : SystematicSampling) (group : Nat) : Nat :=
  let group_size := ss.population_size / ss.num_groups
  let start := (group - 1) * group_size
  let units_digit := (ss.m + group) % 10
  start + units_digit

/-- Theorem stating that the number drawn from the 7th group is 63 -/
theorem seventh_group_draw (ss : SystematicSampling) 
  (h1 : ss.population_size = 100)
  (h2 : ss.num_groups = 10)
  (h3 : ss.sample_size = 10)
  (h4 : ss.m = 6) :
  number_drawn ss 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_seventh_group_draw_l879_87966


namespace NUMINAMATH_CALUDE_count_common_divisors_84_90_l879_87915

def common_divisors (a b : ℕ) : Finset ℕ :=
  (Finset.range (min a b + 1)).filter (fun d => d > 1 ∧ a % d = 0 ∧ b % d = 0)

theorem count_common_divisors_84_90 :
  (common_divisors 84 90).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_common_divisors_84_90_l879_87915


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l879_87964

theorem polynomial_divisibility (C D : ℚ) : 
  (∀ x : ℂ, x^2 - x + 1 = 0 → x^103 + C*x^2 + D = 0) → 
  C = -1 ∧ D = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l879_87964


namespace NUMINAMATH_CALUDE_factory_production_l879_87934

/-- Represents a machine in the factory -/
structure Machine where
  rate : Nat  -- shirts produced per minute
  time_yesterday : Nat  -- minutes worked yesterday
  time_today : Nat  -- minutes worked today

/-- Calculates the total number of shirts produced by all machines -/
def total_shirts (machines : List Machine) : Nat :=
  machines.foldl (fun acc m => acc + m.rate * (m.time_yesterday + m.time_today)) 0

/-- Theorem: Given the specified machines, the total number of shirts produced is 432 -/
theorem factory_production : 
  let machines : List Machine := [
    { rate := 6, time_yesterday := 12, time_today := 10 },  -- Machine A
    { rate := 8, time_yesterday := 10, time_today := 15 },  -- Machine B
    { rate := 5, time_yesterday := 20, time_today := 0 }    -- Machine C
  ]
  total_shirts machines = 432 := by
  sorry


end NUMINAMATH_CALUDE_factory_production_l879_87934


namespace NUMINAMATH_CALUDE_bus_speed_with_stoppages_l879_87923

/-- Calculates the speed of a bus including stoppages -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (h1 : speed_without_stoppages = 50) 
  (h2 : stoppage_time = 8.4 / 60) : 
  ∃ (speed_with_stoppages : ℝ), 
    speed_with_stoppages = speed_without_stoppages * (1 - stoppage_time) ∧ 
    speed_with_stoppages = 43 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_with_stoppages_l879_87923


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l879_87969

theorem nearest_integer_to_power : 
  ∃ n : ℤ, n = 7414 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l879_87969


namespace NUMINAMATH_CALUDE_city_college_juniors_seniors_l879_87912

theorem city_college_juniors_seniors (total : ℕ) (j s : ℕ) : 
  total = 300 →
  j + s = total →
  (1 : ℚ) / 3 * j = (2 : ℚ) / 3 * s →
  j - s = 100 :=
by sorry

end NUMINAMATH_CALUDE_city_college_juniors_seniors_l879_87912


namespace NUMINAMATH_CALUDE_carey_chairs_moved_l879_87940

/-- Proves that Carey moved 28 chairs given the total chairs, Pat's chairs, and remaining chairs. -/
theorem carey_chairs_moved (total : ℕ) (pat_moved : ℕ) (remaining : ℕ) 
  (h1 : total = 74)
  (h2 : pat_moved = 29)
  (h3 : remaining = 17) :
  total - pat_moved - remaining = 28 := by
  sorry

#check carey_chairs_moved

end NUMINAMATH_CALUDE_carey_chairs_moved_l879_87940


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l879_87993

theorem similar_triangle_longest_side 
  (a b c : ℝ) 
  (ha : a = 8) 
  (hb : b = 10) 
  (hc : c = 12) 
  (perimeter : ℝ) 
  (h_perimeter : perimeter = 150) :
  let scale := perimeter / (a + b + c)
  max (scale * a) (max (scale * b) (scale * c)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_l879_87993


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l879_87943

-- Define the sequence c_n
def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c n - 4 * c (n + 1) + 2008

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n ≥ 2 then
    5 * (c (n + 1) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501
  else
    0  -- Define a value for n < 2, though it's not used in the theorem

-- Theorem statement
theorem a_is_perfect_square (n : ℕ) (h : n > 2) :
  ∃ k : ℤ, a n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_a_is_perfect_square_l879_87943


namespace NUMINAMATH_CALUDE_five_digit_divisibility_l879_87987

/-- A five-digit number -/
def FiveDigitNumber : Type := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- A four-digit number -/
def FourDigitNumber : Type := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Extract the four-digit number from a five-digit number by removing the middle digit -/
def extractFourDigit (n : FiveDigitNumber) : FourDigitNumber :=
  sorry

theorem five_digit_divisibility (n : FiveDigitNumber) :
  (∃ (m : FourDigitNumber), m = extractFourDigit n ∧ n.val % m.val = 0) ↔ n.val % 1000 = 0 :=
sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_l879_87987


namespace NUMINAMATH_CALUDE_fraction_simplification_l879_87913

theorem fraction_simplification :
  (1/2 - 1/3) / ((3/7) * (2/8)) = 14/9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l879_87913


namespace NUMINAMATH_CALUDE_expression_value_for_2016_l879_87938

theorem expression_value_for_2016 :
  let x : ℤ := 2016
  (x^2 - x) - (x^2 - 2*x + 1) = 2015 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_for_2016_l879_87938


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l879_87970

theorem pentagon_angle_measure (a b c d e : ℝ) : 
  -- Pentagon angles sum to 540 degrees
  a + b + c + d + e = 540 ∧
  -- Four angles are congruent
  a = b ∧ b = c ∧ c = d ∧
  -- The fifth angle is 50 degrees more than each of the other angles
  e = a + 50 →
  -- The measure of the fifth angle is 148 degrees
  e = 148 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l879_87970


namespace NUMINAMATH_CALUDE_cosine_largest_angle_bound_l879_87920

/-- Represents a sequence of non-degenerate triangles -/
def TriangleSequence := ℕ → (ℝ × ℝ × ℝ)

/-- Conditions for a valid triangle sequence -/
def IsValidTriangleSequence (seq : TriangleSequence) : Prop :=
  ∀ n : ℕ, let (a, b, c) := seq n
    0 < a ∧ a ≤ b ∧ b ≤ c ∧ a + b > c

/-- Sum of the shortest sides of the triangles -/
noncomputable def SumShortestSides (seq : TriangleSequence) : ℝ :=
  ∑' n, (seq n).1

/-- Sum of the second longest sides of the triangles -/
noncomputable def SumSecondLongestSides (seq : TriangleSequence) : ℝ :=
  ∑' n, (seq n).2.1

/-- Sum of the longest sides of the triangles -/
noncomputable def SumLongestSides (seq : TriangleSequence) : ℝ :=
  ∑' n, (seq n).2.2

/-- Cosine of the largest angle of the resultant triangle -/
noncomputable def CosLargestAngle (seq : TriangleSequence) : ℝ :=
  let A := SumShortestSides seq
  let B := SumSecondLongestSides seq
  let C := SumLongestSides seq
  (A^2 + B^2 - C^2) / (2 * A * B)

/-- The main theorem stating that the cosine of the largest angle is bounded below by 1 - √2 -/
theorem cosine_largest_angle_bound (seq : TriangleSequence) 
  (h : IsValidTriangleSequence seq) : 
  CosLargestAngle seq ≥ 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_largest_angle_bound_l879_87920


namespace NUMINAMATH_CALUDE_complex_triple_solution_l879_87983

theorem complex_triple_solution (x y z : ℂ) :
  (x + y)^3 + (y + z)^3 + (z + x)^3 - 3*(x + y)*(y + z)*(z + x) = 0 →
  x^2*(y + z) + y^2*(z + x) + z^2*(x + y) = 0 →
  x + y + z = 0 ∧ x*y*z = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_triple_solution_l879_87983


namespace NUMINAMATH_CALUDE_total_weight_of_aluminum_carbonate_l879_87988

/-- Atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Number of Aluminum atoms in Al2(CO3)3 -/
def Al_count : ℕ := 2

/-- Number of Carbon atoms in Al2(CO3)3 -/
def C_count : ℕ := 3

/-- Number of Oxygen atoms in Al2(CO3)3 -/
def O_count : ℕ := 9

/-- Number of moles of Al2(CO3)3 -/
def moles : ℝ := 6

/-- Calculates the molecular weight of Al2(CO3)3 in g/mol -/
def molecular_weight : ℝ :=
  Al_count * Al_weight + C_count * C_weight + O_count * O_weight

/-- Theorem stating the total weight of 6 moles of Al2(CO3)3 -/
theorem total_weight_of_aluminum_carbonate :
  moles * molecular_weight = 1403.94 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_aluminum_carbonate_l879_87988


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l879_87960

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n-1)

theorem fifth_term_of_sequence (y : ℝ) :
  geometric_sequence 3 (4*y) 5 = 768 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l879_87960


namespace NUMINAMATH_CALUDE_gabrielle_blue_jays_eq_three_l879_87918

/-- The number of blue jays Gabrielle saw -/
def gabrielle_blue_jays : ℕ :=
  let gabrielle_robins : ℕ := 5
  let gabrielle_cardinals : ℕ := 4
  let chase_robins : ℕ := 2
  let chase_blue_jays : ℕ := 3
  let chase_cardinals : ℕ := 5
  let chase_total : ℕ := chase_robins + chase_blue_jays + chase_cardinals
  let gabrielle_total : ℕ := chase_total + chase_total / 5
  gabrielle_total - gabrielle_robins - gabrielle_cardinals

theorem gabrielle_blue_jays_eq_three : gabrielle_blue_jays = 3 := by
  sorry

end NUMINAMATH_CALUDE_gabrielle_blue_jays_eq_three_l879_87918


namespace NUMINAMATH_CALUDE_perfect_squares_equivalence_l879_87994

theorem perfect_squares_equivalence (n : ℕ+) :
  (∃ k : ℕ, 2 * n + 1 = k^2) ∧ (∃ t : ℕ, 3 * n + 1 = t^2) ↔
  (∃ k : ℕ, n + 1 = k^2 + (k + 1)^2) ∧ 
  (∃ t : ℕ, n + 1 = (t - 1)^2 + 2 * t^2 ∨ n + 1 = (t + 1)^2 + 2 * t^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_equivalence_l879_87994


namespace NUMINAMATH_CALUDE_largest_possible_z_value_l879_87939

open Complex

theorem largest_possible_z_value (a b c d z w : ℂ) 
  (h1 : abs a = abs b)
  (h2 : abs b = abs c)
  (h3 : abs c = abs d)
  (h4 : abs a > 0)
  (h5 : a * z^3 + b * w * z^2 + c * z + d = 0)
  (h6 : abs w = 1/2) :
  abs z ≤ 1 ∧ ∃ a b c d z w : ℂ, 
    abs a = abs b ∧ 
    abs b = abs c ∧ 
    abs c = abs d ∧ 
    abs a > 0 ∧
    a * z^3 + b * w * z^2 + c * z + d = 0 ∧
    abs w = 1/2 ∧
    abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_possible_z_value_l879_87939


namespace NUMINAMATH_CALUDE_inserted_eights_composite_l879_87909

def insert_eights (n : ℕ) : ℕ :=
  2000 * 10^n + 8 * ((10^n - 1) / 9) + 21

theorem inserted_eights_composite (n : ℕ) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ insert_eights n = a * b :=
sorry

end NUMINAMATH_CALUDE_inserted_eights_composite_l879_87909


namespace NUMINAMATH_CALUDE_hyperbola_focus_equation_l879_87979

/-- Given a hyperbola of the form x²/m - y² = 1 with one focus at (-2√2, 0),
    prove that m = 7 -/
theorem hyperbola_focus_equation (m : ℝ) : 
  (∃ (x y : ℝ), x^2 / m - y^2 = 1) →  -- hyperbola equation
  ((-2 * Real.sqrt 2, 0) : ℝ × ℝ) ∈ {(x, y) | x^2 / m - y^2 = 1} →  -- focus condition
  m = 7 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_equation_l879_87979


namespace NUMINAMATH_CALUDE_paul_pencil_days_l879_87946

/-- Calculates the number of days Paul makes pencils in a week -/
def pencil_making_days (
  pencils_per_day : ℕ) 
  (initial_stock : ℕ) 
  (pencils_sold : ℕ) 
  (final_stock : ℕ) : ℕ :=
  (final_stock + pencils_sold - initial_stock) / pencils_per_day

theorem paul_pencil_days : 
  pencil_making_days 100 80 350 230 = 5 := by sorry

end NUMINAMATH_CALUDE_paul_pencil_days_l879_87946


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l879_87944

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) :
  perimeter = 40 →
  area = (perimeter / 4) ^ 2 →
  area = 100 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l879_87944


namespace NUMINAMATH_CALUDE_kyle_shooting_time_ratio_l879_87916

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Kyle's basketball practice schedule -/
structure BasketballPractice where
  totalTime : ℕ  -- in minutes
  weightliftingTime : ℕ  -- in minutes
  runningTime : ℕ  -- in minutes
  shootingTime : ℕ  -- in minutes

def KylesPractice : BasketballPractice :=
  { totalTime := 120,  -- 2 hours = 120 minutes
    weightliftingTime := 20,
    runningTime := 40,  -- twice the weightlifting time
    shootingTime := 60  -- calculated as totalTime - (weightliftingTime + runningTime)
  }

/-- Calculates the ratio of shooting time to total practice time -/
def shootingTimeRatio (practice : BasketballPractice) : Ratio :=
  { numerator := practice.shootingTime,
    denominator := practice.totalTime
  }

theorem kyle_shooting_time_ratio :
  shootingTimeRatio KylesPractice = Ratio.mk 1 2 := by sorry

end NUMINAMATH_CALUDE_kyle_shooting_time_ratio_l879_87916


namespace NUMINAMATH_CALUDE_square_land_side_length_l879_87996

theorem square_land_side_length (area : ℝ) (h : area = Real.sqrt 625) :
  ∃ (side : ℝ), side * side = area ∧ side = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l879_87996


namespace NUMINAMATH_CALUDE_horner_operations_count_l879_87973

/-- Represents a univariate polynomial --/
structure UnivariatePoly (α : Type*) where
  coeffs : List α

/-- Horner's method for polynomial evaluation --/
def hornerMethod (p : UnivariatePoly ℤ) : ℕ × ℕ :=
  (p.coeffs.length - 1, p.coeffs.length - 1)

/-- The given polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 --/
def f : UnivariatePoly ℤ :=
  ⟨[1, 1, 2, 3, 4, 5]⟩

theorem horner_operations_count :
  hornerMethod f = (5, 5) := by sorry

end NUMINAMATH_CALUDE_horner_operations_count_l879_87973


namespace NUMINAMATH_CALUDE_frog_escapes_in_18_days_l879_87953

/-- Represents the depth of the well in meters -/
def well_depth : ℕ := 20

/-- Represents the distance the frog climbs up each day in meters -/
def climb_distance : ℕ := 3

/-- Represents the distance the frog slips down each day in meters -/
def slip_distance : ℕ := 2

/-- Represents the net distance the frog climbs each day in meters -/
def net_daily_progress : ℕ := climb_distance - slip_distance

/-- Theorem stating that the frog can climb out of the well in 18 days -/
theorem frog_escapes_in_18_days :
  ∃ (n : ℕ), n = 18 ∧ n * net_daily_progress + climb_distance ≥ well_depth :=
sorry

end NUMINAMATH_CALUDE_frog_escapes_in_18_days_l879_87953


namespace NUMINAMATH_CALUDE_robs_reading_l879_87941

/-- Given Rob's planned reading time, actual reading time as a fraction of planned time,
    and his reading speed, calculate the number of pages he read. -/
theorem robs_reading (planned_hours : ℝ) (actual_fraction : ℝ) (pages_per_minute : ℝ) : 
  planned_hours = 3 →
  actual_fraction = 3/4 →
  pages_per_minute = 1/15 →
  (planned_hours * actual_fraction * 60) * pages_per_minute = 9 := by
  sorry

end NUMINAMATH_CALUDE_robs_reading_l879_87941


namespace NUMINAMATH_CALUDE_infiniteSeries_eq_three_halves_l879_87956

/-- The sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ -/
noncomputable def infiniteSeries : ℝ := ∑' k, k / (3 ^ k)

/-- Theorem stating that the sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ is equal to 3/2 -/
theorem infiniteSeries_eq_three_halves : infiniteSeries = 3/2 := by sorry

end NUMINAMATH_CALUDE_infiniteSeries_eq_three_halves_l879_87956


namespace NUMINAMATH_CALUDE_rational_function_sum_l879_87951

/-- Given rational functions p(x) and q(x) satisfying certain conditions,
    prove that their sum has a specific form. -/
theorem rational_function_sum (p q : ℝ → ℝ) : 
  (∀ x, ∃ y, q x = y * (x + 1) * (x - 2) * (x - 3)) →  -- q(x) is cubic with specific factors
  (∀ x, ∃ y, p x = y * (x + 1) * (x - 2)) →  -- p(x) is quadratic with specific factors
  p 2 = 2 →  -- p(2) = 2
  q (-1) = -1 →  -- q(-1) = -1
  ∀ x, p x + q x = x^3 - 3*x^2 + 4*x + 4 := by
sorry

end NUMINAMATH_CALUDE_rational_function_sum_l879_87951


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l879_87971

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/4 + 1/5, 1/4 + 1/6, 1/4 + 1/3, 1/4 + 1/8, 1/4 + 1/7]
  (∀ x ∈ sums, x ≤ (1/4 + 1/3)) ∧ (1/4 + 1/3 = 7/12) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l879_87971


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l879_87942

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 3 * y - 1 = 0) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2^x + 8^y ≥ z → z ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l879_87942


namespace NUMINAMATH_CALUDE_cow_count_l879_87975

theorem cow_count (D C : ℕ) : 
  2 * D + 4 * C = 2 * (D + C) + 30 → C = 15 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l879_87975


namespace NUMINAMATH_CALUDE_f_properties_l879_87948

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 + x) * Real.cos (Real.pi / 2 - x)

theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ = -x₂ → f x₁ = -f x₂) ∧
  (∃ T : ℝ, T > 0 ∧ T < 2 * Real.pi ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x y : ℝ, -Real.pi/4 ≤ x ∧ x < y ∧ y ≤ Real.pi/4 → f x < f y) ∧
  (∀ x : ℝ, f (3 * Real.pi / 2 - x) = f (3 * Real.pi / 2 + x)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l879_87948


namespace NUMINAMATH_CALUDE_bird_difference_l879_87978

/-- Proves the difference between white birds and original grey birds -/
theorem bird_difference (initial_grey : ℕ) (total_remaining : ℕ) 
  (h1 : initial_grey = 40)
  (h2 : total_remaining = 66) :
  total_remaining - initial_grey / 2 - initial_grey = 6 := by
  sorry

end NUMINAMATH_CALUDE_bird_difference_l879_87978


namespace NUMINAMATH_CALUDE_correct_product_with_decimals_l879_87931

theorem correct_product_with_decimals (x y : ℚ) (z : ℕ) : 
  x = 0.035 → y = 3.84 → z = 13440 → x * y = 0.1344 := by
  sorry

end NUMINAMATH_CALUDE_correct_product_with_decimals_l879_87931


namespace NUMINAMATH_CALUDE_local_minimum_implies_c_equals_two_l879_87950

/-- The function f(x) defined as x(x - c)² --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) --/
def f_deriv (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

theorem local_minimum_implies_c_equals_two :
  ∀ c : ℝ, (∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f c x ≥ f c 2) →
  f_deriv c 2 = 0 →
  c = 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_implies_c_equals_two_l879_87950


namespace NUMINAMATH_CALUDE_roots_of_cubic_equations_l879_87927

theorem roots_of_cubic_equations (p q r s : ℂ) (m : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^3 + q * m^2 + r * m + s = 0)
  (h2 : q * m^3 + r * m^2 + s * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I := by
sorry

end NUMINAMATH_CALUDE_roots_of_cubic_equations_l879_87927


namespace NUMINAMATH_CALUDE_battle_station_staffing_l879_87911

def n : ℕ := 20
def k : ℕ := 5

theorem battle_station_staffing :
  (n.factorial) / ((n - k).factorial) = 930240 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l879_87911


namespace NUMINAMATH_CALUDE_youngest_age_in_office_l879_87947

/-- Proves that in a group of 4 people whose ages form an arithmetic sequence,
    if the oldest person is 50 years old and the sum of their ages is 158 years,
    then the youngest person is 29 years old. -/
theorem youngest_age_in_office (ages : Fin 4 → ℕ) 
  (arithmetic_sequence : ∀ i j k : Fin 4, i < j → j < k → 
    ages j - ages i = ages k - ages j)
  (oldest_age : ages 3 = 50)
  (sum_of_ages : (Finset.univ.sum ages) = 158) :
  ages 0 = 29 := by
sorry

end NUMINAMATH_CALUDE_youngest_age_in_office_l879_87947


namespace NUMINAMATH_CALUDE_card_distribution_theorem_l879_87921

/-- Represents the state of card distribution among points -/
structure CardState (n : ℕ) where
  cards_at_A : Fin n → ℕ
  cards_at_O : ℕ

/-- Represents a move in the game -/
inductive Move (n : ℕ)
  | outer (i : Fin n) : Move n
  | inner : Move n

/-- Applies a move to a card state -/
def apply_move (n : ℕ) (state : CardState n) (move : Move n) : CardState n :=
  sorry

/-- Checks if a state is valid according to the game rules -/
def is_valid_state (n : ℕ) (state : CardState n) : Prop :=
  sorry

/-- Checks if a state is the goal state (all points have ≥ n+1 cards) -/
def is_goal_state (n : ℕ) (state : CardState n) : Prop :=
  sorry

/-- The main theorem to be proved -/
theorem card_distribution_theorem (n : ℕ) (h_n : n ≥ 3) (T : ℕ) (h_T : T ≥ n^2 + 3*n + 1)
  (initial_state : CardState n) (h_initial : is_valid_state n initial_state) :
  ∃ (moves : List (Move n)), 
    is_goal_state n (moves.foldl (apply_move n) initial_state) :=
  sorry

end NUMINAMATH_CALUDE_card_distribution_theorem_l879_87921


namespace NUMINAMATH_CALUDE_max_intersecting_chords_2017_l879_87949

/-- Given a circle with n distinct points, this function calculates the maximum number of chords
    intersecting a line through one point, not passing through any other points. -/
def max_intersecting_chords (n : ℕ) : ℕ :=
  let k := (n - 1) / 2
  k * (n - 1 - k) + (n - 1)

/-- The theorem states that for a circle with 2017 points, the maximum number of
    intersecting chords is 1018080. -/
theorem max_intersecting_chords_2017 :
  max_intersecting_chords 2017 = 1018080 := by sorry

end NUMINAMATH_CALUDE_max_intersecting_chords_2017_l879_87949


namespace NUMINAMATH_CALUDE_equation_solution_l879_87925

theorem equation_solution : ∃ x : ℝ, 4 * (4^x) + Real.sqrt (16 * (16^x)) = 32 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l879_87925


namespace NUMINAMATH_CALUDE_quadratic_decrease_interval_l879_87902

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_decrease_interval (b c : ℝ) :
  f b c 1 = 0 → f b c 3 = 0 → 
  ∀ x y : ℝ, x < y → y < 2 → f b c x > f b c y := by sorry

end NUMINAMATH_CALUDE_quadratic_decrease_interval_l879_87902


namespace NUMINAMATH_CALUDE_tom_apple_count_l879_87952

/-- The number of apples each person has -/
structure AppleCount where
  phillip : ℕ
  ben : ℕ
  tom : ℕ

/-- The conditions of the problem -/
def problem_conditions (ac : AppleCount) : Prop :=
  ac.phillip = 40 ∧
  ac.ben = ac.phillip + 8 ∧
  ac.tom = (3 * ac.ben) / 8

/-- The theorem stating that Tom has 18 apples given the problem conditions -/
theorem tom_apple_count (ac : AppleCount) (h : problem_conditions ac) : ac.tom = 18 := by
  sorry

#check tom_apple_count

end NUMINAMATH_CALUDE_tom_apple_count_l879_87952


namespace NUMINAMATH_CALUDE_least_four_digit_11_heavy_l879_87967

def is_11_heavy (n : ℕ) : Prop := n % 11 > 7

theorem least_four_digit_11_heavy : 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 1000 → ¬(is_11_heavy m)) ∧ is_11_heavy 1000 :=
sorry

end NUMINAMATH_CALUDE_least_four_digit_11_heavy_l879_87967


namespace NUMINAMATH_CALUDE_complex_magnitude_l879_87998

theorem complex_magnitude (z : ℂ) (h : z + Complex.abs z = 2 + 8 * Complex.I) : Complex.abs z = 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l879_87998


namespace NUMINAMATH_CALUDE_total_corn_harvest_l879_87959

-- Define the cornfield properties
def johnson_field : ℝ := 1
def johnson_yield : ℝ := 80
def johnson_period : ℝ := 2

def smith_field : ℝ := 2
def smith_yield_factor : ℝ := 2

def brown_field : ℝ := 1.5
def brown_yield : ℝ := 50
def brown_period : ℝ := 3

def taylor_field : ℝ := 0.5
def taylor_yield : ℝ := 30
def taylor_period : ℝ := 1

def total_months : ℝ := 6

-- Define the theorem
theorem total_corn_harvest :
  let johnson_total := (total_months / johnson_period) * johnson_yield
  let smith_total := (total_months / johnson_period) * (smith_field * smith_yield_factor * johnson_yield)
  let brown_total := (total_months / brown_period) * (brown_field * brown_yield)
  let taylor_total := (total_months / taylor_period) * taylor_yield
  johnson_total + smith_total + brown_total + taylor_total = 1530 := by
  sorry

end NUMINAMATH_CALUDE_total_corn_harvest_l879_87959


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l879_87962

/-- 
An arithmetic sequence is defined by its first term, common difference, and last term.
This theorem proves that for an arithmetic sequence with first term 15, 
common difference 4, and last term 159, the number of terms is 37.
-/
theorem arithmetic_sequence_terms (first_term : ℕ) (common_diff : ℕ) (last_term : ℕ) :
  first_term = 15 → common_diff = 4 → last_term = 159 →
  (last_term - first_term) / common_diff + 1 = 37 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l879_87962


namespace NUMINAMATH_CALUDE_weight_loss_program_l879_87981

def initial_weight : ℕ := 250
def weeks_phase1 : ℕ := 4
def loss_per_week_phase1 : ℕ := 3
def weeks_phase2 : ℕ := 8
def loss_per_week_phase2 : ℕ := 2

theorem weight_loss_program (w : ℕ) :
  w = initial_weight - (weeks_phase1 * loss_per_week_phase1 + weeks_phase2 * loss_per_week_phase2) →
  w = 222 :=
by sorry

end NUMINAMATH_CALUDE_weight_loss_program_l879_87981


namespace NUMINAMATH_CALUDE_range_of_m_l879_87995

-- Define the sets A and B
def A (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}
def B := {x : ℝ | x^2 - 2*x - 15 ≤ 0}

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∃ x, x ∈ A m) ∧ -- A is non-empty
  (∃ x, x ∈ B) ∧ -- B is non-empty
  (∀ x, x ∈ A m → x ∈ B) → -- A ⊆ B
  2 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l879_87995


namespace NUMINAMATH_CALUDE_angle_difference_range_l879_87990

theorem angle_difference_range (α β : Real) (h1 : -π/2 < α) (h2 : α < β) (h3 : β < π) :
  ∃ (x : Real), -3*π/2 < x ∧ x < 0 ∧ ∀ (y : Real), (-3*π/2 < y ∧ y < 0) → ∃ (α' β' : Real),
    -π/2 < α' ∧ α' < β' ∧ β' < π ∧ y = α' - β' :=
by sorry

end NUMINAMATH_CALUDE_angle_difference_range_l879_87990


namespace NUMINAMATH_CALUDE_bank_robbery_culprits_l879_87935

theorem bank_robbery_culprits (Alexey Boris Veniamin Grigory : Prop) :
  (¬Grigory → Boris ∧ ¬Alexey) →
  (Veniamin → ¬Alexey ∧ ¬Boris) →
  (Grigory → Boris) →
  (Boris → Alexey ∨ Veniamin) →
  (Alexey ∧ Boris ∧ Grigory ∧ ¬Veniamin) :=
by sorry

end NUMINAMATH_CALUDE_bank_robbery_culprits_l879_87935


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_zero_l879_87977

theorem tan_alpha_2_implies_expression_zero (α : Real) (h : Real.tan α = 2) :
  2 * (Real.sin α)^2 - 3 * Real.sin α * Real.cos α - 2 * (Real.cos α)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_zero_l879_87977


namespace NUMINAMATH_CALUDE_journey_distance_l879_87999

theorem journey_distance : ∀ (D : ℝ),
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 12 = D →
  D = 90 := by sorry

end NUMINAMATH_CALUDE_journey_distance_l879_87999


namespace NUMINAMATH_CALUDE_chord_slope_l879_87914

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x^2 = -2*y

/-- Point P on the parabola -/
def P : ℝ × ℝ := (2, -2)

/-- Complementary angles of inclination -/
def complementary_angles (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The theorem statement -/
theorem chord_slope : 
  ∃ (A B : ℝ × ℝ), 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧
    parabola P.1 P.2 ∧
    (∃ (m_PA m_PB : ℝ), complementary_angles m_PA m_PB) ∧
    (B.2 - A.2) / (B.1 - A.1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_slope_l879_87914


namespace NUMINAMATH_CALUDE_half_vector_AB_l879_87974

/-- Given two vectors OA and OB in ℝ², prove that half of vector AB equals (1/2, 5/2) -/
theorem half_vector_AB (OA OB : ℝ × ℝ) (h1 : OA = (3, 2)) (h2 : OB = (4, 7)) :
  (1 / 2 : ℝ) • (OB - OA) = (1/2, 5/2) := by sorry

end NUMINAMATH_CALUDE_half_vector_AB_l879_87974


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l879_87908

/-- The function that constructs the number 534n given n -/
def number (n : ℕ) : ℕ := 5340 + n

/-- Predicate to check if a number is four-digit -/
def is_four_digit (x : ℕ) : Prop := 1000 ≤ x ∧ x < 10000

theorem divisible_by_twelve (n : ℕ) : 
  (is_four_digit (number n)) → 
  (n < 10) → 
  ((number n) % 12 = 0 ↔ n = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l879_87908


namespace NUMINAMATH_CALUDE_rectangle_area_l879_87905

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 166) : L * B = 1590 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l879_87905


namespace NUMINAMATH_CALUDE_particular_number_problem_l879_87985

theorem particular_number_problem (x : ℤ) : 
  (x - 29 + 64 = 76) → x = 41 := by
sorry

end NUMINAMATH_CALUDE_particular_number_problem_l879_87985


namespace NUMINAMATH_CALUDE_nonconvex_quadrilateral_theorem_l879_87968

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Checks if a quadrilateral is nonconvex -/
def is_nonconvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the angle at a vertex of a quadrilateral -/
def angle_at_vertex (q : Quadrilateral) (v : Point2D) : ℝ := sorry

/-- Finds the intersection point of two lines -/
def line_intersection (p1 p2 q1 q2 : Point2D) : Point2D := sorry

/-- Checks if a point lies on a line segment -/
def point_on_segment (p : Point2D) (a b : Point2D) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point2D) : ℝ := sorry

theorem nonconvex_quadrilateral_theorem (ABCD : Quadrilateral) 
  (hnonconvex : is_nonconvex ABCD)
  (hC_angle : angle_at_vertex ABCD ABCD.C > 180)
  (F : Point2D) (hF : F = line_intersection ABCD.D ABCD.C ABCD.A ABCD.B)
  (E : Point2D) (hE : E = line_intersection ABCD.B ABCD.C ABCD.A ABCD.D)
  (K L J I : Point2D)
  (hK : point_on_segment K ABCD.A ABCD.B)
  (hL : point_on_segment L ABCD.A ABCD.D)
  (hJ : point_on_segment J ABCD.B ABCD.C)
  (hI : point_on_segment I ABCD.C ABCD.D)
  (hDI_CF : distance ABCD.D I = distance ABCD.C F)
  (hBJ_CE : distance ABCD.B J = distance ABCD.C E) :
  distance K J = distance I L := by sorry

end NUMINAMATH_CALUDE_nonconvex_quadrilateral_theorem_l879_87968


namespace NUMINAMATH_CALUDE_floor_painting_possibilities_l879_87919

theorem floor_painting_possibilities :
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      (p.2 > p.1 ∧ 
       (p.1 - 4) * (p.2 - 4) = 2 * p.1 * p.2 / 3 ∧
       p.1 > 0 ∧ p.2 > 0)) ∧
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_possibilities_l879_87919


namespace NUMINAMATH_CALUDE_bears_per_shelf_l879_87986

theorem bears_per_shelf (initial_stock : ℕ) (new_shipment : ℕ) (num_shelves : ℕ) 
  (h1 : initial_stock = 5)
  (h2 : new_shipment = 7)
  (h3 : num_shelves = 2)
  : (initial_stock + new_shipment) / num_shelves = 6 := by
  sorry

end NUMINAMATH_CALUDE_bears_per_shelf_l879_87986


namespace NUMINAMATH_CALUDE_remainder_problem_l879_87980

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 7 = 3) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l879_87980


namespace NUMINAMATH_CALUDE_prime_divides_binomial_l879_87929

theorem prime_divides_binomial (n k : ℕ) (h_prime : Nat.Prime n) (h_k_pos : 0 < k) (h_k_lt_n : k < n) :
  n ∣ Nat.choose n k := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_binomial_l879_87929


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l879_87922

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  (∃ m : ℕ, m > 12 ∧ (∀ k : ℕ, k > 0 → m ∣ (k * (k + 1) * (k + 2) * (k + 3)))) →
  False :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l879_87922


namespace NUMINAMATH_CALUDE_kanul_cash_theorem_l879_87928

/-- Represents the total amount of cash Kanul had -/
def T : ℝ := sorry

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 5000

/-- The amount spent on machinery -/
def machinery : ℝ := 200

/-- The percentage of total cash spent -/
def percentage_spent : ℝ := 0.30

theorem kanul_cash_theorem : 
  T = (raw_materials + machinery) / (1 - percentage_spent) :=
by sorry

end NUMINAMATH_CALUDE_kanul_cash_theorem_l879_87928
