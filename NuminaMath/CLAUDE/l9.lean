import Mathlib

namespace NUMINAMATH_CALUDE_workday_meetings_percentage_l9_901

def workday_hours : ℕ := 10
def first_meeting_minutes : ℕ := 60

theorem workday_meetings_percentage :
  let workday_minutes : ℕ := workday_hours * 60
  let second_meeting_minutes : ℕ := 3 * first_meeting_minutes
  let total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes
  (total_meeting_minutes : ℚ) / (workday_minutes : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_workday_meetings_percentage_l9_901


namespace NUMINAMATH_CALUDE_usb_storage_capacity_l9_958

/-- Represents the capacity of a storage device in gigabytes -/
def StorageCapacityGB : ℕ := 2

/-- Represents the size of one gigabyte in megabytes -/
def GBtoMB : ℕ := 2^10

/-- Represents the file size of each photo in megabytes -/
def PhotoSizeMB : ℕ := 16

/-- Calculates the number of photos that can be stored -/
def NumberOfPhotos : ℕ := 2^7

theorem usb_storage_capacity :
  StorageCapacityGB * GBtoMB / PhotoSizeMB = NumberOfPhotos :=
sorry

end NUMINAMATH_CALUDE_usb_storage_capacity_l9_958


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l9_994

theorem square_difference_divided_by_nine : (121^2 - 112^2) / 9 = 233 := by sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l9_994


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l9_911

noncomputable def p : ℝ := 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 2 * Real.sqrt 6
noncomputable def q : ℝ := -2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 2 * Real.sqrt 6
noncomputable def r : ℝ := 2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 2 * Real.sqrt 6
noncomputable def s : ℝ := -2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 2 * Real.sqrt 6

theorem sum_of_reciprocals_squared :
  (1/p + 1/q + 1/r + 1/s)^2 = 3/16 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l9_911


namespace NUMINAMATH_CALUDE_segments_parallel_iff_m_equals_two_thirds_l9_933

/-- Given points A, B, C, D on a Cartesian plane, prove that segments AB and CD are parallel
    if and only if m = 2/3 -/
theorem segments_parallel_iff_m_equals_two_thirds 
  (A B C D : ℝ × ℝ) 
  (hA : A = (1, -1)) 
  (hB : B = (4, -2)) 
  (hC : C = (-1, 2)) 
  (hD : D = (3, m)) 
  (m : ℝ) : 
  (∃ k : ℝ, B.1 - A.1 = k * (D.1 - C.1) ∧ B.2 - A.2 = k * (D.2 - C.2)) ↔ m = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_segments_parallel_iff_m_equals_two_thirds_l9_933


namespace NUMINAMATH_CALUDE_salary_increase_with_manager_l9_955

/-- Calculates the increase in average salary when a manager's salary is added to a group of employees. -/
theorem salary_increase_with_manager 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 20 →
  avg_salary = 1500 →
  manager_salary = 22500 →
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 1000 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_with_manager_l9_955


namespace NUMINAMATH_CALUDE_equation_solution_l9_957

theorem equation_solution : 
  ∃ x : ℚ, (2*x - 30) / 3 = (5 - 3*x) / 4 + 1 ∧ x = 147 / 17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l9_957


namespace NUMINAMATH_CALUDE_inequality_proof_l9_941

theorem inequality_proof (x y : ℝ) (n k : ℕ) 
  (h1 : x > y) (h2 : y > 0) (h3 : n > k) :
  (x^k - y^k)^n < (x^n - y^n)^k := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l9_941


namespace NUMINAMATH_CALUDE_sqrt_17_irrational_l9_954

theorem sqrt_17_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ) / q = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_17_irrational_l9_954


namespace NUMINAMATH_CALUDE_min_value_theorem_l9_945

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) : 
  (∃ (x y : ℝ) (hx : x > 0) (hy : y > 0), 2/x + 1/y < 2/a + 1/b) → 
  2/a + 1/b ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l9_945


namespace NUMINAMATH_CALUDE_reflection_matrix_squared_is_identity_l9_977

/-- Reflection matrix over a non-zero vector -/
def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  sorry

/-- Theorem: The square of a reflection matrix is the identity matrix -/
theorem reflection_matrix_squared_is_identity (v : ℝ × ℝ) (h : v ≠ (0, 0)) :
  (reflection_matrix v) ^ 2 = !![1, 0; 0, 1] :=
sorry

end NUMINAMATH_CALUDE_reflection_matrix_squared_is_identity_l9_977


namespace NUMINAMATH_CALUDE_population_ratio_x_to_z_l9_922

/-- Represents the population of a city. -/
structure CityPopulation where
  value : ℕ

/-- Represents the ratio between two city populations. -/
structure PopulationRatio where
  numerator : ℕ
  denominator : ℕ

/-- Given three cities X, Y, and Z, where X's population is 8 times Y's,
    and Y's population is twice Z's, prove that the ratio of X's population
    to Z's population is 16:1. -/
theorem population_ratio_x_to_z
  (pop_x pop_y pop_z : CityPopulation)
  (h1 : pop_x.value = 8 * pop_y.value)
  (h2 : pop_y.value = 2 * pop_z.value) :
  PopulationRatio.mk 16 1 = PopulationRatio.mk (pop_x.value / pop_z.value) 1 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_x_to_z_l9_922


namespace NUMINAMATH_CALUDE_room_length_proof_l9_913

/-- The length of a rectangular room given its width, number of tiles, and tile size. -/
theorem room_length_proof (width : ℝ) (num_tiles : ℕ) (tile_size : ℝ) 
  (h1 : width = 12)
  (h2 : num_tiles = 6)
  (h3 : tile_size = 4)
  : width * (num_tiles * tile_size / width) = 2 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l9_913


namespace NUMINAMATH_CALUDE_count_specific_divisors_l9_953

theorem count_specific_divisors (p q : ℕ+) : 
  let n := 2^(p : ℕ) * 3^(q : ℕ)
  (∃ (s : Finset ℕ), s.card = p * q ∧ 
    (∀ d ∈ s, d ∣ n^2 ∧ d < n ∧ ¬(d ∣ n))) :=
by sorry

end NUMINAMATH_CALUDE_count_specific_divisors_l9_953


namespace NUMINAMATH_CALUDE_intersection_count_504_220_l9_985

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A line segment from (0,0) to (a,b) -/
structure LineSegment where
  a : ℤ
  b : ℤ

/-- Count of intersections with squares and circles -/
structure IntersectionCount where
  squares : ℕ
  circles : ℕ

/-- Function to count intersections of a line segment with squares and circles -/
def countIntersections (l : LineSegment) : IntersectionCount :=
  sorry

theorem intersection_count_504_220 :
  let l : LineSegment := ⟨504, 220⟩
  let count : IntersectionCount := countIntersections l
  count.squares + count.circles = 255 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_504_220_l9_985


namespace NUMINAMATH_CALUDE_A_intersect_Z_l9_980

def A : Set ℝ := {x : ℝ | |x - 1| < 2}

theorem A_intersect_Z : A ∩ Set.range (Int.cast : ℤ → ℝ) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_Z_l9_980


namespace NUMINAMATH_CALUDE_compare_negative_decimals_l9_973

theorem compare_negative_decimals : -0.5 > -0.75 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_decimals_l9_973


namespace NUMINAMATH_CALUDE_triangle_inequality_l9_909

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l9_909


namespace NUMINAMATH_CALUDE_musical_chairs_theorem_l9_952

/-- A function is a derangement if it has no fixed points -/
def IsDerangement {α : Type*} (f : α → α) : Prop :=
  ∀ x, f x ≠ x

/-- A positive integer is a prime power if it's of the form p^k where p is prime and k > 0 -/
def IsPrimePower (n : ℕ) : Prop :=
  ∃ (p k : ℕ), Prime p ∧ k > 0 ∧ n = p^k

theorem musical_chairs_theorem (n m : ℕ) 
    (h1 : m > 1) 
    (h2 : m ≤ n) 
    (h3 : ¬ IsPrimePower m) : 
    ∃ (f : Fin n → Fin n), 
      Function.Bijective f ∧ 
      IsDerangement f ∧ 
      (∀ x, (f^[m]) x = x) ∧ 
      (∀ (k : ℕ) (hk : k < m), ∃ x, (f^[k]) x ≠ x) := by
  sorry

end NUMINAMATH_CALUDE_musical_chairs_theorem_l9_952


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l9_998

theorem perpendicular_vectors_k_value (k : ℝ) : 
  let a : Fin 2 → ℝ := ![6, 2]
  let b : Fin 2 → ℝ := ![-3, k]
  (∀ i, i < 2 → a i * b i = 0) → k = 9 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l9_998


namespace NUMINAMATH_CALUDE_dragon_defeat_probability_is_one_l9_975

/-- Represents the state of the dragon's heads -/
structure DragonState where
  heads : ℕ

/-- Represents the possible outcomes after chopping off a head -/
inductive ChopOutcome
  | TwoHeadsGrow
  | OneHeadGrows
  | NoHeadGrows

/-- The probability distribution of chop outcomes -/
def chopProbability : ChopOutcome → ℚ
  | ChopOutcome.TwoHeadsGrow => 1/4
  | ChopOutcome.OneHeadGrows => 1/3
  | ChopOutcome.NoHeadGrows => 5/12

/-- The transition function for the dragon state after a chop -/
def transition (state : DragonState) (outcome : ChopOutcome) : DragonState :=
  match outcome with
  | ChopOutcome.TwoHeadsGrow => ⟨state.heads + 1⟩
  | ChopOutcome.OneHeadGrows => state
  | ChopOutcome.NoHeadGrows => ⟨state.heads - 1⟩

/-- The probability of eventually defeating the dragon -/
noncomputable def defeatProbability (initialState : DragonState) : ℝ :=
  sorry

/-- Theorem stating that the probability of defeating the dragon is 1 -/
theorem dragon_defeat_probability_is_one :
  defeatProbability ⟨3⟩ = 1 := by sorry

end NUMINAMATH_CALUDE_dragon_defeat_probability_is_one_l9_975


namespace NUMINAMATH_CALUDE_probability_not_perfect_power_l9_992

/-- A number is a perfect power if it can be expressed as x^y where x and y are integers and y > 1 -/
def IsPerfectPower (n : ℕ) : Prop :=
  ∃ x y : ℕ, y > 1 ∧ n = x^y

/-- The count of numbers from 1 to 200 that are perfect powers -/
def PerfectPowerCount : ℕ := 21

/-- The total count of numbers from 1 to 200 -/
def TotalCount : ℕ := 200

/-- The probability of selecting a number that is not a perfect power -/
def ProbabilityNotPerfectPower : ℚ :=
  (TotalCount - PerfectPowerCount : ℚ) / TotalCount

theorem probability_not_perfect_power :
  ProbabilityNotPerfectPower = 179 / 200 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_perfect_power_l9_992


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l9_979

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 / a^2 → a^2 > 1 / a) ∧
  (∃ a, a^2 > 1 / a ∧ a ≤ 1 / a^2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l9_979


namespace NUMINAMATH_CALUDE_problem_solution_l9_944

theorem problem_solution (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4)
  (h2 : c ^ 3 = d ^ 2)
  (h3 : c - a = 19) :
  d - b = 757 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l9_944


namespace NUMINAMATH_CALUDE_remaining_capacity_theorem_l9_993

/-- Represents the meal capacity and consumption for a trekking group --/
structure MealCapacity where
  adult_capacity : ℕ
  child_capacity : ℕ
  adults_eaten : ℕ

/-- Calculates the number of children that can be catered with the remaining food --/
def remaining_child_capacity (m : MealCapacity) : ℕ :=
  sorry

/-- Theorem stating that given the specific meal capacity and consumption, 
    the remaining food can cater to 45 children --/
theorem remaining_capacity_theorem (m : MealCapacity) 
  (h1 : m.adult_capacity = 70)
  (h2 : m.child_capacity = 90)
  (h3 : m.adults_eaten = 35) :
  remaining_child_capacity m = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_capacity_theorem_l9_993


namespace NUMINAMATH_CALUDE_plot_length_is_63_meters_l9_976

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length_difference : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Calculates the length of the plot given its properties. -/
def calculate_length (plot : RectangularPlot) : ℝ :=
  plot.breadth + plot.length_difference

/-- Calculates the perimeter of the plot. -/
def calculate_perimeter (plot : RectangularPlot) : ℝ :=
  2 * (calculate_length plot + plot.breadth)

/-- Theorem stating that under given conditions, the length of the plot is 63 meters. -/
theorem plot_length_is_63_meters (plot : RectangularPlot) 
  (h1 : plot.length_difference = 26)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : calculate_perimeter plot = plot.total_fencing_cost / plot.fencing_cost_per_meter) :
  calculate_length plot = 63 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_63_meters_l9_976


namespace NUMINAMATH_CALUDE_rectangle_max_area_l9_926

/-- The maximum area of a rectangle with perimeter 40 meters is 100 square meters. -/
theorem rectangle_max_area (x : ℝ) :
  let perimeter := 40
  let width := x
  let length := (perimeter / 2) - x
  let area := width * length
  (∀ y, 0 < y ∧ y < perimeter / 2 → area ≥ y * (perimeter / 2 - y)) →
  area ≤ 100 ∧ ∃ z, 0 < z ∧ z < perimeter / 2 ∧ z * (perimeter / 2 - z) = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l9_926


namespace NUMINAMATH_CALUDE_intersection_M_N_l9_946

def N : Set ℝ := {x | x^2 ≤ 1}

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l9_946


namespace NUMINAMATH_CALUDE_rachels_homework_l9_906

theorem rachels_homework (math_pages reading_pages : ℕ) : 
  math_pages = 7 → 
  math_pages = reading_pages + 4 → 
  reading_pages = 3 := by
sorry

end NUMINAMATH_CALUDE_rachels_homework_l9_906


namespace NUMINAMATH_CALUDE_counterfeit_coin_strategy_exists_l9_991

/-- Represents a weighing operation that can compare two groups of coins. -/
def Weighing := List Nat → List Nat → Ordering

/-- Represents a strategy for finding the counterfeit coin. -/
def Strategy := List Nat → List Weighing → Option Nat

/-- The number of coins. -/
def n : Nat := 81

/-- The maximum number of weighings allowed. -/
def max_weighings : Nat := 4

/-- Theorem stating that there exists a strategy to find the counterfeit coin. -/
theorem counterfeit_coin_strategy_exists :
  ∃ (s : Strategy),
    ∀ (counterfeit : Nat),
      counterfeit < n →
      ∃ (weighings : List Weighing),
        weighings.length ≤ max_weighings ∧
        s (List.range n) weighings = some counterfeit :=
by sorry

end NUMINAMATH_CALUDE_counterfeit_coin_strategy_exists_l9_991


namespace NUMINAMATH_CALUDE_michaels_matchsticks_l9_995

theorem michaels_matchsticks (total : ℕ) : 
  (30 * 10 + 20 * 15 + 10 * 25 : ℕ) = (2 * total) / 3 → total = 1275 := by
  sorry

end NUMINAMATH_CALUDE_michaels_matchsticks_l9_995


namespace NUMINAMATH_CALUDE_l_shape_subdivision_l9_964

/-- An L shape made of three congruent squares -/
structure LShape where
  -- We don't need to define the internal structure for this problem

/-- The number of L shapes with the same orientation as the original after n subdivisions -/
def same_orientation (n : ℕ) : ℕ :=
  4^(n-1) + 2^(n-1)

/-- The total number of L shapes after n subdivisions -/
def total_shapes (n : ℕ) : ℕ :=
  4^n

theorem l_shape_subdivision (n : ℕ) :
  n > 0 → same_orientation n ≤ total_shapes n ∧
  same_orientation n = (total_shapes (n-1) + 2^(n-1)) := by
  sorry

#eval same_orientation 2005

end NUMINAMATH_CALUDE_l_shape_subdivision_l9_964


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l9_948

/-- The volume of a sphere inscribed in a cube with side length 8 inches -/
theorem volume_of_inscribed_sphere (π : ℝ) : ℝ := by
  -- Define the side length of the cube
  let cube_side : ℝ := 8

  -- Define the radius of the inscribed sphere
  let sphere_radius : ℝ := cube_side / 2

  -- Define the volume of the sphere
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3

  -- Prove that the volume equals (256/3)π cubic inches
  have : sphere_volume = (256 / 3) * π := by sorry

  -- Return the result
  exact (256 / 3) * π

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l9_948


namespace NUMINAMATH_CALUDE_pen_pencil_price_ratio_l9_947

theorem pen_pencil_price_ratio :
  ∀ (pen_price pencil_price total_price : ℚ),
    pencil_price = 8 →
    total_price = 12 →
    total_price = pen_price + pencil_price →
    pen_price / pencil_price = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_price_ratio_l9_947


namespace NUMINAMATH_CALUDE_tanα_tanβ_value_l9_937

theorem tanα_tanβ_value (α β : ℝ) 
  (h1 : Real.cos (α + β) = 1/5)
  (h2 : Real.cos (α - β) = 3/5) :
  Real.tan α * Real.tan β = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tanα_tanβ_value_l9_937


namespace NUMINAMATH_CALUDE_vector_sum_range_l9_916

theorem vector_sum_range (A B : ℝ × ℝ) : 
  ((A.1 - 2)^2 + A.2^2 = 1) →
  ((B.1 - 2)^2 + B.2^2 = 1) →
  (A ≠ B) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 2) →
  (4 - Real.sqrt 2 ≤ ((A.1 + B.1)^2 + (A.2 + B.2)^2).sqrt) ∧
  (((A.1 + B.1)^2 + (A.2 + B.2)^2).sqrt ≤ 4 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_range_l9_916


namespace NUMINAMATH_CALUDE_tia_walking_time_l9_990

/-- Represents a person's walking characteristics and time to destination -/
structure Walker where
  steps_per_minute : ℝ
  step_length : ℝ
  time_to_destination : ℝ

/-- Calculates the distance walked based on walking characteristics and time -/
def distance (w : Walker) : ℝ :=
  w.steps_per_minute * w.step_length * w.time_to_destination

theorem tia_walking_time (ella tia : Walker)
  (h1 : ella.steps_per_minute = 80)
  (h2 : ella.step_length = 80)
  (h3 : ella.time_to_destination = 20)
  (h4 : tia.steps_per_minute = 120)
  (h5 : tia.step_length = 70)
  (h6 : distance ella = distance tia) :
  tia.time_to_destination = 15.24 := by
  sorry

end NUMINAMATH_CALUDE_tia_walking_time_l9_990


namespace NUMINAMATH_CALUDE_min_value_of_expression_l9_934

theorem min_value_of_expression (x y : ℝ) : (x^2*y - 2)^2 + (x^2 + y)^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l9_934


namespace NUMINAMATH_CALUDE_correct_converses_l9_943

-- Proposition 1
def prop1 (x : ℝ) : Prop := x^2 - 3*x + 2 = 0 → x = 1 ∨ x = 2

-- Proposition 2
def prop2 (x : ℝ) : Prop := -2 ≤ x ∧ x < 3 → (x + 2) * (x - 3) ≤ 0

-- Proposition 3
def prop3 (x y : ℝ) : Prop := x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Proposition 4
def prop4 (x y : ℕ) : Prop := x ≠ 0 ∧ y ≠ 0 ∧ Even x ∧ Even y → Even (x + y)

-- Converses of the propositions
def conv1 (x : ℝ) : Prop := x = 1 ∨ x = 2 → x^2 - 3*x + 2 = 0

def conv2 (x : ℝ) : Prop := (x + 2) * (x - 3) ≤ 0 → -2 ≤ x ∧ x < 3

def conv3 (x y : ℝ) : Prop := x^2 + y^2 = 0 → x = 0 ∧ y = 0

def conv4 (x y : ℕ) : Prop := x ≠ 0 ∧ y ≠ 0 ∧ Even (x + y) → Even x ∧ Even y

theorem correct_converses :
  (∀ x, conv1 x) ∧
  (∀ x y, conv3 x y) ∧
  ¬(∀ x, conv2 x) ∧
  ¬(∀ x y, conv4 x y) :=
by sorry

end NUMINAMATH_CALUDE_correct_converses_l9_943


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l9_930

/-- A number is prime or the sum of two consecutive primes -/
def IsPrimeOrSumOfConsecutivePrimes (n : ℕ) : Prop :=
  Nat.Prime n ∨ ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ q = p + 1 ∧ n = p + q

/-- The theorem statement -/
theorem rectangular_solid_surface_area 
  (a b c : ℕ) 
  (ha : IsPrimeOrSumOfConsecutivePrimes a) 
  (hb : IsPrimeOrSumOfConsecutivePrimes b)
  (hc : IsPrimeOrSumOfConsecutivePrimes c)
  (hv : a * b * c = 399) : 
  2 * (a * b + b * c + c * a) = 422 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l9_930


namespace NUMINAMATH_CALUDE_jill_watching_time_l9_971

/-- The total time Jill spent watching shows -/
def total_time (first_show_duration : ℕ) (multiplier : ℕ) : ℕ :=
  first_show_duration + first_show_duration * multiplier

/-- Proof that Jill spent 150 minutes watching shows -/
theorem jill_watching_time : total_time 30 4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_jill_watching_time_l9_971


namespace NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l9_950

/-- Given the conversion rates between bahs, rahs, and yahs, 
    prove that 500 yahs are equal in value to 100 bahs. -/
theorem yahs_to_bahs_conversion 
  (bah_to_rah : ℚ) (rah_to_yah : ℚ)
  (h1 : bah_to_rah = 30 / 10)  -- 10 bahs = 30 rahs
  (h2 : rah_to_yah = 10 / 6)   -- 6 rahs = 10 yahs
  : 500 * (1 / rah_to_yah) * (1 / bah_to_rah) = 100 := by
  sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_conversion_l9_950


namespace NUMINAMATH_CALUDE_meaningful_sqrt_range_l9_959

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (2 / (x - 1))) → x > 1 := by
sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_range_l9_959


namespace NUMINAMATH_CALUDE_a_work_time_l9_989

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 20

/-- The number of days it takes A and B together to complete the work -/
def ab_days : ℝ := 8.571428571428571

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 15

/-- Theorem stating that given the conditions, A can complete the work alone in 15 days -/
theorem a_work_time :
  (1 / a_days + 1 / b_days = 1 / ab_days) → a_days = 15 :=
by sorry

end NUMINAMATH_CALUDE_a_work_time_l9_989


namespace NUMINAMATH_CALUDE_intersects_three_points_iff_m_range_l9_931

/-- A quadratic function f(x) = x^2 + 2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

/-- Predicate indicating if f intersects the coordinate axes at 3 points -/
def intersects_at_three_points (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0 ∧ f m 0 ≠ 0

/-- Theorem stating the range of m for which f intersects the coordinate axes at 3 points -/
theorem intersects_three_points_iff_m_range (m : ℝ) :
  intersects_at_three_points m ↔ m < 1 ∧ m ≠ 0 := by sorry

end NUMINAMATH_CALUDE_intersects_three_points_iff_m_range_l9_931


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l9_997

theorem cube_root_equation_solution :
  let x : ℝ := 168 / 5
  (15 * x + (15 * x + 8) ^ (1/3)) ^ (1/3) = 8 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l9_997


namespace NUMINAMATH_CALUDE_combined_return_percentage_l9_910

theorem combined_return_percentage (investment1 investment2 : ℝ) 
  (return_rate1 return_rate2 : ℝ) :
  investment1 = 500 →
  investment2 = 1500 →
  return_rate1 = 0.07 →
  return_rate2 = 0.27 →
  let total_investment := investment1 + investment2
  let total_return := investment1 * return_rate1 + investment2 * return_rate2
  let combined_return_rate := total_return / total_investment
  combined_return_rate = 0.22 := by
sorry

end NUMINAMATH_CALUDE_combined_return_percentage_l9_910


namespace NUMINAMATH_CALUDE_fraction_evaluation_l9_935

theorem fraction_evaluation (a b : ℝ) (h1 : a = 5) (h2 : b = 3) :
  2 / (a - b) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l9_935


namespace NUMINAMATH_CALUDE_mistaken_division_l9_912

theorem mistaken_division (n : ℕ) : 
  (n / 9 = 8 ∧ n % 9 = 6) → n / 6 = 13 := by
sorry

end NUMINAMATH_CALUDE_mistaken_division_l9_912


namespace NUMINAMATH_CALUDE_y_equation_solution_l9_966

theorem y_equation_solution (y : ℝ) (c d : ℕ+) 
  (h1 : y^2 + 4*y + 4/y + 1/y^2 = 35)
  (h2 : y = c + Real.sqrt d) : 
  c + d = 42 := by sorry

end NUMINAMATH_CALUDE_y_equation_solution_l9_966


namespace NUMINAMATH_CALUDE_gcf_of_154_308_462_l9_963

theorem gcf_of_154_308_462 : Nat.gcd 154 (Nat.gcd 308 462) = 154 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_154_308_462_l9_963


namespace NUMINAMATH_CALUDE_max_value_on_circle_l9_904

/-- The maximum value of x^2 + y^2 for points on the circle x^2 - 4x - 4 + y^2 = 0 -/
theorem max_value_on_circle : 
  ∀ x y : ℝ, x^2 - 4*x - 4 + y^2 = 0 → x^2 + y^2 ≤ 12 + 8*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l9_904


namespace NUMINAMATH_CALUDE_blue_pill_cost_l9_972

def treatment_duration : ℕ := 21 -- 3 weeks * 7 days

def daily_blue_pills : ℕ := 2
def daily_orange_pills : ℕ := 1

def total_cost : ℕ := 966

theorem blue_pill_cost (orange_pill_cost : ℕ) 
  (h1 : orange_pill_cost + 2 = 16) 
  (h2 : (daily_blue_pills * (orange_pill_cost + 2) + daily_orange_pills * orange_pill_cost) * treatment_duration = total_cost) : 
  orange_pill_cost + 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_blue_pill_cost_l9_972


namespace NUMINAMATH_CALUDE_probability_divisible_by_four_probability_calculation_l9_936

def fair_12_sided_die := Finset.range 12

theorem probability_divisible_by_four (a b : ℕ) : 
  a ∈ fair_12_sided_die → b ∈ fair_12_sided_die →
  (a % 4 = 0 ∧ b % 4 = 0) ↔ (10 * a + b) % 4 = 0 ∧ a % 4 = 0 ∧ b % 4 = 0 :=
by sorry

theorem probability_calculation :
  (Finset.filter (λ x : ℕ × ℕ => x.1 % 4 = 0 ∧ x.2 % 4 = 0) (fair_12_sided_die.product fair_12_sided_die)).card /
  (fair_12_sided_die.card * fair_12_sided_die.card : ℚ) = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_four_probability_calculation_l9_936


namespace NUMINAMATH_CALUDE_second_day_student_tickets_second_day_student_tickets_is_ten_l9_969

/-- The price of a student ticket -/
def student_ticket_price : ℕ := 9

/-- The total revenue from the first day of sales -/
def first_day_revenue : ℕ := 79

/-- The total revenue from the second day of sales -/
def second_day_revenue : ℕ := 246

/-- The number of senior citizen tickets sold on the first day -/
def first_day_senior_tickets : ℕ := 4

/-- The number of student tickets sold on the first day -/
def first_day_student_tickets : ℕ := 3

/-- The number of senior citizen tickets sold on the second day -/
def second_day_senior_tickets : ℕ := 12

/-- Calculates the price of a senior citizen ticket based on the first day's sales -/
def senior_ticket_price : ℕ := 
  (first_day_revenue - student_ticket_price * first_day_student_tickets) / first_day_senior_tickets

theorem second_day_student_tickets : ℕ := 
  (second_day_revenue - senior_ticket_price * second_day_senior_tickets) / student_ticket_price

theorem second_day_student_tickets_is_ten : second_day_student_tickets = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_day_student_tickets_second_day_student_tickets_is_ten_l9_969


namespace NUMINAMATH_CALUDE_james_bags_given_away_l9_914

def bags_given_away (initial_marbles : ℕ) (initial_bags : ℕ) (remaining_marbles : ℕ) : ℕ :=
  (initial_marbles - remaining_marbles) / (initial_marbles / initial_bags)

theorem james_bags_given_away :
  let initial_marbles : ℕ := 28
  let initial_bags : ℕ := 4
  let remaining_marbles : ℕ := 21
  bags_given_away initial_marbles initial_bags remaining_marbles = 1 := by
sorry

end NUMINAMATH_CALUDE_james_bags_given_away_l9_914


namespace NUMINAMATH_CALUDE_max_prob_second_game_C_l9_942

variable (p₁ p₂ p₃ : ℝ)

-- Define the probabilities of winning against each player
def prob_A := p₁
def prob_B := p₂
def prob_C := p₃

-- Define the conditions
axiom prob_order : 0 < p₁ ∧ p₁ < p₂ ∧ p₂ < p₃

-- Define the probability of winning two consecutive games for each scenario
def P_A := 2 * (p₁ * (p₂ + p₃) - 2 * p₁ * p₂ * p₃)
def P_B := 2 * (p₂ * (p₁ + p₃) - 2 * p₁ * p₂ * p₃)
def P_C := 2 * (p₁ * p₃ + p₂ * p₃ - 2 * p₁ * p₂ * p₃)

-- Theorem statement
theorem max_prob_second_game_C :
  P_C > P_A ∧ P_C > P_B :=
sorry

end NUMINAMATH_CALUDE_max_prob_second_game_C_l9_942


namespace NUMINAMATH_CALUDE_band_gigs_played_l9_929

theorem band_gigs_played (earnings_per_member : ℕ) (num_members : ℕ) (total_earnings : ℕ) : 
  earnings_per_member = 20 →
  num_members = 4 →
  total_earnings = 400 →
  total_earnings / (earnings_per_member * num_members) = 5 := by
sorry

end NUMINAMATH_CALUDE_band_gigs_played_l9_929


namespace NUMINAMATH_CALUDE_tetrahedron_cube_volume_ratio_l9_982

theorem tetrahedron_cube_volume_ratio :
  let cube_side : ℝ := x
  let cube_volume := cube_side ^ 3
  let tetrahedron_side := cube_side * Real.sqrt 3 / 2
  let tetrahedron_volume := tetrahedron_side ^ 3 * Real.sqrt 2 / 12
  tetrahedron_volume / cube_volume = Real.sqrt 6 / 32 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_cube_volume_ratio_l9_982


namespace NUMINAMATH_CALUDE_astronaut_selection_probability_l9_900

/-- The probability of selecting one male and one female astronaut from a group of two male and two female astronauts -/
theorem astronaut_selection_probability : 
  let total_astronauts : ℕ := 4
  let male_astronauts : ℕ := 2
  let female_astronauts : ℕ := 2
  let selected_astronauts : ℕ := 2
  
  (Nat.choose male_astronauts 1 * Nat.choose female_astronauts 1) / 
  Nat.choose total_astronauts selected_astronauts = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_astronaut_selection_probability_l9_900


namespace NUMINAMATH_CALUDE_decreasing_function_on_positive_reals_l9_927

/-- The function f(x) = 9 - x² is decreasing on the interval (0, +∞) -/
theorem decreasing_function_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → (9 - x^2 : ℝ) > (9 - y^2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_on_positive_reals_l9_927


namespace NUMINAMATH_CALUDE_plane_perpendicular_through_perpendicular_line_line_not_perpendicular_in_perpendicular_planes_l9_924

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (perpendicular : Plane → Plane → Prop)
variable (passes_through : Plane → Line → Prop)
variable (perpendicular_line : Line → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (line_of_intersection : Plane → Plane → Line)
variable (perpendicular_to_line : Line → Line → Prop)

-- Proposition 2
theorem plane_perpendicular_through_perpendicular_line 
  (p1 p2 : Plane) (l : Line) :
  perpendicular_line l p2 → passes_through p1 l → perpendicular p1 p2 :=
sorry

-- Proposition 4
theorem line_not_perpendicular_in_perpendicular_planes 
  (p1 p2 : Plane) (l : Line) :
  perpendicular p1 p2 →
  in_plane l p1 →
  ¬ perpendicular_to_line l (line_of_intersection p1 p2) →
  ¬ perpendicular_line l p2 :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_through_perpendicular_line_line_not_perpendicular_in_perpendicular_planes_l9_924


namespace NUMINAMATH_CALUDE_correct_subtraction_l9_920

theorem correct_subtraction (x : ℤ) (h : x - 63 = 8) : x - 36 = 35 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l9_920


namespace NUMINAMATH_CALUDE_cos_two_alpha_l9_938

theorem cos_two_alpha (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_l9_938


namespace NUMINAMATH_CALUDE_sales_equation_l9_917

/-- Represents the salesman's total sales -/
def S : ℝ := sorry

/-- Old commission rate -/
def old_rate : ℝ := 0.05

/-- New fixed salary -/
def new_fixed_salary : ℝ := 1300

/-- New commission rate -/
def new_rate : ℝ := 0.025

/-- Sales threshold for new commission -/
def threshold : ℝ := 4000

/-- Difference in remuneration between new and old schemes -/
def remuneration_difference : ℝ := 600

/-- Theorem stating the equation that the salesman's total sales must satisfy -/
theorem sales_equation : 
  new_fixed_salary + new_rate * (S - threshold) = old_rate * S + remuneration_difference :=
sorry

end NUMINAMATH_CALUDE_sales_equation_l9_917


namespace NUMINAMATH_CALUDE_equation_solution_l9_974

theorem equation_solution : 
  ∀ x y z : ℕ+, 
  (x : ℚ) / 21 * (y : ℚ) / 189 + (z : ℚ) = 1 → 
  x = 21 ∧ y = 567 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l9_974


namespace NUMINAMATH_CALUDE_tan_2alpha_value_l9_996

theorem tan_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 3 * Real.cos (2 * α) - 4 * Real.sin α = 1) : 
  Real.tan (2 * α) = -4 * Real.sqrt 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_2alpha_value_l9_996


namespace NUMINAMATH_CALUDE_complex_trig_simplification_l9_915

open Complex

theorem complex_trig_simplification (θ : ℝ) :
  let z : ℂ := (cos θ - I * sin θ)^8 * (1 + I * tan θ)^5 / ((cos θ + I * sin θ)^2 * (tan θ + I))
  z = -1 / (cos θ)^4 * (sin (4*θ) + I * cos (4*θ)) :=
sorry

end NUMINAMATH_CALUDE_complex_trig_simplification_l9_915


namespace NUMINAMATH_CALUDE_f_period_and_range_l9_970

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3 * (Real.sin x)^2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_period_and_range :
  (∃ T > 0, is_periodic f T ∧ ∀ T' > 0, is_periodic f T' → T ≤ T') ∧
  (∀ y ∈ Set.range (f ∘ (fun x => x * π / 3)), -Real.sqrt 3 ≤ y ∧ y ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_f_period_and_range_l9_970


namespace NUMINAMATH_CALUDE_prime_expressions_l9_988

theorem prime_expressions (p : ℤ) : 
  Prime p ∧ Prime (2*p + 1) ∧ Prime (4*p + 1) ∧ Prime (6*p + 1) ↔ p = -2 ∨ p = -3 ∨ p = 3 :=
sorry

end NUMINAMATH_CALUDE_prime_expressions_l9_988


namespace NUMINAMATH_CALUDE_opposite_of_negative_abs_two_l9_940

theorem opposite_of_negative_abs_two : -(- |(-2)|) = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_abs_two_l9_940


namespace NUMINAMATH_CALUDE_autobiography_to_fiction_ratio_l9_967

theorem autobiography_to_fiction_ratio
  (total_books : ℕ)
  (fiction_books : ℕ)
  (nonfiction_books : ℕ)
  (picture_books : ℕ)
  (h_total : total_books = 35)
  (h_fiction : fiction_books = 5)
  (h_nonfiction : nonfiction_books = fiction_books + 4)
  (h_picture : picture_books = 11)
  : (total_books - fiction_books - nonfiction_books - picture_books) / fiction_books = 2 := by
  sorry

end NUMINAMATH_CALUDE_autobiography_to_fiction_ratio_l9_967


namespace NUMINAMATH_CALUDE_tan_beta_plus_pi_sixth_l9_905

open Real

theorem tan_beta_plus_pi_sixth (α β : ℝ) 
  (h1 : tan (α - π/6) = 2) 
  (h2 : tan (α + β) = -3) : 
  tan (β + π/6) = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_plus_pi_sixth_l9_905


namespace NUMINAMATH_CALUDE_pandemic_cut_fifty_percent_l9_951

/-- Represents a car factory with its production details -/
structure CarFactory where
  doorsPerCar : ℕ
  initialProduction : ℕ
  metalShortageDecrease : ℕ
  finalDoorProduction : ℕ

/-- Calculates the percentage of production cut due to a pandemic -/
def pandemicProductionCutPercentage (factory : CarFactory) : ℚ :=
  let productionAfterMetalShortage := factory.initialProduction - factory.metalShortageDecrease
  let finalCarProduction := factory.finalDoorProduction / factory.doorsPerCar
  let pandemicCut := productionAfterMetalShortage - finalCarProduction
  (pandemicCut / productionAfterMetalShortage) * 100

/-- Theorem stating that the pandemic production cut percentage is 50% for the given factory conditions -/
theorem pandemic_cut_fifty_percent (factory : CarFactory) 
  (h1 : factory.doorsPerCar = 5)
  (h2 : factory.initialProduction = 200)
  (h3 : factory.metalShortageDecrease = 50)
  (h4 : factory.finalDoorProduction = 375) :
  pandemicProductionCutPercentage factory = 50 := by
  sorry

#eval pandemicProductionCutPercentage ⟨5, 200, 50, 375⟩

end NUMINAMATH_CALUDE_pandemic_cut_fifty_percent_l9_951


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l9_932

theorem binomial_coefficient_divisibility (p k : ℕ) (hp : Nat.Prime p) (hk : 1 ≤ k ∧ k ≤ p - 1) :
  p ∣ Nat.choose p k := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l9_932


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l9_956

-- Define the function f(x) = 2x^2
def f (x : ℝ) : ℝ := 2 * x^2

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l9_956


namespace NUMINAMATH_CALUDE_equal_area_rectangle_width_l9_923

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangle_width (r1 r2 : Rectangle) 
  (h1 : r1.length = 12)
  (h2 : r1.width = 10)
  (h3 : r2.length = 24)
  (h4 : area r1 = area r2) :
  r2.width = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangle_width_l9_923


namespace NUMINAMATH_CALUDE_preimage_of_neg_one_plus_two_i_l9_983

/-- The complex transformation f(Z) = (1+i)Z -/
def f (Z : ℂ) : ℂ := (1 + Complex.I) * Z

/-- Theorem: The pre-image of -1+2i under f is (1+3i)/2 -/
theorem preimage_of_neg_one_plus_two_i :
  f ((1 + 3 * Complex.I) / 2) = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_neg_one_plus_two_i_l9_983


namespace NUMINAMATH_CALUDE_game_ends_after_46_rounds_game_doesnt_end_before_46_rounds_l9_918

/-- Represents the state of the game at any point --/
structure GameState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a single round of the game --/
def play_round (state : GameState) : GameState :=
  if state.a ≥ state.b ∧ state.a ≥ state.c then
    { a := state.a - 3, b := state.b + 1, c := state.c + 1 }
  else if state.b ≥ state.a ∧ state.b ≥ state.c then
    { a := state.a + 1, b := state.b - 3, c := state.c + 1 }
  else
    { a := state.a + 1, b := state.b + 1, c := state.c - 3 }

/-- Plays the game for a given number of rounds --/
def play_game (initial_state : GameState) (rounds : ℕ) : GameState :=
  match rounds with
  | 0 => initial_state
  | n + 1 => play_round (play_game initial_state n)

/-- The main theorem stating that the game ends after 46 rounds --/
theorem game_ends_after_46_rounds :
  let initial_state := { a := 18, b := 17, c := 16 : GameState }
  let final_state := play_game initial_state 46
  final_state.a = 0 ∨ final_state.b = 0 ∨ final_state.c = 0 :=
by sorry

/-- The game doesn't end before 46 rounds --/
theorem game_doesnt_end_before_46_rounds :
  let initial_state := { a := 18, b := 17, c := 16 : GameState }
  ∀ n < 46, let state := play_game initial_state n
    state.a > 0 ∧ state.b > 0 ∧ state.c > 0 :=
by sorry

end NUMINAMATH_CALUDE_game_ends_after_46_rounds_game_doesnt_end_before_46_rounds_l9_918


namespace NUMINAMATH_CALUDE_s13_is_constant_l9_986

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Given S_4 + a_25 = 5, S_13 is constant -/
theorem s13_is_constant (seq : ArithmeticSequence) 
    (h : seq.S 4 + seq.a 25 = 5) : 
  ∃ c : ℝ, seq.S 13 = c := by
  sorry

end NUMINAMATH_CALUDE_s13_is_constant_l9_986


namespace NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l9_907

/-- Given a rectangle EFGH and an ellipse, prove the perimeter of the rectangle -/
theorem rectangle_ellipse_perimeter (p q c d : ℝ) : 
  p > 0 → q > 0 → c > 0 → d > 0 →
  p * q = 4032 →
  π * c * d = 2016 * π →
  p + q = 2 * c →
  p^2 + q^2 = 4 * (c^2 - d^2) →
  2 * (p + q) = 8 * Real.sqrt 2016 := by
sorry


end NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l9_907


namespace NUMINAMATH_CALUDE_lcm_of_5_6_9_21_l9_968

theorem lcm_of_5_6_9_21 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 9 21)) = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_9_21_l9_968


namespace NUMINAMATH_CALUDE_problems_per_worksheet_l9_978

theorem problems_per_worksheet
  (total_worksheets : ℕ)
  (graded_worksheets : ℕ)
  (remaining_problems : ℕ)
  (h1 : total_worksheets = 15)
  (h2 : graded_worksheets = 7)
  (h3 : remaining_problems = 24)
  : (remaining_problems / (total_worksheets - graded_worksheets) : ℚ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_problems_per_worksheet_l9_978


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l9_999

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  cylinder_height = 6 →
  cube_side ^ 3 - π * cylinder_radius ^ 2 * cylinder_height = 216 - 54 * π :=
by sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l9_999


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l9_961

/-- An isosceles triangle with sides 4 and 9 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 4 ∧ b = 9 ∧ c = 9 →  -- Two sides are 9, one side is 4
  a < b + c ∧ b < a + c ∧ c < a + b →  -- Triangle inequality
  a + b + c = 22 :=  -- Perimeter is 22
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l9_961


namespace NUMINAMATH_CALUDE_student_ticket_price_l9_965

theorem student_ticket_price (senior_price student_price : ℚ) : 
  (4 * senior_price + 3 * student_price = 79) →
  (12 * senior_price + 10 * student_price = 246) →
  student_price = 9 := by
sorry

end NUMINAMATH_CALUDE_student_ticket_price_l9_965


namespace NUMINAMATH_CALUDE_no_three_common_tangents_l9_984

/-- Two circles in the same plane with different radii -/
structure TwoCircles where
  plane : Type*
  circle1 : Set plane
  circle2 : Set plane
  radius1 : ℝ
  radius2 : ℝ
  different_radii : radius1 ≠ radius2

/-- A common tangent to two circles -/
def CommonTangent (tc : TwoCircles) (line : Set tc.plane) : Prop := sorry

/-- The number of common tangents to two circles -/
def NumCommonTangents (tc : TwoCircles) : ℕ := sorry

/-- Theorem: Two circles with different radii cannot have exactly 3 common tangents -/
theorem no_three_common_tangents (tc : TwoCircles) : 
  NumCommonTangents tc ≠ 3 := by sorry

end NUMINAMATH_CALUDE_no_three_common_tangents_l9_984


namespace NUMINAMATH_CALUDE_equation_system_solutions_l9_921

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(1, 5, 2, 3), (1, 5, 3, 2), (5, 1, 2, 3), (5, 1, 3, 2),
   (2, 3, 1, 5), (2, 3, 5, 1), (3, 2, 1, 5), (3, 2, 5, 1),
   (2, 2, 2, 2)}

theorem equation_system_solutions :
  ∀ x y z t : ℕ,
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 →
    x + y = z * t ∧ z + t = x * y ↔ (x, y, z, t) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_equation_system_solutions_l9_921


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l9_925

def a (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_terms :
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k ∧
  ∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k ∧
  k = 3 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l9_925


namespace NUMINAMATH_CALUDE_sequence_formula_correct_l9_908

/-- The general term formula for the sequence -1/2, 1/4, -1/8, 1/16, ... -/
def sequence_formula (n : ℕ) : ℚ := (-1)^(n+1) / (2^n)

/-- The nth term of the sequence -1/2, 1/4, -1/8, 1/16, ... -/
def sequence_term (n : ℕ) : ℚ := 
  if n % 2 = 1 
  then -1 / (2^n) 
  else 1 / (2^n)

theorem sequence_formula_correct : 
  ∀ n : ℕ, n > 0 → sequence_formula n = sequence_term n :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_correct_l9_908


namespace NUMINAMATH_CALUDE_equal_probability_sums_l9_981

/-- Represents a standard six-sided die -/
def Die := Fin 6

/-- The number of dice being rolled -/
def numDice : ℕ := 8

/-- The sum we're comparing to -/
def targetSum : ℕ := 12

/-- Function to calculate the complementary sum -/
def complementarySum (n : ℕ) : ℕ := 2 * (numDice * 3 + numDice) - n

/-- Theorem stating that the sum of 44 occurs with the same probability as the sum of 12 -/
theorem equal_probability_sums :
  complementarySum targetSum = 44 := by
  sorry

end NUMINAMATH_CALUDE_equal_probability_sums_l9_981


namespace NUMINAMATH_CALUDE_sine_HAC_specific_prism_l9_987

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  a : ℝ  -- length
  b : ℝ  -- width
  c : ℝ  -- height
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D
  F : Point3D
  G : Point3D
  H : Point3D

/-- Calculate the sine of the angle HAC in a rectangular prism -/
def sineHAC (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem: The sine of angle HAC in the given rectangular prism is √143 / 13 -/
theorem sine_HAC_specific_prism :
  let prism : RectangularPrism := {
    a := 2,
    b := 2,
    c := 3,
    A := ⟨0, 0, 0⟩,
    B := ⟨2, 0, 0⟩,
    C := ⟨2, 2, 0⟩,
    D := ⟨0, 2, 0⟩,
    E := ⟨0, 0, 3⟩,
    F := ⟨2, 0, 3⟩,
    G := ⟨2, 2, 3⟩,
    H := ⟨0, 2, 3⟩
  }
  sineHAC prism = Real.sqrt 143 / 13 := by
  sorry

end NUMINAMATH_CALUDE_sine_HAC_specific_prism_l9_987


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_on_parabola_l9_960

/-- Given points A and B on the parabola y = -2x^2 forming an isosceles right triangle ABO 
    with O at the origin, prove that the length of OA (equal to OB) is √5 when a = 1. -/
theorem isosceles_right_triangle_on_parabola :
  ∀ (a : ℝ), 
  let A : ℝ × ℝ := (a, -2 * a^2)
  let B : ℝ × ℝ := (-a, -2 * a^2)
  let O : ℝ × ℝ := (0, 0)
  -- A and B are on the parabola y = -2x^2
  (A.2 = -2 * A.1^2 ∧ B.2 = -2 * B.1^2) →
  -- ABO is an isosceles right triangle with right angle at O
  (Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) = Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2)) →
  (A.1 - O.1)^2 + (A.2 - O.2)^2 + (B.1 - O.1)^2 + (B.2 - O.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 →
  -- When a = 1, the length of OA (equal to OB) is √5
  a = 1 → Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) = Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_right_triangle_on_parabola_l9_960


namespace NUMINAMATH_CALUDE_hallie_net_earnings_l9_962

def hourly_rate : ℝ := 10

def monday_hours : ℝ := 7
def monday_tips : ℝ := 18

def tuesday_hours : ℝ := 5
def tuesday_tips : ℝ := 12

def wednesday_hours : ℝ := 7
def wednesday_tips : ℝ := 20

def thursday_hours : ℝ := 8
def thursday_tips : ℝ := 25

def friday_hours : ℝ := 6
def friday_tips : ℝ := 15

def discount_rate : ℝ := 0.05

def total_earnings : ℝ := 
  (monday_hours * hourly_rate + monday_tips) +
  (tuesday_hours * hourly_rate + tuesday_tips) +
  (wednesday_hours * hourly_rate + wednesday_tips) +
  (thursday_hours * hourly_rate + thursday_tips) +
  (friday_hours * hourly_rate + friday_tips)

def discount_amount : ℝ := total_earnings * discount_rate

def net_earnings : ℝ := total_earnings - discount_amount

theorem hallie_net_earnings : net_earnings = 399 := by
  sorry

end NUMINAMATH_CALUDE_hallie_net_earnings_l9_962


namespace NUMINAMATH_CALUDE_largest_812_double_l9_928

/-- Converts a natural number to its base-8 representation as a list of digits --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Interprets a list of digits as a base-12 number --/
def fromBase12 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a number is an 8-12 double --/
def is812Double (n : ℕ) : Prop :=
  fromBase12 (toBase8 n) = 3 * n

theorem largest_812_double :
  ∀ n : ℕ, n > 3 → ¬(is812Double n) :=
sorry

end NUMINAMATH_CALUDE_largest_812_double_l9_928


namespace NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l9_919

-- Define an arithmetic sequence
def is_arithmetic_sequence (a b c : ℤ) : Prop :=
  b - a = c - b

-- State the theorem
theorem middle_term_of_arithmetic_sequence :
  ∀ y : ℤ, is_arithmetic_sequence (3^2) y (3^4) → y = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l9_919


namespace NUMINAMATH_CALUDE_sin_arctan_equality_l9_902

theorem sin_arctan_equality : ∃ (x : ℝ), x > 0 ∧ Real.sin (Real.arctan x) = x := by
  let x := Real.sqrt ((-1 + Real.sqrt 5) / 2)
  use x
  have h1 : x > 0 := sorry
  have h2 : Real.sin (Real.arctan x) = x := sorry
  exact ⟨h1, h2⟩

#check sin_arctan_equality

end NUMINAMATH_CALUDE_sin_arctan_equality_l9_902


namespace NUMINAMATH_CALUDE_total_interest_is_860_l9_939

def inheritance : ℝ := 12000
def investment1 : ℝ := 5000
def rate1 : ℝ := 0.06
def rate2 : ℝ := 0.08

def total_interest : ℝ :=
  investment1 * rate1 + (inheritance - investment1) * rate2

theorem total_interest_is_860 : total_interest = 860 := by sorry

end NUMINAMATH_CALUDE_total_interest_is_860_l9_939


namespace NUMINAMATH_CALUDE_division_result_l9_903

theorem division_result : (64 : ℝ) / 0.08 = 800 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l9_903


namespace NUMINAMATH_CALUDE_infinite_primes_l9_949

theorem infinite_primes : ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_l9_949
