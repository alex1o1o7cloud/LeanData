import Mathlib

namespace NUMINAMATH_CALUDE_subway_construction_equation_l2821_282146

/-- Represents the subway construction scenario -/
structure SubwayConstruction where
  total_length : ℝ
  extra_meters_per_day : ℝ
  days_saved : ℝ
  original_plan : ℝ

/-- The equation holds for the given subway construction scenario -/
def equation_holds (sc : SubwayConstruction) : Prop :=
  sc.total_length / sc.original_plan - sc.total_length / (sc.original_plan + sc.extra_meters_per_day) = sc.days_saved

/-- Theorem stating that the equation holds for the specific scenario described in the problem -/
theorem subway_construction_equation :
  ∀ (sc : SubwayConstruction),
    sc.total_length = 120 ∧
    sc.extra_meters_per_day = 5 ∧
    sc.days_saved = 4 →
    equation_holds sc :=
by
  sorry

#check subway_construction_equation

end NUMINAMATH_CALUDE_subway_construction_equation_l2821_282146


namespace NUMINAMATH_CALUDE_coefficient_x4y_value_l2821_282101

/-- The coefficient of x^4y in the expansion of (x^2 + y + 3)^6 -/
def coefficient_x4y (x y : ℕ) : ℕ :=
  (Nat.choose 6 4) * (Nat.choose 4 3) * (3^3)

/-- Theorem stating that the coefficient of x^4y in (x^2 + y + 3)^6 is 1620 -/
theorem coefficient_x4y_value :
  ∀ x y, coefficient_x4y x y = 1620 := by
  sorry

#eval coefficient_x4y 0 0  -- To check the result

end NUMINAMATH_CALUDE_coefficient_x4y_value_l2821_282101


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2821_282137

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 9| = |x - 3| := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2821_282137


namespace NUMINAMATH_CALUDE_train_crossing_time_l2821_282118

/-- Proves that a train with given length and speed takes the specified time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 3 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2821_282118


namespace NUMINAMATH_CALUDE_s₁_less_than_s₂_l2821_282185

/-- Centroid of a triangle -/
structure Centroid (Point : Type*) (Triangle : Type*) where
  center : Point
  triangle : Triangle

/-- Calculate s₁ for a triangle with its centroid -/
def s₁ {Point : Type*} {Triangle : Type*} (c : Centroid Point Triangle) (distance : Point → Point → ℝ) : ℝ :=
  let G := c.center
  let A := sorry
  let B := sorry
  let C := sorry
  2 * (distance G A + distance G B + distance G C)

/-- Calculate s₂ for a triangle -/
def s₂ {Point : Type*} {Triangle : Type*} (t : Triangle) (distance : Point → Point → ℝ) : ℝ :=
  let A := sorry
  let B := sorry
  let C := sorry
  3 * (distance A B + distance B C + distance C A)

/-- The main theorem: s₁ < s₂ for any triangle with its centroid -/
theorem s₁_less_than_s₂ {Point : Type*} {Triangle : Type*} 
  (c : Centroid Point Triangle) (distance : Point → Point → ℝ) :
  s₁ c distance < s₂ c.triangle distance :=
sorry

end NUMINAMATH_CALUDE_s₁_less_than_s₂_l2821_282185


namespace NUMINAMATH_CALUDE_partition_infinite_multiples_l2821_282172

-- Define a partition of Natural Numbers
def Partition (A : ℕ → Set ℕ) (k : ℕ) : Prop :=
  (∀ n, ∃! i, i ≤ k ∧ n ∈ A i) ∧
  (∀ i, i ≤ k → Set.Nonempty (A i))

-- Define what it means for a set to contain infinitely many multiples of a number
def InfiniteMultiples (S : Set ℕ) (x : ℕ) : Prop :=
  Set.Infinite {n ∈ S | ∃ k, n = k * x}

-- Main theorem
theorem partition_infinite_multiples 
  {A : ℕ → Set ℕ} {k : ℕ} (h : Partition A k) :
  ∃ i, i ≤ k ∧ ∀ x : ℕ, x > 0 → InfiniteMultiples (A i) x :=
sorry

end NUMINAMATH_CALUDE_partition_infinite_multiples_l2821_282172


namespace NUMINAMATH_CALUDE_trip_savings_l2821_282123

/-- The amount Trip can save by going to the earlier movie. -/
def total_savings (evening_ticket_cost : ℚ) (food_combo_cost : ℚ) 
  (ticket_discount_percent : ℚ) (food_discount_percent : ℚ) : ℚ :=
  (ticket_discount_percent / 100) * evening_ticket_cost + 
  (food_discount_percent / 100) * food_combo_cost

/-- Proof that Trip can save $7 by going to the earlier movie. -/
theorem trip_savings : 
  total_savings 10 10 20 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_trip_savings_l2821_282123


namespace NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l2821_282169

/-- Given a line segment connecting (1, -3) and (-4, 9), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that a^2 + b^2 + c^2 + d^2 = 179 -/
theorem line_segment_parameter_sum_of_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t → t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = 1 ∧ d = -3) →
  (a + b = -4 ∧ c + d = 9) →
  a^2 + b^2 + c^2 + d^2 = 179 := by
sorry

end NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l2821_282169


namespace NUMINAMATH_CALUDE_orange_shelves_l2821_282176

/-- The number of oranges on the nth shelf -/
def oranges_on_shelf (n : ℕ) : ℕ := 3 + 5 * (n - 1)

/-- The total number of oranges on n shelves -/
def total_oranges (n : ℕ) : ℕ := n * (oranges_on_shelf 1 + oranges_on_shelf n) / 2

theorem orange_shelves :
  ∃ n : ℕ, n > 0 ∧ total_oranges n = 325 :=
sorry

end NUMINAMATH_CALUDE_orange_shelves_l2821_282176


namespace NUMINAMATH_CALUDE_ibrahim_purchase_l2821_282126

/-- The amount of money Ibrahim lacks to purchase an MP3 player and a CD -/
def money_lacking (mp3_cost cd_cost savings father_contribution : ℕ) : ℕ :=
  (mp3_cost + cd_cost) - (savings + father_contribution)

/-- Theorem: Ibrahim lacks 64 euros -/
theorem ibrahim_purchase :
  money_lacking 120 19 55 20 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ibrahim_purchase_l2821_282126


namespace NUMINAMATH_CALUDE_complement_B_union_A_when_a_is_1_A_subset_B_iff_a_in_range_l2821_282151

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2*x + a ∧ 2*x + a ≤ 3}
def B : Set ℝ := {x | -1/2 < x ∧ x < 2}

-- Theorem for part (1)
theorem complement_B_union_A_when_a_is_1 :
  (Set.univ \ B) ∪ A 1 = {x | x ≤ 1 ∨ x ≥ 2} := by sorry

-- Theorem for part (2)
theorem A_subset_B_iff_a_in_range (a : ℝ) :
  A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_complement_B_union_A_when_a_is_1_A_subset_B_iff_a_in_range_l2821_282151


namespace NUMINAMATH_CALUDE_number_divided_by_eight_l2821_282157

theorem number_divided_by_eight : ∀ x : ℝ, x / 8 = 4 → x = 32 := by sorry

end NUMINAMATH_CALUDE_number_divided_by_eight_l2821_282157


namespace NUMINAMATH_CALUDE_fair_haired_women_percentage_l2821_282106

/-- Given a company where:
  * 20% of employees are women with fair hair
  * 50% of employees have fair hair
  Prove that 40% of fair-haired employees are women -/
theorem fair_haired_women_percentage
  (total_employees : ℕ)
  (women_fair_hair_percent : ℚ)
  (fair_hair_percent : ℚ)
  (h1 : women_fair_hair_percent = 20 / 100)
  (h2 : fair_hair_percent = 50 / 100) :
  (women_fair_hair_percent * total_employees) / (fair_hair_percent * total_employees) = 40 / 100 :=
sorry

end NUMINAMATH_CALUDE_fair_haired_women_percentage_l2821_282106


namespace NUMINAMATH_CALUDE_constant_function_invariant_l2821_282144

-- Define g as a function from ℝ to ℝ
def g : ℝ → ℝ := λ x => 5

-- Theorem statement
theorem constant_function_invariant (x : ℝ) : g (3 * x - 7) = 5 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_invariant_l2821_282144


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2821_282147

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2821_282147


namespace NUMINAMATH_CALUDE_magic_square_theorem_l2821_282117

/-- A type representing a 3x3 grid -/
def Grid := Fin 3 → Fin 3 → ℤ

/-- The set of numbers to be used in the grid -/
def GridNumbers : Finset ℤ := {-3, -2, -1, 0, 1, 2, 3, 4, 5}

/-- The sum of each row, column, and diagonal is equal -/
def is_magic (g : Grid) : Prop :=
  let sum := g 0 0 + g 0 1 + g 0 2
  ∀ i j, (i = j → g i 0 + g i 1 + g i 2 = sum) ∧
         (i = j → g 0 j + g 1 j + g 2 j = sum) ∧
         ((i = 0 ∧ j = 0) → g 0 0 + g 1 1 + g 2 2 = sum) ∧
         ((i = 0 ∧ j = 2) → g 0 2 + g 1 1 + g 2 0 = sum)

/-- The theorem to be proved -/
theorem magic_square_theorem (g : Grid) 
  (h1 : g 0 0 = -2)
  (h2 : g 0 2 = 0)
  (h3 : g 2 2 = 4)
  (h4 : is_magic g)
  (h5 : ∀ i j, g i j ∈ GridNumbers)
  (h6 : ∀ x, x ∈ GridNumbers → ∃! i j, g i j = x) :
  ∃ a b c, g 0 1 = a ∧ g 2 1 = b ∧ g 2 0 = c ∧ a - b - c = 4 :=
sorry

end NUMINAMATH_CALUDE_magic_square_theorem_l2821_282117


namespace NUMINAMATH_CALUDE_choir_members_count_l2821_282116

theorem choir_members_count :
  ∃ n : ℕ,
    (n + 4) % 10 = 0 ∧
    (n + 5) % 11 = 0 ∧
    200 < n ∧
    n < 300 ∧
    n = 226 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l2821_282116


namespace NUMINAMATH_CALUDE_quadratic_root_difference_squares_l2821_282167

theorem quadratic_root_difference_squares (b : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - b*x₁ + 12 = 0 ∧ x₂^2 - b*x₂ + 12 = 0 ∧ x₁^2 - x₂^2 = 7) → 
  b = 7 ∨ b = -7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_squares_l2821_282167


namespace NUMINAMATH_CALUDE_expression_value_l2821_282154

theorem expression_value : ∀ a b : ℝ, 
  (a * (1 : ℝ)^4 + b * (1 : ℝ)^2 + 2 = -3) → 
  (a * (-1 : ℝ)^4 + b * (-1 : ℝ)^2 - 2 = -7) := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2821_282154


namespace NUMINAMATH_CALUDE_downward_parabola_m_range_l2821_282105

/-- A parabola that opens downwards -/
structure DownwardParabola where
  m : ℝ
  eq : ℝ → ℝ := fun x ↦ (m + 3) * x^2 + 1
  opens_downward : m + 3 < 0

/-- The range of m for a downward opening parabola -/
theorem downward_parabola_m_range (p : DownwardParabola) : p.m < -3 := by
  sorry

end NUMINAMATH_CALUDE_downward_parabola_m_range_l2821_282105


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2821_282125

theorem proposition_equivalence (A : Set α) (x y : α) :
  (x ∈ A → y ∉ A) ↔ (y ∈ A → x ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2821_282125


namespace NUMINAMATH_CALUDE_fourth_group_size_l2821_282199

theorem fourth_group_size (total : ℕ) (group1 group2 group3 : ℕ) 
  (h1 : total = 24) 
  (h2 : group1 = 5) 
  (h3 : group2 = 8) 
  (h4 : group3 = 7) : 
  total - (group1 + group2 + group3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_group_size_l2821_282199


namespace NUMINAMATH_CALUDE_complex_division_example_l2821_282168

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that the complex number 2i/(1+i) equals 1+i -/
theorem complex_division_example : (2 * i) / (1 + i) = 1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_example_l2821_282168


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l2821_282143

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x = 0 ∨ 3*x - 4*y - 8 = 0

-- Theorem statement
theorem circle_and_line_properties :
  ∃ (a : ℝ),
    -- Circle C has its center on the x-axis
    (∀ x y : ℝ, circle_C x y → y = 0) ∧
    -- Circle C passes through the point (0, √3)
    circle_C 0 (Real.sqrt 3) ∧
    -- Circle C is tangent to the line x=-1
    (∀ y : ℝ, circle_C (-1) y → (x : ℝ) → x = -1 → (circle_C x y → x = -1)) ∧
    -- Line l passes through the point (0,-2)
    line_l 0 (-2) ∧
    -- The chord intercepted by circle C on line l has a length of 2√3
    (∃ x₁ y₁ x₂ y₂ : ℝ, 
      line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ 
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l2821_282143


namespace NUMINAMATH_CALUDE_expected_wealth_difference_10_days_l2821_282195

/-- Represents the daily outcome for an agent --/
inductive DailyOutcome
  | Win
  | Lose
  | Reset

/-- Represents the state of wealth for both agents --/
structure WealthState :=
  (cat : ℤ)
  (fox : ℤ)

/-- Defines the probability distribution for daily outcomes --/
def dailyProbability : DailyOutcome → ℝ
  | DailyOutcome.Win => 0.25
  | DailyOutcome.Lose => 0.25
  | DailyOutcome.Reset => 0.5

/-- Updates the wealth state based on the daily outcome --/
def updateWealth (state : WealthState) (outcome : DailyOutcome) : WealthState :=
  match outcome with
  | DailyOutcome.Win => { cat := state.cat + 1, fox := state.fox }
  | DailyOutcome.Lose => { cat := state.cat, fox := state.fox + 1 }
  | DailyOutcome.Reset => { cat := 0, fox := 0 }

/-- Calculates the expected value of the absolute difference in wealth after n days --/
def expectedWealthDifference (n : ℕ) : ℝ :=
  sorry

theorem expected_wealth_difference_10_days :
  expectedWealthDifference 10 = 1 :=
sorry

end NUMINAMATH_CALUDE_expected_wealth_difference_10_days_l2821_282195


namespace NUMINAMATH_CALUDE_stating_trapezoid_equal_area_segment_length_trapezoid_floor_x_squared_div_150_l2821_282188

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  -- Length of the shorter base
  b : ℝ
  -- Height of the trapezoid
  h : ℝ
  -- The segment joining midpoints of legs divides area in 3:4 ratio
  midpoint_divides_area : (2 * b + 75) / (2 * b + 225) = 3 / 4
  -- Length of segment parallel to bases dividing area equally
  x : ℝ
  -- x divides the trapezoid into two equal areas
  x_divides_equally : x * (b + x / 2) = 150

/-- 
Theorem stating that for a trapezoid with the given properties,
the length of the segment dividing it into equal areas is 75.
-/
theorem trapezoid_equal_area_segment_length (t : Trapezoid) : t.x = 75 := by
  sorry

/-- 
Corollary stating that the greatest integer not exceeding x^2/150 is 37.
-/
theorem trapezoid_floor_x_squared_div_150 (t : Trapezoid) : 
  ⌊(t.x^2 / 150 : ℝ)⌋ = 37 := by
  sorry

end NUMINAMATH_CALUDE_stating_trapezoid_equal_area_segment_length_trapezoid_floor_x_squared_div_150_l2821_282188


namespace NUMINAMATH_CALUDE_rectangle_area_sum_l2821_282119

theorem rectangle_area_sum (a b c d : ℝ) 
  (ha : a = 20) (hb : b = 40) (hc : c = 48) (hd : d = 42) :
  a + b + c + d = 150 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_sum_l2821_282119


namespace NUMINAMATH_CALUDE_cube_root_of_four_fifth_powers_l2821_282138

theorem cube_root_of_four_fifth_powers (x : ℝ) :
  x = (5^6 + 5^6 + 5^6 + 5^6)^(1/3) → x = 25 * 2^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_four_fifth_powers_l2821_282138


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2821_282149

theorem polynomial_divisibility (r s : ℝ) : 
  (∀ x : ℝ, (x - 2) * (x + 1) ∣ (x^6 - x^5 + 3*x^4 - r*x^3 + s*x^2 + 3*x - 7)) ↔ 
  (r = 33/4 ∧ s = -13/4) := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2821_282149


namespace NUMINAMATH_CALUDE_square_area_ratio_l2821_282191

theorem square_area_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (4 * a = 4 * (4 * b)) → a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2821_282191


namespace NUMINAMATH_CALUDE_base_conversion_l2821_282150

theorem base_conversion (b : ℕ) (h1 : b > 0) : 
  (5 * 6 + 2 = b * b + b + 1) → b = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l2821_282150


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2821_282183

theorem absolute_value_inequality (x : ℝ) : 
  3 ≤ |x - 3| ∧ |x - 3| ≤ 7 ↔ ((-4 ≤ x ∧ x ≤ 0) ∨ (6 ≤ x ∧ x ≤ 10)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2821_282183


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l2821_282190

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel_lines m n → 
  parallel_line_plane n β → 
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l2821_282190


namespace NUMINAMATH_CALUDE_binomial_multiplication_l2821_282127

theorem binomial_multiplication (x : ℝ) : (4*x + 3) * (2*x - 7) = 8*x^2 - 22*x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_multiplication_l2821_282127


namespace NUMINAMATH_CALUDE_fibSum_eq_five_nineteenths_l2821_282135

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of the infinite series ∑(n=0 to ∞) F_n / 5^n -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / 5^n

/-- Theorem stating that the sum of the infinite series equals 5/19 -/
theorem fibSum_eq_five_nineteenths : fibSum = 5 / 19 := by sorry

end NUMINAMATH_CALUDE_fibSum_eq_five_nineteenths_l2821_282135


namespace NUMINAMATH_CALUDE_find_constant_b_l2821_282155

theorem find_constant_b (d e : ℚ) :
  (∀ x : ℚ, (7 * x^2 - 2 * x + 4/3) * (d * x^2 + b * x + e) = 28 * x^4 - 10 * x^3 + 18 * x^2 - 8 * x + 5/3) →
  b = -2/7 := by
sorry

end NUMINAMATH_CALUDE_find_constant_b_l2821_282155


namespace NUMINAMATH_CALUDE_reflection_sum_l2821_282111

/-- Given that the point (-4, 2) is reflected across the line y = mx + b to the point (6, -2),
    prove that m + b = 0 -/
theorem reflection_sum (m b : ℝ) : 
  (∃ (x y : ℝ), y = m * x + b ∧ 
   (x - (-4))^2 + (y - 2)^2 = (x - 6)^2 + (y - (-2))^2 ∧
   (x - (-4)) * (6 - x) + (y - 2) * (-2 - y) = 0) →
  m + b = 0 := by
sorry

end NUMINAMATH_CALUDE_reflection_sum_l2821_282111


namespace NUMINAMATH_CALUDE_min_sum_squares_complex_l2821_282141

theorem min_sum_squares_complex (w : ℂ) (h : Complex.abs (w - (3 - 2*I)) = 4) :
  ∃ (min : ℝ), min = 48 ∧ 
  ∀ (z : ℂ), Complex.abs (z - (3 - 2*I)) = 4 →
    Complex.abs (z + (1 + 2*I))^2 + Complex.abs (z - (7 + 2*I))^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_complex_l2821_282141


namespace NUMINAMATH_CALUDE_greatest_whole_number_inequality_l2821_282107

theorem greatest_whole_number_inequality :
  ∀ x : ℤ, (5 * x - 4 < 3 - 2 * x) → x ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_whole_number_inequality_l2821_282107


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2821_282187

/-- The diameter of a circle with area 50.26548245743669 square meters is 8 meters. -/
theorem circle_diameter_from_area :
  let area : Real := 50.26548245743669
  let diameter : Real := 8
  diameter = 2 * Real.sqrt (area / Real.pi) := by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2821_282187


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2821_282156

/-- The quadratic equation ax^2 - x - 1 = 0 has exactly one solution in the interval (0, 1) if and only if a > 2 -/
theorem quadratic_equation_unique_solution (a : ℝ) : 
  (∃! x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 - x - 1 = 0) ↔ a > 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2821_282156


namespace NUMINAMATH_CALUDE_smallest_y_for_inequality_l2821_282103

theorem smallest_y_for_inequality : ∃ (y : ℕ), y > 0 ∧ (y^6 : ℚ) / (y^3 : ℚ) > 80 ∧ ∀ (z : ℕ), z > 0 → (z^6 : ℚ) / (z^3 : ℚ) > 80 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_for_inequality_l2821_282103


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2821_282193

theorem smallest_solution_of_equation :
  let f : ℝ → ℝ := λ x => 1 / (x - 3) + 1 / (x - 5) - 4 / (x - 4)
  ∃ x : ℝ, f x = 0 ∧ x = 5 - 2 * Real.sqrt 2 ∧ ∀ y : ℝ, f y = 0 → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2821_282193


namespace NUMINAMATH_CALUDE_apples_given_theorem_l2821_282180

/-- Represents the number of apples Joan gave to Melanie -/
def apples_given_to_melanie (initial_apples current_apples : ℕ) : ℕ :=
  initial_apples - current_apples

theorem apples_given_theorem (initial_apples current_apples : ℕ) 
  (h1 : initial_apples = 43)
  (h2 : current_apples = 16) :
  apples_given_to_melanie initial_apples current_apples = 27 := by
  sorry

end NUMINAMATH_CALUDE_apples_given_theorem_l2821_282180


namespace NUMINAMATH_CALUDE_probability_kings_or_aces_l2821_282177

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Number of cards drawn -/
def CardsDrawn : ℕ := 3

/-- Probability of drawing three kings or at least two aces -/
def prob_kings_or_aces : ℚ := 6 / 425

/-- Theorem stating the probability of drawing three kings or at least two aces -/
theorem probability_kings_or_aces :
  (KingsInDeck.choose CardsDrawn) / (StandardDeck.choose CardsDrawn) +
  ((AcesInDeck.choose 2 * (StandardDeck - AcesInDeck).choose 1) +
   AcesInDeck.choose 3) / (StandardDeck.choose CardsDrawn) = prob_kings_or_aces := by
  sorry

end NUMINAMATH_CALUDE_probability_kings_or_aces_l2821_282177


namespace NUMINAMATH_CALUDE_max_volume_triangular_pyramid_l2821_282102

/-- A triangular pyramid with vertex S and base ABC -/
structure TriangularPyramid where
  SA : ℝ
  SB : ℝ
  SC : ℝ
  AB : ℝ
  BC : ℝ
  AC : ℝ

/-- The volume of a triangular pyramid -/
def volume (t : TriangularPyramid) : ℝ := sorry

/-- The conditions given in the problem -/
def satisfiesConditions (t : TriangularPyramid) : Prop :=
  t.SA = 4 ∧
  t.SB ≥ 7 ∧
  t.SC ≥ 9 ∧
  t.AB = 5 ∧
  t.BC ≤ 6 ∧
  t.AC ≤ 8

/-- The theorem stating the maximum volume of the triangular pyramid -/
theorem max_volume_triangular_pyramid :
  ∀ t : TriangularPyramid, satisfiesConditions t → volume t ≤ 8 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_max_volume_triangular_pyramid_l2821_282102


namespace NUMINAMATH_CALUDE_desk_purchase_price_l2821_282173

/-- Given a desk with a selling price that includes a 25% markup and results in a gross profit of $33.33, prove that the purchase price of the desk is $99.99. -/
theorem desk_purchase_price (purchase_price selling_price : ℝ) : 
  selling_price = purchase_price + 0.25 * selling_price →
  selling_price - purchase_price = 33.33 →
  purchase_price = 99.99 := by
sorry

end NUMINAMATH_CALUDE_desk_purchase_price_l2821_282173


namespace NUMINAMATH_CALUDE_cubic_expression_equality_l2821_282134

theorem cubic_expression_equality (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 73/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equality_l2821_282134


namespace NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_91_l2821_282129

theorem greatest_consecutive_integers_sum_91 :
  (∀ n : ℕ, n > 182 → ¬ (∃ a : ℤ, (Finset.range n).sum (λ i => a + i) = 91)) ∧
  (∃ a : ℤ, (Finset.range 182).sum (λ i => a + i) = 91) :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_91_l2821_282129


namespace NUMINAMATH_CALUDE_tangent_problem_l2821_282131

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α/2 + β) = 1/2) 
  (h2 : Real.tan (β - α/2) = 1/3) : 
  Real.tan α = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l2821_282131


namespace NUMINAMATH_CALUDE_sector_area_l2821_282181

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 4) (h2 : θ = π / 3) :
  (1 / 2) * r * θ * r = (8 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2821_282181


namespace NUMINAMATH_CALUDE_tv_weight_difference_l2821_282159

/-- The difference in weight between two TVs with given dimensions -/
theorem tv_weight_difference : 
  let bill_length : ℕ := 48
  let bill_width : ℕ := 100
  let bob_length : ℕ := 70
  let bob_width : ℕ := 60
  let weight_per_sq_inch : ℚ := 4 / 1
  let oz_per_pound : ℕ := 16
  let bill_area : ℕ := bill_length * bill_width
  let bob_area : ℕ := bob_length * bob_width
  let bill_weight_oz : ℚ := bill_area * weight_per_sq_inch
  let bob_weight_oz : ℚ := bob_area * weight_per_sq_inch
  let bill_weight_lbs : ℚ := bill_weight_oz / oz_per_pound
  let bob_weight_lbs : ℚ := bob_weight_oz / oz_per_pound
  bill_weight_lbs - bob_weight_lbs = 150
  := by sorry

end NUMINAMATH_CALUDE_tv_weight_difference_l2821_282159


namespace NUMINAMATH_CALUDE_cameron_tour_theorem_l2821_282139

def cameron_tour_problem (questions_per_tourist : ℕ) (total_tours : ℕ) 
  (group1_size : ℕ) (group2_size : ℕ) (group3_size : ℕ) 
  (inquisitive_factor : ℕ) (total_questions : ℕ) : Prop :=
  let group1_questions := group1_size * questions_per_tourist
  let group2_questions := group2_size * questions_per_tourist
  let group3_questions := group3_size * questions_per_tourist + 
                          (inquisitive_factor - 1) * questions_per_tourist
  let remaining_questions := total_questions - (group1_questions + group2_questions + group3_questions)
  let last_group_size := remaining_questions / questions_per_tourist
  last_group_size = 7

theorem cameron_tour_theorem : 
  cameron_tour_problem 2 4 6 11 8 3 68 := by
  sorry

end NUMINAMATH_CALUDE_cameron_tour_theorem_l2821_282139


namespace NUMINAMATH_CALUDE_steven_peaches_difference_l2821_282196

theorem steven_peaches_difference (jake steven jill : ℕ) 
  (h1 : jake + 5 = steven)
  (h2 : jill = 87)
  (h3 : jake = jill + 13) :
  steven - jill = 18 := by sorry

end NUMINAMATH_CALUDE_steven_peaches_difference_l2821_282196


namespace NUMINAMATH_CALUDE_largest_common_divisor_462_330_l2821_282198

theorem largest_common_divisor_462_330 : Nat.gcd 462 330 = 66 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_462_330_l2821_282198


namespace NUMINAMATH_CALUDE_correct_quotient_calculation_l2821_282182

theorem correct_quotient_calculation (dividend : ℕ) (incorrect_quotient : ℕ) : 
  dividend % 21 = 0 →
  dividend = 12 * incorrect_quotient →
  incorrect_quotient = 56 →
  dividend / 21 = 32 := by
sorry

end NUMINAMATH_CALUDE_correct_quotient_calculation_l2821_282182


namespace NUMINAMATH_CALUDE_smiley_face_tulips_l2821_282133

theorem smiley_face_tulips : 
  let red_tulips_per_eye : ℕ := 8
  let red_tulips_for_smile : ℕ := 18
  let yellow_tulips_multiplier : ℕ := 9
  let number_of_eyes : ℕ := 2

  let total_red_tulips : ℕ := red_tulips_per_eye * number_of_eyes + red_tulips_for_smile
  let total_yellow_tulips : ℕ := yellow_tulips_multiplier * red_tulips_for_smile
  let total_tulips : ℕ := total_red_tulips + total_yellow_tulips

  total_tulips = 196 := by sorry

end NUMINAMATH_CALUDE_smiley_face_tulips_l2821_282133


namespace NUMINAMATH_CALUDE_fraction_decomposition_l2821_282153

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 7) (h2 : x ≠ -2) :
  let f := (2 * x + 4) / (x^2 - 5*x - 14)
  let g := 2 / (x - 7) + 0 / (x + 2)
  (x^2 - 5*x - 14 = (x - 7) * (x + 2)) → f = g :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l2821_282153


namespace NUMINAMATH_CALUDE_reflection_composition_l2821_282122

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ := 
  let p' := (p.1, p.2 - 2)
  let p'' := (p'.2, p'.1)
  (p''.1, p''.2 + 2)

theorem reflection_composition (D : ℝ × ℝ) (h : D = (5, 2)) : 
  reflect_line (reflect_x D) = (-4, 7) := by sorry

end NUMINAMATH_CALUDE_reflection_composition_l2821_282122


namespace NUMINAMATH_CALUDE_only_5_6_10_forms_triangle_l2821_282170

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: Among the given sets, only (5, 6, 10) can form a triangle -/
theorem only_5_6_10_forms_triangle :
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 5 6 11 ∧
  can_form_triangle 5 6 10 ∧
  ¬ can_form_triangle 4 4 8 :=
sorry

end NUMINAMATH_CALUDE_only_5_6_10_forms_triangle_l2821_282170


namespace NUMINAMATH_CALUDE_sum_of_powers_of_three_l2821_282124

theorem sum_of_powers_of_three : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_three_l2821_282124


namespace NUMINAMATH_CALUDE_corner_start_winning_strategy_adjacent_start_winning_strategy_l2821_282166

/-- Represents the players in the game -/
inductive Player
| A
| B

/-- Represents the game state -/
structure GameState where
  n : Nat
  currentPosition : Nat × Nat
  visitedPositions : Set (Nat × Nat)
  currentPlayer : Player

/-- Defines a winning strategy for a player -/
def HasWinningStrategy (player : Player) (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Nat × Nat),
    ∀ (gameState : GameState),
      gameState.currentPlayer = player →
      (strategy gameState ∉ gameState.visitedPositions) →
      (∃ (nextState : GameState), 
        nextState.currentPosition = strategy gameState ∧
        nextState.visitedPositions = insert gameState.currentPosition gameState.visitedPositions ∧
        nextState.currentPlayer ≠ player)

theorem corner_start_winning_strategy :
  ∀ (n : Nat),
    (n % 2 = 0 → HasWinningStrategy Player.A (GameState.mk n (0, 0) {(0, 0)} Player.A)) ∧
    (n % 2 = 1 → HasWinningStrategy Player.B (GameState.mk n (0, 0) {(0, 0)} Player.A)) :=
sorry

theorem adjacent_start_winning_strategy :
  ∀ (n : Nat) (startPos : Nat × Nat),
    (startPos = (0, 1) ∨ startPos = (1, 0)) →
    HasWinningStrategy Player.A (GameState.mk n startPos {startPos} Player.A) :=
sorry

end NUMINAMATH_CALUDE_corner_start_winning_strategy_adjacent_start_winning_strategy_l2821_282166


namespace NUMINAMATH_CALUDE_jack_flyers_l2821_282121

theorem jack_flyers (total : ℕ) (rose : ℕ) (left : ℕ) (h1 : total = 1236) (h2 : rose = 320) (h3 : left = 796) :
  total - (rose + left) = 120 := by
  sorry

end NUMINAMATH_CALUDE_jack_flyers_l2821_282121


namespace NUMINAMATH_CALUDE_kids_bike_wheels_count_l2821_282114

/-- The number of wheels on a kid's bike -/
def kids_bike_wheels : ℕ := 4

/-- The number of regular bikes -/
def regular_bikes : ℕ := 7

/-- The number of children's bikes -/
def children_bikes : ℕ := 11

/-- The number of wheels on a regular bike -/
def regular_bike_wheels : ℕ := 2

/-- The total number of wheels observed -/
def total_wheels : ℕ := 58

theorem kids_bike_wheels_count :
  regular_bikes * regular_bike_wheels + children_bikes * kids_bike_wheels = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_kids_bike_wheels_count_l2821_282114


namespace NUMINAMATH_CALUDE_book_page_difference_l2821_282194

/-- The number of pages in Selena's book -/
def selena_pages : ℕ := 400

/-- The number of pages in Harry's book -/
def harry_pages : ℕ := 180

/-- The difference between half of Selena's pages and Harry's pages -/
def page_difference : ℕ := selena_pages / 2 - harry_pages

theorem book_page_difference : page_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_book_page_difference_l2821_282194


namespace NUMINAMATH_CALUDE_half_perimeter_area_rectangle_existence_l2821_282162

/-- Given a rectangle with sides a and b, this theorem proves the existence of another rectangle
    with half the perimeter and half the area, based on the discriminant of the resulting quadratic equation. -/
theorem half_perimeter_area_rectangle_existence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = (a + b) / 2 ∧ x * y = (a * b) / 2 ↔
  ((a + b)^2 - 4 * (a * b)) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_half_perimeter_area_rectangle_existence_l2821_282162


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2821_282197

theorem cube_root_equation_solution (x p : ℝ) : 
  (Real.rpow (1 - x) (1/3 : ℝ)) + (Real.rpow (1 + x) (1/3 : ℝ)) = p → 
  (x = 0 ∧ p = -1) → True :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2821_282197


namespace NUMINAMATH_CALUDE_cube_symmetry_properties_change_l2821_282163

/-- Represents the symmetrical properties of a geometric object -/
structure SymmetryProperties where
  planes : ℕ
  axes : ℕ
  center : Bool

/-- Represents the different painting configurations of a cube -/
inductive CubePainting
  | Unpainted
  | OneFace
  | TwoFacesParallel
  | TwoFacesAdjacent
  | ThreeFacesMeetingAtVertex
  | ThreeFacesNotMeetingAtVertex

/-- Returns the symmetry properties for a given cube painting configuration -/
def symmetryPropertiesForCube (painting : CubePainting) : SymmetryProperties :=
  match painting with
  | .Unpainted => { planes := 9, axes := 9, center := true }
  | .OneFace => { planes := 4, axes := 1, center := false }
  | .TwoFacesParallel => { planes := 5, axes := 3, center := true }
  | .TwoFacesAdjacent => { planes := 2, axes := 1, center := false }
  | .ThreeFacesMeetingAtVertex => { planes := 3, axes := 0, center := false }
  | .ThreeFacesNotMeetingAtVertex => { planes := 2, axes := 1, center := false }

theorem cube_symmetry_properties_change (painting : CubePainting) :
  symmetryPropertiesForCube painting ≠ symmetryPropertiesForCube CubePainting.Unpainted :=
by sorry

end NUMINAMATH_CALUDE_cube_symmetry_properties_change_l2821_282163


namespace NUMINAMATH_CALUDE_square_roots_equality_l2821_282100

theorem square_roots_equality (m : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (2*m - 4)^2 = k ∧ (3*m - 1)^2 = k) → (m = -3 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_square_roots_equality_l2821_282100


namespace NUMINAMATH_CALUDE_transmission_time_is_128_seconds_l2821_282136

/-- The number of blocks to be sent -/
def num_blocks : ℕ := 80

/-- The number of chunks in each block -/
def chunks_per_block : ℕ := 256

/-- The transmission rate in chunks per second -/
def transmission_rate : ℕ := 160

/-- The time it takes to send all blocks in seconds -/
def transmission_time : ℕ := num_blocks * chunks_per_block / transmission_rate

theorem transmission_time_is_128_seconds : transmission_time = 128 := by
  sorry

end NUMINAMATH_CALUDE_transmission_time_is_128_seconds_l2821_282136


namespace NUMINAMATH_CALUDE_total_pencils_l2821_282112

/-- Given the number of pencils in different locations, prove the total number of pencils. -/
theorem total_pencils (drawer : ℕ) (desk_initial : ℕ) (desk_added : ℕ) :
  drawer = 43 →
  desk_initial = 19 →
  desk_added = 16 →
  drawer + desk_initial + desk_added = 78 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l2821_282112


namespace NUMINAMATH_CALUDE_value_of_expression_l2821_282104

theorem value_of_expression (m n : ℤ) (h : m - n = 1) : (m - n)^2 - 2*m + 2*n = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2821_282104


namespace NUMINAMATH_CALUDE_jellybean_probability_l2821_282171

/-- The probability of selecting exactly 2 red and 1 green jellybean when picking 4 jellybeans randomly without replacement from a bowl containing 5 red, 3 blue, 2 green, and 5 white jellybeans (15 total) -/
theorem jellybean_probability : 
  let total := 15
  let red := 5
  let blue := 3
  let green := 2
  let white := 5
  let pick := 4
  Nat.choose total pick ≠ 0 →
  (Nat.choose red 2 * Nat.choose green 1 * Nat.choose (blue + white) 1) / Nat.choose total pick = 32 / 273 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l2821_282171


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l2821_282161

/-- An isosceles triangle with specific measurements -/
structure IsoscelesTriangle where
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The inradius of the triangle -/
  inradius : ℝ
  /-- One of the angles of the triangle in degrees -/
  angle : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : Bool
  /-- The perimeter is 20 cm -/
  perimeterIs20 : perimeter = 20
  /-- The inradius is 2.5 cm -/
  inradiusIs2_5 : inradius = 2.5
  /-- One angle is 40 degrees -/
  angleIs40 : angle = 40
  /-- The triangle is confirmed to be isosceles -/
  isIsoscelesTrue : isIsosceles = true

/-- The area of the isosceles triangle is 25 cm² -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) : 
  t.inradius * (t.perimeter / 2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l2821_282161


namespace NUMINAMATH_CALUDE_length_ad_is_12_95_l2821_282140

/-- A quadrilateral ABCD with specific properties -/
structure Quadrilateral where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Length of side CD -/
  cd : ℝ
  /-- Angle B in radians -/
  angle_b : ℝ
  /-- Angle C in radians -/
  angle_c : ℝ
  /-- Condition: AB = 6 -/
  hab : ab = 6
  /-- Condition: BC = 8 -/
  hbc : bc = 8
  /-- Condition: CD = 15 -/
  hcd : cd = 15
  /-- Condition: Angle B is obtuse -/
  hb_obtuse : π / 2 < angle_b ∧ angle_b < π
  /-- Condition: Angle C is obtuse -/
  hc_obtuse : π / 2 < angle_c ∧ angle_c < π
  /-- Condition: sin C = 4/5 -/
  hsin_c : Real.sin angle_c = 4/5
  /-- Condition: cos B = -4/5 -/
  hcos_b : Real.cos angle_b = -4/5

/-- The length of side AD in the quadrilateral ABCD -/
def lengthAD (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that the length of side AD is 12.95 -/
theorem length_ad_is_12_95 (q : Quadrilateral) : lengthAD q = 12.95 := by
  sorry

end NUMINAMATH_CALUDE_length_ad_is_12_95_l2821_282140


namespace NUMINAMATH_CALUDE_unknown_number_problem_l2821_282130

theorem unknown_number_problem (x : ℚ) : 
  x + (2/3) * x - (1/3) * (x + (2/3) * x) = 10 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_problem_l2821_282130


namespace NUMINAMATH_CALUDE_sum_mod_nine_l2821_282184

theorem sum_mod_nine : (2469 + 2470 + 2471 + 2472 + 2473 + 2474) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l2821_282184


namespace NUMINAMATH_CALUDE_solve_abc_l2821_282179

def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

theorem solve_abc (a b c : ℝ) :
  A a ≠ B b c ∧
  A a ∩ B b c = {-3} ∧
  A a ∪ B b c = {-3, 1, 4} →
  a = -1 ∧ b = 2 ∧ c = -3 := by
sorry

end NUMINAMATH_CALUDE_solve_abc_l2821_282179


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2821_282189

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence (a₁ a₂ : ℝ) (h : a₂ = a₁ + 4) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 119 :=
by sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2821_282189


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_one_l2821_282108

/-- Two lines in the form ax - y + b = 0 and cx - y + d = 0 are parallel if and only if a = c -/
def are_parallel (a c : ℝ) : Prop := a = c

/-- The condition for the given lines to be parallel -/
def parallel_condition (a : ℝ) : Prop := are_parallel a (1/a)

theorem parallel_iff_a_eq_one (a : ℝ) : 
  parallel_condition a ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_one_l2821_282108


namespace NUMINAMATH_CALUDE_election_invalid_votes_l2821_282164

theorem election_invalid_votes 
  (total_polled : ℕ) 
  (losing_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h_total : total_polled = 90083)
  (h_losing : losing_percentage = 45 / 100)
  (h_difference : vote_difference = 9000) :
  total_polled - (vote_difference / (1/2 - losing_percentage)) = 83 := by
sorry

end NUMINAMATH_CALUDE_election_invalid_votes_l2821_282164


namespace NUMINAMATH_CALUDE_sqrt_ab_is_integer_l2821_282110

theorem sqrt_ab_is_integer (a b n : ℕ+) 
  (h : (a : ℚ) / b = ((a : ℚ)^2 + (n : ℚ)^2) / ((b : ℚ)^2 + (n : ℚ)^2)) : 
  ∃ k : ℕ, k^2 = a * b := by
sorry

end NUMINAMATH_CALUDE_sqrt_ab_is_integer_l2821_282110


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l2821_282109

theorem complex_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z ^ 2 = 3 - 5*I) : 
  Complex.abs z ^ 2 = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l2821_282109


namespace NUMINAMATH_CALUDE_range_of_A_l2821_282113

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_A : ∀ a : ℝ, a ∈ A ↔ -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_A_l2821_282113


namespace NUMINAMATH_CALUDE_forgot_lawns_count_l2821_282165

def lawn_problem (total_lawns : ℕ) (earnings_per_lawn : ℕ) (actual_earnings : ℕ) : ℕ :=
  total_lawns - (actual_earnings / earnings_per_lawn)

theorem forgot_lawns_count :
  lawn_problem 17 4 32 = 9 := by
  sorry

end NUMINAMATH_CALUDE_forgot_lawns_count_l2821_282165


namespace NUMINAMATH_CALUDE_binomial_rv_p_value_l2821_282142

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean : ℝ
  std_dev : ℝ

/-- Theorem: For a binomial random variable with mean 200 and standard deviation 10, p = 1/2 -/
theorem binomial_rv_p_value (X : BinomialRV) 
  (h_mean : X.mean = 200)
  (h_std_dev : X.std_dev = 10) :
  X.p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_rv_p_value_l2821_282142


namespace NUMINAMATH_CALUDE_manicure_total_cost_l2821_282152

theorem manicure_total_cost (manicure_cost : ℝ) (tip_percentage : ℝ) : 
  manicure_cost = 30 →
  tip_percentage = 30 →
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := by
  sorry

end NUMINAMATH_CALUDE_manicure_total_cost_l2821_282152


namespace NUMINAMATH_CALUDE_grunters_win_probability_l2821_282175

/-- The probability of the Grunters winning a single game -/
def p : ℚ := 3/5

/-- The number of games played -/
def n : ℕ := 5

/-- The probability of winning all games -/
def win_all : ℚ := p^n

theorem grunters_win_probability : win_all = 243/3125 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l2821_282175


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2821_282132

/-- 
Given an arithmetic sequence {a_n} with first term a₁ = 19 and integer common difference d,
if the 6th term is negative and the 5th term is non-negative, then the common difference is -4.
-/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h1 : a 1 = 19)
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : a 6 < 0)
  (h4 : a 5 ≥ 0) :
  d = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2821_282132


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l2821_282115

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l2821_282115


namespace NUMINAMATH_CALUDE_combined_research_degrees_l2821_282148

def total_percentage : ℝ := 100
def microphotonics_percentage : ℝ := 10
def home_electronics_percentage : ℝ := 24
def food_additives_percentage : ℝ := 15
def genetically_modified_microorganisms_percentage : ℝ := 29
def industrial_lubricants_percentage : ℝ := 8
def nanotechnology_percentage : ℝ := 7

def basic_astrophysics_percentage : ℝ :=
  total_percentage - (microphotonics_percentage + home_electronics_percentage + 
  food_additives_percentage + genetically_modified_microorganisms_percentage + 
  industrial_lubricants_percentage + nanotechnology_percentage)

def combined_percentage : ℝ := basic_astrophysics_percentage + nanotechnology_percentage

def degrees_in_circle : ℝ := 360

theorem combined_research_degrees :
  combined_percentage * (degrees_in_circle / total_percentage) = 50.4 := by
  sorry

end NUMINAMATH_CALUDE_combined_research_degrees_l2821_282148


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2821_282174

/-- A quadratic function that takes specific values for consecutive natural numbers. -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = -9 ∧ f (n + 1) = -9 ∧ f (n + 2) = -15

/-- The maximum value of a quadratic function with the given properties. -/
theorem quadratic_max_value (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∧ f x = -33/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2821_282174


namespace NUMINAMATH_CALUDE_solution_set_linear_inequalities_l2821_282178

theorem solution_set_linear_inequalities :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_linear_inequalities_l2821_282178


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2821_282158

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 18 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 18 → (x : ℤ) + y ≤ (a : ℤ) + b) →
  (x : ℤ) + y = 75 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2821_282158


namespace NUMINAMATH_CALUDE_consecutive_non_multiple_of_five_product_l2821_282160

theorem consecutive_non_multiple_of_five_product (k : ℤ) :
  (∃ m : ℤ, (5*k + 1) * (5*k + 2) * (5*k + 3) = 5*m + 1) ∨
  (∃ n : ℤ, (5*k + 2) * (5*k + 3) * (5*k + 4) = 5*n - 1) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_non_multiple_of_five_product_l2821_282160


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l2821_282128

/-- Given a large rectangle of dimensions A × B containing a smaller rectangle of dimensions a × b,
    the difference between the total area of yellow regions and green regions is A × b - a × B. -/
theorem rectangle_area_difference (A B a b : ℝ) (hA : A > 0) (hB : B > 0) (ha : a > 0) (hb : b > 0)
  (ha_le_A : a ≤ A) (hb_le_B : b ≤ B) :
  A * b - a * B = A * b - a * B := by sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l2821_282128


namespace NUMINAMATH_CALUDE_eggs_produced_this_year_l2821_282120

/-- Calculates the total egg production for this year given last year's production and additional eggs produced. -/
def total_eggs_this_year (last_year_production additional_eggs : ℕ) : ℕ :=
  last_year_production + additional_eggs

/-- Theorem stating that the total eggs produced this year is 4636. -/
theorem eggs_produced_this_year : 
  total_eggs_this_year 1416 3220 = 4636 := by
  sorry

end NUMINAMATH_CALUDE_eggs_produced_this_year_l2821_282120


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l2821_282145

/-- Converts a speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 1.02

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 3672 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l2821_282145


namespace NUMINAMATH_CALUDE_sequence_value_l2821_282192

/-- Given a sequence {aₙ} where a₁ = 3 and 2aₙ₊₁ - 2aₙ = 1 for all n ≥ 1,
    prove that a₉₉ = 52. -/
theorem sequence_value (a : ℕ → ℝ) (h1 : a 1 = 3) 
    (h2 : ∀ n : ℕ, 2 * a (n + 1) - 2 * a n = 1) : 
  a 99 = 52 := by
  sorry

end NUMINAMATH_CALUDE_sequence_value_l2821_282192


namespace NUMINAMATH_CALUDE_orange_distribution_l2821_282186

theorem orange_distribution (x : ℚ) : 
  (x/2 + 1/2) + (1/2 * (x/2 - 1/2) + 1/2) + (1/2 * (x/4 - 3/4) + 1/2) = x → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l2821_282186
