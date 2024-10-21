import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_bound_l163_16337

/-- The ellipse C: x²/4 + y²/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The right focus of the ellipse -/
def right_focus : ℝ × ℝ := (1, 0)

/-- A line with slope k passing through the right focus -/
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

/-- The ratio |DP|/|AB| for given k -/
noncomputable def ratio (k : ℝ) : ℝ := (1/4) * Real.sqrt (1 - 1/(k^2 + 1))

theorem ellipse_ratio_bound (k : ℝ) (hk : k ≠ 0) :
  0 < ratio k ∧ ratio k < 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_bound_l163_16337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l163_16358

theorem triangle_sine_inequality (A B C : ℝ) : 
  A + B + C = π → 0 < A → 0 < B → 0 < C → A ≤ B → B ≤ C → 
  0 < Real.sin A + Real.sin B - Real.sin C ∧ Real.sin A + Real.sin B - Real.sin C ≤ Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_inequality_l163_16358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l163_16300

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + a = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 4*x + a + 5 = 0}

-- Define the condition that exactly one of A and B is empty
def exactly_one_empty (a : ℝ) : Prop :=
  (A a = ∅ ∧ B a ≠ ∅) ∨ (A a ≠ ∅ ∧ B a = ∅)

-- State the theorem
theorem range_of_a (a : ℝ) :
  exactly_one_empty a → a ∈ Set.Ioc (-1) 0 ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l163_16300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l163_16354

/-- Represents a rectangle in the coordinate plane --/
structure Rectangle where
  width : ℝ
  height : ℝ
  lower_left : ℝ × ℝ

/-- Represents a line in the coordinate plane --/
structure Line where
  start : ℝ × ℝ
  end_ : ℝ × ℝ

/-- Calculates the area of a triangle given its base and height --/
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (1/2) * base * height

/-- Calculates the area of a rectangle --/
noncomputable def rectangle_area (rect : Rectangle) : ℝ :=
  rect.width * rect.height

/-- Checks if a line divides a rectangle into two equal areas --/
def divides_equally (rect : Rectangle) (line : Line) : Prop :=
  let total_area := rectangle_area rect
  let triangle_area := triangle_area (line.end_.1 - line.start.1) (line.end_.2 - line.start.2)
  2 * triangle_area = total_area

/-- The main theorem to be proved --/
theorem equal_area_division (c : ℝ) : 
  let rect : Rectangle := { width := 3, height := 2, lower_left := (0, 0) }
  let line : Line := { start := (c, 0), end_ := (4, 4) }
  divides_equally rect line ↔ c = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l163_16354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_physical_exercise_preference_l163_16318

/-- Represents the contingency table for student preferences on physical exercise --/
structure ContingencyTable where
  male_like : ℕ
  female_like : ℕ
  male_dislike : ℕ
  female_dislike : ℕ

/-- Calculates the chi-square statistic --/
noncomputable def chi_square (ct : ContingencyTable) : ℝ :=
  let n := (ct.male_like + ct.female_like + ct.male_dislike + ct.female_dislike : ℝ)
  let ad := (ct.male_like * ct.female_dislike : ℝ)
  let bc := (ct.female_like * ct.male_dislike : ℝ)
  n * (ad - bc)^2 / ((ct.male_like + ct.female_like) * (ct.male_dislike + ct.female_dislike) *
    (ct.male_like + ct.male_dislike) * (ct.female_like + ct.female_dislike))

theorem physical_exercise_preference (p q : ℕ) 
  (h1 : (280 + q : ℝ) / (400 + p + q) = 4/7)
  (h2 : (p : ℝ) / (p + 120) = 3/5) :
  p = 180 ∧ q = 120 ∧ 
  chi_square {male_like := 280, female_like := 180, male_dislike := 120, female_dislike := 120} < 10.828 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_physical_exercise_preference_l163_16318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_exit_path_exists_l163_16326

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a half-plane forest -/
structure HalfPlaneForest where
  boundary : Set Point

/-- Represents a path in the forest -/
inductive ForestPath
  | Empty : ForestPath
  | Cons : Point → ForestPath → ForestPath

/-- Function to calculate the length of a path -/
def pathLength : ForestPath → ℝ
  | ForestPath.Empty => 0
  | ForestPath.Cons p rest => sorry

/-- Function to check if a path exits the forest -/
def exitsForest (f : HalfPlaneForest) : ForestPath → Prop
  | ForestPath.Empty => false
  | ForestPath.Cons p rest => sorry

/-- Theorem: There exists a path of length (1+√3+7π/6)d that guarantees exit from the forest -/
theorem shorter_exit_path_exists (f : HalfPlaneForest) (d : ℝ) (start : Point) 
    (h : ∃ (b : Point), b ∈ f.boundary ∧ (start.x - b.x)^2 + (start.y - b.y)^2 = d^2) :
    ∃ (p : ForestPath), 
      pathLength p = (1 + Real.sqrt 3 + 7 * Real.pi / 6) * d ∧ 
      exitsForest f p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_exit_path_exists_l163_16326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_answer_is_c_l163_16379

/-- The function we're maximizing -/
noncomputable def f (t : ℝ) : ℝ := (3^t - 4*t) * t / 9^t

/-- The theorem stating the maximum value of the function -/
theorem f_max_value :
  (∀ t : ℝ, f t ≤ 1/16) ∧ (∃ t : ℝ, f t = 1/16) := by
  sorry

/-- The theorem stating that the answer choice (C) is correct -/
theorem answer_is_c :
  ∃ (t : ℝ), f t = 1/16 ∧ ∀ (s : ℝ), f s ≤ f t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_answer_is_c_l163_16379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fewer_heads_probability_l163_16338

/-- The probability of getting heads for a fair coin -/
noncomputable def fairCoinProbability : ℝ := 1/2

/-- The probability of getting heads for the biased coin -/
noncomputable def biasedCoinProbability : ℝ := 3/4

/-- The total number of coins -/
def totalCoins : ℕ := 12

/-- The number of fair coins -/
def fairCoins : ℕ := 11

/-- The number of biased coins -/
def biasedCoins : ℕ := 1

/-- The probability of getting fewer heads than tails when flipping the coins -/
noncomputable def probabilityFewerHeads : ℝ := 3172/8192

theorem fewer_heads_probability :
  probabilityFewerHeads = 
    (1 - (Nat.choose fairCoins 6 * (fairCoinProbability ^ 6 * (1 - fairCoinProbability) ^ 5) * biasedCoinProbability +
          Nat.choose fairCoins 5 * (fairCoinProbability ^ 5 * (1 - fairCoinProbability) ^ 6) * (1 - biasedCoinProbability))) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fewer_heads_probability_l163_16338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_product_range_l163_16310

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if 0 < x ∧ x ≤ 9 then |Real.log x / Real.log 3 - 1|
  else if x > 9 then 4 - Real.sqrt x
  else 0  -- This case should never occur in our problem

-- State the theorem
theorem abc_product_range (a b c : ℝ) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  0 < a ∧ 0 < b ∧ 0 < c →
  f a = f b ∧ f b = f c →
  81 < a * b * c ∧ a * b * c < 144 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_product_range_l163_16310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l163_16376

-- Define the function f(x) = a^x + x - b
noncomputable def f (a b x : ℝ) : ℝ := a^x + x - b

-- State the theorem
theorem zero_point_interval (a b : ℝ) (ha : 1 < a) (hb : 0 < b) (hb1 : b < 1) :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo (-1 : ℝ) 0 ∧ f a b x₀ = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_interval_l163_16376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_of_A_l163_16389

-- Define the set A
def A (x y : ℝ) : Set ℝ := {Real.log x, Real.log y, Real.log (x + y/x)}

-- Define the theorem
theorem max_element_of_A (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : ∃ (a : ℝ), a ∈ A x y ∧ a = 0) 
  (h4 : ∃ (b : ℝ), b ∈ A x y ∧ b = 1) :
  ∃ (m : ℝ), m ∈ A x y ∧ ∀ (z : ℝ), z ∈ A x y → z ≤ m ∧ m = Real.log 11 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_of_A_l163_16389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_scenario_l163_16332

open Set

noncomputable section

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := (x + a) * (x^2 + b*x + c)
def g (x : ℝ) : ℝ := (a*x + 1) * (c*x^2 + b*x + 1)

def S (a b c : ℝ) : Set ℝ := {x : ℝ | f a b c x = 0}
def T (a b c : ℝ) : Set ℝ := {x : ℝ | g a b c x = 0}

theorem impossible_scenario : ∀ a b c : ℝ, ¬(Finite (S a b c) ∧ Finite (T a b c) ∧ (Nat.card (S a b c) = 2 ∧ Nat.card (T a b c) = 3)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_scenario_l163_16332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebooks_theorem_l163_16378

/-- The maximum number of notebooks that can be obtained given the conditions --/
def max_notebooks (total_money : ℕ) (notebook_cost : ℕ) (stickers_for_notebook : ℕ) : ℕ :=
  let initial_notebooks := total_money / notebook_cost
  let initial_stickers := initial_notebooks
  let rec additional_notebooks (notebooks : ℕ) (stickers : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then notebooks
    else if stickers ≥ stickers_for_notebook then
      additional_notebooks (notebooks + 1) (stickers - stickers_for_notebook + 1) (fuel - 1)
    else
      notebooks
  initial_notebooks + additional_notebooks initial_notebooks initial_stickers initial_notebooks

/-- Theorem stating that given the conditions, the maximum number of notebooks is 46 --/
theorem notebooks_theorem :
  max_notebooks 150 4 5 = 46 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebooks_theorem_l163_16378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_and_tangent_l163_16370

/-- Given line l -/
def l (x y : ℝ) : Prop := y = 2 * x + 3

/-- Given circle C -/
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

/-- The tangent line t -/
noncomputable def t (x y : ℝ) : Prop := 2*x - y = Real.sqrt 5 ∨ 2*x - y = -Real.sqrt 5

theorem tangent_line_parallel_and_tangent :
  (∀ x y, l x y → ∃ k, t x (y + k)) ∧
  (∃ x y, t x y ∧ C x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_and_tangent_l163_16370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dentist_work_hours_l163_16325

/-- Calculates the number of hours a dentist works each day given the following conditions:
  * 2 toothbrushes are given to every patient
  * Each visit takes 0.5 hours
  * It's a 5-day work week
  * 160 toothbrushes are given away in a week
-/
theorem dentist_work_hours 
  (toothbrushes_per_patient : ℕ) 
  (visit_duration : ℚ) 
  (workdays_per_week : ℕ) 
  (total_toothbrushes : ℕ) 
  (hours_per_day : ℚ)
  (h1 : toothbrushes_per_patient = 2)
  (h2 : visit_duration = 1/2)
  (h3 : workdays_per_week = 5)
  (h4 : total_toothbrushes = 160)
  : hours_per_day = 8 := by
  sorry

#check dentist_work_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dentist_work_hours_l163_16325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l163_16384

noncomputable def sequence_a (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => (5 * sequence_a a n - 8) / (sequence_a a n - 1)

noncomputable def transformed_sequence (a : ℝ) (n : ℕ) : ℝ :=
  (sequence_a a n - 2) / (sequence_a a n - 4)

theorem sequence_properties (a : ℝ) :
  (a = 3 → ∀ n : ℕ, n > 0 → transformed_sequence a (n + 1) = -3 * transformed_sequence a n) ∧
  ((∀ n : ℕ, sequence_a a n > 3) → a > 3) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l163_16384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_cost_l163_16387

/-- The cost of each candy bar given the conditions of Benny's purchase -/
theorem candy_bar_cost 
  (num_soft_drinks : ℕ) 
  (cost_per_soft_drink : ℚ) 
  (num_candy_bars : ℕ) 
  (total_spent : ℚ) 
  (h1 : num_soft_drinks = 2)
  (h2 : cost_per_soft_drink = 4)
  (h3 : num_candy_bars = 5)
  (h4 : total_spent = 28)
  : (total_spent - num_soft_drinks * cost_per_soft_drink) / num_candy_bars = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_cost_l163_16387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l163_16315

noncomputable def arithmetic_sequence (d a₁ : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def arithmetic_sum (d a₁ : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_geometric_ratio 
  (d a₁ : ℝ) (h_d : d ≠ 0) :
  let a := arithmetic_sequence d a₁
  let S := arithmetic_sum d a₁
  (a 1 * a 4 = a 3 * a 3) →
  (S 3 - S 2) / (S 5 - S 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l163_16315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curve_line_and_axis_l163_16394

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := (x + Real.sqrt (x^2 + 1))^(1/3) + (x - Real.sqrt (x^2 + 1))^(1/3)

/-- The line function -/
def g (x : ℝ) : ℝ := x - 1

/-- The intersection point of the curve and the line -/
def intersection_point : ℝ × ℝ := (2, 1)

/-- The bounded area -/
def bounded_area : ℝ := 5/8

theorem area_between_curve_line_and_axis :
  (∫ (x : ℝ) in Set.Icc 0 1, g x - f x) = bounded_area := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curve_line_and_axis_l163_16394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_values_l163_16312

theorem quadratic_function_values (b c m x₁ x₂ : ℝ) :
  (∀ x, x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  m < x₁ →
  x₁ < x₂ →
  x₂ < m + 1 →
  (m^2 + b*m + c < 1/4) ∧ ((m+1)^2 + b*(m+1) + c < 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_values_l163_16312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_change_theorem_l163_16353

def has_unique_zero_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

noncomputable def sign (x : ℝ) : ℝ :=
  if x < 0 then -1 else if x > 0 then 1 else 0

theorem sign_change_theorem (f : ℝ → ℝ) 
  (h_cont : Continuous f)
  (h_zero1 : has_unique_zero_in f 0 4)
  (h_zero2 : has_unique_zero_in f 0 2)
  (h_zero3 : has_unique_zero_in f 1 (3/2))
  (h_zero4 : has_unique_zero_in f (5/4) (3/2)) :
  (sign (f 0) ≠ sign (f 4)) ∧
  (sign (f 0) ≠ sign (f 2)) ∧
  (sign (f 0) = sign (f 1)) ∧
  (sign (f 0) ≠ sign (f (3/2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_change_theorem_l163_16353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_mpg_l163_16365

-- Define the variables
noncomputable def highway_miles_per_tank : ℝ := 462
noncomputable def city_miles_per_tank : ℝ := 336
noncomputable def city_mpg_difference : ℝ := 3

-- Define the functions
noncomputable def highway_mpg (tank_size : ℝ) : ℝ := highway_miles_per_tank / tank_size
noncomputable def city_mpg (tank_size : ℝ) : ℝ := city_miles_per_tank / tank_size

-- State the theorem
theorem car_city_mpg :
  ∃ (tank_size : ℝ), 
    tank_size > 0 ∧ 
    city_mpg tank_size = highway_mpg tank_size - city_mpg_difference ∧
    city_mpg tank_size = 8 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_mpg_l163_16365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_range_l163_16327

-- Define the arithmetic sequence a_n
noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ := a + n - 1

-- Define b_n in terms of a_n
noncomputable def b_n (a : ℝ) (n : ℕ) : ℝ := (1 + a_n a n) / (a_n a n)

-- State the theorem
theorem arithmetic_sequence_range (a : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → b_n a n ≥ b_n a 8) → 
  -8 < a ∧ a < -7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_range_l163_16327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l163_16390

/-- A function that checks if a number has three different digits in increasing order -/
def has_increasing_digits (n : ℕ) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 < d2 && d2 < d3

/-- The set of integers between 200 and 250 with three different digits in increasing order -/
def valid_numbers : Finset ℕ :=
  Finset.filter (λ n => 200 ≤ n && n ≤ 250 && has_increasing_digits n) (Finset.range 251)

theorem count_valid_numbers : Finset.card valid_numbers = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l163_16390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_fraction_sum_finite_special_fraction_sum_count_l163_16303

def is_special (a b : ℕ+) : Prop := a.val + b.val = 15

def special_fraction_sum : Set ℕ := {n | ∃ (a₁ b₁ a₂ b₂ : ℕ+), 
  is_special a₁ b₁ ∧ is_special a₂ b₂ ∧ n = (a₁.val * b₂.val + a₂.val * b₁.val) / (b₁.val * b₂.val)}

-- We need to prove that special_fraction_sum is finite before we can use Fintype.card
theorem special_fraction_sum_finite : Finite special_fraction_sum := by sorry

-- Now we can state our theorem using Fintype.card
theorem special_fraction_sum_count : Fintype.card (Set.Finite.toFinset special_fraction_sum_finite) = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_fraction_sum_finite_special_fraction_sum_count_l163_16303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_four_l163_16349

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 9

-- Define a line passing through (1,1)
def line_through_point (m : ℝ) (x y : ℝ) : Prop := y - 1 = m * (x - 1)

-- Define the intersection points A and B
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ my_circle x y ∧ line_through_point m x y}

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem min_distance_is_four :
  ∀ m : ℝ, ∀ A B : ℝ × ℝ, A ∈ intersection_points m → B ∈ intersection_points m →
    ∃ (A' B' : ℝ × ℝ), A' ∈ intersection_points m ∧ B' ∈ intersection_points m ∧
      distance A' B' ≤ distance A B ∧ distance A' B' = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_four_l163_16349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l163_16336

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := (1/2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 1

-- Define the set of x values where f reaches its maximum
noncomputable def max_set : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6}

-- Define the transformed function
noncomputable def g (x : ℝ) : ℝ := (1/2) * Real.sin (2*x + Real.pi/6) + 5/4

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f x ≤ f (Real.pi / 6)) ∧
  (∀ x : ℝ, x ∈ max_set ↔ f x = f (Real.pi / 6)) ∧
  (∀ x : ℝ, f x = g x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l163_16336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l163_16399

theorem tan_value_from_sin_cos_sum (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin α + Real.cos α = -7/13) : Real.tan α = -5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l163_16399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_remainder_theorem_application_l163_16313

theorem chinese_remainder_theorem_application :
  let S : Finset ℕ := Finset.filter (fun n => 1 ≤ n ∧ n ≤ 2016 ∧ n % 3 = 1 ∧ n % 5 = 1) (Finset.range 2017)
  Finset.card S = 135 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_remainder_theorem_application_l163_16313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagonal_roof_cost_l163_16335

/-- The cost to cover an octagonal roof with shingles -/
noncomputable def shingle_cost (num_triangles : ℕ) (triangle_base : ℝ) (triangle_height : ℝ) 
                 (shingle_size : ℝ) (shingle_cost : ℝ) : ℝ :=
  let triangle_area := 0.5 * triangle_base * triangle_height
  let total_area := num_triangles * triangle_area
  let sections_needed := Int.ceil (total_area / (shingle_size * shingle_size))
  (sections_needed : ℝ) * shingle_cost

/-- Theorem stating the cost to cover the specific octagonal roof -/
theorem octagonal_roof_cost : 
  shingle_cost 8 3 5 10 35 = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagonal_roof_cost_l163_16335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l163_16398

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 → f y < f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_decreasing_l163_16398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_consumption_approx_3_1_gallons_l163_16311

/-- Calculates the total water consumption of a traveler and a camel in gallons -/
noncomputable def total_water_consumption (traveler_weight : ℝ) (traveler_percent : ℝ) 
  (camel_weight : ℝ) (camel_percent : ℝ) (ounces_per_gallon : ℝ) : ℝ :=
  let traveler_water := traveler_weight * traveler_percent / 100 * 16
  let camel_water := camel_weight * camel_percent / 100 * 16
  (traveler_water + camel_water) / ounces_per_gallon

/-- Theorem stating that the total water consumption is approximately 3.1 gallons -/
theorem water_consumption_approx_3_1_gallons : 
  ∃ ε > 0, |total_water_consumption 160 0.5 1200 2 128 - 3.1| < ε := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_consumption_approx_3_1_gallons_l163_16311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_couples_remain_l163_16345

/-- The number of couples and points on the circumference. -/
def n : ℕ := 2018

/-- The total number of minutes the dance lasts. -/
def total_minutes : ℕ := n ^ 2

/-- The function that determines the starting point for a couple at minute i. -/
def s (i : ℕ) : Fin n := ⟨i % n, by
  apply Nat.mod_lt
  exact Nat.zero_lt_succ _⟩

/-- The function that determines the ending point for a couple at minute i. -/
def r (i : ℕ) : Fin n := ⟨(2 * i) % n, by
  apply Nat.mod_lt
  exact Nat.zero_lt_succ _⟩

/-- Represents the state of the dance at any given minute. -/
def DanceState := Fin n → Bool

/-- The initial state of the dance, with all points occupied. -/
def initial_state : DanceState := fun _ => true

/-- Updates the dance state for a single minute. -/
def update_state (state : DanceState) (i : ℕ) : DanceState :=
  fun p =>
    if p = r i then
      if s i ≠ r i then false  -- Couple drops out if s i ≠ r i
      else state p  -- Couple stays if s i = r i
    else if p = s i then state (r i)
    else state p

/-- The final state of the dance after all minutes. -/
def final_state : DanceState :=
  (List.range total_minutes).foldl update_state initial_state

/-- The theorem stating that no couples remain at the end of the dance. -/
theorem no_couples_remain : ∀ p, ¬(final_state p) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_couples_remain_l163_16345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_even_integer_l163_16377

theorem largest_consecutive_even_integer (n : ℕ) (sum : ℕ) (largest : ℕ) : 
  n = 30 →
  sum = 12000 →
  (∃ (start : ℤ), sum = (start + start + 2 * (n - 1)) * n / 2 ∧ 
    start % 2 = 0 ∧ 
    largest = (start + 2 * (n - 1)).toNat) →
  largest = 429 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_even_integer_l163_16377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_probability_l163_16355

/-- Represents a cube with 8 vertices -/
structure Cube :=
  (vertices : Fin 8)

/-- Represents a path on the cube -/
def CubePath (cube : Cube) := List (Fin 8)

/-- A function that checks if a path is valid according to the cube's structure -/
def isValidPath (cube : Cube) (path : CubePath cube) : Prop := sorry

/-- A function that checks if a path is a Hamiltonian cycle -/
def isHamiltonianCycle (cube : Cube) (path : CubePath cube) : Prop := sorry

/-- The probability of choosing a specific edge at each move -/
def edgeProbability : ℚ := 1 / 3

/-- The total number of possible paths after 8 moves -/
def totalPaths : ℕ := 3^8

/-- The number of valid Hamiltonian cycles in a cube -/
def validHamiltonianCycles : ℕ := 12

theorem bug_probability (cube : Cube) :
  (validHamiltonianCycles : ℚ) / totalPaths = 4 / 2187 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_probability_l163_16355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_range_of_k_l163_16367

-- Define the power function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1)^2 * x^(m^2 - 4*m + 2)

-- Define the function g
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2^x - k

-- Theorem for part (I)
theorem unique_m_value :
  ∃! m : ℝ, ∀ x y : ℝ, 0 < x ∧ x < y → f m x < f m y :=
sorry

-- Theorem for part (II)
theorem range_of_k (m : ℝ) (h : ∀ x y : ℝ, 0 < x ∧ x < y → f m x < f m y) :
  ∃ a b : ℝ, a = 0 ∧ b = 1 ∧
  ∀ k : ℝ, a ≤ k ∧ k ≤ b ↔
    ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 →
      ∃ y : ℝ, 1 ≤ y ∧ y ≤ 2 ∧ g k x = f m y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_range_of_k_l163_16367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_no_zeros_iff_k_range_sum_of_logs_gt_two_l163_16371

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x - k * x

-- Theorem for part (1)
theorem tangent_line_at_point (k : ℝ) :
  k = 2 → ∃ (m : ℝ), ∀ (x y : ℝ), 
    y = f k x ∧ (x, y) = (1, -2) → 
    y - (-2) = m * (x - 1) ∧ m = -1 := by
  sorry

-- Theorem for part (2)
theorem no_zeros_iff_k_range (k : ℝ) :
  (∀ x > 0, f k x ≠ 0) ↔ k > Real.exp (-1) := by
  sorry

-- Theorem for part (3)
theorem sum_of_logs_gt_two (k : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 →
  Real.log x₁ + Real.log x₂ > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_no_zeros_iff_k_range_sum_of_logs_gt_two_l163_16371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_second_quadrant_l163_16393

theorem tan_double_angle_second_quadrant (α : Real) :
  (π / 2 < α) ∧ (α < π) →  -- α is in the second quadrant
  Real.sin α = 3 / 5 →
  Real.tan (2 * α) = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_second_quadrant_l163_16393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_angle_l163_16362

/-- Given a triangle with area 24, side length 8, and median to that side of length 7.5,
    prove that the sine of the angle between the side and the median is 4/5 -/
theorem triangle_sine_angle (A a m : ℝ) (h1 : A = 24) (h2 : a = 8) (h3 : m = 7.5) :
  Real.sin (Real.arcsin ((2 * A) / (a * m))) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_angle_l163_16362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_value_l163_16380

/-- A function f(x) defined as C - x^2 / 2, where C is a constant. -/
noncomputable def f (C : ℝ) (x : ℝ) : ℝ := C - x^2 / 2

/-- Theorem stating that if f(2k) = 3k for some k and f(6) = 9, then the constant C in f(x) is 27. -/
theorem constant_value (C : ℝ) (k : ℝ) (h1 : f C (2 * k) = 3 * k) (h2 : f C 6 = 9) : C = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_value_l163_16380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l163_16307

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  property : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * C.p * x

/-- Focus of a parabola -/
noncomputable def focus (C : Parabola) : ℝ × ℝ := (C.p/2, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Centroid of a triangle -/
noncomputable def centroid (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Origin point -/
def origin : ℝ × ℝ := (0, 0)

/-- Main theorem -/
theorem parabola_focus_property (C : Parabola) 
  (A B : PointOnParabola C) :
  distance (A.x, A.y) (focus C) + distance (B.x, B.y) (focus C) = 10 →
  centroid origin (A.x, A.y) (B.x, B.y) = focus C →
  C.p = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_property_l163_16307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_arccot_three_fifths_l163_16350

theorem tan_arccot_three_fifths : 
  Real.tan (Real.arctan (5 / 3)) = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_arccot_three_fifths_l163_16350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_matches_count_l163_16339

/-- The number of ways to arrange 5 lids on 5 numbered teacups -/
def totalArrangements : ℕ := 120 -- 5! = 5 * 4 * 3 * 2 * 1 = 120

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange 5 lids on 5 numbered teacups with at least two matches -/
def arrangementsWithAtLeastTwoMatches : ℕ :=
  choose 5 2 * 2 + choose 5 3 * 1 + 1

theorem at_least_two_matches_count :
  arrangementsWithAtLeastTwoMatches = 31 := by
  -- Unfold the definition of arrangementsWithAtLeastTwoMatches
  unfold arrangementsWithAtLeastTwoMatches
  -- Evaluate the expressions
  simp [choose]
  -- The result should now be 31
  rfl

#eval arrangementsWithAtLeastTwoMatches

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_matches_count_l163_16339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_faster_than_car_l163_16397

/-- The speed of a vehicle given its distance traveled and time taken -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem train_faster_than_car : 
  let car_distance : ℝ := 200
  let car_time : ℝ := 4
  let train_distance : ℝ := 210
  let train_time : ℝ := 3
  speed train_distance train_time > speed car_distance car_time := by
  -- Proof goes here
  sorry

#eval "QED"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_faster_than_car_l163_16397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_theorem_l163_16359

-- Define the rectangle ABCD and point P
variable (A B C D P : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_rectangle (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def on_side (P B C : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def distance (X Y : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

def angle (X Y Z : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- State the theorem
theorem rectangle_point_theorem 
  (h_rect : is_rectangle A B C D)
  (h_on_side : on_side P B C)
  (h_BP : distance B P = 24)
  (h_CP : distance C P = 6)
  (h_tan : Real.tan (angle A P D) = 4) :
  distance A B = 13.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_theorem_l163_16359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_equation_l163_16347

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := -Real.sqrt ((2 / x^2) - 1)

-- Define the derivative of y
noncomputable def y_derivative (x : ℝ) : ℝ := 
  (2 / (x^3 * Real.sqrt ((2 / x^2) - 1)))

-- Theorem statement
theorem y_satisfies_equation (x : ℝ) (h : x ≠ 0) :
  1 + (y x)^2 + x * (y x) * (y_derivative x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_satisfies_equation_l163_16347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_owens_turtles_formula_l163_16369

/-- Calculates the total number of turtles Owen has after 1 month -/
noncomputable def owens_turtles_after_one_month (G : ℝ) (X : ℕ) (Y : ℝ) : ℝ :=
  let owen_initial : ℕ := 21
  let johanna_initial : ℕ := owen_initial - 5
  let owen_after_growth : ℝ := 2 * (owen_initial : ℝ) * G
  let johanna_remaining : ℕ := johanna_initial / 2
  let liam_contribution : ℝ := (Y / 100) * (X : ℝ)
  owen_after_growth + (johanna_remaining : ℝ) + liam_contribution

/-- Theorem stating that Owen's total number of turtles after 1 month
    is equal to (42 * G) + 8 + (Y/100) * X -/
theorem owens_turtles_formula (G : ℝ) (X : ℕ) (Y : ℝ) :
  owens_turtles_after_one_month G X Y = 42 * G + 8 + (Y / 100) * (X : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_owens_turtles_formula_l163_16369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_valid_l163_16316

/-- The plane equation coefficients -/
def plane_equation : Fin 4 → ℤ := ![1, 2, -2, 6]

/-- The three points on the plane -/
def points : Fin 3 → Fin 3 → ℤ := ![![2, -1, 3], ![4, -1, 5], ![5, -3, 4]]

/-- Check if a point satisfies the plane equation -/
def satisfies_equation (p : Fin 3 → ℤ) : Prop :=
  plane_equation 0 * p 0 + plane_equation 1 * p 1 + plane_equation 2 * p 2 + plane_equation 3 = 0

theorem plane_equation_valid :
  (∀ i : Fin 3, satisfies_equation (points i)) ∧
  plane_equation 0 > 0 ∧
  Nat.gcd (Int.natAbs (plane_equation 0)) 
    (Nat.gcd (Int.natAbs (plane_equation 1)) 
      (Nat.gcd (Int.natAbs (plane_equation 2)) 
        (Int.natAbs (plane_equation 3)))) = 1 :=
by sorry

#check plane_equation_valid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_valid_l163_16316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_four_times_negative_four_equals_one_l163_16382

theorem power_four_times_negative_four_equals_one (a : ℚ) (h : a ≠ 0) :
  a^4 * a^(-4 : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_four_times_negative_four_equals_one_l163_16382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_has_property_P_infinitely_many_composite_with_property_P_l163_16321

-- Define property P
def has_property_P (n : ℕ) : Prop :=
  ∀ a : ℕ, (n ∣ a^n - 1) → (n^2 ∣ a^n - 1)

-- Theorem 1: Every prime number has property P
theorem prime_has_property_P (p : ℕ) (hp : Nat.Prime p) : has_property_P p := by
  sorry

-- Theorem 2: There are infinitely many composite numbers with property P
theorem infinitely_many_composite_with_property_P :
  ∃ S : Set ℕ, (∀ n ∈ S, ¬Nat.Prime n ∧ has_property_P n) ∧ Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_has_property_P_infinitely_many_composite_with_property_P_l163_16321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l163_16366

def A : Fin 3 → ℝ := ![3, 3, -1]
def B : Fin 3 → ℝ := ![5, 5, -2]
def C : Fin 3 → ℝ := ![4, 1, 1]

def vectorAB : Fin 3 → ℝ := fun i => B i - A i
def vectorAC : Fin 3 → ℝ := fun i => C i - A i

def dotProduct (v1 v2 : Fin 3 → ℝ) : ℝ := (v1 0) * (v2 0) + (v1 1) * (v2 1) + (v1 2) * (v2 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ := Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

theorem cosine_of_angle_between_vectors : 
  dotProduct vectorAB vectorAC / (magnitude vectorAB * magnitude vectorAC) = -4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l163_16366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l163_16388

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x - 6) / ((x - 3)^2)

-- State the theorem
theorem inequality_solution (x : ℝ) : 
  x ≠ 3 → (f x < 0 ↔ x ∈ Set.Iio 3 ∪ Set.Ioo 3 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l163_16388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sum_minimum_exponential_sum_equality_l163_16317

theorem exponential_sum_minimum (x : ℝ) : (2 : ℝ)^x + (2 : ℝ)^(-x) ≥ 2 := by
  sorry

theorem exponential_sum_equality : (2 : ℝ)^(0 : ℝ) + (2 : ℝ)^(-(0 : ℝ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sum_minimum_exponential_sum_equality_l163_16317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l163_16383

/-- Calculates the number of bricks required to pave a rectangular area -/
def bricks_required (courtyard_length courtyard_width brick_length brick_width : ℚ) : ℕ :=
  ⌊(courtyard_length * courtyard_width) / (brick_length * brick_width)⌋.toNat

/-- Theorem: The number of bricks required to pave a 20m by 16m courtyard with 20cm by 10cm bricks is 16000 -/
theorem courtyard_paving :
  bricks_required 20 16 (1/5) (1/10) = 16000 := by
  sorry

#eval bricks_required 20 16 (1/5) (1/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l163_16383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_is_2_sqrt_10_l163_16372

/-- The distance between the foci of a hyperbola defined by xy = 4 -/
noncomputable def hyperbola_foci_distance : ℝ := 2 * Real.sqrt 10

/-- Theorem: The distance between the foci of a hyperbola defined by xy = 4 is 2√10 -/
theorem hyperbola_foci_distance_is_2_sqrt_10 :
  let h : ℝ → ℝ → Prop := λ x y ↦ x * y = 4
  ∃ f₁ f₂ : ℝ × ℝ, (h f₁.1 f₁.2 ∧ h f₂.1 f₂.2) ∧
    Real.sqrt ((f₂.1 - f₁.1)^2 + (f₂.2 - f₁.2)^2) = hyperbola_foci_distance :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_is_2_sqrt_10_l163_16372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_existence_l163_16331

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- An ellipse in a 2D plane --/
structure Ellipse where
  center : ℝ × ℝ
  majorAxis : ℝ
  minorAxis : ℝ

/-- A line in a 2D plane represented by ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three points are collinear --/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Check if a line is tangent to a circle --/
def isTangentToCircle (l : Line) (c : Circle) : Prop := sorry

/-- Check if a line is tangent to an ellipse --/
def isTangentToEllipse (l : Line) (e : Ellipse) : Prop := sorry

/-- Check if a point lies on an ellipse --/
def onEllipse (p : ℝ × ℝ) (e : Ellipse) : Prop := sorry

/-- Check if a point lies on a circle --/
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

theorem ellipse_existence 
  (c : Circle) 
  (t : Line) 
  (p1 p2 : ℝ × ℝ) 
  (h1 : isTangentToCircle t c)
  (h2 : onCircle p1 c)
  (h3 : onCircle p2 c)
  (h4 : ¬ collinear p1 p2 c.center) :
  ∃ e : Ellipse, 
    e.center = c.center ∧ 
    isTangentToEllipse t e ∧ 
    onEllipse p1 e ∧ 
    onEllipse p2 e :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_existence_l163_16331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l163_16341

-- Define the sets A, B, and C
def A : Set ℝ := {x | (2 - x) / (3 + x) ≥ 0}
def B : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - (2*a + 1)*x + a*(a + 1) < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : C a ⊆ (A ∩ B) → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l163_16341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_per_gallon_l163_16304

noncomputable def city_efficiency : ℝ := 30
noncomputable def highway_efficiency : ℝ := 40
noncomputable def city_distance : ℝ := 60
noncomputable def highway_distance : ℝ := 200
noncomputable def total_cost : ℝ := 42

noncomputable def total_gallons : ℝ := 
  2 * (city_distance / city_efficiency + highway_distance / highway_efficiency)

theorem gas_cost_per_gallon : 
  total_cost / total_gallons = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_cost_per_gallon_l163_16304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_and_B_selected_is_three_tenths_l163_16305

def num_students : ℕ := 5
def num_selected : ℕ := 3

def probability_A_and_B_selected : ℚ := 3 / 10

theorem prob_A_and_B_selected_is_three_tenths :
  (Nat.choose (num_students - 2) (num_selected - 2) : ℚ) / (Nat.choose num_students num_selected : ℚ) = probability_A_and_B_selected := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_and_B_selected_is_three_tenths_l163_16305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_area_le_two_l163_16324

/-- A lattice point in a 2D plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The set of 6 lattice points satisfying the given conditions -/
def LatticePointSet : Type :=
  { points : Finset LatticePoint //
    points.card = 6 ∧
    (∀ p ∈ points, |p.x| ≤ 2 ∧ |p.y| ≤ 2) ∧
    (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
      (p2.y - p1.y) * (p3.x - p1.x) ≠ (p3.y - p1.y) * (p2.x - p1.x)) }

/-- The area of a triangle formed by three lattice points -/
def triangleArea (p1 p2 p3 : LatticePoint) : ℚ :=
  (1/2) * |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)|

/-- The main theorem -/
theorem exists_triangle_area_le_two (s : LatticePointSet) :
    ∃ p1 p2 p3, p1 ∈ s.val ∧ p2 ∈ s.val ∧ p3 ∈ s.val ∧ triangleArea p1 p2 p3 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_triangle_area_le_two_l163_16324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l163_16381

/-- Given a hyperbola M with eccentricity √3 and distance from one of its foci to an asymptote equal to 2,
    prove that its standard equation is either x²/2 - y²/4 = 1 or y²/2 - x²/4 = 1 -/
theorem hyperbola_standard_equation :
  ∃ (e d : ℝ), e = Real.sqrt 3 ∧ d = 2 →
  (∃ (x y : ℝ → ℝ), (∀ t, x t ^ 2 / 2 - y t ^ 2 / 4 = 1) ∨ 
   (∀ t, y t ^ 2 / 2 - x t ^ 2 / 4 = 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l163_16381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_x_axis_l163_16385

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-coordinate of the intersection point of a line with the x-axis -/
noncomputable def xAxisIntersection (l : Line) : ℝ :=
  (l.y₁ * l.x₂ - l.y₂ * l.x₁) / (l.y₁ - l.y₂)

theorem line_intersects_x_axis (l : Line) :
  l.x₁ = 2 ∧ l.y₁ = 8 ∧ l.x₂ = 6 ∧ l.y₂ = 0 →
  xAxisIntersection l = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_x_axis_l163_16385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_area_l163_16391

/-- The area of a regular polygon inscribed in a circle --/
theorem regular_polygon_area (n : ℕ) (R : ℝ) (h : n > 0) :
  let perimeter := 12 * R
  let side_length := perimeter / n
  let apothem := R * Real.cos (Real.pi / n)
  let area := n * side_length * apothem / 2
  ∃ ε > 0, |area - 5.8 * R^2| < ε * R^2 := by
  sorry

#check regular_polygon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_area_l163_16391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequalities_l163_16342

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) / x

-- State the theorem
theorem f_inequalities :
  (∀ x > 0, f x > 2 / (x + 2)) ∧
  (∀ x > -1, x ≠ 0 → f x < (1 + (1/2) * x) / (1 + x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequalities_l163_16342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_positive_sum_l163_16346

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) a

theorem greatest_positive_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 > 0 →
  a 2011 + a 2012 > 0 →
  a 2011 * a 2012 < 0 →
  (∀ n > 4022, sum_of_terms a n ≤ 0) ∧ sum_of_terms a 4022 > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_positive_sum_l163_16346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l163_16334

structure GeometricSpace where
  Line : Type
  Plane : Type
  perp_line_line : Line → Line → Prop
  perp_line_plane : Line → Plane → Prop
  parallel_line_line : Line → Line → Prop
  parallel_line_plane : Line → Plane → Prop
  parallel_plane_plane : Plane → Plane → Prop
  line_in_plane : Line → Plane → Prop

variable (S : GeometricSpace)

theorem all_propositions_false
  (a b : S.Line)
  (α β : S.Plane)
  (h_ab : a ≠ b)
  (h_αβ : α ≠ β) :
  (¬ (S.perp_line_line a b ∧ S.perp_line_plane a α → S.parallel_line_plane b α)) ∧
  (¬ (S.parallel_line_plane a α ∧ S.perp_line_plane a β → S.parallel_line_plane a β)) ∧
  (¬ (S.perp_line_plane a β ∧ S.perp_line_plane a α → S.parallel_line_plane a α)) ∧
  (¬ (S.parallel_line_line a b ∧ S.parallel_line_plane a α ∧ S.parallel_line_plane b β → S.parallel_plane_plane α β)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l163_16334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_MAB_l163_16323

/-- Curve C in the Cartesian coordinate system -/
noncomputable def curve_C (α : Real) : Real × Real := (3 + 5 * Real.cos α, 4 + 5 * Real.sin α)

/-- Point A on curve C -/
noncomputable def point_A : Real × Real := curve_C (Real.pi / 3)

/-- Point B on curve C -/
noncomputable def point_B : Real × Real := curve_C (Real.pi / 2)

/-- Center M of curve C -/
def point_M : Real × Real := (3, 4)

/-- Theorem: Area of triangle MAB is 25√3/4 -/
theorem area_triangle_MAB :
  let d_AB := Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2)
  let d_MA := Real.sqrt ((point_M.1 - point_A.1)^2 + (point_M.2 - point_A.2)^2)
  let d_MB := Real.sqrt ((point_M.1 - point_B.1)^2 + (point_M.2 - point_B.2)^2)
  let s := (d_AB + d_MA + d_MB) / 2
  let area := Real.sqrt (s * (s - d_AB) * (s - d_MA) * (s - d_MB))
  area = 25 * Real.sqrt 3 / 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_MAB_l163_16323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l163_16386

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.log (5 - x^2))

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -2 ≤ x ∧ x ≤ 2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l163_16386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l163_16301

def math_books : ℕ := 4
def history_books : ℕ := 6
def total_books : ℕ := math_books + history_books

theorem book_arrangement_count : 
  (history_books * Nat.factorial (total_books - 1)) = 30240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_count_l163_16301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_data_l163_16343

noncomputable def data : List ℝ := [5, 7, 7, 8, 10, 11]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ) ^ 2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem standard_deviation_of_data : standardDeviation data = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_of_data_l163_16343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_average_cost_l163_16375

/-- The total cost function for a factory's annual production. -/
noncomputable def total_cost (x : ℝ) : ℝ := x^2 / 10 - 30 * x + 4000

/-- The average cost per ton for a factory's annual production. -/
noncomputable def average_cost (x : ℝ) : ℝ := total_cost x / x

theorem lowest_average_cost :
  ∀ x : ℝ, 150 ≤ x → x ≤ 250 →
  average_cost x ≥ 10 ∧
  average_cost 200 = 10 ∧
  (average_cost x = 10 → x = 200) := by
  sorry

#check lowest_average_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_average_cost_l163_16375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l163_16395

-- Define the function f(x) = ln(1+x) - ln(1-x)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - Real.log (1 - x)

-- State the theorem
theorem f_properties :
  -- f is defined on (-1, 1)
  (∀ x : ℝ, -1 < x ∧ x < 1 → f x ≠ 0 → f x = f x) ∧
  -- f is an odd function
  (∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x) ∧
  -- f is increasing on (0, 1)
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 → f x < f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l163_16395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l163_16306

/-- Triangle ABC with given properties -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = π
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.C = 2 * π / 3) 
  (h2 : t.a = 6) : 
  (t.c = 14 → t.A.sin = (3 * Real.sqrt 3) / 14) ∧ 
  (t.a * t.b * t.C.sin / 2 = 3 * Real.sqrt 3 → t.c = 2 * Real.sqrt 13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l163_16306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l163_16308

/-- Given a hyperbola C with equation y²/6 - x²/b² = 1 and an asymptote y = √3 x,
    prove that the eccentricity of C is 2√3/3 -/
theorem hyperbola_eccentricity (b : ℝ) :
  (∃ C : Set (ℝ × ℝ), C = {(x, y) | y^2 / 6 - x^2 / b^2 = 1}) →
  (∃ asymptote : Set (ℝ × ℝ), asymptote = {(x, y) | y = Real.sqrt 3 * x}) →
  ∃ e : ℝ, e = (2 * Real.sqrt 3) / 3 ∧ e > 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l163_16308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_on_line_l163_16329

-- Define the type for a sample point
structure SamplePoint where
  x : ℝ
  y : ℝ

-- Define the type for a linear regression line
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

-- Define a function to calculate the residual for a point
def calcResidual (point : SamplePoint) (line : RegressionLine) : ℝ :=
  point.y - (line.slope * point.x + line.intercept)

-- Define a function to calculate the sum of squared residuals
def sumSquaredResiduals (points : List SamplePoint) (line : RegressionLine) : ℝ :=
  (points.map (fun p => (calcResidual p line) ^ 2)).sum

-- Theorem statement
theorem all_points_on_line
  (points : List SamplePoint)
  (line : RegressionLine)
  (h_linear : ∃ (a b : ℝ), ∀ p ∈ points, p.y = a * p.x + b)
  (h_zero_residuals : sumSquaredResiduals points line = 0) :
  ∀ p ∈ points, p.y = line.slope * p.x + line.intercept := by
  sorry

#check all_points_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_on_line_l163_16329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l163_16351

/-- Calculates the simple interest rate for a given principal, interest amount, and time period. -/
noncomputable def calculate_interest_rate (principal : ℝ) (interest : ℝ) (time : ℝ) : ℝ :=
  (interest * 100) / (principal * time)

/-- Represents the simple interest problem with given conditions -/
theorem simple_interest_problem (principal : ℝ) :
  (principal * 5 * 8 / 100 = 840) →
  (calculate_interest_rate principal 840 5 = 8) :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check simple_interest_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_problem_l163_16351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l163_16396

-- Define a power function
noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_through_point :
  ∃ α : ℝ, powerFunction α 2 = Real.sqrt 2 → 
  (∀ x : ℝ, x > 0 → powerFunction α x = Real.sqrt x) := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l163_16396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_voters_population_percentage_l163_16348

/-- The percentage of women in the population -/
noncomputable def women_percentage : ℝ := 52

/-- The percentage of women who are voters -/
noncomputable def women_voters_percentage : ℝ := 40

/-- The percentage of the population that consists of women voters -/
noncomputable def population_women_voters_percentage : ℝ := women_percentage * women_voters_percentage / 100

theorem women_voters_population_percentage :
  population_women_voters_percentage = 20.8 := by
  -- Unfold the definitions
  unfold population_women_voters_percentage women_percentage women_voters_percentage
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_voters_population_percentage_l163_16348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_parts_sqrt_6_13_parts_opposite_x_minus_y_l163_16330

noncomputable def intPart (x : ℝ) : ℤ := ⌊x⌋

noncomputable def decPart (x : ℝ) : ℝ := x - intPart x

theorem sqrt_10_parts :
  intPart (Real.sqrt 10) = 3 ∧ decPart (Real.sqrt 10) = Real.sqrt 10 - 3 := by sorry

theorem sqrt_6_13_parts :
  let a := decPart (Real.sqrt 6)
  let b := intPart (Real.sqrt 13)
  a + b - Real.sqrt 6 = 1 := by sorry

theorem opposite_x_minus_y :
  ∀ x y : ℝ, (x : ℝ) ∈ Set.range (Int.cast : ℤ → ℝ) → 0 < y → y < 1 → 12 + Real.sqrt 3 = x + y →
  -(x - y) = Real.sqrt 3 - 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_parts_sqrt_6_13_parts_opposite_x_minus_y_l163_16330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_square_perimeter_difference_infinite_impossible_d_values_l163_16328

theorem pentagon_square_perimeter_difference (d : ℕ+) : 
  (∃ (s p : ℝ), s > 0 ∧ p > s ∧ 5 * p - 4 * s = 2023 ∧ p - s = d) →
  d ≤ 404 :=
by sorry

theorem infinite_impossible_d_values : 
  Set.Infinite {d : ℕ+ | ¬(∃ (s p : ℝ), s > 0 ∧ p > s ∧ 5 * p - 4 * s = 2023 ∧ p - s = d)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_square_perimeter_difference_infinite_impossible_d_values_l163_16328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l163_16373

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (2^x + 1)

theorem f_properties :
  (f 1 = -1/3) ∧
  (∀ a, f a = (1 - 2^a) / (2^a + 1)) ∧
  (∀ x, f (-x) = -f x) :=
by
  constructor
  · -- Proof for f 1 = -1/3
    sorry
  constructor
  · -- Proof for ∀ a, f a = (1 - 2^a) / (2^a + 1)
    sorry
  · -- Proof for ∀ x, f (-x) = -f x
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l163_16373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_l163_16392

/-- A sequence satisfying the given recursive relation -/
def a : ℕ+ → ℕ := sorry

/-- The first term of the sequence is 1 -/
axiom a_1 : a 1 = 1

/-- The recursive relation for the sequence -/
axiom a_rec (m n : ℕ+) : a (n + m) = a n + a m + (n.val * m.val)

/-- The 100th term of the sequence is 5050 -/
theorem a_100 : a 100 = 5050 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_l163_16392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_negative_product_l163_16356

/-- Sequence definition for {aₙ} -/
def a : ℕ → ℚ
  | 0 => 15  -- Define for 0 to cover all natural numbers
  | n + 1 => (3 * a n - 2) / 3

/-- Proposition: The 23rd and 24th terms are the first adjacent terms with negative product -/
theorem adjacent_negative_product : 
  (∀ k < 23, a k * a (k + 1) > 0) ∧ (a 23 * a 24 < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_negative_product_l163_16356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_ratio_l163_16357

theorem circle_radii_ratio (α r R : Real) :
  α = 60 * Real.pi / 180 →  -- Convert 60° to radians
  r = R * Real.tan (α / 2) →  -- Relationship between radii and angle
  r / R = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_ratio_l163_16357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_optimal_price_reduction_value_l163_16340

-- Define the given constants
noncomputable def original_price : ℝ := 40
noncomputable def new_price : ℝ := 32.4
noncomputable def cost_price : ℝ := 30
def original_sales : ℕ := 48
noncomputable def profit_target : ℝ := 512
noncomputable def sales_increase_rate : ℝ := 4 / 0.5

-- Define the percentage decrease
def percentage_decrease (x : ℝ) : Prop :=
  original_price * (1 - x)^2 = new_price

-- Define the optimal price reduction
def optimal_price_reduction (y : ℝ) : Prop :=
  (original_price - cost_price - y) * (y * sales_increase_rate + original_sales) = profit_target

-- Theorem statements
theorem price_decrease_percentage :
  ∃ x : ℝ, percentage_decrease x ∧ x = 0.1 := by sorry

theorem optimal_price_reduction_value :
  ∃ y : ℝ, optimal_price_reduction y ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_optimal_price_reduction_value_l163_16340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_max_product_l163_16319

/-- Given two vectors a and b in ℝ², prove that their dot product being zero
    implies that the maximum value of the product of their second coordinates is 1/2. -/
theorem perpendicular_vectors_max_product (x y : ℝ) : 
  (1 * y + (x - 1) * 2 = 0) → x * y ≤ 1/2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_max_product_l163_16319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_case_l163_16364

/-- The length of the tangent line from a point to a circle -/
noncomputable def tangent_length (px py cx cy r : ℝ) : ℝ :=
  Real.sqrt ((px - cx)^2 + (py - cy)^2 - r^2)

/-- Theorem: The length of the tangent line from A(-1, 4) to the circle (x-2)^2 + (y-3)^2 = 1 is 3 -/
theorem tangent_length_specific_case :
  tangent_length (-1) 4 2 3 1 = 3 := by
  -- Unfold the definition of tangent_length
  unfold tangent_length
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_case_l163_16364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pieCuttingAlgorithmIsFair_l163_16322

/-- Represents a person's valuation of a piece of pie -/
def PersonValuation := ℝ → ℝ

/-- Represents a piece of pie -/
structure PiecePie where
  size : ℝ

/-- The pie-cutting algorithm -/
noncomputable def pieCuttingAlgorithm (n : ℕ) (people : Fin n → PersonValuation) : Fin n → PiecePie :=
  sorry

/-- Fairness property: each person believes they got at least 1/n of the pie -/
def isFairDivision (n : ℕ) (people : Fin n → PersonValuation) (division : Fin n → PiecePie) : Prop :=
  ∀ i : Fin n, people i (division i).size ≥ 1 / n

/-- Theorem: The pie-cutting algorithm results in a fair division -/
theorem pieCuttingAlgorithmIsFair (n : ℕ) (people : Fin n → PersonValuation) :
  isFairDivision n people (pieCuttingAlgorithm n people) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pieCuttingAlgorithmIsFair_l163_16322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshevT_positive_integer_sequence_equals_chebyshevT_all_terms_positive_integers_l163_16309

noncomputable def chebyshevT (c : ℕ+) : ℕ → ℤ
  | 0 => 1
  | 1 => c
  | n+2 => 2 * c * chebyshevT c (n+1) - chebyshevT c n

theorem chebyshevT_positive_integer (c : ℕ+) (n : ℕ) : 
  ∃ k : ℕ+, (chebyshevT c n : ℤ) = k :=
by sorry

theorem sequence_equals_chebyshevT (c : ℕ+) (n : ℕ) :
  chebyshevT c n = c * chebyshevT c (n-1) + 
    Int.sqrt ((c^2 - 1) * ((chebyshevT c (n-1))^2 - 1)) :=
by sorry

theorem all_terms_positive_integers (c : ℕ+) (n : ℕ) :
  ∃ k : ℕ+, (chebyshevT c n : ℤ) = k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chebyshevT_positive_integer_sequence_equals_chebyshevT_all_terms_positive_integers_l163_16309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_theorem_l163_16320

/-- Represents the spending pattern described in the problem --/
noncomputable def spending_pattern (n : ℝ) : ℝ :=
  let after_hardware := (3/4) * n
  let after_cleaners := after_hardware - 9
  let after_grocery := (1/2) * after_cleaners
  let after_bookstall := (2/3) * after_grocery
  (4/5) * after_bookstall

/-- Theorem stating that the spending pattern results in $27 if and only if the initial amount is $72 --/
theorem spending_theorem :
  ∀ n : ℝ, spending_pattern n = 27 ↔ n = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_theorem_l163_16320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celsius_five_times_l163_16314

/-- Converts Celsius temperature to Fahrenheit -/
noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ := c * (9/5) + 32

/-- The Celsius temperature at which the Fahrenheit temperature is exactly 5 times the Celsius temperature -/
theorem celsius_five_times : ∃ (c : ℝ), celsius_to_fahrenheit c = 5 * c ∧ c = 10 := by
  use 10
  constructor
  · simp [celsius_to_fahrenheit]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_celsius_five_times_l163_16314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dacid_marks_theorem_l163_16360

/-- Represents the marks obtained in five subjects -/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average of two numbers -/
def average (a b : ℕ) : ℚ := (a + b : ℚ) / 2

/-- Calculates the overall average of all subjects -/
def overallAverage (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.chemistry + m.biology : ℚ) / 5

/-- Theorem stating that no two-subject combination averages 90% or above,
    and the overall average is 85% -/
theorem dacid_marks_theorem (m : Marks)
    (h1 : m.english = 86)
    (h2 : m.mathematics = 89)
    (h3 : m.physics = 82)
    (h4 : m.chemistry = 87)
    (h5 : m.biology = 81) :
    (∀ (a b : ℕ), (a = m.english ∧ b = m.mathematics) ∨
                  (a = m.english ∧ b = m.physics) ∨
                  (a = m.english ∧ b = m.chemistry) ∨
                  (a = m.english ∧ b = m.biology) ∨
                  (a = m.mathematics ∧ b = m.physics) ∨
                  (a = m.mathematics ∧ b = m.chemistry) ∨
                  (a = m.mathematics ∧ b = m.biology) ∨
                  (a = m.physics ∧ b = m.chemistry) ∨
                  (a = m.physics ∧ b = m.biology) ∨
                  (a = m.chemistry ∧ b = m.biology) →
                  average a b < 90) ∧
    overallAverage m = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dacid_marks_theorem_l163_16360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crow_worm_trips_l163_16363

noncomputable def distance_to_ditch : ℝ := 200
noncomputable def time_gathering : ℝ := 1.5
noncomputable def crow_speed : ℝ := 4

noncomputable def round_trip_distance : ℝ := 2 * distance_to_ditch

noncomputable def speed_meters_per_minute : ℝ := (crow_speed * 1000) / 60

noncomputable def time_for_round_trip : ℝ := round_trip_distance / speed_meters_per_minute

noncomputable def total_time_minutes : ℝ := time_gathering * 60

theorem crow_worm_trips : 
  ⌊total_time_minutes / time_for_round_trip⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crow_worm_trips_l163_16363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_l163_16374

def P : Set ℕ := {1, 2, 3, 4}

def Q : Set ℝ := {x : ℝ | |x| ≤ 2}

def P_real : Set ℝ := {1, 2, 3, 4}

theorem intersection_P_Q : P_real ∩ Q = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_l163_16374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elsa_eva_never_meet_l163_16361

/-- Represents a point on the integer coordinate plane -/
structure Point where
  x : Int
  y : Int

/-- Represents the state of both walkers at any given time -/
structure WalkerState where
  elsa : Point
  eva : Point

/-- Checks if a move is valid (to an adjacent point) -/
def is_valid_move (p q : Point) : Prop :=
  (abs (p.x - q.x) + abs (p.y - q.y) = 1)

/-- Represents a sequence of valid moves for both walkers -/
def valid_walk (initial final : WalkerState) : Prop :=
  ∃ (n : Nat), ∃ (walk : Nat → WalkerState),
    walk 0 = initial ∧
    walk n = final ∧
    ∀ i, i < n → (is_valid_move (walk i).elsa (walk (i+1)).elsa) ∧
                 (is_valid_move (walk i).eva (walk (i+1)).eva)

/-- The main theorem: Elsa and Éva can never meet -/
theorem elsa_eva_never_meet :
  ¬∃ (final : WalkerState),
    valid_walk
      { elsa := { x := 0, y := 0 }, eva := { x := 0, y := 1 } }
      { elsa := final.elsa, eva := final.eva } ∧
    final.elsa = final.eva := by
  sorry

/-- Helper lemma: The sum of coordinates always remains odd -/
lemma sum_coordinates_odd (initial final : WalkerState)
    (h : valid_walk initial final) :
    (initial.elsa.x + initial.elsa.y + initial.eva.x + initial.eva.y) % 2 =
    (final.elsa.x + final.elsa.y + final.eva.x + final.eva.y) % 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elsa_eva_never_meet_l163_16361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_ratio_equal_surface_area_l163_16302

/-- Surface area of a regular tetrahedron with edge length a -/
noncomputable def tetrahedron_area (a : ℝ) : ℝ := Real.sqrt 3 * a^2

/-- Surface area of a regular octahedron with edge length b -/
noncomputable def octahedron_area (b : ℝ) : ℝ := 2 * Real.sqrt 3 * b^2

/-- Surface area of a regular icosahedron with edge length c -/
noncomputable def icosahedron_area (c : ℝ) : ℝ := 5 * Real.sqrt 3 * c^2

theorem edge_ratio_equal_surface_area (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_equal_area : tetrahedron_area a = octahedron_area b ∧ octahedron_area b = icosahedron_area c) :
  a / c = 2 * Real.sqrt 10 ∧ b / c = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_ratio_equal_surface_area_l163_16302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_l163_16352

theorem apple_cost (cost_two_dozen : ℝ) (h : cost_two_dozen = 15.60) :
  4 * (cost_two_dozen / 2) = 31.20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_l163_16352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_mass_theorem_l163_16333

/-- The mass of water in a bucket as a function of time -/
noncomputable def water_mass (capacity : ℝ) (flow_rate : ℝ) (t : ℝ) : ℝ :=
  min (flow_rate * t) capacity

/-- Theorem: The mass of water in the bucket follows the given function -/
theorem water_mass_theorem (t : ℝ) (h : 0 ≤ t ∧ t ≤ 5) :
  water_mass 10 4 t = if t < 2.5 then 4 * t else 10 := by
  sorry

#check water_mass_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_mass_theorem_l163_16333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_triples_satisfying_conditions_l163_16368

theorem infinitely_many_triples_satisfying_conditions :
  ∀ u : ℕ, 
  ∃ (x y z : ℕ),
    x = u^3 + 2*u ∧
    y = u^3 ∧
    z = 2*u^2 + 1 ∧
    (x^2 + 1 = y^2 + z^2) ∧
    (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z) ∧
    (x > 0) ∧ (y > 0) ∧ (z > 0) ∧
    (∃ (n : ℕ), n > u ∧
      ∃ (x' y' z' : ℕ),
        x' = n^3 + 2*n ∧
        y' = n^3 ∧
        z' = 2*n^2 + 1 ∧
        (x'^2 + 1 = y'^2 + z'^2) ∧
        (x' ≠ y') ∧ (y' ≠ z') ∧ (x' ≠ z') ∧
        (x' > 0) ∧ (y' > 0) ∧ (z' > 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_triples_satisfying_conditions_l163_16368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_coincide_l163_16344

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

/-- The equation of the parabola -/
def parabola_equation (x y p : ℝ) : Prop := x^2 = 2 * p * y

/-- The coordinates of the lower focus of the ellipse -/
noncomputable def ellipse_lower_focus : ℝ × ℝ := (0, -1)

/-- The coordinates of the focus of the parabola -/
noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_ellipse_focus_coincide (p : ℝ) : 
  parabola_focus p = ellipse_lower_focus → p = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_coincide_l163_16344
