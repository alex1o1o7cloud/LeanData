import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_angle_calculation_l43_4335

theorem right_triangle_angle_calculation (x : ℝ) : 
  (90 : ℝ) = 90 ∧ 3 * x + (4 * x - 10) = 90 → x = 100 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_angle_calculation_l43_4335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_shells_cost_l43_4369

/-- The cost of the taco shells given the total cost and the costs of other items. -/
theorem taco_shells_cost (total_cost : ℝ) (bell_pepper_cost : ℝ) (bell_pepper_count : ℕ) 
  (meat_cost_per_pound : ℝ) (meat_pounds : ℝ) : 
  total_cost - (bell_pepper_cost * (bell_pepper_count : ℝ) + meat_cost_per_pound * meat_pounds) = 5 :=
by
  sorry

#check taco_shells_cost 17 1.5 4 3 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_shells_cost_l43_4369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pastries_to_break_l43_4396

/-- The number of different fillings available -/
def num_fillings : ℕ := 10

/-- The total number of pastries -/
def total_pastries : ℕ := 45

/-- A pastry is represented as a pair of different fillings -/
def Pastry := {p : Fin num_fillings × Fin num_fillings // p.1 ≠ p.2}

/-- The set of all pastries -/
def all_pastries : Finset Pastry := sorry

/-- The number of pastries that need to be broken -/
def n : ℕ := 36

/-- Given a set of broken pastries, this function determines if it's possible
    to identify at least one filling for any remaining pastry -/
def can_identify_filling (broken : Finset Pastry) : Prop :=
  ∀ p ∈ all_pastries, p ∉ broken → ∃ f : Fin num_fillings, f ∈ [p.val.1, p.val.2]

theorem min_pastries_to_break :
  (∀ broken : Finset Pastry, broken.card = n → can_identify_filling broken) ∧
  (∀ m : ℕ, m < n → ∃ broken : Finset Pastry, broken.card = m ∧ ¬can_identify_filling broken) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pastries_to_break_l43_4396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_servings_is_26_l43_4349

/-- Represents the recipe for fruit punch -/
structure Recipe where
  servings : ℕ
  oranges : ℕ
  juice : ℚ
  soda : ℚ

/-- Represents the available ingredients -/
structure Ingredients where
  oranges : ℕ
  juice : ℚ
  soda : ℚ

/-- Calculates the maximum number of servings that can be prepared -/
def max_servings (recipe : Recipe) (ingredients : Ingredients) : ℕ :=
  min
    (ingredients.oranges * recipe.servings / recipe.oranges)
    (min
      (Nat.floor (ingredients.juice * recipe.servings / recipe.juice))
      (Nat.floor (ingredients.soda * recipe.servings / recipe.soda)))

/-- The given recipe -/
def fruit_punch_recipe : Recipe :=
  { servings := 8
  , oranges := 3
  , juice := 2
  , soda := 1 }

/-- Kim's available ingredients -/
def kim_ingredients : Ingredients :=
  { oranges := 10
  , juice := 12
  , soda := 5 }

theorem max_servings_is_26 :
  max_servings fruit_punch_recipe kim_ingredients = 26 := by
  sorry

#eval max_servings fruit_punch_recipe kim_ingredients

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_servings_is_26_l43_4349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_approx_116_44_l43_4380

/-- Regular hexagonal pyramid with a point P on its lateral edge -/
structure HexagonalPyramid where
  /-- Side length of the base hexagon -/
  a : ℝ
  /-- Height of the pyramid -/
  h : ℝ
  /-- Ratio of SP to PB on edge SB -/
  n : ℝ

/-- Ratio of volumes in a hexagonal pyramid divided by a plane -/
noncomputable def volumeRatio (pyramid : HexagonalPyramid) : ℝ :=
  let b := 1
  let c := pyramid.n
  let totalVolume := (Real.sqrt 3 * pyramid.a^2 * pyramid.h) / 2
  let prismVolume := (b^2 * c * Real.sqrt 3 * pyramid.a^2 * pyramid.h) / (2 * (b + c)^3)
  let smallerPyramidVolume := ((b / (b + c))^3 * Real.sqrt 3 * pyramid.a^2 * pyramid.h) / 3
  (totalVolume - prismVolume - smallerPyramidVolume) / (prismVolume + smallerPyramidVolume)

/-- Theorem stating the volume ratio for a specific hexagonal pyramid -/
theorem volume_ratio_approx_116_44 (pyramid : HexagonalPyramid) 
    (h1 : pyramid.n = 10) : 
    116.43 < volumeRatio pyramid ∧ volumeRatio pyramid < 116.45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_approx_116_44_l43_4380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_exponential_equation_l43_4381

theorem unique_solution_for_exponential_equation :
  ∀ (a b x y : ℕ), 
    (a > 0) → (b > 0) → (x > 0) → (y > 0) →
    x^(a + b) + y = x^a * y^b →
    a = 1 ∧ b = 1 ∧ x = 2 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_exponential_equation_l43_4381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_point_circle_theorem_l43_4351

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
def Point (c : Circle) : Type := { p : ℝ × ℝ // (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 }

-- Define a function to check if two lines are parallel
def parallel (c : Circle) (p1 p2 p3 p4 : Point c) : Prop :=
  (p2.1.1 - p1.1.1) * (p4.1.2 - p3.1.2) = (p2.1.2 - p1.1.2) * (p4.1.1 - p3.1.1)

-- Define a function to check if a line passes through a point
def passes_through (c : Circle) (p1 p2 p3 : Point c) : Prop :=
  (p3.1.1 - p1.1.1) * (p2.1.2 - p1.1.2) = (p3.1.2 - p1.1.2) * (p2.1.1 - p1.1.1)

-- Define a function to check if two points are the same
def same_point (c : Circle) (p1 p2 : Point c) : Prop :=
  p1.1 = p2.1

-- Main theorem
theorem seven_point_circle_theorem 
  (c : Circle)
  (p1 p2 p3 p4 p5 p6 p7 : Point c)
  (h1 : parallel c p4 p5 p1 p2)
  (h2 : parallel c p5 p6 p2 p3)
  (h3 : parallel c p6 p7 p3 p4)
  (h4 : passes_through c p4 p5 p5)
  (h5 : passes_through c p5 p6 p6)
  (h6 : passes_through c p6 p7 p7) :
  same_point c p7 p1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_point_circle_theorem_l43_4351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_sum_at_6_l43_4345

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem greatest_sum_at_6 (a₁ d : ℝ) (h₁ : a₁ = 11) (h₂ : d = -2) :
  ∀ k : ℕ, sum_arithmetic_sequence a₁ d 6 ≥ sum_arithmetic_sequence a₁ d k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_sum_at_6_l43_4345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_typing_theorem_l43_4318

/-- The time it takes for three people to type a document together -/
noncomputable def combined_typing_time (jonathan_time susan_time jack_time document_pages : ℝ) : ℝ :=
  document_pages / (document_pages / jonathan_time + document_pages / susan_time + document_pages / jack_time)

/-- Theorem: Given the individual typing times for a 10-page document,
    the combined typing time for Jonathan, Susan, and Jack is 10 minutes -/
theorem combined_typing_theorem (jonathan_time susan_time jack_time : ℝ) 
    (h1 : jonathan_time = 40)
    (h2 : susan_time = 30)
    (h3 : jack_time = 24) :
  combined_typing_time jonathan_time susan_time jack_time 10 = 10 := by
  sorry

#check combined_typing_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_typing_theorem_l43_4318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l43_4322

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := (1/2) * Real.cos (2*x - φ)

noncomputable def g (x : ℝ) : ℝ := (1/2) * Real.cos (4*x - Real.pi/3)

theorem problem_solution :
  (∀ φ : ℝ, 0 < φ ∧ φ < Real.pi ∧ f φ (Real.pi/6) = 1/2 → φ = Real.pi/3) ∧
  (Set.Icc 0 (Real.pi/4)).image g = Set.Icc (-1/4) (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l43_4322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_miles_theorem_l43_4383

/-- Calculates the additional miles needed to achieve a target average speed -/
noncomputable def additional_miles_for_average_speed (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (target_average_speed : ℝ) : ℝ :=
  let initial_time := initial_distance / initial_speed
  let h := (target_average_speed * initial_time - initial_distance) / (second_speed - target_average_speed)
  second_speed * h

theorem additional_miles_theorem (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (target_average_speed : ℝ) 
  (h_initial_distance : initial_distance = 15)
  (h_initial_speed : initial_speed = 30)
  (h_second_speed : second_speed = 55)
  (h_target_average_speed : target_average_speed = 50) :
  additional_miles_for_average_speed initial_distance initial_speed second_speed target_average_speed = 110 := by
  sorry

-- Remove the #eval statement as it's not necessary for compilation
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_miles_theorem_l43_4383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_ant_fly_ratio_l43_4310

/-- Represents a hexagonal prism with given dimensions -/
structure HexagonalPrism where
  height : ℝ
  base_side : ℝ

/-- Calculates the distance flown by the fly through the prism -/
noncomputable def fly_distance (prism : HexagonalPrism) : ℝ :=
  Real.sqrt ((2 * prism.base_side)^2 + prism.height^2)

/-- Calculates the distance crawled by the ant around the prism -/
noncomputable def ant_distance (prism : HexagonalPrism) (n : ℕ) : ℝ :=
  Real.sqrt (((6 * n + 3) * prism.base_side)^2 + prism.height^2)

/-- Theorem statement -/
theorem smallest_n_for_ant_fly_ratio (prism : HexagonalPrism) 
    (h1 : prism.height = 165)
    (h2 : prism.base_side = 30) :
    (∀ k : ℕ, k < 19 → ant_distance prism k ≤ 20 * fly_distance prism) ∧
    ant_distance prism 19 > 20 * fly_distance prism := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_ant_fly_ratio_l43_4310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_numbers_l43_4324

noncomputable def maxDistance (z : ℂ) : ℝ :=
  Complex.abs ((2 + 5*Complex.I) * z^4 - z^6)

theorem max_distance_complex_numbers :
  ∃ (c : ℝ), c = 81 * Real.sqrt 29 * |Real.sqrt 29 - 9| ∧
  (∀ (z : ℂ), Complex.abs z = 3 → maxDistance z ≤ c) ∧
  (∃ (z : ℂ), Complex.abs z = 3 ∧ maxDistance z = c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_numbers_l43_4324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_increase_three_days_l43_4379

/-- Represents the daily transactions of bicycles in Hank's store -/
structure DailyTransactions where
  sold : ℕ
  bought : ℕ

/-- Calculates the net change in bicycle inventory for a single day -/
def netChange (t : DailyTransactions) : ℤ :=
  t.bought - t.sold

/-- Represents the transactions over the three-day period -/
def weekendTransactions : List DailyTransactions :=
  [
    { sold := 10, bought := 15 },  -- Friday
    { sold := 12, bought := 8 },   -- Saturday
    { sold := 9, bought := 11 }    -- Sunday
  ]

/-- Theorem stating the net increase in bicycles over the three days -/
theorem net_increase_three_days :
  (weekendTransactions.map netChange).sum = 3 := by
  sorry

#eval (weekendTransactions.map netChange).sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_increase_three_days_l43_4379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadratic_expression_l43_4373

theorem unique_quadratic_expression (a b t : ℝ) (h1 : |a + 1| = t) (h2 : |Real.sin b| = t) :
  ∃! x : ℝ, x = a^2 + 2*a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_quadratic_expression_l43_4373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_time_theorem_l43_4327

/-- Represents a clock time in HH:MM format -/
structure ClockTime where
  hours : Nat
  minutes : Nat
  hh_valid : hours < 24
  mm_valid : minutes < 60

/-- Represents a single digit on the clock display -/
def Digit := Fin 10

/-- Function to check if a digit could have resulted from increasing or decreasing by 1 -/
def digit_adjacent (original current : Digit) : Prop :=
  (original.val + 1) % 10 = current.val ∨ (original.val + 9) % 10 = current.val

/-- Function to check if two ClockTimes could be related by the malfunction -/
def time_adjacent (original current : ClockTime) : Prop :=
  digit_adjacent ⟨original.hours / 10, by sorry⟩ ⟨current.hours / 10, by sorry⟩ ∧
  digit_adjacent ⟨original.hours % 10, by sorry⟩ ⟨current.hours % 10, by sorry⟩ ∧
  digit_adjacent ⟨original.minutes / 10, by sorry⟩ ⟨current.minutes / 10, by sorry⟩ ∧
  digit_adjacent ⟨original.minutes % 10, by sorry⟩ ⟨current.minutes % 10, by sorry⟩

theorem storm_time_theorem (malfunctioned : ClockTime) 
  (h_malfunction : malfunctioned.hours = 0 ∧ malfunctioned.minutes = 59) :
  ∃ (original : ClockTime), time_adjacent original malfunctioned ∧ 
    original.hours = 11 ∧ original.minutes = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_time_theorem_l43_4327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_sixth_value_l43_4301

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.tan (ω * x)

theorem tan_pi_sixth_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x : ℝ, f ω x = 2 → f ω (x + π / 2) = 2) : 
  f ω (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_sixth_value_l43_4301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l43_4314

/-- A rectangle ABCD is divided into four identical squares and has a perimeter of 160 cm. -/
structure Rectangle where
  side : ℝ
  side_positive : side > 0
  perimeter_eq : 4 * side = 160

/-- The area of the rectangle ABCD is 6400/9 square centimeters. -/
theorem rectangle_area (ABCD : Rectangle) : ABCD.side^2 * 4 = 6400 / 9 := by
  have h1 : ABCD.side = 40 / 3 := by
    -- Proof that side length is 40/3
    sorry
  
  calc
    ABCD.side^2 * 4 = (40 / 3)^2 * 4 := by rw [h1]
    _ = 1600 / 9 * 4 := by ring
    _ = 6400 / 9 := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l43_4314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_inequality_find_k_range_l43_4316

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 3|

-- Theorem 1: Solve f(x) > 2
theorem solve_inequality : ∀ x : ℝ, f x > 2 ↔ x < -2 := by sorry

-- Theorem 2: Find the range of k
theorem find_k_range : ∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-3) (-1) → f x ≤ k * x + 1) ↔ k ≤ -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_inequality_find_k_range_l43_4316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l43_4356

/-- Calculates the time (in seconds) needed for a train to cross a stationary train -/
noncomputable def timeToCrossStationaryTrain (trainSpeed : ℝ) (timeToPassPole : ℝ) (stationaryTrainLength : ℝ) : ℝ :=
  let speedInMPS := trainSpeed * 1000 / 3600
  let movingTrainLength := speedInMPS * timeToPassPole
  let totalLength := movingTrainLength + stationaryTrainLength
  totalLength / speedInMPS

/-- Theorem stating that a train with given parameters takes 27 seconds to cross a stationary train -/
theorem train_crossing_time :
  let trainSpeed : ℝ := 72 -- km/h
  let timeToPassPole : ℝ := 12 -- seconds
  let stationaryTrainLength : ℝ := 300 -- meters
  timeToCrossStationaryTrain trainSpeed timeToPassPole stationaryTrainLength = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l43_4356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_approx_l43_4317

/-- The average speed of a round trip given uphill and downhill speeds -/
noncomputable def average_round_trip_speed (uphill_speed downhill_speed : ℝ) : ℝ :=
  2 / (1 / uphill_speed + 1 / downhill_speed)

/-- Theorem stating that the average speed for the given round trip is approximately 9.52 mph -/
theorem round_trip_speed_approx :
  let uphill_speed := (5 : ℝ)
  let downhill_speed := (100 : ℝ)
  let result := average_round_trip_speed uphill_speed downhill_speed
  abs (result - 9.52) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval average_round_trip_speed 5 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_approx_l43_4317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l43_4385

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - 4 * x + a) / Real.log 10

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, f a x = y

def q (a : ℝ) : Prop := ∀ x : ℝ, x < -1 → 2 * x^2 + x > 2 + a * x

-- State the theorem
theorem a_range (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (1 ≤ a ∧ a ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l43_4385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_l43_4308

theorem salary_increase (S : ℝ) (h : S > 0) : 
  let annual_raise := 1.08
  let bonus := 1.05
  let final_salary := S * annual_raise^3 * bonus
  let percentage_increase := (final_salary / S - 1) * 100
  ∃ ε > 0, |percentage_increase - 32.27| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_l43_4308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l43_4331

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1 / 2^x) - x^(1/2)

-- State the theorem
theorem root_in_interval :
  (f 0 > 0) → (f 1 < 0) → ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l43_4331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_is_20_l43_4348

-- Define the setup
def sphere_radius_1 : ℝ := 20
def sphere_radius_2 : ℝ := 40
def sphere_radius_3 : ℝ := 40
def cone_base_radius : ℝ := 21

-- Define the property that the spheres and cone touch externally
def touch_externally : Prop := sorry

-- Define the height of the cone
noncomputable def cone_height : ℝ := sorry

-- Theorem statement
theorem cone_height_is_20 :
  touch_externally →
  cone_height = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_is_20_l43_4348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircles_cover_interior_l43_4337

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

-- Define a semicircle
def Semicircle (center : ℝ × ℝ) (diameter : ℝ) : Set (ℝ × ℝ) :=
  {p | dist p center ≤ diameter / 2 ∧ (p.1 - center.1) * (p.2 - center.2) ≥ 0}

-- Theorem statement
theorem semicircles_cover_interior (Q : ConvexQuadrilateral) :
  ∀ p ∈ interior (Set.range Q.vertices),
  ∃ i : Fin 4, p ∈ Semicircle
    ((Q.vertices i + Q.vertices (i.succ)) / 2)
    (dist (Q.vertices i) (Q.vertices (i.succ))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircles_cover_interior_l43_4337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_approx_l43_4375

/-- An equilateral triangle with a point inside satisfying specific distances --/
structure SpecialTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  Q : ℝ × ℝ
  is_equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                   (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  Q_inside : True  -- Placeholder condition, replace with actual condition if needed
  QA_dist : (Q.1 - A.1)^2 + (Q.2 - A.2)^2 = 36
  QB_dist : (Q.1 - B.1)^2 + (Q.2 - B.2)^2 = 64
  QC_dist : (Q.1 - C.1)^2 + (Q.2 - C.2)^2 = 100

/-- The area of the special triangle is approximately 67 --/
theorem special_triangle_area_approx (t : SpecialTriangle) :
  ∃ (area : ℝ), abs (area - 67) < 0.5 ∧
  area = (1/4) * Real.sqrt 3 * ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_area_approx_l43_4375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l43_4330

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- Check if a point lies on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The given line l -/
def l : Line :=
  { slope := -2, intercept := 10 }

/-- The point through which l' passes -/
def p : Point :=
  { x := -10, y := 0 }

/-- The perpendicular line l' -/
def l' : Line :=
  { slope := 1/2, intercept := 5 }

/-- The intersection point of l and l' -/
def intersection : Point :=
  { x := 2, y := 6 }

theorem intersection_point :
  perpendicular l l' ∧
  on_line p l' ∧
  on_line intersection l ∧
  on_line intersection l' := by
  sorry

#eval l.slope
#eval l'.slope
#eval intersection.x
#eval intersection.y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l43_4330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_ratio_product_l43_4353

theorem square_ratio_product (c d : ℝ) (hc : c = 45) (hd : d = 50) :
  (c^2 / d^2) * ((4*c) / (4*d)) = 729 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_ratio_product_l43_4353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_theorem_l43_4332

theorem special_number_theorem (N P Q q r : ℕ) :
  N = P * 10^q + Q →
  N = 2 * P * Q →
  1 ≤ r →
  r ≤ q →
  P = (5^r + 1) / 2 →
  Q = 2^(q-1) * 5^(q-r) * (5^r + 1) →
  N = 2^(q-1) * 5^(q-r) * (5^r + 1)^2 ∧
  (Nat.sqrt N * Nat.sqrt N = N ↔ q % 2 = 1 ∧ r % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_theorem_l43_4332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_midpoint_theorem_l43_4347

-- Define the basic structure
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define vector (we'll use this instead of redefining midpoint)
def vector (p q : ℝ × ℝ) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

-- Theorem statement
theorem triangle_midpoint_theorem (ABC : Triangle) 
  (hP : (ABC.A.1 + ABC.B.1) / 2 = (ABC.A.1 + (-1)) / 2 ∧ (ABC.A.2 + ABC.B.2) / 2 = (ABC.A.2 + (-2)) / 2)
  (hQ : (ABC.A.1 + ABC.C.1) / 2 = (ABC.A.1 + 3) / 2 ∧ (ABC.A.2 + ABC.C.2) / 2 = (ABC.A.2 + 4) / 2)
  (hPQ : vector ((ABC.A.1 + ABC.B.1) / 2, (ABC.A.2 + ABC.B.2) / 2) 
               ((ABC.A.1 + ABC.C.1) / 2, (ABC.A.2 + ABC.C.2) / 2) = (2, 3))
  (hB : ABC.B = (-1, -2)) :
  ABC.C = (3, 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_midpoint_theorem_l43_4347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_slope_three_points_l43_4343

/-- The slope of the best fit line for three equally spaced points -/
theorem best_fit_slope_three_points 
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h_order : x₁ < x₂ ∧ x₂ < x₃) 
  (h_spacing : x₃ - x₂ = x₂ - x₁) :
  let points := [(x₁, y₁), (x₂, y₂), (x₃, y₃)]
  let best_fit_slope := 
    (List.sum (points.map (fun p => (p.1 - x₂) * (p.2 - y₂)))) / 
    (List.sum (points.map (fun p => (p.1 - x₂)^2)))
  best_fit_slope = (y₃ - y₁) / (x₃ - x₁) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fit_slope_three_points_l43_4343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_prime_iff_k_eq_two_l43_4390

def N (k : ℕ) : ℕ :=
  10^(2*k - 1) + (Finset.range k).sum (fun i => 10^(2*(k-i) - 1))

theorem N_prime_iff_k_eq_two (k : ℕ) (h : k ≥ 2) :
  Nat.Prime (N k) ↔ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_prime_iff_k_eq_two_l43_4390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_bisecting_circle_diameter_l43_4394

/-- Represents a frustum (truncated cone) --/
structure Frustum where
  R : ℝ  -- radius of the bottom base
  r : ℝ  -- radius of the top base
  h : ℝ  -- height of the frustum

/-- The diameter of the circle that bisects the volume of a frustum --/
noncomputable def bisectingDiameter (f : Frustum) : ℝ :=
  2 * ((f.R^3 + f.r^3) / 2) ^ (1/3)

/-- Theorem: The diameter of the circle that bisects the volume of a frustum
    is equal to 2 * ∛((R³ + r³) / 2) --/
theorem frustum_bisecting_circle_diameter (f : Frustum) :
  let V := (f.h * Real.pi / 3) * (f.R^2 + f.R*f.r + f.r^2)  -- Volume of the frustum
  ∃ (x : ℝ), 0 < x ∧ x < f.h ∧
    ((x * Real.pi / 3) * (f.r^2 + f.r*(bisectingDiameter f)/2 + ((bisectingDiameter f)/2)^2)) = V/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_bisecting_circle_diameter_l43_4394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_ratio_sum_of_k_l43_4313

/-- Given a quadratic equation k(x^2 - x) + 2x + 7 = 0 with roots a and b,
    this function returns the sum of the ratios of the roots. -/
noncomputable def root_ratio_sum (k : ℝ) (a b : ℝ) : ℝ :=
  a / b + b / a

/-- Given a quadratic equation k(x^2 - x) + 2x + 7 = 0,
    this function checks if the roots satisfy the condition a/b + b/a = 3. -/
def satisfies_condition (k : ℝ) : Prop :=
  ∃ (a b : ℝ), k * (a^2 - a) + 2 * a + 7 = 0 ∧
                k * (b^2 - b) + 2 * b + 7 = 0 ∧
                root_ratio_sum k a b = 3

/-- k1 and k2 are the values of k that satisfy the condition. -/
axiom exists_k1_k2 : ∃ (k1 k2 : ℝ), satisfies_condition k1 ∧ satisfies_condition k2 ∧ k1 ≠ k2

/-- The main theorem stating that k1/k2 + k2/k1 = 1513/4. -/
theorem root_ratio_sum_of_k (k1 k2 : ℝ) 
  (h1 : satisfies_condition k1) 
  (h2 : satisfies_condition k2) 
  (h3 : k1 ≠ k2) : 
  k1 / k2 + k2 / k1 = 1513 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_ratio_sum_of_k_l43_4313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l43_4386

theorem smallest_number_divisible (n : ℕ) : n = 4722 ↔ 
  (∀ m : ℕ, m < n → 
    (¬(27 ∣ (m + 3)) ∨ 
     ¬(35 ∣ (m + 3)) ∨ 
     ¬(25 ∣ (m + 3)) ∨ 
     ¬(21 ∣ (m + 3)))) ∧ 
  (27 ∣ (n + 3)) ∧ 
  (35 ∣ (n + 3)) ∧ 
  (25 ∣ (n + 3)) ∧ 
  (21 ∣ (n + 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l43_4386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_reflection_distance_E_to_E_prime_l43_4300

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection (x y : ℝ) : 
  Real.sqrt ((x - x)^2 + ((-y) - y)^2) = 2 * abs y := by sorry

/-- The distance between E(2, 3) and its reflection E'(2, -3) is 6 --/
theorem distance_E_to_E_prime : 
  Real.sqrt ((2 - 2)^2 + ((-3) - 3)^2) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_reflection_distance_E_to_E_prime_l43_4300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_heptagon_interior_angle_measure_l43_4346

/-- The measure of an interior angle of a regular heptagon -/
noncomputable def regular_heptagon_interior_angle : ℝ := 900 / 7

/-- Theorem: The measure of an interior angle of a regular heptagon is (900/7) degrees -/
theorem regular_heptagon_interior_angle_measure :
  regular_heptagon_interior_angle = 900 / 7 := by
  -- Unfold the definition of regular_heptagon_interior_angle
  unfold regular_heptagon_interior_angle
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_heptagon_interior_angle_measure_l43_4346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangents_range_l43_4315

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

-- Define the theorem
theorem three_tangents_range (t : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ i ∈ ({x₁, x₂, x₃} : Set ℝ),
      (f i - t) / (i - 1) = (6 * i^2 - 3) ∧
      (∀ x : ℝ, x ≠ i → (f x - t) / (x - 1) ≠ (6 * i^2 - 3)))) →
  -3 < t ∧ t < -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tangents_range_l43_4315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_f_eq_zero_l43_4307

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then -x^2 - 2 else x/3 + 2

theorem no_solutions_for_f_eq_zero :
  ¬∃ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_f_eq_zero_l43_4307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_specific_triangle_area_l43_4372

/-- An ellipse with semi-major axis a and semi-minor axis b --/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The right focus of an ellipse --/
noncomputable def right_focus (e : Ellipse) : ℝ × ℝ := (Real.sqrt (e.a^2 - e.b^2), 0)

/-- A point on an ellipse --/
def point_on_ellipse (e : Ellipse) (P : ℝ × ℝ) : Prop :=
  (P.1^2 / e.a^2) + (P.2^2 / e.b^2) = 1

/-- The area of a right triangle formed by a point on the ellipse, the center, and the right focus --/
noncomputable def triangle_area (e : Ellipse) (P : ℝ × ℝ) : ℝ :=
  (1/2) * P.2 * (right_focus e).1

theorem ellipse_with_specific_triangle_area (e : Ellipse) :
  ∃ P : ℝ × ℝ, point_on_ellipse e P ∧ triangle_area e P = Real.sqrt 3 →
  e.a^2 = 2 * Real.sqrt 3 + 4 ∧ e.b^2 = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_with_specific_triangle_area_l43_4372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_sine_l43_4357

theorem root_sum_sine (x₁ x₂ m : ℝ) : 
  x₁ ∈ Set.Icc 0 (π / 2) →
  x₂ ∈ Set.Icc 0 (π / 2) →
  2 * Real.sin (2 * x₁) + Real.cos (2 * x₁) = m →
  2 * Real.sin (2 * x₂) + Real.cos (2 * x₂) = m →
  x₁ ≠ x₂ →
  Real.sin (x₁ + x₂) = 2 * Real.sqrt 5 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_sine_l43_4357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_transformation_l43_4388

-- Define the original and target functions
noncomputable def f (x : ℝ) := Real.cos x
noncomputable def g (x : ℝ) := Real.cos (2*x + Real.pi)

-- Define the transformation
noncomputable def transform (x : ℝ) : ℝ := (x + Real.pi) / 2

theorem cos_graph_transformation :
  ∀ x : ℝ, g x = f (transform x) := by
  intro x
  simp [g, f, transform]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_graph_transformation_l43_4388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_eq_chord_length_l43_4311

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + t, -1 + Real.sqrt 3 * t)

noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sin θ + 4 * Real.cos θ

def rect_eq_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 5

theorem curve_C_eq : ∀ x y : ℝ, 
  (∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ↔ rect_eq_C x y := by sorry

theorem chord_length : ∃ t₁ t₂ : ℝ, 
  rect_eq_C (line_l t₁).1 (line_l t₁).2 ∧ 
  rect_eq_C (line_l t₂).1 (line_l t₂).2 ∧ 
  ((line_l t₁).1 - (line_l t₂).1)^2 + ((line_l t₁).2 - (line_l t₂).2)^2 = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_eq_chord_length_l43_4311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l43_4334

noncomputable def curve (x : ℝ) : ℝ := x * Real.exp (-x)

def line (x : ℝ) : ℝ := x + 3

noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

theorem min_distance_curve_to_line :
  ∃ (x₀ : ℝ), ∀ (x y : ℝ),
    distance_point_to_line x (curve x) 1 (-1) 3 ≥
    distance_point_to_line x₀ (curve x₀) 1 (-1) 3 ∧
    distance_point_to_line x₀ (curve x₀) 1 (-1) 3 = (3 * Real.sqrt 2) / 2 :=
by
  sorry

#check min_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l43_4334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_17_l43_4366

-- Define the sequence a_n
def a : ℕ → ℤ
  | 0 => 2  -- Add a case for 0 to cover all natural numbers
  | 1 => 2
  | n + 2 => 2 * a (n + 1) - 1

-- Theorem statement
theorem a_5_equals_17 : a 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_17_l43_4366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_trig_function_l43_4376

theorem max_min_trig_function (α β : Real) (h : 2 * α + β = Real.pi) :
  (∀ x y : Real, 2 * x + y = Real.pi → Real.cos y - 6 * Real.sin x ≤ 7) ∧
  (∀ x y : Real, 2 * x + y = Real.pi → Real.cos y - 6 * Real.sin x ≥ -5) ∧
  (∃ x y : Real, 2 * x + y = Real.pi ∧ Real.cos y - 6 * Real.sin x = 7) ∧
  (∃ x y : Real, 2 * x + y = Real.pi ∧ Real.cos y - 6 * Real.sin x = -5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_trig_function_l43_4376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_catchup_time_correct_waiting_time_l43_4374

/-- Proves that the time required for a hiker to catch up with a cyclist is 30 minutes,
    given their speeds and the cyclist's travel time before stopping. -/
theorem hiker_catchup_time (hiker_speed : ℝ) (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) :
  hiker_speed = 4 →
  cyclist_speed = 24 →
  cyclist_travel_time = 5 / 60 →
  (cyclist_speed * cyclist_travel_time) / hiker_speed = 1 / 2 := by
  sorry

/-- Calculates the waiting time for the cyclist in minutes. -/
noncomputable def cyclist_waiting_time (hiker_speed : ℝ) (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) : ℝ :=
  (cyclist_speed * cyclist_travel_time) / hiker_speed * 60

/-- Proves that the waiting time for the cyclist is 30 minutes. -/
theorem correct_waiting_time (hiker_speed : ℝ) (cyclist_speed : ℝ) (cyclist_travel_time : ℝ) :
  hiker_speed = 4 →
  cyclist_speed = 24 →
  cyclist_travel_time = 5 / 60 →
  cyclist_waiting_time hiker_speed cyclist_speed cyclist_travel_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_catchup_time_correct_waiting_time_l43_4374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_password_reveal_l43_4391

-- Define the correspondence between expressions and words
def word_correspondence (expr : String) (word : String) : Prop := sorry

-- Define the expression to be factorized
def expression (a b x : ℝ) : ℝ := 3 * a * (x^2 - 1) - 3 * b * (x^2 - 1)

-- Theorem statement
theorem password_reveal (a b x : ℝ) :
  word_correspondence "a - b" "you" ∧
  word_correspondence "x - 1" "love" ∧
  word_correspondence "3" "China" ∧
  word_correspondence "x^2 + 1" "math" ∧
  word_correspondence "a" "study" ∧
  word_correspondence "x + 1" "country" →
  ∃ (f1 f2 f3 : String),
    expression a b x = 3 * (x + 1) * (x - 1) * (a - b) ∧
    word_correspondence f1 "China" ∧
    word_correspondence f2 "love" ∧
    word_correspondence f3 "you" :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_password_reveal_l43_4391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l43_4352

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - 3 * Real.pi / 2)
noncomputable def g (x : ℝ) : ℝ := Real.tan (x + Real.pi / 2)
noncomputable def h (x : ℝ) : ℝ := Real.cos (x + Real.pi / 2) / Real.cos (x - Real.pi)

theorem functions_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ x, g (-x) = -g x) ∧
  (∀ x, g (x + Real.pi) = g x) ∧
  (∀ x, h (-x) = -h x) ∧
  (∀ x, h (x + Real.pi) = h x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l43_4352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l43_4336

/-- The length of a train in meters -/
noncomputable def train_length : ℝ := 300

/-- The time taken for the train to cross a platform in seconds -/
noncomputable def time_cross_platform : ℝ := 38

/-- The time taken for the train to cross a signal pole in seconds -/
noncomputable def time_cross_pole : ℝ := 18

/-- The speed of the train in meters per second -/
noncomputable def train_speed : ℝ := train_length / time_cross_pole

/-- The length of the platform in meters -/
noncomputable def platform_length : ℝ := train_speed * time_cross_platform - train_length

theorem platform_length_calculation :
  ∃ ε > 0, |platform_length - 333.46| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l43_4336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincide_l43_4326

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = -2*p*x ∧ p > 0

-- Define the hyperbola
noncomputable def hyperbola (x y : ℝ) : Prop := x^2/3 - y^2 = 1

-- Define the focus of the parabola
noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (-p/2, 0)

-- Define the left focus of the hyperbola
def hyperbola_left_focus : ℝ × ℝ := (-2, 0)

-- Theorem statement
theorem parabola_hyperbola_focus_coincide (p : ℝ) :
  (∃ x y : ℝ, parabola p x y ∧ hyperbola x y) →
  parabola_focus p = hyperbola_left_focus →
  p = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincide_l43_4326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_dividing_power_minus_one_l43_4339

theorem smallest_n_dividing_power_minus_one (m : ℕ) (h1 : m % 2 = 1) (h2 : m > 2) :
  (∃ n : ℕ, n > 0 ∧ (m^n - 1) % 2^1989 = 0) ∧
  (∀ k : ℕ, 0 < k ∧ k < 2^1988 → (m^k - 1) % 2^1989 ≠ 0) ∧
  (m^(2^1988) - 1) % 2^1989 = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_dividing_power_minus_one_l43_4339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_distance_l43_4360

/-- The shortest distance from a point on the parabola x^2 = y to the line y = 2x + m is √5 -/
noncomputable def shortest_distance (m : ℝ) : ℝ := Real.sqrt 5

/-- The parabola is defined by x^2 = y -/
def on_parabola (x y : ℝ) : Prop := x^2 = y

/-- The line is defined by y = 2x + m -/
def on_line (x y m : ℝ) : Prop := y = 2*x + m

/-- The distance from a point (x, y) to the line y = 2x + m -/
noncomputable def distance_to_line (x y m : ℝ) : ℝ := 
  |2*x - y + m| / Real.sqrt 5

theorem parabola_line_distance (m : ℝ) : 
  (∀ x y : ℝ, on_parabola x y → 
    distance_to_line x y m ≥ shortest_distance m) ∧
  (∃ x y : ℝ, on_parabola x y ∧ 
    distance_to_line x y m = shortest_distance m) →
  m = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_distance_l43_4360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l43_4399

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the cube -/
def cube : Set Point3D :=
  {p | p.x ∈ ({0, 6} : Set ℝ) ∧ p.y ∈ ({0, 6} : Set ℝ) ∧ p.z ∈ ({0, 6} : Set ℝ)}

/-- Defines the plane intersecting the cube -/
def plane : Set Point3D :=
  {p | 3 * p.x - 2 * p.y - 2 * p.z = -6}

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- The main theorem -/
theorem intersection_distance :
  ∃ (S T : Point3D),
    S ∈ cube ∩ plane ∧
    T ∈ cube ∩ plane ∧
    S ≠ ⟨0, 3, 0⟩ ∧ S ≠ ⟨2, 0, 0⟩ ∧ S ≠ ⟨2, 6, 6⟩ ∧
    T ≠ ⟨0, 3, 0⟩ ∧ T ≠ ⟨2, 0, 0⟩ ∧ T ≠ ⟨2, 6, 6⟩ ∧
    distance S T = 7 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l43_4399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_l43_4377

/-- Given two lines p and q that intersect at (2, 7), prove that the slope of q is 3 -/
theorem intersection_slope (m : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + 3 → (x, y) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 + 3}) →  -- Line p
  (∀ x y : ℝ, y = m * x + 1 → (x, y) ∈ {q : ℝ × ℝ | q.2 = m * q.1 + 1}) →  -- Line q
  (2, 7) ∈ {p : ℝ × ℝ | p.2 = 2 * p.1 + 3} →  -- Intersection point on p
  (2, 7) ∈ {q : ℝ × ℝ | q.2 = m * q.1 + 1} →  -- Intersection point on q
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_l43_4377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_eighteen_l43_4384

/-- Represents a finite arithmetic sequence -/
structure ArithmeticSequence where
  a : ℝ  -- first term
  d : ℝ  -- common difference
  n : ℕ  -- number of terms

/-- Sum of the first k terms of an arithmetic sequence -/
noncomputable def sumFirstK (seq : ArithmeticSequence) (k : ℕ) : ℝ :=
  k * (2 * seq.a + (k - 1) * seq.d) / 2

/-- Sum of the last k terms of an arithmetic sequence -/
noncomputable def sumLastK (seq : ArithmeticSequence) (k : ℕ) : ℝ :=
  k * (2 * (seq.a + (seq.n - 1) * seq.d) - (k - 1) * seq.d) / 2

/-- The nth term of an arithmetic sequence -/
noncomputable def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1) * seq.d

theorem seventh_term_is_eighteen (seq : ArithmeticSequence) 
  (h1 : sumFirstK seq 5 = 34)
  (h2 : sumLastK seq 5 = 146)
  (h3 : sumFirstK seq seq.n = 234) :
  nthTerm seq 7 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_eighteen_l43_4384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_sum_2014_l43_4312

/-- Definition of the point P_n in the plane -/
noncomputable def P (n : ℤ) : ℝ × ℝ := (n - 2014/3, (n - 2014/3)^3)

/-- Three points are collinear if the determinant of their coordinates is zero -/
def collinear (p q r : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂) = 0

/-- The main theorem to be proved -/
theorem collinear_iff_sum_2014 :
  ∀ (a b c : ℤ), a ≠ b → b ≠ c → a ≠ c →
    (collinear (P a) (P b) (P c) ↔ a + b + c = 2014) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_sum_2014_l43_4312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_is_lattice_point_l43_4367

/-- A point in the 2D plane with integer coordinates -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three lattice points -/
structure Triangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- The area of a triangle given by the determinant formula -/
def triangleArea (t : Triangle) : ℚ :=
  (1 / 2) * |t.A.x * (t.B.y - t.C.y) + t.B.x * (t.C.y - t.A.y) + t.C.x * (t.A.y - t.B.y)|

/-- The orthocenter of a triangle -/
noncomputable def orthocenter (t : Triangle) : LatticePoint :=
  sorry

/-- Theorem: If a triangle has lattice point vertices and area 1/2, its orthocenter is a lattice point -/
theorem orthocenter_is_lattice_point (t : Triangle) (h : triangleArea t = 1/2) :
  ∃ (x y : ℤ), orthocenter t = LatticePoint.mk x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_is_lattice_point_l43_4367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_cm_distance_l43_4302

/-- Represents the scale of a map --/
structure MapScale where
  cm : ℚ  -- length in centimeters
  km : ℚ  -- corresponding length in kilometers

/-- Calculates the distance in kilometers for a given length in centimeters --/
def calculate_distance (scale : MapScale) (length : ℚ) : ℚ :=
  (scale.km / scale.cm) * length

/-- Theorem: On a map where 10 cm represents 80 km, 18 cm represents 144 km --/
theorem eighteen_cm_distance (scale : MapScale) 
  (h1 : scale.cm = 10) 
  (h2 : scale.km = 80) : 
  calculate_distance scale 18 = 144 := by
  -- Unfold the definition of calculate_distance
  unfold calculate_distance
  -- Simplify the expression
  simp [h1, h2]
  -- Perform the arithmetic
  norm_num
  -- QED
  
#eval calculate_distance { cm := 10, km := 80 } 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_cm_distance_l43_4302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_phi_l43_4355

noncomputable def f (w φ x : ℝ) : ℝ := Real.sin (w * x + φ)
noncomputable def g (w x : ℝ) : ℝ := Real.sin (w * x)

theorem find_phi (w φ : ℝ) : 
  w > 0 ∧ 
  |φ| < π/2 ∧
  (∀ x, f w φ (x + 6*π) = f w φ x) ∧
  (∀ x, f w φ (x + 3*π/8) = g w x) →
  φ = π/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_phi_l43_4355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seohyun_marbles_l43_4340

theorem seohyun_marbles :
  ∀ (original : ℕ),
    let after_jihoon := original / 2;
    let after_loss := (2 * after_jihoon) / 3;
    (after_jihoon > 0 ∧ after_loss = 12) → original = 36 :=
by
  intro original
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seohyun_marbles_l43_4340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_max_a_l43_4350

/-- The function f(x) = e^x(-x^2 + 2x + a) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (-x^2 + 2*x + a)

/-- The maximum value of a for which f is monotonically increasing on [a, a+1] -/
noncomputable def max_a : ℝ := (Real.sqrt 5 - 1) / 2

theorem monotone_increasing_max_a :
  ∀ a : ℝ, (∀ x y : ℝ, a ≤ x → x ≤ y → y ≤ a + 1 → f a x ≤ f a y) →
  a ≤ max_a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_max_a_l43_4350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_g_solution_set_of_g_l43_4344

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 3 * x + 6 else 2 * x - 13

-- Theorem statement
theorem solutions_of_g (x : ℝ) : g x = 3 ↔ x = -1 ∨ x = 8 := by
  sorry

-- Alternative formulation using a set
theorem solution_set_of_g : {x : ℝ | g x = 3} = {-1, 8} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_g_solution_set_of_g_l43_4344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_sum_bijection_l43_4323

/-- Representation of a number as an ordered sum of integers -/
def OrderedSum := List ℕ

/-- Predicate for a valid ordered sum of 1s and 2s -/
def IsValidSum1And2 (s : OrderedSum) : Prop :=
  s.all (λ x => x = 1 ∨ x = 2) ∧ s.sum = s.length

/-- Predicate for a valid ordered sum of integers greater than 1 -/
def IsValidSumGreaterThan1 (s : OrderedSum) : Prop :=
  s.all (λ x => x > 1) ∧ s.sum = s.length + 2

/-- The set of all valid ordered sums of 1s and 2s for a given n -/
def ValidSums1And2 (n : ℕ) : Set OrderedSum :=
  {s | IsValidSum1And2 s ∧ s.sum = n}

/-- The set of all valid ordered sums of integers greater than 1 for a given n -/
def ValidSumsGreaterThan1 (n : ℕ) : Set OrderedSum :=
  {s | IsValidSumGreaterThan1 s ∧ s.sum = n}

/-- The main theorem stating the existence of a bijection between the two sets -/
theorem ordered_sum_bijection (n : ℕ) :
  ∃ f : ValidSums1And2 n → ValidSumsGreaterThan1 (n + 2),
    Function.Bijective f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_sum_bijection_l43_4323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_is_arithmetic_mean_l43_4382

/-- Given a circle of radius r, prove that the side length of an inscribed square
    is the arithmetic mean of the side lengths of a circumscribed square and an inscribed regular octagon. -/
theorem inscribed_square_side_is_arithmetic_mean (r : ℝ) (r_pos : r > 0) : 
  r * Real.sqrt 2 = ((2 * r) + (r * Real.sqrt (2 - Real.sqrt 2))) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_side_is_arithmetic_mean_l43_4382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_theorem_l43_4393

/-- Represents the state of the urn -/
structure UrnState where
  red : ℕ
  blue : ℕ

/-- Represents one operation of drawing and adding a ball -/
def draw_and_add (state : UrnState) : UrnState → Prop :=
  sorry

/-- Represents the probability of a specific sequence of draws -/
def sequence_probability (initial : UrnState) (final : UrnState) (n : ℕ) : ℚ → Prop :=
  sorry

/-- The main theorem to be proved -/
theorem urn_probability_theorem :
  let initial_state : UrnState := ⟨2, 1⟩
  let final_state : UrnState := ⟨3, 5⟩
  let num_operations : ℕ := 5
  sequence_probability initial_state final_state num_operations (4 / 21) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_theorem_l43_4393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l43_4362

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_sum_magnitude (a b : V) 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 1) 
  (h3 : inner a b = -(1/2 : ℝ)) : 
  ‖a + (2 : ℝ) • b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l43_4362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_properties_l43_4359

/-- A function satisfying the given conditions -/
def f_conditions (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  (∀ x, f (x - 2) = f (-x)) ∧
  (∀ x, 2 * x + 1 = f 1 + (deriv f 1) * (x - 1))

/-- The main theorem -/
theorem tangent_line_properties {f : ℝ → ℝ} (hf : f_conditions f) : 
  deriv f 1 = 2 ∧ 
  (∀ x, -2 * x - 3 = f (-3) + (deriv f (-3)) * (x - (-3))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_properties_l43_4359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_binomial_coeff_count_is_power_of_two_odd_binomial_coeff_count_equals_two_pow_binary_ones_l43_4370

/-- The number of 1's in the binary representation of a natural number -/
def binaryOnes (n : ℕ) : ℕ :=
  (n.digits 2).count 1

/-- The count of odd binomial coefficients for a given n -/
def oddBinomialCoeffCount (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (fun h => Nat.choose n h % 2 = 1) |>.length

/-- Theorem: The count of odd binomial coefficients is a power of 2 -/
theorem odd_binomial_coeff_count_is_power_of_two (n : ℕ) :
  ∃ k, oddBinomialCoeffCount n = 2^k :=
by
  use binaryOnes n
  sorry

/-- Theorem: The count of odd binomial coefficients equals 2^k,
    where k is the number of 1's in the binary representation of n -/
theorem odd_binomial_coeff_count_equals_two_pow_binary_ones (n : ℕ) :
  oddBinomialCoeffCount n = 2^(binaryOnes n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_binomial_coeff_count_is_power_of_two_odd_binomial_coeff_count_equals_two_pow_binary_ones_l43_4370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l43_4325

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l43_4325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_root_angle_l43_4363

theorem repeated_root_angle (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : ∃ x : Real, x^2 + 4*x*Real.cos θ + Real.tan (π/2 - θ) = 0 ∧ 
    ∀ y : Real, y^2 + 4*y*Real.cos θ + Real.tan (π/2 - θ) = 0 → y = x) : 
  θ = π / 12 ∨ θ = 5*π / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_root_angle_l43_4363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l43_4305

theorem sin_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.cos (α + π / 4) = 5 / 13) : Real.sin α = 7 * Real.sqrt 2 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l43_4305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yan_distance_ratio_l43_4387

theorem yan_distance_ratio (w x z : ℝ) 
  (hw : w > 0) (hx : x > 0) (hz : z > 0) 
  (h_between : x + z > 0)
  (h_equal_time : z / w = x / w + (x + z) / (5 * w)) : 
  x / z = 2 / 3 := by
  sorry

#check yan_distance_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yan_distance_ratio_l43_4387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alberto_bjorn_distance_difference_l43_4319

-- Define the constants for the problem
noncomputable def time : ℝ := 5
noncomputable def bjorn_distance : ℝ := 75
noncomputable def alberto_distance : ℝ := 100

-- Define the speeds of Bjorn and Alberto
noncomputable def bjorn_speed : ℝ := bjorn_distance / time
noncomputable def alberto_speed : ℝ := alberto_distance / time

-- Theorem to prove
theorem alberto_bjorn_distance_difference :
  alberto_distance - bjorn_distance = 25 := by
  -- Unfold the definitions
  unfold alberto_distance bjorn_distance
  -- Perform the subtraction
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alberto_bjorn_distance_difference_l43_4319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_discount_l43_4397

noncomputable def discountedPrice (price : ℝ) : ℝ :=
  if price ≤ 600 then price
  else if price ≤ 900 then price * 0.8
  else 900 * 0.8 + (price - 900) * 0.6

noncomputable def originalPrice (discountedPrice : ℝ) : ℝ :=
  if discountedPrice ≤ 600 then discountedPrice
  else discountedPrice / 0.8

theorem shopping_discount (xiaoWangPayment motherPayment : ℝ) 
  (h1 : xiaoWangPayment = 560)
  (h2 : motherPayment = 640) :
  (discountedPrice (originalPrice xiaoWangPayment + originalPrice motherPayment) = 996) ∨
  (discountedPrice (originalPrice xiaoWangPayment + originalPrice motherPayment) = 1080) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_discount_l43_4397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l43_4338

theorem largest_expression : 
  let a := 3 + 1 + 4 + 5
  let b := 3 * 1 + 4 + 5
  let c := 3 + 1 * 4 + 5
  let d := 3 + 1 + 4 * 5
  let e := 3 * 1 * 4 * 5
  max a (max b (max c (max d e))) = e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l43_4338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_100ft_optimal_parallel_side_l43_4365

/-- Represents the dimensions and constraints of a rectangular cow pasture -/
structure Pasture where
  barn_length : ℝ
  fence_cost_per_foot : ℝ
  total_fence_cost : ℝ

/-- Calculates the area of the pasture given the length of the side parallel to the barn -/
noncomputable def pasture_area (p : Pasture) (parallel_side : ℝ) : ℝ :=
  let perpendicular_side := (p.total_fence_cost / p.fence_cost_per_foot - parallel_side) / 2
  parallel_side * perpendicular_side

/-- Theorem stating that the area is maximized when the parallel side is 100 feet -/
theorem max_area_at_100ft (p : Pasture) 
    (h1 : p.barn_length = 300)
    (h2 : p.fence_cost_per_foot = 10)
    (h3 : p.total_fence_cost = 2000) :
    ∀ x, x ≠ 100 → pasture_area p 100 ≥ pasture_area p x := by
  sorry

/-- Corollary: The length of the side parallel to the barn that maximizes the area is 100 feet -/
theorem optimal_parallel_side (p : Pasture) 
    (h1 : p.barn_length = 300)
    (h2 : p.fence_cost_per_foot = 10)
    (h3 : p.total_fence_cost = 2000) :
    ∃! x, ∀ y, pasture_area p x ≥ pasture_area p y ∧ x = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_100ft_optimal_parallel_side_l43_4365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_weight_exceeds_35_l43_4303

def weights_property (weights : List ℕ) : Prop :=
  ∀ (s₁ s₂ : List ℕ), s₁ ⊆ weights → s₂ ⊆ weights → s₁.length > s₂.length →
    s₁.sum > s₂.sum

theorem max_weight_exceeds_35 (weights : List ℕ) :
  weights.length = 11 →
  weights.Nodup →
  weights_property weights →
  ∃ w ∈ weights, w > 35 := by
  sorry

#check max_weight_exceeds_35

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_weight_exceeds_35_l43_4303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l43_4304

theorem circumscribed_sphere_surface_area (edge_length : ℝ) (h : edge_length = Real.sqrt 3) :
  4 * Real.pi * (Real.sqrt (3 * edge_length^2) / 2)^2 = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l43_4304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_beats_B_by_20_seconds_l43_4364

-- Define the race parameters
noncomputable def race_distance : ℝ := 1000
noncomputable def A_time : ℝ := 380
noncomputable def lead_distance : ℝ := 50

-- Define the speeds of runners A and B
noncomputable def speed_A : ℝ := race_distance / A_time
noncomputable def speed_B : ℝ := (race_distance - lead_distance) / A_time

-- Define the time difference between A and B
noncomputable def time_difference : ℝ := race_distance / speed_B - A_time

-- Theorem statement
theorem A_beats_B_by_20_seconds : 
  race_distance = 1000 ∧ 
  A_time = 380 ∧ 
  lead_distance = 50 → 
  time_difference = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_beats_B_by_20_seconds_l43_4364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cyclic_product_sum_l43_4378

theorem max_cyclic_product_sum (a b c d e : ℕ) : 
  a ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  b ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  c ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  d ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  e ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e →
  (∀ (x y z w v : ℕ), 
    ({x, y, z, w, v} : Set ℕ) = ({1, 2, 3, 4, 5} : Set ℕ) →
    x * y + y * z + z * w + w * v + v * x ≤ 59) ∧
  (∃ (x y z w v : ℕ), 
    ({x, y, z, w, v} : Set ℕ) = ({1, 2, 3, 4, 5} : Set ℕ) ∧
    x * y + y * z + z * w + w * v + v * x = 59) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cyclic_product_sum_l43_4378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_and_perimeter_l43_4320

/-- A triangle with side lengths 10, 12, and 12 -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  h_side1 : side1 = 12
  h_side2 : side2 = 12
  h_base : base = 10

/-- The area of the isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ := 5 * Real.sqrt 119

/-- The perimeter of the isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.side1 + t.side2 + t.base

theorem isosceles_triangle_area_and_perimeter (t : IsoscelesTriangle) :
  area t = 5 * Real.sqrt 119 ∧ perimeter t = 34 := by
  constructor
  · -- Proof for area
    rfl
  · -- Proof for perimeter
    simp [perimeter, t.h_side1, t.h_side2, t.h_base]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_and_perimeter_l43_4320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_and_distance_l43_4368

/-- Line l with parametric equation x = -2 - t, y = 2 - √3t -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-2 - t, 2 - Real.sqrt 3 * t)

/-- Curve C: (y-2)^2 - x^2 = 1 -/
def curve_C (p : ℝ × ℝ) : Prop := (p.2 - 2)^2 - p.1^2 = 1

/-- Point P in polar coordinates (2√2, 3π/4) -/
def point_P : ℝ × ℝ := (-2, 2)

/-- Theorem stating the length of AB and distance from P to midpoint M -/
theorem line_curve_intersection_and_distance :
  ∃ (A B : ℝ × ℝ) (t₁ t₂ : ℝ),
    line_l t₁ = A ∧ line_l t₂ = B ∧
    curve_C A ∧ curve_C B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 14 ∧
    let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
    Real.sqrt ((point_P.1 - M.1)^2 + (point_P.2 - M.2)^2) = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_and_distance_l43_4368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_tetrahedron_l43_4389

/-- Represents a tetrahedron ABCD with opposite edges AB and CD -/
structure Tetrahedron where
  AB : ℝ
  CD : ℝ
  d : ℝ  -- distance between skew lines AB and CD
  θ : ℝ  -- angle between AB and CD

/-- The volume of a tetrahedron given its parameters -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  (1/6) * t.AB * t.CD * t.d * Real.sin t.θ

/-- Theorem stating the volume of the specific tetrahedron ABCD -/
theorem volume_of_specific_tetrahedron :
  let t : Tetrahedron := {
    AB := 1,
    CD := 2,
    d := Real.sqrt 3,
    θ := π/6
  }
  tetrahedronVolume t = Real.sqrt 3 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_tetrahedron_l43_4389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_coefficient_l43_4371

/-- Given two lines l₁ and l₂, if they are perpendicular, then the coefficient a in l₂ is -1 -/
theorem perpendicular_lines_coefficient :
  ∀ a : ℝ, 
  (∀ x y : ℝ, x - 3*y + 2 = 0 ∧ 3*x - a*y - 1 = 0 → (1 : ℝ) * (1/3) * (3/a) = -1) → 
  a = -1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_coefficient_l43_4371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_theorem_l43_4309

/-- Represents the properties of a coal-powered train -/
structure Train where
  total_weight : ℝ
  remaining_coal : ℝ
  coal_consumption_rate : ℝ → ℝ

/-- Calculates the distance a train can travel before running out of fuel -/
noncomputable def distance_before_fuel_depletion (t : Train) : ℝ :=
  t.remaining_coal / (t.coal_consumption_rate t.total_weight)

/-- Theorem stating the distance a specific train can travel before running out of fuel -/
theorem train_distance_theorem (t : Train) 
  (h1 : t.remaining_coal = 160)
  (h2 : t.coal_consumption_rate = λ w => 0.004 * w) :
  distance_before_fuel_depletion t = 40000 / t.total_weight := by
  sorry

#check train_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_theorem_l43_4309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_geometric_sequence_l43_4333

theorem cosine_geometric_sequence (b : ℝ) : 
  0 < b → b < 2 * Real.pi →
  (∃ r : ℝ, Real.cos b ≠ 0 ∧ Real.cos (2 * b) = r * Real.cos b ∧ Real.cos (3 * b) = r * Real.cos (2 * b)) →
  b = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_geometric_sequence_l43_4333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_b_in_range_l43_4328

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then (b - 3/2) * x + b - 1
  else -x^2 + (2 + b) * x

theorem f_increasing_iff_b_in_range (b : ℝ) :
  (∀ x y : ℝ, x < y → f b x < f b y) ↔ (3/2 < b ∧ b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_b_in_range_l43_4328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l43_4306

theorem trigonometric_problem (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = 3/5)
  (h4 : Real.tan (α - β) = -1/3) : 
  Real.sin (α - β) = -Real.sqrt 10 / 10 ∧ 
  Real.cos β = 9 * Real.sqrt 10 / 50 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l43_4306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l43_4358

noncomputable def z : ℂ := 1 / (2 + Complex.I)

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l43_4358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l43_4341

-- Define the function f(x) = 1 / (x - 1)
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, IsRegular (f x) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l43_4341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_eq_imag_part_of_complex_l43_4398

theorem real_eq_imag_part_of_complex (a : ℝ) : 
  (a : ℂ) + Complex.I = Complex.mk a 1 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_eq_imag_part_of_complex_l43_4398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_range_l43_4321

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the theorem
theorem exponential_function_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x < 2) ↔ 
  a ∈ Set.Ioo (Real.sqrt 2 / 2) 1 ∪ Set.Ioo 1 (Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_range_l43_4321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l43_4354

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculates the area of a triangle given two sides and the included angle -/
noncomputable def triangleArea (side1 : ℝ) (side2 : ℝ) (angle : ℝ) : ℝ :=
  1/2 * side1 * side2 * Real.sin angle

/-- Main theorem -/
theorem hyperbola_triangle_area 
  (h : Hyperbola) 
  (f1 f2 p : Point) 
  (h_on_hyperbola : isOnHyperbola h p)
  (h_foci : f1.x = -h.a * Real.sqrt (1 + h.b^2 / h.a^2) ∧ 
            f1.y = 0 ∧ 
            f2.x = h.a * Real.sqrt (1 + h.b^2 / h.a^2) ∧ 
            f2.y = 0)
  (h_angle_diff : ∃ θ1 θ2, θ2 - θ1 = π/3 ∧ 
                  Real.tan θ1 = (p.y - f1.y) / (p.x - f1.x) ∧
                  Real.tan θ2 = (p.y - f2.y) / (p.x - f2.x)) :
  triangleArea (Real.sqrt ((f1.x - p.x)^2 + (f1.y - p.y)^2))
               (Real.sqrt ((f2.x - p.x)^2 + (f2.y - p.y)^2))
               (π/3) = 16 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l43_4354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l43_4395

/-- A hyperbola with equation 4x^2 - 2y^2 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1^2 - 2 * p.2^2 = 1}

/-- The foci of the hyperbola -/
def Foci : Set (ℝ × ℝ) := sorry

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- The angle F1PF2 is 60 degrees -/
def angle_F1PF2_60 : ℝ := sorry

/-- The area of triangle F1PF2 -/
def area_F1PF2 : ℝ := sorry

theorem hyperbola_triangle_area :
  P ∈ Hyperbola →
  ∃ F1 F2, F1 ∈ Foci ∧ F2 ∈ Foci ∧ F1 ≠ F2 ∧
  angle_F1PF2_60 = 60 * π / 180 →
  area_F1PF2 = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l43_4395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_a_equation_b_equation_c_l43_4329

-- Define the function f(x)
noncomputable def f (x : ℝ) := Real.sqrt (x + Real.sqrt (2*x - 1)) + Real.sqrt (x - Real.sqrt (2*x - 1))

-- Theorem for equation (a)
theorem equation_a (x : ℝ) (h : 1/2 ≤ x ∧ x ≤ 1) : f x = Real.sqrt 2 := by sorry

-- Theorem for equation (b)
theorem equation_b : ¬ ∃ x : ℝ, x ≥ 1/2 ∧ f x = 1 := by sorry

-- Theorem for equation (c)
theorem equation_c : f (3/2) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_a_equation_b_equation_c_l43_4329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equation_solution_l43_4342

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.sqrt ((x + 5) / 5)

-- State the theorem
theorem g_equation_solution :
  ∃ x : ℝ, g (2 * x) = 3 * g x ∧ x = -40 / 7 :=
by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equation_solution_l43_4342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_valued_polynomials_l43_4392

/-- A number is a palindrome if it reads the same backwards as forwards -/
def IsPalindrome (n : ℤ) : Prop := sorry

/-- A polynomial with integer coefficients -/
def IntPolynomial : Type := ℕ → ℤ

/-- Evaluation of a polynomial at a given point -/
def EvaluatePolynomial (P : IntPolynomial) (n : ℕ) : ℤ := sorry

/-- A polynomial P is palindrome-valued if P(n) is a palindrome for all n ≥ 0 -/
def IsPalindromeValued (P : IntPolynomial) : Prop :=
  ∀ n : ℕ, IsPalindrome (EvaluatePolynomial P n)

/-- A polynomial is constant if it has the same value for all inputs -/
def IsConstantPolynomial (P : IntPolynomial) : Prop :=
  ∀ m n : ℕ, EvaluatePolynomial P m = EvaluatePolynomial P n

theorem palindrome_valued_polynomials (P : IntPolynomial) :
  IsPalindromeValued P → IsConstantPolynomial P ∧ IsPalindrome (EvaluatePolynomial P 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_valued_polynomials_l43_4392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_and_complete_sequences_l43_4361

theorem disjoint_and_complete_sequences (α β : ℝ) 
  (h_pos_α : α > 0) (h_pos_β : β > 0)
  (h_irr_α : Irrational α) (h_irr_β : Irrational β)
  (h_sum : 1 / α + 1 / β = 1) :
  let a : ℕ+ → ℕ := fun n ↦ Int.toNat ⌊(n : ℝ) * α⌋
  let b : ℕ+ → ℕ := fun n ↦ Int.toNat ⌊(n : ℝ) * β⌋
  (∀ m n : ℕ+, a m ≠ b n) ∧ 
  (∀ k : ℕ+, ∃ n : ℕ+, a n = k ∨ b n = k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_and_complete_sequences_l43_4361
