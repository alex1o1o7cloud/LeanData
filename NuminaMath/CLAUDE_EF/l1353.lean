import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_card_gt_1000_l1353_135342

/-- The set of triplets (a, b, c) of natural numbers satisfying a^15 + b^15 = c^16 -/
def S : Set (ℕ × ℕ × ℕ) :=
  {abc : ℕ × ℕ × ℕ | let (a, b, c) := abc
                     a^15 + b^15 = c^16}

/-- Theorem stating that the set S has cardinality greater than 1000 -/
theorem S_card_gt_1000 : Nat.card S > 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_card_gt_1000_l1353_135342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1353_135388

noncomputable def f (x : ℝ) : ℝ := (4*x^2 + 2*x + 5) / (x^2 + x + 1)

theorem min_value_of_f :
  ∃ (y : ℝ), y = (16 - 2 * Real.sqrt 7) / 3 ∧
  (∀ (x : ℝ), x > 1 → f x ≥ y) ∧
  (∃ (x : ℝ), x > 1 ∧ f x = y) := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1353_135388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_angles_cosine_l1353_135356

theorem chord_angles_cosine (r : ℝ) (α β : ℝ) : 
  r > 0 ∧ 
  3 = 2 * r * Real.sin (α / 2) ∧ 
  4 = 2 * r * Real.sin (β / 2) ∧ 
  5 = 2 * r * Real.sin ((α + β) / 2) ∧ 
  0 < α ∧ 0 < β ∧ α + β < π →
  Real.cos α = 1/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_angles_cosine_l1353_135356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_l1353_135373

theorem triangle_tangent (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (cosine_relation : a^2 + b^2 - c^2 = -2/3 * a * b) :
  Real.tan (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_l1353_135373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_integers_with_thousands_digit_three_l1353_135329

def four_digit_integers_with_thousands_digit_three : Finset ℕ :=
  Finset.filter (λ n => 3000 ≤ n ∧ n ≤ 3999) (Finset.range 10000)

theorem count_four_digit_integers_with_thousands_digit_three :
  Finset.card four_digit_integers_with_thousands_digit_three = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_four_digit_integers_with_thousands_digit_three_l1353_135329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1353_135314

-- Define the distance between two parallel lines
noncomputable def distance_between_lines (a b c1 c2 : ℝ) : ℝ :=
  |c1 - c2| / Real.sqrt (a^2 + b^2)

-- Define the theorem
theorem parallel_lines_distance (c : ℝ) :
  distance_between_lines 1 (-2) (-1) (-c) = 2 * Real.sqrt 5 ↔ c = 11 ∨ c = -9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l1353_135314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminating_decimal_expansion_13_200_l1353_135352

theorem terminating_decimal_expansion_13_200 : 
  ∃ (n : ℕ) (k : ℤ), (13 : ℚ) / 200 = (k : ℚ) / (10 ^ n) ∧ (k : ℚ) / (10 ^ n) = 26 / 100 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminating_decimal_expansion_13_200_l1353_135352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1353_135378

noncomputable def f (x : ℝ) := Real.log (-3 * x^2 + 4 * x + 4)

theorem f_monotone_increasing :
  MonotoneOn f (Set.Ioo (-2/3) (2/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1353_135378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_conditions_l1353_135318

theorem system_solution_conditions (n p : ℕ) (hn : n > 0) (hp : p > 1) :
  (∃! (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + p * y = n ∧ x + y = p ^ z) ↔
  (∃ z : ℕ, z > 0 ∧ n % (p - 1) = (p ^ z) % (p - 1) ∧ n > p ^ z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_conditions_l1353_135318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_20_002_l1353_135309

/-- The length of the train in meters -/
noncomputable def train_length : ℝ := 110

/-- The length of the bridge in meters -/
noncomputable def bridge_length : ℝ := 142

/-- The time taken by the train to cross the bridge in seconds -/
noncomputable def crossing_time : ℝ := 12.598992080633549

/-- The total distance covered by the train while crossing the bridge -/
noncomputable def total_distance : ℝ := train_length + bridge_length

/-- The speed of the train in meters per second -/
noncomputable def train_speed : ℝ := total_distance / crossing_time

/-- Theorem stating that the train's speed is approximately 20.002 m/s -/
theorem train_speed_approx_20_002 : 
  |train_speed - 20.002| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_20_002_l1353_135309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l1353_135390

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 4 / (exp x + 1)

-- Define the derivative of the function
noncomputable def f_derivative (x : ℝ) : ℝ := -4 * exp x / (exp (2 * x) + 2 * exp x + 1)

theorem tangent_slope_range :
  ∀ x : ℝ, -1 ≤ f_derivative x ∧ f_derivative x < 0 := by
  sorry

#check tangent_slope_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l1353_135390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_equal_intercepts_l1353_135341

-- Define a structure for a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a point is on a line
def PointOnLine (l : Line) (x : ℝ) (y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

-- Define a function to check if x-intercept equals y-intercept
def EqualIntercepts (l : Line) : Prop :=
  ∃ t : ℝ, l.a * t + l.c = 0 ∧ l.b * t + l.c = 0

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∃ (l₁ l₂ : Line),
    (PointOnLine l₁ 2 5 ∧ EqualIntercepts l₁ ∧ l₁.a = 5 ∧ l₁.b = -2 ∧ l₁.c = 0) ∧
    (PointOnLine l₂ 2 5 ∧ EqualIntercepts l₂ ∧ l₂.a = 1 ∧ l₂.b = 1 ∧ l₂.c = -7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_equal_intercepts_l1353_135341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_store_sale_price_percentage_l1353_135354

/-- Calculates the lowest possible sale price for an item -/
noncomputable def lowestSalePrice (listPrice : ℝ) (minDiscount : ℝ) (additionalDiscount : ℝ) : ℝ :=
  listPrice * (1 - minDiscount) * (1 - additionalDiscount)

/-- Calculates the percentage of the total list price -/
noncomputable def percentageOfListPrice (salePrice : ℝ) (totalListPrice : ℝ) : ℝ :=
  (salePrice / totalListPrice) * 100

theorem soccer_store_sale_price_percentage : 
  let jerseyPrice := (80 : ℝ)
  let ballPrice := (40 : ℝ)
  let cleatsPrice := (100 : ℝ)
  let jerseyMinDiscount := (0.3 : ℝ)
  let ballMinDiscount := (0.4 : ℝ)
  let cleatsMinDiscount := (0.2 : ℝ)
  let jerseySummerDiscount := (0.2 : ℝ)
  let ballSummerDiscount := (0.25 : ℝ)
  let cleatsSummerDiscount := (0.15 : ℝ)
  
  let jerseyLowestPrice := lowestSalePrice jerseyPrice jerseyMinDiscount jerseySummerDiscount
  let ballLowestPrice := lowestSalePrice ballPrice ballMinDiscount ballSummerDiscount
  let cleatsLowestPrice := lowestSalePrice cleatsPrice cleatsMinDiscount cleatsSummerDiscount
  
  let totalLowestPrice := jerseyLowestPrice + ballLowestPrice + cleatsLowestPrice
  let totalListPrice := jerseyPrice + ballPrice + cleatsPrice
  
  let percentage := percentageOfListPrice totalLowestPrice totalListPrice
  
  abs (percentage - 54.09) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_store_sale_price_percentage_l1353_135354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_difference_l1353_135301

theorem integer_difference (x y : ℕ) (h1 : x + y = 20) (h2 : x * y = 96) : 
  x ≠ y → (x : ℤ) - y = 4 ∨ (x : ℤ) - y = -4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_difference_l1353_135301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_negative_330_deg_l1353_135328

-- Define secant function
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Define degree to radian conversion
noncomputable def deg_to_rad (θ : ℝ) : ℝ := θ * (Real.pi / 180)

theorem sec_negative_330_deg :
  sec (deg_to_rad (-330)) = (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_negative_330_deg_l1353_135328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_9_value_l1353_135375

noncomputable def S (m : ℕ) (x : ℝ) : ℝ := x^m + 1/x^m

theorem S_9_value (x : ℝ) (h : x + 1/x = 4) : S 9 x = 140248 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_9_value_l1353_135375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l1353_135377

def U : Set ℝ := Set.univ
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

theorem complement_of_M :
  U \ M = {x : ℝ | x < -1 ∨ x > 3} := by
  ext x
  simp [U, M, Set.mem_diff, Set.mem_univ, Set.mem_setOf]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_l1353_135377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_translates_l1353_135365

theorem disjoint_translates (A : Finset ℕ) (hAcard : A.card = 101) 
  (hAsub : A ⊆ Finset.range 1000000) : 
  ∃ (t : Fin 100 → ℕ), ∀ i j, i ≠ j → 
    (A.image (· + t i)) ∩ (A.image (· + t j)) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disjoint_translates_l1353_135365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_f_3_equals_108_l1353_135326

def f (x : ℝ) : ℝ := 2 * x + 4

def g (x : ℝ) : ℝ := 5 * x + 2

theorem f_g_f_3_equals_108 : f (g (f 3)) = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_f_3_equals_108_l1353_135326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_is_10_l1353_135307

/-- A ticket with a positive integer number -/
structure Ticket where
  number : ℕ+

/-- A collection of tickets -/
def TicketCollection := List Ticket

/-- The sum of numbers on a list of tickets -/
def sumTickets (tickets : TicketCollection) : ℕ := 
  tickets.map (λ t => t.number.val) |>.sum

instance : HasSubset TicketCollection := ⟨List.Subset⟩

theorem max_n_is_10 (n : ℕ) (tickets : TicketCollection) 
    (h1 : tickets.length = 2 * n + 1)
    (h2 : sumTickets tickets ≤ 2330)
    (h3 : ∀ (subset : TicketCollection), subset.length = n → subset ⊆ tickets → sumTickets subset > 1165) :
    n ≤ 10 ∧ ∃ (tickets : TicketCollection), 
      tickets.length = 2 * 10 + 1 ∧ 
      sumTickets tickets ≤ 2330 ∧ 
      (∀ (subset : TicketCollection), subset.length = 10 → subset ⊆ tickets → sumTickets subset > 1165) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_is_10_l1353_135307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1353_135393

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (x^2 - 3*x + a)

theorem f_properties :
  ∀ a : ℝ,
  (∀ x : ℝ, x < -1 ∨ x > 3 → StrictMono (f (-9) ∘ (fun y => max x y))) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 1 2 → ¬(Monotone (f a)) ↔ a ≤ 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo 0 2 ∧ x₂ ∈ Set.Ioo 0 2 ∧
    x₁ ≠ x₂ ∧ 
    (∀ x ∈ Set.Ioo x₁ x₂, (f a x₁ ≤ f a x ∧ f a x₂ ≤ f a x) ∨ (f a x₁ ≥ f a x ∧ f a x₂ ≥ f a x)) ∧
    |f a x₁ - f a x₂| > |f a x₁ + f a x₂|
   ↔ 0 < a ∧ a < 9/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1353_135393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekday_rate_is_420_l1353_135315

/-- The weekday rental rate for an Airbnb at Lake Tahoe -/
noncomputable def weekday_rate : ℚ :=
  let num_people : ℕ := 6
  let num_weekdays : ℕ := 2
  let num_weekend_days : ℕ := 2
  let weekend_rate : ℚ := 540
  let per_person_total : ℚ := 320
  ((num_people * per_person_total) - (num_weekend_days * weekend_rate)) / num_weekdays

theorem weekday_rate_is_420 : weekday_rate = 420 := by
  -- Proof goes here
  sorry

-- Remove #eval as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekday_rate_is_420_l1353_135315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_monotonically_decreasing_k_range_l1353_135333

noncomputable def f (a b x : ℝ) : ℝ := (b - 2^x) / (2^(x+1) + a)

theorem odd_function_values (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) → a = 2 ∧ b = 1 := by sorry

theorem monotonically_decreasing (a b : ℝ) :
  a = 2 ∧ b = 1 → (∀ x y, x < y → f a b x > f a b y) := by sorry

theorem k_range (a b k : ℝ) :
  a = 2 ∧ b = 1 →
  (∀ x : ℝ, x ≥ 1 → f a b (k * 3^x) + f a b (3^x - 9^x + 2) > 0) ↔
  k < 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_values_monotonically_decreasing_k_range_l1353_135333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_sqrt_sum_primes_l1353_135311

theorem infinitely_many_sqrt_sum_primes :
  ∀ k : ℕ, ∃ n p : ℕ, 
    p > k ∧ 
    Nat.Prime p ∧ 
    (Real.sqrt (p + n : ℝ) + Real.sqrt (n : ℝ) : ℝ) = p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_sqrt_sum_primes_l1353_135311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1353_135334

noncomputable def f (a b x : ℝ) : ℝ := (a * x^2 + b) / Real.sqrt (x^2 + 1)

theorem min_value_theorem (a b : ℝ) :
  (∃ m : ℝ, m = 3 ∧ ∀ x : ℝ, f a b x ≥ m) →
  (b ≥ 3 ∧ a = (b - Real.sqrt (b^2 - 9)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1353_135334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_is_52_seconds_l1353_135372

/-- The time (in seconds) for a train to cross a man moving in the opposite direction -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_ms := relative_speed * 1000 / 3600
  train_length / relative_speed_ms

/-- Proof that the train crossing time is 52 seconds given the specified conditions -/
theorem train_crossing_time_is_52_seconds :
  train_crossing_time 390 25 2 = 52 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_crossing_time 390 25 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_is_52_seconds_l1353_135372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_odd_l1353_135361

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -Real.sin (x + Real.pi / 2)

-- Theorem statement
theorem f_not_odd : ¬(∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_odd_l1353_135361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_iff_magnitude_one_vector_b_is_unit_vector_c_is_unit_vector_d_is_unit_l1353_135371

def is_unit_vector (a : ℝ × ℝ) : Prop :=
  Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 1

theorem unit_vector_iff_magnitude_one (a : ℝ × ℝ) :
  is_unit_vector a ↔ Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 1 := by
  sorry

-- Examples from the problem
def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (-1, 0)

noncomputable def vector_c : ℝ × ℝ := (Real.cos (38 * Real.pi / 180), Real.sin (52 * Real.pi / 180))

-- For vector_d, we need to define it as a function
noncomputable def vector_d (m : ℝ × ℝ) (h : Real.sqrt ((m.1 ^ 2) + (m.2 ^ 2)) ≠ 0) : ℝ × ℝ :=
  let magnitude := Real.sqrt ((m.1 ^ 2) + (m.2 ^ 2))
  (m.1 / magnitude, m.2 / magnitude)

-- Theorems for each vector
theorem vector_b_is_unit : is_unit_vector vector_b := by
  sorry

theorem vector_c_is_unit : is_unit_vector vector_c := by
  sorry

theorem vector_d_is_unit (m : ℝ × ℝ) (h : Real.sqrt ((m.1 ^ 2) + (m.2 ^ 2)) ≠ 0) :
  is_unit_vector (vector_d m h) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_iff_magnitude_one_vector_b_is_unit_vector_c_is_unit_vector_d_is_unit_l1353_135371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1353_135344

/-- Given a triangle ABC with the following properties:
  - Vectors m = (a, cos A) and n = (sin B, √3b) are perpendicular
  - a = √7
  - b + c = 3
  Prove that angle A is 2π/3 and the area of the triangle is √3/2 -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  let m : ℝ × ℝ := (a, Real.cos A)
  let n : ℝ × ℝ := (Real.sin B, Real.sqrt 3 * b)
  (m.1 * n.1 + m.2 * n.2 = 0) →  -- m ⊥ n
  (a = Real.sqrt 7) →
  (b + c = 3) →
  (A = 2 * Real.pi / 3 ∧ 
   (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1353_135344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_1060_l1353_135355

-- Define the sequence
def mySequence : ℕ → ℕ
| 0 => 1
| n + 1 => 
  let block := (n + 1) / 2 + 1
  let position := (n + 1) % (block + 1)
  if position == 0 then 1 else position + 1

-- Define the sum of the first n terms
def mySequenceSum (n : ℕ) : ℕ :=
  (List.range n).map mySequence |>.sum

-- Theorem statement
theorem sequence_sum_1060 : mySequenceSum 1060 = 9870 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_1060_l1353_135355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_is_empty_l1353_135353

open Set Real

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define set A
def A : Set ℝ :=
  {x | (floor x)^2 - 2 * (floor x) = 3}

-- Define set B
def B : Set ℝ :=
  {x | (2 : ℝ)^x > 8}

-- Theorem statement
theorem A_intersect_B_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_is_empty_l1353_135353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_forest_trees_l1353_135391

def forest_length : ℝ := 100
def forest_width : ℝ := 0.5
def plot_length : ℝ := 1
def plot_width : ℝ := 0.5
def sample_counts : List ℕ := [65110, 63200, 64600, 64700, 67300, 63300, 65100, 66600, 62800, 65500]

theorem estimate_forest_trees :
  let avg_trees := (sample_counts.sum / sample_counts.length : ℝ)
  let total_plots := (forest_length * forest_width) / (plot_length * plot_width)
  Int.floor (avg_trees * total_plots) = 6482100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_forest_trees_l1353_135391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tylers_puppies_l1353_135340

theorem tylers_puppies (total_dogs : ℕ) 
                       (group1_dogs group2_dogs : ℕ) 
                       (group1_puppies group2_puppies group3_puppies : ℚ) 
                       (additional_dogs : ℕ) 
                       (additional_puppies : ℚ) 
                       (h1 : total_dogs = 35)
                       (h2 : group1_dogs = 15)
                       (h3 : group2_dogs = 10)
                       (h4 : group1_puppies = 5.5)
                       (h5 : group2_puppies = 8)
                       (h6 : group3_puppies = 6)
                       (h7 : additional_dogs = 5)
                       (h8 : additional_puppies = 2.5) :
  (group1_dogs * group1_puppies + 
   group2_dogs * group2_puppies + 
   (total_dogs - group1_dogs - group2_dogs) * group3_puppies + 
   additional_dogs * additional_puppies) = 235 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tylers_puppies_l1353_135340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_speed_l1353_135395

theorem bike_ride_speed (joann_speed joann_time fran_time fran_speed : ℝ) 
  (h1 : joann_speed = 12)
  (h2 : joann_time = 4.5)
  (h3 : fran_time = 4)
  (h4 : joann_speed * joann_time = fran_time * fran_speed) :
  fran_speed = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_ride_speed_l1353_135395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_present_expenditure_is_1000_l1353_135324

/-- Represents the yearly increase rate of expenditure -/
noncomputable def yearly_increase_rate : ℝ := 1.30

/-- Represents the expenditure after 3 years in rupees -/
def expenditure_after_3_years : ℝ := 2197

/-- Calculates the present expenditure given the expenditure after 3 years and the yearly increase rate -/
noncomputable def calculate_present_expenditure (future_expenditure : ℝ) (rate : ℝ) : ℝ :=
  future_expenditure / (rate ^ 3)

/-- Theorem stating that the present expenditure is approximately 1000 rupees -/
theorem present_expenditure_is_1000 :
  ∃ ε > 0, |calculate_present_expenditure expenditure_after_3_years yearly_increase_rate - 1000| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_present_expenditure_is_1000_l1353_135324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1353_135325

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * |x + a| - (1/2) * Real.log x

-- State the theorem
theorem f_properties (a : ℝ) :
  (∀ x > 0, f a x > 0) ↔ a > -1 ∧
  (a < -2 →
    ∃ x₁ x₂ x₃ : ℝ,
      x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧
      x₁ < x₂ ∧ x₂ < x₃ ∧
      (∀ y > 0, f a y ≥ f a x₁) ∧
      (∀ y > 0, f a y ≤ f a x₂) ∧
      (∀ y > 0, f a y ≥ f a x₃)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1353_135325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1353_135362

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B as a function of a
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Theorem for problem 1
theorem problem_1 : 
  A ∪ B 2 = {x : ℝ | x ≥ -1} ∧ A ∩ (U \ B 2) = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for problem 2
theorem problem_2 : 
  ∀ a : ℝ, A ∩ B a = A ↔ a < -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1353_135362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_range_l1353_135327

noncomputable section

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define point P
def point_P : ℝ × ℝ := (0, Real.sqrt 2)

-- Define a line through P
def line_through_P (θ : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * Real.cos θ, Real.sqrt 2 + t * Real.sin θ)

-- Define the intersection points of the line with C
def intersection_points (θ : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_through_P θ t ∧ curve_C p.1 p.2}

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Main theorem
theorem intersection_distance_product_range :
  ∀ θ : ℝ, ∀ A B : ℝ × ℝ, A ∈ intersection_points θ → B ∈ intersection_points θ →
    A ≠ B →
    1 ≤ distance point_P A * distance point_P B ∧
    distance point_P A * distance point_P B ≤ 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_range_l1353_135327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_theorem_r33_l1353_135338

-- Define a type for people
def Person : Type := ℕ

-- Define the acquaintance relation
def acquainted : Person → Person → Prop := sorry

-- Theorem statement
theorem ramsey_theorem_r33 :
  ∀ (S : Finset Person), S.card = 6 →
  ∃ (T : Finset Person), T ⊆ S ∧ T.card = 3 ∧
  ((∀ x y, x ∈ T → y ∈ T → x ≠ y → acquainted x y) ∨
   (∀ x y, x ∈ T → y ∈ T → x ≠ y → ¬acquainted x y)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramsey_theorem_r33_l1353_135338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_pair_20th_row_l1353_135379

/-- Represents a pair of integers -/
structure IntPair :=
  (first : Int)
  (second : Int)

/-- Generates the nth row of the table -/
def generateRow (n : Nat) : List IntPair :=
  List.range n |> List.map (fun i => ⟨i + 1, n - i⟩)

/-- The main theorem statement -/
theorem tenth_pair_20th_row :
  (generateRow 20).get? 9 = some ⟨10, 11⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_pair_20th_row_l1353_135379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_value_at_2_l1353_135366

-- Define the function q
noncomputable def q : ℝ → ℝ := sorry

-- State the theorem
theorem q_value_at_2 : q 2 = 5 := by
  -- Assume that (2,5) is on the graph of q
  have h1 : q 2 = 5 := sorry
  -- Assume that q(2) is an integer
  have h2 : ∃ n : ℤ, q 2 = n := sorry
  -- Prove that q(2) = 5
  exact h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_value_at_2_l1353_135366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1353_135367

theorem angle_in_second_quadrant (α : Real) 
  (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  ∃ x y : Real, x < 0 ∧ y > 0 ∧ Real.cos α = x ∧ Real.sin α = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_second_quadrant_l1353_135367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_shirt_price_l1353_135310

/-- Calculates the discounted price of a shirt given the original price and discount rate. -/
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

/-- Rounds a real number to the nearest integer. -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem discounted_shirt_price :
  round_to_nearest (discounted_price 955.88 0.32) = 650 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discounted_shirt_price_l1353_135310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1353_135347

def b (n k : ℕ+) (a : ℕ+ → ℕ) : ℕ := a n + a (n + k)

theorem sequence_properties (a : ℕ+ → ℕ) 
  (h1 : ∀ n : ℕ+, b n 2 a - b n 1 a = 1)
  (h2 : a 1 = 2)
  (h3 : ∀ n k : ℕ+, b (n + 1) k a = 2 * b n k a) :
  (∀ n : ℕ+, b n 4 a - b n 1 a = 3) ∧
  (∀ n : ℕ+, a n = 2^(n:ℕ)) ∧
  (∀ k : ℕ+, {b n k a | n : ℕ+} ∩ {5 * b n (k + 2) a | n : ℕ+} = ∅) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1353_135347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l1353_135332

/-- Given vectors in ℝ² -/
def α : Fin 2 → ℝ := ![1, -3]
def β : Fin 2 → ℝ := ![4, -2]

/-- Dot product of two 2D vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

/-- The statement to be proved -/
theorem perpendicular_condition :
  ∃! l : ℝ, dot_product (l • α + β) α = 0 ∧ l = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l1353_135332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_interior_surface_area_l1353_135389

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a triangle -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
noncomputable def rectangleArea (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (t.base * t.height) / 2

/-- Calculates the surface area of the interior of the box -/
noncomputable def interiorSurfaceArea (sheet : Rectangle) (corner : Triangle) : ℝ :=
  rectangleArea sheet - 4 * triangleArea corner

/-- The theorem to be proved -/
theorem box_interior_surface_area :
  let sheet : Rectangle := ⟨40, 25⟩
  let corner : Triangle := ⟨4, 4⟩
  interiorSurfaceArea sheet corner = 968 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_interior_surface_area_l1353_135389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l1353_135368

theorem product_of_roots (α β : ℝ) : 
  α > 0 → β > 0 → 
  (α^2 - Real.sqrt 13 * α^(Real.log α / Real.log 13) = 0) →
  (β^2 - Real.sqrt 13 * β^(Real.log β / Real.log 13) = 0) →
  α * β = 169 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l1353_135368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_length_l1353_135387

-- Define necessary structures and functions
structure Circle (α : Type*) where
  center : α × α
  radius : ℝ

structure Line (α : Type*) where
  point1 : α × α
  point2 : α × α

def ExternalTangent (c1 c2 : Circle ℝ) (l : Line ℝ) : Prop := sorry
def PerpendicularAt (l1 l2 : Line ℝ) (p : ℝ × ℝ) : Prop := sorry
def IntersectionPoint (l1 l2 : Line ℝ) : ℝ × ℝ := sorry
def DistanceBetweenTangencyPoints (l1 l2 : Line ℝ) : ℝ := sorry

theorem external_tangent_length (R r : ℝ) (h : R > r) :
  let L := 2 * Real.sqrt (R * r)
  ∃ (c₁ c₂ : Circle ℝ) (t₁ t₂ : Line ℝ),
    c₁.radius = R ∧
    c₂.radius = r ∧
    ExternalTangent c₁ c₂ t₁ ∧
    ExternalTangent c₁ c₂ t₂ ∧
    PerpendicularAt t₁ t₂ (IntersectionPoint t₁ t₂) ∧
    DistanceBetweenTangencyPoints t₁ t₂ = L :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_length_l1353_135387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_identification_l1353_135351

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop := ∃ (y : ℝ), y ≥ 0 ∧ x = Real.sqrt y

-- Theorem statement
theorem quadratic_radical_identification :
  is_quadratic_radical (Real.sqrt 3) ∧
  ¬ (∀ (a : ℝ), is_quadratic_radical (Real.sqrt a)) ∧
  ¬ is_quadratic_radical (Real.rpow 5 (1/3)) ∧
  ¬ is_quadratic_radical (Real.sqrt (-3/5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_identification_l1353_135351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_l1353_135376

/-- Regular hexagonal pyramid with inscribed sphere -/
structure HexagonalPyramid where
  /-- Base side length -/
  a : ℝ
  /-- Inscribed sphere radius -/
  r : ℝ
  /-- Base side length is positive -/
  a_pos : 0 < a
  /-- Inscribed sphere radius is positive -/
  r_pos : 0 < r

/-- Volume of a regular hexagonal pyramid with inscribed sphere -/
noncomputable def volume (pyramid : HexagonalPyramid) : ℝ :=
  (Real.sqrt 3 / 2) * pyramid.a^2 * pyramid.r

/-- Theorem: The volume of a regular hexagonal pyramid with base side length a
    and an inscribed sphere of radius r is (√3/2) * a^2 * r -/
theorem hexagonal_pyramid_volume (pyramid : HexagonalPyramid) :
    volume pyramid = (Real.sqrt 3 / 2) * pyramid.a^2 * pyramid.r := by
  sorry

#check hexagonal_pyramid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_l1353_135376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_points_to_identify_polynomial_l1353_135303

/-- A quadratic polynomial of the form ax^2 + bx + c -/
def QuadraticPolynomial (α : Type*) [Field α] := α → α

/-- Given two quadratic polynomials and a set of points, 
    determine if it's possible to identify one of the polynomials -/
def can_identify_polynomial (f g : QuadraticPolynomial ℝ) (points : Finset ℝ) : Prop :=
  ∃ (h : QuadraticPolynomial ℝ), (h = f ∨ h = g) ∧
    ∀ x ∈ points, ∃ y, (f x = y ∨ g x = y) →
      ∀ k : QuadraticPolynomial ℝ, (k = f ∨ k = g) →
        (∀ z ∈ points, k z = y ∨ (f z = y ∨ g z = y)) → k = h

theorem min_points_to_identify_polynomial :
  ∃ n : ℕ, (∀ (f g : QuadraticPolynomial ℝ) (points : Finset ℝ),
    points.card = n →
    can_identify_polynomial f g points) ∧
  (∀ m : ℕ, m < n →
    ∃ (f g : QuadraticPolynomial ℝ) (points : Finset ℝ),
      points.card = m ∧
      ¬can_identify_polynomial f g points) ∧
  n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_points_to_identify_polynomial_l1353_135303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_properties_l1353_135337

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem cos_2x_properties :
  (∃ (p : ℝ), p > 0 ∧ p ≤ π ∧ ∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (x : ℝ), f x = f (-x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_properties_l1353_135337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1353_135308

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arcsin (x/3))^2 - Real.pi * Real.arccos (x/3) + (Real.arccos (x/3))^2 + (Real.pi^2/18) * (x^2 - 3*x + 9)

theorem g_range : 
  Set.range g = Set.Icc (2*Real.pi^2/3) (5*Real.pi^2/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1353_135308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_iff_neg_one_one_in_second_quadrant_l1353_135320

/-- Definition: A point is in the second quadrant -/
def IsInSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- A point is in the second quadrant of the Cartesian coordinate system if and only if
    its x-coordinate is negative and its y-coordinate is positive. -/
theorem second_quadrant_iff (x y : ℝ) : 
  (x < 0 ∧ y > 0) ↔ IsInSecondQuadrant (x, y) := by
  sorry

/-- The point (-1, 1) is in the second quadrant -/
theorem neg_one_one_in_second_quadrant : 
  IsInSecondQuadrant (-1, 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_iff_neg_one_one_in_second_quadrant_l1353_135320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_mod_100_l1353_135364

/-- Sequence definition for a_n -/
def a : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n+2 => (n+2) * a (n+1)

/-- Theorem: The 9th term of the sequence is congruent to 80 modulo 100 -/
theorem a_9_mod_100 : a 9 ≡ 80 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_mod_100_l1353_135364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_l1353_135384

def my_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) - a n = 2 * n

theorem general_term (a : ℕ → ℕ) (h : my_sequence a) :
  ∀ n, a n = n^2 - n + 1 :=
by
  sorry

#check general_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_l1353_135384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sonikas_deposit_l1353_135398

/-- Represents the simple interest calculation --/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Represents the problem of finding Sonika's initial deposit --/
theorem sonikas_deposit (initial_amount : ℝ) (rate : ℝ) : 
  (initial_amount + simple_interest initial_amount rate 2 = 8400) →
  (initial_amount + simple_interest initial_amount (rate + 4) 2 = 8760) →
  initial_amount = 2250 := by
  sorry

#check sonikas_deposit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sonikas_deposit_l1353_135398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1353_135304

theorem range_of_m (p : ℝ → Prop) (h1 : ¬(p 1)) (h2 : p 2) : 
  ∃ m : ℝ, (∀ x : ℝ, p x ↔ x^2 + 2*x - m > 0) ∧ m ∈ Set.Icc 3 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1353_135304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_intersection_area_l1353_135369

/-- A cube with edge length a -/
structure Cube (a : ℝ) where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Point P that bisects edge AB -/
noncomputable def P (a : ℝ) : ℝ × ℝ × ℝ := (a/2, 0, 0)

/-- Point Q that divides edge AC into thirds -/
noncomputable def Q (a : ℝ) : ℝ × ℝ × ℝ := (a/3, a, 0)

/-- A plane passing through line PQ -/
structure Plane (a : ℝ) where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- The intersection of a plane with the cube -/
def intersection (a : ℝ) (cube : Cube a) (plane : Plane a) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- The area of a quadrilateral intersection -/
def intersectionArea (a : ℝ) (cube : Cube a) (plane : Plane a) : ℝ :=
  sorry

/-- The quadrilateral PQFK -/
def PQFK (a : ℝ) (cube : Cube a) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem stating that PQFK has the largest area among all quadrilateral intersections -/
theorem largest_intersection_area (a : ℝ) (cube : Cube a) :
  ∀ (plane : Plane a),
    intersectionArea a cube plane ≤ intersectionArea a cube (Plane.mk sorry sorry) ∧
    intersectionArea a cube (Plane.mk sorry sorry) = (a^2 * Real.sqrt 14) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_intersection_area_l1353_135369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_ST_l1353_135313

-- Define the isosceles triangle PQR
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

-- Define the line segment ST
structure LineSegment where
  length : ℝ

-- Define the isosceles trapezoid formed by ST
structure Trapezoid where
  area : ℝ

-- Define the smaller isosceles triangle formed by ST
structure SmallTriangle where
  area : ℝ

-- Define the main triangle PQR
def triangle_PQR : IsoscelesTriangle := ⟨12, 30⟩

-- Define the line segment ST
def segment_ST : LineSegment := ⟨6⟩

-- Define the trapezoid
def trapezoid : Trapezoid := ⟨135⟩

-- Define the smaller triangle
def small_triangle : SmallTriangle := ⟨45⟩

-- Theorem statement
theorem length_of_ST (h1 : triangle_PQR.base = 12) 
                     (h2 : triangle_PQR.height = 30)
                     (h3 : 0.5 * triangle_PQR.base * triangle_PQR.height = 180)
                     (h4 : trapezoid.area = 135)
                     (h5 : small_triangle.area = 180 - trapezoid.area)
                     (h6 : small_triangle.area / 180 = 1 / 4)
                     (h7 : segment_ST.length = triangle_PQR.base / 2) : 
  segment_ST.length = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_ST_l1353_135313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1353_135330

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 17) / Real.log (1/2)

-- State the theorem
theorem f_range : 
  Set.range f = Set.Iic (-3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1353_135330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_perimeter_sum_l1353_135370

noncomputable def heptagon_vertices : List (ℝ × ℝ) :=
  [(0, 1), (1, 2), (2, 2), (2, 1), (3, 0), (3, -1), (2, -2), (0, 1)]

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def perimeter (vertices : List (ℝ × ℝ)) : ℝ :=
  List.sum (List.zipWith distance vertices (vertices.tail ++ [vertices.head!]))

theorem heptagon_perimeter_sum (a b c d : ℤ) :
  perimeter heptagon_vertices = a + b * Real.sqrt 2 + c * Real.sqrt 3 + d * Real.sqrt 5 →
  a + b + c + d = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_perimeter_sum_l1353_135370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_prime_l1353_135348

theorem divisibility_implies_prime (m : ℕ) (h : m > 1) :
  ((Nat.factorial (m - 1) + 1) % m = 0) → Nat.Prime m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_prime_l1353_135348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_billable_minutes_l1353_135343

-- Define the parameters
def monthly_fee : ℚ := 5
def per_minute_rate : ℚ := 1/4
def total_bill : ℚ := 12.02

-- Define the function to calculate billable minutes
def billable_minutes (fee : ℚ) (rate : ℚ) (bill : ℚ) : ℕ :=
  (((bill - fee) / rate).floor : ℤ).natAbs

-- Theorem statement
theorem john_billable_minutes :
  billable_minutes monthly_fee per_minute_rate total_bill = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_billable_minutes_l1353_135343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_L_is_correct_l1353_135358

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Calculate the distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (l1 l2 : Line) : ℝ :=
  abs (l2.b - l1.b) / Real.sqrt (l1.m^2 + 1)

/-- The given line y = 3/4x + 6 -/
noncomputable def given_line : Line := { m := 3/4, b := 6 }

/-- The line we want to prove is correct -/
noncomputable def line_L : Line := { m := 3/4, b := 1 }

theorem line_L_is_correct : 
  (line_L.m = given_line.m) ∧ 
  (distance_between_parallel_lines given_line line_L = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_L_is_correct_l1353_135358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_saved_driving_faster_l1353_135336

/-- Calculates the time taken to travel a given distance at a given speed -/
noncomputable def travelTime (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- The problem statement -/
theorem time_saved_driving_faster (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : distance = 1200)
  (h2 : speed1 = 50)
  (h3 : speed2 = 60) :
  travelTime distance speed1 - travelTime distance speed2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_saved_driving_faster_l1353_135336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_length_l1353_135323

/-- The length of the tangent line from the origin to a circle -/
theorem tangent_line_length (x y : ℝ) :
  (x - 6)^2 + y^2 = 4 →
  ∃ t : ℝ, t^2 = 32 ∧ t = Real.sqrt (x^2 + y^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_length_l1353_135323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l1353_135316

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x / 6 - Real.pi / 3)

theorem sum_of_max_min_f :
  (⨆ x ∈ Set.Icc (0 : ℝ) 9, f x) + (⨅ x ∈ Set.Icc (0 : ℝ) 9, f x) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l1353_135316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_intercept_l1353_135331

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- The line equation in vector form -/
def line_equation (p : Vector2D) : Prop :=
  dot_product ⟨3, -4⟩ (Vector2D.mk (p.x - 2) (p.y + 3)) = 0

/-- The slope-intercept form of a line -/
def slope_intercept_form (m b : ℝ) (p : Vector2D) : Prop :=
  p.y = m * p.x + b

theorem line_slope_intercept :
  ∀ p : Vector2D, line_equation p ↔ slope_intercept_form (3/4) (-4.5) p := by
  sorry

#check line_slope_intercept

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_intercept_l1353_135331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_problem_l1353_135350

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the focus
def focus : ℝ × ℝ := (3, 0)

-- Define the point M on the parabola
def point_on_parabola (x y : ℝ) : Prop := parabola x y

-- Define the distance between M and F
noncomputable def distance_MF (x y : ℝ) : ℝ := Real.sqrt ((x - 3)^2 + y^2)

-- Define the area of triangle OMF
noncomputable def area_OMF (x y : ℝ) : ℝ := (1/2) * 3 * abs y

theorem parabola_problem (x y : ℝ) :
  point_on_parabola x y ∧ distance_MF x y = 5 →
  x = 8 ∧ area_OMF x y = 6 * Real.sqrt 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_problem_l1353_135350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1353_135349

-- Define the sequence a_n
def a : ℕ → ℤ
  | 0 => -3  -- Add this case to handle n = 0
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + 3

-- State the theorem
theorem sequence_formula (n : ℕ) : a n = 2^(n+1) - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1353_135349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foma_wait_probability_l1353_135392

-- Define the arrival times as real numbers
def ivan_arrival (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 10

def foma_arrival (x y : ℝ) : Prop := x < y ∧ y ≤ 10

-- Define the condition for Foma waiting no more than 4 minutes
def foma_wait_time (x y : ℝ) : Prop := y - x ≤ 4

-- Define the probability measure on the sample space
noncomputable def probability_measure : Set (ℝ × ℝ) → ℝ := sorry

-- State the theorem
theorem foma_wait_probability :
  probability_measure {p : ℝ × ℝ | ivan_arrival p.1 ∧ foma_arrival p.1 p.2 ∧ foma_wait_time p.1 p.2} =
  (3/4) * probability_measure {p : ℝ × ℝ | ivan_arrival p.1 ∧ foma_arrival p.1 p.2} := by
  sorry

#check foma_wait_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_foma_wait_probability_l1353_135392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1353_135380

/-- A four-digit number with ordered digits -/
structure OrderedFourDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  is_four_digit : 1000 ≤ a * 1000 + b * 100 + c * 10 + d
  is_ordered : a ≤ b ∧ b ≤ c ∧ c ≤ d
  are_digits : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9

/-- The expression to be maximized -/
def expression (n : OrderedFourDigitNumber) : ℕ :=
  (n.d - n.a) + (n.d - n.a)

/-- The theorem stating the maximum value of the expression -/
theorem max_expression_value :
  ∀ n : OrderedFourDigitNumber, expression n ≤ 16 ∧ ∃ m : OrderedFourDigitNumber, expression m = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1353_135380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_students_l1353_135319

/-- Represents a class at Pythagoras Academy -/
inductive PythagorasClass
  | Euler
  | Ramanujan
  | Gauss

/-- The number of students in each class -/
def class_size (c : PythagorasClass) : Nat :=
  match c with
  | PythagorasClass.Euler => 15
  | PythagorasClass.Ramanujan => 12
  | PythagorasClass.Gauss => 10

/-- The number of students in both Euler's and Ramanujan's classes -/
def euler_ramanujan_overlap : Nat := 3

/-- The number of students in both Ramanujan's and Gauss's classes -/
def ramanujan_gauss_overlap : Nat := 2

/-- The theorem stating the number of distinct students taking the contest -/
theorem distinct_students :
  (class_size PythagorasClass.Euler + class_size PythagorasClass.Ramanujan + class_size PythagorasClass.Gauss) -
  (euler_ramanujan_overlap + ramanujan_gauss_overlap) = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_students_l1353_135319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1353_135339

noncomputable section

-- Define the hyperbola C
def hyperbola_C (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1 ∧ a > 0 ∧ b > 0

-- Define point B
def point_B (b : ℝ) : ℝ × ℝ := (0, (Real.sqrt 15 / 3) * b)

-- Define the eccentricity
def eccentricity (c a : ℝ) : ℝ := c / a

-- Theorem statement
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (h_hyperbola : ∀ x y, hyperbola_C a b x y → x^2 / a^2 - y^2 / b^2 = 1)
  (h_left_vertex : ∃ A : ℝ × ℝ, A.1 = -a ∧ A.2 = 0)
  (h_point_B : point_B b = (0, (Real.sqrt 15 / 3) * b))
  (h_perpendicular_bisector : ∃ F : ℝ × ℝ, F.1 > 0 ∧ 
    -- The perpendicular bisector of AB passes through F
    (F.1 - (-a))^2 + F.2^2 = (F.1 - 0)^2 + (F.2 - (Real.sqrt 15 / 3) * b)^2)
  (h_focus : c^2 = a^2 + b^2) :
  eccentricity c a = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1353_135339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1353_135317

/-- The equation of the tangent line to the curve y = 4x - x³ at the point (-1, -3) is y = x - 2 -/
theorem tangent_line_equation : 
  let f (x : ℝ) := 4*x - x^3
  let p : ℝ × ℝ := (-1, -3)
  let tangent_line (x : ℝ) := x - 2
  (∀ x, (tangent_line x - f p.1) = (deriv f p.1) * (x - p.1)) ∧ 
  f p.1 = p.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1353_135317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_free_bound_l1353_135399

/-- A natural number is squarefree if it's not divisible by any perfect square greater than 1. -/
def Nat.squarefree (n : ℕ) : Prop :=
  ∀ (d : ℕ), d > 1 → d^2 ∣ n → d = 1

theorem factorial_square_free_bound (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    ∀ a b : ℕ, (Nat.squarefree a ∧ n.factorial = a * b^2) →
      (2 : ℝ)^((1 - ε) * n) < a ∧ a < (2 : ℝ)^((1 + ε) * n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_square_free_bound_l1353_135399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l1353_135382

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^(-(6:ℤ)) - b^(-(6:ℤ))) / (a^(-(3:ℤ)) - b^(-(3:ℤ))) = a^(-(6:ℤ)) + a^(-(3:ℤ)) * b^(-(3:ℤ)) + b^(-(6:ℤ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_simplification_l1353_135382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1353_135394

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1/4, 0)

-- Define the point M on the parabola
def M (y₀ : ℝ) : ℝ × ℝ := (y₀^2, y₀)

-- Define the line x = t
def vertical_line (t : ℝ) (x : ℝ) : Prop := x = t

-- Define the intersections A and B
def A (t y₀ yA : ℝ) : ℝ × ℝ := (t, yA)
def B (t y₀ yB : ℝ) : ℝ × ℝ := (t, yB)

-- Define points P and Q
def P : ℝ × ℝ := (1, 1)
def Q : ℝ × ℝ := (1, -1)

theorem parabola_properties :
  -- Part 1
  (∀ y₀ : ℝ, y₀ = Real.sqrt 2 → ‖M y₀ - focus‖ = 9/4) ∧
  -- Part 2
  (∀ y₀ yA yB : ℝ, 
    vertical_line (-1) (A (-1) y₀ yA).1 ∧
    vertical_line (-1) (B (-1) y₀ yB).1 ∧
    parabola P.1 P.2 ∧ parabola Q.1 Q.2 →
    yA * yB = -1) ∧
  -- Part 3
  (∃ t : ℝ, t = 1 ∧
    (∀ y₀ yA yB yP yQ : ℝ,
      vertical_line t (A t y₀ yA).1 ∧
      vertical_line t (B t y₀ yB).1 ∧
      parabola (M y₀).1 (M y₀).2 →
      yA * yB = 1 ∧ yP * yQ = 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1353_135394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_perpendicular_l1353_135357

theorem vectors_perpendicular (a b : ℝ × ℝ × ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ‖a + b‖ = ‖a - b‖ → a • b = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_perpendicular_l1353_135357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_l1353_135306

theorem trigonometric_equation (A ω φ b : ℝ) (h : A > 0) :
  (∀ x, 2 * (Real.cos x)^2 + Real.sin (2*x) = A * Real.sin (ω*x + φ) + b) →
  A = Real.sqrt 2 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_l1353_135306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_negative_three_to_five_l1353_135360

/-- A random variable following a normal distribution with mean 1 and variance 4 -/
noncomputable def ξ : Real → Real := sorry

/-- The probability density function of the standard normal distribution -/
noncomputable def standard_normal_pdf : Real → Real := sorry

/-- The cumulative distribution function of the standard normal distribution -/
noncomputable def standard_normal_cdf : Real → Real := sorry

/-- The probability that a standard normal random variable is between -1 and 1 -/
axiom prob_within_one_sigma : standard_normal_cdf 1 - standard_normal_cdf (-1) = 0.6826

/-- The probability that a standard normal random variable is between -2 and 2 -/
axiom prob_within_two_sigma : standard_normal_cdf 2 - standard_normal_cdf (-2) = 0.9544

/-- The probability that a standard normal random variable is between -3 and 3 -/
axiom prob_within_three_sigma : standard_normal_cdf 3 - standard_normal_cdf (-3) = 0.9974

/-- The mean of the normal distribution ξ follows -/
def μ : ℝ := 1

/-- The standard deviation of the normal distribution ξ follows -/
def σ : ℝ := 2

/-- ξ follows a normal distribution with mean μ and standard deviation σ -/
axiom ξ_distribution : ∀ (x : ℝ), ξ x = standard_normal_pdf ((x - μ) / σ) / σ

/-- The main theorem to prove -/
theorem prob_negative_three_to_five : 
  standard_normal_cdf ((5 - μ) / σ) - standard_normal_cdf ((-3 - μ) / σ) = 0.9544 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_negative_three_to_five_l1353_135360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_cells_fraction_l1353_135321

theorem marked_cells_fraction (a b N : ℕ) (h1 : a < b) (h2 : b < 2 * a) :
  let α : ℝ := 1 / (2 * a^2 - 2 * a * b + b^2)
  ∀ (marking : ℕ → ℕ → Bool),
    (∀ (x y : ℕ), ∃ (i j : ℕ), i < a ∧ j < b ∧ marking (x + i) (y + j) = true) →
    (∀ (x y : ℕ), ∃ (i j : ℕ), i < b ∧ j < a ∧ marking (x + i) (y + j) = true) →
    ∃ (marked_count : ℕ), 
      (marked_count : ℝ) ≥ α * N^2 ∧
      marked_count = (Finset.sum (Finset.range N) (λ i => 
        Finset.sum (Finset.range N) (λ j => if marking i j then 1 else 0))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_cells_fraction_l1353_135321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_tangent_points_l1353_135386

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define an arc
structure Arc where
  circle : Circle
  start : Point
  stop : Point

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

-- Define the power of a point with respect to a circle
noncomputable def powerOfPoint (p : Point) (c : Circle) : ℝ := sorry

-- Define the locus of points
def isInLocus (x : Point) (arc : Arc) : Prop :=
  let c := arc.circle
  let r := c.radius
  (distance x c.center > r ∨ distance x c.center = r) ∧
  (distance x arc.start = r ∨ distance x arc.stop = r) ∧
  powerOfPoint x c = (distance x c.center)^2 - r^2

-- Theorem statement
theorem locus_of_tangent_points (arc : Arc) (x : Point) :
  isInLocus x arc ↔ 
    ∃ (p q : Point), p ≠ q ∧ 
      distance p arc.circle.center = arc.circle.radius ∧
      distance q arc.circle.center = arc.circle.radius ∧
      (distance x p)^2 = powerOfPoint x arc.circle ∧
      (distance x q)^2 = powerOfPoint x arc.circle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_tangent_points_l1353_135386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_equation_l1353_135381

open Nat

theorem solution_count_equation : 
  ∃ (solutions : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ solutions ↔ 20 * m + 12 * n = 2012 ∧ m > 0 ∧ n > 0) ∧
    Finset.card solutions = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_equation_l1353_135381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_property_l1353_135335

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 6)

-- State the theorem
theorem inverse_g_property :
  ∃ (a b c d : ℝ), 
    (∀ x, x ≠ 3 → g⁻¹ x = (a * x + b) / (c * x + d)) ∧ 
    (a / c = -6) := by
  -- We'll use existence introduction to provide the values
  use (-6 : ℝ), (-2 : ℝ), (1 : ℝ), (-3 : ℝ)
  -- The proof goes here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_property_l1353_135335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_40_deg_l1353_135374

theorem trig_identity_40_deg : 
  (Real.tan (40 * π / 180))^2 - (Real.sin (40 * π / 180))^2 = 
  (Real.tan (40 * π / 180))^2 * (Real.sin (40 * π / 180))^2 := by
  sorry

#check trig_identity_40_deg

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_40_deg_l1353_135374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l1353_135397

-- Define the basic geometric objects
structure Plane : Type
structure Line : Type

-- Define the relationships between planes and lines
axiom perpendicular : Plane → Plane → Prop
axiom parallel_planes : Plane → Plane → Prop
axiom parallel_line_plane : Line → Plane → Prop
axiom parallel_lines : Line → Line → Prop
axiom line_in_plane : Line → Plane → Prop
axiom projection : Line → Plane → Line

-- Define the propositions
def proposition1 (α β : Plane) : Prop :=
  perpendicular α β ↔ ∃ l, line_in_plane l α ∧ perpendicular (Plane.mk) β

def proposition2 (α β : Plane) : Prop :=
  (∃ (S : Set Line), Set.Infinite S ∧ ∀ l ∈ S, line_in_plane l α ∧ parallel_line_plane l β) →
  parallel_planes α β

def proposition3 (a : Line) (α : Plane) : Prop :=
  (∃ l, line_in_plane l α ∧ parallel_lines a l) →
  parallel_line_plane a α

def proposition4 (a b : Line) (α : Plane) : Prop :=
  ¬(parallel_lines (projection a α) (projection b α) ↔ parallel_lines a b)

-- The main theorem
theorem geometric_propositions :
  ∃ (α β : Plane) (a b : Line),
    proposition1 α β ∧ proposition2 α β ∧ ¬proposition3 a α ∧ proposition4 a b α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_propositions_l1353_135397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_is_zero_l1353_135312

-- Define the constants as noncomputable
noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10

-- State the theorem
theorem sum_of_reciprocals_squared_is_zero : 
  (1/a + 1/b + 1/c + 1/d)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_squared_is_zero_l1353_135312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_lower_bound_l1353_135359

def f (n : ℕ) : ℚ := (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

theorem harmonic_sum_lower_bound (n : ℕ) (h : n > 1) : 
  f (2^n) ≥ (n + 2 : ℚ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_sum_lower_bound_l1353_135359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_in_interval_l1353_135383

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

-- State the theorem
theorem max_value_f_in_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  f x = π / 6 + Real.sqrt 3 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ f x := by
  sorry

#check max_value_f_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_in_interval_l1353_135383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1353_135385

/-- The number of days it takes for A and B to complete a work together, given that A is thrice as fast as B and A alone can do the work in 20 days. -/
noncomputable def days_to_complete_work (a_rate : ℝ) (b_rate : ℝ) : ℝ :=
  1 / (a_rate + b_rate)

/-- Theorem stating that A and B can complete the work in 15 days under the given conditions. -/
theorem work_completion_time :
  ∀ (a_rate : ℝ) (b_rate : ℝ),
    a_rate > 0 →
    b_rate > 0 →
    a_rate = 3 * b_rate →
    a_rate = 1 / 20 →
    days_to_complete_work a_rate b_rate = 15 :=
by
  intros a_rate b_rate h1 h2 h3 h4
  unfold days_to_complete_work
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1353_135385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_isosceles_solutions_l1353_135363

/-- A function that returns the number of integer values of x that satisfy
    the conditions for a non-degenerate isosceles triangle with two sides
    of length x and one side of length 24. -/
def isosceles_triangle_solutions : ℕ :=
  Finset.card (Finset.filter (fun x => x > 12 ∧ x < 24 ∧ 2 * x > 24 ∧ x + 24 > x) (Finset.range 24))

/-- Theorem stating that there are exactly 11 integer solutions for x -/
theorem eleven_isosceles_solutions : isosceles_triangle_solutions = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_isosceles_solutions_l1353_135363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_range_l1353_135305

-- Define the line l
noncomputable def line_l (α : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, t * Real.sin α)

-- Define the curve C in polar form
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 4*ρ*(Real.cos θ) + 3 = 0

-- Define the range of α
def α_range (α : ℝ) : Prop :=
  0 < α ∧ α < Real.pi/6

-- State the theorem
theorem intersection_distance_range (α : ℝ) (h : α_range α) :
  ∃ (A B : ℝ × ℝ),
    (∃ (t₁ t₂ : ℝ), line_l α t₁ = A ∧ line_l α t₂ = B) ∧
    (∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ), 
      curve_C ρ₁ θ₁ ∧ curve_C ρ₂ θ₂ ∧
      (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁) = A ∧
      (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂) = B) ∧
    2 * Real.sqrt 3 < Real.sqrt (A.1^2 + A.2^2) + Real.sqrt (B.1^2 + B.2^2) ∧
    Real.sqrt (A.1^2 + A.2^2) + Real.sqrt (B.1^2 + B.2^2) < 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_range_l1353_135305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_vector_magnitude_l1353_135346

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Circle type -/
structure Circle where
  equation : ℝ → ℝ → Prop
  center : ℝ × ℝ

/-- Point type -/
def Point := ℝ × ℝ

/-- Vector between two points -/
def vector (p q : Point) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Magnitude of a vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- Main theorem -/
theorem parabola_circle_intersection_vector_magnitude 
  (p : Parabola) 
  (c : Circle) 
  (A B : Point) : 
  p.equation = fun x y => y^2 = 8*x →
  c.equation = fun x y => x^2 + y^2 + 2*x - 8 = 0 →
  c.center = (-1, 0) →
  (∃ x, p.equation x A.2 ∧ c.equation x A.2 ∧ 
        p.equation x B.2 ∧ c.equation x B.2) →
  magnitude (vector A B) = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_vector_magnitude_l1353_135346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_three_classes_l1353_135302

/-- Represents a math class with its number of students and total score. -/
structure MathClass where
  students : ℚ
  totalScore : ℚ

/-- Calculates the average score of a math class. -/
def averageScore (c : MathClass) : ℚ :=
  c.totalScore / c.students

/-- Calculates the combined average score of two math classes. -/
def combinedAverage (c1 c2 : MathClass) : ℚ :=
  (c1.totalScore + c2.totalScore) / (c1.students + c2.students)

theorem average_of_three_classes
  (x y z : MathClass)
  (hx : averageScore x = 83)
  (hy : averageScore y = 76)
  (hz : averageScore z = 85)
  (hxy : combinedAverage x y = 79)
  (hyz : combinedAverage y z = 81) :
  (x.totalScore + y.totalScore + z.totalScore) / (x.students + y.students + z.students) = 81.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_three_classes_l1353_135302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_product_l1353_135345

def v_p (p a : ℕ) : ℕ := Nat.factorization a p

theorem gcd_lcm_product (p m n : ℕ) (hp : Nat.Prime p) :
  (∀ q : ℕ, Nat.Prime q → v_p q (Nat.gcd m n) = min (v_p q m) (v_p q n)) ∧
  (∀ q : ℕ, Nat.Prime q → v_p q (Nat.lcm m n) = max (v_p q m) (v_p q n)) ∧
  Nat.gcd m n * Nat.lcm m n = m * n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_product_l1353_135345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_product_inequality_l1353_135300

theorem sine_product_inequality (x : ℝ) : 
  Real.sin x * Real.sin (1755 * x) * Real.sin (2011 * x) ≥ 1 ↔ 
  ∃ n : ℤ, x = π / 2 + 2 * π * ↑n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_product_inequality_l1353_135300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_inequality_equality_condition_l1353_135322

open Real BigOperators

theorem power_sum_inequality (n : ℕ) (a b : Fin n → ℝ) (m : ℝ) 
  (h_pos : ∀ i, a i > 0 ∧ b i > 0) (h_m : m > 0 ∨ m < -1) :
  ∑ i, (a i ^ (m + 1)) / (b i ^ m) ≥ 
  (∑ i, a i) ^ (m + 1) / (∑ i, b i) ^ m := by
  sorry

theorem equality_condition (n : ℕ) (a b : Fin n → ℝ) (m : ℝ) 
  (h_pos : ∀ i, a i > 0 ∧ b i > 0) (h_m : m > 0 ∨ m < -1) :
  (∑ i, (a i ^ (m + 1)) / (b i ^ m) = 
   (∑ i, a i) ^ (m + 1) / (∑ i, b i) ^ m) ↔ 
  (∃ c : ℝ, ∀ i, a i / b i = c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_inequality_equality_condition_l1353_135322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_99_50_l1353_135396

/-- The infinite series defined in the problem -/
noncomputable def series_sum : ℝ := ∑' n, if n ≥ 2 then (n^4 + 4*n^2 + 15*n + 15) / (2^n * (n^4 + 9)) else 0

/-- The theorem stating that the sum of the infinite series is equal to 99/50 -/
theorem series_sum_equals_99_50 : series_sum = 99/50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_99_50_l1353_135396
