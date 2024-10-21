import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_specific_vectors_l1315_131558

/-- The cross product of two specific vectors is equal to a specific result vector. -/
theorem cross_product_specific_vectors :
  let v₁ : Fin 3 → ℝ := ![4, -3, 5]
  let v₂ : Fin 3 → ℝ := ![2, -2, 7]
  let result : Fin 3 → ℝ := ![-11, -18, -2]
  (v₁ 1 * v₂ 2 - v₁ 2 * v₂ 1,
   v₁ 2 * v₂ 0 - v₁ 0 * v₂ 2,
   v₁ 0 * v₂ 1 - v₁ 1 * v₂ 0) = (result 0, result 1, result 2) := by
  sorry

#check cross_product_specific_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_specific_vectors_l1315_131558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_2_minus_x_l1315_131500

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (2 - x)

-- State the theorem
theorem domain_of_log_2_minus_x :
  {x : ℝ | ∃ y, f x = y} = Set.Iio 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_2_minus_x_l1315_131500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_problem_minimum_value_minimum_value_achievable_l1315_131514

-- Problem 1
theorem factorization_problem (m n : ℝ) : m^2 - 4*m*n + 3*n^2 = (m - 3*n) * (m - n) := by sorry

-- Problem 2
theorem minimum_value (m : ℝ) : m^2 - 3*m + 2015 ≥ 2012 + 3/4 := by sorry

theorem minimum_value_achievable : ∃ m : ℝ, m^2 - 3*m + 2015 = 2012 + 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_problem_minimum_value_minimum_value_achievable_l1315_131514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_symmetric_g_l1315_131553

/-- The function f(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x)

/-- The function g(x) obtained by translating f(x) left by φ units -/
noncomputable def g (x φ : ℝ) : ℝ := f (x + φ)

/-- Symmetry condition for g(x) about the y-axis -/
def is_symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

theorem min_phi_for_symmetric_g :
  ∀ φ : ℝ, φ > 0 →
  is_symmetric_about_y_axis (g · φ) →
  (∀ ψ : ℝ, ψ > 0 → is_symmetric_about_y_axis (g · ψ) → φ ≤ ψ) →
  φ = 5 * Real.pi / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_symmetric_g_l1315_131553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1315_131541

/-- The sum of the first n terms of a geometric sequence. -/
def S (n : ℕ) : ℝ := sorry

/-- 
Theorem: For a geometric sequence with sum S_n, if S_10 = 10 and S_20 = 30, then S_30 = 70.
-/
theorem geometric_sequence_sum :
  S 10 = 10 → S 20 = 30 → S 30 = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1315_131541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1315_131567

/-- Given two workers who can complete a task individually, calculate the time they take to complete the task together -/
theorem work_completion_time (a_time b_time : ℝ) (ha : a_time > 0) (hb : b_time > 0) :
  (a_time = 10 ∧ b_time = 15) →
  (1 / (1 / a_time + 1 / b_time)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1315_131567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_restricted_polynomial_l1315_131533

/-- A polynomial with coefficients in {0, -1, 1} -/
def RestrictedPoly (R : Type*) [Ring R] := {p : Polynomial R // ∀ i, p.coeff i ∈ ({0, -1, 1} : Set R)}

theorem existence_of_restricted_polynomial (n : ℕ) :
  ∃ (P : RestrictedPoly ℚ), 
    P.val ≠ 0 ∧ 
    P.val.degree ≤ 2^n ∧ 
    (Polynomial.X - 1 : Polynomial ℚ)^n ∣ P.val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_restricted_polynomial_l1315_131533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_analysis_l1315_131589

def selling_price (a : ℝ) (x : ℝ) : ℝ := a + |x - 20|

def sales_volume (x : ℝ) : ℝ := 50 - |x - 16|

def sales_revenue (a : ℝ) (x : ℝ) : ℝ := selling_price a x * sales_volume x

theorem sales_analysis 
  (a : ℝ)
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 30)
  (h2 : sales_revenue a 18 = 2016) :
  a = 40 ∧ 
  sales_revenue a 20 = 1840 ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 30 → sales_revenue a x ≤ sales_revenue a 13) ∧
  sales_revenue a 13 = 2209 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_analysis_l1315_131589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_path_increase_l1315_131525

/-- Given a linear path where y increases by 6 units for every 4 units increase in x,
    prove that y increases by 18 units when x increases by 12 units. -/
theorem linear_path_increase (f : ℝ → ℝ) (h : ∀ x, f (x + 4) - f x = 6) :
  ∀ x, f (x + 12) - f x = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_path_increase_l1315_131525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l1315_131563

/-- The lower bound of the domain -/
noncomputable def lower_bound : ℝ := 1/2

/-- The upper bound of the domain -/
noncomputable def upper_bound : ℝ := 2

/-- The function f(x) = a/x + x ln x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a/x + x * Real.log x

/-- The function g(x) = x³ - x² - 5 -/
def g (x : ℝ) : ℝ := x^3 - x^2 - 5

/-- The main theorem -/
theorem a_lower_bound (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, lower_bound ≤ x₁ ∧ x₁ ≤ upper_bound ∧ 
                lower_bound ≤ x₂ ∧ x₂ ≤ upper_bound → 
                f a x₁ - g x₂ ≥ 2) → 
  a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l1315_131563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_series_l1315_131596

def a (n : ℕ+) : ℚ := ((-1 : ℤ)^(n.val + 1 : ℕ) * n.val) / (n.val^2 + 1 : ℚ)

theorem tenth_term_of_series : a 10 = -10 / 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_series_l1315_131596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l1315_131552

/-- The distance of a race where A beats B by 100 meters -/
def D : ℝ := sorry

/-- The ratio of distances covered by C and B in an 800-meter race where B beats C by 100 meters -/
noncomputable def ratio : ℝ := 700 / 800

theorem race_distance : D = 1000 := by
  have h1 : ratio * (D - 100) = D - 212.5 := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l1315_131552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_formula_l1315_131547

/-- The area of a sector with given arc length and radius -/
noncomputable def sectorArea (l r : ℝ) : ℝ := (1/2) * l * r

/-- Theorem: The area of a sector with arc length l and radius r is (1/2)lr -/
theorem sector_area_formula (l r : ℝ) (h1 : l > 0) (h2 : r > 0) :
  sectorArea l r = (1/2) * l * r :=
by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- The rest of the proof is trivial since it's just the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_formula_l1315_131547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1315_131543

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

/-- Theorem: For a geometric sequence with first term a and common ratio r,
    if S_3 = 7 and S_6 = 63, then a = 1 -/
theorem geometric_sequence_first_term
  (a : ℝ) (r : ℝ) (h1 : r ≠ 1) (h2 : geometricSum a r 3 = 7) (h3 : geometricSum a r 6 = 63) :
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1315_131543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skaters_meeting_distance_l1315_131574

/-- The distance between two skaters on a flat, frozen lake --/
def distance_AB : ℝ := 150

/-- Allie's skating speed in meters per second --/
def speed_Allie : ℝ := 8

/-- Billie's skating speed in meters per second --/
def speed_Billie : ℝ := 7

/-- The angle between Allie's path and line AB in radians --/
noncomputable def angle_Allie : ℝ := Real.pi / 4

/-- The time at which Allie and Billie meet --/
def meeting_time : ℝ := 40

theorem skaters_meeting_distance :
  speed_Allie * meeting_time = 320 := by
  -- The proof goes here
  sorry

#eval speed_Allie * meeting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skaters_meeting_distance_l1315_131574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1315_131528

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x-2)^2 + (y+3)^2 = 9

-- Define the intersection points
def intersection_points (E F : ℝ × ℝ) : Prop :=
  line E.1 E.2 ∧ circle_eq E.1 E.2 ∧
  line F.1 F.2 ∧ circle_eq F.1 F.2 ∧
  E ≠ F

-- Theorem statement
theorem chord_length (E F : ℝ × ℝ) : 
  intersection_points E F → dist E F = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1315_131528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_document_retyping_reduction_l1315_131503

/-- Calculates the percentage reduction in sheets when retyping a document -/
theorem document_retyping_reduction (original_sheets : ℕ) (original_lines_per_sheet : ℕ) 
  (original_chars_per_line : ℕ) (new_lines_per_sheet : ℕ) (new_chars_per_line : ℕ) :
  let total_chars := original_sheets * original_lines_per_sheet * original_chars_per_line
  let chars_per_new_sheet := new_lines_per_sheet * new_chars_per_line
  let new_sheets := (total_chars + chars_per_new_sheet - 1) / chars_per_new_sheet
  let reduction := original_sheets - new_sheets
  let percentage_reduction := (reduction : ℝ) / (original_sheets : ℝ) * 100
  original_sheets = 750 ∧ 
  original_lines_per_sheet = 150 ∧ 
  original_chars_per_line = 200 ∧
  new_lines_per_sheet = 250 ∧
  new_chars_per_line = 220 →
  45.32 < percentage_reduction ∧ percentage_reduction < 45.34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_document_retyping_reduction_l1315_131503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_a_upper_bound_l1315_131551

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x + 1 - Real.log x

-- Part I
theorem interval_of_increase (x : ℝ) :
  x ∈ Set.Ioo (1/2 : ℝ) 1 ↔ (deriv (f 3)) x > 0 :=
by sorry

-- Part II
theorem a_upper_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2 : ℝ), (deriv (f a)) x ≤ 0) →
  a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_a_upper_bound_l1315_131551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_rate_l1315_131591

/-- The rate of drawing barbed wire per meter for a square field -/
theorem barbed_wire_rate (area : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  area = 3136 →
  total_cost = 666 →
  gate_width = 1 →
  num_gates = 2 →
  (total_cost / (4 * Real.sqrt area - gate_width * (num_gates : ℝ))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbed_wire_rate_l1315_131591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_range_l1315_131512

noncomputable section

/-- Piecewise function definition -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + (4*a - 3)*x + 3*a
  else if 0 ≤ x ∧ x < Real.pi/2 then -Real.sin x
  else 0  -- undefined for x ≥ π/2

/-- Monotonically decreasing function -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem monotonic_decreasing_range (a : ℝ) :
  MonoDecreasing (f a) → a ∈ Set.Icc 0 (4/3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_range_l1315_131512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_arrangements_count_dianes_stamp_arrangements_l1315_131564

/-- Represents the number of stamps of each denomination -/
structure StampInventory where
  one_cent : Nat
  two_cent : Nat
  four_cent : Nat
  ten_cent : Nat

/-- Represents a combination of stamps -/
structure StampCombination where
  one_cent : Nat
  two_cent : Nat
  four_cent : Nat
  ten_cent : Nat

/-- Calculates the total value of a stamp combination in cents -/
def combinationValue (c : StampCombination) : Nat :=
  c.one_cent + 2 * c.two_cent + 4 * c.four_cent + 10 * c.ten_cent

/-- Checks if a stamp combination is valid given the inventory -/
def isValidCombination (inv : StampInventory) (c : StampCombination) : Prop :=
  c.one_cent ≤ inv.one_cent ∧
  c.two_cent ≤ inv.two_cent ∧
  c.four_cent ≤ inv.four_cent ∧
  c.ten_cent ≤ inv.ten_cent

/-- Calculates the number of unique arrangements for a given stamp combination -/
def uniqueArrangements (c : StampCombination) : Nat :=
  sorry

/-- The main theorem stating that there are exactly 45 unique arrangements -/
theorem stamp_arrangements_count (inv : StampInventory) : Nat :=
  sorry

/-- The specific inventory given in the problem -/
def dianeInventory : StampInventory where
  one_cent := 1
  two_cent := 2
  four_cent := 4
  ten_cent := 5

/-- The final theorem applied to Diane's specific inventory -/
theorem dianes_stamp_arrangements : stamp_arrangements_count dianeInventory = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_arrangements_count_dianes_stamp_arrangements_l1315_131564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sequence_properties_l1315_131526

def x : ℕ → ℚ
  | 0 => 2  -- Add this case to handle Nat.zero
  | 1 => 2
  | n + 1 => (2 + x n) / (1 - 2 * x n)

theorem x_sequence_properties :
  (∀ n : ℕ, x n ≠ 0) ∧
  ¬ (∃ T : ℕ, ∀ n : ℕ, x (n + T) = x n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_sequence_properties_l1315_131526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_of_fractional_parts_l1315_131585

noncomputable def floor (x : ℝ) := ⌊x⌋
noncomputable def frac (x : ℝ) := x - floor x

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem arithmetic_sequence_of_fractional_parts (x : ℝ) :
  x > 0 ∧ frac x + floor x = x →
  (is_arithmetic_sequence (frac x) (floor x) x ∨
   is_arithmetic_sequence (frac x) x (floor x) ∨
   is_arithmetic_sequence (floor x) (frac x) x) →
  x = 3/2 := by
  sorry

#check arithmetic_sequence_of_fractional_parts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_of_fractional_parts_l1315_131585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marc_speed_is_twelve_thirteenths_l1315_131529

/-- Represents the hiking scenario with Chantal and Marc -/
structure HikingScenario where
  d : ℝ  -- Half of the total distance from trailhead to peak
  chantal_speed_first_half : ℝ  -- Chantal's speed for the first half
  chantal_speed_second_half : ℝ  -- Chantal's speed for the second half (steep part)
  chantal_speed_descent : ℝ  -- Chantal's descending speed on the steep part

/-- Calculates Marc's average speed given the hiking scenario -/
noncomputable def marc_average_speed (scenario : HikingScenario) : ℝ :=
  (2/3 * scenario.d) / ((scenario.d / scenario.chantal_speed_first_half) +
                        (scenario.d / scenario.chantal_speed_second_half) +
                        ((1/3 * scenario.d) / scenario.chantal_speed_descent))

/-- Theorem stating that Marc's average speed is 12/13 miles per hour -/
theorem marc_speed_is_twelve_thirteenths (scenario : HikingScenario)
  (h1 : scenario.chantal_speed_first_half = 3)
  (h2 : scenario.chantal_speed_second_half = 1.5)
  (h3 : scenario.chantal_speed_descent = 2) :
  marc_average_speed scenario = 12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marc_speed_is_twelve_thirteenths_l1315_131529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_two_equals_two_l1315_131507

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else Real.log x / Real.log a

-- State the theorem
theorem f_f_two_equals_two (a : ℝ) (h : a > 2) : f a (f a 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_two_equals_two_l1315_131507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1315_131575

theorem problem_solution : 
  ((3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) = 14/3) ∧
  ((Real.sqrt 6 + Real.sqrt 3 - Real.sqrt 2) * (Real.sqrt 6 - Real.sqrt 3 + Real.sqrt 2) = 1 + 2 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1315_131575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_not_occupied_by_flowerbeds_l1315_131516

-- Define the garden dimensions
def garden_length : ℝ := 30
def garden_width : ℝ := 24

-- Define the number of quadrants
def num_quadrants : ℕ := 4

-- Theorem statement
theorem garden_area_not_occupied_by_flowerbeds :
  let total_area := garden_length * garden_width
  let circle_radius := garden_width / 4
  let circle_area := π * circle_radius^2
  let total_circle_area := (num_quadrants : ℝ) * circle_area
  total_area - total_circle_area = 720 - 144 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_not_occupied_by_flowerbeds_l1315_131516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_short_film_votes_l1315_131549

theorem short_film_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 120 ∧ 
  like_percentage = 3/4 →
  (score / (like_percentage - (1 - like_percentage))).floor = 240 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_short_film_votes_l1315_131549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_rewrite_same_terminal_side_l1315_131501

noncomputable def α : ℝ := 1200 * (Real.pi / 180)

-- Theorem 1
theorem alpha_rewrite :
  ∃ (k : ℤ), α = 2*Real.pi/3 + 2*Real.pi*k ∧ Real.pi/2 < 2*Real.pi/3 ∧ 2*Real.pi/3 < Real.pi :=
by sorry

-- Theorem 2
theorem same_terminal_side :
  (∃ (k : ℤ), α = 2*Real.pi/3 + 2*Real.pi*k) →
  -10*Real.pi/3 ∈ Set.Icc (-4*Real.pi) Real.pi ∧
  -4*Real.pi/3 ∈ Set.Icc (-4*Real.pi) Real.pi ∧
  ∃ (k₁ k₂ : ℤ), α = -10*Real.pi/3 + 2*Real.pi*k₁ ∧ α = -4*Real.pi/3 + 2*Real.pi*k₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_rewrite_same_terminal_side_l1315_131501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_score_students_value_l1315_131535

def zero_score_students (total_students : ℕ) (perfect_score_students : ℕ) 
  (perfect_score : ℕ) (rest_average : ℕ) (class_average : ℕ) : ℕ :=
  let zero_score_students := total_students - perfect_score_students - 
    ((total_students * class_average - perfect_score_students * perfect_score) / rest_average)
  zero_score_students

#check zero_score_students

theorem zero_score_students_value : 
  zero_score_students 20 2 100 40 40 = 3 := by
  sorry

#check zero_score_students_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_score_students_value_l1315_131535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_PQ_l1315_131555

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define point M
def point_M (a : ℝ) : ℝ × ℝ := (a, 2)

-- Define that M is on the parabola
def M_on_parabola (a : ℝ) : Prop :=
  parabola (point_M a).1 (point_M a).2

-- Define points P and Q
noncomputable def point_P : ℝ × ℝ := sorry
noncomputable def point_Q : ℝ × ℝ := sorry

-- Define that P and Q are on the parabola
def P_on_parabola : Prop := parabola point_P.1 point_P.2
def Q_on_parabola : Prop := parabola point_Q.1 point_Q.2

-- Define the slopes of MP and MQ
noncomputable def slope_MP : ℝ := sorry
noncomputable def slope_MQ : ℝ := sorry

-- Define that the sum of slopes of MP and MQ is π
def sum_of_slopes : Prop := slope_MP + slope_MQ = Real.pi

-- Define the slope of PQ
noncomputable def slope_PQ : ℝ := 
  (point_Q.2 - point_P.2) / (point_Q.1 - point_P.1)

-- The theorem to be proved
theorem slope_of_PQ 
  (a : ℝ) 
  (h1 : M_on_parabola a) 
  (h2 : P_on_parabola) 
  (h3 : Q_on_parabola) 
  (h4 : sum_of_slopes) : 
  slope_PQ = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_PQ_l1315_131555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_total_distance_l1315_131580

/-- A hiker's walking journey over three days -/
structure HikerJourney where
  day1_distance : ℕ
  day1_speed : ℕ
  day2_speed_increase : ℕ
  day2_time_decrease : ℕ
  day3_speed : ℕ
  day3_time : ℕ

/-- Calculate the total distance walked by the hiker -/
def total_distance (j : HikerJourney) : ℕ :=
  let day1 := j.day1_distance
  let day2 := (j.day1_speed + j.day2_speed_increase) * (j.day1_distance / j.day1_speed - j.day2_time_decrease)
  let day3 := j.day3_speed * j.day3_time
  day1 + day2 + day3

/-- The theorem stating the total distance walked by the hiker -/
theorem hiker_total_distance :
  ∃ (j : HikerJourney),
    j.day1_distance = 18 ∧
    j.day1_speed = 3 ∧
    j.day2_speed_increase = 1 ∧
    j.day2_time_decrease = 1 ∧
    j.day3_speed = 5 ∧
    j.day3_time = 3 ∧
    total_distance j = 53 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiker_total_distance_l1315_131580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l1315_131581

/-- An arithmetic sequence is a sequence where the difference between 
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence. -/
noncomputable def arithmetic_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_10 (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 2 + a 4 = 4 →
  a 3 + a 5 = 10 →
  arithmetic_sum a 10 = 95 :=
by
  sorry

#check arithmetic_sequence_sum_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l1315_131581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1315_131571

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi)
  (h_cosine_law : a^2 = b^2 + c^2 - 2*b*c*Real.cos A ∧
                  b^2 = c^2 + a^2 - 2*c*a*Real.cos B ∧
                  c^2 = a^2 + b^2 - 2*a*b*Real.cos C) : 
  (3 : ℝ)/2 ≤ a^2/(b^2+c^2) + b^2/(c^2+a^2) + c^2/(a^2+b^2) ∧
  a^2/(b^2+c^2) + b^2/(c^2+a^2) + c^2/(a^2+b^2) ≤ 2*(Real.cos A^2 + Real.cos B^2 + Real.cos C^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1315_131571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strong_integer_characterization_l1315_131546

theorem strong_integer_characterization (k : ℕ) :
  (∃ (r s : ℕ), r > 0 ∧ s > 0 ∧ (k^2 - 6*k + 11)^(r - 1) = (2*k - 7)^s) ↔ k ∈ ({2, 3, 4, 8} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strong_integer_characterization_l1315_131546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_of_tangents_l1315_131562

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point P
noncomputable def P : ℝ × ℝ := (1, Real.sqrt 3)

-- Define the tangent property
def is_tangent (p a : ℝ × ℝ) : Prop :=
  is_on_circle a.1 a.2 ∧ ((p.1 - a.1) * a.1 + (p.2 - a.2) * a.2 = 0)

-- Theorem statement
theorem chord_length_of_tangents :
  ∃ (A B : ℝ × ℝ),
    is_tangent P A ∧
    is_tangent P B ∧
    A ≠ B ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_of_tangents_l1315_131562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_roots_l1315_131530

theorem max_distance_between_roots (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 0) (h4 : a ≠ 0) :
  ∃ d : ℝ, d = Real.sqrt 2 * (3/2) ∧ 
    ∀ x₁ x₂ : ℝ, a * x₁^2 + b * x₁ + c = 0 → a * x₂^2 + b * x₂ + c = 0 → x₁ ≠ x₂ →
      Real.sqrt ((x₁ - x₂)^2 + (x₂ - x₁)^2) ≤ d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_roots_l1315_131530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1315_131521

/-- The area of a circular sector with given central angle and radius -/
noncomputable def sectorArea (θ : ℝ) (r : ℝ) : ℝ := (1/2) * θ * r^2

/-- Theorem: The area of a circular sector with central angle 3/2 radians and radius 12 cm is 108 cm² -/
theorem sector_area_example : sectorArea (3/2) 12 = 108 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_example_l1315_131521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessary_nor_sufficient_condition_l1315_131539

theorem not_necessary_nor_sufficient_condition (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) : 
  ¬(∀ a b : ℝ, a > 0 → b > 0 → a > b → Real.log b / Real.log a < 1) ∧ 
  ¬(∀ a b : ℝ, a > 0 → b > 0 → Real.log b / Real.log a < 1 → a > b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessary_nor_sufficient_condition_l1315_131539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_six_equals_63_16_l1315_131545

/-- A decreasing geometric sequence with specific properties -/
structure DecreasingGeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_decreasing : 0 < q ∧ q < 1
  h_geometric : ∀ n, a (n + 1) = a n * q
  h_positive : 0 < a 1
  h_a1a3 : a 1 * a 3 = 1
  h_a2a4 : a 2 + a 4 = 5 / 4

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (s : DecreasingGeometricSequence) (n : ℕ) : ℝ :=
  (s.a 1) * (1 - s.q ^ n) / (1 - s.q)

/-- The main theorem: S_6 = 63/16 for the given sequence -/
theorem sum_six_equals_63_16 (s : DecreasingGeometricSequence) :
    geometricSum s 6 = 63 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_six_equals_63_16_l1315_131545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_differences_sum_l1315_131548

theorem product_of_differences_sum (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  (6 - a) * (6 - b) * (6 - c) * (6 - d) = 9 →
  a + b + c + d = 24 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_differences_sum_l1315_131548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_greater_than_two_l1315_131538

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2 - a * x

-- Define is_extreme_point
def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (∀ y, y ≠ x → f y ≤ f x) ∨ (∀ y, y ≠ x → f y ≥ f x)

theorem extreme_points_sum_greater_than_two (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a > 1)
  (h₂ : x₁ ≠ x₂)
  (h₃ : is_extreme_point (f a) x₁)
  (h₄ : is_extreme_point (f a) x₂) :
  f a x₁ + f a x₂ > 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_greater_than_two_l1315_131538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_types_possible_l1315_131578

-- Define a triangle ABC with angles A, B, C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the condition from the problem
def satisfies_condition (t : Triangle) : Prop :=
  1007 * t.A^2 + 1009 * t.B^2 = 2016 * t.C^2

-- Define triangle types
def is_acute (t : Triangle) : Prop := t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2
def is_right (t : Triangle) : Prop := t.A = Real.pi/2 ∨ t.B = Real.pi/2 ∨ t.C = Real.pi/2
def is_obtuse (t : Triangle) : Prop := t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

-- Theorem statement
theorem triangle_types_possible :
  ∃ t₁ t₂ t₃ : Triangle,
    satisfies_condition t₁ ∧ is_acute t₁ ∧
    satisfies_condition t₂ ∧ is_right t₂ ∧
    satisfies_condition t₃ ∧ is_obtuse t₃ :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_types_possible_l1315_131578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_trapezoid_area_l1315_131568

/-- Represents a trapezoid in a 2D coordinate system -/
structure Trapezoid where
  lower_left : ℝ × ℝ
  lower_right : ℝ × ℝ
  upper_left : ℝ × ℝ
  upper_right : ℝ × ℝ

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  let a := (t.lower_right.1 - t.lower_left.1)
  let b := (t.upper_right.1 - t.upper_left.1)
  let h := (t.upper_left.2 - t.lower_left.2)
  (a + b) * h / 2

/-- The specific trapezoid described in the problem -/
def problemTrapezoid : Trapezoid :=
  { lower_left := (0, 3)
    lower_right := (3, 3)
    upper_left := (0, 8)
    upper_right := (8, 8) }

/-- Theorem stating that the area of the problem trapezoid is 27.5 -/
theorem problem_trapezoid_area :
  trapezoidArea problemTrapezoid = 27.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval trapezoidArea problemTrapezoid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_trapezoid_area_l1315_131568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rider_problem_l1315_131599

noncomputable def distance (p1 p2 : ℝ) : ℝ := abs (p2 - p1)

noncomputable def speed (d t : ℝ) : ℝ := d / t

theorem rider_problem (A B C : ℝ) :
  (distance A B) + 20 = distance C B →
  (∃ (t : ℝ), t > 0 ∧ distance A B / t = 5) →
  (∃ (s : ℝ), s > 0 ∧ speed (distance C B) s = speed (distance A B) 5 + 1/75) →
  distance C B = 80 := by
  sorry

#check rider_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rider_problem_l1315_131599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_a_range_l1315_131577

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (x - 2) * Real.exp x + a * x^2 + b * x

-- Part 1
theorem monotonicity_intervals (x : ℝ) :
  let a : ℝ := 1
  let b : ℝ := -2
  (∀ y z, y < 1 ∧ z < 1 ∧ y < z → f a b y < f a b z) ∧
  (∀ y z, y > 1 ∧ z > 1 ∧ y < z → f a b y < f a b z) :=
by sorry

-- Part 2
theorem a_range (a : ℝ) :
  (∀ x : ℝ, f a (-2*a) x ≥ f a (-2*a) 1) ↔ a > -(Real.exp 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_a_range_l1315_131577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1315_131523

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x - Real.pi/6) + Real.cos (2*x) + 1/4

theorem f_properties :
  ∃ (k : ℤ), 
    (∀ x ∈ Set.Icc (k * Real.pi - 5*Real.pi/3) (k * Real.pi + Real.pi/12), 
      MonotoneOn f (Set.Icc (k * Real.pi - 5*Real.pi/3) (k * Real.pi + Real.pi/12))) ∧
    (∀ x ∈ Set.Icc (-Real.pi/12) (5*Real.pi/12), f x ≤ Real.sqrt 3/2) ∧
    (∀ x ∈ Set.Icc (-Real.pi/12) (5*Real.pi/12), f x ≥ -Real.sqrt 3/4) ∧
    (∃ x₁ ∈ Set.Icc (-Real.pi/12) (5*Real.pi/12), f x₁ = Real.sqrt 3/2) ∧
    (∃ x₂ ∈ Set.Icc (-Real.pi/12) (5*Real.pi/12), f x₂ = -Real.sqrt 3/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1315_131523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_l1315_131517

theorem sin_cos_fourth_power (θ : ℝ) (h : Real.cos (2 * θ) = 1/3) : 
  Real.sin θ^4 + Real.cos θ^4 = 5/9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_l1315_131517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1315_131565

-- Define the original radius
variable (r : ℝ)

-- Assume r is positive (implicit in the original problem)
variable (hr : r > 0)

-- Define the ratio of new area to original area
noncomputable def area_ratio (r : ℝ) : ℝ := (Real.pi * (r + 2)^2) / (Real.pi * r^2)

-- Theorem statement
theorem circle_area_ratio : 
  area_ratio r = 1 + 4 * (r + 1) / r^2 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1315_131565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercept_l1315_131537

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  distance p e.focus1 + distance p e.focus2 =
  distance (Point.mk 0 0) e.focus1 + distance (Point.mk 0 0) e.focus2

/-- The main theorem to prove -/
theorem ellipse_x_intercept (e : Ellipse) :
  e.focus1 = Point.mk (-2) 3 →
  e.focus2 = Point.mk 4 1 →
  isOnEllipse e (Point.mk 0 0) →
  isOnEllipse e (Point.mk 6 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_x_intercept_l1315_131537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_arccot_three_fifths_l1315_131502

theorem tan_arccot_three_fifths (a b : ℝ) (h1 : a = 3) (h2 : b = 5) :
  Real.tan (Real.arctan⁻¹ (a / b)) = b / a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_arccot_three_fifths_l1315_131502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_relation_l1315_131570

-- Define the variables and constants
variable (α β : ℝ)
variable (m : ℝ)
variable (x y : ℝ)

-- Define the conditions
theorem tangent_relation
  (h1 : 0 < α ∧ α < Real.pi/2)
  (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : m > 0)
  (h4 : α + β ≠ Real.pi/2)
  (h5 : Real.sin β = m * Real.cos (α + β) * Real.sin α)
  (h6 : x = Real.tan α)
  (h7 : y = Real.tan β)
  (h8 : Real.pi/4 ≤ α ∧ α < Real.pi/2) :
  -- Statement 1: Relation between y and x
  y = (m * x) / (1 + (m + 1) * x^2) ∧
  -- Statement 2: Maximum value of y
  y ≤ m / (m + 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_relation_l1315_131570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_is_zero_l1315_131513

/-- Given real numbers a, b, c, and d, prove that the product of the matrices
    [0, 2c, -2b; -2c, 0, 2a; 2b, -2a, 0] and [a^2+d, ab, ac; ab, b^2+d, bc; ac, bc, c^2+d]
    is equal to the 3x3 zero matrix. -/
theorem matrix_product_is_zero (a b c d : ℝ) : 
  (![![0, 2*c, -2*b], ![-2*c, 0, 2*a], ![2*b, -2*a, 0]] : Matrix (Fin 3) (Fin 3) ℝ) *
  (![![a^2 + d, a*b, a*c], ![a*b, b^2 + d, b*c], ![a*c, b*c, c^2 + d]] : Matrix (Fin 3) (Fin 3) ℝ)
  = (0 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_is_zero_l1315_131513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_theta_is_plane_l1315_131573

/-- In spherical coordinates (ρ, θ, φ), the equation θ = c (where c is a constant) describes a plane. -/
theorem constant_theta_is_plane (c : ℝ) :
  let spherical_surface := {p : ℝ × ℝ × ℝ | p.2.1 = c}
  ∃ (normal : ℝ × ℝ × ℝ) (d : ℝ),
    ∀ (p : ℝ × ℝ × ℝ), p ∈ spherical_surface ↔ normal.1 * p.1 + normal.2.1 * p.2.1 + normal.2.2 * p.2.2 = d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_theta_is_plane_l1315_131573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_semi_major_axis_l1315_131522

/-- Given a line and an ellipse intersecting at two points A and B, 
    where OA is perpendicular to OB (O being the origin), 
    and the eccentricity of the ellipse is within a certain range, 
    prove that the maximum value of the semi-major axis 'a' is √(5/2). -/
theorem max_semi_major_axis 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_line : ∀ x y : ℝ, y = -x + 1 → x^2 / a^2 + y^2 / b^2 = 1 → 
    ∃ A B : ℝ × ℝ, A.fst * B.fst + A.snd * B.snd = 0) 
  (h_ecc : ∃ e : ℝ, e^2 = (a^2 - b^2) / a^2 ∧ 1/2 ≤ e ∧ e ≤ Real.sqrt 3/2) : 
  a ≤ Real.sqrt (5/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_semi_major_axis_l1315_131522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_approx_ten_l1315_131557

/-- The distance between two points in a 2D plane. -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem: The sum of distances AD and BD is approximately 10. -/
theorem sum_distances_approx_ten :
  let A : ℝ × ℝ := (8, 0)
  let B : ℝ × ℝ := (0, 5)
  let D : ℝ × ℝ := (1, 3)
  let AD := distance A.1 A.2 D.1 D.2
  let BD := distance B.1 B.2 D.1 D.2
  abs ((AD + BD) - 10) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_approx_ten_l1315_131557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_volume_l1315_131509

/-- The volume of a square-based pyramid frustum -/
noncomputable def frustum_volume (a b h : ℝ) : ℝ := (h / 3) * (a^2 + a*b + b^2)

/-- Theorem: The volume of a specific square-based pyramid frustum -/
theorem specific_frustum_volume :
  let a : ℝ := 6 * Real.sqrt 2
  let b : ℝ := 12 * Real.sqrt 2
  let h : ℝ := 12
  frustum_volume a b h = 2016 := by
  -- Unfold the definition of frustum_volume
  unfold frustum_volume
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry

#check specific_frustum_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_frustum_volume_l1315_131509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_speed_l1315_131576

/-- Calculate the combined average speed of three objects in miles per hour -/
theorem combined_average_speed (distance_A distance_B distance_C : ℝ)
                                (time_A time_B time_C : ℝ)
                                (feet_per_mile : ℝ) :
  distance_A = 300 →
  distance_B = 400 →
  distance_C = 500 →
  time_A = 6 →
  time_B = 8 →
  time_C = 10 →
  feet_per_mile = 5280 →
  abs ((distance_A + distance_B + distance_C) / feet_per_mile /
       ((time_A + time_B + time_C) / 3600) - 34.09) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_average_speed_l1315_131576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l1315_131511

theorem existence_of_special_set (n : ℕ+) :
  ∃ (S : Finset ℕ+), 
    (Finset.card S = n) ∧ 
    (∀ (a b : ℕ+), a ∈ S → b ∈ S → a ≠ b → 
      (∃ k : ℤ, (a : ℤ) - (b : ℤ) = k * (a : ℤ)) ∧ 
      (∃ m : ℤ, (a : ℤ) - (b : ℤ) = m * (b : ℤ)) ∧ 
      (∀ c ∈ S, c ≠ a ∧ c ≠ b → ¬∃ l : ℤ, (a : ℤ) - (b : ℤ) = l * (c : ℤ))) := by
  sorry

#check existence_of_special_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l1315_131511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_in_interval_l1315_131527

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

-- State the theorem
theorem f_has_max_in_interval :
  ∃ (c : ℝ), c ∈ Set.Ioo (-Real.pi/6) (Real.pi/3) ∧
  ∀ (x : ℝ), x ∈ Set.Ioo (-Real.pi/6) (Real.pi/3) → f x ≤ f c :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_in_interval_l1315_131527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_divisibility_property_l1315_131556

theorem function_divisibility_property (k l : ℕ) :
  (∃ f : ℕ → ℕ, ∀ m n : ℕ, (f m + f n) ∣ (m + n + l)^k) ↔ 
  (∃ c : ℕ, l = 2 * c) ∧ 
  (∃ f : ℕ → ℕ, ∀ n : ℕ, f n = n + l / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_divisibility_property_l1315_131556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_y_slower_than_x_l1315_131508

/-- Represents a route with distance and speed sections -/
structure Route where
  sections : List (ℝ × ℝ)  -- List of (distance, speed) pairs

/-- Calculate the time taken for a route in hours -/
noncomputable def routeTime (r : Route) : ℝ :=
  r.sections.foldr (fun (d, s) acc => acc + d / s) 0

/-- Convert hours to minutes -/
noncomputable def hoursToMinutes (h : ℝ) : ℝ := h * 60

theorem route_y_slower_than_x : 
  let routeX : Route := ⟨[(12, 45)]⟩
  let routeY : Route := ⟨[(9, 50), (1, 10)]⟩
  hoursToMinutes (routeTime routeY - routeTime routeX) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_y_slower_than_x_l1315_131508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1315_131559

theorem expression_value : 
  (1 / (1 - Real.rpow 3 (1/4))) + (1 / (1 + Real.rpow 3 (1/4))) + (2 / (1 + Real.sqrt 3)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1315_131559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finland_forest_calculation_l1315_131515

/-- The total forested area of the world in hectares -/
noncomputable def world_forest_area : ℝ := 8076000000

/-- The percentage of world's forested area represented by Finland -/
noncomputable def finland_forest_percentage : ℝ := 0.66

/-- The forested area of Finland in hectares -/
noncomputable def finland_forest_area : ℝ := world_forest_area * (finland_forest_percentage / 100)

/-- Theorem stating that Finland's forested area is approximately 53,301,600 hectares -/
theorem finland_forest_calculation : 
  ∃ ε > 0, |finland_forest_area - 53301600| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finland_forest_calculation_l1315_131515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_profit_percentage_l1315_131594

theorem original_profit_percentage
  (cost : ℝ)
  (reduced_selling_price : ℝ)
  (original_profit_percentage : ℝ)
  (h1 : cost = 30)
  (h2 : reduced_selling_price = (1 + 0.3) * (0.8 * cost))
  (h3 : reduced_selling_price = (cost * (1 + original_profit_percentage / 100)) - 6.3)
  : original_profit_percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_profit_percentage_l1315_131594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_corners_theorem_l1315_131592

-- Define the structure for a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the structure for a house
structure House where
  corner1 : Point
  corner2 : Point
  corner3 : Point
  corner4 : Point

-- Define the function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Define the theorem
theorem house_corners_theorem (h : House) :
  h.corner1 = ⟨1, 14⟩ →
  h.corner2 = ⟨17, 2⟩ →
  distance h.corner1 h.corner3 = (distance h.corner1 h.corner2) / 2 →
  distance h.corner2 h.corner4 = (distance h.corner1 h.corner2) / 2 →
  (h.corner3 = ⟨-5, 6⟩ ∧ h.corner4 = ⟨11, -6⟩) ∨
  (h.corner3 = ⟨11, -6⟩ ∧ h.corner4 = ⟨-5, 6⟩) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_corners_theorem_l1315_131592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_problem_l1315_131550

def is_divisible (f g : ℝ → ℝ) : Prop :=
  ∃ h : ℝ → ℝ, ∀ x, f x = g x * h x

theorem quadratic_polynomial_problem (q : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c)
  (h2 : is_divisible (λ x ↦ (q x)^2 - x^2) (λ x ↦ (x - 2) * (x + 2) * (x - 5))) :
  q 10 = 250 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_problem_l1315_131550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l1315_131582

noncomputable def cycle_a_cost : ℝ := 4000
noncomputable def cycle_b_cost : ℝ := 6000
noncomputable def cycle_c_cost : ℝ := 8000
noncomputable def cycle_a_sale : ℝ := 4500
noncomputable def cycle_b_sale : ℝ := 6500
noncomputable def cycle_c_sale : ℝ := 8500

noncomputable def total_cost : ℝ := cycle_a_cost + cycle_b_cost + cycle_c_cost
noncomputable def total_sale : ℝ := cycle_a_sale + cycle_b_sale + cycle_c_sale
noncomputable def total_gain : ℝ := total_sale - total_cost

noncomputable def gain_percentage : ℝ := (total_gain / total_cost) * 100

theorem overall_gain_percentage :
  abs (gain_percentage - 8.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_l1315_131582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1315_131542

/-- The distance between two parallel lines ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₁ - c₂) / Real.sqrt (a^2 + b^2)

/-- The first line equation: 3x + 4y - 3 = 0 -/
def line1 : ℝ → ℝ → Prop :=
  fun x y => 3 * x + 4 * y - 3 = 0

/-- The second line equation: 3x + 4y + 7 = 0 -/
def line2 : ℝ → ℝ → Prop :=
  fun x y => 3 * x + 4 * y + 7 = 0

theorem distance_between_given_lines :
  distance_between_parallel_lines 3 4 (-3) 7 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1315_131542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1315_131554

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.log x + Real.sqrt (2 - x)

-- Define the domain of f
def domain (x : ℝ) : Prop := (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2)

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, (∃ y, f x = y) ↔ domain x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1315_131554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hd_ha_ratio_is_zero_l1315_131583

/-- A triangle with sides 8, 15, and 17 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_eq : a = 8
  b_eq : b = 15
  c_eq : c = 17
  pythagorean : a^2 + b^2 = c^2

/-- The altitude from vertex A to the side of length 15 -/
noncomputable def altitude (t : RightTriangle) : ℝ := 2 * (t.a * t.b) / (2 * t.b)

/-- The point where the altitudes meet -/
def orthocenter (_ : RightTriangle) : ℝ × ℝ := (0, 0)

/-- The foot of the altitude from A to the side of length 15 -/
def altitude_foot (_ : RightTriangle) : ℝ × ℝ := (0, 0)

/-- The ratio of HD to HA -/
noncomputable def hd_ha_ratio (t : RightTriangle) : ℝ :=
  let h := orthocenter t
  let d := altitude_foot t
  let a := (t.c, 0)
  Real.sqrt ((h.1 - d.1)^2 + (h.2 - d.2)^2) / Real.sqrt ((h.1 - a.1)^2 + (h.2 - a.2)^2)

theorem hd_ha_ratio_is_zero (t : RightTriangle) : hd_ha_ratio t = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hd_ha_ratio_is_zero_l1315_131583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1315_131584

-- Define a, b, and c
def a : ℚ := -4
def b : ℚ := (1 : ℚ) / 4
def c : ℚ := -(1 : ℚ) / 4

-- State the theorem
theorem order_of_abc : a < c ∧ c < b := by
  -- Split the conjunction into two parts
  constructor
  -- Prove a < c
  · simp [a, c]
    norm_num
  -- Prove c < b
  · simp [b, c]
    norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1315_131584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1315_131505

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x + 1)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^(x - 2023) + 1

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 2^(-x^2 + a*x)

theorem problem_solution :
  (¬ (∀ x y : ℝ, x < y → (x < -1 ∨ x > -1) → (y < -1 ∨ y > -1) → f x ≥ f y)) ∧
  (∀ a : ℝ, a > 0 → a ≠ 1 → g a 2023 = 2) ∧
  ((¬ ∃ x : ℝ, x^2 + a*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 > 0)) ∧
  (∀ a : ℝ, (∀ x y : ℝ, x < y → x < 1 → y < 1 → h a x < h a y) → ¬ (a > 2 ∧ ∀ b : ℝ, b > a → b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1315_131505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_result_domain_of_f_composition_result_l1315_131566

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)
def g (x : ℝ) : ℝ := x^2 + 2

-- Theorem for the calculation
theorem calculation_result : 
  ((-4 : ℝ)^3)^(1/3) - (1/2)^0 + 25^(1/2) = 1 := by sorry

-- Theorem for the domain of f
theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 ↔ x ≠ -1 := by sorry

-- Theorem for the composition of f and g
theorem composition_result :
  f (g 2) = 1/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_result_domain_of_f_composition_result_l1315_131566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_calculation_l1315_131588

/-- Represents the loan details and calculation --/
structure Loan where
  principal : ℚ
  rate : ℚ
  time : ℚ
  total_amount : ℚ

/-- Calculates the total amount after simple interest --/
def calculate_total_amount (loan : Loan) : ℚ :=
  loan.principal * (1 + loan.rate * loan.time)

/-- Theorem stating the relationship between the total amount and the principal --/
theorem borrowed_amount_calculation (loan : Loan) 
  (h1 : loan.rate = 6 / 100)
  (h2 : loan.time = 9)
  (h3 : loan.total_amount = 8410)
  : ∃ (ε : ℚ), abs (loan.principal - 5461) < ε ∧ ε < 1 := by
  sorry

#check borrowed_amount_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowed_amount_calculation_l1315_131588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_calculation_l1315_131598

/-- Represents a vessel with a capacity and alcohol percentage -/
structure Vessel where
  capacity : ℚ
  alcoholPercentage : ℚ

/-- Calculates the new concentration of alcohol in a mixture -/
noncomputable def newConcentration (vessels : List Vessel) (finalCapacity : ℚ) : ℚ :=
  let totalAlcohol := vessels.foldl (fun acc v => acc + v.capacity * v.alcoholPercentage) 0
  let totalVolume := finalCapacity
  totalAlcohol / totalVolume

theorem concentration_calculation (vessels : List Vessel) (finalCapacity : ℚ) :
  vessels = [
    { capacity := 3, alcoholPercentage := 1/4 },
    { capacity := 5, alcoholPercentage := 2/5 },
    { capacity := 7, alcoholPercentage := 3/5 },
    { capacity := 4, alcoholPercentage := 3/20 }
  ] ∧ finalCapacity = 25 →
  newConcentration vessels finalCapacity = 151/500 := by
  sorry

#eval (151 : ℚ) / 500

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentration_calculation_l1315_131598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_for_juice_calculation_l1315_131597

/-- The amount of oranges used for juice given total production and export percentage -/
def oranges_for_juice (total_production : ℝ) (export_percentage : ℝ) : ℝ :=
  total_production * (1 - export_percentage) * 0.6

theorem oranges_for_juice_calculation :
  let total_production : ℝ := 7
  let export_percentage : ℝ := 0.3
  ∃ (x : ℝ), x = oranges_for_juice total_production export_percentage ∧ 
             2.85 ≤ x ∧ x < 2.95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_for_juice_calculation_l1315_131597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_sin_double_plus_pi_third_l1315_131572

/-- Given an angle θ with vertex at the origin, initial side along the positive x-axis,
    and terminal side on the line y = 3x, prove that sin(2θ + π/3) = (3 - 4√3) / 10 -/
theorem angle_on_line_sin_double_plus_pi_third (θ : ℝ) :
  (∃ (x y : ℝ), y = 3 * x ∧ x * Real.cos θ = x ∧ y * Real.sin θ = y) →
  Real.sin (2 * θ + π / 3) = (3 - 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_sin_double_plus_pi_third_l1315_131572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l1315_131504

theorem arithmetic_geometric_sequence_ratio 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (harith : 2 * a = b + c)
  (hgeom : a * a = b * c) : 
  ∃ (k : ℝ), k > 0 ∧ a = 2 * k ∧ b = 4 * k ∧ c = k := by
  sorry

#check arithmetic_geometric_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l1315_131504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1315_131519

theorem propositions_truth : 
  (¬ ∃ x₀ : ℝ, Real.sin x₀ = Real.sqrt 5 / 2) ∧ 
  (∀ x ∈ Set.Ioo 0 (Real.pi / 2), x > Real.sin x) ∧
  (¬¬ ∀ x ∈ Set.Ioo 0 (Real.pi / 2), x > Real.sin x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1315_131519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1315_131544

theorem simplify_expression (y : ℝ) (h : y ≠ 0) :
  y^(-2 : ℤ) - 2 = (1 - 2*y^2) / y^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1315_131544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_even_number_in_sequence_l1315_131531

theorem third_even_number_in_sequence (seq : List Nat) : 
  seq.length = 6 ∧ 
  (∀ i, i < 5 → seq.get! (i + 1) = seq.get! i + 2) ∧
  (∀ n ∈ seq, Even n) ∧
  seq.sum = 180 →
  seq.get! 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_even_number_in_sequence_l1315_131531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_in_second_quadrant_l1315_131579

-- Define the function as noncomputable
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k / x

-- State the theorem
theorem function_in_second_quadrant (k : ℝ) :
  (∀ x₁ x₂, x₁ < 0 ∧ x₂ < 0 ∧ x₁ < x₂ → f k x₁ < f k x₂) →
  (∀ x, x < 0 → x < 0 ∧ f k x > 0) :=
by
  -- Introduce the hypothesis
  intro h
  -- Introduce the variable x and the assumption x < 0
  intro x hx
  -- Split the goal into two parts
  constructor
  -- Prove the first part: x < 0
  exact hx
  -- For the second part: f k x > 0, we use sorry to skip the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_in_second_quadrant_l1315_131579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_and_g_inequality_l1315_131518

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.log x - 2 * a * x

def g (x : ℝ) : ℝ := f a x + (1/2) * x^2

theorem f_max_value_and_g_inequality (x₀ : ℝ) :
  (∀ x > 0, HasDerivAt (f a) (-1) 1) →
  (∀ x > 0, HasDerivAt (g a) 0 x₀) →
  (∀ x > 0, g a x ≤ g a x₀) →
  (∃ x_max > 0, ∀ x > 0, f a x ≤ f a x_max ∧ f a x_max = -1 - Real.log 2) ∧
  (x₀ * f a x₀ + 1 + a * x₀^2 > 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_and_g_inequality_l1315_131518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_rectangles_and_triangles_axisymmetric_l1315_131595

-- Define what it means for a figure to be axisymmetric
def is_axisymmetric (Figure : Type) : Prop :=
  ∃ (line : Set (ℝ × ℝ)), ∀ (f : Figure), 
    ∃ (folded_f : Figure), (∃ (fold : Figure → Set (ℝ × ℝ) → Figure), folded_f = fold f line) ∧ folded_f = f

-- Define rectangles and triangles as types
variable (Rectangle Triangle : Type)

-- State the theorem
theorem not_all_rectangles_and_triangles_axisymmetric :
  ¬(∀ (r : Rectangle), is_axisymmetric Rectangle ∧ 
     ∀ (t : Triangle), is_axisymmetric Triangle) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_rectangles_and_triangles_axisymmetric_l1315_131595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression1_evaluate_expression2_l1315_131532

-- Define the first expression
noncomputable def expression1 (α : Real) : Real :=
  (Real.tan (3 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-α - Real.pi) * Real.sin (-Real.pi + α) * Real.cos (α + 5 * Real.pi / 2))

-- Define the second expression
noncomputable def expression2 (α : Real) : Real :=
  1 / (2 * Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α)

-- Theorem for the first expression
theorem simplify_expression1 (α : Real) :
  expression1 α = -1 / Real.sin α := by sorry

-- Theorem for the second expression
theorem evaluate_expression2 :
  Real.tan α = 1 / 4 → expression2 α = 17 / 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression1_evaluate_expression2_l1315_131532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gizmo_production_l1315_131560

/-- Represents the production rates in the factory -/
structure ProductionRates where
  gadget_rate : ℚ
  gizmo_rate : ℚ

/-- Represents a production scenario -/
structure Scenario where
  workers : ℕ
  hours : ℕ
  gadgets : ℕ
  gizmos : ℕ

/-- Calculates the production rates based on a given scenario -/
noncomputable def calculate_rates (s : Scenario) : ProductionRates :=
  { gadget_rate := (s.gadgets : ℚ) / ((s.workers * s.hours) : ℚ)
  , gizmo_rate := (s.gizmos : ℚ) / ((s.workers * s.hours) : ℚ) }

/-- Verifies if a scenario is consistent with given production rates -/
def is_consistent (s : Scenario) (r : ProductionRates) : Prop :=
  (s.gadgets : ℚ) = (s.workers * s.hours : ℚ) * r.gadget_rate ∧
  (s.gizmos : ℚ) = (s.workers * s.hours : ℚ) * r.gizmo_rate

/-- The main theorem to prove -/
theorem gizmo_production
  (s1 : Scenario)
  (s2 : Scenario)
  (s3 : Scenario)
  (h1 : s1 = { workers := 80, hours := 2, gadgets := 320, gizmos := 480 })
  (h2 : s2 = { workers := 100, hours := 3, gadgets := 600, gizmos := 900 })
  (h3 : s3 = { workers := 40, hours := 4, gadgets := 160, gizmos := s3.gizmos })
  (h_consistent : is_consistent s1 (calculate_rates s1) ∧
                  is_consistent s2 (calculate_rates s1) ∧
                  is_consistent s3 (calculate_rates s1)) :
  s3.gizmos = 480 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gizmo_production_l1315_131560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1315_131587

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the point through which tangents pass
def tangent_point : ℝ × ℝ := (1, -2)

-- Define the line equation
def line_equation (y : ℝ) : Prop := y = -1/2

-- Theorem statement
theorem tangent_line_equation :
  ∃ (A B : ℝ × ℝ),
    (circle_eq A.1 A.2) ∧
    (circle_eq B.1 B.2) ∧
    (∃ (t₁ t₂ : ℝ), 
      (tangent_point.1 - t₁ * (A.1 - 1) = A.1) ∧
      (tangent_point.2 - t₁ * A.2 = A.2) ∧
      (tangent_point.1 - t₂ * (B.1 - 1) = B.1) ∧
      (tangent_point.2 - t₂ * B.2 = B.2)) →
    (∀ (x y : ℝ), (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) → line_equation y) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1315_131587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_buildings_arrangement_exists_five_buildings_arrangement_not_exists_l1315_131520

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing an arrangement of buildings -/
def Arrangement (n : ℕ) := Fin n → Point

/-- 
  A predicate that checks if a sequence of points can be viewed in order 
  by rotating either clockwise or counterclockwise from a given viewpoint
-/
def ViewableInOrder (viewpoint : Point) (clockwise : Bool) (seq : Fin n → Point) : Prop :=
  sorry -- Definition omitted for brevity

/-- 
  A predicate that checks if a given arrangement allows viewing all buildings 
  in any arbitrary order from some point
-/
def AllOrdersViewable (arr : Arrangement n) : Prop :=
  ∀ (perm : Fin n → Fin n), ∃ (viewpoint : Point), 
    ∃ (clockwise : Bool), ViewableInOrder viewpoint clockwise (arr ∘ perm)

theorem four_buildings_arrangement_exists :
  ∃ (arr : Arrangement 4), AllOrdersViewable arr := by
  sorry

theorem five_buildings_arrangement_not_exists :
  ¬ ∃ (arr : Arrangement 5), AllOrdersViewable arr := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_buildings_arrangement_exists_five_buildings_arrangement_not_exists_l1315_131520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_formation_has_12_colorings_l1315_131593

/-- Represents the orientation of a mobot -/
inductive MobotOrientation
  | North
  | East

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents a mobot with its position and orientation -/
structure Mobot where
  pos : Position
  orient : MobotOrientation

/-- The grid size -/
def gridSize : Nat := 4

/-- The number of colors -/
def numColors : Nat := 3

/-- The specific formation of mobots -/
def mobotFormation : List Mobot := [
  { pos := ⟨0, 0⟩, orient := MobotOrientation.North },
  { pos := ⟨1, 0⟩, orient := MobotOrientation.North },
  { pos := ⟨2, 2⟩, orient := MobotOrientation.North },
  { pos := ⟨3, 2⟩, orient := MobotOrientation.North },
  { pos := ⟨2, 0⟩, orient := MobotOrientation.East },
  { pos := ⟨2, 1⟩, orient := MobotOrientation.East }
]

/-- Function to count the number of valid colorings -/
def countValidColorings (formation : List Mobot) : Nat :=
  sorry

/-- Theorem stating that the specific formation has exactly 12 valid colorings -/
theorem specific_formation_has_12_colorings :
  countValidColorings mobotFormation = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_formation_has_12_colorings_l1315_131593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_value_cos_value_l1315_131586

-- Define α as a real number
variable (α : Real)

-- Define the condition that tan α = 2
def tan_alpha : Prop := Real.tan α = 2

-- Define that α is in the third quadrant
def third_quadrant : Prop := Real.pi < α ∧ α < 3*Real.pi/2

-- Theorem 1
theorem fraction_value (h : tan_alpha α) :
  (3 * Real.sin α - 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 := by sorry

-- Theorem 2
theorem cos_value (h1 : tan_alpha α) (h2 : third_quadrant α) :
  Real.cos α = -Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_value_cos_value_l1315_131586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_pi_third_l1315_131506

theorem tan_sum_pi_third (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + π/3) = -(6 * Real.sqrt 3 + 2) / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_pi_third_l1315_131506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_inside_circle_l1315_131510

-- Define the point P
def P : ℝ × ℝ := (3, 2)

-- Define the center of the circle
def center : ℝ × ℝ := (2, 3)

-- Define the radius of the circle
def radius : ℝ := 2

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem: P is inside the circle
theorem P_inside_circle :
  distance P center < radius := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_inside_circle_l1315_131510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_painted_faces_l1315_131536

/-- Represents a cube with painted faces -/
structure PaintedCube where
  total : Nat
  twoPainted : Nat

/-- Calculates the probability of selecting two cubes with two painted faces -/
def probabilityTwoPainted (cube : PaintedCube) : Rat :=
  (Nat.choose cube.twoPainted 2 : Rat) / (Nat.choose cube.total 2 : Rat)

/-- Theorem stating the probability of selecting two cubes with two painted faces -/
theorem probability_two_painted_faces (cube : PaintedCube) 
  (h1 : cube.total = 27) 
  (h2 : cube.twoPainted = 12) : 
  probabilityTwoPainted cube = 22 / 117 := by
  sorry

#eval probabilityTwoPainted { total := 27, twoPainted := 12 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_painted_faces_l1315_131536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l1315_131524

-- Define the logarithm base 2 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 * lg 9

-- State the theorem
theorem f_sum_equals_two : f 2 + f 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_two_l1315_131524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_occupied_chairs_l1315_131534

/-- Represents a row of chairs -/
def ChairRow := List Bool

/-- The number of chairs in the row -/
def numChairs : Nat := 30

/-- Checks if a given chair configuration is valid according to the rules -/
def isValidConfiguration (chairs : ChairRow) : Bool :=
  chairs.length = numChairs &&
  ¬(List.zip chairs (chairs.tail!)).any (fun (a, b) => a && b)

/-- Counts the number of occupied chairs in a configuration -/
def countOccupied (chairs : ChairRow) : Nat :=
  chairs.filter id |>.length

/-- Theorem: The maximum number of occupied chairs is 29 -/
theorem max_occupied_chairs :
  (∃ (chairs : ChairRow), isValidConfiguration chairs ∧ countOccupied chairs = 29) ∧
  (∀ (chairs : ChairRow), isValidConfiguration chairs → countOccupied chairs ≤ 29) := by
  sorry

#eval numChairs  -- This line is added to check if the code compiles and runs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_occupied_chairs_l1315_131534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_exp_range_range_implies_increasing_l1315_131540

-- Define the exponential function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2*a - 1)^x

-- State the theorem
theorem increasing_exp_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → a > 1 :=
by
  -- The proof is omitted using 'sorry'
  sorry

-- State the converse theorem
theorem range_implies_increasing (a : ℝ) :
  a > 1 → (∀ x y : ℝ, x < y → f a x < f a y) :=
by
  -- The proof is omitted using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_exp_range_range_implies_increasing_l1315_131540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l1315_131569

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -2 * x + 2 else (1/3)^x + 1

-- State the theorem
theorem inequality_range (x : ℝ) :
  f x + f (x + 1/2) > 2 ↔ x < 1/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l1315_131569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_division_l1315_131590

theorem weight_division (weights : Fin 2009 → ℕ) 
  (weight_bound : ∀ i, weights i ≤ 1000)
  (neighbor_diff : ∀ i : Fin 2008, weights i.succ = weights i + 1 ∨ weights i = weights i.succ + 1)
  (total_even : Even (Finset.sum Finset.univ weights)) :
  ∃ (group : Fin 2009 → Bool),
    Finset.sum (Finset.filter (λ i => group i = true) Finset.univ) weights = 
    Finset.sum (Finset.filter (λ i => group i = false) Finset.univ) weights :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_division_l1315_131590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_open_cube_formations_l1315_131561

/-- Represents a square in a 2D plane -/
structure Square where
  side : ℝ
  position : ℝ × ℝ

/-- Represents an L-shaped polygon composed of 4 congruent squares -/
structure LShape where
  squares : Fin 4 → Square
  congruent : ∀ i j : Fin 4, (squares i).side = (squares j).side

/-- Represents a position where an additional square can be attached -/
structure AttachmentPosition where
  coord : ℝ × ℝ

/-- Represents a polygon formed by attaching an additional square to the L-shape -/
structure ResultingPolygon where
  base : LShape
  additional_square : Square
  attachment_position : AttachmentPosition

/-- Predicate to determine if a resulting polygon can be folded into a cube with one face missing -/
def can_fold_to_open_cube (p : ResultingPolygon) : Prop :=
  sorry -- Definition of this predicate would involve complex geometric reasoning

/-- The main theorem stating that exactly 5 resulting polygons can form an open cube -/
theorem five_open_cube_formations
  (l : LShape)
  (positions : Fin 9 → AttachmentPosition) :
  ∃! (valid_positions : Finset AttachmentPosition),
    Finset.card valid_positions = 5 ∧
    (∀ pos, pos ∈ valid_positions ↔
      can_fold_to_open_cube ⟨l, l.squares 0, pos⟩) := by
  sorry

#check five_open_cube_formations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_open_cube_formations_l1315_131561
