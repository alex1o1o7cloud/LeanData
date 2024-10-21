import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_image_of_point_6_neg3_l1147_114784

noncomputable def f (x y : ℝ) : ℝ × ℝ := (Real.sqrt (1 / (x * y + 6 * y^2)), x^2 + y^3)

theorem image_of_point_6_neg3 : f 6 (-3) = (1/6, 9) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_image_of_point_6_neg3_l1147_114784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_regular_l1147_114781

-- Define the parameter t
variable (t : ℝ)

-- Define x and y as functions of t
noncomputable def x (t : ℝ) : ℝ := Real.sqrt t + 1
noncomputable def y (t : ℝ) : ℝ := 1 - 2 * Real.sqrt t

-- Theorem statement
theorem parametric_to_regular :
  (∀ t ≥ 0, 2 * x t + y t - 3 = 0) ∧
  (∀ t ≥ 0, x t ≥ 1) := by
  sorry

#check parametric_to_regular

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_regular_l1147_114781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1147_114722

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - 2*x else ((-x)^2 - 2*(-x))

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) : 
  (∀ y : ℝ, f y = f (-y)) →  -- f is even
  (∀ z : ℝ, z ≥ 0 → f z = z^2 - 2*z) →  -- definition for x ≥ 0
  (f (x + 1) < 3 ↔ -4 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1147_114722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_of_interest_is_four_percent_l1147_114766

/-- Calculate the rate of interest given principal and simple interest --/
noncomputable def calculate_rate_of_interest (principal : ℝ) (time : ℝ) (simple_interest : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

/-- Theorem: Given the conditions, the rate of interest is 4% --/
theorem rate_of_interest_is_four_percent (principal : ℝ) (time : ℝ) (simple_interest : ℝ)
  (h1 : principal = 2400)
  (h2 : time = 5)
  (h3 : simple_interest = principal - 1920) :
  calculate_rate_of_interest principal time simple_interest = 4 := by
  sorry

#check rate_of_interest_is_four_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_of_interest_is_four_percent_l1147_114766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_jungle_exploration_time_l1147_114764

/-- 
Given:
- John spent some time exploring the jungle.
- He then spent half as much time writing up notes about his travels.
- It took 0.5 years to write his book once he was done with the notes.
- He spent a total of 5 years on his book and exploring.

Prove that John spent 3 years exploring the jungle.
-/
theorem john_jungle_exploration_time : 
  ∀ (exploration_time : ℝ),
  exploration_time > 0 →
  let notes_time := exploration_time / 2
  let book_writing_time := 0.5
  exploration_time + notes_time + book_writing_time = 5 →
  exploration_time = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_jungle_exploration_time_l1147_114764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z1_div_z2_in_fourth_quadrant_l1147_114729

def Z₁ : ℂ := 2 + Complex.I
def Z₂ : ℂ := 1 + Complex.I

theorem z1_div_z2_in_fourth_quadrant :
  let z := Z₁ / Z₂
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z1_div_z2_in_fourth_quadrant_l1147_114729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1147_114789

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (x + 1) else -x^2 + 2*x

-- State the theorem
theorem inequality_solution_set :
  ∀ x : ℝ, f (2*x - 1) > f (2 - x) ↔ x > 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1147_114789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_12_sample_l1147_114743

def total_sample : ℕ := 1200
def grade_10_sample : ℕ := 240

def grade_ratio (k : ℕ) : Fin 3 → ℕ
| 0 => k
| 1 => 5
| 2 => 3

def grade_ratio_sum (k : ℕ) : ℕ := (grade_ratio k 0) + (grade_ratio k 1) + (grade_ratio k 2)

theorem grade_12_sample (k : ℕ) 
  (h1 : (grade_10_sample : ℚ) / total_sample = (grade_ratio k 0 : ℚ) / grade_ratio_sum k) : 
  (total_sample : ℚ) * (grade_ratio k 2) / grade_ratio_sum k = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_12_sample_l1147_114743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1147_114755

-- Define the constants
noncomputable def a : ℝ := (3/2) ^ (1/10 : ℝ)
noncomputable def b : ℝ := (3/2) ^ (1/5 : ℝ)
noncomputable def c : ℝ := (9/4) ^ (1/25 : ℝ)

-- State the theorem
theorem relationship_abc : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1147_114755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_x_y_value_l1147_114739

theorem log_x_y_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2*y^2 + x^2*y - 4*x*y - Real.exp 1*x*y*(Real.log x) + 4 = 0) : 
  Real.log y / Real.log x = Real.log 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_x_y_value_l1147_114739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_removal_calculation_l1147_114769

/-- Represents the dimensions and properties of a swimming pool and hot tub system -/
structure PoolSystem where
  pool_length : ℝ
  pool_width : ℝ
  pool_depth_change : ℝ
  hot_tub_diameter : ℝ
  hot_tub_depth_change : ℝ
  pi : ℝ
  cubic_feet_to_gallons : ℝ

/-- Calculates the total volume of water to be removed from the pool system in gallons -/
noncomputable def total_water_removed (s : PoolSystem) : ℝ :=
  let pool_volume := s.pool_length * s.pool_width * s.pool_depth_change
  let hot_tub_volume := s.pi * (s.hot_tub_diameter / 2)^2 * s.hot_tub_depth_change
  (pool_volume + hot_tub_volume) * s.cubic_feet_to_gallons

/-- Theorem stating the total volume of water to be removed from the pool system -/
theorem water_removal_calculation (s : PoolSystem) 
  (h1 : s.pool_length = 100)
  (h2 : s.pool_width = 50)
  (h3 : s.pool_depth_change = 2)
  (h4 : s.hot_tub_diameter = 10)
  (h5 : s.hot_tub_depth_change = 1)
  (h6 : s.pi = 3.1416)
  (h7 : s.cubic_feet_to_gallons = 7.5) :
  total_water_removed s = 75589.05 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_removal_calculation_l1147_114769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_implies_m_geq_4_l1147_114700

theorem sufficient_condition_implies_m_geq_4 (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 8 < 0 → x < m) ∧ 
  ¬(∀ x : ℝ, x < m → x^2 - 2*x - 8 < 0) →
  m ≥ 4 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_condition_implies_m_geq_4_l1147_114700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_contains_equal_differences_l1147_114712

open Set Finset

def S : Finset ℕ := range 20

theorem subset_contains_equal_differences (A : Finset ℕ) (hA : A ⊆ S) (hCard : A.card = 10) :
  ∃ (a b c d : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a - b = c - d :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_contains_equal_differences_l1147_114712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_price_theorem_l1147_114791

def original_suit_price : ℝ := 200
def original_tie_price : ℝ := 50
def suit_price_increase : ℝ := 0.3
def tie_price_increase : ℝ := 0.2
def suit_discount : ℝ := 0.3
def tie_discount : ℝ := 0.1
def sales_tax_rate : ℝ := 0.07

def increased_suit_price : ℝ := original_suit_price * (1 + suit_price_increase)
def increased_tie_price : ℝ := original_tie_price * (1 + tie_price_increase)

def discounted_suit_price : ℝ := increased_suit_price * (1 - suit_discount)
def discounted_tie_price : ℝ := increased_tie_price * (1 - tie_discount)

def combined_discounted_price : ℝ := discounted_suit_price + discounted_tie_price

def final_price_with_tax : ℝ := combined_discounted_price * (1 + sales_tax_rate)

theorem final_price_theorem : ∃ (ε : ℝ), abs (final_price_with_tax - 252.52) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_price_theorem_l1147_114791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1147_114719

-- Define the length of the train in meters
noncomputable def train_length : ℝ := 140

-- Define the time taken to cross the telegraph post in seconds
noncomputable def crossing_time : ℝ := 16

-- Define the conversion factor from m/s to km/h
noncomputable def ms_to_kmh : ℝ := 3.6

-- Define the function to calculate the speed in km/h
noncomputable def train_speed : ℝ :=
  (train_length / crossing_time) * ms_to_kmh

-- Theorem statement
theorem train_speed_calculation :
  train_speed = 31.5 := by
  -- Unfold the definitions
  unfold train_speed train_length crossing_time ms_to_kmh
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1147_114719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_AE_length_l1147_114779

structure Quadrilateral (A B C D E : ℝ × ℝ) : Prop where
  diagonals_intersect : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧
    A + t • (C - A) = B + t • (D - B)
  AB_eq_BE : ‖A - B‖ = ‖B - E‖
  AB_eq_5 : ‖A - B‖ = 5
  EC_eq_CD : ‖E - C‖ = ‖C - D‖
  EC_eq_7 : ‖E - C‖ = 7
  BC_eq_11 : ‖B - C‖ = 11

theorem quadrilateral_AE_length 
  (A B C D E : ℝ × ℝ) 
  (h : Quadrilateral A B C D E) : 
  ‖A - E‖ = 55 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_AE_length_l1147_114779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sides_l1147_114730

theorem triangle_sine_sides (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Triangle side lengths are positive
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  (a + b + c ≤ 2 * Real.pi) →  -- Perimeter condition
  (Real.sin a + Real.sin b > Real.sin c ∧ 
   Real.sin b + Real.sin c > Real.sin a ∧ 
   Real.sin c + Real.sin a > Real.sin b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sides_l1147_114730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1147_114799

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) →
  1/7 ≤ a ∧ a < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1147_114799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l1147_114780

-- Define the function f(x) = (x+1)^(-1)
noncomputable def f (x : ℝ) : ℝ := (x + 1)⁻¹

-- State the theorem
theorem f_decreasing_on_interval : 
  ∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f y < f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l1147_114780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_radius_l1147_114770

/-- Given a square inscribed in a circle, if the perimeter of the square
    in inches equals the area of the circle in square inches, then the
    radius of the circle is 2√2/π inches. -/
theorem square_circle_radius (s r : ℝ) :
  s > 0 →
  r > 0 →
  s * Real.sqrt 2 = 2 * r →
  4 * s = π * r^2 →
  r = 2 * Real.sqrt 2 / π :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_radius_l1147_114770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ending_digits_l1147_114742

def ends_with_same_digits (M : ℕ) : Prop :=
  ∃ (abcdef : ℕ), 0 < abcdef ∧ abcdef < 1000000 ∧
  M % 1000000 = abcdef ∧ (M * M) % 1000000 = abcdef

theorem unique_ending_digits (M : ℕ) (hM : M > 0) :
  ends_with_same_digits M → (M % 100000 = 60937) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ending_digits_l1147_114742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_area_ratio_l1147_114765

/-- A particle moving along the perimeter of a square -/
structure Particle where
  position : ℝ × ℝ
  velocity : ℝ

/-- The square on which the particles move -/
structure Square where
  side_length : ℝ

/-- The region traced by the midpoint of the line segment joining the two particles -/
def traced_region (p1 p2 : Particle) (s : Square) : Set (ℝ × ℝ) :=
  sorry

/-- The volume (area in 2D) of a set in ℝ × ℝ -/
noncomputable def volume (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem particle_movement_area_ratio :
  ∀ (s : Square) (p1 p2 : Particle) (v : ℝ),
    p1.position = (0, 0) →
    p2.position = (s.side_length, 0) →
    p1.velocity = v →
    p2.velocity = 2 * v →
    v > 0 →
    (volume (traced_region p1 p2 s)) / (s.side_length ^ 2) = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_area_ratio_l1147_114765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l1147_114782

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ Real.sqrt (-x^2 + x + 2)

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  ∃ (a b : ℝ), a = 1/2 ∧ b = 2 ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  (∀ x, x < a → ¬(∀ y, x < y → f x < f y)) ∧
  (∀ x, b < x → ¬(∀ y, x < y → f x < f y)) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l1147_114782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_asymptotes_specific_hyperbola_l1147_114741

/-- The angle between the asymptotes of a hyperbola -/
noncomputable def angle_between_asymptotes (a b : ℝ) : ℝ :=
  2 * Real.arctan (b / a)

/-- Theorem: The angle between the asymptotes of the hyperbola x^2 - y^2/3 = 1 is π/3 -/
theorem angle_asymptotes_specific_hyperbola :
  angle_between_asymptotes 1 (Real.sqrt 3) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_asymptotes_specific_hyperbola_l1147_114741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_calculation_l1147_114713

/-- Water pricing policy and usage calculation --/
theorem water_usage_calculation (a : ℝ) (h : a > 0) :
  ∃ x : ℝ, x > 0 ∧ 
    (if x ≤ 10 then a * x else a * 10 + 2 * a * (x - 10)) = 16 * a ∧ 
    x = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_usage_calculation_l1147_114713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l1147_114747

def inner_radius : ℝ := 6
def outer_radius : ℝ := 9

theorem area_between_circles : 
  (π * outer_radius^2) - (π * inner_radius^2) = 45 * π := by
  -- Expand the expressions
  have h1 : π * outer_radius^2 = 81 * π := by
    rw [outer_radius]
    ring
  have h2 : π * inner_radius^2 = 36 * π := by
    rw [inner_radius]
    ring
  -- Substitute and simplify
  rw [h1, h2]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_l1147_114747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_intersection_l1147_114776

-- Define the line l
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 1 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l'
def line_l' (m x y : ℝ) : Prop := m * x + y + (1/2) * m = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circle_tangent_and_intersection :
  ∀ m : ℝ,
  (∃ x y : ℝ, line_l x y ∧ circle_eq x y) ∧  -- Circle is tangent to line l
  (∃ x : ℝ, circle_eq x 0) ∧                 -- Circle is tangent to y-axis
  (∀ x y : ℝ, circle_eq x y → x ≥ 0) →       -- Center is on positive x-axis
  (∃ x1 y1 x2 y2 : ℝ,
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧
    line_l' m x1 y1 ∧ line_l' m x2 y2 ∧
    distance x1 y1 x2 y2 = Real.sqrt 3) ↔
  m = Real.sqrt 2 / 4 ∨ m = -Real.sqrt 2 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_intersection_l1147_114776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABG_in_regular_heptagon_l1147_114757

/-- The measure of an angle in a regular heptagon -/
noncomputable def regular_heptagon_angle : ℝ := (5 * 180) / 7

/-- The measure of angle ABG in a regular heptagon ABCDEFG -/
noncomputable def angle_ABG : ℝ := (180 - regular_heptagon_angle) / 2

/-- Theorem stating the measure of angle ABG in a regular heptagon ABCDEFG -/
theorem angle_ABG_in_regular_heptagon :
  angle_ABG = (180 - (5 * 180) / 7) / 2 :=
by
  -- Unfold the definitions
  unfold angle_ABG regular_heptagon_angle
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABG_in_regular_heptagon_l1147_114757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1147_114746

noncomputable section

open Real

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C

-- Define the main theorem
theorem triangle_theorem (t : Triangle) 
  (h : t.a * sin t.A + t.c * sin t.C - sqrt 2 * t.a * sin t.C = t.b * sin t.B) : 
  t.B = π/4 ∧ 
  (t.A = 5*π/12 ∧ t.b = 2 → t.a = 1 + sqrt 3 ∧ t.c = sqrt 6) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1147_114746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barrel_filling_trips_l1147_114749

noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

noncomputable def trips_needed (barrel_radius barrel_height bucket_radius : ℝ) : ℕ :=
  Int.toNat ⌈(cylinder_volume barrel_radius barrel_height) / (hemisphere_volume bucket_radius)⌉

theorem barrel_filling_trips :
  trips_needed 5 10 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barrel_filling_trips_l1147_114749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operations_result_l1147_114772

/-- Custom addition operation -/
noncomputable def custom_add (a b : ℝ) : ℝ := a + b - min a b

/-- Custom multiplication operation -/
noncomputable def custom_mul (a b : ℝ) : ℝ := a * b + max a b

/-- Theorem statement -/
theorem custom_operations_result :
  let a : ℝ := 3
  let b : ℝ := 5
  let c : ℝ := 2
  (custom_mul (custom_add a b) c) = 15 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_operations_result_l1147_114772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_winnings_is_four_fifths_l1147_114716

/-- A peculiar die with specific probabilities and payoffs -/
structure PeculiarDie where
  /-- Probability of rolling a 6 -/
  prob_six : ℚ
  /-- Probability of rolling any number from 1 to 5 -/
  prob_others : ℚ
  /-- Winnings when rolling a 6 -/
  win_six : ℚ
  /-- Losses when rolling an even number (2 or 4) -/
  loss_even : ℚ
  /-- Winnings when rolling an odd number (1, 3, or 5) -/
  win_odd : ℚ

/-- The specific peculiar die described in the problem -/
def specificDie : PeculiarDie :=
  { prob_six := 1/4
  , prob_others := 1/5
  , win_six := 4
  , loss_even := -2
  , win_odd := 1 }

/-- Expected winnings from rolling the peculiar die -/
def expectedWinnings (d : PeculiarDie) : ℚ :=
  d.prob_six * d.win_six +
  2 * d.prob_others * d.loss_even +
  3 * d.prob_others * d.win_odd

/-- Theorem stating that the expected winnings from rolling the specific die is 4/5 -/
theorem expected_winnings_is_four_fifths :
  expectedWinnings specificDie = 4/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_winnings_is_four_fifths_l1147_114716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_calculation_l1147_114702

-- Define the ※ operation
noncomputable def star (m n : ℝ) : ℝ :=
  if m ≥ n then Real.sqrt m - Real.sqrt n else Real.sqrt m + Real.sqrt n

-- Theorem statement
theorem star_calculation :
  (star 27 18) * (star 2 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_calculation_l1147_114702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equality_in_subset_l1147_114727

theorem sum_equality_in_subset (S : Finset ℕ) : 
  S.card = 16 → (∀ n, n ∈ S → n ≤ 100) → 
  ∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a + b = c + d :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equality_in_subset_l1147_114727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABF_l1147_114777

/-- Given a square ABCD with area 12, where AE = ED and EF = 2FC, prove that the area of triangle ABF is 5. -/
theorem area_triangle_ABF (A B C D E F : ℝ × ℝ) : 
  let square_area : ℝ := 12
  let square_side : ℝ := Real.sqrt square_area
  let AE_length : ℝ := square_side / 2
  let EC_length : ℝ := square_side
  let EF_length : ℝ := (2 / 3) * EC_length
  let FC_length : ℝ := (1 / 3) * EC_length
  ∀ (area_ABCD : ℝ) (length_AE length_ED length_EF length_FC : ℝ),
  area_ABCD = square_area →
  length_AE = length_ED →
  length_EF = 2 * length_FC →
  5 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_ABF_l1147_114777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_value_l1147_114763

/-- The function f(x, y) represents the left side of the inequality -/
noncomputable def f (x y : ℝ) : ℝ := y^2 + (2*x - 5)*y - x^2 * (Real.log x - Real.log y)

/-- Theorem stating that 5/2 is the only value of x that satisfies the inequality for all y > 0 -/
theorem unique_x_value : 
  (∃ x : ℝ, ∀ y : ℝ, y > 0 → f x y ≤ 0) ↔ (∃! x : ℝ, x = 5/2 ∧ ∀ y : ℝ, y > 0 → f x y ≤ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_value_l1147_114763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_l1147_114718

/-- The value of a as defined in the problem -/
noncomputable def a : ℝ := (1 / Real.pi) * ∫ (x : ℝ) in Set.Icc (-1) 1, (Real.sqrt (1 - x^2) + Real.sin x)

/-- The binomial expression -/
noncomputable def binomial (x : ℝ) : ℝ := (2*x - a/x^2)^9

/-- Theorem stating the constant term in the expansion of the binomial -/
theorem constant_term_of_binomial : 
  ∃ (p : ℝ → ℝ), (∀ x, binomial x = p x) ∧ (p 0 = -672) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_l1147_114718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_malfunctioning_clock_l1147_114775

/-- Represents a time in HH:MM format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Function to check if a digit can result in the given digit after malfunction -/
def can_result_in (original current : Digit) : Prop :=
  (original.val + 1 = current.val) ∨ (original.val + 9 = current.val) ∨
  (original.val = 9 ∧ current.val = 0) ∨ (original.val = 0 ∧ current.val = 9)

/-- Theorem stating that if 09:09 is displayed after malfunction, the original time was 18:18 -/
theorem malfunctioning_clock (displayed : Time) 
  (h : displayed.hours = 9 ∧ displayed.minutes = 9) :
  ∃ (original : Time),
    (can_result_in ⟨1, by norm_num⟩ ⟨0, by norm_num⟩) ∧
    (can_result_in ⟨8, by norm_num⟩ ⟨9, by norm_num⟩) ∧
    (can_result_in ⟨1, by norm_num⟩ ⟨0, by norm_num⟩) ∧
    (can_result_in ⟨8, by norm_num⟩ ⟨9, by norm_num⟩) ∧
    original.hours = 18 ∧ original.minutes = 18 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_malfunctioning_clock_l1147_114775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_point_centers_coincide_l1147_114701

-- Define the basic elements
variable (A B C I O : ℝ × ℝ)
variable (A' B' C' : ℝ × ℝ)

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Set (ℝ × ℝ) := {p | p = A ∨ p = B ∨ p = C}

-- Define incenter and circumcenter
def is_incenter (I : ℝ × ℝ) (triangle : Set (ℝ × ℝ)) : Prop := sorry

def is_circumcenter (O : ℝ × ℝ) (triangle : Set (ℝ × ℝ)) : Prop := sorry

-- Define reflection across a line
def reflect_across_line (point line_start line_end : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define nine-point center
noncomputable def nine_point_center (triangle : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- State the theorem
theorem nine_point_centers_coincide 
  (h_incenter : is_incenter I (triangle_ABC A B C))
  (h_circumcenter : is_circumcenter O (triangle_ABC A B C))
  (h_A' : A' = reflect_across_line O A I)
  (h_B' : B' = reflect_across_line O B I)
  (h_C' : C' = reflect_across_line O C I) :
  nine_point_center (triangle_ABC A B C) = nine_point_center (triangle_ABC A' B' C') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_point_centers_coincide_l1147_114701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_classification_l1147_114783

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

/-- Sequence A -/
def seq_A : ℕ → ℝ
| 0 => 1
| 1 => 4
| 2 => 7
| 3 => 10
| _ => 0  -- for completeness, though we only care about the first 4 terms

/-- Sequence B -/
noncomputable def seq_B : ℕ → ℝ
| 0 => Real.log 2
| 1 => Real.log 4
| 2 => Real.log 8
| 3 => Real.log 16
| _ => 0  -- for completeness, though we only care about the first 4 terms

/-- Sequence C -/
def seq_C : ℕ → ℝ
| 0 => 2^5
| 1 => 2^4
| 2 => 2^3
| 3 => 2^2
| _ => 0  -- for completeness, though we only care about the first 4 terms

/-- Sequence D -/
def seq_D : ℕ → ℝ
| 0 => 10
| 1 => 8
| 2 => 6
| 3 => 4
| 4 => 2
| _ => 0  -- for completeness, though we only care about the first 5 terms

theorem arithmetic_sequence_classification :
  is_arithmetic_sequence seq_A ∧
  is_arithmetic_sequence seq_B ∧
  ¬is_arithmetic_sequence seq_C ∧
  is_arithmetic_sequence seq_D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_classification_l1147_114783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_child_cakes_l1147_114778

/-- The number of cakes Martha bought -/
def total_cakes : ℕ := 60

/-- The number of children Martha has -/
def num_children : ℕ := 5

/-- The fraction of cakes the oldest child gets -/
def oldest_fraction : ℚ := 1/4

/-- The fraction of cakes the second oldest child gets -/
def second_oldest_fraction : ℚ := 3/10

/-- The fraction of cakes the middle child gets -/
def middle_fraction : ℚ := 1/6

/-- The fraction of cakes the second youngest child gets -/
def second_youngest_fraction : ℚ := 1/5

/-- The number of cakes the youngest child gets -/
def youngest_cakes : ℕ := total_cakes - 
  (Int.toNat ⌊oldest_fraction * total_cakes⌋ + 
   Int.toNat ⌊second_oldest_fraction * total_cakes⌋ + 
   Int.toNat ⌊middle_fraction * total_cakes⌋ + 
   Int.toNat ⌊second_youngest_fraction * total_cakes⌋)

theorem youngest_child_cakes : youngest_cakes = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_child_cakes_l1147_114778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l1147_114704

def N : ℕ := 2^5 * 3^3 * 5^3 * 7^2 * 11

theorem number_of_factors : (Finset.filter (· ∣ N) (Finset.range (N + 1))).card = 576 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_l1147_114704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_magnitude_l1147_114767

theorem order_of_magnitude : 
  let a : ℝ := (3/4)^(-(1/3 : ℝ))
  let b : ℝ := (3/4)^(-(1/4 : ℝ))
  let c : ℝ := (3/2)^(-(1/4 : ℝ))
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_magnitude_l1147_114767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l1147_114748

def numbers : List ℕ := [12, 13, 14, 15, 16, 17, 18, 19, 20]

def is_valid_arrangement (arr : List ℕ) : Prop :=
  arr.length = 9 ∧
  (∀ n ∈ arr, n ∈ numbers) ∧
  (∀ i, i < 7 → (arr.get? i).isSome → (arr.get? (i+1)).isSome → (arr.get? (i+2)).isSome →
    ((arr.get? i).get! + (arr.get? (i+1)).get! + (arr.get? (i+2)).get!) % 3 = 0)

def count_valid_arrangements : ℕ := sorry

theorem valid_arrangements_count :
  count_valid_arrangements = 216 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l1147_114748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_extreme_points_inequality_l1147_114760

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log (x + 1)

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2 * x + a / (x + 1)

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ici (1/2), f_deriv a x ≥ 0) → a ≥ -3/2 := by sorry

theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0 →
  x₁ * Real.log (2 / Real.sqrt (Real.exp 1)) < f a x₂ ∧ f a x₂ < 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_extreme_points_inequality_l1147_114760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1147_114759

/-- The number of days it takes A to finish the work alone -/
noncomputable def days_A : ℝ := sorry

/-- The fraction of work A can do in one day -/
noncomputable def work_rate_A : ℝ := 1 / days_A

/-- The fraction of work B can do in one day -/
noncomputable def work_rate_B : ℝ := 2 / days_A

/-- The combined work rate of A and B -/
noncomputable def combined_work_rate : ℝ := work_rate_A + work_rate_B

theorem work_completion_time :
  combined_work_rate = 0.375 → days_A = 8 := by
  sorry

#eval "Theorem defined successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1147_114759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_3_l1147_114706

def a : ℕ → ℚ
  | 0 => 3  -- Add this case to handle Nat.zero
  | 1 => 3
  | n + 1 => (a n - 1) / a n

theorem a_2017_equals_3 : a 2017 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_3_l1147_114706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l1147_114709

-- Define a structure for a triangle with integral sides
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  sum_eq_12 : a + b + c = 12
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the area function for a triangle
noncomputable def area (t : IntTriangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

-- Theorem statement
theorem triangle_area_is_six :
  ∀ t : IntTriangle, area t = 6 := by
  sorry

#check triangle_area_is_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l1147_114709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_rose_shampoo_l1147_114750

/-- The amount of rose shampoo Janet has -/
def rose_shampoo : ℚ := sorry

/-- The total amount of shampoo Janet has -/
def total_shampoo : ℚ := sorry

/-- The daily usage of shampoo -/
def daily_usage : ℚ := 1/12

/-- The number of days the shampoo will last -/
def days : ℕ := 7

/-- The amount of jasmine shampoo Janet has -/
def jasmine_shampoo : ℚ := 1/4

theorem janet_rose_shampoo :
  (total_shampoo = rose_shampoo + jasmine_shampoo) →
  (total_shampoo = days * daily_usage) →
  rose_shampoo = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_rose_shampoo_l1147_114750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_match_probability_l1147_114796

-- Define the probability of player A winning a single game
def p_win : ℚ := 2/3

-- Define the number of games needed to win the match
def games_to_win : ℕ := 3

-- Define the total number of games played in a 3:1 victory
def total_games : ℕ := 4

-- Define the number of games A loses in a 3:1 victory
def games_lost : ℕ := 1

-- Theorem statement
theorem badminton_match_probability :
  (Nat.choose total_games games_lost : ℚ) * p_win^(total_games - games_lost) * (1 - p_win)^games_lost = 8/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_match_probability_l1147_114796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_packaging_analysis_l1147_114731

/-- Represents the difference from standard weight and the number of bags with that difference -/
structure WeightDifference :=
  (difference : Int)
  (count : Nat)

/-- The problem statement -/
theorem food_packaging_analysis 
  (differences : List WeightDifference)
  (h_differences : differences = [
    ⟨-5, 1⟩, ⟨-2, 4⟩, ⟨0, 3⟩, ⟨1, 4⟩, ⟨3, 5⟩, ⟨6, 3⟩
  ])
  (standard_weight : Nat)
  (h_standard_weight : standard_weight = 450)
  (total_bags : Nat)
  (h_total_bags : total_bags = List.sum (List.map WeightDifference.count differences))
  (h_twenty_bags : total_bags = 20) :
  let total_weight := List.sum (List.map (fun d => (standard_weight + d.difference.toNat) * d.count) differences)
  let average_weight : Rat := total_weight / total_bags
  let acceptable_bags := List.sum (List.map WeightDifference.count (List.filter (fun d => d.difference.natAbs ≤ 5) differences))
  (total_weight = 9024 ∧ 
   average_weight = 451.2 ∧ 
   (acceptable_bags : Rat) / total_bags = 17/20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_packaging_analysis_l1147_114731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_distance_bound_l1147_114773

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.exp x) - x^2 + Real.pi * x

theorem zeros_distance_bound (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ ∈ Set.Icc 0 Real.pi →
  x₂ ∈ Set.Icc 0 Real.pi →
  f x₁ = m →
  f x₂ = m →
  x₁ ≠ x₂ →
  |x₂ - x₁| ≤ Real.pi - (2 * m) / (Real.pi + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_distance_bound_l1147_114773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Y_Z_calculation_l1147_114723

-- Define operation Y
def Y (a b : Int) : Int := a^2 - 2*a*b + b^2

-- Define operation Z
def Z (a b : Int) : Int := a*b + a + b

-- Theorem statement
theorem Y_Z_calculation : Z (Y 5 3) (Y 2 1) = 9 := by
  -- Evaluate Y 5 3
  have h1 : Y 5 3 = 4 := by
    unfold Y
    norm_num
  
  -- Evaluate Y 2 1
  have h2 : Y 2 1 = 1 := by
    unfold Y
    norm_num
  
  -- Evaluate Z (Y 5 3) (Y 2 1)
  unfold Z
  rw [h1, h2]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Y_Z_calculation_l1147_114723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1147_114752

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x - 1 / 2

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  f A = 1 →
  0 < A →
  A < π / 2 →
  a = 2 * Real.sqrt 3 →
  c = 4 →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  A = π / 3 ∧ b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1147_114752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repunit_sum_seq_111_l1147_114720

/-- A repunit is a positive integer consisting entirely of the digit 1 -/
def is_repunit (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

/-- The sequence of repunits -/
def repunit (n : ℕ) : ℕ :=
  (10^(n+1) - 1) / 9

/-- The sequence of positive integers that can be expressed as the sum of distinct repunits -/
def repunit_sum_seq : ℕ → ℕ
| 0 => 0
| (n+1) => 
    let binary := Nat.digits 2 n
    (List.range binary.length).filter (λ i => binary.get? i = some 1)
      |>.map repunit
      |>.sum

/-- The 111th term of the repunit sum sequence is 1223456 -/
theorem repunit_sum_seq_111 : repunit_sum_seq 111 = 1223456 :=
by sorry

#eval repunit_sum_seq 111

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repunit_sum_seq_111_l1147_114720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthic_inradius_bound_l1147_114708

/-- Given a triangle ABC with circumradius 1, this structure represents its properties --/
structure Triangle where
  r : ℝ  -- inradius of triangle ABC
  p : ℝ  -- inradius of the orthic triangle of ABC

/-- The theorem states that for any triangle with circumradius 1, 
    the inradius of its orthic triangle is bounded by 1 - (1 + r)² --/
theorem orthic_inradius_bound (t : Triangle) : t.p ≤ 1 - (1 + t.r)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthic_inradius_bound_l1147_114708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_k_l1147_114717

-- Define the solution set A
noncomputable def A (k : ℝ) : Set ℝ := {x : ℝ | x * (1 + Real.log x) + 2 * k > k * x}

-- Define the function h(x)
noncomputable def h (x : ℝ) : ℝ := (x * (1 + Real.log x)) / (x - 2)

theorem max_integer_k : ∃ (k : ℤ), (∀ (k' : ℤ), (∀ (x : ℝ), x > 2 → x ∈ A k') → k' ≤ k) ∧ 
                                   (∀ (x : ℝ), x > 2 → x ∈ A k) ∧
                                   k = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_k_l1147_114717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sine_cosine_l1147_114774

theorem max_value_sine_cosine :
  ∃ (max_x : Real), 
    (∀ x : Real, 0 ≤ x ∧ x < 2 * Real.pi → 
      Real.sin x - Real.sqrt 3 * Real.cos x ≤ Real.sin max_x - Real.sqrt 3 * Real.cos max_x) ∧
    max_x = 5 * Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sine_cosine_l1147_114774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_probabilities_l1147_114705

structure MathCompetition where
  p_algebra : ℚ
  p_geometry : ℚ
  p_combinatorics : ℚ
  points_algebra : ℕ
  points_geometry : ℕ
  points_combinatorics : ℕ
  deduction : ℕ
  starting_score : ℕ

def MathCompetition.default : MathCompetition where
  p_algebra := 3/4
  p_geometry := 2/3
  p_combinatorics := 1/2
  points_algebra := 30
  points_geometry := 30
  points_combinatorics := 40
  deduction := 10
  starting_score := 0

theorem math_competition_probabilities (comp : MathCompetition := MathCompetition.default) 
  (h_all_correct : comp.p_algebra * comp.p_geometry * comp.p_combinatorics = 1/6) :
  let p_one_correct := comp.p_algebra * (1 - comp.p_geometry) * (1 - comp.p_combinatorics) +
                       (1 - comp.p_algebra) * comp.p_geometry * (1 - comp.p_combinatorics) +
                       (1 - comp.p_algebra) * (1 - comp.p_geometry) * comp.p_combinatorics
  let p_prize := comp.p_algebra * comp.p_geometry * comp.p_combinatorics +
                 comp.p_algebra * comp.p_geometry * (1 - comp.p_combinatorics) +
                 comp.p_algebra * (1 - comp.p_geometry) * comp.p_combinatorics +
                 (1 - comp.p_algebra) * comp.p_geometry * comp.p_combinatorics
  p_one_correct = 11/36 ∧ p_prize = 3/4 := by
  sorry

#check math_competition_probabilities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_probabilities_l1147_114705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1147_114703

noncomputable def line_equation (α : ℝ) (b : ℝ) (x y : ℝ) : Prop :=
  x * Real.sin α + Real.sqrt 3 * y - b = 0

noncomputable def inclination_angle (α : ℝ) : ℝ :=
  Real.arctan (-Real.sin α / Real.sqrt 3)

theorem inclination_angle_range :
  ∀ α b : ℝ, 
    (∃ x y : ℝ, line_equation α b x y) →
    (0 ≤ inclination_angle α ∧ inclination_angle α ≤ Real.pi / 6) ∨
    (5 * Real.pi / 6 ≤ inclination_angle α ∧ inclination_angle α < Real.pi) :=
by
  sorry

#check inclination_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l1147_114703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1147_114798

/-- An ellipse in a Cartesian coordinate plane -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt ((e.a^2 - e.b^2) / e.a^2)

/-- Helper function to define points on the ellipse -/
def ellipse_set (e : Ellipse) : Set (ℝ × ℝ) :=
  {p | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1}

/-- Theorem: Eccentricity of an ellipse with specific geometric properties -/
theorem ellipse_eccentricity (e : Ellipse) 
  (h_perp : ∃ (B C : ℝ × ℝ), B ∈ ellipse_set e ∧ C ∈ ellipse_set e ∧ 
    (B.1 - (-e.a * eccentricity e))^2 + B.2^2 = (e.a * eccentricity e)^2)
  (h_right_angle : ∃ (F₂ : ℝ × ℝ), F₂ = (e.a * eccentricity e, 0) ∧
    ∃ (B C : ℝ × ℝ), B ∈ ellipse_set e ∧ C ∈ ellipse_set e ∧
    (B.1 - F₂.1) * (C.1 - F₂.1) + (B.2 - F₂.2) * (C.2 - F₂.2) = 0) :
  eccentricity e = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1147_114798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cut_condition_l1147_114710

/-- A rectangle can be cut into smaller rectangles if and only if the dimensions satisfy certain divisibility conditions. -/
theorem rectangle_cut_condition (a b α β : ℝ) (ha : a > 0) (hb : b > 0) (hα : α > 0) (hβ : β > 0) :
  (∃ (m n : ℕ), a = m * α ∧ b = n * β) ∨ (∃ (m n : ℕ), a = m * β ∧ b = n * α) ↔
  (∃ (k : ℕ+), a = k * α ∧ ∃ (l : ℕ+), b = l * β) ∨ (∃ (k : ℕ+), a = k * β ∧ ∃ (l : ℕ+), b = l * α) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_cut_condition_l1147_114710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_exponential_equation_l1147_114715

theorem infinite_solutions_exponential_equation :
  {(x, y) : ℝ × ℝ | (9 : ℝ)^(x^2 - y) + (9 : ℝ)^(x - y^2) = 1}.Infinite :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_exponential_equation_l1147_114715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_triangle_l1147_114762

/-- The radius of the incircle of a triangle with sides 5, 12, and 13 is 2 -/
theorem incircle_radius_of_triangle (a b c : ℝ) (ha : a = 5) (hb : b = 12) (hc : c = 13) :
  let s := (a + b + c) / 2
  let A := (s * (s - a) * (s - b) * (s - c)).sqrt
  A / s = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_triangle_l1147_114762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_one_range_l1147_114793

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x - 7 else Real.sqrt x

-- State the theorem
theorem f_less_than_one_range :
  {a : ℝ | f a < 1} = Set.Ioo (-3) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_one_range_l1147_114793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1147_114714

-- Define the functions f and g
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the transformation from g to f
noncomputable def transform (h : ℝ → ℝ) (x : ℝ) : ℝ := h ((x - Real.pi/6) / 2)

-- State the theorem
theorem function_transformation (x : ℝ) : 
  transform g x = Real.sin (4 * x + Real.pi/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l1147_114714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_l1147_114787

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle_between (a b : V) : ℝ := Real.arccos ((inner a b) / (norm a * norm b))

theorem magnitude_of_b (a b : V) 
  (h1 : angle_between a b = π / 4)
  (h2 : ‖a‖ = 1)
  (h3 : ‖2 • a - b‖ = Real.sqrt 10) :
  ‖b‖ = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_b_l1147_114787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_equals_4a_plus_4a_sqrt_2_l1147_114761

/-- The sum of perimeters of an infinite sequence of isosceles right triangles -/
noncomputable def sumOfPerimeters (a : ℝ) : ℝ :=
  let initialPerimeter := 2 * a + Real.sqrt 2 * a
  let ratio := (1 : ℝ) / 2
  initialPerimeter / (1 - ratio)

/-- The theorem stating the sum of perimeters equals 4a(1 + √2) -/
theorem sum_of_perimeters_equals_4a_plus_4a_sqrt_2 (a : ℝ) (h : a > 0) :
  sumOfPerimeters a = 4 * a * (1 + Real.sqrt 2) := by
  sorry

#check sum_of_perimeters_equals_4a_plus_4a_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_equals_4a_plus_4a_sqrt_2_l1147_114761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_volume_l1147_114721

/-- The volume of a cuboid with edges 4 cm, 5 cm, and 6 cm is 120 cubic centimeters. -/
theorem cuboid_volume (x y z : ℝ) : 
  x = 4 ∧ y = 5 ∧ z = 6 → x * y * z = 120 := by
  intro h
  rw [h.1, h.2.1, h.2.2]
  norm_num

#check cuboid_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_volume_l1147_114721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_implies_identity_l1147_114736

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f (x + 1) + y - 1) = f x + y) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_implies_identity_l1147_114736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_secant_theorem_l1147_114738

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the tangent line at A
variable (tangent : Set (ℝ × ℝ))

-- Define the line parallel to BC
variable (parallel_line : Set (ℝ × ℝ))

-- Define the intersection points
variable (M N P Q S : ℝ × ℝ)

-- Define the inscribed property
def is_inscribed (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) : Prop :=
  triangle.1 ∈ circle ∧ triangle.2.1 ∈ circle ∧ triangle.2.2 ∈ circle

-- Define the tangent property
def is_tangent (line : Set (ℝ × ℝ)) (point : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : Prop :=
  point ∈ line ∧ point ∈ circle ∧ ∀ x, x ∈ line → x ≠ point → x ∉ circle

-- Define the parallel property
def is_parallel (line1 line2 : Set (ℝ × ℝ)) : Prop :=
  ∃ v : ℝ × ℝ, ∀ x y, x ∈ line1 → y ∈ line1 → ∃ t : ℝ, y - x = t • v ∧
              ∀ x y, x ∈ line2 → y ∈ line2 → ∃ t : ℝ, y - x = t • v

-- Define the intersection property
def intersects_at (line : Set (ℝ × ℝ)) (point : ℝ × ℝ) (set : Set (ℝ × ℝ)) : Prop :=
  point ∈ line ∧ point ∈ set

-- Theorem statement
theorem tangent_secant_theorem
  (h_inscribed : is_inscribed (A, B, C) circle)
  (h_tangent : is_tangent tangent A circle)
  (h_parallel : is_parallel parallel_line {B, C})
  (h_M : intersects_at parallel_line M {A, B})
  (h_N : intersects_at parallel_line N {A, C})
  (h_P : intersects_at parallel_line P circle)
  (h_Q : intersects_at parallel_line Q circle)
  (h_S : intersects_at parallel_line S tangent) :
  (S.1 - M.1) * (S.2 - M.2) * ((S.1 - N.1) * (S.2 - N.2)) =
  (S.1 - P.1) * (S.2 - P.2) * ((S.1 - Q.1) * (S.2 - Q.2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_secant_theorem_l1147_114738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_f_zero_l1147_114707

-- Define the function f(x, y) = x^3 - xy
def f (x y : ℝ) : ℝ := x^3 - x*y

-- Define the maximum of |f(x, y)| over x ∈ [0, 1]
noncomputable def max_f (y : ℝ) : ℝ := 
  ⨆ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1), |f x y|

-- State the theorem
theorem min_max_f_zero : 
  ∃ (y : ℝ), max_f y = 0 ∧ ∀ (z : ℝ), max_f z ≥ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_f_zero_l1147_114707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_equals_n_l1147_114740

theorem subset_sum_equals_n (n : ℕ) (A B : Finset ℕ) 
  (h_subset_A : A ⊆ Finset.range n) 
  (h_subset_B : B ⊆ Finset.range n)
  (h_sum_cardinality : A.card + B.card > n - 1) :
  ∃ (a b : ℕ), a ∈ A ∧ b ∈ B ∧ a + b = n := by
  sorry

#check subset_sum_equals_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_equals_n_l1147_114740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_in_interval_l1147_114725

-- Define the function f
def f (b c x : ℝ) : ℝ := x^3 + b*x + c

-- State the theorem
theorem unique_root_in_interval (b c : ℝ) :
  (∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x < y → f b c x < f b c y) →  -- f is increasing on [-1, 1]
  (f b c (-1) * f b c 1 < 0) →                           -- f(-1) * f(1) < 0
  ∃! x, x ∈ Set.Icc (-1) 1 ∧ f b c x = 0 :=              -- Unique root in [-1, 1]
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_in_interval_l1147_114725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_triangle_perimeter_l1147_114751

/-- Triangle XYZ with inscribed circle --/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle --/
  r : ℝ
  /-- The length of XP --/
  xp : ℝ
  /-- The length of PY --/
  py : ℝ

/-- The perimeter of the triangle --/
noncomputable def perimeter (t : InscribedCircleTriangle) : ℝ :=
  2 * (t.xp + t.py + (t.xp + t.py) * t.r / (t.xp + t.py - t.r))

theorem inscribed_circle_triangle_perimeter :
  let t : InscribedCircleTriangle := { r := 15, xp := 30, py := 36 }
  perimeter t = 83.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_triangle_perimeter_l1147_114751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_error_l1147_114768

/-- Represents a regular hexagon -/
structure RegularHexagon where
  side : ℝ
  apothem : ℝ

/-- Calculates the area of a regular hexagon -/
noncomputable def area (h : RegularHexagon) : ℝ := 3 * h.side * h.apothem

/-- Calculates the area with measurement errors -/
noncomputable def areaWithErrors (h : RegularHexagon) : ℝ :=
  3 * (h.side * 1.06) * (h.apothem * 0.96)

/-- Calculates the percentage error in area -/
noncomputable def percentageError (h : RegularHexagon) : ℝ :=
  ((areaWithErrors h - area h) / area h) * 100

theorem hexagon_area_error (h : RegularHexagon) :
  ⌊percentageError h⌋₊ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_error_l1147_114768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_sum_l1147_114744

-- Define the triangle vertices
def D : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := (8, 0)
def F : ℝ × ℝ := (5, 7)

-- Define the estimated Fermat point
def P : ℝ × ℝ := (3, 3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem fermat_point_sum : 
  ∃ (p q r s : ℕ), 
    distance P D + distance P E + distance P F = p * Real.sqrt q + r * Real.sqrt s ∧ 
    p + r = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_sum_l1147_114744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_radius_properties_l1147_114732

/-- The radius of the smallest circle containing n points with given constraints -/
noncomputable def smallest_circle_radius (n : ℕ) : ℝ :=
  if 2 ≤ n ∧ n ≤ 7 then 1
  else if 8 ≤ n ∧ n ≤ 11 then 1 / (2 * Real.sin (Real.pi / (n - 1)))
  else 0  -- undefined for other values of n

/-- Theorem stating the properties of the smallest circle radius -/
theorem smallest_circle_radius_properties (n : ℕ) (h2 : 2 ≤ n) (h11 : n ≤ 11) :
  ∃ (points : Finset (ℝ × ℝ)),
    (points.card = n) ∧
    (∃ center : ℝ × ℝ, center ∈ points ∧ center = (0, 0)) ∧
    (∀ p q : ℝ × ℝ, p ∈ points → q ∈ points → p ≠ q → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ 1) ∧
    (∀ p : ℝ × ℝ, p ∈ points → p.1^2 + p.2^2 ≤ (smallest_circle_radius n)^2) ∧
    (∀ r : ℝ, r < smallest_circle_radius n →
      ¬∃ (points' : Finset (ℝ × ℝ)),
        (points'.card = n) ∧
        (∃ center : ℝ × ℝ, center ∈ points' ∧ center = (0, 0)) ∧
        (∀ p q : ℝ × ℝ, p ∈ points' → q ∈ points' → p ≠ q → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ 1) ∧
        (∀ p : ℝ × ℝ, p ∈ points' → p.1^2 + p.2^2 ≤ r^2)) := by
  sorry

#check smallest_circle_radius
#check smallest_circle_radius_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_radius_properties_l1147_114732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_81_is_5_l1147_114788

/-- The 81st digit after the decimal point in the decimal expansion of 325/999 -/
def digit_81 : ℕ :=
  let expansion := 325 / 999
  -- We'll use a placeholder function for now
  let nthDigit (n : ℕ) := n % 10  -- This is just a placeholder
  nthDigit 81

theorem digit_81_is_5 : digit_81 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_81_is_5_l1147_114788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_on_other_items_is_eight_percent_l1147_114758

/-- Represents the shopping breakdown and tax information --/
structure ShoppingInfo where
  clothing_percent : ℚ
  food_percent : ℚ
  other_percent : ℚ
  clothing_tax_rate : ℚ
  total_tax_rate : ℚ

/-- Calculates the tax rate on other items --/
def tax_rate_on_other_items (info : ShoppingInfo) : ℚ :=
  ((info.total_tax_rate - info.clothing_tax_rate * info.clothing_percent) / info.other_percent) * 100

/-- Theorem stating that the tax rate on other items is 8% --/
theorem tax_rate_on_other_items_is_eight_percent (info : ShoppingInfo)
  (h1 : info.clothing_percent = 1/2)
  (h2 : info.food_percent = 1/5)
  (h3 : info.other_percent = 3/10)
  (h4 : info.clothing_tax_rate = 1/25)
  (h5 : info.total_tax_rate = 11/250) :
  tax_rate_on_other_items info = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_on_other_items_is_eight_percent_l1147_114758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_difference_l1147_114745

-- Define the parabola and points
def parabola (x y : ℝ) : Prop := x^2 = 2*y

def point_P (x y : ℝ) : Prop := parabola x y

def point_Q (x y : ℝ) : Prop := x^2 = 4*y

def point_M : ℝ × ℝ := (-4, 4)

def point_N : ℝ × ℝ := (4, 5)

-- Define the line passing through N with slope k
def line_through_N (k x y : ℝ) : Prop := y = k*(x - 4) + 5

-- Define the slopes of MA and MB
noncomputable def slope_MA (x y : ℝ) : ℝ := (y - 4) / (x + 4)

noncomputable def slope_MB (x y : ℝ) : ℝ := (y - 4) / (x + 4)

-- Theorem statement
theorem min_slope_difference :
  ∀ k k1 k2 x1 y1 x2 y2 : ℝ,
  point_Q x1 y1 → point_Q x2 y2 →
  line_through_N k x1 y1 → line_through_N k x2 y2 →
  k1 = slope_MA x1 y1 → k2 = slope_MB x2 y2 →
  ∀ k' k1' k2' x1' y1' x2' y2' : ℝ,
  point_Q x1' y1' → point_Q x2' y2' →
  line_through_N k' x1' y1' → line_through_N k' x2' y2' →
  k1' = slope_MA x1' y1' → k2' = slope_MB x2' y2' →
  |k1 - k2| ≤ |k1' - k2'| →
  |k1 - k2| = 1 :=
by sorry

#check min_slope_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_slope_difference_l1147_114745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_trigonometric_expression_l1147_114786

open Real BigOperators

theorem sum_trigonometric_expression :
  (∑ x in Finset.range 44, 3 * sin (x + 3 : ℝ) * cos 1 * (1 + 1 / (sin ((x + 3 : ℝ) - 2) * sin ((x + 3 : ℝ) + 2)))) =
  (3 / 2) * (sin 4 - sin 2 + sin 4 / (sin 1 * sin 5) - sin 2 / (sin 45 * sin 47)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_trigonometric_expression_l1147_114786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l1147_114771

theorem diophantine_equation_solution :
  ∀ x y : ℕ, x > 0 → y > 0 →
  (x : ℤ)^(2*y - 1) + (x + 1 : ℤ)^(2*y - 1) = (x + 2 : ℤ)^(2*y - 1) → x = 1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l1147_114771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounded_iff_special_values_l1147_114728

noncomputable def sequenceA (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => let prev := sequenceA a n
              if prev = 0 then 0 else prev - 1 / prev

theorem sequence_bounded_iff_special_values (a : ℝ) :
  (∀ n : ℕ, |sequenceA a n| < 1) ↔ a = 0 ∨ a = Real.sqrt 2 / 2 ∨ a = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounded_iff_special_values_l1147_114728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1147_114724

theorem lambda_range (l : ℝ) : 
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 + 2 > l*(a + b)) → l < 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1147_114724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approximately_eight_l1147_114794

-- Define the cycling parameters
noncomputable def cycling_time : ℝ := 45 / 60  -- Convert 45 minutes to hours
def cycling_speed : ℝ := 12

-- Define the jogging parameters
def jogging_time : ℝ := 2
def jogging_speed : ℝ := 6

-- Calculate the total distance
noncomputable def total_distance : ℝ := cycling_time * cycling_speed + jogging_time * jogging_speed

-- Calculate the total time
noncomputable def total_time : ℝ := cycling_time + jogging_time

-- Calculate the average speed
noncomputable def average_speed : ℝ := total_distance / total_time

-- Theorem to prove
theorem average_speed_approximately_eight :
  ∃ ε > 0, |average_speed - 8| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approximately_eight_l1147_114794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1147_114734

/-- Given a rectangle ABCD with area 108 square meters, where E is the midpoint of AD,
    G is one-third of the way along CD from C to D, and F is the midpoint of BC,
    the area of trapezoid DEFG is 18 + 18/a square meters, where 'a' is the length of side AD. -/
theorem trapezoid_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 108) :
  (a / 2 + b / 3) * b / 2 = 18 + 18 / a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1147_114734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guards_cover_all_approaches_l1147_114753

-- Define the type for a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the type for a guard
structure Guard where
  position : Point
  direction : ℝ  -- angle in radians

-- Define the object to be guarded
def object : Point := { x := 0, y := 0 }

-- Define the visibility range of a guard
def visibilityRange : ℝ := 100

-- Function to check if a point is visible to a guard
def isVisible (guard : Guard) (point : Point) : Prop :=
  sorry

-- Function to check if a point is visible to any guard in a list
def isVisibleToAnyGuard (guards : List Guard) (point : Point) : Prop :=
  ∃ g ∈ guards, isVisible g point

-- Theorem stating that it's possible to position guards to cover all approaches
theorem guards_cover_all_approaches :
  ∃ (guards : List Guard),
    (∀ p : Point, (p.x - object.x) ^ 2 + (p.y - object.y) ^ 2 ≤ visibilityRange ^ 2 →
      isVisibleToAnyGuard guards p) ∧
    (∀ g : Guard, g ∈ guards → 
      ∀ p : Point, (p.x - g.position.x) ^ 2 + (p.y - g.position.y) ^ 2 ≤ visibilityRange ^ 2 →
        isVisibleToAnyGuard guards p) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_guards_cover_all_approaches_l1147_114753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1147_114792

/-- Represents the monthly sales volume in ten thousand units -/
noncomputable def x : ℝ → ℝ := sorry

/-- Represents the cost of investing in physical store experience installation in ten thousand yuan -/
noncomputable def t : ℝ → ℝ := sorry

/-- The relationship between monthly sales volume and cost investment -/
axiom sales_cost_relation (x' t' : ℝ) : x' = 3 - 2 / (t' + 1) → t' = 2 / (3 - x') - 1

/-- The fixed monthly expenses for the online store in ten thousand yuan -/
def fixed_expenses : ℝ := 3

/-- The purchase price per ten thousand units in ten thousand yuan -/
def purchase_price : ℝ := 32

/-- The selling price function in ten thousand yuan -/
noncomputable def selling_price (x' t' : ℝ) : ℝ := 1.5 * purchase_price + t' / (2 * x')

/-- The monthly profit function in ten thousand yuan -/
noncomputable def profit (x' : ℝ) : ℝ := 
  let t' := 2 / (3 - x') - 1
  selling_price x' t' * x' - purchase_price * x' - fixed_expenses - t'

/-- The theorem stating that the maximum monthly profit is 37.5 ten thousand yuan -/
theorem max_profit : 
  ∃ x' : ℝ, 1 < x' ∧ x' < 3 ∧ ∀ y : ℝ, 1 < y ∧ y < 3 → profit x' ≥ profit y ∧ profit x' = 37.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l1147_114792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_inequality_l1147_114790

theorem contrapositive_inequality :
  (∀ (a b c : ℝ), a > b → a + c > b + c) ↔ (∀ (a b c : ℝ), a + c ≤ b + c → a ≤ b) :=
by
  apply Iff.intro
  · intro h a b c hab
    contrapose! hab
    exact h a b c hab
  · intro h a b c hab
    contrapose! hab
    exact h a b c hab

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_inequality_l1147_114790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_sin_l1147_114711

theorem negation_of_existential_sin : 
  (¬ ∃ x : ℝ, Real.sin x > 1) ↔ (∀ x : ℝ, Real.sin x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_sin_l1147_114711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1147_114754

noncomputable def f (a x : ℝ) := Real.log (-x^2 + 4*a*x - 3*a^2)

def P (a x : ℝ) : Prop := -x^2 + 4*a*x - 3*a^2 > 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem problem_statement (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, a = 1 → P a x ∧ q x → 2 < x ∧ x < 3) ∧
  (Set.Icc 1 2 = {a : ℝ | a > 0 ∧ 
    (∀ x : ℝ, -x^2 + 4*a*x - 3*a^2 ≥ 0 → x ≤ 2 ∨ x ≥ 3) ∧
    ¬(∀ x : ℝ, x ≤ 2 ∨ x ≥ 3 → -x^2 + 4*a*x - 3*a^2 ≥ 0)}) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1147_114754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reach_C_l1147_114795

/-- Represents a dot on the lattice -/
inductive Dot
| Red
| Blue

/-- Represents a position on the lattice -/
structure Position where
  x : Int
  y : Int

/-- Represents the lattice structure -/
def LatticeStructure := Position → Dot

/-- Represents a path on the lattice -/
def PathOnLattice := List Position

/-- Defines the starting position A -/
def A : Position := ⟨0, 0⟩

/-- Defines the target position C -/
def C : Position := ⟨1, 1⟩

/-- Defines the position of dot E -/
def E : Position := ⟨0, 1⟩

/-- Checks if a path is valid according to the movement rules -/
def isValidPath (path : PathOnLattice) (lattice : LatticeStructure) : Prop := sorry

/-- Counts the number of valid paths to C -/
def countValidPathsToC (lattice : LatticeStructure) : Nat := sorry

/-- Calculates the total number of possible paths in 6 moves -/
def totalPossiblePaths : Nat := 4^6

/-- Theorem: The probability of reaching C is the ratio of valid paths to total possible paths -/
theorem probability_reach_C (lattice : LatticeStructure) :
  (countValidPathsToC lattice : ℚ) / totalPossiblePaths = (2 : ℚ) / 4^6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reach_C_l1147_114795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_2_pow_positive_l1147_114737

theorem negation_of_forall_2_pow_positive :
  (¬ (∀ x : ℝ, (2 : ℝ)^x > 0)) ↔ (∃ x₀ : ℝ, (2 : ℝ)^x₀ ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_2_pow_positive_l1147_114737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_treasure_dig_theorem_l1147_114756

/-- Represents a square on the board -/
structure Square where
  x : Fin 8
  y : Fin 8

/-- Represents the board state -/
structure Board where
  treasure : Square
  stepCount : Square → Nat

/-- Checks if two squares are adjacent -/
def adjacent (s1 s2 : Square) : Prop :=
  (s1.x = s2.x ∧ s1.y.val + 1 = s2.y.val) ∨
  (s1.x = s2.x ∧ s1.y.val = s2.y.val + 1) ∨
  (s1.x.val + 1 = s2.x.val ∧ s1.y = s2.y) ∨
  (s1.x.val = s2.x.val + 1 ∧ s1.y = s2.y)

/-- Checks if the step count is valid for a given square -/
def validStepCount (b : Board) (s : Square) : Prop :=
  s ≠ b.treasure →
  ∃ path : List Square,
    path.head? = some s ∧
    path.getLast? = some b.treasure ∧
    path.length - 1 = b.stepCount s ∧
    ∀ i : Fin (path.length - 1), adjacent (path[i]) (path[i.val + 1])

/-- The main theorem: at least 3 squares must be dug to guarantee finding the treasure -/
theorem treasure_dig_theorem (b : Board) 
  (h : ∀ s : Square, validStepCount b s) :
  ∀ dig_set : Finset Square,
    (∀ board : Board, (∀ s, validStepCount board s) → 
      board.treasure ∈ dig_set) →
    dig_set.card ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_treasure_dig_theorem_l1147_114756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1147_114785

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 2^x + 2^(a*x + b)

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (f a b 1 = 5/2) ∧
    (f a b 2 = 17/4) ∧
    (a = -1 ∧ b = 0) ∧
    (∀ x, f a b x = f a b (-x)) ∧
    (∀ x y, x < y → x < 0 → y < 0 → f a b x > f a b y) :=
by
  -- We use 'sorry' to skip the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1147_114785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contestant_arrangements_l1147_114726

/-- The number of possible arrangements for 5 contestants,
    where one contestant cannot go first and another cannot go last. -/
theorem contestant_arrangements :
  let n : ℕ := 5  -- Total number of contestants
  let restricted_first : ℕ := 1  -- Number of contestants who cannot go first
  let restricted_last : ℕ := 1  -- Number of contestants who cannot go last
  ∃ (total_arrangements : ℕ),
    total_arrangements = 78 ∧
    total_arrangements = (n - restricted_first - restricted_last).factorial +
                         (n - restricted_first - 1) * (n - restricted_last - 1) * (n - 2).factorial :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_contestant_arrangements_l1147_114726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_topsoil_cost_l1147_114797

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℚ := 8

/-- The volume of topsoil in cubic yards -/
def volume_cubic_yards : ℚ := 7

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_feet : ℚ := 27

/-- The discount rate for purchases exceeding 100 cubic feet -/
def discount_rate : ℚ := 1/10

/-- The minimum volume in cubic feet to qualify for the discount -/
def discount_threshold : ℚ := 100

/-- The total cost of topsoil after applying the discount -/
noncomputable def total_cost : ℚ :=
  let volume_cubic_feet := volume_cubic_yards * cubic_yards_to_feet
  let base_cost := volume_cubic_feet * cost_per_cubic_foot
  let discount := if volume_cubic_feet > discount_threshold then discount_rate else 0
  base_cost * (1 - discount)

theorem topsoil_cost : total_cost = 13608/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_topsoil_cost_l1147_114797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_floors_l1147_114735

/-- Represents the number of floors in a building -/
structure Building where
  floors : ℕ

/-- Represents the current floor of an elevator -/
structure Floor where
  level : ℕ

/-- The starting floor of the elevator -/
def start_floor : Floor := ⟨9⟩

/-- The number of floors the elevator goes down -/
def down_floors : ℕ := 7

/-- The number of floors the elevator goes up first -/
def up_floors_1 : ℕ := 3

/-- The number of floors the elevator goes up second -/
def up_floors_2 : ℕ := 8

/-- Function to move the elevator -/
def move_elevator (current : Floor) (delta : Int) : Floor :=
  ⟨(max 1 (current.level + delta)).toNat⟩

/-- Theorem stating the number of floors in the building -/
theorem building_floors (b : Building) : 
  let f1 := move_elevator start_floor (-down_floors)
  let f2 := move_elevator f1 up_floors_1
  let f3 := move_elevator f2 up_floors_2
  f3.level = b.floors → b.floors = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_floors_l1147_114735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_fraction_l1147_114733

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 4)) ↔ x ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meaningful_fraction_l1147_114733
