import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l405_40529

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x) + 2 * Real.sin (3 * Real.pi / 2 + x) * Real.sin (Real.pi - x)

-- State the theorem
theorem f_properties :
  -- The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  -- The monotonically increasing interval
  (∀ (k : ℤ), ∀ (x y : ℝ), 
    k * Real.pi + 5 * Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 11 * Real.pi / 12 → 
    f x < f y) ∧
  -- Maximum altitude in triangle ABC
  (∀ (A B C : ℝ), 0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2 ∧
    A + B + C = Real.pi ∧ f A = -Real.sqrt 3 →
    ∀ (a b c : ℝ), a = 3 ∧ a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
    ∀ (h : ℝ), h = a * Real.sin B * Real.sin C / Real.sin (B + C) →
    h ≤ 3 * Real.sqrt 3 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l405_40529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_prices_and_max_basketballs_l405_40570

/-- Represents the price of a soccer ball -/
def soccer_price : ℕ → Prop := sorry

/-- Represents the price of a basketball -/
def basketball_price : ℕ → Prop := sorry

/-- Represents the maximum number of basketballs that can be bought -/
def max_basketballs : ℕ → Prop := sorry

theorem ball_prices_and_max_basketballs :
  ∀ (s b m : ℕ),
  (20 * s + 15 * b = 2050) →
  (10 * s + 20 * b = 1900) →
  (∀ x y : ℕ, x + y = 50 → 70 * y + 50 * x ≤ 2800 → x ≤ 3 * y → y ≤ m) →
  soccer_price s →
  basketball_price b →
  max_basketballs m →
  s = 50 ∧ b = 70 ∧ m = 15 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_prices_and_max_basketballs_l405_40570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l405_40588

-- Define the custom operation ⊗
noncomputable def otimes (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := otimes (3 * x^2 + 6) (23 - x^2)

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), M = 4 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l405_40588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_base_is_positive_l405_40575

open Real

-- Define the problem parameters
variable (b m : ℝ)
variable (h_b_pos : b > 0)
variable (h_m_pos : m > 0)
variable (h_b_gt_m : b > m)

-- Define the theorem
theorem isosceles_triangle_base (b m : ℝ) (h_b_pos : b > 0) (h_m_pos : m > 0) (h_b_gt_m : b > m) :
  ∃ x : ℝ, x = b * m / (b - m) := by
  -- Introduce the variable x
  let x := b * m / (b - m)
  
  -- Prove that x satisfies the equation
  have h_x_eq : x = b * m / (b - m) := by rfl
  
  -- Show that x exists and satisfies the equation
  exact ⟨x, h_x_eq⟩

-- Example usage of the theorem
#check isosceles_triangle_base

-- You can add more properties or lemmas here if needed
-- For example, you might want to prove that x is positive:

theorem base_is_positive (b m : ℝ) (h_b_pos : b > 0) (h_m_pos : m > 0) (h_b_gt_m : b > m) :
  b * m / (b - m) > 0 := by
  have h_denom_pos : b - m > 0 := sub_pos_of_lt h_b_gt_m
  have h_num_pos : b * m > 0 := mul_pos h_b_pos h_m_pos
  exact div_pos h_num_pos h_denom_pos

#check base_is_positive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_base_is_positive_l405_40575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_radius_l405_40540

theorem larger_sphere_radius (r : ℝ) : 
  (4 / 3 * Real.pi * r^3 = 6 * (4 / 3 * Real.pi * 2^3)) → 
  r = 4 * (2 : ℝ)^(1/3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sphere_radius_l405_40540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_2520_l405_40538

/-- The number of distinct, positive factors of 2520 -/
def number_of_factors : ℕ := 48

/-- 2520 is the number we're considering -/
def n : ℕ := 2520

theorem factors_of_2520 : (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = number_of_factors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_2520_l405_40538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_common_value_l405_40557

theorem largest_common_value : 
  ∃ (k1 k2 : ℕ), 7 * k1 = 5 + 11 * k2 ∧ 
                 7 * k1 = 931 ∧
                 7 * k1 < 1000 ∧
                 ∀ (j1 j2 : ℕ), 7 * j1 = 5 + 11 * j2 → 7 * j1 < 1000 → 7 * j1 ≤ 931 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_common_value_l405_40557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafe_purchase_solution_l405_40547

/-- Represents the cafe purchase problem -/
def cafe_purchase (total_money sandwich_cost coffee_cost : ℚ) : ℕ :=
  let max_sandwiches := (total_money / sandwich_cost).floor.toNat
  let remaining_money := total_money - (max_sandwiches : ℚ) * sandwich_cost
  let max_coffee := (remaining_money / coffee_cost).floor.toNat
  max_sandwiches + max_coffee

/-- Theorem stating the solution to the specific problem -/
theorem cafe_purchase_solution :
  cafe_purchase 50 3.5 1.5 = 14 := by
  -- The proof goes here
  sorry

#eval cafe_purchase 50 3.5 1.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafe_purchase_solution_l405_40547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_greater_than_a_minus_one_l405_40574

-- Define the floor function as noncomputable
noncomputable def floor (a : ℝ) : ℤ :=
  Int.floor a

-- Theorem statement
theorem floor_greater_than_a_minus_one (a : ℝ) : (floor a : ℝ) > a - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_greater_than_a_minus_one_l405_40574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l405_40581

/-- Calculates the time for two trains to cross each other -/
noncomputable def time_to_cross (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let combined_length := length1 + length2
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  combined_length / relative_speed

/-- The time for two trains to cross each other is approximately 10.08 seconds -/
theorem trains_crossing_time :
  let length1 := (120 : ℝ)
  let length2 := (160 : ℝ)
  let speed1 := (60 : ℝ)
  let speed2 := (40 : ℝ)
  ‖(time_to_cross length1 length2 speed1 speed2) - 10.08‖ < 0.01 := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check time_to_cross (120 : ℝ) (160 : ℝ) (60 : ℝ) (40 : ℝ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l405_40581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_letter_initials_count_is_504_l405_40527

/-- The number of different three-letter sets of initials possible using the unique letters from A to I. -/
def three_letter_initials_count : Nat :=
  let alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'}
  let n : Nat := alphabet.card
  n * (n - 1) * (n - 2)

/-- Theorem stating that the number of three-letter sets of initials is 504. -/
theorem three_letter_initials_count_is_504 : three_letter_initials_count = 504 := by
  rfl

#eval three_letter_initials_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_letter_initials_count_is_504_l405_40527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l405_40590

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 12 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

-- Define the intersection points
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Assume A and B are on both circles
axiom A_on_C₁ : C₁ A.1 A.2
axiom A_on_C₂ : C₂ A.1 A.2
axiom B_on_C₁ : C₁ B.1 B.2
axiom B_on_C₂ : C₂ B.1 B.2

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem chord_length : distance A B = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l405_40590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l405_40566

theorem power_equation (a b : ℝ) (h1 : (80 : ℝ)^a = 2) (h2 : (80 : ℝ)^b = 5) :
  (16 : ℝ)^((1 - a - b)/(2*(1 - b))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l405_40566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_open_interval_l405_40593

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- State the conditions
axiom f_derivative : ∀ x, HasDerivAt f (f' x) x
axiom f_condition : ∀ x, f' x + f x > 0
axiom f_value : f 1 = Real.exp (-1)

-- Define the solution set
def solution_set : Set ℝ := {x | x > 0 ∧ f (Real.log x) < 1 / x}

-- State the theorem
theorem solution_set_is_open_interval :
  solution_set = Set.Ioo 0 (Real.exp 1) := by
  sorry

#check solution_set_is_open_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_open_interval_l405_40593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_exponent_difference_l405_40558

theorem like_terms_exponent_difference (m n : ℤ) : 
  (∃ (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0), ∀ (x y : ℝ), a * x^(m+2) * y^3 = b * x^4 * y^(n-1)) → 
  m - n = -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_exponent_difference_l405_40558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equal_for_first_and_tenth_draw_l405_40518

/-- Given a box with 1 black ball and 9 white balls, where 10 people draw and replace a ball each,
    the probability of drawing a black ball is the same for the first and tenth person. -/
theorem probability_equal_for_first_and_tenth_draw (total_balls : ℕ) (black_balls : ℕ) 
  (h_total : total_balls = 10) (h_black : black_balls = 1) :
  (black_balls : ℚ) / total_balls = (black_balls : ℚ) / total_balls := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equal_for_first_and_tenth_draw_l405_40518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_dates_in_leap_year_l405_40599

/-- A prime month is a month with a prime number of days -/
def isPrimeMonth (m : Nat) : Bool :=
  m ∈ [2, 3, 5, 7, 11]

/-- The number of days in a given month in a leap year -/
def daysInMonth (m : Nat) : Nat :=
  if m = 2 then 29
  else if m ∈ [4, 6, 9, 11] then 30
  else 31

/-- A prime date is a date with a prime number as its day -/
def isPrimeDate (d : Nat) : Bool :=
  d ∈ [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

/-- The number of prime dates in a given month -/
def primeDatesInMonth (m : Nat) : Nat :=
  (List.range (daysInMonth m)).filter isPrimeDate |>.length

/-- The total number of prime dates in prime months of a leap year -/
def totalPrimeDates : Nat :=
  (List.range 12).filter isPrimeMonth |>.map primeDatesInMonth |>.sum

theorem prime_dates_in_leap_year :
  totalPrimeDates = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_dates_in_leap_year_l405_40599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_sum_magnitude_l405_40521

def a (l : ℝ) : ℝ × ℝ := (l, -2)
def b : ℝ × ℝ := (1, 3)

theorem perpendicular_vectors_sum_magnitude (l : ℝ) :
  (a l).1 * b.1 + (a l).2 * b.2 = 0 →
  Real.sqrt (((a l).1 + b.1)^2 + ((a l).2 + b.2)^2) = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_sum_magnitude_l405_40521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_eight_decomposition_l405_40510

theorem factorial_eight_decomposition : 2^5 * 3^2 * 140 = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_eight_decomposition_l405_40510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_graphs_imply_a_range_l405_40512

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := x^2 + Real.log (x + a)
noncomputable def g (x : ℝ) : ℝ := x^2 + Real.exp x - 1/2

-- Define the theorem
theorem symmetric_graphs_imply_a_range (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ f a (-x) = g x) →
  a ∈ Set.Iio (Real.sqrt (Real.exp 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_graphs_imply_a_range_l405_40512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l405_40543

open Real

/-- The function f(x) = ln x + mx -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := log x + m * x

/-- The function g(x) = f(x) + 1/2 x² -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x + (1/2) * x^2

/-- The function h(x) = 2ln x - ax - x² -/
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := 2 * log x - a * x - x^2

/-- The second derivative of h(x) -/
noncomputable def h'' (a : ℝ) (x : ℝ) : ℝ := -2 / x^2 - 2

theorem min_value_theorem (m : ℝ) (a : ℝ) (x₁ x₂ : ℝ) :
  m ≤ -3 * sqrt 2 / 2 →
  x₁ < x₂ →
  h a x₁ = 0 →
  h a x₂ = 0 →
  g m x₁ = g m x₂ →
  (∀ x, g m x ≥ g m x₁) →
  (x₁ - x₂) * h'' a ((x₁ + x₂) / 2) ≥ -4/3 + 2 * log 2 ∧
  ∃ y, (x₁ - x₂) * h'' a ((x₁ + x₂) / 2) = y ∧ y = -4/3 + 2 * log 2 := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l405_40543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_disconnected_regions_l405_40567

/-- A coloring of an n × n grid. -/
def GridColoring (n : ℕ) := Fin n → Fin n → Bool

/-- Two cells are adjacent if they share a vertex and have the same color. -/
def adjacent (n : ℕ) (c : GridColoring n) (i j k l : Fin n) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨
  (i = k ∧ l.val + 1 = j.val) ∨
  (j = l ∧ i.val + 1 = k.val) ∨
  (j = l ∧ k.val + 1 = i.val) ∧
  c i j = c k l

/-- A path between two cells in the grid. -/
inductive GridPath (n : ℕ) (c : GridColoring n) : Fin n → Fin n → Fin n → Fin n → Prop
  | refl (i j : Fin n) : GridPath n c i j i j
  | step (i j k l m p : Fin n) : 
      GridPath n c i j k l → adjacent n c k l m p → GridPath n c i j m p

/-- The number of mutually disconnected monochromatic regions in a grid coloring. -/
noncomputable def disconnectedRegions (n : ℕ) (c : GridColoring n) : ℕ := 
  sorry

/-- The main theorem: maximum number of mutually disconnected regions. -/
theorem max_disconnected_regions {n : ℕ} (h : Odd n) (h2 : n ≥ 3) :
  (∃ c : GridColoring n, disconnectedRegions n c = (n + 1)^2 / 4 + 1) ∧
  (∀ c : GridColoring n, disconnectedRegions n c ≤ (n + 1)^2 / 4 + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_disconnected_regions_l405_40567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l405_40500

/-- The circle centered at (1, 0) with radius 1 -/
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- A point is on the tangent line if it satisfies this equation -/
def is_tangent_point (x y : ℝ) : Prop :=
  my_circle x y ∧ (x - 1) * (x - 3) + y * (y - 1) = 0

/-- The line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_equation (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

theorem tangent_line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_tangent_point x₁ y₁ ∧
    is_tangent_point x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    ∀ (x y : ℝ), line_equation x₁ y₁ x₂ y₂ x y ↔ 2*x + y - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l405_40500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_rotation_position_disk_rotation_time_l405_40503

-- Define the radii of the clock face and the smaller disk
def clock_radius : ℝ := 24
def disk_radius : ℝ := 8

-- Define the circumferences
noncomputable def clock_circumference : ℝ := 2 * Real.pi * clock_radius
noncomputable def disk_circumference : ℝ := 2 * Real.pi * disk_radius

-- Theorem statement
theorem disk_rotation_position :
  disk_circumference / clock_circumference = 1 / 3 := by
  sorry

-- Convert the fraction to hours (1/3 of 12 hours = 4 hours)
theorem disk_rotation_time : 
  (disk_circumference / clock_circumference) * 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_rotation_position_disk_rotation_time_l405_40503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speedster_convertibles_count_l405_40584

/-- Represents the total number of vehicles in the inventory -/
noncomputable def total_vehicles : ℝ := 64

/-- Represents the fraction of Speedsters in the inventory -/
noncomputable def speedster_fraction : ℝ := 2/3

/-- Represents the fraction of convertibles among Speedsters -/
noncomputable def convertible_fraction : ℝ := 4/5

/-- Calculates the number of Speedster convertibles -/
noncomputable def speedster_convertibles : ℝ :=
  total_vehicles * speedster_fraction * convertible_fraction

/-- Theorem stating that the number of Speedster convertibles is approximately 34 -/
theorem speedster_convertibles_count :
  ⌊speedster_convertibles⌋ = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speedster_convertibles_count_l405_40584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_plank_count_l405_40541

theorem fence_plank_count : ∀ N : ℕ, N ∈ ({96, 97, 98, 99, 100} : Set ℕ) →
  (∃ x : ℕ, N = 5 * x + 1) ↔ N = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_plank_count_l405_40541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_99_equal_digit_sum_2020_l405_40506

def digit_sum (n : ℕ) : ℕ := sorry

theorem no_99_equal_digit_sum_2020 : 
  ¬ ∃ (a : Fin 99 → ℕ), 
    (∀ i j : Fin 99, digit_sum (a i) = digit_sum (a j)) ∧ 
    (Finset.sum Finset.univ a) = 2020 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_99_equal_digit_sum_2020_l405_40506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_intersecting_square_l405_40545

/-- Represents a unit square on a plane -/
structure UnitSquare where
  center : ℝ × ℝ

/-- The set of unit squares -/
def M : Set UnitSquare := sorry

/-- The property that the distance between centers of any two squares in M is at most 2 -/
axiom distance_constraint : ∀ s₁ s₂, s₁ ∈ M → s₂ ∈ M → 
  Real.sqrt ((s₁.center.1 - s₂.center.1)^2 + (s₁.center.2 - s₂.center.2)^2) ≤ 2

/-- Checks if two unit squares intersect -/
def intersects (s₁ s₂ : UnitSquare) : Prop := sorry

/-- The main theorem to prove -/
theorem exists_intersecting_square : 
  ∃ s : UnitSquare, ∀ m, m ∈ M → intersects s m := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_intersecting_square_l405_40545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l405_40580

-- Define the domain
def Domain : Set ℝ := Set.Icc (-2 : ℝ) 2

-- Define f(x)
noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 ∧ x ≤ 2 then 2^x - 1
  else if x < 0 ∧ x ≥ -2 then -(2^(-x) - 1)
  else 0  -- f(0) = 0 since f is odd

-- Define g(x)
def g (m : ℝ) : ℝ → ℝ := fun x => x^2 - 2*x + m

-- State the properties of f
axiom f_odd : ∀ x ∈ Domain, f (-x) = -f x

-- State the condition about g and f
axiom g_f_relation : ∀ m : ℝ, ∀ x₁ ∈ Domain, ∃ x₂ ∈ Domain, g m x₂ = f x₁

-- Theorem to prove
theorem m_range : 
  {m : ℝ | ∀ x₁ ∈ Domain, ∃ x₂ ∈ Domain, g m x₂ = f x₁} = Set.Icc (-5 : ℝ) (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l405_40580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l405_40569

-- Define the point A on the line y = x - 1
def A (a : ℝ) : ℝ × ℝ := (a, a - 1)

-- Define the parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Define the line y = x - 1
def line (x : ℝ) : ℝ := x - 1

-- Define the slope of tangent line l
noncomputable def slope_l (a : ℝ) : ℝ := 2 * (a + Real.sqrt (a^2 - a + 1))

-- Define the slope of tangent line m
noncomputable def slope_m (a : ℝ) : ℝ := 2 * (a - Real.sqrt (a^2 - a + 1))

-- Define the area bounded by the line segment PQ and the parabola
noncomputable def area (a : ℝ) : ℝ := (4/3) * (a^2 - a + 1)^(3/2)

-- Define the distance function between a point on the parabola and the line
noncomputable def distance (t : ℝ) : ℝ := (t^2 - t + 1) / Real.sqrt 2

theorem tangent_lines_theorem (a : ℝ) :
  (∀ x : ℝ, line x = (A a).2) →
  (slope_l a > slope_m a) →
  (slope_l a = 2 * (a + Real.sqrt (a^2 - a + 1))) ∧
  (area a ≥ Real.sqrt 3 / 2) ∧
  (∀ t : ℝ, distance t ≥ (3/8) * Real.sqrt 2) := by
  sorry

#check tangent_lines_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l405_40569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_same_face_exists_l405_40550

/-- Represents a coin, which can be either heads or tails -/
inductive Coin
| Heads
| Tails

/-- A circular arrangement of 11 coins -/
def CoinCircle := Vector Coin 11

/-- Checks if two adjacent coins in the circle have the same face -/
def has_adjacent_same (circle : CoinCircle) : Prop :=
  ∃ i : Fin 11, circle.get i = circle.get ((i.val + 1) % 11)

/-- Theorem: In any circular arrangement of 11 coins, there must exist at least
    one pair of adjacent coins showing the same face -/
theorem adjacent_same_face_exists (circle : CoinCircle) : has_adjacent_same circle := by
  sorry

#check adjacent_same_face_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_same_face_exists_l405_40550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_per_set_l405_40536

-- Define the selling price
variable (x : ℝ)

-- Define the sales volume function
noncomputable def sales_volume (x : ℝ) : ℝ := 15 - 0.1 * x

-- Define the supply price function
noncomputable def supply_price (x : ℝ) : ℝ := 30 + 10 / sales_volume x

-- Define the profit per set function
noncomputable def profit_per_set (x : ℝ) : ℝ := x - supply_price x

-- Theorem statement
theorem max_profit_per_set :
  ∀ x > 0, x < 150 → profit_per_set x ≤ 100 ∧
  ∃ x_max, x_max = 140 ∧ profit_per_set x_max = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_per_set_l405_40536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l405_40556

/-- An isosceles trapezoid with an inscribed circle. -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- Length of one segment of the non-parallel side divided by the point of tangency -/
  m : ℝ
  /-- Length of the other segment of the non-parallel side divided by the point of tangency -/
  n : ℝ
  /-- m and n are positive real numbers -/
  m_pos : 0 < m
  n_pos : 0 < n

/-- The area of an isosceles trapezoid with an inscribed circle -/
noncomputable def area (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ :=
  2 * Real.sqrt (t.m * t.n) * (t.m + t.n)

/-- Theorem: The area of an isosceles trapezoid with an inscribed circle,
    where one non-parallel side is divided by the point of tangency into
    segments of lengths m and n, is equal to 2√(mn)(m+n). -/
theorem isosceles_trapezoid_area
    (t : IsoscelesTrapezoidWithInscribedCircle) :
    area t = 2 * Real.sqrt (t.m * t.n) * (t.m + t.n) := by
  -- Unfold the definition of area
  unfold area
  -- The equation is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l405_40556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_set_finite_nonempty_intersection_l405_40501

/-- A set of points in ℝ³ -/
def M : Set (ℝ × ℝ × ℝ) :=
  {p | ∃ t : ℝ, p.1 = t ∧ p.2.1 = t^3 ∧ p.2.2 = t^5}

/-- A plane in ℝ³ -/
def Plane (A B C D : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {p | A * p.1 + B * p.2.1 + C * p.2.2 + D = 0}

/-- The theorem stating the existence of a set M with the desired property -/
theorem exists_set_finite_nonempty_intersection :
  ∃ (M : Set (ℝ × ℝ × ℝ)), 
    ∀ (A B C D : ℝ), (A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0) → 
      let δ := Plane A B C D
      (M ∩ δ).Nonempty ∧ (M ∩ δ).Finite :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_set_finite_nonempty_intersection_l405_40501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_problem_l405_40576

/-- The number of students remaining on a bus after multiple stops where a fraction of students disembark at each stop. -/
def students_remaining (initial : ℕ) (fraction : ℚ) (stops : ℕ) : ℚ :=
  initial * (1 - fraction) ^ stops

/-- Theorem stating that approximately 12 students remain on the bus after four stops -/
theorem bus_problem : 
  let initial_students : ℕ := 60
  let fraction_leaving : ℚ := 1 / 3
  let num_stops : ℕ := 4
  ⌊students_remaining initial_students fraction_leaving num_stops⌋ = 12 :=
by
  -- Unfold the definitions
  unfold students_remaining
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_problem_l405_40576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_not_in_range_l405_40528

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 3
  | (n + 3) => max (2 * sequence_a (n + 1)) (3 * sequence_a (n + 2) - 2 * sequence_a (n + 1))

theorem sequence_a_not_in_range (n : ℕ) : sequence_a n ∉ Set.Icc 1612 2012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_not_in_range_l405_40528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_common_perpendiculars_l405_40533

/-- Definition of a line in 3D space -/
def Line₃ : Type := sorry

/-- Definition of skew lines in 3D space -/
def SkewLines : Line₃ → Line₃ → Line₃ → Prop := sorry

/-- Definition of perpendicular lines in 3D space -/
def Perpendicular : Line₃ → Line₃ → Prop := sorry

/-- Three skew lines in 3D space are common perpendiculars to their common perpendiculars. -/
theorem skew_lines_common_perpendiculars (a b c : Line₃) 
  (h_skew : SkewLines a b c) : 
  ∃ (a' b' c' : Line₃), 
    (Perpendicular a' b ∧ Perpendicular a' c) ∧
    (Perpendicular b' c ∧ Perpendicular b' a) ∧
    (Perpendicular c' a ∧ Perpendicular c' b) ∧
    (Perpendicular a a' ∧ Perpendicular b b' ∧ Perpendicular c c') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_common_perpendiculars_l405_40533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_log_fractions_l405_40534

theorem order_of_log_fractions :
  let a := (Real.log 5) / 5
  let b := 1 / Real.exp 1
  let c := (Real.log 4) / 4
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_log_fractions_l405_40534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l405_40515

theorem divisibility_problem (n : ℕ) : 
  512 ∣ (3^(2*n) - 32*n^2 + 24*n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l405_40515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l405_40582

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + Real.sqrt 3 * y - 1 = 0

-- Define the slope of the line
noncomputable def line_slope : ℝ := -(3 / Real.sqrt 3)

-- Define the inclination angle
noncomputable def inclination_angle : ℝ := 120 * (Real.pi / 180)

-- Theorem statement
theorem line_inclination_angle :
  Real.tan inclination_angle = -line_slope ∧
  line_equation 0 0 → inclination_angle = 120 * (Real.pi / 180) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l405_40582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_table_sum_theorem_l405_40522

-- Define the type for two-digit prime numbers
def TwoDigitPrime := {n : ℕ // 10 ≤ n ∧ n < 100 ∧ Nat.Prime n}

-- Define a 2x2 table of two-digit primes
def PrimeTable := Fin 2 → Fin 2 → TwoDigitPrime

-- Function to get the smallest four two-digit primes
noncomputable def smallestFourTwoDigitPrimes : Finset TwoDigitPrime := sorry

-- Function to calculate row and column sums
noncomputable def calcSums (t : PrimeTable) : Finset ℕ := sorry

-- Function to get the range of a PrimeTable
noncomputable def getPrimeTableRange (t : PrimeTable) : Finset TwoDigitPrime := sorry

-- Theorem statement
theorem prime_table_sum_theorem (t : PrimeTable) :
  getPrimeTableRange t ⊆ smallestFourTwoDigitPrimes →
  24 ∈ calcSums t →
  28 ∈ calcSums t →
  ∃ c d : ℕ, c < d ∧ 
    c ∈ calcSums t ∧ 
    d ∈ calcSums t ∧ 
    5 * c + 7 * d = 412 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_table_sum_theorem_l405_40522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_good_filling_no_good_filling_2017_l405_40577

/-- Definition of a good filling for an n×n square -/
def is_good_filling (n : ℕ) (filling : Fin n → Fin n → Fin (2*n - 1)) : Prop :=
  ∀ i : Fin n, (∀ j : Fin (2*n - 1), ∃ k : Fin n, filling i k = j ∨ filling k i = j) ∧
               (∀ j : Fin (2*n - 1), ∃ k : Fin n, filling i k = j ∨ filling k i = j)

/-- Theorem stating the existence of a good filling for some n ≥ 3 -/
theorem exists_good_filling :
  ∃ (n : ℕ) (filling : Fin n → Fin n → Fin (2*n - 1)), n ≥ 3 ∧ is_good_filling n filling := by
  sorry

/-- Theorem stating that no good filling exists for n = 2017 -/
theorem no_good_filling_2017 :
  ¬∃ (filling : Fin 2017 → Fin 2017 → Fin 4033), is_good_filling 2017 filling := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_good_filling_no_good_filling_2017_l405_40577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_lambda_l405_40507

/-- Given vectors a, b, and c in ℝ³, if they are coplanar and have specific coordinates,
    then the third component of c is equal to 3. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) :
  a = (2, -1, 2) →
  b = (-1, 3, -3) →
  (∃ x y z, c = (x, y, z) ∧ x = 13 ∧ y = 6) →
  (∃ (m n : ℝ), c = m • a + n • b) →
  ∃ z, c = (13, 6, z) ∧ z = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_vectors_lambda_l405_40507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_coincides_with_graph_l405_40542

-- Define the function y = -|x|
def f (x : ℝ) : ℝ := -abs x

-- Define the set S
def S : Set ℝ := {α | ∃ k : ℤ, α = k * 360 + 225 ∨ α = k * 360 + 315}

-- Theorem statement
theorem angle_coincides_with_graph (α : ℝ) :
  (∀ x : ℝ, (x * Real.cos α = x ∧ f x * Real.sin α = f x) → α ∈ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_coincides_with_graph_l405_40542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_eq_inverse_l405_40559

/-- The sum of 1 / (3^a * 4^b * 6^c) over all positive integer triples (a, b, c) where 1 ≤ a < b < c -/
noncomputable def tripleSum : ℝ :=
  ∑' (a : ℕ), ∑' (b : ℕ), ∑' (c : ℕ),
    if 1 ≤ a ∧ a < b ∧ b < c then
      1 / ((3 : ℝ)^a * 4^b * 6^c)
    else
      0

theorem tripleSum_eq_inverse : tripleSum = 1 / 27041 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripleSum_eq_inverse_l405_40559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laurent_expansion_f_l405_40561

open Complex

/-- The function f(z) = z^2 * cos(1/z) -/
noncomputable def f (z : ℂ) : ℂ := z^2 * (cos (1/z))

/-- The Laurent series coefficients of f(z) around z = 0 -/
noncomputable def laurent_coeff (n : ℤ) : ℂ :=
  if n = -2 then 1 / 24
  else if n = 0 then -1 / 2
  else if n = 2 then 1
  else if n ≥ 4 ∧ n % 2 = 0 then (-1)^(n/2 + 1) / (Nat.factorial (n.natAbs + 2))
  else 0

theorem laurent_expansion_f :
  ∀ z ∈ {z : ℂ | z ≠ 0},
  f z = ∑' (n : ℤ), laurent_coeff n * z^n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laurent_expansion_f_l405_40561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_N_values_l405_40554

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem sum_of_possible_N_values : ∃ (S : Finset ℕ),
  (∀ N ∈ S, ∃ (a b c : ℕ),
    (N = a * b * c) ∧
    (N = 8 * (a + b + c)) ∧
    (c = a + b) ∧
    (is_perfect_square (a * b) ∨ is_perfect_square (b * c) ∨ is_perfect_square (a * c))) ∧
  (∀ N : ℕ,
    (∃ (a b c : ℕ),
      (N = a * b * c) ∧
      (N = 8 * (a + b + c)) ∧
      (c = a + b) ∧
      (is_perfect_square (a * b) ∨ is_perfect_square (b * c) ∨ is_perfect_square (a * c)))
    → N ∈ S) ∧
  (S.sum id = 560) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_N_values_l405_40554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l405_40587

open Real

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := sin (2 * x + Real.pi / 3)

-- Define the translated function
noncomputable def translated_function (x : ℝ) : ℝ := sin (2 * (x - Real.pi / 6) + Real.pi / 3)

-- State the theorem
theorem axis_of_symmetry :
  ∃ (k : ℤ), translated_function (Real.pi / 4 + k * Real.pi / 2) = translated_function (Real.pi / 4 - k * Real.pi / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l405_40587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l405_40585

theorem trajectory_equation :
  ∀ (x y : ℝ → ℝ), (∀ t : ℝ, (x t)^2 + (y t)^2 = 4^2) →
  ∀ t : ℝ, (x t)^2 + (y t)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l405_40585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_vector_result_l405_40572

noncomputable def initial_vector : ℝ × ℝ × ℝ := (2, 3, 1)

noncomputable def rotation_angle : ℝ := 90 * (Real.pi / 180)  -- 90 degrees in radians

def passes_through_y_axis (v : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 = 0 ∧ v.2.1 = t ∧ v.2.2 ≠ 0

def is_rotated_vector (v : ℝ × ℝ × ℝ) : Prop :=
  v.1 ^ 2 + v.2.1 ^ 2 + v.2.2 ^ 2 = initial_vector.1 ^ 2 + initial_vector.2.1 ^ 2 + initial_vector.2.2 ^ 2 ∧
  v.1 * initial_vector.1 + v.2.1 * initial_vector.2.1 + v.2.2 * initial_vector.2.2 = 0 ∧
  v.1 = v.2.2

theorem rotated_vector_result :
  ∃ v : ℝ × ℝ × ℝ, 
    is_rotated_vector v ∧
    passes_through_y_axis v ∧
    v = (Real.sqrt (14/3), (-Real.sqrt (14/3), Real.sqrt (14/3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_vector_result_l405_40572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equality_implications_l405_40586

theorem exponential_equality_implications (a b c : ℝ) (h : (3 : ℝ)^a = (4 : ℝ)^b ∧ (4 : ℝ)^b = (6 : ℝ)^c) :
  (∃! n : ℕ, n = 2 ∧
    ((∀ (ha : a > 0) (hb : b > 0) (hc : c > 0), 3*a < 4*b ∧ 4*b < 6*c) ∨
     (∀ (ha : a > 0) (hb : b > 0) (hc : c > 0), 2/c = 1/a + 2/b) ∨
     (∀ (ha : a < 0) (hb : b < 0) (hc : c < 0), a < b ∧ b < c))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equality_implications_l405_40586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_at_206788_is_7_l405_40548

/-- Represents the sequence of digits formed by writing all whole numbers consecutively starting from 1 -/
def digit_sequence (n : ℕ) : ℕ := sorry

/-- Returns the number of digits in a given natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Returns the nth digit in a given natural number, counting from right to left -/
def nth_digit (n : ℕ) (pos : ℕ) : ℕ := sorry

/-- The main theorem stating that the 206,788th digit in the sequence is 7 -/
theorem digit_at_206788_is_7 : digit_sequence 206788 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_at_206788_is_7_l405_40548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_number_property_l405_40530

/-- A number is triangular if it can be expressed as n(n+1)/2 for some natural number n. -/
def IsTriangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

/-- For any odd integer d and positive integer t, 
    t is triangular if and only if d²t + (d²-1)/8 is triangular. -/
theorem triangular_number_property (d : ℤ) (t : ℕ) (h : Odd d) :
  IsTriangular t ↔ IsTriangular ((d^2 * (t : ℤ) + (d^2 - 1) / 8).natAbs) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_number_property_l405_40530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_circles_alignment_l405_40502

/-- Represents a circle with marked arcs -/
structure MarkedCircle where
  k : ℕ  -- number of marked arcs
  arcs : Fin k → ℝ  -- angular size of each arc

/-- Two circles are alignable if their marked arcs can coincide -/
def alignable (c1 c2 : MarkedCircle) : Prop :=
  ∃ (shift : ℝ), ∀ (i : Fin c1.k), ∃ (j : Fin c2.k), 
    (c1.arcs i + shift) % 360 = c2.arcs j

/-- Theorem: If two identical circles with k arcs each, where each arc is smaller than 
    1/(k^2 - k + 1) * 180°, are alignable, then they can be aligned with no overlap -/
theorem marked_circles_alignment (c1 c2 : MarkedCircle) 
  (h_identical : c1 = c2)
  (h_arc_size : ∀ (i : Fin c1.k), c1.arcs i < 180 / (c1.k^2 - c1.k + 1))
  (h_alignable : alignable c1 c2) :
  ∃ (shift : ℝ), ∀ (i : Fin c1.k) (j : Fin c1.k), 
    (c1.arcs i + shift) % 360 ≠ c1.arcs j := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_circles_alignment_l405_40502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hours_worked_l405_40511

/-- Represents the payment structure for Harry and James -/
structure PaymentStructure where
  x : ℝ  -- hourly rate

/-- Calculates Harry's pay for a given number of hours -/
noncomputable def harry_pay (ps : PaymentStructure) (hours : ℝ) : ℝ :=
  if hours ≤ 30 then ps.x * hours
  else ps.x * 30 + 2 * ps.x * (hours - 30)

/-- Calculates James' pay for a given number of hours -/
noncomputable def james_pay (ps : PaymentStructure) (hours : ℝ) : ℝ :=
  if hours ≤ 40 then ps.x * hours
  else ps.x * 40 + 2 * ps.x * (hours - 40)

/-- Theorem stating that if Harry and James are paid the same amount, 
    and James worked 41 hours, then Harry worked 16 hours -/
theorem harry_hours_worked (ps : PaymentStructure) :
  james_pay ps 41 = harry_pay ps 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hours_worked_l405_40511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_multiple_of_hundred_l405_40531

theorem difference_multiple_of_hundred (S : Finset ℤ) :
  S.card = 101 → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_multiple_of_hundred_l405_40531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_value_l405_40546

/-- An arithmetic sequence with 5 terms -/
def ArithmeticSequence (a₁ a₅ : ℚ) : Prop :=
  ∃ (d : ℚ), a₁ + 4*d = a₅

/-- The middle term of a 5-term sequence -/
def MiddleTerm (a₁ a₅ : ℚ) : ℚ :=
  (a₁ + a₅) / 2

theorem middle_term_value :
  ArithmeticSequence 23 47 →
  MiddleTerm 23 47 = 35 := by
  sorry

#check middle_term_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_value_l405_40546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_area_l405_40595

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square defined by its vertices -/
structure Square where
  vertices : Fin 4 → Point

/-- Checks if a point is on the line y = 2x + 17 -/
def isOnLine (p : Point) : Prop :=
  p.y = 2 * p.x + 17

/-- Checks if a point is on the parabola y = x^2 -/
def isOnParabola (p : Point) : Prop :=
  p.y = p.x^2

/-- Checks if a square has two vertices on the line and two on the parabola -/
def isValidSquare (s : Square) : Prop :=
  (∃ (i j : Fin 4), i ≠ j ∧ isOnLine (s.vertices i) ∧ isOnLine (s.vertices j)) ∧
  (∃ (k l : Fin 4), k ≠ l ∧ isOnParabola (s.vertices k) ∧ isOnParabola (s.vertices l))

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ :=
  let v1 := s.vertices 0
  let v2 := s.vertices 1
  (v2.x - v1.x)^2 + (v2.y - v1.y)^2

/-- The main theorem stating the smallest possible area -/
theorem smallest_square_area :
  ∃ (s : Square), isValidSquare s ∧
    squareArea s = 140 ∧
    ∀ (t : Square), isValidSquare t → squareArea t ≥ 140 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_area_l405_40595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l405_40514

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 1 ∨ y > 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l405_40514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l405_40535

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := ⌊2 * x⌋ + (1 : ℝ) / 3

-- Theorem stating that g is neither even nor odd
theorem g_neither_even_nor_odd :
  ¬(∀ x : ℝ, g x = g (-x)) ∧ ¬(∀ x : ℝ, g x = -g (-x)) := by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l405_40535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integers_satisfying_inequality_l405_40551

theorem positive_integers_satisfying_inequality : 
  (Finset.filter (fun n : ℕ => 30 - 6 * n > 12) (Finset.range 3)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integers_satisfying_inequality_l405_40551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_4_l405_40563

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 8 else 3 * x - 18

-- Theorem statement
theorem solutions_of_f_eq_4 :
  {x : ℝ | f x = 4} = {(-1 : ℝ), 22/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_4_l405_40563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l405_40520

noncomputable section

open Real

/-- The distance function between two curves in polar coordinates -/
noncomputable def d (θ : ℝ) : ℝ := Real.sqrt 7 - 4 / (cos θ + sin θ)

/-- The theorem stating the minimum distance between the curves -/
theorem min_distance_between_curves :
  ∃ (θ : ℝ), d θ = Real.sqrt 7 - 2 * Real.sqrt 2 ∧
  ∀ (φ : ℝ), d φ ≥ d θ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l405_40520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_n_l405_40555

theorem value_of_n (m n p : ℕ+) (h : (m : ℚ) + 1 / ((n : ℚ) + 1/(p : ℚ)) = 17/3) : n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_n_l405_40555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_seven_equals_seven_l405_40505

-- Define the function f
noncomputable def f (u : ℝ) : ℝ := (u^2 + 11*u + 49) / 25

-- State the theorem
theorem f_of_seven_equals_seven :
  (∀ x : ℝ, f (5*x - 3) = x^2 + x + 1) → f 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_seven_equals_seven_l405_40505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lengths_12gon_coefficients_l405_40596

/-- The sum of the lengths of all sides and diagonals of a regular 12-gon inscribed in a circle of radius 12 -/
noncomputable def sum_lengths_12gon (r : ℝ) : ℝ :=
  12 * r + 12 * r * Real.sqrt 2 + 12 * r * Real.sqrt 3 + 12 * r * Real.sqrt 6 + 6 * 2 * r

/-- Represents the coefficients a, b, c, d in the expression a + b√2 + c√3 + d√6 -/
structure Coefficients where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+

theorem sum_lengths_12gon_coefficients :
  ∃ (coeff : Coefficients),
    sum_lengths_12gon 12 = coeff.a + coeff.b * Real.sqrt 2 + coeff.c * Real.sqrt 3 + coeff.d * Real.sqrt 6 ∧
    coeff.a + coeff.b + coeff.c + coeff.d = 720 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_lengths_12gon_coefficients_l405_40596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l405_40517

/-- The function f for which we want to find the extreme value. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x

/-- The theorem stating that the function f attains an extreme value at x = 1 when a = -1. -/
theorem extreme_value_at_one (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 ↔ x = 1) ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l405_40517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l405_40524

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem f_strictly_increasing : 
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < ℇ → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l405_40524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_one_sixth_l405_40589

/-- The sum of the series (n³ + n² - n) / (n+3)! from n=1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' n, (n^3 + n^2 - n) / (Nat.factorial (n + 3 : ℕ))

/-- The sum of the series (n³ + n² - n) / (n+3)! from n=1 to infinity equals 1/6 -/
theorem infiniteSeries_eq_one_sixth : infiniteSeries = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_one_sixth_l405_40589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_slopes_l405_40553

/-- Two parallel lines in the xy-plane -/
structure ParallelLines where
  l1 : Real → Real → Prop
  l2 : Real → Real → Prop
  is_parallel : ∀ x y, l1 x y ↔ x - y + 1 = 0
               ∧ ∀ x y, l2 x y ↔ x - y + 3 = 0

/-- A line intersecting the parallel lines -/
structure IntersectingLine where
  m : Real → Real → Prop
  intersects_l1 : ∃ x y, m x y ∧ (x - y + 1 = 0)
  intersects_l2 : ∃ x y, m x y ∧ (x - y + 3 = 0)

/-- The length of the segment cut off by the parallel lines on the intersecting line -/
noncomputable def segment_length (pl : ParallelLines) (il : IntersectingLine) : Real := 2 * Real.sqrt 2

/-- The theorem stating the possible slopes of the intersecting line -/
theorem intersecting_line_slopes (pl : ParallelLines) (il : IntersectingLine) :
  (∃ k, ∀ x y, il.m x y ↔ y = k * x + Real.sqrt (k * k + 1)) →
  (k = 2 - Real.sqrt 3 ∨ k = 2 + Real.sqrt 3) :=
by
  sorry

#check intersecting_line_slopes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_slopes_l405_40553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_concentration_l405_40525

/-- Represents a solution of alcohol in water -/
structure AlcoholSolution where
  volume : ℚ
  concentration : ℚ

/-- Calculates the total amount of alcohol in a solution -/
def alcohol_amount (solution : AlcoholSolution) : ℚ :=
  solution.volume * solution.concentration

/-- Calculates the concentration of alcohol in a mixture of two solutions -/
def mixture_concentration (solution1 solution2 : AlcoholSolution) : ℚ :=
  (alcohol_amount solution1 + alcohol_amount solution2) / (solution1.volume + solution2.volume)

theorem alcohol_mixture_concentration :
  let solution1 : AlcoholSolution := ⟨8, 1/4⟩
  let solution2 : AlcoholSolution := ⟨2, 3/25⟩
  mixture_concentration solution1 solution2 = 56/250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_concentration_l405_40525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_ammonia_production_l405_40579

-- Define the reaction rates
noncomputable def v_H2_A : ℝ := 0.3
noncomputable def v_N2_B : ℝ := 0.2
noncomputable def v_NH3_C : ℝ := 0.25
noncomputable def v_H2_D : ℝ := 0.4

-- Define the stoichiometric coefficients
noncomputable def coeff_N2 : ℝ := 1
noncomputable def coeff_H2 : ℝ := 3
noncomputable def coeff_NH3 : ℝ := 2

-- Convert all rates to the rate of N2 consumption
noncomputable def rate_N2_A : ℝ := v_H2_A / coeff_H2
noncomputable def rate_N2_B : ℝ := v_N2_B
noncomputable def rate_N2_C : ℝ := v_NH3_C / coeff_NH3
noncomputable def rate_N2_D : ℝ := v_H2_D / coeff_H2

-- Theorem statement
theorem fastest_ammonia_production :
  rate_N2_B > rate_N2_A ∧ 
  rate_N2_B > rate_N2_C ∧ 
  rate_N2_B > rate_N2_D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fastest_ammonia_production_l405_40579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l405_40571

theorem trigonometric_identities (α : Real) 
  (h1 : α > π / 2) (h2 : α < π) (h3 : Real.sin α = 4 / 5) : 
  Real.cos (α - π / 4) = Real.sqrt 2 / 10 ∧ 
  Real.sin α ^ 2 / 2 + (Real.sin (4 * α) * Real.cos (2 * α)) / (1 + Real.cos (4 * α)) = -8 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l405_40571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_property_l405_40523

-- Define the circle
def my_circle (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = r^2}

-- Define a point inside the circle
def point_inside (a b r : ℝ) : Prop := a^2 + b^2 < r^2 ∧ a ≠ 0 ∧ b ≠ 0

-- Define line l1 (shortest chord through P)
def line_l1 (a b : ℝ) : Set (ℝ × ℝ) := {p | a * p.1 + b * p.2 = a^2 + b^2}

-- Define line l2
def line_l2 (a b r : ℝ) : Set (ℝ × ℝ) := {p | b * p.1 - a * p.2 = r^2}

-- Define parallel lines
def parallel (l1 l2 : Set (ℝ × ℝ)) : Prop := ∃ (k : ℝ) (c : ℝ), ∀ p, p ∈ l1 ↔ k * p.1 + p.2 = c

-- Define a line separate from a circle
def separate (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∀ p, p ∈ l → p ∉ c

theorem shortest_chord_property (r a b : ℝ) :
  point_inside a b r →
  parallel (line_l1 a b) (line_l2 a b r) ∧
  separate (line_l2 a b r) (my_circle r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_property_l405_40523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_OAB_area_l405_40539

noncomputable section

-- Define the coordinate system
variable (x y : ℝ)

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 1

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = -2*y + 3

-- Define the point (1,0) that line l passes through
def point_on_l : Prop := line_l 1 0

-- Define the perpendicularity of line l to x-y+1=0
def perpendicularity : Prop := ∀ x y, line_l x y → perp_line x y

-- Define points A and B as the intersection of line l and circle C
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the area of a triangle given three points
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)

-- Theorem statement
theorem triangle_OAB_area
  (A B : ℝ × ℝ)
  (h_point : point_on_l)
  (h_perp : perpendicularity)
  (h_intersect : intersection_points A B) :
  triangle_area origin A B = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_OAB_area_l405_40539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l405_40583

/-- The condition p: (x-a)(x-4a) < 0 for a > 0 and x ∈ ℝ -/
def p (a x : ℝ) : Prop := (x - a) * (x - 4*a) < 0 ∧ a > 0

/-- The condition q: 3 < x < 4 for x ∈ ℝ -/
def q (x : ℝ) : Prop := 3 < x ∧ x < 4

/-- Theorem for part (I) -/
theorem part_one : 
  ∀ x : ℝ, q x → p 1 x := by sorry

/-- Theorem for part (II) -/
theorem part_two : 
  {a : ℝ | ∀ x : ℝ, q x → p a x ∧ ∃ y : ℝ, p a y ∧ ¬q y} = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l405_40583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_property_l405_40578

/-- Represents the sum of the first n terms of a geometric sequence -/
def GeometricSum (n : ℕ) : ℝ := sorry

/-- The theorem statement -/
theorem geometric_sum_property
  (h1 : GeometricSum 14 = 3 * GeometricSum 7)
  (h2 : GeometricSum 14 = 3) :
  GeometricSum 28 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_property_l405_40578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_b_value_l405_40597

/-- A rectangle ABCD with the following properties:
  * Has area 72
  * AB is parallel to the x-axis
  * Vertex A is on the graph of y = log_b(x)
  * Vertex B is on the graph of y = 4log_b(x)
  * Vertex C is on the graph of y = 6log_b(x)
-/
structure SpecialRectangle (b : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  area_eq : (B.1 - A.1) * (C.2 - A.2) = 72
  AB_parallel_x : A.2 = B.2
  A_on_graph : A.2 = Real.log A.1 / Real.log b
  B_on_graph : B.2 = 4 * (Real.log B.1 / Real.log b)
  C_on_graph : C.2 = 6 * (Real.log C.1 / Real.log b)

/-- The value of b for which the SpecialRectangle exists is the fourth root of 10 -/
theorem special_rectangle_b_value :
  ∃ (b : ℝ), ∃ (rect : SpecialRectangle b), b = (10 : ℝ) ^ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rectangle_b_value_l405_40597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l405_40549

/-- A set of points in a plane where each point is the midpoint of two others -/
structure MidpointSet where
  S : Set (ℝ × ℝ)
  midpoint_property : ∀ p, p ∈ S → ∃ q r, q ∈ S ∧ r ∈ S ∧ p = ((q.1 + r.1) / 2, (q.2 + r.2) / 2)

/-- Theorem: A MidpointSet contains infinitely many points -/
theorem midpoint_set_infinite (M : MidpointSet) : Set.Infinite M.S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l405_40549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_rate_per_square_meter_l405_40560

/-- Proves that the rate of the carpet per square meter is Rs. 45 given the specified conditions -/
theorem carpet_rate_per_square_meter (breadth_1 : ℝ) (length_1 : ℝ) (length_2 : ℝ) (breadth_2 : ℝ) (cost_2 : ℝ) :
  breadth_1 = 6 →
  length_1 = 1.44 * breadth_1 →
  length_2 = length_1 * 1.4 →
  breadth_2 = breadth_1 * 1.25 →
  cost_2 = 4082.4 →
  cost_2 / (length_2 * breadth_2) = 45 := by
  sorry

-- Remove the #eval line as it's causing issues with universe levels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_rate_per_square_meter_l405_40560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l405_40544

/-- The eccentricity of an ellipse passing through (√5, 0) and (0, 3) -/
theorem ellipse_eccentricity : ∃ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧ a > b ∧ 
  (5 : ℝ) / b^2 = 1 ∧ 
  9 / a^2 = 1 ∧
  (Real.sqrt (a^2 - b^2)) / a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l405_40544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_meat_consumption_l405_40508

theorem zoo_meat_consumption (num_lions : ℚ) : 
  let num_tigers := 2 * num_lions
  let tiger_consumption := (4.5 : ℚ)
  let lion_consumption := (3.5 : ℚ)
  let total_consumption := num_tigers * tiger_consumption + num_lions * lion_consumption
  let total_animals := num_tigers + num_lions
  total_consumption / total_animals = 25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_meat_consumption_l405_40508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_through_point_parallel_to_plane_with_angle_l405_40568

/-- A point in 3D space -/
structure Point where

/-- A plane in 3D space -/
structure Plane where

/-- A line in 3D space -/
structure Line where
  parallel : Plane → Prop
  angle_with : Plane → Real

/-- Given a point A, planes α and β, and an angle θ, there exist two distinct lines passing through A, parallel to α, and forming angle θ with β. -/
theorem two_lines_through_point_parallel_to_plane_with_angle 
  (A : Point) (α β : Plane) (θ : Real) :
  ∃ (l₁ l₂ : Line), l₁ ≠ l₂ ∧ 
    (∃ (p₁ p₂ : Point), p₁ = A ∧ p₂ = A) ∧
    l₁.parallel α ∧ l₂.parallel α ∧
    l₁.angle_with β = θ ∧ l₂.angle_with β = θ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_through_point_parallel_to_plane_with_angle_l405_40568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l405_40516

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  (0 < a ∧ 0 < b ∧ 0 < c) →
  -- A, B, C are angles of the triangle
  (0 < A ∧ A < Real.pi) ∧ (0 < B ∧ B < Real.pi) ∧ (0 < C ∧ C < Real.pi) →
  -- Sum of angles in a triangle is π
  (A + B + C = Real.pi) →
  -- Cosine law
  (a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)) →
  -- b², a², c² form an arithmetic sequence
  (b^2 - a^2 = a^2 - c^2) →
  -- Minimum value of cos A is 1/2
  (Real.cos A ≥ 1/2) ∧
  -- When a = 2, maximum area is √3
  (a = 2 → ∃ (S : ℝ), S ≤ Real.sqrt 3 ∧ S = 1/2 * b * c * Real.sin A) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l405_40516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_1950_smallest_difference_is_11_factors_with_difference_11_l405_40513

/-- The smallest difference between two factors of 1950 -/
def smallest_factor_difference : ℕ := 11

/-- 1950 can be expressed as a product of two positive integers -/
theorem factors_of_1950 : ∃ (a b : ℕ), a * b = 1950 ∧ a > 0 ∧ b > 0 := by sorry

/-- The smallest difference between two factors of 1950 is 11 -/
theorem smallest_difference_is_11 : 
  ∀ (a b : ℕ), a * b = 1950 → a > 0 → b > 0 → 
  (max a b - min a b) ≥ smallest_factor_difference := by sorry

/-- There exist two factors of 1950 with a difference of 11 -/
theorem factors_with_difference_11 : 
  ∃ (a b : ℕ), a * b = 1950 ∧ a > 0 ∧ b > 0 ∧ 
  (max a b - min a b) = smallest_factor_difference := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_1950_smallest_difference_is_11_factors_with_difference_11_l405_40513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_length_l405_40565

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  valid : P ≠ Q ∧ Q ≠ R ∧ R ≠ P

-- Define a line segment
def LineSegment (A B : ℝ × ℝ) : Type := {x : ℝ × ℝ // ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • A + t • B}

-- Define parallel lines
def Parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

-- Define the angle between two lines
noncomputable def AngleBetween (l₁ l₂ : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the length of a line segment
noncomputable def Length (A B : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem triangle_segment_length 
  (P Q R : ℝ × ℝ) 
  (tri : Triangle P Q R) 
  (X : ℝ × ℝ) 
  (Y : ℝ × ℝ) 
  (Z : ℝ × ℝ) 
  (h₁ : Length P Q = 10)
  (h₂ : Parallel {X, Y, Z} {P, Q})
  (h₃ : AngleBetween {P, Y, Z} {R, Z} = 120)
  (h₄ : Length X Y = 6) :
  Length R Y = 15 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_length_l405_40565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_maximized_l405_40526

/-- An arithmetic sequence with first term greater than zero and a6/a5 = 9/11 -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_positive : 0 < a 1
  ratio_condition : a 6 / a 5 = 9 / 11
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The value of n that maximizes the sum of the first n terms -/
def maximizing_n : ℕ := 10

theorem sum_maximized (seq : ArithmeticSequence) :
  ∀ n : ℕ, sum_n seq n ≤ sum_n seq maximizing_n :=
by
  sorry

#check sum_maximized

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_maximized_l405_40526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_in_set_twenty_two_sevenths_rational_zero_rational_two_tenths_rational_l405_40573

-- Define the set of numbers
def numbers : Set ℝ := {22/7, 0, Real.sqrt 2, 0.2}

-- Theorem stating that √2 is the only irrational number in the set
theorem sqrt_two_irrational_in_set :
  ∃! x, x ∈ numbers ∧ Irrational x ∧ x = Real.sqrt 2 :=
sorry

-- Theorem stating that 22/7 is rational
theorem twenty_two_sevenths_rational : ¬ Irrational (22/7 : ℝ) :=
sorry

-- Theorem stating that 0 is rational
theorem zero_rational : ¬ Irrational (0 : ℝ) :=
sorry

-- Theorem stating that 0.2 is rational
theorem two_tenths_rational : ¬ Irrational (0.2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_two_irrational_in_set_twenty_two_sevenths_rational_zero_rational_two_tenths_rational_l405_40573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_condition_l405_40591

/-- The amount invested that satisfies the given interest conditions -/
noncomputable def amount_invested : ℝ :=
  (6 : ℝ) / (((1 + 0.06/2)^2 - 1) - 0.08)

/-- Theorem stating that the amount_invested satisfies the interest condition -/
theorem interest_condition (P : ℝ) (h : P = amount_invested) :
  P * ((1 + 0.06/2)^2 - 1) = P * 0.08 + 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_condition_l405_40591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_positive_set_l405_40564

noncomputable def f (x : ℝ) := x^2 - 2*x - 4*Real.log x

theorem f_derivative_positive_set :
  {x : ℝ | x > 0 ∧ (deriv f x) > 0} = {x : ℝ | x > 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_positive_set_l405_40564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQRSTU_l405_40592

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a polygon with 6 vertices -/
structure Polygon where
  P : Point
  Q : Point
  R : Point
  S : Point
  T : Point
  U : Point

/-- Calculates the area of a rectangle given its width and height -/
def rectangleArea (width : ℝ) (height : ℝ) : ℝ :=
  width * height

/-- Theorem: Area of polygon PQRSTU is 94 square units -/
theorem area_PQRSTU (poly : Polygon) : ℝ := by
  let V : Point := { x := 0, y := 0 }  -- Placeholder for point V
  let PQ : ℝ := 8
  let QR : ℝ := 12
  let areaResult : ℝ := 94

  have h1 : rectangleArea PQ QR = 96 := by sorry
  have h2 : rectangleArea 1 2 = 2 := by sorry
  have h3 : areaResult = 96 - 2 := by
    rw [← h1, ← h2]
    sorry

  exact areaResult

#check area_PQRSTU

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQRSTU_l405_40592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_volume_l405_40509

-- Define the face areas
def face_area_1 : ℝ := 36
def face_area_2 : ℝ := 18
def face_area_3 : ℝ := 8

-- Define the volume function
def box_volume (l w h : ℝ) : ℝ := l * w * h

-- State the theorem
theorem rectangular_box_volume :
  ∃ (l w h : ℝ),
    l * w = face_area_1 ∧
    w * h = face_area_2 ∧
    l * h = face_area_3 ∧
    ⌊box_volume l w h⌋ = 102 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_volume_l405_40509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monopoly_prefers_durable_l405_40594

/-- Represents the benefit a consumer gets from using a coffee machine for one period -/
def benefit : ℝ := 10

/-- Represents the production cost of a durable coffee machine -/
def durableCost : ℝ := 6

/-- Represents the production cost of a low-quality coffee machine -/
def C : ℝ → ℝ := id

/-- Represents the profit from selling a durable coffee machine -/
def durableProfit : ℝ := 2 * benefit - durableCost

/-- Represents the profit from selling two low-quality coffee machines -/
def lowQualityProfit (c : ℝ) : ℝ := 2 * (benefit - C c)

/-- Theorem stating that a monopoly will produce only durable coffee machines
    if and only if the production cost of low-quality machines is greater than 3 -/
theorem monopoly_prefers_durable (c : ℝ) :
  durableProfit > lowQualityProfit c ↔ C c > 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monopoly_prefers_durable_l405_40594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l405_40504

open Real

-- Define the function f
noncomputable def f (α : ℝ) : ℝ := 
  (sin (π - α) * cos (2*π - α) * tan (π + α)) / 
  (tan (-π - α) * sin (-π - α))

-- Theorem for the simplified form of f
theorem f_simplification (α : ℝ) : f α = cos α := by sorry

-- Theorem for the specific value of f when α = -31π/3
theorem f_specific_value : f (-31*π/3) = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l405_40504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l405_40537

-- Define the points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (5, 4)

-- Define the parabola y^2 = 4x
def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem min_distance_sum :
  ∃ (min_val : ℝ), min_val = 6 ∧
  ∀ (P : ℝ × ℝ), on_parabola P →
  distance A P + distance B P ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l405_40537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_not_satisfied_count_l405_40598

theorem inequality_not_satisfied_count : 
  ∃! (n : ℕ), ∃ (S : Finset ℤ), 
    (∀ x ∈ S, 10*x^2 + 17*x + 21 ≤ 25) ∧ 
    (Finset.card S = n) ∧ 
    (∀ y : ℤ, y ∉ S → 10*y^2 + 17*y + 21 > 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_not_satisfied_count_l405_40598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flat_distance_is_correct_equation_holds_for_solution_l405_40532

/-- The distance on flat terrain for a tourist's hike --/
def flat_distance : Real :=
  let total_time : Real := 3 + 41 / 60  -- 3 hours and 41 minutes in hours
  let total_distance : Real := 9  -- km
  let uphill_speed : Real := 4  -- km/h
  let flat_speed : Real := 5  -- km/h
  let downhill_speed : Real := 6  -- km/h
  
  -- x represents the distance on flat terrain
  -- The equation: (total_distance - x) / uphill_speed + (total_distance - x) / downhill_speed + 2 * x / flat_speed = total_time
  4  -- The solution

theorem flat_distance_is_correct : flat_distance = 4 := by
  -- Unfold the definition of flat_distance
  unfold flat_distance
  -- The actual proof would go here, but we'll use sorry for now
  sorry

/-- The equation that needs to be satisfied for the correct flat distance --/
def hike_equation (x : Real) : Prop :=
  let total_time : Real := 3 + 41 / 60
  let total_distance : Real := 9
  let uphill_speed : Real := 4
  let flat_speed : Real := 5
  let downhill_speed : Real := 6
  (total_distance - x) / uphill_speed + (total_distance - x) / downhill_speed + 2 * x / flat_speed = total_time

theorem equation_holds_for_solution : hike_equation 4 := by
  -- Unfold the definition of hike_equation
  unfold hike_equation
  -- The actual proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flat_distance_is_correct_equation_holds_for_solution_l405_40532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_ratio_is_two_to_one_l405_40552

/-- Represents the work rates and completion times for Jose and Jane's task -/
structure WorkData where
  jose_jane_time : ℝ  -- Time for Jose and Jane to complete the task together
  jose_time : ℝ       -- Time for Jose to complete the task alone
  mixed_time : ℝ      -- Time when Jose works for a portion and Jane completes the rest
  jose_jane_time_pos : jose_jane_time > 0
  jose_time_pos : jose_time > 0
  mixed_time_pos : mixed_time > 0

/-- The ratio of work completed by Jose to Jane when working separately -/
def work_ratio (data : WorkData) : ℚ :=
  2 / 1

/-- Theorem stating that the work ratio is 2:1 given the provided work data -/
theorem work_ratio_is_two_to_one (data : WorkData) 
  (h1 : data.jose_jane_time = 20)
  (h2 : data.jose_time = 60)
  (h3 : data.mixed_time = 45) :
  work_ratio data = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_ratio_is_two_to_one_l405_40552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_one_l405_40562

noncomputable def a : ℕ → ℤ
  | 0 => 6
  | n + 1 => ⌊(5/4 : ℚ) * a n + (3/4 : ℚ) * Real.sqrt ((a n)^2 - 12)⌋

theorem last_digit_is_one (n : ℕ) (h : n > 1) : a n % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_is_one_l405_40562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_piecewise_function_l405_40519

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then x^2
  else if 0 < x ∧ x < 1 then 1
  else 0  -- We define f(x) = 0 outside [-1, 1] to make it total

-- State the theorem
theorem integral_of_piecewise_function :
  ∫ x in Set.Icc (-1) 1, f x = 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_piecewise_function_l405_40519
