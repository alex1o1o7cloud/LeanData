import Mathlib

namespace NUMINAMATH_CALUDE_defective_units_shipped_percentage_l1699_169929

/-- Given that 6% of units produced are defective and 0.24% of total units
    are defective and shipped for sale, prove that 4% of defective units
    are shipped for sale. -/
theorem defective_units_shipped_percentage
  (total_defective_percent : ℝ)
  (defective_shipped_percent : ℝ)
  (h1 : total_defective_percent = 6)
  (h2 : defective_shipped_percent = 0.24) :
  defective_shipped_percent / total_defective_percent * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_percentage_l1699_169929


namespace NUMINAMATH_CALUDE_original_number_proof_l1699_169910

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 10^4 * x = 4 * (1/x)) : x = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1699_169910


namespace NUMINAMATH_CALUDE_total_spears_l1699_169913

/-- The number of spears that can be made from one sapling -/
def spears_per_sapling : ℕ := 3

/-- The number of spears that can be made from one log -/
def spears_per_log : ℕ := 9

/-- The number of saplings available -/
def num_saplings : ℕ := 6

/-- The number of logs available -/
def num_logs : ℕ := 1

/-- Theorem: The total number of spears Marcy can make is 27 -/
theorem total_spears : 
  spears_per_sapling * num_saplings + spears_per_log * num_logs = 27 := by
  sorry


end NUMINAMATH_CALUDE_total_spears_l1699_169913


namespace NUMINAMATH_CALUDE_nutmeg_amount_l1699_169949

theorem nutmeg_amount (cinnamon : Float) (difference : Float) (nutmeg : Float) : 
  cinnamon = 0.67 → 
  difference = 0.17 →
  cinnamon = nutmeg + difference →
  nutmeg = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_nutmeg_amount_l1699_169949


namespace NUMINAMATH_CALUDE_angle_bisector_inequality_l1699_169938

-- Define a triangle with side lengths and angle bisectors
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  t_a : ℝ
  t_b : ℝ
  t_c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b
  bisector_a : t_a = (b * c * (a + c - b).sqrt) / (b + c)
  bisector_b : t_b = (a * c * (b + c - a).sqrt) / (a + c)

-- State the theorem
theorem angle_bisector_inequality (t : Triangle) : (t.t_a + t.t_b) / (t.a + t.b) < 4/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_inequality_l1699_169938


namespace NUMINAMATH_CALUDE_triangle_zero_sum_implies_zero_function_l1699_169986

/-- A function f: ℝ² → ℝ with the property that the sum of its values
    at the vertices of any equilateral triangle with side length 1 is zero. -/
def TriangleZeroSum (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ A B C : ℝ × ℝ, 
    (dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1) → 
    f A + f B + f C = 0

/-- Theorem stating that any function with the TriangleZeroSum property
    is identically zero everywhere. -/
theorem triangle_zero_sum_implies_zero_function 
  (f : ℝ × ℝ → ℝ) (h : TriangleZeroSum f) : 
  ∀ x : ℝ × ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_zero_sum_implies_zero_function_l1699_169986


namespace NUMINAMATH_CALUDE_quadratic_equation_completing_square_l1699_169982

theorem quadratic_equation_completing_square (x : ℝ) :
  ∃ (q t : ℝ), (16 * x^2 - 32 * x - 512 = 0) ↔ ((x + q)^2 = t) ∧ t = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_completing_square_l1699_169982


namespace NUMINAMATH_CALUDE_x_divisibility_l1699_169945

def x : ℕ := 128 + 192 + 256 + 320 + 576 + 704 + 6464

theorem x_divisibility :
  (∃ k : ℕ, x = 8 * k) ∧
  (∃ k : ℕ, x = 16 * k) ∧
  (∃ k : ℕ, x = 32 * k) ∧
  (∃ k : ℕ, x = 64 * k) :=
by sorry

end NUMINAMATH_CALUDE_x_divisibility_l1699_169945


namespace NUMINAMATH_CALUDE_quadratic_properties_l1699_169980

/-- The quadratic function y = mx^2 - x - m + 1 where m ≠ 0 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - x - m + 1

theorem quadratic_properties (m : ℝ) (hm : m ≠ 0) :
  (∀ x, f m x = 0 → x = 1 ∨ x = (1 - m) / m) ∧
  (m < 0 → ∀ a b, f m a = 0 → f m b = 0 → a ≠ b → |a - b| > 2) ∧
  (m > 1 → ∀ x > 1, ∀ y > x, f m y > f m x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1699_169980


namespace NUMINAMATH_CALUDE_function_composition_equality_l1699_169987

theorem function_composition_equality (a b : ℝ) :
  (∀ x, (3 * ((a * x + b) : ℝ) - 4 = 4 * x + 5)) →
  a + b = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1699_169987


namespace NUMINAMATH_CALUDE_bucket_problem_l1699_169994

/-- Given two buckets A and B with unknown amounts of water, if transferring 6 liters from A to B
    results in A containing one-third of B's new amount, and transferring 6 liters from B to A
    results in B containing one-half of A's new amount, then A initially contains 13.2 liters of water. -/
theorem bucket_problem (A B : ℝ) 
    (h1 : A - 6 = (1/3) * (B + 6))
    (h2 : B - 6 = (1/2) * (A + 6)) :
    A = 13.2 := by
  sorry

end NUMINAMATH_CALUDE_bucket_problem_l1699_169994


namespace NUMINAMATH_CALUDE_hot_dog_problem_l1699_169951

theorem hot_dog_problem (cost_per_hot_dog : ℕ) (total_paid : ℕ) (h1 : cost_per_hot_dog = 50) (h2 : total_paid = 300) :
  total_paid / cost_per_hot_dog = 6 := by
sorry

end NUMINAMATH_CALUDE_hot_dog_problem_l1699_169951


namespace NUMINAMATH_CALUDE_square_area_ratio_l1699_169953

theorem square_area_ratio (y : ℝ) (hy : y > 0) : 
  (y^2) / ((3*y)^2) = 1/9 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1699_169953


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l1699_169933

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equals_i : (1 + i^2017) / (1 - i) = i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l1699_169933


namespace NUMINAMATH_CALUDE_opposite_plus_two_equals_zero_l1699_169926

theorem opposite_plus_two_equals_zero (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_plus_two_equals_zero_l1699_169926


namespace NUMINAMATH_CALUDE_base8_subtraction_to_base4_l1699_169954

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 --/
def base10ToBase4 (n : ℕ) : List ℕ := sorry

/-- Subtracts two numbers in base 8 --/
def subtractBase8 (a b : ℕ) : ℕ := sorry

theorem base8_subtraction_to_base4 :
  let a := 643
  let b := 257
  let result := subtractBase8 a b
  base10ToBase4 (base8ToBase10 result) = [3, 3, 1, 1, 0] := by sorry

end NUMINAMATH_CALUDE_base8_subtraction_to_base4_l1699_169954


namespace NUMINAMATH_CALUDE_smallest_number_l1699_169942

def number_set : Finset ℤ := {0, -3, 2, -2}

theorem smallest_number : 
  ∀ x ∈ number_set, -3 ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1699_169942


namespace NUMINAMATH_CALUDE_sum_of_distinct_integers_l1699_169985

theorem sum_of_distinct_integers (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120 →
  a + b + c + d + e = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_integers_l1699_169985


namespace NUMINAMATH_CALUDE_rectangle_area_stage_7_l1699_169963

/-- The side length of each square in inches -/
def square_side : ℝ := 4

/-- The number of squares at Stage 7 -/
def num_squares : ℕ := 7

/-- The area of the rectangle at Stage 7 in square inches -/
def rectangle_area : ℝ := (square_side ^ 2) * num_squares

/-- Theorem: The area of the rectangle at Stage 7 is 112 square inches -/
theorem rectangle_area_stage_7 : rectangle_area = 112 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_stage_7_l1699_169963


namespace NUMINAMATH_CALUDE_multiple_of_smaller_number_l1699_169959

theorem multiple_of_smaller_number (s l : ℝ) (h1 : s + l = 24) (h2 : s = 10) : ∃ m : ℝ, m * s = 5 * l ∧ m = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_smaller_number_l1699_169959


namespace NUMINAMATH_CALUDE_dot_product_range_l1699_169927

/-- The ellipse equation -/
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 16) + (P.2^2 / 15) = 1

/-- The circle equation -/
def is_on_circle (Q : ℝ × ℝ) : Prop :=
  (Q.1 - 1)^2 + Q.2^2 = 4

/-- Definition of a diameter of the circle -/
def is_diameter (E F : ℝ × ℝ) : Prop :=
  is_on_circle E ∧ is_on_circle F ∧ 
  (E.1 + F.1 = 2) ∧ (E.2 + F.2 = 0)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem dot_product_range (P E F : ℝ × ℝ) :
  is_on_ellipse P → is_diameter E F →
  5 ≤ dot_product (E.1 - P.1, E.2 - P.2) (F.1 - P.1, F.2 - P.2) ∧
  dot_product (E.1 - P.1, E.2 - P.2) (F.1 - P.1, F.2 - P.2) ≤ 21 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l1699_169927


namespace NUMINAMATH_CALUDE_sine_inequality_solution_l1699_169911

theorem sine_inequality_solution (x y : Real) :
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), 
   ∀ y ∈ Set.Icc 0 (2 * Real.pi), 
   Real.sin (x + y) ≤ Real.sin x + Real.sin y) ↔ 
  y ∈ Set.Icc 0 Real.pi :=
sorry

end NUMINAMATH_CALUDE_sine_inequality_solution_l1699_169911


namespace NUMINAMATH_CALUDE_fuel_tank_ethanol_percentage_l1699_169932

/-- Fuel tank problem -/
theorem fuel_tank_ethanol_percentage
  (tank_capacity : ℝ)
  (fuel_a_ethanol_percentage : ℝ)
  (total_ethanol : ℝ)
  (fuel_a_volume : ℝ)
  (h1 : tank_capacity = 214)
  (h2 : fuel_a_ethanol_percentage = 12 / 100)
  (h3 : total_ethanol = 30)
  (h4 : fuel_a_volume = 106) :
  (total_ethanol - fuel_a_ethanol_percentage * fuel_a_volume) / (tank_capacity - fuel_a_volume) = 16 / 100 := by
sorry

end NUMINAMATH_CALUDE_fuel_tank_ethanol_percentage_l1699_169932


namespace NUMINAMATH_CALUDE_fish_eaten_ratio_l1699_169906

def total_rocks : ℕ := 10
def rocks_left : ℕ := 7
def rocks_spit : ℕ := 2

def rocks_eaten : ℕ := total_rocks - rocks_left + rocks_spit

theorem fish_eaten_ratio :
  (rocks_eaten : ℚ) / total_rocks = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fish_eaten_ratio_l1699_169906


namespace NUMINAMATH_CALUDE_odd_function_geometric_sequence_l1699_169914

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x + a / x
  else -(Real.log (-x) + a / (-x))

theorem odd_function_geometric_sequence (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    x₁ + x₄ = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧
      f a x₂ / f a x₁ = q ∧
      f a x₃ / f a x₂ = q ∧
      f a x₄ / f a x₃ = q) →
  a ≤ Real.sqrt 3 / (2 * Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_odd_function_geometric_sequence_l1699_169914


namespace NUMINAMATH_CALUDE_basketball_score_possibilities_count_basketball_scores_l1699_169918

def basketball_scores (n : ℕ) : Finset ℕ :=
  Finset.image (λ k => 3 * k + 2 * (n - k)) (Finset.range (n + 1))

theorem basketball_score_possibilities :
  basketball_scores 5 = {10, 11, 12, 13, 14, 15} :=
sorry

theorem count_basketball_scores :
  (basketball_scores 5).card = 6 :=
sorry

end NUMINAMATH_CALUDE_basketball_score_possibilities_count_basketball_scores_l1699_169918


namespace NUMINAMATH_CALUDE_pavilion_pillar_height_l1699_169988

-- Define the regular octagon
structure RegularOctagon where
  side_length : ℝ
  center : ℝ × ℝ

-- Define a pillar
structure Pillar where
  base : ℝ × ℝ
  height : ℝ

-- Define the pavilion
structure Pavilion where
  octagon : RegularOctagon
  pillars : Fin 8 → Pillar

-- Define the theorem
theorem pavilion_pillar_height 
  (pav : Pavilion) 
  (h_a : (pav.pillars 0).height = 15)
  (h_b : (pav.pillars 1).height = 11)
  (h_c : (pav.pillars 2).height = 13) :
  (pav.pillars 5).height = 32 :=
sorry

end NUMINAMATH_CALUDE_pavilion_pillar_height_l1699_169988


namespace NUMINAMATH_CALUDE_roots_and_element_imply_value_l1699_169924

theorem roots_and_element_imply_value (a : ℝ) :
  let A := {x : ℝ | (x - a) * (x - a + 1) = 0}
  2 ∈ A → (a = 2 ∨ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_roots_and_element_imply_value_l1699_169924


namespace NUMINAMATH_CALUDE_randy_practice_hours_l1699_169901

/-- Calculates the number of hours per day Randy needs to practice piano to become an expert --/
def hours_per_day_to_expert (current_age : ℕ) (target_age : ℕ) (practice_days_per_week : ℕ) (vacation_weeks : ℕ) (hours_to_expert : ℕ) : ℚ :=
  let years_to_practice := target_age - current_age
  let weeks_per_year := 52
  let practice_weeks := weeks_per_year - vacation_weeks
  let practice_days_per_year := practice_weeks * practice_days_per_week
  let total_practice_days := years_to_practice * practice_days_per_year
  hours_to_expert / total_practice_days

/-- Theorem stating that Randy needs to practice 5 hours per day to become a piano expert --/
theorem randy_practice_hours :
  hours_per_day_to_expert 12 20 5 2 10000 = 5 := by
  sorry

end NUMINAMATH_CALUDE_randy_practice_hours_l1699_169901


namespace NUMINAMATH_CALUDE_problem_solution_l1699_169996

noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log x) / x + b

def has_tangent_line (f : ℝ → ℝ) (x₀ y₀ m : ℝ) : Prop :=
  ∃ (f' : ℝ → ℝ), (∀ x, HasDerivAt f (f' x) x) ∧ f' x₀ = m ∧ f x₀ = y₀

noncomputable def g (c x : ℝ) : ℝ := Real.log x / Real.log c - x

def has_zero_point (g : ℝ → ℝ) : Prop := ∃ x > 0, g x = 0

theorem problem_solution :
  ∀ (a b : ℝ),
    (has_tangent_line (f a b) 1 0 1) →
    (a = 1 ∧ b = 0) ∧
    (∀ x > 0, f 1 0 x ≤ 1 / Real.exp 1) ∧
    (∀ c > 0, c ≠ 1 → has_zero_point (g c) → c ≤ Real.exp (1 / Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1699_169996


namespace NUMINAMATH_CALUDE_part_one_part_two_l1699_169948

-- Define the function f
def f (a : ℝ) (n : ℕ+) (x : ℝ) : ℝ := a * x^n.val * (1 - x)

-- Part 1
theorem part_one (a : ℝ) :
  (∀ x > 0, f a 2 x ≤ 4/27) ∧ (∃ x > 0, f a 2 x = 4/27) → a = 1 := by sorry

-- Part 2
theorem part_two (n : ℕ+) (m : ℝ) :
  (∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f 1 n x = m ∧ f 1 n y = m) →
  0 < m ∧ m < (n.val ^ n.val : ℝ) / ((n.val + 1 : ℕ) ^ (n.val + 1)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1699_169948


namespace NUMINAMATH_CALUDE_power_function_property_l1699_169909

/-- Given a function f(x) = x^α where f(2) = 4, prove that f(-1) = 1 -/
theorem power_function_property (α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x ^ α) 
  (h2 : f 2 = 4) : 
  f (-1) = 1 := by
sorry

end NUMINAMATH_CALUDE_power_function_property_l1699_169909


namespace NUMINAMATH_CALUDE_parabola_focus_focus_of_y_eq_4x_squared_l1699_169976

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1 / (4 * a))
  ∀ x y : ℝ, y = a * x^2 → (x - f.1)^2 + (y - f.2)^2 = (y - f.2 + 1 / (4 * a))^2 :=
sorry

/-- The focus of the parabola y = 4x^2 is at (0, 1/16) -/
theorem focus_of_y_eq_4x_squared :
  let f : ℝ × ℝ := (0, 1/16)
  ∀ x y : ℝ, y = 4 * x^2 → (x - f.1)^2 + (y - f.2)^2 = (y - f.2 + 1/16)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_focus_of_y_eq_4x_squared_l1699_169976


namespace NUMINAMATH_CALUDE_no_four_integers_product_plus_2002_square_l1699_169989

theorem no_four_integers_product_plus_2002_square : 
  ¬ ∃ (n₁ n₂ n₃ n₄ : ℕ+), 
    (∀ (i j : Fin 4), i ≠ j → ∃ (m : ℕ), (n₁ :: n₂ :: n₃ :: n₄ :: []).get i * (n₁ :: n₂ :: n₃ :: n₄ :: []).get j + 2002 = m^2) :=
by sorry

end NUMINAMATH_CALUDE_no_four_integers_product_plus_2002_square_l1699_169989


namespace NUMINAMATH_CALUDE_reduce_to_single_digit_l1699_169908

/-- Represents the operation of splitting digits and summing -/
def digitSplitSum (n : ℕ) : ℕ → ℕ :=
  sorry

/-- Predicate for a number being single-digit -/
def isSingleDigit (n : ℕ) : Prop :=
  n < 10

/-- Theorem stating that any natural number can be reduced to a single digit in at most 15 steps -/
theorem reduce_to_single_digit (N : ℕ) :
  ∃ (seq : Fin 16 → ℕ), seq 0 = N ∧ isSingleDigit (seq 15) ∧
  ∀ i : Fin 15, seq (i + 1) = digitSplitSum (seq i) (seq i) :=
sorry

end NUMINAMATH_CALUDE_reduce_to_single_digit_l1699_169908


namespace NUMINAMATH_CALUDE_vector_subtraction_l1699_169974

/-- Given vectors a and b in ℝ², prove that a - 2b = (7, 3) -/
theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1699_169974


namespace NUMINAMATH_CALUDE_paris_saturday_study_hours_l1699_169936

/-- The number of hours Paris studies on Saturdays during the semester -/
def saturday_study_hours (
  semester_weeks : ℕ)
  (weekday_study_hours : ℕ)
  (sunday_study_hours : ℕ)
  (total_study_hours : ℕ) : ℕ :=
  total_study_hours - (semester_weeks * 5 * weekday_study_hours) - (semester_weeks * sunday_study_hours)

/-- Theorem stating that Paris studies 60 hours on Saturdays during the semester -/
theorem paris_saturday_study_hours :
  saturday_study_hours 15 3 5 360 = 60 := by
  sorry


end NUMINAMATH_CALUDE_paris_saturday_study_hours_l1699_169936


namespace NUMINAMATH_CALUDE_parabola_single_intersection_l1699_169975

/-- A parabola defined by y = x^2 + 2x + c + 1 -/
def parabola (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c + 1

/-- A horizontal line defined by y = 1 -/
def line : ℝ → ℝ := λ _ => 1

/-- The condition for the parabola to intersect the line at only one point -/
def single_intersection (c : ℝ) : Prop :=
  ∃! x, parabola c x = line x

theorem parabola_single_intersection :
  ∀ c : ℝ, single_intersection c ↔ c = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_single_intersection_l1699_169975


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l1699_169905

/-- The line x + y = c is a perpendicular bisector of the line segment from (2,4) to (6,8) -/
def is_perpendicular_bisector (c : ℝ) : Prop :=
  let midpoint := ((2 + 6) / 2, (4 + 8) / 2)
  (midpoint.1 + midpoint.2 = c) ∧
  (∀ (x y : ℝ), x + y = c → (x - 2)^2 + (y - 4)^2 = (x - 6)^2 + (y - 8)^2)

/-- If the line x + y = c is a perpendicular bisector of the line segment from (2,4) to (6,8), then c = 10 -/
theorem perpendicular_bisector_c_value :
  ∃ c, is_perpendicular_bisector c → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l1699_169905


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1699_169967

theorem circle_equation_proof (x y : ℝ) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 4*x + 6*y - 3 = 0
  let M : ℝ × ℝ := (-1, 1)
  let new_circle : ℝ → ℝ → Prop := λ x y => (x - 2)^2 + (y + 3)^2 = 25
  (∃ h k r : ℝ, ∀ x y : ℝ, C x y ↔ (x - h)^2 + (y - k)^2 = r^2) →
  (new_circle M.1 M.2) ∧
  (∀ x y : ℝ, C x y ↔ (x - 2)^2 + (y + 3)^2 = r^2) →
  (∀ x y : ℝ, new_circle x y ↔ (x - 2)^2 + (y + 3)^2 = 25) :=
by sorry


end NUMINAMATH_CALUDE_circle_equation_proof_l1699_169967


namespace NUMINAMATH_CALUDE_function_range_l1699_169971

/-- The function y = (3 * sin(x) + 1) / (sin(x) + 2) has a range of [-2, 4/3] -/
theorem function_range (x : ℝ) : 
  let y := (3 * Real.sin x + 1) / (Real.sin x + 2)
  ∃ (a b : ℝ), a = -2 ∧ b = 4/3 ∧ a ≤ y ∧ y ≤ b ∧
  (∃ (x₁ x₂ : ℝ), 
    (3 * Real.sin x₁ + 1) / (Real.sin x₁ + 2) = a ∧
    (3 * Real.sin x₂ + 1) / (Real.sin x₂ + 2) = b) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l1699_169971


namespace NUMINAMATH_CALUDE_weaving_increase_proof_l1699_169921

/-- Represents the daily increase in cloth production -/
def daily_increase : ℚ := 16 / 29

/-- Represents the number of days in a month -/
def days_in_month : ℕ := 30

/-- Represents the initial daily production in meters -/
def initial_production : ℚ := 5

/-- Represents the total production in meters for the month -/
def total_production : ℚ := 390

/-- Theorem stating that given the initial conditions, the daily increase in production is 16/29 meters -/
theorem weaving_increase_proof :
  initial_production * days_in_month + 
  (days_in_month * (days_in_month - 1) / 2) * daily_increase = 
  total_production :=
by
  sorry


end NUMINAMATH_CALUDE_weaving_increase_proof_l1699_169921


namespace NUMINAMATH_CALUDE_power_function_through_point_l1699_169969

theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^a) → f 2 = 8 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1699_169969


namespace NUMINAMATH_CALUDE_b_score_is_93_l1699_169943

/-- Represents the scores of five people in an exam -/
structure ExamScores where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- The average score of all five people is 90 -/
def average_all (scores : ExamScores) : Prop :=
  (scores.A + scores.B + scores.C + scores.D + scores.E) / 5 = 90

/-- The average score of A, B, and C is 86 -/
def average_ABC (scores : ExamScores) : Prop :=
  (scores.A + scores.B + scores.C) / 3 = 86

/-- The average score of B, D, and E is 95 -/
def average_BDE (scores : ExamScores) : Prop :=
  (scores.B + scores.D + scores.E) / 3 = 95

/-- Theorem: Given the conditions, B's score is 93 -/
theorem b_score_is_93 (scores : ExamScores) 
  (h1 : average_all scores) 
  (h2 : average_ABC scores) 
  (h3 : average_BDE scores) : 
  scores.B = 93 := by
  sorry

end NUMINAMATH_CALUDE_b_score_is_93_l1699_169943


namespace NUMINAMATH_CALUDE_double_layer_cake_cost_double_layer_cake_cost_is_seven_l1699_169990

theorem double_layer_cake_cost (single_layer_cost : ℝ) 
                               (single_layer_quantity : ℕ) 
                               (double_layer_quantity : ℕ) 
                               (total_paid : ℝ) 
                               (change_received : ℝ) : ℝ :=
  let total_spent := total_paid - change_received
  let single_layer_total := single_layer_cost * single_layer_quantity
  let double_layer_total := total_spent - single_layer_total
  double_layer_total / double_layer_quantity

theorem double_layer_cake_cost_is_seven :
  double_layer_cake_cost 4 7 5 100 37 = 7 := by
  sorry

end NUMINAMATH_CALUDE_double_layer_cake_cost_double_layer_cake_cost_is_seven_l1699_169990


namespace NUMINAMATH_CALUDE_digit_150_is_7_l1699_169915

/-- The decimal representation of 17/70 -/
def decimal_rep : ℚ := 17 / 70

/-- The length of the repeating sequence in the decimal representation of 17/70 -/
def repeat_length : ℕ := 6

/-- The nth digit after the decimal point in the decimal representation of 17/70 -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Theorem: The 150th digit after the decimal point in the decimal representation of 17/70 is 7 -/
theorem digit_150_is_7 : nth_digit 150 = 7 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_7_l1699_169915


namespace NUMINAMATH_CALUDE_weighted_am_gm_inequality_l1699_169952

theorem weighted_am_gm_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  0.2 * a + 0.3 * b + 0.5 * c ≥ (a * b * c) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_weighted_am_gm_inequality_l1699_169952


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1699_169937

/-- Represents the atomic weight of an element in atomic mass units (amu) -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "Al" => 26.98
  | "O"  => 16.00
  | "C"  => 12.01
  | "N"  => 14.01
  | "H"  => 1.008
  | _    => 0  -- Default case, though not used in this problem

/-- Calculates the molecular weight of a compound given its composition -/
def molecular_weight (Al O C N H : ℕ) : ℝ :=
  Al * atomic_weight "Al" +
  O  * atomic_weight "O"  +
  C  * atomic_weight "C"  +
  N  * atomic_weight "N"  +
  H  * atomic_weight "H"

/-- Theorem stating that the molecular weight of the given compound is 146.022 amu -/
theorem compound_molecular_weight :
  molecular_weight 2 3 1 2 4 = 146.022 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1699_169937


namespace NUMINAMATH_CALUDE_train_crossing_time_l1699_169907

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_speed_kmph : ℝ) (train_length : ℝ) (platform_length : ℝ) :
  train_speed_kmph = 72 →
  train_length = 280.0416 →
  platform_length = 240 →
  (train_length + platform_length) / (train_speed_kmph * (1 / 3.6)) = 26.00208 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1699_169907


namespace NUMINAMATH_CALUDE_base8_243_equals_base10_163_l1699_169995

def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base8_243_equals_base10_163 :
  base8_to_base10 [3, 4, 2] = 163 := by
  sorry

end NUMINAMATH_CALUDE_base8_243_equals_base10_163_l1699_169995


namespace NUMINAMATH_CALUDE_folded_paper_corner_distance_l1699_169939

/-- Represents a square sheet of paper with white front and black back -/
structure Paper where
  side : ℝ
  area : ℝ
  area_eq : area = side * side

/-- Represents the folded state of the paper -/
structure FoldedPaper where
  paper : Paper
  fold_length : ℝ
  black_area : ℝ
  white_area : ℝ
  black_twice_white : black_area = 2 * white_area
  areas_sum : black_area + white_area = paper.area

/-- The theorem to be proved -/
theorem folded_paper_corner_distance 
  (p : Paper) 
  (fp : FoldedPaper) 
  (h_area : p.area = 18) 
  (h_fp_paper : fp.paper = p) :
  Real.sqrt 2 * fp.fold_length = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_corner_distance_l1699_169939


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l1699_169993

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc (-2) 3

-- Define the new function g
def g (x : ℝ) : ℝ := f (2 * x - 1)

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc (-1/2) 2 := by sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l1699_169993


namespace NUMINAMATH_CALUDE_combined_height_problem_l1699_169928

/-- Given that Mr. Martinez is two feet taller than Chiquita and Chiquita is 5 feet tall,
    prove that their combined height is 12 feet. -/
theorem combined_height_problem (chiquita_height : ℝ) (martinez_height : ℝ) :
  chiquita_height = 5 →
  martinez_height = chiquita_height + 2 →
  chiquita_height + martinez_height = 12 :=
by sorry

end NUMINAMATH_CALUDE_combined_height_problem_l1699_169928


namespace NUMINAMATH_CALUDE_line_intercept_sum_l1699_169919

/-- A line in the x-y plane -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℚ := -l.y_intercept / l.slope

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℚ := l.y_intercept

/-- The sum of x-intercept and y-intercept of a line -/
def intercept_sum (l : Line) : ℚ := x_intercept l + y_intercept l

theorem line_intercept_sum :
  ∃ (l : Line), l.slope = -3 ∧ l.y_intercept = -13 ∧ intercept_sum l = -52/3 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l1699_169919


namespace NUMINAMATH_CALUDE_exists_universal_program_l1699_169955

/-- Represents a position on the chessboard --/
structure Position :=
  (x : Fin 8) (y : Fin 8)

/-- Represents a labyrinth configuration --/
def Labyrinth := Fin 8 → Fin 8 → Bool

/-- Represents a move command --/
inductive Command
  | RIGHT
  | LEFT
  | UP
  | DOWN

/-- Represents a program as a list of commands --/
def Program := List Command

/-- Checks if a square is accessible in the labyrinth --/
def isAccessible (lab : Labyrinth) (pos : Position) : Bool :=
  lab pos.x pos.y

/-- Executes a single command on a position in a labyrinth --/
def executeCommand (lab : Labyrinth) (pos : Position) (cmd : Command) : Position :=
  sorry

/-- Executes a program on a position in a labyrinth --/
def executeProgram (lab : Labyrinth) (pos : Position) (prog : Program) : Position :=
  sorry

/-- Checks if a program visits all accessible squares in a labyrinth from a given starting position --/
def visitsAllAccessible (lab : Labyrinth) (start : Position) (prog : Program) : Prop :=
  sorry

/-- The main theorem: there exists a program that visits all accessible squares
    for any labyrinth and starting position --/
theorem exists_universal_program :
  ∃ (prog : Program),
    ∀ (lab : Labyrinth) (start : Position),
      visitsAllAccessible lab start prog :=
sorry

end NUMINAMATH_CALUDE_exists_universal_program_l1699_169955


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1699_169957

-- Define the set of real numbers between -3 and 2
def OpenInterval : Set ℝ := {x | -3 < x ∧ x < 2}

-- Define the quadratic function ax^2 + bx + c
def QuadraticF (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the reversed quadratic function cx^2 + bx + a
def ReversedQuadraticF (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + b * x + a

-- Define the solution set of the reversed quadratic inequality
def ReversedSolutionSet : Set ℝ := {x | x < -1/3 ∨ x > 1/2}

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x ∈ OpenInterval, QuadraticF a b c x > 0) :
  ∀ x, ReversedQuadraticF a b c x > 0 ↔ x ∈ ReversedSolutionSet :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1699_169957


namespace NUMINAMATH_CALUDE_meal_combinations_count_l1699_169965

/-- Represents the number of main dishes available -/
def num_main_dishes : ℕ := 2

/-- Represents the number of stir-fry dishes available -/
def num_stir_fry_dishes : ℕ := 4

/-- Calculates the total number of meal combinations -/
def total_combinations : ℕ := num_main_dishes * num_stir_fry_dishes

/-- Theorem stating that the total number of meal combinations is 8 -/
theorem meal_combinations_count : total_combinations = 8 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_count_l1699_169965


namespace NUMINAMATH_CALUDE_trig_product_equality_l1699_169940

theorem trig_product_equality : 
  Real.sin (4 * Real.pi / 3) * Real.cos (5 * Real.pi / 6) * Real.tan (-4 * Real.pi / 3) = -3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equality_l1699_169940


namespace NUMINAMATH_CALUDE_sin_2017pi_over_6_l1699_169900

theorem sin_2017pi_over_6 : Real.sin ((2017 * π) / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2017pi_over_6_l1699_169900


namespace NUMINAMATH_CALUDE_find_y_l1699_169930

theorem find_y (c d : ℝ) (y : ℝ) (h1 : d > 0) : 
  ((3 * c) ^ (3 * d) = c^d * y^d) → y = 27 * c^2 := by
sorry

end NUMINAMATH_CALUDE_find_y_l1699_169930


namespace NUMINAMATH_CALUDE_equal_color_squares_count_l1699_169992

/-- Represents a 5x5 grid with some cells painted black -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Counts the number of squares in the grid with equal black and white cells -/
def countEqualColorSquares (g : Grid) : ℕ :=
  let count2x2 := (5 - 2 + 1)^2 - 2  -- Total 2x2 squares minus those containing the center
  let count4x4 := 2  -- Lower two 4x4 squares meet the criterion
  count2x2 + count4x4

/-- Theorem stating that there are exactly 16 squares with equal black and white cells -/
theorem equal_color_squares_count (g : Grid) : countEqualColorSquares g = 16 := by
  sorry

end NUMINAMATH_CALUDE_equal_color_squares_count_l1699_169992


namespace NUMINAMATH_CALUDE_zongzi_pricing_and_purchase_l1699_169920

-- Define the total number of zongzi and total cost
def total_zongzi : ℕ := 1100
def total_cost : ℚ := 3000

-- Define the price ratio between type A and B
def price_ratio : ℚ := 1.2

-- Define the new total number of zongzi and budget
def new_total_zongzi : ℕ := 2600
def new_budget : ℚ := 7000

-- Define the unit prices of type A and B zongzi
def unit_price_B : ℚ := 2.5
def unit_price_A : ℚ := 3

-- Define the maximum number of type A zongzi in the second scenario
def max_type_A : ℕ := 1000

theorem zongzi_pricing_and_purchase :
  -- The cost of purchasing type A is the same as type B
  (total_cost / 2) / unit_price_A = (total_cost / 2) / unit_price_B ∧
  -- The unit price of type A is 1.2 times the unit price of type B
  unit_price_A = price_ratio * unit_price_B ∧
  -- The total number of zongzi purchased is 1100
  (total_cost / 2) / unit_price_A + (total_cost / 2) / unit_price_B = total_zongzi ∧
  -- The maximum number of type A zongzi in the second scenario is 1000
  max_type_A * unit_price_A + (new_total_zongzi - max_type_A) * unit_price_B ≤ new_budget ∧
  ∀ n : ℕ, n > max_type_A → n * unit_price_A + (new_total_zongzi - n) * unit_price_B > new_budget :=
by sorry

end NUMINAMATH_CALUDE_zongzi_pricing_and_purchase_l1699_169920


namespace NUMINAMATH_CALUDE_doubling_base_theorem_l1699_169999

theorem doubling_base_theorem (a b x : ℝ) (h1 : b ≠ 0) :
  (2 * a) ^ b = a ^ b * x ^ b → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_doubling_base_theorem_l1699_169999


namespace NUMINAMATH_CALUDE_expression_equality_l1699_169935

theorem expression_equality (x : ℝ) : x*(x*(x*(3-2*x)-4)+8)+3*x^2 = -2*x^4 + 3*x^3 - x^2 + 8*x := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1699_169935


namespace NUMINAMATH_CALUDE_inequality_proof_l1699_169978

theorem inequality_proof (a b : ℝ) (θ : ℝ) : 
  abs a + abs b ≤ 
  Real.sqrt (a^2 * Real.cos θ^2 + b^2 * Real.sin θ^2) + 
  Real.sqrt (a^2 * Real.sin θ^2 + b^2 * Real.cos θ^2) ∧
  Real.sqrt (a^2 * Real.cos θ^2 + b^2 * Real.sin θ^2) + 
  Real.sqrt (a^2 * Real.sin θ^2 + b^2 * Real.cos θ^2) ≤ 
  Real.sqrt (2 * (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1699_169978


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1699_169931

/-- Given a geometric sequence {a_n} where a₁ = 3 and a₄ = 24, prove that a₃ + a₄ + a₅ = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                                  -- a₁ = 3
  a 4 = 24 →                                 -- a₄ = 24
  a 3 + a 4 + a 5 = 84 :=                    -- prove a₃ + a₄ + a₅ = 84
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1699_169931


namespace NUMINAMATH_CALUDE_symmetric_points_on_circle_l1699_169944

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 1 = 0

-- Define the line equation
def line_equation (x y : ℝ) (c : ℝ) : Prop :=
  2*x + y + c = 0

-- Theorem statement
theorem symmetric_points_on_circle (c : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_equation x₁ y₁ ∧
    circle_equation x₂ y₂ ∧
    (∃ (x_mid y_mid : ℝ),
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 ∧
      line_equation x_mid y_mid c)) →
  c = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_on_circle_l1699_169944


namespace NUMINAMATH_CALUDE_red_window_exchange_equations_l1699_169984

/-- Represents the relationship between online and offline booth transactions -/
theorem red_window_exchange_equations 
  (x y : ℝ)  -- Total transaction amounts for online (x) and offline (y) booths
  (online_booths : ℕ := 44)  -- Number of online booths
  (offline_booths : ℕ := 71)  -- Number of offline booths
  (h1 : y - 7 * x = 1.8)  -- Relationship between total transaction amounts
  (h2 : y / offline_booths - x / online_booths = 0.3)  -- Difference in average transaction amounts
  : ∃ (system : ℝ × ℝ → Prop), 
    system (x, y) ∧ 
    (∀ (a b : ℝ), system (a, b) ↔ (b - 7 * a = 1.8 ∧ b / offline_booths - a / online_booths = 0.3)) :=
by
  sorry


end NUMINAMATH_CALUDE_red_window_exchange_equations_l1699_169984


namespace NUMINAMATH_CALUDE_race_finish_count_l1699_169998

/-- Calculates the number of men who finished a race given specific conditions --/
def men_finished_race (total_men : ℕ) : ℕ :=
  let tripped := total_men / 4
  let tripped_finished := tripped / 3
  let remaining_after_trip := total_men - tripped
  let dehydrated := remaining_after_trip * 2 / 3
  let dehydrated_finished := dehydrated * 4 / 5
  let remaining_after_dehydration := remaining_after_trip - dehydrated
  let lost := remaining_after_dehydration * 12 / 100
  let lost_finished := lost / 2
  let remaining_after_lost := remaining_after_dehydration - lost
  let faced_obstacle := remaining_after_lost * 3 / 8
  let obstacle_finished := faced_obstacle * 2 / 5
  tripped_finished + dehydrated_finished + lost_finished + obstacle_finished

/-- Theorem stating that given 80 men in the race, 41 men finished --/
theorem race_finish_count : men_finished_race 80 = 41 := by
  sorry

#eval men_finished_race 80

end NUMINAMATH_CALUDE_race_finish_count_l1699_169998


namespace NUMINAMATH_CALUDE_least_colors_for_hidden_edges_l1699_169912

/-- The size of the grid (both width and height) -/
def gridSize : ℕ := 7

/-- The total number of edges in the grid -/
def totalEdges : ℕ := 2 * gridSize * (gridSize - 1)

/-- The expected number of hidden edges given N colors -/
def expectedHiddenEdges (N : ℕ) : ℚ := totalEdges / N

/-- Theorem stating the least N for which the expected number of hidden edges is less than 3 -/
theorem least_colors_for_hidden_edges :
  ∀ N : ℕ, N ≥ 29 ↔ expectedHiddenEdges N < 3 :=
sorry

end NUMINAMATH_CALUDE_least_colors_for_hidden_edges_l1699_169912


namespace NUMINAMATH_CALUDE_plate_arrangement_circular_table_l1699_169977

def plate_arrangement (b r g o y : ℕ) : ℕ :=
  let total := b + r + g + o + y
  let all_arrangements := Nat.factorial (total - 1) / (Nat.factorial b * Nat.factorial r * Nat.factorial g * Nat.factorial o * Nat.factorial y)
  let adjacent_green := Nat.factorial (total - g + 1) / (Nat.factorial b * Nat.factorial r * Nat.factorial o * Nat.factorial y) * Nat.factorial g
  all_arrangements - adjacent_green

theorem plate_arrangement_circular_table :
  plate_arrangement 6 3 3 2 2 = 
    Nat.factorial 15 / (Nat.factorial 6 * Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 2) - 
    (Nat.factorial 14 / (Nat.factorial 6 * Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1) * Nat.factorial 3) :=
by
  sorry

end NUMINAMATH_CALUDE_plate_arrangement_circular_table_l1699_169977


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1699_169934

def A : Set ℤ := {-2, -1, 3, 4}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1699_169934


namespace NUMINAMATH_CALUDE_combined_discount_optimal_l1699_169970

/-- Represents the cost calculation for a clothing purchase with discount options -/
def ClothingPurchase (x : ℕ) : Prop :=
  x > 30 ∧
  let jacket_price : ℕ := 100
  let tshirt_price : ℕ := 60
  let option1_cost : ℕ := 3000 + 60 * (x - 30)
  let option2_cost : ℕ := 2400 + 48 * x
  let combined_cost : ℕ := 3000 + 48 * (x - 30)
  combined_cost ≤ min option1_cost option2_cost

/-- Theorem stating that the combined discount strategy is optimal for any valid x -/
theorem combined_discount_optimal (x : ℕ) : ClothingPurchase x := by
  sorry

end NUMINAMATH_CALUDE_combined_discount_optimal_l1699_169970


namespace NUMINAMATH_CALUDE_left_handed_women_percentage_l1699_169923

/-- Represents the population distribution in Smithtown -/
structure SmithtownPopulation where
  right_handed : ℕ
  left_handed : ℕ
  men : ℕ
  women : ℕ

/-- Conditions for Smithtown's population distribution -/
def valid_distribution (p : SmithtownPopulation) : Prop :=
  p.right_handed = 3 * p.left_handed ∧
  p.men = 3 * p.women / 2 ∧
  p.right_handed + p.left_handed = p.men + p.women ∧
  p.right_handed ≥ p.men

/-- Theorem: In a valid Smithtown population distribution, 
    left-handed women constitute 25% of the total population -/
theorem left_handed_women_percentage 
  (p : SmithtownPopulation) 
  (h : valid_distribution p) : 
  (p.left_handed : ℚ) / (p.right_handed + p.left_handed : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_women_percentage_l1699_169923


namespace NUMINAMATH_CALUDE_triangle_construction_valid_l1699_169902

/-- A triangle can be constructed with perimeter k, one side c, and angle difference δ
    between angles opposite the other two sides if and only if 2c < k. -/
theorem triangle_construction_valid (k c : ℝ) (δ : ℝ) :
  (∃ (a b : ℝ) (α β γ : ℝ),
    a + b + c = k ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    α + β + γ = π ∧
    α - β = δ ∧
    0 < α ∧ α < π ∧
    0 < β ∧ β < π ∧
    0 < γ ∧ γ < π) ↔
  2 * c < k :=
by sorry


end NUMINAMATH_CALUDE_triangle_construction_valid_l1699_169902


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l1699_169997

theorem max_value_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  ∃ (M : ℝ), M = 12 ∧ ∀ z : ℂ, Complex.abs z = 2 →
    Complex.abs ((z - 2)^2 * (z + 2)) ≤ M ∧
    ∃ w₀ : ℂ, Complex.abs w₀ = 2 ∧ Complex.abs ((w₀ - 2)^2 * (w₀ + 2)) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l1699_169997


namespace NUMINAMATH_CALUDE_divisibility_implication_l1699_169972

theorem divisibility_implication (k : ℕ) : 
  (∃ k, 7^17 + 17 * 3 - 1 = 9 * k) → 
  (∃ m, 7^18 + 18 * 3 - 1 = 9 * m) := by
sorry

end NUMINAMATH_CALUDE_divisibility_implication_l1699_169972


namespace NUMINAMATH_CALUDE_cubic_root_sum_squared_l1699_169956

theorem cubic_root_sum_squared (p q r t : ℝ) : 
  (p + q + r = 8) →
  (p * q + p * r + q * r = 14) →
  (p * q * r = 2) →
  (t = Real.sqrt p + Real.sqrt q + Real.sqrt r) →
  (t^4 - 16*t^2 - 12*t = -8) := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squared_l1699_169956


namespace NUMINAMATH_CALUDE_max_values_on_unit_circle_l1699_169947

theorem max_values_on_unit_circle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2 = 1) :
  (∀ x y, x > 0 → y > 0 → x^2 + y^2 = 1 → a + b ≥ x + y) ∧
  (a + b ≤ Real.sqrt 2) ∧
  (∀ x y, x > 0 → y > 0 → x^2 + y^2 = 1 → a * b ≥ x * y) ∧
  (a * b ≤ 1/2) := by
sorry


end NUMINAMATH_CALUDE_max_values_on_unit_circle_l1699_169947


namespace NUMINAMATH_CALUDE_multiple_birth_statistics_l1699_169916

theorem multiple_birth_statistics (total_babies : ℕ) 
  (twins triplets quadruplets quintuplets : ℕ) : 
  total_babies = 1250 →
  quadruplets = 3 * quintuplets →
  triplets = 2 * quadruplets →
  twins = 2 * triplets →
  2 * twins + 3 * triplets + 4 * quadruplets + 5 * quintuplets = total_babies →
  5 * quintuplets = 6250 / 59 := by
  sorry

end NUMINAMATH_CALUDE_multiple_birth_statistics_l1699_169916


namespace NUMINAMATH_CALUDE_complex_square_root_l1699_169925

theorem complex_square_root : 
  ∃ (z₁ z₂ : ℂ), z₁^2 = -45 + 28*I ∧ z₂^2 = -45 + 28*I ∧ 
  z₁ = 2 + 7*I ∧ z₂ = -2 - 7*I ∧
  ∀ (z : ℂ), z^2 = -45 + 28*I → z = z₁ ∨ z = z₂ := by
sorry

end NUMINAMATH_CALUDE_complex_square_root_l1699_169925


namespace NUMINAMATH_CALUDE_system_solutions_l1699_169946

def has_solution (a : ℝ) : Prop :=
  ∃ (x y : ℝ), y > 0 ∧ x ≥ 0 ∧ y - 2 = a * (x - 4) ∧ 2 * x / (|y| + y) = Real.sqrt x

theorem system_solutions (a : ℝ) :
  (a ≤ 0 ∨ a = 1/4) →
    (has_solution a ∧
     ∃ (x y : ℝ), (x = 0 ∧ y = 2 - 4*a) ∨ (x = 4 ∧ y = 2)) ∧
  ((0 < a ∧ a < 1/4) ∨ (1/4 < a ∧ a < 1/2)) →
    (has_solution a ∧
     ∃ (x y : ℝ), (x = 0 ∧ y = 2 - 4*a) ∨ (x = 4 ∧ y = 2) ∨ (x = ((1-2*a)/a)^2 ∧ y = (1-2*a)/a)) ∧
  (a ≥ 1/2) →
    (has_solution a ∧
     ∃ (x y : ℝ), x = 4 ∧ y = 2) :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_l1699_169946


namespace NUMINAMATH_CALUDE_triangle_inequality_l1699_169960

/-- A complete graph K_n with n vertices, where each edge is colored either red, green, or blue. -/
structure ColoredCompleteGraph (n : ℕ) where
  n_ge_3 : n ≥ 3

/-- The number of triangles in K_n with all edges of the same color. -/
def monochromatic_triangles (G : ColoredCompleteGraph n) : ℕ := sorry

/-- The number of triangles in K_n with all edges of different colors. -/
def trichromatic_triangles (G : ColoredCompleteGraph n) : ℕ := sorry

/-- Theorem stating the relationship between monochromatic and trichromatic triangles. -/
theorem triangle_inequality (G : ColoredCompleteGraph n) :
  trichromatic_triangles G ≤ 2 * monochromatic_triangles G + n * (n - 1) / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1699_169960


namespace NUMINAMATH_CALUDE_chord_equation_l1699_169961

/-- Given a circle and a chord, prove the equation of the chord --/
theorem chord_equation (P Q : ℝ × ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 9 → (x - 1)^2 + (y - 2)^2 = (x - P.1)^2 + (y - P.2)^2) →
  (∀ (x y : ℝ), x^2 + y^2 = 9 → (x - 1)^2 + (y - 2)^2 = (x - Q.1)^2 + (y - Q.2)^2) →
  (P.1 + Q.1) / 2 = 1 →
  (P.2 + Q.2) / 2 = 2 →
  ∃ (k : ℝ), ∀ (x y : ℝ), (y - P.2) = k * (x - P.1) ∧ (y - Q.2) = k * (x - Q.1) →
    x + 2*y - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l1699_169961


namespace NUMINAMATH_CALUDE_abs_equation_solution_l1699_169981

theorem abs_equation_solution :
  ∃! y : ℝ, |y - 6| + 3*y = 12 :=
by
  -- The unique solution is y = 3
  use 3
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l1699_169981


namespace NUMINAMATH_CALUDE_never_equal_amounts_l1699_169917

/-- Represents the currencies in Dillie and Dallie -/
inductive Currency
| Diller
| Daller

/-- Represents the state of the financier's money -/
structure MoneyState :=
  (dillers : ℕ)
  (dallers : ℕ)

/-- Represents a currency exchange -/
inductive Exchange
| ToDallers
| ToDillers

/-- The exchange rate from dillers to dallers -/
def dillerToDallerRate : ℕ := 10

/-- The exchange rate from dallers to dillers -/
def dallerToDillerRate : ℕ := 10

/-- Perform a single exchange -/
def performExchange (state : MoneyState) (exchange : Exchange) : MoneyState :=
  match exchange with
  | Exchange.ToDallers => 
      { dillers := state.dillers / dillerToDallerRate,
        dallers := state.dallers + state.dillers * dillerToDallerRate }
  | Exchange.ToDillers => 
      { dillers := state.dillers + state.dallers * dallerToDillerRate,
        dallers := state.dallers / dallerToDillerRate }

/-- The initial state of the financier's money -/
def initialState : MoneyState := { dillers := 1, dallers := 0 }

/-- The main theorem to prove -/
theorem never_equal_amounts (exchanges : List Exchange) :
  let finalState := exchanges.foldl performExchange initialState
  finalState.dillers ≠ finalState.dallers :=
sorry

end NUMINAMATH_CALUDE_never_equal_amounts_l1699_169917


namespace NUMINAMATH_CALUDE_sqrt_six_equals_r_squared_minus_five_over_two_l1699_169968

theorem sqrt_six_equals_r_squared_minus_five_over_two :
  Real.sqrt 6 = ((Real.sqrt 2 + Real.sqrt 3)^2 - 5) / 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_six_equals_r_squared_minus_five_over_two_l1699_169968


namespace NUMINAMATH_CALUDE_parallelogram_segment_sum_l1699_169904

/-- A grid of equilateral triangles -/
structure TriangularGrid where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A parallelogram on the triangular grid -/
structure Parallelogram (grid : TriangularGrid) where
  vertices : Fin 4 → ℕ × ℕ  -- Grid coordinates of the vertices
  area : ℝ

/-- The possible sums of lengths of grid segments inside the parallelogram -/
def possible_segment_sums (grid : TriangularGrid) (p : Parallelogram grid) : Set ℝ :=
  {3, 4, 5, 6}

theorem parallelogram_segment_sum 
  (grid : TriangularGrid) 
  (p : Parallelogram grid) 
  (h_side_length : grid.side_length = 1) 
  (h_area : p.area = Real.sqrt 3) :
  ∃ (sum : ℝ), sum ∈ possible_segment_sums grid p :=
sorry

end NUMINAMATH_CALUDE_parallelogram_segment_sum_l1699_169904


namespace NUMINAMATH_CALUDE_emma_remaining_amount_l1699_169983

def calculate_remaining_amount (initial_amount furniture_cost fraction_given : ℚ) : ℚ :=
  let remaining_after_furniture := initial_amount - furniture_cost
  let amount_given := fraction_given * remaining_after_furniture
  remaining_after_furniture - amount_given

theorem emma_remaining_amount :
  calculate_remaining_amount 2000 400 (3/4) = 400 := by
  sorry

end NUMINAMATH_CALUDE_emma_remaining_amount_l1699_169983


namespace NUMINAMATH_CALUDE_intersection_A_B_l1699_169922

-- Define set A
def A : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (x + 1)}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1699_169922


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1699_169941

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + y = 3) :
  2^x + 2^y ≥ 4 * Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 3 ∧ 2^x₀ + 2^y₀ = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1699_169941


namespace NUMINAMATH_CALUDE_license_plate_count_l1699_169962

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The number of possible letters -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 5

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 3

/-- The number of possible positions for the letter block -/
def block_positions : ℕ := digits_in_plate + 1

/-- The total number of distinct license plates -/
def total_plates : ℕ := block_positions * (num_digits ^ digits_in_plate) * (num_letters ^ letters_in_plate)

theorem license_plate_count : total_plates = 105456000 := by sorry

end NUMINAMATH_CALUDE_license_plate_count_l1699_169962


namespace NUMINAMATH_CALUDE_jane_drawing_paper_l1699_169973

/-- The number of old, brown sheets of drawing paper Jane has -/
def brown_sheets : ℕ := 28

/-- The number of old, yellow sheets of drawing paper Jane has -/
def yellow_sheets : ℕ := 27

/-- The total number of drawing paper sheets Jane has -/
def total_sheets : ℕ := brown_sheets + yellow_sheets

theorem jane_drawing_paper : total_sheets = 55 := by
  sorry

end NUMINAMATH_CALUDE_jane_drawing_paper_l1699_169973


namespace NUMINAMATH_CALUDE_discount_comparison_l1699_169958

/-- The cost difference between Option 2 and Option 1 for buying suits and ties -/
def cost_difference (x : ℝ) : ℝ :=
  (3600 + 36*x) - (40*x + 3200)

theorem discount_comparison (x : ℝ) (h : x > 20) :
  cost_difference x ≥ 0 ∧ cost_difference 30 > 0 := by
  sorry

#eval cost_difference 30

end NUMINAMATH_CALUDE_discount_comparison_l1699_169958


namespace NUMINAMATH_CALUDE_indigo_restaurant_rating_l1699_169903

/-- Calculates the average star rating for a restaurant given the number of reviews for each star rating. -/
def averageStarRating (five_star : ℕ) (four_star : ℕ) (three_star : ℕ) (two_star : ℕ) : ℚ :=
  let total_stars := 5 * five_star + 4 * four_star + 3 * three_star + 2 * two_star
  let total_reviews := five_star + four_star + three_star + two_star
  (total_stars : ℚ) / total_reviews

/-- The average star rating for Indigo Restaurant is 4 stars. -/
theorem indigo_restaurant_rating :
  averageStarRating 6 7 4 1 = 4 := by
  sorry


end NUMINAMATH_CALUDE_indigo_restaurant_rating_l1699_169903


namespace NUMINAMATH_CALUDE_concert_hall_audience_l1699_169950

theorem concert_hall_audience (total_seats : ℕ) 
  (h_total : total_seats = 1260)
  (h_glasses : (7 : ℚ) / 18 * total_seats = number_with_glasses)
  (h_male_no_glasses : (6 : ℚ) / 11 * (total_seats - number_with_glasses) = number_male_no_glasses) :
  number_male_no_glasses = 420 := by
  sorry

end NUMINAMATH_CALUDE_concert_hall_audience_l1699_169950


namespace NUMINAMATH_CALUDE_number_in_bases_is_61_l1699_169964

/-- Represents a number in different bases -/
def NumberInBases (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    (0 ≤ a ∧ a < 6) ∧
    (0 ≤ b ∧ b < 6) ∧
    n = 36 * a + 6 * b + a ∧
    n = 15 * b + a

theorem number_in_bases_is_61 :
  ∃ (n : ℕ), NumberInBases n ∧ n = 61 :=
sorry

end NUMINAMATH_CALUDE_number_in_bases_is_61_l1699_169964


namespace NUMINAMATH_CALUDE_power_of_three_difference_l1699_169966

theorem power_of_three_difference : 3^(1+3+4) - (3^1 + 3^3 + 3^4) = 6450 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_difference_l1699_169966


namespace NUMINAMATH_CALUDE_consecutive_product_prime_power_and_perfect_power_l1699_169991

theorem consecutive_product_prime_power_and_perfect_power (m : ℕ) : m ≥ 1 → (
  (∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ m * (m + 1) = p ^ k) ↔ m = 1
) ∧ (
  ¬∃ (a k : ℕ), a ≥ 1 ∧ k ≥ 2 ∧ m * (m + 1) = a ^ k
) := by sorry

end NUMINAMATH_CALUDE_consecutive_product_prime_power_and_perfect_power_l1699_169991


namespace NUMINAMATH_CALUDE_pauls_coupon_percentage_l1699_169979

def initial_cost : ℝ := 350
def store_discount_percent : ℝ := 20
def final_price : ℝ := 252

theorem pauls_coupon_percentage :
  ∃ (coupon_percent : ℝ),
    final_price = initial_cost * (1 - store_discount_percent / 100) * (1 - coupon_percent / 100) ∧
    coupon_percent = 10 := by
  sorry

end NUMINAMATH_CALUDE_pauls_coupon_percentage_l1699_169979
