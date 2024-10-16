import Mathlib

namespace NUMINAMATH_CALUDE_division_problem_l2719_271977

theorem division_problem (x : ℝ) : (x / 0.08 = 800) → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2719_271977


namespace NUMINAMATH_CALUDE_centered_hexagonal_characterization_l2719_271928

/-- Definition of centered hexagonal number -/
def is_centered_hexagonal (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3 * k^2 - 3 * k + 1

/-- Definition of arithmetic sequence -/
def is_arithmetic_seq (x y z : ℕ) : Prop :=
  y - x = z - y

/-- Definition of geometric sequence -/
def is_geometric_seq (x y z : ℕ) : Prop :=
  x * z = y^2

theorem centered_hexagonal_characterization
  (a b c d : ℕ) 
  (h_arith : is_arithmetic_seq 1 a b)
  (h_geom : is_geometric_seq 1 c d)
  (h_sum : a + b = c + d) :
  is_centered_hexagonal a ↔ ∃ k : ℕ, a = 3 * k^2 - 3 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_centered_hexagonal_characterization_l2719_271928


namespace NUMINAMATH_CALUDE_a_card_is_one_three_l2719_271971

structure Card where
  n1 : Nat
  n2 : Nat
  deriving Repr

structure Person where
  name : String
  card : Card
  deriving Repr

def validCards : List Card := [⟨1, 2⟩, ⟨1, 3⟩, ⟨2, 3⟩]

def commonNumber (c1 c2 : Card) : Nat :=
  if c1.n1 = c2.n1 ∨ c1.n1 = c2.n2 then c1.n1
  else if c1.n2 = c2.n1 ∨ c1.n2 = c2.n2 then c1.n2
  else 0

theorem a_card_is_one_three 
  (a b c : Person)
  (h1 : a.card ∈ validCards ∧ b.card ∈ validCards ∧ c.card ∈ validCards)
  (h2 : a.card ≠ b.card ∧ b.card ≠ c.card ∧ a.card ≠ c.card)
  (h3 : commonNumber a.card b.card ≠ 2)
  (h4 : commonNumber b.card c.card ≠ 1)
  (h5 : c.card.n1 + c.card.n2 ≠ 5) :
  a.card = ⟨1, 3⟩ := by
sorry

end NUMINAMATH_CALUDE_a_card_is_one_three_l2719_271971


namespace NUMINAMATH_CALUDE_meeting_point_27_blocks_l2719_271926

/-- Two people walking around a circular loop -/
def CircularWalk (total_blocks : ℕ) (speed_ratio : ℚ) : Prop :=
  ∃ (meeting_point : ℚ),
    meeting_point > 0 ∧
    meeting_point < total_blocks ∧
    meeting_point = total_blocks / (1 + speed_ratio)

/-- Theorem: In a 27-block loop with a 3:1 speed ratio, the meeting point is at 27/4 blocks -/
theorem meeting_point_27_blocks :
  CircularWalk 27 3 → (27 : ℚ) / 4 = 27 / (1 + 3) :=
by
  sorry

#check meeting_point_27_blocks

end NUMINAMATH_CALUDE_meeting_point_27_blocks_l2719_271926


namespace NUMINAMATH_CALUDE_cube_cutting_l2719_271945

theorem cube_cutting (n s : ℕ) : n > s → n^3 - s^3 = 152 → n = 6 := by sorry

end NUMINAMATH_CALUDE_cube_cutting_l2719_271945


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l2719_271974

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def sides_per_die : ℕ := 6

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 1/2

/-- The number of ways to choose half the dice to show even numbers -/
def ways_to_choose_half : ℕ := Nat.choose num_dice (num_dice / 2)

/-- Theorem: The probability of rolling 8 six-sided dice and getting an equal number of even and odd results is 35/128 -/
theorem equal_even_odd_probability : 
  (ways_to_choose_half : ℚ) * prob_even^num_dice = 35/128 := by sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l2719_271974


namespace NUMINAMATH_CALUDE_copper_alloy_impossibility_l2719_271915

/-- Proves the impossibility of creating a specific copper alloy mixture --/
theorem copper_alloy_impossibility : ∀ (x : ℝ),
  0 ≤ x ∧ x ≤ 100 →
  32 * 0.25 + 8 * (x / 100) ≠ 40 * 0.45 :=
by
  sorry

#check copper_alloy_impossibility

end NUMINAMATH_CALUDE_copper_alloy_impossibility_l2719_271915


namespace NUMINAMATH_CALUDE_sqrt_square_abs_l2719_271916

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_l2719_271916


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2719_271960

theorem inequality_system_solution (x : ℝ) :
  (4 * x - 2 ≥ 3 * (x - 1)) ∧ ((x - 5) / 2 > x - 4) → -1 ≤ x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2719_271960


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l2719_271969

theorem point_on_terminal_side (m : ℝ) (α : ℝ) : 
  (3 : ℝ) / Real.sqrt ((3 : ℝ)^2 + m^2) = (3 : ℝ) / 5 → m = 4 ∨ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l2719_271969


namespace NUMINAMATH_CALUDE_sqrt_2023_irrational_not_perfect_square_2023_l2719_271967

theorem sqrt_2023_irrational : Irrational (Real.sqrt 2023) := by sorry

theorem not_perfect_square_2023 : ¬ ∃ n : ℕ, n ^ 2 = 2023 := by sorry

end NUMINAMATH_CALUDE_sqrt_2023_irrational_not_perfect_square_2023_l2719_271967


namespace NUMINAMATH_CALUDE_inequality_condition_l2719_271992

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ is increasing on [0, +∞) if f(x) ≤ f(y) for all 0 ≤ x ≤ y -/
def IsIncreasingNonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

/-- The main theorem stating the condition for the inequality to hold -/
theorem inequality_condition (f : ℝ → ℝ) (h_odd : IsOdd f) (h_incr : IsIncreasingNonneg f) :
  (∀ θ, f (Real.cos (2 * θ) - 3) + f (4 * m - 2 * m * Real.cos θ) > 0) ↔ m > 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l2719_271992


namespace NUMINAMATH_CALUDE_students_both_swim_and_gymnastics_l2719_271957

theorem students_both_swim_and_gymnastics 
  (total : ℕ) 
  (swim : ℕ) 
  (gymnastics : ℕ) 
  (neither : ℕ) 
  (h1 : total = 60) 
  (h2 : swim = 27) 
  (h3 : gymnastics = 28) 
  (h4 : neither = 15) : 
  total - ((total - swim) + (total - gymnastics) - neither) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_both_swim_and_gymnastics_l2719_271957


namespace NUMINAMATH_CALUDE_boys_in_line_l2719_271985

/-- If a boy in a single line is 19th from both ends, then the total number of boys is 37 -/
theorem boys_in_line (n : ℕ) (h : n > 0) : 
  (∃ k : ℕ, k > 0 ∧ k ≤ n ∧ k = 19 ∧ n - k + 1 = 19) → n = 37 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_line_l2719_271985


namespace NUMINAMATH_CALUDE_complex_z_magnitude_l2719_271924

theorem complex_z_magnitude (z : ℂ) (h : (1 + Complex.I)^2 * z = 1 - Complex.I^3) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_z_magnitude_l2719_271924


namespace NUMINAMATH_CALUDE_angle_opposite_geometric_mean_side_at_most_60_degrees_l2719_271964

/-- 
If in a triangle ABC, side a is the geometric mean of sides b and c,
then the angle A opposite to side a is less than or equal to 60°.
-/
theorem angle_opposite_geometric_mean_side_at_most_60_degrees 
  (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_geometric_mean : a^2 = b*c) : 
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))
  A ≤ π/3 := by sorry

end NUMINAMATH_CALUDE_angle_opposite_geometric_mean_side_at_most_60_degrees_l2719_271964


namespace NUMINAMATH_CALUDE_unique_solution_l2719_271906

def a : Fin 3 → ℝ := ![2, 2, 2]
def b : Fin 3 → ℝ := ![3, -2, 1]
def c : Fin 3 → ℝ := ![3, 3, -4]

def orthogonal (u v : Fin 3 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) + (u 2) * (v 2) = 0

theorem unique_solution :
  orthogonal a b ∧ orthogonal b c ∧ orthogonal a c →
  ∃! (p q r : ℝ), ∀ i : Fin 3,
    (![3, -1, 8] i) = p * (a i) + q * (b i) + r * (c i) ∧
    p = 5/3 ∧ q = 0 ∧ r = -10/17 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2719_271906


namespace NUMINAMATH_CALUDE_shirt_cost_difference_l2719_271995

/-- The difference in cost between two shirts -/
def cost_difference (total_cost first_shirt_cost : ℕ) : ℕ :=
  first_shirt_cost - (total_cost - first_shirt_cost)

/-- Proof that the difference in cost between two shirts is $6 -/
theorem shirt_cost_difference :
  let total_cost : ℕ := 24
  let first_shirt_cost : ℕ := 15
  first_shirt_cost > total_cost - first_shirt_cost →
  cost_difference total_cost first_shirt_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_difference_l2719_271995


namespace NUMINAMATH_CALUDE_g_minimum_value_l2719_271904

noncomputable def g (x : ℝ) : ℝ := x + x / (x^2 + 2) + x * (x + 5) / (x^2 + 3) + 3 * (x + 3) / (x * (x^2 + 3))

theorem g_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_g_minimum_value_l2719_271904


namespace NUMINAMATH_CALUDE_certain_number_proof_l2719_271914

theorem certain_number_proof (N : ℕ) (h1 : N < 81) 
  (h2 : ∀ k : ℕ, k ∈ Finset.range 15 → N + k + 1 < 81) 
  (h3 : N + 16 ≥ 81) : N = 65 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2719_271914


namespace NUMINAMATH_CALUDE_max_value_x_2y_plus_1_l2719_271955

theorem max_value_x_2y_plus_1 (x y : ℝ) 
  (hx : |x - 1| ≤ 1) 
  (hy : |y - 2| ≤ 1) : 
  ∃ (M : ℝ), M = 5 ∧ 
  (∀ z, |x - 2*y + 1| ≤ z ↔ M ≤ z) :=
sorry

end NUMINAMATH_CALUDE_max_value_x_2y_plus_1_l2719_271955


namespace NUMINAMATH_CALUDE_set_operations_and_range_l2719_271997

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a - 1}

-- State the theorem
theorem set_operations_and_range :
  (A ∩ B = {x : ℝ | 3 < x ∧ x < 6}) ∧
  ((Set.univ \ A) ∪ (Set.univ \ B) = {x : ℝ | x ≤ 3 ∨ x ≥ 6}) ∧
  (∀ a : ℝ, (B ∪ C a = B) ↔ (a ≤ 1 ∨ (2 ≤ a ∧ a ≤ 5))) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l2719_271997


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2719_271907

-- Problem 1
theorem problem_1 (a b : ℝ) (h1 : a > b) (h2 : a^2 + b^2 = 13) (h3 : a * b = 6) :
  a - b = 1 := by sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : a^2 + b^2 = c^2)  -- Pythagorean theorem for right triangle
  (h3 : a^2 + b^2 + 25 = 6*a + 8*b) :
  a + b + c = 12 ∨ a + b + c = 7 + Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2719_271907


namespace NUMINAMATH_CALUDE_charles_chocolate_syrup_l2719_271986

/-- The amount of chocolate milk in each glass (in ounces) -/
def glass_size : ℝ := 8

/-- The amount of milk in each glass (in ounces) -/
def milk_per_glass : ℝ := 6.5

/-- The amount of chocolate syrup in each glass (in ounces) -/
def syrup_per_glass : ℝ := 1.5

/-- The total amount of milk Charles has (in ounces) -/
def total_milk : ℝ := 130

/-- The total amount of chocolate milk Charles will drink (in ounces) -/
def total_drink : ℝ := 160

/-- The theorem stating the amount of chocolate syrup Charles has -/
theorem charles_chocolate_syrup : 
  ∃ (syrup : ℝ), 
    (total_drink / glass_size) * syrup_per_glass = syrup ∧ 
    syrup = 30 := by sorry

end NUMINAMATH_CALUDE_charles_chocolate_syrup_l2719_271986


namespace NUMINAMATH_CALUDE_circle_area_difference_l2719_271939

/-- Given two circles where the smaller circle has radius 4 and the center of the larger circle
    is on the circumference of the smaller circle, the difference in areas between the larger
    and smaller circles is 48π. -/
theorem circle_area_difference (r : ℝ) (h : r = 4) : 
  π * (2 * r)^2 - π * r^2 = 48 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l2719_271939


namespace NUMINAMATH_CALUDE_min_value_a1a3_l2719_271937

theorem min_value_a1a3 (a₁ a₂ a₃ : ℝ) (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (h_a₂ : a₂ = 6) 
  (h_arithmetic : ∃ d : ℝ, 1 / (a₃ + 3) - 1 / (a₂ + 2) = 1 / (a₂ + 2) - 1 / (a₁ + 1)) :
  a₁ * a₃ ≥ 16 * Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a1a3_l2719_271937


namespace NUMINAMATH_CALUDE_favorite_subject_fraction_l2719_271943

theorem favorite_subject_fraction (total_students : ℕ) 
  (math_fraction : ℚ) (english_fraction : ℚ) (no_favorite : ℕ) : 
  total_students = 30 →
  math_fraction = 1 / 5 →
  english_fraction = 1 / 3 →
  no_favorite = 12 →
  let math_students := total_students * math_fraction
  let english_students := total_students * english_fraction
  let students_with_favorite := total_students - no_favorite
  let science_students := students_with_favorite - math_students - english_students
  let remaining_students := students_with_favorite - math_students - english_students
  science_students / remaining_students = 1 := by
sorry

end NUMINAMATH_CALUDE_favorite_subject_fraction_l2719_271943


namespace NUMINAMATH_CALUDE_current_population_calculation_l2719_271934

def initial_population : ℕ := 4399
def bombardment_percentage : ℚ := 1/10
def fear_percentage : ℚ := 1/5

theorem current_population_calculation :
  let remaining_after_bombardment := initial_population - ⌊initial_population * bombardment_percentage⌋
  let current_population := remaining_after_bombardment - ⌊remaining_after_bombardment * fear_percentage⌋
  current_population = 3167 := by sorry

end NUMINAMATH_CALUDE_current_population_calculation_l2719_271934


namespace NUMINAMATH_CALUDE_f_min_at_4_l2719_271936

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 19

/-- Theorem stating that f attains its minimum at x = 4 -/
theorem f_min_at_4 : ∀ x : ℝ, f x ≥ f 4 := by sorry

end NUMINAMATH_CALUDE_f_min_at_4_l2719_271936


namespace NUMINAMATH_CALUDE_andrew_vacation_days_in_march_l2719_271919

/-- Calculates the number of vacation days taken in March given the conditions of Andrew's work and vacation schedule. -/
def vacation_days_in_march (days_worked : ℕ) (days_per_vacation : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_vacation_days := days_worked / days_per_vacation
  let used_vacation_days := total_vacation_days - remaining_days
  used_vacation_days / 3

theorem andrew_vacation_days_in_march :
  vacation_days_in_march 300 10 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_andrew_vacation_days_in_march_l2719_271919


namespace NUMINAMATH_CALUDE_special_ring_classification_l2719_271909

universe u

/-- A ring satisfying the given property -/
class SpecialRing (A : Type u) extends Ring A where
  special_property : ∀ (x : A), x ≠ 0 → x^(2^n + 1) = 1
  n : ℕ
  n_pos : n ≥ 1

/-- The theorem stating that any SpecialRing is isomorphic to F₂ or F₄ -/
theorem special_ring_classification (A : Type u) [SpecialRing A] :
  (∃ (f : A ≃+* Fin 2), Function.Bijective f) ∨
  (∃ (g : A ≃+* Fin 4), Function.Bijective g) :=
sorry

end NUMINAMATH_CALUDE_special_ring_classification_l2719_271909


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l2719_271976

-- Define the hyperbola C
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the line l
def line (t : ℝ) (x y : ℝ) : Prop := x = t * y + 2

-- Define the condition for the circle with diameter MN passing through A(2,-2)
def circle_condition (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 - 2) * (x2 - 2) + (y1 + 2) * (y2 + 2) = 0

theorem hyperbola_line_intersection :
  (hyperbola 3 4) →
  (hyperbola (Real.sqrt 2) (Real.sqrt 2)) →
  ∀ t : ℝ,
    (∃ x1 y1 x2 y2 : ℝ,
      x1 ≠ x2 ∧
      hyperbola x1 y1 ∧
      hyperbola x2 y2 ∧
      line t x1 y1 ∧
      line t x2 y2 ∧
      circle_condition x1 y1 x2 y2) →
    (t = 1 ∨ t = 1/7) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l2719_271976


namespace NUMINAMATH_CALUDE_angle_from_elevation_and_depression_l2719_271923

/-- Given an elevation angle and a depression angle observed from a point,
    calculate the angle between the two observed points and the observing point. -/
theorem angle_from_elevation_and_depression
  (elevation_angle : ℝ) (depression_angle : ℝ) :
  elevation_angle = 60 →
  depression_angle = 70 →
  elevation_angle + depression_angle = 130 :=
by sorry

end NUMINAMATH_CALUDE_angle_from_elevation_and_depression_l2719_271923


namespace NUMINAMATH_CALUDE_phil_quarters_proof_l2719_271900

/-- Represents the number of quarters Phil collected every third month in the third year -/
def quarters_collected_third_year : ℕ := sorry

/-- The initial number of quarters Phil had -/
def initial_quarters : ℕ := 50

/-- The number of quarters Phil had after the first year -/
def quarters_after_first_year : ℕ := 2 * initial_quarters

/-- The number of quarters Phil collected in the second year -/
def quarters_collected_second_year : ℕ := 3 * 12

/-- The number of quarters Phil had after the second year -/
def quarters_after_second_year : ℕ := quarters_after_first_year + quarters_collected_second_year

/-- The number of quarters Phil had after the third year -/
def quarters_after_third_year : ℕ := quarters_after_second_year + 4 * quarters_collected_third_year

/-- The number of quarters Phil had after losing some in the fourth year -/
def final_quarters : ℕ := 105

theorem phil_quarters_proof :
  (3 * (quarters_after_third_year : ℚ)) / 4 = final_quarters :=
sorry

end NUMINAMATH_CALUDE_phil_quarters_proof_l2719_271900


namespace NUMINAMATH_CALUDE_remainder_3973_div_28_l2719_271979

theorem remainder_3973_div_28 : 3973 % 28 = 9 := by sorry

end NUMINAMATH_CALUDE_remainder_3973_div_28_l2719_271979


namespace NUMINAMATH_CALUDE_alexander_payment_l2719_271961

/-- The cost of tickets at an amusement park -/
def ticket_cost (child_cost adult_cost : ℕ) (alexander_child alexander_adult anna_child anna_adult : ℕ) : Prop :=
  let alexander_total := child_cost * alexander_child + adult_cost * alexander_adult
  let anna_total := child_cost * anna_child + adult_cost * anna_adult
  (child_cost = 600) ∧
  (alexander_child = 2) ∧
  (alexander_adult = 3) ∧
  (anna_child = 3) ∧
  (anna_adult = 2) ∧
  (alexander_total = anna_total + 200)

theorem alexander_payment :
  ∀ (child_cost adult_cost : ℕ),
  ticket_cost child_cost adult_cost 2 3 3 2 →
  child_cost * 2 + adult_cost * 3 = 3600 :=
by
  sorry

end NUMINAMATH_CALUDE_alexander_payment_l2719_271961


namespace NUMINAMATH_CALUDE_cos_half_alpha_l2719_271910

theorem cos_half_alpha (α : Real) 
  (h1 : 25 * (Real.sin α)^2 + Real.sin α - 24 = 0)
  (h2 : π / 2 < α ∧ α < π) :
  Real.cos (α / 2) = 3 / 5 ∨ Real.cos (α / 2) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_half_alpha_l2719_271910


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2719_271925

def polynomial (a₂ a₁ : ℤ) (x : ℤ) : ℤ := x^3 + a₂*x^2 + a₁*x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  ∀ r : ℤ, polynomial a₂ a₁ r = 0 → r ∈ possible_roots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2719_271925


namespace NUMINAMATH_CALUDE_distance_traveled_l2719_271981

/-- Given a speed of 25 km/hr and a time of 5 hr, the distance traveled is 125 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 25) 
  (h2 : time = 5) 
  (h3 : distance = speed * time) : 
  distance = 125 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l2719_271981


namespace NUMINAMATH_CALUDE_set_operations_l2719_271947

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

theorem set_operations :
  (A ∩ B = {x | 1 ≤ x ∧ x ≤ 4}) ∧
  ((Set.univ : Set ℝ) \ (A ∪ B) = {x | x < -1 ∨ x > 5}) ∧
  (((Set.univ : Set ℝ) \ A) ∪ ((Set.univ : Set ℝ) \ B) = {x | x < 1 ∨ x > 4}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2719_271947


namespace NUMINAMATH_CALUDE_mike_seashells_count_l2719_271998

/-- The number of seashells Mike found initially -/
def initial_seashells : ℝ := 6.0

/-- The number of seashells Mike found later -/
def later_seashells : ℝ := 4.0

/-- The total number of seashells Mike found -/
def total_seashells : ℝ := initial_seashells + later_seashells

theorem mike_seashells_count : total_seashells = 10.0 := by sorry

end NUMINAMATH_CALUDE_mike_seashells_count_l2719_271998


namespace NUMINAMATH_CALUDE_total_cans_collected_l2719_271983

/-- The number of cans collected by LaDonna -/
def ladonna_cans : ℕ := 25

/-- The number of cans collected by Prikya -/
def prikya_cans : ℕ := 2 * ladonna_cans

/-- The number of cans collected by Yoki -/
def yoki_cans : ℕ := 10

/-- The total number of cans collected -/
def total_cans : ℕ := ladonna_cans + prikya_cans + yoki_cans

theorem total_cans_collected : total_cans = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_collected_l2719_271983


namespace NUMINAMATH_CALUDE_fraction_of_one_third_is_one_eighth_l2719_271946

theorem fraction_of_one_third_is_one_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_one_third_is_one_eighth_l2719_271946


namespace NUMINAMATH_CALUDE_first_company_base_rate_l2719_271901

/-- The base rate of the first telephone company -/
def base_rate_1 : ℝ := 7

/-- The per-minute rate of the first telephone company -/
def rate_1 : ℝ := 0.25

/-- The base rate of the second telephone company -/
def base_rate_2 : ℝ := 12

/-- The per-minute rate of the second telephone company -/
def rate_2 : ℝ := 0.20

/-- The number of minutes for which the bills are equal -/
def minutes : ℝ := 100

theorem first_company_base_rate :
  base_rate_1 + rate_1 * minutes = base_rate_2 + rate_2 * minutes →
  base_rate_1 = 7 := by
sorry

end NUMINAMATH_CALUDE_first_company_base_rate_l2719_271901


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2719_271948

theorem lcm_hcf_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 83 → 
  a = 210 → 
  b = 913 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2719_271948


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2719_271950

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ a < -1 ∨ a > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2719_271950


namespace NUMINAMATH_CALUDE_vincent_songs_l2719_271959

/-- The number of songs Vincent knows after summer camp -/
def total_songs (initial_songs : ℕ) (new_songs : ℕ) : ℕ :=
  initial_songs + new_songs

/-- Theorem stating that Vincent knows 74 songs after summer camp -/
theorem vincent_songs : total_songs 56 18 = 74 := by
  sorry

end NUMINAMATH_CALUDE_vincent_songs_l2719_271959


namespace NUMINAMATH_CALUDE_queen_mary_legs_l2719_271968

/-- The total number of legs on the Queen Mary II -/
def total_legs : ℕ := 41

/-- The total number of heads on the ship -/
def total_heads : ℕ := 14

/-- The number of cats on the ship -/
def num_cats : ℕ := 7

/-- The number of legs a cat has -/
def cat_legs : ℕ := 4

/-- The number of legs a normal human has -/
def human_legs : ℕ := 2

/-- The number of legs the captain has -/
def captain_legs : ℕ := 1

/-- Theorem stating the total number of legs on the ship -/
theorem queen_mary_legs : 
  total_legs = 
    (num_cats * cat_legs) + 
    ((total_heads - num_cats - 1) * human_legs) + 
    captain_legs :=
by sorry

end NUMINAMATH_CALUDE_queen_mary_legs_l2719_271968


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l2719_271989

theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) : 
  small_side = 2 → large_side = 12 → 
  (large_side / small_side) ^ 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l2719_271989


namespace NUMINAMATH_CALUDE_sqrt_a_plus_b_equals_two_l2719_271929

theorem sqrt_a_plus_b_equals_two (a b : ℝ) (h : |a - 1| + (b - 3)^2 = 0) : 
  Real.sqrt (a + b) = 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_b_equals_two_l2719_271929


namespace NUMINAMATH_CALUDE_fraction_equality_l2719_271991

theorem fraction_equality (a b : ℝ) (h : a / (a + 2 * b) = 3 / 5) : a / b = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2719_271991


namespace NUMINAMATH_CALUDE_complex_sum_equality_l2719_271973

/-- Given complex numbers a and b, prove that 2a + 3b = 1 + i -/
theorem complex_sum_equality (a b : ℂ) (ha : a = 2 - I) (hb : b = -1 + I) :
  2 * a + 3 * b = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l2719_271973


namespace NUMINAMATH_CALUDE_sum_after_2015_iterations_l2719_271975

/-- The process of adding digits and appending the sum -/
def process (n : ℕ) : ℕ := sorry

/-- The result of applying the process n times to the initial number -/
def iterate_process (initial : ℕ) (n : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_after_2015_iterations :
  sum_of_digits (iterate_process 2015 2015) = 8065 := by sorry

end NUMINAMATH_CALUDE_sum_after_2015_iterations_l2719_271975


namespace NUMINAMATH_CALUDE_calculation_proof_l2719_271949

theorem calculation_proof : (-3 : ℚ) * 6 / (-2) * (1/2) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2719_271949


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l2719_271902

/-- 
Given a quadratic equation (m-1)x^2 - 2mx + m + 3 = 0,
prove that it has two real roots if and only if m ≤ 3/2 and m ≠ 1.
-/
theorem quadratic_two_real_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    (m - 1) * x^2 - 2 * m * x + m + 3 = 0 ∧ 
    (m - 1) * y^2 - 2 * m * y + m + 3 = 0) ↔ 
  (m ≤ 3/2 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l2719_271902


namespace NUMINAMATH_CALUDE_sqrt_x_minus_8_real_l2719_271917

theorem sqrt_x_minus_8_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 8) ↔ x ≥ 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_8_real_l2719_271917


namespace NUMINAMATH_CALUDE_baseball_cards_count_l2719_271918

theorem baseball_cards_count (num_friends : ℕ) (cards_per_friend : ℕ) : 
  num_friends = 5 → cards_per_friend = 91 → num_friends * cards_per_friend = 455 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_count_l2719_271918


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2719_271933

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (9 - 5 * x) = 8 → x = -11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2719_271933


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l2719_271931

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 62 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 62 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l2719_271931


namespace NUMINAMATH_CALUDE_q_range_l2719_271993

def q (x : ℝ) : ℝ := (x^2 - 2)^2

theorem q_range : 
  ∀ y : ℝ, (∃ x : ℝ, q x = y) ↔ y ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_q_range_l2719_271993


namespace NUMINAMATH_CALUDE_is_reflection_l2719_271980

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, b],
    ![3/7, -4/7]]

theorem is_reflection : 
  let R := reflection_matrix (4/7) (-16/21)
  R * R = 1 :=
sorry

end NUMINAMATH_CALUDE_is_reflection_l2719_271980


namespace NUMINAMATH_CALUDE_statistical_properties_l2719_271927

def data1 : List ℝ := [3, 5, 7, 9]
def data2 : List ℝ := [6, 10, 14, 18]
def data3 : List ℝ := [4, 6, 7, 7, 9, 4]

def standardDeviation (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

theorem statistical_properties :
  (standardDeviation data1 = (1/2) * standardDeviation data2) ∧
  (median data3 = 6.5) := by
  sorry

end NUMINAMATH_CALUDE_statistical_properties_l2719_271927


namespace NUMINAMATH_CALUDE_not_always_geometric_b_l2719_271988

/-- A sequence is geometric if there exists a common ratio q such that a(n+1) = q * a(n) for all n -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Definition of the sequence b_n in terms of a_n -/
def B (a : ℕ → ℝ) (n : ℕ) : ℝ := a (2*n - 1) + a (2*n)

theorem not_always_geometric_b (a : ℕ → ℝ) :
  IsGeometric a → ¬ (∀ a : ℕ → ℝ, IsGeometric a → IsGeometric (B a)) :=
by
  sorry

end NUMINAMATH_CALUDE_not_always_geometric_b_l2719_271988


namespace NUMINAMATH_CALUDE_calories_burned_l2719_271987

/-- The number of times players run up and down the bleachers -/
def num_runs : ℕ := 40

/-- The number of stairs in one direction -/
def stairs_one_way : ℕ := 32

/-- The number of calories burned per stair -/
def calories_per_stair : ℕ := 2

/-- The total number of calories burned during the exercise -/
def total_calories : ℕ := num_runs * (2 * stairs_one_way) * calories_per_stair

theorem calories_burned :
  total_calories = 5120 :=
by sorry

end NUMINAMATH_CALUDE_calories_burned_l2719_271987


namespace NUMINAMATH_CALUDE_intersection_point_d_l2719_271932

/-- A function g(x) = 2x + c with c being an integer -/
def g (c : ℤ) : ℝ → ℝ := λ x ↦ 2 * x + c

/-- The inverse function of g -/
noncomputable def g_inv (c : ℤ) : ℝ → ℝ := λ x ↦ (x - c) / 2

theorem intersection_point_d (c : ℤ) (d : ℤ) :
  g c (-4) = d ∧ g_inv c (-4) = d → d = -4 := by sorry

end NUMINAMATH_CALUDE_intersection_point_d_l2719_271932


namespace NUMINAMATH_CALUDE_clock_painting_theorem_clock_painting_57_clock_painting_1913_l2719_271958

/-- The number of distinct painted numbers on a clock face -/
def distinctPaintedNumbers (clockHours : ℕ) (paintInterval : ℕ) : ℕ :=
  clockHours / Nat.gcd clockHours paintInterval

theorem clock_painting_theorem (clockHours : ℕ) (paintInterval : ℕ) 
  (h : clockHours > 0) : 
  distinctPaintedNumbers clockHours paintInterval = 
  clockHours / Nat.gcd clockHours paintInterval := by
  sorry

theorem clock_painting_57 : 
  distinctPaintedNumbers 12 57 = 4 := by
  sorry

theorem clock_painting_1913 : 
  distinctPaintedNumbers 12 1913 = 12 := by
  sorry

end NUMINAMATH_CALUDE_clock_painting_theorem_clock_painting_57_clock_painting_1913_l2719_271958


namespace NUMINAMATH_CALUDE_ending_number_proof_l2719_271911

theorem ending_number_proof (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k) →  -- n is divisible by 7
  n ≥ 21 →               -- n is at least 21 (first number after 18 divisible by 7)
  (21 + n) / 2 = 77/2 →  -- average of arithmetic sequence is 38.5
  n = 56 := by
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l2719_271911


namespace NUMINAMATH_CALUDE_total_candies_l2719_271941

/-- The number of candies in each gift box -/
def candies_per_box : ℕ := 156

/-- The number of children receiving gift boxes -/
def num_children : ℕ := 20

/-- Theorem: The total number of candies needed is 3120 -/
theorem total_candies : candies_per_box * num_children = 3120 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l2719_271941


namespace NUMINAMATH_CALUDE_sin_210_degrees_l2719_271953

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l2719_271953


namespace NUMINAMATH_CALUDE_order_of_special_roots_l2719_271990

theorem order_of_special_roots : ∃ (a b c : ℝ), 
  a = (2 : ℝ) ^ (1/2) ∧ 
  b = Real.exp (1/Real.exp 1) ∧ 
  c = (3 : ℝ) ^ (1/3) ∧ 
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_order_of_special_roots_l2719_271990


namespace NUMINAMATH_CALUDE_total_candles_l2719_271962

theorem total_candles (bedroom : ℕ) (living_room : ℕ) (donovan : ℕ) : 
  bedroom = 20 →
  bedroom = 2 * living_room →
  donovan = 20 →
  bedroom + living_room + donovan = 50 := by
sorry

end NUMINAMATH_CALUDE_total_candles_l2719_271962


namespace NUMINAMATH_CALUDE_anands_age_l2719_271922

theorem anands_age (anand_age bala_age : ℕ) : 
  (anand_age - 10 = (bala_age - 10) / 3) →
  (bala_age = anand_age + 10) →
  anand_age = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_anands_age_l2719_271922


namespace NUMINAMATH_CALUDE_triangle_area_l2719_271965

/-- The area of a triangle with sides 5, 5, and 6 units is 12 square units. -/
theorem triangle_area : ∀ (a b c : ℝ), a = 5 ∧ b = 5 ∧ c = 6 →
  ∃ (s : ℝ), s = (a + b + c) / 2 ∧ Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2719_271965


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2719_271942

/-- Given two vectors a = (-1, 2) and b = (m, 3) where m ∈ ℝ, 
    if a ⊥ b, then m = 6 -/
theorem perpendicular_vectors_m_value :
  ∀ (m : ℝ), 
  let a : Fin 2 → ℝ := ![(-1), 2]
  let b : Fin 2 → ℝ := ![m, 3]
  (∀ (i j : Fin 2), i ≠ j → a i * b j = a j * b i) →
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2719_271942


namespace NUMINAMATH_CALUDE_triangle_area_specific_l2719_271982

/-- The area of a triangle with two sides of length 31 and one side of length 40 is 474 -/
theorem triangle_area_specific : ∃ (A : ℝ), 
  A = (Real.sqrt (51 * (51 - 31) * (51 - 31) * (51 - 40)) : ℝ) ∧ A = 474 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_specific_l2719_271982


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l2719_271940

def binary_to_nat (s : String) : Nat :=
  s.foldl (fun acc c => 2 * acc + c.toNat - '0'.toNat) 0

theorem binary_arithmetic_equality : 
  let a := binary_to_nat "1011101"
  let b := binary_to_nat "1101"
  let c := binary_to_nat "101010"
  let d := binary_to_nat "110"
  let result := binary_to_nat "1110111100"
  ((a + b) * c) / d = result := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l2719_271940


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l2719_271978

theorem two_digit_number_puzzle :
  ∀ (x : ℕ),
  x < 10 →
  let original := 21 * x
  let reversed := 12 * x
  original < 100 →
  original - reversed = 27 →
  original = 63 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l2719_271978


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2719_271912

theorem smallest_integer_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  use 400
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2719_271912


namespace NUMINAMATH_CALUDE_smallest_integer_above_sum_of_roots_l2719_271996

theorem smallest_integer_above_sum_of_roots : ∃ n : ℕ, n = 2703 ∧ 
  (∀ m : ℕ, (m : ℝ) > (Real.sqrt 4 + Real.sqrt 3)^6 → m ≥ n) ∧
  ((n : ℝ) - 1 ≤ (Real.sqrt 4 + Real.sqrt 3)^6) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_above_sum_of_roots_l2719_271996


namespace NUMINAMATH_CALUDE_greatest_power_of_seven_l2719_271935

def r : ℕ := (List.range 50).foldl (· * ·) 1

theorem greatest_power_of_seven (k : ℕ) : k ≤ 8 ↔ (7^k : ℕ) ∣ r :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_seven_l2719_271935


namespace NUMINAMATH_CALUDE_fraction_of_sum_l2719_271966

theorem fraction_of_sum (m n : ℝ) (a b c : ℝ) 
  (h1 : a = (b + c) / m)
  (h2 : b = (c + a) / n)
  (h3 : m ≠ 0)
  (h4 : n ≠ 0) :
  (m * n ≠ 1 → c / (a + b) = (m * n - 1) / (m + n + 2)) ∧
  (m = -1 ∧ n = -1 → c / (a + b) = -1) :=
sorry

end NUMINAMATH_CALUDE_fraction_of_sum_l2719_271966


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l2719_271921

theorem company_picnic_attendance
  (total_employees : ℕ)
  (men_percentage : ℝ)
  (men_attendance_rate : ℝ)
  (women_attendance_rate : ℝ)
  (h1 : men_percentage = 0.55)
  (h2 : men_attendance_rate = 0.2)
  (h3 : women_attendance_rate = 0.4) :
  let women_percentage : ℝ := 1 - men_percentage
  let men_count : ℝ := men_percentage * total_employees
  let women_count : ℝ := women_percentage * total_employees
  let men_attended : ℝ := men_attendance_rate * men_count
  let women_attended : ℝ := women_attendance_rate * women_count
  let total_attended : ℝ := men_attended + women_attended
  total_attended / total_employees = 0.29 := by
sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l2719_271921


namespace NUMINAMATH_CALUDE_distance_between_points_l2719_271905

def point1 : ℝ × ℝ := (3, 7)
def point2 : ℝ × ℝ := (3, -4)

theorem distance_between_points : 
  |point1.2 - point2.2| = 11 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2719_271905


namespace NUMINAMATH_CALUDE_height_prediction_at_10_l2719_271956

/-- Represents a linear regression model for height vs age -/
structure HeightModel where
  slope : ℝ
  intercept : ℝ

/-- Calculates the predicted height for a given age using the model -/
def predictHeight (model : HeightModel) (age : ℝ) : ℝ :=
  model.slope * age + model.intercept

/-- Defines what it means for a prediction to be "around" a value -/
def isAround (predicted : ℝ) (target : ℝ) (tolerance : ℝ) : Prop :=
  abs (predicted - target) ≤ tolerance

theorem height_prediction_at_10 (model : HeightModel) 
  (h1 : model.slope = 7.19) 
  (h2 : model.intercept = 73.93) : 
  ∃ (tolerance : ℝ), tolerance > 0 ∧ isAround (predictHeight model 10) 145.83 tolerance :=
sorry

end NUMINAMATH_CALUDE_height_prediction_at_10_l2719_271956


namespace NUMINAMATH_CALUDE_check_mistake_l2719_271903

theorem check_mistake (x y : ℕ) : 
  (100 * y + x) - (100 * x + y) = 1368 → y = x + 14 := by
  sorry

end NUMINAMATH_CALUDE_check_mistake_l2719_271903


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2719_271930

theorem sum_of_a_and_b (a b : ℝ) : 
  ({a, a^2} : Set ℝ) = ({1, b} : Set ℝ) → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2719_271930


namespace NUMINAMATH_CALUDE_coincide_points_l2719_271999

/-- A point on the coordinate plane with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- A vector between two integer points -/
def vector (a b : IntPoint) : IntPoint :=
  ⟨b.x - a.x, b.y - a.y⟩

/-- Move a point by a vector -/
def movePoint (p v : IntPoint) : IntPoint :=
  ⟨p.x + v.x, p.y + v.y⟩

/-- The main theorem stating that any two points can be made to coincide -/
theorem coincide_points (a b c d : IntPoint) :
  ∃ (moves : List (IntPoint → IntPoint)),
    ∃ (p q : IntPoint),
      (p ∈ [a, b, c, d]) ∧
      (q ∈ [a, b, c, d]) ∧
      (p ≠ q) ∧
      (moves.foldl (λ acc f => f acc) p = moves.foldl (λ acc f => f acc) q) :=
by sorry

end NUMINAMATH_CALUDE_coincide_points_l2719_271999


namespace NUMINAMATH_CALUDE_max_cute_pairs_is_43_l2719_271920

/-- A pair of ages (a, b) is cute if each person is at least seven years older than half the age of the other person. -/
def is_cute_pair (a b : ℕ) : Prop :=
  a ≥ b / 2 + 7 ∧ b ≥ a / 2 + 7

/-- The set of ages from 1 to 100. -/
def age_set : Finset ℕ :=
  Finset.range 100

/-- A function that returns the maximum number of pairwise disjoint cute pairs that can be formed from a set of ages. -/
def max_disjoint_cute_pairs (ages : Finset ℕ) : ℕ :=
  sorry

theorem max_cute_pairs_is_43 :
  max_disjoint_cute_pairs age_set = 43 :=
sorry

end NUMINAMATH_CALUDE_max_cute_pairs_is_43_l2719_271920


namespace NUMINAMATH_CALUDE_sin_2x_eq_cos_x_div_2_solutions_l2719_271944

theorem sin_2x_eq_cos_x_div_2_solutions (x : ℝ) : 
  ∃! (s : Finset ℝ), s.card = 4 ∧ (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧
    (∀ x ∈ s, Real.sin (2 * x) = Real.cos (x / 2)) ∧
    (∀ y, 0 ≤ y ∧ y ≤ 2 * Real.pi ∧ Real.sin (2 * y) = Real.cos (y / 2) → y ∈ s) :=
sorry

end NUMINAMATH_CALUDE_sin_2x_eq_cos_x_div_2_solutions_l2719_271944


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2719_271913

theorem complex_fraction_simplification :
  (3 / 7 + 5 / 8) / (5 / 12 + 7 / 15) = 15 / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2719_271913


namespace NUMINAMATH_CALUDE_max_gcd_15n_plus_4_8n_plus_1_l2719_271951

theorem max_gcd_15n_plus_4_8n_plus_1 :
  ∃ (k : ℕ), k > 0 ∧ Nat.gcd (15 * k + 4) (8 * k + 1) = 17 ∧
  ∀ (n : ℕ), n > 0 → Nat.gcd (15 * n + 4) (8 * n + 1) ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_15n_plus_4_8n_plus_1_l2719_271951


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2719_271952

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : isArithmeticSequence a)
  (h_fifth : a 5 = 10)
  (h_sum : a 1 + a 2 + a 3 = 3) :
  a 1 = -2 ∧ ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
  sorry

#check arithmetic_sequence_property

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2719_271952


namespace NUMINAMATH_CALUDE_string_folding_l2719_271954

theorem string_folding (initial_length : ℝ) (folded_twice : ℕ) : 
  initial_length = 12 ∧ folded_twice = 2 → initial_length / (2^folded_twice) = 3 := by
  sorry

end NUMINAMATH_CALUDE_string_folding_l2719_271954


namespace NUMINAMATH_CALUDE_water_amount_for_scaled_solution_l2719_271972

theorem water_amount_for_scaled_solution 
  (chemical_a : Real) 
  (water : Real) 
  (total : Real) 
  (new_total : Real) 
  (h1 : chemical_a + water = total)
  (h2 : chemical_a = 0.07)
  (h3 : water = 0.03)
  (h4 : total = 0.1)
  (h5 : new_total = 0.6) : 
  (water / total) * new_total = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_for_scaled_solution_l2719_271972


namespace NUMINAMATH_CALUDE_adjacent_chair_subsets_l2719_271970

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- The function that calculates the number of subsets with at least three adjacent chairs -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
  -- Subsets with exactly 3, 4, 5, or 6 adjacent chairs
  4 * n +
  -- Subsets with 7 or more chairs (always contain at least 3 adjacent)
  (Nat.choose n 7) + (Nat.choose n 8) + (Nat.choose n 9) +
  (Nat.choose n 10) + (Nat.choose n 11) + (Nat.choose n 12)

/-- The theorem stating that for 12 chairs, there are 1634 subsets with at least three adjacent chairs -/
theorem adjacent_chair_subsets :
  subsets_with_adjacent_chairs n = 1634 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_chair_subsets_l2719_271970


namespace NUMINAMATH_CALUDE_dave_tickets_l2719_271984

/-- The number of tickets Dave initially won at the arcade -/
def initial_tickets : ℕ := 25

/-- The number of tickets Dave spent on a beanie -/
def spent_tickets : ℕ := 22

/-- The number of additional tickets Dave won later -/
def additional_tickets : ℕ := 15

/-- The number of tickets Dave has now -/
def current_tickets : ℕ := 18

theorem dave_tickets : 
  initial_tickets - spent_tickets + additional_tickets = current_tickets :=
by sorry

end NUMINAMATH_CALUDE_dave_tickets_l2719_271984


namespace NUMINAMATH_CALUDE_special_numbers_theorem_l2719_271994

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n < 2024 ∧ n % (39 * sum_of_digits n) = 0

theorem special_numbers_theorem :
  {n : ℕ | satisfies_condition n} = {351, 702, 1053, 1404} := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_theorem_l2719_271994


namespace NUMINAMATH_CALUDE_kanul_raw_materials_expenditure_l2719_271938

/-- The problem of calculating Kanul's expenditure on raw materials -/
theorem kanul_raw_materials_expenditure
  (total : ℝ)
  (machinery : ℝ)
  (cash_percentage : ℝ)
  (h1 : total = 7428.57)
  (h2 : machinery = 200)
  (h3 : cash_percentage = 0.30)
  (h4 : ∃ (raw_materials : ℝ), raw_materials + machinery + cash_percentage * total = total) :
  ∃ (raw_materials : ℝ), raw_materials = 5000 :=
sorry

end NUMINAMATH_CALUDE_kanul_raw_materials_expenditure_l2719_271938


namespace NUMINAMATH_CALUDE_gcd_problem_l2719_271963

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 7768) :
  Int.gcd (7 * b^2 + 55 * b + 125) (3 * b + 10) = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2719_271963


namespace NUMINAMATH_CALUDE_problem_solution_l2719_271908

theorem problem_solution (x : ℝ) (h : 3 * x - 45 = 159) : (x + 32) * 12 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2719_271908
