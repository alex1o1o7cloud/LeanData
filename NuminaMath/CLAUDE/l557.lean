import Mathlib

namespace NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l557_55752

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_implies_perp_line 
  (α β : Plane) (m n : Line) 
  (h_diff_planes : α ≠ β)
  (h_diff_lines : m ≠ n)
  (h_m_subset_β : subset m β)
  (h_n_subset_β : subset n β)
  (h_m_subset_α : subset m α)
  (h_n_perp_α : perp n α) :
  perpLine n m :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_perp_line_l557_55752


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l557_55784

/-- Given a line L1 with equation x + y - 5 = 0 and a point P (2, -1),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation x - y - 3 = 0 -/
theorem perpendicular_line_equation (L1 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  L1 = {(x, y) | x + y - 5 = 0} →
  P = (2, -1) →
  ∃ L2 : Set (ℝ × ℝ),
    (P ∈ L2) ∧
    (∀ (A B : ℝ × ℝ), A ∈ L1 → B ∈ L1 → A ≠ B →
      ∀ (C D : ℝ × ℝ), C ∈ L2 → D ∈ L2 → C ≠ D →
        ((A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) = 0)) ∧
    L2 = {(x, y) | x - y - 3 = 0} :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l557_55784


namespace NUMINAMATH_CALUDE_box_makers_l557_55727

/-- Represents the possible makers of the boxes -/
inductive Maker
| Cellini
| CelliniSon
| Bellini
| BelliniSon

/-- Represents a box with its inscription and actual maker -/
structure Box where
  color : String
  inscription : Prop
  maker : Maker

/-- The setup of the problem with two boxes -/
def boxSetup (goldBox silverBox : Box) : Prop :=
  (goldBox.color = "gold" ∧ silverBox.color = "silver") ∧
  (goldBox.inscription = (goldBox.maker = Maker.Cellini ∨ goldBox.maker = Maker.CelliniSon) ∧
                         (silverBox.maker = Maker.Cellini ∨ silverBox.maker = Maker.CelliniSon)) ∧
  (silverBox.inscription = (goldBox.maker ≠ Maker.CelliniSon ∧ goldBox.maker ≠ Maker.BelliniSon) ∧
                           (silverBox.maker ≠ Maker.CelliniSon ∧ silverBox.maker ≠ Maker.BelliniSon)) ∧
  (goldBox.inscription ≠ silverBox.inscription)

theorem box_makers (goldBox silverBox : Box) :
  boxSetup goldBox silverBox →
  (goldBox.maker = Maker.Cellini ∧ silverBox.maker = Maker.Bellini) :=
by
  sorry

end NUMINAMATH_CALUDE_box_makers_l557_55727


namespace NUMINAMATH_CALUDE_individual_test_scores_l557_55796

/-- Represents a student's test score -/
structure TestScore where
  value : ℝ

/-- Represents the population of students -/
def Population : Type := Fin 2100

/-- Represents the sample of students -/
def Sample : Type := Fin 100

/-- A function that assigns a test score to each student in the population -/
def scoreAssignment : Population → TestScore := sorry

/-- A function that selects the sample from the population -/
def sampleSelection : Sample → Population := sorry

theorem individual_test_scores 
  (p : Population) 
  (s : Sample) : 
  scoreAssignment p ≠ scoreAssignment (sampleSelection s) → p ≠ sampleSelection s := by
  sorry

end NUMINAMATH_CALUDE_individual_test_scores_l557_55796


namespace NUMINAMATH_CALUDE_computer_operations_l557_55724

/-- Represents the computer's specifications and performance --/
structure ComputerSpec where
  mult_rate : ℕ  -- multiplications per second
  add_rate : ℕ   -- additions per second
  switch_time : ℕ  -- time in seconds when switching from multiplications to additions
  total_time : ℕ  -- total operation time in seconds

/-- Calculates the total number of operations performed by the computer --/
def total_operations (spec : ComputerSpec) : ℕ :=
  let mult_ops := spec.mult_rate * spec.switch_time
  let add_ops := spec.add_rate * (spec.total_time - spec.switch_time)
  mult_ops + add_ops

/-- Theorem stating that the computer performs 63,000,000 operations in 2 hours --/
theorem computer_operations :
  let spec : ComputerSpec := {
    mult_rate := 5000,
    add_rate := 10000,
    switch_time := 1800,
    total_time := 7200
  }
  total_operations spec = 63000000 := by
  sorry


end NUMINAMATH_CALUDE_computer_operations_l557_55724


namespace NUMINAMATH_CALUDE_side_length_S2_correct_l557_55785

/-- The side length of square S2 in a specific rectangular arrangement -/
def side_length_S2 : ℕ := 650

/-- The width of the overall rectangle -/
def total_width : ℕ := 3400

/-- The height of the overall rectangle -/
def total_height : ℕ := 2100

/-- Theorem stating that the side length of S2 is correct given the constraints -/
theorem side_length_S2_correct :
  ∃ (r : ℕ),
    (2 * r + side_length_S2 = total_height) ∧
    (2 * r + 3 * side_length_S2 = total_width) := by
  sorry

end NUMINAMATH_CALUDE_side_length_S2_correct_l557_55785


namespace NUMINAMATH_CALUDE_a_greater_than_b_l557_55730

theorem a_greater_than_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 12345 = (111 + a) * (111 - b)) : a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l557_55730


namespace NUMINAMATH_CALUDE_rabbit_speed_l557_55709

theorem rabbit_speed (x : ℕ) : ((2 * x + 4) * 2 = 188) ↔ (x = 45) := by
  sorry

end NUMINAMATH_CALUDE_rabbit_speed_l557_55709


namespace NUMINAMATH_CALUDE_jason_seashells_theorem_l557_55766

/-- Calculates the number of seashells Jason gave to Tim -/
def seashells_given_to_tim (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Proves that the number of seashells Jason gave to Tim is correct -/
theorem jason_seashells_theorem (initial_seashells current_seashells : ℕ) 
  (h1 : initial_seashells = 49)
  (h2 : current_seashells = 36)
  (h3 : initial_seashells ≥ current_seashells) :
  seashells_given_to_tim initial_seashells current_seashells = 13 := by
  sorry

#eval seashells_given_to_tim 49 36

end NUMINAMATH_CALUDE_jason_seashells_theorem_l557_55766


namespace NUMINAMATH_CALUDE_geometric_sequence_sufficient_not_necessary_l557_55711

/-- Defines a geometric sequence -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- Defines the condition b^2 = ac -/
def condition_b_squared_eq_ac (a b c : ℝ) : Prop :=
  b^2 = a * c

/-- Theorem stating that "a, b, c form a geometric sequence" is sufficient 
    but not necessary for "b^2 = ac" -/
theorem geometric_sequence_sufficient_not_necessary :
  (∀ a b c : ℝ, is_geometric_sequence a b c → condition_b_squared_eq_ac a b c) ∧
  ¬(∀ a b c : ℝ, condition_b_squared_eq_ac a b c → is_geometric_sequence a b c) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sufficient_not_necessary_l557_55711


namespace NUMINAMATH_CALUDE_mary_max_earnings_l557_55706

/-- Calculates the maximum weekly earnings for a worker with the given conditions --/
def maxWeeklyEarnings (maxHours : ℕ) (regularHours : ℕ) (regularRate : ℚ) (overtimeRateIncrease : ℚ) : ℚ :=
  let overtimeHours := maxHours - regularHours
  let overtimeRate := regularRate * (1 + overtimeRateIncrease)
  regularHours * regularRate + overtimeHours * overtimeRate

/-- Theorem stating that Mary's maximum weekly earnings are $410 --/
theorem mary_max_earnings :
  maxWeeklyEarnings 45 20 8 (1/4) = 410 := by
  sorry

#eval maxWeeklyEarnings 45 20 8 (1/4)

end NUMINAMATH_CALUDE_mary_max_earnings_l557_55706


namespace NUMINAMATH_CALUDE_toothpick_grid_30_15_l557_55763

/-- Represents a rectangular grid made of toothpicks -/
structure ToothpickGrid where
  height : ℕ  -- Number of toothpicks in height
  width : ℕ   -- Number of toothpicks in width

/-- Calculates the total number of toothpicks in a grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  (grid.height + 1) * grid.width + (grid.width + 1) * grid.height

/-- Theorem: A 30x15 toothpick grid uses 945 toothpicks -/
theorem toothpick_grid_30_15 :
  totalToothpicks { height := 30, width := 15 } = 945 := by
  sorry


end NUMINAMATH_CALUDE_toothpick_grid_30_15_l557_55763


namespace NUMINAMATH_CALUDE_souvenir_distribution_solution_l557_55700

/-- Represents the souvenir distribution problem -/
structure SouvenirDistribution where
  total_items : ℕ
  total_cost : ℕ
  type_a_cost : ℕ
  type_a_price : ℕ
  type_b_cost : ℕ
  type_b_price : ℕ

/-- Theorem stating the solution to the souvenir distribution problem -/
theorem souvenir_distribution_solution (sd : SouvenirDistribution)
  (h1 : sd.total_items = 100)
  (h2 : sd.total_cost = 6200)
  (h3 : sd.type_a_cost = 50)
  (h4 : sd.type_a_price = 100)
  (h5 : sd.type_b_cost = 70)
  (h6 : sd.type_b_price = 90) :
  ∃ (type_a type_b : ℕ),
    type_a + type_b = sd.total_items ∧
    type_a * sd.type_a_cost + type_b * sd.type_b_cost = sd.total_cost ∧
    type_a = 40 ∧
    type_b = 60 ∧
    (type_a * (sd.type_a_price - sd.type_a_cost) + type_b * (sd.type_b_price - sd.type_b_cost)) = 3200 :=
by
  sorry

end NUMINAMATH_CALUDE_souvenir_distribution_solution_l557_55700


namespace NUMINAMATH_CALUDE_probability_of_colored_ball_l557_55726

def urn_total : ℕ := 30
def red_balls : ℕ := 10
def blue_balls : ℕ := 5
def white_balls : ℕ := 15

theorem probability_of_colored_ball :
  (red_balls + blue_balls : ℚ) / urn_total = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_probability_of_colored_ball_l557_55726


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l557_55778

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 6 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l557_55778


namespace NUMINAMATH_CALUDE_debby_flour_purchase_l557_55721

/-- Given that Debby initially had 12 pounds of flour and ended up with 16 pounds in total,
    prove that she bought 4 pounds of flour. -/
theorem debby_flour_purchase (initial_flour : ℕ) (total_flour : ℕ) (purchased_flour : ℕ) :
  initial_flour = 12 →
  total_flour = 16 →
  total_flour = initial_flour + purchased_flour →
  purchased_flour = 4 := by
  sorry

end NUMINAMATH_CALUDE_debby_flour_purchase_l557_55721


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l557_55795

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_30 : a 30 = 100)
  (h_100 : a 100 = 30) :
  ∃ d : ℝ, d = -1 ∧ ∀ n, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l557_55795


namespace NUMINAMATH_CALUDE_rational_terms_not_adjacent_probability_l557_55734

theorem rational_terms_not_adjacent_probability (n : ℕ) (rational_terms : ℕ) :
  n = 9 ∧ rational_terms = 3 →
  (Nat.factorial 6 * (Nat.factorial 7 / (Nat.factorial 4 * Nat.factorial 3))) / Nat.factorial 9 = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_rational_terms_not_adjacent_probability_l557_55734


namespace NUMINAMATH_CALUDE_inverse_prop_problem_l557_55773

/-- Two numbers are inversely proportional if their product is constant -/
def inverse_proportional (a b : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ x : ℝ, a x * b x = k

theorem inverse_prop_problem (a b : ℝ → ℝ) 
  (h1 : inverse_proportional a b)
  (h2 : ∃ x : ℝ, a x + b x = 60 ∧ a x = 3 * b x) :
  b (-12) = -56.25 := by
  sorry


end NUMINAMATH_CALUDE_inverse_prop_problem_l557_55773


namespace NUMINAMATH_CALUDE_lcm_48_180_l557_55759

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l557_55759


namespace NUMINAMATH_CALUDE_triangle_max_area_l557_55750

/-- Given two positive real numbers a and b representing the lengths of two sides of a triangle,
    the area of the triangle is maximized when these sides are perpendicular. -/
theorem triangle_max_area (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ θ : ℝ, 0 < θ ∧ θ < π → (1/2) * a * b * Real.sin θ ≤ (1/2) * a * b := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l557_55750


namespace NUMINAMATH_CALUDE_complex_square_roots_l557_55732

theorem complex_square_roots (z : ℂ) : 
  z ^ 2 = -104 + 63 * I ∧ (5 + 8 * I) ^ 2 = -104 + 63 * I → 
  (-5 - 8 * I) ^ 2 = -104 + 63 * I := by
sorry

end NUMINAMATH_CALUDE_complex_square_roots_l557_55732


namespace NUMINAMATH_CALUDE_race_catchup_time_l557_55760

/-- Proves that Nicky runs for 48 seconds before Cristina catches up to him in a 500-meter race --/
theorem race_catchup_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 500)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) :
  let catchup_time := head_start + (head_start * nicky_speed) / (cristina_speed - nicky_speed)
  catchup_time = 48 := by
  sorry

end NUMINAMATH_CALUDE_race_catchup_time_l557_55760


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l557_55710

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  x + 1 / (x - 2) ≥ 4 := by
  sorry

theorem min_value_achieved (x : ℝ) (h : x > 2) :
  ∃ x₀ > 2, x₀ + 1 / (x₀ - 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l557_55710


namespace NUMINAMATH_CALUDE_paths_through_F_and_H_l557_55735

/-- Represents a point on the grid -/
structure Point where
  x : Nat
  y : Nat

/-- Calculates the number of paths between two points on a grid -/
def numPaths (start finish : Point) : Nat :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The grid dimensions -/
def gridWidth : Nat := 7
def gridHeight : Nat := 6

/-- The points on the grid -/
def E : Point := ⟨0, 5⟩
def F : Point := ⟨4, 4⟩
def H : Point := ⟨5, 2⟩
def G : Point := ⟨6, 0⟩

/-- Theorem: The number of 12-step paths from E to G passing through F and then H is 135 -/
theorem paths_through_F_and_H : 
  numPaths E F * numPaths F H * numPaths H G = 135 := by
  sorry

end NUMINAMATH_CALUDE_paths_through_F_and_H_l557_55735


namespace NUMINAMATH_CALUDE_number_equation_solution_l557_55775

theorem number_equation_solution : 
  ∃ x : ℝ, (3 * x - 5 = 40) ∧ (x = 15) := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l557_55775


namespace NUMINAMATH_CALUDE_sets_theorem_l557_55792

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem sets_theorem (a : ℝ) :
  (A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 10}) ∧
  ((Set.compl A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10}) ∧
  ((A ∩ C a).Nonempty ↔ a > -3) :=
by sorry

end NUMINAMATH_CALUDE_sets_theorem_l557_55792


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_l557_55740

theorem wire_length_around_square_field (area : ℝ) (num_rounds : ℕ) : 
  area = 27889 ∧ num_rounds = 11 → 
  Real.sqrt area * 4 * num_rounds = 7348 := by
sorry

end NUMINAMATH_CALUDE_wire_length_around_square_field_l557_55740


namespace NUMINAMATH_CALUDE_soccer_ball_max_height_l557_55723

/-- The height function of a soccer ball's trajectory -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 11

/-- Theorem stating the maximum height of the soccer ball -/
theorem soccer_ball_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 136 :=
sorry

end NUMINAMATH_CALUDE_soccer_ball_max_height_l557_55723


namespace NUMINAMATH_CALUDE_exam_score_proof_l557_55772

/-- Proves that the average score of students who took the exam on the assigned day was 65% -/
theorem exam_score_proof (total_students : ℕ) (assigned_day_percentage : ℚ) 
  (makeup_score : ℚ) (class_average : ℚ) : 
  total_students = 100 →
  assigned_day_percentage = 70 / 100 →
  makeup_score = 95 / 100 →
  class_average = 74 / 100 →
  (assigned_day_percentage * total_students * assigned_day_score + 
   (1 - assigned_day_percentage) * total_students * makeup_score) / total_students = class_average →
  assigned_day_score = 65 / 100 :=
by
  sorry

#check exam_score_proof

end NUMINAMATH_CALUDE_exam_score_proof_l557_55772


namespace NUMINAMATH_CALUDE_ray_nickels_left_l557_55790

def nickel_value : ℕ := 5
def initial_cents : ℕ := 95
def cents_to_peter : ℕ := 25

theorem ray_nickels_left : 
  let initial_nickels := initial_cents / nickel_value
  let nickels_to_peter := cents_to_peter / nickel_value
  let cents_to_randi := 2 * cents_to_peter
  let nickels_to_randi := cents_to_randi / nickel_value
  initial_nickels - nickels_to_peter - nickels_to_randi = 4 := by
  sorry

end NUMINAMATH_CALUDE_ray_nickels_left_l557_55790


namespace NUMINAMATH_CALUDE_slope_inequality_l557_55743

open Real

theorem slope_inequality (x₁ x₂ : ℝ) (hx : 0 < x₁ ∧ x₁ < x₂) :
  let f := λ x : ℝ => Real.log x
  let k := (f x₂ - f x₁) / (x₂ - x₁)
  1 / x₂ < k ∧ k < 1 / x₁ := by
  sorry

end NUMINAMATH_CALUDE_slope_inequality_l557_55743


namespace NUMINAMATH_CALUDE_same_color_probability_l557_55786

/-- The probability of drawing two balls of the same color from a bag with 6 green and 7 white balls -/
theorem same_color_probability (total_balls : ℕ) (green_balls : ℕ) (white_balls : ℕ)
  (h1 : total_balls = green_balls + white_balls)
  (h2 : green_balls = 6)
  (h3 : white_balls = 7) :
  (green_balls * (green_balls - 1) + white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1)) = 6 / 13 := by
sorry

end NUMINAMATH_CALUDE_same_color_probability_l557_55786


namespace NUMINAMATH_CALUDE_union_of_sets_l557_55738

theorem union_of_sets : 
  let A : Set ℕ := {2, 3}
  let B : Set ℕ := {1, 2}
  A ∪ B = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l557_55738


namespace NUMINAMATH_CALUDE_theater_line_up_ways_l557_55788

theorem theater_line_up_ways : 
  let number_of_windows : ℕ := 2
  let number_of_people : ℕ := 6
  number_of_windows ^ number_of_people * Nat.factorial number_of_people = 46080 :=
by sorry

end NUMINAMATH_CALUDE_theater_line_up_ways_l557_55788


namespace NUMINAMATH_CALUDE_gcd_128_144_360_l557_55751

theorem gcd_128_144_360 : Nat.gcd 128 (Nat.gcd 144 360) = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_128_144_360_l557_55751


namespace NUMINAMATH_CALUDE_remaining_lawn_after_one_hour_l557_55731

/-- Given that Mary can mow the entire lawn in 3 hours, 
    this function calculates the fraction of the lawn mowed in a given time. -/
def fraction_mowed (hours : ℚ) : ℚ := hours / 3

/-- This theorem states that if Mary works for 1 hour, 
    then 2/3 of the lawn remains to be mowed. -/
theorem remaining_lawn_after_one_hour : 
  1 - (fraction_mowed 1) = 2/3 := by sorry

end NUMINAMATH_CALUDE_remaining_lawn_after_one_hour_l557_55731


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l557_55768

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 4 + seq.a 14 = 2 → S seq 17 = 17) ∧
  (seq.a 11 = 10 → S seq 21 = 210) ∧
  (S seq 11 = 55 → seq.a 6 = 5) ∧
  (S seq 8 = 100 ∧ S seq 16 = 392 → S seq 24 = 876) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l557_55768


namespace NUMINAMATH_CALUDE_cafeteria_seats_available_l557_55736

theorem cafeteria_seats_available 
  (total_tables : ℕ) 
  (seats_per_table : ℕ) 
  (people_dining : ℕ) : 
  total_tables = 40 → 
  seats_per_table = 12 → 
  people_dining = 325 → 
  total_tables * seats_per_table - people_dining = 155 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_seats_available_l557_55736


namespace NUMINAMATH_CALUDE_people_who_left_line_l557_55781

theorem people_who_left_line (initial_people : ℕ) (joined : ℕ) (final_people : ℕ) 
  (h1 : initial_people = 9)
  (h2 : joined = 3)
  (h3 : final_people = 6)
  : initial_people - (initial_people - joined + final_people) = 6 := by
  sorry

end NUMINAMATH_CALUDE_people_who_left_line_l557_55781


namespace NUMINAMATH_CALUDE_line_through_points_and_midpoint_l557_55729

/-- Given a line y = ax + b passing through (2, 3) and (10, 19) with their midpoint on the line, a - b = 3 -/
theorem line_through_points_and_midpoint (a b : ℝ) : 
  (3 = a * 2 + b) → 
  (19 = a * 10 + b) → 
  (11 = a * 6 + b) → 
  a - b = 3 := by sorry

end NUMINAMATH_CALUDE_line_through_points_and_midpoint_l557_55729


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l557_55716

/-- A boat travels on a river with a current. -/
structure Boat :=
  (speed : ℝ)  -- Speed of the boat in still water in km/h

/-- A river with a current. -/
structure River :=
  (current : ℝ)  -- Speed of the river current in km/h

/-- The distance traveled by a boat on a river in one hour. -/
def distanceTraveled (b : Boat) (r : River) (withCurrent : Bool) : ℝ :=
  if withCurrent then b.speed + r.current else b.speed - r.current

theorem boat_distance_along_stream 
  (b : Boat) 
  (r : River) 
  (h1 : b.speed = 8) 
  (h2 : distanceTraveled b r false = 5) : 
  distanceTraveled b r true = 11 := by
  sorry

#check boat_distance_along_stream

end NUMINAMATH_CALUDE_boat_distance_along_stream_l557_55716


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l557_55703

theorem maintenance_check_increase (original : ℝ) (new : ℝ) 
  (h1 : original = 30)
  (h2 : new = 60) :
  (new - original) / original * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l557_55703


namespace NUMINAMATH_CALUDE_carpet_cost_calculation_l557_55764

/-- Calculate the cost of a carpet given its dimensions and the price per square meter -/
def calculate_carpet_cost (length width price_per_sqm : ℝ) : ℝ :=
  length * width * price_per_sqm

/-- The problem statement -/
theorem carpet_cost_calculation :
  let first_carpet_breadth : ℝ := 6
  let first_carpet_length : ℝ := 1.44 * first_carpet_breadth
  let second_carpet_length : ℝ := first_carpet_length * 1.427
  let second_carpet_breadth : ℝ := first_carpet_breadth * 1.275
  let price_per_sqm : ℝ := 46.35
  
  abs (calculate_carpet_cost second_carpet_length second_carpet_breadth price_per_sqm - 4371.78) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_carpet_cost_calculation_l557_55764


namespace NUMINAMATH_CALUDE_parallel_lines_count_l557_55753

/-- Given two sets of intersecting parallel lines in a plane, where one set has 8 lines
    and the intersection forms 588 parallelograms, prove that the other set has 85 lines. -/
theorem parallel_lines_count (n : ℕ) 
  (h1 : n > 0)
  (h2 : (n - 1) * 7 = 588) : 
  n = 85 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_count_l557_55753


namespace NUMINAMATH_CALUDE_trig_simplification_l557_55704

theorem trig_simplification (α : ℝ) :
  Real.sin (α - 4 * Real.pi) * Real.sin (Real.pi - α) -
  2 * (Real.cos ((3 * Real.pi) / 2 + α))^2 -
  Real.sin (α + Real.pi) * Real.cos (Real.pi / 2 + α) =
  -2 * (Real.sin α)^2 := by sorry

end NUMINAMATH_CALUDE_trig_simplification_l557_55704


namespace NUMINAMATH_CALUDE_fraction_domain_l557_55756

theorem fraction_domain (x : ℝ) : 
  (∃ y : ℝ, y = 5 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_domain_l557_55756


namespace NUMINAMATH_CALUDE_constant_d_value_l557_55749

-- Define the problem statement
theorem constant_d_value (a d : ℝ) :
  (∀ x : ℝ, (x + 3) * (x + a) = x^2 + d*x + 12) →
  d = 7 :=
by sorry

end NUMINAMATH_CALUDE_constant_d_value_l557_55749


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l557_55758

theorem fifteenth_student_age (total_students : ℕ) (avg_age : ℕ) 
  (group1_size : ℕ) (group1_avg : ℕ) (group2_size : ℕ) (group2_avg : ℕ) :
  total_students = 15 →
  avg_age = 15 →
  group1_size = 3 →
  group1_avg = 14 →
  group2_size = 11 →
  group2_avg = 16 →
  total_students * avg_age - (group1_size * group1_avg + group2_size * group2_avg) = 7 := by
  sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_fifteenth_student_age_l557_55758


namespace NUMINAMATH_CALUDE_complex_conversion_l557_55725

theorem complex_conversion (z : ℂ) : z = Complex.exp (13 * Real.pi * Complex.I / 4) * (Real.sqrt 3) →
  z = Complex.mk (Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_conversion_l557_55725


namespace NUMINAMATH_CALUDE_f_is_odd_l557_55767

def f (p : ℝ) (x : ℝ) : ℝ := x * |x| + p * x

theorem f_is_odd (p : ℝ) : 
  ∀ x : ℝ, f p (-x) = -(f p x) := by
sorry

end NUMINAMATH_CALUDE_f_is_odd_l557_55767


namespace NUMINAMATH_CALUDE_rearrangement_time_l557_55745

/-- The time required to write all rearrangements of a name -/
theorem rearrangement_time (name_length : ℕ) (writing_speed : ℕ) (h1 : name_length = 8) (h2 : writing_speed = 15) :
  (name_length.factorial / writing_speed : ℚ) / 60 = 44.8 := by
sorry

end NUMINAMATH_CALUDE_rearrangement_time_l557_55745


namespace NUMINAMATH_CALUDE_root_of_polynomial_l557_55791

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 16*x^2 + 4

-- State the theorem
theorem root_of_polynomial :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 16*x^2 + 4) ∧
  -- The polynomial has degree 4
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- √3 + √5 is a root of the polynomial
  p (Real.sqrt 3 + Real.sqrt 5) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l557_55791


namespace NUMINAMATH_CALUDE_angle_B_measure_l557_55702

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.b * Real.sin t.B - t.c * Real.sin t.C = t.a ∧
  (t.b^2 + t.c^2 - t.a^2) / 4 = 1/2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem angle_B_measure (t : Triangle) :
  satisfies_conditions t → t.B = 77.5 * π / 180 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l557_55702


namespace NUMINAMATH_CALUDE_symmetry_axis_of_translated_sine_function_l557_55741

/-- Given a function f(x) = 2sin(2x + π/6), g(x) is obtained by translating
    the graph of f(x) to the right by π/6 units. This theorem states that
    x = π/3 is an equation of one symmetry axis of g(x). -/
theorem symmetry_axis_of_translated_sine_function :
  ∀ (f g : ℝ → ℝ),
  (∀ x, f x = 2 * Real.sin (2 * x + π / 6)) →
  (∀ x, g x = f (x - π / 6)) →
  (∃ k : ℤ, 2 * (π / 3) - π / 6 = π / 2 + k * π) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_of_translated_sine_function_l557_55741


namespace NUMINAMATH_CALUDE_power_of_two_difference_l557_55739

theorem power_of_two_difference (n : ℕ) (h : n > 0) : 2^n - 2^(n-1) = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_difference_l557_55739


namespace NUMINAMATH_CALUDE_inequality_proof_l557_55770

theorem inequality_proof (x a : ℝ) (hx : x > 0 ∧ x ≠ 1) (ha : a < 1) :
  (1 - x^a) / (1 - x) < (1 + x)^(a - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l557_55770


namespace NUMINAMATH_CALUDE_max_guaranteed_score_is_four_l557_55774

/-- Represents a player in the game -/
inductive Player : Type
| B : Player
| R : Player

/-- Represents a color of a square -/
inductive Color : Type
| White : Color
| Blue : Color
| Red : Color

/-- Represents a square on the infinite grid -/
structure Square :=
  (x : ℤ)
  (y : ℤ)

/-- Represents the game state -/
structure GameState :=
  (grid : Square → Color)
  (currentPlayer : Player)

/-- Represents a simple polygon on the grid -/
structure SimplePolygon :=
  (squares : Set Square)

/-- The score of player B is the area of the largest simple polygon of blue squares -/
def score (state : GameState) : ℕ :=
  sorry

/-- A strategy for player B -/
def Strategy : Type :=
  GameState → Square

/-- The maximum guaranteed score for player B -/
def maxGuaranteedScore : ℕ :=
  sorry

/-- The main theorem stating that the maximum guaranteed score for B is 4 -/
theorem max_guaranteed_score_is_four :
  maxGuaranteedScore = 4 :=
sorry

end NUMINAMATH_CALUDE_max_guaranteed_score_is_four_l557_55774


namespace NUMINAMATH_CALUDE_replaced_student_weight_l557_55737

theorem replaced_student_weight 
  (n : ℕ) 
  (new_weight : ℝ) 
  (avg_decrease : ℝ) : 
  n = 8 → 
  new_weight = 46 → 
  avg_decrease = 5 → 
  ∃ (old_weight : ℝ), old_weight = n * avg_decrease + new_weight :=
by
  sorry

end NUMINAMATH_CALUDE_replaced_student_weight_l557_55737


namespace NUMINAMATH_CALUDE_dance_group_composition_l557_55742

/-- Represents a dance group --/
structure DanceGroup where
  boy_dancers : ℕ
  girl_dancers : ℕ
  boy_escorts : ℕ
  girl_escorts : ℕ

/-- The problem statement --/
theorem dance_group_composition 
  (group_a group_b : DanceGroup)
  (h1 : group_a.boy_dancers + group_a.girl_dancers = group_b.boy_dancers + group_b.girl_dancers + 1)
  (h2 : group_a.boy_escorts + group_a.girl_escorts = group_b.boy_escorts + group_b.girl_escorts + 1)
  (h3 : group_a.boy_dancers + group_b.boy_dancers = group_a.girl_dancers + group_b.girl_dancers + 1)
  (h4 : (group_a.boy_dancers + group_b.boy_dancers) * (group_a.girl_dancers + group_b.girl_dancers) = 484)
  (h5 : (group_a.boy_dancers + group_a.boy_escorts) * (group_b.girl_dancers + group_b.girl_escorts) +
        (group_b.boy_dancers + group_b.boy_escorts) * (group_a.girl_dancers + group_a.girl_escorts) = 246)
  (h6 : (group_a.boy_dancers + group_b.boy_dancers) * (group_a.girl_dancers + group_b.girl_dancers) = 306)
  (h7 : group_a.boy_dancers * group_a.girl_dancers + group_b.boy_dancers * group_b.girl_dancers = 150)
  (h8 : let total := group_a.boy_dancers + group_a.girl_dancers + group_a.boy_escorts + group_a.girl_escorts +
                     group_b.boy_dancers + group_b.girl_dancers + group_b.boy_escorts + group_b.girl_escorts
        (total * (total - 1)) / 2 = 946) :
  group_a = { boy_dancers := 8, girl_dancers := 10, boy_escorts := 2, girl_escorts := 3 } ∧
  group_b = { boy_dancers := 10, girl_dancers := 7, boy_escorts := 2, girl_escorts := 2 } :=
by sorry

end NUMINAMATH_CALUDE_dance_group_composition_l557_55742


namespace NUMINAMATH_CALUDE_particular_number_problem_l557_55755

theorem particular_number_problem (x : ℚ) (h : (x + 10) / 5 = 4) : 3 * x - 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_problem_l557_55755


namespace NUMINAMATH_CALUDE_word_sum_proof_l557_55754

theorem word_sum_proof :
  ∀ A B C : ℕ,
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 →
    A ≠ B ∧ B ≠ C ∧ A ≠ C →
    A < 10 ∧ B < 10 ∧ C < 10 →
    100 * A + 10 * B + C +
    100 * B + 10 * C + A +
    100 * C + 10 * A + B =
    1000 * A + 100 * B + 10 * B + C →
    A = 1 ∧ B = 9 ∧ C = 8 := by
sorry

end NUMINAMATH_CALUDE_word_sum_proof_l557_55754


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l557_55746

/-- Given a triangle with area t, angle α, and angle β, prove that the sides a, b, and c have the specified lengths. -/
theorem triangle_side_lengths 
  (t : ℝ) 
  (α β : Real) 
  (h_t : t = 4920)
  (h_α : α = 43 + 36 / 60 + 10 / 3600)
  (h_β : β = 72 + 23 / 60 + 11 / 3600) :
  ∃ (a b c : ℝ), 
    (abs (a - 89) < 1) ∧ 
    (abs (b - 123) < 1) ∧ 
    (abs (c - 116) < 1) ∧
    (a > 0) ∧ (b > 0) ∧ (c > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_side_lengths_l557_55746


namespace NUMINAMATH_CALUDE_min_value_theorem_l557_55761

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_constraint : a + b + c = 5) :
  (9 / a) + (16 / b) + (25 / c^2) ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l557_55761


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l557_55705

/-- 
An arithmetic sequence is defined by its first term, common difference, and last term.
This theorem proves that an arithmetic sequence with first term -6, last term 38,
and common difference 4 has exactly 12 terms.
-/
theorem arithmetic_sequence_length 
  (a : ℤ) (d : ℤ) (l : ℤ) (n : ℕ) 
  (h1 : a = -6) 
  (h2 : d = 4) 
  (h3 : l = 38) 
  (h4 : l = a + (n - 1) * d) : n = 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l557_55705


namespace NUMINAMATH_CALUDE_power_23_2005_mod_36_l557_55719

theorem power_23_2005_mod_36 : 23^2005 % 36 = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_23_2005_mod_36_l557_55719


namespace NUMINAMATH_CALUDE_concentric_squares_ratio_l557_55799

/-- Given two concentric squares ABCD (outer) and EFGH (inner) with side lengths a and b
    respectively, if the area of the shaded region between them is p% of the area of ABCD,
    then a/b = 1/sqrt(1-p/100). -/
theorem concentric_squares_ratio (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : 0 < p ∧ p < 100) :
  (a^2 - b^2) / a^2 = p / 100 → a / b = 1 / Real.sqrt (1 - p / 100) := by
  sorry

end NUMINAMATH_CALUDE_concentric_squares_ratio_l557_55799


namespace NUMINAMATH_CALUDE_andrews_mangoes_l557_55780

/-- Given Andrew's purchase of grapes and mangoes, prove the amount of mangoes bought -/
theorem andrews_mangoes :
  ∀ (mango_kg : ℝ),
  let grape_kg : ℝ := 8
  let grape_price : ℝ := 70
  let mango_price : ℝ := 55
  let total_cost : ℝ := 1055
  (grape_kg * grape_price + mango_kg * mango_price = total_cost) →
  mango_kg = 9 := by
sorry

end NUMINAMATH_CALUDE_andrews_mangoes_l557_55780


namespace NUMINAMATH_CALUDE_quadratic_sum_equality_l557_55794

/-- A quadratic function satisfying specific conditions -/
def P : ℝ → ℝ := fun x ↦ 6 * x^2 - 3 * x + 7

/-- The theorem statement -/
theorem quadratic_sum_equality (a b c : ℤ) :
  P 0 = 7 ∧ P 1 = 10 ∧ P 2 = 25 ∧
  (∀ x : ℝ, 0 < x → x < 1 →
    (∑' n, P n * x^n) = (a * x^2 + b * x + c) / (1 - x)^3) →
  (a, b, c) = (16, -11, 7) := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_equality_l557_55794


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l557_55762

theorem sqrt_fraction_equality : 
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (25 + 36)) = (17 * Real.sqrt 61) / 61 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l557_55762


namespace NUMINAMATH_CALUDE_supplementary_angles_theorem_l557_55748

theorem supplementary_angles_theorem (A B : ℝ) : 
  A + B = 180 →  -- angles A and B are supplementary
  A = 4 * B →    -- measure of angle A is 4 times angle B
  A = 144 :=     -- measure of angle A is 144 degrees
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_theorem_l557_55748


namespace NUMINAMATH_CALUDE_sin_160_cos_10_plus_cos_20_sin_10_l557_55708

theorem sin_160_cos_10_plus_cos_20_sin_10 :
  Real.sin (160 * π / 180) * Real.cos (10 * π / 180) +
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_160_cos_10_plus_cos_20_sin_10_l557_55708


namespace NUMINAMATH_CALUDE_equation_represents_intersecting_lines_l557_55787

-- Define the equation
def equation (x y : ℝ) : Prop := x^2 - y^2 = 0

-- Theorem statement
theorem equation_represents_intersecting_lines :
  ∃ (f g : ℝ → ℝ), 
    (∀ x, f x = x ∧ g x = -x) ∧
    (∀ x y, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_intersecting_lines_l557_55787


namespace NUMINAMATH_CALUDE_train_length_l557_55714

/-- The length of a train given its crossing times over a post and a platform -/
theorem train_length (post_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  post_time = 15 →
  platform_length = 100 →
  platform_time = 25 →
  ∃ (train_length : ℝ),
    train_length / post_time = (train_length + platform_length) / platform_time ∧
    train_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l557_55714


namespace NUMINAMATH_CALUDE_election_votes_l557_55783

theorem election_votes (total_members : ℕ) (winner_percentage : ℚ) (winner_total_percentage : ℚ) :
  total_members = 1600 →
  winner_percentage = 60 / 100 →
  winner_total_percentage = 19.6875 / 100 →
  (↑total_members * winner_total_percentage : ℚ) / winner_percentage = 525 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l557_55783


namespace NUMINAMATH_CALUDE_difference_p_q_l557_55720

theorem difference_p_q (p q : ℚ) (hp : 3 / p = 8) (hq : 3 / q = 18) : p - q = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_difference_p_q_l557_55720


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l557_55779

theorem repeating_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) = (14 / 37 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l557_55779


namespace NUMINAMATH_CALUDE_tv_sales_effect_l557_55715

theorem tv_sales_effect (P Q : ℝ) (h_P : P > 0) (h_Q : Q > 0) : 
  let new_price := 0.82 * P
  let new_quantity := 1.88 * Q
  let original_value := P * Q
  let new_value := new_price * new_quantity
  (new_value / original_value - 1) * 100 = 54.26 := by
sorry

end NUMINAMATH_CALUDE_tv_sales_effect_l557_55715


namespace NUMINAMATH_CALUDE_square_to_obtuse_triangle_l557_55744

/-- Represents a part of a square -/
structure SquarePart where
  -- Add necessary fields to represent a part of a square
  -- This is a placeholder and should be defined more precisely based on the problem requirements

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields to represent a triangle
  -- This is a placeholder and should be defined more precisely based on the problem requirements

/-- Determines if a triangle is obtuse -/
def is_obtuse (t : Triangle) : Prop :=
  -- Add the condition for a triangle to be obtuse
  -- This is a placeholder and should be defined more precisely based on the problem requirements
  sorry

/-- Determines if parts can form a triangle -/
def can_form_triangle (parts : List SquarePart) : Prop :=
  -- Add the condition for parts to be able to form a triangle
  -- This is a placeholder and should be defined more precisely based on the problem requirements
  sorry

/-- Theorem stating that a square can be cut into 3 parts that form an obtuse triangle -/
theorem square_to_obtuse_triangle :
  ∃ (parts : List SquarePart), parts.length = 3 ∧
    ∃ (t : Triangle), can_form_triangle parts ∧ is_obtuse t :=
sorry

end NUMINAMATH_CALUDE_square_to_obtuse_triangle_l557_55744


namespace NUMINAMATH_CALUDE_lesser_number_l557_55701

theorem lesser_number (x y : ℝ) (h1 : x + y = 60) (h2 : 3 * (x - y) = 9) : min x y = 28.5 := by
  sorry

end NUMINAMATH_CALUDE_lesser_number_l557_55701


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l557_55747

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → a ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l557_55747


namespace NUMINAMATH_CALUDE_problem_solution_l557_55793

theorem problem_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c - d + 3) → d = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l557_55793


namespace NUMINAMATH_CALUDE_snow_probability_l557_55797

theorem snow_probability (p1 p2 p3 : ℚ) 
  (h1 : p1 = 1/2) 
  (h2 : p2 = 3/4) 
  (h3 : p3 = 2/3) : 
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 23/24 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l557_55797


namespace NUMINAMATH_CALUDE_inequality_proof_l557_55717

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l557_55717


namespace NUMINAMATH_CALUDE_system_solution_l557_55777

theorem system_solution (x y : ℚ) : 
  (x + y = x^2 + 2*x*y + y^2 ∧ x - y = x^2 - 2*x*y + y^2) ↔ 
  ((x = 1/2 ∧ y = -1/2) ∨ (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 1/2 ∧ y = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l557_55777


namespace NUMINAMATH_CALUDE_maintenance_team_journey_l557_55722

/-- Represents the direction of travel --/
inductive Direction
  | East
  | West

/-- Represents a segment of the journey --/
structure Segment where
  distance : ℝ
  direction : Direction

/-- Calculates the net distance traveled given a list of segments --/
def netDistance (journey : List Segment) : ℝ := sorry

/-- Calculates the total distance traveled given a list of segments --/
def totalDistance (journey : List Segment) : ℝ := sorry

/-- Theorem: The maintenance team's final position and fuel consumption --/
theorem maintenance_team_journey 
  (journey : List Segment)
  (fuel_rate : ℝ)
  (h1 : journey = [
    ⟨12, Direction.East⟩, 
    ⟨6, Direction.West⟩, 
    ⟨4, Direction.East⟩, 
    ⟨2, Direction.West⟩, 
    ⟨8, Direction.West⟩, 
    ⟨13, Direction.East⟩, 
    ⟨2, Direction.West⟩
  ])
  (h2 : fuel_rate = 0.2) :
  netDistance journey = 11 ∧ 
  totalDistance journey * fuel_rate * 2 = 11.6 := by sorry

end NUMINAMATH_CALUDE_maintenance_team_journey_l557_55722


namespace NUMINAMATH_CALUDE_min_value_of_y_l557_55782

-- Define a function that calculates the sum of squares of 11 consecutive integers
def sumOfSquares (x : ℤ) : ℤ := (x-5)^2 + (x-4)^2 + (x-3)^2 + (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2

-- Theorem statement
theorem min_value_of_y (y : ℤ) : (∃ x : ℤ, y^2 = sumOfSquares x) → y ≥ -11 ∧ (∃ x : ℤ, (-11)^2 = sumOfSquares x) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_y_l557_55782


namespace NUMINAMATH_CALUDE_parabola_line_intersection_sum_l557_55769

/-- Parabola P with equation y = x^2 -/
def P : ℝ → ℝ := fun x ↦ x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- Line through Q with slope m -/
def line_through_Q (m : ℝ) : ℝ → ℝ := fun x ↦ m * (x - Q.1) + Q.2

/-- The line does not intersect the parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line_through_Q m x

/-- Theorem stating that r + s = 40 -/
theorem parabola_line_intersection_sum :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) → r + s = 40 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_sum_l557_55769


namespace NUMINAMATH_CALUDE_one_billion_scientific_notation_l557_55765

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- One billion -/
def oneBillion : ℕ := 1000000000

/-- Theorem: The scientific notation of one billion is 1 × 10^9 -/
theorem one_billion_scientific_notation :
  ∃ (sn : ScientificNotation), sn.a = 1 ∧ sn.n = 9 ∧ (sn.a * (10 : ℝ) ^ sn.n = oneBillion) :=
sorry

end NUMINAMATH_CALUDE_one_billion_scientific_notation_l557_55765


namespace NUMINAMATH_CALUDE_solution_pairs_l557_55713

/-- Sum of factorials from 1 to k -/
def sumFactorials (k : ℕ) : ℕ :=
  (List.range k).map Nat.factorial |>.sum

/-- Sum of integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The set of pairs (k, n) that satisfy the equation -/
def solutionSet : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ sumFactorials p.1 = sumIntegers p.2}

theorem solution_pairs : solutionSet = {(1, 1), (2, 2), (5, 17)} := by
  sorry


end NUMINAMATH_CALUDE_solution_pairs_l557_55713


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l557_55718

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l557_55718


namespace NUMINAMATH_CALUDE_product_remainder_l557_55789

theorem product_remainder (a b m : ℕ) (ha : a = 103) (hb : b = 107) (hm : m = 13) :
  (a * b) % m = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l557_55789


namespace NUMINAMATH_CALUDE_dime_exchange_theorem_l557_55776

/-- Represents the number of dimes each person has at each stage -/
structure DimeState :=
  (a : ℤ) (b : ℤ) (c : ℤ)

/-- Represents the transactions between A, B, and C -/
def exchange (state : DimeState) : DimeState :=
  let state1 := DimeState.mk (state.a - state.b - state.c) (2 * state.b) (2 * state.c)
  let state2 := DimeState.mk (2 * state1.a) (state1.b - state1.a - state1.c) (2 * state1.c)
  DimeState.mk (2 * state2.a) (2 * state2.b) (state2.c - state2.a - state2.b)

theorem dime_exchange_theorem (initial : DimeState) :
  exchange initial = DimeState.mk 36 36 36 → initial.a = 36 :=
by sorry

end NUMINAMATH_CALUDE_dime_exchange_theorem_l557_55776


namespace NUMINAMATH_CALUDE_backpack_price_increase_l557_55728

theorem backpack_price_increase 
  (original_backpack_price : ℕ)
  (original_binder_price : ℕ)
  (num_binders : ℕ)
  (binder_price_reduction : ℕ)
  (total_spent : ℕ)
  (h1 : original_backpack_price = 50)
  (h2 : original_binder_price = 20)
  (h3 : num_binders = 3)
  (h4 : binder_price_reduction = 2)
  (h5 : total_spent = 109)
  : ∃ (price_increase : ℕ), 
    original_backpack_price + price_increase + 
    num_binders * (original_binder_price - binder_price_reduction) = total_spent ∧
    price_increase = 5 := by
  sorry

end NUMINAMATH_CALUDE_backpack_price_increase_l557_55728


namespace NUMINAMATH_CALUDE_opposite_gender_selections_l557_55771

def society_size : ℕ := 24
def male_count : ℕ := 14
def female_count : ℕ := 10

theorem opposite_gender_selections :
  (male_count * female_count) + (female_count * male_count) = 280 := by
  sorry

end NUMINAMATH_CALUDE_opposite_gender_selections_l557_55771


namespace NUMINAMATH_CALUDE_subsets_with_three_adjacent_12_chairs_l557_55707

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets of n chairs arranged in a circle
    that contain at least three adjacent chairs -/
def subsets_with_three_adjacent (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of subsets of 12 chairs arranged in a circle
    that contain at least three adjacent chairs is 2056 -/
theorem subsets_with_three_adjacent_12_chairs :
  subsets_with_three_adjacent n = 2056 := by sorry

end NUMINAMATH_CALUDE_subsets_with_three_adjacent_12_chairs_l557_55707


namespace NUMINAMATH_CALUDE_slope_product_is_negative_one_l557_55757

/-- Parabola C: y^2 = 2px (p > 0) passing through (2, 2) -/
def parabola_C (p : ℝ) : Set (ℝ × ℝ) :=
  {point | point.2^2 = 2 * p * point.1 ∧ p > 0}

/-- Point Q on the parabola -/
def point_Q : ℝ × ℝ := (2, 2)

/-- Point M through which the intersecting line passes -/
def point_M : ℝ × ℝ := (2, 0)

/-- Origin O -/
def origin : ℝ × ℝ := (0, 0)

/-- Theorem: The product of slopes of OA and OB is -1 -/
theorem slope_product_is_negative_one
  (p : ℝ)
  (h_p : p > 0)
  (h_Q : point_Q ∈ parabola_C p)
  (A B : ℝ × ℝ)
  (h_A : A ∈ parabola_C p)
  (h_B : B ∈ parabola_C p)
  (h_line : ∃ (m : ℝ), A.1 = m * A.2 + 2 ∧ B.1 = m * B.2 + 2)
  (k1 : ℝ) (h_k1 : k1 = (A.2 - origin.2) / (A.1 - origin.1))
  (k2 : ℝ) (h_k2 : k2 = (B.2 - origin.2) / (B.1 - origin.1)) :
  k1 * k2 = -1 := by sorry

end NUMINAMATH_CALUDE_slope_product_is_negative_one_l557_55757


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_l557_55798

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ) 
  (h1 : square_area = 1600)
  (h2 : rectangle_area = 160)
  (h3 : rectangle_breadth = 10) : 
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_l557_55798


namespace NUMINAMATH_CALUDE_erwin_chocolate_consumption_l557_55733

/-- Represents Erwin's chocolate consumption pattern and total chocolates eaten --/
structure ChocolateConsumption where
  weekday_consumption : ℕ  -- chocolates eaten per weekday
  weekend_consumption : ℕ  -- chocolates eaten per weekend day
  total_chocolates : ℕ     -- total chocolates eaten

/-- Calculates the number of weeks it took to eat all chocolates --/
def weeks_to_finish (consumption : ChocolateConsumption) : ℚ :=
  consumption.total_chocolates / (5 * consumption.weekday_consumption + 2 * consumption.weekend_consumption)

/-- Theorem stating it took Erwin 2 weeks to finish the chocolates --/
theorem erwin_chocolate_consumption :
  let consumption : ChocolateConsumption := {
    weekday_consumption := 2,
    weekend_consumption := 1,
    total_chocolates := 24
  }
  weeks_to_finish consumption = 2 := by sorry

end NUMINAMATH_CALUDE_erwin_chocolate_consumption_l557_55733


namespace NUMINAMATH_CALUDE_part_one_part_two_range_of_m_l557_55712

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
sorry

-- Part 2
theorem part_two :
  ∀ x : ℝ, f 1 x + f 1 (x + 5) ≥ 5 :=
sorry

-- Range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, f 1 x + f 1 (x + 5) ≥ m) ↔ m ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_range_of_m_l557_55712
