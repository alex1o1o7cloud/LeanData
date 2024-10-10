import Mathlib

namespace sequence_contains_intermediate_value_l669_66970

theorem sequence_contains_intermediate_value 
  (n : ℕ) 
  (a : ℕ → ℤ) 
  (A B : ℤ) 
  (h1 : a 1 < A ∧ A < B ∧ B < a n) 
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i < n → a (i + 1) - a i ≤ 1) :
  ∀ C : ℤ, A ≤ C ∧ C ≤ B → ∃ i₀ : ℕ, 1 < i₀ ∧ i₀ < n ∧ a i₀ = C := by
  sorry

end sequence_contains_intermediate_value_l669_66970


namespace circle_triangle_perpendiculars_l669_66904

-- Define the basic structures
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)
structure Circle := (center : Point) (radius : ℝ)

-- Define the triangle
structure Triangle := (A B C : Point)

-- Define the intersection points
structure IntersectionPoints := 
  (A₁ A₂ B₁ B₂ C₁ C₂ : Point)

-- Define a function to check if three lines are concurrent
def are_concurrent (l₁ l₂ l₃ : Line) : Prop := sorry

-- Define a function to create a perpendicular line
def perpendicular_at (l : Line) (p : Point) : Line := sorry

-- Main theorem
theorem circle_triangle_perpendiculars 
  (triangle : Triangle) 
  (circle : Circle) 
  (intersections : IntersectionPoints) : 
  are_concurrent 
    (perpendicular_at (Line.mk 0 1 0) intersections.A₁)
    (perpendicular_at (Line.mk 1 0 0) intersections.B₁)
    (perpendicular_at (Line.mk 1 1 0) intersections.C₁) →
  are_concurrent 
    (perpendicular_at (Line.mk 0 1 0) intersections.A₂)
    (perpendicular_at (Line.mk 1 0 0) intersections.B₂)
    (perpendicular_at (Line.mk 1 1 0) intersections.C₂) :=
by
  sorry

end circle_triangle_perpendiculars_l669_66904


namespace length_of_bd_l669_66984

-- Define the equilateral triangle
def EquilateralTriangle (side_length : ℝ) : Prop :=
  side_length > 0

-- Define points A and C on the sides of the triangle
def PointA (a1 a2 : ℝ) (side_length : ℝ) : Prop :=
  a1 > 0 ∧ a2 > 0 ∧ a1 + a2 = side_length

def PointC (c1 c2 : ℝ) (side_length : ℝ) : Prop :=
  c1 > 0 ∧ c2 > 0 ∧ c1 + c2 = side_length

-- Define the line segment AB and BD
def LineSegments (ab bd : ℝ) : Prop :=
  ab > 0 ∧ bd > 0

-- Theorem statement
theorem length_of_bd
  (side_length : ℝ)
  (a1 a2 c1 c2 ab : ℝ)
  (h1 : EquilateralTriangle side_length)
  (h2 : PointA a1 a2 side_length)
  (h3 : PointC c1 c2 side_length)
  (h4 : LineSegments ab bd)
  (h5 : side_length = 26)
  (h6 : a1 = 3 ∧ a2 = 22)
  (h7 : c1 = 3 ∧ c2 = 23)
  (h8 : ab = 6)
  : bd = 3 := by
  sorry

end length_of_bd_l669_66984


namespace distribute_researchers_count_l669_66963

/-- The number of ways to distribute 4 researchers to 3 schools -/
def distribute_researchers : ℕ :=
  -- Number of ways to divide 4 researchers into 3 groups (one group of 2, two groups of 1)
  (Nat.choose 4 2) *
  -- Number of ways to assign 3 groups to 3 schools
  (Nat.factorial 3)

/-- Theorem stating that the number of distribution schemes is 36 -/
theorem distribute_researchers_count :
  distribute_researchers = 36 := by
  sorry

end distribute_researchers_count_l669_66963


namespace cubic_function_minimum_l669_66996

/-- The function f(x) = x³ - 3x² + 1 reaches its global minimum at x = 2 -/
theorem cubic_function_minimum (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 3*x^2 + 1
  ∀ y : ℝ, f 2 ≤ f y := by sorry

end cubic_function_minimum_l669_66996


namespace certain_number_proof_l669_66903

theorem certain_number_proof (n : ℕ) : 
  (∃ k : ℕ, n = 127 * k + 6) →
  (∃ m : ℕ, 2037 = 127 * m + 5) →
  (∀ d : ℕ, d > 127 → (n % d ≠ 6 ∨ 2037 % d ≠ 5)) →
  n = 2038 := by
sorry

end certain_number_proof_l669_66903


namespace line_slope_through_points_l669_66992

/-- The slope of a line passing through points (1,3) and (5,7) is 1 -/
theorem line_slope_through_points : 
  let x1 : ℝ := 1
  let y1 : ℝ := 3
  let x2 : ℝ := 5
  let y2 : ℝ := 7
  let slope := (y2 - y1) / (x2 - x1)
  slope = 1 := by sorry

end line_slope_through_points_l669_66992


namespace larger_interior_angle_measure_l669_66911

/-- A circular monument consisting of congruent isosceles trapezoids. -/
structure CircularMonument where
  /-- The number of trapezoids in the monument. -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of each trapezoid in degrees. -/
  larger_interior_angle : ℝ

/-- The properties of the circular monument. -/
def monument_properties (m : CircularMonument) : Prop :=
  m.num_trapezoids = 12 ∧
  m.larger_interior_angle > 0 ∧
  m.larger_interior_angle < 180

/-- Theorem stating the measure of the larger interior angle in the monument. -/
theorem larger_interior_angle_measure (m : CircularMonument) 
  (h : monument_properties m) : m.larger_interior_angle = 97.5 := by
  sorry

#check larger_interior_angle_measure

end larger_interior_angle_measure_l669_66911


namespace three_white_balls_probability_l669_66979

/-- The number of white balls in the urn -/
def white_balls : ℕ := 6

/-- The total number of balls in the urn -/
def total_balls : ℕ := 21

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- Probability of drawing 3 white balls without replacement -/
def prob_without_replacement : ℚ := 2 / 133

/-- Probability of drawing 3 white balls with replacement -/
def prob_with_replacement : ℚ := 8 / 343

/-- Probability of drawing 3 white balls simultaneously -/
def prob_simultaneous : ℚ := 2 / 133

/-- Theorem stating the probabilities of drawing 3 white balls under different conditions -/
theorem three_white_balls_probability :
  (Nat.choose white_balls drawn_balls / Nat.choose total_balls drawn_balls : ℚ) = prob_without_replacement ∧
  ((white_balls : ℚ) / total_balls) ^ drawn_balls = prob_with_replacement ∧
  (Nat.choose white_balls drawn_balls / Nat.choose total_balls drawn_balls : ℚ) = prob_simultaneous :=
sorry

end three_white_balls_probability_l669_66979


namespace hex_B1F_to_dec_l669_66987

def hex_to_dec (hex : String) : ℕ :=
  hex.foldr (fun c acc => 16 * acc + 
    match c with
    | 'A' => 10
    | 'B' => 11
    | 'C' => 12
    | 'D' => 13
    | 'E' => 14
    | 'F' => 15
    | _ => c.toNat - '0'.toNat
  ) 0

theorem hex_B1F_to_dec : hex_to_dec "B1F" = 2847 := by
  sorry

end hex_B1F_to_dec_l669_66987


namespace min_stamps_for_50_cents_l669_66985

/-- Represents the number of stamps and their total value -/
structure StampCombination :=
  (threes : ℕ)
  (fours : ℕ)

/-- Calculates the total value of stamps in cents -/
def total_value (s : StampCombination) : ℕ :=
  3 * s.threes + 4 * s.fours

/-- Checks if a stamp combination is valid (totals 50 cents) -/
def is_valid (s : StampCombination) : Prop :=
  total_value s = 50

/-- Theorem: The minimum number of stamps to make 50 cents using 3 cent and 4 cent stamps is 13 -/
theorem min_stamps_for_50_cents :
  ∃ (s : StampCombination), is_valid s ∧
    (∀ (t : StampCombination), is_valid t → s.threes + s.fours ≤ t.threes + t.fours) ∧
    s.threes + s.fours = 13 :=
  sorry

end min_stamps_for_50_cents_l669_66985


namespace largest_package_size_l669_66915

def markers_elliot : ℕ := 60
def markers_tara : ℕ := 36
def markers_sam : ℕ := 90

theorem largest_package_size : ∃ (n : ℕ), n > 0 ∧ 
  markers_elliot % n = 0 ∧ 
  markers_tara % n = 0 ∧ 
  markers_sam % n = 0 ∧
  ∀ (m : ℕ), m > n → 
    (markers_elliot % m = 0 ∧ 
     markers_tara % m = 0 ∧ 
     markers_sam % m = 0) → False :=
by
  -- Proof goes here
  sorry

end largest_package_size_l669_66915


namespace intersection_A_B_union_A_B_l669_66975

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = Set.univ := by sorry

end intersection_A_B_union_A_B_l669_66975


namespace integer_quotient_problem_l669_66930

theorem integer_quotient_problem (x y : ℤ) : 
  1996 * x + y / 96 = x + y → y / x = 2016 ∨ x / y = 1 / 2016 := by
  sorry

end integer_quotient_problem_l669_66930


namespace calculation_result_l669_66966

theorem calculation_result : (0.0077 : ℝ) * 4.5 / (0.05 * 0.1 * 0.007) = 990 := by
  sorry

end calculation_result_l669_66966


namespace consecutive_integers_base_equation_l669_66945

/-- Given two consecutive positive integers A and B that satisfy the equation
    132_A + 43_B = 69_(A+B), prove that A + B = 13 -/
theorem consecutive_integers_base_equation (A B : ℕ) : 
  A > 0 ∧ B > 0 ∧ (B = A + 1 ∨ A = B + 1) →
  (A^2 + 3*A + 2) + (4*B + 3) = 6*(A + B) + 9 →
  A + B = 13 := by
sorry

end consecutive_integers_base_equation_l669_66945


namespace initial_drawer_pencils_count_l669_66916

/-- The number of pencils initially in the drawer -/
def initial_drawer_pencils : ℕ := sorry

/-- The number of pencils initially on the desk -/
def initial_desk_pencils : ℕ := 19

/-- The number of pencils added to the desk -/
def added_desk_pencils : ℕ := 16

/-- The total number of pencils after the addition -/
def total_pencils : ℕ := 78

theorem initial_drawer_pencils_count : 
  initial_drawer_pencils = 43 :=
by sorry

end initial_drawer_pencils_count_l669_66916


namespace order_of_abc_l669_66924

theorem order_of_abc (a b c : ℝ) 
  (h1 : Real.sqrt (1 + 2*a) = Real.exp b)
  (h2 : Real.exp b = 1 / (1 - c))
  (h3 : 1 / (1 - c) = 1.01) : 
  a > b ∧ b > c := by
  sorry

end order_of_abc_l669_66924


namespace triangle_perimeter_l669_66912

theorem triangle_perimeter (a b c : ℝ) : 
  a = 2 ∧ b = 7 ∧ 
  (∃ n : ℕ, c = 2 * n + 1) ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 16 := by
sorry

end triangle_perimeter_l669_66912


namespace sequence_ratio_theorem_l669_66973

theorem sequence_ratio_theorem (d : ℝ) (q : ℚ) :
  d ≠ 0 →
  q > 0 →
  let a : ℕ → ℝ := λ n => d * n
  let b : ℕ → ℝ := λ n => d^2 * q^(n-1)
  ∃ k : ℕ+, (a 1)^2 + (a 2)^2 + (a 3)^2 = k * ((b 1) + (b 2) + (b 3)) →
  q = 2 ∨ q = 1/2 := by
  sorry

end sequence_ratio_theorem_l669_66973


namespace min_students_l669_66967

/-- Represents a student in the math competition -/
structure Student where
  solved : Finset (Fin 6)

/-- Represents the math competition -/
structure MathCompetition where
  students : Finset Student
  problem_count : Nat
  students_per_problem : Nat

/-- The conditions of the math competition -/
def validCompetition (c : MathCompetition) : Prop :=
  c.problem_count = 6 ∧
  c.students_per_problem = 500 ∧
  (∀ p : Fin 6, (c.students.filter (fun s => p ∈ s.solved)).card = c.students_per_problem) ∧
  (∀ s₁ s₂ : Student, s₁ ∈ c.students → s₂ ∈ c.students → s₁ ≠ s₂ → 
    ∃ p : Fin 6, p ∉ s₁.solved ∧ p ∉ s₂.solved)

/-- The theorem to be proved -/
theorem min_students (c : MathCompetition) (h : validCompetition c) : 
  c.students.card ≥ 1000 := by
  sorry

end min_students_l669_66967


namespace product_of_primes_sum_99_l669_66999

theorem product_of_primes_sum_99 (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 99 → p * q = 194 := by sorry

end product_of_primes_sum_99_l669_66999


namespace sum_of_altitudes_l669_66959

/-- The sum of altitudes of a triangle formed by the line 10x + 8y = 80 and the coordinate axes --/
theorem sum_of_altitudes (x y : ℝ) : 
  (10 * x + 8 * y = 80) →
  (∃ (a b c : ℝ), 
    (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) ∧
    (10 * a + 8 * b = 80) ∧
    (a + b + c = (18 * Real.sqrt 41 + 40) / Real.sqrt 41) ∧
    (c = 40 / Real.sqrt 41)) := by
  sorry

end sum_of_altitudes_l669_66959


namespace sum_of_seven_thirds_l669_66957

theorem sum_of_seven_thirds (x : ℚ) : 
  x = 1 / 3 → x + x + x + x + x + x + x = 7 * (1 / 3) := by
  sorry

end sum_of_seven_thirds_l669_66957


namespace largest_square_area_l669_66910

theorem largest_square_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  a^2 + b^2 + c^2 = 450 →  -- sum of square areas
  a^2 = 100 →  -- area of square on AB
  c^2 = 225 :=  -- area of largest square (on BC)
by
  sorry

end largest_square_area_l669_66910


namespace ellipse_eccentricity_range_l669_66988

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e,
    if there exists a point P on the ellipse such that the angle F₁PF₂ is 60°,
    then the eccentricity e is in the range [1/2, 1). -/
theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∃ (e : ℝ), e = Real.sqrt (1 - b^2 / a^2) ∧
    ∃ (F1 F2 : ℝ × ℝ),
      F1 = (-a * e, 0) ∧ F2 = (a * e, 0) ∧
      Real.cos (60 * π / 180) = ((x - F1.1)^2 + (y - F1.2)^2 + (x - F2.1)^2 + (y - F2.2)^2 - 4 * a^2 * e^2) /
        (2 * Real.sqrt ((x - F1.1)^2 + (y - F1.2)^2) * Real.sqrt ((x - F2.1)^2 + (y - F2.2)^2))) →
  1/2 ≤ e ∧ e < 1 :=
sorry

end ellipse_eccentricity_range_l669_66988


namespace quadratic_inequality_solution_fraction_inequality_solution_l669_66993

-- Part 1
def quadratic_inequality (x : ℝ) : Prop := x^2 + 3*x - 4 > 0

theorem quadratic_inequality_solution :
  ∀ x : ℝ, quadratic_inequality x ↔ (x > 1 ∨ x < -4) :=
by sorry

-- Part 2
def fraction_inequality (x : ℝ) : Prop := x ≠ 5 ∧ (1 - x) / (x - 5) ≥ 1

theorem fraction_inequality_solution :
  ∀ x : ℝ, fraction_inequality x ↔ (3 ≤ x ∧ x < 5) :=
by sorry

end quadratic_inequality_solution_fraction_inequality_solution_l669_66993


namespace final_position_l669_66990

-- Define the ant's position type
def Position := ℤ × ℤ

-- Define the direction type
inductive Direction
| East
| North
| West
| South

-- Define the function to get the next direction
def nextDirection (d : Direction) : Direction :=
  match d with
  | Direction.East => Direction.North
  | Direction.North => Direction.West
  | Direction.West => Direction.South
  | Direction.South => Direction.East

-- Define the function to move in a direction
def move (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.East => (p.1 + 1, p.2)
  | Direction.North => (p.1, p.2 + 1)
  | Direction.West => (p.1 - 1, p.2)
  | Direction.South => (p.1, p.2 - 1)

-- Define the function to calculate the position after n steps
def positionAfterSteps (n : ℕ) : Position :=
  sorry -- Proof implementation goes here

-- The main theorem
theorem final_position : positionAfterSteps 2015 = (13, -22) := by
  sorry -- Proof implementation goes here

end final_position_l669_66990


namespace cat_food_consumption_l669_66960

/-- Represents the amount of food eaten by the cat each day -/
def daily_consumption : ℚ := 1/3 + 1/4

/-- Represents the total number of cans available -/
def total_cans : ℚ := 6

/-- Represents the day on which the cat finishes all the food -/
def finish_day : ℕ := 4

theorem cat_food_consumption :
  ∃ (n : ℕ), n * daily_consumption > total_cans ∧ (n - 1) * daily_consumption ≤ total_cans ∧ n = finish_day :=
by sorry

end cat_food_consumption_l669_66960


namespace square_root_of_121_l669_66991

theorem square_root_of_121 : ∀ x : ℝ, x^2 = 121 ↔ x = 11 ∨ x = -11 := by sorry

end square_root_of_121_l669_66991


namespace cube_sum_given_sum_and_product_l669_66922

theorem cube_sum_given_sum_and_product (a b : ℝ) 
  (h1 : a + b = 11) (h2 : a * b = 20) : 
  a^3 + b^3 = 671 := by sorry

end cube_sum_given_sum_and_product_l669_66922


namespace pencils_count_l669_66964

/-- The number of pencils originally in the jar -/
def original_pencils : ℕ := 87

/-- The number of pencils removed from the jar -/
def removed_pencils : ℕ := 4

/-- The number of pencils left in the jar after removal -/
def remaining_pencils : ℕ := 83

/-- Theorem stating that the original number of pencils equals the sum of removed and remaining pencils -/
theorem pencils_count : original_pencils = removed_pencils + remaining_pencils := by
  sorry

end pencils_count_l669_66964


namespace least_subtraction_for_divisibility_l669_66953

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 1415 ∧ 
  (2500000 - n) % 1423 = 0 ∧ 
  ∀ (m : ℕ), m < n → (2500000 - m) % 1423 ≠ 0 := by
  sorry

end least_subtraction_for_divisibility_l669_66953


namespace consumption_increase_after_tax_reduction_l669_66951

/-- 
Given a commodity with tax and consumption, prove that if the tax is reduced by 20% 
and the revenue decreases by 8%, then the consumption must have increased by 15%.
-/
theorem consumption_increase_after_tax_reduction (T C : ℝ) 
  (h1 : T > 0) (h2 : C > 0) : 
  (0.80 * T) * (C * (1 + 15/100)) = 0.92 * (T * C) := by
  sorry

end consumption_increase_after_tax_reduction_l669_66951


namespace inequality_proof_l669_66913

theorem inequality_proof (x y z t : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ t ≥ 0) 
  (h_sum : x + y + z + t = 4) : 
  Real.sqrt (x^2 + t^2) + Real.sqrt (z^2 + 1) + Real.sqrt (z^2 + t^2) + 
  Real.sqrt (y^2 + x^2) + Real.sqrt (y^2 + 64) ≥ 13 := by
  sorry

end inequality_proof_l669_66913


namespace geometric_sequence_11th_term_l669_66935

/-- Given a geometric sequence where a₅ = 16 and a₈ = 4√2, prove that a₁₁ = 2√2 -/
theorem geometric_sequence_11th_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n) ^ m)
  (a5 : a 5 = 16)
  (a8 : a 8 = 4 * Real.sqrt 2) :
  a 11 = 2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_11th_term_l669_66935


namespace concept_laws_theorem_l669_66914

/-- Probability of M laws being included in the Concept -/
def prob_M_laws_included (K N M : ℕ) (p : ℝ) : ℝ :=
  Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- Expected number of laws included in the Concept -/
def expected_laws_included (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

/-- Theorem stating the probability of M laws being included and the expected number of laws -/
theorem concept_laws_theorem (K N M : ℕ) (p : ℝ) 
    (hK : K > 0) (hN : N > 0) (hM : M ≤ K) (hp : 0 ≤ p ∧ p ≤ 1) :
  prob_M_laws_included K N M p = Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M) ∧
  expected_laws_included K N p = K * (1 - (1 - p)^N) := by
  sorry

end concept_laws_theorem_l669_66914


namespace function_inequality_implies_b_bound_l669_66944

open Real

theorem function_inequality_implies_b_bound (b : ℝ) :
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, exp x * (x - b) + x * exp x * (x + 1 - b) > 0) →
  b < 8/3 := by
sorry

end function_inequality_implies_b_bound_l669_66944


namespace fraction_comparison_l669_66943

theorem fraction_comparison (a b m : ℝ) (ha : a > b) (hb : b > 0) (hm : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end fraction_comparison_l669_66943


namespace complex_fraction_simplification_l669_66997

/-- Given a complex number i such that i^2 = -1, 
    prove that (2-i)/(1+4i) = -2/17 - (9/17)i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + 4*i) = -2/17 - (9/17)*i := by sorry

end complex_fraction_simplification_l669_66997


namespace population_after_two_years_l669_66980

def initial_population : ℕ := 1000
def year1_increase : ℚ := 20 / 100
def year2_increase : ℚ := 30 / 100

theorem population_after_two_years :
  let year1_population := initial_population * (1 + year1_increase)
  let year2_population := year1_population * (1 + year2_increase)
  ↑(round year2_population) = 1560 := by sorry

end population_after_two_years_l669_66980


namespace total_pay_is_550_l669_66928

/-- The total weekly pay for two employees, where one is paid 150% of the other -/
def total_weekly_pay (b_pay : ℚ) : ℚ :=
  b_pay + (150 / 100) * b_pay

/-- Theorem: Given B is paid 220 per week, the total pay for both employees is 550 -/
theorem total_pay_is_550 : total_weekly_pay 220 = 550 := by
  sorry

end total_pay_is_550_l669_66928


namespace lcm_of_9_12_15_l669_66908

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end lcm_of_9_12_15_l669_66908


namespace cube_sum_simplification_l669_66907

theorem cube_sum_simplification (a b c : ℕ) (ha : a = 43) (hb : b = 26) (hc : c = 17) :
  (a^3 + c^3) / (a^3 + b^3) = (a + c) / (a + b) := by
  sorry

end cube_sum_simplification_l669_66907


namespace frankie_pet_count_l669_66958

/-- Represents the number of pets Frankie has of each type -/
structure PetCounts where
  dogs : Nat
  cats : Nat
  parrots : Nat
  snakes : Nat

/-- Calculates the total number of pets -/
def totalPets (p : PetCounts) : Nat :=
  p.dogs + p.cats + p.parrots + p.snakes

/-- Represents the conditions given in the problem -/
structure PetConditions (p : PetCounts) : Prop where
  dog_count : p.dogs = 2
  four_legged : p.dogs + p.cats = 6
  parrot_count : p.parrots = p.cats - 1
  snake_count : p.snakes = p.cats + 6

/-- Theorem stating that given the conditions, Frankie has 19 pets in total -/
theorem frankie_pet_count (p : PetCounts) (h : PetConditions p) : totalPets p = 19 := by
  sorry


end frankie_pet_count_l669_66958


namespace positive_correlation_missing_data_point_l669_66936

-- Define the regression line
def regression_line (x : ℝ) : ℝ := 6.5 * x + 17.5

-- Define the data points
def data_points : List (ℝ × ℝ) := [(2, 30), (4, 40), (5, 60), (6, 50), (8, 70)]

-- Theorem 1: Positive correlation
theorem positive_correlation : 
  ∀ x₁ x₂, x₁ < x₂ → regression_line x₁ < regression_line x₂ :=
by sorry

-- Theorem 2: Missing data point
theorem missing_data_point : 
  ∃ y, (2, y) ∈ data_points ∧ y = 30 :=
by sorry

end positive_correlation_missing_data_point_l669_66936


namespace subset_implies_a_geq_two_l669_66952

def A : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - (a+1)*x + a ≤ 0}

theorem subset_implies_a_geq_two (a : ℝ) (h : A ⊆ B a) : a ≥ 2 := by
  sorry

end subset_implies_a_geq_two_l669_66952


namespace point_distributive_l669_66978

/-- Addition of two points in the plane -/
noncomputable def point_add (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Multiplication (midpoint) of two points in the plane -/
noncomputable def point_mul (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: A × (B + C) = (B + A) × (A + C) for any three points A, B, C in the plane -/
theorem point_distributive (A B C : ℝ × ℝ) :
  point_mul A (point_add B C) = point_mul (point_add B A) (point_add A C) :=
sorry

end point_distributive_l669_66978


namespace largest_prime_factor_largest_prime_factor_of_expression_l669_66919

theorem largest_prime_factor (n : ℕ) : ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p :=
  sorry

theorem largest_prime_factor_of_expression : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (18^3 + 12^4 - 6^5) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (18^3 + 12^4 - 6^5) → q ≤ p ∧ p = 23 :=
sorry

end largest_prime_factor_largest_prime_factor_of_expression_l669_66919


namespace f_minimum_and_g_inequality_l669_66965

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a * x + x * Real.cos x + 1)

theorem f_minimum_and_g_inequality :
  (∃! x : ℝ, ∀ y : ℝ, f y ≥ f x) ∧ f 0 = 0 ∧
  ∀ a : ℝ, a > -1 → ∀ x : ℝ, x > 0 ∧ x < 1 → g a x > 1 :=
sorry

end f_minimum_and_g_inequality_l669_66965


namespace combined_resistance_of_parallel_resistors_l669_66948

def parallel_resistance (r1 r2 r3 : ℚ) : ℚ := 1 / (1 / r1 + 1 / r2 + 1 / r3)

theorem combined_resistance_of_parallel_resistors :
  let r1 : ℚ := 2
  let r2 : ℚ := 5
  let r3 : ℚ := 6
  let r : ℚ := parallel_resistance r1 r2 r3
  r = 15 / 13 := by sorry

end combined_resistance_of_parallel_resistors_l669_66948


namespace annie_initial_money_l669_66982

/-- The amount of money Annie had initially -/
def initial_money : ℕ := 132

/-- The price of a hamburger -/
def hamburger_price : ℕ := 4

/-- The price of a milkshake -/
def milkshake_price : ℕ := 5

/-- The number of hamburgers Annie bought -/
def hamburgers_bought : ℕ := 8

/-- The number of milkshakes Annie bought -/
def milkshakes_bought : ℕ := 6

/-- The amount of money Annie had left -/
def money_left : ℕ := 70

theorem annie_initial_money :
  initial_money = 
    hamburger_price * hamburgers_bought + 
    milkshake_price * milkshakes_bought + 
    money_left :=
by sorry

end annie_initial_money_l669_66982


namespace helen_cookies_l669_66938

/-- The total number of chocolate chip cookies Helen baked -/
def total_cookies (yesterday today : ℕ) : ℕ := yesterday + today

/-- Theorem stating that Helen baked 1081 chocolate chip cookies in total -/
theorem helen_cookies : total_cookies 527 554 = 1081 := by
  sorry

end helen_cookies_l669_66938


namespace equation_transformation_l669_66942

theorem equation_transformation (x : ℝ) (y : ℝ) (h1 : x ≠ 0) (h2 : x^2 ≠ 2) :
  y = (x^2 - 2) / x ∧ (x^2 - 2) / x + 2 * x / (x^2 - 2) = 5 → y^2 - 5*y + 2 = 0 := by
  sorry

end equation_transformation_l669_66942


namespace F_of_2_f_of_3_equals_15_l669_66946

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 2*a
def F (a b : ℝ) : ℝ := b^2 + a*b

-- State the theorem
theorem F_of_2_f_of_3_equals_15 : F 2 (f 3) = 15 := by
  sorry

end F_of_2_f_of_3_equals_15_l669_66946


namespace complex_multiplication_sum_l669_66931

theorem complex_multiplication_sum (z : ℂ) (a b : ℝ) :
  z = 5 + 3 * I →
  I * z = a + b * I →
  a + b = 2 := by
sorry

end complex_multiplication_sum_l669_66931


namespace inequality_theorem_l669_66969

theorem inequality_theorem (p q r : ℝ) (n : ℕ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0) (h_pqr : p * q * r = 1) : 
  (1 / (p^n + q^n + 1)) + (1 / (q^n + r^n + 1)) + (1 / (r^n + p^n + 1)) ≤ 1 := by
  sorry

end inequality_theorem_l669_66969


namespace triangle_area_bounds_l669_66933

def triangle_area (s : ℝ) : ℝ := (s + 2)^(4/3)

theorem triangle_area_bounds :
  ∀ s : ℝ, (2^(1/2) * 3^(1/4) ≤ s ∧ s ≤ 3^(2/3) * 2^(1/3)) ↔
    (12 ≤ triangle_area s ∧ triangle_area s ≤ 72) :=
by sorry

end triangle_area_bounds_l669_66933


namespace sally_buttons_count_l669_66906

/-- The number of buttons Sally needs for all shirts -/
def total_buttons (monday_shirts tuesday_shirts wednesday_shirts buttons_per_shirt : ℕ) : ℕ :=
  (monday_shirts + tuesday_shirts + wednesday_shirts) * buttons_per_shirt

/-- Theorem stating that Sally needs 45 buttons for all shirts -/
theorem sally_buttons_count : total_buttons 4 3 2 5 = 45 := by
  sorry

end sally_buttons_count_l669_66906


namespace half_plus_five_equals_fifteen_l669_66986

theorem half_plus_five_equals_fifteen (n : ℝ) : (1/2) * n + 5 = 15 → n = 20 := by
  sorry

end half_plus_five_equals_fifteen_l669_66986


namespace no_valid_operation_l669_66998

-- Define the set of standard arithmetic operations
inductive ArithOp
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an arithmetic operation
def applyOp (op : ArithOp) (a b : ℤ) : ℚ :=
  match op with
  | ArithOp.Add => a + b
  | ArithOp.Sub => a - b
  | ArithOp.Mul => a * b
  | ArithOp.Div => (a : ℚ) / (b : ℚ)

-- Theorem statement
theorem no_valid_operation : ∀ (op : ArithOp), 
  (applyOp op 9 3 : ℚ) + 5 - (4 - 2) ≠ 7 := by
  sorry

end no_valid_operation_l669_66998


namespace cookies_eaten_l669_66921

theorem cookies_eaten (charlie_cookies : ℕ) (father_cookies : ℕ) (mother_cookies : ℕ)
  (h1 : charlie_cookies = 15)
  (h2 : father_cookies = 10)
  (h3 : mother_cookies = 5) :
  charlie_cookies + father_cookies + mother_cookies = 30 := by
sorry

end cookies_eaten_l669_66921


namespace angle_C_measure_l669_66994

/-- Given a triangle ABC, if 3 sin A + 4 cos B = 6 and 3 cos A + 4 sin B = 1, then ∠C = π/6 -/
theorem angle_C_measure (A B C : ℝ) (hsum : A + B + C = π) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 3 * Real.cos A + 4 * Real.sin B = 1) : C = π / 6 := by
  sorry

end angle_C_measure_l669_66994


namespace second_quadrant_angle_ratio_l669_66918

theorem second_quadrant_angle_ratio (x : Real) : 
  (π/2 < x) ∧ (x < π) →  -- x is in the second quadrant
  (Real.tan x)^2 + 3*(Real.tan x) - 4 = 0 → 
  (Real.sin x + Real.cos x) / (2*(Real.sin x) - Real.cos x) = 1/3 := by
sorry

end second_quadrant_angle_ratio_l669_66918


namespace second_tree_height_l669_66971

/-- Given two trees casting shadows under the same conditions, 
    this theorem calculates the height of the second tree. -/
theorem second_tree_height
  (h1 : ℝ) -- Height of the first tree
  (s1 : ℝ) -- Shadow length of the first tree
  (s2 : ℝ) -- Shadow length of the second tree
  (h1_positive : h1 > 0)
  (s1_positive : s1 > 0)
  (s2_positive : s2 > 0)
  (h1_value : h1 = 28)
  (s1_value : s1 = 30)
  (s2_value : s2 = 45) :
  ∃ (h2 : ℝ), h2 = 42 ∧ h2 / s2 = h1 / s1 := by
  sorry


end second_tree_height_l669_66971


namespace max_value_quadratic_constraint_l669_66917

theorem max_value_quadratic_constraint (x y z w : ℝ) 
  (h : 9*x^2 + 4*y^2 + 25*z^2 + 16*w^2 = 4) : 
  (∃ (a b c d : ℝ), 9*a^2 + 4*b^2 + 25*c^2 + 16*d^2 = 4 ∧ 
  2*a + 3*b + 5*c - 4*d = 6*Real.sqrt 6) ∧ 
  (∀ (x y z w : ℝ), 9*x^2 + 4*y^2 + 25*z^2 + 16*w^2 = 4 → 
  2*x + 3*y + 5*z - 4*w ≤ 6*Real.sqrt 6) := by
sorry

end max_value_quadratic_constraint_l669_66917


namespace solution_existence_conditions_l669_66983

theorem solution_existence_conditions (a b : ℝ) :
  (∃ x y : ℝ, (Real.tan x) * (Real.tan y) = a ∧ (Real.sin x)^2 + (Real.sin y)^2 = b^2) ↔
  (1 < b^2 ∧ b^2 < 2*a/(a+1)) ∨ (1 < b^2 ∧ b^2 < 2*a/(a-1)) := by
  sorry

end solution_existence_conditions_l669_66983


namespace jane_is_26_l669_66955

/-- Given Danny's current age and the age difference between Danny and Jane 19 years ago,
    calculates Jane's current age. -/
def janes_current_age (dannys_current_age : ℕ) (years_ago : ℕ) : ℕ :=
  let dannys_age_then := dannys_current_age - years_ago
  let janes_age_then := dannys_age_then / 3
  janes_age_then + years_ago

/-- Proves that Jane's current age is 26, given the problem conditions. -/
theorem jane_is_26 :
  janes_current_age 40 19 = 26 := by
  sorry


end jane_is_26_l669_66955


namespace sum_and_count_theorem_l669_66934

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  let x := sum_integers 40 60
  let y := count_even_integers 40 60
  x + y = 1061 := by sorry

end sum_and_count_theorem_l669_66934


namespace unique_positive_zero_iff_a_lt_neg_two_l669_66940

/-- The function f(x) = ax³ - 3x² + 1 has a unique positive zero if and only if a ∈ (-∞, -2) -/
theorem unique_positive_zero_iff_a_lt_neg_two (a : ℝ) :
  (∃! x : ℝ, x > 0 ∧ a * x^3 - 3 * x^2 + 1 = 0) ↔ a < -2 :=
by sorry

end unique_positive_zero_iff_a_lt_neg_two_l669_66940


namespace total_schedules_l669_66949

/-- Represents the number of classes to be scheduled -/
def num_classes : ℕ := 4

/-- Represents the number of classes that can be scheduled in the first period -/
def first_period_options : ℕ := 3

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: The total number of different possible schedules is 18 -/
theorem total_schedules : 
  first_period_options * factorial (num_classes - 1) = 18 := by
  sorry

end total_schedules_l669_66949


namespace power_five_mod_150_l669_66905

theorem power_five_mod_150 : 5^2023 % 150 = 5 := by
  sorry

end power_five_mod_150_l669_66905


namespace zeros_imply_a_range_l669_66950

theorem zeros_imply_a_range (a : ℝ) : 
  (∃ x y, x ∈ (Set.Ioo 0 1) ∧ y ∈ (Set.Ioo 1 2) ∧ 
    x^2 - 2*a*x + 1 = 0 ∧ y^2 - 2*a*y + 1 = 0) → 
  a ∈ (Set.Ioo 1 (5/4)) := by
sorry

end zeros_imply_a_range_l669_66950


namespace M_greater_than_N_l669_66989

theorem M_greater_than_N : ∀ a : ℝ, 2 * a * (a - 2) > (a + 1) * (a - 3) := by
  sorry

end M_greater_than_N_l669_66989


namespace eriks_remaining_money_is_43_47_l669_66902

/-- Calculates the amount of money Erik has left after his purchase --/
def eriks_remaining_money (initial_amount : ℚ) (bread_price carton_price egg_price chocolate_price : ℚ)
  (bread_quantity carton_quantity egg_quantity chocolate_quantity : ℕ)
  (discount_rate tax_rate : ℚ) : ℚ :=
  let total_cost := bread_price * bread_quantity + carton_price * carton_quantity +
                    egg_price * egg_quantity + chocolate_price * chocolate_quantity
  let discounted_cost := total_cost * (1 - discount_rate)
  let final_cost := discounted_cost * (1 + tax_rate)
  initial_amount - final_cost

/-- Theorem stating that Erik has $43.47 left after his purchase --/
theorem eriks_remaining_money_is_43_47 :
  eriks_remaining_money 86 3 6 4 2 3 3 2 5 (1/10) (1/20) = 43.47 := by
  sorry

end eriks_remaining_money_is_43_47_l669_66902


namespace robot_gloves_rings_arrangements_l669_66962

/-- Represents the number of arms of the robot -/
def num_arms : ℕ := 6

/-- Represents the total number of items (gloves and rings) -/
def total_items : ℕ := 2 * num_arms

/-- Represents the number of valid arrangements for putting on gloves and rings -/
def valid_arrangements : ℕ := (Nat.factorial total_items) / (2^num_arms)

/-- Theorem stating the number of valid arrangements for the robot to put on gloves and rings -/
theorem robot_gloves_rings_arrangements :
  valid_arrangements = (Nat.factorial total_items) / (2^num_arms) :=
by sorry

end robot_gloves_rings_arrangements_l669_66962


namespace carton_width_l669_66901

/-- Represents the dimensions of a rectangular carton -/
structure CartonDimensions where
  length : ℝ
  width : ℝ

/-- Given a carton with dimensions 25 inches by 60 inches, its width is 25 inches -/
theorem carton_width (c : CartonDimensions) 
  (h1 : c.length = 60) 
  (h2 : c.width = 25) : 
  c.width = 25 := by
  sorry

end carton_width_l669_66901


namespace max_a_inequality_max_a_equality_l669_66976

theorem max_a_inequality (a : ℝ) : 
  (∀ x > 0, Real.log (a * x) + a * x ≤ x + Real.exp x) → a ≤ Real.exp 1 := by
  sorry

theorem max_a_equality : 
  ∃ a : ℝ, a = Real.exp 1 ∧ (∀ x > 0, Real.log (a * x) + a * x ≤ x + Real.exp x) := by
  sorry

end max_a_inequality_max_a_equality_l669_66976


namespace midpoint_of_complex_line_segment_l669_66977

theorem midpoint_of_complex_line_segment :
  let z₁ : ℂ := -5 + 7*I
  let z₂ : ℂ := 9 - 3*I
  let midpoint := (z₁ + z₂) / 2
  midpoint = 2 + 2*I := by
  sorry

end midpoint_of_complex_line_segment_l669_66977


namespace locus_is_extended_rectangle_l669_66941

/-- A line in a plane --/
structure Line where
  -- We assume some representation of a line
  mk :: (dummy : Unit)

/-- Distance between a point and a line --/
noncomputable def dist (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- The locus of points with constant difference of distances from two lines --/
def locus (l₁ l₂ : Line) (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |dist p l₁ - dist p l₂| = a}

/-- A rectangle in a plane --/
structure Rectangle where
  -- We assume some representation of a rectangle
  mk :: (dummy : Unit)

/-- The sides of a rectangle extended infinitely --/
def extended_sides (r : Rectangle) : Set (ℝ × ℝ) := sorry

/-- Construct a rectangle from two lines and a distance --/
def construct_rectangle (l₁ l₂ : Line) (a : ℝ) : Rectangle := sorry

/-- The main theorem --/
theorem locus_is_extended_rectangle (l₁ l₂ : Line) (a : ℝ) :
  locus l₁ l₂ a = extended_sides (construct_rectangle l₁ l₂ a) := by sorry

end locus_is_extended_rectangle_l669_66941


namespace solution_set_f_positive_range_of_m_l669_66920

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1/3 ∨ x > 3} := by sorry

-- Theorem for part II
theorem range_of_m (h : ∃ x₀ : ℝ, f x₀ + 2*m^2 < 4*m) :
  -1/2 < m ∧ m < 5/2 := by sorry

end solution_set_f_positive_range_of_m_l669_66920


namespace stacy_paper_pages_per_day_l669_66981

/-- Given a paper with a certain number of pages and a number of days to complete it,
    calculate the number of pages that need to be written per day. -/
def pagesPerDay (totalPages : ℕ) (days : ℕ) : ℕ :=
  totalPages / days

theorem stacy_paper_pages_per_day :
  pagesPerDay 33 3 = 11 := by
  sorry

end stacy_paper_pages_per_day_l669_66981


namespace sandy_clothes_cost_l669_66900

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℚ := 13.99

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℚ := 12.14

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℚ := 7.43

/-- The total amount Sandy spent on clothes -/
def total_cost : ℚ := shorts_cost + shirt_cost + jacket_cost

/-- Theorem stating that the total amount Sandy spent on clothes is $33.56 -/
theorem sandy_clothes_cost : total_cost = 33.56 := by
  sorry

end sandy_clothes_cost_l669_66900


namespace square_difference_identity_simplify_expression_l669_66926

theorem square_difference_identity (a b : ℝ) : (a - b)^2 = a^2 + b^2 - 2*a*b := by sorry

theorem simplify_expression : 2021^2 - 2021 * 4034 + 2017^2 = 16 := by
  have h : ∀ (x y : ℝ), (x - y)^2 = x^2 + y^2 - 2*x*y := square_difference_identity
  sorry

end square_difference_identity_simplify_expression_l669_66926


namespace cubic_root_product_theorem_l669_66995

/-- The cubic polynomial x^3 - 2x^2 + x + k -/
def cubic (k : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + x + k

/-- The condition that the product of roots equals the square of the difference between max and min real roots -/
def root_product_condition (k : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (∀ x, cubic k x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    a * b * c = (max a (max b c) - min a (min b c))^2

theorem cubic_root_product_theorem : 
  ∀ k : ℝ, root_product_condition k ↔ k = -2 :=
sorry

end cubic_root_product_theorem_l669_66995


namespace donut_distribution_l669_66954

/-- The number of ways to distribute n indistinguishable objects among k distinct boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem donut_distribution :
  let n : ℕ := 3  -- number of additional donuts to distribute
  let k : ℕ := 5  -- number of donut kinds
  distribute n k = choose (n + k - 1) n ∧
  choose (n + k - 1) n = 35 :=
by sorry

end donut_distribution_l669_66954


namespace zinc_copper_mixture_weight_l669_66927

/-- Proves that the weight of a zinc-copper mixture is 74 kg given the specified conditions -/
theorem zinc_copper_mixture_weight :
  ∀ (zinc copper total : ℝ),
  zinc = 33.3 →
  zinc / copper = 9 / 11 →
  total = zinc + copper →
  total = 74 := by
sorry

end zinc_copper_mixture_weight_l669_66927


namespace sheep_purchase_equation_l669_66929

/-- Represents a group of people jointly buying sheep -/
structure SheepPurchase where
  x : ℕ  -- number of people
  price : ℕ  -- price of the sheep

/-- The equation holds for the given conditions -/
theorem sheep_purchase_equation (sp : SheepPurchase) : 
  (5 * sp.x + 45 = sp.price) ∧ (7 * sp.x - 3 = sp.price) → 5 * sp.x + 45 = 7 * sp.x + 3 :=
by
  sorry

end sheep_purchase_equation_l669_66929


namespace prohibited_items_most_suitable_for_census_l669_66923

/-- Represents a survey type -/
inductive SurveyType
  | CrashResistance
  | ProhibitedItems
  | AppleSweetness
  | WetlandSpecies

/-- Determines if a survey type is suitable for a census -/
def isSuitableForCensus (survey : SurveyType) : Prop :=
  match survey with
  | .ProhibitedItems => true
  | _ => false

/-- Theorem: The survey about prohibited items on high-speed trains is the most suitable for a census -/
theorem prohibited_items_most_suitable_for_census :
  ∀ (survey : SurveyType), isSuitableForCensus survey → survey = SurveyType.ProhibitedItems :=
by
  sorry

#check prohibited_items_most_suitable_for_census

end prohibited_items_most_suitable_for_census_l669_66923


namespace max_different_ages_is_17_l669_66909

/-- Represents the problem of finding the maximum number of different ages within a range --/
def MaxDifferentAges (average : ℕ) (stdDev : ℕ) : ℕ :=
  (average + stdDev) - (average - stdDev) + 1

/-- Theorem stating that for the given conditions, the maximum number of different ages is 17 --/
theorem max_different_ages_is_17 (average : ℕ) (stdDev : ℕ)
    (h_average : average = 20)
    (h_stdDev : stdDev = 8) :
    MaxDifferentAges average stdDev = 17 := by
  sorry

#eval MaxDifferentAges 20 8  -- Should output 17

end max_different_ages_is_17_l669_66909


namespace symmetric_points_sqrt_l669_66939

/-- Given that point P(3, -1) is symmetric to point Q(a+b, 1-b) about the y-axis,
    prove that the square root of -ab equals √10. -/
theorem symmetric_points_sqrt (a b : ℝ) : 
  (3 = -(a + b) ∧ -1 = 1 - b) → Real.sqrt (-a * b) = Real.sqrt 10 := by
  sorry

end symmetric_points_sqrt_l669_66939


namespace d17_value_l669_66947

def is_divisor_of (d n : ℕ) : Prop := n % d = 0

theorem d17_value (n : ℕ) (d : ℕ → ℕ) :
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 17 → d i < d j) →
  (∀ i, 1 ≤ i ∧ i ≤ 17 → is_divisor_of (d i) n) →
  d 1 = 1 →
  (d 7)^2 + (d 15)^2 = (d 16)^2 →
  d 17 = 28 :=
sorry

end d17_value_l669_66947


namespace bulls_and_heat_wins_l669_66932

/-- The number of games won by the Chicago Bulls and Miami Heat combined in 2010 -/
theorem bulls_and_heat_wins (bulls_wins : ℕ) (heat_wins : ℕ) : 
  bulls_wins = 70 →
  heat_wins = bulls_wins + 5 →
  bulls_wins + heat_wins = 145 := by
  sorry

end bulls_and_heat_wins_l669_66932


namespace percentage_within_one_std_dev_l669_66937

-- Define a symmetric distribution
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percentage_less_than_mean_plus_std : ℝ

-- Theorem statement
theorem percentage_within_one_std_dev 
  (d : SymmetricDistribution) 
  (h1 : d.is_symmetric = true) 
  (h2 : d.percentage_less_than_mean_plus_std = 84) : 
  (d.percentage_less_than_mean_plus_std - (100 - d.percentage_less_than_mean_plus_std)) = 68 := by
  sorry

end percentage_within_one_std_dev_l669_66937


namespace tree_height_difference_l669_66972

-- Define the heights of the trees
def birch_height : ℚ := 12 + 1/4
def maple_height : ℚ := 20 + 2/5

-- Define the height difference
def height_difference : ℚ := maple_height - birch_height

-- Theorem to prove
theorem tree_height_difference :
  height_difference = 8 + 3/20 := by sorry

end tree_height_difference_l669_66972


namespace triangle_abc_properties_l669_66956

/-- Given a triangle ABC with the following properties:
  - b = √2
  - c = 3
  - B + C = 3A
  Prove the following:
  1. a = √5
  2. sin(B + 3π/4) = √10/10
-/
theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  b = Real.sqrt 2 →
  c = 3 →
  B + C = 3 * A →
  a = Real.sqrt 5 ∧ Real.sin (B + 3 * Real.pi / 4) = Real.sqrt 10 / 10 := by
  sorry

end triangle_abc_properties_l669_66956


namespace hyperbola_equation_l669_66968

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 64

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 0

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 36 - y^2 / 12 = 1

-- Theorem statement
theorem hyperbola_equation 
  (h1 : ∀ x y, ellipse x y ↔ hyperbola x y)  -- Same foci condition
  (h2 : ∃ x y, hyperbola x y ∧ asymptote x y)  -- Asymptote condition
  : ∀ x y, hyperbola x y ↔ x^2 / 36 - y^2 / 12 = 1 :=
by sorry

end hyperbola_equation_l669_66968


namespace water_tank_capacity_l669_66974

theorem water_tank_capacity (w c : ℝ) (h1 : w / c = 1 / 5) (h2 : (w + 3) / c = 1 / 4) : c = 60 := by
  sorry

end water_tank_capacity_l669_66974


namespace boat_downstream_distance_l669_66925

/-- Calculates the distance traveled downstream by a boat given its own speed, speed against current, and time. -/
def distance_downstream (boat_speed : ℝ) (speed_against_current : ℝ) (time : ℝ) : ℝ :=
  let current_speed : ℝ := boat_speed - speed_against_current
  let downstream_speed : ℝ := boat_speed + current_speed
  downstream_speed * time

/-- Proves that a boat with given specifications travels 255 km downstream in 6 hours. -/
theorem boat_downstream_distance :
  distance_downstream 40 37.5 6 = 255 := by
  sorry

end boat_downstream_distance_l669_66925


namespace hyperbola_standard_form_l669_66961

theorem hyperbola_standard_form (x y : ℝ) :
  x^2 - 15 * y^2 = 15 ↔ x^2 / 15 - y^2 = 1 := by sorry

end hyperbola_standard_form_l669_66961
