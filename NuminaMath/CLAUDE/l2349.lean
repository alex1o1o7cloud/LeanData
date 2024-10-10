import Mathlib

namespace ellipse_foci_range_ellipse_or_quadratic_range_l2349_234907

/-- Definition of an ellipse with semi-major axis 5 and semi-minor axis √a -/
def is_ellipse (a : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / 5 + y^2 / a = 1

/-- The foci of the ellipse are on the x-axis -/
def foci_on_x_axis (a : ℝ) : Prop :=
  is_ellipse a ∧ ∃ c : ℝ, c^2 = 5 - a ∧ c ≥ 0

/-- The quadratic inequality holds for all real x -/
def quadratic_inequality_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + 2 * a * x + 3 ≥ 0

theorem ellipse_foci_range (a : ℝ) :
  foci_on_x_axis a → 0 < a ∧ a < 5 :=
sorry

theorem ellipse_or_quadratic_range (a : ℝ) :
  (foci_on_x_axis a ∨ quadratic_inequality_holds a) ∧
  ¬(foci_on_x_axis a ∧ quadratic_inequality_holds a) →
  (3 < a ∧ a < 5) ∨ (-3 ≤ a ∧ a ≤ 0) :=
sorry

end ellipse_foci_range_ellipse_or_quadratic_range_l2349_234907


namespace initial_boarders_l2349_234930

theorem initial_boarders (initial_boarders day_students new_boarders : ℕ) : 
  initial_boarders > 0 →
  day_students > 0 →
  new_boarders = 44 →
  initial_boarders * 12 = day_students * 5 →
  (initial_boarders + new_boarders) * 2 = day_students * 1 →
  initial_boarders = 220 := by
sorry

end initial_boarders_l2349_234930


namespace matrix_equation_equivalence_l2349_234974

theorem matrix_equation_equivalence 
  (n : ℕ) 
  (A B C : Matrix (Fin n) (Fin n) ℝ) 
  (h_inv : IsUnit A) 
  (h_eq : (A - B) * C = B * A⁻¹) : 
  C * (A - B) = A⁻¹ * B := by
  sorry

end matrix_equation_equivalence_l2349_234974


namespace count_specific_divisors_l2349_234909

theorem count_specific_divisors (p q : ℕ+) : 
  let n := 2^(p : ℕ) * 3^(q : ℕ)
  (∃ (s : Finset ℕ), s.card = p * q ∧ 
    (∀ d ∈ s, d ∣ n^2 ∧ d < n ∧ ¬(d ∣ n))) :=
by sorry

end count_specific_divisors_l2349_234909


namespace tangent_line_equation_l2349_234939

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2 + 3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 4 * x

-- Define the point of tangency
def x₀ : ℝ := -1

-- Define the slope of the tangent line at x₀
def m : ℝ := f' x₀

-- Define a point on the curve at x₀
def p : ℝ × ℝ := (x₀, f x₀)

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, y - p.2 = m * (x - p.1) ↔ y = -4 * x + 1 :=
by sorry

end tangent_line_equation_l2349_234939


namespace sphere_wedge_volume_l2349_234951

/-- Given a sphere with circumference 18π inches cut into 6 congruent wedges,
    the volume of one wedge is 162π cubic inches. -/
theorem sphere_wedge_volume :
  ∀ (r : ℝ),
  2 * π * r = 18 * π →
  (4 / 3 * π * r^3) / 6 = 162 * π := by
sorry

end sphere_wedge_volume_l2349_234951


namespace green_ball_probability_l2349_234946

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- Theorem: The probability of selecting a green ball is 53/96 -/
theorem green_ball_probability :
  let containers : List Container := [
    ⟨10, 5⟩,  -- Container I
    ⟨3, 5⟩,   -- Container II
    ⟨2, 6⟩,   -- Container III
    ⟨4, 4⟩    -- Container IV
  ]
  let totalContainers : ℕ := containers.length
  let containerProbability : ℚ := 1 / totalContainers
  let totalProbability : ℚ := (containers.map greenProbability).sum * containerProbability
  totalProbability = 53 / 96 := by
  sorry

end green_ball_probability_l2349_234946


namespace trajectory_of_point_P_l2349_234903

/-- The trajectory of point P satisfying given conditions -/
theorem trajectory_of_point_P (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (x - a) * b = y * a ∧  -- P lies on line AB
    (x - 0)^2 + (y - b)^2 = 4 * ((a - x)^2 + y^2) ∧  -- BP = 2PA
    (-x) * (-a) + y * b = 1)  -- OQ · AB = 1
  → 3/2 * x^2 + 3 * y^2 = 1 := by
  sorry

end trajectory_of_point_P_l2349_234903


namespace average_of_remaining_numbers_l2349_234964

theorem average_of_remaining_numbers 
  (total_average : ℝ) 
  (avg_group1 : ℝ) 
  (avg_group2 : ℝ) 
  (h1 : total_average = 3.9) 
  (h2 : avg_group1 = 3.4) 
  (h3 : avg_group2 = 3.85) : 
  (6 * total_average - 2 * avg_group1 - 2 * avg_group2) / 2 = 4.45 := by
sorry

#eval (6 * 3.9 - 2 * 3.4 - 2 * 3.85) / 2

end average_of_remaining_numbers_l2349_234964


namespace first_three_decimal_digits_l2349_234998

theorem first_three_decimal_digits (n : ℕ) (x : ℝ) : 
  n = 2003 → 
  x = (10^n + 1)^(11/7) → 
  ∃ (k : ℕ), x = k + 0.571 + r ∧ 0 ≤ r ∧ r < 0.001 :=
sorry

end first_three_decimal_digits_l2349_234998


namespace range_of_a_l2349_234970

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 4*x else 4*x - x^2

theorem range_of_a (a : ℝ) (h : f (a - 2) + f a > 0) : a > 1 := by
  sorry

end range_of_a_l2349_234970


namespace a_2_times_a_3_l2349_234980

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

theorem a_2_times_a_3 : a 2 * a 3 = 20 := by
  sorry

end a_2_times_a_3_l2349_234980


namespace triangle_problem_l2349_234948

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where c = √3, b = 1, and B = 30°, prove that C is either 60° or 120°,
    and the corresponding area S is either √3/2 or √3/4. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  c = Real.sqrt 3 →
  b = 1 →
  B = 30 * π / 180 →
  ((C = 60 * π / 180 ∧ S = Real.sqrt 3 / 2) ∨
   (C = 120 * π / 180 ∧ S = Real.sqrt 3 / 4)) :=
by sorry

end triangle_problem_l2349_234948


namespace unique_prime_triplet_l2349_234950

theorem unique_prime_triplet :
  ∃! (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    3 * p^4 - 5 * q^4 - 4 * r^2 = 26 :=
by
  -- The proof goes here
  sorry

end unique_prime_triplet_l2349_234950


namespace correct_assignment_count_l2349_234987

def num_rooms : ℕ := 6
def num_friends : ℕ := 6
def max_occupancy : ℕ := 3
def min_occupancy : ℕ := 1
def num_inseparable_friends : ℕ := 2

-- Function to calculate the number of ways to assign friends to rooms
def assignment_ways : ℕ := sorry

-- Theorem statement
theorem correct_assignment_count :
  assignment_ways = 3600 :=
sorry

end correct_assignment_count_l2349_234987


namespace checker_rearrangement_impossible_l2349_234996

/-- Represents a 5x5 chessboard -/
def Chessboard := Fin 5 → Fin 5 → Bool

/-- A function that determines if two positions are adjacent (horizontally or vertically) -/
def adjacent (p1 p2 : Fin 5 × Fin 5) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- A function representing the initial placement of checkers -/
def initial_placement : Fin 5 × Fin 5 → Fin 5 × Fin 5 :=
  fun p => p

/-- A function representing the final placement of checkers -/
def final_placement : Fin 5 × Fin 5 → Fin 5 × Fin 5 :=
  sorry

/-- Theorem stating that it's impossible to rearrange the checkers as required -/
theorem checker_rearrangement_impossible :
  ¬ (∀ p : Fin 5 × Fin 5, adjacent (initial_placement p) (final_placement p)) ∧
    (∀ p : Fin 5 × Fin 5, ∃ q, final_placement q = p) :=
sorry

end checker_rearrangement_impossible_l2349_234996


namespace abs_one_point_five_minus_sqrt_two_l2349_234999

theorem abs_one_point_five_minus_sqrt_two :
  |1.5 - Real.sqrt 2| = 1.5 - Real.sqrt 2 := by sorry

end abs_one_point_five_minus_sqrt_two_l2349_234999


namespace anne_tom_age_sum_l2349_234976

theorem anne_tom_age_sum : 
  ∀ (A T : ℝ),
  A = T + 9 →
  A + 7 = 5 * (T - 3) →
  A + T = 24.5 :=
by
  sorry

end anne_tom_age_sum_l2349_234976


namespace distinct_tetrahedrons_count_l2349_234904

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of vertices required to form a tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of non-tetrahedral configurations -/
def non_tetrahedral_configurations : ℕ := 12

/-- The number of distinct tetrahedrons that can be formed using the vertices of a cube -/
def distinct_tetrahedrons : ℕ :=
  Nat.choose cube_vertices tetrahedron_vertices - non_tetrahedral_configurations

theorem distinct_tetrahedrons_count : distinct_tetrahedrons = 58 := by
  sorry

end distinct_tetrahedrons_count_l2349_234904


namespace beat_kevin_record_l2349_234954

/-- The number of additional wings Alan must eat per minute to beat Kevin's record -/
def additional_wings_per_minute (
  kevin_record : ℕ
  ) (
  alan_current_rate : ℕ
  ) (
  time_frame : ℕ
  ) : ℕ :=
  ((kevin_record + 1) - alan_current_rate * time_frame + time_frame - 1) / time_frame

theorem beat_kevin_record (
  kevin_record : ℕ
  ) (
  alan_current_rate : ℕ
  ) (
  time_frame : ℕ
  ) (
  h1 : kevin_record = 64
  ) (
  h2 : alan_current_rate = 5
  ) (
  h3 : time_frame = 8
  ) : additional_wings_per_minute kevin_record alan_current_rate time_frame = 3 := by
  sorry

end beat_kevin_record_l2349_234954


namespace tan_pi_4_plus_2alpha_l2349_234988

open Real

theorem tan_pi_4_plus_2alpha (α : ℝ) : 
  π < α ∧ α < 3*π/2 →  -- α is in the third quadrant
  cos (2*α) = -3/5 → 
  tan (π/4 + 2*α) = -1/7 := by sorry

end tan_pi_4_plus_2alpha_l2349_234988


namespace unique_double_rectangle_with_perimeter_72_l2349_234929

/-- A rectangle with integer dimensions where one side is twice the other. -/
structure DoubleRectangle where
  shorter : ℕ
  longer : ℕ
  longer_is_double : longer = 2 * shorter

/-- The perimeter of a DoubleRectangle. -/
def perimeter (r : DoubleRectangle) : ℕ := 2 * (r.shorter + r.longer)

/-- The set of all DoubleRectangles with a perimeter of 72 inches. -/
def rectangles_with_perimeter_72 : Set DoubleRectangle :=
  {r : DoubleRectangle | perimeter r = 72}

theorem unique_double_rectangle_with_perimeter_72 :
  ∃! (r : DoubleRectangle), r ∈ rectangles_with_perimeter_72 := by
  sorry

#check unique_double_rectangle_with_perimeter_72

end unique_double_rectangle_with_perimeter_72_l2349_234929


namespace complex_equality_implies_a_value_l2349_234966

theorem complex_equality_implies_a_value (a : ℝ) : 
  (Complex.re ((1 + 2*I) * (2*a + I)) = Complex.im ((1 + 2*I) * (2*a + I))) → 
  a = -5/2 := by
sorry

end complex_equality_implies_a_value_l2349_234966


namespace megans_earnings_l2349_234943

/-- Calculates the total earnings for a given work schedule and hourly rate -/
def total_earnings (hours_per_day : ℕ) (hourly_rate : ℚ) (days_per_month : ℕ) (months : ℕ) : ℚ :=
  hours_per_day * hourly_rate * days_per_month * months

/-- Proves that Megan's total earnings for two months equal $2400 -/
theorem megans_earnings :
  let hours_per_day : ℕ := 8
  let hourly_rate : ℚ := 15/2
  let days_per_month : ℕ := 20
  let months : ℕ := 2
  total_earnings hours_per_day hourly_rate days_per_month months = 2400 := by
  sorry

#eval total_earnings 8 (15/2) 20 2

end megans_earnings_l2349_234943


namespace infinite_n_squared_plus_one_divides_and_not_divides_factorial_l2349_234922

theorem infinite_n_squared_plus_one_divides_and_not_divides_factorial :
  (∃ S : Set ℤ, Set.Infinite S ∧ ∀ n ∈ S, (n^2 + 1) ∣ n!) ∧
  (∃ T : Set ℤ, Set.Infinite T ∧ ∀ n ∈ T, ¬((n^2 + 1) ∣ n!)) :=
by sorry

end infinite_n_squared_plus_one_divides_and_not_divides_factorial_l2349_234922


namespace units_digit_of_quotient_l2349_234944

theorem units_digit_of_quotient (n : ℕ) : 
  (4^1993 + 5^1993) % 3 = 0 ∧ ((4^1993 + 5^1993) / 3) % 10 = 3 :=
by sorry

end units_digit_of_quotient_l2349_234944


namespace coefficient_of_x_cubed_in_expansion_l2349_234917

-- Define n such that 2^n = 64
def n : ℕ := 6

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Theorem statement
theorem coefficient_of_x_cubed_in_expansion :
  ∃ (coeff : ℤ), 
    (2^n = 64) ∧
    (coeff = (-1)^3 * binomial n 3 * 2^(n-3)) ∧
    (coeff = -160) := by
  sorry

end coefficient_of_x_cubed_in_expansion_l2349_234917


namespace circular_plate_ratio_l2349_234959

theorem circular_plate_ratio (radius : ℝ) (circumference : ℝ) 
  (h1 : radius = 15)
  (h2 : circumference = 90) :
  circumference / (2 * radius) = 3 := by
  sorry

end circular_plate_ratio_l2349_234959


namespace roots_of_polynomial_l2349_234906

theorem roots_of_polynomial (a b : ℝ) : 
  (a + 3 * Complex.I) * (b + 6 * Complex.I) = 52 + 105 * Complex.I ∧
  (a + 3 * Complex.I) + (b + 6 * Complex.I) = 12 + 15 * Complex.I →
  a = 23 ∧ b = -11 := by
  sorry

end roots_of_polynomial_l2349_234906


namespace largest_hope_number_proof_l2349_234912

/-- A Hope Number is a natural number with an odd number of divisors --/
def isHopeNumber (n : ℕ) : Prop := Odd (Nat.divisors n).card

/-- The largest Hope Number within 1000 --/
def largestHopeNumber : ℕ := 961

theorem largest_hope_number_proof :
  (∀ m : ℕ, m ≤ 1000 → isHopeNumber m → m ≤ largestHopeNumber) ∧
  isHopeNumber largestHopeNumber ∧
  largestHopeNumber ≤ 1000 :=
sorry

end largest_hope_number_proof_l2349_234912


namespace sum_of_digits_1024_base5_l2349_234984

/-- Converts a natural number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_1024_base5 :
  sumList (toBase5 1024) = 12 := by
  sorry

end sum_of_digits_1024_base5_l2349_234984


namespace cubic_factorization_l2349_234945

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end cubic_factorization_l2349_234945


namespace quadratic_roots_difference_squared_l2349_234992

theorem quadratic_roots_difference_squared : 
  ∀ Φ φ : ℝ, 
  (Φ ^ 2 = Φ + 2) → 
  (φ ^ 2 = φ + 2) → 
  (Φ ≠ φ) → 
  (Φ - φ) ^ 2 = 9 := by
sorry

end quadratic_roots_difference_squared_l2349_234992


namespace age_ratio_theorem_l2349_234924

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the ages of Bipin, Alok, and Chandan -/
structure Ages where
  bipin : Age
  alok : Age
  chandan : Age

/-- The conditions given in the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.alok.years = 5 ∧
  ages.chandan.years = 10 ∧
  ages.bipin.years + 10 = 2 * (ages.chandan.years + 10)

/-- The theorem to prove -/
theorem age_ratio_theorem (ages : Ages) :
  problem_conditions ages →
  (ages.bipin.years : ℚ) / ages.alok.years = 6 / 1 := by
  sorry


end age_ratio_theorem_l2349_234924


namespace order_of_fractions_with_exponents_l2349_234908

theorem order_of_fractions_with_exponents :
  (1/5 : ℝ)^(2/3) < (1/2 : ℝ)^(2/3) ∧ (1/2 : ℝ)^(2/3) < (1/2 : ℝ)^(1/3) := by
  sorry

end order_of_fractions_with_exponents_l2349_234908


namespace no_solution_to_equation_l2349_234933

theorem no_solution_to_equation : ¬∃ x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3*x^2 - 15*x) / (x^2 - 5*x) = x - 2 := by
  sorry

end no_solution_to_equation_l2349_234933


namespace complex_equal_parts_l2349_234971

theorem complex_equal_parts (a : ℝ) : 
  let z : ℂ := (1 - a * Complex.I) / (2 + Complex.I)
  (z.re = z.im) → a = -3 := by
sorry

end complex_equal_parts_l2349_234971


namespace integer_pairs_satisfying_equation_l2349_234914

theorem integer_pairs_satisfying_equation :
  {(x, y) : ℤ × ℤ | x^2 = y^2 + 2*y + 13} =
  {(4, -3), (4, 1), (-4, 1), (-4, -3)} := by
  sorry

end integer_pairs_satisfying_equation_l2349_234914


namespace opposite_gold_is_black_l2349_234925

-- Define the set of colors
inductive Color
  | Blue
  | Orange
  | Yellow
  | Black
  | Silver
  | Gold

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  right : Face
  left : Face

-- Define the three views of the cube
def view1 (c : Cube) : Prop :=
  c.top.color = Color.Black ∧ c.front.color = Color.Blue ∧ c.right.color = Color.Orange

def view2 (c : Cube) : Prop :=
  c.top.color = Color.Black ∧ c.front.color = Color.Yellow ∧ c.right.color = Color.Orange

def view3 (c : Cube) : Prop :=
  c.top.color = Color.Black ∧ c.front.color = Color.Silver ∧ c.right.color = Color.Orange

-- Theorem statement
theorem opposite_gold_is_black (c : Cube) :
  view1 c → view2 c → view3 c → c.bottom.color = Color.Gold → c.top.color = Color.Black :=
by sorry

end opposite_gold_is_black_l2349_234925


namespace train_bus_cost_l2349_234915

theorem train_bus_cost (bus_cost : ℝ) (train_extra_cost : ℝ) : 
  bus_cost = 1.40 →
  train_extra_cost = 6.85 →
  bus_cost + (bus_cost + train_extra_cost) = 9.65 := by
sorry

end train_bus_cost_l2349_234915


namespace jellybean_problem_l2349_234902

theorem jellybean_problem (initial_quantity : ℝ) : 
  (initial_quantity * (1 - 0.3)^4 = 48) → initial_quantity = 200 := by
  sorry

end jellybean_problem_l2349_234902


namespace polynomial_simplification_l2349_234936

theorem polynomial_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x - a)^4 / ((a - b) * (a - c)) + (x - b)^4 / ((b - a) * (b - c)) + (x - c)^4 / ((c - a) * (c - b)) =
  x^4 - 2*(a+b+c)*x^3 + (a^2+b^2+c^2+2*a*b+2*b*c+2*c*a)*x^2 - 2*a*b*c*x := by
sorry

end polynomial_simplification_l2349_234936


namespace average_difference_l2349_234972

theorem average_difference (a b c : ℝ) 
  (hab : (a + b) / 2 = 110) 
  (hbc : (b + c) / 2 = 150) : 
  a - c = -80 := by sorry

end average_difference_l2349_234972


namespace rectangle_area_change_l2349_234926

/-- Given a rectangle with area 540 square centimeters, if its length is increased by 15%
    and its width is decreased by 20%, then its new area is 496.8 square centimeters. -/
theorem rectangle_area_change (l w : ℝ) (h1 : l * w = 540) : 
  (1.15 * l) * (0.8 * w) = 496.8 := by
  sorry

end rectangle_area_change_l2349_234926


namespace peter_is_18_l2349_234905

-- Define Peter's current age
def peter_current_age : ℕ := sorry

-- Define Ivan's current age
def ivan_current_age : ℕ := sorry

-- Define Peter's past age when Ivan was Peter's current age
def peter_past_age : ℕ := sorry

-- Condition 1: Ivan's current age is twice Peter's past age
axiom ivan_age_relation : ivan_current_age = 2 * peter_past_age

-- Condition 2: Sum of their ages will be 54 when Peter reaches Ivan's current age
axiom future_age_sum : ivan_current_age + ivan_current_age = 54

-- Condition 3: The time difference between Peter's current age and past age
-- is equal to the time difference between Ivan's current age and Peter's current age
axiom age_difference_relation : ivan_current_age - peter_current_age = peter_current_age - peter_past_age

theorem peter_is_18 : peter_current_age = 18 := by
  sorry

end peter_is_18_l2349_234905


namespace nested_expression_value_l2349_234957

def nested_expression : ℕ :=
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))))

theorem nested_expression_value : nested_expression = 87380 := by
  sorry

end nested_expression_value_l2349_234957


namespace right_pentagonal_pyramid_base_side_length_l2349_234990

/-- Represents a right pyramid with a regular pentagonal base -/
structure RightPentagonalPyramid where
  base_side_length : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- 
Theorem: For a right pyramid with a regular pentagonal base, 
if the area of one lateral face is 120 square meters and the slant height is 40 meters, 
then the length of the side of its base is 6 meters.
-/
theorem right_pentagonal_pyramid_base_side_length 
  (pyramid : RightPentagonalPyramid) 
  (h1 : pyramid.lateral_face_area = 120) 
  (h2 : pyramid.slant_height = 40) : 
  pyramid.base_side_length = 6 := by
  sorry

end right_pentagonal_pyramid_base_side_length_l2349_234990


namespace march_greatest_drop_l2349_234921

/-- Represents the months from January to August --/
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August

/-- Price change for each month --/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January => -1.00
  | Month.February => 1.50
  | Month.March => -3.00
  | Month.April => 2.00
  | Month.May => -0.75
  | Month.June => 1.00
  | Month.July => -2.50
  | Month.August => -2.00

/-- Definition of a price drop --/
def is_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- Theorem: March has the greatest monthly drop in price --/
theorem march_greatest_drop :
  ∀ m : Month, is_price_drop m → price_change Month.March ≤ price_change m :=
by sorry

end march_greatest_drop_l2349_234921


namespace manager_average_salary_l2349_234961

/-- Proves that the average salary of managers is $90,000 given the company's employee structure and salary information --/
theorem manager_average_salary 
  (num_managers : ℕ) 
  (num_associates : ℕ) 
  (associate_avg_salary : ℝ) 
  (company_avg_salary : ℝ) : 
  num_managers = 15 → 
  num_associates = 75 → 
  associate_avg_salary = 30000 → 
  company_avg_salary = 40000 → 
  (num_managers * (num_managers * company_avg_salary - num_associates * associate_avg_salary) / 
   (num_managers * (num_managers + num_associates))) = 90000 := by
  sorry

end manager_average_salary_l2349_234961


namespace max_small_boxes_in_large_box_l2349_234991

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ := m * 100

/-- The dimensions of the large wooden box in meters -/
def largeBoxDimMeters : BoxDimensions := {
  length := 4,
  width := 2,
  height := 4
}

/-- The dimensions of the large wooden box in centimeters -/
def largeBoxDimCm : BoxDimensions := {
  length := metersToCentimeters largeBoxDimMeters.length,
  width := metersToCentimeters largeBoxDimMeters.width,
  height := metersToCentimeters largeBoxDimMeters.height
}

/-- The dimensions of the small rectangular box in centimeters -/
def smallBoxDimCm : BoxDimensions := {
  length := 4,
  width := 2,
  height := 2
}

theorem max_small_boxes_in_large_box :
  (boxVolume largeBoxDimCm) / (boxVolume smallBoxDimCm) = 2000000 := by
  sorry

end max_small_boxes_in_large_box_l2349_234991


namespace quadratic_inequality_l2349_234960

-- Define the set [1,3]
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - 2

-- Define the solution set
def X : Set ℝ := {x : ℝ | x < -1 ∨ x > 2/3}

-- State the theorem
theorem quadratic_inequality (x : ℝ) :
  (∃ a ∈ A, f a x > 0) → x ∈ X :=
sorry

end quadratic_inequality_l2349_234960


namespace boat_speed_problem_l2349_234955

/-- Proves that given a lake of width 60 miles, a boat traveling at 30 mph,
    and a waiting time of 3 hours for another boat to arrive,
    the speed of the second boat is 12 mph. -/
theorem boat_speed_problem (lake_width : ℝ) (janet_speed : ℝ) (waiting_time : ℝ) :
  lake_width = 60 →
  janet_speed = 30 →
  waiting_time = 3 →
  ∃ (sister_speed : ℝ),
    sister_speed = lake_width / (lake_width / janet_speed + waiting_time) ∧
    sister_speed = 12 := by sorry

end boat_speed_problem_l2349_234955


namespace complex_simplification_l2349_234931

/-- Given that i^2 = -1, prove that 3(4-2i) - 2i(3-i) + i(1+2i) = 8 - 11i -/
theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  3 * (4 - 2*i) - 2*i*(3 - i) + i*(1 + 2*i) = 8 - 11*i :=
by sorry

end complex_simplification_l2349_234931


namespace ratio_problem_l2349_234997

theorem ratio_problem (N : ℝ) (h1 : (1/1) * (1/3) * (2/5) * N = 25) (h2 : (40/100) * N = 300) :
  (25 : ℝ) / ((1/3) * (2/5) * N) = 1/4 := by
  sorry

end ratio_problem_l2349_234997


namespace intersection_circles_sum_l2349_234981

/-- Given two circles intersecting at (2,3) and (m,2), with centers on the line x+y+n=0, prove m+n = -2 -/
theorem intersection_circles_sum (m n : ℝ) : 
  (∃ (c₁ c₂ : ℝ × ℝ), 
    (c₁.1 + c₁.2 + n = 0) ∧ 
    (c₂.1 + c₂.2 + n = 0) ∧
    ((2 - c₁.1)^2 + (3 - c₁.2)^2 = (2 - c₂.1)^2 + (3 - c₂.2)^2) ∧
    ((m - c₁.1)^2 + (2 - c₁.2)^2 = (m - c₂.1)^2 + (2 - c₂.2)^2) ∧
    ((2 - c₁.1)^2 + (3 - c₁.2)^2 = (m - c₁.1)^2 + (2 - c₁.2)^2) ∧
    ((2 - c₂.1)^2 + (3 - c₂.2)^2 = (m - c₂.1)^2 + (2 - c₂.2)^2)) →
  m + n = -2 := by
sorry

end intersection_circles_sum_l2349_234981


namespace least_possible_difference_l2349_234983

theorem least_possible_difference (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → Odd y → Odd z → 
  (∀ (x' y' z' : ℤ), x' < y' → y' < z' → y' - x' > 5 → Even x' → Odd y' → Odd z' → z' - x' ≥ z - x) →
  z - x = 9 := by sorry

end least_possible_difference_l2349_234983


namespace car_pushing_speed_l2349_234949

theorem car_pushing_speed (total_distance total_time first_segment second_segment third_segment : ℝ)
  (second_speed third_speed : ℝ) :
  total_distance = 10 ∧
  total_time = 2 ∧
  first_segment = 3 ∧
  second_segment = 3 ∧
  third_segment = 4 ∧
  second_speed = 3 ∧
  third_speed = 8 ∧
  total_time = first_segment / v + second_segment / second_speed + third_segment / third_speed →
  v = 6 :=
by
  sorry

end car_pushing_speed_l2349_234949


namespace x_plus_2y_equals_8_l2349_234962

theorem x_plus_2y_equals_8 (x y : ℝ) 
  (h1 : (x + y) / 3 = 1.6666666666666667)
  (h2 : 2 * x + y = 7) : 
  x + 2 * y = 8 := by
  sorry

end x_plus_2y_equals_8_l2349_234962


namespace cos_90_degrees_l2349_234928

theorem cos_90_degrees : Real.cos (π / 2) = 0 := by
  sorry

end cos_90_degrees_l2349_234928


namespace mean_temperature_is_88_75_l2349_234935

def temperatures : List ℚ := [85, 84, 85, 88, 91, 93, 94, 90]

theorem mean_temperature_is_88_75 :
  (temperatures.sum / temperatures.length : ℚ) = 355/4 := by sorry

end mean_temperature_is_88_75_l2349_234935


namespace caleb_hamburger_cost_l2349_234916

def total_burgers : ℕ := 50
def single_burger_cost : ℚ := 1
def double_burger_cost : ℚ := 1.5
def double_burgers_bought : ℕ := 29

theorem caleb_hamburger_cost :
  let single_burgers := total_burgers - double_burgers_bought
  let total_cost := (single_burgers : ℚ) * single_burger_cost +
                    (double_burgers_bought : ℚ) * double_burger_cost
  total_cost = 64.5 := by sorry

end caleb_hamburger_cost_l2349_234916


namespace system_solution_l2349_234918

theorem system_solution :
  let x₁ := Real.sqrt 2 / Real.sqrt 5
  let x₂ := -Real.sqrt 2 / Real.sqrt 5
  let y₁ := 2 * Real.sqrt 2 / Real.sqrt 5
  let y₂ := -2 * Real.sqrt 2 / Real.sqrt 5
  let condition₁ (x y : ℝ) := x^2 + y^2 ≤ 2
  let condition₂ (x y : ℝ) := x^4 - 8*x^2*y^2 + 16*y^4 - 20*x^2 - 80*y^2 + 100 = 0
  (condition₁ x₁ y₁ ∧ condition₂ x₁ y₁) ∧
  (condition₁ x₁ y₂ ∧ condition₂ x₁ y₂) ∧
  (condition₁ x₂ y₁ ∧ condition₂ x₂ y₁) ∧
  (condition₁ x₂ y₂ ∧ condition₂ x₂ y₂) :=
by sorry


end system_solution_l2349_234918


namespace debate_team_grouping_l2349_234901

theorem debate_team_grouping (boys girls groups : ℕ) (h1 : boys = 28) (h2 : girls = 4) (h3 : groups = 8) :
  (boys + girls) / groups = 4 := by
  sorry

end debate_team_grouping_l2349_234901


namespace prob_two_non_defective_pens_l2349_234979

/-- Probability of selecting two non-defective pens from a box of pens -/
theorem prob_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (h1 : total_pens = 12) 
  (h2 : defective_pens = 6) : 
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 5 / 22 := by
  sorry

end prob_two_non_defective_pens_l2349_234979


namespace distance_to_black_planet_l2349_234967

/-- The distance to a black planet given spaceship and light travel times -/
theorem distance_to_black_planet 
  (v_ship : ℝ) -- speed of spaceship
  (v_light : ℝ) -- speed of light
  (t_total : ℝ) -- total time of travel and light reflection
  (h_v_ship : v_ship = 100000) -- spaceship speed in km/s
  (h_v_light : v_light = 300000) -- light speed in km/s
  (h_t_total : t_total = 100) -- total time in seconds
  : ∃ d : ℝ, d = 1500 * 10000 ∧ t_total = (d + v_ship * t_total) / v_light + d / v_light :=
by sorry

end distance_to_black_planet_l2349_234967


namespace difference_of_largest_and_smallest_l2349_234965

def digits : List ℕ := [2, 7, 4, 9]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ 
  ∃ (a b c : ℕ), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = 100 * a + 10 * b + c

def largest_number : ℕ := 974
def smallest_number : ℕ := 247

theorem difference_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_number) ∧
  (∀ n : ℕ, is_valid_number n → n ≥ smallest_number) ∧
  largest_number - smallest_number = 727 :=
sorry

end difference_of_largest_and_smallest_l2349_234965


namespace money_left_calculation_l2349_234985

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 4 * p
  let total_spent := 5 * drink_cost + 2 * medium_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Theorem stating that the money left is equal to 50 - 15p -/
theorem money_left_calculation (p : ℝ) : money_left p = 50 - 15 * p := by
  sorry

end money_left_calculation_l2349_234985


namespace lisa_dvd_rental_l2349_234923

theorem lisa_dvd_rental (total_cost : ℚ) (cost_per_dvd : ℚ) (h1 : total_cost = 4.80) (h2 : cost_per_dvd = 1.20) :
  total_cost / cost_per_dvd = 4 := by
  sorry

end lisa_dvd_rental_l2349_234923


namespace inequality_solution_l2349_234913

theorem inequality_solution (n : ℕ+) : 2*n - 5 < 5 - 2*n ↔ n = 1 ∨ n = 2 := by
  sorry

end inequality_solution_l2349_234913


namespace isosceles_trapezoid_area_l2349_234958

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  long_base : ℝ

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating that the area of the given isosceles trapezoid is approximately 2457.2 -/
theorem isosceles_trapezoid_area :
  let t : IsoscelesTrapezoid := ⟨40, 50, 65⟩
  ∃ ε > 0, |area t - 2457.2| < ε :=
sorry

end isosceles_trapezoid_area_l2349_234958


namespace am_gm_inequality_and_specific_case_l2349_234952

theorem am_gm_inequality_and_specific_case :
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 → (a + b + c) / 3 ≥ (a * b * c) ^ (1/3)) ∧
  ((4 + 9 + 16) / 3 - (4 * 9 * 16) ^ (1/3) ≠ 1) := by
  sorry

end am_gm_inequality_and_specific_case_l2349_234952


namespace cos_equality_solution_l2349_234982

theorem cos_equality_solution (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end cos_equality_solution_l2349_234982


namespace parabola_directrix_l2349_234973

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -1/2

/-- Theorem: The directrix of the given parabola is y = -1/2 -/
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola x y → (∃ (d : ℝ), directrix d ∧ 
    d = y - (1/4) ∧ 
    ∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y → 
      (p.1 - 4)^2 + (p.2 - 0)^2 = (p.2 - d)^2) :=
by sorry

end parabola_directrix_l2349_234973


namespace painter_scenario_proof_l2349_234986

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculate the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Theorem: For the given painting scenario, the time to paint the remaining rooms is 49 hours. -/
theorem painter_scenario_proof :
  time_to_paint_remaining 12 7 5 = 49 := by
  sorry

end painter_scenario_proof_l2349_234986


namespace inequality_relationship_l2349_234994

theorem inequality_relationship (a b : ℝ) : 
  ¬(((2 : ℝ)^a > (2 : ℝ)^b → (1 : ℝ)/a < (1 : ℝ)/b) ∧ 
    ((1 : ℝ)/a < (1 : ℝ)/b → (2 : ℝ)^a > (2 : ℝ)^b)) :=
sorry

end inequality_relationship_l2349_234994


namespace solve_cubic_equation_l2349_234911

theorem solve_cubic_equation (y : ℝ) : 
  5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3) → y = 1000 := by
  sorry

end solve_cubic_equation_l2349_234911


namespace swimming_speed_problem_l2349_234919

/-- Given a person who swims for 2 hours at speed S and runs for 1 hour at speed 4S, 
    if the total distance covered is 12 miles, then S must equal 2 miles per hour. -/
theorem swimming_speed_problem (S : ℝ) : 
  (2 * S + 1 * (4 * S) = 12) → S = 2 := by
  sorry

end swimming_speed_problem_l2349_234919


namespace real_part_of_z_l2349_234968

def complex_number_z : ℂ → Prop :=
  λ z ↦ z * Complex.I = 2 * Complex.I

theorem real_part_of_z (z : ℂ) (h : complex_number_z z) :
  z.re = 3/2 := by sorry

end real_part_of_z_l2349_234968


namespace existence_of_abc_l2349_234940

theorem existence_of_abc (p : ℕ) (hp : p.Prime) (hp_gt_2011 : p > 2011) :
  ∃ (a b c : ℕ+), (¬(p ∣ a) ∨ ¬(p ∣ b) ∨ ¬(p ∣ c)) ∧
    ∀ (n : ℕ+), p ∣ (n^4 - 2*n^2 + 9) → p ∣ (24*a*n^2 + 5*b*n + 2011*c) := by
  sorry

end existence_of_abc_l2349_234940


namespace square_sum_given_sum_and_product_l2349_234941

theorem square_sum_given_sum_and_product (a b : ℝ) 
  (h1 : a + b = 12) (h2 : a * b = 20) : a^2 + b^2 = 104 := by
  sorry

end square_sum_given_sum_and_product_l2349_234941


namespace unique_number_proof_l2349_234953

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ n / 1000 = 6

def move_first_to_last (n : ℕ) : ℕ :=
  (n % 1000) * 10 + (n / 1000)

theorem unique_number_proof :
  ∃! n : ℕ, is_valid_number n ∧ move_first_to_last n = n - 1152 :=
by
  use 6538
  sorry

end unique_number_proof_l2349_234953


namespace saree_sale_price_l2349_234942

/-- The sale price of a saree after successive discounts -/
theorem saree_sale_price (original_price : ℝ) (discount1 discount2 discount3 discount4 : ℝ) :
  original_price = 400 →
  discount1 = 0.20 →
  discount2 = 0.05 →
  discount3 = 0.10 →
  discount4 = 0.15 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) * (1 - discount4) = 232.56 := by
  sorry

end saree_sale_price_l2349_234942


namespace sum_a_b_equals_negative_one_l2349_234975

theorem sum_a_b_equals_negative_one (a b : ℝ) :
  |a - 2| + (b + 3)^2 = 0 → a + b = -1 := by
  sorry

end sum_a_b_equals_negative_one_l2349_234975


namespace taehyung_calculation_l2349_234963

theorem taehyung_calculation (x : ℝ) (h : 5 * x = 30) : x / 6 = 1 := by
  sorry

end taehyung_calculation_l2349_234963


namespace sin_15_degrees_l2349_234932

theorem sin_15_degrees : Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end sin_15_degrees_l2349_234932


namespace power_multiplication_l2349_234937

theorem power_multiplication (a : ℝ) : a * a^3 = a^4 := by
  sorry

end power_multiplication_l2349_234937


namespace bacon_suggestion_l2349_234934

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes : ℕ := 408

/-- The difference between the number of students who suggested mashed potatoes and bacon -/
def difference : ℕ := 366

/-- The number of students who suggested adding bacon -/
def bacon : ℕ := mashed_potatoes - difference

theorem bacon_suggestion :
  bacon = 42 :=
by sorry

end bacon_suggestion_l2349_234934


namespace caleb_gallons_per_trip_l2349_234969

/-- Prove that Caleb adds 7 gallons per trip to fill a pool --/
theorem caleb_gallons_per_trip 
  (pool_capacity : ℕ) 
  (cynthia_gallons : ℕ) 
  (total_trips : ℕ) 
  (h1 : pool_capacity = 105)
  (h2 : cynthia_gallons = 8)
  (h3 : total_trips = 7)
  : ∃ (caleb_gallons : ℕ), 
    caleb_gallons * total_trips + cynthia_gallons * total_trips = pool_capacity ∧ 
    caleb_gallons = 7 := by
  sorry

end caleb_gallons_per_trip_l2349_234969


namespace greatest_sum_consecutive_integers_l2349_234927

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → m ≤ n) → n + (n + 1) = 43 :=
by sorry

end greatest_sum_consecutive_integers_l2349_234927


namespace expression_factorization_l2349_234989

theorem expression_factorization (b : ℝ) :
  (4 * b^3 + 126 * b^2 - 9) - (-9 * b^3 + 2 * b^2 - 9) = b^2 * (13 * b + 124) := by
  sorry

end expression_factorization_l2349_234989


namespace optimal_fare_and_passenger_change_l2349_234956

/-- Demand function -/
def demand (p : ℝ) : ℝ := 4200 - 100 * p

/-- Train fare -/
def train_fare : ℝ := 4

/-- Train capacity -/
def train_capacity : ℝ := 800

/-- Bus company cost function -/
def bus_cost (y : ℝ) : ℝ := 10 * y + 225

/-- Optimal bus fare -/
def optimal_bus_fare : ℝ := 22

/-- Change in total passengers if train service closes -/
def passenger_change : ℝ := -400

/-- Theorem stating the optimal bus fare and passenger change -/
theorem optimal_fare_and_passenger_change :
  (∃ (p : ℝ), p = optimal_bus_fare ∧
    ∀ (p' : ℝ), p' > train_fare →
      p * (demand p - train_capacity) - bus_cost (demand p - train_capacity) ≥
      p' * (demand p' - train_capacity) - bus_cost (demand p' - train_capacity)) ∧
  (demand (26) - (demand optimal_bus_fare - train_capacity + train_capacity) = passenger_change) :=
sorry

end optimal_fare_and_passenger_change_l2349_234956


namespace initial_average_age_proof_l2349_234900

/-- Proves that the initial average age of a group is 16 years, given the conditions of the problem -/
theorem initial_average_age_proof (initial_count : ℕ) (new_count : ℕ) (new_avg_age : ℚ) (final_avg_age : ℚ) :
  initial_count = 12 →
  new_count = 12 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  (initial_count * (initial_count * final_avg_age - new_count * new_avg_age) / (initial_count * initial_count)) = 16 := by
  sorry

end initial_average_age_proof_l2349_234900


namespace right_triangle_ab_length_l2349_234995

/-- Given a right triangle ABC in the x-y plane where:
  - ∠B = 90°
  - The length of AC is 100
  - The slope of line segment AC is 4/3
  Prove that the length of AB is 80 -/
theorem right_triangle_ab_length 
  (A B C : ℝ × ℝ) 
  (right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0)
  (ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 100)
  (ac_slope : (C.2 - A.2) / (C.1 - A.1) = 4/3) :
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 80 := by
  sorry

end right_triangle_ab_length_l2349_234995


namespace at_most_one_point_inside_plane_l2349_234920

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Checks if a point is on a line -/
def isPointOnLine (p : Point3D) (l : Line3D) : Prop := sorry

/-- Checks if a point is on a plane -/
def isPointOnPlane (p : Point3D) (pl : Plane3D) : Prop := sorry

/-- Checks if a point is outside a plane -/
def isPointOutsidePlane (p : Point3D) (pl : Plane3D) : Prop := 
  ¬(isPointOnPlane p pl)

/-- The main theorem -/
theorem at_most_one_point_inside_plane 
  (l : Line3D) (pl : Plane3D) 
  (p1 p2 : Point3D) 
  (h1 : isPointOnLine p1 l) 
  (h2 : isPointOnLine p2 l) 
  (h3 : isPointOutsidePlane p1 pl) 
  (h4 : isPointOutsidePlane p2 pl) : 
  ∃! p, isPointOnLine p l ∧ isPointOnPlane p pl :=
sorry

end at_most_one_point_inside_plane_l2349_234920


namespace inverse_variation_problem_l2349_234993

/-- Given that x^2 and ∛y vary inversely, and x = 3 when y = 64, prove that y = 15 * ∛15 when xy = 90 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) :
  (∀ x y, x^2 * y^(1/3) = k) →  -- x^2 and ∛y vary inversely
  (3^2 * 64^(1/3) = k) →        -- x = 3 when y = 64
  (x * y = 90) →                -- xy = 90
  (y = 15 * 15^(1/5)) :=        -- y = 15 * ∛15
by sorry

end inverse_variation_problem_l2349_234993


namespace square_field_area_l2349_234978

/-- The area of a square field with side length 14 meters is 196 square meters. -/
theorem square_field_area : 
  ∀ (side_length area : ℝ), 
  side_length = 14 → 
  area = side_length ^ 2 → 
  area = 196 :=
by
  sorry

end square_field_area_l2349_234978


namespace zoey_lottery_split_l2349_234938

theorem zoey_lottery_split (lottery_amount : ℕ) (h1 : lottery_amount = 7348340) :
  ∃ (num_friends : ℕ), 
    (lottery_amount + 1) % (num_friends + 1) = 0 ∧ 
    num_friends = 7348340 := by
  sorry

end zoey_lottery_split_l2349_234938


namespace sports_club_tennis_players_l2349_234910

/-- Given a sports club with the following properties:
  * There are 30 members in total
  * 17 members play badminton
  * 2 members do not play either badminton or tennis
  * 10 members play both badminton and tennis
  Prove that 21 members play tennis -/
theorem sports_club_tennis_players :
  ∀ (total_members badminton_players neither_players both_players : ℕ),
    total_members = 30 →
    badminton_players = 17 →
    neither_players = 2 →
    both_players = 10 →
    ∃ (tennis_players : ℕ),
      tennis_players = 21 ∧
      tennis_players = total_members - neither_players - (badminton_players - both_players) :=
by sorry

end sports_club_tennis_players_l2349_234910


namespace toms_age_ratio_l2349_234977

theorem toms_age_ratio (T N : ℚ) : 
  (T > 0) →  -- Tom's age is positive
  (N > 0) →  -- N is positive (years in the past)
  (∃ (a b c d : ℚ), a + b + c + d = T ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) →  -- Four children's ages sum to T
  (T - N = 3 * (T - 4 * N)) →  -- Condition from N years ago
  T / N = 11 / 2 := by
sorry

end toms_age_ratio_l2349_234977


namespace magazine_cost_l2349_234947

theorem magazine_cost (book magazine : ℚ)
  (h1 : 2 * book + 2 * magazine = 26)
  (h2 : book + 3 * magazine = 27) :
  magazine = 7 := by
sorry

end magazine_cost_l2349_234947
