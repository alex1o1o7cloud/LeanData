import Mathlib

namespace spring_outing_speeds_l1956_195661

theorem spring_outing_speeds (distance : ℝ) (bus_head_start : ℝ) (car_earlier_arrival : ℝ) :
  distance = 90 →
  bus_head_start = 0.5 →
  car_earlier_arrival = 0.25 →
  ∃ (bus_speed car_speed : ℝ),
    car_speed = 1.5 * bus_speed ∧
    distance / bus_speed - distance / car_speed = bus_head_start + car_earlier_arrival ∧
    bus_speed = 40 ∧
    car_speed = 60 := by
  sorry

end spring_outing_speeds_l1956_195661


namespace subset_range_equivalence_l1956_195671

theorem subset_range_equivalence (a : ℝ) :
  ({x : ℝ | x^2 + 2*(1-a)*x + (3-a) ≤ 0} ⊆ Set.Icc 0 3) ↔ (-1 ≤ a ∧ a ≤ 18/7) := by
  sorry

end subset_range_equivalence_l1956_195671


namespace quadratic_equations_roots_l1956_195624

theorem quadratic_equations_roots :
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = -5 ∧ x₁^2 + 6*x₁ + 5 = 0 ∧ x₂^2 + 6*x₂ + 5 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 2 + Real.sqrt 5 ∧ y₂ = 2 - Real.sqrt 5 ∧ y₁^2 - 4*y₁ - 1 = 0 ∧ y₂^2 - 4*y₂ - 1 = 0) :=
by sorry

end quadratic_equations_roots_l1956_195624


namespace area_of_region_l1956_195649

-- Define the region
def region (x y : ℝ) : Prop :=
  Real.sqrt (Real.arcsin y) ≤ Real.sqrt (Real.arccos x) ∧ 
  -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1

-- State the theorem
theorem area_of_region : 
  MeasureTheory.volume {p : ℝ × ℝ | region p.1 p.2} = 1 + π / 4 := by
  sorry

end area_of_region_l1956_195649


namespace average_of_x_and_y_l1956_195607

theorem average_of_x_and_y (x y : ℝ) : 
  (4 + 6 + 8 + x + y) / 5 = 20 → (x + y) / 2 = 41 := by
  sorry

end average_of_x_and_y_l1956_195607


namespace min_value_theorem_l1956_195683

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  ∃ (m : ℝ), m = 24 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a * b * c = 64 → x + 4*y + 8*z ≤ a + 4*b + 8*c ∧
  (x + 4*y + 8*z = m ∨ a + 4*b + 8*c > m) := by
  sorry

end min_value_theorem_l1956_195683


namespace set_intersection_complement_l1956_195658

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem set_intersection_complement :
  A ∩ (Set.univ \ B) = {x | 3 < x ∧ x < 4} := by sorry

end set_intersection_complement_l1956_195658


namespace chocolate_bars_unsold_l1956_195644

theorem chocolate_bars_unsold (total_bars : ℕ) (price_per_bar : ℕ) (revenue : ℕ) : 
  total_bars = 11 → price_per_bar = 4 → revenue = 16 → total_bars - (revenue / price_per_bar) = 7 := by
  sorry

end chocolate_bars_unsold_l1956_195644


namespace geometric_sequence_sum_l1956_195641

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a n > 0) →  -- All terms are positive
  a 1 = 3 →  -- First term is 3
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 + a 2 + a 3 = 21 →  -- Sum of first three terms is 21
  a 3 + a 4 + a 5 = 84 := by
  sorry

end geometric_sequence_sum_l1956_195641


namespace amaya_total_marks_l1956_195621

/-- Represents the marks scored in different subjects -/
structure Marks where
  music : ℕ
  social_studies : ℕ
  arts : ℕ
  maths : ℕ

/-- Calculates the total marks scored in all subjects -/
def total_marks (m : Marks) : ℕ :=
  m.music + m.social_studies + m.arts + m.maths

/-- Theorem stating the total marks Amaya scored -/
theorem amaya_total_marks (m : Marks) 
  (h1 : m.maths + 20 = m.arts)
  (h2 : m.social_studies = m.music + 10)
  (h3 : m.maths = m.arts - m.arts / 10)
  (h4 : m.music = 70) :
  total_marks m = 530 := by
  sorry

#eval total_marks ⟨70, 80, 200, 180⟩

end amaya_total_marks_l1956_195621


namespace expression_value_l1956_195648

theorem expression_value : (85 + 32 / 113) * 113 = 9635 := by
  sorry

end expression_value_l1956_195648


namespace point_on_x_axis_l1956_195669

theorem point_on_x_axis (x : ℝ) : 
  (x^2 + 2 + 9 = 12) → (x = 1 ∨ x = -1) := by
  sorry

#check point_on_x_axis

end point_on_x_axis_l1956_195669


namespace problem_1_l1956_195672

theorem problem_1 : 12 - (-10) + 7 = 29 := by sorry

end problem_1_l1956_195672


namespace quadratic_integer_roots_l1956_195615

theorem quadratic_integer_roots (a : ℤ) :
  (∃ x y : ℤ, (a + 1) * x^2 - (a^2 + 1) * x + 2 * a^3 - 6 = 0 ∧
               (a + 1) * y^2 - (a^2 + 1) * y + 2 * a^3 - 6 = 0 ∧
               x ≠ y) ↔
  (a = 0 ∨ a = 1) :=
by sorry

end quadratic_integer_roots_l1956_195615


namespace max_blue_points_l1956_195690

/-- The maximum number of blue points when 2016 spheres are colored red or green -/
theorem max_blue_points (total_spheres : Nat) (h : total_spheres = 2016) :
  ∃ (red_spheres : Nat),
    red_spheres ≤ total_spheres ∧
    red_spheres * (total_spheres - red_spheres) = 1016064 ∧
    ∀ (x : Nat), x ≤ total_spheres →
      x * (total_spheres - x) ≤ 1016064 := by
  sorry

end max_blue_points_l1956_195690


namespace roots_equation_relation_l1956_195678

theorem roots_equation_relation (p q a b c : ℝ) : 
  (a^2 + p*a + 1 = 0) → 
  (b^2 + p*b + 1 = 0) → 
  (b^2 + q*b + 2 = 0) → 
  (c^2 + q*c + 2 = 0) → 
  (b-a)*(b-c) = p*q - 6 := by
  sorry

end roots_equation_relation_l1956_195678


namespace congruence_solution_a_l1956_195655

theorem congruence_solution_a (x : Int) : 
  (8 * x) % 13 = 3 % 13 ↔ x % 13 = 2 % 13 := by sorry

end congruence_solution_a_l1956_195655


namespace min_value_of_function_l1956_195677

theorem min_value_of_function (x y : ℝ) : 2*x^2 + 3*y^2 + 8*x - 6*y + 5*x*y + 36 ≥ 10 := by
  sorry

end min_value_of_function_l1956_195677


namespace sum_of_specific_terms_l1956_195659

/-- Arithmetic sequence with first term 1 and common difference 2 -/
def arithmetic_seq (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- Geometric sequence with first term 1 and common ratio 2 -/
def geometric_seq (n : ℕ) : ℕ := 2^(n - 1)

/-- The sum of specific terms in the arithmetic sequence -/
theorem sum_of_specific_terms :
  arithmetic_seq (geometric_seq 2) + 
  arithmetic_seq (geometric_seq 3) + 
  arithmetic_seq (geometric_seq 4) = 25 :=
by sorry

end sum_of_specific_terms_l1956_195659


namespace marble_ratio_proof_l1956_195603

def marble_problem (initial_marbles : ℕ) (lost_marbles : ℕ) (final_marbles : ℕ) : Prop :=
  let marbles_after_loss := initial_marbles - lost_marbles
  let marbles_given_away := 2 * lost_marbles
  let marbles_before_dog_ate := marbles_after_loss - marbles_given_away
  let marbles_eaten_by_dog := marbles_before_dog_ate - final_marbles
  (2 * marbles_eaten_by_dog = lost_marbles) ∧ (marbles_eaten_by_dog > 0) ∧ (lost_marbles > 0)

theorem marble_ratio_proof : marble_problem 24 4 10 := by sorry

end marble_ratio_proof_l1956_195603


namespace complex_fraction_simplification_l1956_195665

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * Complex.I
  let z₂ : ℂ := 4 - 7 * Complex.I
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 := by
  sorry

end complex_fraction_simplification_l1956_195665


namespace triangle_perimeter_bound_l1956_195692

theorem triangle_perimeter_bound (A B C : ℝ) (R : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  ConvexOn ℝ (Set.Ioo 0 π) Real.sin →
  2 * R * (Real.sin A + Real.sin B + Real.sin C) ≤ 3 * Real.sqrt 3 * R :=
sorry

end triangle_perimeter_bound_l1956_195692


namespace sunflower_height_l1956_195687

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet, rounding down -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem sunflower_height
  (sister_height_feet : ℕ)
  (sister_height_inches : ℕ)
  (sunflower_diff : ℕ)
  (h1 : sister_height_feet = 4)
  (h2 : sister_height_inches = 3)
  (h3 : sunflower_diff = 21) :
  inches_to_feet (feet_inches_to_inches sister_height_feet sister_height_inches + sunflower_diff) = 6 := by
  sorry

end sunflower_height_l1956_195687


namespace other_x_intercept_of_quadratic_l1956_195686

/-- Given a quadratic function with vertex (6, -2) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 11. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = -2 ↔ x = 6) →  -- vertex condition
  a * 1^2 + b * 1 + c = 0 →                  -- x-intercept condition
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 11 := by
  sorry

end other_x_intercept_of_quadratic_l1956_195686


namespace polynomial_nonnegative_min_value_of_f_l1956_195656

-- Part a
theorem polynomial_nonnegative (x : ℝ) (h : x ≥ 1) :
  x^3 - 5*x^2 + 8*x - 4 ≥ 0 := by sorry

-- Part b
def f (a b : ℝ) := a*b*(a+b-10) + 8*(a+b)

theorem min_value_of_f :
  ∃ (min : ℝ), min = 8 ∧ 
  ∀ (a b : ℝ), a ≥ 1 → b ≥ 1 → f a b ≥ min := by sorry

end polynomial_nonnegative_min_value_of_f_l1956_195656


namespace translated_line_equation_l1956_195660

/-- 
Theorem: The equation of a line with slope 2 passing through the point (2, 5) is y = 2x + 1.
-/
theorem translated_line_equation (x y : ℝ) : 
  (∃ b : ℝ, y = 2 * x + b) ∧ (2 = 2 ∧ 5 = 2 * 2 + y - 2 * x) → y = 2 * x + 1 := by
  sorry

end translated_line_equation_l1956_195660


namespace quadrilateral_is_parallelogram_l1956_195637

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define an acute angle
structure AcuteAngle where
  vertex : Point2D
  side1 : Line2D
  side2 : Line2D

-- Function to calculate the distance from a point to a line
def distancePointToLine (p : Point2D) (l : Line2D) : ℝ :=
  sorry

-- Function to check if a quadrilateral is convex
def isConvex (q : Quadrilateral) : Prop :=
  sorry

-- Function to check if a quadrilateral is within an angle
def isWithinAngle (q : Quadrilateral) (angle : AcuteAngle) : Prop :=
  sorry

-- Function to check if a quadrilateral is a parallelogram
def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem quadrilateral_is_parallelogram
  (q : Quadrilateral)
  (angle : AcuteAngle)
  (h_convex : isConvex q)
  (h_within : isWithinAngle q angle)
  (h_distance1 : distancePointToLine q.A angle.side1 + distancePointToLine q.C angle.side1 =
                 distancePointToLine q.B angle.side1 + distancePointToLine q.D angle.side1)
  (h_distance2 : distancePointToLine q.A angle.side2 + distancePointToLine q.C angle.side2 =
                 distancePointToLine q.B angle.side2 + distancePointToLine q.D angle.side2) :
  isParallelogram q :=
by
  sorry

end quadrilateral_is_parallelogram_l1956_195637


namespace unknown_van_capacity_l1956_195633

/-- Represents the fleet of vans with their capacities -/
structure Fleet :=
  (total_vans : Nat)
  (standard_capacity : Nat)
  (large_vans : Nat)
  (small_vans : Nat)
  (unknown_van : Nat)
  (total_capacity : Nat)

/-- Theorem stating the capacity of the unknown van -/
theorem unknown_van_capacity (f : Fleet)
  (h1 : f.total_vans = 6)
  (h2 : f.standard_capacity = 8000)
  (h3 : f.large_vans = 3)
  (h4 : f.small_vans = 2)
  (h5 : f.unknown_van = 1)
  (h6 : f.total_capacity = 57600)
  (h7 : f.large_vans * (f.standard_capacity + f.standard_capacity / 2) +
        f.small_vans * f.standard_capacity +
        (f.total_capacity - (f.large_vans * (f.standard_capacity + f.standard_capacity / 2) +
                             f.small_vans * f.standard_capacity)) =
        f.total_capacity) :
  (f.total_capacity - (f.large_vans * (f.standard_capacity + f.standard_capacity / 2) +
                       f.small_vans * f.standard_capacity)) =
  (f.standard_capacity * 7 / 10) :=
by sorry

end unknown_van_capacity_l1956_195633


namespace exists_x0_where_P_less_than_Q_l1956_195611

/-- Given two polynomials P and Q, and an interval (r,s) satisfying certain conditions,
    there exists a real x₀ such that P(x₀) < Q(x₀) -/
theorem exists_x0_where_P_less_than_Q
  (a b c d p q r s : ℝ)
  (P : ℝ → ℝ)
  (Q : ℝ → ℝ)
  (h_P : ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + d)
  (h_Q : ∀ x, Q x = x^2 + p*x + q)
  (h_interval : s - r > 2)
  (h_negative : ∀ x, r < x ∧ x < s → P x < 0 ∧ Q x < 0)
  (h_positive_right : ∀ x, x > s → P x > 0 ∧ Q x > 0)
  (h_positive_left : ∀ x, x < r → P x > 0 ∧ Q x > 0) :
  ∃ x₀, P x₀ < Q x₀ :=
by sorry

end exists_x0_where_P_less_than_Q_l1956_195611


namespace carpenter_problem_solution_l1956_195675

/-- Represents the carpenter problem -/
def CarpenterProblem (x : ℝ) : Prop :=
  let first_carpenter_rate := 1 / (x + 4)
  let second_carpenter_rate := 1 / 5
  let combined_rate := first_carpenter_rate + second_carpenter_rate
  2 * combined_rate = 4 * first_carpenter_rate

/-- The solution to the carpenter problem is 1 day -/
theorem carpenter_problem_solution :
  ∃ (x : ℝ), CarpenterProblem x ∧ x = 1 :=
sorry

end carpenter_problem_solution_l1956_195675


namespace percentage_decrease_in_z_l1956_195676

/-- Given positive real numbers x and z, and a real number q, if x and (z+10) are inversely 
    proportional, and x increases by q%, then the percentage decrease in z is q(z+10)/(100+q)% -/
theorem percentage_decrease_in_z (x z q : ℝ) (hx : x > 0) (hz : z > 0) (hq : q ≠ -100) :
  (∃ k : ℝ, k > 0 ∧ x * (z + 10) = k) →
  let x' := x * (1 + q / 100)
  let z' := (100 / (100 + q)) * (z + 10) - 10
  (z - z') / z * 100 = q * (z + 10) / (100 + q) := by
  sorry

end percentage_decrease_in_z_l1956_195676


namespace trees_survived_difference_l1956_195654

theorem trees_survived_difference (initial_trees died_trees : ℕ) 
  (h1 : initial_trees = 11)
  (h2 : died_trees = 2) :
  initial_trees - died_trees - died_trees = 7 := by
  sorry

end trees_survived_difference_l1956_195654


namespace vegetable_project_profit_l1956_195613

-- Define the constants
def initial_investment : ℝ := 600000
def first_year_expense : ℝ := 80000
def annual_expense_increase : ℝ := 20000
def annual_income : ℝ := 260000

-- Define the net profit function
def f (n : ℝ) : ℝ := -n^2 + 19*n - 60

-- Define the theorem to prove
theorem vegetable_project_profit (n : ℝ) :
  f n = n * annual_income - 
    (n * first_year_expense + (n * (n - 1) / 2) * annual_expense_increase) - 
    initial_investment / 10000 ∧
  (∀ m : ℝ, m < 5 → f m ≤ 0) ∧
  (∀ m : ℝ, m ≥ 5 → f m > 0) :=
sorry

end vegetable_project_profit_l1956_195613


namespace increasing_function_t_bound_l1956_195612

def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 3

theorem increasing_function_t_bound (t : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → f t x₁ < f t x₂) →
  t ≤ 1 :=
by sorry

end increasing_function_t_bound_l1956_195612


namespace unique_solution_l1956_195674

theorem unique_solution : ∃! (x y : ℝ), 
  (2 * x + 3 * y = (7 - 2 * x) + (7 - 3 * y)) ∧ 
  (x - 2 * y = (x - 2) + (2 * y - 2)) ∧
  x = 2 ∧ y = 1 := by sorry

end unique_solution_l1956_195674


namespace floor_equation_solution_l1956_195699

theorem floor_equation_solution (a b : ℕ+) :
  (Int.floor (a^2 / b : ℚ) + Int.floor (b^2 / a : ℚ) = 
   Int.floor ((a^2 + b^2) / (a * b) : ℚ) + a * b) ↔ 
  (a = b^2 + 1) :=
by sorry

end floor_equation_solution_l1956_195699


namespace problem_solution_l1956_195662

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

theorem problem_solution (a : ℝ) (h : a > 0) :
  -- Part 1
  (∀ x, f 1 x ≥ 3 * x + 2 ↔ x ≥ 3 ∨ x ≤ -1) ∧
  -- Part 2
  ((∀ x, f a x ≤ 0 ↔ x ≤ -1) → a = 2) :=
sorry

end problem_solution_l1956_195662


namespace train_bridge_crossing_time_l1956_195638

/-- Calculates the time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (lamp_post_time : ℝ) 
  (h1 : train_length = 75) 
  (h2 : bridge_length = 150) 
  (h3 : lamp_post_time = 2.5) : 
  (train_length + bridge_length) / (train_length / lamp_post_time) = 7.5 := by
  sorry


end train_bridge_crossing_time_l1956_195638


namespace function_and_intersection_points_l1956_195685

noncomputable def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem function_and_intersection_points 
  (b c d : ℝ) 
  (h1 : f b c d 0 = 2) 
  (h2 : (6 : ℝ) * (-1) - f b c d (-1) + 7 = 0) 
  (h3 : (6 : ℝ) = (3 * (-1)^2 + 2*b*(-1) + c)) :
  (∀ x, f b c d x = x^3 - 3*x^2 - 3*x + 2) ∧
  (∀ a, (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f b c d x₁ = (3/2)*x₁^2 - 9*x₁ + a + 2 ∧
    f b c d x₂ = (3/2)*x₂^2 - 9*x₂ + a + 2 ∧
    f b c d x₃ = (3/2)*x₃^2 - 9*x₃ + a + 2) →
  2 < a ∧ a < 5/2) :=
by sorry

end function_and_intersection_points_l1956_195685


namespace parallel_condition_l1956_195663

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ ≠ 0 ∧ m₂ ≠ 0

/-- The first line: ax - y + 3 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x - y + 3 = 0

/-- The second line: 2x - (a + 1)y + 4 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  2 * x - (a + 1) * y + 4 = 0

/-- a = -2 is a sufficient but not necessary condition for the lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  (a = -2 → are_parallel a (-1) 3 2 (-(a + 1)) 4) ∧
  ¬(are_parallel a (-1) 3 2 (-(a + 1)) 4 → a = -2) :=
sorry

end parallel_condition_l1956_195663


namespace some_number_calculation_l1956_195691

theorem some_number_calculation (X : ℝ) : 
  2 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1600.0000000000002 → X = 0.3 := by
  sorry

end some_number_calculation_l1956_195691


namespace unique_function_is_lcm_l1956_195601

def satisfies_conditions (f : ℕ → ℕ → ℕ) : Prop :=
  (∀ m n, f m n = f n m) ∧
  (∀ n, f n n = n) ∧
  (∀ m n, n > m → (n - m) * f m n = n * f m (n - m))

theorem unique_function_is_lcm :
  ∀ f : ℕ → ℕ → ℕ, satisfies_conditions f → ∀ m n, f m n = Nat.lcm m n := by
  sorry

end unique_function_is_lcm_l1956_195601


namespace n_squared_not_divides_n_factorial_l1956_195642

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem n_squared_not_divides_n_factorial (n : ℕ) :
  (¬ divides (n^2) (n!)) ↔ (Nat.Prime n ∨ n = 4) := by sorry

end n_squared_not_divides_n_factorial_l1956_195642


namespace quadratic_sum_l1956_195623

/-- Given a quadratic expression x^2 - 20x + 49, prove that when written in the form (x+b)^2 + c,
    the sum of b and c is equal to -61. -/
theorem quadratic_sum (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 49 = (x + b)^2 + c) → b + c = -61 := by
  sorry

end quadratic_sum_l1956_195623


namespace number_of_baskets_is_one_l1956_195608

-- Define the total number of peaches
def total_peaches : ℕ := 10

-- Define the number of red peaches per basket
def red_peaches_per_basket : ℕ := 4

-- Define the number of green peaches per basket
def green_peaches_per_basket : ℕ := 6

-- Theorem to prove
theorem number_of_baskets_is_one :
  let peaches_per_basket := red_peaches_per_basket + green_peaches_per_basket
  total_peaches / peaches_per_basket = 1 := by
  sorry

end number_of_baskets_is_one_l1956_195608


namespace unique_number_outside_range_l1956_195604

theorem unique_number_outside_range 
  (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = (a * x + b) / (c * x + d))
  (hf19 : f 19 = 19)
  (hf97 : f 97 = 97)
  (hfinv : ∀ x, x ≠ -d/c → f (f x) = x) :
  ∃! y, ∀ x, f x ≠ y ∧ y = 58 :=
by sorry

end unique_number_outside_range_l1956_195604


namespace right_triangle_properties_l1956_195627

/-- Properties of a specific right triangle -/
theorem right_triangle_properties :
  ∀ (a b c : ℝ),
  a = 24 →
  b = 2 * a + 10 →
  c^2 = a^2 + b^2 →
  (1/2 * a * b = 696) ∧ (c = Real.sqrt 3940) := by
  sorry

end right_triangle_properties_l1956_195627


namespace cats_at_rescue_center_l1956_195610

/-- The number of puppies Hartley has -/
def num_puppies : ℕ := 4

/-- The weight of each puppy in kilograms -/
def puppy_weight : ℚ := 7.5

/-- The weight of each cat in kilograms -/
def cat_weight : ℚ := 2.5

/-- The difference in total weight between cats and puppies in kilograms -/
def weight_difference : ℚ := 5

/-- The number of cats at the rescue center -/
def num_cats : ℕ := 14

theorem cats_at_rescue_center : 
  (↑num_cats : ℚ) * cat_weight = 
    ↑num_puppies * puppy_weight + weight_difference := by
  sorry

end cats_at_rescue_center_l1956_195610


namespace fraction_equality_l1956_195645

theorem fraction_equality (a b : ℝ) (h : (4*a + 3*b) / (4*a - 3*b) = 4) : a / b = 5/4 := by
  sorry

end fraction_equality_l1956_195645


namespace digit_cancellation_fractions_l1956_195696

theorem digit_cancellation_fractions :
  let valid_fractions : List (ℕ × ℕ) := [(26, 65), (16, 64), (19, 95), (49, 98)]
  ∀ a b c : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 →
    (10 * a + b) * c = (10 * b + c) * a →
    a < c →
    (10 * a + b, 10 * b + c) ∈ valid_fractions :=
by sorry

end digit_cancellation_fractions_l1956_195696


namespace product_trailing_zeros_l1956_195635

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 45 and 320 -/
def product : ℕ := 45 * 320

/-- Theorem: The number of trailing zeros in the product 45 × 320 is 2 -/
theorem product_trailing_zeros : trailingZeros product = 2 := by sorry

end product_trailing_zeros_l1956_195635


namespace dance_event_relation_l1956_195666

/-- Represents a dance event with boys and girls -/
structure DanceEvent where
  b : ℕ  -- Total number of boys
  g : ℕ  -- Total number of girls

/-- The number of girls the nth boy dances with -/
def girlsForBoy (n : ℕ) : ℕ := 7 + 2 * (n - 1)

/-- Axiom: The last boy dances with all girls -/
axiom last_boy_dances_all (event : DanceEvent) : girlsForBoy event.b = event.g

/-- Theorem: The relationship between boys and girls in the dance event -/
theorem dance_event_relation (event : DanceEvent) : event.b = (event.g - 5) / 2 := by
  sorry


end dance_event_relation_l1956_195666


namespace island_perimeter_calculation_l1956_195670

/-- The perimeter of a rectangular island -/
def island_perimeter (width : ℝ) (length : ℝ) : ℝ :=
  2 * (width + length)

/-- Theorem: The perimeter of a rectangular island with width 4 miles and length 7 miles is 22 miles -/
theorem island_perimeter_calculation :
  island_perimeter 4 7 = 22 := by
  sorry

end island_perimeter_calculation_l1956_195670


namespace trumpets_fraction_in_band_l1956_195629

-- Define the total number of each instrument
def total_flutes : ℕ := 20
def total_clarinets : ℕ := 30
def total_trumpets : ℕ := 60
def total_pianists : ℕ := 20

-- Define the fraction of each instrument that got in
def flutes_fraction : ℚ := 4/5  -- 80%
def clarinets_fraction : ℚ := 1/2
def pianists_fraction : ℚ := 1/10

-- Define the total number of people in the band
def total_in_band : ℕ := 53

-- Theorem to prove
theorem trumpets_fraction_in_band : 
  (total_in_band - 
   (flutes_fraction * total_flutes + 
    clarinets_fraction * total_clarinets + 
    pianists_fraction * total_pianists)) / total_trumpets = 1/3 := by
  sorry

end trumpets_fraction_in_band_l1956_195629


namespace complex_number_magnitude_l1956_195618

theorem complex_number_magnitude (a : ℝ) :
  let i : ℂ := Complex.I
  let z : ℂ := (a - i)^2
  (∃ b : ℝ, z = b * i) → Complex.abs z = 2 := by
  sorry

end complex_number_magnitude_l1956_195618


namespace original_number_is_107_l1956_195679

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def increase_digits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  (hundreds + 3) * 100 + (tens + 2) * 10 + (units + 1)

theorem original_number_is_107 :
  is_three_digit_number 107 ∧ increase_digits 107 = 4 * 107 :=
sorry

end original_number_is_107_l1956_195679


namespace max_ratio_concentric_circles_polyline_l1956_195606

/-- The maximum common ratio for concentric circles allowing a closed polyline -/
theorem max_ratio_concentric_circles_polyline :
  ∃ (q : ℝ), q = (Real.sqrt 5 + 1) / 2 ∧
  ∀ (r : ℝ) (A : Fin 5 → ℝ × ℝ),
  (∀ i : Fin 5, ‖A i‖ = r * q ^ i.val) →
  (∀ i : Fin 5, ‖A i - A (i + 1)‖ = ‖A 0 - A 1‖) →
  (A 0 = A 4) →
  ∀ q' > q, ¬∃ (A' : Fin 5 → ℝ × ℝ),
    (∀ i : Fin 5, ‖A' i‖ = r * q' ^ i.val) ∧
    (∀ i : Fin 5, ‖A' i - A' (i + 1)‖ = ‖A' 0 - A' 1‖) ∧
    (A' 0 = A' 4) :=
by sorry

end max_ratio_concentric_circles_polyline_l1956_195606


namespace triangle_abc_properties_l1956_195693

/-- Triangle ABC with given properties -/
structure TriangleABC where
  /-- Point A coordinates -/
  A : ℝ × ℝ
  /-- Equation of median CM: 5√3x + 9y - 18 = 0 -/
  CM : ℝ → ℝ → Prop
  /-- Equation of angle bisector BT: y = 1 -/
  BT : ℝ → ℝ → Prop

/-- Properties of the given triangle -/
def triangle_properties (t : TriangleABC) : Prop :=
  t.A = (Real.sqrt 3, 3) ∧
  (∀ x y, t.CM x y ↔ 5 * Real.sqrt 3 * x + 9 * y - 18 = 0) ∧
  (∀ x y, t.BT x y ↔ y = 1)

/-- Theorem stating the properties of vertex B and the area of the triangle -/
theorem triangle_abc_properties (t : TriangleABC) 
  (h : triangle_properties t) : 
  (∃ B : ℝ × ℝ, B = (-Real.sqrt 3, 1)) ∧ 
  (∃ area : ℝ, area = 8 * Real.sqrt 3) :=
sorry

end triangle_abc_properties_l1956_195693


namespace football_team_girls_l1956_195684

theorem football_team_girls (total : ℕ) (attended : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 30 →
  attended = 18 →
  girls + boys = total →
  attended = boys + (girls / 3) →
  girls = 18 :=
by sorry

end football_team_girls_l1956_195684


namespace three_digit_squares_ending_1001_l1956_195673

theorem three_digit_squares_ending_1001 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → (n^2 % 10000 = 1001 ↔ n = 501 ∨ n = 749) :=
by sorry

end three_digit_squares_ending_1001_l1956_195673


namespace subtract_sqrt_25_equals_negative_2_l1956_195617

theorem subtract_sqrt_25_equals_negative_2 : 3 - Real.sqrt 25 = -2 := by
  sorry

end subtract_sqrt_25_equals_negative_2_l1956_195617


namespace initial_crayons_count_l1956_195632

/-- The initial number of crayons in the drawer -/
def initial_crayons : ℕ := sorry

/-- The number of pencils in the drawer -/
def pencils : ℕ := 26

/-- The number of crayons added to the drawer -/
def added_crayons : ℕ := 12

/-- The total number of crayons after adding -/
def total_crayons : ℕ := 53

theorem initial_crayons_count : initial_crayons = 41 := by
  sorry

end initial_crayons_count_l1956_195632


namespace part_one_part_two_l1956_195664

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | (x - a) * (x + 1) > 0}
def Q : Set ℝ := {x | |x - 1| ≤ 1}

-- Part 1
theorem part_one : (Set.univ \ P 1) ∪ Q = {x | -1 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : P a ∩ Q = ∅) : a > 2 := by sorry

end part_one_part_two_l1956_195664


namespace excellent_round_probability_l1956_195646

/-- Represents the result of a single dart throw -/
inductive DartThrow
| Miss  : DartThrow  -- Didn't land in 8th ring or higher
| Hit   : DartThrow  -- Landed in 8th ring or higher

/-- Represents a round of 3 dart throws -/
def Round := (DartThrow × DartThrow × DartThrow)

/-- Determines if a round is excellent (at least 2 hits) -/
def is_excellent (r : Round) : Bool :=
  match r with
  | (DartThrow.Hit, DartThrow.Hit, _) => true
  | (DartThrow.Hit, _, DartThrow.Hit) => true
  | (_, DartThrow.Hit, DartThrow.Hit) => true
  | _ => false

/-- The total number of rounds in the experiment -/
def total_rounds : Nat := 20

/-- The number of excellent rounds observed -/
def excellent_rounds : Nat := 12

/-- Theorem: The probability of an excellent round is 0.6 -/
theorem excellent_round_probability :
  (excellent_rounds : ℚ) / total_rounds = 0.6 := by sorry

end excellent_round_probability_l1956_195646


namespace expression_simplification_l1956_195626

theorem expression_simplification (a : ℝ) (h : a = 1 - Real.sqrt 3) :
  (1 - (2 * a - 1) / (a ^ 2)) / ((a - 1) / (a ^ 2)) = -Real.sqrt 3 := by
  sorry

end expression_simplification_l1956_195626


namespace parabola_translation_l1956_195614

/-- Given a parabola y = 3x² in the original coordinate system,
    if the x-axis is translated 2 units up and the y-axis is translated 2 units to the right,
    then the equation of the parabola in the new coordinate system is y = 3(x + 2)² - 2 -/
theorem parabola_translation (x y : ℝ) :
  (y = 3 * x^2) →
  (∀ x' y', x' = x - 2 ∧ y' = y - 2) →
  (y = 3 * (x + 2)^2 - 2) :=
by sorry

end parabola_translation_l1956_195614


namespace no_rain_percentage_l1956_195668

theorem no_rain_percentage (p_monday : ℝ) (p_tuesday : ℝ) (p_both : ℝ) 
  (h_monday : p_monday = 0.7)
  (h_tuesday : p_tuesday = 0.55)
  (h_both : p_both = 0.6) :
  1 - (p_monday + p_tuesday - p_both) = 0.35 := by
  sorry

end no_rain_percentage_l1956_195668


namespace soccer_team_subjects_l1956_195698

theorem soccer_team_subjects (total : ℕ) (history : ℕ) (both : ℕ) (geography : ℕ) : 
  total = 18 → 
  history = 10 → 
  both = 6 → 
  geography = total - (history - both) → 
  geography = 14 := by
sorry

end soccer_team_subjects_l1956_195698


namespace equation_equality_l1956_195622

theorem equation_equality (x y z : ℝ) (h : x / y = 3 / z) : 9 * y^2 = x^2 * z^2 := by
  sorry

end equation_equality_l1956_195622


namespace fundraising_problem_l1956_195634

/-- Represents a student with their fundraising goal -/
structure Student where
  name : String
  goal : ℕ

/-- Represents a day's fundraising activity -/
structure FundraisingDay where
  income : ℕ
  expense : ℕ

/-- The fundraising problem -/
theorem fundraising_problem 
  (students : List Student)
  (collective_goal : ℕ)
  (fundraising_days : List FundraisingDay)
  (h1 : students.length = 8)
  (h2 : collective_goal = 3500)
  (h3 : fundraising_days.length = 5)
  (h4 : students.map Student.goal = [350, 450, 500, 550, 600, 650, 450, 550])
  (h5 : fundraising_days.map FundraisingDay.income = [800, 950, 500, 700, 550])
  (h6 : fundraising_days.map FundraisingDay.expense = [100, 150, 50, 75, 100]) :
  (students.map Student.goal = [350, 450, 500, 550, 600, 650, 450, 550]) ∧
  ((fundraising_days.map (λ d => d.income - d.expense)).sum + 3975 = collective_goal + (students.map Student.goal).sum) :=
sorry

end fundraising_problem_l1956_195634


namespace pen_price_theorem_l1956_195697

/-- Given the conditions of the pen and pencil purchase, prove the average price of a pen. -/
theorem pen_price_theorem (total_pens : ℕ) (total_pencils : ℕ) (total_cost : ℚ) (avg_pencil_price : ℚ) :
  total_pens = 30 →
  total_pencils = 75 →
  total_cost = 510 →
  avg_pencil_price = 2 →
  (total_cost - total_pencils * avg_pencil_price) / total_pens = 12 :=
by sorry

end pen_price_theorem_l1956_195697


namespace lcm_problem_l1956_195631

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 36 := by
  sorry

end lcm_problem_l1956_195631


namespace parallelogram_side_length_comparison_l1956_195653

/-- Represents a parallelogram in 2D space -/
structure Parallelogram :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Checks if parallelogram inner is inside parallelogram outer -/
def is_inside (inner outer : Parallelogram) : Prop := sorry

/-- Checks if the vertices of inner are on the edges of outer -/
def vertices_on_edges (inner outer : Parallelogram) : Prop := sorry

/-- Checks if the sides of para1 are parallel to the sides of para2 -/
def sides_parallel (para1 para2 : Parallelogram) : Prop := sorry

/-- Computes the length of a side of a parallelogram -/
def side_length (p : Parallelogram) (side : Fin 4) : ℝ := sorry

theorem parallelogram_side_length_comparison 
  (P1 P2 P3 : Parallelogram) 
  (h1 : is_inside P3 P2)
  (h2 : is_inside P2 P1)
  (h3 : vertices_on_edges P3 P2)
  (h4 : vertices_on_edges P2 P1)
  (h5 : sides_parallel P3 P1) :
  ∃ (side : Fin 4), side_length P3 side ≥ (side_length P1 side) / 2 := by
  sorry

end parallelogram_side_length_comparison_l1956_195653


namespace smallest_prime_ten_less_square_l1956_195681

theorem smallest_prime_ten_less_square : ∃ (n : ℕ), 
  (∀ m : ℕ, m < n → ¬(Nat.Prime m ∧ ∃ k : ℕ, m = k^2 - 10)) ∧ 
  (Nat.Prime n ∧ ∃ k : ℕ, n = k^2 - 10) :=
by sorry

end smallest_prime_ten_less_square_l1956_195681


namespace product_inequality_l1956_195650

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end product_inequality_l1956_195650


namespace matrix_product_l1956_195628

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 4]

theorem matrix_product :
  A * B = !![17, -5; 16, -20] := by sorry

end matrix_product_l1956_195628


namespace path_area_is_78_l1956_195640

/-- Represents the dimensions of a garden with flower beds and paths. -/
structure GardenDimensions where
  rows : Nat
  columns : Nat
  bedLength : Nat
  bedWidth : Nat
  pathWidth : Nat

/-- Calculates the total area of paths in a garden given its dimensions. -/
def pathArea (g : GardenDimensions) : Nat :=
  let totalWidth := g.pathWidth + g.columns * g.bedLength + (g.columns - 1) * g.pathWidth + g.pathWidth
  let totalHeight := g.pathWidth + g.rows * g.bedWidth + (g.rows - 1) * g.pathWidth + g.pathWidth
  let totalArea := totalWidth * totalHeight
  let bedArea := g.rows * g.columns * g.bedLength * g.bedWidth
  totalArea - bedArea

/-- Theorem stating that the path area for the given garden dimensions is 78 square feet. -/
theorem path_area_is_78 (g : GardenDimensions) 
    (h1 : g.rows = 3) 
    (h2 : g.columns = 2) 
    (h3 : g.bedLength = 6) 
    (h4 : g.bedWidth = 2) 
    (h5 : g.pathWidth = 1) : 
  pathArea g = 78 := by
  sorry

end path_area_is_78_l1956_195640


namespace same_price_at_12_sheets_unique_equal_price_at_12_sheets_l1956_195643

/-- Represents the pricing structure of a photo company -/
structure PhotoCompany where
  perSheetCost : ℚ
  sittingFee : ℚ

/-- Calculates the total cost for a given number of sheets -/
def totalCost (company : PhotoCompany) (sheets : ℚ) : ℚ :=
  company.perSheetCost * sheets + company.sittingFee

/-- John's Photo World pricing -/
def johnsPhotoWorld : PhotoCompany :=
  { perSheetCost := 2.75, sittingFee := 125 }

/-- Sam's Picture Emporium pricing -/
def samsPictureEmporium : PhotoCompany :=
  { perSheetCost := 1.50, sittingFee := 140 }

/-- Theorem stating that the companies charge the same for 12 sheets -/
theorem same_price_at_12_sheets :
  totalCost johnsPhotoWorld 12 = totalCost samsPictureEmporium 12 := by
  sorry

/-- Theorem stating that 12 is the unique number of sheets where prices are equal -/
theorem unique_equal_price_at_12_sheets :
  ∀ x : ℚ, totalCost johnsPhotoWorld x = totalCost samsPictureEmporium x ↔ x = 12 := by
  sorry

end same_price_at_12_sheets_unique_equal_price_at_12_sheets_l1956_195643


namespace binary_to_quinary_conversion_l1956_195689

/-- Converts a natural number from base 2 to base 10 -/
def base2_to_base10 (n : List Bool) : ℕ :=
  n.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number from base 10 to base 5 -/
def base10_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_to_quinary_conversion :
  base10_to_base5 (base2_to_base10 [true, false, true, true, true]) = [1, 0, 4] := by
  sorry

end binary_to_quinary_conversion_l1956_195689


namespace benny_savings_theorem_l1956_195657

/-- The amount of money Benny adds to his piggy bank in January -/
def january_savings : ℕ := 19

/-- The amount of money Benny adds to his piggy bank in February -/
def february_savings : ℕ := january_savings

/-- The amount of money Benny adds to his piggy bank in March -/
def march_savings : ℕ := 8

/-- The total amount of money in Benny's piggy bank by the end of March -/
def total_savings : ℕ := january_savings + february_savings + march_savings

/-- Theorem stating that the total amount in Benny's piggy bank by the end of March is $46 -/
theorem benny_savings_theorem : total_savings = 46 := by
  sorry

end benny_savings_theorem_l1956_195657


namespace two_digit_numbers_problem_l1956_195647

theorem two_digit_numbers_problem :
  ∃ (x y : ℕ), 
    x > y ∧ 
    x ≥ 10 ∧ x < 100 ∧ 
    y ≥ 10 ∧ y < 100 ∧ 
    1000 * x + y = 2 * (1000 * y + 10 * x) + 590 ∧
    2 * x + 3 * y = 72 ∧
    x = 21 ∧ 
    y = 10 ∧
    ∀ (a b : ℕ), 
      (a > b ∧ 
       a ≥ 10 ∧ a < 100 ∧ 
       b ≥ 10 ∧ b < 100 ∧ 
       1000 * a + b = 2 * (1000 * b + 10 * a) + 590 ∧
       2 * a + 3 * b = 72) → 
      (a = 21 ∧ b = 10) :=
by sorry


end two_digit_numbers_problem_l1956_195647


namespace sandy_correct_sums_l1956_195605

theorem sandy_correct_sums :
  ∀ (c i : ℕ),
    c + i = 50 →
    3 * c - 2 * i - (50 - c) = 100 →
    c = 25 := by
  sorry

end sandy_correct_sums_l1956_195605


namespace equation_solution_l1956_195630

theorem equation_solution (x : ℝ) : 
  x ≠ 1 → -x^2 = (2*x + 4)/(x - 1) → x = -2 ∨ x = 1 := by
  sorry

end equation_solution_l1956_195630


namespace two_sided_icing_count_l1956_195600

/-- Represents a cubic cake with icing on specific faces -/
structure CubeCake where
  size : Nat
  has_top_icing : Bool
  has_bottom_icing : Bool
  has_side_icing : Bool
  has_middle_layer_icing : Bool

/-- Counts the number of 1×1×1 sub-cubes with icing on exactly two sides -/
def count_two_sided_icing (cake : CubeCake) : Nat :=
  sorry

/-- The main theorem stating that a 5×5×5 cake with specific icing has 24 sub-cubes with icing on two sides -/
theorem two_sided_icing_count :
  let cake : CubeCake := {
    size := 5,
    has_top_icing := true,
    has_bottom_icing := false,
    has_side_icing := true,
    has_middle_layer_icing := true
  }
  count_two_sided_icing cake = 24 := by sorry

end two_sided_icing_count_l1956_195600


namespace avery_chicken_count_l1956_195652

theorem avery_chicken_count :
  ∀ (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) (filled_cartons : ℕ),
    eggs_per_chicken = 6 →
    eggs_per_carton = 12 →
    filled_cartons = 10 →
    filled_cartons * eggs_per_carton / eggs_per_chicken = 20 :=
by sorry

end avery_chicken_count_l1956_195652


namespace base_value_l1956_195609

theorem base_value (b x y : ℕ) (h1 : b^x * 4^y = 59049) (h2 : x - y = 10) (h3 : x = 10) : b = 3 := by
  sorry

end base_value_l1956_195609


namespace female_rabbits_count_l1956_195680

theorem female_rabbits_count (white_rabbits black_rabbits male_rabbits : ℕ) 
  (h1 : white_rabbits = 11)
  (h2 : black_rabbits = 13)
  (h3 : male_rabbits = 15) : 
  white_rabbits + black_rabbits - male_rabbits = 9 := by
  sorry

end female_rabbits_count_l1956_195680


namespace sqrt_equality_implies_t_equals_three_l1956_195602

theorem sqrt_equality_implies_t_equals_three (t : ℝ) : 
  Real.sqrt (2 * Real.sqrt (t - 2)) = (7 - t) ^ (1/4) → t = 3 := by
  sorry

end sqrt_equality_implies_t_equals_three_l1956_195602


namespace matchstick_subtraction_theorem_l1956_195616

/-- Represents a collection of matchsticks -/
structure MatchstickSet :=
  (count : ℕ)

/-- Represents a Roman numeral -/
inductive RomanNumeral
  | I
  | V
  | X
  | L
  | C
  | D
  | M

/-- Function to determine if a given number of matchsticks can form a Roman numeral -/
def can_form_roman_numeral (m : MatchstickSet) (r : RomanNumeral) : Prop :=
  match r with
  | RomanNumeral.I => m.count ≥ 1
  | RomanNumeral.V => m.count ≥ 2
  | _ => false  -- For simplicity, we only consider I and V in this problem

/-- The main theorem to prove -/
theorem matchstick_subtraction_theorem :
  ∀ (initial : MatchstickSet),
    initial.count = 10 →
    ∃ (removed : MatchstickSet) (remaining : MatchstickSet),
      removed.count = 7 ∧
      remaining.count = initial.count - removed.count ∧
      can_form_roman_numeral remaining RomanNumeral.I ∧
      can_form_roman_numeral remaining RomanNumeral.V :=
sorry

end matchstick_subtraction_theorem_l1956_195616


namespace sector_central_angle_measures_l1956_195651

/-- A circular sector with given perimeter and area -/
structure Sector where
  perimeter : ℝ
  area : ℝ

/-- The possible radian measures of the central angle for a sector with given perimeter and area -/
def centralAngleMeasures (s : Sector) : Set ℝ :=
  {α : ℝ | ∃ r : ℝ, r > 0 ∧ 2 * r + α * r = s.perimeter ∧ 1 / 2 * α * r^2 = s.area}

/-- Theorem: For a sector with perimeter 6 and area 2, the central angle measure is either 1 or 4 radians -/
theorem sector_central_angle_measures :
  centralAngleMeasures ⟨6, 2⟩ = {1, 4} := by
  sorry

end sector_central_angle_measures_l1956_195651


namespace sqrt_difference_equals_five_sixths_l1956_195639

theorem sqrt_difference_equals_five_sixths : 
  Real.sqrt (9 / 4) - Real.sqrt (4 / 9) = 5 / 6 := by
  sorry

end sqrt_difference_equals_five_sixths_l1956_195639


namespace binomial_coefficient_7_4_l1956_195620

theorem binomial_coefficient_7_4 : Nat.choose 7 4 = 35 := by sorry

end binomial_coefficient_7_4_l1956_195620


namespace paulines_potato_count_l1956_195695

/-- Represents Pauline's vegetable garden --/
structure Garden where
  rows : Nat
  spacesPerRow : Nat
  tomatoKinds : Nat
  tomatoesPerKind : Nat
  cucumberKinds : Nat
  cucumbersPerKind : Nat
  availableSpaces : Nat

/-- Calculates the number of potatoes in the garden --/
def potatoCount (g : Garden) : Nat :=
  g.rows * g.spacesPerRow - 
  (g.tomatoKinds * g.tomatoesPerKind + g.cucumberKinds * g.cucumbersPerKind) - 
  g.availableSpaces

/-- Theorem stating the number of potatoes in Pauline's garden --/
theorem paulines_potato_count :
  let g : Garden := {
    rows := 10,
    spacesPerRow := 15,
    tomatoKinds := 3,
    tomatoesPerKind := 5,
    cucumberKinds := 5,
    cucumbersPerKind := 4,
    availableSpaces := 85
  }
  potatoCount g = 30 := by
  sorry

end paulines_potato_count_l1956_195695


namespace one_twelfth_day_in_minutes_l1956_195667

/-- Proves that 1/12 of a day is equal to 120 minutes -/
theorem one_twelfth_day_in_minutes : 
  (∀ (hours_per_day minutes_per_hour : ℕ), 
    hours_per_day = 24 → 
    minutes_per_hour = 60 → 
    (1 / 12 : ℚ) * (hours_per_day * minutes_per_hour) = 120) := by
  sorry

#check one_twelfth_day_in_minutes

end one_twelfth_day_in_minutes_l1956_195667


namespace trigonometric_expression_equality_trigonometric_identities_l1956_195694

-- Part 1
theorem trigonometric_expression_equality :
  (Real.sqrt 3 * Real.sin (-20/3 * π)) / Real.tan (11/3 * π) - 
  Real.cos (13/4 * π) * Real.tan (-37/4 * π) = 
  (Real.sqrt 3 - Real.sqrt 2) / 2 := by sorry

-- Part 2
theorem trigonometric_identities (a : Real) (h : Real.tan a = 4/3) :
  (Real.sin a ^ 2 + 2 * Real.sin a * Real.cos a) / (2 * Real.cos a ^ 2 - Real.sin a ^ 2) = 20 ∧
  Real.sin a * Real.cos a = 12/25 := by sorry

end trigonometric_expression_equality_trigonometric_identities_l1956_195694


namespace digit_subtraction_reaches_zero_l1956_195682

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The sequence obtained by repeatedly subtracting the sum of digits -/
def digitSubtractionSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => digitSubtractionSequence n k - sumOfDigits (digitSubtractionSequence n k)

/-- The theorem stating that the digit subtraction sequence always reaches 0 -/
theorem digit_subtraction_reaches_zero (n : ℕ) :
  ∃ k : ℕ, digitSubtractionSequence n k = 0 :=
sorry

end digit_subtraction_reaches_zero_l1956_195682


namespace prop_p_and_q_false_iff_a_gt_1_l1956_195636

-- Define the propositions p and q
def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → a^x > a^y

def q (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, Real.log (a*x^2 - x + a) = y

-- State the theorem
theorem prop_p_and_q_false_iff_a_gt_1 :
  ∀ a : ℝ, (¬(p a ∧ q a)) ↔ a > 1 := by sorry

end prop_p_and_q_false_iff_a_gt_1_l1956_195636


namespace part_one_part_two_l1956_195619

-- Part I
theorem part_one (a : ℝ) (h : ∀ x, x ∈ Set.Icc (-6 : ℝ) 2 ↔ |a * x - 1| ≤ 2) : 
  a = -1/2 := by sorry

-- Part II
theorem part_two (m : ℝ) (h : ∃ x : ℝ, |4 * x + 1| - |2 * x - 3| ≤ 7 - 3 * m) : 
  m ∈ Set.Iic (7/2 : ℝ) := by sorry

end part_one_part_two_l1956_195619


namespace daleyza_project_units_l1956_195688

/-- Calculates the total number of units in a three-building construction project -/
def total_units (first_building : ℕ) : ℕ :=
  let second_building := (2 : ℕ) * first_building / 5
  let third_building := (6 : ℕ) * second_building / 5
  first_building + second_building + third_building

/-- Theorem stating that given the specific conditions of Daleyza's project, 
    the total number of units is 7520 -/
theorem daleyza_project_units : total_units 4000 = 7520 := by
  sorry

end daleyza_project_units_l1956_195688


namespace book_division_proof_l1956_195625

def number_of_divisions (total : ℕ) (target : ℕ) : ℕ :=
  if total ≤ target then 0
  else 1 + number_of_divisions (total / 2) target

theorem book_division_proof :
  number_of_divisions 400 25 = 4 :=
by sorry

end book_division_proof_l1956_195625
