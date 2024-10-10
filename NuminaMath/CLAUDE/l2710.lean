import Mathlib

namespace charles_initial_bananas_l2710_271028

/-- The initial number of bananas Willie has -/
def willie_initial : ℕ := 48

/-- The final number of bananas Willie has -/
def willie_final : ℕ := 13

/-- The number of bananas Charles loses -/
def charles_loss : ℕ := 35

/-- The initial number of bananas Charles has -/
def charles_initial : ℕ := charles_loss

theorem charles_initial_bananas :
  charles_initial = 35 :=
sorry

end charles_initial_bananas_l2710_271028


namespace wright_brothers_first_flight_l2710_271096

/-- Represents the different groups of brothers mentioned in the problem -/
inductive Brothers
  | Bell
  | Hale
  | Wright
  | Leon

/-- Represents an aircraft -/
structure Aircraft where
  name : String

/-- Represents a flight achievement -/
structure FlightAchievement where
  date : String
  aircraft : Aircraft
  achievers : Brothers

/-- The first powered human flight -/
def first_powered_flight : FlightAchievement :=
  { date := "December 1903"
  , aircraft := { name := "Flyer 1" }
  , achievers := Brothers.Wright }

/-- Theorem stating that the Wright Brothers achieved the first powered human flight -/
theorem wright_brothers_first_flight :
  first_powered_flight.achievers = Brothers.Wright :=
by sorry

end wright_brothers_first_flight_l2710_271096


namespace lizard_spot_wrinkle_ratio_l2710_271039

/-- Represents a three-eyed lizard with wrinkles and spots. -/
structure Lizard where
  eyes : ℕ
  wrinkles : ℕ
  spots : ℕ

/-- The properties of our specific lizard. -/
def specialLizard : Lizard where
  eyes := 3
  wrinkles := 3 * 3
  spots := 3 + (3 * 3) - 69 + 3

theorem lizard_spot_wrinkle_ratio (l : Lizard) 
  (h1 : l.eyes = 3)
  (h2 : l.wrinkles = 3 * l.eyes)
  (h3 : l.eyes = l.spots + l.wrinkles - 69) :
  l.spots / l.wrinkles = 7 := by
  sorry

#eval specialLizard.spots / specialLizard.wrinkles

end lizard_spot_wrinkle_ratio_l2710_271039


namespace race_equation_theorem_l2710_271025

/-- Represents a runner's performance in a race before and after training -/
structure RunnerPerformance where
  distance : ℝ
  speedIncrease : ℝ
  timeImprovement : ℝ
  initialSpeed : ℝ

/-- Checks if the given runner performance satisfies the race equation -/
def satisfiesRaceEquation (perf : RunnerPerformance) : Prop :=
  perf.distance / perf.initialSpeed - 
  perf.distance / (perf.initialSpeed * (1 + perf.speedIncrease)) = 
  perf.timeImprovement

/-- Theorem stating that a runner with the given performance satisfies the race equation -/
theorem race_equation_theorem (perf : RunnerPerformance) 
  (h1 : perf.distance = 3000)
  (h2 : perf.speedIncrease = 0.25)
  (h3 : perf.timeImprovement = 3) :
  satisfiesRaceEquation perf := by
  sorry

end race_equation_theorem_l2710_271025


namespace physics_marks_l2710_271019

theorem physics_marks (P C M : ℝ) 
  (avg_total : (P + C + M) / 3 = 55)
  (avg_pm : (P + M) / 2 = 90)
  (avg_pc : (P + C) / 2 = 70) :
  P = 155 := by
sorry

end physics_marks_l2710_271019


namespace three_digit_number_appended_l2710_271008

theorem three_digit_number_appended (n : ℕ) : 
  100 ≤ n ∧ n < 1000 → 1000 * n + n = 1001 * n := by
  sorry

end three_digit_number_appended_l2710_271008


namespace joey_age_l2710_271061

theorem joey_age : 
  let ages : List ℕ := [3, 5, 7, 9, 11, 13]
  let movie_pair : ℕ × ℕ := (3, 13)
  let baseball_pair : ℕ × ℕ := (7, 9)
  let stay_home : ℕ × ℕ := (5, 11)
  (∀ (a b : ℕ), a ∈ ages ∧ b ∈ ages ∧ a + b = 16 → (a, b) = movie_pair) ∧
  (∀ (a b : ℕ), a ∈ ages ∧ b ∈ ages ∧ a < 10 ∧ b < 10 ∧ (a, b) ≠ movie_pair → (a, b) = baseball_pair) ∧
  (∀ (a : ℕ), a ∈ ages ∧ a ∉ [movie_pair.1, movie_pair.2, baseball_pair.1, baseball_pair.2, 5] → a = 11) →
  stay_home.2 = 11 :=
by sorry

end joey_age_l2710_271061


namespace workers_in_second_group_l2710_271022

theorem workers_in_second_group 
  (wages_group1 : ℕ) 
  (workers_group1 : ℕ) 
  (days_group1 : ℕ) 
  (wages_group2 : ℕ) 
  (days_group2 : ℕ) 
  (h1 : wages_group1 = 9450) 
  (h2 : workers_group1 = 15) 
  (h3 : days_group1 = 6) 
  (h4 : wages_group2 = 9975) 
  (h5 : days_group2 = 5) : 
  (wages_group2 / (wages_group1 / (workers_group1 * days_group1) * days_group2)) = 19 :=
by
  sorry

end workers_in_second_group_l2710_271022


namespace isosceles_triangle_third_side_l2710_271049

/-- An isosceles triangle with side lengths a, b, and c, where a = b -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : a = b
  isPositive : 0 < a ∧ 0 < b ∧ 0 < c
  triangleInequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: In an isosceles triangle with two sides of lengths 13 and 6, the third side is 13 -/
theorem isosceles_triangle_third_side 
  (t : IsoscelesTriangle) 
  (h1 : t.a = 13 ∨ t.b = 13 ∨ t.c = 13) 
  (h2 : t.a = 6 ∨ t.b = 6 ∨ t.c = 6) : 
  t.c = 13 := by
  sorry

#check isosceles_triangle_third_side

end isosceles_triangle_third_side_l2710_271049


namespace gasoline_tank_capacity_gasoline_tank_capacity_proof_l2710_271047

theorem gasoline_tank_capacity : ℝ → Prop :=
  fun capacity =>
    let initial_fraction : ℝ := 5/6
    let final_fraction : ℝ := 1/3
    let used_amount : ℝ := 15
    initial_fraction * capacity - final_fraction * capacity = used_amount →
    capacity = 30

-- The proof goes here
theorem gasoline_tank_capacity_proof : gasoline_tank_capacity 30 := by
  sorry

end gasoline_tank_capacity_gasoline_tank_capacity_proof_l2710_271047


namespace two_solutions_exist_l2710_271054

/-- A structure representing a triangle with integer side lengths -/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  triangle_inequality : a.val < b.val + c.val ∧ b.val < a.val + c.val ∧ c.val < a.val + b.val

/-- The condition from the original problem -/
def satisfies_equation (t : IntegerTriangle) : Prop :=
  (t.a.val * t.b.val * t.c.val : ℕ) = 2 * (t.a.val - 1) * (t.b.val - 1) * (t.c.val - 1)

/-- The main theorem stating that there are exactly two solutions -/
theorem two_solutions_exist : 
  (∃ (t1 t2 : IntegerTriangle), 
    satisfies_equation t1 ∧ 
    satisfies_equation t2 ∧ 
    t1 ≠ t2 ∧ 
    (∀ (t : IntegerTriangle), satisfies_equation t → (t = t1 ∨ t = t2))) ∧
  (∃ (t1 : IntegerTriangle), t1.a = 8 ∧ t1.b = 7 ∧ t1.c = 3 ∧ satisfies_equation t1) ∧
  (∃ (t2 : IntegerTriangle), t2.a = 6 ∧ t2.b = 5 ∧ t2.c = 4 ∧ satisfies_equation t2) :=
by sorry


end two_solutions_exist_l2710_271054


namespace expression_simplification_l2710_271087

theorem expression_simplification (x : ℤ) 
  (h1 : -1 ≤ x) (h2 : x < 2) (h3 : x ≠ 1) : 
  ((x + 1) / (x^2 - 1) + x / (x - 1)) / ((x + 1) / (x^2 - 2*x + 1)) = x - 1 :=
by sorry

end expression_simplification_l2710_271087


namespace wire_cutting_l2710_271006

/-- Given a wire of length 50 feet cut into three pieces, prove the lengths of the pieces. -/
theorem wire_cutting (x : ℝ) 
  (h1 : x + (x + 2) + (2*x - 3) = 50) -- Total length equation
  (h2 : x > 0) -- Ensure positive length
  : x = 12.75 ∧ x + 2 = 14.75 ∧ 2*x - 3 = 22.5 := by
  sorry

end wire_cutting_l2710_271006


namespace vector_magnitude_l2710_271081

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def b : Fin 2 → ℝ := ![2, 1]
def c (x : ℝ) : Fin 2 → ℝ := ![3, x]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ i, v i = k * w i

-- State the theorem
theorem vector_magnitude (x : ℝ) :
  parallel (a x) b →
  ‖b + c x‖ = 5 * Real.sqrt 2 := by
  sorry

end vector_magnitude_l2710_271081


namespace apprentice_work_time_l2710_271042

/-- Proves that given the master's and apprentice's production rates, 
    the apprentice needs 4 hours to match the master's 3-hour output. -/
theorem apprentice_work_time 
  (master_rate : ℕ) 
  (apprentice_rate : ℕ) 
  (master_time : ℕ) 
  (h1 : master_rate = 64)
  (h2 : apprentice_rate = 48)
  (h3 : master_time = 3) :
  (master_rate * master_time) / apprentice_rate = 4 := by
  sorry

#check apprentice_work_time

end apprentice_work_time_l2710_271042


namespace unique_digit_for_divisibility_by_nine_l2710_271010

def sum_of_digits (n : ℕ) : ℕ := 8 + 6 + 5 + n + 7 + 4 + 3 + 2

theorem unique_digit_for_divisibility_by_nine :
  ∃! n : ℕ, n ≤ 9 ∧ (sum_of_digits n) % 9 = 0 ∧ n = 1 := by
sorry

end unique_digit_for_divisibility_by_nine_l2710_271010


namespace square_divisibility_l2710_271040

theorem square_divisibility (n : ℕ+) (h : ∀ q : ℕ+, q ∣ n → q ≤ 12) :
  144 ∣ n^2 := by
sorry

end square_divisibility_l2710_271040


namespace sum_of_roots_quadratic_l2710_271034

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x + y = 1 := by
  sorry

end sum_of_roots_quadratic_l2710_271034


namespace unique_partition_l2710_271063

/-- Represents the number of caps collected by each girl -/
def caps : List Nat := [20, 29, 31, 49, 51]

/-- Represents a partition of the caps into two boxes -/
structure Partition where
  red : List Nat
  blue : List Nat
  sum_red : red.sum = 60
  sum_blue : blue.sum = 120
  partition_complete : red ++ blue = caps

/-- The theorem to be proved -/
theorem unique_partition : ∃! p : Partition, True := by sorry

end unique_partition_l2710_271063


namespace no_zonk_probability_l2710_271004

theorem no_zonk_probability : 
  let num_tables : ℕ := 3
  let boxes_per_table : ℕ := 3
  let prob_no_zonk_per_table : ℚ := 2 / 3
  (prob_no_zonk_per_table ^ num_tables : ℚ) = 8 / 27 := by
sorry

end no_zonk_probability_l2710_271004


namespace rectangular_prism_space_diagonal_l2710_271023

/-- A rectangular prism with given surface area and edge length sum has a space diagonal of length 5 -/
theorem rectangular_prism_space_diagonal : 
  ∀ (x y z : ℝ), 
  (2 * x * y + 2 * y * z + 2 * x * z = 11) →
  (4 * (x + y + z) = 24) →
  Real.sqrt (x^2 + y^2 + z^2) = 5 := by
sorry


end rectangular_prism_space_diagonal_l2710_271023


namespace max_value_fraction_l2710_271076

theorem max_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 9 * a^2 + b^2 = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x * y) / (3 * x + y) ≤ (a * b) / (3 * a + b)) →
  (a * b) / (3 * a + b) = Real.sqrt 2 / 12 :=
sorry

end max_value_fraction_l2710_271076


namespace unique_prime_power_sum_l2710_271011

theorem unique_prime_power_sum (p q : ℕ) : 
  Prime p → Prime q → Prime (p^q + q^p) → (p = 2 ∧ q = 3) := by
sorry

end unique_prime_power_sum_l2710_271011


namespace T_is_far_right_l2710_271053

/-- Represents a rectangle with four integer-labeled sides --/
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

/-- Checks if a rectangle is at the far-right end of the row --/
def is_far_right (r : Rectangle) (others : List Rectangle) : Prop :=
  ∀ other ∈ others, r.y ≥ other.y ∧ (r.y = other.y → r.w ≥ other.w)

/-- The given rectangles --/
def P : Rectangle := ⟨3, 0, 9, 5⟩
def Q : Rectangle := ⟨6, 1, 0, 8⟩
def R : Rectangle := ⟨0, 3, 2, 7⟩
def S : Rectangle := ⟨8, 5, 4, 1⟩
def T : Rectangle := ⟨5, 2, 6, 9⟩

theorem T_is_far_right :
  is_far_right T [P, Q, R, S] :=
sorry

end T_is_far_right_l2710_271053


namespace people_counting_ratio_l2710_271074

theorem people_counting_ratio :
  ∀ (day1 day2 : ℕ),
  day2 = 500 →
  day1 + day2 = 1500 →
  ∃ (k : ℕ), day1 = k * day2 →
  day1 / day2 = 2 := by
sorry

end people_counting_ratio_l2710_271074


namespace largest_integer_inequality_l2710_271069

theorem largest_integer_inequality : ∀ x : ℤ, x ≤ 4 ↔ (x : ℚ) / 4 - 3 / 7 < 2 / 3 := by
  sorry

end largest_integer_inequality_l2710_271069


namespace tile_arrangement_count_l2710_271016

/-- The number of distinguishable arrangements of tiles -/
def tile_arrangements (brown green yellow : ℕ) (purple : ℕ) : ℕ :=
  Nat.factorial (brown + green + yellow + purple) /
  (Nat.factorial brown * Nat.factorial green * Nat.factorial yellow * Nat.factorial purple)

/-- Theorem stating that the number of distinguishable arrangements
    of 2 brown, 3 green, 2 yellow, and 1 purple tile is 1680 -/
theorem tile_arrangement_count :
  tile_arrangements 2 3 2 1 = 1680 := by
  sorry

end tile_arrangement_count_l2710_271016


namespace portrait_ratio_l2710_271086

/-- Prove that the ratio of students who had portraits taken before lunch
    to the total number of students is 1:3 -/
theorem portrait_ratio :
  ∀ (before_lunch after_lunch not_taken : ℕ),
  before_lunch + after_lunch + not_taken = 24 →
  after_lunch = 10 →
  not_taken = 6 →
  (before_lunch : ℚ) / 24 = 1 / 3 := by
sorry

end portrait_ratio_l2710_271086


namespace min_value_theorem_l2710_271036

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * x + 1 / x^2 ≥ 4 ∧ 
  (3 * x + 1 / x^2 = 4 ↔ x = 1) := by
  sorry

end min_value_theorem_l2710_271036


namespace mean_of_remaining_numbers_l2710_271038

def numbers : List ℝ := [1030, 1560, 1980, 2025, 2140, 2250, 2450, 2600, 2780, 2910]

theorem mean_of_remaining_numbers :
  let total_sum := numbers.sum
  let seven_mean := 2300
  let seven_sum := 7 * seven_mean
  let remaining_sum := total_sum - seven_sum
  (remaining_sum / 3 : ℝ) = 2108.33 := by
sorry

end mean_of_remaining_numbers_l2710_271038


namespace tan_sum_alpha_beta_l2710_271030

theorem tan_sum_alpha_beta (α β : Real) (h : 2 * Real.tan α = 3 * Real.tan β) :
  Real.tan (α + β) = (5 * Real.sin (2 * β)) / (5 * Real.cos (2 * β) - 1) := by
  sorry

end tan_sum_alpha_beta_l2710_271030


namespace inequalities_hold_l2710_271046

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((a + b) * (1 / a + 1 / b) ≥ 4) ∧
  (a^2 + b^2 + 2 ≥ 2*a + 2*b) ∧
  (Real.sqrt (abs (a - b)) ≥ Real.sqrt a - Real.sqrt b) :=
by sorry

end inequalities_hold_l2710_271046


namespace hotdog_count_l2710_271017

theorem hotdog_count (initial : ℕ) (sold : ℕ) (remaining : ℕ) : 
  sold = 2 → remaining = 97 → initial = remaining + sold :=
by sorry

end hotdog_count_l2710_271017


namespace line_tangent_to_fixed_circle_l2710_271083

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a line -/
structure Line :=
  (p1 : Point)
  (p2 : Point)

/-- Function to check if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Function to get the circumcircle of a triangle -/
def circumcircle (t : Triangle) : Circle := sorry

/-- Function to check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Function to check if a point is in a half-plane relative to a line -/
def isInHalfPlane (p : Point) (l : Line) : Prop := sorry

/-- Function to get the perpendicular bisector of a line segment -/
def perpendicularBisector (p1 : Point) (p2 : Point) : Line := sorry

/-- Function to get the intersection of two lines -/
def lineIntersection (l1 : Line) (l2 : Line) : Point := sorry

/-- Function to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Main theorem -/
theorem line_tangent_to_fixed_circle 
  (A B C : Point) 
  (h1 : isAcuteAngled (Triangle.mk A B C))
  (h2 : ∀ C', isOnCircle C' (circumcircle (Triangle.mk A B C)) → 
              isInHalfPlane C' (Line.mk A B) → 
              ∃ M N : Point,
                M = lineIntersection (perpendicularBisector B C') (Line.mk A C') ∧
                N = lineIntersection (perpendicularBisector A C') (Line.mk B C') ∧
                ∃ fixedCircle : Circle, isTangent (Line.mk M N) fixedCircle) :
  ∃ fixedCircle : Circle, ∀ C' M N : Point,
    isOnCircle C' (circumcircle (Triangle.mk A B C)) →
    isInHalfPlane C' (Line.mk A B) →
    M = lineIntersection (perpendicularBisector B C') (Line.mk A C') →
    N = lineIntersection (perpendicularBisector A C') (Line.mk B C') →
    isTangent (Line.mk M N) fixedCircle :=
by
  sorry

end line_tangent_to_fixed_circle_l2710_271083


namespace job_completion_time_equivalence_l2710_271064

/-- Represents the number of days required to complete a job -/
def days_to_complete (num_men : ℕ) (man_days : ℕ) : ℚ :=
  man_days / num_men

theorem job_completion_time_equivalence :
  let initial_men : ℕ := 30
  let initial_days : ℕ := 8
  let new_men : ℕ := 40
  let man_days : ℕ := initial_men * initial_days
  days_to_complete new_men man_days = 6 := by
  sorry

end job_completion_time_equivalence_l2710_271064


namespace four_digit_sum_l2710_271078

theorem four_digit_sum (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  (6 * (a + b + c + d) * 1111 = 73326) →
  ({a, b, c, d} : Set ℕ) = {1, 2, 3, 5} :=
by sorry

end four_digit_sum_l2710_271078


namespace smallest_factorization_coefficient_l2710_271095

theorem smallest_factorization_coefficient : 
  ∃ (c : ℕ), c > 0 ∧ 
  (∃ (r s : ℤ), x^2 + c*x + 2016 = (x + r) * (x + s)) ∧ 
  (∀ (c' : ℕ), 0 < c' ∧ c' < c → 
    ¬∃ (r' s' : ℤ), x^2 + c'*x + 2016 = (x + r') * (x + s')) ∧
  c = 108 := by
sorry

end smallest_factorization_coefficient_l2710_271095


namespace minimum_buses_needed_l2710_271050

def bus_capacity : ℕ := 48
def total_passengers : ℕ := 1230

def buses_needed (capacity : ℕ) (passengers : ℕ) : ℕ :=
  (passengers + capacity - 1) / capacity

theorem minimum_buses_needed : 
  buses_needed bus_capacity total_passengers = 26 := by
  sorry

end minimum_buses_needed_l2710_271050


namespace equation_solutions_l2710_271029

def is_solution (a b c : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (1 : ℚ) / a + 1 / b + 1 / c = 1

def solution_set : Set (ℤ × ℤ × ℤ) :=
  {(3, 3, 3), (2, 3, 6), (2, 4, 4)} ∪ {(1, t, -t) | t : ℤ}

theorem equation_solutions :
  ∀ (a b c : ℤ), is_solution a b c ↔ (a, b, c) ∈ solution_set :=
sorry

end equation_solutions_l2710_271029


namespace simplify_expression_l2710_271048

theorem simplify_expression : 
  Real.sqrt 15 + Real.sqrt 45 - (Real.sqrt (4/3) - Real.sqrt 108) = 
  Real.sqrt 15 + 3 * Real.sqrt 5 + (16 * Real.sqrt 3) / 3 := by
  sorry

end simplify_expression_l2710_271048


namespace books_per_bookshelf_l2710_271092

theorem books_per_bookshelf 
  (total_books : ℕ) 
  (num_bookshelves : ℕ) 
  (h1 : total_books = 38) 
  (h2 : num_bookshelves = 19) 
  (h3 : num_bookshelves > 0) :
  total_books / num_bookshelves = 2 := by
  sorry

end books_per_bookshelf_l2710_271092


namespace arcsin_symmetry_l2710_271020

theorem arcsin_symmetry (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  Real.arcsin (-x) = -Real.arcsin x := by
  sorry

end arcsin_symmetry_l2710_271020


namespace tuesday_necklaces_l2710_271026

/-- The number of beaded necklaces Kylie made on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of beaded bracelets Kylie made on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of beaded earrings Kylie made on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed to make one beaded necklace -/
def beads_per_necklace : ℕ := 20

/-- The number of beads needed to make one beaded bracelet -/
def beads_per_bracelet : ℕ := 10

/-- The number of beads needed to make one beaded earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads Kylie used to make her jewelry -/
def total_beads : ℕ := 325

/-- Theorem: The number of beaded necklaces Kylie made on Tuesday is 2 -/
theorem tuesday_necklaces : 
  (total_beads - (monday_necklaces * beads_per_necklace + 
    wednesday_bracelets * beads_per_bracelet + 
    wednesday_earrings * beads_per_earring)) / beads_per_necklace = 2 := by
  sorry

end tuesday_necklaces_l2710_271026


namespace pattern_proof_l2710_271007

theorem pattern_proof (n : ℕ) : (2*n - 1) * (2*n + 1) = (2*n)^2 - 1 := by
  sorry

end pattern_proof_l2710_271007


namespace second_box_clay_capacity_l2710_271072

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.height * d.width * d.length

/-- The dimensions of the first box -/
def firstBox : BoxDimensions := {
  height := 3,
  width := 4,
  length := 7
}

/-- The dimensions of the second box -/
def secondBox : BoxDimensions := {
  height := 3 * firstBox.height,
  width := 2 * firstBox.width,
  length := firstBox.length
}

/-- The amount of clay the first box can hold in grams -/
def firstBoxClay : ℝ := 70

/-- Theorem: The second box can hold 420 grams of clay -/
theorem second_box_clay_capacity : 
  (boxVolume secondBox / boxVolume firstBox) * firstBoxClay = 420 := by sorry

end second_box_clay_capacity_l2710_271072


namespace perpendicular_lines_line_slope_l2710_271055

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
theorem perpendicular_lines (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) (hB₁ : B₁ ≠ 0) (hB₂ : B₂ ≠ 0) :
  (A₁ * x + B₁ * y + C₁ = 0 ∧ A₂ * x + B₂ * y + C₂ = 0) →
  ((-A₁ / B₁) * (-A₂ / B₂) = -1 ↔ (A₁ * A₂ + B₁ * B₂ = 0)) :=
by sorry

/-- The slope of a line Ax + By + C = 0 is -A/B -/
theorem line_slope (A B C : ℝ) (hB : B ≠ 0) :
  (A * x + B * y + C = 0) → (y = (-A / B) * x - C / B) :=
by sorry

end perpendicular_lines_line_slope_l2710_271055


namespace worker_daily_rate_l2710_271002

/-- Proves that a worker's daily rate is $150 given the specified conditions -/
theorem worker_daily_rate (daily_rate : ℝ) (overtime_rate : ℝ) (total_days : ℝ) 
  (overtime_hours : ℝ) (total_pay : ℝ) : 
  overtime_rate = 5 →
  total_days = 5 →
  overtime_hours = 4 →
  total_pay = 770 →
  total_pay = daily_rate * total_days + overtime_rate * overtime_hours →
  daily_rate = 150 := by
  sorry

end worker_daily_rate_l2710_271002


namespace anna_initial_ham_slices_l2710_271014

/-- The number of slices of ham Anna puts in each sandwich. -/
def slices_per_sandwich : ℕ := 3

/-- The number of sandwiches Anna wants to make. -/
def total_sandwiches : ℕ := 50

/-- The additional number of ham slices Anna needs. -/
def additional_slices : ℕ := 119

/-- The initial number of ham slices Anna has. -/
def initial_slices : ℕ := total_sandwiches * slices_per_sandwich - additional_slices

theorem anna_initial_ham_slices :
  initial_slices = 31 := by sorry

end anna_initial_ham_slices_l2710_271014


namespace integer_root_values_l2710_271082

def polynomial (x b : ℤ) : ℤ := x^3 + 6*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-217, -74, -43, -31, -22, -19, 19, 22, 31, 43, 74, 217} :=
by sorry

end integer_root_values_l2710_271082


namespace sqrt_equation_solution_l2710_271005

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 9 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 9 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 45 := by
sorry

end sqrt_equation_solution_l2710_271005


namespace triangle_properties_l2710_271071

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.b * Real.sin t.A = 3 * t.c * Real.sin t.B ∧
  t.a = 3 ∧
  Real.cos t.B = 2/3

theorem triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.b = Real.sqrt 6 ∧ 
  (1/2 : ℝ) * t.a * t.c * Real.sin t.B = Real.sqrt 5 / 2 := by
  sorry

end triangle_properties_l2710_271071


namespace solution_range_solution_range_converse_l2710_271093

/-- The system of equations has two distinct solutions -/
def has_two_distinct_solutions (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
  y₁ = Real.sqrt (-x₁^2 - 2*x₁) ∧ x₁ + y₁ - m = 0 ∧
  y₂ = Real.sqrt (-x₂^2 - 2*x₂) ∧ x₂ + y₂ - m = 0

/-- The main theorem -/
theorem solution_range (m : ℝ) : 
  has_two_distinct_solutions m → m ∈ Set.Icc 0 (-1 + Real.sqrt 2) :=
by
  sorry

/-- The converse of the main theorem -/
theorem solution_range_converse (m : ℝ) : 
  m ∈ Set.Ioo 0 (-1 + Real.sqrt 2) → has_two_distinct_solutions m :=
by
  sorry

end solution_range_solution_range_converse_l2710_271093


namespace c_decreases_as_r_increases_l2710_271085

theorem c_decreases_as_r_increases (e n r : ℝ) (h_e : e > 0) (h_n : n > 0) (h_r : r > 0) :
  ∀ (R₁ R₂ : ℝ), R₁ > 0 → R₂ > 0 → R₂ > R₁ →
  (e * n) / (R₁ + n * r) > (e * n) / (R₂ + n * r) := by
sorry

end c_decreases_as_r_increases_l2710_271085


namespace missing_number_proof_l2710_271000

theorem missing_number_proof (x : ℤ) : (4 + 3) + (x - 3 - 1) = 11 → x = 8 := by
  sorry

end missing_number_proof_l2710_271000


namespace parallel_vectors_expression_l2710_271084

noncomputable def θ : ℝ := Real.arctan (3 : ℝ)

theorem parallel_vectors_expression (a b : ℝ × ℝ) :
  a = (3, 1) →
  b = (Real.sin θ, Real.cos θ) →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 11 / 5 := by
  sorry

end parallel_vectors_expression_l2710_271084


namespace train_length_l2710_271067

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 9 → speed_kmh * (1000 / 3600) * time_s = 225 := by
  sorry

end train_length_l2710_271067


namespace school_students_count_l2710_271051

/-- Proves that given the specified conditions, the total number of students in the school is 387 -/
theorem school_students_count : ∃ (boys girls : ℕ), 
  boys ≥ 150 ∧ 
  boys % 6 = 0 ∧ 
  girls = boys + boys / 20 * 3 ∧ 
  boys + girls ≤ 400 ∧
  boys + girls = 387 := by
  sorry

end school_students_count_l2710_271051


namespace imaginary_part_sum_of_complex_fractions_l2710_271091

theorem imaginary_part_sum_of_complex_fractions :
  Complex.im (1 / Complex.ofReal (-2) + Complex.I + 1 / (Complex.ofReal 1 - 2 * Complex.I)) = 1/5 := by
  sorry

end imaginary_part_sum_of_complex_fractions_l2710_271091


namespace triangle_angle_sum_l2710_271052

theorem triangle_angle_sum (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- Ensures angles are positive
  A + B + C = 180 →        -- Sum of angles in a triangle is 180°
  A = 90 →                 -- Given: Angle A is 90°
  B = 50 →                 -- Given: Angle B is 50°
  C = 40 :=                -- To prove: Angle C is 40°
by sorry

end triangle_angle_sum_l2710_271052


namespace group_average_age_l2710_271021

theorem group_average_age 
  (num_women : ℕ) 
  (num_men : ℕ) 
  (avg_age_women : ℚ) 
  (avg_age_men : ℚ) 
  (h1 : num_women = 12) 
  (h2 : num_men = 18) 
  (h3 : avg_age_women = 28) 
  (h4 : avg_age_men = 40) : 
  (num_women * avg_age_women + num_men * avg_age_men) / (num_women + num_men : ℚ) = 352 / 10 := by
  sorry

end group_average_age_l2710_271021


namespace oriented_knight_moves_l2710_271035

/-- An Oriented Knight's move on a chess board -/
inductive OrientedKnightMove
| right_up : OrientedKnightMove  -- Two squares right, one square up
| up_right : OrientedKnightMove  -- Two squares up, one square right

/-- A sequence of Oriented Knight moves -/
def MoveSequence := List OrientedKnightMove

/-- The size of the chess board -/
def boardSize : ℕ := 16

/-- Checks if a sequence of moves is valid (reaches the top-right corner) -/
def isValidSequence (moves : MoveSequence) : Prop :=
  let finalPosition := moves.foldl
    (fun pos move => match move with
      | OrientedKnightMove.right_up => (pos.1 + 2, pos.2 + 1)
      | OrientedKnightMove.up_right => (pos.1 + 1, pos.2 + 2))
    (0, 0)
  finalPosition = (boardSize - 1, boardSize - 1)

/-- The number of valid move sequences for an Oriented Knight -/
def validSequenceCount : ℕ := 252

theorem oriented_knight_moves :
  (validSequences : Finset MoveSequence).card = validSequenceCount :=
by
  sorry

end oriented_knight_moves_l2710_271035


namespace karen_has_32_quarters_l2710_271018

/-- Calculates the number of quarters Karen has given the conditions of the problem -/
def karens_quarters (christopher_quarters : ℕ) (dollar_difference : ℕ) : ℕ :=
  let christopher_value := christopher_quarters * 25  -- Value in cents
  let karen_value := christopher_value - dollar_difference * 100  -- Value in cents
  karen_value / 25  -- Convert back to quarters

/-- Proves that Karen has 32 quarters given the problem conditions -/
theorem karen_has_32_quarters :
  karens_quarters 64 8 = 32 := by sorry

end karen_has_32_quarters_l2710_271018


namespace area_between_concentric_circles_l2710_271070

theorem area_between_concentric_circles
  (r_small : ℝ) (r_large : ℝ) (h1 : r_small * 2 = 6)
  (h2 : r_large = 3 * r_small) :
  π * r_large^2 - π * r_small^2 = 72 * π :=
by sorry

end area_between_concentric_circles_l2710_271070


namespace room_puzzle_solution_l2710_271058

/-- Represents a person who can either be a truth-teller or a liar -/
inductive Person
| TruthTeller
| Liar

/-- Represents a statement made by a person -/
structure Statement where
  content : Prop
  speaker : Person

/-- The environment of the problem -/
structure Room where
  people : Nat
  liars : Nat
  statements : List Statement

/-- The correct solution to the problem -/
def correct_solution : Room := { people := 4, liars := 2, statements := [] }

/-- Checks if a given solution is consistent with the statements made -/
def is_consistent (room : Room) : Prop :=
  let s1 := Statement.mk (room.people ≤ 3 ∧ room.liars = room.people) Person.Liar
  let s2 := Statement.mk (room.people ≤ 4 ∧ room.liars < room.people) Person.TruthTeller
  let s3 := Statement.mk (room.people = 5 ∧ room.liars = 3) Person.Liar
  room.statements = [s1, s2, s3]

/-- The main theorem to prove -/
theorem room_puzzle_solution :
  ∀ room : Room, is_consistent room → room = correct_solution :=
sorry

end room_puzzle_solution_l2710_271058


namespace all_right_angled_isosceles_similar_isosceles_equal_vertex_angle_similar_l2710_271099

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  vertex_angle : ℝ

-- Define a right-angled isosceles triangle
structure RightAngledIsoscelesTriangle extends IsoscelesTriangle where
  is_right_angled : vertex_angle = 90

-- Define similarity for isosceles triangles
def are_similar (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.vertex_angle = t2.vertex_angle

-- Theorem 1: All isosceles right-angled triangles are similar
theorem all_right_angled_isosceles_similar (t1 t2 : RightAngledIsoscelesTriangle) :
  are_similar t1.toIsoscelesTriangle t2.toIsoscelesTriangle :=
sorry

-- Theorem 2: Two isosceles triangles with equal vertex angles are similar
theorem isosceles_equal_vertex_angle_similar (t1 t2 : IsoscelesTriangle)
  (h : t1.vertex_angle = t2.vertex_angle) :
  are_similar t1 t2 :=
sorry

end all_right_angled_isosceles_similar_isosceles_equal_vertex_angle_similar_l2710_271099


namespace prime_square_diff_divisibility_l2710_271024

theorem prime_square_diff_divisibility (p q : ℕ) (k : ℤ) : 
  Prime p → Prime q → p > 5 → q > 5 → p ≠ q → 
  (p^2 : ℤ) - (q^2 : ℤ) = 6 * k → 
  (p^2 : ℤ) - (q^2 : ℤ) ≡ 0 [ZMOD 24] :=
sorry

end prime_square_diff_divisibility_l2710_271024


namespace three_face_painted_subcubes_count_l2710_271041

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube -/
structure PaintedCube (n : ℕ) extends Cube n where
  painted_faces : ℕ := 6

/-- Counts the number of subcubes with at least three painted faces -/
def count_three_face_painted_subcubes (c : PaintedCube 4) : ℕ :=
  8

/-- Theorem: In a 4x4x4 painted cube, there are exactly 8 subcubes with at least three painted faces -/
theorem three_face_painted_subcubes_count (c : PaintedCube 4) :
  count_three_face_painted_subcubes c = 8 := by
  sorry

end three_face_painted_subcubes_count_l2710_271041


namespace base_k_conversion_l2710_271001

theorem base_k_conversion (k : ℕ) : 
  (0 < k ∧ k < 10) → (k^2 + 7*k + 5 = 125) → k = 8 :=
by sorry

end base_k_conversion_l2710_271001


namespace correct_probability_l2710_271037

/-- The number of options for the first three digits -/
def first_three_options : ℕ := 3

/-- The number of permutations of the last four digits (0, 1, 6, 6) -/
def last_four_permutations : ℕ := 12

/-- The total number of possible phone numbers -/
def total_possible_numbers : ℕ := first_three_options * last_four_permutations

/-- The probability of dialing the correct number -/
def probability_correct : ℚ := 1 / total_possible_numbers

theorem correct_probability : probability_correct = 1 / 36 := by
  sorry

end correct_probability_l2710_271037


namespace algebraic_expression_value_l2710_271080

theorem algebraic_expression_value :
  let x : ℚ := 4
  let y : ℚ := -1/5
  ((x + 2*y)^2 - y*(x + 4*y) - x^2) / (-2*y) = -6 := by sorry

end algebraic_expression_value_l2710_271080


namespace problem_solving_probability_l2710_271044

theorem problem_solving_probability :
  let p_A : ℚ := 1/2
  let p_B : ℚ := 1/3
  let p_C : ℚ := 1/4
  let p_at_least_one : ℚ := 1 - (1 - p_A) * (1 - p_B) * (1 - p_C)
  p_at_least_one = 3/4 :=
by sorry

end problem_solving_probability_l2710_271044


namespace accidents_in_four_minutes_l2710_271043

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The duration of the observation period in minutes -/
def observation_period : ℕ := 4

/-- The interval between car collisions in seconds -/
def car_collision_interval : ℕ := 10

/-- The interval between big crashes in seconds -/
def big_crash_interval : ℕ := 20

/-- The total number of accidents in the observation period -/
def total_accidents : ℕ := 36

theorem accidents_in_four_minutes :
  (observation_period * seconds_per_minute) / car_collision_interval +
  (observation_period * seconds_per_minute) / big_crash_interval =
  total_accidents := by
  sorry

end accidents_in_four_minutes_l2710_271043


namespace largest_lcm_with_18_l2710_271077

theorem largest_lcm_with_18 : 
  (Nat.lcm 18 4).max 
    ((Nat.lcm 18 6).max 
      ((Nat.lcm 18 9).max 
        ((Nat.lcm 18 14).max 
          (Nat.lcm 18 18)))) = 126 := by
  sorry

end largest_lcm_with_18_l2710_271077


namespace square_area_from_diagonal_l2710_271075

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 144 := by sorry

end square_area_from_diagonal_l2710_271075


namespace proposition_is_false_l2710_271013

theorem proposition_is_false : ∃ (angle1 angle2 : ℝ),
  angle1 + angle2 = 90 ∧ angle1 = angle2 :=
by sorry

end proposition_is_false_l2710_271013


namespace first_player_can_force_odd_result_l2710_271059

/-- A game where two players insert operations between numbers 1 to 100 --/
def NumberGame : Type := List (Fin 100 → ℕ) → Prop

/-- The set of possible operations in the game --/
inductive Operation
| Add
| Subtract
| Multiply

/-- A strategy for a player in the game --/
def Strategy : Type := List Operation → Operation

/-- The result of applying operations to a list of numbers --/
def applyOperations (nums : List ℕ) (ops : List Operation) : ℕ := sorry

/-- A winning strategy ensures an odd result --/
def winningStrategy (s : Strategy) : Prop :=
  ∀ (opponent : Strategy), 
    ∃ (finalOps : List Operation), 
      Odd (applyOperations (List.range 100) finalOps)

/-- Theorem: There exists a winning strategy for the first player --/
theorem first_player_can_force_odd_result :
  ∃ (s : Strategy), winningStrategy s :=
sorry

end first_player_can_force_odd_result_l2710_271059


namespace expression_value_l2710_271015

theorem expression_value (x y : ℝ) (h : |x + 1| + (y - 2)^2 = 0) :
  4 * x^2 * y - (6 * x * y - 3 * (4 * x * y - 2) - x^2 * y) + 1 = -7 := by
  sorry

end expression_value_l2710_271015


namespace min_value_M_l2710_271097

theorem min_value_M (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) :
  (4 * Real.sqrt 3) / 3 ≤ max (a + 1/b) (max (b + 1/c) (c + 1/a)) ∧
  ∃ a b c, 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 + c^2 = 1 ∧
    (4 * Real.sqrt 3) / 3 = max (a + 1/b) (max (b + 1/c) (c + 1/a)) := by
  sorry

end min_value_M_l2710_271097


namespace number_ordering_l2710_271003

theorem number_ordering : (6 : ℝ)^10 < 3^20 ∧ 3^20 < 2^30 := by
  sorry

end number_ordering_l2710_271003


namespace stating_transfer_equality_l2710_271031

/-- Represents a glass containing a mixture of wine and water -/
structure Glass where
  total_volume : ℝ
  wine_volume : ℝ
  water_volume : ℝ
  volume_constraint : total_volume = wine_volume + water_volume

/-- Represents the state of two glasses after the transfer process -/
structure TransferState where
  wine_glass : Glass
  water_glass : Glass
  volume_conserved : wine_glass.total_volume = water_glass.total_volume

/-- 
Theorem stating that after the transfer process, the volume of wine in the water glass 
is equal to the volume of water in the wine glass 
-/
theorem transfer_equality (state : TransferState) : 
  state.wine_glass.water_volume = state.water_glass.wine_volume := by
  sorry

#check transfer_equality

end stating_transfer_equality_l2710_271031


namespace sqrt_13_parts_sum_l2710_271088

theorem sqrt_13_parts_sum (a b : ℝ) : 
  (3 : ℝ) < Real.sqrt 13 ∧ Real.sqrt 13 < 4 →
  a = ⌊Real.sqrt 13⌋ →
  b = Real.sqrt 13 - ⌊Real.sqrt 13⌋ →
  a^2 + b - Real.sqrt 13 = 6 := by
  sorry

end sqrt_13_parts_sum_l2710_271088


namespace correct_allocation_schemes_l2710_271012

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 6

/-- Represents the number of venues -/
def num_venues : ℕ := 3

/-- Represents the number of volunteers per group -/
def group_size : ℕ := 2

/-- Represents that volunteers A and B must be in the same group -/
def fixed_pair : ℕ := 1

/-- The number of ways to allocate volunteers to venues -/
def allocation_schemes : ℕ := 18

/-- Theorem stating that the number of allocation schemes is correct -/
theorem correct_allocation_schemes :
  (num_volunteers.choose group_size * (num_volunteers - group_size).choose group_size / 2) *
  num_venues.factorial = allocation_schemes := by
  sorry

end correct_allocation_schemes_l2710_271012


namespace count_divisible_sum_l2710_271068

theorem count_divisible_sum : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (10 * n) % (n * (n + 1) / 2) = 0) ∧ 
    (∀ n ∉ S, n > 0 → (10 * n) % (n * (n + 1) / 2) ≠ 0) ∧ 
    Finset.card S = 5 := by
  sorry

end count_divisible_sum_l2710_271068


namespace scarf_wool_calculation_l2710_271027

/-- The number of balls of wool used for a scarf -/
def scarf_wool : ℕ := sorry

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := 8

/-- The number of balls of wool used for a sweater -/
def sweater_wool : ℕ := 4

/-- The total number of balls of wool used -/
def total_wool : ℕ := 82

theorem scarf_wool_calculation :
  scarf_wool * aaron_scarves + 
  sweater_wool * (aaron_sweaters + enid_sweaters) = 
  total_wool ∧ scarf_wool = 3 := by sorry

end scarf_wool_calculation_l2710_271027


namespace log_one_fifth_25_l2710_271056

-- Define the logarithm function for an arbitrary base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem statement
theorem log_one_fifth_25 : log (1/5) 25 = -2 := by sorry

end log_one_fifth_25_l2710_271056


namespace office_call_probabilities_l2710_271009

/-- Represents the probability of a call being for a specific person -/
structure CallProbability where
  A : ℚ
  B : ℚ
  C : ℚ
  sum_to_one : A + B + C = 1

/-- Calculates the probability of all three calls being for the same person -/
def prob_all_same (p : CallProbability) : ℚ :=
  p.A^3 + p.B^3 + p.C^3

/-- Calculates the probability of exactly two out of three calls being for A -/
def prob_two_for_A (p : CallProbability) : ℚ :=
  3 * p.A^2 * (1 - p.A)

theorem office_call_probabilities :
  ∃ (p : CallProbability),
    p.A = 1/6 ∧ p.B = 1/3 ∧ p.C = 1/2 ∧
    prob_all_same p = 1/6 ∧
    prob_two_for_A p = 5/72 := by
  sorry

end office_call_probabilities_l2710_271009


namespace car_B_speed_l2710_271090

/-- Proves that the speed of car B is 90 km/h given the problem conditions -/
theorem car_B_speed (distance : ℝ) (time : ℝ) (speed_ratio : ℝ × ℝ) :
  distance = 88 →
  time = 32 / 60 →
  speed_ratio = (5, 6) →
  ∃ (speed_A speed_B : ℝ),
    speed_A / speed_B = speed_ratio.1 / speed_ratio.2 ∧
    distance = (speed_A + speed_B) * time ∧
    speed_B = 90 := by
  sorry

end car_B_speed_l2710_271090


namespace data_tape_cost_calculation_l2710_271065

/-- The cost of mounting a data tape for a computer program run. -/
def data_tape_cost : ℝ := 5.35

/-- The operating-system overhead cost per run. -/
def os_overhead_cost : ℝ := 1.07

/-- The cost of computer time per millisecond. -/
def computer_time_cost_per_ms : ℝ := 0.023

/-- The total cost for one run of the program. -/
def total_cost : ℝ := 40.92

/-- The duration of the computer program run in seconds. -/
def run_duration_seconds : ℝ := 1.5

theorem data_tape_cost_calculation :
  data_tape_cost = total_cost - (os_overhead_cost + computer_time_cost_per_ms * (run_duration_seconds * 1000)) :=
by sorry

end data_tape_cost_calculation_l2710_271065


namespace root_sum_absolute_value_l2710_271098

theorem root_sum_absolute_value (m : ℤ) (a b c : ℤ) : 
  (∃ (m : ℤ), ∀ (x : ℤ), x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  |a| + |b| + |c| = 102 :=
by sorry

end root_sum_absolute_value_l2710_271098


namespace floor_equality_iff_in_range_l2710_271032

theorem floor_equality_iff_in_range (x : ℝ) : 
  ⌊2 * x + 1/2⌋ = ⌊x + 3⌋ ↔ x ∈ Set.Ici (5/2) ∩ Set.Iio (7/2) := by
  sorry

end floor_equality_iff_in_range_l2710_271032


namespace partner_a_receives_4800_l2710_271066

/-- Calculates the money received by partner a in a business partnership --/
def money_received_by_a (a_investment b_investment total_profit : ℚ) : ℚ :=
  let management_fee := 0.1 * total_profit
  let remaining_profit := total_profit - management_fee
  let total_investment := a_investment + b_investment
  let a_profit_share := (a_investment / total_investment) * remaining_profit
  management_fee + a_profit_share

/-- Theorem stating that given the problem conditions, partner a receives 4800 rs --/
theorem partner_a_receives_4800 :
  money_received_by_a 20000 25000 9600 = 4800 := by
  sorry

end partner_a_receives_4800_l2710_271066


namespace root_expression_value_l2710_271089

theorem root_expression_value (p m n : ℝ) : 
  (m^2 + (p - 2) * m + 1 = 0) → 
  (n^2 + (p - 2) * n + 1 = 0) → 
  (m^2 + p * m + 1) * (n^2 + p * n + 1) - 2 = 2 := by
sorry

end root_expression_value_l2710_271089


namespace triangle_and_squares_area_l2710_271045

theorem triangle_and_squares_area (x : ℝ) : 
  let triangle_area := (1/2) * (3*x) * (4*x)
  let square1_area := (3*x)^2
  let square2_area := (4*x)^2
  let square3_area := (6*x)^2
  let total_area := triangle_area + square1_area + square2_area + square3_area
  total_area = 1288 → x = Real.sqrt (1288/67) := by
sorry

end triangle_and_squares_area_l2710_271045


namespace sqrt_3_minus_2_squared_l2710_271073

theorem sqrt_3_minus_2_squared : (Real.sqrt 3 - 2) * (Real.sqrt 3 - 2) = 7 - 4 * Real.sqrt 3 := by
  sorry

end sqrt_3_minus_2_squared_l2710_271073


namespace not_perfect_square_l2710_271033

theorem not_perfect_square (n : ℤ) (h : n > 4) : ¬∃ (k : ℕ), n^2 - 3*n = k^2 := by
  sorry

end not_perfect_square_l2710_271033


namespace complex_quadrant_l2710_271057

theorem complex_quadrant (z : ℂ) (h : (z + Complex.I) * Complex.I = 1 + z) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_quadrant_l2710_271057


namespace digit_sum_difference_l2710_271060

-- Define a function to calculate the sum of digits of a number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := sorry

-- Define the sum of digits for all even numbers from 1 to 1000
def sumEvenDigits : ℕ := 
  (List.range 1000).filter isEven |>.map sumOfDigits |>.sum

-- Define the sum of digits for all odd numbers from 1 to 1000
def sumOddDigits : ℕ := 
  (List.range 1000).filter (λ n => ¬(isEven n)) |>.map sumOfDigits |>.sum

-- Theorem statement
theorem digit_sum_difference :
  sumOddDigits - sumEvenDigits = 499 := by sorry

end digit_sum_difference_l2710_271060


namespace arithmetic_mean_of_special_set_l2710_271079

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 2) :
  let set := [1 - 1 / n, 1 - 2 / n] ++ List.replicate (n - 2) 1
  List.sum set / n = 1 - 3 / n^2 := by
  sorry

end arithmetic_mean_of_special_set_l2710_271079


namespace gcd_b_always_one_l2710_271094

def b (n : ℕ) : ℤ := (8^n - 1) / 7

theorem gcd_b_always_one (n : ℕ) : Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1))) = 1 := by
  sorry

end gcd_b_always_one_l2710_271094


namespace inscribed_triangle_area_l2710_271062

theorem inscribed_triangle_area (r : ℝ) (A B C : ℝ) :
  r = 18 / Real.pi →
  A = 60 * Real.pi / 180 →
  B = 120 * Real.pi / 180 →
  C = 180 * Real.pi / 180 →
  (1/2) * r^2 * (Real.sin A + Real.sin B + Real.sin C) = 162 * Real.sqrt 3 / Real.pi^2 :=
by sorry

end inscribed_triangle_area_l2710_271062
