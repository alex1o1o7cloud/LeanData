import Mathlib

namespace number_of_schools_l3590_359036

def students_per_school : ℕ := 247
def total_students : ℕ := 6175

theorem number_of_schools : (total_students / students_per_school : ℕ) = 25 := by
  sorry

end number_of_schools_l3590_359036


namespace quadratic_transformation_l3590_359063

/-- Given a quadratic function f(x) = ax² + bx + c passing through specific points,
    prove that g(x) = cx² + 2bx + a has a specific vertex form -/
theorem quadratic_transformation (a b c : ℝ) 
  (h1 : c = 1)
  (h2 : a + b + c = -2)
  (h3 : a - b + c = 2) :
  let f := fun x => a * x^2 + b * x + c
  let g := fun x => c * x^2 + 2 * b * x + a
  ∀ x, g x = (x - 2)^2 - 5 := by sorry

end quadratic_transformation_l3590_359063


namespace specific_triangle_count_is_32_l3590_359082

/-- Represents the count of triangles at different levels in a structure --/
structure TriangleCount where
  smallest : Nat
  intermediate : Nat
  larger : Nat
  even_larger : Nat
  whole_structure : Nat

/-- Calculates the total number of triangles in the structure --/
def total_triangles (count : TriangleCount) : Nat :=
  count.smallest + count.intermediate + count.larger + count.even_larger + count.whole_structure

/-- Theorem stating that for a specific triangle count, the total number of triangles is 32 --/
theorem specific_triangle_count_is_32 :
  ∃ (count : TriangleCount),
    count.smallest = 2 ∧
    count.intermediate = 6 ∧
    count.larger = 6 ∧
    count.even_larger = 6 ∧
    count.whole_structure = 12 ∧
    total_triangles count = 32 := by
  sorry

#eval total_triangles { smallest := 2, intermediate := 6, larger := 6, even_larger := 6, whole_structure := 12 }

end specific_triangle_count_is_32_l3590_359082


namespace average_score_proof_l3590_359025

def student_A_score : ℚ := 92
def student_B_score : ℚ := 75
def student_C_score : ℚ := 98

def number_of_students : ℚ := 3

def average_score : ℚ := (student_A_score + student_B_score + student_C_score) / number_of_students

theorem average_score_proof : average_score = 88.3333333333333 := by
  sorry

end average_score_proof_l3590_359025


namespace relationship_abc_l3590_359088

theorem relationship_abc (a b c : ℝ) (ha : a = Real.sqrt 6 + Real.sqrt 7) 
  (hb : b = Real.sqrt 5 + Real.sqrt 8) (hc : c = 5) : c < b ∧ b < a := by
  sorry

end relationship_abc_l3590_359088


namespace little_john_initial_money_l3590_359007

theorem little_john_initial_money :
  let sweets_cost : ℚ := 1.25
  let friends_count : ℕ := 2
  let money_per_friend : ℚ := 1.20
  let money_left : ℚ := 4.85
  let initial_money : ℚ := sweets_cost + friends_count * money_per_friend + money_left
  initial_money = 8.50 := by sorry

end little_john_initial_money_l3590_359007


namespace largest_integer_less_than_sqrt5_plus_sqrt3_to_6th_l3590_359058

theorem largest_integer_less_than_sqrt5_plus_sqrt3_to_6th (n : ℕ) : 
  n = 3322 ↔ n = ⌊(Real.sqrt 5 + Real.sqrt 3)^6⌋ :=
by sorry

end largest_integer_less_than_sqrt5_plus_sqrt3_to_6th_l3590_359058


namespace cost_price_is_four_l3590_359037

/-- The cost price of a bag of popcorn -/
def cost_price : ℝ := sorry

/-- The selling price of a bag of popcorn -/
def selling_price : ℝ := 8

/-- The number of bags sold -/
def bags_sold : ℝ := 30

/-- The total profit -/
def total_profit : ℝ := 120

/-- Theorem: The cost price of each bag of popcorn is $4 -/
theorem cost_price_is_four :
  cost_price = 4 :=
by
  have h1 : total_profit = bags_sold * (selling_price - cost_price) :=
    sorry
  sorry

end cost_price_is_four_l3590_359037


namespace bookstore_sales_ratio_l3590_359003

theorem bookstore_sales_ratio :
  let tuesday_sales : ℕ := 7
  let wednesday_sales : ℕ := 3 * tuesday_sales
  let total_sales : ℕ := 91
  let thursday_sales : ℕ := total_sales - (tuesday_sales + wednesday_sales)
  (thursday_sales : ℚ) / wednesday_sales = 3 / 1 :=
by sorry

end bookstore_sales_ratio_l3590_359003


namespace not_perfect_square_l3590_359034

theorem not_perfect_square (a b : ℕ+) : ¬∃ k : ℤ, (a : ℤ)^2 + Int.ceil ((4 * (a : ℤ)^2) / (b : ℤ)) = k^2 := by
  sorry

end not_perfect_square_l3590_359034


namespace square_of_linear_expression_l3590_359068

theorem square_of_linear_expression (x : ℝ) :
  x = -2 → (3*x + 4)^2 = 4 := by
  sorry

end square_of_linear_expression_l3590_359068


namespace trig_identity_l3590_359019

theorem trig_identity : Real.cos (70 * π / 180) * Real.sin (50 * π / 180) - 
                        Real.cos (200 * π / 180) * Real.sin (40 * π / 180) = 
                        Real.sqrt 3 / 2 := by
  sorry

end trig_identity_l3590_359019


namespace arithmetic_sequence_sum_l3590_359087

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h4 : seq.S 4 = -4)
    (h6 : seq.S 6 = 6) :
    seq.S 5 = 0 := by
  sorry

end arithmetic_sequence_sum_l3590_359087


namespace project_budget_equality_l3590_359008

/-- Represents the annual budget change for a project -/
structure BudgetChange where
  initial : ℕ  -- Initial budget in dollars
  annual : ℤ   -- Annual change in dollars (positive for increase, negative for decrease)

/-- Calculates the budget after a given number of years -/
def budget_after_years (bc : BudgetChange) (years : ℕ) : ℤ :=
  bc.initial + years * bc.annual

/-- The problem statement -/
theorem project_budget_equality (q v : BudgetChange) 
  (hq_initial : q.initial = 540000)
  (hv_initial : v.initial = 780000)
  (hq_annual : q.annual = 30000)
  (h_equal_after_4 : budget_after_years q 4 = budget_after_years v 4) :
  v.annual = -30000 := by
  sorry

end project_budget_equality_l3590_359008


namespace sum_of_cosines_l3590_359081

theorem sum_of_cosines (z : ℂ) (α : ℝ) (h1 : z^7 = 1) (h2 : z ≠ 1) (h3 : z = Complex.exp (Complex.I * α)) :
  Real.cos α + Real.cos (2 * α) + Real.cos (4 * α) = -1/2 := by
  sorry

end sum_of_cosines_l3590_359081


namespace journey_gas_cost_l3590_359047

/-- Calculates the cost of gas for a journey given odometer readings, fuel efficiency, and gas price -/
def gas_cost (initial_reading : ℕ) (final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  ((final_reading - initial_reading : ℚ) / fuel_efficiency) * gas_price

/-- Proves that the cost of gas for the given journey is $3.46 -/
theorem journey_gas_cost :
  gas_cost 85340 85368 32 (395/100) = 346/100 := by
  sorry

end journey_gas_cost_l3590_359047


namespace mittens_per_box_l3590_359053

/-- Given the conditions of Chloe's winter clothing boxes, prove the number of mittens per box -/
theorem mittens_per_box 
  (num_boxes : ℕ) 
  (scarves_per_box : ℕ) 
  (total_pieces : ℕ) 
  (h1 : num_boxes = 4) 
  (h2 : scarves_per_box = 2) 
  (h3 : total_pieces = 32) : 
  (total_pieces - num_boxes * scarves_per_box) / num_boxes = 6 := by
  sorry

end mittens_per_box_l3590_359053


namespace infinitely_many_primes_3_mod_4_l3590_359091

theorem infinitely_many_primes_3_mod_4 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 4 = 3} := by
  sorry

end infinitely_many_primes_3_mod_4_l3590_359091


namespace milkshake_ice_cream_difference_l3590_359074

/-- Given the number of milkshakes and ice cream cones sold, prove the difference -/
theorem milkshake_ice_cream_difference (milkshakes ice_cream_cones : ℕ) 
  (h1 : milkshakes = 82) (h2 : ice_cream_cones = 67) : 
  milkshakes - ice_cream_cones = 15 := by
  sorry

end milkshake_ice_cream_difference_l3590_359074


namespace foreign_stamps_count_l3590_359021

theorem foreign_stamps_count (total : ℕ) (old : ℕ) (foreign_and_old : ℕ) (neither : ℕ) :
  total = 200 →
  old = 60 →
  foreign_and_old = 20 →
  neither = 70 →
  ∃ foreign : ℕ, foreign = 90 ∧ 
    foreign + old - foreign_and_old = total - neither :=
by sorry

end foreign_stamps_count_l3590_359021


namespace max_beauty_value_bound_l3590_359075

/-- Represents a figure with circles and segments arranged into pentagons -/
structure Figure where
  circles : Nat
  segments : Nat
  pentagons : Nat

/-- Represents a method of filling numbers in the circles -/
def FillingMethod := Fin 15 → Fin 3

/-- Calculates the beauty value of a filling method -/
def beautyValue (f : Figure) (m : FillingMethod) : Nat :=
  sorry

/-- The maximum possible beauty value -/
def maxBeautyValue (f : Figure) : Nat :=
  sorry

theorem max_beauty_value_bound (f : Figure) 
  (h1 : f.circles = 15) 
  (h2 : f.segments = 20) 
  (h3 : f.pentagons = 6) : 
  maxBeautyValue f ≤ 17 := by
  sorry

end max_beauty_value_bound_l3590_359075


namespace dvd_money_calculation_l3590_359064

/-- Given the cost of one pack of DVDs and the number of packs that can be bought,
    calculate the total amount of money available. -/
theorem dvd_money_calculation (cost_per_pack : ℕ) (num_packs : ℕ) :
  cost_per_pack = 12 → num_packs = 11 → cost_per_pack * num_packs = 132 := by
  sorry

end dvd_money_calculation_l3590_359064


namespace product_simplification_l3590_359016

theorem product_simplification (y : ℝ) : (16 * y^3) * (12 * y^5) * (1 / (4 * y)^3) = 3 * y^5 := by
  sorry

end product_simplification_l3590_359016


namespace park_attendance_solution_l3590_359057

/-- Represents the number of people at Minewaska State Park --/
structure ParkAttendance where
  hikers : ℕ
  bikers : ℕ
  kayakers : ℕ

/-- The conditions of the park attendance problem --/
def parkProblem (p : ParkAttendance) : Prop :=
  p.hikers = p.bikers + 178 ∧
  p.kayakers * 2 = p.bikers ∧
  p.hikers + p.bikers + p.kayakers = 920

/-- The theorem stating the solution to the park attendance problem --/
theorem park_attendance_solution :
  ∃ p : ParkAttendance, parkProblem p ∧ p.hikers = 474 := by
  sorry

end park_attendance_solution_l3590_359057


namespace machines_count_l3590_359001

/-- The number of machines that complete a job lot in 6 hours -/
def N : ℕ := 8

/-- The time taken by N machines to complete the job lot -/
def time_N : ℕ := 6

/-- The number of machines in the second scenario -/
def machines_2 : ℕ := 4

/-- The time taken by machines_2 to complete the job lot -/
def time_2 : ℕ := 12

/-- The work rate of a single machine (job lots per hour) -/
def work_rate : ℚ := 1 / 48

theorem machines_count :
  N * work_rate * time_N = 1 ∧
  machines_2 * work_rate * time_2 = 1 :=
sorry

#check machines_count

end machines_count_l3590_359001


namespace hospital_workers_count_l3590_359018

/-- The number of other workers at the hospital -/
def num_other_workers : ℕ := 2

/-- The total number of workers at the hospital -/
def total_workers : ℕ := num_other_workers + 2

/-- The probability of selecting both John and David when choosing 2 workers randomly -/
def prob_select_john_and_david : ℚ := 1 / 6

theorem hospital_workers_count :
  (prob_select_john_and_david = 1 / (total_workers.choose 2)) →
  num_other_workers = 2 := by
sorry

#eval num_other_workers

end hospital_workers_count_l3590_359018


namespace grid_value_bound_l3590_359012

/-- The value of a square in the grid -/
def square_value (is_filled : Bool) (filled_neighbors : Nat) : Nat :=
  if is_filled then 0 else filled_neighbors

/-- The maximum number of neighbors a square can have -/
def max_neighbors : Nat := 8

/-- The function f(m,n) representing the largest total value of squares in the grid -/
noncomputable def f (m n : Nat) : Nat :=
  sorry  -- Definition of f(m,n) is complex and depends on optimal grid configuration

/-- The theorem stating that 2 is the minimal constant C such that f(m,n) / (m*n) ≤ C -/
theorem grid_value_bound (m n : Nat) (hm : m > 0) (hn : n > 0) :
  (f m n : ℝ) / (m * n : ℝ) ≤ 2 ∧ ∀ C : ℝ, (∀ m' n' : Nat, m' > 0 → n' > 0 → (f m' n' : ℝ) / (m' * n' : ℝ) ≤ C) → C ≥ 2 :=
  sorry


end grid_value_bound_l3590_359012


namespace player_playing_time_l3590_359026

/-- Calculates the playing time for each player in a sports tournament. -/
theorem player_playing_time (total_players : ℕ) (players_on_field : ℕ) (match_duration : ℕ) :
  total_players = 10 →
  players_on_field = 8 →
  match_duration = 45 →
  (players_on_field * match_duration) / total_players = 36 := by
  sorry

end player_playing_time_l3590_359026


namespace sum_of_three_smallest_primes_l3590_359023

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def primes_between_1_and_50 : Set ℕ := {n : ℕ | 1 < n ∧ n ≤ 50 ∧ is_prime n}

theorem sum_of_three_smallest_primes :
  ∃ (a b c : ℕ), a ∈ primes_between_1_and_50 ∧
                 b ∈ primes_between_1_and_50 ∧
                 c ∈ primes_between_1_and_50 ∧
                 a < b ∧ b < c ∧
                 (∀ p ∈ primes_between_1_and_50, p ≥ c ∨ p = a ∨ p = b) ∧
                 a + b + c = 10 :=
sorry

end sum_of_three_smallest_primes_l3590_359023


namespace room_ratios_l3590_359033

/-- Represents a rectangular room with given length and width. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Represents a ratio as a pair of natural numbers. -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

theorem room_ratios (room : Rectangle) 
    (h1 : room.length = 24) 
    (h2 : room.width = 14) : 
    (∃ r1 : Ratio, r1.numerator = 6 ∧ r1.denominator = 19 ∧ 
      r1.numerator * perimeter room = r1.denominator * room.length) ∧
    (∃ r2 : Ratio, r2.numerator = 7 ∧ r2.denominator = 38 ∧ 
      r2.numerator * perimeter room = r2.denominator * room.width) := by
  sorry


end room_ratios_l3590_359033


namespace intersection_point_l3590_359045

/-- The slope of the given line -/
def m : ℝ := 2

/-- The y-intercept of the given line -/
def b : ℝ := 5

/-- The x-coordinate of the point on the perpendicular line -/
def x₀ : ℝ := 5

/-- The y-coordinate of the point on the perpendicular line -/
def y₀ : ℝ := 5

/-- The x-coordinate of the claimed intersection point -/
def x_int : ℝ := 1

/-- The y-coordinate of the claimed intersection point -/
def y_int : ℝ := 7

/-- Theorem stating that (x_int, y_int) is the intersection point of the given line
    and its perpendicular line passing through (x₀, y₀) -/
theorem intersection_point :
  (y_int = m * x_int + b) ∧
  (y_int - y₀ = -(1/m) * (x_int - x₀)) ∧
  (∀ x y : ℝ, (y = m * x + b) ∧ (y - y₀ = -(1/m) * (x - x₀)) → x = x_int ∧ y = y_int) :=
by sorry

end intersection_point_l3590_359045


namespace major_axis_length_l3590_359027

/-- Represents a right circular cylinder --/
structure RightCircularCylinder where
  radius : ℝ

/-- Represents an ellipse formed by intersecting a plane with a cylinder --/
structure Ellipse where
  minorAxis : ℝ
  majorAxis : ℝ

/-- The ellipse formed by intersecting a plane with a right circular cylinder --/
def cylinderEllipse (c : RightCircularCylinder) : Ellipse :=
  { minorAxis := 2 * c.radius,
    majorAxis := 3 * c.radius }

theorem major_axis_length (c : RightCircularCylinder) 
  (h : c.radius = 1) :
  (cylinderEllipse c).majorAxis = 3 ∧
  (cylinderEllipse c).majorAxis = 1.5 * (cylinderEllipse c).minorAxis :=
by sorry

end major_axis_length_l3590_359027


namespace parallelogram_area_example_l3590_359009

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 10 meters and height 7 meters is 70 square meters -/
theorem parallelogram_area_example : parallelogram_area 10 7 = 70 := by
  sorry

end parallelogram_area_example_l3590_359009


namespace positive_difference_of_numbers_l3590_359010

theorem positive_difference_of_numbers (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (diff_squares_eq : a^2 - b^2 = 40) : 
  |a - b| = 4 := by
sorry

end positive_difference_of_numbers_l3590_359010


namespace triangle_side_length_l3590_359052

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- Positive angles
  A + B + C = π →  -- Sum of angles in a triangle
  A = π / 3 →  -- 60 degrees in radians
  B = π / 4 →  -- 45 degrees in radians
  b = Real.sqrt 6 →
  a / Real.sin A = b / Real.sin B →  -- Law of Sines
  a = 3 := by
sorry

end triangle_side_length_l3590_359052


namespace xy_max_value_l3590_359015

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 16) :
  x * y ≤ 32 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = 16 ∧ x * y = 32 :=
sorry

end xy_max_value_l3590_359015


namespace divisibility_reversal_implies_factor_of_99_l3590_359000

def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisibility_reversal_implies_factor_of_99 (k : ℕ) :
  (∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) →
  k ∣ 99 := by
  sorry

end divisibility_reversal_implies_factor_of_99_l3590_359000


namespace same_number_probability_l3590_359028

/-- The upper bound for the selected numbers -/
def upperBound : ℕ := 300

/-- Billy's number is a multiple of this value -/
def billyMultiple : ℕ := 36

/-- Bobbi's number is a multiple of this value -/
def bobbiMultiple : ℕ := 48

/-- The probability of Billy and Bobbi selecting the same number -/
def sameProbability : ℚ := 1 / 24

theorem same_number_probability :
  (∃ (b₁ b₂ : ℕ), b₁ > 0 ∧ b₂ > 0 ∧ b₁ < upperBound ∧ b₂ < upperBound ∧
    b₁ % billyMultiple = 0 ∧ b₂ % bobbiMultiple = 0) →
  (∃ (n : ℕ), n > 0 ∧ n < upperBound ∧ n % billyMultiple = 0 ∧ n % bobbiMultiple = 0) →
  sameProbability = (Nat.card {n : ℕ | n > 0 ∧ n < upperBound ∧ n % billyMultiple = 0 ∧ n % bobbiMultiple = 0} : ℚ) /
                    ((Nat.card {n : ℕ | n > 0 ∧ n < upperBound ∧ n % billyMultiple = 0} : ℚ) *
                     (Nat.card {n : ℕ | n > 0 ∧ n < upperBound ∧ n % bobbiMultiple = 0} : ℚ)) :=
by sorry

end same_number_probability_l3590_359028


namespace binomial_expansion_unique_m_l3590_359093

/-- Given constants b and y, and a natural number m, such that the second, third, and fourth terms
    in the expansion of (b + y)^m are 6, 24, and 60 respectively, prove that m = 11. -/
theorem binomial_expansion_unique_m (b y : ℝ) (m : ℕ) 
  (h1 : (m.choose 1) * b^(m-1) * y = 6)
  (h2 : (m.choose 2) * b^(m-2) * y^2 = 24)
  (h3 : (m.choose 3) * b^(m-3) * y^3 = 60) :
  m = 11 := by
  sorry

end binomial_expansion_unique_m_l3590_359093


namespace a_stops_implies_smudged_l3590_359022

-- Define the set of ladies
inductive Lady : Type
  | A
  | B
  | C

-- Define the state of a lady's face
inductive FaceState : Type
  | Clean
  | Smudged

-- Define the laughing state of a lady
inductive LaughState : Type
  | Laughing
  | NotLaughing

-- Function to get the face state of a lady
def faceState : Lady → FaceState
  | Lady.A => FaceState.Smudged
  | Lady.B => FaceState.Smudged
  | Lady.C => FaceState.Smudged

-- Function to get the initial laugh state of a lady
def initialLaughState : Lady → LaughState
  | _ => LaughState.Laughing

-- Function to determine if a lady can see another lady's smudged face
def canSeeSmugedFace (observer viewer : Lady) : Prop :=
  observer ≠ viewer ∧ faceState viewer = FaceState.Smudged

-- Theorem: If A stops laughing, it implies A must have a smudged face
theorem a_stops_implies_smudged :
  (initialLaughState Lady.A = LaughState.Laughing) →
  (∃ (newLaughState : Lady → LaughState),
    newLaughState Lady.A = LaughState.NotLaughing ∧
    (∀ l : Lady, l ≠ Lady.A → newLaughState l = LaughState.Laughing)) →
  faceState Lady.A = FaceState.Smudged :=
by
  sorry


end a_stops_implies_smudged_l3590_359022


namespace matches_per_box_l3590_359099

/-- Given 5 dozen boxes containing a total of 1200 matches, prove that each box contains 20 matches. -/
theorem matches_per_box (dozen_boxes : ℕ) (total_matches : ℕ) : 
  dozen_boxes = 5 → total_matches = 1200 → (dozen_boxes * 12) * 20 = total_matches := by
  sorry

end matches_per_box_l3590_359099


namespace units_digit_of_17_times_27_l3590_359032

theorem units_digit_of_17_times_27 : (17 * 27) % 10 = 9 := by
  sorry

end units_digit_of_17_times_27_l3590_359032


namespace derivative_of_composite_l3590_359049

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the condition that f'(3) = 9
variable (h : deriv f 3 = 9)

-- State the theorem
theorem derivative_of_composite (f : ℝ → ℝ) (h : deriv f 3 = 9) :
  deriv (fun x ↦ f (3 * x^2)) 1 = 54 := by
  sorry

end derivative_of_composite_l3590_359049


namespace units_digit_of_sum_of_cubes_l3590_359014

theorem units_digit_of_sum_of_cubes : ∃ n : ℕ, n < 10 ∧ (41^3 + 23^3) % 10 = n ∧ n = 8 := by
  sorry

end units_digit_of_sum_of_cubes_l3590_359014


namespace set_equality_l3590_359013

open Set

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define sets M and N
def M : Set Nat := {3, 4, 5}
def N : Set Nat := {1, 3, 6}

-- Theorem statement
theorem set_equality : (U \ M) ∩ N = {1, 6} := by sorry

end set_equality_l3590_359013


namespace raul_remaining_money_l3590_359095

def initial_amount : ℕ := 87
def number_of_comics : ℕ := 8
def cost_per_comic : ℕ := 4

theorem raul_remaining_money : 
  initial_amount - (number_of_comics * cost_per_comic) = 55 := by
  sorry

end raul_remaining_money_l3590_359095


namespace p_one_eq_p_two_p_decreasing_l3590_359096

/-- The number of items in the collection -/
def n : ℕ := 10

/-- The probability of finding any specific item in a randomly chosen container -/
def prob_item : ℝ := 0.1

/-- The probability that exactly k items are missing from the second collection when the first collection is completed -/
noncomputable def p (k : ℕ) : ℝ := sorry

/-- Theorem stating that p_1 equals p_2 -/
theorem p_one_eq_p_two : p 1 = p 2 := by sorry

/-- Theorem stating the strict decreasing order of probabilities -/
theorem p_decreasing {i j : ℕ} (h1 : 2 ≤ i) (h2 : i < j) (h3 : j ≤ n) : p i > p j := by sorry

end p_one_eq_p_two_p_decreasing_l3590_359096


namespace last_digit_of_repeated_seven_exponentiation_l3590_359085

def repeated_exponentiation (base : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => base
  | n + 1 => repeated_exponentiation (base^base) n

theorem last_digit_of_repeated_seven_exponentiation :
  repeated_exponentiation 7 1000 % 10 = 1 := by
  sorry

end last_digit_of_repeated_seven_exponentiation_l3590_359085


namespace windows_preference_l3590_359080

theorem windows_preference (total_students : ℕ) (mac_preference : ℕ) (no_preference : ℕ) 
  (h1 : total_students = 210)
  (h2 : mac_preference = 60)
  (h3 : no_preference = 90) :
  total_students - (mac_preference + mac_preference / 3 + no_preference) = 40 := by
  sorry

end windows_preference_l3590_359080


namespace fish_value_in_dragon_scales_l3590_359020

/-- In a magical kingdom with given exchange rates, prove the value of a fish in dragon scales -/
theorem fish_value_in_dragon_scales 
  (fish_to_bread : ℚ) -- Exchange rate of fish to bread
  (bread_to_scales : ℚ) -- Exchange rate of bread to dragon scales
  (h1 : 2 * fish_to_bread = 3) -- Two fish can be exchanged for three loaves of bread
  (h2 : bread_to_scales = 2) -- One loaf of bread can be traded for two dragon scales
  : fish_to_bread * bread_to_scales = 3 := by sorry

end fish_value_in_dragon_scales_l3590_359020


namespace max_k_value_l3590_359030

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 72) →
  k ≤ 2 * Real.sqrt 26 :=
by sorry

end max_k_value_l3590_359030


namespace base_3_division_theorem_l3590_359059

def base_3_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, d) acc => acc + d * (3 ^ i)) 0

def decimal_to_base_3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
  aux n []

theorem base_3_division_theorem :
  let dividend := [1, 0, 2, 1]  -- 1021₃ in reverse order
  let divisor := [1, 1]         -- 11₃ in reverse order
  let quotient := [2, 2]        -- 22₃ in reverse order
  (base_3_to_decimal dividend) / (base_3_to_decimal divisor) = base_3_to_decimal quotient :=
by sorry

end base_3_division_theorem_l3590_359059


namespace equation_solution_l3590_359050

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem equation_solution :
  ∃ (x : ℝ), 2 * (f x) - 16 = f (x - 6) ∧ x = 1 := by
  sorry

end equation_solution_l3590_359050


namespace no_solution_iff_m_leq_three_l3590_359072

/-- Given a real number m, the system of inequalities {x - m > 2, x - 2m < -1} has no solution if and only if m ≤ 3. -/
theorem no_solution_iff_m_leq_three (m : ℝ) : 
  (∀ x : ℝ, ¬(x - m > 2 ∧ x - 2*m < -1)) ↔ m ≤ 3 := by
  sorry

end no_solution_iff_m_leq_three_l3590_359072


namespace laura_garden_area_l3590_359094

/-- Represents a rectangular garden with fence posts --/
structure Garden where
  total_posts : ℕ
  gap : ℕ
  longer_side_post_ratio : ℕ

/-- Calculates the area of the garden given its specifications --/
def garden_area (g : Garden) : ℕ :=
  let shorter_side_posts := (g.total_posts + 4) / (1 + g.longer_side_post_ratio)
  let longer_side_posts := shorter_side_posts * g.longer_side_post_ratio
  let shorter_side_length := (shorter_side_posts - 1) * g.gap
  let longer_side_length := (longer_side_posts - 1) * g.gap
  shorter_side_length * longer_side_length

theorem laura_garden_area :
  let g : Garden := { total_posts := 24, gap := 5, longer_side_post_ratio := 3 }
  garden_area g = 3000 := by
  sorry


end laura_garden_area_l3590_359094


namespace nine_sided_polygon_diagonals_l3590_359086

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 :=
by sorry

end nine_sided_polygon_diagonals_l3590_359086


namespace sum_of_coefficients_l3590_359038

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 12 := by
sorry

end sum_of_coefficients_l3590_359038


namespace other_root_of_quadratic_l3590_359054

theorem other_root_of_quadratic (m : ℝ) : 
  ((-1)^2 + (-1) + m = 0) → 
  (∃ (x : ℝ), x ≠ -1 ∧ x^2 + x + m = 0 ∧ x = 0) :=
by sorry

end other_root_of_quadratic_l3590_359054


namespace symmetric_circle_equation_l3590_359040

/-- Given a circle C that is symmetric to the circle (x+2)^2+(y-1)^2=1 with respect to the origin,
    prove that the equation of circle C is (x-2)^2+(y+1)^2=1 -/
theorem symmetric_circle_equation (C : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (-x, -y) ∈ {(x, y) | (x + 2)^2 + (y - 1)^2 = 1}) →
  C = {(x, y) | (x - 2)^2 + (y + 1)^2 = 1} :=
by sorry

end symmetric_circle_equation_l3590_359040


namespace special_isosceles_triangle_base_length_l3590_359070

/-- An isosceles triangle with side length a and a specific height property -/
structure SpecialIsoscelesTriangle (a : ℝ) where
  -- The triangle is isosceles with side length a
  side_length : ℝ
  is_isosceles : side_length = a
  -- The height dropped onto the base is equal to the segment connecting
  -- the midpoint of the base with the midpoint of the side
  height_property : ℝ → Prop

/-- The base length of the special isosceles triangle is a√3 -/
theorem special_isosceles_triangle_base_length 
  {a : ℝ} (t : SpecialIsoscelesTriangle a) : 
  ∃ (base : ℝ), base = a * Real.sqrt 3 := by
  sorry

end special_isosceles_triangle_base_length_l3590_359070


namespace sin_five_pi_six_plus_two_alpha_l3590_359004

theorem sin_five_pi_six_plus_two_alpha (α : Real) 
  (h : Real.cos (α + π/6) = 1/3) : 
  Real.sin (5*π/6 + 2*α) = -7/9 := by
  sorry

end sin_five_pi_six_plus_two_alpha_l3590_359004


namespace gcf_of_75_and_100_l3590_359092

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end gcf_of_75_and_100_l3590_359092


namespace solution_system_equations_l3590_359078

theorem solution_system_equations :
  ∀ x y z : ℝ,
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4 →
  ((x = 1/3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2)) :=
by
  sorry

end solution_system_equations_l3590_359078


namespace book_sale_loss_percentage_l3590_359005

/-- Proves that when the cost price of 30 books equals the selling price of 40 books, the loss percentage is 25% -/
theorem book_sale_loss_percentage 
  (cost_price selling_price : ℝ) 
  (h : 30 * cost_price = 40 * selling_price) : 
  (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end book_sale_loss_percentage_l3590_359005


namespace bus_wheel_radius_l3590_359098

/-- The radius of a bus wheel given its speed and revolutions per minute -/
theorem bus_wheel_radius 
  (speed_kmh : ℝ) 
  (rpm : ℝ) 
  (h1 : speed_kmh = 66) 
  (h2 : rpm = 175.15923566878982) : 
  ∃ (r : ℝ), abs (r - 99.89) < 0.01 := by
  sorry

end bus_wheel_radius_l3590_359098


namespace sphere_in_cone_l3590_359071

theorem sphere_in_cone (b d g : ℝ) : 
  let cone_base_radius : ℝ := 15
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := b * Real.sqrt d - g
  g = b + 6 →
  sphere_radius = (cone_height * cone_base_radius) / (cone_base_radius + Real.sqrt (cone_base_radius^2 + cone_height^2)) →
  b + d = 12.5 := by
sorry

end sphere_in_cone_l3590_359071


namespace simplify_powers_l3590_359076

theorem simplify_powers (x : ℝ) : x^5 * x^3 * 2 = 2 * x^8 := by
  sorry

end simplify_powers_l3590_359076


namespace histogram_group_width_l3590_359011

/-- Represents a group in a frequency histogram -/
structure HistogramGroup where
  a : ℝ
  b : ℝ
  m : ℝ  -- frequency
  h : ℝ  -- height
  h_pos : h > 0

/-- Theorem: The width of a histogram group is equal to its frequency divided by its height -/
theorem histogram_group_width (g : HistogramGroup) : |g.a - g.b| = g.m / g.h := by
  sorry

end histogram_group_width_l3590_359011


namespace price_difference_l3590_359017

theorem price_difference (P : ℝ) (h : P > 0) :
  let new_price := P * 1.2
  let discounted_price := new_price * 0.8
  new_price - discounted_price = P * 0.24 := by
sorry

end price_difference_l3590_359017


namespace point_in_fourth_quadrant_l3590_359039

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  is_in_fourth_quadrant 3 (-4) :=
sorry

end point_in_fourth_quadrant_l3590_359039


namespace even_digits_finite_fissile_squares_odd_digits_infinite_fissile_squares_l3590_359046

/-- A fissile square is a positive integer which is a perfect square,
    and whose digits form two perfect squares in a row. -/
def is_fissile_square (n : ℕ) : Prop :=
  ∃ (x y r : ℕ) (d : ℕ), 
    n = x^2 ∧ 
    n = 10^d * y^2 + r^2 ∧ 
    y^2 ≠ 0 ∧ r^2 ≠ 0

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Every square with an even number of digits is the right square of only finitely many fissile squares -/
theorem even_digits_finite_fissile_squares (r : ℕ) (h : Even (num_digits (r^2))) :
  {x : ℕ | is_fissile_square (x^2) ∧ ∃ (y : ℕ) (d : ℕ), x^2 = 10^d * y^2 + r^2 ∧ Even d}.Finite :=
sorry

/-- Theorem: Every square with an odd number of digits is the right square of infinitely many fissile squares -/
theorem odd_digits_infinite_fissile_squares (r : ℕ) (h : Odd (num_digits (r^2))) :
  {x : ℕ | is_fissile_square (x^2) ∧ ∃ (y : ℕ) (d : ℕ), x^2 = 10^d * y^2 + r^2 ∧ Odd d}.Infinite :=
sorry

end even_digits_finite_fissile_squares_odd_digits_infinite_fissile_squares_l3590_359046


namespace largest_expression_l3590_359066

-- Define the expressions
def expr_A : ℝ := (7 * 8) ^ (1/6)
def expr_B : ℝ := (8 * 7^(1/3))^(1/2)
def expr_C : ℝ := (7 * 8^(1/3))^(1/2)
def expr_D : ℝ := (7 * 8^(1/2))^(1/3)
def expr_E : ℝ := (8 * 7^(1/2))^(1/3)

-- Theorem statement
theorem largest_expression :
  expr_B = max expr_A (max expr_B (max expr_C (max expr_D expr_E))) :=
by sorry

end largest_expression_l3590_359066


namespace orange_trees_count_l3590_359083

theorem orange_trees_count (total_trees apple_trees : ℕ) 
  (h1 : total_trees = 74) 
  (h2 : apple_trees = 47) : 
  total_trees - apple_trees = 27 := by
  sorry

end orange_trees_count_l3590_359083


namespace polynomial_roots_problem_l3590_359061

/-- 
Given two real numbers u and v that are roots of the polynomial r(x) = x^3 + cx + d,
and u+3 and v-2 are roots of another polynomial s(x) = x^3 + cx + d + 153,
prove that the only possible value for d is 0.
-/
theorem polynomial_roots_problem (u v c d : ℝ) : 
  (u^3 + c*u + d = 0) →
  (v^3 + c*v + d = 0) →
  ((u+3)^3 + c*(u+3) + d + 153 = 0) →
  ((v-2)^3 + c*(v-2) + d + 153 = 0) →
  d = 0 := by
  sorry


end polynomial_roots_problem_l3590_359061


namespace cube_sum_inequality_cube_sum_equality_l3590_359048

theorem cube_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 / (y*z) + y^3 / (z*x) + z^3 / (x*y) ≥ x + y + z :=
sorry

theorem cube_sum_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 / (y*z) + y^3 / (z*x) + z^3 / (x*y) = x + y + z ↔ x = y ∧ y = z :=
sorry

end cube_sum_inequality_cube_sum_equality_l3590_359048


namespace smallest_among_four_rationals_l3590_359056

theorem smallest_among_four_rationals :
  let a : ℚ := -2/3
  let b : ℚ := -1
  let c : ℚ := 0
  let d : ℚ := 1
  b < a ∧ b < c ∧ b < d := by sorry

end smallest_among_four_rationals_l3590_359056


namespace probability_through_x_l3590_359041

structure DirectedGraph where
  vertices : Finset Char
  edges : Finset (Char × Char)

def paths (g : DirectedGraph) (start finish : Char) : Nat :=
  sorry

theorem probability_through_x (g : DirectedGraph) :
  g.vertices = {'A', 'B', 'X', 'Y'} →
  paths g 'A' 'X' = 2 →
  paths g 'X' 'B' = 1 →
  paths g 'X' 'Y' = 1 →
  paths g 'Y' 'B' = 3 →
  paths g 'A' 'Y' = 3 →
  (paths g 'A' 'X' * paths g 'X' 'B' + paths g 'A' 'X' * paths g 'X' 'Y' * paths g 'Y' 'B') / 
  (paths g 'A' 'X' * paths g 'X' 'B' + paths g 'A' 'X' * paths g 'X' 'Y' * paths g 'Y' 'B' + paths g 'A' 'Y' * paths g 'Y' 'B') = 8 / 11 :=
sorry

end probability_through_x_l3590_359041


namespace interest_problem_l3590_359062

/-- Given a sum P at simple interest rate R for 3 years, if increasing the rate by 8%
    results in Rs. 120 more interest, then P = 500. -/
theorem interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  (P * (R + 8) * 3) / 100 = (P * R * 3) / 100 + 120 →
  P = 500 := by
  sorry

end interest_problem_l3590_359062


namespace vector_equations_l3590_359044

def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (-3, -4)

def a : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def b : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def c : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)

theorem vector_equations :
  (3 * a.1 + b.1, 3 * a.2 + b.2) = (9, -18) ∧
  a = (-b.1 - c.1, -b.2 - c.2) :=
sorry

end vector_equations_l3590_359044


namespace distribute_six_interns_three_schools_l3590_359089

/-- The number of ways to distribute n interns among k schools, where each intern is assigned to exactly one school and each school receives at least one intern. -/
def distribute_interns (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 540 ways to distribute 6 interns among 3 schools under the given conditions. -/
theorem distribute_six_interns_three_schools : distribute_interns 6 3 = 540 := by sorry

end distribute_six_interns_three_schools_l3590_359089


namespace four_diamonds_balance_four_bullets_l3590_359090

/-- Represents the balance of symbols in a weighing system -/
structure SymbolBalance where
  delta : ℚ      -- Represents Δ
  diamond : ℚ    -- Represents ♢
  bullet : ℚ     -- Represents •

/-- The balance equations given in the problem -/
def balance_equations (sb : SymbolBalance) : Prop :=
  (2 * sb.delta + 3 * sb.diamond = 12 * sb.bullet) ∧
  (sb.delta = 3 * sb.diamond + 2 * sb.bullet)

/-- The theorem to be proved -/
theorem four_diamonds_balance_four_bullets (sb : SymbolBalance) :
  balance_equations sb → 4 * sb.diamond = 4 * sb.bullet :=
by sorry

end four_diamonds_balance_four_bullets_l3590_359090


namespace triangle_properties_l3590_359002

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, angle A is π/3 and the area is 4√3. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  Real.sin C / (Real.sin A + Real.sin B) + b / (a + c) = 1 →
  |b - a| = 4 →
  Real.cos B + Real.cos C = 1 →
  A = π / 3 ∧ a * c * (Real.sin B) / 2 = 4 * Real.sqrt 3 := by
  sorry


end triangle_properties_l3590_359002


namespace max_non_managers_l3590_359065

theorem max_non_managers (num_managers : ℕ) (ratio_managers : ℚ) (ratio_non_managers : ℚ) :
  num_managers = 8 →
  ratio_managers / ratio_non_managers > 7 / 24 →
  ∃ (max_non_managers : ℕ),
    (↑num_managers : ℚ) / (↑max_non_managers : ℚ) > ratio_managers / ratio_non_managers ∧
    ∀ (n : ℕ), n > max_non_managers →
      (↑num_managers : ℚ) / (↑n : ℚ) ≤ ratio_managers / ratio_non_managers →
      max_non_managers = 27 :=
by sorry

end max_non_managers_l3590_359065


namespace factor_polynomial_l3590_359067

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 300 * x^13 = 75 * x^7 * (1 - 4 * x^6) := by
  sorry

end factor_polynomial_l3590_359067


namespace unique_complex_pair_l3590_359069

theorem unique_complex_pair : 
  ∃! (a b : ℂ), (a^4 * b^3 = 1) ∧ (a^6 * b^7 = 1) :=
by sorry

end unique_complex_pair_l3590_359069


namespace line_passes_through_fixed_point_max_area_difference_l3590_359055

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the vertices of the ellipse
def A : ℝ × ℝ := (-5, 0)
def B : ℝ × ℝ := (5, 0)

-- Define a line
def line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the slope ratio condition
def slope_ratio (k₁ k₂ : ℝ) : Prop := k₁ / k₂ = 1 / 9

-- Define the intersection points
def intersection_points (k m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line k m x₁ y₁ ∧ line k m x₂ y₂

-- Define the triangles' areas
def area_diff (S₁ S₂ : ℝ) : Prop := S₁ - S₂ ≤ 15

-- Theorem 1: The line passes through (4, 0)
theorem line_passes_through_fixed_point (k m : ℝ) :
  intersection_points k m →
  (∃ k₁ k₂ : ℝ, slope_ratio k₁ k₂) →
  line k m 4 0 :=
sorry

-- Theorem 2: Maximum value of S₁ - S₂
theorem max_area_difference :
  ∀ S₁ S₂ : ℝ,
  (∃ k m : ℝ, intersection_points k m ∧ 
   (∃ k₁ k₂ : ℝ, slope_ratio k₁ k₂)) →
  area_diff S₁ S₂ ∧ 
  (∀ S₁' S₂' : ℝ, area_diff S₁' S₂' → S₁ - S₂ ≥ S₁' - S₂') :=
sorry

end line_passes_through_fixed_point_max_area_difference_l3590_359055


namespace mixed_oil_rate_l3590_359073

/-- Calculates the rate of mixed oil per litre given volumes and prices of different oils. -/
theorem mixed_oil_rate (v1 v2 v3 v4 : ℚ) (p1 p2 p3 p4 : ℚ) :
  v1 = 10 ∧ v2 = 5 ∧ v3 = 3 ∧ v4 = 2 ∧
  p1 = 50 ∧ p2 = 66 ∧ p3 = 75 ∧ p4 = 85 →
  (v1 * p1 + v2 * p2 + v3 * p3 + v4 * p4) / (v1 + v2 + v3 + v4) = 61.25 := by
  sorry

#eval (10 * 50 + 5 * 66 + 3 * 75 + 2 * 85) / (10 + 5 + 3 + 2)

end mixed_oil_rate_l3590_359073


namespace abs_negative_2035_l3590_359097

theorem abs_negative_2035 : |(-2035 : ℤ)| = 2035 := by sorry

end abs_negative_2035_l3590_359097


namespace school_student_count_l3590_359035

/-- The number of teachers in the school -/
def num_teachers : ℕ := 9

/-- The number of additional students needed for equal distribution -/
def additional_students : ℕ := 4

/-- The current number of students in the school -/
def current_students : ℕ := 23

/-- Theorem stating that the current number of students is correct -/
theorem school_student_count :
  ∃ (k : ℕ), (current_students + additional_students) = num_teachers * k ∧
             ∀ (m : ℕ), m < current_students →
               ¬(∃ (j : ℕ), (m + additional_students) = num_teachers * j) :=
by sorry

end school_student_count_l3590_359035


namespace min_values_xy_l3590_359079

/-- Given positive real numbers x and y satisfying 2xy = x + 4y + a,
    prove the minimum values for xy and x + y + 2/x + 1/(2y) for different values of a. -/
theorem min_values_xy (x y a : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x * y = x + 4 * y + a) :
  (a = 16 → x * y ≥ 16) ∧
  (a = 0 → x + y + 2 / x + 1 / (2 * y) ≥ 11 / 2) := by
  sorry

end min_values_xy_l3590_359079


namespace problem_solution_l3590_359084

theorem problem_solution (a : ℝ) : 3 ∈ ({a + 3, 2 * a + 1, a^2 + a + 1} : Set ℝ) → a = -2 := by
  sorry

end problem_solution_l3590_359084


namespace ellipse_major_axis_length_l3590_359029

theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / 9 + y^2 / 4 = 1}
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ellipse = {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1} ∧
    2 * a = 6 :=
by sorry

end ellipse_major_axis_length_l3590_359029


namespace pet_shop_birds_l3590_359043

theorem pet_shop_birds (total : ℕ) (kittens : ℕ) (hamsters : ℕ) (birds : ℕ) : 
  total = 77 → kittens = 32 → hamsters = 15 → birds = total - kittens - hamsters → birds = 30 := by
  sorry

end pet_shop_birds_l3590_359043


namespace order_of_numbers_l3590_359077

theorem order_of_numbers (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) :
  a > -b ∧ -b > b ∧ b > -a := by sorry

end order_of_numbers_l3590_359077


namespace sqrt_31_plus_3_tan_56_approx_7_l3590_359024

/-- Prove that the absolute difference between √31 + 3tan(56°) and 7.00 is less than 0.005 -/
theorem sqrt_31_plus_3_tan_56_approx_7 :
  |Real.sqrt 31 + 3 * Real.tan (56 * π / 180) - 7| < 0.005 := by
  sorry

end sqrt_31_plus_3_tan_56_approx_7_l3590_359024


namespace stock_price_return_l3590_359060

theorem stock_price_return (initial_price : ℝ) (h : initial_price > 0) : 
  let increased_price := initial_price * 1.3
  let decrease_percentage := (1 - 1 / 1.3) * 100
  increased_price * (1 - decrease_percentage / 100) = initial_price :=
by sorry

end stock_price_return_l3590_359060


namespace correct_calculation_l3590_359051

theorem correct_calculation (x : ℤ) : 66 + x = 93 → (66 - x) + 21 = 60 := by
  sorry

end correct_calculation_l3590_359051


namespace inverse_f_sum_l3590_359006

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3*x - x^2

theorem inverse_f_sum : ∃ y₁ y₂ y₃ : ℝ, 
  f y₁ = -4 ∧ f y₂ = 1 ∧ f y₃ = 4 ∧ y₁ + y₂ + y₃ = 5 :=
sorry

end inverse_f_sum_l3590_359006


namespace three_distinct_volumes_l3590_359042

/-- A triangular pyramid with specific face conditions -/
structure TriangularPyramid where
  /-- Two lateral faces are isosceles right triangles -/
  has_two_isosceles_right_faces : Bool
  /-- One face is an equilateral triangle with side length 1 -/
  has_equilateral_face : Bool
  /-- The side length of the equilateral face -/
  equilateral_side_length : ℝ

/-- The volume of a triangular pyramid -/
def volume (pyramid : TriangularPyramid) : ℝ := sorry

/-- The set of all possible volumes for triangular pyramids satisfying the conditions -/
def possible_volumes : Set ℝ := sorry

/-- Theorem stating that there are exactly three distinct volumes -/
theorem three_distinct_volumes :
  ∃ (v₁ v₂ v₃ : ℝ), v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃ ∧
  possible_volumes = {v₁, v₂, v₃} :=
sorry

end three_distinct_volumes_l3590_359042


namespace fermat_number_prime_factor_l3590_359031

def F (n : ℕ) : ℕ := 2^(2^n) + 1

theorem fermat_number_prime_factor (n : ℕ) (h : n ≥ 3) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ F n ∧ p > 2^(n+2) * (n+1) := by
  sorry

end fermat_number_prime_factor_l3590_359031
