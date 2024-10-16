import Mathlib

namespace NUMINAMATH_CALUDE_labor_cost_per_hour_l3154_315446

theorem labor_cost_per_hour 
  (total_repair_cost : ℝ)
  (part_cost : ℝ)
  (labor_hours : ℝ)
  (h1 : total_repair_cost = 2400)
  (h2 : part_cost = 1200)
  (h3 : labor_hours = 16) :
  (total_repair_cost - part_cost) / labor_hours = 75 := by
sorry

end NUMINAMATH_CALUDE_labor_cost_per_hour_l3154_315446


namespace NUMINAMATH_CALUDE_new_girl_weight_l3154_315498

theorem new_girl_weight (initial_total_weight : ℝ) : 
  let initial_average := initial_total_weight / 10
  let new_average := initial_average + 5
  let new_total_weight := new_average * 10
  new_total_weight = initial_total_weight - 50 + 100 := by sorry

end NUMINAMATH_CALUDE_new_girl_weight_l3154_315498


namespace NUMINAMATH_CALUDE_max_a_for_real_roots_l3154_315485

theorem max_a_for_real_roots : ∃ (a_max : ℤ), 
  (∀ a : ℤ, (∃ x : ℝ, (a + 1 : ℝ) * x^2 - 2*x + 3 = 0) → a ≤ a_max) ∧ 
  (∃ x : ℝ, (a_max + 1 : ℝ) * x^2 - 2*x + 3 = 0) ∧ 
  a_max = -2 :=
sorry

end NUMINAMATH_CALUDE_max_a_for_real_roots_l3154_315485


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3154_315419

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt x + Real.sqrt (x + 6) = 12 → x = 529 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3154_315419


namespace NUMINAMATH_CALUDE_gcd_of_638_522_406_l3154_315438

theorem gcd_of_638_522_406 : Nat.gcd 638 (Nat.gcd 522 406) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_638_522_406_l3154_315438


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3154_315417

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1 = (-1 + Real.sqrt 17) / 2 ∧ 
                x2 = (-1 - Real.sqrt 17) / 2 ∧ 
                x1^2 + x1 - 4 = 0 ∧ 
                x2^2 + x2 - 4 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 1 ∧ 
                x2 = 2 ∧ 
                (2*x1 + 1)^2 + 15 = 8*(2*x1 + 1) ∧ 
                (2*x2 + 1)^2 + 15 = 8*(2*x2 + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3154_315417


namespace NUMINAMATH_CALUDE_total_marks_is_530_l3154_315410

/-- Calculates the total marks scored by Amaya in all subjects given the following conditions:
  * Amaya scored 20 marks fewer in Maths than in Arts
  * She got 10 marks more in Social Studies than in Music
  * She scored 70 in Music
  * She scored 1/10 less in Maths than in Arts
-/
def totalMarks (musicScore : ℕ) : ℕ :=
  let socialStudiesScore := musicScore + 10
  let artsScore := 200
  let mathsScore := artsScore - 20
  musicScore + socialStudiesScore + artsScore + mathsScore

theorem total_marks_is_530 : totalMarks 70 = 530 := by
  sorry

end NUMINAMATH_CALUDE_total_marks_is_530_l3154_315410


namespace NUMINAMATH_CALUDE_complement_of_S_in_U_l3154_315487

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the set S
def S : Set Nat := {1, 2, 3, 4}

-- Theorem statement
theorem complement_of_S_in_U : 
  (U \ S) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_S_in_U_l3154_315487


namespace NUMINAMATH_CALUDE_stratified_sample_B_size_l3154_315465

/-- Represents the number of individuals in each level -/
structure PopulationLevels where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Calculates the total population -/
def total_population (p : PopulationLevels) : ℕ := p.A + p.B + p.C

/-- Represents a stratified sample -/
structure StratifiedSample where
  total_sample : ℕ
  population : PopulationLevels

/-- Calculates the number of individuals to be sampled from a specific level -/
def sample_size_for_level (s : StratifiedSample) (level_size : ℕ) : ℕ :=
  (s.total_sample * level_size) / (total_population s.population)

theorem stratified_sample_B_size 
  (sample : StratifiedSample) 
  (h1 : sample.population.A = 5 * n)
  (h2 : sample.population.B = 3 * n)
  (h3 : sample.population.C = 2 * n)
  (h4 : sample.total_sample = 150)
  (n : ℕ) :
  sample_size_for_level sample sample.population.B = 45 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_B_size_l3154_315465


namespace NUMINAMATH_CALUDE_yanna_change_l3154_315441

/-- The change Yanna received after buying shirts and sandals -/
def change_received (shirt_price shirt_quantity sandal_price sandal_quantity payment : ℕ) : ℕ :=
  payment - (shirt_price * shirt_quantity + sandal_price * sandal_quantity)

/-- Theorem stating that Yanna received $41 in change -/
theorem yanna_change :
  change_received 5 10 3 3 100 = 41 := by
  sorry

end NUMINAMATH_CALUDE_yanna_change_l3154_315441


namespace NUMINAMATH_CALUDE_sin_plus_cos_shift_l3154_315473

theorem sin_plus_cos_shift (x : ℝ) :
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * x + π / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_shift_l3154_315473


namespace NUMINAMATH_CALUDE_distance_between_vertices_l3154_315484

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 2| = 5

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 2.5)
def vertex2 : ℝ × ℝ := (0, -1.5)

-- Theorem statement
theorem distance_between_vertices : 
  ∀ (v1 v2 : ℝ × ℝ), 
  (∀ x y, parabola_equation x y → (x = v1.1 ∧ y = v1.2) ∨ (x = v2.1 ∧ y = v2.2)) →
  v1 = vertex1 ∧ v2 = vertex2 →
  Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l3154_315484


namespace NUMINAMATH_CALUDE_vector_collinearity_implies_x_value_l3154_315437

theorem vector_collinearity_implies_x_value (x : ℝ) 
  (hx : x > 0) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (8, x/2))
  (hb : b = (x, 1))
  (hcollinear : ∃ (k : ℝ), k ≠ 0 ∧ (a - 2 • b) = k • (2 • a + b)) :
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_implies_x_value_l3154_315437


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l3154_315482

/-- The area of an equilateral triangle with base 10 and height 5√3 is 25√3 -/
theorem equilateral_triangle_area : 
  ∀ (base height area : ℝ),
  base = 10 →
  height = 5 * Real.sqrt 3 →
  area = (1 / 2) * base * height →
  area = 25 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l3154_315482


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l3154_315424

/-- Given a geometric sequence with common ratio not equal to -1,
    prove that if S_12 = 7S_4, then S_8 / S_4 = 3 -/
theorem geometric_sequence_sum_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (hq : q ≠ -1) 
  (h_geom : ∀ n, a (n + 1) = q * a n) 
  (S : ℕ → ℝ) 
  (h_sum : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) 
  (h_ratio : S 12 = 7 * S 4) : 
  S 8 / S 4 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l3154_315424


namespace NUMINAMATH_CALUDE_square_d_perimeter_l3154_315452

def square_perimeter (side_length : ℝ) : ℝ := 4 * side_length

def square_area (side_length : ℝ) : ℝ := side_length ^ 2

theorem square_d_perimeter (perimeter_c : ℝ) (h1 : perimeter_c = 32) :
  let side_c := perimeter_c / 4
  let area_c := square_area side_c
  let area_d := area_c / 3
  let side_d := Real.sqrt area_d
  square_perimeter side_d = (32 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_square_d_perimeter_l3154_315452


namespace NUMINAMATH_CALUDE_zack_andrew_same_team_probability_l3154_315404

-- Define the total number of players
def total_players : ℕ := 27

-- Define the number of teams
def num_teams : ℕ := 3

-- Define the number of players per team
def players_per_team : ℕ := 9

-- Define the set of players
def Player : Type := Fin total_players

-- Define the function that assigns players to teams
def team_assignment : Player → Fin num_teams := sorry

-- Define Zack, Mihir, and Andrew as specific players
def Zack : Player := sorry
def Mihir : Player := sorry
def Andrew : Player := sorry

-- State that Zack and Mihir are on different teams
axiom zack_mihir_different : team_assignment Zack ≠ team_assignment Mihir

-- State that Mihir and Andrew are on different teams
axiom mihir_andrew_different : team_assignment Mihir ≠ team_assignment Andrew

-- Define the probability function
def probability_same_team (p1 p2 : Player) : ℚ := sorry

-- State the theorem to be proved
theorem zack_andrew_same_team_probability :
  probability_same_team Zack Andrew = 8 / 17 := sorry

end NUMINAMATH_CALUDE_zack_andrew_same_team_probability_l3154_315404


namespace NUMINAMATH_CALUDE_stock_transaction_profit_l3154_315458

/-- Represents a stock transaction and calculates the profit -/
def stock_transaction (initial_shares : ℕ) (initial_price : ℚ) (sold_shares : ℕ) (selling_price : ℚ) : ℚ :=
  let initial_cost := initial_shares * initial_price
  let sale_revenue := sold_shares * selling_price
  let remaining_shares := initial_shares - sold_shares
  let final_value := sale_revenue + (remaining_shares * (2 * initial_price))
  final_value - initial_cost

/-- Proves that the profit from the given stock transaction is $40 -/
theorem stock_transaction_profit :
  stock_transaction 20 3 10 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_stock_transaction_profit_l3154_315458


namespace NUMINAMATH_CALUDE_ball_distribution_ratio_l3154_315408

/-- The number of ways to distribute n identical objects into k distinct bins --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n identical objects into k distinct bins,
    where each bin receives a specified number of objects --/
def distributeSpecific (n k : ℕ) (bins : Fin k → ℕ) : ℕ := sorry

theorem ball_distribution_ratio : 
  let total_balls : ℕ := 15
  let num_bins : ℕ := 4
  let pattern1 : Fin num_bins → ℕ := ![3, 6, 3, 3]
  let pattern2 : Fin num_bins → ℕ := ![3, 2, 3, 7]
  
  (distributeSpecific total_balls num_bins pattern1) / 
  (distributeSpecific total_balls num_bins pattern2) = 560 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_ratio_l3154_315408


namespace NUMINAMATH_CALUDE_total_books_calculation_l3154_315489

/-- The total number of books assigned to Mcgregor and Floyd -/
def total_books : ℕ := 89

/-- The number of books Mcgregor finished -/
def mcgregor_books : ℕ := 34

/-- The number of books Floyd finished -/
def floyd_books : ℕ := 32

/-- The number of books remaining to be read -/
def remaining_books : ℕ := 23

/-- Theorem stating that the total number of books is the sum of the books finished by Mcgregor and Floyd, plus the remaining books -/
theorem total_books_calculation : 
  total_books = mcgregor_books + floyd_books + remaining_books :=
by sorry

end NUMINAMATH_CALUDE_total_books_calculation_l3154_315489


namespace NUMINAMATH_CALUDE_scientific_notation_of_149000000_l3154_315478

theorem scientific_notation_of_149000000 :
  149000000 = 1.49 * (10 : ℝ)^8 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_149000000_l3154_315478


namespace NUMINAMATH_CALUDE_triangle_area_l3154_315499

-- Define the curve
def f (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define the x-intercepts
def x_intercept_1 : ℝ := -3
def x_intercept_2 : ℝ := 4

-- Define the y-intercept
def y_intercept : ℝ := f 0

-- Theorem statement
theorem triangle_area : 
  let base := x_intercept_2 - x_intercept_1
  let height := y_intercept
  (1 / 2 : ℝ) * base * height = 168 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3154_315499


namespace NUMINAMATH_CALUDE_factorization_x12_minus_729_l3154_315468

theorem factorization_x12_minus_729 (x : ℝ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3*x^2 + 9) * (x^2 - 3) * (x^4 + 3*x^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x12_minus_729_l3154_315468


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l3154_315414

/-- A quadratic function is a function of the form f(x) = ax^2 + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique 
  (f : ℝ → ℝ) 
  (h_quad : IsQuadratic f) 
  (h_f0 : f 0 = 1) 
  (h_fx1 : ∀ x, f (x + 1) = f x + x + 1) : 
  ∀ x, f x = (1/2) * x^2 + (1/2) * x + 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l3154_315414


namespace NUMINAMATH_CALUDE_modulus_of_3_minus_2i_l3154_315471

theorem modulus_of_3_minus_2i : Complex.abs (3 - 2*Complex.I) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_3_minus_2i_l3154_315471


namespace NUMINAMATH_CALUDE_parentheses_placement_count_l3154_315470

/-- A sequence of prime numbers -/
def primeSequence : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

/-- The operation of placing parentheses in the expression -/
def parenthesesPlacement (seq : List Nat) : Nat :=
  2^(seq.length - 2)

/-- Theorem stating the number of different values obtained by placing parentheses -/
theorem parentheses_placement_count :
  parenthesesPlacement primeSequence = 256 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_placement_count_l3154_315470


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_l3154_315403

/-- A number is 7-heavy if its remainder when divided by 7 is greater than 4 -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- Three-digit numbers are whole numbers from 100 to 999 -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_heavy : 
  ∀ n : ℕ, is_three_digit n → is_7_heavy n → 104 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_l3154_315403


namespace NUMINAMATH_CALUDE_andy_tomato_plants_l3154_315421

theorem andy_tomato_plants :
  ∀ (P : ℕ),
  (∃ (total_tomatoes dried_tomatoes sauce_tomatoes remaining_tomatoes : ℕ),
    total_tomatoes = 7 * P ∧
    dried_tomatoes = total_tomatoes / 2 ∧
    sauce_tomatoes = (total_tomatoes - dried_tomatoes) / 3 ∧
    remaining_tomatoes = total_tomatoes - dried_tomatoes - sauce_tomatoes ∧
    remaining_tomatoes = 42) →
  P = 18 :=
by sorry

end NUMINAMATH_CALUDE_andy_tomato_plants_l3154_315421


namespace NUMINAMATH_CALUDE_total_pages_theorem_l3154_315407

/-- The number of pages Jairus read -/
def jairus_pages : ℕ := 20

/-- The number of pages Arniel read -/
def arniel_pages : ℕ := 2 * jairus_pages + 2

/-- The total number of pages read by Jairus and Arniel -/
def total_pages : ℕ := jairus_pages + arniel_pages

theorem total_pages_theorem : total_pages = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_theorem_l3154_315407


namespace NUMINAMATH_CALUDE_mrs_hilt_apples_l3154_315444

/-- Calculates the total number of apples eaten given a rate and time period. -/
def applesEaten (rate : ℕ) (hours : ℕ) : ℕ := rate * hours

/-- Theorem stating that eating 5 apples per hour for 3 hours results in 15 apples eaten. -/
theorem mrs_hilt_apples : applesEaten 5 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_apples_l3154_315444


namespace NUMINAMATH_CALUDE_centroid_quadrilateral_area_l3154_315467

-- Define the square ABCD
structure Square :=
  (sideLength : ℝ)

-- Define a point P inside the square
structure PointInSquare :=
  (distanceAP : ℝ)
  (distanceBP : ℝ)

-- Define the quadrilateral formed by centroids
structure CentroidQuadrilateral :=
  (diagonalLength : ℝ)

-- Define the theorem
theorem centroid_quadrilateral_area
  (s : Square)
  (p : PointInSquare)
  (q : CentroidQuadrilateral)
  (h1 : s.sideLength = 30)
  (h2 : p.distanceAP = 12)
  (h3 : p.distanceBP = 26)
  (h4 : q.diagonalLength = 20) :
  q.diagonalLength * q.diagonalLength / 2 = 200 :=
sorry

end NUMINAMATH_CALUDE_centroid_quadrilateral_area_l3154_315467


namespace NUMINAMATH_CALUDE_angela_has_eight_more_l3154_315486

/-- The number of marbles each person has -/
structure MarbleCount where
  albert : ℕ
  angela : ℕ
  allison : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : MarbleCount) : Prop :=
  m.albert = 3 * m.angela ∧
  m.angela > m.allison ∧
  m.allison = 28 ∧
  m.albert + m.allison = 136

/-- The theorem stating that Angela has 8 more marbles than Allison -/
theorem angela_has_eight_more (m : MarbleCount) 
  (h : marble_problem m) : m.angela - m.allison = 8 := by
  sorry

end NUMINAMATH_CALUDE_angela_has_eight_more_l3154_315486


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_shorter_base_l3154_315405

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- Length of the shorter base -/
  shorterBase : ℝ
  /-- Length of the line joining the midpoints of the diagonals -/
  midpointLine : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The midpoint line length is half the difference of the bases -/
  midpointProperty : midpointLine = (longerBase - shorterBase) / 2

/-- Theorem: In an isosceles trapezoid where the line joining the midpoints of the diagonals
    has length 4 and the longer base is 100, the shorter base has length 92 -/
theorem isosceles_trapezoid_shorter_base
  (t : IsoscelesTrapezoid)
  (h1 : t.longerBase = 100)
  (h2 : t.midpointLine = 4) :
  t.shorterBase = 92 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_shorter_base_l3154_315405


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_l3154_315454

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_l3154_315454


namespace NUMINAMATH_CALUDE_p_bounds_l3154_315411

/-- Represents the minimum number of reconstructions needed to transform
    one triangulation into another for a convex n-gon. -/
def p (n : ℕ) : ℕ := sorry

/-- Theorem stating the bounds on p(n) for convex n-gons. -/
theorem p_bounds (n : ℕ) : 
  n ≥ 3 → 
  p n ≥ n - 3 ∧ 
  p n ≤ 2*n - 7 ∧ 
  (n ≥ 13 → p n ≤ 2*n - 10) := by sorry


end NUMINAMATH_CALUDE_p_bounds_l3154_315411


namespace NUMINAMATH_CALUDE_equation_solution_l3154_315472

theorem equation_solution (x : ℝ) (h1 : 0 < x) (h2 : x < 12) (h3 : x ≠ 1) :
  (1 + 2 * Real.log 2 / Real.log 9) / (Real.log x / Real.log 9) - 1 = 
  2 * (Real.log 3 / Real.log x) * (Real.log (12 - x) / Real.log 9) → x = 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3154_315472


namespace NUMINAMATH_CALUDE_f_continuous_l3154_315442

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + 3*x + 5

-- State the theorem
theorem f_continuous : Continuous f := by sorry

end NUMINAMATH_CALUDE_f_continuous_l3154_315442


namespace NUMINAMATH_CALUDE_percent_relation_l3154_315496

theorem percent_relation (a b c : ℝ) (h1 : c = 0.25 * a) (h2 : c = 0.50 * b) : b = 0.50 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3154_315496


namespace NUMINAMATH_CALUDE_solve_candy_problem_l3154_315457

def candy_problem (initial_candies : ℕ) (friend_multiplier : ℕ) (friend_eaten : ℕ) : Prop :=
  let friend_brought := initial_candies * friend_multiplier
  let total_candies := initial_candies + friend_brought
  let each_share := total_candies / 2
  let friend_final := each_share - friend_eaten
  friend_final = 65

theorem solve_candy_problem :
  candy_problem 50 2 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l3154_315457


namespace NUMINAMATH_CALUDE_jane_started_at_18_l3154_315416

/-- Represents Jane's babysitting career --/
structure BabysittingCareer where
  current_age : ℕ
  years_since_stopped : ℕ
  oldest_babysat_current_age : ℕ
  start_age : ℕ

/-- Checks if the babysitting career satisfies all conditions --/
def is_valid_career (career : BabysittingCareer) : Prop :=
  career.current_age = 34 ∧
  career.years_since_stopped = 12 ∧
  career.oldest_babysat_current_age = 25 ∧
  career.start_age ≤ career.current_age - career.years_since_stopped ∧
  ∀ (child_age : ℕ), child_age ≤ career.oldest_babysat_current_age →
    2 * (child_age - career.years_since_stopped) ≤ career.current_age - career.years_since_stopped

theorem jane_started_at_18 :
  ∃ (career : BabysittingCareer), is_valid_career career ∧ career.start_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_jane_started_at_18_l3154_315416


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_5_minus_5_4_l3154_315401

theorem least_prime_factor_of_5_5_minus_5_4 :
  Nat.minFac (5^5 - 5^4) = 2 := by
sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_5_minus_5_4_l3154_315401


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l3154_315450

theorem quadratic_function_max_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, -x^2 + 2*a*x + 1 - a ≤ 2) ∧ 
  (∃ x ∈ Set.Icc 0 1, -x^2 + 2*a*x + 1 - a = 2) → 
  a = -1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l3154_315450


namespace NUMINAMATH_CALUDE_custom_op_three_six_l3154_315436

/-- Custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a.val ^ 2 * b.val) / (a.val + b.val)

/-- Theorem stating that 3 @ 6 = 6 -/
theorem custom_op_three_six :
  custom_op 3 6 = 6 := by sorry

end NUMINAMATH_CALUDE_custom_op_three_six_l3154_315436


namespace NUMINAMATH_CALUDE_blanket_collection_proof_l3154_315443

/-- Calculates the total number of blankets collected over three days -/
def totalBlankets (teamSize : ℕ) (firstDayPerPerson : ℕ) (secondDayMultiplier : ℕ) (thirdDayTotal : ℕ) : ℕ :=
  let firstDay := teamSize * firstDayPerPerson
  let secondDay := firstDay * secondDayMultiplier
  firstDay + secondDay + thirdDayTotal

/-- Proves that the total number of blankets collected is 142 given the specific conditions -/
theorem blanket_collection_proof :
  totalBlankets 15 2 3 22 = 142 := by
  sorry

end NUMINAMATH_CALUDE_blanket_collection_proof_l3154_315443


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_l3154_315481

/-- Given a survey of students' pasta preferences, prove the ratio of spaghetti to tortellini preference -/
theorem pasta_preference_ratio 
  (total_students : ℕ) 
  (spaghetti_preference : ℕ) 
  (tortellini_preference : ℕ) 
  (h1 : total_students = 850)
  (h2 : spaghetti_preference = 300)
  (h3 : tortellini_preference = 200) :
  (spaghetti_preference : ℚ) / tortellini_preference = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_l3154_315481


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3154_315469

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_of_A_in_U : 
  Set.compl A = Set.Icc (-1 : ℝ) (3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3154_315469


namespace NUMINAMATH_CALUDE_track_length_l3154_315418

/-- Represents a circular track with two runners -/
structure CircularTrack where
  length : ℝ
  runner1_speed : ℝ
  runner2_speed : ℝ

/-- Theorem stating the length of the track given the conditions -/
theorem track_length (track : CircularTrack) 
  (h1 : track.runner1_speed > 0)
  (h2 : track.runner2_speed > 0)
  (h3 : track.length / 2 = 100)
  (h4 : 200 = track.runner2_speed * (track.length / (track.runner1_speed + track.runner2_speed)))
  : track.length = 200 := by
  sorry

#check track_length

end NUMINAMATH_CALUDE_track_length_l3154_315418


namespace NUMINAMATH_CALUDE_ellipse_foci_condition_l3154_315451

theorem ellipse_foci_condition (α : Real) (h1 : 0 < α) (h2 : α < π / 2) :
  (∀ x y : Real, x^2 / Real.sin α + y^2 / Real.cos α = 1 →
    ∃ c : Real, c > 0 ∧ 
      ∀ x₀ y₀ : Real, (x₀ + c)^2 + y₀^2 + (x₀ - c)^2 + y₀^2 = 
        2 * ((x^2 / Real.sin α + y^2 / Real.cos α) * (1 / Real.sin α + 1 / Real.cos α))) →
  π / 4 < α ∧ α < π / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_condition_l3154_315451


namespace NUMINAMATH_CALUDE_inequality_proof_l3154_315460

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3154_315460


namespace NUMINAMATH_CALUDE_smallest_valid_sequence_length_l3154_315494

def S : Finset Nat := {1, 2, 3, 4}

def IsValidSequence (seq : List Nat) : Prop :=
  ∀ B : Finset Nat, B ⊆ S → B.Nonempty → 
    ∃ i : Nat, i + B.card ≤ seq.length ∧ 
      (seq.take (i + B.card)).drop i = B.toList

theorem smallest_valid_sequence_length : 
  (∃ seq : List Nat, IsValidSequence seq ∧ seq.length = 8) ∧
  (∀ seq : List Nat, IsValidSequence seq → seq.length ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_sequence_length_l3154_315494


namespace NUMINAMATH_CALUDE_fishing_problem_l3154_315428

theorem fishing_problem (a b c d : ℕ) : 
  a + b + c + d = 25 →
  a > b ∧ b > c ∧ c > d →
  a = b + c →
  b = c + d →
  (a = 11 ∧ b = 7 ∧ c = 4 ∧ d = 3) := by
sorry

end NUMINAMATH_CALUDE_fishing_problem_l3154_315428


namespace NUMINAMATH_CALUDE_calcium_atomic_weight_l3154_315464

/-- The atomic weight of Oxygen -/
def atomic_weight_O : ℝ := 16

/-- The molecular weight of Calcium Oxide (CaO) -/
def molecular_weight_CaO : ℝ := 56

/-- The atomic weight of Calcium -/
def atomic_weight_Ca : ℝ := molecular_weight_CaO - atomic_weight_O

/-- Theorem stating that the atomic weight of Calcium is 40 -/
theorem calcium_atomic_weight :
  atomic_weight_Ca = 40 := by sorry

end NUMINAMATH_CALUDE_calcium_atomic_weight_l3154_315464


namespace NUMINAMATH_CALUDE_book_original_price_l3154_315435

/-- Given a book sold for $78 with a 30% profit, prove that the original price was $60 -/
theorem book_original_price (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 78 → profit_percentage = 30 → 
  ∃ (original_price : ℝ), 
    original_price = 60 ∧ 
    selling_price = original_price * (1 + profit_percentage / 100) := by
  sorry

#check book_original_price

end NUMINAMATH_CALUDE_book_original_price_l3154_315435


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3154_315420

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 5 = 14)
  (h_prod : a 2 * a 6 = 33) :
  a 1 * a 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3154_315420


namespace NUMINAMATH_CALUDE_quadratic_order_l3154_315461

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

-- Define the points
def y1 (c : ℝ) : ℝ := f c (-1)
def y2 (c : ℝ) : ℝ := f c 2
def y3 (c : ℝ) : ℝ := f c 5

-- Theorem statement
theorem quadratic_order (c : ℝ) : y1 c > y3 c ∧ y3 c > y2 c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_order_l3154_315461


namespace NUMINAMATH_CALUDE_combined_molecular_weight_mixture_l3154_315483

-- Define atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define molecular weights
def molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O
def molecular_weight_CO2 : ℝ := atomic_weight_C + 2 * atomic_weight_O
def molecular_weight_HNO3 : ℝ := atomic_weight_H + atomic_weight_N + 3 * atomic_weight_O

-- Define the mixture composition
def moles_CaO : ℝ := 5
def moles_CO2 : ℝ := 3
def moles_HNO3 : ℝ := 2

-- Theorem statement
theorem combined_molecular_weight_mixture :
  moles_CaO * molecular_weight_CaO +
  moles_CO2 * molecular_weight_CO2 +
  moles_HNO3 * molecular_weight_HNO3 = 538.45 := by
  sorry

end NUMINAMATH_CALUDE_combined_molecular_weight_mixture_l3154_315483


namespace NUMINAMATH_CALUDE_f_min_at_five_thirds_l3154_315462

/-- The function f(x) = 3x³ - 2x² - 18x + 9 -/
def f (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 - 18 * x + 9

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 9 * x^2 - 4 * x - 18

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 18 * x - 4

theorem f_min_at_five_thirds :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 5/3 ∧ |x - 5/3| < ε → f x > f (5/3) :=
sorry

end NUMINAMATH_CALUDE_f_min_at_five_thirds_l3154_315462


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3154_315456

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a ∈ Set.Ioi 3 ∪ Set.Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3154_315456


namespace NUMINAMATH_CALUDE_female_employees_count_l3154_315488

/-- Proves that the total number of female employees in a company is 500 under given conditions -/
theorem female_employees_count (total_employees : ℕ) (male_employees : ℕ) (female_managers : ℕ) :
  female_managers = 200 →
  (2 : ℚ) / 5 * total_employees = female_managers + (2 : ℚ) / 5 * male_employees →
  total_employees = male_employees + 500 :=
by sorry

end NUMINAMATH_CALUDE_female_employees_count_l3154_315488


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l3154_315466

theorem sum_of_fourth_powers (a b : ℝ) 
  (h1 : a^2 - b^2 = 8) 
  (h2 : a * b = 2) : 
  a^4 + b^4 = 72 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l3154_315466


namespace NUMINAMATH_CALUDE_palindrome_difference_unique_l3154_315449

/-- A four-digit palindromic integer -/
def FourDigitPalindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ ∃ (a d : ℕ), n = 1001 * a + 110 * d ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9

/-- A three-digit palindromic integer -/
def ThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ (c f : ℕ), n = 101 * c + 10 * f ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ f ∧ f ≤ 9

theorem palindrome_difference_unique :
  ∀ A B C : ℕ,
  FourDigitPalindrome A →
  FourDigitPalindrome B →
  ThreeDigitPalindrome C →
  A - B = C →
  C = 121 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_difference_unique_l3154_315449


namespace NUMINAMATH_CALUDE_biffs_drinks_and_snacks_cost_l3154_315463

/-- Represents Biff's expenses and earnings during his bus trip -/
structure BusTrip where
  ticket_cost : ℝ
  headphones_cost : ℝ
  online_rate : ℝ
  wifi_rate : ℝ
  trip_duration : ℝ

/-- Calculates the amount Biff spent on drinks and snacks -/
def drinks_and_snacks_cost (trip : BusTrip) : ℝ :=
  (trip.online_rate - trip.wifi_rate) * trip.trip_duration - 
  (trip.ticket_cost + trip.headphones_cost)

/-- Theorem stating that Biff's expenses on drinks and snacks equal $3 -/
theorem biffs_drinks_and_snacks_cost :
  let trip := BusTrip.mk 11 16 12 2 3
  drinks_and_snacks_cost trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_biffs_drinks_and_snacks_cost_l3154_315463


namespace NUMINAMATH_CALUDE_unit_digit_of_23_power_100000_l3154_315477

theorem unit_digit_of_23_power_100000 : 23^100000 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_23_power_100000_l3154_315477


namespace NUMINAMATH_CALUDE_parallelogram_area_is_three_l3154_315440

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (a b : Fin 2 → ℝ) : ℝ :=
  |a 0 * b 1 - a 1 * b 0|

/-- Given vectors v, w, and u, prove that the area of the parallelogram
    formed by (v + u) and w is 3 -/
theorem parallelogram_area_is_three :
  let v : Fin 2 → ℝ := ![7, -4]
  let w : Fin 2 → ℝ := ![3, 1]
  let u : Fin 2 → ℝ := ![-1, 5]
  parallelogramArea (v + u) w = 3 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_area_is_three_l3154_315440


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l3154_315431

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 15 + Nat.factorial 17) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 15 + Nat.factorial 17) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l3154_315431


namespace NUMINAMATH_CALUDE_age_ratio_after_two_years_l3154_315422

/-- Proves that the ratio of a man's age to his student's age after two years is 2:1,
    given that the man is 26 years older than his 24-year-old student. -/
theorem age_ratio_after_two_years (student_age : ℕ) (man_age : ℕ) : 
  student_age = 24 →
  man_age = student_age + 26 →
  (man_age + 2) / (student_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_after_two_years_l3154_315422


namespace NUMINAMATH_CALUDE_fraction_equality_l3154_315433

theorem fraction_equality (a b : ℝ) (h : a / b = 6 / 5) :
  (5 * a + 4 * b) / (5 * a - 4 * b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3154_315433


namespace NUMINAMATH_CALUDE_pet_store_puppies_sold_l3154_315409

/-- Proves that the number of puppies sold is 1, given the conditions of the pet store problem. -/
theorem pet_store_puppies_sold :
  let kittens_sold : ℕ := 2
  let kitten_price : ℕ := 6
  let puppy_price : ℕ := 5
  let total_earnings : ℕ := 17
  let puppies_sold : ℕ := (total_earnings - kittens_sold * kitten_price) / puppy_price
  puppies_sold = 1 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_sold_l3154_315409


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3154_315493

/-- Given two vectors a and b in R², if a is parallel to b, then |2a - b| = 4√5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a.1 = 1 ∧ a.2 = 2 ∧ b.1 = -2 → a.1 * b.2 = a.2 * b.1 → 
  ‖(2 • a - b : ℝ × ℝ)‖ = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l3154_315493


namespace NUMINAMATH_CALUDE_hypotenuse_of_right_triangle_with_inscribed_circle_l3154_315459

/-- 
Given a right triangle with an inscribed circle, where the point of tangency 
divides one of the legs into segments of lengths m and n (m < n), 
the hypotenuse of the triangle is (m^2 + n^2) / (n - m).
-/
theorem hypotenuse_of_right_triangle_with_inscribed_circle 
  (m n : ℝ) (h : m < n) : ∃ (x : ℝ), 
  x > 0 ∧ 
  x = (m^2 + n^2) / (n - m) ∧
  x^2 = (x - n + m)^2 + (m + n)^2 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_of_right_triangle_with_inscribed_circle_l3154_315459


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3154_315423

theorem min_value_quadratic (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 8*x₀ + 6*y₀ + 25 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3154_315423


namespace NUMINAMATH_CALUDE_paving_cost_l3154_315453

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 1000) :
  length * width * rate = 20625 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l3154_315453


namespace NUMINAMATH_CALUDE_cupcake_package_size_l3154_315445

/-- The number of cupcakes in the smaller package -/
def smaller_package : ℕ := 10

/-- The number of cupcakes in the larger package -/
def larger_package : ℕ := 15

/-- The number of packs of each size bought -/
def packs_bought : ℕ := 4

/-- The total number of children to receive cupcakes -/
def total_children : ℕ := 100

theorem cupcake_package_size :
  packs_bought * larger_package + packs_bought * smaller_package = total_children :=
sorry

end NUMINAMATH_CALUDE_cupcake_package_size_l3154_315445


namespace NUMINAMATH_CALUDE_ant_movement_probability_l3154_315429

/-- A point in a 3D cubic lattice grid -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- The number of adjacent points in a 3D cubic lattice -/
def adjacent_points : ℕ := 6

/-- The number of steps the ant takes -/
def num_steps : ℕ := 4

/-- The probability of moving to a specific adjacent point in one step -/
def step_probability : ℚ := 1 / adjacent_points

/-- 
  Theorem: The probability of an ant moving from point A to point B 
  (directly one floor above A) on a cubic lattice grid in exactly four steps, 
  where each step is to an adjacent point with equal probability, is 1/1296.
-/
theorem ant_movement_probability (A B : Point3D) 
  (h1 : B.x = A.x ∧ B.y = A.y ∧ B.z = A.z + 1) : 
  step_probability ^ num_steps = 1 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_ant_movement_probability_l3154_315429


namespace NUMINAMATH_CALUDE_C_7_3_2_eq_10_l3154_315430

/-- A function that calculates the number of ways to select k elements from a set of n elements
    with a minimum distance of m between selected elements. -/
def C (n k m : ℕ) : ℕ := sorry

/-- The theorem stating that C_7^(3,2) = 10 -/
theorem C_7_3_2_eq_10 : C 7 3 2 = 10 := by sorry

end NUMINAMATH_CALUDE_C_7_3_2_eq_10_l3154_315430


namespace NUMINAMATH_CALUDE_linear_function_slope_l3154_315402

/-- A linear function y = kx + b where y decreases by 2 when x increases by 3 -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem linear_function_slope (k b : ℝ) :
  (∀ x : ℝ, linear_function k b (x + 3) = linear_function k b x - 2) →
  k = -2/3 := by
sorry

end NUMINAMATH_CALUDE_linear_function_slope_l3154_315402


namespace NUMINAMATH_CALUDE_monotonic_decreasing_range_l3154_315400

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- State the theorem
theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) → a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_monotonic_decreasing_range_l3154_315400


namespace NUMINAMATH_CALUDE_weight_after_one_year_l3154_315439

def initial_weight : ℕ := 250

def training_loss : List ℕ := [8, 5, 7, 6, 8, 7, 5, 7, 4, 6, 5, 7]

def diet_loss_per_month : ℕ := 3

def months_in_year : ℕ := 12

theorem weight_after_one_year :
  initial_weight - (training_loss.sum + diet_loss_per_month * months_in_year) = 139 := by
  sorry

end NUMINAMATH_CALUDE_weight_after_one_year_l3154_315439


namespace NUMINAMATH_CALUDE_cube_root_product_l3154_315491

theorem cube_root_product : (4^9 * 5^6 * 7^3 : ℝ)^(1/3) = 11200 := by sorry

end NUMINAMATH_CALUDE_cube_root_product_l3154_315491


namespace NUMINAMATH_CALUDE_divisible_by_six_percentage_l3154_315455

theorem divisible_by_six_percentage (n : ℕ) : n = 150 →
  (((Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))).card : ℚ) / (n : ℚ)) * 100 = 50/3 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_percentage_l3154_315455


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_180_l3154_315434

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_180 :
  rectangle_area 2025 10 = 180 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_180_l3154_315434


namespace NUMINAMATH_CALUDE_johns_allowance_theorem_l3154_315426

/-- The fraction of John's remaining allowance spent at the toy store -/
def toy_store_fraction (total_allowance : ℚ) (arcade_fraction : ℚ) (candy_amount : ℚ) : ℚ :=
  let remaining_after_arcade := total_allowance * (1 - arcade_fraction)
  let toy_store_amount := remaining_after_arcade - candy_amount
  toy_store_amount / remaining_after_arcade

/-- Proof that John spent 1/3 of his remaining allowance at the toy store -/
theorem johns_allowance_theorem :
  toy_store_fraction 3.60 (3/5) 0.96 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_theorem_l3154_315426


namespace NUMINAMATH_CALUDE_greatest_x_lcm_l3154_315413

def is_lcm (a b c m : ℕ) : Prop :=
  m % a = 0 ∧ m % b = 0 ∧ m % c = 0 ∧
  ∀ n : ℕ, (n % a = 0 ∧ n % b = 0 ∧ n % c = 0) → m ≤ n

theorem greatest_x_lcm :
  ∀ x : ℕ, is_lcm x 15 21 105 → x ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_lcm_l3154_315413


namespace NUMINAMATH_CALUDE_carrot_sticks_total_l3154_315448

theorem carrot_sticks_total (before_dinner after_dinner total : ℕ) 
  (h1 : before_dinner = 22)
  (h2 : after_dinner = 15)
  (h3 : total = before_dinner + after_dinner) :
  total = 37 := by sorry

end NUMINAMATH_CALUDE_carrot_sticks_total_l3154_315448


namespace NUMINAMATH_CALUDE_moderator_earnings_per_hour_l3154_315412

/-- Calculates the earnings per hour for a social media post moderator -/
theorem moderator_earnings_per_hour 
  (payment_per_post : ℝ) 
  (time_per_post : ℝ) 
  (seconds_per_hour : ℝ) 
  (h1 : payment_per_post = 0.25)
  (h2 : time_per_post = 10)
  (h3 : seconds_per_hour = 3600) :
  (payment_per_post * (seconds_per_hour / time_per_post)) = 90 := by
  sorry

#check moderator_earnings_per_hour

end NUMINAMATH_CALUDE_moderator_earnings_per_hour_l3154_315412


namespace NUMINAMATH_CALUDE_corner_sum_is_168_l3154_315476

def checkerboard_size : Nat := 9

def min_number : Nat := 2
def max_number : Nat := 82

def top_left : Nat := min_number
def top_right : Nat := min_number + checkerboard_size - 1
def bottom_left : Nat := max_number - checkerboard_size + 1
def bottom_right : Nat := max_number

theorem corner_sum_is_168 :
  top_left + top_right + bottom_left + bottom_right = 168 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_168_l3154_315476


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l3154_315425

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 9^n - 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 61 ∧ q ≠ 61 ∧ 
   x = 2 * p * q * 61 ∧ 
   ∀ r : ℕ, Prime r → r ∣ x → (r = 2 ∨ r = p ∨ r = q ∨ r = 61)) →
  x = 59048 := by
sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l3154_315425


namespace NUMINAMATH_CALUDE_direct_proportion_information_needed_l3154_315480

/-- A structure representing a direct proportion between x and y -/
structure DirectProportion where
  k : ℝ  -- Constant of proportionality
  y : ℝ → ℝ  -- Function mapping x to y
  prop : ∀ x, y x = k * x  -- Property of direct proportion

/-- The number of pieces of information needed to determine a direct proportion -/
def informationNeeded : ℕ := 2

/-- Theorem stating that exactly 2 pieces of information are needed to determine a direct proportion -/
theorem direct_proportion_information_needed :
  ∀ (dp : DirectProportion), informationNeeded = 2 :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_information_needed_l3154_315480


namespace NUMINAMATH_CALUDE_ratio_of_a_to_b_l3154_315415

theorem ratio_of_a_to_b (a b : ℚ) (h : (6*a - 5*b) / (8*a - 3*b) = 2/7) : 
  a/b = 29/26 := by sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_b_l3154_315415


namespace NUMINAMATH_CALUDE_smallest_a_minus_b_l3154_315475

theorem smallest_a_minus_b (a b n : ℤ) : 
  (a + b < 11) →
  (a > n) →
  (∀ (c d : ℤ), c + d < 11 → c - d ≥ 4) →
  (a - b = 4) →
  (∀ m : ℤ, a > m → m ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_minus_b_l3154_315475


namespace NUMINAMATH_CALUDE_return_trip_time_l3154_315474

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  p : ℝ  -- Speed of the plane in still air
  w : ℝ  -- Speed of the wind
  d : ℝ  -- Distance between the cities

/-- The conditions of the flight scenario -/
def validFlightScenario (f : FlightScenario) : Prop :=
  f.p > 0 ∧ f.w > 0 ∧ f.d > 0 ∧
  f.d / (f.p - f.w) = 90 ∧
  f.d / (f.p + f.w) = f.d / f.p - 15

/-- The theorem stating that the return trip takes 64 minutes -/
theorem return_trip_time (f : FlightScenario) 
  (h : validFlightScenario f) : 
  f.d / (f.p + f.w) = 64 := by
  sorry

end NUMINAMATH_CALUDE_return_trip_time_l3154_315474


namespace NUMINAMATH_CALUDE_temperature_conversion_l3154_315406

theorem temperature_conversion (C F : ℝ) : 
  C = 35 → C = (4/7) * (F - 40) → F = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l3154_315406


namespace NUMINAMATH_CALUDE_floor_sum_equals_n_l3154_315427

theorem floor_sum_equals_n (n : ℤ) : 
  ⌊n / 2⌋ + ⌊(n + 1) / 2⌋ = n := by sorry

end NUMINAMATH_CALUDE_floor_sum_equals_n_l3154_315427


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3154_315490

/-- The area of a quadrilateral with non-perpendicular diagonals -/
theorem quadrilateral_area (a b c d φ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hφ : 0 < φ ∧ φ < π / 2) :
  let S := Real.tan φ * |a^2 + c^2 - b^2 - d^2| / 4
  ∃ (d₁ d₂ : ℝ), d₁ > 0 ∧ d₂ > 0 ∧ S = d₁ * d₂ * Real.sin φ / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3154_315490


namespace NUMINAMATH_CALUDE_fifteen_clockwise_opposite_l3154_315495

/-- Represents a circle of equally spaced children -/
structure ChildrenCircle where
  num_children : ℕ
  standard_child : ℕ

/-- The child directly opposite another child in the circle -/
def opposite_child (circle : ChildrenCircle) (child : ℕ) : ℕ :=
  (child + circle.num_children / 2) % circle.num_children

theorem fifteen_clockwise_opposite (circle : ChildrenCircle) :
  opposite_child circle circle.standard_child = (circle.standard_child + 15) % circle.num_children →
  circle.num_children = 30 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_clockwise_opposite_l3154_315495


namespace NUMINAMATH_CALUDE_calculation_proof_l3154_315447

theorem calculation_proof : 2 * Real.sin (30 * π / 180) - (8 : ℝ) ^ (1/3) + (2 - Real.pi) ^ 0 + (-1) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3154_315447


namespace NUMINAMATH_CALUDE_investment_interest_proof_l3154_315492

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

theorem investment_interest_proof :
  let principal : ℝ := 1500
  let rate : ℝ := 0.03
  let time : ℕ := 10
  ⌊compound_interest principal rate time⌋ = 516 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_proof_l3154_315492


namespace NUMINAMATH_CALUDE_marble_fraction_after_doubling_red_l3154_315479

theorem marble_fraction_after_doubling_red (total : ℚ) (h : total > 0) :
  let blue := (2 / 3) * total
  let red := total - blue
  let new_red := 2 * red
  let new_total := blue + new_red
  new_red / new_total = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_marble_fraction_after_doubling_red_l3154_315479


namespace NUMINAMATH_CALUDE_finley_class_size_l3154_315432

/-- The number of students in Mrs. Finley's class -/
def finley_class : ℕ := sorry

/-- The number of students in Mr. Johnson's class -/
def johnson_class : ℕ := 22

/-- Mr. Johnson's class has 10 more than half the number in Mrs. Finley's class -/
axiom johnson_class_size : johnson_class = finley_class / 2 + 10

theorem finley_class_size : finley_class = 24 := by sorry

end NUMINAMATH_CALUDE_finley_class_size_l3154_315432


namespace NUMINAMATH_CALUDE_x_in_interval_l3154_315497

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define x as in the problem
noncomputable def x : ℝ := 1 / log (1/2) (1/3) + 1 / log (1/5) (1/3)

-- State the theorem
theorem x_in_interval : 2 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_x_in_interval_l3154_315497
