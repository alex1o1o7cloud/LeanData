import Mathlib

namespace possible_m_values_l2238_223862

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem possible_m_values : 
  {m : ℝ | B m ⊆ A} = {-1/2, 0, 1/3} := by sorry

end possible_m_values_l2238_223862


namespace walking_problem_l2238_223899

/-- Represents the walking problem from "The Nine Chapters on the Mathematical Art" -/
theorem walking_problem (x : ℝ) :
  (∀ d : ℝ, d > 0 → 100 * (d / 60) = d) →  -- Good walker takes 100 steps for every 60 steps of bad walker
  x = 100 + (60 / 100) * x →               -- The equation to be proved
  x = (100 * 100) / 40                     -- The solution (not given in the original problem, but included for completeness)
  := by sorry

end walking_problem_l2238_223899


namespace tournament_participants_l2238_223816

theorem tournament_participants : ∃ (n : ℕ), n > 0 ∧ 
  (n * (n - 1) / 2 : ℚ) = 90 + (n - 10) * (n - 11) ∧ 
  (∀ k : ℕ, k ≠ n → (k * (k - 1) / 2 : ℚ) ≠ 90 + (k - 10) * (k - 11)) := by
  sorry

end tournament_participants_l2238_223816


namespace pentagon_side_length_l2238_223833

/-- A five-sided figure with equal side lengths -/
structure Pentagon where
  side_length : ℝ
  perimeter : ℝ
  side_count : ℕ := 5
  all_sides_equal : perimeter = side_count * side_length

/-- Theorem: Given a pentagon with perimeter 23.4 cm, the length of one side is 4.68 cm -/
theorem pentagon_side_length (p : Pentagon) (h : p.perimeter = 23.4) : p.side_length = 4.68 := by
  sorry

end pentagon_side_length_l2238_223833


namespace like_terms_imply_a_minus_b_eq_two_l2238_223842

/-- Two algebraic expressions are like terms if they have the same variables raised to the same powers. -/
def are_like_terms (expr1 expr2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (m n : ℕ), 
    (∀ x y, expr1 x y = c₁ * x^m * y^n) ∧ 
    (∀ x y, expr2 x y = c₂ * x^m * y^n)

/-- Given that -2.5x^(a+b)y^(a-1) and 3x^2y are like terms, prove that a - b = 2 -/
theorem like_terms_imply_a_minus_b_eq_two 
  (a b : ℝ) 
  (h : are_like_terms (λ x y => -2.5 * x^(a+b) * y^(a-1)) (λ x y => 3 * x^2 * y)) : 
  a - b = 2 := by
sorry


end like_terms_imply_a_minus_b_eq_two_l2238_223842


namespace passed_candidates_count_l2238_223806

/-- Prove the number of passed candidates given total candidates and average marks -/
theorem passed_candidates_count
  (total_candidates : ℕ)
  (avg_all : ℚ)
  (avg_passed : ℚ)
  (avg_failed : ℚ)
  (h_total : total_candidates = 120)
  (h_avg_all : avg_all = 35)
  (h_avg_passed : avg_passed = 39)
  (h_avg_failed : avg_failed = 15) :
  ∃ (passed_candidates : ℕ), passed_candidates = 100 ∧
    passed_candidates ≤ total_candidates ∧
    (passed_candidates : ℚ) * avg_passed +
    (total_candidates - passed_candidates : ℚ) * avg_failed =
    (total_candidates : ℚ) * avg_all :=
by sorry

end passed_candidates_count_l2238_223806


namespace games_in_23_team_tournament_l2238_223814

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_teams : ℕ
  no_ties : Bool

/-- The number of games played in a single-elimination tournament -/
def games_played (t : Tournament) : ℕ := t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams and no ties, 
    the number of games played is 22 -/
theorem games_in_23_team_tournament (t : Tournament) 
  (h1 : t.num_teams = 23) (h2 : t.no_ties = true) : 
  games_played t = 22 := by
  sorry

end games_in_23_team_tournament_l2238_223814


namespace train_final_speed_train_final_speed_zero_l2238_223887

/-- Proves that a train with given initial speed and deceleration comes to a stop before traveling 4 km -/
theorem train_final_speed (v_i : Real) (a : Real) (d : Real) :
  v_i = 189 * (1000 / 3600) →
  a = -0.5 →
  d = 4000 →
  v_i^2 + 2 * a * d < 0 →
  ∃ (d_stop : Real), d_stop < d ∧ v_i^2 + 2 * a * d_stop = 0 :=
by sorry

/-- Proves that the final speed of the train after traveling 4 km is 0 m/s -/
theorem train_final_speed_zero (v_i : Real) (a : Real) (d : Real) (v_f : Real) :
  v_i = 189 * (1000 / 3600) →
  a = -0.5 →
  d = 4000 →
  v_f^2 = v_i^2 + 2 * a * d →
  v_f = 0 :=
by sorry

end train_final_speed_train_final_speed_zero_l2238_223887


namespace floor_sum_eval_l2238_223861

theorem floor_sum_eval : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_eval_l2238_223861


namespace jason_potato_eating_time_l2238_223867

/-- Given that Jason eats 27 potatoes in 3 hours, prove that it takes him 20 minutes to eat 3 potatoes. -/
theorem jason_potato_eating_time :
  ∀ (total_potatoes total_hours potatoes_to_eat : ℕ) (minutes_per_hour : ℕ),
    total_potatoes = 27 →
    total_hours = 3 →
    potatoes_to_eat = 3 →
    minutes_per_hour = 60 →
    (potatoes_to_eat * total_hours * minutes_per_hour) / total_potatoes = 20 := by
  sorry

end jason_potato_eating_time_l2238_223867


namespace movie_marathon_duration_l2238_223875

def movie_marathon (first_movie : ℝ) (second_movie_percentage : ℝ) (third_movie_difference : ℝ) : ℝ :=
  let second_movie := first_movie * (1 + second_movie_percentage)
  let third_movie := first_movie + second_movie - third_movie_difference
  first_movie + second_movie + third_movie

theorem movie_marathon_duration :
  movie_marathon 2 0.5 1 = 9 := by
  sorry

end movie_marathon_duration_l2238_223875


namespace chess_tournament_l2238_223822

theorem chess_tournament (n : ℕ) : 
  (n ≥ 3) →  -- Ensure at least 3 players (2 who withdraw + 1 more)
  ((n - 2) * (n - 3) / 2 + 3 = 81) →  -- Total games equation
  (n = 15) :=
by
  sorry

end chess_tournament_l2238_223822


namespace milton_books_l2238_223852

theorem milton_books (total : ℕ) (zoology : ℕ) (botany : ℕ) : 
  total = 960 → 
  botany = 7 * zoology → 
  total = zoology + botany → 
  zoology = 120 :=
by
  sorry

end milton_books_l2238_223852


namespace quadratic_root_m_value_l2238_223824

theorem quadratic_root_m_value :
  ∀ m : ℝ, (1 : ℝ)^2 + m * 1 + 2 = 0 → m = -3 := by
  sorry

end quadratic_root_m_value_l2238_223824


namespace gcd_of_256_180_600_l2238_223868

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 4 := by
  sorry

end gcd_of_256_180_600_l2238_223868


namespace airplane_stop_time_l2238_223856

/-- The distance function for an airplane after landing -/
def distance (t : ℝ) : ℝ := 75 * t - 1.5 * t^2

/-- The time at which the airplane stops -/
def stop_time : ℝ := 25

theorem airplane_stop_time :
  (∀ t : ℝ, distance t ≤ distance stop_time) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ t, |t - stop_time| < δ → distance stop_time - distance t < ε) :=
sorry

end airplane_stop_time_l2238_223856


namespace inequality_addition_l2238_223881

theorem inequality_addition (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end inequality_addition_l2238_223881


namespace intersection_range_l2238_223830

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the property of having three distinct intersection points
def has_three_distinct_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
  f x₁ = a ∧ f x₂ = a ∧ f x₃ = a

-- Theorem statement
theorem intersection_range (a : ℝ) :
  has_three_distinct_intersections a → -2 < a ∧ a < 2 :=
by sorry

end intersection_range_l2238_223830


namespace P_not_factorable_l2238_223855

/-- The polynomial P(x,y) = x^n + xy + y^n -/
def P (n : ℕ) (x y : ℝ) : ℝ := x^n + x*y + y^n

/-- Theorem stating that P(x,y) cannot be factored into two non-constant real polynomials -/
theorem P_not_factorable (n : ℕ) :
  ¬∃ (G H : ℝ → ℝ → ℝ), 
    (∀ x y, P n x y = G x y * H x y) ∧ 
    (∃ a b c d, G a b ≠ G c d) ∧ 
    (∃ a b c d, H a b ≠ H c d) :=
sorry

end P_not_factorable_l2238_223855


namespace complex_subtraction_simplification_l2238_223812

theorem complex_subtraction_simplification :
  (7 - 3*I) - (2 + 5*I) = 5 - 8*I :=
by sorry

end complex_subtraction_simplification_l2238_223812


namespace square_sum_from_diff_and_product_l2238_223821

theorem square_sum_from_diff_and_product (p q : ℝ) 
  (h1 : p - q = 4) 
  (h2 : p * q = -2) : 
  p^2 + q^2 = 12 := by
  sorry

end square_sum_from_diff_and_product_l2238_223821


namespace abs_neg_four_equals_four_l2238_223886

theorem abs_neg_four_equals_four : |(-4 : ℤ)| = 4 := by
  sorry

end abs_neg_four_equals_four_l2238_223886


namespace katya_magic_pen_problem_l2238_223839

theorem katya_magic_pen_problem (katya_prob : ℚ) (pen_prob : ℚ) (total_problems : ℕ) (min_correct : ℕ) :
  katya_prob = 4/5 →
  pen_prob = 1/2 →
  total_problems = 20 →
  min_correct = 13 →
  ∃ x : ℕ, x ≥ 10 ∧
    (x : ℚ) * katya_prob + (total_problems - x : ℚ) * pen_prob ≥ min_correct ∧
    ∀ y : ℕ, y < 10 →
      (y : ℚ) * katya_prob + (total_problems - y : ℚ) * pen_prob < min_correct :=
by sorry

end katya_magic_pen_problem_l2238_223839


namespace median_and_mode_are_50_l2238_223836

/-- Represents a speed measurement and its frequency --/
structure SpeedData where
  speed : ℕ
  frequency : ℕ

/-- The dataset of vehicle speeds and their frequencies --/
def speedDataset : List SpeedData := [
  ⟨48, 5⟩,
  ⟨49, 4⟩,
  ⟨50, 8⟩,
  ⟨51, 2⟩,
  ⟨52, 1⟩
]

/-- Calculates the median of the dataset --/
def calculateMedian (data : List SpeedData) : ℕ := sorry

/-- Calculates the mode of the dataset --/
def calculateMode (data : List SpeedData) : ℕ := sorry

/-- Theorem stating that the median and mode of the dataset are both 50 --/
theorem median_and_mode_are_50 :
  calculateMedian speedDataset = 50 ∧ calculateMode speedDataset = 50 := by sorry

end median_and_mode_are_50_l2238_223836


namespace orange_stock_proof_l2238_223843

/-- Represents the original stock of oranges in kg -/
def original_stock : ℝ := 2700

/-- Represents the percentage of stock remaining after sale -/
def remaining_percentage : ℝ := 0.25

/-- Represents the amount of oranges remaining after sale in kg -/
def remaining_stock : ℝ := 675

theorem orange_stock_proof :
  remaining_percentage * original_stock = remaining_stock :=
sorry

end orange_stock_proof_l2238_223843


namespace figure_18_to_square_l2238_223898

/-- Represents a figure on a graph paper -/
structure Figure where
  area : ℕ

/-- Represents a cut of the figure -/
structure Cut where
  parts : ℕ

/-- Represents the result of rearranging the cut parts -/
structure Rearrangement where
  is_square : Bool

/-- Function to determine if a figure can be cut and rearranged into a square -/
def can_form_square (f : Figure) (c : Cut) : Prop :=
  ∃ (r : Rearrangement), r.is_square = true

/-- Theorem stating that a figure with area 18 can be cut into 3 parts and rearranged into a square -/
theorem figure_18_to_square :
  ∀ (f : Figure) (c : Cut), 
    f.area = 18 → c.parts = 3 → can_form_square f c :=
by sorry

end figure_18_to_square_l2238_223898


namespace cone_generatrix_length_l2238_223896

theorem cone_generatrix_length (r : ℝ) (h1 : r = Real.sqrt 2) :
  let l := 2 * Real.sqrt 2
  (2 * Real.pi * r = Real.pi * l) → l = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_l2238_223896


namespace quadratic_no_real_roots_l2238_223883

theorem quadratic_no_real_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x + m ≠ 0) → m > 9/4 := by
  sorry

end quadratic_no_real_roots_l2238_223883


namespace cube_folding_l2238_223829

/-- Represents the squares on the flat sheet --/
inductive Square
| A | B | C | D | E | F

/-- Represents the adjacency of squares on the flat sheet --/
def adjacent : Square → Square → Prop :=
  sorry

/-- Represents the opposite faces of the cube after folding --/
def opposite : Square → Square → Prop :=
  sorry

/-- The theorem to be proved --/
theorem cube_folding (h1 : adjacent Square.B Square.A)
                     (h2 : adjacent Square.C Square.B)
                     (h3 : adjacent Square.C Square.A)
                     (h4 : adjacent Square.D Square.C)
                     (h5 : adjacent Square.E Square.A)
                     (h6 : adjacent Square.F Square.D)
                     (h7 : adjacent Square.F Square.E) :
  opposite Square.A Square.D :=
sorry

end cube_folding_l2238_223829


namespace complex_in_fourth_quadrant_l2238_223873

/-- 
Given a real number m < 1, prove that the complex number 1 + (m-1)i 
is located in the fourth quadrant of the complex plane.
-/
theorem complex_in_fourth_quadrant (m : ℝ) (h : m < 1) : 
  let z : ℂ := 1 + (m - 1) * I
  (z.re > 0 ∧ z.im < 0) := by sorry

end complex_in_fourth_quadrant_l2238_223873


namespace train_length_proof_l2238_223811

-- Define the speed of the train in km/hr
def train_speed_kmh : ℝ := 108

-- Define the time it takes for the train to pass the tree in seconds
def passing_time : ℝ := 8

-- Theorem to prove the length of the train
theorem train_length_proof : 
  train_speed_kmh * 1000 / 3600 * passing_time = 240 := by
  sorry

#check train_length_proof

end train_length_proof_l2238_223811


namespace conference_men_count_l2238_223884

/-- The number of men at a climate conference -/
def number_of_men : ℕ := 700

/-- The number of women at the conference -/
def number_of_women : ℕ := 500

/-- The number of children at the conference -/
def number_of_children : ℕ := 800

/-- The percentage of men who were Indian -/
def indian_men_percentage : ℚ := 20 / 100

/-- The percentage of women who were Indian -/
def indian_women_percentage : ℚ := 40 / 100

/-- The percentage of children who were Indian -/
def indian_children_percentage : ℚ := 10 / 100

/-- The percentage of people who were not Indian -/
def non_indian_percentage : ℚ := 79 / 100

theorem conference_men_count :
  let total_people := number_of_men + number_of_women + number_of_children
  let indian_people := (indian_men_percentage * number_of_men) + 
                       (indian_women_percentage * number_of_women) + 
                       (indian_children_percentage * number_of_children)
  (1 - non_indian_percentage) * total_people = indian_people →
  number_of_men = 700 := by sorry

end conference_men_count_l2238_223884


namespace line_relationship_undetermined_l2238_223857

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola defined by y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a parabola -/
def pointOnParabola (p : Point) (para : Parabola) : Prop :=
  p.y = para.a * p.x^2 + para.b * p.x + para.c

/-- The relationship between two lines -/
inductive LineRelationship
  | Parallel
  | Perpendicular
  | Intersecting

/-- Theorem: The relationship between AD and BC cannot be determined -/
theorem line_relationship_undetermined 
  (A B C D : Point) 
  (para : Parabola) 
  (h_a_nonzero : para.a ≠ 0)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_on_parabola : pointOnParabola A para ∧ pointOnParabola B para ∧ 
                   pointOnParabola C para ∧ pointOnParabola D para)
  (h_x_sum : A.x + D.x - B.x + C.x = 0) :
  ∀ r : LineRelationship, ∃ para' : Parabola, 
    para'.a ≠ 0 ∧
    (pointOnParabola A para' ∧ pointOnParabola B para' ∧ 
     pointOnParabola C para' ∧ pointOnParabola D para') ∧
    A.x + D.x - B.x + C.x = 0 :=
  sorry

end line_relationship_undetermined_l2238_223857


namespace triangle_properties_l2238_223804

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : 3 * (t.b^2 + t.c^2) = 3 * t.a^2 + 2 * t.b * t.c)
  (h2 : t.a = 2)
  (h3 : t.b + t.c = 2 * Real.sqrt 2)
  (h4 : Real.sin t.B = Real.sqrt 2 * Real.cos t.C) :
  (∃ S : ℝ, S = Real.sqrt 2 / 2 ∧ S = 1/2 * t.b * t.c * Real.sin t.A) ∧ 
  Real.cos t.C = Real.sqrt 3 / 3 := by
  sorry


end triangle_properties_l2238_223804


namespace choir_members_count_l2238_223813

theorem choir_members_count : ∃! n : ℕ, 
  200 ≤ n ∧ n ≤ 300 ∧ 
  n % 10 = 4 ∧ 
  n % 11 = 5 ∧ 
  n = 234 := by sorry

end choir_members_count_l2238_223813


namespace fraction_to_decimal_l2238_223877

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end fraction_to_decimal_l2238_223877


namespace expression_equality_l2238_223870

theorem expression_equality : (3 / 5 : ℚ) * ((2 / 3 + 3 / 8) / 2) - 1 / 16 = 1 / 4 := by
  sorry

end expression_equality_l2238_223870


namespace restaurant_pie_days_l2238_223831

/-- Given a restaurant that sells a constant number of pies per day,
    calculate the number of days based on the total pies sold. -/
theorem restaurant_pie_days (pies_per_day : ℕ) (total_pies : ℕ) (h1 : pies_per_day = 8) (h2 : total_pies = 56) :
  total_pies / pies_per_day = 7 := by
  sorry

end restaurant_pie_days_l2238_223831


namespace mushroom_count_l2238_223845

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem: Given the conditions, the number of mushrooms is 950 -/
theorem mushroom_count : ∃ (N : ℕ), 
  100 ≤ N ∧ N < 1000 ∧  -- N is a three-digit number
  sumOfDigits N = 14 ∧  -- sum of digits is 14
  N % 50 = 0 ∧  -- N is divisible by 50
  N % 100 = 50 ∧  -- N ends in 50
  N = 950 := by
sorry

end mushroom_count_l2238_223845


namespace angle_F_measure_l2238_223818

-- Define a triangle DEF
structure Triangle :=
  (D E F : ℝ)

-- Define the properties of the triangle
def validTriangle (t : Triangle) : Prop :=
  t.D > 0 ∧ t.E > 0 ∧ t.F > 0 ∧ t.D + t.E + t.F = 180

-- Theorem statement
theorem angle_F_measure (t : Triangle) 
  (h1 : validTriangle t) 
  (h2 : t.D = 3 * t.E) 
  (h3 : t.E = 18) : 
  t.F = 108 := by
  sorry

end angle_F_measure_l2238_223818


namespace tangent_line_implies_k_value_l2238_223805

/-- Given a curve y = 3ln(x) + x + k, where k ∈ ℝ, if there exists a point P(x₀, y₀) on the curve
    such that the tangent line at P has the equation 4x - y - 1 = 0, then k = 2. -/
theorem tangent_line_implies_k_value (k : ℝ) (x₀ y₀ : ℝ) :
  y₀ = 3 * Real.log x₀ + x₀ + k →
  (∀ x y, y = 4 * x - 1 ↔ 4 * x - y - 1 = 0) →
  (∃ m b, ∀ x, 3 / x + 1 = m ∧ y₀ - m * x₀ = b ∧ y₀ = 4 * x₀ - 1) →
  k = 2 :=
by sorry

end tangent_line_implies_k_value_l2238_223805


namespace planes_parallel_transitive_l2238_223841

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_transitive 
  (α β γ : Plane) 
  (h1 : parallel α γ) 
  (h2 : parallel γ β) : 
  parallel α β :=
sorry

end planes_parallel_transitive_l2238_223841


namespace virginia_march_rainfall_l2238_223803

/-- Calculates the rainfall in March given the rainfall amounts for April, May, June, July, and the average rainfall for 5 months. -/
def march_rainfall (april may june july average : ℝ) : ℝ :=
  5 * average - (april + may + june + july)

/-- Theorem stating that the rainfall in March was 3.79 inches given the specified conditions. -/
theorem virginia_march_rainfall :
  let april : ℝ := 4.5
  let may : ℝ := 3.95
  let june : ℝ := 3.09
  let july : ℝ := 4.67
  let average : ℝ := 4
  march_rainfall april may june july average = 3.79 := by
  sorry

end virginia_march_rainfall_l2238_223803


namespace quadratic_roots_sum_of_squares_l2238_223825

theorem quadratic_roots_sum_of_squares (p q r s : ℝ) : 
  (∀ x, x^2 - 2*p*x + 3*q = 0 ↔ x = r ∨ x = s) → 
  r^2 + s^2 = 4*p^2 - 6*q := by
  sorry

end quadratic_roots_sum_of_squares_l2238_223825


namespace polynomial_equality_implies_c_value_l2238_223851

theorem polynomial_equality_implies_c_value (a c : ℚ) 
  (h : ∀ x : ℚ, (x + 3) * (x + a) = x^2 + c*x + 8) : 
  c = 17/3 := by
sorry

end polynomial_equality_implies_c_value_l2238_223851


namespace zero_point_in_interval_l2238_223835

def f (x : ℝ) := -x^3 - 3*x + 5

theorem zero_point_in_interval :
  (∀ x y, x < y → f x > f y) →  -- f is monotonically decreasing
  Continuous f →
  f 1 > 0 →
  f 2 < 0 →
  ∃ c, c ∈ Set.Ioo 1 2 ∧ f c = 0 :=
by sorry

end zero_point_in_interval_l2238_223835


namespace employment_percentage_l2238_223815

theorem employment_percentage (total_population : ℝ) (employed_population : ℝ) :
  (employed_population / total_population = 0.5 / 0.78125) ↔
  (0.5 * total_population = employed_population * (1 - 0.21875)) :=
by sorry

end employment_percentage_l2238_223815


namespace seating_theorem_l2238_223800

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (n : ℕ) (no_adjacent_pair : ℕ) (no_adjacent_triple : ℕ) : ℕ :=
  factorial n - (factorial (n - 1) * factorial 2 + factorial (n - 2) * factorial 3) + 
  factorial (n - 3) * factorial 2 * factorial 3

theorem seating_theorem : seating_arrangements 8 2 3 = 25360 := by
  sorry

end seating_theorem_l2238_223800


namespace quadratic_coefficient_l2238_223838

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x : ℝ, a * x^2 + b * x + c = a * (x - 2)^2 + 3) →
  a * 1^2 + b * 1 + c = 5 →
  a = 2 := by
  sorry

end quadratic_coefficient_l2238_223838


namespace integer_root_of_cubic_l2238_223823

/-- Given a cubic polynomial x^3 + bx + c = 0 with rational coefficients b and c,
    if 5 - √2 is a root, then -10 is also a root. -/
theorem integer_root_of_cubic (b c : ℚ) : 
  (5 - Real.sqrt 2)^3 + b*(5 - Real.sqrt 2) + c = 0 →
  (-10)^3 + b*(-10) + c = 0 := by
sorry

end integer_root_of_cubic_l2238_223823


namespace simplify_and_evaluate_l2238_223834

theorem simplify_and_evaluate (x y : ℚ) 
  (hx : x = 1) (hy : y = 1/2) : 
  (3*x + 2*y) * (3*x - 2*y) - (x - y)^2 = 31/4 := by
  sorry

end simplify_and_evaluate_l2238_223834


namespace midpoint_coordinates_l2238_223837

/-- Given point A and vector AB, prove that the midpoint of segment AB has specific coordinates -/
theorem midpoint_coordinates (A B : ℝ × ℝ) (h1 : A = (-3, 2)) (h2 : B - A = (6, 0)) :
  (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 2 := by
  sorry

#check midpoint_coordinates

end midpoint_coordinates_l2238_223837


namespace price_increase_percentage_l2238_223848

theorem price_increase_percentage (x : ℝ) : 
  (∀ (P : ℝ), P > 0 → P * (1 + x / 100) * (1 - 20 / 100) = P) → x = 25 := by
  sorry

end price_increase_percentage_l2238_223848


namespace complement_of_intersection_l2238_223897

/-- The universal set U -/
def U : Set Nat := {0, 1, 2, 3}

/-- The set M -/
def M : Set Nat := {0, 1, 2}

/-- The set N -/
def N : Set Nat := {1, 2, 3}

/-- Theorem stating that the complement of M ∩ N in U is {0, 3} -/
theorem complement_of_intersection (U M N : Set Nat) (hU : U = {0, 1, 2, 3}) (hM : M = {0, 1, 2}) (hN : N = {1, 2, 3}) :
  (M ∩ N)ᶜ = {0, 3} := by
  sorry

end complement_of_intersection_l2238_223897


namespace count_divisible_by_five_l2238_223810

/-- The set of available digits --/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A function to check if a three-digit number is valid (no leading zero) --/
def isValidNumber (n : Nat) : Bool :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 ≠ 0)

/-- A function to check if a number is formed from distinct digits in the given set --/
def isFromDistinctDigits (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- The set of valid three-digit numbers formed from the given digits --/
def validNumbers : Finset Nat :=
  Finset.filter (fun n => isValidNumber n ∧ isFromDistinctDigits n) (Finset.range 1000)

/-- The theorem to be proved --/
theorem count_divisible_by_five :
  (validNumbers.filter (fun n => n % 5 = 0)).card = 36 := by
  sorry

end count_divisible_by_five_l2238_223810


namespace m_mobile_additional_line_cost_l2238_223819

/-- Represents a mobile phone plan with a base cost and additional line cost -/
structure MobilePlan where
  baseCost : ℕ  -- Cost for first two lines
  addLineCost : ℕ  -- Cost for each additional line

/-- Calculates the total cost for a given number of lines -/
def totalCost (plan : MobilePlan) (lines : ℕ) : ℕ :=
  plan.baseCost + plan.addLineCost * (lines - 2)

theorem m_mobile_additional_line_cost :
  ∃ (mMobileAddCost : ℕ),
    let tMobile : MobilePlan := ⟨50, 16⟩
    let mMobile : MobilePlan := ⟨45, mMobileAddCost⟩
    totalCost tMobile 5 - totalCost mMobile 5 = 11 →
    mMobileAddCost = 14 := by
  sorry


end m_mobile_additional_line_cost_l2238_223819


namespace calvin_insect_collection_l2238_223880

/-- Calculates the total number of insects in Calvin's collection. -/
def total_insects (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := scorpions * 2
  roaches + scorpions + crickets + caterpillars

/-- Proves that Calvin has 27 insects in his collection. -/
theorem calvin_insect_collection : total_insects 12 3 = 27 := by
  sorry

end calvin_insect_collection_l2238_223880


namespace complex_product_PRS_l2238_223888

theorem complex_product_PRS : 
  let P : ℂ := 3 + 4 * Complex.I
  let R : ℂ := 2 * Complex.I
  let S : ℂ := 3 - 4 * Complex.I
  P * R * S = 50 * Complex.I :=
by sorry

end complex_product_PRS_l2238_223888


namespace rainfall_volume_calculation_l2238_223808

-- Define the rainfall in centimeters
def rainfall_cm : ℝ := 5

-- Define the ground area in hectares
def ground_area_hectares : ℝ := 1.5

-- Define the conversion factor from hectares to square meters
def hectares_to_sqm : ℝ := 10000

-- Define the conversion factor from centimeters to meters
def cm_to_m : ℝ := 0.01

-- Theorem statement
theorem rainfall_volume_calculation :
  let rainfall_m := rainfall_cm * cm_to_m
  let ground_area_sqm := ground_area_hectares * hectares_to_sqm
  rainfall_m * ground_area_sqm = 750 := by
  sorry

end rainfall_volume_calculation_l2238_223808


namespace sugar_left_l2238_223892

/-- Given that Pamela bought 9.8 ounces of sugar and spilled 5.2 ounces,
    prove that the amount of sugar left is 4.6 ounces. -/
theorem sugar_left (bought spilled : ℝ) (h1 : bought = 9.8) (h2 : spilled = 5.2) :
  bought - spilled = 4.6 := by
  sorry

end sugar_left_l2238_223892


namespace passes_through_origin_parallel_to_line_intersects_below_l2238_223858

/-- Linear function definition -/
def linear_function (m x : ℝ) : ℝ := (2*m + 1)*x + m - 3

/-- Theorem for when the function passes through the origin -/
theorem passes_through_origin (m : ℝ) : 
  linear_function m 0 = 0 ↔ m = 3 := by sorry

/-- Theorem for when the function is parallel to y = 3x - 3 -/
theorem parallel_to_line (m : ℝ) :
  (2*m + 1 = 3) ↔ m = 1 := by sorry

/-- Theorem for when the function intersects y-axis below x-axis -/
theorem intersects_below (m : ℝ) :
  (linear_function m 0 < 0 ∧ 2*m + 1 ≠ 0) ↔ (m < 3 ∧ m ≠ -1/2) := by sorry

end passes_through_origin_parallel_to_line_intersects_below_l2238_223858


namespace complement_union_equality_l2238_223882

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {0, 3, 4}

-- Define set B
def B : Set ℕ := {1, 3}

-- Theorem statement
theorem complement_union_equality : (U \ A) ∪ B = {1, 2, 3} := by
  sorry

end complement_union_equality_l2238_223882


namespace division_remainder_problem_l2238_223853

theorem division_remainder_problem (N : ℕ) (Q2 : ℕ) :
  (∃ R1 : ℕ, N = 44 * 432 + R1) ∧ 
  (∃ Q2 : ℕ, N = 38 * Q2 + 8) →
  N % 44 = 0 :=
sorry

end division_remainder_problem_l2238_223853


namespace scarves_per_box_l2238_223878

theorem scarves_per_box (num_boxes : ℕ) (total_pieces : ℕ) : 
  num_boxes = 6 → 
  total_pieces = 60 → 
  ∃ (scarves_per_box : ℕ), 
    scarves_per_box * num_boxes * 2 = total_pieces ∧ 
    scarves_per_box = 5 :=
by sorry

end scarves_per_box_l2238_223878


namespace composite_probability_is_point_68_l2238_223865

/-- The number of natural numbers from 1 to 50 -/
def total_numbers : ℕ := 50

/-- The number of composite numbers from 1 to 50 -/
def composite_count : ℕ := 34

/-- The probability of selecting a composite number from the first 50 natural numbers -/
def composite_probability : ℚ := composite_count / total_numbers

/-- Theorem: The probability of selecting a composite number from the first 50 natural numbers is 0.68 -/
theorem composite_probability_is_point_68 : composite_probability = 68 / 100 := by
  sorry

end composite_probability_is_point_68_l2238_223865


namespace syllogism_conclusion_l2238_223866

-- Define the sets
variable (U : Type) -- Universe set
variable (Mem : Set U) -- Set of Mems
variable (En : Set U) -- Set of Ens
variable (Veen : Set U) -- Set of Veens

-- Define the hypotheses
variable (h1 : Mem ⊆ En) -- All Mems are Ens
variable (h2 : ∃ x, x ∈ En ∩ Veen) -- Some Ens are Veens

-- Define the conclusions to be proved
def conclusion1 : Prop := ∃ x, x ∈ Mem ∩ Veen -- Some Mems are Veens
def conclusion2 : Prop := ∃ x, x ∈ Veen \ Mem -- Some Veens are not Mems

-- Theorem statement
theorem syllogism_conclusion (U : Type) (Mem En Veen : Set U) 
  (h1 : Mem ⊆ En) (h2 : ∃ x, x ∈ En ∩ Veen) : 
  conclusion1 U Mem Veen ∧ conclusion2 U Mem Veen := by
  sorry

end syllogism_conclusion_l2238_223866


namespace cafeteria_green_apples_l2238_223847

theorem cafeteria_green_apples :
  ∀ (red_apples students_wanting_fruit extra_apples green_apples : ℕ),
    red_apples = 25 →
    students_wanting_fruit = 10 →
    extra_apples = 32 →
    red_apples + green_apples - students_wanting_fruit = extra_apples →
    green_apples = 17 := by
  sorry

end cafeteria_green_apples_l2238_223847


namespace paperboy_delivery_12_l2238_223849

def paperboy_delivery (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | m + 4 => paperboy_delivery m + paperboy_delivery (m + 1) + paperboy_delivery (m + 2) + paperboy_delivery (m + 3)

theorem paperboy_delivery_12 : paperboy_delivery 12 = 2873 := by
  sorry

end paperboy_delivery_12_l2238_223849


namespace win_sector_area_l2238_223864

/-- Given a circular spinner with radius 8 cm and a probability of winning 1/4,
    prove that the area of the WIN sector is 16π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 1/4) :
  p * π * r^2 = 16 * π := by
  sorry

end win_sector_area_l2238_223864


namespace min_value_sum_reciprocals_l2238_223890

theorem min_value_sum_reciprocals (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_one : x + y + z + w = 1) :
  1/(x+y) + 1/(x+z) + 1/(x+w) + 1/(y+z) + 1/(y+w) + 1/(z+w) ≥ 18 ∧
  (1/(x+y) + 1/(x+z) + 1/(x+w) + 1/(y+z) + 1/(y+w) + 1/(z+w) = 18 ↔ x = 1/4 ∧ y = 1/4 ∧ z = 1/4 ∧ w = 1/4) :=
by sorry

end min_value_sum_reciprocals_l2238_223890


namespace max_value_of_function_sum_of_powers_greater_than_one_l2238_223879

-- Part 1
theorem max_value_of_function (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∃ M : ℝ, M = 1 ∧ ∀ x > -1, (1 + x)^a - a * x ≤ M :=
sorry

-- Part 2
theorem sum_of_powers_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^b + b^a > 1 :=
sorry

end max_value_of_function_sum_of_powers_greater_than_one_l2238_223879


namespace total_cost_of_three_items_l2238_223874

/-- The total cost of three items is the sum of their individual costs -/
theorem total_cost_of_three_items
  (cost_watch : ℕ)
  (cost_bracelet : ℕ)
  (cost_necklace : ℕ)
  (h1 : cost_watch = 144)
  (h2 : cost_bracelet = 250)
  (h3 : cost_necklace = 190) :
  cost_watch + cost_bracelet + cost_necklace = 584 :=
by sorry

end total_cost_of_three_items_l2238_223874


namespace billys_age_l2238_223860

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe)
  (h2 : billy + joe = 60)
  (h3 : billy > 30) :
  billy = 45 := by
sorry

end billys_age_l2238_223860


namespace range_of_3x_minus_2y_l2238_223876

theorem range_of_3x_minus_2y (x y : ℝ) 
  (h1 : -1 ≤ x + y ∧ x + y ≤ 1) 
  (h2 : 1 ≤ x - y ∧ x - y ≤ 5) : 
  2 ≤ 3*x - 2*y ∧ 3*x - 2*y ≤ 13 := by
  sorry

end range_of_3x_minus_2y_l2238_223876


namespace rationalize_denominator_sqrt35_l2238_223820

theorem rationalize_denominator_sqrt35 : 
  (35 : ℝ) / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end rationalize_denominator_sqrt35_l2238_223820


namespace sum_distinct_digits_mod_1000_l2238_223863

/-- The sum of all four-digit positive integers with distinct digits -/
def T : ℕ := sorry

/-- Predicate to check if a number has distinct digits -/
def has_distinct_digits (n : ℕ) : Prop := sorry

theorem sum_distinct_digits_mod_1000 : 
  T % 1000 = 400 :=
sorry

end sum_distinct_digits_mod_1000_l2238_223863


namespace moles_of_CO₂_equals_two_l2238_223869

/-- Represents a chemical compound --/
inductive Compound
| HNO₃
| NaHCO₃
| NH₄Cl
| NaNO₃
| H₂O
| CO₂
| NH₄NO₃
| HCl

/-- Represents a chemical reaction --/
structure Reaction :=
(reactants : List (Compound × ℕ))
(products : List (Compound × ℕ))

/-- Represents the two-step reaction process --/
def two_step_reaction : List Reaction :=
[
  { reactants := [(Compound.NaHCO₃, 1), (Compound.HNO₃, 1)],
    products := [(Compound.NaNO₃, 1), (Compound.H₂O, 1), (Compound.CO₂, 1)] },
  { reactants := [(Compound.NH₄Cl, 1), (Compound.HNO₃, 1)],
    products := [(Compound.NH₄NO₃, 1), (Compound.HCl, 1)] }
]

/-- Initial amounts of compounds --/
def initial_amounts : List (Compound × ℕ) :=
[(Compound.HNO₃, 2), (Compound.NaHCO₃, 2), (Compound.NH₄Cl, 1)]

/-- Calculates the moles of CO₂ formed in the two-step reaction --/
def moles_of_CO₂_formed (reactions : List Reaction) (initial : List (Compound × ℕ)) : ℕ :=
sorry

/-- Theorem stating that the moles of CO₂ formed is 2 --/
theorem moles_of_CO₂_equals_two :
  moles_of_CO₂_formed two_step_reaction initial_amounts = 2 :=
sorry

end moles_of_CO₂_equals_two_l2238_223869


namespace positive_integer_pairs_l2238_223809

theorem positive_integer_pairs : 
  ∀ (a b : ℕ+), 
    (∃ (k : ℕ+), k * a = b^4 + 1) → 
    (∃ (l : ℕ+), l * b = a^4 + 1) → 
    (Int.floor (Real.sqrt a.val) = Int.floor (Real.sqrt b.val)) → 
    ((a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1)) := by
  sorry

end positive_integer_pairs_l2238_223809


namespace x_mod_105_l2238_223832

theorem x_mod_105 (x : ℤ) 
  (h1 : (3 + x) % (3^3) = 2^2 % (3^3))
  (h2 : (5 + x) % (5^3) = 3^2 % (5^3))
  (h3 : (7 + x) % (7^3) = 5^2 % (7^3)) :
  x % 105 = 4 := by
  sorry

end x_mod_105_l2238_223832


namespace value_of_expression_l2238_223840

theorem value_of_expression (x : ℝ) (h : x^2 - 2*x = 3) : 3*x^2 - 6*x - 4 = 5 := by
  sorry

end value_of_expression_l2238_223840


namespace trajectory_is_parabola_l2238_223826

/-- A circle that passes through a fixed point and is tangent to a line -/
structure MovingCircle where
  center : ℝ × ℝ
  passes_through : center.1^2 + (center.2 - 1)^2 = (center.2 + 1)^2

/-- The trajectory of the center of the moving circle -/
def trajectory (c : MovingCircle) : Prop :=
  c.center.1^2 = 4 * c.center.2

/-- Theorem stating that the trajectory of the center is x^2 = 4y -/
theorem trajectory_is_parabola (c : MovingCircle) : trajectory c := by
  sorry

end trajectory_is_parabola_l2238_223826


namespace trigonometric_equation_solution_l2238_223871

theorem trigonometric_equation_solution (k : ℤ) :
  (∃ x : ℝ, 4 - Real.sin x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) - Real.cos (7 * x) ^ 2 = 
   Real.cos (Real.pi * k / 2021) ^ 2) ↔ 
  (∃ m : ℤ, k = 2021 * m) ∧ 
  (∀ x : ℝ, 4 - Real.sin x ^ 2 + Real.cos (4 * x) + Real.cos (2 * x) + 
   2 * Real.sin (3 * x) * Real.sin (7 * x) - Real.cos (7 * x) ^ 2 = 
   Real.cos (Real.pi * k / 2021) ^ 2 → 
   ∃ n : ℤ, x = Real.pi / 4 + n * Real.pi / 2) :=
by sorry

end trigonometric_equation_solution_l2238_223871


namespace at_least_one_non_negative_l2238_223894

theorem at_least_one_non_negative (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) 
  (h₁ : a₁ ≠ 0) (h₂ : a₂ ≠ 0) (h₃ : a₃ ≠ 0) (h₄ : a₄ ≠ 0) 
  (h₅ : a₅ ≠ 0) (h₆ : a₆ ≠ 0) (h₇ : a₇ ≠ 0) (h₈ : a₈ ≠ 0) : 
  max (a₁ * a₃ + a₂ * a₄) (max (a₁ * a₅ + a₂ * a₆) (max (a₁ * a₇ + a₂ * a₈) 
    (max (a₃ * a₅ + a₄ * a₆) (max (a₃ * a₇ + a₄ * a₈) (a₅ * a₇ + a₆ * a₈))))) ≥ 0 :=
by sorry

end at_least_one_non_negative_l2238_223894


namespace theater_ticket_sales_l2238_223895

/-- Theater ticket sales problem -/
theorem theater_ticket_sales
  (orchestra_price : ℕ)
  (balcony_price : ℕ)
  (total_tickets : ℕ)
  (balcony_orchestra_diff : ℕ)
  (ho : orchestra_price = 12)
  (hb : balcony_price = 8)
  (ht : total_tickets = 340)
  (hd : balcony_orchestra_diff = 40) :
  let orchestra_tickets := (total_tickets - balcony_orchestra_diff) / 2
  let balcony_tickets := total_tickets - orchestra_tickets
  orchestra_tickets * orchestra_price + balcony_tickets * balcony_price = 3320 :=
by sorry

end theater_ticket_sales_l2238_223895


namespace circle_ratio_after_increase_l2238_223807

theorem circle_ratio_after_increase (r : ℝ) : 
  let new_radius : ℝ := r + 1
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end circle_ratio_after_increase_l2238_223807


namespace tangent_line_implies_a_minus_b_zero_l2238_223859

-- Define the curve
def curve (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem tangent_line_implies_a_minus_b_zero (a b : ℝ) :
  (∃ x y, tangent_line x y ∧ y = curve a b x) →
  (tangent_line 0 b) →
  a - b = 0 :=
by sorry

end tangent_line_implies_a_minus_b_zero_l2238_223859


namespace sum_of_integers_l2238_223846

theorem sum_of_integers (a b : ℤ) (h : 6 * a * b = 9 * a - 10 * b + 16) : a + b = -1 := by
  sorry

end sum_of_integers_l2238_223846


namespace inequality_proof_l2238_223817

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) :
  (a^5 - 1) / (a^4 - 1) * (b^5 - 1) / (b^4 - 1) > 25/64 * (a + 1) * (b + 1) := by
  sorry

end inequality_proof_l2238_223817


namespace intersection_equals_interval_l2238_223801

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}
def N : Set ℝ := {x | 2*x - x^2 ≥ 0}

-- Define the open interval (1, 2]
def open_closed_interval : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem intersection_equals_interval : M ∩ N = open_closed_interval := by sorry

end intersection_equals_interval_l2238_223801


namespace like_terms_exponent_sum_l2238_223850

theorem like_terms_exponent_sum (m n : ℤ) : 
  (m + 2 = 6 ∧ n + 1 = 3) → (-m)^3 + n^2 = -60 := by
  sorry

end like_terms_exponent_sum_l2238_223850


namespace paint_set_cost_l2238_223891

def total_spent : ℕ := 80
def num_classes : ℕ := 6
def folders_per_class : ℕ := 1
def pencils_per_class : ℕ := 3
def pencils_per_eraser : ℕ := 6
def folder_cost : ℕ := 6
def pencil_cost : ℕ := 2
def eraser_cost : ℕ := 1

def total_folders : ℕ := num_classes * folders_per_class
def total_pencils : ℕ := num_classes * pencils_per_class
def total_erasers : ℕ := total_pencils / pencils_per_eraser

def supplies_cost : ℕ := 
  total_folders * folder_cost + 
  total_pencils * pencil_cost + 
  total_erasers * eraser_cost

theorem paint_set_cost : total_spent - supplies_cost = 5 := by
  sorry

end paint_set_cost_l2238_223891


namespace a_range_l2238_223844

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ 2^x

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 4*x + a = 0

-- Theorem statement
theorem a_range (a : ℝ) (hp : p a) (hq : q a) : 2 ≤ a ∧ a ≤ 4 := by
  sorry

end a_range_l2238_223844


namespace connie_calculation_l2238_223889

theorem connie_calculation (x : ℝ) : 4 * x = 200 → x / 4 + 10 = 22.5 := by
  sorry

end connie_calculation_l2238_223889


namespace max_abs_z_on_circle_l2238_223828

open Complex

theorem max_abs_z_on_circle (z : ℂ) : 
  (abs (z - 2*I) = 1) → (abs z ≤ 3) ∧ ∃ w : ℂ, abs (w - 2*I) = 1 ∧ abs w = 3 :=
by sorry

end max_abs_z_on_circle_l2238_223828


namespace point_in_region_l2238_223885

def satisfies_inequality (x y : ℝ) : Prop := 3 + 2*y < 6

theorem point_in_region :
  satisfies_inequality 1 1 ∧
  ¬(satisfies_inequality 0 0 ∧ satisfies_inequality 0 2 ∧ satisfies_inequality 2 0) :=
by sorry

end point_in_region_l2238_223885


namespace disrespectful_polynomial_max_value_l2238_223893

/-- A quadratic polynomial with real coefficients and leading coefficient 1 -/
structure QuadraticPolynomial where
  b : ℝ
  c : ℝ

/-- Evaluation of a quadratic polynomial at a point x -/
def evaluate (q : QuadraticPolynomial) (x : ℝ) : ℝ :=
  x^2 + q.b * x + q.c

/-- A quadratic polynomial is disrespectful if q(q(x)) = 0 has exactly three distinct real roots -/
def isDisrespectful (q : QuadraticPolynomial) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    evaluate q (evaluate q x) = 0 ∧
    evaluate q (evaluate q y) = 0 ∧
    evaluate q (evaluate q z) = 0 ∧
    ∀ w : ℝ, evaluate q (evaluate q w) = 0 → w = x ∨ w = y ∨ w = z

theorem disrespectful_polynomial_max_value :
  ∀ q : QuadraticPolynomial, isDisrespectful q → evaluate q 2 ≤ 45/16 :=
sorry

end disrespectful_polynomial_max_value_l2238_223893


namespace max_prob_highest_second_l2238_223827

/-- Represents a player in the chess game -/
structure Player where
  winProb : ℝ
  winProb_pos : winProb > 0

/-- Represents the chess game with three players -/
structure ChessGame where
  p₁ : Player
  p₂ : Player
  p₃ : Player
  prob_order : p₃.winProb > p₂.winProb ∧ p₂.winProb > p₁.winProb

/-- Calculates the probability of winning two consecutive games given the order of players -/
def probTwoConsecutiveWins (game : ChessGame) (second : Player) : ℝ :=
  2 * (second.winProb * (game.p₁.winProb + game.p₂.winProb + game.p₃.winProb - second.winProb) - 
       2 * game.p₁.winProb * game.p₂.winProb * game.p₃.winProb)

/-- Theorem stating that the probability of winning two consecutive games is maximized 
    when the player with the highest winning probability is played second -/
theorem max_prob_highest_second (game : ChessGame) :
  probTwoConsecutiveWins game game.p₃ ≥ probTwoConsecutiveWins game game.p₂ ∧
  probTwoConsecutiveWins game game.p₃ ≥ probTwoConsecutiveWins game game.p₁ := by
  sorry


end max_prob_highest_second_l2238_223827


namespace johns_quarters_l2238_223854

theorem johns_quarters (quarters dimes nickels : ℕ) : 
  quarters + dimes + nickels = 63 →
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters = 22 :=
by
  sorry

end johns_quarters_l2238_223854


namespace horner_v3_value_l2238_223802

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 5x^5 + 6x^4 + x^2 + 3x + 2 -/
def f : List ℤ := [2, 3, 1, 0, 6, -5, 1]

/-- Theorem: Horner's method for f(x) at x = -2 gives v₃ = -40 -/
theorem horner_v3_value :
  let coeffs := f.take 4
  horner coeffs (-2) = -40 := by sorry

end horner_v3_value_l2238_223802


namespace prob_all_boys_prob_two_boys_one_girl_l2238_223872

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- The number of people to be selected -/
def select_num : ℕ := 3

/-- The probability of selecting 3 boys out of the total 6 people -/
theorem prob_all_boys : 
  (Nat.choose num_boys select_num : ℚ) / (Nat.choose total_people select_num) = 1 / 5 := by
  sorry

/-- The probability of selecting 2 boys and 1 girl out of the total 6 people -/
theorem prob_two_boys_one_girl : 
  ((Nat.choose num_boys 2 * Nat.choose num_girls 1) : ℚ) / (Nat.choose total_people select_num) = 3 / 5 := by
  sorry

end prob_all_boys_prob_two_boys_one_girl_l2238_223872
