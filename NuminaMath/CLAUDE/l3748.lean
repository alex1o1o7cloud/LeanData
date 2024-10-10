import Mathlib

namespace sum_60_is_neg_300_l3748_374842

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  /-- The first term of the progression -/
  a : ℚ
  /-- The common difference of the progression -/
  d : ℚ
  /-- The sum of the first 15 terms is 150 -/
  sum_15 : (15 : ℚ) / 2 * (2 * a + 14 * d) = 150
  /-- The sum of the first 45 terms is 0 -/
  sum_45 : (45 : ℚ) / 2 * (2 * a + 44 * d) = 0

/-- The sum of the first 60 terms of the arithmetic progression is -300 -/
theorem sum_60_is_neg_300 (ap : ArithmeticProgression) :
  (60 : ℚ) / 2 * (2 * ap.a + 59 * ap.d) = -300 := by
  sorry


end sum_60_is_neg_300_l3748_374842


namespace geometric_sequence_fourth_term_l3748_374818

theorem geometric_sequence_fourth_term
  (a₁ a₅ : ℝ)
  (h₁ : a₁ = 5)
  (h₂ : a₅ = 10240)
  (h₃ : ∃ (r : ℝ), ∀ (n : ℕ), n ≤ 5 → a₁ * r^(n-1) = a₅^((n-1)/4) * a₁^(1-(n-1)/4)) :
  ∃ (a₄ : ℝ), a₄ = 2560 ∧ a₄ = a₁ * (a₅ / a₁)^(3/4) := by
  sorry

end geometric_sequence_fourth_term_l3748_374818


namespace cupcakes_frosted_proof_l3748_374869

/-- Cagney's frosting rate in cupcakes per second -/
def cagney_rate : ℚ := 1 / 25

/-- Lacey's frosting rate in cupcakes per second -/
def lacey_rate : ℚ := 1 / 35

/-- Total working time in seconds -/
def total_time : ℕ := 600

/-- The number of cupcakes frosted when working together -/
def cupcakes_frosted : ℕ := 41

theorem cupcakes_frosted_proof :
  ⌊(cagney_rate + lacey_rate) * total_time⌋ = cupcakes_frosted := by
  sorry

end cupcakes_frosted_proof_l3748_374869


namespace min_k_value_k_range_l3748_374809

/-- Given that for all a ∈ (-∞, 0) and all x ∈ (0, +∞), 
    the inequality x^2 + (3-a)x + 3 - 2a^2 < ke^x holds,
    prove that the minimum value of k is 3. -/
theorem min_k_value (k : ℝ) : 
  (∀ a < 0, ∀ x > 0, x^2 + (3-a)*x + 3 - 2*a^2 < k * Real.exp x) → 
  k ≥ 3 := by
  sorry

/-- The range of k is [3, +∞) -/
theorem k_range (k : ℝ) : 
  (∀ a < 0, ∀ x > 0, x^2 + (3-a)*x + 3 - 2*a^2 < k * Real.exp x) ↔ 
  k ≥ 3 := by
  sorry

end min_k_value_k_range_l3748_374809


namespace questions_left_blank_l3748_374898

/-- Represents the math test structure and Steve's performance --/
structure MathTest where
  totalQuestions : ℕ
  wordProblems : ℕ
  addSubProblems : ℕ
  algebraProblems : ℕ
  geometryProblems : ℕ
  totalTime : ℕ
  timePerWordProblem : ℚ
  timePerAddSubProblem : ℚ
  timePerAlgebraProblem : ℕ
  timePerGeometryProblem : ℕ
  wordProblemsAnswered : ℕ
  addSubProblemsAnswered : ℕ
  algebraProblemsAnswered : ℕ
  geometryProblemsAnswered : ℕ

/-- Theorem stating the number of questions left blank --/
theorem questions_left_blank (test : MathTest)
  (h1 : test.totalQuestions = 60)
  (h2 : test.wordProblems = 20)
  (h3 : test.addSubProblems = 25)
  (h4 : test.algebraProblems = 10)
  (h5 : test.geometryProblems = 5)
  (h6 : test.totalTime = 90)
  (h7 : test.timePerWordProblem = 2)
  (h8 : test.timePerAddSubProblem = 3/2)
  (h9 : test.timePerAlgebraProblem = 3)
  (h10 : test.timePerGeometryProblem = 4)
  (h11 : test.wordProblemsAnswered = 15)
  (h12 : test.addSubProblemsAnswered = 22)
  (h13 : test.algebraProblemsAnswered = 8)
  (h14 : test.geometryProblemsAnswered = 3) :
  test.totalQuestions - (test.wordProblemsAnswered + test.addSubProblemsAnswered + test.algebraProblemsAnswered + test.geometryProblemsAnswered) = 12 := by
  sorry

end questions_left_blank_l3748_374898


namespace quadratic_inequality_range_l3748_374813

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 := by
  sorry

end quadratic_inequality_range_l3748_374813


namespace yard_area_l3748_374830

/-- The area of a rectangular yard with a rectangular cutout -/
theorem yard_area (length width cutout_length cutout_width : ℝ) 
  (h1 : length = 20)
  (h2 : width = 15)
  (h3 : cutout_length = 4)
  (h4 : cutout_width = 2) :
  length * width - cutout_length * cutout_width = 292 := by
  sorry

end yard_area_l3748_374830


namespace lost_episodes_proof_l3748_374851

/-- Represents the number of episodes lost per season after a computer failure --/
def episodes_lost_per_season (series1_seasons series2_seasons episodes_per_season remaining_episodes : ℕ) : ℕ :=
  let total_episodes := (series1_seasons + series2_seasons) * episodes_per_season
  let lost_episodes := total_episodes - remaining_episodes
  lost_episodes / (series1_seasons + series2_seasons)

/-- Theorem stating that given the problem conditions, 2 episodes were lost per season --/
theorem lost_episodes_proof :
  episodes_lost_per_season 12 14 16 364 = 2 := by
  sorry

end lost_episodes_proof_l3748_374851


namespace carla_food_bank_theorem_l3748_374820

/-- Represents the food bank scenario with Carla --/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day1_restock : ℕ
  day2_people : ℕ
  day2_cans_per_person : ℕ
  day2_restock : ℕ

/-- Calculates the total number of cans given away --/
def total_cans_given_away (fb : FoodBank) : ℕ :=
  fb.day1_people * fb.day1_cans_per_person + fb.day2_people * fb.day2_cans_per_person

/-- Theorem stating that the total cans given away is 2500 --/
theorem carla_food_bank_theorem (fb : FoodBank) 
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day1_restock = 1500)
  (h5 : fb.day2_people = 1000)
  (h6 : fb.day2_cans_per_person = 2)
  (h7 : fb.day2_restock = 3000) :
  total_cans_given_away fb = 2500 := by
  sorry

end carla_food_bank_theorem_l3748_374820


namespace sum_equals_932_l3748_374856

-- Define the value of a number in a given base
def value_in_base (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

-- Define the given numbers
def num1 : ℕ := value_in_base [3, 5, 1] 7
def num2 : ℕ := value_in_base [13, 12, 4] 13

-- Theorem to prove
theorem sum_equals_932 : num1 + num2 = 932 := by
  sorry

end sum_equals_932_l3748_374856


namespace complement_implies_a_value_l3748_374839

def I (a : ℤ) : Set ℤ := {2, 4, a^2 - a - 3}
def A (a : ℤ) : Set ℤ := {4, 1 - a}

theorem complement_implies_a_value (a : ℤ) :
  (I a) \ (A a) = {-1} → a = -1 := by
  sorry

end complement_implies_a_value_l3748_374839


namespace probability_of_perfect_square_l3748_374819

theorem probability_of_perfect_square :
  ∀ (P : ℝ),
  (P * 50 + 3 * P * 50 = 1) →
  (∃ (perfect_squares_le_50 perfect_squares_gt_50 : ℕ),
    perfect_squares_le_50 = 7 ∧ perfect_squares_gt_50 = 3) →
  (perfect_squares_le_50 * P + perfect_squares_gt_50 * 3 * P) / 100 = 0.08 :=
by sorry

end probability_of_perfect_square_l3748_374819


namespace stream_speed_calculation_l3748_374822

/-- Given a boat traveling downstream, calculates the speed of the stream. -/
theorem stream_speed_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) 
  (h1 : boat_speed = 13)
  (h2 : downstream_distance = 69)
  (h3 : downstream_time = 3.6315789473684212) :
  let downstream_speed := downstream_distance / downstream_time
  let stream_speed := downstream_speed - boat_speed
  stream_speed = 6 := by sorry

end stream_speed_calculation_l3748_374822


namespace system_of_equations_solutions_l3748_374887

theorem system_of_equations_solutions :
  -- First system
  (∃ x y : ℝ, 4 * x - y = 1 ∧ y = 2 * x + 3 ∧ x = 2 ∧ y = 7) ∧
  -- Second system
  (∃ x y : ℝ, 2 * x - y = 5 ∧ 7 * x - 3 * y = 20 ∧ x = 5 ∧ y = 5) :=
by
  sorry

#check system_of_equations_solutions

end system_of_equations_solutions_l3748_374887


namespace martyrs_cemetery_distance_l3748_374863

/-- The distance from the school to the Martyrs' Cemetery in meters -/
def distance : ℝ := 180000

/-- The original speed of the car in meters per minute -/
def original_speed : ℝ := 500

/-- The scheduled travel time in minutes -/
def scheduled_time : ℝ := 120

theorem martyrs_cemetery_distance :
  ∃ (d : ℝ) (v : ℝ),
    d = distance ∧
    v = original_speed ∧
    -- Condition 1: Increased speed by 1/5 after 1 hour
    (60 / v + (d - 60 * v) / (6/5 * v) = scheduled_time - 10) ∧
    -- Condition 2: Increased speed by 1/3 after 60 km
    (60000 / v + (d - 60000) / (4/3 * v) = scheduled_time - 20) ∧
    -- Scheduled time is 2 hours
    scheduled_time = 120 :=
by sorry

end martyrs_cemetery_distance_l3748_374863


namespace square_cloth_trimming_l3748_374861

theorem square_cloth_trimming (x : ℝ) : 
  x > 0 →  -- Ensure positive length
  (x - 6) * (x - 5) = 120 → 
  x = 15 := by
sorry

end square_cloth_trimming_l3748_374861


namespace rectangular_prism_properties_l3748_374844

/-- A rectangular prism with dimensions 12, 16, and 21 inches has a diagonal length of 29 inches
    and a surface area of 1560 square inches. -/
theorem rectangular_prism_properties :
  let a : ℝ := 12
  let b : ℝ := 16
  let c : ℝ := 21
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let surface_area := 2 * (a*b + b*c + c*a)
  diagonal = 29 ∧ surface_area = 1560 := by
  sorry

#check rectangular_prism_properties

end rectangular_prism_properties_l3748_374844


namespace a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l3748_374884

theorem a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq :
  ¬(∀ a b : ℝ, a > b → a^2 > b^2) ∧ ¬(∀ a b : ℝ, a^2 > b^2 → a > b) := by
  sorry

end a_gt_b_not_sufficient_nor_necessary_for_a_sq_gt_b_sq_l3748_374884


namespace sum_reciprocal_inequality_l3748_374871

theorem sum_reciprocal_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c = 1/a + 1/b + 1/c) : a + b + c ≥ 3/(a*b*c) := by
  sorry

end sum_reciprocal_inequality_l3748_374871


namespace donation_relationship_l3748_374843

/-- Represents the relationship between the number of girls and the total donation in a class. -/
def donation_function (x : ℕ) : ℝ :=
  -5 * x + 1125

/-- Theorem stating the relationship between the number of girls and the total donation. -/
theorem donation_relationship (x : ℕ) (y : ℝ) 
  (h1 : x ≤ 45)  -- Ensure the number of girls is not more than the total number of students
  (h2 : y = 20 * x + 25 * (45 - x)) :  -- Total donation calculation
  y = donation_function x :=
by
  sorry

#check donation_relationship

end donation_relationship_l3748_374843


namespace ellen_lego_count_l3748_374833

/-- Calculates the final number of legos Ellen has after a series of transactions -/
def final_lego_count (initial : ℕ) : ℕ :=
  let after_week1 := initial - initial / 5
  let after_week2 := after_week1 + after_week1 / 4
  let after_week3 := after_week2 - 57
  after_week3 + after_week3 / 10

/-- Theorem stating that Ellen ends up with 355 legos -/
theorem ellen_lego_count : final_lego_count 380 = 355 := by
  sorry


end ellen_lego_count_l3748_374833


namespace smallest_number_range_l3748_374811

theorem smallest_number_range 
  (a b c d e : ℝ) 
  (h_distinct : a < b ∧ b < c ∧ c < d ∧ d < e) 
  (h_sum1 : a + b = 20) 
  (h_sum2 : a + c = 200) 
  (h_sum3 : d + e = 2014) 
  (h_sum4 : c + e = 2000) : 
  -793 < a ∧ a < 10 := by
sorry

end smallest_number_range_l3748_374811


namespace tetrahedron_colorings_l3748_374873

/-- Represents a coloring of the tetrahedron -/
def Coloring := Fin 7 → Bool

/-- The group of rotational symmetries of a tetrahedron -/
def TetrahedronSymmetry : Type := Unit -- Placeholder, actual implementation would be more complex

/-- Action of a symmetry on a coloring -/
def symmetryAction (s : TetrahedronSymmetry) (c : Coloring) : Coloring :=
  sorry

/-- A coloring is considered fixed under a symmetry if it's unchanged by the symmetry's action -/
def isFixed (s : TetrahedronSymmetry) (c : Coloring) : Prop :=
  symmetryAction s c = c

/-- The number of distinct colorings under rotational symmetry -/
def numDistinctColorings : ℕ :=
  sorry

theorem tetrahedron_colorings : numDistinctColorings = 48 := by
  sorry

end tetrahedron_colorings_l3748_374873


namespace infinite_solutions_imply_b_value_l3748_374855

theorem infinite_solutions_imply_b_value :
  ∀ b : ℚ, (∀ x : ℚ, 4 * (3 * x - b) = 3 * (4 * x + 7)) → b = -21/4 := by
  sorry

end infinite_solutions_imply_b_value_l3748_374855


namespace sum_of_cubes_and_values_positive_l3748_374885

theorem sum_of_cubes_and_values_positive (a b c : ℝ) 
  (hab : a + b > 0) (hac : a + c > 0) (hbc : b + c > 0) : 
  (a^3 + a) + (b^3 + b) + (c^3 + c) > 0 := by
  sorry

end sum_of_cubes_and_values_positive_l3748_374885


namespace total_dogs_l3748_374824

/-- The number of dogs that can fetch -/
def fetch : ℕ := 55

/-- The number of dogs that can roll over -/
def roll : ℕ := 32

/-- The number of dogs that can play dead -/
def play : ℕ := 40

/-- The number of dogs that can fetch and roll over -/
def fetch_roll : ℕ := 20

/-- The number of dogs that can fetch and play dead -/
def fetch_play : ℕ := 18

/-- The number of dogs that can roll over and play dead -/
def roll_play : ℕ := 15

/-- The number of dogs that can do all three tricks -/
def all_tricks : ℕ := 12

/-- The number of dogs that can do no tricks -/
def no_tricks : ℕ := 14

/-- Theorem stating the total number of dogs in the center -/
theorem total_dogs : 
  fetch + roll + play - fetch_roll - fetch_play - roll_play + all_tricks + no_tricks = 100 := by
  sorry

end total_dogs_l3748_374824


namespace triangle_side_length_l3748_374894

/-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively,
    if a = 4√5, b = 5, and cos A = 3/5, then c = 11. -/
theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 4 * Real.sqrt 5 →
  b = 5 →
  Real.cos A = 3 / 5 →
  c = 11 := by
  sorry

end triangle_side_length_l3748_374894


namespace count_monomials_l3748_374889

-- Define what a monomial is
def is_monomial (term : String) : Bool :=
  match term with
  | "0" => true  -- 0 is considered a monomial
  | t => (t.count '+' = 0) ∧ (t.count '-' ≤ 1) ∧ (t.count '/' = 0)  -- Simplified check for monomials

-- Define the list of terms in the expression
def expression : List String := ["1/x", "x+y", "0", "-a", "-3x^2y", "(x+1)/3"]

-- State the theorem
theorem count_monomials : 
  (expression.filter is_monomial).length = 3 := by sorry

end count_monomials_l3748_374889


namespace same_range_implies_b_constraint_l3748_374823

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 2

-- Define the function g
def g (b : ℝ) (x : ℝ) : ℝ := f b (f b x)

-- State the theorem
theorem same_range_implies_b_constraint (b : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f b x = y) ↔ (∀ y : ℝ, ∃ x : ℝ, g b x = y) →
  b ≥ 4 ∨ b ≤ -2 :=
sorry

end same_range_implies_b_constraint_l3748_374823


namespace orange_seller_gain_percentage_l3748_374896

theorem orange_seller_gain_percentage 
  (loss_rate : ℝ) 
  (initial_sale_quantity : ℝ) 
  (new_sale_quantity : ℝ) 
  (loss_percentage : ℝ) : 
  loss_rate = 0.1 → 
  initial_sale_quantity = 10 → 
  new_sale_quantity = 6 → 
  loss_percentage = 10 → 
  ∃ (G : ℝ), G = 50 ∧ 
    (1 + G / 100) * (1 - loss_rate) * initial_sale_quantity / new_sale_quantity = 1 := by
  sorry

end orange_seller_gain_percentage_l3748_374896


namespace rate_of_discount_l3748_374834

/-- Calculate the rate of discount given the marked price and selling price -/
theorem rate_of_discount (marked_price selling_price : ℝ) 
  (h1 : marked_price = 150)
  (h2 : selling_price = 120) : 
  (marked_price - selling_price) / marked_price * 100 = 20 := by
  sorry

end rate_of_discount_l3748_374834


namespace sector_area_l3748_374808

theorem sector_area (θ : Real) (L : Real) (A : Real) :
  θ = π / 6 →
  L = 2 * π / 3 →
  A = 4 * π / 3 :=
by sorry

end sector_area_l3748_374808


namespace girls_same_color_marble_l3748_374886

-- Define the total number of marbles
def total_marbles : ℕ := 4

-- Define the number of white marbles
def white_marbles : ℕ := 2

-- Define the number of black marbles
def black_marbles : ℕ := 2

-- Define the number of girls selecting marbles
def girls : ℕ := 2

-- Define the probability of both girls selecting the same colored marble
def prob_same_color : ℚ := 1 / 3

-- Theorem statement
theorem girls_same_color_marble :
  (total_marbles = white_marbles + black_marbles) →
  (white_marbles = black_marbles) →
  (girls = 2) →
  (prob_same_color = 1 / 3) := by
sorry

end girls_same_color_marble_l3748_374886


namespace can_measure_fifteen_minutes_l3748_374825

/-- Represents an hourglass with a specific duration. -/
structure Hourglass where
  duration : ℕ

/-- Represents the state of measuring time with two hourglasses. -/
structure MeasurementState where
  time : ℕ
  hg7 : ℕ
  hg11 : ℕ

/-- Defines a single step in the measurement process. -/
inductive MeasurementStep
  | FlipHg7
  | FlipHg11
  | Wait

/-- Applies a measurement step to the current state. -/
def applyStep (state : MeasurementState) (step : MeasurementStep) : MeasurementState :=
  sorry

/-- Checks if the given sequence of steps results in exactly 15 minutes. -/
def measuresFifteenMinutes (steps : List MeasurementStep) : Prop :=
  sorry

/-- Theorem stating that it's possible to measure 15 minutes with 7 and 11-minute hourglasses. -/
theorem can_measure_fifteen_minutes :
  ∃ (steps : List MeasurementStep), measuresFifteenMinutes steps :=
  sorry

end can_measure_fifteen_minutes_l3748_374825


namespace sphere_triangle_distance_theorem_l3748_374870

/-- The distance between the center of a sphere and the plane of a right triangle tangent to the sphere. -/
def sphere_triangle_distance (sphere_radius : ℝ) (triangle_side1 triangle_side2 triangle_side3 : ℝ) : ℝ :=
  sorry

/-- Theorem stating the distance between the center of a sphere and the plane of a right triangle tangent to the sphere. -/
theorem sphere_triangle_distance_theorem :
  sphere_triangle_distance 10 8 15 17 = Real.sqrt 91 := by
  sorry

end sphere_triangle_distance_theorem_l3748_374870


namespace cos_4050_degrees_l3748_374815

theorem cos_4050_degrees : Real.cos (4050 * π / 180) = 0 := by
  sorry

end cos_4050_degrees_l3748_374815


namespace roots_opposite_signs_l3748_374802

theorem roots_opposite_signs (n : ℝ) : 
  n^2 + n - 1 = 0 → 
  ∃ (x : ℝ), x ≠ 0 ∧ 
    (x^2 + (n-2)*x) / (2*n*x - 4) = (n+1) / (n-1) ∧
    (-x^2 + (n-2)*(-x)) / (2*n*(-x) - 4) = (n+1) / (n-1) := by
  sorry

end roots_opposite_signs_l3748_374802


namespace base4_division_theorem_l3748_374853

/-- Converts a number from base 4 to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Performs division in base 4 --/
def divBase4 (a b : ℕ) : ℕ := sorry

theorem base4_division_theorem :
  let dividend := 1302
  let divisor := 12
  let quotient := 103
  divBase4 dividend divisor = quotient := by sorry

end base4_division_theorem_l3748_374853


namespace expected_moves_is_six_l3748_374814

/-- Represents the state of the glasses --/
inductive GlassState
| Full
| Empty

/-- Represents the configuration of the 4 glasses --/
structure GlassConfig :=
(glass1 : GlassState)
(glass2 : GlassState)
(glass3 : GlassState)
(glass4 : GlassState)

/-- The initial configuration --/
def initialConfig : GlassConfig :=
{ glass1 := GlassState.Full,
  glass2 := GlassState.Empty,
  glass3 := GlassState.Full,
  glass4 := GlassState.Empty }

/-- The target configuration --/
def targetConfig : GlassConfig :=
{ glass1 := GlassState.Empty,
  glass2 := GlassState.Full,
  glass3 := GlassState.Empty,
  glass4 := GlassState.Full }

/-- Represents a valid move (pouring from a full glass to an empty one) --/
inductive ValidMove : GlassConfig → GlassConfig → Prop

/-- The expected number of moves to reach the target configuration --/
noncomputable def expectedMoves : ℝ := 6

/-- Main theorem: The expected number of moves from initial to target config is 6 --/
theorem expected_moves_is_six :
  expectedMoves = 6 :=
sorry

end expected_moves_is_six_l3748_374814


namespace integer_fraction_characterization_l3748_374837

def is_integer_fraction (m n : ℕ+) : Prop :=
  ∃ k : ℤ, (n.val ^ 3 + 1 : ℤ) = k * (m.val * n.val - 1)

def solution_set : Set (ℕ+ × ℕ+) :=
  {(2, 1), (3, 1), (1, 2), (2, 2), (5, 2), (1, 3), (5, 3), (3, 5)}

theorem integer_fraction_characterization :
  ∀ m n : ℕ+, is_integer_fraction m n ↔ (m, n) ∈ solution_set := by
  sorry

end integer_fraction_characterization_l3748_374837


namespace trigonometric_identity_l3748_374805

theorem trigonometric_identity (α : Real) 
  (h : (Real.sin (11 * Real.pi - α) - Real.cos (-α)) / Real.cos ((7 * Real.pi / 2) + α) = 3) : 
  (Real.tan α = -1/2) ∧ (Real.sin (2*α) + Real.cos (2*α) = -1/5) := by
  sorry

end trigonometric_identity_l3748_374805


namespace ten_people_no_adjacent_standing_prob_l3748_374803

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => validArrangements (n + 1) + validArrangements (n + 2)

/-- The probability of no two adjacent people standing up in a circular arrangement of n people, each flipping a fair coin. -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n : ℚ)

theorem ten_people_no_adjacent_standing_prob :
  noAdjacentStandingProb 10 = 123 / 1024 := by
  sorry

end ten_people_no_adjacent_standing_prob_l3748_374803


namespace pokemon_cards_distribution_l3748_374817

theorem pokemon_cards_distribution (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 432)
  (h2 : num_friends = 4)
  (h3 : total_cards % num_friends = 0)
  (h4 : (total_cards / num_friends) % 12 = 0) :
  (total_cards / num_friends) / 12 = 9 := by
  sorry

end pokemon_cards_distribution_l3748_374817


namespace middle_school_sample_size_l3748_374829

/-- Represents the number of schools to be sampled in a stratified sampling scenario -/
def stratified_sample (total : ℕ) (category : ℕ) (sample_size : ℕ) : ℕ :=
  (category * sample_size) / total

/-- Theorem stating the correct number of middle schools to be sampled -/
theorem middle_school_sample_size :
  let total_schools : ℕ := 700
  let middle_schools : ℕ := 200
  let sample_size : ℕ := 70
  stratified_sample total_schools middle_schools sample_size = 20 := by
  sorry


end middle_school_sample_size_l3748_374829


namespace base_4_minus_base_9_digits_l3748_374883

-- Define a function to calculate the number of digits in a given base
def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

-- State the theorem
theorem base_4_minus_base_9_digits : 
  num_digits 1024 4 - num_digits 1024 9 = 2 := by
  sorry

end base_4_minus_base_9_digits_l3748_374883


namespace circle_placement_l3748_374807

theorem circle_placement (rectangle_width : ℝ) (rectangle_height : ℝ) 
  (num_squares : ℕ) (square_size : ℝ) (circle_diameter : ℝ) :
  rectangle_width = 20 ∧ 
  rectangle_height = 25 ∧ 
  num_squares = 120 ∧ 
  square_size = 1 ∧ 
  circle_diameter = 1 →
  ∃ (x y : ℝ), 
    0 ≤ x ∧ x ≤ rectangle_width ∧ 
    0 ≤ y ∧ y ≤ rectangle_height ∧ 
    ∀ (i : ℕ), i < num_squares →
      ∃ (sx sy : ℝ), 
        0 ≤ sx ∧ sx + square_size ≤ rectangle_width ∧
        0 ≤ sy ∧ sy + square_size ≤ rectangle_height ∧
        (x - sx)^2 + (y - sy)^2 ≥ (circle_diameter / 2 + square_size / 2)^2 :=
by
  sorry

end circle_placement_l3748_374807


namespace geometric_figure_pieces_l3748_374864

/-- Calculates the sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the number of rows in the geometric figure -/
def num_rows : ℕ := 10

/-- Calculates the number of rods in the geometric figure -/
def num_rods : ℕ := 3 * triangular_number num_rows

/-- Calculates the number of connectors in the geometric figure -/
def num_connectors : ℕ := triangular_number (num_rows + 1)

/-- Calculates the number of unit squares in the geometric figure -/
def num_squares : ℕ := triangular_number num_rows

/-- The total number of pieces in the geometric figure -/
def total_pieces : ℕ := num_rods + num_connectors + num_squares

theorem geometric_figure_pieces :
  total_pieces = 286 :=
sorry

end geometric_figure_pieces_l3748_374864


namespace equation_solution_l3748_374832

theorem equation_solution : 
  let y : ℝ := -33/2
  ∀ x : ℝ, (8*x^2 + 78*x + 5) / (2*x + 19) = 4*x + 2 → x = y := by
  sorry

end equation_solution_l3748_374832


namespace disk_color_difference_l3748_374858

theorem disk_color_difference (total : ℕ) (blue_ratio yellow_ratio green_ratio : ℕ) :
  total = 144 →
  blue_ratio = 3 →
  yellow_ratio = 7 →
  green_ratio = 8 →
  let total_ratio := blue_ratio + yellow_ratio + green_ratio
  let disks_per_part := total / total_ratio
  let blue_disks := blue_ratio * disks_per_part
  let green_disks := green_ratio * disks_per_part
  green_disks - blue_disks = 40 :=
by
  sorry

end disk_color_difference_l3748_374858


namespace remainder_of_3_pow_2n_plus_8_l3748_374821

theorem remainder_of_3_pow_2n_plus_8 (n : ℕ) : (3^(2*n) + 8) % 8 = 1 := by
  sorry

end remainder_of_3_pow_2n_plus_8_l3748_374821


namespace square_sum_product_l3748_374845

theorem square_sum_product (x y : ℝ) (hx : x = Real.sqrt 5 + Real.sqrt 3) (hy : y = Real.sqrt 5 - Real.sqrt 3) :
  x^2 + x*y + y^2 = 18 := by
  sorry

end square_sum_product_l3748_374845


namespace sister_chromatid_separation_in_second_division_sister_chromatid_separation_not_in_other_stages_l3748_374848

/-- Represents the stages of meiosis --/
inductive MeiosisStage
  | Interphase
  | TetradFormation
  | FirstDivision
  | SecondDivision

/-- Represents the events that occur during meiosis --/
inductive MeiosisEvent
  | ChromosomeReplication
  | HomologousPairing
  | ChromatidSeparation

/-- Defines the characteristics of each meiosis stage --/
def stageCharacteristics : MeiosisStage → List MeiosisEvent
  | MeiosisStage.Interphase => [MeiosisEvent.ChromosomeReplication]
  | MeiosisStage.TetradFormation => [MeiosisEvent.HomologousPairing]
  | MeiosisStage.FirstDivision => []
  | MeiosisStage.SecondDivision => [MeiosisEvent.ChromatidSeparation]

/-- Theorem: Sister chromatid separation occurs during the second meiotic division --/
theorem sister_chromatid_separation_in_second_division :
  MeiosisEvent.ChromatidSeparation ∈ stageCharacteristics MeiosisStage.SecondDivision :=
by sorry

/-- Corollary: Sister chromatid separation does not occur in other stages --/
theorem sister_chromatid_separation_not_in_other_stages :
  ∀ stage, stage ≠ MeiosisStage.SecondDivision →
    MeiosisEvent.ChromatidSeparation ∉ stageCharacteristics stage :=
by sorry

end sister_chromatid_separation_in_second_division_sister_chromatid_separation_not_in_other_stages_l3748_374848


namespace grid_paths_6x5_10_l3748_374804

/-- The number of different paths on a grid --/
def grid_paths (width height path_length : ℕ) : ℕ :=
  Nat.choose path_length height

/-- Theorem: The number of different paths on a 6x5 grid with path length 10 is 210 --/
theorem grid_paths_6x5_10 :
  grid_paths 6 5 10 = 210 := by
  sorry

end grid_paths_6x5_10_l3748_374804


namespace chloe_final_score_is_86_l3748_374836

/-- Chloe's final score in a trivia game -/
def chloeFinalScore (firstRoundScore secondRoundScore lastRoundLoss : ℕ) : ℕ :=
  firstRoundScore + secondRoundScore - lastRoundLoss

/-- Theorem: Chloe's final score is 86 points -/
theorem chloe_final_score_is_86 :
  chloeFinalScore 40 50 4 = 86 := by
  sorry

#eval chloeFinalScore 40 50 4

end chloe_final_score_is_86_l3748_374836


namespace wickets_before_last_match_l3748_374875

/-- Represents the number of wickets taken before the last match -/
def W : ℕ := sorry

/-- The initial bowling average -/
def initial_average : ℚ := 12.4

/-- The number of wickets taken in the last match -/
def last_match_wickets : ℕ := 7

/-- The number of runs conceded in the last match -/
def last_match_runs : ℕ := 26

/-- The decrease in average after the last match -/
def average_decrease : ℚ := 0.4

/-- The new average after the last match -/
def new_average : ℚ := initial_average - average_decrease

theorem wickets_before_last_match :
  (initial_average * W + last_match_runs : ℚ) / (W + last_match_wickets) = new_average →
  W = 145 := by sorry

end wickets_before_last_match_l3748_374875


namespace square_side_length_l3748_374854

theorem square_side_length (rectangle_width : ℝ) (rectangle_length : ℝ) 
  (h1 : rectangle_width = 3)
  (h2 : rectangle_length = 3)
  (h3 : square_area = rectangle_width * rectangle_length) : 
  ∃ (square_side : ℝ), square_side^2 = square_area ∧ square_side = 3 := by
  sorry

end square_side_length_l3748_374854


namespace max_x0_value_l3748_374801

def max_x0 (x : Fin 1997 → ℝ) : Prop :=
  (∀ i, x i > 0) ∧
  x 0 = x 1995 ∧
  (∀ i ∈ Finset.range 1995, x i + 2 / x i = 2 * x (i + 1) + 1 / x (i + 1))

theorem max_x0_value (x : Fin 1997 → ℝ) (h : max_x0 x) :
  x 0 ≤ 2^997 ∧ ∃ y : Fin 1997 → ℝ, max_x0 y ∧ y 0 = 2^997 :=
sorry

end max_x0_value_l3748_374801


namespace program_cost_calculation_l3748_374847

-- Define constants
def millisecond_to_second : Real := 0.001
def minute_to_millisecond : Nat := 60000
def os_overhead_cost : Real := 1.07
def computer_time_cost_per_ms : Real := 0.023
def data_tape_cost : Real := 5.35
def memory_cost_per_mb : Real := 0.15
def electricity_cost_per_kwh : Real := 0.02
def program_runtime_minutes : Nat := 45
def program_memory_gb : Real := 3.5
def program_electricity_kwh : Real := 2
def gb_to_mb : Nat := 1024

-- Define the theorem
theorem program_cost_calculation :
  let total_milliseconds := program_runtime_minutes * minute_to_millisecond
  let computer_time_cost := total_milliseconds * computer_time_cost_per_ms
  let memory_usage_mb := program_memory_gb * gb_to_mb
  let memory_cost := memory_usage_mb * memory_cost_per_mb
  let electricity_cost := program_electricity_kwh * electricity_cost_per_kwh
  let total_cost := os_overhead_cost + computer_time_cost + data_tape_cost + memory_cost + electricity_cost
  total_cost = 62644.06 := by
  sorry

end program_cost_calculation_l3748_374847


namespace largest_special_number_l3748_374840

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_special_number : 
  let n := 4731
  ∀ m : ℕ, m > n → 
    ¬(m < 10000 ∧ 
      (∀ i j : ℕ, i < 4 → j < 4 → i ≠ j → (m / 10^i % 10) ≠ (m / 10^j % 10)) ∧
      is_prime (n / 100) ∧
      is_prime ((n / 1000) * 10 + (n % 10)) ∧
      is_prime ((n / 1000) * 10 + (n / 10 % 10)) ∧
      n % 3 = 0 ∧
      ¬(is_prime n)) :=
by sorry

end largest_special_number_l3748_374840


namespace line_through_points_l3748_374860

/-- Given two points A and D in 3D space, this theorem proves that the parametric equations
    of the line passing through these points are of the form x = -3 + 4t, y = 3t, z = 1 + t. -/
theorem line_through_points (A D : ℝ × ℝ × ℝ) (h : A = (-3, 0, 1) ∧ D = (1, 3, 2)) :
  ∃ (f : ℝ → ℝ × ℝ × ℝ), ∀ t : ℝ,
    f t = (-3 + 4*t, 3*t, 1 + t) ∧
    (∃ t₁ t₂ : ℝ, f t₁ = A ∧ f t₂ = D) :=
by sorry

end line_through_points_l3748_374860


namespace jennas_age_l3748_374841

/-- Given that Jenna is 5 years older than Darius, their ages sum to 21, and Darius is 8 years old,
    prove that Jenna is 13 years old. -/
theorem jennas_age (jenna_age darius_age : ℕ) 
  (h1 : jenna_age = darius_age + 5)
  (h2 : jenna_age + darius_age = 21)
  (h3 : darius_age = 8) :
  jenna_age = 13 := by
  sorry

end jennas_age_l3748_374841


namespace volcano_ash_height_l3748_374827

theorem volcano_ash_height (radius : ℝ) (height : ℝ) : 
  radius = 2700 → 2 * radius = 18 * height → height = 300 := by
  sorry

end volcano_ash_height_l3748_374827


namespace modulus_of_z_l3748_374890

/-- The modulus of the complex number z = 2/(1-i) + (1-i)^2 is equal to √2 -/
theorem modulus_of_z (i : ℂ) (h : i^2 = -1) :
  Complex.abs (2 / (1 - i) + (1 - i)^2) = Real.sqrt 2 := by
  sorry

end modulus_of_z_l3748_374890


namespace largest_solution_is_57_98_l3748_374876

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

/-- The equation from the problem -/
def equation (x : ℝ) : Prop := (floor x : ℝ) = 8 + 50 * (frac x)

/-- The theorem statement -/
theorem largest_solution_is_57_98 :
  ∃ (x : ℝ), equation x ∧ (∀ (y : ℝ), equation y → y ≤ x) ∧ x = 57.98 :=
sorry

end largest_solution_is_57_98_l3748_374876


namespace selling_price_equal_profit_loss_l3748_374865

def cost_price : ℕ := 49
def loss_price : ℕ := 42

def profit (selling_price : ℕ) : ℤ := selling_price - cost_price
def loss (selling_price : ℕ) : ℤ := cost_price - selling_price

theorem selling_price_equal_profit_loss : 
  ∃ (sp : ℕ), profit sp = loss loss_price ∧ profit sp > 0 := by
  sorry

end selling_price_equal_profit_loss_l3748_374865


namespace loser_received_35_percent_l3748_374849

/-- Given a total number of votes and the difference between winner and loser,
    calculate the percentage of votes received by the losing candidate. -/
def loser_vote_percentage (total_votes : ℕ) (vote_difference : ℕ) : ℚ :=
  (total_votes - vote_difference) / (2 * total_votes) * 100

/-- Theorem stating that given 4500 total votes and a 1350 vote difference,
    the losing candidate received 35% of the votes. -/
theorem loser_received_35_percent :
  loser_vote_percentage 4500 1350 = 35 := by
  sorry

end loser_received_35_percent_l3748_374849


namespace inequality_proof_l3748_374826

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a > b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
  sorry

end inequality_proof_l3748_374826


namespace correct_num_buckets_l3748_374831

/-- The number of crab buckets Tom has -/
def num_buckets : ℕ := 56

/-- The number of crabs in each bucket -/
def crabs_per_bucket : ℕ := 12

/-- The price of each crab in dollars -/
def price_per_crab : ℕ := 5

/-- Tom's weekly earnings in dollars -/
def weekly_earnings : ℕ := 3360

/-- Theorem stating that the number of crab buckets is correct -/
theorem correct_num_buckets : 
  num_buckets = weekly_earnings / (crabs_per_bucket * price_per_crab) := by
  sorry

end correct_num_buckets_l3748_374831


namespace arithmetic_sequence_ratio_l3748_374846

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1  -- Arithmetic property
  h_sum : ∀ n, S n = n * (a 1 + a n) / 2  -- Sum formula

/-- Theorem: For an arithmetic sequence with S₁ = 1 and S₄/S₂ = 4, S₆/S₄ = 9/4 -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence)
    (h1 : seq.S 1 = 1)
    (h2 : seq.S 4 / seq.S 2 = 4) :
  seq.S 6 / seq.S 4 = 9/4 := by
  sorry

end arithmetic_sequence_ratio_l3748_374846


namespace mother_triple_age_l3748_374810

/-- Represents the age difference between Serena and her mother -/
def age_difference : ℕ := 30

/-- Represents Serena's current age -/
def serena_age : ℕ := 9

/-- Represents the number of years until Serena's mother is three times as old as Serena -/
def years_until_triple : ℕ := 6

/-- Theorem stating that after 'years_until_triple' years, Serena's mother will be three times as old as Serena -/
theorem mother_triple_age :
  serena_age + years_until_triple = (serena_age + age_difference + years_until_triple) / 3 :=
sorry

end mother_triple_age_l3748_374810


namespace least_congruent_number_proof_l3748_374878

/-- The least five-digit positive integer congruent to 7 (mod 18) and 4 (mod 9) -/
def least_congruent_number : ℕ := 10012

theorem least_congruent_number_proof :
  (least_congruent_number ≥ 10000) ∧
  (least_congruent_number < 100000) ∧
  (least_congruent_number % 18 = 7) ∧
  (least_congruent_number % 9 = 4) ∧
  (∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 18 = 7 ∧ n % 9 = 4 → n ≥ least_congruent_number) :=
by sorry

end least_congruent_number_proof_l3748_374878


namespace exponent_division_l3748_374881

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^3 / a = a^2 :=
sorry

end exponent_division_l3748_374881


namespace total_wallpaper_removal_time_l3748_374806

-- Define the structure for a room
structure Room where
  name : String
  walls : Nat
  time_per_wall : List Float

-- Define the rooms
def dining_room : Room := { name := "Dining Room", walls := 3, time_per_wall := [1.5, 1.5, 1.5] }
def living_room : Room := { name := "Living Room", walls := 4, time_per_wall := [1, 1, 2.5, 2.5] }
def bedroom : Room := { name := "Bedroom", walls := 3, time_per_wall := [3, 3, 3] }
def hallway : Room := { name := "Hallway", walls := 5, time_per_wall := [4, 2, 2, 2, 2] }
def kitchen : Room := { name := "Kitchen", walls := 4, time_per_wall := [3, 1.5, 1.5, 2] }
def bathroom : Room := { name := "Bathroom", walls := 2, time_per_wall := [2, 3] }

-- Define the list of all rooms
def all_rooms : List Room := [dining_room, living_room, bedroom, hallway, kitchen, bathroom]

-- Function to calculate total time for a room
def room_time (room : Room) : Float :=
  room.time_per_wall.sum

-- Theorem: The total time to remove wallpaper from all rooms is 45.5 hours
theorem total_wallpaper_removal_time :
  (all_rooms.map room_time).sum = 45.5 := by
  sorry


end total_wallpaper_removal_time_l3748_374806


namespace solve_rental_problem_l3748_374867

def rental_problem (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : Prop :=
  daily_rate * days + mileage_rate * miles = 275

theorem solve_rental_problem :
  rental_problem 30 0.25 5 500 := by
  sorry

end solve_rental_problem_l3748_374867


namespace log_cutting_problem_l3748_374812

/-- Represents the number of cuts needed to split a log into 1-meter pieces -/
def cuts_needed (length : ℕ) : ℕ := length - 1

/-- Represents the total number of logs -/
def total_logs : ℕ := 30

/-- Represents the total length of all logs in meters -/
def total_length : ℕ := 100

/-- Represents the possible lengths of logs in meters -/
inductive LogLength
| short : LogLength  -- 3 meters
| long : LogLength   -- 4 meters

/-- Calculates the minimum number of cuts needed for the given log configuration -/
def min_cuts (x y : ℕ) : Prop :=
  x + y = total_logs ∧
  3 * x + 4 * y = total_length ∧
  x * cuts_needed 3 + y * cuts_needed 4 = 70

theorem log_cutting_problem :
  ∃ x y : ℕ, min_cuts x y :=
sorry

end log_cutting_problem_l3748_374812


namespace sams_initial_points_l3748_374879

theorem sams_initial_points :
  ∀ initial_points : ℕ,
  initial_points + 3 = 95 →
  initial_points = 92 :=
by
  sorry

end sams_initial_points_l3748_374879


namespace evaluate_fraction_l3748_374882

theorem evaluate_fraction : (0.4 ^ 4) / (0.04 ^ 3) = 400 := by
  sorry

end evaluate_fraction_l3748_374882


namespace exchange_theorem_l3748_374850

/-- Represents the number of exchanges between Xiao Zhang and Xiao Li -/
def num_exchanges : ℕ := 4

/-- Initial number of pencils Xiao Zhang has -/
def initial_pencils : ℕ := 200

/-- Initial number of fountain pens Xiao Li has -/
def initial_pens : ℕ := 20

/-- Number of pencils exchanged per transaction -/
def pencils_per_exchange : ℕ := 6

/-- Number of pens exchanged per transaction -/
def pens_per_exchange : ℕ := 1

/-- Ratio of pencils to pens after exchanges -/
def final_ratio : ℕ := 11

theorem exchange_theorem : 
  initial_pencils - num_exchanges * pencils_per_exchange = 
  final_ratio * (initial_pens - num_exchanges * pens_per_exchange) :=
by sorry


end exchange_theorem_l3748_374850


namespace train_crossing_time_l3748_374874

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 135 →
  train_speed_kmh = 54 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 9 := by
  sorry

#check train_crossing_time

end train_crossing_time_l3748_374874


namespace investment_growth_rate_l3748_374880

theorem investment_growth_rate (initial_investment : ℝ) (final_investment : ℝ) (x : ℝ) :
  initial_investment = 2500 ∧ 
  final_investment = 3600 ∧ 
  final_investment = initial_investment * (1 + x)^2 →
  2500 * (1 + x)^2 = 3600 :=
by sorry

end investment_growth_rate_l3748_374880


namespace homer_candy_crush_score_l3748_374800

theorem homer_candy_crush_score (first_try : ℕ) (second_try : ℕ) (third_try : ℕ) 
  (h1 : first_try = 400)
  (h2 : second_try < first_try)
  (h3 : third_try = 2 * second_try)
  (h4 : first_try + second_try + third_try = 1390) :
  second_try = 330 := by
  sorry

end homer_candy_crush_score_l3748_374800


namespace fraction_sum_equality_l3748_374828

theorem fraction_sum_equality (p q : ℚ) (h : p / q = 4 / 5) :
  4 / 7 + (2 * q - p) / (2 * q + p) = 1 := by sorry

end fraction_sum_equality_l3748_374828


namespace place_values_and_names_l3748_374893

/-- Represents a place in a base-10 positional number system -/
inductive Place : Nat → Type where
  | units : Place 1
  | next (n : Nat) : Place n → Place (n + 1)

/-- The value of a place in a base-10 positional number system -/
def placeValue : ∀ n, Place n → Nat
  | _, Place.units => 1
  | _, Place.next _ p => 10 * placeValue _ p

/-- The name of a place in a base-10 positional number system -/
def placeName : ∀ n, Place n → String
  | _, Place.units => "units"
  | _, Place.next _ p =>
    let prev := placeName _ p
    if prev = "units" then "tens"
    else if prev = "tens" then "hundreds"
    else if prev = "hundreds" then "thousands"
    else if prev = "thousands" then "ten thousands"
    else if prev = "ten thousands" then "hundred thousands"
    else if prev = "hundred thousands" then "millions"
    else if prev = "millions" then "ten millions"
    else if prev = "ten millions" then "hundred millions"
    else "billion"

theorem place_values_and_names :
  ∃ (fifth tenth : Nat) (p5 : Place fifth) (p10 : Place tenth),
    fifth = 5 ∧
    tenth = 10 ∧
    placeName _ p5 = "ten thousands" ∧
    placeName _ p10 = "billion" ∧
    placeValue _ p5 = 10000 ∧
    placeValue _ p10 = 1000000000 := by
  sorry

end place_values_and_names_l3748_374893


namespace justin_and_tim_games_l3748_374891

/-- The number of players in the league -/
def total_players : ℕ := 10

/-- The number of players in each game -/
def players_per_game : ℕ := 5

/-- The number of games where two specific players play together -/
def games_together : ℕ := 56

/-- The total number of possible game combinations -/
def total_combinations : ℕ := Nat.choose total_players players_per_game

theorem justin_and_tim_games :
  games_together = (players_per_game - 1) * total_combinations / (total_players - 1) :=
sorry

end justin_and_tim_games_l3748_374891


namespace matrix_equation_solution_l3748_374895

theorem matrix_equation_solution : 
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 0.5, 1]
  M^4 - 3 • M^3 + 2 • M^2 = !![6, 12; 3, 6] := by
  sorry

end matrix_equation_solution_l3748_374895


namespace debate_only_count_l3748_374859

/-- Represents the number of pupils in a class with debate and singing activities -/
structure ClassActivities where
  total : ℕ
  singing_only : ℕ
  both : ℕ
  debate_only : ℕ

/-- The number of pupils in debate only is 37 -/
theorem debate_only_count (c : ClassActivities) 
  (h1 : c.total = 55)
  (h2 : c.singing_only = 18)
  (h3 : c.both = 17)
  (h4 : c.total = c.debate_only + c.singing_only + c.both) : 
  c.debate_only = 37 := by
  sorry

end debate_only_count_l3748_374859


namespace chef_wage_difference_chef_earns_less_l3748_374866

def manager_wage : ℚ := 17/2

theorem chef_wage_difference : ℚ :=
  let dishwasher_wage := manager_wage / 2
  let chef_wage := dishwasher_wage * (1 + 1/4)
  manager_wage - chef_wage

theorem chef_earns_less (h : chef_wage_difference = 255/80) : True := by
  sorry

end chef_wage_difference_chef_earns_less_l3748_374866


namespace tan_alpha_value_implies_expression_value_l3748_374857

theorem tan_alpha_value_implies_expression_value (α : Real) 
  (h : Real.tan α = -1/2) : 
  (Real.sin (2 * α) + 2 * Real.cos (2 * α)) / 
  (4 * Real.cos (2 * α) - 4 * Real.sin (2 * α)) = 1/14 := by
  sorry

end tan_alpha_value_implies_expression_value_l3748_374857


namespace base_five_product_l3748_374816

/-- Converts a base 5 number to decimal --/
def baseToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 5 --/
def decimalToBase (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base_five_product :
  let a := [2, 3, 1] -- represents 132₅ in reverse order
  let b := [2, 1]    -- represents 12₅ in reverse order
  let product := [4, 3, 1, 2] -- represents 2134₅ in reverse order
  (baseToDecimal a) * (baseToDecimal b) = baseToDecimal product ∧
  decimalToBase ((baseToDecimal a) * (baseToDecimal b)) = product.reverse :=
sorry

end base_five_product_l3748_374816


namespace function_properties_l3748_374892

def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (-(x + 1))

def is_odd_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f (-(x + 2))

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def equation_solutions (f : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x = x ∧ f y = y ∧ f z = z ∧
    ∀ w, f w = w → w = x ∨ w = y ∨ w = z

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even_shifted f)
  (h2 : is_odd_shifted f)
  (h3 : ∀ x ∈ Set.Icc 0 1, f x = 2^x - 1) :
  symmetric_about f 1 ∧ equation_solutions f := by
sorry

end function_properties_l3748_374892


namespace divisibility_by_37_l3748_374872

theorem divisibility_by_37 : ∃ k : ℤ, 333^555 + 555^333 = 37 * k := by sorry

end divisibility_by_37_l3748_374872


namespace smallest_multiple_l3748_374899

theorem smallest_multiple (x : ℕ+) : (∀ y : ℕ+, 450 * y.val % 625 = 0 → x ≤ y) ∧ 450 * x.val % 625 = 0 := by
  sorry

end smallest_multiple_l3748_374899


namespace parabola_directrix_l3748_374838

/-- The equation of the directrix of the parabola y = 4x^2 is y = -1/16 -/
theorem parabola_directrix (x y : ℝ) : 
  y = 4 * x^2 → (∃ (k : ℝ), k = -1/16 ∧ y = k) :=
sorry

end parabola_directrix_l3748_374838


namespace range_of_a_l3748_374897

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 65

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, f a x > 0) → -16 < a ∧ a < 16 := by
  sorry

end range_of_a_l3748_374897


namespace airplane_seats_l3748_374868

theorem airplane_seats : ∃ (total : ℝ), 
  (30 : ℝ) + 0.2 * total + 0.75 * total = total ∧ total = 600 := by
  sorry

end airplane_seats_l3748_374868


namespace find_d_l3748_374862

theorem find_d : ∃ d : ℝ, 
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * n^2 + 19 * n - 84 = 0) ∧ 
  (5 * (d - ⌊d⌋)^2 - 26 * (d - ⌊d⌋) + 12 = 0) ∧ 
  (0 ≤ d - ⌊d⌋ ∧ d - ⌊d⌋ < 1) ∧
  d = 3.44 := by
  sorry

end find_d_l3748_374862


namespace coefficient_x_squared_l3748_374877

theorem coefficient_x_squared (n : ℕ) : 
  (2 : ℤ) * 4 * (Nat.choose 6 2) - 2 * (Nat.choose 6 1) + 1 = 109 := by
  sorry

#check coefficient_x_squared

end coefficient_x_squared_l3748_374877


namespace marble_arrangement_l3748_374888

/-- Represents the color of a marble -/
inductive Color
  | Green
  | Blue
  | Red

/-- Represents an arrangement of marbles -/
def Arrangement := List Color

/-- Checks if an arrangement satisfies the equal neighbor condition -/
def satisfiesCondition (arr : Arrangement) : Bool :=
  sorry

/-- Counts the number of valid arrangements for a given number of marbles -/
def countArrangements (totalMarbles : Nat) : Nat :=
  sorry

theorem marble_arrangement :
  let greenMarbles : Nat := 6
  let m : Nat := 12  -- maximum number of additional blue and red marbles
  let totalMarbles : Nat := greenMarbles + m
  let N : Nat := countArrangements totalMarbles
  N = 924 ∧ N % 1000 = 924 := by
  sorry

end marble_arrangement_l3748_374888


namespace smallest_b_not_prime_nine_satisfies_condition_nine_is_smallest_l3748_374852

theorem smallest_b_not_prime (b : ℕ) (h : b > 8) :
  (∀ x : ℤ, ¬ Prime (x^4 + b^2 : ℤ)) →
  b ≥ 9 :=
by sorry

theorem nine_satisfies_condition :
  ∀ x : ℤ, ¬ Prime (x^4 + 9^2 : ℤ) :=
by sorry

theorem nine_is_smallest :
  ∀ b : ℕ, b > 8 →
  (∀ x : ℤ, ¬ Prime (x^4 + b^2 : ℤ)) →
  b ≥ 9 :=
by sorry

end smallest_b_not_prime_nine_satisfies_condition_nine_is_smallest_l3748_374852


namespace sufficient_not_necessary_l3748_374835

theorem sufficient_not_necessary (a b : ℝ) : 
  (a < 0 ∧ -1 < b ∧ b < 0) → 
  (∀ x y : ℝ, x < 0 ∧ -1 < y ∧ y < 0 → x + x * y < 0) ∧
  ¬(∀ x y : ℝ, x + x * y < 0 → x < 0 ∧ -1 < y ∧ y < 0) :=
by sorry

end sufficient_not_necessary_l3748_374835
