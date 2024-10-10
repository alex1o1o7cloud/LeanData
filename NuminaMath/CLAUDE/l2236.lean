import Mathlib

namespace no_solution_to_equation_l2236_223677

theorem no_solution_to_equation :
  ¬ ∃ (x : ℝ), x ≠ 2 ∧ (1 / (x - 2) = (1 - x) / (2 - x) - 3) := by
  sorry

end no_solution_to_equation_l2236_223677


namespace one_tails_after_flips_l2236_223611

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a circular arrangement of coins -/
def CoinCircle (n : ℕ) := Fin (2*n+1) → CoinState

/-- The initial state of the coin circle where all coins show heads -/
def initialState (n : ℕ) : CoinCircle n :=
  λ _ => CoinState.Heads

/-- The position of the k-th flip in the circle -/
def flipPosition (n k : ℕ) : Fin (2*n+1) :=
  ⟨k * (k + 1) / 2, sorry⟩

/-- Applies a single flip to a coin state -/
def flipCoin : CoinState → CoinState
| CoinState.Heads => CoinState.Tails
| CoinState.Tails => CoinState.Heads

/-- Applies the flipping process to the coin circle -/
def applyFlips (n : ℕ) (state : CoinCircle n) : CoinCircle n :=
  sorry

/-- Counts the number of tails in the final state -/
def countTails (n : ℕ) (state : CoinCircle n) : ℕ :=
  sorry

/-- The main theorem stating that exactly one coin shows tails after the process -/
theorem one_tails_after_flips (n : ℕ) :
  countTails n (applyFlips n (initialState n)) = 1 :=
sorry

end one_tails_after_flips_l2236_223611


namespace trig_identity_l2236_223632

theorem trig_identity (x y : ℝ) : 
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.cos x ^ 2 := by
  sorry

end trig_identity_l2236_223632


namespace unique_solution_condition_l2236_223696

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, 3 * x - 5 + a = b * x + 1) ↔ b ≠ 3 := by sorry

end unique_solution_condition_l2236_223696


namespace chromium_percentage_calculation_l2236_223626

/-- The percentage of chromium in the first alloy -/
def chromium_percentage_first : ℝ := 10

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_second : ℝ := 6

/-- The weight of the first alloy in kg -/
def weight_first : ℝ := 15

/-- The weight of the second alloy in kg -/
def weight_second : ℝ := 35

/-- The percentage of chromium in the resulting alloy -/
def chromium_percentage_result : ℝ := 7.2

theorem chromium_percentage_calculation :
  (chromium_percentage_first * weight_first + chromium_percentage_second * weight_second) / (weight_first + weight_second) = chromium_percentage_result :=
by sorry

end chromium_percentage_calculation_l2236_223626


namespace wall_height_is_ten_l2236_223663

-- Define the dimensions of the rooms
def livingRoomSide : ℝ := 40
def bedroomLength : ℝ := 12
def bedroomWidth : ℝ := 10

-- Define the number of walls to be painted in each room
def livingRoomWalls : ℕ := 3
def bedroomWalls : ℕ := 4

-- Define the total area to be painted
def totalAreaToPaint : ℝ := 1640

-- Theorem statement
theorem wall_height_is_ten :
  let livingRoomPerimeter := livingRoomSide * 4
  let livingRoomPaintPerimeter := livingRoomPerimeter - livingRoomSide
  let bedroomPerimeter := 2 * (bedroomLength + bedroomWidth)
  let totalPerimeterToPaint := livingRoomPaintPerimeter + bedroomPerimeter
  totalAreaToPaint / totalPerimeterToPaint = 10 := by
  sorry

end wall_height_is_ten_l2236_223663


namespace average_and_square_multiple_l2236_223650

theorem average_and_square_multiple (n : ℝ) (m : ℝ) (h1 : n ≠ 0) (h2 : n = 9) 
  (h3 : (n + n^2) / 2 = m * n) : m = 5 := by
  sorry

end average_and_square_multiple_l2236_223650


namespace derivative_equals_function_implies_zero_at_two_l2236_223644

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem derivative_equals_function_implies_zero_at_two 
  (h : ∀ x, deriv f x = f x) : 
  deriv f 2 = 0 := by
  sorry

end derivative_equals_function_implies_zero_at_two_l2236_223644


namespace isosceles_triangle_condition_l2236_223643

/-- Proves that if in a triangle with sides a, b, c and opposite angles α, β, γ,
    the equation a + b = tan(γ/2) * (a * tan(α) + b * tan(β)) holds, then α = β. -/
theorem isosceles_triangle_condition 
  (a b c : ℝ) 
  (α β γ : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_angles : α + β + γ = Real.pi)
  (h_condition : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β)) :
  α = β := by
  sorry


end isosceles_triangle_condition_l2236_223643


namespace m_range_theorem_l2236_223666

/-- The range of m values satisfying the given conditions -/
def m_range : Set ℝ :=
  Set.Ioc 1 2 ∪ Set.Ici 3

/-- Condition for the first equation to have two distinct negative roots -/
def has_two_negative_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- Condition for the second equation to have no real roots -/
def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- Main theorem statement -/
theorem m_range_theorem (m : ℝ) :
  (has_two_negative_roots m ∨ has_no_real_roots m) ∧
  ¬(has_two_negative_roots m ∧ has_no_real_roots m) ↔
  m ∈ m_range :=
sorry

end m_range_theorem_l2236_223666


namespace exam_question_distribution_l2236_223676

theorem exam_question_distribution (total_questions : ℕ) 
  (group_a_marks group_b_marks group_c_marks : ℕ) 
  (group_b_questions : ℕ) :
  total_questions = 100 →
  group_a_marks = 1 →
  group_b_marks = 2 →
  group_c_marks = 3 →
  group_b_questions = 23 →
  (∀ a b c : ℕ, a + b + c = total_questions → a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1) →
  (∀ a b c : ℕ, a + b + c = total_questions → 
    a * group_a_marks ≥ (6 * (a * group_a_marks + b * group_b_marks + c * group_c_marks)) / 10) →
  ∃! c : ℕ, c = 1 ∧ ∃ a : ℕ, a + group_b_questions + c = total_questions :=
by sorry

end exam_question_distribution_l2236_223676


namespace bus_speeds_l2236_223687

theorem bus_speeds (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ)
  (h1 : distance = 48)
  (h2 : time_difference = 1/6)
  (h3 : speed_difference = 4) :
  ∃ (speed1 speed2 : ℝ),
    speed1 = 36 ∧
    speed2 = 32 ∧
    distance / speed1 + time_difference = distance / speed2 ∧
    speed1 = speed2 + speed_difference :=
by sorry

end bus_speeds_l2236_223687


namespace smallest_X_value_l2236_223688

/-- A function that checks if a natural number consists only of 0s and 1s in its decimal representation -/
def onlyZerosAndOnes (n : ℕ) : Prop := sorry

/-- The smallest positive integer T that satisfies the given conditions -/
def T : ℕ := sorry

theorem smallest_X_value :
  T > 0 ∧
  onlyZerosAndOnes T ∧
  T % 15 = 0 ∧
  (∀ t : ℕ, t > 0 → onlyZerosAndOnes t → t % 15 = 0 → t ≥ T) →
  T / 15 = 74 := by sorry

end smallest_X_value_l2236_223688


namespace complement_M_intersect_N_l2236_223625

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 3, 4}

theorem complement_M_intersect_N :
  (Set.compl M ∩ N) = {0, 4} := by sorry

end complement_M_intersect_N_l2236_223625


namespace prob_first_class_is_072_l2236_223672

/-- Represents a batch of products -/
structure ProductBatch where
  defectiveRate : ℝ
  firstClassRateAmongQualified : ℝ

/-- Calculates the probability of selecting a first-class product from a batch -/
def probabilityFirstClass (batch : ProductBatch) : ℝ :=
  (1 - batch.defectiveRate) * batch.firstClassRateAmongQualified

/-- Theorem: The probability of selecting a first-class product from the given batch is 0.72 -/
theorem prob_first_class_is_072 (batch : ProductBatch) 
    (h1 : batch.defectiveRate = 0.04)
    (h2 : batch.firstClassRateAmongQualified = 0.75) : 
    probabilityFirstClass batch = 0.72 := by
  sorry

#eval probabilityFirstClass { defectiveRate := 0.04, firstClassRateAmongQualified := 0.75 }

end prob_first_class_is_072_l2236_223672


namespace race_distance_l2236_223659

theorem race_distance (d : ℝ) (vA vB vC : ℝ) 
  (h1 : d / vA = (d - 20) / vB)
  (h2 : d / vB = (d - 10) / vC)
  (h3 : d / vA = (d - 28) / vC)
  (h4 : d > 0) (h5 : vA > 0) (h6 : vB > 0) (h7 : vC > 0) : d = 100 := by
  sorry

end race_distance_l2236_223659


namespace parallel_vectors_trig_identity_l2236_223680

/-- Given two parallel vectors a and b, prove that cos(2α) + sin(2α) = -7/5 -/
theorem parallel_vectors_trig_identity (α : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (Real.sin α - Real.cos α, Real.sin α + Real.cos α)
  (∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1) →  -- parallel vectors condition
  Real.cos (2 * α) + Real.sin (2 * α) = -7/5 := by
  sorry

end parallel_vectors_trig_identity_l2236_223680


namespace f_expression_range_f_transformed_is_l2236_223653

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The property that f(0) = 0 -/
axiom f_zero : f 0 = 0

/-- The property that f(x+1) = f(x) + x + 1 for all x -/
axiom f_next (x : ℝ) : f (x + 1) = f x + x + 1

/-- f is a quadratic function -/
axiom f_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

/-- Theorem: f(x) = (1/2)x^2 + (1/2)x -/
theorem f_expression : ∀ x, f x = (1/2) * x^2 + (1/2) * x := sorry

/-- The range of y = f(x^2 - 2) -/
def range_f_transformed : Set ℝ := {y | ∃ x, y = f (x^2 - 2)}

/-- Theorem: The range of y = f(x^2 - 2) is [-1/8, +∞) -/
theorem range_f_transformed_is : range_f_transformed = {y | y ≥ -1/8} := sorry

end f_expression_range_f_transformed_is_l2236_223653


namespace negation_of_forall_positive_square_plus_one_l2236_223664

theorem negation_of_forall_positive_square_plus_one (P : Real → Prop) : 
  (¬ ∀ x > 1, x^2 + 1 ≥ 0) ↔ (∃ x > 1, x^2 + 1 < 0) :=
by sorry

end negation_of_forall_positive_square_plus_one_l2236_223664


namespace asymptotes_sum_l2236_223651

theorem asymptotes_sum (A B C : ℤ) : 
  (∀ x, x^3 + A*x^2 + B*x + C = (x + 1)*(x - 3)*(x - 4)) → 
  A + B + C = 11 := by
sorry

end asymptotes_sum_l2236_223651


namespace exponent_simplification_l2236_223629

theorem exponent_simplification : ((-2 : ℝ) ^ 3) ^ (1/3) - (-1 : ℝ) ^ 0 = -3 := by
  sorry

end exponent_simplification_l2236_223629


namespace max_value_problem_l2236_223675

theorem max_value_problem (X Y Z : ℕ) (h : 2 * X + 3 * Y + Z = 18) :
  (∀ X' Y' Z' : ℕ, 2 * X' + 3 * Y' + Z' = 18 →
    X' * Y' * Z' + X' * Y' + Y' * Z' + Z' * X' ≤ X * Y * Z + X * Y + Y * Z + Z * X) →
  X * Y * Z + X * Y + Y * Z + Z * X = 24 :=
sorry

end max_value_problem_l2236_223675


namespace wendy_bouquets_l2236_223630

/-- Given the initial number of flowers, flowers per bouquet, and number of wilted flowers,
    calculate the number of bouquets that can be made. -/
def bouquets_remaining (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (initial_flowers - wilted_flowers) / flowers_per_bouquet

/-- Prove that Wendy can make 2 bouquets with the remaining flowers. -/
theorem wendy_bouquets :
  bouquets_remaining 45 5 35 = 2 := by
  sorry

end wendy_bouquets_l2236_223630


namespace perpendicular_line_equation_l2236_223628

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the point through which the perpendicular line passes
def point : ℝ × ℝ := (-1, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  (∀ x y : ℝ, perpendicular_line x y ↔ 
    (∃ m b : ℝ, y = m*x + b ∧ 
      (perpendicular_line point.1 point.2) ∧
      (m * (1/2) = -1))) :=
sorry

end perpendicular_line_equation_l2236_223628


namespace last_digit_of_77_in_binary_l2236_223694

theorem last_digit_of_77_in_binary (n : Nat) : n = 77 → n % 2 = 1 := by
  sorry

end last_digit_of_77_in_binary_l2236_223694


namespace box_calories_l2236_223681

/-- Calculates the total calories in a box of cookies -/
def total_calories (cookies_per_bag : ℕ) (bags_per_box : ℕ) (calories_per_cookie : ℕ) : ℕ :=
  cookies_per_bag * bags_per_box * calories_per_cookie

/-- Theorem: The total calories in a box of cookies is 1600 -/
theorem box_calories :
  total_calories 20 4 20 = 1600 := by
  sorry

end box_calories_l2236_223681


namespace stratified_sampling_most_suitable_l2236_223693

/-- Represents the age groups in the population -/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Other

/-- Represents the population structure -/
structure Population where
  total : Nat
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Determines the most suitable sampling method given a population and sample size -/
def mostSuitableSamplingMethod (pop : Population) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- The theorem stating that stratified sampling is the most suitable method for the given population and sample size -/
theorem stratified_sampling_most_suitable :
  let pop : Population := { total := 163, elderly := 28, middleAged := 54, young := 81 }
  let sampleSize : Nat := 36
  mostSuitableSamplingMethod pop sampleSize = SamplingMethod.Stratified :=
sorry

end stratified_sampling_most_suitable_l2236_223693


namespace bond_face_value_l2236_223615

/-- The face value of a bond -/
def face_value : ℝ := 5000

/-- The interest rate as a percentage of face value -/
def interest_rate : ℝ := 0.05

/-- The selling price of the bond -/
def selling_price : ℝ := 3846.153846153846

/-- The interest amount as a percentage of selling price -/
def interest_percentage : ℝ := 0.065

theorem bond_face_value :
  face_value = selling_price * interest_percentage / interest_rate :=
by sorry

end bond_face_value_l2236_223615


namespace triangle_angle_from_area_and_dot_product_l2236_223600

theorem triangle_angle_from_area_and_dot_product 
  (A B C : ℝ × ℝ) -- Points in 2D plane
  (area : ℝ) 
  (dot_product : ℝ) :
  area = Real.sqrt 3 / 2 →
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = dot_product →
  dot_product = -3 →
  let angle := Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / 
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))
  angle = 5 * Real.pi / 6 :=
by sorry

end triangle_angle_from_area_and_dot_product_l2236_223600


namespace counterexamples_exist_l2236_223655

theorem counterexamples_exist : ∃ (a b c : ℝ),
  -- Statement A is not always true
  (a * b ≠ 0 ∧ a < b ∧ (1 / a) ≤ (1 / b)) ∧
  -- Statement C is not always true
  (a > b ∧ b > 0 ∧ ((b + 1) / (a + 1)) ≥ (b / a)) ∧
  -- Statement D is not always true
  (c < b ∧ b < a ∧ a * c < 0 ∧ c * b^2 ≥ a * b^2) :=
sorry

end counterexamples_exist_l2236_223655


namespace cake_cutting_theorem_l2236_223686

/-- Represents a rectangular cake -/
structure Cake where
  length : ℕ
  width : ℕ
  pieces : ℕ

/-- The maximum number of pieces obtainable with one straight cut -/
def max_pieces_one_cut (c : Cake) : ℕ := sorry

/-- The minimum number of cuts required to ensure each piece is cut -/
def min_cuts_all_pieces (c : Cake) : ℕ := sorry

/-- Theorem for the cake cutting problem -/
theorem cake_cutting_theorem (c : Cake) 
  (h1 : c.length = 5 ∧ c.width = 2) 
  (h2 : c.pieces = 10) : 
  max_pieces_one_cut c = 16 ∧ min_cuts_all_pieces c = 2 := by sorry

end cake_cutting_theorem_l2236_223686


namespace problem_solution_l2236_223699

theorem problem_solution (x y : ℝ) 
  (h1 : |x| + x + y = 8) 
  (h2 : x + |y| - y = 10) : 
  x + y = 14/5 := by
  sorry

end problem_solution_l2236_223699


namespace candy_distribution_l2236_223646

theorem candy_distribution (total : ℕ) (a b c d : ℕ) : 
  total = 2013 →
  a + b + c + d = total →
  a = 2 * b + 10 →
  a = 3 * c + 18 →
  a = 5 * d - 55 →
  a = 990 :=
by sorry

end candy_distribution_l2236_223646


namespace john_light_bulbs_l2236_223622

/-- Calculates the number of light bulbs left after using some and giving away half of the remainder -/
def lightBulbsLeft (initial : ℕ) (used : ℕ) : ℕ :=
  let remaining := initial - used
  remaining - remaining / 2

/-- Proves that starting with 40 light bulbs, using 16, and giving away half of the remainder leaves 12 bulbs -/
theorem john_light_bulbs : lightBulbsLeft 40 16 = 12 := by
  sorry

end john_light_bulbs_l2236_223622


namespace radical_simplification_l2236_223671

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (12 * p) * Real.sqrt (7 * p^3) * Real.sqrt (15 * p^5) = 6 * p^4 * Real.sqrt (35 * p) := by
  sorry

end radical_simplification_l2236_223671


namespace balls_distribution_l2236_223609

-- Define the number of balls and boxes
def num_balls : ℕ := 6
def num_boxes : ℕ := 3

-- Define the function to calculate the number of ways to distribute balls
def distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  Nat.choose (balls + boxes - 1) (boxes - 1)

-- Theorem statement
theorem balls_distribution :
  distribute_balls num_balls num_boxes = 28 := by
  sorry

end balls_distribution_l2236_223609


namespace real_part_of_complex_fraction_l2236_223648

theorem real_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  (2 * i / (1 + i)).re = 1 := by
  sorry

end real_part_of_complex_fraction_l2236_223648


namespace outfits_count_l2236_223695

/-- The number of different outfits that can be made -/
def num_outfits (num_shirts : ℕ) (num_ties : ℕ) (num_pants : ℕ) (num_belts : ℕ) : ℕ :=
  num_shirts * num_pants * (num_ties + 1) * (num_belts + 1)

/-- Theorem stating that the number of outfits is 360 given the specific conditions -/
theorem outfits_count :
  num_outfits 5 5 4 2 = 360 :=
by sorry

end outfits_count_l2236_223695


namespace iguana_feed_cost_l2236_223690

/-- The monthly cost to feed each iguana, given the number of pets, 
    the cost to feed geckos and snakes, and the total annual cost for all pets. -/
theorem iguana_feed_cost 
  (num_geckos num_iguanas num_snakes : ℕ)
  (gecko_cost snake_cost : ℚ)
  (total_annual_cost : ℚ)
  (h1 : num_geckos = 3)
  (h2 : num_iguanas = 2)
  (h3 : num_snakes = 4)
  (h4 : gecko_cost = 15)
  (h5 : snake_cost = 10)
  (h6 : total_annual_cost = 1140)
  : ∃ (iguana_cost : ℚ), 
    iguana_cost = 5 ∧
    (num_geckos : ℚ) * gecko_cost + 
    (num_iguanas : ℚ) * iguana_cost + 
    (num_snakes : ℚ) * snake_cost = 
    total_annual_cost / 12 :=
sorry

end iguana_feed_cost_l2236_223690


namespace terrell_lifting_equivalence_l2236_223682

/-- The number of times Terrell lifts the weights in the initial setup -/
def initial_lifts : ℕ := 10

/-- The weight of each dumbbell in the initial setup (in pounds) -/
def initial_weight : ℕ := 25

/-- The number of dumbbells used in the initial setup -/
def initial_dumbbells : ℕ := 2

/-- The weight of the single dumbbell in the new setup (in pounds) -/
def new_weight : ℕ := 20

/-- The total weight lifted in the initial setup (in pounds) -/
def total_weight : ℕ := initial_dumbbells * initial_weight * initial_lifts

/-- The number of times Terrell must lift the new weight to achieve the same total weight -/
def required_lifts : ℕ := total_weight / new_weight

theorem terrell_lifting_equivalence :
  required_lifts = 25 := by sorry

end terrell_lifting_equivalence_l2236_223682


namespace least_subtraction_for_divisibility_l2236_223612

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 29 ∧ (10154 - x) % 30 = 0 ∧ ∀ y : ℕ, y < x → (10154 - y) % 30 ≠ 0 :=
sorry

end least_subtraction_for_divisibility_l2236_223612


namespace student_calculation_error_l2236_223683

def correct_calculation : ℚ := (3/4 * 16 - 7/8 * 8) / (3/10 - 1/8)

def incorrect_calculation : ℚ := (3/4 * 16 - 7/8 * 8) * (3/5)

def percentage_error (correct incorrect : ℚ) : ℚ :=
  abs (correct - incorrect) / correct * 100

theorem student_calculation_error :
  abs (percentage_error correct_calculation incorrect_calculation - 89.47) < 0.01 := by
  sorry

end student_calculation_error_l2236_223683


namespace arithmetic_sequence_property_l2236_223602

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - a 8 = -8 :=
sorry

end arithmetic_sequence_property_l2236_223602


namespace nathaniel_tickets_l2236_223603

/-- The number of tickets Nathaniel gives to each friend -/
def tickets_per_friend : ℕ := 2

/-- The number of Nathaniel's best friends -/
def num_friends : ℕ := 4

/-- The number of tickets Nathaniel has left after giving away -/
def tickets_left : ℕ := 3

/-- The initial number of tickets Nathaniel had -/
def initial_tickets : ℕ := tickets_per_friend * num_friends + tickets_left

theorem nathaniel_tickets : initial_tickets = 11 := by
  sorry

end nathaniel_tickets_l2236_223603


namespace necessary_not_sufficient_l2236_223614

theorem necessary_not_sufficient (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → (a / b + b / a ≥ 2) → (a^2 + b^2 ≥ 2*a*b)) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (a^2 + b^2 ≥ 2*a*b) ∧ (a / b + b / a < 2)) :=
by sorry

end necessary_not_sufficient_l2236_223614


namespace percentage_failed_hindi_l2236_223679

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  failed_english = 44 →
  failed_both = 22 →
  passed_both = 44 →
  ∃ failed_hindi : ℝ, failed_hindi = 34 :=
by
  sorry

end percentage_failed_hindi_l2236_223679


namespace a2_4_sufficient_not_necessary_for_a3_16_l2236_223610

/-- A geometric sequence with first term 1 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The property that a₂ = 4 is sufficient but not necessary for a₃ = 16 -/
theorem a2_4_sufficient_not_necessary_for_a3_16 :
  ∀ a : ℕ → ℝ, GeometricSequence a →
    (∀ a : ℕ → ℝ, GeometricSequence a → a 2 = 4 → a 3 = 16) ∧
    ¬(∀ a : ℕ → ℝ, GeometricSequence a → a 3 = 16 → a 2 = 4) :=
by sorry

end a2_4_sufficient_not_necessary_for_a3_16_l2236_223610


namespace peter_bought_five_large_glasses_l2236_223636

/-- Represents the purchase of glasses by Peter -/
structure GlassesPurchase where
  small_cost : ℕ             -- Cost of a small glass
  large_cost : ℕ             -- Cost of a large glass
  total_money : ℕ            -- Total money Peter has
  small_bought : ℕ           -- Number of small glasses bought
  change : ℕ                 -- Money left as change

/-- Calculates the number of large glasses Peter bought -/
def large_glasses_bought (purchase : GlassesPurchase) : ℕ :=
  (purchase.total_money - purchase.small_cost * purchase.small_bought - purchase.change) / purchase.large_cost

/-- Theorem stating that Peter bought 5 large glasses -/
theorem peter_bought_five_large_glasses :
  ∀ (purchase : GlassesPurchase),
    purchase.small_cost = 3 →
    purchase.large_cost = 5 →
    purchase.total_money = 50 →
    purchase.small_bought = 8 →
    purchase.change = 1 →
    large_glasses_bought purchase = 5 := by
  sorry


end peter_bought_five_large_glasses_l2236_223636


namespace systematic_sampling_missiles_l2236_223667

/-- Represents a systematic sampling sequence -/
def SystematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => start + i * (total / sampleSize))

/-- The problem statement -/
theorem systematic_sampling_missiles :
  let total := 50
  let sampleSize := 5
  let start := 3
  SystematicSample total sampleSize start = [3, 13, 23, 33, 43] := by
  sorry

#eval SystematicSample 50 5 3

end systematic_sampling_missiles_l2236_223667


namespace triangle_altitude_and_median_equations_l2236_223641

/-- Triangle ABC with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle -/
def triangle : Triangle := { A := (4, 0), B := (6, 7), C := (0, 3) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of the altitude from B to AC -/
def altitudeEquation : LineEquation := { a := 3, b := 2, c := -12 }

/-- The equation of the median from B to AC -/
def medianEquation : LineEquation := { a := 5, b := 1, c := -20 }

theorem triangle_altitude_and_median_equations :
  let t := triangle
  let alt := altitudeEquation
  let med := medianEquation
  (∀ x y : ℝ, alt.a * x + alt.b * y + alt.c = 0 ↔ 
    (x - t.B.1) * (t.A.1 - t.C.1) + (y - t.B.2) * (t.A.2 - t.C.2) = 0) ∧
  (∀ x y : ℝ, med.a * x + med.b * y + med.c = 0 ↔ 
    2 * (x - t.B.1) = t.A.1 - t.C.1 ∧ 2 * (y - t.B.2) = t.A.2 - t.C.2) := by
  sorry

end triangle_altitude_and_median_equations_l2236_223641


namespace essay_competition_probability_l2236_223608

theorem essay_competition_probability (n : ℕ) (h : n = 6) :
  let total_outcomes := n * n
  let favorable_outcomes := n * (n - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 6 :=
by sorry

end essay_competition_probability_l2236_223608


namespace dog_to_cats_ratio_is_two_to_one_l2236_223635

/-- The weight of Christine's first cat in pounds -/
def cat1_weight : ℕ := 7

/-- The weight of Christine's second cat in pounds -/
def cat2_weight : ℕ := 10

/-- The combined weight of Christine's cats in pounds -/
def cats_combined_weight : ℕ := cat1_weight + cat2_weight

/-- The weight of Christine's dog in pounds -/
def dog_weight : ℕ := 34

/-- The ratio of the dog's weight to the combined weight of the cats -/
def dog_to_cats_ratio : ℚ := dog_weight / cats_combined_weight

theorem dog_to_cats_ratio_is_two_to_one :
  dog_to_cats_ratio = 2 := by sorry

end dog_to_cats_ratio_is_two_to_one_l2236_223635


namespace matching_probability_theorem_l2236_223685

/-- Represents the distribution of shoe pairs by color -/
structure ShoeDistribution where
  black : Nat
  brown : Nat
  gray : Nat
  red : Nat

/-- Calculates the total number of individual shoes -/
def totalShoes (d : ShoeDistribution) : Nat :=
  2 * (d.black + d.brown + d.gray + d.red)

/-- Calculates the probability of selecting a matching pair -/
def matchingProbability (d : ShoeDistribution) : Rat :=
  let total := totalShoes d
  let numerator := 
    d.black * (d.black - 1) + 
    d.brown * (d.brown - 1) + 
    d.gray * (d.gray - 1) + 
    d.red * (d.red - 1)
  (numerator : Rat) / (total * (total - 1))

/-- John's shoe distribution -/
def johnsShoes : ShoeDistribution :=
  { black := 8, brown := 4, gray := 3, red := 1 }

theorem matching_probability_theorem : 
  matchingProbability johnsShoes = 45 / 248 := by
  sorry

end matching_probability_theorem_l2236_223685


namespace half_abs_diff_squares_23_19_l2236_223668

theorem half_abs_diff_squares_23_19 : (1 / 2 : ℝ) * |23^2 - 19^2| = 84 := by
  sorry

end half_abs_diff_squares_23_19_l2236_223668


namespace greatest_power_of_four_dividing_16_factorial_l2236_223645

theorem greatest_power_of_four_dividing_16_factorial :
  (∃ k : ℕ+, k.val = 7 ∧ 
   ∀ m : ℕ+, (4 ^ m.val ∣ Nat.factorial 16) → m.val ≤ 7) :=
by sorry

end greatest_power_of_four_dividing_16_factorial_l2236_223645


namespace window_area_calculation_l2236_223697

/-- Calculates the area of a window given its length in meters and width in feet -/
def windowArea (lengthMeters : ℝ) (widthFeet : ℝ) : ℝ :=
  let meterToFeet : ℝ := 3.28084
  let lengthFeet : ℝ := lengthMeters * meterToFeet
  lengthFeet * widthFeet

theorem window_area_calculation :
  windowArea 2 15 = 98.4252 := by
  sorry

end window_area_calculation_l2236_223697


namespace unique_triple_solution_l2236_223640

theorem unique_triple_solution : 
  ∃! (a b c : ℤ), (|a - b| + c = 23) ∧ (a^2 - b*c = 119) :=
sorry

end unique_triple_solution_l2236_223640


namespace complex_multiplication_l2236_223670

theorem complex_multiplication :
  (1 + Complex.I) * (2 - Complex.I) = 3 + Complex.I := by
  sorry

end complex_multiplication_l2236_223670


namespace waiter_customers_l2236_223658

/-- The number of customers who didn't leave a tip -/
def no_tip : ℕ := 34

/-- The number of customers who left a tip -/
def left_tip : ℕ := 15

/-- The number of customers added during the lunch rush -/
def added_customers : ℕ := 20

/-- The number of customers before the lunch rush -/
def customers_before : ℕ := 29

theorem waiter_customers :
  customers_before = (no_tip + left_tip) - added_customers :=
by sorry

end waiter_customers_l2236_223658


namespace expression_equality_l2236_223637

theorem expression_equality : 
  Real.sqrt 12 + 2⁻¹ + Real.cos (60 * π / 180) - 3 * Real.tan (30 * π / 180) = Real.sqrt 3 + 1 := by
  sorry

end expression_equality_l2236_223637


namespace mad_hatter_march_hare_meeting_time_difference_l2236_223642

/-- Represents a clock with a specific rate of time change per hour -/
structure Clock where
  rate : ℚ

/-- Calculates the actual time passed for a given clock time -/
def actual_time (c : Clock) (clock_time : ℚ) : ℚ :=
  clock_time * c.rate

theorem mad_hatter_march_hare_meeting_time_difference : 
  let mad_hatter_clock : Clock := ⟨60 / 75⟩
  let march_hare_clock : Clock := ⟨60 / 50⟩
  let meeting_time : ℚ := 5

  actual_time march_hare_clock meeting_time - actual_time mad_hatter_clock meeting_time = 2 := by
  sorry

end mad_hatter_march_hare_meeting_time_difference_l2236_223642


namespace cone_lateral_surface_area_l2236_223619

/-- The lateral surface area of a cone with base radius 3 cm and lateral surface forming a semicircle when unfolded -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), 
    r = 3 → -- base radius is 3 cm
    l = 6 → -- slant height is 6 cm (derived from the semicircle condition)
    (1/2 : ℝ) * Real.pi * l^2 = 18 * Real.pi :=
by sorry

end cone_lateral_surface_area_l2236_223619


namespace stephanies_remaining_payment_l2236_223684

/-- The total amount Stephanie still needs to pay to finish her bills -/
def remaining_payment (electricity gas water internet : ℝ) 
                      (gas_paid_fraction : ℝ) 
                      (gas_additional_payment : ℝ) 
                      (water_paid_fraction : ℝ) 
                      (internet_payments : ℕ) 
                      (internet_payment_amount : ℝ) : ℝ :=
  (gas - gas_paid_fraction * gas - gas_additional_payment) +
  (water - water_paid_fraction * water) +
  (internet - internet_payments * internet_payment_amount)

/-- Stephanie's remaining bill payment theorem -/
theorem stephanies_remaining_payment :
  remaining_payment 60 40 40 25 (3/4) 5 (1/2) 4 5 = 30 := by
  sorry

end stephanies_remaining_payment_l2236_223684


namespace gcd_987654_123456_l2236_223638

theorem gcd_987654_123456 : Nat.gcd 987654 123456 = 6 := by
  sorry

end gcd_987654_123456_l2236_223638


namespace log_inequality_range_l2236_223662

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_inequality_range (x : ℝ) :
  lg (x + 1) < lg (3 - x) ↔ -1 < x ∧ x < 1 :=
by sorry

end log_inequality_range_l2236_223662


namespace isabellas_hair_growth_l2236_223634

/-- Given Isabella's initial and final hair lengths, prove the amount of hair growth. -/
theorem isabellas_hair_growth 
  (initial_length : ℝ) 
  (final_length : ℝ) 
  (h1 : initial_length = 18) 
  (h2 : final_length = 24) : 
  final_length - initial_length = 6 := by
  sorry

end isabellas_hair_growth_l2236_223634


namespace probability_other_side_red_l2236_223661

structure Card where
  side1 : String
  side2 : String

def Box : List Card := [
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "red"},
  {side1 := "black", side2 := "red"},
  {side1 := "black", side2 := "red"},
  {side1 := "red", side2 := "red"},
  {side1 := "red", side2 := "red"},
  {side1 := "blue", side2 := "blue"}
]

def isRed (s : String) : Bool := s == "red"

def countRedSides (cards : List Card) : Nat :=
  cards.foldl (fun acc card => acc + (if isRed card.side1 then 1 else 0) + (if isRed card.side2 then 1 else 0)) 0

def countBothRedCards (cards : List Card) : Nat :=
  cards.foldl (fun acc card => acc + (if isRed card.side1 && isRed card.side2 then 1 else 0)) 0

theorem probability_other_side_red (box : List Card := Box) :
  let totalRedSides := countRedSides box
  let bothRedCards := countBothRedCards box
  (2 * bothRedCards : Rat) / totalRedSides = 4 / 7 := by sorry

end probability_other_side_red_l2236_223661


namespace arithmetic_sequence_property_l2236_223674

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 200 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 + a 7 + a 9 + a 11 = 200

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  4 * a 5 - 2 * a 3 = 80 := by
  sorry

end arithmetic_sequence_property_l2236_223674


namespace age_ratio_proof_l2236_223647

/-- Proves that the ratio of Saras's age to the combined age of Kul and Ani is 1:2 -/
theorem age_ratio_proof (kul_age saras_age ani_age : ℕ) 
  (h1 : kul_age = 22)
  (h2 : saras_age = 33)
  (h3 : ani_age = 44) : 
  (saras_age : ℚ) / (kul_age + ani_age : ℚ) = 1 / 2 := by
  sorry

#check age_ratio_proof

end age_ratio_proof_l2236_223647


namespace senate_subcommittee_count_l2236_223616

theorem senate_subcommittee_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 8
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (Nat.choose total_republicans subcommittee_republicans) *
  (Nat.choose total_democrats subcommittee_democrats) = 11760 := by
  sorry

end senate_subcommittee_count_l2236_223616


namespace haley_shirts_l2236_223657

/-- The number of shirts Haley bought -/
def shirts_bought : ℕ := 11

/-- The number of shirts Haley returned -/
def shirts_returned : ℕ := 6

/-- The number of shirts Haley ended up with -/
def shirts_remaining : ℕ := shirts_bought - shirts_returned

theorem haley_shirts : shirts_remaining = 5 := by
  sorry

end haley_shirts_l2236_223657


namespace complex_quadratic_roots_l2236_223621

theorem complex_quadratic_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = Complex.I * 2 ∧ 
  z₂ = -2 - Complex.I * 2 ∧ 
  z₁^2 + 2*z₁ = -3 + Complex.I * 4 ∧ 
  z₂^2 + 2*z₂ = -3 + Complex.I * 4 := by
  sorry

end complex_quadratic_roots_l2236_223621


namespace parabola_equation_l2236_223605

/-- Given a parabola y² = 2mx (m ≠ 0) intersected by the line y = x - 4,
    if the length of the chord formed by this intersection is 6√2,
    then the equation of the parabola is either y² = (-4 + √34)x or y² = (-4 - √34)x. -/
theorem parabola_equation (m : ℝ) (h1 : m ≠ 0) :
  let f (x : ℝ) := 2 * m * x
  let g (x : ℝ) := x - 4
  let chord_length := (∃ x₁ x₂, x₁ ≠ x₂ ∧ f (g x₁) = (g x₁)^2 ∧ f (g x₂) = (g x₂)^2 ∧
    Real.sqrt ((x₁ - x₂)^2 + (g x₁ - g x₂)^2) = 6 * Real.sqrt 2)
  chord_length →
    (∀ x, f x = (-4 + Real.sqrt 34) * x) ∨ (∀ x, f x = (-4 - Real.sqrt 34) * x) := by
  sorry

end parabola_equation_l2236_223605


namespace interest_group_signup_ways_l2236_223691

theorem interest_group_signup_ways (num_students : ℕ) (num_groups : ℕ) : 
  num_students = 4 → num_groups = 3 → (num_groups ^ num_students : ℕ) = 81 := by
  sorry

end interest_group_signup_ways_l2236_223691


namespace simplify_complex_expression_l2236_223652

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_expression :
  3 * (4 - 2*i) + 2*i * (3 + 2*i) = 8 :=
by
  sorry

end simplify_complex_expression_l2236_223652


namespace harvest_season_duration_l2236_223627

theorem harvest_season_duration (regular_earnings overtime_earnings total_earnings : ℕ) 
  (h1 : regular_earnings = 28)
  (h2 : overtime_earnings = 939)
  (h3 : total_earnings = 1054997) : 
  total_earnings / (regular_earnings + overtime_earnings) = 1091 := by
  sorry

end harvest_season_duration_l2236_223627


namespace chad_bbq_ice_cost_l2236_223613

/-- The cost of ice for Chad's BBQ -/
def bbq_ice_cost (total_people : ℕ) (ice_per_person : ℕ) (package_size : ℕ) (cost_per_package : ℚ) : ℚ :=
  let total_ice := total_people * ice_per_person
  let packages_needed := (total_ice + package_size - 1) / package_size
  packages_needed * cost_per_package

/-- Theorem: The cost of ice for Chad's BBQ is $27 -/
theorem chad_bbq_ice_cost :
  bbq_ice_cost 20 3 10 (4.5 : ℚ) = 27 := by
  sorry

end chad_bbq_ice_cost_l2236_223613


namespace max_factors_x8_minus_1_l2236_223631

theorem max_factors_x8_minus_1 : 
  ∃ (k : ℕ), k = 5 ∧ 
  (∀ (p : List (Polynomial ℝ)), 
    (∀ q ∈ p, q.degree > 0) → -- Each factor is non-constant
    (List.prod p = Polynomial.X ^ 8 - 1) → -- The product of factors equals x^8 - 1
    List.length p ≤ k) ∧ -- The number of factors is at most k
  (∃ (p : List (Polynomial ℝ)), 
    (∀ q ∈ p, q.degree > 0) ∧ 
    (List.prod p = Polynomial.X ^ 8 - 1) ∧ 
    List.length p = k) -- There exists a factorization with exactly k factors
  := by sorry

end max_factors_x8_minus_1_l2236_223631


namespace smallest_sum_of_reciprocals_l2236_223689

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
  (x : ℤ) + y ≤ (a : ℤ) + b :=
by sorry

end smallest_sum_of_reciprocals_l2236_223689


namespace brother_father_age_ratio_l2236_223654

/-- Represents the ages of family members and total family age --/
structure FamilyAges where
  total : ℕ
  father : ℕ
  mother : ℕ
  sister : ℕ
  kaydence : ℕ

/-- Theorem stating the ratio of brother's age to father's age --/
theorem brother_father_age_ratio (f : FamilyAges) 
  (h1 : f.total = 200)
  (h2 : f.father = 60)
  (h3 : f.mother = f.father - 2)
  (h4 : f.sister = 40)
  (h5 : f.kaydence = 12) :
  ∃ (brother_age : ℕ), 
    brother_age = f.total - (f.father + f.mother + f.sister + f.kaydence) ∧
    2 * brother_age = f.father :=
by
  sorry

end brother_father_age_ratio_l2236_223654


namespace original_fraction_value_l2236_223624

theorem original_fraction_value (x : ℚ) : 
  (x + 1) / (x + 8) = 11 / 17 → x / (x + 7) = 71 / 113 := by
  sorry

end original_fraction_value_l2236_223624


namespace sum_equals_16x_l2236_223665

/-- Given real numbers x, y, z, and w, where y = 2x, z = 3y, and w = z + x,
    prove that x + y + z + w = 16x -/
theorem sum_equals_16x (x y z w : ℝ) 
    (h1 : y = 2 * x) 
    (h2 : z = 3 * y) 
    (h3 : w = z + x) : 
  x + y + z + w = 16 * x := by
  sorry

end sum_equals_16x_l2236_223665


namespace triangle_perimeter_l2236_223649

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 5 ∧ c^2 - 3*c = c - 3 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b →
  a + b + c = 11 := by
  sorry

end triangle_perimeter_l2236_223649


namespace smallest_angle_in_triangle_with_ratio_l2236_223617

theorem smallest_angle_in_triangle_with_ratio (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- angles are positive
  a + b + c = 180 →  -- sum of angles is 180°
  b = 2 * a →  -- ratio condition
  c = 3 * a →  -- ratio condition
  a = 30 := by
sorry

end smallest_angle_in_triangle_with_ratio_l2236_223617


namespace arithmetic_geometric_ratio_l2236_223601

/-- An arithmetic sequence with common difference d ≠ 0 where a₁, a₄, and a₁₀ form a geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  d : ℝ      -- Common difference
  hd : d ≠ 0 -- d is non-zero
  arithmetic_seq : ∀ n, a (n + 1) = a n + d  -- Arithmetic sequence property
  geometric_seq : (a 4) ^ 2 = a 1 * a 10     -- Geometric sequence property for a₁, a₄, a₁₀

/-- The ratio of the first term to the common difference is 3 -/
theorem arithmetic_geometric_ratio (seq : ArithmeticGeometricSequence) : seq.a 1 / seq.d = 3 := by
  sorry

end arithmetic_geometric_ratio_l2236_223601


namespace chocolate_candy_difference_l2236_223660

/-- The cost difference between chocolate and candy bar -/
def cost_difference (chocolate_cost candy_cost : ℕ) : ℕ :=
  chocolate_cost - candy_cost

/-- Theorem stating the cost difference between chocolate and candy bar -/
theorem chocolate_candy_difference :
  cost_difference 3 2 = 1 := by
  sorry

end chocolate_candy_difference_l2236_223660


namespace dvd_average_price_l2236_223692

/-- Calculates the average price of DVDs bought from two boxes with different prices -/
theorem dvd_average_price (box1_count : ℕ) (box1_price : ℚ) (box2_count : ℕ) (box2_price : ℚ) :
  box1_count = 10 →
  box1_price = 2 →
  box2_count = 5 →
  box2_price = 5 →
  (box1_count * box1_price + box2_count * box2_price) / (box1_count + box2_count : ℚ) = 3 := by
  sorry

#check dvd_average_price

end dvd_average_price_l2236_223692


namespace minimum_at_two_l2236_223673

/-- The function f(x) with parameter t -/
def f (t : ℝ) (x : ℝ) : ℝ := x^3 - 2*t*x^2 + t^2*x

/-- The derivative of f(x) with respect to x -/
def f' (t : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*t*x + t^2

theorem minimum_at_two (t : ℝ) : 
  (∀ x : ℝ, f t x ≥ f t 2) ↔ t = 2 := by sorry

end minimum_at_two_l2236_223673


namespace student_calculation_l2236_223620

theorem student_calculation (x : ℕ) (h : x = 120) : 2 * x - 138 = 102 := by
  sorry

end student_calculation_l2236_223620


namespace intersection_of_M_and_N_l2236_223678

def M : Set ℤ := {0, 1, 2, -1}
def N : Set ℤ := {0, 1, 2, 3}

theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} := by
  sorry

end intersection_of_M_and_N_l2236_223678


namespace minute_hand_half_circle_time_l2236_223607

/-- Represents the number of small divisions on a clock face -/
def clock_divisions : ℕ := 60

/-- Represents the number of minutes the minute hand moves for each small division -/
def minutes_per_division : ℕ := 1

/-- Represents the number of small divisions in half a circle -/
def half_circle_divisions : ℕ := 30

/-- Represents half an hour in minutes -/
def half_hour_minutes : ℕ := 30

theorem minute_hand_half_circle_time :
  half_circle_divisions * minutes_per_division = half_hour_minutes :=
sorry

end minute_hand_half_circle_time_l2236_223607


namespace cost_effectiveness_l2236_223618

/-- Represents the cost-effective choice between two malls --/
inductive Choice
  | MallA
  | MallB
  | Either

/-- Calculates the price per unit for Mall A based on the number of items --/
def mall_a_price (n : ℕ) : ℚ :=
  if n * 4 ≤ 40 then 80 - n * 4
  else 40

/-- Calculates the price per unit for Mall B --/
def mall_b_price : ℚ := 80 * (1 - 0.3)

/-- Determines the cost-effective choice based on the number of employees --/
def cost_effective_choice (num_employees : ℕ) : Choice :=
  if num_employees < 6 then Choice.MallB
  else if num_employees = 6 then Choice.Either
  else Choice.MallA

theorem cost_effectiveness 
  (num_employees : ℕ) : 
  (cost_effective_choice num_employees = Choice.MallB ↔ num_employees < 6) ∧
  (cost_effective_choice num_employees = Choice.Either ↔ num_employees = 6) ∧
  (cost_effective_choice num_employees = Choice.MallA ↔ num_employees > 6) :=
sorry

end cost_effectiveness_l2236_223618


namespace ice_cream_theorem_l2236_223656

theorem ice_cream_theorem (n : ℕ) (h : n > 7) :
  ∃ x y : ℕ, 3 * x + 5 * y = n := by
sorry

end ice_cream_theorem_l2236_223656


namespace fraction_equals_zero_l2236_223669

theorem fraction_equals_zero (x : ℝ) : (x - 2) / (x + 3) = 0 → x = 2 := by
  sorry

end fraction_equals_zero_l2236_223669


namespace blueberry_muffin_probability_l2236_223698

theorem blueberry_muffin_probability :
  let n : ℕ := 7
  let k : ℕ := 5
  let p : ℚ := 3/4
  let q : ℚ := 1 - p
  Nat.choose n k * p^k * q^(n-k) = 5103/16384 := by
  sorry

end blueberry_muffin_probability_l2236_223698


namespace sequence_is_arithmetic_l2236_223623

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of the sequence. -/
def a (n : ℕ) (p q : ℝ) : ℝ := p * n + q

theorem sequence_is_arithmetic (p q : ℝ) :
  IsArithmeticSequence (a · p q) := by
  sorry

end sequence_is_arithmetic_l2236_223623


namespace lines_parallel_iff_m_eq_3_or_neg_1_l2236_223639

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 7 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ x y, line1 m x y ↔ ∃ k, line2 m (x + k) (y + k)

-- Theorem statement
theorem lines_parallel_iff_m_eq_3_or_neg_1 :
  ∀ m : ℝ, parallel m ↔ m = 3 ∨ m = -1 := by sorry

end lines_parallel_iff_m_eq_3_or_neg_1_l2236_223639


namespace train_length_l2236_223604

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 360 → time = 30 → speed * time * (1000 / 3600) = 3000 := by
  sorry

#check train_length

end train_length_l2236_223604


namespace twentieth_term_of_sequence_l2236_223633

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- State the theorem
theorem twentieth_term_of_sequence : 
  arithmetic_sequence 2 4 20 = 78 := by
  sorry

end twentieth_term_of_sequence_l2236_223633


namespace complex_expression_evaluation_l2236_223606

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_expression_evaluation :
  i^3 * (1 - i)^2 = -2 := by sorry

end complex_expression_evaluation_l2236_223606
