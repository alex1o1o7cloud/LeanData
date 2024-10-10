import Mathlib

namespace characterize_M_l3184_318462

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 1 = 0}

-- Define the set M
def M : Set ℝ := {m : ℝ | A ∩ B m = B m}

-- Theorem statement
theorem characterize_M : M = {0, 1/2, 1/3} := by sorry

end characterize_M_l3184_318462


namespace roof_length_width_difference_l3184_318486

/-- Represents a rectangular roof -/
structure RectangularRoof where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Theorem: For a rectangular roof with length 4 times the width and area 784 sq ft,
    the difference between length and width is 42 ft -/
theorem roof_length_width_difference 
  (roof : RectangularRoof)
  (h1 : roof.length = 4 * roof.width)
  (h2 : roof.area = 784)
  (h3 : roof.area = roof.length * roof.width) :
  roof.length - roof.width = 42 := by
  sorry


end roof_length_width_difference_l3184_318486


namespace work_completion_time_l3184_318495

theorem work_completion_time (b_alone : ℝ) (together_time : ℝ) (b_remaining : ℝ) 
  (h1 : b_alone = 28)
  (h2 : together_time = 3)
  (h3 : b_remaining = 21) :
  ∃ a_alone : ℝ, 
    a_alone = 21 ∧ 
    together_time * (1 / a_alone + 1 / b_alone) + b_remaining * (1 / b_alone) = 1 := by
  sorry

end work_completion_time_l3184_318495


namespace unique_solution_values_l3184_318443

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop := a * x^2 - 2 * x + 1 = 0

-- Define the property of having exactly one solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x, quadratic_equation a x

-- Theorem statement
theorem unique_solution_values :
  ∀ a : ℝ, has_unique_solution a ↔ (a = 0 ∨ a = 1) :=
sorry

end unique_solution_values_l3184_318443


namespace xyz_value_l3184_318478

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by
sorry

end xyz_value_l3184_318478


namespace train_length_calculation_l3184_318425

/-- Given a train traveling at a certain speed that crosses a bridge of known length in a specific time, this theorem calculates the length of the train. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (crossing_time : Real) :
  train_speed = 90 * (1000 / 3600) → -- Convert 90 km/hr to m/s
  bridge_length = 275 →
  crossing_time = 30 →
  (train_speed * crossing_time) - bridge_length = 475 := by
  sorry

#check train_length_calculation

end train_length_calculation_l3184_318425


namespace hyperbola_eccentricity_l3184_318469

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and a point P(x₀, y₀) on its right branch such that the difference between
    its distances to the left and right foci is 8, and the product of its
    distances to the two asymptotes is 16/5, prove that the eccentricity of
    the hyperbola is √5/2. -/
theorem hyperbola_eccentricity (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0)
    (heq : x₀^2 / a^2 - y₀^2 / b^2 = 1)
    (hright : x₀ > 0)
    (hfoci : 2 * a = 8)
    (hasymptotes : (b * x₀ - a * y₀) * (b * x₀ + a * y₀) / (a^2 + b^2) = 16/5) :
    let c := Real.sqrt (a^2 + b^2)
    c / a = Real.sqrt 5 / 2 := by
  sorry

end hyperbola_eccentricity_l3184_318469


namespace points_difference_is_integer_impossible_score_difference_l3184_318400

/-- Represents the possible outcomes of a chess game -/
inductive GameOutcome
  | Victory
  | Draw
  | Defeat

/-- Calculates the points scored for a given game outcome -/
def points_scored (outcome : GameOutcome) : ℚ :=
  match outcome with
  | GameOutcome.Victory => 1
  | GameOutcome.Draw => 1/2
  | GameOutcome.Defeat => 0

/-- Calculates the points lost for a given game outcome -/
def points_lost (outcome : GameOutcome) : ℚ :=
  match outcome with
  | GameOutcome.Victory => 0
  | GameOutcome.Draw => 1/2
  | GameOutcome.Defeat => 1

/-- Represents a sequence of game outcomes in a chess tournament -/
def Tournament := List GameOutcome

/-- Calculates the total points scored in a tournament -/
def total_points_scored (tournament : Tournament) : ℚ :=
  tournament.map points_scored |>.sum

/-- Calculates the total points lost in a tournament -/
def total_points_lost (tournament : Tournament) : ℚ :=
  tournament.map points_lost |>.sum

/-- Theorem: The difference between points scored and points lost in any chess tournament is always an integer -/
theorem points_difference_is_integer (tournament : Tournament) :
  ∃ n : ℤ, total_points_scored tournament - total_points_lost tournament = n :=
sorry

/-- Corollary: It's impossible to have scored exactly 3.5 points more than lost -/
theorem impossible_score_difference (tournament : Tournament) :
  total_points_scored tournament - total_points_lost tournament ≠ 7/2 :=
sorry

end points_difference_is_integer_impossible_score_difference_l3184_318400


namespace gcd_180_270_l3184_318422

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end gcd_180_270_l3184_318422


namespace ellipse_eccentricity_l3184_318470

/-- The eccentricity of the ellipse x²/25 + y²/16 = 1 is 3/5 -/
theorem ellipse_eccentricity :
  let e : ℝ := Real.sqrt (1 - 16 / 25)
  e = 3 / 5 := by sorry

end ellipse_eccentricity_l3184_318470


namespace rectangle_square_assembly_l3184_318426

structure Rectangle where
  width : ℝ
  height : ℝ

def Square (s : ℝ) : Rectangle :=
  { width := s, height := s }

def totalArea (rectangles : List Rectangle) : ℝ :=
  rectangles.map (λ r => r.width * r.height) |>.sum

def isSquare (area : ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ s * s = area

theorem rectangle_square_assembly
  (s : ℝ)
  (r1 : Rectangle)
  (r2 : Rectangle)
  (h1 : r1.width = 10 ∧ r1.height = 24)
  (h2 : r2 ∈ [
    Rectangle.mk 2 24,
    Rectangle.mk 19 17.68,
    Rectangle.mk 34 10,
    Rectangle.mk 34 44,
    Rectangle.mk 14 24,
    Rectangle.mk 14 17,
    Rectangle.mk 24 38
  ]) :
  isSquare (totalArea [Square s, Square s, r1, r2]) :=
by sorry

end rectangle_square_assembly_l3184_318426


namespace mod_23_equivalence_l3184_318464

theorem mod_23_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 57846 ≡ n [ZMOD 23] ∧ n = 1 := by
  sorry

end mod_23_equivalence_l3184_318464


namespace F_of_4_f_of_5_equals_21_l3184_318468

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 2
def F (a b : ℝ) : ℝ := a * b + b^2

-- State the theorem
theorem F_of_4_f_of_5_equals_21 : F 4 (f 5) = 21 := by
  sorry

end F_of_4_f_of_5_equals_21_l3184_318468


namespace one_third_of_6_3_l3184_318439

theorem one_third_of_6_3 : (6.3 : ℚ) / 3 = 21 / 10 := by
  sorry

end one_third_of_6_3_l3184_318439


namespace mutually_exclusive_events_l3184_318421

/-- Represents the outcome of a single shot --/
inductive ShotOutcome
  | Hit
  | Miss

/-- Represents the outcome of two shots --/
def TwoShotOutcome := (ShotOutcome × ShotOutcome)

/-- The event of hitting the target at least once --/
def hitAtLeastOnce (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Hit ∨ outcome.2 = ShotOutcome.Hit

/-- The event of missing the target both times --/
def missBothTimes (outcome : TwoShotOutcome) : Prop :=
  outcome.1 = ShotOutcome.Miss ∧ outcome.2 = ShotOutcome.Miss

/-- Theorem stating that "missing the target both times" is the mutually exclusive event of "hitting the target at least once" --/
theorem mutually_exclusive_events :
  ∀ (outcome : TwoShotOutcome), hitAtLeastOnce outcome ↔ ¬(missBothTimes outcome) :=
sorry

end mutually_exclusive_events_l3184_318421


namespace inverse_proportion_problem_l3184_318406

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ → ℝ) 
  (h1 : InverselyProportional (x 40) (y 5))
  (h2 : x 40 = 40)
  (h3 : y 5 = 5) :
  y 8 = 25 := by
  sorry

end inverse_proportion_problem_l3184_318406


namespace rectangle_area_l3184_318490

/-- Given a rectangle divided into 18 congruent squares, where the length is three times
    the width and the diagonal of one small square is 5 cm, the area of the entire
    rectangular region is 112.5 square cm. -/
theorem rectangle_area (n m : ℕ) (s : ℝ) : 
  n * m = 18 →
  n = 2 * m →
  s^2 + s^2 = 5^2 →
  (n * s) * (m * s) = 112.5 := by
  sorry

end rectangle_area_l3184_318490


namespace sum_of_distances_inequality_minimum_value_of_expression_l3184_318429

-- Part 1
theorem sum_of_distances_inequality (x y : ℝ) :
  Real.sqrt (x^2 + y^2) + Real.sqrt ((x-1)^2 + y^2) +
  Real.sqrt (x^2 + (y-1)^2) + Real.sqrt ((x-1)^2 + (y-1)^2) ≥ 2 * Real.sqrt 2 :=
sorry

-- Part 2
theorem minimum_value_of_expression :
  ∃ (min : ℝ), min = 8 ∧
  ∀ (a b : ℝ), abs a ≤ Real.sqrt 2 → b > 0 →
  (a - b)^2 + (Real.sqrt (2 - a^2) - 9 / b)^2 ≥ min :=
sorry

end sum_of_distances_inequality_minimum_value_of_expression_l3184_318429


namespace two_from_four_combination_l3184_318461

theorem two_from_four_combination : Nat.choose 4 2 = 6 := by sorry

end two_from_four_combination_l3184_318461


namespace original_time_calculation_l3184_318456

theorem original_time_calculation (original_speed : ℝ) (original_time : ℝ) 
  (h1 : original_speed > 0) (h2 : original_time > 0) : 
  (original_time / 0.8 = original_time + 10) → original_time = 40 := by
  sorry

end original_time_calculation_l3184_318456


namespace sunglasses_and_caps_probability_l3184_318440

theorem sunglasses_and_caps_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_cap_given_sunglasses : ℚ) : 
  total_sunglasses = 60 → 
  total_caps = 40 → 
  prob_cap_given_sunglasses = 1/3 → 
  (total_sunglasses * prob_cap_given_sunglasses : ℚ) / total_caps = 1/2 :=
by sorry

end sunglasses_and_caps_probability_l3184_318440


namespace mean_equality_problem_l3184_318459

theorem mean_equality_problem (z : ℝ) : 
  (7 + 11 + 5 + 9) / 4 = (15 + z) / 2 → z = 1 := by
  sorry

end mean_equality_problem_l3184_318459


namespace min_cookies_eaten_l3184_318404

/-- Represents the number of cookies at each stage of the process -/
structure CookieCount where
  initial : ℕ
  after_first : ℕ
  after_second : ℕ
  after_third : ℕ
  evening : ℕ

/-- Defines the cookie distribution process -/
def distribute_cookies (c : CookieCount) : Prop :=
  c.after_first = (2 * (c.initial - 1)) / 3 ∧
  c.after_second = (2 * (c.after_first - 1)) / 3 ∧
  c.after_third = (2 * (c.after_second - 1)) / 3 ∧
  c.evening = c.after_third - 1

/-- Defines the evening distribution condition -/
def evening_distribution (c : CookieCount) (n : ℕ) : Prop :=
  c.evening = 3 * n

/-- Defines the condition that no cookies are broken -/
def no_broken_cookies (c : CookieCount) : Prop :=
  c.initial % 1 = 0 ∧
  c.after_first % 1 = 0 ∧
  c.after_second % 1 = 0 ∧
  c.after_third % 1 = 0 ∧
  c.evening % 1 = 0

/-- Theorem stating the minimum number of cookies Xiao Wang could have eaten -/
theorem min_cookies_eaten (c : CookieCount) (n : ℕ) :
  distribute_cookies c →
  evening_distribution c n →
  no_broken_cookies c →
  (c.initial - c.after_first) = 6 ∧ n = 7 := by
  sorry

#check min_cookies_eaten

end min_cookies_eaten_l3184_318404


namespace sum_middle_m_value_l3184_318444

/-- An arithmetic sequence with 3m terms -/
structure ArithmeticSequence (m : ℕ) where
  sum_first_2m : ℝ
  sum_last_2m : ℝ

/-- The sum of the middle m terms in an arithmetic sequence -/
def sum_middle_m (seq : ArithmeticSequence m) : ℝ := sorry

theorem sum_middle_m_value {m : ℕ} (seq : ArithmeticSequence m)
  (h1 : seq.sum_first_2m = 100)
  (h2 : seq.sum_last_2m = 200) :
  sum_middle_m seq = 75 := by sorry

end sum_middle_m_value_l3184_318444


namespace circle_tangents_k_range_l3184_318493

-- Define the circle C
def C (k : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*k*x + 2*y + k^2 = 0

-- Define the point P
def P : ℝ × ℝ := (1, -1)

-- Define the condition for two tangents
def has_two_tangents (k : ℝ) : Prop := ∃ (t1 t2 : ℝ × ℝ), t1 ≠ t2 ∧ C k t1.1 t1.2 ∧ C k t2.1 t2.2

-- Theorem statement
theorem circle_tangents_k_range :
  ∀ k : ℝ, has_two_tangents k → (k > 0 ∨ k < -2) :=
by sorry

end circle_tangents_k_range_l3184_318493


namespace unspent_portion_after_transfer_l3184_318488

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Calculates the unspent portion of a credit card's limit after a balance transfer -/
def unspentPortionAfterTransfer (gold : CreditCard) (platinum : CreditCard) : ℝ :=
  sorry

/-- Theorem stating the unspent portion of the platinum card's limit after transfer -/
theorem unspent_portion_after_transfer
  (gold : CreditCard)
  (platinum : CreditCard)
  (h1 : platinum.limit = 2 * gold.limit)
  (h2 : ∃ X : ℝ, gold.balance = X * gold.limit)
  (h3 : platinum.balance = (1 / 7) * platinum.limit) :
  unspentPortionAfterTransfer gold platinum = (12 - 7 * (gold.balance / gold.limit)) / 14 :=
  sorry

end unspent_portion_after_transfer_l3184_318488


namespace solution_difference_l3184_318449

theorem solution_difference (p q : ℝ) : 
  ((p - 5) * (p + 5) = 26 * p - 130) →
  ((q - 5) * (q + 5) = 26 * q - 130) →
  p ≠ q →
  p > q →
  p - q = 16 := by sorry

end solution_difference_l3184_318449


namespace finite_squared_nilpotent_matrices_l3184_318491

/-- Given a 3x3 matrix A with real entries such that A^4 = 0, 
    the set of all possible A^2 matrices is finite. -/
theorem finite_squared_nilpotent_matrices 
  (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : 
  Set.Finite {B : Matrix (Fin 3) (Fin 3) ℝ | ∃ (A : Matrix (Fin 3) (Fin 3) ℝ), A ^ 4 = 0 ∧ B = A ^ 2} :=
sorry

end finite_squared_nilpotent_matrices_l3184_318491


namespace dans_money_was_three_l3184_318428

/-- Dan's initial amount of money, given he bought a candy bar and has some money left -/
def dans_initial_money (candy_bar_cost : ℝ) (money_left : ℝ) : ℝ :=
  candy_bar_cost + money_left

/-- Theorem stating Dan's initial money was $3 -/
theorem dans_money_was_three :
  dans_initial_money 1 2 = 3 := by
  sorry

end dans_money_was_three_l3184_318428


namespace sqrt_18_times_sqrt_72_l3184_318460

theorem sqrt_18_times_sqrt_72 : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end sqrt_18_times_sqrt_72_l3184_318460


namespace cistern_filling_time_l3184_318465

/-- Given a cistern that can be emptied by a tap in 10 hours, and when both this tap and another tap
    are opened simultaneously the cistern gets filled in 30/7 hours, prove that the time it takes
    for the other tap alone to fill the cistern is 3 hours. -/
theorem cistern_filling_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) :
  empty_rate = 10 →
  combined_fill_time = 30 / 7 →
  1 / fill_time - 1 / empty_rate = 1 / combined_fill_time →
  fill_time = 3 := by
  sorry

end cistern_filling_time_l3184_318465


namespace lottery_tax_percentage_l3184_318438

/-- Proves that the percentage of lottery winnings paid for tax is 20% given the specified conditions --/
theorem lottery_tax_percentage (winnings : ℝ) (processing_fee : ℝ) (take_home : ℝ) : 
  winnings = 50 → processing_fee = 5 → take_home = 35 → 
  (winnings - (take_home + processing_fee)) / winnings * 100 = 20 := by
sorry

end lottery_tax_percentage_l3184_318438


namespace complement_intersection_theorem_l3184_318432

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end complement_intersection_theorem_l3184_318432


namespace positive_solution_sum_l3184_318484

theorem positive_solution_sum (a b : ℕ+) (x : ℝ) : 
  x^2 + 10*x = 93 →
  x > 0 →
  x = Real.sqrt a - b →
  a + b = 123 := by
sorry

end positive_solution_sum_l3184_318484


namespace test_total_points_l3184_318454

theorem test_total_points (total_questions : ℕ) (four_point_questions : ℕ) 
  (h1 : total_questions = 40)
  (h2 : four_point_questions = 10) :
  let two_point_questions := total_questions - four_point_questions
  total_questions * 2 + four_point_questions * 2 = 100 :=
by sorry

end test_total_points_l3184_318454


namespace three_digit_divisibility_l3184_318409

theorem three_digit_divisibility (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : (a + b) % 7 = 0) : (101 * a + 10 * b) % 7 = 0 := by
  sorry

end three_digit_divisibility_l3184_318409


namespace opposite_face_of_four_l3184_318419

/-- Represents the six faces of a cube -/
inductive Face
| A | B | C | D | E | F

/-- Assigns numbers to the faces of the cube -/
def face_value : Face → ℕ
| Face.A => 3
| Face.B => 4
| Face.C => 5
| Face.D => 6
| Face.E => 7
| Face.F => 8

/-- Defines the opposite face relation -/
def opposite : Face → Face
| Face.A => Face.F
| Face.B => Face.E
| Face.C => Face.D
| Face.D => Face.C
| Face.E => Face.B
| Face.F => Face.A

theorem opposite_face_of_four (h : ∀ (f : Face), face_value f + face_value (opposite f) = 11) :
  face_value (opposite Face.B) = 7 := by
  sorry

end opposite_face_of_four_l3184_318419


namespace cottage_pie_mince_usage_l3184_318405

/-- Given information about a school cafeteria's use of ground mince for lasagnas and cottage pies,
    prove that each cottage pie uses 3 pounds of ground mince. -/
theorem cottage_pie_mince_usage
  (total_dishes : Nat)
  (lasagna_count : Nat)
  (cottage_pie_count : Nat)
  (total_mince : Nat)
  (mince_per_lasagna : Nat)
  (h1 : total_dishes = lasagna_count + cottage_pie_count)
  (h2 : total_dishes = 100)
  (h3 : lasagna_count = 100)
  (h4 : cottage_pie_count = 100)
  (h5 : total_mince = 500)
  (h6 : mince_per_lasagna = 2) :
  (total_mince - lasagna_count * mince_per_lasagna) / cottage_pie_count = 3 := by
  sorry

end cottage_pie_mince_usage_l3184_318405


namespace quadratic_inequality_solution_set_l3184_318413

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end quadratic_inequality_solution_set_l3184_318413


namespace problem_solution_l3184_318455

theorem problem_solution (x y z : ℝ) 
  (h1 : 3 = 0.15 * x)
  (h2 : 3 = 0.25 * y)
  (h3 : z = 0.30 * y) :
  x - y + z = 11.6 := by
sorry

end problem_solution_l3184_318455


namespace task_completion_choices_l3184_318457

theorem task_completion_choices (method1 method2 : Finset Nat) : 
  method1.card = 3 → method2.card = 5 → method1 ∩ method2 = ∅ → 
  (method1 ∪ method2).card = 8 :=
by sorry

end task_completion_choices_l3184_318457


namespace octagon_area_theorem_l3184_318497

/-- Represents an octagon with given width and height -/
structure Octagon where
  width : ℕ
  height : ℕ

/-- Calculates the area of the octagon -/
def octagonArea (o : Octagon) : ℕ :=
  -- The actual calculation is not provided, as it should be part of the proof
  sorry

/-- Theorem stating that an octagon with width 5 and height 8 has an area of 30 square units -/
theorem octagon_area_theorem (o : Octagon) (h1 : o.width = 5) (h2 : o.height = 8) : 
  octagonArea o = 30 := by
  sorry

end octagon_area_theorem_l3184_318497


namespace jackson_chairs_l3184_318492

/-- The number of chairs Jackson needs to buy for his restaurant -/
def total_chairs (tables_with_4_seats tables_with_6_seats : ℕ) : ℕ :=
  tables_with_4_seats * 4 + tables_with_6_seats * 6

/-- Proof that Jackson needs to buy 96 chairs -/
theorem jackson_chairs : total_chairs 6 12 = 96 := by
  sorry

end jackson_chairs_l3184_318492


namespace arithmetic_geometric_sequence_l3184_318450

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 = 1 →                     -- first term condition
  d ≠ 0 →                       -- common difference not zero
  (a 2)^2 = a 1 * a 5 →         -- geometric sequence condition
  d = 2 := by
sorry

end arithmetic_geometric_sequence_l3184_318450


namespace circle_problem_l3184_318403

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 = c.radius^2

def tangent_to_line (c : Circle) (a b d : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x + b * y = d ∧ (c.center.1 - x)^2 + (c.center.2 - y)^2 = c.radius^2

def center_on_line (c : Circle) (m b : ℝ) : Prop :=
  c.center.2 = m * c.center.1 + b

-- Define the theorem
theorem circle_problem :
  ∃ (c : Circle),
    passes_through c (2, -1) ∧
    tangent_to_line c 1 1 1 ∧
    center_on_line c (-2) 0 ∧
    c.center = (1, -2) ∧
    c.radius^2 = 2 ∧
    (∀ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = 2 ↔ passes_through c (x, y)) ∧
    (let chord_length := 2 * Real.sqrt (c.radius^2 - (3 * c.center.1 + 4 * c.center.2)^2 / 25);
     chord_length = 2) :=
by sorry

end circle_problem_l3184_318403


namespace equation_solution_l3184_318485

theorem equation_solution (x y z : ℚ) 
  (eq1 : x - 4*y - 2*z = 0) 
  (eq2 : 3*x + 2*y - z = 0) 
  (z_neq_zero : z ≠ 0) : 
  (x^2 - 5*x*y) / (2*y^2 + z^2) = 164/147 := by
  sorry

end equation_solution_l3184_318485


namespace distribution_problem_l3184_318401

/-- The number of ways to distribute n indistinguishable objects among k distinct groups,
    with each group receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The problem statement -/
theorem distribution_problem :
  distribute 12 6 = 462 := by sorry

end distribution_problem_l3184_318401


namespace largest_prime_factor_of_1001_l3184_318446

theorem largest_prime_factor_of_1001 : 
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ 1001 ∧ ∀ (q : Nat), Nat.Prime q → q ∣ 1001 → q ≤ p ∧ p = 13 :=
by sorry

end largest_prime_factor_of_1001_l3184_318446


namespace opposite_of_sqrt_three_l3184_318416

theorem opposite_of_sqrt_three : -(Real.sqrt 3) = -Real.sqrt 3 := by
  sorry

end opposite_of_sqrt_three_l3184_318416


namespace liam_target_time_l3184_318476

-- Define Mia's run
def mia_distance : ℕ := 5
def mia_time : ℕ := 45

-- Define Liam's initial run
def liam_initial_distance : ℕ := 3

-- Define the relationship between Liam and Mia's times
def liam_initial_time : ℚ := mia_time / 3

-- Define Liam's target distance
def liam_target_distance : ℕ := 7

-- Theorem to prove
theorem liam_target_time : 
  (liam_target_distance : ℚ) * (liam_initial_time / liam_initial_distance) = 35 := by
  sorry

end liam_target_time_l3184_318476


namespace train_delay_l3184_318480

theorem train_delay (car_time train_time : ℝ) : 
  car_time = 4.5 → 
  car_time + train_time = 11 → 
  train_time - car_time = 2 :=
by
  sorry

end train_delay_l3184_318480


namespace triangle_equality_equilateral_is_isosceles_equilateral_is_acute_equilateral_is_oblique_l3184_318420

/-- A triangle with side lengths satisfying a² + b² + c² = ab + bc + ca is equilateral -/
theorem triangle_equality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (eq : a^2 + b^2 + c^2 = a*b + b*c + c*a) : a = b ∧ b = c := by
  sorry

/-- An equilateral triangle is isosceles -/
theorem equilateral_is_isosceles (a b c : ℝ) (h : a = b ∧ b = c) : 
  a = b ∨ b = c ∨ a = c := by
  sorry

/-- An equilateral triangle is acute-angled -/
theorem equilateral_is_acute (a b c : ℝ) (h : a = b ∧ b = c) (pos : 0 < a) : 
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2 := by
  sorry

/-- An equilateral triangle is oblique (not right-angled) -/
theorem equilateral_is_oblique (a b c : ℝ) (h : a = b ∧ b = c) (pos : 0 < a) : 
  a^2 + b^2 ≠ c^2 ∧ b^2 + c^2 ≠ a^2 ∧ c^2 + a^2 ≠ b^2 := by
  sorry

end triangle_equality_equilateral_is_isosceles_equilateral_is_acute_equilateral_is_oblique_l3184_318420


namespace expression_simplification_l3184_318437

theorem expression_simplification (x y : ℝ) (hx : x ≥ 0) :
  (3 / 5) * Real.sqrt (x * y^2) / (-(4 / 15) * Real.sqrt (y / x)) * (-(5 / 6) * Real.sqrt (x^3 * y)) =
  (15 * x^2 * y * Real.sqrt x) / 8 :=
by sorry

end expression_simplification_l3184_318437


namespace largest_common_number_l3184_318417

def is_in_first_sequence (n : ℕ) : Prop := ∃ k : ℕ, n = 1 + 8 * k

def is_in_second_sequence (n : ℕ) : Prop := ∃ m : ℕ, n = 4 + 9 * m

def is_in_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 250

theorem largest_common_number :
  (is_in_first_sequence 193 ∧ is_in_second_sequence 193 ∧ is_in_range 193) ∧
  ∀ n : ℕ, is_in_first_sequence n → is_in_second_sequence n → is_in_range n → n ≤ 193 :=
by sorry

end largest_common_number_l3184_318417


namespace continuous_stripe_probability_l3184_318467

/-- A regular tetrahedron -/
structure Tetrahedron :=
  (faces : Fin 4)
  (vertices : Fin 4)
  (edges : Fin 6)

/-- A stripe configuration on a tetrahedron -/
def StripeConfiguration := Tetrahedron → Fin 3

/-- Predicate for a continuous stripe encircling the tetrahedron -/
def IsContinuousStripe (config : StripeConfiguration) : Prop :=
  sorry

/-- The total number of possible stripe configurations -/
def TotalConfigurations : ℕ := 3^4

/-- The number of stripe configurations that result in a continuous stripe -/
def ContinuousStripeConfigurations : ℕ := 2^4

/-- The probability of a continuous stripe encircling the tetrahedron -/
def ProbabilityContinuousStripe : ℚ :=
  ContinuousStripeConfigurations / TotalConfigurations

theorem continuous_stripe_probability :
  ProbabilityContinuousStripe = 16 / 81 :=
sorry

end continuous_stripe_probability_l3184_318467


namespace zeros_of_g_l3184_318487

-- Define the power function f
def f : ℝ → ℝ := fun x => x^3

-- Define the function g
def g : ℝ → ℝ := fun x => f x - x

-- State the theorem
theorem zeros_of_g :
  (f 2 = 8) →
  (∀ x : ℝ, g x = 0 ↔ x = 0 ∨ x = 1 ∨ x = -1) :=
by sorry

end zeros_of_g_l3184_318487


namespace min_value_theorem_l3184_318499

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (min : ℝ), min = 5 + 2 * Real.sqrt 6 ∧ ∀ (x : ℝ), (3 / a + 2 / b) ≥ x := by
  sorry

end min_value_theorem_l3184_318499


namespace purchase_in_fourth_month_l3184_318466

/-- Represents the financial state of the family --/
structure FamilyFinance where
  monthlyIncome : ℕ
  monthlyExpenses : ℕ
  initialSavings : ℕ
  furnitureCost : ℕ

/-- Calculates the month when the family can make the purchase --/
def purchaseMonth (finance : FamilyFinance) : ℕ :=
  let monthlySavings := finance.monthlyIncome - finance.monthlyExpenses
  let additionalRequired := finance.furnitureCost - finance.initialSavings
  (additionalRequired + monthlySavings - 1) / monthlySavings + 1

/-- The main theorem stating that the family can make the purchase in the 4th month --/
theorem purchase_in_fourth_month (finance : FamilyFinance) 
  (h1 : finance.monthlyIncome = 150000)
  (h2 : finance.monthlyExpenses = 115000)
  (h3 : finance.initialSavings = 45000)
  (h4 : finance.furnitureCost = 127000) :
  purchaseMonth finance = 4 := by
  sorry

#eval purchaseMonth { 
  monthlyIncome := 150000, 
  monthlyExpenses := 115000, 
  initialSavings := 45000, 
  furnitureCost := 127000 
}

end purchase_in_fourth_month_l3184_318466


namespace problem_solution_l3184_318458

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + x + 1 ≥ 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, (x > 1 → x > 2) ∧ ¬(x > 2 → x > 1)

-- Theorem to prove
theorem problem_solution : p ∧ ¬q := by sorry

end problem_solution_l3184_318458


namespace line_slope_intercept_product_l3184_318496

/-- Given a line passing through points (0, -4) and (4, 4), prove that the product of its slope and y-intercept equals -8. -/
theorem line_slope_intercept_product : 
  ∀ (m b : ℝ), 
  (∀ x y : ℝ, y = m * x + b → (x = 0 ∧ y = -4) ∨ (x = 4 ∧ y = 4)) → 
  m * b = -8 := by
sorry

end line_slope_intercept_product_l3184_318496


namespace bill_face_value_l3184_318427

/-- Proves that given a true discount of 360 and a banker's discount of 432, 
    the face value of the bill is 1800. -/
theorem bill_face_value (TD : ℕ) (BD : ℕ) (FV : ℕ) : 
  TD = 360 → BD = 432 → FV = (TD^2) / (BD - TD) → FV = 1800 := by
  sorry

#check bill_face_value

end bill_face_value_l3184_318427


namespace trishul_investment_percentage_l3184_318483

/-- Represents the investment amounts of Vishal, Trishul, and Raghu -/
structure Investments where
  vishal : ℝ
  trishul : ℝ
  raghu : ℝ

/-- The conditions of the investment problem -/
def InvestmentConditions (i : Investments) : Prop :=
  i.vishal = 1.1 * i.trishul ∧
  i.raghu = 2300 ∧
  i.vishal + i.trishul + i.raghu = 6647

/-- The theorem stating that Trishul invested 10% less than Raghu -/
theorem trishul_investment_percentage (i : Investments) 
  (h : InvestmentConditions i) : 
  (i.raghu - i.trishul) / i.raghu = 0.1 := by
  sorry

end trishul_investment_percentage_l3184_318483


namespace digit_2000th_position_l3184_318442

/-- The sequence of digits formed by concatenating consecutive positive integers starting from 1 -/
def concatenatedSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => (concatenatedSequence n) * 10 + ((n + 1) % 10)

/-- The digit at a given position in the concatenated sequence -/
def digitAtPosition (pos : ℕ) : ℕ :=
  (concatenatedSequence pos) % 10

theorem digit_2000th_position :
  digitAtPosition 1999 = 0 := by
  sorry

end digit_2000th_position_l3184_318442


namespace unique_valid_n_l3184_318412

def is_valid_n (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    n = 10 * a + b ∧
    100 * a + 10 * c + b = 6 * n

theorem unique_valid_n :
  ∃! n : ℕ, n ≥ 10 ∧ is_valid_n n ∧ n = 18 :=
sorry

end unique_valid_n_l3184_318412


namespace x_positive_necessary_not_sufficient_for_abs_x_minus_one_less_than_one_l3184_318445

theorem x_positive_necessary_not_sufficient_for_abs_x_minus_one_less_than_one :
  (∃ x : ℝ, x > 0 ∧ ¬(|x - 1| < 1)) ∧
  (∀ x : ℝ, |x - 1| < 1 → x > 0) :=
by sorry

end x_positive_necessary_not_sufficient_for_abs_x_minus_one_less_than_one_l3184_318445


namespace sqrt_four_cubes_sum_l3184_318415

theorem sqrt_four_cubes_sum : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end sqrt_four_cubes_sum_l3184_318415


namespace correlation_count_correlated_relationships_l3184_318411

/-- Represents a relationship between two quantities -/
structure Relationship where
  name : String
  has_correlation : Bool

/-- The set of relationships given in the problem -/
def relationships : List Relationship := [
  ⟨"cube volume-edge length", false⟩,
  ⟨"yield-fertilizer", true⟩,
  ⟨"height-age", true⟩,
  ⟨"expenses-income", true⟩,
  ⟨"electricity consumption-price", false⟩
]

/-- The correct answer is that exactly three relationships have correlations -/
theorem correlation_count :
  (relationships.filter (fun r => r.has_correlation)).length = 3 := by
  sorry

/-- The relationships with correlations are yield-fertilizer, height-age, and expenses-income -/
theorem correlated_relationships :
  (relationships.filter (fun r => r.has_correlation)).map (fun r => r.name) =
    ["yield-fertilizer", "height-age", "expenses-income"] := by
  sorry

end correlation_count_correlated_relationships_l3184_318411


namespace sugar_salt_difference_is_two_l3184_318453

/-- A baking recipe with specified amounts of ingredients -/
structure Recipe where
  sugar : ℕ
  flour : ℕ
  salt : ℕ

/-- The amount of ingredients Mary has already added -/
structure Added where
  flour : ℕ

/-- Calculate the difference between required sugar and salt -/
def sugarSaltDifference (recipe : Recipe) : ℤ :=
  recipe.sugar - recipe.salt

/-- Theorem: The difference between required sugar and salt is 2 cups -/
theorem sugar_salt_difference_is_two (recipe : Recipe) (added : Added) :
  recipe.sugar = 11 →
  recipe.flour = 6 →
  recipe.salt = 9 →
  added.flour = 12 →
  sugarSaltDifference recipe = 2 := by
  sorry

#eval sugarSaltDifference { sugar := 11, flour := 6, salt := 9 }

end sugar_salt_difference_is_two_l3184_318453


namespace at_least_one_leq_neg_two_l3184_318472

theorem at_least_one_leq_neg_two (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by
sorry

end at_least_one_leq_neg_two_l3184_318472


namespace basketball_team_points_l3184_318498

theorem basketball_team_points (x : ℚ) (y : ℕ) : 
  (1 / 3 : ℚ) * x + (1 / 5 : ℚ) * x + 18 + y = x → 
  y ≤ 21 → 
  y = 15 :=
by sorry

end basketball_team_points_l3184_318498


namespace cube_diagonal_length_l3184_318423

theorem cube_diagonal_length (surface_area : ℝ) (h : surface_area = 864) :
  let side_length := Real.sqrt (surface_area / 6)
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 12 * Real.sqrt 3 := by
  sorry

end cube_diagonal_length_l3184_318423


namespace total_beads_count_l3184_318477

/-- Represents the number of beads of each color in Sue's necklace --/
structure BeadCounts where
  purple : ℕ
  blue : ℕ
  green : ℕ
  red : ℕ

/-- Defines the conditions for Sue's necklace --/
def necklace_conditions (counts : BeadCounts) : Prop :=
  counts.purple = 7 ∧
  counts.blue = 2 * counts.purple ∧
  counts.green = counts.blue + 11 ∧
  counts.red = counts.green / 2 ∧
  (counts.purple + counts.blue + counts.green + counts.red) % 2 = 0

/-- The theorem to be proved --/
theorem total_beads_count (counts : BeadCounts) 
  (h : necklace_conditions counts) : 
  counts.purple + counts.blue + counts.green + counts.red = 58 := by
  sorry

end total_beads_count_l3184_318477


namespace largest_integer_squared_less_than_ten_million_l3184_318441

theorem largest_integer_squared_less_than_ten_million :
  ∃ (n : ℕ), n > 0 ∧ n^2 < 10000000 ∧ ∀ (m : ℕ), m > n → m^2 ≥ 10000000 :=
by sorry

end largest_integer_squared_less_than_ten_million_l3184_318441


namespace multiplication_equation_l3184_318471

theorem multiplication_equation (m : ℕ) : 72519 * m = 724827405 → m = 9999 := by
  sorry

end multiplication_equation_l3184_318471


namespace shortest_light_path_length_shortest_light_path_equals_12_l3184_318452

/-- The shortest path length of a light ray reflecting off the x-axis -/
theorem shortest_light_path_length : ℝ :=
  let A : ℝ × ℝ := (-3, 9)
  let C : ℝ × ℝ := (2, 3)  -- Center of the circle
  let r : ℝ := 1  -- Radius of the circle
  let C' : ℝ × ℝ := (2, -3)  -- Reflection of C across x-axis
  let AC' : ℝ := Real.sqrt ((-3 - 2)^2 + (9 - (-3))^2)
  AC' - r

theorem shortest_light_path_equals_12 :
  shortest_light_path_length = 12 := by sorry

end shortest_light_path_length_shortest_light_path_equals_12_l3184_318452


namespace job_choice_diploma_percentage_l3184_318479

theorem job_choice_diploma_percentage :
  let total_population : ℝ := 100
  let no_diploma_with_job : ℝ := 18
  let with_job_choice : ℝ := 40
  let with_diploma : ℝ := 37
  let without_job_choice : ℝ := total_population - with_job_choice
  let with_diploma_without_job : ℝ := with_diploma - (with_job_choice - no_diploma_with_job)
  (with_diploma_without_job / without_job_choice) * 100 = 25 := by
sorry

end job_choice_diploma_percentage_l3184_318479


namespace odometer_sum_squares_l3184_318489

/-- Represents a 3-digit number abc where a, b, c are single digits --/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  a_positive : a ≥ 1
  sum_constraint : a + b + c = 8

/-- Represents the odometer readings for Denise's trip --/
structure OdometerReadings where
  initial : ThreeDigitNumber
  final : ThreeDigitNumber
  final_swap : final.a = initial.b ∧ final.b = initial.a ∧ final.c = initial.c

/-- Represents Denise's trip --/
structure Trip where
  readings : OdometerReadings
  hours : ℕ
  hours_positive : hours > 0
  speed : ℕ
  speed_eq : speed = 48
  distance_constraint : 90 * (readings.initial.b - readings.initial.a) = hours * speed

theorem odometer_sum_squares (t : Trip) : 
  t.readings.initial.a ^ 2 + t.readings.initial.b ^ 2 + t.readings.initial.c ^ 2 = 26 := by
  sorry

end odometer_sum_squares_l3184_318489


namespace dog_food_duration_l3184_318410

/-- Given a dog's feeding schedule and a bag of dog food, calculate how many days the food will last. -/
theorem dog_food_duration (morning_food evening_food bag_size : ℕ) : 
  morning_food = 1 → 
  evening_food = 1 → 
  bag_size = 32 → 
  (bag_size / (morning_food + evening_food) : ℕ) = 16 := by
sorry

end dog_food_duration_l3184_318410


namespace problem_solution_l3184_318430

theorem problem_solution : 
  ∃ x : ℝ, (28 + x / 69) * 69 = 1980 ∧ x = 1952 := by
  sorry

end problem_solution_l3184_318430


namespace pizza_piece_cost_l3184_318431

/-- Represents the cost of pizzas and their division into pieces -/
structure PizzaPurchase where
  totalCost : ℕ        -- Total cost in dollars
  numPizzas : ℕ        -- Number of pizzas
  piecesPerPizza : ℕ   -- Number of pieces each pizza is cut into

/-- Calculates the cost per piece of pizza -/
def costPerPiece (purchase : PizzaPurchase) : ℚ :=
  (purchase.totalCost : ℚ) / (purchase.numPizzas * purchase.piecesPerPizza)

/-- Theorem: Given 4 pizzas cost $80 and each pizza is cut into 5 pieces, 
    the cost per piece is $4 -/
theorem pizza_piece_cost : 
  let purchase := PizzaPurchase.mk 80 4 5
  costPerPiece purchase = 4 := by
  sorry

end pizza_piece_cost_l3184_318431


namespace sin_50_cos_80_cos_160_l3184_318448

theorem sin_50_cos_80_cos_160 :
  Real.sin (50 * π / 180) * Real.cos (80 * π / 180) * Real.cos (160 * π / 180) = -1/8 := by
sorry

end sin_50_cos_80_cos_160_l3184_318448


namespace bread_lasts_three_days_l3184_318463

/-- Represents the number of days bread will last for a household -/
def days_bread_lasts (
  household_members : ℕ)
  (breakfast_slices_per_member : ℕ)
  (snack_slices_per_member : ℕ)
  (slices_per_loaf : ℕ)
  (number_of_loaves : ℕ) : ℕ :=
  let total_slices := number_of_loaves * slices_per_loaf
  let daily_consumption := household_members * (breakfast_slices_per_member + snack_slices_per_member)
  total_slices / daily_consumption

/-- Theorem stating that 5 loaves of bread will last 3 days for a family of 4 -/
theorem bread_lasts_three_days :
  days_bread_lasts 4 3 2 12 5 = 3 := by
  sorry

end bread_lasts_three_days_l3184_318463


namespace unique_solution_l3184_318433

/-- The number of communications between any n-2 people -/
def communications (n : ℕ) : ℕ := 3^(Nat.succ 0)

/-- The theorem stating that 5 is the only solution -/
theorem unique_solution :
  ∀ n : ℕ,
  (n > 0) →
  (∀ m : ℕ, m = communications n) →
  (∀ i j : Fin n, i ≠ j → (∃! x : ℕ, x ≤ 1)) →
  n = 5 :=
sorry

end unique_solution_l3184_318433


namespace intersection_nonempty_implies_b_range_l3184_318481

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_nonempty_implies_b_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) →
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
sorry

end intersection_nonempty_implies_b_range_l3184_318481


namespace sqrt_65_minus_1_bound_l3184_318475

theorem sqrt_65_minus_1_bound (n : ℕ) (hn : 0 < n) :
  (n : ℝ) < Real.sqrt 65 - 1 ∧ Real.sqrt 65 - 1 < (n : ℝ) + 1 → n = 7 := by
  sorry

end sqrt_65_minus_1_bound_l3184_318475


namespace gain_percentage_proof_l3184_318402

/-- Proves that the gain percentage is 20% when selling 20 articles for $60,
    given that selling 29.99999625000047 articles for $60 would result in a 20% loss. -/
theorem gain_percentage_proof (articles_sold : ℝ) (total_price : ℝ) (loss_articles : ℝ) 
  (h1 : articles_sold = 20)
  (h2 : total_price = 60)
  (h3 : loss_articles = 29.99999625000047)
  (h4 : (0.8 * (loss_articles * (total_price / articles_sold))) = total_price) :
  (((total_price / articles_sold) - (total_price / loss_articles)) / (total_price / loss_articles)) * 100 = 20 :=
by sorry

end gain_percentage_proof_l3184_318402


namespace combination_equality_solutions_l3184_318451

theorem combination_equality_solutions (x : ℕ) : 
  (Nat.choose 25 (2*x) = Nat.choose 25 (x + 4)) ↔ (x = 4 ∨ x = 7) := by
  sorry

end combination_equality_solutions_l3184_318451


namespace complex_equation_sum_l3184_318424

theorem complex_equation_sum (a b : ℝ) : 
  (2 : ℂ) / (1 - Complex.I) = Complex.mk a b → a + b = 2 := by
  sorry

end complex_equation_sum_l3184_318424


namespace max_distance_circle_to_line_l3184_318474

theorem max_distance_circle_to_line : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let line := {p : ℝ × ℝ | p.1 = 2}
  ∀ p ∈ circle, ∀ q ∈ line, 
    ∃ r ∈ circle, Real.sqrt ((r.1 - q.1)^2 + (r.2 - q.2)^2) = 3 ∧
    ∀ s ∈ circle, Real.sqrt ((s.1 - q.1)^2 + (s.2 - q.2)^2) ≤ 3 :=
by
  sorry

end max_distance_circle_to_line_l3184_318474


namespace cincinnati_to_nyc_distance_l3184_318447

/-- The total distance between Cincinnati and New York City -/
def total_distance (day1 day2 day3 remaining : ℕ) : ℕ :=
  day1 + day2 + day3 + remaining

/-- The distance walked on the second day -/
def day2_distance (day1 : ℕ) : ℕ :=
  day1 / 2 - 6

theorem cincinnati_to_nyc_distance :
  total_distance 20 (day2_distance 20) 10 36 = 70 := by
  sorry

end cincinnati_to_nyc_distance_l3184_318447


namespace square_area_is_36_l3184_318473

/-- A square in the coordinate plane with specific y-coordinates -/
structure SquareInPlane where
  -- Define the y-coordinates of the vertices
  y1 : ℝ := 0
  y2 : ℝ := 3
  y3 : ℝ := 0
  y4 : ℝ := -3

/-- The area of the square -/
def squareArea (s : SquareInPlane) : ℝ := 36

/-- Theorem: The area of the square with given y-coordinates is 36 -/
theorem square_area_is_36 (s : SquareInPlane) : squareArea s = 36 := by
  sorry


end square_area_is_36_l3184_318473


namespace allie_toys_count_l3184_318482

/-- Proves that given a set of toys with specified values, the total number of toys is correct -/
theorem allie_toys_count (total_worth : ℕ) (special_toy_worth : ℕ) (regular_toy_worth : ℕ) :
  total_worth = 52 →
  special_toy_worth = 12 →
  regular_toy_worth = 5 →
  ∃ (n : ℕ), n * regular_toy_worth + special_toy_worth = total_worth ∧ n + 1 = 9 :=
by sorry

end allie_toys_count_l3184_318482


namespace at_least_one_not_less_than_two_l3184_318418

theorem at_least_one_not_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/a ≥ 2) := by
  sorry

end at_least_one_not_less_than_two_l3184_318418


namespace car_speed_calculation_l3184_318494

/-- Proves that a car's speed is 104 miles per hour given specific conditions -/
theorem car_speed_calculation (fuel_efficiency : ℝ) (fuel_consumed : ℝ) (time : ℝ)
  (h1 : fuel_efficiency = 64) -- km per liter
  (h2 : fuel_consumed = 3.9) -- gallons
  (h3 : time = 5.7) -- hours
  (h4 : (1 : ℝ) / 3.8 = 1 / 3.8) -- 1 gallon = 3.8 liters
  (h5 : (1 : ℝ) / 1.6 = 1 / 1.6) -- 1 mile = 1.6 kilometers
  : (fuel_efficiency * fuel_consumed * 3.8) / (time * 1.6) = 104 := by
  sorry

end car_speed_calculation_l3184_318494


namespace two_in_A_l3184_318435

def A : Set ℝ := {x | x > 1}

theorem two_in_A : 2 ∈ A := by
  sorry

end two_in_A_l3184_318435


namespace min_shared_side_length_l3184_318434

/-- Given three triangles ABC, DBC, and EBC sharing side BC, prove that the minimum possible
    integer length of BC is 8 cm. -/
theorem min_shared_side_length
  (AB : ℝ) (AC : ℝ) (DC : ℝ) (BD : ℝ) (EC : ℝ)
  (h_AB : AB = 7)
  (h_AC : AC = 15)
  (h_DC : DC = 9)
  (h_BD : BD = 12)
  (h_EC : EC = 11)
  : ∃ (BC : ℕ), BC ≥ 8 ∧ 
    (∀ (BC' : ℕ), BC' ≥ 8 → BC' > AC - AB) ∧
    (∀ (BC' : ℕ), BC' ≥ 8 → BC' > BD - DC) ∧
    (∀ (BC' : ℕ), BC' ≥ 8 → BC' > 0) ∧
    (∀ (BC'' : ℕ), BC'' < BC → 
      (BC'' ≤ AC - AB ∨ BC'' ≤ BD - DC ∨ BC'' ≤ 0)) :=
by sorry

end min_shared_side_length_l3184_318434


namespace partnership_profit_theorem_l3184_318408

/-- Represents the profit distribution in a partnership --/
structure Partnership where
  investment_ratio : ℝ  -- Ratio of A's investment to B's investment
  time_ratio : ℝ        -- Ratio of A's investment time to B's investment time
  b_profit : ℝ          -- B's profit

/-- Calculates the total profit of the partnership --/
def total_profit (p : Partnership) : ℝ :=
  let a_profit := p.b_profit * p.investment_ratio * p.time_ratio
  a_profit + p.b_profit

/-- Theorem stating that under the given conditions, the total profit is 28000 --/
theorem partnership_profit_theorem (p : Partnership) 
  (h1 : p.investment_ratio = 3)
  (h2 : p.time_ratio = 2)
  (h3 : p.b_profit = 4000) :
  total_profit p = 28000 := by
  sorry

#eval total_profit { investment_ratio := 3, time_ratio := 2, b_profit := 4000 }

end partnership_profit_theorem_l3184_318408


namespace uncool_parents_count_l3184_318436

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ)
  (h1 : total = 50)
  (h2 : cool_dads = 25)
  (h3 : cool_moms = 30)
  (h4 : both_cool = 15) :
  total - (cool_dads - both_cool + cool_moms - both_cool + both_cool) = 10 := by
  sorry

end uncool_parents_count_l3184_318436


namespace contrapositive_equality_l3184_318414

theorem contrapositive_equality (a b : ℝ) :
  (¬(a * b = 0) ↔ (a ≠ 0 ∧ b ≠ 0)) ↔
  ((a * b = 0) → (a = 0 ∨ b = 0)) :=
by sorry

end contrapositive_equality_l3184_318414


namespace first_volume_pages_l3184_318407

/-- Given a two-volume book with a total of 999 digits used for page numbers,
    where the first volume has 9 more pages than the second volume,
    prove that the number of pages in the first volume is 207. -/
theorem first_volume_pages (total_digits : ℕ) (page_difference : ℕ) 
  (h1 : total_digits = 999)
  (h2 : page_difference = 9) :
  ∃ (first_volume second_volume : ℕ),
    first_volume = second_volume + page_difference ∧
    first_volume = 207 :=
by sorry

end first_volume_pages_l3184_318407
