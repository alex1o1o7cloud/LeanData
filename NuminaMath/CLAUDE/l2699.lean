import Mathlib

namespace students_not_in_biology_l2699_269909

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880) 
  (h2 : biology_percentage = 275 / 1000) : 
  total_students - (total_students * biology_percentage).floor = 638 := by
  sorry

end students_not_in_biology_l2699_269909


namespace intersection_nonempty_iff_a_in_range_l2699_269927

def set_A (a : ℝ) : Set ℝ := {y | y > a^2 + 1 ∨ y < a}
def set_B : Set ℝ := {y | 2 ≤ y ∧ y ≤ 4}

theorem intersection_nonempty_iff_a_in_range (a : ℝ) :
  (set_A a ∩ set_B).Nonempty ↔ (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) ∨ a > 2 :=
by sorry

end intersection_nonempty_iff_a_in_range_l2699_269927


namespace sum_4_equivalence_l2699_269913

-- Define the type for dice outcomes
def DiceOutcome := Fin 6

-- Define the type for a pair of dice outcomes
def DicePair := DiceOutcome × DiceOutcome

-- Define the sum of a pair of dice
def diceSum (pair : DicePair) : Nat :=
  pair.1.val + pair.2.val + 2

-- Define the event ξ = 4
def sumIs4 (pair : DicePair) : Prop :=
  diceSum pair = 4

-- Define the event where one die shows 3 and the other shows 1
def oneThreeOneOne (pair : DicePair) : Prop :=
  (pair.1.val = 2 ∧ pair.2.val = 0) ∨ (pair.1.val = 0 ∧ pair.2.val = 2)

-- Define the event where both dice show 2
def bothTwo (pair : DicePair) : Prop :=
  pair.1.val = 1 ∧ pair.2.val = 1

-- Theorem: ξ = 4 is equivalent to (one die shows 3 and the other shows 1) or (both dice show 2)
theorem sum_4_equivalence (pair : DicePair) :
  sumIs4 pair ↔ oneThreeOneOne pair ∨ bothTwo pair :=
by sorry

end sum_4_equivalence_l2699_269913


namespace circle_radius_l2699_269902

theorem circle_radius (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y = 1 ↔ (x + 2)^2 + (y - 1)^2 = r^2) → 
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y = 1 ↔ (x + 2)^2 + (y - 1)^2 = r^2 ∧ r = Real.sqrt 6) :=
by sorry

end circle_radius_l2699_269902


namespace sqrt_product_simplification_l2699_269904

theorem sqrt_product_simplification :
  Real.sqrt (12 + 1/9) * Real.sqrt 3 = Real.sqrt 327 / 3 := by
  sorry

end sqrt_product_simplification_l2699_269904


namespace largest_solution_of_equation_l2699_269906

theorem largest_solution_of_equation : 
  ∃ (b : ℝ), (3 * b + 4) * (b - 2) = 9 * b ∧ 
  ∀ (x : ℝ), (3 * x + 4) * (x - 2) = 9 * x → x ≤ b ∧ 
  b = 4 := by
sorry

end largest_solution_of_equation_l2699_269906


namespace negation_equivalence_l2699_269971

theorem negation_equivalence :
  (¬ ∀ x : ℝ, ∃ n : ℕ+, (n : ℝ) ≥ x^2) ↔ (∃ x : ℝ, ∀ n : ℕ+, (n : ℝ) < x^2) := by
  sorry

end negation_equivalence_l2699_269971


namespace even_function_derivative_odd_function_derivative_l2699_269914

variable (f : ℝ → ℝ)

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem for even functions
theorem even_function_derivative (h : IsEven f) (h' : Differentiable ℝ f) :
  IsOdd (deriv f) := by sorry

-- Theorem for odd functions
theorem odd_function_derivative (h : IsOdd f) (h' : Differentiable ℝ f) :
  IsEven (deriv f) := by sorry

end even_function_derivative_odd_function_derivative_l2699_269914


namespace complement_of_intersection_l2699_269989

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {1, 2, 3}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 4, 5} := by sorry

end complement_of_intersection_l2699_269989


namespace red_square_area_equals_cross_area_l2699_269955

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  /-- Side length of the flag -/
  side : ℝ
  /-- Ratio of the cross arm width to the flag side length -/
  arm_ratio : ℝ
  /-- The cross (arms + center) occupies 49% of the flag area -/
  cross_area_constraint : 4 * arm_ratio * (1 - arm_ratio) = 0.49

theorem red_square_area_equals_cross_area (flag : CrossFlag) :
  4 * flag.arm_ratio^2 = 4 * flag.arm_ratio * (1 - flag.arm_ratio) := by
  sorry

#check red_square_area_equals_cross_area

end red_square_area_equals_cross_area_l2699_269955


namespace bottle_arrangement_l2699_269975

theorem bottle_arrangement (x : ℕ) : 
  (x^2 + 36 = (x + 1)^2 + 3) → (x^2 + 36 = 292) :=
by
  sorry

end bottle_arrangement_l2699_269975


namespace factorization_x4_minus_64_l2699_269916

theorem factorization_x4_minus_64 (x : ℝ) : x^4 - 64 = (x^2 - 8) * (x^2 + 8) := by
  sorry

end factorization_x4_minus_64_l2699_269916


namespace tunnel_length_l2699_269992

/-- The length of a tunnel given train parameters -/
theorem tunnel_length (train_length : ℝ) (train_speed : ℝ) (time : ℝ) :
  train_length = 90 →
  train_speed = 160 →
  time = 3 →
  train_speed * time - train_length = 390 := by
  sorry

end tunnel_length_l2699_269992


namespace cos_pi_third_minus_alpha_l2699_269999

theorem cos_pi_third_minus_alpha (α : Real) 
  (h : Real.sin (π / 6 + α) = 1 / 3) : 
  Real.cos (π / 3 - α) = 1 / 3 := by
  sorry

end cos_pi_third_minus_alpha_l2699_269999


namespace line_equations_specific_line_equations_l2699_269934

/-- Definition of a line passing through two points -/
def Line (A B : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | (y - A.2) * (B.1 - A.1) = (x - A.1) * (B.2 - A.2)}

theorem line_equations (A B : ℝ × ℝ) (h : A ≠ B) :
  let l := Line A B
  -- Two-point form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ (y - B.2) / (A.2 - B.2) = (x - B.1) / (A.1 - B.1) ∧
  -- Point-slope form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y - B.2 = ((A.2 - B.2) / (A.1 - B.1)) * (x - B.1) ∧
  -- Slope-intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y = ((A.2 - B.2) / (A.1 - B.1)) * x + (B.2 - ((A.2 - B.2) / (A.1 - B.1)) * B.1) ∧
  -- Intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ x / (-B.1 + (A.1 * B.2 - A.2 * B.1) / (A.2 - B.2)) + 
                             y / ((A.1 * B.2 - A.2 * B.1) / (A.1 - B.1)) = 1 :=
by
  sorry

-- Specific instance for points A(-2, 3) and B(4, -1)
theorem specific_line_equations :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (4, -1)
  let l := Line A B
  -- Two-point form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ (y + 1) / 4 = (x - 4) / (-6) ∧
  -- Point-slope form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y + 1 = -2/3 * (x - 4) ∧
  -- Slope-intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y = -2/3 * x + 5/3 ∧
  -- Intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ x / (5/2) + y / (5/3) = 1 :=
by
  sorry

end line_equations_specific_line_equations_l2699_269934


namespace election_winner_percentage_l2699_269973

theorem election_winner_percentage 
  (total_votes : ℕ) 
  (majority : ℕ) 
  (winning_percentage : ℚ) :
  total_votes = 6500 →
  majority = 1300 →
  winning_percentage * total_votes = (total_votes + majority) / 2 →
  winning_percentage = 60 / 100 :=
by sorry

end election_winner_percentage_l2699_269973


namespace valid_lineups_count_l2699_269953

-- Define the total number of players
def total_players : ℕ := 15

-- Define the number of players in a starting lineup
def lineup_size : ℕ := 6

-- Define the number of players who can't play together
def restricted_players : ℕ := 3

-- Define the function to calculate the number of valid lineups
def valid_lineups : ℕ := sorry

-- Theorem statement
theorem valid_lineups_count :
  valid_lineups = 3300 :=
sorry

end valid_lineups_count_l2699_269953


namespace system_solution_existence_l2699_269911

theorem system_solution_existence (m : ℝ) : 
  (m ≠ 1) ↔ (∃ (x y : ℝ), y = m * x + 5 ∧ y = (3 * m - 2) * x + 7) :=
by sorry

end system_solution_existence_l2699_269911


namespace penny_sock_cost_l2699_269958

/-- Given Penny's shopping scenario, prove the cost of each pair of socks. -/
theorem penny_sock_cost (initial_amount : ℚ) (num_sock_pairs : ℕ) (hat_cost remaining_amount : ℚ) :
  initial_amount = 20 →
  num_sock_pairs = 4 →
  hat_cost = 7 →
  remaining_amount = 5 →
  ∃ (sock_cost : ℚ), 
    initial_amount - hat_cost - (num_sock_pairs : ℚ) * sock_cost = remaining_amount ∧
    sock_cost = 2 := by
  sorry

end penny_sock_cost_l2699_269958


namespace ellipse_parabola_intersection_l2699_269972

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line
def line (x y : ℝ) : Prop := y = x + 1

theorem ellipse_parabola_intersection :
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    ellipse 0 1 ∧  -- Vertex of ellipse at (0, 1)
    parabola 0 1 ∧  -- Focus of parabola at (0, 1)
    line x₁ y₁ ∧
    line x₂ y₂ ∧
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    x₁ * x₂ = -4 := by
  sorry

end ellipse_parabola_intersection_l2699_269972


namespace equal_roots_condition_l2699_269926

/-- The quadratic equation x^2 - nx + 9 = 0 has two equal real roots if and only if n = 6 or n = -6 -/
theorem equal_roots_condition (n : ℝ) : 
  (∃ x : ℝ, x^2 - n*x + 9 = 0 ∧ (∀ y : ℝ, y^2 - n*y + 9 = 0 → y = x)) ↔ 
  (n = 6 ∨ n = -6) := by
sorry

end equal_roots_condition_l2699_269926


namespace sequence_inequality_l2699_269976

theorem sequence_inequality (a : ℕ → ℕ) (n N : ℕ) 
  (h1 : ∀ m k, a (m + k) ≤ a m + a k) 
  (h2 : N ≥ n) :
  a n + a N ≤ n * a 1 + (N / n) * a n :=
sorry

end sequence_inequality_l2699_269976


namespace triangle_equality_l2699_269923

theorem triangle_equality (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = a + c ∨ c = a + b := by
  sorry

end triangle_equality_l2699_269923


namespace jeromes_contacts_l2699_269936

theorem jeromes_contacts (classmates : ℕ) (total_contacts : ℕ) :
  classmates = 20 →
  total_contacts = 33 →
  3 = total_contacts - (classmates + classmates / 2) :=
by sorry

end jeromes_contacts_l2699_269936


namespace factorization_example_l2699_269997

theorem factorization_example (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

#check factorization_example

end factorization_example_l2699_269997


namespace smallest_positive_period_of_even_odd_functions_l2699_269922

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function g is odd if g(x) = -g(-x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

/-- The period of a function f is p if f(x + p) = f(x) for all x -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

/-- The smallest positive period of a function -/
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ IsPeriod f p ∧ ∀ q, 0 < q ∧ q < p → ¬IsPeriod f q

theorem smallest_positive_period_of_even_odd_functions
  (f g : ℝ → ℝ) (c : ℝ)
  (hf : IsEven f)
  (hg : IsOdd g)
  (h : ∀ x, f x = -g (x + c))
  (hc : c > 0) :
  SmallestPositivePeriod f (4 * c) := by
  sorry

end smallest_positive_period_of_even_odd_functions_l2699_269922


namespace triangle_inequality_l2699_269948

theorem triangle_inequality (a b c : ℝ) 
  (triangle_cond : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (abc_cond : a * b * c = 1) : 
  (Real.sqrt (b + c - a)) / a + (Real.sqrt (c + a - b)) / b + (Real.sqrt (a + b - c)) / c ≥ a + b + c :=
sorry

end triangle_inequality_l2699_269948


namespace decimal_51_to_binary_binary_to_decimal_51_l2699_269901

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- Converts a list of bits to a natural number -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_51_to_binary :
  to_binary 51 = [true, true, false, false, true, true] :=
by sorry

theorem binary_to_decimal_51 :
  from_binary [true, true, false, false, true, true] = 51 :=
by sorry

end decimal_51_to_binary_binary_to_decimal_51_l2699_269901


namespace triangle_coordinate_difference_l2699_269980

/-- Triangle ABC with vertices A(0,10), B(4,0), C(10,0), and a vertical line
    intersecting AC at R and BC at S. If the area of triangle RSC is 20,
    then the positive difference between the x and y coordinates of R is 4√10 - 10. -/
theorem triangle_coordinate_difference (R : ℝ × ℝ) :
  let A : ℝ × ℝ := (0, 10)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (10, 0)
  let S : ℝ × ℝ := (R.1, 0)  -- S has same x-coordinate as R and y-coordinate 0
  -- R is on line AC
  (10 - R.1) / (0 - R.2) = 1 →
  -- RS is vertical (same x-coordinate)
  R.1 = S.1 →
  -- Area of triangle RSC is 20
  abs ((R.1 - 10) * R.2) / 2 = 20 →
  -- The positive difference between x and y coordinates of R
  abs (R.2 - R.1) = 4 * Real.sqrt 10 - 10 :=
by sorry

end triangle_coordinate_difference_l2699_269980


namespace max_correct_answers_l2699_269932

theorem max_correct_answers (total_questions : Nat) (correct_points : Int) (incorrect_points : Int) (total_score : Int) :
  total_questions = 30 →
  correct_points = 4 →
  incorrect_points = -3 →
  total_score = 72 →
  ∃ (correct incorrect unanswered : Nat),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_points + incorrect * incorrect_points = total_score ∧
    correct ≤ 21 ∧
    ∀ (c i u : Nat),
      c + i + u = total_questions →
      c * correct_points + i * incorrect_points = total_score →
      c ≤ 21 :=
by sorry

end max_correct_answers_l2699_269932


namespace tens_digit_of_special_two_digit_number_l2699_269915

/-- The product of the digits of a two-digit number -/
def P (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens * ones

/-- The sum of the digits of a two-digit number -/
def S (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens + ones

/-- A two-digit number N satisfying N = P(N)^2 + S(N) has 1 as its tens digit -/
theorem tens_digit_of_special_two_digit_number :
  ∃ N : ℕ, 
    10 ≤ N ∧ N < 100 ∧ 
    N = (P N)^2 + S N ∧
    N / 10 = 1 := by
  sorry

end tens_digit_of_special_two_digit_number_l2699_269915


namespace sum_equals_thirteen_thousand_two_hundred_l2699_269933

theorem sum_equals_thirteen_thousand_two_hundred : 9773 + 3427 = 13200 := by
  sorry

end sum_equals_thirteen_thousand_two_hundred_l2699_269933


namespace class_fund_total_l2699_269970

theorem class_fund_total (ten_bills : ℕ) (twenty_bills : ℕ) : 
  ten_bills = 2 * twenty_bills →
  twenty_bills = 3 →
  ten_bills * 10 + twenty_bills * 20 = 120 :=
by
  sorry

end class_fund_total_l2699_269970


namespace workshop_workers_count_l2699_269990

/-- Proves that the total number of workers in a workshop is 14 given specific salary conditions -/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  -- Average salary of all workers is 9000
  W * 9000 = 7 * 12000 + N * 6000 →
  -- Total workers is sum of technicians and non-technicians
  W = 7 + N →
  -- Conclusion: Total number of workers is 14
  W = 14 := by
  sorry

end workshop_workers_count_l2699_269990


namespace hyperbola_asymptote_ratio_l2699_269905

/-- Given a hyperbola x²/a² - y²/b² = 1 with a > b, if the angle between its asymptotes is 45°, then a/b = √2 + 1 -/
theorem hyperbola_asymptote_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.arctan (2 * (b/a) / (1 - (b/a)^2)) = π/4) →
  a/b = Real.sqrt 2 + 1 := by
sorry

end hyperbola_asymptote_ratio_l2699_269905


namespace line_not_in_second_quadrant_l2699_269943

/-- A line with equation y = 3x - 2 does not pass through the second quadrant. -/
theorem line_not_in_second_quadrant :
  ∀ x y : ℝ, y = 3 * x - 2 → ¬(x > 0 ∧ y > 0) :=
by
  sorry

end line_not_in_second_quadrant_l2699_269943


namespace gnollish_sentences_l2699_269949

/-- The number of words in the Gnollish language -/
def num_words : ℕ := 4

/-- The length of a sentence in the Gnollish language -/
def sentence_length : ℕ := 3

/-- The number of invalid sentence patterns due to the restriction -/
def num_invalid_patterns : ℕ := 2

/-- The number of choices for the unrestricted word in an invalid pattern -/
def choices_for_unrestricted : ℕ := num_words

theorem gnollish_sentences :
  (num_words ^ sentence_length) - (num_invalid_patterns * choices_for_unrestricted) = 56 := by
  sorry

end gnollish_sentences_l2699_269949


namespace complex_fraction_equality_l2699_269954

theorem complex_fraction_equality : (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end complex_fraction_equality_l2699_269954


namespace part_one_part_two_l2699_269964

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (3 - 2*a) * x - 6

-- Part 1
theorem part_one :
  f 1 x > 0 ↔ x < -3 ∨ x > 2 :=
sorry

-- Part 2
theorem part_two (a : ℝ) (h : a < 0) :
  (a < -3/2 → (f a x < 0 ↔ x < -3/a ∨ x > 2)) ∧
  (a = -3/2 → (f a x < 0 ↔ x ≠ 2)) ∧
  (-3/2 < a → (f a x < 0 ↔ x < 2 ∨ x > -3/a)) :=
sorry

end part_one_part_two_l2699_269964


namespace solution_set_f_greater_than_4_range_of_m_l2699_269962

-- Define the function f
def f (x : ℝ) := |x - 2| + 2 * |x - 1|

-- Theorem for the solution set of f(x) > 4
theorem solution_set_f_greater_than_4 :
  {x : ℝ | f x > 4} = {x : ℝ | x < 0 ∨ x > 0} :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, f x > 2 * m^2 - 7 * m + 4} = {m : ℝ | 1/2 < m ∧ m < 3} :=
sorry

end solution_set_f_greater_than_4_range_of_m_l2699_269962


namespace min_unboxed_balls_tennis_balls_storage_l2699_269998

theorem min_unboxed_balls (total_balls : ℕ) (big_box_size small_box_size : ℕ) : ℕ :=
  let min_unboxed := total_balls % big_box_size
  let remaining_after_big := total_balls % big_box_size
  let min_unboxed_small := remaining_after_big % small_box_size
  min min_unboxed min_unboxed_small

theorem tennis_balls_storage :
  min_unboxed_balls 104 25 20 = 4 := by
  sorry

end min_unboxed_balls_tennis_balls_storage_l2699_269998


namespace polygonal_chains_10_9_l2699_269910

/-- The number of points on the circle -/
def n : ℕ := 10

/-- The number of sides in each polygonal chain -/
def sides : ℕ := 9

/-- A function that calculates the number of non-closed, non-self-intersecting 
    polygonal chains with 'sides' sides that can be formed from 'n' points on a circle -/
def polygonal_chains (n : ℕ) (sides : ℕ) : ℕ :=
  if sides ≤ n ∧ sides > 2 then
    (n * 2^(sides - 1)) / 2
  else
    0

/-- Theorem stating that the number of non-closed, non-self-intersecting 9-sided 
    polygonal chains that can be formed with 10 points on a circle as vertices is 1280 -/
theorem polygonal_chains_10_9 : polygonal_chains n sides = 1280 := by
  sorry

end polygonal_chains_10_9_l2699_269910


namespace ellipse_and_chord_problem_l2699_269917

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.2^2 / a^2) + (p.1^2 / b^2) = 1}

-- Define the circle
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = r^2}

theorem ellipse_and_chord_problem 
  (e : ℝ) (f : ℝ × ℝ) 
  (h_e : e = 2 * Real.sqrt 2 / 3)
  (h_f : f = (0, 2 * Real.sqrt 2))
  (h_foci : ∃ (f' : ℝ × ℝ), f'.1 = 0 ∧ f'.2 = -f.2) :
  -- Standard equation of the ellipse
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Ellipse a b = Ellipse 3 1 ∧
  -- Maximum length of CP
  ∃ (C : ℝ × ℝ) (P : ℝ × ℝ), 
    C ∈ Ellipse 3 1 ∧ 
    P = (-1, 0) ∧
    ∀ (C' : ℝ × ℝ), C' ∈ Ellipse 3 1 → 
      Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2) ≥ 
      Real.sqrt ((C'.1 - P.1)^2 + (C'.2 - P.2)^2) ∧
    Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2) = 9 * Real.sqrt 2 / 4 ∧
  -- Length of AB when CP is maximum
  ∃ (A B : ℝ × ℝ),
    A ∈ Circle 2 ∧
    B ∈ Circle 2 ∧
    (A.2 - B.2) * (C.1 - P.1) + (B.1 - A.1) * (C.2 - P.2) = 0 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 62 / 2 :=
sorry

end ellipse_and_chord_problem_l2699_269917


namespace find_y_value_l2699_269978

theorem find_y_value (x y z : ℝ) 
  (h1 : x^2 * y = z) 
  (h2 : x / y = 36)
  (h3 : Real.sqrt (x * y) = z)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) :
  y = 1 / 14.7 := by
sorry

end find_y_value_l2699_269978


namespace peanut_butter_servings_l2699_269908

-- Define the initial amount of peanut butter in tablespoons
def initial_amount : ℚ := 35 + 2/3

-- Define the amount used for the recipe in tablespoons
def amount_used : ℚ := 5 + 1/3

-- Define the serving size in tablespoons
def serving_size : ℚ := 3

-- Theorem to prove
theorem peanut_butter_servings :
  ⌊(initial_amount - amount_used) / serving_size⌋ = 10 := by
  sorry

end peanut_butter_servings_l2699_269908


namespace work_rate_increase_l2699_269994

theorem work_rate_increase (total_time hours_worked : ℝ)
  (original_items additional_items : ℕ) :
  total_time = 10 ∧ 
  hours_worked = 6 ∧ 
  original_items = 1250 ∧ 
  additional_items = 150 →
  let original_rate := original_items / total_time
  let items_processed := original_rate * hours_worked
  let remaining_items := original_items - items_processed + additional_items
  let remaining_time := total_time - hours_worked
  let new_rate := remaining_items / remaining_time
  (new_rate - original_rate) / original_rate * 100 = 30 := by
sorry

end work_rate_increase_l2699_269994


namespace infinite_geometric_series_first_term_l2699_269981

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -1/3) (h2 : S = 9) :
  let a := S * (1 - r)
  a = 12 := by sorry

end infinite_geometric_series_first_term_l2699_269981


namespace aaron_initial_cards_l2699_269993

theorem aaron_initial_cards (found : ℕ) (final : ℕ) (h1 : found = 62) (h2 : final = 67) :
  final - found = 5 := by
  sorry

end aaron_initial_cards_l2699_269993


namespace intersection_A_complement_B_equals_zero_one_l2699_269930

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Theorem statement
theorem intersection_A_complement_B_equals_zero_one :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x ≤ 1} :=
sorry

end intersection_A_complement_B_equals_zero_one_l2699_269930


namespace nine_twin_functions_l2699_269946

-- Define the function f(x) = 2x^2 + 1
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the range set
def range_set : Set ℝ := {5, 19}

-- Define the property for a valid domain
def is_valid_domain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x ∈ range_set) ∧ 
  (∀ y ∈ range_set, ∃ x ∈ D, f x = y)

-- State the theorem
theorem nine_twin_functions :
  ∃! (domains : Finset (Set ℝ)), 
    Finset.card domains = 9 ∧ 
    (∀ D ∈ domains, is_valid_domain D) ∧
    (∀ D : Set ℝ, is_valid_domain D → D ∈ domains) :=
sorry

end nine_twin_functions_l2699_269946


namespace right_triangle_inequality_l2699_269907

theorem right_triangle_inequality (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b - a)^2 * (c^2 + 4*a*b)^2 ≤ 2*c^6 := by
  sorry

end right_triangle_inequality_l2699_269907


namespace sufficient_condition_quadratic_l2699_269918

theorem sufficient_condition_quadratic (a : ℝ) :
  a ≥ 0 → ∃ x : ℝ, a * x^2 + x + 1 ≥ 0 := by
  sorry

end sufficient_condition_quadratic_l2699_269918


namespace train_passing_jogger_time_l2699_269969

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9)
  (h2 : train_speed = 45)
  (h3 : train_length = 120)
  (h4 : initial_distance = 240)
  : (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 36 :=
by
  sorry

#check train_passing_jogger_time

end train_passing_jogger_time_l2699_269969


namespace parabola_properties_l2699_269920

/-- A function representing a parabola -/
def f (x : ℝ) : ℝ := -x^2 + 1

/-- The parabola opens downwards -/
def opens_downwards (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a < 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The parabola intersects the y-axis at (0,1) -/
def intersects_y_axis_at_0_1 (f : ℝ → ℝ) : Prop :=
  f 0 = 1

theorem parabola_properties :
  opens_downwards f ∧ intersects_y_axis_at_0_1 f :=
sorry

end parabola_properties_l2699_269920


namespace smallest_x_congruence_l2699_269937

theorem smallest_x_congruence :
  ∃ (x : ℕ), x > 0 ∧ (725 * x) % 35 = (1165 * x) % 35 ∧
  ∀ (y : ℕ), y > 0 → (725 * y) % 35 = (1165 * y) % 35 → x ≤ y :=
by sorry

end smallest_x_congruence_l2699_269937


namespace solutions_eq_divisors_l2699_269919

/-- The number of integer solutions to the equation xy + ax + by = c -/
def num_solutions (a b c : ℤ) : ℕ :=
  2 * (Nat.divisors (a * b + c).natAbs).card

/-- The number of divisors (positive and negative) of an integer n -/
def num_divisors (n : ℤ) : ℕ :=
  2 * (Nat.divisors n.natAbs).card

theorem solutions_eq_divisors (a b c : ℤ) :
  num_solutions a b c = num_divisors (a * b + c) :=
sorry

end solutions_eq_divisors_l2699_269919


namespace range_of_a_l2699_269957

/-- Given a system of equations and an inequality, prove the range of values for a. -/
theorem range_of_a (a x y : ℝ) 
  (eq1 : x + y = 3 * a + 4)
  (eq2 : x - y = 7 * a - 4)
  (ineq : 3 * x - 2 * y < 11) :
  a < 1 := by
  sorry

end range_of_a_l2699_269957


namespace complex_division_result_l2699_269979

theorem complex_division_result : (1 + 2*I) / (1 - 2*I) = -3/5 + 4/5*I := by
  sorry

end complex_division_result_l2699_269979


namespace repeating_decimal_problem_l2699_269947

theorem repeating_decimal_problem (a b : ℕ) (h1 : a < 10) (h2 : b < 10) : 
  66 * (1 + (10 * a + b) / 99) - 66 * (1 + (10 * a + b) / 100) = 1/2 → 
  10 * a + b = 75 := by sorry

end repeating_decimal_problem_l2699_269947


namespace jake_not_drop_coffee_l2699_269985

/-- The probability of Jake tripping over his dog in the morning -/
def prob_trip : ℝ := 0.4

/-- The probability of Jake dropping his coffee when he trips -/
def prob_drop_given_trip : ℝ := 0.25

/-- The probability of Jake not dropping his coffee in the morning -/
def prob_not_drop : ℝ := 1 - prob_trip * prob_drop_given_trip

theorem jake_not_drop_coffee : prob_not_drop = 0.9 := by
  sorry

end jake_not_drop_coffee_l2699_269985


namespace harmonic_mean_inequality_l2699_269983

theorem harmonic_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 := by
  sorry

end harmonic_mean_inequality_l2699_269983


namespace pinterest_group_pins_l2699_269966

/-- Calculates the number of pins in a Pinterest group after one month -/
def pinsAfterOneMonth (
  groupSize : ℕ
  ) (averageDailyContribution : ℕ
  ) (weeklyDeletionRate : ℕ
  ) (initialPins : ℕ
  ) : ℕ :=
  let daysInMonth : ℕ := 30
  let weeksInMonth : ℕ := 4
  let monthlyContribution := groupSize * averageDailyContribution * daysInMonth
  let monthlyDeletion := groupSize * weeklyDeletionRate * weeksInMonth
  initialPins + monthlyContribution - monthlyDeletion

theorem pinterest_group_pins :
  pinsAfterOneMonth 20 10 5 1000 = 6600 := by
  sorry

end pinterest_group_pins_l2699_269966


namespace smallest_integer_with_given_remainders_l2699_269977

theorem smallest_integer_with_given_remainders : ∃ x : ℕ+, 
  (x : ℕ) % 6 = 5 ∧ 
  (x : ℕ) % 7 = 6 ∧ 
  (x : ℕ) % 8 = 7 ∧ 
  ∀ y : ℕ+, 
    (y : ℕ) % 6 = 5 → 
    (y : ℕ) % 7 = 6 → 
    (y : ℕ) % 8 = 7 → 
    x ≤ y :=
by
  -- Proof goes here
  sorry

end smallest_integer_with_given_remainders_l2699_269977


namespace joe_initial_cars_l2699_269942

/-- The number of cars Joe will have after getting more -/
def total_cars : ℕ := 62

/-- The number of additional cars Joe will get -/
def additional_cars : ℕ := 12

/-- Joe's initial number of cars -/
def initial_cars : ℕ := total_cars - additional_cars

theorem joe_initial_cars : initial_cars = 50 := by sorry

end joe_initial_cars_l2699_269942


namespace ellipse_parallelogram_condition_l2699_269929

/-- The condition for an ellipse to have a parallelogram with a vertex on the ellipse, 
    tangent to the unit circle externally, and inscribed in the ellipse for any point on the ellipse -/
theorem ellipse_parallelogram_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 → 
    ∃ (p q r s : ℝ × ℝ), 
      (p.1^2 + p.2^2 = 1) ∧ 
      (q.1^2 + q.2^2 = 1) ∧ 
      (r.1^2 + r.2^2 = 1) ∧ 
      (s.1^2 + s.2^2 = 1) ∧
      (x^2/a^2 + y^2/b^2 = 1) ∧
      (p.1 - x = s.1 - r.1) ∧ 
      (p.2 - y = s.2 - r.2) ∧
      (q.1 - x = r.1 - s.1) ∧ 
      (q.2 - y = r.2 - s.2)) ↔ 
  (1/a^2 + 1/b^2 = 1) := by
sorry

end ellipse_parallelogram_condition_l2699_269929


namespace rectangle_perimeter_l2699_269939

theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b → -- non-square condition
  a * b = 3 * (2 * a + 2 * b) → -- area equals 3 times perimeter
  2 * a + 2 * b = 36 ∨ 2 * a + 2 * b = 28 := by
  sorry

end rectangle_perimeter_l2699_269939


namespace unique_solution_equation_l2699_269903

theorem unique_solution_equation (x : ℝ) : 
  (8^x * (3*x + 1) = 4) ↔ (x = 1/3) := by sorry

end unique_solution_equation_l2699_269903


namespace complex_value_at_angle_l2699_269935

/-- The value of 1-2i at an angle of 267.5° is equal to -√2/2 -/
theorem complex_value_at_angle : 
  let z : ℂ := 1 - 2*I
  let angle : Real := 267.5 * (π / 180)  -- Convert to radians
  Complex.abs z * Complex.exp (I * angle) = -Real.sqrt 2 / 2 := by
  sorry

end complex_value_at_angle_l2699_269935


namespace boy_girl_ratio_l2699_269940

/-- Represents the number of students in the class -/
def total_students : ℕ := 25

/-- Represents the difference between the number of boys and girls -/
def boy_girl_difference : ℕ := 9

/-- Theorem stating that the ratio of boys to girls is 17:8 -/
theorem boy_girl_ratio :
  ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    boys = girls + boy_girl_difference ∧
    boys = 17 ∧
    girls = 8 :=
by sorry

end boy_girl_ratio_l2699_269940


namespace root_problem_l2699_269931

-- Define the polynomials
def p (c d : ℝ) (x : ℝ) : ℝ := (x + c) * (x + d) * (x + 15)
def q (c d : ℝ) (x : ℝ) : ℝ := (x + 3 * c) * (x + 5) * (x + 9)

-- State the theorem
theorem root_problem (c d : ℝ) :
  (∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
    p c d r1 = 0 ∧ p c d r2 = 0 ∧ p c d r3 = 0) ∧
  (∃! (r : ℝ), q c d r = 0) ∧
  c ≠ d ∧ c ≠ 4 ∧ c ≠ 15 ∧ d ≠ 4 ∧ d ≠ 15 ∧ d ≠ 5 →
  100 * c + d = 157 := by
sorry

end root_problem_l2699_269931


namespace willowton_vampires_l2699_269951

/-- The number of vampires after a given number of nights -/
def vampires_after_nights (initial_population : ℕ) (initial_vampires : ℕ) (turned_per_night : ℕ) (nights : ℕ) : ℕ :=
  initial_vampires + nights * (initial_vampires * turned_per_night)

/-- Theorem stating the number of vampires after two nights in Willowton -/
theorem willowton_vampires :
  vampires_after_nights 300 2 5 2 = 72 := by
  sorry

#eval vampires_after_nights 300 2 5 2

end willowton_vampires_l2699_269951


namespace square_inequality_l2699_269974

theorem square_inequality (a b c A B C : ℝ) 
  (h1 : b^2 < a*c) 
  (h2 : a*C - 2*b*B + c*A = 0) : 
  B^2 ≥ A*C := by
  sorry

end square_inequality_l2699_269974


namespace frequency_distribution_correct_for_proportions_l2699_269963

/-- Represents the score ranges for the math test. -/
inductive ScoreRange
  | AboveOrEqual120
  | Between90And120
  | Between75And90
  | Between60And75
  | Below60

/-- Represents statistical methods that can be applied to analyze test scores. -/
inductive StatisticalMethod
  | SampleAndEstimate
  | CalculateAverage
  | FrequencyDistribution
  | CalculateVariance

/-- The total number of students who took the test. -/
def totalStudents : ℕ := 800

/-- Determines if a given statistical method is correct for finding proportions of students in different score ranges. -/
def isCorrectMethodForProportions (method : StatisticalMethod) : Prop :=
  method = StatisticalMethod.FrequencyDistribution

/-- Theorem stating that frequency distribution is the correct method for finding proportions of students in different score ranges. -/
theorem frequency_distribution_correct_for_proportions :
  isCorrectMethodForProportions StatisticalMethod.FrequencyDistribution :=
sorry

end frequency_distribution_correct_for_proportions_l2699_269963


namespace decreasing_function_inequality_l2699_269991

/-- A function f is decreasing on an open interval (a,b) if for all x, y in (a,b), x < y implies f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f x > f y

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h1 : DecreasingOn f (-1) 1)
  (h2 : f (1 - a) < f (3 * a - 1)) :
  0 < a ∧ a < 1/2 := by
  sorry

end decreasing_function_inequality_l2699_269991


namespace sophia_transactions_l2699_269950

theorem sophia_transactions (mabel anthony cal jade sophia : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 19 →
  sophia = jade + jade / 2 →
  sophia = 128 :=
by sorry

end sophia_transactions_l2699_269950


namespace band_repertoire_average_l2699_269938

theorem band_repertoire_average (total_songs : ℕ) (first_set : ℕ) (second_set : ℕ) (encore : ℕ) : 
  total_songs = 30 →
  first_set = 5 →
  second_set = 7 →
  encore = 2 →
  (total_songs - (first_set + second_set + encore)) / 2 = 8 :=
by sorry

end band_repertoire_average_l2699_269938


namespace reunion_handshakes_l2699_269984

theorem reunion_handshakes (n : ℕ) : n > 0 → (n * (n - 1)) / 2 = 45 → n = 10 := by
  sorry

end reunion_handshakes_l2699_269984


namespace euler_formula_imaginary_part_l2699_269924

open Complex

theorem euler_formula_imaginary_part :
  let z : ℂ := Complex.exp (I * Real.pi / 4)
  let w : ℂ := z / (1 - I)
  Complex.im w = Real.sqrt 2 / 2 := by
  sorry

end euler_formula_imaginary_part_l2699_269924


namespace limit_special_function_l2699_269912

/-- The limit of (7^(3x) - 3^(2x)) / (tan(x) + x^3) as x approaches 0 is ln(343/9) -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ →
    |((7^(3*x) - 3^(2*x)) / (Real.tan x + x^3)) - Real.log (343/9)| < ε := by
  sorry

end limit_special_function_l2699_269912


namespace sin_axis_of_symmetry_l2699_269982

/-- Proves that x = π/12 is one of the axes of symmetry for the function y = sin(2x + π/3) -/
theorem sin_axis_of_symmetry :
  ∃ (k : ℤ), 2 * (π/12 : ℝ) + π/3 = π/2 + k*π := by sorry

end sin_axis_of_symmetry_l2699_269982


namespace bob_rock_skips_bob_rock_skips_solution_l2699_269925

theorem bob_rock_skips (jim_skips : ℕ) (rocks_each : ℕ) (total_skips : ℕ) : ℕ :=
  let bob_skips := (total_skips - jim_skips * rocks_each) / rocks_each
  bob_skips

#check @bob_rock_skips

theorem bob_rock_skips_solution :
  bob_rock_skips 15 10 270 = 12 := by
  sorry

end bob_rock_skips_bob_rock_skips_solution_l2699_269925


namespace edwards_initial_money_l2699_269952

theorem edwards_initial_money (spent_first spent_second remaining : ℕ) : 
  spent_first = 9 → 
  spent_second = 8 → 
  remaining = 17 → 
  spent_first + spent_second + remaining = 34 :=
by sorry

end edwards_initial_money_l2699_269952


namespace quadratic_equation_solution_l2699_269921

theorem quadratic_equation_solution (y : ℝ) : 
  (((8 * y^2 + 50 * y + 5) / (3 * y + 21)) = 4 * y + 3) ↔ 
  (y = (-43 + Real.sqrt 921) / 8 ∨ y = (-43 - Real.sqrt 921) / 8) := by
sorry

end quadratic_equation_solution_l2699_269921


namespace commute_time_difference_l2699_269967

theorem commute_time_difference (distance : Real) (walk_speed : Real) (train_speed : Real) (time_difference : Real) :
  distance = 1.5 ∧ 
  walk_speed = 3 ∧ 
  train_speed = 20 ∧ 
  time_difference = 25 →
  ∃ x : Real, 
    (distance / walk_speed) * 60 = (distance / train_speed) * 60 + x + time_difference ∧ 
    x = 0.5 := by
  sorry


end commute_time_difference_l2699_269967


namespace second_triangle_weight_l2699_269996

/-- Represents an equilateral triangle with given side length and weight -/
structure EquilateralTriangle where
  side_length : ℝ
  weight : ℝ

/-- Calculate the weight of a second equilateral triangle given the properties of a first triangle -/
def calculate_second_triangle_weight (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side_length = 2 ∧ 
  t1.weight = 20 ∧ 
  t2.side_length = 4 ∧ 
  t2.weight = 80

theorem second_triangle_weight (t1 t2 : EquilateralTriangle) : 
  calculate_second_triangle_weight t1 t2 := by
  sorry

end second_triangle_weight_l2699_269996


namespace unique_solution_trigonometric_equation_l2699_269928

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 110 := by
sorry

end unique_solution_trigonometric_equation_l2699_269928


namespace fraction_equation_solution_l2699_269900

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 5) / (x - 3) = 4 ∧ x = 17/3 := by
  sorry

end fraction_equation_solution_l2699_269900


namespace five_points_two_small_triangles_l2699_269941

-- Define a triangular region with unit area
def UnitTriangle : Set (ℝ × ℝ) := sorry

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem five_points_two_small_triangles 
  (points : Finset (ℝ × ℝ)) 
  (h1 : points.card = 5) 
  (h2 : ∀ p ∈ points, p ∈ UnitTriangle) : 
  ∃ (t1 t2 : Finset (ℝ × ℝ)), 
    t1 ⊆ points ∧ t2 ⊆ points ∧ 
    t1.card = 3 ∧ t2.card = 3 ∧ 
    t1 ≠ t2 ∧
    (∃ (p1 p2 p3 : ℝ × ℝ), p1 ∈ t1 ∧ p2 ∈ t1 ∧ p3 ∈ t1 ∧ triangleArea p1 p2 p3 ≤ 1/4) ∧
    (∃ (q1 q2 q3 : ℝ × ℝ), q1 ∈ t2 ∧ q2 ∈ t2 ∧ q3 ∈ t2 ∧ triangleArea q1 q2 q3 ≤ 1/4) :=
by sorry

end five_points_two_small_triangles_l2699_269941


namespace solution_is_correct_l2699_269961

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(8, 3, 3, 1), (5, 4, 3, 1), (3, 2, 2, 2), (7, 6, 2, 1), (9, 5, 2, 1), (15, 4, 2, 1),
   (1, 1, 1, 7), (2, 1, 1, 5), (3, 2, 1, 3), (8, 3, 1, 2), (5, 4, 1, 2)}

def satisfies_conditions (x y z t : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧
  x * y * z = Nat.factorial t ∧
  (x + 1) * (y + 1) * (z + 1) = Nat.factorial (t + 1)

theorem solution_is_correct :
  ∀ x y z t, (x, y, z, t) ∈ solution_set ↔ satisfies_conditions x y z t := by
  sorry

end solution_is_correct_l2699_269961


namespace isosceles_triangle_sides_l2699_269986

/-- Represents the sides of an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- Checks if the given sides form a valid isosceles triangle -/
def is_valid_isosceles (t : IsoscelesTriangle) : Prop :=
  t.base > 0 ∧ t.leg > 0 ∧ t.leg + t.leg > t.base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.base + 2 * t.leg

theorem isosceles_triangle_sides (p : ℝ) (s : ℝ) :
  p = 26 ∧ s = 11 →
  ∃ (t1 t2 : IsoscelesTriangle),
    (perimeter t1 = p ∧ (t1.base = s ∨ t1.leg = s) ∧ is_valid_isosceles t1) ∧
    (perimeter t2 = p ∧ (t2.base = s ∨ t2.leg = s) ∧ is_valid_isosceles t2) ∧
    ((t1.base = 11 ∧ t1.leg = 7.5) ∨ (t1.leg = 11 ∧ t1.base = 4)) ∧
    ((t2.base = 11 ∧ t2.leg = 7.5) ∨ (t2.leg = 11 ∧ t2.base = 4)) ∧
    t1 ≠ t2 :=
by sorry

end isosceles_triangle_sides_l2699_269986


namespace ones_digit_of_largest_power_of_2_dividing_32_factorial_l2699_269987

/-- The largest power of 2 that divides n! -/
def largestPowerOf2DividingFactorial (n : ℕ) : ℕ :=
  (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32)

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  onesDigit (2^(largestPowerOf2DividingFactorial 32)) = 8 :=
sorry

end ones_digit_of_largest_power_of_2_dividing_32_factorial_l2699_269987


namespace sqrt_four_cubed_sum_l2699_269959

theorem sqrt_four_cubed_sum : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end sqrt_four_cubed_sum_l2699_269959


namespace third_divisor_l2699_269968

theorem third_divisor (n : ℕ) (h1 : n = 200) 
  (h2 : ∃ k₁ k₂ k₃ k₄ : ℕ, n + 20 = 15 * k₁ ∧ n + 20 = 30 * k₂ ∧ n + 20 = 60 * k₄) 
  (h3 : ∃ x : ℕ, x ≠ 15 ∧ x ≠ 30 ∧ x ≠ 60 ∧ ∃ k : ℕ, n + 20 = x * k) : 
  ∃ x : ℕ, x = 11 ∧ x ≠ 15 ∧ x ≠ 30 ∧ x ≠ 60 ∧ ∃ k : ℕ, n + 20 = x * k :=
sorry

end third_divisor_l2699_269968


namespace compute_expression_l2699_269944

theorem compute_expression : 20 * (256 / 4 + 64 / 16 + 16 / 64 + 2) = 1405 := by
  sorry

end compute_expression_l2699_269944


namespace room_tiles_theorem_l2699_269995

/-- Given a room with length and width in centimeters, 
    calculate the least number of square tiles required to cover the floor. -/
def leastNumberOfTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  (length / tileSize) * (width / tileSize)

/-- Theorem stating that for a room with length 720 cm and width 432 cm,
    the least number of square tiles required is 15. -/
theorem room_tiles_theorem :
  leastNumberOfTiles 720 432 = 15 := by
  sorry

#eval leastNumberOfTiles 720 432

end room_tiles_theorem_l2699_269995


namespace kids_savings_l2699_269956

-- Define the number of coins each child has
def teagan_pennies : ℕ := 200
def rex_nickels : ℕ := 100
def toni_dimes : ℕ := 330

-- Define the value of each coin type in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10

-- Define the total savings in cents
def total_savings : ℕ := 
  teagan_pennies * penny_value + 
  rex_nickels * nickel_value + 
  toni_dimes * dime_value

-- Theorem to prove
theorem kids_savings : total_savings = 4000 := by
  sorry

end kids_savings_l2699_269956


namespace restaurant_seating_capacity_l2699_269965

theorem restaurant_seating_capacity :
  ∀ (new_tables original_tables : ℕ),
    new_tables + original_tables = 40 →
    new_tables = original_tables + 12 →
    6 * new_tables + 4 * original_tables = 212 :=
by
  sorry

end restaurant_seating_capacity_l2699_269965


namespace fourth_person_height_l2699_269945

theorem fourth_person_height :
  ∀ (h₁ h₂ h₃ h₄ : ℝ),
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →
  h₂ - h₁ = 2 →
  h₃ - h₂ = 2 →
  h₄ - h₃ = 6 →
  (h₁ + h₂ + h₃ + h₄) / 4 = 79 →
  h₄ = 85 :=
by
  sorry

end fourth_person_height_l2699_269945


namespace inequality_solution_count_l2699_269960

theorem inequality_solution_count : 
  ∃! (s : Finset ℤ), 
    (∀ n : ℤ, n ∈ s ↔ Real.sqrt (3*n - 1) ≤ Real.sqrt (5*n - 7) ∧ 
                       Real.sqrt (5*n - 7) < Real.sqrt (3*n + 8)) ∧ 
    s.card = 5 :=
sorry

end inequality_solution_count_l2699_269960


namespace box_matching_problem_l2699_269988

/-- Represents the problem of matching box bodies and bottoms --/
theorem box_matching_problem (total_tinplates : ℕ) 
  (bodies_per_tinplate : ℕ) (bottoms_per_tinplate : ℕ) 
  (bottoms_per_body : ℕ) (bodies_tinplates : ℕ) (bottoms_tinplates : ℕ) :
  total_tinplates = 36 →
  bodies_per_tinplate = 25 →
  bottoms_per_tinplate = 40 →
  bottoms_per_body = 2 →
  bodies_tinplates = 16 →
  bottoms_tinplates = 20 →
  bodies_tinplates + bottoms_tinplates = total_tinplates ∧
  bodies_per_tinplate * bodies_tinplates * bottoms_per_body = 
    bottoms_per_tinplate * bottoms_tinplates :=
by sorry

end box_matching_problem_l2699_269988
