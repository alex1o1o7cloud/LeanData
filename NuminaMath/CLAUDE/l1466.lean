import Mathlib

namespace three_digit_swap_subtraction_l1466_146683

theorem three_digit_swap_subtraction (a b c : ℕ) : 
  a ≤ 9 → b ≤ 9 → c ≤ 9 → a ≠ 0 → a = c + 3 →
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) ≡ 7 [MOD 10] :=
by sorry

end three_digit_swap_subtraction_l1466_146683


namespace consecutive_integers_sqrt_3_l1466_146654

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) →  -- a and b are consecutive integers
  (↑a < Real.sqrt 3) →  -- a < sqrt(3)
  (Real.sqrt 3 < ↑b) →  -- sqrt(3) < b
  a + b = 3 := by sorry

end consecutive_integers_sqrt_3_l1466_146654


namespace smallest_four_digit_divisible_by_37_l1466_146620

theorem smallest_four_digit_divisible_by_37 : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧  -- four-digit number
  n % 37 = 0 ∧              -- divisible by 37
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) ∧ m % 37 = 0 → n ≤ m) ∧  -- smallest such number
  n = 1036 :=               -- the answer is 1036
by sorry

end smallest_four_digit_divisible_by_37_l1466_146620


namespace consecutive_product_l1466_146669

theorem consecutive_product (t : ℤ) :
  let n : ℤ := t * (t + 1) - 1
  (n^2 - 1 : ℤ) = (t - 1) * t * (t + 1) * (t + 2) :=
by sorry

end consecutive_product_l1466_146669


namespace sqrt_equation_solution_l1466_146651

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (7 - 5 * z) = 10 :=
by
  sorry

end sqrt_equation_solution_l1466_146651


namespace arccos_sum_solution_l1466_146686

theorem arccos_sum_solution (x : ℝ) : 
  Real.arccos (2 * x) + Real.arccos (3 * x) = π / 2 → x = 1 / Real.sqrt 13 ∨ x = -1 / Real.sqrt 13 :=
by sorry

end arccos_sum_solution_l1466_146686


namespace garden_transformation_l1466_146680

/-- Represents a rectangular garden --/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Represents a square garden --/
structure SquareGarden where
  side : ℝ

/-- Calculates the perimeter of a rectangular garden --/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.width)

/-- Calculates the area of a rectangular garden --/
def areaRectangular (garden : RectangularGarden) : ℝ :=
  garden.length * garden.width

/-- Calculates the area of a square garden --/
def areaSquare (garden : SquareGarden) : ℝ :=
  garden.side * garden.side

/-- Theorem: Changing a 60x20 rectangular garden to a square with the same perimeter
    results in a 40x40 square garden and increases the area by 400 square feet --/
theorem garden_transformation (original : RectangularGarden) 
    (h1 : original.length = 60)
    (h2 : original.width = 20) :
    ∃ (new : SquareGarden),
      perimeter original = 4 * new.side ∧
      new.side = 40 ∧
      areaSquare new - areaRectangular original = 400 := by
  sorry

end garden_transformation_l1466_146680


namespace geometric_sequence_problem_l1466_146677

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 210 * r = b ∧ b * r = 135 / 56) →
  b = 22.5 := by
sorry

end geometric_sequence_problem_l1466_146677


namespace distinct_combinations_count_l1466_146642

def word : String := "BIOLOGY"

def num_vowels : Nat := 3
def num_consonants : Nat := 3

def is_vowel (c : Char) : Bool :=
  c = 'I' || c = 'O'

def is_consonant (c : Char) : Bool :=
  c = 'B' || c = 'L' || c = 'G'

def indistinguishable (c : Char) : Bool :=
  c = 'I' || c = 'G'

theorem distinct_combinations_count :
  (∃ (vowel_combs consonant_combs : Nat),
    vowel_combs * consonant_combs = 12 ∧
    vowel_combs = (word.toList.filter is_vowel).length.choose num_vowels ∧
    consonant_combs = (word.toList.filter is_consonant).length.choose num_consonants) :=
by sorry

end distinct_combinations_count_l1466_146642


namespace product_xy_equals_one_l1466_146696

theorem product_xy_equals_one (x y : ℝ) (h_distinct : x ≠ y) 
    (h_eq : (1 / (1 + x^2)) + (1 / (1 + y^2)) = 2 / (1 + x*y)) : x * y = 1 := by
  sorry

end product_xy_equals_one_l1466_146696


namespace power_of_power_l1466_146626

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end power_of_power_l1466_146626


namespace sheep_purchase_l1466_146648

/-- Calculates the number of sheep Mary needs to buy to have 69 fewer sheep than Bob -/
theorem sheep_purchase (mary_initial : ℕ) (bob_multiplier : ℕ) (bob_additional : ℕ) (target_difference : ℕ) : 
  mary_initial = 300 →
  bob_multiplier = 2 →
  bob_additional = 35 →
  target_difference = 69 →
  (mary_initial + (bob_multiplier * mary_initial + bob_additional - target_difference - mary_initial)) = 566 :=
by sorry

end sheep_purchase_l1466_146648


namespace chinese_coin_problem_l1466_146663

/-- Represents an arithmetic sequence of 5 terms -/
structure ArithmeticSequence :=
  (a : ℚ) -- First term
  (d : ℚ) -- Common difference

/-- Properties of the specific arithmetic sequence in the problem -/
def ProblemSequence (seq : ArithmeticSequence) : Prop :=
  -- Sum of all terms is 5
  seq.a - 2*seq.d + seq.a - seq.d + seq.a + seq.a + seq.d + seq.a + 2*seq.d = 5 ∧
  -- Sum of first two terms equals sum of last three terms
  seq.a - 2*seq.d + seq.a - seq.d = seq.a + seq.a + seq.d + seq.a + 2*seq.d

theorem chinese_coin_problem (seq : ArithmeticSequence) 
  (h : ProblemSequence seq) : seq.a - seq.d = 7/6 := by
  sorry

end chinese_coin_problem_l1466_146663


namespace symmetric_circle_l1466_146604

/-- Given a circle and a line of symmetry, find the equation of the symmetric circle -/
theorem symmetric_circle (x y : ℝ) : 
  (∀ x y, (x + 2)^2 + y^2 = 2016) →  -- Original circle equation
  (∀ x y, x - y + 1 = 0) →           -- Line of symmetry
  (∀ x y, (x + 1)^2 + (y + 1)^2 = 2016) -- Symmetric circle equation
:= by sorry

end symmetric_circle_l1466_146604


namespace expression_value_l1466_146695

theorem expression_value (x y : ℝ) (h : x + y = 3) : 2*x + 2*y - 1 = 5 := by
  sorry

end expression_value_l1466_146695


namespace total_average_marks_l1466_146633

/-- Given two classes with different numbers of students and average marks,
    calculate the total average marks of all students in both classes. -/
theorem total_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 / ((n1 : ℚ) + (n2 : ℚ)) =
  ((39 : ℚ) * 45 + (35 : ℚ) * 70) / ((39 : ℚ) + (35 : ℚ)) :=
by
  sorry

#eval ((39 : ℚ) * 45 + (35 : ℚ) * 70) / ((39 : ℚ) + (35 : ℚ))

end total_average_marks_l1466_146633


namespace line_circle_distance_theorem_l1466_146664

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Length of tangent from a point to a circle -/
def tangentLength (p : ℝ × ℝ) (c : Circle) : ℝ := sorry

/-- Whether a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop := sorry

/-- Whether a point is on a line -/
def onLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

theorem line_circle_distance_theorem (c : Circle) (l : Line) :
  (¬ intersects l c) →
  (∀ (A B : ℝ × ℝ), onLine A l → onLine B l →
    (distance A B > |tangentLength A c - tangentLength B c| ∧
     distance A B < tangentLength A c + tangentLength B c)) ∧
  (∃ (A B : ℝ × ℝ), onLine A l → onLine B l →
    (distance A B ≤ |tangentLength A c - tangentLength B c| ∨
     distance A B ≥ tangentLength A c + tangentLength B c) →
    intersects l c) :=
by sorry

end line_circle_distance_theorem_l1466_146664


namespace largest_ratio_in_arithmetic_sequence_l1466_146618

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    if S_15 > 0 and S_16 < 0, then S_8/a_8 is the largest among S_1/a_1, S_2/a_2, ..., S_15/a_15 -/
theorem largest_ratio_in_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (h_S15 : S 15 > 0) 
  (h_S16 : S 16 < 0) : 
  ∀ k ∈ Finset.range 15, S 8 / a 8 ≥ S (k + 1) / a (k + 1) :=
by sorry

end largest_ratio_in_arithmetic_sequence_l1466_146618


namespace intercept_sum_l1466_146624

-- Define the two lines
def line1 (x y : ℝ) : Prop := 20 * x + 16 * y - 40 = 0
def line2 (x y : ℝ) : Prop := 20 * x + 16 * y - 64 = 0

-- Define x-intercept of line1
def x_intercept : ℝ := 2

-- Define y-intercept of line2
def y_intercept : ℝ := 4

-- Theorem statement
theorem intercept_sum : x_intercept + y_intercept = 6 := by sorry

end intercept_sum_l1466_146624


namespace jean_initial_candy_l1466_146644

/-- The number of candy pieces Jean had initially -/
def initial_candy : ℕ := sorry

/-- The number of candy pieces Jean gave to her first friend -/
def first_friend : ℕ := 18

/-- The number of candy pieces Jean gave to her second friend -/
def second_friend : ℕ := 12

/-- The number of candy pieces Jean gave to her third friend -/
def third_friend : ℕ := 25

/-- The number of candy pieces Jean bought -/
def bought : ℕ := 10

/-- The number of candy pieces Jean ate -/
def ate : ℕ := 7

/-- The number of candy pieces Jean has left -/
def left : ℕ := 16

theorem jean_initial_candy : 
  initial_candy = first_friend + second_friend + third_friend + left + ate - bought :=
by sorry

end jean_initial_candy_l1466_146644


namespace original_egg_count_l1466_146688

/-- Given a jar of eggs, prove that the original number of eggs is 27 
    when 7 eggs are removed and 20 eggs remain. -/
theorem original_egg_count (removed : ℕ) (remaining : ℕ) : removed = 7 → remaining = 20 → removed + remaining = 27 := by
  sorry

end original_egg_count_l1466_146688


namespace min_value_a_l1466_146668

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, |x + a| - |x + 1| ≤ 2*a) ↔ a ≥ 1/3 :=
sorry

end min_value_a_l1466_146668


namespace arithmetic_sequence_inequality_l1466_146627

/-- Four distinct positive real numbers form an arithmetic sequence -/
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r

theorem arithmetic_sequence_inequality (a b c d : ℝ) 
  (h : is_arithmetic_sequence a b c d) : (a + d) / 2 > Real.sqrt (b * c) := by
  sorry

end arithmetic_sequence_inequality_l1466_146627


namespace max_value_of_f_range_of_k_l1466_146670

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x - 9 / x

theorem max_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ f x = (1 : ℝ) / Real.exp 10 ∧ ∀ y > 0, f y ≤ f x :=
sorry

theorem range_of_k :
  ∀ k : ℝ,
  (∀ x : ℝ, x ≥ 1 → x^2 * ((Real.log x) / x - k / x) + 1 / (x + 1) ≥ 0) →
  (∀ x : ℝ, x ≥ 1 → k ≥ (1/2) * x^2 + (Real.exp 2 - 2) * x - Real.exp x - 7) →
  (Real.exp 2 - 9 ≤ k ∧ k ≤ 1/2) :=
sorry

end max_value_of_f_range_of_k_l1466_146670


namespace simplify_expression_l1466_146611

theorem simplify_expression (y : ℝ) : 8*y - 3 + 2*y + 15 = 10*y + 12 := by
  sorry

end simplify_expression_l1466_146611


namespace sum_of_reciprocals_l1466_146621

theorem sum_of_reciprocals (a b : ℝ) (h1 : a ≠ b) (h2 : a/b + a = b/a + b) : 1/a + 1/b = -1 := by
  sorry

end sum_of_reciprocals_l1466_146621


namespace football_outcomes_count_l1466_146650

def FootballOutcome := Nat × Nat × Nat

def total_matches (outcome : FootballOutcome) : Nat :=
  outcome.1 + outcome.2.1 + outcome.2.2

def total_points (outcome : FootballOutcome) : Nat :=
  3 * outcome.1 + outcome.2.1

def is_valid_outcome (outcome : FootballOutcome) : Prop :=
  total_matches outcome = 14 ∧ total_points outcome = 19

theorem football_outcomes_count :
  ∃! n : Nat, ∃ outcomes : Finset FootballOutcome,
    outcomes.card = n ∧
    (∀ o : FootballOutcome, o ∈ outcomes ↔ is_valid_outcome o) ∧
    n = 4 := by sorry

end football_outcomes_count_l1466_146650


namespace distance_between_points_l1466_146687

theorem distance_between_points : Real.sqrt ((7 - (-5))^2 + (3 - (-2))^2) = 13 := by
  sorry

end distance_between_points_l1466_146687


namespace no_real_solutions_for_inequality_l1466_146665

theorem no_real_solutions_for_inequality :
  ¬∃ (x : ℝ), x^2 - 4*x + 4 < 0 :=
by sorry

end no_real_solutions_for_inequality_l1466_146665


namespace not_right_triangle_l1466_146678

theorem not_right_triangle (a b c : ℝ) (h : a = 3 ∧ b = 5 ∧ c = 7) : 
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end not_right_triangle_l1466_146678


namespace age_problem_l1466_146674

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 := by
sorry

end age_problem_l1466_146674


namespace box_triangle_area_theorem_l1466_146697

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  width : ℝ
  length : ℝ
  height : ℚ

/-- Calculates the area of the triangle formed by the center points of three faces meeting at a corner -/
def triangleArea (d : BoxDimensions) : ℝ :=
  sorry

/-- Checks if two integers are relatively prime -/
def relativelyPrime (m n : ℕ) : Prop :=
  sorry

theorem box_triangle_area_theorem 
  (d : BoxDimensions)
  (m n : ℕ)
  (h1 : d.width = 15)
  (h2 : d.length = 20)
  (h3 : d.height = m / n)
  (h4 : relativelyPrime m n)
  (h5 : triangleArea d = 40) :
  m + n = 69 :=
sorry

end box_triangle_area_theorem_l1466_146697


namespace pearl_division_l1466_146647

theorem pearl_division (n : ℕ) : 
  (n > 0) →
  (n % 8 = 6) → 
  (n % 7 = 5) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 9 = 1) := by
sorry

end pearl_division_l1466_146647


namespace no_real_solution_l1466_146653

theorem no_real_solution :
  ¬∃ (x y z : ℝ), x^2 + 4*y*z + 2*z = 0 ∧ x + 2*x*y + 2*z^2 = 0 ∧ 2*x*z + y^2 + y + 1 = 0 := by
  sorry

end no_real_solution_l1466_146653


namespace complement_A_intersect_B_l1466_146612

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | -x^2 + 6*x - 8 > 0}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Ioc 2 3 := by sorry

end complement_A_intersect_B_l1466_146612


namespace paco_cookies_l1466_146616

theorem paco_cookies (initial_cookies : ℕ) : initial_cookies = 7 :=
by
  -- Define the number of cookies eaten initially
  let initial_eaten : ℕ := 5
  -- Define the number of cookies bought
  let bought : ℕ := 3
  -- Define the number of cookies eaten after buying
  let later_eaten : ℕ := bought + 2
  -- Assert that all cookies were eaten
  have h : initial_cookies - initial_eaten + bought - later_eaten = 0 := by sorry
  -- Prove that initial_cookies = 7
  sorry

end paco_cookies_l1466_146616


namespace wire_cutting_l1466_146605

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 60 →
  ratio = 2 / 5 →
  shorter_piece + shorter_piece / ratio = total_length →
  shorter_piece = 120 / 7 := by
sorry

end wire_cutting_l1466_146605


namespace ellipse_condition_l1466_146685

def represents_ellipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  ∀ (x y : ℝ), x^2 / (5 - m) + y^2 / (m + 3) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_condition (m : ℝ) : 
  represents_ellipse m → m > -3 ∧ m < 5 :=
sorry

end ellipse_condition_l1466_146685


namespace dvd_price_calculation_l1466_146671

theorem dvd_price_calculation (num_dvd : ℕ) (num_bluray : ℕ) (bluray_price : ℚ) (avg_price : ℚ) :
  num_dvd = 8 →
  num_bluray = 4 →
  bluray_price = 18 →
  avg_price = 14 →
  ∃ (dvd_price : ℚ),
    dvd_price * num_dvd + bluray_price * num_bluray = avg_price * (num_dvd + num_bluray) ∧
    dvd_price = 12 :=
by sorry

end dvd_price_calculation_l1466_146671


namespace multiplicative_inverse_modulo_l1466_146652

def A : ℕ := 123456
def B : ℕ := 171717
def M : ℕ := 1000003
def N : ℕ := 538447

theorem multiplicative_inverse_modulo :
  (A * B * N) % M = 1 := by sorry

end multiplicative_inverse_modulo_l1466_146652


namespace rectangle_area_breadth_ratio_l1466_146632

/-- Proves that for a rectangular plot with breadth 8 meters and length 10 meters more than its breadth, the ratio of its area to its breadth is 18:1. -/
theorem rectangle_area_breadth_ratio :
  let b : ℝ := 8  -- breadth
  let l : ℝ := b + 10  -- length
  let A : ℝ := l * b  -- area
  A / b = 18 := by
  sorry

end rectangle_area_breadth_ratio_l1466_146632


namespace line_through_origin_and_point_l1466_146617

/-- A line passing through two points (0,0) and (-4,3) has the function expression y = -3/4 * x -/
theorem line_through_origin_and_point (x y : ℝ) : 
  (0 : ℝ) = 0 * x + y ∧ 3 = -4 * (-3/4) + y → y = -3/4 * x := by
sorry

end line_through_origin_and_point_l1466_146617


namespace equation_containing_2012_l1466_146622

theorem equation_containing_2012 (n : ℕ) : 
  (n^2 ≤ 2012 ∧ ∀ m : ℕ, m > n → m^2 > 2012) → 
  (n = 44 ∧ 2012 ∈ Finset.range (2*n^2 - n^2 + 1) \ Finset.range (n^2)) := by
  sorry

end equation_containing_2012_l1466_146622


namespace potato_cooking_time_l1466_146643

/-- Given a chef cooking potatoes with the following conditions:
  - Total potatoes to cook is 15
  - Potatoes already cooked is 6
  - Time to cook the remaining potatoes is 72 minutes
  Prove that the time to cook one potato is 8 minutes. -/
theorem potato_cooking_time (total : Nat) (cooked : Nat) (remaining_time : Nat) :
  total = 15 → cooked = 6 → remaining_time = 72 → (remaining_time / (total - cooked) = 8) :=
by sorry

end potato_cooking_time_l1466_146643


namespace track_circumference_l1466_146637

/-- Represents a circular track with two runners -/
structure CircularTrack where
  /-- The circumference of the track in yards -/
  circumference : ℝ
  /-- The distance B travels before the first meeting in yards -/
  first_meeting_distance : ℝ
  /-- The distance A has left to complete a lap at the second meeting in yards -/
  second_meeting_remaining : ℝ

/-- The theorem stating the circumference of the track given the conditions -/
theorem track_circumference (track : CircularTrack)
  (h1 : track.first_meeting_distance = 150)
  (h2 : track.second_meeting_remaining = 90)
  (h3 : track.first_meeting_distance < track.circumference / 2)
  (h4 : track.second_meeting_remaining < track.circumference) :
  track.circumference = 720 := by
  sorry

#check track_circumference

end track_circumference_l1466_146637


namespace kids_difference_l1466_146602

/-- The number of kids Julia played with on different days of the week. -/
structure KidsPlayed where
  monday : ℕ
  wednesday : ℕ

/-- Theorem stating the difference in number of kids played with between Monday and Wednesday. -/
theorem kids_difference (k : KidsPlayed) (h1 : k.monday = 6) (h2 : k.wednesday = 4) :
  k.monday - k.wednesday = 2 := by
  sorry

end kids_difference_l1466_146602


namespace derivative_of_f_l1466_146623

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- State the theorem
theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 3 * x^2 := by sorry

end derivative_of_f_l1466_146623


namespace conference_center_occupancy_l1466_146667

theorem conference_center_occupancy (rooms : Nat) (capacity : Nat) (occupancy_ratio : Rat) : 
  rooms = 12 →
  capacity = 150 →
  occupancy_ratio = 5/7 →
  (rooms * capacity * occupancy_ratio).floor = 1285 := by
  sorry

end conference_center_occupancy_l1466_146667


namespace perimeter_is_64_l1466_146610

/-- A structure formed by nine congruent squares -/
structure SquareStructure where
  /-- The side length of each square in the structure -/
  side_length : ℝ
  /-- The total area of the structure is 576 square centimeters -/
  total_area_eq : side_length ^ 2 * 9 = 576

/-- The perimeter of the square structure -/
def perimeter (s : SquareStructure) : ℝ :=
  8 * s.side_length

/-- Theorem stating that the perimeter of the structure is 64 centimeters -/
theorem perimeter_is_64 (s : SquareStructure) : perimeter s = 64 := by
  sorry

#check perimeter_is_64

end perimeter_is_64_l1466_146610


namespace sqrt_expression_one_sqrt_expression_two_sqrt_expression_three_l1466_146681

-- Problem 1
theorem sqrt_expression_one : 
  Real.sqrt 27 + Real.sqrt 3 - Real.sqrt 12 = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem sqrt_expression_two : 
  1 / Real.sqrt 24 + |Real.sqrt 6 - 3| + (1/2)⁻¹ - 2016^0 = 4 - 11 * Real.sqrt 6 / 12 := by sorry

-- Problem 3
theorem sqrt_expression_three : 
  (Real.sqrt 3 + Real.sqrt 2)^2 - (Real.sqrt 3 - Real.sqrt 2)^2 = 4 * Real.sqrt 6 := by sorry

end sqrt_expression_one_sqrt_expression_two_sqrt_expression_three_l1466_146681


namespace solution_set_f_range_of_m_l1466_146682

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- Theorem for part (1)
theorem solution_set_f (x : ℝ) : f x + x^2 - 1 > 0 ↔ x > 1 ∨ x < 0 := by sorry

-- Theorem for part (2)
theorem range_of_m (m : ℝ) : (∃ x, f x < g x m) ↔ m > 4 := by sorry

end solution_set_f_range_of_m_l1466_146682


namespace intersection_M_N_l1466_146676

-- Define set M
def M : Set ℝ := {x : ℝ | -4 ≤ x ∧ x ≤ 7}

-- Define set N
def N : Set ℝ := {x : ℝ | x^2 - x - 6 > 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x : ℝ | (3 < x ∧ x ≤ 7) ∨ (-4 ≤ x ∧ x < -2)} := by
  sorry

end intersection_M_N_l1466_146676


namespace no_extremum_l1466_146649

open Real

/-- A function satisfying the given differential equation and initial condition -/
def SolutionFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x > 0 → x * (deriv f x) + f x = exp x / x) ∧ f 1 = exp 1

/-- The main theorem stating that the function has no maximum or minimum -/
theorem no_extremum (f : ℝ → ℝ) (hf : SolutionFunction f) :
    (∀ x, x > 0 → ¬ IsLocalMax f x) ∧ (∀ x, x > 0 → ¬ IsLocalMin f x) := by
  sorry


end no_extremum_l1466_146649


namespace cube_root_monotone_l1466_146656

theorem cube_root_monotone {a b : ℝ} (h : a > b) : a^(1/3) > b^(1/3) := by
  sorry

end cube_root_monotone_l1466_146656


namespace magic_polynomial_bound_l1466_146675

open Polynomial
open Nat

theorem magic_polynomial_bound (n : ℕ) (P : Polynomial ℚ) 
  (h_deg : degree P = n) (h_irr : Irreducible P) :
  ∃ (s : Finset (Polynomial ℚ)), 
    (∀ Q ∈ s, degree Q < n ∧ (P ∣ (P.comp Q))) ∧ 
    (∀ Q : Polynomial ℚ, degree Q < n → (P ∣ (P.comp Q)) → Q ∈ s) ∧
    s.card ≤ n := by
  sorry

end magic_polynomial_bound_l1466_146675


namespace expand_expression_l1466_146662

theorem expand_expression (x : ℝ) : (17 * x + 18) * (3 * x + 4) = 51 * x^2 + 122 * x + 72 := by
  sorry

end expand_expression_l1466_146662


namespace cream_ratio_l1466_146609

-- Define the initial quantities
def initial_coffee : ℚ := 20
def john_drink : ℚ := 3
def john_cream : ℚ := 4
def jane_cream : ℚ := 3
def jane_drink : ℚ := 5

-- Calculate the amounts of cream in each cup
def john_cream_amount : ℚ := john_cream

def jane_mixture : ℚ := initial_coffee + jane_cream
def jane_cream_ratio : ℚ := jane_cream / jane_mixture
def jane_remaining : ℚ := jane_mixture - jane_drink
def jane_cream_amount : ℚ := jane_cream_ratio * jane_remaining

-- State the theorem
theorem cream_ratio : john_cream_amount / jane_cream_amount = 46 / 27 := by
  sorry

end cream_ratio_l1466_146609


namespace unique_prime_permutation_residue_system_l1466_146613

theorem unique_prime_permutation_residue_system : ∃! (p : ℕ), 
  Nat.Prime p ∧ 
  p % 2 = 1 ∧
  ∃ (b : Fin (p - 1) → Fin (p - 1)), Function.Bijective b ∧
    (∀ (x : Fin (p - 1)), 
      ∃ (y : Fin (p - 1)), (x.val + 1) ^ (b y).val ≡ (y.val + 1) [ZMOD p]) ∧
  p = 3 := by
sorry

end unique_prime_permutation_residue_system_l1466_146613


namespace solution_x_squared_equals_three_l1466_146672

theorem solution_x_squared_equals_three :
  ∀ x : ℝ, x^2 = 3 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 :=
by sorry

end solution_x_squared_equals_three_l1466_146672


namespace max_y_diff_intersection_points_l1466_146635

/-- The maximum difference between the y-coordinates of the intersection points
    of y = 4 - 2x^2 + x^3 and y = 2 + x^2 + x^3 is 4√2/9. -/
theorem max_y_diff_intersection_points :
  let f (x : ℝ) := 4 - 2 * x^2 + x^3
  let g (x : ℝ) := 2 + x^2 + x^3
  let intersection_points := {x : ℝ | f x = g x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ intersection_points ∧ x₂ ∈ intersection_points ∧
    |f x₁ - f x₂| = 4 * Real.sqrt 2 / 9 ∧
    ∀ (y₁ y₂ : ℝ), y₁ ∈ intersection_points → y₂ ∈ intersection_points →
      |f y₁ - f y₂| ≤ 4 * Real.sqrt 2 / 9 := by
  sorry

end max_y_diff_intersection_points_l1466_146635


namespace min_values_proof_l1466_146606

theorem min_values_proof (a b m x : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) (hx : x > 2) :
  (a + b ≥ 2 * Real.sqrt (a * b)) ∧
  (a + b = 2 * Real.sqrt (a * b) ↔ a = b) →
  ((m + 1 / m ≥ 2) ∧ (∃ m₀ > 0, m₀ + 1 / m₀ = 2)) ∧
  ((x^2 + x - 5) / (x - 2) ≥ 7 ∧ (∃ x₀ > 2, (x₀^2 + x₀ - 5) / (x₀ - 2) = 7)) := by
  sorry

end min_values_proof_l1466_146606


namespace lcm_of_8_and_15_l1466_146608

theorem lcm_of_8_and_15 : Nat.lcm 8 15 = 120 := by
  sorry

end lcm_of_8_and_15_l1466_146608


namespace parallel_line_slope_l1466_146698

/-- Given a line with equation 3x - 6y = 12, prove that the slope of any parallel line is 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → 
  ∃ (m b : ℝ), (y = m * x + b) ∧ (m = 1/2) :=
by sorry

end parallel_line_slope_l1466_146698


namespace sum_product_inequality_l1466_146658

theorem sum_product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := by
sorry

end sum_product_inequality_l1466_146658


namespace fourier_series_sum_l1466_146600

open Real

noncomputable def y (x : ℝ) : ℝ := x * cos x

theorem fourier_series_sum : 
  ∃ (S : ℝ), S = ∑' k, (4 * k^2 + 1) / (4 * k^2 - 1)^2 ∧ S = π^2 / 8 + 1 / 2 :=
by
  sorry

end fourier_series_sum_l1466_146600


namespace transformations_result_l1466_146659

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotates a point 180° around the x-axis -/
def rotateX (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

/-- Reflects a point through the xy-plane -/
def reflectXY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Reflects a point through the yz-plane -/
def reflectYZ (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

/-- Applies the sequence of transformations to a point -/
def applyTransformations (p : Point3D) : Point3D :=
  reflectYZ (rotateX (reflectYZ (reflectXY (rotateX p))))

theorem transformations_result :
  applyTransformations { x := 1, y := 1, z := 1 } = { x := 1, y := 1, z := -1 } := by
  sorry


end transformations_result_l1466_146659


namespace perpendicular_parallel_transitive_l1466_146694

-- Define the types for planes and lines
variable (Plane Line : Type*)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_transitive
  (α β : Plane) (l : Line)
  (h1 : perpendicular l α)
  (h2 : parallel α β) :
  perpendicular l β :=
sorry

end perpendicular_parallel_transitive_l1466_146694


namespace average_equation_solution_l1466_146645

theorem average_equation_solution (x : ℚ) : 
  (1 / 3 : ℚ) * ((3 * x + 8) + (7 * x - 3) + (4 * x + 5)) = 5 * x - 6 → x = -28 := by
  sorry

end average_equation_solution_l1466_146645


namespace base_representation_digit_difference_l1466_146629

theorem base_representation_digit_difference : 
  let n : ℕ := 1234
  let base_5_digits := (Nat.log n 5).succ
  let base_9_digits := (Nat.log n 9).succ
  base_5_digits - base_9_digits = 1 := by
sorry

end base_representation_digit_difference_l1466_146629


namespace small_rectangle_perimeter_l1466_146684

/-- Given a square with perimeter 160 units, divided into two congruent rectangles,
    with one of those rectangles further divided into three smaller congruent rectangles,
    the perimeter of one of the three smaller congruent rectangles is equal to 2 * (20 + 40/3) units. -/
theorem small_rectangle_perimeter (s : ℝ) (h1 : s > 0) (h2 : 4 * s = 160) :
  2 * (s / 2 + s / 6) = 2 * (20 + 40 / 3) :=
sorry

end small_rectangle_perimeter_l1466_146684


namespace max_value_on_ellipse_l1466_146603

theorem max_value_on_ellipse (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_ellipse : 4 * x^2 + 9 * y^2 = 36) : 
  ∀ u v : ℝ, u > 0 → v > 0 → 4 * u^2 + 9 * v^2 = 36 → x + 2*y ≤ 5 ∧ x + 2*y = 5 → u + 2*v ≤ 5 := by
sorry

end max_value_on_ellipse_l1466_146603


namespace smallest_square_ending_644_l1466_146666

theorem smallest_square_ending_644 :
  ∀ n : ℕ+, n.val < 194 → (n.val ^ 2) % 1000 ≠ 644 ∧ (194 ^ 2) % 1000 = 644 := by
  sorry

end smallest_square_ending_644_l1466_146666


namespace scientific_notation_216000_l1466_146641

theorem scientific_notation_216000 : 216000 = 2.16 * (10 ^ 5) := by
  sorry

end scientific_notation_216000_l1466_146641


namespace sum_of_roots_of_equation_l1466_146619

theorem sum_of_roots_of_equation (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ - 3)^2 = 16 ∧ (x₂ - 3)^2 = 16 ∧ x₁ + x₂ = 6) := by
  sorry

end sum_of_roots_of_equation_l1466_146619


namespace equal_even_odd_probability_l1466_146630

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling 8 six-sided dice and getting an equal number of even and odd results -/
def prob_equal_even_odd : ℚ := 35/128

theorem equal_even_odd_probability :
  (Nat.choose num_dice (num_dice / 2)) * (prob_even ^ num_dice) = prob_equal_even_odd := by
  sorry

end equal_even_odd_probability_l1466_146630


namespace sine_equality_l1466_146693

theorem sine_equality (α β γ τ : ℝ) 
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0 ∧ τ > 0) 
  (h_eq : ∀ x, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (τ * x)) : 
  α = γ ∨ α = τ := by
sorry

end sine_equality_l1466_146693


namespace train_length_l1466_146601

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length 
  (speed : ℝ) 
  (time_to_cross : ℝ) 
  (bridge_length : ℝ) 
  (h1 : speed = 45) -- km/hr
  (h2 : time_to_cross = 30 / 3600) -- 30 seconds converted to hours
  (h3 : bridge_length = 265) -- meters
  : ∃ (train_length : ℝ), train_length = 110 := by
  sorry

end train_length_l1466_146601


namespace multiply_82519_9999_l1466_146689

theorem multiply_82519_9999 : 82519 * 9999 = 825107481 := by
  sorry

end multiply_82519_9999_l1466_146689


namespace probability_highest_is_four_value_l1466_146636

def number_of_balls : ℕ := 5
def balls_drawn : ℕ := 3

def probability_highest_is_four : ℚ :=
  (Nat.choose (number_of_balls - 2) (balls_drawn - 1)) / (Nat.choose number_of_balls balls_drawn)

theorem probability_highest_is_four_value : 
  probability_highest_is_four = 3 / 10 := by
  sorry

end probability_highest_is_four_value_l1466_146636


namespace pure_imaginary_complex_number_l1466_146625

theorem pure_imaginary_complex_number (a : ℝ) : 
  (a^3 - a = 0 ∧ a/(1-a) ≠ 0) → a = -1 :=
sorry

end pure_imaginary_complex_number_l1466_146625


namespace chord_length_is_six_l1466_146638

/-- A circle with equation x^2 + y^2 + 8x - 10y + 41 = r^2 that is tangent to the x-axis --/
structure TangentCircle where
  r : ℝ
  tangent_to_x_axis : r = 5

/-- The length of the chord intercepted by the circle on the y-axis --/
def chord_length (c : TangentCircle) : ℝ :=
  let y₁ := 2
  let y₂ := 8
  |y₁ - y₂|

/-- Theorem stating that the length of the chord intercepted by the circle on the y-axis is 6 --/
theorem chord_length_is_six (c : TangentCircle) : chord_length c = 6 := by
  sorry

end chord_length_is_six_l1466_146638


namespace balance_balls_l1466_146634

/-- Given balance conditions between different colored balls, prove the number of blue balls needed to balance a specific combination. -/
theorem balance_balls (red blue orange purple : ℚ) 
  (h1 : 4 * red = 8 * blue)
  (h2 : 3 * orange = 7 * blue)
  (h3 : 8 * blue = 6 * purple) :
  5 * red + 3 * orange + 4 * purple = 67/3 * blue := by
  sorry

end balance_balls_l1466_146634


namespace cube_sum_is_26_l1466_146692

/-- Properties of a cube -/
structure Cube where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- Definition of a standard cube -/
def standardCube : Cube :=
  { faces := 6
  , edges := 12
  , vertices := 8 }

/-- Theorem: The sum of faces, edges, and vertices of a cube is 26 -/
theorem cube_sum_is_26 (c : Cube) (h : c = standardCube) : 
  c.faces + c.edges + c.vertices = 26 := by
  sorry

end cube_sum_is_26_l1466_146692


namespace min_value_of_function_l1466_146614

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  4 * x + 25 / x ≥ 20 ∧ ∃ y > 0, 4 * y + 25 / y = 20 := by
  sorry

end min_value_of_function_l1466_146614


namespace two_digit_number_concatenation_l1466_146660

/-- A two-digit number is an integer between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem two_digit_number_concatenation (x y : ℕ) 
  (hx : TwoDigitNumber x) (hy : TwoDigitNumber y) :
  ∃ (n : ℕ), n = 100 * x + y ∧ 1000 ≤ n ∧ n ≤ 9999 :=
by sorry

end two_digit_number_concatenation_l1466_146660


namespace sqrt_14_between_3_and_4_l1466_146646

theorem sqrt_14_between_3_and_4 : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end sqrt_14_between_3_and_4_l1466_146646


namespace total_income_is_53_l1466_146661

def tshirt_price : ℕ := 5
def pants_price : ℕ := 4
def skirt_price : ℕ := 6
def refurbished_tshirt_price : ℕ := tshirt_price / 2

def tshirts_sold : ℕ := 2
def pants_sold : ℕ := 1
def skirts_sold : ℕ := 4
def refurbished_tshirts_sold : ℕ := 6

def total_income : ℕ := 
  tshirts_sold * tshirt_price + 
  pants_sold * pants_price + 
  skirts_sold * skirt_price + 
  refurbished_tshirts_sold * refurbished_tshirt_price

theorem total_income_is_53 : total_income = 53 := by
  sorry

end total_income_is_53_l1466_146661


namespace coffee_lasts_40_days_l1466_146628

/-- The number of days coffee will last given the amount bought, brewing capacity, and daily consumption. -/
def coffee_duration (pounds_bought : ℕ) (cups_per_pound : ℕ) (cups_per_day : ℕ) : ℕ :=
  (pounds_bought * cups_per_pound) / cups_per_day

/-- Theorem stating that under the given conditions, the coffee will last 40 days. -/
theorem coffee_lasts_40_days :
  coffee_duration 3 40 3 = 40 := by
  sorry

end coffee_lasts_40_days_l1466_146628


namespace min_sum_squares_addends_of_18_l1466_146657

theorem min_sum_squares_addends_of_18 :
  ∀ x y : ℝ, x + y = 18 → x^2 + y^2 ≥ 2 * 9^2 :=
by sorry

end min_sum_squares_addends_of_18_l1466_146657


namespace equation_system_has_real_solution_l1466_146655

theorem equation_system_has_real_solution (x y : ℝ) 
  (h1 : 1 ≤ Real.sqrt x) (h2 : Real.sqrt x ≤ y) (h3 : y ≤ x^2) :
  ∃ (a b c : ℝ),
    a + b + c = (x + x^2 + x^4 + y + y^2 + y^4) / 2 ∧
    a * b + a * c + b * c = (x^3 + x^5 + x^6 + y^3 + y^5 + y^6) / 2 ∧
    a * b * c = (x^7 + y^7) / 2 := by
  sorry

end equation_system_has_real_solution_l1466_146655


namespace ellipse_foci_distance_l1466_146640

/-- The distance between the foci of the ellipse x^2/36 + y^2/9 = 5 is 6√15 -/
theorem ellipse_foci_distance :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/36 + y^2/9 = 5}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ 
    ∀ p ∈ ellipse, ‖p - f₁‖ + ‖p - f₂‖ = 2 * Real.sqrt (180 : ℝ) ∧
    ‖f₁ - f₂‖ = 6 * Real.sqrt 15 :=
by sorry

end ellipse_foci_distance_l1466_146640


namespace class_attendance_multiple_l1466_146639

/-- Proves that the largest whole number multiple of students present yesterday
    that is less than or equal to 90% of students attending today is 0. -/
theorem class_attendance_multiple (total_registered : ℕ) (present_yesterday : ℕ) (absent_today : ℕ)
  (h1 : total_registered = 156)
  (h2 : present_yesterday = 70)
  (h3 : absent_today = 30)
  (h4 : present_yesterday > absent_today) :
  (∀ n : ℕ, n * present_yesterday ≤ (present_yesterday - absent_today) * 9 / 10 → n = 0) :=
by sorry

end class_attendance_multiple_l1466_146639


namespace fraction_equality_l1466_146691

theorem fraction_equality (x y : ℚ) (h : y ≠ 0) (h1 : x / y = 7 / 2) :
  (x - 2 * y) / y = 3 / 2 := by sorry

end fraction_equality_l1466_146691


namespace towel_area_decrease_l1466_146607

theorem towel_area_decrease (length width : ℝ) (h1 : length > 0) (h2 : width > 0) :
  let new_length := 0.9 * length
  let new_width := 0.8 * width
  let original_area := length * width
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.28 := by
sorry

end towel_area_decrease_l1466_146607


namespace max_odd_group_length_l1466_146699

/-- A sequence of consecutive natural numbers where each number
    is a product of prime factors with odd exponents -/
def OddGroup (start : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → 
    ∀ p : ℕ, Nat.Prime p → 
      Odd ((start + k).factorization p)

/-- The maximum length of an OddGroup -/
theorem max_odd_group_length : 
  (∃ (max : ℕ), ∀ (start n : ℕ), OddGroup start n → n ≤ max) ∧ 
  (∃ (start : ℕ), OddGroup start 7) :=
sorry

end max_odd_group_length_l1466_146699


namespace tank_capacity_l1466_146615

theorem tank_capacity : 
  ∀ (initial_fraction final_fraction added_amount total_capacity : ℚ),
  initial_fraction = 1/4 →
  final_fraction = 2/3 →
  added_amount = 160 →
  (final_fraction - initial_fraction) * total_capacity = added_amount →
  total_capacity = 384 := by
sorry

end tank_capacity_l1466_146615


namespace positive_y_solution_l1466_146690

theorem positive_y_solution (x y z : ℝ) 
  (eq1 : x * y = 8 - 3 * x - 2 * y)
  (eq2 : y * z = 15 - 5 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 4 * z)
  (y_pos : y > 0) :
  y = 4 := by
sorry

end positive_y_solution_l1466_146690


namespace distance_to_focus_l1466_146679

/-- Given a parabola x = (1/2)y², prove that the distance from a point P(1, y) on the parabola to its focus F is 3/2 -/
theorem distance_to_focus (y : ℝ) (h : 1 = (1/2) * y^2) : 
  let p : ℝ × ℝ := (1, y)
  let f : ℝ × ℝ := ((1/4), 0)  -- Focus of the parabola x = (1/2)y²
  ‖p - f‖ = 3/2 := by sorry

end distance_to_focus_l1466_146679


namespace sum_of_15th_set_l1466_146631

/-- Represents the sum of elements in the nth set of a sequence of sets,
    where each set contains consecutive integers and has one more element than the previous set. -/
def S (n : ℕ) : ℕ :=
  let first := (n * (n - 1)) / 2 + 1
  let last := first + n - 1
  n * (first + last) / 2

/-- The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end sum_of_15th_set_l1466_146631


namespace stephanie_oranges_l1466_146673

/-- Represents the number of store visits -/
def store_visits : ℕ := 8

/-- Represents the total number of oranges bought -/
def total_oranges : ℕ := 16

/-- Represents the number of oranges bought per visit -/
def oranges_per_visit : ℕ := total_oranges / store_visits

/-- Theorem stating that Stephanie buys 2 oranges each time she goes to the store -/
theorem stephanie_oranges : oranges_per_visit = 2 := by
  sorry

end stephanie_oranges_l1466_146673
