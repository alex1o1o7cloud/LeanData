import Mathlib

namespace sams_test_score_l975_97593

theorem sams_test_score (initial_students : ℕ) (initial_average : ℚ) (new_average : ℚ) 
  (h1 : initial_students = 19)
  (h2 : initial_average = 85)
  (h3 : new_average = 86) :
  (initial_students + 1) * new_average - initial_students * initial_average = 105 :=
by sorry

end sams_test_score_l975_97593


namespace jessica_jelly_bean_guess_l975_97509

/-- Represents the number of jelly beans of each color in a bag -/
structure JellyBeanBag where
  red : ℕ
  black : ℕ
  green : ℕ
  purple : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the number of bags needed to fill the fishbowl -/
def bagsNeeded (bag : JellyBeanBag) (guessRedWhite : ℕ) : ℕ :=
  guessRedWhite / (bag.red + bag.white)

theorem jessica_jelly_bean_guess 
  (bag : JellyBeanBag)
  (guessRedWhite : ℕ)
  (h1 : bag.red = 24)
  (h2 : bag.black = 13)
  (h3 : bag.green = 36)
  (h4 : bag.purple = 28)
  (h5 : bag.yellow = 32)
  (h6 : bag.white = 18)
  (h7 : guessRedWhite = 126) :
  bagsNeeded bag guessRedWhite = 3 := by
  sorry

end jessica_jelly_bean_guess_l975_97509


namespace cheese_problem_l975_97549

theorem cheese_problem (k : ℕ) (h1 : k > 7) : ∃ (initial : ℕ), initial = 11 ∧ 
  (10 : ℚ) / k + 7 * ((5 : ℚ) / k) = initial ∧ (35 : ℕ) % k = 0 := by
  sorry

end cheese_problem_l975_97549


namespace triangle_angles_calculation_l975_97531

-- Define the triangle and its properties
def Triangle (A B C : ℝ) (C_ext : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = 180 ∧
  C_ext = A + B

-- Theorem statement
theorem triangle_angles_calculation 
  (A B C C_ext : ℝ) 
  (h : Triangle A B C C_ext) 
  (hA : A = 64) 
  (hB : B = 33) 
  (hC_ext : C_ext = 120) :
  C = 83 ∧ ∃ D, D = 56 ∧ C_ext = A + D :=
by sorry

end triangle_angles_calculation_l975_97531


namespace right_triangle_hypotenuse_l975_97551

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → c = 13 := by
sorry

end right_triangle_hypotenuse_l975_97551


namespace factorial_ratio_l975_97590

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 47 = 117600 := by sorry

end factorial_ratio_l975_97590


namespace slope_of_line_l975_97515

/-- The slope of the line 6x + 10y = 30 is -3/5 -/
theorem slope_of_line (x y : ℝ) : 6 * x + 10 * y = 30 → (y - 3 = (-3 / 5) * (x - 0)) := by
  sorry

end slope_of_line_l975_97515


namespace third_month_sale_l975_97532

def sale_problem (m1 m2 m4 m5 m6 avg : ℕ) : Prop :=
  let total := avg * 6
  let known_sum := m1 + m2 + m4 + m5 + m6
  total - known_sum = 6200

theorem third_month_sale :
  sale_problem 5420 5660 6350 6500 6470 6100 :=
sorry

end third_month_sale_l975_97532


namespace amount_spent_on_sweets_l975_97545

-- Define the initial amount
def initial_amount : ℚ := 5.10

-- Define the amount given to each friend
def amount_per_friend : ℚ := 1.00

-- Define the number of friends
def number_of_friends : ℕ := 2

-- Define the final amount left
def final_amount : ℚ := 2.05

-- Theorem to prove the amount spent on sweets
theorem amount_spent_on_sweets :
  initial_amount - (amount_per_friend * number_of_friends) - final_amount = 1.05 := by
  sorry

end amount_spent_on_sweets_l975_97545


namespace fixed_point_exponential_function_l975_97513

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 1)
  f (-1) = 1 := by
sorry

end fixed_point_exponential_function_l975_97513


namespace red_shirt_pairs_l975_97562

theorem red_shirt_pairs (total_students : ℕ) (blue_students : ℕ) (red_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) 
  (h1 : total_students = 144)
  (h2 : blue_students = 63)
  (h3 : red_students = 81)
  (h4 : total_pairs = 72)
  (h5 : blue_blue_pairs = 29)
  (h6 : total_students = blue_students + red_students)
  (h7 : total_pairs * 2 = total_students) :
  (red_students - (total_students - blue_blue_pairs * 2 - blue_students)) / 2 = 38 := by
  sorry

end red_shirt_pairs_l975_97562


namespace angle_complement_supplement_l975_97518

theorem angle_complement_supplement (x : ℝ) : 
  (90 - x) = 3 * (180 - x) → (180 - x = 135 ∧ 90 - x = 45) := by
  sorry

end angle_complement_supplement_l975_97518


namespace tangent_line_to_parabola_l975_97583

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
axiom tangent_condition (a b c : ℝ) : 
  b^2 - 4*a*c = 0 ↔ ∃ (x y : ℝ), y^2 = 12*x ∧ y = 3*x + c

/-- If the line y = 3x + d is tangent to the parabola y^2 = 12x, then d = 1 -/
theorem tangent_line_to_parabola (d : ℝ) : 
  (∃ (x y : ℝ), y^2 = 12*x ∧ y = 3*x + d) → d = 1 := by
  sorry

#check tangent_line_to_parabola

end tangent_line_to_parabola_l975_97583


namespace cards_in_boxes_l975_97570

/-- The number of ways to distribute n distinct objects into k distinct boxes with no box left empty -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 4 cards and 3 boxes -/
def num_cards : ℕ := 4
def num_boxes : ℕ := 3

/-- The theorem to prove -/
theorem cards_in_boxes : distribute num_cards num_boxes = 36 := by sorry

end cards_in_boxes_l975_97570


namespace train_length_l975_97524

/-- Calculates the length of a train given its speed and time to pass a fixed point. -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 10 → speed * time * (1000 / 3600) = 175 := by
  sorry

end train_length_l975_97524


namespace marie_stamps_l975_97510

theorem marie_stamps (notebooks : ℕ) (stamps_per_notebook : ℕ) (binders : ℕ) (stamps_per_binder : ℕ) (stamps_given_away : ℕ) :
  notebooks = 4 →
  stamps_per_notebook = 20 →
  binders = 2 →
  stamps_per_binder = 50 →
  stamps_given_away = 135 →
  (notebooks * stamps_per_notebook + binders * stamps_per_binder - stamps_given_away : ℚ) / (notebooks * stamps_per_notebook + binders * stamps_per_binder) = 1 / 4 :=
by sorry

end marie_stamps_l975_97510


namespace stability_comparison_l975_97546

-- Define the Student type
structure Student where
  name : String
  average_score : ℝ
  variance : ℝ

-- Define the concept of stability
def more_stable (a b : Student) : Prop :=
  a.variance < b.variance

-- Theorem statement
theorem stability_comparison (A B : Student) 
  (h1 : A.average_score = B.average_score)
  (h2 : A.variance > B.variance) : 
  more_stable B A := by
  sorry

end stability_comparison_l975_97546


namespace sum_of_a_and_b_is_six_l975_97591

/-- A three-digit number in the form 2a3 -/
def number_2a3 (a : ℕ) : ℕ := 200 + 10 * a + 3

/-- A three-digit number in the form 5b9 -/
def number_5b9 (b : ℕ) : ℕ := 500 + 10 * b + 9

/-- Proposition: If 2a3 + 326 = 5b9 and 5b9 is a multiple of 9, then a + b = 6 -/
theorem sum_of_a_and_b_is_six (a b : ℕ) :
  number_2a3 a + 326 = number_5b9 b →
  (∃ k : ℕ, number_5b9 b = 9 * k) →
  a + b = 6 := by
  sorry

end sum_of_a_and_b_is_six_l975_97591


namespace exponent_multiplication_l975_97584

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l975_97584


namespace job_completion_time_l975_97550

/-- Given two workers can finish a job in 15 days and a third worker can finish the job in 30 days,
    prove that all three workers together can finish the job in 10 days. -/
theorem job_completion_time 
  (work_rate_ab : ℝ) 
  (work_rate_c : ℝ) 
  (h1 : work_rate_ab = 1 / 15) 
  (h2 : work_rate_c = 1 / 30) : 
  1 / (work_rate_ab + work_rate_c) = 10 := by
  sorry

#check job_completion_time

end job_completion_time_l975_97550


namespace prob_three_even_dice_l975_97528

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of sides on each die -/
def num_sides : ℕ := 12

/-- The number of even outcomes on each die -/
def num_even_sides : ℕ := 6

/-- The number of dice showing even numbers -/
def num_even_dice : ℕ := 3

/-- The probability of exactly three dice showing even numbers when six fair 12-sided dice are rolled -/
theorem prob_three_even_dice : 
  (num_dice.choose num_even_dice * (num_even_sides / num_sides) ^ num_even_dice * 
  ((num_sides - num_even_sides) / num_sides) ^ (num_dice - num_even_dice) : ℚ) = 5/16 := by
  sorry

end prob_three_even_dice_l975_97528


namespace number_to_billions_l975_97504

/-- Converts a number to billions -/
def to_billions (n : ℕ) : ℚ :=
  (n : ℚ) / 1000000000

theorem number_to_billions :
  to_billions 640080000 = 0.64008 := by sorry

end number_to_billions_l975_97504


namespace man_money_calculation_l975_97557

/-- Calculates the total amount of money given the number of Rs. 50 and Rs. 500 notes -/
def totalAmount (fiftyNotes : ℕ) (fiveHundredNotes : ℕ) : ℕ :=
  50 * fiftyNotes + 500 * fiveHundredNotes

theorem man_money_calculation (totalNotes : ℕ) (fiftyNotes : ℕ) 
  (h1 : totalNotes = 126)
  (h2 : fiftyNotes = 117)
  (h3 : totalNotes = fiftyNotes + (totalNotes - fiftyNotes)) :
  totalAmount fiftyNotes (totalNotes - fiftyNotes) = 10350 := by
  sorry

#eval totalAmount 117 9

end man_money_calculation_l975_97557


namespace finite_decimal_fraction_l975_97569

def is_finite_decimal (n : ℚ) : Prop :=
  ∃ (a b : ℤ), n = a / (2^b * 5^b)

theorem finite_decimal_fraction :
  (is_finite_decimal (9/12)) ∧
  (¬ is_finite_decimal (11/27)) ∧
  (¬ is_finite_decimal (4/7)) ∧
  (¬ is_finite_decimal (8/15)) :=
by sorry

end finite_decimal_fraction_l975_97569


namespace line_equation_l975_97561

/-- Given a line l: ax + by + 1 = 0 and a circle x² + y² - 6y + 5 = 0, 
    this theorem proves that the line l is x - y + 3 = 0 
    if it's the axis of symmetry of the circle and perpendicular to x + y + 2 = 0 -/
theorem line_equation (a b : ℝ) : 
  (∀ x y : ℝ, a * x + b * y + 1 = 0 → 
    (x^2 + y^2 - 6*y + 5 = 0 → 
      (∃ c : ℝ, c * (a * x + b * y + 1) = x^2 + y^2 - 6*y + 5))) → 
  (a * 1 + b * 1 = -1) → 
  (a * x + b * y + 1 = 0 ↔ x - y + 3 = 0) := by
  sorry

end line_equation_l975_97561


namespace leo_current_weight_l975_97592

def leo_weight_problem (leo_weight kendra_weight : ℝ) : Prop :=
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 150)

theorem leo_current_weight :
  ∃ (leo_weight kendra_weight : ℝ),
    leo_weight_problem leo_weight kendra_weight ∧
    leo_weight = 86 := by
  sorry

end leo_current_weight_l975_97592


namespace b_used_car_for_10_hours_l975_97500

/-- Represents the car hire scenario -/
structure CarHire where
  totalCost : ℕ
  aHours : ℕ
  cHours : ℕ
  bPaid : ℕ

/-- Calculates the number of hours b used the car -/
def calculateBHours (ch : CarHire) : ℕ :=
  let totalHours := ch.aHours + ch.cHours + (ch.bPaid * (ch.aHours + ch.cHours) / (ch.totalCost - ch.bPaid))
  ch.bPaid * totalHours / ch.totalCost

/-- Theorem stating that given the conditions, b used the car for 10 hours -/
theorem b_used_car_for_10_hours (ch : CarHire)
  (h1 : ch.totalCost = 720)
  (h2 : ch.aHours = 9)
  (h3 : ch.cHours = 13)
  (h4 : ch.bPaid = 225) :
  calculateBHours ch = 10 := by
  sorry

#eval calculateBHours ⟨720, 9, 13, 225⟩

end b_used_car_for_10_hours_l975_97500


namespace students_in_neither_art_nor_music_l975_97537

theorem students_in_neither_art_nor_music
  (total : ℕ)
  (art : ℕ)
  (music : ℕ)
  (both : ℕ)
  (h1 : total = 60)
  (h2 : art = 40)
  (h3 : music = 30)
  (h4 : both = 15) :
  total - (art + music - both) = 5 := by
  sorry

end students_in_neither_art_nor_music_l975_97537


namespace intersection_of_A_and_B_l975_97521

-- Define the sets A and B
def A : Set ℝ := {x | Real.log x ≥ 0}
def B : Set ℝ := {x | x^2 < 9}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 1 3 := by
  sorry

end intersection_of_A_and_B_l975_97521


namespace percentage_failed_both_l975_97534

/-- The percentage of students who failed in Hindi -/
def failed_hindi : ℝ := 25

/-- The percentage of students who failed in English -/
def failed_english : ℝ := 35

/-- The percentage of students who passed in both subjects -/
def passed_both : ℝ := 80

/-- The theorem stating the percentage of students who failed in both subjects -/
theorem percentage_failed_both :
  100 - passed_both = failed_hindi + failed_english - 40 := by sorry

end percentage_failed_both_l975_97534


namespace carol_carrot_count_l975_97536

/-- The number of carrots Carol picked -/
def carols_carrots : ℝ := 29.0

/-- The number of carrots Carol's mom picked -/
def moms_carrots : ℝ := 16.0

/-- The number of carrots they picked together -/
def joint_carrots : ℝ := 38.0

/-- The total number of carrots picked -/
def total_carrots : ℝ := 83.0

theorem carol_carrot_count : 
  carols_carrots + moms_carrots + joint_carrots = total_carrots :=
by sorry

end carol_carrot_count_l975_97536


namespace C_symmetric_C_area_inequality_C_perimeter_inequality_l975_97556

-- Define the curve C
def C (a : ℝ) (P : ℝ × ℝ) : Prop :=
  a > 1 ∧ (Real.sqrt ((P.1 + 1)^2 + P.2^2) * Real.sqrt ((P.1 - 1)^2 + P.2^2) = a^2)

-- Define the fixed points
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Theorem for symmetry
theorem C_symmetric (a : ℝ) :
  ∀ P : ℝ × ℝ, C a P ↔ C a (-P.1, -P.2) := by sorry

-- Theorem for area inequality
theorem C_area_inequality (a : ℝ) :
  ∀ P : ℝ × ℝ, C a P → 
    (1/2 * Real.sqrt ((P.1 + 1)^2 + P.2^2) * Real.sqrt ((P.1 - 1)^2 + P.2^2) * 
      Real.sin (Real.arccos ((P.1 + 1) * (P.1 - 1) + P.2^2) / 
        (Real.sqrt ((P.1 + 1)^2 + P.2^2) * Real.sqrt ((P.1 - 1)^2 + P.2^2))))
    ≤ (1/2) * a^2 := by sorry

-- Theorem for perimeter inequality
theorem C_perimeter_inequality (a : ℝ) :
  ∀ P : ℝ × ℝ, C a P → 
    Real.sqrt ((P.1 + 1)^2 + P.2^2) + Real.sqrt ((P.1 - 1)^2 + P.2^2) + 2 ≥ 2*a + 2 := by sorry

end C_symmetric_C_area_inequality_C_perimeter_inequality_l975_97556


namespace five_digit_multiple_of_6_l975_97597

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def digit_sum (n : ℕ) : ℕ := 
  let digits := n.digits 10
  digits.sum

theorem five_digit_multiple_of_6 (n : ℕ) : 
  (∃ d : ℕ, n = 84370 + d ∧ d < 10) → 
  is_multiple_of_6 n → 
  (n.mod 10 = 2 ∨ n.mod 10 = 8) :=
sorry

end five_digit_multiple_of_6_l975_97597


namespace concert_songs_count_l975_97585

/-- Calculates the number of songs in a concert given the total duration,
    intermission time, regular song duration, and special song duration. -/
def number_of_songs (total_duration intermission_time regular_song_duration special_song_duration : ℕ) : ℕ :=
  let singing_time := total_duration - intermission_time
  let regular_songs_time := singing_time - special_song_duration
  (regular_songs_time / regular_song_duration) + 1

/-- Theorem stating that the number of songs in the given concert is 13. -/
theorem concert_songs_count :
  number_of_songs 80 10 5 10 = 13 := by
  sorry

end concert_songs_count_l975_97585


namespace min_value_and_inequality_l975_97501

def f (x : ℝ) := 2 * abs (x + 1) + abs (x - 2)

theorem min_value_and_inequality :
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ y : ℝ, f y = m) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 →
    b^2 / a + c^2 / b + a^2 / c ≥ 3) := by
  sorry

end min_value_and_inequality_l975_97501


namespace polynomial_simplification_l975_97548

theorem polynomial_simplification (p : ℝ) : 
  (5 * p^4 - 4 * p^3 + 3 * p - 7) + (8 - 9 * p^2 + p^4 + 6 * p) = 
  6 * p^4 - 4 * p^3 - 9 * p^2 + 9 * p + 1 := by sorry

end polynomial_simplification_l975_97548


namespace product_sum_fractions_l975_97552

theorem product_sum_fractions : (3 * 4 * 5) * (1 / 3 + 1 / 4 - 1 / 5) = 23 := by sorry

end product_sum_fractions_l975_97552


namespace tangent_points_distance_circle_fixed_point_l975_97522

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Define a point on the directrix
structure PointOnDirectrix where
  x : ℝ
  y : ℝ
  on_directrix : directrix x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the tangent line from a point on the directrix to the parabola
def tangent_line (P : PointOnDirectrix) (Q : PointOnParabola) : Prop :=
  ∃ k : ℝ, Q.y = k * (Q.x - P.x)

-- Theorem 1: Distance between tangent points when P is on x-axis
theorem tangent_points_distance :
  ∀ (P : PointOnDirectrix) (Q R : PointOnParabola),
  P.y = 0 →
  tangent_line P Q →
  tangent_line P R →
  Q ≠ R →
  (Q.x - R.x)^2 + (Q.y - R.y)^2 = 16 :=
sorry

-- Theorem 2: Circle with diameter PQ passes through (1, 0)
theorem circle_fixed_point :
  ∀ (P : PointOnDirectrix) (Q : PointOnParabola),
  tangent_line P Q →
  ∃ (r : ℝ),
    (1 - ((P.x + Q.x) / 2))^2 + (0 - ((P.y + Q.y) / 2))^2 = r^2 ∧
    (P.x - ((P.x + Q.x) / 2))^2 + (P.y - ((P.y + Q.y) / 2))^2 = r^2 :=
sorry

end tangent_points_distance_circle_fixed_point_l975_97522


namespace degenerate_ellipse_l975_97512

/-- An ellipse is degenerate if and only if it consists of a single point -/
theorem degenerate_ellipse (x y c : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * (p.1)^2 + (p.2)^2 + 6 * p.1 - 12 * p.2 + c = 0) ↔ c = -39 := by
  sorry

end degenerate_ellipse_l975_97512


namespace coordinate_points_existence_l975_97520

theorem coordinate_points_existence :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (a₁ - b₁ = 4 ∧ a₁^2 + b₁^2 = 30 ∧ a₁ * b₁ = c₁) ∧
    (a₂ - b₂ = 4 ∧ a₂^2 + b₂^2 = 30 ∧ a₂ * b₂ = c₂) ∧
    a₁ = 2 + Real.sqrt 11 ∧
    b₁ = -2 + Real.sqrt 11 ∧
    c₁ = -15 + 4 * Real.sqrt 11 ∧
    a₂ = 2 - Real.sqrt 11 ∧
    b₂ = -2 - Real.sqrt 11 ∧
    c₂ = -15 - 4 * Real.sqrt 11 :=
by sorry

end coordinate_points_existence_l975_97520


namespace prob_sum_seven_l975_97502

/-- A type representing the possible outcomes of a single dice throw -/
inductive DiceOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- The type of outcomes when throwing a dice twice -/
def TwoThrows := DiceOutcome × DiceOutcome

/-- The sum of points for a pair of dice outcomes -/
def sum_points (throw : TwoThrows) : Nat :=
  match throw with
  | (DiceOutcome.one, b) => 1 + DiceOutcome.toNat b
  | (DiceOutcome.two, b) => 2 + DiceOutcome.toNat b
  | (DiceOutcome.three, b) => 3 + DiceOutcome.toNat b
  | (DiceOutcome.four, b) => 4 + DiceOutcome.toNat b
  | (DiceOutcome.five, b) => 5 + DiceOutcome.toNat b
  | (DiceOutcome.six, b) => 6 + DiceOutcome.toNat b
where
  DiceOutcome.toNat : DiceOutcome → Nat
    | DiceOutcome.one => 1
    | DiceOutcome.two => 2
    | DiceOutcome.three => 3
    | DiceOutcome.four => 4
    | DiceOutcome.five => 5
    | DiceOutcome.six => 6

/-- The set of all possible outcomes when throwing a dice twice -/
def all_outcomes : Finset TwoThrows := sorry

/-- The set of outcomes where the sum of points is 7 -/
def sum_seven_outcomes : Finset TwoThrows := sorry

theorem prob_sum_seven : 
  (Finset.card sum_seven_outcomes : ℚ) / (Finset.card all_outcomes : ℚ) = 1 / 6 := by sorry

end prob_sum_seven_l975_97502


namespace quadratic_negative_root_condition_l975_97511

/-- The quadratic equation ax^2 + 2x + 1 = 0 -/
def quadratic_equation (a : ℝ) (x : ℝ) : Prop := a * x^2 + 2 * x + 1 = 0

/-- A root of the quadratic equation is negative -/
def has_negative_root (a : ℝ) : Prop := ∃ x : ℝ, x < 0 ∧ quadratic_equation a x

theorem quadratic_negative_root_condition :
  (∀ a : ℝ, a < 0 → has_negative_root a) ∧
  (∃ a : ℝ, a ≥ 0 ∧ has_negative_root a) :=
sorry

end quadratic_negative_root_condition_l975_97511


namespace afternoon_rowing_count_l975_97527

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 35

/-- The total number of campers who went rowing -/
def total_campers : ℕ := 62

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := total_campers - morning_campers

theorem afternoon_rowing_count : afternoon_campers = 27 := by
  sorry

end afternoon_rowing_count_l975_97527


namespace digit_difference_when_reversed_l975_97581

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The value of a 3-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed value of a 3-digit number -/
def ThreeDigitNumber.reversed_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

theorem digit_difference_when_reversed (n : ThreeDigitNumber) 
  (h : (n.reversed_value - n.value : ℚ) / 10 = 19.8) : 
  n.hundreds - n.units = 2 := by
  sorry

end digit_difference_when_reversed_l975_97581


namespace projection_result_l975_97567

/-- A projection that takes [2, -4] to [3, -3] -/
def projection (v : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The projection satisfies the given condition -/
axiom projection_condition : projection (2, -4) = (3, -3)

/-- Theorem: The projection takes [3, 5] to [-1, 1] -/
theorem projection_result : projection (3, 5) = (-1, 1) := by sorry

end projection_result_l975_97567


namespace grade_distribution_l975_97560

theorem grade_distribution (n : Nat) (k : Nat) : 
  n = 12 ∧ k = 3 → 
  (k^n : Nat) - k * ((k-1)^n : Nat) + (k * (k-2)^n : Nat) = 519156 := by
  sorry

end grade_distribution_l975_97560


namespace two_numbers_difference_l975_97578

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 23210 →
  b % 5 = 0 →
  a = 2 * (b / 10) →
  b - a = 15480 :=
by sorry

end two_numbers_difference_l975_97578


namespace candidate_count_l975_97594

theorem candidate_count (total : ℕ) : 
  (total * 6 / 100 : ℕ) + 84 = total * 7 / 100 → total = 8400 := by
  sorry

end candidate_count_l975_97594


namespace sum_even_probability_l975_97543

/-- Represents a wheel in the game -/
structure Wheel where
  probability_even : ℝ

/-- The game with two wheels -/
structure Game where
  wheel_a : Wheel
  wheel_b : Wheel

/-- A fair wheel has equal probability of landing on even or odd numbers -/
def is_fair (w : Wheel) : Prop := w.probability_even = 1/2

/-- Probability of the sum of two wheels being even -/
def prob_sum_even (g : Game) : ℝ :=
  g.wheel_a.probability_even * g.wheel_b.probability_even +
  (1 - g.wheel_a.probability_even) * (1 - g.wheel_b.probability_even)

theorem sum_even_probability (g : Game) 
  (h1 : is_fair g.wheel_a) 
  (h2 : g.wheel_b.probability_even = 2/3) : 
  prob_sum_even g = 2/3 := by
  sorry

end sum_even_probability_l975_97543


namespace smallest_fraction_between_l975_97529

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 8 → q ≤ q') →
  q - p = 5 := by
sorry

end smallest_fraction_between_l975_97529


namespace union_of_sets_l975_97503

def A (a b : ℝ) : Set ℝ := {5, b/a, a-b}
def B (a b : ℝ) : Set ℝ := {b, a+b, -1}

theorem union_of_sets (a b : ℝ) (h : A a b ∩ B a b = {2, -1}) :
  A a b ∪ B a b = {-1, 2, 3, 5} := by
  sorry

end union_of_sets_l975_97503


namespace max_sqrt_sum_l975_97533

theorem max_sqrt_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 18) :
  ∃ (d : ℝ), d = 6 ∧ ∀ (a b : ℝ), a ≥ 0 → b ≥ 0 → a + b = 18 → Real.sqrt a + Real.sqrt b ≤ d :=
by sorry

end max_sqrt_sum_l975_97533


namespace medical_staff_composition_l975_97516

theorem medical_staff_composition :
  ∀ (a b c d : ℕ),
    a + b + c + d = 17 →
    a + b ≥ c + d →
    d > a →
    a > b →
    c ≥ 2 →
    a = 5 ∧ b = 4 ∧ c = 2 ∧ d = 6 :=
by sorry

end medical_staff_composition_l975_97516


namespace hyperbola_eccentricity_l975_97579

theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ 
                x = a^2 / c ∧ 
                y = -a * b / c ∧
                ∃ (x_f1 y_f1 : ℝ), (x_f1 = -c ∧ y_f1 = 0) ∧
                                   (x + x_f1) / 2 = a^2 / c ∧
                                   (y + y_f1) / 2 = -a * b / c) →
  c / a = Real.sqrt 5 :=
by sorry

end hyperbola_eccentricity_l975_97579


namespace undeveloped_land_area_l975_97508

theorem undeveloped_land_area (total_area : ℝ) (num_sections : ℕ) 
  (h1 : total_area = 7305)
  (h2 : num_sections = 3) :
  total_area / num_sections = 2435 := by
  sorry

end undeveloped_land_area_l975_97508


namespace curve_c_properties_l975_97573

/-- Curve C defined by mx^2 - ny^2 = 1 -/
structure CurveC (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 - n * y^2 = 1

/-- Definition of a hyperbola -/
def is_hyperbola (C : CurveC m n) : Prop := sorry

/-- Definition of an ellipse with foci on the x-axis -/
def is_ellipse_x_foci (C : CurveC m n) : Prop := sorry

/-- Definition of a circle -/
def is_circle (C : CurveC m n) : Prop := sorry

/-- Definition of two straight lines -/
def is_two_straight_lines (C : CurveC m n) : Prop := sorry

theorem curve_c_properties (m n : ℝ) (C : CurveC m n) :
  (m * n > 0 → is_hyperbola C) ∧
  (m > 0 ∧ m + n < 0 → is_ellipse_x_foci C) ∧
  (¬(m > 0 ∧ n < 0 → ¬is_circle C)) ∧
  (m > 0 ∧ n = 0 → is_two_straight_lines C) := by
  sorry

end curve_c_properties_l975_97573


namespace simplify_expression_l975_97535

theorem simplify_expression (m : ℝ) (h : m > 0) : 
  (Real.sqrt m * 3 * m) / ((6 * m) ^ 5) = 1 := by sorry

end simplify_expression_l975_97535


namespace complex_fraction_equality_l975_97505

theorem complex_fraction_equality : (1 - Complex.I) / (1 + Complex.I) = -Complex.I := by
  sorry

end complex_fraction_equality_l975_97505


namespace december_spending_fraction_l975_97587

def monthly_savings (month : Nat) : Rat :=
  match month with
  | 1 => 1/10
  | 2 => 3/25
  | 3 => 3/20
  | 4 => 1/5
  | m => if m ≤ 12 then (14 + m)/100 else 0

def total_savings : Rat :=
  (List.range 12).map (λ m => monthly_savings (m + 1)) |> List.sum

theorem december_spending_fraction :
  total_savings = 4 * (1 - monthly_savings 12) →
  1 - monthly_savings 12 = 39/50 := by
  sorry

end december_spending_fraction_l975_97587


namespace sequence_must_be_finite_l975_97530

def is_valid_sequence (c : ℕ+) (p : ℕ → ℕ) : Prop :=
  ∀ k, k ≥ 1 →
    Nat.Prime (p k) ∧
    (p (k + 1)) ∣ (p k + c) ∧
    ∀ i, 1 ≤ i ∧ i < k + 1 → p (k + 1) ≠ p i

theorem sequence_must_be_finite (c : ℕ+) :
  ¬∃ p : ℕ → ℕ, is_valid_sequence c p ∧ (∀ n, ∃ k > n, p k ≠ 0) :=
sorry

end sequence_must_be_finite_l975_97530


namespace total_photos_taken_is_46_l975_97517

/-- Represents the number of photos on Toby's camera roll at different stages --/
structure PhotoCount where
  initial : Nat
  deletedBadShots : Nat
  catPictures : Nat
  deletedAfterEditing : Nat
  additionalShots : Nat
  secondSession : Nat
  thirdSession : Nat
  deletedFromSecond : Nat
  deletedFromThird : Nat
  final : Nat

/-- Calculates the total number of photos taken in all photo shoots --/
def totalPhotosTaken (p : PhotoCount) : Nat :=
  let firstSessionPhotos := p.final - p.initial + p.deletedBadShots - p.catPictures + 
                            p.deletedAfterEditing - p.additionalShots - 
                            (p.secondSession - p.deletedFromSecond) - 
                            (p.thirdSession - p.deletedFromThird)
  firstSessionPhotos + p.secondSession + p.thirdSession

/-- Theorem stating that the total number of photos taken in all photo shoots is 46 --/
theorem total_photos_taken_is_46 (p : PhotoCount) 
  (h1 : p.initial = 63)
  (h2 : p.deletedBadShots = 7)
  (h3 : p.catPictures = 15)
  (h4 : p.deletedAfterEditing = 3)
  (h5 : p.additionalShots = 5)
  (h6 : p.secondSession = 11)
  (h7 : p.thirdSession = 8)
  (h8 : p.deletedFromSecond = 6)
  (h9 : p.deletedFromThird = 4)
  (h10 : p.final = 112) :
  totalPhotosTaken p = 46 := by
  sorry

end total_photos_taken_is_46_l975_97517


namespace stormi_car_wash_l975_97564

/-- Proves the number of cars Stormi washed to save for a bicycle --/
theorem stormi_car_wash : 
  ∀ (car_wash_price lawn_mow_income bicycle_price additional_needed : ℕ),
  car_wash_price = 10 →
  lawn_mow_income = 26 →
  bicycle_price = 80 →
  additional_needed = 24 →
  (bicycle_price - additional_needed - lawn_mow_income) / car_wash_price = 3 := by
sorry

end stormi_car_wash_l975_97564


namespace a_4_equals_20_l975_97544

def sequence_a (n : ℕ) : ℕ := n^2 + n

theorem a_4_equals_20 : sequence_a 4 = 20 := by
  sorry

end a_4_equals_20_l975_97544


namespace competition_end_time_l975_97582

-- Define the start time of the competition
def start_time : Nat := 15 * 60  -- 3:00 p.m. in minutes since midnight

-- Define the duration of the competition
def duration : Nat := 875  -- in minutes

-- Define the end time of the competition
def end_time : Nat := (start_time + duration) % (24 * 60)

-- Theorem to prove
theorem competition_end_time :
  end_time = 5 * 60 + 35  -- 5:35 a.m. in minutes since midnight
  := by sorry

end competition_end_time_l975_97582


namespace cylinder_cone_volume_ratio_l975_97595

theorem cylinder_cone_volume_ratio (h r_cylinder r_cone : ℝ) 
  (h_positive : h > 0)
  (r_cylinder_positive : r_cylinder > 0)
  (r_cone_positive : r_cone > 0)
  (cross_section_equal : h * (2 * r_cylinder) = h * r_cone) :
  (π * r_cylinder^2 * h) / ((1/3) * π * r_cone^2 * h) = 3/4 := by
sorry

end cylinder_cone_volume_ratio_l975_97595


namespace teddy_material_in_tons_l975_97514

/-- The amount of fluffy foam material Teddy uses for each pillow in pounds -/
def material_per_pillow : ℝ := 5 - 3

/-- The number of pillows Teddy can make -/
def number_of_pillows : ℕ := 3000

/-- The number of pounds in a ton -/
def pounds_per_ton : ℝ := 2000

/-- The theorem stating the amount of fluffy foam material Teddy has in tons -/
theorem teddy_material_in_tons : 
  (material_per_pillow * number_of_pillows) / pounds_per_ton = 3 := by
  sorry

end teddy_material_in_tons_l975_97514


namespace average_notebooks_sold_per_day_l975_97576

theorem average_notebooks_sold_per_day 
  (total_bundles : ℕ) 
  (days : ℕ) 
  (notebooks_per_bundle : ℕ) 
  (h1 : total_bundles = 15) 
  (h2 : days = 5) 
  (h3 : notebooks_per_bundle = 40) : 
  (total_bundles * notebooks_per_bundle) / days = 120 :=
by sorry

end average_notebooks_sold_per_day_l975_97576


namespace square_area_ratio_l975_97565

theorem square_area_ratio : 
  let small_side : ℝ := 5
  let large_side : ℝ := small_side + 5
  let small_area : ℝ := small_side ^ 2
  let large_area : ℝ := large_side ^ 2
  large_area / small_area = 4 := by
sorry

end square_area_ratio_l975_97565


namespace number_from_hcf_lcm_and_other_l975_97588

theorem number_from_hcf_lcm_and_other (a b : ℕ+) : 
  Nat.gcd a b = 12 →
  Nat.lcm a b = 396 →
  b = 99 →
  a = 48 := by
sorry

end number_from_hcf_lcm_and_other_l975_97588


namespace three_correct_propositions_l975_97574

/-- Represents the type of events --/
inductive EventType
  | Certain
  | Impossible
  | Random

/-- Represents a proposition about an event --/
structure Proposition where
  statement : String
  eventType : EventType

/-- Checks if a proposition is correct --/
def isCorrectProposition (p : Proposition) : Bool :=
  match p.statement, p.eventType with
  | "Placing all three balls into two boxes, there must be one box containing more than one ball", EventType.Certain => true
  | "For some real number x, it can make x^2 < 0", EventType.Impossible => true
  | "It will rain in Guangzhou tomorrow", EventType.Certain => false
  | "Out of 100 light bulbs, there are 5 defective ones. Taking out 5 bulbs and all 5 are defective", EventType.Random => true
  | _, _ => false

/-- The list of propositions --/
def propositions : List Proposition := [
  ⟨"Placing all three balls into two boxes, there must be one box containing more than one ball", EventType.Certain⟩,
  ⟨"For some real number x, it can make x^2 < 0", EventType.Impossible⟩,
  ⟨"It will rain in Guangzhou tomorrow", EventType.Certain⟩,
  ⟨"Out of 100 light bulbs, there are 5 defective ones. Taking out 5 bulbs and all 5 are defective", EventType.Random⟩
]

theorem three_correct_propositions :
  (propositions.filter isCorrectProposition).length = 3 := by
  sorry

end three_correct_propositions_l975_97574


namespace triangle_side_length_l975_97540

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) 
  (h1 : b = 7)
  (h2 : c = 6)
  (h3 : Real.cos (B - C) = 37/40)
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h5 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h6 : A + B + C = π) :
  a = Real.sqrt 66.1 := by
sorry

end triangle_side_length_l975_97540


namespace apps_deleted_l975_97541

theorem apps_deleted (initial_apps final_apps : ℕ) (h1 : initial_apps = 12) (h2 : final_apps = 4) :
  initial_apps - final_apps = 8 := by
  sorry

end apps_deleted_l975_97541


namespace geometric_sequence_common_ratio_l975_97563

/-- A geometric sequence with a_3 = 4 and a_6 = 1/2 has a common ratio of 1/2. -/
theorem geometric_sequence_common_ratio : ∀ (a : ℕ → ℝ), 
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 3 = 4 →                                  -- Condition 1
  a 6 = 1 / 2 →                              -- Condition 2
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ∧ q = 1 / 2) := by
  sorry


end geometric_sequence_common_ratio_l975_97563


namespace min_value_of_y_l975_97507

theorem min_value_of_y (x : ℝ) (h1 : x > 3) :
  let y := x + 1 / (x - 3)
  ∀ z, y ≥ z → z ≤ 5 :=
sorry

end min_value_of_y_l975_97507


namespace joe_ball_choices_l975_97506

/-- The number of balls in the bin -/
def num_balls : ℕ := 18

/-- The number of times a ball is chosen -/
def num_choices : ℕ := 4

/-- The number of different possible lists -/
def num_lists : ℕ := num_balls ^ num_choices

theorem joe_ball_choices :
  num_lists = 104976 := by
  sorry

end joe_ball_choices_l975_97506


namespace pet_store_cats_l975_97555

theorem pet_store_cats (siamese_cats : ℕ) (sold_cats : ℕ) (remaining_cats : ℕ) (house_cats : ℕ) : 
  siamese_cats = 13 → 
  sold_cats = 10 → 
  remaining_cats = 8 → 
  siamese_cats + house_cats - sold_cats = remaining_cats → 
  house_cats = 5 := by
sorry

end pet_store_cats_l975_97555


namespace parallel_vector_sum_l975_97571

/-- Given two vectors in ℝ², prove that if one is parallel to their sum, then the first component of the second vector is 1/2. -/
theorem parallel_vector_sum (a b : ℝ × ℝ) (h : a = (1, 2)) (h' : b.2 = 1) :
  (∃ (k : ℝ), b = k • (a + b)) → b.1 = 1/2 := by
  sorry

end parallel_vector_sum_l975_97571


namespace hexagon_circumference_hexagon_circumference_proof_l975_97566

/-- The circumference of a regular hexagon with side length 5 centimeters is 30 centimeters. -/
theorem hexagon_circumference : ℝ → Prop :=
  fun side_length =>
    side_length = 5 →
    (6 : ℝ) * side_length = 30

-- The proof is omitted
theorem hexagon_circumference_proof : hexagon_circumference 5 :=
  sorry

end hexagon_circumference_hexagon_circumference_proof_l975_97566


namespace board_length_problem_l975_97580

/-- The length of a board before the final cut, given initial length, first cut, and final adjustment cut. -/
def board_length_before_final_cut (initial_length first_cut final_cut : ℕ) : ℕ :=
  initial_length - first_cut + final_cut

/-- Theorem stating that the length of the boards before the final 7 cm cut was 125 cm. -/
theorem board_length_problem :
  board_length_before_final_cut 143 25 7 = 125 := by
  sorry

end board_length_problem_l975_97580


namespace only_statement1_is_true_l975_97596

-- Define the statements
def statement1 (a x y : ℝ) : Prop := a * (x - y) = a * x - a * y
def statement2 (a x y : ℝ) : Prop := a^(x - y) = a^x - a^y
def statement3 (x y : ℝ) : Prop := x > y ∧ y > 0 → Real.log (x - y) = Real.log x - Real.log y
def statement4 (x y : ℝ) : Prop := x > 0 ∧ y > 0 → Real.log x / Real.log y = Real.log x - Real.log y
def statement5 (a x y : ℝ) : Prop := a * (x * y) = (a * x) * (a * y)

-- Theorem stating that only statement1 is true among all statements
theorem only_statement1_is_true :
  (∀ a x y : ℝ, statement1 a x y) ∧
  (∃ a x y : ℝ, ¬ statement2 a x y) ∧
  (∃ x y : ℝ, ¬ statement3 x y) ∧
  (∃ x y : ℝ, ¬ statement4 x y) ∧
  (∃ a x y : ℝ, ¬ statement5 a x y) :=
sorry

end only_statement1_is_true_l975_97596


namespace sample_size_is_ten_l975_97553

/-- A structure representing a quality inspection scenario -/
structure QualityInspection where
  total_products : ℕ
  selected_products : ℕ

/-- Definition of sample size for a quality inspection -/
def sample_size (qi : QualityInspection) : ℕ := qi.selected_products

/-- Theorem stating that for the given scenario, the sample size is 10 -/
theorem sample_size_is_ten (qi : QualityInspection) 
  (h1 : qi.total_products = 80) 
  (h2 : qi.selected_products = 10) : 
  sample_size qi = 10 := by
  sorry

end sample_size_is_ten_l975_97553


namespace largest_package_size_l975_97554

theorem largest_package_size (anna beatrice carlos : ℕ) 
  (h1 : anna = 60) (h2 : beatrice = 45) (h3 : carlos = 75) :
  Nat.gcd anna (Nat.gcd beatrice carlos) = 15 := by
  sorry

end largest_package_size_l975_97554


namespace routes_on_grid_l975_97519

/-- The number of routes from A to B on a 3x3 grid -/
def num_routes : ℕ := 20

/-- The size of the grid -/
def grid_size : ℕ := 3

/-- The number of right moves required -/
def right_moves : ℕ := 3

/-- The number of down moves required -/
def down_moves : ℕ := 3

/-- The total number of moves required -/
def total_moves : ℕ := right_moves + down_moves

theorem routes_on_grid : 
  num_routes = (Nat.choose total_moves right_moves) := by
  sorry

end routes_on_grid_l975_97519


namespace ordering_abc_l975_97599

theorem ordering_abc (a b c : ℝ) : 
  a = 7/9 → b = 0.7 * Real.exp 0.1 → c = Real.cos (2/3) → c > a ∧ a > b :=
sorry

end ordering_abc_l975_97599


namespace larger_fraction_l975_97568

theorem larger_fraction (x y : ℚ) (sum_eq : x + y = 7/8) (prod_eq : x * y = 1/4) :
  max x y = 1/2 := by
  sorry

end larger_fraction_l975_97568


namespace deal_or_no_deal_probability_l975_97538

theorem deal_or_no_deal_probability (total_boxes : ℕ) (high_value_boxes : ℕ) (eliminated_boxes : ℕ) :
  total_boxes = 30 →
  high_value_boxes = 8 →
  eliminated_boxes = 14 →
  (high_value_boxes : ℚ) / (total_boxes - eliminated_boxes : ℚ) ≥ 1/2 :=
by sorry

end deal_or_no_deal_probability_l975_97538


namespace sqrt_six_times_sqrt_two_equals_two_sqrt_three_l975_97572

theorem sqrt_six_times_sqrt_two_equals_two_sqrt_three :
  Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_six_times_sqrt_two_equals_two_sqrt_three_l975_97572


namespace expo_arrangement_plans_l975_97525

/-- Represents the number of volunteers --/
def total_volunteers : ℕ := 5

/-- Represents the number of pavilions to be assigned --/
def pavilions_to_assign : ℕ := 3

/-- Represents the number of volunteers who cannot be assigned to a specific pavilion --/
def restricted_volunteers : ℕ := 2

/-- Represents the total number of arrangement plans --/
def total_arrangements : ℕ := 36

/-- Theorem stating the total number of arrangement plans --/
theorem expo_arrangement_plans :
  (total_volunteers = 5) →
  (pavilions_to_assign = 3) →
  (restricted_volunteers = 2) →
  (total_arrangements = 36) :=
by sorry

end expo_arrangement_plans_l975_97525


namespace total_spent_on_cards_l975_97523

-- Define the prices and tax rates
def football_card_price : ℝ := 2.73
def football_card_tax_rate : ℝ := 0.05
def football_card_quantity : ℕ := 2

def pokemon_card_price : ℝ := 4.01
def pokemon_card_tax_rate : ℝ := 0.08

def baseball_card_original_price : ℝ := 10
def baseball_card_discount_rate : ℝ := 0.10
def baseball_card_tax_rate : ℝ := 0.06

-- Calculate the total cost
def total_cost : ℝ :=
  -- Football cards
  (football_card_price * football_card_quantity) * (1 + football_card_tax_rate) +
  -- Pokemon cards
  pokemon_card_price * (1 + pokemon_card_tax_rate) +
  -- Baseball cards
  (baseball_card_original_price * (1 - baseball_card_discount_rate)) * (1 + baseball_card_tax_rate)

-- Theorem statement
theorem total_spent_on_cards :
  total_cost = 19.6038 := by sorry

end total_spent_on_cards_l975_97523


namespace queen_then_club_probability_l975_97542

-- Define a standard deck of cards
def standardDeck : ℕ := 52

-- Define the number of Queens in a standard deck
def numQueens : ℕ := 4

-- Define the number of clubs in a standard deck
def numClubs : ℕ := 13

-- Define the probability of drawing a Queen first and a club second
def probQueenThenClub : ℚ := 1 / 52

-- Theorem statement
theorem queen_then_club_probability :
  probQueenThenClub = (numQueens / standardDeck) * (numClubs / (standardDeck - 1)) :=
by sorry

end queen_then_club_probability_l975_97542


namespace simplify_expression_l975_97598

theorem simplify_expression (b : ℝ) (h : b ≠ -2/3) :
  3 - 2 / (2 + b / (1 + b)) = 3 - (1 + b) / (2 + 3*b) := by
  sorry

end simplify_expression_l975_97598


namespace symmetry_and_periodicity_l975_97558

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Define even function property
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem symmetry_and_periodicity 
  (h1 : is_even (fun x ↦ f (x - 1)))
  (h2 : is_even (fun x ↦ f (x - 2))) :
  (∀ x, f (-x - 2) = f x) ∧ 
  (∀ x, f (x + 2) = f x) ∧
  (∀ x, f' (-x + 4) = f' x) :=
sorry

end symmetry_and_periodicity_l975_97558


namespace imaginary_part_of_one_plus_i_l975_97575

theorem imaginary_part_of_one_plus_i :
  Complex.im (1 + Complex.I) = 1 := by sorry

end imaginary_part_of_one_plus_i_l975_97575


namespace segment_length_in_triangle_l975_97586

/-- Given a triangle with sides a, b, c, and three lines parallel to the sides
    intersecting at one point, with segments of length x cut off by the sides,
    prove that x = abc / (ab + bc + ac) -/
theorem segment_length_in_triangle (a b c x : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  x = (a * b * c) / (a * b + b * c + a * c) := by
  sorry

end segment_length_in_triangle_l975_97586


namespace shower_water_usage_l975_97589

/-- Calculates the total water usage for showers over a given period --/
def total_water_usage (weeks : ℕ) (shower_duration : ℕ) (water_per_minute : ℕ) : ℕ :=
  let days := weeks * 7
  let showers := days / 2
  let total_minutes := showers * shower_duration
  total_minutes * water_per_minute

theorem shower_water_usage : total_water_usage 4 10 2 = 280 := by
  sorry

end shower_water_usage_l975_97589


namespace greatest_k_value_l975_97547

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 73) →
  k ≤ Real.sqrt 105 :=
sorry

end greatest_k_value_l975_97547


namespace fraction_order_l975_97559

theorem fraction_order : (6/17)^2 < 8/25 ∧ 8/25 < 10/31 := by
  sorry

end fraction_order_l975_97559


namespace gas_station_candy_boxes_l975_97526

/-- The number of boxes of chocolate candy sold by a gas station -/
def chocolate_boxes : ℕ := 9 - (5 + 2)

/-- The total number of boxes sold -/
def total_boxes : ℕ := 9

/-- The number of boxes of sugar candy sold -/
def sugar_boxes : ℕ := 5

/-- The number of boxes of gum sold -/
def gum_boxes : ℕ := 2

theorem gas_station_candy_boxes :
  chocolate_boxes = 2 ∧
  chocolate_boxes + sugar_boxes + gum_boxes = total_boxes :=
sorry

end gas_station_candy_boxes_l975_97526


namespace three_digit_congruence_count_l975_97577

theorem three_digit_congruence_count :
  (∃ (S : Finset Nat), 
    (∀ x ∈ S, 100 ≤ x ∧ x ≤ 999) ∧
    (∀ x ∈ S, (4897 * x + 603) % 29 = 1427 % 29) ∧
    S.card = 28) := by
  sorry

end three_digit_congruence_count_l975_97577


namespace tom_candy_proof_l975_97539

def initial_candy : ℕ := 2
def friend_candy : ℕ := 7
def bought_candy : ℕ := 10
def final_candy : ℕ := 19

theorem tom_candy_proof :
  initial_candy + friend_candy + bought_candy = final_candy :=
by sorry

end tom_candy_proof_l975_97539
