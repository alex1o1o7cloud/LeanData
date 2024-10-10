import Mathlib

namespace hexagon_ring_area_l2804_280486

/-- The area of the ring between the inscribed and circumscribed circles of a regular hexagon -/
theorem hexagon_ring_area (a : ℝ) (h : a > 0) : 
  let r_inscribed := (Real.sqrt 3 / 2) * a
  let r_circumscribed := a
  let area_inscribed := π * r_inscribed ^ 2
  let area_circumscribed := π * r_circumscribed ^ 2
  area_circumscribed - area_inscribed = π * a^2 / 4 := by
  sorry

end hexagon_ring_area_l2804_280486


namespace inequality_solutions_l2804_280442

theorem inequality_solutions :
  (∀ x : ℝ, 3 + 2*x > -x - 6 ↔ x > -3) ∧
  (∀ x : ℝ, (2*x + 1 ≤ x + 3 ∧ (2*x + 1) / 3 > 1) ↔ 1 < x ∧ x ≤ 2) := by
  sorry

end inequality_solutions_l2804_280442


namespace expand_expression_l2804_280426

theorem expand_expression (x : ℝ) : 3 * (x - 6) * (x - 7) = 3 * x^2 - 39 * x + 126 := by
  sorry

end expand_expression_l2804_280426


namespace triangle_abc_properties_l2804_280455

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c = 2√3, sin B = 2 sin A, and C = π/3, then a = 2, b = 4, and the area is 2√3 -/
theorem triangle_abc_properties (a b c A B C : ℝ) : 
  c = 2 * Real.sqrt 3 →
  Real.sin B = 2 * Real.sin A →
  C = π / 3 →
  (a = 2 ∧ b = 4 ∧ (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) :=
by sorry

end triangle_abc_properties_l2804_280455


namespace currency_conversion_l2804_280479

/-- Conversion rates and constants --/
def paise_per_rupee : ℚ := 100
def usd_per_inr : ℚ := 12 / 1000
def eur_per_inr : ℚ := 10 / 1000
def gbp_per_inr : ℚ := 9 / 1000

/-- The value 'a' in paise --/
def a_paise : ℚ := 15000

/-- Theorem stating the correct values of 'a' in different currencies --/
theorem currency_conversion (a : ℚ) 
  (h1 : a * (1/2) / 100 = 75) : 
  a / paise_per_rupee = 150 ∧ 
  a / paise_per_rupee * usd_per_inr = 9/5 ∧ 
  a / paise_per_rupee * eur_per_inr = 3/2 ∧ 
  a / paise_per_rupee * gbp_per_inr = 27/20 := by
  sorry

#check currency_conversion

end currency_conversion_l2804_280479


namespace coin_value_ratio_l2804_280477

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of nickels -/
def num_nickels : ℕ := 4

/-- The number of dimes -/
def num_dimes : ℕ := 6

/-- The number of quarters -/
def num_quarters : ℕ := 2

theorem coin_value_ratio :
  ∃ (k : ℕ), k > 0 ∧
    num_nickels * nickel_value = 2 * k ∧
    num_dimes * dime_value = 6 * k ∧
    num_quarters * quarter_value = 5 * k :=
sorry

end coin_value_ratio_l2804_280477


namespace xy_commutativity_l2804_280481

theorem xy_commutativity (x y : ℝ) : 10 * x * y - 10 * y * x = 0 := by
  sorry

end xy_commutativity_l2804_280481


namespace sam_total_dimes_l2804_280405

def initial_dimes : ℕ := 9
def given_dimes : ℕ := 7

theorem sam_total_dimes : initial_dimes + given_dimes = 16 := by
  sorry

end sam_total_dimes_l2804_280405


namespace convex_nonagon_diagonals_l2804_280476

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem convex_nonagon_diagonals : 
  ∀ (n : ℕ), n = 9 → nonagon_diagonals = n * (n - 3) / 2 :=
by sorry

end convex_nonagon_diagonals_l2804_280476


namespace bird_legs_count_l2804_280447

theorem bird_legs_count (num_birds : ℕ) (legs_per_bird : ℕ) (h1 : num_birds = 5) (h2 : legs_per_bird = 2) :
  num_birds * legs_per_bird = 10 := by
  sorry

end bird_legs_count_l2804_280447


namespace possible_values_of_a_l2804_280416

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | x * a - 1 = 0}

-- State the theorem
theorem possible_values_of_a :
  ∀ a : ℝ, (A ∩ B a = B a) → (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end possible_values_of_a_l2804_280416


namespace total_barks_after_duration_l2804_280470

/-- Represents the number of barks per minute for a single dog -/
def barks_per_minute : ℕ := 30

/-- Represents the number of dogs -/
def num_dogs : ℕ := 2

/-- Represents the duration in minutes -/
def duration : ℕ := 10

/-- Theorem stating that the total number of barks after the given duration is 600 -/
theorem total_barks_after_duration :
  num_dogs * barks_per_minute * duration = 600 := by
  sorry

end total_barks_after_duration_l2804_280470


namespace order_of_a_l2804_280469

theorem order_of_a (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by
  sorry

end order_of_a_l2804_280469


namespace fraction_of_girls_at_dance_l2804_280465

theorem fraction_of_girls_at_dance (
  dalton_total : ℕ) (dalton_ratio_boys : ℕ) (dalton_ratio_girls : ℕ)
  (berkeley_total : ℕ) (berkeley_ratio_boys : ℕ) (berkeley_ratio_girls : ℕ)
  (kingston_total : ℕ) (kingston_ratio_boys : ℕ) (kingston_ratio_girls : ℕ)
  (h1 : dalton_total = 300)
  (h2 : dalton_ratio_boys = 3)
  (h3 : dalton_ratio_girls = 2)
  (h4 : berkeley_total = 210)
  (h5 : berkeley_ratio_boys = 3)
  (h6 : berkeley_ratio_girls = 4)
  (h7 : kingston_total = 240)
  (h8 : kingston_ratio_boys = 5)
  (h9 : kingston_ratio_girls = 7)
  : (dalton_total * dalton_ratio_girls / (dalton_ratio_boys + dalton_ratio_girls) +
     berkeley_total * berkeley_ratio_girls / (berkeley_ratio_boys + berkeley_ratio_girls) +
     kingston_total * kingston_ratio_girls / (kingston_ratio_boys + kingston_ratio_girls)) /
    (dalton_total + berkeley_total + kingston_total) = 38 / 75 :=
by sorry

end fraction_of_girls_at_dance_l2804_280465


namespace edmund_computer_savings_l2804_280478

/-- Represents the savings problem for Edmund's computer purchase. -/
def computer_savings_problem (computer_cost starting_balance monthly_gift : ℚ)
  (part_time_daily_wage part_time_days_per_week : ℚ)
  (extra_chore_wage chores_per_day regular_chores_per_week : ℚ)
  (car_wash_wage car_washes_per_week : ℚ)
  (lawn_mowing_wage lawns_per_week : ℚ) : Prop :=
  let weekly_earnings := 
    part_time_daily_wage * part_time_days_per_week +
    extra_chore_wage * (chores_per_day * 7 - regular_chores_per_week) +
    car_wash_wage * car_washes_per_week +
    lawn_mowing_wage * lawns_per_week
  let weekly_savings := weekly_earnings + monthly_gift / 4
  let days_to_save := 
    (↑(Nat.ceil ((computer_cost - starting_balance) / weekly_savings)) * 7 : ℚ)
  days_to_save = 49

/-- Theorem stating that Edmund will save enough for the computer in 49 days. -/
theorem edmund_computer_savings : 
  computer_savings_problem 750 200 50 10 3 2 4 12 3 2 5 1 := by
  sorry


end edmund_computer_savings_l2804_280478


namespace range_of_a_l2804_280432

def p (a : ℝ) : Prop := ∀ x : ℝ, (a - 3/2) ^ x > 0 ∧ (a - 3/2) ^ x < 1

def q (a : ℝ) : Prop := ∃ f : ℝ → ℝ, (∀ x ∈ [0, a], f x = x^2 - 4*x + 3) ∧
  (∀ y ∈ Set.range f, -1 ≤ y ∧ y ≤ 3)

theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) →
  ((3/2 < a ∧ a < 2) ∨ (5/2 ≤ a ∧ a ≤ 4)) :=
sorry

end range_of_a_l2804_280432


namespace right_triangle_legs_l2804_280401

theorem right_triangle_legs (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  c = 10 → -- hypotenuse length
  r = 2 → -- inscribed circle radius
  a + b - c = 2 * r → -- formula for inscribed circle radius
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  ((a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6)) :=
by
  sorry


end right_triangle_legs_l2804_280401


namespace circle_intersection_and_distance_four_points_distance_l2804_280463

/-- The circle equation parameters -/
structure CircleParams where
  m : ℝ
  h : m < 5

/-- The line equation parameters -/
structure LineParams where
  c : ℝ

/-- The theorem statement -/
theorem circle_intersection_and_distance (p : CircleParams) :
  (∃ M N : ℝ × ℝ, 
    (M.1^2 + M.2^2 - 2*M.1 - 4*M.2 + p.m = 0) ∧
    (N.1^2 + N.2^2 - 2*N.1 - 4*N.2 + p.m = 0) ∧
    (M.1 + 2*M.2 - 4 = 0) ∧
    (N.1 + 2*N.2 - 4 = 0) ∧
    ((M.1 - N.1)^2 + (M.2 - N.2)^2 = (4*Real.sqrt 5/5)^2)) →
  p.m = 4 :=
sorry

/-- The theorem statement for the second part -/
theorem four_points_distance (p : CircleParams) (l : LineParams) :
  (p.m = 4) →
  (∃ A B C D : ℝ × ℝ,
    (A.1^2 + A.2^2 - 2*A.1 - 4*A.2 + p.m = 0) ∧
    (B.1^2 + B.2^2 - 2*B.1 - 4*B.2 + p.m = 0) ∧
    (C.1^2 + C.2^2 - 2*C.1 - 4*C.2 + p.m = 0) ∧
    (D.1^2 + D.2^2 - 2*D.1 - 4*D.2 + p.m = 0) ∧
    ((A.1 - 2*A.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2) ∧
    ((B.1 - 2*B.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2) ∧
    ((C.1 - 2*C.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2) ∧
    ((D.1 - 2*D.2 + l.c)^2 / 5 = (Real.sqrt 5/5)^2)) ↔
  (4 - Real.sqrt 5 < l.c ∧ l.c < 2 + Real.sqrt 5) :=
sorry

end circle_intersection_and_distance_four_points_distance_l2804_280463


namespace system_solution_l2804_280417

theorem system_solution (x y z : ℚ) : 
  x = 2/7 ∧ y = 2/5 ∧ z = 2/3 ↔ 
  1/x + 1/y = 6 ∧ 1/y + 1/z = 4 ∧ 1/z + 1/x = 5 :=
by sorry

end system_solution_l2804_280417


namespace trivia_game_points_per_question_l2804_280402

/-- Given a trivia game where a player answers questions correctly and receives a total score,
    this theorem proves that if the player answers 10 questions correctly and scores 50 points,
    then each question is worth 5 points. -/
theorem trivia_game_points_per_question 
  (total_questions : ℕ) 
  (total_score : ℕ) 
  (points_per_question : ℕ) 
  (h1 : total_questions = 10) 
  (h2 : total_score = 50) : 
  points_per_question = 5 := by
  sorry

#check trivia_game_points_per_question

end trivia_game_points_per_question_l2804_280402


namespace line_point_at_47_l2804_280406

/-- A line passing through three given points -/
structure Line where
  -- Define the line using two points
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ
  -- Ensure the third point lies on the line
  x3 : ℝ
  y3 : ℝ
  point_on_line : (y3 - y1) / (x3 - x1) = (y2 - y1) / (x2 - x1)

/-- Theorem: For the given line, when x = 47, y = 143 -/
theorem line_point_at_47 (l : Line) 
  (h1 : l.x1 = 2 ∧ l.y1 = 8)
  (h2 : l.x2 = 6 ∧ l.y2 = 20)
  (h3 : l.x3 = 10 ∧ l.y3 = 32) :
  let m := (l.y2 - l.y1) / (l.x2 - l.x1)
  let b := l.y1 - m * l.x1
  m * 47 + b = 143 := by
  sorry

end line_point_at_47_l2804_280406


namespace range_of_a_l2804_280464

/-- Given two predicates p and q on real numbers, where p is a sufficient but not necessary condition for q, 
    this theorem states that the range of the parameter a in q is [0, 1/2]. -/
theorem range_of_a (p q : ℝ → Prop) (a : ℝ) : 
  (∀ x, p x ↔ |4*x - 3| ≤ 1) →
  (∀ x, q x ↔ x^2 - (2*a + 1)*x + a^2 + a ≤ 0) →
  (∀ x, p x → q x) →
  (∃ x, q x ∧ ¬p x) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_a_l2804_280464


namespace transaction_error_l2804_280441

theorem transaction_error (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 →
  (100 * y + x) - (100 * x + y) = 5616 →
  y = x + 56 := by
sorry

end transaction_error_l2804_280441


namespace smallest_n_cookies_l2804_280483

theorem smallest_n_cookies (n : ℕ) : (∀ m : ℕ, m > 0 → (15 * m - 3) % 7 ≠ 0) ∨ 
  ((15 * n - 3) % 7 = 0 ∧ n > 0 ∧ ∀ m : ℕ, 0 < m ∧ m < n → (15 * m - 3) % 7 ≠ 0) ↔ n = 3 := by
  sorry

end smallest_n_cookies_l2804_280483


namespace find_a_min_g_l2804_280451

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Define the function g
def g (x : ℝ) : ℝ := f 2 x - |x + 1|

-- Theorem for part (I)
theorem find_a : 
  (∀ x : ℝ, f 2 x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) → 
  ∃ a : ℝ, ∀ x : ℝ, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1 :=
sorry

-- Theorem for part (II)
theorem min_g : 
  ∃ m : ℝ, m = -1/2 ∧ ∀ x : ℝ, g x ≥ m :=
sorry

end find_a_min_g_l2804_280451


namespace difference_is_negative_1200_l2804_280450

def A : ℕ → ℕ
| 0 => 2 * 49
| n + 1 => (2 * n + 1) * (2 * n + 2) + A n

def B : ℕ → ℕ
| 0 => 48 * 49
| n + 1 => (2 * n) * (2 * n + 1) + B n

def difference : ℤ := A 24 - B 24

theorem difference_is_negative_1200 : difference = -1200 := by
  sorry

end difference_is_negative_1200_l2804_280450


namespace range_of_m_l2804_280444

theorem range_of_m (x : ℝ) (m : ℝ) : 
  (∃ x ∈ Set.Icc (-1) 1, x^2 - x - (m + 1) = 0) →
  m ∈ Set.Icc (-5/4) 1 :=
by sorry

end range_of_m_l2804_280444


namespace counterfeit_coin_weighings_l2804_280491

/-- Represents a weighing operation on a balance scale -/
def Weighing := List Nat → List Nat → Bool

/-- Represents a strategy for finding the counterfeit coin -/
def Strategy := List Nat → List (List Nat × List Nat)

/-- The number of coins -/
def n : Nat := 15

/-- The maximum number of weighings needed -/
def max_weighings : Nat := 3

theorem counterfeit_coin_weighings :
  ∃ (s : Strategy),
    ∀ (counterfeit : Fin n),
      ∀ (w : Weighing),
        (∀ i j : Fin n, i ≠ j → w [i.val] [j.val] = true) →
        (∀ i : Fin n, i ≠ counterfeit → w [i.val] [counterfeit.val] = false) →
        (s (List.range n)).length ≤ max_weighings ∧
        ∃ (result : Fin n), result = counterfeit := by sorry

end counterfeit_coin_weighings_l2804_280491


namespace sixth_power_of_complex_number_l2804_280495

theorem sixth_power_of_complex_number :
  let z : ℂ := (Real.sqrt 3 + Complex.I) / 2
  z^6 = -1 := by sorry

end sixth_power_of_complex_number_l2804_280495


namespace megan_remaining_acorns_l2804_280418

def initial_acorns : ℕ := 16
def acorns_given : ℕ := 7

theorem megan_remaining_acorns :
  initial_acorns - acorns_given = 9 := by
  sorry

end megan_remaining_acorns_l2804_280418


namespace video_game_cost_is_60_l2804_280414

/-- Represents the cost of the video game given Bucky's fish-catching earnings --/
def video_game_cost (last_weekend_earnings trout_price bluegill_price total_fish trout_percentage additional_savings : ℝ) : ℝ :=
  let trout_count := trout_percentage * total_fish
  let bluegill_count := total_fish - trout_count
  let sunday_earnings := trout_count * trout_price + bluegill_count * bluegill_price
  last_weekend_earnings + sunday_earnings + additional_savings

/-- Theorem stating the cost of the video game based on given conditions --/
theorem video_game_cost_is_60 :
  video_game_cost 35 5 4 5 0.6 2 = 60 := by
  sorry

end video_game_cost_is_60_l2804_280414


namespace function_equation_solution_l2804_280461

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 4 = x + y + 3) : 
  ∀ x : ℝ, f x = x + 4 := by
  sorry

end function_equation_solution_l2804_280461


namespace difference_of_squares_l2804_280471

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l2804_280471


namespace age_multiplier_proof_l2804_280459

theorem age_multiplier_proof (matt_age john_age : ℕ) (h1 : matt_age = 41) (h2 : john_age = 11) :
  ∃ x : ℚ, matt_age = x * john_age - 3 :=
by
  sorry

end age_multiplier_proof_l2804_280459


namespace greatest_five_digit_sum_l2804_280480

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem greatest_five_digit_sum (M : ℕ) :
  is_five_digit M ∧ 
  digit_product M = 180 ∧ 
  (∀ n : ℕ, is_five_digit n ∧ digit_product n = 180 → n ≤ M) →
  digit_sum M = 20 := by
sorry

end greatest_five_digit_sum_l2804_280480


namespace product_of_numbers_l2804_280427

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := by
  sorry

end product_of_numbers_l2804_280427


namespace sum_of_digits_9ab_is_18000_l2804_280431

/-- 
Represents an integer consisting of n repetitions of a digit d.
For example, repeat_digit 3 2000 represents 333...333 (2000 threes).
-/
def repeat_digit (d : ℕ) (n : ℕ) : ℕ :=
  (d * (10^n - 1)) / 9

/-- 
Calculates the sum of digits of a natural number in base 10.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

theorem sum_of_digits_9ab_is_18000 : 
  let a := repeat_digit 3 2000
  let b := repeat_digit 7 2000
  sum_of_digits (9 * a * b) = 18000 := by
  sorry

end sum_of_digits_9ab_is_18000_l2804_280431


namespace circle_radius_on_right_triangle_l2804_280407

theorem circle_radius_on_right_triangle (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  a = 7.5 →  -- shorter leg
  b = 10 →  -- longer leg (diameter of circle)
  6^2 + (c - r)^2 = r^2 →  -- chord condition
  r = 5 := by
sorry

end circle_radius_on_right_triangle_l2804_280407


namespace decimal_2011_equals_base7_5602_l2804_280433

/-- Converts a base 10 number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to a natural number in base 10 -/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 7 * acc + d) 0

theorem decimal_2011_equals_base7_5602 :
  fromBase7 [2, 0, 6, 5] = 2011 :=
by sorry

end decimal_2011_equals_base7_5602_l2804_280433


namespace defective_units_percentage_l2804_280453

/-- The percentage of defective units that are shipped for sale -/
def shipped_defective_percent : ℝ := 5

/-- The percentage of total units that are defective and shipped for sale -/
def total_defective_shipped_percent : ℝ := 0.5

/-- The percentage of defective units produced -/
def defective_percent : ℝ := 10

theorem defective_units_percentage :
  shipped_defective_percent * defective_percent / 100 = total_defective_shipped_percent := by
  sorry

end defective_units_percentage_l2804_280453


namespace f_max_at_zero_l2804_280468

-- Define the function f and its derivative
def f (x : ℝ) : ℝ := x^4 - 2*x^2 - 5

def f_deriv (x : ℝ) : ℝ := 4*x^3 - 4*x

-- State the theorem
theorem f_max_at_zero :
  (f 0 = -5) →
  (∀ x : ℝ, f_deriv x = 4*x^3 - 4*x) →
  ∀ x : ℝ, f x ≤ f 0 :=
by sorry

end f_max_at_zero_l2804_280468


namespace triangle_max_area_l2804_280429

/-- The maximum area of a triangle ABC where b = 3a and c = 2 -/
theorem triangle_max_area (a b c : ℝ) (h1 : b = 3 * a) (h2 : c = 2) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  let area := (1/2) * b * c * Real.sin A
  ∀ x > 0, area ≤ Real.sqrt 2 / 2 ∧ 
  ∃ a₀ > 0, area = Real.sqrt 2 / 2 := by
  sorry

end triangle_max_area_l2804_280429


namespace perimeter_equals_127_32_l2804_280474

/-- The perimeter of a figure constructed with 6 equilateral triangles, where the first triangle
    has a side length of 1 cm and each subsequent triangle has sides equal to half the length
    of the previous triangle. -/
def perimeter_of_triangles : ℚ :=
  let side_lengths : List ℚ := [1, 1/2, 1/4, 1/8, 1/16, 1/32]
  let unique_segments : List ℚ := [1, 1, 1/2, 1/2, 1/4, 1/4, 1/8, 1/8, 1/16, 1/16, 1/32, 1/32, 1/32]
  unique_segments.sum

theorem perimeter_equals_127_32 : perimeter_of_triangles = 127 / 32 := by
  sorry

end perimeter_equals_127_32_l2804_280474


namespace garage_bikes_l2804_280454

/-- The number of bikes that can be assembled given a certain number of wheels -/
def bikes_assembled (total_wheels : ℕ) (wheels_per_bike : ℕ) : ℕ :=
  total_wheels / wheels_per_bike

/-- Theorem: Given 20 bike wheels and 2 wheels per bike, 10 bikes can be assembled -/
theorem garage_bikes : bikes_assembled 20 2 = 10 := by
  sorry

end garage_bikes_l2804_280454


namespace factor_expression_l2804_280457

theorem factor_expression (x : ℝ) : 72 * x^11 + 162 * x^22 = 18 * x^11 * (4 + 9 * x^11) := by
  sorry

end factor_expression_l2804_280457


namespace distribution_recurrence_l2804_280496

/-- The number of ways to distribute n distinct items to k people,
    such that each person receives at least one item -/
def g (n k : ℕ) : ℕ := sorry

theorem distribution_recurrence (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) :
  g (n + 1) k = k * g n (k - 1) + k * g n k :=
sorry

end distribution_recurrence_l2804_280496


namespace wam_gm_difference_bound_l2804_280443

theorem wam_gm_difference_bound (k b : ℝ) (h1 : 0 < k) (h2 : k < 1) (h3 : b > 0) : 
  let a := k * b
  let c := (a + b) / 2
  let wam := (2 * a + 3 * b + 4 * c) / 9
  let gm := (a * b * c) ^ (1/3 : ℝ)
  (wam - gm = b * ((5 * k + 5) / 9 - ((k * (k + 1) * b^2) / 2) ^ (1/3 : ℝ))) ∧
  (wam - gm < ((1 - k)^2 * b) / (8 * k)) := by sorry

end wam_gm_difference_bound_l2804_280443


namespace ages_sum_l2804_280456

/-- Represents the ages of Samantha, Ravi, and Kim -/
structure Ages where
  samantha : ℝ
  ravi : ℝ
  kim : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.samantha = ages.ravi + 10 ∧
  ages.samantha + 12 = 3 * (ages.ravi - 5) ∧
  ages.kim = ages.ravi / 2

/-- The theorem to be proved -/
theorem ages_sum (ages : Ages) : 
  satisfiesConditions ages → ages.samantha + ages.ravi + ages.kim = 56.25 := by
  sorry


end ages_sum_l2804_280456


namespace both_locks_stall_time_l2804_280423

/-- The time (in minutes) the first lock stalls the raccoons -/
def first_lock_time : ℕ := 5

/-- The time (in minutes) the second lock stalls the raccoons -/
def second_lock_time : ℕ := 3 * first_lock_time - 3

/-- The time (in minutes) both locks together stall the raccoons -/
def both_locks_time : ℕ := 5 * second_lock_time

/-- Theorem stating that both locks together stall the raccoons for 60 minutes -/
theorem both_locks_stall_time : both_locks_time = 60 := by
  sorry

end both_locks_stall_time_l2804_280423


namespace cubic_polynomials_with_constant_difference_l2804_280430

/-- Two monic cubic polynomials with specific roots and a constant difference -/
theorem cubic_polynomials_with_constant_difference 
  (f g : ℝ → ℝ) 
  (r : ℝ) 
  (hf : ∃ a : ℝ, ∀ x, f x = (x - (r + 2)) * (x - (r + 8)) * (x - a))
  (hg : ∃ b : ℝ, ∀ x, g x = (x - (r + 4)) * (x - (r + 10)) * (x - b))
  (h_diff : ∀ x, f x - g x = r) :
  r = 32 := by
sorry

end cubic_polynomials_with_constant_difference_l2804_280430


namespace line_points_k_value_l2804_280482

/-- Given a line with equation x = 2y + 5 and two points (m, n) and (m + 3, n + k) on this line, k = 3/2. -/
theorem line_points_k_value (m n k : ℝ) : 
  (m = 2 * n + 5) → 
  (m + 3 = 2 * (n + k) + 5) → 
  k = 3 / 2 := by
sorry

end line_points_k_value_l2804_280482


namespace max_value_on_line_l2804_280412

/-- Given a point A(3,1) on the line mx + ny + 1 = 0, where mn > 0, 
    the maximum value of 3/m + 1/n is -16. -/
theorem max_value_on_line (m n : ℝ) : 
  (3 * m + n = -1) → 
  (m * n > 0) → 
  (∀ k l : ℝ, (3 * k + l = -1) → (k * l > 0) → (3 / m + 1 / n ≥ 3 / k + 1 / l)) →
  3 / m + 1 / n = -16 := by
  sorry

end max_value_on_line_l2804_280412


namespace complex_modulus_l2804_280460

/-- If z is a complex number satisfying (2+i)z = 5, then |z| = √5 -/
theorem complex_modulus (z : ℂ) (h : (2 + Complex.I) * z = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l2804_280460


namespace fraction_comparison_l2804_280438

theorem fraction_comparison : -3/4 > -4/5 := by sorry

end fraction_comparison_l2804_280438


namespace cost_equation_holds_l2804_280415

/-- Represents the cost equation for notebooks and colored pens --/
def cost_equation (x : ℕ) : Prop :=
  let total_items : ℕ := 20
  let total_cost : ℕ := 50
  let notebook_cost : ℕ := 4
  let pen_cost : ℕ := 2
  2 * (total_items - x) + notebook_cost * x = total_cost

/-- Theorem stating the cost equation holds for the given scenario --/
theorem cost_equation_holds : ∃ x : ℕ, cost_equation x := by sorry

end cost_equation_holds_l2804_280415


namespace equation_solution_l2804_280462

theorem equation_solution :
  ∃ (t₁ t₂ : ℝ), t₁ > t₂ ∧
  (∀ t : ℝ, t ≠ 10 → (t^2 - 3*t - 70) / (t - 10) = 7 / (t + 4) ↔ t = t₁ ∨ t = t₂) ∧
  t₁ = -3 ∧ t₂ = -7 :=
by sorry

end equation_solution_l2804_280462


namespace christen_peeled_22_l2804_280449

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initial_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  time_before_christen : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def potatoes_peeled_by_christen (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Christen peeled 22 potatoes -/
theorem christen_peeled_22 (scenario : PotatoPeeling) 
  (h1 : scenario.initial_potatoes = 60)
  (h2 : scenario.homer_rate = 4)
  (h3 : scenario.christen_rate = 6)
  (h4 : scenario.time_before_christen = 6) :
  potatoes_peeled_by_christen scenario = 22 := by
  sorry

end christen_peeled_22_l2804_280449


namespace sum_of_digits_of_greatest_prime_factor_8191_l2804_280420

def greatest_prime_factor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_greatest_prime_factor_8191 :
  sum_of_digits (greatest_prime_factor 8191) = 7 := by sorry

end sum_of_digits_of_greatest_prime_factor_8191_l2804_280420


namespace parabola_line_slope_l2804_280448

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
def on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Define a point in the first quadrant
def in_first_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 > 0

-- Define the vector relationship
def vector_relation (P Q : ℝ × ℝ) : Prop :=
  3 * (focus.1 - P.1) = Q.1 - focus.1 ∧
  3 * (focus.2 - P.2) = Q.2 - focus.2

-- Main theorem
theorem parabola_line_slope (P Q : ℝ × ℝ) :
  on_parabola P →
  on_parabola Q →
  in_first_quadrant Q →
  vector_relation P Q →
  (Q.2 - P.2) / (Q.1 - P.1) = Real.sqrt 3 :=
sorry

end parabola_line_slope_l2804_280448


namespace product_of_distinct_nonzero_reals_l2804_280425

theorem product_of_distinct_nonzero_reals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h : x - 2 / x = y - 2 / y) : x * y = -2 := by
  sorry

end product_of_distinct_nonzero_reals_l2804_280425


namespace foreign_language_score_foreign_language_score_is_98_l2804_280473

theorem foreign_language_score (average_three : ℝ) (average_two : ℝ) 
  (h1 : average_three = 94) (h2 : average_two = 92) : ℝ :=
  3 * average_three - 2 * average_two

theorem foreign_language_score_is_98 (average_three : ℝ) (average_two : ℝ) 
  (h1 : average_three = 94) (h2 : average_two = 92) : 
  foreign_language_score average_three average_two h1 h2 = 98 := by
  sorry

end foreign_language_score_foreign_language_score_is_98_l2804_280473


namespace total_dog_weight_l2804_280422

/-- The weight of Evan's dog in pounds -/
def evans_dog_weight : ℝ := 63

/-- The ratio of Evan's dog weight to Ivan's dog weight -/
def weight_ratio : ℝ := 7

/-- Theorem: The total weight of Evan's and Ivan's dogs is 72 pounds -/
theorem total_dog_weight : evans_dog_weight + evans_dog_weight / weight_ratio = 72 := by
  sorry

end total_dog_weight_l2804_280422


namespace a_values_l2804_280434

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (2 * x + a) ^ 3

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * (2 * x + a) ^ 2

-- Theorem statement
theorem a_values (a : ℝ) : f_derivative a 1 = 6 → a = -1 ∨ a = -3 := by
  sorry

end a_values_l2804_280434


namespace triangle_perimeter_l2804_280475

/-- Given a triangle with inradius 1.5 cm and area 29.25 cm², its perimeter is 39 cm. -/
theorem triangle_perimeter (inradius : ℝ) (area : ℝ) (perimeter : ℝ) : 
  inradius = 1.5 → area = 29.25 → perimeter = area / inradius * 2 → perimeter = 39 := by
  sorry

end triangle_perimeter_l2804_280475


namespace inradius_bounds_l2804_280493

theorem inradius_bounds (a b c r : ℝ) :
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  r = (a + b - c) / 2 →
  r < c / 4 ∧ r < min a b / 2 := by
  sorry

end inradius_bounds_l2804_280493


namespace dice_probability_l2804_280446

/-- The probability of all dice showing the same number -/
def probability : ℝ := 0.0007716049382716049

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The number of dice thrown -/
def num_dice : ℕ := 5

theorem dice_probability :
  (1 / faces : ℝ) ^ (num_dice - 1) = probability := by sorry

end dice_probability_l2804_280446


namespace inverse_variation_example_l2804_280439

/-- Given two quantities that vary inversely, this function represents their relationship -/
def inverse_variation (k : ℝ) (a b : ℝ) : Prop := a * b = k

/-- Theorem: For inverse variation, if b = 0.5 when a = 800, then b = 0.125 when a = 3200 -/
theorem inverse_variation_example (k : ℝ) :
  inverse_variation k 800 0.5 → inverse_variation k 3200 0.125 := by
  sorry

end inverse_variation_example_l2804_280439


namespace existence_of_n_with_k_prime_factors_l2804_280428

theorem existence_of_n_with_k_prime_factors (k m : ℕ) (hk : k > 0) (hm : m > 0) (hm_odd : Odd m) :
  ∃ n : ℕ, n > 0 ∧ (∃ (p : Finset ℕ), p.card ≥ k ∧ ∀ q ∈ p, Prime q ∧ q ∣ (m^n + n^m)) :=
sorry

end existence_of_n_with_k_prime_factors_l2804_280428


namespace marble_jar_problem_l2804_280499

theorem marble_jar_problem :
  ∀ (total_marbles : ℕ) (blue1 green1 blue2 green2 : ℕ),
    -- Jar 1 ratio condition
    7 * green1 = 2 * blue1 →
    -- Jar 2 ratio condition
    8 * green2 = blue2 →
    -- Equal total marbles in each jar
    blue1 + green1 = blue2 + green2 →
    -- Total green marbles
    green1 + green2 = 135 →
    -- Difference in blue marbles
    blue2 - blue1 = 45 :=
by sorry

end marble_jar_problem_l2804_280499


namespace rectangle_side_length_l2804_280489

/-- Given a square with side length 5 and a rectangle with one side 4,
    if they have the same area, then the other side of the rectangle is 6.25 -/
theorem rectangle_side_length (square_side : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  square_side = 5 →
  rectangle_width = 4 →
  square_side * square_side = rectangle_width * rectangle_length →
  rectangle_length = 6.25 := by
sorry

end rectangle_side_length_l2804_280489


namespace sqrt_calculations_l2804_280472

theorem sqrt_calculations :
  (2 * Real.sqrt 2 + Real.sqrt 27 - Real.sqrt 8 = 3 * Real.sqrt 3) ∧
  ((2 * Real.sqrt 12 - 3 * Real.sqrt (1/3)) * Real.sqrt 6 = 9 * Real.sqrt 2) := by
  sorry

end sqrt_calculations_l2804_280472


namespace ceiling_neg_sqrt_64_over_4_l2804_280487

theorem ceiling_neg_sqrt_64_over_4 : ⌈-Real.sqrt (64 / 4)⌉ = -4 := by sorry

end ceiling_neg_sqrt_64_over_4_l2804_280487


namespace smallest_value_complex_sum_l2804_280452

theorem smallest_value_complex_sum (a b c d : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_omega_power : ω^4 = 1) 
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (z : ℂ), ∀ (x y u v : ℤ), x ≠ y ∧ x ≠ u ∧ x ≠ v ∧ y ≠ u ∧ y ≠ v ∧ u ≠ v →
    Complex.abs (x + y*ω + u*ω^2 + v*ω^3) ≥ Complex.abs z ∧
    Complex.abs z = Real.sqrt 3 :=
sorry

end smallest_value_complex_sum_l2804_280452


namespace problem_solution_l2804_280458

theorem problem_solution (x y : ℚ) (hx : x = 2/3) (hy : y = 3/2) :
  (3/4 : ℚ) * x^4 * y^5 = 9/8 := by sorry

end problem_solution_l2804_280458


namespace total_nails_is_113_l2804_280403

/-- The number of nails Cassie needs to cut for her pets -/
def total_nails_to_cut : ℕ :=
  let num_dogs : ℕ := 4
  let num_parrots : ℕ := 8
  let nails_per_dog_foot : ℕ := 4
  let feet_per_dog : ℕ := 4
  let claws_per_parrot_leg : ℕ := 3
  let legs_per_parrot : ℕ := 2
  let extra_nail : ℕ := 1

  let dog_nails : ℕ := num_dogs * nails_per_dog_foot * feet_per_dog
  let parrot_nails : ℕ := num_parrots * claws_per_parrot_leg * legs_per_parrot
  
  dog_nails + parrot_nails + extra_nail

/-- Theorem stating that the total number of nails Cassie needs to cut is 113 -/
theorem total_nails_is_113 : total_nails_to_cut = 113 := by
  sorry

end total_nails_is_113_l2804_280403


namespace equation_solution_l2804_280485

theorem equation_solution :
  let f (x : ℝ) := x^2 + 2*x + 1
  let g (x : ℝ) := |3*x - 2|
  let sol₁ := (-7 + Real.sqrt 37) / 2
  let sol₂ := (-7 - Real.sqrt 37) / 2
  (∀ x : ℝ, f x = g x ↔ x = sol₁ ∨ x = sol₂) :=
by sorry

end equation_solution_l2804_280485


namespace triangle_perimeter_l2804_280400

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a * Real.cos B = 3 ∧
  b * Real.sin A = 4 ∧
  (1/2) * a * c * Real.sin B = 10 →
  a + b + c = 10 + 2 * Real.sqrt 5 := by
sorry

end triangle_perimeter_l2804_280400


namespace number_subtraction_problem_l2804_280488

theorem number_subtraction_problem (x y : ℝ) : 
  (x - 5) / 7 = 7 → (x - y) / 13 = 4 → y = 2 := by
  sorry

end number_subtraction_problem_l2804_280488


namespace jose_fowls_count_l2804_280436

/-- The number of fowls Jose has is the sum of his chickens and ducks -/
theorem jose_fowls_count :
  let chickens : ℕ := 28
  let ducks : ℕ := 18
  let fowls : ℕ := chickens + ducks
  fowls = 46 := by sorry

end jose_fowls_count_l2804_280436


namespace decagon_diagonals_from_vertex_l2804_280498

/-- The number of diagonals from a single vertex in a decagon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: In a decagon, the number of diagonals from a single vertex is 7 -/
theorem decagon_diagonals_from_vertex : 
  diagonals_from_vertex 10 = 7 := by sorry

end decagon_diagonals_from_vertex_l2804_280498


namespace quadratic_square_of_binomial_l2804_280409

theorem quadratic_square_of_binomial (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 4 * x^2 + 18 * x + a = (2 * x + b)^2) → a = 81 / 4 := by
  sorry

end quadratic_square_of_binomial_l2804_280409


namespace percent_of_whole_l2804_280424

theorem percent_of_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = 50 ↔ part = (1/2) * whole :=
sorry

end percent_of_whole_l2804_280424


namespace price_per_dozen_calculation_l2804_280410

/-- The price per dozen of additional doughnuts -/
def price_per_dozen (first_doughnut_price : ℚ) (total_cost : ℚ) (total_doughnuts : ℕ) : ℚ :=
  (total_cost - first_doughnut_price) / ((total_doughnuts - 1 : ℚ) / 12)

/-- Theorem stating the price per dozen of additional doughnuts -/
theorem price_per_dozen_calculation (first_doughnut_price : ℚ) (total_cost : ℚ) (total_doughnuts : ℕ) 
  (h1 : first_doughnut_price = 1)
  (h2 : total_cost = 24)
  (h3 : total_doughnuts = 48) :
  price_per_dozen first_doughnut_price total_cost total_doughnuts = 276 / 47 :=
by sorry

end price_per_dozen_calculation_l2804_280410


namespace foci_distance_of_problem_ellipse_l2804_280435

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

-- Define the conditions of the problem
def problem_ellipse : Ellipse := {
  center := (5, 2)
  semi_major_axis := 5
  semi_minor_axis := 2
}

-- Theorem statement
theorem foci_distance_of_problem_ellipse :
  let e := problem_ellipse
  let c := Real.sqrt (e.semi_major_axis ^ 2 - e.semi_minor_axis ^ 2)
  c = Real.sqrt 21 := by sorry

end foci_distance_of_problem_ellipse_l2804_280435


namespace geometric_sequence_ratio_l2804_280497

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_a4 : a 4 = 1/4) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 1/2 := by
sorry

end geometric_sequence_ratio_l2804_280497


namespace intersection_of_A_and_B_l2804_280490

-- Define set A
def A : Set ℝ := {x | x^2 + x - 12 < 0}

-- Define set B
def B : Set ℝ := {x | Real.sqrt (x + 2) < 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l2804_280490


namespace circle_center_l2804_280413

/-- A circle passing through (0,0) and tangent to y = x^2 at (1,1) has center (-1, 2) -/
theorem circle_center (c : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x - c.1)^2 + (y - c.2)^2 = (c.1)^2 + (c.2)^2 → (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
  (∃ (r : ℝ), ∀ (x y : ℝ), y = x^2 → ((x - 1)^2 + (y - 1)^2 = r^2 → x = 1 ∧ y = 1)) →
  c = (-1, 2) := by
sorry


end circle_center_l2804_280413


namespace complex_modulus_one_l2804_280408

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_modulus_one_l2804_280408


namespace expense_representation_l2804_280492

-- Define a type for financial transactions
inductive Transaction
| Income : ℕ → Transaction
| Expense : ℕ → Transaction

-- Define a function to represent transactions numerically
def represent : Transaction → ℤ
| Transaction.Income n => n
| Transaction.Expense n => -n

-- State the theorem
theorem expense_representation (amount : ℕ) :
  represent (Transaction.Income amount) = amount →
  represent (Transaction.Expense amount) = -amount :=
by
  sorry

end expense_representation_l2804_280492


namespace only_vertical_angles_always_equal_l2804_280421

-- Define the types for lines and angles
def Line : Type := ℝ → ℝ → Prop
def Angle : Type := ℝ

-- Define the relationships between angles
def are_alternate_interior (a b : Angle) (l1 l2 : Line) : Prop := sorry
def are_consecutive_interior (a b : Angle) (l1 l2 : Line) : Prop := sorry
def are_vertical (a b : Angle) : Prop := sorry
def are_adjacent_supplementary (a b : Angle) : Prop := sorry
def are_corresponding (a b : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the property of being supplementary
def are_supplementary (a b : Angle) : Prop := sorry

-- Theorem stating that only vertical angles are always equal
theorem only_vertical_angles_always_equal :
  ∀ (a b : Angle) (l1 l2 : Line),
    (are_vertical a b → a = b) ∧
    (are_alternate_interior a b l1 l2 → ¬(l1 = l2) → ¬(a = b)) ∧
    (are_consecutive_interior a b l1 l2 → ¬(l1 = l2) → ¬(are_supplementary a b)) ∧
    (are_adjacent_supplementary a b → ¬(a = b)) ∧
    (are_corresponding a b l1 l2 → ¬(l1 = l2) → ¬(a = b)) :=
by
  sorry

end only_vertical_angles_always_equal_l2804_280421


namespace regular_pentagon_not_seamless_l2804_280404

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

def is_divisor_of_360 (angle : ℚ) : Prop := ∃ k : ℕ, 360 = k * angle

theorem regular_pentagon_not_seamless :
  ¬(is_divisor_of_360 (interior_angle 5)) ∧
  (is_divisor_of_360 (interior_angle 3)) ∧
  (is_divisor_of_360 (interior_angle 4)) ∧
  (is_divisor_of_360 (interior_angle 6)) :=
sorry

end regular_pentagon_not_seamless_l2804_280404


namespace max_golf_rounds_is_eight_l2804_280484

/-- Calculates the maximum number of golf rounds that can be played given the specified conditions. -/
def maxGolfRounds (initialCost : ℚ) (membershipFee : ℚ) (budget : ℚ) 
  (discount2nd : ℚ) (discount3rd : ℚ) (discountSubsequent : ℚ) : ℕ :=
  let totalBudget := budget + membershipFee
  let cost1st := initialCost
  let cost2nd := initialCost * (1 - discount2nd)
  let cost3rd := initialCost * (1 - discount3rd)
  let costSubsequent := initialCost * (1 - discountSubsequent)
  let remainingAfter3 := totalBudget - cost1st - cost2nd - cost3rd
  let additionalRounds := (remainingAfter3 / costSubsequent).floor
  3 + additionalRounds.toNat

/-- Theorem stating that the maximum number of golf rounds is 8 under the given conditions. -/
theorem max_golf_rounds_is_eight :
  maxGolfRounds 80 100 400 (1/10) (1/5) (3/10) = 8 := by
  sorry

end max_golf_rounds_is_eight_l2804_280484


namespace largest_integer_satisfying_inequality_l2804_280437

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 7 ↔ (x : ℚ) / 4 + 3 / 7 < 9 / 4 := by
  sorry

end largest_integer_satisfying_inequality_l2804_280437


namespace sequence_formula_l2804_280440

theorem sequence_formula (a : ℕ+ → ℚ) 
  (h1 : a 1 = 1/2)
  (h2 : ∀ n : ℕ+, a n * a (n + 1) = n / (n + 2)) :
  ∀ n : ℕ+, a n = n / (n + 1) := by
sorry

end sequence_formula_l2804_280440


namespace abs_x_minus_two_iff_x_in_range_l2804_280466

theorem abs_x_minus_two_iff_x_in_range (x : ℝ) : |x - 2| ≤ 5 ↔ -3 ≤ x ∧ x ≤ 7 := by
  sorry

end abs_x_minus_two_iff_x_in_range_l2804_280466


namespace eddys_spider_plant_babies_l2804_280467

/-- A spider plant that produces baby plants -/
structure SpiderPlant where
  /-- Number of baby plants produced per cycle -/
  babies_per_cycle : ℕ
  /-- Number of cycles per year -/
  cycles_per_year : ℕ

/-- Calculate the total number of baby plants produced over a given number of years -/
def total_babies (plant : SpiderPlant) (years : ℕ) : ℕ :=
  plant.babies_per_cycle * plant.cycles_per_year * years

/-- Theorem: Eddy's spider plant produces 16 baby plants after 4 years -/
theorem eddys_spider_plant_babies :
  ∃ (plant : SpiderPlant), plant.babies_per_cycle = 2 ∧ plant.cycles_per_year = 2 ∧ total_babies plant 4 = 16 := by
  sorry

end eddys_spider_plant_babies_l2804_280467


namespace oranges_per_box_l2804_280494

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 24) (h2 : num_boxes = 3) :
  total_oranges / num_boxes = 8 := by
  sorry

end oranges_per_box_l2804_280494


namespace rectangular_solid_diagonal_l2804_280445

/-- 
Given a rectangular solid with dimensions x, y, and z,
if the total surface area is 34 cm² and the total length of all edges is 28 cm,
then the length of any interior diagonal is √15 cm.
-/
theorem rectangular_solid_diagonal 
  (x y z : ℝ) 
  (h_surface_area : 2 * (x * y + y * z + z * x) = 34)
  (h_edge_length : 4 * (x + y + z) = 28) :
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 15 := by
  sorry

end rectangular_solid_diagonal_l2804_280445


namespace special_heptagon_perimeter_l2804_280411

/-- A heptagon with six sides of length 3 and one side of length 5 -/
structure SpecialHeptagon where
  side_length_six : ℝ
  side_length_one : ℝ
  is_heptagon : side_length_six = 3 ∧ side_length_one = 5

/-- The perimeter of a SpecialHeptagon -/
def perimeter (h : SpecialHeptagon) : ℝ :=
  6 * h.side_length_six + h.side_length_one

theorem special_heptagon_perimeter (h : SpecialHeptagon) :
  perimeter h = 23 := by
  sorry

end special_heptagon_perimeter_l2804_280411


namespace max_product_constrained_sum_l2804_280419

theorem max_product_constrained_sum (a b : ℝ) : 
  a + b = 1 → (∀ x y : ℝ, x + y = 1 → a * b ≥ x * y) → a * b = 1/4 := by
  sorry

end max_product_constrained_sum_l2804_280419
