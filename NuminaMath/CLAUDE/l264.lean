import Mathlib

namespace NUMINAMATH_CALUDE_box_volume_increase_l264_26456

/-- Given a rectangular box with length l, width w, and height h, 
    if the volume is 5670, surface area is 2534, and sum of edges is 252,
    then increasing each dimension by 1 results in a volume of 7001 -/
theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 5670)
  (hs : 2 * (l * w + w * h + h * l) = 2534)
  (he : 4 * (l + w + h) = 252) :
  (l + 1) * (w + 1) * (h + 1) = 7001 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l264_26456


namespace NUMINAMATH_CALUDE_triangle_properties_l264_26418

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the main results -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.A = 2 * t.B) : 
  (t.b = 2 ∧ t.c = 1 → t.a = Real.sqrt 6) ∧
  (t.b + t.c = Real.sqrt 3 * t.a → t.B = π / 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l264_26418


namespace NUMINAMATH_CALUDE_distribution_methods_eq_72_l264_26429

/-- Number of teachers -/
def num_teachers : ℕ := 3

/-- Number of students -/
def num_students : ℕ := 3

/-- Total number of tickets -/
def total_tickets : ℕ := 6

/-- Function to calculate the number of distribution methods -/
def distribution_methods : ℕ := sorry

/-- Theorem stating that the number of distribution methods is 72 -/
theorem distribution_methods_eq_72 : distribution_methods = 72 := by sorry

end NUMINAMATH_CALUDE_distribution_methods_eq_72_l264_26429


namespace NUMINAMATH_CALUDE_ticket_queue_arrangements_l264_26411

/-- Represents the number of valid arrangements for a ticket queue --/
def validArrangements (n : ℕ) : ℕ :=
  Nat.factorial (2 * n) / (Nat.factorial n * Nat.factorial (n + 1))

/-- Theorem stating the number of valid arrangements for a ticket queue --/
theorem ticket_queue_arrangements (n : ℕ) :
  validArrangements n = 
    let total_people := 2 * n
    let people_with_five_yuan := n
    let people_with_ten_yuan := n
    let ticket_price := 5
    -- The actual number of valid arrangements
    Nat.factorial total_people / (Nat.factorial people_with_five_yuan * Nat.factorial (people_with_ten_yuan + 1)) :=
by sorry

#check ticket_queue_arrangements

end NUMINAMATH_CALUDE_ticket_queue_arrangements_l264_26411


namespace NUMINAMATH_CALUDE_solve_for_Q_l264_26481

theorem solve_for_Q : ∃ Q : ℝ, (Q ^ 4).sqrt = 32 * (64 ^ (1/6)) → Q = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_Q_l264_26481


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l264_26450

theorem quadratic_inequality_condition (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x + 1 > 0) → (-2 ≤ m ∧ m ≤ 2) ∧
  ¬((-2 ≤ m ∧ m ≤ 2) → (∀ x : ℝ, x^2 - m*x + 1 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l264_26450


namespace NUMINAMATH_CALUDE_ellipse_b_value_l264_26407

/-- An ellipse with foci at (1, 1) and (1, -1) passing through (7, 0) -/
structure Ellipse where
  foci1 : ℝ × ℝ := (1, 1)
  foci2 : ℝ × ℝ := (1, -1)
  point : ℝ × ℝ := (7, 0)

/-- The standard form of an ellipse equation -/
def standard_equation (h k a b : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- The theorem stating that b = 6 for the given ellipse -/
theorem ellipse_b_value (e : Ellipse) :
  ∃ (h k a b : ℝ), a > 0 ∧ b > 0 ∧
  standard_equation h k a b (e.point.1) (e.point.2) ∧
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_b_value_l264_26407


namespace NUMINAMATH_CALUDE_beach_problem_l264_26490

/-- The number of people originally in the first row of the beach. -/
def first_row : ℕ := sorry

/-- The number of people originally in the second row of the beach. -/
def second_row : ℕ := 20

/-- The number of people in the third row of the beach. -/
def third_row : ℕ := 18

/-- The number of people who left the first row to wade in the water. -/
def left_first_row : ℕ := 3

/-- The number of people who left the second row to wade in the water. -/
def left_second_row : ℕ := 5

/-- The total number of people left relaxing on the beach. -/
def total_remaining : ℕ := 54

theorem beach_problem :
  first_row = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_beach_problem_l264_26490


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_special_properties_l264_26421

/-- Round a natural number to the nearest ten -/
def roundToNearestTen (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

/-- Check if a natural number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_numbers_with_special_properties (p q : ℕ) :
  isTwoDigit p ∧ isTwoDigit q ∧
  (roundToNearestTen p - roundToNearestTen q = p - q) ∧
  (roundToNearestTen p * roundToNearestTen q = p * q + 184) →
  ((p = 16 ∧ q = 26) ∨ (p = 26 ∧ q = 16)) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_special_properties_l264_26421


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l264_26406

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 3) / Real.log 2}
def B : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > 0} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l264_26406


namespace NUMINAMATH_CALUDE_book_club_unique_books_book_club_unique_books_eq_61_l264_26464

theorem book_club_unique_books : ℕ :=
  let tony_books : ℕ := 23
  let dean_books : ℕ := 20
  let breanna_books : ℕ := 30
  let piper_books : ℕ := 26
  let asher_books : ℕ := 25
  let tony_dean_shared : ℕ := 5
  let breanna_piper_asher_shared : ℕ := 7
  let dean_piper_shared : ℕ := 6
  let dean_piper_tony_shared : ℕ := 3
  let asher_breanna_tony_shared : ℕ := 8
  let all_shared : ℕ := 2
  let breanna_piper_shared : ℕ := 9
  let breanna_piper_dean_shared : ℕ := 4
  let breanna_piper_asher_shared : ℕ := 2

  let total_books : ℕ := tony_books + dean_books + breanna_books + piper_books + asher_books
  let overlaps : ℕ := tony_dean_shared + 
                      2 * breanna_piper_asher_shared + 
                      2 * dean_piper_tony_shared + 
                      (dean_piper_shared - dean_piper_tony_shared) + 
                      2 * (asher_breanna_tony_shared - all_shared) + 
                      4 * all_shared + 
                      (breanna_piper_shared - breanna_piper_dean_shared - breanna_piper_asher_shared) + 
                      2 * breanna_piper_dean_shared + 
                      breanna_piper_asher_shared

  total_books - overlaps
  
theorem book_club_unique_books_eq_61 : book_club_unique_books = 61 := by
  sorry

end NUMINAMATH_CALUDE_book_club_unique_books_book_club_unique_books_eq_61_l264_26464


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l264_26416

/-- Number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute 6 3 = 92 := by sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l264_26416


namespace NUMINAMATH_CALUDE_digit_1234_is_4_l264_26451

/-- The decimal number constructed by concatenating integers from 1 to 499 -/
def x : ℚ :=
  sorry

/-- The nth digit of a rational number -/
def nthDigit (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_1234_is_4 : nthDigit x 1234 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_1234_is_4_l264_26451


namespace NUMINAMATH_CALUDE_smallest_prime_20_less_than_square_l264_26410

theorem smallest_prime_20_less_than_square : 
  ∃ (n : ℕ), 
    5 = n^2 - 20 ∧ 
    Prime 5 ∧ 
    (∀ (m : ℕ) (p : ℕ), p < 5 → p = m^2 - 20 → ¬ Prime p) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_20_less_than_square_l264_26410


namespace NUMINAMATH_CALUDE_power_difference_evaluation_l264_26492

theorem power_difference_evaluation : (3^3)^4 - (4^4)^3 = -16245775 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_evaluation_l264_26492


namespace NUMINAMATH_CALUDE_total_cost_is_985_l264_26499

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.75

/-- The additional cost of a train ride compared to a bus ride -/
def train_additional_cost : ℝ := 6.35

/-- The total cost of one train ride and one bus ride -/
def total_cost : ℝ := bus_cost + (bus_cost + train_additional_cost)

theorem total_cost_is_985 : total_cost = 9.85 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_985_l264_26499


namespace NUMINAMATH_CALUDE_sallys_out_of_pocket_cost_l264_26419

/-- The amount of money Sally needs to pay out of pocket to buy a reading book for each student -/
theorem sallys_out_of_pocket_cost 
  (budget : ℕ) 
  (book_cost : ℕ) 
  (num_students : ℕ) 
  (h1 : budget = 320)
  (h2 : book_cost = 12)
  (h3 : num_students = 30) :
  (book_cost * num_students - budget : ℕ) = 40 := by
  sorry

#check sallys_out_of_pocket_cost

end NUMINAMATH_CALUDE_sallys_out_of_pocket_cost_l264_26419


namespace NUMINAMATH_CALUDE_erica_ice_cream_weeks_l264_26459

/-- The number of weeks Erica buys ice cream -/
def ice_cream_weeks (orange_creamsicle_price : ℚ) 
                    (ice_cream_sandwich_price : ℚ)
                    (nutty_buddy_price : ℚ)
                    (total_spent : ℚ) : ℚ :=
  let weekly_spending := 3 * orange_creamsicle_price + 
                         2 * ice_cream_sandwich_price + 
                         2 * nutty_buddy_price
  total_spent / weekly_spending

/-- Theorem stating that Erica buys ice cream for 6 weeks -/
theorem erica_ice_cream_weeks : 
  ice_cream_weeks 2 1.5 3 90 = 6 := by
  sorry

end NUMINAMATH_CALUDE_erica_ice_cream_weeks_l264_26459


namespace NUMINAMATH_CALUDE_tabithas_initial_money_l264_26475

theorem tabithas_initial_money :
  ∀ (initial : ℚ) 
    (given_to_mom : ℚ) 
    (num_items : ℕ) 
    (item_cost : ℚ) 
    (money_left : ℚ),
  given_to_mom = 8 →
  num_items = 5 →
  item_cost = 1/2 →
  money_left = 6 →
  initial - given_to_mom = 2 * ((initial - given_to_mom) / 2 - num_items * item_cost - money_left) →
  initial = 25 := by
sorry

end NUMINAMATH_CALUDE_tabithas_initial_money_l264_26475


namespace NUMINAMATH_CALUDE_mixed_doubles_count_l264_26466

/-- The number of ways to form a mixed doubles team -/
def mixedDoublesTeams (numMales : ℕ) (numFemales : ℕ) : ℕ :=
  numMales * numFemales

/-- Theorem: The number of ways to form a mixed doubles team
    with 5 male players and 4 female players is 20 -/
theorem mixed_doubles_count :
  mixedDoublesTeams 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mixed_doubles_count_l264_26466


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l264_26491

theorem rectangle_perimeter (length width : ℕ+) : 
  length * width = 24 → 2 * (length + width) ≠ 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l264_26491


namespace NUMINAMATH_CALUDE_max_abs_z_value_l264_26441

/-- Given complex numbers a, b, c, z satisfying the conditions, 
    the maximum value of |z| is 1 + √2 -/
theorem max_abs_z_value (a b c z : ℂ) (r : ℝ) 
  (hr : r > 0)
  (ha : Complex.abs a = r)
  (hb : Complex.abs b = 2*r)
  (hc : Complex.abs c = r)
  (heq : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_value_l264_26441


namespace NUMINAMATH_CALUDE_find_n_l264_26460

theorem find_n : ∃ n : ℕ, (1/5 : ℝ)^n * (1/4 : ℝ)^18 = 1/(2*(10^35)) ∧ n = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l264_26460


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_150_l264_26471

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_odd_number_of_factors (n : ℕ) : Prop := is_perfect_square n

theorem greatest_number_with_odd_factors_under_150 :
  ∃ n : ℕ, n < 150 ∧ has_odd_number_of_factors n ∧
  ∀ m : ℕ, m < 150 → has_odd_number_of_factors m → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_150_l264_26471


namespace NUMINAMATH_CALUDE_square_area_probability_l264_26420

/-- The probability of a randomly chosen point on a line segment of length 12
    forming a square with an area between 36 and 81 -/
theorem square_area_probability : ∀ (AB : ℝ) (lower upper : ℝ),
  AB = 12 →
  lower = 6 →
  upper = 9 →
  (upper - lower) / AB = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_square_area_probability_l264_26420


namespace NUMINAMATH_CALUDE_work_completion_time_l264_26474

theorem work_completion_time 
  (total_work : ℕ) 
  (initial_men : ℕ) 
  (remaining_men : ℕ) 
  (remaining_days : ℕ) :
  initial_men = 100 →
  remaining_men = 50 →
  remaining_days = 40 →
  total_work = remaining_men * remaining_days →
  total_work = initial_men * (total_work / initial_men) →
  total_work / initial_men = 20 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l264_26474


namespace NUMINAMATH_CALUDE_arun_weight_average_l264_26430

theorem arun_weight_average :
  let min_weight := 61
  let max_weight := 64
  let average := (min_weight + max_weight) / 2
  (∀ w, min_weight < w ∧ w ≤ max_weight → 
    w > 60 ∧ w < 70 ∧ w > 61 ∧ w < 72 ∧ w ≤ 64) →
  average = 62.5 := by
sorry

end NUMINAMATH_CALUDE_arun_weight_average_l264_26430


namespace NUMINAMATH_CALUDE_two_digit_perfect_squares_divisible_by_four_l264_26449

theorem two_digit_perfect_squares_divisible_by_four :
  (∃! (s : Finset ℕ), ∀ n, n ∈ s ↔ 
    (10 ≤ n^2 ∧ n^2 ≤ 99) ∧ 4 ∣ n^2) ∧
  (∃ s, (∀ n, n ∈ s ↔ 
    (10 ≤ n^2 ∧ n^2 ≤ 99) ∧ 4 ∣ n^2) ∧ 
    Finset.card s = 3) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_perfect_squares_divisible_by_four_l264_26449


namespace NUMINAMATH_CALUDE_second_quadrant_and_modulus_condition_l264_26446

def complex_i : ℂ := Complex.I

theorem second_quadrant_and_modulus_condition (a : ℝ) : 
  let z₁ : ℂ := a + 2 / (1 - complex_i)
  let z₂ : ℂ := a - complex_i
  (z₁.re < 0 ∧ z₁.im > 0) → Complex.abs z₂ = 2 → a = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_and_modulus_condition_l264_26446


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l264_26424

/-- A parabola in the cartesian coordinate plane with equation y^2 = -16x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  h : equation = fun x y ↦ y^2 = -16*x

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Theorem: The coordinates of the focus of the parabola y^2 = -16x are (-4, 0) -/
theorem parabola_focus_coordinates (p : Parabola) : focus p = (-4, 0) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l264_26424


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l264_26455

structure Ball :=
  (color : String)

def Bag := List Ball

def draw (bag : Bag) : Ball × Bag :=
  match bag with
  | [] => ⟨Ball.mk "empty", []⟩
  | (b::bs) => ⟨b, bs⟩

def Event := Bag → Prop

def mutuallyExclusive (e1 e2 : Event) : Prop :=
  ∀ bag, ¬(e1 bag ∧ e2 bag)

def complementary (e1 e2 : Event) : Prop :=
  ∀ bag, e1 bag ↔ ¬(e2 bag)

def initialBag : Bag :=
  [Ball.mk "red", Ball.mk "blue", Ball.mk "black", Ball.mk "white"]

def ADrawsWhite : Event :=
  λ bag => (draw bag).1.color = "white"

def BDrawsWhite : Event :=
  λ bag => let (_, remainingBag) := draw bag
           (draw remainingBag).1.color = "white"

theorem events_mutually_exclusive_but_not_complementary :
  mutuallyExclusive ADrawsWhite BDrawsWhite ∧
  ¬(complementary ADrawsWhite BDrawsWhite) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l264_26455


namespace NUMINAMATH_CALUDE_fraction_value_at_two_l264_26469

theorem fraction_value_at_two :
  let f (x : ℝ) := (x^10 + 20*x^5 + 100) / (x^5 + 10)
  f 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_two_l264_26469


namespace NUMINAMATH_CALUDE_only_one_and_four_are_propositions_l264_26426

-- Define a type for the statements
inductive Statement
  | EmptySetSubset
  | GreaterThanImplication
  | IsThreeGreaterThanOne
  | NonIntersectingLinesParallel

-- Define a function to check if a statement is a proposition
def isProposition (s : Statement) : Prop :=
  match s with
  | Statement.EmptySetSubset => True
  | Statement.GreaterThanImplication => False
  | Statement.IsThreeGreaterThanOne => False
  | Statement.NonIntersectingLinesParallel => True

-- Theorem stating that only statements ① and ④ are propositions
theorem only_one_and_four_are_propositions :
  (∀ s : Statement, isProposition s ↔ (s = Statement.EmptySetSubset ∨ s = Statement.NonIntersectingLinesParallel)) :=
by sorry

end NUMINAMATH_CALUDE_only_one_and_four_are_propositions_l264_26426


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l264_26461

theorem sqrt_product_equality : 3 * Real.sqrt 2 * (2 * Real.sqrt 3) = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l264_26461


namespace NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l264_26486

/-- The decimal representation of 3/11 as a sequence of digits -/
def decimalRep : ℕ → Fin 10
  | 0 => 2
  | 1 => 7
  | n + 2 => decimalRep n

/-- The period of the decimal representation of 3/11 -/
def period : ℕ := 2

/-- Count of digit 2 in one period of the decimal representation -/
def countOfTwo : ℕ := 1

theorem probability_of_two_in_three_elevenths :
  (countOfTwo : ℚ) / (period : ℚ) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l264_26486


namespace NUMINAMATH_CALUDE_smallest_value_abcd_l264_26447

theorem smallest_value_abcd (a b c d : ℤ) 
  (sum_condition : a + b + c + d < 25)
  (a_condition : a > 8)
  (b_condition : b < 5)
  (c_odd : c % 2 = 1)
  (d_even : d % 2 = 0) :
  (∀ a' b' c' d' : ℤ, 
    a' + b' + c' + d' < 25 → 
    a' > 8 → 
    b' < 5 → 
    c' % 2 = 1 → 
    d' % 2 = 0 → 
    a' - b' + c' - d' ≥ a - b + c - d) →
  a - b + c - d = -4 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_abcd_l264_26447


namespace NUMINAMATH_CALUDE_f_parity_l264_26435

-- Define the function f(x) = x|x| + px^2
def f (p : ℝ) (x : ℝ) : ℝ := x * abs x + p * x^2

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem stating the parity of the function depends on p
theorem f_parity (p : ℝ) :
  (p = 0 → is_odd (f p)) ∧
  (p ≠ 0 → ¬(is_even (f p)) ∧ ¬(is_odd (f p))) :=
sorry

end NUMINAMATH_CALUDE_f_parity_l264_26435


namespace NUMINAMATH_CALUDE_cost_of_traveling_specific_roads_l264_26428

/-- Calculates the cost of traveling two intersecting roads on a rectangular lawn. -/
def cost_of_traveling_roads (lawn_length lawn_width road_width cost_per_sqm : ℝ) : ℝ :=
  let road1_area := road_width * lawn_width
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_road_area := road1_area + road2_area - intersection_area
  total_road_area * cost_per_sqm

/-- Theorem stating the cost of traveling two intersecting roads on a specific rectangular lawn. -/
theorem cost_of_traveling_specific_roads : 
  cost_of_traveling_roads 80 50 10 3 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_traveling_specific_roads_l264_26428


namespace NUMINAMATH_CALUDE_tangent_line_equation_l264_26401

-- Define the function f(x) = x^4 - 2x^3
def f (x : ℝ) : ℝ := x^4 - 2*x^3

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 4*x^3 - 6*x^2

-- Theorem: The equation of the tangent line to f(x) at x = 1 is y = -2x + 1
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -2*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l264_26401


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l264_26473

theorem simplify_and_evaluate (x : ℝ) : 
  (2*x + 1)^2 - (x + 3)*(x - 3) = 30 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l264_26473


namespace NUMINAMATH_CALUDE_expression_evaluation_l264_26403

theorem expression_evaluation (x y : ℝ) 
  (h : (3*x + 1)^2 + |y - 3| = 0) : 
  (x + 2*y) * (x - 2*y) + (x + 2*y)^2 - x * (2*x + 3*y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l264_26403


namespace NUMINAMATH_CALUDE_parallel_transitive_l264_26440

-- Define a type for lines in a plane
variable {Line : Type}

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem parallel_transitive (A B C : Line) :
  parallel A C → parallel B C → parallel A B :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_transitive_l264_26440


namespace NUMINAMATH_CALUDE_consecutive_points_segment_length_l264_26493

/-- Given 5 consecutive points on a straight line, prove the length of a specific segment -/
theorem consecutive_points_segment_length 
  (a b c d e : ℝ) -- Define points as real numbers
  (consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Consecutive points condition
  (bc_eq_3cd : c - b = 3 * (d - c)) -- bc = 3 cd
  (ab_eq_5 : b - a = 5) -- ab = 5
  (ac_eq_11 : c - a = 11) -- ac = 11
  (ae_eq_21 : e - a = 21) -- ae = 21
  : e - d = 8 := by -- de = 8
  sorry

end NUMINAMATH_CALUDE_consecutive_points_segment_length_l264_26493


namespace NUMINAMATH_CALUDE_projection_vector_proof_l264_26405

def line_direction : ℝ × ℝ := (3, 2)

theorem projection_vector_proof :
  ∃ (w : ℝ × ℝ), 
    w.1 + w.2 = 3 ∧ 
    w.1 * line_direction.1 + w.2 * line_direction.2 = 0 ∧
    w = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_projection_vector_proof_l264_26405


namespace NUMINAMATH_CALUDE_intersection_of_A_and_union_of_B_C_l264_26498

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem intersection_of_A_and_union_of_B_C : A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_union_of_B_C_l264_26498


namespace NUMINAMATH_CALUDE_firewood_sacks_filled_l264_26413

theorem firewood_sacks_filled (sack_capacity : ℕ) (father_wood : ℕ) (ranger_wood : ℕ) (worker_wood : ℕ) (num_workers : ℕ) :
  sack_capacity = 20 →
  father_wood = 80 →
  ranger_wood = 80 →
  worker_wood = 120 →
  num_workers = 2 →
  (father_wood + ranger_wood + num_workers * worker_wood) / sack_capacity = 20 :=
by sorry

end NUMINAMATH_CALUDE_firewood_sacks_filled_l264_26413


namespace NUMINAMATH_CALUDE_inverse_f_at_150_l264_26427

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^4 + 6

-- State the theorem
theorem inverse_f_at_150 :
  ∃ (y : ℝ), f y = 150 ∧ y = (48 : ℝ)^(1/4) :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_150_l264_26427


namespace NUMINAMATH_CALUDE_triangle_sin_c_max_l264_26452

/-- Given a triangle ABC with sides a, b, c, prove that if the dot products of its vectors
    satisfy the given condition, then sin C is less than or equal to √7/3 -/
theorem triangle_sin_c_max (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_dot_product : b * c * (b^2 + c^2 - a^2) + 2 * a * c * (a^2 + c^2 - b^2) = 
                   3 * a * b * (a^2 + b^2 - c^2)) :
  Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) ≤ Real.sqrt 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sin_c_max_l264_26452


namespace NUMINAMATH_CALUDE_puppies_percentage_proof_l264_26417

/-- The percentage of students who have puppies in Professor Plum's biology class -/
def percentage_with_puppies : ℝ := 80

theorem puppies_percentage_proof (total_students : ℕ) (both_puppies_parrots : ℕ) 
  (h1 : total_students = 40)
  (h2 : both_puppies_parrots = 8)
  (h3 : (25 : ℝ) / 100 * (percentage_with_puppies / 100 * total_students) = both_puppies_parrots) :
  percentage_with_puppies = 80 := by
  sorry

#check puppies_percentage_proof

end NUMINAMATH_CALUDE_puppies_percentage_proof_l264_26417


namespace NUMINAMATH_CALUDE_new_girl_weight_l264_26443

/-- Given a group of 25 girls, if replacing a 55 kg girl with a new girl increases
    the average weight by 1 kg, then the new girl weighs 80 kg. -/
theorem new_girl_weight (W : ℝ) (x : ℝ) : 
  (W / 25 + 1 = (W - 55 + x) / 25) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_new_girl_weight_l264_26443


namespace NUMINAMATH_CALUDE_planks_per_table_is_15_l264_26400

def planks_per_table (trees : ℕ) (planks_per_tree : ℕ) (table_price : ℕ) (labor_cost : ℕ) (profit : ℕ) : ℕ :=
  let total_planks := trees * planks_per_tree
  let total_revenue := profit + labor_cost
  let num_tables := total_revenue / table_price
  total_planks / num_tables

theorem planks_per_table_is_15 :
  planks_per_table 30 25 300 3000 12000 = 15 := by
  sorry

end NUMINAMATH_CALUDE_planks_per_table_is_15_l264_26400


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l264_26402

theorem diophantine_equation_solutions (x y z : ℤ) :
  (x * y / z : ℚ) + (y * z / x : ℚ) + (z * x / y : ℚ) = 3 ↔
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 1 ∧ y = -1 ∧ z = -1) ∨
  (x = -1 ∧ y = 1 ∧ z = -1) ∨
  (x = -1 ∧ y = -1 ∧ z = 1) := by
  sorry

#check diophantine_equation_solutions

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l264_26402


namespace NUMINAMATH_CALUDE_balance_after_six_months_l264_26425

/-- Calculates the balance after two quarters of compound interest -/
def balance_after_two_quarters (initial_deposit : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let balance_after_first_quarter := initial_deposit * (1 + rate1)
  balance_after_first_quarter * (1 + rate2)

/-- Theorem stating the balance after two quarters with given initial deposit and interest rates -/
theorem balance_after_six_months 
  (initial_deposit : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (h1 : initial_deposit = 5000)
  (h2 : rate1 = 0.07)
  (h3 : rate2 = 0.085) :
  balance_after_two_quarters initial_deposit rate1 rate2 = 5804.25 := by
  sorry

#eval balance_after_two_quarters 5000 0.07 0.085

end NUMINAMATH_CALUDE_balance_after_six_months_l264_26425


namespace NUMINAMATH_CALUDE_daughter_normal_probability_l264_26488

-- Define the inheritance types
inductive InheritanceType
| XLinked
| Autosomal

-- Define the phenotypes
inductive Phenotype
| Normal
| Affected

-- Define the genotypes
structure Genotype where
  hemophilia : Bool  -- true if carrier
  phenylketonuria : Bool  -- true if carrier

-- Define the parents
structure Parents where
  mother : Genotype
  father : Genotype

-- Define the conditions
def conditions (parents : Parents) : Prop :=
  (InheritanceType.XLinked = InheritanceType.XLinked) ∧  -- Hemophilia is X-linked
  (InheritanceType.Autosomal = InheritanceType.Autosomal) ∧  -- Phenylketonuria is autosomal
  (parents.mother.hemophilia = true) ∧  -- Mother is carrier for hemophilia
  (parents.father.hemophilia = false) ∧  -- Father is not affected by hemophilia
  (parents.mother.phenylketonuria = true) ∧  -- Mother is carrier for phenylketonuria
  (parents.father.phenylketonuria = true)  -- Father is carrier for phenylketonuria

-- Define the probability of a daughter being phenotypically normal
def prob_normal_daughter (parents : Parents) : ℚ :=
  3 / 4

-- The theorem to prove
theorem daughter_normal_probability (parents : Parents) :
  conditions parents → prob_normal_daughter parents = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_daughter_normal_probability_l264_26488


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l264_26470

theorem unique_modular_congruence : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -215 ≡ n [ZMOD 23] ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l264_26470


namespace NUMINAMATH_CALUDE_sibling_pizza_order_l264_26433

-- Define the siblings
inductive Sibling
| Alex
| Beth
| Cyril
| Dan
| Emma

-- Define the function that returns the fraction of pizza eaten by each sibling
def pizza_fraction (s : Sibling) : ℚ :=
  match s with
  | Sibling.Alex => 1/6
  | Sibling.Beth => 1/4
  | Sibling.Cyril => 1/5
  | Sibling.Dan => 1/3
  | Sibling.Emma => 1 - (1/6 + 1/4 + 1/5 + 1/3)

-- Define the order of siblings
def sibling_order : List Sibling :=
  [Sibling.Dan, Sibling.Beth, Sibling.Cyril, Sibling.Alex, Sibling.Emma]

-- Theorem statement
theorem sibling_pizza_order : 
  List.Pairwise (λ a b => pizza_fraction a > pizza_fraction b) sibling_order :=
sorry

end NUMINAMATH_CALUDE_sibling_pizza_order_l264_26433


namespace NUMINAMATH_CALUDE_tunnel_length_l264_26487

/-- Calculates the length of a tunnel given train and travel parameters -/
theorem tunnel_length
  (train_length : Real)
  (train_speed : Real)
  (exit_time : Real)
  (h1 : train_length = 1.5)
  (h2 : train_speed = 45)
  (h3 : exit_time = 4 / 60) :
  train_speed * exit_time - train_length = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_l264_26487


namespace NUMINAMATH_CALUDE_sinusoid_amplitude_l264_26415

theorem sinusoid_amplitude (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_sinusoid_amplitude_l264_26415


namespace NUMINAMATH_CALUDE_functions_strictly_greater_iff_no_leq_l264_26404

-- Define the functions f and g with domain ℝ
variable (f g : ℝ → ℝ)

-- State the theorem
theorem functions_strictly_greater_iff_no_leq :
  (∀ x : ℝ, f x > g x) ↔ ¬∃ x : ℝ, f x ≤ g x := by sorry

end NUMINAMATH_CALUDE_functions_strictly_greater_iff_no_leq_l264_26404


namespace NUMINAMATH_CALUDE_vectors_are_coplanar_l264_26448

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
  [Fact (finrank ℝ V = 3)]

def are_coplanar (v₁ v₂ v₃ : V) : Prop :=
  ∃ (a b c : ℝ), a • v₁ + b • v₂ + c • v₃ = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

theorem vectors_are_coplanar (a b : V) : are_coplanar a b (2 • a + 4 • b) := by
  sorry

end NUMINAMATH_CALUDE_vectors_are_coplanar_l264_26448


namespace NUMINAMATH_CALUDE_matching_shoe_probability_l264_26438

/-- The probability of selecting a matching pair of shoes from a box containing 6 pairs -/
theorem matching_shoe_probability (total_shoes : ℕ) (total_pairs : ℕ) (h1 : total_shoes = 12) (h2 : total_pairs = 6) :
  (total_pairs : ℚ) / ((total_shoes.choose 2) : ℚ) = 1 / 11 := by
  sorry

#check matching_shoe_probability

end NUMINAMATH_CALUDE_matching_shoe_probability_l264_26438


namespace NUMINAMATH_CALUDE_square_circles_intersection_area_l264_26462

/-- The area of intersection between a square and four circles --/
theorem square_circles_intersection_area :
  let square_side : ℝ := 8
  let circle_radius : ℝ := 3
  let square_area : ℝ := square_side ^ 2
  let circle_sector_area : ℝ := π * circle_radius ^ 2 / 4
  let total_sector_area : ℝ := 4 * circle_sector_area
  let triangle_area : ℝ := (square_side / 2 - circle_radius) ^ 2 / 2
  let total_triangle_area : ℝ := 4 * triangle_area
  let shaded_area : ℝ := square_area - (total_sector_area + total_triangle_area)
  shaded_area = 64 - 9 * π - 18 :=
by sorry

end NUMINAMATH_CALUDE_square_circles_intersection_area_l264_26462


namespace NUMINAMATH_CALUDE_mary_has_seven_balloons_l264_26476

-- Define the number of balloons for each person
def fred_balloons : ℕ := 5
def sam_balloons : ℕ := 6
def total_balloons : ℕ := 18

-- Define Mary's balloons as the difference between total and the sum of Fred's and Sam's
def mary_balloons : ℕ := total_balloons - (fred_balloons + sam_balloons)

-- Theorem to prove
theorem mary_has_seven_balloons : mary_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_seven_balloons_l264_26476


namespace NUMINAMATH_CALUDE_book_reading_fraction_l264_26412

theorem book_reading_fraction (total_pages : ℕ) (pages_read : ℕ) : 
  total_pages = 300 →
  pages_read = (total_pages - pages_read) + 100 →
  pages_read / total_pages = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_fraction_l264_26412


namespace NUMINAMATH_CALUDE_arctan_special_angle_combination_l264_26465

/-- Proves that arctan(tan 75° - 3tan 15° + tan 45°) = 30° --/
theorem arctan_special_angle_combination :
  Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (15 * π / 180) + Real.tan (45 * π / 180)) = 30 * π / 180 := by
  sorry

#check arctan_special_angle_combination

end NUMINAMATH_CALUDE_arctan_special_angle_combination_l264_26465


namespace NUMINAMATH_CALUDE_quadratic_expression_values_l264_26479

theorem quadratic_expression_values (a c : ℝ) : 
  (∀ x : ℝ, a * x^2 + x + c = 10 → x = 1) →
  (∀ x : ℝ, a * x^2 + x + c = 8 → x = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_values_l264_26479


namespace NUMINAMATH_CALUDE_sequence_relation_l264_26482

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

theorem sequence_relation (n : ℕ) : y n ^ 2 = 3 * x n ^ 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_relation_l264_26482


namespace NUMINAMATH_CALUDE_intersection_has_one_element_l264_26468

/-- The set A in ℝ² defined by the equation x^2 - 3xy + 4y^2 = 7/2 -/
def A : Set (ℝ × ℝ) := {p | p.1^2 - 3*p.1*p.2 + 4*p.2^2 = 7/2}

/-- The set B in ℝ² defined by the equation kx + y = 2, where k > 0 -/
def B (k : ℝ) : Set (ℝ × ℝ) := {p | k*p.1 + p.2 = 2}

/-- The theorem stating that when k = 1/4, the intersection of A and B has exactly one element -/
theorem intersection_has_one_element :
  ∃! p : ℝ × ℝ, p ∈ A ∩ B (1/4) :=
sorry

end NUMINAMATH_CALUDE_intersection_has_one_element_l264_26468


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l264_26494

def P : Set ℝ := {x | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem complement_P_intersect_Q : (Set.univ \ P) ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l264_26494


namespace NUMINAMATH_CALUDE_line_through_P_with_equal_intercepts_l264_26432

/-- A line in the 2D plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The point P(2,3) -/
def P : ℝ × ℝ := (2, 3)

/-- A line passes through a point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- A line has equal intercepts on both coordinate axes -/
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c/l.a = -l.c/l.b

/-- The equation of the line is y = 3/2 * x -/
def is_y_eq_3_2x (l : Line) : Prop :=
  l.a = 2 ∧ l.b = -3 ∧ l.c = 0

/-- The equation of the line is x + y - 5 = 0 -/
def is_x_plus_y_eq_5 (l : Line) : Prop :=
  l.a = 1 ∧ l.b = 1 ∧ l.c = -5

theorem line_through_P_with_equal_intercepts (l : Line) :
  passes_through l P ∧ has_equal_intercepts l →
  is_y_eq_3_2x l ∨ is_x_plus_y_eq_5 l := by
  sorry

end NUMINAMATH_CALUDE_line_through_P_with_equal_intercepts_l264_26432


namespace NUMINAMATH_CALUDE_evaluate_expression_l264_26497

theorem evaluate_expression : 6 - 8 * (5 - 2^3) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l264_26497


namespace NUMINAMATH_CALUDE_equation_solution_l264_26408

theorem equation_solution (x : ℝ) : 
  (x = (((-1 + Real.sqrt 21) / 2) ^ 3) ∨ x = (((-1 - Real.sqrt 21) / 2) ^ 3)) →
  6 * x^(1/3) - 3 * (x / x^(2/3)) + 2 * x^(2/3) = 10 + x^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l264_26408


namespace NUMINAMATH_CALUDE_ant_return_probability_2006_l264_26434

/-- A regular octahedron --/
structure Octahedron :=
  (vertices : Finset (Fin 6))
  (edges : Finset (Fin 6 × Fin 6))
  (is_regular : True)  -- This is a simplification; we're assuming it's regular

/-- An ant's position on the octahedron --/
structure AntPosition (O : Octahedron) :=
  (vertex : Fin 6)

/-- The probability distribution of the ant's position after n moves --/
def probability_distribution (O : Octahedron) (n : ℕ) : AntPosition O → ℝ := sorry

/-- The probability of the ant returning to the starting vertex after n moves --/
def return_probability (O : Octahedron) (n : ℕ) : ℝ := sorry

/-- The main theorem --/
theorem ant_return_probability_2006 (O : Octahedron) :
  return_probability O 2006 = (2^2005 + 1) / (3 * 2^2006) := by sorry

end NUMINAMATH_CALUDE_ant_return_probability_2006_l264_26434


namespace NUMINAMATH_CALUDE_rotten_oranges_percentage_l264_26489

/-- Proves that the percentage of rotten oranges is 15% given the conditions -/
theorem rotten_oranges_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_bananas_percentage : ℝ)
  (good_fruits_percentage : ℝ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_bananas_percentage = 6)
  (h4 : good_fruits_percentage = 88.6)
  : (100 - (good_fruits_percentage * (total_oranges + total_bananas) / total_oranges -
     rotten_bananas_percentage * total_bananas / total_oranges)) = 15 :=
by sorry

end NUMINAMATH_CALUDE_rotten_oranges_percentage_l264_26489


namespace NUMINAMATH_CALUDE_max_profit_at_45_l264_26454

/-- Represents the daily profit function for selling children's toys -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 1000 * x - 21000

/-- Represents the cost price of each toy -/
def cost_price : ℝ := 30

/-- Represents the maximum allowed profit margin -/
def max_profit_margin : ℝ := 0.5

/-- Represents the minimum selling price -/
def min_selling_price : ℝ := 35

/-- Represents the maximum selling price based on the profit margin constraint -/
def max_selling_price : ℝ := cost_price * (1 + max_profit_margin)

theorem max_profit_at_45 :
  ∃ (max_profit : ℝ),
    (∀ x : ℝ, min_selling_price ≤ x ∧ x ≤ max_selling_price →
      profit_function x ≤ profit_function max_selling_price) ∧
    profit_function max_selling_price = max_profit ∧
    max_profit = 3750 := by
  sorry

#eval profit_function max_selling_price

end NUMINAMATH_CALUDE_max_profit_at_45_l264_26454


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l264_26496

/-- A right cone with a sphere inscribed inside it. -/
structure InscribedSphere where
  /-- The base radius of the cone in cm. -/
  base_radius : ℝ
  /-- The height of the cone in cm. -/
  cone_height : ℝ
  /-- The radius of the inscribed sphere in cm. -/
  sphere_radius : ℝ
  /-- The base radius is 9 cm. -/
  base_radius_eq : base_radius = 9
  /-- The cone height is 27 cm. -/
  cone_height_eq : cone_height = 27
  /-- The sphere is inscribed in the cone. -/
  inscribed : sphere_radius ≤ base_radius ∧ sphere_radius ≤ cone_height

/-- The radius of the inscribed sphere is 3√10 - 3 cm. -/
theorem inscribed_sphere_radius (s : InscribedSphere) : 
  s.sphere_radius = 3 * Real.sqrt 10 - 3 := by
  sorry

#check inscribed_sphere_radius

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l264_26496


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l264_26444

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, -9)
  parallel a b → x = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l264_26444


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l264_26463

theorem jelly_bean_problem (b c : ℕ) : 
  b = 3 * c →                  -- Initial condition: 3 times as many blueberry as cherry
  b - 15 = 4 * (c - 15) →      -- Condition after eating 15 of each
  b = 135 :=                   -- Conclusion: original number of blueberry jelly beans
by sorry

end NUMINAMATH_CALUDE_jelly_bean_problem_l264_26463


namespace NUMINAMATH_CALUDE_investment_problem_l264_26445

/-- Proves that the amount invested in the first account is approximately $2336.36 --/
theorem investment_problem (total_interest : ℝ) (second_account_investment : ℝ) 
  (interest_rate_difference : ℝ) (first_account_rate : ℝ) :
  total_interest = 1282 →
  second_account_investment = 8200 →
  interest_rate_difference = 0.015 →
  first_account_rate = 0.11 →
  ∃ x : ℝ, (x * first_account_rate + 
    second_account_investment * (first_account_rate + interest_rate_difference) = total_interest) ∧ 
    (abs (x - 2336.36) < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l264_26445


namespace NUMINAMATH_CALUDE_square_sum_of_sqrt3_plus_minus_2_l264_26458

theorem square_sum_of_sqrt3_plus_minus_2 :
  let a : ℝ := Real.sqrt 3 + 2
  let b : ℝ := Real.sqrt 3 - 2
  a^2 + b^2 = 14 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_sqrt3_plus_minus_2_l264_26458


namespace NUMINAMATH_CALUDE_function_characterization_l264_26485

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, is_perfect_square (f (f a - b) + b * f (2 * a))

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def solution_type_1 (f : ℤ → ℤ) : Prop :=
  (∀ n : ℤ, is_even n → f n = 0) ∧
  (∀ n : ℤ, ¬is_even n → is_perfect_square (f n))

def solution_type_2 (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = n * n

theorem function_characterization (f : ℤ → ℤ) :
  satisfies_condition f → solution_type_1 f ∨ solution_type_2 f :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l264_26485


namespace NUMINAMATH_CALUDE_prime_squares_congruence_l264_26439

theorem prime_squares_congruence (p : ℕ) (hp : Prime p) :
  (∀ a : ℕ, ¬(p ∣ a) → a^2 % p = 1) → p = 2 ∨ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_squares_congruence_l264_26439


namespace NUMINAMATH_CALUDE_simplify_expression_l264_26495

theorem simplify_expression : 
  ((5^2010)^2 - (5^2008)^2) / ((5^2009)^2 - (5^2007)^2) = 25 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l264_26495


namespace NUMINAMATH_CALUDE_systematic_sampling_first_two_samples_l264_26467

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  population_size : Nat
  sample_size : Nat
  last_sampled : Nat

/-- Calculates the interval size for systematic sampling -/
def interval_size (s : SystematicSampling) : Nat :=
  s.population_size / s.sample_size

/-- Calculates the start of the last interval -/
def last_interval_start (s : SystematicSampling) : Nat :=
  s.population_size - (interval_size s) + 1

/-- Calculates the position within an interval -/
def position_in_interval (s : SystematicSampling) : Nat :=
  s.last_sampled - (last_interval_start s) + 1

/-- Calculates the nth sampled number -/
def nth_sample (s : SystematicSampling) (n : Nat) : Nat :=
  (n - 1) * (position_in_interval s)

theorem systematic_sampling_first_two_samples
  (s : SystematicSampling)
  (h1 : s.population_size = 8000)
  (h2 : s.sample_size = 50)
  (h3 : s.last_sampled = 7900) :
  (nth_sample s 1 = 60) ∧ (nth_sample s 2 = 220) := by
  sorry

#check systematic_sampling_first_two_samples

end NUMINAMATH_CALUDE_systematic_sampling_first_two_samples_l264_26467


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l264_26436

/-- An arithmetic sequence with a common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence -/
def arithmetic_term (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_nonconstant : d ≠ 0)
  (h_geom : ∃ r, 
    arithmetic_term (a 1) d 10 = r * arithmetic_term (a 1) d 5 ∧
    arithmetic_term (a 1) d 20 = r * arithmetic_term (a 1) d 10) :
  ∃ r, r = 2 ∧
    arithmetic_term (a 1) d 10 = r * arithmetic_term (a 1) d 5 ∧
    arithmetic_term (a 1) d 20 = r * arithmetic_term (a 1) d 10 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l264_26436


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l264_26478

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  (2 * t.a = t.b + t.c) ∧ 
  ((Real.sin t.A)^2 = Real.sin t.B * Real.sin t.C)

-- Define what it means for a triangle to be equilateral
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- The theorem to be proved
theorem triangle_is_equilateral (t : Triangle) 
  (h : TriangleProperties t) : IsEquilateral t := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l264_26478


namespace NUMINAMATH_CALUDE_percent_change_condition_l264_26480

theorem percent_change_condition (p q N : ℝ) 
  (hp : p > 0) (hq : q > 0) (hN : N > 0) (hq_bound : q < 50) :
  N * (1 + 3 * p / 100) * (1 - 2 * q / 100) > N ↔ p > 100 * q / (147 - 3 * q) :=
sorry

end NUMINAMATH_CALUDE_percent_change_condition_l264_26480


namespace NUMINAMATH_CALUDE_num_sequences_l264_26483

/-- The number of distinct elements in the set -/
def num_elements : ℕ := 5

/-- The length of the sequences to be formed -/
def sequence_length : ℕ := 4

/-- The minimum number of times each element appears -/
def min_appearances : ℕ := 3

/-- Theorem stating the number of possible sequences -/
theorem num_sequences (h : min_appearances ≥ sequence_length) :
  num_elements ^ sequence_length = 625 := by sorry

end NUMINAMATH_CALUDE_num_sequences_l264_26483


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_s_max_min_l264_26431

theorem sum_of_reciprocals_of_s_max_min (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) : 
  let s := x^2 + y^2
  ∃ (s_max s_min : ℝ), (∀ (x' y' : ℝ), 4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5 → x'^2 + y'^2 ≤ s_max) ∧
                       (∀ (x' y' : ℝ), 4 * x'^2 - 5 * x' * y' + 4 * y'^2 = 5 → s_min ≤ x'^2 + y'^2) ∧
                       (1 / s_max + 1 / s_min = 8 / 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_s_max_min_l264_26431


namespace NUMINAMATH_CALUDE_perimeter_is_158_l264_26453

/-- Represents a tile in the figure -/
structure Tile where
  width : ℕ := 2
  height : ℕ := 4

/-- Represents the figure composed of tiles -/
structure Figure where
  tiles : List Tile
  horizontalEdges : ℕ := 45
  verticalEdges : ℕ := 34

/-- Calculates the perimeter of the figure -/
def calculatePerimeter (f : Figure) : ℕ :=
  (f.horizontalEdges + f.verticalEdges) * 2

/-- Theorem stating that the perimeter of the specific figure is 158 -/
theorem perimeter_is_158 (f : Figure) : calculatePerimeter f = 158 := by
  sorry

#check perimeter_is_158

end NUMINAMATH_CALUDE_perimeter_is_158_l264_26453


namespace NUMINAMATH_CALUDE_log_sum_equality_l264_26423

theorem log_sum_equality : 
  Real.sqrt (Real.log 8 / Real.log 4 + Real.log 10 / Real.log 5) = 
  Real.sqrt (5 / 2 + Real.log 2 / Real.log 5) := by
sorry

end NUMINAMATH_CALUDE_log_sum_equality_l264_26423


namespace NUMINAMATH_CALUDE_inequality_proof_l264_26414

theorem inequality_proof (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  abs a + abs b + abs c ≥ 3 * Real.sqrt 3 * (b^2 * c^2 + c^2 * a^2 + a^2 * b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l264_26414


namespace NUMINAMATH_CALUDE_circle_center_proof_l264_26442

/-- The equation of a circle in polar coordinates -/
def polar_circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The center of a circle in polar coordinates -/
def circle_center : ℝ × ℝ := (1, 0)

/-- Theorem stating that the center of the circle ρ = 2cosθ is at (1, 0) in polar coordinates -/
theorem circle_center_proof :
  ∀ ρ θ : ℝ, polar_circle_equation ρ θ → circle_center = (1, 0) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_center_proof_l264_26442


namespace NUMINAMATH_CALUDE_hotdogs_served_during_dinner_l264_26409

/-- The number of hot dogs served during lunch today -/
def lunch_hotdogs : ℕ := 9

/-- The total number of hot dogs served today -/
def total_hotdogs : ℕ := 11

/-- The number of hot dogs served during dinner today -/
def dinner_hotdogs : ℕ := total_hotdogs - lunch_hotdogs

theorem hotdogs_served_during_dinner : dinner_hotdogs = 2 := by
  sorry

end NUMINAMATH_CALUDE_hotdogs_served_during_dinner_l264_26409


namespace NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l264_26484

theorem sqrt_sum_greater_than_sqrt_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l264_26484


namespace NUMINAMATH_CALUDE_cheryl_leftover_material_l264_26457

theorem cheryl_leftover_material :
  let material_type1 : ℚ := 2/9
  let material_type2 : ℚ := 1/8
  let total_bought : ℚ := material_type1 + material_type2
  let material_used : ℚ := 0.125
  let material_leftover : ℚ := total_bought - material_used
  material_leftover = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_leftover_material_l264_26457


namespace NUMINAMATH_CALUDE_possible_values_of_a_l264_26472

theorem possible_values_of_a (x y a : ℝ) :
  (|3 * y - 18| + |a * x - y| = 0) →
  (x > 0) →
  (∃ n : ℕ, x = 2 * n) →
  (x ≤ y) →
  (a = 3 ∨ a = 3/2 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l264_26472


namespace NUMINAMATH_CALUDE_circle_centers_line_l264_26422

/-- The set of circles C_k defined by (x-k+1)^2 + (y-3k)^2 = 2k^4 where k is a positive integer -/
def CircleSet (k : ℕ+) (x y : ℝ) : Prop :=
  (x - k + 1)^2 + (y - 3*k)^2 = 2 * k^4

/-- The center of circle C_k -/
def CircleCenter (k : ℕ+) : ℝ × ℝ := (k - 1, 3*k)

/-- The line on which the centers lie -/
def CenterLine (x y : ℝ) : Prop := y = 3*(x + 1) ∧ x ≠ -1

/-- Theorem: If the centers of the circles C_k lie on a fixed line,
    then that line is y = 3(x+1) where x ≠ -1 -/
theorem circle_centers_line :
  (∀ k : ℕ+, ∃ x y : ℝ, CircleCenter k = (x, y) ∧ CenterLine x y) →
  ∀ x y : ℝ, (∃ k : ℕ+, CircleCenter k = (x, y)) → CenterLine x y :=
sorry

end NUMINAMATH_CALUDE_circle_centers_line_l264_26422


namespace NUMINAMATH_CALUDE_novel_writing_rate_l264_26437

/-- Represents the writing rate of an author -/
def writing_rate (total_words : ℕ) (writing_hours : ℕ) : ℚ :=
  total_words / writing_hours

/-- Proves that the writing rate for a 60,000-word novel written in 100 hours is 600 words per hour -/
theorem novel_writing_rate :
  writing_rate 60000 100 = 600 := by
  sorry

end NUMINAMATH_CALUDE_novel_writing_rate_l264_26437


namespace NUMINAMATH_CALUDE_constant_product_l264_26477

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Define a point on the right branch of the hyperbola
def right_branch (x y : ℝ) : Prop := hyperbola x y ∧ x > 0

-- Define a line passing through a point
def line_through (x₀ y₀ x y : ℝ) : Prop := ∃ (m b : ℝ), y - y₀ = m * (x - x₀) ∧ y = m*x + b

-- Define the midpoint condition
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop := x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

-- Main theorem
theorem constant_product (x₀ y₀ x_A y_A x_B y_B : ℝ) :
  right_branch x₀ y₀ →
  asymptote x_A y_A ∧ asymptote x_B y_B →
  line_through x₀ y₀ x_A y_A ∧ line_through x₀ y₀ x_B y_B →
  is_midpoint x₀ y₀ x_A y_A x_B y_B →
  (x_A^2 + y_A^2) * (x_B^2 + y_B^2) = 25 :=
sorry

end NUMINAMATH_CALUDE_constant_product_l264_26477
