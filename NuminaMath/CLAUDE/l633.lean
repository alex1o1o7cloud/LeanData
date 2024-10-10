import Mathlib

namespace equation_solution_l633_63383

theorem equation_solution (b : ℝ) (hb : b ≠ 0) :
  (0 : ℝ)^2 + 9*b^2 = (3*b - 0)^2 :=
by sorry

end equation_solution_l633_63383


namespace max_value_of_f_in_interval_l633_63310

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- Define the interval
def interval : Set ℝ := Set.Icc (-1) 1

-- Theorem statement
theorem max_value_of_f_in_interval :
  ∃ (m : ℝ), m = 2 ∧ ∀ x ∈ interval, f x ≤ m :=
sorry

end max_value_of_f_in_interval_l633_63310


namespace jeans_discount_percentage_l633_63311

theorem jeans_discount_percentage (original_price : ℝ) (coupon : ℝ) (card_discount : ℝ) (total_savings : ℝ) :
  original_price = 125 →
  coupon = 10 →
  card_discount = 0.1 →
  total_savings = 44 →
  ∃ (sale_discount : ℝ),
    sale_discount = 0.2 ∧
    (original_price - sale_discount * original_price - coupon) * (1 - card_discount) = original_price - total_savings :=
by sorry

end jeans_discount_percentage_l633_63311


namespace cone_base_radius_l633_63353

/-- Given a circle of radius 16 divided into 4 equal parts, if one part forms the lateral surface of a cone, then the radius of the cone's base is 4. -/
theorem cone_base_radius (r : ℝ) (h1 : r = 16) (h2 : r > 0) : 
  (2 * Real.pi * r) / 4 = 2 * Real.pi * 4 := by
  sorry

end cone_base_radius_l633_63353


namespace math_expressions_equality_l633_63318

theorem math_expressions_equality : 
  (∃ (a b c d : ℝ), 
    a = (Real.sqrt 5 - (Real.sqrt 3 + Real.sqrt 15) / (Real.sqrt 6 * Real.sqrt 2)) ∧
    b = ((Real.sqrt 48 - 4 * Real.sqrt (1/8)) - (3 * Real.sqrt (1/3) - 2 * Real.sqrt 0.5)) ∧
    c = ((3 + Real.sqrt 5) * (3 - Real.sqrt 5) - (Real.sqrt 3 - 1)^2) ∧
    d = ((- Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) - Real.sqrt ((-3)^2) + 1 / (2 - Real.sqrt 5)) ∧
    a = -1 ∧
    b = 3 * Real.sqrt 3 ∧
    c = 2 * Real.sqrt 3 ∧
    d = -3 - Real.sqrt 5) := by
  sorry

#check math_expressions_equality

end math_expressions_equality_l633_63318


namespace spelling_contest_total_questions_l633_63362

/-- Represents a participant in the spelling contest -/
structure Participant where
  name : String
  round1_correct : ℕ
  round1_wrong : ℕ
  round2_correct : ℕ
  round2_wrong : ℕ
  round3_correct : ℕ
  round3_wrong : ℕ

/-- Calculates the total number of questions for a participant -/
def totalQuestions (p : Participant) : ℕ :=
  p.round1_correct + p.round1_wrong +
  p.round2_correct + p.round2_wrong +
  p.round3_correct + p.round3_wrong

/-- The spelling contest -/
def spellingContest : Prop :=
  let drew : Participant := {
    name := "Drew"
    round1_correct := 20
    round1_wrong := 6
    round2_correct := 24
    round2_wrong := 9
    round3_correct := 28
    round3_wrong := 14
  }
  let carla : Participant := {
    name := "Carla"
    round1_correct := 14
    round1_wrong := 2 * drew.round1_wrong
    round2_correct := 21
    round2_wrong := 8
    round3_correct := 22
    round3_wrong := 10
  }
  let blake : Participant := {
    name := "Blake"
    round1_correct := 0
    round1_wrong := 0
    round2_correct := 18
    round2_wrong := 11
    round3_correct := 15
    round3_wrong := 16
  }
  
  -- Conditions
  (∀ p : Participant, (p.round1_correct : ℚ) / (p.round1_correct + p.round1_wrong) ≥ 0.7) ∧
  (∀ p : Participant, (p.round2_correct : ℚ) / (p.round2_correct + p.round2_wrong) ≥ 0.7) ∧
  (∀ p : Participant, (p.round3_correct : ℚ) / (p.round3_correct + p.round3_wrong) ≥ 0.7) ∧
  (∀ p : Participant, ((p.round1_correct + p.round2_correct) : ℚ) / (p.round1_correct + p.round1_wrong + p.round2_correct + p.round2_wrong) ≥ 0.75) ∧
  
  -- Theorem to prove
  (totalQuestions drew + totalQuestions carla + totalQuestions blake = 248)

theorem spelling_contest_total_questions : spellingContest := by sorry

end spelling_contest_total_questions_l633_63362


namespace function_identity_l633_63300

theorem function_identity (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x + Real.cos x ^ 2| ≤ 3/4)
  (h2 : ∀ x, |f x - Real.sin x ^ 2| ≤ 1/4) :
  ∀ x, f x = 1/2 - Real.sin x ^ 2 := by
  sorry

end function_identity_l633_63300


namespace remainder_theorem_l633_63340

theorem remainder_theorem (n : ℤ) (h : n % 13 = 3) : (5 * n - 11) % 13 = 4 := by
  sorry

end remainder_theorem_l633_63340


namespace triangle_angle_measure_l633_63338

theorem triangle_angle_measure (A B C : Real) (a b c : Real) :
  A ∈ Set.Ioo 0 π →
  B ∈ Set.Ioo 0 π →
  b * Real.sin A + a * Real.cos B = 0 →
  B = 3 * π / 4 := by
  sorry

end triangle_angle_measure_l633_63338


namespace special_numbers_theorem_l633_63345

def is_special_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 1000 * a + 100 * (b - a) + n % 100 ∧ 
                10 ≤ a ∧ a < 100 ∧ 
                0 ≤ b - a ∧ b - a < 100 ∧
                n = (a + (n % 100))^2

theorem special_numbers_theorem : 
  {n : ℕ | is_special_number n} = {3025, 2025, 9801} := by sorry

end special_numbers_theorem_l633_63345


namespace square_of_sum_l633_63315

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by
  sorry

end square_of_sum_l633_63315


namespace middle_legs_arrangements_adjacent_legs_arrangements_l633_63366

/-- The number of athletes -/
def total_athletes : ℕ := 6

/-- The number of athletes needed for the relay -/
def relay_size : ℕ := 4

/-- The number of ways to arrange n items taken r at a time -/
def permutations (n r : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - r))

/-- The number of ways to choose r items from n items -/
def combinations (n r : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

/-- Theorem for the number of arrangements with A and B running the middle two legs -/
theorem middle_legs_arrangements : 
  permutations 2 2 * permutations (total_athletes - 2) (relay_size - 2) = 24 := by sorry

/-- Theorem for the number of arrangements with A and B running adjacent legs -/
theorem adjacent_legs_arrangements : 
  permutations 2 2 * combinations (total_athletes - 2) (relay_size - 2) * permutations 3 3 = 72 := by sorry

end middle_legs_arrangements_adjacent_legs_arrangements_l633_63366


namespace modular_arithmetic_problem_l633_63341

theorem modular_arithmetic_problem :
  ∃ (x y : ℤ), 
    (7 * x ≡ 1 [ZMOD 56]) ∧ 
    (13 * y ≡ 1 [ZMOD 56]) ∧ 
    (3 * x + 9 * y ≡ 39 [ZMOD 56]) ∧ 
    (0 ≤ (3 * x + 9 * y) % 56) ∧ 
    ((3 * x + 9 * y) % 56 < 56) := by
  sorry

end modular_arithmetic_problem_l633_63341


namespace smallest_class_size_l633_63396

theorem smallest_class_size :
  ∀ n : ℕ,
  (∃ x : ℕ, n = 5 * x + 2 ∧ x > 0) →
  n > 30 →
  n ≥ 32 :=
by
  sorry

end smallest_class_size_l633_63396


namespace planes_parallel_if_perp_to_same_line_l633_63337

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- The main theorem
theorem planes_parallel_if_perp_to_same_line 
  (a : Line) (α β : Plane) 
  (h_diff : α ≠ β) 
  (h_perp_α : perp a α) 
  (h_perp_β : perp a β) : 
  parallel α β :=
sorry

end planes_parallel_if_perp_to_same_line_l633_63337


namespace minimum_h_10_l633_63374

def IsTenuous (h : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, h x + h y > 2 * y.val ^ 2

def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem minimum_h_10 (h : ℕ+ → ℤ) 
  (tenuous : IsTenuous h) 
  (min_sum : ∀ g : ℕ+ → ℤ, IsTenuous g → SumH g ≥ SumH h) : 
  h ⟨10, Nat.succ_pos 9⟩ ≥ 137 := by
  sorry

end minimum_h_10_l633_63374


namespace f_order_l633_63316

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 6 * m * x + 2

-- State the theorem
theorem f_order (m : ℝ) (h : ∀ x, f m x = f m (-x)) :
  f m (-2) < f m 1 ∧ f m 1 < f m 0 := by
  sorry

end f_order_l633_63316


namespace computer_price_theorem_l633_63384

/-- The sticker price of the computer -/
def sticker_price : ℝ := 750

/-- The price at store A after discount and rebate -/
def price_A (x : ℝ) : ℝ := 0.8 * x - 100

/-- The price at store B after discount -/
def price_B (x : ℝ) : ℝ := 0.7 * x

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem computer_price_theorem :
  price_A sticker_price = price_B sticker_price - 25 :=
by sorry

end computer_price_theorem_l633_63384


namespace M_mod_51_l633_63334

def M : ℕ := sorry

theorem M_mod_51 : M % 51 = 15 := by sorry

end M_mod_51_l633_63334


namespace smallest_n_for_trig_inequality_l633_63358

theorem smallest_n_for_trig_inequality :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≤ 2^(1 - n)) ∧
  (∀ (m : ℕ), m > 0 → m < n →
    ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m > 2^(1 - m)) ∧
  n = 2 :=
sorry

end smallest_n_for_trig_inequality_l633_63358


namespace sequence_inequality_l633_63386

/-- The sequence a_n defined by n^2 + kn + 2 -/
def a (n : ℕ) (k : ℝ) : ℝ := n^2 + k * n + 2

/-- Theorem stating that if a_n ≥ a_4 for all n ≥ 4, then k is in [-9, -7] -/
theorem sequence_inequality (k : ℝ) :
  (∀ n : ℕ, n ≥ 4 → a n k ≥ a 4 k) →
  k ∈ Set.Icc (-9 : ℝ) (-7 : ℝ) := by
  sorry

end sequence_inequality_l633_63386


namespace largest_prime_factors_difference_l633_63375

theorem largest_prime_factors_difference (n : Nat) (h : n = 195195) :
  ∃ (p q : Nat), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p > q ∧
    p ∣ n ∧
    q ∣ n ∧
    (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p) ∧
    (∀ r : Nat, Nat.Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
    p - q = 2 := by
  sorry

end largest_prime_factors_difference_l633_63375


namespace tea_box_theorem_l633_63364

/-- The amount of tea leaves in a box, given daily consumption and duration -/
def tea_box_amount (daily_consumption : ℚ) (weeks : ℕ) : ℚ :=
  daily_consumption * 7 * weeks

/-- Theorem: A box of tea leaves containing 28 ounces lasts 20 weeks with 1/5 ounce daily consumption -/
theorem tea_box_theorem :
  tea_box_amount (1/5) 20 = 28 := by
  sorry

#eval tea_box_amount (1/5) 20

end tea_box_theorem_l633_63364


namespace inequality_theorem_l633_63343

theorem inequality_theorem (m n : ℕ) (hm : m > 0) (hn : n > 0) (hmn : m > n) (hn2 : n ≥ 2) :
  ((1 + 1 / m : ℝ) ^ m > (1 + 1 / n : ℝ) ^ n) ∧
  ((1 + 1 / m : ℝ) ^ (m + 1) < (1 + 1 / n : ℝ) ^ (n + 1)) :=
by sorry

end inequality_theorem_l633_63343


namespace cookies_per_neighbor_l633_63357

/-- Proves the number of cookies each neighbor was supposed to take -/
theorem cookies_per_neighbor
  (total_cookies : ℕ)
  (num_neighbors : ℕ)
  (cookies_left : ℕ)
  (sarah_cookies : ℕ)
  (h1 : total_cookies = 150)
  (h2 : num_neighbors = 15)
  (h3 : cookies_left = 8)
  (h4 : sarah_cookies = 12)
  : total_cookies / num_neighbors = 10 := by
  sorry

#check cookies_per_neighbor

end cookies_per_neighbor_l633_63357


namespace unique_rational_root_l633_63324

/-- The polynomial function we're examining -/
def f (x : ℚ) : ℚ := 3 * x^4 - 4 * x^3 - 10 * x^2 + 6 * x + 3

/-- A rational number is a root of f if f(x) = 0 -/
def is_root (x : ℚ) : Prop := f x = 0

/-- The statement that 1/3 is the only rational root of f -/
theorem unique_rational_root : 
  (is_root (1/3)) ∧ (∀ x : ℚ, is_root x → x = 1/3) := by sorry

end unique_rational_root_l633_63324


namespace monotonic_f_range_l633_63385

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 3 else (a + 2) * Real.exp (a * x)

/-- The theorem stating the range of a for which f is monotonic -/
theorem monotonic_f_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end monotonic_f_range_l633_63385


namespace reciprocal_of_repeating_decimal_l633_63398

/-- The decimal representation of the repeating decimal 0.565656... -/
def repeating_decimal : ℚ := 56 / 99

/-- The reciprocal of the repeating decimal 0.565656... -/
def reciprocal : ℚ := 99 / 56

/-- Theorem: The reciprocal of the common fraction form of 0.565656... is 99/56 -/
theorem reciprocal_of_repeating_decimal : (repeating_decimal)⁻¹ = reciprocal := by
  sorry

end reciprocal_of_repeating_decimal_l633_63398


namespace sum_of_digits_is_three_l633_63351

/-- Represents a 100-digit number with repeating pattern 5050 --/
def a : ℕ := 5050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050505050

/-- Represents a 100-digit number with repeating pattern 7070 --/
def b : ℕ := 7070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070

/-- The product of a and b --/
def product : ℕ := a * b

/-- Extracts the thousands digit from a number --/
def thousands_digit (n : ℕ) : ℕ := (n / 1000) % 10

/-- Extracts the units digit from a number --/
def units_digit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the sum of the thousands digit and units digit of the product is 3 --/
theorem sum_of_digits_is_three : 
  thousands_digit product + units_digit product = 3 := by sorry

end sum_of_digits_is_three_l633_63351


namespace solution_set_when_m_is_5_minimum_m_for_intersection_l633_63302

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2*|x + 1|

-- Part I
theorem solution_set_when_m_is_5 :
  {x : ℝ | f 5 x > 2} = {x : ℝ | -4/3 < x ∧ x < 0} := by sorry

-- Part II
-- Define the quadratic function
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem minimum_m_for_intersection :
  ∀ m : ℝ, (∀ x : ℝ, ∃ y : ℝ, f m y = g y) ↔ m ≥ 4 := by sorry

end solution_set_when_m_is_5_minimum_m_for_intersection_l633_63302


namespace arithmetic_sequence_sum_l633_63339

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of three consecutive terms in an arithmetic sequence -/
def sum_three_consecutive (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n + a (n + 1) + a (n + 2)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  sum_three_consecutive a 4 = 36 →
  a 1 + a 9 = 24 := by
  sorry

end arithmetic_sequence_sum_l633_63339


namespace amanda_ticket_sales_l633_63326

/-- Amanda's ticket sales problem -/
theorem amanda_ticket_sales
  (total_tickets : ℕ)
  (first_day_sales : ℕ)
  (third_day_sales : ℕ)
  (h1 : total_tickets = 80)
  (h2 : first_day_sales = 20)
  (h3 : third_day_sales = 28) :
  total_tickets - first_day_sales - third_day_sales = 32 := by
  sorry

#check amanda_ticket_sales

end amanda_ticket_sales_l633_63326


namespace closest_integer_to_cube_root_l633_63397

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7 : ℝ)^3 + (9 : ℝ)^3 - 100 → 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |x^(1/3) - n| ≤ |x^(1/3) - m| :=
sorry

end closest_integer_to_cube_root_l633_63397


namespace intersects_x_axis_once_l633_63370

/-- A function f(x) = (k-3)x^2 + 2x + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 3) * x^2 + 2 * x + 1

/-- The condition for a quadratic function to have exactly one root -/
def has_one_root (k : ℝ) : Prop :=
  (k = 3) ∨ (4 * (k - 3) * 1 = 2^2)

theorem intersects_x_axis_once (k : ℝ) :
  (∃! x, f k x = 0) ↔ has_one_root k := by sorry

end intersects_x_axis_once_l633_63370


namespace jason_total_games_l633_63335

/-- The number of football games Jason attended this month -/
def games_this_month : ℕ := 11

/-- The number of football games Jason attended last month -/
def games_last_month : ℕ := 17

/-- The number of football games Jason plans to attend next month -/
def games_next_month : ℕ := 16

/-- The total number of games Jason will attend -/
def total_games : ℕ := games_this_month + games_last_month + games_next_month

theorem jason_total_games : total_games = 44 := by
  sorry

end jason_total_games_l633_63335


namespace garden_trees_l633_63346

/-- The number of trees in a garden with given specifications -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  yard_length / tree_distance + 1

/-- Theorem stating that the number of trees in the garden is 26 -/
theorem garden_trees : number_of_trees 400 16 = 26 := by
  sorry

end garden_trees_l633_63346


namespace angle_bisector_theorem_l633_63330

-- Define the triangle ABC and point X
structure Triangle :=
  (A B C X : ℝ × ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  let d := λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt
  d t.A t.B = 75 ∧
  d t.A t.C = 45 ∧
  d t.B t.C = 90 ∧
  -- X is on the angle bisector of angle ACB
  (t.X.1 - t.A.1) / (t.C.1 - t.A.1) = (t.X.2 - t.A.2) / (t.C.2 - t.A.2) ∧
  (t.X.1 - t.B.1) / (t.C.1 - t.B.1) = (t.X.2 - t.B.2) / (t.C.2 - t.B.2)

-- Theorem statement
theorem angle_bisector_theorem (t : Triangle) (h : is_valid_triangle t) :
  let d := λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt
  d t.A t.X = 25 :=
sorry

end angle_bisector_theorem_l633_63330


namespace luke_new_cards_l633_63369

/-- The number of new baseball cards Luke had --/
def new_cards (cards_per_page old_cards total_pages : ℕ) : ℕ :=
  cards_per_page * total_pages - old_cards

/-- Theorem stating that Luke had 3 new cards --/
theorem luke_new_cards : new_cards 3 9 4 = 3 := by
  sorry

end luke_new_cards_l633_63369


namespace cookie_brownie_difference_l633_63378

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the initial number of cookies -/
def initialCookies : ℕ := 60

/-- Represents the initial number of brownies -/
def initialBrownies : ℕ := 10

/-- Represents the number of cookies eaten per day -/
def cookiesPerDay : ℕ := 3

/-- Represents the number of brownies eaten per day -/
def browniesPerDay : ℕ := 1

/-- Calculates the remaining cookies after a week -/
def remainingCookies : ℕ := initialCookies - daysInWeek * cookiesPerDay

/-- Calculates the remaining brownies after a week -/
def remainingBrownies : ℕ := initialBrownies - daysInWeek * browniesPerDay

/-- Theorem stating the difference between remaining cookies and brownies after a week -/
theorem cookie_brownie_difference :
  remainingCookies - remainingBrownies = 36 := by
  sorry

end cookie_brownie_difference_l633_63378


namespace g_range_l633_63328

noncomputable def g (x : ℝ) : ℝ := 3 * (x - 2)

theorem g_range :
  Set.range g = {y : ℝ | y ≠ -21} := by sorry

end g_range_l633_63328


namespace generating_function_value_at_one_intersection_point_on_generating_function_l633_63354

/-- Linear function -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- Generating function of two linear functions -/
def generatingFunction (f₁ f₂ : LinearFunction) (m n : ℝ) (x : ℝ) : ℝ :=
  m * (f₁.a * x + f₁.b) + n * (f₂.a * x + f₂.b)

/-- Theorem: The value of the generating function of y = x + 1 and y = 2x when x = 1 is 2 -/
theorem generating_function_value_at_one :
  ∀ (m n : ℝ), m + n = 1 →
  generatingFunction ⟨1, 1⟩ ⟨2, 0⟩ m n 1 = 2 := by
  sorry

/-- Theorem: The intersection point of two linear functions lies on their generating function -/
theorem intersection_point_on_generating_function (f₁ f₂ : LinearFunction) (m n : ℝ) :
  m + n = 1 →
  ∀ (x y : ℝ),
  (f₁.a * x + f₁.b = y ∧ f₂.a * x + f₂.b = y) →
  generatingFunction f₁ f₂ m n x = y := by
  sorry

end generating_function_value_at_one_intersection_point_on_generating_function_l633_63354


namespace greatest_integer_less_than_negative_fifteen_fourths_l633_63379

theorem greatest_integer_less_than_negative_fifteen_fourths :
  ⌊-15/4⌋ = -4 :=
sorry

end greatest_integer_less_than_negative_fifteen_fourths_l633_63379


namespace p_or_q_necessary_not_sufficient_l633_63382

theorem p_or_q_necessary_not_sufficient :
  (∀ p q : Prop, (¬p → (p ∨ q))) ∧
  (∃ p q : Prop, (p ∨ q) ∧ ¬(¬p → False)) :=
by sorry

end p_or_q_necessary_not_sufficient_l633_63382


namespace q_div_p_eq_225_l633_63306

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The number of cards with each number -/
def cards_per_number : ℕ := 5

/-- The probability of drawing five cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (total_cards.choose cards_drawn : ℚ)

/-- The probability of drawing four cards with one number and one card with a different number -/
def q : ℚ := (distinct_numbers * (distinct_numbers - 1) * cards_per_number * cards_per_number : ℚ) / (total_cards.choose cards_drawn : ℚ)

/-- The ratio of q to p is 225 -/
theorem q_div_p_eq_225 : q / p = 225 := by sorry

end q_div_p_eq_225_l633_63306


namespace expression_evaluation_l633_63367

theorem expression_evaluation :
  let sin30 : Real := 1/2
  4 * (Real.sqrt 3 + Real.sqrt 7) / (5 * Real.sqrt (3 + sin30)) = (16 + 8 * Real.sqrt 21) / 35 := by
  sorry

end expression_evaluation_l633_63367


namespace cosine_inequality_l633_63321

theorem cosine_inequality (θ : ℝ) : 5 + 8 * Real.cos θ + 4 * Real.cos (2 * θ) + Real.cos (3 * θ) ≥ 0 := by
  sorry

end cosine_inequality_l633_63321


namespace largest_digit_divisible_by_6_l633_63319

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

theorem largest_digit_divisible_by_6 :
  ∀ N : ℕ, N ≤ 9 →
    (is_divisible_by_6 (2345 * 10 + N) → N ≤ 4) ∧
    (is_divisible_by_6 (2345 * 10 + 4)) :=
by sorry

end largest_digit_divisible_by_6_l633_63319


namespace complex_fraction_simplification_l633_63359

theorem complex_fraction_simplification :
  (Complex.I : ℂ) / (1 - Complex.I) = -1/2 + Complex.I/2 := by
  sorry

end complex_fraction_simplification_l633_63359


namespace circle_to_square_impossible_l633_63391

/-- Represents a piece of paper with a boundary --/
structure PaperPiece where
  boundary : Set ℝ × ℝ

/-- Represents a cut on a paper piece --/
inductive Cut
  | StraightLine : (ℝ × ℝ) → (ℝ × ℝ) → Cut
  | CircularArc : (ℝ × ℝ) → ℝ → ℝ → ℝ → Cut

/-- Represents a transformation of paper pieces --/
def Transform := List PaperPiece → List PaperPiece

/-- Checks if a shape is a circle --/
def is_circle (p : PaperPiece) : Prop := sorry

/-- Checks if a shape is a square --/
def is_square (p : PaperPiece) : Prop := sorry

/-- Calculates the area of a paper piece --/
def area (p : PaperPiece) : ℝ := sorry

/-- Theorem stating the impossibility of transforming a circle to a square of equal area --/
theorem circle_to_square_impossible 
  (initial : PaperPiece) 
  (cuts : List Cut) 
  (transform : Transform) :
  is_circle initial →
  (∃ final, is_square final ∧ area final = area initial ∧ 
    transform [initial] = final :: (transform [initial]).tail) →
  False := by
  sorry

#check circle_to_square_impossible

end circle_to_square_impossible_l633_63391


namespace mike_pears_l633_63372

theorem mike_pears (jason_pears keith_pears total_pears : ℕ)
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : total_pears = 105)
  (h4 : ∃ mike_pears : ℕ, jason_pears + keith_pears + mike_pears = total_pears) :
  ∃ mike_pears : ℕ, mike_pears = 12 ∧ jason_pears + keith_pears + mike_pears = total_pears := by
sorry

end mike_pears_l633_63372


namespace carlton_outfit_combinations_l633_63361

theorem carlton_outfit_combinations 
  (button_up_shirts : ℕ) 
  (sweater_vests : ℕ) 
  (ties : ℕ) 
  (h1 : button_up_shirts = 4)
  (h2 : sweater_vests = 3 * button_up_shirts)
  (h3 : ties = 2 * sweater_vests) : 
  button_up_shirts * sweater_vests * ties = 1152 := by
  sorry

end carlton_outfit_combinations_l633_63361


namespace remainder_problem_l633_63331

theorem remainder_problem :
  {x : ℕ | x < 100 ∧ x % 7 = 3 ∧ x % 9 = 4} = {31, 94} := by
  sorry

end remainder_problem_l633_63331


namespace satellite_sensor_ratio_l633_63304

theorem satellite_sensor_ratio :
  ∀ (S : ℝ) (N : ℝ),
    S > 0 →
    N > 0 →
    S = 0.2 * S + 24 * N →
    N / (0.2 * S) = 1 / 6 :=
by
  sorry

end satellite_sensor_ratio_l633_63304


namespace sin_105_cos_105_l633_63305

theorem sin_105_cos_105 :
  Real.sin (105 * π / 180) * Real.cos (105 * π / 180) = -1/4 := by
  sorry

end sin_105_cos_105_l633_63305


namespace arrange_balls_and_boxes_eq_20_l633_63307

/-- The number of ways to arrange 5 balls in 5 boxes with exactly two matches -/
def arrange_balls_and_boxes : ℕ :=
  let n : ℕ := 5  -- Total number of balls and boxes
  let k : ℕ := 2  -- Number of matches required
  let derangement_3 : ℕ := 2  -- Number of derangements for 3 elements
  (n.choose k) * derangement_3

/-- Theorem stating that the number of arrangements is 20 -/
theorem arrange_balls_and_boxes_eq_20 : arrange_balls_and_boxes = 20 := by
  sorry

end arrange_balls_and_boxes_eq_20_l633_63307


namespace parallelepiped_coverage_l633_63325

/-- Represents a parallelepiped with integer dimensions --/
structure Parallelepiped where
  width : ℕ
  depth : ℕ
  height : ℕ

/-- Represents a square with an integer side length --/
structure Square where
  side : ℕ

/-- Checks if a set of squares can cover a parallelepiped without gaps or overlaps --/
def can_cover (p : Parallelepiped) (squares : List Square) : Prop :=
  let surface_area := 2 * (p.width * p.depth + p.width * p.height + p.depth * p.height)
  let squares_area := squares.map (λ s => s.side * s.side) |>.sum
  surface_area = squares_area

theorem parallelepiped_coverage : 
  let p := Parallelepiped.mk 1 1 4
  let squares := [Square.mk 4, Square.mk 1, Square.mk 1]
  can_cover p squares := by
  sorry

end parallelepiped_coverage_l633_63325


namespace advance_ticket_cost_l633_63368

/-- The cost of advance tickets is $20, given the specified conditions. -/
theorem advance_ticket_cost (same_day_cost : ℕ) (total_tickets : ℕ) (total_receipts : ℕ) (advance_tickets_sold : ℕ) :
  same_day_cost = 30 →
  total_tickets = 60 →
  total_receipts = 1600 →
  advance_tickets_sold = 20 →
  ∃ (advance_cost : ℕ), advance_cost * advance_tickets_sold + same_day_cost * (total_tickets - advance_tickets_sold) = total_receipts ∧ advance_cost = 20 :=
by sorry

end advance_ticket_cost_l633_63368


namespace perpendicular_point_sets_l633_63336

-- Definition of a perpendicular point set
def is_perpendicular_point_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ M, ∃ p₂ ∈ M, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the sets M₃ and M₄
def M₃ : Set (ℝ × ℝ) := {p | p.2 = Real.exp p.1 - 2}
def M₄ : Set (ℝ × ℝ) := {p | p.2 = Real.sin p.1 + 1}

-- Theorem statement
theorem perpendicular_point_sets :
  is_perpendicular_point_set M₃ ∧ is_perpendicular_point_set M₄ := by
  sorry

end perpendicular_point_sets_l633_63336


namespace line_point_k_value_l633_63327

/-- Given a line containing points (2, 9), (10, k), and (25, 4), prove that k = 167/23 -/
theorem line_point_k_value (k : ℚ) : 
  (∃ (m b : ℚ), 9 = m * 2 + b ∧ k = m * 10 + b ∧ 4 = m * 25 + b) → 
  k = 167 / 23 := by
sorry

end line_point_k_value_l633_63327


namespace line_slope_intercept_sum_l633_63312

/-- Given a line passing through points (1, 3) and (3, 7) with equation y = mx + b, 
    the sum of m and b is equal to 3. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) → (7 = m * 3 + b) → m + b = 3 := by
  sorry


end line_slope_intercept_sum_l633_63312


namespace non_right_triangle_l633_63381

theorem non_right_triangle : 
  let triangle_sets : List (ℝ × ℝ × ℝ) := 
    [(6, 8, 10), (1, Real.sqrt 3, 2), (5/4, 1, 3/4), (4, 5, 7)]
  ∀ (a b c : ℝ), (a, b, c) ∈ triangle_sets →
    (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) ↔ (a, b, c) ≠ (4, 5, 7) :=
by sorry

end non_right_triangle_l633_63381


namespace hyperbola_eccentricity_l633_63355

/-- 
Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
if the distance between one of its foci and an asymptote is one-fourth of its focal distance,
then the eccentricity of the hyperbola is 2√3/3.
-/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let focal_distance := 2 * c
  let asymptote_distance := b
  focal_distance = 4 * asymptote_distance →
  c / a = 2 * Real.sqrt 3 / 3 := by
  sorry

end hyperbola_eccentricity_l633_63355


namespace imaginary_part_of_one_minus_three_i_squared_l633_63333

theorem imaginary_part_of_one_minus_three_i_squared : 
  Complex.im ((1 - 3*Complex.I)^2) = -6 := by
  sorry

end imaginary_part_of_one_minus_three_i_squared_l633_63333


namespace buddy_cards_bought_l633_63342

/-- Represents the number of baseball cards Buddy has on each day --/
structure BuddyCards where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Represents the number of cards Buddy bought on Wednesday and Thursday --/
structure CardsBought where
  wednesday : ℕ
  thursday : ℕ

/-- The theorem statement --/
theorem buddy_cards_bought (cards : BuddyCards) (bought : CardsBought) : 
  cards.monday = 30 →
  cards.tuesday = cards.monday / 2 →
  cards.thursday = 32 →
  bought.thursday = cards.tuesday / 3 →
  cards.wednesday = cards.tuesday + bought.wednesday →
  cards.thursday = cards.wednesday + bought.thursday →
  bought.wednesday = 12 := by
  sorry


end buddy_cards_bought_l633_63342


namespace power_of_three_l633_63387

theorem power_of_three (a b : ℕ+) (h : 3^(a : ℕ) * 3^(b : ℕ) = 81) :
  (3^(a : ℕ))^(b : ℕ) = 3^4 := by
sorry

end power_of_three_l633_63387


namespace min_steps_to_one_l633_63376

/-- Represents the allowed operations in one step -/
inductive Operation
  | AddOne
  | DivideByTwo
  | DivideByThree

/-- Applies an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.DivideByTwo => n / 2
  | Operation.DivideByThree => n / 3

/-- Checks if a sequence of operations is valid -/
def isValidSequence (start : ℕ) (ops : List Operation) : Bool :=
  ops.foldl (fun acc op => applyOperation acc op) start = 1

/-- The minimum number of steps to reach 1 from the starting number -/
def minSteps (start : ℕ) : ℕ :=
  sorry

theorem min_steps_to_one :
  minSteps 19 = 6 :=
sorry

end min_steps_to_one_l633_63376


namespace donny_gas_change_l633_63392

/-- Calculates the change Donny receives after filling up his truck's gas tank. -/
theorem donny_gas_change (tank_capacity : ℕ) (initial_fuel : ℕ) (fuel_cost : ℕ) (payment : ℕ) : 
  tank_capacity = 150 →
  initial_fuel = 38 →
  fuel_cost = 3 →
  payment = 350 →
  payment - (tank_capacity - initial_fuel) * fuel_cost = 14 := by
  sorry

#check donny_gas_change

end donny_gas_change_l633_63392


namespace quadratic_equation_completion_l633_63308

theorem quadratic_equation_completion (x k ℓ : ℝ) : 
  (13 * x^2 + 39 * x - 91 = 0) ∧ 
  ((x + k)^2 - |ℓ| = 0) →
  |k + ℓ| = 10.75 := by
  sorry

end quadratic_equation_completion_l633_63308


namespace smallest_invertible_domain_l633_63329

-- Define the function f
def f (x : ℝ) : ℝ := (x + 3)^2 - 7

-- State the theorem
theorem smallest_invertible_domain : 
  ∃ (c : ℝ), c = -3 ∧ 
  (∀ x y, x ≥ c → y ≥ c → f x = f y → x = y) ∧ 
  (∀ c' < c, ∃ x y, x ≥ c' → y ≥ c' → f x = f y ∧ x ≠ y) :=
sorry

end smallest_invertible_domain_l633_63329


namespace smallest_multiple_l633_63313

theorem smallest_multiple (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k) ∧ 
  n ≡ 3 [ZMOD 71] ∧ 
  (∀ m : ℕ, m < n → ¬((∃ k : ℕ, m = 17 * k) ∧ m ≡ 3 [ZMOD 71])) → 
  n = 1139 := by
sorry

end smallest_multiple_l633_63313


namespace ceiling_times_self_182_l633_63371

theorem ceiling_times_self_182 :
  ∃! (x : ℝ), ⌈x⌉ * x = 182 :=
by
  -- The proof goes here
  sorry

end ceiling_times_self_182_l633_63371


namespace min_value_sum_squares_l633_63365

theorem min_value_sum_squares (x₁ x₂ x₃ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 3*x₂ + 4*x₃ = 100) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 10000/29 :=
by sorry

end min_value_sum_squares_l633_63365


namespace sector_max_area_l633_63390

/-- Given a sector with circumference 20cm, its maximum area is 25cm² -/
theorem sector_max_area :
  ∀ r l : ℝ,
  r > 0 →
  l > 0 →
  l + 2 * r = 20 →
  ∀ A : ℝ,
  A = 1/2 * l * r →
  A ≤ 25 :=
by
  sorry

end sector_max_area_l633_63390


namespace work_completion_solution_l633_63314

/-- Represents the work completion problem -/
structure WorkCompletion where
  initial_workers : ℕ
  initial_days : ℕ
  worked_days : ℕ
  added_workers : ℕ

/-- Calculates the total days to complete the work -/
def total_days (w : WorkCompletion) : ℚ :=
  let initial_work_rate : ℚ := 1 / (w.initial_workers * w.initial_days)
  let work_done : ℚ := w.worked_days * w.initial_workers * initial_work_rate
  let remaining_work : ℚ := 1 - work_done
  let new_work_rate : ℚ := (w.initial_workers + w.added_workers) * initial_work_rate
  w.worked_days + remaining_work / new_work_rate

/-- Theorem stating the solution to the work completion problem -/
theorem work_completion_solution :
  ∀ w : WorkCompletion,
    w.initial_workers = 12 →
    w.initial_days = 18 →
    w.worked_days = 6 →
    w.added_workers = 4 →
    total_days w = 24 := by
  sorry

end work_completion_solution_l633_63314


namespace negation_of_p_is_existential_l633_63360

-- Define the set of even numbers
def A : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 2 * k}

-- Define the proposition p
def p : Prop := ∀ x : ℤ, (2 * x) ∈ A

-- Theorem statement
theorem negation_of_p_is_existential :
  ¬p ↔ ∃ x : ℤ, (2 * x) ∉ A := by sorry

end negation_of_p_is_existential_l633_63360


namespace student_council_max_profit_l633_63394

/-- Calculate the maximum amount of money the student council can make from selling erasers --/
theorem student_council_max_profit (
  boxes : ℕ)
  (erasers_per_box : ℕ)
  (price_per_eraser : ℚ)
  (bulk_discount_rate : ℚ)
  (bulk_purchase_threshold : ℕ)
  (sales_tax_rate : ℚ)
  (h1 : boxes = 48)
  (h2 : erasers_per_box = 24)
  (h3 : price_per_eraser = 3/4)
  (h4 : bulk_discount_rate = 1/10)
  (h5 : bulk_purchase_threshold = 10)
  (h6 : sales_tax_rate = 3/50)
  : ∃ (max_profit : ℚ), max_profit = 82426/100 :=
by
  sorry

end student_council_max_profit_l633_63394


namespace third_square_is_G_l633_63303

/-- Represents a 2x2 square -/
structure Square :=
  (label : Char)

/-- Represents the visibility of a square -/
inductive Visibility
  | Full
  | Partial

/-- Represents the position of a square in the 4x4 grid -/
structure Position :=
  (row : Fin 2)
  (col : Fin 2)

/-- Represents the state of the 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Option Square

/-- Represents the sequence of square placements -/
def PlacementSequence := List Square

/-- Determines if a square is in a corner position -/
def isCorner (pos : Position) : Bool :=
  (pos.row = 0 ∨ pos.row = 1) ∧ (pos.col = 0 ∨ pos.col = 1)

/-- The main theorem to prove -/
theorem third_square_is_G 
  (squares : List Square)
  (grid : Grid)
  (sequence : PlacementSequence)
  (visibility : Square → Visibility)
  (position : Square → Position) :
  squares.length = 8 ∧
  (∃ s ∈ squares, s.label = 'E') ∧
  visibility (Square.mk 'E') = Visibility.Full ∧
  (∀ s ∈ squares, s.label ≠ 'E' → visibility s = Visibility.Partial) ∧
  (∃ s ∈ squares, isCorner (position s) ∧ s.label = 'G') ∧
  sequence.length = 8 ∧
  sequence.getLast? = some (Square.mk 'E') →
  (sequence.get? 2 = some (Square.mk 'G')) :=
by sorry

end third_square_is_G_l633_63303


namespace min_value_theorem_l633_63350

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + 3 ≥ 9 ∧
  ((x + y) / z + (x + z) / y + (y + z) / x + 3 = 9 ↔ x = y ∧ y = z) :=
sorry

end min_value_theorem_l633_63350


namespace inscribed_circle_radius_right_triangle_l633_63332

/-- The radius of an inscribed circle in a right triangle -/
theorem inscribed_circle_radius_right_triangle
  (a b c : ℝ)
  (h_right : a^2 + b^2 = c^2)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ r : ℝ, r = (a + b - c) / 2 ∧ r > 0 :=
sorry

end inscribed_circle_radius_right_triangle_l633_63332


namespace quadrilateral_inequality_l633_63377

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define an interior point
def interior_point (q : Quadrilateral) (O : ℝ × ℝ) : Prop := sorry

-- Define distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define area of a triangle
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define s₃ and s₄
def s₃ (q : Quadrilateral) (O : ℝ × ℝ) : ℝ :=
  distance O q.A + distance O q.B + distance O q.C + distance O q.D

def s₄ (q : Quadrilateral) : ℝ :=
  distance q.A q.B + distance q.B q.C + distance q.C q.D + distance q.D q.A

-- State the theorem
theorem quadrilateral_inequality (q : Quadrilateral) (O : ℝ × ℝ) :
  interior_point q O →
  triangle_area O q.A q.B = triangle_area O q.C q.D →
  s₃ q O ≥ (1/2) * s₄ q ∧ s₃ q O ≤ s₄ q :=
by sorry

end quadrilateral_inequality_l633_63377


namespace bd_length_l633_63389

/-- Given four points A, B, C, and D on a line in that order, prove that BD = 6 -/
theorem bd_length 
  (A B C D : ℝ) -- Points represented as real numbers on a line
  (h_order : A ≤ B ∧ B ≤ C ∧ C ≤ D) -- Order of points on the line
  (h_AB : B - A = 2) -- Length of AB
  (h_AC : C - A = 5) -- Length of AC
  (h_CD : D - C = 3) -- Length of CD
  : D - B = 6 := by
  sorry

end bd_length_l633_63389


namespace cos_two_thirds_pi_l633_63344

theorem cos_two_thirds_pi : Real.cos (2/3 * Real.pi) = -(1/2) := by sorry

end cos_two_thirds_pi_l633_63344


namespace equilibrium_force_l633_63347

/-- Given three forces in a 2D plane, prove that a specific fourth force is required for equilibrium. -/
theorem equilibrium_force (f₁ f₂ f₃ f₄ : ℝ × ℝ) : 
  f₁ = (-2, -1) → f₂ = (-3, 2) → f₃ = (4, -3) →
  (f₁.1 + f₂.1 + f₃.1 + f₄.1 = 0 ∧ f₁.2 + f₂.2 + f₃.2 + f₄.2 = 0) →
  f₄ = (1, 2) := by
sorry

end equilibrium_force_l633_63347


namespace soda_bottle_difference_l633_63349

theorem soda_bottle_difference (regular_soda : ℕ) (diet_soda : ℕ)
  (h1 : regular_soda = 60)
  (h2 : diet_soda = 19) :
  regular_soda - diet_soda = 41 := by
  sorry

end soda_bottle_difference_l633_63349


namespace laptop_price_proof_l633_63320

theorem laptop_price_proof (original_price : ℝ) : 
  (0.7 * original_price - (0.8 * original_price - 70) = 20) → 
  original_price = 500 := by
sorry

end laptop_price_proof_l633_63320


namespace ammonia_formation_l633_63395

/-- Represents the chemical reaction between Potassium hydroxide and Ammonium iodide -/
structure ChemicalReaction where
  koh : ℝ  -- moles of Potassium hydroxide
  nh4i : ℝ  -- moles of Ammonium iodide
  nh3 : ℝ  -- moles of Ammonia formed

/-- Theorem stating that the moles of Ammonia formed equals the moles of Ammonium iodide used -/
theorem ammonia_formation (reaction : ChemicalReaction) 
  (h1 : reaction.nh4i = 3)  -- 3 moles of Ammonium iodide are used
  (h2 : reaction.nh3 = 3)   -- The total moles of Ammonia formed is 3
  : reaction.nh3 = reaction.nh4i := by
  sorry


end ammonia_formation_l633_63395


namespace tagged_fish_count_l633_63352

/-- The number of tagged fish found in the second catch -/
def tagged_fish_in_second_catch (total_fish : ℕ) (initially_tagged : ℕ) (second_catch : ℕ) : ℕ :=
  (initially_tagged * second_catch) / total_fish

/-- Proof that the number of tagged fish in the second catch is 2 -/
theorem tagged_fish_count :
  let total_fish : ℕ := 1800
  let initially_tagged : ℕ := 60
  let second_catch : ℕ := 60
  tagged_fish_in_second_catch total_fish initially_tagged second_catch = 2 := by
  sorry

end tagged_fish_count_l633_63352


namespace fast_food_purchase_cost_l633_63301

/-- The cost of a single sandwich -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 6

/-- The number of sodas purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem fast_food_purchase_cost : total_cost = 39 := by
  sorry

end fast_food_purchase_cost_l633_63301


namespace vinegar_left_is_60_l633_63309

/-- Represents the pickle-making scenario with given supplies and constraints. -/
structure PickleSupplies where
  jars : ℕ
  cucumbers : ℕ
  vinegar : ℕ
  pickles_per_cucumber : ℕ
  pickles_per_jar : ℕ
  vinegar_per_jar : ℕ

/-- Calculates the amount of vinegar left after making pickles. -/
def vinegar_left (supplies : PickleSupplies) : ℕ :=
  let jar_capacity := supplies.jars * supplies.pickles_per_jar
  let cucumber_capacity := supplies.cucumbers * supplies.pickles_per_cucumber
  let vinegar_capacity := supplies.vinegar / supplies.vinegar_per_jar * supplies.pickles_per_jar
  let pickles_made := min jar_capacity (min cucumber_capacity vinegar_capacity)
  let jars_used := (pickles_made + supplies.pickles_per_jar - 1) / supplies.pickles_per_jar
  supplies.vinegar - jars_used * supplies.vinegar_per_jar

/-- Theorem stating that given the specific supplies and constraints, 60 ounces of vinegar are left. -/
theorem vinegar_left_is_60 :
  vinegar_left {
    jars := 4,
    cucumbers := 10,
    vinegar := 100,
    pickles_per_cucumber := 6,
    pickles_per_jar := 12,
    vinegar_per_jar := 10
  } = 60 := by
  sorry

end vinegar_left_is_60_l633_63309


namespace complex_square_sum_of_squares_l633_63393

theorem complex_square_sum_of_squares (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (↑a + Complex.I * ↑b)^2 = (3 : ℂ) + Complex.I * 4 →
  a^2 + b^2 = 5 := by sorry

end complex_square_sum_of_squares_l633_63393


namespace equation_solution_l633_63356

theorem equation_solution : ∃! x : ℝ, 7 * (2 * x + 3) - 5 = -3 * (2 - 5 * x) ∧ x = 22 := by
  sorry

end equation_solution_l633_63356


namespace min_cost_1001_grid_square_l633_63317

/-- Represents a grid square with side length n -/
def GridSquare (n : ℕ) := {m : ℕ × ℕ | m.1 ≤ n ∧ m.2 ≤ n}

/-- The cost of coloring a single cell -/
def colorCost : ℕ := 1

/-- The minimum number of cells that need to be colored to create a complete grid square -/
def minColoredCells (n : ℕ) : ℕ := n * n + 2 * n * (n - 1)

theorem min_cost_1001_grid_square :
  minColoredCells 1001 * colorCost = 503000 :=
sorry

end min_cost_1001_grid_square_l633_63317


namespace sequence_286_ends_l633_63323

/-- A sequence of seven positive integers where each number differs by one from its neighbors -/
def ValidSequence (a b c d e f g : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧
  (b = a + 1 ∨ b = a - 1) ∧
  (c = b + 1 ∨ c = b - 1) ∧
  (d = c + 1 ∨ d = c - 1) ∧
  (e = d + 1 ∨ e = d - 1) ∧
  (f = e + 1 ∨ f = e - 1) ∧
  (g = f + 1 ∨ g = f - 1)

theorem sequence_286_ends (a b c d e f g : ℕ) :
  ValidSequence a b c d e f g →
  a + b + c + d + e + f + g = 2017 →
  (a = 286 ∨ g = 286) ∧ b ≠ 286 ∧ c ≠ 286 ∧ d ≠ 286 ∧ e ≠ 286 ∧ f ≠ 286 :=
by sorry

end sequence_286_ends_l633_63323


namespace min_production_time_l633_63348

/-- Represents the production process of ceramic items -/
structure CeramicProduction where
  shapingTime : ℕ := 15
  dryingTime : ℕ := 10
  firingTime : ℕ := 30
  totalItems : ℕ := 75
  totalWorkers : ℕ := 13

/-- Calculates the production time for a given stage -/
def stageTime (itemsPerWorker : ℕ) (timePerItem : ℕ) : ℕ :=
  (itemsPerWorker + 1) * timePerItem

/-- Theorem stating the minimum production time for ceramic items -/
theorem min_production_time (prod : CeramicProduction) :
  ∃ (shapers firers : ℕ),
    shapers + firers = prod.totalWorkers ∧
    (∀ (s f : ℕ),
      s + f = prod.totalWorkers →
      max (stageTime (prod.totalItems / s) prod.shapingTime)
          (stageTime (prod.totalItems / f) prod.firingTime)
      ≥ 325) :=
sorry

end min_production_time_l633_63348


namespace min_triangle_area_l633_63373

open Complex

-- Define the equation solutions
def solutions : Set ℂ := {z : ℂ | (z - 3)^10 = 32}

-- Define the property that solutions form a regular decagon
def is_regular_decagon (s : Set ℂ) : Prop := sorry

-- Define the function to calculate the area of a triangle given three complex points
def triangle_area (a b c : ℂ) : ℝ := sorry

-- Theorem statement
theorem min_triangle_area (h : is_regular_decagon solutions) :
  ∃ (a b c : ℂ), a ∈ solutions ∧ b ∈ solutions ∧ c ∈ solutions ∧
  (∀ (x y z : ℂ), x ∈ solutions → y ∈ solutions → z ∈ solutions →
    triangle_area a b c ≤ triangle_area x y z) ∧
  triangle_area a b c = 2 * Real.sin (18 * π / 180) * Real.sin (36 * π / 180) :=
sorry

end min_triangle_area_l633_63373


namespace cubic_function_properties_l633_63399

/-- A cubic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + b

/-- The derivative of f -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem cubic_function_properties (a b : ℝ) (h_a : a ≠ 0) :
  (f' a 2 = 0 ∧ f a b 2 = 8) →
  (a = 4 ∧ b = 24) ∧
  (∀ x, x < -2 → (f' a x > 0)) ∧
  (∀ x, x > 2 → (f' a x > 0)) ∧
  (∀ x, -2 < x ∧ x < 2 → (f' a x < 0)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - (-2)| ∧ |x - (-2)| < δ → f a b x < f a b (-2)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f a b x > f a b 2) :=
by sorry

end cubic_function_properties_l633_63399


namespace same_solution_implies_m_equals_9_l633_63322

theorem same_solution_implies_m_equals_9 :
  ∀ y m : ℝ,
  (y + 3 * m = 32) ∧ (y - 4 = 1) →
  m = 9 :=
by
  sorry

end same_solution_implies_m_equals_9_l633_63322


namespace rod_cutting_l633_63380

theorem rod_cutting (rod_length_m : ℝ) (piece_length_cm : ℝ) : 
  rod_length_m = 38.25 →
  piece_length_cm = 85 →
  ⌊(rod_length_m * 100) / piece_length_cm⌋ = 45 := by
sorry

end rod_cutting_l633_63380


namespace sum_four_digit_numbers_eq_179982_l633_63363

/-- The sum of all four-digit numbers created using digits 1, 2, and 3 with repetition -/
def sum_four_digit_numbers : ℕ :=
  let digits : List ℕ := [1, 2, 3]
  let total_numbers : ℕ := digits.length ^ 4
  let sum_per_position : ℕ := (digits.sum * total_numbers) / digits.length
  sum_per_position * 1000 + sum_per_position * 100 + sum_per_position * 10 + sum_per_position

theorem sum_four_digit_numbers_eq_179982 :
  sum_four_digit_numbers = 179982 := by
  sorry

#eval sum_four_digit_numbers

end sum_four_digit_numbers_eq_179982_l633_63363


namespace correct_propositions_l633_63388

theorem correct_propositions :
  -- Proposition 1
  (∀ P : Prop, (¬P ↔ P) → ¬P) ∧
  -- Proposition 2 (negation)
  ¬(∀ a : ℕ → ℝ, a 0 = 2 ∧ (∀ n : ℕ, a (n + 1) = a n + (a 2 - a 0) / 2) ∧
    (∃ q : ℝ, a 2 = a 0 * q ∧ a 3 = a 2 * q) →
    a 1 - a 0 = -1/2) ∧
  -- Proposition 3
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 →
    (2/a + 3/b ≥ 5 + 2 * Real.sqrt 6)) ∧
  -- Proposition 4 (negation)
  ¬(∀ A B C : ℝ, 0 ≤ A ∧ A ≤ π ∧ 0 ≤ B ∧ B ≤ π ∧ 0 ≤ C ∧ C ≤ π ∧ A + B + C = π →
    (Real.sin A)^2 < (Real.sin B)^2 + (Real.sin C)^2 →
    A < π/2 ∧ B < π/2 ∧ C < π/2) :=
by sorry

end correct_propositions_l633_63388
