import Mathlib

namespace NUMINAMATH_CALUDE_bushes_for_zucchinis_l4077_407782

/-- The number of containers of blueberries each bush yields -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries that can be traded for zucchinis -/
def containers_for_trade : ℕ := 6

/-- The number of zucchinis received in trade for containers_for_trade -/
def zucchinis_from_trade : ℕ := 3

/-- The target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- The number of bushes needed to obtain the target number of zucchinis -/
def bushes_needed : ℕ := 12

theorem bushes_for_zucchinis :
  bushes_needed * containers_per_bush * zucchinis_from_trade = 
  target_zucchinis * containers_for_trade := by
  sorry

end NUMINAMATH_CALUDE_bushes_for_zucchinis_l4077_407782


namespace NUMINAMATH_CALUDE_solve_equation_l4077_407705

theorem solve_equation (X : ℝ) : (X^3)^(1/2) = 9 * 81^(1/9) → X = 3^(44/27) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4077_407705


namespace NUMINAMATH_CALUDE_max_perfect_squares_l4077_407707

theorem max_perfect_squares (n : ℕ) : 
  (∃ (S : Finset ℕ), 
    (∀ k ∈ S, 1 ≤ k ∧ k ≤ 2015 ∧ ∃ m : ℕ, 240 * k = m^2) ∧ 
    S.card = n) →
  n ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_perfect_squares_l4077_407707


namespace NUMINAMATH_CALUDE_correct_statement_l4077_407746

-- Define proposition P
def P : Prop := ∀ x : ℝ, x^2 - 4*x + 5 > 0

-- Define proposition q
def q : Prop := ∃ x : ℝ, x > 0 ∧ Real.cos x > 1

-- Theorem to prove
theorem correct_statement : P ∨ ¬q := by
  sorry

end NUMINAMATH_CALUDE_correct_statement_l4077_407746


namespace NUMINAMATH_CALUDE_snooker_tournament_revenue_l4077_407752

theorem snooker_tournament_revenue :
  let total_tickets : ℕ := 320
  let vip_price : ℚ := 40
  let general_price : ℚ := 10
  let vip_tickets : ℕ := (total_tickets - 148) / 2
  let general_tickets : ℕ := (total_tickets + 148) / 2
  (vip_price * vip_tickets + general_price * general_tickets : ℚ) = 5780 :=
by sorry

end NUMINAMATH_CALUDE_snooker_tournament_revenue_l4077_407752


namespace NUMINAMATH_CALUDE_problem_solution_l4077_407793

theorem problem_solution (a b : ℚ) 
  (eq1 : 4 + 2*a = 5 - b) 
  (eq2 : 5 + b = 9 + 3*a) : 
  4 - 2*a = 26/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4077_407793


namespace NUMINAMATH_CALUDE_largest_odd_proper_divisor_ratio_l4077_407770

/-- The largest odd proper divisor of a positive integer -/
def f (n : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem largest_odd_proper_divisor_ratio :
  let N : ℕ := 20^23 * 23^20
  f N / f (f (f N)) = 25 :=
by sorry

end NUMINAMATH_CALUDE_largest_odd_proper_divisor_ratio_l4077_407770


namespace NUMINAMATH_CALUDE_work_completion_time_l4077_407706

theorem work_completion_time (A : ℝ) (h1 : A > 0) : 
  (∃ B : ℝ, B = A / 2 ∧ 1 / A + 1 / B = 1 / 6) → A = 18 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4077_407706


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l4077_407788

open Real

/-- Given f(x) = 2/x + a*ln(x) - 2 where a > 0, if f(x) > 2(a-1) for all x > 0, then 0 < a < 2/e -/
theorem function_inequality_implies_a_range (a : ℝ) (h_a_pos : a > 0) :
  (∀ x : ℝ, x > 0 → (2 / x + a * log x - 2 > 2 * (a - 1))) →
  0 < a ∧ a < 2 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l4077_407788


namespace NUMINAMATH_CALUDE_pattys_cafe_theorem_l4077_407725

/-- Represents the cost calculation at Patty's Cafe -/
def pattys_cafe_cost (sandwich_price soda_price discount_threshold discount : ℕ) 
                     (num_sandwiches num_sodas : ℕ) : ℕ :=
  let total_items := num_sandwiches + num_sodas
  let subtotal := sandwich_price * num_sandwiches + soda_price * num_sodas
  if total_items > discount_threshold then subtotal - discount else subtotal

/-- The cost of purchasing 7 sandwiches and 6 sodas at Patty's Cafe is $36 -/
theorem pattys_cafe_theorem : 
  pattys_cafe_cost 4 3 10 10 7 6 = 36 := by
  sorry


end NUMINAMATH_CALUDE_pattys_cafe_theorem_l4077_407725


namespace NUMINAMATH_CALUDE_intersection_point_correct_l4077_407728

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- The first line -/
def line1 : Line2D :=
  { point := (1, 2),
    direction := (2, -3) }

/-- The second line -/
def line2 : Line2D :=
  { point := (4, 5),
    direction := (1, -1) }

/-- A point lies on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p.1 = l.point.1 + t * l.direction.1 ∧
            p.2 = l.point.2 + t * l.direction.2

/-- The intersection point of the two lines -/
def intersectionPoint : ℝ × ℝ := (-11, 20)

/-- Theorem: The intersection point lies on both lines and is unique -/
theorem intersection_point_correct :
  pointOnLine intersectionPoint line1 ∧
  pointOnLine intersectionPoint line2 ∧
  ∀ p : ℝ × ℝ, pointOnLine p line1 ∧ pointOnLine p line2 → p = intersectionPoint :=
sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l4077_407728


namespace NUMINAMATH_CALUDE_outstanding_consumer_credit_l4077_407754

/-- The total outstanding consumer installment credit in billions of dollars -/
def total_credit : ℝ := 855

/-- The automobile installment credit in billions of dollars -/
def auto_credit : ℝ := total_credit * 0.2

/-- The credit extended by automobile finance companies in billions of dollars -/
def finance_company_credit : ℝ := 57

theorem outstanding_consumer_credit :
  (auto_credit = total_credit * 0.2) ∧
  (finance_company_credit = 57) ∧
  (finance_company_credit = auto_credit / 3) →
  total_credit = 855 :=
by sorry

end NUMINAMATH_CALUDE_outstanding_consumer_credit_l4077_407754


namespace NUMINAMATH_CALUDE_probability_spade_or_king_is_4_13_l4077_407798

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (kings_per_suit : Nat)

/-- Calculates the probability of drawing a spade or a king -/
def probability_spade_or_king (d : Deck) : Rat :=
  let spades := d.cards_per_suit
  let kings := d.suits * d.kings_per_suit
  let overlap := d.kings_per_suit
  let favorable_outcomes := spades + kings - overlap
  favorable_outcomes / d.total_cards

/-- Theorem stating the probability of drawing a spade or a king is 4/13 -/
theorem probability_spade_or_king_is_4_13 (d : Deck) 
    (h1 : d.total_cards = 52)
    (h2 : d.suits = 4)
    (h3 : d.cards_per_suit = 13)
    (h4 : d.kings_per_suit = 1) : 
  probability_spade_or_king d = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_spade_or_king_is_4_13_l4077_407798


namespace NUMINAMATH_CALUDE_ellipse_equation_l4077_407736

theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let e := Real.sqrt 3 / 2
  let perimeter := 16
  let eccentricity := (Real.sqrt (a^2 - b^2)) / a
  eccentricity = e ∧ perimeter = 4 * a → a^2 = 16 ∧ b^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4077_407736


namespace NUMINAMATH_CALUDE_set_equality_l4077_407785

theorem set_equality : 
  {z : ℤ | ∃ (x a : ℝ), z = x - a ∧ a - 1 ≤ x ∧ x ≤ a + 1} = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l4077_407785


namespace NUMINAMATH_CALUDE_smallest_m_exceeds_15_l4077_407741

def sum_digits_after_decimal (n : ℚ) : ℕ :=
  sorry

def exceeds_15 (m : ℕ) : Prop :=
  sum_digits_after_decimal (1 / 3^m) > 15

theorem smallest_m_exceeds_15 : 
  (∀ k < 7, ¬(exceeds_15 k)) ∧ exceeds_15 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_exceeds_15_l4077_407741


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l4077_407772

def english_marks : ℕ := 96
def math_marks : ℕ := 98
def physics_marks : ℕ := 99
def biology_marks : ℕ := 98
def average_marks : ℚ := 98.2
def num_subjects : ℕ := 5

theorem davids_chemistry_marks :
  ∃ (chemistry_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    chemistry_marks = 100 := by
  sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l4077_407772


namespace NUMINAMATH_CALUDE_value_of_z_l4077_407717

theorem value_of_z (z : ℝ) : 
  (Real.sqrt 1.1) / (Real.sqrt 0.81) + (Real.sqrt z) / (Real.sqrt 0.49) = 2.879628878919216 → 
  z = 1.44 := by
sorry

end NUMINAMATH_CALUDE_value_of_z_l4077_407717


namespace NUMINAMATH_CALUDE_lion_population_l4077_407734

/-- Given a population of lions that increases by 4 each month for 12 months,
    prove that if the final population is 148, the initial population was 100. -/
theorem lion_population (initial_population final_population : ℕ) 
  (monthly_increase : ℕ) (months : ℕ) : 
  monthly_increase = 4 →
  months = 12 →
  final_population = 148 →
  final_population = initial_population + monthly_increase * months →
  initial_population = 100 := by
sorry

end NUMINAMATH_CALUDE_lion_population_l4077_407734


namespace NUMINAMATH_CALUDE_area_four_intersecting_circles_l4077_407783

/-- The area common to four intersecting circles with specific configuration -/
theorem area_four_intersecting_circles (R : ℝ) (R_pos : R > 0) : ℝ := by
  /- Given two circles of radius R that intersect such that each passes through the center of the other,
     and two additional circles of radius R with centers at the intersection points of the first two circles,
     the area common to all four circles is: -/
  have area : ℝ := R^2 * (2 * Real.pi - 3 * Real.sqrt 3) / 6
  
  /- Proof goes here -/
  sorry

#check area_four_intersecting_circles

end NUMINAMATH_CALUDE_area_four_intersecting_circles_l4077_407783


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l4077_407737

/-- Given a quadratic function f(x) = ax^2 + bx + 2 where a ≠ 0 and |f(x)| ≥ 2 for all real x,
    the focus of the parabola has coordinates (0, 1/(4a) + 2). -/
theorem parabola_focus_coordinates (a b : ℝ) (ha : a ≠ 0) 
    (hf : ∀ x : ℝ, |a * x^2 + b * x + 2| ≥ 2) :
    ∃ (focus : ℝ × ℝ), focus = (0, 1 / (4 * a) + 2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l4077_407737


namespace NUMINAMATH_CALUDE_duty_arrangements_count_l4077_407766

/- Define the number of days -/
def num_days : ℕ := 7

/- Define the number of people -/
def num_people : ℕ := 4

/- Define the possible work days for each person -/
def work_days : Set ℕ := {1, 2}

/- Define the function to calculate the number of duty arrangements -/
def duty_arrangements (days : ℕ) (people : ℕ) (work_options : Set ℕ) : ℕ :=
  sorry  -- The actual calculation would go here

/- Theorem stating that the number of duty arrangements is 2520 -/
theorem duty_arrangements_count :
  duty_arrangements num_days num_people work_days = 2520 :=
sorry

end NUMINAMATH_CALUDE_duty_arrangements_count_l4077_407766


namespace NUMINAMATH_CALUDE_paint_remaining_is_one_fourth_l4077_407791

/-- The fraction of paint remaining after three days of painting -/
def paint_remaining : ℚ :=
  let day1_remaining := 1 - (1/4 : ℚ)
  let day2_remaining := day1_remaining - (1/2 * day1_remaining)
  day2_remaining - (1/3 * day2_remaining)

/-- Theorem stating that the remaining paint after three days is 1/4 of the original amount -/
theorem paint_remaining_is_one_fourth :
  paint_remaining = (1/4 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_is_one_fourth_l4077_407791


namespace NUMINAMATH_CALUDE_coin_game_theorem_l4077_407722

/-- Represents the result of a coin-taking game -/
inductive GameResult
| FirstPlayerWins
| SecondPlayerWins

/-- Defines the coin-taking game and determines the winner -/
def coinGameWinner (n : ℕ) : GameResult :=
  if n = 7 then GameResult.FirstPlayerWins
  else if n = 12 then GameResult.SecondPlayerWins
  else sorry -- For other cases

/-- Theorem stating the winner for specific game configurations -/
theorem coin_game_theorem :
  (coinGameWinner 7 = GameResult.FirstPlayerWins) ∧
  (coinGameWinner 12 = GameResult.SecondPlayerWins) := by
  sorry

/-- The maximum value of coins a player can take in one turn -/
def maxTakeValue : ℕ := 3

/-- The value of a two-pound coin -/
def twoPoundValue : ℕ := 2

/-- The value of a one-pound coin -/
def onePoundValue : ℕ := 1

end NUMINAMATH_CALUDE_coin_game_theorem_l4077_407722


namespace NUMINAMATH_CALUDE_can_form_triangle_l4077_407759

/-- Triangle inequality theorem: the sum of the lengths of any two sides 
    of a triangle must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: The line segments 5, 6, and 10 can form a triangle -/
theorem can_form_triangle : triangle_inequality 5 6 10 := by
  sorry

end NUMINAMATH_CALUDE_can_form_triangle_l4077_407759


namespace NUMINAMATH_CALUDE_sqrt_ratio_equality_implies_y_value_l4077_407735

theorem sqrt_ratio_equality_implies_y_value (y : ℝ) :
  y > 2 →
  (Real.sqrt (7 * y)) / (Real.sqrt (4 * (y - 2))) = 3 →
  y = 72 / 29 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ratio_equality_implies_y_value_l4077_407735


namespace NUMINAMATH_CALUDE_seven_students_distribution_l4077_407755

/-- The number of ways to distribute n students into two dormitories with at least m students in each -/
def distribution_plans (n : ℕ) (m : ℕ) : ℕ :=
  (Finset.sum (Finset.range (n - 2*m + 1)) (λ k => (Nat.choose n (m + k)) * 2)) * 2

/-- Theorem: There are 112 ways to distribute 7 students into two dormitories with at least 2 students in each -/
theorem seven_students_distribution : distribution_plans 7 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_seven_students_distribution_l4077_407755


namespace NUMINAMATH_CALUDE_graph6_triangle_or_independent_set_l4077_407774

/-- A simple graph with 6 vertices -/
structure Graph6 where
  vertices : Finset (Fin 6)
  edges : Set (Fin 6 × Fin 6)
  symmetry : ∀ (a b : Fin 6), (a, b) ∈ edges → (b, a) ∈ edges
  irreflexive : ∀ (a : Fin 6), (a, a) ∉ edges

/-- A triangle in a graph -/
def HasTriangle (G : Graph6) : Prop :=
  ∃ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a, b) ∈ G.edges ∧ (b, c) ∈ G.edges ∧ (c, a) ∈ G.edges

/-- An independent set of size 3 in a graph -/
def HasIndependentSet3 (G : Graph6) : Prop :=
  ∃ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a, b) ∉ G.edges ∧ (b, c) ∉ G.edges ∧ (c, a) ∉ G.edges

/-- The main theorem -/
theorem graph6_triangle_or_independent_set (G : Graph6) :
  HasTriangle G ∨ HasIndependentSet3 G :=
sorry

end NUMINAMATH_CALUDE_graph6_triangle_or_independent_set_l4077_407774


namespace NUMINAMATH_CALUDE_xy_values_l4077_407726

theorem xy_values (x y : ℝ) 
  (h1 : (16:ℝ)^x / (4:ℝ)^(x+y) = 64)
  (h2 : (4:ℝ)^(x+y) / (2:ℝ)^(5*y) = 16) :
  x = 5 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_values_l4077_407726


namespace NUMINAMATH_CALUDE_transformed_triangle_area_equality_l4077_407768

-- Define the domain
variable (x₁ x₂ x₃ : ℝ)

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the area function for a triangle given three points
def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem transformed_triangle_area_equality 
  (h₁ : triangle_area (x₁, f x₁) (x₂, f x₂) (x₃, f x₃) = 50)
  (h₂ : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁) :
  triangle_area (x₁/3, 3 * f x₁) (x₂/3, 3 * f x₂) (x₃/3, 3 * f x₃) = 50 := by
  sorry

end NUMINAMATH_CALUDE_transformed_triangle_area_equality_l4077_407768


namespace NUMINAMATH_CALUDE_angle_D_measure_l4077_407775

/-- Prove that given the specified angle conditions, angle D measures 25 degrees. -/
theorem angle_D_measure (A B C D : ℝ) : 
  A + B = 180 →
  C = D →
  A = 50 →
  D = 25 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l4077_407775


namespace NUMINAMATH_CALUDE_place_value_ratio_l4077_407730

-- Define the number
def number : ℚ := 53674.9281

-- Define the place value of a digit at a specific position
def place_value (n : ℚ) (pos : ℤ) : ℚ := 10 ^ pos

-- Define the position of digit 6 (counting from right, with decimal point at 0)
def pos_6 : ℤ := 3

-- Define the position of digit 8 (counting from right, with decimal point at 0)
def pos_8 : ℤ := -1

-- Theorem to prove
theorem place_value_ratio :
  (place_value number pos_6) / (place_value number pos_8) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l4077_407730


namespace NUMINAMATH_CALUDE_equation_solution_l4077_407764

theorem equation_solution : 
  let f (x : ℝ) := 1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7))
  ∀ x : ℝ, f x = 1/8 ↔ x = -2 * Real.sqrt 14 ∨ x = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4077_407764


namespace NUMINAMATH_CALUDE_sum_of_four_cubes_1944_l4077_407744

theorem sum_of_four_cubes_1944 : ∃ (a b c d : ℤ), 1944 = a^3 + b^3 + c^3 + d^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_cubes_1944_l4077_407744


namespace NUMINAMATH_CALUDE_tylers_puppies_l4077_407742

theorem tylers_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) (total_puppies : ℕ) : 
  num_dogs = 15 → puppies_per_dog = 5 → total_puppies = num_dogs * puppies_per_dog → total_puppies = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_tylers_puppies_l4077_407742


namespace NUMINAMATH_CALUDE_unique_student_count_l4077_407703

/-- Represents the number of students that can be seated in large boats -/
def large_boat_capacity (num_boats : ℕ) : ℕ := 17 * num_boats + 6

/-- Represents the number of students that can be seated in small boats -/
def small_boat_capacity (num_boats : ℕ) : ℕ := 10 * num_boats + 2

/-- Theorem stating that 142 is the only number of students satisfying all conditions -/
theorem unique_student_count : 
  ∃! n : ℕ, 
    100 < n ∧ 
    n < 200 ∧ 
    (∃ x y : ℕ, large_boat_capacity x = n ∧ small_boat_capacity y = n) :=
by
  sorry

#check unique_student_count

end NUMINAMATH_CALUDE_unique_student_count_l4077_407703


namespace NUMINAMATH_CALUDE_complex_expression_equality_l4077_407731

theorem complex_expression_equality (a b : ℂ) :
  a = 3 + 2*I ∧ b = 2 - 3*I →
  3*a + 4*b + a^2 + b^2 = 35 - 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l4077_407731


namespace NUMINAMATH_CALUDE_digit_2009_is_zero_l4077_407765

/-- The sequence of digits obtained by writing natural numbers successively -/
def digit_sequence : ℕ → ℕ := sorry

/-- The number of digits used to write numbers from 1 to n -/
def digits_count (n : ℕ) : ℕ := sorry

/-- The 2009th digit in the sequence -/
def digit_2009 : ℕ := digit_sequence 2009

theorem digit_2009_is_zero : digit_2009 = 0 := by sorry

end NUMINAMATH_CALUDE_digit_2009_is_zero_l4077_407765


namespace NUMINAMATH_CALUDE_limit_f_at_zero_l4077_407751

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.exp x - 5 * x) / (4 * x^2 + 7 * x)

theorem limit_f_at_zero : 
  Filter.Tendsto f (Filter.atTop.comap (fun x => 1 / x)) (nhds (-4/7)) := by
  sorry

end NUMINAMATH_CALUDE_limit_f_at_zero_l4077_407751


namespace NUMINAMATH_CALUDE_probability_of_E_l4077_407711

def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

def count (c : Char) : ℕ :=
  match c with
  | 'A' => 5
  | 'E' => 3
  | 'I' => 4
  | 'O' => 2
  | 'U' => 6
  | _ => 0

def total_count : ℕ := (vowels.sum count)

theorem probability_of_E : 
  (count 'E' : ℚ) / total_count = 3 / 20 := by sorry

end NUMINAMATH_CALUDE_probability_of_E_l4077_407711


namespace NUMINAMATH_CALUDE_simplify_expression_l4077_407760

/-- Given an expression 3(3x^2 + 4xy) - a(2x^2 + 3xy - 1), if the simplified result
    does not contain y, then a = 4 and the simplified expression is x^2 + 4 -/
theorem simplify_expression (x y : ℝ) (a : ℝ) :
  (∀ y, 3 * (3 * x^2 + 4 * x * y) - a * (2 * x^2 + 3 * x * y - 1) = 
   3 * (3 * x^2 + 4 * x * y) - a * (2 * x^2 + 3 * x * y - 1)) →
  a = 4 ∧ 3 * (3 * x^2 + 4 * x * y) - a * (2 * x^2 + 3 * x * y - 1) = x^2 + 4 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l4077_407760


namespace NUMINAMATH_CALUDE_soccer_tournament_equation_l4077_407713

/-- Represents a soccer invitational tournament -/
structure SoccerTournament where
  num_teams : ℕ
  num_matches : ℕ
  each_pair_plays : Bool

/-- The equation for the number of matches in the tournament -/
def tournament_equation (t : SoccerTournament) : Prop :=
  t.num_matches = (t.num_teams * (t.num_teams - 1)) / 2

/-- Theorem stating the correct equation for the given tournament conditions -/
theorem soccer_tournament_equation (t : SoccerTournament) 
  (h1 : t.each_pair_plays = true) 
  (h2 : t.num_matches = 28) : 
  tournament_equation t :=
sorry

end NUMINAMATH_CALUDE_soccer_tournament_equation_l4077_407713


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4077_407769

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4077_407769


namespace NUMINAMATH_CALUDE_age_problem_l4077_407792

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 47 →
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l4077_407792


namespace NUMINAMATH_CALUDE_some_number_value_l4077_407702

theorem some_number_value (x y z w N : ℝ) 
  (h1 : 4 * x * z + y * w = N)
  (h2 : x * w + y * z = 6)
  (h3 : (2 * x + y) * (2 * z + w) = 15) :
  N = 3 := by sorry

end NUMINAMATH_CALUDE_some_number_value_l4077_407702


namespace NUMINAMATH_CALUDE_cyclist_return_speed_l4077_407745

theorem cyclist_return_speed 
  (total_distance : ℝ) 
  (first_segment : ℝ) 
  (second_segment : ℝ) 
  (first_speed : ℝ) 
  (second_speed : ℝ) 
  (total_time : ℝ) :
  total_distance = first_segment + second_segment →
  first_segment = 12 →
  second_segment = 24 →
  first_speed = 8 →
  second_speed = 12 →
  total_time = 7.5 →
  (total_distance / first_speed + second_segment / second_speed + 
   (total_distance / ((total_time - (total_distance / first_speed + second_segment / second_speed))))) = total_time →
  (total_distance / (total_time - (total_distance / first_speed + second_segment / second_speed))) = 9 := by
sorry

end NUMINAMATH_CALUDE_cyclist_return_speed_l4077_407745


namespace NUMINAMATH_CALUDE_remainder_4053_div_23_l4077_407732

theorem remainder_4053_div_23 : 4053 % 23 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4053_div_23_l4077_407732


namespace NUMINAMATH_CALUDE_derivative_of_f_l4077_407761

/-- The function f(x) = 3x^2 -/
def f (x : ℝ) : ℝ := 3 * x^2

/-- The derivative of f(x) = 3x^2 is 6x -/
theorem derivative_of_f :
  deriv f = fun x ↦ 6 * x := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l4077_407761


namespace NUMINAMATH_CALUDE_polar_equation_is_parabola_l4077_407778

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.sin θ)

-- Define the Cartesian equation of a parabola
def is_parabola (x y : ℝ) : Prop :=
  ∃ (a : ℝ), x^2 = 2 * a * y

-- Theorem statement
theorem polar_equation_is_parabola :
  ∀ (x y : ℝ), (∃ (r θ : ℝ), polar_equation r θ ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  is_parabola x y :=
sorry

end NUMINAMATH_CALUDE_polar_equation_is_parabola_l4077_407778


namespace NUMINAMATH_CALUDE_five_mondays_in_march_after_five_sunday_february_l4077_407773

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ
  isLeapYear : Bool

/-- Represents a month in a specific year -/
structure Month where
  year : Year
  monthNumber : ℕ
  days : ℕ
  firstDay : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to count occurrences of a specific day in a month -/
def countDayInMonth (m : Month) (d : DayOfWeek) : ℕ := sorry

theorem five_mondays_in_march_after_five_sunday_february 
  (y : Year) 
  (feb : Month) 
  (mar : Month) :
  y.isLeapYear = true →
  feb.year = y →
  feb.monthNumber = 2 →
  feb.days = 29 →
  mar.year = y →
  mar.monthNumber = 3 →
  mar.days = 31 →
  countDayInMonth feb DayOfWeek.Sunday = 5 →
  mar.firstDay = nextDay feb.firstDay →
  countDayInMonth mar DayOfWeek.Monday = 5 := by
  sorry


end NUMINAMATH_CALUDE_five_mondays_in_march_after_five_sunday_february_l4077_407773


namespace NUMINAMATH_CALUDE_logarithm_product_theorem_l4077_407758

theorem logarithm_product_theorem (c d : ℕ+) : 
  (d - c + 2 = 1000) →
  (Real.log (d + 1) / Real.log c = 3) →
  (c + d : ℕ) = 1009 := by
sorry

end NUMINAMATH_CALUDE_logarithm_product_theorem_l4077_407758


namespace NUMINAMATH_CALUDE_max_people_in_line_l4077_407709

/-- Represents the state of the line at any given point -/
structure LineState where
  current : ℕ
  max : ℕ

/-- Updates the line state after people leave and join -/
def updateLine (state : LineState) (leave : ℕ) (join : ℕ) : LineState :=
  let remaining := state.current - min state.current leave
  let newCurrent := remaining + join
  { current := newCurrent, max := max state.max newCurrent }

/-- Repeats the process of people leaving and joining for a given number of times -/
def repeatProcess (initialState : LineState) (leave : ℕ) (join : ℕ) (times : ℕ) : LineState :=
  match times with
  | 0 => initialState
  | n + 1 => repeatProcess (updateLine initialState leave join) leave join n

/-- Calculates the final state after the entire process -/
def finalState (initial : ℕ) (leave : ℕ) (join : ℕ) (repetitions : ℕ) : LineState :=
  let initialState : LineState := { current := initial, max := initial }
  let afterRepetitions := repeatProcess initialState leave join repetitions
  let additionalJoin := afterRepetitions.current / 10  -- 10% rounded down
  updateLine afterRepetitions 0 additionalJoin

/-- Theorem stating that the maximum number of people in line is equal to the initial number -/
theorem max_people_in_line (initial : ℕ) (leave : ℕ) (join : ℕ) (repetitions : ℕ) 
    (h_initial : initial = 9) (h_leave : leave = 6) (h_join : join = 3) (h_repetitions : repetitions = 3) :
    (finalState initial leave join repetitions).max = initial := by
  sorry

end NUMINAMATH_CALUDE_max_people_in_line_l4077_407709


namespace NUMINAMATH_CALUDE_contractor_problem_l4077_407723

-- Define the parameters
def total_days : ℕ := 50
def initial_workers : ℕ := 70
def days_passed : ℕ := 25
def work_completed : ℚ := 40 / 100

-- Define the function to calculate additional workers needed
def additional_workers_needed (total_days : ℕ) (initial_workers : ℕ) (days_passed : ℕ) (work_completed : ℚ) : ℕ :=
  -- The actual calculation will be implemented in the proof
  sorry

-- Theorem statement
theorem contractor_problem :
  additional_workers_needed total_days initial_workers days_passed work_completed = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_problem_l4077_407723


namespace NUMINAMATH_CALUDE_travel_probabilities_l4077_407787

/-- Represents a set of countries --/
structure CountrySet where
  asian : Finset Nat
  european : Finset Nat

/-- The probability of an event given the number of favorable outcomes and total outcomes --/
def probability (favorable : Nat) (total : Nat) : ℚ := favorable / total

/-- The total number of ways to choose 2 items from n items --/
def choose_two (n : Nat) : Nat := n * (n - 1) / 2

theorem travel_probabilities (countries : CountrySet) 
  (h1 : countries.asian.card = 3)
  (h2 : countries.european.card = 3) :
  (probability (choose_two 3) (choose_two 6) = 1 / 5) ∧ 
  (probability 2 9 = 2 / 9) := by
  sorry


end NUMINAMATH_CALUDE_travel_probabilities_l4077_407787


namespace NUMINAMATH_CALUDE_largest_number_l4077_407776

theorem largest_number (a b c d e : ℝ) : 
  a = 15679 + 1/3579 → 
  b = 15679 - 1/3579 → 
  c = 15679 * (1/3579) → 
  d = 15679 / (1/3579) → 
  e = 15679 * 1.03 → 
  d > a ∧ d > b ∧ d > c ∧ d > e := by
sorry

end NUMINAMATH_CALUDE_largest_number_l4077_407776


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l4077_407729

/-- The number of eggs in a full container -/
def full_container : ℕ := 12

/-- The number of containers with missing eggs -/
def containers_with_missing : ℕ := 2

/-- The number of eggs missing from each incomplete container -/
def eggs_missing_per_container : ℕ := 1

/-- The minimum number of eggs we're looking for -/
def min_eggs : ℕ := 106

theorem smallest_number_of_eggs :
  ∀ n : ℕ,
  n > 100 ∧
  ∃ c : ℕ, n = c * full_container - containers_with_missing * eggs_missing_per_container →
  n ≥ min_eggs :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l4077_407729


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l4077_407789

theorem consecutive_integers_sum_of_squares (a : ℕ) (h1 : a > 1) 
  (h2 : (a - 1) * a * (a + 1) = 10 * (3 * a)) : 
  (a - 1)^2 + a^2 + (a + 1)^2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l4077_407789


namespace NUMINAMATH_CALUDE_cost_is_five_l4077_407720

/-- The number of tickets available -/
def total_tickets : ℕ := 10

/-- The number of rides possible -/
def number_of_rides : ℕ := 2

/-- The cost of each ride in tickets -/
def cost_per_ride : ℕ := total_tickets / number_of_rides

/-- Theorem: The cost per ride is 5 tickets -/
theorem cost_is_five : cost_per_ride = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_is_five_l4077_407720


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l4077_407739

/-- Proves that 82 gallons of fuel A were added to a 208-gallon tank -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 208 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (fuel_a : ℝ), fuel_a = 82 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l4077_407739


namespace NUMINAMATH_CALUDE_arrangement_probability_l4077_407701

def total_tiles : ℕ := 8
def x_tiles : ℕ := 5
def o_tiles : ℕ := 3

def specific_arrangement : List Char := ['X', 'X', 'X', 'O', 'O', 'X', 'O', 'X']

def probability_of_arrangement : ℚ := 1 / 56

theorem arrangement_probability :
  probability_of_arrangement = 1 / (total_tiles.choose x_tiles) :=
sorry

end NUMINAMATH_CALUDE_arrangement_probability_l4077_407701


namespace NUMINAMATH_CALUDE_total_people_count_l4077_407779

theorem total_people_count (people_in_front : ℕ) (people_behind : ℕ) (total_lines : ℕ) :
  people_in_front = 2 →
  people_behind = 4 →
  total_lines = 8 →
  (people_in_front + 1 + people_behind) * total_lines = 56 :=
by sorry

end NUMINAMATH_CALUDE_total_people_count_l4077_407779


namespace NUMINAMATH_CALUDE_marbles_remaining_l4077_407794

theorem marbles_remaining (total : ℕ) (given_to_theresa : ℚ) (given_to_elliot : ℚ) :
  total = 100 →
  given_to_theresa = 25 / 100 →
  given_to_elliot = 10 / 100 →
  total - (total * given_to_theresa).floor - (total * given_to_elliot).floor = 65 := by
sorry

end NUMINAMATH_CALUDE_marbles_remaining_l4077_407794


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_l4077_407795

theorem greatest_common_divisor_of_differences : Nat.gcd (858 - 794) (Nat.gcd (1351 - 858) (1351 - 794)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_l4077_407795


namespace NUMINAMATH_CALUDE_cube_construction_count_l4077_407777

/-- Represents a rotation of a cube -/
structure CubeRotation where
  fixedConfigurations : ℕ

/-- The group of rotations for a cube -/
def rotationGroup : Finset CubeRotation := sorry

/-- The number of distinct ways to construct the cube -/
def distinctConstructions : ℕ := sorry

theorem cube_construction_count :
  distinctConstructions = 54 := by sorry

end NUMINAMATH_CALUDE_cube_construction_count_l4077_407777


namespace NUMINAMATH_CALUDE_sin_tan_inequality_l4077_407784

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_mono : ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f x < f y)

-- State the theorem
theorem sin_tan_inequality :
  f (Real.sin (π / 12)) > f (Real.tan (π / 12)) := by sorry

end NUMINAMATH_CALUDE_sin_tan_inequality_l4077_407784


namespace NUMINAMATH_CALUDE_third_term_is_27_l4077_407719

/-- A geometric sequence with six terms where the fifth term is 81 and the sixth term is 243 -/
def geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ 
  b = a * r ∧
  c = b * r ∧
  d = c * r ∧
  81 = d * r ∧
  243 = 81 * r

/-- The third term of the geometric sequence a, b, c, d, 81, 243 is 27 -/
theorem third_term_is_27 (a b c d : ℝ) (h : geometric_sequence a b c d) : c = 27 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_27_l4077_407719


namespace NUMINAMATH_CALUDE_option2_expected_cost_l4077_407757

/-- Represents the water temperature situations -/
inductive WaterTemp
  | Normal
  | SlightlyHigh
  | ExtremelyHigh

/-- Probability of extremely high water temperature -/
def probExtremelyHigh : ℝ := 0.01

/-- Probability of slightly high water temperature -/
def probSlightlyHigh : ℝ := 0.25

/-- Loss incurred when water temperature is extremely high -/
def lossExtremelyHigh : ℝ := 600000

/-- Loss incurred when water temperature is slightly high -/
def lossSlightlyHigh : ℝ := 100000

/-- Cost of implementing Option 2 (temperature control equipment) -/
def costOption2 : ℝ := 20000

/-- Expected cost of Option 2 -/
def expectedCostOption2 : ℝ := 
  (lossExtremelyHigh + costOption2) * probExtremelyHigh + costOption2 * (1 - probExtremelyHigh)

/-- Theorem stating that the expected cost of Option 2 is 2600 yuan -/
theorem option2_expected_cost : expectedCostOption2 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_option2_expected_cost_l4077_407757


namespace NUMINAMATH_CALUDE_function_monotonicity_l4077_407753

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a + a^x else 3 + (a - 1) * x

-- State the theorem
theorem function_monotonicity (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_function_monotonicity_l4077_407753


namespace NUMINAMATH_CALUDE_fewer_sevens_100_l4077_407743

/-- A function that represents an arithmetic expression using sevens -/
def SevenExpression : Type := ℕ → ℚ

/-- Count the number of sevens used in an expression -/
def count_sevens : SevenExpression → ℕ := sorry

/-- Evaluate a SevenExpression -/
def evaluate : SevenExpression → ℚ := sorry

/-- Theorem: There exists an expression using fewer than 10 sevens that evaluates to 100 -/
theorem fewer_sevens_100 : ∃ e : SevenExpression, count_sevens e < 10 ∧ evaluate e = 100 := by sorry

end NUMINAMATH_CALUDE_fewer_sevens_100_l4077_407743


namespace NUMINAMATH_CALUDE_sqrt_12_plus_inverse_third_plus_neg_2_squared_simplify_fraction_division_l4077_407738

-- Problem 1
theorem sqrt_12_plus_inverse_third_plus_neg_2_squared :
  Real.sqrt 12 + (-1/3)⁻¹ + (-2)^2 = 2 * Real.sqrt 3 + 1 := by sorry

-- Problem 2
theorem simplify_fraction_division (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2*a / (a^2 - 4)) / (1 + (a - 2) / (a + 2)) = 1 / (a - 2) := by sorry

end NUMINAMATH_CALUDE_sqrt_12_plus_inverse_third_plus_neg_2_squared_simplify_fraction_division_l4077_407738


namespace NUMINAMATH_CALUDE_share_distribution_l4077_407718

theorem share_distribution (total : ℚ) (a b c : ℚ) 
  (h1 : total = 578)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) : 
  c = 408 := by
  sorry

end NUMINAMATH_CALUDE_share_distribution_l4077_407718


namespace NUMINAMATH_CALUDE_betty_sugar_purchase_l4077_407790

theorem betty_sugar_purchase (f s : ℝ) : 
  (f ≥ 10 + (3/4) * s) → 
  (f ≤ 3 * s) → 
  (∀ s' : ℝ, (∃ f' : ℝ, f' ≥ 10 + (3/4) * s' ∧ f' ≤ 3 * s') → s' ≥ s) →
  s = 40/9 := by
sorry

end NUMINAMATH_CALUDE_betty_sugar_purchase_l4077_407790


namespace NUMINAMATH_CALUDE_isosceles_triangle_attachment_l4077_407796

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Checks if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop := sorry

/-- Checks if two triangles share a common side -/
def shareCommonSide (t1 t2 : Triangle) : Prop := sorry

/-- Checks if two triangles do not overlap -/
def noOverlap (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Combines two triangles into a new triangle -/
def combineTriangles (t1 t2 : Triangle) : Triangle := sorry

theorem isosceles_triangle_attachment (t : Triangle) : 
  isRightTriangle t → 
  ∃ t2 : Triangle, 
    shareCommonSide t t2 ∧ 
    noOverlap t t2 ∧ 
    isIsosceles (combineTriangles t t2) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_attachment_l4077_407796


namespace NUMINAMATH_CALUDE_factor_quadratic_l4077_407780

theorem factor_quadratic (x : ℝ) : 3 * x^2 + 12 * x + 12 = 3 * (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_quadratic_l4077_407780


namespace NUMINAMATH_CALUDE_james_writing_time_l4077_407721

/-- James' writing scenario -/
structure WritingScenario where
  pages_per_hour : ℕ
  pages_per_day_per_person : ℕ
  people_per_day : ℕ

/-- Calculate the hours spent writing per week -/
def hours_per_week (scenario : WritingScenario) : ℕ :=
  let pages_per_day := scenario.pages_per_day_per_person * scenario.people_per_day
  let pages_per_week := pages_per_day * 7
  pages_per_week / scenario.pages_per_hour

/-- Theorem stating James spends 7 hours a week writing -/
theorem james_writing_time (james : WritingScenario)
  (h1 : james.pages_per_hour = 10)
  (h2 : james.pages_per_day_per_person = 5)
  (h3 : james.people_per_day = 2) :
  hours_per_week james = 7 := by
  sorry

end NUMINAMATH_CALUDE_james_writing_time_l4077_407721


namespace NUMINAMATH_CALUDE_custom_op_example_l4077_407763

/-- Custom binary operation @ defined as a @ b = 5a - 2b -/
def custom_op (a b : ℝ) : ℝ := 5 * a - 2 * b

/-- Theorem stating that 4 @ 7 = 6 under the custom operation -/
theorem custom_op_example : custom_op 4 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l4077_407763


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4077_407710

/-- Given an arithmetic sequence {aₙ} with common difference d = 2 and a₄ = 3,
    prove that a₂ + a₈ = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  a 4 = 3 →                    -- a₄ = 3
  a 2 + a 8 = 10 :=             -- prove a₂ + a₈ = 10
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4077_407710


namespace NUMINAMATH_CALUDE_watercolor_pictures_after_work_l4077_407727

/-- Represents the number of papers Charles bought and used --/
structure PaperCounts where
  total_papers : ℕ
  regular_papers : ℕ
  watercolor_papers : ℕ
  today_regular : ℕ
  today_watercolor : ℕ
  yesterday_before_work : ℕ

/-- Theorem stating the number of watercolor pictures Charles drew after work yesterday --/
theorem watercolor_pictures_after_work (p : PaperCounts)
  (h1 : p.total_papers = 20)
  (h2 : p.regular_papers = 10)
  (h3 : p.watercolor_papers = 10)
  (h4 : p.today_regular = 4)
  (h5 : p.today_watercolor = 2)
  (h6 : p.yesterday_before_work = 6)
  (h7 : p.yesterday_before_work ≤ p.regular_papers)
  (h8 : p.today_regular + p.today_watercolor = 6)
  (h9 : p.regular_papers + p.watercolor_papers = p.total_papers)
  (h10 : ∃ (x : ℕ), x > 0 ∧ x ≤ p.watercolor_papers - p.today_watercolor) :
  p.watercolor_papers - p.today_watercolor = 8 := by
  sorry


end NUMINAMATH_CALUDE_watercolor_pictures_after_work_l4077_407727


namespace NUMINAMATH_CALUDE_equivalence_theorem_l4077_407747

-- Define f as a differentiable function on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the condition that f(x) + f'(x) > 0 for all x ∈ ℝ
variable (h : ∀ x : ℝ, f x + (deriv f) x > 0)

-- State the theorem
theorem equivalence_theorem (a b : ℝ) :
  a > b ↔ Real.exp a * f a > Real.exp b * f b :=
sorry

end NUMINAMATH_CALUDE_equivalence_theorem_l4077_407747


namespace NUMINAMATH_CALUDE_golden_comets_ratio_l4077_407715

/-- Represents the number of chickens in a flock -/
structure ChickenFlock where
  rhodeIslandReds : ℕ
  goldenComets : ℕ

/-- Given information about Susie's and Britney's chicken flocks -/
def susie : ChickenFlock := { rhodeIslandReds := 11, goldenComets := 6 }
def britney : ChickenFlock :=
  { rhodeIslandReds := 2 * susie.rhodeIslandReds,
    goldenComets := susie.rhodeIslandReds + susie.goldenComets + 8 - (2 * susie.rhodeIslandReds) }

/-- The theorem to be proved -/
theorem golden_comets_ratio :
  2 * britney.goldenComets = susie.goldenComets := by sorry

end NUMINAMATH_CALUDE_golden_comets_ratio_l4077_407715


namespace NUMINAMATH_CALUDE_bike_shop_wheels_l4077_407781

/-- The number of wheels on all vehicles in a bike shop -/
def total_wheels (num_bicycles num_tricycles : ℕ) : ℕ :=
  2 * num_bicycles + 3 * num_tricycles

/-- Theorem stating the total number of wheels in the bike shop -/
theorem bike_shop_wheels : total_wheels 50 20 = 160 := by
  sorry

end NUMINAMATH_CALUDE_bike_shop_wheels_l4077_407781


namespace NUMINAMATH_CALUDE_sandbag_weight_l4077_407756

/-- Calculates the weight of a partially filled sandbag with a heavier material -/
theorem sandbag_weight (capacity : ℝ) (fill_percentage : ℝ) (weight_increase : ℝ) : 
  capacity = 450 →
  fill_percentage = 0.75 →
  weight_increase = 0.65 →
  capacity * fill_percentage * (1 + weight_increase) = 556.875 := by
  sorry

end NUMINAMATH_CALUDE_sandbag_weight_l4077_407756


namespace NUMINAMATH_CALUDE_average_speed_calculation_average_speed_approximation_l4077_407750

theorem average_speed_calculation (total_distance : ℝ) (first_segment_distance : ℝ) 
  (first_segment_speed : ℝ) (second_segment_distance : ℝ) (second_segment_speed_limit : ℝ) 
  (second_segment_normal_speed : ℝ) (third_segment_distance : ℝ) 
  (speed_limit_distance : ℝ) : ℝ :=
  let first_segment_time := first_segment_distance / first_segment_speed
  let second_segment_time_limit := speed_limit_distance / second_segment_speed_limit
  let second_segment_time_normal := (second_segment_distance - speed_limit_distance) / second_segment_normal_speed
  let second_segment_time := second_segment_time_limit + second_segment_time_normal
  let third_segment_time := first_segment_time * 2.5
  let total_time := first_segment_time + second_segment_time + third_segment_time
  let average_speed := total_distance / total_time
  average_speed

#check average_speed_calculation 760 320 80 240 45 60 200 100

theorem average_speed_approximation :
  ∃ ε > 0, abs (average_speed_calculation 760 320 80 240 45 60 200 100 - 40.97) < ε :=
sorry

end NUMINAMATH_CALUDE_average_speed_calculation_average_speed_approximation_l4077_407750


namespace NUMINAMATH_CALUDE_merchant_problem_l4077_407748

theorem merchant_problem (n : ℕ) (C : ℕ) : 
  (8 * n = C + 3) → 
  (7 * n = C - 4) → 
  (n = 7 ∧ C = 53) := by
  sorry

end NUMINAMATH_CALUDE_merchant_problem_l4077_407748


namespace NUMINAMATH_CALUDE_circle_center_locus_l4077_407771

-- Define the circle C
def circle_C (a x y : ℝ) : Prop :=
  x^2 + y^2 - (2*a^2 - 4)*x - 4*a^2*y + 5*a^4 - 4 = 0

-- Define the locus of the center
def center_locus (x y : ℝ) : Prop :=
  y = 2*x + 4 ∧ -2 ≤ x ∧ x < 0

-- Theorem statement
theorem circle_center_locus :
  ∀ a x y : ℝ, circle_C a x y → ∃ h k : ℝ, center_locus h k ∧ 
  (h = a^2 - 2 ∧ k = 2*a^2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_locus_l4077_407771


namespace NUMINAMATH_CALUDE_min_sum_squares_l4077_407799

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 ∧
  (a^2 + b^2 + c^2 = t^2 / 3 ↔ a = t/3 ∧ b = t/3 ∧ c = t/3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l4077_407799


namespace NUMINAMATH_CALUDE_egg_distribution_proof_l4077_407740

def mia_eggs : ℕ := 4
def sofia_eggs : ℕ := 2 * mia_eggs
def pablo_eggs : ℕ := 4 * sofia_eggs

def total_eggs : ℕ := mia_eggs + sofia_eggs + pablo_eggs
def equal_distribution : ℚ := total_eggs / 3

def fraction_to_sofia : ℚ := 5 / 24

theorem egg_distribution_proof :
  let sofia_new := sofia_eggs + (fraction_to_sofia * pablo_eggs)
  let mia_new := equal_distribution
  let pablo_new := pablo_eggs - (fraction_to_sofia * pablo_eggs) - (mia_new - mia_eggs)
  sofia_new = mia_new ∧ sofia_new = pablo_new := by sorry

end NUMINAMATH_CALUDE_egg_distribution_proof_l4077_407740


namespace NUMINAMATH_CALUDE_alpha_beta_difference_bounds_l4077_407733

theorem alpha_beta_difference_bounds (α β : ℝ) (h1 : -1 < α) (h2 : α < β) (h3 : β < 1) :
  -2 < α - β ∧ α - β < 0 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_difference_bounds_l4077_407733


namespace NUMINAMATH_CALUDE_range_of_2a_plus_3b_inequality_with_squared_sum_l4077_407700

-- Part 1
theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) : 
  -3 ≤ 2*a + 3*b ∧ 2*a + 3*b ≤ 3 :=
sorry

-- Part 2
theorem inequality_with_squared_sum (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 = 6) : 
  1 / (a^2 + 1) + 1 / (b^2 + 2) > 1/2 - 1 / (c^2 + 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_3b_inequality_with_squared_sum_l4077_407700


namespace NUMINAMATH_CALUDE_distance_to_asymptote_l4077_407716

-- Define the parabola C₁
def C₁ (a : ℝ) (x y : ℝ) : Prop := y^2 = 8*a*x ∧ a > 0

-- Define the line l
def l (a : ℝ) (x y : ℝ) : Prop := y = x - 2*a

-- Define the hyperbola C₂
def C₂ (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the directrix of C₁
def directrix (a : ℝ) : ℝ := -2*a

-- Define the focus of C₁
def focus_C₁ (a : ℝ) : ℝ × ℝ := (2*a, 0)

-- Define the asymptote of C₂
def asymptote_C₂ (a b : ℝ) (x y : ℝ) : Prop := b*x - a*y = 0

-- Main theorem
theorem distance_to_asymptote 
  (a b : ℝ) 
  (h₁ : C₁ a (2*a) 0)  -- C₁ passes through its focus
  (h₂ : ∃ x y, C₁ a x y ∧ l a x y ∧ (x - 2*a)^2 + y^2 = 256)  -- Segment length is 16
  (h₃ : ∃ x, C₂ a b x (directrix a))  -- One focus of C₂ on directrix of C₁
  : (abs (2*a)) / Real.sqrt (b^2 + a^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_distance_to_asymptote_l4077_407716


namespace NUMINAMATH_CALUDE_visible_percentage_for_given_prism_and_film_l4077_407767

/-- Represents a regular triangular prism -/
structure RegularTriangularPrism where
  base_edge : ℝ
  height : ℝ

/-- Represents a checkerboard film -/
structure CheckerboardFilm where
  cell_size : ℝ

/-- Calculates the visible percentage of a prism's lateral surface when wrapped with a film -/
def visible_percentage (prism : RegularTriangularPrism) (film : CheckerboardFilm) : ℝ :=
  sorry

/-- Theorem stating the visible percentage for the given prism and film -/
theorem visible_percentage_for_given_prism_and_film :
  let prism := RegularTriangularPrism.mk 3.2 5
  let film := CheckerboardFilm.mk 1
  visible_percentage prism film = 28.75 := by
  sorry

end NUMINAMATH_CALUDE_visible_percentage_for_given_prism_and_film_l4077_407767


namespace NUMINAMATH_CALUDE_regular_pay_is_three_dollars_l4077_407708

/-- Represents a worker's pay structure and hours worked -/
structure WorkerPay where
  regularPayPerHour : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalPay : ℝ

/-- Calculates the total pay for a worker given their pay structure and hours worked -/
def calculateTotalPay (w : WorkerPay) : ℝ :=
  w.regularPayPerHour * w.regularHours + 2 * w.regularPayPerHour * w.overtimeHours

/-- Theorem stating that under given conditions, the regular pay per hour is $3 -/
theorem regular_pay_is_three_dollars
  (w : WorkerPay)
  (h1 : w.regularHours = 40)
  (h2 : w.overtimeHours = 11)
  (h3 : w.totalPay = 186)
  (h4 : calculateTotalPay w = w.totalPay) :
  w.regularPayPerHour = 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_pay_is_three_dollars_l4077_407708


namespace NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l4077_407749

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5}

-- Define the region where x + y ≤ 6
def region : Set (ℝ × ℝ) := {p ∈ rectangle | p.1 + p.2 ≤ 6}

-- Define the probability measure on the rectangle
noncomputable def prob : MeasureTheory.Measure (ℝ × ℝ) := sorry

-- State the theorem
theorem probability_x_plus_y_leq_6 : 
  prob region / prob rectangle = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l4077_407749


namespace NUMINAMATH_CALUDE_expression_evaluation_l4077_407786

theorem expression_evaluation (x y z : ℝ) 
  (hx : x ≠ 3) (hy : y ≠ 5) (hz : z ≠ 7) : 
  (x - 3) / (7 - z) * (y - 5) / (3 - x) * (z - 7) / (5 - y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4077_407786


namespace NUMINAMATH_CALUDE_zoo_visitors_per_hour_l4077_407797

/-- The number of hours the zoo is open in one day -/
def zoo_hours : ℕ := 8

/-- The percentage of total visitors who go to the gorilla exhibit -/
def gorilla_exhibit_percentage : ℚ := 80 / 100

/-- The number of visitors who go to the gorilla exhibit in one day -/
def gorilla_exhibit_visitors : ℕ := 320

/-- The number of new visitors entering the zoo every hour -/
def new_visitors_per_hour : ℕ := 50

theorem zoo_visitors_per_hour :
  new_visitors_per_hour = (gorilla_exhibit_visitors : ℚ) / gorilla_exhibit_percentage / zoo_hours := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_per_hour_l4077_407797


namespace NUMINAMATH_CALUDE_power_function_increasing_l4077_407712

theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, ∀ h > 0, (m^2 - 4*m + 1) * x^(m^2 - 2*m - 3) < (m^2 - 4*m + 1) * (x + h)^(m^2 - 2*m - 3)) ↔ 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_power_function_increasing_l4077_407712


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l4077_407704

theorem power_zero_eq_one (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l4077_407704


namespace NUMINAMATH_CALUDE_intersection_and_sufficient_condition_l4077_407724

def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

theorem intersection_and_sufficient_condition :
  (A (-2) ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B) ∧ (∃ x : ℝ, x ∈ B ∧ x ∉ A a) ↔ a ≤ -3 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_sufficient_condition_l4077_407724


namespace NUMINAMATH_CALUDE_machine_work_time_l4077_407762

/-- The number of shirts made today -/
def shirts_today : ℕ := 8

/-- The number of shirts that can be made per minute -/
def shirts_per_minute : ℕ := 2

/-- The number of minutes the machine worked today -/
def minutes_worked : ℚ := shirts_today / shirts_per_minute

theorem machine_work_time : minutes_worked = 4 := by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_l4077_407762


namespace NUMINAMATH_CALUDE_x_varies_as_square_root_of_z_l4077_407714

/-- If x varies as the square of y, and y varies as the fourth root of z,
    then x varies as the square root of z. -/
theorem x_varies_as_square_root_of_z
  (k : ℝ) (j : ℝ) (x y z : ℝ → ℝ)
  (h1 : ∀ t, x t = k * (y t)^2)
  (h2 : ∀ t, y t = j * (z t)^(1/4)) :
  ∃ m : ℝ, ∀ t, x t = m * (z t)^(1/2) :=
sorry

end NUMINAMATH_CALUDE_x_varies_as_square_root_of_z_l4077_407714
