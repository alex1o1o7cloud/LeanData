import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2708_270879

theorem equation_solution : ∃ x : ℝ, 13 + Real.sqrt (x + 5 * 3 - 3 * 3) = 14 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2708_270879


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2708_270807

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "Person A gets the red card"
def event_A_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Person D gets the red card"
def event_D_red (d : Distribution) : Prop := d Person.D = Card.Red

-- Statement: The events are mutually exclusive but not complementary
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A_red d ∧ event_D_red d)) ∧
  (∃ d : Distribution, ¬event_A_red d ∧ ¬event_D_red d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l2708_270807


namespace NUMINAMATH_CALUDE_bisecting_line_tangent_lines_l2708_270800

-- Define the point P and circle C
def P : ℝ × ℝ := (-1, 4)
def C : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

-- Define the center of circle C
def center : ℝ × ℝ := (2, 3)

-- Define the lines
def line1 (x y : ℝ) : Prop := x + 3*y - 11 = 0
def line2 (x y : ℝ) : Prop := y - 4 = 0
def line3 (x y : ℝ) : Prop := 3*x + 4*y - 13 = 0

-- Theorem statements
theorem bisecting_line :
  line1 P.1 P.2 ∧ line1 center.1 center.2 := by sorry

theorem tangent_lines :
  (line2 P.1 P.2 ∧ (∃ (p : ℝ × ℝ), p ∈ C ∧ line2 p.1 p.2 ∧ (∀ (q : ℝ × ℝ), q ∈ C → line2 q.1 q.2 → q = p))) ∧
  (line3 P.1 P.2 ∧ (∃ (p : ℝ × ℝ), p ∈ C ∧ line3 p.1 p.2 ∧ (∀ (q : ℝ × ℝ), q ∈ C → line3 q.1 q.2 → q = p))) := by sorry

end NUMINAMATH_CALUDE_bisecting_line_tangent_lines_l2708_270800


namespace NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_21_l2708_270841

theorem constant_term_expansion (x : ℝ) : 
  (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7) = 2*x^7 + x^5 + 7*x^3 + 2*x^6 + x^4 + 7*x^2 + 6*x^4 + 3*x^2 + 21 :=
by sorry

theorem constant_term_is_21 : 
  ∃ p : Polynomial ℝ, (Polynomial.eval 0 p = 21 ∧ 
    ∀ x : ℝ, p.eval x = (x^3 + x^2 + 3) * (2*x^4 + x^2 + 7)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_21_l2708_270841


namespace NUMINAMATH_CALUDE_b_investment_is_correct_l2708_270802

-- Define the partnership
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  duration : ℕ
  a_share : ℕ
  b_share : ℕ

-- Define the problem
def partnership_problem : Partnership :=
  { a_investment := 7000
  , b_investment := 11000  -- This is what we want to prove
  , c_investment := 18000
  , duration := 8
  , a_share := 1400
  , b_share := 2200 }

-- Theorem statement
theorem b_investment_is_correct (p : Partnership) 
  (h1 : p.a_investment = 7000)
  (h2 : p.c_investment = 18000)
  (h3 : p.duration = 8)
  (h4 : p.a_share = 1400)
  (h5 : p.b_share = 2200) :
  p.b_investment = 11000 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_is_correct_l2708_270802


namespace NUMINAMATH_CALUDE_journey_time_ratio_l2708_270854

/-- Proves that the ratio of the time taken for the journey back to the time taken for the journey to San Francisco is 3:2, given the average speeds -/
theorem journey_time_ratio (distance : ℝ) (speed_to_sf : ℝ) (avg_speed : ℝ)
  (h1 : speed_to_sf = 45)
  (h2 : avg_speed = 30)
  (h3 : distance > 0) :
  (distance / avg_speed - distance / speed_to_sf) / (distance / speed_to_sf) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_journey_time_ratio_l2708_270854


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017th_term_l2708_270899

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_sequence_2017th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_geom : geometric_sequence (λ n => match n with
    | 1 => a 1 - 1
    | 2 => a 3
    | 3 => a 5 + 5
    | _ => 0
  )) :
  a 2017 = 1010 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017th_term_l2708_270899


namespace NUMINAMATH_CALUDE_age_condition_amount_per_year_is_five_l2708_270880

/-- Mikail's age on his birthday -/
def age : ℕ := 9

/-- The total amount Mikail receives on his birthday -/
def total_amount : ℕ := 45

/-- The condition that Mikail's age is 3 times as old as he was when he was three -/
theorem age_condition : age = 3 * 3 := by sorry

/-- The amount Mikail receives per year of his age -/
def amount_per_year : ℚ := total_amount / age

/-- Proof that the amount Mikail receives per year is $5 -/
theorem amount_per_year_is_five : amount_per_year = 5 := by sorry

end NUMINAMATH_CALUDE_age_condition_amount_per_year_is_five_l2708_270880


namespace NUMINAMATH_CALUDE_expression_evaluation_l2708_270816

theorem expression_evaluation :
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  x^2 + y^2 - z^2 + 2*x*y + 2*y*z = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2708_270816


namespace NUMINAMATH_CALUDE_two_white_socks_cost_45_cents_l2708_270865

/-- The cost of a single brown sock in cents -/
def brown_sock_cost : ℕ := 300 / 15

/-- The cost of two white socks in cents -/
def white_socks_cost : ℕ := brown_sock_cost + 25

theorem two_white_socks_cost_45_cents : white_socks_cost = 45 := by
  sorry

#eval white_socks_cost

end NUMINAMATH_CALUDE_two_white_socks_cost_45_cents_l2708_270865


namespace NUMINAMATH_CALUDE_eighth_term_ratio_l2708_270857

/-- Two arithmetic sequences U and V with their partial sums Un and Vn -/
def arithmetic_sequences (U V : ℕ → ℚ) : Prop :=
  ∃ (u f v g : ℚ), ∀ n,
    U n = u + (n - 1) * f ∧
    V n = v + (n - 1) * g

/-- Partial sum of the first n terms of an arithmetic sequence -/
def partial_sum (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem eighth_term_ratio
  (U V : ℕ → ℚ)
  (h_arith : arithmetic_sequences U V)
  (h_ratio : ∀ n, partial_sum U n / partial_sum V n = (5 * n + 5) / (3 * n + 9)) :
  U 8 / V 8 = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_ratio_l2708_270857


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2708_270837

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (4 * Real.sqrt 6 + 5 * Real.sqrt 7) =
    (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧ B = 6 ∧ C = 15 ∧ D = 7 ∧ E = 79 ∧
    Int.gcd A E = 1 ∧ Int.gcd C E = 1 ∧
    Int.gcd B D = 1 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2708_270837


namespace NUMINAMATH_CALUDE_inequality_proof_l2708_270844

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt 2 * a^3 + 3 / (a * b - b^2) ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2708_270844


namespace NUMINAMATH_CALUDE_f_range_l2708_270860

/-- The function f(x) = |x+10| - |3x-1| -/
def f (x : ℝ) : ℝ := |x + 10| - |3*x - 1|

/-- The range of f is (-∞, 31] -/
theorem f_range :
  Set.range f = Set.Iic 31 := by sorry

end NUMINAMATH_CALUDE_f_range_l2708_270860


namespace NUMINAMATH_CALUDE_min_value_fraction_l2708_270805

theorem min_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∃ (m : ℝ), m = (x + y) / x ∧ ∀ (z w : ℝ), (-6 ≤ z ∧ z ≤ -3) → (3 ≤ w ∧ w ≤ 6) → m ≤ (z + w) / z :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2708_270805


namespace NUMINAMATH_CALUDE_line_points_l2708_270870

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def lies_on_line (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

/-- The main theorem -/
theorem line_points : 
  let p1 : Point := ⟨8, 16⟩
  let p2 : Point := ⟨2, 4⟩
  let points_on_line : List Point := [⟨5, 10⟩, ⟨7, 14⟩, ⟨10, 20⟩, ⟨3, 6⟩]
  let point_not_on_line : Point := ⟨4, 7⟩
  (∀ p ∈ points_on_line, lies_on_line p p1 p2) ∧
  ¬(lies_on_line point_not_on_line p1 p2) := by
  sorry

end NUMINAMATH_CALUDE_line_points_l2708_270870


namespace NUMINAMATH_CALUDE_product_modulo_remainder_1491_2001_mod_250_l2708_270828

theorem product_modulo (a b m : ℕ) (h : m > 0) :
  (a * b) % m = ((a % m) * (b % m)) % m :=
by sorry

theorem remainder_1491_2001_mod_250 :
  (1491 * 2001) % 250 = 241 :=
by sorry

end NUMINAMATH_CALUDE_product_modulo_remainder_1491_2001_mod_250_l2708_270828


namespace NUMINAMATH_CALUDE_inequality_proof_l2708_270823

theorem inequality_proof (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x * y * z = 1) : 
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + x) * (1 + z))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2708_270823


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2708_270858

theorem quadratic_inequality (x : ℝ) : x^2 + 4*x - 21 > 0 ↔ x < -7 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2708_270858


namespace NUMINAMATH_CALUDE_carbon_mass_percentage_l2708_270850

/-- The mass percentage of an element in a compound -/
def mass_percentage (element : String) (compound : String) : ℝ := sorry

/-- The given mass percentage of C in the compound -/
def given_percentage : ℝ := 54.55

theorem carbon_mass_percentage (compound : String) :
  mass_percentage "C" compound = given_percentage := by sorry

end NUMINAMATH_CALUDE_carbon_mass_percentage_l2708_270850


namespace NUMINAMATH_CALUDE_area_inside_rectangle_outside_circles_l2708_270893

/-- The area inside a rectangle but outside three quarter circles --/
theorem area_inside_rectangle_outside_circles (CD DA : ℝ) (r₁ r₂ r₃ : ℝ) :
  CD = 3 →
  DA = 5 →
  r₁ = 1 →
  r₂ = 2 →
  r₃ = 3 →
  (CD * DA) - ((π * r₁^2) / 4 + (π * r₂^2) / 4 + (π * r₃^2) / 4) = 15 - (7 * π / 2) :=
by sorry

end NUMINAMATH_CALUDE_area_inside_rectangle_outside_circles_l2708_270893


namespace NUMINAMATH_CALUDE_jake_weight_loss_l2708_270861

/-- Given that Jake and his sister together weigh 156 pounds and Jake's current weight is 108 pounds,
    this theorem proves that Jake needs to lose 12 pounds to weigh twice as much as his sister. -/
theorem jake_weight_loss (total_weight : ℕ) (jake_weight : ℕ) 
  (h1 : total_weight = 156)
  (h2 : jake_weight = 108) :
  jake_weight - (2 * (total_weight - jake_weight)) = 12 := by
  sorry

#check jake_weight_loss

end NUMINAMATH_CALUDE_jake_weight_loss_l2708_270861


namespace NUMINAMATH_CALUDE_max_dot_product_on_circle_l2708_270812

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define points M and N
def M : ℝ × ℝ := (2, 0)
def N : ℝ × ℝ := (0, -2)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem max_dot_product_on_circle :
  ∀ (P : ℝ × ℝ), Circle P.1 P.2 →
    dot_product (P.1 - M.1, P.2 - M.2) (P.1 - N.1, P.2 - N.2) ≤ 4 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_on_circle_l2708_270812


namespace NUMINAMATH_CALUDE_youtube_likes_problem_l2708_270897

theorem youtube_likes_problem (likes dislikes : ℕ) : 
  dislikes = likes / 2 + 100 →
  dislikes + 1000 = 2600 →
  likes = 3000 := by
sorry

end NUMINAMATH_CALUDE_youtube_likes_problem_l2708_270897


namespace NUMINAMATH_CALUDE_integer_set_average_l2708_270840

theorem integer_set_average : ∀ (a b c d : ℤ),
  a < b ∧ b < c ∧ c < d →  -- Ensure the integers are different and ordered
  d = 90 →                 -- The largest integer is 90
  a ≥ 5 →                  -- The smallest integer is at least 5
  (a + b + c + d) / 4 = 27 -- The average is 27
  := by sorry

end NUMINAMATH_CALUDE_integer_set_average_l2708_270840


namespace NUMINAMATH_CALUDE_double_perimeter_polygon_exists_l2708_270801

/-- A grid point in 2D space --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A line segment on the grid --/
inductive GridSegment
  | Horizontal : GridPoint → GridPoint → GridSegment
  | Vertical : GridPoint → GridPoint → GridSegment
  | Diagonal1x1 : GridPoint → GridPoint → GridSegment
  | Diagonal1x2 : GridPoint → GridPoint → GridSegment

/-- A polygon on the grid --/
structure GridPolygon where
  vertices : List GridPoint
  edges : List GridSegment

/-- A triangle on the grid --/
structure GridTriangle where
  vertices : Fin 3 → GridPoint
  edges : Fin 3 → GridSegment

/-- Calculate the perimeter of a grid polygon --/
def perimeterOfPolygon (p : GridPolygon) : ℕ :=
  sorry

/-- Calculate the perimeter of a grid triangle --/
def perimeterOfTriangle (t : GridTriangle) : ℕ :=
  sorry

/-- Check if a polygon has double the perimeter of a triangle --/
def hasDoublePerimeter (p : GridPolygon) (t : GridTriangle) : Prop :=
  perimeterOfPolygon p = 2 * perimeterOfTriangle t

/-- Main theorem: Given a triangle on a grid, there exists a polygon with double its perimeter --/
theorem double_perimeter_polygon_exists (t : GridTriangle) : 
  ∃ (p : GridPolygon), hasDoublePerimeter p t :=
sorry

end NUMINAMATH_CALUDE_double_perimeter_polygon_exists_l2708_270801


namespace NUMINAMATH_CALUDE_tan_pi_third_plus_cos_nineteen_sixths_pi_l2708_270804

theorem tan_pi_third_plus_cos_nineteen_sixths_pi :
  Real.tan (π / 3) + Real.cos (19 * π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_third_plus_cos_nineteen_sixths_pi_l2708_270804


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l2708_270826

theorem range_of_2a_minus_b (a b : ℝ) (ha : 2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) :
  2 < 2 * a - b ∧ 2 * a - b < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l2708_270826


namespace NUMINAMATH_CALUDE_chess_tournament_theorem_l2708_270890

/-- Represents a single-elimination chess tournament -/
structure ChessTournament where
  participants : ℕ
  winner_games : ℕ
  is_power_of_two : ∃ n : ℕ, participants = 2^n
  winner_played_six : winner_games = 6

/-- Number of participants who won at least 2 more games than they lost -/
def participants_with_two_more_wins (t : ChessTournament) : ℕ :=
  8

theorem chess_tournament_theorem (t : ChessTournament) :
  participants_with_two_more_wins t = 8 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_theorem_l2708_270890


namespace NUMINAMATH_CALUDE_parallelogram_area_specific_l2708_270874

/-- The area of a parallelogram with base b and height h. -/
def parallelogram_area (b h : ℝ) : ℝ := b * h

/-- Theorem: The area of a parallelogram with a base of 15 meters and an altitude
    that is twice the base is 450 square meters. -/
theorem parallelogram_area_specific : 
  let base : ℝ := 15
  let height : ℝ := 2 * base
  parallelogram_area base height = 450 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_specific_l2708_270874


namespace NUMINAMATH_CALUDE_smallest_w_sum_of_digits_17_l2708_270820

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest positive integer w such that 10^w - 74 has a sum of digits equal to 17 is 3 -/
theorem smallest_w_sum_of_digits_17 :
  ∀ w : ℕ+, sum_of_digits (10^(w.val) - 74) = 17 → w.val ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_sum_of_digits_17_l2708_270820


namespace NUMINAMATH_CALUDE_balance_theorem_l2708_270853

/-- Represents the weight of a ball in terms of blue balls -/
@[ext] structure BallWeight where
  blue : ℚ

/-- The weight of a red ball in terms of blue balls -/
def red_weight : BallWeight := ⟨2⟩

/-- The weight of a yellow ball in terms of blue balls -/
def yellow_weight : BallWeight := ⟨3⟩

/-- The weight of a white ball in terms of blue balls -/
def white_weight : BallWeight := ⟨5/3⟩

/-- The weight of a blue ball in terms of blue balls -/
def blue_weight : BallWeight := ⟨1⟩

theorem balance_theorem :
  2 * red_weight.blue + 4 * yellow_weight.blue + 3 * white_weight.blue = 21 * blue_weight.blue :=
by sorry

end NUMINAMATH_CALUDE_balance_theorem_l2708_270853


namespace NUMINAMATH_CALUDE_function_characterization_l2708_270884

theorem function_characterization (f : ℝ → ℝ) 
  (h1 : f 0 = 1) 
  (h2 : ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2) / 2) + (x - y)^2) : 
  ∀ x : ℝ, f x = 1 - 2 * x := by
sorry

end NUMINAMATH_CALUDE_function_characterization_l2708_270884


namespace NUMINAMATH_CALUDE_senior_tickets_first_day_l2708_270887

/- Define the variables -/
def student_ticket_price : ℕ := 9
def first_day_student_tickets : ℕ := 3
def first_day_total : ℕ := 79
def second_day_senior_tickets : ℕ := 12
def second_day_student_tickets : ℕ := 10
def second_day_total : ℕ := 246

/- Theorem to prove -/
theorem senior_tickets_first_day :
  ∃ (senior_ticket_price : ℕ) (first_day_senior_tickets : ℕ),
    senior_ticket_price * second_day_senior_tickets + 
    student_ticket_price * second_day_student_tickets = second_day_total ∧
    senior_ticket_price * first_day_senior_tickets + 
    student_ticket_price * first_day_student_tickets = first_day_total ∧
    first_day_senior_tickets = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_senior_tickets_first_day_l2708_270887


namespace NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l2708_270891

/-- A four-digit palindrome is a number of the form abba where a and b are digits and a ≠ 0 -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ (a b : ℕ), 0 < a ∧ a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem all_four_digit_palindromes_divisible_by_11 :
  ∀ n : ℕ, is_four_digit_palindrome n → n % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l2708_270891


namespace NUMINAMATH_CALUDE_college_students_count_l2708_270809

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 135) :
  boys + girls = 351 :=
sorry

end NUMINAMATH_CALUDE_college_students_count_l2708_270809


namespace NUMINAMATH_CALUDE_joan_football_games_l2708_270892

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The total number of football games Joan went to this year and last year -/
def total_games : ℕ := 9

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := total_games - games_this_year

theorem joan_football_games : games_last_year = 5 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l2708_270892


namespace NUMINAMATH_CALUDE_log_stack_sum_l2708_270851

theorem log_stack_sum : ∀ (a l n : ℕ), 
  a = 15 → l = 4 → n = 12 → 
  (n * (a + l)) / 2 = 114 := by
sorry

end NUMINAMATH_CALUDE_log_stack_sum_l2708_270851


namespace NUMINAMATH_CALUDE_optimal_bus_rental_solution_l2708_270824

/-- Represents a bus rental problem with two types of buses -/
structure BusRental where
  tourists : ℕ
  capacity_A : ℕ
  cost_A : ℕ
  capacity_B : ℕ
  cost_B : ℕ
  max_total_buses : ℕ
  max_B_minus_A : ℕ

/-- Represents a solution to the bus rental problem -/
structure BusRentalSolution where
  buses_A : ℕ
  buses_B : ℕ
  total_cost : ℕ

/-- Check if a solution is valid for a given bus rental problem -/
def is_valid_solution (problem : BusRental) (solution : BusRentalSolution) : Prop :=
  solution.buses_A * problem.capacity_A + solution.buses_B * problem.capacity_B ≥ problem.tourists ∧
  solution.buses_A + solution.buses_B ≤ problem.max_total_buses ∧
  solution.buses_B - solution.buses_A ≤ problem.max_B_minus_A ∧
  solution.total_cost = solution.buses_A * problem.cost_A + solution.buses_B * problem.cost_B

/-- The main theorem stating that the given solution is optimal -/
theorem optimal_bus_rental_solution (problem : BusRental)
  (h_problem : problem = {
    tourists := 900,
    capacity_A := 36,
    cost_A := 1600,
    capacity_B := 60,
    cost_B := 2400,
    max_total_buses := 21,
    max_B_minus_A := 7
  })
  (solution : BusRentalSolution)
  (h_solution : solution = {
    buses_A := 5,
    buses_B := 12,
    total_cost := 36800
  }) :
  is_valid_solution problem solution ∧
  ∀ (other : BusRentalSolution), is_valid_solution problem other → other.total_cost ≥ solution.total_cost :=
by sorry


end NUMINAMATH_CALUDE_optimal_bus_rental_solution_l2708_270824


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2708_270829

theorem complex_equation_solution :
  ∃ z : ℂ, (3 : ℂ) - 2 * Complex.I * z = -2 + 3 * Complex.I * z ∧ z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2708_270829


namespace NUMINAMATH_CALUDE_sherry_opposite_vertex_probability_l2708_270852

/-- The probability that Sherry is at the opposite vertex after k minutes on a triangle -/
def P (k : ℕ) : ℚ :=
  1/6 + 1/(3 * (-2)^k)

/-- Theorem: For k > 0, the probability that Sherry is at the opposite vertex after k minutes on a triangle is P(k) -/
theorem sherry_opposite_vertex_probability (k : ℕ) (h : k > 0) : 
  (1/6 : ℚ) + 1/(3 * (-2)^k) = P k := by
sorry

end NUMINAMATH_CALUDE_sherry_opposite_vertex_probability_l2708_270852


namespace NUMINAMATH_CALUDE_function_domain_implies_k_range_l2708_270896

theorem function_domain_implies_k_range
  (a : ℝ) (k : ℝ)
  (h_a_pos : a > 0)
  (h_a_neq_one : a ≠ 1)
  (h_defined : ∀ x : ℝ, x^2 - 2*k*x + 2*k + 3 > 0) :
  -1 < k ∧ k < 3 :=
sorry

end NUMINAMATH_CALUDE_function_domain_implies_k_range_l2708_270896


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2708_270847

-- Define the cubic function
def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 + b*x + c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x + b

theorem cubic_function_properties (a b c : ℝ) :
  -- Part 1: If there exists a point where the tangent line is parallel to the x-axis
  (∃ x : ℝ, f' a b x = 0) →
  a^2 ≥ 3*b ∧
  -- Part 2: If f(x) has extreme values at x = -1 and x = 3
  (f' a b (-1) = 0 ∧ f' a b 3 = 0) →
  a = 3 ∧ b = -9 ∧
  -- Part 3: If f(x) < 2c for all x ∈ [-2, 6], then c > 54
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 6 → f a b c x < 2*c) →
  c > 54 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2708_270847


namespace NUMINAMATH_CALUDE_julie_school_work_hours_l2708_270871

-- Define the given parameters
def summer_weeks : ℕ := 10
def summer_hours_per_week : ℕ := 36
def summer_earnings : ℕ := 4500
def school_weeks : ℕ := 45
def school_earnings : ℕ := 4500

-- Define the function to calculate required hours per week
def required_hours_per_week (weeks : ℕ) (total_earnings : ℕ) (hourly_rate : ℚ) : ℚ :=
  (total_earnings : ℚ) / (weeks : ℚ) / hourly_rate

-- Theorem statement
theorem julie_school_work_hours :
  let hourly_rate : ℚ := (summer_earnings : ℚ) / ((summer_weeks * summer_hours_per_week) : ℚ)
  required_hours_per_week school_weeks school_earnings hourly_rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_julie_school_work_hours_l2708_270871


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2708_270867

theorem abs_neg_three_eq_three : abs (-3 : ℤ) = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l2708_270867


namespace NUMINAMATH_CALUDE_pet_store_cages_l2708_270842

/-- The number of bird cages in a pet store --/
def num_cages (num_parrots : Float) (num_parakeets : Float) (avg_birds_per_cage : Float) : Float :=
  (num_parrots + num_parakeets) / avg_birds_per_cage

/-- Theorem: The pet store has approximately 6 bird cages --/
theorem pet_store_cages :
  let num_parrots : Float := 6.0
  let num_parakeets : Float := 2.0
  let avg_birds_per_cage : Float := 1.333333333
  (num_cages num_parrots num_parakeets avg_birds_per_cage).round = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2708_270842


namespace NUMINAMATH_CALUDE_equivalent_angle_proof_l2708_270863

/-- The angle (in degrees) that has the same terminal side as -60° within [0°, 360°) -/
def equivalent_angle : ℝ := 300

theorem equivalent_angle_proof :
  ∃ (k : ℤ), equivalent_angle = k * 360 - 60 ∧ 
  0 ≤ equivalent_angle ∧ equivalent_angle < 360 :=
by sorry

end NUMINAMATH_CALUDE_equivalent_angle_proof_l2708_270863


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l2708_270866

/-- The total amount of oil leaked into the water -/
def total_oil_leaked (pre_repair_leak : ℕ) (during_repair_leak : ℕ) : ℕ :=
  pre_repair_leak + during_repair_leak

/-- Theorem stating the total amount of oil leaked -/
theorem oil_leak_calculation :
  total_oil_leaked 6522 5165 = 11687 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l2708_270866


namespace NUMINAMATH_CALUDE_variance_transformation_l2708_270808

def data_variance (data : List ℝ) : ℝ := sorry

theorem variance_transformation (data : List ℝ) (h : data.length = 2010) 
  (h_var : data_variance data = 2) :
  data_variance (data.map (λ x => -3 * x + 1)) = 18 := by sorry

end NUMINAMATH_CALUDE_variance_transformation_l2708_270808


namespace NUMINAMATH_CALUDE_incorrect_statement_about_immunity_l2708_270868

-- Define the three lines of defense
inductive LineOfDefense
| First
| Second
| Third

-- Define the types of immunity
inductive ImmunityType
| NonSpecific
| Specific

-- Define the components of each line of defense
def componentsOfDefense (line : LineOfDefense) : String :=
  match line with
  | .First => "skin and mucous membranes"
  | .Second => "antimicrobial substances and phagocytic cells in body fluids"
  | .Third => "immune organs and immune cells"

-- Define the type of immunity for each line of defense
def immunityTypeOfDefense (line : LineOfDefense) : ImmunityType :=
  match line with
  | .First => .NonSpecific
  | .Second => .NonSpecific
  | .Third => .Specific

-- Theorem to prove
theorem incorrect_statement_about_immunity :
  ¬(∀ (line : LineOfDefense), immunityTypeOfDefense line = .NonSpecific) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_about_immunity_l2708_270868


namespace NUMINAMATH_CALUDE_min_value_y_l2708_270825

noncomputable def y (x a : ℝ) : ℝ := (Real.exp x - a)^2 + (Real.exp (-x) - a)^2

theorem min_value_y (a : ℝ) (ha : a ≠ 0) :
  (∀ x, y x a ≥ (if a ≥ 2 then a^2 - 2 else 2*(a-1)^2)) ∧
  (∃ x, y x a = (if a ≥ 2 then a^2 - 2 else 2*(a-1)^2)) :=
sorry

end NUMINAMATH_CALUDE_min_value_y_l2708_270825


namespace NUMINAMATH_CALUDE_problem_solution_l2708_270839

theorem problem_solution (x y : ℚ) 
  (eq1 : 5 * x - 3 * y = 8) 
  (eq2 : 2 * x + 7 * y = 1) : 
  3 * (x + y + 4) = 12 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2708_270839


namespace NUMINAMATH_CALUDE_smallest_denominators_sum_l2708_270822

theorem smallest_denominators_sum (p q : ℕ) (h : q > 0) :
  (∃ k : ℕ, Nat.div p q = k ∧ 
   ∃ r : ℕ, r < q ∧ 
   ∃ n : ℕ, Nat.div (r * 1000 + 171) q = n ∧
   ∃ s : ℕ, s < q ∧ Nat.div (s * 1000 + 171) q = n) →
  (∃ q1 q2 : ℕ, q1 < q2 ∧ 
   (∀ q' : ℕ, q' ≠ q1 ∧ q' ≠ q2 → q' > q2) ∧
   q1 + q2 = 99) :=
sorry

end NUMINAMATH_CALUDE_smallest_denominators_sum_l2708_270822


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_fifth_l2708_270855

theorem reciprocal_of_negative_one_fifth : 
  ∀ x : ℚ, x = -1/5 → (∃ y : ℚ, y * x = 1 ∧ y = -5) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_fifth_l2708_270855


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2708_270834

/-- 
Given a hyperbola with equation x^2 + my^2 = 1, where m is a real number,
if its conjugate axis is twice the length of its transverse axis,
then its eccentricity e is equal to √5.
-/
theorem hyperbola_eccentricity (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b = 2*a) →
  (∃ e : ℝ, e = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2708_270834


namespace NUMINAMATH_CALUDE_investment_interest_rate_l2708_270833

theorem investment_interest_rate 
  (principal : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (time : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 900) 
  (h2 : rate1 = 0.04) 
  (h3 : time = 7) 
  (h4 : principal * rate2 * time - principal * rate1 * time = interest_difference) 
  (h5 : interest_difference = 31.50) : 
  rate2 = 0.045 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l2708_270833


namespace NUMINAMATH_CALUDE_books_sold_l2708_270889

theorem books_sold (initial_books : ℕ) (remaining_books : ℕ) (h1 : initial_books = 242) (h2 : remaining_books = 105) :
  initial_books - remaining_books = 137 := by
sorry

end NUMINAMATH_CALUDE_books_sold_l2708_270889


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2708_270819

theorem sum_of_fractions : (1 : ℚ) / 1 + (2 : ℚ) / 2 + (3 : ℚ) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2708_270819


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2708_270849

/-- Given that the total marks in physics, chemistry, and mathematics is 180 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 90. -/
theorem average_marks_chemistry_mathematics (P C M : ℕ) : 
  P + C + M = P + 180 → (C + M) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2708_270849


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_time_to_water_surface_l2708_270875

-- Define the height function
def h (t : ℝ) : ℝ := -4.8 * t^2 + 8 * t + 10

-- Theorem for instantaneous velocity at t = 2
theorem instantaneous_velocity_at_2 : 
  (deriv h) 2 = -11.2 := by sorry

-- Theorem for time when athlete reaches water surface
theorem time_to_water_surface : 
  ∃ t : ℝ, t = 2.5 ∧ h t = 0 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_time_to_water_surface_l2708_270875


namespace NUMINAMATH_CALUDE_production_scale_l2708_270895

/-- Production function that calculates the number of items produced given the number of workers, hours per day, number of days, and production rate. -/
def production (workers : ℕ) (hours_per_day : ℕ) (days : ℕ) (rate : ℚ) : ℚ :=
  (workers : ℚ) * (hours_per_day : ℚ) * (days : ℚ) * rate

/-- Theorem stating that if 8 workers produce 512 items in 8 hours a day for 8 days, 
    then 10 workers working 10 hours a day for 10 days will produce 1000 items, 
    assuming a constant production rate. -/
theorem production_scale (rate : ℚ) : 
  production 8 8 8 rate = 512 → production 10 10 10 rate = 1000 := by
  sorry

#check production_scale

end NUMINAMATH_CALUDE_production_scale_l2708_270895


namespace NUMINAMATH_CALUDE_luke_birthday_stickers_l2708_270862

/-- Represents the number of stickers Luke has at different stages --/
structure StickerCount where
  initial : ℕ
  bought : ℕ
  birthday : ℕ
  given_away : ℕ
  used : ℕ
  final : ℕ

/-- Calculates the number of stickers Luke got for his birthday --/
def birthday_stickers (s : StickerCount) : ℕ :=
  s.final + s.given_away + s.used - s.initial - s.bought

/-- Theorem stating that Luke got 20 stickers for his birthday --/
theorem luke_birthday_stickers :
  ∀ s : StickerCount,
    s.initial = 20 ∧
    s.bought = 12 ∧
    s.given_away = 5 ∧
    s.used = 8 ∧
    s.final = 39 →
    birthday_stickers s = 20 := by
  sorry


end NUMINAMATH_CALUDE_luke_birthday_stickers_l2708_270862


namespace NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l2708_270898

theorem smallest_n_divisible (n : ℕ) : n > 0 ∧ 24 ∣ n^2 ∧ 720 ∣ n^3 → n ≥ 60 :=
by sorry

theorem sixty_satisfies : 24 ∣ 60^2 ∧ 720 ∣ 60^3 :=
by sorry

theorem smallest_n_is_sixty : ∃! n : ℕ, n > 0 ∧ 24 ∣ n^2 ∧ 720 ∣ n^3 ∧ ∀ m : ℕ, (m > 0 ∧ 24 ∣ m^2 ∧ 720 ∣ m^3) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_sixty_satisfies_smallest_n_is_sixty_l2708_270898


namespace NUMINAMATH_CALUDE_equation_solutions_l2708_270827

/-- The equation x^4 * y^4 - 10 * x^2 * y^2 + 9 = 0 -/
def equation (x y : ℕ+) : Prop :=
  (x.val : ℝ)^4 * (y.val : ℝ)^4 - 10 * (x.val : ℝ)^2 * (y.val : ℝ)^2 + 9 = 0

/-- The set of all ordered pairs (x,y) of positive integers satisfying the equation -/
def solution_set : Set (ℕ+ × ℕ+) :=
  {p | equation p.1 p.2}

theorem equation_solutions :
  ∃ (s : Finset (ℕ+ × ℕ+)), s.card = 3 ∧ ↑s = solution_set := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2708_270827


namespace NUMINAMATH_CALUDE_simplify_expression_l2708_270817

theorem simplify_expression (x : ℝ) : 114 * x - 69 * x + 15 = 45 * x + 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2708_270817


namespace NUMINAMATH_CALUDE_rectangle_matchsticks_distribution_l2708_270869

/-- Calculates the total number of matchsticks in the rectangle -/
def total_matchsticks (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Checks if the number of matchsticks can be equally distributed among a given number of children -/
def is_valid_distribution (total_sticks children : ℕ) : Prop :=
  children > 100 ∧ total_sticks % children = 0

theorem rectangle_matchsticks_distribution :
  let total := total_matchsticks 60 10
  ∃ (n : ℕ), is_valid_distribution total n ∧
    ∀ (m : ℕ), m > n → ¬(is_valid_distribution total m) ∧
    n = 127 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_matchsticks_distribution_l2708_270869


namespace NUMINAMATH_CALUDE_dihedral_angle_cosine_value_l2708_270810

/-- Regular triangular pyramid with inscribed sphere -/
structure RegularTriangularPyramid where
  /-- Side length of the base triangle -/
  base_side : ℝ
  /-- Radius of the inscribed sphere -/
  sphere_radius : ℝ
  /-- The sphere radius is one-fourth of the base side length -/
  radius_relation : sphere_radius = base_side / 4

/-- Dihedral angle at the apex of a regular triangular pyramid -/
def dihedral_angle_cosine (pyramid : RegularTriangularPyramid) : ℝ :=
  -- Definition of the dihedral angle cosine
  sorry

/-- Theorem: The cosine of the dihedral angle at the apex is 23/26 -/
theorem dihedral_angle_cosine_value (pyramid : RegularTriangularPyramid) :
  dihedral_angle_cosine pyramid = 23 / 26 :=
by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_cosine_value_l2708_270810


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l2708_270864

theorem theater_ticket_sales (orchestra_price balcony_price premium_price : ℕ)
                             (total_tickets : ℕ) (total_revenue : ℕ)
                             (orchestra balcony premium : ℕ) :
  orchestra_price = 15 →
  balcony_price = 10 →
  premium_price = 25 →
  total_tickets = 550 →
  total_revenue = 9750 →
  orchestra + balcony + premium = total_tickets →
  orchestra_price * orchestra + balcony_price * balcony + premium_price * premium = total_revenue →
  premium = 5 * orchestra →
  orchestra ≥ 50 →
  balcony - orchestra = 179 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l2708_270864


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2708_270883

theorem constant_term_binomial_expansion :
  let f := fun (x : ℝ) => (x - 1 / (2 * Real.sqrt x)) ^ 6
  ∃ (c : ℝ), (∀ (x : ℝ), x > 0 → f x = c + x * (f x - c) / x) ∧ c = 15/16 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2708_270883


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2708_270835

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 5 ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 3*x₀ + 5 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2708_270835


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2708_270832

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x ∈ Set.Ioo 0 3 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ x ∉ Set.Ioo 0 3) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2708_270832


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l2708_270876

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (2 * x - 7) + Real.sqrt (5 - x)) ↔ 3.5 ≤ x ∧ x ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l2708_270876


namespace NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_30_l2708_270814

theorem smallest_of_three_consecutive_sum_30 (x : ℕ) :
  x + (x + 1) + (x + 2) = 30 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_30_l2708_270814


namespace NUMINAMATH_CALUDE_circle_center_correct_l2708_270856

/-- The polar equation of a circle -/
def polar_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

/-- The Cartesian equation of a circle -/
def cartesian_equation (x y : ℝ) : Prop := x^2 + y^2 = 4 * y

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (0, 2)

theorem circle_center_correct :
  (∀ ρ θ : ℝ, polar_equation ρ θ ↔ ∃ x y : ℝ, cartesian_equation x y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (∀ x y : ℝ, cartesian_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 4) :=
sorry

end NUMINAMATH_CALUDE_circle_center_correct_l2708_270856


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_l2708_270877

theorem symmetry_implies_sum (a b : ℝ) :
  (∀ x y : ℝ, y = a * x + 8 ↔ x = -1/2 * y + b) →
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_l2708_270877


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_divisibility_l2708_270886

theorem consecutive_numbers_product_divisibility (n : ℕ) (hn : n > 1) :
  ∃ k : ℕ, ∀ p : ℕ,
    Prime p →
    (∀ i : ℕ, i < n → (p ∣ (k + i + 1))) ↔ p ≤ 2 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_divisibility_l2708_270886


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l2708_270873

theorem negation_of_quadratic_inequality (x : ℝ) :
  ¬(x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l2708_270873


namespace NUMINAMATH_CALUDE_circle_area_isosceles_triangle_l2708_270813

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_isosceles_triangle (a b c : ℝ) (h1 : a = 4) (h2 : b = 4) (h3 : c = 3) :
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = (64 / 13.75) * π :=
sorry

end NUMINAMATH_CALUDE_circle_area_isosceles_triangle_l2708_270813


namespace NUMINAMATH_CALUDE_expansion_property_p_value_l2708_270806

/-- The value of p in the expansion of (x+y)^10 -/
def p : ℚ :=
  8/11

/-- The value of q in the expansion of (x+y)^10 -/
def q : ℚ :=
  3/11

/-- The third term in the expansion of (x+y)^10 -/
def third_term (x y : ℚ) : ℚ :=
  45 * x^8 * y^2

/-- The fourth term in the expansion of (x+y)^10 -/
def fourth_term (x y : ℚ) : ℚ :=
  120 * x^7 * y^3

theorem expansion_property : 
  p + q = 1 ∧ third_term p q = fourth_term p q :=
sorry

theorem p_value : p = 8/11 :=
sorry

end NUMINAMATH_CALUDE_expansion_property_p_value_l2708_270806


namespace NUMINAMATH_CALUDE_literature_tech_cost_difference_l2708_270811

theorem literature_tech_cost_difference :
  let num_books : ℕ := 45
  let lit_cost : ℕ := 7
  let tech_cost : ℕ := 5
  (num_books * lit_cost) - (num_books * tech_cost) = 90 := by
sorry

end NUMINAMATH_CALUDE_literature_tech_cost_difference_l2708_270811


namespace NUMINAMATH_CALUDE_algebraic_inequalities_l2708_270845

theorem algebraic_inequalities :
  (∀ a : ℝ, a^2 + 2 > 2*a) ∧
  (∀ x : ℝ, (x+5)*(x+7) < (x+6)^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_inequalities_l2708_270845


namespace NUMINAMATH_CALUDE_michaels_farm_increase_l2708_270843

/-- Given an initial chicken count, a final chicken count after a certain number of years,
    calculate the annual increase in chickens. -/
def annual_chicken_increase (initial_count final_count : ℕ) (years : ℕ) : ℕ :=
  (final_count - initial_count) / years

/-- Theorem stating that given the specific conditions of Michael's farm,
    the annual increase in chickens is 150. -/
theorem michaels_farm_increase :
  annual_chicken_increase 550 1900 9 = 150 := by
  sorry

end NUMINAMATH_CALUDE_michaels_farm_increase_l2708_270843


namespace NUMINAMATH_CALUDE_carol_wins_probability_l2708_270878

/-- Represents the probability of tossing a six -/
def prob_six : ℚ := 1 / 6

/-- Represents the probability of not tossing a six -/
def prob_not_six : ℚ := 1 - prob_six

/-- Represents the number of players -/
def num_players : ℕ := 4

/-- The probability of Carol winning in one cycle -/
def prob_carol_win_cycle : ℚ := prob_not_six^2 * prob_six * prob_not_six

/-- The probability of no one winning in one cycle -/
def prob_no_win_cycle : ℚ := prob_not_six^num_players

/-- Theorem: The probability of Carol being the first to toss a six 
    in a repeated die-tossing game with four players is 125/671 -/
theorem carol_wins_probability : 
  prob_carol_win_cycle / (1 - prob_no_win_cycle) = 125 / 671 := by
  sorry

end NUMINAMATH_CALUDE_carol_wins_probability_l2708_270878


namespace NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l2708_270815

theorem sum_of_special_primes_is_prime (C D : ℕ+) : 
  Prime C.val → Prime D.val → Prime (C.val - D.val) → Prime (C.val + D.val) →
  Prime (C.val + D.val + (C.val - D.val) + C.val + D.val) := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_primes_is_prime_l2708_270815


namespace NUMINAMATH_CALUDE_right_triangle_ratio_minimum_l2708_270831

theorem right_triangle_ratio_minimum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  c / (a + b) ≥ Real.sqrt 2 / 2 ∧ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + y^2 = z^2 ∧ z / (x + y) = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_minimum_l2708_270831


namespace NUMINAMATH_CALUDE_population_falls_below_threshold_l2708_270848

/-- The annual decrease rate of the finch population -/
def annual_decrease_rate : ℝ := 0.3

/-- The threshold below which we consider the population to have significantly decreased -/
def threshold : ℝ := 0.15

/-- The function that calculates the population after a given number of years -/
def population_after_years (initial_population : ℝ) (years : ℕ) : ℝ :=
  initial_population * (1 - annual_decrease_rate) ^ years

/-- Theorem stating that it takes 6 years for the population to fall below the threshold -/
theorem population_falls_below_threshold (initial_population : ℝ) (h : initial_population > 0) :
  population_after_years initial_population 6 < threshold * initial_population ∧
  population_after_years initial_population 5 ≥ threshold * initial_population :=
by sorry

end NUMINAMATH_CALUDE_population_falls_below_threshold_l2708_270848


namespace NUMINAMATH_CALUDE_range_of_a_l2708_270830

theorem range_of_a : 
  (∀ x, 0 < x ∧ x < 1 → ∀ a, (x - a) * (x - (a + 2)) ≤ 0) ∧ 
  (∃ x a, ¬(0 < x ∧ x < 1) ∧ (x - a) * (x - (a + 2)) ≤ 0) →
  ∀ a, a ∈ Set.Icc (-1 : ℝ) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2708_270830


namespace NUMINAMATH_CALUDE_jellybeans_remaining_l2708_270894

/-- Given a jar of jellybeans and a class of students, calculate the remaining jellybeans after some students eat them. -/
theorem jellybeans_remaining (total_jellybeans : ℕ) (total_students : ℕ) (absent_students : ℕ) (jellybeans_per_student : ℕ)
  (h1 : total_jellybeans = 100)
  (h2 : total_students = 24)
  (h3 : absent_students = 2)
  (h4 : jellybeans_per_student = 3) :
  total_jellybeans - (total_students - absent_students) * jellybeans_per_student = 34 := by
  sorry

end NUMINAMATH_CALUDE_jellybeans_remaining_l2708_270894


namespace NUMINAMATH_CALUDE_sqrt_two_function_value_l2708_270846

/-- Given a function f where f(x-1) = x^2 - 2x for all real x, prove that f(√2) = 1 -/
theorem sqrt_two_function_value (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x - 1) = x^2 - 2*x) : 
  f (Real.sqrt 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_function_value_l2708_270846


namespace NUMINAMATH_CALUDE_innermost_rectangle_length_l2708_270838

def floor_mat (y : ℝ) : Prop :=
  let inner_area := 2 * y
  let middle_area := (y + 4) * 6
  let outer_area := (y + 8) * 10
  (middle_area - inner_area = outer_area - middle_area) ∧
  (inner_area > 0) ∧ (middle_area > inner_area) ∧ (outer_area > middle_area)

theorem innermost_rectangle_length :
  ∃ y : ℝ, floor_mat y ∧ y = 4 := by sorry

end NUMINAMATH_CALUDE_innermost_rectangle_length_l2708_270838


namespace NUMINAMATH_CALUDE_range_of_f_l2708_270818

-- Define the function f
def f (x : ℝ) : ℝ := |1 - x| - |x - 3|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y ∈ Set.range f, -2 ≤ y ∧ y ≤ 2 ∧
  ∃ x₁ x₂ : ℝ, f x₁ = -2 ∧ f x₂ = 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2708_270818


namespace NUMINAMATH_CALUDE_probability_of_favorable_outcome_l2708_270803

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 ∧ a ≠ b

def is_favorable_pair (a b : ℕ) : Prop :=
  is_valid_pair a b ∧ ∃ k : ℕ, a * b + a + b = 7 * k - 1

def total_pairs : ℕ := Nat.choose 60 2

def favorable_pairs : ℕ := 444

theorem probability_of_favorable_outcome :
  (favorable_pairs : ℚ) / total_pairs = 74 / 295 := by sorry

end NUMINAMATH_CALUDE_probability_of_favorable_outcome_l2708_270803


namespace NUMINAMATH_CALUDE_circle_center_sum_l2708_270872

/-- Given a circle defined by the equation x^2 + y^2 + 6x - 4y - 12 = 0,
    if (a, b) is the center of this circle, then a + b = -1. -/
theorem circle_center_sum (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 6*x - 4*y - 12 = 0 ↔ (x - a)^2 + (y - b)^2 = (a^2 + b^2 + 6*a - 4*b - 12)) →
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2708_270872


namespace NUMINAMATH_CALUDE_hundredth_odd_integer_and_divisibility_l2708_270859

theorem hundredth_odd_integer_and_divisibility :
  (∃ n : ℕ, n = 100 ∧ 2 * n - 1 = 199) ∧ ¬(199 % 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_hundredth_odd_integer_and_divisibility_l2708_270859


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l2708_270888

theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  Odd a ∧ Odd b ∧ Odd c ∧  -- a, b, c are odd
  b = a + 2 ∧ c = b + 2 ∧   -- consecutive with difference 2
  a + b + c = 75            -- sum is 75
  → c = 27 := by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l2708_270888


namespace NUMINAMATH_CALUDE_vicente_spent_25_dollars_l2708_270882

-- Define the quantities and prices
def rice_kg : ℕ := 5
def meat_lb : ℕ := 3
def rice_price_per_kg : ℕ := 2
def meat_price_per_lb : ℕ := 5

-- Define the total cost function
def total_cost (rice_kg meat_lb rice_price_per_kg meat_price_per_lb : ℕ) : ℕ :=
  rice_kg * rice_price_per_kg + meat_lb * meat_price_per_lb

-- Theorem statement
theorem vicente_spent_25_dollars :
  total_cost rice_kg meat_lb rice_price_per_kg meat_price_per_lb = 25 := by
  sorry

end NUMINAMATH_CALUDE_vicente_spent_25_dollars_l2708_270882


namespace NUMINAMATH_CALUDE_power_equation_solution_l2708_270821

theorem power_equation_solution : 5^3 - 7 = 6^2 + 82 := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2708_270821


namespace NUMINAMATH_CALUDE_museum_visit_permutations_l2708_270836

theorem museum_visit_permutations :
  Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_museum_visit_permutations_l2708_270836


namespace NUMINAMATH_CALUDE_product_of_reals_l2708_270885

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 8) (cube_sum_eq : a^3 + b^3 = 152) : a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reals_l2708_270885


namespace NUMINAMATH_CALUDE_no_multiple_of_five_2c4_l2708_270881

theorem no_multiple_of_five_2c4 :
  ∀ C : ℕ, C < 10 → ¬(∃ k : ℕ, 200 + 10 * C + 4 = 5 * k) :=
by
  sorry

end NUMINAMATH_CALUDE_no_multiple_of_five_2c4_l2708_270881
