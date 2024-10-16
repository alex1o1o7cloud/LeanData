import Mathlib

namespace NUMINAMATH_CALUDE_adams_money_l3077_307732

/-- Adam's money problem --/
theorem adams_money (initial_amount spent allowance : ℕ) :
  initial_amount = 5 →
  spent = 2 →
  allowance = 5 →
  initial_amount - spent + allowance = 8 := by
  sorry

end NUMINAMATH_CALUDE_adams_money_l3077_307732


namespace NUMINAMATH_CALUDE_trig_expression_range_l3077_307740

theorem trig_expression_range (C : ℝ) (h : 0 < C ∧ C < π) :
  ∃ (lower upper : ℝ), lower = -1 ∧ upper = Real.sqrt 2 ∧
  -1 < (2 * Real.cos (2 * C) / Real.tan C) + 1 ∧
  (2 * Real.cos (2 * C) / Real.tan C) + 1 ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_range_l3077_307740


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l3077_307797

theorem solution_set_reciprocal_inequality (x : ℝ) : 
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l3077_307797


namespace NUMINAMATH_CALUDE_car_departure_time_l3077_307760

/-- 
Given two cars A and B that start simultaneously from locations A and B respectively:
- They will meet at some point
- If Car A departs earlier, they will meet 30 minutes earlier
- Car A travels at 60 kilometers per hour
- Car B travels at 40 kilometers per hour

Prove that Car A needs to depart 50 minutes earlier for them to meet 30 minutes earlier.
-/
theorem car_departure_time (speed_A speed_B : ℝ) (meeting_time_diff : ℝ) :
  speed_A = 60 →
  speed_B = 40 →
  meeting_time_diff = 30 →
  ∃ (departure_time : ℝ), 
    departure_time = 50 ∧
    speed_A * (departure_time / 60) = speed_A * (meeting_time_diff / 60) + speed_B * (meeting_time_diff / 60) :=
by sorry

end NUMINAMATH_CALUDE_car_departure_time_l3077_307760


namespace NUMINAMATH_CALUDE_animal_shelter_cats_l3077_307729

theorem animal_shelter_cats (total : ℕ) (cats dogs : ℕ) : 
  total = 60 →
  cats = dogs + 20 →
  cats + dogs = total →
  cats = 40 := by
sorry

end NUMINAMATH_CALUDE_animal_shelter_cats_l3077_307729


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l3077_307734

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 100) : 
  x^2 + y^2 = 1400 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l3077_307734


namespace NUMINAMATH_CALUDE_vector_sum_closed_polygon_l3077_307700

variable {V : Type*} [AddCommGroup V]

/-- Given vectors AB, CF, BC, and FA in a vector space V, 
    their sum is equal to the zero vector. -/
theorem vector_sum_closed_polygon (AB CF BC FA : V) :
  AB + CF + BC + FA = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_closed_polygon_l3077_307700


namespace NUMINAMATH_CALUDE_figure_segments_length_l3077_307774

theorem figure_segments_length 
  (rectangle_length : ℝ) 
  (rectangle_breadth : ℝ) 
  (square_side : ℝ) 
  (h1 : rectangle_length = 10) 
  (h2 : rectangle_breadth = 6) 
  (h3 : square_side = 4) :
  square_side + 2 * rectangle_length + rectangle_breadth / 2 = 27 :=
by sorry

end NUMINAMATH_CALUDE_figure_segments_length_l3077_307774


namespace NUMINAMATH_CALUDE_seth_yogurt_purchase_l3077_307795

theorem seth_yogurt_purchase (ice_cream_cartons : ℕ) (ice_cream_cost : ℕ) (yogurt_cost : ℕ) (difference : ℕ) :
  ice_cream_cartons = 20 →
  ice_cream_cost = 6 →
  yogurt_cost = 1 →
  ice_cream_cartons * ice_cream_cost = difference + yogurt_cost * (ice_cream_cartons * ice_cream_cost - difference) / yogurt_cost →
  (ice_cream_cartons * ice_cream_cost - difference) / yogurt_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_seth_yogurt_purchase_l3077_307795


namespace NUMINAMATH_CALUDE_soccer_team_points_l3077_307772

theorem soccer_team_points : ∀ (total_games wins losses draws : ℕ)
  (points_per_win points_per_draw points_per_loss : ℕ),
  total_games = 20 →
  wins = 14 →
  losses = 2 →
  draws = total_games - wins - losses →
  points_per_win = 3 →
  points_per_draw = 1 →
  points_per_loss = 0 →
  wins * points_per_win + draws * points_per_draw + losses * points_per_loss = 46 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_points_l3077_307772


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3077_307702

/-- Represents a hyperbola with equation x^2/a^2 - y^2/b^2 = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive_a : a > 0
  h_positive_b : b > 0

/-- Theorem: Given a hyperbola with an asymptote through (2, √3) and a focus at (-√7, 0),
    prove that a = 2 and b = √3 --/
theorem hyperbola_equation (h : Hyperbola)
  (h_asymptote : 2 * h.b = Real.sqrt 3 * h.a)
  (h_focus : h.a ^ 2 - h.b ^ 2 = 7) :
  h.a = 2 ∧ h.b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3077_307702


namespace NUMINAMATH_CALUDE_convex_ngon_coverage_l3077_307752

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool
  area : Real

/-- Represents a triangle in 2D space -/
structure Triangle where
  vertices : List (Real × Real)
  area : Real

/-- Checks if a polygon is covered by a triangle -/
def is_covered (p : ConvexPolygon) (t : Triangle) : Prop :=
  sorry

/-- Main theorem: A convex n-gon with area 1 (n ≥ 6) can be covered by a triangle with area ≤ 2 -/
theorem convex_ngon_coverage (p : ConvexPolygon) :
  p.is_convex ∧ p.area = 1 ∧ p.vertices.length ≥ 6 →
  ∃ t : Triangle, t.area ≤ 2 ∧ is_covered p t :=
sorry

end NUMINAMATH_CALUDE_convex_ngon_coverage_l3077_307752


namespace NUMINAMATH_CALUDE_product_of_numbers_l3077_307745

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3077_307745


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l3077_307799

theorem rational_solutions_quadratic (k : ℕ+) :
  (∃ x : ℚ, k * x^2 + 12 * x + k = 0) ↔ (k = 3 ∨ k = 6) :=
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l3077_307799


namespace NUMINAMATH_CALUDE_olympiad_numbers_equal_divisors_of_1998_l3077_307721

/-- The year of the first Olympiad -/
def firstOlympiadYear : ℕ := 1999

/-- The year of the n-th Olympiad -/
def olympiadYear (n : ℕ) : ℕ := firstOlympiadYear + n - 1

/-- The set of positive integers n such that n divides the year of the n-th Olympiad -/
def validOlympiadNumbers : Set ℕ :=
  {n : ℕ | n > 0 ∧ n ∣ olympiadYear n}

/-- The set of divisors of 1998 -/
def divisorsOf1998 : Set ℕ :=
  {n : ℕ | n > 0 ∧ n ∣ 1998}

theorem olympiad_numbers_equal_divisors_of_1998 :
  validOlympiadNumbers = divisorsOf1998 :=
by sorry

end NUMINAMATH_CALUDE_olympiad_numbers_equal_divisors_of_1998_l3077_307721


namespace NUMINAMATH_CALUDE_prime_quadruples_sum_882_l3077_307727

theorem prime_quadruples_sum_882 :
  ∀ p₁ p₂ p₃ p₄ : ℕ,
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ →
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ →
    p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882 →
    ((p₁ = 2 ∧ p₂ = 5 ∧ p₃ = 19 ∧ p₄ = 37) ∨
     (p₁ = 2 ∧ p₂ = 11 ∧ p₃ = 19 ∧ p₄ = 31) ∨
     (p₁ = 2 ∧ p₂ = 13 ∧ p₃ = 19 ∧ p₄ = 29)) :=
by sorry

end NUMINAMATH_CALUDE_prime_quadruples_sum_882_l3077_307727


namespace NUMINAMATH_CALUDE_triathlon_average_speed_l3077_307754

/-- Calculates the average speed for a triathlete swimming and running equal distances -/
theorem triathlon_average_speed (swim_speed run_speed : ℝ) (h1 : swim_speed = 1) (h2 : run_speed = 7) :
  let total_time := 1 / swim_speed + 1 / run_speed
  let total_distance := 2
  total_distance / total_time = 1.75 := by sorry

end NUMINAMATH_CALUDE_triathlon_average_speed_l3077_307754


namespace NUMINAMATH_CALUDE_correct_addition_l3077_307725

theorem correct_addition (x : ℤ) (h : x + 21 = 52) : x + 40 = 71 := by
  sorry

end NUMINAMATH_CALUDE_correct_addition_l3077_307725


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3077_307733

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 0) : 
  (a ≥ 2 → a ≥ 1) ∧ ¬(a ≥ 1 → a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3077_307733


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_2n_plus_8_l3077_307711

theorem remainder_of_3_pow_2n_plus_8 (n : ℕ) : (3^(2*n) + 8) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_2n_plus_8_l3077_307711


namespace NUMINAMATH_CALUDE_quadrilateral_area_inequalities_l3077_307779

/-- Properties of a quadrilateral -/
structure Quadrilateral where
  S : ℝ  -- Area
  a : ℝ  -- Side length
  b : ℝ  -- Side length
  c : ℝ  -- Side length
  d : ℝ  -- Side length
  e : ℝ  -- Diagonal length
  f : ℝ  -- Diagonal length
  m : ℝ  -- Midpoint segment length
  n : ℝ  -- Midpoint segment length
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hd : 0 < d
  he : 0 < e
  hf : 0 < f
  hm : 0 < m
  hn : 0 < n
  hS : 0 < S

/-- Theorem: Area inequalities for a quadrilateral -/
theorem quadrilateral_area_inequalities (q : Quadrilateral) :
  q.S ≤ (1/4) * (q.e^2 + q.f^2) ∧
  q.S ≤ (1/2) * (q.m^2 + q.n^2) ∧
  q.S ≤ (1/4) * (q.a + q.c) * (q.b + q.d) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_inequalities_l3077_307779


namespace NUMINAMATH_CALUDE_q_range_l3077_307749

-- Define the function q(x)
def q (x : ℝ) : ℝ := x^4 + 4*x^2 + 4

-- State the theorem
theorem q_range :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ q x = y) ↔ y ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_q_range_l3077_307749


namespace NUMINAMATH_CALUDE_mean_temperature_l3077_307757

def temperatures : List ℤ := [-3, 0, 2, -1, 4, 5, 3]

theorem mean_temperature : 
  (List.sum temperatures) / (List.length temperatures) = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l3077_307757


namespace NUMINAMATH_CALUDE_correct_completion_for_two_viewers_l3077_307783

/-- Represents the options for completing the sentence --/
inductive SentenceCompletion
  | NoneOfThem
  | BothOfThem
  | NoneOfWhom
  | NeitherOfWhom

/-- Represents a person who looked at the house --/
structure HouseViewer where
  wantsToBuy : Bool

/-- The correct sentence completion given two house viewers --/
def correctCompletion (viewer1 viewer2 : HouseViewer) : SentenceCompletion :=
  if !viewer1.wantsToBuy ∧ !viewer2.wantsToBuy then
    SentenceCompletion.NeitherOfWhom
  else
    SentenceCompletion.BothOfThem  -- This else case is not actually used in our theorem

theorem correct_completion_for_two_viewers (viewer1 viewer2 : HouseViewer) 
  (h1 : ¬viewer1.wantsToBuy) (h2 : ¬viewer2.wantsToBuy) :
  correctCompletion viewer1 viewer2 = SentenceCompletion.NeitherOfWhom :=
by sorry

end NUMINAMATH_CALUDE_correct_completion_for_two_viewers_l3077_307783


namespace NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l3077_307716

theorem complex_square_root_of_negative_four (z : ℂ) 
  (h1 : z^2 = -4)
  (h2 : z.im > 0) : 
  z = Complex.I * 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l3077_307716


namespace NUMINAMATH_CALUDE_debate_only_count_l3077_307713

/-- Represents the number of pupils in a class with debate and singing activities -/
structure ClassActivities where
  total : ℕ
  singing_only : ℕ
  both : ℕ
  debate_only : ℕ

/-- The number of pupils in debate only is 37 -/
theorem debate_only_count (c : ClassActivities) 
  (h1 : c.total = 55)
  (h2 : c.singing_only = 18)
  (h3 : c.both = 17)
  (h4 : c.total = c.debate_only + c.singing_only + c.both) : 
  c.debate_only = 37 := by
  sorry

end NUMINAMATH_CALUDE_debate_only_count_l3077_307713


namespace NUMINAMATH_CALUDE_specific_prism_volume_l3077_307746

/-- Regular triangular prism inscribed in a sphere -/
structure InscribedPrism where
  -- Radius of the sphere
  R : ℝ
  -- Length of AD
  AD : ℝ
  -- Assertion that CD is a diameter
  is_diameter : Bool

/-- Volume of the inscribed prism -/
def prism_volume (p : InscribedPrism) : ℝ :=
  sorry

/-- Theorem: The volume of the specific inscribed prism is 48√15 -/
theorem specific_prism_volume :
  let p : InscribedPrism := {
    R := 6,
    AD := 4 * Real.sqrt 6,
    is_diameter := true
  }
  prism_volume p = 48 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_prism_volume_l3077_307746


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l3077_307796

theorem max_value_of_sum_of_squares (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (x_ge : x ≥ -1/2)
  (y_ge : y ≥ -3/2)
  (z_ge : z ≥ -1) :
  Real.sqrt (3 * x + 1.5) + Real.sqrt (3 * y + 4.5) + Real.sqrt (3 * z + 3) ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l3077_307796


namespace NUMINAMATH_CALUDE_tanning_salon_revenue_l3077_307717

/-- Calculate the revenue of a tanning salon for a calendar month -/
theorem tanning_salon_revenue 
  (first_visit_cost : ℕ) 
  (subsequent_visit_cost : ℕ) 
  (total_customers : ℕ) 
  (second_visit_customers : ℕ) 
  (third_visit_customers : ℕ)
  (h1 : first_visit_cost = 10)
  (h2 : subsequent_visit_cost = 8)
  (h3 : total_customers = 100)
  (h4 : second_visit_customers = 30)
  (h5 : third_visit_customers = 10)
  (h6 : second_visit_customers ≤ total_customers)
  (h7 : third_visit_customers ≤ second_visit_customers) :
  first_visit_cost * total_customers + 
  subsequent_visit_cost * second_visit_customers + 
  subsequent_visit_cost * third_visit_customers = 1320 :=
by sorry

end NUMINAMATH_CALUDE_tanning_salon_revenue_l3077_307717


namespace NUMINAMATH_CALUDE_alternate_multiple_iff_not_div_20_l3077_307723

/-- A positive integer is alternate if its decimal digits are alternately odd and even. -/
def is_alternate (n : ℕ) : Prop := sorry

/-- A number n has an alternate multiple if there exists a positive integer k such that k * n is alternate. -/
def has_alternate_multiple (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ is_alternate (k * n)

/-- Main theorem: A positive integer n has an alternate multiple if and only if n is not divisible by 20. -/
theorem alternate_multiple_iff_not_div_20 (n : ℕ) (hn : n > 0) :
  has_alternate_multiple n ↔ ¬(20 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_alternate_multiple_iff_not_div_20_l3077_307723


namespace NUMINAMATH_CALUDE_xiaoming_relative_score_l3077_307704

def class_average : ℝ := 90
def xiaoming_score : ℝ := 85

theorem xiaoming_relative_score :
  xiaoming_score - class_average = -5 := by
sorry

end NUMINAMATH_CALUDE_xiaoming_relative_score_l3077_307704


namespace NUMINAMATH_CALUDE_bad_carrots_count_l3077_307781

theorem bad_carrots_count (olivia_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  olivia_carrots = 20 → 
  mom_carrots = 14 → 
  good_carrots = 19 → 
  olivia_carrots + mom_carrots - good_carrots = 15 := by
sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l3077_307781


namespace NUMINAMATH_CALUDE_equation_three_solutions_l3077_307792

/-- The equation has exactly three solutions when a is 0, 5, or 9 -/
theorem equation_three_solutions (x : ℝ) (a : ℝ) : 
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, 
    (Real.sqrt (x - 1) * (|x^2 - 10*x + 16| - a)) / 
    (a*x^2 - 7*x^2 - 10*a*x + 70*x + 21*a - 147) = 0) ↔ 
  (a = 0 ∨ a = 5 ∨ a = 9) :=
sorry

end NUMINAMATH_CALUDE_equation_three_solutions_l3077_307792


namespace NUMINAMATH_CALUDE_power_of_power_three_l3077_307707

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l3077_307707


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_one_range_of_m_for_nonempty_solution_l3077_307776

-- Define the function f
def f (x m : ℝ) : ℝ := |2*x - 2| + |x + m|

-- Theorem for part (1)
theorem solution_set_when_m_is_one :
  ∃ (a b : ℝ), a = 0 ∧ b = 4/3 ∧
  (∀ x, f x 1 ≤ 3 ↔ a ≤ x ∧ x ≤ b) :=
sorry

-- Theorem for part (2)
theorem range_of_m_for_nonempty_solution :
  ∃ (lower upper : ℝ), lower = -4 ∧ upper = 2 ∧
  (∀ m, (∃ x, f x m ≤ 3) ↔ lower ≤ m ∧ m ≤ upper) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_one_range_of_m_for_nonempty_solution_l3077_307776


namespace NUMINAMATH_CALUDE_final_position_l3077_307706

def move_on_number_line (start : ℤ) (right : ℤ) (left : ℤ) : ℤ :=
  start + right - left

theorem final_position :
  move_on_number_line (-2) 3 5 = -4 := by
  sorry

end NUMINAMATH_CALUDE_final_position_l3077_307706


namespace NUMINAMATH_CALUDE_max_value_of_a_plus_inverse_l3077_307737

theorem max_value_of_a_plus_inverse (a : ℝ) (h : a < 0) : 
  ∃ (M : ℝ), M = -2 ∧ ∀ (x : ℝ), x < 0 → x + 1/x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_plus_inverse_l3077_307737


namespace NUMINAMATH_CALUDE_almonds_vs_white_sugar_difference_l3077_307736

-- Define the amounts of ingredients used
def brown_sugar : ℝ := 1.28
def white_sugar : ℝ := 0.75
def ground_almonds : ℝ := 1.56
def cocoa_powder : ℝ := 0.49

-- Theorem statement
theorem almonds_vs_white_sugar_difference :
  ground_almonds - white_sugar = 0.81 := by
  sorry

end NUMINAMATH_CALUDE_almonds_vs_white_sugar_difference_l3077_307736


namespace NUMINAMATH_CALUDE_total_pets_is_54_l3077_307722

/-- The number of pets owned by Teddy, Ben, and Dave -/
def total_pets : ℕ :=
  let teddy_dogs : ℕ := 7
  let teddy_cats : ℕ := 8
  let ben_extra_dogs : ℕ := 9
  let dave_extra_cats : ℕ := 13
  let dave_fewer_dogs : ℕ := 5

  let teddy_pets : ℕ := teddy_dogs + teddy_cats
  let ben_pets : ℕ := (teddy_dogs + ben_extra_dogs)
  let dave_pets : ℕ := (teddy_cats + dave_extra_cats) + (teddy_dogs - dave_fewer_dogs)

  teddy_pets + ben_pets + dave_pets

theorem total_pets_is_54 : total_pets = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_is_54_l3077_307722


namespace NUMINAMATH_CALUDE_marble_arrangement_l3077_307720

/-- Represents the color of a marble -/
inductive Color
  | Green
  | Blue
  | Red

/-- Represents an arrangement of marbles -/
def Arrangement := List Color

/-- Checks if an arrangement satisfies the equal neighbor condition -/
def satisfiesCondition (arr : Arrangement) : Bool :=
  sorry

/-- Counts the number of valid arrangements for a given number of marbles -/
def countArrangements (totalMarbles : Nat) : Nat :=
  sorry

theorem marble_arrangement :
  let greenMarbles : Nat := 6
  let m : Nat := 12  -- maximum number of additional blue and red marbles
  let totalMarbles : Nat := greenMarbles + m
  let N : Nat := countArrangements totalMarbles
  N = 924 ∧ N % 1000 = 924 := by
  sorry

end NUMINAMATH_CALUDE_marble_arrangement_l3077_307720


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_l3077_307778

-- Define the function f
def f (a x : ℝ) : ℝ := |a - 3*x| - |2 + x|

-- Theorem for part (1)
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x ≤ 3} = {x : ℝ | -3/4 ≤ x ∧ x ≤ 7/2} := by sorry

-- Theorem for part (2)
theorem range_of_a :
  {a : ℝ | ∃ x, f a x ≥ 1 ∧ ∃ y, a + 2*|2 + y| = 0} = {a : ℝ | a ≥ -5/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_l3077_307778


namespace NUMINAMATH_CALUDE_option_b_neither_parallel_nor_perpendicular_l3077_307771

/-- Two vectors in R³ -/
structure VectorPair where
  μ : Fin 3 → ℝ
  v : Fin 3 → ℝ

/-- Check if two vectors are parallel -/
def isParallel (pair : VectorPair) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, pair.μ i = k * pair.v i)

/-- Check if two vectors are perpendicular -/
def isPerpendicular (pair : VectorPair) : Prop :=
  (pair.μ 0 * pair.v 0 + pair.μ 1 * pair.v 1 + pair.μ 2 * pair.v 2) = 0

/-- The specific vector pair for option B -/
def optionB : VectorPair where
  μ := ![3, 0, -1]
  v := ![0, 0, 2]

/-- Theorem stating that the vectors in option B are neither parallel nor perpendicular -/
theorem option_b_neither_parallel_nor_perpendicular :
  ¬(isParallel optionB) ∧ ¬(isPerpendicular optionB) := by
  sorry


end NUMINAMATH_CALUDE_option_b_neither_parallel_nor_perpendicular_l3077_307771


namespace NUMINAMATH_CALUDE_system_solutions_l3077_307718

theorem system_solutions (x y z : ℤ) : 
  x^2 - 9*y^2 - z^2 = 0 ∧ z = x - 3*y →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 3 ∧ y = 1 ∧ z = 0) ∨
  (x = 9 ∧ y = 3 ∧ z = 0) := by
sorry

end NUMINAMATH_CALUDE_system_solutions_l3077_307718


namespace NUMINAMATH_CALUDE_sum_of_fractions_zero_l3077_307784

theorem sum_of_fractions_zero (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 1) :
  a / (b - c) + b / (c - a) + c / (a - b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_zero_l3077_307784


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l3077_307719

theorem system_of_equations_solutions :
  -- First system
  (∃ x y : ℝ, 4 * x - y = 1 ∧ y = 2 * x + 3 ∧ x = 2 ∧ y = 7) ∧
  -- Second system
  (∃ x y : ℝ, 2 * x - y = 5 ∧ 7 * x - 3 * y = 20 ∧ x = 5 ∧ y = 5) :=
by
  sorry

#check system_of_equations_solutions

end NUMINAMATH_CALUDE_system_of_equations_solutions_l3077_307719


namespace NUMINAMATH_CALUDE_two_solutions_with_more_sheep_l3077_307775

def budget : ℕ := 800
def goat_cost : ℕ := 15
def sheep_cost : ℕ := 16

def is_valid_solution (g h : ℕ) : Prop :=
  goat_cost * g + sheep_cost * h = budget ∧ h > g

theorem two_solutions_with_more_sheep :
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (g h : ℕ), (g, h) ∈ s ↔ is_valid_solution g h) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_two_solutions_with_more_sheep_l3077_307775


namespace NUMINAMATH_CALUDE_equation_proof_l3077_307756

theorem equation_proof : (5568 / 87)^(1/3) + (72 * 2)^(1/2) = (256)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3077_307756


namespace NUMINAMATH_CALUDE_fraction_difference_equals_square_difference_l3077_307751

theorem fraction_difference_equals_square_difference 
  (x y z v : ℚ) (h : x / y + z / v = 1) : 
  x / y - z / v = (x / y)^2 - (z / v)^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_square_difference_l3077_307751


namespace NUMINAMATH_CALUDE_unique_a_value_l3077_307764

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 4^x else 2^(a - x)

-- State the theorem
theorem unique_a_value (a : ℝ) (h1 : a ≠ 1) :
  f a (1 - a) = f a (a - 1) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l3077_307764


namespace NUMINAMATH_CALUDE_percentage_problem_l3077_307777

theorem percentage_problem :
  let total := 500
  let unknown_percentage := 50
  let given_percentage := 10
  let result := 25
  (given_percentage / 100) * (unknown_percentage / 100) * total = result :=
by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3077_307777


namespace NUMINAMATH_CALUDE_chinese_chess_sets_l3077_307793

theorem chinese_chess_sets (go_cost : ℕ) (chinese_chess_cost : ℕ) (total_sets : ℕ) (total_cost : ℕ) :
  go_cost = 24 →
  chinese_chess_cost = 18 →
  total_sets = 14 →
  total_cost = 300 →
  ∃ (go_sets chinese_chess_sets : ℕ),
    go_sets + chinese_chess_sets = total_sets ∧
    go_cost * go_sets + chinese_chess_cost * chinese_chess_sets = total_cost ∧
    chinese_chess_sets = 6 := by
  sorry

end NUMINAMATH_CALUDE_chinese_chess_sets_l3077_307793


namespace NUMINAMATH_CALUDE_ball_probability_l3077_307767

theorem ball_probability (m : ℕ) : 
  (8 : ℝ) / (8 + m) > (m : ℝ) / (8 + m) → m < 8 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l3077_307767


namespace NUMINAMATH_CALUDE_max_profit_theorem_l3077_307762

/-- Represents the production plan and profit calculation for a company producing two types of crafts. -/
structure ProductionPlan where
  /-- Cost of material A in yuan -/
  cost_A : ℕ
  /-- Cost of material B in yuan -/
  cost_B : ℕ
  /-- Number of units of craft X produced -/
  units_X : ℕ
  /-- Number of units of craft Y produced -/
  units_Y : ℕ
  /-- Condition: Cost of B is 40 yuan more than A -/
  cost_diff : cost_B = cost_A + 40
  /-- Condition: 2 units of A and 3 units of B cost 420 yuan -/
  total_cost : 2 * cost_A + 3 * cost_B = 420
  /-- Condition: Total number of crafts is 560 -/
  total_units : units_X + units_Y = 560
  /-- Condition: X should not exceed 180 units -/
  max_X : units_X ≤ 180

/-- Calculates the profit for a given production plan -/
def profit (plan : ProductionPlan) : ℕ :=
  360 * plan.units_X + 450 * plan.units_Y -
  (plan.cost_A * (plan.units_X + 3 * plan.units_Y) +
   plan.cost_B * (2 * plan.units_X + 2 * plan.units_Y))

/-- Theorem stating the maximum profit and optimal production plan -/
theorem max_profit_theorem (plan : ProductionPlan) : 
  plan.cost_A = 60 ∧ plan.cost_B = 100 ∧ plan.units_X = 180 ∧ plan.units_Y = 380 →
  profit plan = 44600 ∧ ∀ other_plan : ProductionPlan, profit other_plan ≤ profit plan := by
  sorry


end NUMINAMATH_CALUDE_max_profit_theorem_l3077_307762


namespace NUMINAMATH_CALUDE_amoeba_growth_after_week_l3077_307759

def amoeba_population (initial_population : ℕ) (days : ℕ) : ℕ :=
  if days = 0 then
    initial_population
  else if days % 2 = 1 then
    2 * amoeba_population initial_population (days - 1)
  else
    3 * 2 * amoeba_population initial_population (days - 1)

theorem amoeba_growth_after_week :
  amoeba_population 4 7 = 13824 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_growth_after_week_l3077_307759


namespace NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_plus_2x_l3077_307730

open MeasureTheory Interval Real

theorem integral_sqrt_4_minus_x_squared_plus_2x : 
  ∫ x in (-2)..2, (Real.sqrt (4 - x^2) + 2*x) = 2*π := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_plus_2x_l3077_307730


namespace NUMINAMATH_CALUDE_expression_simplification_l3077_307766

theorem expression_simplification (x y : ℝ) 
  (h : (x - 2)^2 + |1 + y| = 0) : 
  ((x - y) * (x + 2*y) - (x + y)^2) / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3077_307766


namespace NUMINAMATH_CALUDE_circle_radius_is_sqrt_2_l3077_307790

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def intersects_x_axis_at (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  (p1.1 - c.center.1)^2 + (p1.2 - c.center.2)^2 = c.radius^2 ∧
  (p2.1 - c.center.1)^2 + (p2.2 - c.center.2)^2 = c.radius^2 ∧
  p1.2 = 0 ∧ p2.2 = 0

def tangent_to_line (c : Circle) : Prop :=
  ∃ (x y : ℝ), (x - y + 1 = 0) ∧
  ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
  (|x - y + 1| / Real.sqrt 2 = c.radius)

-- State the theorem
theorem circle_radius_is_sqrt_2 (c : Circle) :
  is_in_first_quadrant c.center →
  intersects_x_axis_at c (1, 0) (3, 0) →
  tangent_to_line c →
  c.radius = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_sqrt_2_l3077_307790


namespace NUMINAMATH_CALUDE_diamond_and_hearts_balance_l3077_307786

-- Define the symbols
variable (triangle diamond heart dot : ℕ)

-- Define the balance relation
def balances (left right : ℕ) : Prop := left = right

-- State the given conditions
axiom balance1 : balances (4 * triangle + 2 * diamond + heart) (21 * dot)
axiom balance2 : balances (2 * triangle) (diamond + heart + 5 * dot)

-- State the theorem to be proved
theorem diamond_and_hearts_balance : balances (diamond + 2 * heart) (11 * dot) := by sorry

end NUMINAMATH_CALUDE_diamond_and_hearts_balance_l3077_307786


namespace NUMINAMATH_CALUDE_solution_count_of_system_l3077_307728

theorem solution_count_of_system (x y : ℂ) : 
  (y = (x + 1)^2 ∧ x * y + y = 1) → 
  (∃! (xr yr : ℝ), yr = (xr + 1)^2 ∧ xr * yr + yr = 1) ∧
  (∃ (xc1 yc1 xc2 yc2 : ℂ), 
    (xc1 ≠ xc2) ∧
    (yc1 = (xc1 + 1)^2 ∧ xc1 * yc1 + yc1 = 1) ∧
    (yc2 = (xc2 + 1)^2 ∧ xc2 * yc2 + yc2 = 1) ∧
    (xc1.im ≠ 0 ∧ xc2.im ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_solution_count_of_system_l3077_307728


namespace NUMINAMATH_CALUDE_ajax_initial_weight_ajax_initial_weight_is_80_l3077_307763

/-- Proves that Ajax's initial weight is 80 kg given the exercise and weight conditions --/
theorem ajax_initial_weight : ℝ → Prop :=
  fun (initial_weight : ℝ) =>
    let pounds_per_kg : ℝ := 2.2
    let weight_loss_per_hour : ℝ := 1.5
    let hours_per_day : ℝ := 2
    let days : ℝ := 14
    let final_weight_pounds : ℝ := 134
    
    let total_weight_loss : ℝ := weight_loss_per_hour * hours_per_day * days
    let initial_weight_pounds : ℝ := final_weight_pounds + total_weight_loss
    
    initial_weight = initial_weight_pounds / pounds_per_kg ∧ initial_weight = 80

theorem ajax_initial_weight_is_80 : ajax_initial_weight 80 := by
  sorry

end NUMINAMATH_CALUDE_ajax_initial_weight_ajax_initial_weight_is_80_l3077_307763


namespace NUMINAMATH_CALUDE_polynomial_problem_l3077_307715

-- Define the polynomials
def B (x : ℝ) : ℝ := 4 * x^2 - 5 * x - 7

theorem polynomial_problem (A : ℝ → ℝ) 
  (h : ∀ x, A x - 2 * (B x) = -2 * x^2 + 10 * x + 14) :
  (∀ x, A x = 6 * x^2) ∧ 
  (∀ x, A x + 2 * (B x) = 14 * x^2 - 10 * x - 14) ∧
  (A (-1) + 2 * (B (-1)) = 10) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_problem_l3077_307715


namespace NUMINAMATH_CALUDE_at_least_one_square_l3077_307773

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : Nat
  height : Nat
  width_gt_one : width > 1
  height_gt_one : height > 1

/-- Represents a division of a square into rectangles -/
structure SquareDivision where
  side_length : Nat
  rectangles : List Rectangle
  total_rectangles : rectangles.length = 17
  covers_square : (rectangles.map (λ r => r.width * r.height)).sum = side_length * side_length

theorem at_least_one_square (d : SquareDivision) (h : d.side_length = 10) :
  ∃ (r : Rectangle), r ∈ d.rectangles ∧ r.width = r.height := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_square_l3077_307773


namespace NUMINAMATH_CALUDE_tangent_line_min_slope_l3077_307709

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 6*x^2 - x + 6

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 12*x - 1

-- Theorem statement
theorem tangent_line_min_slope :
  ∃ (x₀ y₀ : ℝ),
    f x₀ = y₀ ∧
    (∀ x : ℝ, f' x₀ ≤ f' x) ∧
    (13 * x₀ + y₀ - 14 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_min_slope_l3077_307709


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_5_pow_1000_no_zero_digit_l3077_307742

theorem exists_number_divisible_by_5_pow_1000_no_zero_digit :
  ∃ n : ℕ, (5^1000 ∣ n) ∧ (∀ d : ℕ, d < 10 → d ≠ 0 → ∃ k : ℕ, n / 10^k % 10 = d) :=
sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_5_pow_1000_no_zero_digit_l3077_307742


namespace NUMINAMATH_CALUDE_jessica_coins_value_l3077_307703

/-- Represents the value of a coin in cents -/
def coin_value (is_dime : Bool) : ℕ :=
  if is_dime then 10 else 5

/-- Calculates the total value of coins in cents -/
def total_value (num_nickels num_dimes : ℕ) : ℕ :=
  coin_value false * num_nickels + coin_value true * num_dimes

theorem jessica_coins_value :
  ∀ (num_nickels num_dimes : ℕ),
    num_nickels + num_dimes = 30 →
    total_value num_dimes num_nickels - total_value num_nickels num_dimes = 120 →
    total_value num_nickels num_dimes = 165 := by
  sorry

end NUMINAMATH_CALUDE_jessica_coins_value_l3077_307703


namespace NUMINAMATH_CALUDE_trig_identity_l3077_307785

theorem trig_identity (α : Real) (h : Real.cos α ^ 2 = Real.sin α) :
  1 / Real.sin α + Real.cos α ^ 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3077_307785


namespace NUMINAMATH_CALUDE_cafeteria_earnings_l3077_307705

/-- Calculates the total earnings of a cafeteria from selling fruits --/
theorem cafeteria_earnings
  (initial_apples initial_oranges initial_bananas : ℕ)
  (remaining_apples remaining_oranges remaining_bananas : ℕ)
  (apple_cost orange_cost banana_cost : ℚ) :
  initial_apples = 80 →
  initial_oranges = 60 →
  initial_bananas = 40 →
  remaining_apples = 25 →
  remaining_oranges = 15 →
  remaining_bananas = 5 →
  apple_cost = 1.20 →
  orange_cost = 0.75 →
  banana_cost = 0.55 →
  (initial_apples - remaining_apples) * apple_cost +
  (initial_oranges - remaining_oranges) * orange_cost +
  (initial_bananas - remaining_bananas) * banana_cost = 119 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_earnings_l3077_307705


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l3077_307735

theorem floor_plus_self_eq_seventeen_fourths (x : ℝ) : 
  ⌊x⌋ + x = 17/4 → x = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l3077_307735


namespace NUMINAMATH_CALUDE_triangle_side_sum_max_l3077_307748

/-- In a triangle ABC, prove that given certain conditions, b + c has a maximum value of 6 --/
theorem triangle_side_sum_max (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧ 
  a = 3 ∧ 
  1 + (Real.tan A / Real.tan B) = (2 * c / b) ∧ 
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A → 
  b + c ≤ 6 ∧ ∃ b c, b + c = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_max_l3077_307748


namespace NUMINAMATH_CALUDE_line_contains_both_points_l3077_307741

/-- The line equation is 2 - kx = -4y -/
def line_equation (k x y : ℝ) : Prop := 2 - k * x = -4 * y

/-- The first point (2, -1) -/
def point1 : ℝ × ℝ := (2, -1)

/-- The second point (3, -1.5) -/
def point2 : ℝ × ℝ := (3, -1.5)

/-- The line contains both points when k = -1 -/
theorem line_contains_both_points :
  ∃! k : ℝ, line_equation k point1.1 point1.2 ∧ line_equation k point2.1 point2.2 ∧ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_both_points_l3077_307741


namespace NUMINAMATH_CALUDE_inequality_solution_l3077_307739

theorem inequality_solution (x : ℝ) : 
  (3 + 1 / (3 * x - 2) ≥ 5) ∧ (3 * x - 2 ≠ 0) → 
  x ∈ Set.Iio (2 / 3) ∪ Set.Ioc (2 / 3) (5 / 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3077_307739


namespace NUMINAMATH_CALUDE_hyperbola_sum_theorem_l3077_307708

-- Define the hyperbola equation
def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

-- Define the theorem
theorem hyperbola_sum_theorem (h k a b : ℝ) :
  -- Given conditions
  hyperbola_equation h k h k a b ∧
  (h = 3 ∧ k = -5) ∧
  (2 * a = 10) ∧
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ 2 * c = 14) →
  -- Conclusion
  h + k + a + b = 3 + 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_theorem_l3077_307708


namespace NUMINAMATH_CALUDE_medal_distribution_proof_l3077_307758

def total_sprinters : Nat := 10
def american_sprinters : Nat := 4
def medals : Nat := 3

def ways_to_distribute_medals : Nat :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_medalists := non_american_sprinters * (non_american_sprinters - 1) * (non_american_sprinters - 2)
  let one_american_medalist := american_sprinters * medals * (non_american_sprinters * (non_american_sprinters - 1))
  no_american_medalists + one_american_medalist

theorem medal_distribution_proof : 
  ways_to_distribute_medals = 480 := by sorry

end NUMINAMATH_CALUDE_medal_distribution_proof_l3077_307758


namespace NUMINAMATH_CALUDE_sum_90_is_neg_180_l3077_307744

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

/-- Theorem: For the given arithmetic progression, the sum of the first 90 terms is -180 -/
theorem sum_90_is_neg_180 (ap : ArithmeticProgression) 
  (h15 : sum_n ap 15 = 150)
  (h75 : sum_n ap 75 = 30) : 
  sum_n ap 90 = -180 := by
  sorry

end NUMINAMATH_CALUDE_sum_90_is_neg_180_l3077_307744


namespace NUMINAMATH_CALUDE_pure_imaginary_m_value_l3077_307761

/-- A complex number z is defined as z = (m^2+m-2) + (m^2+4m-5)i -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 2) (m^2 + 4*m - 5)

/-- A complex number is pure imaginary if its real part is zero and imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_m_value :
  ∃! m : ℝ, is_pure_imaginary (z m) ∧ m = -2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_value_l3077_307761


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3077_307787

/-- A geometric sequence with five terms where the first term is -1 and the last term is -2 -/
def GeometricSequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -2 = z * r

/-- The product of the middle three terms of the geometric sequence equals ±2√2 -/
theorem geometric_sequence_product (x y z : ℝ) :
  GeometricSequence x y z → x * y * z = 2 * Real.sqrt 2 ∨ x * y * z = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3077_307787


namespace NUMINAMATH_CALUDE_line_through_points_l3077_307714

/-- Given two points A and D in 3D space, this theorem proves that the parametric equations
    of the line passing through these points are of the form x = -3 + 4t, y = 3t, z = 1 + t. -/
theorem line_through_points (A D : ℝ × ℝ × ℝ) (h : A = (-3, 0, 1) ∧ D = (1, 3, 2)) :
  ∃ (f : ℝ → ℝ × ℝ × ℝ), ∀ t : ℝ,
    f t = (-3 + 4*t, 3*t, 1 + t) ∧
    (∃ t₁ t₂ : ℝ, f t₁ = A ∧ f t₂ = D) :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l3077_307714


namespace NUMINAMATH_CALUDE_OPSQ_configurations_l3077_307747

structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩

def isCollinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

def isParallelogram (p q r s : Point) : Prop :=
  (q.x - p.x = s.x - r.x) ∧ (q.y - p.y = s.y - r.y)

theorem OPSQ_configurations (x₁ y₁ x₂ y₂ : ℝ) :
  let P : Point := ⟨x₁, y₁⟩
  let Q : Point := ⟨x₂, y₂⟩
  let S : Point := ⟨2*x₁, 2*y₁⟩
  (isCollinear O P Q ∨ 
   ¬(isCollinear O P Q) ∨ 
   isParallelogram O P S Q) := by sorry

end NUMINAMATH_CALUDE_OPSQ_configurations_l3077_307747


namespace NUMINAMATH_CALUDE_paths_through_B_l3077_307788

/-- The number of paths between two points on a grid -/
def grid_paths (right : ℕ) (down : ℕ) : ℕ :=
  Nat.choose (right + down) down

/-- The position of point A -/
def point_A : ℕ × ℕ := (0, 0)

/-- The position of point B relative to A -/
def A_to_B : ℕ × ℕ := (4, 2)

/-- The position of point C relative to B -/
def B_to_C : ℕ × ℕ := (3, 2)

/-- The total number of steps from A to C -/
def total_steps : ℕ := A_to_B.1 + A_to_B.2 + B_to_C.1 + B_to_C.2

theorem paths_through_B : 
  grid_paths A_to_B.1 A_to_B.2 * grid_paths B_to_C.1 B_to_C.2 = 150 ∧ 
  total_steps = 11 := by
  sorry

end NUMINAMATH_CALUDE_paths_through_B_l3077_307788


namespace NUMINAMATH_CALUDE_min_roads_theorem_l3077_307710

/-- A graph representing cities and roads -/
structure CityGraph where
  num_cities : ℕ
  num_roads : ℕ
  is_connected : Bool

/-- Check if a given number of roads is sufficient for connectivity -/
def is_sufficient (g : CityGraph) : Prop :=
  g.is_connected = true

/-- The minimum number of roads needed for connectivity -/
def min_roads_for_connectivity (num_cities : ℕ) : ℕ :=
  191

/-- Theorem stating that 191 roads are sufficient and necessary for connectivity -/
theorem min_roads_theorem (g : CityGraph) :
  g.num_cities = 21 → 
  (g.num_roads ≥ 191 → is_sufficient g) ∧
  (is_sufficient g → g.num_roads ≥ 191) :=
sorry

#check min_roads_theorem

end NUMINAMATH_CALUDE_min_roads_theorem_l3077_307710


namespace NUMINAMATH_CALUDE_sock_pairs_theorem_l3077_307780

/-- Given an initial number of sock pairs and a number of lost individual socks,
    calculates the maximum number of complete pairs remaining. -/
def maxRemainingPairs (initialPairs : ℕ) (lostSocks : ℕ) : ℕ :=
  initialPairs - min initialPairs lostSocks

/-- Theorem stating that with 25 initial pairs and 12 lost socks,
    the maximum number of complete pairs remaining is 13. -/
theorem sock_pairs_theorem :
  maxRemainingPairs 25 12 = 13 := by
  sorry

#eval maxRemainingPairs 25 12

end NUMINAMATH_CALUDE_sock_pairs_theorem_l3077_307780


namespace NUMINAMATH_CALUDE_quartic_equation_solutions_l3077_307731

theorem quartic_equation_solutions :
  ∀ x : ℝ, x^4 - x^2 - 2 = 0 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quartic_equation_solutions_l3077_307731


namespace NUMINAMATH_CALUDE_tomato_plants_per_row_l3077_307750

/-- Proves that the number of plants in each row is 10, given the conditions of the tomato planting problem -/
theorem tomato_plants_per_row :
  ∀ (rows : ℕ) (yield_per_plant : ℕ) (total_yield : ℕ),
    rows = 30 →
    yield_per_plant = 20 →
    total_yield = 6000 →
    total_yield = rows * yield_per_plant * (total_yield / (rows * yield_per_plant)) →
    total_yield / (rows * yield_per_plant) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_plants_per_row_l3077_307750


namespace NUMINAMATH_CALUDE_b_minus_d_squared_l3077_307798

theorem b_minus_d_squared (a b c d : ℤ) 
  (eq1 : a - b - c + d = 12)
  (eq2 : a + b - c - d = 6) : 
  (b - d)^2 = 9 := by sorry

end NUMINAMATH_CALUDE_b_minus_d_squared_l3077_307798


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l3077_307712

/-- Given a boat traveling downstream, calculates the speed of the stream. -/
theorem stream_speed_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) 
  (h1 : boat_speed = 13)
  (h2 : downstream_distance = 69)
  (h3 : downstream_time = 3.6315789473684212) :
  let downstream_speed := downstream_distance / downstream_time
  let stream_speed := downstream_speed - boat_speed
  stream_speed = 6 := by sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l3077_307712


namespace NUMINAMATH_CALUDE_initial_average_production_l3077_307791

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) 
  (h1 : n = 9)
  (h2 : today_production = 90)
  (h3 : new_average = 54) :
  ∃ initial_average : ℕ, 
    initial_average * n + today_production = new_average * (n + 1) ∧ 
    initial_average = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_production_l3077_307791


namespace NUMINAMATH_CALUDE_square_inequality_l3077_307782

theorem square_inequality (a x y : ℝ) :
  (2 ≤ x ∧ x ≤ 3) ∧ (3 ≤ y ∧ y ≤ 4) →
  ((3 * x - 2 * y - a) * (3 * x - 2 * y - a^2) ≤ 0 ↔ a ≤ -4) :=
by sorry

end NUMINAMATH_CALUDE_square_inequality_l3077_307782


namespace NUMINAMATH_CALUDE_range_of_a_min_value_of_a_l3077_307789

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Statement 1
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f a x ≤ 3) → 0 ≤ a ∧ a ≤ 4 := by sorry

-- Statement 2
theorem min_value_of_a :
  ∃ a : ℝ, a = 1/3 ∧ (∀ x : ℝ, |x - a| + |x + a| ≥ 1 - a) ∧
  (∀ b : ℝ, (∀ x : ℝ, |x - b| + |x + b| ≥ 1 - b) → a ≤ b) := by sorry

end NUMINAMATH_CALUDE_range_of_a_min_value_of_a_l3077_307789


namespace NUMINAMATH_CALUDE_equation_solution_l3077_307768

theorem equation_solution : (25 - 7 = 3 + x) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3077_307768


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3077_307770

theorem reciprocal_of_negative_two :
  (1 : ℚ) / (-2 : ℚ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3077_307770


namespace NUMINAMATH_CALUDE_village_x_decrease_rate_l3077_307743

def village_x_initial_population : ℕ := 68000
def village_y_initial_population : ℕ := 42000
def village_y_growth_rate : ℕ := 800
def years_until_equal : ℕ := 13

theorem village_x_decrease_rate (village_x_decrease_rate : ℕ) : 
  village_x_initial_population - years_until_equal * village_x_decrease_rate = 
  village_y_initial_population + years_until_equal * village_y_growth_rate → 
  village_x_decrease_rate = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_village_x_decrease_rate_l3077_307743


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3077_307765

theorem min_value_of_expression (x y : ℝ) : (x * y + 2)^2 + (x - y)^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3077_307765


namespace NUMINAMATH_CALUDE_inequality_problem_l3077_307769

theorem inequality_problem (a b : ℝ) (h : a ≠ b) :
  (a^2 + b^2 ≥ 2*(a - b - 1)) ∧
  ¬(∀ a b : ℝ, a + b > 2*b^2) ∧
  ¬(∀ a b : ℝ, a^5 + b^5 > a^3*b^2 + a^2*b^3) ∧
  ¬(∀ a b : ℝ, b/a + a/b > 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l3077_307769


namespace NUMINAMATH_CALUDE_cantaloupe_total_l3077_307794

theorem cantaloupe_total (fred_cantaloupes tim_cantaloupes : ℕ) 
  (h1 : fred_cantaloupes = 38) 
  (h2 : tim_cantaloupes = 44) : 
  fred_cantaloupes + tim_cantaloupes = 82 := by
sorry

end NUMINAMATH_CALUDE_cantaloupe_total_l3077_307794


namespace NUMINAMATH_CALUDE_exists_arrangement_for_23_l3077_307726

/-- Fibonacci-like sequence with a specific recurrence relation -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required property for P = 23 -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), F 12 ≡ 0 [ZMOD 23] := by
  sorry

end NUMINAMATH_CALUDE_exists_arrangement_for_23_l3077_307726


namespace NUMINAMATH_CALUDE_exponential_base_theorem_l3077_307724

theorem exponential_base_theorem (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a^x ≤ max a a⁻¹) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, min a a⁻¹ ≤ a^x) ∧
  (max a a⁻¹ - min a a⁻¹ = 1) →
  a = (Real.sqrt 5 + 1) / 2 ∨ a = (Real.sqrt 5 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_exponential_base_theorem_l3077_307724


namespace NUMINAMATH_CALUDE_johnny_red_pencils_l3077_307755

/-- The number of red pencils Johnny bought -/
def total_red_pencils (total_packs : ℕ) (red_per_pack : ℕ) (special_packs : ℕ) (extra_red : ℕ) : ℕ :=
  total_packs * red_per_pack + special_packs * extra_red

/-- Proof that Johnny bought 21 red pencils -/
theorem johnny_red_pencils :
  total_red_pencils 15 1 3 2 = 21 := by
  sorry

#eval total_red_pencils 15 1 3 2

end NUMINAMATH_CALUDE_johnny_red_pencils_l3077_307755


namespace NUMINAMATH_CALUDE_tori_classroom_trash_l3077_307738

/-- Represents the number of pieces of trash picked up in various locations --/
structure TrashCount where
  total : ℕ
  outside : ℕ

/-- Calculates the number of pieces of trash picked up in the classrooms --/
def classroom_trash (t : TrashCount) : ℕ :=
  t.total - t.outside

/-- Theorem stating that for Tori's specific trash counts, the classroom trash is 344 --/
theorem tori_classroom_trash :
  let tori_trash : TrashCount := { total := 1576, outside := 1232 }
  classroom_trash tori_trash = 344 := by
  sorry

#eval classroom_trash { total := 1576, outside := 1232 }

end NUMINAMATH_CALUDE_tori_classroom_trash_l3077_307738


namespace NUMINAMATH_CALUDE_davids_english_marks_l3077_307701

def marks_math : ℕ := 65
def marks_physics : ℕ := 82
def marks_chemistry : ℕ := 67
def marks_biology : ℕ := 85
def average_marks : ℕ := 72
def num_subjects : ℕ := 5

theorem davids_english_marks :
  ∃ (marks_english : ℕ),
    (marks_english + marks_math + marks_physics + marks_chemistry + marks_biology) / num_subjects = average_marks ∧
    marks_english = 61 := by
  sorry

end NUMINAMATH_CALUDE_davids_english_marks_l3077_307701


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l3077_307753

theorem correct_quotient_proof (N : ℕ) (h1 : N % 21 = 0) (h2 : N / 12 = 49) : N / 21 = 28 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l3077_307753
