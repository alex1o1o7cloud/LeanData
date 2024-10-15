import Mathlib

namespace NUMINAMATH_CALUDE_walter_school_allocation_l1541_154169

/-- Represents Walter's work and school allocation details -/
structure WalterFinances where
  days_per_week : ℕ
  hours_per_day : ℕ
  hourly_rate : ℚ
  school_allocation : ℚ

/-- Calculates the fraction of weekly earnings allocated for schooling -/
def school_allocation_fraction (w : WalterFinances) : ℚ :=
  w.school_allocation / (w.days_per_week * w.hours_per_day * w.hourly_rate)

/-- Theorem stating that Walter allocates 3/4 of his weekly earnings for schooling -/
theorem walter_school_allocation :
  let w : WalterFinances := {
    days_per_week := 5,
    hours_per_day := 4,
    hourly_rate := 5,
    school_allocation := 75
  }
  school_allocation_fraction w = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_walter_school_allocation_l1541_154169


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l1541_154134

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 14 * x + b * y + c * z = 0)
  (eq2 : a * x + 24 * y + c * z = 0)
  (eq3 : a * x + b * y + 43 * z = 0)
  (ha : a ≠ 14)
  (hb : b ≠ 24)
  (hc : c ≠ 43)
  (hx : x ≠ 0) :
  a / (a - 14) + b / (b - 24) + c / (c - 43) = 1 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l1541_154134


namespace NUMINAMATH_CALUDE_train_rate_problem_l1541_154189

/-- The constant rate of Train A when two trains meet under specific conditions -/
theorem train_rate_problem (total_distance : ℝ) (train_b_rate : ℝ) (train_a_distance : ℝ) :
  total_distance = 350 →
  train_b_rate = 30 →
  train_a_distance = 200 →
  ∃ (train_a_rate : ℝ),
    train_a_rate * (total_distance - train_a_distance) / train_b_rate = train_a_distance ∧
    train_a_rate = 40 :=
by sorry

end NUMINAMATH_CALUDE_train_rate_problem_l1541_154189


namespace NUMINAMATH_CALUDE_black_ball_probability_l1541_154117

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  orange : ℕ
  black : ℕ
  white : ℕ

/-- Calculates the probability of picking a ball of a specific color -/
def probability (counts : BallCounts) (color : ℕ) : ℚ :=
  color / (counts.orange + counts.black + counts.white)

/-- The main theorem to be proved -/
theorem black_ball_probability (counts : BallCounts) 
  (h1 : counts.orange = 8)
  (h2 : counts.black = 7)
  (h3 : counts.white = 6) :
  probability counts counts.black = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_black_ball_probability_l1541_154117


namespace NUMINAMATH_CALUDE_forest_tree_count_l1541_154194

/-- Calculates the total number of trees in a forest given the side length of a square street,
    the ratio of forest area to street area, and the tree density in the forest. -/
theorem forest_tree_count (street_side : ℝ) (forest_street_ratio : ℝ) (trees_per_sqm : ℝ) : 
  street_side = 100 →
  forest_street_ratio = 3 →
  trees_per_sqm = 4 →
  (street_side^2 * forest_street_ratio * trees_per_sqm : ℝ) = 120000 := by
  sorry

end NUMINAMATH_CALUDE_forest_tree_count_l1541_154194


namespace NUMINAMATH_CALUDE_inscribed_angles_sum_l1541_154124

/-- Given a circle divided into 18 equal arcs, if central angle x spans 3 arcs
    and central angle y spans 6 arcs, then the sum of the corresponding
    inscribed angles x and y is 90°. -/
theorem inscribed_angles_sum (x y : ℝ) : 
  (18 : ℝ) * x = 360 →  -- The circle is divided into 18 equal arcs
  3 * x = y →           -- Central angle y is twice central angle x
  2 * x = 60 →          -- Central angle x spans 3 arcs (3 * 20° = 60°)
  x / 2 + y / 2 = 90    -- Sum of inscribed angles x and y is 90°
  := by sorry

end NUMINAMATH_CALUDE_inscribed_angles_sum_l1541_154124


namespace NUMINAMATH_CALUDE_cloth_cost_price_l1541_154161

theorem cloth_cost_price (total_length : ℕ) (first_part : ℕ) (remaining_part : ℕ)
  (total_price : ℕ) (profit1 : ℕ) (profit2 : ℕ) (cost_price : ℕ) :
  total_length = first_part + remaining_part →
  total_length = 85 →
  first_part = 50 →
  remaining_part = 35 →
  total_price = 8925 →
  profit1 = 15 →
  profit2 = 20 →
  first_part * (cost_price + profit1) + remaining_part * (cost_price + profit2) = total_price →
  cost_price = 88 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l1541_154161


namespace NUMINAMATH_CALUDE_maryann_working_time_l1541_154102

/-- Maryann's working time calculation -/
theorem maryann_working_time 
  (time_calling : ℕ) 
  (accounting_ratio : ℕ) 
  (h1 : time_calling = 70) 
  (h2 : accounting_ratio = 7) : 
  time_calling + accounting_ratio * time_calling = 560 := by
  sorry

end NUMINAMATH_CALUDE_maryann_working_time_l1541_154102


namespace NUMINAMATH_CALUDE_average_yield_is_15_l1541_154119

def rice_field_yields : List ℝ := [12, 13, 15, 17, 18]

theorem average_yield_is_15 :
  (rice_field_yields.sum / rice_field_yields.length : ℝ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_yield_is_15_l1541_154119


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1541_154198

theorem cylinder_surface_area (r h V : ℝ) : 
  r = 1 → V = 4 * Real.pi → V = Real.pi * r^2 * h → 
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 10 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1541_154198


namespace NUMINAMATH_CALUDE_bacteria_increase_l1541_154143

/-- Given an original bacteria count of 600 and a current count of 8917,
    prove that the increase in bacteria count is 8317. -/
theorem bacteria_increase (original : ℕ) (current : ℕ) 
  (h1 : original = 600) (h2 : current = 8917) : 
  current - original = 8317 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_increase_l1541_154143


namespace NUMINAMATH_CALUDE_fisherman_tuna_count_l1541_154190

/-- The number of Red snappers the fisherman gets every day -/
def red_snappers : ℕ := 8

/-- The cost of a Red snapper in dollars -/
def red_snapper_cost : ℕ := 3

/-- The cost of a Tuna in dollars -/
def tuna_cost : ℕ := 2

/-- The total earnings of the fisherman in dollars per day -/
def total_earnings : ℕ := 52

/-- The number of Tunas the fisherman gets every day -/
def tuna_count : ℕ := (total_earnings - red_snappers * red_snapper_cost) / tuna_cost

theorem fisherman_tuna_count : tuna_count = 14 := by
  sorry

end NUMINAMATH_CALUDE_fisherman_tuna_count_l1541_154190


namespace NUMINAMATH_CALUDE_number_of_players_is_five_l1541_154163

/-- Represents the number of chips each player receives -/
def chips_per_player (m : ℕ) (n : ℕ) : ℕ := n * m

/-- Represents the number of chips taken by the i-th player -/
def chips_taken (i : ℕ) (m : ℕ) (remaining : ℕ) : ℕ :=
  i * m + remaining / 6

/-- The main theorem stating that the number of players is 5 -/
theorem number_of_players_is_five (m : ℕ) (total_chips : ℕ) :
  ∃ (n : ℕ),
    n = 5 ∧
    (∀ i : ℕ, i ≤ n →
      chips_taken i m (total_chips - (chips_per_player m i)) =
      chips_per_player m n) :=
sorry

end NUMINAMATH_CALUDE_number_of_players_is_five_l1541_154163


namespace NUMINAMATH_CALUDE_problem_solution_l1541_154120

theorem problem_solution : 
  ∀ a b : ℝ, 
  (∃ k : ℝ, k^2 = a + b - 5 ∧ (k = 3 ∨ k = -3)) →
  (a - b + 4)^(1/3) = 2 →
  a = 9 ∧ b = 5 ∧ Real.sqrt (4 * (a - b)) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1541_154120


namespace NUMINAMATH_CALUDE_unique_p_q_solution_l1541_154195

theorem unique_p_q_solution :
  ∀ p q : ℝ,
    p ≠ q →
    p > 1 →
    q > 1 →
    1 / p + 1 / q = 1 →
    p * q = 9 →
    ((p = (9 + 3 * Real.sqrt 5) / 2 ∧ q = (9 - 3 * Real.sqrt 5) / 2) ∨
     (p = (9 - 3 * Real.sqrt 5) / 2 ∧ q = (9 + 3 * Real.sqrt 5) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_p_q_solution_l1541_154195


namespace NUMINAMATH_CALUDE_dog_food_bag_weight_l1541_154131

/-- Proves that the weight of each bag of dog food is 20 pounds -/
theorem dog_food_bag_weight :
  let cup_weight : ℚ := 1/4  -- Weight of a cup of dog food in pounds
  let num_dogs : ℕ := 2  -- Number of dogs
  let cups_per_meal : ℕ := 6  -- Cups of food per meal per dog
  let meals_per_day : ℕ := 2  -- Number of meals per day
  let bags_per_month : ℕ := 9  -- Number of bags bought per month
  let days_per_month : ℕ := 30  -- Number of days in a month
  
  let daily_consumption : ℚ := num_dogs * cups_per_meal * meals_per_day * cup_weight
  let monthly_consumption : ℚ := daily_consumption * days_per_month
  let bag_weight : ℚ := monthly_consumption / bags_per_month

  bag_weight = 20 := by
    sorry

end NUMINAMATH_CALUDE_dog_food_bag_weight_l1541_154131


namespace NUMINAMATH_CALUDE_cool_double_l1541_154108

def is_cool (n : ℕ) : Prop := ∃ a b : ℕ, n = a^2 + b^2

theorem cool_double {k : ℕ} (h : is_cool k) : is_cool (2 * k) := by
  sorry

end NUMINAMATH_CALUDE_cool_double_l1541_154108


namespace NUMINAMATH_CALUDE_max_triangles_theorem_l1541_154101

/-- Represents a convex n-gon with diagonals drawn such that no three or more intersect at a single point. -/
structure ConvexPolygonWithDiagonals where
  n : ℕ
  is_convex : Bool
  no_triple_intersection : Bool

/-- Calculates the maximum number of triangles formed by diagonals in a convex n-gon. -/
def max_triangles (polygon : ConvexPolygonWithDiagonals) : ℕ :=
  if polygon.n % 2 = 0 then
    2 * polygon.n - 4
  else
    2 * polygon.n - 5

/-- Theorem stating the maximum number of triangles formed by diagonals in a convex n-gon. -/
theorem max_triangles_theorem (polygon : ConvexPolygonWithDiagonals) :
  polygon.is_convex ∧ polygon.no_triple_intersection →
  max_triangles polygon = if polygon.n % 2 = 0 then 2 * polygon.n - 4 else 2 * polygon.n - 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_triangles_theorem_l1541_154101


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1541_154193

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℚ, 
    (3 * X^5 + 16 * X^4 - 17 * X^3 - 100 * X^2 + 32 * X + 90 : Polynomial ℚ) = 
    (X^3 + 8 * X^2 - X - 6) * q + (422 * X^2 + 48 * X - 294) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1541_154193


namespace NUMINAMATH_CALUDE_ball_distribution_problem_l1541_154166

def total_arrangements : ℕ := 90
def arrangements_with_1_and_2_together : ℕ := 18

theorem ball_distribution_problem :
  let n_balls : ℕ := 6
  let n_boxes : ℕ := 3
  let balls_per_box : ℕ := 2
  total_arrangements - arrangements_with_1_and_2_together = 72 :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_problem_l1541_154166


namespace NUMINAMATH_CALUDE_point_on_line_implies_tan_2theta_l1541_154127

theorem point_on_line_implies_tan_2theta (θ : ℝ) : 
  2 * Real.sin θ + Real.cos θ = 0 → Real.tan (2 * θ) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_implies_tan_2theta_l1541_154127


namespace NUMINAMATH_CALUDE_sequence_periodicity_l1541_154114

/-- A cubic polynomial with rational coefficients -/
def CubicPolynomial (α : ℚ → ℚ) : Prop :=
  ∃ a b c d : ℚ, ∀ x, α x = a * x^3 + b * x^2 + c * x + d

/-- A sequence of rational numbers satisfying the given condition -/
def SequenceSatisfyingCondition (p : ℚ → ℚ) (q : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, q n = p (q (n + 1))

theorem sequence_periodicity
  (p : ℚ → ℚ) (q : ℕ → ℚ)
  (h_cubic : CubicPolynomial p)
  (h_seq : SequenceSatisfyingCondition p q) :
  ∃ k : ℕ, k ≥ 1 ∧ ∀ n : ℕ, n ≥ 1 → q (n + k) = q n :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l1541_154114


namespace NUMINAMATH_CALUDE_odd_function_property_l1541_154121

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_symmetry : ∀ x, f x = f (2 - x))
  (h_value : f (-1) = 1) :
  f 2021 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1541_154121


namespace NUMINAMATH_CALUDE_parallel_line_equation_l1541_154110

/-- Given a line y = (5/3)x + 10, prove that a parallel line L
    that is 5 units away from it has the equation
    y = (5/3)x + (10 ± (5√34)/3) -/
theorem parallel_line_equation (x y : ℝ) :
  let original_line := fun x => (5/3) * x + 10
  let distance := 5
  let slope := 5/3
  let perpendicular_slope := -3/5
  let c := 10
  ∃ L : ℝ → ℝ,
    (∀ x, L x = slope * x + (c + distance * Real.sqrt (slope^2 + 1))) ∨
    (∀ x, L x = slope * x + (c - distance * Real.sqrt (slope^2 + 1))) ∧
    (∀ x, |L x - original_line x| / Real.sqrt (1 + perpendicular_slope^2) = distance) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l1541_154110


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1541_154171

theorem complex_magnitude_problem (z : ℂ) (h : z * (2 + Complex.I) = 2 - Complex.I) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1541_154171


namespace NUMINAMATH_CALUDE_hash_triple_100_l1541_154146

def hash (N : ℝ) : ℝ := 0.5 * N - 2

theorem hash_triple_100 : hash (hash (hash 100)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_hash_triple_100_l1541_154146


namespace NUMINAMATH_CALUDE_work_completion_theorem_l1541_154115

theorem work_completion_theorem (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) (new_men : ℕ) :
  initial_men = 36 →
  initial_days = 18 →
  new_days = 8 →
  initial_men * initial_days = new_men * new_days →
  new_men = 81 := by
sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l1541_154115


namespace NUMINAMATH_CALUDE_halloween_candy_distribution_l1541_154173

theorem halloween_candy_distribution (initial_candy : ℕ) (eaten_candy : ℕ) (num_piles : ℕ) : 
  initial_candy = 32 → 
  eaten_candy = 12 → 
  num_piles = 4 → 
  (initial_candy - eaten_candy) / num_piles = 5 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_distribution_l1541_154173


namespace NUMINAMATH_CALUDE_range_of_m_for_inequality_l1541_154112

theorem range_of_m_for_inequality (m : ℝ) : 
  (∀ x : ℕ+, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) ↔ (3 * (x : ℝ) - 3 * m ≤ -2 * m)) → 
  (12 ≤ m ∧ m < 15) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_inequality_l1541_154112


namespace NUMINAMATH_CALUDE_franks_age_l1541_154187

theorem franks_age (frank : ℕ) (gabriel : ℕ) : 
  gabriel = frank - 3 → 
  frank + gabriel = 17 → 
  frank = 10 := by sorry

end NUMINAMATH_CALUDE_franks_age_l1541_154187


namespace NUMINAMATH_CALUDE_allowance_calculation_l1541_154177

/-- Represents John's weekly allowance in dollars -/
def weekly_allowance : ℝ := 2.10

/-- The fraction of allowance spent at the arcade -/
def arcade_fraction : ℚ := 3/5

/-- The fraction of remaining allowance spent at the toy store -/
def toy_store_fraction : ℚ := 2/7

/-- The fraction of remaining allowance spent at the bookstore -/
def bookstore_fraction : ℚ := 1/3

/-- The amount spent at the candy store in dollars -/
def candy_store_amount : ℝ := 0.40

/-- Theorem stating that given the spending pattern, the initial allowance was $2.10 -/
theorem allowance_calculation (A : ℝ) :
  A * (1 - arcade_fraction) * (1 - toy_store_fraction) * (1 - bookstore_fraction) = candy_store_amount →
  A = weekly_allowance := by
  sorry

#check allowance_calculation

end NUMINAMATH_CALUDE_allowance_calculation_l1541_154177


namespace NUMINAMATH_CALUDE_meeting_unexpectedly_is_random_l1541_154103

/-- Represents an event --/
inductive Event
| WinterToSpring
| FishingMoonInWater
| SeekingFishOnTree
| MeetingUnexpectedly

/-- Defines whether an event is certain --/
def isCertain : Event → Prop
| Event.WinterToSpring => True
| _ => False

/-- Defines whether an event is impossible --/
def isImpossible : Event → Prop
| Event.FishingMoonInWater => True
| Event.SeekingFishOnTree => True
| _ => False

/-- Defines a random event --/
def isRandom (e : Event) : Prop :=
  ¬(isCertain e) ∧ ¬(isImpossible e)

/-- Theorem: Meeting unexpectedly is a random event --/
theorem meeting_unexpectedly_is_random :
  isRandom Event.MeetingUnexpectedly :=
by sorry

end NUMINAMATH_CALUDE_meeting_unexpectedly_is_random_l1541_154103


namespace NUMINAMATH_CALUDE_seven_c_plus_seven_d_equals_five_l1541_154149

-- Define the function h
def h (x : ℝ) : ℝ := 7 * x - 6

-- Define the function f
def f (c d x : ℝ) : ℝ := c * x + d

-- State the theorem
theorem seven_c_plus_seven_d_equals_five 
  (c d : ℝ) 
  (h_def : ∀ x, h x = 7 * x - 6)
  (h_inverse : ∀ x, h x = f c d⁻¹ x - 2)
  (f_inverse : ∀ x, f c d (f c d⁻¹ x) = x) :
  7 * c + 7 * d = 5 := by
sorry

end NUMINAMATH_CALUDE_seven_c_plus_seven_d_equals_five_l1541_154149


namespace NUMINAMATH_CALUDE_ratio_equality_l1541_154197

theorem ratio_equality (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) :
  (a / 3) / (b / 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1541_154197


namespace NUMINAMATH_CALUDE_number_components_l1541_154145

def number : ℕ := 1234000000

theorem number_components : 
  (number / 100000000 = 12) ∧ 
  ((number / 10000000) % 10 = 3) ∧ 
  ((number / 1000000) % 10 = 4) := by
  sorry

end NUMINAMATH_CALUDE_number_components_l1541_154145


namespace NUMINAMATH_CALUDE_kids_in_restaurant_group_l1541_154196

/-- Represents the number of kids in a restaurant group given certain conditions. -/
def number_of_kids (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) : ℕ :=
  total_people - (total_cost / adult_meal_cost)

/-- Theorem stating that given the problem conditions, the number of kids is 9. -/
theorem kids_in_restaurant_group :
  let total_people : ℕ := 13
  let adult_meal_cost : ℕ := 7
  let total_cost : ℕ := 28
  number_of_kids total_people adult_meal_cost total_cost = 9 := by
sorry

#eval number_of_kids 13 7 28

end NUMINAMATH_CALUDE_kids_in_restaurant_group_l1541_154196


namespace NUMINAMATH_CALUDE_parallel_line_through_circle_center_l1541_154160

/-- Given a circle C and a line l1, prove that the line l passing through the center of C and parallel to l1 has the equation 2x - 3y - 8 = 0 -/
theorem parallel_line_through_circle_center 
  (C : (ℝ × ℝ) → Prop)
  (l1 : (ℝ × ℝ) → Prop)
  (hC : C = λ (x, y) => (x - 1)^2 + (y + 2)^2 = 5)
  (hl1 : l1 = λ (x, y) => 2*x - 3*y + 6 = 0) :
  ∃ l : (ℝ × ℝ) → Prop, 
    (∀ p, C p → (p.1 - 1)^2 + (p.2 + 2)^2 = 5) ∧ 
    (∀ p, l1 p → 2*p.1 - 3*p.2 + 6 = 0) ∧
    (l = λ (x, y) => 2*x - 3*y - 8 = 0) ∧
    (∃ c, C c ∧ l c) ∧
    (∀ p q : ℝ × ℝ, l p → l q → l1 ((p.1 + q.1)/2, (p.2 + q.2)/2)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_circle_center_l1541_154160


namespace NUMINAMATH_CALUDE_smallest_positive_t_value_l1541_154109

theorem smallest_positive_t_value (p q r s t : ℤ) : 
  (∀ x : ℝ, p * x^4 + q * x^3 + r * x^2 + s * x + t = 0 ↔ x = -3 ∨ x = 4 ∨ x = 6 ∨ x = 1/2) →
  t > 0 →
  (∀ t' : ℤ, t' > 0 ∧ (∀ x : ℝ, p * x^4 + q * x^3 + r * x^2 + s * x + t' = 0 ↔ x = -3 ∨ x = 4 ∨ x = 6 ∨ x = 1/2) → t' ≥ t) →
  t = 72 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_t_value_l1541_154109


namespace NUMINAMATH_CALUDE_committee_probability_l1541_154168

def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 5

theorem committee_probability :
  (Nat.choose total_members committee_size - 
   (Nat.choose boys committee_size + Nat.choose girls committee_size)) / 
   Nat.choose total_members committee_size = 133146 / 142506 := by
sorry

end NUMINAMATH_CALUDE_committee_probability_l1541_154168


namespace NUMINAMATH_CALUDE_discount_approximation_l1541_154148

/-- Calculates the discount given cost price, markup percentage, and profit percentage -/
def calculate_discount (cost_price : ℝ) (markup_percentage : ℝ) (profit_percentage : ℝ) : ℝ :=
  let marked_price := cost_price * (1 + markup_percentage)
  let selling_price := cost_price * (1 + profit_percentage)
  marked_price - selling_price

/-- Theorem stating that the discount is approximately 50 given the problem conditions -/
theorem discount_approximation :
  let cost_price : ℝ := 180
  let markup_percentage : ℝ := 0.4778
  let profit_percentage : ℝ := 0.20
  let discount := calculate_discount cost_price markup_percentage profit_percentage
  ∃ ε > 0, |discount - 50| < ε :=
sorry

end NUMINAMATH_CALUDE_discount_approximation_l1541_154148


namespace NUMINAMATH_CALUDE_fraction_sum_squared_l1541_154183

theorem fraction_sum_squared (a b c : ℝ) (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_squared_l1541_154183


namespace NUMINAMATH_CALUDE_solution_set_f_inequality_range_of_m_for_nonempty_solution_l1541_154184

-- Define the functions f and g
def f (x : ℝ) := |x - 2|
def g (m : ℝ) (x : ℝ) := -|x + 7| + 3 * m

-- Theorem for the first part of the problem
theorem solution_set_f_inequality (x : ℝ) :
  f x + x^2 - 4 > 0 ↔ x > 2 ∨ x < -1 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_m_for_nonempty_solution (m : ℝ) :
  (∃ x : ℝ, f x < g m x) ↔ m > 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_inequality_range_of_m_for_nonempty_solution_l1541_154184


namespace NUMINAMATH_CALUDE_product_scaling_l1541_154125

theorem product_scaling (a b c : ℕ) (ha : a = 268) (hb : b = 74) (hc : c = 19832) 
  (h : a * b = c) : (2.68 : ℝ) * 0.74 = 1.9832 := by
  sorry

end NUMINAMATH_CALUDE_product_scaling_l1541_154125


namespace NUMINAMATH_CALUDE_function_shift_l1541_154126

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_shift (h : f 0 = 2) : f (-1 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_shift_l1541_154126


namespace NUMINAMATH_CALUDE_number_equals_eight_l1541_154111

theorem number_equals_eight (x : ℝ) : 0.75 * x + 2 = 8 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_eight_l1541_154111


namespace NUMINAMATH_CALUDE_apple_juice_problem_l1541_154170

theorem apple_juice_problem (initial_amount : ℚ) (maria_fraction : ℚ) (john_fraction : ℚ) :
  initial_amount = 3/4 →
  maria_fraction = 1/2 →
  john_fraction = 1/3 →
  let remaining_after_maria := initial_amount - (maria_fraction * initial_amount)
  john_fraction * remaining_after_maria = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_apple_juice_problem_l1541_154170


namespace NUMINAMATH_CALUDE_expression_evaluation_l1541_154159

theorem expression_evaluation :
  let x : ℚ := -1
  let expr := (x - 3) / (2 * x - 4) / ((5 / (x - 2)) - x - 2)
  expr = -1/4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1541_154159


namespace NUMINAMATH_CALUDE_billys_old_score_l1541_154133

/-- Billy's video game score problem -/
theorem billys_old_score (points_per_round : ℕ) (rounds_to_beat : ℕ) (old_score : ℕ) : 
  points_per_round = 2 → rounds_to_beat = 363 → old_score = points_per_round * rounds_to_beat → old_score = 726 := by
  sorry

#check billys_old_score

end NUMINAMATH_CALUDE_billys_old_score_l1541_154133


namespace NUMINAMATH_CALUDE_recipe_total_cups_l1541_154129

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of flour -/
def totalCups (ratio : RecipeRatio) (flourAmount : ℕ) : ℕ :=
  let unitAmount := flourAmount / ratio.flour
  unitAmount * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem: Given the specified ratio and flour amount, the total cups of ingredients is 30 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := { butter := 2, flour := 5, sugar := 3 }
  let flourAmount : ℕ := 15
  totalCups ratio flourAmount = 30 := by
  sorry


end NUMINAMATH_CALUDE_recipe_total_cups_l1541_154129


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l1541_154186

/-- Calculate the net rate of pay for a driver --/
theorem driver_net_pay_rate (travel_time : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (payment_rate : ℝ) (gasoline_cost : ℝ) :
  travel_time = 3 ∧ 
  speed = 45 ∧ 
  fuel_efficiency = 36 ∧ 
  payment_rate = 0.60 ∧ 
  gasoline_cost = 2.50 → 
  (payment_rate * speed * travel_time - 
   (speed * travel_time / fuel_efficiency) * gasoline_cost) / travel_time = 23.875 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l1541_154186


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l1541_154106

/-- A parabola with directrix y = 1/2 has the standard equation x^2 = -2y -/
theorem parabola_standard_equation (p : ℝ) (h : p > 0) :
  let directrix : ℝ → ℝ := λ y => 1/2
  let parabola : ℝ → ℝ → Prop := λ x y => x^2 = -2*p*y
  (∀ x y, parabola x y ↔ y = -(x^2)/(2*p)) ∧ p = 1 → 
  ∀ x y, parabola x y ↔ x^2 = -2*y :=
by sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l1541_154106


namespace NUMINAMATH_CALUDE_number_of_tenths_l1541_154136

theorem number_of_tenths (n : ℚ) : (375 : ℚ) * (1 / 10 : ℚ) = n → n = (37.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_number_of_tenths_l1541_154136


namespace NUMINAMATH_CALUDE_max_monotone_interval_l1541_154181

theorem max_monotone_interval (f : ℝ → ℝ) (h : f = λ x => Real.sin (Real.pi * x - Real.pi / 6)) :
  (∃ m : ℝ, m = 2/3 ∧ 
   (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ m → f x₁ < f x₂) ∧
   (∀ m' : ℝ, m' > m → ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ m' ∧ f x₁ ≥ f x₂)) :=
sorry

end NUMINAMATH_CALUDE_max_monotone_interval_l1541_154181


namespace NUMINAMATH_CALUDE_other_root_is_one_l1541_154105

-- Define the quadratic equation
def quadratic (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 3

-- Theorem statement
theorem other_root_is_one (b : ℝ) :
  (∃ x : ℝ, quadratic b x = 0 ∧ x = 3) →
  (∃ y : ℝ, y ≠ 3 ∧ quadratic b y = 0 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_other_root_is_one_l1541_154105


namespace NUMINAMATH_CALUDE_jefferson_high_club_overlap_l1541_154113

/-- Represents the number of students in both robotics and science clubs -/
def students_in_both_clubs (total : ℕ) (robotics : ℕ) (science : ℕ) (either : ℕ) : ℕ :=
  robotics + science - either

/-- Theorem: Given the conditions from Jefferson High School, 
    prove that there are 20 students in both robotics and science clubs -/
theorem jefferson_high_club_overlap :
  students_in_both_clubs 300 80 130 190 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jefferson_high_club_overlap_l1541_154113


namespace NUMINAMATH_CALUDE_favorite_numbers_sum_of_squares_l1541_154142

/-- The sum of the squares of Misty's, Glory's, and Dawn's favorite numbers -/
def sumOfSquares (gloryFavorite : ℕ) : ℕ :=
  let mistyFavorite := gloryFavorite / 3
  let dawnFavorite := gloryFavorite * 2
  mistyFavorite ^ 2 + gloryFavorite ^ 2 + dawnFavorite ^ 2

/-- Theorem stating that the sum of squares of the favorite numbers is 1,035,000 -/
theorem favorite_numbers_sum_of_squares :
  sumOfSquares 450 = 1035000 := by
  sorry

#eval sumOfSquares 450

end NUMINAMATH_CALUDE_favorite_numbers_sum_of_squares_l1541_154142


namespace NUMINAMATH_CALUDE_problem_solution_l1541_154144

theorem problem_solution :
  ∀ (x y z : ℝ),
  (x + x = y * x) →
  (x + x = z * z) →
  (y = 3) →
  (x * z = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1541_154144


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l1541_154132

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (abs x - 1) / (x - 1) = 0 → x = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l1541_154132


namespace NUMINAMATH_CALUDE_total_celestial_bodies_count_l1541_154165

/-- A galaxy with specific ratios of celestial bodies -/
structure Galaxy where
  planets : ℕ
  solar_systems : ℕ
  stars : ℕ
  solar_system_planet_ratio : solar_systems = 8 * planets
  star_solar_system_ratio : stars = 4 * solar_systems
  planet_count : planets = 20

/-- The total number of celestial bodies in the galaxy -/
def total_celestial_bodies (g : Galaxy) : ℕ :=
  g.planets + g.solar_systems + g.stars

/-- Theorem stating that the total number of celestial bodies is 820 -/
theorem total_celestial_bodies_count (g : Galaxy) :
  total_celestial_bodies g = 820 := by
  sorry

end NUMINAMATH_CALUDE_total_celestial_bodies_count_l1541_154165


namespace NUMINAMATH_CALUDE_quarter_circles_sum_exceeds_circumference_l1541_154118

/-- Theorem: As the number of divisions approaches infinity, the sum of the lengths of quarter-circles
    constructed on equal parts of a circle's circumference exceeds the original circumference. -/
theorem quarter_circles_sum_exceeds_circumference (r : ℝ) (hr : r > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → π * π * r > 2 * π * r := by
  sorry

#check quarter_circles_sum_exceeds_circumference

end NUMINAMATH_CALUDE_quarter_circles_sum_exceeds_circumference_l1541_154118


namespace NUMINAMATH_CALUDE_players_quit_l1541_154153

def video_game_problem (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  initial_players - (total_lives / lives_per_player)

theorem players_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ)
  (h1 : initial_players = 8)
  (h2 : lives_per_player = 5)
  (h3 : total_lives = 15) :
  video_game_problem initial_players lives_per_player total_lives = 5 := by
  sorry

#eval video_game_problem 8 5 15

end NUMINAMATH_CALUDE_players_quit_l1541_154153


namespace NUMINAMATH_CALUDE_intersection_complement_subset_condition_l1541_154172

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1: When a = -2, A ∩ (ℝ \ B) = {x | -1 ≤ x ≤ 1}
theorem intersection_complement (a : ℝ) (h : a = -2) :
  A a ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: A ⊆ B if and only if a ≥ 2
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_subset_condition_l1541_154172


namespace NUMINAMATH_CALUDE_elvis_editing_time_l1541_154152

theorem elvis_editing_time (num_songs : ℕ) (studio_hours : ℕ) (record_time : ℕ) (write_time : ℕ)
  (h1 : num_songs = 10)
  (h2 : studio_hours = 5)
  (h3 : record_time = 12)
  (h4 : write_time = 15) :
  (studio_hours * 60) - (num_songs * write_time + num_songs * record_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_elvis_editing_time_l1541_154152


namespace NUMINAMATH_CALUDE_roots_sum_equation_l1541_154151

theorem roots_sum_equation (a b : ℝ) : 
  (a^2 - 4*a + 4 = 0) → 
  (b^2 - 4*b + 4 = 0) → 
  2*(a + b) = 8 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_equation_l1541_154151


namespace NUMINAMATH_CALUDE_modular_congruence_l1541_154138

theorem modular_congruence (a b n : ℤ) : 
  a % 48 = 25 →
  b % 48 = 80 →
  150 ≤ n →
  n ≤ 191 →
  (a - b) % 48 = n % 48 ↔ n = 185 := by
sorry

end NUMINAMATH_CALUDE_modular_congruence_l1541_154138


namespace NUMINAMATH_CALUDE_f_max_range_l1541_154157

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then a * Real.log x - x^2 - 2
  else if x < 0 then x + 1/x + a
  else 0  -- This value doesn't matter as x ≠ 0 in the problem

theorem f_max_range (a : ℝ) :
  (∀ x : ℝ, f a x ≤ f a (-1)) →
  0 ≤ a ∧ a ≤ 2 * Real.exp 3 :=
sorry

end NUMINAMATH_CALUDE_f_max_range_l1541_154157


namespace NUMINAMATH_CALUDE_chocolate_theorem_l1541_154150

-- Define the parameters of the problem
def chocolate_cost : ℕ := 1
def wrappers_per_exchange : ℕ := 3
def initial_money : ℕ := 15

-- Define a function to calculate the maximum number of chocolates
def max_chocolates (cost : ℕ) (exchange_rate : ℕ) (money : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- State the theorem
theorem chocolate_theorem :
  max_chocolates chocolate_cost wrappers_per_exchange initial_money = 22 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l1541_154150


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1541_154158

/-- Given two positive integers with LCM 2310, HCF 30, and one of them being 210, prove the other is 330 -/
theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  B = 210 →
  A = 330 := by sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1541_154158


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l1541_154178

/-- A quadratic equation of the form x^2 - 8x + c has non-real roots if and only if c > 16 -/
theorem quadratic_non_real_roots (c : ℝ) : 
  (∀ x : ℂ, x^2 - 8*x + c = 0 → x.im ≠ 0) ↔ c > 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l1541_154178


namespace NUMINAMATH_CALUDE_remainder_333_power_333_mod_11_l1541_154199

theorem remainder_333_power_333_mod_11 : 333^333 % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_333_power_333_mod_11_l1541_154199


namespace NUMINAMATH_CALUDE_peach_count_l1541_154180

/-- Given a basket of peaches with red and green peaches, calculate the total number of peaches -/
def total_peaches (red_peaches green_peaches : ℕ) : ℕ :=
  red_peaches + green_peaches

/-- Theorem: Given 1 basket with 4 red peaches and 6 green peaches, the total number of peaches is 10 -/
theorem peach_count : total_peaches 4 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_peach_count_l1541_154180


namespace NUMINAMATH_CALUDE_range_of_m_l1541_154156

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := (m - 2) / (m - 3) ≤ 2/3

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + m^2 > 0

-- Define the set of m values that satisfy the conditions
def M : Set ℝ := {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- State the theorem
theorem range_of_m : M = {m : ℝ | m < -2 ∨ (0 ≤ m ∧ m ≤ 2) ∨ m ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1541_154156


namespace NUMINAMATH_CALUDE_tetrahedron_has_four_faces_l1541_154188

/-- A tetrahedron is a type of pyramid with a triangular base -/
structure Tetrahedron where
  is_pyramid : Bool
  has_triangular_base : Bool

/-- The number of faces in a tetrahedron -/
def num_faces (t : Tetrahedron) : Nat :=
  4

theorem tetrahedron_has_four_faces (t : Tetrahedron) :
  t.is_pyramid = true → t.has_triangular_base = true → num_faces t = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_has_four_faces_l1541_154188


namespace NUMINAMATH_CALUDE_intersection_M_N_l1541_154107

def M : Set ℝ := {x | 2 * x - x^2 ≥ 0}

def N : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - x^2)}

theorem intersection_M_N : M ∩ N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1541_154107


namespace NUMINAMATH_CALUDE_reciprocal_of_proper_fraction_greater_l1541_154191

theorem reciprocal_of_proper_fraction_greater {a b : ℚ} (h1 : 0 < a) (h2 : a < b) :
  b / a > a / b :=
sorry

end NUMINAMATH_CALUDE_reciprocal_of_proper_fraction_greater_l1541_154191


namespace NUMINAMATH_CALUDE_subtraction_to_addition_division_to_multiplication_problem_1_problem_2_l1541_154155

-- Problem 1
theorem subtraction_to_addition (a b : ℤ) : a - b = a + (-b) := by sorry

-- Problem 2
theorem division_to_multiplication (a : ℚ) (b : ℚ) (h : b ≠ 0) :
  a / b = a * (1 / b) := by sorry

-- Specific instances
theorem problem_1 : -8 - 5 = -8 + (-5) := by sorry

theorem problem_2 : (1 : ℚ) / 2 / (-2) = (1 : ℚ) / 2 * (-1 / 2) := by sorry

end NUMINAMATH_CALUDE_subtraction_to_addition_division_to_multiplication_problem_1_problem_2_l1541_154155


namespace NUMINAMATH_CALUDE_athlete_speed_l1541_154192

/-- Given an athlete who runs 200 meters in 40 seconds, prove that their speed is 5 meters per second. -/
theorem athlete_speed (distance : Real) (time : Real) (speed : Real) 
  (h1 : distance = 200) 
  (h2 : time = 40) 
  (h3 : speed = distance / time) : speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_athlete_speed_l1541_154192


namespace NUMINAMATH_CALUDE_max_pies_36_l1541_154123

/-- Calculates the maximum number of pies that can be made given a certain number of apples,
    where every two pies require 12 apples and every third pie needs an extra apple. -/
def max_pies (total_apples : ℕ) : ℕ :=
  let basic_pies := 2 * (total_apples / 12)
  let extra_apples := basic_pies / 3
  let adjusted_apples := total_apples - extra_apples
  let full_sets := adjusted_apples / 12
  let remaining_apples := adjusted_apples % 12
  2 * full_sets + if remaining_apples ≥ 6 then 1 else 0

/-- Theorem stating that given 36 apples, the maximum number of pies that can be made is 9. -/
theorem max_pies_36 : max_pies 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_pies_36_l1541_154123


namespace NUMINAMATH_CALUDE_quadratic_root_square_l1541_154141

theorem quadratic_root_square (p : ℝ) : 
  (∃ a b : ℝ, a ≠ b ∧ 
   a^2 - p*a + p = 0 ∧ 
   b^2 - p*b + p = 0 ∧ 
   (a = b^2 ∨ b = a^2)) ↔ 
  (p = 2 + Real.sqrt 5 ∨ p = 2 - Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_square_l1541_154141


namespace NUMINAMATH_CALUDE_fifth_score_calculation_l1541_154104

theorem fifth_score_calculation (s1 s2 s3 s4 : ℕ) (avg : ℚ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76) (h4 : s4 = 80) (h5 : avg = 76.6) :
  ∃ (s5 : ℕ), s5 = 95 ∧ (s1 + s2 + s3 + s4 + s5 : ℚ) / 5 = avg :=
sorry

end NUMINAMATH_CALUDE_fifth_score_calculation_l1541_154104


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_open_zero_one_closed_l1541_154162

def M : Set ℝ := {x | 0 < Real.log (x + 1) ∧ Real.log (x + 1) < 3}

def N : Set ℝ := {y | ∃ x ∈ M, y = Real.sin x}

theorem M_intersect_N_eq_open_zero_one_closed : M ∩ N = Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_open_zero_one_closed_l1541_154162


namespace NUMINAMATH_CALUDE_number_of_friends_l1541_154135

theorem number_of_friends : ℕ :=
  let melanie_cards : ℕ := sorry
  let benny_cards : ℕ := sorry
  let sally_cards : ℕ := sorry
  let jessica_cards : ℕ := sorry
  have total_cards : melanie_cards + benny_cards + sally_cards + jessica_cards = 12 := by sorry
  4

#check number_of_friends

end NUMINAMATH_CALUDE_number_of_friends_l1541_154135


namespace NUMINAMATH_CALUDE_prob_both_odd_is_one_sixth_l1541_154179

/-- The set of numbers to draw from -/
def S : Finset ℕ := {1, 2, 3, 4}

/-- A function to determine if a number is odd -/
def isOdd (n : ℕ) : Bool := n % 2 = 1

/-- The set of all possible pairs of numbers drawn without replacement -/
def allPairs : Finset (ℕ × ℕ) := S.product S |>.filter (fun (a, b) => a ≠ b)

/-- The set of pairs where both numbers are odd -/
def oddPairs : Finset (ℕ × ℕ) := allPairs.filter (fun (a, b) => isOdd a ∧ isOdd b)

/-- The probability of drawing two odd numbers without replacement -/
def probBothOdd : ℚ := (oddPairs.card : ℚ) / (allPairs.card : ℚ)

theorem prob_both_odd_is_one_sixth : probBothOdd = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_odd_is_one_sixth_l1541_154179


namespace NUMINAMATH_CALUDE_circle_radius_in_ellipse_l1541_154137

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 + 6 * y^2 = 8

-- Define the condition of two circles being externally tangent
def externally_tangent_circles (r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 * r^2

-- Define the condition of a circle being internally tangent to the ellipse
def internally_tangent_to_ellipse (r : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse_equation x y ∧ (x - r)^2 + y^2 = r^2

-- State the theorem
theorem circle_radius_in_ellipse (r : ℝ) :
  externally_tangent_circles r →
  internally_tangent_to_ellipse r →
  r = Real.sqrt 10 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_radius_in_ellipse_l1541_154137


namespace NUMINAMATH_CALUDE_wax_needed_l1541_154139

theorem wax_needed (current_wax total_wax_required : ℕ) 
  (h1 : current_wax = 11)
  (h2 : total_wax_required = 492) : 
  total_wax_required - current_wax = 481 :=
by sorry

end NUMINAMATH_CALUDE_wax_needed_l1541_154139


namespace NUMINAMATH_CALUDE_ninth_term_is_17_l1541_154176

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum_property : a 3 + a 4 = 12
  diff_property : ∀ n, a (n + 1) - a n = d
  d_value : d = 2

/-- The 9th term of the arithmetic sequence is 17 -/
theorem ninth_term_is_17 (seq : ArithmeticSequence) : seq.a 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_17_l1541_154176


namespace NUMINAMATH_CALUDE_bales_stacked_l1541_154130

theorem bales_stacked (initial_bales current_bales : ℕ) 
  (h1 : initial_bales = 54)
  (h2 : current_bales = 82) :
  current_bales - initial_bales = 28 := by
  sorry

end NUMINAMATH_CALUDE_bales_stacked_l1541_154130


namespace NUMINAMATH_CALUDE_jane_pens_after_month_l1541_154154

def alex_pens (week : ℕ) : ℕ := 4 * 2^week

def jane_pens : ℕ := alex_pens 3 - 16

theorem jane_pens_after_month : jane_pens = 16 := by
  sorry

end NUMINAMATH_CALUDE_jane_pens_after_month_l1541_154154


namespace NUMINAMATH_CALUDE_sum_of_roots_l1541_154185

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 14*p*x - 15*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 14*r*x - 15*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 3150 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1541_154185


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1541_154116

def U : Set ℕ := {0, 1, 3, 5, 6, 8}
def A : Set ℕ := {1, 5, 8}
def B : Set ℕ := {2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1541_154116


namespace NUMINAMATH_CALUDE_subset_iff_m_range_l1541_154122

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

theorem subset_iff_m_range (m : ℝ) : B m ⊆ A ↔ m ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_m_range_l1541_154122


namespace NUMINAMATH_CALUDE_max_value_expression_l1541_154128

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, 3 * (a - x) * (2 * x + Real.sqrt (x^2 + 4 * b^2)) ≤ 3 * a^2 + 12 * b^2) ∧
  (∃ x : ℝ, 3 * (a - x) * (2 * x + Real.sqrt (x^2 + 4 * b^2)) = 3 * a^2 + 12 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1541_154128


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l1541_154175

theorem reciprocal_equation_solution (x : ℝ) : 
  (2 - 1 / (2 - x) = 1 / (2 - x)) → x = 1 := by sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l1541_154175


namespace NUMINAMATH_CALUDE_clapping_theorem_l1541_154182

/-- Represents the clapping pattern of a person -/
structure ClappingPattern where
  interval : ℕ
  start_time : ℕ

/-- Checks if a clapping pattern results in a clap at the given time -/
def claps_at (pattern : ClappingPattern) (time : ℕ) : Prop :=
  ∃ k : ℕ, time = pattern.start_time + k * pattern.interval

theorem clapping_theorem (jirka_start petr_start : ℕ) :
  jirka_start ≤ 15 ∧ petr_start ≤ 15 ∧
  claps_at { interval := 7, start_time := jirka_start } 90 ∧
  claps_at { interval := 13, start_time := petr_start } 90 →
  (jirka_start = 6 ∨ jirka_start = 13) ∧ petr_start = 12 := by
  sorry

#check clapping_theorem

end NUMINAMATH_CALUDE_clapping_theorem_l1541_154182


namespace NUMINAMATH_CALUDE_pauls_crayons_l1541_154147

theorem pauls_crayons (crayons_given : ℕ) (crayons_lost : ℕ) (crayons_left : ℕ)
  (h1 : crayons_given = 563)
  (h2 : crayons_lost = 558)
  (h3 : crayons_left = 332) :
  crayons_given + crayons_lost + crayons_left = 1453 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_l1541_154147


namespace NUMINAMATH_CALUDE_tangent_line_implies_m_eq_two_l1541_154140

/-- A circle defined by parametric equations with parameter m > 0 -/
structure ParametricCircle (m : ℝ) where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ φ, x φ = Real.sqrt m * Real.cos φ
  h_y : ∀ φ, y φ = Real.sqrt m * Real.sin φ
  h_m : m > 0

/-- The line x + y = m is tangent to the circle -/
def isTangent (m : ℝ) (circle : ParametricCircle m) : Prop :=
  ∃ φ, circle.x φ + circle.y φ = m ∧
    ∀ ψ, circle.x ψ + circle.y ψ ≤ m

theorem tangent_line_implies_m_eq_two (m : ℝ) (circle : ParametricCircle m)
    (h_tangent : isTangent m circle) : m = 2 := by
  sorry

#check tangent_line_implies_m_eq_two

end NUMINAMATH_CALUDE_tangent_line_implies_m_eq_two_l1541_154140


namespace NUMINAMATH_CALUDE_jo_kate_difference_l1541_154167

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_ten (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

def kate_sum (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_ten |>.sum

theorem jo_kate_difference :
  kate_sum 100 - sum_of_first_n 100 = 500 := by
  sorry

end NUMINAMATH_CALUDE_jo_kate_difference_l1541_154167


namespace NUMINAMATH_CALUDE_obrien_current_hats_l1541_154164

/-- The number of hats Policeman O'Brien has after theft -/
def obrien_hats_after_theft (simpson_hats : ℕ) (stolen_hats : ℕ) : ℕ :=
  2 * simpson_hats + 5 - stolen_hats

/-- Theorem stating the number of hats Policeman O'Brien has after theft -/
theorem obrien_current_hats (simpson_hats stolen_hats : ℕ) 
  (h1 : simpson_hats = 15) :
  obrien_hats_after_theft simpson_hats stolen_hats = 35 - stolen_hats := by
  sorry

#check obrien_current_hats

end NUMINAMATH_CALUDE_obrien_current_hats_l1541_154164


namespace NUMINAMATH_CALUDE_distance_to_market_is_40_l1541_154174

/-- The distance between Andy's house and the market -/
def distance_to_market (distance_to_school : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance - 2 * distance_to_school

/-- Theorem: Given the conditions, the distance to the market is 40 meters -/
theorem distance_to_market_is_40 :
  let distance_to_school : ℕ := 50
  let total_distance : ℕ := 140
  distance_to_market distance_to_school total_distance = 40 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_market_is_40_l1541_154174


namespace NUMINAMATH_CALUDE_base4_1302_equals_base5_424_l1541_154100

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 5) ((m % 5) :: acc)
    go n []

theorem base4_1302_equals_base5_424 :
  base10ToBase5 (base4ToBase10 [2, 0, 3, 1]) = [4, 2, 4] := by
  sorry

end NUMINAMATH_CALUDE_base4_1302_equals_base5_424_l1541_154100
