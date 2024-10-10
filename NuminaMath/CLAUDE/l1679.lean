import Mathlib

namespace age_ratio_problem_l1679_167936

theorem age_ratio_problem (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 22 →
  b = 8 →
  b / c = 2 := by
sorry

end age_ratio_problem_l1679_167936


namespace hyperbola_eccentricity_l1679_167916

/-- Hyperbola with given properties has eccentricity √3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let f (x y : ℝ) := x^2 / a^2 - y^2 / b^2
  let l (x : ℝ) := Real.sqrt 3 / 3 * (x + c)
  c^2 = a^2 + b^2 →
  f c ((2 * Real.sqrt 3 * c) / 3) = 1 →
  l (-c) = 0 →
  l 0 = l c / 2 →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 3 := by
sorry

end hyperbola_eccentricity_l1679_167916


namespace largest_multiple_of_seven_below_negative_85_l1679_167931

theorem largest_multiple_of_seven_below_negative_85 :
  ∀ n : ℤ, n % 7 = 0 ∧ n < -85 → n ≤ -91 :=
by
  sorry

end largest_multiple_of_seven_below_negative_85_l1679_167931


namespace quadratic_roots_imply_a_negative_curve_intersection_not_one_l1679_167940

/-- Represents a quadratic equation of the form x^2 + (a-3)x + a = 0 -/
def QuadraticEquation (a : ℝ) := λ x : ℝ => x^2 + (a-3)*x + a

/-- Represents the curve y = |3-x^2| -/
def Curve := λ x : ℝ => |3 - x^2|

theorem quadratic_roots_imply_a_negative (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ QuadraticEquation a x = 0 ∧ QuadraticEquation a y = 0) →
  a < 0 :=
sorry

theorem curve_intersection_not_one (a : ℝ) :
  ¬(∃! x : ℝ, Curve x = a) :=
sorry

end quadratic_roots_imply_a_negative_curve_intersection_not_one_l1679_167940


namespace pascal_triangle_distinct_elements_l1679_167913

theorem pascal_triangle_distinct_elements :
  ∃ (n : ℕ) (k l m p : ℕ),
    0 < k ∧ k < l ∧ l < m ∧ m < p ∧ p < n ∧
    2 * (n.choose k) = n.choose l ∧
    2 * (n.choose m) = n.choose p ∧
    (n.choose k) ≠ (n.choose l) ∧
    (n.choose k) ≠ (n.choose m) ∧
    (n.choose k) ≠ (n.choose p) ∧
    (n.choose l) ≠ (n.choose m) ∧
    (n.choose l) ≠ (n.choose p) ∧
    (n.choose m) ≠ (n.choose p) := by
  sorry

end pascal_triangle_distinct_elements_l1679_167913


namespace goose_egg_hatching_rate_l1679_167903

theorem goose_egg_hatching_rate : 
  ∀ (total_eggs : ℕ) (hatched_eggs : ℕ),
    (hatched_eggs : ℚ) / total_eggs = 1 →
    (3 : ℚ) / 4 * ((2 : ℚ) / 5 * hatched_eggs) = 180 →
    hatched_eggs ≤ total_eggs →
    (hatched_eggs : ℚ) / total_eggs = 1 := by
  sorry

end goose_egg_hatching_rate_l1679_167903


namespace chocolate_lollipop_cost_equivalence_l1679_167909

/-- Proves that the cost of one pack of chocolate equals the cost of 4 lollipops -/
theorem chocolate_lollipop_cost_equivalence 
  (lollipop_count : ℕ) 
  (chocolate_pack_count : ℕ)
  (lollipop_cost : ℕ)
  (bills_given : ℕ)
  (bill_value : ℕ)
  (change_received : ℕ)
  (h1 : lollipop_count = 4)
  (h2 : chocolate_pack_count = 6)
  (h3 : lollipop_cost = 2)
  (h4 : bills_given = 6)
  (h5 : bill_value = 10)
  (h6 : change_received = 4) :
  (bills_given * bill_value - change_received - lollipop_count * lollipop_cost) / chocolate_pack_count = 4 * lollipop_cost :=
by sorry

end chocolate_lollipop_cost_equivalence_l1679_167909


namespace joshua_shares_with_five_friends_l1679_167930

/-- The number of Skittles Joshua has -/
def total_skittles : ℕ := 40

/-- The number of Skittles each friend receives -/
def skittles_per_friend : ℕ := 8

/-- The number of friends Joshua shares his Skittles with -/
def number_of_friends : ℕ := total_skittles / skittles_per_friend

theorem joshua_shares_with_five_friends :
  number_of_friends = 5 :=
by sorry

end joshua_shares_with_five_friends_l1679_167930


namespace area_of_triangle_abc_is_150_over_7_l1679_167972

/-- Given a circle with center O and radius r, and points A and B on a line passing through O,
    this function calculates the area of triangle ABC, where C is the intersection of tangents
    drawn from A and B to the circle. -/
def triangle_area_from_tangents (r OA AB : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for a circle with radius 12 and points A and B such that OA = 15 and AB = 5,
    the area of triangle ABC formed by the intersection of tangents is 150/7. -/
theorem area_of_triangle_abc_is_150_over_7 :
  triangle_area_from_tangents 12 15 5 = 150 / 7 := by
  sorry

end area_of_triangle_abc_is_150_over_7_l1679_167972


namespace race_course_length_60m_l1679_167939

/-- Represents the race scenario with three runners -/
structure RaceScenario where
  speedB : ℝ     -- Speed of runner B (base speed)
  speedA : ℝ     -- Speed of runner A
  speedC : ℝ     -- Speed of runner C
  headStartA : ℝ -- Head start given by A to B
  headStartC : ℝ -- Head start given by C to B

/-- Calculates the race course length for simultaneous finish -/
def calculateRaceCourseLength (race : RaceScenario) : ℝ :=
  sorry

/-- Theorem stating the race course length for the given scenario -/
theorem race_course_length_60m :
  ∀ (v : ℝ), v > 0 →
  let race : RaceScenario :=
    { speedB := v
      speedA := 4 * v
      speedC := 2 * v
      headStartA := 60
      headStartC := 30 }
  calculateRaceCourseLength race = 60 :=
sorry

end race_course_length_60m_l1679_167939


namespace fifth_month_sale_l1679_167962

-- Define the sales for the first four months
def first_four_sales : List Int := [5420, 5660, 6200, 6350]

-- Define the sale for the sixth month
def sixth_month_sale : Int := 7070

-- Define the average sale for six months
def average_sale : Int := 6200

-- Define the number of months
def num_months : Int := 6

-- Theorem to prove
theorem fifth_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := first_four_sales.sum + sixth_month_sale
  total_sales - known_sales = 6500 := by
  sorry

end fifth_month_sale_l1679_167962


namespace oarsmen_count_l1679_167951

theorem oarsmen_count (weight_old : ℝ) (weight_new : ℝ) (avg_increase : ℝ) :
  weight_old = 53 →
  weight_new = 71 →
  avg_increase = 1.8 →
  ∃ n : ℕ, n > 0 ∧ n * avg_increase = weight_new - weight_old :=
by
  sorry

end oarsmen_count_l1679_167951


namespace complex_pattern_cannot_be_formed_l1679_167912

-- Define the types of shapes
inductive Shape
| Triangle
| Square

-- Define the set of available pieces
def available_pieces : Multiset Shape :=
  Multiset.replicate 8 Shape.Triangle + Multiset.replicate 7 Shape.Square

-- Define the possible figures
inductive Figure
| LargeRectangle
| Triangle
| Square
| ComplexPattern
| LongNarrowRectangle

-- Define a function to check if a figure can be formed
def can_form_figure (pieces : Multiset Shape) (figure : Figure) : Prop :=
  match figure with
  | Figure.LargeRectangle => true
  | Figure.Triangle => true
  | Figure.Square => true
  | Figure.ComplexPattern => false
  | Figure.LongNarrowRectangle => true

-- Theorem statement
theorem complex_pattern_cannot_be_formed :
  ∀ (figure : Figure),
    figure ≠ Figure.ComplexPattern ↔ can_form_figure available_pieces figure :=
by sorry

end complex_pattern_cannot_be_formed_l1679_167912


namespace least_integer_greater_than_two_plus_sqrt_three_squared_l1679_167904

theorem least_integer_greater_than_two_plus_sqrt_three_squared :
  ∃ n : ℤ, (n = 14 ∧ (2 + Real.sqrt 3)^2 < n ∧ ∀ m : ℤ, (2 + Real.sqrt 3)^2 < m → n ≤ m) :=
sorry

end least_integer_greater_than_two_plus_sqrt_three_squared_l1679_167904


namespace triangle_area_product_l1679_167989

theorem triangle_area_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (1/2) * (12/a) * (12/b) = 12) : a * b = 6 := by
  sorry

end triangle_area_product_l1679_167989


namespace mean_equality_implies_z_value_l1679_167938

theorem mean_equality_implies_z_value : ∃ z : ℚ, 
  (8 + 15 + 27) / 3 = (18 + z) / 2 → z = 46 / 3 := by
  sorry

end mean_equality_implies_z_value_l1679_167938


namespace negation_of_cosine_inequality_l1679_167937

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos (2 * x) ≤ Real.cos x ^ 2) ↔
  (∃ x : ℝ, Real.cos (2 * x) > Real.cos x ^ 2) := by sorry

end negation_of_cosine_inequality_l1679_167937


namespace logical_equivalence_l1679_167908

theorem logical_equivalence (P Q : Prop) :
  (¬P → Q) ↔ (¬Q → P) := by sorry

end logical_equivalence_l1679_167908


namespace min_product_sum_l1679_167924

theorem min_product_sum (a b c d : ℕ) : 
  a ∈ ({1, 3, 5, 7} : Set ℕ) → 
  b ∈ ({1, 3, 5, 7} : Set ℕ) → 
  c ∈ ({1, 3, 5, 7} : Set ℕ) → 
  d ∈ ({1, 3, 5, 7} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  48 ≤ a * b + b * c + c * d + d * a :=
by sorry

end min_product_sum_l1679_167924


namespace final_result_calculation_l1679_167928

theorem final_result_calculation (chosen_number : ℕ) : 
  chosen_number = 60 → (chosen_number * 4 - 138 = 102) := by
  sorry

end final_result_calculation_l1679_167928


namespace event_properties_l1679_167963

-- Define the types of events
inductive Event
| Train
| Shooting

-- Define the type of outcomes
inductive Outcome
| Success
| Failure

-- Define a function to get the number of trials for each event
def num_trials (e : Event) : ℕ :=
  match e with
  | Event.Train => 3
  | Event.Shooting => 2

-- Define a function to get the possible outcomes for each event
def possible_outcomes (e : Event) : List Outcome :=
  [Outcome.Success, Outcome.Failure]

-- Theorem statement
theorem event_properties :
  (∀ e : Event, num_trials e > 0) ∧
  (∀ e : Event, possible_outcomes e = [Outcome.Success, Outcome.Failure]) :=
by sorry

end event_properties_l1679_167963


namespace monotone_decreasing_implies_a_positive_l1679_167925

/-- The function f(x) = a(x^3 - x) is monotonically decreasing on the interval (-√3/3, √3/3) -/
def is_monotone_decreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, -Real.sqrt 3 / 3 < x ∧ x < y ∧ y < Real.sqrt 3 / 3 → f x > f y

/-- The definition of the function f(x) = a(x^3 - x) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x^3 - x)

theorem monotone_decreasing_implies_a_positive (a : ℝ) :
  is_monotone_decreasing (f a) a → a > 0 := by
  sorry

end monotone_decreasing_implies_a_positive_l1679_167925


namespace glued_polyhedron_edge_length_l1679_167992

/-- A polyhedron formed by gluing a square-based pyramid to a regular tetrahedron -/
structure GluedPolyhedron where
  -- Square-based pyramid
  pyramid_edge_length : ℝ
  pyramid_edge_count : ℕ
  -- Regular tetrahedron
  tetrahedron_edge_length : ℝ
  tetrahedron_edge_count : ℕ
  -- Gluing properties
  glued_edges : ℕ
  merged_edges : ℕ
  -- Conditions
  pyramid_square_base : pyramid_edge_count = 8
  all_edges_length_2 : pyramid_edge_length = 2 ∧ tetrahedron_edge_length = 2
  tetrahedron_regular : tetrahedron_edge_count = 6
  glued_face_edges : glued_edges = 3
  merged_parallel_edges : merged_edges = 2

/-- The total edge length of the glued polyhedron -/
def totalEdgeLength (p : GluedPolyhedron) : ℝ :=
  (p.pyramid_edge_count + p.tetrahedron_edge_count - p.glued_edges - p.merged_edges) * p.pyramid_edge_length

/-- Theorem stating that the total edge length of the glued polyhedron is 18 -/
theorem glued_polyhedron_edge_length (p : GluedPolyhedron) : totalEdgeLength p = 18 := by
  sorry

end glued_polyhedron_edge_length_l1679_167992


namespace correct_divisor_l1679_167915

theorem correct_divisor (X D : ℕ) (h1 : X % D = 0) (h2 : X = 49 * 12) (h3 : X = 28 * D) : D = 21 := by
  sorry

end correct_divisor_l1679_167915


namespace quadratic_always_positive_l1679_167998

theorem quadratic_always_positive (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) → 0 ≤ a ∧ a < 3 :=
by sorry

end quadratic_always_positive_l1679_167998


namespace special_set_bounds_l1679_167994

/-- A set of points in 3D space satisfying the given conditions -/
def SpecialSet (n : ℕ) (S : Set (ℝ × ℝ × ℝ)) : Prop :=
  (n > 0) ∧ 
  (∀ (planes : Finset (Set (ℝ × ℝ × ℝ))), planes.card = n → 
    ∃ (p : ℝ × ℝ × ℝ), p ∈ S ∧ ∀ (plane : Set (ℝ × ℝ × ℝ)), plane ∈ planes → p ∉ plane) ∧
  (∀ (X : ℝ × ℝ × ℝ), X ∈ S → 
    ∃ (planes : Finset (Set (ℝ × ℝ × ℝ))), planes.card = n ∧ 
      ∀ (Y : ℝ × ℝ × ℝ), Y ∈ S \ {X} → ∃ (plane : Set (ℝ × ℝ × ℝ)), plane ∈ planes ∧ Y ∈ plane)

theorem special_set_bounds (n : ℕ) (S : Set (ℝ × ℝ × ℝ)) (h : SpecialSet n S) :
  (3 * n + 1 : ℕ) ≤ S.ncard ∧ S.ncard ≤ Nat.choose (n + 3) 3 := by
  sorry

end special_set_bounds_l1679_167994


namespace jogger_distance_ahead_l1679_167941

/-- Proves that a jogger is 270 meters ahead of a train given specific conditions -/
theorem jogger_distance_ahead (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5 / 18) →  -- Convert 9 km/hr to m/s
  train_speed = 45 * (5 / 18) →  -- Convert 45 km/hr to m/s
  train_length = 120 →
  passing_time = 39 →
  (train_speed - jogger_speed) * passing_time = train_length + 270 :=
by sorry

end jogger_distance_ahead_l1679_167941


namespace count_odd_numbers_l1679_167933

def digits : Finset Nat := {0, 1, 2, 3, 4}

def is_odd (n : Nat) : Bool := n % 2 = 1

def is_valid_number (n : Nat) : Bool :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100) ∈ digits ∧ ((n / 10) % 10) ∈ digits ∧ (n % 10) ∈ digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧ (n / 100) ≠ (n % 10) ∧ ((n / 10) % 10) ≠ (n % 10)

theorem count_odd_numbers :
  (Finset.filter (λ n => is_valid_number n ∧ is_odd n) (Finset.range 1000)).card = 18 := by
  sorry

end count_odd_numbers_l1679_167933


namespace largest_value_is_B_l1679_167921

theorem largest_value_is_B (a b c e : ℚ) : 
  a = (1/2) / (3/4) →
  b = 1 / ((2/3) / 4) →
  c = ((1/2) / 3) / 4 →
  e = (1 / (2/3)) / 4 →
  b > a ∧ b > c ∧ b > e :=
by sorry

end largest_value_is_B_l1679_167921


namespace min_value_theorem_max_value_theorem_l1679_167969

-- Problem 1
theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x + 4 / x + 5 ≥ 9 :=
sorry

-- Problem 2
theorem max_value_theorem (x : ℝ) (h1 : x > 0) (h2 : x < 1/2) :
  1/2 * x * (1 - 2*x) ≤ 1/16 :=
sorry

end min_value_theorem_max_value_theorem_l1679_167969


namespace quadratic_equivalences_l1679_167974

theorem quadratic_equivalences (x : ℝ) : 
  (((x ≠ 1 ∧ x ≠ 2) → x^2 - 3*x + 2 ≠ 0) ∧
   ((x^2 - 3*x + 2 = 0) → (x = 1 ∨ x = 2)) ∧
   ((x = 1 ∨ x = 2) → x^2 - 3*x + 2 = 0)) := by
  sorry

end quadratic_equivalences_l1679_167974


namespace additional_cars_problem_solution_l1679_167991

theorem additional_cars (front_initial : Nat) (back_initial : Nat) (total_end : Nat) : Nat :=
  let total_initial := front_initial + back_initial
  total_end - total_initial

theorem problem_solution : 
  let front_initial := 100
  let back_initial := 2 * front_initial
  let total_end := 700
  additional_cars front_initial back_initial total_end = 400 := by
sorry

end additional_cars_problem_solution_l1679_167991


namespace line_slope_k_l1679_167910

/-- Given a line passing through the points (-1, -4) and (4, k) with slope k, prove that k = 1 -/
theorem line_slope_k (k : ℝ) : 
  (k - (-4)) / (4 - (-1)) = k → k = 1 := by
  sorry

end line_slope_k_l1679_167910


namespace percentage_difference_l1679_167970

theorem percentage_difference (A B C : ℝ) (hpos : 0 < C ∧ C < B ∧ B < A) :
  let x := 100 * (A - B) / B
  let y := 100 * (A - C) / C
  (∃ k, A = B * (1 + k/100) → x = 100 * (A - B) / B) ∧
  (∃ m, A = C * (1 + m/100) → y = 100 * (A - C) / C) := by
sorry

end percentage_difference_l1679_167970


namespace right_triangle_leg_length_l1679_167983

theorem right_triangle_leg_length
  (hypotenuse : ℝ)
  (leg1 : ℝ)
  (h1 : hypotenuse = 15)
  (h2 : leg1 = 9)
  (h3 : hypotenuse^2 = leg1^2 + leg2^2) :
  leg2 = 12 :=
by
  sorry

end right_triangle_leg_length_l1679_167983


namespace sqrt_two_minus_sqrt_eight_l1679_167988

theorem sqrt_two_minus_sqrt_eight : Real.sqrt 2 - Real.sqrt 8 = -Real.sqrt 2 := by
  sorry

end sqrt_two_minus_sqrt_eight_l1679_167988


namespace event_classification_l1679_167942

-- Define the type for events
inductive Event
| Certain : Event
| Impossible : Event

-- Define a function to classify events
def classify_event (e : Event) : String :=
  match e with
  | Event.Certain => "certain event"
  | Event.Impossible => "impossible event"

-- State the theorem
theorem event_classification :
  (∃ e : Event, e = Event.Certain) ∧ 
  (∃ e : Event, e = Event.Impossible) →
  (classify_event Event.Certain = "certain event") ∧
  (classify_event Event.Impossible = "impossible event") := by
  sorry

end event_classification_l1679_167942


namespace u_n_eq_2n_minus_1_l1679_167986

/-- 
Given a positive integer n, u_n is the smallest positive integer such that 
for any odd integer d, the number of integers in any u_n consecutive odd integers 
that are divisible by d is at least as many as the number of integers among 
1, 3, 5, ..., 2n-1 that are divisible by d.
-/
def u_n (n : ℕ+) : ℕ := sorry

/-- The main theorem stating that u_n is equal to 2n - 1 -/
theorem u_n_eq_2n_minus_1 (n : ℕ+) : u_n n = 2 * n - 1 := by sorry

end u_n_eq_2n_minus_1_l1679_167986


namespace is_focus_of_hyperbola_l1679_167995

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - 3*y^2 + 6*x - 12*y - 8 = 0

/-- The focus point -/
def focus : ℝ × ℝ := (-1, -2)

/-- Theorem stating that the given point is a focus of the hyperbola -/
theorem is_focus_of_hyperbola : 
  ∃ (c : ℝ), c > 0 ∧ 
  ∀ (x y : ℝ), hyperbola_equation x y → 
    (x + 1)^2 + (y + 2)^2 - ((x + 5)^2 + (y + 2)^2) = 4*c := by
  sorry

end is_focus_of_hyperbola_l1679_167995


namespace pi_approximation_l1679_167947

theorem pi_approximation (S : ℝ) (h : S > 0) :
  4 * S = (1 + 1/4) * (π * S) → π = 3 := by
sorry

end pi_approximation_l1679_167947


namespace inequality_system_solution_range_l1679_167996

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ x : ℝ, ax < 1 ∧ x - a < 0) → a ∈ Set.Ici (-1) :=
by
  sorry

end inequality_system_solution_range_l1679_167996


namespace problem_solution_l1679_167961

theorem problem_solution : ∀ A B Y : ℤ,
  A = 3009 / 3 →
  B = A / 3 →
  Y = A - B →
  Y = 669 := by
sorry

end problem_solution_l1679_167961


namespace ab_negative_necessary_not_sufficient_l1679_167926

-- Define what it means for an equation to represent a hyperbola
def represents_hyperbola (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, a * x^2 + b * y^2 = c ∧ 
  ((a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) ∧ 
  c ≠ 0

-- State the theorem
theorem ab_negative_necessary_not_sufficient :
  (∀ a b c : ℝ, represents_hyperbola a b c → a * b < 0) ∧
  ¬(∀ a b c : ℝ, a * b < 0 → represents_hyperbola a b c) :=
by sorry

end ab_negative_necessary_not_sufficient_l1679_167926


namespace second_smallest_five_digit_in_pascal_l1679_167978

/-- Pascal's triangle function -/
def pascal (n k : ℕ) : ℕ := sorry

/-- Predicate to check if a number is in Pascal's triangle -/
def inPascalTriangle (x : ℕ) : Prop :=
  ∃ n k : ℕ, pascal n k = x

/-- Predicate to check if a number is a five-digit number -/
def isFiveDigit (x : ℕ) : Prop :=
  10000 ≤ x ∧ x ≤ 99999

/-- The second smallest five-digit number in Pascal's triangle is 10001 -/
theorem second_smallest_five_digit_in_pascal :
  ∃! x : ℕ, inPascalTriangle x ∧ isFiveDigit x ∧
  (∃! y : ℕ, y < x ∧ inPascalTriangle y ∧ isFiveDigit y) ∧
  x = 10001 := by sorry

end second_smallest_five_digit_in_pascal_l1679_167978


namespace rectangle_area_l1679_167929

/-- The area of a rectangle with length 0.4 meters and width 0.22 meters is 0.088 square meters. -/
theorem rectangle_area : 
  let length : ℝ := 0.4
  let width : ℝ := 0.22
  length * width = 0.088 := by sorry

end rectangle_area_l1679_167929


namespace instantaneous_velocity_at_one_second_l1679_167979

-- Define the height function
def h (t : ℝ) : ℝ := -4.9 * t^2 + 4.8 * t + 11

-- Define the velocity function as the derivative of the height function
def v (t : ℝ) : ℝ := -9.8 * t + 4.8

-- Theorem statement
theorem instantaneous_velocity_at_one_second :
  v 1 = -5 := by sorry

end instantaneous_velocity_at_one_second_l1679_167979


namespace tricycle_count_l1679_167975

/-- Represents the number of wheels for each vehicle type -/
def wheels_per_vehicle : Fin 3 → ℕ
  | 0 => 2  -- bicycles
  | 1 => 3  -- tricycles
  | 2 => 2  -- scooters

/-- Proves that the number of tricycles is 4 given the conditions of the parade -/
theorem tricycle_count (vehicles : Fin 3 → ℕ) 
  (total_children : vehicles 0 + vehicles 1 + vehicles 2 = 10)
  (total_wheels : vehicles 0 * wheels_per_vehicle 0 + 
                  vehicles 1 * wheels_per_vehicle 1 + 
                  vehicles 2 * wheels_per_vehicle 2 = 27) :
  vehicles 1 = 4 := by
  sorry

end tricycle_count_l1679_167975


namespace promotion_b_saves_more_l1679_167959

/-- The cost of a single shirt in dollars -/
def shirtCost : ℝ := 40

/-- The cost of two shirts under Promotion A -/
def promotionACost : ℝ := shirtCost + (shirtCost * 0.75)

/-- The cost of two shirts under Promotion B -/
def promotionBCost : ℝ := shirtCost + (shirtCost - 15)

/-- Theorem stating that Promotion B costs $5 less than Promotion A -/
theorem promotion_b_saves_more :
  promotionACost - promotionBCost = 5 := by
  sorry

end promotion_b_saves_more_l1679_167959


namespace inequality_theta_range_l1679_167999

theorem inequality_theta_range (θ : Real) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔ 
  ∃ k : ℤ, θ ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 12) (2 * k * Real.pi + 5 * Real.pi / 12) :=
by sorry

end inequality_theta_range_l1679_167999


namespace equation_solutions_l1679_167918

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 12*x - 9) + 1 / (x^2 + 3*x - 9) + 1 / (x^2 - 14*x - 9) = 0)} = {1, -9, 3, -3} := by
  sorry

end equation_solutions_l1679_167918


namespace statements_with_nonzero_solutions_l1679_167954

theorem statements_with_nonzero_solutions :
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (Real.sqrt (a^2 + b^2) = 3 * (a * b) ∨ Real.sqrt (a^2 + b^2) = 2 * (a * b)) ∧
  ¬(∃ (c d : ℝ), c ≠ 0 ∧ d ≠ 0 ∧
    (Real.sqrt (c^2 + d^2) = 2 * (c + d) ∨ Real.sqrt (c^2 + d^2) = (1/2) * (c + d))) :=
by
  sorry


end statements_with_nonzero_solutions_l1679_167954


namespace distribute_nine_computers_to_three_schools_l1679_167957

/-- The number of ways to distribute computers to schools -/
def distribute_computers (total_computers : ℕ) (num_schools : ℕ) (min_computers : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem: There are 10 ways to distribute 9 computers to 3 schools with at least 2 per school -/
theorem distribute_nine_computers_to_three_schools : 
  distribute_computers 9 3 2 = 10 := by
  sorry

end distribute_nine_computers_to_three_schools_l1679_167957


namespace root_values_l1679_167946

theorem root_values (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^3 + q * m^2 + r * m + s = 0)
  (h2 : q * m^3 + r * m^2 + (s + m * m) * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I :=
by sorry

end root_values_l1679_167946


namespace gcd_problem_l1679_167980

theorem gcd_problem (b : ℤ) (h : 504 ∣ b) : 
  Nat.gcd (4*b^3 + 2*b^2 + 5*b + 63).natAbs b.natAbs = 63 := by
  sorry

end gcd_problem_l1679_167980


namespace school_math_survey_l1679_167964

theorem school_math_survey (total : ℝ) (math_likers : ℝ) (olympiad_participants : ℝ)
  (h1 : math_likers ≤ total)
  (h2 : olympiad_participants = math_likers + 0.1 * (total - math_likers))
  (h3 : olympiad_participants = 0.46 * total) :
  math_likers = 0.4 * total :=
by sorry

end school_math_survey_l1679_167964


namespace complement_intersection_theorem_l1679_167966

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | (p.2 + 2) / (p.1 - 2) = 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 - 4}

-- Statement to prove
theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {(2, -2)} := by sorry

end complement_intersection_theorem_l1679_167966


namespace total_students_in_high_school_l1679_167911

/-- Proves that the total number of students in a high school is 500 given the number of students in different course combinations. -/
theorem total_students_in_high_school : 
  ∀ (music art both neither : ℕ),
    music = 30 →
    art = 10 →
    both = 10 →
    neither = 470 →
    music + art - both + neither = 500 :=
by
  sorry

end total_students_in_high_school_l1679_167911


namespace two_tails_in_seven_flips_l1679_167967

def unfair_coin_flip (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem two_tails_in_seven_flips :
  unfair_coin_flip 7 2 (3/4) = 189/16384 := by
  sorry

end two_tails_in_seven_flips_l1679_167967


namespace triangle_side_length_l1679_167950

theorem triangle_side_length (AB : ℝ) (cosA sinC : ℝ) (angleADB : ℝ) :
  AB = 30 →
  angleADB = 90 →
  cosA = 4/5 →
  sinC = 2/5 →
  ∃ (AD : ℝ), AD = 24 := by
  sorry

end triangle_side_length_l1679_167950


namespace min_balls_to_guarantee_fifteen_l1679_167934

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee at least n of a single color -/
def minBallsToGuarantee (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem min_balls_to_guarantee_fifteen (counts : BallCounts)
  (h_red : counts.red = 28)
  (h_green : counts.green = 20)
  (h_yellow : counts.yellow = 19)
  (h_blue : counts.blue = 13)
  (h_white : counts.white = 11)
  (h_black : counts.black = 9) :
  minBallsToGuarantee counts 15 = 76 := by
  sorry

end min_balls_to_guarantee_fifteen_l1679_167934


namespace volume_for_weight_less_than_112_l1679_167944

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℝ
  /-- Assumption that k is positive -/
  k_pos : k > 0

/-- The volume of the substance given its weight -/
def volume (s : Substance) (weight : ℝ) : ℝ := s.k * weight

theorem volume_for_weight_less_than_112 (s : Substance) (weight : ℝ) 
  (h1 : volume s 112 = 48) (h2 : 0 < weight) (h3 : weight < 112) :
  volume s weight = (48 / 112) * weight := by
sorry

end volume_for_weight_less_than_112_l1679_167944


namespace ellipse_slope_l1679_167945

/-- Given an ellipse with eccentricity √3/2 and a point P on the ellipse such that
    the sum of tangents of angles formed by PA and PB with the x-axis is 1,
    prove that the slope of PA is (1 ± √2)/2. -/
theorem ellipse_slope (a b : ℝ) (x y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  let e := Real.sqrt 3 / 2
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let A := (-a, 0)
  let B := (a, 0)
  let P := (x, y)
  ∀ (α β : ℝ),
    e = Real.sqrt (a^2 - b^2) / a →
    P ∈ C →
    (y / (x + a)) + (y / (x - a)) = 1 →
    (∃ (k : ℝ), k = y / (x + a) ∧ (k = (1 + Real.sqrt 2) / 2 ∨ k = (1 - Real.sqrt 2) / 2)) :=
by
  sorry


end ellipse_slope_l1679_167945


namespace matrix_inverse_proof_l1679_167948

theorem matrix_inverse_proof :
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![4, 2.5, 0; 3, 2, 0; 0, 0, 1]
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![-4, 5, 0; 6, -8, 0; 0, 0, 1]
  N * A = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end matrix_inverse_proof_l1679_167948


namespace arithmetic_computation_l1679_167997

theorem arithmetic_computation : -10 * 5 - (-8 * -4) + (-12 * -6) + 2 * 7 = 4 := by
  sorry

end arithmetic_computation_l1679_167997


namespace product_from_lcm_hcf_l1679_167927

theorem product_from_lcm_hcf (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 600) 
  (h_hcf : Nat.gcd a b = 30) : 
  a * b = 18000 := by
  sorry

end product_from_lcm_hcf_l1679_167927


namespace product_of_primes_in_equation_l1679_167906

theorem product_of_primes_in_equation (p q : ℕ) : 
  Prime p → Prime q → p * 1 + q = 99 → p * q = 194 := by
  sorry

end product_of_primes_in_equation_l1679_167906


namespace geometric_sequence_seventh_term_l1679_167960

theorem geometric_sequence_seventh_term
  (a₁ : ℝ)
  (a₁₀ : ℝ)
  (h₁ : a₁ = 12)
  (h₂ : a₁₀ = 78732)
  (h₃ : ∀ n : ℕ, 1 ≤ n → n ≤ 10 → ∃ r : ℝ, a₁ * r^(n-1) = a₁₀^((n-1)/9) * a₁^(1-(n-1)/9)) :
  ∃ a₇ : ℝ, a₇ = 8748 ∧ a₁ * (a₁₀ / a₁)^(6/9) = a₇ :=
by
  sorry

end geometric_sequence_seventh_term_l1679_167960


namespace simplify_cube_root_l1679_167949

theorem simplify_cube_root : 
  (20^3 + 30^3 + 40^3 + 60^3 : ℝ)^(1/3) = 10 * 315^(1/3) := by
  sorry

end simplify_cube_root_l1679_167949


namespace sugar_amount_l1679_167982

/-- Represents the amounts of ingredients in pounds -/
structure Ingredients where
  sugar : ℝ
  flour : ℝ
  baking_soda : ℝ

/-- The ratios and conditions given in the problem -/
def satisfies_conditions (i : Ingredients) : Prop :=
  i.sugar / i.flour = 3 / 8 ∧
  i.flour / i.baking_soda = 10 ∧
  i.flour / (i.baking_soda + 60) = 8

/-- The theorem stating that under the given conditions, the amount of sugar is 900 pounds -/
theorem sugar_amount (i : Ingredients) :
  satisfies_conditions i → i.sugar = 900 := by
  sorry

end sugar_amount_l1679_167982


namespace complement_of_37_45_l1679_167902

-- Define angle in degrees and minutes
structure AngleDM where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

-- Define the complement of an angle
def complement (α : AngleDM) : AngleDM :=
  let totalMinutes := (90 * 60) - (α.degrees * 60 + α.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60, by sorry⟩

theorem complement_of_37_45 :
  let α : AngleDM := ⟨37, 45, by sorry⟩
  complement α = ⟨52, 15, by sorry⟩ := by
  sorry

end complement_of_37_45_l1679_167902


namespace rectangle_area_l1679_167971

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 166) : L * B = 1590 := by
  sorry

end rectangle_area_l1679_167971


namespace isosceles_triangle_rational_trig_l1679_167993

/-- An isosceles triangle with integer base and height has rational sine and cosine of vertex angle -/
theorem isosceles_triangle_rational_trig (BC AD : ℤ) (h : BC > 0 ∧ AD > 0) : 
  ∃ (sinA cosA : ℚ), 
    sinA = Real.sin (Real.arccos ((BC * BC - 2 * AD * AD) / (BC * BC + 2 * AD * AD))) ∧
    cosA = (BC * BC - 2 * AD * AD) / (BC * BC + 2 * AD * AD) := by
  sorry

#check isosceles_triangle_rational_trig

end isosceles_triangle_rational_trig_l1679_167993


namespace statement_equivalence_l1679_167923

-- Define the statement as a function that takes a real number y
def statementIsTrue (y : ℝ) : Prop := (1/2 * y + 5) > 0

-- Define the theorem
theorem statement_equivalence :
  ∀ y : ℝ, statementIsTrue y ↔ (1/2 * y + 5 > 0) :=
by
  sorry

end statement_equivalence_l1679_167923


namespace geometric_sequence_sum_ratio_l1679_167977

/-- For a geometric sequence with common ratio 2, the ratio of the sum of the first 3 terms to the first term is 7 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- Geometric sequence with common ratio 2
  (∀ n, S n = (a 1) * (1 - 2^n) / (1 - 2)) →  -- Sum formula for geometric sequence
  S 3 / a 1 = 7 := by
sorry

end geometric_sequence_sum_ratio_l1679_167977


namespace two_million_times_three_million_l1679_167973

theorem two_million_times_three_million : 
  (2 * 1000000) * (3 * 1000000) = 6 * 1000000000000 := by
  sorry

end two_million_times_three_million_l1679_167973


namespace max_value_of_expression_l1679_167919

theorem max_value_of_expression (a b c : ℝ) 
  (ha : -1 < a ∧ a < 1) 
  (hb : -1 < b ∧ b < 1) 
  (hc : -1 < c ∧ c < 1) : 
  1/((1 - a^2)*(1 - b^2)*(1 - c^2)) + 1/((1 + a^2)*(1 + b^2)*(1 + c^2)) ≤ 2 ∧ 
  (1/((1 - 0^2)*(1 - 0^2)*(1 - 0^2)) + 1/((1 + 0^2)*(1 + 0^2)*(1 + 0^2)) = 2) :=
by sorry

end max_value_of_expression_l1679_167919


namespace tan_thirteen_pi_fourth_l1679_167976

theorem tan_thirteen_pi_fourth : Real.tan (13 * π / 4) = -1 := by
  sorry

end tan_thirteen_pi_fourth_l1679_167976


namespace vegetarian_gluten_free_fraction_is_one_twentieth_l1679_167914

/-- Represents a restaurant menu -/
structure Menu :=
  (total_dishes : ℕ)
  (vegetarian_dishes : ℕ)
  (gluten_free_vegetarian_dishes : ℕ)

/-- The fraction of dishes that are both vegetarian and gluten-free -/
def vegetarian_gluten_free_fraction (menu : Menu) : ℚ :=
  menu.gluten_free_vegetarian_dishes / menu.total_dishes

theorem vegetarian_gluten_free_fraction_is_one_twentieth
  (menu : Menu)
  (h1 : menu.vegetarian_dishes = 4)
  (h2 : menu.vegetarian_dishes = menu.total_dishes / 5)
  (h3 : menu.gluten_free_vegetarian_dishes = menu.vegetarian_dishes - 3) :
  vegetarian_gluten_free_fraction menu = 1 / 20 := by
  sorry

end vegetarian_gluten_free_fraction_is_one_twentieth_l1679_167914


namespace inequality_proof_l1679_167985

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) ≥ 8 := by
  sorry

end inequality_proof_l1679_167985


namespace rachel_second_level_treasures_l1679_167990

/-- Represents the video game scoring system and Rachel's performance --/
structure GameScore where
  points_per_treasure : ℕ
  treasures_first_level : ℕ
  total_score : ℕ

/-- Calculates the number of treasures found on the second level --/
def treasures_second_level (game : GameScore) : ℕ :=
  (game.total_score - game.points_per_treasure * game.treasures_first_level) / game.points_per_treasure

/-- Theorem stating that Rachel found 2 treasures on the second level --/
theorem rachel_second_level_treasures :
  let game : GameScore := {
    points_per_treasure := 9,
    treasures_first_level := 5,
    total_score := 63
  }
  treasures_second_level game = 2 := by
  sorry

end rachel_second_level_treasures_l1679_167990


namespace lynn_ogen_interest_l1679_167965

/-- Calculates the total annual interest for Lynn Ogen's investments -/
theorem lynn_ogen_interest (x : ℝ) (h1 : x - 100 = 400) :
  0.09 * x + 0.07 * (x - 100) = 73 := by
  sorry

end lynn_ogen_interest_l1679_167965


namespace largest_divisor_of_consecutive_even_product_l1679_167955

theorem largest_divisor_of_consecutive_even_product : 
  ∃ (d : ℕ), d = 24 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (2*n) * (2*n + 2) * (2*n + 4)) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (2*m) * (2*m + 2) * (2*m + 4))) :=
by sorry

end largest_divisor_of_consecutive_even_product_l1679_167955


namespace min_value_sum_l1679_167968

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 3) + 1 / (b + 3) = 1 / 4) : 
  a + 3 * b ≥ 12 + 16 * Real.sqrt 3 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    1 / (a₀ + 3) + 1 / (b₀ + 3) = 1 / 4 ∧
    a₀ + 3 * b₀ = 12 + 16 * Real.sqrt 3 :=
sorry

end min_value_sum_l1679_167968


namespace unique_non_six_order_l1679_167905

theorem unique_non_six_order (a : ℤ) : 
  (a > 1 ∧ ∀ p : ℕ, Nat.Prime p → ∀ n : ℕ, n > 0 ∧ a^n ≡ 1 [ZMOD p] → n ≠ 6) ↔ a = 2 :=
sorry

end unique_non_six_order_l1679_167905


namespace smallest_proportional_part_l1679_167917

theorem smallest_proportional_part :
  let total : ℕ := 120
  let ratios : List ℕ := [3, 5, 7]
  let parts : List ℕ := ratios.map (λ r => r * (total / ratios.sum))
  parts.minimum? = some 24 := by
  sorry

end smallest_proportional_part_l1679_167917


namespace sqrt_plus_square_zero_l1679_167981

theorem sqrt_plus_square_zero (m n : ℝ) : 
  Real.sqrt (m + 1) + (n - 2)^2 = 0 → m + n = 1 := by
  sorry

end sqrt_plus_square_zero_l1679_167981


namespace fraction_increase_l1679_167907

theorem fraction_increase (x y : ℝ) (h : x + y ≠ 0) :
  (2 * (2 * x) * (2 * y)) / (2 * x + 2 * y) = 2 * ((2 * x * y) / (x + y)) :=
by sorry

end fraction_increase_l1679_167907


namespace equation_solution_l1679_167932

theorem equation_solution : ∃ x : ℝ, 45 - (x - (37 - (15 - 15))) = 54 ∧ x = 28 := by
  sorry

end equation_solution_l1679_167932


namespace correct_calculation_l1679_167953

theorem correct_calculation (x : ℤ) (h : 23 - x = 4) : 23 * x = 437 := by
  sorry

end correct_calculation_l1679_167953


namespace total_limes_is_195_l1679_167943

/-- The number of limes picked by each person -/
def fred_limes : ℕ := 36
def alyssa_limes : ℕ := 32
def nancy_limes : ℕ := 35
def david_limes : ℕ := 42
def eileen_limes : ℕ := 50

/-- The total number of limes picked -/
def total_limes : ℕ := fred_limes + alyssa_limes + nancy_limes + david_limes + eileen_limes

/-- Theorem stating that the total number of limes picked is 195 -/
theorem total_limes_is_195 : total_limes = 195 := by
  sorry

end total_limes_is_195_l1679_167943


namespace scientific_notation_317000_l1679_167984

theorem scientific_notation_317000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 317000 = a * (10 : ℝ) ^ n ∧ a = 3.17 ∧ n = 5 := by
  sorry

end scientific_notation_317000_l1679_167984


namespace curve_intersection_implies_a_equals_one_l1679_167956

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ θ : ℝ) (a : ℝ) : Prop :=
  ρ^2 - 2*ρ*(Real.sin θ) + 1 - a^2 = 0 ∧ a > 0

/-- Curve C₂ in polar coordinates -/
def C₂ (ρ θ : ℝ) : Prop :=
  ρ = 4*(Real.cos θ)

/-- Line C₃ in polar coordinates -/
def C₃ (θ : ℝ) : Prop :=
  ∃ α₀, θ = α₀ ∧ Real.tan α₀ = 2

/-- Common points of C₁ and C₂ lie on C₃ -/
def common_points_on_C₃ (a : ℝ) : Prop :=
  ∀ ρ θ, C₁ ρ θ a ∧ C₂ ρ θ → C₃ θ

theorem curve_intersection_implies_a_equals_one :
  ∀ a, common_points_on_C₃ a → a = 1 :=
sorry

end curve_intersection_implies_a_equals_one_l1679_167956


namespace rhombus_area_l1679_167901

/-- The area of a rhombus with side length 25 and one diagonal 30 is 600 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) (area : ℝ) : 
  side = 25 → 
  diagonal1 = 30 → 
  diagonal2^2 = 4 * (side^2 - (diagonal1/2)^2) → 
  area = (diagonal1 * diagonal2) / 2 → 
  area = 600 := by
  sorry

end rhombus_area_l1679_167901


namespace pulley_center_distance_l1679_167922

def pulley_problem (r₁ r₂ d : ℝ) : Prop :=
  r₁ > 0 ∧ r₂ > 0 ∧ d > 0 ∧
  r₁ = 10 ∧ r₂ = 6 ∧ d = 26 →
  ∃ (center_distance : ℝ),
    center_distance = 2 * Real.sqrt 173

theorem pulley_center_distance :
  ∀ (r₁ r₂ d : ℝ), pulley_problem r₁ r₂ d :=
by sorry

end pulley_center_distance_l1679_167922


namespace fraction_equivalence_l1679_167900

theorem fraction_equivalence : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 :=
by
  use 13 / 2
  sorry

end fraction_equivalence_l1679_167900


namespace interior_perimeter_is_155_l1679_167987

/-- Triangle PQR with parallel lines forming interior triangle --/
structure TriangleWithParallels where
  /-- Side length PQ --/
  pq : ℝ
  /-- Side length QR --/
  qr : ℝ
  /-- Side length PR --/
  pr : ℝ
  /-- Length of intersection of m_P with triangle interior --/
  m_p : ℝ
  /-- Length of intersection of m_Q with triangle interior --/
  m_q : ℝ
  /-- Length of intersection of m_R with triangle interior --/
  m_r : ℝ
  /-- m_P is parallel to QR --/
  m_p_parallel_qr : True
  /-- m_Q is parallel to RP --/
  m_q_parallel_rp : True
  /-- m_R is parallel to PQ --/
  m_r_parallel_pq : True

/-- The perimeter of the interior triangle formed by parallel lines --/
def interiorPerimeter (t : TriangleWithParallels) : ℝ :=
  t.m_p + t.m_q + t.m_r

/-- Theorem: The perimeter of the interior triangle is 155 --/
theorem interior_perimeter_is_155 (t : TriangleWithParallels) 
  (h1 : t.pq = 160) (h2 : t.qr = 300) (h3 : t.pr = 240)
  (h4 : t.m_p = 75) (h5 : t.m_q = 60) (h6 : t.m_r = 20) :
  interiorPerimeter t = 155 := by
  sorry

end interior_perimeter_is_155_l1679_167987


namespace number_problem_l1679_167952

theorem number_problem (x : ℝ) : (1/3) * x - 5 = 10 → x = 45 := by
  sorry

end number_problem_l1679_167952


namespace absolute_value_inequality_l1679_167920

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) → a ≤ 4 := by
  sorry

end absolute_value_inequality_l1679_167920


namespace c_value_theorem_l1679_167958

theorem c_value_theorem : ∃ c : ℝ, 
  (∀ x : ℝ, x * (3 * x + 1) < c ↔ -7/3 < x ∧ x < 2) ∧ c = 14 := by
  sorry

end c_value_theorem_l1679_167958


namespace rectangle_area_l1679_167935

theorem rectangle_area (L W : ℝ) (h1 : L / W = 5 / 3) (h2 : L - 5 = W + 3) : L * W = 240 := by
  sorry

end rectangle_area_l1679_167935
