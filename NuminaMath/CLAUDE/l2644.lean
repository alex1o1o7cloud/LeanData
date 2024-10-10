import Mathlib

namespace simplify_expression_l2644_264448

theorem simplify_expression (b c : ℝ) :
  3 * b * (3 * b^3 + 2 * b) - 2 * b^2 + c * (3 * b^2 - c) = 9 * b^4 + 4 * b^2 + 3 * b^2 * c - c^2 := by
  sorry

end simplify_expression_l2644_264448


namespace sqrt_square_12321_l2644_264420

theorem sqrt_square_12321 : (Real.sqrt 12321)^2 = 12321 := by sorry

end sqrt_square_12321_l2644_264420


namespace rob_unique_cards_rob_doubles_ratio_jess_doubles_ratio_alex_doubles_ratio_l2644_264486

-- Define the friends
structure Friend where
  name : String
  total_cards : ℕ
  doubles : ℕ

-- Define the problem setup
def rob : Friend := { name := "Rob", total_cards := 24, doubles := 8 }
def jess : Friend := { name := "Jess", total_cards := 0, doubles := 40 } -- total_cards unknown
def alex : Friend := { name := "Alex", total_cards := 0, doubles := 0 } -- both unknown

-- Theorem: Rob has 16 unique cards
theorem rob_unique_cards :
  rob.total_cards - rob.doubles = 16 :=
by
  sorry

-- Conditions from the problem
theorem rob_doubles_ratio :
  3 * rob.doubles = rob.total_cards :=
by
  sorry

theorem jess_doubles_ratio :
  jess.doubles = 5 * rob.doubles :=
by
  sorry

theorem alex_doubles_ratio (alex_total : ℕ) :
  4 * alex.doubles = alex_total :=
by
  sorry

end rob_unique_cards_rob_doubles_ratio_jess_doubles_ratio_alex_doubles_ratio_l2644_264486


namespace six_balls_two_boxes_l2644_264411

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 64 ways to distribute 6 distinguishable balls into 2 distinguishable boxes -/
theorem six_balls_two_boxes :
  distribute_balls 6 2 = 64 := by
  sorry

end six_balls_two_boxes_l2644_264411


namespace perfect_square_condition_l2644_264457

theorem perfect_square_condition (Z K : ℤ) : 
  (500 < Z ∧ Z < 1000) →
  K > 1 →
  Z = K * K^2 →
  (∃ n : ℤ, Z = n^2) →
  K = 9 := by
sorry

end perfect_square_condition_l2644_264457


namespace incenter_distance_l2644_264460

/-- Given a triangle PQR with sides PQ = 12, PR = 13, QR = 15, and incenter J, 
    the length of PJ is 7√2. -/
theorem incenter_distance (P Q R J : ℝ × ℝ) : 
  let d := (λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  (d P Q = 12) → (d P R = 13) → (d Q R = 15) → 
  (J.1 = (P.1 + Q.1 + R.1) / 3) → (J.2 = (P.2 + Q.2 + R.2) / 3) →
  d P J = 7 * Real.sqrt 2 := by
  sorry

end incenter_distance_l2644_264460


namespace difference_of_squares_l2644_264451

theorem difference_of_squares (x y : ℝ) : x^2 - 25*y^2 = (x - 5*y) * (x + 5*y) := by
  sorry

end difference_of_squares_l2644_264451


namespace anna_sandwiches_l2644_264477

theorem anna_sandwiches (slices_per_sandwich : ℕ) (current_slices : ℕ) (additional_slices : ℕ) :
  slices_per_sandwich = 3 →
  current_slices = 31 →
  additional_slices = 119 →
  (current_slices + additional_slices) / slices_per_sandwich = 50 :=
by
  sorry

end anna_sandwiches_l2644_264477


namespace point_on_parabola_l2644_264423

theorem point_on_parabola (a : ℝ) : (a, -9) ∈ {(x, y) | y = -x^2} → (a = 3 ∨ a = -3) := by
  sorry

end point_on_parabola_l2644_264423


namespace max_constant_inequality_l2644_264446

theorem max_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x^2 + y^2 = 1) :
  ∃ (c : ℝ), c = 1/2 ∧ x^6 + y^6 ≥ c*x*y ∧ ∀ (d : ℝ), (∀ (a b : ℝ), a > 0 → b > 0 → a^2 + b^2 = 1 → a^6 + b^6 ≥ d*a*b) → d ≤ c :=
by sorry

end max_constant_inequality_l2644_264446


namespace max_value_expression_l2644_264442

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 6*x^2 + 1)) / x ≤ 2/3 ∧
  ∃ x₀ > 0, (x₀^2 + 3 - Real.sqrt (x₀^4 + 6*x₀^2 + 1)) / x₀ = 2/3 :=
by sorry

end max_value_expression_l2644_264442


namespace adult_meals_sold_l2644_264434

theorem adult_meals_sold (kids_meals : ℕ) (adult_meals : ℕ) : 
  (10 : ℚ) / 7 = kids_meals / adult_meals →
  kids_meals = 70 →
  adult_meals = 49 := by
sorry

end adult_meals_sold_l2644_264434


namespace inequality_solution_set_l2644_264431

theorem inequality_solution_set (x : ℝ) : 
  (5 ≤ x / (3 * x - 8) ∧ x / (3 * x - 8) < 10) ↔ (8 / 3 < x ∧ x ≤ 20 / 7) :=
by sorry

end inequality_solution_set_l2644_264431


namespace smallest_odd_triangle_perimeter_l2644_264455

/-- A triangle with consecutive odd integer side lengths. -/
structure OddTriangle where
  a : ℕ
  is_odd : Odd a
  satisfies_inequality : a + (a + 2) > (a + 4) ∧ a + (a + 4) > (a + 2) ∧ (a + 2) + (a + 4) > a

/-- The perimeter of an OddTriangle. -/
def perimeter (t : OddTriangle) : ℕ := t.a + (t.a + 2) + (t.a + 4)

/-- The statement to be proven. -/
theorem smallest_odd_triangle_perimeter :
  (∃ t : OddTriangle, ∀ t' : OddTriangle, perimeter t ≤ perimeter t') ∧
  (∃ t : OddTriangle, perimeter t = 15) :=
sorry

end smallest_odd_triangle_perimeter_l2644_264455


namespace total_running_time_l2644_264445

def track_length : ℝ := 500
def num_laps : ℕ := 7
def first_section_length : ℝ := 200
def second_section_length : ℝ := 300
def first_section_speed : ℝ := 5
def second_section_speed : ℝ := 6

theorem total_running_time :
  (num_laps : ℝ) * (first_section_length / first_section_speed + second_section_length / second_section_speed) = 630 :=
by sorry

end total_running_time_l2644_264445


namespace two_digit_number_difference_l2644_264482

theorem two_digit_number_difference (a b : ℕ) : 
  a ≥ 1 → a ≤ 9 → b ≤ 9 → (10 * a + b) - (10 * b + a) = 45 → a - b = 5 := by
sorry

end two_digit_number_difference_l2644_264482


namespace smallest_positive_omega_l2644_264437

theorem smallest_positive_omega : ∃ (ω : ℝ), 
  (ω > 0) ∧ 
  (∀ x, Real.sin (ω * (x - Real.pi / 6)) = Real.cos (ω * x)) ∧
  (∀ ω' > 0, (∀ x, Real.sin (ω' * (x - Real.pi / 6)) = Real.cos (ω' * x)) → ω ≤ ω') ∧
  ω = 9 := by
  sorry

end smallest_positive_omega_l2644_264437


namespace z_in_second_quadrant_l2644_264405

def z : ℂ := Complex.I * (1 + Complex.I)

theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end z_in_second_quadrant_l2644_264405


namespace sum_of_gcd_values_l2644_264424

theorem sum_of_gcd_values (n : ℕ+) : 
  (Finset.sum (Finset.range 4) (λ i => (Nat.gcd (5 * n + 6) n).succ)) = 12 := by
  sorry

end sum_of_gcd_values_l2644_264424


namespace waterfall_flow_rate_l2644_264467

/-- The waterfall problem -/
theorem waterfall_flow_rate 
  (basin_capacity : ℝ) 
  (leak_rate : ℝ) 
  (fill_time : ℝ) 
  (h1 : basin_capacity = 260) 
  (h2 : leak_rate = 4) 
  (h3 : fill_time = 13) : 
  ∃ (flow_rate : ℝ), flow_rate = 24 ∧ 
  fill_time * flow_rate - fill_time * leak_rate = basin_capacity :=
sorry

end waterfall_flow_rate_l2644_264467


namespace decreasing_reciprocal_function_l2644_264464

theorem decreasing_reciprocal_function (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : 0 < x₂) (h3 : x₁ < x₂) :
  (1 : ℝ) / x₁ > (1 : ℝ) / x₂ := by
  sorry

end decreasing_reciprocal_function_l2644_264464


namespace cauchy_schwarz_and_inequality_proof_l2644_264433

theorem cauchy_schwarz_and_inequality_proof :
  (∀ a b c x y z : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 →
    (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) ≥ (a*x + b*y + c*z)^2 ∧
    ((a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = (a*x + b*y + c*z)^2 ↔ a/x = b/y ∧ b/y = c/z)) ∧
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 →
    Real.sqrt a + Real.sqrt (2*b) + Real.sqrt (3*c) ≤ Real.sqrt 6 ∧
    (Real.sqrt a + Real.sqrt (2*b) + Real.sqrt (3*c) = Real.sqrt 6 ↔ a = 1/6 ∧ b = 1/3 ∧ c = 1/2)) :=
by sorry

end cauchy_schwarz_and_inequality_proof_l2644_264433


namespace expression_evaluation_l2644_264463

theorem expression_evaluation (b : ℕ) (h : b = 2) : (b^3 * b^4) - b^2 = 124 := by
  sorry

end expression_evaluation_l2644_264463


namespace distance_equals_speed_times_time_l2644_264475

/-- The distance between Patrick's house and Aaron's house -/
def distance : ℝ := 14

/-- The time Patrick spent jogging -/
def time : ℝ := 2

/-- Patrick's jogging speed -/
def speed : ℝ := 7

/-- Theorem stating that the distance is equal to speed multiplied by time -/
theorem distance_equals_speed_times_time : distance = speed * time := by
  sorry

end distance_equals_speed_times_time_l2644_264475


namespace no_solution_for_equation_l2644_264430

theorem no_solution_for_equation :
  ¬ ∃ (p q r : ℕ), 2^p + 5^q = 19^r := by
  sorry

end no_solution_for_equation_l2644_264430


namespace max_volume_smaller_pyramid_l2644_264435

/-- Regular square pyramid with base side length 2 and height 3 -/
structure SquarePyramid where
  base_side : ℝ
  height : ℝ
  base_side_eq : base_side = 2
  height_eq : height = 3

/-- Smaller pyramid formed by intersecting the main pyramid with a parallel plane -/
structure SmallerPyramid (p : SquarePyramid) where
  intersection_height : ℝ
  volume : ℝ
  height_bounds : 0 < intersection_height ∧ intersection_height < p.height
  volume_eq : volume = (4 / 27) * intersection_height^3 - (8 / 9) * intersection_height^2 + (4 / 3) * intersection_height

/-- The maximum volume of the smaller pyramid is 16/27 -/
theorem max_volume_smaller_pyramid (p : SquarePyramid) : 
  ∃ (sp : SmallerPyramid p), ∀ (other : SmallerPyramid p), sp.volume ≥ other.volume ∧ sp.volume = 16/27 := by
  sorry

end max_volume_smaller_pyramid_l2644_264435


namespace ellipse_trace_l2644_264472

/-- Given a complex number z with |z| = 3, the locus of points (x, y) satisfying z + 2/z = x + yi forms an ellipse -/
theorem ellipse_trace (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), z + 2 / z = x + y * Complex.I ↔ x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end ellipse_trace_l2644_264472


namespace toucan_count_l2644_264468

theorem toucan_count (initial_toucans : ℕ) (joining_toucans : ℕ) : 
  initial_toucans = 2 → joining_toucans = 1 → initial_toucans + joining_toucans = 3 := by
  sorry

end toucan_count_l2644_264468


namespace sin_thirteen_pi_sixths_l2644_264410

theorem sin_thirteen_pi_sixths : Real.sin (13 * π / 6) = 1 / 2 := by
  sorry

end sin_thirteen_pi_sixths_l2644_264410


namespace problem_solution_l2644_264487

noncomputable section

variables (a b c x : ℝ)

def f (x : ℝ) (b : ℝ) : ℝ := |x + b^2| - |-x + 1|
def g (x a b c : ℝ) : ℝ := |x + a^2 + c^2| + |x - 2*b^2|

theorem problem_solution (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a*b + b*c + a*c = 1) :
  (∀ x, f x 1 ≥ 1 ↔ x ∈ Set.Ici (1/2)) ∧
  (∀ x, f x b ≤ g x a b c) := by sorry

end problem_solution_l2644_264487


namespace angle_double_complement_measure_l2644_264461

theorem angle_double_complement_measure : ∀ x : ℝ, 
  (x = 2 * (90 - x)) → x = 60 := by
  sorry

end angle_double_complement_measure_l2644_264461


namespace square_properties_l2644_264456

/-- Given a square with diagonal length 12√2 cm, prove its perimeter and inscribed circle area -/
theorem square_properties (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  let perimeter := 4 * s
  let r := s / 2
  let inscribed_circle_area := Real.pi * r^2
  (perimeter = 48 ∧ inscribed_circle_area = 36 * Real.pi) := by
  sorry

end square_properties_l2644_264456


namespace prime_factor_sum_l2644_264474

theorem prime_factor_sum (w x y z : ℕ) 
  (h : 2^w * 3^x * 5^y * 11^z = 825) : 
  w + 2*x + 3*y + 4*z = 12 := by
sorry

end prime_factor_sum_l2644_264474


namespace vector_magnitude_l2644_264401

def unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

theorem vector_magnitude (e₁ e₂ : ℝ × ℝ) (h₁ : unit_vector e₁) (h₂ : unit_vector e₂)
  (h₃ : e₁.1 * e₂.1 + e₁.2 * e₂.2 = 1/2) : 
  let a := (2 * e₁.1 + e₂.1, 2 * e₁.2 + e₂.2)
  (a.1^2 + a.2^2) = 7 := by sorry

end vector_magnitude_l2644_264401


namespace some_number_value_l2644_264425

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = some_number * 35 * 45 * 35) : 
  some_number = 1 := by
  sorry

end some_number_value_l2644_264425


namespace johns_father_age_difference_l2644_264465

/-- Given John's age and the sum of John and his father's ages, 
    prove the difference between John's father's age and twice John's age. -/
theorem johns_father_age_difference (john_age : ℕ) (sum_ages : ℕ) 
    (h1 : john_age = 15)
    (h2 : john_age + (2 * john_age + sum_ages - john_age) = sum_ages)
    (h3 : sum_ages = 77) : 
  (2 * john_age + sum_ages - john_age) - 2 * john_age = 32 := by
  sorry

end johns_father_age_difference_l2644_264465


namespace min_transportation_cost_l2644_264404

/-- Represents the transportation problem with two production locations and two delivery venues -/
structure TransportationProblem where
  unitsJ : ℕ  -- Units produced in location J
  unitsY : ℕ  -- Units produced in location Y
  unitsA : ℕ  -- Units delivered to venue A
  unitsB : ℕ  -- Units delivered to venue B
  costJB : ℕ  -- Transportation cost from J to B per unit
  fixedCost : ℕ  -- Fixed overhead cost

/-- Calculates the total transportation cost given the number of units transported from J to A -/
def totalCost (p : TransportationProblem) (x : ℕ) : ℕ :=
  p.costJB * (p.unitsJ - x) + p.fixedCost

/-- Theorem stating the minimum transportation cost -/
theorem min_transportation_cost (p : TransportationProblem) 
    (h1 : p.unitsJ = 17) (h2 : p.unitsY = 15) (h3 : p.unitsA = 18) (h4 : p.unitsB = 14)
    (h5 : p.costJB = 200) (h6 : p.fixedCost = 19300) :
    ∃ (x : ℕ), x ≥ 3 ∧ 
    (∀ (y : ℕ), y ≥ 3 → totalCost p x ≤ totalCost p y) ∧
    totalCost p x = 19900 := by
  sorry

end min_transportation_cost_l2644_264404


namespace lanas_roses_l2644_264492

theorem lanas_roses (tulips : ℕ) (used_flowers : ℕ) (extra_flowers : ℕ) 
  (h1 : tulips = 36)
  (h2 : used_flowers = 70)
  (h3 : extra_flowers = 3) :
  tulips + (used_flowers + extra_flowers - tulips) = 73 :=
by sorry

end lanas_roses_l2644_264492


namespace carl_practice_hours_l2644_264478

/-- The number of weeks Carl practices -/
def total_weeks : ℕ := 8

/-- The required average hours per week -/
def required_average : ℕ := 15

/-- The hours practiced in the first 7 weeks -/
def first_seven_weeks : List ℕ := [14, 16, 12, 18, 15, 13, 17]

/-- The sum of hours practiced in the first 7 weeks -/
def sum_first_seven : ℕ := first_seven_weeks.sum

/-- The number of hours Carl must practice in the 8th week -/
def hours_eighth_week : ℕ := 15

theorem carl_practice_hours :
  (sum_first_seven + hours_eighth_week) / total_weeks = required_average :=
sorry

end carl_practice_hours_l2644_264478


namespace abs_ratio_eq_sqrt_two_l2644_264495

theorem abs_ratio_eq_sqrt_two (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 6*a*b) :
  |((a + b) / (a - b))| = Real.sqrt 2 := by
  sorry

end abs_ratio_eq_sqrt_two_l2644_264495


namespace arithmetic_sequence_12th_term_l2644_264480

def arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : arithmetic_sequence a q)
  (h2 : q > 1)
  (h3 : a 3 * a 7 = 72)
  (h4 : a 2 + a 8 = 27) :
  a 12 = 96 :=
sorry

end arithmetic_sequence_12th_term_l2644_264480


namespace trash_in_classrooms_l2644_264496

theorem trash_in_classrooms (total_trash : ℕ) (outside_trash : ℕ) 
  (h1 : total_trash = 1576) 
  (h2 : outside_trash = 1232) : 
  total_trash - outside_trash = 344 := by
  sorry

end trash_in_classrooms_l2644_264496


namespace rain_free_paths_l2644_264413

/-- The function f representing the amount of rain at point (x,y) -/
def f (x y : ℝ) : ℝ := |x^3 + 2*x^2*y - 5*x*y^2 - 6*y^3|

/-- The theorem stating that the set of m values for which f(x,mx) = 0 for all x
    is exactly {-1, 1/2, -1/3} -/
theorem rain_free_paths (x : ℝ) :
  {m : ℝ | ∀ x, f x (m*x) = 0} = {-1, 1/2, -1/3} := by
  sorry

end rain_free_paths_l2644_264413


namespace quadrilateral_similarity_l2644_264418

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  -- Add necessary fields and properties to define a convex quadrilateral
  -- This is a placeholder and should be expanded based on specific requirements

/-- Construct a new quadrilateral from the given one using perpendicular bisectors -/
def constructNextQuadrilateral (Q : ConvexQuadrilateral) : ConvexQuadrilateral :=
  sorry  -- Definition of the construction process

/-- Two quadrilaterals are similar -/
def isSimilar (Q1 Q2 : ConvexQuadrilateral) : Prop :=
  sorry  -- Definition of similarity for quadrilaterals

theorem quadrilateral_similarity (Q1 : ConvexQuadrilateral) :
  let Q2 := constructNextQuadrilateral Q1
  let Q3 := constructNextQuadrilateral Q2
  isSimilar Q3 Q1 := by
  sorry

end quadrilateral_similarity_l2644_264418


namespace triangle_inequality_satisfied_l2644_264462

theorem triangle_inequality_satisfied (a b c : ℝ) (ha : a = 8) (hb : b = 8) (hc : c = 15) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end triangle_inequality_satisfied_l2644_264462


namespace y1_less_than_y2_l2644_264485

/-- Given a linear function y = 2x + 1 and two points (-1, y₁) and (3, y₂) on its graph,
    prove that y₁ < y₂ -/
theorem y1_less_than_y2 (y₁ y₂ : ℝ) : 
  (y₁ = 2 * (-1) + 1) → (y₂ = 2 * 3 + 1) → y₁ < y₂ := by
  sorry

end y1_less_than_y2_l2644_264485


namespace unique_solution_condition_l2644_264473

/-- The equation has exactly one solution if and only if a is in the specified set -/
theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, a * |2 + x| + (x^2 + x - 12) / (x + 4) = 0) ↔ 
  (a ∈ Set.Ioc (-1) 1 ∪ {7/2}) := by sorry

end unique_solution_condition_l2644_264473


namespace count_divisible_numbers_l2644_264454

theorem count_divisible_numbers (n : ℕ) : 
  (Finset.filter (fun k => (k^2 - 1) % 291 = 0) (Finset.range (291000 + 1))).card = 4000 :=
sorry

end count_divisible_numbers_l2644_264454


namespace min_value_on_line_l2644_264471

theorem min_value_on_line (x y : ℝ) : 
  x + y = 4 → ∀ a b : ℝ, a + b = 4 → x^2 + y^2 ≤ a^2 + b^2 ∧ ∃ c d : ℝ, c + d = 4 ∧ c^2 + d^2 = 8 :=
by sorry

end min_value_on_line_l2644_264471


namespace chocolate_packs_l2644_264429

theorem chocolate_packs (total packs_cookies packs_cake : ℕ) 
  (h_total : total = 42)
  (h_cookies : packs_cookies = 4)
  (h_cake : packs_cake = 22) :
  total - packs_cookies - packs_cake = 16 := by
  sorry

end chocolate_packs_l2644_264429


namespace opposite_of_negative_five_l2644_264426

theorem opposite_of_negative_five : 
  -((-5 : ℤ)) = (5 : ℤ) := by sorry

end opposite_of_negative_five_l2644_264426


namespace sqrt_difference_inequality_l2644_264422

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end sqrt_difference_inequality_l2644_264422


namespace smallest_n_exceeding_15_l2644_264402

def f (n : ℕ+) : ℕ := sorry

theorem smallest_n_exceeding_15 :
  (∀ k : ℕ+, k < 3 → f k ≤ 15) ∧ f 3 > 15 := by sorry

end smallest_n_exceeding_15_l2644_264402


namespace slower_train_speed_l2644_264483

/-- Proves that the speed of the slower train is 36 km/hr given the conditions of the problem --/
theorem slower_train_speed (faster_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 46 →
  passing_time = 36 →
  train_length = 50 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * passing_time * (1000 / 3600) = 2 * train_length :=
by
  sorry


end slower_train_speed_l2644_264483


namespace smallest_norm_u_l2644_264414

theorem smallest_norm_u (u : ℝ × ℝ) (h : ‖u + (4, 2)‖ = 10) :
  ∃ (v : ℝ × ℝ), ‖v‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ (w : ℝ × ℝ), ‖w + (4, 2)‖ = 10 → ‖v‖ ≤ ‖w‖ := by
  sorry

end smallest_norm_u_l2644_264414


namespace chess_tournament_participants_l2644_264490

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 231 → n = 22 := by
  sorry

end chess_tournament_participants_l2644_264490


namespace rabbit_fur_genetics_l2644_264491

/-- Represents the phase of meiotic division --/
inductive MeioticPhase
  | LateFirst
  | LateSecond

/-- Represents the fur length gene --/
inductive FurGene
  | Long
  | Short

/-- Represents a rabbit's genotype for fur length --/
structure RabbitGenotype where
  allele1 : FurGene
  allele2 : FurGene

/-- Represents the genetic characteristics of rabbit fur --/
structure RabbitFurGenetics where
  totalGenes : ℕ
  genesPerOocyte : ℕ
  nucleotideTypes : ℕ
  separationPhase : MeioticPhase

def isHeterozygous (genotype : RabbitGenotype) : Prop :=
  genotype.allele1 ≠ genotype.allele2

def maxShortFurOocytes (genetics : RabbitFurGenetics) (genotype : RabbitGenotype) : ℕ :=
  genetics.totalGenes / genetics.genesPerOocyte

theorem rabbit_fur_genetics 
  (genetics : RabbitFurGenetics) 
  (genotype : RabbitGenotype) :
  isHeterozygous genotype →
  genetics.totalGenes = 20 →
  genetics.genesPerOocyte = 4 →
  genetics.nucleotideTypes = 4 →
  genetics.separationPhase = MeioticPhase.LateFirst →
  maxShortFurOocytes genetics genotype = 5 :=
by sorry

end rabbit_fur_genetics_l2644_264491


namespace negation_of_and_zero_l2644_264488

theorem negation_of_and_zero (x y : ℝ) : ¬(x = 0 ∧ y = 0) ↔ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end negation_of_and_zero_l2644_264488


namespace super_champion_tournament_24_teams_l2644_264444

/-- The number of games played in a tournament with a given number of teams --/
def tournament_games (n : ℕ) : ℕ :=
  n - 1

/-- The total number of games in a tournament with a "Super Champion" game --/
def super_champion_tournament (n : ℕ) : ℕ :=
  tournament_games n + 1

/-- Theorem: A tournament with 24 teams and a "Super Champion" game has 24 total games --/
theorem super_champion_tournament_24_teams :
  super_champion_tournament 24 = 24 := by
  sorry

end super_champion_tournament_24_teams_l2644_264444


namespace modular_inverse_17_mod_800_l2644_264439

theorem modular_inverse_17_mod_800 : ∃ x : ℕ, x < 800 ∧ (17 * x) % 800 = 1 :=
by
  use 47
  sorry

end modular_inverse_17_mod_800_l2644_264439


namespace triangle_inequality_with_constant_l2644_264459

theorem triangle_inequality_with_constant (k : ℕ) : 
  (k > 0) →
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 →
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
    a + b > c ∧ b + c > a ∧ c + a > b) ↔
  k = 6 := by
sorry

end triangle_inequality_with_constant_l2644_264459


namespace dillar_dallar_never_equal_l2644_264409

/-- Represents the state of the financier's money -/
structure MoneyState :=
  (dillars : ℕ)
  (dallars : ℕ)

/-- Represents a currency exchange operation -/
inductive ExchangeOp
  | DillarToDallar : ExchangeOp
  | DallarToDillar : ExchangeOp

/-- Applies an exchange operation to a money state -/
def applyExchange (state : MoneyState) (op : ExchangeOp) : MoneyState :=
  match op with
  | ExchangeOp.DillarToDallar => 
      ⟨state.dillars - 1, state.dallars + 10⟩
  | ExchangeOp.DallarToDillar => 
      ⟨state.dillars + 10, state.dallars - 1⟩

/-- Applies a sequence of exchange operations to an initial state -/
def applyExchanges (initial : MoneyState) (ops : List ExchangeOp) : MoneyState :=
  ops.foldl applyExchange initial

theorem dillar_dallar_never_equal :
  ∀ (ops : List ExchangeOp),
    let finalState := applyExchanges ⟨1, 0⟩ ops
    finalState.dillars ≠ finalState.dallars :=
by sorry

end dillar_dallar_never_equal_l2644_264409


namespace yellow_shirt_pairs_l2644_264484

theorem yellow_shirt_pairs (total_students : ℕ) (blue_shirts : ℕ) (yellow_shirts : ℕ) 
  (total_pairs : ℕ) (blue_pairs : ℕ) 
  (h1 : total_students = 132)
  (h2 : blue_shirts = 57)
  (h3 : yellow_shirts = 75)
  (h4 : total_pairs = 66)
  (h5 : blue_pairs = 23)
  (h6 : total_students = blue_shirts + yellow_shirts)
  (h7 : blue_pairs ≤ total_pairs) :
  ∃ yellow_pairs : ℕ, yellow_pairs = total_pairs - blue_pairs - (blue_shirts - 2 * blue_pairs) :=
by sorry

end yellow_shirt_pairs_l2644_264484


namespace boat_license_combinations_l2644_264458

/-- The number of possible letters for the first character of a boat license -/
def numLetters : ℕ := 3

/-- The number of possible digits for each of the six numeric positions in a boat license -/
def numDigits : ℕ := 10

/-- The number of numeric positions in a boat license -/
def numPositions : ℕ := 6

/-- The total number of unique boat license combinations -/
def totalCombinations : ℕ := numLetters * (numDigits ^ numPositions)

theorem boat_license_combinations :
  totalCombinations = 3000000 := by
  sorry

end boat_license_combinations_l2644_264458


namespace smallest_integer_x_l2644_264466

theorem smallest_integer_x : ∃ x : ℤ, 
  (∀ z : ℤ, (7 - 5*z < 25 ∧ 10 - 3*z > 6) → x ≤ z) ∧ 
  (7 - 5*x < 25 ∧ 10 - 3*x > 6) ∧
  x = -3 := by sorry

end smallest_integer_x_l2644_264466


namespace porter_monthly_earnings_l2644_264400

/-- Calculates the monthly earnings of a worker with overtime -/
def monthlyEarningsWithOvertime (dailyRate : ℕ) (regularDaysPerWeek : ℕ) (overtimeRatePercent : ℕ) (weeksInMonth : ℕ) : ℕ :=
  let regularWeeklyEarnings := dailyRate * regularDaysPerWeek
  let overtimeDailyRate := dailyRate * overtimeRatePercent / 100
  let overtimeWeeklyEarnings := dailyRate + overtimeDailyRate
  (regularWeeklyEarnings + overtimeWeeklyEarnings) * weeksInMonth

/-- Theorem stating that under given conditions, monthly earnings with overtime equal $208 -/
theorem porter_monthly_earnings :
  monthlyEarningsWithOvertime 8 5 150 4 = 208 := by
  sorry

#eval monthlyEarningsWithOvertime 8 5 150 4

end porter_monthly_earnings_l2644_264400


namespace smallest_factor_l2644_264450

theorem smallest_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 → 
    (2^4 ∣ 1452 * m) ∧ 
    (3^3 ∣ 1452 * m) ∧ 
    (13^3 ∣ 1452 * m) → 
    m ≥ n) ↔ 
  n = 676 := by
sorry

end smallest_factor_l2644_264450


namespace certain_number_problem_l2644_264499

theorem certain_number_problem : ∃ x : ℝ, 0.85 * x = (4/5 * 25) + 14 ∧ x = 40 := by
  sorry

end certain_number_problem_l2644_264499


namespace special_function_property_l2644_264470

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧
  (∀ x : ℝ, f (x + 5) ≥ f x + 5) ∧
  (∀ x : ℝ, f (x + 1) ≤ f x + 1)

/-- The function g defined in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1 - x

/-- The main theorem -/
theorem special_function_property (f : ℝ → ℝ) (hf : special_function f) :
  g f 2009 = 1 := by
  sorry

end special_function_property_l2644_264470


namespace inequality_range_l2644_264432

theorem inequality_range (a : ℝ) : 
  (∀ (x y : ℝ), x > 0 ∧ y > 0 → (y / 4) - Real.cos x ^ 2 ≥ a * Real.sin x - (9 / y)) →
  -3 ≤ a ∧ a ≤ 3 :=
by sorry

end inequality_range_l2644_264432


namespace sum_interior_angles_decagon_l2644_264452

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The sum of the interior angles of a decagon is 1440 degrees -/
theorem sum_interior_angles_decagon :
  sum_interior_angles decagon_sides = 1440 := by sorry

end sum_interior_angles_decagon_l2644_264452


namespace quadratic_roots_squared_difference_l2644_264412

theorem quadratic_roots_squared_difference (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (x₁^2 - x₂^2 = c^2 / a^2) ↔ (b^4 - c^4 = 4*a*b^2*c) :=
by sorry

end quadratic_roots_squared_difference_l2644_264412


namespace three_numbers_sum_l2644_264419

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →  -- Ordering of numbers
  b = 10 →  -- Median is 10
  (a + b + c) / 3 = a + 8 →  -- Mean is 8 more than least
  (a + b + c) / 3 = c - 20 →  -- Mean is 20 less than greatest
  a + b + c = 66 := by
sorry

end three_numbers_sum_l2644_264419


namespace area_of_triangle_LGH_l2644_264428

-- Define the circle and points
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def P : ℝ × ℝ := (0, 0)
def L : ℝ × ℝ := (-24, 0)
def M : ℝ × ℝ := (-10, 0)
def N : ℝ × ℝ := (10, 0)

-- Define the chords
def EF : Set (ℝ × ℝ) := {p | p.2 = 6}
def GH : Set (ℝ × ℝ) := {p | p.2 = 8}

-- State the theorem
theorem area_of_triangle_LGH : 
  ∀ (G H : ℝ × ℝ),
  G ∈ GH → H ∈ GH →
  G.1 < H.1 →
  H.1 - G.1 = 16 →
  (∀ (E F : ℝ × ℝ), E ∈ EF → F ∈ EF → F.1 - E.1 = 12) →
  (∀ p, p ∈ Circle P 10) →
  (∀ x, (x, 0) ∈ Set.Icc L N → (x, 0) ∈ Circle P 10) →
  let triangle_area := (1 / 2) * 16 * 6
  triangle_area = 48 := by sorry

end area_of_triangle_LGH_l2644_264428


namespace common_number_in_list_l2644_264449

theorem common_number_in_list (l : List ℝ) 
  (h_length : l.length = 7)
  (h_avg_first : (l.take 4).sum / 4 = 7)
  (h_avg_last : (l.drop 3).sum / 4 = 9)
  (h_avg_all : l.sum / 7 = 8) :
  ∃ x ∈ l.take 4 ∩ l.drop 3, x = 8 := by
  sorry

end common_number_in_list_l2644_264449


namespace sum_of_coefficients_l2644_264417

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + 
    a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + a₉*(x+2)^9 + 
    a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by
  sorry

end sum_of_coefficients_l2644_264417


namespace unit_complex_rational_power_minus_one_is_rational_l2644_264421

/-- A complex number with rational real and imaginary parts and modulus 1 -/
structure UnitComplexRational where
  re : ℚ
  im : ℚ
  norm_sq : re^2 + im^2 = 1

/-- The main theorem: z^(2n) - 1 is rational for any integer n -/
theorem unit_complex_rational_power_minus_one_is_rational
  (z : UnitComplexRational) (n : ℤ) :
  ∃ (q : ℚ), (z.re + z.im * Complex.I) ^ (2 * n) - 1 = q := by
  sorry

end unit_complex_rational_power_minus_one_is_rational_l2644_264421


namespace transport_cost_proof_l2644_264438

def cost_per_kg : ℝ := 18000
def instrument_mass_g : ℝ := 300

theorem transport_cost_proof :
  let instrument_mass_kg : ℝ := instrument_mass_g / 1000
  instrument_mass_kg * cost_per_kg = 5400 := by
  sorry

end transport_cost_proof_l2644_264438


namespace slope_determines_m_l2644_264494

theorem slope_determines_m (m : ℝ) : 
  let A : ℝ × ℝ := (-m, 6)
  let B : ℝ × ℝ := (1, 3*m)
  (B.2 - A.2) / (B.1 - A.1) = 12 → m = -2 := by
sorry

end slope_determines_m_l2644_264494


namespace rationalize_denominator_l2644_264441

theorem rationalize_denominator :
  (Real.sqrt 18 + Real.sqrt 2) / (Real.sqrt 3 + Real.sqrt 2) = 4 * (Real.sqrt 6 - 2) := by
  sorry

end rationalize_denominator_l2644_264441


namespace triangle_property_l2644_264408

open Real

theorem triangle_property (A B C a b c : ℝ) :
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = π →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  1 / tan A + 1 / tan C = 1 / sin B →
  b^2 = a * c :=
by sorry

end triangle_property_l2644_264408


namespace sunflower_majority_on_wednesday_sunflower_proportion_increases_l2644_264453

/-- Represents the amount of sunflower seeds on a given day -/
def sunflower_seeds (day : ℕ) : ℝ :=
  3 * (1 - (0.8 ^ day))

/-- Represents the total amount of seeds on any day after adding new seeds -/
def total_seeds : ℝ := 2

/-- Theorem stating that Wednesday (day 3) is the first day when sunflower seeds exceed half of total seeds -/
theorem sunflower_majority_on_wednesday :
  (∀ d < 3, sunflower_seeds d ≤ total_seeds / 2) ∧
  (sunflower_seeds 3 > total_seeds / 2) :=
by sorry

/-- Helper theorem: The proportion of sunflower seeds increases each day -/
theorem sunflower_proportion_increases (d : ℕ) :
  sunflower_seeds d < sunflower_seeds (d + 1) :=
by sorry

end sunflower_majority_on_wednesday_sunflower_proportion_increases_l2644_264453


namespace shirt_cost_l2644_264489

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 86) : S = 24 := by
  sorry

end shirt_cost_l2644_264489


namespace arctan_sum_equals_pi_half_l2644_264447

theorem arctan_sum_equals_pi_half (n : ℕ+) :
  Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/n) = π/2 ↔ n = 4 :=
by sorry

end arctan_sum_equals_pi_half_l2644_264447


namespace ages_solution_l2644_264443

/-- Represents the ages of three persons --/
structure Ages where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ

/-- Checks if the given ages satisfy the problem conditions --/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.eldest = ages.middle + 16 ∧
  ages.middle = ages.youngest + 8 ∧
  ages.eldest - 6 = 3 * (ages.youngest - 6) ∧
  ages.eldest - 6 = 2 * (ages.middle - 6)

/-- Theorem stating that the ages 18, 26, and 42 satisfy the problem conditions --/
theorem ages_solution :
  ∃ (ages : Ages), satisfiesConditions ages ∧ 
    ages.youngest = 18 ∧ ages.middle = 26 ∧ ages.eldest = 42 := by
  sorry

end ages_solution_l2644_264443


namespace total_puppies_eq_sum_l2644_264497

/-- The number of puppies Alyssa's dog had -/
def total_puppies : ℕ := 23

/-- The number of puppies Alyssa gave to her friends -/
def puppies_given_away : ℕ := 15

/-- The number of puppies Alyssa kept for herself -/
def puppies_kept : ℕ := 8

/-- Theorem stating that the total number of puppies is the sum of puppies given away and kept -/
theorem total_puppies_eq_sum : total_puppies = puppies_given_away + puppies_kept := by
  sorry

end total_puppies_eq_sum_l2644_264497


namespace one_third_of_x_l2644_264493

theorem one_third_of_x (x y : ℚ) : 
  x / y = 15 / 3 → y = 24 → x / 3 = 40 := by sorry

end one_third_of_x_l2644_264493


namespace stockholm_malmo_distance_l2644_264481

/-- The road distance between Stockholm and Malmo in kilometers -/
def road_distance (map_distance : ℝ) (scale : ℝ) (road_factor : ℝ) : ℝ :=
  map_distance * scale * road_factor

/-- Theorem: The road distance between Stockholm and Malmo is 1380 km -/
theorem stockholm_malmo_distance :
  road_distance 120 10 1.15 = 1380 := by
  sorry

end stockholm_malmo_distance_l2644_264481


namespace emberly_walks_l2644_264436

theorem emberly_walks (total_days : Nat) (miles_per_walk : Nat) (total_miles : Nat) :
  total_days = 31 →
  miles_per_walk = 4 →
  total_miles = 108 →
  total_days - (total_miles / miles_per_walk) = 4 :=
by sorry

end emberly_walks_l2644_264436


namespace container_volume_ratio_l2644_264407

theorem container_volume_ratio (container1 container2 : ℝ) : 
  container1 > 0 → container2 > 0 →
  (2/3 : ℝ) * container1 + (1/6 : ℝ) * container1 = (5/6 : ℝ) * container2 →
  container1 = container2 := by
sorry

end container_volume_ratio_l2644_264407


namespace tablecloth_extension_theorem_l2644_264498

/-- Represents a circular table with a square tablecloth placed on it. -/
structure TableWithCloth where
  /-- Diameter of the circular table in meters -/
  table_diameter : ℝ
  /-- Side length of the square tablecloth in meters -/
  cloth_side_length : ℝ
  /-- Extension of one corner beyond the table edge in meters -/
  corner1_extension : ℝ
  /-- Extension of an adjacent corner beyond the table edge in meters -/
  corner2_extension : ℝ

/-- Calculates the extensions of the remaining two corners of the tablecloth. -/
def calculate_remaining_extensions (t : TableWithCloth) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct extensions for the given table and tablecloth configuration. -/
theorem tablecloth_extension_theorem (t : TableWithCloth) 
  (h1 : t.table_diameter = 0.6)
  (h2 : t.cloth_side_length = 1)
  (h3 : t.corner1_extension = 0.5)
  (h4 : t.corner2_extension = 0.3) :
  calculate_remaining_extensions t = (0.33, 0.52) :=
by sorry

end tablecloth_extension_theorem_l2644_264498


namespace circle_symmetry_minimum_l2644_264416

theorem circle_symmetry_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y + 1 = 0 ↔ x^2 + y^2 + 4*x - 2*y + 1 = 0 ∧ a*x - b*y + 1 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → (a' + 2*b') / (a' * b') ≥ (a + 2*b) / (a * b)) →
  (a + 2*b) / (a * b) = 9 :=
sorry

end circle_symmetry_minimum_l2644_264416


namespace painted_cells_count_l2644_264476

-- Define the structure of the grid
structure Grid :=
  (k : ℕ)
  (l : ℕ)

-- Define the properties of the grid
def valid_grid (g : Grid) : Prop :=
  g.k = 2 ∧ g.l = 37

-- Define the number of white cells
def white_cells (g : Grid) : ℕ :=
  g.k * g.l

-- Define the total number of cells
def total_cells (g : Grid) : ℕ :=
  (2 * g.k + 1) * (2 * g.l + 1)

-- Define the number of painted cells
def painted_cells (g : Grid) : ℕ :=
  total_cells g - white_cells g

-- The main theorem
theorem painted_cells_count (g : Grid) :
  valid_grid g → white_cells g = 74 → painted_cells g = 301 :=
by
  sorry


end painted_cells_count_l2644_264476


namespace proportional_value_l2644_264406

-- Define the given ratio
def given_ratio : ℚ := 12 / 6

-- Define the conversion factor from minutes to seconds
def minutes_to_seconds : ℕ := 60

-- Define the target time in minutes
def target_time_minutes : ℕ := 8

-- Define the target time in seconds
def target_time_seconds : ℕ := target_time_minutes * minutes_to_seconds

-- State the theorem
theorem proportional_value :
  (given_ratio * target_time_seconds : ℚ) = 960 := by sorry

end proportional_value_l2644_264406


namespace coloring_satisfies_conditions_l2644_264479

-- Define the color type
inductive Color
  | White
  | Red
  | Black

-- Define the coloring function
def color (x y : Int) : Color :=
  if (x + y) % 2 = 0 then Color.Red
  else if x % 2 = 1 && y % 2 = 0 then Color.White
  else Color.Black

-- Define a lattice point
structure LatticePoint where
  x : Int
  y : Int

-- Define a property that a color appears infinitely many times on infinitely many horizontal lines
def infiniteOccurrence (c : Color) : Prop :=
  ∀ (n : Nat), ∃ (m : Int), ∀ (k : Int), ∃ (x : Int), 
    color x (m + k * n) = c

-- Define the parallelogram property
def parallelogramProperty : Prop :=
  ∀ (A B C : LatticePoint),
    color A.x A.y = Color.White →
    color B.x B.y = Color.Red →
    color C.x C.y = Color.Black →
    ∃ (D : LatticePoint),
      color D.x D.y = Color.Red ∧
      D.x - C.x = A.x - B.x ∧
      D.y - C.y = A.y - B.y

-- The main theorem
theorem coloring_satisfies_conditions :
  (∀ c : Color, infiniteOccurrence c) ∧ parallelogramProperty :=
sorry

end coloring_satisfies_conditions_l2644_264479


namespace robin_bobbin_chickens_l2644_264440

def chickens_eaten_sept_1 (chickens_sept_2 chickens_total_sept_15 : ℕ) : ℕ :=
  let avg_daily_consumption := chickens_total_sept_15 / 15
  let chickens_sept_1_and_2 := 2 * avg_daily_consumption
  chickens_sept_1_and_2 - chickens_sept_2

theorem robin_bobbin_chickens :
  chickens_eaten_sept_1 12 32 = 52 :=
sorry

end robin_bobbin_chickens_l2644_264440


namespace min_value_quadratic_l2644_264415

theorem min_value_quadratic (x y : ℝ) : x^2 + 4*x*y + 5*y^2 - 8*x - 6*y ≥ -41 := by
  sorry

end min_value_quadratic_l2644_264415


namespace total_highlighters_l2644_264403

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 15

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 12

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := 9

/-- The number of green highlighters in the teacher's desk -/
def green_highlighters : ℕ := 7

/-- The number of purple highlighters in the teacher's desk -/
def purple_highlighters : ℕ := 6

/-- Theorem stating that the total number of highlighters is 49 -/
theorem total_highlighters : 
  pink_highlighters + yellow_highlighters + blue_highlighters + green_highlighters + purple_highlighters = 49 := by
  sorry

end total_highlighters_l2644_264403


namespace equations_have_same_solutions_l2644_264427

def daniels_equation (x : ℝ) : Prop := |x - 8| = 3

def emmas_equation (x : ℝ) : Prop := x^2 - 16*x + 55 = 0

theorem equations_have_same_solutions :
  (∀ x : ℝ, daniels_equation x ↔ emmas_equation x) :=
sorry

end equations_have_same_solutions_l2644_264427


namespace decimal_sum_to_fraction_l2644_264469

theorem decimal_sum_to_fraction : 
  0.4 + 0.05 + 0.006 + 0.0007 + 0.00008 = 22839 / 50000 := by
  sorry

end decimal_sum_to_fraction_l2644_264469
