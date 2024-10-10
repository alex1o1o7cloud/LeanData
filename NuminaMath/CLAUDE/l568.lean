import Mathlib

namespace sprite_to_coke_ratio_l568_56801

/-- Represents a drink mixture with three components -/
structure Drink where
  total : ℝ
  coke : ℝ
  sprite : ℝ
  mountainDew : ℝ
  cokeParts : ℝ
  mountainDewParts : ℝ

/-- Theorem stating the ratio of Sprite to Coke in the drink -/
theorem sprite_to_coke_ratio (d : Drink) 
  (h1 : d.total = 18)
  (h2 : d.coke = 6)
  (h3 : d.cokeParts = 2)
  (h4 : d.mountainDewParts = 3)
  (h5 : d.total = d.coke + d.sprite + d.mountainDew)
  (h6 : d.coke / d.cokeParts = d.mountainDew / d.mountainDewParts) : 
  d.sprite / d.coke = 1 / 2 := by
  sorry

end sprite_to_coke_ratio_l568_56801


namespace scout_weights_l568_56804

/-- The weight measurement error of the scale -/
def error : ℝ := 2

/-- Míša's measured weight -/
def misa_measured : ℝ := 30

/-- Emil's measured weight -/
def emil_measured : ℝ := 28

/-- Combined measured weight of Míša and Emil -/
def combined_measured : ℝ := 56

/-- Míša's actual weight -/
def misa_actual : ℝ := misa_measured - error

/-- Emil's actual weight -/
def emil_actual : ℝ := emil_measured - error

theorem scout_weights :
  misa_actual = 28 ∧ emil_actual = 26 ∧
  misa_actual + emil_actual = combined_measured - error := by
  sorry

end scout_weights_l568_56804


namespace graph_reflection_l568_56882

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define the reflection across y-axis
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Statement: The graph of y = g(-x) is the reflection of y = g(x) across the y-axis
theorem graph_reflection (x : ℝ) : 
  reflect_y (x, g x) = (-x, g (-x)) := by sorry

end graph_reflection_l568_56882


namespace root_equality_implies_b_equals_four_l568_56847

theorem root_equality_implies_b_equals_four
  (a b c : ℕ)
  (a_gt_one : a > 1)
  (b_gt_one : b > 1)
  (c_gt_one : c > 1)
  (h : ∀ N : ℝ, N ≠ 1 → N^(1/a + 1/(a*b) + 1/(a*b*c) + 1/(a*b*c^2)) = N^(49/60)) :
  b = 4 := by sorry

end root_equality_implies_b_equals_four_l568_56847


namespace fuel_used_fraction_l568_56866

def car_speed : ℝ := 50
def fuel_efficiency : ℝ := 30
def tank_capacity : ℝ := 15
def travel_time : ℝ := 5

theorem fuel_used_fraction (speed : ℝ) (efficiency : ℝ) (capacity : ℝ) (time : ℝ)
  (h1 : speed = car_speed)
  (h2 : efficiency = fuel_efficiency)
  (h3 : capacity = tank_capacity)
  (h4 : time = travel_time) :
  (speed * time / efficiency) / capacity = 5 / 9 := by
  sorry

end fuel_used_fraction_l568_56866


namespace solution_set_for_a_equals_2_range_of_a_l568_56892

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 3| - |x - a|

-- Theorem for the first part of the problem
theorem solution_set_for_a_equals_2 :
  {x : ℝ | f 2 x ≤ -1/2} = {x : ℝ | x ≥ 11/4} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a :
  {a : ℝ | ∀ x, f a x ≥ a} = Set.Iic (3/2) := by sorry

end solution_set_for_a_equals_2_range_of_a_l568_56892


namespace equal_wealth_after_transfer_l568_56855

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.foma - 70 = w.ierema + 70) ∧ 
  (w.foma - 40 = w.yuliy)

/-- The theorem to be proved -/
theorem equal_wealth_after_transfer (w : MerchantWealth) 
  (h : problem_conditions w) : 
  w.foma - 55 = w.ierema + 55 := by
  sorry

end equal_wealth_after_transfer_l568_56855


namespace new_numbers_mean_l568_56803

/-- Given 7 numbers with mean 36 and 3 new numbers making a total of 10 with mean 48,
    prove that the mean of the 3 new numbers is 76. -/
theorem new_numbers_mean (original_count : Nat) (new_count : Nat) 
  (original_mean : ℝ) (new_mean : ℝ) : 
  original_count = 7 →
  new_count = 3 →
  original_mean = 36 →
  new_mean = 48 →
  (original_count * original_mean + new_count * 
    ((original_count + new_count) * new_mean - original_count * original_mean) / new_count) / 
    new_count = 76 := by
  sorry

end new_numbers_mean_l568_56803


namespace loan_future_value_l568_56816

/-- Represents the relationship between principal and future value for a loan -/
theorem loan_future_value 
  (P A : ℝ) -- Principal and future value
  (r : ℝ) -- Annual interest rate
  (n : ℕ) -- Number of times interest is compounded per year
  (t : ℕ) -- Number of years
  (h1 : r = 0.12) -- Interest rate is 12%
  (h2 : n = 2) -- Compounded half-yearly
  (h3 : t = 20) -- Loan period is 20 years
  : A = P * (1 + r/n)^(n*t) :=
by sorry

end loan_future_value_l568_56816


namespace parabola_circle_intersection_l568_56807

/-- Parabola C₁: y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Circle C₂: (x-1)² + y² = 1 -/
def Circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  hy : y^2 = 2 * C.p * x

/-- Only the vertex of C₁ is on C₂, all other points are outside -/
axiom vertex_on_circle_others_outside (C : Parabola) :
  Circle 0 0 ∧ ∀ (P : PointOnParabola C), P.x ≠ 0 → ¬Circle P.x P.y

/-- Fixed point M on C₁ with y₀ > 0 -/
structure FixedPoint (C : Parabola) extends PointOnParabola C where
  hy_pos : y > 0

/-- Two points A and B on C₁ -/
structure IntersectionPoints (C : Parabola) where
  A : PointOnParabola C
  B : PointOnParabola C

/-- Slopes of MA and MB exist and their angles are complementary -/
axiom complementary_slopes (C : Parabola) (M : FixedPoint C) (I : IntersectionPoints C) :
  ∃ (k : ℝ), k ≠ 0 ∧
    (I.A.y - M.y) / (I.A.x - M.x) = k ∧
    (I.B.y - M.y) / (I.B.x - M.x) = -k

/-- Main theorem -/
theorem parabola_circle_intersection (C : Parabola) (M : FixedPoint C) (I : IntersectionPoints C) :
  C.p ≥ 1 ∧
  ∃ (slope : ℝ), slope = -C.p / M.y ∧ slope ≠ 0 ∧
    (I.B.y - I.A.y) / (I.B.x - I.A.x) = slope := by sorry

end parabola_circle_intersection_l568_56807


namespace eighth_term_is_84_l568_56831

/-- The n-th term of the sequence -/
def S (n : ℕ) : ℚ := (3 * n * (n - 1)) / 2

/-- Theorem: The 8th term of the sequence is 84 -/
theorem eighth_term_is_84 : S 8 = 84 := by sorry

end eighth_term_is_84_l568_56831


namespace arithmetic_mean_problem_l568_56824

theorem arithmetic_mean_problem (p q r : ℝ) : 
  (p + q) / 2 = 10 → 
  (q + r) / 2 = 22 → 
  r - p = 24 → 
  (q + r) / 2 = 22 := by
sorry

end arithmetic_mean_problem_l568_56824


namespace triangle_side_values_l568_56820

theorem triangle_side_values (a b c : ℝ) (A B C : ℝ) :
  -- Define triangle ABC
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = π) →
  -- Area condition
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) →
  -- Given conditions
  (c = 2) →
  (A = π/3) →
  -- Conclusion
  (a = Real.sqrt 3 ∧ b = 1) := by
sorry

end triangle_side_values_l568_56820


namespace gcd_2703_1113_l568_56861

theorem gcd_2703_1113 : Nat.gcd 2703 1113 = 159 := by
  sorry

end gcd_2703_1113_l568_56861


namespace necklace_ratio_l568_56833

/-- The number of necklaces Haley, Jason, and Josh have. -/
structure Necklaces where
  haley : ℕ
  jason : ℕ
  josh : ℕ

/-- The conditions given in the problem. -/
def problem_conditions (n : Necklaces) : Prop :=
  n.haley = n.jason + 5 ∧
  n.haley = 25 ∧
  n.haley = n.josh + 15

/-- The theorem stating that under the given conditions, 
    the ratio of Josh's necklaces to Jason's necklaces is 1:2. -/
theorem necklace_ratio (n : Necklaces) 
  (h : problem_conditions n) : n.josh * 2 = n.jason := by
  sorry


end necklace_ratio_l568_56833


namespace evaluate_expression_l568_56841

theorem evaluate_expression : (3^2)^3 + 2*(3^2 - 2^3) = 731 := by
  sorry

end evaluate_expression_l568_56841


namespace other_candidate_votes_l568_56830

-- Define the total number of votes
def total_votes : ℕ := 7500

-- Define the percentage of invalid votes
def invalid_vote_percentage : ℚ := 20 / 100

-- Define the percentage of votes for the winning candidate
def winning_candidate_percentage : ℚ := 55 / 100

-- Theorem to prove
theorem other_candidate_votes :
  (total_votes * (1 - invalid_vote_percentage) * (1 - winning_candidate_percentage)).floor = 2700 :=
by sorry

end other_candidate_votes_l568_56830


namespace quadratic_negative_on_unit_interval_l568_56857

/-- Given a quadratic function f(x) = ax^2 + bx + c where a > b > c and a + b + c = 0,
    prove that f(x) is negative for all x in the open interval (0,1) -/
theorem quadratic_negative_on_unit_interval 
  (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∀ x ∈ Set.Ioo 0 1, a * x^2 + b * x + c < 0 :=
sorry

end quadratic_negative_on_unit_interval_l568_56857


namespace power_of_power_l568_56813

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end power_of_power_l568_56813


namespace two_m_squared_eq_three_n_cubed_l568_56812

theorem two_m_squared_eq_three_n_cubed (m n : ℕ+) :
  2 * m ^ 2 = 3 * n ^ 3 ↔ ∃ k : ℕ+, m = 18 * k ^ 3 ∧ n = 6 * k ^ 2 := by
  sorry

end two_m_squared_eq_three_n_cubed_l568_56812


namespace nut_distribution_properties_l568_56842

/-- Represents the state of nut distribution among three people -/
structure NutState where
  anya : ℕ
  borya : ℕ
  vitya : ℕ

/-- The nut distribution process -/
def distributeNuts (state : NutState) : NutState :=
  sorry

/-- Predicate to check if at least one nut is eaten during the entire process -/
def atLeastOneNutEaten (initialState : NutState) : Prop :=
  sorry

/-- Predicate to check if not all nuts are eaten during the entire process -/
def notAllNutsEaten (initialState : NutState) : Prop :=
  sorry

/-- Main theorem stating the properties of the nut distribution process -/
theorem nut_distribution_properties {n : ℕ} (h : n > 3) :
  let initialState : NutState := ⟨n, 0, 0⟩
  atLeastOneNutEaten initialState ∧ notAllNutsEaten initialState :=
by
  sorry

end nut_distribution_properties_l568_56842


namespace riverdale_school_theorem_l568_56895

def riverdale_school (total students_in_band students_in_chorus students_in_band_or_chorus : ℕ) : Prop :=
  students_in_band + students_in_chorus - students_in_band_or_chorus = 30

theorem riverdale_school_theorem :
  riverdale_school 250 90 120 180 := by
  sorry

end riverdale_school_theorem_l568_56895


namespace calculate_expression_l568_56886

theorem calculate_expression : (1/3)⁻¹ + Real.sqrt 12 - |Real.sqrt 3 - 2| - (π - 2023)^0 = 3 * Real.sqrt 3 := by
  sorry

end calculate_expression_l568_56886


namespace parallel_vectors_x_value_l568_56826

/-- Given two vectors a and b in ℝ², where a = (2, 3) and b = (x, -9),
    if a is parallel to b, then x = -6. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![x, -9]
  (∃ (k : ℝ), b = k • a) →
  x = -6 :=
by sorry

end parallel_vectors_x_value_l568_56826


namespace equal_selection_probability_l568_56806

/-- Represents the selection process for voluntary labor --/
structure SelectionProcess where
  total_students : ℕ
  excluded : ℕ
  selected : ℕ
  h_total : total_students = 1008
  h_excluded : excluded = 8
  h_selected : selected = 20
  h_remaining : total_students - excluded = 1000

/-- The probability of being selected for an individual student --/
def selection_probability (process : SelectionProcess) : ℚ :=
  process.selected / process.total_students

/-- States that the selection probability is equal for all students --/
theorem equal_selection_probability (process : SelectionProcess) :
  ∀ (student1 student2 : Fin process.total_students),
    selection_probability process = selection_probability process :=
by sorry

end equal_selection_probability_l568_56806


namespace percent_decrease_proof_l568_56817

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 70) :
  (original_price - sale_price) / original_price * 100 = 30 := by
  sorry

end percent_decrease_proof_l568_56817


namespace john_booking_l568_56809

/-- Calculates the number of nights booked given the nightly rate, discount, and total paid -/
def nights_booked (nightly_rate : ℕ) (discount : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid + discount) / nightly_rate

theorem john_booking :
  nights_booked 250 100 650 = 3 := by
  sorry

end john_booking_l568_56809


namespace complex_equation_solution_l568_56825

theorem complex_equation_solution (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + i) * i = b + i →
  a = 1 ∧ b = -1 := by
sorry

end complex_equation_solution_l568_56825


namespace inverse_trig_sum_equals_pi_l568_56880

theorem inverse_trig_sum_equals_pi : 
  let arctan_sqrt3 := π / 3
  let arcsin_neg_half := -π / 6
  let arccos_zero := π / 2
  arctan_sqrt3 - arcsin_neg_half + arccos_zero = π := by sorry

end inverse_trig_sum_equals_pi_l568_56880


namespace circle_line_intersection_l568_56888

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the line l passing through the origin
def line_l (k x y : ℝ) : Prop := y = k * x

-- Define the trajectory Γ
def trajectory_Γ (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0 ∧ 3/2 < x ∧ x ≤ 2

-- Define the line m
def line_m (a x y : ℝ) : Prop := y = a * x + 4

theorem circle_line_intersection
  (k : ℝ) -- Slope of line l
  (a : ℝ) -- Parameter for line m
  : 
  (∃ (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l k x1 y1 ∧ line_l k x2 y2) →
  (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3) ∧
  (∀ x y, trajectory_Γ x y ↔ 
    ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ 
    x = (x1 + x2) / 2 * (1 - t) + 2 * t ∧
    y = (y1 + y2) / 2 * (1 - t)) ∧
  ((∃! x y, trajectory_Γ x y ∧ line_m a x y) →
    (a = -15/8 ∨ (-Real.sqrt 3 - 8)/3 < a ∧ a ≤ (Real.sqrt 3 - 8)/3)) :=
by sorry

end circle_line_intersection_l568_56888


namespace total_time_is_80_minutes_l568_56843

/-- The total time students spend outside of class -/
def total_time_outside_class (recess1 recess2 lunch recess3 : ℕ) : ℕ :=
  recess1 + recess2 + lunch + recess3

/-- Theorem stating that the total time outside class is 80 minutes -/
theorem total_time_is_80_minutes :
  total_time_outside_class 15 15 30 20 = 80 := by
  sorry

end total_time_is_80_minutes_l568_56843


namespace triangle_properties_area_condition1_area_condition2_l568_56881

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)

-- Define the main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.B ≠ π/2)
  (h2 : Real.cos (2 * t.B) = Real.sqrt 3 * Real.cos t.B - 1) :
  t.B = π/6 ∧ 
  ((Real.sin t.A = Real.sqrt 3 * Real.sin t.C ∧ t.b = 2 → t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3) ∨
   (2 * t.b = 3 * t.a ∧ t.b * Real.sin t.A = 1 → t.a * t.c * Real.sin t.B / 2 = (Real.sqrt 3 + 2 * Real.sqrt 2) / 2)) :=
by sorry

-- Define additional theorems for each condition
theorem area_condition1 (t : Triangle) 
  (h1 : t.B = π/6)
  (h2 : Real.sin t.A = Real.sqrt 3 * Real.sin t.C)
  (h3 : t.b = 2) :
  t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3 :=
by sorry

theorem area_condition2 (t : Triangle)
  (h1 : t.B = π/6)
  (h2 : 2 * t.b = 3 * t.a)
  (h3 : t.b * Real.sin t.A = 1) :
  t.a * t.c * Real.sin t.B / 2 = (Real.sqrt 3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end triangle_properties_area_condition1_area_condition2_l568_56881


namespace bombardier_solution_l568_56836

/-- Represents the number of bombs thrown by each bombardier -/
structure BombardierShots where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Defines the conditions of the bombardier problem -/
def satisfiesConditions (shots : BombardierShots) : Prop :=
  (shots.first + shots.second = shots.third + 26) ∧
  (shots.second + shots.third = shots.first + shots.second + 38) ∧
  (shots.first + shots.third = shots.second + 24)

/-- Theorem stating the solution to the bombardier problem -/
theorem bombardier_solution :
  ∃ (shots : BombardierShots), satisfiesConditions shots ∧
    shots.first = 25 ∧ shots.second = 64 ∧ shots.third = 63 := by
  sorry

end bombardier_solution_l568_56836


namespace first_problem_number_l568_56844

/-- Given a sequence of 48 consecutive integers ending with 125, 
    the first number in the sequence is 78. -/
theorem first_problem_number (last_number : ℕ) (total_problems : ℕ) :
  last_number = 125 → total_problems = 48 → 
  (last_number - total_problems + 1 : ℕ) = 78 := by
  sorry

end first_problem_number_l568_56844


namespace largest_four_digit_divisible_by_50_l568_56862

theorem largest_four_digit_divisible_by_50 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 50 = 0 → n ≤ 9950 :=
by sorry

end largest_four_digit_divisible_by_50_l568_56862


namespace hyperbola_eccentricity_l568_56883

/-- A hyperbola centered at the origin -/
structure Hyperbola where
  center : ℝ × ℝ
  asymptotes : Set (ℝ → ℝ)

/-- A circle with equation (x-h)^2 + (y-k)^2 = r^2 -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Definition of eccentricity for a hyperbola -/
def eccentricity (h : Hyperbola) : Set ℝ := sorry

/-- Definition of a line being tangent to a circle -/
def is_tangent (l : ℝ → ℝ) (c : Circle) : Prop := sorry

theorem hyperbola_eccentricity (h : Hyperbola) (c : Circle) :
  h.center = (0, 0) →
  c.center = (2, 0) →
  c.radius = Real.sqrt 3 →
  (∀ a ∈ h.asymptotes, is_tangent a c) →
  eccentricity h = {2, 2 * Real.sqrt 3 / 3} := by sorry

end hyperbola_eccentricity_l568_56883


namespace min_value_of_z_l568_56828

theorem min_value_of_z (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 12) :
  ∃ (z_min : ℝ), z_min = -5 ∧ ∀ (z : ℝ), z = 2*x + Real.sqrt 3 * y → z ≥ z_min :=
sorry

end min_value_of_z_l568_56828


namespace shortest_time_5x6_checkerboard_l568_56818

/-- Represents a checkerboard with alternating black and white squares. -/
structure Checkerboard where
  rows : Nat
  cols : Nat
  squareSize : ℝ
  normalSpeed : ℝ
  slowSpeed : ℝ

/-- Calculates the shortest time to travel from bottom-left to top-right corner of the checkerboard. -/
def shortestTravelTime (board : Checkerboard) : ℝ :=
  sorry

/-- The theorem stating the shortest travel time for the specific checkerboard. -/
theorem shortest_time_5x6_checkerboard :
  let board : Checkerboard := {
    rows := 5
    cols := 6
    squareSize := 1
    normalSpeed := 2
    slowSpeed := 1
  }
  shortestTravelTime board = (1 + 5 * Real.sqrt 2) / 2 := by
  sorry

end shortest_time_5x6_checkerboard_l568_56818


namespace salary_increase_l568_56840

theorem salary_increase (num_employees : ℕ) (initial_avg : ℚ) (manager_salary : ℚ) :
  num_employees = 20 →
  initial_avg = 1600 →
  manager_salary = 3700 →
  let total_salary := num_employees * initial_avg
  let new_total := total_salary + manager_salary
  let new_avg := new_total / (num_employees + 1)
  new_avg - initial_avg = 100 := by
  sorry

end salary_increase_l568_56840


namespace only_rectangle_area_certain_l568_56865

-- Define the events
inductive Event
  | WaterFreeze : Event
  | ExamScore : Event
  | CoinToss : Event
  | RectangleArea : Event

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.WaterFreeze => False
  | Event.ExamScore => False
  | Event.CoinToss => False
  | Event.RectangleArea => True

-- Theorem stating that only RectangleArea is a certain event
theorem only_rectangle_area_certain :
  ∀ (e : Event), is_certain e ↔ e = Event.RectangleArea :=
by sorry

end only_rectangle_area_certain_l568_56865


namespace hyperbola_parabola_intersection_l568_56827

/-- Given a hyperbola and a parabola with specific properties, prove that n = 12 -/
theorem hyperbola_parabola_intersection (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ (x y : ℝ), x^2/m - y^2/n = 1) →  -- hyperbola equation
  (∃ (e : ℝ), e = 2) →  -- eccentricity is 2
  (∃ (x y : ℝ), y^2 = 4*m*x) →  -- parabola equation
  (∃ (c : ℝ), c = m) →  -- focus of hyperbola coincides with focus of parabola
  n = 12 := by
sorry

end hyperbola_parabola_intersection_l568_56827


namespace julia_played_with_two_kids_on_monday_l568_56832

/-- The number of kids Julia played with on Monday and Tuesday -/
def total_kids : ℕ := 16

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 14

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := total_kids - tuesday_kids

theorem julia_played_with_two_kids_on_monday :
  monday_kids = 2 := by sorry

end julia_played_with_two_kids_on_monday_l568_56832


namespace hyperbola_circle_intersection_length_l568_56885

/-- Given a hyperbola and a circle with specific properties, prove the length of the chord formed by their intersection. -/
theorem hyperbola_circle_intersection_length :
  ∀ (a b : ℝ) (A B : ℝ × ℝ),
  a > 0 →
  b > 0 →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (∃ (t : ℝ), y = 2 * x * t ∨ y = -2 * x * t)) →  -- Asymptotes condition
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → Real.sqrt (1 + b^2 / a^2) = Real.sqrt 5) →  -- Eccentricity condition
  (∃ (t : ℝ), (A.1 - 2)^2 + (A.2 - 3)^2 = 1 ∧ 
              (B.1 - 2)^2 + (B.2 - 3)^2 = 1 ∧ 
              (A.2 = 2 * A.1 * t ∨ A.2 = -2 * A.1 * t) ∧ 
              (B.2 = 2 * B.1 * t ∨ B.2 = -2 * B.1 * t)) →  -- Intersection condition
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 / 5 := by
sorry


end hyperbola_circle_intersection_length_l568_56885


namespace cookies_taken_in_seven_days_l568_56874

/-- Represents the number of cookies Jessica takes each day -/
def jessica_daily_cookies : ℝ := 1.5

/-- Represents the number of cookies Sarah takes each day -/
def sarah_daily_cookies : ℝ := 3 * jessica_daily_cookies

/-- Represents the number of cookies Paul takes each day -/
def paul_daily_cookies : ℝ := 2 * sarah_daily_cookies

/-- Represents the total number of cookies in the jar initially -/
def initial_cookies : ℕ := 200

/-- Represents the number of cookies left after 10 days -/
def cookies_left : ℕ := 50

/-- Represents the number of days they took cookies -/
def total_days : ℕ := 10

/-- Represents the number of days we want to calculate for -/
def target_days : ℕ := 7

theorem cookies_taken_in_seven_days :
  (jessica_daily_cookies + sarah_daily_cookies + paul_daily_cookies) * target_days = 105 :=
by sorry

end cookies_taken_in_seven_days_l568_56874


namespace tadd_500th_number_l568_56808

def tadd_sequence (n : ℕ) : ℕ := (3 * n - 2) ^ 2

theorem tadd_500th_number : tadd_sequence 500 = 2244004 := by
  sorry

end tadd_500th_number_l568_56808


namespace parallel_line_plane_l568_56839

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Define the theorem
theorem parallel_line_plane 
  (m n : Line) (α : Plane)
  (distinct_lines : m ≠ n)
  (m_parallel_n : parallel m n)
  (m_parallel_α : parallel_plane m α)
  (n_not_in_α : ¬ contained_in n α) :
  parallel_plane n α :=
sorry

end parallel_line_plane_l568_56839


namespace last_two_digits_of_7_pow_2016_l568_56867

/-- The last two digits of 7^n, for n ≥ 1 -/
def lastTwoDigits (n : ℕ) : ℕ :=
  (7^n) % 100

/-- The period of the last two digits of powers of 7 -/
def period : ℕ := 4

theorem last_two_digits_of_7_pow_2016 :
  lastTwoDigits 2016 = 01 :=
by
  sorry

end last_two_digits_of_7_pow_2016_l568_56867


namespace complement_of_P_l568_56810

def U : Set ℝ := Set.univ

def P : Set ℝ := {x : ℝ | x^2 ≤ 1}

theorem complement_of_P : (Set.univ \ P) = {x : ℝ | x < -1 ∨ x > 1} := by
  sorry

end complement_of_P_l568_56810


namespace larger_number_problem_l568_56848

theorem larger_number_problem (L S : ℕ) (hL : L > S) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 := by
  sorry

end larger_number_problem_l568_56848


namespace cube_red_faces_ratio_l568_56877

/-- Represents a cube with side length n -/
structure Cube where
  n : ℕ

/-- Calculates the number of red faces on the original cube -/
def redFaces (c : Cube) : ℕ := 6 * c.n^2

/-- Calculates the total number of faces of all small cubes -/
def totalFaces (c : Cube) : ℕ := 6 * c.n^3

/-- Theorem: The side length of the cube is 3 if and only if
    exactly one-third of the faces of the small cubes are red -/
theorem cube_red_faces_ratio (c : Cube) : 
  c.n = 3 ↔ 3 * redFaces c = totalFaces c := by
  sorry


end cube_red_faces_ratio_l568_56877


namespace soap_decrease_l568_56899

theorem soap_decrease (x : ℝ) (h : x > 0) : x * (0.8 ^ 2) ≤ (2/3) * x :=
sorry

end soap_decrease_l568_56899


namespace impossible_to_turn_all_off_l568_56822

/-- Represents the state of a lightning bug (on or off) -/
inductive BugState
| On
| Off

/-- Represents a 6x6 grid of lightning bugs -/
def Grid := Fin 6 → Fin 6 → BugState

/-- Represents a move on the grid -/
inductive Move
| Horizontal (row : Fin 6) (start_col : Fin 6)
| Vertical (col : Fin 6) (start_row : Fin 6)

/-- Applies a move to a grid -/
def applyMove (grid : Grid) (move : Move) : Grid :=
  sorry

/-- Checks if all bugs in the grid are off -/
def allOff (grid : Grid) : Prop :=
  ∀ (row col : Fin 6), grid row col = BugState.Off

/-- Initial grid configuration with one bug on -/
def initialGrid : Grid :=
  sorry

/-- Theorem stating the impossibility of turning all bugs off -/
theorem impossible_to_turn_all_off :
  ¬∃ (moves : List Move), allOff (moves.foldl applyMove initialGrid) :=
  sorry

end impossible_to_turn_all_off_l568_56822


namespace median_intersection_l568_56878

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a median
def median (t : Triangle) (v : Point) (m : Point) : Prop :=
  (v = t.A ∨ v = t.B ∨ v = t.C) ∧
  (m = ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2) ∨
   m = ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2) ∨
   m = ((t.C.1 + t.A.1) / 2, (t.C.2 + t.A.2) / 2))

-- Define the ratio of division
def divides_in_ratio (p : Point) (v : Point) (m : Point) : Prop :=
  let d1 := ((p.1 - v.1)^2 + (p.2 - v.2)^2).sqrt
  let d2 := ((m.1 - p.1)^2 + (m.2 - p.2)^2).sqrt
  d1 / d2 = 2 / 1

-- Theorem statement
theorem median_intersection (t : Triangle) : 
  ∃ (O : Point), 
    (∀ (v m : Point), median t v m → divides_in_ratio O v m) ∧
    (∀ (v1 m1 v2 m2 : Point), 
      median t v1 m1 → median t v2 m2 → 
      ∃ (k : ℝ), O = (k * v1.1 + (1 - k) * m1.1, k * v1.2 + (1 - k) * m1.2) ∧
                 O = (k * v2.1 + (1 - k) * m2.1, k * v2.2 + (1 - k) * m2.2)) :=
by
  sorry

end median_intersection_l568_56878


namespace fish_problem_l568_56875

theorem fish_problem (total : ℕ) (carla_fish : ℕ) (kyle_fish : ℕ) (tasha_fish : ℕ) :
  total = 36 →
  carla_fish = 8 →
  kyle_fish = tasha_fish →
  total = carla_fish + kyle_fish + tasha_fish →
  kyle_fish = 14 := by
  sorry

end fish_problem_l568_56875


namespace nabla_computation_l568_56889

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_computation : (nabla (nabla 2 3) 4) = 16777219 := by
  sorry

end nabla_computation_l568_56889


namespace power_calculation_l568_56850

theorem power_calculation : (4^4 / 4^3) * 2^8 = 1024 := by
  sorry

end power_calculation_l568_56850


namespace complex_number_solution_l568_56873

theorem complex_number_solution (z : ℂ) : (Complex.I * z = 1) → z = -Complex.I := by
  sorry

end complex_number_solution_l568_56873


namespace triangle_inequality_l568_56890

/-- For any triangle ABC and real numbers x, y, and z, 
    x^2 + y^2 + z^2 ≥ 2xy cos C + 2yz cos A + 2zx cos B -/
theorem triangle_inequality (A B C : ℝ) (x y z : ℝ) : 
  x^2 + y^2 + z^2 ≥ 2*x*y*(Real.cos C) + 2*y*z*(Real.cos A) + 2*z*x*(Real.cos B) := by
  sorry

end triangle_inequality_l568_56890


namespace annie_candy_cost_l568_56845

/-- Calculates the total cost of candies Annie bought for her class -/
theorem annie_candy_cost (class_size : ℕ) (candies_per_classmate : ℕ) (leftover_candies : ℕ) (candy_cost : ℚ) : 
  class_size = 35 → 
  candies_per_classmate = 2 → 
  leftover_candies = 12 → 
  candy_cost = 1/10 →
  (class_size * candies_per_classmate + leftover_candies) * candy_cost = 82/10 := by
  sorry

end annie_candy_cost_l568_56845


namespace range_of_a_l568_56876

theorem range_of_a (a : ℝ) : 
  (∀ x, -2 < x ∧ x < 3 → -2 < x ∧ x < a) ∧ 
  (∃ x, -2 < x ∧ x < a ∧ ¬(-2 < x ∧ x < 3)) 
  ↔ a > 3 := by sorry

end range_of_a_l568_56876


namespace color_cartridge_cost_l568_56872

/-- The cost of each color cartridge given the total cost, number of cartridges, and cost of black-and-white cartridge. -/
theorem color_cartridge_cost (total_cost : ℕ) (bw_cost : ℕ) (num_color : ℕ) : 
  total_cost = bw_cost + num_color * 32 → 32 = (total_cost - bw_cost) / num_color :=
by sorry

end color_cartridge_cost_l568_56872


namespace negative_thirty_two_to_five_thirds_l568_56871

theorem negative_thirty_two_to_five_thirds :
  (-32 : ℝ) ^ (5/3) = -256 * (2 : ℝ) ^ (1/3) := by
  sorry

end negative_thirty_two_to_five_thirds_l568_56871


namespace sum_remainder_zero_l568_56802

theorem sum_remainder_zero (m : ℤ) : (11 - m + (m + 5)) % 8 = 0 := by
  sorry

end sum_remainder_zero_l568_56802


namespace count_squares_with_six_or_more_black_l568_56823

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  position : Nat × Nat

/-- The checkerboard -/
def checkerboard : Nat := 8

/-- Function to count black squares in a given square -/
def countBlackSquares (s : Square) : Nat :=
  sorry

/-- Function to check if a square is valid (fits on the board) -/
def isValidSquare (s : Square) : Bool :=
  s.size > 0 && s.size ≤ checkerboard &&
  s.position.1 + s.size ≤ checkerboard &&
  s.position.2 + s.size ≤ checkerboard

/-- Function to generate all valid squares on the board -/
def allValidSquares : List Square :=
  sorry

/-- Main theorem -/
theorem count_squares_with_six_or_more_black : 
  (allValidSquares.filter (fun s => isValidSquare s && countBlackSquares s ≥ 6)).length = 55 :=
  sorry

end count_squares_with_six_or_more_black_l568_56823


namespace point_ratio_on_line_l568_56834

/-- Given four points P, Q, R, and S on a line in that order, with specific distances between them,
    prove that the ratio of PR to QS is 7/17. -/
theorem point_ratio_on_line (P Q R S : ℝ) : 
  Q - P = 3 →
  R - Q = 4 →
  S - P = 20 →
  P < Q ∧ Q < R ∧ R < S →
  (R - P) / (S - Q) = 7 / 17 := by
sorry

end point_ratio_on_line_l568_56834


namespace r_value_when_n_is_3_l568_56811

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 5^n + 1
  let r : ℕ := 3^s - 3*s
  r = 3^126 - 378 := by
sorry

end r_value_when_n_is_3_l568_56811


namespace circumcircle_passes_through_O_l568_56846

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric operations
variable (parallelogram : Point → Point → Point → Point → Prop)
variable (intersectionPoint : Point → Point → Point → Point → Point)
variable (circumcircle : Point → Point → Point → Circle)
variable (onCircle : Point → Circle → Prop)
variable (intersectionCircles : Circle → Circle → Point)

-- State the theorem
theorem circumcircle_passes_through_O 
  (A B C D P Q R M N O : Point) :
  parallelogram A B C D →
  O = intersectionPoint A C B D →
  Q = intersectionCircles (circumcircle P O B) (circumcircle O A D) →
  R = intersectionCircles (circumcircle P O C) (circumcircle O A D) →
  Q ≠ O →
  R ≠ O →
  parallelogram P Q A M →
  parallelogram P R D N →
  onCircle O (circumcircle M N P) :=
by sorry

end circumcircle_passes_through_O_l568_56846


namespace intersection_A_B_union_complement_A_B_l568_56896

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | 4 - x^2 ≤ 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

-- Theorem for the union of complements of A and B
theorem union_complement_A_B :
  (Set.univ \ A) ∪ (Set.univ \ B) = {x : ℝ | x < -2 ∨ x > -1} := by sorry

end intersection_A_B_union_complement_A_B_l568_56896


namespace twenty_percent_greater_than_twelve_l568_56869

theorem twenty_percent_greater_than_twelve (x : ℝ) : 
  x = 12 * (1 + 0.2) → x = 14.4 := by
  sorry

end twenty_percent_greater_than_twelve_l568_56869


namespace muffin_spending_l568_56897

theorem muffin_spending (x : ℝ) : 
  (x = 0.9 * x + 15) → (x + 0.9 * x = 285) :=
by sorry

end muffin_spending_l568_56897


namespace expression_simplification_l568_56800

/-- Proves that the given expression simplifies to the expected result. -/
theorem expression_simplification (x y : ℝ) :
  3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + 2 * y) = 9 * x^2 + 6 * x - 2 * y - 3 := by
  sorry

end expression_simplification_l568_56800


namespace max_profit_at_two_l568_56819

noncomputable section

-- Define the sales volume function
def sales_volume (x : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 3 then (x - 4)^2 + 6 / (x - 1)
  else if 3 < x ∧ x ≤ 5 then -x + 7
  else 0

-- Define the profit function
def profit (x : ℝ) : ℝ :=
  (sales_volume x) * (x - 1)

-- Main theorem
theorem max_profit_at_two :
  ∀ x, 1 < x ∧ x ≤ 5 → profit x ≤ profit 2 :=
by sorry

end

end max_profit_at_two_l568_56819


namespace even_function_max_symmetry_l568_56859

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f has a maximum value on an interval [a, b] if there exists
    a point c in [a, b] such that f(c) ≥ f(x) for all x in [a, b] -/
def HasMaximumOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c

/-- If f is an even function and has a maximum value on [1, 7],
    then it also has a maximum value on [-7, -1] -/
theorem even_function_max_symmetry (f : ℝ → ℝ) :
  EvenFunction f → HasMaximumOn f 1 7 → HasMaximumOn f (-7) (-1) :=
by
  sorry

end even_function_max_symmetry_l568_56859


namespace factorization_of_2a2_minus_8b2_l568_56838

theorem factorization_of_2a2_minus_8b2 (a b : ℝ) :
  2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end factorization_of_2a2_minus_8b2_l568_56838


namespace factor_theorem_application_l568_56858

theorem factor_theorem_application (c : ℚ) : 
  (∀ x : ℚ, (x + 5) ∣ (2*c*x^3 + 14*x^2 - 6*c*x + 25)) → c = 75/44 := by
  sorry

end factor_theorem_application_l568_56858


namespace max_value_fraction_l568_56856

theorem max_value_fraction (x : ℝ) (h : x > 1) :
  (x^4 - x^2) / (x^6 + 2*x^3 - 1) ≤ 1/5 := by sorry

end max_value_fraction_l568_56856


namespace inscribed_trapezoid_a_value_l568_56829

/-- Trapezoid inscribed in a parabola -/
structure InscribedTrapezoid where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_a_gt_b : a > b
  h_sides_equal : 2*a + 2*b = 3/4 + Real.sqrt ((a - b)^2 + (a^2 - b^2)^2)
  h_ab : Real.sqrt ((a - b)^2 + (a^2 - b^2)^2) = 3/4

theorem inscribed_trapezoid_a_value (t : InscribedTrapezoid) : t.a = 27/40 := by
  sorry

end inscribed_trapezoid_a_value_l568_56829


namespace dropped_students_score_l568_56821

theorem dropped_students_score (initial_students : ℕ) (remaining_students : ℕ) 
  (initial_average : ℚ) (remaining_average : ℚ) 
  (h1 : initial_students = 30) 
  (h2 : remaining_students = 26) 
  (h3 : initial_average = 60.25) 
  (h4 : remaining_average = 63.75) :
  (initial_students : ℚ) * initial_average - 
  (remaining_students : ℚ) * remaining_average = 150 := by
  sorry


end dropped_students_score_l568_56821


namespace ellipse_major_axis_length_l568_56854

/-- Given an ellipse with equation 2x^2 + 3y^2 = 1, its major axis length is √2 -/
theorem ellipse_major_axis_length :
  let ellipse_eq : ℝ → ℝ → Prop := λ x y => 2 * x^2 + 3 * y^2 = 1
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ x y, ellipse_eq x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    2 * a = Real.sqrt 2 :=
by sorry

end ellipse_major_axis_length_l568_56854


namespace apple_orange_probability_l568_56887

theorem apple_orange_probability (n : ℕ) : 
  (n : ℚ) / (n + 3 : ℚ) = 2 / 3 → n = 6 := by
  sorry

end apple_orange_probability_l568_56887


namespace student_line_count_l568_56864

/-- Given a line of students, if a student is 7th from the left and 5th from the right,
    then the total number of students in the line is 11. -/
theorem student_line_count (n : ℕ) 
  (left_position : ℕ) 
  (right_position : ℕ) 
  (h1 : left_position = 7) 
  (h2 : right_position = 5) : 
  n = left_position + right_position - 1 := by
  sorry

end student_line_count_l568_56864


namespace lcm_factor_proof_l568_56894

theorem lcm_factor_proof (A B : ℕ) : 
  A > 0 ∧ B > 0 ∧ A ≥ B ∧ Nat.gcd A B = 30 ∧ A = 450 → 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ Nat.lcm A B = 30 * x * y ∧ (x = 15 ∨ y = 15) :=
by
  sorry

#check lcm_factor_proof

end lcm_factor_proof_l568_56894


namespace limit_fraction_binomial_sums_l568_56898

def a (n : ℕ+) : ℝ := (3 : ℝ) ^ n.val
def b (n : ℕ+) : ℝ := (2 : ℝ) ^ n.val

theorem limit_fraction_binomial_sums :
  ∀ ε > 0, ∃ N : ℕ+, ∀ n ≥ N,
    |((b (n + 1) - a n) / (a (n + 1) + b n)) + (1 / 3)| < ε :=
sorry

end limit_fraction_binomial_sums_l568_56898


namespace correlation_of_product_l568_56879

-- Define a random function type
def RandomFunction := ℝ → ℝ

-- Define the expectation operator
noncomputable def expectation (X : RandomFunction) : ℝ := sorry

-- Define the correlation function
noncomputable def correlation (X Y : RandomFunction) : ℝ := sorry

-- Define what it means for a random function to be centered
def is_centered (X : RandomFunction) : Prop :=
  expectation X = 0

-- Define what it means for two random functions to be uncorrelated
def are_uncorrelated (X Y : RandomFunction) : Prop :=
  expectation (fun t => X t * Y t) = expectation X * expectation Y

-- State the theorem
theorem correlation_of_product (X Y : RandomFunction) 
  (h1 : is_centered X) (h2 : is_centered Y) (h3 : are_uncorrelated X Y) :
  correlation (fun t => X t * Y t) (fun t => X t * Y t) = 
  correlation X X * correlation Y Y := by sorry

end correlation_of_product_l568_56879


namespace parabola_and_range_l568_56852

-- Define the parabola G
def G (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 4)

-- Define point A
def point_A : ℝ × ℝ := (-4, 0)

-- Define the condition for points B and C
def intersect_points (p k x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  G p x₁ y₁ ∧ G p x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the condition AC = 1/4 * AB when k = 1/2
def vector_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = 4*y₂

-- Define the y-intercept of the perpendicular bisector
def perpendicular_bisector_y_intercept (k : ℝ) : ℝ :=
  2*(k + 1)^2

-- Theorem statement
theorem parabola_and_range :
  ∀ p k x₁ y₁ x₂ y₂,
    G p (-4) 0 →
    intersect_points p k x₁ y₁ x₂ y₂ →
    k = 1/2 →
    vector_condition x₁ y₁ x₂ y₂ →
    (p = 2 ∧ 
     ∀ b, b > 2 ↔ ∃ k', perpendicular_bisector_y_intercept k' = b) :=
by sorry

end parabola_and_range_l568_56852


namespace ellipse_line_segment_no_intersection_l568_56891

theorem ellipse_line_segment_no_intersection (a : ℝ) :
  a > 0 →
  (∀ x y : ℝ, x^2 + (1/2) * y^2 = a^2 →
    ((2 ≤ x ∧ x ≤ 4 ∧ y = (3-1)/(4-2) * (x-2) + 1) → False)) →
  (0 < a ∧ a < 3 * Real.sqrt 2 / 2) ∨ (a > Real.sqrt 82 / 2) :=
sorry

end ellipse_line_segment_no_intersection_l568_56891


namespace white_balls_count_l568_56814

theorem white_balls_count (total : ℕ) (p_yellow : ℚ) : 
  total = 10 → p_yellow = 6/10 → (total : ℚ) * (1 - p_yellow) = 4 := by sorry

end white_balls_count_l568_56814


namespace diagonal_intersection_fixed_point_l568_56868

/-- An ellipse with equation x^2/4 + y^2/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- A point is on the ellipse C -/
def on_ellipse_C (p : ℝ × ℝ) : Prop := ellipse_C p.1 p.2

/-- Quadrilateral MNPQ with vertices on ellipse C -/
structure Quadrilateral_MNPQ where
  M : ℝ × ℝ
  N : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  hM : on_ellipse_C M
  hN : on_ellipse_C N
  hP : on_ellipse_C P
  hQ : on_ellipse_C Q
  hMQ_NP : M.2 + Q.2 = 0 ∧ N.2 + P.2 = 0  -- MQ || NP and MQ ⊥ x-axis
  hS : ∃ t : ℝ, (M.2 - N.2) * (4 - P.1) = (M.1 - 4) * (P.2 - N.2) ∧
                (Q.2 - P.2) * (4 - N.1) = (Q.1 - 4) * (N.2 - P.2)  -- MN and QP intersect at S(4,0)

/-- The theorem to be proved -/
theorem diagonal_intersection_fixed_point (q : Quadrilateral_MNPQ) :
  ∃ (I : ℝ × ℝ), I = (1, 0) ∧
  (q.M.2 - q.P.2) * (I.1 - q.N.1) = (q.M.1 - I.1) * (I.2 - q.N.2) ∧
  (q.N.2 - q.Q.2) * (I.1 - q.M.1) = (q.N.1 - I.1) * (I.2 - q.M.2) := by
  sorry

end diagonal_intersection_fixed_point_l568_56868


namespace line_equivalence_l568_56860

/-- Given a line expressed in vector form, prove it's equivalent to a specific slope-intercept form -/
theorem line_equivalence (x y : ℝ) : 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-4)) = 0 ↔ y = 2 * x - 10 := by sorry

end line_equivalence_l568_56860


namespace valid_window_exists_l568_56837

/-- A region in the window --/
structure Region where
  area : ℝ
  sides_equal : Bool

/-- A window configuration --/
structure Window where
  side_length : ℝ
  regions : List Region

/-- Checks if a window configuration is valid --/
def is_valid_window (w : Window) : Prop :=
  w.side_length = 1 ∧
  w.regions.length = 8 ∧
  w.regions.all (fun r => r.area = 1 / 8) ∧
  w.regions.all (fun r => r.sides_equal)

/-- Theorem: There exists a valid window configuration --/
theorem valid_window_exists : ∃ w : Window, is_valid_window w := by
  sorry


end valid_window_exists_l568_56837


namespace tarun_departure_time_l568_56884

theorem tarun_departure_time 
  (total_work : ℝ) 
  (combined_rate : ℝ) 
  (arun_rate : ℝ) 
  (remaining_days : ℝ) :
  combined_rate = total_work / 10 →
  arun_rate = total_work / 30 →
  remaining_days = 18 →
  ∃ (x : ℝ), x * combined_rate + remaining_days * arun_rate = total_work ∧ x = 4 := by
sorry

end tarun_departure_time_l568_56884


namespace five_from_six_circular_seating_l568_56851

/-- The number of ways to seat 5 people from a group of 6 around a circular table -/
def circular_seating_arrangements (total_people : ℕ) (seated_people : ℕ) : ℕ :=
  (total_people.choose seated_people) * (seated_people - 1).factorial

/-- Theorem stating that the number of ways to seat 5 people from a group of 6 around a circular table is 144 -/
theorem five_from_six_circular_seating :
  circular_seating_arrangements 6 5 = 144 := by
  sorry

end five_from_six_circular_seating_l568_56851


namespace smallest_divisor_property_solution_set_l568_56863

def smallest_divisor (n : ℕ) : ℕ :=
  (Nat.factors n).head!

theorem smallest_divisor_property (n : ℕ) : 
  n > 1 → smallest_divisor n > 1 ∧ n % smallest_divisor n = 0 := by sorry

theorem solution_set : 
  {n : ℕ | n + smallest_divisor n = 30} = {25, 27, 28} := by sorry

end smallest_divisor_property_solution_set_l568_56863


namespace inequality_proof_l568_56835

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) : 
  (a * b / (a^5 + a * b + b^5)) + 
  (b * c / (b^5 + b * c + c^5)) + 
  (c * a / (c^5 + c * a + a^5)) ≤ 1 := by
sorry

end inequality_proof_l568_56835


namespace distance_focus_to_asymptote_l568_56849

/-- The distance from the focus of the parabola x = (1/4)y^2 to the asymptote of the hyperbola x^2 - (y^2/3) = 1 is √3/2 -/
theorem distance_focus_to_asymptote :
  let focus : ℝ × ℝ := (1, 0)
  let asymptote (x : ℝ) : ℝ := Real.sqrt 3 * x
  let distance_point_to_line (p : ℝ × ℝ) (f : ℝ → ℝ) : ℝ :=
    |f p.1 - p.2| / Real.sqrt (1 + (Real.sqrt 3)^2)
  distance_point_to_line focus asymptote = Real.sqrt 3 / 2 := by
  sorry


end distance_focus_to_asymptote_l568_56849


namespace pet_store_combinations_l568_56853

/-- The number of puppies available -/
def num_puppies : ℕ := 10

/-- The number of kittens available -/
def num_kittens : ℕ := 7

/-- The number of hamsters available -/
def num_hamsters : ℕ := 9

/-- The number of birds available -/
def num_birds : ℕ := 5

/-- The number of people buying pets -/
def num_people : ℕ := 4

/-- The number of ways to select one pet of each type and assign them to four different people -/
def num_ways : ℕ := num_puppies * num_kittens * num_hamsters * num_birds * Nat.factorial num_people

theorem pet_store_combinations : num_ways = 75600 := by
  sorry

end pet_store_combinations_l568_56853


namespace arithmetic_sequence_sum_l568_56815

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (∀ n : ℕ+, a n * b n = 2 * n^2 - n) →
  5 * a 4 = 7 * a 3 →
  a 1 + b 1 = 2 →
  a 9 + b 10 = 27 := by
  sorry

end arithmetic_sequence_sum_l568_56815


namespace function_f_properties_l568_56870

/-- A function satisfying the given properties -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y) ∧
  (∀ x, x > 0 → f x > 0)

theorem function_f_properties (f : ℝ → ℝ) (hf : FunctionF f) :
  (f 0 = 0) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, x < y → f x < f y) ∧
  (f 1 = 2 → ∃ a, f (2 - a) = 6 ∧ a = -1) := by
  sorry

end function_f_properties_l568_56870


namespace ghee_composition_l568_56805

theorem ghee_composition (original_quantity : ℝ) (vanaspati_percentage : ℝ) 
  (added_pure_ghee : ℝ) (new_vanaspati_percentage : ℝ) :
  original_quantity = 10 →
  vanaspati_percentage = 40 →
  added_pure_ghee = 10 →
  new_vanaspati_percentage = 20 →
  (vanaspati_percentage / 100) * original_quantity = 
    (new_vanaspati_percentage / 100) * (original_quantity + added_pure_ghee) →
  (100 - vanaspati_percentage) = 60 := by
sorry

end ghee_composition_l568_56805


namespace cubic_root_sum_cubes_l568_56893

/-- Given a cubic equation 5x^3 + 500x + 3005 = 0 with roots a, b, and c,
    prove that (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 -/
theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (5 * a^3 + 500 * a + 3005 = 0) →
  (5 * b^3 + 500 * b + 3005 = 0) →
  (5 * c^3 + 500 * c + 3005 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 := by sorry

end cubic_root_sum_cubes_l568_56893
