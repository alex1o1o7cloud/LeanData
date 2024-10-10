import Mathlib

namespace square_intersection_dot_product_l1038_103855

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 3) ∧ B = (3, 3) ∧ C = (3, 0) ∧ D = (0, 0)

-- Define point E as the midpoint of DC
def Midpoint (E D C : ℝ × ℝ) : Prop :=
  E = ((D.1 + C.1) / 2, (D.2 + C.2) / 2)

-- Define the intersection point F
def Intersection (F : ℝ × ℝ) (A E B D : ℝ × ℝ) : Prop :=
  (F.2 = -2 * F.1 + 3) ∧ (F.2 = F.1)

-- Define the dot product of two 2D vectors
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Main theorem
theorem square_intersection_dot_product 
  (A B C D E F : ℝ × ℝ) : 
  Square A B C D → 
  Midpoint E D C → 
  Intersection F A E B D → 
  DotProduct (F.1 - D.1, F.2 - D.2) (E.1 - D.1, E.2 - D.2) = -3/2 := by
  sorry

end square_intersection_dot_product_l1038_103855


namespace product_remainder_mod_17_l1038_103813

theorem product_remainder_mod_17 : (2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 := by
  sorry

end product_remainder_mod_17_l1038_103813


namespace continuous_at_two_l1038_103801

/-- The function f(x) = -4x^2 - 8 is continuous at x₀ = 2 -/
theorem continuous_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |(-4*x^2 - 8) - (-4*2^2 - 8)| < ε :=
by sorry

end continuous_at_two_l1038_103801


namespace point_coordinate_sum_l1038_103898

/-- Given points A and B, where A is at (0, 0) and B is on the line y = 3,
    if the slope of segment AB is 3/4, then the sum of the x- and y-coordinates of B is 7. -/
theorem point_coordinate_sum (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  (3 - 0) / (x - 0) = 3 / 4 → x + 3 = 7 := by
  sorry

end point_coordinate_sum_l1038_103898


namespace max_value_of_f_l1038_103870

def f (x : ℝ) : ℝ := x^2 + 4*x + 1

theorem max_value_of_f :
  ∃ (m : ℝ), m = 4 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ m :=
sorry

end max_value_of_f_l1038_103870


namespace necessary_condition_for_inequality_l1038_103875

theorem necessary_condition_for_inequality (a b c : ℝ) (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by
  sorry

end necessary_condition_for_inequality_l1038_103875


namespace abc_inequality_l1038_103829

noncomputable def a : ℝ := 2 / Real.log 2
noncomputable def b : ℝ := Real.exp 2 / (4 - Real.log 4)
noncomputable def c : ℝ := 2 * Real.sqrt (Real.exp 1)

theorem abc_inequality : c > a ∧ a > b := by sorry

end abc_inequality_l1038_103829


namespace class_8_3_final_score_l1038_103887

/-- The final score of a choir competition is calculated based on three categories:
    singing quality, spirit, and coordination. Each category has a specific weight
    in the final score calculation. -/
def final_score (singing_quality : ℝ) (spirit : ℝ) (coordination : ℝ)
                (singing_weight : ℝ) (spirit_weight : ℝ) (coordination_weight : ℝ) : ℝ :=
  singing_quality * singing_weight + spirit * spirit_weight + coordination * coordination_weight

/-- Theorem stating that the final score of Class 8-3 in the choir competition is 81.8 points -/
theorem class_8_3_final_score :
  final_score 92 80 70 0.4 0.3 0.3 = 81.8 := by
  sorry

end class_8_3_final_score_l1038_103887


namespace snake_count_l1038_103834

theorem snake_count (breeding_balls : Nat) (snake_pairs : Nat) (total_snakes : Nat) :
  breeding_balls = 3 →
  snake_pairs = 6 →
  total_snakes = 36 →
  ∃ snakes_per_ball : Nat, snakes_per_ball * breeding_balls + snake_pairs * 2 = total_snakes ∧ snakes_per_ball = 8 := by
  sorry

end snake_count_l1038_103834


namespace pastry_count_consistency_l1038_103889

/-- Represents the number of pastries in different states --/
structure Pastries where
  initial : ℕ
  sold : ℕ
  remaining : ℕ

/-- The problem statement --/
theorem pastry_count_consistency (p : Pastries) 
  (h1 : p.initial = 148)
  (h2 : p.sold = 103)
  (h3 : p.remaining = 45) :
  p.initial = p.sold + p.remaining := by
  sorry

end pastry_count_consistency_l1038_103889


namespace min_production_avoid_losses_l1038_103810

/-- The minimum daily production of gloves to avoid losses -/
def min_production : ℕ := 800

/-- The total daily production cost (in yuan) as a function of daily production volume (in pairs) -/
def total_cost (x : ℕ) : ℕ := 5 * x + 4000

/-- The factory price per pair of gloves (in yuan) -/
def price_per_pair : ℕ := 10

/-- The daily revenue (in yuan) as a function of daily production volume (in pairs) -/
def revenue (x : ℕ) : ℕ := price_per_pair * x

/-- Theorem stating that the minimum daily production to avoid losses is 800 pairs -/
theorem min_production_avoid_losses :
  ∀ x : ℕ, x ≥ min_production ↔ revenue x ≥ total_cost x :=
sorry

end min_production_avoid_losses_l1038_103810


namespace car_wash_contribution_l1038_103859

def goal : ℕ := 150
def families_with_known_contribution : ℕ := 15
def known_contribution_per_family : ℕ := 5
def remaining_families : ℕ := 3
def amount_needed : ℕ := 45

theorem car_wash_contribution :
  ∀ (contribution_per_remaining_family : ℕ),
    (families_with_known_contribution * known_contribution_per_family) +
    (remaining_families * contribution_per_remaining_family) =
    goal - amount_needed →
    contribution_per_remaining_family = 10 := by
  sorry

end car_wash_contribution_l1038_103859


namespace largest_multiple_of_15_less_than_450_l1038_103846

theorem largest_multiple_of_15_less_than_450 :
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 450 → n ≤ 435 :=
by sorry

end largest_multiple_of_15_less_than_450_l1038_103846


namespace problem_solution_l1038_103814

theorem problem_solution (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  x = 0.5 := by
sorry

end problem_solution_l1038_103814


namespace heptagon_angle_sums_l1038_103803

/-- A heptagon is a polygon with 7 sides -/
def Heptagon : Nat := 7

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : Nat) : ℝ := (n - 2) * 180

/-- The sum of exterior angles of any polygon -/
def sum_exterior_angles : ℝ := 360

theorem heptagon_angle_sums :
  (sum_interior_angles Heptagon = 900) ∧ (sum_exterior_angles = 360) := by
  sorry

#check heptagon_angle_sums

end heptagon_angle_sums_l1038_103803


namespace shells_found_l1038_103893

def initial_shells : ℕ := 68
def final_shells : ℕ := 89

theorem shells_found (initial : ℕ) (final : ℕ) (h1 : initial = initial_shells) (h2 : final = final_shells) :
  final - initial = 21 := by
  sorry

end shells_found_l1038_103893


namespace petya_running_time_l1038_103897

theorem petya_running_time (a V : ℝ) (h1 : a > 0) (h2 : V > 0) :
  (a / (2.5 * V) + a / (1.6 * V)) > (a / V) := by
  sorry

end petya_running_time_l1038_103897


namespace line_through_quadrants_line_through_fixed_point_point_slope_form_slope_intercept_form_l1038_103844

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- 1. Line passing through first, second, and fourth quadrants
theorem line_through_quadrants (l : Line) :
  (∃ x y, x > 0 ∧ y > 0 ∧ y = l.slope * x + l.intercept) ∧
  (∃ x y, x < 0 ∧ y > 0 ∧ y = l.slope * x + l.intercept) ∧
  (∃ x y, x > 0 ∧ y < 0 ∧ y = l.slope * x + l.intercept) →
  l.slope < 0 ∧ l.intercept > 0 :=
sorry

-- 2. Line passing through a fixed point
theorem line_through_fixed_point (k : ℝ) :
  ∃ x y, k * x - y - 2 * k + 3 = 0 ∧ x = 2 ∧ y = 3 :=
sorry

-- 3. Point-slope form equation
theorem point_slope_form (p : Point) (m : ℝ) :
  p.x = 2 ∧ p.y = -1 ∧ m = -Real.sqrt 3 →
  ∀ x y, y + 1 = -Real.sqrt 3 * (x - 2) ↔ y - p.y = m * (x - p.x) :=
sorry

-- 4. Slope-intercept form equation
theorem slope_intercept_form (l : Line) :
  l.slope = -2 ∧ l.intercept = 3 →
  ∀ x y, y = l.slope * x + l.intercept ↔ y = -2 * x + 3 :=
sorry

end line_through_quadrants_line_through_fixed_point_point_slope_form_slope_intercept_form_l1038_103844


namespace polynomial_identity_sum_l1038_103880

theorem polynomial_identity_sum (a₀ a₁ a₂ a₃ : ℝ) 
  (h : ∀ x : ℝ, 1 + x + x^2 + x^3 = a₀ + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3) : 
  a₁ + a₂ + a₃ = -3 := by
  sorry

end polynomial_identity_sum_l1038_103880


namespace cool_drink_solution_volume_l1038_103851

/-- Represents the cool-drink solution problem --/
theorem cool_drink_solution_volume 
  (initial_jasmine_percent : Real)
  (added_jasmine : Real)
  (added_water : Real)
  (final_jasmine_percent : Real)
  (h1 : initial_jasmine_percent = 0.05)
  (h2 : added_jasmine = 8)
  (h3 : added_water = 2)
  (h4 : final_jasmine_percent = 0.125)
  : ∃ (initial_volume : Real),
    initial_volume * initial_jasmine_percent + added_jasmine = 
    (initial_volume + added_jasmine + added_water) * final_jasmine_percent ∧
    initial_volume = 90 := by
  sorry

end cool_drink_solution_volume_l1038_103851


namespace trapezoid_upper_side_length_l1038_103805

/-- Theorem: For a trapezoid with a base of 25 cm, a height of 13 cm, and an area of 286 cm²,
    the length of the upper side is 19 cm. -/
theorem trapezoid_upper_side_length 
  (base : ℝ) (height : ℝ) (area : ℝ) (upper_side : ℝ) 
  (h1 : base = 25) 
  (h2 : height = 13) 
  (h3 : area = 286) 
  (h4 : area = (1/2) * (base + upper_side) * height) : 
  upper_side = 19 := by
  sorry

#check trapezoid_upper_side_length

end trapezoid_upper_side_length_l1038_103805


namespace function_satisfies_conditions_l1038_103811

-- Define the function f
def f (x : ℤ) : ℤ := x^3 - 3*x^2 + 5*x + 9

-- State the theorem
theorem function_satisfies_conditions : 
  f 3 = 12 ∧ f 4 = 22 ∧ f 5 = 36 ∧ f 6 = 54 ∧ f 7 = 76 := by
  sorry

end function_satisfies_conditions_l1038_103811


namespace proof_by_contradiction_principle_l1038_103878

theorem proof_by_contradiction_principle :
  ∀ (P : Prop), (¬P → False) → P :=
by
  sorry

end proof_by_contradiction_principle_l1038_103878


namespace pairings_equal_25_l1038_103882

/-- The number of bowls and glasses -/
def n : ℕ := 5

/-- The total number of possible pairings of bowls and glasses -/
def total_pairings : ℕ := n * n

/-- Theorem stating that the total number of pairings is 25 -/
theorem pairings_equal_25 : total_pairings = 25 := by
  sorry

end pairings_equal_25_l1038_103882


namespace sqrt_x_minus_2_meaningful_l1038_103817

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
sorry

end sqrt_x_minus_2_meaningful_l1038_103817


namespace completing_square_addition_l1038_103818

theorem completing_square_addition (x : ℝ) : 
  (∃ k : ℝ, (x^2 - 4*x + k)^(1/2) = x - 2) → k = 4 := by
  sorry

end completing_square_addition_l1038_103818


namespace total_legs_calculation_l1038_103809

theorem total_legs_calculation (total_tables : ℕ) (four_legged_tables : ℕ) 
  (h1 : total_tables = 36)
  (h2 : four_legged_tables = 16)
  (h3 : four_legged_tables ≤ total_tables) :
  four_legged_tables * 4 + (total_tables - four_legged_tables) * 3 = 124 := by
  sorry

#check total_legs_calculation

end total_legs_calculation_l1038_103809


namespace circle_radius_nine_iff_k_94_l1038_103861

/-- The equation of a circle in general form --/
def circle_equation (x y k : ℝ) : Prop :=
  2 * x^2 + 20 * x + 3 * y^2 + 18 * y - k = 0

/-- The equation of a circle in standard form with center (h, k) and radius r --/
def standard_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the given equation represents a circle with radius 9 iff k = 94 --/
theorem circle_radius_nine_iff_k_94 :
  (∃ h k : ℝ, ∀ x y : ℝ, circle_equation x y 94 ↔ standard_circle_equation x y h k 9) ↔
  (∀ k : ℝ, (∃ h k : ℝ, ∀ x y : ℝ, circle_equation x y k ↔ standard_circle_equation x y h k 9) → k = 94) :=
sorry

end circle_radius_nine_iff_k_94_l1038_103861


namespace arrangements_theorem_l1038_103881

def number_of_arrangements (n : ℕ) (red yellow blue : ℕ) : ℕ :=
  sorry

theorem arrangements_theorem :
  number_of_arrangements 5 2 2 1 = 48 := by
  sorry

end arrangements_theorem_l1038_103881


namespace exp_iff_gt_l1038_103820

-- Define the exponential function as monotonically increasing on ℝ
axiom exp_monotone : ∀ (x y : ℝ), x < y → Real.exp x < Real.exp y

theorem exp_iff_gt (a b : ℝ) : a > b ↔ Real.exp a > Real.exp b := by
  sorry

end exp_iff_gt_l1038_103820


namespace log_inequality_l1038_103840

theorem log_inequality (x : ℝ) (h : 0 < x) (h' : x < 1) : Real.log (1 + x) > x^3 / 3 := by
  sorry

end log_inequality_l1038_103840


namespace clown_count_l1038_103839

/-- The number of clown mobiles -/
def num_mobiles : ℕ := 5

/-- The number of clowns in each mobile -/
def clowns_per_mobile : ℕ := 28

/-- The total number of clowns in all mobiles -/
def total_clowns : ℕ := num_mobiles * clowns_per_mobile

theorem clown_count : total_clowns = 140 := by
  sorry

end clown_count_l1038_103839


namespace unique_intersection_condition_l1038_103827

/-- A function f(x) = kx^2 + 2(k+1)x + k-1 has only one intersection point with the x-axis if and only if k = 0 or k = -1/3 -/
theorem unique_intersection_condition (k : ℝ) : 
  (∃! x, k * x^2 + 2*(k+1)*x + (k-1) = 0) ↔ (k = 0 ∨ k = -1/3) := by
sorry

end unique_intersection_condition_l1038_103827


namespace min_value_constrained_l1038_103841

/-- Given that x + 2y + 3z = 1, the minimum value of x^2 + y^2 + z^2 is 1/14 -/
theorem min_value_constrained (x y z : ℝ) (h : x + 2*y + 3*z = 1) :
  ∃ (min : ℝ), min = (1 : ℝ) / 14 ∧ 
  ∀ (a b c : ℝ), a + 2*b + 3*c = 1 → x^2 + y^2 + z^2 ≥ min :=
sorry

end min_value_constrained_l1038_103841


namespace joan_apples_l1038_103867

/-- The number of apples Joan has after picking and giving some away -/
def apples_remaining (picked : ℕ) (given_away : ℕ) : ℕ :=
  picked - given_away

/-- Theorem: Joan has 16 apples after picking 43 and giving away 27 -/
theorem joan_apples : apples_remaining 43 27 = 16 := by
  sorry

end joan_apples_l1038_103867


namespace jerrys_painting_time_l1038_103869

theorem jerrys_painting_time (fixing_time painting_time mowing_time hourly_rate total_payment : ℝ) :
  fixing_time = 3 * painting_time →
  mowing_time = 6 →
  hourly_rate = 15 →
  total_payment = 570 →
  hourly_rate * (painting_time + fixing_time + mowing_time) = total_payment →
  painting_time = 8 := by
  sorry

end jerrys_painting_time_l1038_103869


namespace min_buses_needed_l1038_103892

/-- The number of students to be transported -/
def total_students : ℕ := 540

/-- The maximum number of students each bus can hold -/
def bus_capacity : ℕ := 45

/-- The minimum number of buses needed is the ceiling of the quotient of total students divided by bus capacity -/
theorem min_buses_needed : 
  (total_students + bus_capacity - 1) / bus_capacity = 12 := by sorry

end min_buses_needed_l1038_103892


namespace luncheon_no_shows_l1038_103854

/-- Theorem: Number of no-shows at a luncheon --/
theorem luncheon_no_shows 
  (invited : ℕ) 
  (tables : ℕ) 
  (people_per_table : ℕ) 
  (h1 : invited = 47) 
  (h2 : tables = 8) 
  (h3 : people_per_table = 5) : 
  invited - (tables * people_per_table) = 7 := by
  sorry

#check luncheon_no_shows

end luncheon_no_shows_l1038_103854


namespace solution_l1038_103890

def problem (B : ℕ) (A : ℕ) (X : ℕ) : Prop :=
  B = 38 ∧ 
  A = B + 8 ∧ 
  A + 10 = 2 * (B - X)

theorem solution : ∃ X, problem 38 (38 + 8) X ∧ X = 10 := by
  sorry

end solution_l1038_103890


namespace tan_identity_l1038_103899

theorem tan_identity (α : ℝ) (h : Real.tan (π / 3 - α) = 1 / 3) :
  Real.tan (2 * π / 3 + α) = -1 / 3 := by
  sorry

end tan_identity_l1038_103899


namespace bluegrass_percentage_in_x_l1038_103806

/-- Represents a seed mixture with percentages of different grass types -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
def finalMixture (x y : SeedMixture) (xProportion : ℝ) : SeedMixture :=
  { ryegrass := x.ryegrass * xProportion + y.ryegrass * (1 - xProportion)
  , bluegrass := x.bluegrass * xProportion + y.bluegrass * (1 - xProportion)
  , fescue := x.fescue * xProportion + y.fescue * (1 - xProportion) }

theorem bluegrass_percentage_in_x 
  (x : SeedMixture) 
  (y : SeedMixture)
  (h1 : x.ryegrass = 0.4)
  (h2 : x.ryegrass + x.bluegrass = 1)
  (h3 : y.ryegrass = 0.25)
  (h4 : y.fescue = 0.75)
  (h5 : (finalMixture x y (1/3)).ryegrass = 0.3) :
  x.bluegrass = 0.6 := by
sorry

end bluegrass_percentage_in_x_l1038_103806


namespace only_traffic_light_is_random_l1038_103837

/-- Represents a phenomenon that can be observed --/
inductive Phenomenon
  | WaterBoiling : Phenomenon
  | TrafficLight : Phenomenon
  | RectangleArea : Phenomenon
  | LinearEquation : Phenomenon

/-- Determines if a phenomenon is random --/
def isRandom (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.TrafficLight => True
  | _ => False

/-- Theorem stating that only the traffic light phenomenon is random --/
theorem only_traffic_light_is_random :
  ∀ (p : Phenomenon), isRandom p ↔ p = Phenomenon.TrafficLight := by
  sorry


end only_traffic_light_is_random_l1038_103837


namespace circle_center_coordinates_l1038_103835

/-- The center of a circle tangent to two parallel lines and lying on a third line -/
theorem circle_center_coordinates (x y : ℚ) : 
  (∃ (r : ℚ), r > 0 ∧ 
    (∀ (x' y' : ℚ), (x' - x)^2 + (y' - y)^2 = r^2 → 
      (3*x' + 4*y' = 24 ∨ 3*x' + 4*y' = -16))) → 
  x - 3*y = 0 → 
  (x, y) = (12/13, 4/13) := by
sorry

end circle_center_coordinates_l1038_103835


namespace accidental_division_correction_l1038_103896

theorem accidental_division_correction (x : ℝ) : 
  x / 15 = 6 → x * 15 = 1350 := by
  sorry

end accidental_division_correction_l1038_103896


namespace child_ticket_price_l1038_103833

theorem child_ticket_price (total_revenue : ℕ) (adult_price : ℕ) (total_tickets : ℕ) (child_tickets : ℕ) :
  total_revenue = 104 →
  adult_price = 6 →
  total_tickets = 21 →
  child_tickets = 11 →
  ∃ (child_price : ℕ), child_price * child_tickets + adult_price * (total_tickets - child_tickets) = total_revenue ∧ child_price = 4 :=
by sorry

end child_ticket_price_l1038_103833


namespace rotated_point_x_coordinate_l1038_103857

theorem rotated_point_x_coordinate 
  (P : ℝ × ℝ) 
  (h_unit_circle : P.1^2 + P.2^2 = 1) 
  (h_P : P = (4/5, -3/5)) : 
  let Q := (
    P.1 * Real.cos (π/3) - P.2 * Real.sin (π/3),
    P.1 * Real.sin (π/3) + P.2 * Real.cos (π/3)
  )
  Q.1 = (4 + 3 * Real.sqrt 3) / 10 := by
sorry

end rotated_point_x_coordinate_l1038_103857


namespace part_one_part_two_l1038_103843

-- Define propositions p and q
def p (k : ℝ) : Prop := k^2 - 2*k - 24 ≤ 0

def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b < 0 ∧ a = 3 - k ∧ b = 3 + k

-- Part 1
theorem part_one (k : ℝ) : q k → k ∈ Set.Iio (-3) := by
  sorry

-- Part 2
theorem part_two (k : ℝ) : (p k ∨ q k) ∧ ¬(p k ∧ q k) → k ∈ Set.Iio (-4) ∪ Set.Icc (-3) 6 := by
  sorry

end part_one_part_two_l1038_103843


namespace eraser_cost_proof_l1038_103879

/-- Represents the price of a single pencil -/
def pencil_price : ℝ := 2

/-- Represents the price of a single eraser -/
def eraser_price : ℝ := 1

/-- The number of pencils sold -/
def pencils_sold : ℕ := 20

/-- The number of erasers sold -/
def erasers_sold : ℕ := pencils_sold * 2

/-- The total revenue from sales -/
def total_revenue : ℝ := 80

theorem eraser_cost_proof :
  (pencils_sold : ℝ) * pencil_price + (erasers_sold : ℝ) * eraser_price = total_revenue ∧
  2 * eraser_price = pencil_price ∧
  eraser_price = 1 :=
by sorry

end eraser_cost_proof_l1038_103879


namespace min_value_sum_reciprocals_l1038_103815

/-- Given a line ax - by + 1 = 0 (where a > 0 and b > 0) passing through the center of the circle
    x^2 + y^2 + 2x - 4y + 1 = 0, the minimum value of 1/a + 1/b is 3 + 2√2. -/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : a * (-1) - b * 2 + 1 = 0) : 
    (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' * (-1) - b' * 2 + 1 = 0 → 1 / a + 1 / b ≤ 1 / a' + 1 / b') → 
    1 / a + 1 / b = 3 + 2 * Real.sqrt 2 := by
  sorry

end min_value_sum_reciprocals_l1038_103815


namespace betty_doug_age_sum_l1038_103828

/-- The sum of Betty's and Doug's ages given the conditions of the problem -/
theorem betty_doug_age_sum : ∀ (betty_age : ℕ) (doug_age : ℕ),
  doug_age = 40 →
  2 * betty_age * 20 = 2000 →
  betty_age + doug_age = 90 := by
  sorry

end betty_doug_age_sum_l1038_103828


namespace polynomial_roots_l1038_103888

theorem polynomial_roots : ∃ (a b : ℝ), 
  (∀ x : ℝ, 6*x^4 + 25*x^3 - 59*x^2 + 28*x = 0 ↔ 
    x = 0 ∨ x = 1 ∨ x = (-31 + Real.sqrt 1633) / 12 ∨ x = (-31 - Real.sqrt 1633) / 12) ∧
  a = (-31 + Real.sqrt 1633) / 12 ∧
  b = (-31 - Real.sqrt 1633) / 12 := by
  sorry

end polynomial_roots_l1038_103888


namespace problem_statement_l1038_103816

theorem problem_statement (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 7) : 
  x^2*y + x*y^2 = 245/36 := by
sorry

end problem_statement_l1038_103816


namespace competition_problem_l1038_103849

/-- Represents the number of students who solved each combination of problems -/
structure ProblemSolvers where
  onlyA : ℕ
  onlyB : ℕ
  onlyC : ℕ
  AB : ℕ
  AC : ℕ
  BC : ℕ
  ABC : ℕ

/-- The theorem statement -/
theorem competition_problem (s : ProblemSolvers) : s.onlyB = 6 :=
  by
  have total : s.onlyA + s.onlyB + s.onlyC + s.AB + s.AC + s.BC + s.ABC = 25 := by sorry
  have solved_A : s.onlyA = s.AB + s.AC + s.ABC + 1 := by sorry
  have not_A_BC : s.onlyB + s.BC = 2 * (s.onlyC + s.BC) := by sorry
  have only_one_not_A : s.onlyB + s.onlyC = s.onlyA := by sorry
  sorry

end competition_problem_l1038_103849


namespace length_PF_is_16_over_3_l1038_103874

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*(x+2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 0)

-- Define the line through the focus
def line_through_focus (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define the intersection points A and B (we don't calculate them explicitly)
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  line_through_focus A.1 A.2 ∧ line_through_focus B.1 B.2

-- Define point P on x-axis
def point_P (P : ℝ × ℝ) : Prop := P.2 = 0

-- Main theorem
theorem length_PF_is_16_over_3 
  (A B P : ℝ × ℝ) 
  (h_intersect : intersection_points A B)
  (h_P : point_P P)
  (h_perpendicular : sorry) -- Additional hypothesis for P being on the perpendicular bisector
  : ‖P - focus‖ = 16/3 :=
sorry

end length_PF_is_16_over_3_l1038_103874


namespace third_term_of_arithmetic_sequence_l1038_103894

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmeticSequence a)
  (h_first : a 1 = 2)
  (h_second : a 2 = 8) :
  a 3 = 14 := by
  sorry

end third_term_of_arithmetic_sequence_l1038_103894


namespace valid_book_pairs_18_4_l1038_103836

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of different pairs of books that can be chosen from a collection of books,
    given the total number of books and the number of books in a series,
    with the restriction that two books from the series cannot be chosen together. -/
def validBookPairs (totalBooks seriesBooks : ℕ) : ℕ :=
  choose totalBooks 2 - choose seriesBooks 2

theorem valid_book_pairs_18_4 :
  validBookPairs 18 4 = 147 := by sorry

end valid_book_pairs_18_4_l1038_103836


namespace apple_cost_price_l1038_103866

theorem apple_cost_price (SP : ℝ) (CP : ℝ) : SP = 16 ∧ SP = (5/6) * CP → CP = 19.2 := by
  sorry

end apple_cost_price_l1038_103866


namespace complementary_angle_proof_l1038_103860

-- Define complementary angles
def complementary (angle1 angle2 : ℝ) : Prop :=
  angle1 + angle2 = 90

-- Theorem statement
theorem complementary_angle_proof (angle1 angle2 : ℝ) 
  (h1 : complementary angle1 angle2) (h2 : angle1 = 25) : 
  angle2 = 65 := by
  sorry


end complementary_angle_proof_l1038_103860


namespace late_attendees_fraction_l1038_103819

theorem late_attendees_fraction 
  (total : ℕ) 
  (total_pos : total > 0)
  (male_fraction : Rat)
  (male_on_time_fraction : Rat)
  (female_on_time_fraction : Rat)
  (h_male : male_fraction = 2 / 3)
  (h_male_on_time : male_on_time_fraction = 3 / 4)
  (h_female_on_time : female_on_time_fraction = 5 / 6) :
  (1 : Rat) - (male_fraction * male_on_time_fraction + (1 - male_fraction) * female_on_time_fraction) = 2 / 9 := by
  sorry

#check late_attendees_fraction

end late_attendees_fraction_l1038_103819


namespace parallelogram_area_12_8_l1038_103856

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 12 cm and height 8 cm is 96 square centimeters -/
theorem parallelogram_area_12_8 : parallelogramArea 12 8 = 96 := by
  sorry

end parallelogram_area_12_8_l1038_103856


namespace min_value_of_f_l1038_103886

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else x + 6/x - 7

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 2 * Real.sqrt 6 - 7 := by
  sorry

end min_value_of_f_l1038_103886


namespace multiply_and_add_l1038_103883

theorem multiply_and_add : 42 * 52 + 48 * 42 = 4200 := by
  sorry

end multiply_and_add_l1038_103883


namespace relationship_abc_l1038_103812

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem relationship_abc : 
  let a := base_to_decimal [1, 2] 16
  let b := base_to_decimal [2, 5] 7
  let c := base_to_decimal [3, 3] 4
  c < a ∧ a < b := by sorry

end relationship_abc_l1038_103812


namespace journalist_selection_theorem_l1038_103824

-- Define the number of domestic and foreign journalists
def domestic_journalists : ℕ := 5
def foreign_journalists : ℕ := 4

-- Define the total number of journalists to be selected
def selected_journalists : ℕ := 3

-- Function to calculate the number of ways to select and arrange journalists
def select_and_arrange_journalists : ℕ := sorry

-- Theorem stating the correct number of ways
theorem journalist_selection_theorem : 
  select_and_arrange_journalists = 260 := by sorry

end journalist_selection_theorem_l1038_103824


namespace intersection_when_a_is_neg_two_intersection_equals_A_iff_l1038_103850

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x + a < 0}

-- Theorem for part (1)
theorem intersection_when_a_is_neg_two :
  A ∩ B (-2) = {x : ℝ | 1/2 ≤ x ∧ x < 2} := by sorry

-- Theorem for part (2)
theorem intersection_equals_A_iff (a : ℝ) :
  A ∩ B a = A ↔ a < -3 := by sorry

end intersection_when_a_is_neg_two_intersection_equals_A_iff_l1038_103850


namespace m_is_positive_l1038_103862

-- Define the sets M and N
def M (m : ℝ) : Set ℝ := {x | x ≤ m}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}

-- State the theorem
theorem m_is_positive (m : ℝ) (h : (M m) ∩ N ≠ ∅) : m > 0 := by
  sorry

end m_is_positive_l1038_103862


namespace rectangular_to_polar_conversion_l1038_103821

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 2 * Real.sqrt 6
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ r = 2 * Real.sqrt 22 ∧ θ = Real.arctan (Real.sqrt 6 / 4) := by
  sorry

end rectangular_to_polar_conversion_l1038_103821


namespace gcd_factorial_problem_l1038_103852

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 2520 := by
  sorry

end gcd_factorial_problem_l1038_103852


namespace divisor_condition_l1038_103884

theorem divisor_condition (k : ℕ+) :
  (∃ (n : ℕ+), (8 * k * n - 1) ∣ (4 * k^2 - 1)^2) ↔ Even k :=
sorry

end divisor_condition_l1038_103884


namespace cube_volume_from_surface_area_l1038_103823

/-- Given a cube with surface area 24 square centimeters, its volume is 8 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (side_length : ℝ),
  (6 * side_length^2 = 24) →
  side_length^3 = 8 := by
sorry

end cube_volume_from_surface_area_l1038_103823


namespace pet_store_cats_l1038_103822

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℝ := 5.0

/-- The number of Siamese cats initially in the pet store -/
def initial_siamese_cats : ℝ := 13.0

/-- The number of cats added during the purchase -/
def added_cats : ℝ := 10.0

/-- The total number of cats after the addition -/
def total_cats_after : ℝ := 28.0

theorem pet_store_cats :
  initial_house_cats + initial_siamese_cats + added_cats = total_cats_after :=
by sorry

end pet_store_cats_l1038_103822


namespace factorization_problem_1_l1038_103873

theorem factorization_problem_1 (x y : ℝ) : x * y - x + y - 1 = (x + 1) * (y - 1) := by
  sorry

end factorization_problem_1_l1038_103873


namespace skittles_distribution_l1038_103802

/-- Given 25 Skittles distributed among 5 people, prove that each person receives 5 Skittles. -/
theorem skittles_distribution (total_skittles : ℕ) (num_people : ℕ) (skittles_per_person : ℕ) :
  total_skittles = 25 →
  num_people = 5 →
  skittles_per_person = total_skittles / num_people →
  skittles_per_person = 5 :=
by sorry

end skittles_distribution_l1038_103802


namespace simplify_expression_l1038_103831

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 - b^3 = a - b) :
  a / b + b / a + 1 / (a * b) = 2 := by
  sorry

end simplify_expression_l1038_103831


namespace total_count_is_1500_l1038_103895

/-- The number of people counted on the second day -/
def second_day_count : ℕ := 500

/-- The number of people counted on the first day -/
def first_day_count : ℕ := 2 * second_day_count

/-- The total number of people counted over two days -/
def total_count : ℕ := first_day_count + second_day_count

theorem total_count_is_1500 : total_count = 1500 := by
  sorry

end total_count_is_1500_l1038_103895


namespace fraction_multiplication_l1038_103865

theorem fraction_multiplication : (1 : ℚ) / 3 * (3 : ℚ) / 4 * (4 : ℚ) / 5 = (1 : ℚ) / 5 := by
  sorry

end fraction_multiplication_l1038_103865


namespace consecutive_integers_sum_negative_l1038_103826

theorem consecutive_integers_sum_negative : ∃ n : ℤ, 
  (n^2 - 13*n + 36) + ((n+1)^2 - 13*(n+1) + 36) < 0 ∧ n = 4 := by
  sorry

end consecutive_integers_sum_negative_l1038_103826


namespace parabola_line_intersection_l1038_103868

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (20, 14)

/-- The line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - Q.2 = m * (p.1 - Q.1)}

/-- The condition for non-intersection -/
def no_intersection (m : ℝ) : Prop :=
  line_through_Q m ∩ P = ∅

theorem parabola_line_intersection :
  ∃ (r s : ℝ), (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 80 := by
  sorry

end parabola_line_intersection_l1038_103868


namespace toy_store_fraction_l1038_103830

-- Define John's weekly allowance
def weekly_allowance : ℚ := 4.80

-- Define the fraction spent at the arcade
def arcade_fraction : ℚ := 3/5

-- Define the amount spent at the candy store
def candy_store_spending : ℚ := 1.28

-- Theorem statement
theorem toy_store_fraction :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction)
  let toy_store_spending := remaining_after_arcade - candy_store_spending
  toy_store_spending / remaining_after_arcade = 1/3 := by
sorry

end toy_store_fraction_l1038_103830


namespace problem_statement_l1038_103808

theorem problem_statement (number : ℝ) (value : ℝ) : 
  number = 1.375 →
  0.6667 * number + 0.75 = value →
  value = 1.666675 := by
  sorry

end problem_statement_l1038_103808


namespace second_boy_probability_l1038_103807

/-- Represents a student in the classroom -/
inductive Student : Type
| Boy : Student
| Girl : Student

/-- The type of all possible orders in which students can leave -/
def LeaveOrder := List Student

/-- Generate all possible leave orders for 2 boys and 2 girls -/
def allLeaveOrders : List LeaveOrder :=
  sorry

/-- Check if the second student in a leave order is a boy -/
def isSecondBoy (order : LeaveOrder) : Bool :=
  sorry

/-- Count the number of leave orders where the second student is a boy -/
def countSecondBoy (orders : List LeaveOrder) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem second_boy_probability (orders : List LeaveOrder) 
  (h1 : orders = allLeaveOrders) 
  (h2 : orders.length = 6) : 
  (countSecondBoy orders : ℚ) / (orders.length : ℚ) = 1 / 2 :=
sorry

end second_boy_probability_l1038_103807


namespace largest_square_area_l1038_103863

/-- Represents a right triangle with squares on each side -/
structure RightTriangleWithSquares where
  -- Side lengths
  xz : ℝ
  yz : ℝ
  xy : ℝ
  -- Right angle condition
  right_angle : xy^2 = xz^2 + yz^2
  -- Non-negativity of side lengths
  xz_nonneg : xz ≥ 0
  yz_nonneg : yz ≥ 0
  xy_nonneg : xy ≥ 0

/-- The theorem to be proved -/
theorem largest_square_area
  (t : RightTriangleWithSquares)
  (sum_area : t.xz^2 + t.yz^2 + t.xy^2 = 450) :
  t.xy^2 = 225 := by
  sorry

end largest_square_area_l1038_103863


namespace sum_of_transformed_roots_equals_one_l1038_103848

theorem sum_of_transformed_roots_equals_one : 
  ∀ α β γ : ℂ, 
  (α^3 = α + 1) → (β^3 = β + 1) → (γ^3 = γ + 1) →
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 := by
  sorry

end sum_of_transformed_roots_equals_one_l1038_103848


namespace smallest_cube_ending_368_l1038_103825

theorem smallest_cube_ending_368 :
  ∀ n : ℕ+, n.val^3 ≡ 368 [MOD 1000] → n.val ≥ 14 :=
by sorry

end smallest_cube_ending_368_l1038_103825


namespace box_volume_l1038_103864

/-- A rectangular box with specific proportions -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  front_half_top : length * height = 0.5 * (length * width)
  top_1_5_side : length * width = 1.5 * (width * height)
  side_area : width * height = 72

/-- The volume of a box is equal to 648 cubic units -/
theorem box_volume (b : Box) : b.length * b.width * b.height = 648 := by
  sorry

end box_volume_l1038_103864


namespace inequality_proof_l1038_103842

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) :
  (1 / x) + (4 / y) + (9 / z) ≥ 36 := by
sorry

end inequality_proof_l1038_103842


namespace evaluate_expression_l1038_103858

theorem evaluate_expression : -((18 / 3)^2 * 4 - 80 + 5 * 7) = -99 := by
  sorry

end evaluate_expression_l1038_103858


namespace reach_one_l1038_103800

/-- Represents the two possible operations in the game -/
inductive Operation
  | EraseUnitsDigit
  | MultiplyByTwo

/-- Defines a step in the game as applying an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.EraseUnitsDigit => n / 10
  | Operation.MultiplyByTwo => n * 2

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a number -/
def applySequence (n : ℕ) (seq : OperationSequence) : ℕ :=
  seq.foldl applyOperation n

/-- The main theorem stating that for any positive natural number,
    there exists a sequence of operations that transforms it to 1 -/
theorem reach_one (n : ℕ) (h : n > 0) :
  ∃ (seq : OperationSequence), applySequence n seq = 1 := by
  sorry

end reach_one_l1038_103800


namespace same_solution_implies_a_equals_seven_l1038_103891

theorem same_solution_implies_a_equals_seven (a : ℝ) : 
  (∃ x : ℝ, 6 * (x + 8) = 18 * x ∧ 6 * x - 2 * (a - x) = 2 * a + x) → 
  a = 7 := by
  sorry

end same_solution_implies_a_equals_seven_l1038_103891


namespace mitch_macarons_count_l1038_103845

/-- The number of macarons Mitch made -/
def mitch_macarons : ℕ := 20

/-- The number of macarons Joshua made -/
def joshua_macarons : ℕ := mitch_macarons + 6

/-- The number of macarons Miles made -/
def miles_macarons : ℕ := 2 * joshua_macarons

/-- The number of macarons Renz made -/
def renz_macarons : ℕ := (3 * miles_macarons) / 4 - 1

/-- The total number of macarons given to kids -/
def total_macarons : ℕ := 68 * 2

theorem mitch_macarons_count : 
  mitch_macarons + joshua_macarons + miles_macarons + renz_macarons = total_macarons :=
by sorry

end mitch_macarons_count_l1038_103845


namespace quadratic_roots_condition_l1038_103872

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 6 * x + 18 = 0) ∧ 
  (∀ x y : ℝ, 3 * x^2 - m * x - 6 * x + 18 = 0 ∧ 3 * y^2 - m * y - 6 * y + 18 = 0 → x = y) ∧
  ((m + 6) / 3 < -2) →
  m = -6 - 6 * Real.sqrt 6 :=
by sorry

end quadratic_roots_condition_l1038_103872


namespace junyoung_remaining_pencils_l1038_103876

/-- Calculates the number of remaining pencils after giving some away -/
def remaining_pencils (initial_dozens : ℕ) (given_dozens : ℕ) (given_individual : ℕ) : ℕ :=
  initial_dozens * 12 - (given_dozens * 12 + given_individual)

/-- Theorem stating that given the initial conditions, 75 pencils remain -/
theorem junyoung_remaining_pencils :
  remaining_pencils 11 4 9 = 75 := by
  sorry

end junyoung_remaining_pencils_l1038_103876


namespace total_inflation_time_is_900_l1038_103832

/-- The time in minutes it takes to inflate one soccer ball -/
def inflation_time : ℕ := 20

/-- The number of soccer balls Alexia inflates -/
def alexia_balls : ℕ := 20

/-- The number of additional balls Ermias inflates compared to Alexia -/
def ermias_additional_balls : ℕ := 5

/-- The total number of balls Ermias inflates -/
def ermias_balls : ℕ := alexia_balls + ermias_additional_balls

/-- The total time taken by Alexia and Ermias to inflate all soccer balls -/
def total_inflation_time : ℕ := inflation_time * (alexia_balls + ermias_balls)

theorem total_inflation_time_is_900 : total_inflation_time = 900 := by
  sorry

end total_inflation_time_is_900_l1038_103832


namespace tan_half_product_squared_l1038_103885

theorem tan_half_product_squared (a b : Real) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) : 
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 20 := by
  sorry

end tan_half_product_squared_l1038_103885


namespace young_inequality_l1038_103871

theorem young_inequality (a b p q : ℝ) 
  (ha : a > 0) (hb : b > 0) (hp : p > 1) (hq : q > 1) (hpq : 1/p + 1/q = 1) :
  a^(1/p) * b^(1/q) ≤ a/p + b/q := by
  sorry

end young_inequality_l1038_103871


namespace line_does_not_intersect_circle_l1038_103847

/-- Proves that a line does not intersect a circle given the radius and distance from center to line -/
theorem line_does_not_intersect_circle (r d : ℝ) (hr : r = 10) (hd : d = 13) :
  d > r → ¬ (∃ (p : ℝ × ℝ), p.1^2 + p.2^2 = r^2 ∧ d = |p.1|) :=
by sorry

end line_does_not_intersect_circle_l1038_103847


namespace songbook_cost_is_seven_l1038_103838

/-- The cost of Jason's music purchases -/
structure MusicPurchase where
  flute : ℝ
  stand : ℝ
  total : ℝ

/-- The cost of the song book given Jason's other music purchases -/
def songbook_cost (p : MusicPurchase) : ℝ :=
  p.total - (p.flute + p.stand)

/-- Theorem: The cost of the song book is $7.00 -/
theorem songbook_cost_is_seven (p : MusicPurchase)
  (h1 : p.flute = 142.46)
  (h2 : p.stand = 8.89)
  (h3 : p.total = 158.35) :
  songbook_cost p = 7.00 := by
  sorry

#eval songbook_cost { flute := 142.46, stand := 8.89, total := 158.35 }

end songbook_cost_is_seven_l1038_103838


namespace fifth_employee_speed_correct_l1038_103804

/-- Calculates the typing speed of the 5th employee given the team's average and the typing speeds of the other 4 employees. -/
def calculate_fifth_employee_speed (team_size : Nat) (team_average : Nat) (employee1_speed : Nat) (employee2_speed : Nat) (employee3_speed : Nat) (employee4_speed : Nat) : Nat :=
  team_size * team_average - (employee1_speed + employee2_speed + employee3_speed + employee4_speed)

/-- Theorem stating that the calculated speed of the 5th employee is correct given the team's average and the speeds of the other 4 employees. -/
theorem fifth_employee_speed_correct (team_average : Nat) (employee1_speed : Nat) (employee2_speed : Nat) (employee3_speed : Nat) (employee4_speed : Nat) :
  let team_size : Nat := 5
  let fifth_employee_speed := calculate_fifth_employee_speed team_size team_average employee1_speed employee2_speed employee3_speed employee4_speed
  (employee1_speed + employee2_speed + employee3_speed + employee4_speed + fifth_employee_speed) / team_size = team_average :=
by
  sorry

#eval calculate_fifth_employee_speed 5 80 64 76 91 80

end fifth_employee_speed_correct_l1038_103804


namespace bridge_length_is_80_l1038_103877

/-- The length of a bridge given train parameters and crossing time -/
def bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  train_speed * crossing_time - train_length

/-- Theorem: The bridge length is 80 meters given the specified conditions -/
theorem bridge_length_is_80 :
  bridge_length 280 18 20 = 80 := by
  sorry

#eval bridge_length 280 18 20

end bridge_length_is_80_l1038_103877


namespace tournament_participants_l1038_103853

/-- The number of games played in a chess tournament -/
def num_games : ℕ := 136

/-- Calculates the number of games played in a tournament given the number of participants -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Proves that the number of participants in the tournament is 17 -/
theorem tournament_participants : ∃ n : ℕ, n > 0 ∧ games_played n = num_games ∧ n = 17 := by
  sorry

end tournament_participants_l1038_103853
