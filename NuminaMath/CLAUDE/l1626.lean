import Mathlib

namespace NUMINAMATH_CALUDE_f_increasing_interval_l1626_162680

-- Define the function f(x) = 2x³ - ln(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - Real.log x

-- Theorem statement
theorem f_increasing_interval :
  ∀ x : ℝ, x > 0 → (∀ y : ℝ, y > x → f y > f x) ↔ x > (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_interval_l1626_162680


namespace NUMINAMATH_CALUDE_proportion_solution_l1626_162694

theorem proportion_solution (y : ℝ) : 
  (0.75 : ℝ) / 1.2 = y / 8 → y = 5 := by
sorry

end NUMINAMATH_CALUDE_proportion_solution_l1626_162694


namespace NUMINAMATH_CALUDE_equation_solution_l1626_162660

theorem equation_solution (x : ℝ) : (10 - x)^2 = 4 * x^2 ↔ x = 10/3 ∨ x = -10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1626_162660


namespace NUMINAMATH_CALUDE_midpoints_collinear_l1626_162602

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Two mutually perpendicular lines passing through a point -/
structure PerpendicularLines where
  origin : ℝ × ℝ
  direction1 : ℝ × ℝ
  direction2 : ℝ × ℝ
  perpendicular : direction1.1 * direction2.1 + direction1.2 * direction2.2 = 0

/-- Intersection points of lines with triangle sides -/
def intersectionPoints (t : Triangle) (l : PerpendicularLines) : List (ℝ × ℝ) := sorry

/-- Midpoints of segments -/
def midpoints (points : List (ℝ × ℝ)) : List (ℝ × ℝ) := sorry

/-- Check if points are collinear -/
def areCollinear (points : List (ℝ × ℝ)) : Prop := sorry

/-- Main theorem -/
theorem midpoints_collinear (t : Triangle) :
  let o := orthocenter t
  let l := PerpendicularLines.mk o (1, 0) (0, 1) (by simp)
  let intersections := intersectionPoints t l
  let mids := midpoints intersections
  areCollinear mids := by sorry

end NUMINAMATH_CALUDE_midpoints_collinear_l1626_162602


namespace NUMINAMATH_CALUDE_solution_l1626_162673

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.foma - 70 = w.ierema + 70) ∧ 
  (w.foma - 40 = w.yuliy)

/-- The amount of gold coins Foma should give to Ierema to equalize their wealth -/
def coins_to_equalize (w : MerchantWealth) : ℕ :=
  (w.foma - w.ierema) / 2

/-- Theorem stating the solution to the problem -/
theorem solution (w : MerchantWealth) 
  (h : problem_conditions w) : 
  coins_to_equalize w = 55 := by
  sorry

end NUMINAMATH_CALUDE_solution_l1626_162673


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inv_sqrt_x_l1626_162634

theorem sqrt_x_plus_inv_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inv_sqrt_x_l1626_162634


namespace NUMINAMATH_CALUDE_todd_ate_cupcakes_l1626_162601

theorem todd_ate_cupcakes (initial_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) : 
  initial_cupcakes = 18 →
  packages = 5 →
  cupcakes_per_package = 2 →
  initial_cupcakes - packages * cupcakes_per_package = 8 :=
by sorry

end NUMINAMATH_CALUDE_todd_ate_cupcakes_l1626_162601


namespace NUMINAMATH_CALUDE_difference_of_squares_l1626_162672

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1626_162672


namespace NUMINAMATH_CALUDE_square_pyramid_sum_l1626_162631

/-- A square pyramid is a polyhedron with a square base and triangular faces meeting at an apex. -/
structure SquarePyramid where
  -- We don't need to define the internal structure, just the concept

/-- The number of faces in a square pyramid -/
def num_faces (sp : SquarePyramid) : ℕ := 5

/-- The number of edges in a square pyramid -/
def num_edges (sp : SquarePyramid) : ℕ := 8

/-- The number of vertices in a square pyramid -/
def num_vertices (sp : SquarePyramid) : ℕ := 5

/-- The sum of faces, edges, and vertices in a square pyramid is 18 -/
theorem square_pyramid_sum (sp : SquarePyramid) : 
  num_faces sp + num_edges sp + num_vertices sp = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_sum_l1626_162631


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_zero_l1626_162639

/-- Given a quadratic equation (k-1)x^2 + x - k^2 = 0 with a root x = 1, prove that k = 0 -/
theorem quadratic_root_implies_k_zero (k : ℝ) : 
  ((k - 1) * 1^2 + 1 - k^2 = 0) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_zero_l1626_162639


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1626_162669

/-- The length of a bridge given train characteristics and crossing time -/
theorem bridge_length_calculation 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 120)
  (h2 : crossing_time = 26.997840172786177)
  (h3 : train_speed_kmph = 36) :
  ∃ bridge_length : ℝ, 
    bridge_length = 149.97840172786177 ∧ 
    bridge_length = (train_speed_kmph * 1000 / 3600) * crossing_time - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1626_162669


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1626_162625

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def A : Finset ℕ := {2, 3}
def B : Finset ℕ := {3, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1626_162625


namespace NUMINAMATH_CALUDE_nails_per_plank_l1626_162675

theorem nails_per_plank (total_nails : ℕ) (total_planks : ℕ) (h1 : total_nails = 4) (h2 : total_planks = 2) :
  total_nails / total_planks = 2 := by
sorry

end NUMINAMATH_CALUDE_nails_per_plank_l1626_162675


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1626_162648

universe u

def U : Set (Fin 5) := {0, 1, 2, 3, 4}
def M : Set (Fin 5) := {0, 2, 3}
def N : Set (Fin 5) := {1, 3, 4}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1626_162648


namespace NUMINAMATH_CALUDE_theater_ticket_price_l1626_162676

theorem theater_ticket_price (adult_price : ℕ) 
  (total_attendance : ℕ) (total_revenue : ℕ) (child_attendance : ℕ) :
  total_attendance = 280 →
  total_revenue = 14000 →
  child_attendance = 80 →
  (total_attendance - child_attendance) * adult_price + child_attendance * 25 = total_revenue →
  adult_price = 60 := by
sorry

end NUMINAMATH_CALUDE_theater_ticket_price_l1626_162676


namespace NUMINAMATH_CALUDE_original_cost_was_75_l1626_162655

/-- Represents the selling price of a key chain -/
def selling_price : ℝ := 100

/-- Represents the original profit percentage -/
def original_profit_percentage : ℝ := 0.25

/-- Represents the new profit percentage -/
def new_profit_percentage : ℝ := 0.50

/-- Represents the new manufacturing cost -/
def new_manufacturing_cost : ℝ := 50

/-- Calculates the original manufacturing cost based on the given conditions -/
def original_manufacturing_cost : ℝ := selling_price * (1 - original_profit_percentage)

/-- Theorem stating that the original manufacturing cost was $75 -/
theorem original_cost_was_75 : 
  original_manufacturing_cost = 75 := by sorry

end NUMINAMATH_CALUDE_original_cost_was_75_l1626_162655


namespace NUMINAMATH_CALUDE_train_meeting_time_l1626_162621

theorem train_meeting_time (distance : ℝ) (speed_diff : ℝ) (final_speed : ℝ) :
  distance = 450 →
  speed_diff = 6 →
  final_speed = 48 →
  (distance / (final_speed + (final_speed + speed_diff))) = 75 / 17 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_time_l1626_162621


namespace NUMINAMATH_CALUDE_new_person_weight_l1626_162698

theorem new_person_weight (initial_count : ℕ) (weight_removed : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  weight_removed = 65 →
  avg_increase = 2 →
  ∃ (initial_avg : ℝ) (new_weight : ℝ),
    initial_count * (initial_avg + avg_increase) = initial_count * initial_avg - weight_removed + new_weight →
    new_weight = 81 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1626_162698


namespace NUMINAMATH_CALUDE_complex_number_magnitude_squared_l1626_162689

theorem complex_number_magnitude_squared : 
  ∀ z : ℂ, z + Complex.abs z = 5 - 3*I → Complex.abs z^2 = 11.56 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_squared_l1626_162689


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l1626_162640

theorem reciprocal_of_negative_one_third :
  ∀ x : ℚ, x * (-1/3) = 1 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l1626_162640


namespace NUMINAMATH_CALUDE_solve_sock_problem_l1626_162657

def sock_problem (initial_pairs : ℕ) (lost_pairs : ℕ) (purchased_pairs : ℕ) (gifted_pairs : ℕ) (final_pairs : ℕ) : Prop :=
  let remaining_pairs := initial_pairs - lost_pairs
  ∃ (donated_fraction : ℚ),
    0 ≤ donated_fraction ∧
    donated_fraction ≤ 1 ∧
    remaining_pairs * (1 - donated_fraction) + purchased_pairs + gifted_pairs = final_pairs ∧
    donated_fraction = 2/3

theorem solve_sock_problem :
  sock_problem 40 4 10 3 25 :=
sorry

end NUMINAMATH_CALUDE_solve_sock_problem_l1626_162657


namespace NUMINAMATH_CALUDE_check_amount_proof_l1626_162617

theorem check_amount_proof (C : ℝ) 
  (tip_percentage : ℝ) 
  (tip_contribution : ℝ) : 
  tip_percentage = 0.20 → 
  tip_contribution = 40 → 
  tip_percentage * C = tip_contribution → 
  C = 200 := by
sorry

end NUMINAMATH_CALUDE_check_amount_proof_l1626_162617


namespace NUMINAMATH_CALUDE_quadratic_sum_l1626_162614

/-- A quadratic function y = ax^2 + bx + c with a minimum value of 61
    that passes through the points (1,0) and (3,0) -/
def QuadraticFunction (a b c : ℝ) : Prop :=
  (∀ x, a*x^2 + b*x + c ≥ 61) ∧
  (∃ x₀, a*x₀^2 + b*x₀ + c = 61) ∧
  (a*1^2 + b*1 + c = 0) ∧
  (a*3^2 + b*3 + c = 0)

theorem quadratic_sum (a b c : ℝ) :
  QuadraticFunction a b c → a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1626_162614


namespace NUMINAMATH_CALUDE_monthly_fee_calculation_l1626_162685

/-- Represents the long distance phone service billing structure and usage -/
structure PhoneBill where
  monthlyFee : ℝ
  ratePerMinute : ℝ
  minutesUsed : ℕ
  totalBill : ℝ

/-- Theorem stating that given the specific conditions, the monthly fee is $2.00 -/
theorem monthly_fee_calculation (bill : PhoneBill) 
    (h1 : bill.ratePerMinute = 0.12)
    (h2 : bill.minutesUsed = 178)
    (h3 : bill.totalBill = 23.36) :
    bill.monthlyFee = 2.00 := by
  sorry

end NUMINAMATH_CALUDE_monthly_fee_calculation_l1626_162685


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1626_162653

-- Define the triangle ABC
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  a + b = 11 →
  c = 7 →
  Real.cos A = -1/7 →
  -- Conclusions to prove
  a = 8 ∧
  Real.sin C = Real.sqrt 3 / 2 ∧
  (1/2 : ℝ) * a * b * Real.sin C = 6 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l1626_162653


namespace NUMINAMATH_CALUDE_bertha_initial_balls_l1626_162607

def tennis_balls (initial_balls : ℕ) : Prop :=
  let worn_out := 20 / 10
  let lost := 20 / 5
  let bought := (20 / 4) * 3
  initial_balls - worn_out - lost + bought - 1 = 10

theorem bertha_initial_balls :
  ∃ (initial_balls : ℕ), tennis_balls initial_balls ∧ initial_balls = 2 :=
sorry

end NUMINAMATH_CALUDE_bertha_initial_balls_l1626_162607


namespace NUMINAMATH_CALUDE_vector_subtraction_magnitude_l1626_162662

theorem vector_subtraction_magnitude : ∃ (a b : ℝ × ℝ), 
  a = (2, 1) ∧ b = (-2, 4) ∧ 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_magnitude_l1626_162662


namespace NUMINAMATH_CALUDE_not_p_or_not_q_false_implies_p_and_q_and_p_or_q_true_l1626_162637

theorem not_p_or_not_q_false_implies_p_and_q_and_p_or_q_true (p q : Prop) :
  (¬(¬p ∨ ¬q)) → ((p ∧ q) ∧ (p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_false_implies_p_and_q_and_p_or_q_true_l1626_162637


namespace NUMINAMATH_CALUDE_custom_op_result_l1626_162679

-- Define the custom operation
def custom_op (a b c : ℕ) : ℕ := 
  (a * b * 10000) + (a * c * 100) + (a * (b + c))

-- State the theorem
theorem custom_op_result : custom_op 7 2 5 = 143549 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l1626_162679


namespace NUMINAMATH_CALUDE_total_cost_over_two_years_l1626_162668

/-- Represents the number of games attended and their types -/
structure GameAttendance where
  home : Nat
  away : Nat
  homePlayoff : Nat
  awayPlayoff : Nat

/-- Represents the ticket prices for different game types -/
structure TicketPrices where
  home : Nat
  away : Nat
  homePlayoff : Nat
  awayPlayoff : Nat

/-- Calculates the total cost for a given year -/
def calculateYearlyCost (attendance : GameAttendance) (prices : TicketPrices) : Nat :=
  attendance.home * prices.home +
  attendance.away * prices.away +
  attendance.homePlayoff * prices.homePlayoff +
  attendance.awayPlayoff * prices.awayPlayoff

/-- Theorem stating the total cost over two years -/
theorem total_cost_over_two_years
  (prices : TicketPrices)
  (thisYear : GameAttendance)
  (lastYear : GameAttendance)
  (h1 : prices.home = 60)
  (h2 : prices.away = 75)
  (h3 : prices.homePlayoff = 120)
  (h4 : prices.awayPlayoff = 100)
  (h5 : thisYear.home = 2)
  (h6 : thisYear.away = 2)
  (h7 : thisYear.homePlayoff = 1)
  (h8 : thisYear.awayPlayoff = 0)
  (h9 : lastYear.home = 6)
  (h10 : lastYear.away = 3)
  (h11 : lastYear.homePlayoff = 1)
  (h12 : lastYear.awayPlayoff = 1) :
  calculateYearlyCost thisYear prices + calculateYearlyCost lastYear prices = 1195 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_over_two_years_l1626_162668


namespace NUMINAMATH_CALUDE_berry_reading_problem_l1626_162628

theorem berry_reading_problem (pages_per_day : ℕ) (days_in_week : ℕ) 
  (pages_sun : ℕ) (pages_mon : ℕ) (pages_tue : ℕ) (pages_wed : ℕ) 
  (pages_fri : ℕ) (pages_sat : ℕ) :
  pages_per_day = 50 →
  days_in_week = 7 →
  pages_sun = 43 →
  pages_mon = 65 →
  pages_tue = 28 →
  pages_wed = 0 →
  pages_fri = 56 →
  pages_sat = 88 →
  ∃ pages_thu : ℕ, 
    pages_thu = pages_per_day * days_in_week - 
      (pages_sun + pages_mon + pages_tue + pages_wed + pages_fri + pages_sat) ∧
    pages_thu = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_berry_reading_problem_l1626_162628


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_17_l1626_162606

theorem modular_inverse_of_3_mod_17 :
  ∃ (x : ℤ), 0 ≤ x ∧ x ≤ 16 ∧ (3 * x) % 17 = 1 :=
by
  use 6
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_17_l1626_162606


namespace NUMINAMATH_CALUDE_jason_climbing_speed_l1626_162666

/-- Given that Matt climbs at 6 feet per minute and Jason is 42 feet higher than Matt after 7 minutes,
    prove that Jason's climbing speed is 12 feet per minute. -/
theorem jason_climbing_speed (matt_speed : ℝ) (time : ℝ) (height_difference : ℝ) :
  matt_speed = 6 →
  time = 7 →
  height_difference = 42 →
  (time * matt_speed + height_difference) / time = 12 := by
sorry

end NUMINAMATH_CALUDE_jason_climbing_speed_l1626_162666


namespace NUMINAMATH_CALUDE_sector_central_angle_l1626_162604

/-- Given a circular sector with radius r and perimeter 3r, its central angle is 1 radian. -/
theorem sector_central_angle (r : ℝ) (h : r > 0) : 
  (∃ (α : ℝ), α > 0 ∧ r * (2 + α) = 3 * r) → 
  (∃ (α : ℝ), α > 0 ∧ r * (2 + α) = 3 * r ∧ α = 1) :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1626_162604


namespace NUMINAMATH_CALUDE_breaking_process_result_l1626_162674

/-- Represents a triangle with its three angles in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

/-- Determines if a triangle is acute-angled -/
def Triangle.isAcute (t : Triangle) : Prop :=
  t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90

/-- Represents the operation of breaking a triangle -/
def breakTriangle (t : Triangle) : List Triangle :=
  sorry  -- Implementation details omitted

/-- Counts the total number of triangles after breaking process -/
def countTriangles (initial : Triangle) : ℕ :=
  sorry  -- Implementation details omitted

/-- The theorem to be proved -/
theorem breaking_process_result (t : Triangle) 
  (h1 : t.angle1 = 3)
  (h2 : t.angle2 = 88)
  (h3 : t.angle3 = 89) :
  countTriangles t = 11 :=
sorry

end NUMINAMATH_CALUDE_breaking_process_result_l1626_162674


namespace NUMINAMATH_CALUDE_no_valid_area_codes_l1626_162695

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 4 ∨ d = 3 ∨ d = 5

def is_valid_area_code (code : Fin 4 → ℕ) : Prop :=
  ∀ i, is_valid_digit (code i)

def product_of_digits (code : Fin 4 → ℕ) : ℕ :=
  (code 0) * (code 1) * (code 2) * (code 3)

theorem no_valid_area_codes :
  ¬∃ (code : Fin 4 → ℕ), is_valid_area_code code ∧ 13 ∣ product_of_digits code := by
  sorry

end NUMINAMATH_CALUDE_no_valid_area_codes_l1626_162695


namespace NUMINAMATH_CALUDE_unique_solution_system_l1626_162696

theorem unique_solution_system (x y z : ℂ) :
  x + y + z = 3 ∧
  x^2 + y^2 + z^2 = 3 ∧
  x^3 + y^3 + z^3 = 3 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1626_162696


namespace NUMINAMATH_CALUDE_inequality_proof_l1626_162649

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : a * b + b * c + c * a = a + b + c) : 
  a^2 + b^2 + c^2 + 2*a*b*c ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1626_162649


namespace NUMINAMATH_CALUDE_set_operations_l1626_162618

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem set_operations :
  (Set.univ \ (A ∩ B) = {x | x < 3 ∨ x ≥ 7}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 7) ∨ x ≥ 10}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1626_162618


namespace NUMINAMATH_CALUDE_x_fifth_minus_seven_x_equals_222_l1626_162612

theorem x_fifth_minus_seven_x_equals_222 (x : ℝ) (h : x = 3) : x^5 - 7*x = 222 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_minus_seven_x_equals_222_l1626_162612


namespace NUMINAMATH_CALUDE_optimal_purchase_is_cheapest_l1626_162643

/-- Park admission fee per person -/
def individual_fee : ℕ := 5

/-- Group ticket fee -/
def group_fee : ℕ := 40

/-- Maximum number of people allowed per group ticket -/
def group_max : ℕ := 10

/-- Cost function for purchasing tickets -/
def ticket_cost (group_tickets : ℕ) (individual_tickets : ℕ) : ℕ :=
  group_tickets * group_fee + individual_tickets * individual_fee

/-- The most economical way to purchase tickets -/
def optimal_purchase (x : ℕ) : ℕ × ℕ :=
  let a := x / group_max
  let b := x % group_max
  if b < 8 then (a, b)
  else if b = 8 then (a, 8)  -- or (a + 1, 0), both are optimal
  else (a + 1, 0)

theorem optimal_purchase_is_cheapest (x : ℕ) :
  let (g, i) := optimal_purchase x
  ∀ (g' i' : ℕ), g' * group_max + i' ≥ x →
    ticket_cost g i ≤ ticket_cost g' i' :=
sorry

end NUMINAMATH_CALUDE_optimal_purchase_is_cheapest_l1626_162643


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1626_162663

/-- Compound interest calculation --/
theorem compound_interest_rate (P : ℝ) (t : ℝ) (interest : ℝ) (h1 : P > 0) (h2 : t > 0) (h3 : interest > 0) :
  let A := P + interest
  let n := 1
  let r := (((A / P) ^ (1 / (n * t))) - 1) * n
  r = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1626_162663


namespace NUMINAMATH_CALUDE_steven_shirts_l1626_162635

/-- The number of shirts owned by Brian -/
def brian_shirts : ℕ := 3

/-- The number of shirts owned by Andrew relative to Brian -/
def andrew_multiplier : ℕ := 6

/-- The number of shirts owned by Steven relative to Andrew -/
def steven_multiplier : ℕ := 4

/-- Theorem: Given the conditions, Steven has 72 shirts -/
theorem steven_shirts : 
  steven_multiplier * (andrew_multiplier * brian_shirts) = 72 := by
sorry

end NUMINAMATH_CALUDE_steven_shirts_l1626_162635


namespace NUMINAMATH_CALUDE_power_of_two_equality_l1626_162615

theorem power_of_two_equality (n : ℕ) : 2^n = 2 * 16^2 * 4^3 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l1626_162615


namespace NUMINAMATH_CALUDE_hiking_rate_up_l1626_162683

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  rate_up : ℝ
  days_up : ℝ
  route_down_length : ℝ
  rate_down_multiplier : ℝ

/-- The hiking scenario satisfies the given conditions -/
def satisfies_conditions (h : HikingScenario) : Prop :=
  h.days_up = 2 ∧ 
  h.route_down_length = 15 ∧
  h.rate_down_multiplier = 1.5 ∧
  h.rate_up * h.days_up = h.route_down_length / h.rate_down_multiplier

/-- Theorem stating that the rate up the mountain is 5 miles per day -/
theorem hiking_rate_up (h : HikingScenario) 
  (hc : satisfies_conditions h) : h.rate_up = 5 := by
  sorry

#check hiking_rate_up

end NUMINAMATH_CALUDE_hiking_rate_up_l1626_162683


namespace NUMINAMATH_CALUDE_computer_repair_cost_l1626_162661

theorem computer_repair_cost (phone_cost laptop_cost : ℕ) 
  (phone_repairs laptop_repairs computer_repairs : ℕ) (total_earnings : ℕ) :
  phone_cost = 11 →
  laptop_cost = 15 →
  phone_repairs = 5 →
  laptop_repairs = 2 →
  computer_repairs = 2 →
  total_earnings = 121 →
  ∃ (computer_cost : ℕ), 
    phone_cost * phone_repairs + laptop_cost * laptop_repairs + computer_cost * computer_repairs = total_earnings ∧
    computer_cost = 18 :=
by sorry

end NUMINAMATH_CALUDE_computer_repair_cost_l1626_162661


namespace NUMINAMATH_CALUDE_number_problem_l1626_162684

theorem number_problem (x : ℝ) : 0.3 * x - 70 = 20 → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1626_162684


namespace NUMINAMATH_CALUDE_certain_number_proof_l1626_162622

theorem certain_number_proof (x : ℝ) : (3 / 5) * x^2 = 126.15 → x = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1626_162622


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l1626_162641

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x - 6 = 0
def equation2 (x : ℝ) : Prop := x/(x-1) - 1 = 3/(x^2-1)

-- Theorem for the first equation
theorem equation1_solutions :
  ∃ x1 x2 : ℝ, 
    (x1 = 2 + Real.sqrt 10 ∧ equation1 x1) ∧
    (x2 = 2 - Real.sqrt 10 ∧ equation1 x2) :=
sorry

-- Theorem for the second equation
theorem equation2_solution :
  ∃ x : ℝ, x = 2 ∧ equation2 x :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l1626_162641


namespace NUMINAMATH_CALUDE_orchestra_students_l1626_162629

theorem orchestra_students (band_students : ℕ → ℕ) (choir_students : ℕ) (total_students : ℕ) :
  (∀ x : ℕ, band_students x = 2 * x) →
  choir_students = 28 →
  total_students = 88 →
  ∃ x : ℕ, x + band_students x + choir_students = total_students ∧ x = 20 :=
by sorry

end NUMINAMATH_CALUDE_orchestra_students_l1626_162629


namespace NUMINAMATH_CALUDE_largest_perimeter_is_31_l1626_162623

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ

/-- Checks if the given lengths can form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem stating the largest possible perimeter of the triangle --/
theorem largest_perimeter_is_31 :
  ∃ (t : Triangle), t.side1 = 7 ∧ t.side2 = 9 ∧ is_valid_triangle t ∧
  (∀ (t' : Triangle), t'.side1 = 7 ∧ t'.side2 = 9 ∧ is_valid_triangle t' →
    perimeter t' ≤ perimeter t) ∧
  perimeter t = 31 :=
sorry

end NUMINAMATH_CALUDE_largest_perimeter_is_31_l1626_162623


namespace NUMINAMATH_CALUDE_vector_coordinates_l1626_162651

/-- A vector in a 2D Cartesian coordinate system -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The standard basis vectors -/
def i : Vector2D := ⟨1, 0⟩
def j : Vector2D := ⟨0, 1⟩

/-- Vector addition -/
def add (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Scalar multiplication -/
def smul (r : ℝ) (v : Vector2D) : Vector2D :=
  ⟨r * v.x, r * v.y⟩

/-- The main theorem -/
theorem vector_coordinates (x y : ℝ) :
  let a := add (smul x i) (smul y j)
  a = ⟨x, y⟩ := by sorry

end NUMINAMATH_CALUDE_vector_coordinates_l1626_162651


namespace NUMINAMATH_CALUDE_largest_number_comparison_l1626_162682

theorem largest_number_comparison :
  (1/2 : ℝ) > (37.5/100 : ℝ) ∧ (1/2 : ℝ) > (7/22 : ℝ) ∧ (1/2 : ℝ) > (π/10 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_comparison_l1626_162682


namespace NUMINAMATH_CALUDE_point_P_on_circle_M_and_line_L_l1626_162636

/-- Circle M with center (3,2) and radius √2 -/
def circle_M (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 2

/-- Line L with equation x + y - 3 = 0 -/
def line_L (x y : ℝ) : Prop := x + y - 3 = 0

/-- Point P with coordinates (2,1) -/
def point_P : ℝ × ℝ := (2, 1)

theorem point_P_on_circle_M_and_line_L :
  circle_M point_P.1 point_P.2 ∧ line_L point_P.1 point_P.2 := by
  sorry

end NUMINAMATH_CALUDE_point_P_on_circle_M_and_line_L_l1626_162636


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1626_162627

theorem max_value_of_expression (m : ℝ) : 
  (4 - |2 - m|) ≤ 4 ∧ ∃ m : ℝ, 4 - |2 - m| = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1626_162627


namespace NUMINAMATH_CALUDE_problem_solution_l1626_162677

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) : 
  x * y ≤ 4 ∧ (∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 → 1 / (a + 1) + 4 / b ≥ 9 / 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1626_162677


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l1626_162600

theorem abs_inequality_solution_set (x : ℝ) : 
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l1626_162600


namespace NUMINAMATH_CALUDE_second_half_speed_l1626_162638

/-- Given a trip with the following properties:
  * Total distance is 60 km
  * First half of the trip (30 km) is traveled at 48 km/h
  * Average speed of the entire trip is 32 km/h
  Then the speed of the second half of the trip is 24 km/h -/
theorem second_half_speed (total_distance : ℝ) (first_half_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 60 →
  first_half_distance = 30 →
  first_half_speed = 48 →
  average_speed = 32 →
  let second_half_distance := total_distance - first_half_distance
  let total_time := total_distance / average_speed
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := total_time - first_half_time
  let second_half_speed := second_half_distance / second_half_time
  second_half_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_second_half_speed_l1626_162638


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l1626_162658

/-- The equation ({m-2}){x^{m^2-2}}+4x-7=0 is quadratic -/
def is_quadratic (m : ℝ) : Prop :=
  (m^2 - 2 = 2) ∧ (m - 2 ≠ 0)

theorem quadratic_equation_m_value :
  ∀ m : ℝ, is_quadratic m → m = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l1626_162658


namespace NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l1626_162619

theorem arccos_of_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_of_one_eq_zero_l1626_162619


namespace NUMINAMATH_CALUDE_log_cube_of_nine_l1626_162613

-- Define a tolerance for approximation
def tolerance : ℝ := 0.000000000000002

-- Define the approximate equality
def approx_equal (a b : ℝ) : Prop := abs (a - b) < tolerance

theorem log_cube_of_nine (x y : ℝ) :
  approx_equal x 9 → (Real.log x^3 / Real.log 9 = y) → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_cube_of_nine_l1626_162613


namespace NUMINAMATH_CALUDE_circular_cross_section_shapes_l1626_162642

-- Define the geometric shapes
inductive GeometricShape
  | Cube
  | Sphere
  | Cylinder
  | PentagonalPrism

-- Define a function to check if a shape can have a circular cross-section
def canHaveCircularCrossSection (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => true
  | GeometricShape.Cylinder => true
  | _ => false

-- Theorem stating that only sphere and cylinder can have circular cross-sections
theorem circular_cross_section_shapes :
  ∀ (shape : GeometricShape),
    canHaveCircularCrossSection shape ↔ (shape = GeometricShape.Sphere ∨ shape = GeometricShape.Cylinder) :=
by sorry

end NUMINAMATH_CALUDE_circular_cross_section_shapes_l1626_162642


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l1626_162688

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧
  n / 100 + n % 10 = 8

theorem three_digit_number_theorem :
  ∀ n : ℕ, is_valid_number n → (n = 810 ∨ n = 840 ∨ n = 870) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l1626_162688


namespace NUMINAMATH_CALUDE_equation_solutions_l1626_162667

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = 7 ∧ 
    ∀ x : ℝ, 3 * (x - 4) = (x - 4)^2 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ y₁ y₂ : ℝ, y₁ = (-1 + Real.sqrt 10) / 3 ∧ y₂ = (-1 - Real.sqrt 10) / 3 ∧ 
    ∀ x : ℝ, 3 * x^2 + 2 * x - 3 = 0 ↔ x = y₁ ∨ x = y₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1626_162667


namespace NUMINAMATH_CALUDE_xaxaxa_divisible_by_seven_l1626_162670

-- Define a function to create the six-digit number XAXAXA
def makeNumber (X A : Nat) : Nat :=
  100000 * X + 10000 * A + 1000 * X + 100 * A + 10 * X + A

-- Theorem statement
theorem xaxaxa_divisible_by_seven (X A : Nat) (h1 : X < 10) (h2 : A < 10) :
  (makeNumber X A) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_xaxaxa_divisible_by_seven_l1626_162670


namespace NUMINAMATH_CALUDE_prob_different_colors_is_two_thirds_l1626_162644

-- Define the set of cards
def Card : Type := Fin 4

-- Define the color of a card
def color (c : Card) : Bool :=
  c.val < 2  -- First two cards are red, last two are black

-- Define the probability of drawing two cards of different colors
def prob_different_colors : ℚ :=
  let total_outcomes := 4 * 3  -- Total number of ways to draw 2 cards
  let favorable_outcomes := 2 * 2 * 2  -- Number of ways to draw different colors
  favorable_outcomes / total_outcomes

-- Theorem statement
theorem prob_different_colors_is_two_thirds :
  prob_different_colors = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_different_colors_is_two_thirds_l1626_162644


namespace NUMINAMATH_CALUDE_power_product_positive_l1626_162626

theorem power_product_positive (m n : ℕ) (hm : m > 2) :
  ∃ k : ℕ+, (2^m - 1) * (2^n + 1) = k := by
  sorry

end NUMINAMATH_CALUDE_power_product_positive_l1626_162626


namespace NUMINAMATH_CALUDE_vector_parallel_cosine_value_l1626_162697

theorem vector_parallel_cosine_value (θ : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : (9/10, 3) = (Real.cos (θ + π/6), 2)) : 
  Real.cos θ = (4 + 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_cosine_value_l1626_162697


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l1626_162654

/-- Given a triangle ABC with the specified conditions, prove that A = π/6 and a = 2 -/
theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π →
  A + B + C = π →
  Real.sqrt 3 * a * Real.sin C = c * Real.cos A →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  b + c = 2 + 2 * Real.sqrt 3 →
  A = π / 6 ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l1626_162654


namespace NUMINAMATH_CALUDE_classroom_capacity_l1626_162616

/-- The number of rows of desks in the classroom -/
def num_rows : ℕ := 8

/-- The number of desks in the first row -/
def first_row_desks : ℕ := 10

/-- The increase in number of desks for each subsequent row -/
def desk_increase : ℕ := 2

/-- The total number of desks in the classroom -/
def total_desks : ℕ := (num_rows * (2 * first_row_desks + (num_rows - 1) * desk_increase)) / 2

theorem classroom_capacity :
  total_desks = 136 := by sorry

end NUMINAMATH_CALUDE_classroom_capacity_l1626_162616


namespace NUMINAMATH_CALUDE_perimeter_is_72_l1626_162632

/-- A geometric figure formed by six identical squares arranged into a larger rectangle,
    with two smaller identical squares placed inside. -/
structure GeometricFigure where
  /-- The side length of each of the six identical squares forming the larger rectangle -/
  side_length : ℝ
  /-- The total area of the figure -/
  total_area : ℝ
  /-- The area of the figure is equal to the area of six squares -/
  area_eq : total_area = 6 * side_length^2

/-- The perimeter of the geometric figure -/
def perimeter (fig : GeometricFigure) : ℝ :=
  2 * (3 * fig.side_length + 2 * fig.side_length) + 2 * fig.side_length

theorem perimeter_is_72 (fig : GeometricFigure) (h : fig.total_area = 216) :
  perimeter fig = 72 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_72_l1626_162632


namespace NUMINAMATH_CALUDE_floor_length_is_20_l1626_162608

/-- Proves that the length of a rectangular floor is 20 meters given the specified conditions -/
theorem floor_length_is_20 (breadth : ℝ) (length : ℝ) (area : ℝ) (total_cost : ℝ) (rate : ℝ) : 
  length = breadth + 2 * breadth →  -- length is 200% more than breadth
  area = length * breadth →         -- area formula
  area = total_cost / rate →        -- area from cost and rate
  total_cost = 400 →                -- given total cost
  rate = 3 →                        -- given rate per square meter
  length = 20 := by
sorry

end NUMINAMATH_CALUDE_floor_length_is_20_l1626_162608


namespace NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l1626_162686

theorem zeroth_power_of_nonzero_rational (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l1626_162686


namespace NUMINAMATH_CALUDE_sum_of_roots_l1626_162611

theorem sum_of_roots (h b : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂)
  (h2 : 3 * x₁^2 - h * x₁ = b)
  (h3 : 3 * x₂^2 - h * x₂ = b) : 
  x₁ + x₂ = h / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1626_162611


namespace NUMINAMATH_CALUDE_high_school_population_l1626_162647

/-- Represents a high school with three grades and a stratified sampling method. -/
structure HighSchool where
  grade10_students : ℕ
  total_sample : ℕ
  grade11_sample : ℕ
  grade12_sample : ℕ

/-- Calculates the total number of students in the high school based on stratified sampling. -/
def total_students (hs : HighSchool) : ℕ :=
  let grade10_sample := hs.total_sample - hs.grade11_sample - hs.grade12_sample
  (hs.grade10_students * hs.total_sample) / grade10_sample

/-- Theorem stating that given the specific conditions, the total number of students is 1800. -/
theorem high_school_population (hs : HighSchool)
  (h1 : hs.grade10_students = 600)
  (h2 : hs.total_sample = 45)
  (h3 : hs.grade11_sample = 20)
  (h4 : hs.grade12_sample = 10) :
  total_students hs = 1800 := by
  sorry

#eval total_students { grade10_students := 600, total_sample := 45, grade11_sample := 20, grade12_sample := 10 }

end NUMINAMATH_CALUDE_high_school_population_l1626_162647


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1626_162687

def U : Finset ℤ := {0, -1, -2, -3, -4}
def M : Finset ℤ := {0, -1, -2}

theorem complement_of_M_in_U : U \ M = {-3, -4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1626_162687


namespace NUMINAMATH_CALUDE_sum_123_even_descending_l1626_162693

/-- The sum of the first n even natural numbers in descending order -/
def sumEvenDescending (n : ℕ) : ℕ :=
  n * (2 * n + 2) / 2

/-- Theorem: The sum of the first 123 even natural numbers in descending order is 15252 -/
theorem sum_123_even_descending : sumEvenDescending 123 = 15252 := by
  sorry

end NUMINAMATH_CALUDE_sum_123_even_descending_l1626_162693


namespace NUMINAMATH_CALUDE_reduced_oil_price_l1626_162605

/-- Represents the price and quantity of oil before and after a price reduction --/
structure OilPriceReduction where
  original_price : ℝ
  reduced_price : ℝ
  original_quantity : ℝ
  additional_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating that given the conditions, the reduced price of oil is 60 --/
theorem reduced_oil_price
  (oil : OilPriceReduction)
  (price_reduction : oil.reduced_price = 0.75 * oil.original_price)
  (quantity_increase : oil.additional_quantity = 5)
  (cost_equality : oil.original_quantity * oil.original_price = 
                   (oil.original_quantity + oil.additional_quantity) * oil.reduced_price)
  (total_cost : oil.total_cost = 1200)
  : oil.reduced_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_reduced_oil_price_l1626_162605


namespace NUMINAMATH_CALUDE_cars_in_driveway_three_cars_in_driveway_l1626_162692

/-- Calculates the number of cars in the driveway given the total number of wheels and the number of wheels for each item. -/
theorem cars_in_driveway (total_wheels : ℕ) (car_wheels bike_wheels trash_can_wheels tricycle_wheels roller_skate_wheels : ℕ)
  (num_bikes num_trash_cans num_tricycles num_roller_skate_pairs : ℕ) : ℕ :=
  let other_wheels := num_bikes * bike_wheels + num_trash_cans * trash_can_wheels +
                      num_tricycles * tricycle_wheels + num_roller_skate_pairs * roller_skate_wheels
  let remaining_wheels := total_wheels - other_wheels
  remaining_wheels / car_wheels

/-- Proves that there are 3 cars in the driveway given the specific conditions. -/
theorem three_cars_in_driveway :
  cars_in_driveway 25 4 2 2 3 4 2 1 1 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cars_in_driveway_three_cars_in_driveway_l1626_162692


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l1626_162664

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 7 indistinguishable balls into 3 indistinguishable boxes,
    with each box containing at least one ball, is equal to 4. -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l1626_162664


namespace NUMINAMATH_CALUDE_ellen_dough_balls_l1626_162652

/-- Represents the time it takes for a ball of dough to rise -/
def rise_time : ℕ := 3

/-- Represents the time it takes to bake a ball of dough -/
def bake_time : ℕ := 2

/-- Represents the total time for the entire baking process -/
def total_time : ℕ := 20

/-- Calculates the total time taken for a given number of dough balls -/
def time_for_n_balls (n : ℕ) : ℕ :=
  rise_time + bake_time + (n - 1) * rise_time

/-- The theorem stating the number of dough balls Ellen makes -/
theorem ellen_dough_balls :
  ∃ n : ℕ, n > 0 ∧ time_for_n_balls n = total_time ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellen_dough_balls_l1626_162652


namespace NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l1626_162620

theorem subset_sum_divisible_by_2n (n : ℕ) (a : Fin n → ℕ) 
  (h1 : n ≥ 4)
  (h2 : ∀ i j, i ≠ j → a i ≠ a j)
  (h3 : ∀ i, 0 < a i ∧ a i < 2*n) :
  ∃ (i j : Fin n), i < j ∧ (2*n) ∣ (a i + a j) :=
sorry

end NUMINAMATH_CALUDE_subset_sum_divisible_by_2n_l1626_162620


namespace NUMINAMATH_CALUDE_pipe_B_rate_correct_l1626_162633

/-- The rate at which pipe B fills the tank -/
def pipe_B_rate : ℝ := 30

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 750

/-- The rate at which pipe A fills the tank in liters per minute -/
def pipe_A_rate : ℝ := 40

/-- The rate at which pipe C drains the tank in liters per minute -/
def pipe_C_rate : ℝ := 20

/-- The time in minutes it takes to fill the tank -/
def fill_time : ℝ := 45

/-- The duration of each pipe's operation in a cycle in minutes -/
def cycle_duration : ℝ := 3

/-- Theorem stating that the calculated rate of pipe B is correct -/
theorem pipe_B_rate_correct : 
  pipe_B_rate = (tank_capacity - fill_time / cycle_duration * (pipe_A_rate - pipe_C_rate)) / 
                (fill_time / cycle_duration) :=
by sorry

end NUMINAMATH_CALUDE_pipe_B_rate_correct_l1626_162633


namespace NUMINAMATH_CALUDE_ant_position_after_2020_moves_l1626_162645

/-- Represents the direction the ant is facing -/
inductive Direction
| East
| North
| West
| South

/-- Represents the position and state of the ant -/
structure AntState :=
  (x : Int) (y : Int) (direction : Direction) (moveCount : Nat)

/-- Function to update the ant's state after one move -/
def move (state : AntState) : AntState :=
  match state.direction with
  | Direction.East => { x := state.x + state.moveCount + 1, y := state.y, direction := Direction.North, moveCount := state.moveCount + 1 }
  | Direction.North => { x := state.x, y := state.y + state.moveCount + 1, direction := Direction.West, moveCount := state.moveCount + 1 }
  | Direction.West => { x := state.x - state.moveCount - 1, y := state.y, direction := Direction.South, moveCount := state.moveCount + 1 }
  | Direction.South => { x := state.x, y := state.y - state.moveCount - 1, direction := Direction.East, moveCount := state.moveCount + 1 }

/-- Function to update the ant's state after n moves -/
def moveN (state : AntState) (n : Nat) : AntState :=
  match n with
  | 0 => state
  | Nat.succ m => move (moveN state m)

/-- The main theorem to prove -/
theorem ant_position_after_2020_moves :
  let initialState : AntState := { x := -20, y := 20, direction := Direction.East, moveCount := 0 }
  let finalState := moveN initialState 2020
  finalState.x = -1030 ∧ finalState.y = -990 := by sorry

end NUMINAMATH_CALUDE_ant_position_after_2020_moves_l1626_162645


namespace NUMINAMATH_CALUDE_intersection_theorem_l1626_162646

-- Define the hyperbola and line equations
def hyperbola (x y : ℝ) : Prop := y = 9 / (x^2 + 1)
def line (x y : ℝ) : Prop := x + y = 4

-- Define the intersection points
def intersection_points : Set ℝ := {1, (3 + Real.sqrt 29) / 2, (3 - Real.sqrt 29) / 2}

-- Theorem statement
theorem intersection_theorem :
  ∀ x ∈ intersection_points, ∃ y, hyperbola x y ∧ line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l1626_162646


namespace NUMINAMATH_CALUDE_ellipse_symmetric_points_exist_l1626_162671

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x + 1/3

-- Define point Q
def Q : ℝ × ℝ := (0, 3)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition for symmetry with respect to the line
def symmetric_points (A B : ℝ × ℝ) : Prop :=
  ∃ (P : ℝ × ℝ), line P.1 P.2 ∧ 
    A.1 + B.1 = 2 * P.1 ∧ A.2 + B.2 = 2 * P.2

-- State the theorem
theorem ellipse_symmetric_points_exist : 
  ∃ (A B : ℝ × ℝ), 
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    symmetric_points A B ∧ 
    3 * dot_product (A.1 - Q.1, A.2 - Q.2) (B.1 - Q.1, B.2 - Q.2) = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_points_exist_l1626_162671


namespace NUMINAMATH_CALUDE_remaining_turnovers_l1626_162603

/-- Represents the number of items that can be made from one jar of plums -/
structure PlumJar where
  turnovers : ℕ
  cakes : ℕ
  fruitBarTrays : ℚ

/-- Theorem stating the number of turnovers that can be made with remaining plums -/
theorem remaining_turnovers (jar : PlumJar) 
  (h1 : jar.turnovers = 16)
  (h2 : jar.cakes = 4)
  (h3 : jar.fruitBarTrays = 1/2)
  (h4 : (4 : ℕ) * jar.fruitBarTrays = 1)
  (h5 : (4 : ℕ) * jar.cakes = 6 + (8 : ℕ)) :
  8 = (4 : ℕ) * jar.turnovers - (1 * jar.turnovers + 6 * (jar.turnovers / jar.cakes)) := by
  sorry

end NUMINAMATH_CALUDE_remaining_turnovers_l1626_162603


namespace NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l1626_162630

theorem min_value_x_plus_four_over_x (x : ℝ) (hx : x > 0) :
  x + 4 / x ≥ 4 ∧ ∃ y > 0, y + 4 / y = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l1626_162630


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1626_162656

theorem inequality_and_equality_condition (a b : ℝ) :
  a^2 + b^2 + 1 ≥ a + b + a*b ∧
  (a^2 + b^2 + 1 = a + b + a*b ↔ a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1626_162656


namespace NUMINAMATH_CALUDE_adoption_time_proof_l1626_162690

/-- The number of days required to adopt all puppies -/
def adoptionDays (initialPuppies : ℕ) (additionalPuppies : ℕ) (adoptionRate : ℕ) : ℕ :=
  (initialPuppies + additionalPuppies) / adoptionRate

/-- Theorem stating that it takes 11 days to adopt all puppies under given conditions -/
theorem adoption_time_proof :
  adoptionDays 15 62 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_adoption_time_proof_l1626_162690


namespace NUMINAMATH_CALUDE_m_range_l1626_162609

def p (m : ℝ) : Prop := ∀ x, |x - m| + |x - 1| > 1

def q (m : ℝ) : Prop := ∀ x > 0, (fun x => Real.log x / Real.log (3 + m)) x > 0

theorem m_range : 
  (∃ m : ℝ, (¬(p m ∧ q m)) ∧ (p m ∨ q m)) ↔ 
  (∃ m : ℝ, (-3 < m ∧ m < -2) ∨ (0 ≤ m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l1626_162609


namespace NUMINAMATH_CALUDE_intersection_A_B_l1626_162624

def A : Set ℤ := {-3, -2, -1, 0, 1}
def B : Set ℤ := {x | x^2 - 4 = 0}

theorem intersection_A_B : A ∩ B = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1626_162624


namespace NUMINAMATH_CALUDE_sunset_delay_theorem_l1626_162691

/-- Calculates the minutes until sunset given the initial sunset time,
    daily sunset delay, days passed, and current time. -/
def minutesUntilSunset (initialSunsetMinutes : ℕ) (dailyDelayMinutes : ℚ)
                       (daysPassed : ℕ) (currentTimeMinutes : ℕ) : ℚ :=
  let newSunsetMinutes : ℚ := initialSunsetMinutes + daysPassed * dailyDelayMinutes
  newSunsetMinutes - currentTimeMinutes

/-- Proves that 40 days after March 1st, at 6:10 PM, 
    there are 38 minutes until sunset. -/
theorem sunset_delay_theorem :
  minutesUntilSunset 1080 1.2 40 1090 = 38 := by
  sorry

#eval minutesUntilSunset 1080 1.2 40 1090

end NUMINAMATH_CALUDE_sunset_delay_theorem_l1626_162691


namespace NUMINAMATH_CALUDE_square_of_product_72519_9999_l1626_162699

theorem square_of_product_72519_9999 : (72519 * 9999)^2 = 525545577128752961 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_72519_9999_l1626_162699


namespace NUMINAMATH_CALUDE_common_solution_y_value_l1626_162665

theorem common_solution_y_value (x y : ℝ) 
  (eq1 : x^2 + y^2 = 25) 
  (eq2 : x^2 + y = 10) : 
  y = (1 - Real.sqrt 61) / 2 := by
sorry

end NUMINAMATH_CALUDE_common_solution_y_value_l1626_162665


namespace NUMINAMATH_CALUDE_baker_pastries_l1626_162610

theorem baker_pastries (cakes : ℕ) (pastry_difference : ℕ) : 
  cakes = 19 → pastry_difference = 112 → cakes + pastry_difference = 131 := by
  sorry

end NUMINAMATH_CALUDE_baker_pastries_l1626_162610


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1626_162659

theorem least_subtraction_for_divisibility (n m : ℕ) : 
  ∃ k, k ≤ m ∧ (n - k) % m = 0 ∧ ∀ j, j < k → (n - j) % m ≠ 0 :=
by sorry

theorem problem_solution : 
  ∃ k, k ≤ 87 ∧ (13604 - k) % 87 = 0 ∧ ∀ j, j < k → (13604 - j) % 87 ≠ 0 ∧ k = 32 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1626_162659


namespace NUMINAMATH_CALUDE_percentage_both_correct_l1626_162650

theorem percentage_both_correct (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) :
  p_first = 0.63 →
  p_second = 0.49 →
  p_neither = 0.20 →
  p_first + p_second - (1 - p_neither) = 0.32 := by
sorry

end NUMINAMATH_CALUDE_percentage_both_correct_l1626_162650


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l1626_162678

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function y = (x + 1)(x - a) -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (x + 1) * (x - a)

theorem even_function_implies_a_zero (a : ℝ) :
  IsEven (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l1626_162678


namespace NUMINAMATH_CALUDE_debby_water_bottles_l1626_162681

/-- The number of water bottles Debby drank per day -/
def bottles_per_day : ℕ := 109

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := 74

/-- The total number of bottles Debby bought -/
def total_bottles : ℕ := bottles_per_day * days_lasted

theorem debby_water_bottles : total_bottles = 8066 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l1626_162681
