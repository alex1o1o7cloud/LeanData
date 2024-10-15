import Mathlib

namespace NUMINAMATH_CALUDE_mans_speed_against_current_l955_95537

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that given the specific speeds, 
    the man's speed against the current is 8.6 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 15 3.2 = 8.6 := by
  sorry


end NUMINAMATH_CALUDE_mans_speed_against_current_l955_95537


namespace NUMINAMATH_CALUDE_stating_boat_speed_with_stream_l955_95536

/-- Represents the speed of a boat in different conditions. -/
structure BoatSpeed where
  stillWater : ℝ
  againstStream : ℝ
  withStream : ℝ

/-- 
Theorem stating that given a man's rowing speed in still water is 6 km/h 
and his speed against the stream is 10 km/h, his speed with the stream is 10 km/h.
-/
theorem boat_speed_with_stream 
  (speed : BoatSpeed) 
  (h1 : speed.stillWater = 6) 
  (h2 : speed.againstStream = 10) : 
  speed.withStream = 10 := by
  sorry

#check boat_speed_with_stream

end NUMINAMATH_CALUDE_stating_boat_speed_with_stream_l955_95536


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l955_95559

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 64 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l955_95559


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l955_95547

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) (h2 : x * y = -4) : x^2 + y^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l955_95547


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l955_95545

def initial_amount : ℚ := 150

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_amount : ℚ := initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction)

theorem jennifer_remaining_money :
  remaining_amount = 20 := by sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l955_95545


namespace NUMINAMATH_CALUDE_cistern_fill_time_l955_95556

/-- Time to fill cistern with all pipes open simultaneously -/
theorem cistern_fill_time (fill_time_A fill_time_B empty_time_C : ℝ) 
  (h_A : fill_time_A = 45)
  (h_B : fill_time_B = 60)
  (h_C : empty_time_C = 72) : 
  (1 / ((1 / fill_time_A) + (1 / fill_time_B) - (1 / empty_time_C))) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l955_95556


namespace NUMINAMATH_CALUDE_inequality_solution_set_l955_95546

theorem inequality_solution_set (x : ℝ) :
  (x + 5) / (x^2 + 3*x + 9) ≥ 0 ↔ x ≥ -5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l955_95546


namespace NUMINAMATH_CALUDE_total_bales_at_end_of_week_l955_95505

def initial_bales : ℕ := 28
def daily_additions : List ℕ := [10, 15, 8, 20, 12, 4, 18]

theorem total_bales_at_end_of_week : 
  initial_bales + daily_additions.sum = 115 := by
  sorry

end NUMINAMATH_CALUDE_total_bales_at_end_of_week_l955_95505


namespace NUMINAMATH_CALUDE_min_payment_amount_l955_95561

/-- Represents the number of bills of each denomination --/
structure BillCount where
  tens : Nat
  fives : Nat
  ones : Nat

/-- Calculates the total value of bills --/
def totalValue (bills : BillCount) : Nat :=
  10 * bills.tens + 5 * bills.fives + bills.ones

/-- Calculates the total count of bills --/
def totalCount (bills : BillCount) : Nat :=
  bills.tens + bills.fives + bills.ones

/-- Represents Tim's initial bill distribution --/
def timsBills : BillCount :=
  { tens := 13, fives := 11, ones := 17 }

/-- Theorem stating the minimum amount Tim can pay using at least 16 bills --/
theorem min_payment_amount (payment : BillCount) : 
  totalCount payment ≥ 16 → 
  totalCount payment ≤ totalCount timsBills → 
  totalValue payment ≥ 40 :=
by sorry

end NUMINAMATH_CALUDE_min_payment_amount_l955_95561


namespace NUMINAMATH_CALUDE_ginas_account_fractions_l955_95595

theorem ginas_account_fractions (betty_balance : ℝ) (gina_combined_balance : ℝ)
  (h1 : betty_balance = 3456)
  (h2 : gina_combined_balance = 1728) :
  ∃ (f1 f2 : ℝ), f1 + f2 = 1/2 ∧ f1 * betty_balance + f2 * betty_balance = gina_combined_balance :=
by sorry

end NUMINAMATH_CALUDE_ginas_account_fractions_l955_95595


namespace NUMINAMATH_CALUDE_max_m_value_l955_95575

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → (4 / (1 - x)) ≥ m - (1 / x)) → 
  m ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_max_m_value_l955_95575


namespace NUMINAMATH_CALUDE_jakes_earnings_l955_95597

/-- Jake's lawn mowing and flower planting problem -/
theorem jakes_earnings (mowing_time mowing_pay planting_time desired_rate : ℝ) 
  (h1 : mowing_time = 1)
  (h2 : mowing_pay = 15)
  (h3 : planting_time = 2)
  (h4 : desired_rate = 20) :
  let total_time := mowing_time + planting_time
  let total_desired_earnings := desired_rate * total_time
  let planting_charge := total_desired_earnings - mowing_pay
  planting_charge = 45 := by sorry

end NUMINAMATH_CALUDE_jakes_earnings_l955_95597


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l955_95541

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  -- Angle between faces ABC and BCD
  angle : ℝ
  -- Area of face ABC
  area_ABC : ℝ
  -- Area of face BCD
  area_BCD : ℝ
  -- Length of edge BC
  length_BC : ℝ

/-- The volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the tetrahedron with given properties -/
theorem tetrahedron_volume :
  ∀ t : Tetrahedron,
    t.angle = π/4 ∧
    t.area_ABC = 150 ∧
    t.area_BCD = 100 ∧
    t.length_BC = 12 →
    volume t = (1250 * Real.sqrt 2) / 3 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l955_95541


namespace NUMINAMATH_CALUDE_two_integers_sum_l955_95527

theorem two_integers_sum (a b : ℕ+) : 
  (a : ℤ) - (b : ℤ) = 3 → a * b = 63 → (a : ℤ) + (b : ℤ) = 17 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_l955_95527


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l955_95543

theorem cubic_equation_solutions :
  ∀ (z : ℂ), z^3 = -27 ↔ z = -3 ∨ z = (3 / 2 : ℂ) + (3 / 2 : ℂ) * Complex.I * Real.sqrt 3 ∨ z = (3 / 2 : ℂ) - (3 / 2 : ℂ) * Complex.I * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l955_95543


namespace NUMINAMATH_CALUDE_parking_lot_revenue_l955_95571

/-- Given a parking lot with the following properties:
  * Total spaces: 1000
  * Section 1: 320 spaces at $5 per hour
  * Section 2: 200 more spaces than Section 3 at $8 per hour
  * Section 3: Remaining spaces at $4 per hour
  Prove that Section 2 has 440 spaces and the total revenue for 5 hours is $30400 -/
theorem parking_lot_revenue 
  (total_spaces : Nat) 
  (section1_spaces : Nat) 
  (section2_price : Nat) 
  (section3_price : Nat) 
  (section1_price : Nat) 
  (hours : Nat) :
  total_spaces = 1000 →
  section1_spaces = 320 →
  section2_price = 8 →
  section3_price = 4 →
  section1_price = 5 →
  hours = 5 →
  ∃ (section2_spaces section3_spaces : Nat),
    section2_spaces = section3_spaces + 200 ∧
    section1_spaces + section2_spaces + section3_spaces = total_spaces ∧
    section2_spaces = 440 ∧
    section1_spaces * section1_price * hours + 
    section2_spaces * section2_price * hours + 
    section3_spaces * section3_price * hours = 30400 := by
  sorry


end NUMINAMATH_CALUDE_parking_lot_revenue_l955_95571


namespace NUMINAMATH_CALUDE_opposite_numbers_properties_l955_95548

theorem opposite_numbers_properties :
  (∀ a b : ℝ, a = -b → a + b = 0) ∧
  (∀ a b : ℝ, a + b = 0 → a = -b) ∧
  (∀ a b : ℝ, b ≠ 0 → (a / b = -1 → a = -b)) :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_properties_l955_95548


namespace NUMINAMATH_CALUDE_expression_evaluation_l955_95558

theorem expression_evaluation :
  68 + (156 / 12) + (11 * 19) - 250 - (450 / 9) = -10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l955_95558


namespace NUMINAMATH_CALUDE_discount_relationship_l955_95577

/-- Represents the banker's discount in Rupees -/
def bankers_discount : ℝ := 78

/-- Represents the true discount in Rupees -/
def true_discount : ℝ := 66

/-- Represents the sum due (present value) in Rupees -/
def sum_due : ℝ := 363

/-- Theorem stating the relationship between banker's discount, true discount, and sum due -/
theorem discount_relationship : 
  bankers_discount = true_discount + (true_discount^2 / sum_due) :=
by sorry

end NUMINAMATH_CALUDE_discount_relationship_l955_95577


namespace NUMINAMATH_CALUDE_division_remainder_problem_l955_95511

theorem division_remainder_problem (P D Q R D' Q' R' C : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  : P % (D * D') = D * R' + R + C := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l955_95511


namespace NUMINAMATH_CALUDE_pauls_score_l955_95573

theorem pauls_score (total_points cousin_points : ℕ) 
  (h1 : total_points = 5816)
  (h2 : cousin_points = 2713) :
  total_points - cousin_points = 3103 := by
  sorry

end NUMINAMATH_CALUDE_pauls_score_l955_95573


namespace NUMINAMATH_CALUDE_airplane_seats_l955_95524

theorem airplane_seats :
  ∀ (total_seats : ℝ),
  (30 : ℝ) + 0.2 * total_seats + 0.75 * total_seats = total_seats →
  total_seats = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l955_95524


namespace NUMINAMATH_CALUDE_factorization_of_4m_squared_minus_64_l955_95560

theorem factorization_of_4m_squared_minus_64 (m : ℝ) : 4 * m^2 - 64 = 4 * (m + 4) * (m - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4m_squared_minus_64_l955_95560


namespace NUMINAMATH_CALUDE_x_over_y_equals_two_l955_95500

theorem x_over_y_equals_two (x y : ℝ) 
  (h1 : 3 < (x^2 - y^2) / (x^2 + y^2))
  (h2 : (x^2 - y^2) / (x^2 + y^2) < 4)
  (h3 : ∃ (n : ℤ), x / y = n) :
  x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_equals_two_l955_95500


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l955_95567

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im ((1 + 2*i) / (2 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l955_95567


namespace NUMINAMATH_CALUDE_impossible_configuration_l955_95557

/-- Represents the sign at a vertex -/
inductive Sign
| Plus
| Minus

/-- Represents the state of the 12-gon -/
def TwelveGonState := Fin 12 → Sign

/-- Initial state of the 12-gon -/
def initialState : TwelveGonState :=
  fun i => if i = 0 then Sign.Minus else Sign.Plus

/-- Applies an operation to change signs at consecutive vertices -/
def applyOperation (state : TwelveGonState) (start : Fin 12) (count : Nat) : TwelveGonState :=
  fun i => if (i - start) % 12 < count then
    match state i with
    | Sign.Plus => Sign.Minus
    | Sign.Minus => Sign.Plus
    else state i

/-- Checks if the state matches the target configuration -/
def isTargetState (state : TwelveGonState) : Prop :=
  state 1 = Sign.Minus ∧ ∀ i : Fin 12, i ≠ 1 → state i = Sign.Plus

/-- The main theorem to be proved -/
theorem impossible_configuration
  (n : Nat)
  (h : n = 6 ∨ n = 4 ∨ n = 3)
  : ¬ ∃ (operations : List (Fin 12)), 
    let finalState := operations.foldl (fun s (start : Fin 12) => applyOperation s start n) initialState
    isTargetState finalState :=
sorry

end NUMINAMATH_CALUDE_impossible_configuration_l955_95557


namespace NUMINAMATH_CALUDE_cubic_equation_root_l955_95512

theorem cubic_equation_root (c d : ℚ) : 
  (3 + Real.sqrt 5 : ℂ) ^ 3 + c * (3 + Real.sqrt 5 : ℂ) ^ 2 + d * (3 + Real.sqrt 5 : ℂ) + 15 = 0 →
  d = -18.5 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l955_95512


namespace NUMINAMATH_CALUDE_distribute_five_projects_three_teams_l955_95508

/-- The number of ways to distribute n distinct projects among k teams,
    where each team must receive at least one project. -/
def distribute_projects (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 projects among 3 teams results in 60 arrangements -/
theorem distribute_five_projects_three_teams :
  distribute_projects 5 3 = 60 := by sorry

end NUMINAMATH_CALUDE_distribute_five_projects_three_teams_l955_95508


namespace NUMINAMATH_CALUDE_waiter_tips_l955_95538

/-- Calculates the total tips earned by a waiter --/
def total_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Proves that the waiter earned $15 in tips --/
theorem waiter_tips : total_tips 10 5 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_l955_95538


namespace NUMINAMATH_CALUDE_tenth_order_magic_constant_l955_95553

/-- The magic constant of an nth-order magic square -/
def magic_constant (n : ℕ) : ℕ :=
  (n * (n^2 + 1)) / 2

/-- Theorem: The magic constant of a 10th-order magic square is 505 -/
theorem tenth_order_magic_constant :
  magic_constant 10 = 505 := by
  sorry

#eval magic_constant 10  -- This will evaluate to 505

end NUMINAMATH_CALUDE_tenth_order_magic_constant_l955_95553


namespace NUMINAMATH_CALUDE_may_savings_l955_95568

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (month 0)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_may_savings_l955_95568


namespace NUMINAMATH_CALUDE_trig_expression_equality_l955_95565

theorem trig_expression_equality : 
  (Real.sin (π/4) + Real.cos (π/6)) / (3 - 2 * Real.cos (π/3)) - 
  Real.sin (π/3) * (1 - Real.sin (π/6)) = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l955_95565


namespace NUMINAMATH_CALUDE_barrel_capacity_l955_95540

def number_of_barrels : ℕ := 4
def flow_rate : ℚ := 7/2
def fill_time : ℕ := 8

theorem barrel_capacity : 
  (flow_rate * fill_time) / number_of_barrels = 7 := by sorry

end NUMINAMATH_CALUDE_barrel_capacity_l955_95540


namespace NUMINAMATH_CALUDE_hyperbola_equation_l955_95517

/-- Given a hyperbola with the general equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is x - √3 y = 0 and one of its foci is on the directrix
    of the parabola y² = -4x, then its equation is 4/3 x² - 4y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (asymptote : ∀ x y : ℝ, x - Real.sqrt 3 * y = 0 → x^2 / a^2 - y^2 / b^2 = 1)
  (focus_on_directrix : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y^2 = -4*x ∧ x = 1) :
  ∀ x y : ℝ, 4/3 * x^2 - 4 * y^2 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l955_95517


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l955_95520

theorem triangle_max_perimeter (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  A = Real.pi / 3 →
  0 < B →
  B < 2 * Real.pi / 3 →
  0 < C →
  C < 2 * Real.pi / 3 →
  A + B + C = Real.pi →
  b = 2 * Real.sin B →
  c = 2 * Real.sin C →
  a + b + c ≤ 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l955_95520


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l955_95539

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 2016) :
  (∃ k : ℕ, k = 334 ∧ 
   (∀ m : ℕ, n^m ∣ n! ↔ m ≤ k)) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l955_95539


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_x_coord_l955_95521

/-- Given vectors a and b in R², if a is perpendicular to (a - b), then the x-coordinate of b is 9. -/
theorem perpendicular_vectors_imply_x_coord (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.2 = -2 →
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) →
  b.1 = 9 := by
  sorry

#check perpendicular_vectors_imply_x_coord

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_x_coord_l955_95521


namespace NUMINAMATH_CALUDE_carol_can_invite_198_friends_l955_95506

/-- The number of invitations in each package -/
def invitations_per_pack : ℕ := 18

/-- The number of packs Carol bought -/
def packs_bought : ℕ := 11

/-- The total number of friends Carol can invite -/
def friends_to_invite : ℕ := invitations_per_pack * packs_bought

/-- Theorem stating that Carol can invite 198 friends -/
theorem carol_can_invite_198_friends : friends_to_invite = 198 := by
  sorry

end NUMINAMATH_CALUDE_carol_can_invite_198_friends_l955_95506


namespace NUMINAMATH_CALUDE_nina_running_distance_l955_95555

theorem nina_running_distance : 
  let first_run : ℝ := 0.08
  let second_run_part1 : ℝ := 0.08
  let second_run_part2 : ℝ := 0.67
  first_run + second_run_part1 + second_run_part2 = 0.83 := by
sorry

end NUMINAMATH_CALUDE_nina_running_distance_l955_95555


namespace NUMINAMATH_CALUDE_cubic_equation_root_l955_95534

theorem cubic_equation_root : 
  ∃ (x : ℝ), x = -4/3 ∧ (x + 1)^(1/3) + (2*x + 3)^(1/3) + 3*x + 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l955_95534


namespace NUMINAMATH_CALUDE_dart_board_probability_l955_95515

/-- The probability of a dart landing within the center square of a regular hexagon dart board -/
theorem dart_board_probability (x : ℝ) (x_pos : x > 0) : 
  let hexagon_area := 3 * Real.sqrt 3 * x^2 / 2
  let square_area := 3 * x^2 / 4
  square_area / hexagon_area = 1 / (2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_dart_board_probability_l955_95515


namespace NUMINAMATH_CALUDE_multiplication_mistake_problem_l955_95526

theorem multiplication_mistake_problem :
  ∃ x : ℝ, (493 * x - 394 * x = 78426) ∧ (x = 792) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_problem_l955_95526


namespace NUMINAMATH_CALUDE_polygon_not_covered_by_homothetic_polygons_l955_95589

/-- A polygon in a 2D space -/
structure Polygon where
  vertices : List (ℝ × ℝ)
  -- Add more properties as needed

/-- Homothetic transformation of a polygon -/
def homothetic_transform (p : Polygon) (center : ℝ × ℝ) (k : ℝ) : Polygon :=
  sorry

/-- Predicate to check if a point is contained in a polygon -/
def point_in_polygon (point : ℝ × ℝ) (p : Polygon) : Prop :=
  sorry

theorem polygon_not_covered_by_homothetic_polygons 
  (M : Polygon) (k : ℝ) (h : 0 < k ∧ k < 1) :
  ∃ (point : ℝ × ℝ), 
    point_in_polygon point M ∧
    ∀ (center1 center2 : ℝ × ℝ),
      ¬(point_in_polygon point (homothetic_transform M center1 k) ∨
        point_in_polygon point (homothetic_transform M center2 k)) :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_not_covered_by_homothetic_polygons_l955_95589


namespace NUMINAMATH_CALUDE_sum_of_obtuse_angles_l955_95501

open Real

theorem sum_of_obtuse_angles (α β : Real) : 
  π < α ∧ α < 2*π → 
  π < β ∧ β < 2*π → 
  sin α = sqrt 5 / 5 → 
  cos β = -(3 * sqrt 10) / 10 → 
  α + β = 7 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_obtuse_angles_l955_95501


namespace NUMINAMATH_CALUDE_derivative_implies_function_l955_95504

open Real

theorem derivative_implies_function (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, deriv f x = 1 + cos x) →
  ∃ C, ∀ x, f x = x + sin x + C :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_implies_function_l955_95504


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l955_95514

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l955_95514


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l955_95599

theorem unique_congruence_in_range : ∃! n : ℕ, 3 ≤ n ∧ n ≤ 10 ∧ n % 7 = 10573 % 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l955_95599


namespace NUMINAMATH_CALUDE_profit_percentage_invariance_l955_95516

theorem profit_percentage_invariance 
  (cost_price : ℝ) 
  (discount_percentage : ℝ) 
  (final_profit_percentage : ℝ) 
  (discount_percentage_pos : 0 < discount_percentage) 
  (discount_percentage_lt_100 : discount_percentage < 100) 
  (final_profit_percentage_pos : 0 < final_profit_percentage) :
  let selling_price := cost_price * (1 + final_profit_percentage / 100)
  let discounted_price := selling_price * (1 - discount_percentage / 100)
  let profit_without_discount := (selling_price - cost_price) / cost_price * 100
  profit_without_discount = final_profit_percentage := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_invariance_l955_95516


namespace NUMINAMATH_CALUDE_hexagon_side_length_l955_95544

theorem hexagon_side_length (perimeter : ℝ) (h : perimeter = 30) : 
  perimeter / 6 = 5 := by sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l955_95544


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l955_95542

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1)
  (h3 : ∀ n : ℕ, a (n + 2) - a n = 6) :
  a 11 = 61 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l955_95542


namespace NUMINAMATH_CALUDE_certain_number_proof_l955_95522

theorem certain_number_proof (x : ℤ) : x + 34 - 53 = 28 ↔ x = 47 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l955_95522


namespace NUMINAMATH_CALUDE_square_sum_greater_than_one_l955_95583

theorem square_sum_greater_than_one
  (x y z t : ℝ)
  (h : (x^2 + y^2 - 1) * (z^2 + t^2 - 1) > (x*z + y*t - 1)^2) :
  x^2 + y^2 > 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_greater_than_one_l955_95583


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l955_95587

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, is_two_digit n ∧ 17 ∣ n → 34 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l955_95587


namespace NUMINAMATH_CALUDE_num_purchasing_methods_eq_seven_l955_95503

/-- The number of purchasing methods for equipment types A and B -/
def num_purchasing_methods : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let (x, y) := p
    600000 * x + 700000 * y ≤ 5000000 ∧
    x ≥ 3 ∧
    y ≥ 2
  ) (Finset.product (Finset.range 10) (Finset.range 10))).card

/-- Theorem stating that the number of purchasing methods is 7 -/
theorem num_purchasing_methods_eq_seven :
  num_purchasing_methods = 7 := by sorry

end NUMINAMATH_CALUDE_num_purchasing_methods_eq_seven_l955_95503


namespace NUMINAMATH_CALUDE_only_contrapositive_correct_l955_95584

theorem only_contrapositive_correct (p q r : Prop) 
  (h : (p ∨ q) → ¬r) : 
  (¬((p ∨ q) → ¬r) ∧ 
   ¬(¬r → p) ∧ 
   ¬(r → ¬(p ∨ q)) ∧ 
   ((¬p ∧ ¬q) → r)) := by
  sorry

end NUMINAMATH_CALUDE_only_contrapositive_correct_l955_95584


namespace NUMINAMATH_CALUDE_polly_tweet_time_l955_95578

/-- Represents the number of tweets per minute in different emotional states -/
structure TweetsPerMinute where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the total number of tweets and the time spent in each state -/
structure TweetData where
  tweets_per_minute : TweetsPerMinute
  total_tweets : ℕ
  time_per_state : ℕ

/-- Theorem: Given Polly's tweet rates and total tweets, prove the time spent in each state -/
theorem polly_tweet_time (data : TweetData)
  (h1 : data.tweets_per_minute.happy = 18)
  (h2 : data.tweets_per_minute.hungry = 4)
  (h3 : data.tweets_per_minute.mirror = 45)
  (h4 : data.total_tweets = 1340)
  (h5 : data.time_per_state * (data.tweets_per_minute.happy + data.tweets_per_minute.hungry + data.tweets_per_minute.mirror) = data.total_tweets) :
  data.time_per_state = 20 := by
  sorry


end NUMINAMATH_CALUDE_polly_tweet_time_l955_95578


namespace NUMINAMATH_CALUDE_original_eq_general_form_l955_95564

/-- The original quadratic equation -/
def original_equation (x : ℝ) : ℝ := 2 * (x + 2)^2 + (x + 3) * (x - 2) + 11

/-- The general form of the quadratic equation -/
def general_form (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 13

/-- Theorem stating the equivalence of the original equation and its general form -/
theorem original_eq_general_form :
  ∀ x, original_equation x = general_form x := by sorry

end NUMINAMATH_CALUDE_original_eq_general_form_l955_95564


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l955_95580

theorem quadratic_equation_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x - 2 = 0 ∧ x = 2) → 
  (b = -1 ∧ ∃ y : ℝ, y^2 + b*y - 2 = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l955_95580


namespace NUMINAMATH_CALUDE_incorrect_number_correction_l955_95513

theorem incorrect_number_correction (n : ℕ) (incorrect_avg correct_avg incorrect_num : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 46)
  (h3 : incorrect_num = 25)
  (h4 : correct_avg = 50) :
  ∃ (actual_num : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * incorrect_avg = actual_num - incorrect_num ∧ 
    actual_num = 65 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_number_correction_l955_95513


namespace NUMINAMATH_CALUDE_solution_difference_l955_95523

theorem solution_difference (r s : ℝ) : 
  ((r - 5) * (r + 5) = 25 * r - 125) →
  ((s - 5) * (s + 5) = 25 * s - 125) →
  r ≠ s →
  r > s →
  r - s = 15 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l955_95523


namespace NUMINAMATH_CALUDE_students_per_bus_l955_95528

theorem students_per_bus (total_students : ℕ) (num_buses : ℕ) (h1 : total_students = 360) (h2 : num_buses = 8) :
  total_students / num_buses = 45 := by
  sorry

end NUMINAMATH_CALUDE_students_per_bus_l955_95528


namespace NUMINAMATH_CALUDE_dessert_cost_calculation_dessert_cost_is_eleven_l955_95552

/-- Calculates the cost of a dessert given the costs of other meal components and the total price --/
theorem dessert_cost_calculation 
  (appetizer_cost : ℝ) 
  (entree_cost : ℝ) 
  (tip_percentage : ℝ) 
  (total_price : ℝ) : ℝ :=
  let base_cost := appetizer_cost + 2 * entree_cost
  let dessert_cost := (total_price - base_cost) / (1 + tip_percentage)
  dessert_cost

/-- Proves that the dessert cost is $11.00 given the specific meal costs --/
theorem dessert_cost_is_eleven :
  dessert_cost_calculation 9 20 0.3 78 = 11 := by
  sorry

end NUMINAMATH_CALUDE_dessert_cost_calculation_dessert_cost_is_eleven_l955_95552


namespace NUMINAMATH_CALUDE_percentage_problem_l955_95533

theorem percentage_problem (n : ℝ) (h : 1.2 * n = 6000) : 0.2 * n = 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l955_95533


namespace NUMINAMATH_CALUDE_congruence_problem_l955_95535

theorem congruence_problem (n : ℤ) : 
  3 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [ZMOD 6] → n = 3 ∨ n = 9 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l955_95535


namespace NUMINAMATH_CALUDE_number_of_women_l955_95574

-- Define the total number of family members
def total_members : ℕ := 15

-- Define the time it takes for a woman to complete the work
def woman_work_days : ℕ := 180

-- Define the time it takes for a man to complete the work
def man_work_days : ℕ := 120

-- Define the time it takes to complete the work with alternating schedule
def alternating_work_days : ℕ := 17

-- Define the function to calculate the number of women
def calculate_women (total : ℕ) (woman_days : ℕ) (man_days : ℕ) (alt_days : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem number_of_women : 
  calculate_women total_members woman_work_days man_work_days alternating_work_days = 3 :=
sorry

end NUMINAMATH_CALUDE_number_of_women_l955_95574


namespace NUMINAMATH_CALUDE_mens_wages_l955_95554

/-- Proves that the wage of one man is 24 Rs given the problem conditions -/
theorem mens_wages (men women boys : ℕ) (total_earnings : ℚ) : 
  men = 5 → 
  boys = 8 → 
  total_earnings = 120 → 
  ∃ (w : ℕ), (5 : ℚ) * (total_earnings / (men + w + boys)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_mens_wages_l955_95554


namespace NUMINAMATH_CALUDE_impossible_sums_l955_95529

-- Define the coin values
def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

-- Define the set of possible coin values
def coin_values : Set ℕ := {penny, nickel, dime, quarter}

-- Define a function to check if a sum is possible with 5 coins
def is_possible_sum (sum : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), 
    a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ d ∈ coin_values ∧ e ∈ coin_values ∧
    a + b + c + d + e = sum

-- Theorem statement
theorem impossible_sums : ¬(is_possible_sum 22) ∧ ¬(is_possible_sum 48) :=
sorry

end NUMINAMATH_CALUDE_impossible_sums_l955_95529


namespace NUMINAMATH_CALUDE_parabola_tangent_slope_l955_95579

-- Define the parabola
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 9

-- Define the derivative of the parabola
def parabola_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem parabola_tangent_slope (a b : ℝ) :
  parabola a b 2 = -1 →
  parabola_derivative a b 2 = 1 →
  a = 3 ∧ b = -11 := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_slope_l955_95579


namespace NUMINAMATH_CALUDE_vector_expression_simplification_l955_95591

variable {V : Type*} [AddCommGroup V]

/-- For any four points A, B, C, D in a vector space,
    the expression AC - BD + CD - AB equals the zero vector. -/
theorem vector_expression_simplification
  (A B C D : V) : (C - A) - (D - B) + (D - C) - (B - A) = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_simplification_l955_95591


namespace NUMINAMATH_CALUDE_sixth_term_is_thirteen_l955_95569

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 7 ∧ 
  a 5 = a 2 + 6

/-- The 6th term of the arithmetic sequence is 13 -/
theorem sixth_term_is_thirteen 
  (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) : 
  a 6 = 13 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_is_thirteen_l955_95569


namespace NUMINAMATH_CALUDE_no_solution_for_all_a_b_l955_95596

theorem no_solution_for_all_a_b : ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧
  ¬∃ (x y : ℝ), (Real.tan (13 * x) * Real.tan (a * y) = 1) ∧
                (Real.tan (21 * x) * Real.tan (b * y) = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_all_a_b_l955_95596


namespace NUMINAMATH_CALUDE_main_theorem_l955_95572

noncomputable section

variable (e : ℝ)
variable (f : ℝ → ℝ)

-- Define the conditions
def non_negative (f : ℝ → ℝ) : Prop := ∀ x ∈ Set.Icc 0 e, f x ≥ 0
def f_e_equals_e : Prop := f e = e
def superadditive (f : ℝ → ℝ) : Prop := 
  ∀ x₁ x₂, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ + x₂ ≤ e → f (x₁ + x₂) ≥ f x₁ + f x₂

-- Define the inequality condition
def inequality_condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 e, 4 * (f x)^2 - 4 * (2 * e - a) * f x + 4 * e^2 - 4 * e * a + 1 ≥ 0

-- Main theorem
theorem main_theorem (h1 : non_negative e f) (h2 : f_e_equals_e e f) (h3 : superadditive e f) :
  (f 0 = 0) ∧
  (∀ x ∈ Set.Icc 0 e, f x ≤ e) ∧
  (∀ a : ℝ, inequality_condition e f a → a ≤ e) := by
  sorry

end

end NUMINAMATH_CALUDE_main_theorem_l955_95572


namespace NUMINAMATH_CALUDE_botany_zoology_ratio_l955_95563

/-- Represents the number of books in Milton's collection. -/
structure BookCollection where
  total : ℕ
  zoology : ℕ
  botany : ℕ
  h_total : total = zoology + botany
  h_botany_multiple : ∃ n : ℕ, botany = n * zoology

/-- The ratio of botany books to zoology books in Milton's collection is 4:1. -/
theorem botany_zoology_ratio (collection : BookCollection)
    (h_total : collection.total = 80)
    (h_zoology : collection.zoology = 16) :
    collection.botany / collection.zoology = 4 := by
  sorry

end NUMINAMATH_CALUDE_botany_zoology_ratio_l955_95563


namespace NUMINAMATH_CALUDE_zeros_in_99999_cubed_l955_95593

-- Define a function to count zeros in a number
def count_zeros (n : ℕ) : ℕ := sorry

-- Define a function to count digits in a number
def count_digits (n : ℕ) : ℕ := sorry

-- Define the given conditions
axiom zeros_9 : count_zeros (9^3) = 0
axiom zeros_99 : count_zeros (99^3) = 2
axiom zeros_999 : count_zeros (999^3) = 3

-- Define the pattern continuation
axiom pattern_continuation (n : ℕ) : 
  n > 999 → count_zeros (n^3) = count_digits n

-- The theorem to prove
theorem zeros_in_99999_cubed : 
  count_zeros ((99999 : ℕ)^3) = count_digits 99999 := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_99999_cubed_l955_95593


namespace NUMINAMATH_CALUDE_property_price_calculation_l955_95531

/-- Calculate the total price of a property given the price per square foot and the sizes of the house and barn. -/
theorem property_price_calculation
  (price_per_sq_ft : ℕ)
  (house_size : ℕ)
  (barn_size : ℕ)
  (h1 : price_per_sq_ft = 98)
  (h2 : house_size = 2400)
  (h3 : barn_size = 1000) :
  price_per_sq_ft * (house_size + barn_size) = 333200 := by
  sorry

#eval 98 * (2400 + 1000) -- Sanity check

end NUMINAMATH_CALUDE_property_price_calculation_l955_95531


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l955_95570

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a + a₁*(x + 2) + a₂*(x + 2)^2 + a₃*(x + 2)^3 + 
    a₄*(x + 2)^4 + a₅*(x + 2)^5 + a₆*(x + 2)^6 + a₇*(x + 2)^7 + a₈*(x + 2)^8 + 
    a₉*(x + 2)^9 + a₁₀*(x + 2)^10 + a₁₁*(x + 2)^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l955_95570


namespace NUMINAMATH_CALUDE_area_between_circles_l955_95594

/-- The area between a circumscribing circle and two externally tangent circles -/
theorem area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) : 
  let R := (r₁ + r₂) / 2
  π * R^2 - (π * r₁^2 + π * r₂^2) = 24 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_circles_l955_95594


namespace NUMINAMATH_CALUDE_intersection_and_parallel_perpendicular_lines_l955_95502

-- Define the lines
def l₁ (x y : ℝ) : Prop := x - 2*y + 4 = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0
def l₃ (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem intersection_and_parallel_perpendicular_lines :
  (∀ x y, l₁ x y ∧ l₂ x y ↔ (x, y) = P) ∧
  (∀ x y, 3*x - 4*y + 8 = 0 ↔ (∃ t, (x, y) = (t*3 + P.1, t*4 + P.2))) ∧
  (∀ x y, 4*x + 3*y - 6 = 0 ↔ (∃ t, (x, y) = (t*4 + P.1, -t*3 + P.2))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_perpendicular_lines_l955_95502


namespace NUMINAMATH_CALUDE_veridux_managers_count_l955_95507

/-- Veridux Corporation employee structure -/
structure VeriduxCorp where
  total_employees : ℕ
  female_employees : ℕ
  male_associates : ℕ
  female_managers : ℕ

/-- Theorem: The total number of managers at Veridux Corporation is 40 -/
theorem veridux_managers_count (v : VeriduxCorp)
  (h1 : v.total_employees = 250)
  (h2 : v.female_employees = 90)
  (h3 : v.male_associates = 160)
  (h4 : v.female_managers = 40)
  (h5 : v.total_employees = v.female_employees + (v.male_associates + v.female_managers)) :
  v.female_managers + (v.total_employees - v.female_employees - v.male_associates) = 40 := by
  sorry

#check veridux_managers_count

end NUMINAMATH_CALUDE_veridux_managers_count_l955_95507


namespace NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l955_95532

theorem smallest_quadratic_coefficient (a : ℕ) : a ≥ 5 ↔ 
  ∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    (0 < x₁ ∧ x₁ < 1) ∧ 
    (0 < x₂ ∧ x₂ < 1) ∧ 
    (x₁ ≠ x₂) ∧
    (∀ x, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) ∧
    a > 0 ∧
    (∀ a' < a, ¬∃ (b' c' : ℤ) (y₁ y₂ : ℝ), 
      (0 < y₁ ∧ y₁ < 1) ∧ 
      (0 < y₂ ∧ y₂ < 1) ∧ 
      (y₁ ≠ y₂) ∧
      (∀ x, a' * x^2 + b' * x + c' = a' * (x - y₁) * (x - y₂)) ∧
      a' > 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l955_95532


namespace NUMINAMATH_CALUDE_hyperbola_n_range_l955_95562

def is_hyperbola (m n : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

def foci_distance (m : ℝ) : ℝ := 4

theorem hyperbola_n_range (m n : ℝ) 
  (h1 : is_hyperbola m n) 
  (h2 : foci_distance m = 4) : 
  -1 < n ∧ n < 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_n_range_l955_95562


namespace NUMINAMATH_CALUDE_parallel_condition_perpendicular_condition_l955_95586

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := (2*a + 1)*x + (a + 2)*y + 3 = 0
def l2 (a x y : ℝ) : Prop := (a - 1)*x - 2*y + 2 = 0

-- Define parallel and perpendicular conditions
def parallel (a : ℝ) : Prop := ∀ x y, l1 a x y ↔ ∃ k, l2 a (x + k * (2*a + 1)) (y + k * (a + 2))

def perpendicular (a : ℝ) : Prop := ∀ x1 y1 x2 y2, 
  l1 a x1 y1 → l2 a x2 y2 → (x2 - x1) * (2*a + 1) + (y2 - y1) * (a + 2) = 0

-- State the theorems
theorem parallel_condition : ∀ a : ℝ, parallel a ↔ a = 0 := by sorry

theorem perpendicular_condition : ∀ a : ℝ, perpendicular a ↔ a = -1 ∨ a = 5/2 := by sorry

end NUMINAMATH_CALUDE_parallel_condition_perpendicular_condition_l955_95586


namespace NUMINAMATH_CALUDE_food_expense_percentage_l955_95509

/-- Represents the percentage of income spent on various expenses --/
structure IncomeDistribution where
  food : ℝ
  education : ℝ
  rent : ℝ
  remaining : ℝ

/-- Proves that the percentage of income spent on food is 50% --/
theorem food_expense_percentage (d : IncomeDistribution) : d.food = 50 :=
  by
  have h1 : d.education = 15 := sorry
  have h2 : d.rent = 50 * (100 - d.food - d.education) / 100 := sorry
  have h3 : d.remaining = 17.5 := sorry
  have h4 : d.food + d.education + d.rent + d.remaining = 100 := sorry
  sorry

#check food_expense_percentage

end NUMINAMATH_CALUDE_food_expense_percentage_l955_95509


namespace NUMINAMATH_CALUDE_kiley_ate_two_slices_l955_95585

/-- Represents a cheesecake with its properties and consumption -/
structure Cheesecake where
  calories_per_slice : ℕ
  total_calories : ℕ
  percent_eaten : ℚ

/-- Calculates the number of slices eaten given a Cheesecake -/
def slices_eaten (c : Cheesecake) : ℚ :=
  (c.total_calories / c.calories_per_slice : ℚ) * c.percent_eaten

/-- Theorem stating that Kiley ate 2 slices of the specified cheesecake -/
theorem kiley_ate_two_slices (c : Cheesecake) 
  (h1 : c.calories_per_slice = 350)
  (h2 : c.total_calories = 2800)
  (h3 : c.percent_eaten = 1/4) : 
  slices_eaten c = 2 := by
  sorry

end NUMINAMATH_CALUDE_kiley_ate_two_slices_l955_95585


namespace NUMINAMATH_CALUDE_march_walking_distance_l955_95510

theorem march_walking_distance (days_in_month : Nat) (miles_per_day : Nat) (skipped_days : Nat) : 
  days_in_month = 31 → miles_per_day = 4 → skipped_days = 4 → 
  (days_in_month - skipped_days) * miles_per_day = 108 := by
  sorry

end NUMINAMATH_CALUDE_march_walking_distance_l955_95510


namespace NUMINAMATH_CALUDE_max_distance_complex_numbers_l955_95590

theorem max_distance_complex_numbers (z : ℂ) (h : Complex.abs z = 3) :
  Complex.abs ((2 + 3*Complex.I) * z^2 - z^4) ≤ 9*Real.sqrt 13 + 81 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_complex_numbers_l955_95590


namespace NUMINAMATH_CALUDE_total_wrappers_collected_l955_95598

theorem total_wrappers_collected (andy_wrappers max_wrappers : ℕ) 
  (h1 : andy_wrappers = 34) 
  (h2 : max_wrappers = 15) : 
  andy_wrappers + max_wrappers = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_wrappers_collected_l955_95598


namespace NUMINAMATH_CALUDE_log_8_1000_equals_inverse_log_10_2_l955_95582

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define log_10 as the natural logarithm divided by ln(10)
noncomputable def log_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_8_1000_equals_inverse_log_10_2 :
  log 8 1000 = 1 / log_10 2 := by
  sorry

end NUMINAMATH_CALUDE_log_8_1000_equals_inverse_log_10_2_l955_95582


namespace NUMINAMATH_CALUDE_nina_widget_purchase_l955_95588

/-- The problem of determining how many widgets Nina can purchase --/
theorem nina_widget_purchase (total_money : ℚ) (reduced_price_quantity : ℕ) (price_reduction : ℚ) : 
  total_money = 27.6 →
  reduced_price_quantity = 8 →
  price_reduction = 1.15 →
  (reduced_price_quantity : ℚ) * ((total_money / (reduced_price_quantity : ℚ)) - price_reduction) = total_money →
  (total_money / (total_money / (reduced_price_quantity : ℚ))).floor = 6 :=
by sorry

end NUMINAMATH_CALUDE_nina_widget_purchase_l955_95588


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l955_95592

/-- Given two vectors a and b in R², if (a + b) is parallel to (m*a - b), then m = -1 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
    (h1 : a = (4, -1))
    (h2 : b = (-5, 2))
    (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (m • a - b)) : 
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l955_95592


namespace NUMINAMATH_CALUDE_sequence_properties_l955_95549

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

theorem sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a : 2 * a 5 - a 3 = 3)
  (h_b2 : b 2 = 1)
  (h_b4 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (∃ q : ℝ, (q = 2 ∨ q = -2) ∧ ∀ n : ℕ, b (n + 1) = b n * q) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l955_95549


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_990_l955_95576

theorem sum_of_largest_and_smallest_prime_factors_of_990 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 990 ∧ largest ∣ 990 ∧
    (∀ p : ℕ, p.Prime → p ∣ 990 → smallest ≤ p ∧ p ≤ largest) ∧
    smallest + largest = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_990_l955_95576


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l955_95530

/-- Given a quadratic function f(x) = 2x^2 - x + 7, when shifted 3 units right and 5 units up,
    the resulting function g(x) = ax^2 + bx + c satisfies a + b + c = 22 -/
theorem quadratic_shift_sum (a b c : ℝ) : 
  (∀ x, 2*(x-3)^2 - (x-3) + 7 + 5 = a*x^2 + b*x + c) → 
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l955_95530


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l955_95519

theorem imaginary_part_of_z (z : ℂ) : (1 + z) * (1 - Complex.I) = 2 → Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l955_95519


namespace NUMINAMATH_CALUDE_min_distance_exp_ln_l955_95581

/-- The minimum distance between a point on y = e^x and a point on y = ln(x) -/
theorem min_distance_exp_ln : ∀ (P Q : ℝ × ℝ),
  (∃ x : ℝ, P = (x, Real.exp x)) →
  (∃ y : ℝ, Q = (Real.exp y, y)) →
  ∃ d : ℝ, d = Real.sqrt 2 ∧ ∀ P' Q', 
    (∃ x' : ℝ, P' = (x', Real.exp x')) →
    (∃ y' : ℝ, Q' = (Real.exp y', y')) →
    d ≤ Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_exp_ln_l955_95581


namespace NUMINAMATH_CALUDE_no_intersection_l955_95525

-- Define the functions
def f (x : ℝ) : ℝ := |3 * x + 4|
def g (x : ℝ) : ℝ := -|4 * x - 1|

-- Theorem statement
theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_l955_95525


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l955_95551

/-- Given an ellipse and a parabola intersecting at two points with a specific distance between them, prove the value of the parabola parameter. -/
theorem ellipse_parabola_intersection (p : ℝ) (h_p_pos : p > 0) : 
  (∃ A B : ℝ × ℝ, 
    A.1^2 / 8 + A.2^2 / 2 = 1 ∧
    B.1^2 / 8 + B.2^2 / 2 = 1 ∧
    A.2^2 = 2 * p * A.1 ∧
    B.2^2 = 2 * p * B.1 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4) →
  p = 1/4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l955_95551


namespace NUMINAMATH_CALUDE_brendan_recharge_ratio_l955_95518

/-- Represents the financial data for Brendan's June earnings and expenses -/
structure FinancialData where
  totalEarnings : ℕ
  carCost : ℕ
  remainingMoney : ℕ

/-- Calculates the amount recharged on the debit card -/
def amountRecharged (data : FinancialData) : ℕ :=
  data.totalEarnings - data.carCost - data.remainingMoney

/-- Represents a ratio as a pair of natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of recharged amount to total earnings -/
def rechargeRatio (data : FinancialData) : Ratio :=
  let recharged := amountRecharged data
  let gcd := Nat.gcd recharged data.totalEarnings
  { numerator := recharged / gcd, denominator := data.totalEarnings / gcd }

/-- Theorem stating that Brendan's recharge ratio is 1:2 -/
theorem brendan_recharge_ratio :
  let data : FinancialData := { totalEarnings := 5000, carCost := 1500, remainingMoney := 1000 }
  let ratio := rechargeRatio data
  ratio.numerator = 1 ∧ ratio.denominator = 2 := by sorry


end NUMINAMATH_CALUDE_brendan_recharge_ratio_l955_95518


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l955_95550

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l955_95550


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l955_95566

theorem complex_magnitude_equality : ∃ t : ℝ, t > 0 ∧ Complex.abs (Complex.mk (-3) t) = 5 * Real.sqrt 5 :=
by
  use 2 * Real.sqrt 29
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l955_95566
