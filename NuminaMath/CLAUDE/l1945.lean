import Mathlib

namespace NUMINAMATH_CALUDE_larger_number_proof_l1945_194573

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 8 * S + 15) :
  L = 1557 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1945_194573


namespace NUMINAMATH_CALUDE_units_digit_of_150_factorial_l1945_194599

theorem units_digit_of_150_factorial (n : ℕ) : n = 150 → (n.factorial % 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_150_factorial_l1945_194599


namespace NUMINAMATH_CALUDE_pool_cannot_be_filled_problem_pool_cannot_be_filled_l1945_194506

/-- Represents the state of a pool being filled -/
structure PoolFilling where
  capacity : ℝ
  num_hoses : ℕ
  flow_rate_per_hose : ℝ
  leakage_rate : ℝ

/-- Determines if a pool can be filled given its filling conditions -/
def can_be_filled (p : PoolFilling) : Prop :=
  p.num_hoses * p.flow_rate_per_hose > p.leakage_rate

/-- Theorem stating that a pool cannot be filled if inflow rate equals leakage rate -/
theorem pool_cannot_be_filled (p : PoolFilling) 
  (h : p.num_hoses * p.flow_rate_per_hose = p.leakage_rate) : 
  ¬(can_be_filled p) := by
  sorry

/-- The specific pool problem instance -/
def problem_pool : PoolFilling := {
  capacity := 48000
  num_hoses := 6
  flow_rate_per_hose := 3
  leakage_rate := 18
}

/-- Theorem for the specific problem instance -/
theorem problem_pool_cannot_be_filled : 
  ¬(can_be_filled problem_pool) := by
  sorry

end NUMINAMATH_CALUDE_pool_cannot_be_filled_problem_pool_cannot_be_filled_l1945_194506


namespace NUMINAMATH_CALUDE_intersection_point_l1945_194597

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := y = 2 * x - 5

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line y = 2x - 5 and the y-axis is (0, -5) -/
theorem intersection_point : 
  ∃ (x y : ℝ), line_equation x y ∧ on_y_axis x y ∧ x = 0 ∧ y = -5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1945_194597


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l1945_194562

theorem probability_of_black_ball (prob_red prob_white : ℝ) 
  (h_red : prob_red = 0.42)
  (h_white : prob_white = 0.28)
  (h_sum : prob_red + prob_white + (1 - prob_red - prob_white) = 1) :
  1 - prob_red - prob_white = 0.30 := by
sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l1945_194562


namespace NUMINAMATH_CALUDE_range_of_a_l1945_194542

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2*a - 6)^x > (2*a - 6)^y

def q (a : ℝ) : Prop := ∃ x y : ℝ, x > 3 ∧ y > 3 ∧ x ≠ y ∧
  x^2 - 3*a*x + 2*a^2 + 1 = 0 ∧ y^2 - 3*a*y + 2*a^2 + 1 = 0

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : a > 3) (h2 : a ≠ 7/2) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a > 7/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1945_194542


namespace NUMINAMATH_CALUDE_blueprint_to_actual_length_l1945_194502

/-- Represents the scale of a blueprint in feet per inch -/
def blueprint_scale : ℝ := 500

/-- Represents the length of a line segment on the blueprint in inches -/
def blueprint_length : ℝ := 6.5

/-- Represents the actual length in feet corresponding to the blueprint length -/
def actual_length : ℝ := blueprint_scale * blueprint_length

theorem blueprint_to_actual_length :
  actual_length = 3250 := by sorry

end NUMINAMATH_CALUDE_blueprint_to_actual_length_l1945_194502


namespace NUMINAMATH_CALUDE_shortest_side_of_triangle_l1945_194598

theorem shortest_side_of_triangle (A : ℝ) (P : ℝ) (D : ℝ) (a b c : ℝ) :
  A = 6 * Real.sqrt 6 →
  P = 18 →
  D = (2 * Real.sqrt 42) / 3 →
  a + b + c = P →
  A = Real.sqrt ((P/2) * ((P/2) - a) * ((P/2) - b) * ((P/2) - c)) →
  D^2 = ((P/2) - b) * ((P/2) - c) / ((P/2) - a) + (A / P)^2 →
  min a (min b c) = 5 := by
sorry

end NUMINAMATH_CALUDE_shortest_side_of_triangle_l1945_194598


namespace NUMINAMATH_CALUDE_post_office_mail_count_l1945_194559

/- Define the daily intake of letters and packages -/
def letters_per_day : ℕ := 60
def packages_per_day : ℕ := 20

/- Define the number of days in a month and the number of months -/
def days_per_month : ℕ := 30
def months : ℕ := 6

/- Define the total pieces of mail per day -/
def mail_per_day : ℕ := letters_per_day + packages_per_day

/- Theorem to prove -/
theorem post_office_mail_count :
  mail_per_day * days_per_month * months = 14400 :=
by sorry

end NUMINAMATH_CALUDE_post_office_mail_count_l1945_194559


namespace NUMINAMATH_CALUDE_total_balls_correct_l1945_194571

/-- The number of yellow balls in the bag -/
def yellow_balls : ℕ := 6

/-- The probability of drawing a yellow ball -/
def yellow_probability : ℚ := 3/10

/-- The total number of balls in the bag -/
def total_balls : ℕ := 20

/-- Theorem stating that the total number of balls is correct given the conditions -/
theorem total_balls_correct : 
  (yellow_balls : ℚ) / total_balls = yellow_probability :=
by sorry

end NUMINAMATH_CALUDE_total_balls_correct_l1945_194571


namespace NUMINAMATH_CALUDE_range_of_c_l1945_194524

def p (c : ℝ) : Prop := c^2 < c

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + 4*c*x + 1 < 0

theorem range_of_c (c : ℝ) (h1 : p c ∨ q c) (h2 : ¬(p c ∧ q c)) :
  c ∈ Set.Icc (1/2) 1 ∪ Set.Ioc (-1/2) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l1945_194524


namespace NUMINAMATH_CALUDE_a_plus_b_value_l1945_194572

theorem a_plus_b_value (a b : ℝ) (ha : |a| = 5) (hb : |b| = 2) (hab : a < b) :
  a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l1945_194572


namespace NUMINAMATH_CALUDE_hamburgers_needed_proof_l1945_194563

/-- Calculates the number of additional hamburgers needed to reach a target revenue -/
def additional_hamburgers_needed (target_revenue : ℕ) (price_per_hamburger : ℕ) (hamburgers_sold : ℕ) : ℕ :=
  ((target_revenue - (price_per_hamburger * hamburgers_sold)) + (price_per_hamburger - 1)) / price_per_hamburger

/-- Proves that 4 additional hamburgers are needed to reach $50 given the conditions -/
theorem hamburgers_needed_proof (target_revenue : ℕ) (price_per_hamburger : ℕ) (hamburgers_sold : ℕ)
  (h1 : target_revenue = 50)
  (h2 : price_per_hamburger = 5)
  (h3 : hamburgers_sold = 6) :
  additional_hamburgers_needed target_revenue price_per_hamburger hamburgers_sold = 4 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_needed_proof_l1945_194563


namespace NUMINAMATH_CALUDE_initial_people_count_l1945_194512

/-- The number of people who left the table -/
def people_left : ℕ := 6

/-- The number of people who remained at the table -/
def people_remained : ℕ := 5

/-- The initial number of people at the table -/
def initial_people : ℕ := people_left + people_remained

theorem initial_people_count : initial_people = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_people_count_l1945_194512


namespace NUMINAMATH_CALUDE_nth_equation_proof_l1945_194580

theorem nth_equation_proof (n : ℕ) (hn : n > 0) : 
  (1 : ℚ) / n * ((n^2 + 2*n) / (n + 1)) - 1 / (n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_proof_l1945_194580


namespace NUMINAMATH_CALUDE_total_mangoes_l1945_194500

theorem total_mangoes (alexis_mangoes : ℕ) (dilan_ashley_mangoes : ℕ) : 
  alexis_mangoes = 60 →
  alexis_mangoes = 4 * dilan_ashley_mangoes →
  alexis_mangoes + dilan_ashley_mangoes = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_total_mangoes_l1945_194500


namespace NUMINAMATH_CALUDE_xy_square_plus_2xy_plus_1_l1945_194539

theorem xy_square_plus_2xy_plus_1 (x y : ℝ) 
  (h : x^2 - 2*x + y^2 - 6*y + 10 = 0) : 
  x^2 * y^2 + 2*x*y + 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_xy_square_plus_2xy_plus_1_l1945_194539


namespace NUMINAMATH_CALUDE_football_game_attendance_l1945_194543

theorem football_game_attendance (saturday : ℕ) (expected_total : ℕ) : 
  saturday = 80 →
  expected_total = 350 →
  (saturday + (saturday - 20) + (saturday - 20 + 50) + (saturday + (saturday - 20))) - expected_total = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_football_game_attendance_l1945_194543


namespace NUMINAMATH_CALUDE_manolo_face_masks_l1945_194505

/-- Calculates the number of face-masks Manolo makes in a four-hour shift -/
def face_masks_in_shift (first_hour_rate : ℕ) (other_hours_rate : ℕ) (shift_duration : ℕ) : ℕ :=
  let first_hour_masks := 60 / first_hour_rate
  let other_hours_masks := (shift_duration - 1) * 60 / other_hours_rate
  first_hour_masks + other_hours_masks

/-- Theorem stating that Manolo makes 45 face-masks in a four-hour shift -/
theorem manolo_face_masks :
  face_masks_in_shift 4 6 4 = 45 :=
by sorry

end NUMINAMATH_CALUDE_manolo_face_masks_l1945_194505


namespace NUMINAMATH_CALUDE_max_catered_children_correct_l1945_194528

structure MealData where
  total_adults : ℕ
  total_children : ℕ
  prepared_veg_adults : ℕ
  prepared_nonveg_adults : ℕ
  prepared_vegan_adults : ℕ
  prepared_veg_children : ℕ
  prepared_nonveg_children : ℕ
  prepared_vegan_children : ℕ
  pref_veg_adults : ℕ
  pref_nonveg_adults : ℕ
  pref_vegan_adults : ℕ
  pref_veg_children : ℕ
  pref_nonveg_children : ℕ
  pref_vegan_children : ℕ
  eaten_veg_adults : ℕ
  eaten_nonveg_adults : ℕ
  eaten_vegan_adults : ℕ

def max_catered_children (data : MealData) : ℕ × ℕ × ℕ :=
  let remaining_veg := data.prepared_veg_adults + data.prepared_veg_children - data.eaten_veg_adults
  let remaining_nonveg := data.prepared_nonveg_adults + data.prepared_nonveg_children - data.eaten_nonveg_adults
  let remaining_vegan := data.prepared_vegan_adults + data.prepared_vegan_children - data.eaten_vegan_adults
  (min remaining_veg data.pref_veg_children,
   min remaining_nonveg data.pref_nonveg_children,
   min remaining_vegan data.pref_vegan_children)

theorem max_catered_children_correct (data : MealData) : 
  data.total_adults = 80 ∧
  data.total_children = 120 ∧
  data.prepared_veg_adults = 70 ∧
  data.prepared_nonveg_adults = 75 ∧
  data.prepared_vegan_adults = 5 ∧
  data.prepared_veg_children = 90 ∧
  data.prepared_nonveg_children = 25 ∧
  data.prepared_vegan_children = 5 ∧
  data.pref_veg_adults = 45 ∧
  data.pref_nonveg_adults = 30 ∧
  data.pref_vegan_adults = 5 ∧
  data.pref_veg_children = 100 ∧
  data.pref_nonveg_children = 15 ∧
  data.pref_vegan_children = 5 ∧
  data.eaten_veg_adults = 42 ∧
  data.eaten_nonveg_adults = 25 ∧
  data.eaten_vegan_adults = 5
  →
  max_catered_children data = (100, 15, 5) := by
sorry

end NUMINAMATH_CALUDE_max_catered_children_correct_l1945_194528


namespace NUMINAMATH_CALUDE_inequality_proof_l1945_194545

theorem inequality_proof (x b : ℝ) (h1 : x < b) (h2 : b < 0) (h3 : b = -2) :
  x^2 > b*x ∧ b*x > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1945_194545


namespace NUMINAMATH_CALUDE_unique_square_double_reverse_l1945_194530

theorem unique_square_double_reverse : ∃! x : ℕ,
  (10 ≤ x^2 ∧ x^2 < 100) ∧
  (10 ≤ 2*x ∧ 2*x < 100) ∧
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ x^2 = 10*a + b ∧ 2*x = 10*b + a) ∧
  x^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_double_reverse_l1945_194530


namespace NUMINAMATH_CALUDE_cool_function_periodic_l1945_194584

/-- A function is cool if there exist real numbers a and b such that
    f(x + a) is even and f(x + b) is odd. -/
def IsCool (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, (∀ x, f (x + a) = f (-x + a)) ∧ (∀ x, f (x + b) = -f (-x + b))

/-- Every cool function is periodic. -/
theorem cool_function_periodic (f : ℝ → ℝ) (h : IsCool f) :
    ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x :=
  sorry

end NUMINAMATH_CALUDE_cool_function_periodic_l1945_194584


namespace NUMINAMATH_CALUDE_tan_graph_property_l1945_194591

theorem tan_graph_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, x ≠ π / 4 → ∃ y, y = a * Real.tan (b * x)) →
  (3 = a * Real.tan (b * π / 8)) →
  ab = 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_graph_property_l1945_194591


namespace NUMINAMATH_CALUDE_next_common_day_l1945_194570

def dance_interval : ℕ := 6
def karate_interval : ℕ := 12
def library_interval : ℕ := 18

theorem next_common_day (dance_interval karate_interval library_interval : ℕ) :
  dance_interval = 6 → karate_interval = 12 → library_interval = 18 →
  Nat.lcm (Nat.lcm dance_interval karate_interval) library_interval = 36 :=
by sorry

end NUMINAMATH_CALUDE_next_common_day_l1945_194570


namespace NUMINAMATH_CALUDE_max_partner_share_l1945_194566

def profit : ℕ := 36000
def ratio : List ℕ := [2, 4, 3, 5, 6]

theorem max_partner_share :
  let total_parts := ratio.sum
  let part_value := profit / total_parts
  let shares := ratio.map (· * part_value)
  shares.maximum? = some 10800 := by sorry

end NUMINAMATH_CALUDE_max_partner_share_l1945_194566


namespace NUMINAMATH_CALUDE_problem_statement_l1945_194513

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (m : ℝ) : Set ℝ := {x | 1 - 2*m ≤ x ∧ x ≤ 2 + m}

theorem problem_statement :
  (∀ m : ℝ, (∀ x : ℝ, x ∈ M → x ∈ N m) ↔ m ≥ 3) ∧
  (∀ m : ℝ, (M ⊂ N m ∧ M ≠ N m) ↔ m ≤ 3/2) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1945_194513


namespace NUMINAMATH_CALUDE_triangle_area_l1945_194526

-- Define the lines
def line1 (x y : ℝ) : Prop := y - 2*x = 3
def line2 (x y : ℝ) : Prop := 2*y - x = 9

-- Define the triangle
def triangle := {(x, y) : ℝ × ℝ | x ≥ 0 ∧ line1 x y ∧ line2 x y}

-- State the theorem
theorem triangle_area : MeasureTheory.volume triangle = 3/4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1945_194526


namespace NUMINAMATH_CALUDE_system_equations_properties_l1945_194556

theorem system_equations_properties (a : ℝ) (x y : ℝ) 
  (h1 : x + 3*y = 4 - a) 
  (h2 : x - y = 3*a) 
  (h3 : -3 ≤ a ∧ a ≤ 1) :
  (a = -2 → x = -y) ∧ 
  (a = 1 → x + y = 3) ∧ 
  (x ≤ 1 → 1 ≤ y ∧ y ≤ 4) := by
sorry

end NUMINAMATH_CALUDE_system_equations_properties_l1945_194556


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_condition_l1945_194540

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x, f a x > 0

-- Define the proposition q
def q (x : ℝ) : Prop := x^2 - 2*x - 8 > 0

theorem quadratic_inequality_and_condition :
  (∃ a, a ∈ Set.Icc 0 4 ∧ p a) ∧
  (∀ x, q x → x > 5) ∧
  (∃ x, q x ∧ x ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_condition_l1945_194540


namespace NUMINAMATH_CALUDE_equation_solution_l1945_194510

theorem equation_solution (x : ℝ) : (6 : ℝ) / (x + 1) = (3 : ℝ) / 2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1945_194510


namespace NUMINAMATH_CALUDE_transistors_in_2010_l1945_194577

/-- Moore's law tripling factor -/
def tripling_factor : ℕ := 3

/-- Years between tripling events -/
def years_per_tripling : ℕ := 3

/-- Initial number of transistors in 1995 -/
def initial_transistors : ℕ := 500000

/-- Years between 1995 and 2010 -/
def years_elapsed : ℕ := 15

/-- Number of tripling events in the given time period -/
def num_triplings : ℕ := years_elapsed / years_per_tripling

/-- Calculates the number of transistors after a given number of tripling events -/
def transistors_after_triplings (initial : ℕ) (triplings : ℕ) : ℕ :=
  initial * tripling_factor ^ triplings

/-- Theorem: The number of transistors in 2010 is 121,500,000 -/
theorem transistors_in_2010 :
  transistors_after_triplings initial_transistors num_triplings = 121500000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_2010_l1945_194577


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1945_194509

/-- Represents the number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 4 distinguishable boxes is 68 -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 68 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1945_194509


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l1945_194537

def U : Set Int := {-3, -2, -1, 0, 1, 2, 3}
def A : Set Int := {-3, -2, 2, 3}
def B : Set Int := {-3, 0, 1, 2}

theorem complement_A_inter_B :
  (U \ A) ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l1945_194537


namespace NUMINAMATH_CALUDE_sum_of_digits_double_equal_sum_of_digits_half_equal_l1945_194575

def sum_of_digits (n : ℕ) : ℕ := sorry

def digit_permutation (a b : ℕ) : Prop := sorry

theorem sum_of_digits_double_equal 
  (a b : ℕ) (h : digit_permutation a b) : 
  sum_of_digits (2 * a) = sum_of_digits (2 * b) := by sorry

theorem sum_of_digits_half_equal 
  (a b : ℕ) (h1 : digit_permutation a b) (h2 : Even a) (h3 : Even b) : 
  sum_of_digits (a / 2) = sum_of_digits (b / 2) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_double_equal_sum_of_digits_half_equal_l1945_194575


namespace NUMINAMATH_CALUDE_product_of_sums_equals_3280_l1945_194552

theorem product_of_sums_equals_3280 :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_3280_l1945_194552


namespace NUMINAMATH_CALUDE_flight_cost_de_l1945_194576

/-- Represents a right-angled triangle DEF -/
structure RightTriangle where
  de : ℝ  -- Distance from D to E
  df : ℝ  -- Distance from D to F
  ef : ℝ  -- Distance from E to F

/-- Represents the cost structure for transportation -/
structure TransportCost where
  bus_rate : ℝ        -- Cost per km for bus
  plane_base_fee : ℝ  -- Base booking fee for plane
  plane_rate : ℝ      -- Cost per km for plane

/-- Calculates the cost of flying between two points -/
def flight_cost (dist : ℝ) (cost : TransportCost) : ℝ :=
  cost.plane_base_fee + cost.plane_rate * dist

theorem flight_cost_de (triangle : RightTriangle) (cost : TransportCost) :
  triangle.de = 3500 →
  triangle.df = 3750 →
  cost.bus_rate = 0.20 →
  cost.plane_base_fee = 120 →
  cost.plane_rate = 0.12 →
  flight_cost triangle.df cost = 570 := by
  sorry


end NUMINAMATH_CALUDE_flight_cost_de_l1945_194576


namespace NUMINAMATH_CALUDE_smallest_number_neg_three_in_set_neg_three_is_smallest_l1945_194504

def number_set : Set ℤ := {-2, 0, -3, 1}

theorem smallest_number (a : ℤ) (ha : a ∈ number_set) :
  -3 ≤ a :=
by sorry

theorem neg_three_in_set : -3 ∈ number_set :=
by sorry

theorem neg_three_is_smallest : ∀ x ∈ number_set, -3 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_neg_three_in_set_neg_three_is_smallest_l1945_194504


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1945_194588

theorem consecutive_odd_integers_sum (x : ℤ) : 
  x % 2 = 1 → -- x is odd
  (x + 4) % 2 = 1 → -- x+4 is odd
  x + (x + 4) = 138 → -- sum of first and third is 138
  x + (x + 2) + (x + 4) = 207 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1945_194588


namespace NUMINAMATH_CALUDE_apple_arrangements_l1945_194536

def word : String := "apple"

/-- The number of distinct letters in the word -/
def distinctLetters : Nat := 4

/-- The total number of letters in the word -/
def totalLetters : Nat := 5

/-- The frequency of the letter 'p' in the word -/
def frequencyP : Nat := 2

/-- The frequency of the letter 'a' in the word -/
def frequencyA : Nat := 1

/-- The frequency of the letter 'l' in the word -/
def frequencyL : Nat := 1

/-- The frequency of the letter 'e' in the word -/
def frequencyE : Nat := 1

/-- The number of distinct arrangements of the letters in the word -/
def distinctArrangements : Nat := 60

theorem apple_arrangements :
  distinctArrangements = Nat.factorial totalLetters / 
    (Nat.factorial frequencyP * Nat.factorial frequencyA * 
     Nat.factorial frequencyL * Nat.factorial frequencyE) := by
  sorry

end NUMINAMATH_CALUDE_apple_arrangements_l1945_194536


namespace NUMINAMATH_CALUDE_fair_coin_prob_heads_l1945_194583

-- Define a fair coin
def fair_coin : Type := Unit

-- Define the probability of landing heads for a fair coin
def prob_heads (c : fair_coin) : ℚ := 1 / 2

-- Define a sequence of coin tosses
def coin_tosses : ℕ → fair_coin
  | _ => ()

-- State the theorem
theorem fair_coin_prob_heads (n : ℕ) : 
  prob_heads (coin_tosses n) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_prob_heads_l1945_194583


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l1945_194525

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 8 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l1945_194525


namespace NUMINAMATH_CALUDE_high_school_students_l1945_194533

theorem high_school_students (total_students : ℕ) : 
  (total_students * 40 / 100 : ℕ) * 70 / 100 = 140 → 
  total_students = 500 := by
sorry

end NUMINAMATH_CALUDE_high_school_students_l1945_194533


namespace NUMINAMATH_CALUDE_parabola_through_origin_l1945_194531

/-- A parabola is defined by the equation y = ax^2 + bx + c, where a, b, and c are real numbers. -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point on a 2D plane is represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin is the point (0, 0) on a 2D plane. -/
def origin : Point := ⟨0, 0⟩

/-- A point lies on a parabola if its coordinates satisfy the parabola's equation. -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = para.a * p.x^2 + para.b * p.x + para.c

/-- Theorem: A parabola passes through the origin if and only if its c coefficient is zero. -/
theorem parabola_through_origin (para : Parabola) :
  lies_on origin para ↔ para.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_origin_l1945_194531


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1945_194574

theorem smallest_k_no_real_roots :
  ∃ k : ℤ, k = 3 ∧ 
  (∀ x : ℝ, (3*k - 2) * x^2 - 15*x + 13 ≠ 0) ∧
  (∀ k' : ℤ, k' < k → ∃ x : ℝ, (3*k' - 2) * x^2 - 15*x + 13 = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l1945_194574


namespace NUMINAMATH_CALUDE_arccos_negative_one_l1945_194548

theorem arccos_negative_one : Real.arccos (-1) = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_one_l1945_194548


namespace NUMINAMATH_CALUDE_basketball_game_theorem_l1945_194514

/-- Represents the scores of a team in a basketball game -/
structure TeamScores :=
  (q1 : ℝ)
  (q2 : ℝ)
  (q3 : ℝ)
  (q4 : ℝ)

/-- The game result -/
def GameResult := TeamScores → TeamScores → Prop

/-- Checks if the scores form a decreasing geometric sequence -/
def is_decreasing_geometric (s : TeamScores) : Prop :=
  ∃ r : ℝ, r > 1 ∧ s.q2 = s.q1 / r ∧ s.q3 = s.q2 / r ∧ s.q4 = s.q3 / r

/-- Checks if the scores form a decreasing arithmetic sequence -/
def is_decreasing_arithmetic (s : TeamScores) : Prop :=
  ∃ d : ℝ, d > 0 ∧ s.q2 = s.q1 - d ∧ s.q3 = s.q2 - d ∧ s.q4 = s.q3 - d

/-- Calculates the total score of a team -/
def total_score (s : TeamScores) : ℝ := s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the score in the second half -/
def second_half_score (s : TeamScores) : ℝ := s.q3 + s.q4

/-- The main theorem to prove -/
theorem basketball_game_theorem (falcons eagles : TeamScores) : 
  (is_decreasing_geometric falcons) →
  (is_decreasing_arithmetic eagles) →
  (falcons.q1 + falcons.q2 = eagles.q1 + eagles.q2) →
  (total_score eagles = total_score falcons + 2) →
  (total_score falcons ≤ 100 ∧ total_score eagles ≤ 100) →
  (second_half_score falcons + second_half_score eagles = 27) := by
  sorry


end NUMINAMATH_CALUDE_basketball_game_theorem_l1945_194514


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1945_194568

theorem min_distance_to_line (x y : ℝ) (h : x + y - 3 = 0) :
  ∃ (min : ℝ), min = Real.sqrt 2 ∧
  ∀ (x' y' : ℝ), x' + y' - 3 = 0 →
  min ≤ Real.sqrt ((x' - 2)^2 + (y' + 1)^2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1945_194568


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l1945_194511

/-- Given that M(3,8) is the midpoint of line segment AB and A(5,6) is one endpoint,
    prove that the product of the coordinates of point B is 10. -/
theorem midpoint_coordinate_product (A B M : ℝ × ℝ) : 
  A = (5, 6) → M = (3, 8) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 * B.2 = 10 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l1945_194511


namespace NUMINAMATH_CALUDE_walking_speed_solution_l1945_194590

/-- Represents the problem of finding A's walking speed -/
def walking_speed_problem (v : ℝ) : Prop :=
  let b_speed : ℝ := 20
  let time_diff : ℝ := 3
  let catch_up_distance : ℝ := 60
  let catch_up_time : ℝ := catch_up_distance / b_speed
  v * (time_diff + catch_up_time) = catch_up_distance ∧ v = 10

/-- Theorem stating that the solution to the walking speed problem is 10 kmph -/
theorem walking_speed_solution :
  ∃ v : ℝ, walking_speed_problem v :=
sorry

end NUMINAMATH_CALUDE_walking_speed_solution_l1945_194590


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l1945_194527

theorem sum_of_four_consecutive_integers (S : ℤ) :
  (∃ n : ℤ, S = n + (n + 1) + (n + 2) + (n + 3)) ↔ (S - 6) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_l1945_194527


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l1945_194549

theorem ratio_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l1945_194549


namespace NUMINAMATH_CALUDE_sams_morning_run_distance_l1945_194521

/-- Represents the distances traveled by Sam during different activities --/
structure SamDistances where
  morning_run : ℝ
  afternoon_walk : ℝ
  evening_bike : ℝ

/-- Theorem stating that given the conditions, Sam's morning run was 2 miles --/
theorem sams_morning_run_distance 
  (total_distance : ℝ) 
  (h1 : total_distance = 18) 
  (h2 : SamDistances → ℝ) 
  (h3 : ∀ d : SamDistances, h2 d = d.morning_run + d.afternoon_walk + d.evening_bike) 
  (h4 : ∀ d : SamDistances, d.afternoon_walk = 2 * d.morning_run) 
  (h5 : ∀ d : SamDistances, d.evening_bike = 12) :
  ∃ d : SamDistances, d.morning_run = 2 ∧ h2 d = total_distance := by
  sorry


end NUMINAMATH_CALUDE_sams_morning_run_distance_l1945_194521


namespace NUMINAMATH_CALUDE_max_guarding_value_l1945_194534

/-- Represents the four possible directions a guard can look --/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a position on the 8x8 board --/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a guard on the board --/
structure Guard :=
  (pos : Position)
  (dir : Direction)

/-- The type of a valid board configuration --/
def BoardConfiguration := Fin 8 → Fin 8 → Guard

/-- Checks if a guard at position (row, col) is guarded by another guard --/
def isGuardedBy (board : BoardConfiguration) (row col : Fin 8) (otherRow otherCol : Fin 8) : Prop :=
  sorry

/-- Counts the number of guards watching a specific position --/
def countGuardingGuards (board : BoardConfiguration) (row col : Fin 8) : Nat :=
  sorry

/-- Checks if all guards are guarded by at least k other guards --/
def allGuardsGuardedByAtLeastK (board : BoardConfiguration) (k : Nat) : Prop :=
  ∀ row col, countGuardingGuards board row col ≥ k

/-- The main theorem stating that 5 is the maximum value of k --/
theorem max_guarding_value :
  (∃ board : BoardConfiguration, allGuardsGuardedByAtLeastK board 5) ∧
  (¬∃ board : BoardConfiguration, allGuardsGuardedByAtLeastK board 6) :=
sorry

end NUMINAMATH_CALUDE_max_guarding_value_l1945_194534


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1945_194546

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1945_194546


namespace NUMINAMATH_CALUDE_probability_three_green_marbles_l1945_194507

/-- The probability of picking exactly k successes in n trials with probability p for each trial. -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The number of green marbles -/
def greenMarbles : ℕ := 8

/-- The number of purple marbles -/
def purpleMarbles : ℕ := 7

/-- The total number of marbles -/
def totalMarbles : ℕ := greenMarbles + purpleMarbles

/-- The number of trials -/
def numTrials : ℕ := 7

/-- The number of green marbles we want to pick -/
def targetGreen : ℕ := 3

/-- The probability of picking a green marble in one trial -/
def probGreen : ℚ := greenMarbles / totalMarbles

theorem probability_three_green_marbles :
  binomialProbability numTrials targetGreen probGreen = 34454336 / 136687500 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_green_marbles_l1945_194507


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l1945_194518

theorem magnitude_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.abs (2 * i / (1 + i)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l1945_194518


namespace NUMINAMATH_CALUDE_circle_m_equation_l1945_194581

/-- A circle M passing through two points with its center on a given line -/
structure CircleM where
  -- Circle M passes through these two points
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  -- The center of circle M lies on this line
  center_line : ℝ → ℝ → ℝ
  -- Conditions from the problem
  h1 : point1 = (0, 2)
  h2 : point2 = (0, 4)
  h3 : ∀ x y, center_line x y = 2*x - y - 1

/-- The equation of circle M -/
def circle_equation (c : CircleM) (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 5

/-- Theorem stating that the given conditions imply the circle equation -/
theorem circle_m_equation (c : CircleM) :
  ∀ x y, circle_equation c x y :=
sorry

end NUMINAMATH_CALUDE_circle_m_equation_l1945_194581


namespace NUMINAMATH_CALUDE_computer_sales_total_l1945_194557

theorem computer_sales_total (total : ℕ) : 
  (total / 2 : ℕ) + (total / 3 : ℕ) + 12 = total → total = 72 := by
  sorry

end NUMINAMATH_CALUDE_computer_sales_total_l1945_194557


namespace NUMINAMATH_CALUDE_second_point_y_coordinate_l1945_194558

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x = 2 * y + 5

-- Define the two points
def point1 (m n : ℝ) : ℝ × ℝ := (m, n)
def point2 (m n k : ℝ) : ℝ × ℝ := (m + 1, n + k)

-- Theorem statement
theorem second_point_y_coordinate 
  (m n : ℝ) 
  (h1 : line_equation m n) 
  (h2 : line_equation (m + 1) (n + 0.5)) : 
  (point2 m n 0.5).2 = n + 0.5 := by
  sorry

end NUMINAMATH_CALUDE_second_point_y_coordinate_l1945_194558


namespace NUMINAMATH_CALUDE_calculate_expression_l1945_194517

theorem calculate_expression : 500 * 997 * 0.0997 * (10^2) = 5 * 997^2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1945_194517


namespace NUMINAMATH_CALUDE_compound_composition_l1945_194554

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  hydrogen : ℕ
  chromium : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (comp : CompoundComposition) (h_weight o_weight cr_weight : ℚ) : ℚ :=
  comp.hydrogen * h_weight + comp.chromium * cr_weight + comp.oxygen * o_weight

/-- States that the compound has the given composition and molecular weight -/
theorem compound_composition (h_weight o_weight cr_weight : ℚ) :
  ∃ (comp : CompoundComposition),
    comp.chromium = 1 ∧
    comp.oxygen = 4 ∧
    molecularWeight comp h_weight o_weight cr_weight = 118 ∧
    h_weight = 1 ∧
    o_weight = 16 ∧
    cr_weight = 52 ∧
    comp.hydrogen = 2 := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l1945_194554


namespace NUMINAMATH_CALUDE_two_thousand_eight_times_two_thousand_six_l1945_194522

theorem two_thousand_eight_times_two_thousand_six (n : ℕ) :
  (2 * 2006 = 1) →
  (∀ n : ℕ, (2*n + 2) * 2006 = 3 * ((2*n) * 2006)) →
  2008 * 2006 = 3^1003 := by
sorry

end NUMINAMATH_CALUDE_two_thousand_eight_times_two_thousand_six_l1945_194522


namespace NUMINAMATH_CALUDE_cube_difference_l1945_194544

theorem cube_difference (x : ℝ) (h : x - 1/x = 3) : x^3 - 1/x^3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l1945_194544


namespace NUMINAMATH_CALUDE_vector_parallel_implies_k_equals_one_l1945_194555

-- Define the vectors
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 3)
def c (k : ℝ) : ℝ × ℝ := (k, 7)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

-- Theorem statement
theorem vector_parallel_implies_k_equals_one (k : ℝ) :
  parallel (a.1 + 2 * (c k).1, a.2 + 2 * (c k).2) b → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_k_equals_one_l1945_194555


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l1945_194567

/-- Given a parabola y = x^2 + 1, prove that the equation of the tangent line
    passing through the point (0,0) is either 2x - y = 0 or 2x + y = 0. -/
theorem parabola_tangent_line (x y : ℝ) :
  y = x^2 + 1 →
  (∃ (m : ℝ), y = m*x ∧ 0 = 0^2 + 1) →
  (y = 2*x ∨ y = -2*x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_l1945_194567


namespace NUMINAMATH_CALUDE_first_day_over_200_paperclips_l1945_194560

def paperclip_count (n : ℕ) : ℕ :=
  if n < 2 then 3 else 3 * 2^(n - 2)

theorem first_day_over_200_paperclips :
  ∀ n : ℕ, n < 9 → paperclip_count n ≤ 200 ∧
  paperclip_count 9 > 200 :=
sorry

end NUMINAMATH_CALUDE_first_day_over_200_paperclips_l1945_194560


namespace NUMINAMATH_CALUDE_runners_speed_ratio_l1945_194515

theorem runners_speed_ratio :
  ∀ (C : ℝ) (v_V v_P : ℝ),
  C > 0 → v_V > 0 → v_P > 0 →
  (∃ (t_1 : ℝ), t_1 > 0 ∧ v_V * t_1 + v_P * t_1 = C) →
  (∃ (t_2 : ℝ), t_2 > 0 ∧ v_V * t_2 = C + v_V * (C / (v_V + v_P)) ∧ v_P * t_2 = C + v_P * (C / (v_V + v_P))) →
  v_V / v_P = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_runners_speed_ratio_l1945_194515


namespace NUMINAMATH_CALUDE_mr_martin_bagels_l1945_194519

/-- Represents the purchase of coffee and bagels -/
structure Purchase where
  coffee : ℕ
  bagels : ℕ
  total : ℚ

/-- Represents the cost of items -/
structure Prices where
  coffee : ℚ
  bagel : ℚ

def mrs_martin : Purchase := { coffee := 3, bagels := 2, total := 12.75 }
def mr_martin (x : ℕ) : Purchase := { coffee := 2, bagels := x, total := 14 }

def prices : Prices := { coffee := 3.25, bagel := 1.5 }

theorem mr_martin_bagels :
  ∃ x : ℕ, 
    (mr_martin x).total = (mr_martin x).coffee • prices.coffee + (mr_martin x).bagels • prices.bagel ∧
    mrs_martin.total = mrs_martin.coffee • prices.coffee + mrs_martin.bagels • prices.bagel ∧
    x = 5 := by
  sorry

#check mr_martin_bagels

end NUMINAMATH_CALUDE_mr_martin_bagels_l1945_194519


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1945_194516

/-- If the quadratic equation mx² + 2x + 1 = 0 has two equal real roots, then m = 1 -/
theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2*x + 1 = 0 ∧ 
   (∀ y : ℝ, m * y^2 + 2*y + 1 = 0 → y = x)) → 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1945_194516


namespace NUMINAMATH_CALUDE_another_divisor_l1945_194569

theorem another_divisor (smallest_number : ℕ) : 
  smallest_number = 44402 →
  (smallest_number + 2) % 12 = 0 →
  (smallest_number + 2) % 30 = 0 →
  (smallest_number + 2) % 48 = 0 →
  (smallest_number + 2) % 74 = 0 →
  (smallest_number + 2) % 22202 = 0 := by
sorry

end NUMINAMATH_CALUDE_another_divisor_l1945_194569


namespace NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l1945_194538

theorem ten_thousandths_place_of_5_32 : ∃ (n : ℕ), (5 : ℚ) / 32 = (n * 10000 + 5) / 100000 :=
by sorry

end NUMINAMATH_CALUDE_ten_thousandths_place_of_5_32_l1945_194538


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_iff_a_in_range_l1945_194585

/-- Piecewise function f(x) defined by parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5*a - 4)*x + 7*a - 3 else (2*a - 1)^x

/-- The range of a for which f is monotonically decreasing -/
def a_range : Set ℝ := Set.Icc (3/5) (4/5)

/-- Theorem stating that f is monotonically decreasing iff a is in the specified range -/
theorem f_monotone_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) ↔ a ∈ a_range :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_iff_a_in_range_l1945_194585


namespace NUMINAMATH_CALUDE_fruit_purchase_change_l1945_194541

/-- The change received when purchasing fruit -/
def change (a : ℝ) : ℝ := 100 - 3 * a

/-- Theorem stating the change received when purchasing fruit -/
theorem fruit_purchase_change (a : ℝ) (h : a ≤ 30) :
  change a = 100 - 3 * a := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_change_l1945_194541


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l1945_194564

theorem six_digit_divisibility (abc : Nat) (h : abc ≥ 100 ∧ abc < 1000) :
  let abcabc := abc * 1000 + abc
  (abcabc % 11 = 0) ∧ (abcabc % 13 = 0) ∧ (abcabc % 1001 = 0) ∧
  ∃ x : Nat, x ≥ 100 ∧ x < 1000 ∧ (x * 1000 + x) % 101 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l1945_194564


namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_is_four_l1945_194532

/-- Represents a triangular pyramid with specific dimensions -/
structure TriangularPyramid where
  base_side_length : ℝ
  perpendicular_edge_length : ℝ

/-- Calculates the radius of the circumscribed sphere around a triangular pyramid -/
def circumscribed_sphere_radius (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating that the radius of the circumscribed sphere is 4 for the given pyramid -/
theorem circumscribed_sphere_radius_is_four :
  let pyramid : TriangularPyramid := { base_side_length := 6, perpendicular_edge_length := 4 }
  circumscribed_sphere_radius pyramid = 4 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_is_four_l1945_194532


namespace NUMINAMATH_CALUDE_trapezium_height_l1945_194553

theorem trapezium_height (a b h : ℝ) (area : ℝ) : 
  a = 20 → b = 18 → area = 209 → (1/2) * (a + b) * h = area → h = 11 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l1945_194553


namespace NUMINAMATH_CALUDE_import_tax_problem_l1945_194520

/-- The import tax rate as a decimal -/
def tax_rate : ℝ := 0.07

/-- The threshold above which the tax is applied -/
def tax_threshold : ℝ := 1000

/-- The amount of tax paid -/
def tax_paid : ℝ := 112.70

/-- The total value of the item -/
def total_value : ℝ := 2610

theorem import_tax_problem :
  tax_rate * (total_value - tax_threshold) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_import_tax_problem_l1945_194520


namespace NUMINAMATH_CALUDE_sum_digits_M_times_2013_l1945_194594

/-- A number composed of n consecutive ones -/
def consecutive_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

/-- Theorem: The sum of digits of M × 2013 is 1200, where M is composed of 200 consecutive ones -/
theorem sum_digits_M_times_2013 :
  sum_of_digits (consecutive_ones 200 * 2013) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_M_times_2013_l1945_194594


namespace NUMINAMATH_CALUDE_marks_team_three_pointers_l1945_194589

/-- Represents the number of 3-pointers scored by Mark's team -/
def marks_three_pointers : ℕ := sorry

/-- The total points scored by both teams -/
def total_points : ℕ := 201

/-- The number of 2-pointers scored by Mark's team -/
def marks_two_pointers : ℕ := 25

/-- The number of free throws scored by Mark's team -/
def marks_free_throws : ℕ := 10

theorem marks_team_three_pointers :
  marks_three_pointers = 8 ∧
  (2 * marks_two_pointers + 3 * marks_three_pointers + marks_free_throws) +
  (2 * (2 * marks_two_pointers) + 3 * (marks_three_pointers / 2) + (marks_free_throws / 2)) = total_points :=
sorry

end NUMINAMATH_CALUDE_marks_team_three_pointers_l1945_194589


namespace NUMINAMATH_CALUDE_ellipse_properties_l1945_194578

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define perpendicular rays from origin
def perpendicular_rays (m1 m2 n1 n2 : ℝ) : Prop :=
  m1 * n1 + m2 * n2 = 0

-- Define points M and N on the ellipse
def points_on_ellipse (m1 m2 n1 n2 : ℝ) : Prop :=
  ellipse m1 m2 ∧ ellipse n1 n2

-- Theorem statement
theorem ellipse_properties :
  ∀ (m1 m2 n1 n2 : ℝ),
  perpendicular_rays m1 m2 n1 n2 →
  points_on_ellipse m1 m2 n1 n2 →
  (∃ (e : ℝ), e = 1/2 ∧ e = Real.sqrt (1 - 3/4)) ∧
  (∃ (d : ℝ), d = 2 * Real.sqrt 21 / 7 ∧
    ∀ (k b : ℝ), (m2 = k * m1 + b ∧ n2 = k * n1 + b) →
      d = |b| / Real.sqrt (k^2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1945_194578


namespace NUMINAMATH_CALUDE_geometric_sequence_and_max_function_l1945_194503

/-- Given that real numbers a, b, c, and d form a geometric sequence, 
    and the function y = ln(x + 2) - x attains its maximum value of c when x = b, 
    prove that ad = -1 -/
theorem geometric_sequence_and_max_function (a b c d : ℝ) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) →
  (∀ x : ℝ, Real.log (x + 2) - x ≤ c) →
  (Real.log (b + 2) - b = c) →
  a * d = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_and_max_function_l1945_194503


namespace NUMINAMATH_CALUDE_infinitely_many_superabundant_numbers_l1945_194593

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Define superabundant numbers
def is_superabundant (m : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k < m → (sigma m : ℚ) / m > (sigma k : ℚ) / k

-- Define the set of superabundant numbers
def superabundant_set : Set ℕ :=
  {m : ℕ | is_superabundant m}

-- Theorem statement
theorem infinitely_many_superabundant_numbers :
  Set.Infinite superabundant_set := by sorry

end NUMINAMATH_CALUDE_infinitely_many_superabundant_numbers_l1945_194593


namespace NUMINAMATH_CALUDE_line_symmetry_l1945_194596

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The property of two lines being symmetrical about the x-axis -/
def symmetrical_about_x_axis (l1 l2 : Line) : Prop :=
  l1.slope = -l2.slope ∧ l1.intercept = -l2.intercept

/-- The given line y = 2x + 1 -/
def given_line : Line :=
  { slope := 2, intercept := 1 }

/-- The proposed symmetrical line y = -2x - 1 -/
def symmetrical_line : Line :=
  { slope := -2, intercept := -1 }

theorem line_symmetry :
  symmetrical_about_x_axis given_line symmetrical_line :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l1945_194596


namespace NUMINAMATH_CALUDE_jessa_cupcakes_l1945_194535

/-- The number of cupcakes Jessa needs to make -/
def total_cupcakes : ℕ := sorry

/-- The number of fourth-grade classes -/
def fourth_grade_classes : ℕ := 12

/-- The number of students in each fourth-grade class -/
def students_per_fourth_grade : ℕ := 45

/-- The number of P.E. classes -/
def pe_classes : ℕ := 2

/-- The number of students in each P.E. class -/
def students_per_pe : ℕ := 90

/-- The number of afterschool clubs -/
def afterschool_clubs : ℕ := 4

/-- The number of students in each afterschool club -/
def students_per_afterschool : ℕ := 60

/-- Theorem stating that the total number of cupcakes Jessa needs to make is 960 -/
theorem jessa_cupcakes : total_cupcakes = 960 := by sorry

end NUMINAMATH_CALUDE_jessa_cupcakes_l1945_194535


namespace NUMINAMATH_CALUDE_toothpick_100th_stage_l1945_194550

/-- Arithmetic sequence with first term 4 and common difference 4 -/
def toothpick_sequence (n : ℕ) : ℕ := 4 + (n - 1) * 4

/-- The 100th term of the toothpick sequence is 400 -/
theorem toothpick_100th_stage : toothpick_sequence 100 = 400 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_100th_stage_l1945_194550


namespace NUMINAMATH_CALUDE_number_calculation_l1945_194565

theorem number_calculation (x : ℝ) : 0.2 * x = 0.4 * 140 + 80 → x = 680 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l1945_194565


namespace NUMINAMATH_CALUDE_cheryl_expense_difference_l1945_194587

def electricity_bill : ℝ := 800
def golf_tournament_payment : ℝ := 1440

def monthly_cell_phone_expenses (x : ℝ) : ℝ := electricity_bill + x

def golf_tournament_cost (x : ℝ) : ℝ := 1.2 * monthly_cell_phone_expenses x

theorem cheryl_expense_difference :
  ∃ x : ℝ, 
    x = 400 ∧ 
    golf_tournament_cost x = golf_tournament_payment :=
sorry

end NUMINAMATH_CALUDE_cheryl_expense_difference_l1945_194587


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1945_194579

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 1 → x - 1/x > 0) ∧
  (∃ x : ℝ, x - 1/x > 0 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1945_194579


namespace NUMINAMATH_CALUDE_gcd_1908_4187_l1945_194508

theorem gcd_1908_4187 : Nat.gcd 1908 4187 = 53 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1908_4187_l1945_194508


namespace NUMINAMATH_CALUDE_range_of_m_l1945_194551

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x - 16

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-25) (-16)) ∧
  (∀ y ∈ Set.Icc (-25) (-16), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 3 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1945_194551


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_squared_equation_solutions_l1945_194582

-- Problem 1
theorem quadratic_equation_roots (x : ℝ) : 
  x^2 - 7*x + 6 = 0 ↔ x = 1 ∨ x = 6 := by sorry

-- Problem 2
theorem squared_equation_solutions (x : ℝ) :
  (2*x + 3)^2 = (x - 3)^2 ↔ x = 0 ∨ x = -6 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_squared_equation_solutions_l1945_194582


namespace NUMINAMATH_CALUDE_kite_area_16_20_l1945_194586

/-- Calculates the area of a kite given its base and height -/
def kite_area (base : ℝ) (height : ℝ) : ℝ :=
  base * height

/-- Theorem: The area of a kite with base 16 inches and height 20 inches is 160 square inches -/
theorem kite_area_16_20 :
  kite_area 16 20 = 160 := by
sorry

end NUMINAMATH_CALUDE_kite_area_16_20_l1945_194586


namespace NUMINAMATH_CALUDE_ratio_of_y_coordinates_l1945_194529

-- Define the ellipse
def Γ (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the point P
def P : ℝ × ℝ := (1, 0)

-- Define the lines l₁ and l₂
def l₁ (x : ℝ) : Prop := x = -2
def l₂ (x : ℝ) : Prop := x = 2

-- Define the line l_CD
def l_CD (x : ℝ) : Prop := x = 1

-- Define the chords AB and CD (implicitly by their properties)
def chord_passes_through_P (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = (1 - t) • P + t • B

-- Define points E and F
def E : ℝ × ℝ := (2, sorry)  -- y-coordinate to be determined
def F : ℝ × ℝ := (-2, sorry) -- y-coordinate to be determined

theorem ratio_of_y_coordinates :
  ∃ (A B C D : ℝ × ℝ),
    Γ A.1 A.2 ∧ Γ B.1 B.2 ∧ Γ C.1 C.2 ∧ Γ D.1 D.2 ∧
    chord_passes_through_P A B ∧ chord_passes_through_P C D ∧
    l_CD C.1 ∧ l_CD D.1 ∧
    (E.2 : ℝ) / (F.2 : ℝ) = -1/3 :=
sorry

end NUMINAMATH_CALUDE_ratio_of_y_coordinates_l1945_194529


namespace NUMINAMATH_CALUDE_euclidean_algorithm_steps_bound_l1945_194547

/-- The number of steps in the Euclidean algorithm for (a, b) -/
def euclidean_steps (a b : ℕ) : ℕ := sorry

/-- The number of digits in the decimal representation of a natural number -/
def decimal_digits (n : ℕ) : ℕ := sorry

theorem euclidean_algorithm_steps_bound (a b : ℕ) (h : a > b) :
  euclidean_steps a b ≤ 5 * decimal_digits b := by sorry

end NUMINAMATH_CALUDE_euclidean_algorithm_steps_bound_l1945_194547


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l1945_194561

theorem divisibility_by_eleven (m : ℕ+) (k : ℕ) (h : 33 ∣ m ^ k) : 11 ∣ m := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l1945_194561


namespace NUMINAMATH_CALUDE_least_candies_to_remove_l1945_194523

theorem least_candies_to_remove (total : Nat) (sisters : Nat) (to_remove : Nat) : 
  total = 24 → 
  sisters = 5 → 
  (total - to_remove) % sisters = 0 → 
  ∀ x : Nat, x < to_remove → (total - x) % sisters ≠ 0 →
  to_remove = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_candies_to_remove_l1945_194523


namespace NUMINAMATH_CALUDE_two_digit_square_sum_equals_concatenation_l1945_194595

theorem two_digit_square_sum_equals_concatenation : 
  {(x, y) : ℕ × ℕ | 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 ∧ (x + y)^2 = 100 * x + y} = 
  {(20, 25), (30, 25)} := by
sorry

end NUMINAMATH_CALUDE_two_digit_square_sum_equals_concatenation_l1945_194595


namespace NUMINAMATH_CALUDE_quadratic_roots_l1945_194501

theorem quadratic_roots : ∀ x : ℝ, x^2 - 49 = 0 ↔ x = 7 ∨ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1945_194501


namespace NUMINAMATH_CALUDE_petyas_chips_l1945_194592

theorem petyas_chips (x : ℕ) (y : ℕ) : 
  y = x - 2 → -- The side of the square has 2 fewer chips than the triangle
  3 * x - 3 = 4 * y - 4 → -- Total chips are the same for both shapes
  3 * x - 3 = 24 -- The total number of chips is 24
  := by sorry

end NUMINAMATH_CALUDE_petyas_chips_l1945_194592
