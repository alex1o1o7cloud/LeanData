import Mathlib

namespace NUMINAMATH_CALUDE_accurate_number_range_l1641_164190

/-- The approximate number obtained by rounding -/
def approximate_number : ℝ := 0.270

/-- A number rounds to the given approximate number if it's within 0.0005 of it -/
def rounds_to (x : ℝ) : Prop :=
  x ≥ approximate_number - 0.0005 ∧ x < approximate_number + 0.0005

/-- The theorem stating the range of the accurate number -/
theorem accurate_number_range (a : ℝ) (h : rounds_to a) :
  a ≥ 0.2695 ∧ a < 0.2705 := by
  sorry

end NUMINAMATH_CALUDE_accurate_number_range_l1641_164190


namespace NUMINAMATH_CALUDE_largest_k_for_positive_root_l1641_164133

/-- The equation in question -/
def equation (k : ℤ) (x : ℝ) : ℝ := 3*x*(2*k*x-5) - 2*x^2 + 8

/-- Predicate for the existence of a positive root -/
def has_positive_root (k : ℤ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ equation k x = 0

/-- The main theorem -/
theorem largest_k_for_positive_root :
  (∀ k : ℤ, k > 1 → ¬(has_positive_root k)) ∧
  has_positive_root 1 := by sorry

end NUMINAMATH_CALUDE_largest_k_for_positive_root_l1641_164133


namespace NUMINAMATH_CALUDE_equation_solution_l1641_164178

theorem equation_solution (x : ℝ) (h : x ≠ 1/3) :
  (6 * x + 1) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 1) ↔
  x = -1 + (2 * Real.sqrt 3) / 3 ∨ x = -1 - (2 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1641_164178


namespace NUMINAMATH_CALUDE_march_greatest_drop_l1641_164174

/-- Represents the months of the year --/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August

/-- Represents the price change for each month --/
def price_change : Month → ℝ
  | Month.January  => -0.75
  | Month.February => 1.50
  | Month.March    => -3.00
  | Month.April    => 2.50
  | Month.May      => -0.25
  | Month.June     => 0.80
  | Month.July     => -2.75
  | Month.August   => -1.20

/-- Determines if a given month has the greatest price drop --/
def has_greatest_drop (m : Month) : Prop :=
  ∀ other : Month, price_change m ≤ price_change other

/-- Theorem stating that March has the greatest price drop --/
theorem march_greatest_drop : has_greatest_drop Month.March :=
sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l1641_164174


namespace NUMINAMATH_CALUDE_double_burger_cost_l1641_164118

/-- Calculates the cost of a double burger given the total spent, total number of hamburgers,
    cost of a single burger, and number of double burgers. -/
def cost_of_double_burger (total_spent : ℚ) (total_burgers : ℕ) (single_burger_cost : ℚ) (double_burgers : ℕ) : ℚ :=
  let single_burgers := total_burgers - double_burgers
  let total_single_cost := single_burgers * single_burger_cost
  let total_double_cost := total_spent - total_single_cost
  total_double_cost / double_burgers

/-- Theorem stating that the cost of a double burger is $1.50 given the specific conditions. -/
theorem double_burger_cost :
  cost_of_double_burger 64.5 50 1 29 = 1.5 := by
  sorry

#eval cost_of_double_burger 64.5 50 1 29

end NUMINAMATH_CALUDE_double_burger_cost_l1641_164118


namespace NUMINAMATH_CALUDE_tangent_secant_theorem_l1641_164189

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Check if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop := sorry

/-- Check if a segment is tangent to a circle -/
def is_tangent (p q : Point) (c : Circle) : Prop := sorry

/-- Check if a segment is a secant of a circle -/
def is_secant (p q r : Point) (c : Circle) : Prop := sorry

theorem tangent_secant_theorem (C : Circle) (Q U M N : Point) :
  is_outside Q C →
  is_tangent Q U C →
  is_secant Q M N C →
  distance Q M < distance Q N →
  distance Q M = 4 →
  distance Q U = distance M N - distance Q M →
  distance Q N = 16 := by sorry

end NUMINAMATH_CALUDE_tangent_secant_theorem_l1641_164189


namespace NUMINAMATH_CALUDE_min_value_theorem_l1641_164136

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 2/b + 3/c = 2) :
  a + 2*b + 3*c ≥ 18 ∧ (a + 2*b + 3*c = 18 ↔ a = 3 ∧ b = 3 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1641_164136


namespace NUMINAMATH_CALUDE_card_count_theorem_l1641_164180

/-- Represents the number of baseball cards each person has -/
structure CardCounts where
  brandon : ℕ
  malcom : ℕ
  ella : ℕ
  lily : ℕ
  mark : ℕ

/-- Calculates the final card counts after all transactions -/
def finalCardCounts (initial : CardCounts) : CardCounts :=
  let malcomToMark := (initial.malcom * 3) / 5
  let ellaToLily := initial.ella / 4
  { brandon := initial.brandon
  , malcom := initial.malcom - malcomToMark
  , ella := initial.ella - ellaToLily
  , lily := initial.lily + ellaToLily
  , mark := malcomToMark + 6 }

/-- Theorem stating the correctness of the final card counts -/
theorem card_count_theorem (initial : CardCounts) :
  initial.brandon = 20 →
  initial.malcom = initial.brandon + 12 →
  initial.ella = initial.malcom - 5 →
  initial.lily = 2 * initial.ella →
  initial.mark = 0 →
  let final := finalCardCounts initial
  final.brandon = 20 ∧
  final.malcom = 13 ∧
  final.ella = 21 ∧
  final.lily = 60 ∧
  final.mark = 25 :=
by
  sorry


end NUMINAMATH_CALUDE_card_count_theorem_l1641_164180


namespace NUMINAMATH_CALUDE_michaels_bills_l1641_164143

def total_amount : ℕ := 280
def bill_denomination : ℕ := 20

theorem michaels_bills : 
  total_amount / bill_denomination = 14 :=
by sorry

end NUMINAMATH_CALUDE_michaels_bills_l1641_164143


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_four_l1641_164125

theorem no_linear_term_implies_m_equals_four :
  ∀ m : ℝ, (∀ x : ℝ, 2*x^2 + m*x = 4*x + 2) →
  (∀ x : ℝ, ∃ a b c : ℝ, a*x^2 + c = 0 ∧ 2*x^2 + m*x = 4*x + 2) →
  m = 4 :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_four_l1641_164125


namespace NUMINAMATH_CALUDE_gcd_105_210_l1641_164106

theorem gcd_105_210 : Nat.gcd 105 210 = 105 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_210_l1641_164106


namespace NUMINAMATH_CALUDE_cricket_innings_calculation_l1641_164112

/-- Given a cricket player's performance data, calculate the number of innings played. -/
theorem cricket_innings_calculation (current_average : ℚ) (next_innings_runs : ℚ) (average_increase : ℚ) : 
  current_average = 35 →
  next_innings_runs = 79 →
  average_increase = 4 →
  ∃ n : ℕ, n > 0 ∧ (n : ℚ) * current_average + next_innings_runs = ((n + 1) : ℚ) * (current_average + average_increase) ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_cricket_innings_calculation_l1641_164112


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l1641_164196

/-- A linear function y = 2x + 1 passing through points (-3, y₁) and (4, y₂) -/
def linear_function (x : ℝ) : ℝ := 2 * x + 1

/-- y₁ is the y-coordinate when x = -3 -/
def y₁ : ℝ := linear_function (-3)

/-- y₂ is the y-coordinate when x = 4 -/
def y₂ : ℝ := linear_function 4

/-- Theorem stating that y₁ < y₂ -/
theorem y₁_less_than_y₂ : y₁ < y₂ := by sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l1641_164196


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1641_164198

theorem quadratic_solution_difference_squared :
  ∀ p q : ℝ,
  (6 * p^2 - 7 * p - 20 = 0) →
  (6 * q^2 - 7 * q - 20 = 0) →
  p ≠ q →
  (p - q)^2 = 529 / 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1641_164198


namespace NUMINAMATH_CALUDE_expression_simplification_l1641_164111

theorem expression_simplification (α : ℝ) :
  4.59 * (Real.cos (2*α) - Real.cos (6*α) + Real.cos (10*α) - Real.cos (14*α)) /
  (Real.sin (2*α) + Real.sin (6*α) + Real.sin (10*α) + Real.sin (14*α)) =
  Real.tan (2*α) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1641_164111


namespace NUMINAMATH_CALUDE_well_diameter_proof_l1641_164116

/-- The volume of a circular well -/
def well_volume : ℝ := 175.92918860102841

/-- The depth of the circular well -/
def well_depth : ℝ := 14

/-- The diameter of the circular well -/
def well_diameter : ℝ := 4

theorem well_diameter_proof :
  well_diameter = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_well_diameter_proof_l1641_164116


namespace NUMINAMATH_CALUDE_heather_walking_distance_l1641_164161

theorem heather_walking_distance (car_to_entrance : ℝ) (entrance_to_rides : ℝ) (rides_to_car : ℝ) 
  (h1 : car_to_entrance = 0.33)
  (h2 : entrance_to_rides = 0.33)
  (h3 : rides_to_car = 0.08) :
  car_to_entrance + entrance_to_rides + rides_to_car = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_heather_walking_distance_l1641_164161


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l1641_164169

def number_of_friends : ℕ := 7

/-- The number of ways to choose 2 people from n friends to sit next to Cara in a circular arrangement -/
def circular_seating_arrangements (n : ℕ) : ℕ := Nat.choose n 2

theorem cara_seating_arrangements :
  circular_seating_arrangements number_of_friends = 21 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l1641_164169


namespace NUMINAMATH_CALUDE_min_notebooks_correct_l1641_164144

/-- The minimum number of notebooks needed to get a discount -/
def min_notebooks : ℕ := 18

/-- The cost of a single pen in yuan -/
def pen_cost : ℕ := 10

/-- The cost of a single notebook in yuan -/
def notebook_cost : ℕ := 4

/-- The number of pens Xiao Wei plans to buy -/
def num_pens : ℕ := 3

/-- The minimum spending amount to get a discount in yuan -/
def discount_threshold : ℕ := 100

/-- Theorem stating that min_notebooks is the minimum number of notebooks
    needed to get the discount -/
theorem min_notebooks_correct : 
  (num_pens * pen_cost + min_notebooks * notebook_cost ≥ discount_threshold) ∧ 
  (∀ n : ℕ, n < min_notebooks → num_pens * pen_cost + n * notebook_cost < discount_threshold) :=
sorry

end NUMINAMATH_CALUDE_min_notebooks_correct_l1641_164144


namespace NUMINAMATH_CALUDE_rabbits_per_cat_l1641_164168

theorem rabbits_per_cat (total_animals : ℕ) (num_cats : ℕ) (hares_per_rabbit : ℕ) :
  total_animals = 37 →
  num_cats = 4 →
  hares_per_rabbit = 3 →
  ∃ (rabbits_per_cat : ℕ),
    total_animals = 1 + num_cats + (num_cats * rabbits_per_cat) + (num_cats * rabbits_per_cat * hares_per_rabbit) ∧
    rabbits_per_cat = 2 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_per_cat_l1641_164168


namespace NUMINAMATH_CALUDE_calvin_mistake_l1641_164105

theorem calvin_mistake (a : ℝ) : 37 + 31 * a = 37 * 31 + a ↔ a = 37 :=
  sorry

end NUMINAMATH_CALUDE_calvin_mistake_l1641_164105


namespace NUMINAMATH_CALUDE_positive_f_one_l1641_164127

def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem positive_f_one (f : ℝ → ℝ) 
    (h_mono : MonoIncreasing f) (h_odd : OddFunction f) : 
    f 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_f_one_l1641_164127


namespace NUMINAMATH_CALUDE_inequality_proof_l1641_164131

theorem inequality_proof (a b c x y z : Real) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a^x = b*c) (eq2 : b^y = c*a) (eq3 : c^z = a*b) :
  (1 / (2 + x)) + (1 / (2 + y)) + (1 / (2 + z)) ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1641_164131


namespace NUMINAMATH_CALUDE_inequality_proof_l1641_164164

theorem inequality_proof (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h : u + v + w + Real.sqrt (u * v * w) = 4) :
  Real.sqrt (v * w / u) + Real.sqrt (u * w / v) + Real.sqrt (u * v / w) ≥ u + v + w := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1641_164164


namespace NUMINAMATH_CALUDE_quadratic_max_sum_roots_l1641_164170

theorem quadratic_max_sum_roots (m : ℝ) :
  let f := fun x : ℝ => 2 * x^2 - 5 * x + m
  let Δ := 25 - 8 * m  -- discriminant
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0) →  -- real roots exist
  (∀ k : ℝ, (∃ y₁ y₂ : ℝ, f y₁ = 0 ∧ f y₂ = 0) → y₁ + y₂ ≤ 5/2) ∧  -- 5/2 is max sum
  (m = 25/8 → ∃ z₁ z₂ : ℝ, f z₁ = 0 ∧ f z₂ = 0 ∧ z₁ + z₂ = 5/2)  -- max sum occurs at m = 25/8
  :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_sum_roots_l1641_164170


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l1641_164102

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l1641_164102


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l1641_164193

def total_shoes : ℕ := 10
def red_shoes : ℕ := 6
def green_shoes : ℕ := 4
def shoes_drawn : ℕ := 2

theorem probability_two_red_shoes :
  (Nat.choose red_shoes shoes_drawn) / (Nat.choose total_shoes shoes_drawn) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l1641_164193


namespace NUMINAMATH_CALUDE_water_transfer_problem_l1641_164158

/-- Represents a rectangular pool with given dimensions -/
structure Pool where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Represents a valve with a specific flow rate -/
structure Valve where
  flow_rate : ℝ

/-- The main theorem to prove -/
theorem water_transfer_problem 
  (pool_A pool_B : Pool)
  (valve_1 valve_2 : Valve)
  (h1 : pool_A.length = 3 ∧ pool_A.width = 2 ∧ pool_A.depth = 1.2)
  (h2 : pool_B.length = 3 ∧ pool_B.width = 2 ∧ pool_B.depth = 1.2)
  (h3 : valve_1.flow_rate * 18 = pool_A.length * pool_A.width * pool_A.depth)
  (h4 : valve_2.flow_rate * 24 = pool_A.length * pool_A.width * pool_A.depth)
  (h5 : 0.4 * pool_A.length * pool_A.width = (valve_1.flow_rate - valve_2.flow_rate) * t)
  (h6 : t > 0) :
  valve_2.flow_rate * t = 7.2 := by sorry


end NUMINAMATH_CALUDE_water_transfer_problem_l1641_164158


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1641_164129

/-- A curve is an ellipse if both coefficients are positive and not equal -/
def is_ellipse (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a ≠ b

/-- The condition that m is between 3 and 7 -/
def m_between_3_and_7 (m : ℝ) : Prop := 3 < m ∧ m < 7

/-- The curve equation in terms of m -/
def curve_equation (m : ℝ) : Prop := is_ellipse (7 - m) (m - 3)

theorem necessary_not_sufficient :
  (∀ m : ℝ, curve_equation m → m_between_3_and_7 m) ∧
  (∃ m : ℝ, m_between_3_and_7 m ∧ ¬curve_equation m) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1641_164129


namespace NUMINAMATH_CALUDE_max_xy_given_constraint_l1641_164159

theorem max_xy_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + 9 * y = 60) :
  x * y ≤ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀ + 9 * y₀ = 60 ∧ x₀ * y₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_given_constraint_l1641_164159


namespace NUMINAMATH_CALUDE_associate_professor_pencils_l1641_164120

theorem associate_professor_pencils 
  (total_people : ℕ) 
  (total_pencils : ℕ) 
  (total_charts : ℕ) 
  (associate_pencils : ℕ) 
  (h1 : total_people = 6) 
  (h2 : total_pencils = 10) 
  (h3 : total_charts = 8) :
  let associate_count := total_people - (total_charts - total_people) / 2
  associate_pencils = (total_pencils - (total_people - associate_count)) / associate_count →
  associate_pencils = 2 := by
sorry

end NUMINAMATH_CALUDE_associate_professor_pencils_l1641_164120


namespace NUMINAMATH_CALUDE_box_width_l1641_164165

/-- The width of a rectangular box with given dimensions and filling rate -/
theorem box_width (fill_rate : ℝ) (length depth : ℝ) (fill_time : ℝ) :
  fill_rate = 4 →
  length = 7 →
  depth = 2 →
  fill_time = 21 →
  (fill_rate * fill_time) / (length * depth) = 6 :=
by sorry

end NUMINAMATH_CALUDE_box_width_l1641_164165


namespace NUMINAMATH_CALUDE_track_length_l1641_164176

/-- Represents a circular track with two runners -/
structure CircularTrack where
  length : ℝ
  runner1_speed : ℝ
  runner2_speed : ℝ

/-- Theorem stating the length of the track given the conditions -/
theorem track_length (track : CircularTrack) 
  (h1 : track.runner1_speed > 0)
  (h2 : track.runner2_speed > 0)
  (h3 : track.length / 2 = 100)
  (h4 : 200 = track.runner2_speed * (track.length / (track.runner1_speed + track.runner2_speed)))
  : track.length = 200 := by
  sorry

#check track_length

end NUMINAMATH_CALUDE_track_length_l1641_164176


namespace NUMINAMATH_CALUDE_circle_radius_l1641_164103

/-- Given a circle centered at (0,k) with k > 4, which is tangent to the lines y=x, y=-x, and y=4,
    the radius of the circle is 4(1+√2). -/
theorem circle_radius (k : ℝ) (h1 : k > 4) : ∃ r : ℝ,
  (∀ x y : ℝ, (x = y ∨ x = -y ∨ y = 4) → (x^2 + (y - k)^2 = r^2)) ∧
  r = 4*(1 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1641_164103


namespace NUMINAMATH_CALUDE_modular_inverse_7_mod_120_l1641_164148

theorem modular_inverse_7_mod_120 :
  ∃ (x : ℕ), x < 120 ∧ (7 * x) % 120 = 1 ∧ x = 103 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_7_mod_120_l1641_164148


namespace NUMINAMATH_CALUDE_expression_two_values_l1641_164119

theorem expression_two_values (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (x y : ℝ), x ≠ y ∧
    ∀ (z : ℝ), z = a / abs a + b / abs b + (a * b) / abs (a * b) → z = x ∨ z = y :=
by sorry

end NUMINAMATH_CALUDE_expression_two_values_l1641_164119


namespace NUMINAMATH_CALUDE_sum_and_equal_numbers_l1641_164134

theorem sum_and_equal_numbers (a b c : ℚ) : 
  a + b + c = 150 →
  a + 10 = b - 3 ∧ b - 3 = 4 * c →
  b = 655 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_and_equal_numbers_l1641_164134


namespace NUMINAMATH_CALUDE_alex_lead_after_even_l1641_164182

/-- Represents the race between Alex and Max -/
structure Race where
  total_length : ℕ
  initial_even : ℕ
  max_ahead : ℕ
  alex_final_ahead : ℕ
  remaining : ℕ

/-- Calculates the distance Alex got ahead of Max after they were even -/
def alex_initial_lead (r : Race) : ℕ :=
  r.total_length - r.remaining - r.initial_even - (r.max_ahead + r.alex_final_ahead)

/-- The theorem stating that Alex got ahead of Max by 300 feet after they were even -/
theorem alex_lead_after_even (r : Race) 
  (h1 : r.total_length = 5000)
  (h2 : r.initial_even = 200)
  (h3 : r.max_ahead = 170)
  (h4 : r.alex_final_ahead = 440)
  (h5 : r.remaining = 3890) :
  alex_initial_lead r = 300 := by
  sorry

#eval alex_initial_lead { total_length := 5000, initial_even := 200, max_ahead := 170, alex_final_ahead := 440, remaining := 3890 }

end NUMINAMATH_CALUDE_alex_lead_after_even_l1641_164182


namespace NUMINAMATH_CALUDE_age_difference_daughter_daughterInLaw_l1641_164194

/-- Represents the ages of family members 5 years ago -/
structure FamilyAges5YearsAgo where
  member1 : ℕ
  member2 : ℕ
  member3 : ℕ
  daughter : ℕ

/-- Represents the current ages of family members -/
structure CurrentFamilyAges where
  member1 : ℕ
  member2 : ℕ
  member3 : ℕ
  daughterInLaw : ℕ

/-- The main theorem stating the difference in ages between daughter and daughter-in-law -/
theorem age_difference_daughter_daughterInLaw 
  (ages5YearsAgo : FamilyAges5YearsAgo)
  (currentAges : CurrentFamilyAges)
  (h1 : ages5YearsAgo.member1 + ages5YearsAgo.member2 + ages5YearsAgo.member3 + ages5YearsAgo.daughter = 114)
  (h2 : currentAges.member1 + currentAges.member2 + currentAges.member3 + currentAges.daughterInLaw = 85)
  (h3 : currentAges.member1 = ages5YearsAgo.member1 + 5)
  (h4 : currentAges.member2 = ages5YearsAgo.member2 + 5)
  (h5 : currentAges.member3 = ages5YearsAgo.member3 + 5) :
  ages5YearsAgo.daughter - currentAges.daughterInLaw = 29 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_daughter_daughterInLaw_l1641_164194


namespace NUMINAMATH_CALUDE_triangle_property_l1641_164126

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h : t.b + t.c = 2 * t.a * Real.sin (t.C + π/6)) : 
  t.A = π/3 ∧ 1 < (t.b^2 + t.c^2) / t.a^2 ∧ (t.b^2 + t.c^2) / t.a^2 ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_property_l1641_164126


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1641_164115

theorem absolute_value_equation (x : ℝ) : 
  |3990 * x + 1995| = 1995 → x = 0 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1641_164115


namespace NUMINAMATH_CALUDE_dorothy_doughnut_profit_l1641_164108

/-- Dorothy's doughnut business problem -/
theorem dorothy_doughnut_profit :
  let ingredient_cost : ℤ := 53
  let rent_utilities : ℤ := 27
  let num_doughnuts : ℕ := 25
  let price_per_doughnut : ℤ := 3
  let total_expenses : ℤ := ingredient_cost + rent_utilities
  let revenue : ℤ := num_doughnuts * price_per_doughnut
  let profit : ℤ := revenue - total_expenses
  profit = -5 := by
sorry


end NUMINAMATH_CALUDE_dorothy_doughnut_profit_l1641_164108


namespace NUMINAMATH_CALUDE_max_sum_under_constraint_l1641_164123

theorem max_sum_under_constraint (m n : ℤ) (h : 205 * m^2 + 409 * n^4 ≤ 20736) :
  m + n ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_sum_under_constraint_l1641_164123


namespace NUMINAMATH_CALUDE_sequence_limit_inequality_l1641_164138

theorem sequence_limit_inequality (a b : ℕ → ℝ) (A B : ℝ) :
  (∀ n : ℕ, a n > b n) →
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - A| < ε) →
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |b n - B| < ε) →
  A ≥ B := by
  sorry

end NUMINAMATH_CALUDE_sequence_limit_inequality_l1641_164138


namespace NUMINAMATH_CALUDE_triangle_problem_l1641_164156

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a = 4 * Real.sqrt 3 →
  b = 6 →
  Real.cos A = -1/3 →
  (c = 2 ∧ Real.cos (2 * B - π/4) = (4 - Real.sqrt 2) / 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1641_164156


namespace NUMINAMATH_CALUDE_tank_problem_l1641_164146

theorem tank_problem (tank1_capacity tank2_capacity tank3_capacity : ℚ)
  (tank1_fill_ratio tank2_fill_ratio : ℚ) (total_water : ℚ) :
  tank1_capacity = 7000 →
  tank2_capacity = 5000 →
  tank3_capacity = 3000 →
  tank1_fill_ratio = 3/4 →
  tank2_fill_ratio = 4/5 →
  total_water = 10850 →
  (total_water - (tank1_capacity * tank1_fill_ratio + tank2_capacity * tank2_fill_ratio)) / tank3_capacity = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_tank_problem_l1641_164146


namespace NUMINAMATH_CALUDE_remaining_ribbon_l1641_164107

/-- Calculates the remaining ribbon length after wrapping gifts -/
theorem remaining_ribbon (num_gifts : ℕ) (ribbon_per_gift : ℚ) (total_ribbon : ℚ) :
  num_gifts = 8 →
  ribbon_per_gift = 3/2 →
  total_ribbon = 15 →
  total_ribbon - (num_gifts : ℚ) * ribbon_per_gift = 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_ribbon_l1641_164107


namespace NUMINAMATH_CALUDE_equal_area_equal_intersection_l1641_164109

/-- A rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A placement of a rectangle in the plane -/
structure PlacedRectangle where
  rect : Rectangle
  center : Point
  angle : ℝ  -- Rotation angle in radians

/-- A horizontal line in the plane -/
structure HorizontalLine where
  y : ℝ

/-- The intersection of a horizontal line with a placed rectangle -/
def intersection (line : HorizontalLine) (pr : PlacedRectangle) : Option ℝ :=
  sorry

theorem equal_area_equal_intersection 
  (r1 r2 : Rectangle) 
  (h : r1.area = r2.area) :
  ∃ (pr1 pr2 : PlacedRectangle),
    pr1.rect = r1 ∧ pr2.rect = r2 ∧
    ∀ (line : HorizontalLine),
      (intersection line pr1).isSome ∨ (intersection line pr2).isSome →
      (intersection line pr1).isSome ∧ (intersection line pr2).isSome ∧
      (intersection line pr1 = intersection line pr2) :=
by sorry

end NUMINAMATH_CALUDE_equal_area_equal_intersection_l1641_164109


namespace NUMINAMATH_CALUDE_plane_equation_l1641_164166

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if two planes are parallel -/
def parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

/-- Check if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Check if coefficients are integers and their GCD is 1 -/
def validCoefficients (plane : Plane) : Prop :=
  Int.gcd (Int.natAbs (Int.floor plane.a)) (Int.gcd (Int.natAbs (Int.floor plane.b)) (Int.gcd (Int.natAbs (Int.floor plane.c)) (Int.natAbs (Int.floor plane.d)))) = 1

theorem plane_equation : ∃ (result : Plane),
  result.a = 2 ∧ result.b = -1 ∧ result.c = 3 ∧ result.d = -14 ∧
  pointOnPlane ⟨2, -1, 3⟩ result ∧
  parallel result ⟨4, -2, 6, -5⟩ ∧
  result.a > 0 ∧
  validCoefficients result :=
sorry

end NUMINAMATH_CALUDE_plane_equation_l1641_164166


namespace NUMINAMATH_CALUDE_special_triangle_properties_l1641_164139

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a = t.b * Real.cos t.C + (Real.sqrt 3 / 3) * t.c * Real.sin t.B ∧
  t.a = 2 ∧
  (1 / 2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.B = π / 3 ∧ t.b = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_properties_l1641_164139


namespace NUMINAMATH_CALUDE_baseball_cards_distribution_l1641_164149

theorem baseball_cards_distribution (n : ℕ) (h : n > 0) :
  ∃ (cards_per_friend : ℕ), 
    cards_per_friend * n = 12 ∧ 
    cards_per_friend = 12 / n :=
sorry

end NUMINAMATH_CALUDE_baseball_cards_distribution_l1641_164149


namespace NUMINAMATH_CALUDE_even_function_increasing_interval_l1641_164160

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The interval (-∞, 0] -/
def NegativeRealsAndZero : Set ℝ := { x | x ≤ 0 }

/-- A function f : ℝ → ℝ is increasing on a set S if f(x) ≤ f(y) for all x, y ∈ S with x ≤ y -/
def IncreasingOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

/-- The main theorem -/
theorem even_function_increasing_interval (a : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (a - 2) * x^2 + (a - 1) * x + 3
  IsEven f →
  IncreasingOn f NegativeRealsAndZero ∧
  ∀ S, IncreasingOn f S → S ⊆ NegativeRealsAndZero :=
sorry

end NUMINAMATH_CALUDE_even_function_increasing_interval_l1641_164160


namespace NUMINAMATH_CALUDE_inradius_properties_l1641_164122

/-- Properties of a triangle ABC --/
structure Triangle where
  /-- Inradius of the triangle --/
  r : ℝ
  /-- Circumradius of the triangle --/
  R : ℝ
  /-- Semiperimeter of the triangle --/
  s : ℝ
  /-- Angle A of the triangle --/
  A : ℝ
  /-- Angle B of the triangle --/
  B : ℝ
  /-- Angle C of the triangle --/
  C : ℝ
  /-- Exradius opposite to angle A --/
  r_a : ℝ
  /-- Exradius opposite to angle B --/
  r_b : ℝ
  /-- Exradius opposite to angle C --/
  r_c : ℝ

/-- Theorem: Properties of inradius in a triangle --/
theorem inradius_properties (t : Triangle) :
  (t.r = 4 * t.R * Real.sin (t.A / 2) * Real.sin (t.B / 2) * Real.sin (t.C / 2)) ∧
  (t.r = t.s * Real.tan (t.A / 2) * Real.tan (t.B / 2) * Real.tan (t.C / 2)) ∧
  (t.r = t.R * (Real.cos t.A + Real.cos t.B + Real.cos t.C - 1)) ∧
  (t.r = t.r_a + t.r_b + t.r_c - 4 * t.R) := by
  sorry

end NUMINAMATH_CALUDE_inradius_properties_l1641_164122


namespace NUMINAMATH_CALUDE_train_length_l1641_164172

/-- Given a train with speed 108 km/hr passing a tree in 9 seconds, its length is 270 meters. -/
theorem train_length (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 108 → time = 9 → length = speed * (1000 / 3600) * time → length = 270 := by sorry

end NUMINAMATH_CALUDE_train_length_l1641_164172


namespace NUMINAMATH_CALUDE_first_load_pieces_l1641_164124

theorem first_load_pieces (total : ℕ) (num_small_loads : ℕ) (pieces_per_small_load : ℕ) 
    (h1 : total = 47)
    (h2 : num_small_loads = 5)
    (h3 : pieces_per_small_load = 6) : 
  total - (num_small_loads * pieces_per_small_load) = 17 := by
  sorry

end NUMINAMATH_CALUDE_first_load_pieces_l1641_164124


namespace NUMINAMATH_CALUDE_tobias_apps_downloaded_l1641_164184

/-- The number of apps downloaded by Tobias -/
def m : ℕ := 24

/-- The base cost of each app in cents -/
def base_cost : ℕ := 200

/-- The tax rate as a percentage -/
def tax_rate : ℕ := 10

/-- The total amount spent in cents -/
def total_spent : ℕ := 5280

/-- Theorem stating that m is the correct number of apps downloaded -/
theorem tobias_apps_downloaded :
  m * (base_cost + base_cost * tax_rate / 100) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_tobias_apps_downloaded_l1641_164184


namespace NUMINAMATH_CALUDE_order_of_xyz_l1641_164192

theorem order_of_xyz (x y z : ℝ) 
  (h : x + 2013 / 2014 = y + 2012 / 2013 ∧ y + 2012 / 2013 = z + 2014 / 2015) : 
  z < y ∧ y < x := by
sorry

end NUMINAMATH_CALUDE_order_of_xyz_l1641_164192


namespace NUMINAMATH_CALUDE_expand_expression_l1641_164142

theorem expand_expression (x : ℝ) : (2*x - 3) * (4*x + 9) = 8*x^2 + 6*x - 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1641_164142


namespace NUMINAMATH_CALUDE_total_crayons_is_18_l1641_164117

/-- The number of crayons each child has -/
def crayons_per_child : ℕ := 3

/-- The number of children -/
def number_of_children : ℕ := 6

/-- The total number of crayons -/
def total_crayons : ℕ := crayons_per_child * number_of_children

theorem total_crayons_is_18 : total_crayons = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_is_18_l1641_164117


namespace NUMINAMATH_CALUDE_system_solution_relation_l1641_164185

theorem system_solution_relation (a₁ a₂ c₁ c₂ : ℝ) :
  (2 * a₁ + 3 = c₁ ∧ 2 * a₂ + 3 = c₂) →
  (∃! (x y : ℝ), a₁ * x + y = a₁ - c₁ ∧ a₂ * x + y = a₂ - c₂ ∧ x = -1 ∧ y = -3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_relation_l1641_164185


namespace NUMINAMATH_CALUDE_soybean_oil_production_l1641_164195

/-- Represents the conversion rates and prices for soybeans, tofu, and soybean oil -/
structure SoybeanProduction where
  soybean_to_tofu : ℝ        -- kg of tofu per kg of soybeans
  soybean_to_oil : ℝ         -- kg of soybeans needed for 1 kg of oil
  tofu_price : ℝ             -- yuan per kg of tofu
  oil_price : ℝ              -- yuan per kg of oil

/-- Represents the batch of soybeans and its processing -/
structure SoybeanBatch where
  total_soybeans : ℝ         -- total kg of soybeans in the batch
  tofu_soybeans : ℝ          -- kg of soybeans used for tofu
  oil_soybeans : ℝ           -- kg of soybeans used for oil
  total_revenue : ℝ          -- total revenue in yuan

/-- Theorem stating that given the conditions, 360 kg of soybeans were used for oil production -/
theorem soybean_oil_production (prod : SoybeanProduction) (batch : SoybeanBatch) :
  prod.soybean_to_tofu = 3 ∧
  prod.soybean_to_oil = 6 ∧
  prod.tofu_price = 3 ∧
  prod.oil_price = 15 ∧
  batch.total_soybeans = 460 ∧
  batch.total_revenue = 1800 ∧
  batch.tofu_soybeans + batch.oil_soybeans = batch.total_soybeans ∧
  batch.total_revenue = (batch.tofu_soybeans * prod.soybean_to_tofu * prod.tofu_price) +
                        (batch.oil_soybeans / prod.soybean_to_oil * prod.oil_price) →
  batch.oil_soybeans = 360 := by
  sorry

end NUMINAMATH_CALUDE_soybean_oil_production_l1641_164195


namespace NUMINAMATH_CALUDE_quadrilateral_dc_length_l1641_164188

theorem quadrilateral_dc_length
  (AB : ℝ) (sinA sinC : ℝ)
  (h1 : AB = 30)
  (h2 : sinA = 1/2)
  (h3 : sinC = 2/5)
  : ∃ (DC : ℝ), DC = 5 * Real.sqrt 47.25 :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_dc_length_l1641_164188


namespace NUMINAMATH_CALUDE_triangle_formation_l1641_164179

/-- Triangle Inequality Theorem: The sum of any two sides of a triangle must be greater than the third side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if a set of three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  ¬ can_form_triangle 3 4 8 ∧
  ¬ can_form_triangle 8 7 15 ∧
  ¬ can_form_triangle 5 5 11 ∧
  can_form_triangle 13 12 20 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l1641_164179


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1641_164100

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  x + y ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1641_164100


namespace NUMINAMATH_CALUDE_billy_sleep_theorem_l1641_164186

def night1_sleep : ℕ := 6

def night2_sleep (n1 : ℕ) : ℕ := n1 + 2

def night3_sleep (n2 : ℕ) : ℕ := n2 / 2

def night4_sleep (n3 : ℕ) : ℕ := n3 * 3

def total_sleep (n1 n2 n3 n4 : ℕ) : ℕ := n1 + n2 + n3 + n4

theorem billy_sleep_theorem :
  let n1 := night1_sleep
  let n2 := night2_sleep n1
  let n3 := night3_sleep n2
  let n4 := night4_sleep n3
  total_sleep n1 n2 n3 n4 = 30 := by sorry

end NUMINAMATH_CALUDE_billy_sleep_theorem_l1641_164186


namespace NUMINAMATH_CALUDE_complex_number_solutions_l1641_164147

theorem complex_number_solutions : 
  ∀ z : ℂ, z^2 = -45 - 28*I ∧ z^3 = 8 + 26*I →
  z = Complex.mk (Real.sqrt 10) (-Real.sqrt 140) ∨
  z = Complex.mk (-Real.sqrt 10) (Real.sqrt 140) := by
sorry

end NUMINAMATH_CALUDE_complex_number_solutions_l1641_164147


namespace NUMINAMATH_CALUDE_f_zero_eq_two_l1641_164113

/-- The function f(x) with parameter a -/
def f (a x : ℝ) : ℝ := (a^2 - 1) * (x^2 - 1) + (a - 1) * (x - 1)

/-- Theorem: If f represents a straight line, then f(0) = 2 -/
theorem f_zero_eq_two (a : ℝ) (h : ∀ x y : ℝ, f a x - f a y = (f a 1 - f a 0) * (x - y)) : 
  f a 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_eq_two_l1641_164113


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1641_164114

theorem equilateral_triangle_perimeter (R : ℝ) (chord_length : ℝ) (chord_distance : ℝ) :
  chord_length = 2 →
  chord_distance = 3 →
  R^2 = chord_distance^2 + (chord_length/2)^2 →
  ∃ (perimeter : ℝ), perimeter = 3 * R * Real.sqrt 3 ∧ perimeter = 3 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1641_164114


namespace NUMINAMATH_CALUDE_factorization_equalities_l1641_164128

theorem factorization_equalities (a b x y : ℝ) : 
  (3 * a * x^2 + 6 * a * x * y + 3 * a * y^2 = 3 * a * (x + y)^2) ∧
  (a^2 * (x - y) - b^2 * (x - y) = (x - y) * (a + b) * (a - b)) ∧
  (a^4 + 3 * a^2 - 4 = (a + 1) * (a - 1) * (a^2 + 4)) ∧
  (4 * x^2 - y^2 - 2 * y - 1 = (2 * x + y + 1) * (2 * x - y - 1)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_equalities_l1641_164128


namespace NUMINAMATH_CALUDE_fraction_unchanged_l1641_164163

theorem fraction_unchanged (x y : ℝ) : (x + y) / (x - 2*y) = ((-x) + (-y)) / ((-x) - 2*(-y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l1641_164163


namespace NUMINAMATH_CALUDE_special_three_digit_numbers_l1641_164181

-- Define a three-digit number
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Define the sum of digits for a three-digit number
def sum_of_digits (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

-- Define the condition for the special property
def has_special_property (n : ℕ) : Prop :=
  n = sum_of_digits n + 2 * (sum_of_digits n)^2

-- Theorem statement
theorem special_three_digit_numbers : 
  ∀ n : ℕ, is_three_digit n ∧ has_special_property n ↔ n = 171 ∨ n = 465 ∨ n = 666 := by
  sorry

end NUMINAMATH_CALUDE_special_three_digit_numbers_l1641_164181


namespace NUMINAMATH_CALUDE_new_average_is_44_l1641_164110

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  lastInningRuns : Nat
  averageIncrease : Nat

/-- Calculates the average score after a given number of innings -/
def calculateAverage (performance : BatsmanPerformance) : Nat :=
  (performance.totalRuns + performance.lastInningRuns) / (performance.innings + 1)

/-- Theorem: Given the specific performance, prove the new average is 44 -/
theorem new_average_is_44 (performance : BatsmanPerformance)
  (h1 : performance.innings = 16)
  (h2 : performance.lastInningRuns = 92)
  (h3 : performance.averageIncrease = 3)
  (h4 : calculateAverage performance = calculateAverage performance - performance.averageIncrease + 3) :
  calculateAverage performance = 44 := by
  sorry

end NUMINAMATH_CALUDE_new_average_is_44_l1641_164110


namespace NUMINAMATH_CALUDE_polygon_sides_l1641_164187

theorem polygon_sides (n : ℕ) (missing_angle : ℝ) : 
  (n ≥ 3) →
  (missing_angle < 170) →
  ((n - 2) * 180 - missing_angle = 2970) →
  n = 19 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l1641_164187


namespace NUMINAMATH_CALUDE_parallel_lines_circle_solution_l1641_164104

/-- A circle intersected by three equally spaced parallel lines -/
structure ParallelLinesCircle where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The lengths of the three chords formed by the intersection -/
  chord1 : ℝ
  chord2 : ℝ
  chord3 : ℝ
  /-- The chords are formed by equally spaced parallel lines -/
  parallel_lines : chord1 = chord3
  /-- The given chord lengths -/
  chord_lengths : chord1 = 40 ∧ chord2 = 36 ∧ chord3 = 40

/-- The theorem stating the distance between lines and radius of the circle -/
theorem parallel_lines_circle_solution (c : ParallelLinesCircle) :
  c.d = Real.sqrt 1188 ∧ c.r = Real.sqrt 357 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_circle_solution_l1641_164104


namespace NUMINAMATH_CALUDE_bulls_ploughing_problem_l1641_164101

/-- Represents the number of fields ploughed by a group of bulls -/
def fields_ploughed (bulls : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℚ :=
  (bulls * days * hours_per_day : ℚ) / 15

/-- The problem statement -/
theorem bulls_ploughing_problem :
  let group1_fields := fields_ploughed 10 3 10
  let group2_fields := fields_ploughed 30 2 8
  group2_fields = 32 →
  group1_fields = 20 := by
sorry


end NUMINAMATH_CALUDE_bulls_ploughing_problem_l1641_164101


namespace NUMINAMATH_CALUDE_sin_cos_225_degrees_l1641_164162

theorem sin_cos_225_degrees : 
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 ∧ 
  Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_225_degrees_l1641_164162


namespace NUMINAMATH_CALUDE_projection_matrix_values_l1641_164155

def projection_matrix (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 10/21; c, 35/63]

theorem projection_matrix_values :
  ∀ a c : ℚ, projection_matrix a c ^ 2 = projection_matrix a c → a = 2/9 ∧ c = 7/6 :=
by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l1641_164155


namespace NUMINAMATH_CALUDE_problem_1_l1641_164197

theorem problem_1 : -3 * (1/4) - (-1/9) + (-3/4) + 1 * (8/9) = -2 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1641_164197


namespace NUMINAMATH_CALUDE_hannah_final_pay_l1641_164171

def calculate_final_pay (hourly_rate : ℚ) (hours_worked : ℕ) (late_penalty : ℚ) 
  (times_late : ℕ) (federal_tax_rate : ℚ) (state_tax_rate : ℚ) (bonus_per_review : ℚ) 
  (qualifying_reviews : ℕ) (total_reviews : ℕ) : ℚ :=
  let gross_pay := hourly_rate * hours_worked
  let total_late_penalty := late_penalty * times_late
  let total_bonus := bonus_per_review * qualifying_reviews
  let adjusted_gross_pay := gross_pay - total_late_penalty + total_bonus
  let federal_tax := adjusted_gross_pay * federal_tax_rate
  let state_tax := adjusted_gross_pay * state_tax_rate
  let total_taxes := federal_tax + state_tax
  adjusted_gross_pay - total_taxes

theorem hannah_final_pay : 
  calculate_final_pay 30 18 5 3 (1/10) (1/20) 15 4 6 = 497.25 := by
  sorry

end NUMINAMATH_CALUDE_hannah_final_pay_l1641_164171


namespace NUMINAMATH_CALUDE_units_digit_not_four_l1641_164175

/-- The set of numbers from which a and b are chosen -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

/-- The units digit of a number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem units_digit_not_four (a b : ℕ) (ha : a ∈ S) (hb : b ∈ S) :
  unitsDigit (2^a + 5^b) ≠ 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_not_four_l1641_164175


namespace NUMINAMATH_CALUDE_jake_sister_weight_ratio_l1641_164130

theorem jake_sister_weight_ratio (J S : ℝ) (hJ : J > 0) (hS : S > 0) 
  (h1 : J + S = 132) (h2 : J - 15 = 2 * S) : (J - 15) / S = 2 := by
  sorry

end NUMINAMATH_CALUDE_jake_sister_weight_ratio_l1641_164130


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1641_164177

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt x + Real.sqrt (x + 6) = 12 → x = 529 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1641_164177


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1641_164199

/-- The parabola function --/
def f (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A point is on the parabola if its y-coordinate equals f(x) --/
def on_parabola (p : ℝ × ℝ) : Prop := p.2 = f p.1

/-- A point is on the x-axis if its y-coordinate is 0 --/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A square is defined by its center and side length --/
structure Square where
  center : ℝ × ℝ
  side : ℝ

/-- A square is inscribed if all its vertices are either on the parabola or on the x-axis --/
def inscribed (s : Square) : Prop :=
  let half_side := s.side / 2
  let left := s.center.1 - half_side
  let right := s.center.1 + half_side
  let top := s.center.2 + half_side
  let bottom := s.center.2 - half_side
  on_x_axis (left, bottom) ∧
  on_x_axis (right, bottom) ∧
  on_parabola (left, top) ∧
  on_parabola (right, top)

/-- The theorem to be proved --/
theorem inscribed_square_area :
  ∃ (s : Square), inscribed s ∧ s.side^2 = 24 - 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1641_164199


namespace NUMINAMATH_CALUDE_japanese_students_fraction_l1641_164191

theorem japanese_students_fraction (J : ℕ) : 
  let S := 2 * J
  let seniors_japanese := (3 * S) / 8
  let juniors_japanese := J / 4
  let total_students := S + J
  let total_japanese := seniors_japanese + juniors_japanese
  (total_japanese : ℚ) / total_students = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_japanese_students_fraction_l1641_164191


namespace NUMINAMATH_CALUDE_function_properties_l1641_164141

noncomputable def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b*x + 1

theorem function_properties (a b : ℝ) :
  (f a b 3 = -26) ∧ 
  (3*(3^2) - 2*a*3 + b = 0) →
  (a = 3 ∧ b = -9) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 4, f 3 (-9) x ≤ 6) ∧
  (∃ x ∈ Set.Icc (-4 : ℝ) 4, f 3 (-9) x = 6) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1641_164141


namespace NUMINAMATH_CALUDE_trig_difference_equals_sqrt_three_l1641_164153

-- Define the problem
theorem trig_difference_equals_sqrt_three :
  (1 / Real.tan (20 * π / 180)) - (1 / Real.cos (10 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_difference_equals_sqrt_three_l1641_164153


namespace NUMINAMATH_CALUDE_expression_evaluation_l1641_164137

theorem expression_evaluation : 4 * (8 - 3 + 2) - 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1641_164137


namespace NUMINAMATH_CALUDE_factorial_ratio_l1641_164152

theorem factorial_ratio (n : ℕ) (h : n > 0) : (n.factorial) / ((n-1).factorial) = n := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1641_164152


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1641_164167

theorem negation_of_proposition (p : ℝ → Prop) :
  (∀ n ∈ Set.Icc 1 2, n^2 < 3*n + 4) ↔ ¬(∃ n ∈ Set.Icc 1 2, n^2 ≥ 3*n + 4) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1641_164167


namespace NUMINAMATH_CALUDE_january_employee_count_l1641_164173

/-- The number of employees in January, given the December count and percentage increase --/
def january_employees (december_count : ℕ) (percent_increase : ℚ) : ℚ :=
  (december_count : ℚ) / (1 + percent_increase)

/-- Theorem stating that given the conditions, the number of employees in January is approximately 408.7 --/
theorem january_employee_count :
  let december_count : ℕ := 470
  let percent_increase : ℚ := 15 / 100
  let january_count := january_employees december_count percent_increase
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.05 ∧ |january_count - 408.7| < ε :=
sorry

end NUMINAMATH_CALUDE_january_employee_count_l1641_164173


namespace NUMINAMATH_CALUDE_beef_pounds_calculation_l1641_164150

theorem beef_pounds_calculation (total_cost : ℝ) (chicken_cost : ℝ) (oil_cost : ℝ) (beef_cost_per_pound : ℝ) :
  total_cost = 16 ∧ 
  chicken_cost = 3 ∧ 
  oil_cost = 1 ∧ 
  beef_cost_per_pound = 4 →
  (total_cost - chicken_cost - oil_cost) / beef_cost_per_pound = 3 := by
  sorry

end NUMINAMATH_CALUDE_beef_pounds_calculation_l1641_164150


namespace NUMINAMATH_CALUDE_parallel_postulate_l1641_164140

/-- A line in a plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane --/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Determines if a point is on a line --/
def Point.isOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Two lines are parallel if they have the same slope --/
def Line.isParallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- The statement of the parallel postulate --/
theorem parallel_postulate (L : Line) (P : Point) 
  (h : ¬ P.isOnLine L) : 
  ∃! (M : Line), M.isParallel L ∧ P.isOnLine M :=
sorry


end NUMINAMATH_CALUDE_parallel_postulate_l1641_164140


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l1641_164132

theorem complex_pure_imaginary (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ (b : ℝ), (a + 3 * Complex.I) / (1 + 2 * Complex.I) = b * Complex.I) →
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l1641_164132


namespace NUMINAMATH_CALUDE_circle_radius_is_five_l1641_164121

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y = 0

/-- The radius of the circle defined by circle_equation -/
def circle_radius : ℝ := 5

/-- Theorem stating that the radius of the circle defined by circle_equation is 5 -/
theorem circle_radius_is_five : 
  ∀ x y : ℝ, circle_equation x y → 
  ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_l1641_164121


namespace NUMINAMATH_CALUDE_twenty_five_percent_of_2004_l1641_164157

theorem twenty_five_percent_of_2004 : (25 : ℚ) / 100 * 2004 = 501 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_of_2004_l1641_164157


namespace NUMINAMATH_CALUDE_distribute_10_3_1_l1641_164154

/-- The number of ways to distribute n identical objects into k identical containers,
    with each container having at least m objects. -/
def distribute (n k m : ℕ) : ℕ := sorry

/-- The number of ways to distribute 10 identical coins into 3 identical bags,
    with each bag having at least 1 coin. -/
theorem distribute_10_3_1 : distribute 10 3 1 = 8 := sorry

end NUMINAMATH_CALUDE_distribute_10_3_1_l1641_164154


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1641_164151

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1641_164151


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l1641_164145

theorem ratio_x_to_y (x y : ℝ) (h : (8*x - 5*y) / (11*x - 3*y) = 4/7) : 
  x/y = 23/12 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l1641_164145


namespace NUMINAMATH_CALUDE_haircut_cost_per_year_l1641_164183

/-- Calculates the total amount spent on haircuts in a year given the specified conditions. -/
theorem haircut_cost_per_year
  (growth_rate : ℝ)
  (initial_length : ℝ)
  (cut_length : ℝ)
  (haircut_cost : ℝ)
  (tip_percentage : ℝ)
  (months_per_year : ℕ)
  (h1 : growth_rate = 1.5)
  (h2 : initial_length = 9)
  (h3 : cut_length = 6)
  (h4 : haircut_cost = 45)
  (h5 : tip_percentage = 0.2)
  (h6 : months_per_year = 12) :
  (haircut_cost * (1 + tip_percentage) * (months_per_year / ((initial_length - cut_length) / growth_rate))) = 324 :=
by sorry

end NUMINAMATH_CALUDE_haircut_cost_per_year_l1641_164183


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_180_l1641_164135

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_180 :
  rectangle_area 2025 10 = 180 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_180_l1641_164135
