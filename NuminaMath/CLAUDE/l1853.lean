import Mathlib

namespace NUMINAMATH_CALUDE_log_8_y_value_l1853_185380

theorem log_8_y_value (y : ℝ) (h : Real.log y / Real.log 8 = 3.25) : y = 32 * Real.sqrt (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_log_8_y_value_l1853_185380


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l1853_185301

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) :
  a > 1 ∨ b > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l1853_185301


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1853_185383

/-- Given a geometric sequence with sum of first m terms Sm, prove S3m = 70 -/
theorem geometric_sequence_sum (m : ℕ) (Sm S2m S3m : ℝ) : 
  Sm = 10 → 
  S2m = 30 → 
  (∃ r : ℝ, r ≠ 0 ∧ S2m - Sm = r * Sm ∧ S3m - S2m = r * (S2m - Sm)) →
  S3m = 70 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1853_185383


namespace NUMINAMATH_CALUDE_students_taking_both_french_and_german_l1853_185329

theorem students_taking_both_french_and_german 
  (total : ℕ) 
  (french : ℕ) 
  (german : ℕ) 
  (neither : ℕ) 
  (h1 : total = 78) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : neither = 24) :
  french + german - (total - neither) = 9 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_both_french_and_german_l1853_185329


namespace NUMINAMATH_CALUDE_sum_of_fractions_minus_ten_equals_zero_l1853_185344

theorem sum_of_fractions_minus_ten_equals_zero :
  5 / 3 + 10 / 6 + 20 / 12 + 40 / 24 + 80 / 48 + 160 / 96 - 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_minus_ten_equals_zero_l1853_185344


namespace NUMINAMATH_CALUDE_extremum_derivative_zero_sufficient_not_necessary_l1853_185395

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the property of having an extremum at a point
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem extremum_derivative_zero_sufficient_not_necessary (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x₀ : ℝ, (deriv f) x₀ = 0 → HasExtremumAt f x₀) ∧
  ¬(∀ x₀ : ℝ, HasExtremumAt f x₀ → (deriv f) x₀ = 0) :=
sorry

end NUMINAMATH_CALUDE_extremum_derivative_zero_sufficient_not_necessary_l1853_185395


namespace NUMINAMATH_CALUDE_det_A_equals_six_l1853_185336

theorem det_A_equals_six (a d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; -3, d]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2*a, 1; -1, d]
  A + B⁻¹ = 0 → Matrix.det A = 6 := by sorry

end NUMINAMATH_CALUDE_det_A_equals_six_l1853_185336


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1853_185345

theorem quadratic_roots_range (m : ℝ) : 
  m > 0 → 
  (∃ x y : ℝ, x ≠ y ∧ x < 1 ∧ y < 1 ∧ 
    m * x^2 + (2*m - 1) * x - m + 2 = 0 ∧
    m * y^2 + (2*m - 1) * y - m + 2 = 0) →
  m > (3 + Real.sqrt 7) / 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1853_185345


namespace NUMINAMATH_CALUDE_circle_transformation_l1853_185392

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given amount -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem -/
theorem circle_transformation :
  let T : ℝ × ℝ := (-2, 6)
  let reflected := reflect_x T
  let final := translate_right reflected 5
  final = (3, -6) := by sorry

end NUMINAMATH_CALUDE_circle_transformation_l1853_185392


namespace NUMINAMATH_CALUDE_min_distance_M_to_F₂_l1853_185334

-- Define the rectangle and its properties
def Rectangle (a b : ℝ) := a > b ∧ a > 0 ∧ b > 0

-- Define the points on the sides of the rectangle
def Points (n : ℕ) (a b : ℝ) := n ≥ 5

-- Define the ellipse F₁
def F₁ (x y a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola F₂
def F₂ (x y a b : ℝ) := x^2 / a^2 - y^2 / b^2 = 1

-- Define the point M on F₁
def M (b : ℝ) := (0, b)

-- Theorem statement
theorem min_distance_M_to_F₂ (n : ℕ) (a b : ℝ) :
  Rectangle a b →
  Points n a b →
  ∀ (x y : ℝ), F₂ x y a b →
  Real.sqrt ((x - 0)^2 + (y - b)^2) ≥ a * Real.sqrt ((a^2 + 2*b^2) / (a^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_min_distance_M_to_F₂_l1853_185334


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1853_185381

theorem absolute_value_inequality (y : ℝ) : 
  |((7 - y) / 4)| ≤ 3 ↔ -5 ≤ y ∧ y ≤ 19 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1853_185381


namespace NUMINAMATH_CALUDE_sin_cube_identity_l1853_185313

theorem sin_cube_identity (θ : Real) : 
  Real.sin θ ^ 3 = -1/4 * Real.sin (3 * θ) + 3/4 * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l1853_185313


namespace NUMINAMATH_CALUDE_triangle_theorem_l1853_185396

/-- Given a triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: In a triangle ABC where c sin B = √3 b cos C and a + b = 6,
    the angle C is π/3 and the minimum value of c is 3 -/
theorem triangle_theorem (t : Triangle) 
    (h1 : t.c * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.C) 
    (h2 : t.a + t.b = 6) : 
    t.C = π / 3 ∧ t.c ≥ 3 ∧ ∃ (t' : Triangle), t'.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1853_185396


namespace NUMINAMATH_CALUDE_twenty_second_visits_l1853_185378

/-- Represents the tanning salon scenario --/
structure TanningSalon where
  total_customers : ℕ
  first_visit_charge : ℕ
  subsequent_visit_charge : ℕ
  third_visit_customers : ℕ
  total_revenue : ℕ

/-- Calculates the number of customers who made a second visit --/
def second_visit_customers (ts : TanningSalon) : ℕ :=
  (ts.total_revenue - ts.total_customers * ts.first_visit_charge - ts.third_visit_customers * ts.subsequent_visit_charge) / ts.subsequent_visit_charge

/-- Theorem stating that 20 customers made a second visit --/
theorem twenty_second_visits (ts : TanningSalon) 
  (h1 : ts.total_customers = 100)
  (h2 : ts.first_visit_charge = 10)
  (h3 : ts.subsequent_visit_charge = 8)
  (h4 : ts.third_visit_customers = 10)
  (h5 : ts.total_revenue = 1240) :
  second_visit_customers ts = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_second_visits_l1853_185378


namespace NUMINAMATH_CALUDE_divisibility_by_three_divisibility_by_eleven_l1853_185316

-- Part (a)
theorem divisibility_by_three (a : ℤ) (h : ∃ k : ℤ, a + 1 = 3 * k) : ∃ m : ℤ, 4 + 7 * a = 3 * m := by
  sorry

-- Part (b)
theorem divisibility_by_eleven (a b : ℤ) (h1 : ∃ m : ℤ, 2 + a = 11 * m) (h2 : ∃ n : ℤ, 35 - b = 11 * n) : ∃ p : ℤ, a + b = 11 * p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_divisibility_by_eleven_l1853_185316


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l1853_185326

theorem rectangular_plot_area 
  (breadth : ℝ) 
  (length : ℝ) 
  (h1 : breadth = 12)
  (h2 : length = 3 * breadth) : 
  breadth * length = 432 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l1853_185326


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l1853_185335

theorem min_sum_dimensions (a b c : ℕ+) : 
  a * b * c = 3003 → a + b + c ≥ 45 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l1853_185335


namespace NUMINAMATH_CALUDE_alberto_engine_spending_l1853_185355

/-- Represents the spending on car maintenance -/
structure CarSpending where
  oil : ℕ
  tires : ℕ
  detailing : ℕ

/-- Calculates the total spending for a CarSpending instance -/
def total_spending (s : CarSpending) : ℕ := s.oil + s.tires + s.detailing

/-- Represents Samara's spending -/
def samara_spending : CarSpending := { oil := 25, tires := 467, detailing := 79 }

/-- The amount Alberto spent more than Samara -/
def alberto_extra_spending : ℕ := 1886

/-- Theorem: Alberto's spending on the new engine is $2457 -/
theorem alberto_engine_spending :
  total_spending samara_spending + alberto_extra_spending = 2457 := by
  sorry

end NUMINAMATH_CALUDE_alberto_engine_spending_l1853_185355


namespace NUMINAMATH_CALUDE_distance_calculation_l1853_185306

/-- Given a journey time of 8 hours and an average speed of 23 miles per hour,
    the distance traveled is 184 miles. -/
theorem distance_calculation (journey_time : ℝ) (average_speed : ℝ) 
  (h1 : journey_time = 8)
  (h2 : average_speed = 23) :
  journey_time * average_speed = 184 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l1853_185306


namespace NUMINAMATH_CALUDE_stevens_collection_group_size_l1853_185327

theorem stevens_collection_group_size :
  let skittles : ℕ := 4502
  let erasers : ℕ := 4276
  let num_groups : ℕ := 154
  let total_items : ℕ := skittles + erasers
  (total_items / num_groups : ℕ) = 57 := by
  sorry

end NUMINAMATH_CALUDE_stevens_collection_group_size_l1853_185327


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l1853_185352

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumradius and inradius
def circumradius (t : Triangle) : ℝ := sorry
def inradius (t : Triangle) : ℝ := sorry

-- Define the points A1, B1, C1
def angle_bisector_points (t : Triangle) : Triangle := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

theorem area_ratio_theorem (t : Triangle) :
  let t1 := angle_bisector_points t
  area t / area t1 = 2 * inradius t / circumradius t := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l1853_185352


namespace NUMINAMATH_CALUDE_point_outside_ellipse_l1853_185399

theorem point_outside_ellipse (m n : ℝ) 
  (h_intersect : ∃ x y : ℝ, m * x + n * y = 4 ∧ x^2 + y^2 = 4) :
  m^2 / 4 + n^2 / 3 > 1 := by
  sorry

end NUMINAMATH_CALUDE_point_outside_ellipse_l1853_185399


namespace NUMINAMATH_CALUDE_inheritance_division_l1853_185386

theorem inheritance_division (total_amount : ℕ) (num_people : ℕ) (amount_per_person : ℕ) :
  total_amount = 527500 →
  num_people = 5 →
  amount_per_person = total_amount / num_people →
  amount_per_person = 105500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_division_l1853_185386


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1853_185384

theorem geometric_sequence_sum (a b c : ℝ) : 
  (1 < a ∧ a < b ∧ b < c ∧ c < 16) →
  (∃ q : ℝ, q ≠ 0 ∧ a = 1 * q ∧ b = a * q ∧ c = b * q ∧ 16 = c * q) →
  (a + c = 10 ∨ a + c = -10) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1853_185384


namespace NUMINAMATH_CALUDE_arccos_sin_three_equals_three_minus_pi_half_l1853_185321

theorem arccos_sin_three_equals_three_minus_pi_half :
  Real.arccos (Real.sin 3) = 3 - π / 2 := by sorry

end NUMINAMATH_CALUDE_arccos_sin_three_equals_three_minus_pi_half_l1853_185321


namespace NUMINAMATH_CALUDE_fruit_distribution_l1853_185330

/-- Given 30 pieces of fruit to be distributed equally among 4 friends,
    the smallest number of pieces to remove for equal distribution is 2. -/
theorem fruit_distribution (total_fruit : Nat) (friends : Nat) (pieces_to_remove : Nat) : 
  total_fruit = 30 →
  friends = 4 →
  pieces_to_remove = 2 →
  (total_fruit - pieces_to_remove) % friends = 0 ∧
  ∀ n : Nat, n < pieces_to_remove → (total_fruit - n) % friends ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_fruit_distribution_l1853_185330


namespace NUMINAMATH_CALUDE_perpendicular_vector_l1853_185391

def vector_AB : Fin 2 → ℝ := ![1, 1]
def vector_AC : Fin 2 → ℝ := ![2, 3]
def vector_BC : Fin 2 → ℝ := ![1, 2]
def vector_D : Fin 2 → ℝ := ![-6, 3]

theorem perpendicular_vector : 
  (vector_AB = ![1, 1]) → 
  (vector_AC = ![2, 3]) → 
  (vector_BC = vector_AC - vector_AB) →
  (vector_D • vector_BC = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vector_l1853_185391


namespace NUMINAMATH_CALUDE_valid_orders_count_l1853_185368

/-- The number of students to select -/
def n : ℕ := 4

/-- The total number of students -/
def total : ℕ := 8

/-- The number of special students (A and B) -/
def special : ℕ := 2

/-- Calculates the number of valid speaking orders -/
def validOrders : ℕ := sorry

theorem valid_orders_count :
  validOrders = 1140 := by sorry

end NUMINAMATH_CALUDE_valid_orders_count_l1853_185368


namespace NUMINAMATH_CALUDE_nearest_integer_to_sum_l1853_185303

def fraction1 : ℚ := 2007 / 2999
def fraction2 : ℚ := 8001 / 5998
def fraction3 : ℚ := 2001 / 3999

def sum : ℚ := fraction1 + fraction2 + fraction3

theorem nearest_integer_to_sum :
  round sum = 3 := by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_sum_l1853_185303


namespace NUMINAMATH_CALUDE_at_least_one_inequality_holds_l1853_185343

-- Define a triangle in 2D space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Function to check if a point is inside a triangle
def isInside (t : Triangle) (p : Point) : Prop := sorry

-- Function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem at_least_one_inequality_holds (t : Triangle) (M N : Point) :
  isInside t M →
  isInside t N →
  M ≠ N →
  (distance t.A N > distance t.A M) ∨
  (distance t.B N > distance t.B M) ∨
  (distance t.C N > distance t.C M) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_inequality_holds_l1853_185343


namespace NUMINAMATH_CALUDE_min_value_implications_l1853_185371

/-- Given a > 0, b > 0, and that the function f(x) = |x+a| + |x-b| has a minimum value of 2,
    prove the following inequalities -/
theorem min_value_implications (a b : ℝ) 
    (ha : a > 0) (hb : b > 0) 
    (hmin : ∀ x, |x + a| + |x - b| ≥ 2) : 
    (3 * a^2 + b^2 ≥ 3) ∧ (4 / (a + 1) + 1 / b ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_implications_l1853_185371


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1853_185339

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (1 - x) / (x + 2)) ↔ x ≠ -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1853_185339


namespace NUMINAMATH_CALUDE_expected_straight_flying_airplanes_l1853_185374

def flyProbability : ℚ := 3/4
def notStraightProbability : ℚ := 5/6
def totalAirplanes : ℕ := 80

theorem expected_straight_flying_airplanes :
  (totalAirplanes : ℚ) * flyProbability * (1 - notStraightProbability) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expected_straight_flying_airplanes_l1853_185374


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1853_185311

theorem smallest_n_congruence (n : ℕ+) : 
  (5 * n.val ≡ 2015 [MOD 26]) ↔ n = 21 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1853_185311


namespace NUMINAMATH_CALUDE_prime_and_power_characterization_l1853_185361

theorem prime_and_power_characterization (n : ℕ) (h : n ≥ 2) :
  (Nat.Prime n ↔ n ∣ (Nat.factorial (n - 1) + 1)) ∧
  (∃ k : ℕ, n^k = Nat.factorial (n - 1) + 1 ↔ n = 2 ∨ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_prime_and_power_characterization_l1853_185361


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1853_185357

/-- The parabola function y = x^2 - 6x + 8 -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- A square inscribed in the region bounded by the parabola and the x-axis -/
structure InscribedSquare where
  side : ℝ
  center_x : ℝ
  lower_left : ℝ × ℝ
  upper_right : ℝ × ℝ
  on_x_axis : lower_left.2 = 0 ∧ upper_right.2 = side
  on_parabola : parabola upper_right.1 = upper_right.2

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∀ (s : InscribedSquare), s.side^2 = 24 - 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1853_185357


namespace NUMINAMATH_CALUDE_x_times_one_minus_f_equals_64_l1853_185397

theorem x_times_one_minus_f_equals_64 :
  let x : ℝ := (2 + Real.sqrt 2) ^ 6
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 64 := by
  sorry

end NUMINAMATH_CALUDE_x_times_one_minus_f_equals_64_l1853_185397


namespace NUMINAMATH_CALUDE_chinese_money_plant_price_is_25_l1853_185370

/-- The price of each potted Chinese money plant -/
def chinese_money_plant_price : ℕ := sorry

/-- The number of orchids sold -/
def orchids_sold : ℕ := 20

/-- The price of each orchid -/
def orchid_price : ℕ := 50

/-- The number of potted Chinese money plants sold -/
def chinese_money_plants_sold : ℕ := 15

/-- The payment for each worker -/
def worker_payment : ℕ := 40

/-- The number of workers -/
def number_of_workers : ℕ := 2

/-- The cost of new pots -/
def new_pots_cost : ℕ := 150

/-- The amount left after expenses -/
def amount_left : ℕ := 1145

theorem chinese_money_plant_price_is_25 :
  chinese_money_plant_price = 25 ∧
  orchids_sold * orchid_price + chinese_money_plants_sold * chinese_money_plant_price =
  amount_left + number_of_workers * worker_payment + new_pots_cost :=
sorry

end NUMINAMATH_CALUDE_chinese_money_plant_price_is_25_l1853_185370


namespace NUMINAMATH_CALUDE_fraction_negative_exponent_l1853_185369

theorem fraction_negative_exponent :
  (2 / 3 : ℚ) ^ (-2 : ℤ) = 9 / 4 := by sorry

end NUMINAMATH_CALUDE_fraction_negative_exponent_l1853_185369


namespace NUMINAMATH_CALUDE_lemonade_stand_profit_is_35_l1853_185337

/-- Lemonade stand profit calculation -/
def lemonade_stand_profit : ℝ :=
  let small_yield_per_gallon : ℝ := 16
  let medium_yield_per_gallon : ℝ := 10
  let large_yield_per_gallon : ℝ := 6

  let small_cost_per_gallon : ℝ := 2.00
  let medium_cost_per_gallon : ℝ := 3.50
  let large_cost_per_gallon : ℝ := 5.00

  let small_price_per_glass : ℝ := 1.00
  let medium_price_per_glass : ℝ := 1.75
  let large_price_per_glass : ℝ := 2.50

  let gallons_made_each_size : ℝ := 2

  let small_glasses_produced : ℝ := small_yield_per_gallon * gallons_made_each_size
  let medium_glasses_produced : ℝ := medium_yield_per_gallon * gallons_made_each_size
  let large_glasses_produced : ℝ := large_yield_per_gallon * gallons_made_each_size

  let small_glasses_unsold : ℝ := 4
  let medium_glasses_unsold : ℝ := 4
  let large_glasses_unsold : ℝ := 2

  let setup_cost : ℝ := 15.00
  let advertising_cost : ℝ := 10.00

  let small_revenue := (small_glasses_produced - small_glasses_unsold) * small_price_per_glass
  let medium_revenue := (medium_glasses_produced - medium_glasses_unsold) * medium_price_per_glass
  let large_revenue := (large_glasses_produced - large_glasses_unsold) * large_price_per_glass

  let small_cost := gallons_made_each_size * small_cost_per_gallon
  let medium_cost := gallons_made_each_size * medium_cost_per_gallon
  let large_cost := gallons_made_each_size * large_cost_per_gallon

  let total_revenue := small_revenue + medium_revenue + large_revenue
  let total_cost := small_cost + medium_cost + large_cost + setup_cost + advertising_cost

  total_revenue - total_cost

theorem lemonade_stand_profit_is_35 : lemonade_stand_profit = 35 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_profit_is_35_l1853_185337


namespace NUMINAMATH_CALUDE_food_drive_mark_cans_l1853_185372

/-- Represents the number of cans brought by each person -/
structure Cans where
  mark : ℕ
  jaydon : ℕ
  sophie : ℕ
  rachel : ℕ

/-- Represents the conditions of the food drive -/
def FoodDrive (c : Cans) : Prop :=
  c.mark = 4 * c.jaydon ∧
  c.jaydon = 2 * c.rachel + 5 ∧
  c.mark + c.jaydon + c.sophie = 225 ∧
  4 * c.jaydon = 3 * c.mark ∧
  3 * c.sophie = 2 * c.mark

theorem food_drive_mark_cans :
  ∀ c : Cans, FoodDrive c → c.mark = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_food_drive_mark_cans_l1853_185372


namespace NUMINAMATH_CALUDE_percentage_problem_l1853_185379

theorem percentage_problem (P : ℝ) : 
  (0.5 * 456 = (P / 100) * 120 + 180) → P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1853_185379


namespace NUMINAMATH_CALUDE_local_minimum_implies_m_equals_two_l1853_185348

/-- The function f(x) = x(x-m)² -/
def f (x m : ℝ) : ℝ := x * (x - m)^2

/-- The derivative of f(x) with respect to x -/
def f_derivative (x m : ℝ) : ℝ := (x - m)^2 + 2*x*(x - m)

theorem local_minimum_implies_m_equals_two :
  ∀ m : ℝ, (∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x m ≥ f 2 m) →
  f_derivative 2 m = 0 →
  m = 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_implies_m_equals_two_l1853_185348


namespace NUMINAMATH_CALUDE_tiffany_bags_on_monday_l1853_185314

/-- The number of bags Tiffany had on Monday -/
def bags_on_monday : ℕ := sorry

/-- The number of bags Tiffany found on Tuesday -/
def bags_on_tuesday : ℕ := 4

/-- The total number of bags Tiffany had -/
def total_bags : ℕ := 8

/-- Theorem: Tiffany had 4 bags on Monday -/
theorem tiffany_bags_on_monday : bags_on_monday = 4 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_on_monday_l1853_185314


namespace NUMINAMATH_CALUDE_roots_sum_value_l1853_185322

-- Define the quadratic equation
def quadratic (x : ℝ) : Prop := x^2 - x - 1 = 0

-- Define the roots a and b
variable (a b : ℝ)

-- State the theorem
theorem roots_sum_value (ha : quadratic a) (hb : quadratic b) (hab : a ≠ b) :
  3 * a^2 + 4 * b + 2 / a^2 = 11 := by sorry

end NUMINAMATH_CALUDE_roots_sum_value_l1853_185322


namespace NUMINAMATH_CALUDE_function_equation_implies_identity_l1853_185338

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_identity_l1853_185338


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_l1853_185307

/-- The number of points marked on the circle -/
def n : ℕ := 20

/-- The number of sets of three intersecting chords with endpoints chosen from n points on a circle -/
def intersecting_chords_count (n : ℕ) : ℕ :=
  Nat.choose n 3 + 
  8 * Nat.choose n 4 + 
  5 * Nat.choose n 5 + 
  Nat.choose n 6

/-- Theorem stating that the number of sets of three intersecting chords 
    with endpoints chosen from 20 points on a circle is 156180 -/
theorem intersecting_chords_theorem : 
  intersecting_chords_count n = 156180 := by sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_l1853_185307


namespace NUMINAMATH_CALUDE_prob_n₂_div_2310_eq_l1853_185351

/-- The product of the first 25 primes -/
def n₀ : ℕ := sorry

/-- The Euler totient function -/
def φ : ℕ → ℕ := sorry

/-- The probability of choosing a divisor n of m, proportional to φ(n) -/
def prob_divisor (n m : ℕ) : ℚ := sorry

/-- The probability that a randomly chosen n₂ (which is a random divisor of n₁, 
    which itself is a random divisor of n₀) is divisible by 2310 -/
def prob_n₂_div_2310 : ℚ := sorry

/-- Main theorem: The probability that n₂ ≡ 0 (mod 2310) is 256/5929 -/
theorem prob_n₂_div_2310_eq : prob_n₂_div_2310 = 256 / 5929 := by sorry

end NUMINAMATH_CALUDE_prob_n₂_div_2310_eq_l1853_185351


namespace NUMINAMATH_CALUDE_max_value_of_a_plus_2b_for_tangent_line_l1853_185359

/-- Given a line ax + by = 1 (where a > 0, b > 0) tangent to the circle x² + y² = 1,
    the maximum value of a + 2b is √5. -/
theorem max_value_of_a_plus_2b_for_tangent_line :
  ∀ a b : ℝ,
  a > 0 →
  b > 0 →
  (∀ x y : ℝ, a * x + b * y = 1 → x^2 + y^2 = 1) →
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  (∀ c : ℝ, c ≥ a + 2*b → c ≥ Real.sqrt 5) ∧
  (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧
    (∀ x y : ℝ, a' * x + b' * y = 1 → x^2 + y^2 = 1) ∧
    (∃ x y : ℝ, a' * x + b' * y = 1 ∧ x^2 + y^2 = 1) ∧
    a' + 2*b' = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_plus_2b_for_tangent_line_l1853_185359


namespace NUMINAMATH_CALUDE_sum_equals_target_l1853_185350

theorem sum_equals_target : 2.75 + 0.003 + 0.158 = 2.911 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_target_l1853_185350


namespace NUMINAMATH_CALUDE_function_symmetric_about_origin_l1853_185340

/-- The function f(x) = x^3 - x is symmetric about the origin. -/
theorem function_symmetric_about_origin (x : ℝ) : let f := λ x : ℝ => x^3 - x
  f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_function_symmetric_about_origin_l1853_185340


namespace NUMINAMATH_CALUDE_tangent_circles_count_l1853_185382

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency relation between two circles
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2 ∨
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius - c2.radius)^2

theorem tangent_circles_count (c1 c2 : Circle) : 
  c1.radius = 1 →
  c2.radius = 1 →
  are_tangent c1 c2 →
  ∃ (s : Finset Circle), 
    s.card = 6 ∧ 
    (∀ c ∈ s, c.radius = 3 ∧ are_tangent c c1 ∧ are_tangent c c2) ∧
    (∀ c : Circle, c.radius = 3 ∧ are_tangent c c1 ∧ are_tangent c c2 → c ∈ s) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l1853_185382


namespace NUMINAMATH_CALUDE_evaluate_expression_l1853_185302

theorem evaluate_expression : 2 + 0 - 2 * 0 = 2 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1853_185302


namespace NUMINAMATH_CALUDE_cube_volume_l1853_185349

/-- Given a cube where the sum of all edge lengths is 48 cm, prove its volume is 64 cm³ -/
theorem cube_volume (total_edge_length : ℝ) (h : total_edge_length = 48) : 
  (total_edge_length / 12)^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l1853_185349


namespace NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l1853_185304

theorem range_of_a_when_p_is_false :
  (¬∃ (x : ℝ), x > 0 ∧ x + 1/x < a) ↔ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l1853_185304


namespace NUMINAMATH_CALUDE_students_math_or_history_not_both_l1853_185323

theorem students_math_or_history_not_both 
  (both : ℕ) 
  (math_total : ℕ) 
  (history_only : ℕ) 
  (h1 : both = 15) 
  (h2 : math_total = 30) 
  (h3 : history_only = 12) : 
  (math_total - both) + history_only = 27 := by
  sorry

end NUMINAMATH_CALUDE_students_math_or_history_not_both_l1853_185323


namespace NUMINAMATH_CALUDE_tricia_age_l1853_185331

/-- Represents the ages of individuals in the problem -/
structure Ages where
  tricia : ℕ
  amilia : ℕ
  yorick : ℕ
  eugene : ℕ
  khloe : ℕ
  rupert : ℕ
  vincent : ℕ
  selena : ℕ
  cora : ℕ
  brody : ℕ

/-- Defines the relationships between ages as given in the problem -/
def valid_ages (a : Ages) : Prop :=
  a.tricia = a.amilia / 3 ∧
  a.amilia = a.yorick / 4 ∧
  a.yorick = 2 * a.eugene ∧
  a.khloe = a.eugene / 3 ∧
  a.rupert = a.khloe + 10 ∧
  a.rupert = a.vincent - 2 ∧
  a.vincent = 22 ∧
  a.yorick = a.selena + 5 ∧
  a.selena = a.amilia + 3 ∧
  a.cora = (a.vincent + a.amilia) / 2 ∧
  a.brody = a.tricia + a.vincent

/-- Theorem stating that if the ages satisfy the given relationships, then Tricia's age is 5 -/
theorem tricia_age (a : Ages) (h : valid_ages a) : a.tricia = 5 := by
  sorry


end NUMINAMATH_CALUDE_tricia_age_l1853_185331


namespace NUMINAMATH_CALUDE_right_triangle_area_l1853_185325

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 8 * Real.sqrt 3 →
  angle = 45 * π / 180 →
  let area := (hypotenuse^2 / 4) * Real.sin angle
  area = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1853_185325


namespace NUMINAMATH_CALUDE_ball_probability_theorem_l1853_185362

/-- Probability of drawing a white ball from the nth box -/
def P (n : ℕ) : ℚ :=
  1/2 * (1/3)^n + 1/2

theorem ball_probability_theorem (n : ℕ) :
  n ≥ 2 →
  (P 2 = 5/9) ∧
  (∀ k : ℕ, k ≥ 2 → P k = 1/2 * (1/3)^k + 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ball_probability_theorem_l1853_185362


namespace NUMINAMATH_CALUDE_age_double_time_l1853_185309

/-- Given two brothers with current ages 15 and 5, this theorem proves that
    it will take 5 years for the older brother's age to be twice the younger brother's age. -/
theorem age_double_time (older_age younger_age : ℕ) (h1 : older_age = 15) (h2 : younger_age = 5) :
  ∃ (years : ℕ), years = 5 ∧ older_age + years = 2 * (younger_age + years) :=
sorry

end NUMINAMATH_CALUDE_age_double_time_l1853_185309


namespace NUMINAMATH_CALUDE_connie_needs_4999_l1853_185388

/-- Calculates the additional amount Connie needs to buy the items --/
def additional_amount_needed (saved : ℚ) (watch_price : ℚ) (strap_original : ℚ) (strap_discount : ℚ) 
  (case_price : ℚ) (protector_price_eur : ℚ) (tax_rate : ℚ) (exchange_rate : ℚ) : ℚ :=
  let strap_price := strap_original * (1 - strap_discount)
  let protector_price_usd := protector_price_eur * exchange_rate
  let subtotal := watch_price + strap_price + case_price + protector_price_usd
  let total_with_tax := subtotal * (1 + tax_rate)
  (total_with_tax - saved).ceil / 100

/-- The theorem stating the additional amount Connie needs --/
theorem connie_needs_4999 : 
  additional_amount_needed 39 55 20 0.25 10 2 0.08 1.2 = 4999 / 100 := by
  sorry

end NUMINAMATH_CALUDE_connie_needs_4999_l1853_185388


namespace NUMINAMATH_CALUDE_intersection_M_N_l1853_185318

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1853_185318


namespace NUMINAMATH_CALUDE_initial_volume_proof_l1853_185385

/-- Proves that the initial volume of a solution is 40 liters given the conditions of the problem -/
theorem initial_volume_proof (V : ℝ) : 
  (0.05 * V + 4.5 = 0.13 * (V + 10)) → V = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_volume_proof_l1853_185385


namespace NUMINAMATH_CALUDE_find_B_l1853_185308

theorem find_B (A B : ℚ) : (1 / 4 : ℚ) * (1 / 8 : ℚ) = 1 / (4 * A) ∧ 1 / (4 * A) = 1 / B → B = 32 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l1853_185308


namespace NUMINAMATH_CALUDE_class_mean_calculation_l1853_185346

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_mean : ℚ)
  (group2_students : ℕ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 40 →
  group2_students = 10 →
  group1_mean = 68 / 100 →
  group2_mean = 74 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 692 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l1853_185346


namespace NUMINAMATH_CALUDE_shaded_area_semicircles_l1853_185328

/-- The area of shaded region formed by semicircles in a pattern -/
theorem shaded_area_semicircles (diameter : ℝ) (pattern_length : ℝ) : 
  diameter = 3 →
  pattern_length = 12 →
  (pattern_length / diameter) * (π * (diameter / 2)^2 / 2) = 9 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircles_l1853_185328


namespace NUMINAMATH_CALUDE_square_root_fraction_simplification_l1853_185366

theorem square_root_fraction_simplification :
  Real.sqrt (7^2 + 24^2) / Real.sqrt (64 + 36) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_simplification_l1853_185366


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l1853_185317

theorem factorization_of_cubic (a : ℝ) : 
  -2 * a^3 + 12 * a^2 - 18 * a = -2 * a * (a - 3)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l1853_185317


namespace NUMINAMATH_CALUDE_days_2000_to_2005_l1853_185373

/-- The number of days in a given range of years -/
def totalDays (totalYears : ℕ) (leapYears : ℕ) (nonLeapDays : ℕ) (leapDays : ℕ) : ℕ :=
  (totalYears - leapYears) * nonLeapDays + leapYears * leapDays

/-- Theorem stating that the total number of days from 2000 to 2005 (inclusive) is 2192 -/
theorem days_2000_to_2005 : totalDays 6 2 365 366 = 2192 := by
  sorry

end NUMINAMATH_CALUDE_days_2000_to_2005_l1853_185373


namespace NUMINAMATH_CALUDE_fraction_difference_equals_negative_one_l1853_185393

theorem fraction_difference_equals_negative_one 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x * y) : 
  1 / x - 1 / y = -1 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_negative_one_l1853_185393


namespace NUMINAMATH_CALUDE_max_a_is_eight_l1853_185363

/-- The quadratic polynomial f(x) = ax^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

/-- The condition that |f(x)| ≤ 1 for all x in [0, 1] -/
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1

/-- The maximum value of a is 8 -/
theorem max_a_is_eight :
  (∃ a : ℝ, condition a) →
  (∀ a : ℝ, condition a → a ≤ 8) ∧
  condition 8 :=
sorry

end NUMINAMATH_CALUDE_max_a_is_eight_l1853_185363


namespace NUMINAMATH_CALUDE_paris_hair_theorem_l1853_185360

theorem paris_hair_theorem (population : ℕ) (max_hair_count : ℕ) 
  (h1 : population > 2000000) 
  (h2 : max_hair_count = 150000) : 
  ∃ (hair_count : ℕ), hair_count ≤ max_hair_count ∧ 
  (∃ (group : Finset (Fin population)), group.card ≥ 14 ∧ 
  ∀ i ∈ group, hair_count = (i : ℕ)) :=
sorry

end NUMINAMATH_CALUDE_paris_hair_theorem_l1853_185360


namespace NUMINAMATH_CALUDE_max_value_on_circle_l1853_185389

theorem max_value_on_circle (x y : ℝ) :
  (x - 1)^2 + y^2 = 1 →
  ∃ (max : ℝ), (∀ (x' y' : ℝ), (x' - 1)^2 + y'^2 = 1 → 2*x' + y' ≤ max) ∧ max = Real.sqrt 5 + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l1853_185389


namespace NUMINAMATH_CALUDE_sum_and_count_integers_l1853_185376

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_integers : sum_integers 60 80 + count_even_integers 60 80 = 1481 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_integers_l1853_185376


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l1853_185342

theorem negation_of_existence_inequality : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l1853_185342


namespace NUMINAMATH_CALUDE_arcsin_neg_half_eq_neg_pi_sixth_l1853_185365

theorem arcsin_neg_half_eq_neg_pi_sixth : 
  Real.arcsin (-1/2) = -π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_neg_half_eq_neg_pi_sixth_l1853_185365


namespace NUMINAMATH_CALUDE_book_difference_l1853_185320

/- Define the number of books for each category -/
def total_books : ℕ := 220
def hardcover_nonfiction : ℕ := 40

/- Define the properties of the book categories -/
def book_categories (paperback_fiction paperback_nonfiction : ℕ) : Prop :=
  paperback_fiction + paperback_nonfiction + hardcover_nonfiction = total_books ∧
  paperback_nonfiction > hardcover_nonfiction ∧
  paperback_fiction = 2 * paperback_nonfiction

/- Theorem statement -/
theorem book_difference :
  ∃ (paperback_fiction paperback_nonfiction : ℕ),
    book_categories paperback_fiction paperback_nonfiction ∧
    paperback_nonfiction - hardcover_nonfiction = 20 :=
by sorry

end NUMINAMATH_CALUDE_book_difference_l1853_185320


namespace NUMINAMATH_CALUDE_aquarium_count_l1853_185394

theorem aquarium_count (total_animals : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : animals_per_aquarium = 2) 
  (h3 : animals_per_aquarium > 0) : 
  total_animals / animals_per_aquarium = 20 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_count_l1853_185394


namespace NUMINAMATH_CALUDE_number_problem_l1853_185312

theorem number_problem : ∃! x : ℚ, (1 / 4 : ℚ) * x > (1 / 5 : ℚ) * (x + 1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1853_185312


namespace NUMINAMATH_CALUDE_trail_mix_fruit_percentage_l1853_185387

-- Define the trail mix compositions
def sue_mix : ℝ := 5
def sue_nuts_percent : ℝ := 0.3
def sue_fruit_percent : ℝ := 0.7

def jane_mix : ℝ := 7
def jane_nuts_percent : ℝ := 0.6

def tom_mix : ℝ := 9
def tom_nuts_percent : ℝ := 0.4
def tom_fruit_percent : ℝ := 0.5

-- Define the combined mixture properties
def combined_nuts_percent : ℝ := 0.45

-- Theorem to prove
theorem trail_mix_fruit_percentage :
  let total_nuts := sue_mix * sue_nuts_percent + jane_mix * jane_nuts_percent + tom_mix * tom_nuts_percent
  let total_weight := total_nuts / combined_nuts_percent
  let total_fruit := sue_mix * sue_fruit_percent + tom_mix * tom_fruit_percent
  let fruit_percentage := total_fruit / total_weight * 100
  abs (fruit_percentage - 38.71) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_trail_mix_fruit_percentage_l1853_185387


namespace NUMINAMATH_CALUDE_cyrus_family_size_cyrus_mosquito_bites_l1853_185354

theorem cyrus_family_size (cyrus_arms_legs : ℕ) (cyrus_body : ℕ) : ℕ :=
  let cyrus_total := cyrus_arms_legs + cyrus_body
  let family_total := cyrus_total / 2
  family_total

theorem cyrus_mosquito_bites : cyrus_family_size 14 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cyrus_family_size_cyrus_mosquito_bites_l1853_185354


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1853_185364

theorem quadratic_inequality_solution_sets (m x : ℝ) :
  let f := fun x => m * x^2 - (m + 1) * x + 1
  (m = 2 → (f x < 0 ↔ 1/2 < x ∧ x < 1)) ∧
  (m > 0 →
    ((0 < m ∧ m < 1) → (f x < 0 ↔ 1 < x ∧ x < 1/m)) ∧
    (m = 1 → ¬∃ x, f x < 0) ∧
    (m > 1 → (f x < 0 ↔ 1/m < x ∧ x < 1))) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1853_185364


namespace NUMINAMATH_CALUDE_circle_has_infinite_symmetry_lines_l1853_185375

-- Define a circle
def Circle : Type := Unit

-- Define a line of symmetry for a circle
def LineOfSymmetry (c : Circle) : Type := Unit

-- Define the property of having an infinite number of lines of symmetry
def HasInfiniteSymmetryLines (c : Circle) : Prop :=
  ∀ (n : ℕ), ∃ (lines : Fin n → LineOfSymmetry c), Function.Injective lines

-- Theorem statement
theorem circle_has_infinite_symmetry_lines (c : Circle) :
  HasInfiniteSymmetryLines c := by sorry

end NUMINAMATH_CALUDE_circle_has_infinite_symmetry_lines_l1853_185375


namespace NUMINAMATH_CALUDE_sphere_volume_l1853_185332

theorem sphere_volume (d : ℝ) (a : ℝ) (h1 : d = 2) (h2 : a = π) :
  let r := Real.sqrt (1^2 + d^2)
  (4 / 3) * π * r^3 = (20 * Real.sqrt 5 * π) / 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_l1853_185332


namespace NUMINAMATH_CALUDE_chelsea_cupcake_time_l1853_185333

/-- The time it takes to make cupcakes given the number of batches and time per batch -/
def cupcake_time (num_batches : ℕ) (bake_time : ℕ) (ice_time : ℕ) : ℕ :=
  num_batches * (bake_time + ice_time)

/-- Theorem: Chelsea's cupcake-making time -/
theorem chelsea_cupcake_time :
  cupcake_time 4 20 30 = 200 := by
  sorry

end NUMINAMATH_CALUDE_chelsea_cupcake_time_l1853_185333


namespace NUMINAMATH_CALUDE_diamond_property_false_l1853_185300

def diamond (x y : ℝ) : ℝ := 2 * |x - y| + 1

theorem diamond_property_false :
  ¬ ∀ x y : ℝ, 3 * (diamond x y) = 3 * (diamond (2*x) (2*y)) :=
sorry

end NUMINAMATH_CALUDE_diamond_property_false_l1853_185300


namespace NUMINAMATH_CALUDE_factor_calculation_l1853_185341

theorem factor_calculation : ∃ f : ℚ, (2 * 7 + 9) * f = 69 ∧ f = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l1853_185341


namespace NUMINAMATH_CALUDE_sum_of_flipped_digits_is_19_l1853_185377

/-- Function to flip a digit upside down -/
def flip_digit (d : ℕ) : ℕ := sorry

/-- Function to flip a number upside down -/
def flip_number (n : ℕ) : ℕ := sorry

/-- Function to sum the digits of a number -/
def sum_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of flipped digits is 19 -/
theorem sum_of_flipped_digits_is_19 :
  sum_digits (flip_number 340) +
  sum_digits (flip_number 24813) +
  sum_digits (flip_number 43323414) = 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_flipped_digits_is_19_l1853_185377


namespace NUMINAMATH_CALUDE_rational_fraction_implies_integer_sum_squares_over_sum_l1853_185315

theorem rational_fraction_implies_integer_sum_squares_over_sum (a b c : ℕ+) :
  (∃ (r s : ℤ), (r : ℚ) / s = (a * Real.sqrt 3 + b) / (b * Real.sqrt 3 + c)) →
  ∃ (k : ℤ), (a ^ 2 + b ^ 2 + c ^ 2 : ℚ) / (a + b + c) = k := by
sorry

end NUMINAMATH_CALUDE_rational_fraction_implies_integer_sum_squares_over_sum_l1853_185315


namespace NUMINAMATH_CALUDE_loan_amount_calculation_l1853_185390

def college_cost : ℝ := 30000
def savings : ℝ := 10000
def grant_percentage : ℝ := 0.4

theorem loan_amount_calculation : 
  let remainder := college_cost - savings
  let grant_amount := remainder * grant_percentage
  let loan_amount := remainder - grant_amount
  loan_amount = 12000 := by sorry

end NUMINAMATH_CALUDE_loan_amount_calculation_l1853_185390


namespace NUMINAMATH_CALUDE_equation_solution_l1853_185356

theorem equation_solution : 
  ∃! x : ℝ, x + 36 / (x - 3) = -9 ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1853_185356


namespace NUMINAMATH_CALUDE_range_of_m_l1853_185398

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x ∈ Set.Icc 0 1, x^2 - m*x - 2 = 0

def q (m : ℝ) : Prop := ∀ x ≥ 1, 
  (∀ y ≥ x, (y^2 - 2*m*y + 1/2) / (x^2 - 2*m*x + 1/2) ≥ 1) ∧ 
  (x^2 - 2*m*x + 1/2 > 0)

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (¬(p m) ∧ (p m ∨ q m)) → (m > -1 ∧ m < 3/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1853_185398


namespace NUMINAMATH_CALUDE_seeds_sown_l1853_185358

/-- Given a farmer who started with 8.75 buckets of seeds and ended with 6 buckets,
    prove that the number of buckets sown is 2.75. -/
theorem seeds_sown (initial : ℝ) (remaining : ℝ) (h1 : initial = 8.75) (h2 : remaining = 6) :
  initial - remaining = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_seeds_sown_l1853_185358


namespace NUMINAMATH_CALUDE_comic_books_liked_by_females_l1853_185324

/-- Given a comic store with the following properties:
  - There are 300 comic books in total
  - Males like 120 comic books
  - 30% of comic books are disliked by both males and females
  Prove that the percentage of comic books liked by females is 30% -/
theorem comic_books_liked_by_females 
  (total_comics : ℕ) 
  (liked_by_males : ℕ) 
  (disliked_percentage : ℚ) :
  total_comics = 300 →
  liked_by_males = 120 →
  disliked_percentage = 30 / 100 →
  (total_comics - (disliked_percentage * total_comics).num - liked_by_males) / total_comics = 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_comic_books_liked_by_females_l1853_185324


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l1853_185367

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (3 * x) % 31 = 15 % 31 ∧ 
  ∀ (y : ℕ), y > 0 ∧ (3 * y) % 31 = 15 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l1853_185367


namespace NUMINAMATH_CALUDE_table_formula_proof_l1853_185347

def f (x : ℤ) : ℤ := x^2 - 4*x + 1

theorem table_formula_proof :
  (f 1 = -2) ∧ 
  (f 2 = 0) ∧ 
  (f 3 = 4) ∧ 
  (f 4 = 10) ∧ 
  (f 5 = 18) := by
  sorry

end NUMINAMATH_CALUDE_table_formula_proof_l1853_185347


namespace NUMINAMATH_CALUDE_unique_tuple_existence_l1853_185319

theorem unique_tuple_existence 
  (p q : ℝ) 
  (h_pos_p : 0 < p) 
  (h_pos_q : 0 < q) 
  (h_sum : p + q = 1) 
  (y : Fin 2017 → ℝ) : 
  ∃! x : Fin 2018 → ℝ, 
    (∀ i : Fin 2017, p * max (x i) (x (i + 1)) + q * min (x i) (x (i + 1)) = y i) ∧ 
    x 0 = x 2017 := by
  sorry

end NUMINAMATH_CALUDE_unique_tuple_existence_l1853_185319


namespace NUMINAMATH_CALUDE_cloak_change_theorem_l1853_185310

/-- Represents the price and change for buying an invisibility cloak -/
structure CloakTransaction where
  silver_paid : ℕ
  gold_change : ℕ

/-- Calculates the change in silver coins when buying a cloak with gold coins -/
def calculate_silver_change (t1 t2 : CloakTransaction) (gold_paid : ℕ) : ℕ :=
  sorry

theorem cloak_change_theorem (t1 t2 : CloakTransaction) 
  (h1 : t1.silver_paid = 20 ∧ t1.gold_change = 4)
  (h2 : t2.silver_paid = 15 ∧ t2.gold_change = 1) :
  calculate_silver_change t1 t2 14 = 10 :=
sorry

end NUMINAMATH_CALUDE_cloak_change_theorem_l1853_185310


namespace NUMINAMATH_CALUDE_water_depth_is_12_feet_l1853_185305

/-- The height of Ron in feet -/
def ron_height : ℝ := 14

/-- The difference in height between Ron and Dean in feet -/
def height_difference : ℝ := 8

/-- The height of Dean in feet -/
def dean_height : ℝ := ron_height - height_difference

/-- The depth of the water as a multiple of Dean's height -/
def water_depth_factor : ℝ := 2

/-- The depth of the water in feet -/
def water_depth : ℝ := water_depth_factor * dean_height

theorem water_depth_is_12_feet : water_depth = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_is_12_feet_l1853_185305


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1853_185353

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2) ∧
  (x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1853_185353
