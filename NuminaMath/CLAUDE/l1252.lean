import Mathlib

namespace NUMINAMATH_CALUDE_rabbit_speed_l1252_125255

def rabbit_speed_equation (x : ℝ) : Prop :=
  2 * (2 * x + 4) = 188

theorem rabbit_speed : ∃ (x : ℝ), rabbit_speed_equation x ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_speed_l1252_125255


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1252_125212

theorem simplify_trig_expression (x : ℝ) : 
  Real.sqrt 2 * Real.cos x + Real.sqrt 6 * Real.sin x = 
  2 * Real.sqrt 2 * Real.cos (π / 3 - x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1252_125212


namespace NUMINAMATH_CALUDE_new_plan_cost_theorem_l1252_125258

def old_phone_plan_cost : ℝ := 150
def old_internet_cost : ℝ := 50
def old_calling_cost : ℝ := 30
def old_streaming_cost : ℝ := 40

def new_phone_plan_increase : ℝ := 0.30
def new_internet_increase : ℝ := 0.20
def new_calling_discount : ℝ := 0.15
def new_streaming_increase : ℝ := 0.25
def promotional_discount : ℝ := 0.10

def new_phone_plan_cost : ℝ := old_phone_plan_cost * (1 + new_phone_plan_increase)
def new_internet_cost : ℝ := old_internet_cost * (1 + new_internet_increase)
def new_calling_cost : ℝ := old_calling_cost * (1 - new_calling_discount)
def new_streaming_cost : ℝ := old_streaming_cost * (1 + new_streaming_increase)

def total_cost_before_discount : ℝ := 
  new_phone_plan_cost + new_internet_cost + new_calling_cost + new_streaming_cost

def total_cost_after_discount : ℝ := 
  total_cost_before_discount * (1 - promotional_discount)

theorem new_plan_cost_theorem : 
  total_cost_after_discount = 297.45 := by sorry

end NUMINAMATH_CALUDE_new_plan_cost_theorem_l1252_125258


namespace NUMINAMATH_CALUDE_min_value_and_nonexistence_l1252_125238

theorem min_value_and_nonexistence (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a + b) * Real.sqrt (a * b) = 1) :
  (∀ x y, x > 0 → y > 0 → (x + y) * Real.sqrt (x * y) = 1 → 1 / x^3 + 1 / y^3 ≥ 1 / a^3 + 1 / b^3) ∧
  1 / a^3 + 1 / b^3 = 4 * Real.sqrt 2 ∧
  ¬∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ 1 / (2 * c) + 1 / (3 * d) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_nonexistence_l1252_125238


namespace NUMINAMATH_CALUDE_opposite_unit_vector_l1252_125208

def a : ℝ × ℝ := (12, 5)

theorem opposite_unit_vector :
  let magnitude := Real.sqrt (a.1^2 + a.2^2)
  (-a.1 / magnitude, -a.2 / magnitude) = (-12/13, -5/13) := by
  sorry

end NUMINAMATH_CALUDE_opposite_unit_vector_l1252_125208


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l1252_125219

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, (x + 2) / (x - 1) = y) ↔ x ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l1252_125219


namespace NUMINAMATH_CALUDE_factor_sum_l1252_125286

/-- If x^2 + 2√2x + 5 is a factor of x^4 + Px^2 + Q, then P + Q = 27 -/
theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, (x^2 + 2 * Real.sqrt 2 * x + 5) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  P + Q = 27 := by
  sorry

end NUMINAMATH_CALUDE_factor_sum_l1252_125286


namespace NUMINAMATH_CALUDE_vw_toyota_ratio_l1252_125206

/-- The number of Dodge trucks in the parking lot -/
def dodge_trucks : ℕ := 60

/-- The number of Volkswagen Bugs in the parking lot -/
def vw_bugs : ℕ := 5

/-- The number of Ford trucks in the parking lot -/
def ford_trucks : ℕ := dodge_trucks / 3

/-- The number of Toyota trucks in the parking lot -/
def toyota_trucks : ℕ := ford_trucks / 2

/-- The ratio of Volkswagen Bugs to Toyota trucks is 1:2 -/
theorem vw_toyota_ratio : 
  (vw_bugs : ℚ) / toyota_trucks = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_vw_toyota_ratio_l1252_125206


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1252_125231

/-- The inclination angle of a line given by the equation x*cos(140°) + y*sin(40°) + 1 = 0 is 50°. -/
theorem line_inclination_angle (x y : ℝ) :
  x * Real.cos (140 * π / 180) + y * Real.sin (40 * π / 180) + 1 = 0 →
  Real.arctan (Real.tan (50 * π / 180)) = 50 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1252_125231


namespace NUMINAMATH_CALUDE_goldbach_counterexample_characterization_l1252_125268

-- Define Goldbach's conjecture
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- Define what constitutes a counterexample to Goldbach's conjecture
def is_goldbach_counterexample (n : ℕ) : Prop :=
  n > 2 ∧ Even n ∧ ¬(∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q)

-- The theorem to prove
theorem goldbach_counterexample_characterization :
  ∀ n : ℕ, is_goldbach_counterexample n ↔ ¬goldbach_conjecture :=
by sorry

end NUMINAMATH_CALUDE_goldbach_counterexample_characterization_l1252_125268


namespace NUMINAMATH_CALUDE_warren_guests_proof_l1252_125299

/-- The number of tables Warren has -/
def num_tables : ℝ := 252.0

/-- The number of guests each table can hold -/
def guests_per_table : ℝ := 4.0

/-- The total number of guests Warren can accommodate -/
def total_guests : ℝ := num_tables * guests_per_table

theorem warren_guests_proof : total_guests = 1008.0 := by
  sorry

end NUMINAMATH_CALUDE_warren_guests_proof_l1252_125299


namespace NUMINAMATH_CALUDE_f_is_odd_l1252_125264

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0

-- The functional equation
axiom functional_equation : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * Real.cos b

-- The theorem to prove
theorem f_is_odd : (∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * Real.cos b) → 
  (∀ x : ℝ, f (-x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_f_is_odd_l1252_125264


namespace NUMINAMATH_CALUDE_original_price_is_20_l1252_125278

/-- Represents the ticket pricing scenario for a concert --/
structure ConcertTickets where
  original_price : ℝ
  total_revenue : ℝ
  total_tickets : ℕ
  discount_40_count : ℕ
  discount_15_count : ℕ

/-- The concert ticket scenario satisfies the given conditions --/
def valid_scenario (c : ConcertTickets) : Prop :=
  c.total_tickets = 50 ∧
  c.discount_40_count = 10 ∧
  c.discount_15_count = 20 ∧
  c.total_revenue = 860 ∧
  c.total_revenue = (c.discount_40_count : ℝ) * (0.6 * c.original_price) +
                    (c.discount_15_count : ℝ) * (0.85 * c.original_price) +
                    ((c.total_tickets - c.discount_40_count - c.discount_15_count) : ℝ) * c.original_price

/-- The original ticket price is $20 --/
theorem original_price_is_20 (c : ConcertTickets) (h : valid_scenario c) :
  c.original_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_price_is_20_l1252_125278


namespace NUMINAMATH_CALUDE_min_formula_l1252_125222

theorem min_formula (a b : ℝ) : min a b = (a + b - Real.sqrt ((a - b)^2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_formula_l1252_125222


namespace NUMINAMATH_CALUDE_exists_n_f_div_g_eq_2012_l1252_125227

/-- The number of divisors of n which are perfect squares -/
def f (n : ℕ+) : ℕ := sorry

/-- The number of divisors of n which are perfect cubes -/
def g (n : ℕ+) : ℕ := sorry

/-- There exists a positive integer n such that f(n) / g(n) = 2012 -/
theorem exists_n_f_div_g_eq_2012 : ∃ n : ℕ+, (f n : ℚ) / (g n : ℚ) = 2012 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_f_div_g_eq_2012_l1252_125227


namespace NUMINAMATH_CALUDE_circumcircle_radius_of_triangle_l1252_125228

theorem circumcircle_radius_of_triangle (a b c : ℚ) :
  a = 15/2 ∧ b = 10 ∧ c = 25/2 →
  a^2 + b^2 = c^2 →
  (c/2 : ℚ) = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_radius_of_triangle_l1252_125228


namespace NUMINAMATH_CALUDE_no_seven_edge_polyhedron_l1252_125234

/-- A polyhedron is a structure with vertices, edges, and faces. -/
structure Polyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces

/-- Euler's formula for polyhedra -/
axiom euler_formula (p : Polyhedron) : p.V - p.E + p.F = 2

/-- Each vertex in a polyhedron has at least 3 edges -/
axiom vertex_edge_count (p : Polyhedron) : p.E * 2 ≥ p.V * 3

/-- A polyhedron must have at least 4 vertices -/
axiom min_vertices (p : Polyhedron) : p.V ≥ 4

/-- Theorem: No polyhedron can have exactly 7 edges -/
theorem no_seven_edge_polyhedron :
  ¬∃ (p : Polyhedron), p.E = 7 := by sorry

end NUMINAMATH_CALUDE_no_seven_edge_polyhedron_l1252_125234


namespace NUMINAMATH_CALUDE_upper_limit_range_l1252_125275

theorem upper_limit_range (n x : ℝ) : 
  3 < n ∧ n < x ∧ 6 < n ∧ n < 10 ∧ n = 7 → x > 7 := by
sorry

end NUMINAMATH_CALUDE_upper_limit_range_l1252_125275


namespace NUMINAMATH_CALUDE_old_refrigerator_cost_proof_l1252_125298

/-- The daily cost of Kurt's new refrigerator in dollars -/
def new_refrigerator_cost : ℝ := 0.45

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The amount Kurt saves in a month with his new refrigerator in dollars -/
def monthly_savings : ℝ := 12

/-- The daily cost of Kurt's old refrigerator in dollars -/
def old_refrigerator_cost : ℝ := 0.85

theorem old_refrigerator_cost_proof : 
  old_refrigerator_cost * days_in_month - new_refrigerator_cost * days_in_month = monthly_savings :=
sorry

end NUMINAMATH_CALUDE_old_refrigerator_cost_proof_l1252_125298


namespace NUMINAMATH_CALUDE_dads_age_l1252_125266

theorem dads_age (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 9 →
  age_difference = 27 →
  (4 : ℕ) * son_age + age_difference = 63 := by
sorry

end NUMINAMATH_CALUDE_dads_age_l1252_125266


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l1252_125280

-- Define the speeds and time
def alice_speed : ℚ := 1 / 12  -- miles per minute
def bob_speed : ℚ := 3 / 20    -- miles per minute
def duration : ℚ := 120        -- minutes (2 hours)

-- Theorem statement
theorem distance_after_two_hours :
  let alice_distance := alice_speed * duration
  let bob_distance := bob_speed * duration
  alice_distance + bob_distance = 28 := by
sorry


end NUMINAMATH_CALUDE_distance_after_two_hours_l1252_125280


namespace NUMINAMATH_CALUDE_second_chord_length_l1252_125225

/-- Represents a chord in a circle -/
structure Chord :=
  (length : ℝ)
  (segment1 : ℝ)
  (segment2 : ℝ)
  (valid : segment1 > 0 ∧ segment2 > 0 ∧ length = segment1 + segment2)

/-- Theorem: Length of the second chord given intersecting chords -/
theorem second_chord_length
  (chord1 : Chord)
  (chord2 : Chord)
  (h1 : chord1.segment1 = 12 ∧ chord1.segment2 = 18)
  (h2 : chord2.segment1 / chord2.segment2 = 3 / 8)
  (h3 : chord1.segment1 * chord1.segment2 = chord2.segment1 * chord2.segment2) :
  chord2.length = 33 :=
sorry

end NUMINAMATH_CALUDE_second_chord_length_l1252_125225


namespace NUMINAMATH_CALUDE_minimum_class_size_minimum_class_size_is_21_l1252_125244

theorem minimum_class_size : ℕ → Prop :=
  fun n =>
    ∃ (boys girls : ℕ),
      boys > 0 ∧ girls > 0 ∧
      (3 * boys = 4 * ((2 * girls) / 3)) ∧
      n = boys + girls + 4 ∧
      ∀ m, m < n →
        ¬∃ (b g : ℕ),
          b > 0 ∧ g > 0 ∧
          (3 * b = 4 * ((2 * g) / 3)) ∧
          m = b + g + 4

theorem minimum_class_size_is_21 :
  minimum_class_size 21 := by
  sorry

end NUMINAMATH_CALUDE_minimum_class_size_minimum_class_size_is_21_l1252_125244


namespace NUMINAMATH_CALUDE_inequality_equiv_range_l1252_125243

/-- The function f(x) = x³ + x + 1 -/
def f (x : ℝ) : ℝ := x^3 + x + 1

/-- The theorem stating the equivalence between the inequality and the range of x -/
theorem inequality_equiv_range :
  ∀ x : ℝ, (f (1 - x) + f (2 * x) > 2) ↔ x > -1 :=
sorry

end NUMINAMATH_CALUDE_inequality_equiv_range_l1252_125243


namespace NUMINAMATH_CALUDE_last_installment_value_is_3300_l1252_125289

/-- Represents the payment structure for a TV set purchase -/
structure TVPayment where
  price : ℕ               -- Price of the TV set in Rupees
  num_installments : ℕ    -- Number of installments
  installment_amount : ℕ  -- Amount of each installment in Rupees
  interest_rate : ℚ       -- Annual interest rate as a rational number
  processing_fee : ℕ      -- Processing fee in Rupees

/-- Calculates the value of the last installment for a TV payment plan -/
def last_installment_value (payment : TVPayment) : ℕ :=
  payment.installment_amount + payment.processing_fee

/-- Theorem stating that the last installment value for the given TV payment plan is 3300 Rupees -/
theorem last_installment_value_is_3300 (payment : TVPayment) 
  (h1 : payment.price = 35000)
  (h2 : payment.num_installments = 36)
  (h3 : payment.installment_amount = 2300)
  (h4 : payment.interest_rate = 9 / 100)
  (h5 : payment.processing_fee = 1000) :
  last_installment_value payment = 3300 := by
  sorry

#eval last_installment_value { 
  price := 35000, 
  num_installments := 36, 
  installment_amount := 2300, 
  interest_rate := 9 / 100, 
  processing_fee := 1000 
}

end NUMINAMATH_CALUDE_last_installment_value_is_3300_l1252_125289


namespace NUMINAMATH_CALUDE_jim_total_cost_l1252_125246

def total_cost (lamp_price bulb_price bedside_table_price decorative_item_price : ℝ)
               (lamp_quantity bulb_quantity bedside_table_quantity decorative_item_quantity : ℕ)
               (lamp_discount bulb_discount bedside_table_discount decorative_item_discount : ℝ)
               (lamp_tax_rate bulb_tax_rate bedside_table_tax_rate decorative_item_tax_rate : ℝ) : ℝ :=
  let lamp_cost := lamp_quantity * lamp_price * (1 - lamp_discount) * (1 + lamp_tax_rate)
  let bulb_cost := bulb_quantity * bulb_price * (1 - bulb_discount) * (1 + bulb_tax_rate)
  let bedside_table_cost := bedside_table_quantity * bedside_table_price * (1 - bedside_table_discount) * (1 + bedside_table_tax_rate)
  let decorative_item_cost := decorative_item_quantity * decorative_item_price * (1 - decorative_item_discount) * (1 + decorative_item_tax_rate)
  lamp_cost + bulb_cost + bedside_table_cost + decorative_item_cost

theorem jim_total_cost :
  total_cost 12 8 25 10 2 6 3 4 0.2 0.3 0 0.15 0.05 0.05 0.06 0.04 = 170.30 := by
  sorry

end NUMINAMATH_CALUDE_jim_total_cost_l1252_125246


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l1252_125214

def is_valid_student_count (n : ℕ) : Prop :=
  n > 0 ∧ 120 % (2 * n) = 0

theorem candy_distribution_theorem :
  ∀ n : ℕ, is_valid_student_count n ↔ n ∈ ({5, 6, 10, 12, 15} : Finset ℕ) :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l1252_125214


namespace NUMINAMATH_CALUDE_pascal_triangle_45th_number_51_entries_l1252_125248

theorem pascal_triangle_45th_number_51_entries : 
  let n : ℕ := 50  -- The row number (0-indexed) with 51 entries
  let k : ℕ := 44  -- The position (0-indexed) of the 45th number
  Nat.choose n k = 19380000 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_45th_number_51_entries_l1252_125248


namespace NUMINAMATH_CALUDE_first_group_size_first_group_size_is_16_l1252_125284

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℝ := 25

/-- The number of men in the second group -/
def men_second_group : ℝ := 15

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℝ := 26.666666666666668

/-- The work done is inversely proportional to the number of days taken -/
axiom work_time_inverse_proportion {m1 m2 d1 d2 : ℝ} :
  m1 * d1 = m2 * d2

theorem first_group_size : ℝ := by
  sorry

theorem first_group_size_is_16 : first_group_size = 16 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_first_group_size_is_16_l1252_125284


namespace NUMINAMATH_CALUDE_stomachion_gray_area_l1252_125281

/-- A square with side length 12 cm divided into 14 polygons -/
structure StomachionPuzzle where
  side_length : ℝ
  num_polygons : ℕ
  h_side : side_length = 12
  h_polygons : num_polygons = 14

/-- A quadrilateral in the Stomachion puzzle -/
structure Quadrilateral (puzzle : StomachionPuzzle) where
  area : ℝ

/-- There exists a quadrilateral in the Stomachion puzzle with an area of 12 cm² -/
theorem stomachion_gray_area (puzzle : StomachionPuzzle) :
  ∃ (q : Quadrilateral puzzle), q.area = 12 := by
  sorry

end NUMINAMATH_CALUDE_stomachion_gray_area_l1252_125281


namespace NUMINAMATH_CALUDE_sine_transformation_l1252_125203

theorem sine_transformation (x : ℝ) : 
  Real.sin (2 * x + (2 * Real.pi) / 3) = Real.sin (2 * (x + Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_sine_transformation_l1252_125203


namespace NUMINAMATH_CALUDE_bruce_books_purchased_l1252_125217

def bruce_purchase (num_books : ℕ) : Prop :=
  let crayon_cost : ℕ := 5 * 5
  let calculator_cost : ℕ := 3 * 5
  let total_cost : ℕ := crayon_cost + calculator_cost + num_books * 5
  let remaining_money : ℕ := 200 - total_cost
  remaining_money = 11 * 10

theorem bruce_books_purchased : ∃ (num_books : ℕ), bruce_purchase num_books ∧ num_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_bruce_books_purchased_l1252_125217


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l1252_125293

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l1252_125293


namespace NUMINAMATH_CALUDE_arc_measure_constant_l1252_125296

/-- A right isosceles triangle with a rotating circle -/
structure RightIsoscelesWithCircle where
  -- The side length of the right isosceles triangle
  s : ℝ
  -- Ensure s is positive
  s_pos : 0 < s

/-- The measure of arc MBM' in degrees -/
def arcMeasure (t : RightIsoscelesWithCircle) : ℝ := 180

/-- Theorem: The arc MBM' always measures 180° -/
theorem arc_measure_constant (t : RightIsoscelesWithCircle) :
  arcMeasure t = 180 := by
  sorry

end NUMINAMATH_CALUDE_arc_measure_constant_l1252_125296


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1252_125224

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 4*x + 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1252_125224


namespace NUMINAMATH_CALUDE_alexandra_magazines_l1252_125226

/-- Alexandra's magazine problem -/
theorem alexandra_magazines :
  let friday_magazines : ℕ := 15
  let saturday_magazines : ℕ := 20
  let sunday_magazines : ℕ := 4 * friday_magazines
  let chewed_magazines : ℕ := 8
  let total_magazines : ℕ := friday_magazines + saturday_magazines + sunday_magazines - chewed_magazines
  total_magazines = 87 := by sorry

end NUMINAMATH_CALUDE_alexandra_magazines_l1252_125226


namespace NUMINAMATH_CALUDE_part1_part2_part3_l1252_125249

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + m - 1

-- Part 1
theorem part1 (m : ℝ) : (∀ x, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

-- Part 2
theorem part2 (m : ℝ) :
  (∀ x, f m x ≥ (m + 1) * x) ↔
  (m = -1 ∧ ∀ x, x ≥ 1) ∨
  (m > -1 ∧ ∀ x, x ≤ (m - 1) / (m + 1) ∨ x ≥ 1) ∨
  (m < -1 ∧ ∀ x, 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) :=
sorry

-- Part 3
theorem part3 (m : ℝ) : (∀ x ∈ Set.Icc (-1/2) (1/2), f m x ≥ 0) ↔ m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l1252_125249


namespace NUMINAMATH_CALUDE_cricket_team_age_theorem_l1252_125251

def cricket_team_age_problem (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) 
  (remaining_players_age_diff : ℕ) (bowlers_count : ℕ) 
  (bowlers_min_age : ℕ) (bowlers_max_age : ℕ) : Prop :=
  let wicket_keeper_age := captain_age + wicket_keeper_age_diff
  let total_age := team_size * 30
  let captain_wicket_keeper_age := captain_age + wicket_keeper_age
  let remaining_players := team_size - 2
  total_age = captain_wicket_keeper_age + remaining_players * (30 - remaining_players_age_diff) ∧
  bowlers_min_age * bowlers_count ≤ bowlers_count * 30 ∧
  bowlers_count * 30 ≤ bowlers_max_age * bowlers_count

theorem cricket_team_age_theorem : 
  cricket_team_age_problem 11 24 3 1 5 18 22 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_theorem_l1252_125251


namespace NUMINAMATH_CALUDE_trig_expression_equals_half_l1252_125254

theorem trig_expression_equals_half : 
  Real.sin (π / 3) - Real.sqrt 3 * Real.cos (π / 3) + (1 / 2) * Real.tan (π / 4) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_half_l1252_125254


namespace NUMINAMATH_CALUDE_combined_tennis_percentage_l1252_125230

theorem combined_tennis_percentage :
  let north_students : ℕ := 1800
  let south_students : ℕ := 2200
  let north_tennis_percentage : ℚ := 25 / 100
  let south_tennis_percentage : ℚ := 35 / 100
  let total_students := north_students + south_students
  let north_tennis_students := (north_students : ℚ) * north_tennis_percentage
  let south_tennis_students := (south_students : ℚ) * south_tennis_percentage
  let total_tennis_students := north_tennis_students + south_tennis_students
  let combined_percentage := total_tennis_students / (total_students : ℚ) * 100
  ⌊combined_percentage⌋ = 31 := by sorry

end NUMINAMATH_CALUDE_combined_tennis_percentage_l1252_125230


namespace NUMINAMATH_CALUDE_home_run_difference_l1252_125232

theorem home_run_difference (aaron_hr winfield_hr : ℕ) : 
  aaron_hr = 755 → winfield_hr = 465 → 2 * winfield_hr - aaron_hr = 175 := by
  sorry

end NUMINAMATH_CALUDE_home_run_difference_l1252_125232


namespace NUMINAMATH_CALUDE_speech_competition_probability_l1252_125215

theorem speech_competition_probability (n : ℕ) (h : n = 5) : 
  let total_arrangements := n.factorial
  let favorable_arrangements := (n - 1).factorial
  let prob_A_before_B := (total_arrangements / 2 : ℚ) / total_arrangements
  let prob_adjacent_and_A_before_B := (favorable_arrangements : ℚ) / total_arrangements
  (prob_adjacent_and_A_before_B / prob_A_before_B) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_speech_competition_probability_l1252_125215


namespace NUMINAMATH_CALUDE_result_of_operation_l1252_125200

theorem result_of_operation (n : ℕ) (h : n = 95) : (n / 5 + 23 : ℚ) = 42 := by
  sorry

end NUMINAMATH_CALUDE_result_of_operation_l1252_125200


namespace NUMINAMATH_CALUDE_nancy_garden_seeds_l1252_125292

theorem nancy_garden_seeds (total_seeds : ℕ) (big_garden_seeds : ℕ) (small_gardens : ℕ) 
  (h1 : total_seeds = 52)
  (h2 : big_garden_seeds = 28)
  (h3 : small_gardens = 6)
  (h4 : big_garden_seeds ≤ total_seeds) :
  (total_seeds - big_garden_seeds) / small_gardens = 4 := by
  sorry

end NUMINAMATH_CALUDE_nancy_garden_seeds_l1252_125292


namespace NUMINAMATH_CALUDE_zero_function_inequality_l1252_125237

theorem zero_function_inequality (f : ℝ → ℝ) :
  (∀ (x y : ℝ), x ≠ 0 → f (x^2 + y) ≥ (1/x + 1) * f y) →
  ∀ x, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_function_inequality_l1252_125237


namespace NUMINAMATH_CALUDE_max_profit_at_110_unique_max_profit_at_110_l1252_125260

/-- Represents the profit function for a new energy company -/
def profit (x : ℕ+) : ℚ :=
  if x < 100 then
    -1/2 * x^2 + 90 * x - 600
  else
    -2 * x - 24200 / x + 4100

/-- Theorem stating the maximum profit occurs at x = 110 -/
theorem max_profit_at_110 :
  ∀ x : ℕ+, profit x ≤ profit 110 ∧ profit 110 = 3660 := by
  sorry

/-- Theorem stating that 110 is the unique maximizer of the profit function -/
theorem unique_max_profit_at_110 :
  ∀ x : ℕ+, x ≠ 110 → profit x < profit 110 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_110_unique_max_profit_at_110_l1252_125260


namespace NUMINAMATH_CALUDE_square_difference_l1252_125276

theorem square_difference (x a : ℝ) : (2*x + a)^2 - (2*x - a)^2 = 8*a*x := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1252_125276


namespace NUMINAMATH_CALUDE_trouser_cost_calculation_final_cost_is_correct_l1252_125288

/-- Calculate the final cost in GBP for three trousers with given prices, discounts, taxes, and fees -/
theorem trouser_cost_calculation (price1 price2 price3 : ℝ) 
  (discount1 discount2 discount3 : ℝ) (global_discount : ℝ) 
  (sales_tax handling_fee conversion_rate : ℝ) : ℝ :=
  let discounted_price1 := price1 * (1 - discount1)
  let discounted_price2 := price2 * (1 - discount2)
  let discounted_price3 := price3 * (1 - discount3)
  let total_discounted := discounted_price1 + discounted_price2 + discounted_price3
  let after_global_discount := total_discounted * (1 - global_discount)
  let after_tax := after_global_discount * (1 + sales_tax)
  let final_usd := after_tax + 3 * handling_fee
  let final_gbp := final_usd * conversion_rate
  final_gbp

/-- The final cost in GBP for the given trouser prices and conditions is £271.87 -/
theorem final_cost_is_correct : 
  trouser_cost_calculation 100 150 200 0.20 0.15 0.25 0.10 0.08 5 0.75 = 271.87 := by
  sorry


end NUMINAMATH_CALUDE_trouser_cost_calculation_final_cost_is_correct_l1252_125288


namespace NUMINAMATH_CALUDE_y_value_at_243_l1252_125282

-- Define the function y in terms of k and x
def y (k : ℝ) (x : ℝ) : ℝ := k * x^(1/5)

-- State the theorem
theorem y_value_at_243 (k : ℝ) :
  y k 32 = 4 → y k 243 = 6 := by
  sorry

end NUMINAMATH_CALUDE_y_value_at_243_l1252_125282


namespace NUMINAMATH_CALUDE_complex_star_angle_sum_l1252_125297

/-- An n-pointed complex star is formed from a regular n-gon by extending every third side --/
structure ComplexStar where
  n : ℕ
  is_even : Even n
  n_ge_6 : n ≥ 6

/-- The sum of interior angles at the n intersections of a complex star --/
def interior_angle_sum (star : ComplexStar) : ℝ :=
  180 * (star.n - 6)

/-- Theorem: The sum of interior angles at the n intersections of a complex star is 180° * (n-6) --/
theorem complex_star_angle_sum (star : ComplexStar) :
  interior_angle_sum star = 180 * (star.n - 6) := by
  sorry

end NUMINAMATH_CALUDE_complex_star_angle_sum_l1252_125297


namespace NUMINAMATH_CALUDE_rectangle_to_squares_l1252_125271

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ

/-- Function to cut a rectangle in half across its length -/
def cutRectangleInHalf (r : Rectangle) : Square :=
  { side := r.width }

theorem rectangle_to_squares (r : Rectangle) 
  (h1 : r.length = 10)
  (h2 : r.width = 5) :
  (cutRectangleInHalf r).side = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_squares_l1252_125271


namespace NUMINAMATH_CALUDE_car_speed_adjustment_l1252_125201

/-- Given a car traveling a fixed distance D at 2 mph for T hours,
    prove that to cover the same distance in 5.0 hours, its speed S should be (2T)/5 mph. -/
theorem car_speed_adjustment (T : ℝ) (h : T > 0) : 
  let D := 2 * T  -- Distance covered at 2 mph for T hours
  let S := (2 * T) / 5  -- New speed to cover the same distance in 5 hours
  D = S * 5 := by sorry

end NUMINAMATH_CALUDE_car_speed_adjustment_l1252_125201


namespace NUMINAMATH_CALUDE_brother_age_twice_sister_l1252_125294

def brother_age_2005 : ℕ := 16
def sister_age_2005 : ℕ := 10
def reference_year : ℕ := 2005

theorem brother_age_twice_sister : 
  ∃ (year : ℕ), year = reference_year - (brother_age_2005 - 2 * sister_age_2005) ∧ year = 2001 :=
sorry

end NUMINAMATH_CALUDE_brother_age_twice_sister_l1252_125294


namespace NUMINAMATH_CALUDE_crayons_left_l1252_125250

theorem crayons_left (initial_crayons : ℕ) (percentage_lost : ℚ) : 
  initial_crayons = 253 → 
  percentage_lost = 35.5 / 100 →
  ↑⌊initial_crayons - percentage_lost * initial_crayons⌋ = 163 := by
  sorry

end NUMINAMATH_CALUDE_crayons_left_l1252_125250


namespace NUMINAMATH_CALUDE_vector_b_solution_l1252_125241

def a : ℝ × ℝ := (1, -2)

theorem vector_b_solution (b : ℝ × ℝ) :
  (b.1^2 + b.2^2 = 20) →  -- |b| = 2√5
  (a.1 * b.2 = a.2 * b.1) →  -- a ∥ b
  (b = (2, -4) ∨ b = (-2, 4)) :=
by sorry

end NUMINAMATH_CALUDE_vector_b_solution_l1252_125241


namespace NUMINAMATH_CALUDE_lance_cents_l1252_125223

/-- Represents the amount of cents each person has -/
structure Cents where
  lance : ℕ
  margaret : ℕ
  guy : ℕ
  bill : ℕ

/-- The problem statement -/
theorem lance_cents (c : Cents) : 
  c.margaret = 75 → -- Margaret has three-fourths of a dollar (75 cents)
  c.guy = 60 → -- Guy has two quarters (50 cents) and a dime (10 cents)
  c.bill = 60 → -- Bill has six dimes (6 * 10 cents)
  c.lance + c.margaret + c.guy + c.bill = 265 → -- Total combined cents
  c.lance = 70 := by
  sorry


end NUMINAMATH_CALUDE_lance_cents_l1252_125223


namespace NUMINAMATH_CALUDE_part_one_part_two_l1252_125235

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := -x^2 + 4*a*x - 3*a^2

-- Define the set q
def q : Set ℝ := {x | -x^2 + 11*x - 18 ≥ 0}

-- Part 1
theorem part_one : 
  {x : ℝ | f x 1 > 0} ∩ q = Set.Icc 2 3 := by sorry

-- Part 2
theorem part_two : 
  {a : ℝ | a > 0 ∧ ∀ x, f x a > 0 → x ∈ Set.Ioo 2 9} = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1252_125235


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_l1252_125236

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem not_all_squares_congruent :
  ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry


end NUMINAMATH_CALUDE_not_all_squares_congruent_l1252_125236


namespace NUMINAMATH_CALUDE_difference_of_y_coordinates_is_two_l1252_125272

noncomputable def e : ℝ := Real.exp 1

theorem difference_of_y_coordinates_is_two :
  ∀ a b : ℝ,
  (a^2 + e^4 = 2 * e^2 * a + 1) →
  (b^2 + e^4 = 2 * e^2 * b + 1) →
  a ≠ b →
  |a - b| = 2 := by
sorry

end NUMINAMATH_CALUDE_difference_of_y_coordinates_is_two_l1252_125272


namespace NUMINAMATH_CALUDE_random_sampling_cannot_prove_inequality_l1252_125274

-- Define the type for inequality proof methods
inductive InequalityProofMethod
  | Comparison
  | Synthetic
  | Analytic
  | Contradiction
  | Scaling
  | RandomSampling

-- Define a predicate for methods that can prove inequalities
def can_prove_inequality (method : InequalityProofMethod) : Prop :=
  match method with
  | InequalityProofMethod.Comparison => True
  | InequalityProofMethod.Synthetic => True
  | InequalityProofMethod.Analytic => True
  | InequalityProofMethod.Contradiction => True
  | InequalityProofMethod.Scaling => True
  | InequalityProofMethod.RandomSampling => False

-- Define random sampling as a sampling method
def is_sampling_method (method : InequalityProofMethod) : Prop :=
  method = InequalityProofMethod.RandomSampling

-- Theorem stating that random sampling cannot be used to prove inequalities
theorem random_sampling_cannot_prove_inequality :
  ∀ (method : InequalityProofMethod),
    is_sampling_method method → ¬(can_prove_inequality method) :=
by sorry

end NUMINAMATH_CALUDE_random_sampling_cannot_prove_inequality_l1252_125274


namespace NUMINAMATH_CALUDE_white_marbles_count_l1252_125277

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (white : ℕ) 
  (h1 : total = 20)
  (h2 : blue = 5)
  (h3 : red = 7)
  (h4 : total = blue + red + white)
  (h5 : (red + white : ℚ) / total = 3/4) : 
  white = 8 := by
sorry

end NUMINAMATH_CALUDE_white_marbles_count_l1252_125277


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1252_125279

theorem circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, (x - 2*a)^2 + (y - (a + 3))^2 = 4 ∧ x^2 + y^2 = 1) →
  -6/5 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1252_125279


namespace NUMINAMATH_CALUDE_alan_market_spend_l1252_125283

/-- The total amount spent by Alan at the market -/
def total_spent (num_eggs : ℕ) (price_per_egg : ℕ) (num_chickens : ℕ) (price_per_chicken : ℕ) : ℕ :=
  num_eggs * price_per_egg + num_chickens * price_per_chicken

/-- Theorem: Alan spent $88 at the market -/
theorem alan_market_spend :
  total_spent 20 2 6 8 = 88 := by
  sorry

end NUMINAMATH_CALUDE_alan_market_spend_l1252_125283


namespace NUMINAMATH_CALUDE_triangle_similarity_l1252_125273

/-- Given five complex numbers a, b, c, u, v representing points on a plane,
    if the ratios (v-a)/(u-a), (u-v)/(b-v), and (c-u)/(v-u) are equal,
    then the ratio (v-a)/(u-a) is equal to (c-a)/(b-a). -/
theorem triangle_similarity (a b c u v : ℂ) :
  (v - a) / (u - a) = (u - v) / (b - v) ∧
  (v - a) / (u - a) = (c - u) / (v - u) →
  (v - a) / (u - a) = (c - a) / (b - a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l1252_125273


namespace NUMINAMATH_CALUDE_cos_eight_arccos_one_fifth_l1252_125220

theorem cos_eight_arccos_one_fifth :
  Real.cos (8 * Real.arccos (1/5)) = -15647/390625 := by
  sorry

end NUMINAMATH_CALUDE_cos_eight_arccos_one_fifth_l1252_125220


namespace NUMINAMATH_CALUDE_smallest_coin_count_fifty_seven_satisfies_conditions_smallest_coin_count_is_57_l1252_125213

theorem smallest_coin_count (n : ℕ) : 
  (n % 5 = 2) ∧ (n % 4 = 1) ∧ (n % 3 = 0) → n ≥ 57 :=
by
  sorry

theorem fifty_seven_satisfies_conditions : 
  (57 % 5 = 2) ∧ (57 % 4 = 1) ∧ (57 % 3 = 0) :=
by
  sorry

theorem smallest_coin_count_is_57 : 
  ∃ (n : ℕ), (n % 5 = 2) ∧ (n % 4 = 1) ∧ (n % 3 = 0) ∧ 
  (∀ (m : ℕ), (m % 5 = 2) ∧ (m % 4 = 1) ∧ (m % 3 = 0) → m ≥ n) ∧
  n = 57 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_coin_count_fifty_seven_satisfies_conditions_smallest_coin_count_is_57_l1252_125213


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1252_125240

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 2 ∧ 
  (13 ∣ (964807 - x)) ∧ 
  ∀ (y : ℕ), y < x → ¬(13 ∣ (964807 - y)) :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1252_125240


namespace NUMINAMATH_CALUDE_perfect_square_mod_three_l1252_125267

theorem perfect_square_mod_three (n : ℤ) : (n^2) % 3 = 0 ∨ (n^2) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_mod_three_l1252_125267


namespace NUMINAMATH_CALUDE_distance_between_vertices_l1252_125229

/-- The equation of the graph -/
def graph_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 1| = 3

/-- The first parabola equation when y ≥ 1 -/
def parabola1 (x y : ℝ) : Prop :=
  y = -1/8 * x^2 + 2

/-- The second parabola equation when y < 1 -/
def parabola2 (x y : ℝ) : Prop :=
  y = 1/4 * x^2 - 1

/-- The vertex of the first parabola -/
def vertex1 : ℝ × ℝ := (0, 2)

/-- The vertex of the second parabola -/
def vertex2 : ℝ × ℝ := (0, -1)

theorem distance_between_vertices :
  |vertex1.2 - vertex2.2| = 3 :=
sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l1252_125229


namespace NUMINAMATH_CALUDE_hexagon_ABCDEF_perimeter_l1252_125252

def hexagon_perimeter (AB BC CD DE EF AF : ℝ) : ℝ :=
  AB + BC + CD + DE + EF + AF

theorem hexagon_ABCDEF_perimeter :
  ∀ (AB BC CD DE EF AF : ℝ),
    AB = 1 → BC = 1 → CD = 1 → DE = 1 → EF = 1 → AF = Real.sqrt 5 →
    hexagon_perimeter AB BC CD DE EF AF = 5 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_ABCDEF_perimeter_l1252_125252


namespace NUMINAMATH_CALUDE_exists_same_color_neighbors_l1252_125287

/-- Represents a color in the grid -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- Represents a position in the grid -/
structure Position where
  x : Fin 50
  y : Fin 50

/-- Represents the coloring of the grid -/
def GridColoring := Position → Color

/-- Checks if a position is valid in the 50x50 grid -/
def isValidPosition (p : Position) : Prop :=
  p.x < 50 ∧ p.y < 50

/-- Gets the color of a cell at a given position -/
def getColor (g : GridColoring) (p : Position) : Color :=
  g p

/-- Checks if a cell has the same color as its four adjacent cells -/
def hasSameColorNeighbors (g : GridColoring) (p : Position) : Prop :=
  isValidPosition p ∧
  isValidPosition ⟨p.x - 1, p.y⟩ ∧
  isValidPosition ⟨p.x + 1, p.y⟩ ∧
  isValidPosition ⟨p.x, p.y - 1⟩ ∧
  isValidPosition ⟨p.x, p.y + 1⟩ ∧
  getColor g p = getColor g ⟨p.x - 1, p.y⟩ ∧
  getColor g p = getColor g ⟨p.x + 1, p.y⟩ ∧
  getColor g p = getColor g ⟨p.x, p.y - 1⟩ ∧
  getColor g p = getColor g ⟨p.x, p.y + 1⟩

/-- Theorem: There exists a cell with four cells on its sides of the same color -/
theorem exists_same_color_neighbors :
  ∀ (g : GridColoring), ∃ (p : Position), hasSameColorNeighbors g p :=
by sorry

end NUMINAMATH_CALUDE_exists_same_color_neighbors_l1252_125287


namespace NUMINAMATH_CALUDE_slope_tangent_ln_at_3_l1252_125295

/-- The slope of the tangent line to y = ln x at x = 3 is 1/3 -/
theorem slope_tangent_ln_at_3 : 
  let f : ℝ → ℝ := λ x => Real.log x
  HasDerivAt f (1/3) 3 := by sorry

end NUMINAMATH_CALUDE_slope_tangent_ln_at_3_l1252_125295


namespace NUMINAMATH_CALUDE_correct_calculation_l1252_125221

theorem correct_calculation (x : ℝ) : x / 12 = 8 → x * 12 = 1152 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1252_125221


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1252_125207

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h1 : ∀ x, f x + (deriv f) x > 1) (h2 : f 0 = 4) :
  {x : ℝ | f x > 3 / Real.exp x + 1} = {x : ℝ | x > 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1252_125207


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1252_125245

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = (a^2 - 3*a + 2) + Complex.I * (a - 1)) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1252_125245


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l1252_125259

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l1252_125259


namespace NUMINAMATH_CALUDE_train_crossing_time_l1252_125210

-- Define the given values
def train_length : Real := 210  -- meters
def train_speed : Real := 25    -- km/h
def man_speed : Real := 2       -- km/h

-- Define the theorem
theorem train_crossing_time :
  let relative_speed : Real := train_speed + man_speed
  let relative_speed_mps : Real := relative_speed * 1000 / 3600
  let time : Real := train_length / relative_speed_mps
  time = 28 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l1252_125210


namespace NUMINAMATH_CALUDE_g_value_at_pi_over_4_l1252_125285

open Real

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - Real.sqrt 3 * (sin x)^2

noncomputable def g (x : ℝ) : ℝ := f (x - π / 12) + Real.sqrt 3 / 2

theorem g_value_at_pi_over_4 : g (π / 4) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_pi_over_4_l1252_125285


namespace NUMINAMATH_CALUDE_incompatible_food_probability_l1252_125262

-- Define the set of foods
def Food : Type := Fin 5

-- Define the incompatibility relation
def incompatible : Food → Food → Prop := sorry

-- Define the number of incompatible pairs
def num_incompatible_pairs : ℕ := 3

-- Define the total number of possible pairs
def total_pairs : ℕ := Nat.choose 5 2

-- State the theorem
theorem incompatible_food_probability :
  (num_incompatible_pairs : ℚ) / (total_pairs : ℚ) = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_incompatible_food_probability_l1252_125262


namespace NUMINAMATH_CALUDE_sin_difference_quotient_zero_l1252_125202

theorem sin_difference_quotient_zero (x y : ℝ) 
  (hx : Real.tan x = x) 
  (hy : Real.tan y = y) 
  (hxy : |x| ≠ |y|) : 
  (Real.sin (x + y)) / (x + y) - (Real.sin (x - y)) / (x - y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_quotient_zero_l1252_125202


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slope_condition_l1252_125211

/-- The slope of a line intersecting an ellipse satisfies a certain condition -/
theorem line_ellipse_intersection_slope_condition 
  (m : ℝ) -- slope of the line
  (h : ∃ (x y : ℝ), y = m * x + 10 ∧ 4 * x^2 + 25 * y^2 = 100) -- line intersects ellipse
  : m^2 ≥ 1/624 := by
  sorry

#check line_ellipse_intersection_slope_condition

end NUMINAMATH_CALUDE_line_ellipse_intersection_slope_condition_l1252_125211


namespace NUMINAMATH_CALUDE_existence_of_pairs_l1252_125269

theorem existence_of_pairs : ∃ (f : Fin 2018 → ℕ × ℕ),
  (∀ i : Fin 2018, (f i).1 ≠ (f i).2) ∧
  (∀ i : Fin 2017, (f i.succ).1 = (f i).1 + 1) ∧
  (∀ i : Fin 2017, (f i.succ).2 = (f i).2 + 1) ∧
  (∀ i : Fin 2018, (f i).1 % (f i).2 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_pairs_l1252_125269


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l1252_125253

theorem sqrt_expression_equality : 
  Real.sqrt 6 * (Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6) - abs (3 * Real.sqrt 2 - 6) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l1252_125253


namespace NUMINAMATH_CALUDE_salesman_profit_l1252_125242

/-- Calculates the profit of a salesman selling backpacks --/
theorem salesman_profit (total_backpacks : ℕ) (case_cost : ℕ)
  (swap_meet_qty : ℕ) (swap_meet_price : ℕ)
  (dept_store_qty : ℕ) (dept_store_price : ℕ)
  (online_qty : ℕ) (online_price : ℕ)
  (online_shipping : ℕ) (local_market_price : ℕ) :
  total_backpacks = 72 →
  case_cost = 1080 →
  swap_meet_qty = 25 →
  swap_meet_price = 20 →
  dept_store_qty = 18 →
  dept_store_price = 30 →
  online_qty = 12 →
  online_price = 28 →
  online_shipping = 40 →
  local_market_price = 24 →
  (swap_meet_qty * swap_meet_price +
   dept_store_qty * dept_store_price +
   online_qty * online_price - online_shipping +
   (total_backpacks - swap_meet_qty - dept_store_qty - online_qty) * local_market_price) -
  case_cost = 664 :=
by sorry

end NUMINAMATH_CALUDE_salesman_profit_l1252_125242


namespace NUMINAMATH_CALUDE_class_average_l1252_125291

theorem class_average (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ)
  (group1_average : ℚ) (group2_average : ℚ) :
  total_students = 40 →
  group1_students = 28 →
  group2_students = 12 →
  group1_average = 68 / 100 →
  group2_average = 77 / 100 →
  let total_score := group1_students * group1_average + group2_students * group2_average
  let class_average := total_score / total_students
  class_average = 707 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l1252_125291


namespace NUMINAMATH_CALUDE_zucchini_weight_l1252_125256

/-- Proves that the weight of zucchini installed is 13 kg -/
theorem zucchini_weight (carrots broccoli half_sold : ℝ) (h1 : carrots = 15) (h2 : broccoli = 8) (h3 : half_sold = 18) :
  ∃ zucchini : ℝ, (carrots + zucchini + broccoli) / 2 = half_sold ∧ zucchini = 13 := by
  sorry

end NUMINAMATH_CALUDE_zucchini_weight_l1252_125256


namespace NUMINAMATH_CALUDE_hiking_problem_l1252_125247

/-- Hiking Problem -/
theorem hiking_problem (up_rate : ℝ) (up_time : ℝ) (down_dist : ℝ) (rate_ratio : ℝ) :
  up_time = 2 →
  down_dist = 18 →
  rate_ratio = 1.5 →
  up_rate * up_time = down_dist / rate_ratio →
  up_rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_hiking_problem_l1252_125247


namespace NUMINAMATH_CALUDE_abigail_score_l1252_125261

theorem abigail_score (n : ℕ) (initial_avg final_avg : ℚ) (abigail_score : ℚ) :
  n = 20 →
  initial_avg = 85 →
  final_avg = 86 →
  (n : ℚ) * initial_avg + abigail_score = (n + 1 : ℚ) * final_avg →
  abigail_score = 106 :=
by sorry

end NUMINAMATH_CALUDE_abigail_score_l1252_125261


namespace NUMINAMATH_CALUDE_pages_used_l1252_125233

def cards_per_page : ℕ := 3
def new_cards : ℕ := 8
def old_cards : ℕ := 16

theorem pages_used (total_cards : ℕ) (h : total_cards = new_cards + old_cards) :
  total_cards / cards_per_page = 8 :=
sorry

end NUMINAMATH_CALUDE_pages_used_l1252_125233


namespace NUMINAMATH_CALUDE_parallelogram_height_l1252_125290

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) 
    (h1 : area = 384) 
    (h2 : base = 24) 
    (h3 : area = base * height) : height = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1252_125290


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l1252_125209

theorem largest_stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 1520)
  (h2 : book2 = 1900)
  (h3 : book3 = 2280) :
  Nat.gcd book1 (Nat.gcd book2 book3) = 380 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l1252_125209


namespace NUMINAMATH_CALUDE_factorization_problem_l1252_125216

theorem factorization_problem (A B : ℤ) :
  (∀ x : ℝ, 10 * x^2 - 31 * x + 21 = (A * x - 7) * (B * x - 3)) →
  A * B + A = 15 := by
sorry

end NUMINAMATH_CALUDE_factorization_problem_l1252_125216


namespace NUMINAMATH_CALUDE_carnival_snack_booth_sales_ratio_l1252_125270

-- Define the constants from the problem
def daily_popcorn_sales : ℚ := 50
def num_days : ℕ := 5
def rent : ℚ := 30
def ingredient_cost : ℚ := 75
def total_earnings : ℚ := 895

-- Define the theorem
theorem carnival_snack_booth_sales_ratio :
  ∃ (daily_cotton_candy_sales : ℚ),
    (daily_cotton_candy_sales * num_days + daily_popcorn_sales * num_days - (rent + ingredient_cost) = total_earnings) ∧
    (daily_cotton_candy_sales / daily_popcorn_sales = 3 / 1) := by
  sorry

end NUMINAMATH_CALUDE_carnival_snack_booth_sales_ratio_l1252_125270


namespace NUMINAMATH_CALUDE_min_nSn_l1252_125257

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_def : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2
  S_10 : S 10 = 0
  S_15 : S 15 = 25

/-- The main theorem -/
theorem min_nSn (seq : ArithmeticSequence) : 
  (∃ n : ℕ, n > 0 ∧ n * seq.S n = -49) ∧ 
  (∀ m : ℕ, m > 0 → m * seq.S m ≥ -49) := by
  sorry

end NUMINAMATH_CALUDE_min_nSn_l1252_125257


namespace NUMINAMATH_CALUDE_triangle_interior_angle_mean_l1252_125239

theorem triangle_interior_angle_mean :
  ∀ (triangle_sum : ℝ) (num_angles : ℕ),
    triangle_sum = 180 →
    num_angles = 3 →
    triangle_sum / num_angles = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_interior_angle_mean_l1252_125239


namespace NUMINAMATH_CALUDE_triangle_area_heron_l1252_125205

/-- Given a triangle ABC with sides a, b, c and area S, prove that under certain conditions, 
    the area S calculated using Heron's formula is equal to 15√7/4 -/
theorem triangle_area_heron (a b c : ℝ) (S : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 * Real.sin C = 24 * Real.sin A →
  a * (Real.sin C - Real.sin B) * (c + b) = (27 - a^2) * Real.sin A →
  S = Real.sqrt ((1/4) * (a^2 * c^2 - ((a^2 + c^2 - b^2) / 2)^2)) →
  S = 15 * Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_heron_l1252_125205


namespace NUMINAMATH_CALUDE_five_digit_number_proof_l1252_125263

theorem five_digit_number_proof (x : ℕ) : 
  x ≥ 10000 ∧ x < 100000 ∧ 10 * x + 1 = 3 * (100000 + x) → x = 42857 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_number_proof_l1252_125263


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1252_125204

theorem yellow_balls_count (total : ℕ) (red : ℕ) (prob_red : ℚ) : 
  red = 10 → prob_red = 1/3 → total - red = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1252_125204


namespace NUMINAMATH_CALUDE_total_short_trees_correct_park_short_trees_after_planting_l1252_125218

/-- Calculates the total number of short trees after planting -/
def total_short_trees (initial_short_trees planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + planted_short_trees

/-- Proves that the total number of short trees after planting is correct -/
theorem total_short_trees_correct (initial_short_trees planted_short_trees : ℕ) :
  total_short_trees initial_short_trees planted_short_trees = initial_short_trees + planted_short_trees :=
by sorry

/-- Proves that the specific case in the problem is correct -/
theorem park_short_trees_after_planting :
  total_short_trees 41 57 = 98 :=
by sorry

end NUMINAMATH_CALUDE_total_short_trees_correct_park_short_trees_after_planting_l1252_125218


namespace NUMINAMATH_CALUDE_games_purchase_l1252_125265

theorem games_purchase (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 42 → spent_amount = 10 → game_cost = 8 → 
  (initial_amount - spent_amount) / game_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_games_purchase_l1252_125265
