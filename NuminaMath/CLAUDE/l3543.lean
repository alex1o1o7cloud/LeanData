import Mathlib

namespace alyssa_total_spending_l3543_354328

/-- Calculates the total cost of Alyssa's toy shopping, including discount and tax --/
def total_cost (football_price teddy_bear_price crayons_price puzzle_price doll_price : ℚ)
  (teddy_bear_discount : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let discounted_teddy_bear := teddy_bear_price * (1 - teddy_bear_discount)
  let subtotal := football_price + discounted_teddy_bear + crayons_price + puzzle_price + doll_price
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax

/-- Theorem stating that Alyssa's total spending matches the calculated amount --/
theorem alyssa_total_spending :
  total_cost 12.99 15.35 4.65 7.85 14.50 0.15 0.08 = 57.23 :=
by sorry

end alyssa_total_spending_l3543_354328


namespace percent_of_number_l3543_354336

theorem percent_of_number (N M : ℝ) (h : M ≠ 0) : (N / M) * 100 = (100 * N) / M := by
  sorry

end percent_of_number_l3543_354336


namespace kids_staying_home_l3543_354342

theorem kids_staying_home (total_kids : ℕ) (kids_at_camp : ℕ) 
  (h1 : total_kids = 898051)
  (h2 : kids_at_camp = 629424) :
  total_kids - kids_at_camp = 268627 := by
  sorry

end kids_staying_home_l3543_354342


namespace tangent_line_reciprocal_function_l3543_354384

/-- The equation of the tangent line to y = 1/x at (1,1) is x + y - 2 = 0 -/
theorem tangent_line_reciprocal_function (x y : ℝ) : 
  (∀ t, t ≠ 0 → y = 1 / t) →  -- Condition: the curve is y = 1/x
  (x = 1 ∧ y = 1) →           -- Condition: the point of tangency is (1,1)
  x + y - 2 = 0               -- Conclusion: equation of the tangent line
  := by sorry

end tangent_line_reciprocal_function_l3543_354384


namespace book_discount_percentage_l3543_354362

theorem book_discount_percentage (marked_price : ℝ) (cost_price : ℝ) (selling_price : ℝ) :
  cost_price = 0.64 * marked_price →
  (selling_price - cost_price) / cost_price = 0.375 →
  (marked_price - selling_price) / marked_price = 0.12 := by
  sorry

end book_discount_percentage_l3543_354362


namespace nested_fraction_equality_l3543_354357

theorem nested_fraction_equality : 
  (1 : ℝ) / (2 - 1 / (2 - 1 / (2 - 1 / 3))) = 5 / 7 := by sorry

end nested_fraction_equality_l3543_354357


namespace total_trade_scientific_notation_l3543_354387

/-- Represents the total bilateral trade in goods in yuan -/
def total_trade : ℝ := 1653 * 1000000000

/-- Represents the scientific notation of the total trade -/
def scientific_notation : ℝ := 1.6553 * (10 ^ 12)

/-- Theorem stating that the total trade is equal to its scientific notation representation -/
theorem total_trade_scientific_notation : total_trade = scientific_notation := by
  sorry

end total_trade_scientific_notation_l3543_354387


namespace dara_waiting_time_l3543_354396

/-- Represents a person's age and employment status -/
structure Person where
  age : ℕ
  employed : Bool

/-- The minimum age required for employment -/
def min_employment_age : ℕ := 25

/-- Calculates the age of a person after a given number of years -/
def age_after (p : Person) (years : ℕ) : ℕ := p.age + years

/-- Jane's current state -/
def jane : Person := { age := 28, employed := true }

/-- Dara's current age -/
def dara_age : ℕ := jane.age + 6 - 2 * (jane.age + 6 - min_employment_age)

/-- Time Dara needs to wait to reach the minimum employment age -/
def waiting_time : ℕ := min_employment_age - dara_age

theorem dara_waiting_time :
  waiting_time = 14 := by sorry

end dara_waiting_time_l3543_354396


namespace tessa_final_debt_l3543_354341

/-- Calculates the final debt after a partial repayment and additional borrowing --/
def calculateFinalDebt (initialDebt : ℚ) (repaymentFraction : ℚ) (additionalBorrowing : ℚ) : ℚ :=
  initialDebt - (repaymentFraction * initialDebt) + additionalBorrowing

/-- Proves that Tessa's final debt to Greg is $30 --/
theorem tessa_final_debt :
  calculateFinalDebt 40 (1/2) 10 = 30 := by
  sorry

end tessa_final_debt_l3543_354341


namespace intersection_of_lines_l3543_354398

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Represents a line in 2D space of the form ax + by = c -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Check if a point lies on a given line -/
def pointOnLine (p : IntersectionPoint) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

theorem intersection_of_lines (line1 line2 : Line)
  (h1 : line1 = ⟨6, -9, 18⟩)
  (h2 : line2 = ⟨8, 2, 20⟩) :
  ∃! p : IntersectionPoint, pointOnLine p line1 ∧ pointOnLine p line2 ∧ p = ⟨18/7, -2/7⟩ := by
  sorry

end intersection_of_lines_l3543_354398


namespace sally_remaining_cards_l3543_354321

def initial_cards : ℕ := 39
def cards_sold : ℕ := 24

theorem sally_remaining_cards : initial_cards - cards_sold = 15 := by
  sorry

end sally_remaining_cards_l3543_354321


namespace fraction_subtraction_l3543_354369

theorem fraction_subtraction : (7 : ℚ) / 3 - 5 / 6 = 3 / 2 := by
  sorry

end fraction_subtraction_l3543_354369


namespace divisors_of_2_pow_18_minus_1_l3543_354339

theorem divisors_of_2_pow_18_minus_1 :
  ∃! (a b : ℕ), 20 < a ∧ a < 30 ∧ 20 < b ∧ b < 30 ∧
  (2^18 - 1) % a = 0 ∧ (2^18 - 1) % b = 0 ∧ a ≠ b ∧
  a = 19 ∧ b = 27 :=
sorry

end divisors_of_2_pow_18_minus_1_l3543_354339


namespace megan_total_songs_l3543_354323

/-- The number of country albums Megan bought -/
def country_albums : ℕ := 2

/-- The number of pop albums Megan bought -/
def pop_albums : ℕ := 8

/-- The number of songs in each album -/
def songs_per_album : ℕ := 7

/-- The total number of songs Megan bought -/
def total_songs : ℕ := (country_albums + pop_albums) * songs_per_album

theorem megan_total_songs : total_songs = 70 := by
  sorry

end megan_total_songs_l3543_354323


namespace sum_of_fractions_sum_equals_14_1_l3543_354352

theorem sum_of_fractions : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (10 / 10 : ℚ) + 
  (11 / 10 : ℚ) + (15 / 10 : ℚ) + (20 / 10 : ℚ) + (25 / 10 : ℚ) + (50 / 10 : ℚ) = 
  (141 : ℚ) / 10 := by
  sorry

theorem sum_equals_14_1 : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (10 / 10 : ℚ) + 
  (11 / 10 : ℚ) + (15 / 10 : ℚ) + (20 / 10 : ℚ) + (25 / 10 : ℚ) + (50 / 10 : ℚ) = 
  14.1 := by
  sorry

end sum_of_fractions_sum_equals_14_1_l3543_354352


namespace cost_price_calculation_l3543_354306

/-- Proves that the cost price of an article is 78.944 given the specified conditions --/
theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  marked_price = 98.68 →
  discount_rate = 0.05 →
  profit_rate = 0.25 →
  ∃ (cost_price : ℝ),
    (1 - discount_rate) * marked_price = cost_price * (1 + profit_rate) ∧
    cost_price = 78.944 := by
  sorry

end cost_price_calculation_l3543_354306


namespace work_days_calculation_l3543_354326

/-- Represents the number of days worked by each person -/
structure DaysWorked where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The problem statement -/
theorem work_days_calculation (days : DaysWorked) (wages : DailyWages) 
    (h1 : days.a = 6)
    (h2 : days.c = 4)
    (h3 : wages.a * 5 = wages.c * 3)
    (h4 : wages.b * 5 = wages.c * 4)
    (h5 : wages.c = 100)
    (h6 : days.a * wages.a + days.b * wages.b + days.c * wages.c = 1480) :
  days.b = 9 := by
  sorry

end work_days_calculation_l3543_354326


namespace babblian_word_count_l3543_354354

def alphabet_size : ℕ := 6
def max_word_length : ℕ := 3

def count_words (alphabet_size : ℕ) (max_word_length : ℕ) : ℕ :=
  (alphabet_size^1 + alphabet_size^2 + alphabet_size^3)

theorem babblian_word_count :
  count_words alphabet_size max_word_length = 258 := by
  sorry

end babblian_word_count_l3543_354354


namespace polynomial_product_expansion_l3543_354348

/-- Given two polynomials (7x^2 + 5) and (3x^3 + 2x + 1), their product is equal to 21x^5 + 29x^3 + 7x^2 + 10x + 5 -/
theorem polynomial_product_expansion (x : ℝ) : 
  (7 * x^2 + 5) * (3 * x^3 + 2 * x + 1) = 21 * x^5 + 29 * x^3 + 7 * x^2 + 10 * x + 5 := by
  sorry


end polynomial_product_expansion_l3543_354348


namespace simplify_radical_product_l3543_354353

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (18 * x) = 30 * x * Real.sqrt (2 * x) :=
by sorry

end simplify_radical_product_l3543_354353


namespace max_boxes_in_wooden_box_l3543_354355

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ :=
  m * 100

theorem max_boxes_in_wooden_box :
  let largeBox : BoxDimensions := {
    length := metersToCentimeters 8,
    width := metersToCentimeters 10,
    height := metersToCentimeters 6
  }
  let smallBox : BoxDimensions := {
    length := 4,
    width := 5,
    height := 6
  }
  (boxVolume largeBox) / (boxVolume smallBox) = 4000000 := by
  sorry

end max_boxes_in_wooden_box_l3543_354355


namespace slower_truck_speed_calculation_l3543_354338

-- Define the length of each truck
def truck_length : ℝ := 250

-- Define the speed of the faster truck
def faster_truck_speed : ℝ := 30

-- Define the time taken for the slower truck to pass the faster one
def passing_time : ℝ := 35.997120230381576

-- Define the speed of the slower truck
def slower_truck_speed : ℝ := 20

-- Theorem statement
theorem slower_truck_speed_calculation :
  let total_length := 2 * truck_length
  let faster_speed_ms := faster_truck_speed * (1000 / 3600)
  let slower_speed_ms := slower_truck_speed * (1000 / 3600)
  let relative_speed := faster_speed_ms + slower_speed_ms
  total_length = relative_speed * passing_time :=
by sorry

end slower_truck_speed_calculation_l3543_354338


namespace imaginary_cube_l3543_354307

theorem imaginary_cube (i : ℂ) : i^2 = -1 → 1 + i^3 = 1 - i := by
  sorry

end imaginary_cube_l3543_354307


namespace inscribed_tetrahedron_volume_l3543_354374

/-- Represents a tetrahedron with a triangular base and square lateral faces -/
structure Tetrahedron where
  base_side_length : ℝ
  has_square_lateral_faces : Bool

/-- Represents a tetrahedron inscribed within another tetrahedron -/
structure InscribedTetrahedron where
  outer : Tetrahedron
  vertices_touch_midpoints : Bool
  base_parallel : Bool

/-- Calculates the volume of an inscribed tetrahedron -/
def volume_inscribed_tetrahedron (t : InscribedTetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the inscribed tetrahedron -/
theorem inscribed_tetrahedron_volume 
  (t : InscribedTetrahedron) 
  (h1 : t.outer.base_side_length = 2) 
  (h2 : t.outer.has_square_lateral_faces = true)
  (h3 : t.vertices_touch_midpoints = true)
  (h4 : t.base_parallel = true) : 
  volume_inscribed_tetrahedron t = Real.sqrt 2 / 12 := by sorry

end inscribed_tetrahedron_volume_l3543_354374


namespace arithmetic_operations_l3543_354377

theorem arithmetic_operations : 
  (400 / 5 = 80) ∧ (3 * 230 = 690) := by sorry

end arithmetic_operations_l3543_354377


namespace gcf_lcm_sum_8_12_l3543_354329

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcf_lcm_sum_8_12_l3543_354329


namespace intersection_A_complement_B_when_m_3_m_value_when_intersection_equals_given_set_l3543_354303

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x - 5 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_A_complement_B_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x : ℝ | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Part 2
theorem m_value_when_intersection_equals_given_set :
  (∃ m : ℝ, A ∩ B m = {x : ℝ | -1 ≤ x ∧ x < 4}) → 
  (∃ m : ℝ, A ∩ B m = {x : ℝ | -1 ≤ x ∧ x < 4} ∧ m = 8) := by sorry

end intersection_A_complement_B_when_m_3_m_value_when_intersection_equals_given_set_l3543_354303


namespace circle_center_transformation_l3543_354390

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given distance -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem stating the transformation of the circle's center -/
theorem circle_center_transformation :
  let original_center : ℝ × ℝ := (-3, 4)
  let reflected_center := reflect_x original_center
  let final_center := translate_right reflected_center 5
  final_center = (2, -4) := by sorry

end circle_center_transformation_l3543_354390


namespace hyperbola_intersection_l3543_354365

/-- Given a triangle AOB with A on the positive y-axis, B on the positive x-axis, and area 9,
    and a hyperbolic function y = k/x intersecting AB at C and D such that CD = 1/3 AB and AC = BD,
    prove that k = 4 -/
theorem hyperbola_intersection (y_A x_B : ℝ) (k : ℝ) : 
  y_A > 0 → x_B > 0 → -- A and B are on positive axes
  1/2 * x_B * y_A = 9 → -- Area of triangle AOB is 9
  ∃ (x_C y_C : ℝ), -- C exists on the line AB and the hyperbola
    0 < x_C ∧ x_C < x_B ∧
    y_C = (y_A / x_B) * (x_B - x_C) ∧ -- C is on line AB
    y_C = k / x_C ∧ -- C is on the hyperbola
    x_C = 1/3 * x_B ∧ -- C is a trisection point
    y_C = 2/3 * y_A → -- C is a trisection point
  k = 4 := by
sorry

end hyperbola_intersection_l3543_354365


namespace quiz_min_correct_answers_l3543_354301

theorem quiz_min_correct_answers 
  (total_questions : ℕ) 
  (points_correct : ℕ) 
  (points_incorrect : ℕ) 
  (target_score : ℕ) 
  (min_correct : ℕ) :
  total_questions = 20 →
  points_correct = 10 →
  points_incorrect = 4 →
  target_score = 88 →
  min_correct = 12 →
  (∀ x : ℕ, x ≥ min_correct ↔ 
    points_correct * x - points_incorrect * (total_questions - x) ≥ target_score) :=
by sorry

end quiz_min_correct_answers_l3543_354301


namespace max_probability_at_20_red_balls_l3543_354386

/-- The probability of winning in one draw -/
def p (n : ℕ) : ℚ := 10 * n / ((n + 5) * (n + 4))

/-- The probability of winning exactly once in three draws -/
def P (n : ℕ) : ℚ := 3 * p n * (1 - p n)^2

theorem max_probability_at_20_red_balls (n : ℕ) (h : n ≥ 5) :
  P n ≤ P 20 ∧ ∃ (m : ℕ), m ≥ 5 ∧ P m = P 20 → m = 20 :=
sorry

end max_probability_at_20_red_balls_l3543_354386


namespace min_pizzas_to_break_even_l3543_354346

def car_cost : ℕ := 6500
def net_profit_per_pizza : ℕ := 7

theorem min_pizzas_to_break_even :
  ∀ n : ℕ, (n * net_profit_per_pizza ≥ car_cost) ∧ 
           (∀ m : ℕ, m < n → m * net_profit_per_pizza < car_cost) →
  n = 929 := by
  sorry

end min_pizzas_to_break_even_l3543_354346


namespace rebeccas_salon_l3543_354322

/-- Rebecca's hair salon problem -/
theorem rebeccas_salon (haircut_price perm_price dye_job_price dye_cost : ℕ)
  (num_perms num_dye_jobs : ℕ) (tips total_revenue : ℕ) :
  haircut_price = 30 →
  perm_price = 40 →
  dye_job_price = 60 →
  dye_cost = 10 →
  num_perms = 1 →
  num_dye_jobs = 2 →
  tips = 50 →
  total_revenue = 310 →
  ∃ (num_haircuts : ℕ),
    num_haircuts * haircut_price +
    num_perms * perm_price +
    num_dye_jobs * dye_job_price -
    num_dye_jobs * dye_cost +
    tips = total_revenue ∧
    num_haircuts = 4 :=
by sorry

end rebeccas_salon_l3543_354322


namespace amount_ratio_l3543_354314

theorem amount_ratio (total : ℚ) (r_amount : ℚ) 
  (h1 : total = 9000)
  (h2 : r_amount = 3600.0000000000005) :
  r_amount / (total - r_amount) = 2 / 3 := by
sorry

end amount_ratio_l3543_354314


namespace xy_sum_inequality_l3543_354311

theorem xy_sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + x*y = 3) :
  (x + y ≥ 2) ∧ (x + y = 2 ↔ x = 1 ∧ y = 1) := by
  sorry

end xy_sum_inequality_l3543_354311


namespace system_solution_l3543_354333

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 7 * x + y = 19
def equation2 (x y : ℝ) : Prop := x + 3 * y = 1
def equation3 (x y z : ℝ) : Prop := 2 * x + y - 4 * z = 10

-- Theorem statement
theorem system_solution (x y z : ℝ) :
  equation1 x y ∧ equation2 x y ∧ equation3 x y z →
  2 * x + y + 3 * z = 1.25 := by
  sorry

end system_solution_l3543_354333


namespace fraction_comparison_l3543_354385

theorem fraction_comparison (x y : ℕ+) (h : y > x) : (x + 1 : ℚ) / (y + 1) > (x : ℚ) / y := by
  sorry

end fraction_comparison_l3543_354385


namespace parabola_vertex_l3543_354310

/-- The vertex of the parabola defined by y² + 10y + 3x + 9 = 0 is (16/3, -5) -/
theorem parabola_vertex :
  let f : ℝ → ℝ → ℝ := λ x y ↦ y^2 + 10*y + 3*x + 9
  ∃! (x₀ y₀ : ℝ), (∀ x y, f x y = 0 → y ≥ y₀) ∧ f x₀ y₀ = 0 ∧ x₀ = 16/3 ∧ y₀ = -5 :=
sorry

end parabola_vertex_l3543_354310


namespace arrangement_theorem_l3543_354364

/-- Represents the number of ways to arrange people in two rows -/
def arrangement_count (total_people : ℕ) (front_row : ℕ) (back_row : ℕ) : ℕ := sorry

/-- Represents whether two people are standing next to each other -/
def standing_next_to (person1 : ℕ) (person2 : ℕ) : Prop := sorry

/-- Represents whether two people are standing apart -/
def standing_apart (person1 : ℕ) (person2 : ℕ) : Prop := sorry

theorem arrangement_theorem :
  ∀ (total_people front_row back_row : ℕ) 
    (person_a person_b person_c : ℕ),
  total_people = 7 →
  front_row = 3 →
  back_row = 4 →
  standing_next_to person_a person_b →
  standing_apart person_a person_c →
  arrangement_count total_people front_row back_row = 1056 := by
  sorry

end arrangement_theorem_l3543_354364


namespace quadratic_inequality_range_l3543_354388

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x - m ≥ 0) → -4 ≤ m ∧ m ≤ 0 :=
by sorry

end quadratic_inequality_range_l3543_354388


namespace quadratic_solutions_parabola_vertex_l3543_354343

-- Part 1: Quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 + 4*x - 2 = 0

theorem quadratic_solutions :
  ∃ x1 x2 : ℝ, x1 = -2 + Real.sqrt 6 ∧ x2 = -2 - Real.sqrt 6 ∧
  quadratic_equation x1 ∧ quadratic_equation x2 :=
sorry

-- Part 2: Parabola vertex
def parabola (x y : ℝ) : Prop :=
  y = 2*x^2 - 4*x + 6

theorem parabola_vertex :
  ∃ x y : ℝ, x = 1 ∧ y = 4 ∧ parabola x y ∧
  ∀ x' y' : ℝ, parabola x' y' → y ≤ y' :=
sorry

end quadratic_solutions_parabola_vertex_l3543_354343


namespace pencil_count_l3543_354319

theorem pencil_count (pens pencils : ℕ) : 
  (5 * pencils = 6 * pens) → 
  (pencils = pens + 7) → 
  pencils = 42 := by sorry

end pencil_count_l3543_354319


namespace measles_cases_1990_l3543_354332

/-- Calculates the number of measles cases in a given year, assuming a linear decrease from 1970 to 2000 -/
def measlesCases (year : ℕ) : ℕ :=
  let initialYear : ℕ := 1970
  let finalYear : ℕ := 2000
  let initialCases : ℕ := 480000
  let finalCases : ℕ := 600
  let yearsPassed : ℕ := year - initialYear
  let totalYears : ℕ := finalYear - initialYear
  let totalDecrease : ℕ := initialCases - finalCases
  let yearlyDecrease : ℕ := totalDecrease / totalYears
  initialCases - (yearsPassed * yearlyDecrease)

theorem measles_cases_1990 : measlesCases 1990 = 160400 := by
  sorry

end measles_cases_1990_l3543_354332


namespace sum_of_xy_l3543_354366

theorem sum_of_xy (x y : ℕ) (hx : 0 < x ∧ x < 20) (hy : 0 < y ∧ y < 20) 
  (h_eq : x + y + x * y = 76) : x + y = 16 := by
  sorry

end sum_of_xy_l3543_354366


namespace four_lines_equal_angles_l3543_354305

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A rectangular box in 3D space -/
structure RectangularBox where
  corner : Point3D
  width : ℝ
  length : ℝ
  height : ℝ

/-- The angle between a line and an edge of the box -/
def angleWithEdge (l : Line3D) (b : RectangularBox) (edge : Fin 12) : ℝ :=
  sorry

/-- A line forms equal angles with all edges of the box -/
def formsEqualAngles (l : Line3D) (b : RectangularBox) : Prop :=
  ∀ (e1 e2 : Fin 12), angleWithEdge l b e1 = angleWithEdge l b e2

/-- The main theorem -/
theorem four_lines_equal_angles (P : Point3D) (b : RectangularBox) :
  ∃! (lines : Finset Line3D), lines.card = 4 ∧ 
    ∀ l ∈ lines, l.point = P ∧ formsEqualAngles l b :=
  sorry

end four_lines_equal_angles_l3543_354305


namespace coin_flip_frequency_l3543_354359

/-- The frequency of an event is the ratio of the number of times the event occurs to the total number of trials. -/
def frequency (occurrences : ℕ) (trials : ℕ) : ℚ :=
  occurrences / trials

/-- In an experiment of flipping a coin 100 times, the frequency of getting "heads" is 49. -/
theorem coin_flip_frequency :
  frequency 49 100 = 49/100 := by
sorry

end coin_flip_frequency_l3543_354359


namespace arbitrarily_large_solution_exists_l3543_354325

theorem arbitrarily_large_solution_exists (N : ℕ) : 
  ∃ (a b c d : ℤ), 
    (a * a + b * b + c * c + d * d = a * b * c + a * b * d + a * c * d + b * c * d) ∧ 
    (min a (min b (min c d)) ≥ N) := by
  sorry

end arbitrarily_large_solution_exists_l3543_354325


namespace nim_max_product_l3543_354375

/-- Nim-sum (bitwise XOR) of two natural numbers -/
def nim_sum (a b : ℕ) : ℕ := a ^^^ b

/-- Check if a given configuration is a losing position in 3-player Nim -/
def is_losing_position (a b c d : ℕ) : Prop :=
  nim_sum (nim_sum (nim_sum a b) c) d = 0

/-- The maximum product of x and y satisfying the game conditions -/
def max_product : ℕ := 7704

/-- The theorem stating the maximum product of x and y in the given Nim game -/
theorem nim_max_product :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧
  is_losing_position 43 99 x y ∧
  x * y = max_product ∧
  ∀ (a b : ℕ), a > 0 → b > 0 → is_losing_position 43 99 a b → a * b ≤ max_product :=
sorry

end nim_max_product_l3543_354375


namespace scooter_initial_cost_l3543_354324

/-- Proves that the initial cost of a scooter is $900 given the conditions of the problem -/
theorem scooter_initial_cost (initial_cost : ℝ) : 
  (∃ (total_cost : ℝ), 
    total_cost = initial_cost + 300 ∧ 
    1500 = 1.25 * total_cost) → 
  initial_cost = 900 :=
by sorry

end scooter_initial_cost_l3543_354324


namespace election_invalid_votes_percentage_l3543_354399

theorem election_invalid_votes_percentage 
  (total_votes : ℕ) 
  (b_votes : ℕ) 
  (h1 : total_votes = 8720)
  (h2 : b_votes = 2834)
  (h3 : ∃ (a_votes : ℕ), a_votes = b_votes + (15 * total_votes) / 100) :
  (total_votes - (b_votes + (b_votes + (15 * total_votes) / 100))) * 100 / total_votes = 20 := by
  sorry

end election_invalid_votes_percentage_l3543_354399


namespace unique_angle_sin_cos_l3543_354315

theorem unique_angle_sin_cos :
  ∃! x : ℝ, 0 ≤ x ∧ x < π / 2 ∧ Real.sin x = 0.6 ∧ Real.cos x = 0.8 := by
  sorry

end unique_angle_sin_cos_l3543_354315


namespace magnitude_product_complex_l3543_354381

theorem magnitude_product_complex : Complex.abs ((7 - 4 * Complex.I) * (3 + 10 * Complex.I)) = Real.sqrt 7085 := by
  sorry

end magnitude_product_complex_l3543_354381


namespace fraction_simplification_fraction_value_at_two_l3543_354376

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ -2) :
  ((3 * x + 4) / (x^2 - 1) - 2 / (x - 1)) / ((x + 2) / (x^2 - 2*x + 1)) = (x - 1) / (x + 1) :=
by sorry

theorem fraction_value_at_two :
  ((3 * 2 + 4) / (2^2 - 1) - 2 / (2 - 1)) / ((2 + 2) / (2^2 - 2*2 + 1)) = 1 / 3 :=
by sorry

end fraction_simplification_fraction_value_at_two_l3543_354376


namespace sum_of_squares_theorem_l3543_354394

theorem sum_of_squares_theorem (x y z a b c k : ℝ) 
  (h1 : x * y = k * a) 
  (h2 : x * z = k * b) 
  (h3 : y * z = k * c) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hk : k ≠ 0) : 
  x^2 + y^2 + z^2 = k * (a * b / c + a * c / b + b * c / a) := by
  sorry

end sum_of_squares_theorem_l3543_354394


namespace blanket_thickness_after_four_foldings_l3543_354397

/-- Represents the thickness of a blanket after a certain number of foldings -/
def blanketThickness (initialThickness : ℕ) (numFoldings : ℕ) : ℕ :=
  initialThickness * 2^numFoldings

/-- Proves that a blanket with initial thickness 3 inches will be 48 inches thick after 4 foldings -/
theorem blanket_thickness_after_four_foldings :
  blanketThickness 3 4 = 48 := by
  sorry

#eval blanketThickness 3 4

end blanket_thickness_after_four_foldings_l3543_354397


namespace arrangement_theorem_l3543_354340

/-- The number of ways to arrange n elements with k special positions --/
def special_arrangements (n : ℕ) (k : ℕ) : ℕ := k * (n - 1).factorial

/-- The number of ways to arrange n elements with two restrictions --/
def restricted_arrangements (n : ℕ) : ℕ := 
  n.factorial - 2 * (n - 1).factorial + (n - 2).factorial

theorem arrangement_theorem (total_students : ℕ) 
  (h_total : total_students = 7) :
  (special_arrangements total_students 3 = 2160) ∧
  (restricted_arrangements total_students = 3720) := by
  sorry

#eval special_arrangements 7 3
#eval restricted_arrangements 7

end arrangement_theorem_l3543_354340


namespace fraction_division_equality_l3543_354316

theorem fraction_division_equality : 
  (-1/42 : ℚ) / ((1/6 : ℚ) - (3/14 : ℚ) + (2/3 : ℚ) - (2/7 : ℚ)) = -1/14 := by
  sorry

end fraction_division_equality_l3543_354316


namespace range_of_a_l3543_354344

-- Define the propositions p and q as functions of x and a
def p (x : ℝ) : Prop := |4*x - 3| ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the condition that not p is necessary but not sufficient for not q
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, ¬(q x a) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q x a)

-- State the theorem
theorem range_of_a :
  {a : ℝ | necessary_not_sufficient a} = {a : ℝ | a < 0 ∨ a > 1} :=
sorry

end range_of_a_l3543_354344


namespace expression_evaluation_l3543_354347

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b - 8)
  (h2 : b = a + 3)
  (h3 : a = 2)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 5 ≠ 0) :
  ((a + 3) / (a + 1)) * ((b - 1) / (b - 3)) * ((c + 7) / (c + 5)) = 20 / 3 := by
  sorry

end expression_evaluation_l3543_354347


namespace mark_change_factor_l3543_354370

theorem mark_change_factor (n : ℕ) (original_avg new_avg : ℚ) (h1 : n = 10) (h2 : original_avg = 80) (h3 : new_avg = 160) :
  ∃ (factor : ℚ), factor * (n * original_avg) = n * new_avg ∧ factor = 2 := by
  sorry

end mark_change_factor_l3543_354370


namespace combined_girls_avg_is_87_l3543_354313

/-- Represents a high school with average scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined average score for boys at both schools -/
def combined_boys_avg : ℝ := 73

/-- Theorem: The combined average score for girls at both schools is 87 -/
theorem combined_girls_avg_is_87 
  (lincoln : School)
  (grant : School)
  (h1 : lincoln.boys_avg = 68)
  (h2 : lincoln.girls_avg = 80)
  (h3 : lincoln.combined_avg = 72)
  (h4 : grant.boys_avg = 75)
  (h5 : grant.girls_avg = 88)
  (h6 : grant.combined_avg = 82)
  (h7 : combined_boys_avg = 73) :
  ∃ (combined_girls_avg : ℝ), combined_girls_avg = 87 := by
  sorry


end combined_girls_avg_is_87_l3543_354313


namespace quadrilateral_formation_l3543_354300

/-- A function that checks if four line segments can form a quadrilateral --/
def can_form_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a

/-- The theorem stating which set of line segments can form a quadrilateral with length 5 --/
theorem quadrilateral_formation :
  ¬(can_form_quadrilateral 1 1 1 5) ∧
  ¬(can_form_quadrilateral 1 1 8 5) ∧
  ¬(can_form_quadrilateral 1 2 2 5) ∧
  can_form_quadrilateral 3 3 3 5 :=
sorry

end quadrilateral_formation_l3543_354300


namespace probability_20th_to_30th_l3543_354393

/-- A sequence of 40 distinct real numbers -/
def Sequence := Fin 40 → ℝ

/-- Predicate to check if a sequence contains distinct elements -/
def IsDistinct (s : Sequence) : Prop :=
  ∀ i j, i ≠ j → s i ≠ s j

/-- The probability that the 20th number ends up in the 30th position after one bubble pass -/
def ProbabilityOf20thTo30th (s : Sequence) : ℚ :=
  1 / 930

/-- Theorem stating the probability of the 20th number ending up in the 30th position -/
theorem probability_20th_to_30th (s : Sequence) (h : IsDistinct s) :
    ProbabilityOf20thTo30th s = 1 / 930 := by
  sorry

end probability_20th_to_30th_l3543_354393


namespace unique_solution_equation_l3543_354335

theorem unique_solution_equation :
  ∃! x : ℝ, 2017 * x^2017 - 2017 + x = (2018 - 2017*x)^(1/2017) :=
by sorry

end unique_solution_equation_l3543_354335


namespace difference_of_percentages_l3543_354309

theorem difference_of_percentages : 
  (75 / 100 * 480) - (3 / 5 * (20 / 100 * 2500)) = 60 := by
  sorry

end difference_of_percentages_l3543_354309


namespace combined_figure_area_l3543_354378

/-- Regular pentagon with side length 3 -/
structure RegularPentagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Square with side length 3 -/
structure Square :=
  (side_length : ℝ)
  (is_square : side_length = 3)

/-- Combined figure of a regular pentagon and a square -/
structure CombinedFigure :=
  (pentagon : RegularPentagon)
  (square : Square)
  (shared_side : pentagon.side_length = square.side_length)

/-- Area of the combined figure -/
def area (figure : CombinedFigure) : ℝ := sorry

/-- Theorem stating the area of the combined figure -/
theorem combined_figure_area (figure : CombinedFigure) :
  area figure = Real.sqrt 81 + Real.sqrt 27 := by sorry

end combined_figure_area_l3543_354378


namespace probability_ace_two_three_four_l3543_354304

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards of each rank (Ace, 2, 3, 4) in a standard deck -/
def cards_per_rank : ℕ := 4

/-- Calculates the probability of drawing a specific sequence of four cards from a standard deck -/
def probability_four_card_sequence : ℚ :=
  (cards_per_rank : ℚ) / deck_size *
  (cards_per_rank : ℚ) / (deck_size - 1) *
  (cards_per_rank : ℚ) / (deck_size - 2) *
  (cards_per_rank : ℚ) / (deck_size - 3)

/-- The probability of drawing an Ace, 2, 3, and 4 in that order from a standard deck of 52 cards, without replacement, is equal to 16/405525 -/
theorem probability_ace_two_three_four : probability_four_card_sequence = 16 / 405525 := by
  sorry

end probability_ace_two_three_four_l3543_354304


namespace max_n_is_81_l3543_354356

/-- The maximum value of n given the conditions -/
def max_n : ℕ := 81

/-- The set of numbers from 1 to 500 -/
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 500}

/-- The probability of selecting a divisor of n from S -/
def prob_divisor (n : ℕ) : ℚ := (Finset.filter (· ∣ n) (Finset.range 500)).card / 500

/-- The theorem stating that 81 is the maximum value satisfying the conditions -/
theorem max_n_is_81 :
  ∀ n : ℕ, n ∈ S → prob_divisor n = 1/100 → n ≤ max_n :=
sorry

end max_n_is_81_l3543_354356


namespace min_value_quadratic_l3543_354337

theorem min_value_quadratic (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^2 - a * x + (a + 1000) ≥ 1 + 999 / a) ∧
  (∃ x : ℝ, a * x^2 - a * x + (a + 1000) = 1 + 999 / a) :=
sorry

end min_value_quadratic_l3543_354337


namespace parabola_focus_coordinates_l3543_354320

/-- Given a quadratic function f(x) = ax^2 + bx + 2 where a ≠ 0,
    and |f(x)| ≥ 2 for all real x, prove that the coordinates of
    the focus of the parabolic curve are (0, 1/(4a) + 2). -/
theorem parabola_focus_coordinates
  (a b : ℝ) (ha : a ≠ 0)
  (hf : ∀ x : ℝ, |a * x^2 + b * x + 2| ≥ 2) :
  ∃ p : ℝ × ℝ, p.1 = 0 ∧ p.2 = 1 / (4 * a) + 2 :=
sorry

end parabola_focus_coordinates_l3543_354320


namespace equal_diagonals_bisect_implies_rectangle_all_sides_equal_implies_rhombus_perpendicular_diagonals_not_imply_rhombus_all_sides_equal_not_imply_square_l3543_354331

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def diagonals_bisect_each_other (q : Quadrilateral) : Prop := sorry
def has_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry
def has_all_sides_equal (q : Quadrilateral) : Prop := sorry

-- Define special quadrilaterals
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem equal_diagonals_bisect_implies_rectangle (q : Quadrilateral) :
  has_equal_diagonals q ∧ diagonals_bisect_each_other q → is_rectangle q :=
sorry

-- Theorem 2
theorem all_sides_equal_implies_rhombus (q : Quadrilateral) :
  has_all_sides_equal q → is_rhombus q :=
sorry

-- Theorem 3
theorem perpendicular_diagonals_not_imply_rhombus :
  ∃ q : Quadrilateral, has_perpendicular_diagonals q ∧ ¬is_rhombus q :=
sorry

-- Theorem 4
theorem all_sides_equal_not_imply_square :
  ∃ q : Quadrilateral, has_all_sides_equal q ∧ ¬is_square q :=
sorry

end equal_diagonals_bisect_implies_rectangle_all_sides_equal_implies_rhombus_perpendicular_diagonals_not_imply_rhombus_all_sides_equal_not_imply_square_l3543_354331


namespace necessary_not_sufficient_condition_l3543_354395

def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem necessary_not_sufficient_condition :
  (∀ a : ℕ → ℝ, IsIncreasing a → ∀ n, |a (n + 1)| > a n) ∧
  (∃ a : ℕ → ℝ, (∀ n, |a (n + 1)| > a n) ∧ ¬IsIncreasing a) :=
sorry

end necessary_not_sufficient_condition_l3543_354395


namespace add_negative_two_l3543_354382

theorem add_negative_two : 3 + (-2) = 1 := by
  sorry

end add_negative_two_l3543_354382


namespace socks_expense_is_eleven_l3543_354350

/-- The amount spent on socks given a budget and other expenses --/
def socks_expense (budget : ℕ) (shirt_cost pants_cost coat_cost belt_cost shoes_cost amount_left : ℕ) : ℕ :=
  budget - (shirt_cost + pants_cost + coat_cost + belt_cost + shoes_cost + amount_left)

/-- Theorem: Given the specific budget and expenses, the amount spent on socks is $11 --/
theorem socks_expense_is_eleven :
  socks_expense 200 30 46 38 18 41 16 = 11 := by
  sorry

end socks_expense_is_eleven_l3543_354350


namespace correct_statement_l3543_354389

def p : Prop := 2017 % 2 = 1
def q : Prop := 2016 % 2 = 0

theorem correct_statement : p ∨ q := by sorry

end correct_statement_l3543_354389


namespace extreme_points_sum_l3543_354318

/-- Given that x = 2 and x = -4 are extreme points of f(x) = x³ + px² + qx, prove that p + q = -21 -/
theorem extreme_points_sum (p q : ℝ) : 
  (∀ x : ℝ, x = 2 ∨ x = -4 → (3*x^2 + 2*p*x + q = 0)) → 
  p + q = -21 := by
sorry

end extreme_points_sum_l3543_354318


namespace x_value_proof_l3543_354368

theorem x_value_proof (x : ℚ) (h : (1/2 : ℚ) - (1/4 : ℚ) + (1/8 : ℚ) = 8/x) : x = 64/3 := by
  sorry

end x_value_proof_l3543_354368


namespace identical_solutions_condition_l3543_354345

theorem identical_solutions_condition (k : ℝ) : 
  (∃! x y : ℝ, y = x^2 ∧ y = 4*x + k) ↔ k = -4 := by
sorry

end identical_solutions_condition_l3543_354345


namespace range_of_x_l3543_354367

-- Define the set of real numbers that satisfy the given condition
def S : Set ℝ := {x | ¬(x ∈ Set.Icc 2 5 ∨ x < 1 ∨ x > 4)}

-- Theorem stating that S is equal to the interval [1,2)
theorem range_of_x : S = Set.Ico 1 2 := by
  sorry

end range_of_x_l3543_354367


namespace sum_of_roots_quadratic_l3543_354317

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 5*x₁ - 1 = 0) → (x₂^2 + 5*x₂ - 1 = 0) → (x₁ + x₂ = -5) := by
  sorry

end sum_of_roots_quadratic_l3543_354317


namespace total_outfits_l3543_354312

/-- Represents the number of shirts available. -/
def num_shirts : ℕ := 7

/-- Represents the number of ties available. -/
def num_ties : ℕ := 5

/-- Represents the number of pairs of pants available. -/
def num_pants : ℕ := 4

/-- Represents the number of shoe types available. -/
def num_shoe_types : ℕ := 2

/-- Calculates the number of outfit combinations with a tie. -/
def outfits_with_tie : ℕ := num_shirts * num_pants * num_ties

/-- Calculates the number of outfit combinations without a tie. -/
def outfits_without_tie : ℕ := num_shirts * num_pants

/-- Theorem stating the total number of different outfits. -/
theorem total_outfits : outfits_with_tie + outfits_without_tie = 168 := by
  sorry

end total_outfits_l3543_354312


namespace garden_length_l3543_354334

/-- Proves that a rectangular garden with length twice its width and perimeter 240 yards has a length of 80 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width → -- length is twice the width
  2 * length + 2 * width = 240 → -- perimeter is 240 yards
  length = 80 := by
sorry

end garden_length_l3543_354334


namespace complex_arithmetic_problem_l3543_354379

theorem complex_arithmetic_problem : 
  (Complex.mk 2 5 + Complex.mk (-1) (-3)) * Complex.mk 3 1 = Complex.mk 1 7 := by
  sorry

end complex_arithmetic_problem_l3543_354379


namespace figure_y_value_l3543_354373

/-- Given a figure with a right triangle and two squares, prove the value of y -/
theorem figure_y_value (y : ℝ) (total_area : ℝ) : 
  total_area = 980 →
  (3 * y)^2 + (6 * y)^2 + (1/2 * 3 * y * 6 * y) = total_area →
  y = 70/9 := by
sorry

end figure_y_value_l3543_354373


namespace parabola_vertex_l3543_354392

/-- The parabola defined by y = (x-1)^2 + 3 has its vertex at (1,3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = (x - 1)^2 + 3 → (∃ (a : ℝ), y = a * (x - 1)^2 + 3 ∧ a ≠ 0) → 
  (1, 3) = (x, y) ∧ (∀ (x' y' : ℝ), y' = (x' - 1)^2 + 3 → y' ≥ y) :=
by sorry

end parabola_vertex_l3543_354392


namespace total_distance_is_250_l3543_354372

/-- Represents a cyclist's journey with specific conditions -/
structure CyclistJourney where
  speed : ℝ
  time_store_to_friend : ℝ
  distance_store_to_friend : ℝ
  h_speed_positive : speed > 0
  h_time_positive : time_store_to_friend > 0
  h_distance_positive : distance_store_to_friend > 0
  h_distance_store_to_friend : distance_store_to_friend = 50
  h_time_relation : 2 * time_store_to_friend = speed * distance_store_to_friend

/-- The total distance cycled in the journey -/
def total_distance (j : CyclistJourney) : ℝ :=
  3 * j.distance_store_to_friend + j.distance_store_to_friend

/-- Theorem stating that the total distance cycled is 250 miles -/
theorem total_distance_is_250 (j : CyclistJourney) : total_distance j = 250 := by
  sorry

end total_distance_is_250_l3543_354372


namespace hour_hand_rotation_l3543_354302

/-- Represents the number of degrees in a complete rotation. -/
def complete_rotation : ℕ := 360

/-- Represents the number of hours in a day. -/
def hours_per_day : ℕ := 24

/-- Represents the number of complete rotations the hour hand makes. -/
def rotations : ℕ := 12

/-- Represents the number of days in which the rotations occur. -/
def days : ℕ := 6

/-- Calculates the number of degrees the hour hand rotates per hour. -/
def degrees_per_hour : ℚ :=
  (rotations * complete_rotation) / (days * hours_per_day)

theorem hour_hand_rotation :
  degrees_per_hour = 30 := by sorry

end hour_hand_rotation_l3543_354302


namespace rectangular_plot_poles_l3543_354383

/-- Calculates the number of fence poles needed for a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  (2 * (length + width)) / pole_distance

/-- Proves that a 90m by 60m plot with poles 5m apart needs 60 poles -/
theorem rectangular_plot_poles : fence_poles 90 60 5 = 60 := by
  sorry

end rectangular_plot_poles_l3543_354383


namespace new_average_production_l3543_354363

theorem new_average_production (n : ℕ) (past_avg : ℝ) (today_prod : ℝ) 
  (h1 : n = 10)
  (h2 : past_avg = 50)
  (h3 : today_prod = 105) :
  (n * past_avg + today_prod) / (n + 1) = 55 := by
sorry

end new_average_production_l3543_354363


namespace elvis_matchsticks_l3543_354391

theorem elvis_matchsticks (total : ℕ) (elvis_squares : ℕ) (ralph_squares : ℕ) 
  (ralph_per_square : ℕ) (leftover : ℕ) :
  total = 50 →
  elvis_squares = 5 →
  ralph_squares = 3 →
  ralph_per_square = 8 →
  leftover = 6 →
  ∃ (elvis_per_square : ℕ), 
    elvis_per_square * elvis_squares + ralph_per_square * ralph_squares + leftover = total ∧
    elvis_per_square = 4 :=
by sorry

end elvis_matchsticks_l3543_354391


namespace equation_solution_l3543_354380

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (3 / x + (4 / x) / (8 / x) = 1.5) ∧ x = 3 := by
  sorry

end equation_solution_l3543_354380


namespace linear_function_property_l3543_354327

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define the inverse function
def InverseFunction (f g : ℝ → ℝ) : Prop := ∀ x, f (g x) = x ∧ g (f x) = x

theorem linear_function_property (f : ℝ → ℝ) 
  (h1 : LinearFunction f) 
  (h2 : ∃ g : ℝ → ℝ, InverseFunction f g ∧ ∀ x, f x = 5 * g x + 8) 
  (h3 : f 1 = 5) : 
  f 3 = 2 * Real.sqrt 5 + 5 := by
sorry

end linear_function_property_l3543_354327


namespace quadratic_solution_average_l3543_354330

theorem quadratic_solution_average (a b : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 3 * a * x + b
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ (x₁ + x₂) / 2 = 3 / 2) :=
by
  sorry

end quadratic_solution_average_l3543_354330


namespace susan_weather_probability_l3543_354360

/-- The probability of having exactly 1 or 2 sunny days in a 3-day period -/
def prob_1_or_2_sunny (p : ℚ) : ℚ :=
  (3 : ℚ) * p * (1 - p)^2 + (3 : ℚ) * p^2 * (1 - p)

/-- The theorem stating the probability of Susan getting her desired weather -/
theorem susan_weather_probability :
  prob_1_or_2_sunny (2/5) = 18/25 := by
  sorry


end susan_weather_probability_l3543_354360


namespace lcm_problem_l3543_354351

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := by
  sorry

end lcm_problem_l3543_354351


namespace parallel_vectors_x_value_l3543_354361

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (1, x) (-2, 3) → x = -3/2 := by
  sorry

end parallel_vectors_x_value_l3543_354361


namespace married_men_fraction_l3543_354358

theorem married_men_fraction (total_women : ℕ) (h_pos : 0 < total_women) :
  let single_women := (3 * total_women : ℕ) / 5
  let married_women := total_women - single_women
  let married_men := married_women
  let total_people := total_women + married_men
  (married_men : ℚ) / total_people = 2 / 7 := by
  sorry

end married_men_fraction_l3543_354358


namespace complex_magnitude_problem_l3543_354371

theorem complex_magnitude_problem (z : ℂ) : z = (2 + I) / (1 - I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_magnitude_problem_l3543_354371


namespace sine_sum_acute_triangle_l3543_354349

theorem sine_sum_acute_triangle (α β γ : Real) 
  (acute_triangle : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π)
  (acute_angles : α < π/2 ∧ β < π/2 ∧ γ < π/2) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end sine_sum_acute_triangle_l3543_354349


namespace optimal_profit_profit_function_correct_l3543_354308

/-- Represents the daily profit function for a factory -/
def daily_profit (x : ℝ) : ℝ := -50 * x^2 + 400 * x + 9000

/-- Represents the optimal price reduction -/
def optimal_reduction : ℝ := 4

/-- Represents the maximum daily profit -/
def max_profit : ℝ := 9800

/-- Theorem stating the optimal price reduction and maximum profit -/
theorem optimal_profit :
  (∀ x : ℝ, daily_profit x ≤ daily_profit optimal_reduction) ∧
  daily_profit optimal_reduction = max_profit := by
  sorry

/-- Theorem stating the correctness of the daily profit function -/
theorem profit_function_correct
  (cost_per_kg : ℝ)
  (initial_price : ℝ)
  (initial_sales : ℝ)
  (sales_increase_rate : ℝ)
  (h1 : cost_per_kg = 30)
  (h2 : initial_price = 48)
  (h3 : initial_sales = 500)
  (h4 : sales_increase_rate = 50) :
  ∀ x : ℝ, daily_profit x =
    (initial_price - x - cost_per_kg) * (initial_sales + sales_increase_rate * x) := by
  sorry

end optimal_profit_profit_function_correct_l3543_354308
