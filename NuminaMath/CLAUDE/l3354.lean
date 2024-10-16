import Mathlib

namespace NUMINAMATH_CALUDE_exchange_rate_change_l3354_335469

theorem exchange_rate_change 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) 
  (h_eq : (1 + x) * (1 + y) * (1 + z) = (1 - x) * (1 - y) * (1 - z)) : 
  (1 - x) * (1 - y) * (1 - z) < 1 :=
by sorry

end NUMINAMATH_CALUDE_exchange_rate_change_l3354_335469


namespace NUMINAMATH_CALUDE_zachary_initial_money_l3354_335451

/-- Calculates Zachary's initial money given the costs of items and additional amount needed --/
theorem zachary_initial_money 
  (football_cost shorts_cost shoes_cost additional_needed : ℚ) 
  (h1 : football_cost = 3.75)
  (h2 : shorts_cost = 2.40)
  (h3 : shoes_cost = 11.85)
  (h4 : additional_needed = 8) :
  football_cost + shorts_cost + shoes_cost - additional_needed = 9 := by
sorry

end NUMINAMATH_CALUDE_zachary_initial_money_l3354_335451


namespace NUMINAMATH_CALUDE_part_one_part_two_l3354_335489

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Define propositions p and q
def p (x : ℝ) (a : ℝ) : Prop := x ∈ A a
def q (x : ℝ) : Prop := x ∈ B

-- Theorem for part I
theorem part_one (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : A a ∪ B = Set.univ) : a = 2 := by
  sorry

-- Theorem for part II
theorem part_two (a : ℝ) (h : ∀ x, p x a → q x) : a ≤ 0 ∨ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3354_335489


namespace NUMINAMATH_CALUDE_no_brownies_left_l3354_335431

/-- Represents the number of brownies left after consumption --/
def brownies_left (total : ℚ) (tina_lunch : ℚ) (tina_dinner : ℚ) (husband : ℚ) (guests : ℚ) (daughter : ℚ) : ℚ :=
  total - (5 * (tina_lunch + tina_dinner) + 5 * husband + 2 * guests + 3 * daughter)

/-- Theorem stating that no brownies are left after consumption --/
theorem no_brownies_left : 
  brownies_left 24 1.5 0.5 0.75 2.5 2 = 0 := by
  sorry

#eval brownies_left 24 1.5 0.5 0.75 2.5 2

end NUMINAMATH_CALUDE_no_brownies_left_l3354_335431


namespace NUMINAMATH_CALUDE_angle_magnification_l3354_335484

theorem angle_magnification (original_angle : ℝ) (magnification : ℝ) :
  original_angle = 20 ∧ magnification = 10 →
  original_angle = original_angle := by sorry

end NUMINAMATH_CALUDE_angle_magnification_l3354_335484


namespace NUMINAMATH_CALUDE_cutlery_theorem_l3354_335422

/-- Calculates the total number of cutlery pieces after purchases -/
def totalCutlery (initialKnives : ℕ) : ℕ :=
  let initialTeaspoons := 2 * initialKnives
  let additionalKnives := initialKnives / 3
  let additionalTeaspoons := (2 * initialTeaspoons) / 3
  let totalKnives := initialKnives + additionalKnives
  let totalTeaspoons := initialTeaspoons + additionalTeaspoons
  totalKnives + totalTeaspoons

/-- Theorem stating that given 24 initial knives, the total cutlery after purchases is 112 -/
theorem cutlery_theorem : totalCutlery 24 = 112 := by
  sorry

end NUMINAMATH_CALUDE_cutlery_theorem_l3354_335422


namespace NUMINAMATH_CALUDE_complex_number_location_l3354_335452

theorem complex_number_location (m : ℝ) (z : ℂ) 
  (h1 : 2/3 < m) (h2 : m < 1) (h3 : z = Complex.mk (m - 1) (3*m - 2)) : 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l3354_335452


namespace NUMINAMATH_CALUDE_ellipse_segment_length_l3354_335421

/-- The length of segment AB for a given ellipse -/
theorem ellipse_segment_length : 
  ∀ (x y : ℝ), 
  (x^2 / 25 + y^2 / 16 = 1) →  -- Ellipse equation
  (∃ (a b : ℝ), 
    (a^2 / 25 + b^2 / 16 = 1) ∧  -- Points A and B satisfy ellipse equation
    (a = 3) ∧  -- x-coordinate of right focus
    (b = 16/5 ∨ b = -16/5)) →  -- y-coordinates of intersection points
  (16/5 - (-16/5) = 32/5) :=  -- Length of segment AB
by
  sorry

end NUMINAMATH_CALUDE_ellipse_segment_length_l3354_335421


namespace NUMINAMATH_CALUDE_parakeets_per_cage_l3354_335496

theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 8)
  (h2 : parrots_per_cage = 2)
  (h3 : total_birds = 72) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 7 := by
  sorry

end NUMINAMATH_CALUDE_parakeets_per_cage_l3354_335496


namespace NUMINAMATH_CALUDE_rowing_distance_calculation_l3354_335420

/-- Represents the problem of calculating the distance to a destination given rowing conditions. -/
theorem rowing_distance_calculation 
  (rowing_speed : ℝ) 
  (current_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : rowing_speed = 10) 
  (h2 : current_speed = 2) 
  (h3 : total_time = 5) : 
  ∃ (distance : ℝ), distance = 24 ∧ 
    distance / (rowing_speed + current_speed) + 
    distance / (rowing_speed - current_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_rowing_distance_calculation_l3354_335420


namespace NUMINAMATH_CALUDE_charles_housesitting_rate_l3354_335499

/-- Represents the earnings of Charles from housesitting and dog walking -/
structure Earnings where
  housesitting_rate : ℝ
  dog_walking_rate : ℝ
  housesitting_hours : ℕ
  dogs_walked : ℕ
  total_earnings : ℝ

/-- Theorem stating that given the conditions, Charles earns $15 per hour for housesitting -/
theorem charles_housesitting_rate (e : Earnings) 
  (h1 : e.dog_walking_rate = 22)
  (h2 : e.housesitting_hours = 10)
  (h3 : e.dogs_walked = 3)
  (h4 : e.total_earnings = 216)
  (h5 : e.housesitting_rate * e.housesitting_hours + e.dog_walking_rate * e.dogs_walked = e.total_earnings) :
  e.housesitting_rate = 15 := by
  sorry

end NUMINAMATH_CALUDE_charles_housesitting_rate_l3354_335499


namespace NUMINAMATH_CALUDE_difference_of_squares_l3354_335401

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3354_335401


namespace NUMINAMATH_CALUDE_inequality_proof_l3354_335474

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * (a + b) + a * c * (a + c) + b * c * (b + c)) / (a * b * c) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3354_335474


namespace NUMINAMATH_CALUDE_water_bucket_problem_l3354_335445

theorem water_bucket_problem (total_parts : Nat) (part_weight : Nat) (remainder : Nat) :
  total_parts = 7 ∧ part_weight = 900 ∧ remainder = 200 →
  (total_parts * part_weight + remainder : ℝ) / 1000 = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_water_bucket_problem_l3354_335445


namespace NUMINAMATH_CALUDE_tan_product_from_cosine_sum_l3354_335454

theorem tan_product_from_cosine_sum (α β : ℝ) 
  (h : 3 * Real.cos (2 * α + β) + 5 * Real.cos β = 0) : 
  Real.tan (α + β) * Real.tan α = -4 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_from_cosine_sum_l3354_335454


namespace NUMINAMATH_CALUDE_simplify_expression_l3354_335453

theorem simplify_expression :
  (((Real.sqrt 5 - 2) ^ (Real.sqrt 3 - 2)) / ((Real.sqrt 5 + 2) ^ (Real.sqrt 3 + 2))) = 41 + 20 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3354_335453


namespace NUMINAMATH_CALUDE_cheese_cost_is_50_l3354_335440

/-- The cost of a sandwich in cents -/
def sandwich_cost : ℕ := 90

/-- The cost of a slice of bread in cents -/
def bread_cost : ℕ := 15

/-- The cost of a slice of ham in cents -/
def ham_cost : ℕ := 25

/-- The cost of a slice of cheese in cents -/
def cheese_cost : ℕ := sandwich_cost - bread_cost - ham_cost

theorem cheese_cost_is_50 : cheese_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_cheese_cost_is_50_l3354_335440


namespace NUMINAMATH_CALUDE_toaster_customers_l3354_335492

/-- Represents the inverse proportionality between customers and cost -/
def inverse_prop (k : ℝ) (p c : ℝ) : Prop := p * c = k

/-- Applies a discount to a given price -/
def apply_discount (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

theorem toaster_customers : 
  ∀ (k : ℝ),
  inverse_prop k 12 600 →
  (∃ (p : ℝ), 
    inverse_prop k p (apply_discount (2 * 400) 0.1) ∧ 
    p = 10) := by
sorry

end NUMINAMATH_CALUDE_toaster_customers_l3354_335492


namespace NUMINAMATH_CALUDE_simple_interest_rate_l3354_335410

/-- Simple interest rate calculation -/
theorem simple_interest_rate 
  (principal : ℝ) 
  (final_amount : ℝ) 
  (time : ℝ) 
  (h1 : principal = 750) 
  (h2 : final_amount = 900) 
  (h3 : time = 8) :
  (final_amount - principal) * 100 / (principal * time) = 2.5 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l3354_335410


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l3354_335404

theorem unique_solution_logarithmic_equation :
  ∃! x : ℝ, x > 0 ∧ x^(Real.log x / Real.log 10) = x^2 / 10 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l3354_335404


namespace NUMINAMATH_CALUDE_port_distance_equation_l3354_335446

/-- The distance between two ports satisfies a specific equation based on ship and river speeds --/
theorem port_distance_equation (ship_speed : ℝ) (current_speed : ℝ) (time_difference : ℝ) 
  (h1 : ship_speed = 26)
  (h2 : current_speed = 2)
  (h3 : time_difference = 3) :
  ∃ x : ℝ, x / (ship_speed + current_speed) = x / (ship_speed - current_speed) - time_difference :=
by sorry

end NUMINAMATH_CALUDE_port_distance_equation_l3354_335446


namespace NUMINAMATH_CALUDE_tangent_line_equation_monotonic_increase_condition_l3354_335473

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

-- Theorem for the tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  f (-1) 1 = 0 ∧ 
  (deriv (f (-1))) 1 = -Real.log 2 →
  (Real.log 2 * x + y - Real.log 2 = 0) ↔ 
  y = (deriv (f (-1))) 1 * (x - 1) + f (-1) 1 :=
sorry

-- Theorem for monotonic increase condition
theorem monotonic_increase_condition (a : ℝ) :
  Monotone (f a) ↔ a ≥ (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_monotonic_increase_condition_l3354_335473


namespace NUMINAMATH_CALUDE_divisibility_rule_37_l3354_335478

/-- Given a positive integer n, returns a list of its three-digit segments from right to left -/
def threeDigitSegments (n : ℕ+) : List ℕ :=
  sorry

/-- The divisibility rule for 37 -/
theorem divisibility_rule_37 (n : ℕ+) : 
  37 ∣ n ↔ 37 ∣ (threeDigitSegments n).sum :=
sorry

end NUMINAMATH_CALUDE_divisibility_rule_37_l3354_335478


namespace NUMINAMATH_CALUDE_area_of_polygon_l3354_335482

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a polygon -/
structure Polygon :=
  (vertices : List Point)

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℝ := sorry

/-- Checks if a quadrilateral is a rectangle -/
def is_rectangle (a b c d : Point) : Prop := sorry

/-- Checks if a quadrilateral is a square -/
def is_square (a b f e : Point) : Prop := sorry

/-- Checks if two line segments are perpendicular -/
def is_perpendicular (a f : Point) (f e : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem area_of_polygon (a b c d e f : Point) :
  is_rectangle a b c d →
  is_square a b f e →
  is_perpendicular a f f e →
  distance a f = 10 →
  distance f e = 15 →
  distance c d = 20 →
  area (Polygon.mk [a, f, e, d, c, b]) = 375 := by
  sorry

end NUMINAMATH_CALUDE_area_of_polygon_l3354_335482


namespace NUMINAMATH_CALUDE_vector_BC_l3354_335417

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (3, 2)
def AC : ℝ × ℝ := (-4, -3)

theorem vector_BC : (B.1 - A.1 + AC.1, B.2 - A.2 + AC.2) = (-7, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_l3354_335417


namespace NUMINAMATH_CALUDE_condition_neither_necessary_nor_sufficient_l3354_335436

-- Define the sets M and P
def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

-- Statement to prove
theorem condition_neither_necessary_nor_sufficient :
  ¬(∀ x : ℝ, (x ∈ M ∩ P → (x ∈ M ∨ x ∈ P)) ∧ ((x ∈ M ∨ x ∈ P) → x ∈ M ∩ P)) :=
sorry

end NUMINAMATH_CALUDE_condition_neither_necessary_nor_sufficient_l3354_335436


namespace NUMINAMATH_CALUDE_collinear_points_sum_l3354_335407

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop := sorry

/-- The theorem states that if the given points are collinear, then p + q = 6. -/
theorem collinear_points_sum (p q : ℝ) : 
  collinear (2, p, q) (p, 3, q) (p, q, 4) → p + q = 6 := by sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l3354_335407


namespace NUMINAMATH_CALUDE_chef_used_one_apple_l3354_335465

/-- The number of apples used by a chef when making pies -/
def apples_used (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that the chef used 1 apple -/
theorem chef_used_one_apple :
  apples_used 40 39 = 1 := by
  sorry

end NUMINAMATH_CALUDE_chef_used_one_apple_l3354_335465


namespace NUMINAMATH_CALUDE_toy_value_proof_l3354_335426

theorem toy_value_proof (total_toys : ℕ) (total_value : ℕ) (special_toy_value : ℕ) :
  total_toys = 9 →
  total_value = 52 →
  special_toy_value = 12 →
  ∃ (other_toy_value : ℕ),
    (total_toys - 1) * other_toy_value + special_toy_value = total_value ∧
    other_toy_value = 5 := by
  sorry

end NUMINAMATH_CALUDE_toy_value_proof_l3354_335426


namespace NUMINAMATH_CALUDE_count_valid_assignments_five_l3354_335495

/-- Represents a valid assignment of students to tests -/
def ValidAssignment (n : ℕ) := Fin n → Fin n → Prop

/-- The number of valid assignments for n students and n tests -/
def CountValidAssignments (n : ℕ) : ℕ := sorry

/-- The condition that each student takes exactly 2 distinct tests -/
def StudentTakesTwoTests (assignment : ValidAssignment 5) : Prop :=
  ∀ s : Fin 5, ∃! t1 t2 : Fin 5, t1 ≠ t2 ∧ assignment s t1 ∧ assignment s t2

/-- The condition that each test is taken by exactly 2 students -/
def TestTakenByTwoStudents (assignment : ValidAssignment 5) : Prop :=
  ∀ t : Fin 5, ∃! s1 s2 : Fin 5, s1 ≠ s2 ∧ assignment s1 t ∧ assignment s2 t

theorem count_valid_assignments_five :
  (∀ assignment : ValidAssignment 5,
    StudentTakesTwoTests assignment ∧ TestTakenByTwoStudents assignment) →
  CountValidAssignments 5 = 2040 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_assignments_five_l3354_335495


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l3354_335433

theorem contrapositive_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + x - a ≠ 0) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l3354_335433


namespace NUMINAMATH_CALUDE_ratio_yz_l3354_335437

theorem ratio_yz (x y z : ℝ) 
  (h1 : (x + 53/18 * y - 143/9 * z) / z = 1)
  (h2 : (3/8 * x - 17/4 * y + z) / y = 1) :
  y / z = 352 / 305 := by
sorry

end NUMINAMATH_CALUDE_ratio_yz_l3354_335437


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3354_335460

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3354_335460


namespace NUMINAMATH_CALUDE_deer_bridge_problem_l3354_335402

theorem deer_bridge_problem (y : ℚ) : 
  (3 * (3 * (3 * y - 50) - 50) - 50) * 4 - 50 = 0 ∧ y > 0 → y = 425 / 18 := by
  sorry

end NUMINAMATH_CALUDE_deer_bridge_problem_l3354_335402


namespace NUMINAMATH_CALUDE_english_only_students_l3354_335461

theorem english_only_students (total : ℕ) (all_three : ℕ) (english_only : ℕ) (french_only : ℕ) :
  total = 35 →
  all_three = 2 →
  english_only = 3 * french_only →
  english_only + french_only + all_three = total →
  english_only - all_three = 23 := by
  sorry

end NUMINAMATH_CALUDE_english_only_students_l3354_335461


namespace NUMINAMATH_CALUDE_stirring_evenly_key_to_representativeness_l3354_335488

/-- Represents a sampling method -/
inductive SamplingMethod
| Lottery
| Other

/-- Represents actions in the lottery method -/
inductive LotteryAction
| MakeTickets
| StirEvenly
| DrawOneByOne
| DrawWithoutReplacement

/-- Represents the property of being representative -/
def IsRepresentative (sample : Set α) : Prop := sorry

/-- The lottery method -/
def lotteryMethod : SamplingMethod := SamplingMethod.Lottery

/-- Function to determine if an action is key to representativeness -/
def isKeyToRepresentativeness (action : LotteryAction) (method : SamplingMethod) : Prop := sorry

/-- Theorem stating that stirring evenly is key to representativeness in the lottery method -/
theorem stirring_evenly_key_to_representativeness :
  isKeyToRepresentativeness LotteryAction.StirEvenly lotteryMethod := by sorry

end NUMINAMATH_CALUDE_stirring_evenly_key_to_representativeness_l3354_335488


namespace NUMINAMATH_CALUDE_visited_neither_country_l3354_335494

theorem visited_neither_country (total : ℕ) (visited_iceland : ℕ) (visited_norway : ℕ) (visited_both : ℕ) :
  total = 90 →
  visited_iceland = 55 →
  visited_norway = 33 →
  visited_both = 51 →
  total - (visited_iceland + visited_norway - visited_both) = 53 := by
sorry

end NUMINAMATH_CALUDE_visited_neither_country_l3354_335494


namespace NUMINAMATH_CALUDE_school_attendance_problem_l3354_335414

theorem school_attendance_problem (boys : ℕ) (girls : ℕ) :
  boys = 2000 →
  (boys + girls : ℝ) = 1.4 * boys →
  girls = 800 := by
sorry

end NUMINAMATH_CALUDE_school_attendance_problem_l3354_335414


namespace NUMINAMATH_CALUDE_profit_and_sales_maximization_max_profit_with_constraints_l3354_335441

/-- Represents the daily sales quantity as a function of the selling price -/
def sales_quantity (x : ℝ) : ℝ := -10 * x + 400

/-- Represents the daily profit as a function of the selling price -/
def profit (x : ℝ) : ℝ := sales_quantity x * (x - 10)

/-- The cost price of the item -/
def cost_price : ℝ := 10

/-- The domain constraints for the selling price -/
def price_domain (x : ℝ) : Prop := 10 < x ∧ x ≤ 40

theorem profit_and_sales_maximization (x : ℝ) 
  (h : price_domain x) : 
  profit x = 1250 ∧ 
  (∀ y, price_domain y → sales_quantity x ≥ sales_quantity y) → 
  x = 15 :=
sorry

theorem max_profit_with_constraints (x : ℝ) :
  28 ≤ x ∧ x ≤ 35 →
  profit x ≤ 2160 :=
sorry

end NUMINAMATH_CALUDE_profit_and_sales_maximization_max_profit_with_constraints_l3354_335441


namespace NUMINAMATH_CALUDE_inverse_variation_y_sqrt_z_l3354_335483

theorem inverse_variation_y_sqrt_z (y z : ℝ) (k : ℝ) (h1 : y^2 * Real.sqrt z = k) 
  (h2 : 3^2 * Real.sqrt 4 = k) (h3 : y = 6) : z = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_y_sqrt_z_l3354_335483


namespace NUMINAMATH_CALUDE_books_before_sale_l3354_335493

theorem books_before_sale (books_bought : ℕ) (total_books : ℕ) 
  (h1 : books_bought = 56) 
  (h2 : total_books = 91) : 
  total_books - books_bought = 35 := by
  sorry

end NUMINAMATH_CALUDE_books_before_sale_l3354_335493


namespace NUMINAMATH_CALUDE_estimate_wild_rabbits_l3354_335439

theorem estimate_wild_rabbits (initial_marked : ℕ) (recaptured : ℕ) (marked_in_recapture : ℕ) :
  initial_marked = 100 →
  recaptured = 40 →
  marked_in_recapture = 5 →
  (recaptured * initial_marked) / marked_in_recapture = 800 :=
by sorry

end NUMINAMATH_CALUDE_estimate_wild_rabbits_l3354_335439


namespace NUMINAMATH_CALUDE_sum_of_selection_l3354_335448

/-- Represents a selection of numbers from an 8x8 grid -/
def Selection := Fin 8 → Fin 8

/-- The sum of numbers in a selection -/
def sum_selection (s : Selection) : ℕ :=
  Finset.sum Finset.univ (λ i => s i + 1 + 8 * i)

/-- Theorem: The sum of any valid selection is 260 -/
theorem sum_of_selection (s : Selection) (h : Function.Injective s) : sum_selection s = 260 := by
  sorry

#eval sum_selection (λ i => i)  -- Should output 260

end NUMINAMATH_CALUDE_sum_of_selection_l3354_335448


namespace NUMINAMATH_CALUDE_hannah_grapes_count_l3354_335411

def sophie_oranges_daily : ℕ := 20
def observation_days : ℕ := 30
def total_fruits : ℕ := 1800

def hannah_grapes_daily : ℕ := (total_fruits - sophie_oranges_daily * observation_days) / observation_days

theorem hannah_grapes_count : hannah_grapes_daily = 40 := by
  sorry

end NUMINAMATH_CALUDE_hannah_grapes_count_l3354_335411


namespace NUMINAMATH_CALUDE_class_size_l3354_335424

theorem class_size (s : ℕ) (r : ℕ) : 
  (0 * 2 + 1 * 12 + 2 * 10 + 3 * r) / s = 2 →
  s = 2 + 12 + 10 + r →
  s = 40 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l3354_335424


namespace NUMINAMATH_CALUDE_steps_calculation_l3354_335412

/-- The number of steps Benjamin took from the hotel to Times Square. -/
def total_steps : ℕ := 582

/-- The number of steps Benjamin took from the hotel to Rockefeller Center. -/
def steps_to_rockefeller : ℕ := 354

/-- The number of steps Benjamin took from Rockefeller Center to Times Square. -/
def steps_rockefeller_to_times_square : ℕ := total_steps - steps_to_rockefeller

theorem steps_calculation :
  steps_rockefeller_to_times_square = 228 :=
by sorry

end NUMINAMATH_CALUDE_steps_calculation_l3354_335412


namespace NUMINAMATH_CALUDE_bank_interest_calculation_l3354_335434

def initial_deposit : ℝ := 5600
def interest_rate : ℝ := 0.07
def time_period : ℕ := 2

theorem bank_interest_calculation :
  let interest_per_year := initial_deposit * interest_rate
  let total_interest := interest_per_year * time_period
  initial_deposit + total_interest = 6384 := by
  sorry

end NUMINAMATH_CALUDE_bank_interest_calculation_l3354_335434


namespace NUMINAMATH_CALUDE_imaginary_part_z_2017_l3354_335464

theorem imaginary_part_z_2017 : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i) / (1 - i)
  Complex.im (z^2017) = Complex.im i := by sorry

end NUMINAMATH_CALUDE_imaginary_part_z_2017_l3354_335464


namespace NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l3354_335419

/-- The number of erasers Jungkook has -/
def jungkook_erasers : ℕ := 6

/-- The number of erasers Jimin has -/
def jimin_erasers : ℕ := jungkook_erasers + 4

/-- The number of erasers Seokjin has -/
def seokjin_erasers : ℕ := jimin_erasers - 3

theorem jungkook_has_fewest_erasers :
  jungkook_erasers < jimin_erasers ∧ jungkook_erasers < seokjin_erasers :=
by sorry

end NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l3354_335419


namespace NUMINAMATH_CALUDE_insert_books_combinations_l3354_335468

theorem insert_books_combinations (n m : ℕ) : 
  n = 5 → m = 3 → (n + 1) * (n + 2) * (n + 3) = 336 := by
  sorry

end NUMINAMATH_CALUDE_insert_books_combinations_l3354_335468


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_factors_of_125_l3354_335498

theorem smallest_sum_of_three_factors_of_125 :
  ∀ a b c : ℕ+,
  a * b * c = 125 →
  ∀ x y z : ℕ+,
  x * y * z = 125 →
  a + b + c ≤ x + y + z →
  a + b + c = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_factors_of_125_l3354_335498


namespace NUMINAMATH_CALUDE_expression_equals_two_l3354_335408

theorem expression_equals_two (x : ℝ) (h : x ≠ -1) :
  ((x - 1) / (x + 1) + 1) / (x / (x + 1)) = 2 := by sorry

end NUMINAMATH_CALUDE_expression_equals_two_l3354_335408


namespace NUMINAMATH_CALUDE_fraction_product_l3354_335480

theorem fraction_product : 
  (4 : ℚ) / 5 * 5 / 6 * 6 / 7 * 7 / 8 * 8 / 9 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3354_335480


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3354_335462

theorem sufficient_not_necessary (a b : ℝ) : 
  ((a > b ∧ b > 1) → (a - b < a^2 - b^2)) ∧ 
  ¬((a - b < a^2 - b^2) → (a > b ∧ b > 1)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3354_335462


namespace NUMINAMATH_CALUDE_largest_x_for_equation_l3354_335487

theorem largest_x_for_equation : 
  (∀ x y : ℤ, x > 3 → x^2 - x*y - 2*y^2 ≠ 9) ∧ 
  (∃ y : ℤ, 3^2 - 3*y - 2*y^2 = 9) :=
sorry

end NUMINAMATH_CALUDE_largest_x_for_equation_l3354_335487


namespace NUMINAMATH_CALUDE_greatest_number_with_conditions_l3354_335435

theorem greatest_number_with_conditions : ∃ n : ℕ, 
  n < 150 ∧ 
  (∃ k : ℕ, n = k^2) ∧ 
  n % 3 = 0 ∧
  ∀ m : ℕ, m < 150 → (∃ k : ℕ, m = k^2) → m % 3 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_conditions_l3354_335435


namespace NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_435_sin_15_eq_zero_l3354_335479

theorem cos_75_cos_15_minus_sin_435_sin_15_eq_zero :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - 
  Real.sin (435 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_435_sin_15_eq_zero_l3354_335479


namespace NUMINAMATH_CALUDE_smallest_angle_25_sided_polygon_l3354_335485

/-- Represents a convex polygon with n sides and angles in an arithmetic sequence --/
structure ConvexPolygon (n : ℕ) where
  -- The common difference of the arithmetic sequence of angles
  d : ℕ
  -- The smallest angle in the polygon
  smallest_angle : ℕ
  -- Ensure the polygon is convex (all angles less than 180°)
  convex : smallest_angle + (n - 1) * d < 180
  -- Ensure the sum of angles is correct for an n-sided polygon
  angle_sum : smallest_angle * n + (n * (n - 1) * d) / 2 = (n - 2) * 180

theorem smallest_angle_25_sided_polygon :
  ∃ (p : ConvexPolygon 25), p.smallest_angle = 154 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_25_sided_polygon_l3354_335485


namespace NUMINAMATH_CALUDE_resulting_shape_volume_l3354_335476

def original_edge_length : ℝ := 5
def small_cube_edge_length : ℝ := 1
def number_of_small_cubes_removed : ℕ := 5

def cube_volume (edge_length : ℝ) : ℝ := edge_length ^ 3

theorem resulting_shape_volume :
  cube_volume original_edge_length - 
  (number_of_small_cubes_removed : ℝ) * cube_volume small_cube_edge_length = 120 := by
  sorry

end NUMINAMATH_CALUDE_resulting_shape_volume_l3354_335476


namespace NUMINAMATH_CALUDE_consecutive_even_integers_square_product_l3354_335491

theorem consecutive_even_integers_square_product : 
  ∀ (a b c : ℤ),
  (b = a + 2 ∧ c = b + 2) →  -- consecutive even integers
  (a * b * c = 12 * (a + b + c)) →  -- product is 12 times their sum
  (a^2 * b^2 * c^2 = 36864) :=  -- product of squares is 36864
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_square_product_l3354_335491


namespace NUMINAMATH_CALUDE_team_a_more_uniform_l3354_335449

/-- Represents a team of girls in a duet --/
structure Team where
  members : Fin 6 → ℝ
  variance : ℝ

/-- The problem setup --/
def problem_setup (team_a team_b : Team) : Prop :=
  team_a.variance = 1.2 ∧ team_b.variance = 2.0

/-- Definition of more uniform heights --/
def more_uniform (team1 team2 : Team) : Prop :=
  team1.variance < team2.variance

/-- The main theorem --/
theorem team_a_more_uniform (team_a team_b : Team) 
  (h : problem_setup team_a team_b) : 
  more_uniform team_a team_b := by
  sorry

#check team_a_more_uniform

end NUMINAMATH_CALUDE_team_a_more_uniform_l3354_335449


namespace NUMINAMATH_CALUDE_language_knowledge_distribution_l3354_335490

/-- Given the distribution of language knowledge among students, prove that
    among those who know both German and French, more than 90% know English. -/
theorem language_knowledge_distribution (a b c d : ℝ) 
    (h1 : a + b ≥ 0.9 * (a + b + c + d))
    (h2 : a + c ≥ 0.9 * (a + b + c + d))
    (h3 : a ≥ 0) (h4 : b ≥ 0) (h5 : c ≥ 0) (h6 : d ≥ 0) : 
    a ≥ 9 * d := by
  sorry


end NUMINAMATH_CALUDE_language_knowledge_distribution_l3354_335490


namespace NUMINAMATH_CALUDE_haley_tv_watching_l3354_335477

/-- Haley's TV watching problem -/
theorem haley_tv_watching (total_hours sunday_hours : ℕ) 
  (h1 : total_hours = 9)
  (h2 : sunday_hours = 3) :
  total_hours - sunday_hours = 6 := by
  sorry

end NUMINAMATH_CALUDE_haley_tv_watching_l3354_335477


namespace NUMINAMATH_CALUDE_inequality_proof_l3354_335481

theorem inequality_proof (a b c d e f : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f) :
  Real.rpow (a * b * c / (a + b + d)) (1/3) + Real.rpow (d * e * f / (c + e + f)) (1/3) 
  < Real.rpow ((a + b + d) * (c + e + f)) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3354_335481


namespace NUMINAMATH_CALUDE_buses_needed_l3354_335403

theorem buses_needed (num_students : ℕ) (seats_per_bus : ℕ) (h1 : num_students = 14) (h2 : seats_per_bus = 2) :
  (num_students + seats_per_bus - 1) / seats_per_bus = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_buses_needed_l3354_335403


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l3354_335400

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 = 1 →
  a 7 = a 5 + 2 * a 3 →
  a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l3354_335400


namespace NUMINAMATH_CALUDE_tower_divisibility_l3354_335425

/-- Represents the number of towers that can be built with cubes up to edge-length n -/
def S : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 1 => S n * (min 5 (n + 1))

/-- The problem statement -/
theorem tower_divisibility : S 9 % 1000 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tower_divisibility_l3354_335425


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_3pi_over_5_l3354_335442

theorem cos_2alpha_minus_3pi_over_5 (α : Real) 
  (h : Real.sin (α + π/5) = Real.sqrt 7 / 3) : 
  Real.cos (2*α - 3*π/5) = 5/9 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_3pi_over_5_l3354_335442


namespace NUMINAMATH_CALUDE_ramp_cost_is_2950_l3354_335418

/-- Calculate the total cost of installing a ramp --/
def total_ramp_cost (permit_cost : ℝ) (contractor_hourly_rate : ℝ) 
  (contractor_days : ℕ) (contractor_hours_per_day : ℕ) 
  (inspector_discount_percent : ℝ) : ℝ :=
  let contractor_total_hours := contractor_days * contractor_hours_per_day
  let contractor_cost := contractor_hourly_rate * contractor_total_hours
  let inspector_cost := contractor_cost * (1 - inspector_discount_percent)
  permit_cost + contractor_cost + inspector_cost

/-- Theorem stating the total cost of installing a ramp is $2950 --/
theorem ramp_cost_is_2950 : 
  total_ramp_cost 250 150 3 5 0.8 = 2950 := by
  sorry

end NUMINAMATH_CALUDE_ramp_cost_is_2950_l3354_335418


namespace NUMINAMATH_CALUDE_exist_tetrahedra_volume_area_paradox_l3354_335447

/-- A tetrahedron in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- Calculate the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Calculate the area of a face of a tetrahedron -/
def face_area (t : Tetrahedron) (face : Fin 4) : ℝ := sorry

/-- Theorem: There exist two tetrahedra such that one has greater volume
    but smaller or equal face areas compared to the other -/
theorem exist_tetrahedra_volume_area_paradox :
  ∃ (t₁ t₂ : Tetrahedron),
    volume t₁ > volume t₂ ∧
    ∀ (face₁ : Fin 4), ∃ (face₂ : Fin 4),
      face_area t₁ face₁ ≤ face_area t₂ face₂ :=
sorry

end NUMINAMATH_CALUDE_exist_tetrahedra_volume_area_paradox_l3354_335447


namespace NUMINAMATH_CALUDE_a_geq_one_l3354_335413

-- Define the conditions
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x > a

-- Define the relationship between p and q
axiom not_p_sufficient_for_not_q : ∀ x a : ℝ, (¬p x → ¬q x a) ∧ ∃ x a : ℝ, ¬p x ∧ q x a

-- Theorem to prove
theorem a_geq_one : ∀ a : ℝ, (∀ x : ℝ, q x a → p x) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_a_geq_one_l3354_335413


namespace NUMINAMATH_CALUDE_cats_remaining_after_sale_l3354_335432

/-- The number of cats remaining after a sale at a pet store. -/
theorem cats_remaining_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 19 → house = 45 → sold = 56 → siamese + house - sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_after_sale_l3354_335432


namespace NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l3354_335429

theorem unique_solution_ceiling_equation :
  ∃! b : ℝ, b + ⌈b⌉ = 19.6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_ceiling_equation_l3354_335429


namespace NUMINAMATH_CALUDE_min_value_product_quotient_l3354_335444

theorem min_value_product_quotient (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x^2 + 4*x + 1) * (y^2 + 4*y + 1) * (z^2 + 4*z + 1)) / (x*y*z) ≥ 216 ∧
  ((1^2 + 4*1 + 1) * (1^2 + 4*1 + 1) * (1^2 + 4*1 + 1)) / (1*1*1) = 216 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_quotient_l3354_335444


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_a_nonzero_b_zero_l3354_335457

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_iff_a_nonzero_b_zero (a b : ℝ) :
  is_purely_imaginary (Complex.mk b (a)) ↔ a ≠ 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_a_nonzero_b_zero_l3354_335457


namespace NUMINAMATH_CALUDE_parabola_y_intercept_l3354_335427

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (a b : ℝ) : 
  (6 = 2^2 + 2*a + b) ∧ (-14 = (-2)^2 + (-2)*a + b) → b = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_intercept_l3354_335427


namespace NUMINAMATH_CALUDE_one_french_horn_player_l3354_335471

/-- Represents the number of players for each instrument in an orchestra -/
structure Orchestra :=
  (total : ℕ)
  (drummer : ℕ)
  (trombone : ℕ)
  (trumpet : ℕ)
  (violin : ℕ)
  (cello : ℕ)
  (contrabass : ℕ)
  (clarinet : ℕ)
  (flute : ℕ)
  (maestro : ℕ)

/-- Theorem stating that there is one French horn player in the orchestra -/
theorem one_french_horn_player (o : Orchestra) 
  (h_total : o.total = 21)
  (h_drummer : o.drummer = 1)
  (h_trombone : o.trombone = 4)
  (h_trumpet : o.trumpet = 2)
  (h_violin : o.violin = 3)
  (h_cello : o.cello = 1)
  (h_contrabass : o.contrabass = 1)
  (h_clarinet : o.clarinet = 3)
  (h_flute : o.flute = 4)
  (h_maestro : o.maestro = 1) :
  o.total = o.drummer + o.trombone + o.trumpet + o.violin + o.cello + o.contrabass + o.clarinet + o.flute + o.maestro + 1 :=
by sorry

end NUMINAMATH_CALUDE_one_french_horn_player_l3354_335471


namespace NUMINAMATH_CALUDE_eighteen_power_mn_l3354_335438

theorem eighteen_power_mn (m n : ℤ) (R S : ℝ) (hR : R = 2^m) (hS : S = 3^n) :
  18^(m+n) = R^n * S^(2*m) := by
  sorry

end NUMINAMATH_CALUDE_eighteen_power_mn_l3354_335438


namespace NUMINAMATH_CALUDE_complement_M_in_U_l3354_335463

-- Define the set U
def U : Set ℕ := {1,2,3,4,5,6,7}

-- Define the set M
def M : Set ℕ := {x ∈ U | x^2 - 6*x + 5 ≤ 0}

-- State the theorem
theorem complement_M_in_U : (U \ M) = {6,7} := by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l3354_335463


namespace NUMINAMATH_CALUDE_set_operations_l3354_335486

def U : Set Int := {x | 0 < x ∧ x ≤ 10}
def A : Set Int := {1, 2, 4, 5, 9}
def B : Set Int := {4, 6, 7, 8, 10}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ (A ∪ B)) = {3}) ∧
  ((U \ A) ∩ (U \ B) = {3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3354_335486


namespace NUMINAMATH_CALUDE_total_tickets_bought_l3354_335467

theorem total_tickets_bought (adult_price children_price total_spent adult_count : ℚ)
  (h1 : adult_price = 5.5)
  (h2 : children_price = 3.5)
  (h3 : total_spent = 83.5)
  (h4 : adult_count = 5) :
  ∃ (children_count : ℚ), adult_count + children_count = 21 ∧
    adult_price * adult_count + children_price * children_count = total_spent :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_bought_l3354_335467


namespace NUMINAMATH_CALUDE_square_division_theorem_l3354_335456

theorem square_division_theorem (s : ℝ) :
  s > 0 →
  (3 * s = 42) →
  s = 14 :=
by sorry

end NUMINAMATH_CALUDE_square_division_theorem_l3354_335456


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l3354_335459

theorem solution_set_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l3354_335459


namespace NUMINAMATH_CALUDE_triangle_position_after_two_moves_l3354_335455

/-- Represents the sides of a square --/
inductive SquareSide
  | Top
  | Right
  | Bottom
  | Left

/-- Represents a regular octagon --/
structure RegularOctagon where
  inner_angle : ℝ
  inner_angle_eq : inner_angle = 135

/-- Represents a square rolling around an octagon --/
structure RollingSquare where
  octagon : RegularOctagon
  rotation_per_move : ℝ
  rotation_per_move_eq : rotation_per_move = 135

/-- The result of rolling a square around an octagon --/
def roll_square (initial_side : SquareSide) (num_moves : ℕ) : SquareSide :=
  sorry

theorem triangle_position_after_two_moves :
  ∀ (octagon : RegularOctagon) (square : RollingSquare),
    roll_square SquareSide.Bottom 2 = SquareSide.Bottom :=
  sorry

end NUMINAMATH_CALUDE_triangle_position_after_two_moves_l3354_335455


namespace NUMINAMATH_CALUDE_hiker_distance_l3354_335466

/-- Calculates the final straight-line distance of a hiker from their starting point
    given their movements in cardinal directions. -/
theorem hiker_distance (north south west east : ℝ) :
  north = 20 ∧ south = 8 ∧ west = 15 ∧ east = 10 →
  Real.sqrt ((north - south)^2 + (west - east)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_l3354_335466


namespace NUMINAMATH_CALUDE_monster_perimeter_l3354_335450

theorem monster_perimeter (r : ℝ) (θ : ℝ) (h1 : r = 2) (h2 : θ = 2 * π / 3) :
  let arc_length := (2 * π - θ) / (2 * π) * (2 * π * r)
  let chord_length := 2 * r * Real.sin (θ / 2)
  arc_length + chord_length = 8 * π / 3 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_monster_perimeter_l3354_335450


namespace NUMINAMATH_CALUDE_valid_numbers_are_unique_l3354_335406

/-- Represents a six-digit number of the form 387abc --/
def SixDigitNumber (a b c : Nat) : Nat :=
  387000 + a * 100 + b * 10 + c

/-- Checks if a natural number is divisible by 5, 6, and 7 --/
def isDivisibleBy567 (n : Nat) : Prop :=
  n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0

/-- The set of valid six-digit numbers --/
def ValidNumbers : Set Nat :=
  {387000, 387210, 387420, 387630, 387840}

/-- Theorem stating that the ValidNumbers are the only six-digit numbers
    of the form 387abc that are divisible by 5, 6, and 7 --/
theorem valid_numbers_are_unique :
  ∀ a b c : Nat, a < 10 ∧ b < 10 ∧ c < 10 →
  isDivisibleBy567 (SixDigitNumber a b c) ↔ SixDigitNumber a b c ∈ ValidNumbers :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_are_unique_l3354_335406


namespace NUMINAMATH_CALUDE_min_organizer_handshakes_l3354_335428

/-- The number of handshakes between players in a chess tournament where each player plays against every other player exactly once -/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of handshakes including those of the organizer -/
def total_handshakes (n : ℕ) (k : ℕ) : ℕ := player_handshakes n + k

/-- Theorem stating that the minimum number of organizer handshakes is 0 given 406 total handshakes -/
theorem min_organizer_handshakes :
  ∃ (n : ℕ), total_handshakes n 0 = 406 ∧ 
  ∀ (m k : ℕ), total_handshakes m k = 406 → k ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_organizer_handshakes_l3354_335428


namespace NUMINAMATH_CALUDE_corresponding_angles_equal_l3354_335415

/-- Given two triangles ABC and A₁B₁C₁, where for each pair of corresponding angles,
    either the angles are equal or their sum is 180°, all corresponding angles are equal. -/
theorem corresponding_angles_equal 
  (α β γ α₁ β₁ γ₁ : ℝ) 
  (triangle_ABC : α + β + γ = 180)
  (triangle_A₁B₁C₁ : α₁ + β₁ + γ₁ = 180)
  (h1 : α = α₁ ∨ α + α₁ = 180)
  (h2 : β = β₁ ∨ β + β₁ = 180)
  (h3 : γ = γ₁ ∨ γ + γ₁ = 180) :
  α = α₁ ∧ β = β₁ ∧ γ = γ₁ := by
  sorry

end NUMINAMATH_CALUDE_corresponding_angles_equal_l3354_335415


namespace NUMINAMATH_CALUDE_large_cheese_block_volume_l3354_335470

/-- Represents the dimensions and volume of a cheese block -/
structure CheeseBlock where
  width : ℝ
  depth : ℝ
  length : ℝ
  volume : ℝ

/-- Theorem: Volume of a large cheese block -/
theorem large_cheese_block_volume
  (normal : CheeseBlock)
  (large : CheeseBlock)
  (h1 : normal.volume = 3)
  (h2 : large.width = 2 * normal.width)
  (h3 : large.depth = 2 * normal.depth)
  (h4 : large.length = 3 * normal.length)
  (h5 : large.volume = large.width * large.depth * large.length) :
  large.volume = 36 := by
  sorry

#check large_cheese_block_volume

end NUMINAMATH_CALUDE_large_cheese_block_volume_l3354_335470


namespace NUMINAMATH_CALUDE_sine_graph_shift_l3354_335497

theorem sine_graph_shift (x : ℝ) :
  2 * Real.sin (3 * (x - 5 * π / 18) + π / 2) = 2 * Real.sin (3 * x - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l3354_335497


namespace NUMINAMATH_CALUDE_sqrt_two_minus_half_sqrt_two_equals_half_sqrt_two_l3354_335405

theorem sqrt_two_minus_half_sqrt_two_equals_half_sqrt_two :
  Real.sqrt 2 - (Real.sqrt 2) / 2 = (Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_half_sqrt_two_equals_half_sqrt_two_l3354_335405


namespace NUMINAMATH_CALUDE_sum_of_cubes_special_case_l3354_335443

theorem sum_of_cubes_special_case (x y : ℝ) (h1 : x + y = 1) (h2 : x * y = 1) : 
  x^3 + y^3 = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_special_case_l3354_335443


namespace NUMINAMATH_CALUDE_petya_final_amount_l3354_335458

/-- Represents the juice distribution problem between Petya and Masha -/
structure JuiceDistribution where
  total : ℝ
  petya_initial : ℝ
  masha_initial : ℝ
  transferred : ℝ
  h_total : total = 10
  h_initial_sum : petya_initial + masha_initial = total
  h_after_transfer : petya_initial + transferred = 3 * (masha_initial - transferred)
  h_masha_reduction : masha_initial - transferred = (1/3) * masha_initial

/-- Theorem stating that Petya's final amount of juice is 7.5 liters -/
theorem petya_final_amount (jd : JuiceDistribution) : 
  jd.petya_initial + jd.transferred = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_petya_final_amount_l3354_335458


namespace NUMINAMATH_CALUDE_min_value_of_function_l3354_335430

theorem min_value_of_function (x : ℝ) (h : x > 5/4) :
  ∃ (y : ℝ), y = 4*x - 1 + 1/(4*x - 5) ∧ y ≥ 6 ∧ (∃ (x₀ : ℝ), x₀ > 5/4 ∧ 4*x₀ - 1 + 1/(4*x₀ - 5) = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3354_335430


namespace NUMINAMATH_CALUDE_a_5_equals_10_l3354_335423

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

theorem a_5_equals_10 (a : ℕ → ℕ) (h1 : arithmetic_sequence a) (h2 : a 1 = 2) :
  a 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_a_5_equals_10_l3354_335423


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3354_335409

/-- Proves that the speed of a boat in still water is 12 mph given certain conditions -/
theorem boat_speed_in_still_water 
  (distance : ℝ) 
  (downstream_time : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : distance = 45) 
  (h2 : downstream_time = 3)
  (h3 : downstream_speed = distance / downstream_time)
  (h4 : ∃ (current_speed : ℝ), downstream_speed = 12 + current_speed) :
  12 = (12 : ℝ) := by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3354_335409


namespace NUMINAMATH_CALUDE_trig_identity_l3354_335472

theorem trig_identity : 
  Real.sin (-1071 * π / 180) * Real.sin (99 * π / 180) + 
  Real.sin (-171 * π / 180) * Real.sin (-261 * π / 180) + 
  Real.tan (-1089 * π / 180) * Real.tan (-540 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3354_335472


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3354_335475

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_z : ℝ), ∀ (z : ℝ), z = 4 * x^2 + 8 * x + 16 → z ≥ min_z ∧ ∃ (x₀ : ℝ), 4 * x₀^2 + 8 * x₀ + 16 = min_z :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3354_335475


namespace NUMINAMATH_CALUDE_factorial_calculation_l3354_335416

theorem factorial_calculation : (Nat.factorial 15) / ((Nat.factorial 7) * (Nat.factorial 8)) * 2 = 1286 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l3354_335416
