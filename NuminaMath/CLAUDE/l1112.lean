import Mathlib

namespace NUMINAMATH_CALUDE_nathaniel_tickets_l1112_111226

/-- Given a person with initial tickets who gives a fixed number of tickets to each of their friends,
    calculate the number of remaining tickets. -/
def remaining_tickets (initial : ℕ) (given_per_friend : ℕ) (num_friends : ℕ) : ℕ :=
  initial - given_per_friend * num_friends

/-- Theorem stating that given 11 initial tickets, giving 2 tickets to each of 4 friends
    results in 3 remaining tickets. -/
theorem nathaniel_tickets : remaining_tickets 11 2 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_nathaniel_tickets_l1112_111226


namespace NUMINAMATH_CALUDE_alex_not_reading_probability_l1112_111276

theorem alex_not_reading_probability (p : ℚ) (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_alex_not_reading_probability_l1112_111276


namespace NUMINAMATH_CALUDE_equation_solution_l1112_111201

/-- The overall substitution method for solving quadratic equations -/
def overall_substitution_method (a b c : ℝ) : Set ℝ :=
  { x | ∃ y, y^2 + b*y + c = 0 ∧ a*x + b = y }

/-- The equation (2x-5)^2 - 2(2x-5) - 3 = 0 has solutions x₁ = 2 and x₂ = 4 -/
theorem equation_solution : 
  overall_substitution_method 2 (-5) (-3) = {2, 4} := by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l1112_111201


namespace NUMINAMATH_CALUDE_no_negative_exponents_l1112_111254

theorem no_negative_exponents (a b c d : ℤ) (h : (4:ℝ)^a + (4:ℝ)^b = (8:ℝ)^c + (27:ℝ)^d) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_no_negative_exponents_l1112_111254


namespace NUMINAMATH_CALUDE_units_digit_of_4_pow_8_cubed_l1112_111255

theorem units_digit_of_4_pow_8_cubed (n : ℕ) : n = 4^(8^3) → n % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_4_pow_8_cubed_l1112_111255


namespace NUMINAMATH_CALUDE_problem_solution_l1112_111260

theorem problem_solution (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (6 * x) * Real.sqrt (5 * x) * Real.sqrt (20 * x) = 20) : 
  x = (1 / 18) ^ (1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1112_111260


namespace NUMINAMATH_CALUDE_parabola_focus_l1112_111248

-- Define the parabola
def parabola (x : ℝ) : ℝ := -4 * x^2 - 8 * x + 1

-- Define the focus of a parabola
def focus (a b c : ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem parabola_focus :
  focus (-4) (-8) 1 = (-1, 79/16) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1112_111248


namespace NUMINAMATH_CALUDE_dot_product_range_l1112_111241

-- Define the parabola M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

-- Define the circle C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 3)^2 = 4}

-- Define the center of the circle
def center : ℝ × ℝ := (0, 3)

-- Define a point on the parabola
def P : ℝ × ℝ := sorry

-- Define tangent points A and B on the circle
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Axiom: P is on the parabola M
axiom P_on_M : P ∈ M

-- Axiom: A and B are on the circle C
axiom A_on_C : A ∈ C
axiom B_on_C : B ∈ C

-- Axiom: PA and PB are tangent to C
axiom PA_tangent : sorry
axiom PB_tangent : sorry

-- Define vectors CA and CB
def CA : ℝ × ℝ := (A.1 - center.1, A.2 - center.2)
def CB : ℝ × ℝ := (B.1 - center.1, B.2 - center.2)

-- Define dot product of CA and CB
def dot_product : ℝ := CA.1 * CB.1 + CA.2 * CB.2

-- Theorem: The dot product of CA and CB is in the range [0, 4)
theorem dot_product_range : 0 ≤ dot_product ∧ dot_product < 4 := by sorry

end NUMINAMATH_CALUDE_dot_product_range_l1112_111241


namespace NUMINAMATH_CALUDE_first_part_length_l1112_111207

/-- Proves that given a 50 km trip where the first part is traveled at 60 km/h,
    the second part at 30 km/h, and the average speed is 40 km/h,
    the length of the first part of the trip is 25 km. -/
theorem first_part_length
  (total_distance : ℝ)
  (speed_first_part : ℝ)
  (speed_second_part : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 50)
  (h2 : speed_first_part = 60)
  (h3 : speed_second_part = 30)
  (h4 : average_speed = 40)
  : ∃ (x : ℝ),
    x / speed_first_part + (total_distance - x) / speed_second_part =
      total_distance / average_speed ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_first_part_length_l1112_111207


namespace NUMINAMATH_CALUDE_trihedral_angle_inequalities_l1112_111281

structure TrihedralAngle where
  SA : Real
  SB : Real
  SC : Real
  α : Real
  β : Real
  γ : Real
  ASB : Real
  BSC : Real
  CSA : Real

def is_acute_dihedral (t : TrihedralAngle) : Prop := sorry

theorem trihedral_angle_inequalities (t : TrihedralAngle) :
  t.α + t.β + t.γ ≤ t.ASB + t.BSC + t.CSA ∧
  (is_acute_dihedral t → t.α + t.β + t.γ ≥ (t.ASB + t.BSC + t.CSA) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trihedral_angle_inequalities_l1112_111281


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1112_111256

def R : Set ℝ := Set.univ

def A : Set ℝ := {1, 2, 3, 4, 5}

def B : Set ℝ := {x : ℝ | x * (4 - x) < 0}

theorem intersection_complement_equality :
  A ∩ (Set.compl B) = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1112_111256


namespace NUMINAMATH_CALUDE_min_y_value_l1112_111216

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 10*x + 36*y) : 
  ∀ z : ℝ, (∃ w : ℝ, w^2 + z^2 = 10*w + 36*z) → y ≤ z → -7 ≤ y :=
sorry

end NUMINAMATH_CALUDE_min_y_value_l1112_111216


namespace NUMINAMATH_CALUDE_tan_ratio_problem_l1112_111217

theorem tan_ratio_problem (x : Real) (h : Real.tan (x + π / 4) = 2) :
  (Real.tan x) / (Real.tan (2 * x)) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_problem_l1112_111217


namespace NUMINAMATH_CALUDE_decagon_ratio_l1112_111205

/-- Represents a decagon with specific properties -/
structure Decagon where
  total_area : ℝ
  squares_below : ℝ
  trapezoid_base1 : ℝ
  trapezoid_base2 : ℝ
  bisector : ℝ → ℝ → Prop

/-- Theorem stating the ratio of XQ to QY in the given decagon -/
theorem decagon_ratio (d : Decagon)
    (h_area : d.total_area = 12)
    (h_squares : d.squares_below = 2)
    (h_base1 : d.trapezoid_base1 = 3)
    (h_base2 : d.trapezoid_base2 = 6)
    (h_bisect : d.bisector (d.squares_below + (d.trapezoid_base1 + d.trapezoid_base2) / 2 * h) (d.total_area / 2))
    (x y : ℝ)
    (h_xy : x + y = 6)
    (h_bisect_xy : d.bisector x y) :
    x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_ratio_l1112_111205


namespace NUMINAMATH_CALUDE_hannah_spending_l1112_111222

def sweatshirt_price : ℝ := 15
def tshirt_price : ℝ := 10
def sock_price : ℝ := 5
def jacket_price : ℝ := 50
def discount_rate : ℝ := 0.1

def num_sweatshirts : ℕ := 3
def num_tshirts : ℕ := 2
def num_socks : ℕ := 4

def total_before_discount : ℝ :=
  sweatshirt_price * num_sweatshirts +
  tshirt_price * num_tshirts +
  sock_price * num_socks +
  jacket_price

def discount_amount : ℝ := total_before_discount * discount_rate

def total_after_discount : ℝ := total_before_discount - discount_amount

theorem hannah_spending :
  total_after_discount = 121.50 := by sorry

end NUMINAMATH_CALUDE_hannah_spending_l1112_111222


namespace NUMINAMATH_CALUDE_sandy_first_shop_books_l1112_111277

/-- Represents the problem of Sandy's book purchases -/
def SandyBookProblem (first_shop_books : ℕ) : Prop :=
  let total_spent : ℕ := 2160
  let second_shop_books : ℕ := 55
  let average_price : ℕ := 18
  (total_spent : ℚ) / (first_shop_books + second_shop_books : ℚ) = average_price

/-- Proves that Sandy bought 65 books from the first shop -/
theorem sandy_first_shop_books :
  SandyBookProblem 65 := by sorry

end NUMINAMATH_CALUDE_sandy_first_shop_books_l1112_111277


namespace NUMINAMATH_CALUDE_geometric_sequence_arithmetic_mean_l1112_111203

/-- Given a geometric sequence {a_n} with common ratio q = -2 and a_3 * a_7 = 4 * a_4,
    prove that the arithmetic mean of a_8 and a_11 is -56. -/
theorem geometric_sequence_arithmetic_mean 
  (a : ℕ → ℝ) -- The geometric sequence
  (h1 : ∀ n, a (n + 1) = a n * (-2)) -- Common ratio q = -2
  (h2 : a 3 * a 7 = 4 * a 4) -- Given condition
  : (a 8 + a 11) / 2 = -56 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_arithmetic_mean_l1112_111203


namespace NUMINAMATH_CALUDE_gdp_growth_time_l1112_111236

theorem gdp_growth_time (initial_gdp : ℝ) (growth_rate : ℝ) (target_gdp : ℝ) :
  initial_gdp = 8000 →
  growth_rate = 0.1 →
  target_gdp = 16000 →
  (∀ n : ℕ, n < 5 → initial_gdp * (1 + growth_rate) ^ n ≤ target_gdp) ∧
  initial_gdp * (1 + growth_rate) ^ 5 > target_gdp :=
by sorry

end NUMINAMATH_CALUDE_gdp_growth_time_l1112_111236


namespace NUMINAMATH_CALUDE_perpendicular_iff_m_eq_neg_one_l1112_111247

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line: x + y = 0 -/
def line1 : Line := ⟨1, 1, 0⟩

/-- The second line: x + my = 0 -/
def line2 (m : ℝ) : Line := ⟨1, m, 0⟩

/-- Theorem: The lines x+y=0 and x+my=0 are perpendicular if and only if m=-1 -/
theorem perpendicular_iff_m_eq_neg_one :
  ∀ m : ℝ, perpendicular line1 (line2 m) ↔ m = -1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_iff_m_eq_neg_one_l1112_111247


namespace NUMINAMATH_CALUDE_existence_of_n_l1112_111269

theorem existence_of_n (p : ℕ) (a k : ℕ+) (h_prime : Nat.Prime p) 
  (h_bound : p^(a : ℕ) < k ∧ k < 2 * p^(a : ℕ)) : 
  ∃ n : ℕ+, n < p^(2 * (a : ℕ)) ∧ 
    (Nat.choose n k : ZMod (p^(a : ℕ))) = n ∧ 
    (n : ZMod (p^(a : ℕ))) = k :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_l1112_111269


namespace NUMINAMATH_CALUDE_car_distance_traveled_l1112_111292

/-- Calculates the distance traveled by a car given its speed and time -/
def distanceTraveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- The actual speed of the car in km/h -/
def actualSpeed : ℚ := 35

/-- The fraction of the actual speed at which the car is traveling -/
def speedFraction : ℚ := 5 / 7

/-- The time the car travels in hours -/
def travelTime : ℚ := 126 / 75

/-- The theorem stating the distance traveled by the car -/
theorem car_distance_traveled :
  distanceTraveled (speedFraction * actualSpeed) travelTime = 42 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l1112_111292


namespace NUMINAMATH_CALUDE_intersection_when_a_is_half_intersection_empty_iff_l1112_111286

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem 1: When a = 1/2, A ∩ B = {x | 0 < x < 1}
theorem intersection_when_a_is_half : 
  A (1/2) ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: A ∩ B = ∅ if and only if a ≤ -1/2 or a ≥ 2
theorem intersection_empty_iff : 
  ∀ a : ℝ, A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_half_intersection_empty_iff_l1112_111286


namespace NUMINAMATH_CALUDE_perpendicular_length_l1112_111219

/-- Given a triangle ABC with angle ABC = 135°, AB = 2, and BC = 5,
    if perpendiculars are constructed to AB at A and to BC at C meeting at point D,
    then CD = 5√2 -/
theorem perpendicular_length (A B C D : ℝ × ℝ) : 
  let angleABC : ℝ := 135 * π / 180
  let AB : ℝ := 2
  let BC : ℝ := 5
  ∀ (perpAB : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0)
    (perpBC : (D.1 - C.1) * (B.1 - C.1) + (D.2 - C.2) * (B.2 - C.2) = 0),
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_length_l1112_111219


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_l1112_111209

theorem sqrt_sum_squares : Real.sqrt (3^2) + (Real.sqrt 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_l1112_111209


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1112_111261

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (f = fun x ↦ x - 1) ∨ (f = fun x ↦ -x - 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1112_111261


namespace NUMINAMATH_CALUDE_function_property_l1112_111266

theorem function_property (f : ℕ → ℕ) :
  (∀ m n : ℕ, (m^2 + f n) ∣ (m * f m + n)) →
  (∀ n : ℕ, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l1112_111266


namespace NUMINAMATH_CALUDE_c_leq_one_sufficient_not_necessary_l1112_111253

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

def sequence_a (c : ℝ) (n : ℕ) : ℝ :=
  |n - c|

theorem c_leq_one_sufficient_not_necessary (c : ℝ) :
  (c ≤ 1 → is_increasing (sequence_a c)) ∧
  ¬(is_increasing (sequence_a c) → c ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_c_leq_one_sufficient_not_necessary_l1112_111253


namespace NUMINAMATH_CALUDE_subtract_from_square_l1112_111215

theorem subtract_from_square (n : ℕ) (h : n = 17) : n^2 - n = 272 := by
  sorry

end NUMINAMATH_CALUDE_subtract_from_square_l1112_111215


namespace NUMINAMATH_CALUDE_sum_base8_equals_1207_l1112_111290

/-- Converts a base-8 number represented as a list of digits to its decimal equivalent -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 8 * acc + d) 0

/-- Converts a decimal number to its base-8 representation as a list of digits -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimalToBase8 (n / 8)

/-- The sum of 527₈, 165₈, and 273₈ in base 8 is equal to 1207₈ -/
theorem sum_base8_equals_1207 :
  let a := base8ToDecimal [7, 2, 5]
  let b := base8ToDecimal [5, 6, 1]
  let c := base8ToDecimal [3, 7, 2]
  decimalToBase8 (a + b + c) = [7, 0, 2, 1] := by
  sorry

end NUMINAMATH_CALUDE_sum_base8_equals_1207_l1112_111290


namespace NUMINAMATH_CALUDE_correct_fraction_proof_l1112_111296

theorem correct_fraction_proof (x y : ℚ) : 
  (5 / 6 : ℚ) * 288 = x / y * 288 + 150 → x / y = 5 / 32 := by
sorry

end NUMINAMATH_CALUDE_correct_fraction_proof_l1112_111296


namespace NUMINAMATH_CALUDE_no_integer_solution_l1112_111289

theorem no_integer_solution : ¬ ∃ (x y : ℤ), 2 * x + 6 * y = 91 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1112_111289


namespace NUMINAMATH_CALUDE_betty_age_l1112_111299

theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 10) :
  betty = 5 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l1112_111299


namespace NUMINAMATH_CALUDE_equal_area_condition_l1112_111211

/-- Represents a configuration of five unit squares in the coordinate plane -/
structure SquareConfiguration where
  /-- The lower left corner of the configuration is at the origin -/
  lower_left_at_origin : Bool

/-- Represents a line in the coordinate plane -/
structure Line where
  /-- The x-coordinate of the point where the line intersects the x-axis -/
  x_intercept : ℝ
  /-- The coordinates of the other point the line passes through -/
  other_point : ℝ × ℝ

/-- Calculates the area of the region divided by the line -/
def divided_area (config : SquareConfiguration) (line : Line) : ℝ :=
  sorry

theorem equal_area_condition (config : SquareConfiguration) (line : Line) :
  config.lower_left_at_origin = true ∧
  line.other_point = (3, 3) →
  (divided_area config line = 5 / 2 ↔ line.x_intercept = 2 / 3) :=
sorry

end NUMINAMATH_CALUDE_equal_area_condition_l1112_111211


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1112_111279

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 0 ∧ x₂ = 3/2) ∧ 
  (∀ x : ℝ, 2*x^2 - 3*x = 0 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1112_111279


namespace NUMINAMATH_CALUDE_no_fraction_satisfies_conditions_l1112_111284

theorem no_fraction_satisfies_conditions : ¬∃ (a b n : ℕ), 
  (a < b) ∧ 
  (n < a) ∧ 
  (n < b) ∧ 
  ((a + n : ℚ) / (b + n)) > (3 / 2) * (a / b) ∧
  ((a - n : ℚ) / (b - n)) > (1 / 2) * (a / b) := by
  sorry

end NUMINAMATH_CALUDE_no_fraction_satisfies_conditions_l1112_111284


namespace NUMINAMATH_CALUDE_plant_prices_and_minimum_cost_l1112_111220

/-- The price of a pot of green radish -/
def green_radish_price : ℝ := 4

/-- The price of a pot of spider plant -/
def spider_plant_price : ℝ := 12

/-- The total number of pots to be purchased -/
def total_pots : ℕ := 120

/-- The number of spider plant pots that minimizes the total cost -/
def optimal_spider_pots : ℕ := 80

/-- The number of green radish pots that minimizes the total cost -/
def optimal_green_radish_pots : ℕ := 40

theorem plant_prices_and_minimum_cost :
  (green_radish_price + spider_plant_price = 16) ∧
  (80 / green_radish_price = 2 * (120 / spider_plant_price)) ∧
  (optimal_spider_pots + optimal_green_radish_pots = total_pots) ∧
  (optimal_green_radish_pots ≤ optimal_spider_pots / 2) ∧
  (∀ a : ℕ, a + (total_pots - a) = total_pots →
    (total_pots - a) ≤ a / 2 →
    spider_plant_price * a + green_radish_price * (total_pots - a) ≥
    spider_plant_price * optimal_spider_pots + green_radish_price * optimal_green_radish_pots) :=
by sorry

end NUMINAMATH_CALUDE_plant_prices_and_minimum_cost_l1112_111220


namespace NUMINAMATH_CALUDE_circle_center_on_line_max_ab_l1112_111234

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x - b*y + 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 2)

theorem circle_center_on_line_max_ab :
  ∀ (a b : ℝ),
  line_equation a b (circle_center.1) (circle_center.2) →
  a * b ≤ 1/8 ∧
  ∀ (ε : ℝ), ε > 0 → ∃ (a' b' : ℝ), 
    line_equation a' b' (circle_center.1) (circle_center.2) ∧
    a' * b' > 1/8 - ε :=
by sorry

end NUMINAMATH_CALUDE_circle_center_on_line_max_ab_l1112_111234


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l1112_111271

theorem rectangular_box_surface_area 
  (x y z : ℝ) 
  (h1 : 4 * x + 4 * y + 4 * z = 160) 
  (h2 : Real.sqrt (x^2 + y^2 + z^2) = 25) : 
  2 * (x * y + y * z + z * x) = 975 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l1112_111271


namespace NUMINAMATH_CALUDE_total_meals_is_48_l1112_111264

/-- Represents the number of entree options --/
def num_entrees : ℕ := 4

/-- Represents the number of drink options --/
def num_drinks : ℕ := 4

/-- Represents the number of dessert options (including "no dessert") --/
def num_desserts : ℕ := 3

/-- Calculates the total number of possible meal combinations --/
def total_meals : ℕ := num_entrees * num_drinks * num_desserts

/-- Theorem stating that the total number of possible meals is 48 --/
theorem total_meals_is_48 : total_meals = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_meals_is_48_l1112_111264


namespace NUMINAMATH_CALUDE_contrapositive_proof_l1112_111297

theorem contrapositive_proof : 
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ 
  (∀ a b : ℝ, a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_proof_l1112_111297


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l1112_111229

/-- 
Given a quadratic expression of the form 3x^2 + nx + 108, 
this theorem states that the largest value of n for which 
the expression can be factored as the product of two linear 
factors with integer coefficients is 325.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (A B : ℤ), 
    (3 * A = 1 ∧ B = 108) → 
    (∀ (x : ℤ), 3 * x^2 + n * x + 108 = (3 * x + A) * (x + B)) ∧
    (∀ (m : ℤ), (∃ (C D : ℤ), ∀ (x : ℤ), 
      3 * x^2 + m * x + 108 = (3 * x + C) * (x + D)) → 
      m ≤ n)) ∧
  (∀ (n : ℤ), (∀ (A B : ℤ), 
    (3 * A = 1 ∧ B = 108) → 
    (∀ (x : ℤ), 3 * x^2 + n * x + 108 = (3 * x + A) * (x + B)) ∧
    (∀ (m : ℤ), (∃ (C D : ℤ), ∀ (x : ℤ), 
      3 * x^2 + m * x + 108 = (3 * x + C) * (x + D)) → 
      m ≤ n)) → 
  n = 325) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l1112_111229


namespace NUMINAMATH_CALUDE_class_selection_ways_l1112_111206

def total_classes : ℕ := 10
def advanced_classes : ℕ := 6
def intro_classes : ℕ := 4
def classes_to_choose : ℕ := 5
def min_advanced : ℕ := 3

theorem class_selection_ways :
  (Nat.choose advanced_classes 3 * Nat.choose intro_classes 2) +
  (Nat.choose advanced_classes 4 * Nat.choose intro_classes 1) +
  (Nat.choose advanced_classes 5) = 186 := by
  sorry

end NUMINAMATH_CALUDE_class_selection_ways_l1112_111206


namespace NUMINAMATH_CALUDE_polynomial_inequality_l1112_111274

theorem polynomial_inequality (a b c : ℝ) 
  (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l1112_111274


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l1112_111212

def days_in_week : ℕ := 7

def probability_sun : ℝ := 0.3
def probability_light_rain : ℝ := 0.5
def probability_heavy_rain : ℝ := 0.2

def light_rain_amount : ℝ := 3
def heavy_rain_amount : ℝ := 8

def daily_expected_rainfall : ℝ :=
  probability_sun * 0 + 
  probability_light_rain * light_rain_amount + 
  probability_heavy_rain * heavy_rain_amount

theorem expected_weekly_rainfall : 
  days_in_week * daily_expected_rainfall = 21.7 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l1112_111212


namespace NUMINAMATH_CALUDE_tan_product_identity_l1112_111267

theorem tan_product_identity : (1 + Real.tan (18 * π / 180)) * (1 + Real.tan (27 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_identity_l1112_111267


namespace NUMINAMATH_CALUDE_colin_speed_l1112_111208

/-- The skipping speeds of Bruce, Tony, Brandon, and Colin -/
structure SkippingSpeeds where
  bruce : ℝ
  tony : ℝ
  brandon : ℝ
  colin : ℝ

/-- The conditions of the skipping speeds problem -/
def skipping_conditions (s : SkippingSpeeds) : Prop :=
  s.bruce = 1 ∧
  s.tony = 2 * s.bruce ∧
  s.brandon = (1/3) * s.tony ∧
  s.colin = 6 * s.brandon

/-- Theorem stating that Colin's skipping speed is 4 miles per hour -/
theorem colin_speed (s : SkippingSpeeds) 
  (h : skipping_conditions s) : s.colin = 4 := by
  sorry


end NUMINAMATH_CALUDE_colin_speed_l1112_111208


namespace NUMINAMATH_CALUDE_total_apples_is_sixteen_l1112_111265

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- The total number of apples picked -/
def total_apples : ℕ := mike_apples + nancy_apples + keith_apples

theorem total_apples_is_sixteen : total_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_sixteen_l1112_111265


namespace NUMINAMATH_CALUDE_three_pipes_used_l1112_111225

def tank_filling_problem (rate_a rate_b rate_c : ℝ) : Prop :=
  let total_rate := rate_a + rate_b + rate_c
  rate_c = 2 * rate_b ∧
  rate_b = 2 * rate_a ∧
  rate_a = 1 / 70 ∧
  total_rate = 1 / 10

theorem three_pipes_used (rate_a rate_b rate_c : ℝ) 
  (h : tank_filling_problem rate_a rate_b rate_c) : 
  ∃ (n : ℕ), n = 3 ∧ n > 0 := by
  sorry

#check three_pipes_used

end NUMINAMATH_CALUDE_three_pipes_used_l1112_111225


namespace NUMINAMATH_CALUDE_inequality_proof_l1112_111227

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2)/(x^2 + y*z) + (y^2 + 2*z^2 + 2*x^2)/(y^2 + z*x) + (z^2 + 2*x^2 + 2*y^2)/(z^2 + x*y) > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1112_111227


namespace NUMINAMATH_CALUDE_max_p_value_l1112_111238

theorem max_p_value (p q r : ℝ) (sum_eq : p + q + r = 10) (prod_sum_eq : p*q + p*r + q*r = 25) :
  p ≤ 20/3 ∧ ∃ q r : ℝ, p = 20/3 ∧ p + q + r = 10 ∧ p*q + p*r + q*r = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_p_value_l1112_111238


namespace NUMINAMATH_CALUDE_equation_solution_l1112_111283

theorem equation_solution : ∃ (a b c d : ℕ+), 
  2014 = (a.val ^ 2 + b.val ^ 2) * (c.val ^ 3 - d.val ^ 3) ∧ 
  a.val = 5 ∧ b.val = 9 ∧ c.val = 3 ∧ d.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1112_111283


namespace NUMINAMATH_CALUDE_count_less_than_04_l1112_111270

def numbers : Finset ℚ := {0.8, 1/2, 0.3, 1/3}

theorem count_less_than_04 : Finset.card (numbers.filter (λ x => x < 0.4)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_less_than_04_l1112_111270


namespace NUMINAMATH_CALUDE_laptop_price_theorem_l1112_111245

/-- The sticker price of the laptop -/
def stickerPrice : ℝ := 900

/-- The price at Store P -/
def storePPrice (price : ℝ) : ℝ := 0.8 * price - 120

/-- The price at Store Q -/
def storeQPrice (price : ℝ) : ℝ := 0.7 * price

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem laptop_price_theorem :
  storeQPrice stickerPrice - storePPrice stickerPrice = 30 := by
  sorry

#check laptop_price_theorem

end NUMINAMATH_CALUDE_laptop_price_theorem_l1112_111245


namespace NUMINAMATH_CALUDE_seven_eighths_of_64_l1112_111202

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_64_l1112_111202


namespace NUMINAMATH_CALUDE_vector_angle_theorem_l1112_111287

-- Define a type for 3D vectors
def Vector3D := ℝ × ℝ × ℝ

-- Define a function to calculate the angle between two vectors
noncomputable def angle (v1 v2 : Vector3D) : ℝ := sorry

-- Define a predicate for non-zero vectors
def nonzero (v : Vector3D) : Prop := v ≠ (0, 0, 0)

theorem vector_angle_theorem (vectors : Fin 30 → Vector3D) 
  (h : ∀ i, nonzero (vectors i)) : 
  ∃ i j, i ≠ j ∧ angle (vectors i) (vectors j) < Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_vector_angle_theorem_l1112_111287


namespace NUMINAMATH_CALUDE_x_intercept_of_specific_line_l1112_111240

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ := sorry

/-- The specific line passing through (2, -2) and (6, 10) -/
def specific_line : Line := { x₁ := 2, y₁ := -2, x₂ := 6, y₂ := 10 }

theorem x_intercept_of_specific_line :
  x_intercept specific_line = 8/3 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_specific_line_l1112_111240


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1112_111218

theorem gcd_of_specific_numbers : 
  let m : ℕ := 55555555
  let n : ℕ := 111111111
  Nat.gcd m n = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1112_111218


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l1112_111242

theorem polygon_interior_angle_sum (n : ℕ) (h : n > 2) :
  (360 / 72 : ℝ) = n →
  (n - 2) * 180 = 540 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l1112_111242


namespace NUMINAMATH_CALUDE_abs_five_minus_e_equals_five_minus_e_l1112_111230

theorem abs_five_minus_e_equals_five_minus_e :
  |5 - Real.exp 1| = 5 - Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_abs_five_minus_e_equals_five_minus_e_l1112_111230


namespace NUMINAMATH_CALUDE_bob_marathon_preparation_l1112_111224

/-- The total miles Bob runs in 3 days -/
def total_miles : ℝ := 70

/-- Miles run on day one -/
def day_one_miles : ℝ := 0.2 * total_miles

/-- Miles run on day two -/
def day_two_miles : ℝ := 0.5 * (total_miles - day_one_miles)

/-- Miles run on day three -/
def day_three_miles : ℝ := 28

theorem bob_marathon_preparation :
  day_one_miles + day_two_miles + day_three_miles = total_miles :=
by sorry

end NUMINAMATH_CALUDE_bob_marathon_preparation_l1112_111224


namespace NUMINAMATH_CALUDE_new_cube_weight_l1112_111262

/-- Given a cube of weight 3 pounds and density D, prove that a new cube with sides twice as long
    and density 1.25D will weigh 30 pounds. -/
theorem new_cube_weight (D : ℝ) (D_pos : D > 0) : 
  let original_weight : ℝ := 3
  let original_volume : ℝ := original_weight / D
  let new_volume : ℝ := 8 * original_volume
  let new_density : ℝ := 1.25 * D
  new_density * new_volume = 30 := by
  sorry


end NUMINAMATH_CALUDE_new_cube_weight_l1112_111262


namespace NUMINAMATH_CALUDE_walk_distance_proof_l1112_111250

/-- The distance Rajesh and Hiro walked together -/
def distance_together : ℝ := 7

/-- Hiro's walking distance -/
def hiro_distance : ℝ := distance_together

/-- Rajesh's walking distance -/
def rajesh_distance : ℝ := 18

theorem walk_distance_proof :
  (4 * hiro_distance - 10 = rajesh_distance) →
  distance_together = 7 := by
  sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l1112_111250


namespace NUMINAMATH_CALUDE_remainder_theorem_l1112_111239

theorem remainder_theorem (n : ℤ) : n % 9 = 5 → (4 * n - 6) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1112_111239


namespace NUMINAMATH_CALUDE_pastry_price_is_18_l1112_111273

/-- The price of a pastry satisfies the given conditions -/
theorem pastry_price_is_18 
  (usual_pastries : ℕ) 
  (usual_bread : ℕ) 
  (today_pastries : ℕ) 
  (today_bread : ℕ) 
  (bread_price : ℕ) 
  (daily_difference : ℕ) 
  (h1 : usual_pastries = 20) 
  (h2 : usual_bread = 10) 
  (h3 : today_pastries = 14) 
  (h4 : today_bread = 25) 
  (h5 : bread_price = 4) 
  (h6 : daily_difference = 48) : 
  ∃ (pastry_price : ℕ), 
    pastry_price = 18 ∧ 
    usual_pastries * pastry_price + usual_bread * bread_price - 
    (today_pastries * pastry_price + today_bread * bread_price) = daily_difference :=
by sorry

end NUMINAMATH_CALUDE_pastry_price_is_18_l1112_111273


namespace NUMINAMATH_CALUDE_largest_s_value_l1112_111295

theorem largest_s_value : ∃ (s : ℝ), 
  (∀ (t : ℝ), (15 * t^2 - 40 * t + 18) / (4 * t - 3) + 6 * t = 7 * t - 1 → t ≤ s) ∧
  (15 * s^2 - 40 * s + 18) / (4 * s - 3) + 6 * s = 7 * s - 1 ∧
  s = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_s_value_l1112_111295


namespace NUMINAMATH_CALUDE_socorro_training_days_l1112_111268

/-- Calculates the number of days required to complete a training program. -/
def trainingDays (totalHours : ℕ) (dailyMinutes : ℕ) : ℕ :=
  (totalHours * 60) / dailyMinutes

/-- Proves that given 5 hours of total training time and 30 minutes of daily training,
    it takes 10 days to complete the training. -/
theorem socorro_training_days :
  trainingDays 5 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_socorro_training_days_l1112_111268


namespace NUMINAMATH_CALUDE_first_number_remainder_l1112_111298

/-- A permutation of numbers from 1 to 2023 -/
def Arrangement := Fin 2023 → Fin 2023

/-- Property that any three numbers with one in between have different remainders when divided by 3 -/
def ValidArrangement (arr : Arrangement) : Prop :=
  ∀ i : Fin 2020, (arr i % 3) ≠ (arr (i + 2) % 3) ∧ (arr i % 3) ≠ (arr (i + 4) % 3) ∧ (arr (i + 2) % 3) ≠ (arr (i + 4) % 3)

/-- Theorem stating that the first number in a valid arrangement must have remainder 1 when divided by 3 -/
theorem first_number_remainder (arr : Arrangement) (h : ValidArrangement arr) : arr 0 % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_number_remainder_l1112_111298


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1112_111251

theorem complex_modulus_problem (z : ℂ) (x : ℝ) 
  (h1 : z * Complex.I = 2 * Complex.I + x)
  (h2 : z.im = 2) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1112_111251


namespace NUMINAMATH_CALUDE_tips_fraction_of_income_l1112_111280

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the total income of a waitress -/
def totalIncome (w : WaitressIncome) : ℚ :=
  w.salary + w.tips

/-- Theorem: If a waitress's tips are 7/4 of her salary, then 7/11 of her income comes from tips -/
theorem tips_fraction_of_income (w : WaitressIncome) 
  (h : w.tips = (7 : ℚ) / 4 * w.salary) : 
  w.tips / totalIncome w = (7 : ℚ) / 11 := by
  sorry

end NUMINAMATH_CALUDE_tips_fraction_of_income_l1112_111280


namespace NUMINAMATH_CALUDE_min_white_surface_3x3x3_l1112_111278

/-- Represents a cube with unit cubes of different colors --/
structure ColoredCube where
  edge_length : ℕ
  total_units : ℕ
  red_units : ℕ
  white_units : ℕ

/-- Calculates the minimum white surface area fraction for a ColoredCube --/
def min_white_surface_fraction (c : ColoredCube) : ℚ :=
  sorry

/-- Theorem: For a 3x3x3 cube with 21 red and 6 white unit cubes,
    the minimum white surface area fraction is 5/54 --/
theorem min_white_surface_3x3x3 :
  let c : ColoredCube := {
    edge_length := 3,
    total_units := 27,
    red_units := 21,
    white_units := 6
  }
  min_white_surface_fraction c = 5 / 54 := by
  sorry

end NUMINAMATH_CALUDE_min_white_surface_3x3x3_l1112_111278


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1112_111294

theorem triangle_abc_properties (a b c A B C : ℝ) (h1 : c * Real.cos A = 5) 
  (h2 : a * Real.sin C = 4) (h3 : (1/2) * a * b * Real.sin C = 16) : 
  c = Real.sqrt 41 ∧ a + b + c = 13 + Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1112_111294


namespace NUMINAMATH_CALUDE_subtracted_amount_l1112_111235

/-- Given a number N = 200, if 95% of N minus some amount A equals 178, then A must be 12. -/
theorem subtracted_amount (N : ℝ) (A : ℝ) : 
  N = 200 → 0.95 * N - A = 178 → A = 12 := by sorry

end NUMINAMATH_CALUDE_subtracted_amount_l1112_111235


namespace NUMINAMATH_CALUDE_scientific_notation_19672_l1112_111249

theorem scientific_notation_19672 :
  19672 = 1.9672 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_19672_l1112_111249


namespace NUMINAMATH_CALUDE_game_theorem_l1112_111288

/-- Represents the game "What? Where? When?" with given conditions -/
structure Game where
  envelopes : ℕ := 13
  win_points : ℕ := 6
  win_prob : ℚ := 1/2

/-- Expected number of points for a single game -/
def expected_points (g : Game) : ℚ := sorry

/-- Expected number of points over 100 games -/
def expected_points_100 (g : Game) : ℚ := 100 * expected_points g

/-- Probability of an envelope being chosen in a game -/
def envelope_prob (g : Game) : ℚ := sorry

theorem game_theorem (g : Game) :
  expected_points_100 g = 465 ∧ envelope_prob g = 12/13 := by sorry

end NUMINAMATH_CALUDE_game_theorem_l1112_111288


namespace NUMINAMATH_CALUDE_megans_acorns_l1112_111275

/-- The initial number of acorns Megan had -/
def T : ℝ := 20

/-- Theorem stating the conditions and the correct answer for Megan's acorn problem -/
theorem megans_acorns :
  (0.35 * T = 7) ∧ (0.45 * T = 9) ∧ (T = 20) := by
  sorry

#check megans_acorns

end NUMINAMATH_CALUDE_megans_acorns_l1112_111275


namespace NUMINAMATH_CALUDE_intersection_S_T_l1112_111258

def S : Set ℤ := {-4, -3, 6, 7}
def T : Set ℤ := {x | x^2 > 4*x}

theorem intersection_S_T : S ∩ T = {-4, -3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l1112_111258


namespace NUMINAMATH_CALUDE_beetle_speed_l1112_111257

/-- Calculates the speed of a beetle given the ant's distance and the beetle's relative speed -/
theorem beetle_speed (ant_distance : Real) (time_minutes : Real) (beetle_relative_speed : Real) :
  let beetle_distance := ant_distance * (1 - beetle_relative_speed)
  let time_hours := time_minutes / 60
  let speed_km_h := (beetle_distance / 1000) / time_hours
  speed_km_h = 2.55 :=
by
  sorry

#check beetle_speed 600 12 0.15

end NUMINAMATH_CALUDE_beetle_speed_l1112_111257


namespace NUMINAMATH_CALUDE_smallest_equal_packages_l1112_111214

/-- The number of pencils in each pack -/
def pencils_per_pack : ℕ := 10

/-- The number of pencil sharpeners in each pack -/
def sharpeners_per_pack : ℕ := 14

/-- The smallest number of pencil sharpener packages needed -/
def min_sharpener_packages : ℕ := 5

theorem smallest_equal_packages :
  ∃ (pencil_packs : ℕ),
    pencil_packs * pencils_per_pack = min_sharpener_packages * sharpeners_per_pack ∧
    ∀ (k : ℕ), k < min_sharpener_packages →
      ¬∃ (m : ℕ), m * pencils_per_pack = k * sharpeners_per_pack :=
by sorry

end NUMINAMATH_CALUDE_smallest_equal_packages_l1112_111214


namespace NUMINAMATH_CALUDE_heptagon_angle_measure_l1112_111232

-- Define the heptagon
structure Heptagon where
  G : ℝ
  E : ℝ
  O : ℝ
  M : ℝ
  T : ℝ
  R : ℝ
  Y : ℝ

-- Define the theorem
theorem heptagon_angle_measure (GEOMETRY : Heptagon) : 
  GEOMETRY.G = GEOMETRY.E ∧ 
  GEOMETRY.G = GEOMETRY.T ∧ 
  GEOMETRY.O + GEOMETRY.Y = 180 ∧
  GEOMETRY.M = GEOMETRY.R ∧
  GEOMETRY.M = 160 →
  GEOMETRY.G = 400 / 3 := by
sorry

end NUMINAMATH_CALUDE_heptagon_angle_measure_l1112_111232


namespace NUMINAMATH_CALUDE_equation_solution_l1112_111237

theorem equation_solution :
  ∃ x : ℚ, (8 * x^2 + 150 * x + 2) / (3 * x + 50) = 4 * x + 2 ∧ x = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1112_111237


namespace NUMINAMATH_CALUDE_point_on_line_p_value_l1112_111272

/-- Given that (m, n) and (m + p, n + 15) both lie on the line x = (y / 5) - (2 / 5),
    prove that p = 3. -/
theorem point_on_line_p_value (m n p : ℝ) : 
  (m = n / 5 - 2 / 5) → 
  (m + p = (n + 15) / 5 - 2 / 5) → 
  p = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_p_value_l1112_111272


namespace NUMINAMATH_CALUDE_reema_loan_interest_l1112_111210

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem reema_loan_interest :
  let principal : ℚ := 1500
  let rate : ℚ := 7
  let time : ℚ := rate
  simple_interest principal rate time = 735 := by sorry

end NUMINAMATH_CALUDE_reema_loan_interest_l1112_111210


namespace NUMINAMATH_CALUDE_triangle_max_area_l1112_111223

theorem triangle_max_area (R : ℝ) (A B C : ℝ) (a b c : ℝ) :
  0 < R →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a = 2 * R * Real.sin (A / 2) →
  b = 2 * R * Real.sin (B / 2) →
  c = 2 * R * Real.sin (C / 2) →
  2 * R * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 2 * a - b) * Real.sin B →
  ∃ (S : ℝ), S ≤ (Real.sqrt 2 + 1) / 2 * R ^ 2 ∧
    ∀ (S' : ℝ), S' = 1 / 2 * a * b * Real.sin C → S' ≤ S :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1112_111223


namespace NUMINAMATH_CALUDE_boys_at_reunion_l1112_111246

/-- The number of handshakes when each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There are 12 boys at the reunion given the conditions. -/
theorem boys_at_reunion : ∃ (n : ℕ), n > 0 ∧ handshakes n = 66 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_boys_at_reunion_l1112_111246


namespace NUMINAMATH_CALUDE_joes_height_l1112_111263

theorem joes_height (sara_height joe_height : ℕ) : 
  sara_height + joe_height = 120 →
  joe_height = 2 * sara_height + 6 →
  joe_height = 82 :=
by
  sorry

end NUMINAMATH_CALUDE_joes_height_l1112_111263


namespace NUMINAMATH_CALUDE_emily_sixth_score_l1112_111259

def emily_scores : List ℕ := [91, 94, 88, 90, 101]
def target_mean : ℕ := 95
def num_quizzes : ℕ := 6

theorem emily_sixth_score :
  ∃ (sixth_score : ℕ),
    (emily_scores.sum + sixth_score) / num_quizzes = target_mean ∧
    sixth_score = 106 := by
  sorry

end NUMINAMATH_CALUDE_emily_sixth_score_l1112_111259


namespace NUMINAMATH_CALUDE_triangle_inequality_minimum_l1112_111228

theorem triangle_inequality_minimum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htri : a + b ≥ c ∧ b + c ≥ a ∧ c + a ≥ b) :
  c / (a + b) + b / c ≥ Real.sqrt 2 - 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_minimum_l1112_111228


namespace NUMINAMATH_CALUDE_product_of_4_7_25_l1112_111244

theorem product_of_4_7_25 : 4 * 7 * 25 = 700 := by
  sorry

end NUMINAMATH_CALUDE_product_of_4_7_25_l1112_111244


namespace NUMINAMATH_CALUDE_magnitude_of_7_minus_24i_l1112_111221

theorem magnitude_of_7_minus_24i : Complex.abs (7 - 24 * Complex.I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_7_minus_24i_l1112_111221


namespace NUMINAMATH_CALUDE_multiply_by_8050_equals_80_5_l1112_111213

theorem multiply_by_8050_equals_80_5 : ∃ x : ℝ, 8050 * x = 80.5 ∧ x = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_8050_equals_80_5_l1112_111213


namespace NUMINAMATH_CALUDE_sector_central_angle_l1112_111204

theorem sector_central_angle (r : ℝ) (θ : ℝ) (h : r > 0) :
  2 * r + r * θ = π * r / 2 → θ = π - 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1112_111204


namespace NUMINAMATH_CALUDE_prime_factors_of_1998_l1112_111233

theorem prime_factors_of_1998 (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ 
  a < b ∧ b < c ∧
  a * b * c = 1998 →
  (b + c)^a = 1600 := by
sorry

end NUMINAMATH_CALUDE_prime_factors_of_1998_l1112_111233


namespace NUMINAMATH_CALUDE_median_salary_is_clerk_salary_l1112_111243

/-- Represents a position in the company -/
inductive Position
  | CEO
  | SeniorManager
  | Manager
  | AssistantManager
  | Clerk

/-- Returns the number of employees for a given position -/
def employeeCount (p : Position) : ℕ :=
  match p with
  | .CEO => 1
  | .SeniorManager => 8
  | .Manager => 12
  | .AssistantManager => 10
  | .Clerk => 40

/-- Returns the salary for a given position -/
def salary (p : Position) : ℕ :=
  match p with
  | .CEO => 180000
  | .SeniorManager => 95000
  | .Manager => 70000
  | .AssistantManager => 55000
  | .Clerk => 28000

/-- The total number of employees in the company -/
def totalEmployees : ℕ := 71

/-- Theorem stating that the median salary is equal to the Clerk's salary -/
theorem median_salary_is_clerk_salary :
  (totalEmployees + 1) / 2 ≤ (employeeCount Position.Clerk) ∧
  (totalEmployees + 1) / 2 > (totalEmployees - employeeCount Position.Clerk) →
  salary Position.Clerk = 28000 := by
  sorry

#check median_salary_is_clerk_salary

end NUMINAMATH_CALUDE_median_salary_is_clerk_salary_l1112_111243


namespace NUMINAMATH_CALUDE_wheel_spinner_probability_l1112_111231

theorem wheel_spinner_probability (p_E p_F p_G p_H p_I : ℚ) : 
  p_E = 1/5 →
  p_F = 1/4 →
  p_G = p_H →
  p_E + p_F + p_G + p_H + p_I = 1 →
  p_H = 9/40 := by
sorry

end NUMINAMATH_CALUDE_wheel_spinner_probability_l1112_111231


namespace NUMINAMATH_CALUDE_data_transmission_time_l1112_111282

/-- Represents the number of blocks to be sent -/
def num_blocks : ℕ := 30

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 1024

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 256

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Proves that the time to send the data is 2 minutes -/
theorem data_transmission_time :
  (num_blocks * chunks_per_block) / transmission_rate / seconds_per_minute = 2 :=
sorry

end NUMINAMATH_CALUDE_data_transmission_time_l1112_111282


namespace NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l1112_111285

/-- Given an equilateral triangle with two vertices at (3,7) and (13,7),
    prove that the y-coordinate of the third vertex in the first quadrant is 7 + 5√3 -/
theorem equilateral_triangle_third_vertex_y_coord :
  ∀ (x y : ℝ),
  let A : ℝ × ℝ := (3, 7)
  let B : ℝ × ℝ := (13, 7)
  let C : ℝ × ℝ := (x, y)
  (x > 0 ∧ y > 0) →  -- C is in the first quadrant
  (dist A B = dist B C ∧ dist B C = dist C A) →  -- Triangle is equilateral
  y = 7 + 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coord_l1112_111285


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1112_111291

theorem min_value_quadratic (x y : ℝ) :
  x^2 + y^2 - 10*x + 6*y + 25 ≥ -9 ∧
  ∃ (a b : ℝ), a^2 + b^2 - 10*a + 6*b + 25 = -9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1112_111291


namespace NUMINAMATH_CALUDE_expression_value_l1112_111293

theorem expression_value : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1112_111293


namespace NUMINAMATH_CALUDE_maximize_product_l1112_111200

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 40) :
  x^6 * y^3 ≤ 24^6 * 16^3 ∧
  (x^6 * y^3 = 24^6 * 16^3 ↔ x = 24 ∧ y = 16) :=
sorry

end NUMINAMATH_CALUDE_maximize_product_l1112_111200


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_320_l1112_111252

def closest_perfect_square (n : ℕ) : ℕ :=
  let root := n.sqrt
  if (root + 1)^2 - n < n - root^2
  then (root + 1)^2
  else root^2

theorem closest_perfect_square_to_320 :
  closest_perfect_square 320 = 324 :=
sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_320_l1112_111252
