import Mathlib

namespace binary_1011_to_decimal_l2412_241250

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1011 -/
def binary_1011 : List Bool := [true, true, false, true]

/-- Theorem stating that the decimal representation of binary 1011 is 11 -/
theorem binary_1011_to_decimal :
  binary_to_decimal binary_1011 = 11 := by sorry

end binary_1011_to_decimal_l2412_241250


namespace g_composition_equals_107_l2412_241298

/-- The function g defined as g(x) = 3x + 2 -/
def g (x : ℝ) : ℝ := 3 * x + 2

/-- Theorem stating that g(g(g(3))) = 107 -/
theorem g_composition_equals_107 : g (g (g 3)) = 107 := by
  sorry

end g_composition_equals_107_l2412_241298


namespace adult_ticket_cost_l2412_241206

theorem adult_ticket_cost (num_students : Nat) (num_adults : Nat) (student_ticket_cost : Nat) (total_cost : Nat) :
  num_students = 12 →
  num_adults = 4 →
  student_ticket_cost = 1 →
  total_cost = 24 →
  (total_cost - num_students * student_ticket_cost) / num_adults = 3 := by
  sorry

end adult_ticket_cost_l2412_241206


namespace average_weight_increase_l2412_241299

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 6 * initial_average
  let final_total := initial_total - 75 + 102
  let final_average := final_total / 6
  final_average - initial_average = 4.5 := by
sorry

end average_weight_increase_l2412_241299


namespace max_value_sum_of_inverses_l2412_241225

theorem max_value_sum_of_inverses (a b : ℝ) (h : a + b = 4) :
  (∀ x y : ℝ, x + y = 4 → 1 / (x^2 + 1) + 1 / (y^2 + 1) ≤ 1 / (a^2 + 1) + 1 / (b^2 + 1)) →
  1 / (a^2 + 1) + 1 / (b^2 + 1) = (Real.sqrt 5 + 2) / 4 :=
by sorry

end max_value_sum_of_inverses_l2412_241225


namespace equilateral_triangle_on_curve_l2412_241237

def curve (x : ℝ) : ℝ := -2 * x^2

theorem equilateral_triangle_on_curve :
  ∃ (P Q : ℝ × ℝ),
    (P.2 = curve P.1) ∧
    (Q.2 = curve Q.1) ∧
    (P.1 = -Q.1) ∧
    (P.2 = Q.2) ∧
    (dist P (0, 0) = dist Q (0, 0)) ∧
    (dist P Q = dist P (0, 0)) ∧
    (dist P (0, 0) = Real.sqrt 3) :=
sorry

end equilateral_triangle_on_curve_l2412_241237


namespace distance_to_focus_l2412_241203

/-- Given a parabola y^2 = 4x and a point M(x₀, 2√3) on it, 
    the distance from M to the focus of the parabola is 4. -/
theorem distance_to_focus (x₀ : ℝ) : 
  (2 * Real.sqrt 3)^2 = 4 * x₀ →   -- Point M is on the parabola
  x₀ + 1 = 4 :=                    -- Distance to focus
by sorry

end distance_to_focus_l2412_241203


namespace remainder_3456_div_97_l2412_241289

theorem remainder_3456_div_97 : 3456 % 97 = 61 := by
  sorry

end remainder_3456_div_97_l2412_241289


namespace jorge_ticket_cost_l2412_241221

def number_of_tickets : ℕ := 24
def price_per_ticket : ℚ := 7
def discount_percentage : ℚ := 50 / 100

def total_cost_with_discount : ℚ :=
  number_of_tickets * price_per_ticket * (1 - discount_percentage)

theorem jorge_ticket_cost : total_cost_with_discount = 84 := by
  sorry

end jorge_ticket_cost_l2412_241221


namespace parabola_transformation_l2412_241214

/-- Represents a parabola in the form y = (x + a)² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Applies a horizontal shift to a parabola -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a - shift, b := p.b }

/-- Applies a vertical shift to a parabola -/
def vertical_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, b := p.b + shift }

/-- The theorem to be proved -/
theorem parabola_transformation (p : Parabola) :
  p.a = 2 ∧ p.b = 3 →
  (vertical_shift (horizontal_shift p 3) (-2)) = { a := -1, b := 1 } := by
  sorry

end parabola_transformation_l2412_241214


namespace smallest_positive_angle_for_negative_2015_l2412_241226

-- Define the concept of angle equivalence
def angle_equivalent (a b : ℤ) : Prop := ∃ k : ℤ, b = a + 360 * k

-- Define the function to find the smallest positive equivalent angle
def smallest_positive_equivalent (a : ℤ) : ℤ :=
  (a % 360 + 360) % 360

-- Theorem statement
theorem smallest_positive_angle_for_negative_2015 :
  smallest_positive_equivalent (-2015) = 145 ∧
  angle_equivalent (-2015) 145 ∧
  ∀ x : ℤ, 0 < x ∧ x < 145 → ¬(angle_equivalent (-2015) x) :=
by sorry

end smallest_positive_angle_for_negative_2015_l2412_241226


namespace min_area_of_rectangle_with_perimeter_120_l2412_241292

-- Define the rectangle type
structure Rectangle where
  length : ℕ
  width : ℕ

-- Define the perimeter function
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

-- Define the area function
def area (r : Rectangle) : ℕ := r.length * r.width

-- Theorem statement
theorem min_area_of_rectangle_with_perimeter_120 :
  ∃ (min_area : ℕ), 
    (∀ (r : Rectangle), perimeter r = 120 → area r ≥ min_area) ∧
    (∃ (r : Rectangle), perimeter r = 120 ∧ area r = min_area) ∧
    min_area = 59 := by
  sorry

end min_area_of_rectangle_with_perimeter_120_l2412_241292


namespace willam_farm_tax_l2412_241232

/-- Farm tax calculation for Mr. Willam -/
theorem willam_farm_tax (total_tax : ℝ) (willam_percentage : ℝ) :
  let willam_tax := total_tax * (willam_percentage / 100)
  willam_tax = total_tax * (willam_percentage / 100) :=
by sorry

#check willam_farm_tax 3840 27.77777777777778

end willam_farm_tax_l2412_241232


namespace initial_rulers_l2412_241258

theorem initial_rulers (taken : ℕ) (remaining : ℕ) : taken = 25 → remaining = 21 → taken + remaining = 46 := by
  sorry

end initial_rulers_l2412_241258


namespace xy_bounds_l2412_241243

theorem xy_bounds (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  -1 ≤ x * y ∧ x * y ≤ 1/3 := by
  sorry

end xy_bounds_l2412_241243


namespace smallest_y_l2412_241248

theorem smallest_y (y : ℕ) 
  (h1 : y % 6 = 5) 
  (h2 : y % 7 = 6) 
  (h3 : y % 8 = 7) : 
  y ≥ 167 ∧ ∃ (z : ℕ), z % 6 = 5 ∧ z % 7 = 6 ∧ z % 8 = 7 ∧ z = 167 :=
sorry

end smallest_y_l2412_241248


namespace quadratic_roots_sum_of_squares_l2412_241209

theorem quadratic_roots_sum_of_squares : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 9*x₁ + 9 = 0) ∧
  (x₂^2 - 9*x₂ + 9 = 0) ∧
  (x₁^2 + x₂^2 = 63) :=
by sorry

end quadratic_roots_sum_of_squares_l2412_241209


namespace centroid_triangle_area_l2412_241273

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define point E on side CD
def E (rect : Rectangle) : ℝ × ℝ :=
  sorry

-- Define the area of a triangle
def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  sorry

-- Define the centroid of a triangle
def centroid (p q r : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem centroid_triangle_area (rect : Rectangle) :
  let G₁ := centroid rect.A rect.D (E rect)
  let G₂ := centroid rect.A rect.B (E rect)
  let G₃ := centroid rect.B rect.C (E rect)
  triangleArea G₁ G₂ G₃ = 1/18 :=
by
  sorry

end centroid_triangle_area_l2412_241273


namespace positive_integer_solution_of_equation_l2412_241227

theorem positive_integer_solution_of_equation (x : ℕ+) :
  (4 * x.val^2 - 16 * x.val - 60 = 0) → x.val = 6 := by
  sorry

end positive_integer_solution_of_equation_l2412_241227


namespace additional_courses_is_two_l2412_241277

/-- Represents the wall construction problem --/
structure WallProblem where
  initial_courses : ℕ
  bricks_per_course : ℕ
  total_bricks : ℕ

/-- Calculates the number of additional courses added to the wall --/
def additional_courses (w : WallProblem) : ℕ :=
  let initial_bricks := w.initial_courses * w.bricks_per_course
  let remaining_bricks := w.total_bricks - initial_bricks + (w.bricks_per_course / 2)
  remaining_bricks / w.bricks_per_course

/-- Theorem stating that the number of additional courses is 2 --/
theorem additional_courses_is_two (w : WallProblem) 
    (h1 : w.initial_courses = 3)
    (h2 : w.bricks_per_course = 400)
    (h3 : w.total_bricks = 1800) : 
  additional_courses w = 2 := by
  sorry

#eval additional_courses { initial_courses := 3, bricks_per_course := 400, total_bricks := 1800 }

end additional_courses_is_two_l2412_241277


namespace correct_parentheses_removal_l2412_241216

theorem correct_parentheses_removal (x : ℝ) :
  2 - 4 * ((1/4) * x + 1) = 2 - x - 4 := by
sorry

end correct_parentheses_removal_l2412_241216


namespace angle_ADC_measure_l2412_241284

theorem angle_ADC_measure (ABC BAC BCA BAD CAD ACD BCD ADC : Real) : 
  ABC = 60 →
  BAC = BAD + CAD →
  BAD = CAD →
  BCA = ACD + BCD →
  ACD = 2 * BCD →
  BAC + ABC + BCA = 180 →
  CAD + ACD + ADC = 180 →
  ADC = 100 :=
by sorry

end angle_ADC_measure_l2412_241284


namespace journey_speed_proof_l2412_241265

/-- Proves that given a journey of 120 miles in 120 minutes, with average speeds of 50 mph and 60 mph
for the first and second 40-minute segments respectively, the average speed for the last 40-minute
segment is 70 mph. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 120 →
  total_time = 120 →
  speed1 = 50 →
  speed2 = 60 →
  ∃ (speed3 : ℝ), speed3 = 70 ∧ (speed1 + speed2 + speed3) / 3 = total_distance / (total_time / 60) :=
by sorry

end journey_speed_proof_l2412_241265


namespace larger_number_proof_l2412_241297

theorem larger_number_proof (L S : ℕ) (hL : L > S) :
  L - S = 1365 → L = 6 * S + 15 → L = 1635 := by
  sorry

end larger_number_proof_l2412_241297


namespace hybrid_cars_with_full_headlights_l2412_241246

/-- Proves that in a car dealership with 600 cars, where 60% are hybrids and 40% of hybrids have only one headlight, the number of hybrids with full headlights is 216. -/
theorem hybrid_cars_with_full_headlights (total_cars : ℕ) (hybrid_percentage : ℚ) (one_headlight_percentage : ℚ) :
  total_cars = 600 →
  hybrid_percentage = 60 / 100 →
  one_headlight_percentage = 40 / 100 →
  (total_cars : ℚ) * hybrid_percentage * (1 - one_headlight_percentage) = 216 := by
  sorry

end hybrid_cars_with_full_headlights_l2412_241246


namespace soccer_goals_mean_l2412_241293

theorem soccer_goals_mean (players_3goals : ℕ) (players_4goals : ℕ) (players_5goals : ℕ) (players_6goals : ℕ) 
  (h1 : players_3goals = 4)
  (h2 : players_4goals = 3)
  (h3 : players_5goals = 1)
  (h4 : players_6goals = 2) :
  let total_goals := 3 * players_3goals + 4 * players_4goals + 5 * players_5goals + 6 * players_6goals
  let total_players := players_3goals + players_4goals + players_5goals + players_6goals
  (total_goals : ℚ) / total_players = 4.1 := by
  sorry

end soccer_goals_mean_l2412_241293


namespace sqrt_200_simplification_l2412_241272

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end sqrt_200_simplification_l2412_241272


namespace absolute_value_fraction_less_than_one_l2412_241262

theorem absolute_value_fraction_less_than_one (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : |(x - y) / (1 - x * y)| < 1 := by
  sorry

end absolute_value_fraction_less_than_one_l2412_241262


namespace union_of_A_and_B_l2412_241283

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 1}
def B : Set ℝ := {x : ℝ | x - 4 < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > 1} := by
  sorry

end union_of_A_and_B_l2412_241283


namespace negation_of_exponential_inequality_l2412_241228

theorem negation_of_exponential_inequality :
  (¬ ∀ a : ℝ, a > 0 → Real.exp a ≥ 1) ↔ (∃ a : ℝ, a > 0 ∧ Real.exp a < 1) := by
  sorry

end negation_of_exponential_inequality_l2412_241228


namespace quadratic_roots_in_unit_interval_l2412_241211

theorem quadratic_roots_in_unit_interval (a b c : ℤ) (ha : a > 0) 
  (h_roots : ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
    (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ) = 0 ∧
    (a : ℝ) * y^2 + (b : ℝ) * y + (c : ℝ) = 0) : 
  a ≥ 5 := by
sorry

end quadratic_roots_in_unit_interval_l2412_241211


namespace gcd_lcm_sum_l2412_241213

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 48 18 = 159 := by sorry

end gcd_lcm_sum_l2412_241213


namespace bridge_crossing_time_l2412_241202

/-- Proves that a man walking at 6 km/hr takes 15 minutes to cross a bridge of 1500 meters in length. -/
theorem bridge_crossing_time : 
  let walking_speed : ℝ := 6 -- km/hr
  let bridge_length : ℝ := 1.5 -- km (1500 meters)
  let crossing_time : ℝ := bridge_length / walking_speed * 60 -- in minutes
  crossing_time = 15 := by sorry

end bridge_crossing_time_l2412_241202


namespace psychology_majors_percentage_l2412_241271

theorem psychology_majors_percentage (total_students : ℝ) (h1 : total_students > 0) : 
  let freshmen := 0.60 * total_students
  let liberal_arts_freshmen := 0.40 * freshmen
  let psych_majors := 0.048 * total_students
  psych_majors / liberal_arts_freshmen = 0.20 := by
sorry

end psychology_majors_percentage_l2412_241271


namespace minimum_cost_purchase_l2412_241245

/-- Represents the unit price and quantity of an ingredient -/
structure Ingredient where
  price : ℝ
  quantity : ℝ

/-- Represents the purchase of two ingredients -/
structure Purchase where
  A : Ingredient
  B : Ingredient

def total_cost (p : Purchase) : ℝ := p.A.price * p.A.quantity + p.B.price * p.B.quantity

def total_quantity (p : Purchase) : ℝ := p.A.quantity + p.B.quantity

theorem minimum_cost_purchase :
  ∀ (p : Purchase),
    p.A.price + p.B.price = 68 →
    5 * p.A.price + 3 * p.B.price = 280 →
    total_quantity p = 36 →
    p.A.quantity ≥ 2 * p.B.quantity →
    total_cost p ≥ 1272 ∧
    (total_cost p = 1272 ↔ p.A.quantity = 24 ∧ p.B.quantity = 12) :=
by sorry

end minimum_cost_purchase_l2412_241245


namespace target_seat_representation_l2412_241275

/-- Represents a seat in a cinema -/
structure CinemaSeat where
  row : Nat
  seatNumber : Nat

/-- Given representation for seat number 4 in row 6 -/
def givenSeat : CinemaSeat := ⟨6, 4⟩

/-- The seat we want to represent (seat number 1 in row 5) -/
def targetSeat : CinemaSeat := ⟨5, 1⟩

/-- Theorem stating that the target seat is correctly represented -/
theorem target_seat_representation : targetSeat = ⟨5, 1⟩ := by
  sorry

end target_seat_representation_l2412_241275


namespace perfect_square_trinomial_condition_l2412_241218

/-- A quadratic trinomial in x and y -/
def QuadraticTrinomial (a b c : ℝ) := fun (x y : ℝ) ↦ a*x^2 + b*x*y + c*y^2

/-- Predicate for a quadratic trinomial being a perfect square -/
def IsPerfectSquare (q : (ℝ → ℝ → ℝ)) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ), q x y = (a*x + b*y)^2

theorem perfect_square_trinomial_condition (m : ℝ) :
  IsPerfectSquare (QuadraticTrinomial 4 m 9) → (m = 12 ∨ m = -12) :=
by sorry

end perfect_square_trinomial_condition_l2412_241218


namespace hacker_guarantee_l2412_241222

/-- A computer network with the given properties -/
structure ComputerNetwork where
  num_computers : ℕ
  is_connected : Bool
  no_shared_cycle_vertices : Bool

/-- The game state -/
structure GameState where
  network : ComputerNetwork
  hacked_computers : ℕ
  protected_computers : ℕ

/-- The game rules -/
def game_rules (state : GameState) : Bool :=
  state.hacked_computers + state.protected_computers ≤ state.network.num_computers

/-- The theorem statement -/
theorem hacker_guarantee (network : ComputerNetwork) 
  (h1 : network.num_computers = 2008)
  (h2 : network.is_connected = true)
  (h3 : network.no_shared_cycle_vertices = true) :
  ∃ (final_state : GameState), 
    final_state.network = network ∧ 
    game_rules final_state ∧ 
    final_state.hacked_computers ≥ 671 :=
sorry

end hacker_guarantee_l2412_241222


namespace planted_field_fraction_l2412_241295

theorem planted_field_fraction (a b c x : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_legs : a = 6 ∧ b = 8) (h_distance : (a - 0.6*x) * (b - 0.8*x) / 2 = 3) :
  (a * b / 2 - x^2) / (a * b / 2) = 5/6 := by
  sorry

end planted_field_fraction_l2412_241295


namespace condition_necessary_not_sufficient_l2412_241255

-- Define the equation of an ellipse
def is_ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / a + y^2 / b = 1 → (a > 0 ∧ b > 0 ∧ a ≠ b)

-- Define the condition ab > 0
def condition (a b : ℝ) : Prop := a * b > 0

-- Theorem stating that the condition is necessary but not sufficient
theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, is_ellipse_equation a b → condition a b) ∧
  (∃ a b : ℝ, condition a b ∧ ¬is_ellipse_equation a b) := by
  sorry

end condition_necessary_not_sufficient_l2412_241255


namespace courtyard_length_l2412_241210

/-- Proves that a courtyard with given dimensions and number of bricks has a specific length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) :
  width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  num_bricks = 14400 →
  width * (num_bricks * brick_length * brick_width / width) = 18 :=
by sorry

end courtyard_length_l2412_241210


namespace reciprocal_of_repeating_decimal_l2412_241230

/-- The common fraction form of the repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 4 / 11

/-- The reciprocal of the common fraction form of 0.363636... -/
def reciprocal : ℚ := 11 / 4

theorem reciprocal_of_repeating_decimal : (repeating_decimal)⁻¹ = reciprocal := by
  sorry

end reciprocal_of_repeating_decimal_l2412_241230


namespace book_profit_rate_l2412_241207

/-- Calculate the rate of profit given the cost price and selling price -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at Rs 50 and sold at Rs 80 is 60% -/
theorem book_profit_rate :
  let cost_price : ℚ := 50
  let selling_price : ℚ := 80
  rate_of_profit cost_price selling_price = 60 := by
  sorry

end book_profit_rate_l2412_241207


namespace first_day_of_month_l2412_241296

/-- Days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after n days -/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (nextDay d) n

/-- Theorem: If the 30th day of a month is a Wednesday, then the 1st day of that month is a Tuesday -/
theorem first_day_of_month (d : DayOfWeek) : 
  dayAfter d 29 = DayOfWeek.Wednesday → d = DayOfWeek.Tuesday := by
  sorry


end first_day_of_month_l2412_241296


namespace cat_teeth_count_l2412_241253

theorem cat_teeth_count (dog_teeth : ℕ) (pig_teeth : ℕ) (num_dogs : ℕ) (num_cats : ℕ) (num_pigs : ℕ) (total_teeth : ℕ) :
  dog_teeth = 42 →
  pig_teeth = 28 →
  num_dogs = 5 →
  num_cats = 10 →
  num_pigs = 7 →
  total_teeth = 706 →
  (total_teeth - num_dogs * dog_teeth - num_pigs * pig_teeth) / num_cats = 30 := by
sorry

end cat_teeth_count_l2412_241253


namespace half_circle_roll_distance_l2412_241254

/-- The length of the path traveled by the center of a half-circle when rolled along a straight line -/
theorem half_circle_roll_distance (r : ℝ) (h : r = 3 / Real.pi) : 
  let roll_distance := r * Real.pi + r
  roll_distance = 3 + 3 / Real.pi := by sorry

end half_circle_roll_distance_l2412_241254


namespace optimal_price_maximizes_profit_l2412_241233

/-- Represents the profit function for helmet sales -/
def profit_function (x : ℝ) : ℝ :=
  -20 * x^2 + 1400 * x - 60000

/-- The optimal selling price for helmets -/
def optimal_price : ℝ := 70

theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit_function optimal_price ≥ profit_function x :=
sorry

#check optimal_price_maximizes_profit

end optimal_price_maximizes_profit_l2412_241233


namespace race_speed_ratio_l2412_241239

/-- The ratio of runner a's speed to runner b's speed in a race -/
def speed_ratio (head_start_percent : ℚ) (winning_distance_percent : ℚ) : ℚ :=
  (1 + head_start_percent) / (1 + winning_distance_percent)

/-- Theorem stating that the speed ratio is 37/35 given the specified conditions -/
theorem race_speed_ratio :
  speed_ratio (48/100) (40/100) = 37/35 := by
  sorry

end race_speed_ratio_l2412_241239


namespace sine_inequality_l2412_241201

theorem sine_inequality (n : ℕ+) (θ : ℝ) : |Real.sin (n * θ)| ≤ n * |Real.sin θ| := by
  sorry

end sine_inequality_l2412_241201


namespace sign_determination_l2412_241259

theorem sign_determination (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ b < 0 := by
  sorry

end sign_determination_l2412_241259


namespace locus_of_vertices_is_parabola_l2412_241219

/-- The locus of vertices of a family of parabolas forms a parabola -/
theorem locus_of_vertices_is_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  ∃ (A B C : ℝ), A ≠ 0 ∧
    (∀ t : ℝ, ∃ (x y : ℝ),
      (y = a * x^2 + (2 * t + 1) * x + c) ∧
      (x = -(2 * t + 1) / (2 * a)) ∧
      (y = A * x^2 + B * x + C)) :=
by sorry

end locus_of_vertices_is_parabola_l2412_241219


namespace carls_yard_area_l2412_241270

/-- Represents a rectangular yard with fence posts. -/
structure FencedYard where
  short_posts : ℕ  -- Number of posts on the shorter side
  long_posts : ℕ   -- Number of posts on the longer side
  post_spacing : ℕ -- Distance between adjacent posts in yards

/-- Calculates the total number of fence posts. -/
def total_posts (yard : FencedYard) : ℕ :=
  2 * (yard.short_posts + yard.long_posts) - 4

/-- Calculates the area of the fenced yard in square yards. -/
def yard_area (yard : FencedYard) : ℕ :=
  (yard.short_posts - 1) * (yard.long_posts - 1) * yard.post_spacing^2

/-- Theorem stating the area of Carl's yard. -/
theorem carls_yard_area :
  ∃ (yard : FencedYard),
    yard.short_posts = 4 ∧
    yard.long_posts = 12 ∧
    yard.post_spacing = 5 ∧
    total_posts yard = 24 ∧
    yard.long_posts = 3 * yard.short_posts ∧
    yard_area yard = 825 :=
by sorry

end carls_yard_area_l2412_241270


namespace compute_b_l2412_241269

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 21

-- State the theorem
theorem compute_b (a b : ℚ) :
  (f a b (3 + Real.sqrt 5) = 0) → b = -27.5 := by
  sorry

end compute_b_l2412_241269


namespace not_cyclically_symmetric_example_cyclically_symmetric_example_difference_of_cyclically_symmetric_triangle_angles_cyclically_symmetric_l2412_241294

-- Definition of cyclically symmetric function
def CyclicallySymmetric (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c, f a b c = f b c a ∧ f a b c = f c a b

-- Statement 1
theorem not_cyclically_symmetric_example :
  ¬CyclicallySymmetric (fun x y z => x^2 - y^2 + z) := by sorry

-- Statement 2
theorem cyclically_symmetric_example :
  CyclicallySymmetric (fun x y z => x^2*(y-z) + y^2*(z-x) + z^2*(x-y)) := by sorry

-- Statement 3
theorem difference_of_cyclically_symmetric (f g : ℝ → ℝ → ℝ → ℝ) :
  CyclicallySymmetric f → CyclicallySymmetric g →
  CyclicallySymmetric (fun x y z => f x y z - g x y z) := by sorry

-- Statement 4
theorem triangle_angles_cyclically_symmetric (A B C : ℝ) :
  A + B + C = π →
  CyclicallySymmetric (fun x y z => 2 + Real.cos z * Real.cos (x-y) - Real.cos z^2) := by sorry

end not_cyclically_symmetric_example_cyclically_symmetric_example_difference_of_cyclically_symmetric_triangle_angles_cyclically_symmetric_l2412_241294


namespace race_result_l2412_241264

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ → ℝ

/-- The race setup -/
structure Race where
  sasha : Runner
  lesha : Runner
  kolya : Runner
  length : ℝ

def Race.valid (race : Race) : Prop :=
  race.sasha.speed > 0 ∧ 
  race.lesha.speed > 0 ∧ 
  race.kolya.speed > 0 ∧
  race.length > 0 ∧
  -- When Sasha finishes, Lesha is 10 meters behind
  race.sasha.position (race.length / race.sasha.speed) = race.length ∧
  race.lesha.position (race.length / race.sasha.speed) = race.length - 10 ∧
  -- When Lesha finishes, Kolya is 10 meters behind
  race.lesha.position (race.length / race.lesha.speed) = race.length ∧
  race.kolya.position (race.length / race.lesha.speed) = race.length - 10 ∧
  -- All runners have constant speeds
  ∀ t, race.sasha.position t = race.sasha.speed * t ∧
       race.lesha.position t = race.lesha.speed * t ∧
       race.kolya.position t = race.kolya.speed * t

theorem race_result (race : Race) (h : race.valid) : 
  race.kolya.position (race.length / race.sasha.speed) = race.length - 19 := by
  sorry

end race_result_l2412_241264


namespace percentage_calculation_l2412_241261

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 4800 → 
  (P / 100) * (30 / 100) * (50 / 100) * N = 108 → 
  P = 15 := by
sorry

end percentage_calculation_l2412_241261


namespace rectangle_side_equality_l2412_241235

theorem rectangle_side_equality (X : ℝ) : 
  (∀ (top bottom : ℝ), top = 5 + X ∧ bottom = 10 ∧ top = bottom) → X = 5 := by
  sorry

end rectangle_side_equality_l2412_241235


namespace books_in_fiction_section_l2412_241205

theorem books_in_fiction_section 
  (initial_books : ℕ) 
  (books_left : ℕ) 
  (history_books : ℕ) 
  (children_books : ℕ) 
  (wrong_place_books : ℕ) 
  (h1 : initial_books = 51) 
  (h2 : books_left = 16) 
  (h3 : history_books = 12) 
  (h4 : children_books = 8) 
  (h5 : wrong_place_books = 4) : 
  initial_books - books_left - history_books - (children_books - wrong_place_books) = 19 := by
  sorry

end books_in_fiction_section_l2412_241205


namespace greatest_integer_satisfying_conditions_l2412_241267

theorem greatest_integer_satisfying_conditions : 
  ∃ (n : ℕ), n < 150 ∧ 
  (∃ (k : ℕ), n = 11 * k - 1) ∧ 
  (∃ (l : ℕ), n = 9 * l + 2) ∧
  (∀ (m : ℕ), m < 150 → 
    (∃ (k' : ℕ), m = 11 * k' - 1) → 
    (∃ (l' : ℕ), m = 9 * l' + 2) → 
    m ≤ n) ∧
  n = 65 :=
by sorry

end greatest_integer_satisfying_conditions_l2412_241267


namespace quadratic_trinomial_pairs_l2412_241290

/-- Represents a quadratic trinomial ax^2 + bx + c -/
structure QuadraticTrinomial (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Checks if two numbers are roots of a quadratic trinomial -/
def areRoots {α : Type*} [Ring α] (t : QuadraticTrinomial α) (r1 r2 : α) : Prop :=
  t.a * r1 * r1 + t.b * r1 + t.c = 0 ∧ t.a * r2 * r2 + t.b * r2 + t.c = 0

theorem quadratic_trinomial_pairs 
  {α : Type*} [Field α] [CharZero α]
  (t1 t2 : QuadraticTrinomial α)
  (h1 : areRoots t2 t1.b t1.c)
  (h2 : areRoots t1 t2.b t2.c) :
  (∃ (a : α), t1 = ⟨1, a, 0⟩ ∧ t2 = ⟨1, -a, 0⟩) ∨
  (t1 = ⟨1, 1, -2⟩ ∧ t2 = ⟨1, 1, -2⟩) := by
  sorry


end quadratic_trinomial_pairs_l2412_241290


namespace total_packs_bought_l2412_241285

/-- The number of index card packs John buys for each student -/
def packs_per_student : ℕ := 2

/-- The number of classes John has -/
def num_classes : ℕ := 6

/-- The number of students in each of John's classes -/
def students_per_class : ℕ := 30

/-- Theorem: John buys 360 packs of index cards in total -/
theorem total_packs_bought : packs_per_student * num_classes * students_per_class = 360 := by
  sorry

end total_packs_bought_l2412_241285


namespace sin_negative_1020_degrees_l2412_241212

theorem sin_negative_1020_degrees : Real.sin ((-1020 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_negative_1020_degrees_l2412_241212


namespace second_vessel_capacity_l2412_241236

/-- Proves that the capacity of the second vessel is 3.625 liters given the conditions of the problem -/
theorem second_vessel_capacity :
  let vessel1_capacity : ℝ := 3
  let vessel1_alcohol_percentage : ℝ := 0.25
  let vessel2_alcohol_percentage : ℝ := 0.40
  let total_liquid : ℝ := 8
  let new_concentration : ℝ := 0.275
  ∃ vessel2_capacity : ℝ,
    vessel2_capacity > 0 ∧
    vessel1_capacity * vessel1_alcohol_percentage + 
    vessel2_capacity * vessel2_alcohol_percentage = 
    total_liquid * new_concentration ∧
    vessel2_capacity = 3.625 := by
  sorry


end second_vessel_capacity_l2412_241236


namespace solution_value_l2412_241287

theorem solution_value (x a : ℝ) (h : x = 5 ∧ a * x - 8 = 20 + a) : a = 7 := by
  sorry

end solution_value_l2412_241287


namespace limit_cosine_fraction_l2412_241291

theorem limit_cosine_fraction :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |((1 - Real.cos (2*x)) / (Real.cos (7*x) - Real.cos (3*x))) + (1/10)| < ε :=
by sorry

end limit_cosine_fraction_l2412_241291


namespace log_product_telescoping_l2412_241238

theorem log_product_telescoping (z : ℝ) : 
  z = (Real.log 4 / Real.log 3) * (Real.log 5 / Real.log 4) * 
      (Real.log 6 / Real.log 5) * (Real.log 7 / Real.log 6) * 
      (Real.log 8 / Real.log 7) * (Real.log 9 / Real.log 8) * 
      (Real.log 10 / Real.log 9) * (Real.log 11 / Real.log 10) * 
      (Real.log 12 / Real.log 11) * (Real.log 13 / Real.log 12) * 
      (Real.log 14 / Real.log 13) * (Real.log 15 / Real.log 14) * 
      (Real.log 16 / Real.log 15) * (Real.log 17 / Real.log 16) * 
      (Real.log 18 / Real.log 17) * (Real.log 19 / Real.log 18) * 
      (Real.log 20 / Real.log 19) * (Real.log 21 / Real.log 20) * 
      (Real.log 22 / Real.log 21) * (Real.log 23 / Real.log 22) * 
      (Real.log 24 / Real.log 23) * (Real.log 25 / Real.log 24) * 
      (Real.log 26 / Real.log 25) * (Real.log 27 / Real.log 26) * 
      (Real.log 28 / Real.log 27) * (Real.log 29 / Real.log 28) * 
      (Real.log 30 / Real.log 29) * (Real.log 31 / Real.log 30) * 
      (Real.log 32 / Real.log 31) * (Real.log 33 / Real.log 32) * 
      (Real.log 34 / Real.log 33) * (Real.log 35 / Real.log 34) * 
      (Real.log 36 / Real.log 35) * (Real.log 37 / Real.log 36) * 
      (Real.log 38 / Real.log 37) * (Real.log 39 / Real.log 38) * 
      (Real.log 40 / Real.log 39) →
  z = (3 * Real.log 2 + Real.log 5) / Real.log 3 := by
  sorry

end log_product_telescoping_l2412_241238


namespace fraction_value_l2412_241229

theorem fraction_value (x : ℝ) : (3 * x^2 + 9 * x + 15) / (3 * x^2 + 9 * x + 5) = 41 := by
  sorry

end fraction_value_l2412_241229


namespace gloria_pencils_l2412_241240

theorem gloria_pencils (G : ℕ) (h : G + 99 = 101) : G = 2 := by
  sorry

end gloria_pencils_l2412_241240


namespace ripe_oranges_per_day_l2412_241257

theorem ripe_oranges_per_day :
  ∀ (daily_ripe_oranges : ℕ),
    daily_ripe_oranges * 73 = 365 →
    daily_ripe_oranges = 5 := by
  sorry

end ripe_oranges_per_day_l2412_241257


namespace fine_on_fifth_day_l2412_241263

/-- Calculates the fine for a given day based on the previous day's fine -/
def nextDayFine (prevFine : ℚ) : ℚ :=
  min (prevFine + 0.3) (prevFine * 2)

/-- Calculates the fine for a specified number of days -/
def fineAfterDays (days : ℕ) : ℚ :=
  match days with
  | 0 => 0
  | 1 => 0.07
  | n + 1 => nextDayFine (fineAfterDays n)

theorem fine_on_fifth_day :
  fineAfterDays 5 = 0.86 := by sorry

end fine_on_fifth_day_l2412_241263


namespace factor_theorem_application_l2412_241249

theorem factor_theorem_application (x t : ℝ) : 
  (∃ k : ℝ, 4 * x^2 + 9 * x - 2 = (x - t) * k) ↔ (t = -1/4 ∨ t = -2) :=
by sorry

end factor_theorem_application_l2412_241249


namespace evaluate_expression_l2412_241234

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (x - 3) - 5) + 9) + 2 = x^4 - 3*x^3 - 5*x^2 + 9*x + 2 := by
  sorry

end evaluate_expression_l2412_241234


namespace reciprocal_sum_identity_l2412_241256

theorem reciprocal_sum_identity (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  1 / x + 1 / y = 1 / z → z = x * y / (y + x) := by
  sorry

end reciprocal_sum_identity_l2412_241256


namespace max_truck_speed_l2412_241266

theorem max_truck_speed (distance : ℝ) (hourly_cost : ℝ) (fixed_cost : ℝ) (max_total_cost : ℝ) :
  distance = 125 →
  hourly_cost = 30 →
  fixed_cost = 1000 →
  max_total_cost = 1200 →
  ∃ (max_speed : ℝ),
    max_speed = 75 ∧
    ∀ (speed : ℝ),
      speed > 0 →
      (distance / speed) * hourly_cost + fixed_cost + 2 * speed ≤ max_total_cost →
      speed ≤ max_speed :=
sorry

end max_truck_speed_l2412_241266


namespace natural_fraction_pairs_l2412_241220

def is_valid_pair (x y : ℕ) : Prop :=
  (∃ k : ℕ, (x + 1) = k * y) ∧ (∃ m : ℕ, (y + 1) = m * x)

theorem natural_fraction_pairs :
  ∀ x y : ℕ, is_valid_pair x y ↔ 
    ((x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) ∨ (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 3)) :=
by sorry

end natural_fraction_pairs_l2412_241220


namespace parallel_line_slope_l2412_241278

/-- Given a line with equation 2x + 4y = -17, prove that its slope (and the slope of any parallel line) is -1/2 -/
theorem parallel_line_slope (x y : ℝ) (h : 2 * x + 4 * y = -17) :
  ∃ m b : ℝ, m = -1/2 ∧ y = m * x + b := by
  sorry

end parallel_line_slope_l2412_241278


namespace point_location_l2412_241200

theorem point_location (α : Real) (h : α = 5 * Real.pi / 8) :
  let P : Real × Real := (Real.sin α, Real.tan α)
  P.1 > 0 ∧ P.2 < 0 :=
sorry

end point_location_l2412_241200


namespace initial_milk_water_ratio_l2412_241252

theorem initial_milk_water_ratio 
  (total_initial_volume : ℝ)
  (additional_water : ℝ)
  (final_ratio : ℝ)
  (h1 : total_initial_volume = 45)
  (h2 : additional_water = 23)
  (h3 : final_ratio = 1.125)
  : ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_initial_volume ∧
    initial_milk / (initial_water + additional_water) = final_ratio ∧
    initial_milk / initial_water = 4 := by
  sorry

end initial_milk_water_ratio_l2412_241252


namespace certain_number_problem_l2412_241274

theorem certain_number_problem (x y z : ℝ) : 
  x + y = 15 →
  y = 7 →
  3 * x = z * y - 11 →
  z = 5 := by
sorry

end certain_number_problem_l2412_241274


namespace part_one_part_two_l2412_241268

-- Define the conditions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := x^2 - 6*x + 8 < 0 ∧ x^2 - 8*x + 15 > 0

-- Part 1
theorem part_one : 
  ∀ x : ℝ, p x 1 ∧ q x → 2 < x ∧ x < 3 :=
sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, q x → (∃ a : ℝ, a > 0 ∧ p x a)) ∧
  (∃ x : ℝ, (∃ a : ℝ, a > 0 ∧ p x a) ∧ ¬q x) ↔
  (∃ a : ℝ, 1 ≤ a ∧ a ≤ 2) :=
sorry

end part_one_part_two_l2412_241268


namespace circus_tent_capacity_l2412_241217

theorem circus_tent_capacity (total_capacity : ℕ) (num_sections : ℕ) (section_capacity : ℕ) : 
  total_capacity = 984 → 
  num_sections = 4 → 
  total_capacity = num_sections * section_capacity → 
  section_capacity = 246 := by
sorry

end circus_tent_capacity_l2412_241217


namespace square_sequence_50th_term_l2412_241281

/-- Represents the number of squares in the nth figure of the sequence -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The theorem states that the 50th term of the sequence is 7651 -/
theorem square_sequence_50th_term :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 50 = 7651 := by
  sorry

end square_sequence_50th_term_l2412_241281


namespace difference_of_squares_factorization_linear_expression_factorization_l2412_241242

-- Problem 1
theorem difference_of_squares_factorization (x : ℝ) :
  4 * x^2 - 9 = (2*x + 3) * (2*x - 3) := by sorry

-- Problem 2
theorem linear_expression_factorization (a b x y : ℝ) :
  2*a*(x - y) - 3*b*(y - x) = (x - y)*(2*a + 3*b) := by sorry

end difference_of_squares_factorization_linear_expression_factorization_l2412_241242


namespace gcd_factorial_eight_ten_l2412_241208

theorem gcd_factorial_eight_ten : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end gcd_factorial_eight_ten_l2412_241208


namespace remainder_of_3_19_times_5_7_mod_100_l2412_241231

theorem remainder_of_3_19_times_5_7_mod_100 : (3^19 * 5^7) % 100 = 75 := by
  sorry

end remainder_of_3_19_times_5_7_mod_100_l2412_241231


namespace picnic_cost_l2412_241260

def sandwich_price : ℚ := 6
def fruit_salad_price : ℚ := 4
def cheese_platter_price : ℚ := 8
def soda_price : ℚ := 2.5
def snack_bag_price : ℚ := 4.5

def num_people : ℕ := 6
def num_sandwiches : ℕ := 6
def num_fruit_salads : ℕ := 4
def num_cheese_platters : ℕ := 3
def num_sodas : ℕ := 12
def num_snack_bags : ℕ := 5

def sandwich_discount (n : ℕ) : ℕ := n / 6
def cheese_platter_discount (n : ℕ) : ℚ := if n ≥ 2 then 0.1 else 0
def soda_discount (n : ℕ) : ℕ := (n / 10) * 2
def snack_bag_discount (n : ℕ) : ℕ := n / 2

def total_cost : ℚ :=
  (num_sandwiches - sandwich_discount num_sandwiches) * sandwich_price +
  num_fruit_salads * fruit_salad_price +
  (num_cheese_platters * cheese_platter_price) * (1 - cheese_platter_discount num_cheese_platters) +
  (num_sodas - soda_discount num_sodas) * soda_price +
  num_snack_bags * snack_bag_price - snack_bag_discount num_snack_bags

theorem picnic_cost : total_cost = 113.1 := by
  sorry

end picnic_cost_l2412_241260


namespace equality_transitivity_add_polynomial_to_equation_l2412_241286

-- Statement 1: Transitivity of equality
theorem equality_transitivity (a b c : ℝ) (h1 : a = b) (h2 : b = c) : a = c := by
  sorry

-- Statement 5: Adding a polynomial to both sides of an equation
theorem add_polynomial_to_equation (f g p : ℝ → ℝ) (h : ∀ x, f x = g x) : 
  ∀ x, f x + p x = g x + p x := by
  sorry

end equality_transitivity_add_polynomial_to_equation_l2412_241286


namespace book_cost_problem_l2412_241241

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) 
  (h1 : total_cost = 360)
  (h2 : loss_percent = 0.15)
  (h3 : gain_percent = 0.19)
  (h4 : ∃ (c1 c2 : ℝ), c1 + c2 = total_cost ∧ 
                       c1 * (1 - loss_percent) = c2 * (1 + gain_percent)) :
  ∃ (loss_book_cost : ℝ), loss_book_cost = 210 ∧ 
    ∃ (c2 : ℝ), loss_book_cost + c2 = total_cost ∧ 
    loss_book_cost * (1 - loss_percent) = c2 * (1 + gain_percent) :=
sorry

end book_cost_problem_l2412_241241


namespace exactly_one_event_probability_l2412_241204

theorem exactly_one_event_probability (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  p₁ * (1 - p₂) + p₂ * (1 - p₁) = 
  (p₁ + p₂) - (p₁ * p₂) := by
  sorry

end exactly_one_event_probability_l2412_241204


namespace arithmetic_sequence_solve_for_y_l2412_241280

/-- Given an arithmetic sequence with the first three terms as specified,
    prove that the value of y is 5/3 -/
theorem arithmetic_sequence_solve_for_y :
  ∀ (seq : ℕ → ℚ),
  (seq 0 = 2/3) →
  (seq 1 = y + 2) →
  (seq 2 = 4*y) →
  (∀ n, seq (n+1) - seq n = seq (n+2) - seq (n+1)) →
  (y = 5/3) := by
sorry

end arithmetic_sequence_solve_for_y_l2412_241280


namespace yellow_marbles_total_l2412_241279

/-- The total number of yellow marbles after redistribution -/
def total_marbles_after_redistribution : ℕ → ℕ → ℕ → ℕ → ℕ
  | mary_initial, joan, john, mary_to_tim =>
    (mary_initial - mary_to_tim) + joan + john + mary_to_tim

/-- Theorem stating the total number of yellow marbles after redistribution -/
theorem yellow_marbles_total
  (mary_initial : ℕ)
  (joan : ℕ)
  (john : ℕ)
  (mary_to_tim : ℕ)
  (h1 : mary_initial = 9)
  (h2 : joan = 3)
  (h3 : john = 7)
  (h4 : mary_to_tim = 4)
  (h5 : mary_initial ≥ mary_to_tim) :
  total_marbles_after_redistribution mary_initial joan john mary_to_tim = 19 :=
by sorry

end yellow_marbles_total_l2412_241279


namespace chocolate_bar_cost_l2412_241247

/-- Proves that the cost of each bar of chocolate is $5 given the conditions of the problem. -/
theorem chocolate_bar_cost
  (num_bars : ℕ)
  (total_selling_price : ℚ)
  (packaging_cost_per_bar : ℚ)
  (total_profit : ℚ)
  (h1 : num_bars = 5)
  (h2 : total_selling_price = 90)
  (h3 : packaging_cost_per_bar = 2)
  (h4 : total_profit = 55) :
  ∃ (cost_per_bar : ℚ), cost_per_bar = 5 ∧
    total_selling_price = num_bars * cost_per_bar + num_bars * packaging_cost_per_bar + total_profit :=
by sorry

end chocolate_bar_cost_l2412_241247


namespace group_messages_in_week_l2412_241276

/-- Calculates the total number of messages sent in a week by remaining members of a group -/
theorem group_messages_in_week 
  (initial_members : ℕ) 
  (removed_members : ℕ) 
  (messages_per_day : ℕ) 
  (days_in_week : ℕ) 
  (h1 : initial_members = 150) 
  (h2 : removed_members = 20) 
  (h3 : messages_per_day = 50) 
  (h4 : days_in_week = 7) :
  (initial_members - removed_members) * messages_per_day * days_in_week = 45500 :=
by sorry

end group_messages_in_week_l2412_241276


namespace high_schooler_pairs_l2412_241282

theorem high_schooler_pairs (n : ℕ) (h : 10 ≤ n ∧ n ≤ 15) : 
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 15 → n * (n - 1) / 2 ≤ m * (m - 1) / 2) → n = 10 ∧
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 15 → m * (m - 1) / 2 ≤ n * (n - 1) / 2) → n = 15 :=
by sorry

end high_schooler_pairs_l2412_241282


namespace square_binomial_simplification_l2412_241224

theorem square_binomial_simplification (x : ℝ) (h : 3 * x^2 - 12 ≥ 0) :
  (7 - Real.sqrt (3 * x^2 - 12))^2 = 3 * x^2 + 37 - 14 * Real.sqrt (3 * x^2 - 12) := by
  sorry

end square_binomial_simplification_l2412_241224


namespace basketball_league_female_fraction_l2412_241223

theorem basketball_league_female_fraction :
  -- Define variables
  let last_year_males : ℕ := 30
  let last_year_females : ℕ := 15  -- Derived from the solution
  let male_increase_rate : ℚ := 11/10
  let female_increase_rate : ℚ := 5/4
  let total_increase_rate : ℚ := 23/20

  -- Define this year's participants
  let this_year_males : ℚ := last_year_males * male_increase_rate
  let this_year_females : ℚ := last_year_females * female_increase_rate
  let this_year_total : ℚ := (last_year_males + last_year_females) * total_increase_rate

  -- The fraction of female participants this year
  this_year_females / this_year_total = 75 / 207 := by
sorry

end basketball_league_female_fraction_l2412_241223


namespace multiples_of_15_between_20_and_205_l2412_241288

theorem multiples_of_15_between_20_and_205 : 
  (Finset.filter (fun x => x % 15 = 0 ∧ x > 20 ∧ x ≤ 205) (Finset.range 206)).card = 12 := by
  sorry

end multiples_of_15_between_20_and_205_l2412_241288


namespace product_of_one_plus_roots_l2412_241244

theorem product_of_one_plus_roots (a b c : ℝ) : 
  (x^3 - 15*x^2 + 25*x - 10 = 0 → x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 51 := by
  sorry

end product_of_one_plus_roots_l2412_241244


namespace seed_mixture_percentage_l2412_241251

/-- Given two seed mixtures X and Y, and a final mixture containing both, 
    this theorem proves the percentage of mixture X in the final mixture. -/
theorem seed_mixture_percentage 
  (x_ryegrass : ℚ) (x_bluegrass : ℚ) (y_ryegrass : ℚ) (y_fescue : ℚ) 
  (final_ryegrass : ℚ) : 
  x_ryegrass = 40 / 100 →
  x_bluegrass = 60 / 100 →
  y_ryegrass = 25 / 100 →
  y_fescue = 75 / 100 →
  final_ryegrass = 38 / 100 →
  x_ryegrass + x_bluegrass = 1 →
  y_ryegrass + y_fescue = 1 →
  ∃ (p : ℚ), p * x_ryegrass + (1 - p) * y_ryegrass = final_ryegrass ∧ 
             p = 260 / 3 :=
by sorry

end seed_mixture_percentage_l2412_241251


namespace power_multiplication_l2412_241215

theorem power_multiplication (a : ℝ) : a ^ 2 * a ^ 5 = a ^ 7 := by
  sorry

end power_multiplication_l2412_241215
