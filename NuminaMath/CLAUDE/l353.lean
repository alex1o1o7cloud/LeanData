import Mathlib

namespace NUMINAMATH_CALUDE_lenora_points_scored_l353_35315

-- Define the types of shots
inductive ShotType
| ThreePoint
| FreeThrow

-- Define the game parameters
def total_shots : ℕ := 40
def three_point_success_rate : ℚ := 1/4
def free_throw_success_rate : ℚ := 1/2

-- Define the point values for each shot type
def point_value (shot : ShotType) : ℕ :=
  match shot with
  | ShotType.ThreePoint => 3
  | ShotType.FreeThrow => 1

-- Define the function to calculate points scored
def points_scored (three_point_attempts : ℕ) (free_throw_attempts : ℕ) : ℚ :=
  (three_point_attempts : ℚ) * three_point_success_rate * (point_value ShotType.ThreePoint) +
  (free_throw_attempts : ℚ) * free_throw_success_rate * (point_value ShotType.FreeThrow)

-- Theorem statement
theorem lenora_points_scored :
  ∀ (three_point_attempts free_throw_attempts : ℕ),
    three_point_attempts + free_throw_attempts = total_shots →
    points_scored three_point_attempts free_throw_attempts = 30 :=
by sorry

end NUMINAMATH_CALUDE_lenora_points_scored_l353_35315


namespace NUMINAMATH_CALUDE_sin_cos_roots_l353_35348

theorem sin_cos_roots (θ : Real) (a : Real) 
  (h1 : x^2 - 2 * Real.sqrt 2 * a * x + a = 0 ↔ x = Real.sin θ ∨ x = Real.cos θ)
  (h2 : -π/2 < θ ∧ θ < 0) : 
  a = -1/4 ∧ Real.sin θ - Real.cos θ = -Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_roots_l353_35348


namespace NUMINAMATH_CALUDE_password_probability_l353_35383

def even_two_digit_numbers : ℕ := 45
def vowels : ℕ := 5
def total_letters : ℕ := 26
def prime_two_digit_numbers : ℕ := 21
def total_two_digit_numbers : ℕ := 90

theorem password_probability :
  (even_two_digit_numbers / total_two_digit_numbers) *
  (vowels / total_letters) *
  (prime_two_digit_numbers / total_two_digit_numbers) =
  7 / 312 := by sorry

end NUMINAMATH_CALUDE_password_probability_l353_35383


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l353_35308

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 16) 
  (h2 : x * y = -8) : 
  x^2 + y^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l353_35308


namespace NUMINAMATH_CALUDE_robotics_club_mentor_age_l353_35326

theorem robotics_club_mentor_age (total_members : ℕ) (avg_age : ℕ) 
  (num_boys num_girls num_mentors : ℕ) (avg_age_boys avg_age_girls : ℕ) :
  total_members = 50 →
  avg_age = 20 →
  num_boys = 25 →
  num_girls = 20 →
  num_mentors = 5 →
  avg_age_boys = 18 →
  avg_age_girls = 19 →
  (total_members * avg_age - num_boys * avg_age_boys - num_girls * avg_age_girls) / num_mentors = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_robotics_club_mentor_age_l353_35326


namespace NUMINAMATH_CALUDE_f_of_one_equals_negative_two_l353_35386

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_of_one_equals_negative_two
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_def : ∀ x, x < 0 → f x = x - x^4) :
  f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_equals_negative_two_l353_35386


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l353_35396

theorem thirty_percent_less_than_ninety (x : ℝ) : x + x / 2 = 63 → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l353_35396


namespace NUMINAMATH_CALUDE_josh_bought_four_cookies_l353_35320

/-- Calculates the number of cookies Josh bought given his initial money,
    the cost of other items, the cost per cookie, and the remaining money. -/
def cookies_bought (initial_money : ℚ) (hat_cost : ℚ) (pencil_cost : ℚ)
                   (cookie_cost : ℚ) (remaining_money : ℚ) : ℚ :=
  ((initial_money - hat_cost - pencil_cost - remaining_money) / cookie_cost)

/-- Proves that Josh bought 4 cookies given the problem conditions. -/
theorem josh_bought_four_cookies :
  cookies_bought 20 10 2 1.25 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_josh_bought_four_cookies_l353_35320


namespace NUMINAMATH_CALUDE_babysitting_cost_difference_l353_35384

/-- Represents the babysitting scenario with given rates and conditions -/
structure BabysittingScenario where
  current_rate : ℕ -- Rate of current babysitter in dollars per hour
  new_base_rate : ℕ -- Base rate of new babysitter in dollars per hour
  new_scream_charge : ℕ -- Extra charge for each scream by new babysitter
  hours : ℕ -- Number of hours of babysitting
  screams : ℕ -- Number of times kids scream during babysitting

/-- Calculates the cost difference between current and new babysitter -/
def costDifference (scenario : BabysittingScenario) : ℕ :=
  scenario.current_rate * scenario.hours - 
  (scenario.new_base_rate * scenario.hours + scenario.new_scream_charge * scenario.screams)

/-- Theorem stating the cost difference for the given scenario -/
theorem babysitting_cost_difference :
  ∃ (scenario : BabysittingScenario),
    scenario.current_rate = 16 ∧
    scenario.new_base_rate = 12 ∧
    scenario.new_scream_charge = 3 ∧
    scenario.hours = 6 ∧
    scenario.screams = 2 ∧
    costDifference scenario = 18 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_cost_difference_l353_35384


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l353_35307

theorem imaginary_part_of_complex_division (z₁ z₂ : ℂ) :
  z₁ = 2 - I → z₂ = 1 - 3*I → Complex.im (z₂ / z₁) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l353_35307


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l353_35356

-- Define the total number of people and the number of math majors
def total_people : ℕ := 12
def math_majors : ℕ := 5

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the probability of math majors sitting consecutively
def prob_consecutive_math_majors : ℚ := (total_people : ℚ) / (choose total_people math_majors : ℚ)

-- State the theorem
theorem math_majors_consecutive_probability :
  prob_consecutive_math_majors = 1 / 66 :=
sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l353_35356


namespace NUMINAMATH_CALUDE_equation_solutions_l353_35357

theorem equation_solutions :
  (∃ x : ℝ, (3 + x) * (30 / 100) = 4.8 ∧ x = 13) ∧
  (∃ x : ℝ, 5 / x = (9 / 2) / (8 / 5) ∧ x = 16 / 9) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l353_35357


namespace NUMINAMATH_CALUDE_height_to_sphere_ratio_l353_35381

/-- A truncated right circular cone with an inscribed sphere -/
structure TruncatedConeWithSphere where
  R : ℝ  -- radius of the larger base
  r : ℝ  -- radius of the smaller base
  H : ℝ  -- height of the truncated cone
  s : ℝ  -- radius of the inscribed sphere
  R_positive : R > 0
  r_positive : r > 0
  H_positive : H > 0
  s_positive : s > 0
  sphere_inscribed : s = Real.sqrt (R * r)
  volume_relation : π * H * (R^2 + R*r + r^2) / 3 = 4 * π * s^3

/-- The ratio of the height of the truncated cone to the radius of the sphere is 4 -/
theorem height_to_sphere_ratio (cone : TruncatedConeWithSphere) : 
  cone.H / cone.s = 4 := by
  sorry

end NUMINAMATH_CALUDE_height_to_sphere_ratio_l353_35381


namespace NUMINAMATH_CALUDE_xy_greater_than_xz_l353_35339

theorem xy_greater_than_xz (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) :
  x * y > x * z := by sorry

end NUMINAMATH_CALUDE_xy_greater_than_xz_l353_35339


namespace NUMINAMATH_CALUDE_not_divisible_by_two_or_five_l353_35338

def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 3)^2 + (n - 1)^2 + (n + 1)^2 + (n + 3)^2}

theorem not_divisible_by_two_or_five :
  ∀ x ∈ T, ¬(∃ k : ℤ, x = 2 * k ∨ x = 5 * k) :=
by sorry

end NUMINAMATH_CALUDE_not_divisible_by_two_or_five_l353_35338


namespace NUMINAMATH_CALUDE_perpendicular_unit_vector_l353_35335

/-- Given a vector a = (2, 1), prove that (√5/5, -2√5/5) is a unit vector perpendicular to a. -/
theorem perpendicular_unit_vector (a : ℝ × ℝ) (h : a = (2, 1)) :
  let b : ℝ × ℝ := (Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5)
  (a.1 * b.1 + a.2 * b.2 = 0) ∧ (b.1^2 + b.2^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vector_l353_35335


namespace NUMINAMATH_CALUDE_chicken_surprise_serving_weight_l353_35336

/-- Represents the recipe for Chicken Surprise -/
structure ChickenSurprise where
  servings : ℕ
  chickenPounds : ℚ
  stuffingOunces : ℕ

/-- Calculates the weight of one serving of Chicken Surprise in ounces -/
def servingWeight (recipe : ChickenSurprise) : ℚ :=
  let totalOunces := recipe.chickenPounds * 16 + recipe.stuffingOunces
  totalOunces / recipe.servings

/-- Theorem stating that one serving of Chicken Surprise is 8 ounces -/
theorem chicken_surprise_serving_weight :
  let recipe := ChickenSurprise.mk 12 (9/2) 24
  servingWeight recipe = 8 := by
  sorry

end NUMINAMATH_CALUDE_chicken_surprise_serving_weight_l353_35336


namespace NUMINAMATH_CALUDE_range_of_a_l353_35361

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x - 2 ≤ 0) → 
  -2 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l353_35361


namespace NUMINAMATH_CALUDE_polynomial_absolute_value_l353_35387

/-- A second-degree polynomial with real coefficients -/
def SecondDegreePolynomial (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

/-- The absolute value of f at 1, 2, and 3 is equal to 9 -/
def AbsValueCondition (f : ℝ → ℝ) : Prop :=
  |f 1| = 9 ∧ |f 2| = 9 ∧ |f 3| = 9

theorem polynomial_absolute_value (f : ℝ → ℝ) 
  (h1 : SecondDegreePolynomial f) 
  (h2 : AbsValueCondition f) : 
  |f 0| = 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_absolute_value_l353_35387


namespace NUMINAMATH_CALUDE_apartment_count_l353_35304

theorem apartment_count (total_keys : ℕ) (keys_per_apartment : ℕ) (num_complexes : ℕ) :
  total_keys = 72 →
  keys_per_apartment = 3 →
  num_complexes = 2 →
  ∃ (apartments_per_complex : ℕ), 
    apartments_per_complex * keys_per_apartment * num_complexes = total_keys ∧
    apartments_per_complex = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_apartment_count_l353_35304


namespace NUMINAMATH_CALUDE_divisibility_statements_l353_35380

theorem divisibility_statements :
  (12 % 2 = 0) ∧
  (123 % 3 = 0) ∧
  (1234 % 4 ≠ 0) ∧
  (12345 % 5 = 0) ∧
  (123456 % 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_statements_l353_35380


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l353_35317

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem fifteenth_term_of_sequence (a₁ a₂ : ℤ) (h : a₂ = a₁ + 1) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 53 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l353_35317


namespace NUMINAMATH_CALUDE_room_length_calculation_l353_35345

theorem room_length_calculation (area : ℝ) (width : ℝ) (length : ℝ) :
  area = 12.0 ∧ width = 8.0 ∧ area = width * length → length = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l353_35345


namespace NUMINAMATH_CALUDE_pool_cost_per_person_l353_35368

theorem pool_cost_per_person
  (total_earnings : ℝ)
  (num_people : ℕ)
  (amount_left : ℝ)
  (h1 : total_earnings = 30)
  (h2 : num_people = 10)
  (h3 : amount_left = 5) :
  (total_earnings - amount_left) / num_people = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_pool_cost_per_person_l353_35368


namespace NUMINAMATH_CALUDE_equation_d_is_linear_l353_35349

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants, and at least one of a or b is non-zero. --/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation x = y + 1 --/
def EquationD (x y : ℝ) : Prop := x = y + 1

theorem equation_d_is_linear : IsLinearEquationInTwoVariables EquationD := by
  sorry

#check equation_d_is_linear

end NUMINAMATH_CALUDE_equation_d_is_linear_l353_35349


namespace NUMINAMATH_CALUDE_intersection_of_circles_l353_35303

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def B (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- Define the theorem
theorem intersection_of_circles (r : ℝ) (hr : r > 0) :
  (∃! p, p ∈ A ∩ B r) → r = 3 ∨ r = 7 := by
  sorry


end NUMINAMATH_CALUDE_intersection_of_circles_l353_35303


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l353_35394

/-- A quadratic function f(x) with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + a

/-- The function g(x) represents f(x) - x -/
def g (a : ℝ) (x : ℝ) : ℝ := f a x - x

theorem quadratic_roots_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : g a x₁ = 0) (h₂ : g a x₂ = 0) (h₃ : 0 < x₁) (h₄ : x₁ < x₂) (h₅ : x₂ < 1) :
  (0 < a ∧ a < 3 - Real.sqrt 2) ∧ f a 0 * f a 1 - f a 0 < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l353_35394


namespace NUMINAMATH_CALUDE_pentagon_rectangle_angle_sum_l353_35365

/-- The sum of an interior angle of a regular pentagon and an interior angle of a rectangle is 198°. -/
theorem pentagon_rectangle_angle_sum : 
  let pentagon_angle : ℝ := 180 * (5 - 2) / 5
  let rectangle_angle : ℝ := 90
  pentagon_angle + rectangle_angle = 198 := by sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_angle_sum_l353_35365


namespace NUMINAMATH_CALUDE_toy_store_inventory_l353_35372

structure Toy where
  name : String
  week1_sales : ℕ
  week2_sales : ℕ
  remaining : ℕ

def initial_stock (t : Toy) : ℕ :=
  t.remaining + t.week1_sales + t.week2_sales

theorem toy_store_inventory (action_figures board_games puzzles stuffed_animals : Toy) 
  (h1 : action_figures.name = "Action Figures" ∧ action_figures.week1_sales = 38 ∧ action_figures.week2_sales = 26 ∧ action_figures.remaining = 19)
  (h2 : board_games.name = "Board Games" ∧ board_games.week1_sales = 27 ∧ board_games.week2_sales = 15 ∧ board_games.remaining = 8)
  (h3 : puzzles.name = "Puzzles" ∧ puzzles.week1_sales = 43 ∧ puzzles.week2_sales = 39 ∧ puzzles.remaining = 12)
  (h4 : stuffed_animals.name = "Stuffed Animals" ∧ stuffed_animals.week1_sales = 20 ∧ stuffed_animals.week2_sales = 18 ∧ stuffed_animals.remaining = 30) :
  initial_stock action_figures = 83 ∧ 
  initial_stock board_games = 50 ∧ 
  initial_stock puzzles = 94 ∧ 
  initial_stock stuffed_animals = 68 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_inventory_l353_35372


namespace NUMINAMATH_CALUDE_cards_given_to_jeff_l353_35369

/-- Nell's initial number of cards -/
def nell_initial : ℕ := 528

/-- Nell's remaining number of cards -/
def nell_remaining : ℕ := 252

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := nell_initial - nell_remaining

theorem cards_given_to_jeff : cards_given = 276 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_to_jeff_l353_35369


namespace NUMINAMATH_CALUDE_line_and_circle_proof_l353_35331

-- Define the lines and circles
def l₁ (x y : ℝ) : Prop := x - 2*y + 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x - y - 2 = 0
def c₁ (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def c₂ (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line that we want to prove
def target_line (x y : ℝ) : Prop := y = x

-- Define the circle that we want to prove
def target_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 11 = 0

-- Define the line on which the center of the target circle should lie
def center_line (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

theorem line_and_circle_proof :
  -- Part 1: The target line passes through the origin and the intersection of l₁ and l₂
  (∃ x y : ℝ, l₁ x y ∧ l₂ x y ∧ target_line x y) ∧
  target_line 0 0 ∧
  -- Part 2: The target circle has its center on the center_line and passes through
  -- the intersection points of c₁ and c₂
  (∃ x y : ℝ, center_line x y ∧ 
    ∀ a b : ℝ, (c₁ a b ∧ c₂ a b) → target_circle a b) :=
sorry

end NUMINAMATH_CALUDE_line_and_circle_proof_l353_35331


namespace NUMINAMATH_CALUDE_score_difference_is_1_25_l353_35388

-- Define the score distribution
def score_distribution : List (ℝ × ℝ) := [
  (0.15, 60),
  (0.20, 75),
  (0.25, 85),
  (0.30, 95),
  (0.10, 100)
]

-- Calculate the mean score
def mean_score : ℝ := 
  (score_distribution.map (λ p => p.1 * p.2)).sum

-- Define the median score
def median_score : ℝ := 85

-- Theorem statement
theorem score_difference_is_1_25 : 
  median_score - mean_score = 1.25 := by sorry

end NUMINAMATH_CALUDE_score_difference_is_1_25_l353_35388


namespace NUMINAMATH_CALUDE_area_bisectors_perpendicular_l353_35313

/-- Triangle with two sides of length 6 and one side of length 8 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : a = b ∧ a = 6 ∧ c = 8

/-- Area bisector of a triangle -/
def AreaBisector (t : IsoscelesTriangle) := ℝ → ℝ

/-- The angle between two lines -/
def AngleBetween (l1 l2 : ℝ → ℝ) : ℝ := sorry

theorem area_bisectors_perpendicular (t : IsoscelesTriangle) 
  (b1 b2 : AreaBisector t) (h : b1 ≠ b2) : 
  AngleBetween b1 b2 = π / 2 := by sorry

end NUMINAMATH_CALUDE_area_bisectors_perpendicular_l353_35313


namespace NUMINAMATH_CALUDE_auto_finance_to_total_auto_ratio_l353_35342

def total_consumer_credit : ℝ := 855
def auto_finance_credit : ℝ := 57
def auto_credit_percentage : ℝ := 0.20

theorem auto_finance_to_total_auto_ratio :
  let total_auto_credit := total_consumer_credit * auto_credit_percentage
  auto_finance_credit / total_auto_credit = 1/3 := by
sorry

end NUMINAMATH_CALUDE_auto_finance_to_total_auto_ratio_l353_35342


namespace NUMINAMATH_CALUDE_church_distance_l353_35352

theorem church_distance (horse_speed : ℝ) (hourly_rate : ℝ) (flat_fee : ℝ) (total_paid : ℝ) 
  (h1 : horse_speed = 10)
  (h2 : hourly_rate = 30)
  (h3 : flat_fee = 20)
  (h4 : total_paid = 80) : 
  (total_paid - flat_fee) / hourly_rate * horse_speed = 20 := by
  sorry

#check church_distance

end NUMINAMATH_CALUDE_church_distance_l353_35352


namespace NUMINAMATH_CALUDE_solution_set_characterization_l353_35312

/-- A function f: ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The set of real numbers x where xf(x) > 0 -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ := {x | x * f x > 0}

/-- The open interval (-∞, -1) ∪ (1, +∞) -/
def TargetSet : Set ℝ := {x | x < -1 ∨ x > 1}

theorem solution_set_characterization (f : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_positive : ∀ x > 0, f x + x * (deriv f x) > 0)
  (h_zero : f 1 = 0) :
  SolutionSet f = TargetSet := by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l353_35312


namespace NUMINAMATH_CALUDE_cube_surface_area_from_volume_l353_35325

theorem cube_surface_area_from_volume (volume : ℝ) (side_length : ℝ) (surface_area : ℝ) : 
  volume = 729 → 
  volume = side_length ^ 3 → 
  surface_area = 6 * side_length ^ 2 → 
  surface_area = 486 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_volume_l353_35325


namespace NUMINAMATH_CALUDE_barrel_filling_time_l353_35371

theorem barrel_filling_time (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  y - x = 1/4 ∧ 
  66/y - 40/x = 3 →
  40/x = 5 ∨ 40/x = 96 :=
by sorry

end NUMINAMATH_CALUDE_barrel_filling_time_l353_35371


namespace NUMINAMATH_CALUDE_simplification_and_sum_of_squares_l353_35399

/-- The polynomial expression to be simplified -/
def original_expression (x : ℝ) : ℝ :=
  5 * (2 * x^3 - 3 * x^2 + 4) - 6 * (x^4 - 2 * x^3 + 3 * x - 2)

/-- The simplified form of the polynomial expression -/
def simplified_expression (x : ℝ) : ℝ :=
  -6 * x^4 + 22 * x^3 - 15 * x^2 - 18 * x + 32

/-- The coefficients of the simplified expression -/
def coefficients : List ℝ := [-6, 22, -15, -18, 32]

/-- Sum of squares of the coefficients -/
def sum_of_squares : ℝ := (coefficients.map (λ c => c^2)).sum

theorem simplification_and_sum_of_squares :
  (∀ x, original_expression x = simplified_expression x) ∧
  sum_of_squares = 2093 := by
  sorry

end NUMINAMATH_CALUDE_simplification_and_sum_of_squares_l353_35399


namespace NUMINAMATH_CALUDE_max_teams_in_tournament_l353_35323

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played -/
def max_games : ℕ := 200

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- Calculates the total number of games for a given number of teams -/
def total_games (n : ℕ) : ℕ := games_between_teams * (n * (n - 1) / 2)

/-- The theorem stating the maximum number of teams that can participate -/
theorem max_teams_in_tournament : 
  ∃ (n : ℕ), n = 7 ∧ 
  total_games n ≤ max_games ∧ 
  ∀ (m : ℕ), m > n → total_games m > max_games :=
sorry

end NUMINAMATH_CALUDE_max_teams_in_tournament_l353_35323


namespace NUMINAMATH_CALUDE_road_project_completion_time_l353_35374

/-- Represents a road construction project -/
structure RoadProject where
  totalLength : ℝ
  initialWorkers : ℕ
  daysWorked : ℝ
  completedLength : ℝ
  extraWorkers : ℕ

/-- Calculates the total number of days required to complete the road project -/
def totalDaysRequired (project : RoadProject) : ℝ :=
  sorry

/-- Theorem stating that given the project conditions, it will be completed in 15 days -/
theorem road_project_completion_time (project : RoadProject)
  (h1 : project.totalLength = 10)
  (h2 : project.initialWorkers = 30)
  (h3 : project.daysWorked = 5)
  (h4 : project.completedLength = 2)
  (h5 : project.extraWorkers = 30) :
  totalDaysRequired project = 15 :=
sorry

end NUMINAMATH_CALUDE_road_project_completion_time_l353_35374


namespace NUMINAMATH_CALUDE_problem_solution_l353_35341

def A (a : ℝ) := { x : ℝ | a - 1 ≤ x ∧ x ≤ a + 1 }
def B := { x : ℝ | -1 ≤ x ∧ x ≤ 4 }

theorem problem_solution :
  (∀ a : ℝ, a = 2 → A a ∪ B = { x : ℝ | -1 ≤ x ∧ x ≤ 4 }) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ A a → x ∈ B) → 0 ≤ a ∧ a ≤ 3) ∧
  (∀ a : ℝ, A a ∪ B = B → 0 ≤ a ∧ a ≤ 3) ∧
  (∀ a : ℝ, A a ∩ B = ∅ → a < -2 ∨ a > 5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l353_35341


namespace NUMINAMATH_CALUDE_problem_solution_l353_35337

theorem problem_solution (a b : ℝ) (h : a^2 - 2*b^2 - 2 = 0) :
  -3*a^2 + 6*b^2 + 2023 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l353_35337


namespace NUMINAMATH_CALUDE_real_roots_of_quartic_equation_l353_35385

theorem real_roots_of_quartic_equation :
  let f : ℝ → ℝ := λ x => 2 * x^4 + 4 * x^3 + 3 * x^2 + x - 1
  let x₁ : ℝ := (-1 + Real.sqrt 3) / 2
  let x₂ : ℝ := (-1 - Real.sqrt 3) / 2
  (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) ∧ (f x₁ = 0 ∧ f x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_of_quartic_equation_l353_35385


namespace NUMINAMATH_CALUDE_rachel_homework_l353_35321

theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 10 →
  math_pages + reading_pages = 23 →
  reading_pages > math_pages →
  reading_pages - math_pages = 3 :=
by sorry

end NUMINAMATH_CALUDE_rachel_homework_l353_35321


namespace NUMINAMATH_CALUDE_linear_system_solution_l353_35389

theorem linear_system_solution (x y : ℝ) 
  (eq1 : x + 3*y = 20) 
  (eq2 : x + y = 10) : 
  x = 5 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l353_35389


namespace NUMINAMATH_CALUDE_sum_of_ages_l353_35362

/-- Given the ages of a father and son, prove that their sum is 55 years. -/
theorem sum_of_ages (father_age son_age : ℕ) 
  (h1 : father_age = 37) 
  (h2 : son_age = 18) : 
  father_age + son_age = 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l353_35362


namespace NUMINAMATH_CALUDE_julia_number_l353_35305

theorem julia_number (j m : ℂ) : 
  j * m = 48 - 24*I → 
  m = 7 + 4*I → 
  j = 432/65 - 360/65*I := by sorry

end NUMINAMATH_CALUDE_julia_number_l353_35305


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l353_35311

/-- A quadratic function with roots at -3 and 5, and a minimum value of 36 -/
def quadratic (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_coefficient_sum (a b c : ℝ) :
  (∀ x, quadratic a b c x ≥ 36) ∧ 
  quadratic a b c (-3) = 0 ∧ 
  quadratic a b c 5 = 0 →
  a + b + c = 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l353_35311


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_geometric_sequence_general_term_l353_35300

/-- Geometric sequence -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

theorem geometric_sequence_sixth_term :
  let a₁ := 3
  let q := -2
  geometric_sequence a₁ q 6 = -96 := by sorry

theorem geometric_sequence_general_term :
  let a₃ := 20
  let a₆ := 160
  ∃ q : ℝ, ∀ n : ℕ, geometric_sequence (a₃ / q^2) q n = 5 * 2^(n - 1) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_geometric_sequence_general_term_l353_35300


namespace NUMINAMATH_CALUDE_square_properties_l353_35344

/-- Properties of a square with side length 30 cm -/
theorem square_properties :
  let s : ℝ := 30
  let area : ℝ := s^2
  let diagonal : ℝ := s * Real.sqrt 2
  (area = 900 ∧ diagonal = 30 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l353_35344


namespace NUMINAMATH_CALUDE_research_team_probabilities_l353_35351

/-- Represents a research team member -/
structure Member where
  gender : Bool  -- true for male, false for female
  speaksEnglish : Bool

/-- Represents a research team -/
def ResearchTeam : Type := List Member

/-- Creates a research team with the given specifications -/
def createTeam : ResearchTeam :=
  [
    { gender := true, speaksEnglish := false },   -- non-English speaking male
    { gender := true, speaksEnglish := true },    -- English speaking male
    { gender := true, speaksEnglish := true },    -- English speaking male
    { gender := true, speaksEnglish := true },    -- English speaking male
    { gender := false, speaksEnglish := false },  -- non-English speaking female
    { gender := false, speaksEnglish := true }    -- English speaking female
  ]

/-- Calculates the probability of selecting two members with a given property -/
def probabilityOfSelection (team : ResearchTeam) (property : Member → Member → Bool) : Rat :=
  sorry

theorem research_team_probabilities (team : ResearchTeam) 
  (h1 : team = createTeam) : 
  (probabilityOfSelection team (fun m1 m2 => m1.gender = m2.gender) = 7/15) ∧ 
  (probabilityOfSelection team (fun m1 m2 => m1.speaksEnglish ∨ m2.speaksEnglish) = 14/15) ∧
  (probabilityOfSelection team (fun m1 m2 => m1.gender ≠ m2.gender ∧ (m1.speaksEnglish ∨ m2.speaksEnglish)) = 7/15) :=
by sorry

end NUMINAMATH_CALUDE_research_team_probabilities_l353_35351


namespace NUMINAMATH_CALUDE_number_of_distributions_l353_35329

/-- The number of ways to distribute 5 students into 3 groups with constraints -/
def distribution_schemes : ℕ :=
  -- The actual calculation would go here, but we don't have the solution steps
  80

/-- Theorem stating the number of distribution schemes -/
theorem number_of_distributions :
  distribution_schemes = 80 :=
by
  -- The proof would go here
  sorry

#check number_of_distributions

end NUMINAMATH_CALUDE_number_of_distributions_l353_35329


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l353_35314

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two convex polygons in a plane -/
structure TwoPolygonConfig where
  Q₁ : ConvexPolygon
  Q₂ : ConvexPolygon
  same_plane : Bool
  m₁_le_m₂ : Q₁.sides ≤ Q₂.sides
  share_at_most_one_vertex : Bool
  share_no_sides : Bool

/-- The maximum number of intersections between two convex polygons -/
def max_intersections (config : TwoPolygonConfig) : ℕ := 
  config.Q₁.sides * config.Q₂.sides

/-- Theorem: The maximum number of intersections between two convex polygons
    under the given conditions is the product of their number of sides -/
theorem max_intersections_theorem (config : TwoPolygonConfig) : 
  max_intersections config = config.Q₁.sides * config.Q₂.sides := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l353_35314


namespace NUMINAMATH_CALUDE_coefficients_of_equation_l353_35346

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation 3x^2 - x - 2 = 0 -/
def equation : QuadraticEquation :=
  { a := 3, b := -1, c := -2 }

theorem coefficients_of_equation :
  equation.a = 3 ∧ equation.b = -1 ∧ equation.c = -2 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_equation_l353_35346


namespace NUMINAMATH_CALUDE_sample_size_is_300_l353_35393

/-- Represents the population ratios of the districts -/
def district_ratios : List ℕ := [2, 3, 5, 2, 6]

/-- The number of individuals contributed by the largest district -/
def largest_district_contribution : ℕ := 100

/-- Calculates the total sample size based on the district ratios and the contribution of the largest district -/
def calculate_sample_size (ratios : List ℕ) (largest_contribution : ℕ) : ℕ :=
  let total_ratio := ratios.sum
  let largest_ratio := ratios.maximum?
  match largest_ratio with
  | some max_ratio => (total_ratio * largest_contribution) / max_ratio
  | none => 0

/-- Theorem stating that the calculated sample size is 300 -/
theorem sample_size_is_300 :
  calculate_sample_size district_ratios largest_district_contribution = 300 := by
  sorry

#eval calculate_sample_size district_ratios largest_district_contribution

end NUMINAMATH_CALUDE_sample_size_is_300_l353_35393


namespace NUMINAMATH_CALUDE_target_probabilities_l353_35306

/-- Probability of hitting a target -/
structure TargetProbability where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Assumptions about the probabilities -/
axiom prob_bounds (p : TargetProbability) :
  0 ≤ p.A ∧ p.A ≤ 1 ∧
  0 ≤ p.B ∧ p.B ≤ 1 ∧
  0 ≤ p.C ∧ p.C ≤ 1

/-- Given probabilities -/
def given_probs : TargetProbability :=
  { A := 0.7, B := 0.6, C := 0.5 }

/-- Probability of at least one person hitting the target -/
def prob_at_least_one (p : TargetProbability) : ℝ :=
  1 - (1 - p.A) * (1 - p.B) * (1 - p.C)

/-- Probability of exactly two people hitting the target -/
def prob_exactly_two (p : TargetProbability) : ℝ :=
  p.A * p.B * (1 - p.C) + p.A * (1 - p.B) * p.C + (1 - p.A) * p.B * p.C

/-- Probability of hitting exactly k times in n trials -/
def prob_k_of_n (p q : ℝ) (n k : ℕ) : ℝ :=
  (n.choose k : ℝ) * p^k * q^(n - k)

theorem target_probabilities (p : TargetProbability) 
  (h : p = given_probs) : 
  prob_at_least_one p = 0.94 ∧ 
  prob_exactly_two p = 0.44 ∧ 
  prob_k_of_n p.A (1 - p.A) 3 2 = 0.441 := by
  sorry

end NUMINAMATH_CALUDE_target_probabilities_l353_35306


namespace NUMINAMATH_CALUDE_line_circle_intersection_count_l353_35328

/-- The number of intersection points between a line and a circle -/
theorem line_circle_intersection_count (k : ℝ) : 
  ∃ (p q : ℝ × ℝ), p ≠ q ∧ 
  (∀ (x y : ℝ), (k * x - y - k = 0 ∧ x^2 + y^2 = 2) ↔ (x, y) = p ∨ (x, y) = q) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_count_l353_35328


namespace NUMINAMATH_CALUDE_overtime_pay_is_correct_l353_35395

/-- Represents the time interval between minute and hour hand overlaps on a normal clock in minutes -/
def normal_overlap : ℚ := 720 / 11

/-- Represents the time interval between minute and hour hand overlaps on the slow clock in minutes -/
def slow_overlap : ℕ := 69

/-- Represents the normal workday duration in hours -/
def normal_workday : ℕ := 8

/-- Represents the regular hourly pay rate in dollars -/
def regular_rate : ℚ := 4

/-- Represents the overtime pay rate multiplier -/
def overtime_multiplier : ℚ := 3/2

/-- Theorem stating that the overtime pay is $2.60 given the specified conditions -/
theorem overtime_pay_is_correct :
  let actual_time_ratio : ℚ := slow_overlap / normal_overlap
  let actual_time_worked : ℚ := normal_workday * actual_time_ratio
  let overtime_hours : ℚ := actual_time_worked - normal_workday
  let overtime_pay : ℚ := overtime_hours * regular_rate * overtime_multiplier
  overtime_pay = 13/5 := by sorry

end NUMINAMATH_CALUDE_overtime_pay_is_correct_l353_35395


namespace NUMINAMATH_CALUDE_division_result_l353_35373

theorem division_result : (4 : ℚ) / (8 / 13) = 13 / 2 := by sorry

end NUMINAMATH_CALUDE_division_result_l353_35373


namespace NUMINAMATH_CALUDE_log_simplification_l353_35391

theorem log_simplification (p q r s z u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (hz : z > 0) (hu : u > 0) : 
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * z / (s * u)) = Real.log (u / z) := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l353_35391


namespace NUMINAMATH_CALUDE_angle_30_less_than_complement_l353_35332

theorem angle_30_less_than_complement (x : ℝ) : x = 90 - x - 30 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_30_less_than_complement_l353_35332


namespace NUMINAMATH_CALUDE_only_A_scored_full_marks_l353_35378

/-- Represents the three students -/
inductive Student
  | A
  | B
  | C

/-- Represents whether a statement is true or false -/
def Statement := Bool

/-- Represents whether a student scored full marks -/
def ScoredFullMarks := Student → Bool

/-- Represents whether a student told the truth -/
def ToldTruth := Student → Bool

/-- A's statement: C did not score full marks -/
def statementA (s : ScoredFullMarks) : Statement :=
  !s Student.C

/-- B's statement: I scored full marks -/
def statementB (s : ScoredFullMarks) : Statement :=
  s Student.B

/-- C's statement: A is telling the truth -/
def statementC (t : ToldTruth) : Statement :=
  t Student.A

theorem only_A_scored_full_marks :
  ∀ (s : ScoredFullMarks) (t : ToldTruth),
    (∃! x : Student, s x = true) →
    (∃! x : Student, t x = false) →
    (t Student.A = (statementA s)) →
    (t Student.B = (statementB s)) →
    (t Student.C = (statementC t)) →
    s Student.A = true :=
sorry

end NUMINAMATH_CALUDE_only_A_scored_full_marks_l353_35378


namespace NUMINAMATH_CALUDE_paving_stone_size_l353_35397

theorem paving_stone_size (length width : ℝ) (num_stones : ℕ) (stone_side : ℝ) : 
  length = 30 → 
  width = 18 → 
  num_stones = 135 → 
  (length * width) = (num_stones : ℝ) * stone_side^2 → 
  stone_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_paving_stone_size_l353_35397


namespace NUMINAMATH_CALUDE_problem_statement_l353_35353

/-- The equation x^2 - x + a^2 - 6a = 0 has one positive root and one negative root. -/
def p (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - x + a^2 - 6*a = 0 ∧ y^2 - y + a^2 - 6*a = 0

/-- The graph of y = x^2 + (a-3)x + 1 has no common points with the x-axis. -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a-3)*x + 1 ≠ 0

/-- The range of values for a is 0 < a ≤ 1 or 5 ≤ a < 6. -/
def range_of_a (a : ℝ) : Prop :=
  (0 < a ∧ a ≤ 1) ∨ (5 ≤ a ∧ a < 6)

theorem problem_statement (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l353_35353


namespace NUMINAMATH_CALUDE_cans_bought_with_euros_l353_35358

/-- The number of cans of soda that can be bought for a given amount of euros. -/
def cans_per_euros (T R E : ℚ) : ℚ :=
  (5 * E * T) / R

/-- Given that T cans of soda can be purchased for R quarters,
    and 1 euro is equivalent to 5 quarters,
    the number of cans of soda that can be bought for E euros is (5ET)/R -/
theorem cans_bought_with_euros (T R E : ℚ) (hT : T > 0) (hR : R > 0) (hE : E ≥ 0) :
  cans_per_euros T R E = (5 * E * T) / R :=
by sorry

end NUMINAMATH_CALUDE_cans_bought_with_euros_l353_35358


namespace NUMINAMATH_CALUDE_two_cyclists_problem_l353_35367

/-- Two cyclists problem -/
theorem two_cyclists_problem (MP : ℝ) : 
  (∀ (t : ℝ), t > 0 → 
    (MP / t = 42 / ((420 / (MP + 30)) + 1/3)) ∧
    (MP + 30) / t = 42 / (420 / MP)) →
  MP = 180 := by
sorry

end NUMINAMATH_CALUDE_two_cyclists_problem_l353_35367


namespace NUMINAMATH_CALUDE_floor_sqrt_50_l353_35360

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_l353_35360


namespace NUMINAMATH_CALUDE_multiply_72515_9999_l353_35398

theorem multiply_72515_9999 : 72515 * 9999 = 725077485 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72515_9999_l353_35398


namespace NUMINAMATH_CALUDE_point_between_parallel_lines_l353_35333

-- Define the two line equations
def line1 (x y : ℝ) : Prop := 6 * x - 8 * y + 1 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

-- Define what it means for a point to be between two lines
def between_lines (x y : ℝ) : Prop :=
  (line1 x y ∧ ¬line2 x y) ∨ (¬line1 x y ∧ line2 x y) ∨ (¬line1 x y ∧ ¬line2 x y)

-- Theorem statement
theorem point_between_parallel_lines :
  between_lines 5 b → b = 4 := by sorry

end NUMINAMATH_CALUDE_point_between_parallel_lines_l353_35333


namespace NUMINAMATH_CALUDE_one_pair_three_different_probability_l353_35324

def total_socks : ℕ := 12
def socks_per_color : ℕ := 3
def num_colors : ℕ := 4
def drawn_socks : ℕ := 5

def probability_one_pair_three_different : ℚ :=
  27 / 66

theorem one_pair_three_different_probability :
  (total_socks = socks_per_color * num_colors) →
  (probability_one_pair_three_different =
    (num_colors * (socks_per_color.choose 2) *
     (socks_per_color ^ (num_colors - 1))) /
    (total_socks.choose drawn_socks)) :=
by sorry

end NUMINAMATH_CALUDE_one_pair_three_different_probability_l353_35324


namespace NUMINAMATH_CALUDE_repeating_decimal_interval_l353_35340

/-- A number is a repeating decimal with period p if it can be expressed as m / (10^p - 1) for some integer m. -/
def is_repeating_decimal (x : ℚ) (p : ℕ) : Prop :=
  ∃ (m : ℤ), x = m / (10^p - 1)

theorem repeating_decimal_interval :
  ∀ n : ℕ,
    n < 2000 →
    is_repeating_decimal (1 / n) 8 →
    is_repeating_decimal (1 / (n + 6)) 6 →
    801 ≤ n ∧ n ≤ 1200 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_interval_l353_35340


namespace NUMINAMATH_CALUDE_total_players_l353_35347

/-- The total number of players in a game scenario -/
theorem total_players (kabaddi : ℕ) (kho_kho_only : ℕ) (both : ℕ) :
  kabaddi = 10 →
  kho_kho_only = 15 →
  both = 5 →
  kabaddi + kho_kho_only - both = 25 := by
sorry

end NUMINAMATH_CALUDE_total_players_l353_35347


namespace NUMINAMATH_CALUDE_age_order_l353_35316

structure Person where
  name : String
  age : ℕ

def age_relationship (sergei sasha tolia : Person) : Prop :=
  sergei.age = 2 * (sergei.age + tolia.age - sergei.age)

theorem age_order (sergei sasha tolia : Person) 
  (h : age_relationship sergei sasha tolia) : 
  sergei.age > tolia.age ∧ tolia.age > sasha.age :=
by
  sorry

#check age_order

end NUMINAMATH_CALUDE_age_order_l353_35316


namespace NUMINAMATH_CALUDE_simplification_proof_l353_35322

theorem simplification_proof (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) : 
  (a - 1) / (a^2 - 1) + 1 / (a + 1) = 2 / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplification_proof_l353_35322


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l353_35318

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, -1 + y)
  are_parallel a b → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l353_35318


namespace NUMINAMATH_CALUDE_dogs_not_liking_any_food_l353_35392

/-- Given a kennel of dogs with specified food preferences, prove the number of dogs
    that don't like any of watermelon, salmon, or chicken. -/
theorem dogs_not_liking_any_food (total : ℕ) (watermelon salmon chicken : ℕ)
  (watermelon_and_salmon watermelon_and_chicken_not_salmon salmon_and_chicken_not_watermelon : ℕ)
  (h1 : total = 80)
  (h2 : watermelon = 21)
  (h3 : salmon = 58)
  (h4 : watermelon_and_salmon = 12)
  (h5 : chicken = 15)
  (h6 : watermelon_and_chicken_not_salmon = 7)
  (h7 : salmon_and_chicken_not_watermelon = 10) :
  total - (watermelon_and_salmon + (salmon - watermelon_and_salmon - salmon_and_chicken_not_watermelon) +
           (watermelon - watermelon_and_salmon - watermelon_and_chicken_not_salmon) +
           salmon_and_chicken_not_watermelon + watermelon_and_chicken_not_salmon) = 13 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_liking_any_food_l353_35392


namespace NUMINAMATH_CALUDE_geometric_sequence_differences_l353_35375

/-- The type of sequences of real numbers of length n -/
def RealSequence (n : ℕ) := Fin n → ℝ

/-- The condition that a sequence is strictly increasing -/
def StrictlyIncreasing {n : ℕ} (a : RealSequence n) : Prop :=
  ∀ i j : Fin n, i < j → a i < a j

/-- The set of differences between elements of a sequence -/
def Differences {n : ℕ} (a : RealSequence n) : Set ℝ :=
  {x : ℝ | ∃ i j : Fin n, i < j ∧ x = a j - a i}

/-- The set of powers of r from 1 to k -/
def PowerSet (r : ℝ) (k : ℕ) : Set ℝ :=
  {x : ℝ | ∃ m : ℕ, m ≤ k ∧ x = r ^ m}

/-- The main theorem -/
theorem geometric_sequence_differences (n : ℕ) (h : n ≥ 2) :
  (∃ (a : RealSequence n) (r : ℝ),
    StrictlyIncreasing a ∧
    r > 0 ∧
    Differences a = PowerSet r (n * (n - 1) / 2)) ↔
  n = 2 ∨ n = 3 ∨ n = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_differences_l353_35375


namespace NUMINAMATH_CALUDE_power_sum_equals_eight_l353_35310

theorem power_sum_equals_eight :
  (-2)^3 + (-2)^2 + (-2)^1 + 2^1 + 2^2 + 2^3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_eight_l353_35310


namespace NUMINAMATH_CALUDE_infinite_product_a_l353_35363

noncomputable def a : ℕ → ℚ
  | 0 => 2/3
  | n+1 => 1 + (a n - 1)^2

theorem infinite_product_a : ∏' n, a n = 1/2 := by sorry

end NUMINAMATH_CALUDE_infinite_product_a_l353_35363


namespace NUMINAMATH_CALUDE_triangle_equilateral_if_angles_arithmetic_and_geometric_l353_35355

theorem triangle_equilateral_if_angles_arithmetic_and_geometric :
  ∀ (a b c : ℝ),
  -- The angles form an arithmetic sequence
  (∃ d : ℝ, b = a + d ∧ c = b + d) →
  -- The angles form a geometric sequence
  (∃ r : ℝ, b = a * r ∧ c = b * r) →
  -- The sum of angles is 180°
  a + b + c = 180 →
  -- The triangle is equilateral (all angles are equal)
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_triangle_equilateral_if_angles_arithmetic_and_geometric_l353_35355


namespace NUMINAMATH_CALUDE_spells_base7_to_base10_l353_35370

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The number of spells in base 7 --/
def spellsBase7 : Nat := 653

/-- Theorem: The number of spells in base 7 (653) is equal to 332 in base 10 --/
theorem spells_base7_to_base10 :
  base7ToBase10 (spellsBase7 / 100) ((spellsBase7 / 10) % 10) (spellsBase7 % 10) = 332 := by
  sorry

end NUMINAMATH_CALUDE_spells_base7_to_base10_l353_35370


namespace NUMINAMATH_CALUDE_sister_age_l353_35376

theorem sister_age (B S : ℕ) (h : B = B * S) : S = 1 := by
  sorry

end NUMINAMATH_CALUDE_sister_age_l353_35376


namespace NUMINAMATH_CALUDE_factor_expression_l353_35382

theorem factor_expression (y z : ℝ) : 3 * y^2 - 75 * z^2 = 3 * (y + 5 * z) * (y - 5 * z) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l353_35382


namespace NUMINAMATH_CALUDE_new_students_count_l353_35377

theorem new_students_count (initial_students : ℕ) (left_students : ℕ) (final_students : ℕ) :
  initial_students = 10 →
  left_students = 4 →
  final_students = 48 →
  final_students - (initial_students - left_students) = 42 :=
by sorry

end NUMINAMATH_CALUDE_new_students_count_l353_35377


namespace NUMINAMATH_CALUDE_complex_equation_solution_l353_35379

theorem complex_equation_solution (z : ℂ) 
  (h : 12 * Complex.abs z ^ 2 = 2 * Complex.abs (z + 2) ^ 2 + Complex.abs (z ^ 2 + 1) ^ 2 + 31) :
  z + 6 / z = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l353_35379


namespace NUMINAMATH_CALUDE_winning_candidate_votes_l353_35319

/-- Proves that the winning candidate received 11628 votes in the described election scenario -/
theorem winning_candidate_votes :
  let total_votes : ℝ := (4136 + 7636) / (1 - 0.4969230769230769)
  let winning_votes : ℝ := 0.4969230769230769 * total_votes
  ⌊winning_votes⌋ = 11628 := by
  sorry

end NUMINAMATH_CALUDE_winning_candidate_votes_l353_35319


namespace NUMINAMATH_CALUDE_base8_digit_sum_l353_35327

/-- Represents a digit in base 8 -/
def Digit8 : Type := { n : ℕ // n > 0 ∧ n < 8 }

/-- Converts a three-digit number in base 8 to its decimal equivalent -/
def toDecimal (p q r : Digit8) : ℕ := 64 * p.val + 8 * q.val + r.val

/-- The sum of three permutations of digits in base 8 -/
def sumPermutations (p q r : Digit8) : ℕ :=
  toDecimal p q r + toDecimal r q p + toDecimal q p r

/-- The value of PPP0 in base 8 -/
def ppp0 (p : Digit8) : ℕ := 512 * p.val + 64 * p.val + 8 * p.val

theorem base8_digit_sum (p q r : Digit8) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_sum : sumPermutations p q r = ppp0 p) : 
  q.val + r.val = 7 := by sorry

end NUMINAMATH_CALUDE_base8_digit_sum_l353_35327


namespace NUMINAMATH_CALUDE_largest_non_odd_units_digit_proof_l353_35359

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

def units_digit (n : ℕ) : ℕ := n % 10

def largest_non_odd_units_digit : ℕ := 8

theorem largest_non_odd_units_digit_proof :
  ∀ d : ℕ, d ≤ 9 →
    (d > largest_non_odd_units_digit →
      ∃ n : ℕ, is_odd n ∧ units_digit n = d) ∧
    (d ≤ largest_non_odd_units_digit →
      d = largest_non_odd_units_digit ∨
      ∀ n : ℕ, is_odd n → units_digit n ≠ d) :=
sorry

end NUMINAMATH_CALUDE_largest_non_odd_units_digit_proof_l353_35359


namespace NUMINAMATH_CALUDE_smallest_root_of_equation_l353_35343

theorem smallest_root_of_equation (x : ℚ) : 
  (x - 5/6)^2 + (x - 5/6)*(x - 2/3) = 0 ∧ x^2 - 2*x + 1 ≥ 0 → 
  x ≥ 5/6 ∧ (∀ y : ℚ, y < 5/6 → (y - 5/6)^2 + (y - 5/6)*(y - 2/3) ≠ 0 ∨ y^2 - 2*y + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_root_of_equation_l353_35343


namespace NUMINAMATH_CALUDE_lottery_possibility_l353_35330

theorem lottery_possibility (win_chance : ℝ) (h : win_chance = 0.01) : 
  ∃ (outcome : Bool), outcome = true :=
sorry

end NUMINAMATH_CALUDE_lottery_possibility_l353_35330


namespace NUMINAMATH_CALUDE_intersection_M_N_l353_35390

def M : Set ℝ := {x | x / (x - 1) ≥ 0}

def N : Set ℝ := {y | ∃ x, y = 3 * x^2 + 1}

theorem intersection_M_N : M ∩ N = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l353_35390


namespace NUMINAMATH_CALUDE_backpack_cost_theorem_l353_35301

/-- Calculates the total cost of personalized backpacks with a discount -/
def total_cost (num_backpacks : ℕ) (original_price : ℚ) (discount_rate : ℚ) (monogram_fee : ℚ) : ℚ :=
  let discounted_price := original_price * (1 - discount_rate)
  let total_discounted := num_backpacks.cast * discounted_price
  let total_monogram := num_backpacks.cast * monogram_fee
  total_discounted + total_monogram

/-- Theorem stating that the total cost of 5 backpacks with given prices and discount is $140.00 -/
theorem backpack_cost_theorem :
  total_cost 5 20 (1/5) 12 = 140 := by
  sorry

end NUMINAMATH_CALUDE_backpack_cost_theorem_l353_35301


namespace NUMINAMATH_CALUDE_water_jars_count_l353_35334

/-- Proves that 7 gallons of water stored in equal numbers of quart, half-gallon, and one-gallon jars results in 12 water-filled jars -/
theorem water_jars_count (total_water : ℚ) (jar_sizes : Fin 3 → ℚ) :
  total_water = 7 →
  jar_sizes 0 = 1/4 →
  jar_sizes 1 = 1/2 →
  jar_sizes 2 = 1 →
  ∃ (x : ℚ), x > 0 ∧ x * (jar_sizes 0 + jar_sizes 1 + jar_sizes 2) = total_water ∧
  (3 * x : ℚ) = 12 :=
by sorry

end NUMINAMATH_CALUDE_water_jars_count_l353_35334


namespace NUMINAMATH_CALUDE_apps_deleted_l353_35302

theorem apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) 
  (h1 : initial_apps = 16) (h2 : remaining_apps = 8) : 
  initial_apps - remaining_apps = 8 := by
  sorry

end NUMINAMATH_CALUDE_apps_deleted_l353_35302


namespace NUMINAMATH_CALUDE_smallest_square_side_l353_35309

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents a partition of a square into smaller squares -/
structure Partition where
  total : ℕ
  unit_squares : ℕ
  other_squares : List ℕ

/-- Checks if a partition is valid for a given square -/
def is_valid_partition (s : Square) (p : Partition) : Prop :=
  p.total = 15 ∧
  p.unit_squares = 12 ∧
  p.other_squares.length = 3 ∧
  (p.unit_squares + p.other_squares.sum) = s.side * s.side ∧
  ∀ x ∈ p.other_squares, x > 0

/-- The theorem stating the smallest possible square side length -/
theorem smallest_square_side : 
  ∃ (s : Square) (p : Partition), 
    is_valid_partition s p ∧ 
    (∀ (s' : Square) (p' : Partition), is_valid_partition s' p' → s.side ≤ s'.side) ∧
    s.side = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_side_l353_35309


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l353_35350

-- Define the coefficients of the two lines as functions of a
def line1_coeff (a : ℝ) : ℝ × ℝ := (1 - a, a)
def line2_coeff (a : ℝ) : ℝ × ℝ := (2*a + 3, a - 1)

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop :=
  (line1_coeff a).1 * (line2_coeff a).1 + (line1_coeff a).2 * (line2_coeff a).2 = 0

-- State the theorem
theorem perpendicular_lines_a_values :
  ∀ a : ℝ, perpendicular a → a = 1 ∨ a = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l353_35350


namespace NUMINAMATH_CALUDE_remainder_problem_l353_35366

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 34 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l353_35366


namespace NUMINAMATH_CALUDE_sam_winning_probability_l353_35354

theorem sam_winning_probability :
  let hit_probability : ℚ := 2/5
  let miss_probability : ℚ := 3/5
  let p : ℚ := p -- p represents the probability of Sam winning
  (hit_probability = 2/5) →
  (miss_probability = 3/5) →
  (p = hit_probability + miss_probability * miss_probability * p) →
  p = 5/8 := by
sorry

end NUMINAMATH_CALUDE_sam_winning_probability_l353_35354


namespace NUMINAMATH_CALUDE_not_right_triangle_l353_35364

theorem not_right_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 3 * C) 
  (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_l353_35364
