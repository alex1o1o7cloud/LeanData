import Mathlib

namespace NUMINAMATH_CALUDE_arc_length_45_degrees_l2574_257435

/-- Given a circle with circumference 72 meters and an arc subtended by a 45° central angle,
    the length of the arc is 9 meters. -/
theorem arc_length_45_degrees (D : ℝ) (EF : ℝ) : 
  D = 72 →  -- circumference of the circle
  EF = (45 / 360) * D →  -- arc length as a fraction of the circumference
  EF = 9 := by sorry

end NUMINAMATH_CALUDE_arc_length_45_degrees_l2574_257435


namespace NUMINAMATH_CALUDE_sin_two_theta_l2574_257400

theorem sin_two_theta (θ : Real) 
  (h : Real.exp (Real.log 2 * (-2 + 3 * Real.sin θ)) + 1 = Real.exp (Real.log 2 * (1/2 + Real.sin θ))) : 
  Real.sin (2 * θ) = 4 * Real.sqrt 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_sin_two_theta_l2574_257400


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l2574_257403

/-- The value of m for which the circle x^2 + y^2 + mx - 1/4 = 0 is tangent to the directrix of the parabola y^2 = 4x -/
theorem circle_tangent_to_parabola_directrix (x y m : ℝ) : 
  (∃ x y, x^2 + y^2 + m*x - 1/4 = 0) → -- Circle equation
  (∃ x y, y^2 = 4*x) → -- Parabola equation
  (∃ x y, x^2 + y^2 + m*x - 1/4 = 0 ∧ x = -1) → -- Circle is tangent to directrix (x = -1)
  m = 3/4 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l2574_257403


namespace NUMINAMATH_CALUDE_union_equals_interval_l2574_257449

def A : Set ℝ := {1, 2, 3, 4}

def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

theorem union_equals_interval (a : ℝ) :
  A ∪ B a = Set.Iic 5 → a = 5 := by sorry

end NUMINAMATH_CALUDE_union_equals_interval_l2574_257449


namespace NUMINAMATH_CALUDE_a_less_than_one_sufficient_not_necessary_l2574_257419

-- Define the equation
def circle_equation (x y a : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 + 2 * a * x + 6 * y + 5 * a = 0

-- Define what it means for the equation to represent a circle
def is_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

-- Theorem stating that a < 1 is sufficient but not necessary
theorem a_less_than_one_sufficient_not_necessary :
  (∀ a : ℝ, a < 1 → is_circle a) ∧
  ¬(∀ a : ℝ, is_circle a → a < 1) :=
sorry

end NUMINAMATH_CALUDE_a_less_than_one_sufficient_not_necessary_l2574_257419


namespace NUMINAMATH_CALUDE_expression_evaluation_l2574_257433

theorem expression_evaluation (x : ℝ) (hx : x^2 - 2*x - 3 = 0) (hx_neq : x ≠ 3) :
  (2 / (x - 3) - 1 / x) * ((x^2 - 3*x) / (x^2 + 6*x + 9)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2574_257433


namespace NUMINAMATH_CALUDE_attendance_decrease_l2574_257425

/-- Proves that given a projected 25 percent increase in attendance and actual attendance being 64 percent of the projected attendance, the actual percent decrease in attendance is 20 percent. -/
theorem attendance_decrease (P : ℝ) (P_positive : P > 0) : 
  let projected_attendance := 1.25 * P
  let actual_attendance := 0.64 * projected_attendance
  let percent_decrease := (P - actual_attendance) / P * 100
  percent_decrease = 20 := by
  sorry

end NUMINAMATH_CALUDE_attendance_decrease_l2574_257425


namespace NUMINAMATH_CALUDE_shekars_social_studies_score_l2574_257405

/-- Given Shekar's scores in four subjects and his average marks across all five subjects,
    prove that his marks in social studies must be 82. -/
theorem shekars_social_studies_score
  (math_score : ℕ)
  (science_score : ℕ)
  (english_score : ℕ)
  (biology_score : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 47)
  (h4 : biology_score = 85)
  (h5 : average_marks = 71)
  (h6 : num_subjects = 5)
  : ∃ (social_studies_score : ℕ),
    social_studies_score = 82 ∧
    (math_score + science_score + english_score + biology_score + social_studies_score) / num_subjects = average_marks :=
by
  sorry


end NUMINAMATH_CALUDE_shekars_social_studies_score_l2574_257405


namespace NUMINAMATH_CALUDE_inspection_arrangements_l2574_257429

/-- Represents the number of liberal arts classes -/
def liberal_arts_classes : ℕ := 2

/-- Represents the number of science classes -/
def science_classes : ℕ := 4

/-- Represents the total number of classes -/
def total_classes : ℕ := liberal_arts_classes + science_classes

/-- Represents the number of ways to choose inspectors from science classes for liberal arts classes -/
def science_to_liberal_arts : ℕ := science_classes * (science_classes - 1)

/-- Represents the number of ways to arrange inspections within science classes -/
def science_arrangements : ℕ := 
  liberal_arts_classes * (liberal_arts_classes - 1) * (science_classes - 2) * (science_classes - 3) +
  liberal_arts_classes * (liberal_arts_classes - 1) +
  liberal_arts_classes * liberal_arts_classes * (science_classes - 2)

/-- The main theorem stating the total number of inspection arrangements -/
theorem inspection_arrangements : 
  science_to_liberal_arts * science_arrangements = 168 := by
  sorry

end NUMINAMATH_CALUDE_inspection_arrangements_l2574_257429


namespace NUMINAMATH_CALUDE_negative_765_degrees_conversion_l2574_257499

theorem negative_765_degrees_conversion :
  ∃ (k : ℤ) (α : ℝ), 
    (-765 : ℝ) * π / 180 = 2 * k * π + α ∧ 
    0 ≤ α ∧ 
    α < 2 * π ∧ 
    k = -3 ∧ 
    α = 7 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_negative_765_degrees_conversion_l2574_257499


namespace NUMINAMATH_CALUDE_evaluate_expression_l2574_257448

theorem evaluate_expression : 
  (125 : ℝ) ^ (1/3) / (64 : ℝ) ^ (1/2) * (81 : ℝ) ^ (1/4) = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2574_257448


namespace NUMINAMATH_CALUDE_course_size_l2574_257445

theorem course_size (num_d : ℕ) (h_d : num_d = 25) :
  ∃ (total : ℕ),
    total > 0 ∧
    (total : ℚ) = num_d + (1/5 : ℚ) * total + (1/4 : ℚ) * total + (1/2 : ℚ) * total ∧
    total = 500 := by
  sorry

end NUMINAMATH_CALUDE_course_size_l2574_257445


namespace NUMINAMATH_CALUDE_ab_value_l2574_257427

theorem ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^2 + b^2 = 9) (h2 : a^4 + b^4 = 65) : a * b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2574_257427


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_l2574_257463

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define the set of ball colors
inductive BallColor : Type
| Red : BallColor
| Black : BallColor
| White : BallColor

-- Define a distribution of balls to people
def Distribution := Person → BallColor

-- Define the event "A receives the white ball"
def event_A_white (d : Distribution) : Prop :=
  d Person.A = BallColor.White

-- Define the event "B receives the white ball"
def event_B_white (d : Distribution) : Prop :=
  d Person.B = BallColor.White

-- Theorem stating that the events are mutually exclusive
theorem events_mutually_exclusive :
  ∀ (d : Distribution), ¬(event_A_white d ∧ event_B_white d) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_l2574_257463


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2574_257415

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2574_257415


namespace NUMINAMATH_CALUDE_unknown_rope_length_l2574_257493

/-- Calculates the length of an unknown rope given other rope lengths and conditions --/
theorem unknown_rope_length
  (known_ropes : List ℝ)
  (knot_loss : ℝ)
  (final_length : ℝ)
  (h1 : known_ropes = [8, 20, 2, 2, 2])
  (h2 : knot_loss = 1.2)
  (h3 : final_length = 35) :
  ∃ x : ℝ, x = 5.8 ∧ 
    final_length + (known_ropes.length * knot_loss) = 
    (known_ropes.sum + x) := by
  sorry


end NUMINAMATH_CALUDE_unknown_rope_length_l2574_257493


namespace NUMINAMATH_CALUDE_quadratic_composition_theorem_l2574_257442

/-- A quadratic polynomial is a polynomial of degree 2 -/
def QuadraticPolynomial (R : Type*) [CommRing R] := {p : Polynomial R // p.degree = 2}

theorem quadratic_composition_theorem {R : Type*} [CommRing R] :
  ∀ (f : QuadraticPolynomial R),
  ∃ (g h : QuadraticPolynomial R),
  (f.val * (f.val.comp (Polynomial.X + 1))) = g.val.comp h.val :=
sorry

end NUMINAMATH_CALUDE_quadratic_composition_theorem_l2574_257442


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2574_257438

theorem negative_fraction_comparison : -5/6 < -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2574_257438


namespace NUMINAMATH_CALUDE_sector_central_angle_l2574_257494

/-- Given a circle with circumference 2π + 2 and a sector of that circle with arc length 2π - 2,
    the central angle of the sector is π - 1. -/
theorem sector_central_angle (r : ℝ) (α : ℝ) 
    (h_circumference : 2 * π * r = 2 * π + 2)
    (h_arc_length : r * α = 2 * π - 2) : 
  α = π - 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2574_257494


namespace NUMINAMATH_CALUDE_toy_car_growth_l2574_257423

theorem toy_car_growth (initial_count : ℕ) (growth_factor : ℚ) (final_multiplier : ℕ) : 
  initial_count = 50 → growth_factor = 5/2 → final_multiplier = 3 →
  (↑initial_count * growth_factor * ↑final_multiplier : ℚ) = 375 := by
  sorry

end NUMINAMATH_CALUDE_toy_car_growth_l2574_257423


namespace NUMINAMATH_CALUDE_negation_equivalence_l2574_257437

-- Define the set [-1, 3]
def interval : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- Define the original proposition
def original_prop (a : ℝ) : Prop := ∀ x ∈ interval, x^2 - a ≥ 0

-- Define the negation of the proposition
def negation_prop (a : ℝ) : Prop := ∃ x ∈ interval, x^2 - a < 0

-- Theorem stating that the negation of the original proposition is equivalent to negation_prop
theorem negation_equivalence (a : ℝ) : ¬(original_prop a) ↔ negation_prop a := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2574_257437


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l2574_257414

theorem mixed_number_calculation : 
  25 * ((5 + 2/7) - (3 + 3/5)) / ((3 + 1/6) + (2 + 1/4)) = 7 + 49/91 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l2574_257414


namespace NUMINAMATH_CALUDE_quadratic_two_roots_range_quadratic_solutions_when_k_is_one_l2574_257401

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := k * x^2 - (2*k + 4) * x + k - 6

-- Theorem for the range of k
theorem quadratic_two_roots_range (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) ↔ 
  (k > -2/5 ∧ k ≠ 0) :=
sorry

-- Theorem for solutions when k = 1
theorem quadratic_solutions_when_k_is_one :
  ∃ x y : ℝ, x ≠ y ∧ 
  quadratic 1 x = 0 ∧ quadratic 1 y = 0 ∧
  x = 3 + Real.sqrt 14 ∧ y = 3 - Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_range_quadratic_solutions_when_k_is_one_l2574_257401


namespace NUMINAMATH_CALUDE_expand_expression_l2574_257481

theorem expand_expression (x : ℝ) : 5 * (4 * x^3 - 3 * x^2 + 2 * x - 7) = 20 * x^3 - 15 * x^2 + 10 * x - 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2574_257481


namespace NUMINAMATH_CALUDE_total_money_value_l2574_257439

def gold_value : ℕ := 75
def silver_value : ℕ := 40
def bronze_value : ℕ := 20
def titanium_value : ℕ := 10

def gold_count : ℕ := 6
def silver_count : ℕ := 8
def bronze_count : ℕ := 10
def titanium_count : ℕ := 4

def cash : ℕ := 45

theorem total_money_value :
  gold_value * gold_count +
  silver_value * silver_count +
  bronze_value * bronze_count +
  titanium_value * titanium_count +
  cash = 1055 := by sorry

end NUMINAMATH_CALUDE_total_money_value_l2574_257439


namespace NUMINAMATH_CALUDE_f_5_equals_142_l2574_257452

-- Define the function f
def f (x : ℝ) (y : ℝ) : ℝ := 2 * x^2 + y

-- Theorem statement
theorem f_5_equals_142 :
  ∃ y : ℝ, f 2 y = 100 → f 5 y = 142 :=
by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_142_l2574_257452


namespace NUMINAMATH_CALUDE_decimal_24_equals_binary_11000_l2574_257469

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_24_equals_binary_11000 : 
  to_binary 24 = [false, false, false, true, true] ∧ 
  from_binary [false, false, false, true, true] = 24 := by
  sorry

end NUMINAMATH_CALUDE_decimal_24_equals_binary_11000_l2574_257469


namespace NUMINAMATH_CALUDE_min_value_expression_l2574_257443

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.sqrt ((2 * x^2 + y^2) * (4 * x^2 + y^2))) / (x * y) ≥ 2 * Real.rpow 3 (1/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2574_257443


namespace NUMINAMATH_CALUDE_opposite_of_blue_is_white_l2574_257441

/-- Represents the colors of the squares --/
inductive Color
| Red | Blue | Orange | Purple | Green | Yellow | White

/-- Represents the positions on the cube --/
inductive Position
| Top | Bottom | Front | Back | Left | Right

/-- Represents a cube configuration --/
structure CubeConfig where
  top : Color
  bottom : Color
  front : Color
  back : Color
  left : Color
  right : Color

/-- Defines the property of opposite faces --/
def isOpposite (p1 p2 : Position) : Prop :=
  (p1 = Position.Top ∧ p2 = Position.Bottom) ∨
  (p1 = Position.Bottom ∧ p2 = Position.Top) ∨
  (p1 = Position.Front ∧ p2 = Position.Back) ∨
  (p1 = Position.Back ∧ p2 = Position.Front) ∨
  (p1 = Position.Left ∧ p2 = Position.Right) ∨
  (p1 = Position.Right ∧ p2 = Position.Left)

/-- The main theorem --/
theorem opposite_of_blue_is_white 
  (cube : CubeConfig)
  (top_is_purple : cube.top = Color.Purple)
  (front_is_green : cube.front = Color.Green)
  (blue_on_side : cube.left = Color.Blue ∨ cube.right = Color.Blue) :
  (cube.left = Color.Blue ∧ cube.right = Color.White) ∨ 
  (cube.right = Color.Blue ∧ cube.left = Color.White) :=
sorry

end NUMINAMATH_CALUDE_opposite_of_blue_is_white_l2574_257441


namespace NUMINAMATH_CALUDE_electric_car_charging_cost_l2574_257402

/-- Calculates the total cost of charging an electric car -/
def total_charging_cost (charges_per_week : ℕ) (num_weeks : ℕ) (cost_per_charge : ℚ) : ℚ :=
  (charges_per_week * num_weeks : ℕ) * cost_per_charge

/-- Proves that the total cost of charging an electric car under given conditions is $121.68 -/
theorem electric_car_charging_cost :
  total_charging_cost 3 52 (78/100) = 12168/100 := by
  sorry

end NUMINAMATH_CALUDE_electric_car_charging_cost_l2574_257402


namespace NUMINAMATH_CALUDE_soda_price_calculation_l2574_257461

def initial_amount : ℕ := 500
def rice_packets : ℕ := 2
def rice_price : ℕ := 20
def wheat_packets : ℕ := 3
def wheat_price : ℕ := 25
def remaining_balance : ℕ := 235

theorem soda_price_calculation :
  ∃ (soda_price : ℕ),
    initial_amount - (rice_packets * rice_price + wheat_packets * wheat_price + soda_price) = remaining_balance ∧
    soda_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_calculation_l2574_257461


namespace NUMINAMATH_CALUDE_two_over_x_is_proper_convert_improper_to_mixed_integer_values_for_integer_result_l2574_257467

-- Define proper and improper expressions
def is_proper_expression (num denom : Polynomial ℚ) : Prop :=
  num.degree < denom.degree

def is_improper_expression (num denom : Polynomial ℚ) : Prop :=
  num.degree ≥ denom.degree

-- Statement 1
theorem two_over_x_is_proper :
  is_proper_expression (2 : Polynomial ℚ) (X : Polynomial ℚ) :=
sorry

-- Statement 2
theorem convert_improper_to_mixed :
  (X^2 - 1) / (X + 2) = X - 2 + 3 / (X + 2) :=
sorry

-- Statement 3
theorem integer_values_for_integer_result :
  {x : ℤ | ∃ (y : ℤ), (2*x - 1) / (x + 1) = y} = {0, -2, 2, -4} :=
sorry

end NUMINAMATH_CALUDE_two_over_x_is_proper_convert_improper_to_mixed_integer_values_for_integer_result_l2574_257467


namespace NUMINAMATH_CALUDE_framed_picture_perimeter_is_six_feet_l2574_257408

/-- Calculates the perimeter of a framed picture given original dimensions and scaling factor. -/
def framedPicturePerimeter (width height scale border : ℚ) : ℚ :=
  2 * (width * scale + height * scale + 2 * border)

/-- Converts inches to feet -/
def inchesToFeet (inches : ℚ) : ℚ :=
  inches / 12

theorem framed_picture_perimeter_is_six_feet :
  let originalWidth : ℚ := 3
  let originalHeight : ℚ := 5
  let scaleFactor : ℚ := 3
  let borderWidth : ℚ := 3
  
  inchesToFeet (framedPicturePerimeter originalWidth originalHeight scaleFactor borderWidth) = 6 := by
  sorry

end NUMINAMATH_CALUDE_framed_picture_perimeter_is_six_feet_l2574_257408


namespace NUMINAMATH_CALUDE_hot_pot_restaurant_problem_l2574_257470

-- Define variables for set prices
variable (price_A price_B : ℚ)

-- Define variables for daily quantities and income
variable (day1_A day1_B day2_A day2_B : ℕ)
variable (income1 income2 : ℚ)

-- Define variables for costs and constraints
variable (cost_A cost_B : ℚ)
variable (max_sets : ℕ)
variable (set_A_ratio : ℚ)

-- Define variables for extra ingredients
variable (extra_cost : ℚ)

-- Define variables for Xiaoming's spending
variable (xiaoming_total : ℚ)
variable (xiaoming_set_A_ratio : ℚ)

-- Theorem statement
theorem hot_pot_restaurant_problem 
  (h1 : day1_A * price_A + day1_B * price_B = income1)
  (h2 : day2_A * price_A + day2_B * price_B = income2)
  (h3 : day1_A = 20 ∧ day1_B = 10 ∧ income1 = 2800)
  (h4 : day2_A = 15 ∧ day2_B = 20 ∧ income2 = 3350)
  (h5 : cost_A = 45 ∧ cost_B = 50)
  (h6 : max_sets = 50)
  (h7 : set_A_ratio = 1/5)
  (h8 : extra_cost = 10)
  (h9 : xiaoming_total = 1610)
  (h10 : xiaoming_set_A_ratio = 1/4) :
  price_A = 90 ∧ 
  price_B = 100 ∧ 
  (∃ (m : ℕ), m ≥ max_sets * set_A_ratio ∧ 
              m ≤ max_sets ∧ 
              (price_A - cost_A) * m + (price_B - cost_B) * (max_sets - m) = 2455) ∧
  (∃ (x y : ℕ), x = xiaoming_set_A_ratio * (x + y) ∧
                90 * x + 100 * y + 110 * (3 * x - y) = xiaoming_total ∧
                3 * x - y = 5) := by
  sorry

end NUMINAMATH_CALUDE_hot_pot_restaurant_problem_l2574_257470


namespace NUMINAMATH_CALUDE_function_bound_l2574_257484

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin x - 1/2 * Real.cos (2*x) + a - 3/a + 1/2

theorem function_bound (a : ℝ) (ha : a ≠ 0) :
  (∀ x, f a x ≤ 0) → 0 < a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_function_bound_l2574_257484


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l2574_257431

/-- The minimum bailing rate problem -/
theorem minimum_bailing_rate
  (distance : ℝ)
  (water_entry_rate : ℝ)
  (max_water_capacity : ℝ)
  (rowing_speed : ℝ)
  (h1 : distance = 2)
  (h2 : water_entry_rate = 8)
  (h3 : max_water_capacity = 50)
  (h4 : rowing_speed = 2)
  : ∃ (bailing_rate : ℝ),
    bailing_rate = 8 ∧
    (∀ r : ℝ, r < 8 →
      (distance / rowing_speed) * (water_entry_rate - r) > max_water_capacity) ∧
    (distance / rowing_speed) * (water_entry_rate - bailing_rate) ≤ max_water_capacity :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l2574_257431


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2574_257495

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x + 2*y - 3 = 0
def l₂ (a x y : ℝ) : Prop := 2*x - a*y + 3 = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, l₁ x y → l₂ a x y → x = x

-- Theorem statement
theorem parallel_lines_a_value :
  ∀ a : ℝ, parallel a → a = -4 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2574_257495


namespace NUMINAMATH_CALUDE_west_movement_representation_l2574_257457

/-- Represents the direction of movement on an east-west road -/
inductive Direction
| East
| West

/-- Represents a movement on the road with a direction and distance -/
structure Movement where
  direction : Direction
  distance : ℝ

/-- Converts a movement to its numerical representation -/
def movementToNumber (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.distance
  | Direction.West => -m.distance

/-- The theorem stating that moving west by 7m should be denoted as -7m -/
theorem west_movement_representation :
  let eastMovement := Movement.mk Direction.East 3
  let westMovement := Movement.mk Direction.West 7
  movementToNumber eastMovement = 3 →
  movementToNumber westMovement = -7 :=
by sorry

end NUMINAMATH_CALUDE_west_movement_representation_l2574_257457


namespace NUMINAMATH_CALUDE_smallest_positive_translation_l2574_257411

theorem smallest_positive_translation (f : ℝ → ℝ) (φ : ℝ) : 
  (f = λ x => Real.sin (2 * x) + Real.cos (2 * x)) →
  (∀ x, f (x - φ) = f (φ - x)) →
  (∀ ψ, 0 < ψ ∧ ψ < φ → ¬(∀ x, f (x - ψ) = f (ψ - x))) →
  φ = 3 * Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_translation_l2574_257411


namespace NUMINAMATH_CALUDE_furniture_markup_l2574_257407

/-- Given a selling price and a cost price, calculate the percentage markup -/
def percentageMarkup (sellingPrice costPrice : ℕ) : ℚ :=
  ((sellingPrice - costPrice : ℚ) / costPrice) * 100

theorem furniture_markup :
  percentageMarkup 5750 5000 = 15 := by sorry

end NUMINAMATH_CALUDE_furniture_markup_l2574_257407


namespace NUMINAMATH_CALUDE_min_value_problem_l2574_257416

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + 2*y + 3*z = 1) :
  16/x^3 + 81/(8*y^3) + 1/(27*z^3) ≥ 1296 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ + 2*y₀ + 3*z₀ = 1 ∧
    16/x₀^3 + 81/(8*y₀^3) + 1/(27*z₀^3) = 1296 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2574_257416


namespace NUMINAMATH_CALUDE_complement_union_equality_l2574_257477

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {0, 1, 2}

-- Define set B
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_union_equality :
  (Set.compl A ∩ U) ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_equality_l2574_257477


namespace NUMINAMATH_CALUDE_total_timeout_time_total_timeout_is_185_minutes_l2574_257451

/-- Calculates the total time students spend in time-out given the number of time-outs for different offenses and the duration of each time-out. -/
theorem total_timeout_time (running_timeouts : ℕ) (timeout_duration : ℕ) : ℕ :=
  let food_timeouts := 5 * running_timeouts - 1
  let swearing_timeouts := food_timeouts / 3
  let total_timeouts := running_timeouts + food_timeouts + swearing_timeouts
  total_timeouts * timeout_duration

/-- Proves that the total time students spend in time-out is 185 minutes under the given conditions. -/
theorem total_timeout_is_185_minutes : total_timeout_time 5 5 = 185 := by
  sorry

end NUMINAMATH_CALUDE_total_timeout_time_total_timeout_is_185_minutes_l2574_257451


namespace NUMINAMATH_CALUDE_right_angles_in_two_days_l2574_257420

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hour_hand : ℕ)
  (minute_hand : ℕ)

/-- Represents the number of minutes in a day -/
def minutes_per_day : ℕ := 24 * 60

/-- Represents the number of right angles formed by clock hands in one day -/
def right_angles_per_day : ℕ := 44

/-- Checks if the clock hands form a right angle -/
def is_right_angle (c : Clock) : Prop :=
  (c.minute_hand - c.hour_hand) % 60 = 15 ∨ (c.hour_hand - c.minute_hand) % 60 = 15

/-- The main theorem: In 2 days, clock hands form a right angle 88 times -/
theorem right_angles_in_two_days :
  (2 * right_angles_per_day = 88) ∧
  (∀ t : ℕ, t < 2 * minutes_per_day →
    (∃ c : Clock, c.hour_hand = t % 720 ∧ c.minute_hand = t % 60 ∧
      is_right_angle c) ↔ t % (minutes_per_day / right_angles_per_day) = 0) :=
sorry

end NUMINAMATH_CALUDE_right_angles_in_two_days_l2574_257420


namespace NUMINAMATH_CALUDE_prob_at_most_one_white_ball_l2574_257434

/-- The number of black balls in the box -/
def black_balls : ℕ := 10

/-- The number of red balls in the box -/
def red_balls : ℕ := 12

/-- The number of white balls in the box -/
def white_balls : ℕ := 4

/-- The total number of balls in the box -/
def total_balls : ℕ := black_balls + red_balls + white_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 2

/-- X represents the number of white balls drawn -/
def X : Fin (drawn_balls + 1) → ℕ := sorry

/-- The probability of drawing at most one white ball -/
def P_X_le_1 : ℚ := sorry

/-- The main theorem to prove -/
theorem prob_at_most_one_white_ball :
  P_X_le_1 = (Nat.choose (total_balls - white_balls) 1 * Nat.choose white_balls 1 + 
              Nat.choose (total_balls - white_balls) 2) / 
             Nat.choose total_balls 2 :=
sorry

end NUMINAMATH_CALUDE_prob_at_most_one_white_ball_l2574_257434


namespace NUMINAMATH_CALUDE_solve_auction_problem_l2574_257453

def auction_problem (starting_price harry_initial_increase second_bidder_multiplier third_bidder_addition harry_final_increase : ℕ) : Prop :=
  let harry_first_bid := starting_price + harry_initial_increase
  let second_bid := harry_first_bid * second_bidder_multiplier
  let third_bid := second_bid + (harry_first_bid * third_bidder_addition)
  let harry_final_bid := third_bid + harry_final_increase
  harry_final_bid = 4000

theorem solve_auction_problem :
  auction_problem 300 200 2 3 1500 := by sorry

end NUMINAMATH_CALUDE_solve_auction_problem_l2574_257453


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l2574_257496

theorem logarithm_expression_equality : 
  (Real.log 8 / Real.log 5 * Real.log 2 / Real.log 5 + 25 ^ (Real.log 3 / Real.log 5)) / 
  (Real.log 4 + Real.log 25) + 5 * Real.log 2 / Real.log 3 - Real.log (32/9) / Real.log 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l2574_257496


namespace NUMINAMATH_CALUDE_product_equals_sum_solution_l2574_257450

theorem product_equals_sum_solution (x y : ℝ) (h1 : x * y = x + y) (h2 : y ≠ 1) :
  x = y / (y - 1) := by
  sorry

end NUMINAMATH_CALUDE_product_equals_sum_solution_l2574_257450


namespace NUMINAMATH_CALUDE_exchange_properties_l2574_257459

/-- Represents a box containing red and yellow balls -/
structure Box where
  red : ℕ
  yellow : ℕ

/-- Calculates the expected number of red balls after exchanging i balls -/
noncomputable def expected_red (box_a box_b : Box) (i : ℕ) : ℚ :=
  sorry

/-- Box A initially contains 3 red balls and 1 yellow ball -/
def initial_box_a : Box := ⟨3, 1⟩

/-- Box B initially contains 1 red ball and 3 yellow balls -/
def initial_box_b : Box := ⟨1, 3⟩

theorem exchange_properties :
  let E₁ := expected_red initial_box_a initial_box_b
  let E₂ := expected_red initial_box_b initial_box_a
  (E₁ 1 > E₂ 1) ∧
  (E₁ 2 = E₂ 2) ∧
  (E₁ 2 = 2) ∧
  (E₁ 1 + E₂ 1 = 4) ∧
  (E₁ 3 = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_exchange_properties_l2574_257459


namespace NUMINAMATH_CALUDE_light_reflection_l2574_257430

-- Define the incident light ray
def incident_ray (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the reflection line
def reflection_line (x y : ℝ) : Prop := y = x

-- Define the reflected light ray
def reflected_ray (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Theorem statement
theorem light_reflection 
  (x y : ℝ) 
  (h_incident : incident_ray x y) 
  (h_reflection : reflection_line x y) : 
  reflected_ray x y :=
sorry

end NUMINAMATH_CALUDE_light_reflection_l2574_257430


namespace NUMINAMATH_CALUDE_no_real_solutions_l2574_257482

theorem no_real_solutions : ∀ s : ℝ, s ≠ 2 → (s^2 - 5*s - 10) / (s - 2) ≠ 3*s + 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2574_257482


namespace NUMINAMATH_CALUDE_arc_measure_is_sixty_l2574_257487

/-- An equilateral triangle with a circle rolling along its side -/
structure TriangleWithCircle where
  /-- Side length of the equilateral triangle -/
  a : ℝ
  /-- Assumption that the side length is positive -/
  a_pos : 0 < a

/-- The angular measure of the arc intercepted on the circle -/
def arcMeasure (t : TriangleWithCircle) : ℝ := 60

/-- Theorem stating that the arc measure is always 60 degrees -/
theorem arc_measure_is_sixty (t : TriangleWithCircle) : arcMeasure t = 60 := by
  sorry

end NUMINAMATH_CALUDE_arc_measure_is_sixty_l2574_257487


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l2574_257497

/-- Given two circles on a 2D plane, prove that the x-coordinate of the point 
where a line tangent to both circles intersects the x-axis (to the right of the origin) 
is equal to 9/2. -/
theorem tangent_line_intersection 
  (r₁ r₂ c : ℝ) 
  (h₁ : r₁ = 3) 
  (h₂ : r₂ = 5) 
  (h₃ : c = 12) : 
  ∃ x : ℝ, x > 0 ∧ x = 9/2 ∧ 
  (∃ y : ℝ, (x - 0)^2 + y^2 = r₁^2 ∧ (x - c)^2 + y^2 = r₂^2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l2574_257497


namespace NUMINAMATH_CALUDE_equivalence_l2574_257417

variable {S : Type*} [Finite S]

def monotonic_decreasing (f : Set S → ℝ) : Prop :=
  ∀ X Y : Set S, X ⊆ Y → f Y ≤ f X

def property_1 (f : Set S → ℝ) : Prop :=
  ∀ X Y : Set S, f (X ∪ Y) + f (X ∩ Y) ≤ f X + f Y

def property_2 (f : Set S → ℝ) : Prop :=
  ∀ a : S, monotonic_decreasing (fun X => f (X ∪ {a}) - f X)

theorem equivalence (f : Set S → ℝ) : property_1 f ↔ property_2 f := by
  sorry

end NUMINAMATH_CALUDE_equivalence_l2574_257417


namespace NUMINAMATH_CALUDE_gwen_race_time_l2574_257474

/-- Represents the time Gwen spent jogging and walking during a race. -/
structure RaceTime where
  jogging : ℕ
  walking : ℕ

/-- Calculates if the given race time satisfies the required ratio and walking time. -/
def is_valid_race_time (rt : RaceTime) : Prop :=
  rt.jogging * 3 = rt.walking * 5 ∧ rt.walking = 9

/-- Theorem stating that the race time with 15 minutes of jogging and 9 minutes of walking
    satisfies the required conditions. -/
theorem gwen_race_time : ∃ (rt : RaceTime), is_valid_race_time rt ∧ rt.jogging = 15 := by
  sorry

end NUMINAMATH_CALUDE_gwen_race_time_l2574_257474


namespace NUMINAMATH_CALUDE_medium_mall_sample_l2574_257447

def stratified_sample (total_sample : ℕ) (ratio : List ℕ) : List ℕ :=
  let total_ratio := ratio.sum
  ratio.map (λ r => (total_sample * r) / total_ratio)

theorem medium_mall_sample :
  let ratio := [2, 4, 9]
  let sample := stratified_sample 45 ratio
  sample[1] = 12 := by sorry

end NUMINAMATH_CALUDE_medium_mall_sample_l2574_257447


namespace NUMINAMATH_CALUDE_triangle_properties_l2574_257479

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle --/
def TriangleABC (t : Triangle) : Prop :=
  t.c = 2 * Real.sqrt 3 ∧
  t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0

theorem triangle_properties (t : Triangle) (h : TriangleABC t) :
  t.C = π / 3 ∧ 
  (∃ (max_area : ℝ), max_area = 3 * Real.sqrt 3 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2574_257479


namespace NUMINAMATH_CALUDE_twins_age_problem_l2574_257456

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 13 → age = 6 := by sorry

end NUMINAMATH_CALUDE_twins_age_problem_l2574_257456


namespace NUMINAMATH_CALUDE_society_officer_selection_l2574_257436

theorem society_officer_selection (n : ℕ) (k : ℕ) : n = 12 ∧ k = 5 →
  (n.factorial / (n - k).factorial) = 95040 := by
  sorry

end NUMINAMATH_CALUDE_society_officer_selection_l2574_257436


namespace NUMINAMATH_CALUDE_CQRP_is_parallelogram_l2574_257458

-- Define the triangle ABC
structure Triangle (α β : ℝ) :=
  (A B C : ℂ)
  (angle_condition : α > 45 ∧ β > 45)

-- Define the construction of points R, P, and Q
def construct_R (t : Triangle α β) : ℂ :=
  t.B + (t.A - t.B) * Complex.I

def construct_P (t : Triangle α β) : ℂ :=
  t.C + (t.B - t.C) * (-Complex.I)

def construct_Q (t : Triangle α β) : ℂ :=
  t.C + (t.A - t.C) * Complex.I

-- State the theorem
theorem CQRP_is_parallelogram (α β : ℝ) (t : Triangle α β) :
  let R := construct_R t
  let P := construct_P t
  let Q := construct_Q t
  (R + P) / 2 = (t.C + Q) / 2 := by sorry

end NUMINAMATH_CALUDE_CQRP_is_parallelogram_l2574_257458


namespace NUMINAMATH_CALUDE_min_rooks_correct_min_rooks_minimal_l2574_257409

/-- A function that returns the minimum number of rooks needed on an n × n board
    to guarantee k non-attacking rooks can be selected. -/
def min_rooks (n k : ℕ) : ℕ :=
  n * (k - 1) + 1

/-- Theorem stating that min_rooks gives the correct minimum number of rooks. -/
theorem min_rooks_correct (n k : ℕ) (h1 : 1 < k) (h2 : k ≤ n) :
  ∀ (m : ℕ), m ≥ min_rooks n k →
    ∀ (placement : Fin m → Fin n × Fin n),
      ∃ (selected : Fin k → Fin m),
        ∀ (i j : Fin k), i ≠ j →
          (placement (selected i)).1 ≠ (placement (selected j)).1 ∧
          (placement (selected i)).2 ≠ (placement (selected j)).2 :=
by
  sorry

/-- Theorem stating that min_rooks gives the smallest such number. -/
theorem min_rooks_minimal (n k : ℕ) (h1 : 1 < k) (h2 : k ≤ n) :
  ∀ (m : ℕ), m < min_rooks n k →
    ∃ (placement : Fin m → Fin n × Fin n),
      ∀ (selected : Fin k → Fin m),
        ∃ (i j : Fin k), i ≠ j ∧
          ((placement (selected i)).1 = (placement (selected j)).1 ∨
           (placement (selected i)).2 = (placement (selected j)).2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_rooks_correct_min_rooks_minimal_l2574_257409


namespace NUMINAMATH_CALUDE_expected_value_Y_l2574_257455

/-- A random variable following a binomial distribution -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ

/-- The expected value of a binomial random variable -/
def expectedValue (X : BinomialRV n p) : ℝ := n * p

/-- Two random variables X and Y satisfying X + Y = 8, where X follows B(10, 0.6) -/
structure RandomVariables where
  X : BinomialRV 10 0.6
  Y : ℝ → ℝ
  sum_constraint : ∀ ω, X.X ω + Y ω = 8

/-- The theorem stating that E(Y) = 2 -/
theorem expected_value_Y (rv : RandomVariables) : 
  ∃ (E_Y : ℝ → ℝ), (∀ ω, E_Y ω = rv.Y ω) ∧ (∀ ω, E_Y ω = 2) :=
sorry

end NUMINAMATH_CALUDE_expected_value_Y_l2574_257455


namespace NUMINAMATH_CALUDE_second_number_value_second_number_proof_l2574_257412

theorem second_number_value : ℝ → Prop :=
  fun second_number =>
    let first_number : ℝ := 40
    (0.65 * first_number = 0.05 * second_number + 23) →
    second_number = 60

-- Proof
theorem second_number_proof : ∃ (x : ℝ), second_number_value x :=
  sorry

end NUMINAMATH_CALUDE_second_number_value_second_number_proof_l2574_257412


namespace NUMINAMATH_CALUDE_cylinder_base_area_at_different_heights_l2574_257489

/-- Represents the properties of a cylinder with constant volume -/
structure Cylinder where
  volume : ℝ
  height : ℝ
  base_area : ℝ
  height_positive : height > 0
  volume_eq : volume = height * base_area

/-- Theorem about the base area of a cylinder with constant volume -/
theorem cylinder_base_area_at_different_heights
  (c : Cylinder)
  (h_initial : c.height = 12)
  (s_initial : c.base_area = 2)
  (h_final : ℝ)
  (h_final_positive : h_final > 0)
  (h_final_value : h_final = 4.8) :
  let s_final := c.volume / h_final
  s_final = 5 := by sorry

end NUMINAMATH_CALUDE_cylinder_base_area_at_different_heights_l2574_257489


namespace NUMINAMATH_CALUDE_find_x_l2574_257410

-- Define the conditions
def condition1 (x : ℕ) : Prop := 3 * x > 0
def condition2 (x : ℕ) : Prop := x ≥ 10
def condition3 (x : ℕ) : Prop := x > 5

-- Theorem statement
theorem find_x : ∃ (x : ℕ), condition1 x ∧ condition2 x ∧ condition3 x ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2574_257410


namespace NUMINAMATH_CALUDE_power_of_power_l2574_257492

theorem power_of_power (a : ℝ) : (a ^ 3) ^ 4 = a ^ 12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2574_257492


namespace NUMINAMATH_CALUDE_garden_area_difference_l2574_257471

/-- Represents a rectangular garden with a pathway around it. -/
structure Garden where
  totalLength : ℕ
  totalWidth : ℕ
  pathwayWidth : ℕ

/-- Calculates the effective gardening area of a garden. -/
def effectiveArea (g : Garden) : ℕ :=
  (g.totalLength - 2 * g.pathwayWidth) * (g.totalWidth - 2 * g.pathwayWidth)

/-- Karl's garden dimensions -/
def karlGarden : Garden :=
  { totalLength := 30
  , totalWidth := 50
  , pathwayWidth := 2 }

/-- Makenna's garden dimensions -/
def makennaGarden : Garden :=
  { totalLength := 35
  , totalWidth := 55
  , pathwayWidth := 3 }

/-- Theorem stating the difference in effective gardening area -/
theorem garden_area_difference :
  effectiveArea makennaGarden - effectiveArea karlGarden = 225 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_difference_l2574_257471


namespace NUMINAMATH_CALUDE_quadratic_condition_l2574_257454

theorem quadratic_condition (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m^2 - 1) * x^2 + x + m = a * x^2 + b * x + c) ↔ 
  (m ≠ 1 ∧ m ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l2574_257454


namespace NUMINAMATH_CALUDE_min_occupied_seats_150_l2574_257426

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure the next person must sit next to someone -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 150 seats, the minimum number of occupied seats
    required to ensure the next person must sit next to someone is 90 -/
theorem min_occupied_seats_150 : min_occupied_seats 150 = 90 := by
  sorry

end NUMINAMATH_CALUDE_min_occupied_seats_150_l2574_257426


namespace NUMINAMATH_CALUDE_intersection_range_l2574_257483

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 + 1 ≥ Real.sqrt (2 * (p.1^2 + p.2^2))}
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1 - a| + |p.2 - 1| ≤ 1}

-- State the theorem
theorem intersection_range (a : ℝ) :
  (M ∩ N a).Nonempty ↔ a ∈ Set.Icc (1 - Real.sqrt 6) (3 + Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l2574_257483


namespace NUMINAMATH_CALUDE_quadratic_roots_when_k_negative_l2574_257480

theorem quadratic_roots_when_k_negative (k : ℝ) (h : k < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 + x₁ + k - 1 = 0) ∧ 
  (x₂^2 + x₂ + k - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_when_k_negative_l2574_257480


namespace NUMINAMATH_CALUDE_distance_traveled_l2574_257440

-- Define the average speed of car R
def speed_R : ℝ := 34.05124837953327

-- Define the time difference between car R and car P
def time_difference : ℝ := 2

-- Define the speed difference between car P and car R
def speed_difference : ℝ := 10

-- Theorem statement
theorem distance_traveled (t : ℝ) (h : t > time_difference) :
  speed_R * t = (speed_R + speed_difference) * (t - time_difference) →
  speed_R * t = 300 :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_l2574_257440


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2574_257476

/-- Given a line L1 with equation 2x - 6y - 8 = 0 and a point P (1, 3),
    prove that the line L2 with equation y + 3x - 6 = 0 passes through P
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 2 * x - 6 * y - 8 = 0
  let L2 : ℝ → ℝ → Prop := λ x y => y + 3 * x - 6 = 0
  let P : ℝ × ℝ := (1, 3)
  (L2 P.1 P.2) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L1 x2 y2 → x1 ≠ x2 →
    let m1 := (y2 - y1) / (x2 - x1)
    let m2 := -3
    m1 * m2 = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2574_257476


namespace NUMINAMATH_CALUDE_white_tshirts_per_pack_is_five_l2574_257462

/-- The number of white T-shirts in one pack -/
def white_tshirts_per_pack : ℕ := sorry

/-- The number of packs of white T-shirts bought -/
def white_packs : ℕ := 2

/-- The number of packs of blue T-shirts bought -/
def blue_packs : ℕ := 4

/-- The number of blue T-shirts in one pack -/
def blue_tshirts_per_pack : ℕ := 3

/-- The cost of one T-shirt in dollars -/
def cost_per_tshirt : ℕ := 3

/-- The total cost of all T-shirts in dollars -/
def total_cost : ℕ := 66

theorem white_tshirts_per_pack_is_five :
  white_tshirts_per_pack = 5 :=
by
  sorry

#check white_tshirts_per_pack_is_five

end NUMINAMATH_CALUDE_white_tshirts_per_pack_is_five_l2574_257462


namespace NUMINAMATH_CALUDE_petya_wins_l2574_257446

/-- Represents a position on the board -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the game state -/
structure GameState :=
  (board : Fin 101 → Fin 101 → Bool)
  (lastMoveLength : Nat)

/-- Represents a move in the game -/
inductive Move
  | Initial : Position → Move
  | Strip : Position → Nat → Move

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  match move with
  | Move.Initial _ => state.lastMoveLength = 0
  | Move.Strip _ n => n = state.lastMoveLength ∨ n = state.lastMoveLength + 1

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a player has a winning strategy from a given state -/
def hasWinningStrategy (state : GameState) (isFirstPlayer : Bool) : Prop :=
  sorry

/-- The main theorem stating that the first player (Petya) has a winning strategy -/
theorem petya_wins :
  ∃ (initialMove : Move),
    isValidMove { board := λ _ _ => false, lastMoveLength := 0 } initialMove ∧
    hasWinningStrategy (applyMove { board := λ _ _ => false, lastMoveLength := 0 } initialMove) true :=
  sorry

end NUMINAMATH_CALUDE_petya_wins_l2574_257446


namespace NUMINAMATH_CALUDE_nested_bracket_evaluation_l2574_257413

-- Define the operation [a, b, c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- Define the main theorem
theorem nested_bracket_evaluation :
  let x := bracket (2^4) (2^3) (2^5)
  let y := bracket (3^2) 3 (3^2 + 1)
  let z := bracket (5^2) 5 (5^2 + 1)
  bracket x y z = 169/100 := by sorry

end NUMINAMATH_CALUDE_nested_bracket_evaluation_l2574_257413


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2574_257424

/-- Given vectors a and b in ℝ², if a is perpendicular to b, then the magnitude of b is √5 -/
theorem perpendicular_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  b.1 = 2 → 
  a.1 * b.1 + a.2 * b.2 = 0 → 
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2574_257424


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_arithmetic_sequence_ratio_l2574_257460

-- Definition of a sequence
def Sequence (α : Type) := ℕ → α

-- Definition of a geometric sequence with common ratio 2
def IsGeometricSequenceWithRatio2 (a : Sequence ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n

-- Definition of the condition a_n = 2a_{n-1} for n ≥ 2
def SatisfiesCondition (a : Sequence ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

-- Definition of an arithmetic sequence
def IsArithmeticSequence (a : Sequence ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d

-- Definition of the sum of first n terms of a sequence
def SumOfFirstNTerms (a : Sequence ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => SumOfFirstNTerms a n + a (n + 1)

-- Theorem 1
theorem necessary_but_not_sufficient :
  (∀ a : Sequence ℝ, IsGeometricSequenceWithRatio2 a → SatisfiesCondition a) ∧
  (∃ a : Sequence ℝ, SatisfiesCondition a ∧ ¬IsGeometricSequenceWithRatio2 a) := by sorry

-- Theorem 2
theorem arithmetic_sequence_ratio :
  ∀ a b : Sequence ℝ,
    IsArithmeticSequence a →
    IsArithmeticSequence b →
    (SumOfFirstNTerms a 5) / (SumOfFirstNTerms b 7) = 15 / 13 →
    a 3 / b 4 = 21 / 13 := by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_arithmetic_sequence_ratio_l2574_257460


namespace NUMINAMATH_CALUDE_minimum_value_of_f_l2574_257475

/-- The function f(x) = -x³ + ax² - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem minimum_value_of_f (a : ℝ) :
  (f_derivative a 2 = 0) →
  (∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → f a m ≥ -4) ∧
  (∃ m : ℝ, -1 ≤ m ∧ m ≤ 1 ∧ f a m = -4) :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_l2574_257475


namespace NUMINAMATH_CALUDE_equation_solution_l2574_257418

theorem equation_solution : ∃! x : ℚ, (4 * x - 12) / 3 = (3 * x + 6) / 5 ∧ x = 78 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2574_257418


namespace NUMINAMATH_CALUDE_divisible_by_65_l2574_257428

theorem divisible_by_65 (n : ℕ) : ∃ k : ℤ, 5^n * (2^(2*n) - 3^n) + 2^n - 7^n = 65 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_65_l2574_257428


namespace NUMINAMATH_CALUDE_cylinder_radius_and_volume_l2574_257485

/-- Properties of a cylinder with given height and surface area -/
def Cylinder (h : ℝ) (s : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ h > 0 ∧ s = 2 * Real.pi * r * h + 2 * Real.pi * r^2

theorem cylinder_radius_and_volume 
  (h : ℝ) (s : ℝ) 
  (hh : h = 8) (hs : s = 130 * Real.pi) : 
  ∃ (r v : ℝ), Cylinder h s ∧ r = 5 ∧ v = 200 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_and_volume_l2574_257485


namespace NUMINAMATH_CALUDE_distinct_colorings_count_l2574_257468

/-- The number of distinct colorings of n points on a circle with k blue points
    and at least p red points between each pair of consecutive blue points. -/
def distinctColorings (n k p : ℕ) : ℚ :=
  if 2 ≤ k ∧ k ≤ n / (p + 1) then
    (1 : ℚ) / k * (Nat.choose (n - k * p - 1) (k - 1) : ℚ)
  else
    0

theorem distinct_colorings_count
  (n k p : ℕ)
  (h1 : 0 < n ∧ 0 < k ∧ 0 < p)
  (h2 : 2 ≤ k)
  (h3 : k ≤ n / (p + 1)) :
  distinctColorings n k p = (1 : ℚ) / k * (Nat.choose (n - k * p - 1) (k - 1) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_distinct_colorings_count_l2574_257468


namespace NUMINAMATH_CALUDE_largest_angle_in_tangent_circles_triangle_l2574_257465

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent --/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is tangent to the x-axis --/
def is_tangent_to_x_axis (c : Circle) : Prop :=
  let (_, y) := c.center
  y = c.radius

/-- Theorem about the largest angle in a triangle formed by centers of three mutually tangent circles --/
theorem largest_angle_in_tangent_circles_triangle (A B C : Circle) :
  are_externally_tangent A B ∧ 
  are_externally_tangent B C ∧ 
  are_externally_tangent C A ∧
  is_tangent_to_x_axis A ∧
  is_tangent_to_x_axis B ∧
  is_tangent_to_x_axis C →
  ∃ γ : ℝ, π/2 < γ ∧ γ ≤ 2 * Real.arcsin (4/5) ∧ 
  γ = max (Real.arccos ((A.center.1 - C.center.1)^2 + (A.center.2 - C.center.2)^2 - A.radius^2 - C.radius^2) / (2 * A.radius * C.radius))
          (max (Real.arccos ((B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 - B.radius^2 - C.radius^2) / (2 * B.radius * C.radius))
               (Real.arccos ((A.center.1 - B.center.1)^2 + (A.center.2 - B.center.2)^2 - A.radius^2 - B.radius^2) / (2 * A.radius * B.radius))) :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_tangent_circles_triangle_l2574_257465


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l2574_257473

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion (x y z : ℝ) (r θ : ℝ) 
  (h1 : x = 6)
  (h2 : y = 6)
  (h3 : z = -10)
  (h4 : r > 0)
  (h5 : 0 ≤ θ ∧ θ < 2 * π)
  (h6 : x = r * Real.cos θ)
  (h7 : y = r * Real.sin θ) :
  r = 6 * Real.sqrt 2 ∧ θ = π / 4 ∧ z = -10 := by
  sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l2574_257473


namespace NUMINAMATH_CALUDE_complex_power_eight_l2574_257498

theorem complex_power_eight (z : ℂ) : 
  z = (1 - Complex.I * Real.sqrt 3) / 2 → 
  z^8 = -(1 + Complex.I * Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_power_eight_l2574_257498


namespace NUMINAMATH_CALUDE_inequality_proof_l2574_257466

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 1) :
  (a + 2*b + 2/(a + 1)) * (b + 2*a + 2/(b + 1)) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2574_257466


namespace NUMINAMATH_CALUDE_burj_khalifa_sears_difference_l2574_257490

/-- The height difference between two buildings -/
def height_difference (h1 h2 : ℕ) : ℕ := h1 - h2

/-- Burj Khalifa's height in meters -/
def burj_khalifa_height : ℕ := 830

/-- Sears Tower's height in meters -/
def sears_tower_height : ℕ := 527

/-- Theorem stating the height difference between Burj Khalifa and Sears Tower -/
theorem burj_khalifa_sears_difference :
  height_difference burj_khalifa_height sears_tower_height = 303 := by
  sorry

end NUMINAMATH_CALUDE_burj_khalifa_sears_difference_l2574_257490


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l2574_257464

theorem fraction_sum_theorem (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = 3)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = -4) :
  b / (a + b) + c / (b + c) + a / (c + a) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l2574_257464


namespace NUMINAMATH_CALUDE_recurring_decimal_one_zero_six_l2574_257488

-- Define the recurring decimal notation
def recurring_decimal (whole : ℕ) (recurring : ℕ) : ℚ :=
  whole + (recurring : ℚ) / 99

-- State the theorem
theorem recurring_decimal_one_zero_six :
  recurring_decimal 1 6 = 35 / 33 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_one_zero_six_l2574_257488


namespace NUMINAMATH_CALUDE_triangle_sequence_properties_l2574_257432

/-- Isosceles triangle with perimeter 2s -/
structure IsoscelesTriangle (s : ℝ) :=
  (base : ℝ)
  (leg : ℝ)
  (perimeter_eq : base + 2 * leg = 2 * s)
  (isosceles : leg ≥ base / 2)

/-- Sequence of isosceles triangles -/
def triangle_sequence (s : ℝ) : ℕ → IsoscelesTriangle s
| 0 => ⟨2, 49, sorry, sorry⟩
| (n + 1) => ⟨(triangle_sequence s n).leg, sorry, sorry, sorry⟩

/-- Angle between the legs of triangle i -/
def angle (s : ℝ) (i : ℕ) : ℝ := sorry

theorem triangle_sequence_properties (s : ℝ) :
  (∀ j : ℕ, angle s (2 * j) < angle s (2 * (j + 1))) ∧
  (∀ j : ℕ, angle s (2 * j + 1) > angle s (2 * (j + 1) + 1)) ∧
  (abs (angle s 11 - Real.pi / 3) < Real.pi / 180) :=
sorry

end NUMINAMATH_CALUDE_triangle_sequence_properties_l2574_257432


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_17_squared_plus_144_squared_l2574_257486

theorem largest_prime_divisor_of_17_squared_plus_144_squared :
  (Nat.factors (17^2 + 144^2)).maximum? = some 29 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_17_squared_plus_144_squared_l2574_257486


namespace NUMINAMATH_CALUDE_parallel_tangents_f_1_equals_1_l2574_257478

def f (a b x : ℝ) : ℝ := x^3 + a*x + b

theorem parallel_tangents_f_1_equals_1 (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 3*a^2 + a = 3*b^2 + a) : f a b 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_f_1_equals_1_l2574_257478


namespace NUMINAMATH_CALUDE_budget_supplies_percent_l2574_257422

theorem budget_supplies_percent (salaries research_dev utilities equipment transportation : ℝ)
  (h1 : salaries = 60)
  (h2 : research_dev = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : transportation = 72 * 100 / 360)
  (h6 : salaries + research_dev + utilities + equipment + transportation < 100) :
  100 - (salaries + research_dev + utilities + equipment + transportation) = 2 := by
  sorry

end NUMINAMATH_CALUDE_budget_supplies_percent_l2574_257422


namespace NUMINAMATH_CALUDE_expression_satisfies_conditions_l2574_257421

def original_expression : ℕ := 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1

def transformed_expression : ℕ := 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1 * 1

theorem expression_satisfies_conditions :
  (original_expression = 11) ∧
  (transformed_expression = 11) :=
by
  sorry

#eval original_expression
#eval transformed_expression

end NUMINAMATH_CALUDE_expression_satisfies_conditions_l2574_257421


namespace NUMINAMATH_CALUDE_relationship_between_A_B_C_l2574_257472

-- Define propositions A, B, and C
variable (A B C : Prop)

-- Define the relationships between A, B, and C
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

def necessary_and_sufficient (P Q : Prop) : Prop :=
  (P ↔ Q)

-- Theorem statement
theorem relationship_between_A_B_C
  (h1 : sufficient_not_necessary A B)
  (h2 : necessary_and_sufficient B C) :
  sufficient_not_necessary C A :=
sorry

end NUMINAMATH_CALUDE_relationship_between_A_B_C_l2574_257472


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2574_257444

theorem complex_equation_sum (x y : ℝ) :
  (x / (1 - Complex.I)) + (y / (1 - 2 * Complex.I)) = 5 / (1 - 3 * Complex.I) →
  x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2574_257444


namespace NUMINAMATH_CALUDE_min_teams_non_negative_balance_l2574_257404

/-- Represents the number of wins in a series --/
inductive SeriesScore
| Four_Zero
| Four_One
| Four_Two
| Four_Three

/-- Represents a team's performance in the tournament --/
structure TeamPerformance where
  wins : ℕ
  losses : ℕ

/-- Represents the NHL playoff tournament --/
structure NHLPlayoffs where
  num_teams : ℕ
  num_rounds : ℕ
  series_scores : List SeriesScore

/-- Defines a non-negative balance of wins --/
def has_non_negative_balance (team : TeamPerformance) : Prop :=
  team.wins ≥ team.losses

/-- Theorem stating the minimum number of teams with non-negative balance --/
theorem min_teams_non_negative_balance (playoffs : NHLPlayoffs) 
  (h1 : playoffs.num_teams = 16)
  (h2 : playoffs.num_rounds = 4)
  (h3 : ∀ s ∈ playoffs.series_scores, s ∈ [SeriesScore.Four_Zero, SeriesScore.Four_One, SeriesScore.Four_Two, SeriesScore.Four_Three]) :
  ∃ (teams : List TeamPerformance), 
    (∀ team ∈ teams, has_non_negative_balance team) ∧ 
    (teams.length = 2) ∧
    (∀ (n : ℕ), n < 2 → ¬∃ (teams' : List TeamPerformance), 
      (∀ team ∈ teams', has_non_negative_balance team) ∧ 
      (teams'.length = n)) :=
by sorry

end NUMINAMATH_CALUDE_min_teams_non_negative_balance_l2574_257404


namespace NUMINAMATH_CALUDE_age_problem_solution_l2574_257406

/-- The ages of the king and queen satisfy the given conditions -/
def age_problem (king_age queen_age : ℕ) : Prop :=
  ∃ (t : ℕ),
    -- The king's current age is twice the queen's age when the king was as old as the queen is now
    king_age = 2 * (queen_age - t) ∧
    -- When the queen is as old as the king is now, their combined ages will be 63 years
    king_age + (king_age + t) = 63 ∧
    -- The age difference
    king_age - queen_age = t

/-- The solution to the age problem -/
theorem age_problem_solution :
  ∃ (king_age queen_age : ℕ), age_problem king_age queen_age ∧ king_age = 28 ∧ queen_age = 21 :=
sorry

end NUMINAMATH_CALUDE_age_problem_solution_l2574_257406


namespace NUMINAMATH_CALUDE_zebra_fox_ratio_l2574_257491

/-- Proves that the ratio of zebras to foxes is 3:1 given the conditions of the problem --/
theorem zebra_fox_ratio :
  let total_animals : ℕ := 100
  let num_cows : ℕ := 20
  let num_foxes : ℕ := 15
  let num_sheep : ℕ := 20
  let num_zebras : ℕ := total_animals - (num_cows + num_foxes + num_sheep)
  (num_zebras : ℚ) / num_foxes = 3 := by sorry

end NUMINAMATH_CALUDE_zebra_fox_ratio_l2574_257491
