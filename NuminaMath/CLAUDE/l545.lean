import Mathlib

namespace NUMINAMATH_CALUDE_simplify_radical_sum_l545_54518

theorem simplify_radical_sum : Real.sqrt 72 + Real.sqrt 32 + Real.sqrt 50 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l545_54518


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l545_54587

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 90)
  (h2 : average_speed = 66) : 
  (2 * average_speed - speed_first_hour) = 42 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l545_54587


namespace NUMINAMATH_CALUDE_scarf_cost_is_two_l545_54595

/-- The cost of a single scarf when Kiki buys scarves and hats under specific conditions. -/
def scarf_cost (total_money : ℚ) (num_scarves : ℕ) (hat_percentage : ℚ) : ℚ :=
  let scarf_percentage := 1 - hat_percentage
  let scarf_total := total_money * scarf_percentage
  scarf_total / num_scarves

/-- Theorem stating that under the given conditions, each scarf costs $2. -/
theorem scarf_cost_is_two :
  scarf_cost 90 18 (60 / 100) = 2 := by
  sorry

#eval scarf_cost 90 18 (60 / 100)

end NUMINAMATH_CALUDE_scarf_cost_is_two_l545_54595


namespace NUMINAMATH_CALUDE_french_toast_weekends_l545_54528

/-- Represents the number of slices used per weekend -/
def slices_per_weekend : ℚ := 3

/-- Represents the number of slices in a loaf of bread -/
def slices_per_loaf : ℕ := 12

/-- Represents the number of loaves of bread used -/
def loaves_used : ℕ := 26

/-- Theorem stating that 26 loaves of bread cover 104 weekends of french toast making -/
theorem french_toast_weekends : 
  (loaves_used : ℚ) * (slices_per_loaf : ℚ) / slices_per_weekend = 104 := by
  sorry

end NUMINAMATH_CALUDE_french_toast_weekends_l545_54528


namespace NUMINAMATH_CALUDE_draw_one_is_random_event_l545_54532

/-- A set of cards numbered from 1 to 10 -/
def CardSet : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}

/-- Definition of a random event -/
def IsRandomEvent (event : Set ℕ → Prop) : Prop :=
  ∃ (s : Set ℕ), event s ∧ ∃ (t : Set ℕ), ¬event t

/-- Drawing a card numbered 1 from the set -/
def DrawOne (s : Set ℕ) : Prop := 1 ∈ s

/-- Theorem: Drawing a card numbered 1 from a set of cards numbered 1 to 10 is a random event -/
theorem draw_one_is_random_event : IsRandomEvent DrawOne :=
sorry

end NUMINAMATH_CALUDE_draw_one_is_random_event_l545_54532


namespace NUMINAMATH_CALUDE_smallest_common_multiple_9_6_l545_54576

theorem smallest_common_multiple_9_6 : ∀ n : ℕ, n > 0 ∧ 9 ∣ n ∧ 6 ∣ n → n ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_9_6_l545_54576


namespace NUMINAMATH_CALUDE_find_z_when_y_is_4_l545_54562

-- Define the relationship between y and z
def inverse_variation (y z : ℝ) (k : ℝ) : Prop :=
  y^3 * z^(1/3) = k

-- Theorem statement
theorem find_z_when_y_is_4 (y z : ℝ) (k : ℝ) :
  inverse_variation 2 1 k →
  inverse_variation 4 z k →
  z = 1 / 512 :=
by
  sorry

end NUMINAMATH_CALUDE_find_z_when_y_is_4_l545_54562


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l545_54564

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_property
  (a : ℕ → ℝ) (q : ℝ)
  (h_geo : geometric_sequence a q)
  (h_neg : a 1 * a 2 < 0) :
  a 1 * a 5 > 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l545_54564


namespace NUMINAMATH_CALUDE_exam_mean_is_115_l545_54578

/-- The mean score of the exam -/
def mean : ℝ := 115

/-- The standard deviation of the exam scores -/
def std_dev : ℝ := 40

/-- Theorem stating that the given conditions imply the mean score is 115 -/
theorem exam_mean_is_115 :
  (55 = mean - 1.5 * std_dev) ∧
  (75 = mean - 2 * std_dev) ∧
  (85 = mean + 1.5 * std_dev) ∧
  (100 = mean + 3.5 * std_dev) →
  mean = 115 := by sorry

end NUMINAMATH_CALUDE_exam_mean_is_115_l545_54578


namespace NUMINAMATH_CALUDE_inequality_one_l545_54519

theorem inequality_one (x : ℝ) : 
  (x + 2) / (x - 4) ≤ 0 ↔ -2 ≤ x ∧ x < 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_one_l545_54519


namespace NUMINAMATH_CALUDE_at_most_one_triangle_l545_54563

/-- Represents a city in Euleria -/
def City : Type := Fin 101

/-- Represents an airline in Euleria -/
def Airline : Type := Fin 99

/-- Represents a flight between two cities operated by an airline -/
def Flight : Type := City × City × Airline

/-- The set of all flights in Euleria -/
def AllFlights : Set Flight := sorry

/-- A function that returns the airline operating a flight between two cities -/
def flightOperator : City → City → Airline := sorry

/-- A predicate that checks if three cities form a triangle -/
def isTriangle (a b c : City) : Prop :=
  flightOperator a b = flightOperator b c ∧ flightOperator b c = flightOperator c a

/-- The main theorem stating that there is at most one triangle in Euleria -/
theorem at_most_one_triangle :
  ∀ a b c d e f : City,
    isTriangle a b c → isTriangle d e f → a = d ∧ b = e ∧ c = f := by sorry

end NUMINAMATH_CALUDE_at_most_one_triangle_l545_54563


namespace NUMINAMATH_CALUDE_power_of_negative_square_l545_54542

theorem power_of_negative_square (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_square_l545_54542


namespace NUMINAMATH_CALUDE_function_monotonicity_implies_c_zero_and_b_positive_l545_54551

-- Define the function f(x)
def f (b c x : ℝ) : ℝ := -x^3 - b*x^2 - 5*c*x

-- State the theorem
theorem function_monotonicity_implies_c_zero_and_b_positive
  (b c : ℝ)
  (h1 : ∀ x ≤ 0, Monotone (fun x => f b c x))
  (h2 : ∀ x ∈ Set.Icc 0 6, StrictMono (fun x => f b c x)) :
  c = 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_monotonicity_implies_c_zero_and_b_positive_l545_54551


namespace NUMINAMATH_CALUDE_costume_material_cost_l545_54557

/-- Calculates the total cost of material for Jenna's costume --/
theorem costume_material_cost : 
  let skirt_length : ℕ := 12
  let skirt_width : ℕ := 4
  let num_skirts : ℕ := 3
  let bodice_area : ℕ := 2
  let sleeve_area : ℕ := 5
  let num_sleeves : ℕ := 2
  let cost_per_sqft : ℕ := 3
  
  skirt_length * skirt_width * num_skirts + 
  bodice_area + 
  sleeve_area * num_sleeves * cost_per_sqft = 468 := by
  sorry

end NUMINAMATH_CALUDE_costume_material_cost_l545_54557


namespace NUMINAMATH_CALUDE_canteen_theorem_l545_54567

/-- Represents the number of dishes available --/
def num_dishes : ℕ := 6

/-- Calculates the maximum number of days based on the number of dishes --/
def max_days (n : ℕ) : ℕ := 2^n

/-- Calculates the average number of dishes per day --/
def avg_dishes_per_day (n : ℕ) : ℚ := n / 2

theorem canteen_theorem :
  max_days num_dishes = 64 ∧ avg_dishes_per_day num_dishes = 3 := by sorry

end NUMINAMATH_CALUDE_canteen_theorem_l545_54567


namespace NUMINAMATH_CALUDE_max_integer_difference_l545_54581

theorem max_integer_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) : 
  (∀ a b : ℤ, 7 < a ∧ a < 9 ∧ 9 < b ∧ b < 15 → y - x ≥ b - a) ∧ y - x = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_integer_difference_l545_54581


namespace NUMINAMATH_CALUDE_intersection_equals_N_l545_54526

-- Define the sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem intersection_equals_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_intersection_equals_N_l545_54526


namespace NUMINAMATH_CALUDE_ellipse_distance_to_y_axis_l545_54571

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the foci
def foci (f : ℝ) : Prop := f^2 = 3

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop := ellipse x y

-- Define the perpendicularity condition
def perpendicular_vectors (x y f : ℝ) : Prop :=
  (x + f) * (x - f) + y * y = 0

-- Theorem statement
theorem ellipse_distance_to_y_axis 
  (x y f : ℝ) 
  (h1 : ellipse x y) 
  (h2 : foci f) 
  (h3 : perpendicular_vectors x y f) : 
  x^2 = 8/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_distance_to_y_axis_l545_54571


namespace NUMINAMATH_CALUDE_employees_in_all_three_proof_l545_54586

/-- The number of employees trained to work in all 3 restaurants -/
def employees_in_all_three : ℕ := 2

theorem employees_in_all_three_proof :
  let total_employees : ℕ := 39
  let min_restaurants : ℕ := 1
  let max_restaurants : ℕ := 3
  let family_buffet : ℕ := 15
  let dining_room : ℕ := 18
  let snack_bar : ℕ := 12
  let in_two_restaurants : ℕ := 4
  employees_in_all_three = 
    total_employees + employees_in_all_three - in_two_restaurants - 
    (family_buffet + dining_room + snack_bar) := by
  sorry

#check employees_in_all_three_proof

end NUMINAMATH_CALUDE_employees_in_all_three_proof_l545_54586


namespace NUMINAMATH_CALUDE_nancy_football_tickets_l545_54555

/-- The total amount Nancy spends on football tickets for three months -/
def total_spent (this_month_games : ℕ) (this_month_price : ℕ) 
                (last_month_games : ℕ) (last_month_price : ℕ) 
                (next_month_games : ℕ) (next_month_price : ℕ) : ℕ :=
  this_month_games * this_month_price + 
  last_month_games * last_month_price + 
  next_month_games * next_month_price

theorem nancy_football_tickets : 
  total_spent 9 5 8 4 7 6 = 119 := by
  sorry

end NUMINAMATH_CALUDE_nancy_football_tickets_l545_54555


namespace NUMINAMATH_CALUDE_cricketer_matches_count_l545_54534

/-- Proves that a cricketer played 10 matches given the average scores for all matches, 
    the first 6 matches, and the last 4 matches. -/
theorem cricketer_matches_count 
  (total_average : ℝ) 
  (first_six_average : ℝ) 
  (last_four_average : ℝ) 
  (h1 : total_average = 38.9)
  (h2 : first_six_average = 42)
  (h3 : last_four_average = 34.25) : 
  ∃ (n : ℕ), n = 10 ∧ 
    n * total_average = 6 * first_six_average + 4 * last_four_average := by
  sorry

#check cricketer_matches_count

end NUMINAMATH_CALUDE_cricketer_matches_count_l545_54534


namespace NUMINAMATH_CALUDE_circle_radius_increment_l545_54510

theorem circle_radius_increment (c₁ c₂ : ℝ) (h₁ : c₁ = 50) (h₂ : c₂ = 60) :
  c₂ / (2 * Real.pi) - c₁ / (2 * Real.pi) = 5 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increment_l545_54510


namespace NUMINAMATH_CALUDE_problem_statement_l545_54569

theorem problem_statement :
  (∀ x : ℝ, (2/3 ≤ x ∧ x ≤ 2) → (-1 ≤ x ∧ x ≤ 2)) ∧
  (∃ x : ℝ, (-1 ≤ x ∧ x ≤ 2) ∧ ¬(2/3 ≤ x ∧ x ≤ 2)) ∧
  (∀ a : ℝ, (∀ x : ℝ, (x ≤ a ∨ x ≥ a + 1) → (2/3 ≤ x ∧ x ≤ 2)) ↔ (a ≥ 2 ∨ a ≤ -1/3)) :=
by sorry


end NUMINAMATH_CALUDE_problem_statement_l545_54569


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l545_54507

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def yIntercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

theorem parallel_line_y_intercept :
  ∀ (b : Line),
    b.slope = 3 →                -- b is parallel to y = 3x - 6
    b.point = (3, 4) →           -- b passes through (3, 4)
    yIntercept b = -5            -- the y-intercept of b is -5
  := by sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l545_54507


namespace NUMINAMATH_CALUDE_parabola_transformation_l545_54505

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = -2x^2 + 1 -/
def original_parabola : Parabola := ⟨-2, 0, 1⟩

/-- Moves a parabola horizontally by h units -/
def move_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  ⟨p.a, -2 * p.a * h + p.b, p.a * h^2 - p.b * h + p.c⟩

/-- Moves a parabola vertically by k units -/
def move_vertical (p : Parabola) (k : ℝ) : Parabola :=
  ⟨p.a, p.b, p.c + k⟩

/-- The final parabola after moving right by 1 and up by 1 -/
def final_parabola : Parabola :=
  move_vertical (move_horizontal original_parabola 1) 1

theorem parabola_transformation :
  final_parabola = ⟨-2, 4, 2⟩ := by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l545_54505


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l545_54538

theorem trader_gain_percentage (cost : ℝ) (h : cost > 0) :
  let gain := 30 * cost
  let cost_price := 100 * cost
  let gain_percentage := (gain / cost_price) * 100
  gain_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l545_54538


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_min_value_sum_fractions_achieved_l545_54549

theorem min_value_sum_fractions (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c) / d + (a + b + d) / c + (a + c + d) / b + (b + c + d) / a ≥ 12 :=
by sorry

theorem min_value_sum_fractions_achieved (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ∃ (x : ℝ), x > 0 ∧ 
    (x + x + x) / x + (x + x + x) / x + (x + x + x) / x + (x + x + x) / x = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_min_value_sum_fractions_achieved_l545_54549


namespace NUMINAMATH_CALUDE_cube_root_problem_l545_54539

theorem cube_root_problem (x : ℝ) : (x + 6) ^ (1/3 : ℝ) = 3 → (x + 6) ^ 3 = 19683 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l545_54539


namespace NUMINAMATH_CALUDE_linear_function_proof_l545_54503

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem linear_function_proof (f : ℝ → ℝ) 
  (h1 : is_linear f) 
  (h2 : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) : 
  ∀ x, f x = 2 * x + 7 := by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l545_54503


namespace NUMINAMATH_CALUDE_investment_interest_rate_l545_54521

/-- Calculates the total interest rate for a two-share investment --/
def total_interest_rate (total_investment : ℚ) (rate1 : ℚ) (rate2 : ℚ) (amount2 : ℚ) : ℚ :=
  let amount1 := total_investment - amount2
  let interest1 := amount1 * rate1
  let interest2 := amount2 * rate2
  let total_interest := interest1 + interest2
  (total_interest / total_investment) * 100

/-- Theorem stating the total interest rate for the given investment scenario --/
theorem investment_interest_rate :
  total_interest_rate 10000 (9/100) (11/100) 3750 = (975/10000) * 100 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l545_54521


namespace NUMINAMATH_CALUDE_family_ages_solution_l545_54588

/-- Represents the ages of a family at a given time --/
structure FamilyAges where
  man : ℕ
  father : ℕ
  sister : ℕ

/-- Checks if the given ages satisfy the problem conditions --/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.man = (2 * ages.father) / 5 ∧
  ages.man + 10 = (ages.father + 10) / 2 ∧
  ages.sister + 10 = (3 * (ages.father + 10)) / 4

/-- The theorem stating the solution to the problem --/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ 
    ages.man = 20 ∧ ages.father = 50 ∧ ages.sister = 35 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l545_54588


namespace NUMINAMATH_CALUDE_fraction_order_l545_54545

theorem fraction_order (a b m n : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : m > 0) (h4 : n > 0) :
  b / a < (b + m) / (a + m) ∧ 
  (b + m) / (a + m) < (a + n) / (b + n) ∧ 
  (a + n) / (b + n) < a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l545_54545


namespace NUMINAMATH_CALUDE_naturalNumberDecimal_irrational_l545_54589

/-- Represents the infinite decimal 0.1234567891011121314... -/
def naturalNumberDecimal : ℝ :=
  sorry

/-- The digits of naturalNumberDecimal after the decimal point consist of all natural numbers in order -/
axiom naturalNumberDecimal_property : ∀ n : ℕ, ∃ k : ℕ, sorry

theorem naturalNumberDecimal_irrational : Irrational naturalNumberDecimal := by
  sorry

end NUMINAMATH_CALUDE_naturalNumberDecimal_irrational_l545_54589


namespace NUMINAMATH_CALUDE_boy_and_bus_speeds_l545_54504

/-- Represents the problem of finding the speeds of a boy and a bus given certain conditions. -/
theorem boy_and_bus_speeds
  (total_distance : ℝ)
  (first_meeting_time : ℝ)
  (boy_additional_distance : ℝ)
  (stop_time : ℝ) :
  total_distance = 4.5 ∧
  first_meeting_time = 0.25 ∧
  boy_additional_distance = 9 / 28 ∧
  stop_time = 4 / 60 →
  ∃ (boy_speed bus_speed : ℝ),
    boy_speed = 3 ∧
    bus_speed = 45 ∧
    boy_speed > 0 ∧
    bus_speed > 0 ∧
    boy_speed * first_meeting_time + boy_additional_distance =
      bus_speed * first_meeting_time - total_distance ∧
    bus_speed * (first_meeting_time + 2 * stop_time) = 2 * total_distance :=
by sorry

end NUMINAMATH_CALUDE_boy_and_bus_speeds_l545_54504


namespace NUMINAMATH_CALUDE_standard_deviation_double_data_l545_54565

def data1 : List ℝ := [2, 3, 4, 5]
def data2 : List ℝ := [4, 6, 8, 10]

def standard_deviation (data : List ℝ) : ℝ := sorry

theorem standard_deviation_double_data :
  standard_deviation data1 = (1 / 2) * standard_deviation data2 := by sorry

end NUMINAMATH_CALUDE_standard_deviation_double_data_l545_54565


namespace NUMINAMATH_CALUDE_sum_modulo_thirteen_l545_54592

theorem sum_modulo_thirteen : (9245 + 9246 + 9247 + 9248 + 9249 + 9250) % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_modulo_thirteen_l545_54592


namespace NUMINAMATH_CALUDE_linear_equation_equivalence_l545_54502

theorem linear_equation_equivalence (x y : ℝ) :
  (3 * x - 2 * y = 6) ↔ (y = (3 / 2) * x - 3) := by sorry

end NUMINAMATH_CALUDE_linear_equation_equivalence_l545_54502


namespace NUMINAMATH_CALUDE_power_of_negative_square_l545_54599

theorem power_of_negative_square (x : ℝ) : (-2 * x^2)^3 = -8 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_square_l545_54599


namespace NUMINAMATH_CALUDE_tray_height_l545_54574

/-- The height of a tray formed from a square paper with specific cuts -/
theorem tray_height (side_length : ℝ) (cut_start : ℝ) (cut_angle : ℝ) : 
  side_length = 50 →
  cut_start = Real.sqrt 5 →
  cut_angle = π / 4 →
  (Real.sqrt 10) / 2 = 
    cut_start * Real.sin (cut_angle / 2) := by
  sorry

end NUMINAMATH_CALUDE_tray_height_l545_54574


namespace NUMINAMATH_CALUDE_regions_in_circle_l545_54554

/-- The number of regions created by radii and concentric circles within a larger circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem: 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (r : ℕ) (c : ℕ) 
    (h1 : r = 16) (h2 : c = 10) : 
    num_regions r c = 176 := by
  sorry

#eval num_regions 16 10

end NUMINAMATH_CALUDE_regions_in_circle_l545_54554


namespace NUMINAMATH_CALUDE_problem_1_l545_54524

def oplus (a b : ℚ) : ℚ := (a * b) / (a + b)

theorem problem_1 : oplus (oplus 3 5) (oplus 5 4) = 60 / 59 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l545_54524


namespace NUMINAMATH_CALUDE_common_rational_root_exists_l545_54522

theorem common_rational_root_exists :
  ∃ (k : ℚ) (a b c d e f g : ℚ),
    k = -1/3 ∧
    k < 0 ∧
    ¬(∃ n : ℤ, k = n) ∧
    90 * k^4 + a * k^3 + b * k^2 + c * k + 18 = 0 ∧
    18 * k^5 + d * k^4 + e * k^3 + f * k^2 + g * k + 90 = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_rational_root_exists_l545_54522


namespace NUMINAMATH_CALUDE_ship_length_l545_54596

/-- The length of a ship given its speed and time to pass a fixed point -/
theorem ship_length (speed : ℝ) (time : ℝ) (h1 : speed = 18) (h2 : time = 20) :
  speed * time * (1000 / 3600) = 100 := by
  sorry

#check ship_length

end NUMINAMATH_CALUDE_ship_length_l545_54596


namespace NUMINAMATH_CALUDE_sets_intersection_empty_l545_54500

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x < 2}

theorem sets_intersection_empty : A ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_sets_intersection_empty_l545_54500


namespace NUMINAMATH_CALUDE_odd_number_1991_in_group_32_l545_54570

/-- The n-th group of odd numbers contains (2n-1) numbers -/
def group_size (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of odd numbers up to the n-th group -/
def sum_up_to_group (n : ℕ) : ℕ := n^2

/-- The position of 1991 in the sequence of odd numbers -/
def target : ℕ := 1991

/-- The theorem stating that 1991 is in the 32nd group -/
theorem odd_number_1991_in_group_32 :
  ∃ (n : ℕ), n = 32 ∧ 
  sum_up_to_group (n - 1) < target ∧ 
  target ≤ sum_up_to_group n :=
sorry

end NUMINAMATH_CALUDE_odd_number_1991_in_group_32_l545_54570


namespace NUMINAMATH_CALUDE_students_taking_paper_c_l545_54577

/-- Represents the systematic sampling setup for the school test -/
structure SchoolSampling where
  total_students : ℕ
  sample_size : ℕ
  first_selected : ℕ
  sampling_interval : ℕ

/-- Calculates the nth term in the arithmetic sequence of selected student numbers -/
def nth_selected (s : SchoolSampling) (n : ℕ) : ℕ :=
  s.first_selected + s.sampling_interval * (n - 1)

/-- Theorem stating the number of students taking test paper C -/
theorem students_taking_paper_c (s : SchoolSampling) 
  (h1 : s.total_students = 800)
  (h2 : s.sample_size = 40)
  (h3 : s.first_selected = 18)
  (h4 : s.sampling_interval = 20) :
  (Finset.filter (fun n => 561 ≤ nth_selected s n ∧ nth_selected s n ≤ 800) 
    (Finset.range s.sample_size)).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_paper_c_l545_54577


namespace NUMINAMATH_CALUDE_cubic_factorization_l545_54598

theorem cubic_factorization (a : ℝ) : a^3 - 9*a = a*(a+3)*(a-3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l545_54598


namespace NUMINAMATH_CALUDE_wednesday_sales_l545_54520

def initial_stock : ℕ := 700
def monday_sales : ℕ := 50
def tuesday_sales : ℕ := 82
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40
def unsold_percentage : ℚ := 60 / 100

theorem wednesday_sales :
  let total_sales := initial_stock - (initial_stock * unsold_percentage).floor
  let other_days_sales := monday_sales + tuesday_sales + thursday_sales + friday_sales
  total_sales - other_days_sales = 60 := by
sorry

end NUMINAMATH_CALUDE_wednesday_sales_l545_54520


namespace NUMINAMATH_CALUDE_full_bucket_weight_l545_54506

/-- Represents the weight of a bucket with water -/
structure BucketWeight where
  empty : ℝ  -- Weight of the empty bucket
  full : ℝ   -- Weight of water when bucket is full

/-- Given conditions about the bucket weights -/
def bucket_conditions (p q : ℝ) (b : BucketWeight) : Prop :=
  b.empty + (3/4 * b.full) = p ∧ b.empty + (1/3 * b.full) = q

/-- Theorem stating the weight of a fully full bucket -/
theorem full_bucket_weight (p q : ℝ) (b : BucketWeight) 
  (h : bucket_conditions p q b) : 
  b.empty + b.full = (8*p - 3*q) / 5 := by
  sorry

end NUMINAMATH_CALUDE_full_bucket_weight_l545_54506


namespace NUMINAMATH_CALUDE_complete_square_sum_l545_54543

theorem complete_square_sum (x : ℝ) : 
  (∃ (a b c : ℤ), a > 0 ∧ 
   64 * x^2 + 96 * x - 128 = 0 ↔ (a * x + b)^2 = c) →
  (∃ (a b c : ℤ), a > 0 ∧ 
   64 * x^2 + 96 * x - 128 = 0 ↔ (a * x + b)^2 = c ∧
   a + b + c = 178) := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l545_54543


namespace NUMINAMATH_CALUDE_turtle_problem_l545_54548

theorem turtle_problem (initial_turtles : ℕ) (h1 : initial_turtles = 9) : 
  let new_turtles := 3 * initial_turtles - 2
  let total_turtles := initial_turtles + new_turtles
  let remaining_turtles := total_turtles / 2
  remaining_turtles = 17 := by
sorry

end NUMINAMATH_CALUDE_turtle_problem_l545_54548


namespace NUMINAMATH_CALUDE_ray_walks_dog_three_times_daily_l545_54560

/-- The number of times Ray walks his dog each day -/
def walks_per_day (route_length total_distance : ℕ) : ℕ :=
  total_distance / route_length

theorem ray_walks_dog_three_times_daily :
  let route_length : ℕ := 4 + 7 + 11
  let total_distance : ℕ := 66
  walks_per_day route_length total_distance = 3 := by
  sorry

end NUMINAMATH_CALUDE_ray_walks_dog_three_times_daily_l545_54560


namespace NUMINAMATH_CALUDE_sam_seashells_l545_54533

/-- The number of seashells Sam found on the beach -/
def total_seashells : ℕ := 35

/-- The number of seashells Sam gave to Joan -/
def seashells_given : ℕ := 18

/-- The number of seashells Sam has now -/
def seashells_remaining : ℕ := 17

/-- Theorem stating that the total number of seashells Sam found is equal to
    the sum of seashells given away and seashells remaining -/
theorem sam_seashells : 
  total_seashells = seashells_given + seashells_remaining := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l545_54533


namespace NUMINAMATH_CALUDE_john_savings_period_l545_54525

/-- Calculates the number of years saved given monthly savings, recent expense, and remaining balance -/
def years_saved (monthly_saving : ℕ) (recent_expense : ℕ) (remaining_balance : ℕ) : ℚ :=
  (recent_expense + remaining_balance) / (monthly_saving * 12)

theorem john_savings_period :
  let monthly_saving : ℕ := 25
  let recent_expense : ℕ := 400
  let remaining_balance : ℕ := 200
  years_saved monthly_saving recent_expense remaining_balance = 2 := by sorry

end NUMINAMATH_CALUDE_john_savings_period_l545_54525


namespace NUMINAMATH_CALUDE_giraffe_contest_minimum_voters_l545_54514

structure VotingSystem where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat

def minimum_voters_to_win (vs : VotingSystem) : Nat :=
  2 * ((vs.num_districts + 1) / 2) * ((vs.sections_per_district + 1) / 2)

theorem giraffe_contest_minimum_voters 
  (vs : VotingSystem)
  (h1 : vs.total_voters = 105)
  (h2 : vs.num_districts = 5)
  (h3 : vs.sections_per_district = 7)
  (h4 : vs.voters_per_section = 3)
  (h5 : vs.total_voters = vs.num_districts * vs.sections_per_district * vs.voters_per_section) :
  minimum_voters_to_win vs = 24 := by
  sorry

#eval minimum_voters_to_win ⟨105, 5, 7, 3⟩

end NUMINAMATH_CALUDE_giraffe_contest_minimum_voters_l545_54514


namespace NUMINAMATH_CALUDE_isosceles_triangle_locus_l545_54552

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def isIsosceles (t : Triangle) : Prop :=
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = (t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2

def satisfiesLocus (C : ℝ × ℝ) : Prop :=
  C.1^2 + C.2^2 - 6*C.1 + 4*C.2 - 5 = 0

theorem isosceles_triangle_locus :
  ∀ t : Triangle,
    t.A = (3, -2) →
    t.B = (0, 1) →
    isIsosceles t →
    t.C ≠ (0, 1) →
    t.C ≠ (6, -5) →
    satisfiesLocus t.C :=
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_locus_l545_54552


namespace NUMINAMATH_CALUDE_eighth_term_is_eight_l545_54585

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference of the sequence
  d : ℝ
  -- The sum of the first six terms is 21
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 21
  -- The seventh term is 7
  seventh_term : a + 6*d = 7

/-- The eighth term of the arithmetic sequence is 8 -/
theorem eighth_term_is_eight (seq : ArithmeticSequence) : seq.a + 7*seq.d = 8 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_eight_l545_54585


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_20000_l545_54590

def hulk_jump (n : ℕ) : ℝ := 3 * (3 ^ (n - 1))

theorem hulk_jump_exceeds_20000 :
  (∀ m : ℕ, m < 10 → hulk_jump m ≤ 20000) ∧
  hulk_jump 10 > 20000 := by
sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_20000_l545_54590


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l545_54582

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 15 (2*x + 1) = Nat.choose 15 (x + 2)) ↔ (x = 1 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l545_54582


namespace NUMINAMATH_CALUDE_part_one_part_two_l545_54566

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ x > 5}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 4}

-- Part 1
theorem part_one :
  (A ∩ B 2 = {x | 5 < x ∧ x < 6}) ∧
  (Set.univ \ A = {x | -1 ≤ x ∧ x ≤ 5}) :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  B a ⊆ (Set.univ \ A) ↔ a ∈ Set.Iic 3 ∪ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l545_54566


namespace NUMINAMATH_CALUDE_system_solution_negative_implies_m_range_l545_54540

theorem system_solution_negative_implies_m_range (m : ℝ) : 
  (∃ x y : ℝ, x - y = 2*m + 7 ∧ x + y = 4*m - 3 ∧ x < 0 ∧ y < 0) → m < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_negative_implies_m_range_l545_54540


namespace NUMINAMATH_CALUDE_sin_sum_max_value_l545_54568

open Real

theorem sin_sum_max_value (x₁ x₂ x₃ : ℝ) 
  (h₁ : 0 < x₁ ∧ x₁ < π) 
  (h₂ : 0 < x₂ ∧ x₂ < π) 
  (h₃ : 0 < x₃ ∧ x₃ < π) 
  (h_sum : x₁ + x₂ + x₃ = π) : 
  sin x₁ + sin x₂ + sin x₃ ≤ 2 * sqrt 3 / 3 := by
  sorry

#check sin_sum_max_value

end NUMINAMATH_CALUDE_sin_sum_max_value_l545_54568


namespace NUMINAMATH_CALUDE_pentagon_cannot_tile_l545_54527

/-- Represents a regular polygon --/
inductive RegularPolygon
  | EquilateralTriangle
  | Square
  | Pentagon
  | Hexagon

/-- Calculates the interior angle of a regular polygon with n sides --/
def interiorAngle (n : ℕ) : ℚ :=
  180 - (360 / n)

/-- Checks if a polygon can tile the plane --/
def canTilePlane (p : RegularPolygon) : Prop :=
  match p with
  | RegularPolygon.EquilateralTriangle => (360 / interiorAngle 3).isInt
  | RegularPolygon.Square => (360 / interiorAngle 4).isInt
  | RegularPolygon.Pentagon => (360 / interiorAngle 5).isInt
  | RegularPolygon.Hexagon => (360 / interiorAngle 6).isInt

/-- Theorem stating that only the pentagon cannot tile the plane --/
theorem pentagon_cannot_tile :
  ∀ p : RegularPolygon,
    ¬(canTilePlane p) ↔ p = RegularPolygon.Pentagon :=
by sorry

end NUMINAMATH_CALUDE_pentagon_cannot_tile_l545_54527


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_chord_intersection_equality_l545_54591

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane. -/
def Point := ℝ × ℝ

/-- The distance between two points. -/
def distance (p q : Point) : ℝ := sorry

/-- Checks if a point lies on a circle. -/
def onCircle (c : Circle) (p : Point) : Prop := 
  distance c.center p = c.radius

/-- Represents a chord of a circle. -/
structure Chord (c : Circle) where
  p1 : Point
  p2 : Point
  h1 : onCircle c p1
  h2 : onCircle c p2

/-- Theorem: For two intersecting chords and a line through their intersection point,
    the product of the distances from the intersection point to the endpoints of one chord
    is equal to the product of the distances from the intersection point to the endpoints of the other chord. -/
theorem intersecting_chords_theorem (c : Circle) (ab cd : Chord c) (e f g h i : Point) : 
  onCircle c f ∧ onCircle c g ∧ onCircle c h ∧ onCircle c i →
  distance e f * distance e g = distance e h * distance e i :=
sorry

/-- Main theorem to prove -/
theorem chord_intersection_equality (c : Circle) (ab cd : Chord c) (e f g h i : Point) : 
  onCircle c f ∧ onCircle c g ∧ onCircle c h ∧ onCircle c i →
  distance f g = distance h i :=
sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_chord_intersection_equality_l545_54591


namespace NUMINAMATH_CALUDE_income_calculation_l545_54558

/-- Proves that given a person's income and expenditure are in the ratio 4:3, 
    and their savings are Rs. 5,000, their income is Rs. 20,000. -/
theorem income_calculation (income expenditure savings : ℕ) : 
  income * 3 = expenditure * 4 →  -- Income and expenditure ratio is 4:3
  income - expenditure = savings → -- Savings definition
  savings = 5000 →                -- Given savings amount
  income = 20000 :=               -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_income_calculation_l545_54558


namespace NUMINAMATH_CALUDE_equation_solution_l545_54553

theorem equation_solution : 
  ∀ x : ℤ, x * (x + 1) = 2014 * 2015 ↔ x = 2014 ∨ x = -2015 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l545_54553


namespace NUMINAMATH_CALUDE_quadratic_equation_implication_l545_54529

theorem quadratic_equation_implication (x : ℝ) : 
  x^2 + 3*x + 5 = 7 → 3*x^2 + 9*x - 11 = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_implication_l545_54529


namespace NUMINAMATH_CALUDE_min_balls_for_three_same_color_60_6_l545_54559

/-- Given a bag with colored balls, returns the minimum number of balls
    that must be picked to ensure at least three balls of the same color are picked. -/
def min_balls_for_three_same_color (total_balls : ℕ) (balls_per_color : ℕ) : ℕ :=
  2 * (total_balls / balls_per_color) + 1

/-- Proves that for a bag with 60 balls and 6 balls of each color,
    the minimum number of balls to pick to ensure at least three of the same color is 21. -/
theorem min_balls_for_three_same_color_60_6 :
  min_balls_for_three_same_color 60 6 = 21 := by
  sorry

#eval min_balls_for_three_same_color 60 6

end NUMINAMATH_CALUDE_min_balls_for_three_same_color_60_6_l545_54559


namespace NUMINAMATH_CALUDE_x_y_inequality_l545_54580

theorem x_y_inequality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + 2 * |y| = 2 * x * y) : 
  (x > 0 → x + y > 3) ∧ (x < 0 → x + y < -3) := by
  sorry

end NUMINAMATH_CALUDE_x_y_inequality_l545_54580


namespace NUMINAMATH_CALUDE_rent_ratio_increase_l545_54573

/-- The ratio of rent spent this year compared to last year, given changes in income and rent percentage --/
theorem rent_ratio_increase (last_year_rent_percent : ℝ) (income_increase_percent : ℝ) (this_year_rent_percent : ℝ) :
  last_year_rent_percent = 0.20 →
  income_increase_percent = 0.15 →
  this_year_rent_percent = 0.25 →
  (this_year_rent_percent * (1 + income_increase_percent)) / last_year_rent_percent = 1.4375 := by
  sorry

end NUMINAMATH_CALUDE_rent_ratio_increase_l545_54573


namespace NUMINAMATH_CALUDE_all_statements_true_l545_54547

def A : Set ℝ := {-1, 2, 3}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

theorem all_statements_true :
  (A ∩ B ≠ A) ∧
  (A ∪ B ≠ B) ∧
  (3 ∉ {x : ℝ | x < -1 ∨ x ≥ 3}) ∧
  (A ∩ {x : ℝ | x < -1 ∨ x ≥ 3} ≠ ∅) := by
  sorry

end NUMINAMATH_CALUDE_all_statements_true_l545_54547


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l545_54513

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_m_value :
  ∀ m : ℝ, collinear (-2) 12 1 3 m (-6) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l545_54513


namespace NUMINAMATH_CALUDE_cube_inequality_l545_54572

theorem cube_inequality (n : ℕ+) : (n + 1)^3 ≠ n^3 + (n - 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l545_54572


namespace NUMINAMATH_CALUDE_polynomial_value_l545_54556

/-- Given a polynomial function f(x) = ax^5 + bx^3 - cx + 2 where f(-3) = 9, 
    prove that f(3) = -5 -/
theorem polynomial_value (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^5 + b * x^3 - c * x + 2
  f (-3) = 9 → f 3 = -5 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_l545_54556


namespace NUMINAMATH_CALUDE_fourth_number_proof_l545_54523

theorem fourth_number_proof (x : ℝ) (fourth_number : ℝ) : 
  (128 + 255 + 511 + 1023 + x) / 5 = 423 →
  (28 + x + 42 + fourth_number + 104) / 5 = 90 →
  fourth_number = 78 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l545_54523


namespace NUMINAMATH_CALUDE_reeta_pencils_l545_54546

theorem reeta_pencils (reeta_pencils : ℕ) 
  (h1 : reeta_pencils + (2 * reeta_pencils + 4) = 64) : 
  reeta_pencils = 20 := by
  sorry

end NUMINAMATH_CALUDE_reeta_pencils_l545_54546


namespace NUMINAMATH_CALUDE_avg_calculation_l545_54583

/-- Calculates the average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Calculates the average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- The main theorem to prove -/
theorem avg_calculation : avg3 (avg3 2 2 0) (avg2 1 2) 1 = 23 / 18 := by
  sorry

end NUMINAMATH_CALUDE_avg_calculation_l545_54583


namespace NUMINAMATH_CALUDE_smallest_m_divisible_by_15_l545_54597

/-- The largest prime with 2015 digits -/
def q : ℕ := sorry

theorem smallest_m_divisible_by_15 :
  ∃ m : ℕ, m > 0 ∧ 15 ∣ (q^2 - m) ∧ ∀ k : ℕ, 0 < k ∧ k < m → ¬(15 ∣ (q^2 - k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_divisible_by_15_l545_54597


namespace NUMINAMATH_CALUDE_box_volume_perimeter_triples_l545_54579

def is_valid_triple (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 4 * (a + b + c)

theorem box_volume_perimeter_triples :
  ∃! (n : ℕ), ∃ (S : Finset (ℕ × ℕ × ℕ)),
    S.card = n ∧
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S ↔ is_valid_triple t.1 t.2.1 t.2.2) ∧
    n = 5 :=
sorry

end NUMINAMATH_CALUDE_box_volume_perimeter_triples_l545_54579


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l545_54584

/-- Represents a right circular cone -/
structure Cone :=
  (diameter : ℝ)
  (altitude : ℝ)

/-- Represents a right circular cylinder -/
structure Cylinder :=
  (radius : ℝ)

/-- 
Theorem: The radius of a cylinder inscribed in a cone
Given:
  - The cylinder's diameter is equal to its height
  - The cone has a diameter of 12 and an altitude of 15
  - The axes of the cylinder and cone coincide
Prove: The radius of the cylinder is 10/3
-/
theorem inscribed_cylinder_radius (cone : Cone) (cyl : Cylinder) :
  cone.diameter = 12 →
  cone.altitude = 15 →
  cyl.radius * 2 = cyl.radius * 2 →  -- cylinder's diameter equals its height
  cyl.radius = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l545_54584


namespace NUMINAMATH_CALUDE_shop_length_l545_54594

/-- Given a shop with the following properties:
  * Monthly rent is 3600 (in some currency)
  * Width is 15 feet
  * Annual rent per square foot is 144 (in the same currency as monthly rent)
  Prove that the length of the shop is 20 feet -/
theorem shop_length (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) :
  monthly_rent = 3600 →
  width = 15 →
  annual_rent_per_sqft = 144 →
  (monthly_rent * 12) / (width * annual_rent_per_sqft) = 20 :=
by sorry

end NUMINAMATH_CALUDE_shop_length_l545_54594


namespace NUMINAMATH_CALUDE_line_through_point_l545_54536

/-- Given a line equation 3bx + (2b-1)y = 5b - 3 that passes through the point (3, -7),
    prove that b = 1. -/
theorem line_through_point (b : ℝ) : 
  (3 * b * 3 + (2 * b - 1) * (-7) = 5 * b - 3) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l545_54536


namespace NUMINAMATH_CALUDE_intersection_points_count_l545_54509

theorem intersection_points_count : ∃! (points : Finset (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ points ↔ (9*x^2 + 4*y^2 = 36 ∧ 4*x^2 + 9*y^2 = 36)) ∧
  points.card = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_points_count_l545_54509


namespace NUMINAMATH_CALUDE_pascals_triangle_20th_row_5th_number_l545_54511

theorem pascals_triangle_20th_row_5th_number : 
  let n : ℕ := 20  -- The row number (0-indexed)
  let k : ℕ := 4   -- The position in the row (0-indexed)
  Nat.choose n k = 4845 := by
sorry

end NUMINAMATH_CALUDE_pascals_triangle_20th_row_5th_number_l545_54511


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l545_54544

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite in sign but equal in magnitude. -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

/-- Given two points M(a,3) and N(4,b) symmetric about the x-axis,
    prove that a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_about_x_axis (a, 3) (4, b)) : a + b = 1 := by
  sorry


end NUMINAMATH_CALUDE_symmetric_points_sum_l545_54544


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l545_54561

theorem roof_dimension_difference :
  ∀ (width length : ℝ),
  width > 0 →
  length = 4 * width →
  width * length = 768 →
  length - width = 24 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l545_54561


namespace NUMINAMATH_CALUDE_circular_garden_area_l545_54501

/-- Proves that a circular garden with radius 6 and fence length equal to 1/3 of its area has an area of 36π square units -/
theorem circular_garden_area (r : ℝ) (h1 : r = 6) : 
  (2 * Real.pi * r = (1/3) * Real.pi * r^2) → Real.pi * r^2 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_area_l545_54501


namespace NUMINAMATH_CALUDE_unique_virtual_square_plus_one_l545_54515

def is_virtual (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * a + b

def is_square_plus_one (n : ℕ) : Prop :=
  ∃ (m : ℕ), n = m^2 + 1

theorem unique_virtual_square_plus_one :
  ∃! (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ is_virtual n ∧ is_square_plus_one n ∧ n = 8282 :=
sorry

end NUMINAMATH_CALUDE_unique_virtual_square_plus_one_l545_54515


namespace NUMINAMATH_CALUDE_intersection_not_solution_quadratic_solution_l545_54575

theorem intersection_not_solution : ∀ x y : ℝ,
  (y = x ∧ y = x - 4) → (x ≠ 2 ∨ y ≠ 2) :=
by sorry

theorem quadratic_solution : ∀ x : ℝ,
  x^2 - 4*x + 4 = 0 → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_not_solution_quadratic_solution_l545_54575


namespace NUMINAMATH_CALUDE_equation_has_four_real_solutions_l545_54517

theorem equation_has_four_real_solutions :
  let f : ℝ → ℝ := λ x => 6*x/(x^2 + x + 1) + 7*x/(x^2 - 7*x + 2) + 5/2
  (∃ (a b c d : ℝ), (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)) ∧
  (∀ (w x y z : ℝ), (∀ r, f r = 0 ↔ r = w ∨ r = x ∨ r = y ∨ r = z) →
    w = a ∧ x = b ∧ y = c ∧ z = d) :=
by sorry

end NUMINAMATH_CALUDE_equation_has_four_real_solutions_l545_54517


namespace NUMINAMATH_CALUDE_room_width_calculation_l545_54537

/-- Given a rectangular room with specified length, paving cost per square meter,
    and total paving cost, prove that the width of the room is as calculated. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : cost_per_sqm = 600)
    (h3 : total_cost = 12375) :
    total_cost / (cost_per_sqm * length) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l545_54537


namespace NUMINAMATH_CALUDE_larger_sphere_radius_l545_54541

/-- The radius of a sphere with volume equal to 12 spheres of radius 0.5 inches is ³√3 inches. -/
theorem larger_sphere_radius (r : ℝ) : 
  (4 / 3 * Real.pi * r^3 = 12 * (4 / 3 * Real.pi * (1 / 2)^3)) → r = (3 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_larger_sphere_radius_l545_54541


namespace NUMINAMATH_CALUDE_negation_equivalence_l545_54593

theorem negation_equivalence :
  (¬(∀ x : ℝ, |x| ≥ 2 → (x ≥ 2 ∨ x ≤ -2))) ↔ (∀ x : ℝ, |x| < 2 → (-2 < x ∧ x < 2)) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l545_54593


namespace NUMINAMATH_CALUDE_ball_count_l545_54531

theorem ball_count (white green yellow red purple : ℕ)
  (h1 : white = 50)
  (h2 : green = 30)
  (h3 : yellow = 10)
  (h4 : red = 7)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 0.9) :
  white + green + yellow + red + purple = 100 := by
sorry

end NUMINAMATH_CALUDE_ball_count_l545_54531


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l545_54530

theorem quadratic_symmetry (a : ℝ) :
  (∃ (a : ℝ), 4 = a * (-2)^2) → (∃ (a : ℝ), 4 = a * 2^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l545_54530


namespace NUMINAMATH_CALUDE_cubic_equation_root_a_value_l545_54508

theorem cubic_equation_root_a_value :
  ∀ a b : ℚ,
  (∃ x : ℝ, x^3 + a*x^2 + b*x - 48 = 0 ∧ x = 2 - 5*Real.sqrt 3) →
  a = -332/71 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_a_value_l545_54508


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l545_54516

/-- Given that 3i - 2 is a root of the quadratic equation 2x^2 + px + q = 0,
    prove that p + q = 34. -/
theorem quadratic_root_sum (p q : ℝ) : 
  (2 * (Complex.I * 3 - 2)^2 + p * (Complex.I * 3 - 2) + q = 0) →
  p + q = 34 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l545_54516


namespace NUMINAMATH_CALUDE_parabola_point_distance_l545_54535

/-- Given a parabola y² = 4x and a point A on the parabola,
    if the distance from A to the focus is 4,
    then the distance from A to the origin is √21. -/
theorem parabola_point_distance (A : ℝ × ℝ) :
  A.1 ≥ 0 →  -- Ensure x-coordinate is non-negative
  A.2^2 = 4 * A.1 →  -- A is on the parabola
  (A.1 - 1)^2 + A.2^2 = 16 →  -- Distance from A to focus (1, 0) is 4
  A.1^2 + A.2^2 = 21 :=  -- Distance from A to origin is √21
by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l545_54535


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l545_54550

/-- The number of bottle caps Danny has after throwing some away and finding new ones -/
def final_bottle_caps (initial : ℕ) (thrown_away : ℕ) (found : ℕ) : ℕ :=
  initial - thrown_away + found

/-- Theorem stating that Danny's final bottle cap count is 67 -/
theorem danny_bottle_caps :
  final_bottle_caps 69 60 58 = 67 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l545_54550


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l545_54512

theorem distinct_prime_factors_count : 
  (Finset.card (Nat.factors (85 * 87 * 91 * 94)).toFinset) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l545_54512
