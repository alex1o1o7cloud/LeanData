import Mathlib

namespace pants_cost_theorem_l2729_272965

/-- Represents the cost and pricing strategy for a pair of pants -/
structure PantsPricing where
  cost : ℝ
  profit_percentage : ℝ
  discount_percentage : ℝ
  final_price : ℝ

/-- Calculates the selling price before discount -/
def selling_price (p : PantsPricing) : ℝ :=
  p.cost * (1 + p.profit_percentage)

/-- Calculates the final selling price after discount -/
def discounted_price (p : PantsPricing) : ℝ :=
  selling_price p * (1 - p.discount_percentage)

/-- Theorem stating the relationship between the cost and final price -/
theorem pants_cost_theorem (p : PantsPricing) 
  (h1 : p.profit_percentage = 0.30)
  (h2 : p.discount_percentage = 0.20)
  (h3 : p.final_price = 130)
  : p.cost = 125 := by
  sorry

end pants_cost_theorem_l2729_272965


namespace money_ratio_proof_l2729_272985

def ram_money : ℝ := 490
def krishan_money : ℝ := 2890

/-- The ratio of money between two people -/
structure MoneyRatio where
  person1 : ℝ
  person2 : ℝ

/-- The condition that two money ratios are equal -/
def equal_ratios (r1 r2 : MoneyRatio) : Prop :=
  r1.person1 / r1.person2 = r2.person1 / r2.person2

theorem money_ratio_proof 
  (ram_gopal : MoneyRatio) 
  (gopal_krishan : MoneyRatio) 
  (h1 : ram_gopal.person1 = ram_money)
  (h2 : gopal_krishan.person2 = krishan_money)
  (h3 : equal_ratios ram_gopal gopal_krishan) :
  ∃ (n : ℕ), 
    ram_gopal.person1 / ram_gopal.person2 = 49 / 119 ∧ 
    n * ram_gopal.person1 = 49 ∧ 
    n * ram_gopal.person2 = 119 := by
  sorry

end money_ratio_proof_l2729_272985


namespace range_of_m_l2729_272951

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 2/x + 1/y = 1) (h2 : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) :
  m ∈ Set.Ioo (-4 : ℝ) 2 :=
sorry

end range_of_m_l2729_272951


namespace inverse_15_mod_1003_l2729_272991

theorem inverse_15_mod_1003 : ∃ x : ℤ, 0 ≤ x ∧ x < 1003 ∧ (15 * x) % 1003 = 1 :=
by
  use 937
  sorry

end inverse_15_mod_1003_l2729_272991


namespace train_length_l2729_272932

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 6 → 
  ∃ (length_m : ℝ), abs (length_m - 100.02) < 0.01 := by sorry

end train_length_l2729_272932


namespace school_boys_count_l2729_272921

theorem school_boys_count (muslim_percent : ℝ) (hindu_percent : ℝ) (sikh_percent : ℝ) (other_count : ℕ) :
  muslim_percent = 44 →
  hindu_percent = 28 →
  sikh_percent = 10 →
  other_count = 54 →
  ∃ (total : ℕ), 
    (muslim_percent + hindu_percent + sikh_percent + (other_count : ℝ) / (total : ℝ) * 100 = 100) ∧
    total = 300 := by
  sorry

end school_boys_count_l2729_272921


namespace min_lines_for_200_intersections_l2729_272963

/-- The number of intersection points for m lines -/
def intersectionPoints (m : ℕ) : ℕ := m * (m - 1) / 2

/-- The minimum number of lines that intersect in exactly 200 points -/
def minLines : ℕ := 21

theorem min_lines_for_200_intersections :
  (intersectionPoints minLines = 200) ∧
  (∀ k : ℕ, k < minLines → intersectionPoints k < 200) := by
  sorry

end min_lines_for_200_intersections_l2729_272963


namespace expression_evaluation_l2729_272930

theorem expression_evaluation : 
  |-2| + (1/4 : ℝ) - 1 - 4 * Real.cos (π/4) + Real.sqrt 8 = 5/4 := by sorry

end expression_evaluation_l2729_272930


namespace quadratic_vertex_l2729_272953

/-- The quadratic function f(x) = 3(x+4)^2 - 5 has vertex at (-4, -5) -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * (x + 4)^2 - 5
  (∀ x, f x = 3 * (x + 4)^2 - 5) →
  ∃! (h k : ℝ), ∀ x, f x = 3 * (x - h)^2 + k ∧ h = -4 ∧ k = -5 :=
by sorry

end quadratic_vertex_l2729_272953


namespace fraction_simplification_l2729_272931

theorem fraction_simplification (y : ℝ) (h : y = 3) :
  (y^8 + 10*y^4 + 25) / (y^4 + 5) = 86 := by
  sorry

end fraction_simplification_l2729_272931


namespace hiker_distance_l2729_272961

theorem hiker_distance (north east south east2 : ℝ) 
  (h1 : north = 15)
  (h2 : east = 8)
  (h3 : south = 9)
  (h4 : east2 = 2) : 
  Real.sqrt ((north - south)^2 + (east + east2)^2) = 2 * Real.sqrt 34 := by
  sorry

end hiker_distance_l2729_272961


namespace jamal_book_cart_l2729_272945

theorem jamal_book_cart (history_books fiction_books childrens_books wrong_place_books remaining_books : ℕ) :
  history_books = 12 →
  fiction_books = 19 →
  childrens_books = 8 →
  wrong_place_books = 4 →
  remaining_books = 16 →
  history_books + fiction_books + childrens_books + wrong_place_books + remaining_books = 59 := by
  sorry

end jamal_book_cart_l2729_272945


namespace regular_polygon_diagonals_l2729_272959

/-- A regular polygon with exterior angle of 36 degrees has 7 diagonals from each vertex -/
theorem regular_polygon_diagonals (n : ℕ) (h_regular : n ≥ 3) :
  (360 : ℝ) / 36 = n → n - 3 = 7 := by sorry

end regular_polygon_diagonals_l2729_272959


namespace candy_mixture_total_candy_mixture_total_proof_l2729_272975

/-- Proves that the total amount of candy needed is 80 pounds given the problem conditions --/
theorem candy_mixture_total (price_cheap price_expensive price_mixture : ℚ) 
                            (amount_cheap : ℚ) (total_amount : ℚ) : Prop :=
  price_cheap = 2 ∧ 
  price_expensive = 3 ∧ 
  price_mixture = 2.2 ∧
  amount_cheap = 64 ∧
  total_amount = 80 ∧
  ∃ (amount_expensive : ℚ),
    amount_cheap + amount_expensive = total_amount ∧
    (amount_cheap * price_cheap + amount_expensive * price_expensive) / total_amount = price_mixture
    
theorem candy_mixture_total_proof : 
  candy_mixture_total 2 3 2.2 64 80 := by
  sorry

end candy_mixture_total_candy_mixture_total_proof_l2729_272975


namespace arithmetic_expression_equality_l2729_272926

theorem arithmetic_expression_equality : 7 ^ 8 - 6 / 2 + 9 ^ 3 + 3 + 12 = 5765542 := by
  sorry

end arithmetic_expression_equality_l2729_272926


namespace production_quantity_for_36000_min_production_for_profit_8500_l2729_272973

-- Define the production cost function
def C (n : ℕ) : ℝ := 4000 + 50 * n

-- Define the profit function
def P (n : ℕ) : ℝ := 40 * n - 4000

-- Theorem 1: Production quantity when cost is 36,000
theorem production_quantity_for_36000 :
  ∃ n : ℕ, C n = 36000 ∧ n = 640 := by sorry

-- Theorem 2: Minimum production for profit ≥ 8,500
theorem min_production_for_profit_8500 :
  ∃ n : ℕ, (∀ m : ℕ, P m ≥ 8500 → m ≥ n) ∧ P n ≥ 8500 ∧ n = 313 := by sorry

end production_quantity_for_36000_min_production_for_profit_8500_l2729_272973


namespace square_sum_pairs_l2729_272997

theorem square_sum_pairs : 
  {(a, b) : ℕ × ℕ | ∃ (m n : ℕ), a^2 + 3*b = m^2 ∧ b^2 + 3*a = n^2} = 
  {(1, 1), (11, 11), (16, 11)} := by
sorry

end square_sum_pairs_l2729_272997


namespace total_paper_pieces_l2729_272966

theorem total_paper_pieces : 
  let olivia_pieces : ℕ := 127
  let edward_pieces : ℕ := 345
  let sam_pieces : ℕ := 518
  olivia_pieces + edward_pieces + sam_pieces = 990 :=
by sorry

end total_paper_pieces_l2729_272966


namespace hyperbola_eccentricity_l2729_272978

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (e : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / a^2 - y₁^2 / b^2 = 1 ∧ 
    x₂^2 / a^2 - y₂^2 / b^2 = 1 ∧
    y₁ - y₂ = x₁ - x₂ ∧
    x₁ > c ∧ x₂ > c) →
  (2 * b^2) / a = (2 * Real.sqrt 2 / 3) * b * e^2 →
  e = Real.sqrt 6 / 2 := by
  sorry

end hyperbola_eccentricity_l2729_272978


namespace birthday_cards_count_l2729_272925

def total_amount_spent : ℕ := 70
def cost_per_card : ℕ := 2
def christmas_cards : ℕ := 20

def total_cards : ℕ := total_amount_spent / cost_per_card

def birthday_cards : ℕ := total_cards - christmas_cards

theorem birthday_cards_count : birthday_cards = 15 := by
  sorry

end birthday_cards_count_l2729_272925


namespace collinearity_of_special_points_l2729_272976

-- Define the triangle and points
variable (A B C A' B' C' A'' B'' C'' : ℝ × ℝ)

-- Define the conditions
def is_scalene_triangle (A B C : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

def is_angle_bisector_point (X Y Z X' : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), X' = t • Y + (1 - t) • Z ∧ 
  (X.1 - Y.1) * (X'.1 - Z.1) = (X.2 - Y.2) * (X'.2 - Z.2)

def is_perpendicular_bisector_point (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2) = 0 ∧
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2

-- State the theorem
theorem collinearity_of_special_points 
  (h_scalene : is_scalene_triangle A B C)
  (h_A' : is_angle_bisector_point A B C A')
  (h_B' : is_angle_bisector_point B C A B')
  (h_C' : is_angle_bisector_point C A B C')
  (h_A'' : is_perpendicular_bisector_point A A' A'')
  (h_B'' : is_perpendicular_bisector_point B B' B'')
  (h_C'' : is_perpendicular_bisector_point C C' C'') :
  ∃ (m b : ℝ), A''.2 = m * A''.1 + b ∧ 
               B''.2 = m * B''.1 + b ∧ 
               C''.2 = m * C''.1 + b :=
sorry

end collinearity_of_special_points_l2729_272976


namespace train_crossing_time_l2729_272962

/-- Time taken for a faster train to cross a man in a slower train -/
theorem train_crossing_time (faster_speed slower_speed : ℝ) (train_length : ℝ) : 
  faster_speed = 54 →
  slower_speed = 36 →
  train_length = 135 →
  (train_length / (faster_speed - slower_speed)) * (3600 / 1000) = 27 := by
  sorry

end train_crossing_time_l2729_272962


namespace quadratic_solutions_inequality_solution_set_l2729_272988

-- Part 1: Quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

theorem quadratic_solutions : 
  ∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = -1 ∧ 
  ∀ x : ℝ, quadratic_equation x ↔ (x = x1 ∨ x = x2) := by sorry

-- Part 2: Inequality system
def inequality_system (x : ℝ) : Prop := 3*x - 1 ≥ 5 ∧ (1 + 2*x) / 3 > x - 1

theorem inequality_solution_set :
  ∀ x : ℝ, inequality_system x ↔ 2 ≤ x ∧ x < 4 := by sorry

end quadratic_solutions_inequality_solution_set_l2729_272988


namespace train_speed_calculation_l2729_272937

/-- Given two trains A and B moving towards each other, calculate the speed of train B. -/
theorem train_speed_calculation (length_A length_B : ℝ) (speed_A : ℝ) (crossing_time : ℝ) 
  (h1 : length_A = 225) 
  (h2 : length_B = 150) 
  (h3 : speed_A = 54) 
  (h4 : crossing_time = 15) : 
  (((length_A + length_B) / crossing_time) * (3600 / 1000) - speed_A) = 36 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l2729_272937


namespace power_of_negative_product_l2729_272982

theorem power_of_negative_product (a : ℝ) : (-2 * a^4)^3 = -8 * a^12 := by
  sorry

end power_of_negative_product_l2729_272982


namespace min_triple_intersection_l2729_272943

theorem min_triple_intersection (U : Finset Nat) (A B C : Finset Nat) : 
  Finset.card U = 30 →
  Finset.card A = 26 →
  Finset.card B = 23 →
  Finset.card C = 21 →
  A ⊆ U →
  B ⊆ U →
  C ⊆ U →
  10 ≤ Finset.card (A ∩ B ∩ C) :=
by sorry

end min_triple_intersection_l2729_272943


namespace marcus_initial_mileage_l2729_272950

/-- Represents the mileage and fuel efficiency of a car --/
structure Car where
  mpg : ℕ  -- Miles per gallon
  tankCapacity : ℕ  -- Gallons
  currentMileage : ℕ  -- Current mileage

/-- Calculates the initial mileage of a car before a road trip --/
def initialMileage (c : Car) (numFillUps : ℕ) : ℕ :=
  c.currentMileage - (c.mpg * c.tankCapacity * numFillUps)

/-- Theorem: Given the conditions of Marcus's road trip, his car's initial mileage was 1728 miles --/
theorem marcus_initial_mileage :
  let marcusCar : Car := { mpg := 30, tankCapacity := 20, currentMileage := 2928 }
  initialMileage marcusCar 2 = 1728 := by
  sorry

#eval initialMileage { mpg := 30, tankCapacity := 20, currentMileage := 2928 } 2

end marcus_initial_mileage_l2729_272950


namespace optimal_selling_price_l2729_272902

-- Define the parameters
def purchasePrice : ℝ := 40
def initialSellingPrice : ℝ := 50
def initialQuantitySold : ℝ := 50

-- Define the relationship between price increase and quantity decrease
def priceIncrease : ℝ → ℝ := λ x => x
def quantityDecrease : ℝ → ℝ := λ x => x

-- Define the selling price and quantity sold as functions of price increase
def sellingPrice : ℝ → ℝ := λ x => initialSellingPrice + priceIncrease x
def quantitySold : ℝ → ℝ := λ x => initialQuantitySold - quantityDecrease x

-- Define the revenue function
def revenue : ℝ → ℝ := λ x => sellingPrice x * quantitySold x

-- Define the cost function
def cost : ℝ → ℝ := λ x => purchasePrice * quantitySold x

-- Define the profit function
def profit : ℝ → ℝ := λ x => revenue x - cost x

-- State the theorem
theorem optimal_selling_price :
  ∃ x : ℝ, x = 20 ∧ sellingPrice x = 70 ∧ 
  ∀ y : ℝ, profit y ≤ profit x :=
sorry

end optimal_selling_price_l2729_272902


namespace midpoint_sum_equals_vertex_sum_l2729_272981

theorem midpoint_sum_equals_vertex_sum (d e f : ℝ) : 
  let vertex_sum := d + e + f
  let midpoint_sum := (d + e) / 2 + (d + f) / 2 + (e + f) / 2
  vertex_sum = midpoint_sum := by sorry

end midpoint_sum_equals_vertex_sum_l2729_272981


namespace product_prime_factors_l2729_272970

theorem product_prime_factors (m n : ℕ) : 
  (∃ p₁ p₂ p₃ p₄ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    m = p₁ * p₂ * p₃ * p₄) →
  (∃ q₁ q₂ q₃ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ 
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧
    n = q₁ * q₂ * q₃) →
  Nat.gcd m n = 15 →
  ∃ r₁ r₂ r₃ r₄ r₅ : ℕ, Prime r₁ ∧ Prime r₂ ∧ Prime r₃ ∧ Prime r₄ ∧ Prime r₅ ∧
    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧ 
    r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧
    r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧
    r₄ ≠ r₅ ∧
    m * n = r₁ * r₂ * r₃ * r₄ * r₅ :=
by
  sorry

end product_prime_factors_l2729_272970


namespace min_value_expression_l2729_272972

theorem min_value_expression (x y : ℝ) (h : 4 - 16*x^2 - 8*x*y - y^2 > 0) :
  (13*x^2 + 24*x*y + 13*y^2 - 14*x - 16*y + 61) / (4 - 16*x^2 - 8*x*y - y^2)^(7/2) ≥ 7/16 := by
  sorry

end min_value_expression_l2729_272972


namespace kat_weekly_training_hours_l2729_272920

/-- Represents Kat's weekly training schedule --/
structure TrainingSchedule where
  strength_sessions : ℕ
  strength_hours_per_session : ℝ
  boxing_sessions : ℕ
  boxing_hours_per_session : ℝ
  cardio_sessions : ℕ
  cardio_hours_per_session : ℝ
  flexibility_sessions : ℕ
  flexibility_hours_per_session : ℝ
  interval_sessions : ℕ
  interval_hours_per_session : ℝ

/-- Calculates the total weekly training hours --/
def total_weekly_hours (schedule : TrainingSchedule) : ℝ :=
  schedule.strength_sessions * schedule.strength_hours_per_session +
  schedule.boxing_sessions * schedule.boxing_hours_per_session +
  schedule.cardio_sessions * schedule.cardio_hours_per_session +
  schedule.flexibility_sessions * schedule.flexibility_hours_per_session +
  schedule.interval_sessions * schedule.interval_hours_per_session

/-- Kat's actual training schedule --/
def kat_schedule : TrainingSchedule := {
  strength_sessions := 3
  strength_hours_per_session := 1
  boxing_sessions := 4
  boxing_hours_per_session := 1.5
  cardio_sessions := 2
  cardio_hours_per_session := 0.5
  flexibility_sessions := 1
  flexibility_hours_per_session := 0.75
  interval_sessions := 1
  interval_hours_per_session := 1.25
}

/-- Theorem stating that Kat's total weekly training time is 12 hours --/
theorem kat_weekly_training_hours :
  total_weekly_hours kat_schedule = 12 := by sorry

end kat_weekly_training_hours_l2729_272920


namespace right_triangle_base_length_l2729_272938

theorem right_triangle_base_length 
  (height : ℝ) 
  (perimeter : ℝ) 
  (is_right_triangle : Bool) 
  (h1 : height = 3) 
  (h2 : perimeter = 12) 
  (h3 : is_right_triangle = true) : 
  ∃ (base : ℝ), base = 4 ∧ 
  ∃ (hypotenuse : ℝ), 
    perimeter = base + height + hypotenuse ∧
    hypotenuse^2 = base^2 + height^2 := by
  sorry

end right_triangle_base_length_l2729_272938


namespace square_sum_divided_l2729_272983

theorem square_sum_divided : (10^2 + 6^2) / 2 = 68 := by
  sorry

end square_sum_divided_l2729_272983


namespace jason_total_games_l2729_272995

/-- The number of games Jason attended this month -/
def games_this_month : ℕ := 11

/-- The number of games Jason attended last month -/
def games_last_month : ℕ := 17

/-- The number of games Jason plans to attend next month -/
def games_next_month : ℕ := 16

/-- The total number of games Jason will attend -/
def total_games : ℕ := games_this_month + games_last_month + games_next_month

theorem jason_total_games : total_games = 44 := by sorry

end jason_total_games_l2729_272995


namespace washing_machine_capacity_l2729_272969

theorem washing_machine_capacity 
  (num_families : ℕ) 
  (people_per_family : ℕ) 
  (vacation_days : ℕ) 
  (towels_per_person_per_day : ℕ) 
  (total_loads : ℕ) 
  (h1 : num_families = 3) 
  (h2 : people_per_family = 4) 
  (h3 : vacation_days = 7) 
  (h4 : towels_per_person_per_day = 1) 
  (h5 : total_loads = 6) : 
  (num_families * people_per_family * vacation_days * towels_per_person_per_day) / total_loads = 14 := by
  sorry

end washing_machine_capacity_l2729_272969


namespace average_income_proof_l2729_272918

def cab_driver_income : List ℝ := [200, 150, 750, 400, 500]

theorem average_income_proof :
  (List.sum cab_driver_income) / (List.length cab_driver_income) = 400 := by
  sorry

end average_income_proof_l2729_272918


namespace attendee_difference_is_25_l2729_272979

/-- The number of attendees from Company A -/
def company_A : ℕ := 30

/-- The number of attendees from Company B -/
def company_B : ℕ := 2 * company_A

/-- The number of attendees from Company C -/
def company_C : ℕ := company_A + 10

/-- The number of attendees from Company D -/
def company_D : ℕ := 185 - (company_A + company_B + company_C + 20)

/-- The difference in attendees between Company C and Company D -/
def attendee_difference : ℕ := company_C - company_D

theorem attendee_difference_is_25 : attendee_difference = 25 := by
  sorry

end attendee_difference_is_25_l2729_272979


namespace sport_water_amount_l2729_272916

/-- Represents the ratios in a flavored drink formulation -/
structure DrinkFormulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard : DrinkFormulation := ⟨1, 12, 30⟩

/-- The sport formulation of the drink -/
def sport : DrinkFormulation :=
  ⟨standard.flavoring, standard.corn_syrup / 3, standard.water * 2⟩

theorem sport_water_amount (corn_syrup_amount : ℚ) :
  corn_syrup_amount = 1 →
  (corn_syrup_amount * sport.water / sport.corn_syrup) = 15 := by
  sorry

end sport_water_amount_l2729_272916


namespace fraction_zero_l2729_272933

theorem fraction_zero (x : ℝ) : x = 1/2 → (2*x - 1) / (x + 2) = 0 := by
  sorry

end fraction_zero_l2729_272933


namespace pencils_to_yuna_l2729_272974

/-- The number of pencils in a dozen -/
def pencils_per_dozen : ℕ := 12

/-- The number of dozens Jimin initially had -/
def initial_dozens : ℕ := 3

/-- The number of pencils Jimin gave to his younger brother -/
def pencils_to_brother : ℕ := 8

/-- The number of pencils Jimin has left -/
def pencils_left : ℕ := 17

/-- Proves that the number of pencils Jimin gave to Yuna is 11 -/
theorem pencils_to_yuna :
  initial_dozens * pencils_per_dozen - pencils_to_brother - pencils_left = 11 := by
  sorry

end pencils_to_yuna_l2729_272974


namespace simplify_expression_l2729_272989

theorem simplify_expression (a b : ℝ) : 
  (1 : ℝ) * (2 * a) * (3 * b) * (4 * a^2 * b) * (5 * a^3 * b^2) = 120 * a^6 * b^4 := by
  sorry

end simplify_expression_l2729_272989


namespace gdp_scientific_notation_l2729_272912

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- The GDP value in billions of yuan -/
def gdp_billions : ℝ := 53100

/-- Theorem stating that the GDP in billions is equal to its scientific notation -/
theorem gdp_scientific_notation : 
  to_scientific_notation gdp_billions = ScientificNotation.mk 5.31 12 (by norm_num) :=
sorry

end gdp_scientific_notation_l2729_272912


namespace matts_writing_speed_l2729_272960

/-- Matt's writing speed problem -/
theorem matts_writing_speed (right_hand_speed : ℕ) (time : ℕ) (difference : ℕ) : 
  right_hand_speed = 10 →
  time = 5 →
  difference = 15 →
  ∃ (left_hand_speed : ℕ), 
    right_hand_speed * time = left_hand_speed * time + difference ∧
    left_hand_speed = 7 :=
by sorry

end matts_writing_speed_l2729_272960


namespace cube_sum_ge_mixed_product_cube_sum_ge_weighted_square_sum_product_l2729_272940

-- Problem 1
theorem cube_sum_ge_mixed_product (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^3 + y^3 ≥ x^2*y + x*y^2 := by sorry

-- Problem 2
theorem cube_sum_ge_weighted_square_sum_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 ≥ (1/3) * (a^2 + b^2 + c^2) * (a + b + c) := by sorry

end cube_sum_ge_mixed_product_cube_sum_ge_weighted_square_sum_product_l2729_272940


namespace total_sandwiches_count_l2729_272994

/-- The number of people going to the zoo -/
def people : ℝ := 219.0

/-- The number of sandwiches per person -/
def sandwiches_per_person : ℝ := 3.0

/-- The total number of sandwiches prepared -/
def total_sandwiches : ℝ := people * sandwiches_per_person

/-- Theorem stating that the total number of sandwiches is 657.0 -/
theorem total_sandwiches_count : total_sandwiches = 657.0 := by
  sorry

end total_sandwiches_count_l2729_272994


namespace class_average_problem_l2729_272917

theorem class_average_problem (total_students : ℕ) (high_scorers : ℕ) (zero_scorers : ℕ) 
  (high_score : ℕ) (class_average : ℚ) :
  total_students = 28 →
  high_scorers = 4 →
  zero_scorers = 3 →
  high_score = 95 →
  class_average = 47.32142857142857 →
  let remaining_students := total_students - high_scorers - zero_scorers
  let total_score := total_students * class_average
  let high_score_total := high_scorers * high_score
  let remaining_score := total_score - high_score_total
  remaining_score / remaining_students = 45 := by
    sorry

#eval (28 : ℚ) * 47.32142857142857 -- To verify the total score

end class_average_problem_l2729_272917


namespace q_value_at_minus_one_l2729_272958

-- Define the polynomial q(x)
def q (a b x : ℤ) : ℤ := x^2 + a*x + b

-- Define the two polynomials that q(x) divides
def p1 (x : ℤ) : ℤ := x^4 + 8*x^2 + 49
def p2 (x : ℤ) : ℤ := 2*x^4 + 5*x^2 + 18*x + 3

-- Theorem statement
theorem q_value_at_minus_one 
  (a b : ℤ) 
  (h1 : ∀ x, (p1 x) % (q a b x) = 0)
  (h2 : ∀ x, (p2 x) % (q a b x) = 0) :
  q a b (-1) = 66 := by
  sorry

end q_value_at_minus_one_l2729_272958


namespace disjoint_sets_imply_a_values_l2729_272910

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | (p.2 - 3) / (p.1 - 1) = 2 ∧ p.1 ≠ 1}
def B (a : ℝ) : Set (ℝ × ℝ) := {p | 4 * p.1 + a * p.2 = 16}

-- State the theorem
theorem disjoint_sets_imply_a_values (a : ℝ) :
  A ∩ B a = ∅ → a = -2 ∨ a = 4 := by
  sorry

end disjoint_sets_imply_a_values_l2729_272910


namespace sphere_volume_equals_surface_area_l2729_272928

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r = 3) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 := by
  sorry

end sphere_volume_equals_surface_area_l2729_272928


namespace siblings_weekly_water_consumption_l2729_272996

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The daily water consumption of the first sibling -/
def sibling1_daily_consumption : ℕ := 8

/-- The daily water consumption of the second sibling -/
def sibling2_daily_consumption : ℕ := 7

/-- The daily water consumption of the third sibling -/
def sibling3_daily_consumption : ℕ := 9

/-- The total water consumption of all siblings in one week -/
def total_weekly_consumption : ℕ :=
  (sibling1_daily_consumption + sibling2_daily_consumption + sibling3_daily_consumption) * days_in_week

theorem siblings_weekly_water_consumption :
  total_weekly_consumption = 168 := by
  sorry

end siblings_weekly_water_consumption_l2729_272996


namespace mark_recapture_not_suitable_for_centipedes_l2729_272903

/-- Represents a method for population quantity experiments -/
inductive PopulationExperimentMethod
| MarkRecapture
| Sampling

/-- Represents an animal type -/
inductive AnimalType
| Centipede
| Rodent

/-- Represents the size of an animal -/
inductive AnimalSize
| Small
| Large

/-- Function to determine if a method is suitable for an animal type -/
def methodSuitability (method : PopulationExperimentMethod) (animal : AnimalType) : Prop :=
  match method, animal with
  | PopulationExperimentMethod.MarkRecapture, AnimalType.Centipede => False
  | PopulationExperimentMethod.Sampling, AnimalType.Centipede => True
  | _, _ => True

/-- Function to determine the size of an animal -/
def animalSize (animal : AnimalType) : AnimalSize :=
  match animal with
  | AnimalType.Centipede => AnimalSize.Small
  | AnimalType.Rodent => AnimalSize.Large

/-- Theorem stating that the mark-recapture method is not suitable for investigating centipedes -/
theorem mark_recapture_not_suitable_for_centipedes :
  ¬(methodSuitability PopulationExperimentMethod.MarkRecapture AnimalType.Centipede) :=
by sorry

end mark_recapture_not_suitable_for_centipedes_l2729_272903


namespace team_combinations_l2729_272911

theorem team_combinations (n m : ℕ) (h1 : n = 7) (h2 : m = 4) : 
  Nat.choose n m = 35 := by
  sorry

end team_combinations_l2729_272911


namespace perfect_numbers_mn_value_S_is_perfect_min_sum_value_l2729_272909

/-- Definition of a perfect number -/
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

/-- Statement 1: 29 and 13 are perfect numbers -/
theorem perfect_numbers : is_perfect_number 29 ∧ is_perfect_number 13 := by sorry

/-- Statement 2: Given equation has mn = ±4 -/
theorem mn_value (m n : ℝ) : 
  (∀ a : ℝ, a^2 - 4*a + 8 = (a - m)^2 + n^2) → m*n = 4 ∨ m*n = -4 := by sorry

/-- Statement 3: S is a perfect number when k = 36 -/
theorem S_is_perfect (a b : ℤ) :
  let S := a^2 + 4*a*b + 5*b^2 - 12*b + 36
  ∃ x y : ℤ, S = x^2 + y^2 := by sorry

/-- Statement 4: Minimum value of a + b is 3 -/
theorem min_sum_value (a b : ℝ) :
  -a^2 + 5*a + b - 7 = 0 → a + b ≥ 3 := by sorry

end perfect_numbers_mn_value_S_is_perfect_min_sum_value_l2729_272909


namespace johns_out_of_pocket_expense_l2729_272955

/-- Calculates the amount John paid out of pocket for a new computer and accessories,
    given the costs and the sale of his PlayStation. -/
theorem johns_out_of_pocket_expense (computer_cost accessories_cost playstation_value : ℕ)
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : playstation_value = 400) :
  computer_cost + accessories_cost - (playstation_value * 80 / 100) = 580 := by
  sorry

#check johns_out_of_pocket_expense

end johns_out_of_pocket_expense_l2729_272955


namespace largest_b_divisible_by_four_l2729_272984

theorem largest_b_divisible_by_four :
  let n : ℕ → ℕ := λ b => 4000000 + b * 100000 + 508632
  ∃ (b : ℕ), b ≤ 9 ∧ n b % 4 = 0 ∧ ∀ (x : ℕ), x ≤ 9 ∧ n x % 4 = 0 → x ≤ b :=
by sorry

end largest_b_divisible_by_four_l2729_272984


namespace survey_result_l2729_272977

/-- The percentage of parents who agree to the tuition fee increase -/
def agree_percentage : ℝ := 0.20

/-- The number of parents who disagree with the tuition fee increase -/
def disagree_count : ℕ := 640

/-- The total number of parents surveyed -/
def total_parents : ℕ := 800

/-- Theorem stating that the total number of parents surveyed is 800 -/
theorem survey_result : total_parents = 800 := by
  sorry

end survey_result_l2729_272977


namespace final_balance_proof_l2729_272980

def bank_transactions (initial_balance : ℚ) : ℚ :=
  let balance1 := initial_balance - 300
  let balance2 := balance1 - 150
  let balance3 := balance2 + (3/5 * balance2)
  let balance4 := balance3 - 250
  balance4 + (2/3 * balance4)

theorem final_balance_proof :
  ∃ (initial_balance : ℚ),
    (300 = (3/7) * initial_balance) ∧
    (150 = (1/3) * (initial_balance - 300)) ∧
    (250 = (1/4) * (initial_balance - 300 - 150 + (3/5) * (initial_balance - 300 - 150))) ∧
    (bank_transactions initial_balance = 1250) :=
by sorry

end final_balance_proof_l2729_272980


namespace modulus_of_z_l2729_272924

open Complex

theorem modulus_of_z (z : ℂ) (h : (1 + I) * (1 - z) = 1) : abs z = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_z_l2729_272924


namespace books_from_first_shop_l2729_272929

theorem books_from_first_shop :
  ∀ (x : ℕ),
    (1000 : ℝ) + 800 = 20 * (x + 40) →
    x = 50 := by
  sorry

end books_from_first_shop_l2729_272929


namespace inequality_system_solution_l2729_272939

/-- Given an inequality system with solution set x < 1, find the range of a -/
theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x - 1 < 0 ∧ x < a + 3) ↔ x < 1) → a ≥ -2 :=
by sorry

end inequality_system_solution_l2729_272939


namespace circumcircle_equation_l2729_272957

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, -1)

-- Define point P on the parabola
def P : ℝ × ℝ := (-4, -4)

-- Define the tangent line at P
def tangent_line (x y : ℝ) : Prop := y = 2*x + 4

-- Define point Q as the intersection of the tangent line and x-axis
def Q : ℝ × ℝ := (-2, 0)

-- Theorem statement
theorem circumcircle_equation :
  ∀ x y : ℝ,
  parabola (P.1) (P.2) →
  tangent_line (Q.1) (Q.2) →
  (x^2 + y^2 + 4*x + 5*y + 4 = 0) ↔ 
  ((x - (-2))^2 + (y - (-5/2))^2 = (5/2)^2) :=
sorry

end circumcircle_equation_l2729_272957


namespace video_game_cost_l2729_272949

def allowance_period1 : ℕ := 8
def allowance_rate1 : ℕ := 5
def allowance_period2 : ℕ := 6
def allowance_rate2 : ℕ := 6
def remaining_money : ℕ := 3

def total_savings : ℕ := allowance_period1 * allowance_rate1 + allowance_period2 * allowance_rate2

def money_after_clothes : ℕ := total_savings / 2

theorem video_game_cost : money_after_clothes - remaining_money = 35 := by
  sorry

end video_game_cost_l2729_272949


namespace sequence_integer_count_l2729_272927

def sequence_term (n : ℕ) : ℚ :=
  15625 / (5 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n))) ∧
  (∀ (m : ℕ), m > 0 →
    ((∀ (n : ℕ), n < m → is_integer (sequence_term n)) ∧
     (∀ (n : ℕ), n ≥ m → ¬ is_integer (sequence_term n)))
    → m = 7) :=
by sorry

end sequence_integer_count_l2729_272927


namespace remaining_calories_l2729_272947

def calories_per_serving : ℕ := 110
def servings_per_block : ℕ := 16
def servings_eaten : ℕ := 5

theorem remaining_calories : 
  (servings_per_block - servings_eaten) * calories_per_serving = 1210 := by
  sorry

end remaining_calories_l2729_272947


namespace crayon_selection_ways_l2729_272971

def total_crayons : ℕ := 15
def red_crayons : ℕ := 4
def selection_size : ℕ := 5

theorem crayon_selection_ways : 
  (Nat.choose total_crayons selection_size) -
  (Nat.choose red_crayons 2 * Nat.choose (total_crayons - red_crayons) (selection_size - 2)) +
  (Nat.choose red_crayons 1 * Nat.choose (total_crayons - red_crayons) (selection_size - 1)) +
  (Nat.choose (total_crayons - red_crayons) selection_size) = 1782 :=
by sorry

end crayon_selection_ways_l2729_272971


namespace intersection_of_M_and_N_l2729_272992

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N : M ∩ N = {0, 2} := by sorry

end intersection_of_M_and_N_l2729_272992


namespace willow_playing_time_l2729_272998

/-- Calculates the total playing time in hours given the time spent on football and basketball in minutes -/
def total_playing_time (football_minutes : ℕ) (basketball_minutes : ℕ) : ℚ :=
  (football_minutes + basketball_minutes : ℚ) / 60

/-- Proves that given Willow played football for 60 minutes and basketball for 60 minutes, 
    the total time he played is 2 hours -/
theorem willow_playing_time :
  total_playing_time 60 60 = 2 := by sorry

end willow_playing_time_l2729_272998


namespace x_value_l2729_272901

theorem x_value (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((5 * x) / 7) = x) : x = 5 / 7 := by
  sorry

end x_value_l2729_272901


namespace sine_power_five_decomposition_l2729_272952

theorem sine_power_five_decomposition (b₁ b₂ b₃ b₄ b₅ : ℝ) : 
  (∀ θ : ℝ, Real.sin θ ^ 5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + 
    b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 = 101 / 256 := by
sorry

end sine_power_five_decomposition_l2729_272952


namespace intersecting_lines_l2729_272934

theorem intersecting_lines (k : ℚ) : 
  (∃! p : ℚ × ℚ, 
    p.1 + k * p.2 = 0 ∧ 
    2 * p.1 + 3 * p.2 + 8 = 0 ∧ 
    p.1 - p.2 - 1 = 0) → 
  k = -1/2 := by
sorry

end intersecting_lines_l2729_272934


namespace z_value_theorem_l2729_272919

theorem z_value_theorem (x y z k : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ k ≠ 0) 
  (eq : 1/x - 1/y = k * 1/z) : z = x*y / (k*(y-x)) := by
  sorry

end z_value_theorem_l2729_272919


namespace nina_weekend_sales_l2729_272954

/-- Calculates the total amount Nina made over the weekend from jewelry sales --/
def total_sales (necklace_price bracelet_price earring_price ensemble_price : ℚ)
  (necklaces_sold bracelets_sold earrings_sold ensembles_sold : ℕ)
  (necklace_discount bracelet_discount ensemble_discount tax_rate : ℚ)
  (necklace_custom_fee bracelet_custom_fee : ℚ)
  (necklace_customs bracelet_customs : ℕ) : ℚ :=
  let necklace_total := necklace_price * necklaces_sold * (1 - necklace_discount) + necklace_custom_fee * necklace_customs
  let bracelet_total := bracelet_price * bracelets_sold * (1 - bracelet_discount) + bracelet_custom_fee * bracelet_customs
  let earring_total := earring_price * earrings_sold
  let ensemble_total := ensemble_price * ensembles_sold * (1 - ensemble_discount)
  let subtotal := necklace_total + bracelet_total + earring_total + ensemble_total
  subtotal * (1 + tax_rate)

/-- The total amount Nina made over the weekend is $585.90 --/
theorem nina_weekend_sales :
  total_sales 25 15 10 45 5 10 20 2 (1/10) (1/20) (3/20) (2/25) 5 3 1 2 = 58590/100 := by
  sorry

end nina_weekend_sales_l2729_272954


namespace two_digit_integer_problem_l2729_272913

theorem two_digit_integer_problem (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧ 
  10 ≤ n ∧ n < 100 ∧ 
  m ≠ n ∧
  (m + n) / 2 = (m : ℚ) + n / 100 →
  min m n = 32 := by
sorry

end two_digit_integer_problem_l2729_272913


namespace unique_function_l2729_272964

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x + 1/x else 0

theorem unique_function (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x : ℝ, f (2*x) = a * f x + b * x) ∧
  (∀ x y : ℝ, y ≠ 0 → f x * f y = f (x*y) + f (x/y)) ∧
  (∀ g : ℝ → ℝ, 
    ((∀ x : ℝ, g (2*x) = a * g x + b * x) ∧
     (∀ x y : ℝ, y ≠ 0 → g x * g y = g (x*y) + g (x/y)))
    → g = f) :=
by sorry

end unique_function_l2729_272964


namespace grid_coloring_count_l2729_272967

/-- Represents the number of valid colorings for a 2 × n grid -/
def num_colorings (n : ℕ) : ℕ :=
  3^(n-1)

/-- Theorem stating the number of distinct colorings for the grid -/
theorem grid_coloring_count (n : ℕ) (h : n ≥ 2) :
  let grid_size := 2 * n
  let colored_endpoints := 3
  let vertices_to_color := grid_size - colored_endpoints
  let num_colors := 3
  num_colorings n = num_colors^(n-1) :=
by sorry

end grid_coloring_count_l2729_272967


namespace geometric_sequence_and_sum_l2729_272944

/-- Represents the sum of the first n terms in a geometric sequence -/
def S (n : ℕ) : ℚ := sorry

/-- Represents the nth term of the geometric sequence -/
def a (n : ℕ) : ℚ := sorry

/-- Represents the nth term of the sequence b_n -/
def b (n : ℕ) : ℚ := 1 / (a n) + n

/-- Represents the sum of the first n terms of the sequence b_n -/
def T (n : ℕ) : ℚ := sorry

theorem geometric_sequence_and_sum :
  (S 3 = 7/2) → (S 6 = 63/16) →
  (∀ n, a n = (1/2)^(n-2)) ∧
  (∀ n, T n = (2^n + n^2 + n - 1) / 2) := by sorry

end geometric_sequence_and_sum_l2729_272944


namespace min_blocks_for_wall_l2729_272906

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  length : ℚ
  height : ℕ

/-- Calculates the number of blocks needed for a wall --/
def calculateBlocksNeeded (wall : WallDimensions) (block1 : BlockDimensions) (block2 : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the minimum number of blocks needed for the specified wall --/
theorem min_blocks_for_wall :
  let wall := WallDimensions.mk 120 8
  let block1 := BlockDimensions.mk 1 1
  let block2 := BlockDimensions.mk (3/2) 1
  calculateBlocksNeeded wall block1 block2 = 648 :=
sorry

end min_blocks_for_wall_l2729_272906


namespace find_number_l2729_272914

theorem find_number : ∃ x : ℕ, x * 9999 = 724777430 ∧ x = 72483 := by
  sorry

end find_number_l2729_272914


namespace right_angled_triangle_l2729_272907

theorem right_angled_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a * Real.cos B + b * Real.cos A = c * Real.sin A →
  a * Real.sin C = b * Real.sin B →
  b * Real.sin C = c * Real.sin A →
  c * Real.sin B = a * Real.sin C →
  A = π / 2 := by sorry

end right_angled_triangle_l2729_272907


namespace alyssa_gave_away_seven_puppies_l2729_272904

/-- The number of puppies Alyssa gave to her friends -/
def puppies_given_away (initial : ℕ) (current : ℕ) : ℕ :=
  initial - current

/-- Theorem stating that Alyssa gave away 7 puppies -/
theorem alyssa_gave_away_seven_puppies :
  puppies_given_away 12 5 = 7 := by
  sorry

end alyssa_gave_away_seven_puppies_l2729_272904


namespace game_configurations_l2729_272908

/-- The number of rows in the grid -/
def m : ℕ := 5

/-- The number of columns in the grid -/
def n : ℕ := 7

/-- The total number of steps needed to reach from bottom-left to top-right -/
def total_steps : ℕ := m + n

/-- The number of unique paths from bottom-left to top-right of an m × n grid -/
def num_paths : ℕ := Nat.choose total_steps n

theorem game_configurations : num_paths = 792 := by sorry

end game_configurations_l2729_272908


namespace quadratic_distinct_roots_range_l2729_272999

theorem quadratic_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x - m = 0 ∧ y^2 - 6*y - m = 0) ↔ m > -9 := by
  sorry

end quadratic_distinct_roots_range_l2729_272999


namespace power_inequality_l2729_272922

theorem power_inequality (n : ℕ) (h : n > 1) : n ^ n > (n + 1) ^ (n - 1) := by
  sorry

end power_inequality_l2729_272922


namespace triangle_side_length_l2729_272956

theorem triangle_side_length (x : ℕ+) : 
  (5 + 15 > x^3 ∧ x^3 + 5 > 15 ∧ x^3 + 15 > 5) ↔ x = 2 := by
  sorry

end triangle_side_length_l2729_272956


namespace goats_and_hens_total_amount_l2729_272987

/-- The total amount spent on goats and hens -/
def total_amount (num_goats num_hens goat_price hen_price : ℕ) : ℕ :=
  num_goats * goat_price + num_hens * hen_price

/-- Theorem: The total amount spent on 5 goats at Rs. 400 each and 10 hens at Rs. 50 each is Rs. 2500 -/
theorem goats_and_hens_total_amount :
  total_amount 5 10 400 50 = 2500 := by
  sorry

end goats_and_hens_total_amount_l2729_272987


namespace smallest_consecutive_primes_sum_after_13_l2729_272942

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def consecutive_primes (p q r s : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
  q = p.succ ∧ r = q.succ ∧ s = r.succ

theorem smallest_consecutive_primes_sum_after_13 :
  ∃ (p q r s : ℕ),
    consecutive_primes p q r s ∧
    p > 13 ∧
    4 ∣ (p + q + r + s) ∧
    (p + q + r + s = 88) ∧
    ∀ (a b c d : ℕ),
      consecutive_primes a b c d → a > 13 → 4 ∣ (a + b + c + d) →
      (a + b + c + d ≥ p + q + r + s) :=
by sorry

end smallest_consecutive_primes_sum_after_13_l2729_272942


namespace coin_game_probability_l2729_272993

def num_players : ℕ := 4
def initial_coins : ℕ := 5
def num_rounds : ℕ := 5
def num_balls : ℕ := 5
def num_green : ℕ := 2
def num_red : ℕ := 1
def num_white : ℕ := 2

def coin_transfer : ℕ := 2

def game_round_probability : ℚ := 1 / 5

theorem coin_game_probability :
  (game_round_probability ^ num_rounds : ℚ) = 1 / 3125 := by
  sorry

end coin_game_probability_l2729_272993


namespace tangent_product_l2729_272935

theorem tangent_product (α β : Real) (h : α + β = 3 * Real.pi / 4) :
  (1 - Real.tan α) * (1 - Real.tan β) = 2 := by
  sorry

end tangent_product_l2729_272935


namespace bob_cannot_win_bob_must_choose_nine_l2729_272915

/-- Represents the possible game numbers -/
inductive GameNumber
| nineteen : GameNumber
| twenty : GameNumber

/-- Represents the possible starting numbers -/
inductive StartNumber
| nine : StartNumber
| ten : StartNumber

/-- Represents a player in the game -/
inductive Player
| alice : Player
| bob : Player

/-- Represents the state of the game after each turn -/
structure GameState where
  current_sum : ℕ
  current_player : Player

/-- Represents the outcome of the game -/
inductive GameOutcome
| alice_wins : GameOutcome
| bob_wins : GameOutcome
| draw : GameOutcome

/-- Simulates a single turn of the game -/
def play_turn (state : GameState) (alice_number : GameNumber) (bob_number : GameNumber) : GameState :=
  sorry

/-- Simulates the entire game until completion -/
def play_game (start : StartNumber) (alice_number : GameNumber) (bob_number : GameNumber) : GameOutcome :=
  sorry

/-- Theorem stating that Bob cannot win -/
theorem bob_cannot_win :
  ∀ (start : StartNumber) (alice_number bob_number : GameNumber),
    play_game start alice_number bob_number ≠ GameOutcome.bob_wins :=
  sorry

/-- Theorem stating that Bob must choose 9 to prevent Alice from winning -/
theorem bob_must_choose_nine :
  (∀ (alice_number bob_number : GameNumber),
    play_game StartNumber.nine alice_number bob_number ≠ GameOutcome.alice_wins) ∧
  (∃ (alice_number bob_number : GameNumber),
    play_game StartNumber.ten alice_number bob_number = GameOutcome.alice_wins) :=
  sorry

end bob_cannot_win_bob_must_choose_nine_l2729_272915


namespace circle_intersection_condition_tangent_length_l2729_272900

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O1 (x y r : ℝ) : Prop := (x-2)^2 + y^2 = r^2
def line (m x y : ℝ) : Prop := m*x + y - m - 1 = 0

-- Statement 1
theorem circle_intersection_condition (r : ℝ) (h : r > 0) :
  (∀ m : ℝ, ∃ x1 y1 x2 y2 : ℝ, 
    x1 ≠ x2 ∧ 
    circle_O1 x1 y1 r ∧ line m x1 y1 ∧
    circle_O1 x2 y2 r ∧ line m x2 y2) ↔ 
  r > Real.sqrt 2 :=
sorry

-- Statement 2
theorem tangent_length (A B : ℝ × ℝ) :
  (∃ t : ℝ, circle_O (A.1) (A.2) ∧ circle_O (B.1) (B.2) ∧
    (∀ x y : ℝ, circle_O x y → (x - 0)*(A.2 - 2) = (y - 2)*(A.1 - 0) ∧
                               (x - 0)*(B.2 - 2) = (y - 2)*(B.1 - 0))) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 :=
sorry

end circle_intersection_condition_tangent_length_l2729_272900


namespace mikes_work_hours_l2729_272923

/-- Given that Mike worked for a total of 15 hours over 5 days, 
    prove that he worked 3 hours each day. -/
theorem mikes_work_hours (total_hours : ℕ) (total_days : ℕ) 
  (h1 : total_hours = 15) (h2 : total_days = 5) :
  total_hours / total_days = 3 := by
  sorry

end mikes_work_hours_l2729_272923


namespace august_tips_multiple_l2729_272946

theorem august_tips_multiple (total_months : Nat) (august_ratio : Real) 
  (h1 : total_months = 7)
  (h2 : august_ratio = 0.4) :
  let other_months := total_months - 1
  let august_tips := august_ratio * total_months
  august_tips / other_months = 2.8 := by
  sorry

end august_tips_multiple_l2729_272946


namespace wendi_chicken_count_l2729_272941

def chicken_count (initial : ℕ) : ℕ :=
  let step1 := initial + (initial / 2) ^ 2
  let step2 := step1 - 3
  let step3 := step2 + ((4 * step2 - 28) / 7)
  step3 + 3

theorem wendi_chicken_count : chicken_count 15 = 105 := by
  sorry

end wendi_chicken_count_l2729_272941


namespace min_value_fraction_l2729_272990

theorem min_value_fraction (x : ℝ) (h : x > -2) :
  (x^2 + 6*x + 9) / (x + 2) ≥ 4 ∧ ∃ y > -2, (y^2 + 6*y + 9) / (y + 2) = 4 :=
by sorry

end min_value_fraction_l2729_272990


namespace remaining_crops_l2729_272948

/-- Calculates the total number of remaining crops for a farmer after pest damage --/
theorem remaining_crops (corn_per_row potato_per_row wheat_per_row : ℕ)
  (corn_destroyed potato_destroyed wheat_destroyed : ℚ)
  (corn_rows potato_rows wheat_rows : ℕ)
  (h_corn : corn_per_row = 12)
  (h_potato : potato_per_row = 40)
  (h_wheat : wheat_per_row = 60)
  (h_corn_dest : corn_destroyed = 30 / 100)
  (h_potato_dest : potato_destroyed = 40 / 100)
  (h_wheat_dest : wheat_destroyed = 25 / 100)
  (h_corn_rows : corn_rows = 25)
  (h_potato_rows : potato_rows = 15)
  (h_wheat_rows : wheat_rows = 20) :
  (corn_rows * corn_per_row * (1 - corn_destroyed)).floor +
  (potato_rows * potato_per_row * (1 - potato_destroyed)).floor +
  (wheat_rows * wheat_per_row * (1 - wheat_destroyed)).floor = 1470 := by
  sorry

end remaining_crops_l2729_272948


namespace cups_sold_calculation_l2729_272986

def lemon_cost : ℕ := 10
def sugar_cost : ℕ := 5
def cup_cost : ℕ := 3
def price_per_cup : ℕ := 4
def profit : ℕ := 66

theorem cups_sold_calculation : 
  ∃ (cups : ℕ), 
    cups * price_per_cup = (lemon_cost + sugar_cost + cup_cost + profit) ∧ 
    cups = 21 := by
  sorry

end cups_sold_calculation_l2729_272986


namespace volunteer_selection_l2729_272968

theorem volunteer_selection (boys girls : ℕ) (positions : ℕ) : 
  boys = 4 → girls = 3 → positions = 4 → 
  (Nat.choose (boys + girls) positions) - (Nat.choose boys positions) = 34 :=
by sorry

end volunteer_selection_l2729_272968


namespace product_of_exponents_l2729_272936

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^5 = 270 →
  2^r + 46 = 78 →
  6^s + 5^4 = 1921 →
  p * r * s = 60 := by
  sorry

end product_of_exponents_l2729_272936


namespace product_of_12_and_3460_l2729_272905

theorem product_of_12_and_3460 : ∃ x : ℕ, x * 12 = x * 240 → 12 * 3460 = 41520 := by
  sorry

end product_of_12_and_3460_l2729_272905
