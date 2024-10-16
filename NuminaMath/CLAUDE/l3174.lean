import Mathlib

namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_hexagonal_pyramid_l3174_317446

/-- The radius of a sphere circumscribed around a regular hexagonal pyramid -/
theorem circumscribed_sphere_radius_hexagonal_pyramid 
  (a b : ℝ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < b) 
  (h₃ : a < b) : 
  ∃ R : ℝ, R = b^2 / (2 * Real.sqrt (b^2 - a^2)) ∧ 
  R > 0 ∧
  R * 2 * Real.sqrt (b^2 - a^2) = b^2 :=
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_hexagonal_pyramid_l3174_317446


namespace NUMINAMATH_CALUDE_power_of_five_mod_ten_thousand_l3174_317410

theorem power_of_five_mod_ten_thousand :
  5^2023 ≡ 8125 [ZMOD 10000] := by sorry

end NUMINAMATH_CALUDE_power_of_five_mod_ten_thousand_l3174_317410


namespace NUMINAMATH_CALUDE_nathaniel_best_friends_l3174_317448

/-- Given that Nathaniel has 37 tickets initially, gives 5 tickets to each best friend,
    and ends up with 2 tickets, prove that he has 7 best friends. -/
theorem nathaniel_best_friends :
  let initial_tickets : ℕ := 37
  let tickets_per_friend : ℕ := 5
  let remaining_tickets : ℕ := 2
  let best_friends : ℕ := (initial_tickets - remaining_tickets) / tickets_per_friend
  best_friends = 7 := by
sorry


end NUMINAMATH_CALUDE_nathaniel_best_friends_l3174_317448


namespace NUMINAMATH_CALUDE_proportional_relationship_l3174_317426

/-- Given that y-2 is directly proportional to x-3, and when x=4, y=8,
    prove the functional relationship and a specific point. -/
theorem proportional_relationship (k : ℝ) :
  (∀ x y : ℝ, y - 2 = k * (x - 3)) →  -- Condition 1
  (8 - 2 = k * (4 - 3)) →             -- Condition 2
  (∀ x y : ℝ, y = 6 * x - 16) ∧       -- Conclusion 1
  (-6 = 6 * (5/3) - 16) :=            -- Conclusion 2
by sorry

end NUMINAMATH_CALUDE_proportional_relationship_l3174_317426


namespace NUMINAMATH_CALUDE_diamond_digit_equality_l3174_317457

theorem diamond_digit_equality (diamond : ℕ) : 
  diamond < 10 →  -- diamond is a digit
  (9 * diamond + 6 = 10 * diamond + 3) →  -- diamond6₉ = diamond3₁₀
  diamond = 3 :=
by sorry

end NUMINAMATH_CALUDE_diamond_digit_equality_l3174_317457


namespace NUMINAMATH_CALUDE_existence_of_small_power_l3174_317430

theorem existence_of_small_power (p e : ℝ) (h1 : 0 < p) (h2 : p < 1) (h3 : e > 0) :
  ∃ n : ℕ, (1 - p) ^ n < e := by
sorry

end NUMINAMATH_CALUDE_existence_of_small_power_l3174_317430


namespace NUMINAMATH_CALUDE_fencing_cost_per_metre_l3174_317481

-- Define the ratio of the sides
def ratio_length : ℚ := 3
def ratio_width : ℚ := 4

-- Define the area of the field
def area : ℚ := 9408

-- Define the total cost of fencing
def total_cost : ℚ := 98

-- Statement to prove
theorem fencing_cost_per_metre :
  let length := (ratio_length * Real.sqrt (area / (ratio_length * ratio_width)))
  let width := (ratio_width * Real.sqrt (area / (ratio_length * ratio_width)))
  let perimeter := 2 * (length + width)
  total_cost / perimeter = 0.25 := by sorry

end NUMINAMATH_CALUDE_fencing_cost_per_metre_l3174_317481


namespace NUMINAMATH_CALUDE_test_questions_count_l3174_317428

/-- Calculates the total number of questions on a test given the time spent answering,
    time per question, and number of unanswered questions. -/
def totalQuestions (hoursSpent : ℕ) (minutesPerQuestion : ℕ) (unansweredQuestions : ℕ) : ℕ :=
  (hoursSpent * 60 / minutesPerQuestion) + unansweredQuestions

/-- Proves that the total number of questions on the test is 100 -/
theorem test_questions_count :
  totalQuestions 2 2 40 = 100 := by sorry

end NUMINAMATH_CALUDE_test_questions_count_l3174_317428


namespace NUMINAMATH_CALUDE_displacement_increment_l3174_317458

/-- Given an object with equation of motion s = 2t^2, 
    prove that the increment of displacement from time t = 2 to t = 2 + d 
    is equal to 8d + 2d^2 -/
theorem displacement_increment (d : ℝ) : 
  let s (t : ℝ) := 2 * t^2
  (s (2 + d) - s 2) = 8*d + 2*d^2 := by
sorry

end NUMINAMATH_CALUDE_displacement_increment_l3174_317458


namespace NUMINAMATH_CALUDE_min_sum_inequality_l3174_317425

theorem min_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 3 / Real.rpow 162 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_inequality_l3174_317425


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l3174_317482

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2015 = -i := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l3174_317482


namespace NUMINAMATH_CALUDE_total_ground_beef_weight_l3174_317463

theorem total_ground_beef_weight (package_weight : ℕ) (butcher1_packages : ℕ) (butcher2_packages : ℕ) (butcher3_packages : ℕ) 
  (h1 : package_weight = 4)
  (h2 : butcher1_packages = 10)
  (h3 : butcher2_packages = 7)
  (h4 : butcher3_packages = 8) :
  package_weight * (butcher1_packages + butcher2_packages + butcher3_packages) = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_ground_beef_weight_l3174_317463


namespace NUMINAMATH_CALUDE_shelves_used_l3174_317469

def initial_stock : ℕ := 5
def new_shipment : ℕ := 7
def bears_per_shelf : ℕ := 6

theorem shelves_used (initial_stock new_shipment bears_per_shelf : ℕ) :
  initial_stock = 5 →
  new_shipment = 7 →
  bears_per_shelf = 6 →
  (initial_stock + new_shipment) / bears_per_shelf = 2 := by
sorry

end NUMINAMATH_CALUDE_shelves_used_l3174_317469


namespace NUMINAMATH_CALUDE_evaluate_expression_l3174_317421

theorem evaluate_expression (b : ℝ) (h : b = 2) : (6*b^2 - 15*b + 7)*(3*b - 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3174_317421


namespace NUMINAMATH_CALUDE_illumination_configurations_count_l3174_317488

/-- The number of different ways to illuminate n traffic lights, each with three possible states. -/
def illumination_configurations (n : ℕ) : ℕ := 3^n

/-- Theorem stating that the number of different ways to illuminate n traffic lights,
    each with three possible states, is 3^n. -/
theorem illumination_configurations_count (n : ℕ) :
  illumination_configurations n = 3^n :=
by sorry

end NUMINAMATH_CALUDE_illumination_configurations_count_l3174_317488


namespace NUMINAMATH_CALUDE_inequality_solution_l3174_317439

theorem inequality_solution (y : ℝ) : 
  (y^2 + 2*y^3 - 3*y^4) / (y + 2*y^2 - 3*y^3) ≥ -1 ↔ 
  (y ∈ Set.Icc (-1) (-1/3) ∪ Set.Ioo (-1/3) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1) ∧ 
  (y ≠ -1/3) ∧ (y ≠ 0) ∧ (y ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3174_317439


namespace NUMINAMATH_CALUDE_min_rings_to_connect_five_links_l3174_317493

/-- Represents a chain link with a specific number of rings -/
structure ChainLink where
  rings : ℕ

/-- Represents a collection of chain links -/
structure ChainCollection where
  links : List ChainLink

/-- Function to calculate the minimum number of rings to separate and reattach -/
def minRingsToConnect (chain : ChainCollection) : ℕ :=
  sorry

/-- Theorem stating the minimum number of rings to separate and reattach for the given problem -/
theorem min_rings_to_connect_five_links :
  let chain := ChainCollection.mk (List.replicate 5 (ChainLink.mk 3))
  minRingsToConnect chain = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_rings_to_connect_five_links_l3174_317493


namespace NUMINAMATH_CALUDE_lcm_equality_pairs_l3174_317405

theorem lcm_equality_pairs (m n : ℕ) : 
  Nat.lcm m n = 3 * m + 2 * n + 1 ↔ (m = 3 ∧ n = 10) ∨ (m = 9 ∧ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_lcm_equality_pairs_l3174_317405


namespace NUMINAMATH_CALUDE_intersection_point_of_three_lines_l3174_317453

theorem intersection_point_of_three_lines (k b : ℝ) :
  (∀ x y : ℝ, (y = k * x + b) ∧ (y = 2 * k * x + 2 * b) ∧ (y = b * x + k)) →
  (k ≠ b) →
  (∃! p : ℝ × ℝ, 
    (p.2 = k * p.1 + b) ∧ 
    (p.2 = 2 * k * p.1 + 2 * b) ∧ 
    (p.2 = b * p.1 + k) ∧
    p = (1, 0)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_three_lines_l3174_317453


namespace NUMINAMATH_CALUDE_quadratic_point_range_l3174_317490

/-- Given a quadratic function y = ax² + 4ax + c with a ≠ 0, and points A, B, C on its graph,
    prove that m < -3 under certain conditions. -/
theorem quadratic_point_range (a c m y₁ y₂ x₀ y₀ : ℝ) : 
  a ≠ 0 →
  y₁ = a * m^2 + 4 * a * m + c →
  y₂ = a * (m + 2)^2 + 4 * a * (m + 2) + c →
  y₀ = a * x₀^2 + 4 * a * x₀ + c →
  x₀ = -2 →
  y₀ ≥ y₂ →
  y₂ > y₁ →
  m < -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_point_range_l3174_317490


namespace NUMINAMATH_CALUDE_inequality_solution_l3174_317472

def solution_set (a : ℝ) : Set ℝ :=
  if 0 < a ∧ a < 2 then {x | 1 < x ∧ x ≤ 2/a}
  else if a = 2 then ∅
  else if a > 2 then {x | 2/a ≤ x ∧ x < 1}
  else ∅

theorem inequality_solution (a : ℝ) (h : a > 0) :
  {x : ℝ | (a + 2) * x - 4 ≤ 2 * (x - 1)} = solution_set a := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3174_317472


namespace NUMINAMATH_CALUDE_population_increase_rate_example_l3174_317484

def population_increase_rate (initial_population final_population : ℕ) : ℚ :=
  (final_population - initial_population : ℚ) / initial_population * 100

theorem population_increase_rate_example :
  population_increase_rate 300 330 = 10 := by
sorry

end NUMINAMATH_CALUDE_population_increase_rate_example_l3174_317484


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l3174_317400

theorem greatest_integer_radius (A : ℝ) (h : A < 60 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi ≤ A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ r ∧ r = 7 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l3174_317400


namespace NUMINAMATH_CALUDE_martha_total_savings_l3174_317476

/-- Represents Martha's savings plan for a month --/
structure SavingsPlan where
  daily_allowance : ℝ
  week1_savings : List ℝ
  week2_savings : List ℝ
  week3_savings : List ℝ
  week4_savings : List ℝ
  week1_expense : ℝ
  week2_expense : ℝ
  week3_expense : ℝ
  week4_expense : ℝ

/-- Calculates the total savings for a given week --/
def weekly_savings (savings : List ℝ) (expense : ℝ) : ℝ :=
  savings.sum - expense

/-- Calculates the total savings for the month --/
def total_monthly_savings (plan : SavingsPlan) : ℝ :=
  weekly_savings plan.week1_savings plan.week1_expense +
  weekly_savings plan.week2_savings plan.week2_expense +
  weekly_savings plan.week3_savings plan.week3_expense +
  weekly_savings plan.week4_savings plan.week4_expense

/-- Martha's specific savings plan --/
def martha_plan : SavingsPlan :=
  { daily_allowance := 15
  , week1_savings := [6, 6, 6, 6, 6, 6, 4.5]
  , week2_savings := [7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 6]
  , week3_savings := [9, 9, 9, 9, 7.5, 9, 9]
  , week4_savings := [10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 9]
  , week1_expense := 20
  , week2_expense := 30
  , week3_expense := 40
  , week4_expense := 50
  }

/-- Theorem stating that Martha's total savings at the end of the month is $106 --/
theorem martha_total_savings :
  total_monthly_savings martha_plan = 106 := by
  sorry

end NUMINAMATH_CALUDE_martha_total_savings_l3174_317476


namespace NUMINAMATH_CALUDE_today_is_wednesday_l3174_317415

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the number of days from Sunday -/
def daysFromSunday (d : DayOfWeek) : Nat :=
  match d with
  | .Sunday => 0
  | .Monday => 1
  | .Tuesday => 2
  | .Wednesday => 3
  | .Thursday => 4
  | .Friday => 5
  | .Saturday => 6

/-- Adds a number of days to a given day, wrapping around the week -/
def addDays (d : DayOfWeek) (n : Int) : DayOfWeek :=
  match (daysFromSunday d + n % 7 + 7) % 7 with
  | 0 => .Sunday
  | 1 => .Monday
  | 2 => .Tuesday
  | 3 => .Wednesday
  | 4 => .Thursday
  | 5 => .Friday
  | _ => .Saturday

/-- The condition given in the problem -/
def satisfiesCondition (today : DayOfWeek) : Prop :=
  let dayAfterTomorrow := addDays today 2
  let yesterday := addDays today (-1)
  let tomorrow := addDays today 1
  daysFromSunday (addDays dayAfterTomorrow 3) = daysFromSunday (addDays yesterday 2)

/-- The theorem to be proved -/
theorem today_is_wednesday : 
  ∃ (d : DayOfWeek), satisfiesCondition d ∧ d = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_today_is_wednesday_l3174_317415


namespace NUMINAMATH_CALUDE_blue_pencil_length_l3174_317450

/-- Given a pencil with a total length of 6 cm, a purple part of 3 cm, and a black part of 2 cm,
    prove that the length of the blue part is 1 cm. -/
theorem blue_pencil_length (total : ℝ) (purple : ℝ) (black : ℝ) (blue : ℝ)
    (h_total : total = 6)
    (h_purple : purple = 3)
    (h_black : black = 2)
    (h_sum : total = purple + black + blue) :
    blue = 1 := by
  sorry

end NUMINAMATH_CALUDE_blue_pencil_length_l3174_317450


namespace NUMINAMATH_CALUDE_twice_x_minus_three_l3174_317444

theorem twice_x_minus_three (x : ℝ) : 2 * x - 3 = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_twice_x_minus_three_l3174_317444


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3174_317416

theorem regular_polygon_interior_angle_sum :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 2 →
    exterior_angle = 24 →
    n * exterior_angle = 360 →
    (n - 2) * 180 = 2340 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3174_317416


namespace NUMINAMATH_CALUDE_total_interest_calculation_l3174_317442

/-- Calculates the total interest for a loan split into two parts with different interest rates -/
theorem total_interest_calculation 
  (A B : ℝ) 
  (h1 : A > 0) 
  (h2 : B > 0) 
  (h3 : A + B = 10000) : 
  ∃ I : ℝ, I = 0.08 * A + 0.1 * B := by
  sorry

#check total_interest_calculation

end NUMINAMATH_CALUDE_total_interest_calculation_l3174_317442


namespace NUMINAMATH_CALUDE_well_depth_proof_l3174_317479

/-- The depth of the well in feet -/
def depth : ℝ := 918.09

/-- The total time from dropping the stone to hearing it hit the bottom, in seconds -/
def total_time : ℝ := 8.5

/-- The speed of sound in feet per second -/
def sound_speed : ℝ := 1100

/-- The function describing the distance fallen by the stone after t seconds -/
def stone_fall (t : ℝ) : ℝ := 16 * t^2

theorem well_depth_proof :
  ∃ (t_fall : ℝ), 
    t_fall > 0 ∧
    stone_fall t_fall = depth ∧
    t_fall + depth / sound_speed = total_time :=
sorry

end NUMINAMATH_CALUDE_well_depth_proof_l3174_317479


namespace NUMINAMATH_CALUDE_abs_equal_abs_neg_l3174_317483

theorem abs_equal_abs_neg (x : ℝ) : |x| = |-x| := by sorry

end NUMINAMATH_CALUDE_abs_equal_abs_neg_l3174_317483


namespace NUMINAMATH_CALUDE_cassette_price_proof_l3174_317445

def total_money : ℕ := 37
def cd_price : ℕ := 14

theorem cassette_price_proof :
  ∃ (cassette_price : ℕ),
    2 * cd_price + cassette_price = total_money ∧
    cd_price + 2 * cassette_price = total_money - 5 ∧
    cassette_price = 9 := by
  sorry

end NUMINAMATH_CALUDE_cassette_price_proof_l3174_317445


namespace NUMINAMATH_CALUDE_troll_ratio_l3174_317447

/-- The number of trolls hiding by the path in the forest -/
def trolls_by_path : ℕ := 6

/-- The total number of trolls counted -/
def total_trolls : ℕ := 33

/-- The number of trolls hiding under the bridge -/
def trolls_under_bridge : ℕ := 18

/-- The number of trolls hiding in the plains -/
def trolls_in_plains : ℕ := trolls_under_bridge / 2

theorem troll_ratio : 
  trolls_by_path + trolls_under_bridge + trolls_in_plains = total_trolls ∧ 
  trolls_under_bridge / trolls_by_path = 3 := by
  sorry

end NUMINAMATH_CALUDE_troll_ratio_l3174_317447


namespace NUMINAMATH_CALUDE_painting_price_change_l3174_317473

theorem painting_price_change (P : ℝ) (h : P > 0) : 
  let first_year_price := 1.30 * P
  let final_price := 1.105 * P
  let second_year_decrease := (first_year_price - final_price) / first_year_price
  second_year_decrease = 0.15 := by sorry

end NUMINAMATH_CALUDE_painting_price_change_l3174_317473


namespace NUMINAMATH_CALUDE_certain_number_problem_l3174_317427

theorem certain_number_problem (x : ℤ) : x + 3 = 226 → 3 * x = 669 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3174_317427


namespace NUMINAMATH_CALUDE_dave_winfield_home_runs_l3174_317465

theorem dave_winfield_home_runs :
  let aaron_hr : ℕ := 755
  let winfield_hr : ℕ := 465
  aaron_hr = 2 * winfield_hr - 175 →
  winfield_hr = 465 :=
by sorry

end NUMINAMATH_CALUDE_dave_winfield_home_runs_l3174_317465


namespace NUMINAMATH_CALUDE_mikaela_paint_containers_l3174_317403

/-- Represents the number of paint containers Mikaela initially bought. -/
def initial_containers : ℕ := 8

/-- Represents the number of walls Mikaela initially planned to paint. -/
def planned_walls : ℕ := 4

/-- Represents the number of containers used for the ceiling. -/
def ceiling_containers : ℕ := 1

/-- Represents the number of containers left over. -/
def leftover_containers : ℕ := 3

/-- Represents the number of walls Mikaela actually painted. -/
def painted_walls : ℕ := 3

theorem mikaela_paint_containers :
  initial_containers = 
    ceiling_containers + leftover_containers + (planned_walls - painted_walls) :=
by sorry

end NUMINAMATH_CALUDE_mikaela_paint_containers_l3174_317403


namespace NUMINAMATH_CALUDE_min_value_E_p_l3174_317424

/-- Given an odd prime p and positive integers x and y, 
    the function E_p(x,y) has a lower bound. -/
theorem min_value_E_p (p : ℕ) (x y : ℕ) 
  (hp : Nat.Prime p ∧ Odd p) (hx : x > 0) (hy : y > 0) : 
  Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 
  Real.sqrt (2 * p) - (Real.sqrt ((p - 1) / 2) + Real.sqrt ((p + 1) / 2)) :=
sorry

end NUMINAMATH_CALUDE_min_value_E_p_l3174_317424


namespace NUMINAMATH_CALUDE_scientific_notation_of_400_million_l3174_317496

theorem scientific_notation_of_400_million :
  (400000000 : ℝ) = 4 * 10^8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_400_million_l3174_317496


namespace NUMINAMATH_CALUDE_value_of_B_l3174_317408

theorem value_of_B : ∃ B : ℚ, (3/4 : ℚ) * (1/8 : ℚ) * B = (1/4 : ℚ) * (1/6 : ℚ) * 48 ∧ B = 64/3 :=
by sorry

end NUMINAMATH_CALUDE_value_of_B_l3174_317408


namespace NUMINAMATH_CALUDE_parallelogram_analogous_to_parallelepiped_l3174_317433

/-- A parallelepiped is a 3D shape with opposite faces parallel -/
structure Parallelepiped :=
  (opposite_faces_parallel : Bool)

/-- A parallelogram is a 2D shape with opposite sides parallel -/
structure Parallelogram :=
  (opposite_sides_parallel : Bool)

/-- An analogy between 3D and 2D shapes -/
def is_analogous (shape3D : Type) (shape2D : Type) : Prop :=
  ∃ (property3D : shape3D → Prop) (property2D : shape2D → Prop),
    ∀ (s3D : shape3D) (s2D : shape2D), property3D s3D ↔ property2D s2D

/-- Theorem: A parallelogram is the most analogous 2D shape to a parallelepiped -/
theorem parallelogram_analogous_to_parallelepiped :
  is_analogous Parallelepiped Parallelogram :=
sorry

end NUMINAMATH_CALUDE_parallelogram_analogous_to_parallelepiped_l3174_317433


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_three_l3174_317492

def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {3, 4}

theorem intersection_implies_m_equals_three (m : ℝ) :
  A m ∩ B = {3} → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_three_l3174_317492


namespace NUMINAMATH_CALUDE_pink_shells_count_l3174_317497

theorem pink_shells_count (total : ℕ) (purple yellow blue orange : ℕ) 
  (h1 : total = 65)
  (h2 : purple = 13)
  (h3 : yellow = 18)
  (h4 : blue = 12)
  (h5 : orange = 14) :
  total - (purple + yellow + blue + orange) = 8 := by
  sorry

end NUMINAMATH_CALUDE_pink_shells_count_l3174_317497


namespace NUMINAMATH_CALUDE_fruit_platter_kiwis_l3174_317462

theorem fruit_platter_kiwis 
  (total : ℕ) 
  (oranges apples bananas kiwis : ℕ) 
  (h_total : oranges + apples + bananas + kiwis = total)
  (h_apples : apples = 3 * oranges)
  (h_bananas : bananas = 4 * apples)
  (h_kiwis : kiwis = 5 * bananas)
  (h_total_value : total = 540) :
  kiwis = 420 := by
  sorry

end NUMINAMATH_CALUDE_fruit_platter_kiwis_l3174_317462


namespace NUMINAMATH_CALUDE_appended_number_theorem_l3174_317429

theorem appended_number_theorem (a x : ℕ) (ha : 0 < a) (hx : x ≤ 9) :
  (10 * a + x - a^2 = (11 - x) * a) ↔ (x = a) := by
sorry

end NUMINAMATH_CALUDE_appended_number_theorem_l3174_317429


namespace NUMINAMATH_CALUDE_price_reduction_equation_l3174_317441

/-- Represents the price reduction percentage -/
def x : ℝ := sorry

/-- The original price of the medicine -/
def original_price : ℝ := 25

/-- The final price of the medicine after two reductions -/
def final_price : ℝ := 16

/-- Theorem stating the relationship between the original price, 
    final price, and the reduction percentage -/
theorem price_reduction_equation : 
  original_price * (1 - x)^2 = final_price := by sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l3174_317441


namespace NUMINAMATH_CALUDE_sqrt_13_between_3_and_4_l3174_317420

theorem sqrt_13_between_3_and_4 (a : ℝ) (h : a = Real.sqrt 13) : 3 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_13_between_3_and_4_l3174_317420


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_inequality_l3174_317434

theorem least_integer_absolute_value_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), |3*y - 4| ≤ 25 → x ≤ y) ∧ |3*x - 4| ≤ 25 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_inequality_l3174_317434


namespace NUMINAMATH_CALUDE_females_soccer_not_basketball_l3174_317464

/-- Represents the number of students in various categories -/
structure SchoolTeams where
  soccer_males : ℕ
  soccer_females : ℕ
  basketball_males : ℕ
  basketball_females : ℕ
  males_in_both : ℕ
  total_students : ℕ

/-- The theorem to be proved -/
theorem females_soccer_not_basketball (teams : SchoolTeams)
  (h1 : teams.soccer_males = 120)
  (h2 : teams.soccer_females = 60)
  (h3 : teams.basketball_males = 100)
  (h4 : teams.basketball_females = 80)
  (h5 : teams.males_in_both = 70)
  (h6 : teams.total_students = 260) :
  teams.soccer_females - (teams.soccer_females + teams.basketball_females - 
    (teams.total_students - (teams.soccer_males + teams.basketball_males - teams.males_in_both))) = 30 := by
  sorry


end NUMINAMATH_CALUDE_females_soccer_not_basketball_l3174_317464


namespace NUMINAMATH_CALUDE_one_intersection_point_l3174_317414

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 6
def line3 (x y : ℝ) : Prop := 6 * x - 9 * y = 12

-- Define a point of intersection
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem one_intersection_point :
  ∃! p : ℝ × ℝ, is_intersection p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_one_intersection_point_l3174_317414


namespace NUMINAMATH_CALUDE_tiffany_sunscreen_cost_l3174_317480

/-- Calculates the total cost of sunscreen for a beach trip -/
def sunscreenCost (reapplyTime hours applicationAmount bottleAmount bottlePrice : ℕ) : ℕ :=
  let applications := hours / reapplyTime
  let totalAmount := applications * applicationAmount
  let bottlesNeeded := (totalAmount + bottleAmount - 1) / bottleAmount  -- Ceiling division
  bottlesNeeded * bottlePrice

/-- Theorem: The total cost of sunscreen for Tiffany's beach trip is $14 -/
theorem tiffany_sunscreen_cost :
  sunscreenCost 2 16 3 12 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_sunscreen_cost_l3174_317480


namespace NUMINAMATH_CALUDE_infinite_complementary_sequences_with_arithmetic_l3174_317475

def is_strictly_increasing (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

def infinite_complementary_sequences (a b : ℕ → ℕ) : Prop :=
  (is_strictly_increasing a) ∧ 
  (is_strictly_increasing b) ∧
  (∀ n : ℕ, ∃ m : ℕ, n = a m ∨ n = b m) ∧
  (∀ n : ℕ, ¬(∃ m k : ℕ, n = a m ∧ n = b k))

def arithmetic_sequence (s : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, s (n + 1) = s n + d

theorem infinite_complementary_sequences_with_arithmetic (a b : ℕ → ℕ) :
  infinite_complementary_sequences a b →
  (∃ d : ℕ, arithmetic_sequence a d) →
  a 16 = 36 →
  (∀ n : ℕ, a n = 2 * n + 4) ∧
  (∀ n : ℕ, b n = if n ≤ 5 then n else 2 * n - 5) :=
sorry

end NUMINAMATH_CALUDE_infinite_complementary_sequences_with_arithmetic_l3174_317475


namespace NUMINAMATH_CALUDE_max_rental_income_l3174_317498

/-- Represents the daily rental income function for a hotel --/
def rental_income (x : ℕ) : ℕ :=
  (100 + 10 * x) * (300 - 10 * x)

/-- Theorem stating the maximum daily rental income and the rent at which it's achieved --/
theorem max_rental_income :
  (∃ x : ℕ, x < 30 ∧ rental_income x = 40000) ∧
  (∀ y : ℕ, y < 30 → rental_income y ≤ 40000) ∧
  (rental_income 10 = 40000) := by
  sorry

#check max_rental_income

end NUMINAMATH_CALUDE_max_rental_income_l3174_317498


namespace NUMINAMATH_CALUDE_largest_last_digit_l3174_317443

def is_valid_digit_string (s : List Nat) : Prop :=
  s.length = 1001 ∧ 
  s.head? = some 3 ∧
  ∀ i, i < 1000 → (s[i]! * 10 + s[i+1]!) % 17 = 0 ∨ (s[i]! * 10 + s[i+1]!) % 23 = 0

theorem largest_last_digit (s : List Nat) (h : is_valid_digit_string s) : 
  s[1000]! ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_largest_last_digit_l3174_317443


namespace NUMINAMATH_CALUDE_grandmothers_current_age_prove_grandmothers_age_l3174_317494

/-- Given Yoojung's current age and her grandmother's future age, calculate the grandmother's current age. -/
theorem grandmothers_current_age (yoojung_current_age : ℕ) (yoojung_future_age : ℕ) (grandmother_future_age : ℕ) : ℕ :=
  grandmother_future_age - (yoojung_future_age - yoojung_current_age)

/-- Prove that given the conditions, the grandmother's current age is 55. -/
theorem prove_grandmothers_age :
  let yoojung_current_age := 5
  let yoojung_future_age := 10
  let grandmother_future_age := 60
  grandmothers_current_age yoojung_current_age yoojung_future_age grandmother_future_age = 55 := by
  sorry

end NUMINAMATH_CALUDE_grandmothers_current_age_prove_grandmothers_age_l3174_317494


namespace NUMINAMATH_CALUDE_arith_seq_ratio_l3174_317495

/-- Two arithmetic sequences and their properties -/
structure ArithSeqPair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  A : ℕ → ℚ  -- Sum of first n terms of a
  B : ℕ → ℚ  -- Sum of first n terms of b
  sum_ratio : ∀ n, A n / B n = (4 * n + 2 : ℚ) / (5 * n - 5 : ℚ)

/-- Main theorem -/
theorem arith_seq_ratio (seq : ArithSeqPair) : 
  (seq.a 5 + seq.a 13) / (seq.b 5 + seq.b 13) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_arith_seq_ratio_l3174_317495


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l3174_317423

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ : ℝ} : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of b for which the given lines are parallel -/
theorem parallel_lines_b_value (b : ℝ) : 
  (∃ c₁ c₂ : ℝ, ∀ x y : ℝ, 3 * y - 3 * b = 9 * x + c₁ ↔ y + 2 = (b + 9) * x + c₂) → 
  b = -6 := by
sorry


end NUMINAMATH_CALUDE_parallel_lines_b_value_l3174_317423


namespace NUMINAMATH_CALUDE_jogger_train_distance_l3174_317437

/-- Proof that a jogger is 240 meters ahead of a train's engine given specific conditions -/
theorem jogger_train_distance (jogger_speed train_speed train_length pass_time : ℝ) 
  (h1 : jogger_speed = 9) -- jogger's speed in km/hr
  (h2 : train_speed = 45) -- train's speed in km/hr
  (h3 : train_length = 120) -- train's length in meters
  (h4 : pass_time = 36) -- time taken for train to pass jogger in seconds
  : (train_speed - jogger_speed) * (5/18) * pass_time - train_length = 240 :=
by
  sorry

#eval (45 - 9) * (5/18) * 36 - 120 -- Should evaluate to 240

end NUMINAMATH_CALUDE_jogger_train_distance_l3174_317437


namespace NUMINAMATH_CALUDE_triangle_half_angle_sine_inequality_l3174_317466

theorem triangle_half_angle_sine_inequality (A B C : Real) 
  (h : A + B + C = π) : 
  Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_half_angle_sine_inequality_l3174_317466


namespace NUMINAMATH_CALUDE_family_size_l3174_317477

theorem family_size (planned_spending : ℝ) (orange_cost : ℝ) (savings_percentage : ℝ) :
  planned_spending = 15 →
  orange_cost = 1.5 →
  savings_percentage = 0.4 →
  (planned_spending * savings_percentage) / orange_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_family_size_l3174_317477


namespace NUMINAMATH_CALUDE_problem_solution_l3174_317491

theorem problem_solution (n : ℝ) : 32 - 16 = n * 4 → (n / 4) + 16 = 17 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3174_317491


namespace NUMINAMATH_CALUDE_maxwell_walking_speed_l3174_317455

/-- Proves that Maxwell's walking speed is 4 km/h given the problem conditions -/
theorem maxwell_walking_speed
  (total_distance : ℝ)
  (brad_speed : ℝ)
  (maxwell_distance : ℝ)
  (h1 : total_distance = 50)
  (h2 : brad_speed = 6)
  (h3 : maxwell_distance = 20)
  : ∃ (maxwell_speed : ℝ), maxwell_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_maxwell_walking_speed_l3174_317455


namespace NUMINAMATH_CALUDE_children_ticket_price_l3174_317404

/-- The price of an adult ticket in dollars -/
def adult_price : ℚ := 8

/-- The total revenue in dollars -/
def total_revenue : ℚ := 236

/-- The total number of tickets sold -/
def total_tickets : ℕ := 34

/-- The number of adult tickets sold -/
def adult_tickets : ℕ := 12

/-- The price of a children's ticket in dollars -/
def children_price : ℚ := (total_revenue - adult_price * adult_tickets) / (total_tickets - adult_tickets)

theorem children_ticket_price :
  children_price = 6.36 := by sorry

end NUMINAMATH_CALUDE_children_ticket_price_l3174_317404


namespace NUMINAMATH_CALUDE_total_fish_caught_l3174_317436

/-- Given 20 fishermen, where 19 caught 400 fish each and the 20th caught 2400,
    prove that the total number of fish caught is 10000. -/
theorem total_fish_caught (total_fishermen : Nat) (fish_per_fisherman : Nat) (fish_last_fisherman : Nat) :
  total_fishermen = 20 →
  fish_per_fisherman = 400 →
  fish_last_fisherman = 2400 →
  (total_fishermen - 1) * fish_per_fisherman + fish_last_fisherman = 10000 :=
by sorry

end NUMINAMATH_CALUDE_total_fish_caught_l3174_317436


namespace NUMINAMATH_CALUDE_impossibleCubePlacement_l3174_317486

/-- A type representing the vertices of a cube --/
inductive CubeVertex
| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8

/-- A function type representing a placement of numbers on the cube vertices --/
def CubePlacement := CubeVertex → Nat

/-- Predicate to check if two vertices are adjacent on a cube --/
def adjacent : CubeVertex → CubeVertex → Prop :=
  sorry

/-- Predicate to check if a number is in the valid range and not divisible by 13 --/
def validNumber (n : Nat) : Prop :=
  1 ≤ n ∧ n ≤ 245 ∧ n % 13 ≠ 0

/-- Predicate to check if two numbers have a common divisor greater than 1 --/
def hasCommonDivisor (a b : Nat) : Prop :=
  ∃ (d : Nat), d > 1 ∧ d ∣ a ∧ d ∣ b

theorem impossibleCubePlacement :
  ¬∃ (p : CubePlacement),
    (∀ v, validNumber (p v)) ∧
    (∀ v1 v2, v1 ≠ v2 → p v1 ≠ p v2) ∧
    (∀ v1 v2, adjacent v1 v2 → hasCommonDivisor (p v1) (p v2)) ∧
    (∀ v1 v2, ¬adjacent v1 v2 → ¬hasCommonDivisor (p v1) (p v2)) :=
by
  sorry


end NUMINAMATH_CALUDE_impossibleCubePlacement_l3174_317486


namespace NUMINAMATH_CALUDE_fraction_multiplication_result_l3174_317460

theorem fraction_multiplication_result : (3 / 4 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * 5100 = 765 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_result_l3174_317460


namespace NUMINAMATH_CALUDE_sequence_general_term_l3174_317432

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 3^n) : 
  ∀ n : ℕ, n ≥ 1 → a n = (3^n - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3174_317432


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l3174_317474

/-- Given a canoe with a speed in still water and a downstream speed, calculate its upstream speed -/
theorem canoe_upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 12.5)
  (h2 : speed_downstream = 16) :
  speed_still - (speed_downstream - speed_still) = 9 := by
  sorry

#check canoe_upstream_speed

end NUMINAMATH_CALUDE_canoe_upstream_speed_l3174_317474


namespace NUMINAMATH_CALUDE_tournament_handshakes_l3174_317440

/-- Represents a women's doubles tennis tournament --/
structure Tournament where
  numTeams : Nat
  playersPerTeam : Nat
  handshakesPerPlayer : Nat

/-- Calculates the total number of handshakes in the tournament --/
def totalHandshakes (t : Tournament) : Nat :=
  (t.numTeams * t.playersPerTeam * t.handshakesPerPlayer) / 2

/-- Theorem stating that the specific tournament configuration results in 24 handshakes --/
theorem tournament_handshakes :
  ∃ (t : Tournament),
    t.numTeams = 4 ∧
    t.playersPerTeam = 2 ∧
    t.handshakesPerPlayer = 6 ∧
    totalHandshakes t = 24 := by
  sorry


end NUMINAMATH_CALUDE_tournament_handshakes_l3174_317440


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l3174_317438

theorem product_of_cubic_fractions :
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 57 / 84 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l3174_317438


namespace NUMINAMATH_CALUDE_mean_calculation_l3174_317417

theorem mean_calculation (x : ℝ) :
  (28 + x + 50 + 78 + 104) / 5 = 62 →
  (48 + 62 + 98 + 124 + x) / 5 = 76.4 := by
sorry

end NUMINAMATH_CALUDE_mean_calculation_l3174_317417


namespace NUMINAMATH_CALUDE_simplify_expression_l3174_317468

theorem simplify_expression (a : ℝ) : (2*a - 3)^2 - (a + 5)*(a - 5) = 3*a^2 - 12*a + 34 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3174_317468


namespace NUMINAMATH_CALUDE_chocolate_ratio_l3174_317499

theorem chocolate_ratio (initial_chocolates : ℕ) (num_sisters : ℕ) (given_to_mother : ℕ) (eaten_by_father : ℕ) (left_with_father : ℕ) :
  initial_chocolates = 20 →
  num_sisters = 4 →
  given_to_mother = 3 →
  eaten_by_father = 2 →
  left_with_father = 5 →
  ∃ (chocolates_per_person : ℕ) (given_to_father : ℕ),
    chocolates_per_person * (num_sisters + 1) = initial_chocolates ∧
    given_to_father = left_with_father + given_to_mother + eaten_by_father ∧
    given_to_father * 2 = initial_chocolates :=
by sorry

end NUMINAMATH_CALUDE_chocolate_ratio_l3174_317499


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l3174_317449

theorem partial_fraction_sum_zero (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l3174_317449


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3174_317418

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, Real.exp x > Real.log x) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ Real.log x₀) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3174_317418


namespace NUMINAMATH_CALUDE_book_survey_difference_l3174_317478

/-- Represents the survey results of students reading books A and B -/
structure BookSurvey where
  total : ℕ
  only_a : ℕ
  only_b : ℕ
  both : ℕ
  h_total : total = only_a + only_b + both
  h_a_both : both = (only_a + both) / 5
  h_b_both : both = (only_b + both) / 4

/-- The difference between students who read only book A and only book B is 75 -/
theorem book_survey_difference (s : BookSurvey) (h_total : s.total = 600) :
  s.only_a - s.only_b = 75 := by
  sorry

end NUMINAMATH_CALUDE_book_survey_difference_l3174_317478


namespace NUMINAMATH_CALUDE_team_organization_theorem_l3174_317409

/-- The number of ways to organize a team of 13 members into a specific hierarchy -/
def team_organization_count : ℕ := 4804800

/-- The total number of team members -/
def total_members : ℕ := 13

/-- The number of project managers -/
def project_managers : ℕ := 3

/-- The number of subordinates per project manager -/
def subordinates_per_manager : ℕ := 3

/-- Theorem stating the correct number of ways to organize the team -/
theorem team_organization_theorem :
  team_organization_count = 
    total_members * 
    (Nat.choose (total_members - 1) project_managers) * 
    (Nat.choose (total_members - 1 - project_managers) subordinates_per_manager) * 
    (Nat.choose (total_members - 1 - project_managers - subordinates_per_manager) subordinates_per_manager) * 
    (Nat.choose (total_members - 1 - project_managers - 2 * subordinates_per_manager) subordinates_per_manager) :=
by
  sorry

#eval team_organization_count

end NUMINAMATH_CALUDE_team_organization_theorem_l3174_317409


namespace NUMINAMATH_CALUDE_zoes_bottles_l3174_317401

/-- Given the initial number of bottles, the number of bottles drunk, and the number of bottles bought,
    calculate the final number of bottles. -/
def finalBottles (initial drunk bought : ℕ) : ℕ :=
  initial - drunk + bought

/-- Prove that for Zoe's specific case, the final number of bottles is 47. -/
theorem zoes_bottles : finalBottles 42 25 30 = 47 := by
  sorry

end NUMINAMATH_CALUDE_zoes_bottles_l3174_317401


namespace NUMINAMATH_CALUDE_jerry_remaining_money_l3174_317461

def mustard_oil_quantity : ℕ := 2
def mustard_oil_price : ℕ := 13
def pasta_quantity : ℕ := 3
def pasta_price : ℕ := 4
def sauce_quantity : ℕ := 1
def sauce_price : ℕ := 5
def total_budget : ℕ := 50

theorem jerry_remaining_money :
  total_budget - (mustard_oil_quantity * mustard_oil_price + 
                  pasta_quantity * pasta_price + 
                  sauce_quantity * sauce_price) = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_remaining_money_l3174_317461


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3174_317454

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with S₃ = 9 and S₆ = 36, a₇ + a₈ + a₉ = 45 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
    (h₃ : seq.S 3 = 9) (h₆ : seq.S 6 = 36) : 
    seq.a 7 + seq.a 8 + seq.a 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3174_317454


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l3174_317459

theorem binomial_coefficient_problem (m : ℕ) :
  (1 : ℚ) / (Nat.choose 5 m) - (1 : ℚ) / (Nat.choose 6 m) = 7 / (10 * Nat.choose 7 m) →
  Nat.choose 8 m = 28 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l3174_317459


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l3174_317487

/-- Given two positive integers with specific LCM and HCF, prove that if one number is 210, the other is 517 -/
theorem lcm_hcf_problem (A B : ℕ+) (h1 : Nat.lcm A B = 2310) (h2 : Nat.gcd A B = 47) (h3 : A = 210) : B = 517 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l3174_317487


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l3174_317470

/-- Given a function f(x) = ax³ + 4x² + 3x, prove that if f'(1) = 2, then a = -3 -/
theorem cubic_function_derivative (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + 4 * x^2 + 3 * x
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 8 * x + 3
  f' 1 = 2 → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l3174_317470


namespace NUMINAMATH_CALUDE_basketball_max_height_l3174_317452

/-- The height function of a basketball -/
def h (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 2

/-- The maximum height reached by the basketball -/
theorem basketball_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 127 :=
sorry

end NUMINAMATH_CALUDE_basketball_max_height_l3174_317452


namespace NUMINAMATH_CALUDE_expression_bounds_l3174_317431

theorem expression_bounds (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  2 * Real.sqrt 2 ≤ 
    Real.sqrt ((a^2)^2 + (b^2 - b^2)^2) + 
    Real.sqrt ((b^2)^2 + (c^2 - b^2)^2) + 
    Real.sqrt ((c^2)^2 + (d^2 - c^2)^2) + 
    Real.sqrt ((d^2)^2 + (a^2 - d^2)^2) ∧
  Real.sqrt ((a^2)^2 + (b^2 - b^2)^2) + 
    Real.sqrt ((b^2)^2 + (c^2 - b^2)^2) + 
    Real.sqrt ((c^2)^2 + (d^2 - c^2)^2) + 
    Real.sqrt ((d^2)^2 + (a^2 - d^2)^2) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l3174_317431


namespace NUMINAMATH_CALUDE_circumscribed_equal_sides_equal_angles_inscribed_equal_sides_not_always_equal_angles_circumscribed_equal_angles_not_always_equal_sides_inscribed_equal_angles_equal_sides_l3174_317485

/- Define the basic structures -/
structure Polygon :=
  (sides : ℕ)
  (isCircumscribed : Bool)
  (hasEqualSides : Bool)
  (hasEqualAngles : Bool)

/- Define the theorems to be proved -/
theorem circumscribed_equal_sides_equal_angles (p : Polygon) :
  p.isCircumscribed ∧ p.hasEqualSides → p.hasEqualAngles :=
sorry

theorem inscribed_equal_sides_not_always_equal_angles :
  ∃ p : Polygon, ¬p.isCircumscribed ∧ p.hasEqualSides ∧ ¬p.hasEqualAngles :=
sorry

theorem circumscribed_equal_angles_not_always_equal_sides :
  ∃ p : Polygon, p.isCircumscribed ∧ p.hasEqualAngles ∧ ¬p.hasEqualSides :=
sorry

theorem inscribed_equal_angles_equal_sides (p : Polygon) :
  ¬p.isCircumscribed ∧ p.hasEqualAngles → p.hasEqualSides :=
sorry

end NUMINAMATH_CALUDE_circumscribed_equal_sides_equal_angles_inscribed_equal_sides_not_always_equal_angles_circumscribed_equal_angles_not_always_equal_sides_inscribed_equal_angles_equal_sides_l3174_317485


namespace NUMINAMATH_CALUDE_chocolate_division_l3174_317471

theorem chocolate_division (total : ℝ) (total_positive : 0 < total) : 
  let al_share := (4 / 10) * total
  let bert_share := (3 / 10) * total
  let carl_share := (2 / 10) * total
  let dana_share := (1 / 10) * total
  al_share + bert_share + carl_share + dana_share = total :=
by sorry

end NUMINAMATH_CALUDE_chocolate_division_l3174_317471


namespace NUMINAMATH_CALUDE_vector_problem_l3174_317412

/-- Given vectors a and b, if |a| = 6 and a ∥ b, then x = 4 and x + y = 8 -/
theorem vector_problem (x y : ℝ) : 
  let a : ℝ × ℝ × ℝ := (2, 4, x)
  let b : ℝ × ℝ × ℝ := (2, y, 2)
  (‖a‖ = 6 ∧ ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → x = 4 ∧ x + y = 8 := by
  sorry


end NUMINAMATH_CALUDE_vector_problem_l3174_317412


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l3174_317406

-- Define a binary number as a list of bits (0 or 1)
def BinaryNumber := List Nat

-- Define a function to convert a binary number to decimal
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + bit * 2^i) 0

-- State the theorem
theorem binary_101_equals_5 :
  binary_to_decimal [1, 0, 1] = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l3174_317406


namespace NUMINAMATH_CALUDE_min_value_x2_plus_y2_l3174_317413

theorem min_value_x2_plus_y2 (x y : ℝ) (h : (x + 1)^2 + y^2 = 1/4) :
  ∃ (min : ℝ), min = 1/4 ∧ ∀ (a b : ℝ), (a + 1)^2 + b^2 = 1/4 → a^2 + b^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_x2_plus_y2_l3174_317413


namespace NUMINAMATH_CALUDE_last_number_is_one_seventh_l3174_317407

/-- A sequence of 100 non-zero real numbers where each number (except the first and last) 
    is the product of its neighbors, and the first number is 7 -/
def SpecialSequence (a : Fin 100 → ℝ) : Prop :=
  a 0 = 7 ∧ 
  (∀ i : Fin 98, a (i + 1) = a i * a (i + 2)) ∧
  (∀ i : Fin 100, a i ≠ 0)

/-- The last number in the sequence is 1/7 -/
theorem last_number_is_one_seventh (a : Fin 100 → ℝ) (h : SpecialSequence a) : 
  a 99 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_last_number_is_one_seventh_l3174_317407


namespace NUMINAMATH_CALUDE_circle_properties_l3174_317435

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + 45 = 0

-- Define the point P
def P : ℝ × ℝ := (9, 1)

-- Theorem statement
theorem circle_properties :
  -- 1. Common chord equation
  (∀ x y : ℝ, C₁ x y ∧ C₂ x y → 4*x + 3*y - 23 = 0) ∧
  -- 2. Length of common chord
  (∃ a b : ℝ, C₁ a b ∧ C₂ a b ∧
    ∃ c d : ℝ, C₁ c d ∧ C₂ c d ∧ (a ≠ c ∨ b ≠ d) ∧
    ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 2 * 7^(1/2 : ℝ)) ∧
  -- 3. Tangent lines
  (∀ x y : ℝ, (x = 9 ∨ 9*x + 40*y - 121 = 0) →
    ((x - P.1)^2 + (y - P.2)^2 = 0 ∨
     ∃ t : ℝ, C₂ (x + t) (y + t * (y - P.2) / (x - P.1)) ∧
              (∀ s : ℝ, s ≠ t → ¬C₂ (x + s) (y + s * (y - P.2) / (x - P.1))))) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3174_317435


namespace NUMINAMATH_CALUDE_history_score_calculation_l3174_317467

theorem history_score_calculation (math_score : ℝ) (third_subject_score : ℝ) (desired_average : ℝ) :
  math_score = 74 →
  third_subject_score = 67 →
  desired_average = 75 →
  (math_score + third_subject_score + (3 * desired_average - math_score - third_subject_score)) / 3 = desired_average :=
by
  sorry

#check history_score_calculation

end NUMINAMATH_CALUDE_history_score_calculation_l3174_317467


namespace NUMINAMATH_CALUDE_expression_evaluation_l3174_317422

theorem expression_evaluation :
  -2^3 + 36 / 3^2 * (-1/2 : ℝ) + |(-5 : ℝ)| = -5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3174_317422


namespace NUMINAMATH_CALUDE_children_group_size_l3174_317419

theorem children_group_size (adults_per_group : ℕ) (total_adults : ℕ) (total_children : ℕ) :
  adults_per_group = 17 →
  total_adults = 255 →
  total_children = total_adults →
  total_adults % adults_per_group = 0 →
  ∃ (children_per_group : ℕ),
    children_per_group > 0 ∧
    total_children % children_per_group = 0 ∧
    total_children / children_per_group = total_adults / adults_per_group ∧
    children_per_group = 17 := by
  sorry

end NUMINAMATH_CALUDE_children_group_size_l3174_317419


namespace NUMINAMATH_CALUDE_honeys_earnings_l3174_317411

/-- Honey's earnings problem -/
theorem honeys_earnings (days : ℕ) (spent : ℕ) (saved : ℕ) (daily_earnings : ℕ) : 
  days = 20 → spent = 1360 → saved = 240 → daily_earnings = 80 → 
  days * daily_earnings = spent + saved :=
by
  sorry

#check honeys_earnings

end NUMINAMATH_CALUDE_honeys_earnings_l3174_317411


namespace NUMINAMATH_CALUDE_oil_after_eight_hours_l3174_317456

/-- Represents the remaining oil in a car's fuel tank as a function of time -/
def remaining_oil (initial_oil : ℝ) (consumption_rate : ℝ) (time : ℝ) : ℝ :=
  initial_oil - consumption_rate * time

theorem oil_after_eight_hours 
  (initial_oil : ℝ) 
  (consumption_rate : ℝ) 
  (h1 : initial_oil = 50) 
  (h2 : consumption_rate = 5) :
  remaining_oil initial_oil consumption_rate 8 = 10 := by
  sorry

#check oil_after_eight_hours

end NUMINAMATH_CALUDE_oil_after_eight_hours_l3174_317456


namespace NUMINAMATH_CALUDE_labourer_income_l3174_317489

/-- Represents the financial situation of a labourer over a 10-month period. -/
structure LabourerFinances where
  monthly_income : ℝ
  first_period_length : ℕ := 6
  second_period_length : ℕ := 4
  first_period_expense : ℝ := 75
  second_period_expense : ℝ := 60
  savings : ℝ := 30

/-- The labourer's finances satisfy the given conditions. -/
def satisfies_conditions (f : LabourerFinances) : Prop :=
  (f.first_period_length * f.monthly_income < f.first_period_length * f.first_period_expense) ∧
  (f.second_period_length * f.monthly_income = 
    f.second_period_length * f.second_period_expense + 
    (f.first_period_length * f.first_period_expense - f.first_period_length * f.monthly_income) + 
    f.savings)

/-- The labourer's monthly income is 72 given the conditions. -/
theorem labourer_income (f : LabourerFinances) (h : satisfies_conditions f) : 
  f.monthly_income = 72 := by
  sorry


end NUMINAMATH_CALUDE_labourer_income_l3174_317489


namespace NUMINAMATH_CALUDE_paycheck_calculation_l3174_317451

def biweekly_gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def tax_deduction : ℝ := 100

theorem paycheck_calculation :
  biweekly_gross_pay * (1 - retirement_rate) - tax_deduction = 740 := by
  sorry

end NUMINAMATH_CALUDE_paycheck_calculation_l3174_317451


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l3174_317402

theorem first_player_winning_strategy (a c : ℤ) : ∃ (x y z : ℤ), 
  x^3 + a*x^2 - x + c = 0 ∧ 
  y^3 + a*y^2 - y + c = 0 ∧ 
  z^3 + a*z^2 - z + c = 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z :=
by sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l3174_317402
