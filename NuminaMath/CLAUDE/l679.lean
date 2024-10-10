import Mathlib

namespace defective_units_shipped_percentage_l679_67938

theorem defective_units_shipped_percentage
  (total_units : ℝ)
  (defective_rate : ℝ)
  (defective_shipped_rate : ℝ)
  (h1 : defective_rate = 0.1)
  (h2 : defective_shipped_rate = 0.005)
  (h3 : total_units > 0) :
  (defective_shipped_rate / defective_rate) * 100 = 5 := by
sorry

end defective_units_shipped_percentage_l679_67938


namespace hyperbola_eccentricity_l679_67923

/-- The eccentricity of the hyperbola x² - 4y² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - 4*y^2 = 1
  ∃ e : ℝ, e = Real.sqrt 5 / 2 ∧ 
    ∀ x y : ℝ, h x y → 
      ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
        x^2 / a^2 - y^2 / b^2 = 1 ∧
        c^2 = a^2 + b^2 ∧
        e = c / a :=
by sorry

end hyperbola_eccentricity_l679_67923


namespace remainder_problem_l679_67921

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1059 % d = r)
  (h3 : 1482 % d = r)
  (h4 : 2340 % d = r) :
  2 * d - r = 6 := by
sorry

end remainder_problem_l679_67921


namespace d_equals_25_l679_67971

theorem d_equals_25 (x : ℝ) (h : x^2 - 2*x - 5 = 0) : 
  x^4 - 2*x^3 + x^2 - 12*x - 5 = 25 := by
  sorry

end d_equals_25_l679_67971


namespace min_sum_squares_l679_67941

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 8 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧ x^2 + y^2 + z^2 ≥ m := by
  sorry

end min_sum_squares_l679_67941


namespace function_satisfying_equation_l679_67977

theorem function_satisfying_equation (f : ℕ → ℕ) :
  (∀ m n : ℕ, f (m * n) + f (m + n) = f m * f n + 1) →
  (∀ n : ℕ, f n = 1 ∨ f n = n + 1) :=
by sorry

end function_satisfying_equation_l679_67977


namespace vacation_emails_l679_67992

theorem vacation_emails (a₁ : ℝ) (r : ℝ) (n : ℕ) (h1 : a₁ = 16) (h2 : r = 1/2) (h3 : n = 4) :
  a₁ * (1 - r^n) / (1 - r) = 30 := by
  sorry

end vacation_emails_l679_67992


namespace max_rental_income_l679_67914

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

end max_rental_income_l679_67914


namespace celias_budget_weeks_l679_67937

/-- Celia's budget problem -/
theorem celias_budget_weeks (weekly_food_budget : ℝ) (rent : ℝ) (streaming : ℝ) (cell_phone : ℝ) 
  (savings_percent : ℝ) (savings_amount : ℝ) :
  weekly_food_budget = 100 →
  rent = 1500 →
  streaming = 30 →
  cell_phone = 50 →
  savings_percent = 0.1 →
  savings_amount = 198 →
  ∃ (weeks : ℕ), 
    savings_amount = savings_percent * (weekly_food_budget * ↑weeks + rent + streaming + cell_phone) ∧
    weeks = 4 := by
  sorry

end celias_budget_weeks_l679_67937


namespace basketball_team_combinations_l679_67978

theorem basketball_team_combinations (n : ℕ) (k : ℕ) (h : n = 12 ∧ k = 6) :
  n * Nat.choose (n - 1) (k - 1) = 5544 :=
sorry

end basketball_team_combinations_l679_67978


namespace point_inside_circle_l679_67942

theorem point_inside_circle (a b : ℝ) : 
  a ≠ b → 
  a^2 - a - Real.sqrt 2 = 0 → 
  b^2 - b - Real.sqrt 2 = 0 → 
  a^2 + b^2 < 8 := by
sorry

end point_inside_circle_l679_67942


namespace total_weight_is_seven_pounds_l679_67927

-- Define the weights of items
def brie_cheese : ℚ := 8 / 16  -- in pounds
def bread : ℚ := 1
def tomatoes : ℚ := 1
def zucchini : ℚ := 2
def chicken_breasts : ℚ := 3 / 2
def raspberries : ℚ := 8 / 16  -- in pounds
def blueberries : ℚ := 8 / 16  -- in pounds

-- Define the conversion factor
def ounces_per_pound : ℚ := 16

-- Theorem statement
theorem total_weight_is_seven_pounds :
  brie_cheese + bread + tomatoes + zucchini + chicken_breasts + raspberries + blueberries = 7 := by
  sorry

end total_weight_is_seven_pounds_l679_67927


namespace minimum_rice_amount_l679_67989

theorem minimum_rice_amount (o r : ℝ) (ho : o ≥ 8 + r / 3) (ho2 : o ≤ 2 * r) :
  ∃ (min_r : ℕ), min_r = 5 ∧ ∀ (r' : ℕ), r' ≥ min_r → ∃ (o' : ℝ), o' ≥ 8 + r' / 3 ∧ o' ≤ 2 * r' :=
sorry

end minimum_rice_amount_l679_67989


namespace marks_garden_flowers_l679_67964

theorem marks_garden_flowers (yellow : ℕ) (purple : ℕ) (green : ℕ) 
  (h1 : yellow = 10)
  (h2 : purple = yellow + yellow * 4 / 5)
  (h3 : green = (yellow + purple) / 4) :
  yellow + purple + green = 35 := by
  sorry

end marks_garden_flowers_l679_67964


namespace tagalong_boxes_per_case_l679_67940

theorem tagalong_boxes_per_case 
  (total_boxes : ℕ) 
  (total_cases : ℕ) 
  (h1 : total_boxes = 36) 
  (h2 : total_cases = 3) 
  (h3 : total_cases > 0) : 
  total_boxes / total_cases = 12 := by
sorry

end tagalong_boxes_per_case_l679_67940


namespace angle_sum_B_plus_D_l679_67936

-- Define the triangle AFG and external angle BFD
structure Triangle :=
  (A B D F G : Real)

-- State the theorem
theorem angle_sum_B_plus_D (t : Triangle) 
  (h1 : t.A = 30) -- Given: Angle A is 30 degrees
  (h2 : t.F = t.G) -- Given: Angle AFG equals Angle AGF
  : t.B + t.D = 75 := by
  sorry


end angle_sum_B_plus_D_l679_67936


namespace min_value_of_expression_equality_achieved_l679_67931

theorem min_value_of_expression (x : ℝ) : (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 ≥ 2094 :=
sorry

theorem equality_achieved : ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 = 2094 :=
sorry

end min_value_of_expression_equality_achieved_l679_67931


namespace rectangle_perimeter_bound_l679_67993

theorem rectangle_perimeter_bound (a b : ℝ) (h : a > 0 ∧ b > 0) (area_gt_perimeter : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 := by
  sorry

end rectangle_perimeter_bound_l679_67993


namespace system_solution_l679_67958

theorem system_solution : 
  ∃! (x y : ℝ), (x + 2 * Real.sqrt y = 2) ∧ (2 * Real.sqrt x + y = 2) ∧ (x = 4 - 2 * Real.sqrt 3) ∧ (y = 4 - 2 * Real.sqrt 3) := by
  sorry

end system_solution_l679_67958


namespace line_proof_l679_67903

-- Define the three given lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := x - 2 * y + 7 = 0

-- Theorem statement
theorem line_proof :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧  -- Intersection point
    (∀ (x y : ℝ), result_line x y → 
      ((y - y₀) = -(1 / (2 : ℝ)) * (x - x₀)) ∧  -- Slope is perpendicular
      (result_line x₀ y₀))  -- Result line passes through intersection
:= by sorry

end line_proof_l679_67903


namespace expression_evaluation_l679_67959

theorem expression_evaluation :
  let x : ℚ := -1/3
  (-5 * x^2 + 4 + x) - 3 * (-2 * x^2 + x - 1) = 70/9 := by
sorry

end expression_evaluation_l679_67959


namespace no_solutions_lcm_gcd_equation_l679_67929

theorem no_solutions_lcm_gcd_equation :
  ¬∃ (n : ℕ), n > 0 ∧ Nat.lcm n 150 = Nat.gcd n 150 + 600 := by
sorry

end no_solutions_lcm_gcd_equation_l679_67929


namespace bakery_storage_ratio_l679_67974

/-- Bakery storage problem -/
theorem bakery_storage_ratio : 
  ∀ (flour baking_soda : ℝ),
  flour / baking_soda = 10 →
  flour / (baking_soda + 60) = 8 →
  ∃ (sugar : ℝ),
  sugar = 6000 ∧
  sugar / flour = 2.5 :=
by sorry

end bakery_storage_ratio_l679_67974


namespace toothbrushes_per_patient_l679_67972

/-- Calculates the number of toothbrushes given to each patient in a dental office -/
theorem toothbrushes_per_patient
  (hours_per_day : ℝ)
  (hours_per_visit : ℝ)
  (days_per_week : ℕ)
  (total_toothbrushes : ℕ)
  (h1 : hours_per_day = 8)
  (h2 : hours_per_visit = 0.5)
  (h3 : days_per_week = 5)
  (h4 : total_toothbrushes = 160) :
  (total_toothbrushes : ℝ) / ((hours_per_day / hours_per_visit) * days_per_week) = 2 := by
  sorry

end toothbrushes_per_patient_l679_67972


namespace problem_solution_l679_67995

theorem problem_solution (y : ℝ) (h : 9 / y^2 = y / 81) : y = 9 := by
  sorry

end problem_solution_l679_67995


namespace james_jello_cost_l679_67970

/-- The cost to fill a bathtub with jello --/
def jello_cost (bathtub_capacity : ℝ) (cubic_foot_to_gallon : ℝ) (gallon_weight : ℝ) 
                (jello_mix_ratio : ℝ) (jello_mix_cost : ℝ) : ℝ :=
  bathtub_capacity * cubic_foot_to_gallon * gallon_weight * jello_mix_ratio * jello_mix_cost

/-- Theorem: The cost to fill James' bathtub with jello is $270 --/
theorem james_jello_cost : 
  jello_cost 6 7.5 8 1.5 0.5 = 270 := by
  sorry

#eval jello_cost 6 7.5 8 1.5 0.5

end james_jello_cost_l679_67970


namespace cement_mixture_weight_l679_67969

/-- Proves that a cement mixture with the given composition weighs 40 pounds -/
theorem cement_mixture_weight :
  ∀ (W : ℝ),
  (1/4 : ℝ) * W + (2/5 : ℝ) * W + 14 = W →
  W = 40 := by
  sorry

end cement_mixture_weight_l679_67969


namespace units_digit_product_l679_67924

theorem units_digit_product (n : ℕ) : 
  (4^150 * 9^151 * 16^152) % 10 = 4 := by sorry

end units_digit_product_l679_67924


namespace circles_tangent_internally_l679_67976

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 17 = 0

-- Define the center and radius of C1
def center_C1 : ℝ × ℝ := (-1, -2)
def radius_C1 : ℝ := 2

-- Define the center and radius of C2
def center_C2 : ℝ × ℝ := (2, -2)
def radius_C2 : ℝ := 5

-- Define the distance between centers
def distance_between_centers : ℝ := 3

-- Theorem stating that the circles are tangent internally
theorem circles_tangent_internally :
  distance_between_centers = abs (radius_C2 - radius_C1) ∧
  distance_between_centers < radius_C1 + radius_C2 :=
sorry

end circles_tangent_internally_l679_67976


namespace parabola_vector_sum_implies_magnitude_sum_l679_67986

noncomputable section

-- Define the parabola
def is_on_parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the vector from focus to a point
def vec_from_focus (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - focus.1, p.2 - focus.2)

-- Define the magnitude of a vector
def vec_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem parabola_vector_sum_implies_magnitude_sum
  (A B C : ℝ × ℝ)
  (hA : is_on_parabola A)
  (hB : is_on_parabola B)
  (hC : is_on_parabola C)
  (h_sum : vec_from_focus A + 2 • vec_from_focus B + 3 • vec_from_focus C = (0, 0)) :
  vec_magnitude (vec_from_focus A) + 2 * vec_magnitude (vec_from_focus B) + 3 * vec_magnitude (vec_from_focus C) = 12 := by
  sorry

end parabola_vector_sum_implies_magnitude_sum_l679_67986


namespace removed_term_is_last_l679_67926

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) : ℕ → ℚ :=
  fun n => a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem removed_term_is_last
  (a₁ : ℚ)
  (avg_11 : ℚ)
  (avg_10 : ℚ)
  (h₁ : a₁ = -5)
  (h₂ : avg_11 = 5)
  (h₃ : avg_10 = 4)
  (h₄ : ∃ d : ℚ, sum_arithmetic_sequence a₁ d 11 = 11 * avg_11) :
  arithmetic_sequence a₁ ((sum_arithmetic_sequence a₁ 2 11 - sum_arithmetic_sequence a₁ 2 10) / 1) 11 =
  sum_arithmetic_sequence a₁ 2 11 - 10 * avg_10 :=
by sorry

end removed_term_is_last_l679_67926


namespace intersection_count_l679_67908

/-- Calculates the number of intersections between two regular polygons inscribed in a circle -/
def intersections (n m : ℕ) : ℕ := 2 * n * m

/-- The set of regular polygons inscribed in the circle -/
def polygons : Finset ℕ := {4, 6, 8, 10}

/-- The set of all pairs of polygons -/
def polygon_pairs : Finset (ℕ × ℕ) := 
  {(4, 6), (4, 8), (4, 10), (6, 8), (6, 10), (8, 10)}

theorem intersection_count :
  (polygon_pairs.sum (fun (p : ℕ × ℕ) => intersections p.1 p.2)) = 568 := by
  sorry

end intersection_count_l679_67908


namespace product_positive_l679_67996

theorem product_positive (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x^4 - y^4 > x) (h2 : y^4 - x^4 > y) : x * y > 0 :=
by sorry

end product_positive_l679_67996


namespace highest_score_is_179_l679_67985

/-- Represents a batsman's statistics --/
structure BatsmanStats where
  totalInnings : ℕ
  overallAverage : ℚ
  highLowDifference : ℕ
  averageExcludingHighLow : ℚ

/-- Calculates the highest score of a batsman given their statistics --/
def highestScore (stats : BatsmanStats) : ℕ :=
  sorry

/-- Theorem stating that the highest score is 179 for the given conditions --/
theorem highest_score_is_179 (stats : BatsmanStats) 
  (h1 : stats.totalInnings = 46)
  (h2 : stats.overallAverage = 60)
  (h3 : stats.highLowDifference = 150)
  (h4 : stats.averageExcludingHighLow = 58) :
  highestScore stats = 179 := by
  sorry

end highest_score_is_179_l679_67985


namespace fraction_calculation_l679_67916

theorem fraction_calculation : (7 / 9 - 5 / 6 + 5 / 18) * 18 = 4 := by
  sorry

end fraction_calculation_l679_67916


namespace area_is_zero_l679_67912

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + 5*y + 50 = 25 + 15*y - y^2

-- Define the line
def line (x : ℝ) : ℝ := x - 4

-- Define the region above the line
def region_above_line (x y : ℝ) : Prop :=
  y > line x

-- Define the area of the region
def area_of_region : ℝ := 0

-- Theorem statement
theorem area_is_zero :
  area_of_region = 0 :=
sorry

end area_is_zero_l679_67912


namespace area_circular_segment_equilateral_triangle_l679_67975

/-- The area of a circular segment cut off by one side of an inscribed equilateral triangle -/
theorem area_circular_segment_equilateral_triangle (a : ℝ) (ha : a > 0) :
  let R := a / Real.sqrt 3
  let sector_area := π * R^2 / 3
  let triangle_area := a * R / 4
  sector_area - triangle_area = (a^2 * (4 * π - 3 * Real.sqrt 3)) / 36 := by
  sorry

end area_circular_segment_equilateral_triangle_l679_67975


namespace solve_for_q_l679_67963

theorem solve_for_q (p q : ℝ) 
  (h1 : p > 1)
  (h2 : q > 1)
  (h3 : 1/p + 1/q = 1)
  (h4 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 := by
sorry

end solve_for_q_l679_67963


namespace max_robot_A_l679_67901

def robot_problem (transport_rate_A transport_rate_B : ℕ) 
                  (price_A price_B total_budget : ℕ) 
                  (total_robots : ℕ) : Prop :=
  (transport_rate_A = transport_rate_B + 30) ∧
  (1500 / transport_rate_A = 1000 / transport_rate_B) ∧
  (price_A = 50000) ∧
  (price_B = 30000) ∧
  (total_robots = 12) ∧
  (total_budget = 450000)

theorem max_robot_A (transport_rate_A transport_rate_B : ℕ) 
                    (price_A price_B total_budget : ℕ) 
                    (total_robots : ℕ) :
  robot_problem transport_rate_A transport_rate_B price_A price_B total_budget total_robots →
  ∀ m : ℕ, m ≤ total_robots ∧ 
           price_A * m + price_B * (total_robots - m) ≤ total_budget →
  m ≤ 4 :=
by sorry

end max_robot_A_l679_67901


namespace card_combination_proof_l679_67905

theorem card_combination_proof : Nat.choose 60 13 = 7446680748480 := by
  sorry

end card_combination_proof_l679_67905


namespace intersection_and_inequality_l679_67954

/-- Given real numbers a, b, c and functions f and g, prove properties about their intersections and inequalities. -/
theorem intersection_and_inequality 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (hf : f = λ x => a * x + b) 
  (hg : g = λ x => a * x^2 + b * x + c) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂) ∧ 
  (∃ d : ℝ, 3/2 < d ∧ d < 2 * Real.sqrt 3 ∧ 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧ d = |x₂ - x₁|) ∧
  (∀ x : ℝ, x ≤ -Real.sqrt 3 → f x < g x) := by
sorry

end intersection_and_inequality_l679_67954


namespace male_contestants_count_l679_67919

theorem male_contestants_count (total : ℕ) (female_ratio : ℚ) : 
  total = 18 → female_ratio = 1/3 → (total : ℚ) * (1 - female_ratio) = 12 := by
  sorry

end male_contestants_count_l679_67919


namespace wire_service_coverage_l679_67939

/-- The percentage of reporters covering local politics in country x -/
def local_politics_coverage : ℝ := 12

/-- The percentage of reporters not covering politics -/
def non_politics_coverage : ℝ := 80

/-- The percentage of reporters covering politics but not local politics in country x -/
def politics_not_local : ℝ := 40

/-- Theorem stating that given the conditions, the percentage of reporters
    who cover politics but not local politics in country x is 40% -/
theorem wire_service_coverage :
  local_politics_coverage = 12 →
  non_politics_coverage = 80 →
  politics_not_local = 40 :=
by
  sorry

#check wire_service_coverage

end wire_service_coverage_l679_67939


namespace line_slope_and_inclination_l679_67949

def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 1 = 0

theorem line_slope_and_inclination :
  ∃ (m θ : ℝ), 
    (∀ x y, line_equation x y → y = m * x + (1 / Real.sqrt 3)) ∧
    m = -Real.sqrt 3 / 3 ∧
    θ = 5 * π / 6 ∧
    Real.tan θ = m := by
  sorry

end line_slope_and_inclination_l679_67949


namespace remainder_3079_div_67_l679_67909

theorem remainder_3079_div_67 : 3079 % 67 = 64 := by sorry

end remainder_3079_div_67_l679_67909


namespace two_talent_students_l679_67910

theorem two_talent_students (total : ℕ) (cant_sing cant_dance cant_act : ℕ) :
  total = 50 ∧
  cant_sing = 20 ∧
  cant_dance = 35 ∧
  cant_act = 15 →
  ∃ (two_talents : ℕ),
    two_talents = 30 ∧
    two_talents = (total - cant_sing) + (total - cant_dance) + (total - cant_act) - total :=
by sorry

end two_talent_students_l679_67910


namespace ring_toss_total_earnings_l679_67946

/-- The ring toss game at a carnival earns a certain amount per day for a given number of days. -/
def carnival_earnings (daily_earnings : ℕ) (num_days : ℕ) : ℕ :=
  daily_earnings * num_days

/-- Theorem: The ring toss game earns 3168 dollars in total when it makes 144 dollars per day for 22 days. -/
theorem ring_toss_total_earnings :
  carnival_earnings 144 22 = 3168 := by
  sorry

end ring_toss_total_earnings_l679_67946


namespace problem_statement_l679_67968

theorem problem_statement (x : ℝ) (h : x^5 + x^4 + x = -1) :
  x^1997 + x^1998 + x^1999 + x^2000 + x^2001 + x^2002 + x^2003 + x^2004 + x^2005 + x^2006 + x^2007 = -1 := by
  sorry

end problem_statement_l679_67968


namespace blanch_snack_slices_l679_67922

/-- Calculates the number of pizza slices Blanch took as a snack -/
def snack_slices (initial : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) (left : ℕ) : ℕ :=
  initial - breakfast - lunch - dinner - left

theorem blanch_snack_slices :
  snack_slices 15 4 2 5 2 = 2 := by
  sorry

end blanch_snack_slices_l679_67922


namespace min_equation_solution_l679_67917

theorem min_equation_solution (x : ℝ) : min (1/2 + x) (x^2) = 1 → x = 1 := by
  sorry

end min_equation_solution_l679_67917


namespace parabola_directrix_parameter_l679_67983

/-- Given a parabola with equation x^2 = ay and directrix y = 1, prove that a = -4 -/
theorem parabola_directrix_parameter (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y) →  -- Equation of the parabola
  (1 : ℝ) = -a/4 →          -- Equation of the directrix (y = 1 is equivalent to 1 = -a/4 for a parabola)
  a = -4 := by
sorry

end parabola_directrix_parameter_l679_67983


namespace imaginary_part_of_complex_fraction_l679_67943

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := -25 * i / (3 + 4 * i)
  Complex.im z = -3 := by sorry

end imaginary_part_of_complex_fraction_l679_67943


namespace john_total_skateboard_distance_l679_67933

/-- The total distance John skateboarded, given his journey to and from the park -/
def total_skateboarded_distance (initial_skate : ℕ) (walk : ℕ) : ℕ :=
  2 * initial_skate

/-- Theorem stating that John skateboarded 20 miles in total -/
theorem john_total_skateboard_distance :
  total_skateboarded_distance 10 4 = 20 := by
  sorry

end john_total_skateboard_distance_l679_67933


namespace smallest_part_of_three_way_division_l679_67990

theorem smallest_part_of_three_way_division (total : ℕ) (a b c : ℕ) : 
  total = 2340 →
  a + b + c = total →
  ∃ (x : ℕ), a = 5 * x ∧ b = 7 * x ∧ c = 11 * x →
  min a (min b c) = 510 :=
by sorry

end smallest_part_of_three_way_division_l679_67990


namespace counterexample_exists_l679_67911

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ a*b := by sorry

end counterexample_exists_l679_67911


namespace smallest_angle_in_triangle_l679_67967

theorem smallest_angle_in_triangle (d e f : ℝ) (F : ℝ) (h1 : d = 2) (h2 : e = 2) (h3 : f > 4 * Real.sqrt 2) :
  let y := Real.pi
  F ≥ y ∧ ∀ z, z < y → F > z :=
by sorry

end smallest_angle_in_triangle_l679_67967


namespace min_a_for_p_true_l679_67997

-- Define the set of x
def X : Set ℝ := { x | 1 ≤ x ∧ x ≤ 9 }

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x ∈ X, x^2 - a*x + 36 ≤ 0

-- Theorem statement
theorem min_a_for_p_true : 
  (∃ a : ℝ, p a) → (∀ a : ℝ, p a → a ≥ 12) ∧ p 12 :=
sorry

end min_a_for_p_true_l679_67997


namespace p_or_q_false_sufficient_not_necessary_l679_67900

-- Define propositions p and q
variable (p q : Prop)

-- Define the statement "p or q is false"
def p_or_q_false : Prop := ¬(p ∨ q)

-- Define the statement "not p is true"
def not_p_true : Prop := ¬p

-- Theorem stating that "p or q is false" is sufficient but not necessary for "not p is true"
theorem p_or_q_false_sufficient_not_necessary :
  (p_or_q_false p q → not_p_true p) ∧
  ¬(not_p_true p → p_or_q_false p q) :=
sorry

end p_or_q_false_sufficient_not_necessary_l679_67900


namespace total_time_calculation_l679_67920

/-- Calculates the total time to complete an assignment and clean sticky keys. -/
theorem total_time_calculation (assignment_time : ℕ) (num_keys : ℕ) (time_per_key : ℕ) :
  assignment_time = 10 ∧ num_keys = 14 ∧ time_per_key = 3 →
  assignment_time + num_keys * time_per_key = 52 := by
  sorry

end total_time_calculation_l679_67920


namespace polynomial_remainder_theorem_l679_67925

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x => c * x^3 - 8 * x^2 + d * x - 7
  (g 2 = -9) ∧ (g (-1) = -19) → c = 19/3 ∧ d = -7/3 := by
  sorry

end polynomial_remainder_theorem_l679_67925


namespace must_divide_seven_l679_67956

theorem must_divide_seven (a b c d : ℕ+) 
  (h1 : Nat.gcd a.val b.val = 30)
  (h2 : Nat.gcd b.val c.val = 45)
  (h3 : Nat.gcd c.val d.val = 60)
  (h4 : 80 < Nat.gcd d.val a.val)
  (h5 : Nat.gcd d.val a.val < 120) :
  7 ∣ a.val := by
  sorry

end must_divide_seven_l679_67956


namespace intersection_of_distinct_planes_is_z_axis_l679_67957

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- A plane in cylindrical coordinates defined by a constant θ value -/
def CylindricalPlane (θ_const : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.θ = θ_const}

/-- The z-axis in cylindrical coordinates -/
def ZAxis : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = 0}

theorem intersection_of_distinct_planes_is_z_axis (θ₁ θ₂ : ℝ) (h : θ₁ ≠ θ₂) :
  (CylindricalPlane θ₁) ∩ (CylindricalPlane θ₂) = ZAxis := by
  sorry

#check intersection_of_distinct_planes_is_z_axis

end intersection_of_distinct_planes_is_z_axis_l679_67957


namespace greatest_y_value_l679_67984

theorem greatest_y_value (x y : ℤ) (h : x * y + 7 * x + 2 * y = -8) :
  y ≤ -1 ∧ ∃ (x₀ y₀ : ℤ), x₀ * y₀ + 7 * x₀ + 2 * y₀ = -8 ∧ y₀ = -1 := by
  sorry

end greatest_y_value_l679_67984


namespace x_eq_3_sufficient_not_necessary_l679_67947

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two 2D vectors are parallel -/
def areParallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

/-- Definition of vector a -/
def a (x : ℝ) : Vector2D :=
  ⟨2, x - 1⟩

/-- Definition of vector b -/
def b (x : ℝ) : Vector2D :=
  ⟨x + 1, 4⟩

/-- Theorem stating that x = 3 is a sufficient but not necessary condition for a ∥ b -/
theorem x_eq_3_sufficient_not_necessary :
  (∃ (x : ℝ), x ≠ 3 ∧ areParallel (a x) (b x)) ∧
  (∀ (x : ℝ), x = 3 → areParallel (a x) (b x)) :=
sorry

end x_eq_3_sufficient_not_necessary_l679_67947


namespace last_two_digits_28_l679_67998

theorem last_two_digits_28 (n : ℕ) (h : Odd n) (h_pos : 0 < n) :
  2^(2*n) * (2^(2*n + 1) - 1) ≡ 28 [ZMOD 100] :=
sorry

end last_two_digits_28_l679_67998


namespace opposite_vector_with_magnitude_l679_67965

/-- Given two vectors a and b in ℝ², where a is (-1, 2) and b is in the opposite direction
    to a with magnitude √5, prove that b = (1, -2) -/
theorem opposite_vector_with_magnitude (a b : ℝ × ℝ) : 
  a = (-1, 2) →
  ∃ k : ℝ, k < 0 ∧ b = k • a →
  ‖b‖ = Real.sqrt 5 →
  b = (1, -2) :=
by sorry

end opposite_vector_with_magnitude_l679_67965


namespace parallel_angle_theorem_l679_67934

theorem parallel_angle_theorem (α β : Real) :
  (α = 60 ∨ β = 60) →  -- One angle is 60°
  (α = β ∨ α + β = 180) →  -- Angles are either equal or supplementary (parallel sides condition)
  (α = 60 ∧ β = 60) ∨ (α = 60 ∧ β = 120) ∨ (α = 120 ∧ β = 60) :=
by sorry

end parallel_angle_theorem_l679_67934


namespace goods_train_speed_l679_67945

/-- Calculates the speed of a goods train given the conditions described in the problem -/
theorem goods_train_speed
  (passenger_train_speed : ℝ)
  (goods_train_length : ℝ)
  (passing_time : ℝ)
  (h_passenger_speed : passenger_train_speed = 100)
  (h_goods_length : goods_train_length = 400)
  (h_passing_time : passing_time = 12) :
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 20 ∧
    (goods_train_speed + passenger_train_speed) * passing_time / 3.6 = goods_train_length :=
by sorry

end goods_train_speed_l679_67945


namespace mirror_country_transfers_l679_67953

-- Define the type for cities
def City : Type := ℕ

-- Define the type for countries
inductive Country
| Wonderland
| Mirrorland

-- Define a function to represent railroad connections
def connected (country : Country) (city1 city2 : City) : Prop := sorry

-- Define a function to represent the "double" of a city in the other country
def double (city : City) (country : Country) : City := sorry

-- Define the number of transfers needed for a journey
def transfers (country : Country) (start finish : City) : ℕ := sorry

-- State the theorem
theorem mirror_country_transfers 
  (A B : City) 
  (h1 : transfers Country.Wonderland A B ≥ 2) 
  (h2 : ∀ (c1 c2 : City), connected Country.Wonderland c1 c2 ↔ ¬connected Country.Mirrorland (double c1 Country.Mirrorland) (double c2 Country.Mirrorland))
  (h3 : ∀ (c : City), ∃ (d : City), d = double c Country.Mirrorland)
  : ∀ (X Y : City), transfers Country.Mirrorland X Y ≤ 2 :=
by sorry

end mirror_country_transfers_l679_67953


namespace spoon_fork_sale_price_comparison_l679_67962

theorem spoon_fork_sale_price_comparison :
  ∃ (initial_price : ℕ),
    initial_price % 10 = 0 ∧
    initial_price > 100 ∧
    initial_price - 100 < initial_price / 10 :=
by
  sorry

end spoon_fork_sale_price_comparison_l679_67962


namespace unique_integer_solution_implies_a_range_l679_67966

theorem unique_integer_solution_implies_a_range (a : ℝ) :
  (∃! x : ℤ, (2 * x + 3 > 5 ∧ x - a ≤ 0)) →
  (2 ≤ a ∧ a < 3) :=
by sorry

end unique_integer_solution_implies_a_range_l679_67966


namespace tangent_condition_l679_67981

theorem tangent_condition (a b : ℝ) : 
  (∃ (x y : ℝ), x + y = 0 ∧ (x - a)^2 + (y - b)^2 = 2) → 
  (a + b = 2 → ∃ (x y : ℝ), x + y = 0 ∧ (x - a)^2 + (y - b)^2 = 2) ∧
  (∃ (a b : ℝ), (∃ (x y : ℝ), x + y = 0 ∧ (x - a)^2 + (y - b)^2 = 2) ∧ a + b ≠ 2) :=
by sorry

end tangent_condition_l679_67981


namespace fraction_subtraction_theorem_l679_67944

theorem fraction_subtraction_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let x := a * b / (a + b)
  let y := a * b * (b + a) / (b^2 + a*b + a^2)
  ((a - x) / (b - x) = (a / b)^2) ∧ ((a - y) / (b - y) = (a / b)^3) := by
  sorry

end fraction_subtraction_theorem_l679_67944


namespace probability_at_least_three_successes_in_five_trials_l679_67915

theorem probability_at_least_three_successes_in_five_trials : 
  let n : ℕ := 5
  let p : ℝ := 1/2
  let binomial_probability (k : ℕ) := (n.choose k) * p^k * (1-p)^(n-k)
  (binomial_probability 3 + binomial_probability 4 + binomial_probability 5) = 1/2 := by
  sorry

end probability_at_least_three_successes_in_five_trials_l679_67915


namespace multiples_count_l679_67979

def count_multiples (n : ℕ) : ℕ := 
  (Finset.filter (λ x => x % 2 = 0 ∨ x % 3 = 0) (Finset.range (n + 1))).card

def count_multiples_not_five (n : ℕ) : ℕ := 
  (Finset.filter (λ x => (x % 2 = 0 ∨ x % 3 = 0) ∧ x % 5 ≠ 0) (Finset.range (n + 1))).card

theorem multiples_count : count_multiples_not_five 100 = 50 := by
  sorry

end multiples_count_l679_67979


namespace collinear_vectors_y_value_l679_67999

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-4, y)
  collinear a b → y = -6 := by
sorry

end collinear_vectors_y_value_l679_67999


namespace jacket_price_reduction_l679_67904

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P > 0 →
  (1 - x) * 0.75 * P * 1.5686274509803921 = P →
  x = 0.15 :=
by sorry

end jacket_price_reduction_l679_67904


namespace grid_configurations_l679_67951

/-- Represents a grid of lightbulbs -/
structure LightbulbGrid where
  rows : Nat
  cols : Nat

/-- Represents the switches for a lightbulb grid -/
structure Switches where
  count : Nat

/-- Calculates the number of distinct configurations for a given lightbulb grid and switches -/
def distinctConfigurations (grid : LightbulbGrid) (switches : Switches) : Nat :=
  2^(switches.count - 1)

/-- Theorem: The number of distinct configurations for a 20x16 grid with 36 switches is 2^35 -/
theorem grid_configurations :
  let grid : LightbulbGrid := ⟨20, 16⟩
  let switches : Switches := ⟨36⟩
  distinctConfigurations grid switches = 2^35 := by
  sorry

#eval distinctConfigurations ⟨20, 16⟩ ⟨36⟩

end grid_configurations_l679_67951


namespace water_added_to_tank_l679_67907

theorem water_added_to_tank (tank_capacity : ℚ) 
  (h1 : tank_capacity = 64) 
  (initial_fraction : ℚ) 
  (h2 : initial_fraction = 3/4) 
  (final_fraction : ℚ) 
  (h3 : final_fraction = 7/8) : 
  (final_fraction - initial_fraction) * tank_capacity = 8 := by
  sorry

end water_added_to_tank_l679_67907


namespace clock_painting_theorem_l679_67960

def clock_numbers : ℕ := 12

def paint_interval_a : ℕ := 57
def paint_interval_b : ℕ := 2005

theorem clock_painting_theorem :
  (∃ (painted_numbers : Finset ℕ),
    painted_numbers.card = 4 ∧
    ∀ n : ℕ, n ∈ painted_numbers ↔ n < clock_numbers ∧ ∃ k : ℕ, (paint_interval_a * k) % clock_numbers = n) ∧
  (∀ n : ℕ, n < clock_numbers → ∃ k : ℕ, (paint_interval_b * k) % clock_numbers = n) :=
by sorry

end clock_painting_theorem_l679_67960


namespace worm_length_difference_l679_67988

theorem worm_length_difference (long_worm short_worm : Real) 
  (h1 : long_worm = 0.8)
  (h2 : short_worm = 0.1) :
  long_worm - short_worm = 0.7 := by
sorry

end worm_length_difference_l679_67988


namespace reseating_women_l679_67991

/-- The number of ways to reseat n women in a line, where each woman can sit in her original seat or within two positions on either side. -/
def T : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 4
  | 3 => 7
  | (n + 4) => T (n + 3) + T (n + 2) + T (n + 1)

/-- Theorem stating that the number of ways to reseat 10 women under the given conditions is 480. -/
theorem reseating_women : T 10 = 480 := by
  sorry

end reseating_women_l679_67991


namespace committee_selection_l679_67928

theorem committee_selection (n m : ℕ) (h1 : n = 15) (h2 : m = 5) : 
  Nat.choose n m = 3003 := by
  sorry

end committee_selection_l679_67928


namespace cube_sum_ratio_equals_product_ratio_l679_67982

theorem cube_sum_ratio_equals_product_ratio 
  (a b c d e f : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : d + e + f = 0) 
  (h3 : d * e * f ≠ 0) : 
  (a^3 + b^3 + c^3) / (d^3 + e^3 + f^3) = a * b * c / (d * e * f) := by
  sorry

end cube_sum_ratio_equals_product_ratio_l679_67982


namespace newer_train_distance_proof_l679_67902

/-- The distance traveled by the newer train -/
def newer_train_distance (older_train_distance : ℝ) : ℝ :=
  older_train_distance * 1.5

theorem newer_train_distance_proof (older_train_distance : ℝ) 
  (h : older_train_distance = 300) :
  newer_train_distance older_train_distance = 450 := by
  sorry

end newer_train_distance_proof_l679_67902


namespace triangle_inequality_fraction_l679_67948

theorem triangle_inequality_fraction (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a + b) / (1 + a + b) > c / (1 + c) := by
  sorry

end triangle_inequality_fraction_l679_67948


namespace unique_hyperdeficient_l679_67950

/-- Sum of all divisors of n including n itself -/
def g (n : ℕ) : ℕ := sorry

/-- A number n is hyperdeficient if g(g(n)) = n + 3 -/
def is_hyperdeficient (n : ℕ) : Prop := g (g n) = n + 3

/-- There exists exactly one hyperdeficient positive integer -/
theorem unique_hyperdeficient : ∃! n : ℕ+, is_hyperdeficient n := by sorry

end unique_hyperdeficient_l679_67950


namespace remainder_101_pow_37_mod_100_l679_67955

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := by
  sorry

end remainder_101_pow_37_mod_100_l679_67955


namespace arithmetic_mean_of_four_numbers_l679_67913

theorem arithmetic_mean_of_four_numbers :
  let numbers : List ℝ := [17, 29, 41, 53]
  (numbers.sum / numbers.length : ℝ) = 35 := by sorry

end arithmetic_mean_of_four_numbers_l679_67913


namespace A_union_B_eq_B_l679_67932

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {y | ∃ x, y = Real.sin x}

theorem A_union_B_eq_B : A ∪ B = B := by
  sorry

end A_union_B_eq_B_l679_67932


namespace product_value_l679_67973

theorem product_value (x y : ℤ) (h1 : x = 12) (h2 : y = 7) :
  (x - y) * (2 * x + 2 * y) = 190 := by
  sorry

end product_value_l679_67973


namespace distribution_count_correct_l679_67930

/-- The number of ways to distribute 4 distinct objects into 4 distinct containers 
    such that exactly one container contains 2 objects and the others contain 1 object each -/
def distributionCount : ℕ := 144

/-- The number of universities -/
def numUniversities : ℕ := 4

/-- The number of students -/
def numStudents : ℕ := 4

theorem distribution_count_correct :
  distributionCount = 
    (numStudents.choose 2) * (numUniversities * (numUniversities - 1) * (numUniversities - 2)) :=
by sorry

end distribution_count_correct_l679_67930


namespace geometry_propositions_l679_67935

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the main theorem
theorem geometry_propositions
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : subset m β) :
  -- Exactly two of the following propositions are correct
  ∃! (correct : Fin 4 → Prop),
    (∀ i, correct i ↔ i.val < 2) ∧
    correct 0 = (parallel α β → line_perpendicular l m) ∧
    correct 1 = (line_perpendicular l m → parallel α β) ∧
    correct 2 = (plane_perpendicular α β → line_parallel l m) ∧
    correct 3 = (line_parallel l m → plane_perpendicular α β) :=
sorry

end geometry_propositions_l679_67935


namespace regular_triangular_pyramid_volume_l679_67952

theorem regular_triangular_pyramid_volume 
  (b : ℝ) (β : ℝ) (h : 0 < β ∧ β < π / 2) :
  let volume := (((36 * b^2 * Real.cos β^2) / (1 + 9 * Real.cos β^2))^(3/2) * Real.tan β) / 24
  ∃ (a : ℝ), 
    a > 0 ∧ 
    volume = (a^3 * Real.tan β) / 24 ∧
    a^2 = (36 * b^2 * Real.cos β^2) / (1 + 9 * Real.cos β^2) :=
by sorry

end regular_triangular_pyramid_volume_l679_67952


namespace age_difference_l679_67918

-- Define variables for ages
variable (a b c : ℕ)

-- Define the condition from the problem
def age_condition (a b c : ℕ) : Prop := a + b = b + c + 12

-- Theorem to prove
theorem age_difference (h : age_condition a b c) : a = c + 12 := by
  sorry

end age_difference_l679_67918


namespace envelope_count_l679_67994

/-- The weight of one envelope in grams -/
def envelope_weight : ℝ := 8.5

/-- The total weight of all envelopes in kilograms -/
def total_weight : ℝ := 7.48

/-- The number of envelopes sent -/
def num_envelopes : ℕ := 880

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℝ := 1000

theorem envelope_count :
  (total_weight * kg_to_g) / envelope_weight = num_envelopes := by
  sorry

end envelope_count_l679_67994


namespace race_finish_orders_l679_67906

-- Define the number of racers
def num_racers : ℕ := 4

-- Define the function to calculate the number of permutations
def num_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Theorem statement
theorem race_finish_orders :
  num_permutations num_racers = 24 := by
  sorry

end race_finish_orders_l679_67906


namespace percentage_decrease_of_b_l679_67987

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- a and b are positive
  a / b = 4 / 5 ∧  -- ratio of a to b is 4 to 5
  x = 1.25 * a ∧  -- x equals a increased by 25 percent of a
  m = b * (1 - p / 100) ∧  -- m equals b decreased by p percent of b
  m / x = 0.4  -- m / x is 0.4
  → p = 60 :=  -- The percentage decrease of b to get m is 60%
by sorry

end percentage_decrease_of_b_l679_67987


namespace andrena_debelyn_difference_l679_67961

/-- Represents the number of dolls each person has -/
structure DollCount where
  debelyn : ℕ
  christel : ℕ
  andrena : ℕ

/-- The initial doll counts before any transfers -/
def initial_count : DollCount :=
  { debelyn := 20, christel := 24, andrena := 0 }

/-- The number of dolls transferred from Debelyn to Andrena -/
def debelyn_transfer : ℕ := 2

/-- The number of dolls transferred from Christel to Andrena -/
def christel_transfer : ℕ := 5

/-- The final doll counts after transfers -/
def final_count : DollCount :=
  { debelyn := initial_count.debelyn - debelyn_transfer,
    christel := initial_count.christel - christel_transfer,
    andrena := initial_count.andrena + debelyn_transfer + christel_transfer }

/-- Andrena has 2 more dolls than Christel after transfers -/
axiom andrena_christel_difference : final_count.andrena = final_count.christel + 2

/-- The theorem to be proved -/
theorem andrena_debelyn_difference :
  final_count.andrena - final_count.debelyn = 3 := by
  sorry

end andrena_debelyn_difference_l679_67961


namespace complex_purely_imaginary_solution_l679_67980

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

/-- Given that z and (z+2)^2 - 8i are both purely imaginary, prove that z = -2i -/
theorem complex_purely_imaginary_solution (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : isPurelyImaginary ((z + 2)^2 - 8*I)) : 
  z = -2*I := by
  sorry

end complex_purely_imaginary_solution_l679_67980
