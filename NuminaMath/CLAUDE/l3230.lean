import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_no_seven_multiple_l3230_323026

/-- Function to check if a natural number contains the digit 7 in its decimal representation -/
def containsSeven (n : ℕ) : Prop := sorry

/-- Function to generate the sequence of numbers obtained by multiplying by 5 k times -/
def multiplyByFive (n : ℕ) (k : ℕ) : List ℕ := sorry

/-- Function to generate the sequence of numbers obtained by multiplying by 2 k times -/
def multiplyByTwo (n : ℕ) (k : ℕ) : List ℕ := sorry

/-- Theorem stating the existence of a number that can be multiplied by 2 k times
    without producing a number containing 7, given a number that can be multiplied
    by 5 k times without producing a number containing 7 -/
theorem existence_of_no_seven_multiple (n : ℕ) (k : ℕ) :
  (∀ m ∈ multiplyByFive n k, ¬containsSeven m) →
  ∃ m : ℕ, ∀ p ∈ multiplyByTwo m k, ¬containsSeven p :=
by sorry

end NUMINAMATH_CALUDE_existence_of_no_seven_multiple_l3230_323026


namespace NUMINAMATH_CALUDE_unique_solution_system_l3230_323010

theorem unique_solution_system (x y z t : ℝ) :
  (x * y + z + t = 1) ∧
  (y * z + t + x = 3) ∧
  (z * t + x + y = -1) ∧
  (t * x + y + z = 1) →
  (x = 1 ∧ y = 0 ∧ z = -1 ∧ t = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3230_323010


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l3230_323044

theorem quadratic_distinct_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0) →
  m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l3230_323044


namespace NUMINAMATH_CALUDE_volunteer_selection_l3230_323034

theorem volunteer_selection (boys girls : ℕ) (positions : ℕ) : 
  boys = 4 → girls = 3 → positions = 4 → 
  (Nat.choose (boys + girls) positions) - (Nat.choose boys positions) = 34 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_selection_l3230_323034


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3230_323061

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.mk 1 2 * a + Complex.mk b 0 = Complex.I * 2) → (a = 1 ∧ b = -1) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3230_323061


namespace NUMINAMATH_CALUDE_area1_is_linear_area2_is_quadratic_l3230_323063

-- Define the rectangles
def rectangle1_length : ℝ := 10
def rectangle1_width : ℝ := 5
def rectangle2_length : ℝ := 30
def rectangle2_width : ℝ := 20

-- Define the area functions
def area1 (x : ℝ) : ℝ := (rectangle1_length - x) * rectangle1_width
def area2 (x : ℝ) : ℝ := (rectangle2_length + x) * (rectangle2_width + x)

-- Theorem statements
theorem area1_is_linear :
  ∃ (m b : ℝ), ∀ x, area1 x = m * x + b :=
sorry

theorem area2_is_quadratic :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, area2 x = a * x^2 + b * x + c) :=
sorry

end NUMINAMATH_CALUDE_area1_is_linear_area2_is_quadratic_l3230_323063


namespace NUMINAMATH_CALUDE_family_movie_night_l3230_323091

theorem family_movie_night (regular_price adult_price elderly_price : ℕ)
  (child_price_diff total_payment change num_adults num_elderly : ℕ) :
  regular_price = 15 →
  adult_price = 12 →
  elderly_price = 10 →
  child_price_diff = 5 →
  total_payment = 150 →
  change = 3 →
  num_adults = 4 →
  num_elderly = 2 →
  ∃ (num_children : ℕ),
    num_children = 11 ∧
    total_payment - change = 
      num_adults * adult_price + 
      num_elderly * elderly_price + 
      num_children * (adult_price - child_price_diff) :=
by sorry

end NUMINAMATH_CALUDE_family_movie_night_l3230_323091


namespace NUMINAMATH_CALUDE_sufficient_unnecessary_condition_l3230_323070

theorem sufficient_unnecessary_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-4) 2 → (1/2) * x^2 - a ≥ 0) ↔ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_unnecessary_condition_l3230_323070


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l3230_323018

theorem gcd_from_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : 
  Nat.gcd X Y = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l3230_323018


namespace NUMINAMATH_CALUDE_cube_difference_divided_problem_solution_l3230_323025

theorem cube_difference_divided (a b : ℕ) (h : a > b) :
  (a^3 - b^3) / (a - b) = a^2 + a*b + b^2 :=
by sorry

theorem problem_solution : (64^3 - 27^3) / 37 = 6553 :=
by
  have h : 64 > 27 := by sorry
  have := cube_difference_divided 64 27 h
  sorry

end NUMINAMATH_CALUDE_cube_difference_divided_problem_solution_l3230_323025


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3230_323099

theorem trigonometric_identity (α : ℝ) : 
  4.6 * (Real.cos (5 * Real.pi / 2 - 6 * α) + Real.sin (Real.pi + 4 * α) + Real.sin (3 * Real.pi - α)) / 
  (Real.sin (5 * Real.pi / 2 + 6 * α) + Real.cos (4 * α - 2 * Real.pi) + Real.cos (α + 2 * Real.pi)) = 
  Real.tan α := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3230_323099


namespace NUMINAMATH_CALUDE_triangle_inequality_l3230_323060

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a^2 + b^2 - a*b = c^2) : 
  (a - c) * (b - c) ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3230_323060


namespace NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l3230_323046

theorem least_n_with_gcd_conditions (n : ℕ) : 
  n > 1500 ∧ 
  Nat.gcd 40 (n + 105) = 10 ∧ 
  Nat.gcd (n + 40) 105 = 35 ∧
  (∀ m : ℕ, m > 1500 → Nat.gcd 40 (m + 105) = 10 → Nat.gcd (m + 40) 105 = 35 → m ≥ n) →
  n = 1511 := by
sorry

end NUMINAMATH_CALUDE_least_n_with_gcd_conditions_l3230_323046


namespace NUMINAMATH_CALUDE_unit_digit_of_3_to_58_l3230_323036

theorem unit_digit_of_3_to_58 : 3^58 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_3_to_58_l3230_323036


namespace NUMINAMATH_CALUDE_candy_mixture_total_candy_mixture_total_proof_l3230_323007

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

end NUMINAMATH_CALUDE_candy_mixture_total_candy_mixture_total_proof_l3230_323007


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3230_323013

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 5 * x) = 3 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3230_323013


namespace NUMINAMATH_CALUDE_composition_value_l3230_323051

/-- Given two functions f and g, and a composition condition, prove that d equals 8 -/
theorem composition_value (c d : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5*x + c)
  (hg : ∀ x, g x = c*x + 1)
  (hcomp : ∀ x, f (g x) = 15*x + d) :
  d = 8 := by
  sorry

end NUMINAMATH_CALUDE_composition_value_l3230_323051


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l3230_323081

theorem complex_magnitude_one (z : ℂ) (p : ℕ) (h : 11 * z^10 + 10 * Complex.I * z^p + 10 * Complex.I * z - 11 = 0) : 
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l3230_323081


namespace NUMINAMATH_CALUDE_coats_collected_at_elementary_schools_l3230_323057

theorem coats_collected_at_elementary_schools 
  (total_coats : ℕ) 
  (high_school_coats : ℕ) 
  (h1 : total_coats = 9437) 
  (h2 : high_school_coats = 6922) : 
  total_coats - high_school_coats = 2515 := by
  sorry

end NUMINAMATH_CALUDE_coats_collected_at_elementary_schools_l3230_323057


namespace NUMINAMATH_CALUDE_xyz_and_fourth_power_sum_l3230_323093

theorem xyz_and_fourth_power_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 1)
  (sum_sq_eq : x^2 + y^2 + z^2 = 2)
  (sum_cube_eq : x^3 + y^3 + z^3 = 3) :
  x * y * z = 1/6 ∧ x^4 + y^4 + z^4 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_xyz_and_fourth_power_sum_l3230_323093


namespace NUMINAMATH_CALUDE_triangle_identity_l3230_323027

/-- Definition of the △ operation for ordered pairs of real numbers -/
def triangle (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, a * d + b * c)

/-- Theorem stating that if (a, b) △ (x, y) = (a, b) for all real a and b, then (x, y) = (1, 0) -/
theorem triangle_identity (x y : ℝ) : 
  (∀ a b : ℝ, triangle a b x y = (a, b)) → (x, y) = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_identity_l3230_323027


namespace NUMINAMATH_CALUDE_william_arrival_time_l3230_323029

/-- Represents time as hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : minutes < 60

/-- Adds hours and minutes to a given time -/
def addTime (t : Time) (h : ℕ) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + h * 60 + m
  let newHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  ⟨newHours % 24, newMinutes, by sorry⟩

/-- Calculates arrival time given departure time, total time on road, and total stop time -/
def calculateArrivalTime (departureTime : Time) (totalTimeOnRoad : ℕ) (totalStopTime : ℕ) : Time :=
  let actualDrivingTime := totalTimeOnRoad - (totalStopTime / 60)
  addTime departureTime actualDrivingTime 0

theorem william_arrival_time :
  let departureTime : Time := ⟨7, 0, by sorry⟩
  let totalTimeOnRoad : ℕ := 12
  let stopTimes : List ℕ := [25, 10, 25]
  let totalStopTime : ℕ := stopTimes.sum
  let arrivalTime := calculateArrivalTime departureTime totalTimeOnRoad totalStopTime
  arrivalTime = ⟨18, 0, by sorry⟩ := by sorry

end NUMINAMATH_CALUDE_william_arrival_time_l3230_323029


namespace NUMINAMATH_CALUDE_R_and_T_largest_area_l3230_323031

/-- Represents a polygon constructed from unit squares and right triangles with legs of length 1 -/
structure Polygon where
  squares : ℕ
  triangles : ℕ

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℚ :=
  p.squares + p.triangles / 2

/-- The five polygons P, Q, R, S, T -/
def P : Polygon := ⟨3, 2⟩
def Q : Polygon := ⟨4, 1⟩
def R : Polygon := ⟨6, 0⟩
def S : Polygon := ⟨2, 4⟩
def T : Polygon := ⟨5, 2⟩

/-- Theorem stating that R and T have the largest area among the five polygons -/
theorem R_and_T_largest_area :
  area R = area T ∧
  area R ≥ area P ∧
  area R ≥ area Q ∧
  area R ≥ area S :=
sorry

end NUMINAMATH_CALUDE_R_and_T_largest_area_l3230_323031


namespace NUMINAMATH_CALUDE_smallest_number_l3230_323096

theorem smallest_number (a b c d : ℝ) (h1 : a = 0) (h2 : b = -1) (h3 : c = -Real.sqrt 3) (h4 : d = 3) :
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3230_323096


namespace NUMINAMATH_CALUDE_four_is_square_root_of_sixteen_l3230_323077

-- Definition of square root
def is_square_root (x y : ℝ) : Prop := y * y = x

-- Theorem to prove
theorem four_is_square_root_of_sixteen : is_square_root 16 4 := by
  sorry

end NUMINAMATH_CALUDE_four_is_square_root_of_sixteen_l3230_323077


namespace NUMINAMATH_CALUDE_distance_P_to_xaxis_l3230_323053

/-- The distance from a point to the x-axis is the absolute value of its y-coordinate -/
def distanceToXAxis (p : ℝ × ℝ) : ℝ := |p.2|

/-- Point P with coordinates (-3, 1) -/
def P : ℝ × ℝ := (-3, 1)

/-- Theorem: The distance from point P to the x-axis is 1 -/
theorem distance_P_to_xaxis : distanceToXAxis P = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_xaxis_l3230_323053


namespace NUMINAMATH_CALUDE_percentage_of_sum_l3230_323014

theorem percentage_of_sum (x y : ℝ) (P : ℝ) 
  (h1 : 0.2 * (x - y) = P / 100 * (x + y))
  (h2 : y = 14.285714285714285 / 100 * x) :
  P = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sum_l3230_323014


namespace NUMINAMATH_CALUDE_train_crossing_time_l3230_323090

/-- Calculates the time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 100 →
  train_speed = 54.99520038396929 →
  man_speed = 5 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3230_323090


namespace NUMINAMATH_CALUDE_money_calculation_l3230_323048

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def totalMoney (n50 : ℕ) (n500 : ℕ) : ℕ :=
  50 * n50 + 500 * n500

/-- Proves that the total amount of money is 10350 rupees given the specified conditions -/
theorem money_calculation :
  ∀ (n50 n500 : ℕ),
    n50 = 37 →
    n50 + n500 = 54 →
    totalMoney n50 n500 = 10350 := by
  sorry

#eval totalMoney 37 17  -- Should output 10350

end NUMINAMATH_CALUDE_money_calculation_l3230_323048


namespace NUMINAMATH_CALUDE_cups_sold_calculation_l3230_323050

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

end NUMINAMATH_CALUDE_cups_sold_calculation_l3230_323050


namespace NUMINAMATH_CALUDE_problem_solution_l3230_323080

theorem problem_solution : ∃ x : ℝ, 
  ((35 * x)^2 / 100) * x = (23/18) / 100 * 9500 - 175 ∧ 
  abs (x + 0.62857) < 0.00001 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3230_323080


namespace NUMINAMATH_CALUDE_parallelogram_base_l3230_323088

theorem parallelogram_base (area height base : ℝ) : 
  area = 336 ∧ height = 24 ∧ area = base * height → base = 14 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l3230_323088


namespace NUMINAMATH_CALUDE_percentage_calculation_l3230_323089

theorem percentage_calculation (x : ℝ) : 
  x = 0.18 * 4750 → 1.5 * x = 1282.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3230_323089


namespace NUMINAMATH_CALUDE_lumber_price_increase_l3230_323032

theorem lumber_price_increase 
  (original_lumber_cost : ℝ)
  (original_nails_cost : ℝ)
  (original_fabric_cost : ℝ)
  (total_cost_increase : ℝ)
  (h1 : original_lumber_cost = 450)
  (h2 : original_nails_cost = 30)
  (h3 : original_fabric_cost = 80)
  (h4 : total_cost_increase = 97) :
  let original_total_cost := original_lumber_cost + original_nails_cost + original_fabric_cost
  let new_total_cost := original_total_cost + total_cost_increase
  let new_lumber_cost := new_total_cost - (original_nails_cost + original_fabric_cost)
  let lumber_cost_increase := new_lumber_cost - original_lumber_cost
  let percentage_increase := (lumber_cost_increase / original_lumber_cost) * 100
  percentage_increase = 21.56 := by
  sorry

end NUMINAMATH_CALUDE_lumber_price_increase_l3230_323032


namespace NUMINAMATH_CALUDE_train_length_calculation_l3230_323033

/-- The length of a train given its speed, the speed of a trolley moving in the opposite direction, and the time it takes for the train to pass the trolley. -/
theorem train_length_calculation (train_speed : ℝ) (trolley_speed : ℝ) (passing_time : ℝ) : 
  train_speed = 60 →
  trolley_speed = 12 →
  passing_time = 5.4995600351971845 →
  ∃ (train_length : ℝ), abs (train_length - 109.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3230_323033


namespace NUMINAMATH_CALUDE_rectangle_areas_sum_l3230_323058

theorem rectangle_areas_sum : 
  let base_width : ℕ := 2
  let lengths : List ℕ := [1, 4, 9, 16, 25, 36, 49]
  let areas : List ℕ := lengths.map (λ l => base_width * l)
  areas.sum = 280 := by sorry

end NUMINAMATH_CALUDE_rectangle_areas_sum_l3230_323058


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3230_323011

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3230_323011


namespace NUMINAMATH_CALUDE_triangle_area_main_theorem_l3230_323062

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the area of a specific triangle -/
theorem triangle_area (t : Triangle) 
  (h1 : t.a^2 = t.b^2 + t.c^2 - t.b * t.c) 
  (h2 : t.b * t.c = 16) : 
  (1/2) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 := by
  sorry

/-- Main theorem proving the area of the triangle -/
theorem main_theorem : 
  ∃ (t : Triangle), 
    t.a^2 = t.b^2 + t.c^2 - t.b * t.c ∧ 
    t.b * t.c = 16 ∧ 
    (1/2) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_main_theorem_l3230_323062


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l3230_323037

-- Define the probability of having a boy or girl
def prob_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- The theorem to prove
theorem prob_at_least_one_boy_and_girl : 
  (1 : ℚ) - 2 * (prob_boy_or_girl ^ num_children) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l3230_323037


namespace NUMINAMATH_CALUDE_blue_pen_cost_is_ten_cents_l3230_323086

/-- The cost of a blue pen given the conditions of Maci's pen purchase. -/
def blue_pen_cost (blue_pens red_pens : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (blue_pens + 2 * red_pens)

/-- Theorem stating that the cost of a blue pen is $0.10 under the given conditions. -/
theorem blue_pen_cost_is_ten_cents :
  blue_pen_cost 10 15 4 = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_blue_pen_cost_is_ten_cents_l3230_323086


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3230_323076

/-- Given two concentric circles where a chord is tangent to the smaller circle,
    this theorem proves the area of the region between the circles. -/
theorem area_between_concentric_circles
  (outer_radius inner_radius chord_length : ℝ)
  (h_outer_positive : 0 < outer_radius)
  (h_inner_positive : 0 < inner_radius)
  (h_outer_greater : inner_radius < outer_radius)
  (h_chord_tangent : chord_length^2 = outer_radius^2 - inner_radius^2)
  (h_chord_length : chord_length = 100) :
  (outer_radius^2 - inner_radius^2) * π = 2000 * π :=
sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3230_323076


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3230_323005

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a+1, a^2+3}

-- State the theorem
theorem intersection_implies_a_value :
  ∀ a : ℝ, (A a ∩ B a = {-3}) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3230_323005


namespace NUMINAMATH_CALUDE_team_formation_count_l3230_323092

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem team_formation_count : 
  let total_boys : ℕ := 4
  let total_girls : ℕ := 5
  let team_size : ℕ := 5
  let ways_3b2g : ℕ := choose total_boys 3 * choose total_girls 2
  let ways_4b1g : ℕ := choose total_boys 4 * choose total_girls 1
  ways_3b2g + ways_4b1g = 45 := by sorry

end NUMINAMATH_CALUDE_team_formation_count_l3230_323092


namespace NUMINAMATH_CALUDE_extra_people_on_train_l3230_323064

theorem extra_people_on_train (current : ℕ) (initial : ℕ) (got_off : ℕ)
  (h1 : current = 63)
  (h2 : initial = 78)
  (h3 : got_off = 27) :
  current - (initial - got_off) = 12 :=
by sorry

end NUMINAMATH_CALUDE_extra_people_on_train_l3230_323064


namespace NUMINAMATH_CALUDE_tournament_games_theorem_l3230_323078

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (num_teams : ℕ)
  (num_players_per_team : ℕ)

/-- Calculates the number of games needed to determine the champion -/
def games_to_champion (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- The theorem stating that a tournament with 128 teams requires 127 games to determine the champion -/
theorem tournament_games_theorem :
  ∀ (t : Tournament), t.num_teams = 128 → t.num_players_per_team = 4 → games_to_champion t = 127 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_theorem_l3230_323078


namespace NUMINAMATH_CALUDE_intersection_x_is_seven_l3230_323074

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- The x-axis -/
def x_axis : Line := { p1 := ⟨0, 0⟩, p2 := ⟨1, 0⟩ }

/-- Point C -/
def C : Point := ⟨7, 5⟩

/-- Point D -/
def D : Point := ⟨7, -3⟩

/-- The line passing through points C and D -/
def line_CD : Line := { p1 := C, p2 := D }

/-- The x-coordinate of the intersection point between a line and the x-axis -/
def intersection_x (l : Line) : ℝ := sorry

theorem intersection_x_is_seven : intersection_x line_CD = 7 := by sorry

end NUMINAMATH_CALUDE_intersection_x_is_seven_l3230_323074


namespace NUMINAMATH_CALUDE_marble_distribution_l3230_323085

theorem marble_distribution (n : ℕ) (x : ℕ) :
  (∀ i : ℕ, i ≤ n → i + (n * x - (i * (i + 1)) / 2) / 10 = x) →
  (n * x = n * x - (n * (n + 1)) / 2) →
  (n = 9 ∧ x = 9) := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l3230_323085


namespace NUMINAMATH_CALUDE_tree_break_height_l3230_323003

/-- Given a tree of height 36 meters that breaks and falls across a road of width 12 meters,
    touching the opposite edge, the height at which the tree broke is 16 meters. -/
theorem tree_break_height :
  ∀ (h : ℝ), 
  h > 0 →
  h < 36 →
  (36 - h)^2 = h^2 + 12^2 →
  h = 16 :=
by sorry

end NUMINAMATH_CALUDE_tree_break_height_l3230_323003


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3230_323075

def P : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def Q : Set ℝ := {x | x ≥ 3}

theorem intersection_of_P_and_Q : P ∩ Q = {x | 3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3230_323075


namespace NUMINAMATH_CALUDE_problem_statement_l3230_323022

/-- Given m > 0, p, and q as defined, prove the conditions for m and x. -/
theorem problem_statement (m : ℝ) (h_m : m > 0) : 
  -- Define p
  let p := fun x : ℝ => (x + 1) * (x - 5) ≤ 0
  -- Define q
  let q := fun x : ℝ => 1 - m ≤ x ∧ x ≤ 1 + m
  -- Part 1: When p is a sufficient condition for q, m ≥ 4
  ((∀ x : ℝ, p x → q x) → m ≥ 4) ∧
  -- Part 2: When m = 5 and (p or q) is true but (p and q) is false, 
  --         x is in the specified range
  (m = 5 → 
    ∀ x : ℝ, ((p x ∨ q x) ∧ ¬(p x ∧ q x)) → 
      ((-4 ≤ x ∧ x < -1) ∨ (5 < x ∧ x < 6))) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3230_323022


namespace NUMINAMATH_CALUDE_smallest_product_of_primes_above_50_l3230_323065

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def first_prime_above_50 : ℕ := 53

def second_prime_above_50 : ℕ := 59

theorem smallest_product_of_primes_above_50 :
  (is_prime first_prime_above_50) ∧
  (is_prime second_prime_above_50) ∧
  (first_prime_above_50 > 50) ∧
  (second_prime_above_50 > 50) ∧
  (first_prime_above_50 < second_prime_above_50) ∧
  (∀ p : ℕ, is_prime p ∧ p > 50 ∧ p ≠ first_prime_above_50 → p ≥ second_prime_above_50) →
  first_prime_above_50 * second_prime_above_50 = 3127 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_of_primes_above_50_l3230_323065


namespace NUMINAMATH_CALUDE_lemonade_water_calculation_l3230_323028

/-- Represents the ratio of water to lemon juice in the lemonade recipe -/
def water_to_juice_ratio : ℚ := 5 / 3

/-- Represents the number of gallons of lemonade to be made -/
def gallons_to_make : ℚ := 2

/-- Represents the number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- Calculates the number of quarts of water needed for the lemonade recipe -/
def quarts_of_water_needed : ℚ :=
  (water_to_juice_ratio * gallons_to_make * quarts_per_gallon) / (water_to_juice_ratio + 1)

/-- Theorem stating that 5 quarts of water are needed for the lemonade recipe -/
theorem lemonade_water_calculation :
  quarts_of_water_needed = 5 := by sorry

end NUMINAMATH_CALUDE_lemonade_water_calculation_l3230_323028


namespace NUMINAMATH_CALUDE_collinearity_of_special_points_l3230_323008

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

end NUMINAMATH_CALUDE_collinearity_of_special_points_l3230_323008


namespace NUMINAMATH_CALUDE_georges_socks_l3230_323084

theorem georges_socks (initial_socks bought_socks dad_socks : ℝ) 
  (h1 : initial_socks = 28.0)
  (h2 : bought_socks = 36.0)
  (h3 : dad_socks = 4.0) :
  initial_socks + bought_socks + dad_socks = 68.0 :=
by sorry

end NUMINAMATH_CALUDE_georges_socks_l3230_323084


namespace NUMINAMATH_CALUDE_new_total_bill_l3230_323066

def original_order_cost : ℝ := 25
def tomatoes_old_price : ℝ := 0.99
def tomatoes_new_price : ℝ := 2.20
def lettuce_old_price : ℝ := 1.00
def lettuce_new_price : ℝ := 1.75
def celery_old_price : ℝ := 1.96
def celery_new_price : ℝ := 2.00
def delivery_tip_cost : ℝ := 8.00

theorem new_total_bill :
  let price_increase := (tomatoes_new_price - tomatoes_old_price) +
                        (lettuce_new_price - lettuce_old_price) +
                        (celery_new_price - celery_old_price)
  let new_food_cost := original_order_cost + price_increase
  let total_bill := new_food_cost + delivery_tip_cost
  total_bill = 35 := by
  sorry

end NUMINAMATH_CALUDE_new_total_bill_l3230_323066


namespace NUMINAMATH_CALUDE_inequality_solution_l3230_323016

theorem inequality_solution (x y : ℝ) :
  (y^2 - 4*x*y + 4*x^2 < x^2) ↔ (x < y ∧ y < 3*x ∧ x > 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3230_323016


namespace NUMINAMATH_CALUDE_tangent_points_line_passes_through_fixed_point_l3230_323047

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Tangent line from a point to a parabola -/
def tangent_line (p : Parabola) (m : Point) : Line :=
  sorry

/-- Tangent point of a line to a parabola -/
def tangent_point (p : Parabola) (l : Line) : Point :=
  sorry

/-- Given a parabola C, a line l, and a point M on l, prove that the line AB 
    formed by the tangent points A and B of the tangent lines from M to C 
    always passes through a fixed point -/
theorem tangent_points_line_passes_through_fixed_point 
  (C : Parabola) 
  (l : Line) 
  (M : Point) 
  (m : ℝ) 
  (h1 : C.equation = fun x y => x^2 = 4*y) 
  (h2 : l.equation = fun x y => y = -m) 
  (h3 : m > 0) 
  (h4 : l.equation M.x M.y) :
  let t1 := tangent_line C M
  let t2 := tangent_line C M
  let A := tangent_point C t1
  let B := tangent_point C t2
  let AB : Line := sorry
  AB.equation 0 m := by sorry

end NUMINAMATH_CALUDE_tangent_points_line_passes_through_fixed_point_l3230_323047


namespace NUMINAMATH_CALUDE_twenty_percent_less_than_sixty_l3230_323006

theorem twenty_percent_less_than_sixty (x : ℝ) : x + (1/3) * x = 60 - 0.2 * 60 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_less_than_sixty_l3230_323006


namespace NUMINAMATH_CALUDE_total_cards_traded_is_128_l3230_323000

/-- Represents the number of cards of each type --/
structure CardCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents a trade of cards --/
structure Trade where
  fromA : ℕ
  fromB : ℕ
  fromC : ℕ
  toA : ℕ
  toB : ℕ
  toC : ℕ

/-- Calculates the total number of cards traded in a single trade --/
def cardsTraded (trade : Trade) : ℕ :=
  trade.fromA + trade.fromB + trade.fromC + trade.toA + trade.toB + trade.toC

/-- Represents the initial card counts and trades for each round --/
structure RoundData where
  initialPadma : CardCounts
  initialRobert : CardCounts
  padmaTrade : Trade
  robertTrade : Trade

theorem total_cards_traded_is_128 
  (round1 : RoundData)
  (round2 : RoundData)
  (round3 : RoundData)
  (h1 : round1.initialPadma = ⟨50, 45, 30⟩)
  (h2 : round1.padmaTrade = ⟨5, 12, 0, 0, 0, 20⟩)
  (h3 : round2.initialRobert = ⟨60, 50, 40⟩)
  (h4 : round2.robertTrade = ⟨10, 3, 15, 8, 18, 0⟩)
  (h5 : round3.padmaTrade = ⟨0, 15, 10, 12, 0, 0⟩) :
  cardsTraded round1.padmaTrade + cardsTraded round2.robertTrade + cardsTraded round3.padmaTrade = 128 := by
  sorry


end NUMINAMATH_CALUDE_total_cards_traded_is_128_l3230_323000


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3230_323039

theorem trigonometric_identities (α : Real) (h_acute : 0 < α ∧ α < Real.pi / 2) 
  (h_sin : Real.sin α = 3 / 5) : 
  (Real.cos α = 4 / 5) ∧ 
  (Real.cos (α + Real.pi / 6) = (4 * Real.sqrt 3 - 3) / 10) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3230_323039


namespace NUMINAMATH_CALUDE_circle_area_tripled_l3230_323038

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → r = n/2 * (Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l3230_323038


namespace NUMINAMATH_CALUDE_nina_weekend_sales_l3230_323021

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

end NUMINAMATH_CALUDE_nina_weekend_sales_l3230_323021


namespace NUMINAMATH_CALUDE_no_roots_implies_negative_a_l3230_323012

theorem no_roots_implies_negative_a :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x - 1/x + a ≠ 0) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_roots_implies_negative_a_l3230_323012


namespace NUMINAMATH_CALUDE_consecutive_points_theorem_l3230_323069

/-- Represents a point on a straight line -/
structure Point where
  x : ℝ

/-- Represents the distance between two points -/
def distance (p q : Point) : ℝ := q.x - p.x

theorem consecutive_points_theorem (a b c d e : Point)
  (consecutive : a.x < b.x ∧ b.x < c.x ∧ c.x < d.x ∧ d.x < e.x)
  (bc_eq_2cd : distance b c = 2 * distance c d)
  (de_eq_8 : distance d e = 8)
  (ac_eq_11 : distance a c = 11)
  (ae_eq_22 : distance a e = 22) :
  distance a b = 5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_points_theorem_l3230_323069


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3230_323041

theorem largest_n_satisfying_inequality : 
  (∀ n : ℕ, n ≤ 7 → (1 : ℚ) / 4 + (n : ℚ) / 6 < 3 / 2) ∧
  (∀ n : ℕ, n > 7 → (1 : ℚ) / 4 + (n : ℚ) / 6 ≥ 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l3230_323041


namespace NUMINAMATH_CALUDE_chord_equation_l3230_323045

theorem chord_equation (m n s t : ℝ) (hm : 0 < m) (hn : 0 < n) (hs : 0 < s) (ht : 0 < t)
  (h1 : m + n = 2) (h2 : m / s + n / t = 9) (h3 : s + t = 4 / 9)
  (h4 : ∃ (x1 y1 x2 y2 : ℝ), 
    x1^2 / 4 + y1^2 / 2 = 1 ∧ 
    x2^2 / 4 + y2^2 / 2 = 1 ∧ 
    (x1 + x2) / 2 = m ∧ 
    (y1 + y2) / 2 = n) :
  ∃ (a b c : ℝ), a * m + b * n + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3 := by
sorry

end NUMINAMATH_CALUDE_chord_equation_l3230_323045


namespace NUMINAMATH_CALUDE_divide_ten_theorem_l3230_323015

theorem divide_ten_theorem (x : ℝ) : 
  x > 0 ∧ x < 10 →
  (10 - x)^2 + x^2 + (10 - x) / x = 72 →
  x = 2 := by
sorry


end NUMINAMATH_CALUDE_divide_ten_theorem_l3230_323015


namespace NUMINAMATH_CALUDE_tangent_line_parallel_point_l3230_323098

theorem tangent_line_parallel_point (x y : ℝ) : 
  y = x^4 - x → -- P(x, y) is on the curve
  (4 * x^3 - 1 : ℝ) = 3 → -- Tangent line is parallel to 3x - y + 1 = 0
  x = 1 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_point_l3230_323098


namespace NUMINAMATH_CALUDE_max_correct_answers_is_19_l3230_323059

/-- Represents the result of an exam -/
structure ExamResult where
  total_questions : Nat
  correct_answers : Nat
  wrong_answers : Nat
  unanswered : Nat
  score : Int

/-- Checks if an ExamResult is valid according to the given scoring system -/
def is_valid_result (result : ExamResult) : Prop :=
  result.total_questions = 25 ∧
  result.correct_answers + result.wrong_answers + result.unanswered = result.total_questions ∧
  4 * result.correct_answers - result.wrong_answers = result.score

/-- Theorem: The maximum number of correct answers for a score of 70 is 19 -/
theorem max_correct_answers_is_19 :
  ∀ result : ExamResult,
    is_valid_result result →
    result.score = 70 →
    result.correct_answers ≤ 19 ∧
    ∃ optimal_result : ExamResult,
      is_valid_result optimal_result ∧
      optimal_result.score = 70 ∧
      optimal_result.correct_answers = 19 :=
by sorry

#check max_correct_answers_is_19

end NUMINAMATH_CALUDE_max_correct_answers_is_19_l3230_323059


namespace NUMINAMATH_CALUDE_soap_brand_survey_l3230_323001

theorem soap_brand_survey (total : ℕ) (only_A : ℕ) (both : ℕ) (only_B_ratio : ℕ) :
  total = 260 →
  only_A = 60 →
  both = 30 →
  only_B_ratio = 3 →
  total - (only_A + both + only_B_ratio * both) = 80 :=
by sorry

end NUMINAMATH_CALUDE_soap_brand_survey_l3230_323001


namespace NUMINAMATH_CALUDE_monomial_type_sum_l3230_323004

/-- Two monomials are of the same type if they have the same variables raised to the same powers -/
def same_type_monomials (m n : ℕ) : Prop :=
  m - 1 = 2 ∧ n + 1 = 2

theorem monomial_type_sum (m n : ℕ) :
  same_type_monomials m n → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_monomial_type_sum_l3230_323004


namespace NUMINAMATH_CALUDE_part_one_part_two_l3230_323043

/-- Definition of a difference solution equation -/
def is_difference_solution_equation (a b : ℝ) : Prop :=
  (b / a) = b - a

/-- Part 1: Prove that 3x = 4.5 is a difference solution equation -/
theorem part_one : is_difference_solution_equation 3 4.5 := by sorry

/-- Part 2: Prove that 5x - m = 1 is a difference solution equation when m = 21/4 -/
theorem part_two : is_difference_solution_equation 5 ((21/4) + 1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3230_323043


namespace NUMINAMATH_CALUDE_quadratic_vertex_l3230_323020

/-- The quadratic function f(x) = 3(x+4)^2 - 5 has vertex at (-4, -5) -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * (x + 4)^2 - 5
  (∀ x, f x = 3 * (x + 4)^2 - 5) →
  ∃! (h k : ℝ), ∀ x, f x = 3 * (x - h)^2 + k ∧ h = -4 ∧ k = -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l3230_323020


namespace NUMINAMATH_CALUDE_orange_juice_price_l3230_323094

def initial_money : ℕ := 86
def bread_price : ℕ := 3
def bread_quantity : ℕ := 3
def juice_quantity : ℕ := 3
def money_left : ℕ := 59

theorem orange_juice_price :
  ∃ (juice_price : ℕ),
    initial_money - (bread_price * bread_quantity + juice_price * juice_quantity) = money_left ∧
    juice_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_price_l3230_323094


namespace NUMINAMATH_CALUDE_diamond_three_four_l3230_323083

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem diamond_three_four : diamond 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l3230_323083


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l3230_323040

/-- A regular polygon with exterior angle of 36 degrees has 7 diagonals from each vertex -/
theorem regular_polygon_diagonals (n : ℕ) (h_regular : n ≥ 3) :
  (360 : ℝ) / 36 = n → n - 3 = 7 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l3230_323040


namespace NUMINAMATH_CALUDE_chord_length_l3230_323079

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r ^ 2 - d ^ 2)
  chord_length = 6 := by sorry

end NUMINAMATH_CALUDE_chord_length_l3230_323079


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3230_323023

theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![3, -1]
  let b : Fin 2 → ℝ := ![1, x]
  (∀ i, i < 2 → a i * b i = 0) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3230_323023


namespace NUMINAMATH_CALUDE_converse_of_quadratic_eq_l3230_323019

theorem converse_of_quadratic_eq (x : ℝ) : x = 1 ∨ x = 2 → x^2 - 3*x + 2 = 0 := by sorry

end NUMINAMATH_CALUDE_converse_of_quadratic_eq_l3230_323019


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3230_323056

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x - 4|

-- Define the solution set
def solution_set : Set ℝ := {x | x < -2 ∨ x > Real.sqrt 2}

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, f (x^2 + 2) > f x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3230_323056


namespace NUMINAMATH_CALUDE_sister_reams_proof_l3230_323049

/-- The number of reams of paper bought for Haley -/
def reams_for_haley : ℕ := 2

/-- The total number of reams of paper bought -/
def total_reams : ℕ := 5

/-- The number of reams of paper bought for Haley's sister -/
def reams_for_sister : ℕ := total_reams - reams_for_haley

theorem sister_reams_proof : reams_for_sister = 3 := by
  sorry

end NUMINAMATH_CALUDE_sister_reams_proof_l3230_323049


namespace NUMINAMATH_CALUDE_fraction_addition_l3230_323054

theorem fraction_addition : 
  (7 : ℚ) / 12 + (11 : ℚ) / 16 = (61 : ℚ) / 48 :=
by sorry

end NUMINAMATH_CALUDE_fraction_addition_l3230_323054


namespace NUMINAMATH_CALUDE_feb_first_is_wednesday_l3230_323073

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the previous day
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

-- Define a function to go back n days
def goBackDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => prevDay (goBackDays d m)

-- Theorem statement
theorem feb_first_is_wednesday (h : DayOfWeek) :
  h = DayOfWeek.Tuesday → goBackDays h 27 = DayOfWeek.Wednesday :=
by
  sorry

end NUMINAMATH_CALUDE_feb_first_is_wednesday_l3230_323073


namespace NUMINAMATH_CALUDE_quadrilateral_angle_sum_my_quadrilateral_invalid_l3230_323095

-- Define a quadrilateral
structure Quadrilateral where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ

-- Theorem: The sum of interior angles of a quadrilateral is 360°
theorem quadrilateral_angle_sum (q : Quadrilateral) : 
  q.angle1 + q.angle2 + q.angle3 + q.angle4 = 360 := by
  sorry

-- Define our specific quadrilateral
def my_quadrilateral : Quadrilateral :=
  { side1 := 15
    side2 := 20
    side3 := 25
    side4 := 33
    angle1 := 100
    angle2 := 130
    angle3 := 105
    angle4 := 125 }

-- Theorem: Our specific quadrilateral violates the angle sum property
theorem my_quadrilateral_invalid : 
  my_quadrilateral.angle1 + my_quadrilateral.angle2 + 
  my_quadrilateral.angle3 + my_quadrilateral.angle4 ≠ 360 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_sum_my_quadrilateral_invalid_l3230_323095


namespace NUMINAMATH_CALUDE_fruit_box_problem_l3230_323072

theorem fruit_box_problem (total_fruit oranges peaches apples : ℕ) : 
  total_fruit = 56 →
  oranges = total_fruit / 4 →
  peaches = oranges / 2 →
  apples = 5 * peaches →
  apples = 35 := by
sorry

end NUMINAMATH_CALUDE_fruit_box_problem_l3230_323072


namespace NUMINAMATH_CALUDE_milk_fraction_after_transfers_l3230_323055

/-- Represents the contents of a cup --/
structure CupContents where
  tea : ℚ
  milk : ℚ

/-- Represents the problem setup --/
def initial_setup : CupContents × CupContents :=
  ({ tea := 8, milk := 0 }, { tea := 0, milk := 8 })

/-- Transfers a fraction of tea from the first cup to the second --/
def transfer_tea (cups : CupContents × CupContents) (fraction : ℚ) : CupContents × CupContents :=
  let (cup1, cup2) := cups
  let transfer_amount := cup1.tea * fraction
  ({ tea := cup1.tea - transfer_amount, milk := cup1.milk },
   { tea := cup2.tea + transfer_amount, milk := cup2.milk })

/-- Transfers a fraction of the mixture from the second cup to the first --/
def transfer_mixture (cups : CupContents × CupContents) (fraction : ℚ) : CupContents × CupContents :=
  let (cup1, cup2) := cups
  let total2 := cup2.tea + cup2.milk
  let transfer_tea := cup2.tea * fraction
  let transfer_milk := cup2.milk * fraction
  ({ tea := cup1.tea + transfer_tea, milk := cup1.milk + transfer_milk },
   { tea := cup2.tea - transfer_tea, milk := cup2.milk - transfer_milk })

/-- Calculates the fraction of milk in a cup --/
def milk_fraction (cup : CupContents) : ℚ :=
  cup.milk / (cup.tea + cup.milk)

/-- The main theorem to prove --/
theorem milk_fraction_after_transfers :
  let cups1 := transfer_tea initial_setup (1/4)
  let cups2 := transfer_mixture cups1 (1/3)
  milk_fraction cups2.fst = 1/3 := by sorry


end NUMINAMATH_CALUDE_milk_fraction_after_transfers_l3230_323055


namespace NUMINAMATH_CALUDE_sum_of_squares_l3230_323017

theorem sum_of_squares (a b c d e f : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0)
  (h1 : b * c * d * e * f / a = 1 / 2)
  (h2 : a * c * d * e * f / b = 1 / 4)
  (h3 : a * b * d * e * f / c = 1 / 8)
  (h4 : a * b * c * e * f / d = 2)
  (h5 : a * b * c * d * f / e = 4)
  (h6 : a * b * c * d * e / f = 8) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 119 / 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3230_323017


namespace NUMINAMATH_CALUDE_sector_central_angle_l3230_323067

/-- Given a sector with perimeter 6 and area 2, its central angle in radians is either 4 or 1 -/
theorem sector_central_angle (r l : ℝ) : 
  2 * r + l = 6 →
  1 / 2 * l * r = 2 →
  l / r = 4 ∨ l / r = 1 :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3230_323067


namespace NUMINAMATH_CALUDE_washing_machine_capacity_l3230_323035

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

end NUMINAMATH_CALUDE_washing_machine_capacity_l3230_323035


namespace NUMINAMATH_CALUDE_election_combinations_l3230_323082

/-- Represents the gender of a club member -/
inductive Gender
| Male
| Female

/-- Represents the club with its members -/
structure Club where
  total_members : Nat
  male_members : Nat
  female_members : Nat
  members_sum : total_members = male_members + female_members

/-- Represents the election positions -/
structure Election where
  president : Nat
  vice_president : Nat
  secretary : Nat

/-- Function to calculate the number of valid election combinations -/
def count_valid_elections (club : Club) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem election_combinations (club : Club)
  (h1 : club.total_members = 30)
  (h2 : club.male_members = 15)
  (h3 : club.female_members = 15)
  : count_valid_elections club = 6300 := by
  sorry

end NUMINAMATH_CALUDE_election_combinations_l3230_323082


namespace NUMINAMATH_CALUDE_sin_two_pi_thirds_l3230_323087

theorem sin_two_pi_thirds : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_pi_thirds_l3230_323087


namespace NUMINAMATH_CALUDE_box_height_is_nine_l3230_323030

/-- A rectangular box containing spheres -/
structure SphereBox where
  height : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ
  large_sphere_count : ℕ
  small_sphere_count : ℕ

/-- The specific box described in the problem -/
def problem_box : SphereBox :=
  { height := 9,
    large_sphere_radius := 3,
    small_sphere_radius := 1.5,
    large_sphere_count := 1,
    small_sphere_count := 8 }

/-- Theorem stating that the height of the box must be 9 -/
theorem box_height_is_nine (box : SphereBox) :
  box.height = 9 ∧
  box.large_sphere_radius = 3 ∧
  box.small_sphere_radius = 1.5 ∧
  box.large_sphere_count = 1 ∧
  box.small_sphere_count = 8 →
  box = problem_box :=
sorry

end NUMINAMATH_CALUDE_box_height_is_nine_l3230_323030


namespace NUMINAMATH_CALUDE_survey_result_l3230_323009

/-- The percentage of parents who agree to the tuition fee increase -/
def agree_percentage : ℝ := 0.20

/-- The number of parents who disagree with the tuition fee increase -/
def disagree_count : ℕ := 640

/-- The total number of parents surveyed -/
def total_parents : ℕ := 800

/-- Theorem stating that the total number of parents surveyed is 800 -/
theorem survey_result : total_parents = 800 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l3230_323009


namespace NUMINAMATH_CALUDE_ratio_equality_l3230_323002

theorem ratio_equality : ∃ x : ℚ, (x / (2/5)) = ((3/7) / (6/5)) ∧ x = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3230_323002


namespace NUMINAMATH_CALUDE_smallest_sum_mn_smallest_sum_is_60_l3230_323042

theorem smallest_sum_mn (m n : ℕ) (h : 3 * n^3 = 5 * m^2) : 
  ∀ (a b : ℕ), 3 * a^3 = 5 * b^2 → m + n ≤ a + b :=
by sorry

theorem smallest_sum_is_60 : 
  ∃ (m n : ℕ), 3 * n^3 = 5 * m^2 ∧ m + n = 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_mn_smallest_sum_is_60_l3230_323042


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_2_range_of_m_for_inequality_l3230_323097

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - 2*m| - |x + m|

-- Part 1
theorem solution_set_when_m_is_2 :
  let m : ℝ := 2
  ∀ x : ℝ, f m x ≥ 1 ↔ -2 < x ∧ x ≤ 1/2 :=
sorry

-- Part 2
theorem range_of_m_for_inequality :
  ∀ m : ℝ, m > 0 →
  (∀ x t : ℝ, f m x ≤ |t + 3| + |t - 2|) ↔
  (0 < m ∧ m ≤ 5/3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_2_range_of_m_for_inequality_l3230_323097


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l3230_323071

theorem product_zero_implies_factor_zero (a b : ℝ) (h : a * b = 0) :
  a = 0 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l3230_323071


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l3230_323024

theorem triangle_sine_inequality (a b c : Real) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h3 : a + b + c ≤ 2 * Real.pi) : 
  Real.sin a + Real.sin b > Real.sin c ∧
  Real.sin b + Real.sin c > Real.sin a ∧
  Real.sin c + Real.sin a > Real.sin b :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l3230_323024


namespace NUMINAMATH_CALUDE_range_of_f_l3230_323068

-- Define the function
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of the function
theorem range_of_f :
  ∀ y : ℝ, (y ≥ -8) ↔ ∃ x : ℝ, f x = y :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l3230_323068


namespace NUMINAMATH_CALUDE_events_independent_prob_A_or_B_l3230_323052

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The set of all ball numbers -/
def ball_numbers : Finset ℕ := Finset.range total_balls

/-- Event A: selecting a ball with an odd number -/
def event_A : Finset ℕ := ball_numbers.filter (λ n => n % 2 = 1)

/-- Event B: selecting a ball with a number that is a multiple of 3 -/
def event_B : Finset ℕ := ball_numbers.filter (λ n => n % 3 = 0)

/-- The probability of an event occurring -/
def prob (event : Finset ℕ) : ℚ := (event.card : ℚ) / total_balls

/-- The intersection of events A and B -/
def event_AB : Finset ℕ := event_A ∩ event_B

/-- Theorem: Events A and B are independent -/
theorem events_independent : prob event_AB = prob event_A * prob event_B := by sorry

/-- Theorem: The probability of A or B occurring is 5/8 -/
theorem prob_A_or_B : prob (event_A ∪ event_B) = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_events_independent_prob_A_or_B_l3230_323052
