import Mathlib

namespace NUMINAMATH_CALUDE_largest_five_digit_code_l1818_181846

def is_power_of_5 (n : Nat) : Prop :=
  ∃ k : Nat, n = 5^k

def is_power_of_2 (n : Nat) : Prop :=
  ∃ k : Nat, n = 2^k

def is_multiple_of_3 (n : Nat) : Prop :=
  ∃ k : Nat, n = 3 * k

def digits_sum (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def has_unique_digits (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

theorem largest_five_digit_code : 
  ∀ n : Nat,
  n ≤ 99999 ∧
  n ≥ 10000 ∧
  (∀ d : Nat, d ∈ n.digits 10 → d ≠ 0) ∧
  is_power_of_5 (n / 1000) ∧
  is_power_of_2 (n % 100) ∧
  is_multiple_of_3 ((n / 100) % 10) ∧
  Odd (digits_sum n) ∧
  has_unique_digits n
  →
  n ≤ 25916 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_code_l1818_181846


namespace NUMINAMATH_CALUDE_fertilizer_production_range_l1818_181829

/-- Represents the production capacity and constraints of a fertilizer plant --/
structure FertilizerPlant where
  maxWorkers : Nat
  maxHoursPerWorker : Nat
  minExpectedSales : Nat
  hoursPerBag : Nat
  rawMaterialPerBag : Nat
  initialRawMaterial : Nat
  usedRawMaterial : Nat
  supplementedRawMaterial : Nat

/-- Calculates the maximum number of bags that can be produced given the plant's constraints --/
def maxBagsProduced (plant : FertilizerPlant) : Nat :=
  min
    (plant.maxWorkers * plant.maxHoursPerWorker / plant.hoursPerBag)
    ((plant.initialRawMaterial - plant.usedRawMaterial + plant.supplementedRawMaterial) * 1000 / plant.rawMaterialPerBag)

/-- Theorem stating the range of bags that can be produced --/
theorem fertilizer_production_range (plant : FertilizerPlant)
  (h1 : plant.maxWorkers = 200)
  (h2 : plant.maxHoursPerWorker = 2100)
  (h3 : plant.minExpectedSales = 80000)
  (h4 : plant.hoursPerBag = 4)
  (h5 : plant.rawMaterialPerBag = 20)
  (h6 : plant.initialRawMaterial = 800)
  (h7 : plant.usedRawMaterial = 200)
  (h8 : plant.supplementedRawMaterial = 1200) :
  80000 ≤ maxBagsProduced plant ∧ maxBagsProduced plant = 90000 := by
  sorry

#check fertilizer_production_range

end NUMINAMATH_CALUDE_fertilizer_production_range_l1818_181829


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_l1818_181862

/-- A type representing a point on a circle -/
def CirclePoint : Type := Nat

/-- The total number of points on the circle -/
def total_points : Nat := 2023

/-- A type representing a selection of 6 distinct points -/
def Sextuple : Type := Fin 6 → CirclePoint

/-- Predicate to check if two chords intersect -/
def chords_intersect (a b c d : CirclePoint) : Prop := sorry

/-- The probability of selecting a sextuple where AB intersects both CD and EF -/
def intersecting_chords_probability : ℚ := 1 / 72

theorem intersecting_chords_theorem (s : Sextuple) :
  (∀ i j : Fin 6, i ≠ j → s i ≠ s j) →  -- all points are distinct
  (∀ s : Sextuple, (∀ i j : Fin 6, i ≠ j → s i ≠ s j) → 
    chords_intersect (s 0) (s 1) (s 2) (s 3) ∧ 
    chords_intersect (s 2) (s 3) (s 4) (s 5)) →  -- definition of intersecting chords
  intersecting_chords_probability = 1 / 72 := by sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_l1818_181862


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1818_181828

theorem algebraic_expression_value (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (sum_condition : a + b + c + d = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 45) :
  (a^5 / ((a-b)*(a-c)*(a-d))) + (b^5 / ((b-a)*(b-c)*(b-d))) + 
  (c^5 / ((c-a)*(c-b)*(c-d))) + (d^5 / ((d-a)*(d-b)*(d-c))) = -9 :=
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1818_181828


namespace NUMINAMATH_CALUDE_square_root_of_four_l1818_181890

theorem square_root_of_four : ∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l1818_181890


namespace NUMINAMATH_CALUDE_sine_function_vertical_shift_l1818_181811

/-- Given a sine function y = a * sin(b * x) + d with positive constants a, b, and d,
    if the maximum value of y is 4 and the minimum value of y is -2, then d = 1. -/
theorem sine_function_vertical_shift 
  (a b d : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hd : d > 0) 
  (hmax : ∀ x, a * Real.sin (b * x) + d ≤ 4)
  (hmin : ∀ x, a * Real.sin (b * x) + d ≥ -2)
  (hex_max : ∃ x, a * Real.sin (b * x) + d = 4)
  (hex_min : ∃ x, a * Real.sin (b * x) + d = -2) : 
  d = 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_vertical_shift_l1818_181811


namespace NUMINAMATH_CALUDE_quarters_percentage_is_fifty_percent_l1818_181883

/-- The number of dimes -/
def num_dimes : ℕ := 50

/-- The number of quarters -/
def num_quarters : ℕ := 20

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of all coins in cents -/
def total_value : ℕ := num_dimes * dime_value + num_quarters * quarter_value

/-- The value of quarters in cents -/
def quarters_value : ℕ := num_quarters * quarter_value

/-- Theorem stating that the percentage of the total value in quarters is 50% -/
theorem quarters_percentage_is_fifty_percent :
  (quarters_value : ℚ) / (total_value : ℚ) * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_quarters_percentage_is_fifty_percent_l1818_181883


namespace NUMINAMATH_CALUDE_qingming_festival_probability_l1818_181875

/-- Represents the number of students in each grade --/
structure GradeDistribution where
  grade7 : ℕ
  grade8 : ℕ
  grade9 : ℕ

/-- Represents the participation methods for each grade --/
structure ParticipationMethods where
  memorial_hall : GradeDistribution
  online : GradeDistribution

/-- The main theorem to prove --/
theorem qingming_festival_probability 
  (total_students : GradeDistribution)
  (participation : ParticipationMethods)
  (h1 : total_students.grade7 = 4 * k)
  (h2 : total_students.grade8 = 5 * k)
  (h3 : total_students.grade9 = 6 * k)
  (h4 : participation.memorial_hall.grade7 = 2 * a - 1)
  (h5 : participation.memorial_hall.grade8 = 8)
  (h6 : participation.memorial_hall.grade9 = 10)
  (h7 : participation.online.grade7 = a)
  (h8 : participation.online.grade8 = b)
  (h9 : participation.online.grade9 = 2)
  (h10 : total_students.grade7 = participation.memorial_hall.grade7 + participation.online.grade7)
  (h11 : total_students.grade8 = participation.memorial_hall.grade8 + participation.online.grade8)
  (h12 : total_students.grade9 = participation.memorial_hall.grade9 + participation.online.grade9)
  : ℚ :=
  5/21

/-- Auxiliary function to calculate combinations --/
def combinations (n : ℕ) (r : ℕ) : ℕ := sorry

#check qingming_festival_probability

end NUMINAMATH_CALUDE_qingming_festival_probability_l1818_181875


namespace NUMINAMATH_CALUDE_pencil_cost_is_25_l1818_181860

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := sorry

/-- The cost of a pen in cents -/
def pen_cost : ℕ := 80

/-- The total number of items (pens and pencils) bought -/
def total_items : ℕ := 36

/-- The number of pencils bought -/
def pencils_bought : ℕ := 16

/-- The total amount spent in cents -/
def total_spent : ℕ := 2000  -- 20 dollars = 2000 cents

theorem pencil_cost_is_25 : 
  pencil_cost = 25 ∧ 
  pencil_cost * pencils_bought + pen_cost * (total_items - pencils_bought) = total_spent :=
sorry

end NUMINAMATH_CALUDE_pencil_cost_is_25_l1818_181860


namespace NUMINAMATH_CALUDE_eva_total_marks_l1818_181841

def eva_marks (maths_second science_second arts_second : ℕ) : Prop :=
  let maths_first := maths_second + 10
  let arts_first := arts_second - 15
  let science_first := science_second - (science_second / 3)
  let total_first := maths_first + arts_first + science_first
  let total_second := maths_second + science_second + arts_second
  total_first + total_second = 485

theorem eva_total_marks :
  eva_marks 80 90 90 := by sorry

end NUMINAMATH_CALUDE_eva_total_marks_l1818_181841


namespace NUMINAMATH_CALUDE_relationship_holds_l1818_181876

/-- A function representing the relationship between x and y -/
def f (x : ℕ) : ℕ := x^2 + 3*x + 1

/-- The set of x values given in the problem -/
def X : Finset ℕ := {1, 2, 3, 4, 5}

/-- The set of y values given in the problem -/
def Y : Finset ℕ := {5, 11, 19, 29, 41}

/-- Theorem stating that the function f correctly relates all given x and y values -/
theorem relationship_holds : ∀ x ∈ X, f x ∈ Y :=
  sorry

end NUMINAMATH_CALUDE_relationship_holds_l1818_181876


namespace NUMINAMATH_CALUDE_donut_distribution_l1818_181847

/-- The number of ways to distribute n indistinguishable objects into k distinguishable categories,
    with at least one object in each category. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem stating that there are 35 ways to distribute 8 donuts into 5 varieties
    with at least one donut of each variety. -/
theorem donut_distribution : distribute 8 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_donut_distribution_l1818_181847


namespace NUMINAMATH_CALUDE_age_ratio_correct_l1818_181840

/-- Represents the ages and relationship between a mother and daughter -/
structure FamilyAges where
  mother_current_age : ℕ
  daughter_future_age : ℕ
  years_to_future : ℕ
  multiple : ℝ

/-- Calculates the ratio of mother's age to daughter's age at a past time -/
def age_ratio (f : FamilyAges) : ℝ × ℝ :=
  (f.multiple, 1)

/-- Theorem stating that the age ratio is correct given the family ages -/
theorem age_ratio_correct (f : FamilyAges) 
  (h1 : f.mother_current_age = 41)
  (h2 : f.daughter_future_age = 26)
  (h3 : f.years_to_future = 3)
  (h4 : ∃ (x : ℕ), f.mother_current_age - x = f.multiple * (f.daughter_future_age - f.years_to_future - x)) :
  age_ratio f = (f.multiple, 1) := by
  sorry

#check age_ratio_correct

end NUMINAMATH_CALUDE_age_ratio_correct_l1818_181840


namespace NUMINAMATH_CALUDE_total_with_tax_calculation_l1818_181853

def total_before_tax : ℝ := 150
def sales_tax_rate : ℝ := 0.08

theorem total_with_tax_calculation :
  total_before_tax * (1 + sales_tax_rate) = 162 := by
  sorry

end NUMINAMATH_CALUDE_total_with_tax_calculation_l1818_181853


namespace NUMINAMATH_CALUDE_prices_and_schemes_l1818_181885

def soccer_ball_price : ℕ := 60
def basketball_price : ℕ := 80

def initial_purchase_cost : ℕ := 1600
def initial_soccer_balls : ℕ := 8
def initial_basketballs : ℕ := 14

def total_balls : ℕ := 50
def min_budget : ℕ := 3200
def max_budget : ℕ := 3240

theorem prices_and_schemes :
  (initial_soccer_balls * soccer_ball_price + initial_basketballs * basketball_price = initial_purchase_cost) ∧
  (basketball_price = soccer_ball_price + 20) ∧
  (∀ y : ℕ, y ≤ total_balls →
    (y * soccer_ball_price + (total_balls - y) * basketball_price ≥ min_budget ∧
     y * soccer_ball_price + (total_balls - y) * basketball_price ≤ max_budget)
    ↔ (y = 38 ∨ y = 39 ∨ y = 40)) :=
by sorry

end NUMINAMATH_CALUDE_prices_and_schemes_l1818_181885


namespace NUMINAMATH_CALUDE_corey_candy_count_l1818_181872

/-- Given that Tapanga and Corey have a total of 66 candies, and Tapanga has 8 more candies than Corey,
    prove that Corey has 29 candies. -/
theorem corey_candy_count (total : ℕ) (difference : ℕ) (corey : ℕ) : 
  total = 66 → difference = 8 → total = corey + (corey + difference) → corey = 29 := by
  sorry

end NUMINAMATH_CALUDE_corey_candy_count_l1818_181872


namespace NUMINAMATH_CALUDE_first_group_size_l1818_181857

def work_rate (people : ℕ) (time : ℕ) : ℚ := 1 / (people * time)

theorem first_group_size :
  ∀ (p : ℕ),
  (work_rate p 60 = work_rate 16 30) →
  p = 8 := by
sorry

end NUMINAMATH_CALUDE_first_group_size_l1818_181857


namespace NUMINAMATH_CALUDE_jermaine_earnings_difference_l1818_181877

def total_earnings : ℕ := 90
def terrence_earnings : ℕ := 30
def emilee_earnings : ℕ := 25

theorem jermaine_earnings_difference : 
  ∃ (jermaine_earnings : ℕ), 
    jermaine_earnings > terrence_earnings ∧
    jermaine_earnings + terrence_earnings + emilee_earnings = total_earnings ∧
    jermaine_earnings - terrence_earnings = 5 :=
by sorry

end NUMINAMATH_CALUDE_jermaine_earnings_difference_l1818_181877


namespace NUMINAMATH_CALUDE_potato_price_correct_l1818_181842

/-- The price of potatoes per kilo -/
def potato_price : ℝ := 2

theorem potato_price_correct (
  initial_amount : ℝ)
  (potato_kilos : ℝ)
  (tomato_kilos : ℝ)
  (cucumber_kilos : ℝ)
  (banana_kilos : ℝ)
  (tomato_price : ℝ)
  (cucumber_price : ℝ)
  (banana_price : ℝ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 500)
  (h2 : potato_kilos = 6)
  (h3 : tomato_kilos = 9)
  (h4 : cucumber_kilos = 5)
  (h5 : banana_kilos = 3)
  (h6 : tomato_price = 3)
  (h7 : cucumber_price = 4)
  (h8 : banana_price = 5)
  (h9 : remaining_amount = 426)
  (h10 : initial_amount - (potato_kilos * potato_price + tomato_kilos * tomato_price + 
         cucumber_kilos * cucumber_price + banana_kilos * banana_price) = remaining_amount) :
  potato_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_potato_price_correct_l1818_181842


namespace NUMINAMATH_CALUDE_power_of_two_geq_n_l1818_181855

theorem power_of_two_geq_n (n : ℕ) (h : n ≥ 1) : 2^n ≥ n := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_geq_n_l1818_181855


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l1818_181850

/-- If the cost price of 29 articles equals the selling price of 24 articles,
    then the percentage of profit is 5/24 * 100. -/
theorem merchant_profit_percentage (C S : ℝ) (h : 29 * C = 24 * S) :
  (S - C) / C * 100 = 5 / 24 * 100 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l1818_181850


namespace NUMINAMATH_CALUDE_sector_area_l1818_181804

/-- Given a sector with central angle 2 radians and arc length 4, its area is equal to 4 -/
theorem sector_area (θ : Real) (L : Real) (r : Real) (A : Real) : 
  θ = 2 → L = 4 → L = θ * r → A = 1/2 * θ * r^2 → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1818_181804


namespace NUMINAMATH_CALUDE_book_price_proof_l1818_181803

theorem book_price_proof (P V T : ℝ) 
  (vasya_short : V + 150 = P)
  (tolya_short : T + 200 = P)
  (exchange_scenario : V + T / 2 - P = 100) : 
  P = 700 := by
  sorry

end NUMINAMATH_CALUDE_book_price_proof_l1818_181803


namespace NUMINAMATH_CALUDE_expression_equality_l1818_181806

theorem expression_equality : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1818_181806


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1818_181861

/-- For a parabola with equation y = 4x^2, the distance from the focus to the directrix is 1/8 -/
theorem parabola_focus_directrix_distance :
  let parabola := fun (x : ℝ) => 4 * x^2
  ∃ (focus : ℝ × ℝ) (directrix : ℝ → ℝ),
    (∀ x, parabola x = (x - focus.1)^2 / (4 * (focus.2 - directrix 0))) ∧
    (focus.2 - directrix 0 = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1818_181861


namespace NUMINAMATH_CALUDE_square_area_is_26_l1818_181879

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The square defined by its four vertices -/
structure Square where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Calculate the area of a square given its vertices -/
def squareArea (sq : Square) : ℝ := 
  let dx := sq.p.x - sq.q.x
  let dy := sq.p.y - sq.q.y
  (dx * dx + dy * dy)

/-- The theorem stating that the area of the given square is 26 -/
theorem square_area_is_26 : 
  let sq := Square.mk 
    (Point.mk 2 3) 
    (Point.mk (-3) 4) 
    (Point.mk (-2) 9) 
    (Point.mk 3 8)
  squareArea sq = 26 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_26_l1818_181879


namespace NUMINAMATH_CALUDE_problem_solution_l1818_181880

def f (x : ℝ) : ℝ := x^2 + 10

def g (x : ℝ) : ℝ := x^2 - 6

theorem problem_solution (a : ℝ) (h1 : a > 3) (h2 : f (g a) = 16) : a = Real.sqrt (Real.sqrt 6 + 6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1818_181880


namespace NUMINAMATH_CALUDE_max_value_implies_m_value_l1818_181892

def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem max_value_implies_m_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-2) 2, f m x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-2) 2, f m x = 3) →
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_m_value_l1818_181892


namespace NUMINAMATH_CALUDE_train_length_l1818_181896

/-- Given a train traveling at 45 km/hour that passes a 140-meter bridge in 56 seconds,
    prove that the length of the train is 560 meters. -/
theorem train_length (speed : ℝ) (bridge_length : ℝ) (time : ℝ) :
  speed = 45 → bridge_length = 140 → time = 56 →
  speed * (1000 / 3600) * time - bridge_length = 560 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1818_181896


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l1818_181866

theorem solution_set_reciprocal_inequality (x : ℝ) :
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l1818_181866


namespace NUMINAMATH_CALUDE_pump_fill_time_pump_fill_time_proof_l1818_181817

theorem pump_fill_time (fill_time_with_leak : ℚ) (leak_drain_time : ℕ) : ℚ :=
  let fill_rate_with_leak : ℚ := 1 / fill_time_with_leak
  let leak_rate : ℚ := 1 / leak_drain_time
  let pump_rate : ℚ := fill_rate_with_leak + leak_rate
  1 / pump_rate

theorem pump_fill_time_proof :
  pump_fill_time (13/6) 26 = 2 := by sorry

end NUMINAMATH_CALUDE_pump_fill_time_pump_fill_time_proof_l1818_181817


namespace NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_over_six_l1818_181830

theorem arcsin_one_half_equals_pi_over_six : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_over_six_l1818_181830


namespace NUMINAMATH_CALUDE_solution_set_correct_l1818_181812

/-- The solution set of the inequality -x^2 + 2x > 0 -/
def SolutionSet : Set ℝ := {x | 0 < x ∧ x < 2}

/-- Theorem stating that SolutionSet is the correct solution to the inequality -/
theorem solution_set_correct :
  ∀ x : ℝ, x ∈ SolutionSet ↔ -x^2 + 2*x > 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_correct_l1818_181812


namespace NUMINAMATH_CALUDE_dolphin_training_ratio_l1818_181835

theorem dolphin_training_ratio (total : ℕ) (fully_trained_fraction : ℚ) (to_be_trained : ℕ) :
  total = 20 →
  fully_trained_fraction = 1/4 →
  to_be_trained = 5 →
  (total - (total * fully_trained_fraction).num - to_be_trained : ℚ) / to_be_trained = 2 := by
sorry

end NUMINAMATH_CALUDE_dolphin_training_ratio_l1818_181835


namespace NUMINAMATH_CALUDE_correct_remaining_insects_l1818_181818

/-- Calculates the number of remaining insects in the playground -/
def remaining_insects (spiders ants ladybugs flown_away : ℕ) : ℕ :=
  spiders + ants + ladybugs - flown_away

/-- Theorem stating that the number of remaining insects is correct -/
theorem correct_remaining_insects :
  remaining_insects 3 12 8 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_remaining_insects_l1818_181818


namespace NUMINAMATH_CALUDE_equal_intercept_line_equations_l1818_181871

/-- A line with equal absolute intercepts passing through (3, -2) -/
structure EqualInterceptLine where
  -- The slope of the line
  m : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (3, -2)
  point_condition : -2 = m * 3 + b
  -- The line has equal absolute intercepts
  intercept_condition : ∃ (k : ℝ), k ≠ 0 ∧ (k = -b/m ∨ k = b)

/-- The possible equations of the line -/
def possible_equations (l : EqualInterceptLine) : Prop :=
  (2 * l.m + 3 = 0 ∧ l.b = 0) ∨
  (l.m = -1 ∧ l.b = 1) ∨
  (l.m = 1 ∧ l.b = 5)

theorem equal_intercept_line_equations :
  ∀ (l : EqualInterceptLine), possible_equations l :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equations_l1818_181871


namespace NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l1818_181856

/-- The number of chocolate bars in the large box -/
def total_chocolate_bars (num_small_boxes : ℕ) (bars_per_small_box : ℕ) : ℕ :=
  num_small_boxes * bars_per_small_box

/-- Theorem stating the total number of chocolate bars in the large box -/
theorem chocolate_bars_in_large_box :
  total_chocolate_bars 20 32 = 640 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l1818_181856


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1818_181845

theorem inequality_equivalence (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (c - a) * (c - b) * (b - a) < 0 ↔ b*c^2 + c*a^2 + a*b^2 < b^2*c + c^2*a + a^2*b :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1818_181845


namespace NUMINAMATH_CALUDE_lcm_5_6_10_12_l1818_181864

theorem lcm_5_6_10_12 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 12)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_6_10_12_l1818_181864


namespace NUMINAMATH_CALUDE_larger_number_proof_l1818_181833

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 5) : L = 1637 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1818_181833


namespace NUMINAMATH_CALUDE_inequality_proof_l1818_181897

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a^2 + 1 / b^2 + a * b ≥ 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1818_181897


namespace NUMINAMATH_CALUDE_probability_greater_than_400_probability_even_l1818_181819

def digits : List ℕ := [1, 5, 6]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (∀ d, d ∈ digits → (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d))

def valid_numbers : List ℕ := [156, 165, 516, 561, 615, 651]

theorem probability_greater_than_400 :
  (valid_numbers.filter (λ n => n > 400)).length / valid_numbers.length = 2 / 3 := by sorry

theorem probability_even :
  (valid_numbers.filter (λ n => n % 2 = 0)).length / valid_numbers.length = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_greater_than_400_probability_even_l1818_181819


namespace NUMINAMATH_CALUDE_arcsin_arccos_eq_arctan_pi_fourth_l1818_181821

theorem arcsin_arccos_eq_arctan_pi_fourth :
  ∃ x : ℝ, x = 0 ∧ Real.arcsin x + Real.arccos (1 - x) = Real.arctan x + π / 4 :=
by sorry

end NUMINAMATH_CALUDE_arcsin_arccos_eq_arctan_pi_fourth_l1818_181821


namespace NUMINAMATH_CALUDE_park_area_l1818_181815

/-- The area of a rectangular park with perimeter 80 meters and length three times the width is 300 square meters. -/
theorem park_area (width length : ℝ) (h_perimeter : 2 * (width + length) = 80) (h_length : length = 3 * width) :
  width * length = 300 :=
sorry

end NUMINAMATH_CALUDE_park_area_l1818_181815


namespace NUMINAMATH_CALUDE_qt_plus_q_plus_t_not_two_l1818_181825

theorem qt_plus_q_plus_t_not_two :
  ∀ q t : ℕ+, q * t + q + t ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_qt_plus_q_plus_t_not_two_l1818_181825


namespace NUMINAMATH_CALUDE_molecular_weight_AlPO4_correct_l1818_181859

/-- The molecular weight of AlPO4 in grams per mole -/
def molecular_weight_AlPO4 : ℝ := 122

/-- The number of moles given in the problem -/
def moles : ℝ := 4

/-- The total weight of the given moles of AlPO4 in grams -/
def total_weight : ℝ := 488

/-- Theorem: The molecular weight of AlPO4 is correct given the total weight of 4 moles -/
theorem molecular_weight_AlPO4_correct : 
  molecular_weight_AlPO4 * moles = total_weight :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_AlPO4_correct_l1818_181859


namespace NUMINAMATH_CALUDE_system_of_inequalities_l1818_181873

theorem system_of_inequalities (x : ℝ) :
  (2 * x ≤ 6 - x) ∧ (3 * x + 1 > 2 * (x - 1)) → -3 < x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l1818_181873


namespace NUMINAMATH_CALUDE_vector_inequality_l1818_181843

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable (hd : FiniteDimensional.finrank ℝ V = 2)

/-- Given four vectors a, b, c, d in a 2D real vector space such that their sum is zero,
    prove that the sum of their norms is greater than or equal to the sum of the norms
    of their pairwise sums with d. -/
theorem vector_inequality (a b c d : V) (h : a + b + c + d = 0) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖d‖ ≥ ‖a + d‖ + ‖b + d‖ + ‖c + d‖ :=
sorry

end NUMINAMATH_CALUDE_vector_inequality_l1818_181843


namespace NUMINAMATH_CALUDE_impossible_arrangement_l1818_181891

theorem impossible_arrangement (n : Nat) (h : n = 2002) : 
  ¬ ∃ (A : Fin n → Fin n → Fin (n^2)),
    (∀ i j : Fin n, A i j < n^2) ∧ 
    (∀ i j : Fin n, ∃ k₁ k₂ : Fin n, 
      (A i k₁ * A i k₂ * A i j ≤ n^2 ∨ A k₁ j * A k₂ j * A i j ≤ n^2)) ∧
    (∀ x : Fin (n^2), ∃ i j : Fin n, A i j = x) := by
  sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l1818_181891


namespace NUMINAMATH_CALUDE_largest_fraction_l1818_181848

theorem largest_fraction : 
  let fractions := [5/11, 9/20, 23/47, 105/209, 205/409]
  ∀ x ∈ fractions, (105 : ℚ) / 209 ≥ x := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l1818_181848


namespace NUMINAMATH_CALUDE_expression_evaluation_l1818_181858

theorem expression_evaluation : -3^2 + (-12) * |-(1/2)| - 6 / (-1) = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1818_181858


namespace NUMINAMATH_CALUDE_conjugate_sum_product_l1818_181886

theorem conjugate_sum_product (c d : ℝ) :
  (c + Real.sqrt d + (c - Real.sqrt d) = -8) →
  ((c + Real.sqrt d) * (c - Real.sqrt d) = 4) →
  c + d = 8 := by
sorry

end NUMINAMATH_CALUDE_conjugate_sum_product_l1818_181886


namespace NUMINAMATH_CALUDE_existence_of_m_n_l1818_181801

theorem existence_of_m_n (p : ℕ) (hp : p.Prime) (hp_gt_10 : p > 10) :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m + n < p ∧ (5^m * 7^n - 1) % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l1818_181801


namespace NUMINAMATH_CALUDE_problem_solution_l1818_181878

theorem problem_solution : (-1)^2023 + |Real.sqrt 3 - 3| + Real.sqrt 9 - (-4) * (1/2) = 7 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1818_181878


namespace NUMINAMATH_CALUDE_two_special_numbers_l1818_181870

/-- A three-digit number divisible by 5 that can be represented as n^3 + n^2 -/
def special_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧  -- three-digit number
  x % 5 = 0 ∧           -- divisible by 5
  ∃ n : ℕ, x = n^3 + n^2  -- can be represented as n^3 + n^2

/-- There are exactly two numbers satisfying the special_number property -/
theorem two_special_numbers : ∃! (a b : ℕ), a ≠ b ∧ special_number a ∧ special_number b ∧
  ∀ x, special_number x → (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_two_special_numbers_l1818_181870


namespace NUMINAMATH_CALUDE_gcd_values_count_l1818_181838

theorem gcd_values_count (a b : ℕ+) (h : Nat.gcd a.val b.val * Nat.lcm a.val b.val = 180) :
  ∃ S : Finset ℕ+, (∀ x ∈ S, x = Nat.gcd a.val b.val) ∧ S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_values_count_l1818_181838


namespace NUMINAMATH_CALUDE_red_beads_count_l1818_181820

/-- The total number of beads in the string -/
def total_beads : ℕ := 85

/-- The number of green beads in one pattern cycle -/
def green_in_cycle : ℕ := 3

/-- The number of red beads in one pattern cycle -/
def red_in_cycle : ℕ := 4

/-- The number of yellow beads in one pattern cycle -/
def yellow_in_cycle : ℕ := 1

/-- The total number of beads in one pattern cycle -/
def beads_per_cycle : ℕ := green_in_cycle + red_in_cycle + yellow_in_cycle

/-- The number of complete cycles in the string -/
def complete_cycles : ℕ := total_beads / beads_per_cycle

/-- The number of beads remaining after complete cycles -/
def remaining_beads : ℕ := total_beads % beads_per_cycle

/-- The number of red beads in the remaining portion -/
def red_in_remaining : ℕ := min remaining_beads (red_in_cycle)

/-- Theorem: The total number of red beads in the string is 42 -/
theorem red_beads_count : 
  complete_cycles * red_in_cycle + red_in_remaining = 42 := by
sorry

end NUMINAMATH_CALUDE_red_beads_count_l1818_181820


namespace NUMINAMATH_CALUDE_triangle_perimeter_upper_bound_l1818_181831

theorem triangle_perimeter_upper_bound (a b c : ℝ) : 
  a = 8 → b = 15 → a + b > c → a + c > b → b + c > a → 
  ∃ n : ℕ, n = 46 ∧ ∀ m : ℕ, (m : ℝ) > a + b + c → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_upper_bound_l1818_181831


namespace NUMINAMATH_CALUDE_train_car_count_l1818_181813

theorem train_car_count (total_cars : ℕ) (passenger_cars : ℕ) (cargo_cars : ℕ) : 
  total_cars = 71 →
  cargo_cars = passenger_cars / 2 + 3 →
  total_cars = passenger_cars + cargo_cars + 2 →
  passenger_cars = 44 := by
sorry

end NUMINAMATH_CALUDE_train_car_count_l1818_181813


namespace NUMINAMATH_CALUDE_dart_board_probability_l1818_181834

/-- The probability of a dart landing within the center square of a regular hexagonal dart board -/
theorem dart_board_probability (s : ℝ) (h : s > 0) : 
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let square_area := s^2
  square_area / hexagon_area = 2 * Real.sqrt 3 / 9 := by sorry

end NUMINAMATH_CALUDE_dart_board_probability_l1818_181834


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1818_181865

open Set

universe u

theorem complement_intersection_theorem (U M N : Set ℕ) :
  U = {0, 1, 2, 3, 4} →
  M = {0, 1, 2} →
  N = {2, 3} →
  (U \ M) ∩ N = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1818_181865


namespace NUMINAMATH_CALUDE_walk_distance_difference_l1818_181868

theorem walk_distance_difference (total_distance susan_distance : ℕ) 
  (h1 : total_distance = 15)
  (h2 : susan_distance = 9) :
  susan_distance - (total_distance - susan_distance) = 3 :=
by sorry

end NUMINAMATH_CALUDE_walk_distance_difference_l1818_181868


namespace NUMINAMATH_CALUDE_service_center_location_l1818_181827

/-- The location of the service center on a highway given the locations of two exits -/
theorem service_center_location 
  (fourth_exit_location : ℝ) 
  (twelfth_exit_location : ℝ) 
  (h1 : fourth_exit_location = 50)
  (h2 : twelfth_exit_location = 190)
  (service_center_location : ℝ) 
  (h3 : service_center_location = fourth_exit_location + (twelfth_exit_location - fourth_exit_location) / 2) :
  service_center_location = 120 := by
sorry

end NUMINAMATH_CALUDE_service_center_location_l1818_181827


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1818_181851

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (2 + I) / I
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1818_181851


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1818_181854

/-- The discriminant of the quadratic equation x^2 - (2m+1)x + m^2 + m = 0 -/
def discriminant (m : ℝ) : ℝ := (2*m+1)^2 - 4*(m^2 + m)

/-- The sum of roots of the quadratic equation -/
def sum_of_roots (m : ℝ) : ℝ := 2*m + 1

/-- The product of roots of the quadratic equation -/
def product_of_roots (m : ℝ) : ℝ := m^2 + m

theorem quadratic_equation_properties (m : ℝ) :
  (∀ m, discriminant m > 0) ∧
  (∃ m, (2*(sum_of_roots m) + product_of_roots m)*(sum_of_roots m + 2*(product_of_roots m)) = 20 → m = -2 ∨ m = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1818_181854


namespace NUMINAMATH_CALUDE_angle_point_m_value_l1818_181824

theorem angle_point_m_value (θ : Real) (m : Real) :
  let P : Prod Real Real := (-Real.sqrt 3, m)
  (∃ (r : Real), r > 0 ∧ P.1^2 + P.2^2 = r^2) →  -- P is on the terminal side of θ
  Real.sin θ = Real.sqrt 13 / 13 →
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_angle_point_m_value_l1818_181824


namespace NUMINAMATH_CALUDE_probability_to_reach_target_is_79_1024_l1818_181867

/-- Represents a step direction --/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- Represents a position on the coordinate plane --/
structure Position :=
  (x : Int) (y : Int)

/-- The probability of a single step in any direction --/
def stepProbability : ℚ := 1/4

/-- The starting position --/
def start : Position := ⟨0, 0⟩

/-- The target position --/
def target : Position := ⟨3, 1⟩

/-- The maximum number of steps allowed --/
def maxSteps : ℕ := 6

/-- Calculates the probability of reaching the target position in at most maxSteps steps --/
noncomputable def probabilityToReachTarget (start : Position) (target : Position) (maxSteps : ℕ) : ℚ :=
  sorry

/-- The main theorem to prove --/
theorem probability_to_reach_target_is_79_1024 :
  probabilityToReachTarget start target maxSteps = 79/1024 :=
by sorry

end NUMINAMATH_CALUDE_probability_to_reach_target_is_79_1024_l1818_181867


namespace NUMINAMATH_CALUDE_unknown_number_value_l1818_181844

theorem unknown_number_value : ∃ x : ℝ, 5 + 2 * (8 - x) = 15 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l1818_181844


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l1818_181869

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 21) (Nat.lcm 14 15) = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l1818_181869


namespace NUMINAMATH_CALUDE_break_even_is_80_weeks_l1818_181893

/-- Represents the chicken and egg problem --/
structure ChickenEggProblem where
  num_chickens : ℕ
  chicken_cost : ℚ
  weekly_feed_cost : ℚ
  eggs_per_chicken : ℕ
  eggs_bought_weekly : ℕ
  egg_cost_per_dozen : ℚ

/-- Calculates the break-even point in weeks --/
def break_even_weeks (problem : ChickenEggProblem) : ℕ :=
  sorry

/-- Theorem stating that the break-even point is 80 weeks for the given problem --/
theorem break_even_is_80_weeks (problem : ChickenEggProblem)
  (h1 : problem.num_chickens = 4)
  (h2 : problem.chicken_cost = 20)
  (h3 : problem.weekly_feed_cost = 1)
  (h4 : problem.eggs_per_chicken = 3)
  (h5 : problem.eggs_bought_weekly = 12)
  (h6 : problem.egg_cost_per_dozen = 2) :
  break_even_weeks problem = 80 :=
sorry

end NUMINAMATH_CALUDE_break_even_is_80_weeks_l1818_181893


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1818_181822

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse is 6√2
  A = (h^2) / 4 →        -- area formula for isosceles right triangle
  A = 18 := by
    sorry

#check isosceles_right_triangle_area

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1818_181822


namespace NUMINAMATH_CALUDE_class_gender_composition_l1818_181895

theorem class_gender_composition (num_boys num_girls : ℕ) :
  num_boys = 2 * num_girls →
  num_boys = num_girls + 7 →
  num_girls - 1 = 6 := by
sorry

end NUMINAMATH_CALUDE_class_gender_composition_l1818_181895


namespace NUMINAMATH_CALUDE_square_root_plus_nine_l1818_181810

theorem square_root_plus_nine : ∃! x : ℝ, x > 0 ∧ x + 9 = x^2 := by sorry

end NUMINAMATH_CALUDE_square_root_plus_nine_l1818_181810


namespace NUMINAMATH_CALUDE_exists_region_with_min_area_l1818_181881

/-- Represents a line segment in a unit square --/
structure Segment where
  length : ℝ
  parallel_to_side : Bool

/-- Represents a configuration of segments in a unit square --/
structure SquareConfiguration where
  segments : List Segment
  total_length : ℝ
  total_length_eq : total_length = (segments.map Segment.length).sum
  total_length_bound : total_length = 18
  segments_within_square : ∀ s ∈ segments, s.length ≤ 1

/-- Represents a region formed by the segments --/
structure Region where
  area : ℝ

/-- The theorem to be proved --/
theorem exists_region_with_min_area (config : SquareConfiguration) :
  ∃ (r : Region), r.area ≥ 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exists_region_with_min_area_l1818_181881


namespace NUMINAMATH_CALUDE_beam_buying_problem_l1818_181826

/-- Represents the problem of buying beams as described in "Si Yuan Yu Jian" -/
theorem beam_buying_problem (x : ℕ) :
  (3 * x * (x - 1) = 6210) ↔
  (x > 0 ∧
   3 * x = 6210 / x +
   3 * (x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_beam_buying_problem_l1818_181826


namespace NUMINAMATH_CALUDE_sum_of_ages_l1818_181814

/-- Represents the ages of Alex, Chris, and Bella -/
structure Ages where
  alex : ℕ
  chris : ℕ
  bella : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.alex = ages.chris + 8 ∧
  ages.alex + 10 = 3 * (ages.chris - 6) ∧
  ages.bella = 2 * ages.chris

/-- The theorem to prove -/
theorem sum_of_ages (ages : Ages) :
  satisfiesConditions ages →
  ages.alex + ages.chris + ages.bella = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1818_181814


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1818_181836

theorem algebraic_expression_value (x y : ℝ) 
  (h : Real.sqrt (x - 3) + y^2 - 4*y + 4 = 0) : 
  (x^2 - y^2) / (x*y) * (1 / (x^2 - 2*x*y + y^2)) / (x / (x^2*y - x*y^2)) - 1 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1818_181836


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_specific_case_l1818_181839

theorem simplify_and_evaluate (x y : ℝ) :
  (x - y) * (x + y) + y^2 = x^2 :=
sorry

theorem evaluate_specific_case :
  let x : ℝ := 2
  let y : ℝ := 2023
  (x - y) * (x + y) + y^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_specific_case_l1818_181839


namespace NUMINAMATH_CALUDE_solution_set_exponential_inequality_l1818_181874

theorem solution_set_exponential_inequality :
  ∀ x : ℝ, (6 : ℝ) ^ (x - 2) < 1 ↔ x < 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_exponential_inequality_l1818_181874


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l1818_181802

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l1818_181802


namespace NUMINAMATH_CALUDE_dots_erased_l1818_181889

/-- Checks if a number contains the digit 2 in its base-3 representation -/
def containsTwo (n : Nat) : Bool :=
  let rec aux (m : Nat) : Bool :=
    if m = 0 then false
    else if m % 3 = 2 then true
    else aux (m / 3)
  aux n

/-- Counts the number of integers from 0 to n (inclusive) whose base-3 representation contains at least one digit '2' -/
def countNumbersWithTwo (n : Nat) : Nat :=
  (List.range (n + 1)).filter containsTwo |>.length

theorem dots_erased (total_dots : Nat) : 
  total_dots = 1000 → countNumbersWithTwo (total_dots - 1) = 895 := by sorry

end NUMINAMATH_CALUDE_dots_erased_l1818_181889


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1818_181800

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ,
  X^4 = (X^2 + 3*X + 2) * q + (-18*X - 16) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1818_181800


namespace NUMINAMATH_CALUDE_odd_decreasing_function_inequality_l1818_181809

open Set
open Function

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsDecreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

theorem odd_decreasing_function_inequality (f : ℝ → ℝ) (m : ℝ) 
  (h_odd : IsOdd f) 
  (h_decr : IsDecreasing f) 
  (h_domain : ∀ x ∈ Set.Ioo (-2 : ℝ) 2, f x ∈ Set.univ)
  (h_ineq : f (m - 1) + f (2 * m - 1) > 0) :
  m ∈ Set.Ioo (-1/2 : ℝ) (2/3) :=
sorry

end NUMINAMATH_CALUDE_odd_decreasing_function_inequality_l1818_181809


namespace NUMINAMATH_CALUDE_intersection_complement_equals_specific_set_l1818_181884

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 3, 6, 7}

theorem intersection_complement_equals_specific_set :
  B ∩ (U \ A) = {6, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_specific_set_l1818_181884


namespace NUMINAMATH_CALUDE_same_gender_probability_l1818_181894

/-- The probability of selecting 2 students of the same gender from a group of 5 students with 3 boys and 2 girls -/
theorem same_gender_probability (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_students = 5)
  (h2 : boys = 3)
  (h3 : girls = 2)
  (h4 : total_students = boys + girls) :
  (Nat.choose boys 2 + Nat.choose girls 2) / Nat.choose total_students 2 = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_same_gender_probability_l1818_181894


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l1818_181899

theorem smaller_solution_quadratic (x : ℝ) : 
  x^2 + 15*x + 36 = 0 ∧ x ≠ -3 → x = -12 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l1818_181899


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1818_181898

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a b : ℝ, ∀ x y : ℝ, 16*x^2 + m*x*y + 25*y^2 = (a*x + b*y)^2) → 
  m = 40 ∨ m = -40 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1818_181898


namespace NUMINAMATH_CALUDE_worker_distribution_l1818_181823

theorem worker_distribution (total_workers : ℕ) (male_days female_days : ℝ) 
  (h_total : total_workers = 20)
  (h_male_days : male_days = 2)
  (h_female_days : female_days = 3)
  (h_work_rate : ∀ (x y : ℕ), x + y = total_workers → 
    (x : ℝ) / male_days + (y : ℝ) / female_days = 1) :
  ∃ (male_workers female_workers : ℕ),
    male_workers + female_workers = total_workers ∧
    male_workers = 12 ∧
    female_workers = 8 :=
sorry

end NUMINAMATH_CALUDE_worker_distribution_l1818_181823


namespace NUMINAMATH_CALUDE_only_3_and_4_propositional_l1818_181807

-- Define a type for statements
inductive Statement
| Question : Statement
| Imperative : Statement
| Declarative : Statement

-- Define a function to check if a statement is propositional
def isPropositional (s : Statement) : Prop :=
  match s with
  | Statement.Declarative => True
  | _ => False

-- Define our four statements
def statement1 : Statement := Statement.Question
def statement2 : Statement := Statement.Imperative
def statement3 : Statement := Statement.Declarative
def statement4 : Statement := Statement.Declarative

-- Theorem to prove
theorem only_3_and_4_propositional :
  isPropositional statement1 = False ∧
  isPropositional statement2 = False ∧
  isPropositional statement3 = True ∧
  isPropositional statement4 = True := by
  sorry


end NUMINAMATH_CALUDE_only_3_and_4_propositional_l1818_181807


namespace NUMINAMATH_CALUDE_total_balls_count_l1818_181887

-- Define the number of boxes
def num_boxes : ℕ := 3

-- Define the number of balls in each box
def balls_per_box : ℕ := 5

-- Theorem to prove
theorem total_balls_count : num_boxes * balls_per_box = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_count_l1818_181887


namespace NUMINAMATH_CALUDE_prob_queen_first_three_cards_l1818_181837

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : Type := Unit

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The probability of drawing at least one Queen in the first three cards -/
def prob_at_least_one_queen (d : StandardDeck) : ℚ :=
  1 - (deck_size - num_queens) * (deck_size - num_queens - 1) * (deck_size - num_queens - 2) /
      (deck_size * (deck_size - 1) * (deck_size - 2))

theorem prob_queen_first_three_cards :
  ∀ d : StandardDeck, prob_at_least_one_queen d = 2174 / 10000 :=
by sorry

end NUMINAMATH_CALUDE_prob_queen_first_three_cards_l1818_181837


namespace NUMINAMATH_CALUDE_farmer_apples_l1818_181849

/-- The number of apples the farmer has after giving some away -/
def remaining_apples (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that the farmer has 39 apples after giving some away -/
theorem farmer_apples : remaining_apples 127 88 = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l1818_181849


namespace NUMINAMATH_CALUDE_muscle_gain_percentage_l1818_181816

/-- Proves that the percentage of body weight gained in muscle is 20% -/
theorem muscle_gain_percentage
  (initial_weight : ℝ)
  (final_weight : ℝ)
  (h1 : initial_weight = 120)
  (h2 : final_weight = 150)
  (h3 : ∀ (x : ℝ), x * initial_weight + (x / 4) * initial_weight = final_weight - initial_weight) :
  ∃ (muscle_gain_percent : ℝ), muscle_gain_percent = 20 := by
  sorry

end NUMINAMATH_CALUDE_muscle_gain_percentage_l1818_181816


namespace NUMINAMATH_CALUDE_john_double_sam_age_l1818_181863

/-- The number of years until John is twice as old as Sam -/
def years_until_double : ℕ := 9

/-- Sam's current age -/
def sam_age : ℕ := 9

/-- John's current age -/
def john_age : ℕ := 3 * sam_age

theorem john_double_sam_age :
  john_age + years_until_double = 2 * (sam_age + years_until_double) :=
by sorry

end NUMINAMATH_CALUDE_john_double_sam_age_l1818_181863


namespace NUMINAMATH_CALUDE_point_on_line_l1818_181882

/-- Given a line passing through points (2, 1) and (10, 5), 
    prove that the point (14, 7) lies on this line. -/
theorem point_on_line : ∀ (t : ℝ), 
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (10, 5)
  let p3 : ℝ × ℝ := (t, 7)
  -- Check if p3 is on the line through p1 and p2
  (p3.2 - p1.2) * (p2.1 - p1.1) = (p3.1 - p1.1) * (p2.2 - p1.2) →
  t = 14 :=
by
  sorry

#check point_on_line

end NUMINAMATH_CALUDE_point_on_line_l1818_181882


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1818_181832

theorem polynomial_simplification (x : ℝ) :
  x * (4 * x^3 + 3 * x^2 - 5) - 7 * (x^3 - 4 * x^2 + 2 * x - 6) =
  4 * x^4 - 4 * x^3 + 28 * x^2 - 19 * x + 42 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1818_181832


namespace NUMINAMATH_CALUDE_factorization_sum_l1818_181888

theorem factorization_sum (a b c : ℤ) : 
  (∀ x, x^2 + 9*x + 20 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 60 = (x + b) * (x - c)) →
  a + b + c = 14 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l1818_181888


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l1818_181808

theorem combined_mean_of_two_sets (set1_count set2_count : ℕ) 
  (set1_mean set2_mean : ℚ) : 
  set1_count = 7 → 
  set2_count = 8 → 
  set1_mean = 15 → 
  set2_mean = 30 → 
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 23 := by
sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l1818_181808


namespace NUMINAMATH_CALUDE_balloons_left_l1818_181805

theorem balloons_left (round_bags : ℕ) (round_per_bag : ℕ) (long_bags : ℕ) (long_per_bag : ℕ) (burst : ℕ) : 
  round_bags = 5 → 
  round_per_bag = 20 → 
  long_bags = 4 → 
  long_per_bag = 30 → 
  burst = 5 → 
  round_bags * round_per_bag + long_bags * long_per_bag - burst = 215 := by
sorry

end NUMINAMATH_CALUDE_balloons_left_l1818_181805


namespace NUMINAMATH_CALUDE_boat_distance_theorem_l1818_181852

/-- Calculates the distance a boat travels along a stream in one hour, given its speed in still water and its distance against the stream in one hour. -/
def distance_along_stream (boat_speed : ℝ) (distance_against : ℝ) : ℝ :=
  let stream_speed := boat_speed - distance_against
  boat_speed + stream_speed

/-- Theorem stating that a boat with a speed of 7 km/hr in still water,
    traveling 3 km against the stream in one hour,
    will travel 11 km along the stream in one hour. -/
theorem boat_distance_theorem :
  distance_along_stream 7 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_theorem_l1818_181852
