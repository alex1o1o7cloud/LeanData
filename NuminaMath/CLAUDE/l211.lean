import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l211_21168

theorem inequality_proof (a b c d : ℝ) :
  (a^8 + b^3 + c^8 + d^3)^2 ≤ 4 * (a^4 + b^8 + c^8 + d^4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l211_21168


namespace NUMINAMATH_CALUDE_expression_simplification_l211_21117

theorem expression_simplification :
  ∀ p : ℝ, ((7*p+3)-3*p*5)*(2)+(5-2/4)*(8*p-12) = 20*p - 48 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l211_21117


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l211_21138

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 4| + 3 * y = 15 :=
by
  -- The unique solution is y = 4.75
  use 4.75
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l211_21138


namespace NUMINAMATH_CALUDE_ladder_problem_l211_21131

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ base_distance : ℝ, base_distance^2 + height^2 = ladder_length^2 ∧ base_distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l211_21131


namespace NUMINAMATH_CALUDE_distance_between_cities_l211_21174

/-- The distance between City A and City B -/
def distance : ℝ := 427.5

/-- The time for the first trip in hours -/
def time_first_trip : ℝ := 6

/-- The time for the return trip in hours -/
def time_return_trip : ℝ := 4.5

/-- The time saved on each trip in hours -/
def time_saved_per_trip : ℝ := 0.5

/-- The speed of the round trip if time was saved, in miles per hour -/
def speed_with_time_saved : ℝ := 90

theorem distance_between_cities :
  distance = 427.5 ∧
  (2 * distance) / (time_first_trip + time_return_trip - 2 * time_saved_per_trip) = speed_with_time_saved :=
by sorry

end NUMINAMATH_CALUDE_distance_between_cities_l211_21174


namespace NUMINAMATH_CALUDE_arithmetic_proof_l211_21144

theorem arithmetic_proof : 4 * (8 - 6) - 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l211_21144


namespace NUMINAMATH_CALUDE_max_midpoints_on_circle_l211_21135

/-- A regular n-gon with n ≥ 3 -/
structure RegularNGon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The set of midpoints of all sides and diagonals of a regular n-gon -/
def midpoints (ngon : RegularNGon) : Set (ℝ × ℝ) :=
  sorry

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of points from a set that lie on a given circle -/
def pointsOnCircle (S : Set (ℝ × ℝ)) (c : Circle) : ℕ :=
  sorry

/-- The maximum number of points from a set that lie on any circle -/
def maxPointsOnCircle (S : Set (ℝ × ℝ)) : ℕ :=
  sorry

/-- Theorem: The maximum number of marked midpoints that lie on the same circle is n -/
theorem max_midpoints_on_circle (ngon : RegularNGon) :
    maxPointsOnCircle (midpoints ngon) = ngon.n :=
  sorry

end NUMINAMATH_CALUDE_max_midpoints_on_circle_l211_21135


namespace NUMINAMATH_CALUDE_dinner_cost_theorem_l211_21153

/-- Calculate the total amount Bret spends on dinner -/
def dinner_cost : ℝ :=
  let team_a_size : ℕ := 4
  let team_b_size : ℕ := 4
  let main_meal_cost : ℝ := 12.00
  let team_a_appetizers : ℕ := 2
  let team_a_appetizer_cost : ℝ := 6.00
  let team_b_appetizers : ℕ := 3
  let team_b_appetizer_cost : ℝ := 8.00
  let sharing_plates : ℕ := 4
  let sharing_plate_cost : ℝ := 10.00
  let tip_percentage : ℝ := 0.20
  let rush_order_fee : ℝ := 5.00
  let sales_tax_rate : ℝ := 0.07

  let main_meals_cost := (team_a_size + team_b_size) * main_meal_cost
  let appetizers_cost := team_a_appetizers * team_a_appetizer_cost + team_b_appetizers * team_b_appetizer_cost
  let sharing_plates_cost := sharing_plates * sharing_plate_cost
  let food_cost := main_meals_cost + appetizers_cost + sharing_plates_cost
  let tip := food_cost * tip_percentage
  let subtotal := food_cost + tip + rush_order_fee
  let sales_tax := (food_cost + tip) * sales_tax_rate
  food_cost + tip + rush_order_fee + sales_tax

theorem dinner_cost_theorem : dinner_cost = 225.85 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_theorem_l211_21153


namespace NUMINAMATH_CALUDE_quadratic_inequality_l211_21177

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- Theorem: If f(-3) = f(1) for a quadratic function f(x) = x^2 + bx + c, 
    then f(1) > c > f(-1) -/
theorem quadratic_inequality (b c : ℝ) : 
  f b c (-3) = f b c 1 → f b c 1 > c ∧ c > f b c (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l211_21177


namespace NUMINAMATH_CALUDE_shelves_count_l211_21129

/-- The number of shelves in a library --/
def number_of_shelves (books_per_shelf : ℕ) (total_round_trip_distance : ℕ) : ℕ :=
  (total_round_trip_distance / 2) / books_per_shelf

/-- Theorem: The number of shelves is 4 --/
theorem shelves_count :
  number_of_shelves 400 3200 = 4 := by
  sorry

end NUMINAMATH_CALUDE_shelves_count_l211_21129


namespace NUMINAMATH_CALUDE_divisibility_by_x2_plus_x_plus_1_l211_21197

theorem divisibility_by_x2_plus_x_plus_1 (n : ℕ) (hn : n > 0) :
  ∃ q : Polynomial ℚ, (X + 1 : Polynomial ℚ)^(2*n + 1) + X^(n + 2) = (X^2 + X + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_x2_plus_x_plus_1_l211_21197


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l211_21111

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a → a 6 = 6 → a 9 = 9 → a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l211_21111


namespace NUMINAMATH_CALUDE_julia_age_after_ten_years_l211_21188

/-- Given the ages and relationships of siblings, calculate Julia's age after 10 years -/
theorem julia_age_after_ten_years 
  (justin_age : ℕ)
  (jessica_age_when_justin_born : ℕ)
  (james_age_diff_jessica : ℕ)
  (julia_age_diff_justin : ℕ)
  (h1 : justin_age = 26)
  (h2 : jessica_age_when_justin_born = 6)
  (h3 : james_age_diff_jessica = 7)
  (h4 : julia_age_diff_justin = 8) :
  justin_age - julia_age_diff_justin + 10 = 28 :=
by sorry

end NUMINAMATH_CALUDE_julia_age_after_ten_years_l211_21188


namespace NUMINAMATH_CALUDE_time_to_write_117639_l211_21179

def digits_count (n : ℕ) : ℕ := 
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else if n < 10000 then 4
  else if n < 100000 then 5
  else 6

def total_digits (n : ℕ) : ℕ := 
  (List.range n).map digits_count |>.sum

def time_to_write (n : ℕ) (digits_per_minute : ℕ) : ℕ := 
  (total_digits n + digits_per_minute - 1) / digits_per_minute

theorem time_to_write_117639 : 
  time_to_write 117639 93 = 4 * 24 * 60 + 10 * 60 + 34 := by sorry

end NUMINAMATH_CALUDE_time_to_write_117639_l211_21179


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l211_21128

/-- Given 5 persons, if replacing one person with a new person weighing 95.5 kg
    increases the average weight by 5.5 kg, then the weight of the replaced person was 68 kg. -/
theorem weight_of_replaced_person (initial_count : ℕ) (new_person_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 5 →
  new_person_weight = 95.5 →
  avg_increase = 5.5 →
  (new_person_weight - initial_count * avg_increase : ℝ) = 68 := by
sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l211_21128


namespace NUMINAMATH_CALUDE_roots_sum_magnitude_l211_21157

theorem roots_sum_magnitude (p : ℝ) (r₁ r₂ : ℝ) : 
  (∃ x : ℝ, x^2 + p*x + 12 = 0) →
  r₁^2 + p*r₁ + 12 = 0 →
  r₂^2 + p*r₂ + 12 = 0 →
  r₁ ≠ r₂ →
  |r₁ + r₂| > 6 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_magnitude_l211_21157


namespace NUMINAMATH_CALUDE_calculation_proof_l211_21182

theorem calculation_proof :
  (- 2^2 * (1/4) + 4 / (4/9) + (-1)^2023 = 7) ∧
  (- 1^4 + |2 - (-3)^2| + (1/2) / (-(3/2)) = 17/3) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l211_21182


namespace NUMINAMATH_CALUDE_time_per_cut_l211_21119

/-- Given 3 pieces of wood, each cut into 3 sections, in 18 minutes total, prove the time per cut is 3 minutes -/
theorem time_per_cut (num_pieces : ℕ) (sections_per_piece : ℕ) (total_time : ℕ) :
  num_pieces = 3 →
  sections_per_piece = 3 →
  total_time = 18 →
  (total_time : ℚ) / (num_pieces * (sections_per_piece - 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_per_cut_l211_21119


namespace NUMINAMATH_CALUDE_lcm_problem_l211_21190

theorem lcm_problem (a b : ℕ+) : 
  a = 1491 → Nat.lcm a b = 5964 → b = 4 := by sorry

end NUMINAMATH_CALUDE_lcm_problem_l211_21190


namespace NUMINAMATH_CALUDE_solve_puppy_problem_l211_21183

def puppyProblem (initialPuppies : ℕ) (givenAway : ℕ) (kept : ℕ) (sellingPrice : ℕ) (profit : ℕ) : Prop :=
  let remainingAfterGiveaway := initialPuppies - givenAway
  let soldPuppies := remainingAfterGiveaway - kept
  let revenue := soldPuppies * sellingPrice
  let amountToStud := revenue - profit
  amountToStud = 300

theorem solve_puppy_problem :
  puppyProblem 8 4 1 600 1500 := by
  sorry

end NUMINAMATH_CALUDE_solve_puppy_problem_l211_21183


namespace NUMINAMATH_CALUDE_markers_final_count_l211_21114

def markers_problem (initial : ℕ) (robert_gave : ℕ) (sarah_took : ℕ) (teacher_multiplier : ℕ) : ℕ :=
  let after_robert := initial + robert_gave
  let after_sarah := after_robert - sarah_took
  let after_teacher := after_sarah + teacher_multiplier * after_sarah
  (after_teacher) / 2

theorem markers_final_count : 
  markers_problem 217 109 35 3 = 582 := by sorry

end NUMINAMATH_CALUDE_markers_final_count_l211_21114


namespace NUMINAMATH_CALUDE_stock_price_change_l211_21108

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_decrease := initial_price * (1 - 0.05)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l211_21108


namespace NUMINAMATH_CALUDE_cookie_difference_l211_21158

theorem cookie_difference (initial_sweet initial_salty sweet_eaten salty_eaten : ℕ) :
  initial_sweet = 39 →
  initial_salty = 6 →
  sweet_eaten = 32 →
  salty_eaten = 23 →
  sweet_eaten - salty_eaten = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l211_21158


namespace NUMINAMATH_CALUDE_area_outside_squares_inside_triangle_l211_21127

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents the problem setup -/
structure SquareProblem where
  bigSquare : Square
  smallSquare1 : Square
  smallSquare2 : Square

/-- The main theorem stating the area of the region -/
theorem area_outside_squares_inside_triangle (p : SquareProblem) : 
  p.bigSquare.sideLength = 6 ∧ 
  p.smallSquare1.sideLength = 2 ∧ 
  p.smallSquare2.sideLength = 3 →
  let triangleArea := (p.bigSquare.sideLength ^ 2) / 2
  let smallSquaresArea := p.smallSquare1.sideLength ^ 2 + p.smallSquare2.sideLength ^ 2
  triangleArea - smallSquaresArea = 5 := by
  sorry

end NUMINAMATH_CALUDE_area_outside_squares_inside_triangle_l211_21127


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l211_21141

def f (x : ℝ) := x^3 + x - 1

theorem root_exists_in_interval :
  Continuous f ∧ f 0 < 0 ∧ f 1 > 0 →
  ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l211_21141


namespace NUMINAMATH_CALUDE_expression_simplification_l211_21175

theorem expression_simplification (y : ℝ) : 
  (3 - 4*y) * (3 + 4*y) + (3 + 4*y)^2 = 18 + 24*y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l211_21175


namespace NUMINAMATH_CALUDE_david_scott_age_difference_l211_21166

/-- Represents the ages of three brothers -/
structure BrothersAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : BrothersAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- The theorem to prove -/
theorem david_scott_age_difference (ages : BrothersAges) 
  (h : satisfiesConditions ages) : ages.david - ages.scott = 8 := by
  sorry

end NUMINAMATH_CALUDE_david_scott_age_difference_l211_21166


namespace NUMINAMATH_CALUDE_men_to_women_ratio_l211_21101

/-- Proves that the ratio of men to women workers is 1:3 given the problem conditions -/
theorem men_to_women_ratio (woman_wage : ℝ) (num_women : ℕ) : 
  let man_wage := 2 * woman_wage
  let women_earnings := num_women * woman_wage * 30
  let men_earnings := (num_women / 3) * man_wage * 20
  women_earnings = 21600 → men_earnings = 14400 := by
  sorry

end NUMINAMATH_CALUDE_men_to_women_ratio_l211_21101


namespace NUMINAMATH_CALUDE_right_angled_triangles_with_special_property_l211_21186

theorem right_angled_triangles_with_special_property :
  {(a, b, c) : ℕ × ℕ × ℕ | 
    0 < a ∧ a < b ∧ b < c ∧
    a * b = 4 * (a + b + c) ∧
    a * a + b * b = c * c} =
  {(10, 24, 26), (12, 16, 20), (9, 40, 41)} :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangles_with_special_property_l211_21186


namespace NUMINAMATH_CALUDE_function_values_and_range_l211_21143

noncomputable def f (b c x : ℝ) : ℝ := -1/3 * x^3 + b * x^2 + c * x + b * c

noncomputable def g (a x : ℝ) : ℝ := a * x^2 - 2 * Real.log x

theorem function_values_and_range :
  ∀ b c : ℝ,
  (∃ x : ℝ, f b c x = -4/3 ∧ ∀ y : ℝ, f b c y ≤ f b c x) →
  (b = -1 ∧ c = 3) ∧
  ∀ a : ℝ,
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 3 ∧ 0 < x₂ ∧ x₂ < 3 ∧ |f b c x₁ - g a x₂| < 1) →
  (2 * Real.log 3 - 13) / 9 ≤ a ∧ a ≤ (6 * Real.log 3 - 1) / 27 :=
by sorry

end NUMINAMATH_CALUDE_function_values_and_range_l211_21143


namespace NUMINAMATH_CALUDE_max_reflections_l211_21148

theorem max_reflections (angle : ℝ) (h : angle = 8) : 
  ∃ (n : ℕ), n ≤ 10 ∧ n * angle < 90 ∧ ∀ m : ℕ, m > n → m * angle ≥ 90 :=
by sorry

end NUMINAMATH_CALUDE_max_reflections_l211_21148


namespace NUMINAMATH_CALUDE_complete_square_factorization_l211_21191

theorem complete_square_factorization :
  ∀ x : ℝ, x^2 + 4 + 4*x = (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_factorization_l211_21191


namespace NUMINAMATH_CALUDE_average_income_P_R_l211_21142

def average_income (x y : ℕ) : ℚ := (x + y) / 2

theorem average_income_P_R (P Q R : ℕ) : 
  average_income P Q = 5050 →
  average_income Q R = 6250 →
  P = 4000 →
  average_income P R = 5200 := by
sorry

end NUMINAMATH_CALUDE_average_income_P_R_l211_21142


namespace NUMINAMATH_CALUDE_divisibility_of_sum_l211_21199

theorem divisibility_of_sum : ∃ y : ℕ,
  y = 112 + 160 + 272 + 432 + 1040 + 1264 + 4256 ∧
  16 ∣ y ∧ 8 ∣ y ∧ 4 ∣ y ∧ 2 ∣ y :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_l211_21199


namespace NUMINAMATH_CALUDE_door_purchase_savings_l211_21150

/-- Calculates the cost of purchasing doors with the "buy 3 get 1 free" offer -/
def cost_with_offer (num_doors : ℕ) (price_per_door : ℕ) : ℕ :=
  ((num_doors + 3) / 4) * 3 * price_per_door

/-- Calculates the regular cost of purchasing doors without any offer -/
def regular_cost (num_doors : ℕ) (price_per_door : ℕ) : ℕ :=
  num_doors * price_per_door

/-- Calculates the savings when purchasing doors with the offer -/
def savings (num_doors : ℕ) (price_per_door : ℕ) : ℕ :=
  regular_cost num_doors price_per_door - cost_with_offer num_doors price_per_door

theorem door_purchase_savings :
  let alice_doors := 6
  let bob_doors := 9
  let price_per_door := 120
  let total_doors := alice_doors + bob_doors
  savings total_doors price_per_door = 600 :=
by sorry

end NUMINAMATH_CALUDE_door_purchase_savings_l211_21150


namespace NUMINAMATH_CALUDE_problem_solution_l211_21126

-- Definition of A_n^m
def A (n m : ℕ) : ℕ := n.factorial / (n - m).factorial

-- Theorem statement
theorem problem_solution (n : ℕ) : A (n + 1) 2 - A n 2 = 10 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l211_21126


namespace NUMINAMATH_CALUDE_square_difference_evaluation_l211_21162

theorem square_difference_evaluation : 81^2 - (45 + 9)^2 = 3645 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_evaluation_l211_21162


namespace NUMINAMATH_CALUDE_daily_rental_cost_satisfies_conditions_l211_21149

/-- Represents the daily rental cost of a car in dollars -/
def daily_rental_cost : ℝ := 30

/-- Represents the cost per mile in dollars -/
def cost_per_mile : ℝ := 0.18

/-- Represents the total budget in dollars -/
def total_budget : ℝ := 75

/-- Represents the number of miles that can be driven -/
def miles_driven : ℝ := 250

/-- Theorem stating that the daily rental cost satisfies the given conditions -/
theorem daily_rental_cost_satisfies_conditions :
  daily_rental_cost + (cost_per_mile * miles_driven) = total_budget :=
by sorry

end NUMINAMATH_CALUDE_daily_rental_cost_satisfies_conditions_l211_21149


namespace NUMINAMATH_CALUDE_find_a_l211_21189

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => 
  if x > 0 then a^x else -a^(-x)

-- State the theorem
theorem find_a : 
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ 
  (∀ x, f a (-x) = -(f a x)) ∧  -- odd function property
  (∀ x > 0, f a x = a^x) ∧      -- definition for x > 0
  f a (Real.log 2 / Real.log (1/2)) = -3 ∧ -- given condition
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_find_a_l211_21189


namespace NUMINAMATH_CALUDE_revenue_change_l211_21146

/-- Given a price increase of 80% and a sales decrease of 35%, prove that revenue increases by 17% -/
theorem revenue_change (P Q : ℝ) (h_P : P > 0) (h_Q : Q > 0) : 
  let R := P * Q
  let P_new := P * (1 + 0.80)
  let Q_new := Q * (1 - 0.35)
  let R_new := P_new * Q_new
  (R_new - R) / R = 0.17 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_l211_21146


namespace NUMINAMATH_CALUDE_garden_area_l211_21193

-- Define the rectangle garden
def rectangular_garden (length width : ℝ) := length * width

-- Theorem statement
theorem garden_area : rectangular_garden 12 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l211_21193


namespace NUMINAMATH_CALUDE_find_n_l211_21165

/-- Given that P = s / ((1 + k)^n + m), prove that n = (log((s/P) - m)) / (log(1 + k)) -/
theorem find_n (P s k m n : ℝ) (h : P = s / ((1 + k)^n + m)) (h1 : k > -1) (h2 : P > 0) (h3 : s > 0) :
  n = (Real.log ((s/P) - m)) / (Real.log (1 + k)) := by
  sorry

end NUMINAMATH_CALUDE_find_n_l211_21165


namespace NUMINAMATH_CALUDE_hike_length_is_seven_l211_21124

/-- Calculates the length of a hike given water consumption and time information -/
def hikeLength (initialWater : ℚ) (duration : ℚ) (remainingWater : ℚ) (leakRate : ℚ) 
                (lastMileConsumption : ℚ) (firstPartConsumptionRate : ℚ) : ℚ :=
  let totalWaterLost := initialWater - remainingWater
  let leakageLoss := leakRate * duration
  let firstPartConsumption := totalWaterLost - leakageLoss - lastMileConsumption
  let firstPartDistance := firstPartConsumption / firstPartConsumptionRate
  firstPartDistance + 1

/-- Theorem stating that given the specific conditions, the hike length is 7 miles -/
theorem hike_length_is_seven :
  hikeLength 9 2 3 1 2 (2/3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_hike_length_is_seven_l211_21124


namespace NUMINAMATH_CALUDE_student_age_problem_l211_21161

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The problem statement -/
theorem student_age_problem :
  ∃! n : ℕ, 1900 ≤ n ∧ n < 1960 ∧ (1960 - n = sum_of_digits n) := by sorry

end NUMINAMATH_CALUDE_student_age_problem_l211_21161


namespace NUMINAMATH_CALUDE_equation_roots_exist_l211_21178

/-- Proves that the equation x|x| + px + q = 0 can have real roots even when p^2 - 4q < 0 -/
theorem equation_roots_exist (p q : ℝ) (h : p^2 - 4*q < 0) : 
  ∃ x : ℝ, x * |x| + p * x + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_exist_l211_21178


namespace NUMINAMATH_CALUDE_optimal_sale_exists_l211_21152

/-- Represents the selling price and number of items that maximize profit while meeting constraints. -/
def OptimalSale : Type :=
  { price : ℕ // price > 0 } × { quantity : ℕ // quantity > 0 }

/-- Calculates the profit given the selling price and quantity. -/
def profit (costPrice : ℕ) (sale : OptimalSale) : ℤ :=
  (sale.1.val - costPrice) * sale.2.val

/-- Calculates the total cost given the cost price and quantity. -/
def totalCost (costPrice : ℕ) (sale : OptimalSale) : ℕ :=
  costPrice * sale.2.val

/-- Calculates the quantity sold based on the price change from the initial price. -/
def quantitySold (initialQuantity : ℕ) (initialPrice : ℕ) (newPrice : ℕ) : ℤ :=
  initialQuantity - 10 * (newPrice - initialPrice)

theorem optimal_sale_exists (initialPrice initialQuantity costPrice : ℕ) 
    (h1 : initialPrice > costPrice)
    (h2 : initialQuantity > 0)
    (h3 : costPrice > 0) :
  ∃ (sale : OptimalSale),
    profit costPrice sale = 8000 ∧ 
    totalCost costPrice sale < 10000 ∧
    sale.2.val = quantitySold initialQuantity initialPrice sale.1.val :=
by
  -- The proof would go here
  sorry

#check optimal_sale_exists

end NUMINAMATH_CALUDE_optimal_sale_exists_l211_21152


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l211_21170

theorem complex_magnitude_problem (a b : ℝ) (h : a^2 - 4 + b * Complex.I - Complex.I = 0) :
  Complex.abs (a + b * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l211_21170


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l211_21120

/-- The degree of a polynomial -/
def degree (p : Polynomial ℂ) : ℕ := sorry

theorem polynomial_remainder_theorem :
  ∃ (Q R : Polynomial ℂ),
    (X : Polynomial ℂ)^2023 + 1 = (X^2 + X + 1) * Q + R ∧
    degree R < 2 ∧
    R = -X + 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l211_21120


namespace NUMINAMATH_CALUDE_number_operations_l211_21163

theorem number_operations (x : ℚ) : x = 192 → 6 * (((x/8) + 8) - 30) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l211_21163


namespace NUMINAMATH_CALUDE_boundary_length_of_modified_square_l211_21113

-- Define the square's area
def square_area : ℝ := 256

-- Define the number of divisions per side
def divisions : ℕ := 4

-- Theorem statement
theorem boundary_length_of_modified_square :
  let side_length := Real.sqrt square_area
  let segment_length := side_length / divisions
  let arc_length := 2 * Real.pi * segment_length
  let straight_segments_length := 2 * divisions * segment_length
  abs ((arc_length + straight_segments_length) - 57.1) < 0.05 := by
sorry

end NUMINAMATH_CALUDE_boundary_length_of_modified_square_l211_21113


namespace NUMINAMATH_CALUDE_y_sum_equals_4360_l211_21176

/-- Given real numbers y₁ to y₈ satisfying four equations, 
    prove that a specific linear combination of these numbers equals 4360 -/
theorem y_sum_equals_4360 
  (y₁ y₂ y₃ y₄ y₅ y₆ y₇ y₈ : ℝ) 
  (eq1 : y₁ + 4*y₂ + 9*y₃ + 16*y₄ + 25*y₅ + 36*y₆ + 49*y₇ + 64*y₈ = 2)
  (eq2 : 4*y₁ + 9*y₂ + 16*y₃ + 25*y₄ + 36*y₅ + 49*y₆ + 64*y₇ + 81*y₈ = 15)
  (eq3 : 9*y₁ + 16*y₂ + 25*y₃ + 36*y₄ + 49*y₅ + 64*y₆ + 81*y₇ + 100*y₈ = 156)
  (eq4 : 16*y₁ + 25*y₂ + 36*y₃ + 49*y₄ + 64*y₅ + 81*y₆ + 100*y₇ + 121*y₈ = 1305) :
  25*y₁ + 36*y₂ + 49*y₃ + 64*y₄ + 81*y₅ + 100*y₆ + 121*y₇ + 144*y₈ = 4360 := by
  sorry


end NUMINAMATH_CALUDE_y_sum_equals_4360_l211_21176


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l211_21145

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (Real.sqrt 3, 1))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a + 2 • b))^2 + (Prod.snd (a + 2 • b))^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l211_21145


namespace NUMINAMATH_CALUDE_olivia_spent_15_dollars_l211_21154

/-- The amount spent at a supermarket, given the initial amount and the amount left after spending. -/
def amount_spent (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

/-- Proves that Olivia spent 15 dollars at the supermarket. -/
theorem olivia_spent_15_dollars : amount_spent 78 63 = 15 := by
  sorry

end NUMINAMATH_CALUDE_olivia_spent_15_dollars_l211_21154


namespace NUMINAMATH_CALUDE_trigonometric_identities_l211_21112

theorem trigonometric_identities : 
  (2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1/2) ∧ 
  (Real.sin (45 * π / 180) * Real.cos (15 * π / 180) - 
   Real.cos (45 * π / 180) * Real.sin (15 * π / 180) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l211_21112


namespace NUMINAMATH_CALUDE_max_braking_distance_l211_21137

/-- The braking distance function for a car -/
def s (t : ℝ) : ℝ := 15 * t - 6 * t^2

/-- The maximum distance traveled by the car before stopping -/
theorem max_braking_distance :
  (∃ t : ℝ, ∀ u : ℝ, s u ≤ s t) ∧ (∃ t : ℝ, s t = 75/8) :=
sorry

end NUMINAMATH_CALUDE_max_braking_distance_l211_21137


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l211_21104

theorem divisibility_equivalence (a b : ℤ) :
  (13 ∣ (2 * a + 3 * b)) ↔ (13 ∣ (2 * b - 3 * a)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l211_21104


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l211_21169

/-- Given a sphere with surface area 256π cm², its volume is (2048/3)π cm³ -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    4 * Real.pi * r^2 = 256 * Real.pi → 
    (4 / 3) * Real.pi * r^3 = (2048 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l211_21169


namespace NUMINAMATH_CALUDE_max_inradius_value_l211_21181

/-- A parabola with equation y^2 = 4x and focus at (1,0) -/
structure Parabola where
  equation : ℝ → ℝ → Prop := fun x y ↦ y^2 = 4*x
  focus : ℝ × ℝ := (1, 0)

/-- The inradius of a triangle formed by a point on the parabola, the focus, and the origin -/
def inradius (p : Parabola) (P : ℝ × ℝ) : ℝ :=
  sorry

/-- The maximum inradius of triangle OPF -/
def max_inradius (p : Parabola) : ℝ :=
  sorry

theorem max_inradius_value (p : Parabola) :
  max_inradius p = 2 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_CALUDE_max_inradius_value_l211_21181


namespace NUMINAMATH_CALUDE_P_infimum_and_no_minimum_l211_21100

/-- The function P : ℝ² → ℝ defined by P(X₁, X₂) = X₁² + (1 - X₁X₂)² -/
def P : ℝ × ℝ → ℝ := fun (X₁, X₂) ↦ X₁^2 + (1 - X₁ * X₂)^2

theorem P_infimum_and_no_minimum :
  (∀ ε > 0, ∃ (X₁ X₂ : ℝ), P (X₁, X₂) < ε) ∧
  (¬∃ (X₁ X₂ : ℝ), ∀ (Y₁ Y₂ : ℝ), P (X₁, X₂) ≤ P (Y₁, Y₂)) := by
  sorry

end NUMINAMATH_CALUDE_P_infimum_and_no_minimum_l211_21100


namespace NUMINAMATH_CALUDE_sin_plus_cos_for_point_two_neg_one_l211_21125

/-- Given an angle α whose terminal side passes through the point (2, -1),
    prove that sin α + cos α = √5/5 -/
theorem sin_plus_cos_for_point_two_neg_one (α : Real) :
  (2 : Real) = Real.cos α * Real.sqrt 5 ∧ (-1 : Real) = Real.sin α * Real.sqrt 5 →
  Real.sin α + Real.cos α = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_for_point_two_neg_one_l211_21125


namespace NUMINAMATH_CALUDE_stratified_sampling_pine_saplings_l211_21196

/-- Calculates the expected number of pine saplings in a stratified sample -/
def expected_pine_saplings (total_saplings : ℕ) (pine_saplings : ℕ) (sample_size : ℕ) : ℕ :=
  (pine_saplings * sample_size) / total_saplings

theorem stratified_sampling_pine_saplings :
  expected_pine_saplings 30000 4000 150 = 20 := by
  sorry

#eval expected_pine_saplings 30000 4000 150

end NUMINAMATH_CALUDE_stratified_sampling_pine_saplings_l211_21196


namespace NUMINAMATH_CALUDE_fraction_multiplication_l211_21130

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 * 5 / 8 = 15 / 77 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l211_21130


namespace NUMINAMATH_CALUDE_largest_square_area_l211_21187

/-- Given a configuration of 7 squares where the smallest square has area 9,
    the largest square has area 324. -/
theorem largest_square_area (num_squares : ℕ) (smallest_area : ℝ) : 
  num_squares = 7 → smallest_area = 9 → ∃ (largest_area : ℝ), largest_area = 324 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l211_21187


namespace NUMINAMATH_CALUDE_triangular_array_coin_sum_l211_21140

/-- The number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_coin_sum :
  ∃ (N : ℕ), triangular_sum N = 2211 ∧ sum_of_digits N = 12 :=
sorry

end NUMINAMATH_CALUDE_triangular_array_coin_sum_l211_21140


namespace NUMINAMATH_CALUDE_sum_of_roots_is_one_l211_21173

-- Define a quadratic polynomial Q(x) = ax^2 + bx + c
def Q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem sum_of_roots_is_one 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, Q a b c (x^4 + x^2) ≥ Q a b c (x^3 + 1)) : 
  (- b) / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_one_l211_21173


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l211_21102

theorem solve_equation_and_evaluate (x : ℝ) : 
  2*x - 7 = 8*x - 1 → 5*(x - 3) = -20 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l211_21102


namespace NUMINAMATH_CALUDE_water_breadth_in_cistern_l211_21164

/-- Calculates the breadth of water in a cistern given its dimensions and wet surface area -/
theorem water_breadth_in_cistern (length width wet_surface_area : ℝ) :
  length = 9 →
  width = 6 →
  wet_surface_area = 121.5 →
  ∃ (breadth : ℝ),
    breadth = 2.25 ∧
    wet_surface_area = length * width + 2 * length * breadth + 2 * width * breadth :=
by sorry

end NUMINAMATH_CALUDE_water_breadth_in_cistern_l211_21164


namespace NUMINAMATH_CALUDE_divisibility_problem_l211_21159

theorem divisibility_problem (a b n : ℤ) : 
  n = 10 * a + b → (17 ∣ (a - 5 * b)) → (17 ∣ n) := by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l211_21159


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l211_21105

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 1}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l211_21105


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l211_21171

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 - a*x + 4 < 0

-- Define the necessary condition
def necessary_condition (a : ℝ) : Prop := a > 3

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, p a → necessary_condition a) ∧
  ¬(∀ a : ℝ, necessary_condition a → p a) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l211_21171


namespace NUMINAMATH_CALUDE_percentage_problem_l211_21118

theorem percentage_problem (x : ℝ) (h : x = 300) : 
  ∃ P : ℝ, P * x / 100 = x / 3 + 110 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l211_21118


namespace NUMINAMATH_CALUDE_lisa_cleaning_time_proof_l211_21151

/-- The time it takes Lisa to clean her room alone -/
def lisa_cleaning_time : ℝ := 8

/-- The time it takes Kay to clean her room alone -/
def kay_cleaning_time : ℝ := 12

/-- The time it takes Lisa and Kay to clean a room together -/
def combined_cleaning_time : ℝ := 4.8

theorem lisa_cleaning_time_proof :
  lisa_cleaning_time = 8 ∧
  (1 / lisa_cleaning_time + 1 / kay_cleaning_time = 1 / combined_cleaning_time) :=
sorry

end NUMINAMATH_CALUDE_lisa_cleaning_time_proof_l211_21151


namespace NUMINAMATH_CALUDE_tennis_preference_theorem_l211_21110

/-- The percentage of students preferring tennis when combining two schools -/
def combined_tennis_preference 
  (central_students : ℕ) 
  (central_tennis_percentage : ℚ)
  (north_students : ℕ)
  (north_tennis_percentage : ℚ) : ℚ :=
  ((central_students : ℚ) * central_tennis_percentage + 
   (north_students : ℚ) * north_tennis_percentage) / 
  ((central_students + north_students) : ℚ)

theorem tennis_preference_theorem : 
  combined_tennis_preference 1800 (25/100) 3000 (35/100) = 31/100 := by
  sorry

end NUMINAMATH_CALUDE_tennis_preference_theorem_l211_21110


namespace NUMINAMATH_CALUDE_negative_square_power_2014_l211_21156

theorem negative_square_power_2014 : -(-(-1)^2)^2014 = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_power_2014_l211_21156


namespace NUMINAMATH_CALUDE_factorization_equality_l211_21133

theorem factorization_equality (a b : ℝ) : (2*a - b)^2 + 8*a*b = (2*a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l211_21133


namespace NUMINAMATH_CALUDE_barbell_cost_l211_21121

/-- Given that John buys barbells, gives money, and receives change, 
    prove the cost of each barbell. -/
theorem barbell_cost (num_barbells : ℕ) (money_given : ℕ) (change_received : ℕ) : 
  num_barbells = 3 → money_given = 850 → change_received = 40 → 
  (money_given - change_received) / num_barbells = 270 := by
  sorry

#check barbell_cost

end NUMINAMATH_CALUDE_barbell_cost_l211_21121


namespace NUMINAMATH_CALUDE_instruction_set_exists_l211_21115

/-- Represents a box that may contain a ball or be empty. -/
inductive Box
| withBall : Box
| empty : Box

/-- Represents an instruction to swap the contents of two boxes. -/
structure SwapInstruction where
  i : Nat
  j : Nat

/-- Represents a configuration of N boxes. -/
def BoxConfiguration (N : Nat) := Fin N → Box

/-- Represents an instruction set. -/
def InstructionSet := List SwapInstruction

/-- Checks if a configuration is sorted (balls to the left of empty boxes). -/
def isSorted (config : BoxConfiguration N) : Prop :=
  ∀ i j, i < j → config i = Box.empty → config j = Box.empty

/-- Applies an instruction set to a configuration. -/
def applyInstructions (config : BoxConfiguration N) (instructions : InstructionSet) : BoxConfiguration N :=
  sorry

/-- The main theorem to be proved. -/
theorem instruction_set_exists (N : Nat) :
  ∃ (instructions : InstructionSet),
    instructions.length ≤ 100 * N ∧
    ∀ (config : BoxConfiguration N),
      ∃ (subset : InstructionSet),
        subset.length ≤ instructions.length ∧
        isSorted (applyInstructions config subset) :=
  sorry

end NUMINAMATH_CALUDE_instruction_set_exists_l211_21115


namespace NUMINAMATH_CALUDE_conference_drinks_l211_21123

theorem conference_drinks (total : ℕ) (coffee : ℕ) (juice : ℕ) (both : ℕ) :
  total = 30 →
  coffee = 15 →
  juice = 18 →
  both = 7 →
  total - (coffee + juice - both) = 4 :=
by sorry

end NUMINAMATH_CALUDE_conference_drinks_l211_21123


namespace NUMINAMATH_CALUDE_infinitely_many_primes_6k_plus_1_l211_21192

theorem infinitely_many_primes_6k_plus_1 :
  ∀ S : Finset Nat, (∀ p ∈ S, Prime p ∧ ∃ k, p = 6*k + 1) →
  ∃ q, Prime q ∧ (∃ m, q = 6*m + 1) ∧ q ∉ S :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_6k_plus_1_l211_21192


namespace NUMINAMATH_CALUDE_same_function_shifted_possible_same_function_different_variable_power_zero_not_always_one_same_domain_range_not_same_function_l211_21136

-- Define a function type
def RealFunction := ℝ → ℝ

-- Statement 1
theorem same_function_shifted_possible : 
  ∃ (f : RealFunction), ∀ x : ℝ, f x = f (x + 1) :=
sorry

-- Statement 2
theorem same_function_different_variable (f : RealFunction) :
  ∀ x t : ℝ, f x = f t :=
sorry

-- Statement 3
theorem power_zero_not_always_one :
  ∃ x : ℝ, x^0 ≠ 1 :=
sorry

-- Statement 4
theorem same_domain_range_not_same_function :
  ∃ (f g : RealFunction), (∀ x : ℝ, ∃ y : ℝ, f x = y ∧ g x = y) ∧ f ≠ g :=
sorry

end NUMINAMATH_CALUDE_same_function_shifted_possible_same_function_different_variable_power_zero_not_always_one_same_domain_range_not_same_function_l211_21136


namespace NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l211_21180

def U : Set Int := {-1, 0, 1, 2, 3, 4, 5}
def A : Set Int := {1, 2, 5}
def B : Set Int := {0, 1, 2, 3}

theorem intersection_of_B_and_complement_of_A : B ∩ (U \ A) = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_B_and_complement_of_A_l211_21180


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_condition_l211_21194

/-- Given three quadratic equations, this theorem states the necessary and sufficient condition
for each equation to have a common root with one another but not all share a single common root. -/
theorem quadratic_equations_common_root_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) : 
  let A := (a₁ + a₂ + a₃) / 2
  ∀ x₁ x₂ x₃ : ℝ,
  (x₁^2 - a₁*x₁ + b₁ = 0 ∧ 
   x₂^2 - a₂*x₂ + b₂ = 0 ∧ 
   x₃^2 - a₃*x₃ + b₃ = 0) →
  ((x₁ = x₂ ∨ x₂ = x₃ ∨ x₃ = x₁) ∧ 
   ¬(x₁ = x₂ ∧ x₂ = x₃)) ↔
  (b₁ = (A - a₂)*(A - a₃) ∧
   b₂ = (A - a₃)*(A - a₁) ∧
   b₃ = (A - a₁)*(A - a₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_condition_l211_21194


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_0000000033_l211_21195

/-- Expresses a given number in scientific notation -/
def scientific_notation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem scientific_notation_of_0_0000000033 :
  scientific_notation 0.0000000033 = (3.3, -9) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_0000000033_l211_21195


namespace NUMINAMATH_CALUDE_divisibility_implies_zero_product_l211_21147

theorem divisibility_implies_zero_product (p q r : ℝ) : 
  (∀ x, ∃ k, x^4 + 6*x^3 + 4*p*x^2 + 2*q*x + r = k * (x^3 + 4*x^2 + 2*x + 1)) →
  (p + q) * r = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_zero_product_l211_21147


namespace NUMINAMATH_CALUDE_rectangle_area_change_l211_21184

theorem rectangle_area_change (L B x : ℝ) (h1 : L > 0) (h2 : B > 0) (h3 : x > 0) : 
  (L + x / 100 * L) * (B - x / 100 * B) = 99 / 100 * (L * B) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l211_21184


namespace NUMINAMATH_CALUDE_jeremy_scrabble_score_l211_21160

/-- Calculates the score for a three-letter word in Scrabble with given letter values and a triple word score -/
def scrabble_score (first_letter_value : ℕ) (middle_letter_value : ℕ) (last_letter_value : ℕ) : ℕ :=
  3 * (first_letter_value + middle_letter_value + last_letter_value)

/-- Theorem: The score for Jeremy's word is 30 points -/
theorem jeremy_scrabble_score :
  scrabble_score 1 8 1 = 30 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_scrabble_score_l211_21160


namespace NUMINAMATH_CALUDE_square_perimeter_diagonal_ratio_l211_21122

theorem square_perimeter_diagonal_ratio 
  (s₁ s₂ : ℝ) 
  (h_positive₁ : s₁ > 0) 
  (h_positive₂ : s₂ > 0) 
  (h_perimeter_ratio : 4 * s₂ = 5 * (4 * s₁)) :
  s₂ * Real.sqrt 2 = 5 * (s₁ * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_diagonal_ratio_l211_21122


namespace NUMINAMATH_CALUDE_maple_trees_remaining_l211_21107

theorem maple_trees_remaining (initial_maples : Real) (cut_maples : Real) (remaining_maples : Real) : 
  initial_maples = 9.0 → cut_maples = 2.0 → remaining_maples = initial_maples - cut_maples → remaining_maples = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_remaining_l211_21107


namespace NUMINAMATH_CALUDE_minimize_distance_l211_21185

/-- Given points P(-2,-3) and Q(5,3) in the xy-plane, and R(2,m) chosen such that PR+RQ is minimized, prove that m = 3/7 -/
theorem minimize_distance (P Q R : ℝ × ℝ) (m : ℝ) :
  P = (-2, -3) →
  Q = (5, 3) →
  R = (2, m) →
  (∀ m' : ℝ, dist P R + dist R Q ≤ dist P (2, m') + dist (2, m') Q) →
  m = 3/7 := by
  sorry


end NUMINAMATH_CALUDE_minimize_distance_l211_21185


namespace NUMINAMATH_CALUDE_building_height_problem_l211_21139

theorem building_height_problem (h_taller h_shorter : ℝ) : 
  h_taller - h_shorter = 36 →
  h_shorter / h_taller = 5 / 7 →
  h_taller = 126 := by
  sorry

end NUMINAMATH_CALUDE_building_height_problem_l211_21139


namespace NUMINAMATH_CALUDE_probability_difference_l211_21172

-- Define the probabilities
def prob_plane : ℝ := 0.7
def prob_train : ℝ := 0.3
def prob_plane_ontime : ℝ := 0.8
def prob_train_ontime : ℝ := 0.9

-- Define the events
def event_plane_ontime : ℝ := prob_plane * prob_plane_ontime
def event_train_ontime : ℝ := prob_train * prob_train_ontime
def event_ontime : ℝ := event_plane_ontime + event_train_ontime

-- Theorem statement
theorem probability_difference :
  (event_plane_ontime / event_ontime) - (event_train_ontime / event_ontime) = 29 / 83 :=
by sorry

end NUMINAMATH_CALUDE_probability_difference_l211_21172


namespace NUMINAMATH_CALUDE_triangle_angle_max_l211_21109

theorem triangle_angle_max (c : ℝ) (X Y Z : ℝ) : 
  0 < X ∧ 0 < Y ∧ 0 < Z →  -- angles are positive
  X + Y + Z = 180 →  -- angle sum in a triangle
  Z ≤ Y ∧ Y ≤ X →  -- given order of angles
  c * X = 6 * Z →  -- given relation between X and Z
  Z ≤ 36 :=  -- maximum value of Z
by sorry

end NUMINAMATH_CALUDE_triangle_angle_max_l211_21109


namespace NUMINAMATH_CALUDE_solution_implies_m_minus_n_abs_l211_21198

/-- Given a system of equations 2x - y = m and x + my = n with solution x = 2 and y = 1, 
    prove that |m - n| = 2 -/
theorem solution_implies_m_minus_n_abs (m n : ℝ) 
  (h1 : 2 * 2 - 1 = m) 
  (h2 : 2 + m * 1 = n) : 
  |m - n| = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_minus_n_abs_l211_21198


namespace NUMINAMATH_CALUDE_solve_brownies_problem_l211_21106

def brownies_problem (initial_brownies : ℕ) (remaining_brownies : ℕ) : Prop :=
  let admin_brownies := initial_brownies / 2
  let after_admin := initial_brownies - admin_brownies
  let carl_brownies := after_admin / 2
  let after_carl := after_admin - carl_brownies
  let final_brownies := 3
  ∃ (simon_brownies : ℕ), 
    simon_brownies = after_carl - final_brownies ∧
    simon_brownies = 2

theorem solve_brownies_problem :
  brownies_problem 20 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_brownies_problem_l211_21106


namespace NUMINAMATH_CALUDE_quartic_polynomial_property_l211_21103

def Q (x : ℝ) (e f : ℝ) : ℝ := 3 * x^4 + 24 * x^3 + e * x^2 + f * x + 16

theorem quartic_polynomial_property (e f : ℝ) :
  (∀ r₁ r₂ r₃ r₄ : ℝ, Q r₁ e f = 0 ∧ Q r₂ e f = 0 ∧ Q r₃ e f = 0 ∧ Q r₄ e f = 0 →
    (-24 / 12 = e / 3) ∧
    (-24 / 12 = 3 + 24 + e + f + 16) ∧
    (e / 3 = 3 + 24 + e + f + 16)) →
  f = -39 := by
sorry

end NUMINAMATH_CALUDE_quartic_polynomial_property_l211_21103


namespace NUMINAMATH_CALUDE_min_sum_with_product_and_even_constraint_l211_21134

theorem min_sum_with_product_and_even_constraint (a b : ℤ) : 
  a * b = 72 → Even a → (∀ (x y : ℤ), x * y = 72 → Even x → a + b ≤ x + y) → a + b = -38 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_product_and_even_constraint_l211_21134


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l211_21167

theorem intersection_implies_a_value (P Q : Set ℕ) (a : ℕ) :
  P = {0, a} →
  Q = {1, 2} →
  (P ∩ Q).Nonempty →
  a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l211_21167


namespace NUMINAMATH_CALUDE_translation_right_2_units_l211_21116

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_2_units :
  let A : Point := { x := 3, y := -2 }
  let A' : Point := translateRight A 2
  A'.x = 5 ∧ A'.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_translation_right_2_units_l211_21116


namespace NUMINAMATH_CALUDE_probability_one_defective_part_l211_21132

/-- The probability of drawing exactly one defective part from a box containing 5 parts,
    of which 2 are defective, when randomly selecting 2 parts. -/
theorem probability_one_defective_part : 
  let total_parts : ℕ := 5
  let defective_parts : ℕ := 2
  let drawn_parts : ℕ := 2
  let total_ways := Nat.choose total_parts drawn_parts
  let favorable_ways := Nat.choose defective_parts 1 * Nat.choose (total_parts - defective_parts) (drawn_parts - 1)
  (favorable_ways : ℚ) / total_ways = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_defective_part_l211_21132


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l211_21155

/-- Given Ed and Sue's rollerblading, biking, and swimming rates and their total distances,
    prove that the sum of squares of the rates is 485. -/
theorem rates_sum_of_squares (r b s : ℕ) : 
  (2 * r + 3 * b + s = 80) →
  (4 * r + 2 * b + 3 * s = 98) →
  r^2 + b^2 + s^2 = 485 := by
  sorry

end NUMINAMATH_CALUDE_rates_sum_of_squares_l211_21155
