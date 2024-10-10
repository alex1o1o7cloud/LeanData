import Mathlib

namespace optimal_price_maximizes_revenue_l1288_128891

/-- Revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (150 - 6 * p)

/-- The optimal price maximizes the revenue --/
theorem optimal_price_maximizes_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → R p ≥ R q ∧
  p = 12.5 := by
  sorry

end optimal_price_maximizes_revenue_l1288_128891


namespace variance_2X_plus_1_l1288_128825

/-- A random variable following a Binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Variance of a Binomial distribution -/
def variance (X : BinomialDistribution) : ℝ :=
  X.n * X.p * (1 - X.p)

/-- Variance of a linear transformation of a random variable -/
def varianceLinearTransform (a b : ℝ) (X : BinomialDistribution) : ℝ :=
  a^2 * variance X

/-- Theorem: Variance of 2X+1 for X ~ B(10, 0.8) equals 6.4 -/
theorem variance_2X_plus_1 (X : BinomialDistribution) 
    (h2 : X.n = 10) (h3 : X.p = 0.8) : 
    varianceLinearTransform 2 1 X = 6.4 := by
  sorry

end variance_2X_plus_1_l1288_128825


namespace train_fraction_is_four_fifths_l1288_128831

/-- Proves that the fraction of the journey traveled by train is 0.8 -/
theorem train_fraction_is_four_fifths
  (D : ℝ) -- Total distance
  (h_D_pos : D > 0) -- Assume distance is positive
  (train_speed : ℝ) -- Train speed
  (h_train_speed : train_speed = 80) -- Train speed is 80 mph
  (car_speed : ℝ) -- Car speed
  (h_car_speed : car_speed = 20) -- Car speed is 20 mph
  (avg_speed : ℝ) -- Average speed
  (h_avg_speed : avg_speed = 50) -- Average speed is 50 mph
  (x : ℝ) -- Fraction of journey by train
  (h_x_range : 0 ≤ x ∧ x ≤ 1) -- x is between 0 and 1
  (h_speed_equation : D / ((x * D / train_speed) + ((1 - x) * D / car_speed)) = avg_speed) -- Speed equation
  : x = 4/5 := by
  sorry

end train_fraction_is_four_fifths_l1288_128831


namespace work_completion_time_l1288_128857

theorem work_completion_time (a b : ℝ) (h1 : b = 20) 
  (h2 : 4 * (1/a + 1/b) = 0.4666666666666667) : a = 15 := by
  sorry

end work_completion_time_l1288_128857


namespace smaller_rectangle_perimeter_is_9_l1288_128895

/-- Represents a rectangle with its dimensions and division properties. -/
structure DividedRectangle where
  perimeter : ℝ
  verticalCuts : ℕ
  horizontalCuts : ℕ
  smallRectangles : ℕ
  totalCutLength : ℝ

/-- Calculates the perimeter of a smaller rectangle given a DividedRectangle. -/
def smallRectanglePerimeter (r : DividedRectangle) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating the perimeter of each smaller rectangle is 9 cm under given conditions. -/
theorem smaller_rectangle_perimeter_is_9 (r : DividedRectangle) 
    (h1 : r.perimeter = 96)
    (h2 : r.verticalCuts = 8)
    (h3 : r.horizontalCuts = 11)
    (h4 : r.smallRectangles = 108)
    (h5 : r.totalCutLength = 438) :
    smallRectanglePerimeter r = 9 := by
  sorry

end smaller_rectangle_perimeter_is_9_l1288_128895


namespace hyperbola_ratio_l1288_128893

/-- Given a point M(x, 5/x) in the first quadrant on the hyperbola y = 5/x,
    with A(x, 0), B(0, 5/x), C(x, 3/x), and D(3/y, y) where y = 5/x,
    prove that the ratio CD:AB = 2:5 -/
theorem hyperbola_ratio (x : ℝ) (hx : x > 0) : 
  let y := 5 / x
  let m := (x, y)
  let a := (x, 0)
  let b := (0, y)
  let c := (x, 3 / x)
  let d := (3 / y, y)
  let cd := Real.sqrt ((x - 3 / y)^2 + (3 / x - y)^2)
  let ab := Real.sqrt ((x - 0)^2 + (0 - y)^2)
  cd / ab = 2 / 5 := by
sorry


end hyperbola_ratio_l1288_128893


namespace E_equals_F_l1288_128814

def E : Set ℝ := {x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3)}

def F : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}

theorem E_equals_F : E = F := by sorry

end E_equals_F_l1288_128814


namespace arrangements_equal_78_l1288_128896

/-- The number of different arrangements to select 2 workers for typesetting and 2 for printing
    from a group of 7 workers, where 5 are proficient in typesetting and 4 are proficient in printing. -/
def num_arrangements (total : ℕ) (typesetters : ℕ) (printers : ℕ) (typeset_needed : ℕ) (print_needed : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of arrangements is 78 -/
theorem arrangements_equal_78 :
  num_arrangements 7 5 4 2 2 = 78 :=
by sorry

end arrangements_equal_78_l1288_128896


namespace hyperbola_proof_l1288_128824

def polar_equation (ρ φ : ℝ) : Prop := ρ = 36 / (4 - 5 * Real.cos φ)

theorem hyperbola_proof (ρ φ : ℝ) (h : polar_equation ρ φ) :
  ∃ (a b : ℝ), 
    (a = 16 ∧ b = 12) ∧ 
    (∃ (e : ℝ), e > 1 ∧ ρ = (e * (b^2 / a)) / (1 - e * Real.cos φ)) :=
by sorry

end hyperbola_proof_l1288_128824


namespace find_number_l1288_128874

theorem find_number : ∃ x : ℝ, (((x - 1.9) * 1.5 + 32) / 2.5) = 20 ∧ x = 13.9 := by
  sorry

end find_number_l1288_128874


namespace flight_departure_time_l1288_128852

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Adds a duration in minutes to a Time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

theorem flight_departure_time :
  let checkInTime : ℕ := 120 -- 2 hours in minutes
  let drivingTime : ℕ := 45
  let parkingTime : ℕ := 15
  let latestDepartureTime : Time := ⟨17, 0, sorry⟩ -- 5:00 pm
  let flightDepartureTime : Time := addMinutes latestDepartureTime (checkInTime + drivingTime + parkingTime)
  flightDepartureTime = ⟨20, 0, sorry⟩ -- 8:00 pm
:= by sorry

end flight_departure_time_l1288_128852


namespace positive_A_value_l1288_128861

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 6 = 200) : A = 2 * Real.sqrt 41 :=
sorry

end positive_A_value_l1288_128861


namespace square_area_is_two_l1288_128856

/-- A complex number z is a vertex of a square with z^2 and z^4 if it satisfies the equation z^3 - iz + i - 1 = 0 -/
def is_square_vertex (z : ℂ) : Prop :=
  z ≠ 0 ∧ z^3 - Complex.I * z + Complex.I - 1 = 0

/-- The area of a square formed by z, z^2, and z^4 in the complex plane -/
noncomputable def square_area (z : ℂ) : ℝ :=
  (1/2) * Complex.abs (z^4 - z)^2

theorem square_area_is_two (z : ℂ) (h : is_square_vertex z) :
  square_area z = 2 :=
sorry

end square_area_is_two_l1288_128856


namespace flu_spread_l1288_128850

theorem flu_spread (initial_infected : ℕ) (total_infected : ℕ) (x : ℝ) : 
  initial_infected = 1 →
  total_infected = 81 →
  (1 + x)^2 = total_infected →
  x ≥ 0 →
  ∃ (y : ℝ), y = x ∧ (initial_infected : ℝ) + y + y^2 = total_infected :=
sorry

end flu_spread_l1288_128850


namespace sqrt_product_plus_one_l1288_128815

theorem sqrt_product_plus_one : 
  Real.sqrt ((25 : ℝ) * 24 * 23 * 22 + 1) = 551 := by sorry

end sqrt_product_plus_one_l1288_128815


namespace imaginary_part_of_z_l1288_128858

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) :
  z.im = -1/5 := by sorry

end imaginary_part_of_z_l1288_128858


namespace john_savings_after_interest_l1288_128855

/-- The percentage of earnings John will have left after one year, given his spending habits and bank interest rate --/
theorem john_savings_after_interest
  (earnings : ℝ)
  (rent_percent : ℝ)
  (dishwasher_percent : ℝ)
  (groceries_percent : ℝ)
  (interest_rate : ℝ)
  (h1 : rent_percent = 0.4)
  (h2 : dishwasher_percent = 0.7 * rent_percent)
  (h3 : groceries_percent = 1.15 * rent_percent)
  (h4 : interest_rate = 0.05)
  : (1 - (rent_percent + dishwasher_percent + groceries_percent)) * (1 + interest_rate) = 0.903 := by
  sorry

end john_savings_after_interest_l1288_128855


namespace algebraic_expression_value_l1288_128818

theorem algebraic_expression_value : 
  let x : ℝ := -1
  3 * x^2 + 2 * x - 1 = 0 := by sorry

end algebraic_expression_value_l1288_128818


namespace min_value_of_fraction_sum_l1288_128866

theorem min_value_of_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x + 9 / y) ≥ 25 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 4 / x + 9 / y = 25 :=
by sorry

end min_value_of_fraction_sum_l1288_128866


namespace angela_december_sleep_l1288_128890

/-- The number of hours Angela slept every night in December -/
def december_sleep_hours : ℝ := sorry

/-- The number of hours Angela slept every night in January -/
def january_sleep_hours : ℝ := 8.5

/-- The number of days in December -/
def december_days : ℕ := 31

/-- The number of days in January -/
def january_days : ℕ := 31

/-- The additional hours of sleep Angela got in January compared to December -/
def additional_sleep : ℝ := 62

theorem angela_december_sleep :
  december_sleep_hours = 6.5 :=
by
  sorry

end angela_december_sleep_l1288_128890


namespace campers_rowing_in_morning_l1288_128884

theorem campers_rowing_in_morning (afternoon_campers : ℕ) (difference : ℕ) 
  (h1 : afternoon_campers = 61)
  (h2 : afternoon_campers = difference + morning_campers) 
  (h3 : difference = 9) : 
  morning_campers = 52 :=
by
  sorry

end campers_rowing_in_morning_l1288_128884


namespace hundred_mile_fare_l1288_128803

/-- Represents the cost of a taxi journey based on distance traveled -/
structure TaxiFare where
  /-- The distance traveled in miles -/
  distance : ℝ
  /-- The cost of the journey in dollars -/
  cost : ℝ

/-- Taxi fare is directly proportional to the distance traveled -/
axiom fare_proportional (d₁ d₂ c₁ c₂ : ℝ) :
  d₁ * c₂ = d₂ * c₁

theorem hundred_mile_fare (f : TaxiFare) (h : f.distance = 80 ∧ f.cost = 192) :
  ∃ (g : TaxiFare), g.distance = 100 ∧ g.cost = 240 :=
sorry

end hundred_mile_fare_l1288_128803


namespace no_coin_exchange_solution_l1288_128836

theorem no_coin_exchange_solution : ¬∃ (x y z : ℕ), 
  x + y + z = 500 ∧ 
  36 * x + 6 * y + z = 3564 ∧ 
  x ≤ 99 := by
sorry

end no_coin_exchange_solution_l1288_128836


namespace discount_difference_l1288_128841

/-- Proves that the difference between the claimed discount and the true discount is 9% -/
theorem discount_difference (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) :
  initial_discount = 0.4 →
  additional_discount = 0.1 →
  claimed_discount = 0.55 →
  claimed_discount - (1 - (1 - initial_discount) * (1 - additional_discount)) = 0.09 := by
  sorry

end discount_difference_l1288_128841


namespace minimal_vertices_2007_gon_l1288_128823

/-- Given a regular polygon with n sides, returns the minimal number k such that
    among every k vertices of the polygon, there always exists 4 vertices forming
    a convex quadrilateral with 3 sides being sides of the polygon. -/
def minimalVerticesForQuadrilateral (n : ℕ) : ℕ :=
  ⌈(3 * n : ℚ) / 4⌉₊

theorem minimal_vertices_2007_gon :
  minimalVerticesForQuadrilateral 2007 = 1506 := by
  sorry

#eval minimalVerticesForQuadrilateral 2007

end minimal_vertices_2007_gon_l1288_128823


namespace smallest_coprime_to_180_seven_coprime_to_180_seven_is_smallest_coprime_to_180_l1288_128804

theorem smallest_coprime_to_180 : ∀ x : ℕ, x > 1 ∧ x < 7 → Nat.gcd x 180 ≠ 1 :=
by
  sorry

theorem seven_coprime_to_180 : Nat.gcd 7 180 = 1 :=
by
  sorry

theorem seven_is_smallest_coprime_to_180 : ∀ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 → x ≥ 7 :=
by
  sorry

end smallest_coprime_to_180_seven_coprime_to_180_seven_is_smallest_coprime_to_180_l1288_128804


namespace steven_seed_collection_l1288_128842

/-- Represents the number of seeds in different fruits -/
structure FruitSeeds where
  apple : ℕ
  pear : ℕ
  grape : ℕ

/-- Represents the number of fruits Steven has -/
structure FruitCount where
  apples : ℕ
  pears : ℕ
  grapes : ℕ

/-- Calculates the total number of seeds Steven needs to collect -/
def totalSeedsNeeded (avg : FruitSeeds) (count : FruitCount) (additional : ℕ) : ℕ :=
  avg.apple * count.apples + avg.pear * count.pears + avg.grape * count.grapes + additional

/-- Theorem: Steven needs to collect 60 seeds in total -/
theorem steven_seed_collection :
  let avg : FruitSeeds := ⟨6, 2, 3⟩
  let count : FruitCount := ⟨4, 3, 9⟩
  let additional : ℕ := 3
  totalSeedsNeeded avg count additional = 60 := by
  sorry


end steven_seed_collection_l1288_128842


namespace total_hike_length_l1288_128876

/-- The length of a hike given the distance hiked on the first day and the remaining distance. -/
def hike_length (first_day_distance : ℕ) (remaining_distance : ℕ) : ℕ :=
  first_day_distance + remaining_distance

/-- Theorem stating that the total length of the hike is 36 miles. -/
theorem total_hike_length :
  hike_length 9 27 = 36 := by
  sorry

end total_hike_length_l1288_128876


namespace d_magnitude_when_Q_has_five_roots_l1288_128883

/-- The polynomial Q(x) -/
def Q (d : ℂ) (x : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - d*x + 5) * (x^2 - 5*x + 10)

/-- The theorem stating that if Q has exactly 5 distinct roots, then |d| = √28 -/
theorem d_magnitude_when_Q_has_five_roots (d : ℂ) :
  (∃ (S : Finset ℂ), S.card = 5 ∧ (∀ x : ℂ, x ∈ S ↔ Q d x = 0) ∧ (∀ x y : ℂ, x ∈ S → y ∈ S → x ≠ y → Q d x = 0 → Q d y = 0 → x ≠ y)) →
  Complex.abs d = Real.sqrt 28 := by
sorry

end d_magnitude_when_Q_has_five_roots_l1288_128883


namespace hotel_charge_comparison_l1288_128868

theorem hotel_charge_comparison (P R G : ℝ) 
  (h1 : P = R - 0.55 * R) 
  (h2 : P = G - 0.1 * G) : 
  (R - G) / G = 1 := by
sorry

end hotel_charge_comparison_l1288_128868


namespace greatest_3digit_base9_divisible_by_7_l1288_128873

/-- Converts a base 9 number to decimal --/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (9 ^ i)) 0

/-- Checks if a number is a valid 3-digit base 9 number --/
def isValid3DigitBase9 (n : Nat) : Prop :=
  ∃ (d₁ d₂ d₃ : Nat), d₁ ≠ 0 ∧ d₁ < 9 ∧ d₂ < 9 ∧ d₃ < 9 ∧ n = base9ToDecimal [d₃, d₂, d₁]

theorem greatest_3digit_base9_divisible_by_7 :
  let n := base9ToDecimal [8, 8, 8]
  isValid3DigitBase9 n ∧ n % 7 = 0 ∧
  ∀ m, isValid3DigitBase9 m → m % 7 = 0 → m ≤ n :=
by sorry

end greatest_3digit_base9_divisible_by_7_l1288_128873


namespace output_is_fifteen_l1288_128862

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 2
  if step1 > 18 then step1 - 5 else step1 + 8

theorem output_is_fifteen : function_machine 10 = 15 := by sorry

end output_is_fifteen_l1288_128862


namespace equation_solution_l1288_128886

theorem equation_solution : ∃ x : ℝ, x ≠ 1 ∧ (x / (x - 1) + 2 / (1 - x) = 2) ∧ x = 0 := by
  sorry

end equation_solution_l1288_128886


namespace new_person_weight_l1288_128816

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 105 :=
by sorry

end new_person_weight_l1288_128816


namespace sum_of_digits_c_equals_five_l1288_128827

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def a : ℕ := sum_of_digits (4568^777)
def b : ℕ := sum_of_digits a
def c : ℕ := sum_of_digits b

theorem sum_of_digits_c_equals_five : c = 5 := by
  sorry

end sum_of_digits_c_equals_five_l1288_128827


namespace molly_rode_3285_miles_l1288_128800

/-- The number of miles Molly rode her bike from her 13th to 16th birthday -/
def molly_bike_miles : ℕ :=
  let start_age : ℕ := 13
  let end_age : ℕ := 16
  let years_riding : ℕ := end_age - start_age
  let days_per_year : ℕ := 365
  let miles_per_day : ℕ := 3
  years_riding * days_per_year * miles_per_day

/-- Theorem stating that Molly rode her bike for 3285 miles -/
theorem molly_rode_3285_miles : molly_bike_miles = 3285 := by
  sorry

end molly_rode_3285_miles_l1288_128800


namespace power_function_odd_l1288_128864

/-- A power function f(x) = (a-1)x^b passing through the point (a, 1/8) is odd. -/
theorem power_function_odd (a b : ℝ) (h : (a - 1) * a^b = 1/8) : 
  ∀ x ≠ 0, (a - 1) * (-x)^b = -((a - 1) * x^b) :=
by sorry

end power_function_odd_l1288_128864


namespace sum_of_three_numbers_l1288_128807

theorem sum_of_three_numbers (x y z : ℝ) : 
  z / x = 18.48 / 15.4 →
  z = 0.4 * y →
  x + y = 400 →
  x + y + z = 520 := by
sorry

end sum_of_three_numbers_l1288_128807


namespace f_composition_result_l1288_128821

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^3 else z^3

theorem f_composition_result :
  f (f (f (f (-1 + I)))) = -1.79841759e14 - 2.75930025e10 * I :=
by sorry

end f_composition_result_l1288_128821


namespace green_face_probability_l1288_128828

/-- The probability of rolling a green face on a 10-sided die with 3 green faces is 3/10. -/
theorem green_face_probability (total_faces : ℕ) (green_faces : ℕ) 
  (h1 : total_faces = 10) (h2 : green_faces = 3) : 
  (green_faces : ℚ) / total_faces = 3 / 10 := by
  sorry

end green_face_probability_l1288_128828


namespace john_boxes_l1288_128851

/-- The number of boxes each person has -/
structure Boxes where
  stan : ℕ
  joseph : ℕ
  jules : ℕ
  john : ℕ

/-- The conditions of the problem -/
def problem_conditions (b : Boxes) : Prop :=
  b.stan = 100 ∧
  b.joseph = b.stan - (80 * b.stan / 100) ∧
  b.jules = b.joseph + 5 ∧
  b.john > b.jules

/-- The theorem to prove -/
theorem john_boxes (b : Boxes) (h : problem_conditions b) : b.john = 30 := by
  sorry

end john_boxes_l1288_128851


namespace min_value_quadratic_max_value_quadratic_l1288_128830

-- Question 1
theorem min_value_quadratic (m : ℝ) : m^2 - 6*m + 10 ≥ 1 := by sorry

-- Question 2
theorem max_value_quadratic (x : ℝ) : -2*x^2 - 4*x + 3 ≤ 5 := by sorry

end min_value_quadratic_max_value_quadratic_l1288_128830


namespace square_side_difference_l1288_128834

/-- Given four squares with side lengths s₁ ≥ s₂ ≥ s₃ ≥ s₄, prove that s₁ - s₄ = 29 -/
theorem square_side_difference (s₁ s₂ s₃ s₄ : ℝ) 
  (h₁ : s₁ ≥ s₂) (h₂ : s₂ ≥ s₃) (h₃ : s₃ ≥ s₄)
  (ab : s₁ - s₂ = 11) (cd : s₂ - s₃ = 5) (fe : s₃ - s₄ = 13) :
  s₁ - s₄ = 29 := by
  sorry

end square_side_difference_l1288_128834


namespace distribution_proportion_l1288_128889

theorem distribution_proportion (total : ℚ) (p q r s : ℚ) : 
  total = 1000 →
  p = 2 * q →
  s = 4 * r →
  s - p = 250 →
  p + q + r + s = total →
  q / r = 1 := by
  sorry

end distribution_proportion_l1288_128889


namespace exists_unique_marking_scheme_l1288_128881

/-- Represents a cell in the grid -/
structure Cell :=
  (row : Nat)
  (col : Nat)

/-- Represents a marking scheme for the grid -/
def MarkingScheme := Set Cell

/-- Represents a 10x10 sub-square in the grid -/
structure SubSquare :=
  (topLeft : Cell)

/-- Counts the number of marked cells in a sub-square -/
def countMarkedCells (scheme : MarkingScheme) (square : SubSquare) : Nat :=
  sorry

/-- Checks if all sub-squares have unique counts -/
def allSubSquaresUnique (scheme : MarkingScheme) : Prop :=
  sorry

/-- Main theorem: There exists a marking scheme where all sub-squares have unique counts -/
theorem exists_unique_marking_scheme :
  ∃ (scheme : MarkingScheme),
    (∀ c : Cell, c.row < 19 ∧ c.col < 19) →
    (∀ s : SubSquare, s.topLeft.row ≤ 9 ∧ s.topLeft.col ≤ 9) →
    allSubSquaresUnique scheme :=
  sorry

end exists_unique_marking_scheme_l1288_128881


namespace repeating_decimal_proof_l1288_128867

/-- The repeating decimal 0.76204̄ as a rational number -/
def repeating_decimal : ℚ := 761280 / 999000

theorem repeating_decimal_proof : repeating_decimal = 0.76 + (204 : ℚ) / 999000 := by sorry


end repeating_decimal_proof_l1288_128867


namespace third_number_is_41_l1288_128863

/-- A sequence of six numbers with specific properties -/
def GoldStickerSequence (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₁ = 29 ∧ 
  a₂ = 35 ∧ 
  a₄ = 47 ∧ 
  a₅ = 53 ∧ 
  a₆ = 59 ∧ 
  a₂ - a₁ = 6 ∧ 
  a₄ - a₂ = 12 ∧ 
  a₆ - a₄ = 12

theorem third_number_is_41 {a₁ a₂ a₃ a₄ a₅ a₆ : ℕ} 
  (h : GoldStickerSequence a₁ a₂ a₃ a₄ a₅ a₆) : a₃ = 41 :=
by
  sorry

end third_number_is_41_l1288_128863


namespace pq_is_one_eighth_of_rs_l1288_128880

-- Define the line segment RS and points P and Q on it
structure LineSegment where
  length : ℝ

structure Point where
  position : ℝ

-- Define the problem setup
def problem (RS : LineSegment) (P Q : Point) : Prop :=
  -- P and Q lie on RS
  0 ≤ P.position ∧ P.position ≤ RS.length ∧
  0 ≤ Q.position ∧ Q.position ≤ RS.length ∧
  -- RP is 3 times PS
  P.position = (3/4) * RS.length ∧
  -- RQ is 7 times QS
  Q.position = (7/8) * RS.length

-- Theorem to prove
theorem pq_is_one_eighth_of_rs (RS : LineSegment) (P Q : Point) 
  (h : problem RS P Q) : 
  abs (Q.position - P.position) = (1/8) * RS.length :=
sorry

end pq_is_one_eighth_of_rs_l1288_128880


namespace y_value_l1288_128801

theorem y_value (m : ℕ) (y : ℝ) 
  (h1 : ((1 ^ m) / (y ^ m)) * ((1 ^ 16) / (4 ^ 16)) = 1 / (2 * (10 ^ 31)))
  (h2 : m = 31) : 
  y = 5 := by
  sorry

end y_value_l1288_128801


namespace double_and_square_reverse_digits_l1288_128835

/-- For any base greater than 2, doubling (base - 1) and squaring (base - 1) 
    result in numbers with the same digits in reverse order. -/
theorem double_and_square_reverse_digits (a : ℕ) (h : a > 2) :
  ∃ (d₁ d₂ : ℕ), d₁ < a ∧ d₂ < a ∧ 
  2 * (a - 1) = d₁ * a + d₂ ∧
  (a - 1)^2 = d₂ * a + d₁ :=
sorry

end double_and_square_reverse_digits_l1288_128835


namespace gcd_of_128_144_512_l1288_128899

theorem gcd_of_128_144_512 : Nat.gcd 128 (Nat.gcd 144 512) = 16 := by sorry

end gcd_of_128_144_512_l1288_128899


namespace orange_juice_remaining_l1288_128806

theorem orange_juice_remaining (initial_amount : ℚ) (given_away : ℚ) : 
  initial_amount = 5 → given_away = 18/7 → initial_amount - given_away = 17/7 := by
  sorry

end orange_juice_remaining_l1288_128806


namespace simplify_nested_expression_l1288_128847

theorem simplify_nested_expression (x : ℝ) :
  2 * (1 - (2 * (1 - (1 + (2 - (3 * x)))))) = -10 + 12 * x := by
  sorry

end simplify_nested_expression_l1288_128847


namespace center_equidistant_from_hexagon_vertices_l1288_128822

/-- Represents a nickel coin -/
structure Nickel where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- States that a circle's diameter is equal to a nickel's diameter -/
def circle_diameter_eq_nickel (c : Circle) (n : Nickel) : Prop :=
  c.radius * 2 = n.diameter

/-- States that a hexagon is inscribed in a circle -/
def hexagon_inscribed_in_circle (h : RegularHexagon) (c : Circle) : Prop :=
  ∀ i : Fin 6, dist c.center (h.vertices i) = c.radius

/-- States that a hexagon can be constructed using three nickels -/
def hexagon_constructible_with_nickels (h : RegularHexagon) (n : Nickel) : Prop :=
  ∀ i : Fin 6, ∀ j : Fin 6, i ≠ j → dist (h.vertices i) (h.vertices j) = n.diameter

/-- The main theorem -/
theorem center_equidistant_from_hexagon_vertices
  (c : Circle) (n : Nickel) (h : RegularHexagon)
  (h1 : circle_diameter_eq_nickel c n)
  (h2 : hexagon_inscribed_in_circle h c)
  (h3 : hexagon_constructible_with_nickels h n) :
  ∀ i j : Fin 6, dist c.center (h.vertices i) = dist c.center (h.vertices j) := by
  sorry

end center_equidistant_from_hexagon_vertices_l1288_128822


namespace managers_salary_l1288_128837

/-- Proves that given 24 employees with an average salary of Rs. 2400, 
    if adding a manager's salary increases the average by Rs. 100, 
    then the manager's salary is Rs. 4900. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (salary_increase : ℕ) : 
  num_employees = 24 → 
  avg_salary = 2400 → 
  salary_increase = 100 → 
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - 
   num_employees * avg_salary) = 4900 := by
  sorry

#check managers_salary

end managers_salary_l1288_128837


namespace right_triangle_among_options_l1288_128838

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_among_options : 
  is_right_triangle 1 2 3 ∧ 
  ¬is_right_triangle 3 4 5 ∧ 
  ¬is_right_triangle 6 8 10 ∧ 
  ¬is_right_triangle 5 10 12 :=
by sorry

end right_triangle_among_options_l1288_128838


namespace point_M_on_x_axis_l1288_128808

-- Define a point M with coordinates (a+2, a-3)
def M (a : ℝ) : ℝ × ℝ := (a + 2, a - 3)

-- Define what it means for a point to lie on the x-axis
def lies_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Theorem statement
theorem point_M_on_x_axis :
  ∀ a : ℝ, lies_on_x_axis (M a) → M a = (5, 0) := by
  sorry

end point_M_on_x_axis_l1288_128808


namespace sum_congruence_l1288_128871

def large_sum : ℕ := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem sum_congruence : large_sum % 9 = 6 := by
  sorry

end sum_congruence_l1288_128871


namespace jeanine_has_more_pencils_jeanine_has_more_pencils_proof_l1288_128888

/-- The number of pencils Jeanine has after giving some to Abby is 3 more than Clare's pencils. -/
theorem jeanine_has_more_pencils : ℕ → Prop :=
  fun (initial_pencils : ℕ) =>
    initial_pencils = 18 →
    let clare_pencils := initial_pencils / 2
    let jeanine_remaining := initial_pencils - (initial_pencils / 3)
    jeanine_remaining - clare_pencils = 3

/-- Proof of the theorem -/
theorem jeanine_has_more_pencils_proof : jeanine_has_more_pencils 18 := by
  sorry

#check jeanine_has_more_pencils_proof

end jeanine_has_more_pencils_jeanine_has_more_pencils_proof_l1288_128888


namespace training_hours_calculation_l1288_128802

/-- Calculates the total training hours given daily training hours, initial training days, and additional training days. -/
def total_training_hours (daily_hours : ℕ) (initial_days : ℕ) (additional_days : ℕ) : ℕ :=
  daily_hours * (initial_days + additional_days)

/-- Theorem stating that training 5 hours daily for 30 days and continuing for 12 more days results in 210 total training hours. -/
theorem training_hours_calculation : total_training_hours 5 30 12 = 210 := by
  sorry

end training_hours_calculation_l1288_128802


namespace smallest_product_l1288_128820

def S : Finset ℤ := {-9, -5, -1, 1, 4}

theorem smallest_product (a b : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
  ∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x ≠ y), 
    x * y ≤ a * b ∧ x * y = -36 := by
  sorry

end smallest_product_l1288_128820


namespace least_odd_prime_factor_of_2023_pow_6_plus_1_l1288_128846

theorem least_odd_prime_factor_of_2023_pow_6_plus_1 :
  (Nat.minFac (2023^6 + 1)) = 37 := by
  sorry

end least_odd_prime_factor_of_2023_pow_6_plus_1_l1288_128846


namespace triangle_properties_l1288_128843

theorem triangle_properties (A B C : ℝ × ℝ) (S : ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  AB.1 * AC.1 + AB.2 * AC.2 = S →
  (C.1 - A.1) / Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3/5 →
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 2 →
  Real.tan (2 * Real.arctan ((B.2 - A.2) / (B.1 - A.1))) = -4/3 ∧
  S = 8/5 := by
sorry

end triangle_properties_l1288_128843


namespace absolute_value_sum_l1288_128849

theorem absolute_value_sum (x q : ℝ) : 
  |x - 5| = q ∧ x > 5 → x + q = 2*q + 5 := by
sorry

end absolute_value_sum_l1288_128849


namespace quadratic_equation_roots_l1288_128865

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*x₁ - 9 = 0) ∧ (x₂^2 - 2*x₂ - 9 = 0) := by
  sorry

end quadratic_equation_roots_l1288_128865


namespace sufficient_not_necessary_l1288_128879

theorem sufficient_not_necessary (a b : ℝ) :
  (0 < a ∧ a < b → 1 / a > 1 / b) ∧
  ∃ a b : ℝ, 1 / a > 1 / b ∧ ¬(0 < a ∧ a < b) :=
sorry

end sufficient_not_necessary_l1288_128879


namespace remainder_divisibility_l1288_128844

theorem remainder_divisibility (x y : ℤ) (h : 9 ∣ (x + 2*y)) :
  ∃ k : ℤ, 2*(5*x - 8*y - 4) = 9*k + (-8) ∨ 2*(5*x - 8*y - 4) = 9*k + 1 := by
  sorry

end remainder_divisibility_l1288_128844


namespace tangent_line_equation_l1288_128811

noncomputable def f (x : ℝ) : ℝ := Real.log ((2 - x) / (2 + x))

theorem tangent_line_equation :
  let x₀ : ℝ := -1
  let y₀ : ℝ := f x₀
  let m : ℝ := -4/3
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (y = -4/3 * x + Real.log 3 - 4/3) :=
by sorry

end tangent_line_equation_l1288_128811


namespace geometric_sequence_problem_l1288_128885

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_sum : a 4 + a 8 = -2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
  sorry

end geometric_sequence_problem_l1288_128885


namespace valid_midpoint_on_hyperbola_l1288_128860

/-- The hyperbola equation --/
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

/-- Definition of midpoint --/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

/-- Theorem stating that (-1,-4) is the only valid midpoint --/
theorem valid_midpoint_on_hyperbola :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_hyperbola x₁ y₁ ∧
    is_on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ ∧
    (∀ (x y : ℝ), (x, y) ∈ [(1, 1), (-1, 2), (1, 3)] →
      ¬∃ (x₁' y₁' x₂' y₂' : ℝ),
        is_on_hyperbola x₁' y₁' ∧
        is_on_hyperbola x₂' y₂' ∧
        is_midpoint x y x₁' y₁' x₂' y₂') :=
by sorry

end valid_midpoint_on_hyperbola_l1288_128860


namespace min_cost_is_128_l1288_128870

/-- Represents the cost of each type of flower -/
structure FlowerCost where
  sunflower : ℕ
  tulip : ℕ
  orchid : ℚ
  rose : ℕ
  hydrangea : ℕ

/-- Represents the areas of different regions in the garden -/
structure GardenRegions where
  small_region1 : ℕ
  small_region2 : ℕ
  medium_region : ℕ
  large_region : ℕ

/-- Calculates the minimum cost of the garden given the flower costs and garden regions -/
def min_garden_cost (costs : FlowerCost) (regions : GardenRegions) : ℚ :=
  costs.sunflower * regions.small_region1 +
  costs.sunflower * regions.small_region2 +
  costs.tulip * regions.medium_region +
  costs.hydrangea * regions.large_region

theorem min_cost_is_128 (costs : FlowerCost) (regions : GardenRegions) :
  costs.sunflower = 1 ∧ 
  costs.tulip = 2 ∧ 
  costs.orchid = 5/2 ∧ 
  costs.rose = 3 ∧ 
  costs.hydrangea = 4 ∧
  regions.small_region1 = 8 ∧
  regions.small_region2 = 8 ∧
  regions.medium_region = 6 ∧
  regions.large_region = 25 →
  min_garden_cost costs regions = 128 := by
  sorry

end min_cost_is_128_l1288_128870


namespace min_value_theorem_l1288_128840

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 4) :
  (1 / (4 - x) + 2 / x) ≥ (3 + 2 * Real.sqrt 2) / 4 :=
by sorry

end min_value_theorem_l1288_128840


namespace largest_divided_by_smallest_l1288_128819

theorem largest_divided_by_smallest : 
  let numbers : List ℝ := [10, 11, 12, 13]
  (List.maximum numbers).get! / (List.minimum numbers).get! = 1.3 := by
sorry

end largest_divided_by_smallest_l1288_128819


namespace function_composition_equality_l1288_128887

theorem function_composition_equality (a : ℝ) (h_pos : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x + 1 / Real.sqrt 2
  f (f (1 / Real.sqrt 2)) = f 0 → a = 0 := by
  sorry

end function_composition_equality_l1288_128887


namespace b_52_mod_55_l1288_128892

/-- Definition of b_n as the integer obtained by writing all integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- Theorem stating that the remainder of b_52 divided by 55 is 2 -/
theorem b_52_mod_55 : b 52 % 55 = 2 := by sorry

end b_52_mod_55_l1288_128892


namespace ellipse_foci_l1288_128809

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 25 + y^2 / 169 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(0, 12), (0, -12)}

-- Theorem statement
theorem ellipse_foci :
  ∀ (x y : ℝ), ellipse_equation x y →
  ∃ (f₁ f₂ : ℝ × ℝ), f₁ ∈ foci ∧ f₂ ∈ foci ∧
  (x - f₁.1)^2 + (y - f₁.2)^2 + (x - f₂.1)^2 + (y - f₂.2)^2 =
  4 * Real.sqrt (13^2 * 5^2) :=
sorry

end ellipse_foci_l1288_128809


namespace continuity_at_one_l1288_128805

def f (x : ℝ) := -4 * x^2 - 7

theorem continuity_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end continuity_at_one_l1288_128805


namespace q_div_p_equals_450_l1288_128829

def total_slips : ℕ := 50
def num_range : ℕ := 10
def slips_per_num : ℕ := 5
def drawn_slips : ℕ := 5

def p : ℚ := num_range / (total_slips.choose drawn_slips)
def q : ℚ := (num_range.choose 2) * (slips_per_num.choose 3) * (slips_per_num.choose 2) / (total_slips.choose drawn_slips)

theorem q_div_p_equals_450 : q / p = 450 := by sorry

end q_div_p_equals_450_l1288_128829


namespace math_festival_divisibility_l1288_128832

/-- The year of the first math festival -/
def first_festival_year : ℕ := 1990

/-- The base year for calculating festival years -/
def base_year : ℕ := 1989

/-- Predicate to check if a given ordinal number satisfies the divisibility condition -/
def satisfies_condition (N : ℕ) : Prop :=
  (base_year + N) % N = 0

theorem math_festival_divisibility :
  (∃ (first : ℕ), first > 0 ∧ satisfies_condition first ∧
    ∀ (k : ℕ), 0 < k ∧ k < first → ¬satisfies_condition k) ∧
  (∃ (last : ℕ), satisfies_condition last ∧
    ∀ (k : ℕ), k > last → ¬satisfies_condition k) :=
sorry

end math_festival_divisibility_l1288_128832


namespace ryan_study_difference_l1288_128882

/-- Ryan's daily study schedule -/
structure StudySchedule where
  english_hours : ℕ
  chinese_hours : ℕ

/-- The difference in hours between English and Chinese study time -/
def study_time_difference (schedule : StudySchedule) : ℤ :=
  schedule.english_hours - schedule.chinese_hours

/-- Theorem: Ryan spends 4 more hours on English than Chinese -/
theorem ryan_study_difference :
  ∀ (schedule : StudySchedule),
  schedule.english_hours = 6 →
  schedule.chinese_hours = 2 →
  study_time_difference schedule = 4 := by
sorry

end ryan_study_difference_l1288_128882


namespace f_inequality_solution_f_minimum_value_condition_l1288_128872

def f (a : ℝ) (x : ℝ) : ℝ := |3*x - 1| + a*x + 3

theorem f_inequality_solution (a : ℝ) (h : a = 1) :
  {x : ℝ | f a x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 1/2} :=
sorry

theorem f_minimum_value_condition (a : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a x ≤ f a y) ↔ -3 ≤ a ∧ a ≤ 3 :=
sorry

end f_inequality_solution_f_minimum_value_condition_l1288_128872


namespace student_handshake_problem_l1288_128854

/-- Given an m x n array of students where m, n ≥ 3, if each student shakes hands
    with adjacent students (horizontally, vertically, or diagonally) and there
    are 1020 handshakes in total, then the total number of students N is 140. -/
theorem student_handshake_problem (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (8 * m * n - 6 * m - 6 * n + 4) / 2 = 1020 →
  m * n = 140 := by
  sorry

#check student_handshake_problem

end student_handshake_problem_l1288_128854


namespace count_perfect_square_factors_equals_3850_l1288_128848

def prime_factorization := (2, 12) :: (3, 18) :: (5, 20) :: (7, 8) :: []

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def count_perfect_square_factors (factorization : List (ℕ × ℕ)) : ℕ :=
  factorization.foldl (fun acc (p, e) => acc * ((e / 2) + 1)) 1

theorem count_perfect_square_factors_equals_3850 :
  count_perfect_square_factors prime_factorization = 3850 := by
  sorry

end count_perfect_square_factors_equals_3850_l1288_128848


namespace apple_cost_theorem_l1288_128833

theorem apple_cost_theorem (cost_two_dozen : ℝ) (h : cost_two_dozen = 15.60) :
  let cost_per_dozen : ℝ := cost_two_dozen / 2
  let cost_four_dozen : ℝ := 4 * cost_per_dozen
  cost_four_dozen = 31.20 := by
sorry

end apple_cost_theorem_l1288_128833


namespace box_weight_is_42_l1288_128878

/-- The weight of a box of books -/
def box_weight (book_weight : ℕ) (num_books : ℕ) : ℕ :=
  book_weight * num_books

/-- Theorem: The weight of a box of books is 42 pounds -/
theorem box_weight_is_42 : box_weight 3 14 = 42 := by
  sorry

end box_weight_is_42_l1288_128878


namespace sqrt_equation_solution_l1288_128894

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (9 + 3 * z) = 12 :=
by
  -- Proof goes here
  sorry

end sqrt_equation_solution_l1288_128894


namespace gcd_lcm_product_24_36_l1288_128897

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end gcd_lcm_product_24_36_l1288_128897


namespace equation_simplification_l1288_128812

theorem equation_simplification (y : ℝ) (S : ℝ) :
  5 * (2 * y + 3 * Real.sqrt 3) = S →
  10 * (4 * y + 6 * Real.sqrt 3) = 4 * S := by
sorry

end equation_simplification_l1288_128812


namespace chocolate_theorem_l1288_128898

/-- The number of chocolates Nick has -/
def nick_chocolates : ℕ := 10

/-- The factor by which Alix's chocolates exceed Nick's -/
def alix_factor : ℕ := 3

/-- The number of chocolates mom took from Alix -/
def mom_took : ℕ := 5

/-- The difference in chocolates between Alix and Nick after mom took some -/
def chocolate_difference : ℕ := 15

theorem chocolate_theorem :
  (alix_factor * nick_chocolates - mom_took) - nick_chocolates = chocolate_difference := by
  sorry

end chocolate_theorem_l1288_128898


namespace problem_statement_l1288_128859

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 + 2*a*b = -2) 
  (h2 : a*b - b^2 = -4) : 
  2*a^2 + (7/2)*a*b + (1/2)*b^2 = -2 := by
sorry

end problem_statement_l1288_128859


namespace salary_increase_proof_l1288_128875

/-- Proves that given the conditions of the salary increase problem, the new salary is $90,000 -/
theorem salary_increase_proof (S : ℝ) 
  (h1 : S + 25000 = S * (1 + 0.3846153846153846)) : S + 25000 = 90000 := by
  sorry

end salary_increase_proof_l1288_128875


namespace least_coins_l1288_128826

theorem least_coins (n : ℕ) : 
  (n > 0) → 
  (n % 7 = 3) → 
  (n % 5 = 4) → 
  (∀ m : ℕ, m > 0 → m % 7 = 3 → m % 5 = 4 → n ≤ m) → 
  n = 24 :=
by sorry

end least_coins_l1288_128826


namespace shirt_price_calculation_l1288_128877

def shorts_price : ℝ := 15
def jacket_price : ℝ := 14.82
def total_spent : ℝ := 42.33

theorem shirt_price_calculation : 
  ∃ (shirt_price : ℝ), shirt_price = total_spent - (shorts_price + jacket_price) ∧ shirt_price = 12.51 :=
by sorry

end shirt_price_calculation_l1288_128877


namespace alexander_rearrangements_l1288_128817

theorem alexander_rearrangements (name_length : ℕ) (rearrangements_per_minute : ℕ) : 
  name_length = 9 → rearrangements_per_minute = 15 → 
  (Nat.factorial name_length / rearrangements_per_minute : ℚ) / 60 = 403.2 := by
  sorry

end alexander_rearrangements_l1288_128817


namespace sum_of_B_coordinates_l1288_128845

-- Define the points
def A : ℝ × ℝ := (5, -1)
def M : ℝ × ℝ := (4, 3)

-- Define B as a variable point
variable (B : ℝ × ℝ)

-- Define the midpoint condition
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem sum_of_B_coordinates :
  is_midpoint M A B → B.1 + B.2 = 10 := by
  sorry

end sum_of_B_coordinates_l1288_128845


namespace sum_of_five_terms_l1288_128839

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_five_terms (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 + a 15 = 6 →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by sorry

end sum_of_five_terms_l1288_128839


namespace ab_value_proof_l1288_128853

theorem ab_value_proof (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (2 - i) * (a - b * i) = (-8 - i) * i) : a * b = 42 := by
  sorry

end ab_value_proof_l1288_128853


namespace minimum_room_size_for_table_l1288_128813

theorem minimum_room_size_for_table (table_length : ℝ) (table_width : ℝ) 
  (h1 : table_length = 12) (h2 : table_width = 9) : 
  ∃ (S : ℕ), S = 15 ∧ 
  (∀ (room_size : ℕ), (Real.sqrt (table_length^2 + table_width^2) ≤ room_size) ↔ (S ≤ room_size)) :=
by sorry

end minimum_room_size_for_table_l1288_128813


namespace cone_volume_l1288_128869

/-- Given a cone with generatrix length 2 and unfolded side sector area 2π, its volume is (√3 * π) / 3 -/
theorem cone_volume (generatrix : ℝ) (sector_area : ℝ) :
  generatrix = 2 →
  sector_area = 2 * Real.pi →
  ∃ (volume : ℝ), volume = (Real.sqrt 3 * Real.pi) / 3 :=
by sorry

end cone_volume_l1288_128869


namespace install_remaining_windows_time_l1288_128810

/-- Calculates the time needed to install remaining windows -/
def time_to_install_remaining (total : ℕ) (installed : ℕ) (time_per_window : ℕ) : ℕ :=
  (total - installed) * time_per_window

/-- Proves that the time to install the remaining windows is 20 hours -/
theorem install_remaining_windows_time :
  time_to_install_remaining 10 6 5 = 20 := by
  sorry

end install_remaining_windows_time_l1288_128810
