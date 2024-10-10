import Mathlib

namespace smallest_prime_12_less_than_square_l2304_230450

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem smallest_prime_12_less_than_square : 
  ∃ (n : ℕ) (k : ℕ), 
    n > 0 ∧ 
    is_prime n ∧ 
    n = k^2 - 12 ∧ 
    ∀ (m : ℕ) (j : ℕ), m > 0 → is_prime m → m = j^2 - 12 → n ≤ m :=
by
  sorry

end smallest_prime_12_less_than_square_l2304_230450


namespace first_worker_time_l2304_230459

/-- Given two workers loading a truck, prove that the first worker's time is 5 hours. -/
theorem first_worker_time (T : ℝ) : 
  T > 0 →  -- The time must be positive
  (1 / T + 1 / 4 : ℝ) = 1 / 2.2222222222222223 → 
  T = 5 := by 
sorry

end first_worker_time_l2304_230459


namespace sum_of_two_squares_equivalence_l2304_230445

theorem sum_of_two_squares_equivalence (n : ℕ) (hn : n > 0) :
  (∃ a b : ℤ, n = a^2 + b^2) ↔ (∃ A B : ℤ, 2 * n = A^2 + B^2) :=
sorry

end sum_of_two_squares_equivalence_l2304_230445


namespace car_trip_speed_l2304_230427

/-- Proves that given the conditions of the car trip, the return speed must be 37.5 mph -/
theorem car_trip_speed (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) :
  distance = 150 →
  speed_ab = 75 →
  avg_speed = 50 →
  ∃ speed_ba : ℝ,
    speed_ba = 37.5 ∧
    avg_speed = (2 * distance) / (distance / speed_ab + distance / speed_ba) :=
by sorry

end car_trip_speed_l2304_230427


namespace camera_rental_theorem_l2304_230490

def camera_rental_problem (camera_value : ℝ) (rental_weeks : ℕ) 
  (base_fee_rate : ℝ) (high_demand_rate : ℝ) (low_demand_rate : ℝ)
  (insurance_rate : ℝ) (sales_tax_rate : ℝ)
  (mike_contribution_rate : ℝ) (sarah_contribution_rate : ℝ) (sarah_contribution_cap : ℝ)
  (alex_contribution_rate : ℝ) (alex_contribution_cap : ℝ) : Prop :=
  let base_fee := camera_value * base_fee_rate
  let high_demand_fee := base_fee + (camera_value * high_demand_rate)
  let low_demand_fee := base_fee - (camera_value * low_demand_rate)
  let total_rental_fee := 2 * high_demand_fee + 2 * low_demand_fee
  let insurance_fee := camera_value * insurance_rate
  let subtotal := total_rental_fee + insurance_fee
  let total_cost := subtotal + (subtotal * sales_tax_rate)
  let mike_contribution := total_cost * mike_contribution_rate
  let sarah_contribution := min (total_cost * sarah_contribution_rate) sarah_contribution_cap
  let alex_contribution := min (total_cost * alex_contribution_rate) alex_contribution_cap
  let total_contribution := mike_contribution + sarah_contribution + alex_contribution
  let john_payment := total_cost - total_contribution
  john_payment = 1015.20

theorem camera_rental_theorem : 
  camera_rental_problem 5000 4 0.10 0.03 0.02 0.05 0.08 0.20 0.30 1000 0.10 700 := by
  sorry

#check camera_rental_theorem

end camera_rental_theorem_l2304_230490


namespace cylinder_max_volume_ratio_l2304_230452

/-- The ratio of height to base radius of a cylinder with surface area 6π when its volume is maximized -/
theorem cylinder_max_volume_ratio : 
  ∃ (h r : ℝ), 
    h > 0 ∧ r > 0 ∧  -- Ensure positive height and radius
    2 * π * r^2 + 2 * π * r * h = 6 * π ∧  -- Surface area condition
    (∀ (h' r' : ℝ), 
      h' > 0 ∧ r' > 0 ∧ 
      2 * π * r'^2 + 2 * π * r' * h' = 6 * π → 
      π * r^2 * h ≥ π * r'^2 * h') →  -- Volume maximization condition
    h / r = 2 := by
  sorry

end cylinder_max_volume_ratio_l2304_230452


namespace inequality_proof_l2304_230484

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) := by
  sorry

end inequality_proof_l2304_230484


namespace intersection_points_theorem_l2304_230408

/-- A triangle with marked points on two sides -/
structure MarkedTriangle where
  -- The number of points marked on side BC
  pointsOnBC : ℕ
  -- The number of points marked on side AB
  pointsOnAB : ℕ
  -- Ensure the points are distinct from vertices
  distinctPoints : pointsOnBC > 0 ∧ pointsOnAB > 0

/-- The number of intersection points formed by connecting marked points -/
def intersectionPoints (t : MarkedTriangle) : ℕ := t.pointsOnBC * t.pointsOnAB

/-- Theorem: The number of intersection points in a triangle with 60 points on BC and 50 points on AB is 3000 -/
theorem intersection_points_theorem (t : MarkedTriangle) 
  (h1 : t.pointsOnBC = 60) (h2 : t.pointsOnAB = 50) : 
  intersectionPoints t = 3000 := by
  sorry

end intersection_points_theorem_l2304_230408


namespace mens_tshirt_interval_l2304_230472

/-- Represents the shop selling T-shirts -/
structure TShirtShop where
  womens_interval : ℕ  -- Minutes between women's T-shirt sales
  womens_price : ℕ     -- Price of women's T-shirts in dollars
  mens_price : ℕ        -- Price of men's T-shirts in dollars
  daily_hours : ℕ      -- Hours open per day
  weekly_days : ℕ      -- Days open per week
  weekly_revenue : ℕ   -- Total weekly revenue in dollars

/-- Calculates the interval between men's T-shirt sales -/
def mens_interval (shop : TShirtShop) : ℕ :=
  sorry

/-- Theorem stating that the men's T-shirt sale interval is 40 minutes -/
theorem mens_tshirt_interval (shop : TShirtShop) 
  (h1 : shop.womens_interval = 30)
  (h2 : shop.womens_price = 18)
  (h3 : shop.mens_price = 15)
  (h4 : shop.daily_hours = 12)
  (h5 : shop.weekly_days = 7)
  (h6 : shop.weekly_revenue = 4914) :
  mens_interval shop = 40 := by
    sorry

end mens_tshirt_interval_l2304_230472


namespace middle_number_proof_l2304_230466

theorem middle_number_proof (A B C : ℝ) (h1 : A < B) (h2 : B < C) 
  (h3 : B - C = A - B) (h4 : A * B = 85) (h5 : B * C = 115) : B = 10 := by
  sorry

end middle_number_proof_l2304_230466


namespace shortest_handspan_l2304_230406

def sangwon_handspan : ℝ := 19 + 0.8
def doyoon_handspan : ℝ := 18.9
def changhyeok_handspan : ℝ := 19.3

theorem shortest_handspan :
  doyoon_handspan < sangwon_handspan ∧ doyoon_handspan < changhyeok_handspan :=
by
  sorry

end shortest_handspan_l2304_230406


namespace sam_investment_result_l2304_230458

-- Define the compound interest function
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

-- Define the problem parameters
def initial_investment : ℝ := 10000
def first_rate : ℝ := 0.20
def first_time : ℕ := 3
def multiplier : ℝ := 3
def second_rate : ℝ := 0.15
def second_time : ℕ := 1

-- Theorem statement
theorem sam_investment_result :
  let first_phase := compound_interest initial_investment first_rate first_time
  let second_phase := compound_interest (first_phase * multiplier) second_rate second_time
  second_phase = 59616 := by sorry

end sam_investment_result_l2304_230458


namespace inscribed_circle_angle_theorem_l2304_230455

/-- A triangle with an inscribed circle --/
structure InscribedCircleTriangle where
  /-- The angle at the tangent point on side BC --/
  angle_bc : ℝ
  /-- The angle at the tangent point on side CA --/
  angle_ca : ℝ
  /-- The angle at the tangent point on side AB --/
  angle_ab : ℝ
  /-- The sum of angles at tangent points is 360° --/
  sum_angles : angle_bc + angle_ca + angle_ab = 360

/-- Theorem: If the angles at tangent points are 120°, 130°, and θ°, then θ = 110° --/
theorem inscribed_circle_angle_theorem (t : InscribedCircleTriangle) 
    (h1 : t.angle_bc = 120) (h2 : t.angle_ca = 130) : t.angle_ab = 110 := by
  sorry

end inscribed_circle_angle_theorem_l2304_230455


namespace new_city_buildings_count_l2304_230483

/-- Calculates the total number of buildings for the new city project --/
def new_city_buildings (pittsburgh_stores : ℕ) (pittsburgh_hospitals : ℕ) (pittsburgh_schools : ℕ) (pittsburgh_police : ℕ) : ℕ :=
  (pittsburgh_stores / 2) + (pittsburgh_hospitals * 2) + (pittsburgh_schools - 50) + (pittsburgh_police + 5)

/-- Theorem stating that the total number of buildings for the new city is 2175 --/
theorem new_city_buildings_count : 
  new_city_buildings 2000 500 200 20 = 2175 := by
  sorry

end new_city_buildings_count_l2304_230483


namespace correct_answer_calculation_l2304_230432

theorem correct_answer_calculation (incorrect_answer : ℝ) (h : incorrect_answer = 115.15) :
  let original_value := incorrect_answer / 7
  let correct_answer := original_value / 7
  correct_answer = 2.35 := by
sorry

end correct_answer_calculation_l2304_230432


namespace temperature_decrease_l2304_230477

/-- The temperature that is 6°C lower than -3°C is -9°C. -/
theorem temperature_decrease : ((-3 : ℤ) - 6) = -9 := by
  sorry

end temperature_decrease_l2304_230477


namespace inequality_proof_l2304_230457

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b^5 + b * c^5 + c * a^5 ≥ a * b * c * (a^2 * b + b^2 * c + c^2 * a) := by
  sorry

end inequality_proof_l2304_230457


namespace students_play_both_football_and_cricket_l2304_230473

/-- Represents the number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

/-- Theorem stating that given the conditions, 140 students play both football and cricket -/
theorem students_play_both_football_and_cricket :
  students_play_both 410 325 175 50 = 140 := by
  sorry

end students_play_both_football_and_cricket_l2304_230473


namespace smallest_a_with_single_digit_sum_l2304_230413

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is single-digit -/
def is_single_digit (n : ℕ) : Prop := n < 10

/-- The property we want to prove -/
def has_single_digit_sum (a : ℕ) : Prop :=
  is_single_digit (sum_of_digits (10^a - 74))

theorem smallest_a_with_single_digit_sum :
  (∀ k < 2, ¬ has_single_digit_sum k) ∧ has_single_digit_sum 2 := by sorry

end smallest_a_with_single_digit_sum_l2304_230413


namespace garage_spokes_count_l2304_230499

/-- Represents a bicycle with two wheels -/
structure Bicycle where
  front_spokes : ℕ
  back_spokes : ℕ

/-- Represents a tricycle with three wheels -/
structure Tricycle where
  front_spokes : ℕ
  middle_spokes : ℕ
  back_spokes : ℕ

/-- The total number of spokes in all bicycles and the tricycle -/
def total_spokes (bikes : List Bicycle) (trike : Tricycle) : ℕ :=
  (bikes.map (fun b => b.front_spokes + b.back_spokes)).sum +
  (trike.front_spokes + trike.middle_spokes + trike.back_spokes)

theorem garage_spokes_count :
  let bikes : List Bicycle := [
    { front_spokes := 16, back_spokes := 18 },
    { front_spokes := 20, back_spokes := 22 },
    { front_spokes := 24, back_spokes := 26 },
    { front_spokes := 28, back_spokes := 30 }
  ]
  let trike : Tricycle := { front_spokes := 32, middle_spokes := 34, back_spokes := 36 }
  total_spokes bikes trike = 286 := by
  sorry


end garage_spokes_count_l2304_230499


namespace nancy_crayon_packs_l2304_230465

theorem nancy_crayon_packs (total_crayons : ℕ) (crayons_per_pack : ℕ) 
  (h1 : total_crayons = 615) 
  (h2 : crayons_per_pack = 15) : 
  total_crayons / crayons_per_pack = 41 := by
  sorry

end nancy_crayon_packs_l2304_230465


namespace inequality_proof_l2304_230467

theorem inequality_proof (a b c : ℝ) : a^2 + 4*b^2 + 9*c^2 ≥ 2*a*b + 3*a*c + 6*b*c := by
  sorry

end inequality_proof_l2304_230467


namespace true_discount_example_l2304_230424

/-- Given a banker's discount and sum due, calculate the true discount -/
def true_discount (bankers_discount : ℚ) (sum_due : ℚ) : ℚ :=
  bankers_discount / (1 + bankers_discount / sum_due)

/-- Theorem stating that for a banker's discount of 18 and sum due of 90, the true discount is 15 -/
theorem true_discount_example : true_discount 18 90 = 15 := by
  sorry

end true_discount_example_l2304_230424


namespace octal_7421_to_decimal_l2304_230426

def octal_to_decimal (octal : ℕ) : ℕ :=
  let digits := [7, 4, 2, 1]
  (List.zipWith (λ (d : ℕ) (p : ℕ) => d * (8 ^ p)) digits (List.range 4)).sum

theorem octal_7421_to_decimal :
  octal_to_decimal 7421 = 1937 := by
  sorry

end octal_7421_to_decimal_l2304_230426


namespace polynomial_factorization_l2304_230453

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 4*x + 4 - 81*x^4 = (-9*x^2 + x + 2) * (9*x^2 + x + 2) := by
  sorry

#check polynomial_factorization

end polynomial_factorization_l2304_230453


namespace triangle_ABC_coordinates_l2304_230498

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (a b c : Point) : ℝ := sorry

/-- Checks if a point is on a coordinate axis -/
def isOnAxis (p : Point) : Prop :=
  p.x = 0 ∨ p.y = 0

theorem triangle_ABC_coordinates :
  let a : Point := ⟨2, 0⟩
  let b : Point := ⟨0, 3⟩
  ∀ c : Point,
    triangleArea a b c = 6 ∧ isOnAxis c →
    c = ⟨0, 9⟩ ∨ c = ⟨0, -3⟩ ∨ c = ⟨-2, 0⟩ ∨ c = ⟨6, 0⟩ :=
by sorry

end triangle_ABC_coordinates_l2304_230498


namespace symmetry_about_origin_l2304_230479

/-- A point on the graph of y = 3^x -/
structure PointOn3x where
  x : ℝ
  y : ℝ
  h : y = 3^x

/-- A point on the graph of y = -3^(-x) -/
structure PointOnNeg3NegX where
  x : ℝ
  y : ℝ
  h : y = -3^(-x)

/-- The condition given in the problem -/
axiom symmetry_condition {p : PointOn3x} :
  ∃ (q : PointOnNeg3NegX), q.x = -p.x ∧ q.y = -p.y

/-- The theorem to be proved -/
theorem symmetry_about_origin :
  ∀ (p : PointOn3x), ∃ (q : PointOnNeg3NegX), q.x = -p.x ∧ q.y = -p.y :=
sorry

end symmetry_about_origin_l2304_230479


namespace no_universal_divisibility_l2304_230442

/-- Represents a nonzero digit (1-9) -/
def NonzeroDigit := {d : Nat // d ≥ 1 ∧ d ≤ 9}

/-- Concatenates three numbers to form a new number -/
def concat3 (a : NonzeroDigit) (n : Nat) (b : NonzeroDigit) : Nat :=
  100 * a.val + 10 * n + b.val

/-- Concatenates two numbers to form a new number -/
def concat2 (a b : NonzeroDigit) : Nat :=
  10 * a.val + b.val

/-- Statement: There does not exist a natural number n such that
    for all nonzero digits a and b, concat3 a n b is divisible by concat2 a b -/
theorem no_universal_divisibility :
  ¬ ∃ n : Nat, ∀ (a b : NonzeroDigit), (concat3 a n b) % (concat2 a b) = 0 := by
  sorry

end no_universal_divisibility_l2304_230442


namespace chocolate_division_l2304_230456

theorem chocolate_division (total : ℚ) (piles : ℕ) (keep_fraction : ℚ) :
  total = 72 / 7 →
  piles = 6 →
  keep_fraction = 1 / 3 →
  (total / piles) * (1 - keep_fraction) = 8 / 7 := by
  sorry

end chocolate_division_l2304_230456


namespace balls_triangle_to_square_l2304_230471

theorem balls_triangle_to_square (n : ℕ) (h1 : n * (n + 1) / 2 = 1176) :
  let square_side := n - 8
  square_side * square_side - n * (n + 1) / 2 = 424 := by
  sorry

end balls_triangle_to_square_l2304_230471


namespace fish_count_l2304_230433

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 11

/-- The total number of fish Lilly and Rosy have -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 21 := by
  sorry

end fish_count_l2304_230433


namespace complement_of_A_in_U_l2304_230489

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3}

-- Define set A
def A : Finset Nat := {1, 3}

-- Theorem statement
theorem complement_of_A_in_U : 
  (U \ A) = {0, 2} := by sorry

end complement_of_A_in_U_l2304_230489


namespace boat_distance_against_stream_l2304_230481

/-- Calculates the distance a boat travels against a stream in one hour, given its speed in still water and its distance traveled along the stream in one hour. -/
def distance_against_stream (speed_still_water : ℝ) (distance_along_stream : ℝ) : ℝ :=
  speed_still_water - (distance_along_stream - speed_still_water)

/-- Theorem stating that a boat with a speed of 8 km/hr in still water, which travels 11 km along a stream in one hour, will travel 5 km against the stream in one hour. -/
theorem boat_distance_against_stream :
  distance_against_stream 8 11 = 5 := by
  sorry

#eval distance_against_stream 8 11

end boat_distance_against_stream_l2304_230481


namespace garden_ratio_maintenance_l2304_230443

/-- Represents a garden with tulips and daisies -/
structure Garden where
  tulips : ℕ
  daisies : ℕ

/-- Calculates the number of tulips needed to maintain a 3:7 ratio with the given number of daisies -/
def tulipsForRatio (daisies : ℕ) : ℕ :=
  (3 * daisies + 6) / 7

theorem garden_ratio_maintenance (initial : Garden) (added_daisies : ℕ) :
  initial.daisies = 35 →
  added_daisies = 30 →
  (3 : ℚ) / 7 = initial.tulips / initial.daisies →
  tulipsForRatio (initial.daisies + added_daisies) = 28 := by
  sorry

end garden_ratio_maintenance_l2304_230443


namespace greater_a_than_c_l2304_230460

theorem greater_a_than_c (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : (a - b) * (b - c) * (c - a) > 0) : 
  a > c := by
  sorry

end greater_a_than_c_l2304_230460


namespace oliver_vowel_learning_time_l2304_230495

/-- The number of days Oliver takes to learn one alphabet -/
def days_per_alphabet : ℕ := 5

/-- The number of vowels in the English alphabet -/
def number_of_vowels : ℕ := 5

/-- The total number of days Oliver needs to finish learning all vowels -/
def total_days : ℕ := days_per_alphabet * number_of_vowels

theorem oliver_vowel_learning_time : total_days = 25 := by
  sorry

end oliver_vowel_learning_time_l2304_230495


namespace geometric_mean_sqrt3_plus_minus_one_l2304_230419

theorem geometric_mean_sqrt3_plus_minus_one : 
  ∃ (x : ℝ), x^2 = (Real.sqrt 3 - 1) * (Real.sqrt 3 + 1) ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) :=
by sorry

end geometric_mean_sqrt3_plus_minus_one_l2304_230419


namespace sector_radius_range_l2304_230431

theorem sector_radius_range (a : ℝ) (m : ℝ) (h1 : a > 0) (h2 : 0 < m) (h3 : m < 360) :
  ∃ R : ℝ, a / (2 * (1 + π)) < R ∧ R < a / 2 := by
  sorry

end sector_radius_range_l2304_230431


namespace total_cookies_count_l2304_230411

/-- Given 286 bags of cookies with 452 cookies in each bag, 
    prove that the total number of cookies is 129,272. -/
theorem total_cookies_count (bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : bags = 286) (h2 : cookies_per_bag = 452) : 
  bags * cookies_per_bag = 129272 := by
  sorry

end total_cookies_count_l2304_230411


namespace square_approximation_l2304_230493

theorem square_approximation (x : ℝ) (h : x ≥ 1/2) :
  ∃ n : ℤ, |x - (n : ℝ)^2| ≤ Real.sqrt (x - 1/4) := by
  sorry

end square_approximation_l2304_230493


namespace ken_steak_change_l2304_230417

/-- Calculates the change Ken will receive when buying steak -/
def calculate_change (price_per_pound : ℕ) (pounds_bought : ℕ) (payment : ℕ) : ℕ :=
  payment - (price_per_pound * pounds_bought)

/-- Proves that Ken will receive $6 in change -/
theorem ken_steak_change :
  let price_per_pound : ℕ := 7
  let pounds_bought : ℕ := 2
  let payment : ℕ := 20
  calculate_change price_per_pound pounds_bought payment = 6 := by
sorry

end ken_steak_change_l2304_230417


namespace expression_evaluation_l2304_230405

theorem expression_evaluation : 
  (2 + 1/4)^(1/2) - 0.3^0 - 16^(-3/4) = 3/8 := by sorry

end expression_evaluation_l2304_230405


namespace complex_multiplication_l2304_230480

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (1 - i)^2 * (1 + i) = 2 - 2*i := by
  sorry

end complex_multiplication_l2304_230480


namespace homothetic_image_containment_l2304_230485

-- Define a convex polygon
def ConvexPolygon (P : Set (Point)) : Prop := sorry

-- Define a homothetic transformation
def HomotheticTransformation (center : Point) (k : ℝ) (P : Set Point) : Set Point := sorry

-- Define that a set is contained within another set
def IsContainedIn (A B : Set Point) : Prop := sorry

-- The theorem statement
theorem homothetic_image_containment 
  (P : Set Point) (h : ConvexPolygon P) :
  ∃ (center : Point), 
    IsContainedIn (HomotheticTransformation center (1/2) P) P := by
  sorry

end homothetic_image_containment_l2304_230485


namespace absolute_difference_60th_terms_l2304_230401

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem absolute_difference_60th_terms : 
  let C := arithmetic_sequence 25 15
  let D := arithmetic_sequence 40 (-15)
  |C 60 - D 60| = 1755 := by
sorry

end absolute_difference_60th_terms_l2304_230401


namespace gcd_1785_840_l2304_230423

theorem gcd_1785_840 : Nat.gcd 1785 840 = 105 := by
  sorry

end gcd_1785_840_l2304_230423


namespace john_twice_frank_age_l2304_230446

/-- Given that Frank is 15 years younger than John and Frank will be 16 in 4 years,
    prove that John will be twice as old as Frank in 3 years. -/
theorem john_twice_frank_age (frank_age john_age x : ℕ) : 
  john_age = frank_age + 15 →
  frank_age + 4 = 16 →
  john_age + x = 2 * (frank_age + x) →
  x = 3 := by sorry

end john_twice_frank_age_l2304_230446


namespace union_complement_equal_set_l2304_230476

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_equal_set : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_equal_set_l2304_230476


namespace equal_distribution_of_drawings_l2304_230438

/-- Given 54 animal drawings distributed equally among 6 neighbors, prove that each neighbor receives 9 drawings. -/
theorem equal_distribution_of_drawings (total_drawings : ℕ) (num_neighbors : ℕ) (drawings_per_neighbor : ℕ) : 
  total_drawings = 54 → 
  num_neighbors = 6 → 
  total_drawings = num_neighbors * drawings_per_neighbor →
  drawings_per_neighbor = 9 :=
by
  sorry

end equal_distribution_of_drawings_l2304_230438


namespace trapezoid_xy_length_l2304_230461

/-- Represents a trapezoid WXYZ with specific properties -/
structure Trapezoid where
  -- Points W, X, Y, Z
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  -- WX is parallel to ZY
  parallel_WX_ZY : (X.1 - W.1) * (Y.2 - Z.2) = (X.2 - W.2) * (Y.1 - Z.1)
  -- WY is perpendicular to ZY
  perpendicular_WY_ZY : (Y.1 - W.1) * (Y.1 - Z.1) + (Y.2 - W.2) * (Y.2 - Z.2) = 0
  -- YZ = 20
  yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 20
  -- tan Z = 2
  tan_Z : (Y.2 - Z.2) / (Y.1 - Z.1) = 2
  -- tan X = 2.5
  tan_X : (Y.2 - X.2) / (X.1 - Y.1) = 2.5

/-- The length of XY in the trapezoid is 4√116 -/
theorem trapezoid_xy_length (t : Trapezoid) : 
  Real.sqrt ((t.X.1 - t.Y.1)^2 + (t.X.2 - t.Y.2)^2) = 4 * Real.sqrt 116 := by
  sorry


end trapezoid_xy_length_l2304_230461


namespace grandpa_lou_movie_time_l2304_230441

theorem grandpa_lou_movie_time :
  ∀ (tuesday_movies : ℕ),
    (tuesday_movies + 2 * tuesday_movies ≤ 9) →
    (tuesday_movies * 90 = 270) :=
by
  sorry

end grandpa_lou_movie_time_l2304_230441


namespace keith_missed_games_l2304_230463

theorem keith_missed_games (total_games : ℕ) (attended_games : ℕ) 
  (h1 : total_games = 8)
  (h2 : attended_games = 4) :
  total_games - attended_games = 4 := by
  sorry

end keith_missed_games_l2304_230463


namespace square_sum_xy_l2304_230422

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 30) 
  (h2 : y * (x + y) = 60) : 
  (x + y)^2 = 90 := by
  sorry

end square_sum_xy_l2304_230422


namespace factorial_sum_division_l2304_230449

theorem factorial_sum_division (n : ℕ) : (Nat.factorial 8 + Nat.factorial 9) / Nat.factorial 6 = 560 := by
  sorry

end factorial_sum_division_l2304_230449


namespace fraction_ordering_l2304_230440

theorem fraction_ordering : (8 : ℚ) / 25 < 6 / 17 ∧ 6 / 17 < 11 / 29 := by
  sorry

end fraction_ordering_l2304_230440


namespace sine_cosine_sum_equals_root_two_over_two_l2304_230470

theorem sine_cosine_sum_equals_root_two_over_two :
  Real.sin (30 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (30 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end sine_cosine_sum_equals_root_two_over_two_l2304_230470


namespace elises_initial_money_l2304_230409

/-- Proves that Elise's initial amount of money was $8 --/
theorem elises_initial_money :
  ∀ (initial savings comic_cost puzzle_cost final : ℕ),
  savings = 13 →
  comic_cost = 2 →
  puzzle_cost = 18 →
  final = 1 →
  initial + savings - comic_cost - puzzle_cost = final →
  initial = 8 := by
sorry

end elises_initial_money_l2304_230409


namespace square_equation_solution_l2304_230475

theorem square_equation_solution :
  ∃! x : ℚ, (2015 + x)^2 = x^2 ∧ x = -2015/2 := by sorry

end square_equation_solution_l2304_230475


namespace volume_removed_tetrahedra_l2304_230400

/-- The volume of tetrahedra removed from a cube when slicing corners to form octagonal faces -/
theorem volume_removed_tetrahedra (cube_edge : ℝ) (h : cube_edge = 2) :
  let octagon_side := 2 * (Real.sqrt 2 - 1)
  let tetrahedron_height := 2 / Real.sqrt 2
  let base_area := 2 * (3 - 2 * Real.sqrt 2)
  let single_tetrahedron_volume := (1 / 3) * base_area * tetrahedron_height
  8 * single_tetrahedron_volume = (32 * (3 - 2 * Real.sqrt 2)) / 3 :=
by sorry

end volume_removed_tetrahedra_l2304_230400


namespace coin_machine_possible_amount_l2304_230421

theorem coin_machine_possible_amount :
  ∃ (m n p : ℕ), 298 = 5 + 25 * m + 2 * n + 29 * p :=
sorry

end coin_machine_possible_amount_l2304_230421


namespace constant_term_binomial_expansion_l2304_230434

theorem constant_term_binomial_expansion (x : ℝ) : 
  let binomial := (x - 1 / (2 * Real.sqrt x)) ^ 9
  ∃ c : ℝ, c = 21/16 ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |binomial - c| < ε) :=
by
  sorry

end constant_term_binomial_expansion_l2304_230434


namespace rectangle_breadth_l2304_230497

theorem rectangle_breadth (area : ℝ) (length_ratio : ℝ) (breadth : ℝ) : 
  area = 460 →
  length_ratio = 1.15 →
  area = (length_ratio * breadth) * breadth →
  breadth = 20 := by
sorry

end rectangle_breadth_l2304_230497


namespace graph_intersection_symmetry_l2304_230444

/-- Given real numbers a, b, c, and d, if the graphs of 
    y = 2a + 1/(x-b) and y = 2c + 1/(x-d) have exactly one common point, 
    then the graphs of y = 2b + 1/(x-a) and y = 2d + 1/(x-c) 
    also have exactly one common point. -/
theorem graph_intersection_symmetry (a b c d : ℝ) :
  (∃! x : ℝ, 2*a + 1/(x-b) = 2*c + 1/(x-d)) →
  (∃! x : ℝ, 2*b + 1/(x-a) = 2*d + 1/(x-c)) :=
by sorry

end graph_intersection_symmetry_l2304_230444


namespace largest_divisor_five_consecutive_integers_l2304_230469

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k > 24 ∧ ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ m : ℤ, m ≤ 24 → (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end largest_divisor_five_consecutive_integers_l2304_230469


namespace phi_value_l2304_230430

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ)

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := f x φ + (deriv (f · φ)) x

theorem phi_value (φ : ℝ) 
  (h1 : -π < φ ∧ φ < 0) 
  (h2 : ∀ x, g x φ = g (-x) φ) : 
  φ = -π/3 := by
sorry

end phi_value_l2304_230430


namespace quentavious_gum_pieces_l2304_230462

/-- Given the initial number of nickels, the number of nickels left, and the number of gum pieces per nickel,
    calculate the total number of gum pieces received. -/
def gumPiecesReceived (initialNickels : ℕ) (nickelsLeft : ℕ) (gumPiecesPerNickel : ℕ) : ℕ :=
  (initialNickels - nickelsLeft) * gumPiecesPerNickel

/-- Theorem: The number of gum pieces Quentavious received is 6, given the problem conditions. -/
theorem quentavious_gum_pieces :
  gumPiecesReceived 5 2 2 = 6 := by
  sorry

end quentavious_gum_pieces_l2304_230462


namespace inequality_solution_set_l2304_230414

theorem inequality_solution_set (x : ℝ) :
  (x - 3) / (x^2 - 2*x + 11) ≥ 0 ↔ x ≥ 3 := by
sorry

end inequality_solution_set_l2304_230414


namespace infinitely_many_squares_2012_2013_divisibility_condition_l2304_230491

-- Part (a)
theorem infinitely_many_squares_2012_2013 :
  ∀ k : ℕ, ∃ t > k, ∃ a b : ℕ,
    2012 * t + 1 = a^2 ∧ 2013 * t + 1 = b^2 :=
sorry

-- Part (b)
theorem divisibility_condition (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∃ x y : ℕ, m * n + 1 = x^2 ∧ m * n + n + 1 = y^2) →
  8 * (2 * m + 1) ∣ n :=
sorry

end infinitely_many_squares_2012_2013_divisibility_condition_l2304_230491


namespace chairs_per_table_l2304_230468

theorem chairs_per_table (indoor_tables outdoor_tables total_chairs : ℕ) 
  (h1 : indoor_tables = 8)
  (h2 : outdoor_tables = 12)
  (h3 : total_chairs = 60) :
  ∃ (chairs_per_table : ℕ), 
    chairs_per_table * (indoor_tables + outdoor_tables) = total_chairs ∧ 
    chairs_per_table = 3 := by
  sorry

end chairs_per_table_l2304_230468


namespace no_x_squared_term_l2304_230474

theorem no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + 2) * (2*x - 4) = 2*x^3 + (2*a - 4)*x^2 + (4 - 4*a)*x - 8) →
  (2*a - 4 = 0 ↔ a = 2) :=
by sorry

end no_x_squared_term_l2304_230474


namespace cistern_filling_time_l2304_230428

theorem cistern_filling_time (T : ℝ) : 
  T > 0 →  -- T must be positive
  (1 / 4 : ℝ) - (1 / T) = (3 / 28 : ℝ) → 
  T = 7 :=
by sorry

end cistern_filling_time_l2304_230428


namespace sum_square_bound_l2304_230478

/-- The sum of integers from 1 to n -/
def sum_to (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate for a natural number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem sum_square_bound :
  ∀ K : ℕ, K > 0 →
    (is_perfect_square (sum_to K) ∧
     ∃ N : ℕ, sum_to K = N * N ∧ N + K < 120) ↔
    (K = 1 ∨ K = 8 ∨ K = 49) :=
sorry

end sum_square_bound_l2304_230478


namespace cow_value_increase_l2304_230407

/-- Calculates the increase in value of a cow after weight gain -/
theorem cow_value_increase (initial_weight : ℝ) (weight_factor : ℝ) (price_per_pound : ℝ)
  (h1 : initial_weight = 400)
  (h2 : weight_factor = 1.5)
  (h3 : price_per_pound = 3) :
  (initial_weight * weight_factor - initial_weight) * price_per_pound = 600 := by
  sorry

#check cow_value_increase

end cow_value_increase_l2304_230407


namespace inequalities_hold_l2304_230454

theorem inequalities_hold (a b : ℝ) (h : a ≠ b) : 
  (a^2 - 4*a + 5 > 0) ∧ (a^2 + b^2 ≥ 2*(a - b - 1)) := by
  sorry

end inequalities_hold_l2304_230454


namespace modular_congruence_solution_l2304_230418

theorem modular_congruence_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 27 ∧ n ≡ -3456 [ZMOD 28] ∧ n = 12 := by
  sorry

end modular_congruence_solution_l2304_230418


namespace sum_seven_terms_l2304_230437

/-- An arithmetic sequence with specific terms. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  second_term : a 2 = 5 / 3
  sixth_term : a 6 = -7 / 3

/-- The sum of the first n terms of an arithmetic sequence. -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The theorem stating that the sum of the first 7 terms is -7/3. -/
theorem sum_seven_terms (seq : ArithmeticSequence) : sum_n_terms seq 7 = -7 / 3 := by
  sorry


end sum_seven_terms_l2304_230437


namespace quadratic_coefficient_l2304_230487

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x : ℝ, a * (x - 1)^2 + 3 = a * x^2 + b * x + c) →
  a * 0^2 + b * 0 + c = 1 →
  a = -2 := by
  sorry

end quadratic_coefficient_l2304_230487


namespace solution_set_inequality_l2304_230415

theorem solution_set_inequality (x : ℝ) :
  x * (x - 1) < 0 ↔ 0 < x ∧ x < 1 := by sorry

end solution_set_inequality_l2304_230415


namespace rectangle_area_perimeter_sum_l2304_230494

theorem rectangle_area_perimeter_sum (a b : ℕ+) : 
  let A := (a : ℝ) * (b : ℝ)
  let P := 2 * (a : ℝ) + 2 * (b : ℝ) + 2
  A + P ≠ 114 :=
by sorry

end rectangle_area_perimeter_sum_l2304_230494


namespace expression_evaluation_l2304_230488

theorem expression_evaluation : 
  Real.sqrt 5 * 5^(1/2 : ℝ) + 18 / 3 * 2 - 9^(3/2 : ℝ) + 10 = 0 := by
  sorry

end expression_evaluation_l2304_230488


namespace coplanar_condition_l2304_230448

open Real

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the origin and points
variable (O A B C D : V)

-- Define the condition for coplanarity
def are_coplanar (A B C D : V) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧ 
  a • (A - O) + b • (B - O) + c • (C - O) + d • (D - O) = 0

-- State the theorem
theorem coplanar_condition (k' : ℝ) :
  (4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k' • (D - O) = 0) →
  (are_coplanar O A B C D ↔ k' = -7) := by
  sorry

end coplanar_condition_l2304_230448


namespace min_value_x_plus_2y_l2304_230402

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : 1 / (2 * x + y) + 1 / (y + 1) = 1) : 
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 1 / (2 * x' + y') + 1 / (y' + 1) = 1 → x + 2 * y ≤ x' + 2 * y') ∧ 
  x + 2 * y = Real.sqrt 3 + 1 / 2 := by
sorry

end min_value_x_plus_2y_l2304_230402


namespace both_questions_correct_percentage_l2304_230412

theorem both_questions_correct_percentage
  (p_first : ℝ)
  (p_second : ℝ)
  (p_neither : ℝ)
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.65)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.60 :=
by
  sorry

end both_questions_correct_percentage_l2304_230412


namespace half_day_division_ways_l2304_230416

/-- The number of ways to express 43200 as a product of two positive integers -/
def num_factor_pairs : ℕ := 72

/-- Half a day in seconds -/
def half_day_seconds : ℕ := 43200

theorem half_day_division_ways :
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = half_day_seconds) (Finset.product (Finset.range (half_day_seconds + 1)) (Finset.range (half_day_seconds + 1)))).card = num_factor_pairs :=
sorry

end half_day_division_ways_l2304_230416


namespace train_arrangement_count_l2304_230420

/-- Represents the number of trains -/
def total_trains : ℕ := 8

/-- Represents the number of trains in each group -/
def trains_per_group : ℕ := 4

/-- Calculates the number of ways to arrange the trains according to the given conditions -/
def train_arrangements : ℕ := sorry

/-- Theorem stating that the number of train arrangements is 720 -/
theorem train_arrangement_count : train_arrangements = 720 := by sorry

end train_arrangement_count_l2304_230420


namespace harrison_extra_pages_l2304_230496

def minimum_pages : ℕ := 25
def sam_pages : ℕ := 100

def pam_pages (sam : ℕ) : ℕ := sam / 2

def harrison_pages (pam : ℕ) : ℕ := pam - 15

theorem harrison_extra_pages :
  harrison_pages (pam_pages sam_pages) - minimum_pages = 10 :=
by sorry

end harrison_extra_pages_l2304_230496


namespace expanded_garden_perimeter_l2304_230439

/-- Given a square garden with an area of 49 square meters, if each side is expanded by 4 meters
    to form a new square garden, the perimeter of the new garden is 44 meters. -/
theorem expanded_garden_perimeter : ∀ (original_side : ℝ),
  original_side^2 = 49 →
  (4 * (original_side + 4) = 44) :=
by
  sorry

end expanded_garden_perimeter_l2304_230439


namespace smallest_positive_integer_3003m_55555n_l2304_230435

theorem smallest_positive_integer_3003m_55555n :
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 3003 * m + 55555 * n) ∧
  ∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 3003 * x + 55555 * y) → k ≤ j :=
by sorry

end smallest_positive_integer_3003m_55555n_l2304_230435


namespace visible_sum_range_l2304_230436

/-- Represents a die with 6 faces -/
structure Die :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)
  (face_range : ∀ i : Fin 6, 1 ≤ faces i ∧ faces i ≤ 6)

/-- Represents the larger 3x3x3 cube made of 27 dice -/
def LargeCube := Fin 3 → Fin 3 → Fin 3 → Die

/-- Calculates the sum of visible face values on the larger cube -/
def visible_sum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the range of possible sums of visible face values -/
theorem visible_sum_range (cube : LargeCube) :
  90 ≤ visible_sum cube ∧ visible_sum cube ≤ 288 :=
sorry

end visible_sum_range_l2304_230436


namespace not_proportional_l2304_230429

-- Define the properties of direct and inverse proportionality
def is_directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x * x = k

-- Define the function representing y = 3x + 2
def f (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem not_proportional :
  ¬(is_directly_proportional f) ∧ ¬(is_inversely_proportional f) :=
by sorry

end not_proportional_l2304_230429


namespace min_sum_of_ten_numbers_l2304_230451

theorem min_sum_of_ten_numbers (S : Finset ℕ) : 
  S.card = 10 → 
  (∀ T ⊆ S, T.card = 5 → (T.prod id) % 2 = 0) → 
  (S.sum id) % 2 = 1 → 
  ∃ min_sum : ℕ, 
    (S.sum id = min_sum) ∧ 
    (∀ S' : Finset ℕ, S'.card = 10 → 
      (∀ T' ⊆ S', T'.card = 5 → (T'.prod id) % 2 = 0) → 
      (S'.sum id) % 2 = 1 → 
      S'.sum id ≥ min_sum) ∧
    min_sum = 51 :=
sorry

end min_sum_of_ten_numbers_l2304_230451


namespace race_probability_l2304_230486

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℝ) : 
  total_cars = 18 → 
  prob_Y = 1/12 → 
  prob_Z = 1/6 → 
  prob_XYZ = 0.375 → 
  ∃ prob_X : ℝ, 
    prob_X + prob_Y + prob_Z = prob_XYZ ∧ 
    prob_X = 0.125 := by
  sorry

end race_probability_l2304_230486


namespace inequality_theorem_l2304_230404

theorem inequality_theorem (a b : ℝ) : 
  |2*a - 2| < |a - 4| → |2*b - 2| < |b - 4| → 2*|a + b| < |4 + a*b| := by
sorry

end inequality_theorem_l2304_230404


namespace work_completion_time_l2304_230482

/-- Given that two workers A and B can complete a task together in a certain time,
    and B can complete the task alone in a known time,
    this theorem proves how long it takes A to complete the task alone. -/
theorem work_completion_time
  (joint_time : ℝ)
  (b_time : ℝ)
  (h_joint : joint_time = 8.571428571428571)
  (h_b : b_time = 20)
  : ∃ (a_time : ℝ), a_time = 15 ∧ 1 / a_time + 1 / b_time = 1 / joint_time :=
sorry

end work_completion_time_l2304_230482


namespace cards_per_page_l2304_230403

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 3) 
  (h2 : old_cards = 9) 
  (h3 : pages = 4) : 
  (new_cards + old_cards) / pages = 3 := by
  sorry

end cards_per_page_l2304_230403


namespace power_one_sixth_equals_one_l2304_230410

def is_greatest_power_of_two_factor (a : ℕ) : Prop :=
  2^a ∣ 180 ∧ ∀ k > a, ¬(2^k ∣ 180)

def is_greatest_power_of_three_factor (b : ℕ) : Prop :=
  3^b ∣ 180 ∧ ∀ k > b, ¬(3^k ∣ 180)

theorem power_one_sixth_equals_one (a b : ℕ) 
  (h1 : is_greatest_power_of_two_factor a) 
  (h2 : is_greatest_power_of_three_factor b) : 
  (1/6 : ℚ)^(b - a) = 1 := by
  sorry

end power_one_sixth_equals_one_l2304_230410


namespace fraction_five_thirteenths_digit_sum_l2304_230447

theorem fraction_five_thirteenths_digit_sum : 
  ∃ (a b c d : ℕ), 
    (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧ 
    (5 : ℚ) / 13 = (a * 1000 + b * 100 + c * 10 + d) / 9999 ∧ 
    a + b + c + d = 20 := by
  sorry

end fraction_five_thirteenths_digit_sum_l2304_230447


namespace parabola_vertex_condition_l2304_230492

/-- A parabola with equation y = x^2 + 2x + a -/
structure Parabola where
  a : ℝ

/-- The vertex of a parabola y = x^2 + 2x + a is below the x-axis -/
def vertex_below_x_axis (p : Parabola) : Prop :=
  let x := -1  -- x-coordinate of the vertex
  let y := x^2 + 2*x + p.a  -- y-coordinate of the vertex
  y < 0

/-- If the vertex of the parabola y = x^2 + 2x + a is below the x-axis, then a < 1 -/
theorem parabola_vertex_condition (p : Parabola) : vertex_below_x_axis p → p.a < 1 := by
  sorry

end parabola_vertex_condition_l2304_230492


namespace identify_brothers_l2304_230464

-- Define the brothers
inductive Brother
| trulya
| tralya

-- Define a function to represent whether a brother tells the truth
def tellsTruth : Brother → Prop
| Brother.trulya => true
| Brother.tralya => false

-- Define the statements made by the brothers
def firstBrotherStatement (first second : Brother) : Prop :=
  first = Brother.trulya

def secondBrotherStatement (first second : Brother) : Prop :=
  second = Brother.tralya

def cardSuitStatement : Prop := false  -- Cards are not of the same suit

-- The main theorem
theorem identify_brothers :
  ∃ (first second : Brother),
    first ≠ second ∧
    (tellsTruth first → firstBrotherStatement first second) ∧
    (tellsTruth second → secondBrotherStatement first second) ∧
    (tellsTruth first → cardSuitStatement) ∧
    first = Brother.tralya ∧
    second = Brother.trulya :=
  sorry

end identify_brothers_l2304_230464


namespace smallest_n_for_factorization_l2304_230425

theorem smallest_n_for_factorization : 
  ∀ n : ℤ, 
  (∃ A B : ℤ, ∀ x : ℝ, 3 * x^2 + n * x + 72 = (3 * x + A) * (x + B)) → 
  n ≥ 35 :=
by sorry

end smallest_n_for_factorization_l2304_230425
