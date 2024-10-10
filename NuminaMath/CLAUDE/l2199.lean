import Mathlib

namespace range_of_m_l2199_219974

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = (1/2)^x}

-- Define the set N
def N (m : ℝ) : Set ℝ := {y | ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ y = (1/(m-1) + 1)*(x-1) + (|m|-1)*(x-2)}

-- Theorem statement
theorem range_of_m : 
  ∀ m : ℝ, (∀ y ∈ N m, y ∈ M) ↔ -1 < m ∧ m < 0 :=
sorry

end range_of_m_l2199_219974


namespace bug_meeting_point_l2199_219970

/-- Represents a triangle with given side lengths -/
structure Triangle where
  pq : ℝ
  qr : ℝ
  pr : ℝ

/-- Represents a bug moving along the perimeter of a triangle -/
structure Bug where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Calculates the meeting point of two bugs on a triangle's perimeter -/
def meetingPoint (t : Triangle) (b1 b2 : Bug) : ℝ :=
  sorry

theorem bug_meeting_point (t : Triangle) (b1 b2 : Bug) :
  t.pq = 8 ∧ t.qr = 10 ∧ t.pr = 12 ∧
  b1.speed = 2 ∧ b2.speed = 3 ∧
  b1.direction ≠ b2.direction →
  meetingPoint t b1 b2 = 3 :=
sorry

end bug_meeting_point_l2199_219970


namespace books_ratio_l2199_219914

/-- Given the number of books for Loris, Lamont, and Darryl, 
    prove that the ratio of Lamont's books to Darryl's books is 2:1 -/
theorem books_ratio (Loris Lamont Darryl : ℕ) : 
  Loris + 3 = Lamont →  -- Loris needs three more books to have the same as Lamont
  Darryl = 20 →  -- Darryl has 20 books
  Loris + Lamont + Darryl = 97 →  -- Total number of books is 97
  Lamont / Darryl = 2 := by
sorry

end books_ratio_l2199_219914


namespace positive_sum_product_iff_l2199_219923

theorem positive_sum_product_iff (a b : ℝ) : (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end positive_sum_product_iff_l2199_219923


namespace three_digit_primes_exist_l2199_219980

theorem three_digit_primes_exist : 
  ∃ (S : Finset Nat), 
    (1 ≤ S.card ∧ S.card ≤ 10) ∧ 
    (∀ p ∈ S, 100 ≤ p ∧ p ≤ 999 ∧ Nat.Prime p) :=
by sorry

end three_digit_primes_exist_l2199_219980


namespace quadratic_root_condition_l2199_219912

theorem quadratic_root_condition (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ x₂ > 1 ∧ 
    3 * x₁^2 + a * (a - 6) * x₁ - 3 = 0 ∧ 
    3 * x₂^2 + a * (a - 6) * x₂ - 3 = 0) ↔ 
  (0 < a ∧ a < 6) :=
sorry

end quadratic_root_condition_l2199_219912


namespace green_ratio_l2199_219930

theorem green_ratio (total : ℕ) (girls : ℕ) (yellow : ℕ) 
  (h_total : total = 30)
  (h_girls : girls = 18)
  (h_yellow : yellow = 9)
  (h_pink : girls / 3 = 6) :
  (total - (girls / 3 + yellow)) / total = 1 / 2 := by
  sorry

end green_ratio_l2199_219930


namespace cubic_sequence_with_two_squares_exists_l2199_219984

/-- A cubic sequence is a sequence of integers given by a_n = n^3 + bn^2 + cn + d,
    where b, c, and d are integer constants and n ranges over all integers. -/
def CubicSequence (b c d : ℤ) : ℤ → ℤ := fun n ↦ n^3 + b*n^2 + c*n + d

/-- A number is a perfect square if there exists an integer whose square equals the number. -/
def IsPerfectSquare (x : ℤ) : Prop := ∃ k : ℤ, k^2 = x

theorem cubic_sequence_with_two_squares_exists : ∃ b c d : ℤ,
  let a := CubicSequence b c d
  IsPerfectSquare (a 2015) ∧
  IsPerfectSquare (a 2016) ∧
  (∀ n : ℤ, n ≠ 2015 ∧ n ≠ 2016 → ¬ IsPerfectSquare (a n)) ∧
  a 2015 * a 2016 = 0 :=
sorry

end cubic_sequence_with_two_squares_exists_l2199_219984


namespace smallest_sum_l2199_219936

/-- Given positive integers A, B, C, and D satisfying certain conditions,
    the smallest possible sum A + B + C + D is 43. -/
theorem smallest_sum (A B C D : ℕ+) : 
  (∃ r : ℚ, B.val - A.val = r ∧ C.val - B.val = r) →  -- arithmetic sequence condition
  (∃ q : ℚ, C.val / B.val = q ∧ D.val / C.val = q) →  -- geometric sequence condition
  C.val / B.val = 4 / 3 →                             -- given ratio
  A.val + B.val + C.val + D.val ≥ 43 :=               -- smallest possible sum
by sorry

end smallest_sum_l2199_219936


namespace greatest_difference_of_valid_units_digits_l2199_219910

/-- A function that checks if a number is divisible by 4 -/
def isDivisibleBy4 (n : ℕ) : Prop := n % 4 = 0

/-- The set of all possible three-digit numbers starting with 47 -/
def threeDigitNumbers : Set ℕ := {n : ℕ | 470 ≤ n ∧ n ≤ 479}

/-- The set of all three-digit numbers starting with 47 that are divisible by 4 -/
def divisibleNumbers : Set ℕ := {n ∈ threeDigitNumbers | isDivisibleBy4 n}

/-- The set of units digits of numbers in divisibleNumbers -/
def validUnitsDigits : Set ℕ := {x : ℕ | ∃ n ∈ divisibleNumbers, n % 10 = x}

theorem greatest_difference_of_valid_units_digits :
  ∃ (a b : ℕ), a ∈ validUnitsDigits ∧ b ∈ validUnitsDigits ∧ 
  ∀ (x y : ℕ), x ∈ validUnitsDigits → y ∈ validUnitsDigits → 
  (max a b - min a b : ℤ) ≥ (max x y - min x y) ∧
  (max a b - min a b : ℤ) = 4 :=
sorry

end greatest_difference_of_valid_units_digits_l2199_219910


namespace sum_real_imag_parts_l2199_219967

theorem sum_real_imag_parts (z : ℂ) : z = 1 + I → (z.re + z.im = 2) := by
  sorry

end sum_real_imag_parts_l2199_219967


namespace product_digit_sum_l2199_219938

def digit_repeat (d₁ d₂ d₃ : ℕ) (n : ℕ) : ℕ :=
  (d₁ * 10^2 + d₂ * 10 + d₃) * (10^(3*n) - 1) / 999

def a : ℕ := digit_repeat 3 0 3 33
def b : ℕ := digit_repeat 5 0 5 33

theorem product_digit_sum :
  (a * b % 10) + ((a * b / 1000) % 10) = 8 := by sorry

end product_digit_sum_l2199_219938


namespace grocery_store_bottles_l2199_219953

theorem grocery_store_bottles (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 28) (h2 : diet_soda = 2) : 
  regular_soda + diet_soda = 30 := by
  sorry

end grocery_store_bottles_l2199_219953


namespace ball_distance_theorem_l2199_219991

def initial_height : ℚ := 120
def rebound_fraction : ℚ := 1/3
def num_bounces : ℕ := 4

def descent_distance (n : ℕ) : ℚ :=
  initial_height * (rebound_fraction ^ n)

def total_distance : ℚ :=
  2 * (initial_height * (1 - rebound_fraction^(num_bounces + 1)) / (1 - rebound_fraction)) - initial_height

theorem ball_distance_theorem :
  total_distance = 5000 / 27 := by sorry

end ball_distance_theorem_l2199_219991


namespace work_completion_time_l2199_219958

theorem work_completion_time (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (5 : ℝ) / 12 + 4 / b + 3 / c = 1 →
  1 / ((1 / b) + (1 / c)) = 12 := by
  sorry

end work_completion_time_l2199_219958


namespace ben_old_car_sale_amount_l2199_219911

def old_car_cost : ℕ := 1900
def remaining_debt : ℕ := 2000

def new_car_cost : ℕ := 2 * old_car_cost

def amount_paid_off : ℕ := new_car_cost - remaining_debt

theorem ben_old_car_sale_amount : amount_paid_off = 1800 := by
  sorry

end ben_old_car_sale_amount_l2199_219911


namespace similar_triangles_AB_length_l2199_219998

/-- Two similar triangles with given side lengths and angles -/
structure SimilarTriangles where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  DE : ℝ
  EF : ℝ
  DF : ℝ
  angleBAC : ℝ
  angleEDF : ℝ

/-- Theorem stating that for the given similar triangles, AB = 75/17 -/
theorem similar_triangles_AB_length (t : SimilarTriangles)
  (h1 : t.AB = 5)
  (h2 : t.BC = 17)
  (h3 : t.AC = 12)
  (h4 : t.DE = 9)
  (h5 : t.EF = 15)
  (h6 : t.DF = 12)
  (h7 : t.angleBAC = 120)
  (h8 : t.angleEDF = 120) :
  t.AB = 75 / 17 := by
  sorry

end similar_triangles_AB_length_l2199_219998


namespace surface_area_is_14_l2199_219964

/-- The surface area of a rectangular prism formed by joining three 1x1x1 cubes side by side -/
def surface_area_of_prism : ℕ :=
  let length : ℕ := 3
  let width : ℕ := 1
  let height : ℕ := 1
  2 * (length * width + length * height + width * height)

/-- Theorem stating that the surface area of the prism is 14 -/
theorem surface_area_is_14 : surface_area_of_prism = 14 := by
  sorry

end surface_area_is_14_l2199_219964


namespace lucky_larry_coincidence_l2199_219986

theorem lucky_larry_coincidence :
  let a : ℤ := 2
  let b : ℤ := 3
  let c : ℤ := 4
  let d : ℤ := 5
  ∃ f : ℤ, (a + b - c + d - f = a + (b - (c + (d - f)))) ∧ f = 5 :=
by
  sorry

end lucky_larry_coincidence_l2199_219986


namespace distance_AB_l2199_219982

/-- The distance between points A(2,1) and B(5,-1) is √13. -/
theorem distance_AB : Real.sqrt 13 = Real.sqrt ((5 - 2)^2 + (-1 - 1)^2) := by
  sorry

end distance_AB_l2199_219982


namespace min_value_x_plus_y_l2199_219981

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 1) (h2 : x * y = 2 * x + y + 2) :
  x + y ≥ 7 ∧ ∃ x0 y0, x0 > 1 ∧ x0 * y0 = 2 * x0 + y0 + 2 ∧ x0 + y0 = 7 := by
  sorry

end min_value_x_plus_y_l2199_219981


namespace problem_statement_l2199_219971

theorem problem_statement (A B : ℝ) :
  (∀ x : ℝ, x ≠ 5 → A / (x - 5) + B * (x + 1) = (-2 * x^2 + 16 * x + 18) / (x - 5)) →
  A + B = 0 := by
  sorry

end problem_statement_l2199_219971


namespace max_concert_tickets_l2199_219961

theorem max_concert_tickets (ticket_price : ℕ) (budget : ℕ) : 
  ticket_price = 15 → budget = 120 → 
  ∃ (max_tickets : ℕ), max_tickets = 8 ∧ 
    (∀ n : ℕ, n * ticket_price ≤ budget → n ≤ max_tickets) :=
by sorry

end max_concert_tickets_l2199_219961


namespace product_remainder_eleven_l2199_219951

theorem product_remainder_eleven : (1010 * 1011 * 1012 * 1013 * 1014) % 11 = 0 := by
  sorry

end product_remainder_eleven_l2199_219951


namespace blanket_price_problem_l2199_219992

theorem blanket_price_problem (price1 price2 avg_price : ℕ) 
  (count1 count2 count_unknown : ℕ) :
  price1 = 100 →
  price2 = 150 →
  avg_price = 150 →
  count1 = 3 →
  count2 = 3 →
  count_unknown = 2 →
  (count1 * price1 + count2 * price2 + count_unknown * 
    ((count1 + count2 + count_unknown) * avg_price - 
     count1 * price1 - count2 * price2) / count_unknown) / 
    (count1 + count2 + count_unknown) = avg_price →
  ((count1 + count2 + count_unknown) * avg_price - 
   count1 * price1 - count2 * price2) / count_unknown = 225 :=
by sorry

end blanket_price_problem_l2199_219992


namespace profit_maximizing_price_l2199_219983

/-- Given the initial conditions of a pricing problem, prove that the profit-maximizing price is 95 yuan. -/
theorem profit_maximizing_price 
  (initial_cost : ℝ)
  (initial_price : ℝ)
  (initial_units : ℝ)
  (price_increase : ℝ)
  (units_decrease : ℝ)
  (h1 : initial_cost = 80)
  (h2 : initial_price = 90)
  (h3 : initial_units = 400)
  (h4 : price_increase = 1)
  (h5 : units_decrease = 20)
  : ∃ (max_price : ℝ), max_price = 95 ∧ 
    ∀ (x : ℝ), 
      (initial_price + x) * (initial_units - units_decrease * x) - initial_cost * (initial_units - units_decrease * x) ≤ 
      (initial_price + (max_price - initial_price)) * (initial_units - units_decrease * (max_price - initial_price)) - 
      initial_cost * (initial_units - units_decrease * (max_price - initial_price)) :=
by sorry

end profit_maximizing_price_l2199_219983


namespace parabola_line_intersection_l2199_219916

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {(x, y) | y = x^2}

/-- The point Q -/
def Q : ℝ × ℝ := (20, 14)

/-- The line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y - Q.2 = m * (x - Q.1)}

/-- The condition for non-intersection -/
def no_intersection (m : ℝ) : Prop :=
  line_through_Q m ∩ P = ∅

/-- The theorem statement -/
theorem parabola_line_intersection :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 80 := by sorry

end parabola_line_intersection_l2199_219916


namespace fence_length_l2199_219993

/-- Given a straight wire fence with 12 equally spaced posts, where the distance between
    the third and the sixth post is 3.3 m, the total length of the fence is 12.1 meters. -/
theorem fence_length (num_posts : ℕ) (distance_3_to_6 : ℝ) :
  num_posts = 12 →
  distance_3_to_6 = 3.3 →
  (num_posts - 1 : ℝ) * (distance_3_to_6 / 3) = 12.1 := by
  sorry

end fence_length_l2199_219993


namespace children_boarding_bus_l2199_219960

theorem children_boarding_bus (initial_children final_children : ℕ) 
  (h1 : initial_children = 18)
  (h2 : final_children = 25) :
  final_children - initial_children = 7 := by
  sorry

end children_boarding_bus_l2199_219960


namespace diagonals_intersect_l2199_219913

-- Define a regular 30-sided polygon
def RegularPolygon30 : Type := Unit

-- Define the sine function (simplified for this context)
noncomputable def sin (angle : ℝ) : ℝ := sorry

-- Define the cosine function (simplified for this context)
noncomputable def cos (angle : ℝ) : ℝ := sorry

-- Theorem statement
theorem diagonals_intersect (polygon : RegularPolygon30) : 
  (sin (6 * π / 180) * sin (18 * π / 180) * sin (84 * π / 180) = 
   sin (12 * π / 180) * sin (12 * π / 180) * sin (48 * π / 180)) ∧
  (sin (6 * π / 180) * sin (36 * π / 180) * sin (54 * π / 180) = 
   sin (30 * π / 180) * sin (12 * π / 180) * sin (12 * π / 180)) ∧
  (sin (36 * π / 180) * sin (18 * π / 180) * sin (6 * π / 180) = 
   cos (36 * π / 180) * cos (36 * π / 180)) :=
by sorry

end diagonals_intersect_l2199_219913


namespace vivi_fabric_purchase_l2199_219956

/-- The total yards of fabric Vivi bought -/
def total_yards (checkered_cost plain_cost cost_per_yard : ℚ) : ℚ :=
  checkered_cost / cost_per_yard + plain_cost / cost_per_yard

/-- Proof that Vivi bought 16 yards of fabric -/
theorem vivi_fabric_purchase :
  total_yards 75 45 (7.5 : ℚ) = 16 := by
  sorry

end vivi_fabric_purchase_l2199_219956


namespace square_minus_product_plus_square_l2199_219968

theorem square_minus_product_plus_square (a b : ℝ) 
  (sum_eq : a + b = 6) 
  (product_eq : a * b = 3) : 
  a^2 - a*b + b^2 = 27 := by sorry

end square_minus_product_plus_square_l2199_219968


namespace alcohol_percentage_in_mixture_l2199_219990

/-- Given a mixture of water and an alcohol solution, calculate the percentage of alcohol in the new mixture. -/
theorem alcohol_percentage_in_mixture 
  (water_volume : ℝ) 
  (solution_volume : ℝ) 
  (original_alcohol_percentage : ℝ) : 
  water_volume = 16 → 
  solution_volume = 24 → 
  original_alcohol_percentage = 90 → 
  (original_alcohol_percentage / 100 * solution_volume) / (water_volume + solution_volume) * 100 = 54 := by
  sorry

end alcohol_percentage_in_mixture_l2199_219990


namespace shari_walked_13_miles_l2199_219901

/-- Represents Shari's walking pattern -/
structure WalkingPattern where
  rate1 : ℝ  -- Rate for the first phase in miles per hour
  time1 : ℝ  -- Time for the first phase in hours
  rate2 : ℝ  -- Rate for the second phase in miles per hour
  time2 : ℝ  -- Time for the second phase in hours

/-- Calculates the total distance walked given a WalkingPattern -/
def totalDistance (w : WalkingPattern) : ℝ :=
  w.rate1 * w.time1 + w.rate2 * w.time2

/-- Shari's actual walking pattern -/
def sharisWalk : WalkingPattern :=
  { rate1 := 4
    time1 := 2
    rate2 := 5
    time2 := 1 }

/-- Theorem stating that Shari walked 13 miles in total -/
theorem shari_walked_13_miles :
  totalDistance sharisWalk = 13 := by
  sorry


end shari_walked_13_miles_l2199_219901


namespace polynomial_factorization_l2199_219921

theorem polynomial_factorization (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  -(a - b) * (b - c) * (c - a) * (a^2 + a*b + b^2 + b*c + c^2 + a*c) := by
  sorry

end polynomial_factorization_l2199_219921


namespace tims_coins_value_l2199_219989

/-- Represents the number of coins Tim has -/
def total_coins : ℕ := 18

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the number of dimes Tim has -/
def num_dimes : ℕ := 8

/-- Represents the number of quarters Tim has -/
def num_quarters : ℕ := 10

/-- Theorem stating the total value of Tim's coins -/
theorem tims_coins_value :
  (num_dimes * dime_value + num_quarters * quarter_value = 330) ∧
  (num_dimes + num_quarters = total_coins) ∧
  (num_dimes + 2 = num_quarters) :=
sorry

end tims_coins_value_l2199_219989


namespace range_of_a_eq_l2199_219965

/-- Proposition p: The solution set of the inequality x^2 + (a-1)x + a^2 < 0 is empty. -/
def prop_p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + a^2 ≥ 0

/-- Quadratic function f(x) = x^2 - mx + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2

/-- Proposition q: f(3/2 + x) = f(3/2 - x), and max(f(x)) = 2 for x ∈ [0, a] -/
def prop_q (a : ℝ) : Prop :=
  ∃ m, (∀ x, f m ((3:ℝ)/2 + x) = f m ((3:ℝ)/2 - x)) ∧
       (∀ x, x ∈ Set.Icc 0 a → f m x ≤ 2) ∧
       (∃ x, x ∈ Set.Icc 0 a ∧ f m x = 2)

/-- The range of a given the conditions -/
def range_of_a : Set ℝ :=
  {a | (¬(prop_p a ∧ prop_q a)) ∧ (prop_p a ∨ prop_q a)}

theorem range_of_a_eq :
  range_of_a = Set.Iic (-1) ∪ Set.Ioo 0 (1/3) ∪ Set.Ioi 3 :=
sorry

end range_of_a_eq_l2199_219965


namespace min_cost_stationery_l2199_219947

/-- Represents the cost and quantity of stationery items --/
structure Stationery where
  costA : ℕ  -- Cost of item A
  costB : ℕ  -- Cost of item B
  totalItems : ℕ  -- Total number of items to purchase
  minCost : ℕ  -- Minimum total cost
  maxCost : ℕ  -- Maximum total cost

/-- Theorem stating the minimum cost for the stationery purchase --/
theorem min_cost_stationery (s : Stationery) 
  (h1 : 2 * s.costA + s.costB = 35)
  (h2 : s.costA + 3 * s.costB = 30)
  (h3 : s.totalItems = 120)
  (h4 : s.minCost = 955)
  (h5 : s.maxCost = 1000) :
  ∃ (x : ℕ), x ≥ 36 ∧ 
             10 * x + 600 = 960 ∧ 
             ∀ (y : ℕ), y ≥ 36 → 10 * y + 600 ≥ 960 := by
  sorry

end min_cost_stationery_l2199_219947


namespace discount_problem_l2199_219959

/-- Proves that if a 25% discount on a purchase is $40, then the total amount paid after the discount is $120. -/
theorem discount_problem (original_price : ℝ) (discount_amount : ℝ) (discount_percentage : ℝ) 
  (h1 : discount_amount = 40)
  (h2 : discount_percentage = 0.25)
  (h3 : discount_amount = discount_percentage * original_price) :
  original_price - discount_amount = 120 := by
  sorry

#check discount_problem

end discount_problem_l2199_219959


namespace triangle_altitude_proof_l2199_219937

theorem triangle_altitude_proof (a b c h : ℝ) : 
  a = 13 ∧ b = 15 ∧ c = 22 →
  a + b > c ∧ a + c > b ∧ b + c > a →
  h = (30 * Real.sqrt 10) / 11 →
  (1 / 2) * c * h = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) :=
by sorry

end triangle_altitude_proof_l2199_219937


namespace prob_a_not_less_than_b_expected_tests_scheme_b_l2199_219934

/-- Represents the two testing schemes -/
inductive TestScheme
| A
| B

/-- Represents the possible outcomes of a test -/
inductive TestResult
| Positive
| Negative

/-- The total number of swimmers -/
def totalSwimmers : ℕ := 5

/-- The number of swimmers who have taken stimulants -/
def stimulantUsers : ℕ := 1

/-- The number of swimmers tested in the first step of Scheme B -/
def schemeBFirstTest : ℕ := 3

/-- Function to calculate the probability that Scheme A requires no fewer tests than Scheme B -/
def probANotLessThanB : ℚ :=
  18/25

/-- Function to calculate the expected number of tests in Scheme B -/
def expectedTestsSchemeB : ℚ :=
  2.4

/-- Theorem stating the probability that Scheme A requires no fewer tests than Scheme B -/
theorem prob_a_not_less_than_b :
  probANotLessThanB = 18/25 := by sorry

/-- Theorem stating the expected number of tests in Scheme B -/
theorem expected_tests_scheme_b :
  expectedTestsSchemeB = 2.4 := by sorry

end prob_a_not_less_than_b_expected_tests_scheme_b_l2199_219934


namespace final_number_independent_of_operations_l2199_219939

/-- Represents the state of the blackboard with counts of 0, 1, and 2 --/
structure BoardState where
  count0 : Nat
  count1 : Nat
  count2 : Nat

/-- Represents a single operation of replacing two numbers with the third --/
inductive Operation
  | replace01with2
  | replace02with1
  | replace12with0

/-- Applies an operation to a board state --/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.replace01with2 => { count0 := state.count0 - 1, count1 := state.count1 - 1, count2 := state.count2 + 1 }
  | Operation.replace02with1 => { count0 := state.count0 - 1, count1 := state.count1 + 1, count2 := state.count2 - 1 }
  | Operation.replace12with0 => { count0 := state.count0 + 1, count1 := state.count1 - 1, count2 := state.count2 - 1 }

/-- Checks if the board state has only one number remaining --/
def isFinalState (state : BoardState) : Bool :=
  (state.count0 > 0 && state.count1 = 0 && state.count2 = 0) ||
  (state.count0 = 0 && state.count1 > 0 && state.count2 = 0) ||
  (state.count0 = 0 && state.count1 = 0 && state.count2 > 0)

/-- Gets the final number on the board --/
def getFinalNumber (state : BoardState) : Nat :=
  if state.count0 > 0 then 0
  else if state.count1 > 0 then 1
  else 2

/-- Theorem: The final number is determined by initial counts and their parities --/
theorem final_number_independent_of_operations (initialState : BoardState) 
  (ops1 ops2 : List Operation) 
  (h1 : isFinalState (ops1.foldl applyOperation initialState))
  (h2 : isFinalState (ops2.foldl applyOperation initialState)) :
  getFinalNumber (ops1.foldl applyOperation initialState) = 
  getFinalNumber (ops2.foldl applyOperation initialState) := by
  sorry

#check final_number_independent_of_operations

end final_number_independent_of_operations_l2199_219939


namespace big_fifteen_games_l2199_219920

/-- Represents the Big Fifteen Basketball Conference -/
structure BigFifteenConference where
  numDivisions : Nat
  teamsPerDivision : Nat
  intraDivisionGames : Nat
  interDivisionGames : Nat
  nonConferenceGames : Nat

/-- Calculates the total number of games in the conference -/
def totalGames (conf : BigFifteenConference) : Nat :=
  let intraDivisionTotal := conf.numDivisions * (conf.teamsPerDivision.choose 2) * conf.intraDivisionGames
  let interDivisionTotal := conf.numDivisions * conf.teamsPerDivision * (conf.numDivisions - 1) * conf.teamsPerDivision / 2
  let nonConferenceTotal := conf.numDivisions * conf.teamsPerDivision * conf.nonConferenceGames
  intraDivisionTotal + interDivisionTotal + nonConferenceTotal

/-- Theorem stating that the total number of games in the Big Fifteen Conference is 270 -/
theorem big_fifteen_games :
  totalGames {
    numDivisions := 3,
    teamsPerDivision := 5,
    intraDivisionGames := 3,
    interDivisionGames := 1,
    nonConferenceGames := 2
  } = 270 := by sorry


end big_fifteen_games_l2199_219920


namespace problem_one_problem_two_l2199_219906

-- Define a function to represent mixed numbers
def mixed_number (whole : Int) (numerator : Int) (denominator : Int) : Rat :=
  whole + (numerator : Rat) / (denominator : Rat)

-- Problem 1
theorem problem_one : 
  mixed_number 28 5 7 + mixed_number (-25) (-1) 7 = mixed_number 3 4 7 := by
  sorry

-- Problem 2
theorem problem_two :
  mixed_number (-2022) (-2) 7 + mixed_number (-2023) (-4) 7 + (4046 : Rat) + (-1 : Rat) / 7 = 0 := by
  sorry

end problem_one_problem_two_l2199_219906


namespace chocolate_bars_count_l2199_219929

/-- Represents the number of chocolate bars in the colossal box -/
def chocolate_bars_in_colossal_box : ℕ :=
  let sizable_boxes : ℕ := 350
  let small_boxes_per_sizable : ℕ := 49
  let chocolate_bars_per_small : ℕ := 75
  sizable_boxes * small_boxes_per_sizable * chocolate_bars_per_small

/-- Proves that the number of chocolate bars in the colossal box is 1,287,750 -/
theorem chocolate_bars_count : chocolate_bars_in_colossal_box = 1287750 := by
  sorry

end chocolate_bars_count_l2199_219929


namespace arithmetic_sequence_general_term_l2199_219966

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_mean1 : (a 2 + a 6) / 2 = 5)
  (h_mean2 : (a 3 + a 7) / 2 = 7) :
  ∃ (b c : ℝ), ∀ n : ℕ, a n = b * n + c ∧ b = 2 ∧ c = -3 :=
sorry

end arithmetic_sequence_general_term_l2199_219966


namespace zoo_new_species_l2199_219935

theorem zoo_new_species (initial_types : ℕ) (time_per_type : ℕ) (total_time_after : ℕ) : 
  initial_types = 5 → 
  time_per_type = 6 → 
  total_time_after = 54 → 
  (initial_types + (total_time_after / time_per_type - initial_types)) = 9 :=
by sorry

end zoo_new_species_l2199_219935


namespace bond_investment_problem_l2199_219915

theorem bond_investment_problem (interest_income : ℝ) (rate1 rate2 : ℝ) (amount1 : ℝ) :
  interest_income = 1900 →
  rate1 = 0.0575 →
  rate2 = 0.0625 →
  amount1 = 20000 →
  ∃ amount2 : ℝ,
    amount1 * rate1 + amount2 * rate2 = interest_income ∧
    amount1 + amount2 = 32000 := by
  sorry

#check bond_investment_problem

end bond_investment_problem_l2199_219915


namespace janet_horses_count_l2199_219925

def fertilizer_per_horse_per_day : ℕ := 5
def total_acres : ℕ := 20
def fertilizer_per_acre : ℕ := 400
def acres_fertilized_per_day : ℕ := 4
def days_to_fertilize : ℕ := 25

def janet_horses : ℕ := 64

theorem janet_horses_count : janet_horses = 
  (total_acres * fertilizer_per_acre) / 
  (fertilizer_per_horse_per_day * days_to_fertilize) := by
  sorry

end janet_horses_count_l2199_219925


namespace linear_function_quadrants_l2199_219948

/-- A linear function f(x) = mx + b passes through a quadrant if there exists a point (x, y) in that quadrant such that y = mx + b -/
def passes_through_quadrant (m b : ℝ) (quad : Nat) : Prop :=
  match quad with
  | 1 => ∃ x y, x > 0 ∧ y > 0 ∧ y = m * x + b
  | 2 => ∃ x y, x < 0 ∧ y > 0 ∧ y = m * x + b
  | 3 => ∃ x y, x < 0 ∧ y < 0 ∧ y = m * x + b
  | 4 => ∃ x y, x > 0 ∧ y < 0 ∧ y = m * x + b
  | _ => False

/-- The graph of y = -5x + 5 passes through Quadrants I, II, and IV -/
theorem linear_function_quadrants :
  passes_through_quadrant (-5) 5 1 ∧
  passes_through_quadrant (-5) 5 2 ∧
  passes_through_quadrant (-5) 5 4 :=
by sorry

end linear_function_quadrants_l2199_219948


namespace mike_total_spent_l2199_219918

def trumpet_price : Float := 267.35
def song_book_price : Float := 12.95
def trumpet_case_price : Float := 74.50
def cleaning_kit_price : Float := 28.99
def valve_oils_price : Float := 18.75

theorem mike_total_spent : 
  trumpet_price + song_book_price + trumpet_case_price + cleaning_kit_price + valve_oils_price = 402.54 := by
  sorry

end mike_total_spent_l2199_219918


namespace line_moved_down_l2199_219941

/-- Given a line with equation y = 2x + 3, prove that moving it down by 5 units
    results in the equation y = 2x - 2. -/
theorem line_moved_down (x y : ℝ) :
  (y = 2 * x + 3) → (y - 5 = 2 * x - 2) := by
  sorry

end line_moved_down_l2199_219941


namespace expand_product_l2199_219996

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9*x + 18 := by
  sorry

end expand_product_l2199_219996


namespace soy_sauce_bottle_ounces_l2199_219955

/-- Represents the number of ounces in one cup -/
def ounces_per_cup : ℕ := 8

/-- Represents the number of cups of soy sauce required for the first recipe -/
def recipe1_cups : ℕ := 2

/-- Represents the number of cups of soy sauce required for the second recipe -/
def recipe2_cups : ℕ := 1

/-- Represents the number of cups of soy sauce required for the third recipe -/
def recipe3_cups : ℕ := 3

/-- Represents the number of bottles Stephanie needs to buy -/
def bottles_needed : ℕ := 3

/-- Theorem stating that one bottle of soy sauce contains 16 ounces -/
theorem soy_sauce_bottle_ounces : 
  (recipe1_cups + recipe2_cups + recipe3_cups) * ounces_per_cup / bottles_needed = 16 := by
  sorry

end soy_sauce_bottle_ounces_l2199_219955


namespace ben_needs_14_eggs_l2199_219932

/-- Represents the weekly egg requirements for a community -/
structure EggRequirements where
  saly : ℕ
  ben : ℕ
  ked : ℕ
  total_monthly : ℕ

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- Checks if the given egg requirements are valid -/
def is_valid_requirements (req : EggRequirements) : Prop :=
  req.saly = 10 ∧
  req.ked = req.ben / 2 ∧
  req.total_monthly = weeks_in_month * (req.saly + req.ben + req.ked)

/-- Theorem stating that Ben needs 14 eggs per week -/
theorem ben_needs_14_eggs (req : EggRequirements) 
  (h : is_valid_requirements req) (h_total : req.total_monthly = 124) : 
  req.ben = 14 := by
  sorry


end ben_needs_14_eggs_l2199_219932


namespace equal_face_areas_not_imply_equal_volumes_l2199_219907

/-- A tetrahedron with its volume and face areas -/
structure Tetrahedron where
  volume : ℝ
  face_areas : Fin 4 → ℝ

/-- Two tetrahedrons have equal face areas -/
def equal_face_areas (t1 t2 : Tetrahedron) : Prop :=
  ∀ i : Fin 4, t1.face_areas i = t2.face_areas i

/-- Theorem stating that equal face areas do not imply equal volumes -/
theorem equal_face_areas_not_imply_equal_volumes :
  ∃ (t1 t2 : Tetrahedron), equal_face_areas t1 t2 ∧ t1.volume ≠ t2.volume :=
sorry

end equal_face_areas_not_imply_equal_volumes_l2199_219907


namespace evaluate_expression_power_sum_given_equation_l2199_219917

-- Problem 1
theorem evaluate_expression (x y : ℝ) (hx : x = 0.5) (hy : y = -1) :
  (x - 5*y) * (-x - 5*y) - (-x + 5*y)^2 = -5.5 := by sorry

-- Problem 2
theorem power_sum_given_equation (a b : ℝ) (h : a^2 - 2*a + b^2 + 4*b + 5 = 0) :
  (a + b)^2013 = -1 := by sorry

end evaluate_expression_power_sum_given_equation_l2199_219917


namespace harper_gift_cost_l2199_219957

/-- Harper's gift-buying problem -/
theorem harper_gift_cost (son_teachers daughter_teachers total_spent : ℕ) 
  (h1 : son_teachers = 3)
  (h2 : daughter_teachers = 4)
  (h3 : total_spent = 70) :
  total_spent / (son_teachers + daughter_teachers) = 10 := by
  sorry

#check harper_gift_cost

end harper_gift_cost_l2199_219957


namespace two_sqrt_three_in_set_l2199_219950

theorem two_sqrt_three_in_set : 2 * Real.sqrt 3 ∈ {x : ℝ | x < 4} := by
  sorry

end two_sqrt_three_in_set_l2199_219950


namespace four_tangent_lines_l2199_219988

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if two circles are on the same side of a line -/
def sameSideOfLine (A B : Circle) (m : Line) : Prop := sorry

/-- Predicate to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Function to reflect a line over another line -/
def reflectLine (l : Line) (m : Line) : Line := sorry

/-- The main theorem -/
theorem four_tangent_lines (A B : Circle) (m : Line) 
  (h : sameSideOfLine A B m) : 
  ∃ (l₁ l₂ l₃ l₄ : Line), 
    (isTangent l₁ A ∧ isTangent (reflectLine l₁ m) B) ∧
    (isTangent l₂ A ∧ isTangent (reflectLine l₂ m) B) ∧
    (isTangent l₃ A ∧ isTangent (reflectLine l₃ m) B) ∧
    (isTangent l₄ A ∧ isTangent (reflectLine l₄ m) B) ∧
    (l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₁ ≠ l₄ ∧ l₂ ≠ l₃ ∧ l₂ ≠ l₄ ∧ l₃ ≠ l₄) :=
by sorry

end four_tangent_lines_l2199_219988


namespace largest_angle_of_triangle_l2199_219926

/-- Given a triangle DEF with side lengths d, e, and f satisfying certain conditions,
    prove that its largest angle is 120°. -/
theorem largest_angle_of_triangle (d e f : ℝ) (h1 : d + 3*e + 3*f = d^2) (h2 : d + 3*e - 3*f = -4) :
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A ≤ 120 ∧ B ≤ 120 ∧ max A (max B C) = 120 := by
  sorry

end largest_angle_of_triangle_l2199_219926


namespace polynomial_divisibility_l2199_219997

theorem polynomial_divisibility : ∀ (x : ℂ),
  (x^5 + x^4 + x^3 + x^2 + x + 1 = 0) →
  (x^55 + x^44 + x^33 + x^22 + x^11 + 1 = 0) := by
sorry

end polynomial_divisibility_l2199_219997


namespace converse_proposition_l2199_219903

theorem converse_proposition :
  (∀ x y : ℝ, (x ≤ 2 ∨ y ≤ 2) → x + y ≤ 4) ↔
  (¬∀ x y : ℝ, (x > 2 ∧ y > 2) → x + y > 4) :=
by sorry

end converse_proposition_l2199_219903


namespace sum_first_seven_primes_mod_eighth_prime_l2199_219972

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]
def eighth_prime : Nat := 19

theorem sum_first_seven_primes_mod_eighth_prime :
  (first_seven_primes.sum) % eighth_prime = 1 := by
  sorry

end sum_first_seven_primes_mod_eighth_prime_l2199_219972


namespace sum_of_two_with_prime_bound_l2199_219977

theorem sum_of_two_with_prime_bound (n : ℕ) (h : n ≥ 50) :
  ∃ x y : ℕ, n = x + y ∧
    ∀ p : ℕ, p.Prime → (p ∣ x ∨ p ∣ y) → (n : ℝ).sqrt ≥ p :=
  sorry

end sum_of_two_with_prime_bound_l2199_219977


namespace grading_problem_solution_l2199_219931

/-- Represents the grading scenario of Teacher Wang --/
structure GradingScenario where
  initial_rate : ℕ            -- Initial grading rate (assignments per hour)
  new_rate : ℕ                -- New grading rate (assignments per hour)
  change_time : ℕ             -- Time at which the grading rate changed (in hours)
  time_saved : ℕ              -- Time saved due to rate change (in hours)
  total_assignments : ℕ       -- Total number of assignments in the batch

/-- Theorem stating the solution to the grading problem --/
theorem grading_problem_solution (scenario : GradingScenario) : 
  scenario.initial_rate = 6 →
  scenario.new_rate = 8 →
  scenario.change_time = 2 →
  scenario.time_saved = 3 →
  scenario.total_assignments = 84 :=
by sorry

end grading_problem_solution_l2199_219931


namespace straight_line_no_dot_l2199_219933

/-- Represents the properties of an alphabet with dots and straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  dotOnly : ℕ
  allHaveEither : Bool

/-- Theorem: In the given alphabet, the number of letters with a straight line but no dot is 36 -/
theorem straight_line_no_dot (a : Alphabet) 
  (h1 : a.total = 60)
  (h2 : a.both = 20)
  (h3 : a.dotOnly = 4)
  (h4 : a.allHaveEither = true) : 
  a.total - a.both - a.dotOnly = 36 := by
  sorry

#check straight_line_no_dot

end straight_line_no_dot_l2199_219933


namespace dan_placed_16_pencils_l2199_219995

/-- The number of pencils Dan placed on the desk -/
def pencils_dan_placed (drawer : ℕ) (desk_initial : ℕ) (total_after : ℕ) : ℕ :=
  total_after - (drawer + desk_initial)

/-- Theorem stating that Dan placed 16 pencils on the desk -/
theorem dan_placed_16_pencils : 
  pencils_dan_placed 43 19 78 = 16 := by
  sorry

end dan_placed_16_pencils_l2199_219995


namespace square_area_equal_perimeter_triangle_l2199_219944

theorem square_area_equal_perimeter_triangle (a b c : ℝ) (square_side : ℝ) : 
  a = 5.8 ∧ b = 7.5 ∧ c = 10.7 →
  4 * square_side = a + b + c →
  square_side ^ 2 = 36 := by sorry

end square_area_equal_perimeter_triangle_l2199_219944


namespace cube_difference_l2199_219928

theorem cube_difference (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
sorry

end cube_difference_l2199_219928


namespace h_perimeter_is_26_l2199_219963

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def rectanglePerimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Calculates the perimeter of an H-shaped figure formed by three rectangles -/
def hPerimeter (r : Rectangle) : ℝ :=
  2 * r.length + 4 * r.width + 2 * r.length

/-- Theorem: The perimeter of an H-shaped figure formed by three 3x5 inch rectangles is 26 inches -/
theorem h_perimeter_is_26 :
  let r : Rectangle := { length := 5, width := 3 }
  hPerimeter r = 26 := by
  sorry

end h_perimeter_is_26_l2199_219963


namespace student_grouping_l2199_219949

/-- Calculates the minimum number of groups needed to split students -/
def minGroups (totalStudents : ℕ) (maxGroupSize : ℕ) : ℕ :=
  (totalStudents + maxGroupSize - 1) / maxGroupSize

theorem student_grouping (totalStudents : ℕ) (maxGroupSize : ℕ) 
  (h1 : totalStudents = 30) (h2 : maxGroupSize = 12) :
  minGroups totalStudents maxGroupSize = 3 := by
  sorry

#eval minGroups 30 12  -- Should output 3

end student_grouping_l2199_219949


namespace log_inequality_l2199_219902

theorem log_inequality (x : ℝ) (hx : x > 0) : Real.log (x + 1) ≥ x - (1/2) * x^2 := by
  sorry

end log_inequality_l2199_219902


namespace smoothie_ingredients_total_l2199_219994

theorem smoothie_ingredients_total (strawberries yogurt orange_juice : ℚ) 
  (h1 : strawberries = 0.2)
  (h2 : yogurt = 0.1)
  (h3 : orange_juice = 0.2) :
  strawberries + yogurt + orange_juice = 0.5 := by
  sorry

end smoothie_ingredients_total_l2199_219994


namespace quadratic_function_range_l2199_219985

/-- A quadratic function with specific properties -/
def f (a b x : ℝ) : ℝ := -x^2 + a*x + b^2 - b + 1

/-- The theorem statement -/
theorem quadratic_function_range (a b : ℝ) :
  (∀ x, f a b (1 - x) = f a b (1 + x)) →
  (∀ x ∈ Set.Icc (-1) 1, f a b x > 0) →
  b < -1 ∨ b > 2 := by
  sorry

end quadratic_function_range_l2199_219985


namespace second_last_digit_of_power_of_three_is_even_l2199_219922

/-- The second-to-last digit of a natural number -/
def secondLastDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- A natural number is even if it's divisible by 2 -/
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem second_last_digit_of_power_of_three_is_even (n : ℕ) (h : n > 2) :
  isEven (secondLastDigit (3^n)) := by sorry

end second_last_digit_of_power_of_three_is_even_l2199_219922


namespace pascal_triangle_ratio_l2199_219942

/-- Binomial coefficient -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The row number in Pascal's Triangle -/
def n : ℕ := 53

/-- The position of the first entry in the consecutive trio -/
def r : ℕ := 23

/-- Theorem stating that three consecutive entries in row 53 of Pascal's Triangle are in the ratio 4:5:6 -/
theorem pascal_triangle_ratio :
  ∃ (r : ℕ), r < n ∧ 
    (choose n r : ℚ) / (choose n (r + 1)) = 4 / 5 ∧
    (choose n (r + 1) : ℚ) / (choose n (r + 2)) = 5 / 6 := by
  sorry


end pascal_triangle_ratio_l2199_219942


namespace union_equality_implies_m_values_l2199_219909

def A : Set ℝ := {2, 3}
def B (m : ℝ) : Set ℝ := {x | m * x - 1 = 0}

theorem union_equality_implies_m_values (m : ℝ) :
  A ∪ B m = A → m = 1/2 ∨ m = 1/3 := by
  sorry

end union_equality_implies_m_values_l2199_219909


namespace annulus_area_l2199_219979

theorem annulus_area (r₁ r₂ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 2) :
  π * r₂^2 - π * r₁^2 = 3 * π := by sorry

end annulus_area_l2199_219979


namespace max_value_expression_l2199_219999

theorem max_value_expression (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0) :
  (k * x + y)^2 / (x^2 + y^2) ≤ k^2 + 1 := by
  sorry

end max_value_expression_l2199_219999


namespace alberto_bjorn_distance_difference_l2199_219943

/-- The difference in miles biked between Alberto and Bjorn after four hours -/
theorem alberto_bjorn_distance_difference :
  let alberto_distance : ℕ := 60
  let bjorn_distance : ℕ := 45
  alberto_distance - bjorn_distance = 15 := by
sorry

end alberto_bjorn_distance_difference_l2199_219943


namespace train_passing_time_l2199_219946

/-- The time for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 110 →
  train_speed = 65 * (5/18) →
  man_speed = 7 * (5/18) →
  (train_length / (train_speed + man_speed)) = 5.5 := by sorry

end train_passing_time_l2199_219946


namespace certain_multiple_proof_l2199_219962

theorem certain_multiple_proof (n : ℝ) (m : ℝ) (h1 : n = 5) (h2 : 7 * n - 15 = m * n + 10) : m = 2 := by
  sorry

end certain_multiple_proof_l2199_219962


namespace min_value_sqrt_inverse_equality_condition_l2199_219904

theorem min_value_sqrt_inverse (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 4 / x^2 ≥ 4 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 4 / x^2 = 4 * Real.sqrt 2 ↔ x = 2^(4/3) :=
by sorry

end min_value_sqrt_inverse_equality_condition_l2199_219904


namespace sequence_problem_l2199_219973

theorem sequence_problem (a b : ℝ) 
  (h1 : 0 < 2 ∧ 0 < a ∧ 0 < b ∧ 0 < 9)
  (h2 : a - 2 = b - a)  -- arithmetic sequence condition
  (h3 : a / 2 = b / a ∧ b / a = 9 / b)  -- geometric sequence condition
  : a = 4 ∧ b = 6 := by
  sorry

end sequence_problem_l2199_219973


namespace maria_alice_ages_sum_l2199_219945

/-- Maria and Alice's ages problem -/
theorem maria_alice_ages_sum : 
  ∀ (maria alice : ℕ), 
    maria = alice + 8 →  -- Maria is eight years older than Alice
    maria + 10 = 3 * (alice - 6) →  -- Ten years from now, Maria will be three times as old as Alice was six years ago
    maria + alice = 44  -- The sum of their current ages is 44
    := by sorry

end maria_alice_ages_sum_l2199_219945


namespace probability_three_girls_l2199_219900

/-- The probability of choosing 3 girls from a group of 15 members (8 girls and 7 boys) -/
theorem probability_three_girls (total : ℕ) (girls : ℕ) (boys : ℕ) (chosen : ℕ) : 
  total = 15 → girls = 8 → boys = 7 → chosen = 3 →
  (Nat.choose girls chosen : ℚ) / (Nat.choose total chosen : ℚ) = 8 / 65 := by
sorry

end probability_three_girls_l2199_219900


namespace partnership_profit_calculation_l2199_219987

/-- Profit calculation for Mary and Harry's partnership --/
theorem partnership_profit_calculation
  (mary_investment harry_investment : ℚ)
  (effort_share investment_share : ℚ)
  (mary_extra : ℚ)
  (h1 : mary_investment = 700)
  (h2 : harry_investment = 300)
  (h3 : effort_share = 1/3)
  (h4 : investment_share = 2/3)
  (h5 : mary_extra = 800) :
  ∃ (P : ℚ),
    P = 3000 ∧
    (P/6 + (mary_investment / (mary_investment + harry_investment)) * (investment_share * P)) -
    (P/6 + (harry_investment / (mary_investment + harry_investment)) * (investment_share * P)) = mary_extra :=
by sorry

end partnership_profit_calculation_l2199_219987


namespace wrong_mark_calculation_l2199_219927

theorem wrong_mark_calculation (n : Nat) (initial_avg correct_avg : ℝ) (correct_mark : ℝ) :
  n = 10 ∧ 
  initial_avg = 100 ∧ 
  correct_avg = 95 ∧ 
  correct_mark = 10 →
  ∃ wrong_mark : ℝ,
    n * initial_avg = (n - 1) * correct_avg + wrong_mark ∧
    wrong_mark = 60 := by
  sorry

end wrong_mark_calculation_l2199_219927


namespace quadratic_equation_solution_l2199_219954

theorem quadratic_equation_solution :
  ∀ (a b : ℝ),
  (∀ x : ℝ, x^2 - 6*x + 18 = 28 ↔ (x = a ∨ x = b)) →
  a ≥ b →
  a + 3*b = 12 - 2*Real.sqrt 19 :=
by sorry

end quadratic_equation_solution_l2199_219954


namespace composite_rectangle_area_l2199_219969

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- A larger rectangle composed of three identical smaller rectangles -/
structure CompositeRectangle where
  smallRectangle : Rectangle
  count : ℕ

/-- The area of the composite rectangle -/
def CompositeRectangle.area (cr : CompositeRectangle) : ℝ :=
  cr.smallRectangle.area * cr.count

theorem composite_rectangle_area :
  ∀ (r : Rectangle),
    r.width = 8 →
    (CompositeRectangle.area { smallRectangle := r, count := 3 }) = 384 :=
by
  sorry

end composite_rectangle_area_l2199_219969


namespace off_road_vehicle_cost_l2199_219975

theorem off_road_vehicle_cost 
  (dirt_bike_cost : ℕ) 
  (dirt_bike_count : ℕ) 
  (off_road_count : ℕ) 
  (registration_fee : ℕ) 
  (total_cost : ℕ) 
  (h1 : dirt_bike_cost = 150)
  (h2 : dirt_bike_count = 3)
  (h3 : off_road_count = 4)
  (h4 : registration_fee = 25)
  (h5 : total_cost = 1825)
  (h6 : total_cost = dirt_bike_cost * dirt_bike_count + 
                     off_road_count * x + 
                     registration_fee * (dirt_bike_count + off_road_count)) :
  x = 300 := by
  sorry


end off_road_vehicle_cost_l2199_219975


namespace log_equation_solution_l2199_219905

theorem log_equation_solution (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∀ x > 1, 3 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 10 * (Real.log x)^2 / (Real.log a + Real.log b)) →
  b = a^((5 + Real.sqrt 10) / 3) ∨ b = a^((5 - Real.sqrt 10) / 3) := by
  sorry

end log_equation_solution_l2199_219905


namespace possible_values_of_a_l2199_219908

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 19*x^3) 
  (h3 : a - b = x) : 
  a = 3*x ∨ a = -2*x := by
sorry

end possible_values_of_a_l2199_219908


namespace mushroom_distribution_l2199_219952

theorem mushroom_distribution (morning_mushrooms afternoon_mushrooms : ℕ) 
  (rabbit_count : ℕ) (h1 : morning_mushrooms = 94) (h2 : afternoon_mushrooms = 85) 
  (h3 : rabbit_count = 8) :
  let total_mushrooms := morning_mushrooms + afternoon_mushrooms
  (total_mushrooms / rabbit_count = 22) ∧ (total_mushrooms % rabbit_count = 3) :=
by
  sorry

end mushroom_distribution_l2199_219952


namespace team_formation_count_l2199_219940

def num_boys : ℕ := 10
def num_girls : ℕ := 12
def boys_to_select : ℕ := 5
def girls_to_select : ℕ := 3

theorem team_formation_count : 
  (Nat.choose num_boys boys_to_select) * (Nat.choose num_girls girls_to_select) = 55440 := by
  sorry

end team_formation_count_l2199_219940


namespace ellipse_k_range_l2199_219924

-- Define the equation of the ellipse
def is_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) + y^2 / (9 - k) = 1

-- Define the range of k
def valid_k_range (k : ℝ) : Prop :=
  (1 < k ∧ k < 5) ∨ (5 < k ∧ k < 9)

-- Theorem stating the relationship between the ellipse equation and the range of k
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ↔ valid_k_range k :=
sorry

end ellipse_k_range_l2199_219924


namespace quadratic_root_relation_l2199_219978

/-- Given two quadratic equations with coefficients a, b, c, d where the roots of
    a²x² + bx + c = 0 are 2011 times the roots of cx² + dx + a = 0,
    prove that b² = d² -/
theorem quadratic_root_relation (a b c d : ℝ) 
  (h : ∀ (x₁ x₂ : ℝ), c * x₁^2 + d * x₁ + a = 0 ∧ c * x₂^2 + d * x₂ + a = 0 → 
       a^2 * (2011 * x₁)^2 + b * (2011 * x₁) + c = 0 ∧ 
       a^2 * (2011 * x₂)^2 + b * (2011 * x₂) + c = 0) : 
  b^2 = d^2 := by
  sorry

end quadratic_root_relation_l2199_219978


namespace expression_evaluation_l2199_219976

theorem expression_evaluation : 7 ^ 8 - 6 / 2 + 9 ^ 3 + 3 + 12 = 5765542 := by
  sorry

end expression_evaluation_l2199_219976


namespace leonards_age_l2199_219919

theorem leonards_age (leonard nina jerome peter natasha : ℕ) : 
  nina = leonard + 4 →
  nina = jerome / 2 →
  peter = 2 * leonard →
  natasha = peter - 3 →
  leonard + nina + jerome + peter + natasha = 75 →
  leonard = 11 :=
by
  sorry

end leonards_age_l2199_219919
