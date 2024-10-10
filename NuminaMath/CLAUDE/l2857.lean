import Mathlib

namespace average_cost_is_two_l2857_285741

/-- Calculates the average cost per fruit given the costs and quantities of apples, bananas, and oranges. -/
def average_cost_per_fruit (apple_cost banana_cost orange_cost : ℚ) 
                           (apple_qty banana_qty orange_qty : ℕ) : ℚ :=
  let total_cost := apple_cost * apple_qty + banana_cost * banana_qty + orange_cost * orange_qty
  let total_qty := apple_qty + banana_qty + orange_qty
  total_cost / total_qty

/-- Proves that the average cost per fruit is $2 given the specific costs and quantities. -/
theorem average_cost_is_two :
  average_cost_per_fruit 2 1 3 12 4 4 = 2 := by
  sorry

#eval average_cost_per_fruit 2 1 3 12 4 4

end average_cost_is_two_l2857_285741


namespace minimum_gift_cost_l2857_285707

structure Store :=
  (name : String)
  (mom_gift : Nat)
  (dad_gift : Nat)
  (brother_gift : Nat)
  (sister_gift : Nat)
  (shopping_time : Nat)

def stores : List Store := [
  ⟨"Romashka", 1000, 750, 930, 850, 35⟩,
  ⟨"Oduvanchik", 1050, 790, 910, 800, 30⟩,
  ⟨"Nezabudka", 980, 810, 925, 815, 40⟩,
  ⟨"Landysh", 1100, 755, 900, 820, 25⟩
]

def travel_time : Nat := 30
def start_time : Nat := 16 * 60 + 35
def close_time : Nat := 20 * 60

def is_valid_shopping_plan (plan : List Store) : Bool :=
  let total_time := plan.foldl (fun acc s => acc + s.shopping_time) 0 + (plan.length - 1) * travel_time
  start_time + total_time ≤ close_time

def gift_cost (plan : List Store) : Nat :=
  plan.foldl (fun acc s => acc + s.mom_gift + s.dad_gift + s.brother_gift + s.sister_gift) 0

theorem minimum_gift_cost :
  ∃ (plan : List Store),
    plan.length = 4 ∧
    (∀ s : Store, s ∈ plan → s ∈ stores) ∧
    is_valid_shopping_plan plan ∧
    gift_cost plan = 3435 ∧
    (∀ other_plan : List Store,
      other_plan.length = 4 →
      (∀ s : Store, s ∈ other_plan → s ∈ stores) →
      is_valid_shopping_plan other_plan →
      gift_cost other_plan ≥ 3435) :=
sorry

end minimum_gift_cost_l2857_285707


namespace ribbon_division_l2857_285718

theorem ribbon_division (total_ribbon : ℚ) (num_boxes : ℕ) (ribbon_per_box : ℚ) : 
  total_ribbon = 5/12 → 
  num_boxes = 5 → 
  total_ribbon = num_boxes * ribbon_per_box → 
  ribbon_per_box = 1/12 := by
  sorry

end ribbon_division_l2857_285718


namespace gcd_39_91_l2857_285734

theorem gcd_39_91 : Nat.gcd 39 91 = 13 := by
  sorry

end gcd_39_91_l2857_285734


namespace x_over_3_is_directly_proportional_l2857_285740

/-- A function f : ℝ → ℝ is directly proportional if there exists a non-zero constant k such that f(x) = k * x for all x -/
def IsDirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function f(x) = x/3 is directly proportional -/
theorem x_over_3_is_directly_proportional :
  IsDirectlyProportional (fun x => x / 3) := by
  sorry

end x_over_3_is_directly_proportional_l2857_285740


namespace andrey_numbers_l2857_285708

/-- Represents a five-digit number as a tuple of five natural numbers, each between 0 and 9 inclusive. -/
def FiveDigitNumber := (Nat × Nat × Nat × Nat × Nat)

/-- Checks if a given FiveDigitNumber is valid (all digits between 0 and 9). -/
def isValidFiveDigitNumber (n : FiveDigitNumber) : Prop :=
  let (a, b, c, d, e) := n
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9

/-- Converts a FiveDigitNumber to its numerical value. -/
def toNumber (n : FiveDigitNumber) : Nat :=
  let (a, b, c, d, e) := n
  10000 * a + 1000 * b + 100 * c + 10 * d + e

/-- Checks if two FiveDigitNumbers differ by exactly two digits. -/
def differByTwoDigits (n1 n2 : FiveDigitNumber) : Prop :=
  let (a1, b1, c1, d1, e1) := n1
  let (a2, b2, c2, d2, e2) := n2
  (a1 ≠ a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 ≠ b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 = b2 ∧ c1 ≠ c2 ∧ d1 = d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 ≠ d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 ≠ e2)

/-- Checks if a FiveDigitNumber contains a zero. -/
def containsZero (n : FiveDigitNumber) : Prop :=
  let (a, b, c, d, e) := n
  a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0 ∨ e = 0

theorem andrey_numbers (n1 n2 : FiveDigitNumber) 
  (h1 : isValidFiveDigitNumber n1)
  (h2 : isValidFiveDigitNumber n2)
  (h3 : differByTwoDigits n1 n2)
  (h4 : toNumber n1 + toNumber n2 = 111111) :
  containsZero n1 ∨ containsZero n2 := by
  sorry

end andrey_numbers_l2857_285708


namespace quadratic_function_properties_l2857_285727

/-- A quadratic function with the given properties -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The symmetry property of f -/
def symmetry_property (b c : ℝ) : Prop :=
  ∀ x, f b c (2 + x) = f b c (2 - x)

theorem quadratic_function_properties (b c : ℝ) 
  (h : symmetry_property b c) : 
  b = 4 ∧ 
  (∀ a : ℝ, f b c (5/4) ≥ f b c (-a^2 - a + 1)) ∧
  (∀ a : ℝ, f b c (5/4) = f b c (-a^2 - a + 1) ↔ a = -1/2) :=
sorry

end quadratic_function_properties_l2857_285727


namespace circle_through_line_intersections_l2857_285735

/-- Given a line that intersects the coordinate axes, prove that the circle passing through
    the origin and the intersection points has a specific equation. -/
theorem circle_through_line_intersections (x y : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1 / 2 - A.2 / 4 = 1) ∧ 
    (B.1 / 2 - B.2 / 4 = 1) ∧ 
    (A.2 = 0) ∧ 
    (B.1 = 0) ∧
    ((x - 1)^2 + (y + 2)^2 = 5) ↔ 
    (x^2 + y^2 = A.1^2 + A.2^2 ∧ 
     x^2 + y^2 = B.1^2 + B.2^2)) :=
by sorry


end circle_through_line_intersections_l2857_285735


namespace sum_of_a_and_b_l2857_285715

theorem sum_of_a_and_b (a b : ℝ) (h1 : a + 4 * b = 33) (h2 : 6 * a + 3 * b = 51) : 
  a + b = 12 := by
sorry

end sum_of_a_and_b_l2857_285715


namespace first_question_percentage_l2857_285779

/-- The percentage of students who answered the first question correctly -/
def first_question_percent : ℝ := 80

/-- The percentage of students who answered the second question correctly -/
def second_question_percent : ℝ := 55

/-- The percentage of students who answered neither question correctly -/
def neither_question_percent : ℝ := 20

/-- The percentage of students who answered both questions correctly -/
def both_questions_percent : ℝ := 55

theorem first_question_percentage :
  first_question_percent = 100 - neither_question_percent - second_question_percent + both_questions_percent :=
by sorry

end first_question_percentage_l2857_285779


namespace sum_zero_ratio_theorem_l2857_285789

theorem sum_zero_ratio_theorem (x y z w : ℝ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w) 
  (h_sum_zero : x + y + z + w = 0) : 
  (x*y + y*z + z*x + w*x + w*y + w*z) / (x^2 + y^2 + z^2 + w^2) = -1/2 := by
sorry

end sum_zero_ratio_theorem_l2857_285789


namespace line_intersection_y_axis_l2857_285742

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line by two points
structure Line where
  p1 : Point2D
  p2 : Point2D

-- Define the y-axis
def yAxis : Line := { p1 := ⟨0, 0⟩, p2 := ⟨0, 1⟩ }

-- Function to check if a point is on a line
def isPointOnLine (p : Point2D) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

-- Function to check if a point is on the y-axis
def isPointOnYAxis (p : Point2D) : Prop :=
  p.x = 0

-- Theorem statement
theorem line_intersection_y_axis :
  let l : Line := { p1 := ⟨2, 9⟩, p2 := ⟨4, 13⟩ }
  let intersection : Point2D := ⟨0, 5⟩
  isPointOnLine intersection l ∧ isPointOnYAxis intersection := by
  sorry

end line_intersection_y_axis_l2857_285742


namespace quadratic_rewrite_l2857_285709

theorem quadratic_rewrite (b : ℝ) (h1 : b < 0) :
  (∃ n : ℝ, ∀ x : ℝ, x^2 + b*x + 2/3 = (x + n)^2 + 1/4) →
  b = -Real.sqrt 15 / 3 := by
sorry

end quadratic_rewrite_l2857_285709


namespace probability_white_after_red_20_balls_l2857_285721

/-- The probability of drawing a white ball after a red ball has been drawn -/
def probability_white_after_red (total : ℕ) (red : ℕ) (white : ℕ) : ℚ :=
  if total = red + white ∧ red > 0 then
    white / (total - 1 : ℚ)
  else
    0

theorem probability_white_after_red_20_balls :
  probability_white_after_red 20 10 10 = 10 / 19 := by
  sorry

end probability_white_after_red_20_balls_l2857_285721


namespace power_of_product_exponent_l2857_285754

theorem power_of_product_exponent (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end power_of_product_exponent_l2857_285754


namespace like_terms_imply_exponents_l2857_285763

/-- Two algebraic terms are considered like terms if they have the same variables with the same exponents. -/
def are_like_terms (term1 term2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (p q : ℕ), ∀ (a b : ℝ), term1 a b = c₁ * a^p * b^q ∧ term2 a b = c₂ * a^p * b^q

/-- The theorem states that if the given terms are like terms, then m = 4 and n = 2. -/
theorem like_terms_imply_exponents 
  (m n : ℕ) 
  (h : are_like_terms (λ a b => (1/3) * a^2 * b^m) (λ a b => (-1/2) * a^n * b^4)) : 
  m = 4 ∧ n = 2 := by
  sorry

end like_terms_imply_exponents_l2857_285763


namespace cement_mixture_weight_l2857_285792

/-- Proves that a cement mixture with given proportions weighs 48 pounds -/
theorem cement_mixture_weight (sand_fraction : ℚ) (water_fraction : ℚ) (gravel_weight : ℚ) :
  sand_fraction = 1/3 →
  water_fraction = 1/2 →
  gravel_weight = 8 →
  sand_fraction + water_fraction + gravel_weight / (sand_fraction + water_fraction + gravel_weight) = 1 →
  sand_fraction + water_fraction + gravel_weight = 48 := by
  sorry

end cement_mixture_weight_l2857_285792


namespace sum_of_real_solutions_l2857_285776

theorem sum_of_real_solutions (a : ℝ) (h : a > 1/2) :
  ∃ (x₁ x₂ : ℝ), 
    (Real.sqrt (3 * a - Real.sqrt (2 * a + x₁)) = x₁) ∧
    (Real.sqrt (3 * a - Real.sqrt (2 * a + x₂)) = x₂) ∧
    (x₁ + x₂ = Real.sqrt (3 * a + Real.sqrt (2 * a)) + Real.sqrt (3 * a - Real.sqrt (2 * a))) :=
by sorry

end sum_of_real_solutions_l2857_285776


namespace monitor_horizontal_length_l2857_285774

/-- Given a rectangle with a 16:9 aspect ratio and a diagonal of 32 inches,
    prove that the horizontal length is (16 * 32) / sqrt(337) --/
theorem monitor_horizontal_length (h w d : ℝ) : 
  h / w = 9 / 16 → 
  h^2 + w^2 = d^2 → 
  d = 32 → 
  w = (16 * 32) / Real.sqrt 337 := by
  sorry

end monitor_horizontal_length_l2857_285774


namespace inequality_solution_l2857_285700

def inequality (x : ℝ) : Prop :=
  2 / (x + 2) + 4 / (x + 8) ≥ 3 / 4

theorem inequality_solution :
  ∀ x : ℝ, inequality x ↔ (x > -2 ∧ x ≤ -8/3) ∨ x ≥ 4 :=
by sorry

end inequality_solution_l2857_285700


namespace cone_rolling_theorem_l2857_285732

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Predicate to check if a number is not divisible by the square of any prime -/
def notDivisibleBySquareOfPrime (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^2 ∣ n)

/-- The main theorem -/
theorem cone_rolling_theorem (cone : RightCircularCone) 
  (m n : ℕ) (h_sqrt : cone.h / cone.r = m * Real.sqrt n) 
  (h_prime : notDivisibleBySquareOfPrime n) 
  (h_rotations : (2 * Real.pi * Real.sqrt (cone.r^2 + cone.h^2)) = 50 * cone.r * Real.pi) :
  m + n = 50 := by sorry

end cone_rolling_theorem_l2857_285732


namespace clusters_per_spoonful_l2857_285746

/-- Represents the number of clusters of oats in a box of cereal -/
def clusters_per_box : ℕ := 500

/-- Represents the number of bowlfuls in a box of cereal -/
def bowlfuls_per_box : ℕ := 5

/-- Represents the number of spoonfuls in a bowl of cereal -/
def spoonfuls_per_bowl : ℕ := 25

/-- Theorem stating that the number of clusters of oats in each spoonful is 4 -/
theorem clusters_per_spoonful :
  clusters_per_box / (bowlfuls_per_box * spoonfuls_per_bowl) = 4 := by
  sorry

end clusters_per_spoonful_l2857_285746


namespace hawkeye_remaining_money_l2857_285758

/-- Calculates the remaining money after battery charging -/
def remaining_money (charge_cost : ℚ) (num_charges : ℕ) (budget : ℚ) : ℚ :=
  budget - charge_cost * num_charges

/-- Theorem: Given the specified conditions, the remaining money is $6 -/
theorem hawkeye_remaining_money :
  remaining_money 3.5 4 20 = 6 := by
  sorry

end hawkeye_remaining_money_l2857_285758


namespace sin_y_in_terms_of_c_and_d_l2857_285788

theorem sin_y_in_terms_of_c_and_d (c d y : ℝ) 
  (h1 : c > d) (h2 : d > 0) (h3 : 0 < y) (h4 : y < π / 2)
  (h5 : Real.tan y = (3 * c * d) / (c^2 - d^2)) :
  Real.sin y = (3 * c * d) / Real.sqrt (c^4 + 7 * c^2 * d^2 + d^4) := by
  sorry

end sin_y_in_terms_of_c_and_d_l2857_285788


namespace distribute_five_students_three_classes_l2857_285755

/-- The number of ways to distribute n students among k classes with a maximum of m students per class -/
def distributeStudents (n k m : ℕ) : ℕ := sorry

/-- Theorem: Distributing 5 students among 3 classes with at most 2 students per class yields 90 possibilities -/
theorem distribute_five_students_three_classes : distributeStudents 5 3 2 = 90 := by sorry

end distribute_five_students_three_classes_l2857_285755


namespace cube_difference_fifty_l2857_285712

/-- The sum of cubes of the first n positive integers -/
def sumOfPositiveCubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

/-- The sum of cubes of the first n negative integers -/
def sumOfNegativeCubes (n : ℕ) : ℤ := -(sumOfPositiveCubes n)

/-- The difference between the sum of cubes of the first n positive integers
    and the sum of cubes of the first n negative integers -/
def cubeDifference (n : ℕ) : ℤ := (sumOfPositiveCubes n : ℤ) - sumOfNegativeCubes n

theorem cube_difference_fifty : cubeDifference 50 = 3251250 := by sorry

end cube_difference_fifty_l2857_285712


namespace weighted_mean_car_sales_approx_l2857_285799

/-- Represents the car sales data for a week -/
structure CarSalesWeek where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  tuesday_discount : ℚ
  wednesday_commission : ℚ
  friday_discount : ℚ
  saturday_commission : ℚ

/-- Calculates the weighted mean of car sales for a week -/
def weightedMeanCarSales (sales : CarSalesWeek) : ℚ :=
  let monday_weighted := sales.monday
  let tuesday_weighted := sales.tuesday * (1 - sales.tuesday_discount)
  let wednesday_weighted := sales.wednesday * (1 + sales.wednesday_commission)
  let thursday_weighted := sales.thursday
  let friday_weighted := sales.friday * (1 - sales.friday_discount)
  let saturday_weighted := sales.saturday * (1 + sales.saturday_commission)
  let total_weighted := monday_weighted + tuesday_weighted + wednesday_weighted + 
                        thursday_weighted + friday_weighted + saturday_weighted
  total_weighted / 6

/-- Theorem: The weighted mean of car sales for the given week is approximately 5.48 -/
theorem weighted_mean_car_sales_approx (sales : CarSalesWeek) 
  (h1 : sales.monday = 8)
  (h2 : sales.tuesday = 3)
  (h3 : sales.wednesday = 10)
  (h4 : sales.thursday = 4)
  (h5 : sales.friday = 4)
  (h6 : sales.saturday = 4)
  (h7 : sales.tuesday_discount = 1/10)
  (h8 : sales.wednesday_commission = 1/20)
  (h9 : sales.friday_discount = 3/20)
  (h10 : sales.saturday_commission = 7/100) :
  ∃ ε > 0, |weightedMeanCarSales sales - 548/100| < ε :=
sorry


end weighted_mean_car_sales_approx_l2857_285799


namespace birthday_candles_ratio_l2857_285781

theorem birthday_candles_ratio (ambika_candles : ℕ) (total_candles : ℕ) : 
  ambika_candles = 4 → total_candles = 14 → 
  ∃ (aniyah_ratio : ℚ), aniyah_ratio = 2.5 ∧ 
  ambika_candles * (1 + aniyah_ratio) = total_candles :=
sorry

end birthday_candles_ratio_l2857_285781


namespace number_problem_l2857_285716

theorem number_problem (x : ℝ) : (0.2 * x = 0.2 * 650 + 190) → x = 1600 := by
  sorry

end number_problem_l2857_285716


namespace two_heads_probability_l2857_285704

/-- The probability of getting heads on a single fair coin toss -/
def prob_heads : ℚ := 1 / 2

/-- The probability of getting two heads when tossing two fair coins simultaneously -/
def prob_two_heads : ℚ := prob_heads * prob_heads

/-- Theorem: The probability of getting two heads when tossing two fair coins simultaneously is 1/4 -/
theorem two_heads_probability : prob_two_heads = 1 / 4 := by
  sorry

end two_heads_probability_l2857_285704


namespace tuesday_rainfall_l2857_285767

/-- Rainfall problem -/
theorem tuesday_rainfall (total_rainfall average_rainfall : ℝ) 
  (h1 : total_rainfall = 7 * average_rainfall)
  (h2 : average_rainfall = 3)
  (h3 : ∃ tuesday_rainfall : ℝ, 
    tuesday_rainfall = total_rainfall - tuesday_rainfall) :
  ∃ tuesday_rainfall : ℝ, tuesday_rainfall = 10.5 := by
  sorry

end tuesday_rainfall_l2857_285767


namespace number_divisible_by_5_power_1000_without_zero_digit_l2857_285768

theorem number_divisible_by_5_power_1000_without_zero_digit :
  ∃ n : ℕ, (5^1000 ∣ n) ∧ (∀ d : ℕ, d < 10 → (n.digits 10).all (λ x => x ≠ 0)) := by
  sorry

end number_divisible_by_5_power_1000_without_zero_digit_l2857_285768


namespace geometric_sequence_sum_l2857_285753

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

theorem geometric_sequence_sum (a : ℕ → ℝ) (n : ℕ) :
  (∀ k, a k > 0) →
  geometric_sequence a →
  a 2 = 3 →
  a 1 + a 3 = 10 →
  (∃ S_n : ℝ, S_n = (27/2) - (1/2) * 3^(n-3) ∨ S_n = (3^n - 1) / 2) :=
sorry

end geometric_sequence_sum_l2857_285753


namespace mod_nine_power_four_l2857_285706

theorem mod_nine_power_four (m : ℕ) : 
  14^4 % 9 = m ∧ 0 ≤ m ∧ m < 9 → m = 5 := by
  sorry

end mod_nine_power_four_l2857_285706


namespace factorial_simplification_l2857_285796

theorem factorial_simplification : (12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / 
  ((10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) + 3 * (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) = 4 / 3 := by
  sorry

end factorial_simplification_l2857_285796


namespace license_plate_difference_l2857_285780

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of possible license plates for Georgia (LLDLLL format). -/
def georgia_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible license plates for Nebraska (LLDDDDD format). -/
def nebraska_plates : ℕ := num_letters^2 * num_digits^5

/-- The difference between the number of possible license plates for Nebraska and Georgia. -/
theorem license_plate_difference : nebraska_plates - georgia_plates = 21902400 := by
  sorry

end license_plate_difference_l2857_285780


namespace rectangle_construction_l2857_285723

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Checks if four points form a rectangle -/
def isRectangle (r : Rectangle) : Prop := sorry

/-- Calculates the aspect ratio of a rectangle -/
def aspectRatio (r : Rectangle) : ℝ := sorry

/-- Checks if a point lies on a line segment between two other points -/
def onSegment (p q r : Point) : Prop := sorry

/-- Theorem: A rectangle with a given aspect ratio can be constructed
    given one point on each of its sides -/
theorem rectangle_construction
  (a : ℝ)
  (A B C D : Point)
  (h_a : a > 0) :
  ∃ (r : Rectangle),
    isRectangle r ∧
    aspectRatio r = a ∧
    onSegment r.P A r.Q ∧
    onSegment r.Q B r.R ∧
    onSegment r.R C r.S ∧
    onSegment r.S D r.P :=
by sorry

end rectangle_construction_l2857_285723


namespace quadratic_inequality_range_l2857_285783

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) := by
sorry

end quadratic_inequality_range_l2857_285783


namespace min_value_theorem_l2857_285724

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 8) 
  (h2 : t * u * v * w = 27) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 96 :=
sorry

end min_value_theorem_l2857_285724


namespace ceiling_equality_iff_x_in_range_l2857_285777

theorem ceiling_equality_iff_x_in_range (x : ℝ) : 
  ⌈⌈3*x⌉ + 1/2⌉ = ⌈x - 2⌉ ↔ x ∈ Set.Icc (-1) (-2/3) :=
sorry

end ceiling_equality_iff_x_in_range_l2857_285777


namespace radio_cost_price_l2857_285752

/-- Proves that the cost price of a radio is 2400 given the selling price and loss percentage --/
theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) : 
  selling_price = 2100 → loss_percentage = 12.5 → 
  ∃ (cost_price : ℝ), cost_price = 2400 ∧ selling_price = cost_price * (1 - loss_percentage / 100) :=
by
  sorry

#check radio_cost_price

end radio_cost_price_l2857_285752


namespace smallest_flock_size_l2857_285749

theorem smallest_flock_size (total_sparrows : ℕ) (parrot_flock_size : ℕ) : 
  total_sparrows = 182 →
  parrot_flock_size = 14 →
  ∃ (P : ℕ), total_sparrows = parrot_flock_size * P →
  (∀ (S : ℕ), S > 0 ∧ S ∣ total_sparrows ∧ (∃ (Q : ℕ), S ∣ (parrot_flock_size * Q)) → S ≥ 14) ∧
  14 ∣ total_sparrows ∧ (∃ (R : ℕ), 14 ∣ (parrot_flock_size * R)) :=
by sorry

#check smallest_flock_size

end smallest_flock_size_l2857_285749


namespace five_thousand_five_hundred_scientific_notation_l2857_285793

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem five_thousand_five_hundred_scientific_notation :
  toScientificNotation 5500 = ScientificNotation.mk 5.5 3 (by norm_num) :=
sorry

end five_thousand_five_hundred_scientific_notation_l2857_285793


namespace f_inequality_range_l2857_285794

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

theorem f_inequality_range :
  ∀ x : ℝ, (f x + f (x - 1/2) > 1) ↔ (x > -1/4) :=
by sorry

end f_inequality_range_l2857_285794


namespace total_flowers_l2857_285714

def roses : ℕ := 5
def lilies : ℕ := 2

theorem total_flowers : roses + lilies = 7 := by
  sorry

end total_flowers_l2857_285714


namespace max_digits_after_subtraction_l2857_285784

theorem max_digits_after_subtraction :
  ∀ (a b c : ℕ),
  10000 ≤ a ∧ a ≤ 99999 →
  1000 ≤ b ∧ b ≤ 9999 →
  0 ≤ c ∧ c ≤ 9 →
  (Nat.digits 10 (a * b - c)).length ≤ 9 ∧
  ∃ (x y z : ℕ),
    10000 ≤ x ∧ x ≤ 99999 ∧
    1000 ≤ y ∧ y ≤ 9999 ∧
    0 ≤ z ∧ z ≤ 9 ∧
    (Nat.digits 10 (x * y - z)).length = 9 :=
by sorry

end max_digits_after_subtraction_l2857_285784


namespace sum_of_squares_of_roots_l2857_285720

theorem sum_of_squares_of_roots (a : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, (x₁^4 + a*x₁^2 - 2017 = 0) ∧ 
                      (x₂^4 + a*x₂^2 - 2017 = 0) ∧ 
                      (x₃^4 + a*x₃^2 - 2017 = 0) ∧ 
                      (x₄^4 + a*x₄^2 - 2017 = 0) ∧ 
                      (x₁^2 + x₂^2 + x₃^2 + x₄^2 = 4)) → 
  a = 1006.5 :=
by sorry

end sum_of_squares_of_roots_l2857_285720


namespace mike_current_salary_l2857_285797

def mike_salary_five_months_ago : ℕ := 10000
def fred_salary_five_months_ago : ℕ := 1000
def salary_increase_percentage : ℕ := 40

theorem mike_current_salary :
  let total_salary_five_months_ago := mike_salary_five_months_ago + fred_salary_five_months_ago
  let salary_increase := (salary_increase_percentage * total_salary_five_months_ago) / 100
  mike_salary_five_months_ago + salary_increase = 15400 := by
  sorry

end mike_current_salary_l2857_285797


namespace range_of_m_l2857_285717

/-- The proposition p: x^2 - 8x - 20 ≤ 0 -/
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

/-- The proposition q: [x-(1+m)][x-(1-m)] ≤ 0 -/
def q (x m : ℝ) : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0

/-- p is a sufficient condition for q -/
def p_sufficient_for_q (m : ℝ) : Prop :=
  ∀ x, p x → q x m

/-- p is not a necessary condition for q -/
def p_not_necessary_for_q (m : ℝ) : Prop :=
  ∃ x, q x m ∧ ¬(p x)

/-- m is positive -/
def m_positive (m : ℝ) : Prop := m > 0

theorem range_of_m :
  ∀ m : ℝ, (m_positive m ∧ p_sufficient_for_q m ∧ p_not_necessary_for_q m) ↔ m ≥ 9 :=
sorry

end range_of_m_l2857_285717


namespace gcf_18_30_l2857_285786

theorem gcf_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcf_18_30_l2857_285786


namespace chess_draw_probability_l2857_285701

theorem chess_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.3)
  (h_not_lose : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end chess_draw_probability_l2857_285701


namespace parabola_sum_l2857_285762

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_sum (p : Parabola) 
  (vertex_condition : p.y_at 1 = 3 ∧ (- p.b / (2 * p.a)) = 1)
  (point_condition : p.y_at 0 = 2) :
  p.a + p.b + p.c = 3 := by
  sorry

end parabola_sum_l2857_285762


namespace mark_and_carolyn_money_sum_l2857_285705

theorem mark_and_carolyn_money_sum : (5 : ℚ) / 8 + (7 : ℚ) / 20 = 0.975 := by
  sorry

end mark_and_carolyn_money_sum_l2857_285705


namespace line_equation_midpoint_line_equation_max_distance_l2857_285702

/-- A line passing through point M(1, 2) and intersecting the x-axis and y-axis -/
structure Line where
  m : ℝ × ℝ := (1, 2)
  intersects_x_axis : ℝ → Prop
  intersects_y_axis : ℝ → Prop

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : LineEquation) : ℝ := sorry

theorem line_equation_midpoint (l : Line) : 
  (∃ p q : ℝ × ℝ, l.intersects_x_axis p.1 ∧ l.intersects_y_axis q.2 ∧ 
   l.m = ((p.1 + q.1) / 2, (p.2 + q.2) / 2)) → 
  ∃ eq : LineEquation, eq.a = 2 ∧ eq.b = 1 ∧ eq.c = -4 :=
sorry

theorem line_equation_max_distance (l : Line) :
  (∀ eq : LineEquation, distance_point_to_line (0, 0) eq ≤ 
   distance_point_to_line (0, 0) ⟨1, 2, -5⟩) →
  ∃ eq : LineEquation, eq.a = 1 ∧ eq.b = 2 ∧ eq.c = -5 :=
sorry

end line_equation_midpoint_line_equation_max_distance_l2857_285702


namespace square_region_perimeter_l2857_285739

theorem square_region_perimeter (total_area : ℝ) (num_squares : ℕ) (h1 : total_area = 144) (h2 : num_squares = 4) :
  let square_area : ℝ := total_area / num_squares
  let side_length : ℝ := Real.sqrt square_area
  let perimeter : ℝ := 2 * side_length * num_squares
  perimeter = 48 := by
  sorry

end square_region_perimeter_l2857_285739


namespace quadratic_even_iff_b_zero_l2857_285722

/-- A quadratic function f(x) = ax² + bx + c is even if and only if b = 0 -/
theorem quadratic_even_iff_b_zero (a b c : ℝ) :
  (∀ x, (a * x^2 + b * x + c) = (a * (-x)^2 + b * (-x) + c)) ↔ b = 0 := by
  sorry

end quadratic_even_iff_b_zero_l2857_285722


namespace sum_of_lg2_and_lg5_power_of_8_two_thirds_l2857_285791

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem 1: lg2 + lg5 = 1
theorem sum_of_lg2_and_lg5 : lg 2 + lg 5 = 1 := by sorry

-- Theorem 2: 8^(2/3) = 4
theorem power_of_8_two_thirds : (8 : ℝ) ^ (2/3) = 4 := by sorry

end sum_of_lg2_and_lg5_power_of_8_two_thirds_l2857_285791


namespace hyperbolas_same_asymptotes_l2857_285725

/-- Two hyperbolas have the same asymptotes if and only if M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) ↔ M = 225 / 16 :=
by sorry

end hyperbolas_same_asymptotes_l2857_285725


namespace cricket_team_right_handed_players_l2857_285756

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 58) 
  (h2 : throwers = 37) 
  (h3 : throwers ≤ total_players) 
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures one-third of non-throwers can be left-handed
  : (throwers + ((total_players - throwers) - (total_players - throwers) / 3)) = 51 := by
  sorry

end cricket_team_right_handed_players_l2857_285756


namespace sqrt_equation_solution_l2857_285744

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (5 * x + 9) = 11 → x = 112 / 5 := by
  sorry

end sqrt_equation_solution_l2857_285744


namespace one_absent_out_of_three_l2857_285773

def probability_absent : ℚ := 1 / 40

def probability_present : ℚ := 1 - probability_absent

def probability_one_absent_two_present : ℚ :=
  3 * probability_absent * probability_present * probability_present

theorem one_absent_out_of_three (ε : ℚ) (h : ε > 0) :
  |probability_one_absent_two_present - 4563 / 64000| < ε :=
sorry

end one_absent_out_of_three_l2857_285773


namespace solution_range_l2857_285766

-- Define the equation
def equation (m x : ℝ) : Prop :=
  m / (x - 2) + 1 = x / (2 - x)

-- Define the theorem
theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ equation m x) ↔ (m ≤ 2 ∧ m ≠ -2) := by
  sorry

end solution_range_l2857_285766


namespace max_value_of_expression_l2857_285769

theorem max_value_of_expression (x : ℝ) :
  ∃ (max : ℝ), max = (1 / 4 : ℝ) ∧ ∀ y : ℝ, 10^y - 100^y ≤ max :=
sorry

end max_value_of_expression_l2857_285769


namespace weird_calculator_theorem_l2857_285770

/-- Represents the calculator operations -/
inductive Operation
| DSharp : Operation  -- doubles and adds 1
| DFlat  : Operation  -- doubles and subtracts 1

/-- Applies a single operation to a number -/
def apply_operation (op : Operation) (x : ℕ) : ℕ :=
  match op with
  | Operation.DSharp => 2 * x + 1
  | Operation.DFlat  => 2 * x - 1

/-- Applies a sequence of operations to a number -/
def apply_sequence (ops : List Operation) (x : ℕ) : ℕ :=
  match ops with
  | [] => x
  | op :: rest => apply_sequence rest (apply_operation op x)

/-- The set of all possible results after 8 operations starting from 1 -/
def possible_results : Set ℕ :=
  {n | ∃ (ops : List Operation), ops.length = 8 ∧ apply_sequence ops 1 = n}

theorem weird_calculator_theorem :
  possible_results = {n : ℕ | n < 512 ∧ n % 2 = 1} :=
sorry

end weird_calculator_theorem_l2857_285770


namespace probability_both_colors_drawn_l2857_285731

def total_balls : ℕ := 16
def black_balls : ℕ := 10
def white_balls : ℕ := 6
def drawn_balls : ℕ := 3

theorem probability_both_colors_drawn : 
  (1 : ℚ) - (Nat.choose black_balls drawn_balls + Nat.choose white_balls drawn_balls : ℚ) / 
  (Nat.choose total_balls drawn_balls : ℚ) = 3/4 :=
sorry

end probability_both_colors_drawn_l2857_285731


namespace cos_sin_sum_l2857_285747

theorem cos_sin_sum (φ : Real) (h : Real.cos (π / 2 + φ) = Real.sqrt 3 / 2) :
  Real.cos (3 * π / 2 - φ) + Real.sin (φ - π) = Real.sqrt 3 := by
  sorry

end cos_sin_sum_l2857_285747


namespace certain_number_proof_l2857_285761

theorem certain_number_proof (a b x : ℝ) (h1 : x * a = 3 * b) (h2 : a * b ≠ 0) (h3 : a / 3 = b / 2) : x = 2 := by
  sorry

end certain_number_proof_l2857_285761


namespace other_x_intercept_of_quadratic_l2857_285726

/-- Given a quadratic function with vertex (5, 12) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 12 - a * (x - 5)^2) →  -- vertex form
  a * 1^2 + b * 1 + c = 0 →                          -- x-intercept at (1, 0)
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9 :=    -- other x-intercept at 9
by sorry

end other_x_intercept_of_quadratic_l2857_285726


namespace positive_reals_inequality_arithmetic_geometric_mean_inequality_l2857_285760

theorem positive_reals_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end positive_reals_inequality_arithmetic_geometric_mean_inequality_l2857_285760


namespace inequality_always_holds_l2857_285719

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : a * |c| ≥ b * |c| := by
  sorry

end inequality_always_holds_l2857_285719


namespace angle_WYZ_measure_l2857_285765

-- Define the angles
def angle_XYZ : ℝ := 40
def angle_XYW : ℝ := 15

-- Define the theorem
theorem angle_WYZ_measure :
  let angle_WYZ := angle_XYZ - angle_XYW
  angle_WYZ = 25 := by sorry

end angle_WYZ_measure_l2857_285765


namespace square_difference_divided_problem_solution_l2857_285759

theorem square_difference_divided (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (125^2 - 105^2) / 20 = 230 := by
  have h : 125 > 105 := by sorry
  have key := square_difference_divided 125 105 h
  sorry

end square_difference_divided_problem_solution_l2857_285759


namespace tetrahedron_height_formula_l2857_285743

/-- Configuration of four mutually tangent spheres -/
structure SpheresConfiguration where
  small_radius : ℝ
  large_radius : ℝ
  small_spheres_count : ℕ
  on_flat_floor : Prop

/-- Tetrahedron circumscribing the spheres configuration -/
def circumscribing_tetrahedron (config : SpheresConfiguration) : Prop :=
  sorry

/-- Height of the tetrahedron from the floor to the opposite vertex -/
noncomputable def tetrahedron_height (config : SpheresConfiguration) : ℝ :=
  sorry

/-- Theorem stating the height of the tetrahedron -/
theorem tetrahedron_height_formula (config : SpheresConfiguration) 
  (h1 : config.small_radius = 2)
  (h2 : config.large_radius = 3)
  (h3 : config.small_spheres_count = 3)
  (h4 : config.on_flat_floor)
  (h5 : circumscribing_tetrahedron config) :
  tetrahedron_height config = (Real.sqrt 177 + 9 * Real.sqrt 3) / 3 :=
sorry

end tetrahedron_height_formula_l2857_285743


namespace bounded_area_calculation_l2857_285736

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bounded by two circles and two vertical lines -/
def boundedArea (c1 c2 : Circle) (x1 x2 : ℝ) : ℝ :=
  sorry

theorem bounded_area_calculation :
  let c1 : Circle := { center := (4, 4), radius := 4 }
  let c2 : Circle := { center := (12, 12), radius := 4 }
  let x1 : ℝ := 4
  let x2 : ℝ := 12
  boundedArea c1 c2 x1 x2 = 64 - 8 * Real.pi := by
  sorry

end bounded_area_calculation_l2857_285736


namespace star_perimeter_sum_l2857_285703

theorem star_perimeter_sum (X Y Z : ℕ) : 
  Prime X → Prime Y → Prime Z →
  X < Z → Z < Y → X + Y < 2 * Z →
  X + Y + Z ≥ 20 := by
sorry

end star_perimeter_sum_l2857_285703


namespace paul_initial_stock_l2857_285795

/-- The number of pencils Paul makes in a day -/
def daily_production : ℕ := 100

/-- The number of days Paul works in a week -/
def working_days : ℕ := 5

/-- The number of pencils Paul sold during the week -/
def pencils_sold : ℕ := 350

/-- The number of pencils in stock at the end of the week -/
def end_stock : ℕ := 230

/-- The number of pencils Paul had at the beginning of the week -/
def initial_stock : ℕ := daily_production * working_days + end_stock - pencils_sold

theorem paul_initial_stock :
  initial_stock = 380 :=
sorry

end paul_initial_stock_l2857_285795


namespace flag_height_l2857_285713

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- The three fabric squares Bobby has -/
def fabric1 : Rectangle := ⟨8, 5⟩
def fabric2 : Rectangle := ⟨10, 7⟩
def fabric3 : Rectangle := ⟨5, 5⟩

/-- The desired length of the flag -/
def flagLength : ℝ := 15

/-- Theorem stating that the height of the flag will be 9 feet -/
theorem flag_height :
  (area fabric1 + area fabric2 + area fabric3) / flagLength = 9 := by
  sorry

end flag_height_l2857_285713


namespace percent_relation_l2857_285785

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) : 
  (4 * b) / a = 10/3 := by sorry

end percent_relation_l2857_285785


namespace work_completion_time_l2857_285728

theorem work_completion_time (a_time b_time initial_days : ℝ) 
  (ha : a_time = 12)
  (hb : b_time = 6)
  (hi : initial_days = 3) :
  let a_rate := 1 / a_time
  let b_rate := 1 / b_time
  let initial_work := a_rate * initial_days
  let remaining_work := 1 - initial_work
  let combined_rate := a_rate + b_rate
  (remaining_work / combined_rate) = 3 := by
sorry

end work_completion_time_l2857_285728


namespace important_rectangle_difference_l2857_285745

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (isBlack : Nat → Nat → Bool)

/-- Represents a rectangle on the chessboard -/
structure Rectangle :=
  (top : Nat)
  (left : Nat)
  (bottom : Nat)
  (right : Nat)

/-- Checks if a rectangle is important -/
def isImportantRectangle (board : Chessboard) (rect : Rectangle) : Bool :=
  board.isBlack rect.top rect.left &&
  board.isBlack rect.top rect.right &&
  board.isBlack rect.bottom rect.left &&
  board.isBlack rect.bottom rect.right

/-- Counts the number of important rectangles containing a square -/
def countImportantRectangles (board : Chessboard) (row : Nat) (col : Nat) : Nat :=
  sorry

/-- Sums the counts for all squares of a given color -/
def sumCounts (board : Chessboard) (isBlack : Bool) : Nat :=
  sorry

/-- The main theorem -/
theorem important_rectangle_difference (board : Chessboard) :
  board.size = 8 →
  (∀ i j, board.isBlack i j = ((i + j) % 2 = 0)) →
  (sumCounts board true) - (sumCounts board false) = 36 :=
sorry

end important_rectangle_difference_l2857_285745


namespace gain_percent_example_l2857_285798

/-- Calculates the gain percent given the cost price and selling price -/
def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that the gain percent is 50% when an article is bought for $10 and sold for $15 -/
theorem gain_percent_example : gain_percent 10 15 = 50 := by
  sorry

end gain_percent_example_l2857_285798


namespace corn_preference_percentage_l2857_285711

theorem corn_preference_percentage (peas carrots corn : ℕ) : 
  peas = 6 → carrots = 9 → corn = 5 → 
  (corn : ℚ) / (peas + carrots + corn : ℚ) * 100 = 25 := by
  sorry

end corn_preference_percentage_l2857_285711


namespace similar_triangles_problem_l2857_285729

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- Given two similar triangles satisfying certain conditions, 
    prove that the corresponding side of the larger triangle is 12 feet -/
theorem similar_triangles_problem 
  (small large : Triangle)
  (area_diff : large.area - small.area = 72)
  (area_ratio : ∃ k : ℕ, large.area / small.area = k^2)
  (small_area_int : ∃ n : ℕ, small.area = n)
  (small_side : small.side = 6)
  : large.side = 12 := by
  sorry

end similar_triangles_problem_l2857_285729


namespace flowers_per_vase_is_nine_l2857_285787

/-- The number of carnations -/
def carnations : ℕ := 4

/-- The number of roses -/
def roses : ℕ := 23

/-- The total number of vases needed -/
def vases : ℕ := 3

/-- The total number of flowers -/
def total_flowers : ℕ := carnations + roses

/-- The number of flowers one vase can hold -/
def flowers_per_vase : ℕ := total_flowers / vases

theorem flowers_per_vase_is_nine : flowers_per_vase = 9 := by
  sorry

end flowers_per_vase_is_nine_l2857_285787


namespace complex_equality_l2857_285738

theorem complex_equality (u v : ℂ) 
  (h1 : 3 * Complex.abs (u + 1) * Complex.abs (v + 1) ≥ Complex.abs (u * v + 5 * u + 5 * v + 1))
  (h2 : Complex.abs (u + v) = Complex.abs (u * v + 1)) :
  u = 1 ∨ v = 1 := by
sorry

end complex_equality_l2857_285738


namespace sack_lunch_cost_l2857_285751

/-- The cost of each sack lunch for a field trip -/
theorem sack_lunch_cost (num_children : ℕ) (num_chaperones : ℕ) (num_teachers : ℕ) (num_additional : ℕ) (total_cost : ℚ) : 
  num_children = 35 →
  num_chaperones = 5 →
  num_teachers = 1 →
  num_additional = 3 →
  total_cost = 308 →
  total_cost / (num_children + num_chaperones + num_teachers + num_additional) = 7 := by
sorry

end sack_lunch_cost_l2857_285751


namespace part_one_part_two_l2857_285778

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

-- Part 1
theorem part_one (a : ℝ) : 
  (∀ x, f x a - |x - a| ≤ 2 ↔ x ∈ Set.Icc (-5) (-1)) → a = 2 := by sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∃ x₀ : ℝ, f x₀ 2 < 4*m + m^2) → m < -5 ∨ m > 1 := by sorry

end part_one_part_two_l2857_285778


namespace complex_sum_magnitude_l2857_285748

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 1) :
  Complex.abs (a + b + c) = 1 ∨ Complex.abs (a + b + c) = 3 := by
  sorry

end complex_sum_magnitude_l2857_285748


namespace sum_of_intercepts_l2857_285750

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3 * x - 2 * y - 6 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -3

/-- Theorem: The sum of the x-intercept and y-intercept of the line 3x - 2y - 6 = 0 is -1 -/
theorem sum_of_intercepts :
  line_equation x_intercept 0 ∧ 
  line_equation 0 y_intercept ∧ 
  x_intercept + y_intercept = -1 := by
  sorry

end sum_of_intercepts_l2857_285750


namespace javier_first_throw_distance_l2857_285775

/-- Represents the distances of Javier's three javelin throws -/
structure JavelinThrows where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Theorem stating the distance of Javier's first throw given the conditions -/
theorem javier_first_throw_distance (throws : JavelinThrows) :
  throws.first = 2 * throws.second ∧
  throws.first = throws.third / 2 ∧
  throws.first + throws.second + throws.third = 1050 →
  throws.first = 300 := by
  sorry

end javier_first_throw_distance_l2857_285775


namespace simplify_expression_l2857_285730

theorem simplify_expression : (7^5 + 2^8) * (2^3 - (-2)^3)^7 = 0 := by
  sorry

end simplify_expression_l2857_285730


namespace puzzle_pieces_count_l2857_285733

theorem puzzle_pieces_count (pieces_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) 
  (num_500_piece_puzzles : ℕ) (num_unknown_piece_puzzles : ℕ) :
  pieces_per_hour = 100 →
  hours_per_day = 7 →
  days = 7 →
  num_500_piece_puzzles = 5 →
  num_unknown_piece_puzzles = 8 →
  (pieces_per_hour * hours_per_day * days - num_500_piece_puzzles * 500) / num_unknown_piece_puzzles = 300 := by
  sorry

end puzzle_pieces_count_l2857_285733


namespace statement_is_valid_assignment_l2857_285764

/-- Represents a variable in an assignment statement -/
structure Variable where
  name : String

/-- Represents an expression in an assignment statement -/
inductive Expression where
  | Var : Variable → Expression
  | Const : ℕ → Expression
  | Add : Expression → Expression → Expression

/-- Represents an assignment statement -/
structure AssignmentStatement where
  lhs : Variable
  rhs : Expression

/-- Checks if a given statement is a valid assignment statement -/
def isValidAssignmentStatement (stmt : AssignmentStatement) : Prop :=
  ∃ (v : Variable) (e : Expression), stmt.lhs = v ∧ stmt.rhs = e

/-- The statement "S = a + 1" -/
def statement : AssignmentStatement :=
  { lhs := ⟨"S"⟩,
    rhs := Expression.Add (Expression.Var ⟨"a"⟩) (Expression.Const 1) }

/-- Theorem: The statement "S = a + 1" is a valid assignment statement -/
theorem statement_is_valid_assignment : isValidAssignmentStatement statement := by
  sorry


end statement_is_valid_assignment_l2857_285764


namespace square_of_r_minus_three_l2857_285782

theorem square_of_r_minus_three (r : ℝ) (h : r^2 - 6*r + 5 = 0) : (r - 3)^2 = 4 := by
  sorry

end square_of_r_minus_three_l2857_285782


namespace line_slope_l2857_285772

/-- The slope of a line given by the equation 3y + 4x = 12 is -4/3 -/
theorem line_slope (x y : ℝ) : 3 * y + 4 * x = 12 → (y - 4) / (x - 0) = -4/3 := by
  sorry

end line_slope_l2857_285772


namespace equation_solution_l2857_285737

theorem equation_solution : ∃! (x y : ℝ), 3*x^2 + 14*y^2 - 12*x*y + 6*x - 20*y + 11 = 0 := by
  sorry

end equation_solution_l2857_285737


namespace min_value_fraction_sum_l2857_285757

theorem min_value_fraction_sum (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : b + c ≥ a + d) : 
  b / (c + d) + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end min_value_fraction_sum_l2857_285757


namespace system_solution_l2857_285710

theorem system_solution (x y : ℝ) : 
  (x^2 + y^2 ≤ 1 ∧ 
   16 * x^4 - 8 * x^2 * y^2 + y^4 - 40 * x^2 - 10 * y^2 + 25 = 0) ↔ 
  ((x = -2 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) ∨
   (x = -2 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
   (x = 2 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) ∨
   (x = 2 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5)) :=
by sorry

end system_solution_l2857_285710


namespace unique_solution_lcm_gcd_equation_l2857_285790

theorem unique_solution_lcm_gcd_equation : 
  ∃! n : ℕ+, Nat.lcm n 120 = Nat.gcd n 120 + 300 ∧ n = 180 := by sorry

end unique_solution_lcm_gcd_equation_l2857_285790


namespace tan_135_degrees_l2857_285771

theorem tan_135_degrees : Real.tan (135 * π / 180) = -1 := by
  sorry

end tan_135_degrees_l2857_285771
