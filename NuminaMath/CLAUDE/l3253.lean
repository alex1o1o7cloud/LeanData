import Mathlib

namespace equation_has_real_solution_l3253_325304

theorem equation_has_real_solution (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  ∃ x : ℝ, (a * b^x)^(x + 1) = c := by
  sorry

end equation_has_real_solution_l3253_325304


namespace july_rainfall_l3253_325344

theorem july_rainfall (march april may june : ℝ) (h1 : march = 3.79) (h2 : april = 4.5) 
  (h3 : may = 3.95) (h4 : june = 3.09) (h5 : (march + april + may + june + july) / 5 = 4) : 
  july = 4.67 := by
  sorry

end july_rainfall_l3253_325344


namespace common_factor_of_polynomials_l3253_325302

theorem common_factor_of_polynomials (a b : ℝ) :
  ∃ (k₁ k₂ : ℝ), (4 * a^2 - 2 * a * b = (2 * a - b) * k₁) ∧
                 (4 * a^2 - b^2 = (2 * a - b) * k₂) := by
  sorry

end common_factor_of_polynomials_l3253_325302


namespace three_intersection_points_k_value_l3253_325356

/-- Curve C1 -/
def C1 (k : ℝ) (x y : ℝ) : Prop :=
  y = k * abs x + 2

/-- Curve C2 -/
def C2 (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 4

/-- Number of intersection points between C1 and C2 -/
def intersectionPoints (k : ℝ) : ℕ :=
  sorry -- This would require a complex implementation to count intersection points

theorem three_intersection_points_k_value :
  ∀ k : ℝ, intersectionPoints k = 3 → k = -4/3 :=
by sorry

end three_intersection_points_k_value_l3253_325356


namespace one_fourth_in_one_eighth_l3253_325300

theorem one_fourth_in_one_eighth :
  (1 / 8 : ℚ) / (1 / 4 : ℚ) = 1 / 2 := by
  sorry

end one_fourth_in_one_eighth_l3253_325300


namespace min_equal_triangles_is_18_l3253_325380

/-- A non-convex hexagon representing a chessboard with one corner square cut out. -/
structure CutoutChessboard :=
  (area : ℝ)
  (is_non_convex : Bool)

/-- The minimum number of equal triangles into which the cutout chessboard can be divided. -/
def min_equal_triangles (board : CutoutChessboard) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of equal triangles is 18 for a cutout chessboard with area 63. -/
theorem min_equal_triangles_is_18 (board : CutoutChessboard) 
  (h1 : board.area = 63)
  (h2 : board.is_non_convex = true) : 
  min_equal_triangles board = 18 :=
sorry

end min_equal_triangles_is_18_l3253_325380


namespace function_inequality_relation_l3253_325390

theorem function_inequality_relation (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 3 * x + 1) →
  a > 0 →
  b > 0 →
  (∀ x, |x - 1| < b → |f x - 4| < a) →
  a ≥ 3 * b :=
sorry

end function_inequality_relation_l3253_325390


namespace trees_per_sharpening_l3253_325337

def cost_per_sharpening : ℕ := 5
def total_sharpening_cost : ℕ := 35
def min_trees_chopped : ℕ := 91

theorem trees_per_sharpening :
  ∃ (x : ℕ), x > 0 ∧ 
    x * (total_sharpening_cost / cost_per_sharpening) ≥ min_trees_chopped ∧
    ∀ (y : ℕ), y > 0 → y * (total_sharpening_cost / cost_per_sharpening) ≥ min_trees_chopped → y ≥ x :=
by sorry

end trees_per_sharpening_l3253_325337


namespace upstream_downstream_time_ratio_l3253_325306

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 -/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 36) 
  (h2 : stream_speed = 12) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

#check upstream_downstream_time_ratio

end upstream_downstream_time_ratio_l3253_325306


namespace total_reimbursement_is_correct_l3253_325317

/-- Reimbursement rate for industrial clients on weekdays -/
def industrial_rate : ℚ := 36 / 100

/-- Reimbursement rate for commercial clients on weekdays -/
def commercial_rate : ℚ := 42 / 100

/-- Reimbursement rate for any clients on weekends -/
def weekend_rate : ℚ := 45 / 100

/-- Mileage for industrial clients on Monday -/
def monday_industrial : ℕ := 10

/-- Mileage for commercial clients on Monday -/
def monday_commercial : ℕ := 8

/-- Mileage for industrial clients on Tuesday -/
def tuesday_industrial : ℕ := 12

/-- Mileage for commercial clients on Tuesday -/
def tuesday_commercial : ℕ := 14

/-- Mileage for industrial clients on Wednesday -/
def wednesday_industrial : ℕ := 15

/-- Mileage for commercial clients on Wednesday -/
def wednesday_commercial : ℕ := 5

/-- Mileage for commercial clients on Thursday -/
def thursday_commercial : ℕ := 20

/-- Mileage for industrial clients on Friday -/
def friday_industrial : ℕ := 8

/-- Mileage for commercial clients on Friday -/
def friday_commercial : ℕ := 8

/-- Mileage for commercial clients on Saturday -/
def saturday_commercial : ℕ := 12

/-- Calculate the total reimbursement for the week -/
def total_reimbursement : ℚ :=
  industrial_rate * (monday_industrial + tuesday_industrial + wednesday_industrial + friday_industrial) +
  commercial_rate * (monday_commercial + tuesday_commercial + wednesday_commercial + thursday_commercial + friday_commercial) +
  weekend_rate * saturday_commercial

/-- Theorem stating that the total reimbursement is equal to $44.70 -/
theorem total_reimbursement_is_correct : total_reimbursement = 4470 / 100 := by
  sorry

end total_reimbursement_is_correct_l3253_325317


namespace sum_difference_odd_even_l3253_325351

theorem sum_difference_odd_even : 
  let range := Finset.Icc 372 506
  let odd_sum := (range.filter (λ n => n % 2 = 1)).sum id
  let even_sum := (range.filter (λ n => n % 2 = 0)).sum id
  odd_sum - even_sum = 439 := by sorry

end sum_difference_odd_even_l3253_325351


namespace point_in_second_quadrant_l3253_325362

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  is_in_second_quadrant (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end point_in_second_quadrant_l3253_325362


namespace tan_660_degrees_l3253_325369

theorem tan_660_degrees : Real.tan (660 * π / 180) = -Real.sqrt 3 := by
  sorry

end tan_660_degrees_l3253_325369


namespace meal_combinations_eq_sixty_l3253_325328

/-- The number of menu items in the restaurant -/
def total_menu_items : ℕ := 12

/-- The number of vegetarian dishes available -/
def vegetarian_dishes : ℕ := 5

/-- The number of different meal combinations for Elena and Nasir -/
def meal_combinations : ℕ := total_menu_items * vegetarian_dishes

/-- Theorem stating that the number of meal combinations is 60 -/
theorem meal_combinations_eq_sixty :
  meal_combinations = 60 := by sorry

end meal_combinations_eq_sixty_l3253_325328


namespace fraction_equation_solution_l3253_325303

theorem fraction_equation_solution :
  ∃! x : ℚ, (x - 4) / (x + 3) = (x + 2) / (x - 1) ∧ x = -1/5 := by
  sorry

end fraction_equation_solution_l3253_325303


namespace ratio_approximation_l3253_325398

def geometric_sum (n : ℕ) : ℚ :=
  (10^n - 1) / 9

def ratio (n : ℕ) : ℚ :=
  (10^n * 9) / (10^n - 1)

theorem ratio_approximation :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |ratio 8 - 9| < ε :=
sorry

end ratio_approximation_l3253_325398


namespace planes_perpendicular_parallel_l3253_325360

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_parallel 
  (a b : Line) (α β γ : Plane) 
  (h1 : perpendicular α γ) 
  (h2 : parallel β γ) : 
  perpendicular α β :=
sorry

end planes_perpendicular_parallel_l3253_325360


namespace watch_cost_price_l3253_325385

/-- The cost price of a watch satisfying certain conditions -/
theorem watch_cost_price : ∃ (cp : ℝ), 
  (cp > 0) ∧ 
  (0.80 * cp + 520 = 1.06 * cp) ∧ 
  (cp = 2000) := by
  sorry

end watch_cost_price_l3253_325385


namespace car_lot_total_l3253_325381

theorem car_lot_total (air_bags : ℕ) (power_windows : ℕ) (both : ℕ) (neither : ℕ)
  (h1 : air_bags = 45)
  (h2 : power_windows = 30)
  (h3 : both = 12)
  (h4 : neither = 2) :
  air_bags + power_windows - both + neither = 65 := by
  sorry

end car_lot_total_l3253_325381


namespace bankers_discount_example_l3253_325305

/-- Calculates the banker's discount given the face value and true discount of a bill. -/
def bankers_discount (face_value : ℚ) (true_discount : ℚ) : ℚ :=
  let present_value := face_value - true_discount
  (true_discount / present_value) * face_value

/-- Theorem stating that for a bill with face value 2660 and true discount 360,
    the banker's discount is approximately 416.35. -/
theorem bankers_discount_example :
  ∃ ε > 0, |bankers_discount 2660 360 - 416.35| < ε :=
by
  sorry

#eval bankers_discount 2660 360

end bankers_discount_example_l3253_325305


namespace smallest_k_satisfies_condition_no_smaller_k_satisfies_condition_l3253_325376

/-- The smallest positive real number k satisfying the given condition -/
def smallest_k : ℝ := 4

/-- Predicate to check if a quadratic equation has two distinct real roots -/
def has_distinct_real_roots (p q : ℝ) : Prop :=
  p^2 - 4*q > 0

/-- Predicate to check if four real numbers are distinct -/
def are_distinct (a b c d : ℝ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Main theorem stating that smallest_k satisfies the required condition -/
theorem smallest_k_satisfies_condition :
  ∀ a b c d : ℝ,
  are_distinct a b c d →
  a ≥ smallest_k → b ≥ smallest_k → c ≥ smallest_k → d ≥ smallest_k →
  ∃ p q r s : ℝ,
    ({p, q, r, s} : Set ℝ) = {a, b, c, d} ∧
    has_distinct_real_roots p q ∧
    has_distinct_real_roots r s ∧
    (∀ x : ℝ, (x^2 + p*x + q = 0 ∨ x^2 + r*x + s = 0) →
      (∀ y : ℝ, y ≠ x → (y^2 + p*y + q ≠ 0 ∧ y^2 + r*y + s ≠ 0))) :=
by sorry

/-- Theorem stating that no smaller positive real number than smallest_k satisfies the condition -/
theorem no_smaller_k_satisfies_condition :
  ∀ k : ℝ, 0 < k → k < smallest_k →
  ∃ a b c d : ℝ,
    are_distinct a b c d ∧
    a ≥ k ∧ b ≥ k ∧ c ≥ k ∧ d ≥ k ∧
    (∀ p q r s : ℝ,
      ({p, q, r, s} : Set ℝ) = {a, b, c, d} →
      ¬(has_distinct_real_roots p q ∧
        has_distinct_real_roots r s ∧
        (∀ x : ℝ, (x^2 + p*x + q = 0 ∨ x^2 + r*x + s = 0) →
          (∀ y : ℝ, y ≠ x → (y^2 + p*y + q ≠ 0 ∧ y^2 + r*y + s ≠ 0))))) :=
by sorry

end smallest_k_satisfies_condition_no_smaller_k_satisfies_condition_l3253_325376


namespace public_transport_support_percentage_l3253_325326

theorem public_transport_support_percentage
  (gov_employees : ℕ) (gov_support_rate : ℚ)
  (citizens : ℕ) (citizen_support_rate : ℚ) :
  gov_employees = 150 →
  gov_support_rate = 70 / 100 →
  citizens = 800 →
  citizen_support_rate = 60 / 100 →
  let total_surveyed := gov_employees + citizens
  let total_supporters := gov_employees * gov_support_rate + citizens * citizen_support_rate
  (total_supporters / total_surveyed : ℚ) = 6158 / 10000 := by
  sorry

end public_transport_support_percentage_l3253_325326


namespace crates_lost_l3253_325332

/-- Proves the number of crates lost given initial conditions --/
theorem crates_lost (initial_crates : ℕ) (total_cost : ℚ) (selling_price : ℚ) (profit_percentage : ℚ) : 
  initial_crates = 10 →
  total_cost = 160 →
  selling_price = 25 →
  profit_percentage = 25 / 100 →
  ∃ (lost_crates : ℕ), lost_crates = 2 ∧ 
    selling_price * (initial_crates - lost_crates) = total_cost * (1 + profit_percentage) :=
by sorry

end crates_lost_l3253_325332


namespace six_digit_divisibility_theorem_l3253_325339

/-- Represents a 6-digit number in the form 739ABC -/
def SixDigitNumber (a b c : Nat) : Nat :=
  739000 + 100 * a + 10 * b + c

/-- Checks if a number is divisible by 7, 8, and 9 -/
def isDivisibleBy789 (n : Nat) : Prop :=
  n % 7 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0

/-- The main theorem stating the possible values for A, B, and C -/
theorem six_digit_divisibility_theorem :
  ∀ a b c : Nat,
  a < 10 ∧ b < 10 ∧ c < 10 →
  isDivisibleBy789 (SixDigitNumber a b c) →
  (a = 3 ∧ b = 6 ∧ c = 8) ∨ (a = 8 ∧ b = 7 ∧ c = 2) :=
by sorry

end six_digit_divisibility_theorem_l3253_325339


namespace range_of_a_l3253_325329

open Set

def A : Set ℝ := {x | -5 < x ∧ x < 1}
def B : Set ℝ := {x | -2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : A ∩ B ⊆ C a) : a ≥ 1 := by
  sorry

end range_of_a_l3253_325329


namespace tree_planting_equation_holds_l3253_325367

/-- Represents the tree planting project with increased efficiency -/
structure TreePlantingProject where
  total_trees : ℕ
  efficiency_increase : ℝ
  days_ahead : ℕ
  trees_per_day : ℝ

/-- The equation holds for the given tree planting project -/
theorem tree_planting_equation_holds (project : TreePlantingProject) 
  (h1 : project.total_trees = 20000)
  (h2 : project.efficiency_increase = 0.25)
  (h3 : project.days_ahead = 5) :
  project.total_trees / project.trees_per_day - 
  project.total_trees / (project.trees_per_day * (1 + project.efficiency_increase)) = 
  project.days_ahead := by
  sorry

end tree_planting_equation_holds_l3253_325367


namespace valid_numbers_count_l3253_325348

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧
  100 ≤ n^2 ∧ n^2 < 1000 ∧
  100 ≤ (10 * (n % 10) + n / 10)^2 ∧ (10 * (n % 10) + n / 10)^2 < 1000 ∧
  n^2 = (10 * (n % 10) + n / 10)^2 % 10 * 100 + ((10 * (n % 10) + n / 10)^2 / 10 % 10) * 10 + (10 * (n % 10) + n / 10)^2 / 100

theorem valid_numbers_count :
  ∃ (S : Finset ℕ), S.card = 4 ∧ (∀ n, n ∈ S ↔ is_valid_number n) :=
sorry

end valid_numbers_count_l3253_325348


namespace steve_coins_problem_l3253_325334

/-- Represents the number of coins of each type -/
structure CoinCount where
  dimes : ℕ
  nickels : ℕ

/-- Represents the value of coins in cents -/
def coinValue (c : CoinCount) : ℕ := c.dimes * 10 + c.nickels * 5

theorem steve_coins_problem :
  ∃ (c : CoinCount),
    c.dimes + c.nickels = 36 ∧
    coinValue c = 310 ∧
    c.dimes = 26 := by
  sorry

end steve_coins_problem_l3253_325334


namespace range_of_m_l3253_325315

/-- The statement p: The equation x^2 + mx + 1 = 0 has two distinct negative real roots -/
def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- The statement q: The equation 4x^2 + 4(m-2)x + 1 = 0 has no real roots -/
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m :
  (∃ m : ℝ, ¬(p m) ∧ q m) →
  (∃ m : ℝ, 1 < m ∧ m ≤ 2) ∧ (∀ m : ℝ, (1 < m ∧ m ≤ 2) → (¬(p m) ∧ q m)) :=
by sorry

end range_of_m_l3253_325315


namespace unclaimed_candy_fraction_l3253_325375

/-- Represents the order of arrival -/
inductive Participant
| Charlie
| Alice
| Bob

/-- The fraction of candy each participant should receive based on the 4:3:2 ratio -/
def intended_share (p : Participant) : ℚ :=
  match p with
  | Participant.Charlie => 2/9
  | Participant.Alice => 4/9
  | Participant.Bob => 1/3

/-- The actual amount of candy taken by each participant -/
def actual_take (p : Participant) : ℚ :=
  match p with
  | Participant.Charlie => 2/9
  | Participant.Alice => 28/81
  | Participant.Bob => 17/81

theorem unclaimed_candy_fraction :
  1 - (actual_take Participant.Charlie + actual_take Participant.Alice + actual_take Participant.Bob) = 2/9 := by
  sorry

end unclaimed_candy_fraction_l3253_325375


namespace average_price_approx_1_70_l3253_325377

/-- The average price per bottle given the purchase of large and small bottles -/
def average_price_per_bottle (large_bottles : ℕ) (small_bottles : ℕ) 
  (large_price : ℚ) (small_price : ℚ) : ℚ :=
  ((large_bottles : ℚ) * large_price + (small_bottles : ℚ) * small_price) / 
  ((large_bottles : ℚ) + (small_bottles : ℚ))

/-- Theorem stating that the average price per bottle is approximately $1.70 -/
theorem average_price_approx_1_70 :
  let large_bottles : ℕ := 1300
  let small_bottles : ℕ := 750
  let large_price : ℚ := 189/100  -- $1.89
  let small_price : ℚ := 138/100  -- $1.38
  abs (average_price_per_bottle large_bottles small_bottles large_price small_price - 17/10) < 1/100
  := by sorry

end average_price_approx_1_70_l3253_325377


namespace quadratic_solution_l3253_325389

theorem quadratic_solution (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : p^2 + 2*p*p + q = 0)
  (h2 : q^2 + 2*p*q + q = 0) :
  p = 1 ∧ q = -3 := by
  sorry

end quadratic_solution_l3253_325389


namespace smallest_period_sin_polar_l3253_325325

theorem smallest_period_sin_polar (t : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → 
    ∃ r : ℝ, r = Real.sin θ ∧ 
    (∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ)) → 
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ 
    x = (Real.sin θ) * (Real.cos θ) ∧ 
    y = (Real.sin θ) * (Real.sin θ)) →
  t ≥ π :=
sorry

end smallest_period_sin_polar_l3253_325325


namespace right_triangle_side_length_l3253_325310

theorem right_triangle_side_length (D E F : ℝ) : 
  -- DEF is a right triangle with angle E being right
  (D^2 + E^2 = F^2) →
  -- cos(D) = (8√85)/85
  (Real.cos D = (8 * Real.sqrt 85) / 85) →
  -- EF:DF = 1:2
  (E / F = 1 / 2) →
  -- The length of DF is 2√85
  F = 2 * Real.sqrt 85 := by
  sorry

end right_triangle_side_length_l3253_325310


namespace rain_duration_l3253_325331

/-- Given a 9-hour period where it did not rain for 5 hours, prove that it rained for 4 hours. -/
theorem rain_duration (total_hours : ℕ) (no_rain_hours : ℕ) (h1 : total_hours = 9) (h2 : no_rain_hours = 5) :
  total_hours - no_rain_hours = 4 := by
  sorry

end rain_duration_l3253_325331


namespace floor_equation_solutions_l3253_325346

theorem floor_equation_solutions (x y : ℝ) :
  (∀ n : ℕ+, x * ⌊n * y⌋ = y * ⌊n * x⌋) ↔
  (x = y ∨ x = 0 ∨ y = 0 ∨ (∃ a b : ℤ, x = a ∧ y = b)) :=
sorry

end floor_equation_solutions_l3253_325346


namespace largest_number_l3253_325352

/-- Represents a number with a repeating decimal expansion -/
structure RepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : List ℕ
  repeatingPart : List ℕ

/-- Convert a RepeatingDecimal to a real number -/
noncomputable def toReal (x : RepeatingDecimal) : ℝ := sorry

/-- The numbers given in the problem -/
def a : ℝ := 9.12344
def b : RepeatingDecimal := ⟨9, [1, 2, 3], [4]⟩
def c : RepeatingDecimal := ⟨9, [1, 2], [3, 4]⟩
def d : RepeatingDecimal := ⟨9, [1], [2, 3, 4]⟩
def e : RepeatingDecimal := ⟨9, [], [1, 2, 3, 4]⟩

/-- Theorem stating that 9.123̄4 is the largest among the given numbers -/
theorem largest_number : 
  toReal b > a ∧ 
  toReal b > toReal c ∧ 
  toReal b > toReal d ∧ 
  toReal b > toReal e :=
by sorry

end largest_number_l3253_325352


namespace perfect_square_from_condition_l3253_325394

theorem perfect_square_from_condition (a b c : ℤ) (h : a * b + b * c + c * a = 1) :
  ∃ n : ℤ, (1 + a^2) * (1 + b^2) * (1 + c^2) = n^2 := by
  sorry

end perfect_square_from_condition_l3253_325394


namespace row_6_seat_16_notation_l3253_325312

/-- Represents a seat in a movie theater -/
structure Seat :=
  (row : ℕ)
  (number : ℕ)

/-- The format for denoting a seat -/
def seatNotation (s : Seat) : ℕ × ℕ := (s.row, s.number)

/-- Given condition: "row 10, seat 3" is denoted as (10,3) -/
axiom example_seat : seatNotation { row := 10, number := 3 } = (10, 3)

/-- Theorem: "row 6, seat 16" is denoted as (6,16) -/
theorem row_6_seat_16_notation :
  seatNotation { row := 6, number := 16 } = (6, 16) := by
  sorry


end row_6_seat_16_notation_l3253_325312


namespace third_valid_number_is_105_l3253_325345

def is_valid_number (n : ℕ) : Bool :=
  n < 600

def find_third_valid_number (sequence : List ℕ) : Option ℕ :=
  let valid_numbers := sequence.filter is_valid_number
  valid_numbers.get? 2

theorem third_valid_number_is_105 (sequence : List ℕ) :
  sequence = [59, 16, 95, 55, 67, 19, 98, 10, 50, 71] →
  find_third_valid_number sequence = some 105 := by
  sorry

end third_valid_number_is_105_l3253_325345


namespace waiter_customers_l3253_325322

/-- Given a number of customers who left and the number of remaining customers,
    calculate the initial number of customers. -/
def initial_customers (left : ℕ) (remaining : ℕ) : ℕ := left + remaining

/-- Theorem: Given that 9 customers left and 12 remained, 
    prove that there were initially 21 customers. -/
theorem waiter_customers : initial_customers 9 12 = 21 := by
  sorry

end waiter_customers_l3253_325322


namespace other_bill_value_l3253_325382

/-- Represents the class fund with two types of bills -/
structure ClassFund where
  total_amount : ℕ
  num_other_bills : ℕ
  value_ten_dollar_bill : ℕ

/-- Theorem stating the value of the other type of bills -/
theorem other_bill_value (fund : ClassFund)
  (h1 : fund.total_amount = 120)
  (h2 : fund.num_other_bills = 3)
  (h3 : fund.value_ten_dollar_bill = 10)
  (h4 : 2 * fund.num_other_bills = (fund.total_amount - fund.num_other_bills * (fund.total_amount / fund.num_other_bills)) / fund.value_ten_dollar_bill) :
  fund.total_amount / fund.num_other_bills = 40 := by
sorry

end other_bill_value_l3253_325382


namespace rachel_math_homework_l3253_325361

/-- The number of pages of reading homework Rachel had to complete -/
def reading_homework : ℕ := 3

/-- The additional pages of math homework compared to reading homework -/
def additional_math_pages : ℕ := 4

/-- The total number of pages of math homework Rachel had to complete -/
def math_homework : ℕ := reading_homework + additional_math_pages

theorem rachel_math_homework :
  math_homework = 7 :=
by sorry

end rachel_math_homework_l3253_325361


namespace thomas_weekly_wage_l3253_325366

/-- Calculates the weekly wage given the monthly wage and number of weeks in a month. -/
def weekly_wage (monthly_wage : ℕ) (weeks_per_month : ℕ) : ℕ :=
  monthly_wage / weeks_per_month

/-- Proves that given a monthly wage of 19500 and 4 weeks in a month, the weekly wage is 4875. -/
theorem thomas_weekly_wage :
  weekly_wage 19500 4 = 4875 := by
  sorry

#eval weekly_wage 19500 4

end thomas_weekly_wage_l3253_325366


namespace apple_count_l3253_325373

theorem apple_count (apples oranges : ℕ) 
  (h1 : apples = oranges + 27)
  (h2 : apples + oranges = 301) : 
  apples = 164 := by
sorry

end apple_count_l3253_325373


namespace investment_sum_l3253_325363

/-- Given a sum invested at different interest rates, prove the sum's value --/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 720) → P = 12000 := by
  sorry

end investment_sum_l3253_325363


namespace speed_ratio_with_head_start_l3253_325393

/-- The ratio of speeds between two runners in a race with a head start --/
theorem speed_ratio_with_head_start (va vb : ℝ) (h : va > 0 ∧ vb > 0) :
  (∃ k : ℝ, va = k * vb) →
  (va * (1 - 0.15625) = vb) →
  va / vb = 32 / 27 := by
sorry

end speed_ratio_with_head_start_l3253_325393


namespace share_calculation_l3253_325324

/-- Given a total amount divided among three parties with specific ratios, 
    prove that the first party's share is a certain value. -/
theorem share_calculation (total : ℚ) (a b c : ℚ) : 
  total = 700 →
  a + b + c = total →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a = 280 := by
  sorry

end share_calculation_l3253_325324


namespace sum_of_xyz_l3253_325314

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*x/(y+z) + z*y/(z+x) = -5)
  (h2 : y*z/(x+y) + z*x/(y+z) + x*y/(z+x) = 7) :
  x + y + z = 2 := by
  sorry

end sum_of_xyz_l3253_325314


namespace smallest_divisor_cube_sum_l3253_325307

theorem smallest_divisor_cube_sum (n : ℕ) : n ≥ 2 →
  (∃ m : ℕ, m > 0 ∧ m ∣ n ∧
    (∃ d : ℕ, d > 1 ∧ d ∣ n ∧
      (∀ k : ℕ, k > 1 ∧ k ∣ n → k ≥ d) ∧
      n = d^3 + m^3)) →
  n = 16 ∨ n = 72 ∨ n = 520 :=
by sorry

end smallest_divisor_cube_sum_l3253_325307


namespace lisa_decorative_spoons_l3253_325341

/-- The number of children Lisa has -/
def num_children : ℕ := 4

/-- The number of baby spoons each child had -/
def baby_spoons_per_child : ℕ := 3

/-- The number of large spoons in the new cutlery set -/
def new_large_spoons : ℕ := 10

/-- The number of teaspoons in the new cutlery set -/
def new_teaspoons : ℕ := 15

/-- The total number of spoons Lisa has now -/
def total_spoons : ℕ := 39

/-- The number of decorative spoons Lisa created -/
def decorative_spoons : ℕ := total_spoons - (new_large_spoons + new_teaspoons) - (num_children * baby_spoons_per_child)

theorem lisa_decorative_spoons : decorative_spoons = 2 := by
  sorry

end lisa_decorative_spoons_l3253_325341


namespace sum_of_roots_l3253_325311

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x + y = 1 := by
  sorry

end sum_of_roots_l3253_325311


namespace rotation_matrix_correct_l3253_325396

def A : Fin 2 → ℝ := ![1, 1]
def B : Fin 2 → ℝ := ![-1, 1]
def C : Fin 2 → ℝ := ![-1, -1]
def D : Fin 2 → ℝ := ![1, -1]

def N : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; -1, 0]

theorem rotation_matrix_correct :
  N.mulVec A = D ∧
  N.mulVec B = A ∧
  N.mulVec C = B ∧
  N.mulVec D = C := by
  sorry

end rotation_matrix_correct_l3253_325396


namespace incorrect_transformation_l3253_325323

theorem incorrect_transformation (x y m : ℝ) :
  ¬(∀ (x y m : ℝ), x = y → x / m = y / m) :=
sorry

end incorrect_transformation_l3253_325323


namespace gcf_of_270_108_150_l3253_325342

theorem gcf_of_270_108_150 : Nat.gcd 270 (Nat.gcd 108 150) = 30 := by
  sorry

end gcf_of_270_108_150_l3253_325342


namespace smallest_rectangular_block_l3253_325347

theorem smallest_rectangular_block (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 462 → 
  l * m * n ≥ 672 ∧ 
  ∃ (l' m' n' : ℕ), (l' - 1) * (m' - 1) * (n' - 1) = 462 ∧ l' * m' * n' = 672 :=
by sorry

end smallest_rectangular_block_l3253_325347


namespace library_problem_l3253_325397

theorem library_problem (total_books : ℕ) (books_per_student : ℕ) 
  (day1_students : ℕ) (day2_students : ℕ) (day4_students : ℕ) :
  total_books = 120 →
  books_per_student = 5 →
  day1_students = 4 →
  day2_students = 5 →
  day4_students = 9 →
  ∃ (day3_students : ℕ),
    day3_students = 6 ∧
    total_books = (day1_students + day2_students + day3_students + day4_students) * books_per_student :=
by sorry

end library_problem_l3253_325397


namespace numerator_increase_percentage_l3253_325350

theorem numerator_increase_percentage (P : ℚ) : 
  (1 + P / 100) / ((3 / 4) * 12) = 2 / 15 → P = 20 := by
  sorry

end numerator_increase_percentage_l3253_325350


namespace border_length_is_even_l3253_325343

/-- Represents a domino on the board -/
inductive Domino
| Horizontal
| Vertical

/-- Represents the board -/
def Board := Fin 2010 → Fin 2011 → Domino

/-- The border length between horizontal and vertical dominoes -/
def borderLength (board : Board) : ℕ := sorry

/-- Theorem stating that the border length is even -/
theorem border_length_is_even (board : Board) : 
  Even (borderLength board) := by sorry

end border_length_is_even_l3253_325343


namespace colored_ngon_at_most_two_colors_l3253_325371

/-- A regular n-gon with colored sides and diagonals -/
structure ColoredNGon where
  n : ℕ
  vertices : Fin n → Point
  colors : ℕ
  coloring : (Fin n × Fin n) → Fin colors

/-- The coloring satisfies the first condition -/
def satisfies_condition1 (R : ColoredNGon) : Prop :=
  ∀ c : Fin R.colors, ∀ A B : Fin R.n,
    (R.coloring (A, B) = c) ∨
    (∃ C : Fin R.n, R.coloring (A, C) = c ∧ R.coloring (B, C) = c)

/-- The coloring satisfies the second condition -/
def satisfies_condition2 (R : ColoredNGon) : Prop :=
  ∀ A B C : Fin R.n,
    (R.coloring (A, B) ≠ R.coloring (B, C)) →
    (R.coloring (A, C) = R.coloring (A, B) ∨ R.coloring (A, C) = R.coloring (B, C))

/-- Main theorem: If a ColoredNGon satisfies both conditions, then it has at most 2 colors -/
theorem colored_ngon_at_most_two_colors (R : ColoredNGon)
  (h1 : satisfies_condition1 R) (h2 : satisfies_condition2 R) :
  R.colors ≤ 2 :=
sorry

end colored_ngon_at_most_two_colors_l3253_325371


namespace karen_total_distance_l3253_325378

/-- The number of shelves in the library. -/
def num_shelves : ℕ := 4

/-- The number of books on each shelf. -/
def books_per_shelf : ℕ := 400

/-- The total number of books in the library. -/
def total_books : ℕ := num_shelves * books_per_shelf

/-- The distance in miles from the library to Karen's home. -/
def distance_to_home : ℕ := total_books

/-- The total distance Karen bikes from home to library and back. -/
def total_distance : ℕ := 2 * distance_to_home

/-- Theorem stating that the total distance Karen bikes is 3200 miles. -/
theorem karen_total_distance : total_distance = 3200 := by
  sorry

end karen_total_distance_l3253_325378


namespace smallest_possible_a_l3253_325355

theorem smallest_possible_a (a b : ℤ) (x : ℝ) (h1 : a > x) (h2 : a < 41)
  (h3 : b > 39) (h4 : b < 51)
  (h5 : (↑40 / ↑40 : ℚ) - (↑a / ↑50 : ℚ) = 2/5) : a ≥ 30 := by
  sorry

end smallest_possible_a_l3253_325355


namespace centroid_unique_point_l3253_325353

/-- Definition of a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Definition of the centroid of a triangle -/
def centroid (t : Triangle) : Point :=
  sorry

/-- Definition of a point being inside or on the boundary of a triangle -/
def insideOrOnBoundary (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Definition of the area of a triangle -/
def area (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating that the centroid is the unique point satisfying the condition -/
theorem centroid_unique_point (t : Triangle) :
  ∃! M, insideOrOnBoundary M t ∧
    ∀ N, insideOrOnBoundary N t →
      ∃ P, insideOrOnBoundary P t ∧
        area (Triangle.mk M N P) ≥ (1/6 : ℝ) * area t :=
  sorry

end centroid_unique_point_l3253_325353


namespace vasyas_numbers_l3253_325364

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
sorry

end vasyas_numbers_l3253_325364


namespace integral_x_plus_inverse_x_l3253_325384

open Real MeasureTheory

theorem integral_x_plus_inverse_x : ∫ x in (1 : ℝ)..2, (x + 1/x) = 3/2 + Real.log 2 := by sorry

end integral_x_plus_inverse_x_l3253_325384


namespace circle_equation_l3253_325374

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition for a circle to be tangent to the y-axis
def tangentToYAxis (c : Circle) : Prop :=
  c.center.1 = c.radius

-- Define the condition for the center to be on the line 3x - y = 0
def centerOnLine (c : Circle) : Prop :=
  c.center.2 = 3 * c.center.1

-- Define the condition for the circle to pass through point (2,3)
def passesThrough (c : Circle) : Prop :=
  (c.center.1 - 2)^2 + (c.center.2 - 3)^2 = c.radius^2

-- Define the equation of the circle
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- State the theorem
theorem circle_equation :
  ∀ c : Circle,
  tangentToYAxis c → centerOnLine c → passesThrough c →
  (∀ x y : ℝ, circleEquation c x y ↔ 
    ((x - 1)^2 + (y - 3)^2 = 1) ∨ 
    ((x - 13/9)^2 + (y - 13/3)^2 = 169/81)) :=
by sorry

end circle_equation_l3253_325374


namespace only_shape3_symmetric_shape3_is_symmetric_other_shapes_not_symmetric_l3253_325392

-- Define the type for L-like shapes
inductive LLikeShape
| Shape1
| Shape2
| Shape3
| Shape4
| Shape5

-- Define a function to check if a shape is symmetric to the original
def isSymmetric (shape : LLikeShape) : Prop :=
  match shape with
  | LLikeShape.Shape3 => True
  | _ => False

-- Theorem stating that only Shape3 is symmetric
theorem only_shape3_symmetric :
  ∀ (shape : LLikeShape), isSymmetric shape ↔ shape = LLikeShape.Shape3 :=
by sorry

-- Theorem stating that Shape3 is indeed symmetric
theorem shape3_is_symmetric : isSymmetric LLikeShape.Shape3 :=
by sorry

-- Theorem stating that other shapes are not symmetric
theorem other_shapes_not_symmetric :
  ∀ (shape : LLikeShape), shape ≠ LLikeShape.Shape3 → ¬(isSymmetric shape) :=
by sorry

end only_shape3_symmetric_shape3_is_symmetric_other_shapes_not_symmetric_l3253_325392


namespace smallest_angle_in_triangle_l3253_325379

theorem smallest_angle_in_triangle (a b c : ℝ) (C : ℝ) : 
  a = 2 →
  b = 2 →
  c ≥ 4 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  C ≥ 120 * Real.pi / 180 :=
by sorry

end smallest_angle_in_triangle_l3253_325379


namespace intersection_value_l3253_325340

/-- Given a proportional function y = kx (k ≠ 0) and an inverse proportional function y = -5/x
    intersecting at points A(x₁, y₁) and B(x₂, y₂), the value of x₁y₂ - 3x₂y₁ is equal to 10. -/
theorem intersection_value (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : 
  k ≠ 0 →
  y₁ = k * x₁ →
  y₁ = -5 / x₁ →
  y₂ = k * x₂ →
  y₂ = -5 / x₂ →
  x₁ * y₂ - 3 * x₂ * y₁ = 10 := by
  sorry

end intersection_value_l3253_325340


namespace april_largest_difference_l3253_325327

/-- Represents the months of cookie sales --/
inductive Month
| january
| february
| march
| april
| may

/-- Calculates the percentage difference between two sales values --/
def percentageDifference (x y : ℕ) : ℚ :=
  (max x y - min x y : ℚ) / (min x y : ℚ) * 100

/-- Returns the sales data for Rangers and Scouts for a given month --/
def salesData (m : Month) : ℕ × ℕ :=
  match m with
  | .january => (5, 4)
  | .february => (6, 4)
  | .march => (5, 5)
  | .april => (7, 4)
  | .may => (3, 5)

/-- Theorem: April has the largest percentage difference in cookie sales --/
theorem april_largest_difference :
  ∀ m : Month, m ≠ Month.april →
    percentageDifference (salesData Month.april).1 (salesData Month.april).2 ≥
    percentageDifference (salesData m).1 (salesData m).2 :=
by sorry

end april_largest_difference_l3253_325327


namespace tangent_line_equation_l3253_325357

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * log x

theorem tangent_line_equation :
  let p : ℝ × ℝ := (2, f 2)
  let m : ℝ := (deriv f) 2
  let tangent_eq (x y : ℝ) : Prop := x - y + 2 * log 2 - 2 = 0
  tangent_eq p.1 p.2 ∧ ∀ x y, tangent_eq x y ↔ y - p.2 = m * (x - p.1) :=
by sorry

end tangent_line_equation_l3253_325357


namespace different_monotonicity_implies_inequality_l3253_325308

/-- Given a > 1, a ≠ 2, and (a-1)^x and (1/a)^x have different monotonicities,
    prove that (a-1)^(1/3) > (1/a)^3 -/
theorem different_monotonicity_implies_inequality (a : ℝ) 
  (h1 : a > 1) 
  (h2 : a ≠ 2) 
  (h3 : ∀ x y : ℝ, (∃ ε > 0, ∀ δ ∈ Set.Ioo (x - ε) (x + ε), 
    ((a - 1) ^ δ - (a - 1) ^ x) * ((1 / a) ^ δ - (1 / a) ^ x) < 0)) :
  (a - 1) ^ (1 / 3) > (1 / a) ^ 3 := by
  sorry

end different_monotonicity_implies_inequality_l3253_325308


namespace remainder_777_power_777_mod_13_l3253_325333

theorem remainder_777_power_777_mod_13 : 777^777 ≡ 12 [ZMOD 13] := by
  sorry

end remainder_777_power_777_mod_13_l3253_325333


namespace simultaneous_equations_imply_quadratic_l3253_325372

theorem simultaneous_equations_imply_quadratic (x y : ℝ) :
  (2 * x^2 + 6 * x + 5 * y + 1 = 0) →
  (2 * x + y + 3 = 0) →
  (y^2 + 10 * y - 7 = 0) :=
by
  sorry

end simultaneous_equations_imply_quadratic_l3253_325372


namespace area_between_curves_l3253_325354

-- Define the functions for the curves
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x^3

-- Define the bounds of integration
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- State the theorem
theorem area_between_curves : 
  (∫ x in lower_bound..upper_bound, f x - g x) = 1/12 := by sorry

end area_between_curves_l3253_325354


namespace triangle_minimum_value_l3253_325370

theorem triangle_minimum_value (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < B) → (B < Real.pi / 2) →
  (Real.cos B)^2 + (1/2) * Real.sin (2 * B) = 1 →
  -- |BC + AB| = 3
  b = 3 →
  -- Minimum value of 16b/(ac)
  (∀ x y z : Real, x > 0 → y > 0 → z > 0 →
    (Real.cos x)^2 + (1/2) * Real.sin (2 * x) = 1 →
    y = 3 →
    16 * y / (z * x) ≥ 16 * (2 - Real.sqrt 2) / 3) ∧
  (∃ x y z : Real, x > 0 ∧ y > 0 ∧ z > 0 ∧
    (Real.cos x)^2 + (1/2) * Real.sin (2 * x) = 1 ∧
    y = 3 ∧
    16 * y / (z * x) = 16 * (2 - Real.sqrt 2) / 3) :=
by sorry

end triangle_minimum_value_l3253_325370


namespace number_of_hens_l3253_325318

/-- Given a farm with hens and cows, prove that the number of hens is 24 -/
theorem number_of_hens (hens cows : ℕ) : 
  hens + cows = 44 →  -- Total number of heads
  2 * hens + 4 * cows = 128 →  -- Total number of feet
  hens = 24 := by
  sorry

end number_of_hens_l3253_325318


namespace line_perpendicular_to_plane_l3253_325349

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α β : Plane) 
  (h1 : parallel m n) 
  (h2 : plane_perpendicular α β) 
  (h3 : perpendicular m α) : 
  perpendicular n α :=
sorry

end line_perpendicular_to_plane_l3253_325349


namespace quadratic_roots_sum_product_l3253_325388

theorem quadratic_roots_sum_product (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x - 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + x₂ - x₁*x₂ = 5 →
  m = -2 := by
sorry

end quadratic_roots_sum_product_l3253_325388


namespace constant_term_of_product_l3253_325320

def is_monic (p : Polynomial ℝ) : Prop :=
  p.leadingCoeff = 1

theorem constant_term_of_product (p q : Polynomial ℝ) :
  is_monic p →
  is_monic q →
  p.degree = 3 →
  q.degree = 3 →
  (∃ c : ℝ, c > 0 ∧ p.coeff 0 = c ∧ q.coeff 0 = c) →
  (∃ a : ℝ, p.coeff 1 = a ∧ q.coeff 1 = a) →
  p * q = Polynomial.monomial 6 1 + Polynomial.monomial 5 2 + Polynomial.monomial 4 1 +
          Polynomial.monomial 3 2 + Polynomial.monomial 2 9 + Polynomial.monomial 1 12 +
          Polynomial.monomial 0 36 →
  p.coeff 0 = 6 ∧ q.coeff 0 = 6 :=
by sorry

end constant_term_of_product_l3253_325320


namespace population_change_l3253_325387

/-- Represents the population changes over 5 years -/
structure PopulationChange where
  year1 : Real
  year2 : Real
  year3 : Real
  year4 : Real
  year5 : Real

/-- Calculates the final population given an initial population and population changes -/
def finalPopulation (initialPop : Real) (changes : PopulationChange) : Real :=
  initialPop * (1 + changes.year1) * (1 + changes.year2) * (1 + changes.year3) * (1 + changes.year4) * (1 + changes.year5)

/-- The theorem to be proved -/
theorem population_change (changes : PopulationChange) 
  (h1 : changes.year1 = 0.10)
  (h2 : changes.year2 = -0.08)
  (h3 : changes.year3 = 0.15)
  (h4 : changes.year4 = -0.06)
  (h5 : changes.year5 = 0.12)
  (h6 : finalPopulation 13440 changes = 16875) : 
  ∃ (initialPop : Real), abs (initialPop - 13440) < 1 ∧ finalPopulation initialPop changes = 16875 := by
  sorry

end population_change_l3253_325387


namespace linda_travel_distance_l3253_325330

/-- Represents the travel data for one day -/
structure DayTravel where
  minutes_per_mile : ℕ
  distance : ℕ

/-- Calculates the distance traveled in one hour given the minutes per mile -/
def distance_traveled (minutes_per_mile : ℕ) : ℕ :=
  60 / minutes_per_mile

/-- Generates the travel data for four days -/
def generate_four_days (initial_minutes_per_mile : ℕ) : List DayTravel :=
  [0, 1, 2, 3].map (λ i =>
    { minutes_per_mile := initial_minutes_per_mile + i * 5,
      distance := distance_traveled (initial_minutes_per_mile + i * 5) })

theorem linda_travel_distance :
  ∃ (initial_minutes_per_mile : ℕ),
    let four_days := generate_four_days initial_minutes_per_mile
    four_days.length = 4 ∧
    (∀ day ∈ four_days, day.minutes_per_mile > 0 ∧ day.minutes_per_mile ≤ 60) ∧
    (∀ day ∈ four_days, day.distance > 0) ∧
    (List.sum (four_days.map (λ day => day.distance)) = 25) := by
  sorry

end linda_travel_distance_l3253_325330


namespace not_divisible_power_ten_plus_one_l3253_325309

theorem not_divisible_power_ten_plus_one (m n : ℕ) :
  ¬ ∃ (k : ℕ), (10^m + 1) = k * (10^n - 1) := by
  sorry

end not_divisible_power_ten_plus_one_l3253_325309


namespace min_value_expression_l3253_325383

theorem min_value_expression (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  1 / (x + y)^2 + 1 / (x - y)^2 ≥ 1 := by
  sorry

end min_value_expression_l3253_325383


namespace playground_area_is_297_l3253_325301

/-- Calculates the area of a rectangular playground given the specified conditions --/
def playground_area (total_posts : ℕ) (post_spacing : ℕ) : ℕ :=
  let shorter_side_posts := 4  -- Including corners
  let longer_side_posts := 3 * shorter_side_posts
  let shorter_side_length := post_spacing * (shorter_side_posts - 1)
  let longer_side_length := post_spacing * (longer_side_posts - 1)
  shorter_side_length * longer_side_length

/-- Theorem stating that the area of the playground under given conditions is 297 square yards --/
theorem playground_area_is_297 :
  playground_area 24 3 = 297 := by
  sorry

end playground_area_is_297_l3253_325301


namespace cubic_sum_equals_265_l3253_325391

theorem cubic_sum_equals_265 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = -14) :
  a^3 + a^2*b + a*b^2 + b^3 = 265 := by sorry

end cubic_sum_equals_265_l3253_325391


namespace alley_width_l3253_325386

/-- Given a ladder of length l placed in an alley, touching one wall at a 60° angle
    and the other wall at a 30° angle with the ground, the width w of the alley
    is equal to l(√3 + 1)/2. -/
theorem alley_width (l : ℝ) (h : l > 0) :
  let w := l * (Real.sqrt 3 + 1) / 2
  let angle_A := 60 * π / 180
  let angle_B := 30 * π / 180
  ∃ (m : ℝ), m > 0 ∧ w = l * Real.sin angle_A + l * Real.sin angle_B :=
sorry

end alley_width_l3253_325386


namespace min_max_sum_l3253_325395

theorem min_max_sum (a b c d e : ℕ+) (h_sum : a + b + c + d + e = 2018) :
  673 ≤ max (a + b) (max (b + c) (max (c + d) (d + e))) ∧
  ∃ (a' b' c' d' e' : ℕ+), a' + b' + c' + d' + e' = 2018 ∧
    max (a' + b') (max (b' + c') (max (c' + d') (d' + e'))) = 673 :=
by sorry

end min_max_sum_l3253_325395


namespace kaleb_tickets_proof_l3253_325336

/-- The number of tickets Kaleb initially bought at the fair -/
def initial_tickets : ℕ := 6

/-- The cost of each ticket in dollars -/
def ticket_cost : ℕ := 9

/-- The amount Kaleb spent on the ferris wheel in dollars -/
def ferris_wheel_cost : ℕ := 27

/-- The number of tickets Kaleb had left after riding the ferris wheel -/
def remaining_tickets : ℕ := 3

/-- Theorem stating that the initial number of tickets is correct given the conditions -/
theorem kaleb_tickets_proof :
  initial_tickets = (ferris_wheel_cost / ticket_cost) + remaining_tickets :=
by sorry

end kaleb_tickets_proof_l3253_325336


namespace men_left_hostel_l3253_325359

/-- Proves that 50 men left the hostel given the initial and final conditions -/
theorem men_left_hostel (initial_men : ℕ) (initial_days : ℕ) (final_days : ℕ) 
  (h1 : initial_men = 250)
  (h2 : initial_days = 40)
  (h3 : final_days = 50)
  (h4 : initial_men * initial_days = (initial_men - men_left) * final_days) :
  men_left = 50 := by
  sorry

#check men_left_hostel

end men_left_hostel_l3253_325359


namespace probability_all_odd_is_one_forty_second_l3253_325365

def total_slips : ℕ := 10
def odd_slips : ℕ := 5
def draws : ℕ := 4

def probability_all_odd : ℚ := (odd_slips.choose draws) / (total_slips.choose draws)

theorem probability_all_odd_is_one_forty_second :
  probability_all_odd = 1 / 42 := by sorry

end probability_all_odd_is_one_forty_second_l3253_325365


namespace passenger_catches_train_l3253_325335

/-- Represents the problem of a passenger trying to catch a train --/
theorem passenger_catches_train 
  (train_delay : ℝ) 
  (train_speed : ℝ) 
  (distance : ℝ) 
  (train_stop_time : ℝ) 
  (passenger_delay : ℝ) 
  (passenger_speed : ℝ) 
  (h1 : train_delay = 11) 
  (h2 : train_speed = 10) 
  (h3 : distance = 1.5) 
  (h4 : train_stop_time = 14.5) 
  (h5 : passenger_delay = 12) 
  (h6 : passenger_speed = 4) :
  passenger_delay + distance / passenger_speed * 60 ≤ 
  train_delay + distance / train_speed * 60 + train_stop_time := by
  sorry

#check passenger_catches_train

end passenger_catches_train_l3253_325335


namespace isosceles_triangle_base_length_l3253_325321

/-- An isosceles triangle with congruent sides of length 7 cm and perimeter 23 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length : ∀ (base : ℝ), 
  base > 0 → -- The base length is positive
  7 > 0 → -- The congruent side length is positive
  2 * 7 + base = 23 → -- The perimeter is 23 cm
  base = 9 := by
  sorry

end isosceles_triangle_base_length_l3253_325321


namespace floor_double_floor_eq_42_l3253_325319

theorem floor_double_floor_eq_42 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 42 ↔ 7 ≤ x ∧ x < 43/6 := by
  sorry

end floor_double_floor_eq_42_l3253_325319


namespace square_perimeter_l3253_325358

/-- The perimeter of a square with side length 19 cm is 76 cm. -/
theorem square_perimeter : 
  ∀ (s : ℝ), s = 19 → 4 * s = 76 := by
  sorry

end square_perimeter_l3253_325358


namespace eliminate_x_from_system_l3253_325338

theorem eliminate_x_from_system (x y : ℝ) :
  (5 * x - 3 * y = -5) ∧ (5 * x + 4 * y = -1) → 7 * y = 4 := by
sorry

end eliminate_x_from_system_l3253_325338


namespace circle_properties_l3253_325313

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 9
def C₂ (x y : ℝ) : Prop := (x-1)^2 + (y+1)^2 = 16

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

theorem circle_properties :
  -- 1. The equation of the line passing through the centers of C₁ and C₂ is y = -x
  (∃ m b : ℝ, ∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = -1) → y = m * x + b) ∧
  (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = -1) → y = -x) ∧
  -- 2. The circles intersect and the length of their common chord is √94/2
  (∃ x₁ y₁ x₂ y₂ : ℝ, C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
    ((x₂ - x₁)^2 + (y₂ - y₁)^2)^(1/2) = (94)^(1/2)/2) ∧
  -- 3. There exist exactly 4 points on C₂ that are at a distance of 2 from the line y = x
  (∃! (a b c d : ℝ × ℝ), 
    C₂ a.1 a.2 ∧ C₂ b.1 b.2 ∧ C₂ c.1 c.2 ∧ C₂ d.1 d.2 ∧
    (∀ x y : ℝ, line_y_eq_x x y → 
      ((a.1 - x)^2 + (a.2 - y)^2)^(1/2) = 2 ∧
      ((b.1 - x)^2 + (b.2 - y)^2)^(1/2) = 2 ∧
      ((c.1 - x)^2 + (c.2 - y)^2)^(1/2) = 2 ∧
      ((d.1 - x)^2 + (d.2 - y)^2)^(1/2) = 2)) :=
by
  sorry

end circle_properties_l3253_325313


namespace min_value_fraction_l3253_325316

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 2) : 
  (x + y) / (x * y * z) ≥ 4 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 2 ∧ (a + b) / (a * b * c) = 4 :=
sorry

end min_value_fraction_l3253_325316


namespace bellas_age_l3253_325368

theorem bellas_age : 
  ∀ (bella_age : ℕ), 
  (bella_age + (bella_age + 9) + (bella_age / 2) = 27) → 
  bella_age = 6 := by
sorry

end bellas_age_l3253_325368


namespace sarah_desserts_l3253_325399

theorem sarah_desserts (michael_cookies : ℕ) (sarah_cupcakes : ℕ) :
  michael_cookies = 5 →
  sarah_cupcakes = 9 →
  sarah_cupcakes / 3 = sarah_cupcakes - (sarah_cupcakes / 3) →
  michael_cookies + (sarah_cupcakes - (sarah_cupcakes / 3)) = 11 :=
by sorry

end sarah_desserts_l3253_325399
