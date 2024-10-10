import Mathlib

namespace minimum_guests_l3261_326150

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (min_guests : ℕ) :
  total_food = 411 →
  max_per_guest = 2.5 →
  min_guests = ⌈total_food / max_per_guest⌉ →
  min_guests = 165 := by
  sorry

end minimum_guests_l3261_326150


namespace mindy_emails_l3261_326121

theorem mindy_emails (e m : ℕ) (h1 : e = 9 * m - 7) (h2 : e + m = 93) : e = 83 := by
  sorry

end mindy_emails_l3261_326121


namespace karlson_max_candies_l3261_326114

/-- The number of vertices in the complete graph -/
def n : ℕ := 29

/-- The maximum number of candies Karlson could eat -/
def max_candies : ℕ := 406

/-- Theorem stating the maximum number of candies Karlson could eat -/
theorem karlson_max_candies :
  (n * (n - 1)) / 2 = max_candies := by
  sorry

end karlson_max_candies_l3261_326114


namespace inequality_solution_l3261_326120

theorem inequality_solution (x : ℕ+) : 4 * x - 3 < 2 * x + 1 ↔ x = 1 := by sorry

end inequality_solution_l3261_326120


namespace derivative_f_at_one_l3261_326153

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.sin x

theorem derivative_f_at_one :
  deriv f 1 = 3 * Real.sin 1 + Real.cos 1 := by sorry

end derivative_f_at_one_l3261_326153


namespace binomial_odd_iff_binary_condition_l3261_326107

def has_one_at_position (m : ℕ) (pos : ℕ) : Prop :=
  (m / 2^pos) % 2 = 1

theorem binomial_odd_iff_binary_condition (n k : ℕ) :
  Nat.choose n k % 2 = 1 ↔ ∀ pos, has_one_at_position k pos → has_one_at_position n pos :=
sorry

end binomial_odd_iff_binary_condition_l3261_326107


namespace regular_hexagon_radius_l3261_326106

/-- The radius of a regular hexagon with perimeter 12a is 2a -/
theorem regular_hexagon_radius (a : ℝ) (h : a > 0) :
  let perimeter := 12 * a
  ∃ (radius : ℝ), radius = 2 * a ∧ 
    (∃ (side : ℝ), side * 6 = perimeter ∧ radius = side) := by
  sorry

end regular_hexagon_radius_l3261_326106


namespace factorial_500_trailing_zeros_l3261_326137

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 500! has 124 trailing zeros -/
theorem factorial_500_trailing_zeros :
  trailingZeros 500 = 124 := by
  sorry

end factorial_500_trailing_zeros_l3261_326137


namespace partial_fraction_decomposition_l3261_326136

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  (3 * x^2 - 2 * x) / ((x - 4) * (x - 2)^2) = 
  10 / (x - 4) + (-7) / (x - 2) + (-4) / (x - 2)^2 := by
  sorry

end partial_fraction_decomposition_l3261_326136


namespace computer_employee_savings_l3261_326170

/-- Calculates the employee savings on a computer purchase given the initial cost,
    markup percentage, and employee discount percentage. -/
def employeeSavings (initialCost : ℝ) (markupPercentage : ℝ) (discountPercentage : ℝ) : ℝ :=
  let retailPrice := initialCost * (1 + markupPercentage)
  retailPrice * discountPercentage

/-- Theorem stating that an employee saves $86.25 when buying a computer
    with a 15% markup and 15% employee discount, given an initial cost of $500. -/
theorem computer_employee_savings :
  employeeSavings 500 0.15 0.15 = 86.25 := by
  sorry


end computer_employee_savings_l3261_326170


namespace total_amount_is_234_l3261_326113

/-- Represents the division of money among three parties -/
structure MoneyDivision where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Theorem stating the total amount given the conditions -/
theorem total_amount_is_234 
  (div : MoneyDivision) 
  (h1 : div.y = div.x * (45/100))  -- y gets 45 paisa for each rupee x gets
  (h2 : div.z = div.x * (50/100))  -- z gets 50 paisa for each rupee x gets
  (h3 : div.y = 54)                -- The share of y is Rs. 54
  : div.x + div.y + div.z = 234 := by
  sorry

#check total_amount_is_234

end total_amount_is_234_l3261_326113


namespace square_circle_union_area_l3261_326133

/-- The area of the union of a square and a circle with specific properties -/
theorem square_circle_union_area :
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 144 + 108 * π := by
  sorry

end square_circle_union_area_l3261_326133


namespace adams_goats_l3261_326160

theorem adams_goats (adam andrew ahmed : ℕ) 
  (andrew_eq : andrew = 2 * adam + 5)
  (ahmed_eq : ahmed = andrew - 6)
  (ahmed_count : ahmed = 13) : 
  adam = 7 := by
  sorry

end adams_goats_l3261_326160


namespace deer_cheetah_time_difference_l3261_326135

/-- Proves the time difference between a deer and cheetah passing a point, given their speeds and catch-up time. -/
theorem deer_cheetah_time_difference 
  (deer_speed : ℝ) 
  (cheetah_speed : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : deer_speed = 50) 
  (h2 : cheetah_speed = 60) 
  (h3 : catch_up_time = 1) : 
  ∃ (time_difference : ℝ), 
    time_difference = 4 ∧ 
    deer_speed * (catch_up_time + time_difference) = cheetah_speed * catch_up_time :=
by sorry

end deer_cheetah_time_difference_l3261_326135


namespace speed_ratio_problem_l3261_326189

/-- 
Given two people traveling in opposite directions for one hour, 
if one person takes 35 minutes longer to reach the other's destination when they swap,
then the ratio of their speeds is 3:4.
-/
theorem speed_ratio_problem (v₁ v₂ : ℝ) : 
  v₁ > 0 → v₂ > 0 → 
  (60 * v₂ = 60 * v₁ / v₂ + 35 * v₁) → 
  v₁ / v₂ = 3 / 4 :=
by sorry

end speed_ratio_problem_l3261_326189


namespace house_sale_percentage_l3261_326168

theorem house_sale_percentage (market_value : ℝ) (num_people : ℕ) (after_tax_per_person : ℝ) (tax_rate : ℝ) :
  market_value = 500000 →
  num_people = 4 →
  after_tax_per_person = 135000 →
  tax_rate = 0.1 →
  ((num_people * after_tax_per_person / (1 - tax_rate) - market_value) / market_value) * 100 = 20 := by
  sorry

end house_sale_percentage_l3261_326168


namespace polynomial_functional_equation_l3261_326119

theorem polynomial_functional_equation (p : ℝ → ℝ) :
  (∀ x : ℝ, p (x^3) - p (x^3 - 2) = (p x)^2 + 18) →
  (∃ a : ℝ, a^2 = 30 ∧ (∀ x : ℝ, p x = 6 * x^3 + a)) :=
by sorry

end polynomial_functional_equation_l3261_326119


namespace min_selling_price_A_l3261_326161

/-- Represents the number of units of model A purchased -/
def units_A : ℕ := 100

/-- Represents the number of units of model B purchased -/
def units_B : ℕ := 160 - units_A

/-- Represents the cost price of model A in yuan -/
def cost_A : ℕ := 150

/-- Represents the cost price of model B in yuan -/
def cost_B : ℕ := 350

/-- Represents the total cost of purchasing both models in yuan -/
def total_cost : ℕ := 36000

/-- Represents the minimum required gross profit in yuan -/
def min_gross_profit : ℕ := 11000

/-- Theorem stating that the minimum selling price of model A is 200 yuan -/
theorem min_selling_price_A : 
  ∃ (selling_price_A : ℕ), 
    selling_price_A = 200 ∧ 
    units_A * cost_A + units_B * cost_B = total_cost ∧
    units_A * (selling_price_A - cost_A) + units_B * (2 * (selling_price_A - cost_A)) ≥ min_gross_profit ∧
    ∀ (price : ℕ), price < selling_price_A → 
      units_A * (price - cost_A) + units_B * (2 * (price - cost_A)) < min_gross_profit :=
by
  sorry


end min_selling_price_A_l3261_326161


namespace special_ellipse_properties_l3261_326156

/-- An ellipse with one vertex at (0,1) and focus on the x-axis -/
structure SpecialEllipse where
  /-- The right focus of the ellipse -/
  focus : ℝ × ℝ
  /-- The distance from the right focus to the line x-y+2√2=0 is 3 -/
  focus_distance : (|focus.1 + 2 * Real.sqrt 2| : ℝ) / Real.sqrt 2 = 3

/-- The equation of the ellipse -/
def ellipse_equation (e : SpecialEllipse) (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

/-- A line passing through (0,1) -/
structure LineThroughA where
  /-- The slope of the line -/
  k : ℝ

/-- The equation of a line passing through (0,1) -/
def line_equation (l : LineThroughA) (x y : ℝ) : Prop :=
  y = l.k * x + 1

/-- The theorem to be proved -/
theorem special_ellipse_properties (e : SpecialEllipse) :
  (∀ x y, ellipse_equation e x y ↔ x^2 / 3 + y^2 = 1) ∧
  (∀ l : LineThroughA, line_equation l = line_equation ⟨1⟩ ∨ line_equation l = line_equation ⟨-1⟩ →
    ∀ l' : LineThroughA, ∃ x y, ellipse_equation e x y ∧ line_equation l x y ∧ line_equation l' x y →
      ∀ x' y', ellipse_equation e x' y' ∧ line_equation l' x' y' →
        (x - 0)^2 + (y - 1)^2 ≥ (x' - 0)^2 + (y' - 1)^2) :=
by sorry

end special_ellipse_properties_l3261_326156


namespace circle_equation_proof_l3261_326191

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x/4 + y/2 = 1

/-- Point A is where the line intersects the x-axis -/
def point_A : ℝ × ℝ := (4, 0)

/-- Point B is where the line intersects the y-axis -/
def point_B : ℝ × ℝ := (0, 2)

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

/-- Theorem: The equation of the circle with diameter AB is x^2 + y^2 - 4x - 2y = 0 -/
theorem circle_equation_proof :
  ∀ x y : ℝ, line_equation x y →
  (∃ t : ℝ, x = t * (point_B.1 - point_A.1) + point_A.1 ∧
            y = t * (point_B.2 - point_A.2) + point_A.2) →
  circle_equation x y :=
sorry

end circle_equation_proof_l3261_326191


namespace ellipse_foci_distance_l3261_326146

/-- The distance between the foci of an ellipse described by the equation
    √((x-4)² + (y+5)²) + √((x+6)² + (y-7)²) = 22 is equal to 2√2. -/
theorem ellipse_foci_distance : 
  let ellipse := {p : ℝ × ℝ | Real.sqrt ((p.1 - 4)^2 + (p.2 + 5)^2) + 
                               Real.sqrt ((p.1 + 6)^2 + (p.2 - 7)^2) = 22}
  let foci := ((4, -5), (-6, 7))
  ∃ (d : ℝ), d = Real.sqrt 8 ∧ 
    d = Real.sqrt ((foci.1.1 - foci.2.1)^2 + (foci.1.2 - foci.2.2)^2) := by
  sorry

end ellipse_foci_distance_l3261_326146


namespace bom_watermelon_seeds_l3261_326110

/-- Given the number of watermelon seeds for Bom, Gwi, and Yeon, prove that Bom has 300 seeds. -/
theorem bom_watermelon_seeds :
  ∀ (bom gwi yeon : ℕ),
  gwi = bom + 40 →
  yeon = 3 * gwi →
  bom + gwi + yeon = 1660 →
  bom = 300 := by
sorry

end bom_watermelon_seeds_l3261_326110


namespace jenny_distance_difference_l3261_326177

theorem jenny_distance_difference (run_distance walk_distance : ℝ) 
  (h1 : run_distance = 0.6)
  (h2 : walk_distance = 0.4) :
  run_distance - walk_distance = 0.2 := by
  sorry

end jenny_distance_difference_l3261_326177


namespace sequence_is_increasing_l3261_326176

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem sequence_is_increasing (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) - a n - 2 = 0) : 
  is_increasing a :=
sorry

end sequence_is_increasing_l3261_326176


namespace magic_square_sum_l3261_326134

theorem magic_square_sum (b c d e g h : ℕ) : 
  b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ g > 0 ∧ h > 0 →
  30 * b * c = d * e * 3 →
  30 * b * c = g * h * 3 →
  30 * b * c = 30 * e * 3 →
  30 * b * c = b * e * h →
  30 * b * c = c * 3 * 3 →
  30 * b * c = 30 * e * g →
  30 * b * c = c * e * 3 →
  (∃ g₁ g₂ : ℕ, g = g₁ ∨ g = g₂) →
  g₁ + g₂ = 25 :=
by sorry

end magic_square_sum_l3261_326134


namespace inequality_proof_l3261_326162

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1 / (2 * b^2)) * (b + 1 / (2 * a^2)) ≥ 25 / 4 := by
  sorry

end inequality_proof_l3261_326162


namespace min_value_rational_function_l3261_326194

theorem min_value_rational_function :
  ∀ x : ℝ, (3 * x^2 + 6 * x + 5) / (0.5 * x^2 + x + 1) ≥ 4 := by
  sorry

end min_value_rational_function_l3261_326194


namespace total_distance_traveled_l3261_326172

/-- Calculate the total distance traveled given cycling, walking, and jogging durations and speeds -/
theorem total_distance_traveled
  (cycling_time : ℚ) (cycling_speed : ℚ)
  (walking_time : ℚ) (walking_speed : ℚ)
  (jogging_time : ℚ) (jogging_speed : ℚ)
  (h1 : cycling_time = 20 / 60)
  (h2 : cycling_speed = 12)
  (h3 : walking_time = 40 / 60)
  (h4 : walking_speed = 3)
  (h5 : jogging_time = 50 / 60)
  (h6 : jogging_speed = 7) :
  let total_distance := cycling_time * cycling_speed + walking_time * walking_speed + jogging_time * jogging_speed
  ∃ ε > 0, |total_distance - 11.8333| < ε :=
sorry

#eval (20/60 : ℚ) * 12 + (40/60 : ℚ) * 3 + (50/60 : ℚ) * 7

end total_distance_traveled_l3261_326172


namespace circle_equation_satisfies_conditions_l3261_326199

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def PointOnLine (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

theorem circle_equation_satisfies_conditions :
  ∃ (h k r : ℝ),
    -- The circle's equation
    (∀ x y, CircleEquation h k r x y ↔ (x - 2)^2 + (y + 1)^2 = 5) ∧
    -- The center lies on the line 3x + y - 5 = 0
    PointOnLine 3 1 (-5) h k ∧
    -- The circle passes through (0, 0)
    CircleEquation h k r 0 0 ∧
    -- The circle passes through (4, 0)
    CircleEquation h k r 4 0 :=
by sorry

end circle_equation_satisfies_conditions_l3261_326199


namespace range_of_a_l3261_326140

theorem range_of_a (x a : ℝ) : 
  (∀ x, (x - 1) / (x - 3) < 0 → |x - a| < 2) ∧ 
  (∃ x, |x - a| < 2 ∧ (x - 1) / (x - 3) ≥ 0) →
  a ∈ Set.Icc 1 3 :=
by sorry

end range_of_a_l3261_326140


namespace equation_solution_l3261_326163

theorem equation_solution (y : ℚ) : 
  (y ≠ 5) → (y ≠ (3/2 : ℚ)) → 
  ((y^2 - 12*y + 35) / (y - 5) + (2*y^2 + 9*y - 18) / (2*y - 3) = 0) → 
  y = (1/2 : ℚ) := by
sorry

end equation_solution_l3261_326163


namespace distinct_values_of_binomial_sum_l3261_326184

theorem distinct_values_of_binomial_sum : ∃ (S : Finset ℕ),
  (∀ r : ℕ, r > 0 ∧ r + 1 ≤ 10 ∧ 17 - r ≤ 10 →
    (Nat.choose 10 (r + 1) + Nat.choose 10 (17 - r)) ∈ S) ∧
  Finset.card S = 2 := by
  sorry

end distinct_values_of_binomial_sum_l3261_326184


namespace regions_theorem_l3261_326118

/-- The number of regions formed by n lines in a plane -/
def total_regions (n : ℕ) : ℚ :=
  (n^2 + n + 2) / 2

/-- The number of bounded regions formed by n lines in a plane -/
def bounded_regions (n : ℕ) : ℚ :=
  (n^2 - 3*n + 2) / 2

/-- Theorem stating the formulas for total and bounded regions -/
theorem regions_theorem (n : ℕ) :
  (total_regions n = (n^2 + n + 2) / 2) ∧
  (bounded_regions n = (n^2 - 3*n + 2) / 2) := by
  sorry

end regions_theorem_l3261_326118


namespace linear_equation_condition_l3261_326190

theorem linear_equation_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ k m n : ℝ, (a - 2) * x^(|a| - 1) + 3 * y = k * x + m * y + n) → 
  a = -2 :=
by sorry

end linear_equation_condition_l3261_326190


namespace repeating_decimal_equals_fraction_l3261_326115

/-- Represents a repeating decimal where the whole number part is 7 and the repeating part is 182. -/
def repeating_decimal : ℚ := 7 + 182 / 999

/-- The fraction representation of the repeating decimal. -/
def fraction : ℚ := 7175 / 999

/-- Theorem stating that the repeating decimal 7.182182... is equal to the fraction 7175/999. -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end repeating_decimal_equals_fraction_l3261_326115


namespace events_mutually_exclusive_not_contradictory_l3261_326101

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the events
def EventA (d : Distribution) : Prop := d Person.A = Card.Red
def EventB (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem events_mutually_exclusive_not_contradictory :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventA d ∧ EventB d)) ∧
  -- The events are not contradictory
  (∃ d : Distribution, EventA d) ∧
  (∃ d : Distribution, EventB d) ∧
  -- There exists a distribution where neither event occurs
  (∃ d : Distribution, ¬EventA d ∧ ¬EventB d) :=
sorry

end events_mutually_exclusive_not_contradictory_l3261_326101


namespace intersection_of_A_and_B_l3261_326127

-- Define the sets A and B
def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-2) 1 := by sorry

end intersection_of_A_and_B_l3261_326127


namespace probability_two_even_balls_l3261_326158

def total_balls : ℕ := 17
def even_balls : ℕ := 8

theorem probability_two_even_balls :
  (even_balls : ℚ) / total_balls * (even_balls - 1) / (total_balls - 1) = 7 / 34 := by
  sorry

end probability_two_even_balls_l3261_326158


namespace festival_attendance_theorem_l3261_326139

/-- Represents the attendance for each day of a four-day music festival --/
structure FestivalAttendance where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Calculates the total attendance for all four days --/
def totalAttendance (attendance : FestivalAttendance) : ℕ :=
  attendance.day1 + attendance.day2 + attendance.day3 + attendance.day4

/-- Theorem stating that the total attendance for the festival is 3600 --/
theorem festival_attendance_theorem (attendance : FestivalAttendance) :
  (attendance.day2 = attendance.day1 / 2) →
  (attendance.day3 = attendance.day1 * 3) →
  (attendance.day4 = attendance.day2 * 2) →
  (totalAttendance attendance = 3600) :=
by
  sorry

#check festival_attendance_theorem

end festival_attendance_theorem_l3261_326139


namespace inequality_solution_range_l3261_326104

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 := by
  sorry

end inequality_solution_range_l3261_326104


namespace equal_ratios_imply_p_equals_13_l3261_326179

theorem equal_ratios_imply_p_equals_13 
  (a b c p : ℝ) 
  (h1 : (5 : ℝ) / (a + b) = p / (a + c)) 
  (h2 : p / (a + c) = (8 : ℝ) / (c - b)) : 
  p = 13 := by
  sorry

end equal_ratios_imply_p_equals_13_l3261_326179


namespace white_balls_remain_odd_one_white_ball_left_l3261_326178

/-- Represents the state of the bag with black and white balls -/
structure BagState where
  white : Nat
  black : Nat

/-- The process of drawing two balls and applying the rules -/
def drawBalls (state : BagState) : BagState :=
  sorry

/-- Predicate to check if the process has ended (0 or 1 ball left) -/
def processEnded (state : BagState) : Prop :=
  state.white + state.black ≤ 1

/-- Theorem stating that the number of white balls remains odd throughout the process -/
theorem white_balls_remain_odd (initial : BagState) (final : BagState) 
    (h_initial : initial.white = 2007 ∧ initial.black = 2007)
    (h_process : final = drawBalls initial ∨ (∃ intermediate, final = drawBalls intermediate ∧ ¬processEnded intermediate)) :
  Odd final.white :=
  sorry

/-- Main theorem proving that one white ball is left at the end -/
theorem one_white_ball_left (initial : BagState) (final : BagState)
    (h_initial : initial.white = 2007 ∧ initial.black = 2007)
    (h_process : final = drawBalls initial ∨ (∃ intermediate, final = drawBalls intermediate ∧ ¬processEnded intermediate))
    (h_ended : processEnded final) :
  final.white = 1 ∧ final.black = 0 :=
  sorry

end white_balls_remain_odd_one_white_ball_left_l3261_326178


namespace division_of_fraction_by_integer_l3261_326108

theorem division_of_fraction_by_integer : 
  (3 : ℚ) / 7 / 4 = (3 : ℚ) / 28 := by
sorry

end division_of_fraction_by_integer_l3261_326108


namespace distribute_five_balls_four_boxes_l3261_326195

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    with at least one box remaining empty -/
def distributeWithEmptyBox (n k : ℕ) : ℕ :=
  if n < k then distribute n k else distribute n k

theorem distribute_five_balls_four_boxes :
  distributeWithEmptyBox 5 4 = 1024 := by sorry

end distribute_five_balls_four_boxes_l3261_326195


namespace min_nSn_arithmetic_sequence_l3261_326183

def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

theorem min_nSn_arithmetic_sequence (a₁ d : ℤ) :
  (arithmetic_sequence a₁ d 7 = 5) →
  (sum_arithmetic_sequence a₁ d 5 = -55) →
  (∀ n : ℕ, n > 0 → n * (sum_arithmetic_sequence a₁ d n) ≥ -343) ∧
  (∃ n : ℕ, n > 0 ∧ n * (sum_arithmetic_sequence a₁ d n) = -343) :=
sorry

end min_nSn_arithmetic_sequence_l3261_326183


namespace three_digit_one_more_than_multiple_l3261_326142

/-- The least common multiple of 2, 3, 5, and 7 -/
def lcm_2357 : ℕ := 210

/-- Checks if a number is a three-digit positive integer -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Checks if a number is one more than a multiple of 2, 3, 5, and 7 -/
def is_one_more_than_multiple (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * lcm_2357 + 1

theorem three_digit_one_more_than_multiple :
  ∀ n : ℕ, is_three_digit n ∧ is_one_more_than_multiple n ↔ n = 211 ∨ n = 421 := by
  sorry

end three_digit_one_more_than_multiple_l3261_326142


namespace zeros_after_one_in_10000_pow_50_l3261_326117

theorem zeros_after_one_in_10000_pow_50 :
  ∃ (n : ℕ), 10000^50 = 10^n ∧ n = 200 := by
  sorry

end zeros_after_one_in_10000_pow_50_l3261_326117


namespace boat_speed_in_still_water_l3261_326105

/-- Represents the speed of a boat in still water and a stream -/
structure BoatAndStreamSpeeds where
  boat : ℝ
  stream : ℝ

/-- Represents the time taken to travel upstream and downstream -/
structure TravelTimes where
  downstream : ℝ
  upstream : ℝ

/-- The problem statement -/
theorem boat_speed_in_still_water 
  (speeds : BoatAndStreamSpeeds)
  (times : TravelTimes)
  (h1 : speeds.stream = 13)
  (h2 : times.upstream = 2 * times.downstream)
  (h3 : (speeds.boat + speeds.stream) * times.downstream = 
        (speeds.boat - speeds.stream) * times.upstream) :
  speeds.boat = 39 := by
sorry

end boat_speed_in_still_water_l3261_326105


namespace range_of_a_l3261_326149

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + a| + |x - 1| + a < 2011) ↔ a < 1005 :=
sorry

end range_of_a_l3261_326149


namespace expression_value_l3261_326186

theorem expression_value (b : ℚ) (h : b = 1/3) : 
  (3 * b⁻¹ - b⁻¹ / 3) / b^2 = 72 := by sorry

end expression_value_l3261_326186


namespace tan_15_30_product_equals_two_l3261_326187

theorem tan_15_30_product_equals_two :
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 :=
by
  have tan_45_eq_1 : Real.tan (45 * π / 180) = 1 := by sorry
  have tan_sum_15_30 : Real.tan ((15 + 30) * π / 180) = 
    (Real.tan (15 * π / 180) + Real.tan (30 * π / 180)) / 
    (1 - Real.tan (15 * π / 180) * Real.tan (30 * π / 180)) := by sorry
  sorry

end tan_15_30_product_equals_two_l3261_326187


namespace sunglasses_and_caps_probability_l3261_326145

theorem sunglasses_and_caps_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (prob_sunglasses_also_cap : ℚ) :
  total_sunglasses = 80 →
  total_caps = 45 →
  prob_sunglasses_also_cap = 3/8 →
  (total_sunglasses * prob_sunglasses_also_cap : ℚ) / total_caps = 2/3 :=
by sorry

end sunglasses_and_caps_probability_l3261_326145


namespace reverse_digits_when_multiplied_by_nine_l3261_326159

theorem reverse_digits_when_multiplied_by_nine : ∃ n : ℕ, 
  (100000 ≤ n ∧ n < 1000000) ∧  -- six-digit number
  (n * 9 = 
    ((n % 10) * 100000 + 
     ((n / 10) % 10) * 10000 + 
     ((n / 100) % 10) * 1000 + 
     ((n / 1000) % 10) * 100 + 
     ((n / 10000) % 10) * 10 + 
     (n / 100000))) :=
by
  -- Proof goes here
  sorry

end reverse_digits_when_multiplied_by_nine_l3261_326159


namespace johns_croissants_l3261_326129

theorem johns_croissants :
  ∀ (c : ℕ) (k : ℕ),
  c + k = 5 →
  (88 * c + 44 * k) % 100 = 0 →
  c = 3 :=
by
  sorry

end johns_croissants_l3261_326129


namespace paint_usage_l3261_326152

/-- Calculates the total amount of paint used by an artist for large and small canvases -/
theorem paint_usage (large_paint : ℕ) (small_paint : ℕ) (large_count : ℕ) (small_count : ℕ) :
  large_paint = 3 →
  small_paint = 2 →
  large_count = 3 →
  small_count = 4 →
  large_paint * large_count + small_paint * small_count = 17 := by
  sorry


end paint_usage_l3261_326152


namespace work_completion_time_l3261_326182

theorem work_completion_time (y_completion_time x_remaining_time : ℕ) 
  (y_worked_days : ℕ) (h1 : y_completion_time = 16) (h2 : y_worked_days = 10) 
  (h3 : x_remaining_time = 9) : 
  (y_completion_time * x_remaining_time) / 
  (y_completion_time - y_worked_days) = 24 := by
  sorry

end work_completion_time_l3261_326182


namespace regular_17gon_symmetry_sum_l3261_326122

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry for a regular polygon (in degrees) -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_17gon_symmetry_sum :
  ∀ (p : RegularPolygon 17),
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 649 / 17 := by
  sorry

end regular_17gon_symmetry_sum_l3261_326122


namespace congruent_rectangle_perimeter_l3261_326167

/-- Given a rectangle of width y and height 2y divided into a square of side x
    and four congruent rectangles, the perimeter of one of the congruent rectangles
    is 3y - 2x. -/
theorem congruent_rectangle_perimeter
  (y : ℝ) (x : ℝ)
  (h1 : y > 0)
  (h2 : x > 0)
  (h3 : x < y)
  (h4 : x < 2*y) :
  ∃ (l w : ℝ),
    l > 0 ∧ w > 0 ∧
    x + 2*l = y ∧
    x + 2*w = 2*y ∧
    2*l + 2*w = 3*y - 2*x :=
by sorry

end congruent_rectangle_perimeter_l3261_326167


namespace igneous_sedimentary_ratio_l3261_326130

/-- Represents Cliff's rock collection --/
structure RockCollection where
  igneous : ℕ
  sedimentary : ℕ
  shinyIgneous : ℕ
  shinySedimentary : ℕ

/-- The properties of Cliff's rock collection --/
def isValidCollection (c : RockCollection) : Prop :=
  c.shinyIgneous = (2 * c.igneous) / 3 ∧
  c.shinySedimentary = c.sedimentary / 5 ∧
  c.shinyIgneous = 40 ∧
  c.igneous + c.sedimentary = 180

/-- The theorem stating the ratio of igneous to sedimentary rocks --/
theorem igneous_sedimentary_ratio (c : RockCollection) 
  (h : isValidCollection c) : c.igneous * 2 = c.sedimentary := by
  sorry


end igneous_sedimentary_ratio_l3261_326130


namespace three_minus_a_equals_four_l3261_326157

theorem three_minus_a_equals_four (a b : ℝ) 
  (eq1 : 3 + a = 4 - b) 
  (eq2 : 4 + b = 7 + a) : 
  3 - a = 4 := by
sorry

end three_minus_a_equals_four_l3261_326157


namespace youth_gathering_count_l3261_326151

/-- The number of youths at a gathering, given the conditions from the problem. -/
def total_youths (male_youths : ℕ) : ℕ := 2 * male_youths + 12

/-- The theorem stating the total number of youths at the gathering. -/
theorem youth_gathering_count : 
  ∃ (male_youths : ℕ), 
    (male_youths : ℚ) / (total_youths male_youths : ℚ) = 9 / 20 ∧ 
    total_youths male_youths = 120 := by
  sorry


end youth_gathering_count_l3261_326151


namespace total_earnings_l3261_326128

def markese_earnings : ℕ := 16
def difference : ℕ := 5

theorem total_earnings : 
  markese_earnings + (markese_earnings + difference) = 37 := by
  sorry

end total_earnings_l3261_326128


namespace bella_win_probability_l3261_326155

theorem bella_win_probability (lose_prob : ℚ) (no_tie : Bool) : lose_prob = 5/11 ∧ no_tie = true → 1 - lose_prob = 6/11 := by
  sorry

end bella_win_probability_l3261_326155


namespace tangent_ratio_given_sine_condition_l3261_326198

theorem tangent_ratio_given_sine_condition (α : Real) 
  (h : 5 * Real.sin (2 * α) = Real.sin (2 * Real.pi / 180)) : 
  Real.tan (α + Real.pi / 180) / Real.tan (α - Real.pi / 180) = -3/2 := by
  sorry

end tangent_ratio_given_sine_condition_l3261_326198


namespace students_agreement_count_l3261_326180

theorem students_agreement_count :
  let third_grade_count : ℕ := 154
  let fourth_grade_count : ℕ := 237
  third_grade_count + fourth_grade_count = 391 :=
by
  sorry

end students_agreement_count_l3261_326180


namespace bus_passengers_second_stop_l3261_326181

/-- Given a bus with the following properties:
  * 23 rows of 4 seats each
  * 16 people board at the start
  * At the first stop, 15 people board and 3 get off
  * At the second stop, 17 people board
  * There are 57 empty seats after the second stop
  Prove that 10 people got off at the second stop. -/
theorem bus_passengers_second_stop 
  (total_seats : ℕ) 
  (initial_passengers : ℕ) 
  (first_stop_on : ℕ) 
  (first_stop_off : ℕ) 
  (second_stop_on : ℕ) 
  (empty_seats_after_second : ℕ) 
  (h1 : total_seats = 23 * 4)
  (h2 : initial_passengers = 16)
  (h3 : first_stop_on = 15)
  (h4 : first_stop_off = 3)
  (h5 : second_stop_on = 17)
  (h6 : empty_seats_after_second = 57) :
  ∃ (second_stop_off : ℕ), 
    second_stop_off = 10 ∧ 
    empty_seats_after_second = total_seats - (initial_passengers + first_stop_on - first_stop_off + second_stop_on - second_stop_off) :=
by sorry

end bus_passengers_second_stop_l3261_326181


namespace quadratic_no_real_roots_l3261_326164

/-- If a, b, and c form a geometric sequence, then ax^2 + bx + c has no real roots -/
theorem quadratic_no_real_roots (a b c : ℝ) (h : b^2 = a*c) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 := by
sorry

end quadratic_no_real_roots_l3261_326164


namespace trajectory_and_no_line_exist_l3261_326126

-- Define the points and vectors
def A : ℝ × ℝ := (8, 0)
def Q : ℝ × ℝ := (-1, 0)

-- Define the conditions
def condition1 (B P : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BP := (P.1 - B.1, P.2 - B.2)
  AB.1 * BP.1 + AB.2 * BP.2 = 0

def condition2 (B C P : ℝ × ℝ) : Prop :=
  (C.1 - B.1, C.2 - B.2) = (P.1 - C.1, P.2 - C.2)

def on_y_axis (B : ℝ × ℝ) : Prop := B.1 = 0
def on_x_axis (C : ℝ × ℝ) : Prop := C.2 = 0

-- Define the trajectory
def trajectory (P : ℝ × ℝ) : Prop := P.2^2 = -4 * P.1

-- Define the line passing through A
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k * x - 8 * k

-- Define the dot product condition
def dot_product_condition (M N : ℝ × ℝ) : Prop :=
  let QM := (M.1 - Q.1, M.2 - Q.2)
  let QN := (N.1 - Q.1, N.2 - Q.2)
  QM.1 * QN.1 + QM.2 * QN.2 = 97

-- The main theorem
theorem trajectory_and_no_line_exist :
  ∀ B C P, on_y_axis B → on_x_axis C →
  condition1 B P → condition2 B C P →
  (trajectory P ∧
   ¬∃ k M N, line_through_A k M.1 M.2 ∧ line_through_A k N.1 N.2 ∧
              trajectory M ∧ trajectory N ∧ dot_product_condition M N) :=
sorry

end trajectory_and_no_line_exist_l3261_326126


namespace inverse_expression_equals_one_fifth_l3261_326109

theorem inverse_expression_equals_one_fifth :
  (3 - 4 * (3 - 5)⁻¹)⁻¹ = (1 : ℝ) / 5 := by
  sorry

end inverse_expression_equals_one_fifth_l3261_326109


namespace unique_solution_exponential_equation_l3261_326124

theorem unique_solution_exponential_equation :
  ∃! y : ℝ, (10 : ℝ)^(2*y) * (100 : ℝ)^y = (1000 : ℝ)^3 * (10 : ℝ)^y :=
by
  sorry

end unique_solution_exponential_equation_l3261_326124


namespace square_sum_equals_4014_l3261_326132

theorem square_sum_equals_4014 (a : ℝ) (h : (2006 - a) * (2004 - a) = 2005) :
  (2006 - a)^2 + (2004 - a)^2 = 4014 := by
  sorry

end square_sum_equals_4014_l3261_326132


namespace brother_age_proof_l3261_326102

/-- Trevor's current age -/
def Trevor_current_age : ℕ := 11

/-- Trevor's future age when the condition is met -/
def Trevor_future_age : ℕ := 24

/-- Trevor's older brother's current age -/
def Brother_current_age : ℕ := 20

theorem brother_age_proof :
  (Trevor_future_age - Trevor_current_age = Brother_current_age - Trevor_current_age) ∧
  (Brother_current_age + (Trevor_future_age - Trevor_current_age) = 3 * Trevor_current_age) :=
by sorry

end brother_age_proof_l3261_326102


namespace sequence_terms_l3261_326196

def a (n : ℕ) : ℤ := (-1)^(n+1) * (3*n - 2)

theorem sequence_terms : 
  (a 1 = 1) ∧ (a 2 = -4) ∧ (a 3 = 7) ∧ (a 4 = -10) ∧ (a 5 = 13) := by
  sorry

end sequence_terms_l3261_326196


namespace max_y_value_l3261_326111

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = 8) : 
  y ≤ 43 ∧ ∃ (x₀ y₀ : ℤ), x₀ * y₀ + 7 * x₀ + 6 * y₀ = 8 ∧ y₀ = 43 :=
by sorry

end max_y_value_l3261_326111


namespace divisibility_prime_factorization_l3261_326147

theorem divisibility_prime_factorization (a b : ℕ) : 
  (a ∣ b) ↔ (∀ p : ℕ, ∀ k : ℕ, Prime p → (p^k ∣ a) → (p^k ∣ b)) :=
by sorry

end divisibility_prime_factorization_l3261_326147


namespace M_subset_N_l3261_326171

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def N : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem M_subset_N : M ⊆ N := by sorry

end M_subset_N_l3261_326171


namespace three_digit_factorial_sum_l3261_326125

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem three_digit_factorial_sum : ∃ a b c : ℕ, 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  100 * a + 10 * b + c = factorial a + factorial b + factorial c := by
  sorry

end three_digit_factorial_sum_l3261_326125


namespace train_speed_l3261_326148

/-- The speed of a train given its length, time to cross a walking man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 400 →
  crossing_time = 23.998 →
  man_speed_kmh = 3 →
  ∃ (train_speed_kmh : ℝ), 
    (train_speed_kmh ≥ 63.004 ∧ train_speed_kmh ≤ 63.006) ∧
    train_speed_kmh * 1000 / 3600 = 
      train_length / crossing_time + man_speed_kmh * 1000 / 3600 :=
by sorry

end train_speed_l3261_326148


namespace simplify_fraction_product_l3261_326143

theorem simplify_fraction_product : 8 * (15 / 4) * (-25 / 45) = -50 / 3 := by
  sorry

end simplify_fraction_product_l3261_326143


namespace spiral_stripe_length_l3261_326166

/-- The length of a spiral stripe on a right circular cylinder -/
theorem spiral_stripe_length (base_circumference height : ℝ) (h1 : base_circumference = 18) (h2 : height = 8) :
  let stripe_length := Real.sqrt (height^2 + (2 * base_circumference)^2)
  stripe_length = Real.sqrt 1360 := by
sorry

end spiral_stripe_length_l3261_326166


namespace cost_per_item_jings_purchase_l3261_326185

/-- Given a total cost and number of identical items, prove that the cost per item
    is equal to the total cost divided by the number of items. -/
theorem cost_per_item (total_cost : ℝ) (num_items : ℕ) (h : num_items > 0) :
  let cost_per_item := total_cost / num_items
  cost_per_item = total_cost / num_items :=
by
  sorry

/-- For Jing's purchase of 8 identical items with a total cost of $26,
    prove that the cost per item is $26 divided by 8. -/
theorem jings_purchase :
  let total_cost : ℝ := 26
  let num_items : ℕ := 8
  let cost_per_item := total_cost / num_items
  cost_per_item = 26 / 8 :=
by
  sorry

end cost_per_item_jings_purchase_l3261_326185


namespace least_cookies_l3261_326173

theorem least_cookies (b : ℕ) : 
  b > 0 ∧ 
  b % 6 = 5 ∧ 
  b % 8 = 3 ∧ 
  b % 9 = 7 ∧
  (∀ c : ℕ, c > 0 ∧ c % 6 = 5 ∧ c % 8 = 3 ∧ c % 9 = 7 → b ≤ c) → 
  b = 179 := by
sorry

end least_cookies_l3261_326173


namespace average_difference_l3261_326174

theorem average_difference (x : ℝ) : (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 8 ↔ x = 16 := by
  sorry

end average_difference_l3261_326174


namespace range_of_a_l3261_326123

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + a > 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a :
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∃ a : ℝ, a ≤ 0 ∨ (1/4 < a ∧ a < 4)) :=
sorry

end range_of_a_l3261_326123


namespace math_competition_participation_l3261_326131

theorem math_competition_participation (total_students : ℕ) (non_participants : ℕ) 
  (h1 : total_students = 39) (h2 : non_participants = 26) :
  (total_students - non_participants : ℚ) / total_students = 1 / 3 := by
  sorry

end math_competition_participation_l3261_326131


namespace problem_solution_l3261_326188

theorem problem_solution (x : ℚ) : (1/2 * (12*x + 3) = 3*x + 2) → x = 1/6 := by
  sorry

end problem_solution_l3261_326188


namespace sqrt_eleven_between_integers_l3261_326116

theorem sqrt_eleven_between_integers (a : ℤ) : 3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4 →
  (↑a < Real.sqrt 11 ∧ Real.sqrt 11 < ↑a + 1) ↔ a = 3 := by
  sorry

end sqrt_eleven_between_integers_l3261_326116


namespace course_selection_ways_l3261_326193

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem course_selection_ways (type_a type_b total : ℕ) : 
  type_a = 3 → type_b = 4 → total = 3 →
  (choose type_a 2 * choose type_b 1 + choose type_a 1 * choose type_b 2) = 30 := by
  sorry

end course_selection_ways_l3261_326193


namespace bob_baked_36_more_l3261_326138

/-- The number of additional peanut butter cookies Bob baked after the accident -/
def bob_additional_cookies (alice_initial : ℕ) (bob_initial : ℕ) (lost : ℕ) (alice_additional : ℕ) (final_total : ℕ) : ℕ :=
  final_total - ((alice_initial + bob_initial - lost) + alice_additional)

/-- Theorem stating that Bob baked 36 additional cookies given the problem conditions -/
theorem bob_baked_36_more (alice_initial bob_initial lost alice_additional final_total : ℕ) 
  (h1 : alice_initial = 74)
  (h2 : bob_initial = 7)
  (h3 : lost = 29)
  (h4 : alice_additional = 5)
  (h5 : final_total = 93) :
  bob_additional_cookies alice_initial bob_initial lost alice_additional final_total = 36 := by
  sorry

end bob_baked_36_more_l3261_326138


namespace line_intersects_ellipse_l3261_326175

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

/-- The equation of the line -/
def line (m x y : ℝ) : Prop := (m+2)*x - (m+4)*y + 2-m = 0

/-- Theorem stating that the line always intersects the ellipse -/
theorem line_intersects_ellipse :
  ∀ m : ℝ, ∃ x y : ℝ, ellipse x y ∧ line m x y := by sorry

end line_intersects_ellipse_l3261_326175


namespace equation_solution_l3261_326154

theorem equation_solution : 
  ∃ x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 2) ∧ x = -1/3 := by
  sorry

end equation_solution_l3261_326154


namespace sliding_chord_annulus_area_l3261_326103

/-- The area of the annulus formed by a sliding chord on a circle -/
theorem sliding_chord_annulus_area
  (R : ℝ) -- radius of the outer circle
  (a b : ℝ) -- distances from point C to ends A and B of the chord
  (h1 : R > 0) -- radius is positive
  (h2 : a > 0) -- distance a is positive
  (h3 : b > 0) -- distance b is positive
  (h4 : a + b ≤ 2 * R) -- chord length constraint
  : ∃ (r : ℝ), -- radius of the inner circle
    r^2 = R^2 - a * b ∧ 
    π * R^2 - π * r^2 = π * a * b :=
by sorry

end sliding_chord_annulus_area_l3261_326103


namespace book_sale_loss_l3261_326100

/-- Given that the cost price of 15 books equals the selling price of 20 books,
    prove that there is a 25% loss. -/
theorem book_sale_loss (C S : ℝ) (h : 15 * C = 20 * S) :
  (C - S) / C * 100 = 25 :=
sorry

end book_sale_loss_l3261_326100


namespace triangle_side_relation_l3261_326197

/-- Given a triangle ABC with side lengths a, b, c satisfying the equation
    a² - 16b² - c² + 6ab + 10bc = 0, prove that a + c = 2b. -/
theorem triangle_side_relation (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a^2 - 16*b^2 - c^2 + 6*a*b + 10*b*c = 0) :
  a + c = 2*b :=
sorry

end triangle_side_relation_l3261_326197


namespace boat_speed_in_still_water_l3261_326192

/-- Proves that the speed of a boat in still water is 24 km/hr, given the speed of the stream and the boat's downstream travel information. -/
theorem boat_speed_in_still_water :
  let stream_speed : ℝ := 4
  let downstream_distance : ℝ := 140
  let downstream_time : ℝ := 5
  let downstream_speed : ℝ := downstream_distance / downstream_time
  let boat_speed_still_water : ℝ := downstream_speed - stream_speed
  boat_speed_still_water = 24 :=
by sorry

end boat_speed_in_still_water_l3261_326192


namespace equal_numbers_l3261_326144

theorem equal_numbers (x y z : ℕ) 
  (h1 : x ∣ Nat.gcd y z)
  (h2 : y ∣ Nat.gcd x z)
  (h3 : z ∣ Nat.gcd x y)
  (h4 : x ∣ Nat.lcm y z)
  (h5 : y ∣ Nat.lcm x z)
  (h6 : z ∣ Nat.lcm x y) :
  x = y ∧ y = z := by
  sorry

end equal_numbers_l3261_326144


namespace best_athlete_is_A_l3261_326169

structure Athlete where
  name : String
  average : ℝ
  variance : ℝ

def betterPerformance (a b : Athlete) : Prop :=
  a.average > b.average ∨ (a.average = b.average ∧ a.variance < b.variance)

theorem best_athlete_is_A :
  let A := Athlete.mk "A" 185 3.6
  let B := Athlete.mk "B" 180 3.6
  let C := Athlete.mk "C" 185 7.4
  let D := Athlete.mk "D" 180 8.1
  ∀ x ∈ [B, C, D], betterPerformance A x := by
  sorry

end best_athlete_is_A_l3261_326169


namespace min_digit_divisible_by_72_l3261_326141

theorem min_digit_divisible_by_72 :
  ∃ (x : ℕ), x < 10 ∧ (983480 + x) % 72 = 0 ∧
  ∀ (y : ℕ), y < x → (983480 + y) % 72 ≠ 0 :=
by sorry

end min_digit_divisible_by_72_l3261_326141


namespace tangent_line_sum_l3261_326112

/-- Given a function f where the tangent line at x=2 is 2x+y-3=0, prove f(2) + f'(2) = -3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : DifferentiableAt ℝ f 2) 
  (h_tangent : ∀ x y, y = f x → (x = 2 → 2*x + y - 3 = 0)) : 
  f 2 + deriv f 2 = -3 := by
  sorry

end tangent_line_sum_l3261_326112


namespace carries_trip_l3261_326165

theorem carries_trip (day1 : ℕ) (day3 : ℕ) (day4 : ℕ) (charge_distance : ℕ) (charge_count : ℕ) : 
  day1 = 135 → 
  day3 = 159 → 
  day4 = 189 → 
  charge_distance = 106 → 
  charge_count = 7 → 
  ∃ day2 : ℕ, day2 - day1 = 124 ∧ day1 + day2 + day3 + day4 = charge_distance * charge_count :=
by sorry

end carries_trip_l3261_326165
