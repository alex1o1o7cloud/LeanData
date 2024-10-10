import Mathlib

namespace tan_alpha_plus_pi_fourth_l3459_345905

theorem tan_alpha_plus_pi_fourth (x y : ℝ) (α : ℝ) : 
  (x < 0 ∧ y > 0) →  -- terminal side in second quadrant
  (3 * x + 4 * y = 0) →  -- m ⊥ OA
  (Real.tan α = -3/4) →  -- derived from m ⊥ OA
  Real.tan (α + π/4) = 1/7 := by
  sorry

end tan_alpha_plus_pi_fourth_l3459_345905


namespace train_length_l3459_345961

/-- Calculates the length of a train given the time it takes to cross a bridge and a lamp post -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ) :
  bridge_length = 1500 →
  bridge_time = 70 →
  post_time = 20 →
  ∃ (train_length : ℝ),
    train_length / post_time = (train_length + bridge_length) / bridge_time ∧
    train_length = 600 := by
  sorry

end train_length_l3459_345961


namespace fruit_cost_problem_l3459_345967

/-- The cost of fruits problem -/
theorem fruit_cost_problem (apple_price pear_price mango_price : ℝ) 
  (h1 : 5 * apple_price + 4 * pear_price = 48)
  (h2 : 2 * apple_price + 3 * mango_price = 33)
  (h3 : mango_price = pear_price + 2.5) :
  3 * apple_price + 3 * pear_price = 31.5 := by
  sorry

end fruit_cost_problem_l3459_345967


namespace quadratic_inequality_l3459_345909

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by sorry

end quadratic_inequality_l3459_345909


namespace max_sum_is_24_l3459_345940

def numbers : Finset ℕ := {1, 4, 7, 10, 13}

def valid_arrangement (a b c d e : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ d ∈ numbers ∧ e ∈ numbers ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  a + b + e = a + c + e

def sum_of_arrangement (a b c d e : ℕ) : ℕ := a + b + e

theorem max_sum_is_24 :
  ∀ a b c d e : ℕ, valid_arrangement a b c d e →
    sum_of_arrangement a b c d e ≤ 24 :=
sorry

end max_sum_is_24_l3459_345940


namespace power_mod_thirteen_l3459_345992

theorem power_mod_thirteen : 2^2010 ≡ 12 [ZMOD 13] := by sorry

end power_mod_thirteen_l3459_345992


namespace max_value_when_a_is_one_range_of_a_for_two_roots_l3459_345971

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 + 4 * x - 3 - a

-- Theorem for the maximum value of f when a = 1
theorem max_value_when_a_is_one :
  ∃ (max : ℝ), max = 2 ∧ ∀ x ∈ Set.Icc (-1) 1, f 1 x ≤ max :=
sorry

-- Theorem for the range of a when f has two distinct roots
theorem range_of_a_for_two_roots :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 
    a ∈ Set.Ioi 0 ∪ Set.Ioo (-1) 0 ∪ Set.Iic (-2) :=
sorry

end max_value_when_a_is_one_range_of_a_for_two_roots_l3459_345971


namespace jordan_income_proof_l3459_345948

-- Define the daily incomes and work days
def terry_daily_income : ℝ := 24
def work_days : ℕ := 7
def weekly_income_difference : ℝ := 42

-- Define Jordan's daily income as a variable
def jordan_daily_income : ℝ := 30

-- Theorem to prove
theorem jordan_income_proof :
  jordan_daily_income * work_days - terry_daily_income * work_days = weekly_income_difference :=
by sorry

end jordan_income_proof_l3459_345948


namespace range_of_f_l3459_345924

noncomputable def f (x : ℝ) : ℝ := x + 1 / (2 * x)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≤ -Real.sqrt 2 ∨ y ≥ Real.sqrt 2 := by
  sorry

end range_of_f_l3459_345924


namespace geometric_sequence_middle_term_l3459_345953

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (h_seq : ∃ r : ℝ, b = a * r ∧ c = b * r) 
  (h_a : a = 7 + 4 * Real.sqrt 3) 
  (h_c : c = 7 - 4 * Real.sqrt 3) : 
  b = 1 ∨ b = -1 := by
sorry

end geometric_sequence_middle_term_l3459_345953


namespace appropriate_sampling_methods_l3459_345987

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents income levels --/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents a community with different income levels --/
structure Community where
  highIncome : Nat
  middleIncome : Nat
  lowIncome : Nat

/-- Represents a school class --/
structure SchoolClass where
  totalStudents : Nat
  specialtyType : String

/-- Determines the most appropriate sampling method for a community survey --/
def communitySamplingMethod (community : Community) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Determines the most appropriate sampling method for a school class survey --/
def schoolClassSamplingMethod (schoolClass : SchoolClass) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Theorem stating the appropriate sampling methods for the given surveys --/
theorem appropriate_sampling_methods
  (community : Community)
  (schoolClass : SchoolClass) :
  communitySamplingMethod {highIncome := 125, middleIncome := 280, lowIncome := 95} 100 = SamplingMethod.Stratified ∧
  schoolClassSamplingMethod {totalStudents := 15, specialtyType := "art"} 3 = SamplingMethod.SimpleRandom :=
  sorry

end appropriate_sampling_methods_l3459_345987


namespace cubes_with_five_neighbors_count_l3459_345969

/-- Represents a large cube assembled from unit cubes -/
structure LargeCube where
  sideLength : ℕ

/-- The number of unit cubes with exactly 4 neighbors in the large cube -/
def cubesWithFourNeighbors (c : LargeCube) : ℕ := 12 * (c.sideLength - 2)

/-- The number of unit cubes with exactly 5 neighbors in the large cube -/
def cubesWithFiveNeighbors (c : LargeCube) : ℕ := 6 * (c.sideLength - 2)^2

/-- Theorem stating the relationship between cubes with 4 and 5 neighbors -/
theorem cubes_with_five_neighbors_count (c : LargeCube) 
  (h : cubesWithFourNeighbors c = 132) : 
  cubesWithFiveNeighbors c = 726 := by
  sorry

end cubes_with_five_neighbors_count_l3459_345969


namespace triangle_abc_properties_l3459_345938

/-- Triangle ABC with given vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The altitude from a point to a line -/
def altitude (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Main theorem about the triangle ABC -/
theorem triangle_abc_properties :
  let t : Triangle := { A := (-2, 4), B := (-3, -1), C := (1, 3) }
  let alt_B_AC : ℝ → ℝ := altitude t.B (fun x => x - 1)  -- Line AC: y = x - 1
  ∀ x y, alt_B_AC x = y ↔ x + y - 2 = 0 ∧ triangleArea t = 8 := by sorry

end triangle_abc_properties_l3459_345938


namespace sum_of_digits_9_pow_1001_l3459_345951

theorem sum_of_digits_9_pow_1001 : ∃ (n : ℕ), 
  (9^1001 : ℕ) % 100 = n ∧ (n / 10 + n % 10 = 9) := by sorry

end sum_of_digits_9_pow_1001_l3459_345951


namespace diamond_six_three_l3459_345960

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := 4 * a - 2 * b

-- Theorem statement
theorem diamond_six_three : diamond 6 3 = 18 := by sorry

end diamond_six_three_l3459_345960


namespace train_length_calculation_l3459_345941

theorem train_length_calculation (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 ∧ crossing_time = 45 ∧ train_speed = 55.99999999999999 →
  2220 = train_speed * crossing_time - bridge_length :=
by
  sorry

end train_length_calculation_l3459_345941


namespace smallest_five_digit_negative_congruent_to_one_mod_seventeen_l3459_345985

theorem smallest_five_digit_negative_congruent_to_one_mod_seventeen :
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 1 [ZMOD 17] → n ≥ -10011 :=
by sorry

end smallest_five_digit_negative_congruent_to_one_mod_seventeen_l3459_345985


namespace T_is_three_rays_l3459_345994

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    (5 = p.1 + 3 ∧ 5 ≥ p.2 - 6) ∨
    (5 = p.2 - 6 ∧ 5 ≥ p.1 + 3) ∨
    (p.1 + 3 = p.2 - 6 ∧ 5 ≥ p.1 + 3)}

-- Define the three rays
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 11}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≤ 2 ∧ p.2 = 11}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 9 ∧ p.1 ≤ 2 ∧ p.2 ≤ 11}

-- Theorem statement
theorem T_is_three_rays : T = ray1 ∪ ray2 ∪ ray3 := by
  sorry

end T_is_three_rays_l3459_345994


namespace arthur_leftover_is_four_l3459_345921

/-- The amount of money Arthur has leftover after selling his basketball cards and buying comic books -/
def arthursLeftover (cardValue : ℚ) (numCards : ℕ) (comicBookPrice : ℚ) : ℚ :=
  let totalCardValue := cardValue * numCards
  let numComicBooks := (totalCardValue / comicBookPrice).floor
  totalCardValue - numComicBooks * comicBookPrice

/-- Theorem stating that Arthur will have $4 leftover -/
theorem arthur_leftover_is_four :
  arthursLeftover (5/100) 2000 6 = 4 := by
  sorry

end arthur_leftover_is_four_l3459_345921


namespace march_largest_drop_l3459_345900

/-- Represents the months in the first half of 1994 -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June

/-- Returns the price change for a given month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January  => -1.00
  | Month.February => 0.50
  | Month.March    => -3.00
  | Month.April    => 2.00
  | Month.May      => -1.50
  | Month.June     => -0.75

/-- Determines if a given month has the largest price drop -/
def has_largest_drop (m : Month) : Prop :=
  ∀ (other : Month), price_change m ≤ price_change other

theorem march_largest_drop :
  has_largest_drop Month.March :=
sorry

end march_largest_drop_l3459_345900


namespace distance_to_y_axis_l3459_345936

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate. -/
theorem distance_to_y_axis (A : ℝ × ℝ) : 
  A.1 = -3 → A.2 = 4 → |A.1| = 3 := by sorry

end distance_to_y_axis_l3459_345936


namespace p_current_age_l3459_345999

theorem p_current_age (p q : ℕ) : 
  (p - 3) / (q - 3) = 4 / 3 →
  (p + 6) / (q + 6) = 7 / 6 →
  p = 15 := by
sorry

end p_current_age_l3459_345999


namespace distance_between_A_and_B_l3459_345935

-- Define the position of point A
def A : ℝ := 3

-- Define the possible positions of point B
def B : Set ℝ := {-9, 9}

-- Define the distance function
def distance (x y : ℝ) : ℝ := |x - y|

-- Theorem statement
theorem distance_between_A_and_B :
  ∀ b ∈ B, distance A b = 6 ∨ distance A b = 12 := by
  sorry

end distance_between_A_and_B_l3459_345935


namespace A_intersect_B_eq_open_interval_l3459_345986

-- Define set A
def A : Set ℝ := {x | (x + 2) / (x - 3) < 0}

-- Define set B
def B : Set ℝ := {x | x > 0}

-- Theorem statement
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 0 3 := by sorry

end A_intersect_B_eq_open_interval_l3459_345986


namespace difference_y_coordinates_l3459_345939

/-- Given a line with equation x = 2y + 5 and two points (m, n) and (m + 4, n + k) on this line,
    the value of k is 2. -/
theorem difference_y_coordinates (m n k : ℝ) : 
  (m = 2*n + 5) → (m + 4 = 2*(n + k) + 5) → k = 2 := by
  sorry

end difference_y_coordinates_l3459_345939


namespace not_in_fourth_quadrant_l3459_345978

/-- A linear function defined by y = 3x + 2 -/
def linear_function (x : ℝ) : ℝ := 3 * x + 2

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Theorem stating that the linear function y = 3x + 2 does not pass through the fourth quadrant -/
theorem not_in_fourth_quadrant :
  ∀ x : ℝ, ¬(fourth_quadrant x (linear_function x)) :=
by sorry

end not_in_fourth_quadrant_l3459_345978


namespace intersection_nonempty_implies_a_value_l3459_345901

theorem intersection_nonempty_implies_a_value (a : ℝ) : 
  let P : Set ℝ := {0, a}
  let Q : Set ℝ := {1, 2}
  (P ∩ Q).Nonempty → a = 1 ∨ a = 2 :=
by sorry

end intersection_nonempty_implies_a_value_l3459_345901


namespace simultaneous_equations_solution_l3459_345930

theorem simultaneous_equations_solution (m : ℝ) :
  ∃ (x y : ℝ), y = 3 * m * x + 4 ∧ y = (3 * m - 1) * x + 3 := by
  sorry

end simultaneous_equations_solution_l3459_345930


namespace worker_payment_schedule_l3459_345932

/-- Proves that the amount to return for each day not worked is $25 --/
theorem worker_payment_schedule (total_days : Nat) (days_not_worked : Nat) (payment_per_day : Nat) (total_earnings : Nat) :
  total_days = 30 →
  days_not_worked = 24 →
  payment_per_day = 100 →
  total_earnings = 0 →
  (total_days - days_not_worked) * payment_per_day = days_not_worked * 25 := by
  sorry

end worker_payment_schedule_l3459_345932


namespace julie_hourly_rate_l3459_345937

/-- Calculates the hourly rate given the following conditions:
  * Hours worked per day
  * Days worked per week
  * Monthly salary when missing one day of work
  * Number of weeks in a month
-/
def calculate_hourly_rate (hours_per_day : ℕ) (days_per_week : ℕ) 
  (monthly_salary_missing_day : ℕ) (weeks_per_month : ℕ) : ℚ :=
  let total_hours := hours_per_day * days_per_week * weeks_per_month - hours_per_day
  monthly_salary_missing_day / total_hours

theorem julie_hourly_rate :
  calculate_hourly_rate 8 6 920 4 = 5 := by
  sorry

end julie_hourly_rate_l3459_345937


namespace gcd_g_x_l3459_345991

def g (x : ℤ) : ℤ := (5*x+3)*(11*x+2)*(7*x+4)^2*(8*x+5)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 360 * k) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 120 := by
sorry

end gcd_g_x_l3459_345991


namespace alice_savings_l3459_345902

/-- Alice's savings problem -/
theorem alice_savings (total_days : ℕ) (total_dimes : ℕ) (dime_value : ℚ) (daily_savings : ℚ) : 
  total_days = 40 →
  total_dimes = 4 →
  dime_value = 1/10 →
  daily_savings = (total_dimes : ℚ) * dime_value / total_days →
  daily_savings = 1/100 := by
  sorry

#check alice_savings

end alice_savings_l3459_345902


namespace apple_basket_problem_l3459_345993

theorem apple_basket_problem (n : ℕ) (h1 : n > 1) : 
  (2 : ℝ) / n = (2 : ℝ) / 5 → n = 5 := by
  sorry

end apple_basket_problem_l3459_345993


namespace count_hexagons_l3459_345989

/-- The number of regular hexagons in a larger hexagon -/
def num_hexagons (n : ℕ+) : ℚ :=
  (n^2 + n : ℚ)^2 / 4

/-- Theorem: The number of regular hexagons with vertices among the vertices of equilateral triangles
    in a regular hexagon of side length n is (n² + n)² / 4 -/
theorem count_hexagons (n : ℕ+) :
  num_hexagons n = (n^2 + n : ℚ)^2 / 4 := by
  sorry

end count_hexagons_l3459_345989


namespace circle_equation_l3459_345929

/-- A circle with center on y = 3x and tangent to x-axis -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.2 = 3 * center.1
  tangent_to_x_axis : center.2 = radius

/-- The line 2x + y - 10 = 0 -/
def intercepting_line (x y : ℝ) : Prop := 2 * x + y - 10 = 0

/-- The chord intercepted by the line has length 4 -/
def chord_length (c : TangentCircle) : ℝ := 4

theorem circle_equation (c : TangentCircle) 
  (h : ∃ (x y : ℝ), intercepting_line x y ∧ 
       ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
       ((x - c.center.1)^2 + (y - c.center.2)^2 = (chord_length c / 2)^2)) :
  ((c.center.1 = 1 ∧ c.center.2 = 3 ∧ c.radius = 3) ∨
   (c.center.1 = -6 ∧ c.center.2 = -18 ∧ c.radius = 18)) :=
sorry

end circle_equation_l3459_345929


namespace tourist_tax_calculation_l3459_345912

theorem tourist_tax_calculation (tax_free_amount tax_rate total_tax : ℝ) 
  (h1 : tax_free_amount = 600)
  (h2 : tax_rate = 0.07)
  (h3 : total_tax = 78.4) : 
  ∃ (total_value : ℝ), 
    total_value > tax_free_amount ∧ 
    tax_rate * (total_value - tax_free_amount) = total_tax ∧ 
    total_value = 1720 := by
  sorry

end tourist_tax_calculation_l3459_345912


namespace sum_of_cubes_l3459_345976

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 11) (h2 : a * b = 21) :
  a^3 + b^3 = 638 := by sorry

end sum_of_cubes_l3459_345976


namespace product_of_numbers_l3459_345933

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 42) (h2 : |x - y| = 4) : x * y = 437 := by
  sorry

end product_of_numbers_l3459_345933


namespace triangle_perimeter_l3459_345973

/-- Given a triangle with sides a, b, and c, where a = 3, b = 5, and c is a root of x^2 - 5x + 4 = 0
    that satisfies the triangle inequality, prove that the perimeter is 12. -/
theorem triangle_perimeter (a b c : ℝ) : 
  a = 3 → b = 5 → c^2 - 5*c + 4 = 0 → 
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 12 := by sorry

end triangle_perimeter_l3459_345973


namespace system_of_inequalities_l3459_345998

theorem system_of_inequalities (x : ℝ) :
  (2 + x < 6 - 3 * x) ∧ (x ≤ (4 + x) / 2) → x < 1 := by
  sorry

end system_of_inequalities_l3459_345998


namespace water_speed_calculation_l3459_345954

/-- The speed of water in a river where a person who can swim at 12 km/h in still water
    takes 1 hour to swim 10 km against the current. -/
def water_speed : ℝ := 2

theorem water_speed_calculation (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : still_water_speed = 12)
  (h2 : distance = 10)
  (h3 : time = 1)
  (h4 : distance / time = still_water_speed - water_speed) : 
  water_speed = 2 := by
  sorry

#check water_speed_calculation

end water_speed_calculation_l3459_345954


namespace marble_173_is_gray_l3459_345988

/-- Represents the color of a marble -/
inductive MarbleColor
| Gray
| White
| Black

/-- Defines the pattern of marbles -/
def marblePattern : List MarbleColor :=
  List.replicate 6 MarbleColor.Gray ++
  List.replicate 3 MarbleColor.White ++
  List.replicate 5 MarbleColor.Black

/-- Determines the color of the nth marble in the sequence -/
def nthMarbleColor (n : Nat) : MarbleColor :=
  let patternLength := marblePattern.length
  let indexInPattern := (n - 1) % patternLength
  marblePattern[indexInPattern]'
    (by
      have h : indexInPattern < patternLength := Nat.mod_lt _ (Nat.zero_lt_of_ne_zero (by decide))
      exact h
    )

/-- Theorem: The 173rd marble is gray -/
theorem marble_173_is_gray : nthMarbleColor 173 = MarbleColor.Gray := by
  sorry

end marble_173_is_gray_l3459_345988


namespace solution_set_equality_l3459_345957

theorem solution_set_equality : {x : ℝ | x^2 - 2*x + 1 = 0} = {1} := by
  sorry

end solution_set_equality_l3459_345957


namespace vector_properties_l3459_345972

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (1, 2)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 4) ∧
  ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2 = 2) ∧
  ((a.1 + b.1) * c.1 + (a.2 + b.2) * c.2 = 0) :=
sorry

end vector_properties_l3459_345972


namespace triangle_inequality_l3459_345942

open Real

theorem triangle_inequality (A B C : ℝ) (R r : ℝ) :
  R > 0 ∧ r > 0 →
  (3 * Real.sqrt 3 * r^2) / (2 * R^2) ≤ Real.sin A * Real.sin B * Real.sin C ∧
  Real.sin A * Real.sin B * Real.sin C ≤ (3 * Real.sqrt 3 * r) / (4 * R) ∧
  (3 * Real.sqrt 3 * r) / (4 * R) ≤ 3 * Real.sqrt 3 / 8 :=
by sorry

end triangle_inequality_l3459_345942


namespace inequality_solution_parity_of_f_l3459_345914

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a/x

theorem inequality_solution :
  (∀ x, 0 < x ∧ x < 1 ↔ f x 2 - f (x-1) 2 > 2*x - 1) :=
sorry

theorem parity_of_f :
  (∀ x ≠ 0, f (-x) 0 = f x 0) ∧
  (∀ a ≠ 0, ∃ x ≠ 0, f (-x) a ≠ f x a ∧ f (-x) a ≠ -f x a) :=
sorry

end inequality_solution_parity_of_f_l3459_345914


namespace average_pencils_per_box_l3459_345968

theorem average_pencils_per_box : 
  let pencil_counts : List Nat := [12, 14, 14, 15, 15, 15, 16, 16, 17, 18]
  let total_boxes : Nat := pencil_counts.length
  let total_pencils : Nat := pencil_counts.sum
  (total_pencils : ℚ) / total_boxes = 15.2 := by
  sorry

end average_pencils_per_box_l3459_345968


namespace polynomial_expansion_l3459_345950

theorem polynomial_expansion (x : ℝ) : 
  (x^3 - 3*x + 3) * (x^2 + 3*x + 3) = x^5 + 3*x^4 - 6*x^2 + 9 := by
  sorry

end polynomial_expansion_l3459_345950


namespace distance_before_collision_value_l3459_345959

/-- Two boats moving towards each other -/
structure BoatSystem where
  boat1_speed : ℝ
  boat2_speed : ℝ
  initial_distance : ℝ

/-- Calculate the distance between boats one minute before collision -/
def distance_before_collision (bs : BoatSystem) : ℝ :=
  sorry

/-- Theorem stating the distance between boats one minute before collision -/
theorem distance_before_collision_value (bs : BoatSystem)
  (h1 : bs.boat1_speed = 5)
  (h2 : bs.boat2_speed = 25)
  (h3 : bs.initial_distance = 20) :
  distance_before_collision bs = 0.5 :=
sorry

end distance_before_collision_value_l3459_345959


namespace fruit_filling_probability_is_five_eighths_l3459_345908

/-- The number of fruit types available -/
def num_fruits : ℕ := 5

/-- The number of meat types available -/
def num_meats : ℕ := 4

/-- The number of ingredient types required for a filling -/
def ingredients_per_filling : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of making a mooncake with fruit filling -/
def fruit_filling_probability : ℚ :=
  choose num_fruits ingredients_per_filling /
  (choose num_fruits ingredients_per_filling + choose num_meats ingredients_per_filling)

theorem fruit_filling_probability_is_five_eighths :
  fruit_filling_probability = 5 / 8 := by
  sorry

end fruit_filling_probability_is_five_eighths_l3459_345908


namespace function_domain_range_implies_b_equals_two_l3459_345981

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the properties of the function
def has_domain_range (b : ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 1 b ↔ f x ∈ Set.Icc 1 b) ∧
  (∀ y ∈ Set.Icc 1 b, ∃ x ∈ Set.Icc 1 b, f x = y)

-- Theorem statement
theorem function_domain_range_implies_b_equals_two :
  ∃ b : ℝ, has_domain_range b → b = 2 := by sorry

end function_domain_range_implies_b_equals_two_l3459_345981


namespace point_on_double_angle_l3459_345946

/-- Given a point P(-1, 2) on the terminal side of angle α, 
    prove that the point (-3, -4) lies on the terminal side of angle 2α. -/
theorem point_on_double_angle (α : ℝ) :
  let P : ℝ × ℝ := (-1, 2)
  let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
  let cos_α : ℝ := P.1 / r
  let sin_α : ℝ := P.2 / r
  let cos_2α : ℝ := cos_α^2 - sin_α^2
  let sin_2α : ℝ := 2 * sin_α * cos_α
  let Q : ℝ × ℝ := (-3, -4)
  (∃ k : ℝ, k > 0 ∧ Q.1 = k * cos_2α ∧ Q.2 = k * sin_2α) :=
by
  sorry

end point_on_double_angle_l3459_345946


namespace constant_term_expansion_l3459_345947

/-- The constant term in the expansion of (x - 3/x^2)^6 -/
def constant_term : ℕ := 135

/-- The binomial coefficient function -/
def binomial_coeff (n k : ℕ) : ℕ := sorry

/-- Theorem: The constant term in the expansion of (x - 3/x^2)^6 is 135 -/
theorem constant_term_expansion :
  constant_term = binomial_coeff 6 2 * 3^2 :=
by sorry

end constant_term_expansion_l3459_345947


namespace theater_probability_ratio_l3459_345928

theorem theater_probability_ratio : 
  let n : ℕ := 4  -- number of sections and acts
  let p : ℝ := 1 / 4  -- probability of moving in a given act
  let q : ℝ := 1 - p  -- probability of not moving in a given act
  let prob_move_once : ℝ := n * p * q^(n-1)  -- probability of moving exactly once
  let prob_move_twice : ℝ := (n.choose 2) * p^2 * q^(n-2)  -- probability of moving exactly twice
  prob_move_twice / prob_move_once = 1 / 2 := by
sorry

end theater_probability_ratio_l3459_345928


namespace claire_earnings_l3459_345964

-- Define the given quantities
def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def small_red_roses : ℕ := 40
def medium_red_roses : ℕ := 60

-- Define the prices
def price_small : ℚ := 3/4
def price_medium : ℚ := 1
def price_large : ℚ := 5/4

-- Calculate the number of roses and red roses
def roses : ℕ := total_flowers - tulips
def red_roses : ℕ := roses - white_roses

-- Calculate the number of large red roses
def large_red_roses : ℕ := red_roses - small_red_roses - medium_red_roses

-- Define the function to calculate earnings
def earnings : ℚ :=
  (small_red_roses / 2 : ℚ) * price_small +
  (medium_red_roses / 2 : ℚ) * price_medium +
  (large_red_roses / 2 : ℚ) * price_large

-- Theorem statement
theorem claire_earnings : earnings = 215/2 := by sorry

end claire_earnings_l3459_345964


namespace camping_items_l3459_345979

theorem camping_items (total_items : ℕ) 
  (tent_stakes : ℕ) 
  (drink_mix : ℕ) 
  (water_bottles : ℕ) 
  (food_cans : ℕ) : 
  total_items = 32 → 
  drink_mix = 2 * tent_stakes → 
  water_bottles = tent_stakes + 2 → 
  food_cans * 2 = tent_stakes → 
  tent_stakes + drink_mix + water_bottles + food_cans = total_items → 
  tent_stakes = 6 := by
sorry

end camping_items_l3459_345979


namespace quadratic_vertex_value_and_range_l3459_345925

/-- The quadratic function y = ax^2 + 2ax + a -/
def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + a

/-- The x-coordinate of the vertex of the quadratic function -/
def vertex_x (a : ℝ) : ℝ := -1

theorem quadratic_vertex_value_and_range (a : ℝ) :
  f a (vertex_x a) = 0 ∧ f a (vertex_x a) ≤ 1 :=
sorry

end quadratic_vertex_value_and_range_l3459_345925


namespace cyclic_ratio_inequality_l3459_345956

theorem cyclic_ratio_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / b + b / c + c / d + d / a ≥ 4 := by sorry

end cyclic_ratio_inequality_l3459_345956


namespace quadratic_always_positive_l3459_345975

/-- If x² + 2x + m > 0 for all real x, then m > 1 -/
theorem quadratic_always_positive (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 1 := by
  sorry

end quadratic_always_positive_l3459_345975


namespace impossible_corner_cut_l3459_345970

theorem impossible_corner_cut (a b c : ℝ) : 
  a^2 + b^2 = 25 ∧ b^2 + c^2 = 36 ∧ c^2 + a^2 = 64 → False :=
by
  sorry

#check impossible_corner_cut

end impossible_corner_cut_l3459_345970


namespace smallest_prime_sum_l3459_345990

def digit_set : Set Nat := {1, 2, 3, 5}

def is_valid_prime (p : Nat) (used_digits : Set Nat) : Prop :=
  Nat.Prime p ∧ 
  (p % 10) ∈ digit_set ∧
  (p % 10) ∉ used_digits ∧
  (∀ d ∈ digit_set, d ≠ p % 10 → ¬ (∃ k, p / 10^k % 10 = d))

def valid_prime_triple (p q r : Nat) : Prop :=
  is_valid_prime p ∅ ∧
  is_valid_prime q {p % 10} ∧
  is_valid_prime r {p % 10, q % 10}

theorem smallest_prime_sum :
  ∀ p q r, valid_prime_triple p q r → p + q + r ≥ 71 :=
sorry

end smallest_prime_sum_l3459_345990


namespace integral_comparison_l3459_345943

theorem integral_comparison : ∫ x in (0:ℝ)..1, x > ∫ x in (0:ℝ)..1, x^3 := by
  sorry

end integral_comparison_l3459_345943


namespace determinant_of_cubic_roots_l3459_345997

theorem determinant_of_cubic_roots (p q r : ℝ) (a b c : ℝ) : 
  (∀ x : ℝ, x^3 + 3*p*x^2 + q*x + r = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  let matrix := !![a, b, c; b, c, a; c, a, b]
  Matrix.det matrix = 3*p*q := by
sorry

end determinant_of_cubic_roots_l3459_345997


namespace tangent_line_and_root_range_l3459_345963

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x

theorem tangent_line_and_root_range :
  -- Part 1: Tangent line equation
  (∀ x y : ℝ, y = f x → (x = 2 → 12 * x - y - 17 = 0)) ∧
  -- Part 2: Range of m for three distinct real roots
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔ -3 < m ∧ m < -2) :=
by sorry

end tangent_line_and_root_range_l3459_345963


namespace fraction_sum_reciprocal_l3459_345904

theorem fraction_sum_reciprocal (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  1 / x + 1 / y = 1 / z → z = (x * y) / (y + x) := by
  sorry

end fraction_sum_reciprocal_l3459_345904


namespace circle_area_triple_radius_l3459_345907

theorem circle_area_triple_radius (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := π * (3*r)^2
  A' = 9 * A := by sorry

end circle_area_triple_radius_l3459_345907


namespace complex_equation_solution_l3459_345962

theorem complex_equation_solution (x₁ x₂ A : ℂ) (h_distinct : x₁ ≠ x₂)
  (h_eq1 : x₁ * (x₁ + 1) = A)
  (h_eq2 : x₂ * (x₂ + 1) = A)
  (h_eq3 : x₁^4 + 3*x₁^3 + 5*x₁ = x₂^4 + 3*x₂^3 + 5*x₂) :
  A = -7 := by sorry

end complex_equation_solution_l3459_345962


namespace emma_bank_account_l3459_345918

theorem emma_bank_account (initial_amount : ℝ) : 
  let withdrawal := 60
  let deposit := 2 * withdrawal
  let final_balance := 290
  (initial_amount - withdrawal + deposit = final_balance) → initial_amount = 230 := by
sorry

end emma_bank_account_l3459_345918


namespace license_plate_count_l3459_345958

/-- Number of digits in the license plate -/
def num_digits : ℕ := 5

/-- Number of letters in the license plate -/
def num_letters : ℕ := 3

/-- Number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- Number of possible letters (A-Z) -/
def letter_choices : ℕ := 26

/-- Number of positions where the consecutive letters can be placed -/
def letter_positions : ℕ := num_digits + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := letter_positions * digit_choices^num_digits * letter_choices^num_letters

theorem license_plate_count : total_license_plates = 105456000 := by
  sorry

end license_plate_count_l3459_345958


namespace no_intersection_l3459_345952

/-- Represents a 2D point or vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Represents a parametric line in 2D -/
structure ParamLine where
  origin : Vec2D
  direction : Vec2D

/-- The first line -/
def line1 : ParamLine :=
  { origin := { x := 1, y := 4 }
    direction := { x := -2, y := 6 } }

/-- The second line -/
def line2 : ParamLine :=
  { origin := { x := 3, y := 10 }
    direction := { x := -1, y := 3 } }

/-- Checks if two parametric lines intersect -/
def linesIntersect (l1 l2 : ParamLine) : Prop :=
  ∃ (s t : ℝ), l1.origin.x + s * l1.direction.x = l2.origin.x + t * l2.direction.x ∧
                l1.origin.y + s * l1.direction.y = l2.origin.y + t * l2.direction.y

theorem no_intersection : ¬ linesIntersect line1 line2 := by
  sorry

end no_intersection_l3459_345952


namespace problem_l3459_345996

/-- Given m > 0, prove the following statements -/
theorem problem (m : ℝ) (hm : m > 0) :
  /- If (x+2)(x-6) ≤ 0 implies 2-m ≤ x ≤ 2+m for all x, then m ≥ 4 -/
  ((∀ x, (x + 2) * (x - 6) ≤ 0 → 2 - m ≤ x ∧ x ≤ 2 + m) → m ≥ 4) ∧
  /- If m = 5, and for all x, ((x+2)(x-6) ≤ 0) ∨ (-3 ≤ x ≤ 7) is true, 
     and ((x+2)(x-6) ≤ 0) ∧ (-3 ≤ x ≤ 7) is false, 
     then x ∈ [-3,-2) ∪ (6,7] -/
  (m = 5 → 
    (∀ x, ((x + 2) * (x - 6) ≤ 0 ∨ (-3 ≤ x ∧ x ≤ 7)) ∧
           ¬((x + 2) * (x - 6) ≤ 0 ∧ -3 ≤ x ∧ x ≤ 7)) →
    (∀ x, x ∈ Set.Ioo (-3) (-2) ∪ Set.Ioc 6 7)) :=
by sorry

end problem_l3459_345996


namespace not_all_cells_tetraploid_l3459_345949

/-- Represents a watermelon plant --/
structure WatermelonPlant where
  /-- The number of chromosome sets in somatic cells --/
  somaticChromosomeSets : ℕ
  /-- The number of chromosome sets in root cells --/
  rootChromosomeSets : ℕ

/-- Represents the process of culturing and treating watermelon plants --/
def cultureAndTreat (original : WatermelonPlant) : WatermelonPlant :=
  { somaticChromosomeSets := 2 * original.somaticChromosomeSets,
    rootChromosomeSets := original.rootChromosomeSets }

/-- Theorem: Not all cells in a watermelon plant obtained from treating diploid seedlings
    with colchicine contain four sets of chromosomes --/
theorem not_all_cells_tetraploid (original : WatermelonPlant)
    (h_diploid : original.somaticChromosomeSets = 2)
    (h_root_untreated : (cultureAndTreat original).rootChromosomeSets = original.rootChromosomeSets) :
    ∃ (cell_type : WatermelonPlant → ℕ),
      cell_type (cultureAndTreat original) ≠ 4 :=
  sorry


end not_all_cells_tetraploid_l3459_345949


namespace congruence_solution_l3459_345965

theorem congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 14567 [MOD 16] ∧ n = 7 := by
  sorry

end congruence_solution_l3459_345965


namespace natural_number_pair_product_sum_gcd_lcm_l3459_345910

theorem natural_number_pair_product_sum_gcd_lcm : 
  ∀ a b : ℕ, 
    a > 0 ∧ b > 0 → 
    (a * b - (a + b) = Nat.gcd a b + Nat.lcm a b) ↔ 
    ((a = 6 ∧ b = 3) ∨ (a = 6 ∧ b = 4) ∨ (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 6)) :=
by sorry

end natural_number_pair_product_sum_gcd_lcm_l3459_345910


namespace rectangle_diagonal_estimate_l3459_345917

theorem rectangle_diagonal_estimate (length width diagonal : ℝ) : 
  length = 3 → width = 2 → diagonal^2 = length^2 + width^2 → 
  3.6 < diagonal ∧ diagonal < 3.7 := by
  sorry

end rectangle_diagonal_estimate_l3459_345917


namespace parabola_directrix_intersection_l3459_345903

/-- The parabola equation: x^2 = 4y -/
def parabola_equation (x y : ℝ) : Prop := x^2 = 4*y

/-- The directrix equation for a parabola with equation x^2 = 4ay -/
def directrix_equation (a y : ℝ) : Prop := y = -a

/-- The y-axis equation -/
def y_axis (x : ℝ) : Prop := x = 0

theorem parabola_directrix_intersection :
  ∃ (a : ℝ), a = 1 ∧
  (∀ x y : ℝ, parabola_equation x y ↔ x^2 = 4*a*y) ∧
  (∃ y : ℝ, directrix_equation a y ∧ y_axis 0 ∧ y = -1) :=
sorry

end parabola_directrix_intersection_l3459_345903


namespace popcorn_probability_l3459_345977

theorem popcorn_probability (white_ratio : ℚ) (yellow_ratio : ℚ) 
  (white_pop_prob : ℚ) (yellow_pop_prob : ℚ) :
  white_ratio = 3/4 →
  yellow_ratio = 1/4 →
  white_pop_prob = 1/3 →
  yellow_pop_prob = 3/4 →
  let white_and_pop := white_ratio * white_pop_prob
  let yellow_and_pop := yellow_ratio * yellow_pop_prob
  let total_pop := white_and_pop + yellow_and_pop
  (white_and_pop / total_pop) = 4/7 := by
  sorry

end popcorn_probability_l3459_345977


namespace number_calculation_l3459_345982

theorem number_calculation (n : ℝ) : 
  (0.20 * 0.45 * 0.60 * 0.75 * n = 283.5) → n = 7000 := by
  sorry

end number_calculation_l3459_345982


namespace greatest_multiple_of_5_and_6_less_than_1000_l3459_345919

theorem greatest_multiple_of_5_and_6_less_than_1000 : 
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 6 = 0 → n ≤ 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l3459_345919


namespace quadratic_real_roots_k_range_l3459_345984

theorem quadratic_real_roots_k_range (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ (k ≥ -9/4 ∧ k ≠ 0) :=
sorry

end quadratic_real_roots_k_range_l3459_345984


namespace slope_of_line_l_l3459_345920

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

-- Define the center of the circle
def center : ℝ × ℝ := (3, 5)

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 5 = k * (x - 3)

-- Define points A, B, and P
variables (A B P : ℝ × ℝ)

-- State that A and B are on the circle C
axiom A_on_circle : circle_C A.1 A.2
axiom B_on_circle : circle_C B.1 B.2

-- State that A, B, and P are on line l
axiom A_on_line : ∃ k, line_l k A.1 A.2
axiom B_on_line : ∃ k, line_l k B.1 B.2
axiom P_on_line : ∃ k, line_l k P.1 P.2

-- State that P is on the y-axis
axiom P_on_y_axis : P.1 = 0

-- State the vector relationship
axiom vector_relation : 2 * (A.1 - P.1, A.2 - P.2) = (B.1 - P.1, B.2 - P.2)

-- Theorem to prove
theorem slope_of_line_l : ∃ k, (k = 2 ∨ k = -2) ∧ line_l k A.1 A.2 ∧ line_l k B.1 B.2 ∧ line_l k P.1 P.2 :=
sorry

end slope_of_line_l_l3459_345920


namespace all_statements_false_l3459_345927

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Define the theorem
theorem all_statements_false :
  ¬(∀ (m n : Line) (α : Plane), 
    parallel_line_plane m α → parallel_line_plane n α → parallel_lines m n) ∧
  ¬(∀ (m n : Line) (α : Plane), 
    perpendicular_line_plane m α → perpendicular_lines m n → parallel_line_plane n α) ∧
  ¬(∀ (m n : Line) (α β : Plane), 
    perpendicular_line_plane m α → perpendicular_line_plane n β → 
    perpendicular_lines m n → perpendicular_planes α β) ∧
  ¬(∀ (m : Line) (α β : Plane), 
    line_in_plane m β → parallel_planes α β → parallel_line_plane m α) :=
by sorry

end all_statements_false_l3459_345927


namespace zaras_estimate_bound_l3459_345983

theorem zaras_estimate_bound (x y ε : ℝ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : x - y < ε) 
  (h4 : ε > 0) : 
  (x + 2*ε) - (y - ε) < 2*ε := by
sorry

end zaras_estimate_bound_l3459_345983


namespace sphere_volume_from_surface_area_l3459_345906

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
  (4 * Real.pi * r ^ 2 = 16 * Real.pi) →
  (4 / 3 * Real.pi * r ^ 3 = 32 / 3 * Real.pi) := by
  sorry

end sphere_volume_from_surface_area_l3459_345906


namespace probabilities_correct_l3459_345944

/-- Represents the color of a ball -/
inductive Color
  | Black
  | White

/-- Represents a bag containing balls -/
structure Bag where
  black : ℕ
  white : ℕ

/-- Calculate the probability of drawing a ball of a specific color from a bag -/
def prob_color (b : Bag) (c : Color) : ℚ :=
  match c with
  | Color.Black => b.black / (b.black + b.white)
  | Color.White => b.white / (b.black + b.white)

/-- The contents of bag A -/
def bag_A : Bag := ⟨2, 2⟩

/-- The contents of bag B -/
def bag_B : Bag := ⟨2, 1⟩

theorem probabilities_correct :
  (prob_color bag_A Color.Black * prob_color bag_B Color.Black = 1/3) ∧
  (prob_color bag_A Color.White * prob_color bag_B Color.White = 1/6) ∧
  (prob_color bag_A Color.White * prob_color bag_B Color.White +
   prob_color bag_A Color.White * prob_color bag_B Color.Black +
   prob_color bag_A Color.Black * prob_color bag_B Color.White = 2/3) :=
by sorry

end probabilities_correct_l3459_345944


namespace problem_solution_l3459_345922

theorem problem_solution (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h1 : a^b = b^a) (h2 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end problem_solution_l3459_345922


namespace chess_pieces_arrangement_l3459_345934

theorem chess_pieces_arrangement (total : ℕ) 
  (h1 : ∃ inner : ℕ, total = inner + 60)
  (h2 : ∃ outer : ℕ, 60 = outer + 32) : 
  total = 80 := by
sorry

end chess_pieces_arrangement_l3459_345934


namespace matrix_vector_computation_l3459_345923

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (v w u : Fin 2 → ℝ)

theorem matrix_vector_computation
  (hv : M.mulVec v = ![2, 6])
  (hw : M.mulVec w = ![3, -5])
  (hu : M.mulVec u = ![-1, 4]) :
  M.mulVec (2 • v - w + 4 • u) = ![-3, 33] := by sorry

end matrix_vector_computation_l3459_345923


namespace binomial_20_choose_6_l3459_345915

theorem binomial_20_choose_6 : Nat.choose 20 6 = 38760 := by sorry

end binomial_20_choose_6_l3459_345915


namespace pentagon_triangle_intersections_pentagon_quadrilateral_intersections_l3459_345966

/-- A polygon in a plane -/
class Polygon :=
  (sides : ℕ)

/-- A pentagon is a polygon with 5 sides -/
def Pentagon : Polygon :=
  { sides := 5 }

/-- A triangle is a polygon with 3 sides -/
def Triangle : Polygon :=
  { sides := 3 }

/-- A quadrilateral is a polygon with 4 sides -/
def Quadrilateral : Polygon :=
  { sides := 4 }

/-- The maximum number of intersection points between the sides of two polygons -/
def maxIntersections (P Q : Polygon) : ℕ := sorry

/-- Theorem: Maximum intersections between a pentagon and a triangle -/
theorem pentagon_triangle_intersections :
  maxIntersections Pentagon Triangle = 10 := by sorry

/-- Theorem: Maximum intersections between a pentagon and a quadrilateral -/
theorem pentagon_quadrilateral_intersections :
  maxIntersections Pentagon Quadrilateral = 16 := by sorry

end pentagon_triangle_intersections_pentagon_quadrilateral_intersections_l3459_345966


namespace great_pyramid_tallest_duration_l3459_345926

/-- Represents the dimensions and historical facts about the Great Pyramid of Giza -/
structure GreatPyramid where
  height : ℕ
  width : ℕ
  year_built : Int
  year_surpassed : Int
  height_above_500 : height = 500 + 20
  width_relation : width = height + 234
  sum_height_width : height + width = 1274
  built_BC : year_built < 0
  surpassed_AD : year_surpassed > 0

/-- Theorem stating the duration for which the Great Pyramid was the tallest structure -/
theorem great_pyramid_tallest_duration (p : GreatPyramid) : 
  p.year_surpassed - p.year_built = 3871 :=
sorry

end great_pyramid_tallest_duration_l3459_345926


namespace profit_achieved_l3459_345995

/-- The number of pencils purchased -/
def num_purchased : ℕ := 1800

/-- The cost of each pencil when purchased -/
def cost_per_pencil : ℚ := 15 / 100

/-- The selling price of each pencil -/
def selling_price : ℚ := 30 / 100

/-- The desired profit -/
def desired_profit : ℚ := 150

/-- The number of pencils that must be sold to make the desired profit -/
def num_sold : ℕ := 1400

theorem profit_achieved : 
  (num_sold : ℚ) * selling_price - (num_purchased : ℚ) * cost_per_pencil = desired_profit := by
  sorry

end profit_achieved_l3459_345995


namespace soap_box_height_l3459_345916

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: Given the dimensions of a carton and soap boxes, and the maximum number of soap boxes
    that can fit in the carton, prove that the height of a soap box is 1 inch. -/
theorem soap_box_height
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (max_boxes : ℕ)
  (h_carton_length : carton.length = 30)
  (h_carton_width : carton.width = 42)
  (h_carton_height : carton.height = 60)
  (h_soap_length : soap.length = 7)
  (h_soap_width : soap.width = 6)
  (h_max_boxes : max_boxes = 360)
  : soap.height = 1 :=
by sorry

end soap_box_height_l3459_345916


namespace largest_equal_cost_number_l3459_345931

/-- Sum of digits in decimal representation -/
def sumOfDecimalDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDecimalDigits (n / 10)

/-- Sum of digits in binary representation -/
def sumOfBinaryDigits (n : Nat) : Nat :=
  if n = 0 then 0 else (n % 2) + sumOfBinaryDigits (n / 2)

/-- Cost calculation for Option 1 -/
def option1Cost (n : Nat) : Nat :=
  2 * sumOfDecimalDigits n

/-- Cost calculation for Option 2 -/
def option2Cost (n : Nat) : Nat :=
  sumOfBinaryDigits n

theorem largest_equal_cost_number :
  ∀ n : Nat, n < 2000 → n > 1023 →
    option1Cost n ≠ option2Cost n ∧
    option1Cost 1023 = option2Cost 1023 :=
by sorry

end largest_equal_cost_number_l3459_345931


namespace circle_line_intersection_and_min_chord_l3459_345911

/-- Circle C: x^2 + y^2 - 4x - 2y - 20 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

/-- Line l: mx - y - m + 3 = 0 (m ∈ ℝ) -/
def line_l (m x y : ℝ) : Prop := m*x - y - m + 3 = 0

theorem circle_line_intersection_and_min_chord :
  (∀ m : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_l m x y) ∧
  (∃ min_length : ℝ, min_length = 4 * Real.sqrt 5 ∧
    ∀ m : ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ,
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l m x₁ y₁ ∧ line_l m x₂ y₂ →
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ min_length) ∧
  (∃ x y : ℝ, circle_C x y ∧ x - 2*y + 5 = 0 ∧
    ∀ x' y' : ℝ, circle_C x' y' ∧ x' - 2*y' + 5 = 0 →
      Real.sqrt ((x - x')^2 + (y - y')^2) = 4 * Real.sqrt 5) :=
by sorry

end circle_line_intersection_and_min_chord_l3459_345911


namespace fraction_calculation_l3459_345955

theorem fraction_calculation : (2 / 8 : ℚ) + (4 / 16 : ℚ) * (3 / 9 : ℚ) = 1 / 3 := by
  sorry

end fraction_calculation_l3459_345955


namespace inverse_g_84_l3459_345913

theorem inverse_g_84 (g : ℝ → ℝ) (h : ∀ x, g x = 3 * x^3 + 3) :
  g 3 = 84 ∧ (∀ y, g y = 84 → y = 3) :=
sorry

end inverse_g_84_l3459_345913


namespace expression_simplification_l3459_345980

theorem expression_simplification (x y : ℝ) :
  (x^3 - 9*x*y^2) / (9*y^2 + x^2) * ((x + 3*y) / (x^2 - 3*x*y) + (x - 3*y) / (x^2 + 3*x*y)) = x - 3*y :=
by sorry

end expression_simplification_l3459_345980


namespace ball_probability_l3459_345974

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 20)
  (h_green : green = 30)
  (h_yellow : yellow = 10)
  (h_red : red = 37)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 0.6 := by
  sorry

end ball_probability_l3459_345974


namespace equation_solution_l3459_345945

theorem equation_solution :
  ∀ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) → x = -48 / 23 := by
  sorry

end equation_solution_l3459_345945
