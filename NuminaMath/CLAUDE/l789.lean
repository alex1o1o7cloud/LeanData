import Mathlib

namespace total_squares_is_83_l789_78965

/-- Represents the count of squares of a specific size in the figure -/
structure SquareCount where
  size : Nat
  count : Nat

/-- Represents the figure composed of squares and isosceles right triangles -/
structure Figure where
  squareCounts : List SquareCount

/-- Calculates the total number of squares in the figure -/
def totalSquares (f : Figure) : Nat :=
  f.squareCounts.foldl (fun acc sc => acc + sc.count) 0

/-- The specific figure described in the problem -/
def problemFigure : Figure :=
  { squareCounts := [
      { size := 1, count := 40 },
      { size := 2, count := 25 },
      { size := 3, count := 12 },
      { size := 4, count := 5 },
      { size := 5, count := 1 }
    ] }

theorem total_squares_is_83 : totalSquares problemFigure = 83 := by
  sorry

end total_squares_is_83_l789_78965


namespace sum_second_largest_second_smallest_l789_78924

/-- A function that generates all valid three-digit numbers using digits 0 to 9 (each digit used only once) -/
def generateNumbers : Finset Nat := sorry

/-- The second smallest number in the set of generated numbers -/
def secondSmallest : Nat := sorry

/-- The second largest number in the set of generated numbers -/
def secondLargest : Nat := sorry

/-- Theorem stating that the sum of the second largest and second smallest numbers is 1089 -/
theorem sum_second_largest_second_smallest :
  secondLargest + secondSmallest = 1089 := by sorry

end sum_second_largest_second_smallest_l789_78924


namespace gcd_16_12_l789_78917

def operation_process : List (Nat × Nat) := [(16, 12), (4, 12), (4, 8), (4, 4)]

theorem gcd_16_12 : Nat.gcd 16 12 = 4 := by
  sorry

end gcd_16_12_l789_78917


namespace monday_pages_to_reach_average_l789_78961

def target_average : ℕ := 50
def days_in_week : ℕ := 7
def known_pages : List ℕ := [43, 28, 0, 70, 56, 88]

theorem monday_pages_to_reach_average :
  ∃ (monday_pages : ℕ),
    (monday_pages + known_pages.sum) / days_in_week = target_average ∧
    monday_pages = 65 := by
  sorry

end monday_pages_to_reach_average_l789_78961


namespace inscribed_circle_radius_345_triangle_l789_78980

/-- A triangle with side lengths 3, 4, and 5 has an inscribed circle with radius 1. -/
theorem inscribed_circle_radius_345_triangle :
  ∀ (a b c r : ℝ),
  a = 3 ∧ b = 4 ∧ c = 5 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 1 := by
sorry

end inscribed_circle_radius_345_triangle_l789_78980


namespace existence_of_n_l789_78912

theorem existence_of_n : ∃ n : ℕ, n > 0 ∧ (1.001 : ℝ)^n > 10 ∧ (0.999 : ℝ)^n < 0.1 := by
  sorry

end existence_of_n_l789_78912


namespace cable_lengths_theorem_l789_78934

/-- Given two pieces of cable with specific mass and length relationships,
    prove that their lengths are either (5, 8) meters or (19.5, 22.5) meters. -/
theorem cable_lengths_theorem (mass1 mass2 : ℝ) (length_diff mass_per_meter_diff : ℝ) :
  mass1 = 65 →
  mass2 = 120 →
  length_diff = 3 →
  mass_per_meter_diff = 2 →
  ∃ (l1 l2 : ℝ),
    ((l1 = 5 ∧ l2 = 8) ∨ (l1 = 19.5 ∧ l2 = 22.5)) ∧
    (mass1 / l1 + mass_per_meter_diff) * (l1 + length_diff) = mass2 :=
by sorry

end cable_lengths_theorem_l789_78934


namespace union_equality_implies_a_values_l789_78979

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Theorem statement
theorem union_equality_implies_a_values (a : ℝ) : 
  A a ∪ B = B → a = 0 ∨ a = 1 ∨ a = 1/2 := by
  sorry

end union_equality_implies_a_values_l789_78979


namespace larger_integer_problem_l789_78916

theorem larger_integer_problem (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) (h4 : y - x = 8) (h5 : x * y = 272) : y = 17 := by
  sorry

end larger_integer_problem_l789_78916


namespace sandra_leftover_money_l789_78968

def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def candy_cost : ℚ := 1/2
def jelly_bean_cost : ℚ := 1/5
def candy_count : ℕ := 14
def jelly_bean_count : ℕ := 20

theorem sandra_leftover_money :
  (sandra_savings + mother_gift + father_gift : ℚ) - 
  (candy_count * candy_cost + jelly_bean_count * jelly_bean_cost) = 11 :=
by sorry

end sandra_leftover_money_l789_78968


namespace sqrt_27_div_sqrt_3_eq_3_l789_78926

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end sqrt_27_div_sqrt_3_eq_3_l789_78926


namespace antonia_pills_left_l789_78927

/-- Calculates the number of pills left after taking supplements for two weeks -/
def pills_left (bottles_120 : Nat) (bottles_30 : Nat) (supplements : Nat) (weeks : Nat) : Nat :=
  let total_pills := bottles_120 * 120 + bottles_30 * 30
  let days := weeks * 7
  let pills_used := days * supplements
  total_pills - pills_used

/-- Theorem stating that given the specific conditions, the number of pills left is 350 -/
theorem antonia_pills_left :
  pills_left 3 2 5 2 = 350 := by
  sorry

end antonia_pills_left_l789_78927


namespace cube_root_neg_eight_plus_sqrt_nine_equals_one_l789_78942

theorem cube_root_neg_eight_plus_sqrt_nine_equals_one :
  ((-8 : ℝ) ^ (1/3 : ℝ)) + (9 : ℝ).sqrt = 1 := by
  sorry

end cube_root_neg_eight_plus_sqrt_nine_equals_one_l789_78942


namespace mary_cut_ten_roses_l789_78928

/-- The number of roses Mary cut from her flower garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Proof that Mary cut 10 roses from her flower garden -/
theorem mary_cut_ten_roses (initial_roses final_roses : ℕ) 
  (h1 : initial_roses = 6) 
  (h2 : final_roses = 16) : 
  roses_cut initial_roses final_roses = 10 := by
  sorry

#check mary_cut_ten_roses

end mary_cut_ten_roses_l789_78928


namespace radical_simplification_l789_78933

theorem radical_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (98 * x) = 210 * x * Real.sqrt (2 * x) := by
  sorry

end radical_simplification_l789_78933


namespace waiter_initial_customers_l789_78994

def initial_customers (customers_left : ℕ) (remaining_tables : ℕ) (people_per_table : ℕ) : ℕ :=
  remaining_tables * people_per_table + customers_left

theorem waiter_initial_customers : 
  initial_customers 12 4 8 = 44 := by
  sorry

end waiter_initial_customers_l789_78994


namespace quadratic_equation_proof_l789_78900

theorem quadratic_equation_proof (a : ℝ) :
  (a^2 - 4*a + 5 ≠ 0) ∧
  (∀ x : ℝ, (2^2 - 4*2 + 5)*x^2 + 2*2*x + 4 = 0 ↔ x = -2) :=
by sorry

end quadratic_equation_proof_l789_78900


namespace f_and_g_odd_and_increasing_l789_78903

-- Define the functions
def f (x : ℝ) : ℝ := 6 * x
def g (x : ℝ) : ℝ := x * |x|

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Theorem statement
theorem f_and_g_odd_and_increasing :
  (is_odd f ∧ is_increasing f) ∧ (is_odd g ∧ is_increasing g) := by sorry

end f_and_g_odd_and_increasing_l789_78903


namespace intersection_implies_a_range_l789_78947

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def N (a : ℝ) : Set ℝ := {x : ℝ | 1 - 3*a < x ∧ x ≤ 2*a}

theorem intersection_implies_a_range (a : ℝ) : M ∩ N a = M → a ∈ Set.Ici 1 := by
  sorry

end intersection_implies_a_range_l789_78947


namespace savings_calculation_l789_78911

theorem savings_calculation (initial_savings : ℝ) : 
  let february_spend := 0.20 * initial_savings
  let march_spend := 0.40 * initial_savings
  let april_spend := 1500
  let remaining := 2900
  february_spend + march_spend + april_spend + remaining = initial_savings →
  initial_savings = 11000 := by
sorry

end savings_calculation_l789_78911


namespace recruit_count_l789_78971

theorem recruit_count (peter nikolai denis total : ℕ) : 
  peter = 50 →
  nikolai = 100 →
  denis = 170 →
  (total - peter - 1 = 4 * (total - denis - 1) ∨
   total - nikolai - 1 = 4 * (total - denis - 1) ∨
   total - peter - 1 = 4 * (total - nikolai - 1)) →
  total = 213 :=
by sorry

end recruit_count_l789_78971


namespace correct_arrangements_l789_78953

def num_seats : ℕ := 5
def num_teachers : ℕ := 4

/-- The number of arrangements where Teacher A is to the left of Teacher B -/
def arrangements_a_left_of_b : ℕ := 60

theorem correct_arrangements :
  arrangements_a_left_of_b = (num_seats.factorial / (num_seats - num_teachers).factorial) / 2 :=
sorry

end correct_arrangements_l789_78953


namespace parallelepiped_volume_l789_78956

theorem parallelepiped_volume (base_area : ℝ) (angle : ℝ) (lateral_area1 : ℝ) (lateral_area2 : ℝ) :
  base_area = 4 →
  angle = 30 * π / 180 →
  lateral_area1 = 6 →
  lateral_area2 = 12 →
  ∃ (a b c : ℝ),
    a * b * Real.sin angle = base_area ∧
    a * c = lateral_area1 ∧
    b * c = lateral_area2 ∧
    a * b * c = 12 := by
  sorry

#check parallelepiped_volume

end parallelepiped_volume_l789_78956


namespace remainder_seven_n_l789_78920

theorem remainder_seven_n (n : ℤ) (h : n ≡ 3 [ZMOD 4]) : 7*n ≡ 1 [ZMOD 4] := by
  sorry

end remainder_seven_n_l789_78920


namespace quadratic_no_real_roots_l789_78952

/-- Given a, b, c form a geometric sequence, the quadratic function f(x) = ax^2 + bx + c has no real roots -/
theorem quadratic_no_real_roots (a b c : ℝ) (h_geo : b^2 = a*c) (h_pos : a*c > 0) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
by sorry

end quadratic_no_real_roots_l789_78952


namespace inequality_proof_l789_78993

theorem inequality_proof (a b c A α : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hα : α > 0)
  (hsum : a + b + c = A) (hA : A ≤ 1) :
  (1/a - a)^α + (1/b - b)^α + (1/c - c)^α ≥ 3 * (3/A - A/3)^α := by
  sorry

end inequality_proof_l789_78993


namespace pseudoprime_construction_infinite_pseudoprimes_l789_78981

/-- A number n is a pseudoprime to base a if it's composite and a^(n-1) ≡ 1 (mod n) -/
def IsPseudoprime (n : ℕ) (a : ℕ) : Prop :=
  ¬ Nat.Prime n ∧ a^(n-1) % n = 1

/-- Given a pseudoprime m, 2^m - 1 is also a pseudoprime -/
theorem pseudoprime_construction (m : ℕ) (a : ℕ) (h : IsPseudoprime m a) :
  ∃ b : ℕ, IsPseudoprime (2^m - 1) b :=
sorry

/-- There are infinitely many pseudoprimes -/
theorem infinite_pseudoprimes : ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ∃ a : ℕ, IsPseudoprime m a :=
sorry

end pseudoprime_construction_infinite_pseudoprimes_l789_78981


namespace custom_mult_theorem_l789_78909

/-- Custom multiplication operation for integers -/
def customMult (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that if y10 = 90 under the custom multiplication, then y = 11 -/
theorem custom_mult_theorem (y : ℤ) (h : customMult y 10 = 90) : y = 11 := by
  sorry

end custom_mult_theorem_l789_78909


namespace point_c_coordinates_l789_78975

/-- Given points A, B, and C on a line, where C divides AB in the ratio 2:1,
    prove that C has the specified coordinates. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) : 
  A = (-3, -2) →
  B = (5, 10) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) →  -- C is on line segment AB
  dist A C = 2 * dist C B →                             -- AC = 2CB
  C = (11/3, 8) := by
  sorry


end point_c_coordinates_l789_78975


namespace zeoland_speeding_fine_l789_78905

/-- The speeding fine structure in Zeoland -/
structure SpeedingFine where
  totalFine : ℕ      -- Total fine amount
  speedLimit : ℕ     -- Posted speed limit
  actualSpeed : ℕ    -- Actual speed of the driver
  finePerMph : ℕ     -- Fine per mile per hour over the limit

/-- Theorem: Given Jed's speeding fine details, prove the fine per mph over the limit -/
theorem zeoland_speeding_fine (fine : SpeedingFine) 
  (h1 : fine.totalFine = 256)
  (h2 : fine.speedLimit = 50)
  (h3 : fine.actualSpeed = 66) :
  fine.finePerMph = 16 := by
  sorry


end zeoland_speeding_fine_l789_78905


namespace group_size_l789_78998

theorem group_size (N : ℝ) 
  (h1 : N / 5 = N * (1 / 5))  -- 1/5 of the group plays at least one instrument
  (h2 : N * (1 / 5) - 128 = N * 0.04)  -- Probability of playing exactly one instrument is 0.04
  : N = 800 := by
  sorry

end group_size_l789_78998


namespace min_value_sum_squares_l789_78949

theorem min_value_sum_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 11) :
  x^2 + y^2 + z^2 ≥ 121/29 := by
sorry

end min_value_sum_squares_l789_78949


namespace minutes_before_noon_l789_78974

theorem minutes_before_noon (x : ℕ) : 
  (180 - (x + 40) = 3 * x) →  -- Condition 1 and 3
  x = 35                      -- The result we want to prove
  := by sorry

end minutes_before_noon_l789_78974


namespace investment_average_rate_l789_78938

/-- Proves that for a $6000 investment split between 3% and 5.5% interest rates
    with equal annual returns, the average interest rate is 3.88% -/
theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) :
  total = 6000 →
  rate1 = 0.03 →
  rate2 = 0.055 →
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x < total ∧
    rate1 * (total - x) = rate2 * x →
  (rate1 * (total - x) + rate2 * x) / total = 0.0388 :=
by sorry


end investment_average_rate_l789_78938


namespace arccos_gt_arctan_iff_l789_78991

theorem arccos_gt_arctan_iff (x : ℝ) : Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1) (1/2) := by
  sorry

end arccos_gt_arctan_iff_l789_78991


namespace quadratic_equation_roots_range_l789_78958

/-- The range of m for which the quadratic equation (m-3)x^2 + 4x + 1 = 0 has two real roots -/
theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (m - 3) * x₁^2 + 4 * x₁ + 1 = 0 ∧ (m - 3) * x₂^2 + 4 * x₂ + 1 = 0) ↔ 
  (m ≤ 7 ∧ m ≠ 3) :=
sorry

end quadratic_equation_roots_range_l789_78958


namespace right_triangle_to_square_l789_78936

theorem right_triangle_to_square (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  b = 10 →           -- longer leg is 10
  b = 2*a →          -- condition for forming a square
  a = 5 := by sorry

end right_triangle_to_square_l789_78936


namespace fence_cost_l789_78913

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 144 → price_per_foot = 58 → cost = 2784 → 
  cost = 4 * Real.sqrt area * price_per_foot := by
  sorry

#check fence_cost

end fence_cost_l789_78913


namespace p_squared_plus_98_composite_l789_78986

theorem p_squared_plus_98_composite (p : ℕ) (h : Prime p) : ¬ Prime (p^2 + 98) := by
  sorry

end p_squared_plus_98_composite_l789_78986


namespace range_of_m_l789_78931

-- Define proposition p
def p (m : ℝ) : Prop := ∀ x y : ℝ, x^2/2 + y^2/(m-1) = 1 → (m - 1 > 2)

-- Define proposition q
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + 4*m ≠ 0

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (¬p m) ∧ (p m ∨ q m) → m ∈ Set.Ioo (1/4 : ℝ) 3 :=
sorry

end range_of_m_l789_78931


namespace line_mb_equals_two_l789_78937

/-- Given a line with equation y = mx + b passing through points (0, 1) and (1, 3), prove that mb = 2 -/
theorem line_mb_equals_two (m b : ℝ) : 
  (1 = m * 0 + b) →  -- The line passes through (0, 1)
  (3 = m * 1 + b) →  -- The line passes through (1, 3)
  m * b = 2 := by
sorry

end line_mb_equals_two_l789_78937


namespace grapes_purchased_l789_78944

/-- Represents the price of grapes per kilogram -/
def grape_price : ℕ := 70

/-- Represents the price of mangoes per kilogram -/
def mango_price : ℕ := 55

/-- Represents the amount of mangoes purchased in kilograms -/
def mango_amount : ℕ := 11

/-- Represents the total amount paid -/
def total_paid : ℕ := 1165

/-- Theorem stating that the amount of grapes purchased is 8 kg -/
theorem grapes_purchased : ∃ (g : ℕ), g * grape_price + mango_amount * mango_price = total_paid ∧ g = 8 := by
  sorry

end grapes_purchased_l789_78944


namespace min_difference_l789_78921

open Real

noncomputable def f (x : ℝ) : ℝ := exp (x - 1)

noncomputable def g (x : ℝ) : ℝ := 1/2 + log (x/2)

theorem min_difference (a b : ℝ) (h : f a = g b) :
  ∃ (min : ℝ), min = 1 + log 2 ∧ ∀ (a' b' : ℝ), f a' = g b' → b' - a' ≥ min :=
sorry

end min_difference_l789_78921


namespace tim_initial_amount_l789_78941

/-- Tim's initial amount of money in cents -/
def initial_amount : ℕ := sorry

/-- Amount Tim paid for the candy bar in cents -/
def candy_bar_cost : ℕ := 45

/-- Amount Tim received as change in cents -/
def change_received : ℕ := 5

/-- Theorem stating that Tim's initial amount equals 50 cents -/
theorem tim_initial_amount : initial_amount = candy_bar_cost + change_received := by sorry

end tim_initial_amount_l789_78941


namespace arithmetic_calculation_l789_78955

theorem arithmetic_calculation : 5 * 12 + 6 * 11 + 13 * 5 + 7 * 9 = 254 := by
  sorry

end arithmetic_calculation_l789_78955


namespace min_value_w_l789_78935

theorem min_value_w (x y : ℝ) : 
  3 * x^2 + 5 * y^2 + 12 * x - 10 * y + 45 ≥ 28 ∧ 
  ∃ (a b : ℝ), 3 * a^2 + 5 * b^2 + 12 * a - 10 * b + 45 = 28 := by
sorry

end min_value_w_l789_78935


namespace donut_purchase_proof_l789_78940

/-- Represents the number of items purchased over the week -/
def total_items : ℕ := 4

/-- Price of a croissant in cents -/
def croissant_price : ℕ := 60

/-- Price of a donut in cents -/
def donut_price : ℕ := 90

/-- Represents the number of donuts purchased -/
def num_donuts : ℕ := sorry

/-- Represents the number of croissants purchased -/
def num_croissants : ℕ := total_items - num_donuts

/-- Total cost in cents -/
def total_cost : ℕ := num_donuts * donut_price + num_croissants * croissant_price

theorem donut_purchase_proof : 
  (num_donuts + num_croissants = total_items) ∧ 
  (total_cost % 100 = 0) ∧ 
  (num_donuts = 2) := by sorry

end donut_purchase_proof_l789_78940


namespace rotate_parabola_180_l789_78951

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a point 180 degrees around the origin -/
def rotate180 (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- The original parabola -/
def originalParabola : Parabola := ⟨1, -5, 9⟩

/-- Theorem stating that rotating the original parabola 180 degrees results in the new parabola -/
theorem rotate_parabola_180 :
  let (x, y) := rotate180 x y
  y = -(originalParabola.a * x^2 + originalParabola.b * x + originalParabola.c) :=
by sorry

end rotate_parabola_180_l789_78951


namespace sine_amplitude_l789_78985

/-- Given a sine function y = a * sin(bx + c) + d where a, b, c, and d are positive constants,
    if the graph oscillates between 5 and -3, then a = 4 -/
theorem sine_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) :
  a = 4 := by sorry

end sine_amplitude_l789_78985


namespace inequality_proof_l789_78948

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
  sorry

end inequality_proof_l789_78948


namespace pen_pencil_difference_l789_78972

theorem pen_pencil_difference :
  ∀ (pens pencils : ℕ),
    pens * 6 = pencils * 5 →  -- ratio of pens to pencils is 5:6
    pencils = 30 →            -- there are 30 pencils
    pencils - pens = 5        -- prove that there are 5 more pencils than pens
:= by sorry

end pen_pencil_difference_l789_78972


namespace hexadecagon_area_theorem_l789_78969

/-- A hexadecagon inscribed in a square with specific properties -/
structure InscribedHexadecagon where
  /-- The perimeter of the square in which the hexadecagon is inscribed -/
  square_perimeter : ℝ
  /-- The property that every side of the square is trisected twice equally -/
  trisected_twice : Prop

/-- The area of the inscribed hexadecagon -/
def hexadecagon_area (h : InscribedHexadecagon) : ℝ := sorry

/-- Theorem stating the area of the inscribed hexadecagon with given properties -/
theorem hexadecagon_area_theorem (h : InscribedHexadecagon) 
  (h_perimeter : h.square_perimeter = 160) : hexadecagon_area h = 1344 := by sorry

end hexadecagon_area_theorem_l789_78969


namespace specific_factory_production_l789_78997

/-- A factory that produces toys -/
structure ToyFactory where
  workingDaysPerWeek : ℕ
  dailyProduction : ℕ
  constantProduction : Prop

/-- Calculate the weekly production of a toy factory -/
def weeklyProduction (factory : ToyFactory) : ℕ :=
  factory.workingDaysPerWeek * factory.dailyProduction

/-- Theorem stating the weekly production of a specific factory -/
theorem specific_factory_production :
  ∀ (factory : ToyFactory),
    factory.workingDaysPerWeek = 4 →
    factory.dailyProduction = 1375 →
    factory.constantProduction →
    weeklyProduction factory = 5500 := by
  sorry

end specific_factory_production_l789_78997


namespace pencil_cartons_theorem_l789_78987

/-- Represents the purchase of school supplies -/
structure SchoolSupplies where
  pencil_boxes_per_carton : ℕ
  pencil_box_cost : ℕ
  marker_cartons : ℕ
  marker_boxes_per_carton : ℕ
  marker_carton_cost : ℕ
  total_spent : ℕ

/-- Calculates the number of pencil cartons bought -/
def pencil_cartons_bought (s : SchoolSupplies) : ℕ :=
  (s.total_spent - s.marker_cartons * s.marker_carton_cost) / (s.pencil_boxes_per_carton * s.pencil_box_cost)

/-- Theorem stating the number of pencil cartons bought -/
theorem pencil_cartons_theorem (s : SchoolSupplies) 
  (h1 : s.pencil_boxes_per_carton = 10)
  (h2 : s.pencil_box_cost = 2)
  (h3 : s.marker_cartons = 10)
  (h4 : s.marker_boxes_per_carton = 5)
  (h5 : s.marker_carton_cost = 4)
  (h6 : s.total_spent = 600) :
  pencil_cartons_bought s = 10 := by
  sorry

end pencil_cartons_theorem_l789_78987


namespace f_symmetry_l789_78929

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem f_symmetry (a : ℝ) : f a - f (-a) = 0 := by
  sorry

end f_symmetry_l789_78929


namespace quadratic_function_uniqueness_quadratic_function_coefficient_range_l789_78957

-- Part 1
def is_symmetric_about_negative_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-1 - x) = f (-1 + x)

theorem quadratic_function_uniqueness
  (f : ℝ → ℝ)
  (h1 : ∃ a b c, ∀ x, f x = a * x^2 + b * x + c)
  (h2 : is_symmetric_about_negative_one f)
  (h3 : f 0 = 1)
  (h4 : ∃ x_min, ∀ x, f x ≥ f x_min ∧ f x_min = 0) :
  ∀ x, f x = (x + 1)^2 := by sorry

-- Part 2
theorem quadratic_function_coefficient_range
  (b : ℝ)
  (h : ∀ x ∈ Set.Ioo 0 1, |x^2 + b*x| ≤ 1) :
  b ∈ Set.Icc (-2) 0 := by sorry

end quadratic_function_uniqueness_quadratic_function_coefficient_range_l789_78957


namespace linear_relationship_values_l789_78964

/-- Given a linear relationship between x and y, prove the values of y for specific x values -/
theorem linear_relationship_values (x y : ℝ) :
  (y = 3 * x - 1) →
  (x = 1 → y = 2) ∧ (x = 5 → y = 14) := by
  sorry

end linear_relationship_values_l789_78964


namespace locus_of_centers_l789_78923

/-- The locus of centers of circles externally tangent to C1 and internally tangent to C2 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, 
    (a^2 + b^2 = (r + 1)^2) ∧ 
    ((a - 2)^2 + b^2 = (5 - r)^2)) →
  8 * a^2 + 9 * b^2 - 16 * a - 64 = 0 :=
by sorry

end locus_of_centers_l789_78923


namespace johnson_family_seating_l789_78970

/-- The number of ways to seat 5 boys and 4 girls in a row of 9 chairs such that at least 2 boys are next to each other -/
def seating_arrangements (num_boys num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - (Nat.factorial num_boys * Nat.factorial num_girls)

theorem johnson_family_seating :
  seating_arrangements 5 4 = 360000 := by
  sorry

end johnson_family_seating_l789_78970


namespace perpendicular_sum_maximized_l789_78910

theorem perpendicular_sum_maximized (r : ℝ) (α : ℝ) :
  let s := r * (Real.sin α + Real.cos α)
  ∀ β, 0 ≤ β ∧ β ≤ 2 * Real.pi → s ≤ r * (Real.sin (Real.pi / 4) + Real.cos (Real.pi / 4)) :=
by sorry

end perpendicular_sum_maximized_l789_78910


namespace book_sale_percentage_gain_l789_78976

/-- Calculates the percentage gain for a book sale given the number of books purchased,
    the number of books whose selling price equals the total cost price,
    and the total number of books purchased. -/
def calculatePercentageGain (booksPurchased : ℕ) (booksSoldForCost : ℕ) : ℚ :=
  ((booksPurchased : ℚ) / booksSoldForCost - 1) * 100

/-- Theorem stating that the percentage gain for the given book sale scenario is (3/7) * 100. -/
theorem book_sale_percentage_gain :
  calculatePercentageGain 50 35 = (3/7) * 100 := by
  sorry

#eval calculatePercentageGain 50 35

end book_sale_percentage_gain_l789_78976


namespace power_four_squared_cubed_minus_four_l789_78988

theorem power_four_squared_cubed_minus_four : (4^2)^3 - 4 = 4092 := by
  sorry

end power_four_squared_cubed_minus_four_l789_78988


namespace quadratic_roots_difference_squared_l789_78950

theorem quadratic_roots_difference_squared :
  ∀ α β : ℝ, 
  (α^2 - 3*α + 1 = 0) → 
  (β^2 - 3*β + 1 = 0) → 
  (α ≠ β) →
  (α - β)^2 = 5 := by
  sorry

end quadratic_roots_difference_squared_l789_78950


namespace complex_fraction_equals_i_l789_78973

theorem complex_fraction_equals_i : (Complex.I + 3) / (1 - 3 * Complex.I) = Complex.I := by
  sorry

end complex_fraction_equals_i_l789_78973


namespace race_even_distance_l789_78925

/-- The distance Alex and Max were even at the beginning of the race -/
def even_distance : ℕ := sorry

/-- The total race distance in feet -/
def total_race_distance : ℕ := 5000

/-- The distance left for Max to catch up at the end of the race -/
def distance_left : ℕ := 3890

/-- Alex's first lead over Max in feet -/
def alex_first_lead : ℕ := 300

/-- Max's lead over Alex in feet -/
def max_lead : ℕ := 170

/-- Alex's final lead over Max in feet -/
def alex_final_lead : ℕ := 440

theorem race_even_distance :
  even_distance = 540 ∧
  even_distance + alex_first_lead - max_lead + alex_final_lead = total_race_distance - distance_left :=
by sorry

end race_even_distance_l789_78925


namespace infinitely_many_prime_divisors_l789_78967

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => (a n)^2 + 1

def is_prime_divisor_of_sequence (p : ℕ) : Prop :=
  ∃ n : ℕ, p.Prime ∧ p ∣ a n

theorem infinitely_many_prime_divisors :
  ∀ m : ℕ, ∃ p : ℕ, p > m ∧ is_prime_divisor_of_sequence p :=
sorry

end infinitely_many_prime_divisors_l789_78967


namespace clarence_oranges_l789_78922

-- Define the initial number of oranges
def initial_oranges : ℝ := 5.0

-- Define the number of oranges given away
def oranges_given : ℝ := 3.0

-- Define the number of Skittles bought (not used in the calculation, but mentioned in the problem)
def skittles_bought : ℝ := 9.0

-- Define the function to calculate the remaining oranges
def remaining_oranges : ℝ := initial_oranges - oranges_given

-- Theorem to prove
theorem clarence_oranges : remaining_oranges = 2.0 := by
  sorry

end clarence_oranges_l789_78922


namespace solutions_of_x_fourth_minus_16_l789_78908

theorem solutions_of_x_fourth_minus_16 :
  {x : ℂ | x^4 - 16 = 0} = {2, -2, 2*I, -2*I} := by sorry

end solutions_of_x_fourth_minus_16_l789_78908


namespace sum_of_common_ratios_is_three_l789_78978

/-- Given two nonconstant geometric sequences with different common ratios,
    if a specific condition is met, prove that the sum of their common ratios is 3. -/
theorem sum_of_common_ratios_is_three
  (k a₂ a₃ b₂ b₃ : ℝ)
  (hk : k ≠ 0)
  (ha : ∃ p : ℝ, p ≠ 1 ∧ a₂ = k * p ∧ a₃ = k * p^2)
  (hb : ∃ r : ℝ, r ≠ 1 ∧ b₂ = k * r ∧ b₃ = k * r^2)
  (hdiff : ∀ p r : ℝ, (a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2) → p ≠ r)
  (hcond : a₃ - b₃ = 3 * (a₂ - b₂)) :
  ∃ p r : ℝ, (a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2) ∧ p + r = 3 :=
by sorry

end sum_of_common_ratios_is_three_l789_78978


namespace rectangle_area_42_implies_y_7_l789_78930

/-- Rectangle PQRS with vertices P(0, 0), Q(0, 6), R(y, 6), and S(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of a rectangle is the product of its length and width -/
def area (rect : Rectangle) : ℝ := 6 * rect.y

theorem rectangle_area_42_implies_y_7 (rect : Rectangle) (h_area : area rect = 42) : rect.y = 7 := by
  sorry

end rectangle_area_42_implies_y_7_l789_78930


namespace acme_cheaper_at_min_shirts_l789_78954

/-- Acme T-Shirt Company's pricing function -/
def acme_price (x : ℕ) : ℕ := 80 + 10 * x

/-- Beta T-Shirt Company's pricing function -/
def beta_price (x : ℕ) : ℕ := 20 + 15 * x

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_acme_cheaper : ℕ := 13

theorem acme_cheaper_at_min_shirts :
  acme_price min_shirts_acme_cheaper < beta_price min_shirts_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_acme_cheaper → acme_price n ≥ beta_price n :=
by sorry

end acme_cheaper_at_min_shirts_l789_78954


namespace locus_is_circle_l789_78906

/-- An equilateral triangle in a 2D plane -/
structure EquilateralTriangle where
  s : ℝ  -- side length
  A : ℝ × ℝ  -- coordinates of vertex A
  B : ℝ × ℝ  -- coordinates of vertex B
  C : ℝ × ℝ  -- coordinates of vertex C
  is_equilateral : 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = s^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = s^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = s^2

/-- The locus of points with constant sum of squared distances to triangle vertices -/
def ConstantSumLocus (tri : EquilateralTriangle) (a : ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | 
    (P.1 - tri.A.1)^2 + (P.2 - tri.A.2)^2 + 
    (P.1 - tri.B.1)^2 + (P.2 - tri.B.2)^2 + 
    (P.1 - tri.C.1)^2 + (P.2 - tri.C.2)^2 = a}

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

theorem locus_is_circle (tri : EquilateralTriangle) (a : ℝ) (h : a > tri.s^2) :
  ∃ (c : Circle), ConstantSumLocus tri a = {P : ℝ × ℝ | (P.1 - c.center.1)^2 + (P.2 - c.center.2)^2 = c.radius^2} :=
sorry

end locus_is_circle_l789_78906


namespace quadratic_function_value_l789_78977

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

-- Define the derivative of f(x)
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 8 * x - m

theorem quadratic_function_value : 
  ∀ (m : ℝ), 
  (∀ x : ℝ, x ≥ -2 → (f_derivative m) x ≥ 0) →  -- f(x) is increasing on [−2, +∞)
  (∀ x : ℝ, x < -2 → (f_derivative m) x < 0) →  -- f(x) is decreasing on (-∞, −2)
  f m 1 = 25 := by
  sorry

end quadratic_function_value_l789_78977


namespace spoiled_cross_to_square_l789_78995

/-- Represents a symmetrical Greek cross -/
structure GreekCross where
  arm_length : ℝ
  arm_width : ℝ
  symmetrical : arm_length > 0 ∧ arm_width > 0

/-- Represents a square -/
structure Square where
  side_length : ℝ
  is_positive : side_length > 0

/-- Represents a Greek cross with a square cut out -/
structure SpoiledGreekCross where
  cross : GreekCross
  cut_out : Square
  fits_end : cut_out.side_length = cross.arm_width

/-- Represents a piece obtained from cutting the spoiled Greek cross -/
structure Piece where
  area : ℝ
  is_positive : area > 0

/-- Theorem stating that a spoiled Greek cross can be cut into four pieces
    that can be reassembled into a square -/
theorem spoiled_cross_to_square (sc : SpoiledGreekCross) :
  ∃ (p1 p2 p3 p4 : Piece) (result : Square),
    p1.area + p2.area + p3.area + p4.area = result.side_length ^ 2 :=
sorry

end spoiled_cross_to_square_l789_78995


namespace smallest_base_not_divisible_by_five_l789_78946

theorem smallest_base_not_divisible_by_five : 
  ∃ (b : ℕ), b > 2 ∧ b = 6 ∧ ¬(5 ∣ (2 * b^3 - 1)) ∧
  ∀ (k : ℕ), 2 < k ∧ k < b → (5 ∣ (2 * k^3 - 1)) :=
by sorry

end smallest_base_not_divisible_by_five_l789_78946


namespace inequality_solution_length_l789_78914

theorem inequality_solution_length (k : ℝ) : 
  (∃ a b : ℝ, a < b ∧ 
    (∀ x : ℝ, a ≤ x ∧ x ≤ b ↔ 1 ≤ x^2 - 3*x + k ∧ x^2 - 3*x + k ≤ 5) ∧
    b - a = 8) →
  k = 9/4 := by sorry

end inequality_solution_length_l789_78914


namespace point_on_transformed_plane_l789_78999

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Apply a similarity transformation to a plane -/
def similarityTransform (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Check if a point lies on a plane -/
def pointOnPlane (point : Point) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem point_on_transformed_plane :
  let originalPlane : Plane := { a := 3, b := -1, c := 2, d := 4 }
  let k : ℝ := 1/2
  let transformedPlane := similarityTransform originalPlane k
  let pointA : Point := { x := -1, y := 1, z := 1 }
  pointOnPlane pointA transformedPlane := by sorry

end point_on_transformed_plane_l789_78999


namespace min_value_S_n_l789_78984

/-- The sum of the first n terms of the sequence -/
def S_n (n : ℕ+) : ℤ := n^2 - 12*n

/-- The minimum value of S_n for positive integers n -/
def min_S_n : ℤ := -36

theorem min_value_S_n :
  ∀ n : ℕ+, S_n n ≥ min_S_n ∧ ∃ m : ℕ+, S_n m = min_S_n :=
sorry

end min_value_S_n_l789_78984


namespace projectile_meeting_time_l789_78932

/-- Given two objects traveling towards each other, calculate the time it takes for them to meet. -/
theorem projectile_meeting_time 
  (initial_distance : ℝ) 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (h1 : initial_distance = 1182) 
  (h2 : speed1 = 460) 
  (h3 : speed2 = 525) : 
  (initial_distance / (speed1 + speed2)) * 60 = 72 := by
  sorry

end projectile_meeting_time_l789_78932


namespace circle_roll_position_l789_78902

theorem circle_roll_position (d : ℝ) (start : ℝ) (h_d : d = 1) (h_start : start = 3) : 
  let circumference := π * d
  let end_position := start - circumference
  end_position = 3 - π := by
sorry

end circle_roll_position_l789_78902


namespace eleven_twelfths_squared_between_half_and_one_l789_78904

theorem eleven_twelfths_squared_between_half_and_one :
  (11 / 12 : ℚ)^2 > 1/2 ∧ (11 / 12 : ℚ)^2 < 1 := by
  sorry

end eleven_twelfths_squared_between_half_and_one_l789_78904


namespace slope_of_line_l789_78996

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 := by
  sorry

end slope_of_line_l789_78996


namespace smallest_AAB_value_l789_78960

def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem smallest_AAB_value :
  ∀ A B : ℕ,
  is_digit A →
  is_digit B →
  two_digit (10 * A + B) →
  three_digit (100 * A + 10 * A + B) →
  (10 * A + B : ℚ) = (1 / 7) * (100 * A + 10 * A + B) →
  ∀ A' B' : ℕ,
  is_digit A' →
  is_digit B' →
  two_digit (10 * A' + B') →
  three_digit (100 * A' + 10 * A' + B') →
  (10 * A' + B' : ℚ) = (1 / 7) * (100 * A' + 10 * A' + B') →
  100 * A + 10 * A + B ≤ 100 * A' + 10 * A' + B' →
  100 * A + 10 * A + B = 332 :=
by sorry

end smallest_AAB_value_l789_78960


namespace discounted_notebooks_cost_l789_78982

/-- The total cost of purchasing discounted notebooks -/
theorem discounted_notebooks_cost 
  (x : ℝ) -- original price of a notebook in yuan
  (y : ℝ) -- discount amount in yuan
  : 5 * (x - y) = 5 * x - 5 * y := by
  sorry

end discounted_notebooks_cost_l789_78982


namespace vectors_form_basis_vectors_not_collinear_basis_iff_not_collinear_l789_78901

def e₁ : Fin 2 → ℝ := ![(-1 : ℝ), 2]
def e₂ : Fin 2 → ℝ := ![(5 : ℝ), -1]

theorem vectors_form_basis (v : Fin 2 → ℝ) : 
  ∃ (a b : ℝ), v = fun i => a * e₁ i + b * e₂ i :=
sorry

theorem vectors_not_collinear : 
  e₁ 0 * e₂ 1 ≠ e₁ 1 * e₂ 0 :=
sorry

theorem basis_iff_not_collinear :
  (∀ (v : Fin 2 → ℝ), ∃ (a b : ℝ), v = fun i => a * e₁ i + b * e₂ i) ↔
  (e₁ 0 * e₂ 1 ≠ e₁ 1 * e₂ 0) :=
sorry

end vectors_form_basis_vectors_not_collinear_basis_iff_not_collinear_l789_78901


namespace three_tangents_implies_a_greater_than_three_l789_78918

/-- A curve of the form y = x³ + ax² + bx -/
structure Curve where
  a : ℝ
  b : ℝ

/-- The number of tangent lines to the curve that pass through (0,-1) -/
noncomputable def numTangentLines (c : Curve) : ℕ := sorry

/-- Theorem stating that if there are exactly three tangent lines passing through (0,-1), then a > 3 -/
theorem three_tangents_implies_a_greater_than_three (c : Curve) :
  numTangentLines c = 3 → c.a > 3 := by sorry

end three_tangents_implies_a_greater_than_three_l789_78918


namespace motorboat_travel_time_l789_78962

/-- Represents the time (in hours) for the motorboat to travel from dock C to dock D -/
def motorboat_time_to_D : ℝ := 5.5

/-- Represents the total journey time in hours -/
def total_journey_time : ℝ := 12

/-- Represents the time (in hours) the motorboat stops at dock E -/
def stop_time_at_E : ℝ := 1

theorem motorboat_travel_time :
  motorboat_time_to_D = (total_journey_time - stop_time_at_E) / 2 :=
sorry

end motorboat_travel_time_l789_78962


namespace water_bottles_stolen_solve_water_bottle_theft_l789_78966

theorem water_bottles_stolen (initial_bottles : ℕ) (lost_bottles : ℕ) (stickers_per_bottle : ℕ) (total_stickers : ℕ) : ℕ :=
  let remaining_after_loss := initial_bottles - lost_bottles
  let remaining_after_theft := total_stickers / stickers_per_bottle
  remaining_after_loss - remaining_after_theft

theorem solve_water_bottle_theft : water_bottles_stolen 10 2 3 21 = 1 := by
  sorry

end water_bottles_stolen_solve_water_bottle_theft_l789_78966


namespace pure_imaginary_complex_number_l789_78983

theorem pure_imaginary_complex_number (m : ℝ) : 
  (m^2 - 3*m : ℂ) + (m^2 - 5*m + 6 : ℂ)*Complex.I = Complex.I * ((m^2 - 5*m + 6 : ℂ)) → m = 0 := by
  sorry

end pure_imaginary_complex_number_l789_78983


namespace smallest_value_l789_78915

def x : ℝ := 4
def y : ℝ := 2

theorem smallest_value : 
  min (x + y) (min (x * y) (min (x - y) (min (x / y) (y / x)))) = y / x :=
by sorry

end smallest_value_l789_78915


namespace min_club_members_l789_78939

theorem min_club_members : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), m > 0 ∧ 2/5 < m/n ∧ m/n < 1/2) ∧ 
  (∀ (k : ℕ), 0 < k ∧ k < n → ¬∃ (j : ℕ), j > 0 ∧ 2/5 < j/k ∧ j/k < 1/2) ∧ n = 7 := by
  sorry

end min_club_members_l789_78939


namespace sum_of_cubes_squares_and_product_l789_78919

theorem sum_of_cubes_squares_and_product : (3 + 7)^3 + (3^2 + 7^2) + 3 * 7 = 1079 := by
  sorry

end sum_of_cubes_squares_and_product_l789_78919


namespace tangent_and_normal_lines_l789_78992

-- Define the curve
def x (t : ℝ) : ℝ := t - t^4
def y (t : ℝ) : ℝ := t^2 - t^3

-- Define the parameter value
def t₀ : ℝ := 1

-- State the theorem
theorem tangent_and_normal_lines :
  let x₀ := x t₀
  let y₀ := y t₀
  let dx := deriv x t₀
  let dy := deriv y t₀
  let m_tangent := dy / dx
  let m_normal := -1 / m_tangent
  (∀ t : ℝ, y t - y₀ = m_tangent * (x t - x₀)) ∧
  (∀ t : ℝ, y t - y₀ = m_normal * (x t - x₀)) := by
  sorry

end tangent_and_normal_lines_l789_78992


namespace decimal_123_in_base7_has_three_consecutive_digits_l789_78963

/-- Represents a number in base 7 --/
def Base7 := Nat

/-- Converts a decimal number to base 7 --/
def toBase7 (n : Nat) : Base7 :=
  sorry

/-- Checks if a Base7 number has three consecutive digits --/
def hasThreeConsecutiveDigits (n : Base7) : Prop :=
  sorry

/-- The decimal number we're working with --/
def decimalNumber : Nat := 123

theorem decimal_123_in_base7_has_three_consecutive_digits :
  hasThreeConsecutiveDigits (toBase7 decimalNumber) :=
sorry

end decimal_123_in_base7_has_three_consecutive_digits_l789_78963


namespace complex_magnitude_l789_78945

theorem complex_magnitude (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l789_78945


namespace herd_size_l789_78959

theorem herd_size (herd : ℕ) : 
  (herd / 3 + herd / 6 + herd / 8 + herd / 24 + 15 = herd) → 
  herd = 45 := by
  sorry

end herd_size_l789_78959


namespace cyclist_distance_l789_78989

/-- Represents a cyclist's journey -/
structure CyclistJourney where
  v : ℝ  -- speed in mph
  t : ℝ  -- time in hours
  d : ℝ  -- distance in miles

/-- Conditions for the cyclist's journey -/
def journeyConditions (j : CyclistJourney) : Prop :=
  j.d = j.v * j.t ∧
  j.d = (j.v + 1) * (3/4 * j.t) ∧
  j.d = (j.v - 1) * (j.t + 3)

theorem cyclist_distance (j : CyclistJourney) 
  (h : journeyConditions j) : j.d = 36 := by
  sorry

end cyclist_distance_l789_78989


namespace otimes_h_otimes_h_l789_78907

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x^2 - y

-- Theorem statement
theorem otimes_h_otimes_h (h : ℝ) : otimes h (otimes h h) = h := by
  sorry

end otimes_h_otimes_h_l789_78907


namespace parallelogram_perimeter_l789_78990

theorem parallelogram_perimeter (a b : ℝ) (ha : a = Real.sqrt 20) (hb : b = Real.sqrt 125) :
  2 * (a + b) = 14 * Real.sqrt 5 := by
  sorry

end parallelogram_perimeter_l789_78990


namespace cos_330_degrees_l789_78943

theorem cos_330_degrees : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l789_78943
