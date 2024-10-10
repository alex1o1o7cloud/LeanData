import Mathlib

namespace banana_arrangements_l963_96347

-- Define the word and its properties
def banana_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

-- Theorem statement
theorem banana_arrangements : 
  (banana_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end banana_arrangements_l963_96347


namespace fifth_minus_fourth_cube_volume_l963_96381

/-- The volume of a cube with side length n -/
def cube_volume (n : ℕ) : ℕ := n ^ 3

/-- The difference in volume between two cubes in the sequence -/
def volume_difference (m n : ℕ) : ℕ := cube_volume m - cube_volume n

theorem fifth_minus_fourth_cube_volume : volume_difference 5 4 = 61 := by
  sorry

end fifth_minus_fourth_cube_volume_l963_96381


namespace intersection_A_B_l963_96313

def A : Set Int := {1, 2, 3, 4, 5}

def B : Set Int := {x | (x - 1) / (4 - x) > 0}

theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end intersection_A_B_l963_96313


namespace sarah_apples_left_l963_96370

def apples_left (initial : ℕ) (teachers : ℕ) (friends : ℕ) (eaten : ℕ) : ℕ :=
  initial - (teachers + friends + eaten)

theorem sarah_apples_left :
  apples_left 25 16 5 1 = 3 := by
  sorry

end sarah_apples_left_l963_96370


namespace polynomial_division_remainder_l963_96394

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℂ, X^44 + X^33 + X^22 + X^11 + 1 = (X^4 + X^3 + X^2 + X + 1) * q :=
by
  sorry

end polynomial_division_remainder_l963_96394


namespace stationery_box_sheets_l963_96335

/-- Represents the contents of a stationery box -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents a person who uses the stationery box -/
structure Person where
  name : String
  box : StationeryBox
  pagesPerLetter : ℕ

theorem stationery_box_sheets (ann sue : Person) : 
  ann.name = "Ann" →
  sue.name = "Sue" →
  ann.pagesPerLetter = 1 →
  sue.pagesPerLetter = 3 →
  ann.box = sue.box →
  ann.box.sheets - ann.box.envelopes = 50 →
  sue.box.envelopes - sue.box.sheets / 3 = 50 →
  ann.box.sheets = 150 := by
sorry

end stationery_box_sheets_l963_96335


namespace steven_has_14_peaches_l963_96356

/-- The number of peaches each person has -/
structure PeachCount where
  steven : ℕ
  jake : ℕ
  jill : ℕ

/-- Given conditions about peach counts -/
def peach_conditions (p : PeachCount) : Prop :=
  p.jake + 6 = p.steven ∧ 
  p.jake = p.jill + 3 ∧ 
  p.jill = 5

/-- Theorem stating Steven has 14 peaches -/
theorem steven_has_14_peaches (p : PeachCount) 
  (h : peach_conditions p) : p.steven = 14 := by
  sorry

end steven_has_14_peaches_l963_96356


namespace zeros_of_log_linear_function_l963_96332

open Real

theorem zeros_of_log_linear_function (m : ℝ) (x₁ x₂ : ℝ) 
  (hm : m > 0) 
  (hx : x₁ < x₂) 
  (hz₁ : m * log x₁ = x₁) 
  (hz₂ : m * log x₂ = x₂) : 
  x₁ < exp 1 ∧ exp 1 < x₂ := by
sorry

end zeros_of_log_linear_function_l963_96332


namespace unit_circle_problem_l963_96383

theorem unit_circle_problem (y₀ : ℝ) (B : ℝ × ℝ) :
  (-3/5)^2 + y₀^2 = 1 →  -- A is on the unit circle
  y₀ > 0 →  -- A is in the second quadrant
  ((-3/5) * B.1 + y₀ * B.2) / ((-3/5)^2 + y₀^2) = 1/2 →  -- Angle between OA and OB is 60°
  B.1^2 + B.2^2 = 4 →  -- |OB| = 2
  (2 * y₀^2 + 2 * (-3/5) * y₀ = 8/25) ∧  -- Part 1: 2sin²α + sin2α = 8/25
  ((B.2 - y₀) / (B.1 + 3/5) = 3/4)  -- Part 2: Slope of AB = 3/4
  := by sorry

end unit_circle_problem_l963_96383


namespace average_of_three_quantities_l963_96363

theorem average_of_three_quantities 
  (total_count : Nat) 
  (total_average : ℚ) 
  (subset_count : Nat) 
  (subset_average : ℚ) 
  (h1 : total_count = 5) 
  (h2 : total_average = 11) 
  (h3 : subset_count = 2) 
  (h4 : subset_average = 21.5) : 
  (total_count * total_average - subset_count * subset_average) / (total_count - subset_count) = 4 := by
  sorry

end average_of_three_quantities_l963_96363


namespace base9_521_equals_base10_424_l963_96300

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 9^2 + tens * 9^1 + ones * 9^0

/-- The theorem stating that 521 in base 9 is equal to 424 in base 10 -/
theorem base9_521_equals_base10_424 :
  base9ToBase10 5 2 1 = 424 := by
  sorry

end base9_521_equals_base10_424_l963_96300


namespace problem_solution_l963_96341

theorem problem_solution :
  ∀ a b c d : ℝ,
  (100 * a = 35^2 - 15^2) →
  ((a - 1)^2 = 3^(4 * b)) →
  (b^2 + c * b - 5 = 0) →
  (∃ k : ℝ, 2 * (x^2) + 3 * x + 4 * d = (x + c) * k) →
  (a = 10 ∧ b = 1 ∧ c = 4 ∧ d = -5) :=
by sorry

end problem_solution_l963_96341


namespace company_blocks_l963_96339

/-- Calculates the number of blocks in a company based on gift budget and workers per block -/
theorem company_blocks (total_amount : ℝ) (gift_worth : ℝ) (workers_per_block : ℝ) :
  total_amount = 4000 ∧ gift_worth = 4 ∧ workers_per_block = 100 →
  (total_amount / gift_worth) / workers_per_block = 10 := by
  sorry

end company_blocks_l963_96339


namespace exactly_two_primes_probability_l963_96390

-- Define a 12-sided die
def Die := Fin 12

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := sorry

-- Define the probability of rolling a prime number on a single die
def probPrime : ℚ := 5 / 12

-- Define the probability of not rolling a prime number on a single die
def probNotPrime : ℚ := 7 / 12

-- Define the number of dice
def numDice : ℕ := 3

-- Define the number of dice that should show prime numbers
def numPrimeDice : ℕ := 2

-- Theorem statement
theorem exactly_two_primes_probability :
  (numDice.choose numPrimeDice : ℚ) * probPrime ^ numPrimeDice * probNotPrime ^ (numDice - numPrimeDice) = 525 / 1728 :=
sorry

end exactly_two_primes_probability_l963_96390


namespace difference_of_squares_l963_96397

theorem difference_of_squares (m : ℝ) : m^2 - 9 = (m + 3) * (m - 3) := by
  sorry

end difference_of_squares_l963_96397


namespace max_visible_sum_is_164_l963_96395

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- The set of numbers used to form each cube --/
def cube_numbers : Finset ℕ := {1, 2, 4, 8, 16, 32}

/-- A cube is valid if it uses exactly the numbers in cube_numbers --/
def valid_cube (c : Cube) : Prop :=
  (Finset.image c.faces (Finset.univ : Finset (Fin 6))) = cube_numbers

/-- The sum of visible faces when a cube is stacked --/
def visible_sum (c : Cube) (top : Bool) : ℕ :=
  if top then
    c.faces 0 + c.faces 1 + c.faces 2 + c.faces 3 + c.faces 4
  else
    c.faces 1 + c.faces 2 + c.faces 3 + c.faces 4

/-- The theorem to be proved --/
theorem max_visible_sum_is_164 :
  ∃ (c1 c2 c3 : Cube),
    valid_cube c1 ∧ valid_cube c2 ∧ valid_cube c3 ∧
    visible_sum c1 false + visible_sum c2 false + visible_sum c3 true = 164 ∧
    ∀ (d1 d2 d3 : Cube),
      valid_cube d1 → valid_cube d2 → valid_cube d3 →
      visible_sum d1 false + visible_sum d2 false + visible_sum d3 true ≤ 164 := by
  sorry

end max_visible_sum_is_164_l963_96395


namespace geometric_progression_relation_l963_96308

/-- Given two geometric progressions, prove that their first terms are related as stated. -/
theorem geometric_progression_relation (a b q : ℝ) (n : ℕ) (h : q ≠ 1) :
  (a * (q^(2*n) - 1)) / (q - 1) = (b * (q^(2*n) - 1)) / (q^2 - 1) →
  b = a + a * q :=
by sorry

end geometric_progression_relation_l963_96308


namespace problem_statement_l963_96334

theorem problem_statement (a b : ℝ) 
  (h1 : |a| = 4) 
  (h2 : |b| = 6) : 
  (ab > 0 → (a - b = 2 ∨ a - b = -2)) ∧ 
  (|a + b| = -(a + b) → (a + b = -10 ∨ a + b = -2)) := by
  sorry

end problem_statement_l963_96334


namespace correct_calculation_l963_96312

theorem correct_calculation (x : ℚ) : x - 13/5 = 9/7 → x + 13/5 = 227/35 := by
  sorry

end correct_calculation_l963_96312


namespace lunch_meeting_probability_l963_96391

/-- The probability of Janet and Donald meeting for lunch -/
theorem lunch_meeting_probability :
  let arrival_interval : ℝ := 60
  let janet_wait_time : ℝ := 15
  let donald_wait_time : ℝ := 5
  let meeting_condition (x y : ℝ) : Prop := |x - y| ≤ min donald_wait_time janet_wait_time
  let total_area : ℝ := arrival_interval ^ 2
  let meeting_area : ℝ := arrival_interval * (2 * min donald_wait_time janet_wait_time)
  (meeting_area / total_area : ℝ) = 1/6 := by
sorry

end lunch_meeting_probability_l963_96391


namespace simplify_expression_l963_96362

theorem simplify_expression (a b : ℝ) : 
  -2 * (a^3 - 3*b^2) + 4 * (-b^2 + a^3) = 2*a^3 + 2*b^2 := by
  sorry

end simplify_expression_l963_96362


namespace bakery_boxes_l963_96349

theorem bakery_boxes (total_muffins : ℕ) (muffins_per_box : ℕ) (available_boxes : ℕ) : 
  total_muffins = 95 → 
  muffins_per_box = 5 → 
  available_boxes = 10 → 
  (total_muffins - available_boxes * muffins_per_box + muffins_per_box - 1) / muffins_per_box = 9 := by
  sorry

end bakery_boxes_l963_96349


namespace total_candies_is_96_l963_96368

/-- The number of candies Adam has -/
def adam_candies : ℕ := 6

/-- The number of candies James has -/
def james_candies : ℕ := 3 * adam_candies

/-- The number of candies Rubert has -/
def rubert_candies : ℕ := 4 * james_candies

/-- The total number of candies -/
def total_candies : ℕ := adam_candies + james_candies + rubert_candies

theorem total_candies_is_96 : total_candies = 96 := by
  sorry

end total_candies_is_96_l963_96368


namespace triangle_area_from_parametric_lines_l963_96388

/-- The area of a triangle formed by two points on given lines and the origin -/
theorem triangle_area_from_parametric_lines (t s : ℝ) : 
  let l : ℝ × ℝ → Prop := λ p => ∃ t, p.1 = 3 + 5*t ∧ p.2 = 2 + 4*t
  let m : ℝ × ℝ → Prop := λ p => ∃ s, p.1 = 2 + 5*s ∧ p.2 = 3 + 4*s
  let C : ℝ × ℝ := (3 + 5*t, 2 + 4*t)
  let D : ℝ × ℝ := (2 + 5*s, 3 + 4*s)
  let O : ℝ × ℝ := (0, 0)
  l C → m D → 
  (1/2 : ℝ) * |5 + 2*s + 7*t| = 
  (1/2 : ℝ) * |C.1 * D.2 - C.2 * D.1| :=
by sorry

end triangle_area_from_parametric_lines_l963_96388


namespace fourth_term_is_seven_l963_96359

-- Define the sequence sum function
def S (n : ℕ) : ℕ := n^2

-- Define the sequence term function
def a (n : ℕ) : ℤ := S n - S (n-1)

-- Theorem statement
theorem fourth_term_is_seven : a 4 = 7 := by
  sorry

end fourth_term_is_seven_l963_96359


namespace car_distance_proof_l963_96371

/-- Proves that the distance a car needs to cover is 630 km, given the original time, 
    new time factor, and new speed. -/
theorem car_distance_proof (original_time : ℝ) (new_time_factor : ℝ) (new_speed : ℝ) : 
  original_time = 6 → 
  new_time_factor = 3 / 2 → 
  new_speed = 70 → 
  original_time * new_time_factor * new_speed = 630 := by
  sorry

#check car_distance_proof

end car_distance_proof_l963_96371


namespace greatest_integer_satisfying_inequality_l963_96382

theorem greatest_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ y : ℤ, 4 * |y| - 6 < 34 → y ≤ x) ∧ (4 * |x| - 6 < 34) :=
by sorry

end greatest_integer_satisfying_inequality_l963_96382


namespace bus_interval_l963_96325

/-- Given a circular bus route where two buses have an interval of 21 minutes between them,
    prove that three buses on the same route will have an interval of 14 minutes between them. -/
theorem bus_interval (total_time : ℕ) (two_bus_interval : ℕ) (three_bus_interval : ℕ) : 
  two_bus_interval = 21 → 
  total_time = 2 * two_bus_interval → 
  three_bus_interval = total_time / 3 → 
  three_bus_interval = 14 := by
  sorry

#eval 42 / 3  -- This should output 14

end bus_interval_l963_96325


namespace find_k_l963_96367

theorem find_k (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h1 : a^2 + b^2 + c^2 = 49)
    (h2 : x^2 + y^2 + z^2 = 64)
    (h3 : a*x + b*y + c*z = 56)
    (h4 : ∃ k, a = k*x ∧ b = k*y ∧ c = k*z) :
  ∃ k, a = k*x ∧ b = k*y ∧ c = k*z ∧ k = 7/8 := by
sorry

end find_k_l963_96367


namespace function_inequality_l963_96351

theorem function_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 3 * x + 2) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 2| < b → |f x + 4| < a) ↔
  b ≤ a / 3 :=
by sorry

end function_inequality_l963_96351


namespace triangle_angle_identity_l963_96385

theorem triangle_angle_identity (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_less_than_pi : α < Real.pi ∧ β < Real.pi ∧ γ < Real.pi) : 
  (Real.cos α) / (Real.sin β * Real.sin γ) + 
  (Real.cos β) / (Real.sin α * Real.sin γ) + 
  (Real.cos γ) / (Real.sin α * Real.sin β) = 2 := by
sorry


end triangle_angle_identity_l963_96385


namespace max_value_expression_l963_96326

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (⨆ x, 2 * (a - x) * (x + c * Real.sqrt (x^2 + b^2))) = a^2 + c^2 * b^2 := by
  sorry

end max_value_expression_l963_96326


namespace balloon_count_l963_96375

/-- The number of filled water balloons Max and Zach have in total -/
def total_balloons (max_rate : ℕ) (max_time : ℕ) (zach_rate : ℕ) (zach_time : ℕ) (popped : ℕ) : ℕ :=
  max_rate * max_time + zach_rate * zach_time - popped

/-- Theorem stating the total number of filled water balloons Max and Zach have -/
theorem balloon_count : total_balloons 2 30 3 40 10 = 170 := by
  sorry

end balloon_count_l963_96375


namespace max_swaps_is_19_l963_96392

/-- A permutation of the numbers 1 to 20 -/
def Permutation := Fin 20 → Fin 20

/-- The identity permutation -/
def id_perm : Permutation := fun i => i

/-- A swap operation on a permutation -/
def swap (p : Permutation) (i j : Fin 20) : Permutation :=
  fun k => if k = i then p j else if k = j then p i else p k

/-- The minimum number of swaps needed to transform a permutation into the identity permutation -/
def min_swaps (p : Permutation) : ℕ := sorry

/-- Theorem: The maximum number of swaps needed for any permutation is 19 -/
theorem max_swaps_is_19 :
  ∃ (p : Permutation), min_swaps p = 19 ∧ 
  ∀ (q : Permutation), min_swaps q ≤ 19 :=
sorry

end max_swaps_is_19_l963_96392


namespace outfit_count_l963_96364

/-- The number of different outfits that can be made with the given clothing items. -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (belts : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (belts + 1)

/-- Theorem stating the number of outfits for the given clothing items. -/
theorem outfit_count :
  number_of_outfits 7 4 5 2 = 504 := by
  sorry

end outfit_count_l963_96364


namespace smallest_valid_number_last_four_digits_l963_96380

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ),
    n % 4 = 0 ∧
    n % 9 = 0 ∧
    n = 94444 + 90000 * k ∧
    k ≥ 0

theorem smallest_valid_number_last_four_digits :
  ∃ (n : ℕ),
    is_valid_number n ∧
    (∀ m, is_valid_number m → n ≤ m) ∧
    n % 10000 = 4444 :=
sorry

end smallest_valid_number_last_four_digits_l963_96380


namespace arctan_sum_special_case_l963_96317

theorem arctan_sum_special_case : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end arctan_sum_special_case_l963_96317


namespace inequality_proof_l963_96398

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_3 : a + b + c = 3) :
  (a^2 + 9) / (2*a^2 + (b + c)^2) + (b^2 + 9) / (2*b^2 + (c + a)^2) + (c^2 + 9) / (2*c^2 + (a + b)^2) ≤ 5 := by
  sorry

end inequality_proof_l963_96398


namespace sum_of_fractions_between_18_and_19_l963_96336

theorem sum_of_fractions_between_18_and_19 :
  let a : ℚ := 2 + 3/8
  let b : ℚ := 4 + 1/3
  let c : ℚ := 5 + 2/21
  let d : ℚ := 6 + 1/11
  18 < a + b + c + d ∧ a + b + c + d < 19 := by
  sorry

end sum_of_fractions_between_18_and_19_l963_96336


namespace johnsons_class_size_l963_96346

theorem johnsons_class_size (finley_class : ℕ) (johnson_class : ℕ) 
  (h1 : finley_class = 24) 
  (h2 : johnson_class = finley_class / 2 + 10) : 
  johnson_class = 22 := by
  sorry

end johnsons_class_size_l963_96346


namespace mrs_dunbar_roses_l963_96329

/-- Calculates the total number of white roses needed for a wedding arrangement -/
def total_roses (num_bouquets : ℕ) (num_table_decorations : ℕ) (roses_per_bouquet : ℕ) (roses_per_table_decoration : ℕ) : ℕ :=
  num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration

/-- Proves that the total number of white roses needed for Mrs. Dunbar's wedding arrangement is 109 -/
theorem mrs_dunbar_roses : total_roses 5 7 5 12 = 109 := by
  sorry

end mrs_dunbar_roses_l963_96329


namespace perpendicular_condition_l963_96399

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Constructs the first line l1 given parameter a -/
def line1 (a : ℝ) : Line :=
  { a := a, b := a + 2, c := 1 }

/-- Constructs the second line l2 given parameter a -/
def line2 (a : ℝ) : Line :=
  { a := 1, b := a, c := 2 }

/-- States that a = -3 is a sufficient but not necessary condition for perpendicularity -/
theorem perpendicular_condition :
  (∀ a : ℝ, a = -3 → are_perpendicular (line1 a) (line2 a)) ∧
  (∃ a : ℝ, a ≠ -3 ∧ are_perpendicular (line1 a) (line2 a)) := by
  sorry

end perpendicular_condition_l963_96399


namespace circle_symmetry_l963_96320

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 2016

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 2016

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ),
  original_circle x y ∧ symmetry_line x y →
  ∃ (x' y' : ℝ), symmetric_circle x' y' ∧ symmetry_line ((x + x') / 2) ((y + y') / 2) :=
by sorry

end circle_symmetry_l963_96320


namespace quadratic_inequality_range_l963_96365

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4 * x + a > 0) ↔ a > 2 := by sorry

end quadratic_inequality_range_l963_96365


namespace solve_refrigerator_problem_l963_96354

def refrigerator_problem (purchase_price installation_cost transport_cost selling_price : ℚ) : Prop :=
  let discount_rate : ℚ := 20 / 100
  let profit_rate : ℚ := 10 / 100
  let labelled_price : ℚ := purchase_price / (1 - discount_rate)
  let total_cost : ℚ := labelled_price + installation_cost + transport_cost
  (1 + profit_rate) * total_cost = selling_price

theorem solve_refrigerator_problem :
  refrigerator_problem 17500 250 125 24475 := by sorry

end solve_refrigerator_problem_l963_96354


namespace estimate_white_balls_l963_96316

/-- The number of red balls in the bag -/
def red_balls : ℕ := 6

/-- The probability of drawing a red ball -/
def prob_red : ℚ := 1/5

/-- The number of white balls in the bag -/
def white_balls : ℕ := 24

theorem estimate_white_balls :
  (red_balls : ℚ) / (red_balls + white_balls) = prob_red := by
  sorry

end estimate_white_balls_l963_96316


namespace odd_integer_quadratic_function_property_l963_96314

theorem odd_integer_quadratic_function_property (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  (Nat.gcd a n = 1) ∧ (Nat.gcd b n = 1) ∧
  (∃ (k : ℕ), n * k = (a^2 + b)) ∧
  (∀ (x : ℕ), x ≥ 1 → ∃ (p : ℕ), Prime p ∧ p ∣ ((x + a)^2 + b) ∧ ¬(p ∣ n)) := by
  sorry

end odd_integer_quadratic_function_property_l963_96314


namespace height_comparison_l963_96366

theorem height_comparison (a b : ℝ) (h : a = 0.8 * b) : b = 1.25 * a := by
  sorry

end height_comparison_l963_96366


namespace unique_triangle_configuration_l963_96330

/-- Represents a stick with a positive length -/
structure Stick where
  length : ℝ
  positive : length > 0

/-- Represents a triangle formed by three sticks -/
structure Triangle where
  a : Stick
  b : Stick
  c : Stick
  valid : a.length + b.length > c.length ∧
          a.length + c.length > b.length ∧
          b.length + c.length > a.length

/-- A configuration of 15 sticks forming 5 triangles -/
structure Configuration where
  sticks : Fin 15 → Stick
  triangles : Fin 5 → Triangle
  uses_all_sticks : ∀ s : Fin 15, ∃ t : Fin 5, (triangles t).a = sticks s ∨
                                               (triangles t).b = sticks s ∨
                                               (triangles t).c = sticks s

/-- Theorem stating that there's only one way to form 5 triangles from 15 sticks -/
theorem unique_triangle_configuration (c1 c2 : Configuration) : c1 = c2 := by
  sorry

end unique_triangle_configuration_l963_96330


namespace inequality_proof_l963_96384

theorem inequality_proof (a m : ℝ) (ha : a > 0) :
  abs (m + a) + abs (m + 1 / a) + abs (-1 / m + a) + abs (-1 / m + 1 / a) ≥ 4 := by
  sorry

end inequality_proof_l963_96384


namespace cube_sum_divided_l963_96393

theorem cube_sum_divided (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 219 / 9 := by
  sorry

end cube_sum_divided_l963_96393


namespace last_digit_of_multiple_of_six_l963_96331

theorem last_digit_of_multiple_of_six (x : ℕ) :
  x < 10 →
  (43560 + x) % 6 = 0 →
  x = 0 ∨ x = 6 := by
sorry

end last_digit_of_multiple_of_six_l963_96331


namespace stream_speed_l963_96378

/-- Given a canoe that rows upstream at 6 km/hr and downstream at 10 km/hr, 
    the speed of the stream is 2 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 6)
  (h2 : downstream_speed = 10) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 2 := by
  sorry


end stream_speed_l963_96378


namespace pizza_count_l963_96360

def num_toppings : ℕ := 8

def zero_topping_pizzas : ℕ := 1

def one_topping_pizzas (n : ℕ) : ℕ := n

def two_topping_pizzas (n : ℕ) : ℕ := n.choose 2

def total_pizzas (n : ℕ) : ℕ :=
  zero_topping_pizzas + one_topping_pizzas n + two_topping_pizzas n

theorem pizza_count : total_pizzas num_toppings = 37 := by
  sorry

end pizza_count_l963_96360


namespace x1_value_l963_96306

theorem x1_value (x₁ x₂ x₃ : ℝ) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h2 : (1 - x₁)^2 + 2*(x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/2) :
  x₁ = (3 * Real.sqrt 2 - 3) / 7 := by
  sorry

end x1_value_l963_96306


namespace average_age_decrease_l963_96315

/-- Given a group of 10 persons with an unknown average age A,
    prove that replacing a person aged 40 with a person aged 10
    decreases the average age by 3 years. -/
theorem average_age_decrease (A : ℝ) : 
  A - ((10 * A - 30) / 10) = 3 := by
  sorry

end average_age_decrease_l963_96315


namespace semicircle_perimeter_l963_96369

/-- The perimeter of a semicircle with radius 14 cm is equal to 14π + 28 cm. -/
theorem semicircle_perimeter :
  let r : ℝ := 14
  let π : ℝ := Real.pi
  let semicircle_perimeter : ℝ := r * π + 2 * r
  semicircle_perimeter = 14 * π + 28 := by
  sorry

end semicircle_perimeter_l963_96369


namespace line_passes_through_fixed_point_l963_96324

/-- The line y = mx + (2m+1) always passes through the point (-2, 1) for any real m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ((-2 : ℝ) * m + (2 * m + 1) = 1) := by
  sorry

end line_passes_through_fixed_point_l963_96324


namespace min_selling_price_l963_96328

/-- The minimum selling price for a product line given specific conditions --/
theorem min_selling_price (n : ℕ) (avg_price : ℝ) (low_price_count : ℕ) (max_price : ℝ) :
  n = 20 →
  avg_price = 1200 →
  low_price_count = 10 →
  max_price = 11000 →
  ∃ (min_price : ℝ),
    min_price = 400 ∧
    min_price * low_price_count + 1000 * (n - low_price_count - 1) + max_price = n * avg_price :=
by sorry

end min_selling_price_l963_96328


namespace projectile_max_height_l963_96358

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 116

/-- Theorem stating that the maximum height reached by the projectile is 116 feet -/
theorem projectile_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ max_height :=
sorry

end projectile_max_height_l963_96358


namespace trains_crossing_time_l963_96327

/-- The time it takes for two trains moving in opposite directions to cross each other -/
theorem trains_crossing_time (length_A length_B speed_A speed_B : ℝ) : 
  length_A = 108 →
  length_B = 112 →
  speed_A = 50 * (1000 / 3600) →
  speed_B = 82 * (1000 / 3600) →
  let total_length := length_A + length_B
  let relative_speed := speed_A + speed_B
  let crossing_time := total_length / relative_speed
  ∃ ε > 0, |crossing_time - 6| < ε :=
by
  sorry

#check trains_crossing_time

end trains_crossing_time_l963_96327


namespace find_m_l963_96338

def U : Set Nat := {0, 1, 2, 3}

def A (m : ℝ) : Set Nat := {x ∈ U | x^2 + m * x = 0}

def complement_A : Set Nat := {1, 2}

theorem find_m :
  ∃ m : ℝ, (A m = U \ complement_A) ∧ (m = -3) :=
sorry

end find_m_l963_96338


namespace complex_simplification_l963_96322

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that the simplification of the given complex expression equals 30 -/
theorem complex_simplification : 6 * (4 - 2 * i) + 2 * i * (6 - 3 * i) = 30 := by
  sorry

end complex_simplification_l963_96322


namespace integer_solutions_of_equation_l963_96374

theorem integer_solutions_of_equation :
  ∀ m n : ℤ, m^5 - n^5 = 16*m*n ↔ (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = -2) ∨ (m = -2 ∧ n = 2) :=
by sorry

end integer_solutions_of_equation_l963_96374


namespace washer_cost_l963_96350

/-- Given a washer-dryer combination costing 1200 dollars, where the washer costs 220 dollars more than the dryer, prove that the washer costs 710 dollars. -/
theorem washer_cost (total : ℝ) (difference : ℝ) (washer : ℝ) (dryer : ℝ) : 
  total = 1200 →
  difference = 220 →
  washer = dryer + difference →
  total = washer + dryer →
  washer = 710 := by
sorry

end washer_cost_l963_96350


namespace probability_three_tails_one_head_probability_three_tails_one_head_proof_l963_96301

/-- The probability of getting exactly three tails and one head when tossing four coins simultaneously -/
theorem probability_three_tails_one_head : ℚ :=
  1 / 4

/-- Proof that the probability of getting exactly three tails and one head when tossing four coins simultaneously is 1/4 -/
theorem probability_three_tails_one_head_proof :
  probability_three_tails_one_head = 1 / 4 := by
  sorry

end probability_three_tails_one_head_probability_three_tails_one_head_proof_l963_96301


namespace complement_union_theorem_l963_96318

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | |x| < 2}
def N : Set ℝ := {y | ∃ x, y = 2^x - 1}

-- State the theorem
theorem complement_union_theorem :
  (Set.compl M ∪ Set.compl N) = {x | x ≤ -1 ∨ x ≥ 2} := by sorry

end complement_union_theorem_l963_96318


namespace polynomial_factorization_l963_96377

theorem polynomial_factorization (a b c : ℝ) : 
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  (a - b) * (b - c) * (c - a) * (a*b^2 + b*c^2 + c*a^2) := by
  sorry

end polynomial_factorization_l963_96377


namespace jerry_sticker_count_jerry_has_36_stickers_l963_96357

theorem jerry_sticker_count (fred_stickers : ℕ) (george_diff : ℕ) (jerry_multiplier : ℕ) : ℕ :=
  let george_stickers := fred_stickers - george_diff
  let jerry_stickers := jerry_multiplier * george_stickers
  jerry_stickers

theorem jerry_has_36_stickers : 
  jerry_sticker_count 18 6 3 = 36 := by
  sorry

end jerry_sticker_count_jerry_has_36_stickers_l963_96357


namespace largest_equal_digit_sums_l963_96342

/-- Calculates the sum of digits of a natural number in a given base. -/
def digit_sum (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Checks if a number has equal digit sums in base 10 and base 3. -/
def equal_digit_sums (n : ℕ) : Prop :=
  digit_sum n 10 = digit_sum n 3

theorem largest_equal_digit_sums :
  ∀ m : ℕ, m < 1000 → m > 310 → ¬(equal_digit_sums m) ∧ equal_digit_sums 310 := by sorry

end largest_equal_digit_sums_l963_96342


namespace candy_bar_weight_reduction_l963_96387

theorem candy_bar_weight_reduction 
  (original_weight : ℝ) 
  (original_price : ℝ) 
  (new_weight : ℝ) 
  (h1 : original_weight > 0) 
  (h2 : original_price > 0) 
  (h3 : new_weight > 0) 
  (h4 : new_weight < original_weight) 
  (h5 : original_price / new_weight = (1 + 1/3) * (original_price / original_weight)) :
  (original_weight - new_weight) / original_weight = 1/4 := by
sorry

end candy_bar_weight_reduction_l963_96387


namespace exponent_subtraction_minus_fifteen_l963_96311

theorem exponent_subtraction_minus_fifteen :
  (23^11 / 23^8) - 15 = 12152 := by sorry

end exponent_subtraction_minus_fifteen_l963_96311


namespace min_value_of_ab_l963_96355

theorem min_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 8) :
  a * b ≥ 16 := by
  sorry

end min_value_of_ab_l963_96355


namespace dogs_running_l963_96337

theorem dogs_running (total : ℕ) (playing : ℕ) (barking : ℕ) (idle : ℕ) : 
  total = 88 →
  playing = total / 2 →
  barking = total / 4 →
  idle = 10 →
  total - playing - barking - idle = 12 :=
by
  sorry

end dogs_running_l963_96337


namespace hyperbola_eccentricity_l963_96396

/-- The eccentricity of a hyperbola passing through the focus of a specific parabola -/
theorem hyperbola_eccentricity (a : ℝ) (h_a : a > 0) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 = 1
  let parabola := fun (x y : ℝ) => y^2 = 8 * x
  let focus : ℝ × ℝ := (2, 0)
  hyperbola focus.1 focus.2 →
  let c := Real.sqrt (a^2 + 1)
  c / a = Real.sqrt 5 / 2 := by
sorry

end hyperbola_eccentricity_l963_96396


namespace set_T_is_hexagon_l963_96389

/-- The set T of points (x, y) satisfying the given conditions forms a hexagon -/
theorem set_T_is_hexagon (b : ℝ) (hb : b > 0) :
  let T : Set (ℝ × ℝ) :=
    {p | b ≤ p.1 ∧ p.1 ≤ 3*b ∧
         b ≤ p.2 ∧ p.2 ≤ 3*b ∧
         p.1 + p.2 ≥ 2*b ∧
         p.1 + 2*b ≥ 2*p.2 ∧
         p.2 + 2*b ≥ 2*p.1}
  ∃ (vertices : Finset (ℝ × ℝ)), vertices.card = 6 ∧
    ∀ p ∈ T, p ∈ convexHull ℝ (vertices : Set (ℝ × ℝ)) :=
by
  sorry

end set_T_is_hexagon_l963_96389


namespace sum_of_squares_is_integer_l963_96303

theorem sum_of_squares_is_integer 
  (a b c : ℚ) 
  (h1 : ∃ k : ℤ, (a + b + c : ℚ) = k)
  (h2 : ∃ m : ℤ, (a * b + b * c + c * a) / (a + b + c) = m) :
  ∃ n : ℤ, (a^2 + b^2 + c^2) / (a + b + c) = n := by
  sorry

end sum_of_squares_is_integer_l963_96303


namespace percentage_and_absolute_difference_l963_96343

/-- Given two initial values and an annual percentage increase, 
    calculate the percentage difference and the absolute difference after 5 years. -/
theorem percentage_and_absolute_difference 
  (initial_value1 : ℝ) 
  (initial_value2 : ℝ) 
  (annual_increase : ℝ) 
  (h1 : initial_value1 = 0.60 * 5000) 
  (h2 : initial_value2 = 0.42 * 3000) :
  let difference := initial_value1 - initial_value2
  let percentage_difference := (difference / initial_value1) * 100
  let new_difference := difference * (1 + annual_increase / 100) ^ 5
  percentage_difference = 58 ∧ 
  new_difference = 1740 * (1 + annual_increase / 100) ^ 5 := by
sorry

end percentage_and_absolute_difference_l963_96343


namespace cube_coloring_count_l963_96307

/-- Represents a coloring scheme for a cube -/
structure CubeColoring where
  /-- The number of faces on the cube -/
  faces : Nat
  /-- The number of available colors -/
  colors : Nat
  /-- The number of faces already colored -/
  colored_faces : Nat
  /-- Function to check if a coloring scheme is valid -/
  is_valid : (List Nat) → Bool

/-- Counts the number of valid coloring schemes for a cube -/
def count_valid_colorings (c : CubeColoring) : Nat :=
  sorry

/-- Theorem stating that there are exactly 13 valid coloring schemes for a cube
    with 6 faces, 5 colors, and 3 faces already colored -/
theorem cube_coloring_count :
  ∃ (c : CubeColoring),
    c.faces = 6 ∧
    c.colors = 5 ∧
    c.colored_faces = 3 ∧
    count_valid_colorings c = 13 :=
  sorry

end cube_coloring_count_l963_96307


namespace matrix_product_equality_l963_96372

theorem matrix_product_equality : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 3, -1; 1, 5, -2; 0, 6, 2]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![3, -1, 0; 2, 1, -4; 5, 0, 1]
  A * B = !![7, 1, -13; 3, 4, -22; 22, 6, -22] := by sorry

end matrix_product_equality_l963_96372


namespace smallest_sum_of_c_and_d_l963_96323

theorem smallest_sum_of_c_and_d (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h1 : ∃ x : ℝ, x^2 + c*x + 3*d = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*d*x + c = 0) :
  c + d ≥ (4*Real.sqrt 3 + 4/3) / Real.sqrt 3 := by
  sorry

end smallest_sum_of_c_and_d_l963_96323


namespace jasons_quarters_l963_96310

/-- Given an initial amount of quarters and an additional amount,
    calculate the total number of quarters. -/
def total_quarters (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Jason's total number of quarters -/
theorem jasons_quarters :
  total_quarters 49 25 = 74 := by
  sorry

end jasons_quarters_l963_96310


namespace sequence_inequality_l963_96309

theorem sequence_inequality (x : ℕ → ℝ) 
  (h1 : x 1 = 3)
  (h2 : ∀ n : ℕ, 4 * x (n + 1) - 3 * x n < 2)
  (h3 : ∀ n : ℕ, 2 * x (n + 1) - x n < 2) :
  ∀ n : ℕ, 2 + (1/2)^n < x (n + 1) ∧ x (n + 1) < 2 + (3/4)^n :=
by sorry

end sequence_inequality_l963_96309


namespace baseball_games_played_l963_96345

theorem baseball_games_played (wins losses played : ℕ) : 
  wins = 5 → 
  played = wins + losses → 
  played = 2 * losses → 
  played = 10 := by
sorry

end baseball_games_played_l963_96345


namespace stamp_price_l963_96376

theorem stamp_price (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 6 → purchase_price = (1 / 5) * original_price → original_price = 30 := by
  sorry

end stamp_price_l963_96376


namespace red_spools_count_l963_96305

-- Define the variables
def spools_per_beret : ℕ := 3
def black_spools : ℕ := 15
def blue_spools : ℕ := 6
def total_berets : ℕ := 11

-- Define the theorem
theorem red_spools_count : 
  ∃ (red_spools : ℕ), 
    red_spools + black_spools + blue_spools = spools_per_beret * total_berets ∧ 
    red_spools = 12 := by
  sorry

end red_spools_count_l963_96305


namespace average_of_a_and_b_l963_96348

theorem average_of_a_and_b (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 70)
  (h3 : c - a = 50) :
  (a + b) / 2 = 45 := by
sorry

end average_of_a_and_b_l963_96348


namespace sphere_radius_in_truncated_cone_l963_96304

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottomRadius : ℝ
  topRadius : ℝ
  sphereRadius : ℝ
  isTangent : Bool

/-- The theorem stating the radius of the sphere tangent to a specific truncated cone -/
theorem sphere_radius_in_truncated_cone
  (cone : TruncatedConeWithSphere)
  (h1 : cone.bottomRadius = 18)
  (h2 : cone.topRadius = 2)
  (h3 : cone.isTangent = true) :
  cone.sphereRadius = 6 := by
  sorry

end sphere_radius_in_truncated_cone_l963_96304


namespace specific_gathering_interactions_l963_96340

/-- The number of interactions in a gathering of witches and zombies -/
def interactions (num_witches num_zombies : ℕ) : ℕ :=
  (num_zombies * (num_zombies - 1)) / 2 + num_witches * num_zombies

/-- Theorem stating the number of interactions in a specific gathering -/
theorem specific_gathering_interactions :
  interactions 25 18 = 603 := by
  sorry

end specific_gathering_interactions_l963_96340


namespace bowling_ball_weighs_18_pounds_l963_96333

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := sorry

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := sorry

/-- Theorem stating the weight of one bowling ball is 18 pounds -/
theorem bowling_ball_weighs_18_pounds :
  (10 * bowling_ball_weight = 6 * canoe_weight) →
  (4 * canoe_weight = 120) →
  bowling_ball_weight = 18 := by sorry

end bowling_ball_weighs_18_pounds_l963_96333


namespace expression_evaluation_l963_96344

theorem expression_evaluation : 3 - (5 : ℝ)^(3-3) = 2 := by sorry

end expression_evaluation_l963_96344


namespace function_inequality_range_l963_96361

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem function_inequality_range 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f (2*x - 1) < f (1/3)} = Set.Ioo (1/3) (2/3) := by
  sorry

end function_inequality_range_l963_96361


namespace binary_digit_difference_l963_96379

theorem binary_digit_difference (n m : ℕ) (hn : n = 1280) (hm : m = 320) :
  (Nat.log 2 n + 1) - (Nat.log 2 m + 1) = 2 := by
  sorry

end binary_digit_difference_l963_96379


namespace strawberry_harvest_l963_96321

/-- Calculates the expected strawberry harvest from a rectangular garden. -/
theorem strawberry_harvest (length width plants_per_sqft berries_per_plant : ℕ) :
  length = 10 →
  width = 12 →
  plants_per_sqft = 5 →
  berries_per_plant = 8 →
  length * width * plants_per_sqft * berries_per_plant = 4800 := by
  sorry

#check strawberry_harvest

end strawberry_harvest_l963_96321


namespace contractor_engagement_l963_96352

/-- Represents the contractor's engagement problem -/
def ContractorProblem (daily_wage : ℚ) (daily_fine : ℚ) (total_earnings : ℚ) (absent_days : ℕ) : Prop :=
  ∃ (working_days : ℕ),
    daily_wage * working_days - daily_fine * absent_days = total_earnings ∧
    working_days + absent_days = 30

/-- The theorem states that given the problem conditions, the total engagement days is 30 -/
theorem contractor_engagement :
  ContractorProblem 25 7.5 425 10 :=
by
  sorry

end contractor_engagement_l963_96352


namespace min_value_a_plus_2b_l963_96319

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = x + y + 1 → a + 2 * b ≤ x + 2 * y ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ + 1 ∧ a₀ + 2 * b₀ = 7 :=
by sorry

end min_value_a_plus_2b_l963_96319


namespace professors_simultaneous_probability_l963_96353

/-- The duration the cafeteria is open, in minutes -/
def cafeteria_open_duration : ℕ := 120

/-- The duration of each professor's lunch, in minutes -/
def lunch_duration : ℕ := 15

/-- The latest possible start time for lunch, in minutes after the cafeteria opens -/
def latest_start_time : ℕ := cafeteria_open_duration - lunch_duration

/-- The probability that two professors are in the cafeteria simultaneously -/
theorem professors_simultaneous_probability : 
  (lunch_duration * latest_start_time : ℚ) / (latest_start_time^2 : ℚ) = 2/7 := by
  sorry

end professors_simultaneous_probability_l963_96353


namespace complex_power_one_minus_i_six_l963_96302

theorem complex_power_one_minus_i_six :
  (1 - Complex.I) ^ 6 = 8 * Complex.I :=
by sorry

end complex_power_one_minus_i_six_l963_96302


namespace polygon_14_diagonals_interior_angles_sum_l963_96386

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

theorem polygon_14_diagonals_interior_angles_sum :
  ∃ n : ℕ, num_diagonals n = 14 ∧ sum_interior_angles n = 900 :=
sorry

end polygon_14_diagonals_interior_angles_sum_l963_96386


namespace line_through_two_points_l963_96373

/-- Given a line passing through points (-3, 5) and (0, -4), prove that m + b = -7 
    where y = mx + b is the equation of the line. -/
theorem line_through_two_points (m b : ℝ) : 
  (5 = -3 * m + b) ∧ (-4 = 0 * m + b) → m + b = -7 := by
  sorry

end line_through_two_points_l963_96373
