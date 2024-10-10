import Mathlib

namespace polynomial_factorization_l1492_149285

theorem polynomial_factorization (a b c : ℝ) :
  a^3 * (b^2 - c^2) + b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2) =
  (a - b)^2 * (b - c) * (c - a) * (a*b + b*c + c*a) := by
  sorry

end polynomial_factorization_l1492_149285


namespace jenny_tim_age_difference_l1492_149202

/-- Represents the ages of family members --/
structure FamilyAges where
  tim : ℕ
  rommel : ℕ
  jenny : ℕ
  uncle : ℕ
  aunt : ℚ

/-- Defines the relationships between family members' ages --/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.tim = 5 ∧
  ages.rommel = 3 * ages.tim ∧
  ages.jenny = ages.rommel + 2 ∧
  ages.uncle = 2 * (ages.rommel + ages.jenny) ∧
  ages.aunt = (ages.uncle + ages.jenny) / 2

/-- Theorem stating the age difference between Jenny and Tim --/
theorem jenny_tim_age_difference (ages : FamilyAges) 
  (h : validFamilyAges ages) : ages.jenny - ages.tim = 12 := by
  sorry

end jenny_tim_age_difference_l1492_149202


namespace simplify_equation_l1492_149278

theorem simplify_equation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 7*x^2 + x + 1 = 0 ↔ x^2*(y^2 + y - 9) = 0 :=
by sorry

end simplify_equation_l1492_149278


namespace polynomial_value_at_one_l1492_149212

-- Define the polynomial P(x)
def P (r : ℝ) (x : ℝ) : ℝ := x^3 + x^2 - r^2*x - 2020

-- Define the roots of P(x)
variable (r s t : ℝ)

-- State the theorem
theorem polynomial_value_at_one (hr : P r r = 0) (hs : P r s = 0) (ht : P r t = 0) :
  P r 1 = -4038 := by
  sorry

end polynomial_value_at_one_l1492_149212


namespace translation_right_5_units_l1492_149214

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_5_units :
  let P : Point := { x := -2, y := -3 }
  let P' : Point := translateRight P 5
  P'.x = 3 ∧ P'.y = -3 := by
  sorry

end translation_right_5_units_l1492_149214


namespace regular_polygon_sides_l1492_149252

theorem regular_polygon_sides (central_angle : ℝ) (h : central_angle = 36) :
  (360 : ℝ) / central_angle = 10 := by
  sorry

end regular_polygon_sides_l1492_149252


namespace binary_string_power_of_two_sum_l1492_149269

/-- A binary string is represented as a list of booleans, where true represents 1 and false represents 0. -/
def BinaryString := List Bool

/-- Count the number of ones in a binary string. -/
def countOnes (s : BinaryString) : Nat :=
  s.filter id |>.length

/-- Represents a way of inserting plus signs into a binary string. 
    true means "insert a plus sign after this digit", false means "don't insert". -/
def PlusInsertion := List Bool

/-- Compute the sum of a binary string with plus signs inserted according to a PlusInsertion. -/
def computeSum (s : BinaryString) (insertion : PlusInsertion) : Nat :=
  sorry  -- Implementation details omitted for brevity

/-- Check if a number is a power of two. -/
def isPowerOfTwo (n : Nat) : Prop :=
  ∃ k : Nat, n = 2^k

/-- The main theorem statement. -/
theorem binary_string_power_of_two_sum 
  (s : BinaryString) 
  (h : countOnes s ≥ 2017) : 
  ∃ insertion : PlusInsertion, isPowerOfTwo (computeSum s insertion) := by
  sorry


end binary_string_power_of_two_sum_l1492_149269


namespace andy_late_time_l1492_149292

def school_start_time : Nat := 8 * 60  -- 8:00 AM in minutes
def normal_travel_time : Nat := 30
def red_light_delay : Nat := 3
def num_red_lights : Nat := 4
def construction_delay : Nat := 10
def departure_time : Nat := 7 * 60 + 15  -- 7:15 AM in minutes

def total_delay : Nat := red_light_delay * num_red_lights + construction_delay

def actual_travel_time : Nat := normal_travel_time + total_delay

def arrival_time : Nat := departure_time + actual_travel_time

theorem andy_late_time : arrival_time - school_start_time = 7 := by
  sorry

end andy_late_time_l1492_149292


namespace two_x_equals_two_l1492_149263

theorem two_x_equals_two (h : 1 = x) : 2 * x = 2 := by
  sorry

end two_x_equals_two_l1492_149263


namespace cindy_calculation_l1492_149257

theorem cindy_calculation (x : ℝ) : (2 * (x - 9)) / 6 = 36 → (x - 12) / 8 = 13.125 := by
  sorry

end cindy_calculation_l1492_149257


namespace school_students_count_l1492_149215

theorem school_students_count (blue_percent : ℚ) (red_percent : ℚ) (green_percent : ℚ) (other_count : ℕ) :
  blue_percent = 44/100 →
  red_percent = 28/100 →
  green_percent = 10/100 →
  other_count = 162 →
  ∃ (total : ℕ), 
    (blue_percent + red_percent + green_percent < 1) ∧
    (1 - (blue_percent + red_percent + green_percent)) * total = other_count ∧
    total = 900 :=
by
  sorry

end school_students_count_l1492_149215


namespace lukes_fishing_days_l1492_149286

/-- Proves the number of days Luke catches fish given the conditions -/
theorem lukes_fishing_days (fish_per_day : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ) : 
  fish_per_day = 2 → 
  fillets_per_fish = 2 → 
  total_fillets = 120 → 
  (total_fillets / fillets_per_fish) / fish_per_day = 30 := by
  sorry

end lukes_fishing_days_l1492_149286


namespace power_sum_equality_l1492_149259

theorem power_sum_equality : (3^2)^3 + (2^3)^2 = 793 := by
  sorry

end power_sum_equality_l1492_149259


namespace arithmetic_geometric_sequence_l1492_149216

theorem arithmetic_geometric_sequence (d : ℝ) (a : ℕ → ℝ) :
  d ≠ 0 ∧
  (∀ n, a (n + 1) = a n + d) ∧
  a 1 = 1 ∧
  (a 3) ^ 2 = a 1 * a 13 →
  d = 2 := by
sorry

end arithmetic_geometric_sequence_l1492_149216


namespace sqrt_product_plus_one_l1492_149243

theorem sqrt_product_plus_one : 
  Real.sqrt ((35 : ℝ) * 34 * 33 * 32 + 1) = 1121 := by sorry

end sqrt_product_plus_one_l1492_149243


namespace inequality_proof_l1492_149265

theorem inequality_proof (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 3) : 
  a^b + b^c + c^a ≤ a^2 + b^2 + c^2 := by
  sorry

end inequality_proof_l1492_149265


namespace min_sum_distances_l1492_149283

variable {α : Type*} [LinearOrder α] [AddCommGroup α] [OrderedAddCommGroup α]

def points (P₁ P₂ P₃ P₄ P₅ P₆ P₇ : α) : Prop :=
  P₁ < P₂ ∧ P₂ < P₃ ∧ P₃ < P₄ ∧ P₄ < P₅ ∧ P₅ < P₆ ∧ P₆ < P₇

def distance (x y : α) : α := abs (x - y)

def sum_distances (P : α) (P₁ P₂ P₃ P₄ P₅ P₆ P₇ : α) : α :=
  distance P P₁ + distance P P₂ + distance P P₃ + distance P P₄ +
  distance P P₅ + distance P P₆ + distance P P₇

theorem min_sum_distances
  (P₁ P₂ P₃ P₄ P₅ P₆ P₇ : α)
  (h : points P₁ P₂ P₃ P₄ P₅ P₆ P₇) :
  ∀ P, sum_distances P P₁ P₂ P₃ P₄ P₅ P₆ P₇ ≥ sum_distances P₄ P₁ P₂ P₃ P₄ P₅ P₆ P₇ ∧
  (sum_distances P P₁ P₂ P₃ P₄ P₅ P₆ P₇ = sum_distances P₄ P₁ P₂ P₃ P₄ P₅ P₆ P₇ ↔ P = P₄) :=
by sorry

end min_sum_distances_l1492_149283


namespace women_to_total_ratio_l1492_149280

theorem women_to_total_ratio (total_passengers : ℕ) (seated_men : ℕ) : 
  total_passengers = 48 →
  seated_men = 14 →
  ∃ (women men standing_men : ℕ),
    women + men = total_passengers ∧
    standing_men + seated_men = men ∧
    standing_men = men / 8 ∧
    women * 3 = total_passengers * 2 := by
  sorry

end women_to_total_ratio_l1492_149280


namespace sum_odd_9_to_39_l1492_149260

/-- Sum of first n consecutive odd integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n ^ 2

/-- The nth odd integer -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

/-- Sum of odd integers from a to b inclusive -/
def sum_odd_range (a b : ℕ) : ℕ :=
  sum_first_n_odd ((b - 1) / 2 + 1) - sum_first_n_odd ((a - 1) / 2)

theorem sum_odd_9_to_39 :
  sum_odd_range 9 39 = 384 :=
sorry

end sum_odd_9_to_39_l1492_149260


namespace friday_night_revenue_l1492_149240

/-- Represents the revenue calculation for a movie theater --/
def theater_revenue (matinee_price evening_price opening_price popcorn_price : ℕ)
                    (matinee_customers evening_customers opening_customers : ℕ) : ℕ :=
  let total_customers := matinee_customers + evening_customers + opening_customers
  let popcorn_sales := total_customers / 2
  (matinee_price * matinee_customers) +
  (evening_price * evening_customers) +
  (opening_price * opening_customers) +
  (popcorn_price * popcorn_sales)

/-- Theorem stating the total revenue of the theater on Friday night --/
theorem friday_night_revenue :
  theater_revenue 5 7 10 10 32 40 58 = 1670 := by
  sorry

end friday_night_revenue_l1492_149240


namespace nested_series_sum_l1492_149277

def nested_series (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | k + 1 => 2 * (1 + nested_series k)

theorem nested_series_sum : nested_series 5 = 126 := by
  sorry

end nested_series_sum_l1492_149277


namespace largest_c_for_f_range_containing_2_l1492_149207

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

-- State the theorem
theorem largest_c_for_f_range_containing_2 :
  (∃ (c : ℝ), ∀ (d : ℝ), 
    (∃ (x : ℝ), f d x = 2) → d ≤ c) ∧
  (∃ (x : ℝ), f 11 x = 2) :=
sorry

end largest_c_for_f_range_containing_2_l1492_149207


namespace product_divisible_by_five_l1492_149226

theorem product_divisible_by_five (a b : ℕ+) :
  (5 ∣ (a * b)) → ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) := by
  sorry

end product_divisible_by_five_l1492_149226


namespace quadratic_inequality_equivalence_l1492_149234

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < 9 ↔ x ∈ Set.Ioo (-5/2) 3 := by
sorry

end quadratic_inequality_equivalence_l1492_149234


namespace room_area_l1492_149294

/-- The area of a room given the costs of floor replacement -/
theorem room_area (removal_cost : ℝ) (per_sqft_cost : ℝ) (total_cost : ℝ) : 
  removal_cost = 50 →
  per_sqft_cost = 1.25 →
  total_cost = 120 →
  (total_cost - removal_cost) / per_sqft_cost = 56 := by
sorry

end room_area_l1492_149294


namespace equation_always_has_real_root_l1492_149273

theorem equation_always_has_real_root (K : ℝ) : 
  ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) :=
by
  sorry

end equation_always_has_real_root_l1492_149273


namespace cube_number_sum_l1492_149267

theorem cube_number_sum : 
  ∀ (n : ℤ),
  (∀ (i : Fin 6), i.val < 6 → ∃ (face : ℤ), face = n + i.val) →
  (∃ (s : ℤ), s % 2 = 1 ∧ 
    (n + (n + 5) = s) ∧ 
    ((n + 1) + (n + 4) = s) ∧ 
    ((n + 2) + (n + 3) = s)) →
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 27) :=
by sorry


end cube_number_sum_l1492_149267


namespace curve_equation_and_no_fixed_point_l1492_149261

-- Define the circle C2
def C2 (x y : ℝ) : Prop := x^2 + (y-2)^2 = 1

-- Define the curve C1
def C1 (x y : ℝ) : Prop :=
  ∀ (x' y' : ℝ), C2 x' y' → (x - x')^2 + (y - y')^2 > 0 ∧
  (y + 1 = Real.sqrt ((x - x')^2 + (y - y')^2) - 1)

-- Define the point N
def N (b : ℝ) : ℝ × ℝ := (0, b)

-- Define the angle equality condition
def angle_equality (P Q : ℝ × ℝ) (N : ℝ × ℝ) : Prop :=
  (P.1 - N.1)^2 + (P.2 - N.2)^2 = (Q.1 - N.1)^2 + (Q.2 - N.2)^2

theorem curve_equation_and_no_fixed_point :
  (∀ x y : ℝ, C1 x y ↔ x^2 = 8*y) ∧
  (∀ b : ℝ, b < 0 →
    ∀ P Q : ℝ × ℝ, C1 P.1 P.2 → C1 Q.1 Q.2 → P ≠ Q →
    angle_equality P Q (N b) →
    ¬∃ F : ℝ × ℝ, ∀ P Q : ℝ × ℝ, C1 P.1 P.2 → C1 Q.1 Q.2 → P ≠ Q →
      angle_equality P Q (N b) → (Q.2 - P.2) * F.1 = (Q.1 - P.1) * F.2 + (P.1 * Q.2 - Q.1 * P.2)) :=
sorry

end curve_equation_and_no_fixed_point_l1492_149261


namespace cubic_sum_equals_one_l1492_149239

theorem cubic_sum_equals_one (a b : ℝ) (h : a + b = 1) : a^3 + 3*a*b + b^3 = 1 := by
  sorry

end cubic_sum_equals_one_l1492_149239


namespace worker_room_arrangement_l1492_149244

/-- The number of rooms and workers -/
def n : ℕ := 5

/-- The number of unchosen rooms -/
def k : ℕ := 2

/-- Represents whether each room choice is equally likely -/
def equal_probability : Prop := sorry

/-- Represents the condition that unchosen rooms are not adjacent -/
def non_adjacent_unchosen : Prop := sorry

/-- The number of ways to arrange workers in rooms with given conditions -/
def arrangement_count : ℕ := sorry

theorem worker_room_arrangement :
  arrangement_count = 900 :=
sorry

end worker_room_arrangement_l1492_149244


namespace quadratic_minimum_value_l1492_149229

theorem quadratic_minimum_value :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + 5
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 1 :=
by sorry

end quadratic_minimum_value_l1492_149229


namespace ab_equals_zero_l1492_149264

theorem ab_equals_zero (a b : ℝ) 
  (h1 : (4 : ℝ) ^ a = 256 ^ (b + 1))
  (h2 : (27 : ℝ) ^ b = 3 ^ (a - 2)) : 
  a * b = 0 := by sorry

end ab_equals_zero_l1492_149264


namespace special_numbers_theorem_l1492_149242

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  90 ≤ n ∧ n ≤ 150 ∧ digit_sum (digit_sum n) = 1

theorem special_numbers_theorem : 
  {n : ℕ | satisfies_condition n} = {91, 100, 109, 118, 127, 136, 145} := by
  sorry

end special_numbers_theorem_l1492_149242


namespace square_sum_theorem_l1492_149236

theorem square_sum_theorem (x y : ℝ) 
  (h1 : (x + y)^4 + (x - y)^4 = 4112)
  (h2 : x^2 - y^2 = 16) : 
  x^2 + y^2 = 34 := by
sorry

end square_sum_theorem_l1492_149236


namespace basketball_substitutions_remainder_l1492_149210

/-- Number of ways to make substitutions in a basketball game -/
def substitution_ways (total_players starters max_substitutions : ℕ) : ℕ :=
  let substitutes := total_players - starters
  let a0 := 1  -- No substitutions
  let a1 := starters * substitutes  -- One substitution
  let a2 := a1 * (starters - 1) * (substitutes - 1)  -- Two substitutions
  let a3 := a2 * (starters - 2) * (substitutes - 2)  -- Three substitutions
  let a4 := a3 * (starters - 3) * (substitutes - 3)  -- Four substitutions
  a0 + a1 + a2 + a3 + a4

/-- Theorem stating the remainder when the number of substitution ways is divided by 1000 -/
theorem basketball_substitutions_remainder :
  substitution_ways 14 5 4 % 1000 = 606 := by
  sorry

end basketball_substitutions_remainder_l1492_149210


namespace square_side_length_of_unit_area_l1492_149245

/-- The side length of a square with area 1 is 1. -/
theorem square_side_length_of_unit_area : 
  ∀ s : ℝ, s > 0 → s * s = 1 → s = 1 := by
  sorry

end square_side_length_of_unit_area_l1492_149245


namespace complex_equation_sum_l1492_149251

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 →
  (a + 2 * i) / i = b - i →
  a + b = 3 := by sorry

end complex_equation_sum_l1492_149251


namespace simplify_expression_l1492_149256

theorem simplify_expression : (9 * 10^8) / (3 * 10^3) = 300000 := by
  sorry

end simplify_expression_l1492_149256


namespace slope_parallel_sufficient_not_necessary_l1492_149205

-- Define a structure for a line with a slope
structure Line where
  slope : ℝ

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem slope_parallel_sufficient_not_necessary :
  ∃ (l1 l2 : Line),
    (parallel l1 l2 → l1.slope = l2.slope) ∧
    ∃ (l3 l4 : Line), l3.slope = l4.slope ∧ ¬ parallel l3 l4 := by
  sorry

end slope_parallel_sufficient_not_necessary_l1492_149205


namespace triangle_problem_l1492_149220

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) →
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  A + B + C = π →
  2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c →
  c = Real.sqrt 7 →
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →
  C = π/3 ∧ a + b = 5 := by
sorry


end triangle_problem_l1492_149220


namespace complex_number_equality_l1492_149224

theorem complex_number_equality : ∀ (i : ℂ), i * i = -1 →
  (2 * i) / (1 + i) = 1 + i := by sorry

end complex_number_equality_l1492_149224


namespace count_numbers_3000_l1492_149289

/-- Returns true if the given number contains the digit '2' in its base-10 representation -/
def contains_two (n : ℕ) : Bool :=
  sorry

/-- Returns the count of numbers less than or equal to n that contain '2' and are divisible by 3 -/
def count_numbers (n : ℕ) : ℕ :=
  sorry

theorem count_numbers_3000 : count_numbers 3000 = 384 :=
  sorry

end count_numbers_3000_l1492_149289


namespace efficient_elimination_of_y_l1492_149287

theorem efficient_elimination_of_y (x y : ℝ) :
  (3 * x - 2 * y = 3) →
  (4 * x + y = 15) →
  ∃ k : ℝ, (2 * (4 * x + y) + (3 * x - 2 * y) = k) ∧ (11 * x = 33) :=
by
  sorry

end efficient_elimination_of_y_l1492_149287


namespace todd_ate_eight_cupcakes_l1492_149228

/-- The number of cupcakes Todd ate -/
def cupcakes_eaten (initial : ℕ) (packages : ℕ) (per_package : ℕ) : ℕ :=
  initial - (packages * per_package)

/-- Proof that Todd ate 8 cupcakes -/
theorem todd_ate_eight_cupcakes :
  cupcakes_eaten 18 5 2 = 8 := by
  sorry

end todd_ate_eight_cupcakes_l1492_149228


namespace dans_balloons_l1492_149217

theorem dans_balloons (dans_balloons : ℕ) (tims_balloons : ℕ) : 
  tims_balloons = 203 → 
  tims_balloons = 7 * dans_balloons → 
  dans_balloons = 29 := by
sorry

end dans_balloons_l1492_149217


namespace f_sum_theorem_l1492_149297

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_sum_theorem (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 2)
  (h_def : ∀ x ∈ Set.Ioo 0 1, f x = (4:ℝ) ^ x) :
  f (-5/2) + f 1 = -2 := by
sorry

end f_sum_theorem_l1492_149297


namespace marathon_total_distance_l1492_149201

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a total distance in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ
  h : yards < 1760

def marathon_length : Marathon := { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 10

theorem marathon_total_distance :
  ∃ (m : ℕ) (y : ℕ) (h : y < 1760),
    (m * yards_per_mile + y) = 
      (num_marathons * marathon_length.miles * yards_per_mile + 
       num_marathons * marathon_length.yards) ∧
    y = 330 := by sorry

end marathon_total_distance_l1492_149201


namespace rectangular_box_volume_l1492_149258

/-- The volume of a rectangular box given the areas of its faces -/
theorem rectangular_box_volume (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (area1 : a * b = 36) (area2 : b * c = 12) (area3 : a * c = 9) : 
  a * b * c = 144 * Real.sqrt 3 := by
  sorry

end rectangular_box_volume_l1492_149258


namespace genevieve_error_count_l1492_149204

/-- The number of lines of code Genevieve has written -/
def total_lines : ℕ := 4300

/-- The number of lines per debug block -/
def lines_per_block : ℕ := 100

/-- The number of errors found in the first block -/
def initial_errors : ℕ := 3

/-- The increase in errors found per block -/
def error_increase : ℕ := 1

/-- The number of completed debug blocks -/
def num_blocks : ℕ := total_lines / lines_per_block

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a₁ : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The total number of errors fixed -/
def total_errors : ℕ := arithmetic_sum num_blocks initial_errors error_increase

theorem genevieve_error_count :
  total_errors = 1032 := by sorry

end genevieve_error_count_l1492_149204


namespace inscribed_squares_problem_l1492_149203

theorem inscribed_squares_problem (a b : ℝ) : 
  let small_area : ℝ := 16
  let large_area : ℝ := 18
  let rotation_angle : ℝ := 30 * π / 180
  let small_side : ℝ := Real.sqrt small_area
  let large_side : ℝ := Real.sqrt large_area
  a + b = large_side ∧ 
  Real.sqrt (a^2 + b^2) = 2 * small_side * Real.cos rotation_angle →
  a * b = -15 := by
sorry

end inscribed_squares_problem_l1492_149203


namespace friends_weekly_biking_distance_l1492_149222

/-- The total distance two friends bike in a week -/
def total_distance_biked (onur_daily_distance : ℕ) (hanil_extra_distance : ℕ) (days_per_week : ℕ) : ℕ :=
  (onur_daily_distance * days_per_week) + ((onur_daily_distance + hanil_extra_distance) * days_per_week)

/-- Theorem stating the total distance biked by Onur and Hanil in a week -/
theorem friends_weekly_biking_distance :
  total_distance_biked 250 40 5 = 2700 := by
  sorry

#eval total_distance_biked 250 40 5

end friends_weekly_biking_distance_l1492_149222


namespace hyperbola_param_sum_l1492_149266

/-- A hyperbola with given center, focus, and vertex -/
structure Hyperbola where
  center : ℝ × ℝ
  focus : ℝ × ℝ
  vertex : ℝ × ℝ

/-- Parameters of the hyperbola equation -/
structure HyperbolaParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given a hyperbola, compute its equation parameters -/
def computeParams (hyp : Hyperbola) : HyperbolaParams := sorry

theorem hyperbola_param_sum :
  let hyp : Hyperbola := {
    center := (1, -1),
    focus := (1, 5),
    vertex := (1, 1)
  }
  let params := computeParams hyp
  params.h + params.k + params.a + params.b = 2 + 4 * Real.sqrt 2 := by sorry

end hyperbola_param_sum_l1492_149266


namespace prime_quadratic_roots_l1492_149271

theorem prime_quadratic_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x ≠ y ∧ 
    x^2 - 2*p*x + p^2 - 5*p - 1 = 0 ∧ 
    y^2 - 2*p*y + p^2 - 5*p - 1 = 0) → 
  p = 3 ∨ p = 7 := by
sorry

end prime_quadratic_roots_l1492_149271


namespace car_average_speed_l1492_149253

theorem car_average_speed 
  (total_time : ℝ) 
  (initial_time : ℝ) 
  (initial_speed : ℝ) 
  (remaining_speed : ℝ) 
  (h1 : total_time = 24) 
  (h2 : initial_time = 4) 
  (h3 : initial_speed = 35) 
  (h4 : remaining_speed = 53) : 
  (initial_speed * initial_time + remaining_speed * (total_time - initial_time)) / total_time = 50 := by
  sorry

end car_average_speed_l1492_149253


namespace desktop_computers_sold_l1492_149221

theorem desktop_computers_sold (total : ℕ) (laptops : ℕ) (netbooks : ℕ) (desktops : ℕ)
  (h1 : total = 72)
  (h2 : laptops = total / 2)
  (h3 : netbooks = total / 3)
  (h4 : desktops = total - laptops - netbooks) :
  desktops = 12 := by
  sorry

end desktop_computers_sold_l1492_149221


namespace nap_period_days_l1492_149291

-- Define the given conditions
def naps_per_week : ℕ := 3
def hours_per_nap : ℕ := 2
def total_nap_hours : ℕ := 60

-- Define the theorem
theorem nap_period_days : 
  (total_nap_hours / hours_per_nap / naps_per_week) * 7 = 70 := by
  sorry

end nap_period_days_l1492_149291


namespace school_absence_percentage_l1492_149254

theorem school_absence_percentage (total_students boys girls : ℕ) 
  (h_total : total_students = 180)
  (h_boys : boys = 100)
  (h_girls : girls = 80)
  (h_sum : total_students = boys + girls)
  (absent_boys : ℕ := boys / 5)
  (absent_girls : ℕ := girls / 4)
  (total_absent : ℕ := absent_boys + absent_girls) :
  (total_absent : ℚ) / total_students * 100 = 40 / 180 * 100 := by
sorry

end school_absence_percentage_l1492_149254


namespace arithmetic_calculations_l1492_149270

theorem arithmetic_calculations :
  (1405 - (816 + 487) = 102) ∧
  (3450 - 107 * 13 = 2059) ∧
  (48306 / (311 - 145) = 291) := by
sorry

end arithmetic_calculations_l1492_149270


namespace intersection_A_B_complement_B_in_A_union_A_l1492_149211

-- Define set A as positive integers less than 9
def A : Set ℕ := {x | x > 0 ∧ x < 9}

-- Define set B
def B : Set ℕ := {1, 2, 3}

-- Define set A for the second part
def A' : Set ℝ := {x | -3 < x ∧ x < 1}

-- Define set B for the second part
def B' : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {1, 2, 3} := by sorry

-- Theorem for complement of B in A
theorem complement_B_in_A : A \ B = {4, 5, 6, 7, 8} := by sorry

-- Theorem for A' ∪ B'
theorem union_A'_B' : A' ∪ B' = {x | -3 < x ∧ x < 1 ∨ 2 < x ∧ x < 10} := by sorry

end intersection_A_B_complement_B_in_A_union_A_l1492_149211


namespace sufficient_condition_for_square_inequality_l1492_149296

theorem sufficient_condition_for_square_inequality (a b : ℝ) :
  a > b ∧ b > 0 → a^2 > b^2 := by
  sorry

end sufficient_condition_for_square_inequality_l1492_149296


namespace sum_greater_than_2e_squared_l1492_149206

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem sum_greater_than_2e_squared (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) 
  (hf₁ : f a x₁ = 1) (hf₂ : f a x₂ = 1) : 
  x₁ + x₂ > 2 * (Real.exp 1) ^ 2 := by
  sorry

end sum_greater_than_2e_squared_l1492_149206


namespace real_part_of_reciprocal_l1492_149290

theorem real_part_of_reciprocal (z : ℂ) (h1 : z ≠ 1) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 1) :
  (1 / (1 - z)).re = 1 / 2 := by
  sorry

end real_part_of_reciprocal_l1492_149290


namespace cos_300_degrees_l1492_149295

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_degrees_l1492_149295


namespace regular_polygon_with_144_degree_angle_l1492_149272

theorem regular_polygon_with_144_degree_angle (n : ℕ) :
  n > 2 →
  (n - 2) * 180 = 144 * n →
  n = 10 :=
by sorry

end regular_polygon_with_144_degree_angle_l1492_149272


namespace cube_inequality_l1492_149237

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_inequality_l1492_149237


namespace circle_points_speeds_l1492_149200

/-- Two points moving along a unit circle -/
structure CirclePoints where
  v₁ : ℝ  -- Speed of the first point
  v₂ : ℝ  -- Speed of the second point

/-- Conditions for the circle points -/
def satisfies_conditions (cp : CirclePoints) : Prop :=
  cp.v₁ > 0 ∧ cp.v₂ > 0 ∧  -- Positive speeds
  cp.v₁ - cp.v₂ = 1 / 720 ∧  -- Meet every 12 minutes (720 seconds)
  1 / cp.v₂ - 1 / cp.v₁ = 10  -- First point is 10 seconds faster

/-- The theorem to be proved -/
theorem circle_points_speeds (cp : CirclePoints) 
  (h : satisfies_conditions cp) : cp.v₁ = 1/80 ∧ cp.v₂ = 1/90 := by
  sorry

end circle_points_speeds_l1492_149200


namespace product_equals_zero_l1492_149246

theorem product_equals_zero (a : ℤ) (h : a = 11) :
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 := by
  sorry

end product_equals_zero_l1492_149246


namespace no_double_composition_square_minus_two_l1492_149227

theorem no_double_composition_square_minus_two :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x^2 - 2 := by
  sorry

end no_double_composition_square_minus_two_l1492_149227


namespace empty_solution_set_implies_a_leq_5_l1492_149282

-- Define the function representing the sum of distances
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem statement
theorem empty_solution_set_implies_a_leq_5 :
  ∀ a : ℝ, (∀ x : ℝ, f x ≥ a) → a ≤ 5 := by
  sorry

end empty_solution_set_implies_a_leq_5_l1492_149282


namespace expression_simplification_l1492_149274

theorem expression_simplification (a : ℝ) (h : a^2 - a - (7/2) = 0) :
  a^2 - (a - (2*a)/(a+1)) / ((a^2 - 2*a + 1)/(a^2 - 1)) = 7/2 := by
  sorry

end expression_simplification_l1492_149274


namespace intersection_when_m_neg_three_subset_condition_l1492_149230

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Theorem 1: A ∩ B when m = -3
theorem intersection_when_m_neg_three :
  A ∩ B (-3) = {x | -3 ≤ x ∧ x ≤ -2} := by sorry

-- Theorem 2: B ⊆ A iff m ≥ -1
theorem subset_condition :
  ∀ m : ℝ, B m ⊆ A ↔ m ≥ -1 := by sorry

end intersection_when_m_neg_three_subset_condition_l1492_149230


namespace projectile_max_height_l1492_149247

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 10

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 41.25

/-- Theorem stating that the maximum value of h(t) is equal to max_height -/
theorem projectile_max_height : 
  ∀ t : ℝ, h t ≤ max_height ∧ ∃ t₀ : ℝ, h t₀ = max_height :=
sorry

end projectile_max_height_l1492_149247


namespace external_tangent_points_theorem_l1492_149284

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def intersect (c1 c2 : Circle) : Prop := sorry

def touches (c1 c2 : Circle) (p : Point) : Prop := sorry

def on_line (p : Point) (l : Line) : Prop := sorry

def passes_through (l : Line) (p : Point) : Prop := sorry

-- Main theorem
theorem external_tangent_points_theorem 
  (C C' : Circle) (X Y : Point) 
  (h1 : intersect C C') 
  (h2 : on_line X (Line.mk 0 1 0)) 
  (h3 : on_line Y (Line.mk 0 1 0)) :
  ∃ (T1 T2 T3 T4 : Point),
    ∀ (P Q R S : Point) (third_circle : Circle),
      touches C third_circle P →
      touches C' third_circle Q →
      on_line R (Line.mk 0 1 0) →
      on_line S (Line.mk 0 1 0) →
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) ∧
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) ∧
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) ∧
      (passes_through (Line.mk 1 0 0) T1 ∨
       passes_through (Line.mk 1 0 0) T2 ∨
       passes_through (Line.mk 1 0 0) T3 ∨
       passes_through (Line.mk 1 0 0) T4) := by
  sorry

end external_tangent_points_theorem_l1492_149284


namespace not_solution_one_l1492_149218

theorem not_solution_one (x : ℂ) (h1 : x^2 + x + 1 = 0) (h2 : x ≠ 0) : x ≠ 1 := by
  sorry

end not_solution_one_l1492_149218


namespace min_sum_of_sides_l1492_149279

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (a + b)^2 - c^2 = 4 and C = 60°, then the minimum value of a + b is 4√3/3 -/
theorem min_sum_of_sides (a b c : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : Real.cos (Real.pi / 3) = (a^2 + b^2 - c^2) / (2 * a * b)) :
  ∃ (min_sum : ℝ), min_sum = 4 * Real.sqrt 3 / 3 ∧ ∀ x y, (x + y)^2 - c^2 = 4 → x + y ≥ min_sum :=
sorry


end min_sum_of_sides_l1492_149279


namespace vector_subtraction_scalar_multiplication_l1492_149219

def vector_operation (v1 v2 : ℝ × ℝ) (s : ℝ) : ℝ × ℝ :=
  (v1.1 - s * v2.1, v1.2 - s * v2.2)

theorem vector_subtraction_scalar_multiplication :
  vector_operation (3, -8) (2, -6) 5 = (-7, 22) := by
  sorry

end vector_subtraction_scalar_multiplication_l1492_149219


namespace cubic_fraction_factorization_l1492_149235

theorem cubic_fraction_factorization (x y z : ℝ) :
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) = (x + y) * (y + z) * (z + x) :=
by sorry

end cubic_fraction_factorization_l1492_149235


namespace cubic_inches_in_cubic_foot_l1492_149268

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Theorem statement
theorem cubic_inches_in_cubic_foot :
  (inches_per_foot ^ 3 : ℕ) = 1728 :=
sorry

end cubic_inches_in_cubic_foot_l1492_149268


namespace min_sum_squares_l1492_149288

theorem min_sum_squares (a b : ℝ) (ha : a ≠ 0) :
  (∃ x ∈ Set.Icc 3 4, (a + 2) / x = a * x + 2 * b + 1) →
  ∃ m : ℝ, m = 1 / 100 ∧ ∀ a' b' : ℝ, 
    (∃ x ∈ Set.Icc 3 4, (a' + 2) / x = a' * x + 2 * b' + 1) → 
    a' ^ 2 + b' ^ 2 ≥ m :=
by sorry

end min_sum_squares_l1492_149288


namespace data_analytics_course_hours_l1492_149241

/-- Calculates the total hours spent on a course given the course duration and weekly time commitments. -/
def total_course_hours (weeks : ℕ) (class_hours_1 : ℕ) (class_hours_2 : ℕ) (class_hours_3 : ℕ) (homework_hours : ℕ) : ℕ :=
  weeks * (class_hours_1 + class_hours_2 + class_hours_3 + homework_hours)

/-- Theorem stating that the total hours spent on the described course is 336. -/
theorem data_analytics_course_hours : 
  total_course_hours 24 3 3 4 4 = 336 := by
  sorry

end data_analytics_course_hours_l1492_149241


namespace scaled_cylinder_volume_l1492_149232

/-- Theorem: Scaling a cylindrical container -/
theorem scaled_cylinder_volume (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 3 →
  π * (2*r)^2 * (4*h) = 48 := by
  sorry

end scaled_cylinder_volume_l1492_149232


namespace power_division_seventeen_l1492_149208

theorem power_division_seventeen : (17 : ℕ)^9 / (17 : ℕ)^7 = 289 := by sorry

end power_division_seventeen_l1492_149208


namespace alice_oranges_sold_l1492_149213

/-- Given that Alice sold twice as many oranges as Emily, and they sold 180 oranges in total,
    prove that Alice sold 120 oranges. -/
theorem alice_oranges_sold (emily : ℕ) (h1 : emily + 2 * emily = 180) : 2 * emily = 120 := by
  sorry

end alice_oranges_sold_l1492_149213


namespace negation_of_existential_negation_of_specific_existential_l1492_149293

theorem negation_of_existential (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_specific_existential :
  (¬ ∃ x : ℝ, x^2 - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2 > 0) :=
by sorry

end negation_of_existential_negation_of_specific_existential_l1492_149293


namespace polynomial_divisibility_l1492_149275

theorem polynomial_divisibility (x : ℝ) (m : ℝ) : 
  (5 * x^3 - 3 * x^2 - 12 * x + m) % (x - 4) = 0 ↔ m = -224 := by
  sorry

end polynomial_divisibility_l1492_149275


namespace prob_A_miss_at_least_once_prob_A_hit_twice_B_hit_once_l1492_149223

-- Define the probabilities
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define the number of shots
def num_shots : ℕ := 3

-- Theorem for part (I)
theorem prob_A_miss_at_least_once :
  1 - prob_A_hit ^ num_shots = 19/27 := by sorry

-- Theorem for part (II)
theorem prob_A_hit_twice_B_hit_once :
  (Nat.choose num_shots 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit) *
  (Nat.choose num_shots 1 : ℚ) * prob_B_hit * (1 - prob_B_hit)^2 = 1/16 := by sorry

end prob_A_miss_at_least_once_prob_A_hit_twice_B_hit_once_l1492_149223


namespace gcd_65_130_l1492_149298

theorem gcd_65_130 : Nat.gcd 65 130 = 65 := by
  sorry

end gcd_65_130_l1492_149298


namespace triangle_type_l1492_149233

theorem triangle_type (A B C : ℝ) (a b c : ℝ) 
  (h : a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.sin C) :
  A = π / 4 ∧ B = π / 4 ∧ C = π / 2 := by
  sorry

#check triangle_type

end triangle_type_l1492_149233


namespace correct_sums_count_l1492_149299

theorem correct_sums_count (total : ℕ) (correct : ℕ) (incorrect : ℕ)
  (h1 : incorrect = 2 * correct)
  (h2 : total = correct + incorrect)
  (h3 : total = 24) :
  correct = 8 := by
  sorry

end correct_sums_count_l1492_149299


namespace license_plate_combinations_l1492_149209

/-- The number of consonants available for the license plate. -/
def num_consonants : ℕ := 21

/-- The number of vowels available for the license plate. -/
def num_vowels : ℕ := 5

/-- The number of digits available for the license plate. -/
def num_digits : ℕ := 10

/-- The total number of possible license plate combinations. -/
def total_combinations : ℕ := num_consonants^2 * num_vowels^2 * num_digits

/-- Theorem stating that the total number of license plate combinations is 110,250. -/
theorem license_plate_combinations : total_combinations = 110250 := by
  sorry

end license_plate_combinations_l1492_149209


namespace max_value_expression_l1492_149231

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({1, 2, 3, 4} : Set ℕ) →
  b ∈ ({1, 2, 3, 4} : Set ℕ) →
  c ∈ ({1, 2, 3, 4} : Set ℕ) →
  d ∈ ({1, 2, 3, 4} : Set ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  c * a^b - d ≤ 127 :=
by sorry

end max_value_expression_l1492_149231


namespace sum_of_sequence_equals_11920_l1492_149225

def integerSequence : List Nat := List.range 40 |>.map (fun i => 103 + 10 * i)

theorem sum_of_sequence_equals_11920 : (integerSequence.sum = 11920) := by
  sorry

end sum_of_sequence_equals_11920_l1492_149225


namespace immediate_prepayment_better_l1492_149262

/-- Represents a mortgage loan with fixed interest rate and annuity payments -/
structure MortgageLoan where
  S : ℝ  -- Initial loan balance
  T : ℝ  -- Monthly payment amount
  r : ℝ  -- Interest rate for the period
  (T_positive : T > 0)
  (r_nonnegative : r ≥ 0)
  (r_less_than_one : r < 1)

/-- Calculates the final balance after immediate partial prepayment -/
def final_balance_immediate (loan : MortgageLoan) : ℝ :=
  loan.S - 2 * loan.T + loan.r * loan.S - 0.5 * loan.r * loan.T + (0.5 * loan.r * loan.S)^2

/-- Calculates the final balance when waiting until the end of the period -/
def final_balance_waiting (loan : MortgageLoan) : ℝ :=
  loan.S - 2 * loan.T + loan.r * loan.S

/-- Theorem stating that immediate partial prepayment results in a lower final balance -/
theorem immediate_prepayment_better (loan : MortgageLoan) :
  final_balance_immediate loan < final_balance_waiting loan :=
by sorry

end immediate_prepayment_better_l1492_149262


namespace negation_of_implication_l1492_149238

theorem negation_of_implication (x y : ℝ) :
  ¬(x + y ≤ 0 → x ≤ 0 ∨ y ≤ 0) ↔ (x + y > 0 → x > 0 ∧ y > 0) :=
by sorry

end negation_of_implication_l1492_149238


namespace simon_age_is_10_l1492_149250

def alvin_age : ℕ := 30

def simon_age : ℕ := alvin_age / 2 - 5

theorem simon_age_is_10 : simon_age = 10 := by
  sorry

end simon_age_is_10_l1492_149250


namespace jeremy_purchase_l1492_149248

theorem jeremy_purchase (computer_price : ℝ) (accessory_percentage : ℝ) (initial_money_factor : ℝ) : 
  computer_price = 3000 →
  accessory_percentage = 0.1 →
  initial_money_factor = 2 →
  let accessory_price := computer_price * accessory_percentage
  let initial_money := computer_price * initial_money_factor
  let total_spent := computer_price + accessory_price
  initial_money - total_spent = 2700 := by
sorry

end jeremy_purchase_l1492_149248


namespace triangle_side_difference_l1492_149255

theorem triangle_side_difference (x : ℤ) : 
  (x > 5 ∧ x < 11) → (11 - 6 = 4) := by
  sorry

end triangle_side_difference_l1492_149255


namespace dance_pairing_l1492_149276

-- Define the types for boys and girls
variable {Boy Girl : Type}

-- Define the dancing relation
variable (danced_with : Boy → Girl → Prop)

-- State the theorem
theorem dance_pairing
  (h1 : ∀ b : Boy, ∃ g : Girl, ¬danced_with b g)
  (h2 : ∀ g : Girl, ∃ b : Boy, danced_with b g)
  : ∃ (g g' : Boy) (f f' : Girl),
    danced_with g f ∧ ¬danced_with g f' ∧
    danced_with g' f' ∧ ¬danced_with g' f :=
by sorry

end dance_pairing_l1492_149276


namespace fourth_part_diminished_l1492_149281

theorem fourth_part_diminished (x : ℝ) (y : ℝ) (h : x = 280) (h2 : x/5 + 7 = x/4 - y) : y = 7 := by
  sorry

end fourth_part_diminished_l1492_149281


namespace not_all_sqrt5_periodic_all_sqrt3_periodic_l1492_149249

-- Define the function types
def RealFunction := ℝ → ℝ

-- Define the functional equations
def SatisfiesSqrt5Equation (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (x - 1) + f (x + 1) = Real.sqrt 5 * f x

def SatisfiesSqrt3Equation (g : RealFunction) : Prop :=
  ∀ x : ℝ, g (x - 1) + g (x + 1) = Real.sqrt 3 * g x

-- Define periodicity
def IsPeriodic (f : RealFunction) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = f x

-- Theorem statements
theorem not_all_sqrt5_periodic :
  ∃ f : RealFunction, SatisfiesSqrt5Equation f ∧ ¬IsPeriodic f :=
sorry

theorem all_sqrt3_periodic :
  ∀ g : RealFunction, SatisfiesSqrt3Equation g → IsPeriodic g :=
sorry

end not_all_sqrt5_periodic_all_sqrt3_periodic_l1492_149249
