import Mathlib

namespace water_level_change_time_correct_l393_39384

noncomputable def water_level_change_time (S H h s V g : ℝ) : ℝ :=
  let a := S / (0.6 * s * Real.sqrt (2 * g))
  let b := V / (0.6 * s * Real.sqrt (2 * g))
  2 * a * (Real.sqrt H - Real.sqrt (H - h) + b * Real.log (abs ((Real.sqrt H - b) / (Real.sqrt (H - h) - b))))

theorem water_level_change_time_correct
  (S H h s V g : ℝ)
  (h_S : S > 0)
  (h_H : H > 0)
  (h_h : 0 < h ∧ h < H)
  (h_s : s > 0)
  (h_V : V ≥ 0)
  (h_g : g > 0) :
  ∃ T : ℝ, T = water_level_change_time S H h s V g ∧ T > 0 :=
sorry

end water_level_change_time_correct_l393_39384


namespace cereal_box_servings_l393_39355

def cereal_box_problem (total_cups : ℕ) (cups_per_serving : ℕ) : ℕ :=
  total_cups / cups_per_serving

theorem cereal_box_servings :
  cereal_box_problem 18 2 = 9 := by
  sorry

end cereal_box_servings_l393_39355


namespace negation_equivalence_l393_39364

/-- An exponential function -/
def ExponentialFunction (f : ℝ → ℝ) : Prop := sorry

/-- A monotonic function -/
def MonotonicFunction (f : ℝ → ℝ) : Prop := sorry

/-- The statement "All exponential functions are monotonic functions" -/
def AllExponentialAreMonotonic : Prop :=
  ∀ f : ℝ → ℝ, ExponentialFunction f → MonotonicFunction f

/-- The negation of "All exponential functions are monotonic functions" -/
def NegationAllExponentialAreMonotonic : Prop :=
  ∃ f : ℝ → ℝ, ExponentialFunction f ∧ ¬MonotonicFunction f

/-- Theorem: The negation of "All exponential functions are monotonic functions"
    is equivalent to "There exists at least one exponential function that is not a monotonic function" -/
theorem negation_equivalence :
  ¬AllExponentialAreMonotonic ↔ NegationAllExponentialAreMonotonic :=
sorry

end negation_equivalence_l393_39364


namespace smallest_with_twelve_divisors_l393_39324

/-- The number of positive integer divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly 12 divisors -/
def has_twelve_divisors (n : ℕ+) : Prop :=
  num_divisors n = 12

theorem smallest_with_twelve_divisors :
  ∃ (n : ℕ+), has_twelve_divisors n ∧ ∀ (m : ℕ+), has_twelve_divisors m → n ≤ m :=
by
  use 72
  sorry

end smallest_with_twelve_divisors_l393_39324


namespace queen_then_diamond_probability_l393_39399

/-- Standard deck of cards --/
def standard_deck : ℕ := 52

/-- Number of Queens in a standard deck --/
def num_queens : ℕ := 4

/-- Number of diamonds in a standard deck --/
def num_diamonds : ℕ := 13

/-- Probability of drawing a Queen as the first card and a diamond as the second --/
def prob_queen_then_diamond : ℚ := 52 / 221

theorem queen_then_diamond_probability :
  prob_queen_then_diamond = (num_queens / standard_deck) * (num_diamonds / (standard_deck - 1)) :=
sorry

end queen_then_diamond_probability_l393_39399


namespace y_derivative_l393_39369

noncomputable def y (x : ℝ) : ℝ := 
  (3 / (8 * Real.sqrt 2)) * Real.log ((Real.sqrt 2 + Real.tanh x) / (Real.sqrt 2 - Real.tanh x)) - 
  (Real.tanh x) / (4 * (2 - Real.tanh x ^ 2))

theorem y_derivative (x : ℝ) : 
  deriv y x = 1 / (2 + Real.cosh x ^ 2) ^ 2 :=
sorry

end y_derivative_l393_39369


namespace circle_radius_tangent_to_square_sides_l393_39351

open Real

theorem circle_radius_tangent_to_square_sides (a : ℝ) :
  a = Real.sqrt (2 + Real.sqrt 2) →
  ∃ (R : ℝ),
    R = Real.sqrt 2 + Real.sqrt (2 - Real.sqrt 2) ∧
    (Real.sin (π / 8) = Real.sqrt (2 - Real.sqrt 2) / 2) ∧
    (∃ (O : ℝ × ℝ) (C : ℝ × ℝ),
      -- O is the center of the circle, C is a vertex of the square
      -- The distance between O and C is related to R and the sine of 22.5°
      Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2) = 4 * R / Real.sqrt (2 - Real.sqrt 2) ∧
      -- The angle between the tangents from C is 45°
      Real.arctan (R / (Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2) - R)) = π / 8) :=
by
  sorry

end circle_radius_tangent_to_square_sides_l393_39351


namespace starting_lineup_count_l393_39349

def team_size : ℕ := 15
def lineup_size : ℕ := 7
def all_stars : ℕ := 3
def guards : ℕ := 5

theorem starting_lineup_count :
  (Finset.sum (Finset.range 3) (λ i =>
    Nat.choose guards (i + 2) * Nat.choose (team_size - all_stars - guards) (lineup_size - all_stars - (i + 2)))) = 285 := by
  sorry

end starting_lineup_count_l393_39349


namespace unique_solution_quadratic_inequality_l393_39331

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, |x^2 + a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
sorry

end unique_solution_quadratic_inequality_l393_39331


namespace system_solution_l393_39365

theorem system_solution : 
  ∃ (x y z : ℝ), 
    (x = 1/2 ∧ y = 0 ∧ z = 0) ∧
    (2*x + 3*y + z = 1) ∧
    (4*x - y + 2*z = 2) ∧
    (8*x + 5*y + 3*z = 4) := by
  sorry

end system_solution_l393_39365


namespace quadratic_equation_roots_l393_39332

/-- The quadratic equation 3x^2 - 4x + 1 = 0 has two distinct real roots -/
theorem quadratic_equation_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 3 * x₁^2 - 4 * x₁ + 1 = 0 ∧ 3 * x₂^2 - 4 * x₂ + 1 = 0 := by
  sorry

end quadratic_equation_roots_l393_39332


namespace parallel_vectors_solution_l393_39339

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

theorem parallel_vectors_solution (a : ℝ) :
  let m : Vector2D := ⟨a, -2⟩
  let n : Vector2D := ⟨1, 1 - a⟩
  parallel m n → a = 2 ∨ a = -1 := by
  sorry

end parallel_vectors_solution_l393_39339


namespace cake_supplies_cost_l393_39321

/-- Proves that the cost of supplies for a cake is $54 given the specified conditions -/
theorem cake_supplies_cost (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℕ) (profit : ℕ) : 
  hours_per_day = 2 →
  days_worked = 4 →
  hourly_rate = 22 →
  profit = 122 →
  (hours_per_day * days_worked * hourly_rate) - profit = 54 :=
by sorry

end cake_supplies_cost_l393_39321


namespace max_value_of_expression_l393_39386

def S : Finset ℕ := {1, 2, 3, 4}

def f (k x y z : ℕ) : ℕ := k * x^y - z

theorem max_value_of_expression :
  ∃ (k x y z : ℕ), k ∈ S ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S ∧
    f k x y z = 127 ∧
    ∀ (k' x' y' z' : ℕ), k' ∈ S → x' ∈ S → y' ∈ S → z' ∈ S →
      f k' x' y' z' ≤ 127 :=
by sorry

end max_value_of_expression_l393_39386


namespace danny_soda_consumption_l393_39389

theorem danny_soda_consumption (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 →
  (1 - x / 100) + (0.3 + 0.3) = 0.7 →
  x = 90 := by
sorry

end danny_soda_consumption_l393_39389


namespace cheaper_lens_price_l393_39379

theorem cheaper_lens_price (original_price : ℝ) (discount_rate : ℝ) (savings : ℝ) : 
  original_price = 300 →
  discount_rate = 0.2 →
  savings = 20 →
  original_price * (1 - discount_rate) - savings = 220 := by
sorry

end cheaper_lens_price_l393_39379


namespace square_sum_equals_four_l393_39307

theorem square_sum_equals_four (x y : ℝ) (h1 : x + y = -4) (h2 : x = 6 / y) : x^2 + y^2 = 4 := by
  sorry

end square_sum_equals_four_l393_39307


namespace nigels_money_theorem_l393_39330

/-- Represents the amount of money Nigel has at different stages --/
structure NigelsMoney where
  initial : ℕ
  afterFirstGiveaway : ℕ
  afterMotherGift : ℕ
  final : ℕ

/-- Theorem stating Nigel's final amount is $10 more than twice his initial amount --/
theorem nigels_money_theorem (n : NigelsMoney) (h1 : n.initial = 45)
  (h2 : n.afterMotherGift = n.afterFirstGiveaway + 80)
  (h3 : n.final = n.afterMotherGift - 25)
  (h4 : n.afterFirstGiveaway < n.initial) :
  n.final = 2 * n.initial + 10 := by
  sorry

end nigels_money_theorem_l393_39330


namespace candidate_vote_difference_l393_39381

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 7800 →
  candidate_percentage = 35 / 100 →
  (total_votes : ℚ) * candidate_percentage < (total_votes : ℚ) * (1 - candidate_percentage) →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 2340 := by
  sorry

end candidate_vote_difference_l393_39381


namespace intersection_equals_interval_l393_39359

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x + 4}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the interval [4, +∞)
def interval_four_to_inf : Set ℝ := {x | x ≥ 4}

-- Theorem statement
theorem intersection_equals_interval : A_intersect_B = interval_four_to_inf := by
  sorry

end intersection_equals_interval_l393_39359


namespace triangular_display_total_l393_39371

/-- Represents a triangular display of cans -/
structure CanDisplay where
  bottom_layer : ℕ
  second_layer : ℕ
  top_layer : ℕ

/-- Calculates the total number of cans in the display -/
def total_cans (d : CanDisplay) : ℕ :=
  sorry

/-- Theorem stating that the specific triangular display contains 165 cans -/
theorem triangular_display_total (d : CanDisplay) 
  (h1 : d.bottom_layer = 30)
  (h2 : d.second_layer = 27)
  (h3 : d.top_layer = 3) :
  total_cans d = 165 := by
  sorry

end triangular_display_total_l393_39371


namespace expression_simplification_l393_39390

theorem expression_simplification (a b c : ℝ) 
  (ha : a ≠ 2) (hb : b ≠ 3) (hc : c ≠ 6) : 
  ((a - 2) / (6 - c)) * ((b - 3) / (2 - a)) * ((c - 6) / (3 - b)) = -1 := by
  sorry

end expression_simplification_l393_39390


namespace distance_to_line_l393_39338

/-- The smallest distance from (0, 0) to the line y = 4/3 * x - 100 -/
def smallest_distance : ℝ := 60

/-- The equation of the line in the form Ax + By + C = 0 -/
def line_equation (x y : ℝ) : Prop := -4 * x + 3 * y + 300 = 0

/-- The point from which we're measuring the distance -/
def origin : ℝ × ℝ := (0, 0)

theorem distance_to_line :
  smallest_distance = 
    (‖-4 * origin.1 + 3 * origin.2 + 300‖ : ℝ) / Real.sqrt ((-4)^2 + 3^2) :=
sorry

end distance_to_line_l393_39338


namespace complex_product_real_implies_ratio_l393_39308

theorem complex_product_real_implies_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (r : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = r) : p / q = 3 / 4 := by
  sorry

end complex_product_real_implies_ratio_l393_39308


namespace total_transport_is_405_l393_39314

/-- Calculates the total number of people transported by two boats over two days -/
def total_people_transported (boat_a_capacity : ℕ) (boat_b_capacity : ℕ)
  (day1_a_trips : ℕ) (day1_b_trips : ℕ)
  (day2_a_trips : ℕ) (day2_b_trips : ℕ) : ℕ :=
  (boat_a_capacity * day1_a_trips + boat_b_capacity * day1_b_trips) +
  (boat_a_capacity * day2_a_trips + boat_b_capacity * day2_b_trips)

/-- Theorem stating that the total number of people transported is 405 -/
theorem total_transport_is_405 :
  total_people_transported 20 15 7 5 5 6 = 405 := by
  sorry

#eval total_people_transported 20 15 7 5 5 6

end total_transport_is_405_l393_39314


namespace fruit_punch_theorem_l393_39352

/-- Calculates the total amount of fruit punch given the amount of orange punch -/
def total_fruit_punch (orange_punch : ℝ) : ℝ :=
  let cherry_punch := 2 * orange_punch
  let apple_juice := cherry_punch - 1.5
  orange_punch + cherry_punch + apple_juice

/-- Theorem stating that given 4.5 liters of orange punch, the total fruit punch is 21 liters -/
theorem fruit_punch_theorem : total_fruit_punch 4.5 = 21 := by
  sorry

end fruit_punch_theorem_l393_39352


namespace isosceles_triangle_perimeter_l393_39309

/-- An isosceles triangle with side lengths 2 and 4 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  is_isosceles : base = 2 ∧ leg = 4

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.base + 2 * t.leg

/-- Theorem: The perimeter of an isosceles triangle with side lengths 2 and 4 is 10 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, perimeter t = 10 := by
  sorry

#check isosceles_triangle_perimeter

end isosceles_triangle_perimeter_l393_39309


namespace rectangle_partition_into_L_shapes_rectangle_1985_1987_not_partitionable_rectangle_1987_1989_partitionable_l393_39336

/-- An L-shape is a figure composed of 3 unit squares -/
def LShape : Nat := 3

/-- Checks if a number is divisible by 3 -/
def isDivisibleBy3 (n : Nat) : Prop := n % 3 = 0

/-- Checks if a number leaves a remainder of 2 when divided by 3 -/
def hasRemainder2 (n : Nat) : Prop := n % 3 = 2

/-- Theorem: A rectangle can be partitioned into L-shapes iff
    1) Its area is divisible by 3, and
    2) At least one side is divisible by 3, or both sides have remainder 2 when divided by 3 -/
theorem rectangle_partition_into_L_shapes (m n : Nat) :
  (isDivisibleBy3 (m * n)) ∧ 
  (isDivisibleBy3 m ∨ isDivisibleBy3 n ∨ (hasRemainder2 m ∧ hasRemainder2 n)) ↔ 
  ∃ (k : Nat), m * n = k * LShape := by sorry

/-- Corollary: 1985 × 1987 rectangle cannot be partitioned into L-shapes -/
theorem rectangle_1985_1987_not_partitionable :
  ¬ ∃ (k : Nat), 1985 * 1987 = k * LShape := by sorry

/-- Corollary: 1987 × 1989 rectangle can be partitioned into L-shapes -/
theorem rectangle_1987_1989_partitionable :
  ∃ (k : Nat), 1987 * 1989 = k * LShape := by sorry

end rectangle_partition_into_L_shapes_rectangle_1985_1987_not_partitionable_rectangle_1987_1989_partitionable_l393_39336


namespace floor_ceil_sum_l393_39340

theorem floor_ceil_sum : ⌊(-3.75 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ + (1/2 : ℝ) = 31.5 := by
  sorry

end floor_ceil_sum_l393_39340


namespace fibConversionAccuracy_l393_39313

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define the Fibonacci representation of a number
def fibRep (n : ℕ) : List ℕ := sorry

-- Define the conversion function using Fibonacci representation
def kmToMilesFib (km : ℕ) : ℕ := sorry

-- Define the exact conversion from km to miles
def kmToMilesExact (km : ℕ) : ℚ :=
  (km : ℚ) / 1.609

-- Main theorem
theorem fibConversionAccuracy :
  ∀ n : ℕ, n ≤ 100 →
    |((kmToMilesFib n : ℚ) - kmToMilesExact n)| < 2/3 := by sorry

end fibConversionAccuracy_l393_39313


namespace imaginary_part_of_complex_fraction_l393_39377

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  (2 / (1 - i)).im = 1 := by sorry

end imaginary_part_of_complex_fraction_l393_39377


namespace unique_code_l393_39303

/-- Represents a three-digit code --/
structure Code where
  A : Nat
  B : Nat
  C : Nat
  h1 : A < 10
  h2 : B < 10
  h3 : C < 10
  h4 : A ≠ B
  h5 : A ≠ C
  h6 : B ≠ C

/-- The conditions for the code --/
def satisfiesConditions (code : Code) : Prop :=
  code.B > code.A ∧
  code.A < code.C ∧
  code.B * 10 + code.B + code.A * 10 + code.A = code.C * 10 + code.C ∧
  code.B * 10 + code.B + code.A * 10 + code.A = 242

theorem unique_code :
  ∃! code : Code, satisfiesConditions code ∧ code.A = 2 ∧ code.B = 3 ∧ code.C = 2 := by
  sorry

end unique_code_l393_39303


namespace sum_remainder_mod_seven_l393_39316

theorem sum_remainder_mod_seven : (2^2003 + 2003^2) % 7 = 5 := by
  sorry

end sum_remainder_mod_seven_l393_39316


namespace new_encoding_correct_l393_39370

-- Define the encoding function
def encode (c : Char) : String :=
  match c with
  | 'A' => "21"
  | 'B' => "122"
  | 'C' => "1"
  | _ => ""

-- Define the decoding function (simplified for this problem)
def decode (s : String) : String :=
  if s = "011011010011" then "ABCBA" else ""

-- Theorem statement
theorem new_encoding_correct : 
  let original := "011011010011"
  let decoded := decode original
  String.join (List.map encode decoded.data) = "211221121" := by
  sorry


end new_encoding_correct_l393_39370


namespace unique_albums_count_l393_39361

/-- The number of albums in either Andrew's, John's, or Sarah's collection, but not in all three -/
def unique_albums (shared_albums andrew_total john_unique sarah_unique : ℕ) : ℕ :=
  (andrew_total - shared_albums) + john_unique + sarah_unique

/-- Theorem stating the number of unique albums across the three collections -/
theorem unique_albums_count :
  unique_albums 10 20 5 3 = 18 := by
  sorry

end unique_albums_count_l393_39361


namespace brenda_skittles_l393_39346

theorem brenda_skittles (x : ℕ) : x + 8 = 15 → x = 7 := by
  sorry

end brenda_skittles_l393_39346


namespace probability_one_has_no_growth_pie_l393_39319

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := total_pies - growth_pies
def pies_given : ℕ := 3

def probability_no_growth_pie : ℚ := 7/10

theorem probability_one_has_no_growth_pie :
  (1 : ℚ) - (Nat.choose shrink_pies (pies_given - 1) : ℚ) / (Nat.choose total_pies pies_given : ℚ) = probability_no_growth_pie :=
sorry

end probability_one_has_no_growth_pie_l393_39319


namespace exists_valid_arrangement_l393_39347

/-- A permutation of integers from 1 to n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Check if a number is divisible by either 4 or 7 -/
def isDivisibleBy4Or7 (n : ℕ) : Prop := n % 4 = 0 ∨ n % 7 = 0

/-- Check if a permutation satisfies the adjacency condition when arranged in a circle -/
def isValidCircularArrangement (p : Permutation 2015) : Prop :=
  ∀ i : Fin 2015, isDivisibleBy4Or7 ((p i).val + (p (i + 1)).val)

theorem exists_valid_arrangement : ∃ p : Permutation 2015, isValidCircularArrangement p := by
  sorry

end exists_valid_arrangement_l393_39347


namespace dishwasher_manager_ratio_l393_39387

/-- The hourly wages of employees at Joe's Steakhouse -/
structure Wages where
  manager : ℝ
  chef : ℝ
  dishwasher : ℝ

/-- The conditions of the wages at Joe's Steakhouse -/
def wage_conditions (w : Wages) : Prop :=
  w.chef = w.dishwasher * 1.22 ∧
  w.manager = 8.50 ∧
  w.manager = w.chef + 3.315

/-- The theorem stating the ratio of dishwasher to manager wages -/
theorem dishwasher_manager_ratio (w : Wages) :
  wage_conditions w → w.dishwasher / w.manager = 0.5 := by
  sorry


end dishwasher_manager_ratio_l393_39387


namespace angle_C_is_120_degrees_l393_39397

theorem angle_C_is_120_degrees 
  (A B : ℝ) 
  (m : ℝ × ℝ) 
  (n : ℝ × ℝ) 
  (h1 : m = (Real.sqrt 3 * Real.sin A, Real.sin B))
  (h2 : n = (Real.cos B, Real.sqrt 3 * Real.cos A))
  (h3 : m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B))
  : ∃ C : ℝ, C = 2 * π / 3 :=
by sorry

end angle_C_is_120_degrees_l393_39397


namespace rectangle_ratio_l393_39343

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0) : 
  (s + 2*y = 3*s) →  -- Outer square side length
  (x + s = 3*s) →    -- Perpendicular arrangement
  ((3*s)^2 = 9*s^2)  -- Area ratio
  → x / y = 2 := by
sorry

end rectangle_ratio_l393_39343


namespace laptop_price_proof_l393_39394

theorem laptop_price_proof (sticker_price : ℝ) : 
  (0.9 * sticker_price - 100 = 0.8 * sticker_price - 20) → 
  sticker_price = 800 := by
sorry

end laptop_price_proof_l393_39394


namespace circles_intersect_l393_39304

/-- Two circles in a plane -/
structure TwoCircles where
  /-- The first circle: x² + y² - 2x = 0 -/
  c1 : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 1}
  /-- The second circle: x² + y² + 4y = 0 -/
  c2 : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 + 2)^2 = 4}

/-- The circles intersect if there exists a point that belongs to both circles -/
def intersect (tc : TwoCircles) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ tc.c1 ∧ p ∈ tc.c2

/-- Theorem stating that the two given circles intersect -/
theorem circles_intersect : ∀ tc : TwoCircles, intersect tc := by
  sorry

end circles_intersect_l393_39304


namespace jake_and_sister_weight_l393_39342

theorem jake_and_sister_weight (jake_weight : ℕ) (sister_weight : ℕ) : 
  jake_weight = 108 →
  jake_weight - 12 = 2 * sister_weight →
  jake_weight + sister_weight = 156 :=
by sorry

end jake_and_sister_weight_l393_39342


namespace isosceles_triangle_quadratic_roots_l393_39395

theorem isosceles_triangle_quadratic_roots (k : ℝ) : 
  (∃ (a b : ℝ), 
    -- a and b are the roots of the quadratic equation
    a^2 - 12*a + k = 0 ∧ 
    b^2 - 12*b + k = 0 ∧ 
    -- a and b are equal (isosceles triangle)
    a = b ∧ 
    -- triangle inequality
    3 + a > b ∧ 3 + b > a ∧ a + b > 3 ∧
    -- one side is 3
    3 > 0) → 
  k = 36 := by
sorry

end isosceles_triangle_quadratic_roots_l393_39395


namespace complex_simplification_l393_39337

theorem complex_simplification :
  (-5 + 3 * Complex.I) - (2 - 7 * Complex.I) + (1 + 2 * Complex.I) = -6 + 12 * Complex.I :=
by sorry

end complex_simplification_l393_39337


namespace max_prize_winners_l393_39366

/-- Represents a tournament with given number of players and point thresholds. -/
structure Tournament :=
  (num_players : ℕ)
  (win_points : ℕ)
  (draw_points : ℕ)
  (loss_points : ℕ)
  (prize_threshold : ℕ)

/-- Calculates the total number of games in a round-robin tournament. -/
def total_games (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Calculates the total points available in the tournament. -/
def total_points (t : Tournament) : ℕ :=
  total_games t.num_players * t.win_points

/-- Theorem stating the maximum number of prize winners in the specific tournament. -/
theorem max_prize_winners (t : Tournament) 
  (h1 : t.num_players = 15)
  (h2 : t.win_points = 2)
  (h3 : t.draw_points = 1)
  (h4 : t.loss_points = 0)
  (h5 : t.prize_threshold = 20) :
  ∃ (max_winners : ℕ), max_winners = 9 ∧ 
  (∀ (n : ℕ), n > max_winners → 
    n * t.prize_threshold > total_points t) :=
sorry

end max_prize_winners_l393_39366


namespace area_difference_zero_l393_39310

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- A point inside the polygon -/
def InteriorPoint (p : RegularPolygon n) := ℝ × ℝ

/-- The area difference function between black and white triangles -/
def areaDifference (p : RegularPolygon n) (point : InteriorPoint p) : ℝ := sorry

/-- Theorem stating that the area difference is always zero -/
theorem area_difference_zero (n : ℕ) (p : RegularPolygon n) (point : InteriorPoint p) :
  areaDifference p point = 0 := by sorry

end area_difference_zero_l393_39310


namespace gold_alloy_percentage_l393_39376

/-- Proves that adding pure gold to an alloy results in a specific gold percentage -/
theorem gold_alloy_percentage 
  (original_weight : ℝ) 
  (original_percentage : ℝ) 
  (added_gold : ℝ) 
  (h1 : original_weight = 48) 
  (h2 : original_percentage = 0.25) 
  (h3 : added_gold = 12) : 
  (original_percentage * original_weight + added_gold) / (original_weight + added_gold) = 0.4 := by
  sorry

end gold_alloy_percentage_l393_39376


namespace brownies_before_division_l393_39301

def initial_brownies : ℕ := 24  -- 2 dozen

def father_ate (n : ℕ) : ℕ := n / 3

def mooney_ate (n : ℕ) : ℕ := n / 4  -- 25% = 1/4

def benny_ate (n : ℕ) : ℕ := n * 2 / 5

def snoopy_ate : ℕ := 3

def mother_baked_wednesday : ℕ := 18  -- 1.5 dozen

def mother_baked_thursday : ℕ := 36  -- 3 dozen

def final_brownies : ℕ :=
  let after_father := initial_brownies - father_ate initial_brownies
  let after_mooney := after_father - mooney_ate after_father
  let after_benny := after_mooney - benny_ate after_mooney
  let after_snoopy := after_benny - snoopy_ate
  after_snoopy + mother_baked_wednesday + mother_baked_thursday

theorem brownies_before_division :
  final_brownies = 59 := by sorry

end brownies_before_division_l393_39301


namespace quadratic_inequality_and_negation_l393_39372

theorem quadratic_inequality_and_negation :
  (∀ x : ℝ, x^2 + 2*x + 3 > 0) ∧
  (¬(∀ x : ℝ, x^2 + 2*x + 3 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 3 ≤ 0) :=
by sorry

end quadratic_inequality_and_negation_l393_39372


namespace function_composition_equality_l393_39350

/-- Given real numbers a, b, c, d, and functions f and h, 
    prove that f(h(x)) = h(f(x)) for all x if and only if a = c or b = d -/
theorem function_composition_equality 
  (a b c d : ℝ) 
  (f : ℝ → ℝ) 
  (h : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x + b) 
  (hh : ∀ x, h x = c * x + d) : 
  (∀ x, f (h x) = h (f x)) ↔ (a = c ∨ b = d) := by
  sorry

end function_composition_equality_l393_39350


namespace vacation_cost_division_l393_39306

theorem vacation_cost_division (total_cost : ℕ) (n : ℕ) : 
  total_cost = 1000 → 
  (total_cost / 5 + 50 = total_cost / n) → 
  n = 4 := by
sorry

end vacation_cost_division_l393_39306


namespace product_65_35_l393_39345

theorem product_65_35 : 65 * 35 = 2275 := by
  sorry

end product_65_35_l393_39345


namespace chocolate_bars_bought_correct_number_of_bars_l393_39333

def sugar_per_chocolate_bar : ℕ := 10
def sugar_in_lollipop : ℕ := 37
def calories_in_lollipop : ℕ := 190
def total_sugar : ℕ := 177

theorem chocolate_bars_bought : ℕ :=
  (total_sugar - sugar_in_lollipop) / sugar_per_chocolate_bar

theorem correct_number_of_bars : chocolate_bars_bought = 14 := by
  sorry

end chocolate_bars_bought_correct_number_of_bars_l393_39333


namespace root_exists_in_interval_l393_39388

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 8

-- State the theorem
theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  -- Assuming f(1) < 0 and f(2) > 0
  have h1 : f 1 < 0 := by sorry
  have h2 : f 2 > 0 := by sorry
  
  -- The proof would go here
  sorry

end root_exists_in_interval_l393_39388


namespace infinite_perfect_squares_l393_39375

theorem infinite_perfect_squares (k : ℕ+) : 
  ∃ n : ℕ+, ∃ m : ℕ, (n * 2^k.val - 7 : ℤ) = m^2 :=
sorry

end infinite_perfect_squares_l393_39375


namespace luke_connor_sleep_difference_l393_39373

theorem luke_connor_sleep_difference (connor_sleep : ℕ) (puppy_sleep : ℕ) : 
  connor_sleep = 6 →
  puppy_sleep = 16 →
  puppy_sleep = 2 * (connor_sleep + (puppy_sleep / 2 - connor_sleep)) →
  puppy_sleep / 2 - connor_sleep = 2 :=
by sorry

end luke_connor_sleep_difference_l393_39373


namespace min_value_of_f_l393_39362

/-- The polynomial function in two variables -/
def f (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

/-- The minimum value of the polynomial function -/
theorem min_value_of_f :
  ∀ x y : ℝ, f x y ≥ -18 ∧ ∃ a b : ℝ, f a b = -18 :=
by sorry

end min_value_of_f_l393_39362


namespace quadratic_inequality_range_l393_39327

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, m * x^2 + m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by
  sorry

end quadratic_inequality_range_l393_39327


namespace total_paper_clips_l393_39356

/-- The number of boxes used to distribute paper clips -/
def num_boxes : ℕ := 9

/-- The number of paper clips in each box -/
def clips_per_box : ℕ := 9

/-- Theorem: The total number of paper clips collected is 81 -/
theorem total_paper_clips : num_boxes * clips_per_box = 81 := by
  sorry

end total_paper_clips_l393_39356


namespace a_alone_time_equals_b_alone_time_l393_39354

/-- Two workers finishing a job -/
structure WorkerPair where
  total_time : ℝ
  b_alone_time : ℝ
  work : ℝ

/-- The time it takes for worker a to finish the job alone -/
def a_alone_time (w : WorkerPair) : ℝ :=
  w.b_alone_time

theorem a_alone_time_equals_b_alone_time (w : WorkerPair)
  (h1 : w.total_time = 10)
  (h2 : w.b_alone_time = 20) :
  a_alone_time w = w.b_alone_time :=
by
  sorry

#check a_alone_time_equals_b_alone_time

end a_alone_time_equals_b_alone_time_l393_39354


namespace spherical_coordinate_transformation_l393_39391

/-- Given a point with rectangular coordinates (-5, -7, 4) and spherical coordinates (ρ, θ, φ),
    prove that the point with spherical coordinates (ρ, θ + π, -φ) has rectangular coordinates (5, 7, 4). -/
theorem spherical_coordinate_transformation (ρ θ φ : ℝ) :
  (ρ * Real.sin φ * Real.cos θ = -5) →
  (ρ * Real.sin φ * Real.sin θ = -7) →
  (ρ * Real.cos φ = 4) →
  (ρ * Real.sin (-φ) * Real.cos (θ + π) = 5) ∧
  (ρ * Real.sin (-φ) * Real.sin (θ + π) = 7) ∧
  (ρ * Real.cos (-φ) = 4) :=
by sorry

end spherical_coordinate_transformation_l393_39391


namespace blue_lights_l393_39383

/-- The number of blue lights on a Christmas tree -/
theorem blue_lights (total : ℕ) (red : ℕ) (yellow : ℕ) 
  (h1 : total = 95)
  (h2 : red = 26)
  (h3 : yellow = 37) :
  total - (red + yellow) = 32 := by
  sorry

end blue_lights_l393_39383


namespace divisibility_problem_l393_39368

theorem divisibility_problem (n : ℕ) (h : n = (List.range 2001).foldl (· * ·) 1) :
  ∃ k : ℤ, n + (4003 * n - 4002) = 4003 * k := by
  sorry

end divisibility_problem_l393_39368


namespace pirate_loot_sum_l393_39341

/-- Converts a base 5 number (represented as a list of digits) to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The loot values in base 5 --/
def silverware : List Nat := [3, 1, 2, 4]
def diamondTiaras : List Nat := [1, 0, 1, 3]
def silkScarves : List Nat := [2, 0, 2]

/-- The theorem to prove --/
theorem pirate_loot_sum :
  base5ToBase10 silverware + base5ToBase10 diamondTiaras + base5ToBase10 silkScarves = 1011 := by
  sorry

end pirate_loot_sum_l393_39341


namespace ammonium_hydroxide_formation_l393_39396

/-- Represents a chemical compound in a reaction --/
structure Compound where
  name : String
  moles : ℚ

/-- Represents a chemical reaction --/
structure Reaction where
  reactants : List Compound
  products : List Compound

/-- Finds the number of moles of a specific compound in a list of compounds --/
def findMoles (compounds : List Compound) (name : String) : ℚ :=
  match compounds.find? (fun c => c.name = name) with
  | some compound => compound.moles
  | none => 0

/-- The chemical reaction --/
def reaction : Reaction :=
  { reactants := [
      { name := "NH4Cl", moles := 1 },
      { name := "NaOH", moles := 1 }
    ],
    products := [
      { name := "NH4OH", moles := 1 },
      { name := "NaCl", moles := 1 }
    ]
  }

theorem ammonium_hydroxide_formation :
  findMoles reaction.products "NH4OH" = 1 :=
by sorry

end ammonium_hydroxide_formation_l393_39396


namespace largest_expression_l393_39315

theorem largest_expression : 
  let a := 2 + 0 + 1 + 3
  let b := 2 * 0 + 1 + 3
  let c := 2 + 0 * 1 + 3
  let d := 2 + 0 + 1 * 3
  let e := 2 * 0 * 1 * 3
  (a ≥ b) ∧ (a ≥ c) ∧ (a ≥ d) ∧ (a ≥ e) :=
by sorry

end largest_expression_l393_39315


namespace projectile_meeting_time_l393_39363

theorem projectile_meeting_time (initial_distance : ℝ) (speed1 speed2 : ℝ) :
  initial_distance = 1182 →
  speed1 = 460 →
  speed2 = 525 →
  (initial_distance / (speed1 + speed2)) * 60 = 72 := by
  sorry

end projectile_meeting_time_l393_39363


namespace problem_statement_l393_39312

theorem problem_statement (x y : ℝ) (hx : x = 1/3) (hy : y = 3) :
  (1/4) * x^3 * y^8 = 60.75 := by
  sorry

end problem_statement_l393_39312


namespace pulley_centers_distance_l393_39392

theorem pulley_centers_distance (r₁ r₂ contact_distance : ℝ) 
  (h₁ : r₁ = 10)
  (h₂ : r₂ = 6)
  (h₃ : contact_distance = 26) :
  let center_distance := Real.sqrt ((contact_distance ^ 2) + ((r₁ - r₂) ^ 2))
  center_distance = 2 * Real.sqrt 173 := by
  sorry

end pulley_centers_distance_l393_39392


namespace ball_color_difference_l393_39353

theorem ball_color_difference (m n : ℕ) (h1 : m > n) (h2 : n > 0) :
  (m * (m - 1) + n * (n - 1) : ℚ) / ((m + n) * (m + n - 1)) = 
  (2 * m * n : ℚ) / ((m + n) * (m + n - 1)) →
  ∃ a : ℕ, a > 1 ∧ m - n = a :=
sorry

end ball_color_difference_l393_39353


namespace crayon_distribution_l393_39318

theorem crayon_distribution (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) : 
  total_crayons = 24 → num_people = 3 → crayons_per_person = total_crayons / num_people → crayons_per_person = 8 := by
  sorry

end crayon_distribution_l393_39318


namespace combined_weight_theorem_l393_39323

/-- The combined weight that Rodney, Roger, and Ron can lift -/
def combinedWeight (rodney roger ron : ℕ) : ℕ := rodney + roger + ron

/-- Theorem stating the combined weight that Rodney, Roger, and Ron can lift -/
theorem combined_weight_theorem :
  ∀ (ron : ℕ),
  let roger := 4 * ron - 7
  let rodney := 2 * roger
  rodney = 146 →
  combinedWeight rodney roger ron = 239 := by
sorry

end combined_weight_theorem_l393_39323


namespace sequence_properties_l393_39385

def sequence_a (n : ℕ) : ℝ := 2 * n - 1

def sum_S (n : ℕ) : ℝ := n^2

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → 4 * (sum_S n) = (sequence_a n + 1)^2) →
  (∀ n : ℕ, n > 0 → sequence_a n = 2 * n - 1) ∧
  (sequence_a 1 = 1) ∧
  (sum_S 20 = 400) :=
by sorry

end sequence_properties_l393_39385


namespace group_size_l393_39302

/-- The number of members in the group -/
def n : ℕ := sorry

/-- The total collection in paise -/
def total_collection : ℕ := 1369

/-- Each member contributes as many paise as there are members -/
axiom member_contribution : n = n

/-- The total collection is the product of the number of members and their contribution -/
axiom total_collection_eq : n * n = total_collection

theorem group_size : n = 37 := by sorry

end group_size_l393_39302


namespace max_ellipse_area_in_rectangle_l393_39358

/-- The maximum area of an ellipse inside a rectangle -/
theorem max_ellipse_area_in_rectangle (π : ℝ) (rectangle_length rectangle_width : ℝ) :
  rectangle_length = 18 ∧ rectangle_width = 14 →
  let semi_major_axis := rectangle_length / 2
  let semi_minor_axis := rectangle_width / 2
  let max_area := π * semi_major_axis * semi_minor_axis
  max_area = π * 63 :=
by sorry

end max_ellipse_area_in_rectangle_l393_39358


namespace not_perfect_squares_l393_39334

theorem not_perfect_squares : 
  ¬(∃ n : ℕ, n^2 = 12345678) ∧ 
  ¬(∃ n : ℕ, n^2 = 987654) ∧ 
  ¬(∃ n : ℕ, n^2 = 1234560) ∧ 
  ¬(∃ n : ℕ, n^2 = 98765445) := by
sorry

end not_perfect_squares_l393_39334


namespace a_minus_b_value_l393_39398

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 4) (h2 : b^2 = 9) (h3 : a/b > 0) :
  a - b = 1 ∨ a - b = -1 := by
  sorry

end a_minus_b_value_l393_39398


namespace triangle_with_pi_power_sum_is_acute_l393_39378

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

-- Define the property of being an acute triangle
def IsAcute (t : Triangle) : Prop :=
  t.a^2 + t.b^2 > t.c^2

-- State the theorem
theorem triangle_with_pi_power_sum_is_acute (t : Triangle) 
  (h : t.a^Real.pi + t.b^Real.pi = t.c^Real.pi) : IsAcute t := by
  sorry

end triangle_with_pi_power_sum_is_acute_l393_39378


namespace consecutive_pages_sum_l393_39335

theorem consecutive_pages_sum (x : ℕ) : x > 0 ∧ x + (x + 1) = 137 → x + 1 = 69 := by
  sorry

end consecutive_pages_sum_l393_39335


namespace x_value_l393_39311

theorem x_value : ∃ (x : ℝ), x > 0 ∧ Real.sqrt ((4 * x) / 3) = x ∧ x = 4 / 3 := by
  sorry

end x_value_l393_39311


namespace sons_ages_l393_39326

theorem sons_ages (x y : ℕ+) (h1 : x < y) (h2 : y ≤ 4) 
  (h3 : ∃ (a b : ℕ+), a ≠ x ∧ b ≠ y ∧ a * b = x * y)
  (h4 : x ≠ y → (x = 1 ∧ y = 4)) :
  x = 1 ∧ y = 4 := by
sorry

end sons_ages_l393_39326


namespace parallel_lines_b_value_l393_39374

theorem parallel_lines_b_value (b : ℝ) : 
  (∃ (m₁ m₂ : ℝ), (∀ x y : ℝ, 3 * y - 4 * b = 9 * x ↔ y = m₁ * x + (4 * b / 3)) ∧
                   (∀ x y : ℝ, y - 2 = (b + 10) * x ↔ y = m₂ * x + 2) ∧
                   m₁ = m₂) →
  b = -7 := by
sorry

end parallel_lines_b_value_l393_39374


namespace percentOutsideC_eq_61_11_l393_39367

def gradeScale : List (Char × (Int × Int)) := [
  ('A', (94, 100)),
  ('B', (86, 93)),
  ('C', (76, 85)),
  ('D', (65, 75)),
  ('F', (0, 64))
]

def scores : List Int := [98, 73, 55, 100, 76, 93, 88, 72, 77, 65, 82, 79, 68, 85, 91, 56, 81, 89]

def isOutsideC (score : Int) : Bool :=
  score < 76 || score > 85

def countOutsideC : Nat :=
  scores.filter isOutsideC |>.length

theorem percentOutsideC_eq_61_11 :
  (countOutsideC : ℚ) / scores.length * 100 = 61.11 := by
  sorry

end percentOutsideC_eq_61_11_l393_39367


namespace stratified_sampling_sizes_l393_39305

/-- Represents the income groups in the community -/
inductive IncomeGroup
  | High
  | Middle
  | Low

/-- Calculates the sample size for a given income group -/
def sampleSize (totalPopulation : ℕ) (groupPopulation : ℕ) (totalSample : ℕ) : ℕ :=
  (groupPopulation * totalSample) / totalPopulation

/-- Theorem stating the correct sample sizes for each income group -/
theorem stratified_sampling_sizes :
  let totalPopulation := 600
  let highIncome := 230
  let middleIncome := 290
  let lowIncome := 80
  let totalSample := 60
  (sampleSize totalPopulation highIncome totalSample = 23) ∧
  (sampleSize totalPopulation middleIncome totalSample = 29) ∧
  (sampleSize totalPopulation lowIncome totalSample = 8) :=
by
  sorry

#check stratified_sampling_sizes

end stratified_sampling_sizes_l393_39305


namespace expected_rainfall_theorem_l393_39360

/-- Represents the daily rainfall probabilities and amounts -/
structure DailyRainfall where
  sun_prob : ℝ
  light_rain_prob : ℝ
  heavy_rain_prob : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculates the expected total rainfall over a given number of days -/
def expected_total_rainfall (daily : DailyRainfall) (days : ℕ) : ℝ :=
  days * (daily.light_rain_prob * daily.light_rain_amount + daily.heavy_rain_prob * daily.heavy_rain_amount)

/-- The main theorem stating the expected total rainfall over 10 days -/
theorem expected_rainfall_theorem (daily : DailyRainfall)
  (h1 : daily.sun_prob = 0.5)
  (h2 : daily.light_rain_prob = 0.3)
  (h3 : daily.heavy_rain_prob = 0.2)
  (h4 : daily.light_rain_amount = 3)
  (h5 : daily.heavy_rain_amount = 6)
  : expected_total_rainfall daily 10 = 21 := by
  sorry

end expected_rainfall_theorem_l393_39360


namespace solution_characterization_l393_39322

/-- The set of solutions to the system of equations:
    a² + b = c²
    b² + c = a²
    c² + a = b²
-/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (0, 1, -1), (-1, 0, 1), (1, -1, 0)}

/-- A triplet (a, b, c) satisfies the system of equations -/
def SatisfiesSystem (t : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := t
  a^2 + b = c^2 ∧ b^2 + c = a^2 ∧ c^2 + a = b^2

theorem solution_characterization :
  ∀ t : ℝ × ℝ × ℝ, SatisfiesSystem t ↔ t ∈ SolutionSet := by
  sorry


end solution_characterization_l393_39322


namespace nabla_equation_solution_l393_39317

-- Define the ∇ operation
def nabla (a b : ℤ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem nabla_equation_solution :
  ∀ b : ℤ, b ≠ 3 → nabla 3 b = -4 → b = 5 := by
  sorry

end nabla_equation_solution_l393_39317


namespace last_digit_sum_l393_39320

def is_valid_pair (a b : Nat) : Prop :=
  (a * 10 + b) % 17 = 0 ∨ (a * 10 + b) % 23 = 0

def valid_sequence (s : List Nat) : Prop :=
  s.length = 2000 ∧
  s.head? = some 3 ∧
  ∀ i, i < 1999 → is_valid_pair (s.get! i) (s.get! (i + 1))

theorem last_digit_sum (s : List Nat) (a b : Nat) :
  valid_sequence s →
  (s.getLast? = some a ∨ s.getLast? = some b) →
  a + b = 7 := by
sorry

end last_digit_sum_l393_39320


namespace probabilities_sum_to_one_l393_39344

def p₁ : ℝ := 0.22
def p₂ : ℝ := 0.31
def p₃ : ℝ := 0.47

theorem probabilities_sum_to_one : p₁ + p₂ + p₃ = 1 := by
  sorry

end probabilities_sum_to_one_l393_39344


namespace chicken_farm_theorem_l393_39329

/-- Represents the chicken farm problem -/
structure ChickenFarm where
  totalChicks : ℕ
  costA : ℕ
  costB : ℕ
  survivalRateA : ℚ
  survivalRateB : ℚ

/-- The solution to the chicken farm problem -/
def chickenFarmSolution (farm : ChickenFarm) : Prop :=
  -- Total number of chicks is 2000
  farm.totalChicks = 2000 ∧
  -- Cost of type A chick is 2 yuan
  farm.costA = 2 ∧
  -- Cost of type B chick is 3 yuan
  farm.costB = 3 ∧
  -- Survival rate of type A chicks is 94%
  farm.survivalRateA = 94/100 ∧
  -- Survival rate of type B chicks is 99%
  farm.survivalRateB = 99/100 ∧
  -- Question 1
  (∃ (x y : ℕ), x + y = farm.totalChicks ∧ 
    farm.costA * x + farm.costB * y = 4500 ∧
    x = 1500 ∧ y = 500) ∧
  -- Question 2
  (∃ (x : ℕ), x ≥ 1300 ∧
    ∀ (y : ℕ), y + x = farm.totalChicks →
      farm.costA * x + farm.costB * y ≤ 4700) ∧
  -- Question 3
  (∃ (x y : ℕ), x + y = farm.totalChicks ∧
    farm.survivalRateA * x + farm.survivalRateB * y ≥ 96/100 * farm.totalChicks ∧
    x = 1200 ∧ y = 800 ∧
    farm.costA * x + farm.costB * y = 4800 ∧
    ∀ (x' y' : ℕ), x' + y' = farm.totalChicks →
      farm.survivalRateA * x' + farm.survivalRateB * y' ≥ 96/100 * farm.totalChicks →
      farm.costA * x' + farm.costB * y' ≥ 4800)

theorem chicken_farm_theorem (farm : ChickenFarm) : chickenFarmSolution farm := by
  sorry

end chicken_farm_theorem_l393_39329


namespace wedge_volume_l393_39380

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (θ : ℝ) : 
  d = 12 →  -- diameter of the log
  θ = π/4 →  -- angle between the two cuts (45° in radians)
  (1/2) * π * (d/2)^2 * d = 216 * π := by
  sorry

end wedge_volume_l393_39380


namespace magic_square_d_plus_e_l393_39382

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  sum_eq_row1 : sum = 30 + e + 24
  sum_eq_row2 : sum = 15 + c + d
  sum_eq_row3 : sum = a + 28 + b
  sum_eq_col1 : sum = 30 + 15 + a
  sum_eq_col2 : sum = e + c + 28
  sum_eq_col3 : sum = 24 + d + b
  sum_eq_diag1 : sum = 30 + c + b
  sum_eq_diag2 : sum = a + c + 24

theorem magic_square_d_plus_e (sq : MagicSquare) : sq.d + sq.e = 48 := by
  sorry

end magic_square_d_plus_e_l393_39382


namespace large_cube_volume_l393_39300

theorem large_cube_volume (small_cube_surface_area : ℝ) (num_small_cubes : ℕ) :
  small_cube_surface_area = 96 →
  num_small_cubes = 8 →
  let small_cube_side := Real.sqrt (small_cube_surface_area / 6)
  let large_cube_side := small_cube_side * 2
  large_cube_side ^ 3 = 512 :=
by
  sorry

end large_cube_volume_l393_39300


namespace impossible_heart_and_club_l393_39357

-- Define a standard deck of cards
def StandardDeck : Type := Fin 52

-- Define suits
inductive Suit
| Hearts
| Clubs
| Diamonds
| Spades

-- Define a function to get the suit of a card
def getSuit : StandardDeck → Suit := sorry

-- Theorem: The probability of drawing a card that is both Hearts and Clubs is 0
theorem impossible_heart_and_club (card : StandardDeck) : 
  ¬(getSuit card = Suit.Hearts ∧ getSuit card = Suit.Clubs) := by
  sorry

end impossible_heart_and_club_l393_39357


namespace lillian_candy_count_l393_39325

theorem lillian_candy_count (initial_candies : ℕ) (additional_candies : ℕ) : 
  initial_candies = 88 → additional_candies = 5 → initial_candies + additional_candies = 93 := by
  sorry

end lillian_candy_count_l393_39325


namespace quadratic_minimum_quadratic_minimum_achieved_l393_39348

theorem quadratic_minimum (x : ℝ) : x^2 - 4*x - 2019 ≥ -2023 := by
  sorry

theorem quadratic_minimum_achieved : ∃ x : ℝ, x^2 - 4*x - 2019 = -2023 := by
  sorry

end quadratic_minimum_quadratic_minimum_achieved_l393_39348


namespace triangle_properties_l393_39328

open Real

theorem triangle_properties (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C ∧
  1 + (tan C / tan B) = 2 * a / b ∧
  cos (B + π/6) = 1/3 ∧
  (a + b)^2 - c^2 = 4 →
  C = π/3 ∧ 
  sin A = (2 * sqrt 6 + 1) / 6 ∧
  ∀ x y, x > 0 ∧ y > 0 ∧ (x + y)^2 - c^2 = 4 → 3*x + y ≥ 4 := by sorry

end triangle_properties_l393_39328


namespace min_y_over_x_on_ellipse_l393_39393

/-- The minimum value of y/x for points on the ellipse 4(x-2)^2 + y^2 = 4 -/
theorem min_y_over_x_on_ellipse :
  ∃ (min : ℝ), min = -2 * Real.sqrt 3 / 3 ∧
  ∀ (x y : ℝ), 4 * (x - 2)^2 + y^2 = 4 →
  y / x ≥ min :=
by sorry

end min_y_over_x_on_ellipse_l393_39393
