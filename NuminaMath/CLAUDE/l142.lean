import Mathlib

namespace square_root_three_squared_l142_14229

theorem square_root_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end square_root_three_squared_l142_14229


namespace tomato_harvest_ratio_l142_14247

/-- Proves that the ratio of tomatoes harvested on Wednesday to Thursday is 2:1 --/
theorem tomato_harvest_ratio :
  ∀ (thursday_harvest : ℕ),
  400 + thursday_harvest + (700 + 700) = 2000 →
  (400 : ℚ) / thursday_harvest = 2 / 1 :=
by
  sorry

end tomato_harvest_ratio_l142_14247


namespace odd_function_property_l142_14284

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the theorem
theorem odd_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h1 : OddFunction f) 
  (h2 : a = 2) 
  (h3 : f (-2) = 11) : 
  f a = -11 := by
  sorry

end odd_function_property_l142_14284


namespace fourth_week_sales_l142_14223

def chocolate_sales (week1 week2 week3 week4 week5 : ℕ) : Prop :=
  let total := week1 + week2 + week3 + week4 + week5
  (total : ℚ) / 5 = 71

theorem fourth_week_sales :
  ∀ week4 : ℕ,
  chocolate_sales 75 67 75 week4 68 →
  week4 = 70 := by
sorry

end fourth_week_sales_l142_14223


namespace shopkeeper_red_cards_l142_14295

/-- Represents the number of decks for each type of playing cards --/
structure DeckCounts where
  standard : Nat
  special : Nat
  custom : Nat

/-- Represents the number of red cards in each type of deck --/
structure RedCardCounts where
  standard : Nat
  special : Nat
  custom : Nat

/-- Calculates the total number of red cards given the deck counts and red card counts --/
def totalRedCards (decks : DeckCounts) (redCards : RedCardCounts) : Nat :=
  decks.standard * redCards.standard +
  decks.special * redCards.special +
  decks.custom * redCards.custom

/-- Theorem stating that the shopkeeper has 178 red cards in total --/
theorem shopkeeper_red_cards :
  let decks : DeckCounts := { standard := 3, special := 2, custom := 2 }
  let redCards : RedCardCounts := { standard := 26, special := 30, custom := 20 }
  totalRedCards decks redCards = 178 := by
  sorry

end shopkeeper_red_cards_l142_14295


namespace different_tens_digit_probability_l142_14243

/-- The number of integers to be chosen -/
def n : ℕ := 6

/-- The lower bound of the range (inclusive) -/
def lower_bound : ℕ := 10

/-- The upper bound of the range (inclusive) -/
def upper_bound : ℕ := 79

/-- The total number of integers in the range -/
def total_numbers : ℕ := upper_bound - lower_bound + 1

/-- The number of different tens digits in the range -/
def tens_digits : ℕ := 7

/-- The probability of choosing n different integers from the range
    such that they each have a different tens digit -/
def probability : ℚ := 1750 / 2980131

theorem different_tens_digit_probability :
  probability = (tens_digits.choose n * (10 ^ n : ℕ)) / total_numbers.choose n :=
sorry

end different_tens_digit_probability_l142_14243


namespace class_average_problem_l142_14270

/-- Given a class of 50 students with an overall average of 92 and the first 30 students
    having an average of 90, the average of the remaining 20 students is 95. -/
theorem class_average_problem :
  ∀ (total_score first_group_score last_group_score : ℝ),
  (50 : ℝ) * 92 = total_score →
  (30 : ℝ) * 90 = first_group_score →
  total_score = first_group_score + last_group_score →
  last_group_score / (20 : ℝ) = 95 := by
sorry

end class_average_problem_l142_14270


namespace problem_solution_l142_14272

theorem problem_solution (a b : ℝ) : 
  a = 105 ∧ a^3 = 21 * 49 * 45 * b → b = 12.5 := by sorry

end problem_solution_l142_14272


namespace job_completion_time_l142_14224

/-- If a group can complete a job in 20 days, twice the group can do half the job in 5 days -/
theorem job_completion_time (people : ℕ) (work : ℝ) : 
  (people * work = 20) → (2 * people) * (work / 2) = 5 :=
by
  sorry

end job_completion_time_l142_14224


namespace valid_paths_count_l142_14240

-- Define the grid dimensions
def rows : Nat := 5
def cols : Nat := 7

-- Define the blocked paths
def blocked_path1 : (Nat × Nat) × (Nat × Nat) := ((4, 2), (5, 2))
def blocked_path2 : (Nat × Nat) × (Nat × Nat) := ((2, 7), (3, 7))

-- Define a function to calculate valid paths
def valid_paths (r : Nat) (c : Nat) (blocked1 blocked2 : (Nat × Nat) × (Nat × Nat)) : Nat :=
  sorry

-- Theorem statement
theorem valid_paths_count : 
  valid_paths rows cols blocked_path1 blocked_path2 = 546 := by sorry

end valid_paths_count_l142_14240


namespace product_sum_and_reciprocals_bound_l142_14255

theorem product_sum_and_reciprocals_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 ∧
  ∀ ε > 0, ∃ a' b' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    (a' + b' + c') * (1/a' + 1/b' + 1/c') < 9 + ε :=
by sorry

end product_sum_and_reciprocals_bound_l142_14255


namespace bicycle_price_calculation_l142_14282

theorem bicycle_price_calculation (initial_cost : ℝ) (profit1 : ℝ) (profit2 : ℝ) : 
  initial_cost = 150 ∧ profit1 = 0.20 ∧ profit2 = 0.25 →
  (initial_cost * (1 + profit1)) * (1 + profit2) = 225 :=
by
  sorry

end bicycle_price_calculation_l142_14282


namespace modulo_nine_sum_product_l142_14207

theorem modulo_nine_sum_product : 
  (2 * (1 + 222 + 3333 + 44444 + 555555 + 6666666 + 77777777 + 888888888)) % 9 = 1 := by
  sorry

end modulo_nine_sum_product_l142_14207


namespace equation_solution_for_all_y_l142_14276

theorem equation_solution_for_all_y :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0 :=
by
  -- The proof goes here
  sorry

end equation_solution_for_all_y_l142_14276


namespace line_intersects_circle_shortest_chord_l142_14227

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1 - 2 * k

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 7 = 0

-- Theorem 1: Line l always intersects circle C
theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, line_l k x y ∧ circle_C x y :=
sorry

-- Theorem 2: The line x + 2y - 4 = 0 produces the shortest chord
theorem shortest_chord :
  ∀ k : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    line_l k x₁ y₁ ∧ circle_C x₁ y₁ ∧
    line_l k x₂ y₂ ∧ circle_C x₂ y₂) →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧
    x₁ + 2*y₁ - 4 = 0 ∧ circle_C x₁ y₁ ∧
    x₂ + 2*y₂ - 4 = 0 ∧ circle_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 ≤ (x₁ - x₂)^2 + (y₁ - y₂)^2) :=
sorry

end line_intersects_circle_shortest_chord_l142_14227


namespace factorial_fraction_simplification_l142_14220

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end factorial_fraction_simplification_l142_14220


namespace min_max_cubic_minus_xy_squared_l142_14241

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := |x^3 - x*y^2|

/-- The theorem statement -/
theorem min_max_cubic_minus_xy_squared :
  (∃ (m : ℝ), ∀ (y : ℝ), m ≤ (⨆ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2), f x y)) ∧
  (∀ (m : ℝ), (∀ (y : ℝ), m ≤ (⨆ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2), f x y)) → 8 ≤ m) :=
sorry

end min_max_cubic_minus_xy_squared_l142_14241


namespace xyz_value_l142_14297

theorem xyz_value (x y z : ℝ) : 
  4 * (Real.sqrt x + Real.sqrt (y - 1) + Real.sqrt (z - 2)) = x + y + z + 9 →
  x * y * z = 120 := by
sorry

end xyz_value_l142_14297


namespace train_length_calculation_l142_14277

-- Define the given values
def train_speed : Real := 63  -- km/hr
def man_speed : Real := 3     -- km/hr
def crossing_time : Real := 29.997600191984642  -- seconds

-- Define the theorem
theorem train_length_calculation :
  let relative_speed := (train_speed - man_speed) * 1000 / 3600  -- Convert to m/s
  let train_length := relative_speed * crossing_time
  ∃ ε > 0, abs (train_length - 500) < ε :=
by sorry

end train_length_calculation_l142_14277


namespace equation_solutions_l142_14222

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 10 ∧ x₂ = 3 - Real.sqrt 10 ∧
    x₁^2 - 6*x₁ = 1 ∧ x₂^2 - 6*x₂ = 1) ∧
  (∃ x₃ x₄ : ℝ, x₃ = 2/3 ∧ x₄ = -4 ∧
    (x₃ - 3)^2 = (2*x₃ + 1)^2 ∧ (x₄ - 3)^2 = (2*x₄ + 1)^2) :=
by sorry

end equation_solutions_l142_14222


namespace sqrt_meaningful_iff_l142_14206

theorem sqrt_meaningful_iff (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 1) ↔ x ≥ -1 := by sorry

end sqrt_meaningful_iff_l142_14206


namespace polynomial_factor_coefficients_l142_14231

theorem polynomial_factor_coefficients :
  ∀ (a b : ℚ),
  (∃ (c d : ℚ), ∀ (x : ℚ),
    a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 9 =
    (4 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 4.5)) →
  a = 11 ∧ b = -121/4 := by
  sorry

end polynomial_factor_coefficients_l142_14231


namespace more_wins_probability_correct_l142_14239

/-- The probability of winning, losing, or tying a single match -/
def match_probability : ℚ := 1/3

/-- The number of matches played -/
def num_matches : ℕ := 6

/-- The probability of finishing with more wins than losses -/
def more_wins_probability : ℚ := 98/243

theorem more_wins_probability_correct :
  let outcomes := 3^num_matches
  let equal_wins_losses := (num_matches.choose (num_matches/2))
                         + (num_matches.choose ((num_matches-2)/2)) * (num_matches.choose 2)
                         + (num_matches.choose ((num_matches-4)/2)) * (num_matches.choose 4)
                         + 1
  (1 - equal_wins_losses / outcomes) / 2 = more_wins_probability :=
sorry

end more_wins_probability_correct_l142_14239


namespace unique_three_digit_factorial_sum_l142_14236

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumOfDigitFactorials (n : ℕ) : ℕ :=
  (n.digits 10).map factorial |>.sum

def hasDigit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ n.digits 10

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = sumOfDigitFactorials n ∧ hasDigit n 3 :=
by
  sorry

end unique_three_digit_factorial_sum_l142_14236


namespace cow_chicken_problem_l142_14293

theorem cow_chicken_problem (cows chickens : ℕ) : 
  4 * cows + 2 * chickens = 14 + 2 * (cows + chickens) → cows = 7 := by
  sorry

end cow_chicken_problem_l142_14293


namespace no_integer_points_on_circle_l142_14211

theorem no_integer_points_on_circle : 
  ∀ x : ℤ, (x - 3)^2 + (3*x + 1)^2 > 16 :=
by sorry

end no_integer_points_on_circle_l142_14211


namespace ivan_number_properties_l142_14260

def sum_of_digits (n : ℕ) : ℕ := sorry

def num_digits (n : ℕ) : ℕ := sorry

theorem ivan_number_properties (n : ℕ) (h : n > 0) :
  let x := (sum_of_digits n)^2
  (num_digits n ≤ 3 → x < 730) ∧
  (num_digits n = 4 → x < n) ∧
  (num_digits n ≥ 5 → x < n) ∧
  (∀ m : ℕ, m > 0 → (sum_of_digits x)^2 = m → (m = 1 ∨ m = 81)) :=
by sorry

#check ivan_number_properties

end ivan_number_properties_l142_14260


namespace append_two_digit_numbers_formula_l142_14205

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≥ 0 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- Appends one two-digit number after another -/
def append_two_digit_numbers (n1 n2 : TwoDigitNumber) : Nat :=
  1000 * n1.tens + 100 * n1.units + 10 * n2.tens + n2.units

/-- Theorem: Appending two two-digit numbers results in the expected formula -/
theorem append_two_digit_numbers_formula (n1 n2 : TwoDigitNumber) :
  append_two_digit_numbers n1 n2 = 1000 * n1.tens + 100 * n1.units + 10 * n2.tens + n2.units :=
by sorry

end append_two_digit_numbers_formula_l142_14205


namespace chord_slope_l142_14234

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 20 + y^2 / 16 = 1

-- Define the point P
def P : ℝ × ℝ := (3, -2)

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := y - P.2 = m * (x - P.1)

-- Define the midpoint property
def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem chord_slope :
  ∃ (A B : ℝ × ℝ) (m : ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    line_l m A.1 A.2 ∧
    line_l m B.1 B.2 ∧
    is_midpoint P A B ∧
    m = 6/5 := by sorry

end chord_slope_l142_14234


namespace eighth_equation_sum_l142_14230

theorem eighth_equation_sum (a t : ℝ) (ha : a > 0) (ht : t > 0) :
  (8 + a / t).sqrt = 8 * (a / t).sqrt → a + t = 71 := by
  sorry

end eighth_equation_sum_l142_14230


namespace count_1973_in_I_1000000_l142_14267

-- Define the sequence type
def Sequence := List Nat

-- Define the initial sequence
def I₀ : Sequence := [1, 1]

-- Define the rule for generating the next sequence
def nextSequence (I : Sequence) : Sequence :=
  sorry

-- Define the n-th sequence
def Iₙ (n : Nat) : Sequence :=
  sorry

-- Define the count of a number in a sequence
def count (m : Nat) (I : Sequence) : Nat :=
  sorry

-- Euler's totient function
def φ (n : Nat) : Nat :=
  sorry

-- The main theorem
theorem count_1973_in_I_1000000 :
  count 1973 (Iₙ 1000000) = φ 1973 :=
sorry

end count_1973_in_I_1000000_l142_14267


namespace inequality_not_always_correct_l142_14204

theorem inequality_not_always_correct 
  (x y z : ℝ) (k : ℤ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z ≠ 0) (hk : k ≠ 0) :
  ¬ (∀ (x y z : ℝ) (k : ℤ), x > 0 → y > 0 → x > y → z ≠ 0 → k ≠ 0 → x / z^k > y / z^k) :=
by sorry

end inequality_not_always_correct_l142_14204


namespace all_statements_false_l142_14289

-- Define the concepts of lines and planes
variable (Line Plane : Type)

-- Define the concept of parallelism between lines
variable (parallel_lines : Line → Line → Prop)

-- Define the concept of parallelism between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the concept of perpendicularity between lines
variable (perpendicular : Line → Line → Prop)

-- Define the concept of a line having no common points with another line
variable (no_common_points : Line → Line → Prop)

-- Define the concept of a line having no common points with countless lines in a plane
variable (no_common_points_with_plane_lines : Line → Plane → Prop)

theorem all_statements_false :
  (∀ (l₁ l₂ : Line) (p : Plane), parallel_line_plane l₁ p → parallel_line_plane l₂ p → parallel_lines l₁ l₂) = False ∧
  (∀ (l₁ l₂ : Line), no_common_points l₁ l₂ → parallel_lines l₁ l₂) = False ∧
  (∀ (l₁ l₂ l₃ : Line), perpendicular l₁ l₃ → perpendicular l₂ l₃ → parallel_lines l₁ l₂) = False ∧
  (∀ (l : Line) (p : Plane), no_common_points_with_plane_lines l p → parallel_line_plane l p) = False :=
sorry

end all_statements_false_l142_14289


namespace square_area_proof_l142_14280

/-- Given a square with side length equal to both 5x - 21 and 29 - 2x,
    prove that its area is 10609/49 square meters. -/
theorem square_area_proof (x : ℝ) (h : 5 * x - 21 = 29 - 2 * x) :
  (5 * x - 21) ^ 2 = 10609 / 49 := by
  sorry

end square_area_proof_l142_14280


namespace inscribed_tetrahedron_volume_ratio_l142_14201

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Given two regular tetrahedra where one is inscribed inside the other
    such that its vertices are at the midpoints of the edges of the larger tetrahedron,
    the ratio of their volumes is 1/8 -/
theorem inscribed_tetrahedron_volume_ratio
  (large : RegularTetrahedron) (small : RegularTetrahedron)
  (h : small.sideLength = large.sideLength / 2) :
  (small.sideLength ^ 3) / (large.sideLength ^ 3) = 1 / 8 := by
  sorry

#check inscribed_tetrahedron_volume_ratio

end inscribed_tetrahedron_volume_ratio_l142_14201


namespace factorial_difference_quotient_l142_14259

theorem factorial_difference_quotient : (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end factorial_difference_quotient_l142_14259


namespace symmetric_decreasing_property_l142_14228

/-- A function f: ℝ → ℝ that is decreasing on (4, +∞) and symmetric about x = 4 -/
def SymmetricDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 4 ∧ y > x → f y < f x) ∧
  (∀ x, f (4 + x) = f (4 - x))

/-- Given a symmetric decreasing function f, prove that f(3) > f(6) -/
theorem symmetric_decreasing_property (f : ℝ → ℝ) 
  (h : SymmetricDecreasingFunction f) : f 3 > f 6 := by
  sorry

end symmetric_decreasing_property_l142_14228


namespace triangle_side_b_value_l142_14218

theorem triangle_side_b_value (A B C : ℝ) (a b c : ℝ) :
  c = Real.sqrt 6 →
  Real.cos C = -(1/4 : ℝ) →
  Real.sin A = 2 * Real.sin B →
  b = 1 :=
by sorry

end triangle_side_b_value_l142_14218


namespace equation_solution_l142_14216

theorem equation_solution (x : ℚ) :
  x ≠ 2/3 →
  ((3*x + 2) / (3*x^2 + 4*x - 4) = 3*x / (3*x - 2)) ↔ (x = 1/3 ∨ x = -2) :=
by sorry

end equation_solution_l142_14216


namespace square_difference_l142_14283

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end square_difference_l142_14283


namespace isosceles_base_angle_l142_14265

-- Define an isosceles triangle with a 30° vertex angle
def IsoscelesTriangle (α β γ : ℝ) : Prop :=
  α = 30 ∧ β = γ ∧ α + β + γ = 180

-- Theorem: In an isosceles triangle with a 30° vertex angle, each base angle is 75°
theorem isosceles_base_angle (α β γ : ℝ) (h : IsoscelesTriangle α β γ) : β = 75 := by
  sorry

end isosceles_base_angle_l142_14265


namespace cayley_hamilton_for_A_l142_14208

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 2, 3],
    ![2, 1, 2],
    ![3, 2, 1]]

theorem cayley_hamilton_for_A :
  A^3 + (-8 : ℤ) • A^2 + (-2 : ℤ) • A + (-8 : ℤ) • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := by
  sorry

end cayley_hamilton_for_A_l142_14208


namespace sum_first_six_primes_mod_seventh_prime_l142_14249

theorem sum_first_six_primes_mod_seventh_prime : 
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 := by
  sorry

end sum_first_six_primes_mod_seventh_prime_l142_14249


namespace river_road_cars_l142_14235

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 13 →
  buses = cars - 60 →
  cars = 65 := by
sorry

end river_road_cars_l142_14235


namespace additional_lawn_to_mow_l142_14281

/-- The problem of calculating additional square feet to mow -/
theorem additional_lawn_to_mow 
  (rate : ℚ) 
  (book_cost : ℚ) 
  (lawns_mowed : ℕ) 
  (lawn_length : ℕ) 
  (lawn_width : ℕ) : 
  rate = 1/10 → 
  book_cost = 150 → 
  lawns_mowed = 3 → 
  lawn_length = 20 → 
  lawn_width = 15 → 
  (book_cost - lawns_mowed * lawn_length * lawn_width * rate) / rate = 600 := by
  sorry

#check additional_lawn_to_mow

end additional_lawn_to_mow_l142_14281


namespace problem_solution_l142_14254

theorem problem_solution : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end problem_solution_l142_14254


namespace cody_money_l142_14271

def final_money (initial : ℕ) (birthday : ℕ) (game_cost : ℕ) : ℕ :=
  initial + birthday - game_cost

theorem cody_money : final_money 45 9 19 = 35 := by
  sorry

end cody_money_l142_14271


namespace exam_score_deviation_l142_14263

/-- Given an exam with mean score 74 and standard deviation σ, 
    prove that 58 is 2 standard deviations below the mean. -/
theorem exam_score_deviation :
  ∀ σ : ℝ,
  74 + 3 * σ = 98 →
  74 - 2 * σ = 58 :=
by
  sorry

end exam_score_deviation_l142_14263


namespace ant_path_impossibility_l142_14268

/-- Represents a vertex of a cube --/
inductive Vertex
| V1 | V2 | V3 | V4 | V5 | V6 | V7 | V8

/-- Represents the label of a vertex (+1 or -1) --/
def vertexLabel (v : Vertex) : Int :=
  match v with
  | Vertex.V1 | Vertex.V3 | Vertex.V6 | Vertex.V8 => 1
  | Vertex.V2 | Vertex.V4 | Vertex.V5 | Vertex.V7 => -1

/-- Represents a path of an ant on the cube --/
def AntPath := List Vertex

/-- Checks if the path is valid (no backtracking) --/
def isValidPath (path : AntPath) : Prop :=
  sorry

/-- Counts the number of visits to each vertex --/
def countVisits (path : AntPath) : Vertex → Nat :=
  sorry

/-- The main theorem to prove --/
theorem ant_path_impossibility :
  ¬ ∃ (path : AntPath),
    isValidPath path ∧
    (∃ (v : Vertex),
      countVisits path v = 25 ∧
      ∀ (w : Vertex), w ≠ v → countVisits path w = 20) :=
sorry

end ant_path_impossibility_l142_14268


namespace certain_number_multiplied_by_p_l142_14225

theorem certain_number_multiplied_by_p (x : ℕ+) (p : ℕ) (n : ℕ) : 
  Nat.Prime p → 
  (x : ℕ) / (n * p) = 2 → 
  x ≥ 48 → 
  (∀ y : ℕ+, y < x → (y : ℕ) / (n * p) ≠ 2) →
  n = 12 := by sorry

end certain_number_multiplied_by_p_l142_14225


namespace certain_number_solution_l142_14273

theorem certain_number_solution : 
  ∃ x : ℝ, (0.02^2 + 0.52^2 + x^2) = 100 * (0.002^2 + 0.052^2 + 0.0035^2) ∧ x = 0.035 := by
  sorry

end certain_number_solution_l142_14273


namespace house_rent_percentage_l142_14244

def total_income : ℝ := 1000
def petrol_percentage : ℝ := 0.3
def petrol_expenditure : ℝ := 300
def house_rent : ℝ := 210

theorem house_rent_percentage : 
  (house_rent / (total_income * (1 - petrol_percentage))) * 100 = 30 := by
  sorry

end house_rent_percentage_l142_14244


namespace candy_distribution_l142_14286

theorem candy_distribution (total_candy : ℕ) (num_people : ℕ) (bags_per_person : ℕ) : 
  total_candy = 648 → num_people = 4 → bags_per_person = 8 →
  (total_candy / num_people / bags_per_person : ℕ) = 20 := by
  sorry

end candy_distribution_l142_14286


namespace max_arrangement_is_eight_l142_14296

/-- Represents a valid arrangement of balls -/
def ValidArrangement (arrangement : List Nat) : Prop :=
  (∀ n ∈ arrangement, 1 ≤ n ∧ n ≤ 9) ∧
  (5 ∈ arrangement → (arrangement.indexOf 5).pred = arrangement.indexOf 1 ∨ 
                     (arrangement.indexOf 5).succ = arrangement.indexOf 1) ∧
  (7 ∈ arrangement → (arrangement.indexOf 7).pred = arrangement.indexOf 1 ∨ 
                     (arrangement.indexOf 7).succ = arrangement.indexOf 1)

/-- The maximum number of balls that can be arranged -/
def MaxArrangement : Nat := 8

/-- Theorem stating that the maximum number of balls that can be arranged is 8 -/
theorem max_arrangement_is_eight :
  (∃ arrangement : List Nat, arrangement.length = MaxArrangement ∧ ValidArrangement arrangement) ∧
  (∀ arrangement : List Nat, arrangement.length > MaxArrangement → ¬ValidArrangement arrangement) := by
  sorry

end max_arrangement_is_eight_l142_14296


namespace baker_cakes_l142_14213

/-- Calculates the final number of cakes a baker has after selling some and buying new ones. -/
def final_cakes (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  initial - sold + bought

/-- Proves that for the given numbers, the baker ends up with 186 cakes. -/
theorem baker_cakes : final_cakes 121 105 170 = 186 := by
  sorry

end baker_cakes_l142_14213


namespace power_calculation_l142_14256

theorem power_calculation : 27^3 * 9^2 / 3^15 = (1 : ℚ) / 9 := by sorry

end power_calculation_l142_14256


namespace stating_third_number_formula_l142_14209

/-- 
Given a triangular array of positive odd numbers arranged as follows:
1
3  5
7  9  11
13 15 17 19
...
This function returns the third number from the left in the nth row.
-/
def thirdNumberInRow (n : ℕ) : ℕ :=
  n^2 - n + 5

/-- 
Theorem stating that for n ≥ 3, the third number from the left 
in the nth row of the described triangular array is n^2 - n + 5.
-/
theorem third_number_formula (n : ℕ) (h : n ≥ 3) : 
  thirdNumberInRow n = n^2 - n + 5 := by
  sorry

end stating_third_number_formula_l142_14209


namespace min_sum_of_integers_l142_14253

theorem min_sum_of_integers (m n : ℕ) : 
  m < n → 
  m > 0 → 
  n > 0 → 
  m * n = (m - 20) * (n + 23) → 
  ∀ k l : ℕ, k < l → k > 0 → l > 0 → k * l = (k - 20) * (l + 23) → m + n ≤ k + l →
  m + n = 321 := by
sorry

end min_sum_of_integers_l142_14253


namespace segments_form_quadrilateral_l142_14215

/-- A function that checks if three line segments can form a quadrilateral with a fourth segment -/
def can_form_quadrilateral (a b c d : ℝ) : Prop :=
  a + b + c > d

/-- Theorem stating that line segments of length 2, 2, 2 can form a quadrilateral with a segment of length 5 -/
theorem segments_form_quadrilateral :
  can_form_quadrilateral 2 2 2 5 := by
  sorry

end segments_form_quadrilateral_l142_14215


namespace amy_total_tickets_l142_14233

/-- Amy's initial number of tickets -/
def initial_tickets : ℕ := 33

/-- Number of tickets Amy bought additionally -/
def additional_tickets : ℕ := 21

/-- Theorem stating the total number of tickets Amy has -/
theorem amy_total_tickets : initial_tickets + additional_tickets = 54 := by
  sorry

end amy_total_tickets_l142_14233


namespace female_officers_count_l142_14264

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 204 →
  female_on_duty_ratio = 1/2 →
  female_ratio = 17/100 →
  ∃ (total_female : ℕ), total_female = 600 ∧ 
    (female_ratio * total_female : ℚ) = (female_on_duty_ratio * total_on_duty : ℚ) := by
  sorry

end female_officers_count_l142_14264


namespace scaling_transformation_maps_line_l142_14288

/-- A scaling transformation in 2D space -/
structure ScalingTransformation where
  x_scale : ℝ
  y_scale : ℝ

/-- The original line equation -/
def original_line (x y : ℝ) : Prop := x + y + 2 = 0

/-- The transformed line equation -/
def transformed_line (x y : ℝ) : Prop := 8*x + y + 8 = 0

/-- Theorem stating that the given scaling transformation maps the original line to the transformed line -/
theorem scaling_transformation_maps_line :
  ∃ (t : ScalingTransformation),
    (∀ (x y : ℝ), original_line x y ↔ transformed_line (t.x_scale * x) (t.y_scale * y)) ∧
    t.x_scale = 1/2 ∧ t.y_scale = 4 := by
  sorry

end scaling_transformation_maps_line_l142_14288


namespace circuit_length_difference_l142_14202

/-- The length of the small circuit in meters -/
def small_circuit_length : ℕ := 400

/-- The number of laps Jana runs -/
def jana_laps : ℕ := 3

/-- The number of laps Father runs -/
def father_laps : ℕ := 4

/-- The total distance Jana runs in meters -/
def jana_distance : ℕ := small_circuit_length * jana_laps

/-- The total distance Father runs in meters -/
def father_distance : ℕ := 2 * jana_distance

/-- The length of the large circuit in meters -/
def large_circuit_length : ℕ := father_distance / father_laps

theorem circuit_length_difference :
  large_circuit_length - small_circuit_length = 200 := by
  sorry

end circuit_length_difference_l142_14202


namespace greatest_x_with_lcm_l142_14219

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (lcm : ℕ), lcm = Nat.lcm x (Nat.lcm 12 18) ∧ lcm = 108) →
  x ≤ 108 ∧ ∃ (y : ℕ), y = 108 ∧ Nat.lcm y (Nat.lcm 12 18) = 108 :=
by sorry

end greatest_x_with_lcm_l142_14219


namespace fruit_difference_l142_14299

theorem fruit_difference (apples : ℕ) (peach_multiplier : ℕ) : 
  apples = 60 → peach_multiplier = 3 → 
  (peach_multiplier * apples) - apples = 120 := by
  sorry

end fruit_difference_l142_14299


namespace candy_cost_calculation_l142_14226

/-- The problem of calculating the total cost of candy -/
theorem candy_cost_calculation (cost_per_piece : ℕ) (num_gumdrops : ℕ) (total_cost : ℕ) : 
  cost_per_piece = 8 → num_gumdrops = 28 → total_cost = cost_per_piece * num_gumdrops → total_cost = 224 :=
by sorry

end candy_cost_calculation_l142_14226


namespace min_garden_cost_is_108_l142_14269

/-- Represents the cost of each flower type in dollars -/
structure FlowerCost where
  asters : ℝ
  begonias : ℝ
  cannas : ℝ
  dahlias : ℝ
  easterLilies : ℝ

/-- Represents the dimensions of each region in the flower bed -/
structure RegionDimensions where
  region1 : ℝ × ℝ
  region2 : ℝ × ℝ
  region3 : ℝ × ℝ
  region4 : ℝ × ℝ
  region5 : ℝ × ℝ

/-- Calculates the minimum cost of the garden given the flower costs and region dimensions -/
def minGardenCost (costs : FlowerCost) (dimensions : RegionDimensions) : ℝ :=
  sorry

/-- Theorem stating that the minimum cost of the garden is $108 -/
theorem min_garden_cost_is_108 (costs : FlowerCost) (dimensions : RegionDimensions) :
  costs.asters = 1 ∧ 
  costs.begonias = 1.5 ∧ 
  costs.cannas = 2 ∧ 
  costs.dahlias = 2.5 ∧ 
  costs.easterLilies = 3 ∧
  dimensions.region1 = (3, 4) ∧
  dimensions.region2 = (2, 3) ∧
  dimensions.region3 = (3, 5) ∧
  dimensions.region4 = (4, 5) ∧
  dimensions.region5 = (3, 7) →
  minGardenCost costs dimensions = 108 :=
by
  sorry

end min_garden_cost_is_108_l142_14269


namespace a_greater_than_b_l142_14200

theorem a_greater_than_b (n : ℕ) (a b : ℝ) 
  (h_n : n > 1)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_eq : a^n = a + 1)
  (h_b_eq : b^(2*n) = b + 3*a) :
  a > b :=
by sorry

end a_greater_than_b_l142_14200


namespace equation_D_is_linear_l142_14257

/-- Definition of a linear equation in two variables -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ (x y : ℝ), f x y = a * x + b * y + c

/-- The specific equation we want to prove is linear -/
def equation_D (x y : ℝ) : ℝ := 2 * x + y - 5

/-- Theorem stating that equation_D is a linear equation in two variables -/
theorem equation_D_is_linear : is_linear_equation equation_D := by
  sorry


end equation_D_is_linear_l142_14257


namespace min_value_inequality_l142_14251

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 4 / y) ≥ 9/4 := by
  sorry

end min_value_inequality_l142_14251


namespace johns_pace_l142_14238

/-- Given the conditions of a race between John and Steve, prove that John's pace during his final push was 178 / 42.5 m/s. -/
theorem johns_pace (john_initial_behind : ℝ) (steve_speed : ℝ) (john_final_ahead : ℝ) (push_duration : ℝ) :
  john_initial_behind = 15 →
  steve_speed = 3.8 →
  john_final_ahead = 2 →
  push_duration = 42.5 →
  (john_initial_behind + john_final_ahead + steve_speed * push_duration) / push_duration = 178 / 42.5 := by
  sorry

#eval (178 : ℚ) / 42.5

end johns_pace_l142_14238


namespace polynomial_roots_nature_l142_14287

def P (x : ℝ) : ℝ := x^6 - 5*x^5 + 3*x^2 - 8*x + 16

theorem polynomial_roots_nature :
  (∀ x < 0, P x > 0) ∧ 
  (∃ a b, 0 < a ∧ a < b ∧ P a * P b < 0) := by
  sorry

end polynomial_roots_nature_l142_14287


namespace ram_money_calculation_l142_14291

theorem ram_money_calculation (ram gopal krishan : ℕ) 
  (h1 : ram * 17 = gopal * 7)
  (h2 : gopal * 17 = krishan * 7)
  (h3 : krishan = 4335) : 
  ram = 735 := by
sorry

end ram_money_calculation_l142_14291


namespace system_solution_l142_14285

theorem system_solution (x y z : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x^2 + y^2 = -x + 3*y + z ∧
  y^2 + z^2 = x + 3*y - z ∧
  z^2 + x^2 = 2*x + 2*y - z →
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
by sorry

end system_solution_l142_14285


namespace errors_per_debug_session_l142_14262

theorem errors_per_debug_session 
  (total_lines : ℕ) 
  (debug_interval : ℕ) 
  (total_errors : ℕ) 
  (h1 : total_lines = 4300)
  (h2 : debug_interval = 100)
  (h3 : total_errors = 129) :
  total_errors / (total_lines / debug_interval) = 3 := by
sorry

end errors_per_debug_session_l142_14262


namespace unique_element_implies_a_equals_four_l142_14214

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + a * x + 1 = 0}

-- State the theorem
theorem unique_element_implies_a_equals_four :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ A a) → a = 4 := by sorry

end unique_element_implies_a_equals_four_l142_14214


namespace widget_sales_sum_l142_14258

def arithmetic_sequence (n : ℕ) : ℕ := 3 * n - 1

def sum_arithmetic_sequence (n : ℕ) : ℕ :=
  n * (arithmetic_sequence 1 + arithmetic_sequence n) / 2

theorem widget_sales_sum :
  sum_arithmetic_sequence 15 = 345 := by
  sorry

end widget_sales_sum_l142_14258


namespace car_dealer_sales_l142_14278

theorem car_dealer_sales (x : ℕ) (a b : ℤ) : 
  x > 0 ∧ 
  (7 : ℚ) = (x : ℚ)⁻¹ * (7 * x : ℚ) ∧ 
  (8 : ℚ) = ((x - 1) : ℚ)⁻¹ * ((7 * x - a) : ℚ) ∧ 
  (5 : ℚ) = ((x - 1) : ℚ)⁻¹ * ((7 * x - b) : ℚ) ∧ 
  (23 : ℚ) / 4 = ((x - 2) : ℚ)⁻¹ * ((7 * x - a - b) : ℚ) →
  7 * x = 42 := by
  sorry

end car_dealer_sales_l142_14278


namespace exponential_distribution_expected_value_l142_14261

/-- The expected value of an exponentially distributed random variable -/
theorem exponential_distribution_expected_value (α : ℝ) (hα : α > 0) :
  let X : ℝ → ℝ := λ x => if x ≥ 0 then α * Real.exp (-α * x) else 0
  ∫ x in Set.Ici 0, x * X x = 1 / α :=
sorry

end exponential_distribution_expected_value_l142_14261


namespace sum_equals_two_thirds_l142_14246

theorem sum_equals_two_thirds :
  let original_sum := (1:ℚ)/3 + 1/6 + 1/9 + 1/12 + 1/15 + 1/18
  let removed_terms := 1/12 + 1/15
  let remaining_sum := original_sum - removed_terms
  remaining_sum = 2/3 := by
sorry

end sum_equals_two_thirds_l142_14246


namespace arrangement_counts_l142_14203

def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3
def total_people : ℕ := number_of_boys + number_of_girls

theorem arrangement_counts :
  (∃ (arrange_four : ℕ) (arrange_two_rows : ℕ) (girls_together : ℕ) 
      (boys_not_adjacent : ℕ) (a_not_ends : ℕ) (a_not_left_b_not_right : ℕ) 
      (a_b_c_order : ℕ),
    arrange_four = 120 ∧
    arrange_two_rows = 120 ∧
    girls_together = 36 ∧
    boys_not_adjacent = 72 ∧
    a_not_ends = 72 ∧
    a_not_left_b_not_right = 78 ∧
    a_b_c_order = 20) :=
by
  sorry


end arrangement_counts_l142_14203


namespace x_squared_mod_25_l142_14290

theorem x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x ^ 2 ≡ 0 [ZMOD 25] := by
  sorry

end x_squared_mod_25_l142_14290


namespace option_D_not_suitable_for_comprehensive_survey_l142_14292

-- Define the type for survey options
inductive SurveyOption
| A -- Security check for passengers before boarding a plane
| B -- School recruiting teachers and conducting interviews for applicants
| C -- Understanding the extracurricular reading time of seventh-grade students in a school
| D -- Understanding the service life of a batch of light bulbs

-- Define a function to check if an option is suitable for a comprehensive survey
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => True
  | SurveyOption.B => True
  | SurveyOption.C => True
  | SurveyOption.D => False

-- Theorem stating that option D is not suitable for a comprehensive survey
theorem option_D_not_suitable_for_comprehensive_survey :
  ¬(isSuitableForComprehensiveSurvey SurveyOption.D) :=
by sorry

end option_D_not_suitable_for_comprehensive_survey_l142_14292


namespace davids_age_l142_14298

/-- Given the age relationships between Anna, Ben, Carla, and David, prove David's age. -/
theorem davids_age 
  (anna ben carla david : ℕ)  -- Define variables for ages
  (h1 : anna = ben - 5)       -- Anna is five years younger than Ben
  (h2 : ben = carla + 2)      -- Ben is two years older than Carla
  (h3 : david = carla + 4)    -- David is four years older than Carla
  (h4 : anna = 12)            -- Anna is 12 years old
  : david = 19 :=             -- Prove David is 19 years old
by
  sorry  -- Proof omitted

end davids_age_l142_14298


namespace vector_addition_result_l142_14275

theorem vector_addition_result :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-3, 4)
  2 • a + b = (1, 2) := by
sorry

end vector_addition_result_l142_14275


namespace counterexample_exists_l142_14217

theorem counterexample_exists : ∃ (x y : ℝ), x > y ∧ x^2 ≤ y^2 := by
  sorry

end counterexample_exists_l142_14217


namespace number_difference_problem_l142_14210

theorem number_difference_problem : ∃ (a b : ℕ), 
  a + b = 25650 ∧ 
  a % 100 = 0 ∧ 
  a / 100 = b ∧ 
  a - b = 25146 := by
sorry

end number_difference_problem_l142_14210


namespace prob_blue_or_purple_l142_14266

/-- A bag of jelly beans with different colors -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  purple : ℕ

/-- The probability of selecting either a blue or purple jelly bean -/
def bluePurpleProbability (bag : JellyBeanBag) : ℚ :=
  (bag.blue + bag.purple : ℚ) / (bag.red + bag.green + bag.yellow + bag.blue + bag.purple : ℚ)

/-- Theorem stating the probability of selecting a blue or purple jelly bean from the given bag -/
theorem prob_blue_or_purple (bag : JellyBeanBag) 
    (h : bag = { red := 7, green := 8, yellow := 9, blue := 10, purple := 4 }) : 
    bluePurpleProbability bag = 7 / 19 := by
  sorry

#eval bluePurpleProbability { red := 7, green := 8, yellow := 9, blue := 10, purple := 4 }

end prob_blue_or_purple_l142_14266


namespace paco_cookies_l142_14212

def cookies_eaten (initial : ℕ) (given : ℕ) (left : ℕ) : ℕ :=
  initial - given - left

theorem paco_cookies : cookies_eaten 36 14 12 = 10 := by
  sorry

end paco_cookies_l142_14212


namespace arithmetic_to_harmonic_progression_l142_14274

/-- Three non-zero real numbers form an arithmetic progression if and only if
    the difference between the second and first is equal to the difference between the third and second. -/
def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- Three non-zero real numbers form a harmonic progression if and only if
    the reciprocal of the middle term is the arithmetic mean of the reciprocals of the other two terms. -/
def is_harmonic_progression (a b c : ℝ) : Prop :=
  2 / b = 1 / a + 1 / c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- If three non-zero real numbers form an arithmetic progression,
    then their reciprocals form a harmonic progression. -/
theorem arithmetic_to_harmonic_progression (a b c : ℝ) :
  is_arithmetic_progression a b c → is_harmonic_progression (1/a) (1/b) (1/c) := by
  sorry

end arithmetic_to_harmonic_progression_l142_14274


namespace expression_simplification_l142_14237

theorem expression_simplification (m n x : ℝ) :
  (3 * m^2 + 2 * m * n - 5 * m^2 + 3 * m * n = -2 * m^2 + 5 * m * n) ∧
  ((x^2 + 2 * x) - 2 * (x^2 - x) = -x^2 + 4 * x) := by
  sorry

end expression_simplification_l142_14237


namespace solution_set_when_a_is_one_range_of_a_when_inequality_holds_l142_14250

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_when_inequality_holds :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, f a x > x) ↔ a ∈ Set.Ioc 0 2 := by sorry

end solution_set_when_a_is_one_range_of_a_when_inequality_holds_l142_14250


namespace projectile_max_height_l142_14232

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 30

/-- The maximum height of the projectile -/
def max_height : ℝ := 155

/-- Theorem: The maximum height of the projectile is 155 feet -/
theorem projectile_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
by
  sorry

end projectile_max_height_l142_14232


namespace football_field_area_l142_14294

-- Define the football field and fertilizer properties
def total_fertilizer : ℝ := 800
def partial_fertilizer : ℝ := 300
def partial_area : ℝ := 3600

-- Define the theorem
theorem football_field_area :
  (total_fertilizer * partial_area) / partial_fertilizer = 9600 := by
  sorry

end football_field_area_l142_14294


namespace min_value_of_sum_equality_condition_l142_14221

theorem min_value_of_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b) / c + (a + c) / d + (b + d) / a + (c + d) / b ≥ 8 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b) / c + (a + c) / d + (b + d) / a + (c + d) / b = 8 ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end min_value_of_sum_equality_condition_l142_14221


namespace smallest_sum_arithmetic_geometric_sequence_l142_14252

theorem smallest_sum_arithmetic_geometric_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →
  (∃ r : ℚ, C - B = B - A ∧ C = B * r ∧ D = C * r) →
  C = (5 : ℚ) / 3 * B →
  A + B + C + D ≥ 52 ∧ (∃ A' B' C' D' : ℤ, 
    A' > 0 ∧ B' > 0 ∧ C' > 0 ∧
    (∃ r' : ℚ, C' - B' = B' - A' ∧ C' = B' * r' ∧ D' = C' * r') ∧
    C' = (5 : ℚ) / 3 * B' ∧
    A' + B' + C' + D' = 52) := by
  sorry

end smallest_sum_arithmetic_geometric_sequence_l142_14252


namespace triangle_count_equality_l142_14242

/-- The number of non-congruent triangles with positive area and integer side lengths summing to n -/
def T (n : ℕ) : ℕ := sorry

/-- The statement to prove -/
theorem triangle_count_equality : T 2022 = T 2019 := by sorry

end triangle_count_equality_l142_14242


namespace hyperbola_asymptote_angle_l142_14248

theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (Real.arctan ((b/a) / (1 - (b/a)^2)) * 2 = π / 4) →
  a / b = Real.sqrt 2 + 1 := by
sorry

end hyperbola_asymptote_angle_l142_14248


namespace rock_collecting_contest_l142_14245

theorem rock_collecting_contest (sydney_initial conner_initial : ℕ)
  (sydney_day1 conner_day1_multiplier : ℕ)
  (sydney_day3_multiplier conner_day3 : ℕ) :
  sydney_initial = 837 →
  conner_initial = 723 →
  sydney_day1 = 4 →
  conner_day1_multiplier = 8 →
  sydney_day3_multiplier = 2 →
  conner_day3 = 27 →
  ∃ (conner_day2 : ℕ),
    sydney_initial + sydney_day1 + sydney_day3_multiplier * (conner_day1_multiplier * sydney_day1) ≤
    conner_initial + (conner_day1_multiplier * sydney_day1) + conner_day2 + conner_day3 ∧
    conner_day2 = 123 :=
by sorry

end rock_collecting_contest_l142_14245


namespace inequalities_hold_for_all_reals_l142_14279

-- Define the two quadratic functions
def f (x : ℝ) := x^2 + 6*x + 10
def g (x : ℝ) := -x^2 + x - 2

-- Theorem stating that both inequalities hold for all real numbers
theorem inequalities_hold_for_all_reals :
  (∀ x : ℝ, f x > 0) ∧ (∀ x : ℝ, g x < 0) :=
sorry

end inequalities_hold_for_all_reals_l142_14279
