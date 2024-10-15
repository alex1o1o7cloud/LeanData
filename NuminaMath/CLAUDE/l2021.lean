import Mathlib

namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2021_202150

/-- The speed of a boat in still water, given its downstream and upstream distances in one hour -/
theorem boat_speed_in_still_water 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (h1 : downstream_distance = 7) 
  (h2 : upstream_distance = 5) : 
  ∃ (boat_speed stream_speed : ℝ), 
    boat_speed + stream_speed = downstream_distance ∧ 
    boat_speed - stream_speed = upstream_distance ∧
    boat_speed = 6 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2021_202150


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2021_202100

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 16 :=
by
  -- The unique solution is z = -251/4
  use -251/4
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2021_202100


namespace NUMINAMATH_CALUDE_no_whole_number_57_times_less_l2021_202160

theorem no_whole_number_57_times_less : ¬ ∃ (N : ℕ) (n : ℕ) (a : Fin 10),
  N ≥ 10 ∧ 
  a.val ≠ 0 ∧
  N = a.val * 10^n + (N / 57) :=
sorry

end NUMINAMATH_CALUDE_no_whole_number_57_times_less_l2021_202160


namespace NUMINAMATH_CALUDE_hyperbola_condition_l2021_202159

/-- Represents the equation of a conic section --/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 + m) + y^2 / (m + 1) = 1 ∧ 
  ((2 + m > 0 ∧ m + 1 < 0) ∨ (2 + m < 0 ∧ m + 1 > 0))

/-- The main theorem stating the condition for the equation to represent a hyperbola --/
theorem hyperbola_condition (m : ℝ) : 
  is_hyperbola m ↔ -2 < m ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l2021_202159


namespace NUMINAMATH_CALUDE_updated_p_value_l2021_202105

/-- Given the equation fp - w = 20000, where f = 10 and w = 10 + 250i, prove that p = 2001 + 25i -/
theorem updated_p_value (f w p : ℂ) : 
  f = 10 → w = 10 + 250 * Complex.I → f * p - w = 20000 → p = 2001 + 25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_updated_p_value_l2021_202105


namespace NUMINAMATH_CALUDE_matrix_product_50_l2021_202128

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range n).foldl
    (fun acc k => acc * !![1, 2*(k+1); 0, 1])
    !![1, 0; 0, 1]

theorem matrix_product_50 :
  matrix_product 50 = !![1, 2550; 0, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_product_50_l2021_202128


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2021_202162

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  S : ℕ → ℚ  -- Sum function
  is_arithmetic : ∀ n : ℕ, S (n + 1) - S n = S (n + 2) - S (n + 1)

/-- Theorem: If S_2 / S_4 = 1/3, then S_4 / S_8 = 3/10 for an arithmetic sequence -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) 
  (h : seq.S 2 / seq.S 4 = 1 / 3) : 
  seq.S 4 / seq.S 8 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2021_202162


namespace NUMINAMATH_CALUDE_f_max_min_values_l2021_202142

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - x^3

-- State the theorem
theorem f_max_min_values :
  (∃ x : ℝ, f x = 2 ∧ ∀ y : ℝ, f y ≤ 2) ∧
  (∃ x : ℝ, f x = -2 ∧ ∀ y : ℝ, f y ≥ -2) := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_values_l2021_202142


namespace NUMINAMATH_CALUDE_remainder_b_sixth_l2021_202187

theorem remainder_b_sixth (n : ℕ+) (b : ℤ) (h : b^3 ≡ 1 [ZMOD n]) : b^6 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_remainder_b_sixth_l2021_202187


namespace NUMINAMATH_CALUDE_distance_is_correct_l2021_202198

def point : ℝ × ℝ × ℝ := (2, 3, 4)
def line_point : ℝ × ℝ × ℝ := (4, 5, 6)
def line_direction : ℝ × ℝ × ℝ := (4, 1, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_direction : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_is_correct : 
  distance_to_line point line_point line_direction = (9 * Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_distance_is_correct_l2021_202198


namespace NUMINAMATH_CALUDE_polynomial_composition_difference_l2021_202176

theorem polynomial_composition_difference (f : Polynomial ℝ) :
  ∃ (g h : Polynomial ℝ), f = g.comp h - h.comp g := by
  sorry

end NUMINAMATH_CALUDE_polynomial_composition_difference_l2021_202176


namespace NUMINAMATH_CALUDE_special_function_value_l2021_202114

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2))

/-- The main theorem stating that f(2009) = 2 for any function satisfying SpecialFunction -/
theorem special_function_value :
    ∀ f : ℝ → ℝ, SpecialFunction f → f 2009 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l2021_202114


namespace NUMINAMATH_CALUDE_monotone_function_characterization_l2021_202120

/-- A monotone function from integers to integers -/
def MonotoneIntFunction (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, x ≤ y → f x ≤ f y

/-- The functional equation that f must satisfy -/
def SatisfiesFunctionalEquation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (x^2005 + y^2005) = (f x)^2005 + (f y)^2005

/-- The main theorem statement -/
theorem monotone_function_characterization (f : ℤ → ℤ) 
  (hm : MonotoneIntFunction f) (hf : SatisfiesFunctionalEquation f) :
  (∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x) :=
sorry

end NUMINAMATH_CALUDE_monotone_function_characterization_l2021_202120


namespace NUMINAMATH_CALUDE_lcm_24_36_45_l2021_202172

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_45_l2021_202172


namespace NUMINAMATH_CALUDE_nancy_album_pictures_l2021_202167

theorem nancy_album_pictures (total : ℕ) (num_albums : ℕ) (pics_per_album : ℕ) 
  (h1 : total = 51)
  (h2 : num_albums = 8)
  (h3 : pics_per_album = 5) :
  total - (num_albums * pics_per_album) = 11 := by
sorry

end NUMINAMATH_CALUDE_nancy_album_pictures_l2021_202167


namespace NUMINAMATH_CALUDE_midpoint_sum_midpoint_sum_specific_l2021_202126

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (3, -1) and (11, 21) is 17. -/
theorem midpoint_sum : ℝ × ℝ → ℝ × ℝ → ℝ
  | (x₁, y₁) => λ (x₂, y₂) => (x₁ + x₂) / 2 + (y₁ + y₂) / 2

#check midpoint_sum (3, -1) (11, 21) = 17

theorem midpoint_sum_specific :
  midpoint_sum (3, -1) (11, 21) = 17 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_midpoint_sum_specific_l2021_202126


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2021_202107

/-- Represents the speed of a swimmer in still water and the speed of the stream -/
structure SwimmerSpeeds where
  manSpeed : ℝ
  streamSpeed : ℝ

/-- Calculates the effective speed given the swimmer's speed and stream speed -/
def effectiveSpeed (speeds : SwimmerSpeeds) (isDownstream : Bool) : ℝ :=
  if isDownstream then speeds.manSpeed + speeds.streamSpeed else speeds.manSpeed - speeds.streamSpeed

/-- Theorem stating the speed of the man in still water given the problem conditions -/
theorem swimmer_speed_in_still_water
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (time : ℝ)
  (h_downstream : downstream_distance = 28)
  (h_upstream : upstream_distance = 12)
  (h_time : time = 2)
  (speeds : SwimmerSpeeds)
  (h_downstream_speed : effectiveSpeed speeds true = downstream_distance / time)
  (h_upstream_speed : effectiveSpeed speeds false = upstream_distance / time) :
  speeds.manSpeed = 10 := by
sorry


end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l2021_202107


namespace NUMINAMATH_CALUDE_alloy_composition_ratio_l2021_202118

/-- Given two alloys A and B, with known compositions and mixture properties,
    prove that the ratio of tin to copper in alloy B is 1:4. -/
theorem alloy_composition_ratio :
  -- Define the masses of alloys
  ∀ (mass_A mass_B : ℝ),
  -- Define the ratio of lead to tin in alloy A
  ∀ (lead_ratio tin_ratio : ℝ),
  -- Define the total amount of tin in the mixture
  ∀ (total_tin : ℝ),
  -- Conditions
  mass_A = 60 →
  mass_B = 100 →
  lead_ratio = 3 →
  tin_ratio = 2 →
  total_tin = 44 →
  -- Calculate tin in alloy A
  let tin_A := (tin_ratio / (lead_ratio + tin_ratio)) * mass_A
  -- Calculate tin in alloy B
  let tin_B := total_tin - tin_A
  -- Calculate copper in alloy B
  let copper_B := mass_B - tin_B
  -- Prove the ratio
  tin_B / copper_B = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_alloy_composition_ratio_l2021_202118


namespace NUMINAMATH_CALUDE_inequality_proof_l2021_202141

def f (a x : ℝ) : ℝ := |x - a|

theorem inequality_proof (a s t : ℝ) (h1 : ∀ x, f a x ≤ 4 ↔ -1 ≤ x ∧ x ≤ 7) 
    (h2 : s > 0) (h3 : t > 0) (h4 : 2*s + t = a) : 
    1/s + 8/t ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2021_202141


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l2021_202134

/-- Given positive numbers a, b, c, d with b < d, 
    the maximum value of y = a√(x - b) + c√(d - x) is √((d-b)(a²+c²)) -/
theorem max_value_of_sum_of_roots (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hbd : b < d) :
  (∀ x, b ≤ x ∧ x ≤ d → a * Real.sqrt (x - b) + c * Real.sqrt (d - x) ≤ Real.sqrt ((d - b) * (a^2 + c^2))) ∧
  (∃ x, b < x ∧ x < d ∧ a * Real.sqrt (x - b) + c * Real.sqrt (d - x) = Real.sqrt ((d - b) * (a^2 + c^2))) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_roots_l2021_202134


namespace NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l2021_202110

theorem percentage_of_red_non_honda_cars 
  (total_cars : ℕ) 
  (honda_cars : ℕ) 
  (honda_red_ratio : ℚ) 
  (total_red_ratio : ℚ) 
  (h1 : total_cars = 9000)
  (h2 : honda_cars = 5000)
  (h3 : honda_red_ratio = 90 / 100)
  (h4 : total_red_ratio = 60 / 100)
  : (↑(total_cars * total_red_ratio - honda_cars * honda_red_ratio) / ↑(total_cars - honda_cars) : ℚ) = 225 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l2021_202110


namespace NUMINAMATH_CALUDE_propositions_true_l2021_202171

theorem propositions_true :
  (∀ a b c : ℝ, c ≠ 0 → a * c^2 > b * c^2 → a > b) ∧
  (∀ a : ℝ, 1 / a > 1 → 0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_propositions_true_l2021_202171


namespace NUMINAMATH_CALUDE_ninth_observation_l2021_202184

theorem ninth_observation (n : ℕ) (original_avg new_avg : ℚ) (decrease : ℚ) :
  n = 8 →
  original_avg = 15 →
  decrease = 2 →
  new_avg = original_avg - decrease →
  (n * original_avg + (n + 1) * new_avg) / (2 * n + 1) - original_avg = -3 :=
by sorry

end NUMINAMATH_CALUDE_ninth_observation_l2021_202184


namespace NUMINAMATH_CALUDE_nuts_in_boxes_l2021_202143

theorem nuts_in_boxes (x y z : ℕ) 
  (h1 : x + 6 = y + z) 
  (h2 : y + 10 = x + z) : 
  z = 8 := by
sorry

end NUMINAMATH_CALUDE_nuts_in_boxes_l2021_202143


namespace NUMINAMATH_CALUDE_base12_addition_correct_l2021_202177

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

/-- Converts a Digit12 to its decimal value --/
def toDecimal (d : Digit12) : Nat :=
  match d with
  | Digit12.D0 => 0
  | Digit12.D1 => 1
  | Digit12.D2 => 2
  | Digit12.D3 => 3
  | Digit12.D4 => 4
  | Digit12.D5 => 5
  | Digit12.D6 => 6
  | Digit12.D7 => 7
  | Digit12.D8 => 8
  | Digit12.D9 => 9
  | Digit12.A => 10
  | Digit12.B => 11

/-- Represents a number in base 12 --/
def Base12 := List Digit12

/-- Converts a Base12 number to its decimal value --/
def base12ToDecimal (n : Base12) : Nat :=
  n.foldr (fun d acc => toDecimal d + 12 * acc) 0

/-- Addition in base 12 --/
def addBase12 (a b : Base12) : Base12 :=
  sorry -- Implementation details omitted

theorem base12_addition_correct :
  addBase12 [Digit12.D8, Digit12.A, Digit12.D2] [Digit12.D3, Digit12.B, Digit12.D7] =
  [Digit12.D1, Digit12.D0, Digit12.D9, Digit12.D9] :=
by sorry

end NUMINAMATH_CALUDE_base12_addition_correct_l2021_202177


namespace NUMINAMATH_CALUDE_fourth_quadrant_a_range_l2021_202173

-- Define the complex number z
def z (a : ℝ) : ℂ := (1 - 2*Complex.I) * (a + Complex.I)

-- Define the point M
def M (a : ℝ) : ℝ × ℝ := (a + 2, 1 - 2*a)

-- Theorem statement
theorem fourth_quadrant_a_range (a : ℝ) :
  (M a).1 > 0 ∧ (M a).2 < 0 → a > 1/2 := by sorry

end NUMINAMATH_CALUDE_fourth_quadrant_a_range_l2021_202173


namespace NUMINAMATH_CALUDE_stability_ratio_calculation_l2021_202104

theorem stability_ratio_calculation (T H L : ℚ) : 
  T = 3 → H = 9 → L = (30 * T^3) / H^3 → L = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_stability_ratio_calculation_l2021_202104


namespace NUMINAMATH_CALUDE_original_number_is_35_l2021_202108

-- Define a two-digit number type
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

-- Define functions to get tens and units digits
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

-- Define a function to swap digits
def swap_digits (n : TwoDigitNumber) : TwoDigitNumber :=
  ⟨10 * (units_digit n) + (tens_digit n), by sorry⟩

-- Theorem statement
theorem original_number_is_35 (n : TwoDigitNumber) 
  (h1 : tens_digit n + units_digit n = 8)
  (h2 : (swap_digits n).val = n.val + 18) : 
  n.val = 35 := by sorry

end NUMINAMATH_CALUDE_original_number_is_35_l2021_202108


namespace NUMINAMATH_CALUDE_lamp_arrangement_l2021_202151

theorem lamp_arrangement (n : ℕ) (k : ℕ) (h : n = 6 ∧ k = 2) :
  (Finset.range (n - k + 1)).card.choose k = 10 := by
  sorry

end NUMINAMATH_CALUDE_lamp_arrangement_l2021_202151


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l2021_202158

theorem sin_two_alpha_value (α : ℝ) (h : Real.sin α - Real.cos α = 4/3) : 
  Real.sin (2 * α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l2021_202158


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l2021_202157

open Real

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_strictly_increasing :
  (∀ x > 0, x^2 * (deriv f x) + 2*x * f x = exp x / x) →
  f 2 = exp 2 / 8 →
  StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l2021_202157


namespace NUMINAMATH_CALUDE_salary_change_percentage_l2021_202145

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * 1.5
  let final_salary := increased_salary * 0.9
  (final_salary - initial_salary) / initial_salary * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l2021_202145


namespace NUMINAMATH_CALUDE_chocolate_boxes_given_away_l2021_202175

theorem chocolate_boxes_given_away (total_boxes : ℕ) (pieces_per_box : ℕ) (remaining_pieces : ℕ) : 
  total_boxes = 14 → pieces_per_box = 6 → remaining_pieces = 54 → 
  (total_boxes * pieces_per_box - remaining_pieces) / pieces_per_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_boxes_given_away_l2021_202175


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l2021_202112

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l2021_202112


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_70_factorial_l2021_202179

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  let m := n % 100
  if m ≠ 0 then m else last_two_nonzero_digits (n / 10)

theorem last_two_nonzero_digits_70_factorial :
  ∃ n : ℕ, last_two_nonzero_digits (factorial 70) = n ∧ n < 100 := by
  sorry

#eval last_two_nonzero_digits (factorial 70)

end NUMINAMATH_CALUDE_last_two_nonzero_digits_70_factorial_l2021_202179


namespace NUMINAMATH_CALUDE_street_crossing_time_l2021_202124

/-- Proves that a person walking at 5.4 km/h takes 12 minutes to cross a 1080 m street -/
theorem street_crossing_time :
  let street_length : ℝ := 1080  -- length in meters
  let speed_kmh : ℝ := 5.4       -- speed in km/h
  let speed_mpm : ℝ := speed_kmh * 1000 / 60  -- speed in meters per minute
  let time_minutes : ℝ := street_length / speed_mpm
  time_minutes = 12 := by sorry

end NUMINAMATH_CALUDE_street_crossing_time_l2021_202124


namespace NUMINAMATH_CALUDE_expression_evaluation_l2021_202139

theorem expression_evaluation (x : ℝ) (h : x = -1) :
  (((x - 2) / x - x / (x + 2)) / ((x + 2) / (x^2 + 4*x + 4))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2021_202139


namespace NUMINAMATH_CALUDE_problem_statement_l2021_202133

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.exp x = 0.1

def l₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ x - a * y = 0

def l₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ 2 * x + a * y - 1 = 0

def q : Prop := ∀ a : ℝ, (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y → a = Real.sqrt 2)

theorem problem_statement : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2021_202133


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2021_202194

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 1) : a^3 + b^3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2021_202194


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_l2021_202111

theorem max_sum_under_constraints (a b : ℝ) :
  4 * a + 3 * b ≤ 10 →
  3 * a + 5 * b ≤ 11 →
  a + b ≤ 156 / 55 := by
sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_l2021_202111


namespace NUMINAMATH_CALUDE_smallest_alpha_beta_inequality_optimal_alpha_beta_optimal_beta_value_l2021_202147

theorem smallest_alpha_beta_inequality (α : ℝ) (β : ℝ) :
  (α > 0 ∧ β > 0 ∧
   ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^α / β) →
  α ≥ 2 :=
by sorry

theorem optimal_alpha_beta :
  ∃ β : ℝ, β > 0 ∧
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 →
    Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β :=
by sorry

theorem optimal_beta_value (β : ℝ) :
  (β > 0 ∧
   ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 →
     Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β) →
  β ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_alpha_beta_inequality_optimal_alpha_beta_optimal_beta_value_l2021_202147


namespace NUMINAMATH_CALUDE_emily_income_l2021_202102

/-- Represents the tax structure and Emily's tax payment --/
structure TaxSystem where
  q : ℝ  -- Base tax rate
  income : ℝ  -- Emily's annual income
  total_tax : ℝ  -- Total tax paid by Emily

/-- The tax system satisfies the given conditions --/
def valid_tax_system (ts : TaxSystem) : Prop :=
  ts.total_tax = 
    (0.01 * ts.q * 35000 + 
     0.01 * (ts.q + 3) * 15000 + 
     0.01 * (ts.q + 5) * (ts.income - 50000)) *
    (if ts.income > 50000 then 1 else 0) +
    (0.01 * ts.q * 35000 + 
     0.01 * (ts.q + 3) * (ts.income - 35000)) *
    (if ts.income > 35000 ∧ ts.income ≤ 50000 then 1 else 0) +
    (0.01 * ts.q * ts.income) *
    (if ts.income ≤ 35000 then 1 else 0)

/-- Emily's total tax is (q + 0.75)% of her income --/
def emily_tax_condition (ts : TaxSystem) : Prop :=
  ts.total_tax = 0.01 * (ts.q + 0.75) * ts.income

/-- Theorem: Emily's income is $48235 --/
theorem emily_income (ts : TaxSystem) 
  (h1 : valid_tax_system ts) 
  (h2 : emily_tax_condition ts) : 
  ts.income = 48235 :=
sorry

end NUMINAMATH_CALUDE_emily_income_l2021_202102


namespace NUMINAMATH_CALUDE_team_can_have_odd_and_even_points_l2021_202122

/-- Represents a football team in the tournament -/
structure Team :=
  (id : Nat)
  (points : Nat)

/-- Represents the football tournament -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : Nat)
  (points_for_win : Nat)
  (points_for_draw : Nat)
  (bonus_points : Nat)

/-- Definition of the specific tournament conditions -/
def specific_tournament : Tournament :=
  { teams := sorry,
    num_teams := 10,
    points_for_win := 3,
    points_for_draw := 1,
    bonus_points := 5 }

/-- Theorem stating that a team can end with both odd and even points -/
theorem team_can_have_odd_and_even_points (t : Tournament) 
  (h1 : t.num_teams = 10)
  (h2 : t.points_for_win = 3)
  (h3 : t.points_for_draw = 1)
  (h4 : t.bonus_points = 5) :
  ∃ (team1 team2 : Team), 
    team1 ∈ t.teams ∧ 
    team2 ∈ t.teams ∧ 
    Odd team1.points ∧ 
    Even team2.points :=
sorry

end NUMINAMATH_CALUDE_team_can_have_odd_and_even_points_l2021_202122


namespace NUMINAMATH_CALUDE_sum_exterior_angles_quadrilateral_l2021_202103

/-- A quadrilateral is a polygon with four sides. -/
def Quadrilateral : Type := Unit  -- Placeholder definition

/-- The sum of exterior angles of a polygon. -/
def sum_exterior_angles (p : Type) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of a quadrilateral is 360 degrees. -/
theorem sum_exterior_angles_quadrilateral :
  sum_exterior_angles Quadrilateral = 360 := by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_quadrilateral_l2021_202103


namespace NUMINAMATH_CALUDE_min_ratio_of_integers_with_mean_l2021_202170

theorem min_ratio_of_integers_with_mean (x y : ℤ) : 
  10 ≤ x ∧ x ≤ 150 → 
  10 ≤ y ∧ y ≤ 150 → 
  (x + y) / 2 = 75 → 
  ∃ (x' y' : ℤ), 
    10 ≤ x' ∧ x' ≤ 150 ∧ 
    10 ≤ y' ∧ y' ≤ 150 ∧ 
    (x' + y') / 2 = 75 ∧ 
    x' / y' ≤ x / y ∧
    x' / y' = 1 / 14 :=
by sorry

end NUMINAMATH_CALUDE_min_ratio_of_integers_with_mean_l2021_202170


namespace NUMINAMATH_CALUDE_remaining_area_after_cutting_triangles_l2021_202190

/-- The area of a square with side length n -/
def square_area (n : ℕ) : ℕ := n * n

/-- The area of a rectangle with width w and height h -/
def rectangle_area (w h : ℕ) : ℕ := w * h

theorem remaining_area_after_cutting_triangles :
  let total_area := square_area 6
  let dark_gray_area := rectangle_area 1 3
  let light_gray_area := rectangle_area 2 3
  total_area - (dark_gray_area + light_gray_area) = 27 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_after_cutting_triangles_l2021_202190


namespace NUMINAMATH_CALUDE_square_root_problem_l2021_202132

theorem square_root_problem (m : ℝ) (h1 : m > 0) (h2 : ∃ a : ℝ, (3 - a)^2 = m ∧ (2*a + 1)^2 = m) : m = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l2021_202132


namespace NUMINAMATH_CALUDE_parabola_vertex_l2021_202197

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -3 * (x - 1)^2 + 4

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 4)

/-- Theorem: The vertex of the parabola y = -3(x-1)^2 + 4 is at the point (1,4) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2021_202197


namespace NUMINAMATH_CALUDE_lesser_number_proof_l2021_202129

theorem lesser_number_proof (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : 
  min x y = 25 := by
sorry

end NUMINAMATH_CALUDE_lesser_number_proof_l2021_202129


namespace NUMINAMATH_CALUDE_line_y_intercept_l2021_202117

/-- Proves that for a line ax + y + 2 = 0 with an inclination angle of 3π/4, the y-intercept is -2 -/
theorem line_y_intercept (a : ℝ) : 
  (∀ x y : ℝ, a * x + y + 2 = 0) → 
  (Real.tan (3 * Real.pi / 4) = -a) → 
  (∃ x : ℝ, 0 * x + (-2) + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l2021_202117


namespace NUMINAMATH_CALUDE_tank_full_time_l2021_202135

/-- Represents the state of a water tank system -/
structure TankSystem where
  capacity : ℕ
  fill_rate_a : ℕ
  fill_rate_b : ℕ
  drain_rate : ℕ

/-- Calculates the time required to fill the tank -/
def time_to_fill (system : TankSystem) : ℕ :=
  let net_fill_per_cycle := system.fill_rate_a + system.fill_rate_b - system.drain_rate
  let cycles := system.capacity / net_fill_per_cycle
  cycles * 3 - 1

/-- Theorem stating that the tank will be full in 56 minutes -/
theorem tank_full_time (system : TankSystem) 
    (h1 : system.capacity = 950)
    (h2 : system.fill_rate_a = 40)
    (h3 : system.fill_rate_b = 30)
    (h4 : system.drain_rate = 20) :
  time_to_fill system = 56 := by
  sorry

#eval time_to_fill { capacity := 950, fill_rate_a := 40, fill_rate_b := 30, drain_rate := 20 }

end NUMINAMATH_CALUDE_tank_full_time_l2021_202135


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2021_202188

/-- A regular polygon with an exterior angle of 10 degrees has 36 sides. -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 10 → n = 36 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2021_202188


namespace NUMINAMATH_CALUDE_unique_n_for_total_digits_l2021_202183

/-- Sum of digits function for a natural number -/
def sumOfDigits (k : ℕ) : ℕ := sorry

/-- Total sum of digits for all numbers from 1 to n -/
def totalSumOfDigits (n : ℕ) : ℕ := 
  (Finset.range n).sum (fun i => sumOfDigits (i + 1))

/-- The theorem statement -/
theorem unique_n_for_total_digits : 
  ∃! n : ℕ, totalSumOfDigits n = 777 := by sorry

end NUMINAMATH_CALUDE_unique_n_for_total_digits_l2021_202183


namespace NUMINAMATH_CALUDE_jimmy_flour_amount_l2021_202166

/-- The amount of flour Jimmy bought initially -/
def initial_flour (working_hours : ℕ) (minutes_per_pizza : ℕ) (flour_per_pizza : ℚ) (leftover_pizzas : ℕ) : ℚ :=
  let pizzas_per_hour : ℕ := 60 / minutes_per_pizza
  let total_pizzas : ℕ := working_hours * pizzas_per_hour + leftover_pizzas
  total_pizzas * flour_per_pizza

/-- Theorem stating that Jimmy bought 22 kg of flour initially -/
theorem jimmy_flour_amount :
  initial_flour 7 10 (1/2) 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_flour_amount_l2021_202166


namespace NUMINAMATH_CALUDE_frog_jump_probability_l2021_202106

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the rectangular garden -/
structure Garden where
  bottomLeft : Point
  topRight : Point

/-- Represents the possible jump directions -/
inductive JumpDirection
  | Up
  | Down
  | Left
  | Right
  | NorthEast
  | NorthWest
  | SouthEast
  | SouthWest

/-- Represents the possible jump lengths -/
inductive JumpLength
  | One
  | Two

/-- Function to calculate the probability of ending on a horizontal side -/
def probabilityHorizontalEnd (garden : Garden) (start : Point) : ℝ :=
  sorry

/-- Theorem stating the probability of ending on a horizontal side is 0.4 -/
theorem frog_jump_probability (garden : Garden) (start : Point) :
  garden.bottomLeft = ⟨1, 1⟩ ∧
  garden.topRight = ⟨5, 6⟩ ∧
  start = ⟨2, 3⟩ →
  probabilityHorizontalEnd garden start = 0.4 :=
sorry


end NUMINAMATH_CALUDE_frog_jump_probability_l2021_202106


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l2021_202149

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  x + y = 2 * a - b - 4 ∧ x - y = b - 4

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

-- Part 1
theorem part_one (a b : ℝ) (P : Point) :
  a = 1 → b = 2 → system a b P.x P.y → P.x = -3 ∧ P.y = -1 := by sorry

-- Part 2
theorem part_two (a b : ℝ) (P : Point) :
  system a b P.x P.y →
  P.x < 0 ∧ P.y > 0 →
  (∃ (n : ℕ), n = 4 ∧ ∀ (m : ℤ), (∃ (a' : ℝ), a' = a ∧ system a' b P.x P.y) → m ≤ n) →
  -1 ≤ b ∧ b < 0 := by sorry

-- Part 3
theorem part_three (a b t : ℝ) (P : Point) :
  system a b P.x P.y →
  (∃! (z : ℝ), z = 2 ∧ P.y * z + P.x + 4 = 0) →
  (a * t > b ↔ t > 3/2 ∨ t < 3/2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l2021_202149


namespace NUMINAMATH_CALUDE_constant_distance_to_line_l2021_202174

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := y^2 / 2 + x^2 = 1

-- Define the related circle E
def related_circle_E (x y : ℝ) : Prop := x^2 + y^2 = 2/3

-- Define the line l
def line_l (k m x y : ℝ) : Prop := y = k * x + m

-- Theorem statement
theorem constant_distance_to_line
  (k m x1 y1 x2 y2 : ℝ)
  (h1 : ellipse_C x1 y1)
  (h2 : ellipse_C x2 y2)
  (h3 : line_l k m x1 y1)
  (h4 : line_l k m x2 y2)
  (h5 : ∃ (x y : ℝ), related_circle_E x y ∧ line_l k m x y) :
  ∃ (d : ℝ), d = Real.sqrt 6 / 3 ∧
  (∀ (x y : ℝ), line_l k m x y → (x^2 + y^2 = d^2)) :=
sorry

end NUMINAMATH_CALUDE_constant_distance_to_line_l2021_202174


namespace NUMINAMATH_CALUDE_num_technicians_is_eight_l2021_202130

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := sorry

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 24

/-- Represents the average salary of all workers -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technician workers -/
def avg_salary_others : ℕ := 6000

/-- Theorem stating that the number of technicians is 8 given the workshop conditions -/
theorem num_technicians_is_eight :
  num_technicians = 8 ∧
  num_technicians + (total_workers - num_technicians) = total_workers ∧
  num_technicians * avg_salary_technicians +
    (total_workers - num_technicians) * avg_salary_others =
    total_workers * avg_salary_all :=
by sorry

end NUMINAMATH_CALUDE_num_technicians_is_eight_l2021_202130


namespace NUMINAMATH_CALUDE_bacteria_growth_7_hours_l2021_202109

/-- Calculates the number of bacteria after a given number of hours, 
    given an initial count and doubling time. -/
def bacteria_growth (initial_count : ℕ) (hours : ℕ) : ℕ :=
  initial_count * 2^hours

/-- Theorem stating that after 7 hours, starting with 10 bacteria, 
    the population will be 1280 bacteria. -/
theorem bacteria_growth_7_hours : 
  bacteria_growth 10 7 = 1280 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_7_hours_l2021_202109


namespace NUMINAMATH_CALUDE_tickets_left_to_sell_l2021_202115

theorem tickets_left_to_sell (total : ℕ) (first_week : ℕ) (second_week : ℕ) 
  (h1 : total = 90) 
  (h2 : first_week = 38) 
  (h3 : second_week = 17) :
  total - (first_week + second_week) = 35 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_to_sell_l2021_202115


namespace NUMINAMATH_CALUDE_inequality_proof_l2021_202199

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let K := a^4*(b^2*c + b*c^2) + a^3*(b^3*c + b*c^3) + a^2*(b^3*c^2 + b^2*c^3 + b^2*c + b*c^2) + a*(b^3*c + b*c^3) + (b^3*c^2 + b^2*c^3)
  K ≥ 12*a^2*b^2*c^2 ∧ (K = 12*a^2*b^2*c^2 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2021_202199


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l2021_202186

theorem min_value_of_function (x : ℝ) (hx : x < 0) :
  (1 - 2*x - 3/x) ≥ 1 + 2*Real.sqrt 6 := by
  sorry

theorem min_value_achieved (x : ℝ) (hx : x < 0) :
  ∃ x₀, x₀ < 0 ∧ (1 - 2*x₀ - 3/x₀) = 1 + 2*Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achieved_l2021_202186


namespace NUMINAMATH_CALUDE_gray_eyed_black_haired_count_l2021_202182

theorem gray_eyed_black_haired_count : ∀ (total red_haired black_haired green_eyed gray_eyed green_eyed_red_haired : ℕ),
  total = 60 →
  red_haired + black_haired = total →
  green_eyed + gray_eyed = total →
  green_eyed_red_haired = 20 →
  black_haired = 40 →
  gray_eyed = 25 →
  gray_eyed - (red_haired - green_eyed_red_haired) = 25 := by
  sorry

end NUMINAMATH_CALUDE_gray_eyed_black_haired_count_l2021_202182


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2021_202181

theorem smallest_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  ∀ m : ℕ, m > 0 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2021_202181


namespace NUMINAMATH_CALUDE_least_common_solution_l2021_202154

theorem least_common_solution : ∃ b : ℕ, b > 0 ∧ 
  b % 7 = 6 ∧ 
  b % 11 = 10 ∧ 
  b % 13 = 12 ∧
  (∀ c : ℕ, c > 0 ∧ c % 7 = 6 ∧ c % 11 = 10 ∧ c % 13 = 12 → b ≤ c) ∧
  b = 1000 := by
  sorry

end NUMINAMATH_CALUDE_least_common_solution_l2021_202154


namespace NUMINAMATH_CALUDE_ara_height_is_60_l2021_202155

/-- Represents the heights of Shea and Ara --/
structure Heights where
  initial : ℝ  -- Initial height of both Shea and Ara
  shea_current : ℝ  -- Shea's current height
  shea_growth_rate : ℝ  -- Shea's growth rate as a decimal
  ara_growth_difference : ℝ  -- Difference between Shea and Ara's growth in inches

/-- Calculates Ara's current height given the initial conditions --/
def ara_current_height (h : Heights) : ℝ :=
  h.initial + (h.shea_current - h.initial) - h.ara_growth_difference

/-- Theorem stating that Ara's current height is 60 inches --/
theorem ara_height_is_60 (h : Heights)
  (h_shea_current : h.shea_current = 65)
  (h_shea_growth : h.shea_growth_rate = 0.3)
  (h_ara_diff : h.ara_growth_difference = 5) :
  ara_current_height h = 60 := by
  sorry

#eval ara_current_height { initial := 50, shea_current := 65, shea_growth_rate := 0.3, ara_growth_difference := 5 }

end NUMINAMATH_CALUDE_ara_height_is_60_l2021_202155


namespace NUMINAMATH_CALUDE_cos_sin_transformation_l2021_202123

theorem cos_sin_transformation (x : Real) : 
  Real.sqrt 2 * Real.cos x = Real.sqrt 2 * Real.sin (2 * (x + Real.pi/4) + Real.pi/4) := by
sorry

end NUMINAMATH_CALUDE_cos_sin_transformation_l2021_202123


namespace NUMINAMATH_CALUDE_zach_needs_six_more_l2021_202121

/-- Calculates how much more money Zach needs to buy a bike --/
def money_needed_for_bike (bike_cost allowance lawn_pay babysit_rate current_savings babysit_hours : ℕ) : ℕ :=
  let total_earnings := allowance + lawn_pay + babysit_rate * babysit_hours
  let total_savings := current_savings + total_earnings
  if total_savings ≥ bike_cost then 0
  else bike_cost - total_savings

/-- Proves that Zach needs $6 more to buy the bike --/
theorem zach_needs_six_more : 
  money_needed_for_bike 100 5 10 7 65 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_zach_needs_six_more_l2021_202121


namespace NUMINAMATH_CALUDE_sum_below_threshold_equals_14_tenths_l2021_202185

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

def sum_below_threshold (nums : List ℚ) (t : ℚ) : ℚ :=
  (nums.filter (· ≤ t)).sum

theorem sum_below_threshold_equals_14_tenths :
  sum_below_threshold numbers threshold = 14/10 := by
  sorry

end NUMINAMATH_CALUDE_sum_below_threshold_equals_14_tenths_l2021_202185


namespace NUMINAMATH_CALUDE_all_red_final_state_l2021_202169

/-- Represents the possible colors of chameleons -/
inductive Color
  | Yellow
  | Green
  | Red

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : ℕ
  green : ℕ
  red : ℕ

/-- The initial state of chameleons on the island -/
def initial_state : ChameleonState :=
  { yellow := 7, green := 10, red := 17 }

/-- The total number of chameleons on the island -/
def total_chameleons : ℕ := 34

/-- Represents a meeting between two chameleons of different colors -/
def meet (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.Yellow, Color.Green => Color.Red
  | Color.Yellow, Color.Red => Color.Green
  | Color.Green, Color.Red => Color.Yellow
  | Color.Green, Color.Yellow => Color.Red
  | Color.Red, Color.Yellow => Color.Green
  | Color.Red, Color.Green => Color.Yellow
  | _, _ => c1  -- If same color, no change

/-- The invariant quantity Delta -/
def Delta (state : ChameleonState) : ℤ :=
  state.red - state.green

/-- Theorem: The only possible final state is all chameleons being red -/
theorem all_red_final_state :
  ∀ (final_state : ChameleonState),
    final_state.yellow + final_state.green + final_state.red = total_chameleons →
    (final_state.yellow = 0 ∧ final_state.green = 0 ∧ final_state.red = total_chameleons) ∨
    (final_state.yellow ≠ 0 ∨ final_state.green ≠ 0 ∨ final_state.red ≠ total_chameleons) :=
sorry


end NUMINAMATH_CALUDE_all_red_final_state_l2021_202169


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l2021_202192

theorem ones_digit_of_large_power : ∃ n : ℕ, n < 10 ∧ 34^(34 * 17^17) ≡ n [ZMOD 10] ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l2021_202192


namespace NUMINAMATH_CALUDE_expression_value_l2021_202101

theorem expression_value (a b m n x : ℝ) : 
  (a = -b) →                   -- a and b are opposite numbers
  (m * n = 1) →                -- m and n are reciprocal numbers
  (m - n ≠ 0) →                -- given condition
  (abs x = 2) →                -- absolute value of x is 2
  (-2 * m * n + (b + a) / (m - n) - x = -4 ∨ 
   -2 * m * n + (b + a) / (m - n) - x = 0) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2021_202101


namespace NUMINAMATH_CALUDE_determinant_zero_implies_y_eq_neg_b_l2021_202191

variable (b y : ℝ)

def matrix (b y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![y + b, y, y],
    ![y, y + b, y],
    ![y, y, y + b]]

theorem determinant_zero_implies_y_eq_neg_b (h1 : b ≠ 0) 
  (h2 : Matrix.det (matrix b y) = 0) : y = -b := by
  sorry

end NUMINAMATH_CALUDE_determinant_zero_implies_y_eq_neg_b_l2021_202191


namespace NUMINAMATH_CALUDE_expression_equality_l2021_202180

theorem expression_equality : 
  (-(-2) + (1 + Real.pi) ^ 0 - |1 - Real.sqrt 2| + Real.sqrt 8 - Real.cos (45 * π / 180)) = 
  2 + 5 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2021_202180


namespace NUMINAMATH_CALUDE_rectangle_area_l2021_202127

/-- The area of a rectangle with length 8m and width 50dm is 40 m² -/
theorem rectangle_area : 
  let length : ℝ := 8
  let width_dm : ℝ := 50
  let width_m : ℝ := width_dm / 10
  let area : ℝ := length * width_m
  area = 40 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2021_202127


namespace NUMINAMATH_CALUDE_oxygen_weight_in_compound_l2021_202140

/-- The atomic weight of hydrogen -/
def hydrogen_weight : ℝ := 1

/-- The atomic weight of chlorine -/
def chlorine_weight : ℝ := 35.5

/-- The total molecular weight of the compound -/
def total_weight : ℝ := 68

/-- The number of hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of chlorine atoms in the compound -/
def chlorine_count : ℕ := 1

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- Theorem: The atomic weight of oxygen in the given compound is 15.75 -/
theorem oxygen_weight_in_compound : 
  ∃ (oxygen_weight : ℝ), 
    (hydrogen_count : ℝ) * hydrogen_weight + 
    (chlorine_count : ℝ) * chlorine_weight + 
    (oxygen_count : ℝ) * oxygen_weight = total_weight ∧ 
    oxygen_weight = 15.75 := by sorry

end NUMINAMATH_CALUDE_oxygen_weight_in_compound_l2021_202140


namespace NUMINAMATH_CALUDE_water_depth_when_upright_l2021_202131

/-- Represents a right cylindrical water tank -/
structure WaterTank where
  height : ℝ
  baseDiameter : ℝ

/-- Calculates the volume of water in the tank when horizontal -/
def horizontalWaterVolume (tank : WaterTank) (depth : ℝ) : ℝ :=
  sorry

/-- Calculates the depth of water when the tank is upright -/
def uprightWaterDepth (tank : WaterTank) (horizontalDepth : ℝ) : ℝ :=
  sorry

theorem water_depth_when_upright 
  (tank : WaterTank) 
  (h1 : tank.height = 20)
  (h2 : tank.baseDiameter = 6)
  (h3 : horizontalWaterVolume tank 4 = π * (tank.baseDiameter / 2)^2 * tank.height) :
  uprightWaterDepth tank 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_when_upright_l2021_202131


namespace NUMINAMATH_CALUDE_investment_problem_l2021_202148

theorem investment_problem (total : ℝ) (rate_greater rate_smaller : ℝ) (income_diff : ℝ) :
  total = 10000 ∧ 
  rate_greater = 0.06 ∧ 
  rate_smaller = 0.05 ∧ 
  income_diff = 160 →
  ∃ (greater smaller : ℝ),
    greater + smaller = total ∧
    rate_greater * greater = rate_smaller * smaller + income_diff ∧
    smaller = 4000 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l2021_202148


namespace NUMINAMATH_CALUDE_max_xy_value_l2021_202144

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : 
  x * y ≤ 168 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l2021_202144


namespace NUMINAMATH_CALUDE_bike_riders_proportion_l2021_202116

theorem bike_riders_proportion (total_students bus_riders walkers : ℕ) 
  (h1 : total_students = 92)
  (h2 : bus_riders = 20)
  (h3 : walkers = 27) :
  (total_students - bus_riders - walkers : ℚ) / (total_students - bus_riders : ℚ) = 45 / 72 :=
by sorry

end NUMINAMATH_CALUDE_bike_riders_proportion_l2021_202116


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2021_202164

theorem quadratic_factorization (x : ℂ) : 
  2 * x^2 - 4 * x + 5 = (Real.sqrt 2 * x - Real.sqrt 2 + Complex.I * Real.sqrt 3) * 
                        (Real.sqrt 2 * x - Real.sqrt 2 - Complex.I * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2021_202164


namespace NUMINAMATH_CALUDE_expression_equivalence_l2021_202168

theorem expression_equivalence (x y : ℝ) (h : x * y ≠ 0) :
  ((x^4 + 1) / x^2) * ((y^4 + 1) / y^2) + ((x^4 - 1) / y^2) * ((y^4 - 1) / x^2) = 2 * x^2 * y^2 + 2 / (x^2 * y^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l2021_202168


namespace NUMINAMATH_CALUDE_inequality_proof_l2021_202113

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2 * x * y) / (x + y) + Real.sqrt ((x^2 + y^2) / 2) ≥ (x + y) / 2 + Real.sqrt (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2021_202113


namespace NUMINAMATH_CALUDE_lcm_of_36_and_154_l2021_202163

theorem lcm_of_36_and_154 :
  let a := 36
  let b := 154
  let hcf := 14
  hcf = Nat.gcd a b →
  Nat.lcm a b = 396 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_154_l2021_202163


namespace NUMINAMATH_CALUDE_min_distinct_values_l2021_202189

/-- Represents a list of positive integers -/
def IntegerList := List Nat

/-- Checks if a given value is the unique mode of the list occurring exactly n times -/
def isUniqueMode (list : IntegerList) (mode : Nat) (n : Nat) : Prop :=
  (list.count mode = n) ∧ 
  ∀ x, x ≠ mode → list.count x < n

/-- Theorem: The minimum number of distinct values in a list of 2018 positive integers
    with a unique mode occurring exactly 10 times is 225 -/
theorem min_distinct_values (list : IntegerList) (mode : Nat) :
  list.length = 2018 →
  isUniqueMode list mode 10 →
  list.toFinset.card ≥ 225 :=
sorry

end NUMINAMATH_CALUDE_min_distinct_values_l2021_202189


namespace NUMINAMATH_CALUDE_f_2_3_neg1_eq_5_3_l2021_202165

-- Define the function f
def f (a b c : ℚ) : ℚ := (a + b) / (a - c)

-- State the theorem
theorem f_2_3_neg1_eq_5_3 : f 2 3 (-1) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_f_2_3_neg1_eq_5_3_l2021_202165


namespace NUMINAMATH_CALUDE_hcf_problem_l2021_202152

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 2562) (h2 : Nat.lcm a b = 183) :
  Nat.gcd a b = 14 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l2021_202152


namespace NUMINAMATH_CALUDE_committee_selection_l2021_202146

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) :
  Nat.choose n k = 792 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l2021_202146


namespace NUMINAMATH_CALUDE_runners_passing_count_l2021_202137

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in meters per minute
  radius : ℝ  -- radius of the track in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other -/
def passingCount (r1 r2 : Runner) (duration : ℝ) : ℕ :=
  sorry

theorem runners_passing_count :
  let odell : Runner := { speed := 260, radius := 55, direction := 1 }
  let kershaw : Runner := { speed := 280, radius := 65, direction := -1 }
  passingCount odell kershaw 30 = 126 :=
sorry

end NUMINAMATH_CALUDE_runners_passing_count_l2021_202137


namespace NUMINAMATH_CALUDE_sin_intersection_sum_l2021_202178

open Real

theorem sin_intersection_sum (f : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) :
  (∀ x ∈ Set.Icc 0 (7 * π / 6), f x = Real.sin (2 * x + π / 6)) →
  x₁ < x₂ →
  x₂ < x₃ →
  x₁ ∈ Set.Icc 0 (7 * π / 6) →
  x₂ ∈ Set.Icc 0 (7 * π / 6) →
  x₃ ∈ Set.Icc 0 (7 * π / 6) →
  f x₁ = f x₂ →
  f x₂ = f x₃ →
  x₁ + 2 * x₂ + x₃ = 5 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_sin_intersection_sum_l2021_202178


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l2021_202196

/-- Probability of selecting an individual in systematic sampling -/
theorem systematic_sampling_probability
  (population_size : ℕ)
  (sample_size : ℕ)
  (h1 : population_size = 1001)
  (h2 : sample_size = 50)
  (h3 : population_size > 0)
  (h4 : sample_size ≤ population_size) :
  (sample_size : ℚ) / population_size = 50 / 1001 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l2021_202196


namespace NUMINAMATH_CALUDE_integer_root_pairs_l2021_202136

/-- A function that checks if all roots of a quadratic polynomial ax^2 + bx + c are integers -/
def allRootsInteger (a b c : ℤ) : Prop :=
  ∃ x y : ℤ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y

/-- The main theorem stating the only valid pairs (p,q) -/
theorem integer_root_pairs :
  ∀ p q : ℤ,
    (allRootsInteger 1 p q ∧ allRootsInteger 1 q p) ↔
    ((p = 4 ∧ q = 4) ∨ (p = 9 ∧ q = 8) ∨ (p = 8 ∧ q = 9)) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_pairs_l2021_202136


namespace NUMINAMATH_CALUDE_min_value_f_neg_three_range_of_a_l2021_202161

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem 1: Minimum value of f when a = -3
theorem min_value_f_neg_three :
  ∃ (m : ℝ), m = 4 ∧ ∀ (x : ℝ), f (-3) x ≥ m :=
sorry

-- Theorem 2: Range of a when f(x) ≤ 2a + 2|x-1| for all x
theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), f a x ≤ 2 * a + 2 * |x - 1|) → a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_neg_three_range_of_a_l2021_202161


namespace NUMINAMATH_CALUDE_fraction_inequality_l2021_202119

theorem fraction_inequality (a b c d p q : ℕ+) 
  (h1 : a * d - b * c = 1)
  (h2 : (a : ℚ) / b > (p : ℚ) / q)
  (h3 : (p : ℚ) / q > (c : ℚ) / d) : 
  q ≥ b + d ∧ (q = b + d → p = a + c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2021_202119


namespace NUMINAMATH_CALUDE_ray_fish_market_l2021_202193

/-- Calculates the number of tuna needed to serve customers in a fish market -/
def tuna_needed (total_customers : ℕ) (unsatisfied_customers : ℕ) (pounds_per_customer : ℕ) (pounds_per_tuna : ℕ) : ℕ :=
  ((total_customers - unsatisfied_customers) * pounds_per_customer) / pounds_per_tuna

theorem ray_fish_market :
  tuna_needed 100 20 25 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ray_fish_market_l2021_202193


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2021_202153

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -30 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2021_202153


namespace NUMINAMATH_CALUDE_overall_score_calculation_l2021_202138

theorem overall_score_calculation (score1 score2 score3 : ℚ) 
  (problems1 problems2 problems3 : ℕ) : 
  score1 = 60 / 100 →
  score2 = 75 / 100 →
  score3 = 85 / 100 →
  problems1 = 15 →
  problems2 = 25 →
  problems3 = 20 →
  (score1 * problems1 + score2 * problems2 + score3 * problems3) / 
  (problems1 + problems2 + problems3 : ℚ) = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_overall_score_calculation_l2021_202138


namespace NUMINAMATH_CALUDE_max_xy_value_min_inverse_sum_l2021_202125

-- Part 1
theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 2 * y = 12) :
  xy ≤ 3 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 12 ∧ x * y = 3 :=
sorry

-- Part 2
theorem min_inverse_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 3) :
  1 / x + 1 / y ≥ 1 + 2 * Real.sqrt 2 / 3 ∧
  ∃ x y, x > 0 ∧ y > 0 ∧ x + 2 * y = 3 ∧ 1 / x + 1 / y = 1 + 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_min_inverse_sum_l2021_202125


namespace NUMINAMATH_CALUDE_initial_discount_percentage_l2021_202156

-- Define the original price of the dress
variable (d : ℝ)
-- Define the initial discount percentage
variable (x : ℝ)

-- Theorem statement
theorem initial_discount_percentage
  (h1 : d > 0)  -- Assuming the original price is positive
  (h2 : 0 ≤ x ∧ x ≤ 100)  -- The discount percentage is between 0 and 100
  (h3 : d * (1 - x / 100) * (1 - 40 / 100) = d * 0.33)  -- The equation representing the final price
  : x = 45 := by
  sorry

end NUMINAMATH_CALUDE_initial_discount_percentage_l2021_202156


namespace NUMINAMATH_CALUDE_betty_needs_five_more_l2021_202195

def wallet_cost : ℕ := 100
def betty_initial_savings : ℕ := wallet_cost / 2
def parents_contribution : ℕ := 15
def grandparents_contribution : ℕ := 2 * parents_contribution

theorem betty_needs_five_more :
  wallet_cost - (betty_initial_savings + parents_contribution + grandparents_contribution) = 5 := by
  sorry

end NUMINAMATH_CALUDE_betty_needs_five_more_l2021_202195
