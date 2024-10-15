import Mathlib

namespace NUMINAMATH_CALUDE_barge_unloading_time_l1332_133227

/-- Represents the unloading scenario of a barge with different crane configurations -/
structure BargeUnloading where
  /-- Time (in hours) for one crane of greater capacity to unload the barge alone -/
  x : ℝ
  /-- Time (in hours) for one crane of lesser capacity to unload the barge alone -/
  y : ℝ
  /-- Time (in hours) for one crane of greater capacity and one of lesser capacity to unload together -/
  z : ℝ

/-- The main theorem about the barge unloading scenario -/
theorem barge_unloading_time (b : BargeUnloading) : b.z = 14.4 :=
  sorry

end NUMINAMATH_CALUDE_barge_unloading_time_l1332_133227


namespace NUMINAMATH_CALUDE_power_product_evaluation_l1332_133209

theorem power_product_evaluation : 
  let a : ℕ := 2
  a^3 * a^4 = 128 := by sorry

end NUMINAMATH_CALUDE_power_product_evaluation_l1332_133209


namespace NUMINAMATH_CALUDE_lottery_probability_l1332_133256

theorem lottery_probability : 
  let mega_balls : ℕ := 30
  let winner_balls : ℕ := 50
  let picked_winner_balls : ℕ := 5
  let mega_prob : ℚ := 1 / mega_balls
  let winner_prob : ℚ := 1 / (winner_balls.choose picked_winner_balls)
  mega_prob * winner_prob = 1 / 63562800 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l1332_133256


namespace NUMINAMATH_CALUDE_factorization_equality_l1332_133214

theorem factorization_equality (x y : ℝ) : x^2 * y - 4*y = y*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1332_133214


namespace NUMINAMATH_CALUDE_system_solution_l1332_133261

theorem system_solution (x y : ℝ) :
  (y = x + 1) ∧ (y = -x + 2) ∧ (x = 1/2) ∧ (y = 3/2) →
  (y - x - 1 = 0) ∧ (y + x - 2 = 0) ∧ (x = 1/2) ∧ (y = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1332_133261


namespace NUMINAMATH_CALUDE_number_ratio_l1332_133288

theorem number_ratio (x y z : ℝ) (k : ℝ) : 
  x = 18 →
  y = k * x →
  z = 2 * y →
  (x + y + z) / 3 = 78 →
  y / x = 4 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l1332_133288


namespace NUMINAMATH_CALUDE_product_of_numbers_l1332_133240

theorem product_of_numbers (x y : ℝ) 
  (h1 : x - y = 15) 
  (h2 : x^2 + y^2 = 578) : 
  x * y = (931 - 15 * Real.sqrt 931) / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1332_133240


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1332_133231

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  condition1 : a 3 + a 5 = 8
  condition2 : a 1 * a 5 = 4

/-- The ratio of the 13th term to the 9th term is 9 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : seq.a 13 / seq.a 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1332_133231


namespace NUMINAMATH_CALUDE_dormitory_students_l1332_133291

theorem dormitory_students (total : ℝ) (total_pos : 0 < total) : 
  let first_year := total / 2
  let second_year := total / 2
  let first_year_undeclared := 4 / 5 * first_year
  let first_year_declared := first_year - first_year_undeclared
  let second_year_declared := 1 / 3 * (first_year_declared / first_year) * second_year
  let second_year_undeclared := second_year - second_year_declared
  second_year_undeclared / total = 7 / 15 := by
sorry

end NUMINAMATH_CALUDE_dormitory_students_l1332_133291


namespace NUMINAMATH_CALUDE_smallest_marble_set_marble_set_existence_l1332_133206

theorem smallest_marble_set (n : ℕ) : n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n → n ≥ 210 := by
  sorry

theorem marble_set_existence : ∃ n : ℕ, n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ n = 210 := by
  sorry

end NUMINAMATH_CALUDE_smallest_marble_set_marble_set_existence_l1332_133206


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1332_133219

theorem hyperbola_condition (m : ℝ) :
  m > 0 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), x^2 / (2 + m) - y^2 / (1 + m) = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1332_133219


namespace NUMINAMATH_CALUDE_largest_integer_less_than_95_with_remainder_5_mod_7_l1332_133263

theorem largest_integer_less_than_95_with_remainder_5_mod_7 :
  ∃ n : ℤ, n < 95 ∧ n % 7 = 5 ∧ ∀ m : ℤ, m < 95 ∧ m % 7 = 5 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_95_with_remainder_5_mod_7_l1332_133263


namespace NUMINAMATH_CALUDE_probability_white_then_red_l1332_133299

/-- Probability of drawing a white marble first and then a red marble from a bag -/
theorem probability_white_then_red (total_marbles : ℕ) (red_marbles : ℕ) (white_marbles : ℕ) :
  total_marbles = red_marbles + white_marbles →
  red_marbles = 4 →
  white_marbles = 6 →
  (white_marbles : ℚ) / (total_marbles : ℚ) * (red_marbles : ℚ) / ((total_marbles - 1) : ℚ) = 4 / 15 :=
by sorry

end NUMINAMATH_CALUDE_probability_white_then_red_l1332_133299


namespace NUMINAMATH_CALUDE_exists_valid_division_l1332_133259

/-- A grid-based figure --/
structure GridFigure where
  cells : ℕ

/-- Represents a division of a grid figure --/
structure Division where
  removed : ℕ
  part1 : ℕ
  part2 : ℕ

/-- Checks if a division is valid for a given grid figure --/
def is_valid_division (g : GridFigure) (d : Division) : Prop :=
  d.removed = 1 ∧ 
  d.part1 = d.part2 ∧
  d.part1 + d.part2 + d.removed = g.cells

/-- Theorem stating that a valid division exists for any grid figure --/
theorem exists_valid_division (g : GridFigure) : 
  ∃ (d : Division), is_valid_division g d :=
sorry

end NUMINAMATH_CALUDE_exists_valid_division_l1332_133259


namespace NUMINAMATH_CALUDE_evenness_of_k_l1332_133287

theorem evenness_of_k (a b n k : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) (hk : 0 < k)
  (h1 : 2^n - 1 = a * b)
  (h2 : (a * b + a - b - 1) % 2^k = 0)
  (h3 : (a * b + a - b - 1) % 2^(k+1) ≠ 0) :
  Even k := by sorry

end NUMINAMATH_CALUDE_evenness_of_k_l1332_133287


namespace NUMINAMATH_CALUDE_three_integers_product_2700_sum_56_l1332_133212

theorem three_integers_product_2700_sum_56 :
  ∃ (a b c : ℕ),
    a > 1 ∧ b > 1 ∧ c > 1 ∧
    Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1 ∧
    a * b * c = 2700 ∧
    a + b + c = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_integers_product_2700_sum_56_l1332_133212


namespace NUMINAMATH_CALUDE_quadratic_completion_l1332_133224

theorem quadratic_completion (p : ℝ) (n : ℝ) : 
  (∀ x, x^2 + p*x + 1/4 = (x+n)^2 - 1/16) → 
  p < 0 → 
  p = -Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l1332_133224


namespace NUMINAMATH_CALUDE_overall_discount_rate_l1332_133251

def bag_marked : ℕ := 200
def shirt_marked : ℕ := 80
def shoes_marked : ℕ := 150
def hat_marked : ℕ := 50
def jacket_marked : ℕ := 220

def bag_sold : ℕ := 120
def shirt_sold : ℕ := 60
def shoes_sold : ℕ := 105
def hat_sold : ℕ := 40
def jacket_sold : ℕ := 165

def total_marked : ℕ := bag_marked + shirt_marked + shoes_marked + hat_marked + jacket_marked
def total_sold : ℕ := bag_sold + shirt_sold + shoes_sold + hat_sold + jacket_sold

theorem overall_discount_rate :
  (1 - (total_sold : ℚ) / total_marked) * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_overall_discount_rate_l1332_133251


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l1332_133280

theorem common_number_in_overlapping_lists (nums : List ℝ) : 
  nums.length = 9 ∧ 
  (nums.take 5).sum / 5 = 7 ∧ 
  (nums.drop 4).sum / 5 = 9 ∧ 
  nums.sum / 9 = 73 / 9 →
  ∃ x ∈ nums.take 5 ∩ nums.drop 4, x = 7 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l1332_133280


namespace NUMINAMATH_CALUDE_box_third_side_length_l1332_133221

/-- Proves that the third side of a rectangular box is 9 cm, given specific conditions. -/
theorem box_third_side_length (num_cubes : ℕ) (cube_volume : ℝ) (side1 side2 : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  side1 = 8 →
  side2 = 9 →
  (num_cubes : ℝ) * cube_volume = side1 * side2 * 9 :=
by sorry

end NUMINAMATH_CALUDE_box_third_side_length_l1332_133221


namespace NUMINAMATH_CALUDE_periodic_sequence_quadratic_root_l1332_133225

def is_periodic (x : ℕ → ℝ) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, x (n + p) = x n

def sequence_property (x : ℕ → ℝ) : Prop :=
  x 0 > 1 ∧ ∀ n : ℕ, x (n + 1) = 1 / (x n - ⌊x n⌋)

def is_quadratic_root (r : ℝ) : Prop :=
  ∃ a b c : ℤ, a ≠ 0 ∧ a * r^2 + b * r + c = 0

theorem periodic_sequence_quadratic_root (x : ℕ → ℝ) :
  is_periodic x → sequence_property x → is_quadratic_root (x 0) := by
  sorry

end NUMINAMATH_CALUDE_periodic_sequence_quadratic_root_l1332_133225


namespace NUMINAMATH_CALUDE_inequality_proof_l1332_133255

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  (1/x^2 + x) * (1/y^2 + y) * (1/z^2 + z) ≥ (28/3)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1332_133255


namespace NUMINAMATH_CALUDE_f_composition_equals_pi_squared_plus_one_l1332_133298

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_equals_pi_squared_plus_one :
  f (f (f (-2016))) = Real.pi^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_pi_squared_plus_one_l1332_133298


namespace NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1332_133266

/-- The number of rectangles in a row of length n -/
def rectangles_in_row (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of rectangles in a grid of width w and height h -/
def total_rectangles (w h : ℕ) : ℕ :=
  w * rectangles_in_row h + h * rectangles_in_row w - w * h

theorem rectangles_in_5x4_grid :
  total_rectangles 5 4 = 24 := by sorry

end NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1332_133266


namespace NUMINAMATH_CALUDE_volunteer_distribution_l1332_133235

theorem volunteer_distribution (n : ℕ) (h : n = 5) :
  (n.choose 1) * ((n - 1).choose 2 / 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l1332_133235


namespace NUMINAMATH_CALUDE_march_largest_drop_l1332_133258

/-- Represents the months of interest --/
inductive Month
  | january
  | february
  | march
  | april

/-- The price change for each month --/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.january => -1.25
  | Month.february => 0.75
  | Month.march => -3.00
  | Month.april => 0.25

/-- A month has the largest price drop if its price change is negative and smaller than or equal to all other negative price changes --/
def has_largest_price_drop (m : Month) : Prop :=
  price_change m < 0 ∧
  ∀ n : Month, price_change n < 0 → price_change m ≤ price_change n

theorem march_largest_drop :
  has_largest_price_drop Month.march :=
sorry


end NUMINAMATH_CALUDE_march_largest_drop_l1332_133258


namespace NUMINAMATH_CALUDE_arrangements_without_adjacent_l1332_133220

def total_people : ℕ := 5

theorem arrangements_without_adjacent (A B : ℕ) (h1 : A ≤ total_people) (h2 : B ≤ total_people) (h3 : A ≠ B) :
  (Nat.factorial total_people) - (2 * Nat.factorial (total_people - 1)) = 72 :=
sorry

end NUMINAMATH_CALUDE_arrangements_without_adjacent_l1332_133220


namespace NUMINAMATH_CALUDE_room_length_to_perimeter_ratio_l1332_133286

/-- The ratio of a rectangular room's length to its perimeter -/
theorem room_length_to_perimeter_ratio :
  let length : ℚ := 23
  let width : ℚ := 13
  let perimeter : ℚ := 2 * (length + width)
  (length : ℚ) / perimeter = 23 / 72 := by sorry

end NUMINAMATH_CALUDE_room_length_to_perimeter_ratio_l1332_133286


namespace NUMINAMATH_CALUDE_sqrt_193_between_13_and_14_l1332_133281

theorem sqrt_193_between_13_and_14 : 13 < Real.sqrt 193 ∧ Real.sqrt 193 < 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_193_between_13_and_14_l1332_133281


namespace NUMINAMATH_CALUDE_inscribed_triangle_angle_l1332_133257

/-- A triangle ABC inscribed in the parabola y = x^2 with specific properties -/
structure InscribedTriangle where
  /-- x-coordinate of point A -/
  a : ℝ
  /-- x-coordinate of point C -/
  c : ℝ
  /-- A and B have the same y-coordinate (AB parallel to x-axis) -/
  hParallel : a > 0
  /-- C is closer to x-axis than AB -/
  hCloser : 0 ≤ c ∧ c < a
  /-- Length of AB is 1 shorter than altitude CH -/
  hAltitude : a^2 - c^2 = 2*a + 1

/-- The angle ACB of the inscribed triangle is π/4 -/
theorem inscribed_triangle_angle (t : InscribedTriangle) : 
  Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_angle_l1332_133257


namespace NUMINAMATH_CALUDE_expression_simplification_l1332_133265

theorem expression_simplification (a x : ℝ) (h : a^2 + x^3 > 0) :
  (Real.sqrt (a^2 + x^3) - (x^3 - a^2) / Real.sqrt (a^2 + x^3)) / (a^2 + x^3) =
  2 * a^2 / (a^2 + x^3)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1332_133265


namespace NUMINAMATH_CALUDE_first_year_interest_l1332_133228

def initial_deposit : ℝ := 1000
def first_year_balance : ℝ := 1100
def second_year_increase_rate : ℝ := 0.20
def total_increase_rate : ℝ := 0.32

theorem first_year_interest :
  let second_year_balance := first_year_balance * (1 + second_year_increase_rate)
  second_year_balance = initial_deposit * (1 + total_increase_rate) →
  first_year_balance - initial_deposit = 100 := by
sorry

end NUMINAMATH_CALUDE_first_year_interest_l1332_133228


namespace NUMINAMATH_CALUDE_sin_300_deg_l1332_133250

theorem sin_300_deg : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_deg_l1332_133250


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l1332_133249

def M : Set ℝ := {-1, 1, 2, 4}
def N : Set ℝ := {x | x^2 - 2*x ≥ 3}

theorem intersection_complement_equals_set : M ∩ (Set.univ \ N) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l1332_133249


namespace NUMINAMATH_CALUDE_candies_in_box_l1332_133267

def initial_candies : ℕ := 88
def diana_takes : ℕ := 6
def john_adds : ℕ := 12
def sara_takes : ℕ := 20

theorem candies_in_box : 
  initial_candies - diana_takes + john_adds - sara_takes = 74 :=
by sorry

end NUMINAMATH_CALUDE_candies_in_box_l1332_133267


namespace NUMINAMATH_CALUDE_brendas_mother_cookies_l1332_133217

/-- The number of people Brenda's mother made cookies for -/
def num_people (total_cookies : ℕ) (cookies_per_person : ℕ) : ℕ :=
  total_cookies / cookies_per_person

theorem brendas_mother_cookies : num_people 35 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_brendas_mother_cookies_l1332_133217


namespace NUMINAMATH_CALUDE_sin_difference_product_l1332_133292

theorem sin_difference_product (a b : ℝ) :
  Real.sin (2 * a + b) - Real.sin b = 2 * Real.cos (a + b) * Real.sin a := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_product_l1332_133292


namespace NUMINAMATH_CALUDE_not_all_zero_iff_at_least_one_nonzero_l1332_133236

theorem not_all_zero_iff_at_least_one_nonzero (a b c : ℝ) :
  ¬(a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_not_all_zero_iff_at_least_one_nonzero_l1332_133236


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1332_133260

theorem simplify_and_evaluate :
  (∀ x : ℝ, x = -1 → (-x^2 + 5*x) - (x - 3) - 4*x = 2) ∧
  (∀ m n : ℝ, m = -1/2 ∧ n = 1/3 → 5*(3*m^2*n - m*n^2) - (m*n^2 + 3*m^2*n) = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1332_133260


namespace NUMINAMATH_CALUDE_calculator_correction_l1332_133215

theorem calculator_correction : (0.024 * 3.08) / 0.4 = 0.1848 := by
  sorry

end NUMINAMATH_CALUDE_calculator_correction_l1332_133215


namespace NUMINAMATH_CALUDE_min_omega_value_l1332_133246

theorem min_omega_value (ω : ℝ) (f g : ℝ → ℝ) : 
  (ω > 0) →
  (∀ x, f x = Real.sin (ω * x)) →
  (∀ x, g x = f (x - π / 12)) →
  (∃ k : ℤ, ω * π / 3 - ω * π / 12 = k * π + π / 2) →
  ω ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_value_l1332_133246


namespace NUMINAMATH_CALUDE_ironman_age_is_48_l1332_133248

-- Define the ages as natural numbers
def thor_age : ℕ := 1456
def captain_america_age : ℕ := thor_age / 13
def peter_parker_age : ℕ := captain_america_age / 7
def doctor_strange_age : ℕ := peter_parker_age * 4
def ironman_age : ℕ := peter_parker_age + 32

-- State the theorem
theorem ironman_age_is_48 :
  (thor_age = 13 * captain_america_age) ∧
  (captain_america_age = 7 * peter_parker_age) ∧
  (4 * peter_parker_age = doctor_strange_age) ∧
  (ironman_age = peter_parker_age + 32) ∧
  (thor_age = 1456) →
  ironman_age = 48 := by
  sorry

end NUMINAMATH_CALUDE_ironman_age_is_48_l1332_133248


namespace NUMINAMATH_CALUDE_ellipse_foci_l1332_133207

/-- The equation of an ellipse -/
def is_ellipse (x y : ℝ) : Prop := y^2 / 3 + x^2 / 2 = 1

/-- The coordinates of a point -/
def Point := ℝ × ℝ

/-- The foci of an ellipse -/
def are_foci (p1 p2 : Point) : Prop :=
  p1 = (0, -1) ∧ p2 = (0, 1)

/-- Theorem: The foci of the given ellipse are (0, -1) and (0, 1) -/
theorem ellipse_foci :
  ∃ (p1 p2 : Point), (∀ x y : ℝ, is_ellipse x y → are_foci p1 p2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l1332_133207


namespace NUMINAMATH_CALUDE_equation_solution_l1332_133223

theorem equation_solution : ∃ (x y : ℕ), 1984 * x - 1983 * y = 1985 ∧ x = 27764 ∧ y = 27777 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1332_133223


namespace NUMINAMATH_CALUDE_tourism_max_value_l1332_133205

noncomputable def f (x : ℝ) : ℝ := (51/50) * x - 0.01 * x^2 - Real.log x + Real.log 10

theorem tourism_max_value (x : ℝ) (h1 : 6 < x) (h2 : x ≤ 12) :
  ∃ (y : ℝ), y = f 12 ∧ ∀ z ∈ Set.Ioo 6 12, f z ≤ y := by
  sorry

end NUMINAMATH_CALUDE_tourism_max_value_l1332_133205


namespace NUMINAMATH_CALUDE_person_b_correct_probability_l1332_133242

theorem person_b_correct_probability 
  (prob_a_correct : ℝ) 
  (prob_b_correct_given_a_incorrect : ℝ) 
  (h1 : prob_a_correct = 0.4) 
  (h2 : prob_b_correct_given_a_incorrect = 0.5) : 
  (1 - prob_a_correct) * prob_b_correct_given_a_incorrect = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_person_b_correct_probability_l1332_133242


namespace NUMINAMATH_CALUDE_partner_investment_period_l1332_133218

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where q invests for 16 months, this theorem proves that p invests for 8 months. -/
theorem partner_investment_period (x : ℝ) (t : ℝ) : 
  (7 * x * t) / (5 * x * 16) = 7 / 10 → t = 8 := by
  sorry

end NUMINAMATH_CALUDE_partner_investment_period_l1332_133218


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1332_133201

def f (a : ℝ) (x : ℝ) := x^2 - 2*a*x + 1

theorem necessary_but_not_sufficient (a : ℝ) :
  (a ≤ 0 → ∀ x y, 1 ≤ x → x < y → f a x < f a y) ∧
  (∃ a > 0, ∀ x y, 1 ≤ x → x < y → f a x < f a y) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1332_133201


namespace NUMINAMATH_CALUDE_inscribed_sphere_sum_l1332_133203

/-- A sphere inscribed in a right cone with base radius 15 cm and height 30 cm -/
structure InscribedSphere where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  b : ℝ
  d : ℝ
  base_radius_eq : base_radius = 15
  height_eq : height = 30
  sphere_radius_eq : sphere_radius = b * (Real.sqrt d - 1)

/-- Theorem stating that b + d = 11.75 for the given inscribed sphere -/
theorem inscribed_sphere_sum (s : InscribedSphere) : s.b + s.d = 11.75 := by
  sorry

#check inscribed_sphere_sum

end NUMINAMATH_CALUDE_inscribed_sphere_sum_l1332_133203


namespace NUMINAMATH_CALUDE_liza_final_balance_l1332_133237

/-- Calculates the final balance in Liza's account after a series of transactions --/
def final_balance (initial_balance rent paycheck electricity internet phone : ℤ) : ℤ :=
  initial_balance - rent + paycheck - electricity - internet - phone

/-- Theorem stating that Liza's final account balance is 1563 --/
theorem liza_final_balance :
  final_balance 800 450 1500 117 100 70 = 1563 := by sorry

end NUMINAMATH_CALUDE_liza_final_balance_l1332_133237


namespace NUMINAMATH_CALUDE_parallel_lines_max_distance_l1332_133275

/-- Two parallel lines with maximum distance -/
theorem parallel_lines_max_distance :
  ∃ (k b₁ b₂ : ℝ),
    -- Line equations
    (∀ x y, y = k * x + b₁ ↔ 3 * x + 5 * y + 16 = 0) ∧
    (∀ x y, y = k * x + b₂ ↔ 3 * x + 5 * y - 18 = 0) ∧
    -- Lines pass through given points
    (-2 = k * (-2) + b₁) ∧
    (3 = k * 1 + b₂) ∧
    -- Lines are parallel
    (∀ x y₁ y₂, y₁ = k * x + b₁ ∧ y₂ = k * x + b₂ → y₂ - y₁ = b₂ - b₁) ∧
    -- Distance between lines is maximum
    (∀ k' b₁' b₂',
      ((-2 = k' * (-2) + b₁') ∧ (3 = k' * 1 + b₂')) →
      |b₂ - b₁| / Real.sqrt (1 + k^2) ≥ |b₂' - b₁'| / Real.sqrt (1 + k'^2)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_max_distance_l1332_133275


namespace NUMINAMATH_CALUDE_range_of_x_for_meaningful_sqrt_l1332_133262

theorem range_of_x_for_meaningful_sqrt (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3*x - 2) → x ≥ 2/3 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_for_meaningful_sqrt_l1332_133262


namespace NUMINAMATH_CALUDE_uncovered_cells_bound_l1332_133216

/-- Represents a rectangular board with dominoes -/
structure Board where
  m : ℕ  -- width of the board
  n : ℕ  -- height of the board
  uncovered : ℕ  -- number of uncovered cells

/-- Theorem stating that the number of uncovered cells is less than both mn/4 and mn/5 -/
theorem uncovered_cells_bound (b : Board) : 
  b.uncovered < min (b.m * b.n / 4) (b.m * b.n / 5) := by
  sorry

#check uncovered_cells_bound

end NUMINAMATH_CALUDE_uncovered_cells_bound_l1332_133216


namespace NUMINAMATH_CALUDE_prime_exponent_assignment_l1332_133230

theorem prime_exponent_assignment (k : ℕ) (p : Fin k → ℕ) 
  (h_prime : ∀ i, Prime (p i)) 
  (h_distinct : ∀ i j, i ≠ j → p i ≠ p j) :
  (Finset.univ : Finset (Fin k → Fin k)).card = k ^ k :=
sorry

end NUMINAMATH_CALUDE_prime_exponent_assignment_l1332_133230


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_five_l1332_133276

theorem three_digit_divisible_by_five (n : ℕ) :
  300 ≤ n ∧ n < 400 →
  (n % 5 = 0 ↔ n % 100 = 5 ∧ n / 100 = 3) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_five_l1332_133276


namespace NUMINAMATH_CALUDE_min_probability_alex_dylan_same_team_l1332_133295

/-- The probability that Alex and Dylan are on the same team given that Alex picks one of the cards a or a+7, and Dylan picks the other. -/
def p (a : ℕ) : ℚ :=
  (Nat.choose (32 - a) 2 + Nat.choose (a - 1) 2) / 703

/-- The statement to be proved -/
theorem min_probability_alex_dylan_same_team :
  (∃ a : ℕ, a ≤ 40 ∧ a + 7 ≤ 40 ∧ p a ≥ 1/2) ∧
  (∀ a : ℕ, a ≤ 40 ∧ a + 7 ≤ 40 ∧ p a ≥ 1/2 → p a ≥ 497/703) ∧
  (∃ a : ℕ, a ≤ 40 ∧ a + 7 ≤ 40 ∧ p a = 497/703) :=
sorry

end NUMINAMATH_CALUDE_min_probability_alex_dylan_same_team_l1332_133295


namespace NUMINAMATH_CALUDE_bus_speed_with_stoppages_l1332_133202

/-- Given a bus that travels at 32 km/hr excluding stoppages and stops for 30 minutes per hour,
    the speed of the bus including stoppages is 16 km/hr. -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stop_time : ℝ) 
  (h1 : speed_without_stoppages = 32)
  (h2 : stop_time = 0.5) : 
  speed_without_stoppages * (1 - stop_time) = 16 := by
  sorry

#check bus_speed_with_stoppages

end NUMINAMATH_CALUDE_bus_speed_with_stoppages_l1332_133202


namespace NUMINAMATH_CALUDE_f_derivative_at_pi_half_l1332_133278

noncomputable def f (x : ℝ) : ℝ := x / Real.sin x

theorem f_derivative_at_pi_half :
  deriv f (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_pi_half_l1332_133278


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l1332_133222

theorem no_real_roots_for_nonzero_k (k : ℝ) (h : k ≠ 0) :
  ∀ x : ℝ, x^2 + 2*k*x + 3*k^2 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l1332_133222


namespace NUMINAMATH_CALUDE_triangle_angle_measurement_l1332_133270

theorem triangle_angle_measurement (A B C : ℝ) : 
  A = 70 ∧ 
  B = 2 * C + 30 ∧ 
  A + B + C = 180 →
  C = 80 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measurement_l1332_133270


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1332_133254

/-- Given an arithmetic sequence {a_n} with a_3 = 5 and a_15 = 41, 
    prove that the common difference d is equal to 3. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a3 : a 3 = 5) 
  (h_a15 : a 15 = 41) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1332_133254


namespace NUMINAMATH_CALUDE_trapezoid_area_division_l1332_133296

/-- Given a trapezoid with specific properties, prove that the largest integer less than x^2/50 is 72 -/
theorem trapezoid_area_division (b : ℝ) (h : ℝ) (x : ℝ) : 
  b > 0 ∧ h > 0 ∧
  (b + 12.5) / (b + 37.5) = 3 / 5 ∧
  x > 0 ∧
  (25 + x) * ((x - 25) / 50) = 50 ∧
  x^2 - 75*x + 3125 = 0 →
  ⌊x^2 / 50⌋ = 72 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_division_l1332_133296


namespace NUMINAMATH_CALUDE_expression_simplification_l1332_133239

theorem expression_simplification (x y : ℝ) 
  (hx : x = (Real.sqrt 5 + 1) / 2) 
  (hy : y = (Real.sqrt 5 - 1) / 2) : 
  (x - 2*y)^2 + x*(5*y - x) - 4*y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1332_133239


namespace NUMINAMATH_CALUDE_area_of_trapezoid_TUVW_l1332_133211

/-- Represents a triangle in the problem -/
structure Triangle where
  isIsosceles : Bool
  area : ℝ

/-- Represents the large triangle XYZ -/
def XYZ : Triangle where
  isIsosceles := true
  area := 135

/-- Represents a small triangle -/
def SmallTriangle : Triangle where
  isIsosceles := true
  area := 3

/-- The number of small triangles in XYZ -/
def numSmallTriangles : ℕ := 9

/-- The number of small triangles in trapezoid TUVW -/
def numSmallTrianglesInTUVW : ℕ := 4

/-- The area of trapezoid TUVW -/
def areaTUVW : ℝ := numSmallTrianglesInTUVW * SmallTriangle.area

theorem area_of_trapezoid_TUVW : areaTUVW = 123 := by
  sorry

end NUMINAMATH_CALUDE_area_of_trapezoid_TUVW_l1332_133211


namespace NUMINAMATH_CALUDE_factors_of_2520_l1332_133245

/-- The number of distinct, positive factors of 2520 -/
def num_factors_2520 : ℕ :=
  (Finset.filter (· ∣ 2520) (Finset.range 2521)).card

/-- Theorem stating that the number of distinct, positive factors of 2520 is 48 -/
theorem factors_of_2520 : num_factors_2520 = 48 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_2520_l1332_133245


namespace NUMINAMATH_CALUDE_xiaopang_mom_money_l1332_133243

/-- The price of apples per kilogram -/
def apple_price : ℝ := 5

/-- The amount of money Xiaopang's mom had -/
def total_money : ℝ := 21.5

/-- The amount of apples Xiaopang's mom wanted to buy initially -/
def initial_amount : ℝ := 5

/-- The amount of apples Xiaopang's mom actually bought -/
def actual_amount : ℝ := 4

/-- The amount of money Xiaopang's mom was short for the initial amount -/
def short_amount : ℝ := 3.5

/-- The amount of money Xiaopang's mom had left after buying the actual amount -/
def left_amount : ℝ := 1.5

theorem xiaopang_mom_money :
  total_money = actual_amount * apple_price + left_amount ∧
  total_money = initial_amount * apple_price - short_amount :=
by sorry

end NUMINAMATH_CALUDE_xiaopang_mom_money_l1332_133243


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1332_133210

def C : Finset Nat := {51, 53, 54, 56, 57}

def has_smallest_prime_factor (n : Nat) (s : Finset Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, (Nat.minFac n ≤ Nat.minFac m)

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 54 C := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1332_133210


namespace NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l1332_133297

theorem largest_integer_for_negative_quadratic :
  ∀ m : ℤ, m^2 - 11*m + 24 < 0 → m ≤ 7 ∧ 
  ∃ n : ℤ, n^2 - 11*n + 24 < 0 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_for_negative_quadratic_l1332_133297


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1332_133213

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x^2 + x - 2 < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ x^2 + x - 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1332_133213


namespace NUMINAMATH_CALUDE_root_values_l1332_133274

theorem root_values (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^3 + q * m^2 + r * m + s = 0)
  (h2 : q * m^3 + r * m^2 + s * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_root_values_l1332_133274


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l1332_133285

/-- A rectangle with perimeter 46 and area 108 has a shorter side of 9 -/
theorem rectangle_shorter_side : ∃ (a b : ℝ), 
  a > 0 ∧ b > 0 ∧  -- positive sides
  a ≥ b ∧          -- a is the longer side
  2 * (a + b) = 46 ∧  -- perimeter condition
  a * b = 108 ∧    -- area condition
  b = 9 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l1332_133285


namespace NUMINAMATH_CALUDE_second_run_time_l1332_133253

/-- Represents the time in seconds for various parts of the obstacle course challenge -/
structure ObstacleCourseTime where
  totalSecondRun : ℕ
  doorOpenTime : ℕ

/-- Calculates the time for the second run without backpack -/
def secondRunWithoutBackpack (t : ObstacleCourseTime) : ℕ :=
  t.totalSecondRun - t.doorOpenTime

/-- Theorem stating that for the given times, the second run without backpack takes 801 seconds -/
theorem second_run_time (t : ObstacleCourseTime) 
    (h1 : t.totalSecondRun = 874)
    (h2 : t.doorOpenTime = 73) : 
  secondRunWithoutBackpack t = 801 := by
  sorry

end NUMINAMATH_CALUDE_second_run_time_l1332_133253


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l1332_133264

theorem polynomial_root_sum (A B C D : ℤ) : 
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ z : ℂ, z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36 = 0 ↔ 
      z = r₁ ∨ z = r₂ ∨ z = r₃ ∨ z = r₄ ∨ z = r₅ ∨ z = r₆) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12) →
  B = -76 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l1332_133264


namespace NUMINAMATH_CALUDE_teamA_win_probability_l1332_133208

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  numTeams : Nat
  noTies : Bool
  equalWinChance : Bool
  teamAWonFirst : Bool

/-- Calculates the probability that Team A finishes with more points than Team B -/
def probabilityTeamAWins (tournament : SoccerTournament) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem teamA_win_probability 
  (tournament : SoccerTournament) 
  (h1 : tournament.numTeams = 9)
  (h2 : tournament.noTies = true)
  (h3 : tournament.equalWinChance = true)
  (h4 : tournament.teamAWonFirst = true) :
  probabilityTeamAWins tournament = 9714 / 8192 :=
sorry

end NUMINAMATH_CALUDE_teamA_win_probability_l1332_133208


namespace NUMINAMATH_CALUDE_green_pill_cost_calculation_l1332_133294

def green_pill_cost (total_cost : ℚ) (days : ℕ) (green_daily : ℕ) (pink_daily : ℕ) : ℚ :=
  (total_cost / days + 2 * pink_daily) / (green_daily + pink_daily)

theorem green_pill_cost_calculation :
  let total_cost : ℚ := 600
  let days : ℕ := 10
  let green_daily : ℕ := 2
  let pink_daily : ℕ := 1
  green_pill_cost total_cost days green_daily pink_daily = 62/3 := by
sorry

end NUMINAMATH_CALUDE_green_pill_cost_calculation_l1332_133294


namespace NUMINAMATH_CALUDE_rental_cost_calculation_l1332_133284

/-- Calculates the total cost of renting a truck given the daily rate, mileage rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that the total rental cost for the given conditions is $230. -/
theorem rental_cost_calculation :
  let daily_rate : ℚ := 35
  let mileage_rate : ℚ := 1/4
  let days : ℕ := 3
  let miles : ℕ := 500
  total_rental_cost daily_rate mileage_rate days miles = 230 := by
sorry


end NUMINAMATH_CALUDE_rental_cost_calculation_l1332_133284


namespace NUMINAMATH_CALUDE_triangle_count_is_48_l1332_133241

/-- Represents the configuration of the rectangle and its divisions -/
structure RectangleConfig where
  vertical_divisions : Nat
  horizontal_divisions : Nat
  additional_horizontal_divisions : Nat

/-- Calculates the number of triangles in the described figure -/
def count_triangles (config : RectangleConfig) : Nat :=
  let initial_rectangles := config.vertical_divisions * config.horizontal_divisions
  let initial_triangles := 2 * initial_rectangles
  let additional_rectangles := initial_rectangles * config.additional_horizontal_divisions
  let additional_triangles := 2 * additional_rectangles
  initial_triangles + additional_triangles

/-- The specific configuration described in the problem -/
def problem_config : RectangleConfig :=
  { vertical_divisions := 3
  , horizontal_divisions := 2
  , additional_horizontal_divisions := 2 }

/-- Theorem stating that the number of triangles in the described figure is 48 -/
theorem triangle_count_is_48 : count_triangles problem_config = 48 := by
  sorry


end NUMINAMATH_CALUDE_triangle_count_is_48_l1332_133241


namespace NUMINAMATH_CALUDE_intersectionRangeOfB_l1332_133234

/-- Two lines y = 2x + 1 and y = 3x + b intersect in the third quadrant -/
def linesIntersectInThirdQuadrant (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = 2*x + 1 ∧ y = 3*x + b ∧ x < 0 ∧ y < 0

/-- The range of b for which the lines intersect in the third quadrant -/
theorem intersectionRangeOfB :
  ∀ b : ℝ, linesIntersectInThirdQuadrant b ↔ b > 3/2 :=
sorry

end NUMINAMATH_CALUDE_intersectionRangeOfB_l1332_133234


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l1332_133200

theorem closest_integer_to_cube_root_150 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (150 : ℝ)^(1/3)| ≤ |m - (150 : ℝ)^(1/3)| ∧ n = 5 :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l1332_133200


namespace NUMINAMATH_CALUDE_irrationality_of_root_sum_squares_l1332_133290

theorem irrationality_of_root_sum_squares (a b c : ℤ) (r : ℝ) 
  (h1 : a * r^2 + b * r + c = 0)
  (h2 : a * c ≠ 0) : 
  Irrational (Real.sqrt (r^2 + c^2)) :=
by sorry

end NUMINAMATH_CALUDE_irrationality_of_root_sum_squares_l1332_133290


namespace NUMINAMATH_CALUDE_integral_second_derivative_car_acceleration_l1332_133226

-- Part 1
theorem integral_second_derivative {f : ℝ → ℝ} {a b : ℝ} (h₁ : Continuous (deriv (deriv f))) 
  (h₂ : deriv f a = 0) (h₃ : deriv f b = 0) (h₄ : a < b) :
  f b - f a = ∫ x in a..b, ((a + b) / 2 - x) * deriv (deriv f) x := by sorry

-- Part 2
theorem car_acceleration {f : ℝ → ℝ} {L T : ℝ} (h₁ : f 0 = 0) (h₂ : f T = L) 
  (h₃ : deriv f 0 = 0) (h₄ : deriv f T = 0) (h₅ : T > 0) (h₆ : L > 0) :
  ∃ t : ℝ, t ∈ Set.Icc 0 T ∧ |deriv (deriv f) t| ≥ 4 * L / T^2 := by sorry

end NUMINAMATH_CALUDE_integral_second_derivative_car_acceleration_l1332_133226


namespace NUMINAMATH_CALUDE_sum_of_squares_equation_l1332_133272

theorem sum_of_squares_equation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a^2 + a*b + b^2 = 1)
  (eq2 : b^2 + b*c + c^2 = 3)
  (eq3 : c^2 + c*a + a^2 = 4) :
  a + b + c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equation_l1332_133272


namespace NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l1332_133252

/-- Given a triangle ABC with centroid G, prove that if GA^2 + GB^2 + GC^2 = 58, 
    then AB^2 + AC^2 + BC^2 = 174. -/
theorem triangle_centroid_distance_sum (A B C G : ℝ × ℝ) : 
  (G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) →  -- G is the centroid
  (dist G A)^2 + (dist G B)^2 + (dist G C)^2 = 58 →       -- Given condition
  (dist A B)^2 + (dist A C)^2 + (dist B C)^2 = 174 :=     -- Conclusion to prove
by
  sorry

#check triangle_centroid_distance_sum

end NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l1332_133252


namespace NUMINAMATH_CALUDE_participation_related_to_city_probability_one_from_each_city_l1332_133229

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![60, 40],
    ![30, 70]]

-- Define the K^2 formula
def K_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99.9% certainty
def critical_value : ℚ := 10828 / 1000

-- Theorem for part 1
theorem participation_related_to_city :
  let a := contingency_table 0 0
  let b := contingency_table 0 1
  let c := contingency_table 1 0
  let d := contingency_table 1 1
  K_squared a b c d > critical_value :=
sorry

-- Define the number of people from each city
def city_A_count : ℕ := 4
def city_B_count : ℕ := 2
def total_count : ℕ := city_A_count + city_B_count

-- Theorem for part 2
theorem probability_one_from_each_city :
  (Nat.choose city_A_count 1 * Nat.choose city_B_count 1 : ℚ) / Nat.choose total_count 2 = 8 / 15 :=
sorry

end NUMINAMATH_CALUDE_participation_related_to_city_probability_one_from_each_city_l1332_133229


namespace NUMINAMATH_CALUDE_distribute_and_simplify_l1332_133283

theorem distribute_and_simplify (a b : ℝ) : 3*a*(2*a - b) = 6*a^2 - 3*a*b := by
  sorry

end NUMINAMATH_CALUDE_distribute_and_simplify_l1332_133283


namespace NUMINAMATH_CALUDE_count_ballpoint_pens_l1332_133268

/-- The total number of school supplies -/
def total_supplies : ℕ := 60

/-- The number of pencils -/
def pencils : ℕ := 5

/-- The number of notebooks -/
def notebooks : ℕ := 10

/-- The number of erasers -/
def erasers : ℕ := 32

/-- The number of ballpoint pens -/
def ballpoint_pens : ℕ := total_supplies - (pencils + notebooks + erasers)

theorem count_ballpoint_pens : ballpoint_pens = 13 := by
  sorry

end NUMINAMATH_CALUDE_count_ballpoint_pens_l1332_133268


namespace NUMINAMATH_CALUDE_train_speed_problem_l1332_133273

theorem train_speed_problem (x : ℝ) (v : ℝ) :
  x > 0 →
  (x / v + 2 * x / 20 = 4 * x / 32) →
  v = 8.8 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1332_133273


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1332_133269

-- Define the complex number -1-2i
def z : ℂ := -1 - 2 * Complex.I

-- Theorem stating that the imaginary part of z is -2
theorem imaginary_part_of_z :
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1332_133269


namespace NUMINAMATH_CALUDE_p_plus_q_equals_21_over_2_l1332_133204

theorem p_plus_q_equals_21_over_2 (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 81 = 0)
  (hq : 9*q^3 - 81*q^2 - 243*q + 3645 = 0) : 
  p + q = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_21_over_2_l1332_133204


namespace NUMINAMATH_CALUDE_video_votes_l1332_133289

theorem video_votes (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ) (score : ℤ) : 
  total_votes = likes + dislikes →
  likes = (6 : ℕ) * total_votes / 10 →
  dislikes = (4 : ℕ) * total_votes / 10 →
  score = likes - dislikes →
  score = 150 →
  total_votes = 750 := by
sorry


end NUMINAMATH_CALUDE_video_votes_l1332_133289


namespace NUMINAMATH_CALUDE_equation_real_root_l1332_133247

theorem equation_real_root (x m : ℝ) (i : ℂ) : 
  (∃ x : ℝ, x^2 + (1 - 2*i)*x + 3*m - i = 0) → m = 1/12 :=
by sorry

end NUMINAMATH_CALUDE_equation_real_root_l1332_133247


namespace NUMINAMATH_CALUDE_shaded_area_proof_l1332_133277

theorem shaded_area_proof (t : ℝ) (h : t = 5) : 
  let larger_side := 2 * t - 4
  let smaller_side := 4
  (larger_side ^ 2) - (smaller_side ^ 2) = 20 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l1332_133277


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1332_133244

theorem trigonometric_identity : 
  Real.cos (6 * π / 180) * Real.cos (36 * π / 180) + 
  Real.sin (6 * π / 180) * Real.cos (54 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1332_133244


namespace NUMINAMATH_CALUDE_cyclic_fraction_inequality_l1332_133238

theorem cyclic_fraction_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 2*y) / (z + 2*x + 3*y) + (y + 2*z) / (x + 2*y + 3*z) + (z + 2*x) / (y + 2*z + 3*x) ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_cyclic_fraction_inequality_l1332_133238


namespace NUMINAMATH_CALUDE_certain_value_proof_l1332_133233

theorem certain_value_proof (n : ℤ) (x : ℤ) 
  (h1 : 101 * n^2 ≤ x)
  (h2 : ∀ m : ℤ, 101 * m^2 ≤ x → m ≤ 7) :
  x = 4979 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_proof_l1332_133233


namespace NUMINAMATH_CALUDE_x_squared_coefficient_is_13_l1332_133271

/-- The coefficient of x^2 in the expansion of ((1-x)^3 * (2x^2+1)^5) is 13 -/
theorem x_squared_coefficient_is_13 : 
  let f : Polynomial ℚ := (1 - X)^3 * (2*X^2 + 1)^5
  (f.coeff 2) = 13 := by sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_is_13_l1332_133271


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1332_133282

def polynomial (x : ℝ) : ℝ := 4 * (x^3 - 2*x^2) + 3 * (x^2 - x^3 + 2*x^4) - 5 * (x^4 - 2*x^3)

theorem coefficient_of_x_cubed :
  ∃ (a b c d : ℝ), ∀ x, polynomial x = a*x^4 + 11*x^3 + b*x^2 + c*x + d :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1332_133282


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_3465_is_9_l1332_133232

/-- The largest perfect square factor of 3465 -/
def largest_perfect_square_factor_of_3465 : ℕ := 9

/-- Theorem stating that the largest perfect square factor of 3465 is 9 -/
theorem largest_perfect_square_factor_of_3465_is_9 :
  ∀ n : ℕ, n^2 ∣ 3465 → n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_3465_is_9_l1332_133232


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1332_133293

theorem polynomial_remainder : ∀ x : ℝ, 
  (4 * x^8 - 3 * x^6 - 6 * x^4 + x^3 + 5 * x^2 - 9) = 
  (x - 1) * (4 * x^7 + 4 * x^6 + x^5 - 2 * x^4 - 2 * x^3 + 4 * x^2 + 4 * x + 4) + (-9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1332_133293


namespace NUMINAMATH_CALUDE_linear_function_property_l1332_133279

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- Given a linear function g where g(8) - g(3) = 15, prove that g(20) - g(3) = 51. -/
theorem linear_function_property (g : ℝ → ℝ) (h1 : LinearFunction g) (h2 : g 8 - g 3 = 15) :
  g 20 - g 3 = 51 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l1332_133279
