import Mathlib

namespace NUMINAMATH_CALUDE_travel_time_ratio_l2763_276326

theorem travel_time_ratio : 
  let distance : ℝ := 600
  let initial_time : ℝ := 5
  let new_speed : ℝ := 80
  let new_time : ℝ := distance / new_speed
  new_time / initial_time = 1.5 := by sorry

end NUMINAMATH_CALUDE_travel_time_ratio_l2763_276326


namespace NUMINAMATH_CALUDE_xyz_value_l2763_276341

theorem xyz_value (a b c x y z : ℂ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq1 : a = (b + c) / (x + 1))
  (eq2 : b = (a + c) / (y + 1))
  (eq3 : c = (a + b) / (z + 1))
  (sum_prod : x * y + x * z + y * z = 9)
  (sum : x + y + z = 5) :
  x * y * z = 13 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l2763_276341


namespace NUMINAMATH_CALUDE_scientific_notation_of_105_9_billion_l2763_276318

theorem scientific_notation_of_105_9_billion : 
  (105.9 : ℝ) * 1000000000 = 1.059 * (10 : ℝ)^10 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_105_9_billion_l2763_276318


namespace NUMINAMATH_CALUDE_min_value_theorem_l2763_276368

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  second_term : a 2 = 4
  tenth_sum : S 10 = 110

/-- The theorem statement -/
theorem min_value_theorem (seq : ArithmeticSequence) :
  ∃ (n : ℕ), ∀ (m : ℕ), (seq.S m + 64) / seq.a m ≥ 17 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2763_276368


namespace NUMINAMATH_CALUDE_unique_m_value_l2763_276345

-- Define the set A
def A (m : ℚ) : Set ℚ := {m + 2, 2 * m^2 + m}

-- Theorem statement
theorem unique_m_value : ∃! m : ℚ, 3 ∈ A m ∧ (∀ x ∈ A m, x = m + 2 ∨ x = 2 * m^2 + m) :=
by sorry

end NUMINAMATH_CALUDE_unique_m_value_l2763_276345


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2763_276378

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x^2 + (2 - a) * x - 2 < 0}
  (a = 0 → S = {x : ℝ | x < 1}) ∧
  (-2 < a ∧ a < 0 → S = {x : ℝ | x < 1 ∨ x > -2/a}) ∧
  (a = -2 → S = {x : ℝ | x ≠ 1}) ∧
  (a < -2 → S = {x : ℝ | x < -2/a ∨ x > 1}) ∧
  (a > 0 → S = {x : ℝ | -2/a < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2763_276378


namespace NUMINAMATH_CALUDE_gift_cost_theorem_l2763_276311

def polo_price : ℚ := 26
def necklace_price : ℚ := 83
def game_price : ℚ := 90
def sock_price : ℚ := 7
def book_price : ℚ := 15
def scarf_price : ℚ := 22

def polo_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def game_quantity : ℕ := 1
def sock_quantity : ℕ := 4
def book_quantity : ℕ := 3
def scarf_quantity : ℕ := 2

def sales_tax_rate : ℚ := 13 / 200  -- 6.5%
def book_discount_rate : ℚ := 1 / 10  -- 10%
def rebate : ℚ := 12

def total_cost : ℚ :=
  polo_price * polo_quantity +
  necklace_price * necklace_quantity +
  game_price * game_quantity +
  sock_price * sock_quantity +
  book_price * book_quantity +
  scarf_price * scarf_quantity

def discounted_book_cost : ℚ := book_price * book_quantity * (1 - book_discount_rate)

def total_cost_after_book_discount : ℚ :=
  total_cost - (book_price * book_quantity) + discounted_book_cost

def total_cost_with_tax : ℚ :=
  total_cost_after_book_discount * (1 + sales_tax_rate)

def final_cost : ℚ := total_cost_with_tax - rebate

theorem gift_cost_theorem :
  final_cost = 46352 / 100 := by sorry

end NUMINAMATH_CALUDE_gift_cost_theorem_l2763_276311


namespace NUMINAMATH_CALUDE_five_twos_equal_twentyfour_l2763_276338

theorem five_twos_equal_twentyfour : ∃ (f : ℕ → ℕ → ℕ) (g : ℕ → ℕ → ℕ), 
  f (g 2 2 + 2) (g 2 2) = 24 :=
by sorry

end NUMINAMATH_CALUDE_five_twos_equal_twentyfour_l2763_276338


namespace NUMINAMATH_CALUDE_at_most_one_zero_point_l2763_276350

/-- A decreasing function on a closed interval has at most one zero point -/
theorem at_most_one_zero_point 
  {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b) (h_decr : ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x) :
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0 ∨ ∀ x, a ≤ x → x ≤ b → f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_zero_point_l2763_276350


namespace NUMINAMATH_CALUDE_unique_prime_in_range_l2763_276352

theorem unique_prime_in_range : 
  ∃! n : ℕ, 30 < n ∧ n ≤ 43 ∧ Prime n ∧ n % 9 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_in_range_l2763_276352


namespace NUMINAMATH_CALUDE_truncated_pyramid_edges_and_height_l2763_276333

theorem truncated_pyramid_edges_and_height :
  ∃ (x y z u r s t : ℤ),
    x = 4 * r * t ∧
    y = 4 * s * t ∧
    z = (r - s)^2 - 2 * t^2 ∧
    u = (r - s)^2 + 2 * t^2 ∧
    (x - y)^2 + 2 * z^2 = 2 * u^2 :=
by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_edges_and_height_l2763_276333


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2763_276376

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x : ℝ | x < 6}

theorem complement_union_theorem :
  (Set.univ \ B) ∪ A = {x : ℝ | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2763_276376


namespace NUMINAMATH_CALUDE_fred_final_cards_l2763_276365

/-- The number of baseball cards Fred has after various transactions -/
def fred_cards (initial : ℕ) (given_away : ℕ) (new_cards : ℕ) : ℕ :=
  initial - given_away + new_cards

/-- Theorem stating that Fred ends up with 48 cards given the specific numbers in the problem -/
theorem fred_final_cards : fred_cards 26 18 40 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fred_final_cards_l2763_276365


namespace NUMINAMATH_CALUDE_translation_result_l2763_276330

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  ⟨p.x + dx, p.y⟩

/-- Translates a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  ⟨p.x, p.y + dy⟩

/-- The initial point P(2,3) -/
def initialPoint : Point :=
  ⟨2, 3⟩

/-- The final point after translation -/
def finalPoint : Point :=
  translateVertical (translateHorizontal initialPoint (-3)) (-4)

theorem translation_result :
  finalPoint = ⟨-1, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l2763_276330


namespace NUMINAMATH_CALUDE_units_digit_of_F_F7_l2763_276300

-- Define the modified Fibonacci sequence
def modifiedFib : ℕ → ℕ
  | 0 => 3
  | 1 => 5
  | (n + 2) => modifiedFib (n + 1) + modifiedFib n

-- Function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_F_F7 :
  unitsDigit (modifiedFib (modifiedFib 7)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_F_F7_l2763_276300


namespace NUMINAMATH_CALUDE_linear_function_theorem_l2763_276395

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Define the domain and range conditions
def domain_condition (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1
def range_condition (y : ℝ) : Prop := 1 ≤ y ∧ y ≤ 9

-- Theorem statement
theorem linear_function_theorem (k b : ℝ) :
  (∀ x, domain_condition x → range_condition (linear_function k b x)) →
  ((k = 2 ∧ b = 7) ∨ (k = -2 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l2763_276395


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2763_276323

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_1 + a_4 + a_10 + a_16 + a_19 = 150,
    prove that a_18 - 2a_14 = -30 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 1 + a 4 + a 10 + a 16 + a 19 = 150) : 
    a 18 - 2 * a 14 = -30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2763_276323


namespace NUMINAMATH_CALUDE_system_solution_unique_l2763_276366

theorem system_solution_unique :
  ∃! (x y : ℚ), 3 * x + y = 2 ∧ 2 * x - y = 8 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2763_276366


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_l2763_276359

def original_number : ℕ := 8679921
def divisor : ℕ := 330

theorem sum_of_prime_factors : 
  ∃ (n : ℕ), 
    n ≥ original_number ∧ 
    n % divisor = 0 ∧
    (∀ m : ℕ, m ≥ original_number ∧ m % divisor = 0 → m ≥ n) ∧
    (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id = 284) :=
sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_l2763_276359


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2763_276337

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2763_276337


namespace NUMINAMATH_CALUDE_pairing_ways_eq_5040_l2763_276336

/-- Represents the number of students with each grade -/
structure GradeDistribution where
  grade5 : Nat
  grade4 : Nat
  grade3 : Nat

/-- Calculates the number of ways to form pairs of students with different grades -/
def pairingWays (dist : GradeDistribution) : Nat :=
  Nat.choose dist.grade4 dist.grade5 * Nat.factorial dist.grade5

/-- The given grade distribution in the problem -/
def problemDistribution : GradeDistribution :=
  { grade5 := 6, grade4 := 7, grade3 := 1 }

/-- Theorem stating that the number of pairing ways for the given distribution is 5040 -/
theorem pairing_ways_eq_5040 :
  pairingWays problemDistribution = 5040 := by
  sorry

end NUMINAMATH_CALUDE_pairing_ways_eq_5040_l2763_276336


namespace NUMINAMATH_CALUDE_book_selling_price_l2763_276324

/-- Calculates the selling price of an item given its cost price and profit rate -/
def selling_price (cost_price : ℚ) (profit_rate : ℚ) : ℚ :=
  cost_price * (1 + profit_rate)

/-- Theorem: The selling price of a book with cost price Rs 50 and profit rate 40% is Rs 70 -/
theorem book_selling_price :
  selling_price 50 (40 / 100) = 70 := by
  sorry

end NUMINAMATH_CALUDE_book_selling_price_l2763_276324


namespace NUMINAMATH_CALUDE_possible_days_l2763_276386

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def Anya_lies (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Wednesday ∨ d = Day.Thursday

def Vanya_lies (d : Day) : Prop :=
  d = Day.Thursday ∨ d = Day.Friday ∨ d = Day.Saturday

def Anya_says_Friday (d : Day) : Prop :=
  (Anya_lies d ∧ d ≠ Day.Friday) ∨ (¬Anya_lies d ∧ d = Day.Friday)

def Vanya_says_Tuesday (d : Day) : Prop :=
  (Vanya_lies d ∧ d ≠ Day.Tuesday) ∨ (¬Vanya_lies d ∧ d = Day.Tuesday)

theorem possible_days :
  ∀ d : Day, (Anya_says_Friday d ∧ Vanya_says_Tuesday d) ↔ 
    (d = Day.Tuesday ∨ d = Day.Thursday ∨ d = Day.Friday) :=
by sorry

end NUMINAMATH_CALUDE_possible_days_l2763_276386


namespace NUMINAMATH_CALUDE_corrected_mean_is_89_42857142857143_l2763_276391

def initial_scores : List ℝ := [85, 90, 87, 93, 89, 84, 88]

def corrected_scores : List ℝ := 
  [85, 90, 87 + 5, 93, 89, 84 + 5, 88]

theorem corrected_mean_is_89_42857142857143 : 
  (corrected_scores.sum / corrected_scores.length : ℝ) = 89.42857142857143 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_is_89_42857142857143_l2763_276391


namespace NUMINAMATH_CALUDE_austins_highest_wave_l2763_276389

/-- Represents the height of a surfer and related wave measurements. -/
structure SurferMeasurements where
  surfboard_length : ℝ
  shortest_wave : ℝ
  highest_wave : ℝ
  surfer_height : ℝ

/-- Calculates the height of the highest wave caught by a surfer given the measurements. -/
def highest_wave_height (m : SurferMeasurements) : Prop :=
  m.surfboard_length = 7 ∧
  m.shortest_wave = m.surfboard_length + 3 ∧
  m.shortest_wave = m.surfer_height + 4 ∧
  m.highest_wave = 4 * m.surfer_height + 2 ∧
  m.highest_wave = 26

/-- Theorem stating that the highest wave Austin caught was 26 feet tall. -/
theorem austins_highest_wave :
  ∃ m : SurferMeasurements, highest_wave_height m :=
sorry

end NUMINAMATH_CALUDE_austins_highest_wave_l2763_276389


namespace NUMINAMATH_CALUDE_equation_solution_l2763_276304

theorem equation_solution (x : ℝ) :
  (∃ (n : ℤ), x = π / 18 + 2 * π * n / 9) ∨ (∃ (s : ℤ), x = 2 * π * s / 3) ↔
  (((1 - (Real.cos (15 * x))^7 * (Real.cos (9 * x))^2)^(1/4) = Real.sin (9 * x)) ∧
   Real.sin (9 * x) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2763_276304


namespace NUMINAMATH_CALUDE_walter_seal_time_l2763_276344

/-- Given Walter's zoo visit, prove he spent 13 minutes looking at seals. -/
theorem walter_seal_time : ∀ (S : ℕ), 
  S + 8 * S + 13 = 130 → S = 13 := by
  sorry

end NUMINAMATH_CALUDE_walter_seal_time_l2763_276344


namespace NUMINAMATH_CALUDE_roxy_garden_problem_l2763_276301

def garden_problem (initial_flowering : ℕ) (initial_fruiting_factor : ℕ) 
  (bought_flowering : ℕ) (bought_fruiting : ℕ) 
  (given_fruiting : ℕ) (total_remaining : ℕ) : Prop :=
  let initial_fruiting := initial_flowering * initial_fruiting_factor
  let after_purchase_flowering := initial_flowering + bought_flowering
  let after_purchase_fruiting := initial_fruiting + bought_fruiting
  let remaining_fruiting := after_purchase_fruiting - given_fruiting
  let remaining_flowering := total_remaining - remaining_fruiting
  let given_flowering := after_purchase_flowering - remaining_flowering
  given_flowering = 1

theorem roxy_garden_problem : 
  garden_problem 7 2 3 2 4 21 :=
sorry

end NUMINAMATH_CALUDE_roxy_garden_problem_l2763_276301


namespace NUMINAMATH_CALUDE_hex_to_decimal_l2763_276315

/-- 
Given a hexadecimal number of the form $10k5_{(6)}$ where $k$ is a positive integer,
if this number equals 239 when converted to decimal, then $k = 3$.
-/
theorem hex_to_decimal (k : ℕ+) : 
  (1 * 6^3 + k.val * 6^1 + 5 = 239) → k = 3 := by sorry

end NUMINAMATH_CALUDE_hex_to_decimal_l2763_276315


namespace NUMINAMATH_CALUDE_rectangle_tiling_l2763_276399

theorem rectangle_tiling (n : ℕ) : 
  (∃ (a b : ℕ), 3 * a + 2 * b + 3 * n = 63 ∧ b + n = 20) ↔ 
  n ∈ ({2, 5, 8, 11, 14, 17, 20} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_tiling_l2763_276399


namespace NUMINAMATH_CALUDE_min_value_problem_l2763_276305

theorem min_value_problem (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) :
  (9 * z) / (3 * x + y) + (9 * x) / (y + 3 * z) + (4 * y) / (x + z) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2763_276305


namespace NUMINAMATH_CALUDE_cubic_function_property_l2763_276388

/-- Given a function f(x) = ax³ + bx + 1, prove that if f(a) = 8, then f(-a) = -6 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + b * x + 1
  f a = 8 → f (-a) = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2763_276388


namespace NUMINAMATH_CALUDE_fencing_cost_per_metre_l2763_276349

/-- Proof of fencing cost per metre for a rectangular field -/
theorem fencing_cost_per_metre
  (ratio_length_width : ℚ) -- Ratio of length to width
  (area : ℝ) -- Area of the field in square meters
  (total_cost : ℝ) -- Total cost of fencing
  (h_ratio : ratio_length_width = 3 / 4) -- The ratio of length to width is 3:4
  (h_area : area = 10092) -- The area is 10092 sq. m
  (h_cost : total_cost = 101.5) -- The total cost is 101.5
  : ∃ (length width : ℝ),
    length / width = ratio_length_width ∧
    length * width = area ∧
    (2 * (length + width)) * (total_cost / (2 * (length + width))) = total_cost ∧
    total_cost / (2 * (length + width)) = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_per_metre_l2763_276349


namespace NUMINAMATH_CALUDE_angle_convergence_point_l2763_276371

theorem angle_convergence_point (y : ℝ) : 
  y > 0 ∧ y + y + 140 = 360 → y = 110 := by sorry

end NUMINAMATH_CALUDE_angle_convergence_point_l2763_276371


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l2763_276303

/-- Properties of a regular polygon with 24-degree exterior angles -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (exterior_angle : ℝ),
  exterior_angle = 24 →
  n * exterior_angle = 360 →
  (180 * (n - 2) = 2340 ∧ (n * (n - 3)) / 2 = 90) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l2763_276303


namespace NUMINAMATH_CALUDE_sum_of_roots_of_sum_l2763_276339

/-- Given two quadratic polynomials with the same leading coefficient, 
    if the sum of their four roots is p and their sum has two roots, 
    then the sum of the roots of their sum is p/2 -/
theorem sum_of_roots_of_sum (f g : ℝ → ℝ) (a b₁ b₂ c₁ c₂ p : ℝ) :
  (∀ x, f x = a * x^2 + b₁ * x + c₁) →
  (∀ x, g x = a * x^2 + b₂ * x + c₂) →
  (-b₁ / a - b₂ / a = p) →
  (∃ x y, ∀ z, f z + g z = 2 * a * (z - x) * (z - y)) →
  -(b₁ + b₂) / (2 * a) = p / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_sum_l2763_276339


namespace NUMINAMATH_CALUDE_mrs_hilt_money_left_l2763_276335

def initial_amount : ℕ := 10
def truck_cost : ℕ := 3
def pencil_case_cost : ℕ := 2

theorem mrs_hilt_money_left : 
  initial_amount - (truck_cost + pencil_case_cost) = 5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_money_left_l2763_276335


namespace NUMINAMATH_CALUDE_chord_length_is_16_l2763_276387

/-- Represents a line in polar form -/
structure PolarLine where
  equation : ℝ → ℝ → ℝ

/-- Represents a circle in parametric form -/
structure ParametricCircle where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Calculates the length of a chord on a circle cut by a line -/
noncomputable def chordLength (l : PolarLine) (c : ParametricCircle) : ℝ :=
  sorry

/-- The main theorem stating that the chord length is 16 -/
theorem chord_length_is_16 :
  let l : PolarLine := { equation := λ ρ θ => ρ * Real.sin (θ - Real.pi / 3) - 6 }
  let c : ParametricCircle := { x := λ θ => 10 * Real.cos θ, y := λ θ => 10 * Real.sin θ }
  chordLength l c = 16 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_is_16_l2763_276387


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l2763_276334

theorem chocolate_bar_cost (total_bars : ℕ) (unsold_bars : ℕ) (total_revenue : ℕ) : 
  total_bars = 11 → 
  unsold_bars = 7 → 
  total_revenue = 16 → 
  (total_revenue : ℚ) / ((total_bars - unsold_bars) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l2763_276334


namespace NUMINAMATH_CALUDE_pilot_fish_speed_is_30_l2763_276398

/-- Calculates the speed of a pilot fish given initial conditions -/
def pilotFishSpeed (keanuSpeed : ℝ) (sharkSpeedMultiplier : ℝ) (pilotFishIncreaseFactor : ℝ) : ℝ :=
  let sharkSpeedIncrease := keanuSpeed * (sharkSpeedMultiplier - 1)
  keanuSpeed + pilotFishIncreaseFactor * sharkSpeedIncrease

/-- Theorem stating that under given conditions, the pilot fish's speed is 30 mph -/
theorem pilot_fish_speed_is_30 :
  pilotFishSpeed 20 2 (1/2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_pilot_fish_speed_is_30_l2763_276398


namespace NUMINAMATH_CALUDE_infinite_product_value_l2763_276357

noncomputable def infiniteProduct (n : ℕ) : ℝ := (3 ^ (2 ^ n)) ^ (1 / 2 ^ (n + 1))

theorem infinite_product_value :
  (∏' n, infiniteProduct n) = (9 : ℝ) ^ (1/3) := by sorry

end NUMINAMATH_CALUDE_infinite_product_value_l2763_276357


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2763_276377

theorem parallelogram_base_length
  (area : ℝ)
  (base : ℝ)
  (altitude : ℝ)
  (h1 : area = 98)
  (h2 : altitude = 2 * base)
  (h3 : area = base * altitude) :
  base = 7 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2763_276377


namespace NUMINAMATH_CALUDE_expression_value_theorem_l2763_276351

theorem expression_value_theorem (a b c d m : ℝ) :
  (a = -b) →
  (c * d = 1) →
  (|m| = 5) →
  (2 * (a + b) - 3 * c * d + m = 2 ∨ 2 * (a + b) - 3 * c * d + m = -8) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_theorem_l2763_276351


namespace NUMINAMATH_CALUDE_min_value_of_function_l2763_276325

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  4 * x + 9 / x^2 ≥ 3 * (36 : ℝ)^(1/3) ∧ 
  ∃ y > 0, 4 * y + 9 / y^2 = 3 * (36 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2763_276325


namespace NUMINAMATH_CALUDE_power_product_eq_7776_l2763_276319

theorem power_product_eq_7776 : 3^5 * 2^5 = 7776 := by
  sorry

end NUMINAMATH_CALUDE_power_product_eq_7776_l2763_276319


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l2763_276362

theorem last_three_digits_of_5_to_9000 (h : 5^300 ≡ 1 [MOD 800]) :
  5^9000 ≡ 1 [MOD 800] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l2763_276362


namespace NUMINAMATH_CALUDE_percentage_difference_l2763_276363

theorem percentage_difference : 
  let sixty_percent_of_fifty : ℝ := (60 / 100) * 50
  let fifty_percent_of_thirty : ℝ := (50 / 100) * 30
  sixty_percent_of_fifty - fifty_percent_of_thirty = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2763_276363


namespace NUMINAMATH_CALUDE_function_range_and_inequality_l2763_276353

theorem function_range_and_inequality (a b c m : ℝ) : 
  (∀ x, -x^2 + a*x + b ≤ 0) →
  (∀ x, -x^2 + a*x + b > c - 1 ↔ m - 4 < x ∧ x < m + 1) →
  c = 29/4 := by sorry

end NUMINAMATH_CALUDE_function_range_and_inequality_l2763_276353


namespace NUMINAMATH_CALUDE_fifth_point_coordinate_l2763_276384

/-- A sequence of 16 numbers where each number (except the first and last) is the average of its two adjacent numbers -/
def ArithmeticSequence (a : Fin 16 → ℝ) : Prop :=
  a 0 = 2 ∧ 
  a 15 = 47 ∧ 
  ∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2

theorem fifth_point_coordinate (a : Fin 16 → ℝ) (h : ArithmeticSequence a) : a 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fifth_point_coordinate_l2763_276384


namespace NUMINAMATH_CALUDE_unique_root_in_unit_interval_l2763_276367

theorem unique_root_in_unit_interval :
  ∃! α : ℝ, |α| < 1 ∧ α^3 - 2*α + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_root_in_unit_interval_l2763_276367


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_equals_one_l2763_276343

/-- Two vectors a and b in R² -/
def a : ℝ × ℝ := (3, 1)
def b : ℝ → ℝ × ℝ := λ x ↦ (x, -3)

/-- The dot product of two vectors in R² -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem: If a and b are perpendicular, then x = 1 -/
theorem perpendicular_vectors_x_equals_one :
  (∃ x : ℝ, dot_product a (b x) = 0) → 
  (∃ x : ℝ, b x = (1, -3)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_equals_one_l2763_276343


namespace NUMINAMATH_CALUDE_boat_license_combinations_l2763_276314

/-- The number of possible letters for a boat license. -/
def num_letters : ℕ := 4

/-- The number of digits in a boat license. -/
def num_digits : ℕ := 6

/-- The number of possible digits for each position (0-9). -/
def digits_per_position : ℕ := 10

/-- The total number of possible boat license combinations. -/
def total_combinations : ℕ := num_letters * (digits_per_position ^ num_digits)

/-- Theorem stating that the total number of boat license combinations is 4,000,000. -/
theorem boat_license_combinations :
  total_combinations = 4000000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_combinations_l2763_276314


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2763_276328

theorem tan_alpha_value (h : Real.tan (π - π/4) = 1/6) : Real.tan α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2763_276328


namespace NUMINAMATH_CALUDE_rectangle_area_l2763_276361

/-- Proves that a rectangle with width to height ratio of 7:5 and perimeter 48 cm has an area of 140 cm² -/
theorem rectangle_area (width height : ℝ) : 
  width / height = 7 / 5 →
  2 * (width + height) = 48 →
  width * height = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2763_276361


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l2763_276374

theorem min_sum_dimensions (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 → a * b * c = 3003 → a + b + c ≥ 57 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l2763_276374


namespace NUMINAMATH_CALUDE_investment_proportional_to_profit_share_q_investment_l2763_276354

/-- Represents the investment and profit share of an investor -/
structure Investor where
  investment : ℕ
  profitShare : ℕ

/-- Given two investors with their investments and profit shares, 
    proves that their investments are proportional to their profit shares -/
theorem investment_proportional_to_profit_share 
  (p q : Investor) 
  (h1 : p.investment = 500000) 
  (h2 : p.profitShare = 2) 
  (h3 : q.profitShare = 4) : 
  q.investment = 1000000 := by
sorry

/-- Main theorem that proves Q's investment given P's investment and their profit sharing ratio -/
theorem q_investment 
  (p q : Investor) 
  (h1 : p.investment = 500000) 
  (h2 : p.profitShare = 2) 
  (h3 : q.profitShare = 4) : 
  q.investment = 1000000 := by
sorry

end NUMINAMATH_CALUDE_investment_proportional_to_profit_share_q_investment_l2763_276354


namespace NUMINAMATH_CALUDE_three_hour_therapy_charge_l2763_276370

def therapy_charge (first_hour_rate : ℕ) (additional_hour_rate : ℕ) (hours : ℕ) : ℕ :=
  first_hour_rate + (hours - 1) * additional_hour_rate

theorem three_hour_therapy_charge :
  ∀ (first_hour_rate additional_hour_rate : ℕ),
    first_hour_rate = additional_hour_rate + 20 →
    therapy_charge first_hour_rate additional_hour_rate 5 = 300 →
    therapy_charge first_hour_rate additional_hour_rate 3 = 188 :=
by
  sorry

#check three_hour_therapy_charge

end NUMINAMATH_CALUDE_three_hour_therapy_charge_l2763_276370


namespace NUMINAMATH_CALUDE_johns_wrong_marks_l2763_276355

/-- Proves that John's wrongly entered marks are 102 given the conditions of the problem -/
theorem johns_wrong_marks (n : ℕ) (actual_marks wrong_marks : ℝ) 
  (h1 : n = 80)  -- Number of students in the class
  (h2 : actual_marks = 62)  -- John's actual marks
  (h3 : (wrong_marks - actual_marks) / n = 1/2)  -- Average increase due to wrong entry
  : wrong_marks = 102 := by
  sorry

end NUMINAMATH_CALUDE_johns_wrong_marks_l2763_276355


namespace NUMINAMATH_CALUDE_number_of_children_l2763_276392

/-- The number of children in a family satisfying certain conditions -/
theorem number_of_children : ∃ (T : ℕ), T < 19 ∧ (∃ (S C B : ℕ), S = 3 * C ∧ B = 2 * S ∧ T = B + S + 1 ∧ T = 10) := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l2763_276392


namespace NUMINAMATH_CALUDE_no_real_solutions_l2763_276381

theorem no_real_solutions : ¬∃ (x y : ℝ), x^2 + 3*y^2 - 8*x - 12*y + 36 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2763_276381


namespace NUMINAMATH_CALUDE_fraction_to_decimal_subtraction_l2763_276340

theorem fraction_to_decimal_subtraction : (3 : ℚ) / 40 - 0.005 = 0.070 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_subtraction_l2763_276340


namespace NUMINAMATH_CALUDE_max_point_difference_is_n_l2763_276360

/-- A soccer tournament with n teams -/
structure SoccerTournament where
  n : ℕ  -- number of teams
  n_pos : 0 < n  -- number of teams is positive

/-- The result of a match between two teams -/
inductive MatchResult
  | Win
  | Loss
  | Draw

/-- Points awarded for each match result -/
def pointsForResult (result : MatchResult) : ℕ :=
  match result with
  | MatchResult.Win => 2
  | MatchResult.Loss => 0
  | MatchResult.Draw => 1

/-- The maximum possible difference in points between adjacent teams -/
def maxPointDifference (tournament : SoccerTournament) : ℕ :=
  tournament.n

theorem max_point_difference_is_n (tournament : SoccerTournament) :
  ∃ (team1 team2 : ℕ),
    team1 < tournament.n ∧
    team2 < tournament.n ∧
    team1 + 1 = team2 ∧
    ∃ (points1 points2 : ℕ),
      points1 - points2 = maxPointDifference tournament :=
by
  sorry

#check max_point_difference_is_n

end NUMINAMATH_CALUDE_max_point_difference_is_n_l2763_276360


namespace NUMINAMATH_CALUDE_oranges_used_proof_l2763_276385

/-- Calculates the total number of oranges used to make juice -/
def total_oranges (oranges_per_glass : ℕ) (glasses : ℕ) : ℕ :=
  oranges_per_glass * glasses

/-- Proves that the total number of oranges used is 30 -/
theorem oranges_used_proof (oranges_per_glass : ℕ) (glasses : ℕ) 
  (h1 : oranges_per_glass = 3) 
  (h2 : glasses = 10) : 
  total_oranges oranges_per_glass glasses = 30 := by
sorry

end NUMINAMATH_CALUDE_oranges_used_proof_l2763_276385


namespace NUMINAMATH_CALUDE_carpet_price_not_152_l2763_276308

/-- Represents the price of a flying carpet over time -/
structure CarpetPrice where
  /-- The initial price of the carpet in dinars -/
  initial : ℕ
  /-- The number of years the price increases -/
  years : ℕ
  /-- The year in which the price triples (1-indexed) -/
  tripleYear : ℕ

/-- Calculates the final price of the carpet given the initial conditions -/
def finalPrice (c : CarpetPrice) : ℕ :=
  let priceBeforeTriple := c.initial + c.tripleYear - 1
  let priceAfterTriple := 3 * priceBeforeTriple
  priceAfterTriple + (c.years - c.tripleYear)

/-- Theorem stating that the final price cannot be 152 dinars given the conditions -/
theorem carpet_price_not_152 (c : CarpetPrice) 
  (h1 : c.initial = 1)
  (h2 : c.years = 99)
  (h3 : c.tripleYear > 0)
  (h4 : c.tripleYear ≤ c.years) :
  finalPrice c ≠ 152 := by
  sorry

#eval finalPrice { initial := 1, years := 99, tripleYear := 27 }
#eval finalPrice { initial := 1, years := 99, tripleYear := 26 }

end NUMINAMATH_CALUDE_carpet_price_not_152_l2763_276308


namespace NUMINAMATH_CALUDE_fair_coin_five_tosses_l2763_276313

/-- A fair coin is a coin that has an equal probability of landing on either side when tossed. -/
def FairCoin : Type := Unit

/-- The probability of a fair coin landing on one specific side in a single toss. -/
def singleTossProbability (coin : FairCoin) : ℚ := 1 / 2

/-- The number of tosses. -/
def numTosses : ℕ := 5

/-- The probability of a fair coin landing on the same side for a given number of tosses. -/
def sameSideProbability (coin : FairCoin) (n : ℕ) : ℚ :=
  (singleTossProbability coin) ^ n

theorem fair_coin_five_tosses (coin : FairCoin) :
  sameSideProbability coin numTosses = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_five_tosses_l2763_276313


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l2763_276307

theorem line_circle_intersection_range (a : ℝ) :
  (∃ x y : ℝ, x - y - a = 0 ∧ (x - 1)^2 + y^2 = 2) →
  -1 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l2763_276307


namespace NUMINAMATH_CALUDE_company_blocks_l2763_276379

/-- Given a company with the following properties:
  - The total amount for gifts is $4000
  - Each gift costs $4
  - There are approximately 100 workers per block
  Prove that the number of blocks in the company is 10 -/
theorem company_blocks (total_amount : ℕ) (gift_cost : ℕ) (workers_per_block : ℕ) : 
  total_amount = 4000 →
  gift_cost = 4 →
  workers_per_block = 100 →
  (total_amount / gift_cost) / workers_per_block = 10 := by
  sorry

end NUMINAMATH_CALUDE_company_blocks_l2763_276379


namespace NUMINAMATH_CALUDE_differential_of_y_l2763_276372

noncomputable def y (x : ℝ) : ℝ := Real.arctan (Real.sinh x) + (Real.sinh x) * Real.log (Real.cosh x)

theorem differential_of_y (x : ℝ) :
  deriv y x = Real.cosh x * (1 + Real.log (Real.cosh x)) :=
by sorry

end NUMINAMATH_CALUDE_differential_of_y_l2763_276372


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2763_276306

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2763_276306


namespace NUMINAMATH_CALUDE_remainder_of_7n_mod_4_l2763_276390

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 2) : (7 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_7n_mod_4_l2763_276390


namespace NUMINAMATH_CALUDE_number_proof_l2763_276329

theorem number_proof (N p q : ℝ) 
  (h1 : N / p = 6)
  (h2 : N / q = 18)
  (h3 : p - q = 1 / 3) : 
  N = 3 := by
sorry

end NUMINAMATH_CALUDE_number_proof_l2763_276329


namespace NUMINAMATH_CALUDE_unique_integer_proof_l2763_276327

theorem unique_integer_proof : ∃! n : ℕ+, 
  (∃ k : ℕ, n = 18 * k) ∧ 
  (24.7 < (n : ℝ).sqrt ∧ (n : ℝ).sqrt < 25) :=
by
  use 612
  sorry

end NUMINAMATH_CALUDE_unique_integer_proof_l2763_276327


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l2763_276397

/-- Linear function f(x) = -2x - 7 -/
def f (x : ℝ) : ℝ := -2 * x - 7

theorem y1_less_than_y2 (x₁ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f (x₁ - 1) = y₂) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l2763_276397


namespace NUMINAMATH_CALUDE_exists_special_multiple_l2763_276346

/-- A function that returns true if all digits of a natural number are in the set {0, 1, 8, 9} -/
def valid_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ ({0, 1, 8, 9} : Set ℕ)

/-- The main theorem stating the existence of a number with the required properties -/
theorem exists_special_multiple : ∃ n : ℕ, 
  2003 ∣ n ∧ n < 10^11 ∧ valid_digits n :=
sorry

end NUMINAMATH_CALUDE_exists_special_multiple_l2763_276346


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2763_276331

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 - 5 * x) = 8 → x = -12 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2763_276331


namespace NUMINAMATH_CALUDE_exists_special_subset_l2763_276342

/-- Given a set of 40 elements and a function that maps each 19-element subset to a unique element (common friend), 
    there exists a 20-element subset M₀ such that for all a ∈ M₀, the common friend of M₀ \ {a} is not a. -/
theorem exists_special_subset (I : Finset Nat) (f : Finset Nat → Nat) : 
  I.card = 40 → 
  (∀ A : Finset Nat, A ⊆ I → A.card = 19 → f A ∈ I) →
  (∀ A : Finset Nat, A ⊆ I → A.card = 19 → f A ∉ A) →
  ∃ M₀ : Finset Nat, M₀ ⊆ I ∧ M₀.card = 20 ∧ 
    ∀ a ∈ M₀, f (M₀ \ {a}) ≠ a := by
  sorry

end NUMINAMATH_CALUDE_exists_special_subset_l2763_276342


namespace NUMINAMATH_CALUDE_distance_on_line_l2763_276382

/-- Given two points (a, b) and (c, d) on a line y = mx + k, 
    the distance between them is |a - c|√(1 + m²) -/
theorem distance_on_line (m k a b c d : ℝ) :
  b = m * a + k →
  d = m * c + k →
  Real.sqrt ((c - a)^2 + (d - b)^2) = |a - c| * Real.sqrt (1 + m^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_on_line_l2763_276382


namespace NUMINAMATH_CALUDE_expression_equality_l2763_276394

theorem expression_equality : 3 * 257 + 4 * 257 + 2 * 257 + 258 = 2571 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2763_276394


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2763_276309

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₂.a * l₁.b

/-- The problem statement -/
theorem parallel_lines_m_value (m : ℝ) :
  let l₁ : Line := ⟨m - 2, -1, 5⟩
  let l₂ : Line := ⟨m - 2, 3 - m, 2⟩
  parallel l₁ l₂ → m = 2 ∨ m = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_m_value_l2763_276309


namespace NUMINAMATH_CALUDE_right_triangle_area_l2763_276393

theorem right_triangle_area (DE DF : ℝ) (h1 : DE = 30) (h2 : DF = 24) : 
  (1/2) * DE * DF = 360 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2763_276393


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2763_276332

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = x * y) :
  x + 2 * y ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2763_276332


namespace NUMINAMATH_CALUDE_order_of_trig_values_l2763_276312

theorem order_of_trig_values :
  let a := Real.tan (70 * π / 180)
  let b := Real.sin (25 * π / 180)
  let c := Real.cos (25 * π / 180)
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_order_of_trig_values_l2763_276312


namespace NUMINAMATH_CALUDE_inequality_solution_l2763_276396

theorem inequality_solution (x : ℝ) :
  (1 + x) / 2 - (2 * x + 1) / 3 ≤ 1 ↔ x ≥ -5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2763_276396


namespace NUMINAMATH_CALUDE_arc_length_30_degree_sector_l2763_276320

/-- The length of an arc in a sector with radius 1 cm and central angle 30° is π/6 cm. -/
theorem arc_length_30_degree_sector (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 1 → θ = 30 * π / 180 → l = r * θ → l = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_30_degree_sector_l2763_276320


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2763_276310

theorem smallest_n_congruence : ∃! n : ℕ+, 
  (∀ m : ℕ+, 5 * m ≡ 2023 [MOD 26] → n ≤ m) ∧ 
  5 * n ≡ 2023 [MOD 26] := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2763_276310


namespace NUMINAMATH_CALUDE_cricket_team_size_l2763_276373

/-- The number of players on a cricket team satisfying specific conditions -/
theorem cricket_team_size :
  ∀ (total_players throwers non_throwers right_handed : ℕ),
    throwers = 37 →
    non_throwers = total_players - throwers →
    right_handed = 51 →
    right_handed = throwers + (2 * non_throwers / 3) →
    total_players = 58 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l2763_276373


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l2763_276380

theorem factorization_difference_of_squares (m n : ℝ) : (m + n)^2 - (m - n)^2 = 4 * m * n := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l2763_276380


namespace NUMINAMATH_CALUDE_cubelets_one_color_count_l2763_276364

/-- Represents a cube divided into cubelets -/
structure CubeletCube where
  size : Nat
  total_cubelets : Nat
  painted_faces : Fin 3 → Fin 6

/-- The number of cubelets painted with exactly one color -/
def cubelets_with_one_color (c : CubeletCube) : Nat :=
  6 * (c.size - 2) * (c.size - 2)

/-- Theorem: In a 6x6x6 cube painted as described, 96 cubelets are painted with exactly one color -/
theorem cubelets_one_color_count :
  ∀ c : CubeletCube, c.size = 6 → c.total_cubelets = 216 → cubelets_with_one_color c = 96 := by
  sorry

end NUMINAMATH_CALUDE_cubelets_one_color_count_l2763_276364


namespace NUMINAMATH_CALUDE_min_value_problem_l2763_276348

theorem min_value_problem (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_6 : x + y + z = 6) : 
  (x^2 + 2*y^2)/(x + y) + (x^2 + 2*z^2)/(x + z) + (y^2 + 2*z^2)/(y + z) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2763_276348


namespace NUMINAMATH_CALUDE_line_points_comparison_l2763_276302

theorem line_points_comparison (m n b : ℝ) : 
  (m = -3 * (-2) + b) → (n = -3 * 3 + b) → m > n := by
  sorry

end NUMINAMATH_CALUDE_line_points_comparison_l2763_276302


namespace NUMINAMATH_CALUDE_inconsistent_inventory_report_max_consistent_statements_no_more_than_three_consistent_l2763_276347

theorem inconsistent_inventory_report (n : ℕ) (h_n : n ≥ 1000) : 
  ¬(n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 2 ∧ n % 5 = 2) :=
sorry

theorem max_consistent_statements : 
  ∃ (n : ℕ), n ≥ 1000 ∧ 
  ((n % 2 = 1 ∧ n % 3 = 1 ∧ n % 5 = 2) ∨
   (n % 3 = 1 ∧ n % 4 = 2 ∧ n % 5 = 2)) :=
sorry

theorem no_more_than_three_consistent (n : ℕ) (h_n : n ≥ 1000) :
  ¬∃ (a b c d : Bool), a ∧ b ∧ c ∧ d ∧
  (a → n % 2 = 1) ∧
  (b → n % 3 = 1) ∧
  (c → n % 4 = 2) ∧
  (d → n % 5 = 2) ∧
  (a.toNat + b.toNat + c.toNat + d.toNat > 3) :=
sorry

end NUMINAMATH_CALUDE_inconsistent_inventory_report_max_consistent_statements_no_more_than_three_consistent_l2763_276347


namespace NUMINAMATH_CALUDE_impossible_shape_l2763_276316

/-- Represents a square sheet of paper -/
structure Paper :=
  (side : ℝ)
  (is_square : side > 0)

/-- Represents a folded paper -/
structure FoldedPaper :=
  (paper : Paper)
  (folds : ℕ)
  (is_valid : folds ≤ 2)

/-- Represents a shape cut from the paper -/
inductive Shape
  | CrossesBothFolds
  | CrossesOneFold
  | CrossesNoFolds
  | ContainsCenter

/-- Represents a cut made on the folded paper -/
structure Cut :=
  (folded_paper : FoldedPaper)
  (resulting_shape : Shape)

/-- Theorem stating that a shape crossing both folds without containing the center is impossible -/
theorem impossible_shape (p : Paper) (fp : FoldedPaper) (c : Cut) :
  fp.paper = p →
  fp.folds = 2 →
  c.folded_paper = fp →
  c.resulting_shape = Shape.CrossesBothFolds →
  ¬(c.resulting_shape = Shape.ContainsCenter) →
  False :=
sorry

end NUMINAMATH_CALUDE_impossible_shape_l2763_276316


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2763_276356

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let roots := {x : ℝ | f x = 0}
  (∀ x ∈ roots, f x = 0) →
  (∃ x₁ x₂ : ℝ, roots = {x₁, x₂}) →
  (∀ x, f x = 0 ↔ x^2 + 5*x - 24 = 4*x + 38) →
  ∃ x₁ x₂ : ℝ, x₁ + x₂ = -1 ∧ roots = {x₁, x₂} :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2763_276356


namespace NUMINAMATH_CALUDE_ellipse_tangent_existence_l2763_276375

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is outside the ellipse -/
def isOutside (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) > 1

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Checks if a line is tangent to the ellipse -/
def isTangent (e : Ellipse) (l : Line) : Prop :=
  ∃ θ : ℝ, l.p2.x = e.a * Real.cos θ ∧ l.p2.y = e.b * Real.sin θ ∧
    (l.p1.x * l.p2.x / e.a^2) + (l.p1.y * l.p2.y / e.b^2) = 1

/-- Main theorem: For any ellipse and point outside it, there exist two tangent lines -/
theorem ellipse_tangent_existence (e : Ellipse) (p : Point) (h : isOutside e p) :
  ∃ l1 l2 : Line, l1 ≠ l2 ∧ l1.p1 = p ∧ l2.p1 = p ∧ isTangent e l1 ∧ isTangent e l2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_tangent_existence_l2763_276375


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2763_276321

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2763_276321


namespace NUMINAMATH_CALUDE_physics_majors_consecutive_probability_l2763_276358

/-- The number of people sitting at the round table -/
def total_people : ℕ := 10

/-- The number of physics majors -/
def physics_majors : ℕ := 3

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of chemistry majors -/
def chemistry_majors : ℕ := 2

/-- The number of biology majors -/
def biology_majors : ℕ := 1

/-- The probability of all physics majors sitting in consecutive seats -/
def consecutive_physics_probability : ℚ := 1 / 24

theorem physics_majors_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let valid_arrangements := 3 * Nat.factorial (total_people - physics_majors)
  consecutive_physics_probability = (valid_arrangements : ℚ) / total_arrangements :=
sorry

end NUMINAMATH_CALUDE_physics_majors_consecutive_probability_l2763_276358


namespace NUMINAMATH_CALUDE_quadratic_function_coefficient_l2763_276322

theorem quadratic_function_coefficient (a b c : ℤ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (2, -3) = (-(b / (2 * a)), -(b^2 - 4 * a * c) / (4 * a)) →
  1 = a * 0^2 + b * 0 + c →
  6 = a * 5^2 + b * 5 + c →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_coefficient_l2763_276322


namespace NUMINAMATH_CALUDE_average_difference_l2763_276369

theorem average_difference : 
  let m := (12 + 15 + 9 + 14 + 10) / 5
  let n := (24 + 8 + 8 + 12) / 4
  n - m = 1 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l2763_276369


namespace NUMINAMATH_CALUDE_corner_sum_is_ten_l2763_276317

/-- Represents a Go board as a function from coordinates to real numbers -/
def GoBoard : Type := Fin 18 → Fin 18 → ℝ

/-- The property that any 2x2 square on the board sums to 10 -/
def valid_board (board : GoBoard) : Prop :=
  ∀ i j, i < 17 → j < 17 →
    board i j + board (i+1) j + board i (j+1) + board (i+1) (j+1) = 10

/-- The sum of the four corner squares -/
def corner_sum (board : GoBoard) : ℝ :=
  board 0 0 + board 0 17 + board 17 0 + board 17 17

/-- Theorem: For any valid Go board, the sum of the four corners is 10 -/
theorem corner_sum_is_ten (board : GoBoard) (h : valid_board board) :
  corner_sum board = 10 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_ten_l2763_276317


namespace NUMINAMATH_CALUDE_snowfall_difference_theorem_l2763_276383

/-- Snowfall data for various mountains -/
structure SnowfallData where
  bald_mountain : Real
  billy_mountain : Real
  mount_pilot : Real
  rockstone_peak : Real
  sunset_ridge : Real

/-- Conversion factors -/
def meters_to_cm : Real := 100
def mm_to_cm : Real := 0.1

/-- Calculate total snowfall difference -/
def snowfall_difference (data : SnowfallData) : Real :=
  (data.billy_mountain * meters_to_cm +
   data.mount_pilot +
   data.rockstone_peak * mm_to_cm +
   data.sunset_ridge * meters_to_cm) -
  (data.bald_mountain * meters_to_cm)

/-- Theorem stating the snowfall difference -/
theorem snowfall_difference_theorem (data : SnowfallData)
  (h1 : data.bald_mountain = 1.5)
  (h2 : data.billy_mountain = 3.5)
  (h3 : data.mount_pilot = 126)
  (h4 : data.rockstone_peak = 5250)
  (h5 : data.sunset_ridge = 2.25) :
  snowfall_difference data = 1076 := by
  sorry

end NUMINAMATH_CALUDE_snowfall_difference_theorem_l2763_276383
