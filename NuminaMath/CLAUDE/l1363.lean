import Mathlib

namespace NUMINAMATH_CALUDE_sum_2012_terms_eq_4021_l1363_136354

/-- A sequence where each term (after the second) is the sum of its previous and next terms -/
def SpecialSequence (a₀ a₁ : ℤ) : ℕ → ℤ
  | 0 => a₀
  | 1 => a₁
  | n + 2 => SpecialSequence a₀ a₁ (n + 1) + SpecialSequence a₀ a₁ n

/-- The sum of the first n terms of the special sequence -/
def SequenceSum (a₀ a₁ : ℤ) (n : ℕ) : ℤ :=
  (List.range n).map (SpecialSequence a₀ a₁) |>.sum

theorem sum_2012_terms_eq_4021 :
  SequenceSum 2010 2011 2012 = 4021 := by
  sorry

end NUMINAMATH_CALUDE_sum_2012_terms_eq_4021_l1363_136354


namespace NUMINAMATH_CALUDE_events_related_confidence_l1363_136369

-- Define the confidence level
def confidence_level : ℝ := 0.95

-- Define the critical value for 95% confidence
def critical_value : ℝ := 3.841

-- Define the relationship between events A and B
def events_related (K : ℝ) : Prop := K^2 > critical_value

-- Theorem statement
theorem events_related_confidence (K : ℝ) :
  events_related K ↔ confidence_level = 0.95 :=
sorry

end NUMINAMATH_CALUDE_events_related_confidence_l1363_136369


namespace NUMINAMATH_CALUDE_total_weight_carrots_cucumbers_l1363_136310

theorem total_weight_carrots_cucumbers : 
  ∀ (weight_carrots : ℝ) (weight_ratio : ℝ),
    weight_carrots = 250 →
    weight_ratio = 2.5 →
    weight_carrots + weight_ratio * weight_carrots = 875 :=
by
  sorry

end NUMINAMATH_CALUDE_total_weight_carrots_cucumbers_l1363_136310


namespace NUMINAMATH_CALUDE_compute_alpha_l1363_136322

variable (α β : ℂ)

theorem compute_alpha (h1 : (α + β).re > 0)
                       (h2 : (Complex.I * (α - 3 * β)).re > 0)
                       (h3 : β = 4 + 3 * Complex.I) :
  α = 12 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_compute_alpha_l1363_136322


namespace NUMINAMATH_CALUDE_number_difference_l1363_136318

theorem number_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 2 / 3) (h4 : a^3 + b^3 = 945) : b - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1363_136318


namespace NUMINAMATH_CALUDE_consecutive_zeros_in_power_of_five_l1363_136344

theorem consecutive_zeros_in_power_of_five : ∃ n : ℕ, n < 10^6 ∧ 5^n % 10^20 < 10^14 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_zeros_in_power_of_five_l1363_136344


namespace NUMINAMATH_CALUDE_sum_210_72_in_base5_l1363_136365

/-- Converts a decimal number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of base 5 digits to a decimal number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_210_72_in_base5 :
  toBase5 (210 + 72) = [2, 0, 6, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_210_72_in_base5_l1363_136365


namespace NUMINAMATH_CALUDE_mary_sugar_calculation_l1363_136358

/-- The amount of sugar Mary needs to add to her cake -/
def remaining_sugar (total_required : ℕ) (already_added : ℕ) : ℕ :=
  total_required - already_added

/-- Proof that Mary needs to add 11 more cups of sugar -/
theorem mary_sugar_calculation : remaining_sugar 13 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mary_sugar_calculation_l1363_136358


namespace NUMINAMATH_CALUDE_probability_four_twos_value_l1363_136308

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def num_success : ℕ := 4

def probability_exactly_four_twos : ℚ :=
  (num_dice.choose num_success) * 
  ((1 : ℚ) / num_sides) ^ num_success * 
  ((num_sides - 1 : ℚ) / num_sides) ^ (num_dice - num_success)

theorem probability_four_twos_value : 
  probability_exactly_four_twos = 168070 / 16777216 := by sorry

end NUMINAMATH_CALUDE_probability_four_twos_value_l1363_136308


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l1363_136370

-- Part 1
theorem calculation_proof : 
  |(-Real.sqrt 3)| + (3 - Real.pi)^(0 : ℝ) + (1/3)^(-2 : ℝ) = Real.sqrt 3 + 10 := by sorry

-- Part 2
theorem inequality_system_solution :
  {x : ℝ | 3*x + 1 > 2*(x - 1) ∧ x - 1 ≤ 3*x + 3} = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l1363_136370


namespace NUMINAMATH_CALUDE_brick_length_is_20_l1363_136350

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 20

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 10

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 7.5

/-- The length of the wall in meters -/
def wall_length : ℝ := 29

/-- The width of the wall in meters -/
def wall_width : ℝ := 2

/-- The height of the wall in meters -/
def wall_height : ℝ := 0.75

/-- The number of bricks in the wall -/
def number_of_bricks : ℕ := 29000

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem brick_length_is_20 :
  brick_length = 20 :=
by
  have h1 : brick_length * brick_width * brick_height = 
    (wall_length * wall_width * wall_height * m_to_cm^3) / number_of_bricks :=
    sorry
  sorry

end NUMINAMATH_CALUDE_brick_length_is_20_l1363_136350


namespace NUMINAMATH_CALUDE_number_of_placements_is_36_l1363_136321

/-- The number of ways to place 3 men and 4 women into groups -/
def number_of_placements : ℕ :=
  let num_men : ℕ := 3
  let num_women : ℕ := 4
  let num_groups_of_two : ℕ := 2
  let num_groups_of_three : ℕ := 1
  let ways_to_choose_man_for_three : ℕ := Nat.choose num_men 1
  let ways_to_choose_women_for_three : ℕ := Nat.choose num_women 2
  let ways_to_pair_remaining : ℕ := 2
  ways_to_choose_man_for_three * ways_to_choose_women_for_three * ways_to_pair_remaining

/-- Theorem stating that the number of placements is 36 -/
theorem number_of_placements_is_36 : number_of_placements = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_of_placements_is_36_l1363_136321


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1363_136394

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃! (a : ℝ), i * (1 + a * i) = 2 + i :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1363_136394


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1363_136338

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, prove its eccentricity and asymptotes. -/
theorem hyperbola_properties :
  let a := 2
  let b := 2 * Real.sqrt 3
  let c := 4
  let e := c / a
  let asymptote (x : ℝ) := Real.sqrt 3 * x
  (∀ x y : ℝ, x^2/4 - y^2/12 = 1 →
    (e = 2 ∧
    (∀ x : ℝ, y = asymptote x ∨ y = -asymptote x))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1363_136338


namespace NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_root_l1363_136304

theorem arithmetic_sequence_quadratic_root (x y z : ℝ) : 
  (∃ d : ℝ, y = x + d ∧ z = x + 2*d) →  -- arithmetic sequence
  x ≤ y ∧ y ≤ z ∧ z ≤ 10 →             -- ordering condition
  (∃! r : ℝ, z*r^2 + y*r + x = 0) →    -- quadratic has exactly one root
  (∃ r : ℝ, z*r^2 + y*r + x = 0 ∧ r = Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_quadratic_root_l1363_136304


namespace NUMINAMATH_CALUDE_log_problem_l1363_136393

theorem log_problem (y : ℝ) (m : ℝ) 
  (h1 : Real.log 5 / Real.log 8 = y)
  (h2 : Real.log 125 / Real.log 2 = m * y) : 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_log_problem_l1363_136393


namespace NUMINAMATH_CALUDE_sum_of_ab_l1363_136327

theorem sum_of_ab (a b : ℝ) (h : a^2 + b^2 + a^2*b^2 = 4*a*b - 1) :
  a + b = 2 ∨ a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ab_l1363_136327


namespace NUMINAMATH_CALUDE_total_marigolds_sold_l1363_136395

/-- The number of marigolds sold during a three-day sale -/
def marigolds_sold (day1 day2 day3 : ℕ) : ℕ := day1 + day2 + day3

/-- Theorem stating the total number of marigolds sold during the sale -/
theorem total_marigolds_sold :
  let day1 := 14
  let day2 := 25
  let day3 := 2 * day2
  marigolds_sold day1 day2 day3 = 89 := by
  sorry

end NUMINAMATH_CALUDE_total_marigolds_sold_l1363_136395


namespace NUMINAMATH_CALUDE_power_of_product_cube_l1363_136324

theorem power_of_product_cube (x : ℝ) : (2 * x^3)^2 = 4 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_cube_l1363_136324


namespace NUMINAMATH_CALUDE_equation_solution_l1363_136372

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 4) * x = 8 ∧ x = 112 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1363_136372


namespace NUMINAMATH_CALUDE_saras_quarters_l1363_136368

theorem saras_quarters (current_quarters borrowed_quarters : ℕ) 
  (h1 : current_quarters = 512)
  (h2 : borrowed_quarters = 271) :
  current_quarters + borrowed_quarters = 783 := by
  sorry

end NUMINAMATH_CALUDE_saras_quarters_l1363_136368


namespace NUMINAMATH_CALUDE_complex_division_l1363_136357

theorem complex_division (i : ℂ) (h : i^2 = -1) : (1 + 2*i) / i = 2 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l1363_136357


namespace NUMINAMATH_CALUDE_expression_evaluation_l1363_136378

theorem expression_evaluation : 
  (4 * 6) / (12 * 15) * (5 * 12 * 15^2) / (2 * 6 * 5) = 2.5 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1363_136378


namespace NUMINAMATH_CALUDE_runner_problem_l1363_136355

theorem runner_problem (v : ℝ) (h : v > 0) :
  (40 / v = 20 / v + 8) → (40 / (v / 2) = 16) := by
  sorry

end NUMINAMATH_CALUDE_runner_problem_l1363_136355


namespace NUMINAMATH_CALUDE_grade_difference_l1363_136373

theorem grade_difference (a b c : ℕ) : 
  a + b + c = 25 → 
  3 * a + 4 * b + 5 * c = 106 → 
  c - a = 6 := by
  sorry

end NUMINAMATH_CALUDE_grade_difference_l1363_136373


namespace NUMINAMATH_CALUDE_friend_consumption_l1363_136386

def total_people : ℕ := 8
def pizzas : ℕ := 5
def slices_per_pizza : ℕ := 8
def pasta_bowls : ℕ := 2
def garlic_breads : ℕ := 12

def ron_scott_pizza : ℕ := 10
def mark_pizza : ℕ := 2
def sam_pizza : ℕ := 4

def ron_scott_pasta_percent : ℚ := 40 / 100
def ron_scott_mark_garlic_percent : ℚ := 25 / 100

theorem friend_consumption :
  let remaining_friends := total_people - 4
  let remaining_pizza := pizzas * slices_per_pizza - (ron_scott_pizza + mark_pizza + sam_pizza)
  let remaining_pasta_percent := 1 - ron_scott_pasta_percent
  let remaining_garlic_percent := 1 - ron_scott_mark_garlic_percent
  (remaining_pizza / remaining_friends = 6) ∧
  (remaining_pasta_percent / (total_people - 2) = 10 / 100) ∧
  (remaining_garlic_percent * garlic_breads / (total_people - 3) = 1.8) := by
  sorry

end NUMINAMATH_CALUDE_friend_consumption_l1363_136386


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1363_136376

theorem algebraic_simplification (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1363_136376


namespace NUMINAMATH_CALUDE_sum_reciprocals_l1363_136389

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l1363_136389


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1363_136300

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  is_right_triangle 3 4 5 ∧
  ¬is_right_triangle 2 3 4 ∧
  ¬is_right_triangle 4 6 7 ∧
  ¬is_right_triangle 5 11 12 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1363_136300


namespace NUMINAMATH_CALUDE_min_value_theorem_l1363_136367

theorem min_value_theorem (a b : ℝ) (ha : a > 0) 
  (h : ∀ x > 0, (a * x - 2) * (x^2 + b * x - 5) ≥ 0) :
  (∃ (b₀ : ℝ), b₀ + 4 / a = 2 * Real.sqrt 5) ∧ 
  (∀ (b₁ : ℝ), b₁ + 4 / a ≥ 2 * Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1363_136367


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_iff_no_intersection_l1363_136319

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Check if a line is parallel to a plane -/
def isParallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Check if a line intersects with another line -/
def intersects (l1 l2 : Line3D) : Prop :=
  sorry

/-- Get a line in a plane -/
def lineInPlane (p : Plane3D) : Line3D :=
  sorry

theorem line_parallel_to_plane_iff_no_intersection (l : Line3D) (p : Plane3D) :
  isParallel l p ↔ ∀ (l' : Line3D), lineInPlane p = l' → ¬ intersects l l' :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_iff_no_intersection_l1363_136319


namespace NUMINAMATH_CALUDE_lines_always_parallel_l1363_136381

/-- A linear function f(x) = kx + b -/
def f (k b x : ℝ) : ℝ := k * x + b

/-- Line l₁ represented by y = f(x) -/
def l₁ (k b : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = f k b x}

/-- Line l₂ defined as y - y₀ = f(x) - f(x₀) -/
def l₂ (k b x₀ y₀ : ℝ) : Set (ℝ × ℝ) := {(x, y) | y - y₀ = f k b x - f k b x₀}

/-- Point P -/
def P (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)

theorem lines_always_parallel (k b x₀ y₀ : ℝ) 
  (h : P x₀ y₀ ∉ l₁ k b) : 
  ∃ (m : ℝ), ∀ (x y : ℝ), 
    ((x, y) ∈ l₁ k b ↔ y = k * x + m) ∧ 
    ((x, y) ∈ l₂ k b x₀ y₀ ↔ y = k * x + (y₀ - k * x₀)) :=
sorry

end NUMINAMATH_CALUDE_lines_always_parallel_l1363_136381


namespace NUMINAMATH_CALUDE_select_real_coins_l1363_136390

/-- Represents the outcome of a weighing on a balance scale -/
inductive WeighingResult
| Equal : WeighingResult
| LeftHeavier : WeighingResult
| RightHeavier : WeighingResult

/-- Represents a group of coins -/
structure CoinGroup where
  total : Nat
  counterfeit : Nat

/-- Represents a weighing action on the balance scale -/
def weighing (left right : CoinGroup) : WeighingResult :=
  sorry

/-- Represents the process of selecting coins -/
def selectCoins (coins : CoinGroup) (weighings : Nat) : Option (Finset Nat) :=
  sorry

theorem select_real_coins 
  (total_coins : Nat)
  (counterfeit_coins : Nat)
  (max_weighings : Nat)
  (coins_to_select : Nat)
  (h1 : total_coins = 40)
  (h2 : counterfeit_coins = 3)
  (h3 : max_weighings = 3)
  (h4 : coins_to_select = 16)
  (h5 : counterfeit_coins < total_coins) :
  ∃ (selected : Finset Nat), 
    (selected.card = coins_to_select) ∧ 
    (∀ c ∈ selected, c ≤ total_coins - counterfeit_coins) ∧
    (selectCoins ⟨total_coins, counterfeit_coins⟩ max_weighings = some selected) :=
sorry

end NUMINAMATH_CALUDE_select_real_coins_l1363_136390


namespace NUMINAMATH_CALUDE_odd_digits_base4_345_l1363_136384

/-- Converts a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 345 is 4 -/
theorem odd_digits_base4_345 : countOddDigits (toBase4 345) = 4 :=
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_345_l1363_136384


namespace NUMINAMATH_CALUDE_caitlin_uniform_number_l1363_136332

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ Nat.Prime n

theorem caitlin_uniform_number
  (a b c : ℕ)
  (ha : is_two_digit_prime a)
  (hb : is_two_digit_prime b)
  (hc : is_two_digit_prime c)
  (hab : a ≠ b)
  (hac : a ≠ c)
  (hbc : b ≠ c)
  (sum_ac : a + c = 24)
  (sum_ab : a + b = 30)
  (sum_bc : b + c = 28) :
  c = 11 := by
  sorry

end NUMINAMATH_CALUDE_caitlin_uniform_number_l1363_136332


namespace NUMINAMATH_CALUDE_square_root_five_plus_one_squared_minus_two_times_plus_seven_equals_eleven_l1363_136330

theorem square_root_five_plus_one_squared_minus_two_times_plus_seven_equals_eleven :
  (Real.sqrt 5 + 1)^2 - 2 * (Real.sqrt 5 + 1) + 7 = 11 := by sorry

end NUMINAMATH_CALUDE_square_root_five_plus_one_squared_minus_two_times_plus_seven_equals_eleven_l1363_136330


namespace NUMINAMATH_CALUDE_roots_of_product_equation_l1363_136363

theorem roots_of_product_equation (p r : ℝ) (f g : ℝ → ℝ) 
  (hp : p > 0) (hr : r > 0)
  (hf : ∀ x, f x = 0 ↔ x = p)
  (hg : ∀ x, g x = 0 ↔ x = r)
  (hlin_f : ∃ a b, ∀ x, f x = a * x + b)
  (hlin_g : ∃ c d, ∀ x, g x = c * x + d) :
  ∀ x, f x * g x = f 0 * g 0 ↔ x = 0 ∨ x = p + r :=
sorry

end NUMINAMATH_CALUDE_roots_of_product_equation_l1363_136363


namespace NUMINAMATH_CALUDE_triangle_inradius_l1363_136306

/-- Given a triangle with perimeter 36 and area 45, prove that its inradius is 2.5 -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) : 
  P = 36 → A = 45 → A = r * (P / 2) → r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l1363_136306


namespace NUMINAMATH_CALUDE_quadratic_other_intercept_l1363_136361

/-- Given a quadratic function f(x) = ax² + bx + c with vertex (5, -3) and 
    one x-intercept at (1, 0), the x-coordinate of the other x-intercept is 9. -/
theorem quadratic_other_intercept 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 1 = 0) 
  (h3 : (5 : ℝ) = -b / (2 * a)) 
  (h4 : f 5 = -3) : 
  ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_other_intercept_l1363_136361


namespace NUMINAMATH_CALUDE_sum_of_g_42_and_neg_42_l1363_136317

/-- Given a function g: ℝ → ℝ defined as g(x) = ax^8 + bx^6 - cx^4 + dx^2 + 5
    where a, b, c, d are real constants, if g(42) = 3,
    then g(42) + g(-42) = 6 -/
theorem sum_of_g_42_and_neg_42 (a b c d : ℝ) (g : ℝ → ℝ)
    (h1 : ∀ x, g x = a * x^8 + b * x^6 - c * x^4 + d * x^2 + 5)
    (h2 : g 42 = 3) :
  g 42 + g (-42) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_g_42_and_neg_42_l1363_136317


namespace NUMINAMATH_CALUDE_log_equation_implies_sum_of_cubes_l1363_136333

theorem log_equation_implies_sum_of_cubes (x y : ℝ) 
  (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^3 + (Real.log y / Real.log 3)^3 = 
       3 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^3 + y^3 = 307 := by
sorry

end NUMINAMATH_CALUDE_log_equation_implies_sum_of_cubes_l1363_136333


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1363_136388

def is_valid_pair (x y : ℕ) : Prop :=
  Nat.Prime x ∧ Nat.Prime y ∧ x ≠ y ∧ (621 * x * y) % (x + y) = 0

theorem count_valid_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, is_valid_pair p.1 p.2) ∧ 
    (∀ x y, is_valid_pair x y → (x, y) ∈ pairs) ∧
    pairs.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1363_136388


namespace NUMINAMATH_CALUDE_discount_problem_l1363_136343

/-- The total cost after discount for a given number of toys, cost per toy, and discount percentage. -/
def totalCostAfterDiscount (numToys : ℕ) (costPerToy : ℚ) (discountPercentage : ℚ) : ℚ :=
  let totalCost := numToys * costPerToy
  let discountAmount := totalCost * (discountPercentage / 100)
  totalCost - discountAmount

/-- Theorem stating that the total cost after a 20% discount for 5 toys costing $3 each is $12. -/
theorem discount_problem : totalCostAfterDiscount 5 3 20 = 12 := by
  sorry

end NUMINAMATH_CALUDE_discount_problem_l1363_136343


namespace NUMINAMATH_CALUDE_kayak_trip_remaining_fraction_l1363_136311

/-- Given a kayak trip with total distance and distance paddled before lunch,
    calculate the fraction of the trip remaining after lunch -/
theorem kayak_trip_remaining_fraction
  (total_distance : ℝ)
  (distance_before_lunch : ℝ)
  (h1 : total_distance = 36)
  (h2 : distance_before_lunch = 12)
  : (total_distance - distance_before_lunch) / total_distance = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_kayak_trip_remaining_fraction_l1363_136311


namespace NUMINAMATH_CALUDE_oak_trees_remaining_l1363_136362

/-- The number of oak trees remaining in the park after cutting down damaged trees -/
def remaining_oak_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of remaining oak trees is 7 -/
theorem oak_trees_remaining :
  remaining_oak_trees 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_remaining_l1363_136362


namespace NUMINAMATH_CALUDE_faraway_impossible_totals_l1363_136303

/-- Represents the number of creatures in Faraway village -/
structure FarawayVillage where
  horses : ℕ
  goats : ℕ

/-- The total number of creatures in Faraway village -/
def total_creatures (v : FarawayVillage) : ℕ :=
  21 * v.horses + 6 * v.goats

/-- Theorem stating that 74 and 89 cannot be the total number of creatures -/
theorem faraway_impossible_totals :
  ¬ ∃ (v : FarawayVillage), total_creatures v = 74 ∨ total_creatures v = 89 := by
  sorry

end NUMINAMATH_CALUDE_faraway_impossible_totals_l1363_136303


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l1363_136301

theorem negation_of_forall_positive (S : Set ℚ) :
  (¬ ∀ x ∈ S, 2 * x + 1 > 0) ↔ (∃ x ∈ S, 2 * x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l1363_136301


namespace NUMINAMATH_CALUDE_function_composition_value_l1363_136309

/-- Given a function g and a composition f[g(x)], prove that f(0) = 4/5 -/
theorem function_composition_value (g : ℝ → ℝ) (f : ℝ → ℝ) :
  (∀ x, g x = 1 - 3 * x) →
  (∀ x, f (g x) = (1 - x^2) / (1 + x^2)) →
  f 0 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_value_l1363_136309


namespace NUMINAMATH_CALUDE_angle_sum_proof_l1363_136316

theorem angle_sum_proof (α β : Real) 
  (h1 : Real.tan α = 1/7) 
  (h2 : Real.tan β = 1/3) : 
  α + 2*β = π/4 := by sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l1363_136316


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1363_136396

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a < 0 ∧ b < 0 → a + b < 0) ∧
  ∃ (x y : ℝ), x + y < 0 ∧ ¬(x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1363_136396


namespace NUMINAMATH_CALUDE_rationalize_and_divide_l1363_136313

theorem rationalize_and_divide : (8 / Real.sqrt 8) / 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_divide_l1363_136313


namespace NUMINAMATH_CALUDE_pole_reconfiguration_l1363_136340

/-- Represents the configuration of electric poles on a road --/
structure RoadConfig where
  length : ℕ
  original_spacing : ℕ
  new_spacing : ℕ

/-- Calculates the number of holes needed for a given spacing --/
def holes_needed (config : RoadConfig) (spacing : ℕ) : ℕ :=
  config.length / spacing + 1

/-- Calculates the number of common holes between two spacings --/
def common_holes (config : RoadConfig) : ℕ :=
  config.length / (Nat.lcm config.original_spacing config.new_spacing) + 1

/-- The main theorem about the number of new holes and abandoned holes --/
theorem pole_reconfiguration (config : RoadConfig) 
  (h_length : config.length = 3000)
  (h_original : config.original_spacing = 50)
  (h_new : config.new_spacing = 60) :
  (holes_needed config config.new_spacing - common_holes config = 40) ∧
  (holes_needed config config.original_spacing - common_holes config = 50) := by
  sorry


end NUMINAMATH_CALUDE_pole_reconfiguration_l1363_136340


namespace NUMINAMATH_CALUDE_value_of_c_l1363_136352

theorem value_of_c (a b c : ℝ) : 
  8 = 0.04 * a → 
  4 = 0.08 * b → 
  c = b / a → 
  c = 0.25 := by
sorry

end NUMINAMATH_CALUDE_value_of_c_l1363_136352


namespace NUMINAMATH_CALUDE_solution_set_M_range_of_k_l1363_136339

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 4| - |x - 3|

-- Theorem for the solution set M
theorem solution_set_M : 
  {x : ℝ | f x ≤ 2} = Set.Icc (-1) 3 := by sorry

-- Theorem for the range of k
theorem range_of_k : 
  {k : ℝ | ∃ x, k^2 - 4*k - 3*f x = 0} = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_M_range_of_k_l1363_136339


namespace NUMINAMATH_CALUDE_figure_x_value_l1363_136377

/-- Given a figure composed of two squares, a right triangle, and a rectangle,
    where:
    - The right triangle has legs measuring 3x and 4x
    - One square has a side length of 4x
    - Another square has a side length of 6x
    - The rectangle has length 3x and width x
    - The total area of the figure is 1100 square inches
    Prove that the value of x is √(1100/61) -/
theorem figure_x_value :
  ∀ x : ℝ,
  (4*x)^2 + (6*x)^2 + (1/2 * 3*x * 4*x) + (3*x * x) = 1100 →
  x = Real.sqrt (1100 / 61) :=
by sorry

end NUMINAMATH_CALUDE_figure_x_value_l1363_136377


namespace NUMINAMATH_CALUDE_melanie_dimes_l1363_136335

theorem melanie_dimes (initial : ℕ) (from_dad : ℕ) (total : ℕ) (from_mom : ℕ) : 
  initial = 7 → from_dad = 8 → total = 19 → from_mom = total - (initial + from_dad) → from_mom = 4 := by sorry

end NUMINAMATH_CALUDE_melanie_dimes_l1363_136335


namespace NUMINAMATH_CALUDE_length_of_projected_segment_l1363_136379

/-- Given two points A and B on the y-axis, and their respective projections A' and B' on the line y = x,
    with AA' and BB' intersecting at point C, prove that the length of A'B' is 2.5√2. -/
theorem length_of_projected_segment (A B A' B' C : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 9) →
  (A'.1 = A'.2) →
  (B'.1 = B'.2) →
  (∃ t : ℝ, A + t • (C - A) = A') →
  (∃ s : ℝ, B + s • (C - B) = B') →
  ‖A' - B'‖ = 2.5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_length_of_projected_segment_l1363_136379


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_five_l1363_136302

theorem four_digit_divisible_by_five (n : ℕ) : 
  (5000 ≤ n ∧ n ≤ 5999) ∧ (n % 5 = 0) → 
  (Finset.filter (λ x : ℕ => (5000 ≤ x ∧ x ≤ 5999) ∧ (x % 5 = 0)) (Finset.range 10000)).card = 200 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_five_l1363_136302


namespace NUMINAMATH_CALUDE_infinite_points_in_circle_l1363_136328

open Set

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 < radius^2}

-- Define the condition for point P
def SatisfiesCondition (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  (p.1 - a.1)^2 + (p.2 - a.2)^2 + (p.1 - b.1)^2 + (p.2 - b.2)^2 ≤ 5

-- Theorem statement
theorem infinite_points_in_circle :
  let center := (0, 0)
  let radius := 2
  let a := (-2, 0)  -- One endpoint of the diameter
  let b := (2, 0)   -- Other endpoint of the diameter
  let valid_points := {p ∈ Circle center radius | SatisfiesCondition p a b}
  Infinite valid_points := by sorry

end NUMINAMATH_CALUDE_infinite_points_in_circle_l1363_136328


namespace NUMINAMATH_CALUDE_opposite_five_fourteen_implies_eighteen_l1363_136331

/-- A structure representing a circle with n equally spaced natural numbers -/
structure NumberCircle where
  n : ℕ
  numbers : Fin n → ℕ
  ordered : ∀ i : Fin n, numbers i = i.val + 1

/-- Definition of opposite numbers on the circle -/
def are_opposite (circle : NumberCircle) (a b : ℕ) : Prop :=
  ∃ i j : Fin circle.n,
    circle.numbers i = a ∧
    circle.numbers j = b ∧
    (j.val + circle.n / 2) % circle.n = i.val

/-- The main theorem -/
theorem opposite_five_fourteen_implies_eighteen (circle : NumberCircle) :
  are_opposite circle 5 14 → circle.n = 18 :=
by sorry

end NUMINAMATH_CALUDE_opposite_five_fourteen_implies_eighteen_l1363_136331


namespace NUMINAMATH_CALUDE_pascal_interior_sum_l1363_136312

/-- Represents the sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum (h : interior_sum 6 = 30) : interior_sum 8 = 126 := by
  sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l1363_136312


namespace NUMINAMATH_CALUDE_f_positive_on_interval_l1363_136383

open Real

noncomputable def f (a x : ℝ) : ℝ := a * log x - x - a / x + 2 * a

theorem f_positive_on_interval (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (exp 1), f a x > 0) ↔ a > (exp 2) / (3 * exp 1 - 1) := by
  sorry

end NUMINAMATH_CALUDE_f_positive_on_interval_l1363_136383


namespace NUMINAMATH_CALUDE_janet_additional_money_needed_l1363_136399

def janet_savings : ℕ := 2225
def monthly_rent : ℕ := 1250
def months_advance : ℕ := 2
def deposit : ℕ := 500
def utility_deposit : ℕ := 300
def moving_costs : ℕ := 150

theorem janet_additional_money_needed :
  janet_savings + (monthly_rent * months_advance + deposit + utility_deposit + moving_costs - janet_savings) = 3450 :=
by sorry

end NUMINAMATH_CALUDE_janet_additional_money_needed_l1363_136399


namespace NUMINAMATH_CALUDE_box_2_neg2_3_l1363_136326

/-- Definition of the box operation for integers -/
def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

/-- Theorem stating that the box operation applied to 2, -2, and 3 equals 69/4 -/
theorem box_2_neg2_3 : box 2 (-2) 3 = 69/4 := by sorry

end NUMINAMATH_CALUDE_box_2_neg2_3_l1363_136326


namespace NUMINAMATH_CALUDE_books_sold_l1363_136353

/-- Given that Tom initially had 5 books, bought 38 new books, and now has 39 books in total,
    prove that the number of books Tom sold is 4. -/
theorem books_sold (initial_books : ℕ) (new_books : ℕ) (total_books : ℕ) (sold_books : ℕ) : 
  initial_books = 5 → new_books = 38 → total_books = 39 → 
  initial_books - sold_books + new_books = total_books →
  sold_books = 4 := by
sorry

end NUMINAMATH_CALUDE_books_sold_l1363_136353


namespace NUMINAMATH_CALUDE_right_triangle_with_35_hypotenuse_l1363_136346

theorem right_triangle_with_35_hypotenuse (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 35 →           -- Hypotenuse length
  b = a + 1 →        -- Consecutive integer legs
  a + b = 51         -- Sum of leg lengths
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_with_35_hypotenuse_l1363_136346


namespace NUMINAMATH_CALUDE_existence_of_separated_points_l1363_136325

/-- A type representing a segment in a plane -/
structure Segment where
  -- Add necessary fields

/-- A type representing a point in a plane -/
structure Point where
  -- Add necessary fields

/-- Checks if two segments are parallel -/
def are_parallel (s1 s2 : Segment) : Prop :=
  sorry

/-- Checks if two segments intersect -/
def intersect (s1 s2 : Segment) : Prop :=
  sorry

/-- Checks if a segment separates two points -/
def separates (s : Segment) (p1 p2 : Point) : Prop :=
  sorry

/-- Main theorem -/
theorem existence_of_separated_points (n : ℕ) (segments : Fin (n^2) → Segment)
  (h1 : ∀ i j, i ≠ j → ¬(are_parallel (segments i) (segments j)))
  (h2 : ∀ i j, i ≠ j → ¬(intersect (segments i) (segments j))) :
  ∃ (points : Fin n → Point),
    ∀ i j, i ≠ j → ∃ k, separates (segments k) (points i) (points j) :=
sorry

end NUMINAMATH_CALUDE_existence_of_separated_points_l1363_136325


namespace NUMINAMATH_CALUDE_tangent_line_slope_is_one_l1363_136348

/-- The slope of a line passing through (-1, 0) and tangent to y = e^x is 1 -/
theorem tangent_line_slope_is_one :
  ∀ (a : ℝ), 
    (∃ (k : ℝ), 
      (∀ x, k * (x + 1) = Real.exp x → x = a) ∧ 
      k * (a + 1) = Real.exp a ∧
      k = Real.exp a) →
    k = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_is_one_l1363_136348


namespace NUMINAMATH_CALUDE_tylenol_interval_l1363_136392

-- Define the problem parameters
def total_hours : ℝ := 12
def tablet_mg : ℝ := 500
def tablets_per_dose : ℝ := 2
def total_grams : ℝ := 3

-- Define the theorem
theorem tylenol_interval :
  let total_mg : ℝ := total_grams * 1000
  let total_tablets : ℝ := total_mg / tablet_mg
  let intervals : ℝ := total_tablets - 1
  total_hours / intervals = 2.4 := by sorry

end NUMINAMATH_CALUDE_tylenol_interval_l1363_136392


namespace NUMINAMATH_CALUDE_simplify_expression_l1363_136374

theorem simplify_expression (x y : ℝ) (h : y ≠ 0) :
  let P := x^2 + y^2
  let Q := x^2 - y^2
  ((P + 3*Q) / (P - Q)) - ((P - 3*Q) / (P + Q)) = (2*x^4 - y^4) / (x^2 * y^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1363_136374


namespace NUMINAMATH_CALUDE_symmetric_origin_implies_sum_zero_l1363_136360

-- Define a property for a function to be symmetric about the origin
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

-- Theorem statement
theorem symmetric_origin_implies_sum_zero
  (f : ℝ → ℝ) (h : SymmetricAboutOrigin f) :
  ∀ x : ℝ, f x + f (-x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_origin_implies_sum_zero_l1363_136360


namespace NUMINAMATH_CALUDE_square_side_length_l1363_136334

theorem square_side_length (area : ℚ) (side : ℚ) (h1 : area = 9/16) (h2 : side * side = area) : side = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1363_136334


namespace NUMINAMATH_CALUDE_division_result_l1363_136345

theorem division_result : (32 / 8 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1363_136345


namespace NUMINAMATH_CALUDE_binomial_variance_transform_l1363_136347

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Variance of a linear transformation of a random variable -/
def varianceLinearTransform (X : BinomialRV) (a b : ℝ) : ℝ := a^2 * variance X

/-- The main theorem to prove -/
theorem binomial_variance_transform (ξ : BinomialRV) 
    (h_n : ξ.n = 100) (h_p : ξ.p = 0.3) : 
    varianceLinearTransform ξ 3 (-5) = 189 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_transform_l1363_136347


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_nonnegative_l1363_136329

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x - a > 0}
def B : Set ℝ := {x | x ≤ 0}

-- State the theorem
theorem intersection_empty_implies_a_nonnegative (a : ℝ) :
  A a ∩ B = ∅ → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_nonnegative_l1363_136329


namespace NUMINAMATH_CALUDE_inequality_preservation_l1363_136314

theorem inequality_preservation (x y : ℝ) (h : x > y) : x - 3 > y - 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1363_136314


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1363_136382

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1363_136382


namespace NUMINAMATH_CALUDE_complex_point_location_l1363_136380

theorem complex_point_location (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (eq : a + i = (b + i) * (2 - i)) : 
  a > 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_location_l1363_136380


namespace NUMINAMATH_CALUDE_average_hiring_per_week_l1363_136351

def employee_hiring (week1 week2 week3 week4 : ℕ) : Prop :=
  (week1 = week2 + 200) ∧
  (week2 + 150 = week3) ∧
  (week4 = 2 * week3) ∧
  (week4 = 400)

theorem average_hiring_per_week 
  (week1 week2 week3 week4 : ℕ) 
  (h : employee_hiring week1 week2 week3 week4) : 
  (week1 + week2 + week3 + week4) / 4 = 225 := by
  sorry

end NUMINAMATH_CALUDE_average_hiring_per_week_l1363_136351


namespace NUMINAMATH_CALUDE_major_axis_length_is_three_l1363_136337

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * (1 + major_minor_ratio)

/-- Theorem: The major axis length is 3 when the cylinder radius is 1 and the major axis is 50% longer than the minor axis -/
theorem major_axis_length_is_three :
  major_axis_length 1 0.5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_is_three_l1363_136337


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l1363_136323

theorem binomial_expansion_constant_term (a : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → f x = (a * x^2 - 2 / Real.sqrt x)^5) ∧ 
   (∃ c, c = 160 ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - c| < ε))) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l1363_136323


namespace NUMINAMATH_CALUDE_professional_ratio_l1363_136366

/-- Represents a professional group with engineers, doctors, and lawyers. -/
structure ProfessionalGroup where
  numEngineers : ℕ
  numDoctors : ℕ
  numLawyers : ℕ

/-- The average age of the entire group -/
def groupAverageAge : ℝ := 45

/-- The average age of engineers -/
def engineerAverageAge : ℝ := 40

/-- The average age of doctors -/
def doctorAverageAge : ℝ := 50

/-- The average age of lawyers -/
def lawyerAverageAge : ℝ := 60

/-- Theorem stating the ratio of professionals in the group -/
theorem professional_ratio (group : ProfessionalGroup) :
  group.numEngineers * (doctorAverageAge - groupAverageAge) =
  group.numDoctors * (groupAverageAge - engineerAverageAge) ∧
  group.numEngineers * (lawyerAverageAge - groupAverageAge) =
  3 * group.numLawyers * (groupAverageAge - engineerAverageAge) :=
sorry

end NUMINAMATH_CALUDE_professional_ratio_l1363_136366


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_540_l1363_136385

/-- The number of ways to allocate teachers to schools -/
def allocation_schemes (math_teachers language_teachers schools : ℕ) : ℕ :=
  (math_teachers.factorial * language_teachers.factorial) / 
  ((math_teachers / schools).factorial ^ schools * 
   (language_teachers / schools).factorial ^ schools * schools.factorial)

/-- Theorem: The number of allocation schemes for the given problem is 540 -/
theorem allocation_schemes_eq_540 : 
  allocation_schemes 3 6 3 = 540 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_540_l1363_136385


namespace NUMINAMATH_CALUDE_max_diagonal_sum_l1363_136397

/-- A rhombus with side length 5 and diagonals d1 and d2 -/
structure Rhombus where
  side_length : ℝ
  d1 : ℝ
  d2 : ℝ
  side_is_5 : side_length = 5
  d1_le_6 : d1 ≤ 6
  d2_ge_6 : d2 ≥ 6

/-- The maximum sum of diagonals in a rhombus with given constraints is 14 -/
theorem max_diagonal_sum (r : Rhombus) : (r.d1 + r.d2 ≤ 14) ∧ (∃ (s : Rhombus), s.d1 + s.d2 = 14) :=
  sorry

end NUMINAMATH_CALUDE_max_diagonal_sum_l1363_136397


namespace NUMINAMATH_CALUDE_sum_of_number_and_reverse_divisible_by_11_l1363_136320

theorem sum_of_number_and_reverse_divisible_by_11 (A B : ℕ) : 
  A < 10 → B < 10 → A ≠ B → 
  11 ∣ ((10 * A + B) + (10 * B + A)) := by
sorry

end NUMINAMATH_CALUDE_sum_of_number_and_reverse_divisible_by_11_l1363_136320


namespace NUMINAMATH_CALUDE_largest_n_multiple_of_four_l1363_136342

def expression (n : ℕ) : ℤ :=
  7 * (n - 3)^4 - n^2 + 12*n - 30

theorem largest_n_multiple_of_four :
  ∀ n : ℕ, n < 100000 →
    (4 ∣ expression n) →
    n ≤ 99999 ∧
    (4 ∣ expression 99999) ∧
    99999 < 100000 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_multiple_of_four_l1363_136342


namespace NUMINAMATH_CALUDE_total_selling_price_l1363_136391

-- Define the cost and loss percentage for each item
def cost1 : ℕ := 750
def cost2 : ℕ := 1200
def cost3 : ℕ := 500
def loss_percent1 : ℚ := 10 / 100
def loss_percent2 : ℚ := 15 / 100
def loss_percent3 : ℚ := 5 / 100

-- Calculate the selling price of an item
def selling_price (cost : ℕ) (loss_percent : ℚ) : ℚ :=
  cost - (cost * loss_percent)

-- Define the theorem
theorem total_selling_price :
  selling_price cost1 loss_percent1 +
  selling_price cost2 loss_percent2 +
  selling_price cost3 loss_percent3 = 2170 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_l1363_136391


namespace NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l1363_136398

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => (k + 1) / 5^(k + 1))

/-- 30! has 7 trailing zeros -/
theorem thirty_factorial_trailing_zeros :
  trailingZeros 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l1363_136398


namespace NUMINAMATH_CALUDE_diagonals_15_sided_polygon_l1363_136349

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex polygon with 15 sides is 90 -/
theorem diagonals_15_sided_polygon : num_diagonals 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_15_sided_polygon_l1363_136349


namespace NUMINAMATH_CALUDE_bobs_spending_ratio_l1363_136307

/-- Proves that given Bob's spending pattern, the ratio of Tuesday's spending to Monday's remaining amount is 1/5 -/
theorem bobs_spending_ratio : 
  ∀ (initial_amount : ℚ) (tuesday_spent : ℚ) (final_amount : ℚ),
  initial_amount = 80 →
  final_amount = 20 →
  tuesday_spent > 0 →
  tuesday_spent < 40 →
  20 = 40 - tuesday_spent - (3/8) * (40 - tuesday_spent) →
  tuesday_spent / 40 = 1/5 := by
sorry

end NUMINAMATH_CALUDE_bobs_spending_ratio_l1363_136307


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1363_136305

theorem arithmetic_calculation : 10 - 9 + 8 * 7 + 6 - 5 * 4 / 2 + 3 - 1 = 55 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1363_136305


namespace NUMINAMATH_CALUDE_abs_neg_three_halves_l1363_136315

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_halves_l1363_136315


namespace NUMINAMATH_CALUDE_distinct_products_count_l1363_136356

def S : Finset ℕ := {1, 3, 7, 9, 13}

def products : Finset ℕ :=
  (S.powerset.filter (λ s => s.card ≥ 2)).image (λ s => s.prod id)

theorem distinct_products_count : products.card = 11 := by
  sorry

end NUMINAMATH_CALUDE_distinct_products_count_l1363_136356


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_squared_l1363_136359

theorem integral_sqrt_one_minus_x_squared_plus_x_squared :
  ∫ x in (-1 : ℝ)..1, (Real.sqrt (1 - x^2) + x^2) = π / 2 + 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_squared_l1363_136359


namespace NUMINAMATH_CALUDE_tan_x_axis_intersection_l1363_136341

theorem tan_x_axis_intersection :
  ∀ (x : ℝ), (∃ (n : ℤ), x = -π/8 + n*π/2) ↔ Real.tan (2*x + π/4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_tan_x_axis_intersection_l1363_136341


namespace NUMINAMATH_CALUDE_tangent_inequality_l1363_136371

theorem tangent_inequality (α β : Real) 
  (h1 : 0 < α) (h2 : α ≤ π/4) (h3 : 0 < β) (h4 : β ≤ π/4) : 
  Real.sqrt (Real.tan α * Real.tan β) ≤ Real.tan ((α + β)/2) ∧ 
  Real.tan ((α + β)/2) ≤ (Real.tan α + Real.tan β)/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_inequality_l1363_136371


namespace NUMINAMATH_CALUDE_shorts_price_is_6_l1363_136375

/-- The price of a single jacket in dollars -/
def jacket_price : ℕ := 10

/-- The number of jackets bought -/
def num_jackets : ℕ := 3

/-- The price of a single pair of pants in dollars -/
def pants_price : ℕ := 12

/-- The number of pairs of pants bought -/
def num_pants : ℕ := 4

/-- The number of pairs of shorts bought -/
def num_shorts : ℕ := 2

/-- The total amount spent in dollars -/
def total_spent : ℕ := 90

theorem shorts_price_is_6 :
  ∃ (shorts_price : ℕ),
    shorts_price * num_shorts + 
    jacket_price * num_jackets + 
    pants_price * num_pants = total_spent ∧
    shorts_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_shorts_price_is_6_l1363_136375


namespace NUMINAMATH_CALUDE_book_club_boys_count_l1363_136387

theorem book_club_boys_count (total_members attendees : ℕ) 
  (h_total : total_members = 30)
  (h_attendees : attendees = 18)
  (h_all_boys_attended : ∃ boys girls : ℕ, 
    boys + girls = total_members ∧
    boys + (girls / 3) = attendees) : 
  ∃ boys : ℕ, boys = 12 ∧ ∃ girls : ℕ, boys + girls = total_members :=
sorry

end NUMINAMATH_CALUDE_book_club_boys_count_l1363_136387


namespace NUMINAMATH_CALUDE_right_triangle_integer_area_l1363_136364

theorem right_triangle_integer_area 
  (a b c : ℕ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : 
  ∃ A : ℕ, 2 * A = a * b :=
sorry

end NUMINAMATH_CALUDE_right_triangle_integer_area_l1363_136364


namespace NUMINAMATH_CALUDE_rachel_math_problems_l1363_136336

/-- The number of math problems Rachel solved in total -/
def total_problems (problems_per_minute : ℕ) (minutes_solved : ℕ) (problems_next_day : ℕ) : ℕ :=
  problems_per_minute * minutes_solved + problems_next_day

/-- Theorem stating that Rachel solved 151 math problems in total -/
theorem rachel_math_problems :
  total_problems 7 18 25 = 151 := by
  sorry

end NUMINAMATH_CALUDE_rachel_math_problems_l1363_136336
