import Mathlib

namespace NUMINAMATH_CALUDE_largest_power_of_three_dividing_expression_l1512_151201

theorem largest_power_of_three_dividing_expression (m : ℕ) : 
  (∃ (k : ℕ), (3^k : ℕ) ∣ (2^(3^m) + 1)) ∧ 
  (∀ (k : ℕ), k > 2 → ¬((3^k : ℕ) ∣ (2^(3^m) + 1))) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_three_dividing_expression_l1512_151201


namespace NUMINAMATH_CALUDE_running_distance_l1512_151272

theorem running_distance (jonathan_distance : ℝ) 
  (h1 : jonathan_distance = 7.5)
  (mercedes_distance : ℝ) 
  (h2 : mercedes_distance = 2 * jonathan_distance)
  (davonte_distance : ℝ) 
  (h3 : davonte_distance = mercedes_distance + 2) :
  mercedes_distance + davonte_distance = 32 := by
sorry

end NUMINAMATH_CALUDE_running_distance_l1512_151272


namespace NUMINAMATH_CALUDE_event_probability_l1512_151299

theorem event_probability (n : ℕ) (k₀ : ℕ) (p : ℝ) 
  (h1 : n = 120) 
  (h2 : k₀ = 32) 
  (h3 : k₀ = Int.floor (n * p)) :
  32 / 121 ≤ p ∧ p ≤ 33 / 121 := by
  sorry

end NUMINAMATH_CALUDE_event_probability_l1512_151299


namespace NUMINAMATH_CALUDE_tim_initial_books_l1512_151232

/-- The number of books Sandy has -/
def sandy_books : ℕ := 10

/-- The number of books Benny lost -/
def benny_lost : ℕ := 24

/-- The number of books they have together after Benny lost some -/
def remaining_books : ℕ := 19

/-- Tim's initial number of books -/
def tim_books : ℕ := 33

theorem tim_initial_books : 
  sandy_books + tim_books - benny_lost = remaining_books :=
by sorry

end NUMINAMATH_CALUDE_tim_initial_books_l1512_151232


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l1512_151212

/-- A coloring of integers from 1 to 2014 using four colors -/
def Coloring := Fin 2014 → Fin 4

/-- An arithmetic progression of length 11 within the range 1 to 2014 -/
structure ArithmeticProgression :=
  (start : Fin 2014)
  (step : Nat)
  (h : ∀ i : Fin 11, (start.val : ℕ) + i.val * step ≤ 2014)

/-- A coloring is valid if no arithmetic progression of length 11 is monochromatic -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ ap : ArithmeticProgression, ∃ i j : Fin 11, i ≠ j ∧ 
    c ⟨(ap.start.val + i.val * ap.step : ℕ), by sorry⟩ ≠ 
    c ⟨(ap.start.val + j.val * ap.step : ℕ), by sorry⟩

/-- There exists a valid coloring of integers from 1 to 2014 using four colors -/
theorem exists_valid_coloring : ∃ c : Coloring, ValidColoring c := by sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l1512_151212


namespace NUMINAMATH_CALUDE_tokens_theorem_l1512_151239

/-- The number of tokens Elsa has -/
def elsa_tokens : ℕ := 60

/-- The number of tokens Angus has -/
def x : ℕ := elsa_tokens - (elsa_tokens / 4)

/-- The number of tokens Bella has -/
def y : ℕ := elsa_tokens + (x^2 - 10)

theorem tokens_theorem : x = 45 ∧ y = 2075 := by
  sorry

end NUMINAMATH_CALUDE_tokens_theorem_l1512_151239


namespace NUMINAMATH_CALUDE_red_blocks_count_l1512_151282

theorem red_blocks_count (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : yellow = red + 7)
  (h2 : blue = red + 14)
  (h3 : red + yellow + blue = 75) :
  red = 18 := by
  sorry

end NUMINAMATH_CALUDE_red_blocks_count_l1512_151282


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l1512_151227

-- Define the binary operation ◇ on nonzero real numbers
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- State the theorem
theorem diamond_equation_solution :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 →
    diamond a (diamond b c) = (diamond a b) * c) →
  (∀ (a : ℝ), a ≠ 0 → diamond a a = 1) →
  (∃! (y : ℝ), diamond 2024 (diamond 8 y) = 200 ∧ y = 200 / 253) :=
by sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l1512_151227


namespace NUMINAMATH_CALUDE_lcm_gcd_product_10_15_l1512_151263

theorem lcm_gcd_product_10_15 : Nat.lcm 10 15 * Nat.gcd 10 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_10_15_l1512_151263


namespace NUMINAMATH_CALUDE_average_marks_bcd_e_is_48_l1512_151211

def average_marks_bcd_e (a b c d e : ℕ) : Prop :=
  -- The average marks of a, b, c is 48
  (a + b + c) / 3 = 48 ∧
  -- When d joins, the average becomes 47
  (a + b + c + d) / 4 = 47 ∧
  -- E has 3 more marks than d
  e = d + 3 ∧
  -- The marks of a is 43
  a = 43 →
  -- The average marks of b, c, d, e is 48
  (b + c + d + e) / 4 = 48

theorem average_marks_bcd_e_is_48 : 
  ∀ (a b c d e : ℕ), average_marks_bcd_e a b c d e :=
sorry

end NUMINAMATH_CALUDE_average_marks_bcd_e_is_48_l1512_151211


namespace NUMINAMATH_CALUDE_sector_properties_l1512_151290

/-- Proves that a circular sector with perimeter 4 and area 1 has radius 1 and central angle 2 -/
theorem sector_properties :
  ∀ r θ : ℝ,
  r > 0 →
  θ > 0 →
  2 * r + θ * r = 4 →
  1 / 2 * θ * r^2 = 1 →
  r = 1 ∧ θ = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_properties_l1512_151290


namespace NUMINAMATH_CALUDE_intersection_distance_l1512_151271

/-- The distance between intersection points of two curves with a ray in polar coordinates --/
theorem intersection_distance (θ : Real) : 
  let ρ₁ : Real := Real.sqrt (2 / (Real.cos θ ^ 2 - Real.sin θ ^ 2))
  let ρ₂ : Real := 4 * Real.cos θ
  θ = π / 6 → abs (ρ₁ - ρ₂) = 2 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l1512_151271


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l1512_151200

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l1512_151200


namespace NUMINAMATH_CALUDE_sheep_grass_consumption_l1512_151225

theorem sheep_grass_consumption 
  (num_sheep : ℕ) 
  (num_bags : ℕ) 
  (num_days : ℕ) 
  (h1 : num_sheep = 40) 
  (h2 : num_bags = 40) 
  (h3 : num_days = 40) :
  num_days = 40 := by
  sorry

end NUMINAMATH_CALUDE_sheep_grass_consumption_l1512_151225


namespace NUMINAMATH_CALUDE_beach_problem_l1512_151219

/-- The number of people originally in the second row of the beach -/
def original_second_row : ℕ := 20

theorem beach_problem :
  let first_row : ℕ := 24
  let first_row_left : ℕ := 3
  let second_row_left : ℕ := 5
  let third_row : ℕ := 18
  let total_remaining : ℕ := 54
  (first_row - first_row_left) + (original_second_row - second_row_left) + third_row = total_remaining :=
by sorry

end NUMINAMATH_CALUDE_beach_problem_l1512_151219


namespace NUMINAMATH_CALUDE_equation_solutions_l1512_151233

def is_solution (m n r k : ℕ+) : Prop :=
  m * n + n * r + m * r = k * (m + n + r)

theorem equation_solutions :
  (∃ (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card = 7 ∧ 
    (∀ x ∈ s, is_solution x.1 x.2.1 x.2.2 2) ∧
    (∀ x : ℕ+ × ℕ+ × ℕ+, is_solution x.1 x.2.1 x.2.2 2 → x ∈ s)) ∧
  (∀ k : ℕ+, k > 1 → 
    ∃ (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card ≥ 3 * k + 1 ∧ 
      ∀ x ∈ s, is_solution x.1 x.2.1 x.2.2 k) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1512_151233


namespace NUMINAMATH_CALUDE_jeremy_age_l1512_151240

/-- Given the ages of Amy, Jeremy, and Chris, prove Jeremy's age --/
theorem jeremy_age (amy jeremy chris : ℕ) 
  (h1 : amy + jeremy + chris = 132)  -- Combined age
  (h2 : amy = jeremy / 3)            -- Amy's age relation to Jeremy
  (h3 : chris = 2 * amy)             -- Chris's age relation to Amy
  : jeremy = 66 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_age_l1512_151240


namespace NUMINAMATH_CALUDE_probability_not_snow_l1512_151203

theorem probability_not_snow (p_snow : ℚ) (h : p_snow = 2 / 5) : 1 - p_snow = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snow_l1512_151203


namespace NUMINAMATH_CALUDE_grazing_months_a_l1512_151298

/-- The number of months a put his oxen for grazing -/
def months_a : ℕ := 7

/-- The number of oxen a put for grazing -/
def oxen_a : ℕ := 10

/-- The number of oxen b put for grazing -/
def oxen_b : ℕ := 12

/-- The number of months b put his oxen for grazing -/
def months_b : ℕ := 5

/-- The number of oxen c put for grazing -/
def oxen_c : ℕ := 15

/-- The number of months c put his oxen for grazing -/
def months_c : ℕ := 3

/-- The total rent of the pasture in rupees -/
def total_rent : ℚ := 245

/-- The share of rent c pays in rupees -/
def c_rent_share : ℚ := 62.99999999999999

theorem grazing_months_a : 
  months_a * oxen_a * total_rent = 
  c_rent_share * (months_a * oxen_a + months_b * oxen_b + months_c * oxen_c) := by
  sorry

end NUMINAMATH_CALUDE_grazing_months_a_l1512_151298


namespace NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1512_151262

/-- The minimum distance from a point on the parabola y = x^2 + 1 to the line y = 2x - 1 is √5/5 -/
theorem min_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2 + 1}
  let line := {p : ℝ × ℝ | p.2 = 2 * p.1 - 1}
  (∀ p ∈ parabola, ∃ q ∈ line, ∀ r ∈ line, dist p q ≤ dist p r) →
  (∃ p ∈ parabola, ∃ q ∈ line, dist p q = Real.sqrt 5 / 5) ∧
  (∀ p ∈ parabola, ∀ q ∈ line, dist p q ≥ Real.sqrt 5 / 5) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1512_151262


namespace NUMINAMATH_CALUDE_min_a_for_g_zeros_l1512_151241

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x| + |x - 1|

-- Define the function g(x) in terms of f(x) and a
def g (a : ℝ) (x : ℝ) : ℝ := f x - a

-- Theorem statement
theorem min_a_for_g_zeros :
  ∃ (a : ℝ), (∃ (x : ℝ), g a x = 0) ∧
  (∀ (b : ℝ), b < a → ¬∃ (x : ℝ), g b x = 0) ∧
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_g_zeros_l1512_151241


namespace NUMINAMATH_CALUDE_linear_function_condition_l1512_151252

/-- A linear function with respect to x of the form y = (m-2)x + 2 -/
def linearFunction (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x + 2

/-- The condition for the function to be linear with respect to x -/
def isLinear (m : ℝ) : Prop := m ≠ 2

theorem linear_function_condition (m : ℝ) :
  (∀ x, ∃ y, y = linearFunction m x) ↔ isLinear m :=
sorry

end NUMINAMATH_CALUDE_linear_function_condition_l1512_151252


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l1512_151224

noncomputable section

def f (x : ℝ) : ℝ := Real.log x + (x + 1) / x

theorem tangent_line_and_inequality (x : ℝ) (hx : x > 0) (hx1 : x ≠ 1) :
  (∃ (m b : ℝ), m * 1 + b = f 1 ∧ m = 2 ∧ ∀ t, f t = m * t + b) ∧
  f x > ((x + 1) * Real.log x) / (x - 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l1512_151224


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_l1512_151284

/-- An arithmetic sequence with given first two terms -/
def arithmetic_sequence (a₁ a₂ : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * (a₂ - a₁)

/-- Check if three numbers form a geometric sequence -/
def is_geometric_sequence (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem arithmetic_to_geometric :
  ∃ x : ℝ, is_geometric_sequence (x - 8) (x + (arithmetic_sequence (-8) (-6) 4))
                                 (x + (arithmetic_sequence (-8) (-6) 5)) ∧
            x = -1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_l1512_151284


namespace NUMINAMATH_CALUDE_prob_different_colors_l1512_151210

/-- Represents the color of a chip -/
inductive ChipColor
  | Blue
  | Red
  | Yellow

/-- Represents the state of the bag after the first draw -/
structure BagState where
  blue : Nat
  red : Nat
  yellow : Nat

/-- The initial state of the bag -/
def initialBag : BagState :=
  { blue := 6, red := 5, yellow := 4 }

/-- The state of the bag after drawing a blue chip -/
def bagAfterBlue : BagState :=
  { blue := 7, red := 5, yellow := 4 }

/-- The probability of drawing two chips of different colors -/
def probDifferentColors : ℚ := 593 / 900

/-- The theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors :
  let totalChips := initialBag.blue + initialBag.red + initialBag.yellow
  let probFirstBlue := initialBag.blue / totalChips
  let probFirstRed := initialBag.red / totalChips
  let probFirstYellow := initialBag.yellow / totalChips
  let probSecondNotBlueAfterBlue := (bagAfterBlue.red + bagAfterBlue.yellow) / (bagAfterBlue.blue + bagAfterBlue.red + bagAfterBlue.yellow)
  let probSecondNotRedAfterRed := (initialBag.blue + initialBag.yellow) / totalChips
  let probSecondNotYellowAfterYellow := (initialBag.blue + initialBag.red) / totalChips
  probFirstBlue * probSecondNotBlueAfterBlue +
  probFirstRed * probSecondNotRedAfterRed +
  probFirstYellow * probSecondNotYellowAfterYellow = probDifferentColors :=
by
  sorry


end NUMINAMATH_CALUDE_prob_different_colors_l1512_151210


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l1512_151267

/-- A type representing a circular arrangement of 9 digits -/
def CircularArrangement := Fin 9 → Fin 9

/-- Checks if a number is composite -/
def is_composite (n : ℕ) : Prop := ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- Checks if two adjacent digits in the arrangement form a composite number -/
def adjacent_composite (arr : CircularArrangement) (i : Fin 9) : Prop :=
  let n := (arr i).val * 10 + (arr ((i.val + 1) % 9)).val
  is_composite n

/-- The main theorem stating the existence of a valid arrangement -/
theorem exists_valid_arrangement : ∃ (arr : CircularArrangement), 
  (∀ i : Fin 9, 1 ≤ (arr i).val ∧ (arr i).val ≤ 9) ∧ 
  (∀ i j : Fin 9, i ≠ j → arr i ≠ arr j) ∧
  (∀ i : Fin 9, adjacent_composite arr i) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l1512_151267


namespace NUMINAMATH_CALUDE_root_between_a_and_b_l1512_151288

theorem root_between_a_and_b (p q a b : ℝ) 
  (ha : a^2 + p*a + q = 0)
  (hb : b^2 - p*b - q = 0)
  (hq : q ≠ 0) :
  ∃ c ∈ Set.Ioo a b, c^2 + 2*p*c + 2*q = 0 := by
sorry

end NUMINAMATH_CALUDE_root_between_a_and_b_l1512_151288


namespace NUMINAMATH_CALUDE_fraction_product_l1512_151206

theorem fraction_product : (2 : ℚ) / 3 * 3 / 5 * 4 / 7 * 5 / 9 = 8 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1512_151206


namespace NUMINAMATH_CALUDE_movie_change_theorem_l1512_151268

/-- The change received by two sisters after buying movie tickets -/
def change_received (ticket_price : ℕ) (money_brought : ℕ) : ℕ :=
  money_brought - (2 * ticket_price)

/-- Theorem: The change received is $9 when tickets cost $8 each and the sisters brought $25 -/
theorem movie_change_theorem : change_received 8 25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_movie_change_theorem_l1512_151268


namespace NUMINAMATH_CALUDE_remaining_liquid_weight_l1512_151264

/-- Proves that the weight of the remaining liquid after evaporation is 6 kg --/
theorem remaining_liquid_weight (initial_weight : ℝ) (evaporated_water : ℝ) (added_solution : ℝ) 
  (initial_x_percent : ℝ) (final_x_percent : ℝ) :
  initial_weight = 8 →
  evaporated_water = 2 →
  added_solution = 2 →
  initial_x_percent = 0.2 →
  final_x_percent = 0.25 →
  ∃ (remaining_weight : ℝ),
    remaining_weight = initial_weight - evaporated_water ∧
    (remaining_weight + added_solution) * final_x_percent = 
      initial_weight * initial_x_percent + added_solution * initial_x_percent ∧
    remaining_weight = 6 :=
by sorry

end NUMINAMATH_CALUDE_remaining_liquid_weight_l1512_151264


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l1512_151286

theorem shaded_area_between_circles (r₁ r₂ : ℝ) : 
  r₁ = Real.sqrt 2 → r₂ = 2 * r₁ → π * r₂^2 - π * r₁^2 = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l1512_151286


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1512_151261

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 9th term of the arithmetic sequence is 5 -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_sum : a 4 + a 6 = 8) :
  a 9 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1512_151261


namespace NUMINAMATH_CALUDE_digit_2500_is_3_l1512_151235

/-- Represents the decimal number obtained by writing integers from 999 down to 1 in reverse order -/
def reverse_decimal : ℚ :=
  sorry

/-- Returns the nth digit after the decimal point in the given rational number -/
def nth_digit (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_2500_is_3 : nth_digit reverse_decimal 2500 = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_2500_is_3_l1512_151235


namespace NUMINAMATH_CALUDE_alternating_sequence_sum_l1512_151293

def sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  let last_term := a₁ + (n - 1) * d
  let sum_of_pairs := ((n - 1) / 2) * (a₁ + last_term - d)
  if n % 2 = 0 then sum_of_pairs else sum_of_pairs + last_term

theorem alternating_sequence_sum :
  sequence_sum 2 3 19 = 29 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sequence_sum_l1512_151293


namespace NUMINAMATH_CALUDE_meeting_distance_l1512_151215

-- Define the speeds and distance
def xiaoBinSpeed : ℝ := 15
def xiaoMingSpeed : ℝ := 5
def distanceToSchool : ℝ := 30

-- Define the theorem
theorem meeting_distance :
  let totalDistance : ℝ := 2 * distanceToSchool
  let meetingTime : ℝ := totalDistance / (xiaoBinSpeed + xiaoMingSpeed)
  let xiaoMingDistance : ℝ := meetingTime * xiaoMingSpeed
  xiaoMingDistance = 15 := by sorry

end NUMINAMATH_CALUDE_meeting_distance_l1512_151215


namespace NUMINAMATH_CALUDE_average_of_set_l1512_151230

theorem average_of_set (S : Finset ℕ) (n : ℕ) (h_nonempty : S.Nonempty) :
  (∃ (max min : ℕ),
    max ∈ S ∧ min ∈ S ∧
    (∀ x ∈ S, x ≤ max) ∧
    (∀ x ∈ S, min ≤ x) ∧
    (S.sum id - max) / (S.card - 1) = 32 ∧
    (S.sum id - max - min) / (S.card - 2) = 35 ∧
    (S.sum id - min) / (S.card - 1) = 40 ∧
    max = min + 72) →
  S.sum id / S.card = 368 / 10 := by
sorry

end NUMINAMATH_CALUDE_average_of_set_l1512_151230


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_square_trisection_l1512_151248

/-- Given an ellipse where the two trisection points on the minor axis and its two foci form a square,
    prove that its eccentricity is √10/10 -/
theorem ellipse_eccentricity_square_trisection (a b c : ℝ) :
  b = 3 * c →                    -- Condition: trisection points and foci form a square
  a ^ 2 = b ^ 2 + c ^ 2 →        -- Definition: relationship between semi-major axis, semi-minor axis, and focal distance
  c / a = (Real.sqrt 10) / 10 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_square_trisection_l1512_151248


namespace NUMINAMATH_CALUDE_expression_perfect_square_iff_l1512_151214

def factorial (n : ℕ) : ℕ := Nat.factorial n

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def expression (n : ℕ) : ℕ := 
  (List.range (2*n + 1)).foldl (λ acc i => acc * factorial i) 1 / factorial (n + 1)

theorem expression_perfect_square_iff (n : ℕ) : 
  is_perfect_square (expression n) ↔ 
  (∃ k : ℕ, n = 4 * k * (k + 1)) ∨ (∃ k : ℕ, n = 2 * k * k - 1) :=
sorry

end NUMINAMATH_CALUDE_expression_perfect_square_iff_l1512_151214


namespace NUMINAMATH_CALUDE_alternating_sum_coefficients_l1512_151283

theorem alternating_sum_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -1 := by
sorry

end NUMINAMATH_CALUDE_alternating_sum_coefficients_l1512_151283


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l1512_151269

/-- Given a purchase with total cost, sales tax, and tax rate, calculate the cost of tax-free items -/
theorem tax_free_items_cost
  (total_cost : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h1 : total_cost = 40)
  (h2 : sales_tax = 0.30)
  (h3 : tax_rate = 0.06)
  : ∃ (tax_free_cost : ℝ), tax_free_cost = total_cost - sales_tax / tax_rate :=
by
  sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l1512_151269


namespace NUMINAMATH_CALUDE_triangle_problem_l1512_151295

noncomputable section

/-- Given a triangle ABC with the following properties:
  BC = √5
  AC = 3
  sin C = 2 * sin A
  Prove that:
  1. AB = 2√5
  2. sin(2A - π/4) = √2/10
-/
theorem triangle_problem (A B C : ℝ) (h1 : Real.sqrt 5 = BC)
  (h2 : 3 = AC) (h3 : Real.sin C = 2 * Real.sin A) :
  AB = 2 * Real.sqrt 5 ∧ Real.sin (2 * A - π / 4) = Real.sqrt 2 / 10 :=
by sorry

end

end NUMINAMATH_CALUDE_triangle_problem_l1512_151295


namespace NUMINAMATH_CALUDE_mean_of_data_l1512_151213

def data : List ℕ := [7, 5, 3, 5, 10]

theorem mean_of_data : (data.sum : ℚ) / data.length = 6 := by sorry

end NUMINAMATH_CALUDE_mean_of_data_l1512_151213


namespace NUMINAMATH_CALUDE_parentheses_placement_l1512_151226

theorem parentheses_placement : 90 - 72 / (6 + 3) = 82 := by sorry

end NUMINAMATH_CALUDE_parentheses_placement_l1512_151226


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1512_151292

theorem min_value_of_expression (x : ℝ) :
  let f := λ x : ℝ => Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((x - 1)^2 + (x - 1)^2)
  (∀ x, f x ≥ 1) ∧ (∃ x, f x = 1) := by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1512_151292


namespace NUMINAMATH_CALUDE_min_value_f_min_value_achieved_l1512_151257

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem min_value_f :
  ∀ x : ℝ, x ≥ 0 → f x ≥ 1 := by
  sorry

theorem min_value_achieved :
  ∃ x : ℝ, x ≥ 0 ∧ f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_min_value_achieved_l1512_151257


namespace NUMINAMATH_CALUDE_farm_animals_count_l1512_151254

theorem farm_animals_count (rabbits chickens : ℕ) : 
  rabbits = chickens + 17 → 
  rabbits = 64 → 
  rabbits + chickens = 111 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_count_l1512_151254


namespace NUMINAMATH_CALUDE_five_classes_in_school_l1512_151296

/-- Represents the number of students in each class -/
def class_sizes (n : ℕ) : ℕ → ℕ
  | 0 => 25
  | i + 1 => class_sizes n i - 2

/-- The total number of students in the school -/
def total_students (n : ℕ) : ℕ :=
  (List.range n).map (class_sizes n) |>.sum

/-- The theorem stating that there are 5 classes in the school -/
theorem five_classes_in_school :
  ∃ n : ℕ, n > 0 ∧ total_students n = 105 ∧ n = 5 :=
sorry

end NUMINAMATH_CALUDE_five_classes_in_school_l1512_151296


namespace NUMINAMATH_CALUDE_lawn_mowing_time_mowing_time_approx_2_3_l1512_151246

/-- Represents the lawn mowing problem -/
theorem lawn_mowing_time (lawn_length lawn_width : ℝ) 
                         (swath_width overlap : ℝ) 
                         (mowing_speed : ℝ) : ℝ :=
  let effective_swath := swath_width - overlap
  let num_strips := lawn_width / effective_swath
  let total_distance := num_strips * lawn_length
  let time_taken := total_distance / mowing_speed
  time_taken

/-- Proves that the time taken to mow the lawn is approximately 2.3 hours -/
theorem mowing_time_approx_2_3 :
  ∃ ε > 0, |lawn_mowing_time 120 180 (30/12) (2/12) 4000 - 2.3| < ε :=
sorry

end NUMINAMATH_CALUDE_lawn_mowing_time_mowing_time_approx_2_3_l1512_151246


namespace NUMINAMATH_CALUDE_negation_of_proposition_is_true_l1512_151276

theorem negation_of_proposition_is_true : 
  (∃ a : ℝ, a > 2 ∧ a^2 ≥ 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_is_true_l1512_151276


namespace NUMINAMATH_CALUDE_kylie_coins_left_l1512_151244

/-- The number of coins Kylie collected and gave away -/
structure CoinCollection where
  piggy_bank : ℕ
  from_brother : ℕ
  from_father : ℕ
  given_away : ℕ

/-- Calculate the number of coins Kylie has left -/
def coins_left (c : CoinCollection) : ℕ :=
  c.piggy_bank + c.from_brother + c.from_father - c.given_away

/-- Theorem stating that Kylie has 15 coins left -/
theorem kylie_coins_left :
  ∀ (c : CoinCollection),
  c.piggy_bank = 15 →
  c.from_brother = 13 →
  c.from_father = 8 →
  c.given_away = 21 →
  coins_left c = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_kylie_coins_left_l1512_151244


namespace NUMINAMATH_CALUDE_parabola_equation_l1512_151238

/-- Given a parabola and a circle, prove the equation of the parabola -/
theorem parabola_equation (p : ℝ) (hp : p ≠ 0) :
  (∀ x y, x^2 = 2*p*y) →  -- Parabola equation
  (∀ x y, (x - 2)^2 + (y - 1)^2 = 1) →  -- Circle equation
  (∃ y, ∀ x, (x - 2)^2 + (y - 1)^2 = 1 ∧ y = -p/2) →  -- Axis of parabola is tangent to circle
  (∀ x y, x^2 = -8*y) :=  -- Conclusion: equation of the parabola
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1512_151238


namespace NUMINAMATH_CALUDE_jason_remaining_cards_l1512_151265

/-- The number of Pokemon cards Jason started with -/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has now -/
def remaining_cards : ℕ := initial_cards - cards_given_away

theorem jason_remaining_cards : remaining_cards = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_remaining_cards_l1512_151265


namespace NUMINAMATH_CALUDE_correct_minus_incorrect_l1512_151294

/-- Calculates the result following the order of operations -/
def J : ℤ := 12 - (3 * 4)

/-- Calculates the result ignoring parentheses and going from left to right -/
def A : ℤ := (12 - 3) * 4

/-- The difference between the correct calculation and the incorrect one -/
theorem correct_minus_incorrect : J - A = -36 := by sorry

end NUMINAMATH_CALUDE_correct_minus_incorrect_l1512_151294


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1512_151291

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1512_151291


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l1512_151275

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem complement_of_union_M_N : 
  (M ∪ N)ᶜ = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l1512_151275


namespace NUMINAMATH_CALUDE_point_on_line_l1512_151255

/-- Given a point P(x, b) on the line x + y = 30, if the slope of OP is 4 (where O is the origin), then b = 24. -/
theorem point_on_line (x b : ℝ) : 
  x + b = 30 →  -- P(x, b) is on the line x + y = 30
  (b / x = 4) →  -- The slope of OP is 4
  b = 24 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l1512_151255


namespace NUMINAMATH_CALUDE_inequality_proof_l1512_151216

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : n ≥ 2) :
  (1 + x)^n + (1 - x)^n < 2^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1512_151216


namespace NUMINAMATH_CALUDE_muffin_ratio_l1512_151250

/-- The number of muffins Sasha made -/
def sasha_muffins : ℕ := 30

/-- The price of each muffin in dollars -/
def muffin_price : ℕ := 4

/-- The total amount raised in dollars -/
def total_raised : ℕ := 900

/-- The number of muffins Melissa made -/
def melissa_muffins : ℕ := 120

/-- The number of muffins Tiffany made -/
def tiffany_muffins : ℕ := (sasha_muffins + melissa_muffins) / 2

/-- The total number of muffins made -/
def total_muffins : ℕ := sasha_muffins + melissa_muffins + tiffany_muffins

theorem muffin_ratio : 
  (total_muffins * muffin_price = total_raised) → 
  (melissa_muffins : ℚ) / sasha_muffins = 4 := by
sorry

end NUMINAMATH_CALUDE_muffin_ratio_l1512_151250


namespace NUMINAMATH_CALUDE_perfect_square_triples_l1512_151280

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem perfect_square_triples :
  ∀ a b c : ℕ,
    (is_perfect_square (a^2 + 2*b + c) ∧
     is_perfect_square (b^2 + 2*c + a) ∧
     is_perfect_square (c^2 + 2*a + b)) →
    ((a = 0 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 1 ∧ c = 1) ∨
     (a = 43 ∧ b = 127 ∧ c = 106)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_triples_l1512_151280


namespace NUMINAMATH_CALUDE_min_class_size_class_size_32_achievable_l1512_151285

/-- Represents the number of people in each group of the class --/
structure ClassGroups where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the tree-planting problem --/
def TreePlantingConditions (g : ClassGroups) : Prop :=
  g.second = (g.first + g.third) / 3 ∧
  4 * g.second = 5 * g.first + 3 * g.third - 72

/-- The theorem stating the minimum number of people in the class --/
theorem min_class_size (g : ClassGroups) 
  (h : TreePlantingConditions g) : 
  g.first + g.second + g.third ≥ 32 := by
  sorry

/-- The theorem stating that 32 is achievable --/
theorem class_size_32_achievable : 
  ∃ g : ClassGroups, TreePlantingConditions g ∧ g.first + g.second + g.third = 32 := by
  sorry

end NUMINAMATH_CALUDE_min_class_size_class_size_32_achievable_l1512_151285


namespace NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l1512_151259

theorem sum_of_squares_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let s₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let s₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → s₁^2 + s₂^2 = (b^2 - 2*a*c) / a^2 := by
  sorry

theorem sum_of_squares_specific_quadratic :
  let s₁ := (15 + Real.sqrt 201) / 2
  let s₂ := (15 - Real.sqrt 201) / 2
  x^2 - 15*x + 6 = 0 → s₁^2 + s₂^2 = 213 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l1512_151259


namespace NUMINAMATH_CALUDE_anne_heavier_than_douglas_l1512_151217

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := 52

/-- The difference in weight between Anne and Douglas -/
def weight_difference : ℕ := anne_weight - douglas_weight

/-- Theorem stating that Anne is 15 pounds heavier than Douglas -/
theorem anne_heavier_than_douglas : weight_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_anne_heavier_than_douglas_l1512_151217


namespace NUMINAMATH_CALUDE_haley_josh_necklace_difference_haley_josh_necklace_difference_proof_l1512_151281

/-- Given the number of necklaces for Haley, Jason, and Josh, prove that Haley has 15 more necklaces than Josh. -/
theorem haley_josh_necklace_difference : ℕ → ℕ → ℕ → Prop :=
  fun haley jason josh =>
    (haley = jason + 5) →
    (josh = jason / 2) →
    (haley = 25) →
    (haley - josh = 15)

/-- Proof of the theorem -/
theorem haley_josh_necklace_difference_proof :
  ∀ haley jason josh, haley_josh_necklace_difference haley jason josh :=
by
  sorry

#check haley_josh_necklace_difference
#check haley_josh_necklace_difference_proof

end NUMINAMATH_CALUDE_haley_josh_necklace_difference_haley_josh_necklace_difference_proof_l1512_151281


namespace NUMINAMATH_CALUDE_impossibleAllGood_l1512_151243

/-- A mushroom is either good or bad -/
inductive MushroomType
  | Good
  | Bad

/-- Definition of a mushroom -/
structure Mushroom where
  wormCount : ℕ
  type : MushroomType

/-- A basket of mushrooms -/
structure Basket where
  mushrooms : List Mushroom

/-- Function to determine if a mushroom is good -/
def isGoodMushroom (m : Mushroom) : Prop :=
  m.wormCount < 10

/-- Initial basket setup -/
def initialBasket : Basket :=
  { mushrooms := List.append
      (List.replicate 100 { wormCount := 10, type := MushroomType.Bad })
      (List.replicate 11 { wormCount := 0, type := MushroomType.Good }) }

/-- Theorem: It's impossible for all mushrooms to become good after redistribution -/
theorem impossibleAllGood (b : Basket) : ¬ ∀ m ∈ b.mushrooms, isGoodMushroom m := by
  sorry

#check impossibleAllGood initialBasket

end NUMINAMATH_CALUDE_impossibleAllGood_l1512_151243


namespace NUMINAMATH_CALUDE_parallel_segments_theorem_l1512_151247

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ

/-- Represents three parallel line segments intersecting another line segment -/
structure ParallelSegments where
  ab : Segment
  ef : Segment
  cd : Segment
  bc : Segment
  ab_parallel_ef : Bool
  ef_parallel_cd : Bool

/-- Given three parallel line segments intersecting another line segment,
    with specific lengths, the middle segment's length is 16 -/
theorem parallel_segments_theorem (p : ParallelSegments)
  (h1 : p.ab_parallel_ef = true)
  (h2 : p.ef_parallel_cd = true)
  (h3 : p.ab.length = 20)
  (h4 : p.cd.length = 80)
  (h5 : p.bc.length = 100) :
  p.ef.length = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_theorem_l1512_151247


namespace NUMINAMATH_CALUDE_log_inequality_l1512_151266

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x) < x := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1512_151266


namespace NUMINAMATH_CALUDE_unsold_books_count_l1512_151202

/-- Proves that the number of unsold books is 36 given the sale conditions --/
theorem unsold_books_count (total_books : ℕ) : 
  (2 : ℚ) / 3 * total_books * (7 : ℚ) / 2 = 252 → 
  (1 : ℚ) / 3 * total_books = 36 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_count_l1512_151202


namespace NUMINAMATH_CALUDE_soccer_stars_draw_points_l1512_151253

/-- Represents a soccer team's season statistics -/
structure SoccerTeamStats where
  total_games : ℕ
  games_won : ℕ
  games_lost : ℕ
  points_per_win : ℕ
  total_points : ℕ

/-- Calculates the points earned for a draw given a team's season statistics -/
def points_per_draw (stats : SoccerTeamStats) : ℕ :=
  let games_drawn := stats.total_games - stats.games_won - stats.games_lost
  let points_from_wins := stats.games_won * stats.points_per_win
  let points_from_draws := stats.total_points - points_from_wins
  points_from_draws / games_drawn

/-- Theorem stating that Team Soccer Stars earns 1 point for each draw -/
theorem soccer_stars_draw_points :
  let stats : SoccerTeamStats := {
    total_games := 20,
    games_won := 14,
    games_lost := 2,
    points_per_win := 3,
    total_points := 46
  }
  points_per_draw stats = 1 := by sorry

end NUMINAMATH_CALUDE_soccer_stars_draw_points_l1512_151253


namespace NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l1512_151209

theorem negation_of_cube_odd_is_odd :
  ¬(∀ x : ℤ, Odd x → Odd (x^3)) ↔ ∃ x : ℤ, Odd x ∧ ¬Odd (x^3) :=
sorry

end NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l1512_151209


namespace NUMINAMATH_CALUDE_z_magnitude_l1512_151221

open Complex

/-- Euler's formula -/
axiom euler_formula (θ : ℝ) : exp (I * θ) = cos θ + I * sin θ

/-- The complex number z satisfies the given equation -/
def z : ℂ := by sorry

/-- The equation that z satisfies -/
axiom z_equation : (exp (I * Real.pi) - I) * z = 1

/-- The magnitude of z is √2/2 -/
theorem z_magnitude : abs z = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_z_magnitude_l1512_151221


namespace NUMINAMATH_CALUDE_initial_water_percentage_in_milk_initial_water_percentage_is_five_percent_l1512_151228

/-- Proves that the initial water percentage in milk is 5% given the specified conditions -/
theorem initial_water_percentage_in_milk : ℝ → Prop :=
  fun initial_percentage =>
    let initial_volume : ℝ := 10
    let pure_milk_added : ℝ := 15
    let final_percentage : ℝ := 2
    let final_volume : ℝ := 25
    let initial_water_volume := (initial_percentage / 100) * initial_volume
    let final_water_volume := (final_percentage / 100) * final_volume
    initial_water_volume = final_water_volume ∧ initial_percentage = 5

/-- The initial water percentage in milk is 5% -/
theorem initial_water_percentage_is_five_percent : 
  initial_water_percentage_in_milk 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_percentage_in_milk_initial_water_percentage_is_five_percent_l1512_151228


namespace NUMINAMATH_CALUDE_value_swap_l1512_151236

theorem value_swap (a b : ℕ) (h1 : a = 1) (h2 : b = 2) :
  let c := a
  let a' := b
  let b' := c
  (a', b', c) = (2, 1, 1) := by sorry

end NUMINAMATH_CALUDE_value_swap_l1512_151236


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l1512_151208

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 0, 4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ c = -1/12 ∧ d = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l1512_151208


namespace NUMINAMATH_CALUDE_jerrys_coins_l1512_151273

theorem jerrys_coins (n d : ℕ) : 
  n + d = 30 →
  5 * n + 10 * d + 140 = 10 * n + 5 * d →
  5 * n + 10 * d = 155 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_coins_l1512_151273


namespace NUMINAMATH_CALUDE_solution_set_implies_range_l1512_151287

/-- The solution set of the inequality ax^2 + ax - 4 < 0 is ℝ -/
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x, a * x^2 + a * x - 4 < 0

/-- The range of a is (-16, 0] -/
def range_of_a : Set ℝ := Set.Ioc (-16) 0

theorem solution_set_implies_range :
  (∃ a, solution_set_is_reals a) → (∀ a, solution_set_is_reals a ↔ a ∈ range_of_a) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_range_l1512_151287


namespace NUMINAMATH_CALUDE_compute_M_v_minus_2w_l1512_151245

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (v w : Fin 2 → ℝ)

axiom Mv : M.mulVec v = ![4, 2]
axiom Mw : M.mulVec w = ![5, 1]

theorem compute_M_v_minus_2w :
  M.mulVec (v - 2 • w) = ![-6, 0] := by sorry

end NUMINAMATH_CALUDE_compute_M_v_minus_2w_l1512_151245


namespace NUMINAMATH_CALUDE_solution_set_intersection_range_l1512_151258

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2*|x + 1|

-- Part I
theorem solution_set (x : ℝ) : 
  x ∈ Set.Ioo (-4/3 : ℝ) 1 ↔ f 5 x > 2 := by sorry

-- Part II
theorem intersection_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = x^2 + 2*x + 3 ∧ y = f m x) ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_intersection_range_l1512_151258


namespace NUMINAMATH_CALUDE_joes_fruit_spending_l1512_151220

theorem joes_fruit_spending (total_money : ℚ) (chocolate_fraction : ℚ) (money_left : ℚ) : 
  total_money = 450 →
  chocolate_fraction = 1/9 →
  money_left = 220 →
  (total_money - chocolate_fraction * total_money - money_left) / total_money = 2/5 := by
sorry

end NUMINAMATH_CALUDE_joes_fruit_spending_l1512_151220


namespace NUMINAMATH_CALUDE_division_of_fractions_l1512_151270

theorem division_of_fractions : (7 : ℚ) / (8 / 13) = 91 / 8 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l1512_151270


namespace NUMINAMATH_CALUDE_four_player_tournament_games_l1512_151251

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 4 players, where each player plays against every
    other player exactly once, the total number of games played is 6. -/
theorem four_player_tournament_games :
  num_games 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_four_player_tournament_games_l1512_151251


namespace NUMINAMATH_CALUDE_joe_hvac_zones_l1512_151231

def hvac_system (total_cost : ℕ) (vents_per_zone : ℕ) (cost_per_vent : ℕ) : ℕ :=
  (total_cost / cost_per_vent) / vents_per_zone

theorem joe_hvac_zones :
  hvac_system 20000 5 2000 = 2 := by
  sorry

end NUMINAMATH_CALUDE_joe_hvac_zones_l1512_151231


namespace NUMINAMATH_CALUDE_smallest_floor_x_l1512_151204

-- Define a tetrahedron type
structure Tetrahedron :=
  (a b c d e x : ℝ)

-- Define the conditions for a valid tetrahedron
def is_valid_tetrahedron (t : Tetrahedron) : Prop :=
  t.a = 4 ∧ t.b = 7 ∧ t.c = 20 ∧ t.d = 22 ∧ t.e = 28 ∧
  t.x > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b ∧
  t.a + t.d > t.e ∧ t.d + t.e > t.a ∧ t.e + t.a > t.d ∧
  t.b + t.d > t.x ∧ t.d + t.x > t.b ∧ t.x + t.b > t.d ∧
  t.b + t.e > t.c ∧ t.e + t.c > t.b ∧ t.c + t.b > t.e ∧
  t.c + t.d > t.x ∧ t.d + t.x > t.c ∧ t.x + t.c > t.d ∧
  t.c + t.e > t.x ∧ t.e + t.x > t.c ∧ t.x + t.c > t.e ∧
  t.d + t.e > t.x ∧ t.e + t.x > t.d ∧ t.x + t.d > t.e

-- Theorem statement
theorem smallest_floor_x (t : Tetrahedron) (h : is_valid_tetrahedron t) :
  ∀ (y : ℝ), (is_valid_tetrahedron {a := t.a, b := t.b, c := t.c, d := t.d, e := t.e, x := y} →
  ⌊t.x⌋ ≥ 8) ∧ (∃ (z : ℝ), is_valid_tetrahedron {a := t.a, b := t.b, c := t.c, d := t.d, e := t.e, x := z} ∧ ⌊z⌋ = 8) :=
sorry

end NUMINAMATH_CALUDE_smallest_floor_x_l1512_151204


namespace NUMINAMATH_CALUDE_age_squares_sum_l1512_151249

theorem age_squares_sum (d t h : ℕ) : 
  t = 2 * d ∧ 
  h^2 + 4 * d = 5 * t ∧ 
  3 * h^2 = 7 * d^2 + 2 * t^2 →
  d^2 + h^2 + t^2 = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_age_squares_sum_l1512_151249


namespace NUMINAMATH_CALUDE_room_dimension_increase_l1512_151234

/-- 
  Given a rectangular room where increasing both length and breadth by y feet
  increases the perimeter by 16 feet, prove that y equals 4 feet.
-/
theorem room_dimension_increase (L B : ℝ) (y : ℝ) 
  (h : 2 * ((L + y) + (B + y)) = 2 * (L + B) + 16) : y = 4 :=
by sorry

end NUMINAMATH_CALUDE_room_dimension_increase_l1512_151234


namespace NUMINAMATH_CALUDE_room_length_proof_l1512_151229

/-- Given a room with known width, total paving cost, and paving rate per square meter,
    prove that the length of the room is 5.5 meters. -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) 
    (h1 : width = 3.75)
    (h2 : total_cost = 16500)
    (h3 : rate_per_sq_meter = 800) : 
  total_cost / rate_per_sq_meter / width = 5.5 := by
  sorry


end NUMINAMATH_CALUDE_room_length_proof_l1512_151229


namespace NUMINAMATH_CALUDE_slope_of_line_l1512_151289

theorem slope_of_line (x y : ℝ) :
  4 * x - 7 * y = 14 → (y - (-2)) / (x - 0) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l1512_151289


namespace NUMINAMATH_CALUDE_polynomial_real_root_l1512_151218

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^4 + a*x^3 - x^2 + a*x + 1 = 0) ↔ 
  (a ≤ -1/2 ∨ a ≥ 1/2) := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l1512_151218


namespace NUMINAMATH_CALUDE_rational_results_l1512_151274

-- Define the natural logarithm (ln) and common logarithm (lg)
noncomputable def ln (x : ℝ) := Real.log x
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the logarithm with an arbitrary base
noncomputable def log (b : ℝ) (x : ℝ) := ln x / ln b

-- State the theorem
theorem rational_results :
  (2 * lg 2 + lg 25 = 2) ∧
  (3^(1 / ln 3) - Real.exp 1 = 0) ∧
  (log 4 3 * log 3 6 * log 6 8 = 3/2) := by sorry

end NUMINAMATH_CALUDE_rational_results_l1512_151274


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1512_151205

theorem right_triangle_sets : ∃! (a b c : ℝ), (a = 6 ∧ b = 8 ∧ c = 13) ∧
  ¬(a^2 + b^2 = c^2) ∧
  (0.3^2 + 0.4^2 = 0.5^2) ∧
  (1^2 + 1^2 = (Real.sqrt 2)^2) ∧
  (8^2 + 15^2 = 17^2) :=
by
  sorry

#check right_triangle_sets

end NUMINAMATH_CALUDE_right_triangle_sets_l1512_151205


namespace NUMINAMATH_CALUDE_solve_star_equation_l1512_151278

/-- Custom binary operation -/
def star (a b : ℚ) : ℚ := a * b + 3 * b - 2 * a

/-- Theorem stating the solution to the equation -/
theorem solve_star_equation : ∃ x : ℚ, star 3 x = 23 ∧ x = 29 / 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l1512_151278


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l1512_151279

/-- Represents the ages of Arun and Deepak -/
structure Ages where
  arun : ℕ
  deepak : ℕ

/-- The ratio of two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of two natural numbers -/
def calculateRatio (a b : ℕ) : Ratio :=
  let gcd := Nat.gcd a b
  { numerator := a / gcd, denominator := b / gcd }

/-- Theorem stating the ratio of Arun's and Deepak's ages -/
theorem age_ratio_theorem (ages : Ages) : 
  ages.deepak = 42 → 
  ages.arun + 6 = 36 → 
  calculateRatio ages.arun ages.deepak = Ratio.mk 5 7 := by
  sorry

#check age_ratio_theorem

end NUMINAMATH_CALUDE_age_ratio_theorem_l1512_151279


namespace NUMINAMATH_CALUDE_each_brother_pays_19_80_l1512_151260

/-- The amount each brother pays when buying cakes and splitting the cost -/
def amount_per_person (num_cakes : ℕ) (price_per_cake : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_before_tax := num_cakes * price_per_cake
  let tax_amount := total_before_tax * tax_rate
  let total_after_tax := total_before_tax + tax_amount
  total_after_tax / 2

/-- Theorem stating that each brother pays $19.80 -/
theorem each_brother_pays_19_80 :
  amount_per_person 3 12 (1/10) = 198/10 := by
  sorry

end NUMINAMATH_CALUDE_each_brother_pays_19_80_l1512_151260


namespace NUMINAMATH_CALUDE_divisor_count_squared_lt_4n_l1512_151242

def divisor_count (n : ℕ+) : ℕ := (Nat.divisors n.val).card

theorem divisor_count_squared_lt_4n (n : ℕ+) : (divisor_count n)^2 < 4 * n.val := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_squared_lt_4n_l1512_151242


namespace NUMINAMATH_CALUDE_matt_weight_matt_weight_is_80kg_l1512_151223

/-- Given Matt's protein intake and requirements, calculate his weight. -/
theorem matt_weight (protein_percentage : ℝ) (protein_per_kg : ℝ) (powder_per_week : ℝ) : ℝ :=
  let protein_per_day := (powder_per_week / 7) * protein_percentage
  protein_per_day / protein_per_kg

/-- Prove that Matt weighs 80 kilograms given his protein intake and requirements. -/
theorem matt_weight_is_80kg : 
  matt_weight 0.80 2 1400 = 80 := by
  sorry

end NUMINAMATH_CALUDE_matt_weight_matt_weight_is_80kg_l1512_151223


namespace NUMINAMATH_CALUDE_dilation_matrix_determinant_l1512_151256

theorem dilation_matrix_determinant :
  ∀ (E : Matrix (Fin 3) (Fin 3) ℝ),
  (∀ i j : Fin 3, i ≠ j → E i j = 0) →
  E 0 0 = 3 →
  E 1 1 = 5 →
  E 2 2 = 7 →
  Matrix.det E = 105 := by
sorry

end NUMINAMATH_CALUDE_dilation_matrix_determinant_l1512_151256


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1512_151222

/-- Given a rectangle with length thrice its breadth and area 588 square meters,
    prove that its perimeter is 112 meters. -/
theorem rectangle_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) : 
  length = 3 * breadth →
  area = 588 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 112 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1512_151222


namespace NUMINAMATH_CALUDE_count_distinct_arrangements_l1512_151237

/-- A regular five-pointed star with 10 positions for placing objects -/
structure StarArrangement where
  positions : Fin 10 → Fin 10

/-- The group of symmetries of a regular five-pointed star -/
def starSymmetryGroup : Fintype G := sorry

/-- The number of distinct arrangements of 10 different objects on a regular five-pointed star,
    considering rotations and reflections as equivalent -/
def distinctArrangements : ℕ := sorry

/-- Theorem stating the number of distinct arrangements -/
theorem count_distinct_arrangements :
  distinctArrangements = Nat.factorial 10 / 10 := by sorry

end NUMINAMATH_CALUDE_count_distinct_arrangements_l1512_151237


namespace NUMINAMATH_CALUDE_certain_number_proof_l1512_151297

theorem certain_number_proof (x : ℝ) (h : x = 3) :
  ∃ y : ℝ, (x + y) / (x + y + 5) = (x + y + 5) / (x + y + 5 + 13) ∧ y = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1512_151297


namespace NUMINAMATH_CALUDE_volume_between_concentric_spheres_l1512_151207

theorem volume_between_concentric_spheres :
  let r₁ : ℝ := 3  -- radius of smaller sphere
  let r₂ : ℝ := 6  -- radius of larger sphere
  let V₁ := (4 / 3) * π * r₁^3  -- volume of smaller sphere
  let V₂ := (4 / 3) * π * r₂^3  -- volume of larger sphere
  V₂ - V₁ = 252 * π := by sorry

end NUMINAMATH_CALUDE_volume_between_concentric_spheres_l1512_151207


namespace NUMINAMATH_CALUDE_range_of_m_l1512_151277

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x - 6

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-10) (-6)) ∧
  (∃ x ∈ Set.Icc 0 m, f x = -10) ∧
  (∃ x ∈ Set.Icc 0 m, f x = -6) →
  m ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1512_151277
