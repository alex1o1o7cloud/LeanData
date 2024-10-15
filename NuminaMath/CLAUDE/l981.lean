import Mathlib

namespace NUMINAMATH_CALUDE_smallest_covering_l981_98121

/-- A rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- A configuration of rectangles covering a larger rectangle -/
structure Configuration where
  covering : Rectangle
  tiles : List Rectangle

/-- The total area covered by a list of rectangles -/
def total_area (tiles : List Rectangle) : ℕ := tiles.foldl (fun acc r => acc + r.area) 0

/-- A valid configuration has no gaps or overhangs -/
def Configuration.valid (c : Configuration) : Prop :=
  c.covering.area = total_area c.tiles

/-- The smallest valid configuration for covering with 3x4 rectangles -/
def smallest_valid_configuration : Configuration :=
  { covering := { width := 6, height := 8 }
  , tiles := List.replicate 4 { width := 3, height := 4 } }

theorem smallest_covering :
  smallest_valid_configuration.valid ∧
  (∀ c : Configuration, c.valid → c.covering.area ≥ smallest_valid_configuration.covering.area) ∧
  smallest_valid_configuration.tiles.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_covering_l981_98121


namespace NUMINAMATH_CALUDE_toys_per_day_l981_98104

/-- A factory produces toys according to the following conditions:
  1. The factory produces 4560 toys per week.
  2. The workers work 4 days a week.
  3. The same number of toys is made every day.
-/
def factory_production (toys_per_day : ℕ) : Prop :=
  toys_per_day * 4 = 4560 ∧ toys_per_day > 0

/-- The number of toys produced each day is 1140. -/
theorem toys_per_day : ∃ (n : ℕ), factory_production n ∧ n = 1140 :=
  sorry

end NUMINAMATH_CALUDE_toys_per_day_l981_98104


namespace NUMINAMATH_CALUDE_marker_distance_l981_98188

theorem marker_distance (k : ℝ) (h1 : k > 0) 
  (h2 : ∀ n : ℕ+, Real.sqrt ((4:ℝ)^2 + (4*k)^2) = 31) : 
  Real.sqrt ((12:ℝ)^2 + (12*k)^2) = 93 := by sorry

end NUMINAMATH_CALUDE_marker_distance_l981_98188


namespace NUMINAMATH_CALUDE_polygon_edges_l981_98191

theorem polygon_edges (n : ℕ) : n ≥ 3 → (
  (n - 2) * 180 = 4 * 360 + 180 ↔ n = 11
) := by sorry

end NUMINAMATH_CALUDE_polygon_edges_l981_98191


namespace NUMINAMATH_CALUDE_gas_supply_equilibrium_l981_98154

/-- The distance between points A and B in kilometers -/
def total_distance : ℝ := 500

/-- The amount of gas extracted from reservoir A in cubic meters per minute -/
def gas_from_A : ℝ := 10000

/-- The rate of gas leakage in cubic meters per kilometer -/
def leakage_rate : ℝ := 4

/-- The distance between point A and city C in kilometers -/
def distance_AC : ℝ := 100

theorem gas_supply_equilibrium :
  let gas_to_C_from_A := gas_from_A - leakage_rate * distance_AC
  let gas_to_C_from_B := (gas_from_A * 1.12) - leakage_rate * (total_distance - distance_AC)
  gas_to_C_from_A = gas_to_C_from_B :=
by sorry

end NUMINAMATH_CALUDE_gas_supply_equilibrium_l981_98154


namespace NUMINAMATH_CALUDE_margarets_mean_score_l981_98194

def scores : List ℝ := [88, 90, 94, 95, 96, 99]

theorem margarets_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ (cyprian_scores : List ℝ) (margaret_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        margaret_scores.length = 2 ∧ 
        cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
        cyprian_scores.length = 4 ∧ 
        cyprian_scores.sum / cyprian_scores.length = 92) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 2 ∧ 
    margaret_scores.sum / margaret_scores.length = 97 := by
  sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l981_98194


namespace NUMINAMATH_CALUDE_fourth_root_of_1250000_l981_98186

theorem fourth_root_of_1250000 : (1250000 : ℝ) ^ (1/4 : ℝ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_1250000_l981_98186


namespace NUMINAMATH_CALUDE_irrational_in_set_l981_98198

-- Define the set of numbers
def numbers : Set ℝ := {0, -2, Real.sqrt 3, 1/2}

-- Define a predicate for rational numbers
def isRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Theorem statement
theorem irrational_in_set :
  ∃ (x : ℝ), x ∈ numbers ∧ ¬(isRational x) ∧
  ∀ (y : ℝ), y ∈ numbers ∧ y ≠ x → isRational y :=
sorry

end NUMINAMATH_CALUDE_irrational_in_set_l981_98198


namespace NUMINAMATH_CALUDE_min_area_line_equation_l981_98155

/-- The equation of the line passing through (3, 1) that minimizes the area of the triangle formed by its x and y intercepts and the origin --/
theorem min_area_line_equation :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ (x y : ℝ), x / a + y / b = 1 → (3 / a + 1 / b = 1)) ∧
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 
    (∀ (x y : ℝ), x / a' + y / b' = 1 → (3 / a' + 1 / b' = 1)) →
    a * b ≤ a' * b') ∧
  a = 6 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_area_line_equation_l981_98155


namespace NUMINAMATH_CALUDE_divisor_problem_l981_98142

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 149 →
  quotient = 9 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 16 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l981_98142


namespace NUMINAMATH_CALUDE_george_marbles_count_l981_98139

/-- The total number of marbles George collected -/
def total_marbles : ℕ := 50

/-- The number of yellow marbles -/
def yellow_marbles : ℕ := 12

/-- The number of red marbles -/
def red_marbles : ℕ := 7

/-- The number of green marbles -/
def green_marbles : ℕ := yellow_marbles / 2

/-- The number of white marbles -/
def white_marbles : ℕ := total_marbles / 2

theorem george_marbles_count :
  total_marbles = white_marbles + yellow_marbles + green_marbles + red_marbles :=
by sorry

end NUMINAMATH_CALUDE_george_marbles_count_l981_98139


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l981_98165

/-- Given a geometric sequence {a_n} with positive terms, prove that if a_6 + a_5 = 4 
    and a_4 + a_3 - a_2 - a_1 = 1, then a_1 = √2 - 1 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence definition
  a 6 + a 5 = 4 →  -- First given equation
  a 4 + a 3 - a 2 - a 1 = 1 →  -- Second given equation
  a 1 = Real.sqrt 2 - 1 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l981_98165


namespace NUMINAMATH_CALUDE_trees_in_park_l981_98187

/-- The number of trees after n years, given an initial number and annual growth rate. -/
def trees_after_years (initial : ℕ) (growth_rate : ℚ) (years : ℕ) : ℚ :=
  initial * (1 + growth_rate) ^ years

/-- Theorem stating that given 5000 trees initially and 30% annual growth,
    the number of trees after 3 years is 10985. -/
theorem trees_in_park (initial : ℕ) (growth_rate : ℚ) (years : ℕ) 
  (h_initial : initial = 5000)
  (h_growth : growth_rate = 3/10)
  (h_years : years = 3) :
  trees_after_years initial growth_rate years = 10985 := by
  sorry

#eval trees_after_years 5000 (3/10) 3

end NUMINAMATH_CALUDE_trees_in_park_l981_98187


namespace NUMINAMATH_CALUDE_delta_y_over_delta_x_l981_98166

/-- The function f(x) = 2x^2 + 5 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 5

/-- Theorem stating that for the given function and points, Δy / Δx = 2Δx + 4 -/
theorem delta_y_over_delta_x (Δx : ℝ) (Δy : ℝ) :
  f 1 = 7 →
  f (1 + Δx) = 7 + Δy →
  Δy / Δx = 2 * Δx + 4 :=
by
  sorry

end NUMINAMATH_CALUDE_delta_y_over_delta_x_l981_98166


namespace NUMINAMATH_CALUDE_ground_beef_cost_l981_98127

/-- The cost of ground beef in dollars per kilogram -/
def price_per_kg : ℝ := 5

/-- The quantity of ground beef in kilograms -/
def quantity : ℝ := 12

/-- The total cost of ground beef -/
def total_cost : ℝ := price_per_kg * quantity

theorem ground_beef_cost : total_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_ground_beef_cost_l981_98127


namespace NUMINAMATH_CALUDE_isabel_paper_count_l981_98147

/-- The number of pieces of paper Isabel used -/
def used : ℕ := 156

/-- The number of pieces of paper Isabel has left -/
def left : ℕ := 744

/-- The initial number of pieces of paper Isabel bought -/
def initial : ℕ := used + left

theorem isabel_paper_count : initial = 900 := by
  sorry

end NUMINAMATH_CALUDE_isabel_paper_count_l981_98147


namespace NUMINAMATH_CALUDE_not_hearing_favorite_song_probability_l981_98108

/-- Represents the duration of a song in seconds -/
def SongDuration := ℕ

/-- Represents a playlist of songs -/
def Playlist := List SongDuration

/-- Calculates the duration of the nth song in the sequence -/
def nthSongDuration (n : ℕ) : SongDuration :=
  45 + 15 * n

/-- Generates a playlist of 12 songs with increasing durations -/
def generatePlaylist : Playlist :=
  List.range 12 |>.map nthSongDuration

/-- The duration of the favorite song in seconds -/
def favoriteSongDuration : SongDuration := 4 * 60

/-- The total duration we're interested in (5 minutes in seconds) -/
def totalDuration : SongDuration := 5 * 60

/-- Calculates the probability of not hearing the entire favorite song 
    within the first 5 minutes of a random playlist -/
def probabilityNotHearingFavoriteSong (playlist : Playlist) (favoriteDuration : SongDuration) (totalDuration : SongDuration) : ℚ :=
  sorry

theorem not_hearing_favorite_song_probability :
  probabilityNotHearingFavoriteSong generatePlaylist favoriteSongDuration totalDuration = 65 / 66 := by
  sorry

end NUMINAMATH_CALUDE_not_hearing_favorite_song_probability_l981_98108


namespace NUMINAMATH_CALUDE_classroom_arrangements_l981_98125

theorem classroom_arrangements (n : Nat) (h : n = 6) : 
  (Finset.range (n + 1)).sum (fun k => Nat.choose n k) - Nat.choose n 1 - Nat.choose n 0 = 57 := by
  sorry

end NUMINAMATH_CALUDE_classroom_arrangements_l981_98125


namespace NUMINAMATH_CALUDE_first_number_proof_l981_98145

theorem first_number_proof : ∃ x : ℝ, x + 2.017 + 0.217 + 2.0017 = 221.2357 ∧ x = 217 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l981_98145


namespace NUMINAMATH_CALUDE_f_neg_one_eq_neg_one_l981_98128

/-- Given a function f(x) = -2x^2 + 1, prove that f(-1) = -1 -/
theorem f_neg_one_eq_neg_one :
  let f : ℝ → ℝ := fun x ↦ -2 * x^2 + 1
  f (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_neg_one_l981_98128


namespace NUMINAMATH_CALUDE_ski_price_after_discounts_l981_98110

def original_price : ℝ := 200
def discount1 : ℝ := 0.40
def discount2 : ℝ := 0.20
def discount3 : ℝ := 0.10

theorem ski_price_after_discounts :
  let price1 := original_price * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let final_price := price2 * (1 - discount3)
  final_price = 86.40 := by sorry

end NUMINAMATH_CALUDE_ski_price_after_discounts_l981_98110


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_difference_l981_98109

theorem square_difference_of_sum_and_difference (x y : ℝ) 
  (h_sum : x + y = 20) (h_diff : x - y = 10) : x^2 - y^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_difference_l981_98109


namespace NUMINAMATH_CALUDE_difference_from_sum_and_difference_of_squares_l981_98168

theorem difference_from_sum_and_difference_of_squares
  (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  x - y = 4 := by
sorry

end NUMINAMATH_CALUDE_difference_from_sum_and_difference_of_squares_l981_98168


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l981_98185

-- Define the parabola
def parabola (x : ℝ) : ℝ := (2 - x) * x

-- State the theorem
theorem axis_of_symmetry :
  (∀ x : ℝ, parabola (1 + x) = parabola (1 - x)) ∧
  (∀ a : ℝ, a ≠ 1 → ∃ x : ℝ, parabola (a + x) ≠ parabola (a - x)) :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l981_98185


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l981_98123

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 7 + a 13 = 20 →
  a 9 + a 10 + a 11 = 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l981_98123


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l981_98197

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, (f x = y ∧ (3 * x^2 + 1 = 4)) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l981_98197


namespace NUMINAMATH_CALUDE_existence_of_m_and_k_l981_98159

def f (p : ℕ × ℕ) : ℕ × ℕ :=
  let (a, b) := p
  if a < b then (2*a, b-a) else (a-b, 2*b)

def iter_f (k : ℕ) : (ℕ × ℕ) → (ℕ × ℕ) :=
  match k with
  | 0 => id
  | k+1 => f ∘ (iter_f k)

theorem existence_of_m_and_k (n : ℕ) (h : n > 1) :
  ∃ (m k : ℕ), m < n ∧ iter_f k (n, m) = (m, n) := by
  sorry

#check existence_of_m_and_k

end NUMINAMATH_CALUDE_existence_of_m_and_k_l981_98159


namespace NUMINAMATH_CALUDE_product_remainder_l981_98103

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

theorem product_remainder (a₁ : ℕ) (d : ℕ) (n : ℕ) (m : ℕ) :
  a₁ = 2 → d = 10 → n = 21 → m = 7 →
  (arithmetic_sequence a₁ d n).prod % m = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l981_98103


namespace NUMINAMATH_CALUDE_triangle_exists_l981_98173

/-- A triangle with given circumradius, centroid-circumcenter distance, and centroid-altitude distance --/
structure TriangleWithCircumcenter where
  r : ℝ  -- radius of circumscribed circle
  KS : ℝ  -- distance from circumcenter to centroid
  d : ℝ  -- distance from centroid to altitude

/-- Conditions for the existence of a triangle with given parameters --/
def triangle_existence_conditions (t : TriangleWithCircumcenter) : Prop :=
  t.d ≤ 2 * t.KS ∧ 
  t.r ≥ 3 * t.d / 2 ∧ 
  |Real.sqrt (4 * t.r^2 - 9 * t.d^2) - 3 * Real.sqrt (4 * t.KS^2 - t.d^2)| < 4 * t.r

/-- Theorem stating the existence of a triangle with given parameters --/
theorem triangle_exists (t : TriangleWithCircumcenter) : 
  triangle_existence_conditions t ↔ ∃ (triangle : Type), true :=
sorry

end NUMINAMATH_CALUDE_triangle_exists_l981_98173


namespace NUMINAMATH_CALUDE_min_distance_vectors_l981_98199

def a (t : ℝ) : Fin 3 → ℝ := ![2, t, t]
def b (t : ℝ) : Fin 3 → ℝ := ![1 - t, 2 * t - 1, 0]

theorem min_distance_vectors (t : ℝ) :
  ∃ (min : ℝ), min = Real.sqrt 2 ∧ ∀ (s : ℝ), ‖b s - a s‖ ≥ min := by sorry

end NUMINAMATH_CALUDE_min_distance_vectors_l981_98199


namespace NUMINAMATH_CALUDE_complex_expression_odd_exponent_l981_98182

theorem complex_expression_odd_exponent (n : ℕ) (h : Odd n) :
  (((1 + Complex.I) / (1 - Complex.I)) ^ (2 * n) + 
   ((1 - Complex.I) / (1 + Complex.I)) ^ (2 * n)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_odd_exponent_l981_98182


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l981_98151

theorem real_part_of_complex_fraction : 
  (5 * Complex.I / (1 + 2 * Complex.I)).re = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l981_98151


namespace NUMINAMATH_CALUDE_circles_intersect_l981_98136

theorem circles_intersect : 
  let c1 : ℝ × ℝ := (-2, 0)
  let r1 : ℝ := 2
  let c2 : ℝ × ℝ := (2, 1)
  let r2 : ℝ := 3
  let d := Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2)
  (abs (r1 - r2) < d) ∧ (d < r1 + r2) :=
by sorry

end NUMINAMATH_CALUDE_circles_intersect_l981_98136


namespace NUMINAMATH_CALUDE_molecular_weight_CH3COOH_is_60_l981_98138

/-- The molecular weight of CH3COOH in grams per mole -/
def molecular_weight_CH3COOH : ℝ := 60

/-- The number of moles in the given sample -/
def sample_moles : ℝ := 6

/-- The total weight of the sample in grams -/
def sample_weight : ℝ := 360

/-- Theorem stating that the molecular weight of CH3COOH is 60 grams/mole -/
theorem molecular_weight_CH3COOH_is_60 :
  molecular_weight_CH3COOH = sample_weight / sample_moles :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_CH3COOH_is_60_l981_98138


namespace NUMINAMATH_CALUDE_pen_cost_is_four_l981_98135

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := 2

/-- The cost of a pen in dollars -/
def pen_cost : ℝ := 2 * pencil_cost

/-- The total cost of a pen and pencil in dollars -/
def total_cost : ℝ := 6

theorem pen_cost_is_four :
  pen_cost = 4 ∧ pencil_cost + pen_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_pen_cost_is_four_l981_98135


namespace NUMINAMATH_CALUDE_b_share_is_180_l981_98183

/-- Represents the rental arrangement for a pasture -/
structure PastureRental where
  total_rent : ℚ
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  c_months : ℕ

/-- Calculates the share of rent for person b given a PastureRental arrangement -/
def calculate_b_share (rental : PastureRental) : ℚ :=
  let total_horse_months := rental.a_horses * rental.a_months +
                            rental.b_horses * rental.b_months +
                            rental.c_horses * rental.c_months
  let cost_per_horse_month := rental.total_rent / total_horse_months
  (rental.b_horses * rental.b_months : ℚ) * cost_per_horse_month

/-- Theorem stating that b's share of the rent is 180 for the given arrangement -/
theorem b_share_is_180 (rental : PastureRental)
  (h1 : rental.total_rent = 435)
  (h2 : rental.a_horses = 12) (h3 : rental.a_months = 8)
  (h4 : rental.b_horses = 16) (h5 : rental.b_months = 9)
  (h6 : rental.c_horses = 18) (h7 : rental.c_months = 6) :
  calculate_b_share rental = 180 := by
  sorry

end NUMINAMATH_CALUDE_b_share_is_180_l981_98183


namespace NUMINAMATH_CALUDE_six_digit_number_exists_l981_98122

/-- A six-digit number is between 100000 and 999999 -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- A five-digit number is between 10000 and 99999 -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- The result of removing one digit from a six-digit number -/
def remove_digit (n : ℕ) : ℕ := n / 10

theorem six_digit_number_exists : 
  ∃! n : ℕ, is_six_digit n ∧ 
    ∃ m : ℕ, is_five_digit m ∧ 
      m = remove_digit n ∧ 
      n - m = 654321 :=
sorry

end NUMINAMATH_CALUDE_six_digit_number_exists_l981_98122


namespace NUMINAMATH_CALUDE_solution_set_x_abs_x_minus_one_l981_98115

theorem solution_set_x_abs_x_minus_one (x : ℝ) :
  {x : ℝ | x * |x - 1| > 0} = {x : ℝ | 0 < x ∧ x ≠ 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_x_abs_x_minus_one_l981_98115


namespace NUMINAMATH_CALUDE_sum_of_partial_fractions_coefficients_l981_98171

theorem sum_of_partial_fractions_coefficients (A B C D E : ℝ) :
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 ∧ x ≠ -6 →
    (x + 1) / ((x + 2) * (x + 3) * (x + 4) * (x + 5) * (x + 6)) =
    A / (x + 2) + B / (x + 3) + C / (x + 4) + D / (x + 5) + E / (x + 6)) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_partial_fractions_coefficients_l981_98171


namespace NUMINAMATH_CALUDE_common_tangent_sum_l981_98178

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = 2 * x^2 + 125 / 100

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = 2 * y^2 + 65 / 4

/-- Common tangent line L -/
def L (x y a b c : ℝ) : Prop := a * x + b * y = c

/-- The slope of L is rational -/
def rational_slope (a b : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ a / b = p / q

theorem common_tangent_sum (a b c : ℕ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧
    L x₁ y₁ a b c ∧ L x₂ y₂ a b c ∧
    rational_slope a b ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.gcd a (Nat.gcd b c) = 1) →
  a + b + c = 289 := by
  sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l981_98178


namespace NUMINAMATH_CALUDE_stellas_clocks_l981_98177

/-- Stella's antique shop inventory problem -/
theorem stellas_clocks :
  ∀ (num_clocks : ℕ),
    (3 * 5 + num_clocks * 15 + 5 * 4 = 40 + 25) →
    num_clocks = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_stellas_clocks_l981_98177


namespace NUMINAMATH_CALUDE_photo_lineup_arrangements_l981_98148

def number_of_male_actors : ℕ := 4
def number_of_female_actors : ℕ := 5

def arrangement_count (m n : ℕ) : ℕ := sorry

theorem photo_lineup_arrangements :
  arrangement_count number_of_male_actors number_of_female_actors =
    (arrangement_count number_of_female_actors number_of_female_actors) *
    (arrangement_count (number_of_female_actors + 1) number_of_male_actors) -
    2 * (arrangement_count (number_of_female_actors - 1) (number_of_female_actors - 1)) *
    (arrangement_count number_of_female_actors number_of_male_actors) :=
  sorry

end NUMINAMATH_CALUDE_photo_lineup_arrangements_l981_98148


namespace NUMINAMATH_CALUDE_cylinder_side_diagonal_l981_98134

theorem cylinder_side_diagonal (h l d : ℝ) (h_height : h = 16) (h_length : l = 12) : 
  d = 20 → d^2 = h^2 + l^2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_side_diagonal_l981_98134


namespace NUMINAMATH_CALUDE_raffle_ticket_sales_l981_98114

theorem raffle_ticket_sales (total_members : ℕ) (male_members : ℕ) (female_members : ℕ) 
  (total_tickets : ℕ) (female_tickets : ℕ) :
  total_members > 0 →
  male_members > 0 →
  female_members = 2 * male_members →
  total_members = male_members + female_members →
  (total_tickets : ℚ) / total_members = 66 →
  (female_tickets : ℚ) / female_members = 70 →
  (total_tickets - female_tickets : ℚ) / male_members = 66 :=
by sorry

end NUMINAMATH_CALUDE_raffle_ticket_sales_l981_98114


namespace NUMINAMATH_CALUDE_permutation_solution_l981_98176

def is_valid_permutation (a : Fin 9 → ℕ) : Prop :=
  (∀ i j : Fin 9, i ≠ j → a i ≠ a j) ∧
  (∀ i : Fin 9, a i ∈ (Set.range (fun i : Fin 9 => i.val + 1)))

def satisfies_conditions (a : Fin 9 → ℕ) : Prop :=
  (a 0 + a 1 + a 2 + a 3 = a 3 + a 4 + a 5 + a 6) ∧
  (a 3 + a 4 + a 5 + a 6 = a 6 + a 7 + a 8 + a 0) ∧
  (a 0^2 + a 1^2 + a 2^2 + a 3^2 = a 3^2 + a 4^2 + a 5^2 + a 6^2) ∧
  (a 3^2 + a 4^2 + a 5^2 + a 6^2 = a 6^2 + a 7^2 + a 8^2 + a 0^2)

def solution : Fin 9 → ℕ := fun i =>
  match i with
  | ⟨0, _⟩ => 2
  | ⟨1, _⟩ => 4
  | ⟨2, _⟩ => 9
  | ⟨3, _⟩ => 5
  | ⟨4, _⟩ => 1
  | ⟨5, _⟩ => 6
  | ⟨6, _⟩ => 8
  | ⟨7, _⟩ => 3
  | ⟨8, _⟩ => 7

theorem permutation_solution :
  is_valid_permutation solution ∧ satisfies_conditions solution :=
by sorry

end NUMINAMATH_CALUDE_permutation_solution_l981_98176


namespace NUMINAMATH_CALUDE_volume_ratio_specific_cone_l981_98162

/-- Represents a right circular cone -/
structure Cone where
  base_diameter : ℝ
  height : ℝ

/-- Represents a plane intersecting the cone -/
structure IntersectingPlane where
  distance_from_apex : ℝ

/-- Calculates the volume ratio of the two parts resulting from intersecting a cone with a plane -/
def volume_ratio (cone : Cone) (plane : IntersectingPlane) : ℝ × ℝ :=
  sorry

/-- Theorem stating the volume ratio for the given cone and intersecting plane -/
theorem volume_ratio_specific_cone :
  let cone : Cone := { base_diameter := 26, height := 39 }
  let plane : IntersectingPlane := { distance_from_apex := 30 }
  volume_ratio cone plane = (0.4941, 0.5059) :=
sorry

end NUMINAMATH_CALUDE_volume_ratio_specific_cone_l981_98162


namespace NUMINAMATH_CALUDE_cafeteria_pies_l981_98152

theorem cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : 
  initial_apples = 62 → 
  handed_out = 8 → 
  apples_per_pie = 9 → 
  (initial_apples - handed_out) / apples_per_pie = 6 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l981_98152


namespace NUMINAMATH_CALUDE_printer_problem_l981_98112

/-- Given a total of 42 pages, where every 7th page is crumpled and every 3rd page is blurred,
    the number of pages that are neither crumpled nor blurred is 24. -/
theorem printer_problem (total_pages : Nat) (crumple_interval : Nat) (blur_interval : Nat)
    (h1 : total_pages = 42)
    (h2 : crumple_interval = 7)
    (h3 : blur_interval = 3) :
    total_pages - (total_pages / crumple_interval + total_pages / blur_interval - total_pages / (crumple_interval * blur_interval)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_printer_problem_l981_98112


namespace NUMINAMATH_CALUDE_book_page_words_l981_98156

theorem book_page_words (total_pages : ℕ) (max_words_per_page : ℕ) (total_words_mod : ℕ) :
  total_pages = 150 →
  max_words_per_page = 120 →
  total_words_mod = 221 →
  ∃ (words_per_page : ℕ),
    words_per_page ≤ max_words_per_page ∧
    Nat.Prime words_per_page ∧
    (total_pages * words_per_page) % total_words_mod = 220 ∧
    words_per_page = 67 :=
by sorry

end NUMINAMATH_CALUDE_book_page_words_l981_98156


namespace NUMINAMATH_CALUDE_segment_construction_l981_98113

/-- Given positive real numbers a, b, c, d, and e, there exists a real number x
    such that x = (a * b * c) / (d * e). -/
theorem segment_construction (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (hd : d > 0) (he : e > 0) : ∃ x : ℝ, x = (a * b * c) / (d * e) := by
  sorry

end NUMINAMATH_CALUDE_segment_construction_l981_98113


namespace NUMINAMATH_CALUDE_simplify_fraction_l981_98126

theorem simplify_fraction : 4 * (14 / 5) * (20 / -42) = -(4 / 15) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l981_98126


namespace NUMINAMATH_CALUDE_petes_number_l981_98193

theorem petes_number : ∃ x : ℝ, 5 * (3 * x - 5) = 200 ∧ x = 15 := by sorry

end NUMINAMATH_CALUDE_petes_number_l981_98193


namespace NUMINAMATH_CALUDE_hall_length_breadth_difference_l981_98167

/-- Represents a rectangular hall -/
structure RectangularHall where
  length : ℝ
  breadth : ℝ
  area : ℝ

/-- Theorem: For a rectangular hall with area 750 m² and length 30 m, 
    the difference between length and breadth is 5 m -/
theorem hall_length_breadth_difference 
  (hall : RectangularHall) 
  (h1 : hall.area = 750) 
  (h2 : hall.length = 30) : 
  hall.length - hall.breadth = 5 := by
  sorry


end NUMINAMATH_CALUDE_hall_length_breadth_difference_l981_98167


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l981_98132

-- Problem 1
theorem problem_1 (m : ℝ) : 
  let A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
  let B : Set ℝ := {x | x^2 + (m+1)*x + m = 0}
  A ∩ B = B → m = 1 ∨ m = 2 := by sorry

-- Problem 2
theorem problem_2 (n : ℝ) :
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
  let B : Set ℝ := {x | n+1 ≤ x ∧ x ≤ 2*n-1}
  B ⊆ A → n ≤ 3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l981_98132


namespace NUMINAMATH_CALUDE_ball_transfer_probability_l981_98172

/-- Represents a bag of balls -/
structure Bag where
  white : ℕ
  red : ℕ

/-- The probability of drawing a red ball from a bag -/
def redProbability (bag : Bag) : ℚ :=
  bag.red / (bag.white + bag.red)

/-- The probability of drawing a white ball from a bag -/
def whiteProbability (bag : Bag) : ℚ :=
  bag.white / (bag.white + bag.red)

/-- The probability of drawing a red ball from the second bag
    after transferring a ball from the first bag -/
def transferAndDrawRed (bagA bagB : Bag) : ℚ :=
  (redProbability bagA) * (redProbability (Bag.mk bagB.white (bagB.red + 1))) +
  (whiteProbability bagA) * (redProbability (Bag.mk (bagB.white + 1) bagB.red))

theorem ball_transfer_probability :
  let bagA : Bag := ⟨2, 3⟩
  let bagB : Bag := ⟨1, 2⟩
  transferAndDrawRed bagA bagB = 13 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ball_transfer_probability_l981_98172


namespace NUMINAMATH_CALUDE_rectangle_length_l981_98180

/-- 
Given a rectangular garden with perimeter 950 meters and breadth 100 meters, 
this theorem proves that its length is 375 meters.
-/
theorem rectangle_length (perimeter breadth : ℝ) 
  (h_perimeter : perimeter = 950)
  (h_breadth : breadth = 100) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter :=
by
  sorry

#check rectangle_length

end NUMINAMATH_CALUDE_rectangle_length_l981_98180


namespace NUMINAMATH_CALUDE_octopus_leg_solution_l981_98189

-- Define the possible number of legs for an octopus
inductive LegCount : Type
  | six : LegCount
  | seven : LegCount
  | eight : LegCount

-- Define the colors of the octopuses
inductive OctopusColor : Type
  | blue : OctopusColor
  | green : OctopusColor
  | yellow : OctopusColor
  | red : OctopusColor

-- Define a function to determine if an octopus is truthful based on its leg count
def isTruthful (legs : LegCount) : Prop :=
  match legs with
  | LegCount.six => True
  | LegCount.seven => False
  | LegCount.eight => True

-- Define a function to convert LegCount to a natural number
def legCountToNat (legs : LegCount) : ℕ :=
  match legs with
  | LegCount.six => 6
  | LegCount.seven => 7
  | LegCount.eight => 8

-- Define the claims made by each octopus
def claim (color : OctopusColor) : ℕ :=
  match color with
  | OctopusColor.blue => 28
  | OctopusColor.green => 27
  | OctopusColor.yellow => 26
  | OctopusColor.red => 25

-- Define the theorem
theorem octopus_leg_solution :
  ∃ (legs : OctopusColor → LegCount),
    (legs OctopusColor.green = LegCount.six) ∧
    (legs OctopusColor.blue = LegCount.seven) ∧
    (legs OctopusColor.yellow = LegCount.seven) ∧
    (legs OctopusColor.red = LegCount.seven) ∧
    (∀ (c : OctopusColor), isTruthful (legs c) ↔ (claim c = legCountToNat (legs OctopusColor.blue) + legCountToNat (legs OctopusColor.green) + legCountToNat (legs OctopusColor.yellow) + legCountToNat (legs OctopusColor.red))) :=
  sorry

end NUMINAMATH_CALUDE_octopus_leg_solution_l981_98189


namespace NUMINAMATH_CALUDE_heathers_weight_l981_98131

/-- Given that Emily weighs 9 pounds and Heather is 78 pounds heavier than Emily,
    prove that Heather weighs 87 pounds. -/
theorem heathers_weight (emily_weight : ℕ) (weight_difference : ℕ) :
  emily_weight = 9 →
  weight_difference = 78 →
  emily_weight + weight_difference = 87 :=
by sorry

end NUMINAMATH_CALUDE_heathers_weight_l981_98131


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l981_98130

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 16 = 48 → Nat.gcd n 16 = 18 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l981_98130


namespace NUMINAMATH_CALUDE_pen_notebook_cost_l981_98143

theorem pen_notebook_cost : ∃ (p n : ℕ), 
  p > 0 ∧ n > 0 ∧ 
  15 * p + 5 * n = 13000 ∧ 
  p > n ∧ 
  p + n = 10 :=
by sorry

end NUMINAMATH_CALUDE_pen_notebook_cost_l981_98143


namespace NUMINAMATH_CALUDE_parallel_line_equation_l981_98164

/-- A line in the Cartesian coordinate system -/
structure CartesianLine where
  slope : ℝ
  y_intercept : ℝ

/-- The equation of a line given its slope and y-intercept -/
def line_equation (l : CartesianLine) (x : ℝ) : ℝ :=
  l.slope * x + l.y_intercept

theorem parallel_line_equation 
  (l : CartesianLine) 
  (h1 : l.slope = -2) 
  (h2 : l.y_intercept = -3) : 
  ∀ x, line_equation l x = -2 * x - 3 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l981_98164


namespace NUMINAMATH_CALUDE_inequality_proof_l981_98107

theorem inequality_proof (a b c A B C k : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_A : 0 < A) (pos_B : 0 < B) (pos_C : 0 < C)
  (sum_a : a + A = k) (sum_b : b + B = k) (sum_c : c + C = k) :
  a * B + b * C + c * A ≤ k^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l981_98107


namespace NUMINAMATH_CALUDE_heat_required_for_temperature_change_l981_98117

/-- Specific heat capacity as a function of temperature -/
def specific_heat_capacity (c₀ α t : ℝ) : ℝ := c₀ * (1 + α * t)

/-- Amount of heat required to change temperature -/
def heat_required (m c_avg Δt : ℝ) : ℝ := m * c_avg * Δt

theorem heat_required_for_temperature_change 
  (m : ℝ) 
  (c₀ : ℝ) 
  (α : ℝ) 
  (t_initial t_final : ℝ) 
  (h_m : m = 3) 
  (h_c₀ : c₀ = 200) 
  (h_α : α = 0.05) 
  (h_t_initial : t_initial = 30) 
  (h_t_final : t_final = 80) :
  heat_required m 
    ((specific_heat_capacity c₀ α t_initial + specific_heat_capacity c₀ α t_final) / 2) 
    (t_final - t_initial) = 112500 := by
  sorry

#check heat_required_for_temperature_change

end NUMINAMATH_CALUDE_heat_required_for_temperature_change_l981_98117


namespace NUMINAMATH_CALUDE_unique_solutions_l981_98190

def system_solution (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 1 ∧
  x^4 - 18*x^2*y^2 + 81*y^4 - 20*x^2 - 180*y^2 + 100 = 0

theorem unique_solutions :
  (∀ x y : ℝ, system_solution x y →
    ((x = -1/Real.sqrt 10 ∧ y = 3/Real.sqrt 10) ∨
     (x = -1/Real.sqrt 10 ∧ y = -3/Real.sqrt 10) ∨
     (x = 1/Real.sqrt 10 ∧ y = 3/Real.sqrt 10) ∨
     (x = 1/Real.sqrt 10 ∧ y = -3/Real.sqrt 10))) ∧
  (system_solution (-1/Real.sqrt 10) (3/Real.sqrt 10)) ∧
  (system_solution (-1/Real.sqrt 10) (-3/Real.sqrt 10)) ∧
  (system_solution (1/Real.sqrt 10) (3/Real.sqrt 10)) ∧
  (system_solution (1/Real.sqrt 10) (-3/Real.sqrt 10)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l981_98190


namespace NUMINAMATH_CALUDE_sum_of_coefficients_bounds_l981_98170

/-- A quadratic function with vertex in the first quadrant passing through (0,1) and (-1,0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  vertex_first_quadrant : -b / (2 * a) > 0
  passes_through_0_1 : c = 1
  passes_through_neg1_0 : a - b + c = 0

/-- The sum of coefficients of a quadratic function -/
def S (f : QuadraticFunction) : ℝ := f.a + f.b + f.c

/-- Theorem: The sum of coefficients S is between 0 and 2 -/
theorem sum_of_coefficients_bounds (f : QuadraticFunction) : 0 < S f ∧ S f < 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_bounds_l981_98170


namespace NUMINAMATH_CALUDE_ratio_s5_s8_l981_98111

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_ratio : (4 * (2 * a 1 + 3 * (a 2 - a 1))) / (6 * (2 * a 1 + 5 * (a 2 - a 1))) = -2/3

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

/-- The main theorem -/
theorem ratio_s5_s8 (seq : ArithmeticSequence) : 
  (sum_n seq 5) / (sum_n seq 8) = 1 / 40.8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_s5_s8_l981_98111


namespace NUMINAMATH_CALUDE_train_B_speed_train_B_speed_is_36_l981_98160

-- Define the problem parameters
def train_A_length : ℝ := 125  -- meters
def train_B_length : ℝ := 150  -- meters
def train_A_speed : ℝ := 54    -- km/hr
def crossing_time : ℝ := 11    -- seconds

-- Define the theorem
theorem train_B_speed : ℝ :=
  let total_distance := train_A_length + train_B_length
  let relative_speed_mps := total_distance / crossing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  relative_speed_kmph - train_A_speed

-- Prove the theorem
theorem train_B_speed_is_36 : train_B_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_B_speed_train_B_speed_is_36_l981_98160


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2023rd_term_l981_98174

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2023rd_term 
  (p q : ℝ) 
  (h1 : arithmetic_sequence p (3*p - q - p) 1 = p)
  (h2 : arithmetic_sequence p (3*p - q - p) 2 = 3*p - q)
  (h3 : arithmetic_sequence p (3*p - q - p) 3 = 9)
  (h4 : arithmetic_sequence p (3*p - q - p) 4 = 3*p + q) :
  arithmetic_sequence p (3*p - q - p) 2023 = 18189 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2023rd_term_l981_98174


namespace NUMINAMATH_CALUDE_find_a_minus_b_l981_98161

-- Define the functions
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 7
def h (a b x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem find_a_minus_b (a b : ℝ) :
  (∀ x, h a b x = x - 9) → a - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_a_minus_b_l981_98161


namespace NUMINAMATH_CALUDE_original_price_from_discounted_l981_98175

/-- 
Given a shirt sold at a discounted price with a known discount percentage, 
this theorem proves the original selling price.
-/
theorem original_price_from_discounted (discounted_price : ℝ) (discount_percent : ℝ) 
  (h1 : discounted_price = 560) 
  (h2 : discount_percent = 20) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_percent / 100) = discounted_price ∧ 
    original_price = 700 := by
  sorry

end NUMINAMATH_CALUDE_original_price_from_discounted_l981_98175


namespace NUMINAMATH_CALUDE_container_capacity_prove_container_capacity_l981_98144

theorem container_capacity : ℝ → Prop :=
  fun capacity =>
    (0.5 * capacity + 20 = 0.75 * capacity) →
    capacity = 80

-- The proof of the theorem
theorem prove_container_capacity : container_capacity 80 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_prove_container_capacity_l981_98144


namespace NUMINAMATH_CALUDE_x_plus_inv_x_eq_five_l981_98169

theorem x_plus_inv_x_eq_five (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_inv_x_eq_five_l981_98169


namespace NUMINAMATH_CALUDE_igor_travel_time_l981_98133

/-- Represents the ski lift system with its properties and functions -/
structure SkiLift where
  total_cabins : Nat
  igor_cabin : Nat
  first_alignment : Nat
  second_alignment : Nat
  alignment_time : Nat

/-- Calculates the time for Igor to reach the top of the mountain -/
def time_to_top (lift : SkiLift) : Nat :=
  let total_distance := lift.total_cabins - lift.igor_cabin + lift.second_alignment
  let speed := (lift.first_alignment - lift.second_alignment) / lift.alignment_time
  (total_distance / 2) * (1 / speed)

/-- Theorem stating that Igor will reach the top in 1035 seconds -/
theorem igor_travel_time (lift : SkiLift) 
  (h1 : lift.total_cabins = 99)
  (h2 : lift.igor_cabin = 42)
  (h3 : lift.first_alignment = 13)
  (h4 : lift.second_alignment = 12)
  (h5 : lift.alignment_time = 15) :
  time_to_top lift = 1035 := by
  sorry

end NUMINAMATH_CALUDE_igor_travel_time_l981_98133


namespace NUMINAMATH_CALUDE_calculator_exam_duration_l981_98124

theorem calculator_exam_duration 
  (full_battery : ℝ) 
  (remaining_battery : ℝ) 
  (exam_duration : ℝ) :
  full_battery = 60 →
  remaining_battery = 13 →
  exam_duration = (1/4 * full_battery) - remaining_battery →
  exam_duration = 2 :=
by sorry

end NUMINAMATH_CALUDE_calculator_exam_duration_l981_98124


namespace NUMINAMATH_CALUDE_new_person_weight_l981_98149

theorem new_person_weight (W : ℝ) :
  let initial_avg := W / 20
  let intermediate_avg := (W - 95) / 19
  let final_avg := initial_avg + 4.2
  let new_person_weight := (final_avg * 20) - (W - 95)
  new_person_weight = 179 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l981_98149


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_20000_l981_98179

def hulk_jump (n : ℕ) : ℝ := 3 * (3 ^ (n - 1))

theorem hulk_jump_exceeds_20000 :
  (∀ m : ℕ, m < 10 → hulk_jump m ≤ 20000) ∧
  hulk_jump 10 > 20000 := by
sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_20000_l981_98179


namespace NUMINAMATH_CALUDE_jane_payment_l981_98100

/-- The amount Jane paid with, given the cost of the apple and the change received. -/
def amount_paid (apple_cost change : ℚ) : ℚ :=
  apple_cost + change

/-- Theorem stating that Jane paid with $5.00, given the conditions of the problem. -/
theorem jane_payment :
  let apple_cost : ℚ := 75 / 100
  let change : ℚ := 425 / 100
  amount_paid apple_cost change = 5 := by
  sorry

end NUMINAMATH_CALUDE_jane_payment_l981_98100


namespace NUMINAMATH_CALUDE_number_of_orders_is_1536_l981_98129

/-- Represents the number of letters --/
def n : ℕ := 10

/-- Represents the number of letters that can be in the stack (excluding 9 and 10) --/
def m : ℕ := 8

/-- Calculates the number of different orders for typing the remaining letters --/
def number_of_orders : ℕ :=
  Finset.sum (Finset.range (m + 1)) (λ k => (Nat.choose m k) * (k + 2))

/-- Theorem stating that the number of different orders is 1536 --/
theorem number_of_orders_is_1536 : number_of_orders = 1536 := by
  sorry

end NUMINAMATH_CALUDE_number_of_orders_is_1536_l981_98129


namespace NUMINAMATH_CALUDE_min_value_problem_max_value_problem_min_sum_problem_l981_98153

-- Problem 1
theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y ≥ 8 := by sorry

-- Problem 2
theorem max_value_problem (x : ℝ) (h : x < 3) :
  4/(x - 3) + x ≤ -1 := by sorry

-- Problem 3
theorem min_sum_problem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = 2) :
  n/m + 1/(2*n) ≥ 5/4 := by sorry

end NUMINAMATH_CALUDE_min_value_problem_max_value_problem_min_sum_problem_l981_98153


namespace NUMINAMATH_CALUDE_inscribed_half_area_rhombus_l981_98181

/-- A centrally symmetric convex polygon -/
structure CentrallySymmetricConvexPolygon where
  -- Add necessary fields and properties
  area : ℝ
  centrally_symmetric : Bool
  convex : Bool

/-- A rhombus -/
structure Rhombus where
  -- Add necessary fields
  area : ℝ

/-- A rhombus is inscribed in a polygon -/
def is_inscribed (r : Rhombus) (p : CentrallySymmetricConvexPolygon) : Prop :=
  sorry

/-- Main theorem: For any centrally symmetric convex polygon, 
    there exists an inscribed rhombus with half the area of the polygon -/
theorem inscribed_half_area_rhombus (p : CentrallySymmetricConvexPolygon) :
  ∃ r : Rhombus, is_inscribed r p ∧ r.area = p.area / 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_half_area_rhombus_l981_98181


namespace NUMINAMATH_CALUDE_product_of_square_roots_l981_98163

theorem product_of_square_roots (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p) * Real.sqrt (10 * p^3) * Real.sqrt (14 * p^5) = 10 * p^4 * Real.sqrt (21 * p) :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l981_98163


namespace NUMINAMATH_CALUDE_triangle_max_area_l981_98118

/-- Given a triangle ABC with area S, prove that the maximum value of S is √3/4
    when 2S + √3(AB · AC) = 0 and |BC| = √3 -/
theorem triangle_max_area (A B C : ℝ × ℝ) (S : ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  2 * S + Real.sqrt 3 * (AB.1 * AC.1 + AB.2 * AC.2) = 0 →
  BC.1^2 + BC.2^2 = 3 →
  S ≤ Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l981_98118


namespace NUMINAMATH_CALUDE_function_properties_l981_98195

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a^(2*x) - 2*a^x + 1

theorem function_properties (a : ℝ) (h_a : a > 1) :
  (∀ y : ℝ, y < 1 → ∃ x : ℝ, f a x = y) ∧
  (∀ x : ℝ, f a x < 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f a x ≥ -7) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ f a x = -7) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l981_98195


namespace NUMINAMATH_CALUDE_inequality_proof_l981_98150

theorem inequality_proof (a b c : ℝ) (h : a * b < 0) :
  a^2 + b^2 + c^2 > 2*a*b + 2*b*c + 2*c*a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l981_98150


namespace NUMINAMATH_CALUDE_power_sum_ratio_l981_98106

theorem power_sum_ratio (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (sum_zero : a + b + c = 0) :
  (a^7 + b^7 + c^7)^2 / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49/60 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_ratio_l981_98106


namespace NUMINAMATH_CALUDE_cross_section_area_l981_98105

-- Define the rectangular parallelepiped
structure RectangularParallelepiped where
  AB : ℝ
  AD : ℝ
  BD : ℝ
  AA₁ : ℝ

-- Define the theorem
theorem cross_section_area (rp : RectangularParallelepiped)
  (h1 : rp.AB = 29)
  (h2 : rp.AD = 36)
  (h3 : rp.BD = 25)
  (h4 : rp.AA₁ = 48) :
  ∃ (area : ℝ), area = 1872 ∧ area = rp.AD * Real.sqrt (rp.AA₁^2 + (Real.sqrt (rp.AD^2 + rp.AB^2 - rp.BD^2))^2) :=
by sorry

end NUMINAMATH_CALUDE_cross_section_area_l981_98105


namespace NUMINAMATH_CALUDE_flour_for_one_loaf_l981_98101

/-- The amount of flour required for one loaf of bread -/
def flour_per_loaf (total_flour : ℕ) (num_loaves : ℕ) : ℕ := total_flour / num_loaves

/-- Theorem: Given 400g of total flour and the ability to make 2 loaves, 
    prove that one loaf requires 200g of flour -/
theorem flour_for_one_loaf : 
  flour_per_loaf 400 2 = 200 := by
sorry

end NUMINAMATH_CALUDE_flour_for_one_loaf_l981_98101


namespace NUMINAMATH_CALUDE_total_marbles_is_90_l981_98157

/-- Represents the number of marbles of each color in the bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The ratio of red:blue:green marbles is 2:4:6 -/
def ratio_constraint (bag : MarbleBag) : Prop :=
  3 * bag.red = bag.blue ∧ 2 * bag.blue = bag.green

/-- There are 30 blue marbles -/
def blue_constraint (bag : MarbleBag) : Prop :=
  bag.blue = 30

/-- The total number of marbles in the bag -/
def total_marbles (bag : MarbleBag) : ℕ :=
  bag.red + bag.blue + bag.green

/-- Theorem stating that the total number of marbles is 90 -/
theorem total_marbles_is_90 (bag : MarbleBag) 
  (h_ratio : ratio_constraint bag) (h_blue : blue_constraint bag) : 
  total_marbles bag = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_90_l981_98157


namespace NUMINAMATH_CALUDE_inequality_theorem_l981_98137

theorem inequality_theorem (a b c : ℝ) (θ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * Real.cos θ ^ 2 + b * Real.sin θ ^ 2 < c) : 
  Real.sqrt a * Real.cos θ ^ 2 + Real.sqrt b * Real.sin θ ^ 2 < Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l981_98137


namespace NUMINAMATH_CALUDE_motel_rent_problem_l981_98196

/-- Represents the total rent charged by a motel on a given night -/
def TotalRent (r40 r60 : ℕ) : ℝ := 40 * r40 + 60 * r60

/-- The problem statement -/
theorem motel_rent_problem (r40 r60 : ℕ) :
  (∃ (total : ℝ), total = TotalRent r40 r60 ∧
    0.8 * total = TotalRent (r40 + 10) (r60 - 10)) →
  TotalRent r40 r60 = 1000 := by
  sorry

#check motel_rent_problem

end NUMINAMATH_CALUDE_motel_rent_problem_l981_98196


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_l981_98158

theorem sin_cos_equation_solution (x y : ℝ) : 
  (Real.sin (x + y))^2 - (Real.cos (x - y))^2 = 1 ↔ 
  (∃ (k l : ℤ), x = Real.pi / 2 * (2 * k + l + 1) ∧ y = Real.pi / 2 * (2 * k - l)) ∨
  (∃ (m n : ℤ), x = Real.pi / 2 * (2 * m + n) ∧ y = Real.pi / 2 * (2 * m - n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_l981_98158


namespace NUMINAMATH_CALUDE_initial_passengers_l981_98116

theorem initial_passengers (P : ℕ) : 
  P % 2 = 0 ∧ 
  (P : ℝ) + 0.08 * (P : ℝ) ≤ 70 ∧ 
  P % 25 = 0 → 
  P = 50 := by
sorry

end NUMINAMATH_CALUDE_initial_passengers_l981_98116


namespace NUMINAMATH_CALUDE_compound_interest_problem_l981_98119

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Total amount calculation --/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

/-- Main theorem --/
theorem compound_interest_problem (principal : ℝ) :
  compound_interest principal 0.1 2 = 420 →
  total_amount principal 420 = 2420 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l981_98119


namespace NUMINAMATH_CALUDE_tan_sum_alpha_beta_l981_98184

theorem tan_sum_alpha_beta (α β : Real) 
  (h1 : Real.sin α + Real.sin β = 1/4)
  (h2 : Real.cos α + Real.cos β = 1/3) : 
  Real.tan (α + β) = 24/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_alpha_beta_l981_98184


namespace NUMINAMATH_CALUDE_factorial_combination_l981_98140

theorem factorial_combination : Nat.factorial 10 / (Nat.factorial 7 * Nat.factorial 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_combination_l981_98140


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l981_98120

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the arithmetic sequence equals 120. -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

/-- The main theorem: If a is an arithmetic sequence satisfying the sum condition,
    then the difference between a_7 and one-third of a_5 is 16. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) (h_sum : SumCondition a) : 
    a 7 - (1/3) * a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l981_98120


namespace NUMINAMATH_CALUDE_library_visitors_l981_98146

/-- Calculates the average number of visitors on non-Sunday days in a library -/
theorem library_visitors (sunday_avg : ℕ) (total_days : ℕ) (month_avg : ℕ) 
  (h1 : sunday_avg = 510)
  (h2 : total_days = 30)
  (h3 : month_avg = 285) :
  let sundays : ℕ := total_days / 7 + 1
  let other_days : ℕ := total_days - sundays
  let other_days_avg : ℕ := (month_avg * total_days - sunday_avg * sundays) / other_days
  other_days_avg = 240 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_l981_98146


namespace NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_for_x_sq_eq_9_l981_98102

theorem x_eq_3_sufficient_not_necessary_for_x_sq_eq_9 :
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) ∧
  (∀ x : ℝ, x = 3 → x^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_3_sufficient_not_necessary_for_x_sq_eq_9_l981_98102


namespace NUMINAMATH_CALUDE_ellipse_locus_and_intercept_range_l981_98192

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point B
def B : ℝ × ℝ := (0, 1)

-- Define the perpendicularity condition
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  (P.2 - B.2) * (Q.2 - B.2) = -(P.1 - B.1) * (Q.1 - B.1)

-- Define the projection M
def M (P Q : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the perpendicular bisector l
def l (P Q : ℝ × ℝ) : ℝ → ℝ := sorry

-- Define the x-intercept of l
def x_intercept (P Q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_locus_and_intercept_range :
  ∀ (P Q : ℝ × ℝ),
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  P ≠ B →
  Q ≠ B →
  perpendicular P Q →
  (∀ (x y : ℝ), 
    (x, y) = M P Q →
    y ≠ 1 →
    x^2 + (y - 1/5)^2 = (4/5)^2) ∧
  (-9/20 ≤ x_intercept P Q ∧ x_intercept P Q ≤ 9/20) :=
sorry

end NUMINAMATH_CALUDE_ellipse_locus_and_intercept_range_l981_98192


namespace NUMINAMATH_CALUDE_angle_between_lines_l981_98141

theorem angle_between_lines (k₁ k₂ : ℝ) (h₁ : 6 * k₁^2 + k₁ - 1 = 0) (h₂ : 6 * k₂^2 + k₂ - 1 = 0) :
  let θ := Real.arctan ((k₁ - k₂) / (1 + k₁ * k₂))
  θ = π / 4 ∨ θ = -π / 4 :=
sorry

end NUMINAMATH_CALUDE_angle_between_lines_l981_98141
