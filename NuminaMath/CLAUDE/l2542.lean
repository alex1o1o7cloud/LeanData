import Mathlib

namespace NUMINAMATH_CALUDE_petes_diner_cost_theorem_l2542_254269

/-- Represents the cost calculation at Pete's Diner -/
def PetesDinerCost (burgerPrice juicePrice discountAmount : ℕ) 
                   (discountThreshold : ℕ) 
                   (burgerCount juiceCount : ℕ) : ℕ :=
  let totalItems := burgerCount + juiceCount
  let subtotal := burgerCount * burgerPrice + juiceCount * juicePrice
  if totalItems > discountThreshold then subtotal - discountAmount else subtotal

/-- Proves that the total cost of 7 burgers and 5 juices at Pete's Diner is 38 dollars -/
theorem petes_diner_cost_theorem : 
  PetesDinerCost 4 3 5 10 7 5 = 38 := by
  sorry

#eval PetesDinerCost 4 3 5 10 7 5

end NUMINAMATH_CALUDE_petes_diner_cost_theorem_l2542_254269


namespace NUMINAMATH_CALUDE_cricket_average_increase_l2542_254262

theorem cricket_average_increase (initial_average : ℝ) : 
  (16 * initial_average + 92) / 17 = initial_average + 4 → 
  (16 * initial_average + 92) / 17 = 28 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_increase_l2542_254262


namespace NUMINAMATH_CALUDE_opposite_of_negative_half_l2542_254245

theorem opposite_of_negative_half : -(-(1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_half_l2542_254245


namespace NUMINAMATH_CALUDE_right_triangular_pyramid_volume_l2542_254203

/-- A right triangular pyramid with base edge length 2 and pairwise perpendicular side edges -/
structure RightTriangularPyramid where
  base_edge_length : ℝ
  side_edges_perpendicular : Prop

/-- The volume of a right triangular pyramid -/
def volume (p : RightTriangularPyramid) : ℝ := sorry

/-- Theorem: The volume of a right triangular pyramid with base edge length 2 
    and pairwise perpendicular side edges is √2/3 -/
theorem right_triangular_pyramid_volume :
  ∀ (p : RightTriangularPyramid), 
    p.base_edge_length = 2 ∧ p.side_edges_perpendicular →
    volume p = Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_right_triangular_pyramid_volume_l2542_254203


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2542_254232

/-- A geometric sequence with specific properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) ∧
  |a 1| = 1 ∧
  a 5 = -8 * a 2 ∧
  a 5 > a 2

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := (-2) ^ (n - 1)

theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  geometric_sequence a → (∀ n : ℕ, a n = general_term n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2542_254232


namespace NUMINAMATH_CALUDE_odd_function_period_range_l2542_254290

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function has period p if f(x + p) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

/-- The smallest positive period of a function -/
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  HasPeriod f p ∧ p > 0 ∧ ∀ q, HasPeriod f q → q > 0 → p ≤ q

theorem odd_function_period_range (f : ℝ → ℝ) (m : ℝ) :
  IsOdd f →
  SmallestPositivePeriod f 3 →
  f 2015 > 1 →
  f 1 = (2 * m + 3) / (m - 1) →
  -2/3 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_period_range_l2542_254290


namespace NUMINAMATH_CALUDE_al_sandwich_options_l2542_254283

/-- Represents the number of different types of bread available. -/
def num_bread : ℕ := 4

/-- Represents the number of different types of meat available. -/
def num_meat : ℕ := 6

/-- Represents the number of different types of cheese available. -/
def num_cheese : ℕ := 5

/-- Represents whether ham is available. -/
def has_ham : Prop := True

/-- Represents whether chicken is available. -/
def has_chicken : Prop := True

/-- Represents whether cheddar cheese is available. -/
def has_cheddar : Prop := True

/-- Represents whether white bread is available. -/
def has_white_bread : Prop := True

/-- Represents the number of sandwiches with ham and cheddar cheese combination. -/
def ham_cheddar_combos : ℕ := num_bread

/-- Represents the number of sandwiches with white bread and chicken combination. -/
def white_chicken_combos : ℕ := num_cheese

/-- Theorem stating the number of different sandwiches Al could order. -/
theorem al_sandwich_options : 
  num_bread * num_meat * num_cheese - ham_cheddar_combos - white_chicken_combos = 111 := by
  sorry

end NUMINAMATH_CALUDE_al_sandwich_options_l2542_254283


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2542_254226

theorem fraction_multiplication : (1 : ℚ) / 3 * (1 : ℚ) / 2 * (3 : ℚ) / 4 * (5 : ℚ) / 6 = (5 : ℚ) / 48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2542_254226


namespace NUMINAMATH_CALUDE_rectangular_field_width_l2542_254258

/-- The width of a rectangular field given its length-to-width ratio and perimeter -/
def field_width (length_width_ratio : ℚ) (perimeter : ℚ) : ℚ :=
  perimeter / (2 * (length_width_ratio + 1))

theorem rectangular_field_width :
  field_width (7/5) 288 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l2542_254258


namespace NUMINAMATH_CALUDE_max_pairs_after_loss_l2542_254244

/-- Given a collection of shoes and a number of lost individual shoes,
    calculate the maximum number of matching pairs remaining. -/
def maxRemainingPairs (totalPairs : ℕ) (lostShoes : ℕ) : ℕ :=
  totalPairs - (lostShoes / 2) - (lostShoes % 2)

/-- Theorem: Given 150 pairs of shoes and a loss of 37 individual shoes,
    the maximum number of matching pairs remaining is 131. -/
theorem max_pairs_after_loss :
  maxRemainingPairs 150 37 = 131 := by
  sorry

#eval maxRemainingPairs 150 37

end NUMINAMATH_CALUDE_max_pairs_after_loss_l2542_254244


namespace NUMINAMATH_CALUDE_n_squared_divisible_by_144_l2542_254253

theorem n_squared_divisible_by_144 (n : ℕ+) (h : ∀ d : ℕ+, d ∣ n → d ≤ 12) :
  144 ∣ n^2 := by
  sorry

end NUMINAMATH_CALUDE_n_squared_divisible_by_144_l2542_254253


namespace NUMINAMATH_CALUDE_inheritance_tax_calculation_l2542_254296

theorem inheritance_tax_calculation (inheritance : ℝ) 
  (federal_tax_rate : ℝ) (state_tax_rate : ℝ) (total_tax : ℝ) : 
  inheritance = 38600 →
  federal_tax_rate = 0.25 →
  state_tax_rate = 0.15 →
  total_tax = 14000 →
  total_tax = inheritance * federal_tax_rate + 
    (inheritance - inheritance * federal_tax_rate) * state_tax_rate :=
by sorry

end NUMINAMATH_CALUDE_inheritance_tax_calculation_l2542_254296


namespace NUMINAMATH_CALUDE_congruent_triangles_on_skew_lines_l2542_254267

/-- Two lines in 3D space are skew if they are not parallel and do not intersect. -/
def are_skew (g l : Line3D) : Prop := sorry

/-- A point lies on a line in 3D space. -/
def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

/-- Two triangles in 3D space are congruent. -/
def triangles_congruent (t1 t2 : Triangle3D) : Prop := sorry

/-- The number of congruent triangles that can be constructed on two skew lines. -/
def num_congruent_triangles_on_skew_lines (g l : Line3D) (abc : Triangle3D) : ℕ := sorry

/-- Theorem: Given two skew lines and a triangle, there exist exactly 16 congruent triangles
    with vertices on the given lines. -/
theorem congruent_triangles_on_skew_lines (g l : Line3D) (abc : Triangle3D) :
  are_skew g l →
  num_congruent_triangles_on_skew_lines g l abc = 16 :=
by sorry

end NUMINAMATH_CALUDE_congruent_triangles_on_skew_lines_l2542_254267


namespace NUMINAMATH_CALUDE_sequence_equality_l2542_254236

/-- Sequence definition -/
def a (x : ℝ) (n : ℕ) : ℝ := 1 + x^(n+1) + x^(n+2)

/-- Main theorem -/
theorem sequence_equality (x : ℝ) :
  (a x 2)^2 = (a x 1) * (a x 3) →
  ∀ n ≥ 3, (a x n)^2 = (a x (n-1)) * (a x (n+1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_equality_l2542_254236


namespace NUMINAMATH_CALUDE_max_rational_products_l2542_254289

/-- Represents a number that can be either rational or irrational -/
inductive Number
| Rational : ℚ → Number
| Irrational : ℝ → Number

/-- Definition of the table structure -/
structure Table :=
  (top : Fin 50 → Number)
  (left : Fin 50 → Number)

/-- Counts the number of rational and irrational numbers in a list -/
def countNumbers (numbers : List Number) : Nat × Nat :=
  numbers.foldl (fun (ratCount, irratCount) n =>
    match n with
    | Number.Rational _ => (ratCount + 1, irratCount)
    | Number.Irrational _ => (ratCount, irratCount + 1)
  ) (0, 0)

/-- Checks if the product of two Numbers is rational -/
def isRationalProduct (a b : Number) : Bool :=
  match a, b with
  | Number.Rational _, Number.Rational _ => true
  | Number.Rational 0, _ => true
  | _, Number.Rational 0 => true
  | _, _ => false

/-- Counts the number of rational products in the table -/
def countRationalProducts (t : Table) : Nat :=
  (List.range 50).foldl (fun count i =>
    (List.range 50).foldl (fun count j =>
      if isRationalProduct (t.top i) (t.left j) then count + 1 else count
    ) count
  ) 0

/-- The main theorem -/
theorem max_rational_products (t : Table) :
  (countNumbers (List.ofFn t.top) = (25, 25) ∧
   countNumbers (List.ofFn t.left) = (25, 25) ∧
   (∀ i j : Fin 50, t.top i ≠ t.left j) ∧
   (∃ i : Fin 50, t.top i = Number.Rational 0)) →
  countRationalProducts t ≤ 1275 :=
sorry

end NUMINAMATH_CALUDE_max_rational_products_l2542_254289


namespace NUMINAMATH_CALUDE_miles_to_tie_l2542_254234

/-- The number of miles Billy runs each day from Sunday to Friday -/
def billy_daily_miles : ℚ := 1

/-- The number of miles Tiffany runs each day from Sunday to Tuesday -/
def tiffany_daily_miles_sun_to_tue : ℚ := 2

/-- The number of miles Tiffany runs each day from Wednesday to Friday -/
def tiffany_daily_miles_wed_to_fri : ℚ := 1/3

/-- The number of days Billy and Tiffany run from Sunday to Tuesday -/
def days_sun_to_tue : ℕ := 3

/-- The number of days Billy and Tiffany run from Wednesday to Friday -/
def days_wed_to_fri : ℕ := 3

theorem miles_to_tie : 
  (tiffany_daily_miles_sun_to_tue * days_sun_to_tue + 
   tiffany_daily_miles_wed_to_fri * days_wed_to_fri) - 
  (billy_daily_miles * (days_sun_to_tue + days_wed_to_fri)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_miles_to_tie_l2542_254234


namespace NUMINAMATH_CALUDE_tick_to_burr_ratio_l2542_254213

/-- Given a dog with burrs and ticks in its fur, prove the ratio of ticks to burrs. -/
theorem tick_to_burr_ratio (num_burrs num_total : ℕ) (h1 : num_burrs = 12) (h2 : num_total = 84) :
  (num_total - num_burrs) / num_burrs = 6 := by
  sorry

end NUMINAMATH_CALUDE_tick_to_burr_ratio_l2542_254213


namespace NUMINAMATH_CALUDE_set_membership_implies_value_l2542_254295

theorem set_membership_implies_value (m : ℤ) : 
  3 ∈ ({1, m+2} : Set ℤ) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_value_l2542_254295


namespace NUMINAMATH_CALUDE_candy_sampling_probability_l2542_254230

theorem candy_sampling_probability :
  let p_choose_A : ℝ := 0.40
  let p_choose_B : ℝ := 0.35
  let p_choose_C : ℝ := 0.25
  let p_sample_A : ℝ := 0.16 + 0.07
  let p_sample_B : ℝ := 0.24 + 0.15
  let p_sample_C : ℝ := 0.31 + 0.22
  let p_sample : ℝ := p_choose_A * p_sample_A + p_choose_B * p_sample_B + p_choose_C * p_sample_C
  p_sample = 0.361 :=
by sorry

end NUMINAMATH_CALUDE_candy_sampling_probability_l2542_254230


namespace NUMINAMATH_CALUDE_line_through_intersection_and_perpendicular_l2542_254271

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 7 * x - 8 * y - 1 = 0
def l2 (x y : ℝ) : Prop := 2 * x + 17 * y + 9 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x - y + 7 = 0

-- Define the resulting line
def result_line (x y : ℝ) : Prop := 27 * x + 54 * y + 37 = 0

-- Theorem statement
theorem line_through_intersection_and_perpendicular :
  ∃ (x y : ℝ),
    (l1 x y ∧ l2 x y) ∧  -- Intersection point satisfies both l1 and l2
    (∀ (x' y' : ℝ), result_line x' y' ↔ 
      (y' - y) = -(1/2) * (x' - x)) ∧  -- Slope of result_line is -1/2
    (∀ (x' y' : ℝ), perp_line x' y' → 
      (y' - y) * (1/2) + (x' - x) = 0)  -- Perpendicular to perp_line
    := by sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_perpendicular_l2542_254271


namespace NUMINAMATH_CALUDE_five_digit_multiples_of_five_l2542_254268

theorem five_digit_multiples_of_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 90000)).card = 18000 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_multiples_of_five_l2542_254268


namespace NUMINAMATH_CALUDE_max_value_inequality_l2542_254278

theorem max_value_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_condition : a + b + c + d ≤ 4) :
  (a * (b + 2 * c)) ^ (1/4) + (b * (c + 2 * d)) ^ (1/4) + 
  (c * (d + 2 * a)) ^ (1/4) + (d * (a + 2 * b)) ^ (1/4) ≤ 4 * 3 ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2542_254278


namespace NUMINAMATH_CALUDE_chang_e_2_orbit_period_l2542_254270

def b (α : ℕ → ℕ) : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1 + 1 / (α n + 1 / b α n)

theorem chang_e_2_orbit_period (α : ℕ → ℕ) :
  b α 4 < b α 7 := by
  sorry

end NUMINAMATH_CALUDE_chang_e_2_orbit_period_l2542_254270


namespace NUMINAMATH_CALUDE_rational_expression_proof_l2542_254224

theorem rational_expression_proof (a b c : ℚ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  ∃ (q : ℚ), q = |1 / (a - b) + 1 / (b - c) + 1 / (c - a)| := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_proof_l2542_254224


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2542_254275

/-- The number of sheep on the Stewart farm -/
def num_sheep : ℕ := 24

/-- The number of horses on the Stewart farm -/
def num_horses : ℕ := 56

/-- The ratio of sheep to horses -/
def sheep_to_horse_ratio : ℚ := 3 / 7

/-- The amount of food each horse eats per day in ounces -/
def horse_food_per_day : ℕ := 230

/-- The total amount of horse food needed per day in ounces -/
def total_horse_food_per_day : ℕ := 12880

theorem stewart_farm_sheep_count :
  (num_sheep : ℚ) / num_horses = sheep_to_horse_ratio ∧
  num_horses * horse_food_per_day = total_horse_food_per_day ∧
  num_sheep = 24 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2542_254275


namespace NUMINAMATH_CALUDE_specific_figure_triangles_l2542_254246

/-- Represents a triangular figure composed of smaller equilateral triangles. -/
structure TriangularFigure where
  row1 : Nat -- Number of triangles in the first row
  row2 : Nat -- Number of triangles in the second row
  row3 : Nat -- Number of triangles in the third row
  has_outer_triangle : Bool -- Whether there's a large triangle spanning all smaller triangles
  has_diagonal_cut : Bool -- Whether there's a diagonal cut over the bottom two rows

/-- Calculates the total number of triangles in the figure. -/
def total_triangles (figure : TriangularFigure) : Nat :=
  sorry

/-- Theorem stating that for the specific triangular figure described,
    the total number of triangles is 11. -/
theorem specific_figure_triangles :
  let figure : TriangularFigure := {
    row1 := 3,
    row2 := 2,
    row3 := 1,
    has_outer_triangle := true,
    has_diagonal_cut := true
  }
  total_triangles figure = 11 := by sorry

end NUMINAMATH_CALUDE_specific_figure_triangles_l2542_254246


namespace NUMINAMATH_CALUDE_four_digit_number_properties_l2542_254264

/-- P function for a four-digit number -/
def P (x : ℕ) : ℤ :=
  let y := (x % 10) * 1000 + x / 10
  (x - y) / 9

/-- Check if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k * k

/-- Definition of s -/
def s (a b : ℕ) : ℕ := 1100 + 20 * a + b

/-- Definition of t -/
def t (a b : ℕ) : ℕ := b * 1000 + a * 100 + 23

/-- Main theorem -/
theorem four_digit_number_properties :
  (P 5324 = 88) ∧
  (∀ a b : ℕ, 1 ≤ a → a ≤ 4 → 1 ≤ b → b ≤ 9 →
    (∃ min_pt : ℤ, min_pt = -161 ∧
      is_perfect_square (P (t a b) - P (s a b) - a - b) ∧
      (∀ a' b' : ℕ, 1 ≤ a' → a' ≤ 4 → 1 ≤ b' → b' ≤ 9 →
        is_perfect_square (P (t a' b') - P (s a' b') - a' - b') →
        P (t a' b') ≥ min_pt))) :=
sorry

end NUMINAMATH_CALUDE_four_digit_number_properties_l2542_254264


namespace NUMINAMATH_CALUDE_max_rectangle_area_max_area_condition_l2542_254231

/-- The maximum area of a rectangle with perimeter 40 meters (excluding one side) is 200 square meters. -/
theorem max_rectangle_area (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 40) : x * y ≤ 200 := by
  sorry

/-- The maximum area is achieved when the length is twice the width. -/
theorem max_area_condition (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 40) : 
  x * y = 200 ↔ x = 2*y := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_max_area_condition_l2542_254231


namespace NUMINAMATH_CALUDE_puppy_food_consumption_l2542_254260

/-- Represents the feeding schedule for a puppy over 4 weeks plus one day -/
structure PuppyFeeding where
  first_two_weeks : ℚ  -- Amount of food per day in first two weeks
  second_two_weeks : ℚ  -- Amount of food per day in second two weeks
  today : ℚ  -- Amount of food given today

/-- Calculates the total amount of food eaten by the puppy over 4 weeks plus one day -/
def total_food (feeding : PuppyFeeding) : ℚ :=
  feeding.first_two_weeks * 14 + feeding.second_two_weeks * 14 + feeding.today

/-- Theorem stating that the puppy will eat 25 cups of food over 4 weeks plus one day -/
theorem puppy_food_consumption :
  let feeding := PuppyFeeding.mk (3/4) 1 (1/2)
  total_food feeding = 25 := by sorry

end NUMINAMATH_CALUDE_puppy_food_consumption_l2542_254260


namespace NUMINAMATH_CALUDE_rotation_of_vector_l2542_254272

/-- Given points O and P in a 2D Cartesian plane, and Q obtained by rotating OP counterclockwise by 3π/4, 
    prove that Q has the specified coordinates. -/
theorem rotation_of_vector (O P Q : ℝ × ℝ) : 
  O = (0, 0) → 
  P = (6, 8) → 
  Q = (Real.cos (3 * Real.pi / 4) * (P.1 - O.1) - Real.sin (3 * Real.pi / 4) * (P.2 - O.2) + O.1,
       Real.sin (3 * Real.pi / 4) * (P.1 - O.1) + Real.cos (3 * Real.pi / 4) * (P.2 - O.2) + O.2) →
  Q = (-7 * Real.sqrt 2, -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_rotation_of_vector_l2542_254272


namespace NUMINAMATH_CALUDE_f_negative_2016_l2542_254215

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 1

theorem f_negative_2016 (a : ℝ) :
  f a 2016 = 5 → f a (-2016) = -7 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_2016_l2542_254215


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l2542_254227

theorem negative_fractions_comparison : -2/3 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l2542_254227


namespace NUMINAMATH_CALUDE_fourier_expansion_arccos_plus_one_l2542_254233

-- Define the Chebyshev polynomials
noncomputable def T (n : ℕ) (x : ℝ) : ℝ := Real.cos (n * Real.arccos x)

-- Define the function to be expanded
noncomputable def f (x : ℝ) : ℝ := Real.arccos x + 1

-- Define the interval
def I : Set ℝ := Set.Ioo (-1) 1

-- Define the Fourier coefficient
noncomputable def a (n : ℕ) : ℝ :=
  if n = 0
  then (Real.pi + 2) / 2
  else 2 / Real.pi * ((-1)^n - 1) / (n^2 : ℝ)

-- State the theorem
theorem fourier_expansion_arccos_plus_one :
  ∀ x ∈ I, f x = (Real.pi + 2) / 2 + ∑' n, a n * T n x :=
sorry

end NUMINAMATH_CALUDE_fourier_expansion_arccos_plus_one_l2542_254233


namespace NUMINAMATH_CALUDE_base_7_to_decimal_l2542_254291

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

theorem base_7_to_decimal :
  to_decimal [6, 5, 7] 7 = 384 := by
  sorry

end NUMINAMATH_CALUDE_base_7_to_decimal_l2542_254291


namespace NUMINAMATH_CALUDE_power_of_two_sum_l2542_254249

theorem power_of_two_sum : 2^3 * 2^4 + 2^5 = 160 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l2542_254249


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2542_254217

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/3
  let S : ℝ := ∑' n, a * r^n
  S = 3/2 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2542_254217


namespace NUMINAMATH_CALUDE_total_paid_is_230_l2542_254208

/-- The cost of an item before tax -/
def cost : ℝ := 200

/-- The tax rate as a decimal -/
def tax_rate : ℝ := 0.15

/-- The total amount paid after tax -/
def total_paid : ℝ := cost + (cost * tax_rate)

/-- Theorem stating that the total amount paid after tax is $230 -/
theorem total_paid_is_230 : total_paid = 230 := by
  sorry

end NUMINAMATH_CALUDE_total_paid_is_230_l2542_254208


namespace NUMINAMATH_CALUDE_solve_equation_l2542_254294

theorem solve_equation : ∃ x : ℚ, (3 * x) / 7 = 6 ∧ x = 14 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2542_254294


namespace NUMINAMATH_CALUDE_divisibility_property_l2542_254285

theorem divisibility_property (n : ℕ) : 
  ∃ (a b : ℕ), (a * n + 1)^6 + b ≡ 0 [MOD (n^2 + n + 1)] :=
by
  use 2, 27
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2542_254285


namespace NUMINAMATH_CALUDE_valid_numbers_with_sum_444_l2542_254219

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 10 ≠ 0 ∧ (n / 10) % 10 ≠ 0 ∧ n / 100 ≠ 0

def sum_of_permutations (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  if a = b ∧ b = c then
    n
  else if a = b ∨ b = c ∨ a = c then
    100 * (a + b + c) + 10 * (a + b + c) + (a + b + c)
  else
    222 * (a + b + c)

theorem valid_numbers_with_sum_444 (n : ℕ) :
  is_valid_number n ∧ sum_of_permutations n = 444 →
  n = 112 ∨ n = 121 ∨ n = 211 ∨ n = 444 :=
by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_with_sum_444_l2542_254219


namespace NUMINAMATH_CALUDE_mutually_exclusive_necessary_not_sufficient_l2542_254299

open Set

variable {Ω : Type*} [MeasurableSpace Ω]

def mutually_exclusive (A₁ A₂ : Set Ω) : Prop := A₁ ∩ A₂ = ∅

def complementary (A₁ A₂ : Set Ω) : Prop := A₁ ∪ A₂ = univ ∧ A₁ ∩ A₂ = ∅

theorem mutually_exclusive_necessary_not_sufficient :
  (∀ A₁ A₂ : Set Ω, complementary A₁ A₂ → mutually_exclusive A₁ A₂) ∧
  (∃ A₁ A₂ : Set Ω, mutually_exclusive A₁ A₂ ∧ ¬complementary A₁ A₂) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_necessary_not_sufficient_l2542_254299


namespace NUMINAMATH_CALUDE_badge_exchange_l2542_254274

theorem badge_exchange (V T : ℕ) : 
  V = T + 5 ∧ 
  (V - V * 24 / 100 + T * 20 / 100 : ℚ) = (T - T * 20 / 100 + V * 24 / 100 : ℚ) - 1 →
  V = 50 ∧ T = 45 :=
by sorry

end NUMINAMATH_CALUDE_badge_exchange_l2542_254274


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2542_254263

theorem simplify_and_evaluate (x : ℝ) (h : x^2 + 4*x - 4 = 0) :
  3*(x-2)^2 - 6*(x+1)*(x-1) = 6 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2542_254263


namespace NUMINAMATH_CALUDE_simplify_expression_l2542_254282

theorem simplify_expression : 4 * (15 / 5) * (24 / -60) = -24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2542_254282


namespace NUMINAMATH_CALUDE_same_units_digit_count_l2542_254287

def old_page_numbers := Finset.range 60

theorem same_units_digit_count :
  (old_page_numbers.filter (λ x => x % 10 = (61 - x) % 10)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_same_units_digit_count_l2542_254287


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2542_254286

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ a : ℝ, x = a^2 + 1}
def Q : Set ℝ := {x | ∃ y : ℝ, y = Real.log (2 - x)}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2542_254286


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2542_254259

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - x - 2 < 0} = Set.Ioo (-1 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2542_254259


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2542_254284

theorem tangent_line_to_circle (a : ℝ) : 
  (∀ x y : ℝ, ax + y + 1 = 0 → (x - 2)^2 + y^2 = 4) →
  (∃! x y : ℝ, ax + y + 1 = 0 ∧ (x - 2)^2 + y^2 = 4) →
  a = 3/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2542_254284


namespace NUMINAMATH_CALUDE_people_after_yoongi_l2542_254288

theorem people_after_yoongi (total : ℕ) (before : ℕ) (h1 : total = 20) (h2 : before = 11) :
  total - (before + 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_people_after_yoongi_l2542_254288


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2542_254279

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/2⌋ = ⌊x + 3⌋ ↔ 2 ≤ x ∧ x < 7/3 := by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2542_254279


namespace NUMINAMATH_CALUDE_specific_coin_expected_value_l2542_254265

/-- A biased coin with probabilities for heads and tails, and associated winnings/losses. -/
structure BiasedCoin where
  prob_heads : ℚ
  prob_tails : ℚ
  win_heads : ℚ
  loss_tails : ℚ

/-- Expected value of winnings for a single flip of a biased coin. -/
def expected_value (coin : BiasedCoin) : ℚ :=
  coin.prob_heads * coin.win_heads + coin.prob_tails * (-coin.loss_tails)

/-- Theorem stating the expected value for the specific coin in the problem. -/
theorem specific_coin_expected_value :
  let coin : BiasedCoin := {
    prob_heads := 1/4,
    prob_tails := 3/4,
    win_heads := 4,
    loss_tails := 3
  }
  expected_value coin = -5/4 := by sorry

end NUMINAMATH_CALUDE_specific_coin_expected_value_l2542_254265


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l2542_254211

theorem wire_cutting_problem (initial_length second_length num_pieces : ℕ) 
  (h1 : initial_length = 1000)
  (h2 : second_length = 1050)
  (h3 : num_pieces = 14)
  (h4 : ∃ (piece_length : ℕ), 
    piece_length * num_pieces = initial_length ∧ 
    piece_length * num_pieces = second_length) :
  ∃ (piece_length : ℕ), piece_length = 71 ∧ 
    piece_length * num_pieces = initial_length ∧ 
    piece_length * num_pieces = second_length :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l2542_254211


namespace NUMINAMATH_CALUDE_leahs_coins_value_l2542_254200

/-- Represents the number and value of coins Leah has -/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total number of coins -/
def CoinCollection.total (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes

/-- Calculates the total value of coins in cents -/
def CoinCollection.value (c : CoinCollection) : ℕ :=
  c.pennies + 5 * c.nickels + 10 * c.dimes

/-- Theorem stating that Leah's coins are worth 66 cents -/
theorem leahs_coins_value (c : CoinCollection) : c.value = 66 :=
  by
  have h1 : c.total = 15 := sorry
  have h2 : c.pennies = c.nickels + 3 := sorry
  sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l2542_254200


namespace NUMINAMATH_CALUDE_unique_bounded_sequence_l2542_254218

def sequence_relation (a : ℕ → ℕ) :=
  ∀ n : ℕ, n ≥ 3 → a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))

theorem unique_bounded_sequence (a : ℕ → ℕ) :
  sequence_relation a →
  (∃ M : ℕ, ∀ n : ℕ, a n ≤ M) →
  (∀ n : ℕ, a n = 2) :=
sorry

end NUMINAMATH_CALUDE_unique_bounded_sequence_l2542_254218


namespace NUMINAMATH_CALUDE_min_cubes_is_60_l2542_254235

/-- The dimensions of the box in centimeters -/
def box_dimensions : Fin 3 → ℕ
| 0 => 30
| 1 => 40
| 2 => 50
| _ => 0

/-- The function to calculate the minimum number of cubes -/
def min_cubes (dimensions : Fin 3 → ℕ) : ℕ :=
  let cube_side := Nat.gcd (dimensions 0) (Nat.gcd (dimensions 1) (dimensions 2))
  (dimensions 0 / cube_side) * (dimensions 1 / cube_side) * (dimensions 2 / cube_side)

/-- Theorem stating that the minimum number of cubes is 60 -/
theorem min_cubes_is_60 : min_cubes box_dimensions = 60 := by
  sorry

#eval min_cubes box_dimensions

end NUMINAMATH_CALUDE_min_cubes_is_60_l2542_254235


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l2542_254201

/-- Represents the profit distribution in a partnership business -/
structure ProfitDistribution where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  c_profit : ℕ

/-- Calculates the total profit given a profit distribution -/
def total_profit (pd : ProfitDistribution) : ℕ :=
  let total_investment := pd.a_investment + pd.b_investment + pd.c_investment
  let c_ratio := pd.c_investment / (total_investment / 20)
  (pd.c_profit * 20) / c_ratio

/-- Theorem stating that given the specific investments and c's profit, 
    the total profit is $60,000 -/
theorem partnership_profit_calculation (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 45000)
  (h2 : pd.b_investment = 63000)
  (h3 : pd.c_investment = 72000)
  (h4 : pd.c_profit = 24000) :
  total_profit pd = 60000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l2542_254201


namespace NUMINAMATH_CALUDE_pages_read_on_tuesday_l2542_254207

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Berry's daily reading goal -/
def daily_goal : ℕ := 50

/-- Pages read on Sunday -/
def sunday_pages : ℕ := 43

/-- Pages read on Monday -/
def monday_pages : ℕ := 65

/-- Pages read on Wednesday -/
def wednesday_pages : ℕ := 0

/-- Pages read on Thursday -/
def thursday_pages : ℕ := 70

/-- Pages read on Friday -/
def friday_pages : ℕ := 56

/-- Pages to be read on Saturday -/
def saturday_pages : ℕ := 88

/-- Theorem stating that Berry must have read 28 pages on Tuesday to achieve his weekly goal -/
theorem pages_read_on_tuesday : 
  ∃ (tuesday_pages : ℕ), 
    (sunday_pages + monday_pages + tuesday_pages + wednesday_pages + 
     thursday_pages + friday_pages + saturday_pages) = 
    (daily_goal * days_in_week) ∧ tuesday_pages = 28 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_on_tuesday_l2542_254207


namespace NUMINAMATH_CALUDE_ball_probability_6_l2542_254281

def ball_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | 1 => 0
  | k + 2 => (1 / 3) * (1 - ball_probability (k + 1))

theorem ball_probability_6 :
  ball_probability 6 = 61 / 243 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_6_l2542_254281


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l2542_254273

structure IsoscelesTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  isIsosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3)
  sumIs180 : angle1 + angle2 + angle3 = 180

theorem isosceles_triangle_base_angle 
  (triangle : IsoscelesTriangle) 
  (has80DegreeAngle : triangle.angle1 = 80 ∨ triangle.angle2 = 80 ∨ triangle.angle3 = 80) :
  (∃ baseAngle : ℝ, (baseAngle = 80 ∨ baseAngle = 50) ∧ 
   ((triangle.angle1 = baseAngle ∧ triangle.angle2 = baseAngle) ∨
    (triangle.angle1 = baseAngle ∧ triangle.angle3 = baseAngle) ∨
    (triangle.angle2 = baseAngle ∧ triangle.angle3 = baseAngle))) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l2542_254273


namespace NUMINAMATH_CALUDE_work_completion_time_l2542_254250

theorem work_completion_time 
  (people : ℕ) 
  (original_time : ℕ) 
  (original_work : ℝ) 
  (h1 : original_time = 16) 
  (h2 : people * original_time = original_work) :
  (2 * people) * 8 = original_work / 2 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2542_254250


namespace NUMINAMATH_CALUDE_box_counting_l2542_254206

theorem box_counting (initial_boxes : Nat) (boxes_per_fill : Nat) (non_empty_boxes : Nat) :
  initial_boxes = 7 →
  boxes_per_fill = 7 →
  non_empty_boxes = 10 →
  initial_boxes + (non_empty_boxes - 1) * boxes_per_fill = 77 := by
  sorry

end NUMINAMATH_CALUDE_box_counting_l2542_254206


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2542_254256

/-- Given a line L1 defined by 4x + 5y = 20, and a perpendicular line L2 with y-intercept -3,
    the x-intercept of L2 is 12/5 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y => 4 * x + 5 * y = 20
  let m1 : ℝ := -4 / 5  -- slope of L1
  let m2 : ℝ := 5 / 4   -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y => y = m2 * x - 3  -- equation of L2
  ∃ x : ℝ, L2 x 0 ∧ x = 12 / 5
  :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l2542_254256


namespace NUMINAMATH_CALUDE_ellipse_equation_l2542_254209

/-- Prove that an ellipse passing through (2,0) with focal distance 2√2 has the equation x²/4 + y²/2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → x^2/4 + y^2/2 = 1) ↔
  (4/a^2 + 0^2/b^2 = 1 ∧ a^2 - b^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2542_254209


namespace NUMINAMATH_CALUDE_cubic_curve_rational_points_l2542_254237

-- Define a cubic curve with rational coefficients
def CubicCurve (f : ℚ → ℚ → ℚ) : Prop :=
  ∀ x y, ∃ a b c d e g h k l : ℚ, 
    f x y = a*x^3 + b*x^2*y + c*x*y^2 + d*y^3 + e*x^2 + g*x*y + h*y^2 + k*x + l*y

-- Define a point on the curve
def PointOnCurve (f : ℚ → ℚ → ℚ) (x y : ℚ) : Prop :=
  f x y = 0

-- Theorem statement
theorem cubic_curve_rational_points 
  (f : ℚ → ℚ → ℚ) 
  (hf : CubicCurve f) 
  (x₀ y₀ : ℚ) 
  (h₀ : PointOnCurve f x₀ y₀) :
  ∃ x' y' : ℚ, x' ≠ x₀ ∧ y' ≠ y₀ ∧ PointOnCurve f x' y' :=
sorry

end NUMINAMATH_CALUDE_cubic_curve_rational_points_l2542_254237


namespace NUMINAMATH_CALUDE_petya_win_probability_l2542_254240

/-- The "Pile of Stones" game --/
structure PileOfStones :=
  (initial_stones : ℕ)
  (min_take : ℕ)
  (max_take : ℕ)

/-- A player in the "Pile of Stones" game --/
inductive Player
| Petya
| Computer

/-- The strategy of a player --/
def Strategy := ℕ → ℕ

/-- The optimal strategy for the second player --/
def optimal_strategy : Strategy := sorry

/-- A random strategy that always takes between min_take and max_take stones --/
def random_strategy (game : PileOfStones) : Strategy := sorry

/-- The probability of winning for a player given their strategy and the opponent's strategy --/
def win_probability (game : PileOfStones) (player : Player) (player_strategy : Strategy) (opponent_strategy : Strategy) : ℚ := sorry

/-- The main theorem: Petya's probability of winning is 1/256 --/
theorem petya_win_probability :
  let game : PileOfStones := ⟨16, 1, 4⟩
  win_probability game Player.Petya (random_strategy game) optimal_strategy = 1 / 256 := by sorry

end NUMINAMATH_CALUDE_petya_win_probability_l2542_254240


namespace NUMINAMATH_CALUDE_book_sale_profit_percentage_l2542_254223

/-- Calculates the profit percentage after tax for a book sale -/
theorem book_sale_profit_percentage 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (tax_rate : ℝ) 
  (h1 : cost_price = 32) 
  (h2 : selling_price = 56) 
  (h3 : tax_rate = 0.07) : 
  (selling_price * (1 - tax_rate) - cost_price) / cost_price * 100 = 62.75 := by
sorry

end NUMINAMATH_CALUDE_book_sale_profit_percentage_l2542_254223


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2542_254238

theorem complex_fraction_equality (z : ℂ) (h : z = 1 - I) : 
  (z^2 - 2*z) / (z - 1) = -1 - I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2542_254238


namespace NUMINAMATH_CALUDE_log_of_geometric_is_arithmetic_l2542_254228

theorem log_of_geometric_is_arithmetic (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_geom : b / a = c / b) : 
  Real.log b - Real.log a = Real.log c - Real.log b :=
sorry

end NUMINAMATH_CALUDE_log_of_geometric_is_arithmetic_l2542_254228


namespace NUMINAMATH_CALUDE_distinct_cube_digits_mod_seven_l2542_254266

theorem distinct_cube_digits_mod_seven :
  ∃! s : Finset ℕ, 
    (∀ n : ℕ, (n^3 % 7) ∈ s) ∧ 
    (∀ m ∈ s, ∃ n : ℕ, n^3 % 7 = m) ∧
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_cube_digits_mod_seven_l2542_254266


namespace NUMINAMATH_CALUDE_evaluate_expression_l2542_254293

theorem evaluate_expression : 
  Real.sqrt ((5 - 3 * Real.sqrt 5) ^ 2) - Real.sqrt ((5 + 3 * Real.sqrt 5) ^ 2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2542_254293


namespace NUMINAMATH_CALUDE_shape_selections_count_l2542_254257

/-- Represents a regular hexagonal grid --/
structure HexagonalGrid :=
  (size : ℕ)

/-- Represents a shape that can be selected from the grid --/
structure Shape :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the number of ways to select a shape from a hexagonal grid --/
def selectionsCount (grid : HexagonalGrid) (shape : Shape) : ℕ :=
  sorry

/-- The number of distinct rotations for a shape in a hexagonal grid --/
def rotationsCount : ℕ := 3

/-- Theorem stating that there are 72 ways to select the given shape from the hexagonal grid --/
theorem shape_selections_count :
  ∀ (grid : HexagonalGrid) (shape : Shape),
  grid.size = 5 →  -- Assuming the grid size is 5 based on the problem description
  shape.width = 2 →  -- Assuming the shape width is 2 based on diagram b
  shape.height = 2 →  -- Assuming the shape height is 2 based on diagram b
  selectionsCount grid shape * rotationsCount = 72 :=
sorry

end NUMINAMATH_CALUDE_shape_selections_count_l2542_254257


namespace NUMINAMATH_CALUDE_percent_decrease_l2542_254248

theorem percent_decrease (X Y : ℝ) (h : Y = 1.2 * X) :
  X = Y * (1 - 1/6) :=
by sorry

end NUMINAMATH_CALUDE_percent_decrease_l2542_254248


namespace NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l2542_254247

-- Define the function f
def f (x c : ℝ) : ℝ := x^2 + 5*x + c

-- State the theorem
theorem largest_c_for_negative_three_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), 
    (∃ (x : ℝ), f x c = -3) → 
    (∃ (x : ℝ), f x d = -3) → 
    d ≤ c) ∧
  (∃ (x : ℝ), f x (13/4) = -3) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_three_in_range_l2542_254247


namespace NUMINAMATH_CALUDE_range_of_y_l2542_254292

theorem range_of_y (x y : ℝ) (h1 : x = 4 - y) (h2 : -2 ≤ x ∧ x ≤ -1) : 5 ≤ y ∧ y ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_y_l2542_254292


namespace NUMINAMATH_CALUDE_dinosaur_weight_theorem_l2542_254241

/-- The combined weight of Barney, five regular dinosaurs, and their food -/
def total_weight (regular_weight food_weight : ℕ) : ℕ :=
  let regular_combined := 5 * regular_weight
  let barney_weight := regular_combined + 1500
  barney_weight + regular_combined + food_weight

/-- Theorem stating the total weight of the dinosaurs and their food -/
theorem dinosaur_weight_theorem (X : ℕ) :
  total_weight 800 X = 9500 + X :=
by
  sorry

end NUMINAMATH_CALUDE_dinosaur_weight_theorem_l2542_254241


namespace NUMINAMATH_CALUDE_triangle_angle_A_l2542_254297

theorem triangle_angle_A (a b : ℝ) (B : ℝ) (hA : 0 < a) (hB : 0 < b) (hC : 0 < B) :
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  B = π / 3 →
  ∃ (A : ℝ), 0 < A ∧ A < 2 * π / 3 ∧ Real.sin A = Real.sqrt 2 / 2 ∧ A = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l2542_254297


namespace NUMINAMATH_CALUDE_marathon_theorem_l2542_254280

def marathon_problem (total_distance : ℝ) (day1_percent : ℝ) (day2_percent : ℝ) : ℝ :=
  let day1_distance := total_distance * day1_percent
  let remaining_after_day1 := total_distance - day1_distance
  let day2_distance := remaining_after_day1 * day2_percent
  let day3_distance := total_distance - day1_distance - day2_distance
  day3_distance

theorem marathon_theorem :
  marathon_problem 70 0.2 0.5 = 28 := by
  sorry

end NUMINAMATH_CALUDE_marathon_theorem_l2542_254280


namespace NUMINAMATH_CALUDE_triangle_max_area_l2542_254222

theorem triangle_max_area (A B C : ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) (h5 : Real.cos A / Real.sin B + Real.cos B / Real.sin A = 2) 
  (h6 : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 12 ∧ 
    a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
  ∃ (S : ℝ), S ≤ 36 * (3 - 2 * Real.sqrt 2) ∧ 
    (∀ (S' : ℝ), (∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 12 ∧ 
      a' / Real.sin A = b' / Real.sin B ∧ b' / Real.sin B = c' / Real.sin C ∧ 
      S' = 1/2 * a' * b' * Real.sin C) → S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2542_254222


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2542_254204

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2542_254204


namespace NUMINAMATH_CALUDE_gcd_three_digit_palindromes_l2542_254216

def three_digit_palindrome (a b : ℕ) : ℕ := 101 * a + 10 * b

theorem gcd_three_digit_palindromes :
  ∃ (d : ℕ), d > 0 ∧
  (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
    d ∣ three_digit_palindrome a b) ∧
  (∀ (d' : ℕ), d' > d →
    ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
      ¬(d' ∣ three_digit_palindrome a b)) ∧
  d = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_three_digit_palindromes_l2542_254216


namespace NUMINAMATH_CALUDE_ball_triangle_ratio_l2542_254205

theorem ball_triangle_ratio (q : ℝ) (hq : q ≠ 1) :
  let r := 2012 / q
  let side1 := r * (1 + q)
  let side2 := 2012 * (1 + q)
  let side3 := 2012 * (1 + q^2) / q
  (side1^2 + side2^2 + side3^2) / (side1 + side2 + side3) = 4024 :=
sorry

end NUMINAMATH_CALUDE_ball_triangle_ratio_l2542_254205


namespace NUMINAMATH_CALUDE_million_place_seven_digits_l2542_254276

/-- A place value in a number system. -/
inductive PlaceValue
  | Units
  | Tens
  | Hundreds
  | Thousands
  | TenThousands
  | HundredThousands
  | Millions

/-- The number of digits in a place value. -/
def PlaceValue.digits : PlaceValue → Nat
  | Units => 1
  | Tens => 2
  | Hundreds => 3
  | Thousands => 4
  | TenThousands => 5
  | HundredThousands => 6
  | Millions => 7

/-- A number with its highest place being the million place has 7 digits. -/
theorem million_place_seven_digits :
  PlaceValue.digits PlaceValue.Millions = 7 := by
  sorry

end NUMINAMATH_CALUDE_million_place_seven_digits_l2542_254276


namespace NUMINAMATH_CALUDE_perpendicular_condition_l2542_254277

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "lies within" relation for a line and a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define non-coincidence for lines
variable (non_coincident : Line → Line → Line → Prop)

theorem perpendicular_condition 
  (l m n : Line) (α : Plane)
  (h_non_coincident : non_coincident l m n)
  (h_m_in_α : line_in_plane m α)
  (h_n_in_α : line_in_plane n α) :
  (perp_line_plane l α → perp_line_line l m ∧ perp_line_line l n) ∧
  ¬(perp_line_line l m ∧ perp_line_line l n → perp_line_plane l α) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l2542_254277


namespace NUMINAMATH_CALUDE_vector_arrangements_l2542_254225

-- Define a structure for a vector in 2D space
structure Vector2D where
  x : ℝ
  y : ℝ

-- Define a function to check if two vectors are parallel
def areParallel (v1 v2 : Vector2D) : Prop :=
  ∃ (k : ℝ), v1.x = k * v2.x ∧ v1.y = k * v2.y

-- Define a function to check if a quadrilateral is non-convex
def isNonConvex (v1 v2 v3 v4 : Vector2D) : Prop :=
  sorry -- Definition of non-convex quadrilateral

-- Define a function to check if a four-segment broken line is self-intersecting
def isSelfIntersecting (v1 v2 v3 v4 : Vector2D) : Prop :=
  sorry -- Definition of self-intersecting broken line

theorem vector_arrangements (v1 v2 v3 v4 : Vector2D) :
  (¬ areParallel v1 v2 ∧ ¬ areParallel v1 v3 ∧ ¬ areParallel v1 v4 ∧
   ¬ areParallel v2 v3 ∧ ¬ areParallel v2 v4 ∧ ¬ areParallel v3 v4) →
  (v1.x + v2.x + v3.x + v4.x = 0 ∧ v1.y + v2.y + v3.y + v4.y = 0) →
  (∃ (a b c d : Vector2D), isNonConvex a b c d) ∧
  (∃ (a b c d : Vector2D), isSelfIntersecting a b c d) :=
by
  sorry


end NUMINAMATH_CALUDE_vector_arrangements_l2542_254225


namespace NUMINAMATH_CALUDE_magnitude_of_Z_l2542_254298

def Z : ℂ := Complex.mk 3 (-4)

theorem magnitude_of_Z : Complex.abs Z = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_Z_l2542_254298


namespace NUMINAMATH_CALUDE_figure_area_theorem_l2542_254214

theorem figure_area_theorem (x : ℝ) :
  let small_square_area := (3 * x)^2
  let large_square_area := (7 * x)^2
  let triangle_area := (1/2) * (3 * x) * (7 * x)
  small_square_area + large_square_area + triangle_area = 2200 →
  x = Real.sqrt (4400 / 137) :=
by sorry

end NUMINAMATH_CALUDE_figure_area_theorem_l2542_254214


namespace NUMINAMATH_CALUDE_k_domain_l2542_254202

-- Define the function h
def h : ℝ → ℝ := sorry

-- Define the domain of h
def h_domain : Set ℝ := Set.Icc (-8) 4

-- Define the function k in terms of h
def k (x : ℝ) : ℝ := h (3 * x + 1)

-- State the theorem
theorem k_domain :
  {x : ℝ | k x ∈ Set.range h} = Set.Icc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_k_domain_l2542_254202


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2542_254252

-- Define the functional equation
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y * f (x - y) = x^2 * f y - y^2 * f x

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, satisfies_equation f →
    (∀ x : ℝ, f x = x ∨ f x = -x ∨ f x = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l2542_254252


namespace NUMINAMATH_CALUDE_glutenNutNonVegan_is_65_l2542_254254

/-- Represents the number of cupcakes with specific properties -/
structure Cupcakes where
  total : ℕ
  glutenFree : ℕ
  vegan : ℕ
  nutFree : ℕ
  glutenFreeVegan : ℕ
  veganNutFree : ℕ

/-- The properties of the cupcakes ordered for the birthday party -/
def birthdayCupcakes : Cupcakes where
  total := 120
  glutenFree := 120 / 3
  vegan := 120 / 4
  nutFree := 120 / 5
  glutenFreeVegan := 15
  veganNutFree := 10

/-- Calculates the number of cupcakes that are gluten, nut, and non-vegan -/
def glutenNutNonVegan (c : Cupcakes) : ℕ :=
  c.total - (c.glutenFree + (c.vegan - c.glutenFreeVegan))

/-- Theorem stating that the number of gluten, nut, and non-vegan cupcakes is 65 -/
theorem glutenNutNonVegan_is_65 : glutenNutNonVegan birthdayCupcakes = 65 := by
  sorry

end NUMINAMATH_CALUDE_glutenNutNonVegan_is_65_l2542_254254


namespace NUMINAMATH_CALUDE_sum_range_l2542_254212

theorem sum_range (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let S := a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)
  1 ≤ S ∧ S ≤ 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_range_l2542_254212


namespace NUMINAMATH_CALUDE_box_storage_calculation_l2542_254243

/-- Calculates the total number of boxes stored on a rectangular piece of land over two days -/
theorem box_storage_calculation (land_width land_length : ℕ) 
  (box_dimension : ℕ) (day1_layers day2_layers : ℕ) : 
  land_width = 44 → 
  land_length = 35 → 
  box_dimension = 1 → 
  day1_layers = 7 → 
  day2_layers = 3 → 
  (land_width / box_dimension) * (land_length / box_dimension) * (day1_layers + day2_layers) = 15400 :=
by
  sorry

#check box_storage_calculation

end NUMINAMATH_CALUDE_box_storage_calculation_l2542_254243


namespace NUMINAMATH_CALUDE_solve_boat_speed_l2542_254210

def boat_speed_problem (stream_speed : ℝ) (distance : ℝ) (total_time : ℝ) : Prop :=
  ∃ (boat_speed : ℝ),
    boat_speed > stream_speed ∧
    (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = total_time ∧
    boat_speed = 9

theorem solve_boat_speed : boat_speed_problem 1.5 105 24 := by
  sorry

end NUMINAMATH_CALUDE_solve_boat_speed_l2542_254210


namespace NUMINAMATH_CALUDE_mary_berry_cost_l2542_254251

/-- The amount Mary paid for berries, given her total payment, peach cost, and change received. -/
theorem mary_berry_cost (total_paid change peach_cost : ℚ) 
  (h1 : total_paid = 20)
  (h2 : change = 598/100)
  (h3 : peach_cost = 683/100) :
  total_paid - change - peach_cost = 719/100 := by
  sorry


end NUMINAMATH_CALUDE_mary_berry_cost_l2542_254251


namespace NUMINAMATH_CALUDE_floor_e_equals_two_l2542_254261

theorem floor_e_equals_two : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_e_equals_two_l2542_254261


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2542_254255

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n : ℕ, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d) →  -- arithmetic sequence
  b 5 * b 6 = 35 →
  b 4 * b 7 = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2542_254255


namespace NUMINAMATH_CALUDE_circle_construction_cases_l2542_254221

/-- Two lines in a plane -/
structure Line where
  -- Add necessary fields for a line

/-- A point in a plane -/
structure Point where
  -- Add necessary fields for a point

/-- A circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point is on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  sorry

/-- Predicate to check if two lines intersect -/
def lines_intersect (l1 l2 : Line) : Prop :=
  sorry

/-- Predicate to check if two lines are perpendicular -/
def lines_perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- Predicate to check if a circle is tangent to a line -/
def circle_tangent_to_line (c : Circle) (l : Line) : Prop :=
  sorry

/-- Predicate to check if a point is on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop :=
  sorry

/-- Main theorem -/
theorem circle_construction_cases
  (a b : Line) (P : Point)
  (h1 : lines_intersect a b)
  (h2 : point_on_line P b) :
  (∃ c1 c2 : Circle,
    c1 ≠ c2 ∧
    circle_tangent_to_line c1 a ∧
    circle_tangent_to_line c2 a ∧
    point_on_circle P c1 ∧
    point_on_circle P c2 ∧
    point_on_line c1.center b ∧
    point_on_line c2.center b) ∨
  (∃ Q : Point, point_on_line Q a ∧ point_on_line Q b ∧ P = Q) ∨
  (lines_perpendicular a b) :=
sorry

end NUMINAMATH_CALUDE_circle_construction_cases_l2542_254221


namespace NUMINAMATH_CALUDE_norma_bananas_l2542_254239

theorem norma_bananas (initial : ℕ) (lost : ℕ) (final : ℕ) :
  initial = 47 →
  lost = 45 →
  final = initial - lost →
  final = 2 :=
by sorry

end NUMINAMATH_CALUDE_norma_bananas_l2542_254239


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2542_254242

/-- The equation x^2 + ax + b = 0 has two distinct positive roots less than 1 -/
def has_two_distinct_positive_roots_less_than_one (a b : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
    x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0

/-- p is a necessary but not sufficient condition for q -/
theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (has_two_distinct_positive_roots_less_than_one a b → -2 < a ∧ a < 0 ∧ 0 < b ∧ b < 1) ∧
  ¬((-2 < a ∧ a < 0 ∧ 0 < b ∧ b < 1) → has_two_distinct_positive_roots_less_than_one a b) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2542_254242


namespace NUMINAMATH_CALUDE_point_not_outside_implies_on_or_inside_l2542_254220

-- Define a circle in a 2D plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the possible relationships between a point and a circle
inductive PointCircleRelation
  | Inside
  | On
  | Outside

-- Function to determine the relation between a point and a circle
def pointCircleRelation (p : ℝ × ℝ) (c : Circle) : PointCircleRelation :=
  sorry

-- Theorem statement
theorem point_not_outside_implies_on_or_inside
  (p : ℝ × ℝ) (c : Circle) :
  pointCircleRelation p c ≠ PointCircleRelation.Outside →
  (pointCircleRelation p c = PointCircleRelation.On ∨
   pointCircleRelation p c = PointCircleRelation.Inside) :=
by sorry

end NUMINAMATH_CALUDE_point_not_outside_implies_on_or_inside_l2542_254220


namespace NUMINAMATH_CALUDE_f_properties_l2542_254229

/-- The function f(x) = mx² + 1 + ln x -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 1 + Real.log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := 2 * m * x + 1 / x

/-- Theorem stating the main properties to be proved -/
theorem f_properties (m : ℝ) (n : ℝ) (a b : ℝ) :
  (∃ (t : ℝ), t = f_deriv m 1 ∧ 2 = f m 1 + t * (-2)) →  -- Tangent line condition
  (f m a = n ∧ f m b = n ∧ a < b) →                      -- Roots condition
  (∀ x > 0, f m x ≤ 1 - x) ∧                             -- Property 1
  (b - a < 1 - 2 * n)                                    -- Property 2
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l2542_254229
