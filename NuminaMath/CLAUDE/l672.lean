import Mathlib

namespace NUMINAMATH_CALUDE_average_and_difference_l672_67213

theorem average_and_difference (y : ℝ) : 
  (46 + y) / 2 = 52 → |y - 46| = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l672_67213


namespace NUMINAMATH_CALUDE_joan_gave_63_seashells_l672_67214

/-- The number of seashells Joan gave to Mike -/
def seashells_given_to_mike (initial_seashells : ℕ) (remaining_seashells : ℕ) : ℕ :=
  initial_seashells - remaining_seashells

/-- Theorem: Joan gave 63 seashells to Mike -/
theorem joan_gave_63_seashells :
  seashells_given_to_mike 79 16 = 63 := by
  sorry

end NUMINAMATH_CALUDE_joan_gave_63_seashells_l672_67214


namespace NUMINAMATH_CALUDE_jaspers_refreshments_l672_67299

theorem jaspers_refreshments (chips drinks : ℕ) (h1 : chips = 27) (h2 : drinks = 31) :
  let hot_dogs := drinks - 12
  chips - hot_dogs = 8 := by
  sorry

end NUMINAMATH_CALUDE_jaspers_refreshments_l672_67299


namespace NUMINAMATH_CALUDE_even_digits_in_base7_of_315_l672_67243

/-- Converts a natural number from base 10 to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of digits --/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a digit is even in base 7 --/
def isEvenInBase7 (digit : ℕ) : Bool :=
  sorry

theorem even_digits_in_base7_of_315 :
  let base7Repr := toBase7 315
  countEvenDigits (base7Repr.filter isEvenInBase7) = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_in_base7_of_315_l672_67243


namespace NUMINAMATH_CALUDE_exists_common_element_l672_67220

/-- A collection of 1978 sets, each containing 40 elements -/
def SetCollection := Fin 1978 → Finset (Fin (1978 * 40))

/-- The property that any two sets in the collection have exactly one common element -/
def OneCommonElement (collection : SetCollection) : Prop :=
  ∀ i j, i ≠ j → (collection i ∩ collection j).card = 1

/-- The theorem stating that there exists an element in all sets of the collection -/
theorem exists_common_element (collection : SetCollection)
  (h1 : ∀ i, (collection i).card = 40)
  (h2 : OneCommonElement collection) :
  ∃ x, ∀ i, x ∈ collection i :=
sorry

end NUMINAMATH_CALUDE_exists_common_element_l672_67220


namespace NUMINAMATH_CALUDE_matthews_crackers_l672_67273

/-- The number of friends Matthew has -/
def num_friends : ℕ := 4

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := 8

/-- The number of cakes each person ate -/
def cakes_eaten_per_person : ℕ := 2

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 8

theorem matthews_crackers :
  initial_crackers = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_matthews_crackers_l672_67273


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l672_67247

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term
  (d : ℚ)
  (h1 : d = 5)
  (h2 : ∃ (c : ℚ), ∀ (n : ℕ), n > 0 → arithmetic_sum a d (4 * n) / arithmetic_sum a d n = c) :
  a = 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l672_67247


namespace NUMINAMATH_CALUDE_all_natural_numbers_have_P_structure_l672_67284

/-- The set of all squares of positive integers -/
def P : Set ℕ := {n : ℕ | ∃ k : ℕ+, n = k^2}

/-- A number n has a P structure if it can be expressed as a sum of some distinct elements from P -/
def has_P_structure (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ), (∀ s ∈ S, s ∈ P) ∧ (S.sum id = n)

/-- Every natural number has a P structure -/
theorem all_natural_numbers_have_P_structure :
  ∀ n : ℕ, has_P_structure n :=
sorry

end NUMINAMATH_CALUDE_all_natural_numbers_have_P_structure_l672_67284


namespace NUMINAMATH_CALUDE_product_of_points_on_line_l672_67218

/-- A line in the coordinate plane passing through the origin with slope 1/2 -/
def line_k (x y : ℝ) : Prop := y = (1/2) * x

theorem product_of_points_on_line :
  ∀ x y : ℝ,
  line_k x 6 →
  line_k 10 y →
  x * y = 60 := by
sorry

end NUMINAMATH_CALUDE_product_of_points_on_line_l672_67218


namespace NUMINAMATH_CALUDE_cuboid_area_example_l672_67241

/-- The surface area of a cuboid with given dimensions -/
def cuboid_surface_area (length breadth height : ℝ) : ℝ :=
  2 * (length * breadth + breadth * height + length * height)

/-- Theorem: The surface area of a cuboid with length 8 cm, breadth 6 cm, and height 9 cm is 348 cm² -/
theorem cuboid_area_example : cuboid_surface_area 8 6 9 = 348 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_area_example_l672_67241


namespace NUMINAMATH_CALUDE_packing_peanuts_calculation_l672_67262

/-- The amount of packing peanuts (in grams) needed for each large order -/
def large_order_peanuts : ℕ := sorry

/-- The total amount of packing peanuts (in grams) used -/
def total_peanuts : ℕ := 800

/-- The number of large orders -/
def num_large_orders : ℕ := 3

/-- The number of small orders -/
def num_small_orders : ℕ := 4

/-- The amount of packing peanuts (in grams) needed for each small order -/
def small_order_peanuts : ℕ := 50

theorem packing_peanuts_calculation :
  large_order_peanuts * num_large_orders + small_order_peanuts * num_small_orders = total_peanuts ∧
  large_order_peanuts = 200 := by sorry

end NUMINAMATH_CALUDE_packing_peanuts_calculation_l672_67262


namespace NUMINAMATH_CALUDE_linear_quadratic_intersection_l672_67255

-- Define the functions f and g
def f (k b x : ℝ) : ℝ := k * x + b
def g (x : ℝ) : ℝ := x^2 - x - 6

-- State the theorem
theorem linear_quadratic_intersection (k b : ℝ) :
  (∃ A B : ℝ × ℝ, 
    f k b A.1 = 0 ∧ 
    f k b 0 = B.2 ∧ 
    B.1 - A.1 = 2 ∧ 
    B.2 - A.2 = 2) →
  (k = 1 ∧ b = 2) ∧
  (∀ x : ℝ, f k b x > g x → (g x + 1) / (f k b x) ≥ -3) ∧
  (∃ x : ℝ, f k b x > g x ∧ (g x + 1) / (f k b x) = -3) :=
by sorry

end NUMINAMATH_CALUDE_linear_quadratic_intersection_l672_67255


namespace NUMINAMATH_CALUDE_derivative_y_l672_67288

noncomputable def y (x : ℝ) : ℝ := -1/2 * Real.log (Real.tanh (x/2)) - Real.cosh x / (2 * Real.sinh x ^ 2)

theorem derivative_y (x : ℝ) : 
  deriv y x = 1 / Real.sinh x ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_derivative_y_l672_67288


namespace NUMINAMATH_CALUDE_function_equivalence_l672_67265

theorem function_equivalence
  (f g h : ℕ → ℕ)
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 :=
by sorry

end NUMINAMATH_CALUDE_function_equivalence_l672_67265


namespace NUMINAMATH_CALUDE_cone_height_calculation_l672_67206

theorem cone_height_calculation (cylinder_base_area cone_base_area cylinder_height : ℝ) 
  (h1 : cylinder_base_area * cylinder_height = (1/3) * cone_base_area * cone_height)
  (h2 : cylinder_base_area / cone_base_area = 3/5)
  (h3 : cylinder_height = 8) : 
  cone_height = 14.4 := by
  sorry

#check cone_height_calculation

end NUMINAMATH_CALUDE_cone_height_calculation_l672_67206


namespace NUMINAMATH_CALUDE_final_lives_calculation_l672_67270

def calculate_final_lives (initial_lives lives_lost gain_factor : ℕ) : ℕ :=
  initial_lives - lives_lost + gain_factor * lives_lost

theorem final_lives_calculation (initial_lives lives_lost gain_factor : ℕ) :
  calculate_final_lives initial_lives lives_lost gain_factor =
  initial_lives - lives_lost + gain_factor * lives_lost :=
by
  sorry

-- Example usage
example : calculate_final_lives 75 28 3 = 131 :=
by
  sorry

end NUMINAMATH_CALUDE_final_lives_calculation_l672_67270


namespace NUMINAMATH_CALUDE_linear_equation_solution_l672_67267

theorem linear_equation_solution (a : ℝ) : 
  (∃ (x y : ℝ), a * x - 2 * y = 2 ∧ x = 4 ∧ y = 5) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l672_67267


namespace NUMINAMATH_CALUDE_reading_time_difference_l672_67250

/-- Proves that the difference in reading time between two readers is 144 minutes given their reading rates and book length. -/
theorem reading_time_difference 
  (xanthia_rate : ℝ) 
  (molly_rate : ℝ) 
  (book_pages : ℝ) 
  (h1 : xanthia_rate = 75) 
  (h2 : molly_rate = 45) 
  (h3 : book_pages = 270) : 
  (book_pages / molly_rate - book_pages / xanthia_rate) * 60 = 144 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l672_67250


namespace NUMINAMATH_CALUDE_trajectory_E_equation_max_area_AMBN_l672_67269

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P on circle C
def P (x y : ℝ) : Prop := C x y

-- Define the point H as the foot of the perpendicular from P to x-axis
def H (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define the point Q
def Q (x y : ℝ) : Prop := ∃ (px py : ℝ), P px py ∧ x = (px + (H px).1) / 2 ∧ y = (py + (H px).2) / 2

-- Define the trajectory E
def E (x y : ℝ) : Prop := Q x y

-- Define the line y = kx
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * x ∧ k > 0

-- Theorem for the equation of trajectory E
theorem trajectory_E_equation : ∀ x y : ℝ, E x y ↔ x^2/4 + y^2 = 1 :=
sorry

-- Theorem for the maximum area of quadrilateral AMBN
theorem max_area_AMBN : ∃ (max_area : ℝ), 
  (∀ k x1 y1 x2 y2 : ℝ, E x1 y1 ∧ E x2 y2 ∧ Line k x1 y1 ∧ Line k x2 y2 → 
    abs (x1 * y2 - x2 * y1) ≤ max_area) ∧
  max_area = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_trajectory_E_equation_max_area_AMBN_l672_67269


namespace NUMINAMATH_CALUDE_expression_equality_l672_67230

theorem expression_equality : 
  3 + Real.sqrt 3 + (3 + Real.sqrt 3)⁻¹ + (Real.sqrt 3 - 3)⁻¹ = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l672_67230


namespace NUMINAMATH_CALUDE_rope_length_is_35_l672_67280

/-- The length of the rope in meters -/
def rope_length : ℝ := 35

/-- The time ratio between walking with and against the tractor -/
def time_ratio : ℝ := 7

/-- The equation for walking in the same direction as the tractor -/
def same_direction_equation (x S : ℝ) : Prop :=
  x + time_ratio * S = 140

/-- The equation for walking in the opposite direction of the tractor -/
def opposite_direction_equation (x S : ℝ) : Prop :=
  x - S = 20

theorem rope_length_is_35 :
  ∃ S : ℝ, same_direction_equation rope_length S ∧ opposite_direction_equation rope_length S :=
sorry

end NUMINAMATH_CALUDE_rope_length_is_35_l672_67280


namespace NUMINAMATH_CALUDE_max_subsequent_voters_l672_67248

/-- Represents a movie rating system where:
  * Ratings are integers from 0 to 10
  * At moment T, the rating was an integer
  * After moment T, each subsequent voter decreased the rating by one unit
-/
structure MovieRating where
  initial_rating : ℕ
  initial_voters : ℕ
  subsequent_votes : List ℕ

/-- The rating at any given moment is the sum of all scores divided by their quantity -/
def current_rating (mr : MovieRating) : ℚ :=
  (mr.initial_rating * mr.initial_voters + mr.subsequent_votes.sum) / 
  (mr.initial_voters + mr.subsequent_votes.length)

/-- The condition that the rating decreases by 1 unit after each vote -/
def decreasing_by_one (mr : MovieRating) : Prop :=
  ∀ i, i < mr.subsequent_votes.length →
    current_rating { mr with 
      subsequent_votes := mr.subsequent_votes.take i
    } - current_rating { mr with 
      subsequent_votes := mr.subsequent_votes.take (i + 1)
    } = 1

/-- The main theorem: The maximum number of viewers who could have voted after moment T is 5 -/
theorem max_subsequent_voters (mr : MovieRating) 
    (h1 : mr.initial_rating ∈ Set.range (fun i => i : ℕ → ℕ) ∩ Set.Icc 0 10)
    (h2 : ∀ v ∈ mr.subsequent_votes, v ∈ Set.range (fun i => i : ℕ → ℕ) ∩ Set.Icc 0 10)
    (h3 : decreasing_by_one mr) :
    mr.subsequent_votes.length ≤ 5 :=
  sorry

end NUMINAMATH_CALUDE_max_subsequent_voters_l672_67248


namespace NUMINAMATH_CALUDE_coin_flip_experiment_l672_67227

theorem coin_flip_experiment (total_flips : ℕ) (heads_count : ℕ) (is_fair : Bool) :
  total_flips = 800 →
  heads_count = 440 →
  is_fair = true →
  (heads_count : ℚ) / (total_flips : ℚ) = 11/20 ∧ 
  (1 : ℚ) / 2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_experiment_l672_67227


namespace NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l672_67245

theorem sqrt_t6_plus_t4 (t : ℝ) : Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l672_67245


namespace NUMINAMATH_CALUDE_customer_difference_l672_67205

theorem customer_difference (initial_customers remaining_customers : ℕ) 
  (h1 : initial_customers = 19) 
  (h2 : remaining_customers = 4) : 
  initial_customers - remaining_customers = 15 := by
sorry

end NUMINAMATH_CALUDE_customer_difference_l672_67205


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l672_67244

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - Complex.I) / (1 - 2 * Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l672_67244


namespace NUMINAMATH_CALUDE_ice_cream_sales_for_games_l672_67240

/-- The number of ice creams needed to be sold to buy two games -/
def ice_creams_needed (game_cost : ℕ) (ice_cream_price : ℕ) : ℕ :=
  2 * game_cost / ice_cream_price

/-- Proof that 24 ice creams are needed to buy two $60 games when each ice cream is $5 -/
theorem ice_cream_sales_for_games : ice_creams_needed 60 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_for_games_l672_67240


namespace NUMINAMATH_CALUDE_imma_fraction_is_83_125_l672_67212

/-- Represents the rose distribution problem --/
structure RoseDistribution where
  total_money : ℕ
  rose_price : ℕ
  roses_to_friends : ℕ
  jenna_fraction : ℚ

/-- Calculates the fraction of roses Imma receives --/
def imma_fraction (rd : RoseDistribution) : ℚ :=
  sorry

/-- Theorem stating the fraction of roses Imma receives --/
theorem imma_fraction_is_83_125 (rd : RoseDistribution) 
  (h1 : rd.total_money = 300)
  (h2 : rd.rose_price = 2)
  (h3 : rd.roses_to_friends = 125)
  (h4 : rd.jenna_fraction = 1/3) :
  imma_fraction rd = 83/125 :=
sorry

end NUMINAMATH_CALUDE_imma_fraction_is_83_125_l672_67212


namespace NUMINAMATH_CALUDE_circle_center_l672_67201

/-- The center of the circle described by the equation x² - 6x + y² + 2y = 20 is (3, -1) -/
theorem circle_center (x y : ℝ) : x^2 - 6*x + y^2 + 2*y = 20 → (3, -1) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l672_67201


namespace NUMINAMATH_CALUDE_square_pyramid_volume_l672_67283

/-- The volume of a regular square pyramid with base edge length 1 and height 3 is 1 -/
theorem square_pyramid_volume :
  let base_edge : ℝ := 1
  let height : ℝ := 3
  let base_area : ℝ := base_edge ^ 2
  let volume : ℝ := (1 / 3) * base_area * height
  volume = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_volume_l672_67283


namespace NUMINAMATH_CALUDE_divisors_of_5_pow_30_minus_1_l672_67261

theorem divisors_of_5_pow_30_minus_1 :
  ∀ n : ℕ, 90 < n → n < 100 → (5^30 - 1) % n = 0 ↔ n = 91 ∨ n = 97 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_5_pow_30_minus_1_l672_67261


namespace NUMINAMATH_CALUDE_tan_2_implies_sin_cos_2_5_l672_67254

theorem tan_2_implies_sin_cos_2_5 (x : ℝ) (h : Real.tan x = 2) : 
  Real.sin x * Real.cos x = 2/5 := by
sorry

end NUMINAMATH_CALUDE_tan_2_implies_sin_cos_2_5_l672_67254


namespace NUMINAMATH_CALUDE_inequality_proof_l672_67208

theorem inequality_proof (a b c x : ℝ) :
  (a + c) / 2 - (1 / 2) * Real.sqrt ((a - c)^2 + b^2) ≤ 
  a * (Real.cos x)^2 + b * Real.cos x * Real.sin x + c * (Real.sin x)^2 ∧
  a * (Real.cos x)^2 + b * Real.cos x * Real.sin x + c * (Real.sin x)^2 ≤ 
  (a + c) / 2 + (1 / 2) * Real.sqrt ((a - c)^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l672_67208


namespace NUMINAMATH_CALUDE_correct_calculation_l672_67259

theorem correct_calculation (x : ℝ) : 
  (x / 2 + 45 = 85) → (2 * x - 45 = 115) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l672_67259


namespace NUMINAMATH_CALUDE_intersection_equality_condition_l672_67296

theorem intersection_equality_condition (M N P : Set α) :
  (M = N → M ∩ P = N ∩ P) ∧
  ∃ M N P : Set ℕ, (M ∩ P = N ∩ P) ∧ (M ≠ N) := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_condition_l672_67296


namespace NUMINAMATH_CALUDE_number_of_pumice_rocks_l672_67272

/-- The number of slate rocks -/
def slate_rocks : ℕ := 10

/-- The number of granite rocks -/
def granite_rocks : ℕ := 4

/-- The probability of choosing 2 slate rocks at random without replacement -/
def prob_two_slate : ℚ := 15/100

/-- The number of pumice rocks -/
def pumice_rocks : ℕ := 11

theorem number_of_pumice_rocks :
  (slate_rocks : ℚ) * (slate_rocks - 1) / 
  ((slate_rocks + pumice_rocks + granite_rocks) * (slate_rocks + pumice_rocks + granite_rocks - 1)) = 
  prob_two_slate := by sorry

end NUMINAMATH_CALUDE_number_of_pumice_rocks_l672_67272


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l672_67271

theorem smallest_number_divisibility : ∃! n : ℕ, 
  (∀ m : ℕ, m < n → ¬(∀ d ∈ [12, 16, 18, 21, 28, 35, 45], (m - 4) % d = 0)) ∧
  (∀ d ∈ [12, 16, 18, 21, 28, 35, 45], (n - 4) % d = 0) ∧
  n = 5044 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l672_67271


namespace NUMINAMATH_CALUDE_paint_used_approximation_l672_67285

/-- The amount of paint Joe starts with in gallons -/
def initial_paint : ℝ := 720

/-- The fraction of paint used in the first week -/
def first_week_fraction : ℚ := 2/7

/-- The fraction of remaining paint used in the second week -/
def second_week_fraction : ℚ := 3/8

/-- The fraction of remaining paint used in the third week -/
def third_week_fraction : ℚ := 5/11

/-- The fraction of remaining paint used in the fourth week -/
def fourth_week_fraction : ℚ := 4/13

/-- The total amount of paint used after four weeks -/
def total_paint_used : ℝ :=
  let first_week := initial_paint * (first_week_fraction : ℝ)
  let second_week := (initial_paint - first_week) * (second_week_fraction : ℝ)
  let third_week := (initial_paint - first_week - second_week) * (third_week_fraction : ℝ)
  let fourth_week := (initial_paint - first_week - second_week - third_week) * (fourth_week_fraction : ℝ)
  first_week + second_week + third_week + fourth_week

/-- Theorem stating that the total paint used is approximately 598.620 gallons -/
theorem paint_used_approximation : 
  598.619 < total_paint_used ∧ total_paint_used < 598.621 :=
sorry

end NUMINAMATH_CALUDE_paint_used_approximation_l672_67285


namespace NUMINAMATH_CALUDE_train_speed_calculation_l672_67231

/-- The speed of a train given the lengths of two trains, the speed of one train, and the time they take to cross each other when moving in opposite directions. -/
theorem train_speed_calculation (length1 length2 speed1 time_to_cross : ℝ) :
  length1 = 280 →
  length2 = 220.04 →
  speed1 = 120 →
  time_to_cross = 9 →
  ∃ speed2 : ℝ, 
    (length1 + length2) = (speed1 + speed2) * (5 / 18) * time_to_cross ∧ 
    abs (speed2 - 80.016) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l672_67231


namespace NUMINAMATH_CALUDE_divisibility_by_24_l672_67278

theorem divisibility_by_24 (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) : 
  24 ∣ p^2 - 1 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l672_67278


namespace NUMINAMATH_CALUDE_anna_ate_14_apples_l672_67211

def apples_tuesday : ℕ := 4

def apples_wednesday (tuesday : ℕ) : ℕ := 2 * tuesday

def apples_thursday (tuesday : ℕ) : ℕ := tuesday / 2

def total_apples (tuesday wednesday thursday : ℕ) : ℕ := 
  tuesday + wednesday + thursday

theorem anna_ate_14_apples : 
  total_apples apples_tuesday 
               (apples_wednesday apples_tuesday) 
               (apples_thursday apples_tuesday) = 14 := by
  sorry

end NUMINAMATH_CALUDE_anna_ate_14_apples_l672_67211


namespace NUMINAMATH_CALUDE_zoo_with_only_hippos_possible_l672_67226

-- Define the universe of zoos
variable (Z : Type)

-- Define the subsets of zoos with hippos, rhinos, and giraffes
variable (H R G : Set Z)

-- Define the conditions
axiom condition1 : H ∩ R ⊆ Gᶜ
axiom condition2 : R ∩ Gᶜ ⊆ H
axiom condition3 : H ∩ G ⊆ R

-- Theorem to prove
theorem zoo_with_only_hippos_possible :
  ∃ (z : Z), z ∈ H ∧ z ∉ G ∧ z ∉ R :=
sorry

end NUMINAMATH_CALUDE_zoo_with_only_hippos_possible_l672_67226


namespace NUMINAMATH_CALUDE_complex_roots_magnitude_l672_67252

theorem complex_roots_magnitude (p : ℝ) (x₁ x₂ : ℂ) : 
  (∀ x : ℂ, x^2 + p*x + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  Complex.abs x₁ = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_magnitude_l672_67252


namespace NUMINAMATH_CALUDE_f_iterated_four_times_l672_67268

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^2 else z^2

theorem f_iterated_four_times :
  f (f (f (f (1 + 2*I)))) = 165633 - 112896*I := by sorry

end NUMINAMATH_CALUDE_f_iterated_four_times_l672_67268


namespace NUMINAMATH_CALUDE_triangle_angle_f_l672_67295

theorem triangle_angle_f (D E F : Real) : 
  0 < D ∧ 0 < E ∧ 0 < F ∧ D + E + F = Real.pi →
  5 * Real.sin D + 2 * Real.cos E = 8 →
  3 * Real.sin E + 5 * Real.cos D = 2 →
  Real.sin F = 43 / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_f_l672_67295


namespace NUMINAMATH_CALUDE_x_equals_seven_l672_67225

theorem x_equals_seven (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 7 * x^2 + 14 * x * y = x^3 + 3 * x^2 * y) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_seven_l672_67225


namespace NUMINAMATH_CALUDE_initial_money_theorem_l672_67202

def meat_cost : ℕ := 17
def chicken_cost : ℕ := 22
def veggies_cost : ℕ := 43
def eggs_cost : ℕ := 5
def dog_food_cost : ℕ := 45
def cat_food_cost : ℕ := 18
def money_left : ℕ := 35

def total_spent : ℕ := meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost + cat_food_cost

theorem initial_money_theorem : 
  meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost + cat_food_cost + money_left = 185 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_theorem_l672_67202


namespace NUMINAMATH_CALUDE_probability_select_A_l672_67289

/-- The probability of selecting a specific person when choosing 2 from 5 -/
def probability_select_person (total : ℕ) (choose : ℕ) : ℚ :=
  (total - 1).choose (choose - 1) / total.choose choose

/-- The group size -/
def group_size : ℕ := 5

/-- The number of people to choose -/
def choose_size : ℕ := 2

theorem probability_select_A :
  probability_select_person group_size choose_size = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_select_A_l672_67289


namespace NUMINAMATH_CALUDE_fast_food_constant_and_variables_l672_67276

/-- A linear pricing model for fast food boxes -/
structure FastFoodPricing where
  cost_per_box : ℝ  -- Cost per box in yuan
  num_boxes : ℝ     -- Number of boxes purchased
  total_cost : ℝ    -- Total cost in yuan
  pricing_model : total_cost = cost_per_box * num_boxes

/-- Theorem stating that in a FastFoodPricing model, the constant is the cost per box,
    and the variables are the number of boxes and the total cost -/
theorem fast_food_constant_and_variables (model : FastFoodPricing) :
  (∃ (k : ℝ), k = model.cost_per_box ∧ k ≠ 0) ∧
  (∀ (n s : ℝ), n = model.num_boxes ∧ s = model.total_cost →
    s = model.cost_per_box * n) :=
sorry

end NUMINAMATH_CALUDE_fast_food_constant_and_variables_l672_67276


namespace NUMINAMATH_CALUDE_lucas_change_l672_67249

def banana_cost : ℚ := 70 / 100
def orange_cost : ℚ := 80 / 100
def banana_quantity : ℕ := 5
def orange_quantity : ℕ := 2
def paid_amount : ℚ := 10

def total_cost : ℚ := banana_cost * banana_quantity + orange_cost * orange_quantity

theorem lucas_change :
  paid_amount - total_cost = 490 / 100 := by sorry

end NUMINAMATH_CALUDE_lucas_change_l672_67249


namespace NUMINAMATH_CALUDE_probability_at_least_one_switch_closed_l672_67216

theorem probability_at_least_one_switch_closed 
  (p : ℝ) 
  (h1 : 0 < p) 
  (h2 : p < 1) :
  let prob_at_least_one_closed := 4*p - 6*p^2 + 4*p^3 - p^4
  prob_at_least_one_closed = 1 - (1 - p)^4 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_switch_closed_l672_67216


namespace NUMINAMATH_CALUDE_count_valid_pairs_l672_67260

/-- The number of ordered pairs (m, n) of positive integers satisfying m ≥ n and m² - n² = 120 -/
def count_pairs : ℕ := 4

/-- Predicate for valid pairs -/
def is_valid_pair (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 120

theorem count_valid_pairs :
  (∃! (s : Finset (ℕ × ℕ)), s.card = count_pairs ∧ 
    ∀ p : ℕ × ℕ, p ∈ s ↔ is_valid_pair p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l672_67260


namespace NUMINAMATH_CALUDE_average_of_geometric_sequence_l672_67258

/-- The average of the numbers 5y, 10y, 20y, 40y, and 80y is equal to 31y -/
theorem average_of_geometric_sequence (y : ℝ) : 
  (5*y + 10*y + 20*y + 40*y + 80*y) / 5 = 31*y := by
  sorry

end NUMINAMATH_CALUDE_average_of_geometric_sequence_l672_67258


namespace NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l672_67204

/-- Given a quadratic function f(x) = x^2 + ax + b - 3 that passes through the point (2,0),
    the minimum value of a^2 + b^2 is 1. -/
theorem min_value_a_squared_plus_b_squared (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b - 3 = 0 → x = 2) → 
  (∃ m : ℝ, ∀ a' b' : ℝ, (∀ x : ℝ, x^2 + a'*x + b' - 3 = 0 → x = 2) → a'^2 + b'^2 ≥ m) ∧
  (a^2 + b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l672_67204


namespace NUMINAMATH_CALUDE_cd_length_problem_l672_67238

theorem cd_length_problem (x : ℝ) : x > 0 ∧ x + x + 2*x = 6 → x = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_cd_length_problem_l672_67238


namespace NUMINAMATH_CALUDE_student_average_age_l672_67275

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (total_average : ℕ) 
  (h1 : num_students = 30)
  (h2 : teacher_age = 46)
  (h3 : total_average = 16)
  : (((num_students + 1) * total_average - teacher_age) / num_students : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_average_age_l672_67275


namespace NUMINAMATH_CALUDE_part_one_part_two_l672_67297

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) / (2 * x - 1) ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a < 0}

-- Part (I)
theorem part_one :
  (A ∩ B 4 = {x | 1/2 < x ∧ x < 2}) ∧
  (A ∪ B 4 = {x | -2 < x ∧ x ≤ 3}) := by sorry

-- Part (II)
theorem part_two :
  (∀ a, B a ∩ (Set.univ \ A) = B a) →
  {a | a ≤ 1/4} = Set.Iic (1/4) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l672_67297


namespace NUMINAMATH_CALUDE_tip_percentage_is_thirty_percent_l672_67221

/-- Calculates the tip percentage given meal costs and total price --/
def calculate_tip_percentage (appetizer_cost : ℚ) (entree_cost : ℚ) (num_entrees : ℕ) (dessert_cost : ℚ) (total_price : ℚ) : ℚ :=
  let meal_cost := appetizer_cost + entree_cost * num_entrees + dessert_cost
  let tip_amount := total_price - meal_cost
  (tip_amount / meal_cost) * 100

/-- Proves that the tip percentage is 30% given the specific meal costs --/
theorem tip_percentage_is_thirty_percent :
  calculate_tip_percentage 9 20 2 11 78 = 30 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_is_thirty_percent_l672_67221


namespace NUMINAMATH_CALUDE_value_of_a_l672_67263

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x + a*x^3

theorem value_of_a : 
  ∀ a : ℝ, (deriv (f a)) 1 = 5 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l672_67263


namespace NUMINAMATH_CALUDE_triangle_with_perimeter_12_has_area_6_l672_67223

-- Define a triangle with integral sides
def Triangle := (ℕ × ℕ × ℕ)

-- Function to calculate perimeter of a triangle
def perimeter (t : Triangle) : ℕ :=
  let (a, b, c) := t
  a + b + c

-- Function to check if three sides form a valid triangle
def is_valid_triangle (t : Triangle) : Prop :=
  let (a, b, c) := t
  a + b > c ∧ b + c > a ∧ c + a > b

-- Function to calculate the area of a triangle using Heron's formula
noncomputable def area (t : Triangle) : ℝ :=
  let (a, b, c) := t
  let s : ℝ := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem triangle_with_perimeter_12_has_area_6 :
  ∃ (t : Triangle), perimeter t = 12 ∧ is_valid_triangle t ∧ area t = 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_with_perimeter_12_has_area_6_l672_67223


namespace NUMINAMATH_CALUDE_glow_interval_l672_67286

/-- The time interval between glows of a light, given the total time period and number of glows. -/
theorem glow_interval (total_time : ℕ) (num_glows : ℝ) 
  (h1 : total_time = 4969)
  (h2 : num_glows = 382.2307692307692) :
  ∃ (interval : ℝ), abs (interval - 13) < 0.0000001 ∧ interval = total_time / num_glows :=
sorry

end NUMINAMATH_CALUDE_glow_interval_l672_67286


namespace NUMINAMATH_CALUDE_rods_in_mile_l672_67222

/-- Represents the number of furlongs in a mile -/
def furlongs_per_mile : ℕ := 8

/-- Represents the number of rods in a furlong -/
def rods_per_furlong : ℕ := 40

/-- Theorem stating that one mile is equal to 320 rods -/
theorem rods_in_mile : furlongs_per_mile * rods_per_furlong = 320 := by
  sorry

end NUMINAMATH_CALUDE_rods_in_mile_l672_67222


namespace NUMINAMATH_CALUDE_circle_x_axis_intersection_sum_l672_67279

theorem circle_x_axis_intersection_sum (c : ℝ × ℝ) (r : ℝ) : 
  c = (3, -4) → r = 7 → 
  ∃ x₁ x₂ : ℝ, 
    ((x₁ - 3)^2 + 4^2 = r^2) ∧
    ((x₂ - 3)^2 + 4^2 = r^2) ∧
    x₁ + x₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_circle_x_axis_intersection_sum_l672_67279


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l672_67237

/-- Given a geometric sequence with first term 2 and fifth term 18, the third term is 6 -/
theorem geometric_sequence_third_term :
  ∀ (x y z : ℝ), 
  (∃ q : ℝ, q ≠ 0 ∧ x = 2 * q ∧ y = 2 * q^2 ∧ z = 2 * q^3 ∧ 18 = 2 * q^4) →
  y = 6 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l672_67237


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l672_67219

theorem sqrt_equation_solution :
  let f : ℝ → ℝ := λ x => Real.sqrt (x + 9) - 2 * Real.sqrt (x - 2) + 3
  ∃ x₁ x₂ : ℝ, x₁ = 8 + 4 * Real.sqrt 2 ∧ x₂ = 8 - 4 * Real.sqrt 2 ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l672_67219


namespace NUMINAMATH_CALUDE_two_men_absent_l672_67264

/-- Represents the work completion scenario -/
structure WorkCompletion where
  total_men : ℕ
  planned_days : ℕ
  actual_days : ℕ

/-- Calculates the number of absent men given the work completion scenario -/
def calculate_absent_men (w : WorkCompletion) : ℕ :=
  w.total_men - (w.total_men * w.planned_days) / w.actual_days

/-- Theorem stating that 2 men became absent in the given scenario -/
theorem two_men_absent (w : WorkCompletion) 
  (h1 : w.total_men = 22)
  (h2 : w.planned_days = 20)
  (h3 : w.actual_days = 22) : 
  calculate_absent_men w = 2 := by
  sorry

#eval calculate_absent_men ⟨22, 20, 22⟩

end NUMINAMATH_CALUDE_two_men_absent_l672_67264


namespace NUMINAMATH_CALUDE_angle_properties_l672_67274

theorem angle_properties (a θ : ℝ) (h : a > 0) 
  (h_point : ∃ (x y : ℝ), x = 3 * a ∧ y = 4 * a ∧ (Real.cos θ = x / Real.sqrt (x^2 + y^2)) ∧ (Real.sin θ = y / Real.sqrt (x^2 + y^2))) :
  (Real.sin θ = 4/5) ∧ 
  (Real.sin (3 * Real.pi / 2 - θ) + Real.cos (θ - Real.pi) = -6/5) := by
  sorry


end NUMINAMATH_CALUDE_angle_properties_l672_67274


namespace NUMINAMATH_CALUDE_inequality_system_solution_l672_67277

theorem inequality_system_solution (a : ℝ) :
  (∃ x : ℝ, x + a ≥ 0 ∧ 1 - 2*x > x - 2) ↔ a > -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l672_67277


namespace NUMINAMATH_CALUDE_car_lot_problem_l672_67266

theorem car_lot_problem (total : ℕ) (power_steering : ℕ) (power_windows : ℕ) (neither : ℕ) :
  total = 65 →
  power_steering = 45 →
  power_windows = 25 →
  neither = 12 →
  ∃ both : ℕ, both = 17 ∧
    total = power_steering + power_windows - both + neither :=
by sorry

end NUMINAMATH_CALUDE_car_lot_problem_l672_67266


namespace NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l672_67200

/-- Given a square with side length 20 and two congruent equilateral triangles drawn inside,
    where each triangle has one vertex on a vertex of the square and they touch different
    pairs of adjacent vertices, this theorem states the side length of the largest square
    that can be inscribed in the space inside the square and outside the triangles. -/
theorem largest_inscribed_square_side_length :
  ∃ (outer_square_side_length : ℝ) (triangle_side_length : ℝ) (inscribed_square_side_length : ℝ),
    outer_square_side_length = 20 ∧
    triangle_side_length = 10 * (Real.sqrt 3 - 1) ∧
    inscribed_square_side_length = 10 - (5 * Real.sqrt 6 - 5 * Real.sqrt 2) / 2 ∧
    inscribed_square_side_length = 
      (outer_square_side_length * Real.sqrt 2 - triangle_side_length) / (2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l672_67200


namespace NUMINAMATH_CALUDE_clown_balloons_theorem_l672_67215

def balloons_problem (initial_dozens : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  initial_dozens * 12 - (boys + girls)

theorem clown_balloons_theorem :
  balloons_problem 3 3 12 = 21 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_theorem_l672_67215


namespace NUMINAMATH_CALUDE_simplify_polynomial_l672_67228

theorem simplify_polynomial (y : ℝ) :
  (2*y - 1) * (4*y^10 + 2*y^9 + 4*y^8 + 2*y^7) = 8*y^11 + 6*y^9 - 2*y^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l672_67228


namespace NUMINAMATH_CALUDE_jack_weight_jack_weight_proof_l672_67253

/-- Jack and Anna's see-saw problem -/
theorem jack_weight (anna_weight : ℕ) (num_rocks : ℕ) (rock_weight : ℕ) : ℕ :=
  let total_rock_weight := num_rocks * rock_weight
  let jack_weight := anna_weight - total_rock_weight
  jack_weight

/-- Proof of Jack's weight -/
theorem jack_weight_proof :
  jack_weight 40 5 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jack_weight_jack_weight_proof_l672_67253


namespace NUMINAMATH_CALUDE_probability_one_boy_one_girl_l672_67290

/-- The probability of selecting one boy and one girl from a group of 3 boys and 2 girls, when choosing 2 students out of 5 -/
theorem probability_one_boy_one_girl :
  let total_students : ℕ := 5
  let num_boys : ℕ := 3
  let num_girls : ℕ := 2
  let students_to_select : ℕ := 2
  let total_combinations := Nat.choose total_students students_to_select
  let favorable_outcomes := Nat.choose num_boys 1 * Nat.choose num_girls 1
  (favorable_outcomes : ℚ) / total_combinations = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_boy_one_girl_l672_67290


namespace NUMINAMATH_CALUDE_hyperbola_equation_l672_67233

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  ((a - 4)^2 + b^2 = 16) →
  (a^2 + b^2 = 16) →
  (x^2/4 - y^2/12 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l672_67233


namespace NUMINAMATH_CALUDE_total_birds_l672_67203

/-- The number of birds on an oak tree --/
def bird_count (bluebirds cardinals goldfinches sparrows robins swallows : ℕ) : Prop :=
  -- There are twice as many cardinals as bluebirds
  cardinals = 2 * bluebirds ∧
  -- The number of goldfinches is equal to the product of bluebirds and swallows
  goldfinches = bluebirds * swallows ∧
  -- The number of sparrows is half the sum of cardinals and goldfinches
  2 * sparrows = cardinals + goldfinches ∧
  -- The number of robins is 2 less than the quotient of bluebirds divided by swallows
  robins + 2 = bluebirds / swallows ∧
  -- There are 12 swallows
  swallows = 12 ∧
  -- The number of swallows is half as many as the number of bluebirds
  2 * swallows = bluebirds

theorem total_birds (bluebirds cardinals goldfinches sparrows robins swallows : ℕ) :
  bird_count bluebirds cardinals goldfinches sparrows robins swallows →
  bluebirds + cardinals + goldfinches + sparrows + robins + swallows = 540 :=
by sorry

end NUMINAMATH_CALUDE_total_birds_l672_67203


namespace NUMINAMATH_CALUDE_smallest_k_for_two_trailing_zeros_l672_67287

theorem smallest_k_for_two_trailing_zeros : ∃ k : ℕ+, k = 13 ∧ 
  (∀ m : ℕ+, m < k → ¬(100 ∣ Nat.choose (2 * m) m)) ∧ 
  (100 ∣ Nat.choose (2 * k) k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_two_trailing_zeros_l672_67287


namespace NUMINAMATH_CALUDE_males_band_not_orchestra_zero_l672_67292

/-- Represents the membership of students in band and orchestra --/
structure MusicGroups where
  total : ℕ
  females_band : ℕ
  males_band : ℕ
  females_orchestra : ℕ
  males_orchestra : ℕ
  females_both : ℕ

/-- The number of males in the band who are not in the orchestra is 0 --/
theorem males_band_not_orchestra_zero (g : MusicGroups)
  (h1 : g.total = 250)
  (h2 : g.females_band = 120)
  (h3 : g.males_band = 90)
  (h4 : g.females_orchestra = 90)
  (h5 : g.males_orchestra = 120)
  (h6 : g.females_both = 70) :
  g.males_band - (g.males_band + g.males_orchestra - (g.total - (g.females_band + g.females_orchestra - g.females_both))) = 0 := by
  sorry

#check males_band_not_orchestra_zero

end NUMINAMATH_CALUDE_males_band_not_orchestra_zero_l672_67292


namespace NUMINAMATH_CALUDE_parallel_planes_line_parallel_parallel_planes_perpendicular_line_not_always_perpendicular_planes_perpendicular_line_parallel_not_always_perpendicular_planes_line_perpendicular_l672_67235

-- Define the basic types
variable (P : Type) -- Type for planes
variable (L : Type) -- Type for lines

-- Define the relations
variable (parallel_planes : P → P → Prop)
variable (perpendicular_planes : P → P → Prop)
variable (line_in_plane : L → P → Prop)
variable (line_parallel_to_plane : L → P → Prop)
variable (line_perpendicular_to_plane : L → P → Prop)

-- State the theorems
theorem parallel_planes_line_parallel
  (p1 p2 : P) (l : L)
  (h1 : parallel_planes p1 p2)
  (h2 : line_in_plane l p1) :
  line_parallel_to_plane l p2 :=
sorry

theorem parallel_planes_perpendicular_line
  (p1 p2 : P) (l : L)
  (h1 : parallel_planes p1 p2)
  (h2 : line_perpendicular_to_plane l p1) :
  line_perpendicular_to_plane l p2 :=
sorry

theorem not_always_perpendicular_planes_perpendicular_line_parallel
  (p1 p2 : P) (l : L) :
  ¬ (∀ (h1 : perpendicular_planes p1 p2)
      (h2 : line_perpendicular_to_plane l p1),
      line_parallel_to_plane l p2) :=
sorry

theorem not_always_perpendicular_planes_line_perpendicular
  (p1 p2 : P) (l : L) :
  ¬ (∀ (h1 : perpendicular_planes p1 p2)
      (h2 : line_in_plane l p1),
      line_perpendicular_to_plane l p2) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_line_parallel_parallel_planes_perpendicular_line_not_always_perpendicular_planes_perpendicular_line_parallel_not_always_perpendicular_planes_line_perpendicular_l672_67235


namespace NUMINAMATH_CALUDE_shortest_distance_between_tangents_l672_67209

-- Define the parabola C₁
def C₁ (x y : ℝ) : Prop := x^2 = 4*y

-- Define the point P on C₁
def P : ℝ × ℝ := (2, 1)

-- Define the point Q
def Q : ℝ × ℝ := (0, 2)

-- Define the line l (implicitly by Q and its intersection with C₁)
def l (x y : ℝ) : Prop := ∃ (k : ℝ), y - Q.2 = k * (x - Q.1)

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := x^2 = 2*y - 4

-- Define the tangent lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := ∃ (x₃ y₃ : ℝ), C₁ x₃ y₃ ∧ 2*x*x₃ - 2*y - 2*x₃^2 = 0

def l₂ (x y : ℝ) : Prop := ∃ (x₄ y₄ : ℝ), C₂ x₄ y₄ ∧ 2*x*x₄ - 2*y - x₄^2 + 4 = 0

-- The theorem to prove
theorem shortest_distance_between_tangents :
  ∀ (x₃ : ℝ), l₁ x₃ (x₃^2/4) → l₂ (x₃/2) ((x₃/2)^2/2 + 2) →
  (x₃^2 + 4) / (2 * Real.sqrt (x₃^2 + 1)) ≥ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_between_tangents_l672_67209


namespace NUMINAMATH_CALUDE_absolute_value_sum_l672_67242

theorem absolute_value_sum : |(-8 : ℤ)| + |(-4 : ℤ)| = 12 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l672_67242


namespace NUMINAMATH_CALUDE_chickpea_flour_amount_l672_67282

def rye_flour : ℕ := 5
def whole_wheat_bread_flour : ℕ := 10
def whole_wheat_pastry_flour : ℕ := 2
def total_flour : ℕ := 20

theorem chickpea_flour_amount :
  total_flour - (rye_flour + whole_wheat_bread_flour + whole_wheat_pastry_flour) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chickpea_flour_amount_l672_67282


namespace NUMINAMATH_CALUDE_four_male_workers_selected_l672_67256

/-- Represents the number of male workers selected in a stratified sampling -/
def male_workers_selected (total_workers female_workers selected_workers : ℕ) : ℕ :=
  (total_workers - female_workers) * selected_workers / total_workers

/-- Theorem stating that 4 male workers are selected in the given scenario -/
theorem four_male_workers_selected :
  male_workers_selected 30 10 6 = 4 := by
  sorry

#eval male_workers_selected 30 10 6

end NUMINAMATH_CALUDE_four_male_workers_selected_l672_67256


namespace NUMINAMATH_CALUDE_unpainted_area_triangular_board_l672_67298

/-- The area of the unpainted region on a triangular board that intersects with a rectangular board -/
theorem unpainted_area_triangular_board (base height width intersection_angle : ℝ) 
  (h_base : base = 8)
  (h_height : height = 10)
  (h_width : width = 5)
  (h_angle : intersection_angle = 45) :
  base * height / 2 - width * height = 50 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_area_triangular_board_l672_67298


namespace NUMINAMATH_CALUDE_logarithm_sum_l672_67224

theorem logarithm_sum (a : ℝ) (h : 1 + a^3 = 9) : 
  Real.log a / Real.log (1/4) + Real.log 8 / Real.log a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_l672_67224


namespace NUMINAMATH_CALUDE_ahmed_hassan_tree_difference_l672_67293

/-- Represents the number of trees in an orchard -/
structure Orchard :=
  (apple : ℕ)
  (orange : ℕ)

/-- Calculate the total number of trees in an orchard -/
def totalTrees (o : Orchard) : ℕ := o.apple + o.orange

/-- The difference in the number of trees between two orchards -/
def treeDifference (o1 o2 : Orchard) : ℕ := (totalTrees o1) - (totalTrees o2)

theorem ahmed_hassan_tree_difference :
  let ahmed : Orchard := { apple := 4, orange := 8 }
  let hassan : Orchard := { apple := 1, orange := 2 }
  treeDifference ahmed hassan = 9 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_hassan_tree_difference_l672_67293


namespace NUMINAMATH_CALUDE_triangle_height_l672_67291

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 10 → area = 25 → area = (base * height) / 2 → height = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l672_67291


namespace NUMINAMATH_CALUDE_sugar_problem_l672_67234

/-- Calculates the remaining sugar after a bag is torn -/
def remaining_sugar (initial_sugar : ℕ) (num_bags : ℕ) (torn_bags : ℕ) : ℕ :=
  let sugar_per_bag := initial_sugar / num_bags
  let intact_sugar := sugar_per_bag * (num_bags - torn_bags)
  let torn_bag_sugar := sugar_per_bag / 2
  intact_sugar + (torn_bag_sugar * torn_bags)

/-- Theorem stating that 21 kilos of sugar remain after one bag is torn -/
theorem sugar_problem :
  remaining_sugar 24 4 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sugar_problem_l672_67234


namespace NUMINAMATH_CALUDE_perpendicular_slope_l672_67246

/-- The slope of a line perpendicular to a line passing through points (3, -4) and (-6, 2) is 3/2 -/
theorem perpendicular_slope : 
  let p₁ : ℝ × ℝ := (3, -4)
  let p₂ : ℝ × ℝ := (-6, 2)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  (- (1 / m)) = (3 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l672_67246


namespace NUMINAMATH_CALUDE_library_problem_l672_67217

/-- Represents the number of students that can be helped on the fourth day given the initial number of books and the number of students helped in the first three days. -/
def students_helped_fourth_day (total_books : ℕ) (books_per_student : ℕ) (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  (total_books - (day1 + day2 + day3) * books_per_student) / books_per_student

/-- Theorem stating that given the specific conditions of the library problem, 9 students can be helped on the fourth day. -/
theorem library_problem :
  students_helped_fourth_day 120 5 4 5 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_library_problem_l672_67217


namespace NUMINAMATH_CALUDE_mrs_hilt_money_l672_67251

/-- Mrs. Hilt's pencil purchase problem -/
theorem mrs_hilt_money (pencil_cost remaining_money : ℕ) 
  (h1 : pencil_cost = 11)
  (h2 : remaining_money = 4) : 
  pencil_cost + remaining_money = 15 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_money_l672_67251


namespace NUMINAMATH_CALUDE_peter_marbles_l672_67257

/-- The number of marbles Peter lost -/
def lost_marbles : ℕ := 15

/-- The number of marbles Peter currently has -/
def current_marbles : ℕ := 18

/-- The initial number of marbles Peter had -/
def initial_marbles : ℕ := lost_marbles + current_marbles

theorem peter_marbles : initial_marbles = 33 := by
  sorry

end NUMINAMATH_CALUDE_peter_marbles_l672_67257


namespace NUMINAMATH_CALUDE_zeros_in_Q_l672_67281

def R (k : ℕ+) : ℕ := (10^k.val - 1) / 9

def Q : ℕ := R 30 / R 6

def count_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 30 := by sorry

end NUMINAMATH_CALUDE_zeros_in_Q_l672_67281


namespace NUMINAMATH_CALUDE_function_properties_l672_67229

/-- Given a function f(x) = a*sin(2x) + cos(2x) where f(π/3) = (√3 - 1)/2,
    prove properties about the value of a, the maximum value of f(x),
    and the intervals where f(x) is monotonically decreasing. -/
theorem function_properties (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * Real.sin (2 * x) + Real.cos (2 * x)) :
  f (π / 3) = (Real.sqrt 3 - 1) / 2 →
  (a = 1 ∧ 
   (∃ M, M = Real.sqrt 2 ∧ ∀ x, f x ≤ M) ∧
   ∀ k : ℤ, ∀ x ∈ Set.Icc (k * π + π / 4) (k * π + 3 * π / 4), 
     ∀ y ∈ Set.Icc (k * π + π / 4) (k * π + 3 * π / 4), 
       x ≤ y → f y ≤ f x) :=
by
  sorry

end NUMINAMATH_CALUDE_function_properties_l672_67229


namespace NUMINAMATH_CALUDE_resulting_angle_25_2_5_turns_l672_67232

/-- Given an initial angle and a number of clockwise turns, calculate the resulting angle -/
def resulting_angle (initial_angle : ℝ) (clockwise_turns : ℝ) : ℝ :=
  initial_angle - 360 * clockwise_turns

/-- Theorem: The resulting angle after rotating 25° clockwise by 2.5 turns is -875° -/
theorem resulting_angle_25_2_5_turns :
  resulting_angle 25 2.5 = -875 := by
  sorry

end NUMINAMATH_CALUDE_resulting_angle_25_2_5_turns_l672_67232


namespace NUMINAMATH_CALUDE_f_is_odd_l672_67239

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Theorem: f is an odd function
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_l672_67239


namespace NUMINAMATH_CALUDE_locus_of_tangent_circles_l672_67294

-- Define the circles C₃ and C₄
def C₃ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₄ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

-- Define the property of being externally tangent to C₃
def externally_tangent_C₃ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 2)^2

-- Define the property of being internally tangent to C₄
def internally_tangent_C₄ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

-- Define the locus equation
def locus_equation (a b : ℝ) : Prop := a^2 + 5*b^2 - 32*a - 51 = 0

-- State the theorem
theorem locus_of_tangent_circles :
  ∀ a b : ℝ,
  (∃ r : ℝ, externally_tangent_C₃ a b r ∧ internally_tangent_C₄ a b r) ↔
  locus_equation a b :=
sorry

end NUMINAMATH_CALUDE_locus_of_tangent_circles_l672_67294


namespace NUMINAMATH_CALUDE_candy_bar_count_l672_67207

theorem candy_bar_count (bags : ℕ) (candy_per_bag : ℕ) (h1 : bags = 5) (h2 : candy_per_bag = 3) :
  bags * candy_per_bag = 15 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_count_l672_67207


namespace NUMINAMATH_CALUDE_is_projection_matrix_l672_67210

def projection_matrix (M : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  M * M = M

theorem is_projection_matrix : 
  let M : Matrix (Fin 2) (Fin 2) ℚ := !![9/34, 25/34; 3/5, 15/34]
  projection_matrix M := by
  sorry

end NUMINAMATH_CALUDE_is_projection_matrix_l672_67210


namespace NUMINAMATH_CALUDE_exists_decreasing_then_increasing_not_exists_increasing_then_decreasing_l672_67236

-- Define the sequence type
def PowerSumSequence (originalNumbers : List ℝ) : ℕ → ℝ :=
  λ n => (originalNumbers.map (λ x => x ^ n)).sum

-- Theorem for part (a)
theorem exists_decreasing_then_increasing :
  ∃ (originalNumbers : List ℝ),
    (∀ x ∈ originalNumbers, x > 0) ∧
    (let a := PowerSumSequence originalNumbers
     a 1 > a 2 ∧ a 2 > a 3 ∧ a 3 > a 4 ∧ a 4 > a 5 ∧
     ∀ n ≥ 5, a n < a (n + 1)) := by
  sorry

-- Theorem for part (b)
theorem not_exists_increasing_then_decreasing :
  ¬ ∃ (originalNumbers : List ℝ),
    (∀ x ∈ originalNumbers, x > 0) ∧
    (let a := PowerSumSequence originalNumbers
     a 1 < a 2 ∧ a 2 < a 3 ∧ a 3 < a 4 ∧ a 4 < a 5 ∧
     ∀ n ≥ 5, a n > a (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_exists_decreasing_then_increasing_not_exists_increasing_then_decreasing_l672_67236
