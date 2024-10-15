import Mathlib

namespace NUMINAMATH_CALUDE_square_ratio_problem_l3056_305691

theorem square_ratio_problem (area_ratio : ℚ) (a b c : ℕ) :
  area_ratio = 48 / 125 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  a = 4 ∧ b = 15 ∧ c = 25 ∧ a + b + c = 44 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_problem_l3056_305691


namespace NUMINAMATH_CALUDE_christine_stickers_l3056_305688

/-- The number of stickers Christine currently has -/
def current_stickers : ℕ := 11

/-- The number of stickers required for a prize -/
def required_stickers : ℕ := 30

/-- The number of additional stickers Christine needs -/
def additional_stickers : ℕ := required_stickers - current_stickers

theorem christine_stickers : additional_stickers = 19 := by
  sorry

end NUMINAMATH_CALUDE_christine_stickers_l3056_305688


namespace NUMINAMATH_CALUDE_pencil_boxes_filled_l3056_305604

/-- Given 648 pencils and 4 pencils per box, prove that the number of filled boxes is 162. -/
theorem pencil_boxes_filled (total_pencils : ℕ) (pencils_per_box : ℕ) (h1 : total_pencils = 648) (h2 : pencils_per_box = 4) :
  total_pencils / pencils_per_box = 162 := by
  sorry

end NUMINAMATH_CALUDE_pencil_boxes_filled_l3056_305604


namespace NUMINAMATH_CALUDE_expression_evaluation_l3056_305603

theorem expression_evaluation : (4 + 5 + 6) / 3 * 2 - 2 / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3056_305603


namespace NUMINAMATH_CALUDE_weight_of_one_bag_is_five_l3056_305690

-- Define the given values
def total_harvest : ℕ := 405
def juice_amount : ℕ := 90
def restaurant_amount : ℕ := 60
def total_revenue : ℕ := 408
def price_per_bag : ℕ := 8

-- Define the weight of one bag as a function of the given values
def weight_of_one_bag : ℚ :=
  (total_harvest - juice_amount - restaurant_amount) / (total_revenue / price_per_bag)

-- Theorem to prove
theorem weight_of_one_bag_is_five :
  weight_of_one_bag = 5 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_one_bag_is_five_l3056_305690


namespace NUMINAMATH_CALUDE_equation_one_l3056_305654

theorem equation_one (x : ℚ) : 3 * (x + 8) - 5 = 6 * (2 * x - 1) → x = 25 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_l3056_305654


namespace NUMINAMATH_CALUDE_prob_two_high_temp_is_half_l3056_305668

/-- Represents a 3-digit number where each digit is either 0-5 or 6-9 -/
def ThreeDayPeriod := Fin 1000

/-- The probability of a digit being 0-5 (representing a high temperature warning) -/
def p_high_temp : ℚ := 3/5

/-- The number of random samples generated -/
def num_samples : ℕ := 20

/-- Counts the number of digits in a ThreeDayPeriod that are 0-5 -/
def count_high_temp (n : ThreeDayPeriod) : ℕ := sorry

/-- The event of exactly 2 high temperature warnings in a 3-day period -/
def two_high_temp (n : ThreeDayPeriod) : Prop := count_high_temp n = 2

/-- The probability of the event two_high_temp -/
def prob_two_high_temp : ℚ := sorry

theorem prob_two_high_temp_is_half : prob_two_high_temp = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_two_high_temp_is_half_l3056_305668


namespace NUMINAMATH_CALUDE_equation_solutions_l3056_305692

def equation (x : ℝ) : Prop :=
  x ≥ 1 ∧ Real.sqrt (x + 5 - 6 * Real.sqrt (x - 1)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 1)) = 3

theorem equation_solutions :
  {x : ℝ | equation x} = {5, 26} :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3056_305692


namespace NUMINAMATH_CALUDE_photo_rectangle_perimeters_l3056_305616

/-- Represents a photograph with length and width -/
structure Photo where
  length : ℝ
  width : ℝ

/-- Represents a rectangle composed of photographs -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The problem statement -/
theorem photo_rectangle_perimeters 
  (photo : Photo)
  (rect1 rect2 rect3 : Rectangle)
  (h1 : 2 * (photo.length + photo.width) = 20)
  (h2 : 2 * (rect2.length + rect2.width) = 56)
  (h3 : rect1.length = 2 * photo.length ∧ rect1.width = 2 * photo.width)
  (h4 : rect2.length = photo.length ∧ rect2.width = 4 * photo.width)
  (h5 : rect3.length = 4 * photo.length ∧ rect3.width = photo.width) :
  2 * (rect1.length + rect1.width) = 40 ∧ 2 * (rect3.length + rect3.width) = 44 := by
  sorry

end NUMINAMATH_CALUDE_photo_rectangle_perimeters_l3056_305616


namespace NUMINAMATH_CALUDE_all_diagonal_triangles_count_l3056_305611

/-- Represents a convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : ℕ
  is_convex : sides = n

/-- Represents a diagonal division of a polygon -/
structure DiagonalDivision (p : ConvexPolygon n) where
  diagonals : ℕ
  triangles : ℕ
  non_intersecting : Bool
  vertex_diagonals : ℕ → ℕ
  valid_division : diagonals = n - 3 ∧ triangles = n - 2
  valid_vertex_diagonals : ∀ v, vertex_diagonals v = 3 ∨ vertex_diagonals v = 0

/-- Counts the number of triangles with all sides as diagonals -/
def count_all_diagonal_triangles (p : ConvexPolygon 102) (d : DiagonalDivision p) : ℕ :=
  sorry

theorem all_diagonal_triangles_count 
  (p : ConvexPolygon 102) 
  (d : DiagonalDivision p) : 
  count_all_diagonal_triangles p d = 34 :=
sorry

end NUMINAMATH_CALUDE_all_diagonal_triangles_count_l3056_305611


namespace NUMINAMATH_CALUDE_product_equality_l3056_305670

theorem product_equality (h : 213 * 16 = 3408) : 16 * 21.3 = 340.8 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3056_305670


namespace NUMINAMATH_CALUDE_max_points_at_distance_l3056_305659

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in 2D space
def Point : Type := ℝ × ℝ

-- Function to check if a point is outside a circle
def isOutside (p : Point) (c : Circle) : Prop :=
  let (px, py) := p
  let (cx, cy) := c.center
  (px - cx)^2 + (py - cy)^2 > c.radius^2

-- Function to count points on circle at fixed distance from a point
def countPointsAtDistance (c : Circle) (p : Point) (d : ℝ) : ℕ :=
  sorry

-- Theorem statement
theorem max_points_at_distance (c : Circle) (p : Point) (d : ℝ) 
  (h : isOutside p c) : 
  countPointsAtDistance c p d ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_points_at_distance_l3056_305659


namespace NUMINAMATH_CALUDE_rod_cutting_l3056_305671

/-- Given a rod of length 42.5 meters that can be cut into 50 equal pieces,
    prove that the length of each piece is 0.85 meters. -/
theorem rod_cutting (rod_length : Real) (num_pieces : Nat) (piece_length : Real) 
    (h1 : rod_length = 42.5)
    (h2 : num_pieces = 50)
    (h3 : piece_length * num_pieces = rod_length) : 
  piece_length = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l3056_305671


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3056_305620

def arithmeticSequenceSum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_ratio :
  let numerator := arithmeticSequenceSum 4 4 52
  let denominator := arithmeticSequenceSum 6 6 78
  numerator / denominator = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3056_305620


namespace NUMINAMATH_CALUDE_polynomial_Q_l3056_305624

/-- Given a polynomial Q(x) = Q(0) + Q(1)x + Q(3)x³ where Q(-1) = 2,
    prove that Q(x) = -2x + (2/9)x³ - 2/9 -/
theorem polynomial_Q (Q : ℝ → ℝ) : 
  (∀ x, Q x = Q 0 + Q 1 * x + Q 3 * x^3) → 
  Q (-1) = 2 → 
  ∀ x, Q x = -2 * x + (2/9) * x^3 - 2/9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_Q_l3056_305624


namespace NUMINAMATH_CALUDE_oil_transfer_height_l3056_305686

/-- Given a cone with base radius 9 cm and height 27 cm, when its volume is transferred to a cylinder with base radius 18 cm, the height of the liquid in the cylinder is 2.25 cm. -/
theorem oil_transfer_height :
  let cone_radius : ℝ := 9
  let cone_height : ℝ := 27
  let cylinder_radius : ℝ := 18
  let cone_volume : ℝ := (1/3) * Real.pi * cone_radius^2 * cone_height
  let cylinder_height : ℝ := cone_volume / (Real.pi * cylinder_radius^2)
  cylinder_height = 2.25
  := by sorry

end NUMINAMATH_CALUDE_oil_transfer_height_l3056_305686


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3056_305637

/-- Given a line L1 with equation 6x - 3y = 9 and a point P (1, -2),
    prove that the line L2 with equation y = 2x - 4 is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) :
  (6 * x - 3 * y = 9) →  -- Equation of L1
  (y = 2 * x - 4) →      -- Equation of L2
  (2 = (6 : ℝ) / 3) →    -- Slopes are equal (parallel condition)
  (2 * 1 - 4 = -2) →     -- L2 passes through (1, -2)
  ∃ (m b : ℝ), y = m * x + b ∧ m = 2 ∧ b = -4 := by
sorry


end NUMINAMATH_CALUDE_parallel_line_through_point_l3056_305637


namespace NUMINAMATH_CALUDE_kids_left_playing_l3056_305644

theorem kids_left_playing (initial_kids : ℝ) (kids_gone_home : ℝ) 
  (h1 : initial_kids = 22.0) 
  (h2 : kids_gone_home = 14.0) : 
  initial_kids - kids_gone_home = 8.0 := by
sorry

end NUMINAMATH_CALUDE_kids_left_playing_l3056_305644


namespace NUMINAMATH_CALUDE_denis_numbers_sum_l3056_305667

theorem denis_numbers_sum : 
  ∀ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d → 
    a * d = 32 → 
    b * c = 14 → 
    a + b + c + d = 42 := by
  sorry

end NUMINAMATH_CALUDE_denis_numbers_sum_l3056_305667


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l3056_305610

theorem sum_of_squares_theorem (x y z t : ℤ) (h : x + y = z + t) :
  x^2 + y^2 + z^2 + t^2 = (x + y)^2 + (x - z)^2 + (x - t)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l3056_305610


namespace NUMINAMATH_CALUDE_arrangement_theorem_l3056_305655

/-- The number of ways to arrange 5 people in 5 seats with exactly 2 matching --/
def arrangement_count : ℕ := 20

/-- The number of ways to choose 2 items from 5 --/
def choose_two_from_five : ℕ := 10

/-- The number of ways to arrange the remaining 3 people --/
def arrange_remaining : ℕ := 2

theorem arrangement_theorem : 
  arrangement_count = choose_two_from_five * arrange_remaining := by
  sorry


end NUMINAMATH_CALUDE_arrangement_theorem_l3056_305655


namespace NUMINAMATH_CALUDE_characterization_of_special_numbers_l3056_305650

/-- Sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The set of numbers that satisfy n = S(n)^2 - S(n) + 1 -/
def specialNumbers : Set ℕ :=
  {n : ℕ | n = (sumOfDigits n)^2 - sumOfDigits n + 1}

/-- Theorem stating that the set of special numbers is exactly {1, 13, 43, 91, 157} -/
theorem characterization_of_special_numbers :
  specialNumbers = {1, 13, 43, 91, 157} := by sorry

end NUMINAMATH_CALUDE_characterization_of_special_numbers_l3056_305650


namespace NUMINAMATH_CALUDE_four_prime_pairs_sum_50_l3056_305661

/-- A function that returns the number of unordered pairs of prime numbers that sum to a given natural number. -/
def count_prime_pairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range (n/2 + 1))).card

/-- Theorem stating that there are exactly 4 unordered pairs of prime numbers that sum to 50. -/
theorem four_prime_pairs_sum_50 : count_prime_pairs 50 = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_prime_pairs_sum_50_l3056_305661


namespace NUMINAMATH_CALUDE_seven_digit_subtraction_l3056_305634

def is_seven_digit (n : ℕ) : Prop := 1000000 ≤ n ∧ n ≤ 9999999

def digit_sum_except_second (n : ℕ) : ℕ :=
  let digits := (Nat.digits 10 n).reverse
  List.sum (digits.take 1 ++ digits.drop 2)

theorem seven_digit_subtraction (n : ℕ) :
  is_seven_digit n →
  ∃ k, n - k = 9875352 →
  n - digit_sum_except_second n = 9875357 :=
sorry

end NUMINAMATH_CALUDE_seven_digit_subtraction_l3056_305634


namespace NUMINAMATH_CALUDE_pencil_sharpening_ishas_pencil_l3056_305600

/-- The length sharpened off a pencil is equal to the difference between
    the original length and the new length after sharpening. -/
theorem pencil_sharpening (original_length new_length : ℝ) :
  original_length ≥ new_length →
  original_length - new_length = original_length - new_length :=
by
  sorry

/-- Isha's pencil problem -/
theorem ishas_pencil :
  let original_length : ℝ := 31
  let new_length : ℝ := 14
  original_length - new_length = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_ishas_pencil_l3056_305600


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3056_305673

theorem unique_solution_equation (b : ℝ) : 
  (b + ⌈b⌉ = 21.6) ∧ (b - ⌊b⌋ = 0.6) → b = 10.6 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3056_305673


namespace NUMINAMATH_CALUDE_coffee_pastry_budget_l3056_305679

theorem coffee_pastry_budget (B : ℝ) (c p : ℝ) 
  (hc : c = (1/4) * (B - p)) 
  (hp : p = (1/10) * (B - c)) : 
  c + p = (4/13) * B := by
sorry

end NUMINAMATH_CALUDE_coffee_pastry_budget_l3056_305679


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l3056_305602

/-- The area of a square with one side on y = 5 and endpoints on y = x^2 + 3x + 2 is 21 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 3*x₁ + 2 = 5) ∧
  (x₂^2 + 3*x₂ + 2 = 5) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 21) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l3056_305602


namespace NUMINAMATH_CALUDE_rajesh_work_time_l3056_305677

/-- The problem of determining Rajesh's work time -/
theorem rajesh_work_time (rahul_rate : ℝ) (rajesh_rate : ℝ → ℝ) (combined_rate : ℝ → ℝ) 
  (total_payment : ℝ) (rahul_share : ℝ) (R : ℝ) :
  rahul_rate = 1/3 →
  (∀ x, rajesh_rate x = 1/x) →
  (∀ x, combined_rate x = (x + 3) / (3*x)) →
  total_payment = 150 →
  rahul_share = 60 →
  R = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_rajesh_work_time_l3056_305677


namespace NUMINAMATH_CALUDE_factory_sampling_is_systematic_l3056_305693

/-- Represents a sampling method --/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a factory's production and inspection process --/
structure ProductionProcess where
  inspectionInterval : ℕ  -- Interval between inspections in minutes
  samplePosition : ℕ      -- Fixed position on the conveyor belt for sampling

/-- Determines the sampling method based on the production process --/
def determineSamplingMethod (process : ProductionProcess) : SamplingMethod :=
  if process.inspectionInterval > 0 ∧ process.samplePosition > 0 then
    SamplingMethod.Systematic
  else
    SamplingMethod.Other

/-- Theorem stating that the described process is systematic sampling --/
theorem factory_sampling_is_systematic (process : ProductionProcess) 
  (h1 : process.inspectionInterval = 10)
  (h2 : process.samplePosition > 0) :
  determineSamplingMethod process = SamplingMethod.Systematic := by
  sorry


end NUMINAMATH_CALUDE_factory_sampling_is_systematic_l3056_305693


namespace NUMINAMATH_CALUDE_inequality_solution_exists_implies_a_leq_4_l3056_305663

theorem inequality_solution_exists_implies_a_leq_4 :
  (∃ x : ℝ, |x - 2| - |x + 2| ≥ a) → a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_exists_implies_a_leq_4_l3056_305663


namespace NUMINAMATH_CALUDE_difference_between_half_and_sixth_l3056_305642

theorem difference_between_half_and_sixth (x y : ℚ) : x = 1/2 → y = 1/6 → x - y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_half_and_sixth_l3056_305642


namespace NUMINAMATH_CALUDE_nabla_calculation_l3056_305695

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2

-- State the theorem
theorem nabla_calculation : nabla 2 (nabla 0 (nabla 1 7)) = 71859 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l3056_305695


namespace NUMINAMATH_CALUDE_point_coordinates_proof_l3056_305633

/-- Given points A and B, and the relation between vectors AP and AB, 
    prove that P has specific coordinates. -/
theorem point_coordinates_proof (A B P : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (4, -3) → 
  P - A = 3 • (B - A) → 
  P = (8, -15) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_proof_l3056_305633


namespace NUMINAMATH_CALUDE_triangle_side_length_l3056_305625

theorem triangle_side_length (X Y Z : Real) (XY : Real) :
  -- XYZ is a triangle
  X + Y + Z = Real.pi →
  -- cos(2X - Y) + sin(X + Y) = 2
  Real.cos (2 * X - Y) + Real.sin (X + Y) = 2 →
  -- XY = 6
  XY = 6 →
  -- Then YZ = 3√3
  ∃ (YZ : Real), YZ = 3 * Real.sqrt 3 := by
    sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3056_305625


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3056_305675

/-- The number of coins flipped -/
def n : ℕ := 10

/-- The probability of getting heads on a single coin flip -/
def p : ℚ := 1/2

/-- The probability of getting an equal number of heads and tails -/
def prob_equal : ℚ := (n.choose (n/2)) / 2^n

/-- The probability of getting more heads than tails -/
def prob_more_heads : ℚ := (1 - prob_equal) / 2

theorem coin_flip_probability : prob_more_heads = 193/512 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3056_305675


namespace NUMINAMATH_CALUDE_round_trip_speed_calculation_l3056_305628

/-- Represents a round trip journey between two cities -/
structure RoundTrip where
  initial_speed : ℝ
  initial_time : ℝ
  return_time : ℝ
  average_speed : ℝ
  distance : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem round_trip_speed_calculation (trip : RoundTrip) 
  (h1 : trip.return_time = 2 * trip.initial_time)
  (h2 : trip.average_speed = 34)
  (h3 : trip.distance > 0)
  (h4 : trip.initial_speed > 0) :
  trip.initial_speed = 51 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_speed_calculation_l3056_305628


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l3056_305665

/-- A normally distributed random variable -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  mean_pos : 0 < mean
  std_dev_pos : 0 < std_dev

/-- The probability that a normal random variable falls within an interval -/
noncomputable def prob_in_interval (X : NormalDistribution) (lower upper : ℝ) : ℝ := sorry

/-- Theorem: If P(0 < X < a) = 0.3 for X ~ N(a, d²), then P(0 < X < 2a) = 0.6 -/
theorem normal_distribution_probability 
  (X : NormalDistribution) 
  (h : prob_in_interval X 0 X.mean = 0.3) : 
  prob_in_interval X 0 (2 * X.mean) = 0.6 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l3056_305665


namespace NUMINAMATH_CALUDE_book_collection_average_l3056_305648

def arithmeticSequenceSum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def arithmeticSequenceAverage (a d n : ℕ) : ℚ :=
  (arithmeticSequenceSum a d n : ℚ) / n

theorem book_collection_average :
  arithmeticSequenceAverage 12 12 7 = 48 := by
  sorry

end NUMINAMATH_CALUDE_book_collection_average_l3056_305648


namespace NUMINAMATH_CALUDE_shrink_ray_reduction_l3056_305689

/-- The shrink ray problem -/
theorem shrink_ray_reduction (initial_cups : ℕ) (initial_coffee_per_cup : ℝ) (final_total_coffee : ℝ) :
  initial_cups = 5 →
  initial_coffee_per_cup = 8 →
  final_total_coffee = 20 →
  (1 - final_total_coffee / (initial_cups * initial_coffee_per_cup)) * 100 = 50 := by
  sorry

#check shrink_ray_reduction

end NUMINAMATH_CALUDE_shrink_ray_reduction_l3056_305689


namespace NUMINAMATH_CALUDE_rogers_final_balance_theorem_l3056_305680

/-- Calculates Roger's final balance in US dollars after all transactions -/
def rogers_final_balance (initial_balance : ℝ) (video_game_percentage : ℝ) 
  (euros_spent : ℝ) (euro_to_dollar : ℝ) (canadian_dollars_received : ℝ) 
  (canadian_to_dollar : ℝ) : ℝ :=
  let remaining_after_game := initial_balance * (1 - video_game_percentage)
  let remaining_after_euros := remaining_after_game - euros_spent * euro_to_dollar
  remaining_after_euros + canadian_dollars_received * canadian_to_dollar

/-- Theorem stating Roger's final balance after all transactions -/
theorem rogers_final_balance_theorem : 
  rogers_final_balance 45 0.35 20 1.2 46 0.8 = 42.05 := by
  sorry

end NUMINAMATH_CALUDE_rogers_final_balance_theorem_l3056_305680


namespace NUMINAMATH_CALUDE_random_event_identification_l3056_305697

theorem random_event_identification :
  -- Event ①
  (∀ x : ℝ, x^2 + 1 ≠ 0) ∧
  -- Event ②
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x > 1/x ∧ y ≤ 1/y) ∧
  -- Event ③
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → 1/x > 1/y) ∧
  -- Event ④
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_random_event_identification_l3056_305697


namespace NUMINAMATH_CALUDE_apple_distribution_l3056_305617

/-- The number of ways to distribute n apples among k people, with each person receiving at least m apples -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The problem statement -/
theorem apple_distribution :
  distribution_ways 26 3 3 = 171 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3056_305617


namespace NUMINAMATH_CALUDE_smallest_number_l3056_305622

theorem smallest_number (S : Finset ℕ) (h : S = {5, 8, 1, 2}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3056_305622


namespace NUMINAMATH_CALUDE_inequality_chain_l3056_305626

theorem inequality_chain (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) :
  a > -b ∧ -b > b ∧ b > -a := by sorry

end NUMINAMATH_CALUDE_inequality_chain_l3056_305626


namespace NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_l3056_305627

-- Define the polynomial
def cubic_polynomial (a b : ℝ) (x : ℂ) : ℂ := x^3 - 9*x^2 + b*x + a

-- Define the arithmetic progression property for complex roots
def arithmetic_progression (r₁ r₂ r₃ : ℂ) : Prop :=
  ∃ (d : ℝ), r₂ - r₁ = d ∧ r₃ - r₂ = d

-- State the theorem
theorem cubic_roots_arithmetic_progression (a b : ℝ) :
  (∃ (r₁ r₂ r₃ : ℂ), 
    (∀ x : ℂ, cubic_polynomial a b x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧
    arithmetic_progression r₁ r₂ r₃ ∧
    (∃ i : ℝ, r₂ = i * Complex.I)) →
  (a = 27 + 3 * (Real.sqrt ((a - 27) / 3))^2 ∧ b = -27) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_l3056_305627


namespace NUMINAMATH_CALUDE_quotient_problem_l3056_305607

theorem quotient_problem (q d1 d2 : ℝ) 
  (h1 : q = 6 * d1)  -- quotient is 6 times the dividend
  (h2 : q = 15 * d2) -- quotient is 15 times the divisor
  (h3 : d1 / d2 = q) -- definition of quotient
  : q = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l3056_305607


namespace NUMINAMATH_CALUDE_fraction_problem_l3056_305672

theorem fraction_problem : ∃ f : ℚ, f * 1 = (144 : ℚ) / 216 ∧ f = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3056_305672


namespace NUMINAMATH_CALUDE_limit_s_at_zero_is_infinity_l3056_305676

/-- The x coordinate of the left endpoint of the intersection of y = x^3 and y = m -/
noncomputable def P (m : ℝ) : ℝ := -Real.rpow m (1/3)

/-- The function s defined as [P(-m) - P(m)]/m -/
noncomputable def s (m : ℝ) : ℝ := (P (-m) - P m) / m

theorem limit_s_at_zero_is_infinity :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ ∧ -2 < m ∧ m < 2 → |s m| > ε :=
sorry

end NUMINAMATH_CALUDE_limit_s_at_zero_is_infinity_l3056_305676


namespace NUMINAMATH_CALUDE_red_toy_percentage_l3056_305652

/-- Represents a toy production lot -/
structure ToyLot where
  total : ℕ
  red : ℕ
  green : ℕ
  small : ℕ
  large : ℕ
  redSmall : ℕ
  redLarge : ℕ
  greenLarge : ℕ

/-- The conditions of the toy production lot -/
def validToyLot (lot : ToyLot) : Prop :=
  lot.total > 0 ∧
  lot.red + lot.green = lot.total ∧
  lot.small + lot.large = lot.total ∧
  lot.small = lot.large ∧
  lot.redSmall = (lot.total * 10) / 100 ∧
  lot.greenLarge = 40 ∧
  lot.redLarge = 60

/-- The theorem stating the percentage of red toys -/
theorem red_toy_percentage (lot : ToyLot) :
  validToyLot lot → (lot.red : ℚ) / lot.total = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_red_toy_percentage_l3056_305652


namespace NUMINAMATH_CALUDE_fitness_center_membership_ratio_l3056_305647

theorem fitness_center_membership_ratio :
  ∀ (f m c : ℕ), 
  (f > 0) → (m > 0) → (c > 0) →
  (35 * f + 30 * m + 10 * c : ℝ) / (f + m + c : ℝ) = 25 →
  ∃ (k : ℕ), k > 0 ∧ f = 3 * k ∧ m = 6 * k ∧ c = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_fitness_center_membership_ratio_l3056_305647


namespace NUMINAMATH_CALUDE_exponential_equality_l3056_305615

theorem exponential_equality (x a b : ℝ) (h1 : 3^x = a) (h2 : 5^x = b) : 45^x = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_exponential_equality_l3056_305615


namespace NUMINAMATH_CALUDE_point_division_theorem_l3056_305657

/-- Given a line segment AB and a point P on it such that AP:PB = 3:5,
    prove that P = (5/8)*A + (3/8)*B --/
theorem point_division_theorem (A B P : ℝ × ℝ) : 
  (∃ t : ℝ, P = A + t • (B - A)) → -- P is on line segment AB
  (dist A P : ℝ) / (dist P B) = 3 / 5 → -- AP:PB = 3:5
  P = (5/8 : ℝ) • A + (3/8 : ℝ) • B := by sorry

end NUMINAMATH_CALUDE_point_division_theorem_l3056_305657


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3056_305669

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), y = 2*x + 10 ∧ x^2 + y^2 = (a^2 + b^2)) → 
  (b / a = 2) → 
  (a^2 = 5 ∧ b^2 = 20) := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3056_305669


namespace NUMINAMATH_CALUDE_running_time_proof_l3056_305658

/-- Proves that the time taken for Joe and Pete to be 16 km apart is 80 minutes -/
theorem running_time_proof (joe_speed : ℝ) (pete_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  joe_speed = 0.133333333333 →
  pete_speed = joe_speed / 2 →
  distance = 16 →
  time * (joe_speed + pete_speed) = distance →
  time = 80 := by
sorry

end NUMINAMATH_CALUDE_running_time_proof_l3056_305658


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l3056_305653

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 → q > 0 → 
  10 * p^9 * q = 45 * p^8 * q^2 →
  p + 2*q = 1 →
  p = 9/13 := by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l3056_305653


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3056_305618

theorem square_sum_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 20) 
  (h2 : p + q = 10) : 
  p^2 + q^2 = 60 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3056_305618


namespace NUMINAMATH_CALUDE_fraction_equality_l3056_305619

theorem fraction_equality (x y : ℝ) (h : (x - y) / (x + y) = 5) :
  (2 * x + 3 * y) / (3 * x - 2 * y) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3056_305619


namespace NUMINAMATH_CALUDE_vector_subtraction_l3056_305613

def vector_a : ℝ × ℝ := (-1, 2)
def vector_b : ℝ × ℝ := (0, 1)

theorem vector_subtraction :
  vector_a - 2 • vector_b = (-1, 0) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3056_305613


namespace NUMINAMATH_CALUDE_system_solution_unique_l3056_305635

theorem system_solution_unique :
  ∃! (x y : ℚ), 3 * x - 2 * y = 5 ∧ x + 4 * y = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3056_305635


namespace NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l3056_305629

/-- Given a rectangle of dimensions a × b units divided into a smaller rectangle of dimensions p × q units
    and four congruent rectangles, the perimeter of one of the four congruent rectangles is 2(a + b - p - q) units. -/
theorem congruent_rectangle_perimeter
  (a b p q : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : p > 0)
  (h4 : q > 0)
  (h5 : p < a)
  (h6 : q < b) :
  ∃ (l1 l2 : ℝ), l1 = b - q ∧ l2 = a - p ∧ 2 * (l1 + l2) = 2 * (a + b - p - q) :=
sorry

end NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l3056_305629


namespace NUMINAMATH_CALUDE_megan_deleted_files_l3056_305687

/-- Calculates the number of deleted files given the initial number of files,
    number of folders after organizing, and number of files per folder. -/
def deleted_files (initial_files : ℕ) (num_folders : ℕ) (files_per_folder : ℕ) : ℕ :=
  initial_files - (num_folders * files_per_folder)

/-- Proves that Megan deleted 21 files given the problem conditions. -/
theorem megan_deleted_files :
  deleted_files 93 9 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_megan_deleted_files_l3056_305687


namespace NUMINAMATH_CALUDE_base_nine_solution_l3056_305641

/-- Convert a list of digits in base b to its decimal representation -/
def toDecimal (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Check if the equation is valid in base b -/
def isValidEquation (b : ℕ) : Prop :=
  toDecimal [5, 7, 4, 2] b + toDecimal [6, 9, 3, 1] b = toDecimal [1, 2, 7, 7, 3] b

theorem base_nine_solution :
  ∃ (b : ℕ), b > 1 ∧ isValidEquation b ∧ ∀ (x : ℕ), x > 1 ∧ x ≠ b → ¬isValidEquation x :=
by sorry

end NUMINAMATH_CALUDE_base_nine_solution_l3056_305641


namespace NUMINAMATH_CALUDE_evaluate_expression_l3056_305646

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3056_305646


namespace NUMINAMATH_CALUDE_tan_alpha_for_point_on_terminal_side_l3056_305601

theorem tan_alpha_for_point_on_terminal_side (α : Real) :
  let P : ℝ × ℝ := (1, -2)
  (P.1 = 1 ∧ P.2 = -2) →  -- Point P(1, -2) lies on the terminal side of angle α
  Real.tan α = -2 :=
by sorry

end NUMINAMATH_CALUDE_tan_alpha_for_point_on_terminal_side_l3056_305601


namespace NUMINAMATH_CALUDE_football_shaped_area_l3056_305656

/-- The area of two quarter-circle sectors minus two right triangles in a square with side length 4 -/
theorem football_shaped_area (π : ℝ) (h_π : π = Real.pi) : 
  let side_length : ℝ := 4
  let diagonal : ℝ := side_length * Real.sqrt 2
  let quarter_circle_area : ℝ := (π * diagonal^2) / 4
  let triangle_area : ℝ := side_length^2 / 2
  2 * (quarter_circle_area - triangle_area) = 16 * π - 16 := by
sorry

end NUMINAMATH_CALUDE_football_shaped_area_l3056_305656


namespace NUMINAMATH_CALUDE_celine_erasers_l3056_305698

/-- Proves that Celine collected 10 erasers given the conditions of the problem -/
theorem celine_erasers (gabriel : ℕ) (celine : ℕ) (julian : ℕ) : 
  celine = 2 * gabriel → 
  julian = 2 * celine → 
  gabriel + celine + julian = 35 → 
  celine = 10 := by
sorry

end NUMINAMATH_CALUDE_celine_erasers_l3056_305698


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3056_305662

theorem necessary_but_not_sufficient_condition (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, x^2 - a*x + 3 < 0) → (a > 3 ∧ ∃ b > 3, ¬(∀ x ∈ Set.Icc 1 3, x^2 - b*x + 3 < 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3056_305662


namespace NUMINAMATH_CALUDE_geometric_means_l3056_305631

theorem geometric_means (a b : ℝ) (p : ℕ) (ha : 0 < a) (hb : a < b) :
  let r := (b / a) ^ (1 / (p + 1 : ℝ))
  ∀ k : ℕ, k ≥ 1 → k ≤ p →
    a * r ^ k = a * (b / a) ^ (k / (p + 1 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_geometric_means_l3056_305631


namespace NUMINAMATH_CALUDE_range_of_a_l3056_305678

open Real

noncomputable def f (a x : ℝ) : ℝ := x - (a + 1) * log x

noncomputable def g (a x : ℝ) : ℝ := a / x - 3

noncomputable def h (a x : ℝ) : ℝ := f a x - g a x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ g a x) →
  a ∈ Set.Iic (exp 1 * (exp 1 + 2) / (exp 1 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3056_305678


namespace NUMINAMATH_CALUDE_fraction_difference_l3056_305636

theorem fraction_difference (A B C : ℚ) (k m : ℕ) : 
  A = 3 * k / (2 * m) →
  B = 2 * k / (3 * m) →
  C = k / (4 * m) →
  A + B + C = 29 / 60 →
  A - B - C = 7 / 60 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_l3056_305636


namespace NUMINAMATH_CALUDE_penny_frog_count_l3056_305682

/-- The number of tree frogs Penny counted -/
def tree_frogs : ℕ := 55

/-- The number of poison frogs Penny counted -/
def poison_frogs : ℕ := 10

/-- The number of wood frogs Penny counted -/
def wood_frogs : ℕ := 13

/-- The total number of frogs Penny counted -/
def total_frogs : ℕ := tree_frogs + poison_frogs + wood_frogs

theorem penny_frog_count : total_frogs = 78 := by
  sorry

end NUMINAMATH_CALUDE_penny_frog_count_l3056_305682


namespace NUMINAMATH_CALUDE_symmetric_point_origin_symmetric_point_negative_two_five_l3056_305685

def symmetric_point (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem symmetric_point_origin (x y : ℝ) : 
  symmetric_point x y = (-x, -y) := by sorry

theorem symmetric_point_negative_two_five : 
  symmetric_point (-2) 5 = (2, -5) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_origin_symmetric_point_negative_two_five_l3056_305685


namespace NUMINAMATH_CALUDE_press_conference_seating_l3056_305645

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (team_sizes : List ℕ) : ℕ :=
  (factorial team_sizes.length) * (team_sizes.map factorial).prod

theorem press_conference_seating :
  seating_arrangements [3, 3, 2, 2] = 3456 := by
  sorry

end NUMINAMATH_CALUDE_press_conference_seating_l3056_305645


namespace NUMINAMATH_CALUDE_star_properties_l3056_305638

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

theorem star_properties :
  (∀ x y : ℝ, star x y = star y x) ∧
  (∃ x y z : ℝ, star x (y + z) ≠ star x y + star x z) ∧
  (∃ x : ℝ, star (x - 2) (x + 2) ≠ star x x - 2) ∧
  (¬ ∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x) ∧
  (∃ x y z : ℝ, star (star x y) z ≠ star x (star y z)) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l3056_305638


namespace NUMINAMATH_CALUDE_hyperbola_min_value_l3056_305683

/-- The minimum value of (b² + 1) / a for a hyperbola with eccentricity 2 -/
theorem hyperbola_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let c := e * a  -- focal distance
  (c^2 = a^2 + b^2) →  -- hyperbola property
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (b^2 + 1) / a ≥ 2 * Real.sqrt 3) ∧
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ (b^2 + 1) / a = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_min_value_l3056_305683


namespace NUMINAMATH_CALUDE_parabola_c_value_l3056_305639

/-- A parabola with equation x = ay^2 + by + c, vertex (4, 1), and passing through point (-1, 3) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 4
  vertex_y : ℝ := 1
  point_x : ℝ := -1
  point_y : ℝ := 3
  eq_vertex : 4 = a * 1^2 + b * 1 + c
  eq_point : -1 = a * 3^2 + b * 3 + c

/-- The value of c for the given parabola is 11/4 -/
theorem parabola_c_value (p : Parabola) : p.c = 11/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3056_305639


namespace NUMINAMATH_CALUDE_liz_scored_three_three_pointers_l3056_305608

/-- Represents the basketball game scenario described in the problem -/
structure BasketballGame where
  initial_deficit : ℕ
  free_throws : ℕ
  jump_shots : ℕ
  opponent_points : ℕ
  final_deficit : ℕ

/-- Calculates the number of three-pointers Liz scored -/
def three_pointers (game : BasketballGame) : ℕ :=
  let points_needed := game.initial_deficit - game.final_deficit + game.opponent_points
  let points_from_other_shots := game.free_throws + 2 * game.jump_shots
  (points_needed - points_from_other_shots) / 3

/-- Theorem stating that Liz scored 3 three-pointers -/
theorem liz_scored_three_three_pointers :
  let game := BasketballGame.mk 20 5 4 10 8
  three_pointers game = 3 := by sorry

end NUMINAMATH_CALUDE_liz_scored_three_three_pointers_l3056_305608


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l3056_305632

theorem jelly_bean_probability (p_red p_green p_orange_yellow : ℝ) :
  p_red = 0.25 →
  p_green = 0.35 →
  p_red + p_green + p_orange_yellow = 1 →
  p_orange_yellow = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l3056_305632


namespace NUMINAMATH_CALUDE_sequences_properties_l3056_305660

/-- Arithmetic sequence with first term 3 and common difference 2 -/
def a (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- Geometric sequence with first term 1 and common ratio 2 -/
def b (n : ℕ) : ℕ := 2^(n - 1)

/-- Product sequence of a and b -/
def c (n : ℕ) : ℕ := a n * b n

/-- Sum of first n terms of arithmetic sequence a -/
def S (n : ℕ) : ℕ := n * (a 1 + a n) / 2

/-- Sum of first n terms of product sequence c -/
def T (n : ℕ) : ℕ := (2 * n - 1) * 2^n + 1

theorem sequences_properties :
  (a 1 = 3) ∧
  (b 1 = 1) ∧
  (b 2 + S 2 = 10) ∧
  (a 5 - 2 * b 2 = a 3) ∧
  (∀ n : ℕ, a n = 2 * n + 1) ∧
  (∀ n : ℕ, b n = 2^(n - 1)) ∧
  (∀ n : ℕ, T n = (2 * n - 1) * 2^n + 1) := by
  sorry

#check sequences_properties

end NUMINAMATH_CALUDE_sequences_properties_l3056_305660


namespace NUMINAMATH_CALUDE_pastry_difference_l3056_305623

/-- Represents the number of pastries each person has -/
structure Pastries where
  frank : ℕ
  calvin : ℕ
  phoebe : ℕ
  grace : ℕ

/-- The conditions of the pastry problem -/
def PastryProblem (p : Pastries) : Prop :=
  p.calvin = p.frank + 8 ∧
  p.phoebe = p.frank + 8 ∧
  p.grace = 30 ∧
  p.frank + p.calvin + p.phoebe + p.grace = 97

theorem pastry_difference (p : Pastries) (h : PastryProblem p) :
  p.grace - p.calvin = 5 ∧ p.grace - p.phoebe = 5 :=
sorry

end NUMINAMATH_CALUDE_pastry_difference_l3056_305623


namespace NUMINAMATH_CALUDE_square_side_length_l3056_305640

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an octagon -/
structure Octagon :=
  (A B C D E F G H : Point)

/-- Represents a square -/
structure Square :=
  (W X Y Z : Point)

/-- The octagon ABCDEFGH -/
def octagon : Octagon := sorry

/-- The inscribed square WXYZ -/
def square : Square := sorry

/-- W is on BC -/
axiom W_on_BC : square.W.x ≥ octagon.B.x ∧ square.W.x ≤ octagon.C.x ∧ 
                square.W.y = octagon.B.y ∧ square.W.y = octagon.C.y

/-- X is on DE -/
axiom X_on_DE : square.X.x ≥ octagon.D.x ∧ square.X.x ≤ octagon.E.x ∧ 
                square.X.y = octagon.D.y ∧ square.X.y = octagon.E.y

/-- Y is on FG -/
axiom Y_on_FG : square.Y.x ≥ octagon.F.x ∧ square.Y.x ≤ octagon.G.x ∧ 
                square.Y.y = octagon.F.y ∧ square.Y.y = octagon.G.y

/-- Z is on HA -/
axiom Z_on_HA : square.Z.x ≥ octagon.H.x ∧ square.Z.x ≤ octagon.A.x ∧ 
                square.Z.y = octagon.H.y ∧ square.Z.y = octagon.A.y

/-- AB = 50 -/
axiom AB_length : Real.sqrt ((octagon.A.x - octagon.B.x)^2 + (octagon.A.y - octagon.B.y)^2) = 50

/-- GH = 50(√3 - 1) -/
axiom GH_length : Real.sqrt ((octagon.G.x - octagon.H.x)^2 + (octagon.G.y - octagon.H.y)^2) = 50 * (Real.sqrt 3 - 1)

/-- The side length of square WXYZ is 50 -/
theorem square_side_length : 
  Real.sqrt ((square.W.x - square.Z.x)^2 + (square.W.y - square.Z.y)^2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3056_305640


namespace NUMINAMATH_CALUDE_seating_arrangement_probability_l3056_305694

/-- The number of delegates --/
def num_delegates : ℕ := 12

/-- The number of countries --/
def num_countries : ℕ := 3

/-- The number of delegates per country --/
def delegates_per_country : ℕ := 4

/-- The probability that each delegate sits next to at least one delegate from another country --/
def seating_probability : ℚ := 21 / 22

/-- Theorem stating the probability of the seating arrangement --/
theorem seating_arrangement_probability :
  let total_arrangements := (num_delegates.factorial) / (delegates_per_country.factorial ^ num_countries)
  let unwanted_arrangements := num_countries * num_delegates * (num_delegates - delegates_per_country).factorial / 
                               (delegates_per_country.factorial ^ (num_countries - 1)) -
                               (num_countries.choose 2) * num_delegates * delegates_per_country +
                               num_delegates * (num_countries - 1)
  (total_arrangements - unwanted_arrangements) / total_arrangements = seating_probability :=
sorry

end NUMINAMATH_CALUDE_seating_arrangement_probability_l3056_305694


namespace NUMINAMATH_CALUDE_cube_expansion_value_l3056_305630

theorem cube_expansion_value (y : ℝ) (h : y = 50) : 
  y^3 + 3*y^2*(2*y) + 3*y*(2*y)^2 + (2*y)^3 = 3375000 := by
sorry

end NUMINAMATH_CALUDE_cube_expansion_value_l3056_305630


namespace NUMINAMATH_CALUDE_total_results_l3056_305649

theorem total_results (total_sum : ℕ) (total_count : ℕ) 
  (first_six_sum : ℕ) (last_six_sum : ℕ) (sixth_result : ℕ) :
  total_sum / total_count = 60 →
  first_six_sum = 6 * 58 →
  last_six_sum = 6 * 63 →
  sixth_result = 66 →
  total_sum = first_six_sum + last_six_sum - sixth_result →
  total_count = 11 := by
sorry

end NUMINAMATH_CALUDE_total_results_l3056_305649


namespace NUMINAMATH_CALUDE_trig_simplification_l3056_305674

open Real

theorem trig_simplification (α : ℝ) :
  (tan (π/4 - α) / (1 - tan (π/4 - α)^2)) * ((sin α * cos α) / (cos α^2 - sin α^2)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3056_305674


namespace NUMINAMATH_CALUDE_statement_2_statement_3_l3056_305651

-- Define the types for lines and planes
variable {Line Plane : Type}

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)
variable (lineParallelPlane : Line → Plane → Prop)
variable (linePerpendicularPlane : Line → Plane → Prop)

-- Statement 2
theorem statement_2 (α β : Plane) (m : Line) :
  planePerpendicular α β → lineParallelPlane m α → linePerpendicularPlane m β := by
  sorry

-- Statement 3
theorem statement_3 (α β : Plane) (m : Line) :
  linePerpendicularPlane m β → planeParallel β α → planePerpendicular α β := by
  sorry

end NUMINAMATH_CALUDE_statement_2_statement_3_l3056_305651


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3056_305699

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- For any real number m, the point (-1, m^2 + 1) is in the second quadrant -/
theorem point_in_second_quadrant (m : ℝ) : in_second_quadrant (-1) (m^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3056_305699


namespace NUMINAMATH_CALUDE_tables_left_l3056_305684

theorem tables_left (original_tables : ℝ) (customers_per_table : ℝ) (current_customers : ℕ) :
  original_tables = 44.0 →
  customers_per_table = 8.0 →
  current_customers = 256 →
  original_tables - (current_customers : ℝ) / customers_per_table = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_tables_left_l3056_305684


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attainable_l3056_305612

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
  sorry

theorem min_reciprocal_sum_attainable : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ 1/a + 1/b + 1/c = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attainable_l3056_305612


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l3056_305666

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_point_x_coordinate (x : ℝ) (h : x > 0) : 
  (deriv f x = 2) → x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l3056_305666


namespace NUMINAMATH_CALUDE_complex_power_multiply_l3056_305681

theorem complex_power_multiply (i : ℂ) : i^2 = -1 → i^13 * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_multiply_l3056_305681


namespace NUMINAMATH_CALUDE_alternating_digit_sum_2017_l3056_305606

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Alternating sum of digit sums from 1 to n -/
def alternating_digit_sum (n : ℕ) : ℤ :=
  (Finset.range n).sum (fun i => (-1)^i.succ * (digit_sum (i + 1) : ℤ))

/-- The alternating sum of digit sums for integers from 1 to 2017 is equal to 1009 -/
theorem alternating_digit_sum_2017 : alternating_digit_sum 2017 = 1009 := by sorry

end NUMINAMATH_CALUDE_alternating_digit_sum_2017_l3056_305606


namespace NUMINAMATH_CALUDE_expression_evaluation_l3056_305605

theorem expression_evaluation : 
  (2002 : ℤ)^3 - 2001 * 2002^2 - 2001^2 * 2002 + 2001^3 + (2002 - 2001)^3 = 4004 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3056_305605


namespace NUMINAMATH_CALUDE_inequality_solution_l3056_305614

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 1}
  else if a < 0 then {x | x > 1 ∨ x < 1/a}
  else if 0 < a ∧ a < 1 then {x | 1 < x ∧ x < 1/a}
  else if a > 1 then {x | 1/a < x ∧ x < 1}
  else ∅

theorem inequality_solution (a : ℝ) (x : ℝ) :
  a * x^2 - (a + 1) * x + 1 < 0 ↔ x ∈ solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3056_305614


namespace NUMINAMATH_CALUDE_additional_people_for_lawn_mowing_l3056_305696

/-- The number of additional people needed to mow a lawn in a shorter time -/
theorem additional_people_for_lawn_mowing 
  (initial_people : ℕ) 
  (initial_time : ℕ) 
  (new_time : ℕ) 
  (h1 : initial_people > 0)
  (h2 : initial_time > 0)
  (h3 : new_time > 0)
  (h4 : new_time < initial_time) :
  let total_work := initial_people * initial_time
  let new_people := total_work / new_time
  new_people - initial_people = 10 :=
by sorry

end NUMINAMATH_CALUDE_additional_people_for_lawn_mowing_l3056_305696


namespace NUMINAMATH_CALUDE_initial_games_count_l3056_305643

theorem initial_games_count (sold : ℕ) (added : ℕ) (final : ℕ) : 
  sold = 68 → added = 47 → final = 74 → 
  ∃ initial : ℕ, initial - sold + added = final ∧ initial = 95 := by
sorry

end NUMINAMATH_CALUDE_initial_games_count_l3056_305643


namespace NUMINAMATH_CALUDE_value_of_x_l3056_305664

theorem value_of_x (x y z : ℚ) 
  (h1 : x = y / 2) 
  (h2 : y = z / 3) 
  (h3 : z = 100) : 
  x = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3056_305664


namespace NUMINAMATH_CALUDE_glenda_skating_speed_l3056_305621

/-- Prove Glenda's skating speed given the conditions of the problem -/
theorem glenda_skating_speed 
  (ann_speed : ℝ) 
  (time : ℝ) 
  (total_distance : ℝ) 
  (h1 : ann_speed = 6)
  (h2 : time = 3)
  (h3 : total_distance = 42) :
  ∃ (glenda_speed : ℝ), 
    glenda_speed = 8 ∧ 
    ann_speed * time + glenda_speed * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_glenda_skating_speed_l3056_305621


namespace NUMINAMATH_CALUDE_alice_basic_salary_l3056_305609

/-- Calculates the monthly basic salary given total sales, commission rate, and savings. -/
def calculate_basic_salary (total_sales : ℝ) (commission_rate : ℝ) (savings : ℝ) : ℝ :=
  let total_earnings := savings * 10
  let commission := total_sales * commission_rate
  total_earnings - commission

/-- Proves that given the specified conditions, Alice's monthly basic salary is $240. -/
theorem alice_basic_salary :
  let total_sales : ℝ := 2500
  let commission_rate : ℝ := 0.02
  let savings : ℝ := 29
  calculate_basic_salary total_sales commission_rate savings = 240 := by
  sorry

#eval calculate_basic_salary 2500 0.02 29

end NUMINAMATH_CALUDE_alice_basic_salary_l3056_305609
