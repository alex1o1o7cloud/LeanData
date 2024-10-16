import Mathlib

namespace NUMINAMATH_CALUDE_geometric_series_first_term_l1108_110874

theorem geometric_series_first_term (a₁ q : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → ∃ (aₙ : ℝ), aₙ = a₁ * q^(n-1)) →  -- Geometric series condition
  (-1 < q) →                                         -- Convergence condition
  (q < 1) →                                          -- Convergence condition
  (q ≠ 0) →                                          -- Non-zero common ratio
  (a₁ / (1 - q) = 1) →                               -- Sum of series is 1
  (|a₁| / (1 - |q|) = 2) →                           -- Sum of absolute values is 2
  a₁ = 4/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l1108_110874


namespace NUMINAMATH_CALUDE_carmelas_initial_money_l1108_110837

/-- Proves that Carmela's initial amount of money is $7 given the problem conditions --/
theorem carmelas_initial_money :
  ∀ x : ℕ,
  (∃ (final_amount : ℕ),
    -- Carmela's final amount after giving $1 to each of 4 cousins
    x - 4 = final_amount ∧
    -- Each cousin's final amount after receiving $1
    2 + 1 = final_amount) →
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_carmelas_initial_money_l1108_110837


namespace NUMINAMATH_CALUDE_system_of_equations_l1108_110857

theorem system_of_equations (x y a : ℝ) : 
  (2 * x + y = 2 * a + 1) → 
  (x + 2 * y = a - 1) → 
  (x - y = 4) → 
  (a = 2) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l1108_110857


namespace NUMINAMATH_CALUDE_largest_satisfying_n_l1108_110899

/-- A rectangle with sides parallel to the coordinate axes -/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- Two rectangles are disjoint -/
def disjoint (r1 r2 : Rectangle) : Prop :=
  r1.x_max ≤ r2.x_min ∨ r2.x_max ≤ r1.x_min ∨
  r1.y_max ≤ r2.y_min ∨ r2.y_max ≤ r1.y_min

/-- Two rectangles have a common point -/
def have_common_point (r1 r2 : Rectangle) : Prop :=
  ¬(disjoint r1 r2)

/-- The property described in the problem -/
def satisfies_property (n : ℕ) : Prop :=
  ∃ (A B : Fin n → Rectangle),
    (∀ i : Fin n, disjoint (A i) (B i)) ∧
    (∀ i j : Fin n, i ≠ j → have_common_point (A i) (B j))

/-- The main theorem: The largest positive integer satisfying the property is 4 -/
theorem largest_satisfying_n :
  (∃ n : ℕ, n > 0 ∧ satisfies_property n) ∧
  (∀ n : ℕ, satisfies_property n → n ≤ 4) ∧
  satisfies_property 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_satisfying_n_l1108_110899


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l1108_110817

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between planes
variable (planeParallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the containment relation between a line and a plane
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (a : Line) (α β : Plane)
  (h1 : planeParallel α β)
  (h2 : lineInPlane a α) :
  lineParallelPlane a β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l1108_110817


namespace NUMINAMATH_CALUDE_emilias_blueberries_l1108_110808

/-- The number of cartons of berries Emilia needs in total -/
def total_needed : ℕ := 42

/-- The number of cartons of strawberries Emilia has -/
def strawberries : ℕ := 2

/-- The number of cartons of berries Emilia buys at the supermarket -/
def bought : ℕ := 33

/-- The number of cartons of blueberries in Emilia's cupboard -/
def blueberries : ℕ := total_needed - (strawberries + bought)

theorem emilias_blueberries : blueberries = 7 := by
  sorry

end NUMINAMATH_CALUDE_emilias_blueberries_l1108_110808


namespace NUMINAMATH_CALUDE_parabola_with_same_shape_and_vertex_l1108_110869

/-- A parabola with the same shape and opening direction as y = -3x^2 + 1 and vertex at (-1, 2) -/
theorem parabola_with_same_shape_and_vertex (x y : ℝ) : 
  y = -3 * (x + 1)^2 + 2 → 
  (∃ (a b c : ℝ), y = -3 * x^2 + b * x + c) ∧ 
  (y = -3 * (-1)^2 + 2 ∧ ∀ (h : ℝ), y ≤ -3 * (h + 1)^2 + 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_with_same_shape_and_vertex_l1108_110869


namespace NUMINAMATH_CALUDE_sequence_property_l1108_110803

/-- Two infinite sequences of rational numbers -/
def Sequence := ℕ → ℚ

/-- Property that a sequence is nonconstant -/
def Nonconstant (s : Sequence) : Prop :=
  ∃ i j, s i ≠ s j

/-- Property that (sᵢ - sⱼ)(tᵢ - tⱼ) is an integer for all i and j -/
def IntegerProduct (s t : Sequence) : Prop :=
  ∀ i j, ∃ k : ℤ, (s i - s j) * (t i - t j) = k

theorem sequence_property (s t : Sequence) 
  (hs : Nonconstant s) (ht : Nonconstant t) (h : IntegerProduct s t) :
  ∃ r : ℚ, (∀ i j : ℕ, ∃ m n : ℤ, (s i - s j) * r = m ∧ (t i - t j) / r = n) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l1108_110803


namespace NUMINAMATH_CALUDE_remainder_sum_powers_mod_five_l1108_110848

theorem remainder_sum_powers_mod_five :
  (9^7 + 8^8 + 7^9) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_powers_mod_five_l1108_110848


namespace NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l1108_110852

-- Define the condition for an ellipse
def is_ellipse (a b : ℝ) : Prop := ∃ (x y : ℝ), a * x^2 + b * y^2 = 1 ∧ a > 0 ∧ b > 0

-- Theorem stating that ab > 0 is necessary but not sufficient for an ellipse
theorem ab_positive_necessary_not_sufficient :
  (∀ a b : ℝ, is_ellipse a b → a * b > 0) ∧
  ¬(∀ a b : ℝ, a * b > 0 → is_ellipse a b) :=
sorry

end NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l1108_110852


namespace NUMINAMATH_CALUDE_triangles_on_circle_l1108_110898

theorem triangles_on_circle (n : ℕ) (h : n = 15) : 
  (Nat.choose n 3) = 455 := by sorry

end NUMINAMATH_CALUDE_triangles_on_circle_l1108_110898


namespace NUMINAMATH_CALUDE_BE_length_l1108_110853

-- Define the points
variable (A B C D E F G H : Point)

-- Define the square ABCD
def is_square (A B C D : Point) : Prop := sorry

-- Define that E is on the extension of BC
def on_extension (E B C : Point) : Prop := sorry

-- Define the square AEFG
def is_square_AEFG (A E F G : Point) : Prop := sorry

-- Define that A and G are on the same side of BE
def same_side (A G B E : Point) : Prop := sorry

-- Define that H is on the extension of BD and intersects AF
def intersects_extension (H B D A F : Point) : Prop := sorry

-- Define the lengths
def length (P Q : Point) : ℝ := sorry

-- State the theorem
theorem BE_length 
  (h1 : is_square A B C D)
  (h2 : on_extension E B C)
  (h3 : is_square_AEFG A E F G)
  (h4 : same_side A G B E)
  (h5 : intersects_extension H B D A F)
  (h6 : length H D = Real.sqrt 2)
  (h7 : length F H = 5 * Real.sqrt 2) :
  length B E = 8 := by sorry

end NUMINAMATH_CALUDE_BE_length_l1108_110853


namespace NUMINAMATH_CALUDE_return_speed_calculation_l1108_110864

/-- Proves that given a round trip of 4 miles (2 miles each way), where the first half
    takes 1 hour and the average speed for the entire trip is 3 miles/hour,
    the speed for the second half of the trip is 6 miles/hour. -/
theorem return_speed_calculation (total_distance : ℝ) (outbound_distance : ℝ) 
    (outbound_time : ℝ) (average_speed : ℝ) :
  total_distance = 4 →
  outbound_distance = 2 →
  outbound_time = 1 →
  average_speed = 3 →
  ∃ (return_speed : ℝ), 
    return_speed = 6 ∧ 
    average_speed = total_distance / (outbound_time + outbound_distance / return_speed) := by
  sorry


end NUMINAMATH_CALUDE_return_speed_calculation_l1108_110864


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1108_110830

theorem complex_modulus_problem (z : ℂ) : 
  ((1 - Complex.I) / Complex.I) * z = 1 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1108_110830


namespace NUMINAMATH_CALUDE_no_numbers_divisible_by_all_l1108_110891

theorem no_numbers_divisible_by_all : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 →
  ¬(2 ∣ n ∧ 3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_no_numbers_divisible_by_all_l1108_110891


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l1108_110889

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope 3 and x-intercept (4, 0), the y-intercept is (0, -12). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := 3, x_intercept := (4, 0) }
  y_intercept l = (0, -12) := by sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l1108_110889


namespace NUMINAMATH_CALUDE_x_equals_one_ninth_l1108_110825

theorem x_equals_one_ninth (x : ℚ) (h : x - 1/10 = x/10) : x = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_ninth_l1108_110825


namespace NUMINAMATH_CALUDE_town_trash_cans_l1108_110814

theorem town_trash_cans (street_cans : ℕ) (store_cans : ℕ) : 
  street_cans = 14 →
  store_cans = 2 * street_cans →
  street_cans + store_cans = 42 := by
sorry

end NUMINAMATH_CALUDE_town_trash_cans_l1108_110814


namespace NUMINAMATH_CALUDE_dog_bones_proof_l1108_110842

/-- The number of bones the dog dug up -/
def bones_dug_up : ℕ := 367

/-- The total number of bones the dog has now -/
def total_bones_now : ℕ := 860

/-- The initial number of bones the dog had -/
def initial_bones : ℕ := total_bones_now - bones_dug_up

theorem dog_bones_proof : initial_bones = 493 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_proof_l1108_110842


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l1108_110871

/-- Given an arithmetic sequence {a_n} with first term a and common difference d,
    prove that the 11th term is 3.5 under the given conditions. -/
theorem arithmetic_sequence_11th_term
  (a d : ℝ)
  (h1 : a + (a + 3 * d) + (a + 6 * d) = 31.5)
  (h2 : 9 * a + (9 * 8 / 2) * d = 85.5) :
  a + 10 * d = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l1108_110871


namespace NUMINAMATH_CALUDE_andrew_flooring_planks_l1108_110896

/-- The number of planks Andrew bought for his flooring project -/
def total_planks : ℕ := 65

/-- The number of planks used in Andrew's bedroom -/
def bedroom_planks : ℕ := 8

/-- The number of planks used in the living room -/
def living_room_planks : ℕ := 20

/-- The number of planks used in the kitchen -/
def kitchen_planks : ℕ := 11

/-- The number of planks used in the guest bedroom -/
def guest_bedroom_planks : ℕ := bedroom_planks - 2

/-- The number of planks used in each hallway -/
def hallway_planks : ℕ := 4

/-- The number of planks ruined and replaced in each bedroom -/
def ruined_planks_per_bedroom : ℕ := 3

/-- The number of leftover planks -/
def leftover_planks : ℕ := 6

/-- The number of hallways -/
def num_hallways : ℕ := 2

/-- The number of bedrooms -/
def num_bedrooms : ℕ := 2

theorem andrew_flooring_planks :
  total_planks = 
    bedroom_planks + living_room_planks + kitchen_planks + guest_bedroom_planks + 
    (num_hallways * hallway_planks) + (num_bedrooms * ruined_planks_per_bedroom) + 
    leftover_planks :=
by sorry

end NUMINAMATH_CALUDE_andrew_flooring_planks_l1108_110896


namespace NUMINAMATH_CALUDE_seat_to_right_of_xiaofang_l1108_110805

/-- Represents a seat position as an ordered pair of integers -/
structure SeatPosition :=
  (column : ℤ)
  (row : ℤ)

/-- Returns the seat position to the right of a given seat -/
def seatToRight (seat : SeatPosition) : SeatPosition :=
  { column := seat.column + 1, row := seat.row }

/-- Xiaofang's seat position -/
def xiaofangSeat : SeatPosition := { column := 3, row := 5 }

theorem seat_to_right_of_xiaofang :
  seatToRight xiaofangSeat = { column := 4, row := 5 } := by sorry

end NUMINAMATH_CALUDE_seat_to_right_of_xiaofang_l1108_110805


namespace NUMINAMATH_CALUDE_power_multiplication_l1108_110812

theorem power_multiplication (a : ℝ) : a^3 * a^6 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1108_110812


namespace NUMINAMATH_CALUDE_min_value_of_w_min_value_achievable_l1108_110854

theorem min_value_of_w (x y : ℝ) : 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 27 ≥ 81/4 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 27 = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_w_min_value_achievable_l1108_110854


namespace NUMINAMATH_CALUDE_initial_concentrated_kola_percentage_l1108_110804

/-- Proves that the initial percentage of concentrated kola in a 340-liter solution is 5% -/
theorem initial_concentrated_kola_percentage
  (initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_concentrated_kola : ℝ)
  (new_volume : ℝ)
  (new_sugar_percentage : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percentage = 88 / 100)
  (h3 : added_sugar = 3.2)
  (h4 : added_water = 10)
  (h5 : added_concentrated_kola = 6.8)
  (h6 : new_volume = initial_volume + added_sugar + added_water + added_concentrated_kola)
  (h7 : new_sugar_percentage = 7.5 / 100) :
  ∃ (initial_concentrated_kola_percentage : ℝ),
    initial_concentrated_kola_percentage = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_initial_concentrated_kola_percentage_l1108_110804


namespace NUMINAMATH_CALUDE_chinese_character_sum_l1108_110824

theorem chinese_character_sum : ∃! (a b c d e f g : ℕ),
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
   e ≠ f ∧ e ≠ g ∧
   f ≠ g) ∧
  (1000 * a + 100 * b + 10 * c + d + 100 * e + 10 * f + g = 2013) ∧
  (a + b + c + d + e + f + g = 24) :=
by sorry

end NUMINAMATH_CALUDE_chinese_character_sum_l1108_110824


namespace NUMINAMATH_CALUDE_jason_gave_nine_cards_l1108_110845

/-- The number of Pokemon cards Jason started with -/
def initial_cards : ℕ := 13

/-- The number of Pokemon cards Jason has left -/
def remaining_cards : ℕ := 4

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem jason_gave_nine_cards : cards_given = 9 := by sorry

end NUMINAMATH_CALUDE_jason_gave_nine_cards_l1108_110845


namespace NUMINAMATH_CALUDE_hall_length_l1108_110820

/-- Hall represents a rectangular hall with specific properties -/
structure Hall where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ

/-- Properties of the hall -/
def hall_properties (h : Hall) : Prop :=
  h.width = 15 ∧
  h.volume = 1687.5 ∧
  2 * (h.length * h.width) = 2 * (h.length * h.height) + 2 * (h.width * h.height)

/-- Theorem stating that a hall with the given properties has a length of 15 meters -/
theorem hall_length (h : Hall) (hp : hall_properties h) : h.length = 15 := by
  sorry

end NUMINAMATH_CALUDE_hall_length_l1108_110820


namespace NUMINAMATH_CALUDE_vaishalis_total_stripes_l1108_110813

/-- Represents the types of stripes on hats --/
inductive StripeType
  | Solid
  | Zigzag
  | Wavy
  | Other

/-- Represents a hat with its stripe count and type --/
structure Hat :=
  (stripeCount : ℕ)
  (stripeType : StripeType)

/-- Determines if a stripe type should be counted --/
def countStripe (st : StripeType) : Bool :=
  match st with
  | StripeType.Solid => true
  | StripeType.Zigzag => true
  | StripeType.Wavy => true
  | _ => false

/-- Calculates the total number of counted stripes for a list of hats --/
def totalCountedStripes (hats : List Hat) : ℕ :=
  hats.foldl (fun acc hat => 
    if countStripe hat.stripeType then
      acc + hat.stripeCount
    else
      acc
  ) 0

/-- Vaishali's hat collection --/
def vaishalisHats : List Hat := [
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 5, stripeType := StripeType.Zigzag },
  { stripeCount := 5, stripeType := StripeType.Zigzag },
  { stripeCount := 2, stripeType := StripeType.Wavy },
  { stripeCount := 2, stripeType := StripeType.Wavy },
  { stripeCount := 2, stripeType := StripeType.Wavy }
]

theorem vaishalis_total_stripes :
  totalCountedStripes vaishalisHats = 28 := by
  sorry

end NUMINAMATH_CALUDE_vaishalis_total_stripes_l1108_110813


namespace NUMINAMATH_CALUDE_function_value_given_cube_l1108_110816

theorem function_value_given_cube (x : ℝ) (h : x^3 = 8) :
  (x - 1) * (x + 1) * (x^2 + x + 1) = 21 := by
  sorry

end NUMINAMATH_CALUDE_function_value_given_cube_l1108_110816


namespace NUMINAMATH_CALUDE_percentage_problem_l1108_110888

theorem percentage_problem (P : ℝ) :
  (0.15 * 0.30 * (P / 100) * 4800 = 108) → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1108_110888


namespace NUMINAMATH_CALUDE_same_terminal_side_same_quadrant_angle_2010_in_third_quadrant_l1108_110844

-- Define the quadrants
inductive Quadrant
| first
| second
| third
| fourth

-- Define a function to determine the quadrant of an angle
def angle_quadrant (angle : ℝ) : Quadrant := sorry

-- Define the principle that angles with the same terminal side are in the same quadrant
theorem same_terminal_side_same_quadrant (angle1 angle2 : ℝ) :
  angle1 % 360 = angle2 % 360 → angle_quadrant angle1 = angle_quadrant angle2 := sorry

-- State the theorem
theorem angle_2010_in_third_quadrant :
  let angle_2010 : ℝ := 2010
  let angle_210 : ℝ := 210
  angle_2010 = 5 * 360 + angle_210 →
  angle_quadrant angle_210 = Quadrant.third →
  angle_quadrant angle_2010 = Quadrant.third := by
    sorry

end NUMINAMATH_CALUDE_same_terminal_side_same_quadrant_angle_2010_in_third_quadrant_l1108_110844


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1108_110860

/-- A regular polygon with perimeter 150 and side length 15 has 10 sides -/
theorem regular_polygon_sides (p : ℕ) (perimeter side_length : ℝ) 
  (h_regular : p ≥ 3)
  (h_perimeter : perimeter = 150)
  (h_side : side_length = 15)
  (h_relation : perimeter = p * side_length) : p = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1108_110860


namespace NUMINAMATH_CALUDE_octagons_700_sticks_4901_l1108_110883

/-- The number of sticks required to construct a series of octagons -/
def sticks_for_octagons (n : ℕ) : ℕ :=
  if n = 0 then 0
  else 8 + 7 * (n - 1)

/-- Theorem stating that 700 octagons require 4901 sticks -/
theorem octagons_700_sticks_4901 : sticks_for_octagons 700 = 4901 := by
  sorry

end NUMINAMATH_CALUDE_octagons_700_sticks_4901_l1108_110883


namespace NUMINAMATH_CALUDE_solve_equation_l1108_110877

theorem solve_equation (x : ℝ) : ((17.28 / x) / (3.6 * 0.2) = 2) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1108_110877


namespace NUMINAMATH_CALUDE_solve_for_y_l1108_110838

theorem solve_for_y (x y : ℝ) (h1 : x = 4) (h2 : 3 * x + 2 * y = 30) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1108_110838


namespace NUMINAMATH_CALUDE_complete_square_min_value_m_greater_n_right_triangle_l1108_110831

-- 1. Complete the square
theorem complete_square (x : ℝ) : x^2 - 4*x + 5 = (x - 2)^2 + 1 := by sorry

-- 2. Minimum value
theorem min_value : ∃ (m : ℝ), ∀ (x : ℝ), x^2 - 2*x + 3 ≥ m ∧ ∃ (y : ℝ), y^2 - 2*y + 3 = m := by sorry

-- 3. Relationship between M and N
theorem m_greater_n (a : ℝ) : a^2 - a > a - 2 := by sorry

-- 4. Triangle shape
theorem right_triangle (a b c : ℝ) : 
  a^2 + b^2 + c^2 - 6*a - 10*b - 8*c + 50 = 0 → 
  a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 := by sorry

end NUMINAMATH_CALUDE_complete_square_min_value_m_greater_n_right_triangle_l1108_110831


namespace NUMINAMATH_CALUDE_colins_class_girls_l1108_110819

theorem colins_class_girls (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 35 →
  boys > 15 →
  girls + boys = total →
  4 * girls = 3 * boys →
  girls = 15 :=
by sorry

end NUMINAMATH_CALUDE_colins_class_girls_l1108_110819


namespace NUMINAMATH_CALUDE_distinct_triangles_in_regular_ngon_l1108_110863

theorem distinct_triangles_in_regular_ngon (n : ℕ) :
  Nat.choose n 3 = (n * (n - 1) * (n - 2)) / 6 :=
sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_regular_ngon_l1108_110863


namespace NUMINAMATH_CALUDE_replacement_theorem_l1108_110834

/-- Calculates the percentage of chemicals in a solution after replacing part of it with a different solution -/
def resulting_solution_percentage (original_percentage : ℝ) (replacement_percentage : ℝ) (replaced_portion : ℝ) : ℝ :=
  let remaining_portion := 1 - replaced_portion
  let original_chemicals := original_percentage * remaining_portion
  let replacement_chemicals := replacement_percentage * replaced_portion
  (original_chemicals + replacement_chemicals) * 100

/-- Theorem stating that replacing half of an 80% solution with a 20% solution results in a 50% solution -/
theorem replacement_theorem :
  resulting_solution_percentage 0.8 0.2 0.5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_replacement_theorem_l1108_110834


namespace NUMINAMATH_CALUDE_zero_points_inequality_l1108_110850

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - (a / 2) * Real.log x

theorem zero_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x > 0 → f a x = 0 → x = x₁ ∨ x = x₂) →
  x₁ < x₂ →
  f a x₁ = 0 →
  f a x₂ = 0 →
  1 < x₁ ∧ x₁ < a ∧ a < x₂ ∧ x₂ < a^2 :=
by sorry

end NUMINAMATH_CALUDE_zero_points_inequality_l1108_110850


namespace NUMINAMATH_CALUDE_scout_weekend_earnings_l1108_110861

/-- Scout's weekend earnings calculation --/
theorem scout_weekend_earnings 
  (base_pay : ℝ) 
  (tip_per_customer : ℝ)
  (saturday_hours : ℝ) 
  (saturday_customers : ℕ)
  (sunday_hours : ℝ) 
  (sunday_customers : ℕ)
  (h1 : base_pay = 10)
  (h2 : tip_per_customer = 5)
  (h3 : saturday_hours = 4)
  (h4 : saturday_customers = 5)
  (h5 : sunday_hours = 5)
  (h6 : sunday_customers = 8) :
  base_pay * (saturday_hours + sunday_hours) + 
  tip_per_customer * (saturday_customers + sunday_customers) = 155 := by
sorry

end NUMINAMATH_CALUDE_scout_weekend_earnings_l1108_110861


namespace NUMINAMATH_CALUDE_pool_problem_l1108_110839

/-- Given a pool with humans and dogs, calculate the number of dogs -/
def number_of_dogs (total_legs_paws : ℕ) (num_humans : ℕ) (human_legs : ℕ) (dog_paws : ℕ) : ℕ :=
  ((total_legs_paws - (num_humans * human_legs)) / dog_paws)

theorem pool_problem :
  let total_legs_paws : ℕ := 24
  let num_humans : ℕ := 2
  let human_legs : ℕ := 2
  let dog_paws : ℕ := 4
  number_of_dogs total_legs_paws num_humans human_legs dog_paws = 5 := by
  sorry

end NUMINAMATH_CALUDE_pool_problem_l1108_110839


namespace NUMINAMATH_CALUDE_abs_neg_ten_eq_ten_l1108_110829

theorem abs_neg_ten_eq_ten : |(-10 : ℤ)| = 10 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_ten_eq_ten_l1108_110829


namespace NUMINAMATH_CALUDE_conic_eccentricity_l1108_110818

/-- A conic section with foci F₁ and F₂ -/
structure ConicSection where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a conic section -/
def eccentricity (c : ConicSection) : ℝ := sorry

/-- A point on a conic section -/
def Point (c : ConicSection) := ℝ × ℝ

theorem conic_eccentricity (c : ConicSection) :
  ∃ (P : Point c), 
    distance P c.F₁ / distance c.F₁ c.F₂ = 4/3 ∧
    distance c.F₁ c.F₂ / distance P c.F₂ = 3/2 →
    eccentricity c = 1/2 ∨ eccentricity c = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l1108_110818


namespace NUMINAMATH_CALUDE_division_value_proof_l1108_110892

theorem division_value_proof (x : ℝ) : (9 / x) * 12 = 18 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_value_proof_l1108_110892


namespace NUMINAMATH_CALUDE_parabola_directrix_l1108_110884

/-- Given a parabola y = ax^2 with directrix y = -2, prove that a = 1/8 -/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 ∧ y = -2 → a = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1108_110884


namespace NUMINAMATH_CALUDE_smallest_crate_dimension_l1108_110841

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate -/
def cylinderFitsInCrate (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.width) ∨
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.radius * 2 ≤ crate.height)

theorem smallest_crate_dimension (x : ℝ) :
  let crate := CrateDimensions.mk x 8 12
  let cylinder := Cylinder.mk 6 (max x (max 8 12))
  cylinderFitsInCrate crate cylinder →
  min x (min 8 12) = 8 := by
  sorry

#check smallest_crate_dimension

end NUMINAMATH_CALUDE_smallest_crate_dimension_l1108_110841


namespace NUMINAMATH_CALUDE_condition_relationship_l1108_110806

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a + b = 1 → 4 * a * b ≤ 1) ∧
  (∃ a b, 4 * a * b ≤ 1 ∧ a + b ≠ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l1108_110806


namespace NUMINAMATH_CALUDE_blueberry_pie_count_l1108_110823

theorem blueberry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) :
  total_pies = 36 →
  apple_ratio = 2 →
  blueberry_ratio = 5 →
  cherry_ratio = 3 →
  blueberry_ratio * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio) = 18 :=
by sorry

end NUMINAMATH_CALUDE_blueberry_pie_count_l1108_110823


namespace NUMINAMATH_CALUDE_max_diff_color_pairs_l1108_110897

/-- Represents a grid of black and white cells -/
structure Grid :=
  (size : Nat)
  (black_cells : Fin size → Fin size → Bool)

/-- The number of black cells in a given row -/
def row_black_count (g : Grid) (row : Fin g.size) : Nat :=
  (List.range g.size).count (λ col ↦ g.black_cells row col)

/-- The number of black cells in a given column -/
def col_black_count (g : Grid) (col : Fin g.size) : Nat :=
  (List.range g.size).count (λ row ↦ g.black_cells row col)

/-- The number of pairs of adjacent differently colored cells -/
def diff_color_pairs (g : Grid) : Nat :=
  sorry

/-- The theorem statement -/
theorem max_diff_color_pairs :
  ∃ (g : Grid),
    g.size = 100 ∧
    (∀ col₁ col₂ : Fin g.size, col_black_count g col₁ = col_black_count g col₂) ∧
    (∀ row₁ row₂ : Fin g.size, row₁ ≠ row₂ → row_black_count g row₁ ≠ row_black_count g row₂) ∧
    (∀ g' : Grid,
      g'.size = 100 →
      (∀ col₁ col₂ : Fin g'.size, col_black_count g' col₁ = col_black_count g' col₂) →
      (∀ row₁ row₂ : Fin g'.size, row₁ ≠ row₂ → row_black_count g' row₁ ≠ row_black_count g' row₂) →
      diff_color_pairs g' ≤ diff_color_pairs g) ∧
    diff_color_pairs g = 14601 :=
  sorry

end NUMINAMATH_CALUDE_max_diff_color_pairs_l1108_110897


namespace NUMINAMATH_CALUDE_maria_green_towels_l1108_110867

/-- The number of green towels Maria bought -/
def green_towels : ℕ := 35

/-- The number of white towels Maria bought -/
def white_towels : ℕ := 21

/-- The number of towels Maria gave to her mother -/
def towels_given : ℕ := 34

/-- The number of towels Maria ended up with -/
def towels_left : ℕ := 22

/-- Theorem stating that the number of green towels Maria bought is 35 -/
theorem maria_green_towels :
  green_towels = 35 ∧
  green_towels + white_towels - towels_given = towels_left :=
by sorry

end NUMINAMATH_CALUDE_maria_green_towels_l1108_110867


namespace NUMINAMATH_CALUDE_ferris_wheel_cost_calculation_l1108_110801

/-- The cost of a Ferris wheel ride in tickets -/
def ferris_wheel_cost : ℕ := sorry

/-- The number of Ferris wheel rides -/
def ferris_wheel_rides : ℕ := 2

/-- The cost of a roller coaster ride in tickets -/
def roller_coaster_cost : ℕ := 5

/-- The number of roller coaster rides -/
def roller_coaster_rides : ℕ := 3

/-- The cost of a log ride in tickets -/
def log_ride_cost : ℕ := 1

/-- The number of log rides -/
def log_ride_rides : ℕ := 7

/-- The initial number of tickets Dolly has -/
def initial_tickets : ℕ := 20

/-- The number of additional tickets Dolly buys -/
def additional_tickets : ℕ := 6

theorem ferris_wheel_cost_calculation :
  ferris_wheel_cost = 2 :=
sorry

end NUMINAMATH_CALUDE_ferris_wheel_cost_calculation_l1108_110801


namespace NUMINAMATH_CALUDE_library_shelves_l1108_110887

/-- Given a library with 14240 books and shelves that hold 8 books each, 
    the number of shelves required is 1780. -/
theorem library_shelves : 
  ∀ (total_books : ℕ) (books_per_shelf : ℕ),
    total_books = 14240 →
    books_per_shelf = 8 →
    total_books / books_per_shelf = 1780 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_l1108_110887


namespace NUMINAMATH_CALUDE_part_one_part_two_l1108_110893

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem part_one : 
  {x : ℝ | f 1 x ≥ 2} = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, a > 1 → (∀ x : ℝ, f a x + |x - 1| ≥ 2) ↔ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1108_110893


namespace NUMINAMATH_CALUDE_perpendicular_vector_proof_l1108_110862

def line_direction : ℝ × ℝ := (3, 2)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_vector_proof (v : ℝ × ℝ) :
  is_perpendicular v line_direction ∧ v.1 + v.2 = 1 → v = (-2, 3) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_proof_l1108_110862


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1108_110858

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- The original number of sides of the polygon -/
def original_sides : ℕ := 7

/-- The number of sides after doubling -/
def doubled_sides : ℕ := 2 * original_sides

theorem polygon_interior_angles_sum : 
  sum_interior_angles doubled_sides = 2160 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1108_110858


namespace NUMINAMATH_CALUDE_divide_twelve_by_repeating_third_l1108_110882

/-- The repeating decimal 0.3333... --/
def repeating_third : ℚ := 1 / 3

/-- The result of dividing 12 by the repeating decimal 0.3333... --/
theorem divide_twelve_by_repeating_third : 12 / repeating_third = 36 := by sorry

end NUMINAMATH_CALUDE_divide_twelve_by_repeating_third_l1108_110882


namespace NUMINAMATH_CALUDE_flour_already_added_l1108_110811

theorem flour_already_added (total_flour : ℕ) (flour_needed : ℕ) (flour_already_added : ℕ) : 
  total_flour = 9 → flour_needed = 6 → flour_already_added = total_flour - flour_needed → 
  flour_already_added = 3 := by
sorry

end NUMINAMATH_CALUDE_flour_already_added_l1108_110811


namespace NUMINAMATH_CALUDE_board_zeros_l1108_110815

theorem board_zeros (n : ℕ) (pos neg zero : ℕ) : 
  n = 10 → 
  pos + neg + zero = n → 
  pos * neg = 15 → 
  zero = 2 := by sorry

end NUMINAMATH_CALUDE_board_zeros_l1108_110815


namespace NUMINAMATH_CALUDE_star_1993_1935_l1108_110847

-- Define the operation *
def star (x y : ℤ) : ℤ := x - y

-- State the theorem
theorem star_1993_1935 : star 1993 1935 = 58 := by
  -- Assumptions
  have h1 : ∀ x : ℤ, star x x = 0 := by sorry
  have h2 : ∀ x y z : ℤ, star x (star y z) = star (star x y) z := by sorry
  
  -- Proof
  sorry

end NUMINAMATH_CALUDE_star_1993_1935_l1108_110847


namespace NUMINAMATH_CALUDE_green_hats_count_l1108_110859

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_price : ℚ) :
  total_hats = 85 →
  blue_cost = 6 →
  green_cost = 7 →
  total_price = 548 →
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_price ∧
    green_hats = 38 := by
  sorry

end NUMINAMATH_CALUDE_green_hats_count_l1108_110859


namespace NUMINAMATH_CALUDE_number_of_girls_in_college_l1108_110875

theorem number_of_girls_in_college (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) : 
  total_students = 240 → ratio_boys = 5 → ratio_girls = 7 → 
  (ratio_girls * total_students) / (ratio_boys + ratio_girls) = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_college_l1108_110875


namespace NUMINAMATH_CALUDE_ravi_mobile_price_l1108_110873

/-- The purchase price of Ravi's mobile phone -/
def mobile_price : ℝ :=
  -- Define the variable for the mobile phone price
  sorry

/-- The selling price of the refrigerator -/
def fridge_sell_price : ℝ :=
  15000 * (1 - 0.04)

/-- The selling price of the mobile phone -/
def mobile_sell_price : ℝ :=
  mobile_price * 1.10

/-- The total selling price of both items -/
def total_sell_price : ℝ :=
  fridge_sell_price + mobile_sell_price

/-- The total purchase price of both items plus profit -/
def total_purchase_plus_profit : ℝ :=
  15000 + mobile_price + 200

theorem ravi_mobile_price :
  (total_sell_price = total_purchase_plus_profit) →
  mobile_price = 6000 :=
by sorry

end NUMINAMATH_CALUDE_ravi_mobile_price_l1108_110873


namespace NUMINAMATH_CALUDE_cupboard_pricing_l1108_110835

/-- The cost price of a cupboard --/
def C : ℝ := sorry

/-- The selling price of the first cupboard --/
def SP₁ : ℝ := 0.84 * C

/-- The selling price of the second cupboard before tax --/
def SP₂ : ℝ := 0.756 * C

/-- The final selling price of the second cupboard after tax --/
def SP₂' : ℝ := 0.82404 * C

/-- The theorem stating the relationship between the cost price and the selling prices --/
theorem cupboard_pricing :
  2.32 * C - (SP₁ + SP₂') = 1800 :=
sorry

end NUMINAMATH_CALUDE_cupboard_pricing_l1108_110835


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1108_110870

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  (1 / x^2 - 2 / x - 3 < 0) ↔ (x < -1 ∨ x > 1/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1108_110870


namespace NUMINAMATH_CALUDE_barycentric_centroid_vector_relation_l1108_110836

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a triangle ABC, a point X with absolute barycentric coordinates (α:β:γ),
    and M as the centroid of triangle ABC, prove that:
    3 XM⃗ = (α - β)AB⃗ + (β - γ)BC⃗ + (γ - α)CA⃗ -/
theorem barycentric_centroid_vector_relation
  (A B C X M : V) (α β γ : ℝ) :
  X = α • A + β • B + γ • C →
  M = (1/3 : ℝ) • (A + B + C) →
  3 • (X - M) = (α - β) • (B - A) + (β - γ) • (C - B) + (γ - α) • (A - C) := by
  sorry

end NUMINAMATH_CALUDE_barycentric_centroid_vector_relation_l1108_110836


namespace NUMINAMATH_CALUDE_triangle_properties_l1108_110822

-- Define the triangle ABC
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  b = 4 → c = 5 → A = π / 3 →
  -- Properties to prove
  a = Real.sqrt 21 ∧ Real.sin (2 * B) = 4 * Real.sqrt 3 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1108_110822


namespace NUMINAMATH_CALUDE_expression_simplification_l1108_110827

theorem expression_simplification (a b : ℝ) 
  (ha : a = 3 + Real.sqrt 5) 
  (hb : b = 3 - Real.sqrt 5) : 
  ((a^2 - 2*a*b + b^2) / (a^2 - b^2)) * (a*b / (a - b)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1108_110827


namespace NUMINAMATH_CALUDE_inequality_proof_l1108_110866

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  2 * a * b * Real.log (b / a) < b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1108_110866


namespace NUMINAMATH_CALUDE_condo_units_per_floor_l1108_110880

/-- The number of units on each regular floor in a condo development -/
def units_per_regular_floor (total_floors : ℕ) (penthouse_floors : ℕ) (units_per_penthouse : ℕ) (total_units : ℕ) : ℕ :=
  (total_units - penthouse_floors * units_per_penthouse) / (total_floors - penthouse_floors)

/-- Theorem stating that the number of units on each regular floor is 12 -/
theorem condo_units_per_floor :
  units_per_regular_floor 23 2 2 256 = 12 := by
  sorry

end NUMINAMATH_CALUDE_condo_units_per_floor_l1108_110880


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1108_110872

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1108_110872


namespace NUMINAMATH_CALUDE_year_end_bonus_recipients_l1108_110881

/-- The total number of people receiving year-end bonuses in a company. -/
def total_award_recipients : ℕ := by sorry

/-- The amount of the first prize in ten thousands of yuan. -/
def first_prize : ℚ := 1.5

/-- The amount of the second prize in ten thousands of yuan. -/
def second_prize : ℚ := 1

/-- The amount of the third prize in ten thousands of yuan. -/
def third_prize : ℚ := 0.5

/-- The total bonus amount in ten thousands of yuan. -/
def total_bonus : ℚ := 100

theorem year_end_bonus_recipients :
  ∃ (x y z : ℕ),
    (x + y + z = total_award_recipients) ∧
    (first_prize * x + second_prize * y + third_prize * z = total_bonus) ∧
    (93 ≤ z - x) ∧ (z - x < 96) ∧
    (total_award_recipients = 147) := by sorry

end NUMINAMATH_CALUDE_year_end_bonus_recipients_l1108_110881


namespace NUMINAMATH_CALUDE_max_expression_l1108_110868

/-- A permutation of the digits 1 to 9 -/
def Digits := Fin 9 → Fin 9

/-- Check if a permutation is valid (bijective) -/
def is_valid_permutation (p : Digits) : Prop :=
  Function.Bijective p

/-- Convert three consecutive digits in a permutation to a number -/
def to_number (p : Digits) (start : Fin 9) : ℕ :=
  100 * (p start).val + 10 * (p (start + 1)).val + (p (start + 2)).val

/-- The expression to be maximized -/
def expression (p : Digits) : ℤ :=
  (to_number p 0 : ℤ) + (to_number p 3 : ℤ) - (to_number p 6 : ℤ)

/-- The main theorem -/
theorem max_expression :
  ∃ (p : Digits), is_valid_permutation p ∧ 
    (∀ (q : Digits), is_valid_permutation q → expression q ≤ expression p) ∧
    expression p = 1716 := by sorry

end NUMINAMATH_CALUDE_max_expression_l1108_110868


namespace NUMINAMATH_CALUDE_altitudes_sum_lt_perimeter_l1108_110856

/-- A triangle with side lengths and altitudes -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  altitude_positive : 0 < ha ∧ 0 < hb ∧ 0 < hc
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The sum of altitudes is less than the perimeter in any triangle -/
theorem altitudes_sum_lt_perimeter (t : Triangle) : t.ha + t.hb + t.hc < t.a + t.b + t.c := by
  sorry

end NUMINAMATH_CALUDE_altitudes_sum_lt_perimeter_l1108_110856


namespace NUMINAMATH_CALUDE_one_fourth_of_8_8_l1108_110865

theorem one_fourth_of_8_8 : 
  (8.8 : ℚ) / 4 = 11 / 5 := by sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_8_l1108_110865


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1108_110876

theorem mod_equivalence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 15827 [ZMOD 16] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1108_110876


namespace NUMINAMATH_CALUDE_range_of_m_l1108_110849

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 + (2*m - 3)*x₁ + 1 = 0 ∧ x₂^2 + (2*m - 3)*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∃ a b, a > 0 ∧ b > 0 ∧ a ≠ b ∧ ∀ x y, x^2/m + y^2/2 = 1 ↔ (x/a)^2 + (y/b)^2 = 1

-- Define the range of m
def m_range (m : ℝ) : Prop := m < 1/2 ∨ (2 < m ∧ m ≤ 5/2)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1108_110849


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l1108_110810

def g (x : ℝ) : ℝ := -x^3 + x^2 - x + 1

theorem polynomial_value_theorem :
  g 3 = 1 ∧ 12 * (-1) - 6 * 1 + 3 * (-1) - 1 = -22 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l1108_110810


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1108_110846

/-- An arithmetic sequence with a positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  ArithmeticSequence a d →
  (a 1 + a 2 + a 3 = 15) →
  (a 1 * a 2 * a 3 = 80) →
  (a 11 + a 12 + a 13 = 105) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1108_110846


namespace NUMINAMATH_CALUDE_tylers_dogs_l1108_110833

theorem tylers_dogs (puppies_per_dog : ℕ) (total_puppies : ℕ) (initial_dogs : ℕ) : 
  puppies_per_dog = 5 → 
  total_puppies = 75 → 
  initial_dogs * puppies_per_dog = total_puppies → 
  initial_dogs = 15 := by
sorry

end NUMINAMATH_CALUDE_tylers_dogs_l1108_110833


namespace NUMINAMATH_CALUDE_cake_comparison_l1108_110894

theorem cake_comparison : (1 : ℚ) / 3 > (1 : ℚ) / 4 ∧ (1 : ℚ) / 3 > (1 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_cake_comparison_l1108_110894


namespace NUMINAMATH_CALUDE_january_more_expensive_l1108_110800

/-- Represents the cost of purchasing screws and bolts in different months -/
structure CostComparison where
  january_screws_per_dollar : ℕ
  january_bolts_per_dollar : ℕ
  february_set_screws : ℕ
  february_set_bolts : ℕ
  february_set_price : ℕ
  tractor_screws : ℕ
  tractor_bolts : ℕ

/-- Calculates the cost of purchasing screws and bolts for a tractor in January -/
def january_cost (c : CostComparison) : ℚ :=
  (c.tractor_screws : ℚ) / c.january_screws_per_dollar + (c.tractor_bolts : ℚ) / c.january_bolts_per_dollar

/-- Calculates the cost of purchasing screws and bolts for a tractor in February -/
def february_cost (c : CostComparison) : ℚ :=
  (c.february_set_price : ℚ) * (max (c.tractor_screws / c.february_set_screws) (c.tractor_bolts / c.february_set_bolts))

/-- Theorem stating that the cost in January is higher than in February -/
theorem january_more_expensive (c : CostComparison) 
    (h1 : c.january_screws_per_dollar = 40)
    (h2 : c.january_bolts_per_dollar = 60)
    (h3 : c.february_set_screws = 25)
    (h4 : c.february_set_bolts = 25)
    (h5 : c.february_set_price = 1)
    (h6 : c.tractor_screws = 600)
    (h7 : c.tractor_bolts = 600) :
  january_cost c > february_cost c := by
  sorry

end NUMINAMATH_CALUDE_january_more_expensive_l1108_110800


namespace NUMINAMATH_CALUDE_fish_per_family_member_l1108_110843

def fish_distribution (family_size : ℕ) (eyes_eaten : ℕ) (eyes_to_dog : ℕ) (eyes_per_fish : ℕ) : ℕ :=
  let total_eyes := eyes_eaten + eyes_to_dog
  let total_fish := total_eyes / eyes_per_fish
  total_fish / family_size

theorem fish_per_family_member :
  fish_distribution 3 22 2 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fish_per_family_member_l1108_110843


namespace NUMINAMATH_CALUDE_percentage_relation_l1108_110821

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.08 * x) (h2 : b = 0.16 * x) :
  a = 0.5 * b := by sorry

end NUMINAMATH_CALUDE_percentage_relation_l1108_110821


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1108_110826

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 + 3 * x^3 - 5 * x^2 + 8 * x - 6) + (-6 * x^5 + x^3 + 4 * x^2 - 8 * x + 7) =
  -4 * x^5 + 4 * x^3 - x^2 + 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1108_110826


namespace NUMINAMATH_CALUDE_hippopotamus_cards_l1108_110895

theorem hippopotamus_cards (initial_cards remaining_cards : ℕ) : 
  initial_cards = 72 → remaining_cards = 11 → initial_cards - remaining_cards = 61 := by
  sorry

end NUMINAMATH_CALUDE_hippopotamus_cards_l1108_110895


namespace NUMINAMATH_CALUDE_residue_of_7_power_1234_mod_13_l1108_110832

theorem residue_of_7_power_1234_mod_13 : 7^1234 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_7_power_1234_mod_13_l1108_110832


namespace NUMINAMATH_CALUDE_special_triangle_smallest_angle_cos_l1108_110851

/-- A triangle with sides of three consecutive odd numbers where the largest angle is thrice the smallest angle -/
structure SpecialTriangle where
  n : ℕ
  side1 : ℕ := n + 2
  side2 : ℕ := n + 3
  side3 : ℕ := n + 4
  is_valid : side1 + side2 > side3 ∧ side1 + side3 > side2 ∧ side2 + side3 > side1
  largest_angle_triple : Real.cos ((n + 1) / (2 * (n + 2))) = 
    4 * ((n + 5) / (2 * (n + 4))) ^ 3 - 3 * ((n + 5) / (2 * (n + 4)))

/-- The cosine of the smallest angle in a SpecialTriangle is 6/11 -/
theorem special_triangle_smallest_angle_cos (t : SpecialTriangle) : 
  Real.cos ((t.n + 5) / (2 * (t.n + 4))) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_smallest_angle_cos_l1108_110851


namespace NUMINAMATH_CALUDE_train_station_distance_l1108_110855

theorem train_station_distance (speed1 speed2 : ℝ) (time_diff : ℝ) : 
  speed1 = 4 →
  speed2 = 5 →
  time_diff = 12 / 60 →
  (∃ d : ℝ, d / speed1 - d / speed2 = time_diff ∧ d = 4) :=
by sorry

end NUMINAMATH_CALUDE_train_station_distance_l1108_110855


namespace NUMINAMATH_CALUDE_triangle_trig_expression_l1108_110840

theorem triangle_trig_expression (D E F : Real) (h1 : 0 < D) (h2 : 0 < E) (h3 : 0 < F)
  (h4 : D + E + F = Real.pi) 
  (h5 : Real.sin F * 8 = 7) (h6 : Real.sin D * 5 = 8) (h7 : Real.sin E * 7 = 5) :
  (Real.cos ((D - E) / 2) / Real.sin (F / 2)) - (Real.sin ((D - E) / 2) / Real.cos (F / 2)) = 
  1 / Real.sqrt ((1 + Real.sqrt (15 / 64)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_expression_l1108_110840


namespace NUMINAMATH_CALUDE_patricks_class_size_l1108_110802

theorem patricks_class_size :
  ∃! b : ℕ, 100 < b ∧ b < 200 ∧
  ∃ k₁ : ℕ, b = 4 * k₁ - 2 ∧
  ∃ k₂ : ℕ, b = 6 * k₂ - 3 ∧
  ∃ k₃ : ℕ, b = 7 * k₃ - 4 :=
by sorry

end NUMINAMATH_CALUDE_patricks_class_size_l1108_110802


namespace NUMINAMATH_CALUDE_quadratic_circle_theorem_l1108_110886

/-- Quadratic function f(x) = x^2 + 2x + b --/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + b

/-- Circle equation C(x, y) = 0 --/
def C (b : ℝ) (x y : ℝ) : ℝ := x^2 + y^2 + 2*x - (b + 1)*y + b

theorem quadratic_circle_theorem (b : ℝ) (hb : b < 1 ∧ b ≠ 0) :
  /- The function intersects the coordinate axes at three points -/
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f b x₁ = 0 ∧ f b x₂ = 0) ∧
  (∃ y : ℝ, f b 0 = y) ∧
  /- The circle C passing through these three points has the given equation -/
  (∀ x y : ℝ, (f b x = y ∨ (x = 0 ∧ y = f b 0) ∨ (y = 0 ∧ f b x = 0)) → C b x y = 0) ∧
  /- Circle C passes through the fixed points (0, 1) and (-2, 1) for all valid b -/
  C b 0 1 = 0 ∧ C b (-2) 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_circle_theorem_l1108_110886


namespace NUMINAMATH_CALUDE_club_members_after_four_years_l1108_110828

/-- Represents the number of people in the club after k years -/
def club_members (k : ℕ) : ℕ :=
  if k = 0 then
    20
  else
    4 * club_members (k - 1) - 12

/-- The theorem stating the number of club members after 4 years -/
theorem club_members_after_four_years :
  club_members 4 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_club_members_after_four_years_l1108_110828


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1108_110890

/-- Given a square with perimeter 180 units divided into 3 congruent rectangles,
    prove that the perimeter of one rectangle is 120 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (num_rectangles : ℕ) : 
  square_perimeter = 180 →
  num_rectangles = 3 →
  let square_side := square_perimeter / 4
  let rect_length := square_side
  let rect_width := square_side / num_rectangles
  2 * (rect_length + rect_width) = 120 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1108_110890


namespace NUMINAMATH_CALUDE_base_8_to_10_reverse_digits_l1108_110807

theorem base_8_to_10_reverse_digits : ∃ (d e f : ℕ), 
  (0 ≤ d ∧ d ≤ 7) ∧ 
  (0 ≤ e ∧ e ≤ 7) ∧ 
  (0 ≤ f ∧ f ≤ 7) ∧ 
  e = 3 ∧
  (64 * d + 8 * e + f = 100 * f + 10 * e + d) := by
  sorry

end NUMINAMATH_CALUDE_base_8_to_10_reverse_digits_l1108_110807


namespace NUMINAMATH_CALUDE_root_equation_value_l1108_110879

theorem root_equation_value (m : ℝ) : 
  m^2 + m - 1 = 0 → 3*m^2 + 3*m + 2006 = 2009 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l1108_110879


namespace NUMINAMATH_CALUDE_at_operation_difference_l1108_110878

def at_operation (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem at_operation_difference : at_operation 5 3 - at_operation 3 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_at_operation_difference_l1108_110878


namespace NUMINAMATH_CALUDE_min_production_avoids_loss_min_production_is_minimal_l1108_110885

/-- The daily production cost function for a shoe factory -/
def cost (n : ℕ) : ℝ := 4000 + 50 * n

/-- The daily revenue function for a shoe factory -/
def revenue (n : ℕ) : ℝ := 90 * n

/-- The daily profit function for a shoe factory -/
def profit (n : ℕ) : ℝ := revenue n - cost n

/-- The minimum number of pairs of shoes that must be produced daily to avoid loss -/
def min_production : ℕ := 100

theorem min_production_avoids_loss :
  ∀ n : ℕ, n ≥ min_production → profit n ≥ 0 :=
sorry

theorem min_production_is_minimal :
  ∀ m : ℕ, (∀ n : ℕ, n ≥ m → profit n ≥ 0) → m ≥ min_production :=
sorry

end NUMINAMATH_CALUDE_min_production_avoids_loss_min_production_is_minimal_l1108_110885


namespace NUMINAMATH_CALUDE_papi_calot_plants_l1108_110809

/-- The number of plants Papi Calot needs to buy -/
def total_plants (rows : ℕ) (plants_per_row : ℕ) (additional_plants : ℕ) : ℕ :=
  rows * plants_per_row + additional_plants

/-- Proof that Papi Calot needs to buy 141 plants -/
theorem papi_calot_plants : total_plants 7 18 15 = 141 := by
  sorry

end NUMINAMATH_CALUDE_papi_calot_plants_l1108_110809
