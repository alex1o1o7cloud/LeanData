import Mathlib

namespace zack_traveled_18_countries_l2059_205929

/-- The number of countries George traveled to -/
def george_countries : ℕ := 6

/-- The number of countries Joseph traveled to -/
def joseph_countries : ℕ := george_countries / 2

/-- The number of countries Patrick traveled to -/
def patrick_countries : ℕ := 3 * joseph_countries

/-- The number of countries Zack traveled to -/
def zack_countries : ℕ := 2 * patrick_countries

/-- Proof that Zack traveled to 18 countries -/
theorem zack_traveled_18_countries : zack_countries = 18 := by
  sorry

end zack_traveled_18_countries_l2059_205929


namespace sallys_pears_l2059_205936

theorem sallys_pears (sara_pears : ℕ) (total_pears : ℕ) (sally_pears : ℕ) :
  sara_pears = 45 →
  total_pears = 56 →
  sally_pears = total_pears - sara_pears →
  sally_pears = 11 := by
  sorry

end sallys_pears_l2059_205936


namespace min_value_a_plus_b_min_value_achieved_l2059_205923

/-- The function f(x) = |2x-1| - m -/
def f (x m : ℝ) : ℝ := |2*x - 1| - m

/-- The theorem stating the minimum value of a + b -/
theorem min_value_a_plus_b (m : ℝ) (h1 : Set.Icc 0 1 = {x | f x m ≤ 0}) 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h2 : 1/a + 1/(2*b) = m) : 
  a + b ≥ 3/2 + Real.sqrt 2 := by
  sorry

/-- The theorem stating that the minimum value is achieved -/
theorem min_value_achieved (m : ℝ) (h1 : Set.Icc 0 1 = {x | f x m ≤ 0}) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1/a + 1/(2*b) = m ∧ a + b = 3/2 + Real.sqrt 2 := by
  sorry

end min_value_a_plus_b_min_value_achieved_l2059_205923


namespace intersection_complement_equality_l2059_205985

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 2}

-- Define set B
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem intersection_complement_equality :
  A ∩ (U \ B) = {1} := by sorry

end intersection_complement_equality_l2059_205985


namespace pen_discount_theorem_l2059_205994

theorem pen_discount_theorem (marked_price : ℝ) :
  let purchase_quantity : ℕ := 60
  let purchase_price_in_pens : ℕ := 46
  let profit_percent : ℝ := 29.130434782608695

  let cost_price : ℝ := marked_price * purchase_price_in_pens
  let selling_price : ℝ := cost_price * (1 + profit_percent / 100)
  let selling_price_per_pen : ℝ := selling_price / purchase_quantity
  let discount : ℝ := marked_price - selling_price_per_pen
  let discount_percent : ℝ := (discount / marked_price) * 100

  discount_percent = 1 := by sorry

end pen_discount_theorem_l2059_205994


namespace average_of_r_s_t_l2059_205912

theorem average_of_r_s_t (r s t : ℝ) (h : (5 / 4) * (r + s + t - 2) = 15) : 
  (r + s + t) / 3 = 14 / 3 := by
sorry

end average_of_r_s_t_l2059_205912


namespace bricks_to_paint_theorem_l2059_205934

/-- Represents a stack of bricks -/
structure BrickStack :=
  (height : ℕ)
  (width : ℕ)
  (depth : ℕ)
  (total_bricks : ℕ)
  (sides_against_wall : ℕ)

/-- Calculates the number of bricks that need to be painted on their exposed surfaces -/
def bricks_to_paint (stack : BrickStack) : ℕ :=
  let front_face := stack.height * stack.width + stack.depth
  let top_face := stack.width * stack.depth
  front_face * stack.height + top_face * (4 - stack.sides_against_wall)

theorem bricks_to_paint_theorem (stack : BrickStack) :
  stack.height = 4 ∧ 
  stack.width = 3 ∧ 
  stack.depth = 15 ∧ 
  stack.total_bricks = 180 ∧ 
  stack.sides_against_wall = 2 →
  bricks_to_paint stack = 96 :=
by sorry

end bricks_to_paint_theorem_l2059_205934


namespace james_muffins_l2059_205969

theorem james_muffins (arthur_muffins : Float) (ratio : Float) (james_muffins : Float)
  (h1 : arthur_muffins = 115.0)
  (h2 : ratio = 12.0)
  (h3 : arthur_muffins = ratio * james_muffins) :
  james_muffins = arthur_muffins / ratio := by
sorry

end james_muffins_l2059_205969


namespace flower_bed_fraction_l2059_205911

-- Define the dimensions of the yard
def yard_length : ℝ := 30
def yard_width : ℝ := 6

-- Define the lengths of the parallel sides of the trapezoidal remainder
def trapezoid_long_side : ℝ := 30
def trapezoid_short_side : ℝ := 20

-- Define the fraction we want to prove
def target_fraction : ℚ := 5/36

-- Theorem statement
theorem flower_bed_fraction :
  let yard_area := yard_length * yard_width
  let triangle_leg := (trapezoid_long_side - trapezoid_short_side) / 2
  let triangle_area := triangle_leg^2 / 2
  let flower_beds_area := 2 * triangle_area
  flower_beds_area / yard_area = target_fraction := by
  sorry

end flower_bed_fraction_l2059_205911


namespace volume_of_specific_open_box_l2059_205926

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutLength : ℝ) : ℝ :=
  (sheetLength - 2 * cutLength) * (sheetWidth - 2 * cutLength) * cutLength

/-- Theorem stating that the volume of the open box is 5440 m³ given the specified dimensions. -/
theorem volume_of_specific_open_box :
  openBoxVolume 50 36 8 = 5440 := by
  sorry

#eval openBoxVolume 50 36 8

end volume_of_specific_open_box_l2059_205926


namespace cube_root_inequality_l2059_205973

theorem cube_root_inequality (x : ℝ) : 
  x > 0 → (x^(1/3) < 3*x ↔ x > 1/(3*(3^(1/2)))) :=
by sorry

end cube_root_inequality_l2059_205973


namespace max_value_of_linear_function_max_value_achieved_l2059_205959

-- Define the linear function
def f (x : ℝ) : ℝ := -x + 3

-- State the theorem
theorem max_value_of_linear_function :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x ≤ 3 :=
by
  sorry

-- State that the maximum is achieved
theorem max_value_achieved :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = 3 :=
by
  sorry

end max_value_of_linear_function_max_value_achieved_l2059_205959


namespace exterior_angle_measure_l2059_205947

-- Define the nonagon's interior angle
def nonagon_interior_angle : ℝ := 140

-- Define the nonagon's exterior angle
def nonagon_exterior_angle : ℝ := 360 - nonagon_interior_angle

-- Define the square's interior angle
def square_interior_angle : ℝ := 90

-- Theorem statement
theorem exterior_angle_measure :
  nonagon_exterior_angle - square_interior_angle = 130 := by
  sorry

end exterior_angle_measure_l2059_205947


namespace jerry_shelf_difference_l2059_205980

/-- The difference between action figures and books on Jerry's shelf -/
def shelf_difference (initial_figures : ℕ) (initial_books : ℕ) (added_books : ℕ) : ℤ :=
  (initial_figures : ℤ) - ((initial_books : ℤ) + (added_books : ℤ))

/-- Theorem stating the difference between action figures and books on Jerry's shelf -/
theorem jerry_shelf_difference :
  shelf_difference 7 2 4 = 1 := by
  sorry

end jerry_shelf_difference_l2059_205980


namespace base_prime_441_l2059_205946

/-- Definition of base prime representation for a natural number -/
def basePrimeRepresentation (n : ℕ) : List ℕ :=
  sorry

/-- Theorem stating that the base prime representation of 441 is [0, 2, 2, 0] -/
theorem base_prime_441 : basePrimeRepresentation 441 = [0, 2, 2, 0] := by
  sorry

end base_prime_441_l2059_205946


namespace lines_are_parallel_l2059_205920

-- Define the lines
def line1 (a : ℝ) (θ : ℝ) : Prop := θ = a
def line2 (p a θ : ℝ) : Prop := p * Real.sin (θ - a) = 1

-- Theorem statement
theorem lines_are_parallel (a p : ℝ) : 
  ∀ θ, ¬(line1 a θ ∧ line2 p a θ) :=
sorry

end lines_are_parallel_l2059_205920


namespace fraction_change_theorem_l2059_205918

theorem fraction_change_theorem (a b c d e f x y : ℚ) 
  (h1 : a ≠ b) (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + y) = c / d) 
  (h4 : (a + 2*x) / (b + 2*y) = e / f) 
  (h5 : d ≠ c) (h6 : f ≠ e) : 
  x = (b*c - a*d) / (d - c) ∧ 
  y = (b*e - a*f) / (2*f - 2*e) := by
sorry

end fraction_change_theorem_l2059_205918


namespace largest_prime_divisor_to_test_l2059_205995

theorem largest_prime_divisor_to_test (n : ℕ) (h : 500 ≤ n ∧ n ≤ 550) :
  (∀ p : ℕ, p.Prime → p ≤ 23 → ¬(p ∣ n)) → n.Prime ∨ n = 1 :=
sorry

end largest_prime_divisor_to_test_l2059_205995


namespace one_by_one_tile_position_l2059_205901

/-- Represents a tile with width and height -/
structure Tile where
  width : ℕ
  height : ℕ

/-- Represents a square grid -/
structure Square where
  side_length : ℕ

/-- Represents the position of a tile in the square -/
structure TilePosition where
  row : ℕ
  col : ℕ

/-- Checks if a position is in the center or adjacent to the boundary of the square -/
def is_center_or_adjacent_boundary (pos : TilePosition) (square : Square) : Prop :=
  (pos.row = square.side_length / 2 + 1 ∧ pos.col = square.side_length / 2 + 1) ∨
  (pos.row = 1 ∨ pos.row = square.side_length ∨ pos.col = 1 ∨ pos.col = square.side_length)

/-- Theorem: In a 7x7 square formed by sixteen 1x3 tiles and one 1x1 tile,
    the 1x1 tile must be either in the center or adjacent to the boundary -/
theorem one_by_one_tile_position
  (square : Square)
  (large_tiles : Finset Tile)
  (small_tile : Tile)
  (tile_arrangement : Square → Finset Tile → Tile → TilePosition) :
  square.side_length = 7 →
  large_tiles.card = 16 →
  (∀ t ∈ large_tiles, t.width = 1 ∧ t.height = 3) →
  small_tile.width = 1 ∧ small_tile.height = 1 →
  is_center_or_adjacent_boundary (tile_arrangement square large_tiles small_tile) square :=
by sorry

end one_by_one_tile_position_l2059_205901


namespace min_price_with_profit_margin_l2059_205935

theorem min_price_with_profit_margin (marked_price : ℝ) (markup_percentage : ℝ) (min_profit_margin : ℝ) : 
  marked_price = 240 →
  markup_percentage = 0.6 →
  min_profit_margin = 0.1 →
  let cost_price := marked_price / (1 + markup_percentage)
  let min_reduced_price := cost_price * (1 + min_profit_margin)
  min_reduced_price = 165 :=
by sorry

end min_price_with_profit_margin_l2059_205935


namespace perpendicular_vectors_l2059_205963

/-- Given vectors a, b, and c in ℝ², prove that if a is perpendicular to (b - c), then x = 4/3 -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (3, 1)) 
  (h2 : b = (x, -2)) 
  (h3 : c = (0, 2)) 
  (h4 : a • (b - c) = 0) : 
  x = 4/3 := by
  sorry

#check perpendicular_vectors

end perpendicular_vectors_l2059_205963


namespace smallest_sum_of_squares_l2059_205927

theorem smallest_sum_of_squares (x y : ℕ) : x^2 - y^2 = 221 → x^2 + y^2 ≥ 229 := by
  sorry

end smallest_sum_of_squares_l2059_205927


namespace sphere_surface_area_rectangular_solid_l2059_205997

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 200 * Real.pi :=
by sorry

end sphere_surface_area_rectangular_solid_l2059_205997


namespace systematic_sampling_18th_group_l2059_205981

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) (groupNumber : ℕ) : ℕ :=
  firstSelected + (groupNumber - 1) * (totalStudents / sampleSize)

/-- Theorem: Systematic sampling selects student number 872 in the 18th group -/
theorem systematic_sampling_18th_group :
  let totalStudents : ℕ := 1000
  let sampleSize : ℕ := 20
  let groupSize : ℕ := totalStudents / sampleSize
  let thirdGroupStart : ℕ := 2 * groupSize
  let selectedInThirdGroup : ℕ := 122
  let firstSelected : ℕ := selectedInThirdGroup - thirdGroupStart
  systematicSample totalStudents sampleSize firstSelected 18 = 872 := by
sorry

end systematic_sampling_18th_group_l2059_205981


namespace arrangements_not_adjacent_l2059_205992

theorem arrangements_not_adjacent (n : ℕ) (h : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := by
  sorry

end arrangements_not_adjacent_l2059_205992


namespace stone_skipping_l2059_205915

theorem stone_skipping (throw1 throw2 throw3 throw4 throw5 : ℕ) : 
  throw5 = 8 ∧ 
  throw2 = throw1 + 2 ∧ 
  throw3 = 2 * throw2 ∧ 
  throw4 = throw3 - 3 ∧ 
  throw5 = throw4 + 1 →
  throw1 + throw2 + throw3 + throw4 + throw5 = 33 := by
  sorry

end stone_skipping_l2059_205915


namespace rachel_book_count_l2059_205924

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 6

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 2

/-- The total number of books Rachel has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem rachel_book_count : total_books = 72 := by
  sorry

end rachel_book_count_l2059_205924


namespace complex_number_simplification_l2059_205968

theorem complex_number_simplification :
  3 * (2 - 5 * Complex.I) - 4 * (1 + 3 * Complex.I) = 2 - 27 * Complex.I :=
by sorry

end complex_number_simplification_l2059_205968


namespace quadratic_root_theorem_l2059_205977

theorem quadratic_root_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hroot : a * 2^2 - (a + b + c) * 2 + (b + c) = 0) :
  ∃ x : ℝ, x ≠ 2 ∧ a * x^2 - (a + b + c) * x + (b + c) = 0 ∧ x = (b + c - a) / a :=
sorry

end quadratic_root_theorem_l2059_205977


namespace negative_seven_in_A_l2059_205996

def A : Set ℤ := {1, -7}

theorem negative_seven_in_A : -7 ∈ A := by
  sorry

end negative_seven_in_A_l2059_205996


namespace stream_speed_l2059_205965

/-- The speed of a stream given downstream and upstream speeds -/
theorem stream_speed (downstream_speed upstream_speed : ℝ) :
  downstream_speed = 15 →
  upstream_speed = 8 →
  (downstream_speed - upstream_speed) / 2 = 3.5 := by
  sorry

end stream_speed_l2059_205965


namespace brian_stones_l2059_205931

theorem brian_stones (total : ℕ) (grey : ℕ) (green : ℕ) (white : ℕ) (black : ℕ) : 
  total = 100 →
  grey = 40 →
  green = 60 →
  grey + green = total →
  white + black = total →
  (white : ℚ) / total = (green : ℚ) / total →
  white > black →
  white = 60 := by
sorry

end brian_stones_l2059_205931


namespace billion_scientific_notation_l2059_205932

/-- Represents the value of one billion -/
def billion : ℝ := 10^9

/-- The given amount in billions -/
def amount : ℝ := 4.15

theorem billion_scientific_notation : 
  amount * billion = 4.15 * 10^9 := by sorry

end billion_scientific_notation_l2059_205932


namespace min_value_when_a_is_one_range_of_a_l2059_205916

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Part 1: Minimum value when a = 1
theorem min_value_when_a_is_one :
  ∃ (m : ℝ), ∀ (x : ℝ), f 1 x ≥ m ∧ ∃ (y : ℝ), f 1 y = m ∧ m = 1 :=
sorry

-- Part 2: Range of a for f(x) ≥ a when x ∈ [-1, +∞)
theorem range_of_a :
  ∀ (a : ℝ), (∀ (x : ℝ), x ≥ -1 → f a x ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 :=
sorry

end min_value_when_a_is_one_range_of_a_l2059_205916


namespace real_part_of_complex_fraction_l2059_205939

theorem real_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  (Complex.re ((1 : ℂ) + i) / i) = 1 := by
  sorry

end real_part_of_complex_fraction_l2059_205939


namespace coins_sum_theorem_l2059_205987

theorem coins_sum_theorem (stack1 stack2 stack3 stack4 : ℕ) 
  (h1 : stack1 = 12)
  (h2 : stack2 = 17)
  (h3 : stack3 = 23)
  (h4 : stack4 = 8) :
  stack1 + stack2 + stack3 + stack4 = 60 := by
  sorry

end coins_sum_theorem_l2059_205987


namespace five_balls_two_boxes_l2059_205979

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_balls + 1

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 2 distinguishable boxes -/
theorem five_balls_two_boxes : distribute_balls 5 2 = 6 := by
  sorry

end five_balls_two_boxes_l2059_205979


namespace double_reflection_of_D_l2059_205989

/-- Reflects a point across the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point across the line y = -x + 1 -/
def reflect_diagonal (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2 + 1, -p.1 + 1)

/-- The main theorem -/
theorem double_reflection_of_D (D : ℝ × ℝ) (hD : D = (5, 2)) :
  (reflect_diagonal ∘ reflect_x_axis) D = (-3, -4) := by
  sorry

end double_reflection_of_D_l2059_205989


namespace product_equals_10000_l2059_205921

theorem product_equals_10000 : ∃ x : ℕ, 469160 * x = 4691130840 ∧ x = 10000 := by
  sorry

end product_equals_10000_l2059_205921


namespace circle_area_ratio_after_tripling_diameter_l2059_205964

theorem circle_area_ratio_after_tripling_diameter :
  ∀ (r : ℝ), r > 0 →
  (π * r^2) / (π * (3*r)^2) = 1/9 := by
sorry

end circle_area_ratio_after_tripling_diameter_l2059_205964


namespace set_equality_l2059_205903

open Set Real

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | Real.log (x - 2) ≤ 0}

theorem set_equality : (Aᶜ ∪ B) = Icc (-1) 3 := by sorry

end set_equality_l2059_205903


namespace circle_polar_equation_l2059_205960

/-- The polar equation ρ = 2a cos θ represents a circle with center C(a, 0) and radius a -/
theorem circle_polar_equation (a : ℝ) :
  ∀ ρ θ : ℝ, ρ = 2 * a * Real.cos θ ↔ 
  ∃ x y : ℝ, (x - a)^2 + y^2 = a^2 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end circle_polar_equation_l2059_205960


namespace flight_passenger_distribution_l2059_205943

/-- Proof of the flight passenger distribution problem -/
theorem flight_passenger_distribution
  (total_passengers : ℕ)
  (female_percentage : ℚ)
  (first_class_male_ratio : ℚ)
  (coach_females : ℕ)
  (h1 : total_passengers = 120)
  (h2 : female_percentage = 30 / 100)
  (h3 : first_class_male_ratio = 1 / 3)
  (h4 : coach_females = 28)
  : ∃ (first_class_percentage : ℚ), first_class_percentage = 30 / 100 := by
  sorry

end flight_passenger_distribution_l2059_205943


namespace intersection_complement_equality_l2059_205933

universe u

def U : Set (Fin 6) := {1, 2, 3, 4, 5, 6}
def P : Set (Fin 6) := {1, 2, 3, 4}
def Q : Set (Fin 6) := {3, 4, 5, 6}

theorem intersection_complement_equality :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_complement_equality_l2059_205933


namespace quadratic_roots_ratio_l2059_205970

theorem quadratic_roots_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end quadratic_roots_ratio_l2059_205970


namespace rainy_days_exist_l2059_205993

/-- Represents the number of rainy days given the conditions of Mo's drinking habits -/
def rainy_days (n d T H P : ℤ) : Prop :=
  ∃ (R : ℤ),
    (1 ≤ d) ∧ (d ≤ 31) ∧
    (T = 3 * (d - R)) ∧
    (H = n * R) ∧
    (T = H + P) ∧
    (R = (3 * d - P) / (n + 3)) ∧
    (0 ≤ R) ∧ (R ≤ d)

/-- Theorem stating the existence of R satisfying the conditions -/
theorem rainy_days_exist (n d T H P : ℤ) (h1 : 1 ≤ d) (h2 : d ≤ 31) 
  (h3 : T = 3 * (d - (3 * d - P) / (n + 3))) 
  (h4 : H = n * ((3 * d - P) / (n + 3))) 
  (h5 : T = H + P)
  (h6 : (3 * d - P) % (n + 3) = 0)
  (h7 : 0 ≤ (3 * d - P) / (n + 3))
  (h8 : (3 * d - P) / (n + 3) ≤ d) :
  rainy_days n d T H P :=
by
  sorry


end rainy_days_exist_l2059_205993


namespace flagpole_break_height_l2059_205988

/-- 
Given a flagpole of height 8 meters that breaks such that the upper part touches the ground 3 meters from the base, 
the height from the ground to the break point is √73/2 meters.
-/
theorem flagpole_break_height (h : ℝ) (d : ℝ) (x : ℝ) 
  (h_height : h = 8) 
  (d_distance : d = 3) 
  (x_def : x = h - (h^2 - d^2).sqrt / 2) : 
  x = Real.sqrt 73 / 2 := by
  sorry

end flagpole_break_height_l2059_205988


namespace average_brown_mms_l2059_205910

def brown_mms : List Nat := [9, 12, 8, 8, 3]

theorem average_brown_mms :
  (brown_mms.sum / brown_mms.length : ℚ) = 8 := by
  sorry

end average_brown_mms_l2059_205910


namespace parabola_intersection_l2059_205978

theorem parabola_intersection
  (a h d : ℝ) (ha : a ≠ 0) :
  let f (x : ℝ) := a * (x - h)^2 + d
  let g (x : ℝ) := a * ((x + 3) - h)^2 + d
  ∃! x, f x = g x ∧ x = -3/2 :=
sorry

end parabola_intersection_l2059_205978


namespace total_weight_of_fruits_l2059_205967

-- Define the weight of oranges and apples
def orange_weight : ℚ := 24 / 12
def apple_weight : ℚ := 30 / 8

-- Define the number of bags for each fruit
def orange_bags : ℕ := 5
def apple_bags : ℕ := 4

-- Theorem to prove
theorem total_weight_of_fruits :
  orange_bags * orange_weight + apple_bags * apple_weight = 25 := by
  sorry

end total_weight_of_fruits_l2059_205967


namespace x_equals_one_ninth_l2059_205937

theorem x_equals_one_ninth (x : ℚ) (h : x - 1/10 = x/10) : x = 1/9 := by
  sorry

end x_equals_one_ninth_l2059_205937


namespace coloring_scheme_satisfies_conditions_l2059_205948

/-- Represents the three colors used in the coloring scheme. -/
inductive Color
  | White
  | Red
  | Blue

/-- The coloring function that assigns a color to each integral point in the plane. -/
def f : ℤ × ℤ → Color :=
  sorry

/-- Represents an infinite set of integers. -/
def InfiniteSet (s : Set ℤ) : Prop :=
  ∀ n : ℤ, ∃ m ∈ s, m > n

theorem coloring_scheme_satisfies_conditions :
  (∀ c : Color, InfiniteSet {k : ℤ | InfiniteSet {n : ℤ | f (n, k) = c}}) ∧
  (∀ a b c : ℤ × ℤ,
    f a = Color.White → f b = Color.Red → f c = Color.Blue →
    ∃ d : ℤ × ℤ, f d = Color.Red ∧ d = (a.1 + c.1 - b.1, a.2 + c.2 - b.2)) :=
by
  sorry

end coloring_scheme_satisfies_conditions_l2059_205948


namespace completing_square_sum_l2059_205975

theorem completing_square_sum (a b c : ℤ) : 
  (∀ x : ℝ, 49 * x^2 + 70 * x - 121 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 158 := by
  sorry

end completing_square_sum_l2059_205975


namespace two_part_journey_average_speed_l2059_205938

/-- Calculates the average speed of a two-part journey -/
theorem two_part_journey_average_speed 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  distance1 = 360 →
  speed1 = 60 →
  distance2 = 120 →
  speed2 = 40 →
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 480 / 9 :=
by
  sorry

#eval (480 : ℚ) / 9

end two_part_journey_average_speed_l2059_205938


namespace rectangles_in_five_by_five_grid_l2059_205928

/-- The number of dots in each row and column of the square array -/
def gridSize : ℕ := 5

/-- The number of different rectangles that can be formed in a square array of dots -/
def numRectangles (n : ℕ) : ℕ := (n.choose 2) * (n.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_five_by_five_grid :
  numRectangles gridSize = 100 := by
  sorry

end rectangles_in_five_by_five_grid_l2059_205928


namespace range_of_m_l2059_205983

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function has a minimum positive period of p if f(x + p) = f(x) for all x,
    and p is the smallest positive number with this property -/
def HasMinPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  (∀ x, f (x + p) = f x) ∧ ∀ q, 0 < q → q < p → ∃ x, f (x + q) ≠ f x

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
    (h_odd : IsOdd f)
    (h_period : HasMinPeriod f 3)
    (h_2015 : f 2015 > 1)
    (h_1 : f 1 = (2*m + 3)/(m - 1)) :
    -2/3 < m ∧ m < 1 := by
  sorry

end range_of_m_l2059_205983


namespace salary_increase_after_five_years_l2059_205952

theorem salary_increase_after_five_years (annual_raise : Real) 
  (h1 : annual_raise = 0.15) : 
  (1 + annual_raise)^5 > 2 := by
  sorry

#check salary_increase_after_five_years

end salary_increase_after_five_years_l2059_205952


namespace heaviest_tv_weight_difference_l2059_205961

-- Define the dimensions and weight ratios of the TVs
def bill_width : ℝ := 48
def bill_height : ℝ := 100
def bill_weight_ratio : ℝ := 4

def bob_width : ℝ := 70
def bob_height : ℝ := 60
def bob_weight_ratio : ℝ := 3.5

def steve_width : ℝ := 84
def steve_height : ℝ := 92
def steve_weight_ratio : ℝ := 4.5

-- Define the conversion factor from ounces to pounds
def oz_to_lb : ℝ := 16

-- Theorem to prove
theorem heaviest_tv_weight_difference : 
  let bill_area := bill_width * bill_height
  let bob_area := bob_width * bob_height
  let steve_area := steve_width * steve_height
  
  let bill_weight := bill_area * bill_weight_ratio / oz_to_lb
  let bob_weight := bob_area * bob_weight_ratio / oz_to_lb
  let steve_weight := steve_area * steve_weight_ratio / oz_to_lb
  
  let heaviest_weight := max bill_weight (max bob_weight steve_weight)
  let combined_weight := bill_weight + bob_weight
  
  heaviest_weight - combined_weight = 54.75 := by
  sorry

end heaviest_tv_weight_difference_l2059_205961


namespace ball_returns_after_15_throws_l2059_205956

/-- Represents the number of girls to skip in each throw -/
def skip_pattern : ℕ → ℕ
  | n => if n % 2 = 0 then 3 else 4

/-- Calculates the position of the girl who receives the ball after n throws -/
def ball_position (n : ℕ) : Fin 15 :=
  (List.range n).foldl (fun pos _ => 
    (pos + skip_pattern pos + 1 : Fin 15)) 0

theorem ball_returns_after_15_throws :
  ball_position 15 = 0 := by sorry

end ball_returns_after_15_throws_l2059_205956


namespace greatest_x_value_l2059_205976

theorem greatest_x_value : ∃ (x : ℤ), (∀ (y : ℤ), 2.134 * (10 : ℝ) ^ y < 21000 → y ≤ x) ∧ 2.134 * (10 : ℝ) ^ x < 21000 ∧ x = 3 := by
  sorry

end greatest_x_value_l2059_205976


namespace average_annual_growth_rate_l2059_205909

theorem average_annual_growth_rate 
  (p q : ℝ) 
  (hp : p > -1) 
  (hq : q > -1) :
  ∃ x : ℝ, x > -1 ∧ (1 + x)^2 = (1 + p) * (1 + q) ∧ 
  x = Real.sqrt ((1 + p) * (1 + q)) - 1 :=
sorry

end average_annual_growth_rate_l2059_205909


namespace square_sum_inequality_square_sum_equality_l2059_205919

theorem square_sum_inequality {a b c d : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
by sorry

theorem square_sum_equality {a b c d : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end square_sum_inequality_square_sum_equality_l2059_205919


namespace benny_payment_l2059_205951

/-- The cost of a lunch special -/
def lunch_special_cost : ℕ := 8

/-- The number of people in the group -/
def number_of_people : ℕ := 3

/-- The total cost Benny will pay -/
def total_cost : ℕ := number_of_people * lunch_special_cost

theorem benny_payment : total_cost = 24 := by sorry

end benny_payment_l2059_205951


namespace quarterly_insurance_payment_l2059_205914

theorem quarterly_insurance_payment 
  (annual_payment : ℕ) 
  (quarters_per_year : ℕ) 
  (h1 : annual_payment = 1512) 
  (h2 : quarters_per_year = 4) : 
  annual_payment / quarters_per_year = 378 := by
sorry

end quarterly_insurance_payment_l2059_205914


namespace greatest_possible_award_l2059_205908

theorem greatest_possible_award (total_prize : ℕ) (num_winners : ℕ) (min_award : ℕ) :
  total_prize = 600 →
  num_winners = 15 →
  min_award = 15 →
  (2 : ℚ) / 5 * total_prize = (3 : ℚ) / 5 * num_winners * min_award →
  ∃ (max_award : ℕ), max_award = 390 ∧
    max_award ≤ total_prize ∧
    max_award ≥ min_award ∧
    ∃ (other_awards : List ℕ),
      other_awards.length = num_winners - 1 ∧
      (∀ x ∈ other_awards, min_award ≤ x) ∧
      max_award + other_awards.sum = total_prize :=
by
  sorry

end greatest_possible_award_l2059_205908


namespace simplify_and_evaluate_l2059_205966

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end simplify_and_evaluate_l2059_205966


namespace sqrt_product_equals_sqrt_of_product_l2059_205945

theorem sqrt_product_equals_sqrt_of_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end sqrt_product_equals_sqrt_of_product_l2059_205945


namespace stormi_car_wash_price_l2059_205984

/-- The amount Stormi charges for washing each car -/
def car_wash_price : ℝ := 10

/-- The number of cars Stormi washes -/
def num_cars : ℕ := 3

/-- The price Stormi charges for mowing a lawn -/
def lawn_mow_price : ℝ := 13

/-- The number of lawns Stormi mows -/
def num_lawns : ℕ := 2

/-- The cost of the bicycle -/
def bicycle_cost : ℝ := 80

/-- The additional amount Stormi needs to afford the bicycle -/
def additional_amount_needed : ℝ := 24

theorem stormi_car_wash_price :
  car_wash_price * num_cars + lawn_mow_price * num_lawns = bicycle_cost - additional_amount_needed :=
sorry

end stormi_car_wash_price_l2059_205984


namespace digit_sum_equals_78331_l2059_205944

/-- A function that generates all possible natural numbers from a given list of digits,
    where each digit can be used no more than once. -/
def generateNumbers (digits : List Nat) : List Nat :=
  sorry

/-- The sum of all numbers generated from the digits 2, 0, 1, 8. -/
def digitSum : Nat :=
  (generateNumbers [2, 0, 1, 8]).sum

/-- Theorem stating that the sum of all possible natural numbers formed from digits 2, 0, 1, 8,
    where each digit is used no more than once, is equal to 78331. -/
theorem digit_sum_equals_78331 : digitSum = 78331 := by
  sorry

end digit_sum_equals_78331_l2059_205944


namespace sector_area_l2059_205940

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 12) (h2 : central_angle = 2) :
  let radius := perimeter / (2 + central_angle)
  (1/2) * central_angle * radius^2 = 9 := by sorry

end sector_area_l2059_205940


namespace log_simplification_l2059_205958

theorem log_simplification (a b c d x y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : x > 0) (h6 : y > 0) :
  Real.log (2 * a / (3 * b)) + Real.log (3 * b / (4 * c)) + Real.log (4 * c / (5 * d)) - Real.log (10 * a * y / (3 * d * x)) = Real.log (3 * x / (25 * y)) :=
by sorry

end log_simplification_l2059_205958


namespace payment_difference_l2059_205900

/-- Represents the pizza with its properties and how it was shared -/
structure Pizza :=
  (total_slices : ℕ)
  (plain_cost : ℚ)
  (pepperoni_cost : ℚ)
  (mushroom_cost : ℚ)
  (bob_slices : ℕ)
  (charlie_slices : ℕ)
  (alice_slices : ℕ)

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : ℚ :=
  p.plain_cost + p.pepperoni_cost + p.mushroom_cost

/-- Calculates the cost per slice -/
def cost_per_slice (p : Pizza) : ℚ :=
  total_cost p / p.total_slices

/-- Calculates how much Bob paid -/
def bob_payment (p : Pizza) : ℚ :=
  cost_per_slice p * p.bob_slices

/-- Calculates how much Alice paid -/
def alice_payment (p : Pizza) : ℚ :=
  cost_per_slice p * p.alice_slices

/-- The main theorem stating the difference in payment between Bob and Alice -/
theorem payment_difference (p : Pizza) 
  (h1 : p.total_slices = 12)
  (h2 : p.plain_cost = 12)
  (h3 : p.pepperoni_cost = 3)
  (h4 : p.mushroom_cost = 2)
  (h5 : p.bob_slices = 6)
  (h6 : p.charlie_slices = 5)
  (h7 : p.alice_slices = 3) :
  bob_payment p - alice_payment p = 4.26 := by
  sorry


end payment_difference_l2059_205900


namespace smallest_two_digit_integer_with_property_l2059_205962

theorem smallest_two_digit_integer_with_property : ∃ n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (let a := n / 10; let b := n % 10; 10 * b + a + 5 = 2 * n) ∧
  (∀ m : ℕ, m ≥ 10 ∧ m < 100 → 
    (let x := m / 10; let y := m % 10; 10 * y + x + 5 = 2 * m) → 
    m ≥ n) ∧
  n = 69 :=
sorry

end smallest_two_digit_integer_with_property_l2059_205962


namespace max_sum_of_digits_l2059_205999

/-- Represents a nonzero digit (1-9) -/
def NonzeroDigit := {d : ℕ // 1 ≤ d ∧ d ≤ 9}

/-- An is an n-digit integer with all digits equal to a -/
def An (a : NonzeroDigit) (n : ℕ+) : ℕ := a.val * (10^n.val - 1) / 9

/-- Bn is an n-digit integer with all digits equal to b -/
def Bn (b : NonzeroDigit) (n : ℕ+) : ℕ := b.val * (10^n.val - 1) / 9

/-- Cn is a 3n-digit integer with all digits equal to c -/
def Cn (c : NonzeroDigit) (n : ℕ+) : ℕ := c.val * (10^(3*n.val) - 1) / 9

/-- The equation Cn - Bn = An^3 is satisfied for at least two values of n -/
def SatisfiesEquation (a b c : NonzeroDigit) : Prop :=
  ∃ n₁ n₂ : ℕ+, n₁ ≠ n₂ ∧ Cn c n₁ - Bn b n₁ = (An a n₁)^3 ∧ Cn c n₂ - Bn b n₂ = (An a n₂)^3

theorem max_sum_of_digits (a b c : NonzeroDigit) (h : SatisfiesEquation a b c) :
  a.val + b.val + c.val ≤ 19 :=
sorry

end max_sum_of_digits_l2059_205999


namespace range_of_a_l2059_205942

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_monotone_decreasing f (-2) 4)
  (h2 : f (a + 1) > f (2 * a)) :
  1 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l2059_205942


namespace polynomial_expansion_l2059_205957

theorem polynomial_expansion (z : ℂ) : 
  (3 * z^3 + 4 * z^2 - 5 * z + 1) * (2 * z^2 - 3 * z + 4) * (z - 1) = 
  3 * z^6 - 3 * z^5 + 5 * z^4 + 2 * z^3 - 5 * z^2 + 4 * z - 16 := by
  sorry

end polynomial_expansion_l2059_205957


namespace initial_saline_concentration_l2059_205991

theorem initial_saline_concentration 
  (initial_weight : ℝ) 
  (water_added : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_weight = 100)
  (h2 : water_added = 200)
  (h3 : final_concentration = 10)
  : ∃ (initial_concentration : ℝ),
    initial_concentration = 30 ∧ 
    (initial_concentration / 100) * initial_weight = 
    (final_concentration / 100) * (initial_weight + water_added) :=
sorry

end initial_saline_concentration_l2059_205991


namespace jungkook_red_balls_l2059_205998

/-- The number of boxes Jungkook bought -/
def num_boxes : ℕ := 2

/-- The number of red balls in each box -/
def balls_per_box : ℕ := 3

/-- The total number of red balls Jungkook has -/
def total_balls : ℕ := num_boxes * balls_per_box

theorem jungkook_red_balls : total_balls = 6 := by
  sorry

end jungkook_red_balls_l2059_205998


namespace quadratic_no_real_roots_l2059_205925

theorem quadratic_no_real_roots (c : ℤ) : 
  c < 3 → 
  (∀ x : ℝ, x^2 + 2*x + c ≠ 0) → 
  c = 2 := by
sorry

end quadratic_no_real_roots_l2059_205925


namespace population_growth_model_l2059_205953

/-- World population growth model from 1992 to 2000 -/
theorem population_growth_model 
  (initial_population : ℝ) 
  (growth_rate : ℝ) 
  (years : ℕ) 
  (final_population : ℝ) :
  initial_population = 5.48 →
  years = 8 →
  final_population = initial_population * (1 + growth_rate / 100) ^ years :=
by sorry

end population_growth_model_l2059_205953


namespace unknown_number_value_l2059_205904

theorem unknown_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 315 * 7) : x = 25 := by
  sorry

end unknown_number_value_l2059_205904


namespace chess_tournament_games_14_l2059_205905

/-- The number of games played in a chess tournament -/
def chess_tournament_games (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a chess tournament with 14 players where each player plays every other player once,
    the total number of games played is 91. -/
theorem chess_tournament_games_14 :
  chess_tournament_games 14 = 91 := by
  sorry

#eval chess_tournament_games 14  -- This should output 91

end chess_tournament_games_14_l2059_205905


namespace an_is_arithmetic_sequence_l2059_205974

/-- Definition of an arithmetic sequence's general term -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ n, a n = m * n + b

/-- The sequence defined by an = 3n - 1 -/
def a (n : ℕ) : ℝ := 3 * n - 1

/-- Theorem: The sequence defined by an = 3n - 1 is an arithmetic sequence -/
theorem an_is_arithmetic_sequence : is_arithmetic_sequence a := by
  sorry

end an_is_arithmetic_sequence_l2059_205974


namespace two_colonies_limit_time_l2059_205902

/-- Represents the growth of a bacteria colony -/
structure BacteriaColony where
  initialSize : ℕ
  doubleTime : ℕ
  habitatLimit : ℕ

/-- The time it takes for a single colony to reach the habitat limit -/
def singleColonyLimitTime (colony : BacteriaColony) : ℕ := 20

/-- The size of a colony after a given number of days -/
def colonySize (colony : BacteriaColony) (days : ℕ) : ℕ :=
  colony.initialSize * 2^days

/-- Predicate to check if a colony has reached the habitat limit -/
def hasReachedLimit (colony : BacteriaColony) (days : ℕ) : Prop :=
  colonySize colony days ≥ colony.habitatLimit

/-- Theorem: Two colonies reach the habitat limit in the same time as a single colony -/
theorem two_colonies_limit_time (colony1 colony2 : BacteriaColony) :
  (∃ t : ℕ, hasReachedLimit colony1 t ∧ hasReachedLimit colony2 t) →
  (∃ t : ℕ, t = singleColonyLimitTime colony1 ∧ hasReachedLimit colony1 t ∧ hasReachedLimit colony2 t) :=
sorry

end two_colonies_limit_time_l2059_205902


namespace algorithm_structure_logical_judgment_l2059_205941

-- Define the basic algorithm structures
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

-- Define a property for structures requiring logical judgment and different processing
def RequiresLogicalJudgment (s : AlgorithmStructure) : Prop :=
  match s with
  | AlgorithmStructure.Conditional => true
  | AlgorithmStructure.Loop => true
  | _ => false

-- Theorem statement
theorem algorithm_structure_logical_judgment :
  ∀ (s : AlgorithmStructure),
    RequiresLogicalJudgment s ↔ (s = AlgorithmStructure.Conditional ∨ s = AlgorithmStructure.Loop) :=
by sorry

end algorithm_structure_logical_judgment_l2059_205941


namespace chicken_fried_steak_cost_l2059_205930

theorem chicken_fried_steak_cost (steak_egg_cost : ℝ) (james_payment : ℝ) 
  (tip_percentage : ℝ) (chicken_fried_steak_cost : ℝ) :
  steak_egg_cost = 16 →
  james_payment = 21 →
  tip_percentage = 0.20 →
  james_payment = (steak_egg_cost + chicken_fried_steak_cost) / 2 + 
    tip_percentage * (steak_egg_cost + chicken_fried_steak_cost) →
  chicken_fried_steak_cost = 14 := by
  sorry

end chicken_fried_steak_cost_l2059_205930


namespace min_subset_size_for_sum_l2059_205907

theorem min_subset_size_for_sum (n : ℕ+) :
  let M := Finset.range (2 * n)
  ∃ k : ℕ+, (∀ A : Finset ℕ, A ⊆ M → A.card = k →
    ∃ a b c d : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b + c + d = 4 * n + 1) ∧
  (∀ k' : ℕ+, k' < k →
    ∃ A : Finset ℕ, A ⊆ M ∧ A.card = k' ∧
    ∀ a b c d : ℕ, a ∈ A → b ∈ A → c ∈ A → d ∈ A →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    a + b + c + d ≠ 4 * n + 1) ∧
  k = n + 3 :=
by sorry

end min_subset_size_for_sum_l2059_205907


namespace triangle_area_l2059_205917

-- Define the lines that bound the triangle
def line1 (x : ℝ) : ℝ := x
def line2 (x : ℝ) : ℝ := -x
def line3 : ℝ := 8

-- Theorem statement
theorem triangle_area : 
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := |A.1 - B.1|
  let height := |line3 - O.2|
  (1/2 : ℝ) * base * height = 64 := by sorry

end triangle_area_l2059_205917


namespace secretary_project_time_l2059_205913

theorem secretary_project_time (t1 t2 t3 : ℕ) : 
  t1 + t2 + t3 = 120 →
  t2 = 2 * t1 →
  t3 = 5 * t1 →
  t3 = 75 := by
sorry

end secretary_project_time_l2059_205913


namespace tangent_sum_half_pi_l2059_205972

theorem tangent_sum_half_pi (α β γ : Real) 
  (h_acute : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2) 
  (h_sum : α + β + γ = π/2) : 
  Real.tan α * Real.tan β + Real.tan α * Real.tan γ + Real.tan β * Real.tan γ = 1 := by
  sorry

end tangent_sum_half_pi_l2059_205972


namespace quadratic_root_ratio_l2059_205922

theorem quadratic_root_ratio (c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (5 * x₁^2 - 2 * x₁ + c = 0) ∧ 
    (5 * x₂^2 - 2 * x₂ + c = 0) ∧ 
    (x₁ / x₂ = -3/5)) → 
  c = -3 := by
sorry

end quadratic_root_ratio_l2059_205922


namespace particle_position_after_3045_minutes_l2059_205950

/-- Represents the position of a particle -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Calculates the time taken for n rectangles -/
def timeForNRectangles (n : ℕ) : ℕ :=
  (n + 1)^2 - 1

/-- Calculates the position after n complete rectangles -/
def positionAfterNRectangles (n : ℕ) : Position :=
  if n % 2 = 0 then
    ⟨0, n⟩
  else
    ⟨0, n⟩

/-- Calculates the final position after given time -/
def finalPosition (time : ℕ) : Position :=
  let n := (Nat.sqrt (time + 1) : ℕ) - 1
  let remainingTime := time - timeForNRectangles n
  let basePosition := positionAfterNRectangles n
  if n % 2 = 0 then
    ⟨basePosition.x + remainingTime, basePosition.y⟩
  else
    ⟨basePosition.x + remainingTime, basePosition.y⟩

theorem particle_position_after_3045_minutes :
  finalPosition 3045 = ⟨21, 54⟩ := by
  sorry


end particle_position_after_3045_minutes_l2059_205950


namespace circle_slope_bounds_l2059_205990

theorem circle_slope_bounds (x y : ℝ) (h : x^2 + y^2 + 2*x - 4*y + 1 = 0) :
  ∃ (k : ℝ), y = k*(x-4) ∧ -20/21 ≤ k ∧ k ≤ 0 :=
sorry

end circle_slope_bounds_l2059_205990


namespace max_area_inscribed_rectangle_l2059_205906

theorem max_area_inscribed_rectangle (d : ℝ) (h : d = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → x^2 + y^2 = d^2 → x * y ≤ d^2 / 2 :=
by
  sorry

#check max_area_inscribed_rectangle

end max_area_inscribed_rectangle_l2059_205906


namespace tetradecagon_side_length_l2059_205954

/-- A regular tetradecagon is a polygon with 14 sides of equal length -/
def RegularTetradecagon := { n : ℕ // n = 14 }

/-- The perimeter of the tetradecagon table in centimeters -/
def perimeter : ℝ := 154

/-- Theorem: In a regular tetradecagon with a perimeter of 154 cm, the length of each side is 11 cm -/
theorem tetradecagon_side_length (t : RegularTetradecagon) :
  perimeter / t.val = 11 := by sorry

end tetradecagon_side_length_l2059_205954


namespace joan_seashells_l2059_205971

/-- The number of seashells Joan has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proof that Joan has 16 seashells after giving 63 away from her initial 79 -/
theorem joan_seashells : remaining_seashells 79 63 = 16 := by
  sorry

end joan_seashells_l2059_205971


namespace max_value_implies_a_equals_one_l2059_205955

noncomputable def f (x a : ℝ) : ℝ := (1 + Real.cos (2 * x)) * 1 + 1 * (Real.sqrt 3 * Real.sin (2 * x) + a)

theorem max_value_implies_a_equals_one (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = 4) →
  a = 1 :=
by sorry

end max_value_implies_a_equals_one_l2059_205955


namespace train_distance_proof_l2059_205982

/-- Calculates the distance a train can travel given its coal efficiency and available coal. -/
def trainDistance (milesPerCoal : ℚ) (availableCoal : ℚ) : ℚ :=
  milesPerCoal * availableCoal

/-- Proves that a train with given efficiency and coal amount can travel 400 miles. -/
theorem train_distance_proof :
  let milesPerCoal : ℚ := 5 / 2
  let availableCoal : ℚ := 160
  trainDistance milesPerCoal availableCoal = 400 := by
  sorry

#eval trainDistance (5 / 2) 160

end train_distance_proof_l2059_205982


namespace expression_can_be_any_real_l2059_205949

theorem expression_can_be_any_real (x : ℝ) : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  a + b + c = 1 ∧ 
  (a^4 + b^4 + c^4) / (a*b + b*c + c*a) = x :=
sorry

end expression_can_be_any_real_l2059_205949


namespace total_legs_is_108_l2059_205986

-- Define the number of each animal
def num_birds : ℕ := 3
def num_dogs : ℕ := 5
def num_snakes : ℕ := 4
def num_spiders : ℕ := 1
def num_horses : ℕ := 2
def num_rabbits : ℕ := 6
def num_octopuses : ℕ := 3
def num_ants : ℕ := 7

-- Define the number of legs for each animal type
def legs_bird : ℕ := 2
def legs_dog : ℕ := 4
def legs_snake : ℕ := 0
def legs_spider : ℕ := 8
def legs_horse : ℕ := 4
def legs_rabbit : ℕ := 4
def legs_octopus : ℕ := 0
def legs_ant : ℕ := 6

-- Theorem to prove
theorem total_legs_is_108 : 
  num_birds * legs_bird + 
  num_dogs * legs_dog + 
  num_snakes * legs_snake + 
  num_spiders * legs_spider + 
  num_horses * legs_horse + 
  num_rabbits * legs_rabbit + 
  num_octopuses * legs_octopus + 
  num_ants * legs_ant = 108 := by
  sorry

end total_legs_is_108_l2059_205986
