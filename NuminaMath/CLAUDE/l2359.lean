import Mathlib

namespace NUMINAMATH_CALUDE_not_perfect_square_l2359_235992

theorem not_perfect_square (n : ℕ) (h : n > 1) : ¬ ∃ (a : ℕ), 4 * 10^n + 9 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2359_235992


namespace NUMINAMATH_CALUDE_highway_on_map_l2359_235934

/-- Represents the scale of a map as a ratio -/
structure MapScale where
  numerator : ℕ
  denominator : ℕ

/-- Converts kilometers to centimeters -/
def km_to_cm (km : ℕ) : ℕ := km * 100000

/-- Calculates the length on a map given the actual length and map scale -/
def length_on_map (actual_length_km : ℕ) (scale : MapScale) : ℕ :=
  (km_to_cm actual_length_km) * scale.numerator / scale.denominator

/-- Theorem stating that a 155 km highway on a 1:500000 scale map is 31 cm long -/
theorem highway_on_map :
  let actual_length_km : ℕ := 155
  let scale : MapScale := ⟨1, 500000⟩
  length_on_map actual_length_km scale = 31 := by sorry

end NUMINAMATH_CALUDE_highway_on_map_l2359_235934


namespace NUMINAMATH_CALUDE_class_average_problem_l2359_235968

theorem class_average_problem (x : ℝ) :
  (0.2 * x + 0.5 * 60 + 0.3 * 40 = 58) →
  x = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l2359_235968


namespace NUMINAMATH_CALUDE_intersection_parallel_line_l2359_235913

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line -/
theorem intersection_parallel_line (a b c d e f g h i : ℝ) :
  (∃ x y : ℝ, a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →  -- Intersection exists
  (g ≠ 0 ∨ h ≠ 0) →  -- Third line is not degenerate
  (a * h ≠ b * g ∨ d * h ≠ e * g) →  -- At least one of the first two lines is not parallel to the third
  (∃ k : ℝ, k ≠ 0 ∧ 
    ∀ x y : ℝ, (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
    ∃ t : ℝ, g * x + h * y + i + t * (a * x + b * y + c) = 0 ∧
            g * x + h * y + i + t * (d * x + e * y + f) = 0) →
  ∃ j : ℝ, ∀ x y : ℝ, 
    (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
    g * x + h * y + j = 0
  := by sorry

end NUMINAMATH_CALUDE_intersection_parallel_line_l2359_235913


namespace NUMINAMATH_CALUDE_average_first_n_odd_numbers_l2359_235999

/-- The nth odd number -/
def nthOddNumber (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of the first n odd numbers -/
def sumFirstNOddNumbers (n : ℕ) : ℕ := n ^ 2

/-- The average of the first n odd numbers -/
def averageFirstNOddNumbers (n : ℕ) : ℕ := sumFirstNOddNumbers n / n

theorem average_first_n_odd_numbers (n : ℕ) (h : n > 0) :
  averageFirstNOddNumbers n = nthOddNumber n := by
  sorry

end NUMINAMATH_CALUDE_average_first_n_odd_numbers_l2359_235999


namespace NUMINAMATH_CALUDE_circle_polar_equation_l2359_235964

/-- The polar equation ρ = 2a cos θ represents a circle with center C(a, 0) and radius a -/
theorem circle_polar_equation (a : ℝ) :
  ∀ ρ θ : ℝ, ρ = 2 * a * Real.cos θ ↔ 
  ∃ x y : ℝ, (x - a)^2 + y^2 = a^2 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_circle_polar_equation_l2359_235964


namespace NUMINAMATH_CALUDE_tree_prob_five_vertices_l2359_235979

/-- The number of vertices in the graph -/
def n : ℕ := 5

/-- The probability of drawing an edge between any two vertices -/
def edge_prob : ℚ := 1/2

/-- The number of labeled trees on n vertices -/
def num_labeled_trees (n : ℕ) : ℕ := n^(n-2)

/-- The total number of possible graphs on n vertices -/
def total_graphs (n : ℕ) : ℕ := 2^(n.choose 2)

/-- The probability that a randomly generated graph is a tree -/
def tree_probability (n : ℕ) : ℚ := (num_labeled_trees n : ℚ) / (total_graphs n : ℚ)

theorem tree_prob_five_vertices :
  tree_probability n = 125 / 1024 :=
sorry

end NUMINAMATH_CALUDE_tree_prob_five_vertices_l2359_235979


namespace NUMINAMATH_CALUDE_nyc_streetlights_l2359_235905

/-- Given the total number of streetlights bought, the number of squares, and the number of streetlights per square, 
    calculate the number of unused streetlights. -/
def unused_streetlights (total : ℕ) (squares : ℕ) (per_square : ℕ) : ℕ :=
  total - squares * per_square

/-- Theorem stating that with 200 total streetlights, 15 squares, and 12 streetlights per square, 
    there will be 20 unused streetlights. -/
theorem nyc_streetlights : unused_streetlights 200 15 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_nyc_streetlights_l2359_235905


namespace NUMINAMATH_CALUDE_divisibility_pairs_l2359_235925

theorem divisibility_pairs : 
  {p : ℕ × ℕ | p.1 ∣ (2^(Nat.totient p.2) + 1) ∧ p.2 ∣ (2^(Nat.totient p.1) + 1)} = 
  {(1, 1), (1, 3), (3, 1)} := by
sorry

end NUMINAMATH_CALUDE_divisibility_pairs_l2359_235925


namespace NUMINAMATH_CALUDE_ending_number_divisible_by_eleven_l2359_235994

theorem ending_number_divisible_by_eleven (start : Nat) (count : Nat) : 
  start ≥ 29 →
  start % 11 = 0 →
  count = 5 →
  ∀ k, k ∈ Finset.range count → (start + k * 11) % 11 = 0 →
  start + (count - 1) * 11 = 77 :=
by sorry

end NUMINAMATH_CALUDE_ending_number_divisible_by_eleven_l2359_235994


namespace NUMINAMATH_CALUDE_negative_two_x_squared_cubed_l2359_235996

theorem negative_two_x_squared_cubed (x : ℝ) : (-2 * x^2)^3 = -8 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_x_squared_cubed_l2359_235996


namespace NUMINAMATH_CALUDE_garden_area_increase_l2359_235938

theorem garden_area_increase :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter := 2 * (rect_length + rect_width)
  let square_side := rect_perimeter / 4
  let rect_area := rect_length * rect_width
  let square_area := square_side * square_side
  square_area - rect_area = 400 := by
sorry

end NUMINAMATH_CALUDE_garden_area_increase_l2359_235938


namespace NUMINAMATH_CALUDE_danny_found_seven_caps_l2359_235983

/-- The number of bottle caps Danny found at the park -/
def bottleCapsFound (initialCaps currentCaps : ℕ) : ℕ :=
  currentCaps - initialCaps

/-- Proof that Danny found 7 bottle caps at the park -/
theorem danny_found_seven_caps : bottleCapsFound 25 32 = 7 := by
  sorry

end NUMINAMATH_CALUDE_danny_found_seven_caps_l2359_235983


namespace NUMINAMATH_CALUDE_triangle_properties_l2359_235916

theorem triangle_properties (a b c A B C : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (h_sides : a = 2)
  (h_equation : (b + 2) * (Real.sin A - Real.sin B) = c * (Real.sin B + Real.sin C)) :
  A = 2 * π / 3 ∧
  ∃ S : ℝ, S > 0 ∧ S ≤ Real.sqrt 3 / 3 ∧
    S = 1 / 2 * a * b * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2359_235916


namespace NUMINAMATH_CALUDE_pencil_distribution_l2359_235940

/-- Given a total number of pencils and pencils per row, calculate the number of rows -/
def calculate_rows (total_pencils : ℕ) (pencils_per_row : ℕ) : ℕ :=
  total_pencils / pencils_per_row

/-- Theorem: Given 6 pencils distributed equally into rows of 3 pencils each, 
    the number of rows created is 2 -/
theorem pencil_distribution :
  calculate_rows 6 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2359_235940


namespace NUMINAMATH_CALUDE_min_value_of_abs_sum_l2359_235945

theorem min_value_of_abs_sum (x : ℝ) : 
  |x - 4| + |x + 2| + |x - 5| ≥ -1 ∧ ∃ y : ℝ, |y - 4| + |y + 2| + |y - 5| = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_abs_sum_l2359_235945


namespace NUMINAMATH_CALUDE_intersection_count_l2359_235942

-- Define the lines
def line1 (x y : ℝ) : Prop := 3*x + 4*y - 12 = 0
def line2 (x y : ℝ) : Prop := 5*x - 2*y - 10 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨
  (line1 x y ∧ line3 x) ∨
  (line1 x y ∧ line4 y) ∨
  (line2 x y ∧ line3 x) ∨
  (line2 x y ∧ line4 y) ∨
  (line3 x ∧ line4 y)

-- Theorem statement
theorem intersection_count :
  ∃ (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    ∀ (p : ℝ × ℝ), is_intersection p.1 p.2 → p = p1 ∨ p = p2 :=
sorry

end NUMINAMATH_CALUDE_intersection_count_l2359_235942


namespace NUMINAMATH_CALUDE_min_empty_cells_is_three_l2359_235943

/-- Represents a triangular cell arrangement with grasshoppers -/
structure TriangularArrangement where
  up_cells : ℕ  -- Number of upward-pointing cells
  down_cells : ℕ  -- Number of downward-pointing cells
  has_more_up : up_cells = down_cells + 3

/-- The minimum number of empty cells after all grasshoppers have jumped -/
def min_empty_cells (arrangement : TriangularArrangement) : ℕ := 3

/-- Theorem stating that the minimum number of empty cells is always 3 -/
theorem min_empty_cells_is_three (arrangement : TriangularArrangement) :
  min_empty_cells arrangement = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_empty_cells_is_three_l2359_235943


namespace NUMINAMATH_CALUDE_total_amount_divided_l2359_235902

/-- Proves that the total amount divided is 3500, given the specified conditions --/
theorem total_amount_divided (first_part : ℝ) (interest_rate1 : ℝ) (interest_rate2 : ℝ) 
  (total_interest : ℝ) :
  first_part = 1550 →
  interest_rate1 = 0.03 →
  interest_rate2 = 0.05 →
  total_interest = 144 →
  ∃ (total : ℝ), 
    total = 3500 ∧
    first_part * interest_rate1 + (total - first_part) * interest_rate2 = total_interest :=
by
  sorry


end NUMINAMATH_CALUDE_total_amount_divided_l2359_235902


namespace NUMINAMATH_CALUDE_expression_value_l2359_235949

theorem expression_value : 
  (11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2359_235949


namespace NUMINAMATH_CALUDE_find_divisor_l2359_235944

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 144 →
  quotient = 13 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2359_235944


namespace NUMINAMATH_CALUDE_oak_trees_cut_down_l2359_235972

theorem oak_trees_cut_down (initial_trees : ℕ) (remaining_trees : ℕ) :
  initial_trees = 9 →
  remaining_trees = 7 →
  initial_trees - remaining_trees = 2 :=
by sorry

end NUMINAMATH_CALUDE_oak_trees_cut_down_l2359_235972


namespace NUMINAMATH_CALUDE_boy_scout_interest_l2359_235939

/-- Represents the simple interest calculation for a Boy Scout Troop's account --/
theorem boy_scout_interest (final_balance : ℝ) (rate : ℝ) (time : ℝ) (interest : ℝ) : 
  final_balance = 310.45 →
  rate = 0.06 →
  time = 0.25 →
  interest = final_balance - (final_balance / (1 + rate * time)) →
  interest = 4.54 := by
sorry

end NUMINAMATH_CALUDE_boy_scout_interest_l2359_235939


namespace NUMINAMATH_CALUDE_final_bus_count_l2359_235987

def bus_problem (initial : ℕ) (first_stop : ℕ) (second_stop : ℕ) (third_stop : ℕ) : ℕ :=
  initial + first_stop - second_stop + third_stop

theorem final_bus_count :
  bus_problem 128 67 34 54 = 215 := by
  sorry

end NUMINAMATH_CALUDE_final_bus_count_l2359_235987


namespace NUMINAMATH_CALUDE_sin_480_plus_tan_300_l2359_235924

/-- The sum of sine of 480 degrees and tangent of 300 degrees equals negative square root of 3 divided by 2. -/
theorem sin_480_plus_tan_300 : Real.sin (480 * π / 180) + Real.tan (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_480_plus_tan_300_l2359_235924


namespace NUMINAMATH_CALUDE_simplify_expression_l2359_235982

theorem simplify_expression : 
  (((81 : ℝ) ^ (1/4 : ℝ)) + (Real.sqrt (8 + 3/4)))^2 = (71 + 12 * Real.sqrt 35) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2359_235982


namespace NUMINAMATH_CALUDE_max_cosine_difference_value_l2359_235981

def max_cosine_difference (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  a₃ = a₂ + a₁ ∧ 
  a₄ = a₃ + a₂ ∧ 
  ∃ (a b c : ℝ), ∀ n ∈ ({1, 2, 3, 4} : Set ℕ), 
    a * n^2 + b * n + c = Real.cos (if n = 1 then a₁ 
                                    else if n = 2 then a₂ 
                                    else if n = 3 then a₃ 
                                    else a₄)

theorem max_cosine_difference_value :
  ∀ a₁ a₂ a₃ a₄ : ℝ, max_cosine_difference a₁ a₂ a₃ a₄ →
    Real.cos a₁ - Real.cos a₄ ≤ -9 + 3 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_max_cosine_difference_value_l2359_235981


namespace NUMINAMATH_CALUDE_shopper_receive_amount_l2359_235986

/-- The amount of money each person has and donates --/
def problem (isabella sam giselle valentina ethan : ℚ) : Prop :=
  isabella = giselle + 15 ∧
  isabella = sam + 45 ∧
  giselle = 120 ∧
  valentina = 2 * sam ∧
  ethan = isabella - 75

/-- The total donation amount --/
def total_donation (isabella sam giselle valentina ethan : ℚ) : ℚ :=
  0.2 * isabella + 0.15 * sam + 0.1 * giselle + 0.25 * valentina + 0.3 * ethan

/-- The amount each shopper receives after equal distribution --/
def shopper_receive (isabella sam giselle valentina ethan : ℚ) : ℚ :=
  (total_donation isabella sam giselle valentina ethan) / 4

/-- Theorem stating the amount each shopper receives --/
theorem shopper_receive_amount :
  ∀ isabella sam giselle valentina ethan,
  problem isabella sam giselle valentina ethan →
  shopper_receive isabella sam giselle valentina ethan = 28.875 :=
by sorry

end NUMINAMATH_CALUDE_shopper_receive_amount_l2359_235986


namespace NUMINAMATH_CALUDE_angle_sequence_convergence_l2359_235904

noncomputable def angle_sequence (α : ℝ) : ℕ → ℝ
  | 0 => 0  -- Initial value doesn't affect the limit
  | n + 1 => (Real.pi - α - angle_sequence α n) / 2

theorem angle_sequence_convergence (α : ℝ) (h : 0 < α ∧ α < Real.pi) :
  ∃ (L : ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |angle_sequence α n - L| < ε) ∧
             L = (Real.pi - α) / 3 :=
by sorry

end NUMINAMATH_CALUDE_angle_sequence_convergence_l2359_235904


namespace NUMINAMATH_CALUDE_sum_of_x_sixth_powers_l2359_235955

theorem sum_of_x_sixth_powers (x : ℕ) (b : ℕ) :
  (x : ℝ) * (x : ℝ)^6 = (x : ℝ)^b → b = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_sixth_powers_l2359_235955


namespace NUMINAMATH_CALUDE_greatest_integer_with_conditions_l2359_235989

theorem greatest_integer_with_conditions : ∃ n : ℕ, 
  n < 150 ∧ 
  (∃ a b : ℕ, n + 2 = 9 * a ∧ n + 3 = 11 * b) ∧
  (∀ m : ℕ, m < 150 → (∃ c d : ℕ, m + 2 = 9 * c ∧ m + 3 = 11 * d) → m ≤ n) ∧
  n = 142 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_conditions_l2359_235989


namespace NUMINAMATH_CALUDE_senior_teachers_in_sample_l2359_235922

theorem senior_teachers_in_sample
  (total_teachers : ℕ)
  (intermediate_teachers : ℕ)
  (sample_intermediate : ℕ)
  (h_total : total_teachers = 300)
  (h_intermediate : intermediate_teachers = 192)
  (h_sample_intermediate : sample_intermediate = 64)
  (h_ratio : ∃ k : ℕ, k > 0 ∧ total_teachers - intermediate_teachers = 9 * k ∧ 5 * k = 4 * k + k) :
  ∃ sample_size : ℕ,
    sample_size * intermediate_teachers = sample_intermediate * total_teachers ∧
    ∃ sample_senior : ℕ,
      9 * sample_senior = 5 * (sample_size - sample_intermediate) ∧
      sample_senior = 20 :=
sorry

end NUMINAMATH_CALUDE_senior_teachers_in_sample_l2359_235922


namespace NUMINAMATH_CALUDE_line_through_points_l2359_235946

/-- A line passing through two points (1,3) and (4,-2) can be represented by y = mx + b, where m + b = 3 -/
theorem line_through_points (m b : ℚ) : 
  (3 = m * 1 + b) → (-2 = m * 4 + b) → m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2359_235946


namespace NUMINAMATH_CALUDE_xyz_value_l2359_235948

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + 2 * x * y * z = 20) :
  x * y * z = 20 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2359_235948


namespace NUMINAMATH_CALUDE_flower_bed_fraction_l2359_235997

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

end NUMINAMATH_CALUDE_flower_bed_fraction_l2359_235997


namespace NUMINAMATH_CALUDE_return_trip_time_l2359_235970

/-- Represents the time for a plane's journey between two cities -/
structure FlightTime where
  against_wind : ℝ  -- Time flying against the wind
  still_air : ℝ     -- Time flying in still air
  with_wind : ℝ     -- Time flying with the wind

/-- Checks if the flight times are valid according to the problem conditions -/
def is_valid_flight (ft : FlightTime) : Prop :=
  ft.against_wind = 75 ∧ ft.with_wind = ft.still_air - 10

/-- Theorem stating the possible return trip times -/
theorem return_trip_time (ft : FlightTime) :
  is_valid_flight ft → ft.with_wind = 15 ∨ ft.with_wind = 50 := by
  sorry

end NUMINAMATH_CALUDE_return_trip_time_l2359_235970


namespace NUMINAMATH_CALUDE_mandy_nutmeg_amount_l2359_235974

/-- The amount of cinnamon Mandy used in tablespoons -/
def cinnamon : ℚ := 0.6666666666666666

/-- The difference between cinnamon and nutmeg in tablespoons -/
def difference : ℚ := 0.16666666666666666

/-- The amount of nutmeg Mandy used in tablespoons -/
def nutmeg : ℚ := cinnamon - difference

theorem mandy_nutmeg_amount : nutmeg = 0.5 := by sorry

end NUMINAMATH_CALUDE_mandy_nutmeg_amount_l2359_235974


namespace NUMINAMATH_CALUDE_wire_length_l2359_235966

/-- Given that a 75-meter roll of wire weighs 15 kg, 
    this theorem proves that a roll weighing 5 kg is 25 meters long. -/
theorem wire_length (weight : ℝ) (length : ℝ) : 
  (75 : ℝ) / 15 = length / weight → weight = 5 → length = 25 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_l2359_235966


namespace NUMINAMATH_CALUDE_phone_price_reduction_l2359_235991

theorem phone_price_reduction (reduced_price : ℝ) (percentage : ℝ) 
  (h1 : reduced_price = 1800)
  (h2 : percentage = 90/100)
  (h3 : reduced_price = percentage * (reduced_price / percentage)) :
  reduced_price / percentage - reduced_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_phone_price_reduction_l2359_235991


namespace NUMINAMATH_CALUDE_divisible_by_24_sum_of_four_cubes_l2359_235935

theorem divisible_by_24_sum_of_four_cubes (k : ℤ) : 
  24 * k = (k + 2)^3 + (-k)^3 + (-k)^3 + (k - 2)^3 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_24_sum_of_four_cubes_l2359_235935


namespace NUMINAMATH_CALUDE_largest_fourth_side_l2359_235908

/-- A quadrilateral with side lengths a, b, c, and d satisfies the triangle inequality -/
def is_valid_quadrilateral (a b c d : ℕ) : Prop :=
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a

/-- The largest possible integer value for the fourth side of a quadrilateral 
    with other sides 2006, 2007, and 2008 -/
theorem largest_fourth_side : 
  ∀ x : ℕ, is_valid_quadrilateral 2006 2007 2008 x → x ≤ 6020 :=
by sorry

end NUMINAMATH_CALUDE_largest_fourth_side_l2359_235908


namespace NUMINAMATH_CALUDE_triangle_count_properties_l2359_235920

/-- Function that counts the number of congruent integer-sided triangles with perimeter n -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem stating the properties of function f for specific values -/
theorem triangle_count_properties (h : ∀ n : ℕ, n ≥ 3 → f n = f n) :
  (f 1999 > f 1996) ∧ (f 2000 = f 1997) := by sorry

end NUMINAMATH_CALUDE_triangle_count_properties_l2359_235920


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_l2359_235919

/-- A circle with diameter endpoints at (0,0) and (10,0) -/
def circle_with_diameter (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 25

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The intersection point of the circle and y-axis -/
def intersection_point (y : ℝ) : Prop :=
  circle_with_diameter 0 y ∧ y_axis 0

theorem circle_y_axis_intersection :
  ∃ y : ℝ, intersection_point y ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_l2359_235919


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l2359_235900

/-- Given a circle A with center at (-1, 2) tangent to line m: x + 2y + 7 = 0,
    and a moving line l passing through B(-2, 0) intersecting circle A at M and N,
    prove the equation of circle A and find the equations of line l when |MN| = 2√19. -/
theorem circle_and_line_problem :
  ∀ (A : ℝ × ℝ) (m : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) (M N : ℝ × ℝ),
  A = (-1, 2) →
  (∀ x y, m x y ↔ x + 2*y + 7 = 0) →
  (∃ r : ℝ, ∀ x y, (x + 1)^2 + (y - 2)^2 = r^2 ↔ m x y) →
  (∀ x, l x 0 ↔ x = -2) →
  (∃ x y, l x y ∧ (x + 1)^2 + (y - 2)^2 = 20) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 4*19 →
  ((∀ x y, (x + 1)^2 + (y - 2)^2 = 20 ↔ (x - A.1)^2 + (y - A.2)^2 = 20) ∧
   ((∀ x y, l x y ↔ 3*x - 4*y + 6 = 0) ∨ (∀ x y, l x y ↔ x = -2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_problem_l2359_235900


namespace NUMINAMATH_CALUDE_system_solution_proof_l2359_235975

theorem system_solution_proof :
  let x : ℚ := -262/75
  let y : ℚ := -2075/200
  let z : ℚ := -105/100
  (3 * x - 4 * y = 12) ∧
  (-5 * x + 6 * y - z = 9) ∧
  (x + 2 * y + 3 * z = 0) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_proof_l2359_235975


namespace NUMINAMATH_CALUDE_shortest_chord_length_l2359_235933

/-- The shortest chord length of the intersection between a line and a circle -/
theorem shortest_chord_length (m : ℝ) : 
  let l := {(x, y) : ℝ × ℝ | 2 * m * x - y - 8 * m - 3 = 0}
  let C := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6 * x + 12 * y + 20 = 0}
  ∃ (chord_length : ℝ), 
    chord_length = 2 * Real.sqrt 15 ∧ 
    ∀ (other_length : ℝ), 
      (∃ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l ∧ p ∈ C ∧ q ∈ C ∧ 
        other_length = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) →
      other_length ≥ chord_length :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_chord_length_l2359_235933


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2359_235998

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 6*x₁ + 9*k = 0 ∧ x₂^2 - 6*x₂ + 9*k = 0) ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2359_235998


namespace NUMINAMATH_CALUDE_inequality_proof_l2359_235977

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2359_235977


namespace NUMINAMATH_CALUDE_enlarged_poster_height_l2359_235960

-- Define the original poster dimensions
def original_width : ℚ := 3
def original_height : ℚ := 2

-- Define the new width
def new_width : ℚ := 12

-- Define the function to calculate the new height
def calculate_new_height (ow oh nw : ℚ) : ℚ :=
  (nw / ow) * oh

-- Theorem statement
theorem enlarged_poster_height :
  calculate_new_height original_width original_height new_width = 8 := by
  sorry

end NUMINAMATH_CALUDE_enlarged_poster_height_l2359_235960


namespace NUMINAMATH_CALUDE_algebraic_expressions_l2359_235985

variable (a x : ℝ)

theorem algebraic_expressions :
  ((-3 * a^2)^3 - 4 * a^2 * a^4 + 5 * a^9 / a^3 = -26 * a^6) ∧
  (((x + 1) * (x + 2) + 2 * (x - 1)) / x = x + 5) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expressions_l2359_235985


namespace NUMINAMATH_CALUDE_folded_paper_distance_l2359_235958

/-- Given a square sheet of paper with area 18 cm², prove that when folded such that
    the visible black area equals the visible white area, the distance from the folded
    point to its original position is 2√6 cm. -/
theorem folded_paper_distance (side_length : ℝ) (fold_length : ℝ) (distance : ℝ) : 
  side_length^2 = 18 →
  fold_length^2 = 12 →
  (1/2) * fold_length^2 = 18 - fold_length^2 →
  distance^2 = 2 * fold_length^2 →
  distance = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_folded_paper_distance_l2359_235958


namespace NUMINAMATH_CALUDE_two_tshirts_per_package_l2359_235911

/-- Given a number of packages and a total number of t-shirts, 
    calculate the number of t-shirts per package -/
def tshirts_per_package (num_packages : ℕ) (total_tshirts : ℕ) : ℕ :=
  total_tshirts / num_packages

/-- Theorem: Given 28 packages and 56 total t-shirts, 
    each package contains 2 t-shirts -/
theorem two_tshirts_per_package :
  tshirts_per_package 28 56 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_tshirts_per_package_l2359_235911


namespace NUMINAMATH_CALUDE_total_shaded_area_l2359_235941

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a right triangle with two equal legs -/
structure RightTriangle where
  leg : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  r.width * r.height

/-- Calculates the area of a right triangle -/
def rightTriangleArea (t : RightTriangle) : ℝ :=
  0.5 * t.leg * t.leg

/-- Represents the overlap between rectangles -/
def rectangleOverlap : ℝ := 20

/-- Represents the fraction of triangle overlap with rectangles -/
def triangleOverlapFraction : ℝ := 0.5

/-- Theorem stating the total shaded area -/
theorem total_shaded_area (r1 r2 : Rectangle) (t : RightTriangle) :
  let totalArea := rectangleArea r1 + rectangleArea r2 - rectangleOverlap
  let triangleCorrection := triangleOverlapFraction * rightTriangleArea t
  totalArea - triangleCorrection = 70.75 :=
by
  sorry

#check total_shaded_area (Rectangle.mk 4 12) (Rectangle.mk 5 9) (RightTriangle.mk 3)

end NUMINAMATH_CALUDE_total_shaded_area_l2359_235941


namespace NUMINAMATH_CALUDE_opposite_of_neg_one_half_l2359_235952

/-- The opposite of a rational number -/
def opposite (x : ℚ) : ℚ := -x

/-- The property that defines the opposite of a number -/
def is_opposite (x y : ℚ) : Prop := x + y = 0

theorem opposite_of_neg_one_half :
  is_opposite (-1/2 : ℚ) (1/2 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_neg_one_half_l2359_235952


namespace NUMINAMATH_CALUDE_faster_train_length_l2359_235912

/-- Calculates the length of a faster train given the speeds of two trains and the time it takes for the faster train to pass a man in the slower train. -/
theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : faster_speed = 72)
  (h2 : slower_speed = 36)
  (h3 : passing_time = 12)
  (h4 : faster_speed > slower_speed) :
  let relative_speed := faster_speed - slower_speed
  let speed_ms := relative_speed * (5 / 18)
  let train_length := speed_ms * passing_time
  train_length = 120 := by sorry

end NUMINAMATH_CALUDE_faster_train_length_l2359_235912


namespace NUMINAMATH_CALUDE_constant_function_not_decreasing_l2359_235959

def f : ℝ → ℝ := fun _ ↦ 2

theorem constant_function_not_decreasing :
  ¬∃ (a b : ℝ), a < b ∧ ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_constant_function_not_decreasing_l2359_235959


namespace NUMINAMATH_CALUDE_minimum_oranges_l2359_235914

theorem minimum_oranges : ∃ n : ℕ, n > 0 ∧ 
  (n % 5 = 1 ∧ n % 7 = 1 ∧ n % 10 = 1) ∧ 
  ∀ m : ℕ, m > 0 → (m % 5 = 1 ∧ m % 7 = 1 ∧ m % 10 = 1) → m ≥ 71 := by
  sorry

end NUMINAMATH_CALUDE_minimum_oranges_l2359_235914


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2359_235937

theorem smallest_number_proof (x y z : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  (x + y + z) / 3 = 165 →
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2359_235937


namespace NUMINAMATH_CALUDE_interval_equinumerosity_l2359_235953

theorem interval_equinumerosity (a : ℝ) (ha : a > 0) :
  ∃ f : Set.Icc 0 1 → Set.Icc 0 a, Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_interval_equinumerosity_l2359_235953


namespace NUMINAMATH_CALUDE_circular_sign_diameter_ratio_l2359_235918

theorem circular_sign_diameter_ratio (d₁ d₂ : ℝ) (h : d₁ > 0 ∧ d₂ > 0) :
  (π * (d₂ / 2)^2) = 49 * (π * (d₁ / 2)^2) → d₂ = 7 * d₁ := by
  sorry

end NUMINAMATH_CALUDE_circular_sign_diameter_ratio_l2359_235918


namespace NUMINAMATH_CALUDE_missing_number_proof_l2359_235976

theorem missing_number_proof (x : ℝ) : 11 + Real.sqrt (-4 + 6 * x / 3) = 13 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2359_235976


namespace NUMINAMATH_CALUDE_total_spent_is_108_l2359_235956

/-- The total amount spent by Robert and Teddy on snacks -/
def total_spent (pizza_boxes : ℕ) (pizza_price : ℕ) (robert_drinks : ℕ) (drink_price : ℕ)
                (hamburgers : ℕ) (hamburger_price : ℕ) (teddy_drinks : ℕ) : ℕ :=
  pizza_boxes * pizza_price + robert_drinks * drink_price +
  hamburgers * hamburger_price + teddy_drinks * drink_price

/-- Theorem stating that the total amount spent is $108 -/
theorem total_spent_is_108 :
  total_spent 5 10 10 2 6 3 10 = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_108_l2359_235956


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2359_235950

theorem quadratic_roots_relation (a b c d : ℝ) (h : a ≠ 0 ∧ c ≠ 0) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ c * (x/2007)^2 + d * (x/2007) + a = 0) →
  b^2 = d^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2359_235950


namespace NUMINAMATH_CALUDE_jimmy_action_figures_sale_earnings_l2359_235910

theorem jimmy_action_figures_sale_earnings :
  let regular_figure_count : ℕ := 4
  let special_figure_count : ℕ := 1
  let regular_figure_value : ℕ := 15
  let special_figure_value : ℕ := 20
  let discount : ℕ := 5

  let regular_sale_price : ℕ := regular_figure_value - discount
  let special_sale_price : ℕ := special_figure_value - discount

  let total_earnings : ℕ := regular_figure_count * regular_sale_price + special_figure_count * special_sale_price

  total_earnings = 55 := by sorry

end NUMINAMATH_CALUDE_jimmy_action_figures_sale_earnings_l2359_235910


namespace NUMINAMATH_CALUDE_semicircle_radius_l2359_235990

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  right_angle : PQ^2 + QR^2 = PR^2

-- Define the theorem
theorem semicircle_radius (t : RightTriangle) 
  (h1 : (1/2) * π * (t.PQ/2)^2 = 18*π) 
  (h2 : π * (t.QR/2) = 10*π) : 
  t.PR/2 = 4*Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l2359_235990


namespace NUMINAMATH_CALUDE_fish_count_l2359_235917

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 11

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 21 := by sorry

end NUMINAMATH_CALUDE_fish_count_l2359_235917


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_measure_l2359_235969

theorem right_triangle_acute_angle_measure (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b = 90) ∧ (a / b = 5 / 4) → min a b = 40 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_measure_l2359_235969


namespace NUMINAMATH_CALUDE_expected_threes_eight_sided_dice_l2359_235931

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The probability of rolling a 3 on a single die -/
def p : ℚ := 1 / n

/-- The probability of not rolling a 3 on a single die -/
def q : ℚ := 1 - p

/-- The expected number of 3's when rolling two n-sided dice -/
def expected_threes (n : ℕ) : ℚ := 
  2 * (p * p) + 1 * (2 * p * q) + 0 * (q * q)

/-- Theorem: The expected number of 3's when rolling two 8-sided dice is 1/4 -/
theorem expected_threes_eight_sided_dice : expected_threes n = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expected_threes_eight_sided_dice_l2359_235931


namespace NUMINAMATH_CALUDE_poll_size_l2359_235932

theorem poll_size (total : ℕ) (women_in_favor_percent : ℚ) (women_opposed : ℕ) : 
  (2 * women_opposed : ℚ) / (1 - women_in_favor_percent) = total →
  women_in_favor_percent = 35 / 100 →
  women_opposed = 39 →
  total = 120 := by
sorry

end NUMINAMATH_CALUDE_poll_size_l2359_235932


namespace NUMINAMATH_CALUDE_triangle_distance_sum_l2359_235921

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a point is inside a triangle
def isInside (t : Triangle) (p : ℝ × ℝ) : Prop := sorry

-- Define a function to calculate the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_distance_sum (t : Triangle) (M : ℝ × ℝ) :
  isInside t M →
  distance M t.A + distance M t.B + distance M t.C > perimeter t / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_distance_sum_l2359_235921


namespace NUMINAMATH_CALUDE_alvarez_diesel_consumption_l2359_235923

/-- Given that Mr. Alvarez spends $36 on diesel fuel each week and the cost of diesel fuel is $3 per gallon,
    prove that he uses 24 gallons of diesel fuel in two weeks. -/
theorem alvarez_diesel_consumption
  (weekly_expenditure : ℝ)
  (cost_per_gallon : ℝ)
  (h1 : weekly_expenditure = 36)
  (h2 : cost_per_gallon = 3)
  : (weekly_expenditure / cost_per_gallon) * 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_alvarez_diesel_consumption_l2359_235923


namespace NUMINAMATH_CALUDE_matrix_determinant_l2359_235947

theorem matrix_determinant : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -2, 3]
  Matrix.det A = 32 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_l2359_235947


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_by_nine_l2359_235954

theorem least_addition_for_divisibility_by_nine :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), (228712 + m) % 9 = 0 → m ≥ n) ∧
  (228712 + n) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_by_nine_l2359_235954


namespace NUMINAMATH_CALUDE_retail_price_calculation_l2359_235936

/-- Proves that the retail price of a machine is $144 given the specified conditions --/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  wholesale_price = 108 →
  discount_rate = 0.1 →
  profit_rate = 0.2 →
  ∃ (retail_price : ℝ),
    retail_price = 144 ∧
    wholesale_price * (1 + profit_rate) = retail_price * (1 - discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l2359_235936


namespace NUMINAMATH_CALUDE_carly_job_applications_l2359_235984

/-- The number of job applications Carly sent to companies in her state -/
def in_state_applications : ℕ := 200

/-- The number of job applications Carly sent to companies in other states -/
def out_state_applications : ℕ := 2 * in_state_applications

/-- The total number of job applications Carly sent -/
def total_applications : ℕ := in_state_applications + out_state_applications

theorem carly_job_applications : total_applications = 600 := by
  sorry

end NUMINAMATH_CALUDE_carly_job_applications_l2359_235984


namespace NUMINAMATH_CALUDE_prob_four_green_out_of_seven_l2359_235909

/-- The probability of drawing exactly 4 green marbles out of 7 draws, with replacement,
    from a bag containing 10 green marbles and 5 purple marbles. -/
theorem prob_four_green_out_of_seven (total_marbles : ℕ) (green_marbles : ℕ) (purple_marbles : ℕ)
  (h1 : total_marbles = green_marbles + purple_marbles)
  (h2 : green_marbles = 10)
  (h3 : purple_marbles = 5)
  (h4 : total_marbles > 0) :
  (Nat.choose 7 4 : ℚ) * (green_marbles / total_marbles : ℚ)^4 * (purple_marbles / total_marbles : ℚ)^3 =
  35 * (2/3 : ℚ)^4 * (1/3 : ℚ)^3 :=
by sorry

end NUMINAMATH_CALUDE_prob_four_green_out_of_seven_l2359_235909


namespace NUMINAMATH_CALUDE_eighteenth_replacement_in_december_l2359_235926

/-- Represents months as integers from 1 to 12 -/
def Month := Fin 12

/-- Convert a number of months to a Month value -/
def monthsToMonth (n : ℕ) : Month :=
  ⟨(n - 1) % 12 + 1, by sorry⟩

/-- January represented as a Month -/
def january : Month := ⟨1, by sorry⟩

/-- December represented as a Month -/
def december : Month := ⟨12, by sorry⟩

/-- The number of months between replacements -/
def replacementInterval : ℕ := 7

/-- The number of the replacement we're interested in -/
def targetReplacement : ℕ := 18

theorem eighteenth_replacement_in_december :
  monthsToMonth (replacementInterval * (targetReplacement - 1) + 1) = december := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_replacement_in_december_l2359_235926


namespace NUMINAMATH_CALUDE_negative_eight_interpretations_l2359_235915

theorem negative_eight_interpretations :
  (-(- 8) = -(-8)) ∧
  (-(- 8) = -1 * (-8)) ∧
  (-(- 8) = |-8|) ∧
  (-(- 8) = 8) :=
by sorry

end NUMINAMATH_CALUDE_negative_eight_interpretations_l2359_235915


namespace NUMINAMATH_CALUDE_max_good_quadratics_less_than_500_l2359_235927

/-- A good quadratic trinomial has distinct coefficients and two distinct real roots -/
def is_good_quadratic (a b c : ℕ+) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b.val : ℝ)^2 > 4 * (a.val : ℝ) * (c.val : ℝ)

/-- The set of 10 positive integers from which coefficients are chosen -/
def coefficient_set : Finset ℕ+ :=
  sorry

/-- The set of all good quadratic trinomials formed from the coefficient set -/
def good_quadratics : Finset (ℕ+ × ℕ+ × ℕ+) :=
  sorry

theorem max_good_quadratics_less_than_500 :
  Finset.card good_quadratics < 500 :=
sorry

end NUMINAMATH_CALUDE_max_good_quadratics_less_than_500_l2359_235927


namespace NUMINAMATH_CALUDE_sophie_donuts_l2359_235957

/-- The number of donuts left for Sophie after buying boxes and giving some away. -/
def donuts_left (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_given : ℕ) (donuts_given : ℕ) : ℕ :=
  (total_boxes - boxes_given) * donuts_per_box - donuts_given

/-- Theorem stating that Sophie has 30 donuts left. -/
theorem sophie_donuts :
  donuts_left 4 12 1 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sophie_donuts_l2359_235957


namespace NUMINAMATH_CALUDE_initial_men_count_l2359_235973

/-- The number of men initially doing the work -/
def initial_men : ℕ := 36

/-- The time taken by the initial group of men to complete the work -/
def initial_time : ℕ := 25

/-- The number of men in the second group -/
def second_group : ℕ := 15

/-- The time taken by the second group to complete the work -/
def second_time : ℕ := 60

/-- Theorem stating that the initial number of men is 36 -/
theorem initial_men_count : initial_men = 36 := by
  sorry

#check initial_men_count

end NUMINAMATH_CALUDE_initial_men_count_l2359_235973


namespace NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_difference_l2359_235980

theorem quadratic_equation_from_sum_and_difference (x y : ℝ) 
  (sum_cond : x + y = 10) 
  (diff_cond : |x - y| = 12) : 
  (∀ z : ℝ, (z - x) * (z - y) = 0 ↔ z^2 - 10*z - 11 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_difference_l2359_235980


namespace NUMINAMATH_CALUDE_three_buildings_height_l2359_235963

/-- The height of three buildings given specific conditions -/
theorem three_buildings_height 
  (h1 : ℕ) -- Height of the first building
  (h2_eq : h2 = 2 * h1) -- Second building is twice as tall as the first
  (h3_eq : h3 = 3 * (h1 + h2)) -- Third building is three times as tall as the first two combined
  (h1_val : h1 = 600) -- First building is 600 feet tall
  : h1 + h2 + h3 = 7200 := by
  sorry

#check three_buildings_height

end NUMINAMATH_CALUDE_three_buildings_height_l2359_235963


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l2359_235971

theorem perfect_square_quadratic (x k : ℝ) : 
  (∃ b : ℝ, ∀ x, x^2 - 20*x + k = (x + b)^2) ↔ k = 100 := by sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l2359_235971


namespace NUMINAMATH_CALUDE_illumination_theorem_l2359_235901

/-- Represents a rectangular room with a point light source and a mirror --/
structure IlluminatedRoom where
  length : ℝ
  width : ℝ
  height : ℝ
  mirror_width : ℝ
  light_source : ℝ × ℝ × ℝ

/-- Calculates the fraction of walls not illuminated in the room --/
def fraction_not_illuminated (room : IlluminatedRoom) : ℚ :=
  17 / 32

/-- Theorem stating that the fraction of walls not illuminated is 17/32 --/
theorem illumination_theorem (room : IlluminatedRoom) :
  fraction_not_illuminated room = 17 / 32 := by
  sorry

end NUMINAMATH_CALUDE_illumination_theorem_l2359_235901


namespace NUMINAMATH_CALUDE_square_minus_self_sum_l2359_235929

theorem square_minus_self_sum : (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_self_sum_l2359_235929


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_no_negative_nine_l2359_235928

theorem greatest_integer_b_for_no_negative_nine : ∃ (b : ℤ), 
  (∀ x : ℝ, 3 * x^2 + b * x + 15 ≠ -9) ∧
  (∀ c : ℤ, c > b → ∃ x : ℝ, 3 * x^2 + c * x + 15 = -9) ∧
  b = 16 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_no_negative_nine_l2359_235928


namespace NUMINAMATH_CALUDE_intercept_length_is_four_l2359_235907

/-- A parabola with equation y = x² + mx and axis of symmetry x = 2 -/
structure Parabola where
  m : ℝ
  axis_of_symmetry : m = -4

/-- The length of the segment intercepted by the parabola on the x-axis -/
def intercept_length (p : Parabola) : ℝ :=
  let x₁ := 0
  let x₂ := 4
  x₂ - x₁

/-- Theorem stating that the length of the intercepted segment is 4 -/
theorem intercept_length_is_four (p : Parabola) : intercept_length p = 4 := by
  sorry

end NUMINAMATH_CALUDE_intercept_length_is_four_l2359_235907


namespace NUMINAMATH_CALUDE_queens_bounding_rectangle_l2359_235903

theorem queens_bounding_rectangle (a : Fin 2004 → Fin 2004) 
  (h_perm : Function.Bijective a) 
  (h_diag : ∀ i j : Fin 2004, i ≠ j → |a i - a j| ≠ |i - j|) :
  ∃ i j : Fin 2004, |i - j| + |a i - a j| = 2004 := by
  sorry

end NUMINAMATH_CALUDE_queens_bounding_rectangle_l2359_235903


namespace NUMINAMATH_CALUDE_sandbox_area_l2359_235961

/-- The area of a rectangle with length 312 cm and width 146 cm is 45552 square centimeters. -/
theorem sandbox_area :
  let length : ℕ := 312
  let width : ℕ := 146
  length * width = 45552 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_area_l2359_235961


namespace NUMINAMATH_CALUDE_tom_initial_dimes_l2359_235988

/-- Represents the number of coins Tom has -/
structure TomCoins where
  initial_pennies : ℕ
  initial_dimes : ℕ
  dad_dimes : ℕ
  dad_nickels : ℕ
  final_dimes : ℕ

/-- The theorem states that given the conditions from the problem,
    Tom's initial number of dimes was 15 -/
theorem tom_initial_dimes (coins : TomCoins)
  (h1 : coins.initial_pennies = 27)
  (h2 : coins.dad_dimes = 33)
  (h3 : coins.dad_nickels = 49)
  (h4 : coins.final_dimes = 48)
  (h5 : coins.final_dimes = coins.initial_dimes + coins.dad_dimes) :
  coins.initial_dimes = 15 := by
  sorry


end NUMINAMATH_CALUDE_tom_initial_dimes_l2359_235988


namespace NUMINAMATH_CALUDE_remaining_money_for_sharpeners_l2359_235967

def total_money : ℕ := 100
def notebook_price : ℕ := 5
def notebooks_bought : ℕ := 4
def eraser_price : ℕ := 4
def erasers_bought : ℕ := 10
def highlighter_cost : ℕ := 30

def heaven_notebook_cost : ℕ := notebook_price * notebooks_bought
def brother_eraser_cost : ℕ := eraser_price * erasers_bought
def brother_total_cost : ℕ := brother_eraser_cost + highlighter_cost

theorem remaining_money_for_sharpeners :
  total_money - (heaven_notebook_cost + brother_total_cost) = 10 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_for_sharpeners_l2359_235967


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2359_235993

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x < 1}

theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x : ℝ | x < -2} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2359_235993


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l2359_235906

/-- Given points X, Y, Z, and G as the midpoint of XY, prove that the sum of the slope
    and y-intercept of the line passing through Z and G is 18/5 -/
theorem slope_intercept_sum (X Y Z G : ℝ × ℝ) : 
  X = (0, 8) → Y = (0, 0) → Z = (10, 0) → 
  G = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2) →
  let m := (G.2 - Z.2) / (G.1 - Z.1)
  let b := G.2
  m + b = 18 / 5 := by
sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l2359_235906


namespace NUMINAMATH_CALUDE_maria_friends_money_l2359_235965

/-- The amount of money Rene received from Maria -/
def rene_amount : ℕ := 300

/-- The amount of money Florence received from Maria -/
def florence_amount : ℕ := 3 * rene_amount

/-- The amount of money Isha received from Maria -/
def isha_amount : ℕ := florence_amount / 2

/-- The total amount of money Maria gave to her three friends -/
def total_amount : ℕ := isha_amount + florence_amount + rene_amount

/-- Theorem stating that the total amount Maria gave to her friends is $1650 -/
theorem maria_friends_money : total_amount = 1650 := by sorry

end NUMINAMATH_CALUDE_maria_friends_money_l2359_235965


namespace NUMINAMATH_CALUDE_distribution_plans_count_l2359_235978

def num_factories : Nat := 4
def num_classes : Nat := 3

def distribution_plans : Nat :=
  let one_class_A := num_classes * (num_factories - 1)^(num_classes - 1)
  let two_classes_A := (num_classes.choose 2) * (num_factories - 1)
  let all_classes_A := 1
  one_class_A + two_classes_A + all_classes_A

theorem distribution_plans_count : distribution_plans = 37 := by
  sorry

end NUMINAMATH_CALUDE_distribution_plans_count_l2359_235978


namespace NUMINAMATH_CALUDE_digit_selection_theorem_l2359_235951

/-- The number of digits available for selection -/
def n : ℕ := 10

/-- The number of digits to be selected -/
def k : ℕ := 4

/-- Function to calculate the number of permutations without repetition -/
def permutations_without_repetition (n k : ℕ) : ℕ := sorry

/-- Function to calculate the number of four-digit numbers without repetition -/
def four_digit_numbers_without_repetition (n k : ℕ) : ℕ := sorry

/-- Function to calculate the number of even four-digit numbers greater than 3000 without repetition -/
def even_four_digit_numbers_gt_3000_without_repetition (n k : ℕ) : ℕ := sorry

theorem digit_selection_theorem :
  permutations_without_repetition n k = 5040 ∧
  four_digit_numbers_without_repetition n k = 4356 ∧
  even_four_digit_numbers_gt_3000_without_repetition n k = 1792 := by
  sorry

end NUMINAMATH_CALUDE_digit_selection_theorem_l2359_235951


namespace NUMINAMATH_CALUDE_power_comparison_l2359_235962

theorem power_comparison : 3^15 < 10^9 ∧ 10^9 < 5^13 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l2359_235962


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2359_235995

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (2 + 3*I) / (-3 + 2*I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2359_235995


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l2359_235930

/-- The number of ways to divide n distinct objects into k groups of size m each. -/
def divide_into_groups (n k m : ℕ) : ℕ :=
  (Nat.choose n m * Nat.choose (n - m) m * Nat.choose (n - 2*m) m) / (Nat.factorial k)

/-- The number of ways to distribute n distinct objects among k people, with each person receiving m objects. -/
def distribute_among_people (n k m : ℕ) : ℕ :=
  divide_into_groups n k m * (Nat.factorial k)

theorem book_distribution_theorem :
  let n : ℕ := 6  -- number of books
  let k : ℕ := 3  -- number of groups/people
  let m : ℕ := 2  -- number of books per group/person
  divide_into_groups n k m = 15 ∧
  distribute_among_people n k m = 90 := by
  sorry


end NUMINAMATH_CALUDE_book_distribution_theorem_l2359_235930
