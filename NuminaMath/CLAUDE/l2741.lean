import Mathlib

namespace NUMINAMATH_CALUDE_max_cells_hit_five_times_l2741_274118

/-- Represents a triangular cell in the grid -/
structure TriangularCell :=
  (id : ℕ)

/-- Represents the entire triangular grid -/
structure TriangularGrid :=
  (cells : List TriangularCell)

/-- Represents a shot fired by the marksman -/
structure Shot :=
  (target : TriangularCell)

/-- Function to determine if two cells are adjacent -/
def areAdjacent (c1 c2 : TriangularCell) : Bool :=
  sorry

/-- Function to determine where a shot lands -/
def shotLands (s : Shot) (g : TriangularGrid) : TriangularCell :=
  sorry

/-- Function to count the number of hits on a cell -/
def countHits (c : TriangularCell) (shots : List Shot) : ℕ :=
  sorry

/-- Theorem stating the maximum number of cells that can be hit exactly five times -/
theorem max_cells_hit_five_times (g : TriangularGrid) :
  (∃ (shots : List Shot), 
    (∀ c : TriangularCell, c ∈ g.cells → countHits c shots ≤ 5) ∧ 
    (∃ cells : List TriangularCell, 
      cells.length = 25 ∧ 
      (∀ c : TriangularCell, c ∈ cells → countHits c shots = 5))) ∧
  (∀ (shots : List Shot),
    ¬∃ cells : List TriangularCell, 
      cells.length > 25 ∧ 
      (∀ c : TriangularCell, c ∈ cells → countHits c shots = 5)) :=
  sorry

end NUMINAMATH_CALUDE_max_cells_hit_five_times_l2741_274118


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2741_274194

/-- The speed of a boat in still water, given its downstream travel time and distance, and the speed of the stream. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 5)
  (h3 : downstream_distance = 125) :
  downstream_distance = (boat_speed + stream_speed) * downstream_time →
  boat_speed = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2741_274194


namespace NUMINAMATH_CALUDE_direct_proportion_constant_factor_l2741_274146

theorem direct_proportion_constant_factor 
  (k : ℝ) (x y : ℝ → ℝ) (t : ℝ) :
  (∀ t, y t = k * x t) → 
  (∀ t₁ t₂, t₁ ≠ t₂ → x t₁ ≠ x t₂ → y t₁ / x t₁ = y t₂ / x t₂) :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_constant_factor_l2741_274146


namespace NUMINAMATH_CALUDE_min_value_of_two_plus_y_l2741_274105

theorem min_value_of_two_plus_y (x y : ℝ) (h1 : y > 0) (h2 : x^2 + y - 3 = 0) :
  ∀ z, z = 2 + y → z ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_two_plus_y_l2741_274105


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2741_274147

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2741_274147


namespace NUMINAMATH_CALUDE_rachel_treasures_l2741_274115

theorem rachel_treasures (points_per_treasure : ℕ) (second_level_treasures : ℕ) (total_score : ℕ) :
  points_per_treasure = 9 →
  second_level_treasures = 2 →
  total_score = 63 →
  ∃ (first_level_treasures : ℕ),
    first_level_treasures * points_per_treasure + second_level_treasures * points_per_treasure = total_score ∧
    first_level_treasures = 5 :=
by
  sorry

#check rachel_treasures

end NUMINAMATH_CALUDE_rachel_treasures_l2741_274115


namespace NUMINAMATH_CALUDE_flag_making_problem_l2741_274196

/-- The number of students in each group making flags -/
def students_per_group : ℕ := 10

/-- The total number of flags to be made -/
def total_flags : ℕ := 240

/-- The number of groups initially assigned to make flags -/
def initial_groups : ℕ := 3

/-- The number of groups after reassignment -/
def final_groups : ℕ := 2

/-- The additional number of flags each student has to make after reassignment -/
def additional_flags_per_student : ℕ := 4

theorem flag_making_problem :
  (total_flags / final_groups - total_flags / initial_groups) / students_per_group = additional_flags_per_student :=
by sorry

end NUMINAMATH_CALUDE_flag_making_problem_l2741_274196


namespace NUMINAMATH_CALUDE_die_product_divisibility_l2741_274133

def is_divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem die_product_divisibility :
  let die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
  ∀ S : Finset ℕ, S ⊆ die_numbers → S.card = 7 →
    let product := S.prod id
    (is_divisible product 192) ∧
    (∀ n > 192, ∃ T : Finset ℕ, T ⊆ die_numbers ∧ T.card = 7 ∧ ¬(is_divisible (T.prod id) n)) :=
by sorry

end NUMINAMATH_CALUDE_die_product_divisibility_l2741_274133


namespace NUMINAMATH_CALUDE_eleven_billion_scientific_notation_l2741_274148

theorem eleven_billion_scientific_notation :
  (11 : ℝ) * (10 ^ 9 : ℝ) = (1.1 : ℝ) * (10 ^ 10 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_eleven_billion_scientific_notation_l2741_274148


namespace NUMINAMATH_CALUDE_constant_sequence_l2741_274102

theorem constant_sequence (a : ℕ → ℝ) : 
  (∀ (b : ℕ → ℕ), (∀ n : ℕ, b n ≠ b (n + 1) ∧ (b n ∣ b (n + 1))) → 
    ∃ (d : ℝ), ∀ n : ℕ, a (b (n + 1)) - a (b n) = d) →
  ∃ (c : ℝ), ∀ n : ℕ, a n = c :=
by sorry

end NUMINAMATH_CALUDE_constant_sequence_l2741_274102


namespace NUMINAMATH_CALUDE_twenty_fifth_digit_is_zero_l2741_274124

/-- The decimal representation of 1/13 -/
def decimal_1_13 : ℚ := 1 / 13

/-- The decimal representation of 1/11 -/
def decimal_1_11 : ℚ := 1 / 11

/-- The sum of the decimal representations of 1/13 and 1/11 -/
def sum_decimals : ℚ := decimal_1_13 + decimal_1_11

/-- The function that returns the nth digit after the decimal point of a rational number -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem: The 25th digit after the decimal point in the sum of 1/13 and 1/11 is 0 -/
theorem twenty_fifth_digit_is_zero : nth_digit_after_decimal sum_decimals 25 = 0 := by sorry

end NUMINAMATH_CALUDE_twenty_fifth_digit_is_zero_l2741_274124


namespace NUMINAMATH_CALUDE_max_red_squares_l2741_274188

/-- A configuration of red squares on a 5x5 grid -/
def RedConfiguration := Fin 5 → Fin 5 → Bool

/-- Checks if four points form an axis-parallel rectangle -/
def isAxisParallelRectangle (p1 p2 p3 p4 : Fin 5 × Fin 5) : Bool :=
  sorry

/-- Counts the number of red squares in a configuration -/
def countRedSquares (config : RedConfiguration) : Nat :=
  sorry

/-- Checks if a configuration is valid (no axis-parallel rectangles) -/
def isValidConfiguration (config : RedConfiguration) : Prop :=
  ∀ p1 p2 p3 p4 : Fin 5 × Fin 5,
    config p1.1 p1.2 ∧ config p2.1 p2.2 ∧ config p3.1 p3.2 ∧ config p4.1 p4.2 →
    ¬isAxisParallelRectangle p1 p2 p3 p4

/-- The maximum number of red squares in a valid configuration is 12 -/
theorem max_red_squares :
  (∃ config : RedConfiguration, isValidConfiguration config ∧ countRedSquares config = 12) ∧
  (∀ config : RedConfiguration, isValidConfiguration config → countRedSquares config ≤ 12) :=
sorry

end NUMINAMATH_CALUDE_max_red_squares_l2741_274188


namespace NUMINAMATH_CALUDE_difference_of_squares_153_147_l2741_274168

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_153_147_l2741_274168


namespace NUMINAMATH_CALUDE_fish_pond_population_l2741_274138

/-- Proves that given the conditions of the fish tagging problem, the approximate number of fish in the pond is 1250 -/
theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) 
  (h1 : initial_tagged = 50)
  (h2 : second_catch = 50)
  (h3 : tagged_in_second = 2)
  (h4 : (initial_tagged : ℚ) / total_fish = (tagged_in_second : ℚ) / second_catch) :
  total_fish = 1250 := by
  sorry

#check fish_pond_population

end NUMINAMATH_CALUDE_fish_pond_population_l2741_274138


namespace NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l2741_274108

def arithmetic_sequence (a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  (a₂ - a₁) = (a₃ - a₂) ∧ (a₃ - a₂) = (a₄ - a₃) ∧ (a₄ - a₃) = (a₅ - a₄)

theorem middle_term_of_arithmetic_sequence (x z : ℝ) :
  arithmetic_sequence 23 x 38 z 53 → 38 = (23 + 53) / 2 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l2741_274108


namespace NUMINAMATH_CALUDE_living_room_set_cost_l2741_274192

/-- The total cost of a living room set -/
def total_cost (sofa_cost armchair_cost coffee_table_cost : ℕ) (num_armchairs : ℕ) : ℕ :=
  sofa_cost + num_armchairs * armchair_cost + coffee_table_cost

/-- Theorem: The total cost of the specified living room set is $2,430 -/
theorem living_room_set_cost : total_cost 1250 425 330 2 = 2430 := by
  sorry

end NUMINAMATH_CALUDE_living_room_set_cost_l2741_274192


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l2741_274127

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) : 
  total_clips = 81 → num_boxes = 9 → clips_per_box = total_clips / num_boxes → clips_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l2741_274127


namespace NUMINAMATH_CALUDE_mikaela_hourly_rate_l2741_274186

/-- Mikaela's tutoring earnings problem -/
theorem mikaela_hourly_rate :
  ∀ (hourly_rate : ℝ),
  let first_month_hours : ℝ := 35
  let second_month_hours : ℝ := first_month_hours + 5
  let total_hours : ℝ := first_month_hours + second_month_hours
  let total_earnings : ℝ := total_hours * hourly_rate
  let personal_needs_fraction : ℝ := 4/5
  let savings : ℝ := 150
  (personal_needs_fraction * total_earnings + savings = total_earnings) →
  hourly_rate = 10 := by
sorry


end NUMINAMATH_CALUDE_mikaela_hourly_rate_l2741_274186


namespace NUMINAMATH_CALUDE_polynomial_equality_l2741_274126

theorem polynomial_equality : 102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 100406401 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2741_274126


namespace NUMINAMATH_CALUDE_hexagon_four_identical_shapes_l2741_274134

/-- A regular hexagon -/
structure RegularHexagon where
  -- Add necessary fields

/-- A line segment representing a cut in the hexagon -/
structure Cut where
  -- Add necessary fields

/-- Represents a shape resulting from cuts in the hexagon -/
structure Shape where
  -- Add necessary fields

/-- Checks if two shapes are identical -/
def are_identical (s1 s2 : Shape) : Prop := sorry

/-- Checks if a cut is along a symmetry axis of the hexagon -/
def is_symmetry_axis_cut (h : RegularHexagon) (c : Cut) : Prop := sorry

/-- Theorem: A regular hexagon can be divided into four identical shapes by cutting along its symmetry axes -/
theorem hexagon_four_identical_shapes (h : RegularHexagon) :
  ∃ (c1 c2 : Cut) (s1 s2 s3 s4 : Shape),
    is_symmetry_axis_cut h c1 ∧
    is_symmetry_axis_cut h c2 ∧
    are_identical s1 s2 ∧
    are_identical s1 s3 ∧
    are_identical s1 s4 :=
  sorry

end NUMINAMATH_CALUDE_hexagon_four_identical_shapes_l2741_274134


namespace NUMINAMATH_CALUDE_shorter_leg_length_l2741_274117

/-- A right triangle that can be cut and rearranged into a square -/
structure CuttableRightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  is_right_triangle : shorter_leg > 0 ∧ longer_leg > 0
  can_form_square : shorter_leg * 2 = longer_leg

/-- Theorem: If a right triangle with longer leg 10 can be cut and rearranged 
    to form a square, then its shorter leg has length 5 -/
theorem shorter_leg_length (t : CuttableRightTriangle) 
    (h : t.longer_leg = 10) : t.shorter_leg = 5 := by
  sorry

end NUMINAMATH_CALUDE_shorter_leg_length_l2741_274117


namespace NUMINAMATH_CALUDE_gift_payment_l2741_274174

theorem gift_payment (total : ℝ) (alice bob carlos : ℝ) : 
  total = 120 ∧ 
  alice = (1/3) * (bob + carlos) ∧ 
  bob = (1/4) * (alice + carlos) ∧ 
  total = alice + bob + carlos → 
  carlos = 72 := by
sorry

end NUMINAMATH_CALUDE_gift_payment_l2741_274174


namespace NUMINAMATH_CALUDE_digit_125_of_1_17_l2741_274164

/-- The decimal representation of 1/17 -/
def decimal_rep_1_17 : List ℕ := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating sequence in the decimal representation of 1/17 -/
def repeat_length : ℕ := 16

/-- The 125th digit after the decimal point in the decimal representation of 1/17 is 4 -/
theorem digit_125_of_1_17 : 
  (decimal_rep_1_17[(125 - 1) % repeat_length]) = 4 := by sorry

end NUMINAMATH_CALUDE_digit_125_of_1_17_l2741_274164


namespace NUMINAMATH_CALUDE_expression_simplification_l2741_274129

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (x^2 / (x - 2) - x - 2) / (4 * x / (x^2 - 4)) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2741_274129


namespace NUMINAMATH_CALUDE_unsold_books_count_l2741_274166

/-- Proves that the number of unsold books is 36 given the sale conditions --/
theorem unsold_books_count (total_books : ℕ) : 
  (2 : ℚ) / 3 * total_books * (7 : ℚ) / 2 = 252 → 
  (1 : ℚ) / 3 * total_books = 36 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_count_l2741_274166


namespace NUMINAMATH_CALUDE_number_problem_l2741_274171

theorem number_problem : ∃ n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  n / sum = quotient ∧ n % sum = 30 → n = 220030 :=
by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2741_274171


namespace NUMINAMATH_CALUDE_ratio_percentage_difference_l2741_274113

theorem ratio_percentage_difference (A B : ℝ) (h : A / B = 5 / 8) :
  (B - A) / B = 37.5 / 100 ∧ (B - A) / A = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ratio_percentage_difference_l2741_274113


namespace NUMINAMATH_CALUDE_book_pages_maximum_l2741_274153

theorem book_pages_maximum (pages : ℕ) : pages ≤ 208 :=
by
  have h1 : pages ≤ 13 * 16 := by sorry
  have h2 : pages ≤ 11 * 20 := by sorry
  sorry

#check book_pages_maximum

end NUMINAMATH_CALUDE_book_pages_maximum_l2741_274153


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_sum_l2741_274169

/-- A monic cubic polynomial -/
def monicCubicPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, p x = x^3 + a*x^2 + b*x + c

/-- The main theorem -/
theorem monic_cubic_polynomial_sum (p : ℝ → ℝ) 
  (h_monic : monicCubicPolynomial p)
  (h1 : p 1 = 10)
  (h2 : p 2 = 20)
  (h3 : p 3 = 30) :
  p 0 + p 5 = 68 := by sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_sum_l2741_274169


namespace NUMINAMATH_CALUDE_eight_digit_rotation_l2741_274178

def is_coprime (a b : Nat) : Prop := Nat.gcd a b = 1

def rotate_last_to_first (n : Nat) : Nat :=
  let d := n % 10
  let k := n / 10
  d * 10^7 + k

theorem eight_digit_rotation (A B : Nat) :
  (∃ B : Nat, 
    B > 44444444 ∧ 
    is_coprime B 12 ∧ 
    A = rotate_last_to_first B) →
  (A ≤ 99999998 ∧ A ≥ 14444446) :=
by sorry

end NUMINAMATH_CALUDE_eight_digit_rotation_l2741_274178


namespace NUMINAMATH_CALUDE_paco_cookies_l2741_274189

/-- Given that Paco had 22 sweet cookies initially and ate 15 sweet cookies,
    prove that he had 7 sweet cookies left. -/
theorem paco_cookies (initial_sweet : ℕ) (eaten_sweet : ℕ) 
  (h1 : initial_sweet = 22) 
  (h2 : eaten_sweet = 15) : 
  initial_sweet - eaten_sweet = 7 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l2741_274189


namespace NUMINAMATH_CALUDE_mod_congruence_problem_l2741_274114

theorem mod_congruence_problem (n : ℕ) : 
  (123^2 * 947) % 60 = n ∧ 0 ≤ n ∧ n < 60 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_problem_l2741_274114


namespace NUMINAMATH_CALUDE_probability_not_snow_l2741_274167

theorem probability_not_snow (p_snow : ℚ) (h : p_snow = 2 / 5) : 1 - p_snow = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snow_l2741_274167


namespace NUMINAMATH_CALUDE_num_pentagons_from_circle_points_l2741_274111

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct points on the circle -/
def num_points : ℕ := 15

/-- The number of vertices in a pentagon -/
def pentagon_vertices : ℕ := 5

/-- Theorem: The number of different convex pentagons formed by selecting 5 points
    from 15 distinct points on the circumference of a circle is 3003 -/
theorem num_pentagons_from_circle_points :
  choose num_points pentagon_vertices = 3003 := by sorry

end NUMINAMATH_CALUDE_num_pentagons_from_circle_points_l2741_274111


namespace NUMINAMATH_CALUDE_charitable_woman_age_l2741_274187

theorem charitable_woman_age (x : ℚ) : 
  (x / 2 + 1) + ((x / 2 - 1) / 2 + 2) + ((x / 4 - 3 / 2) / 2 + 3) + 1 = x → x = 38 :=
by sorry

end NUMINAMATH_CALUDE_charitable_woman_age_l2741_274187


namespace NUMINAMATH_CALUDE_percentage_of_apples_after_adding_l2741_274183

/-- Given a basket of fruits with the following conditions:
  * x is the initial number of apples
  * y is the initial number of oranges
  * z is the number of oranges added
  * w is the number of apples added
  * The sum of initial apples and oranges is 30
  * The sum of added oranges and apples is 12
  * The ratio of initial apples to initial oranges is 2:1
  * The ratio of added apples to added oranges is 3:1
  Prove that the percentage of apples in the basket after adding extra fruits is (29/42) * 100 -/
theorem percentage_of_apples_after_adding (x y z w : ℕ) : 
  x + y = 30 →
  z + w = 12 →
  x = 2 * y →
  w = 3 * z →
  (x + w : ℚ) / (x + y + z + w) * 100 = 29 / 42 * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_apples_after_adding_l2741_274183


namespace NUMINAMATH_CALUDE_triangle_angle_c_l2741_274139

theorem triangle_angle_c (A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧ 
  3 * Real.sin A + 4 * Real.cos B = 6 ∧
  4 * Real.sin B + 3 * Real.cos A = 1 →
  C = π / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l2741_274139


namespace NUMINAMATH_CALUDE_certain_number_value_l2741_274122

theorem certain_number_value : ∃ x : ℝ, 15 * x = 165 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l2741_274122


namespace NUMINAMATH_CALUDE_probability_same_color_top_three_l2741_274152

def total_cards : ℕ := 52
def cards_per_color : ℕ := 26

theorem probability_same_color_top_three (total : ℕ) (per_color : ℕ) 
  (h1 : total = 52) 
  (h2 : per_color = 26) 
  (h3 : total = 2 * per_color) :
  (2 * (per_color.choose 3)) / (total.choose 3) = 12 / 51 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_top_three_l2741_274152


namespace NUMINAMATH_CALUDE_product_of_fractions_l2741_274109

theorem product_of_fractions : (1 / 3) * (1 / 4) * (1 / 5) * (1 / 6) = 1 / 360 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2741_274109


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2741_274161

def a (m : ℝ) : ℝ × ℝ := (1, m + 2)
def b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem parallel_vectors_magnitude (m : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ a m = k • b m) →
  Real.sqrt ((b m).1^2 + (b m).2^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2741_274161


namespace NUMINAMATH_CALUDE_total_points_is_24_l2741_274185

/-- Calculates points earned based on pounds recycled and points per set of pounds -/
def calculatePoints (pounds : ℕ) (poundsPerSet : ℕ) (pointsPerSet : ℕ) : ℕ :=
  (pounds / poundsPerSet) * pointsPerSet

/-- Represents the recycling problem and calculates total points -/
def recyclingProblem : ℕ :=
  let gwenPoints := calculatePoints 12 4 2
  let lisaPoints := calculatePoints 25 5 3
  let jackPoints := calculatePoints 21 7 1
  gwenPoints + lisaPoints + jackPoints

/-- Theorem stating that the total points earned is 24 -/
theorem total_points_is_24 : recyclingProblem = 24 := by
  sorry


end NUMINAMATH_CALUDE_total_points_is_24_l2741_274185


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2741_274107

/-- A line in 3D space defined by the equation (x-2)/2 = (y-2)/(-1) = (z-4)/3 -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (2 + 2*t, 2 - t, 4 + 3*t)

/-- A plane in 3D space defined by the equation x + 3y + 5z - 42 = 0 -/
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  x + 3*y + 5*z - 42 = 0

/-- The intersection point of the line and the plane -/
def intersection_point : ℝ × ℝ × ℝ := (4, 1, 7)

theorem intersection_point_is_unique :
  ∃! t : ℝ, line t = intersection_point ∧ plane (line t) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2741_274107


namespace NUMINAMATH_CALUDE_cindys_calculation_l2741_274175

theorem cindys_calculation (x : ℝ) : 
  (x - 4) / 7 = 43 → (x - 7) / 4 = 74.5 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l2741_274175


namespace NUMINAMATH_CALUDE_contrapositive_odd_sum_even_l2741_274140

theorem contrapositive_odd_sum_even :
  (¬(∃ (a b : ℤ), Odd a ∧ Odd b ∧ ¬(Even (a + b))) ↔
   (∀ (a b : ℤ), ¬(Even (a + b)) → ¬(Odd a ∧ Odd b))) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_odd_sum_even_l2741_274140


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2741_274101

theorem min_sum_of_squares (x y : ℝ) (h : (x + 5) * (y - 5) = 0) :
  ∃ (m : ℝ), (∀ a b : ℝ, (a + 5) * (b - 5) = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 50 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2741_274101


namespace NUMINAMATH_CALUDE_peter_present_age_l2741_274193

/-- Represents the ages of Peter and Jacob -/
structure Ages where
  peter : ℕ
  jacob : ℕ

/-- The conditions of the problem -/
def age_conditions (a : Ages) : Prop :=
  ∃ (p_past j_past : ℕ),
    p_past = a.peter - 10 ∧
    j_past = a.jacob - 10 ∧
    p_past = j_past / 3 ∧
    a.jacob = a.peter + 12

/-- The theorem stating Peter's present age -/
theorem peter_present_age :
  ∀ a : Ages, age_conditions a → a.peter = 16 := by
  sorry

end NUMINAMATH_CALUDE_peter_present_age_l2741_274193


namespace NUMINAMATH_CALUDE_ellipse_constant_product_l2741_274180

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the moving line
def moving_line (k x y : ℝ) : Prop := y = k * (x - 2)

-- Define the dot product of vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

-- Theorem statement
theorem ellipse_constant_product :
  ∃ (E : ℝ × ℝ),
    E.2 = 0 ∧
    ∀ (k : ℝ) (A B : ℝ × ℝ),
      k ≠ 0 →
      ellipse_C A.1 A.2 →
      ellipse_C B.1 B.2 →
      moving_line k A.1 A.2 →
      moving_line k B.1 B.2 →
      dot_product (A.1 - E.1) (A.2 - E.2) (B.1 - E.1) (B.2 - E.2) = -5/9 :=
sorry

end NUMINAMATH_CALUDE_ellipse_constant_product_l2741_274180


namespace NUMINAMATH_CALUDE_purely_imaginary_reciprocal_l2741_274163

theorem purely_imaginary_reciprocal (m : ℝ) :
  let z : ℂ := m * (m - 1) + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → z⁻¹ = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_reciprocal_l2741_274163


namespace NUMINAMATH_CALUDE_f_max_min_values_l2741_274195

-- Define the function f(x) = |x-2| + |x-3| - |x-1|
def f (x : ℝ) : ℝ := |x - 2| + |x - 3| - |x - 1|

-- Define the condition that |x-2| + |x-3| is minimized
def is_minimized (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem f_max_min_values :
  (∃ (x : ℝ), is_minimized x) →
  (∃ (max min : ℝ), 
    (∀ (y : ℝ), is_minimized y → f y ≤ max) ∧
    (∃ (z : ℝ), is_minimized z ∧ f z = max) ∧
    (∀ (y : ℝ), is_minimized y → min ≤ f y) ∧
    (∃ (z : ℝ), is_minimized z ∧ f z = min) ∧
    max = 0 ∧ min = -1) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_values_l2741_274195


namespace NUMINAMATH_CALUDE_candidate_score_approx_45_l2741_274172

-- Define the maximum marks for Paper I
def max_marks : ℝ := 127.27

-- Define the passing percentage
def passing_percentage : ℝ := 0.55

-- Define the margin by which the candidate failed
def failing_margin : ℝ := 25

-- Define the candidate's score
def candidate_score : ℝ := max_marks * passing_percentage - failing_margin

-- Theorem to prove
theorem candidate_score_approx_45 : 
  ∃ ε > 0, abs (candidate_score - 45) < ε :=
sorry

end NUMINAMATH_CALUDE_candidate_score_approx_45_l2741_274172


namespace NUMINAMATH_CALUDE_cars_without_features_l2741_274184

theorem cars_without_features (total : ℕ) (steering : ℕ) (windows : ℕ) (both : ℕ)
  (h1 : total = 65)
  (h2 : steering = 45)
  (h3 : windows = 25)
  (h4 : both = 17) :
  total - (steering + windows - both) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cars_without_features_l2741_274184


namespace NUMINAMATH_CALUDE_complex_on_line_l2741_274123

theorem complex_on_line (z : ℂ) (a : ℝ) : 
  z = (1 - a * Complex.I) / Complex.I →
  (z.re : ℝ) + 2 * (z.im : ℝ) + 5 = 0 →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_on_line_l2741_274123


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2741_274143

/-- Given two lines that are perpendicular, prove that the value of m is 1/2 -/
theorem perpendicular_lines_m_value (m : ℝ) : 
  (∀ x y : ℝ, x - m * y + 2 * m = 0 ∨ x + 2 * y - m = 0) → 
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ - m * y₁ + 2 * m = 0 → 
    x₂ + 2 * y₂ - m = 0 → 
    (x₁ - x₂) * (y₁ - y₂) = 0) →
  m = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2741_274143


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_one_l2741_274100

theorem smallest_number_with_remainder_one : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 9 = 0 ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → n % k = 1) ∧
  (∀ m : ℕ, 0 < m ∧ m < n → ¬(m % 9 = 0 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → m % k = 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_one_l2741_274100


namespace NUMINAMATH_CALUDE_square_difference_value_l2741_274199

theorem square_difference_value (x y : ℝ) 
  (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) : 
  (x - y)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_square_difference_value_l2741_274199


namespace NUMINAMATH_CALUDE_tan_alpha_implies_sin_2alpha_plus_pi_half_l2741_274150

theorem tan_alpha_implies_sin_2alpha_plus_pi_half (α : Real) 
  (h : Real.tan α = -Real.cos α / (3 + Real.sin α)) : 
  Real.sin (2 * α + π / 2) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_implies_sin_2alpha_plus_pi_half_l2741_274150


namespace NUMINAMATH_CALUDE_banana_cost_proof_l2741_274128

/-- The cost of Tony's purchase in dollars -/
def tony_cost : ℚ := 7

/-- The number of dozen apples Tony bought -/
def tony_apples : ℕ := 2

/-- The cost of Arnold's purchase in dollars -/
def arnold_cost : ℚ := 5

/-- The number of dozen apples Arnold bought -/
def arnold_apples : ℕ := 1

/-- The number of bunches of bananas each person bought -/
def bananas : ℕ := 1

/-- The cost of a bunch of bananas in dollars -/
def banana_cost : ℚ := 3

theorem banana_cost_proof :
  banana_cost = tony_cost - arnold_cost - (tony_apples - arnold_apples) * (tony_cost - arnold_cost) :=
by
  sorry

end NUMINAMATH_CALUDE_banana_cost_proof_l2741_274128


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l2741_274120

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + (2/3) * a 2 = 3 →
  a 4^2 = (1/9) * a 3 * a 7 →
  a 4 = 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l2741_274120


namespace NUMINAMATH_CALUDE_segment_point_relation_l2741_274157

/-- Given a segment AB of length 2 and a point P on AB such that AP² = AB · PB, prove that AP = √5 - 1 -/
theorem segment_point_relation (A B P : ℝ) : 
  (0 ≤ P - A) ∧ (P - A ≤ B - A) ∧  -- P is on segment AB
  (B - A = 2) ∧                    -- AB = 2
  ((P - A)^2 = (B - A) * (B - P))  -- AP² = AB · PB
  → P - A = Real.sqrt 5 - 1 := by sorry

end NUMINAMATH_CALUDE_segment_point_relation_l2741_274157


namespace NUMINAMATH_CALUDE_ant_ratio_l2741_274116

theorem ant_ratio (abe beth cece duke : ℕ) : 
  abe = 4 →
  beth = abe + abe / 2 →
  cece = 2 * abe →
  abe + beth + cece + duke = 20 →
  duke * 2 = abe :=
by
  sorry

end NUMINAMATH_CALUDE_ant_ratio_l2741_274116


namespace NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l2741_274156

theorem ceiling_fraction_evaluation :
  (⌈(25 / 11 : ℚ) - ⌈(35 / 25 : ℚ)⌉⌉ : ℚ) /
  (⌈(35 / 11 : ℚ) + ⌈(11 * 25 / 35 : ℚ)⌉⌉ : ℚ) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l2741_274156


namespace NUMINAMATH_CALUDE_remainder_2022_power_mod_11_l2741_274176

theorem remainder_2022_power_mod_11 : 2022^(2022^2022) ≡ 5 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_remainder_2022_power_mod_11_l2741_274176


namespace NUMINAMATH_CALUDE_square_root_23_minus_one_expression_l2741_274137

theorem square_root_23_minus_one_expression : 
  let x : ℝ := Real.sqrt 23 - 1
  x^2 + 2*x + 2 = 24 := by sorry

end NUMINAMATH_CALUDE_square_root_23_minus_one_expression_l2741_274137


namespace NUMINAMATH_CALUDE_second_square_area_l2741_274131

/-- An isosceles right triangle with two inscribed squares -/
structure IsoscelesRightTriangleWithSquares where
  /-- Side length of the first inscribed square -/
  s : ℝ
  /-- Area of the first inscribed square is 484 -/
  h_area : s^2 = 484
  /-- Side length of the second inscribed square -/
  S : ℝ
  /-- The second square shares one side with the hypotenuse and its opposite vertex touches the midpoint of the hypotenuse -/
  h_S : S = (2 * s * Real.sqrt 2) / 3

/-- The area of the second inscribed square is 3872/9 -/
theorem second_square_area (triangle : IsoscelesRightTriangleWithSquares) : 
  triangle.S^2 = 3872 / 9 := by
  sorry

end NUMINAMATH_CALUDE_second_square_area_l2741_274131


namespace NUMINAMATH_CALUDE_race_track_width_l2741_274132

/-- The width of a circular race track given its inner circumference and outer radius -/
theorem race_track_width (inner_circumference outer_radius : ℝ) :
  inner_circumference = 880 →
  outer_radius = 140.0563499208679 →
  ∃ width : ℝ, abs (width - ((2 * Real.pi * outer_radius - inner_circumference) / 2)) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_race_track_width_l2741_274132


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_100_l2741_274197

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_100 : sum_factorials 100 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_100_l2741_274197


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l2741_274198

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (x + 3) / (x - 2)) ↔ x ≥ -3 ∧ x ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l2741_274198


namespace NUMINAMATH_CALUDE_equation_solution_l2741_274191

theorem equation_solution : 
  ∃! y : ℝ, (y^3 + 3*y^2) / (y^2 + 5*y + 6) + y = -8 ∧ y^2 + 5*y + 6 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2741_274191


namespace NUMINAMATH_CALUDE_berry_collection_theorem_l2741_274151

def berry_collection (total_berries : ℕ) (sergey_speed_ratio : ℕ) : Prop :=
  let sergey_picked := (2 * total_berries) / 3
  let dima_picked := total_berries / 3
  let sergey_collected := sergey_picked / 2
  let dima_collected := (2 * dima_picked) / 3
  sergey_collected - dima_collected = 100

theorem berry_collection_theorem :
  berry_collection 900 2 := by
  sorry

end NUMINAMATH_CALUDE_berry_collection_theorem_l2741_274151


namespace NUMINAMATH_CALUDE_alice_number_puzzle_l2741_274181

theorem alice_number_puzzle (x : ℝ) : 
  (((x + 3) * 3 - 3) / 3 = 10) → x = 8 := by sorry

end NUMINAMATH_CALUDE_alice_number_puzzle_l2741_274181


namespace NUMINAMATH_CALUDE_min_value_expression_l2741_274121

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^3 + 8 * b^3 + 27 * c^3 + 1 / (3 * a * b * c) ≥ 6 ∧
  (4 * a^3 + 8 * b^3 + 27 * c^3 + 1 / (3 * a * b * c) = 6 ↔
    a = 1 / Real.rpow 6 (1/3) ∧ b = 1 / Real.rpow 12 (1/3) ∧ c = 1 / Real.rpow 54 (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2741_274121


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l2741_274141

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_equals_zero :
  i^2022 + i^2023 + i^2024 + i^2025 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l2741_274141


namespace NUMINAMATH_CALUDE_kids_staying_home_l2741_274130

def total_kids : ℕ := 1059955
def camp_kids : ℕ := 564237

theorem kids_staying_home : total_kids - camp_kids = 495718 := by
  sorry

end NUMINAMATH_CALUDE_kids_staying_home_l2741_274130


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2741_274144

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem rectangular_solid_surface_area 
  (l w h : ℕ) 
  (prime_l : is_prime l) 
  (prime_w : is_prime w) 
  (prime_h : is_prime h) 
  (volume_eq : l * w * h = 1001) : 
  2 * (l * w + w * h + h * l) = 622 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2741_274144


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2741_274135

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ∀ x : ℝ, x^3 - 9*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c) ∧
    a + b + c = 9) →
  p + q = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2741_274135


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l2741_274159

/-- Calculates the total number of heartbeats during an athlete's activity --/
def totalHeartbeats (joggingHeartRate walkingHeartRate : ℕ) 
                    (walkingDuration : ℕ) 
                    (joggingDistance joggingPace : ℕ) : ℕ :=
  let joggingDuration := joggingDistance * joggingPace
  let joggingBeats := joggingDuration * joggingHeartRate
  let walkingBeats := walkingDuration * walkingHeartRate
  joggingBeats + walkingBeats

/-- Proves that the total number of heartbeats is 9900 given the specified conditions --/
theorem athlete_heartbeats :
  totalHeartbeats 120 90 30 10 6 = 9900 := by
  sorry

end NUMINAMATH_CALUDE_athlete_heartbeats_l2741_274159


namespace NUMINAMATH_CALUDE_min_value_sum_and_sqrt_l2741_274179

theorem min_value_sum_and_sqrt (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + 1/b + 2 * Real.sqrt (a * b) ≥ 4 ∧
  (1/a + 1/b + 2 * Real.sqrt (a * b) = 4 ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_and_sqrt_l2741_274179


namespace NUMINAMATH_CALUDE_min_sum_with_product_144_l2741_274160

theorem min_sum_with_product_144 :
  (∃ (a b : ℤ), a * b = 144 ∧ a + b = -145) ∧
  (∀ (a b : ℤ), a * b = 144 → a + b ≥ -145) := by
sorry

end NUMINAMATH_CALUDE_min_sum_with_product_144_l2741_274160


namespace NUMINAMATH_CALUDE_waitress_tips_l2741_274154

theorem waitress_tips (salary : ℝ) (tips : ℝ) (h1 : salary > 0) (h2 : tips > 0) :
  tips / (salary + tips) = 1/3 → tips / salary = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_waitress_tips_l2741_274154


namespace NUMINAMATH_CALUDE_milestone_solution_l2741_274158

def milestone_problem (initial_number : ℕ) (second_number : ℕ) (third_number : ℕ) : Prop :=
  let a := initial_number / 10
  let b := initial_number % 10
  (initial_number = 10 * a + b) ∧
  (second_number = 10 * b + a) ∧
  (third_number = 100 * a + b) ∧
  (0 < a) ∧ (a < 10) ∧ (0 < b) ∧ (b < 10)

theorem milestone_solution :
  ∃ (initial_number second_number : ℕ),
    milestone_problem initial_number second_number 106 :=
  sorry

end NUMINAMATH_CALUDE_milestone_solution_l2741_274158


namespace NUMINAMATH_CALUDE_students_not_in_sports_l2741_274119

/-- The number of students in the class -/
def total_students : ℕ := 50

/-- The number of students playing basketball -/
def basketball : ℕ := total_students / 2

/-- The number of students playing volleyball -/
def volleyball : ℕ := total_students / 3

/-- The number of students playing soccer -/
def soccer : ℕ := total_students / 5

/-- The number of students playing badminton -/
def badminton : ℕ := total_students / 8

/-- The number of students playing both basketball and volleyball -/
def basketball_and_volleyball : ℕ := total_students / 10

/-- The number of students playing both basketball and soccer -/
def basketball_and_soccer : ℕ := total_students / 12

/-- The number of students playing both basketball and badminton -/
def basketball_and_badminton : ℕ := total_students / 16

/-- The number of students playing both volleyball and soccer -/
def volleyball_and_soccer : ℕ := total_students / 8

/-- The number of students playing both volleyball and badminton -/
def volleyball_and_badminton : ℕ := total_students / 10

/-- The number of students playing both soccer and badminton -/
def soccer_and_badminton : ℕ := total_students / 20

/-- The number of students playing all four sports -/
def all_four_sports : ℕ := total_students / 25

/-- The theorem stating that 16 students do not engage in any of the four sports -/
theorem students_not_in_sports : 
  total_students - (basketball + volleyball + soccer + badminton 
  - basketball_and_volleyball - basketball_and_soccer - basketball_and_badminton 
  - volleyball_and_soccer - volleyball_and_badminton - soccer_and_badminton 
  + all_four_sports) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_sports_l2741_274119


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l2741_274110

/-- The area of the shaded region in a square with quarter circles at corners -/
theorem shaded_area_square_with_quarter_circles 
  (side_length : ℝ) 
  (radius : ℝ) 
  (h1 : side_length = 12) 
  (h2 : radius = 6) : 
  side_length ^ 2 - π * radius ^ 2 = 144 - 36 * π := by
  sorry

#check shaded_area_square_with_quarter_circles

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l2741_274110


namespace NUMINAMATH_CALUDE_circle_existence_theorem_l2741_274182

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- Check if a point is on a circle -/
def isOn (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Check if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- The main theorem -/
theorem circle_existence_theorem (n : ℕ) (points : Fin n → Point) 
    (h1 : n ≥ 3) 
    (h2 : ∃ p1 p2 p3 : Fin n, ¬areCollinear (points p1) (points p2) (points p3)) :
    ∃ (c : Circle) (p1 p2 p3 : Fin n), 
      p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
      isOn (points p1) c ∧ isOn (points p2) c ∧ isOn (points p3) c ∧
      ∀ (p : Fin n), p ≠ p1 → p ≠ p2 → p ≠ p3 → ¬isInside (points p) c :=
  sorry

end NUMINAMATH_CALUDE_circle_existence_theorem_l2741_274182


namespace NUMINAMATH_CALUDE_rationality_of_given_numbers_l2741_274145

theorem rationality_of_given_numbers :
  (∃ (a b : ℚ), a^2 = 4 ∧ b ≠ 0) ∧  -- √4 is rational
  (∀ (a b : ℚ), a^3 ≠ 0.5 * b^3) ∧  -- ∛0.5 is irrational
  (∃ (a b : ℚ), a^4 = 0.0625 * b^4 ∧ b ≠ 0) ∧  -- ∜0.0625 is rational
  (∃ (a b : ℚ), a^3 = -8 ∧ b^2 = 4 ∧ b ≠ 0) :=  -- ∛(-8) * √((0.25)^(-1)) is rational
by sorry

end NUMINAMATH_CALUDE_rationality_of_given_numbers_l2741_274145


namespace NUMINAMATH_CALUDE_one_correct_description_l2741_274104

/-- Represents an experimental description --/
structure ExperimentalDescription where
  id : Nat
  isCorrect : Bool

/-- The set of all experimental descriptions --/
def experimentDescriptions : Finset ExperimentalDescription := sorry

/-- Theorem stating that there is exactly one correct experimental description --/
theorem one_correct_description :
  (experimentDescriptions.filter (λ d => d.isCorrect)).card = 1 := by sorry

end NUMINAMATH_CALUDE_one_correct_description_l2741_274104


namespace NUMINAMATH_CALUDE_potential_solution_check_l2741_274155

theorem potential_solution_check (x y : ℕ+) (h : 1 + 2^x.val + 2^(2*x.val+1) = y.val^2) : 
  x = 3 ∨ ∃ z : ℕ+, (1 + 2^z.val + 2^(2*z.val+1) = y.val^2 ∧ z ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_potential_solution_check_l2741_274155


namespace NUMINAMATH_CALUDE_remove_fifteen_for_average_seven_point_five_l2741_274112

theorem remove_fifteen_for_average_seven_point_five :
  let sequence := List.range 15
  let sum := sequence.sum
  let removed := 15
  let remaining_sum := sum - removed
  let remaining_count := sequence.length - 1
  (remaining_sum : ℚ) / remaining_count = 15/2 := by
    sorry

end NUMINAMATH_CALUDE_remove_fifteen_for_average_seven_point_five_l2741_274112


namespace NUMINAMATH_CALUDE_chess_match_schedules_count_l2741_274149

/-- Represents a chess match schedule between two schools -/
structure ChessMatchSchedule where
  /-- Number of players per school -/
  players_per_school : Nat
  /-- Number of games played simultaneously in each round -/
  games_per_round : Nat
  /-- Total number of games in the match -/
  total_games : Nat
  /-- Number of rounds in the match -/
  total_rounds : Nat
  /-- Condition: Each player plays against each player from the other school -/
  player_matchup : players_per_school * players_per_school = total_games
  /-- Condition: Games are evenly distributed across rounds -/
  round_distribution : total_games = games_per_round * total_rounds

/-- The number of different ways to schedule the chess match -/
def number_of_schedules (schedule : ChessMatchSchedule) : Nat :=
  Nat.factorial schedule.total_rounds

/-- Theorem stating that there are 24 different ways to schedule the chess match -/
theorem chess_match_schedules_count :
  ∃ (schedule : ChessMatchSchedule),
    schedule.players_per_school = 4 ∧
    schedule.games_per_round = 4 ∧
    number_of_schedules schedule = 24 := by
  sorry

end NUMINAMATH_CALUDE_chess_match_schedules_count_l2741_274149


namespace NUMINAMATH_CALUDE_amount_difference_l2741_274162

theorem amount_difference (x : ℝ) (h : x = 690) : (0.25 * 1500) - (0.5 * x) = 30 := by
  sorry

end NUMINAMATH_CALUDE_amount_difference_l2741_274162


namespace NUMINAMATH_CALUDE_exam_score_percentage_l2741_274173

theorem exam_score_percentage : 
  let score1 : ℕ := 42
  let score2 : ℕ := 33
  let total_score : ℕ := score1 + score2
  (score1 : ℚ) / (total_score : ℚ) * 100 = 56 := by
sorry

end NUMINAMATH_CALUDE_exam_score_percentage_l2741_274173


namespace NUMINAMATH_CALUDE_parallel_iff_parallel_sum_l2741_274170

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def IsParallel (u v : V) : Prop :=
  ∃ (k : ℝ), v = k • u ∨ u = k • v

theorem parallel_iff_parallel_sum {a b : V} (ha : a ≠ 0) (hb : b ≠ 0) :
  IsParallel a b ↔ IsParallel a (a + b) :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_parallel_sum_l2741_274170


namespace NUMINAMATH_CALUDE_uncle_bobs_age_l2741_274106

theorem uncle_bobs_age (anna_age brianna_age caitlin_age bob_age : ℕ) : 
  anna_age = 48 →
  brianna_age = anna_age / 2 →
  caitlin_age = brianna_age - 6 →
  bob_age = 3 * caitlin_age →
  bob_age = 54 := by
sorry

end NUMINAMATH_CALUDE_uncle_bobs_age_l2741_274106


namespace NUMINAMATH_CALUDE_movie_replay_count_l2741_274103

theorem movie_replay_count (movie_length : Real) (ad_length : Real) (theater_hours : Real) :
  movie_length = 1.5 ∧ ad_length = 1/3 ∧ theater_hours = 11 →
  ⌊theater_hours * 60 / (movie_length * 60 + ad_length)⌋ = 6 := by
sorry

end NUMINAMATH_CALUDE_movie_replay_count_l2741_274103


namespace NUMINAMATH_CALUDE_truncated_cone_radius_theorem_l2741_274165

/-- Represents a cone with its base radius -/
structure Cone where
  baseRadius : ℝ

/-- Represents a truncated cone with its smaller base radius -/
structure TruncatedCone where
  smallerBaseRadius : ℝ

/-- Given three cones touching each other and a truncated cone sharing
    a common generatrix with each, compute the smaller base radius of the truncated cone -/
def computeTruncatedConeRadius (c1 c2 c3 : Cone) : ℝ :=
  6

theorem truncated_cone_radius_theorem (c1 c2 c3 : Cone) (tc : TruncatedCone) 
    (h1 : c1.baseRadius = 23)
    (h2 : c2.baseRadius = 46)
    (h3 : c3.baseRadius = 69)
    (h4 : tc.smallerBaseRadius = computeTruncatedConeRadius c1 c2 c3) :
  tc.smallerBaseRadius = 6 := by
  sorry

end NUMINAMATH_CALUDE_truncated_cone_radius_theorem_l2741_274165


namespace NUMINAMATH_CALUDE_die_roll_probability_l2741_274125

/-- The probability of rolling a 5 on a standard die -/
def prob_five : ℚ := 1/6

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The probability of not rolling a 5 in a single roll -/
def prob_not_five : ℚ := 1 - prob_five

theorem die_roll_probability : 
  1 - prob_not_five ^ num_rolls = 1288991/1679616 := by
sorry

end NUMINAMATH_CALUDE_die_roll_probability_l2741_274125


namespace NUMINAMATH_CALUDE_prob_non_first_class_l2741_274177

theorem prob_non_first_class (A B C : ℝ) 
  (hA : A = 0.65) 
  (hB : B = 0.2) 
  (hC : C = 0.1) : 
  1 - A = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_prob_non_first_class_l2741_274177


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2741_274136

/-- Given a square with perimeter 240 units divided into 4 congruent rectangles,
    where each rectangle's width is half the side length of the square,
    the perimeter of one rectangle is 180 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (rectangle_count : ℕ) 
  (h1 : square_perimeter = 240)
  (h2 : rectangle_count = 4) : ℝ :=
by
  -- Define the side length of the square
  let square_side := square_perimeter / 4

  -- Define the dimensions of each rectangle
  let rectangle_width := square_side / 2
  let rectangle_length := square_side

  -- Calculate the perimeter of one rectangle
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)

  -- Prove that the rectangle_perimeter equals 180
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l2741_274136


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2741_274142

theorem function_inequality_implies_a_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ 1) :
  (∀ (x₁ x₂ : ℝ), 1 ≤ x₁ ∧ x₁ ≤ Real.exp 1 ∧ 1 ≤ x₂ ∧ x₂ ≤ Real.exp 1 → 
    x₁ + a / x₁ ≥ x₂ - Real.log x₂) →
  Real.exp 1 - 2 ≤ a ∧ a ≤ 1 := by
  sorry

#check function_inequality_implies_a_range

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2741_274142


namespace NUMINAMATH_CALUDE_base6_addition_l2741_274190

-- Define a function to convert a base-6 number (represented as a list of digits) to a natural number
def base6ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

-- Define a function to convert a natural number to its base-6 representation
def natToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

-- State the theorem
theorem base6_addition :
  base6ToNat [4, 5, 1, 2] + base6ToNat [2, 3, 4, 5, 3] = base6ToNat [3, 4, 4, 0, 5] := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_l2741_274190
