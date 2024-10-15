import Mathlib

namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2039_203987

/-- The number of dots in each row and column of the square array -/
def gridSize : ℕ := 5

/-- The number of different rectangles that can be formed in the grid -/
def numberOfRectangles : ℕ := (gridSize.choose 2) * (gridSize.choose 2)

/-- Theorem stating that the number of rectangles in a 5x5 grid is 100 -/
theorem rectangles_in_5x5_grid : numberOfRectangles = 100 := by sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2039_203987


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l2039_203976

theorem negative_fractions_comparison : -3/4 > -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l2039_203976


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2039_203978

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 6) :
  a 5 = 18 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2039_203978


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2039_203967

theorem trigonometric_simplification (θ : ℝ) : 
  (1 + Real.sin θ + Real.cos θ) / (1 + Real.sin θ - Real.cos θ) + 
  (1 - Real.cos θ + Real.sin θ) / (1 + Real.cos θ + Real.sin θ) = 
  2 * (Real.sin θ)⁻¹ := by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2039_203967


namespace NUMINAMATH_CALUDE_fruit_cost_l2039_203955

/-- The cost of buying apples and oranges at given prices and quantities -/
theorem fruit_cost (apple_price : ℚ) (apple_weight : ℚ) (orange_price : ℚ) (orange_weight : ℚ)
  (apple_buy : ℚ) (orange_buy : ℚ) :
  apple_price = 3 →
  apple_weight = 4 →
  orange_price = 5 →
  orange_weight = 6 →
  apple_buy = 12 →
  orange_buy = 18 →
  (apple_price / apple_weight * apple_buy + orange_price / orange_weight * orange_buy : ℚ) = 24 :=
by sorry

end NUMINAMATH_CALUDE_fruit_cost_l2039_203955


namespace NUMINAMATH_CALUDE_probability_at_least_one_three_l2039_203911

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of at least one die showing a 3 when two fair dice are rolled -/
def prob_at_least_one_three : ℚ := 15 / 64

/-- Theorem stating that the probability of at least one die showing a 3
    when two fair 8-sided dice are rolled is 15/64 -/
theorem probability_at_least_one_three :
  prob_at_least_one_three = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_three_l2039_203911


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2039_203943

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 8 * Real.pi →
  Real.cos (a 2 + a 8) = -1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2039_203943


namespace NUMINAMATH_CALUDE_ratio_problem_l2039_203903

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a * b * c / (d * e * f) = 1.875)
  (h2 : a / b = 5 / 2)
  (h3 : b / c = 1 / 2)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  c / d = 0.375 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l2039_203903


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l2039_203965

theorem product_of_sums_and_differences (P Q R S : ℝ) : 
  P = Real.sqrt 2012 + Real.sqrt 2013 →
  Q = -Real.sqrt 2012 - Real.sqrt 2013 →
  R = Real.sqrt 2012 - Real.sqrt 2013 →
  S = Real.sqrt 2013 - Real.sqrt 2012 →
  P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l2039_203965


namespace NUMINAMATH_CALUDE_square_of_105_l2039_203969

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_square_of_105_l2039_203969


namespace NUMINAMATH_CALUDE_sequence_bound_l2039_203950

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i : ℕ, i > 0 → 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j : ℕ, i > 0 → j > 0 → i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bound_l2039_203950


namespace NUMINAMATH_CALUDE_c_investment_value_l2039_203939

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Theorem stating that given the conditions of the problem, c's investment is $72,000 --/
theorem c_investment_value (p : Partnership)
  (h1 : p.a_investment = 45000)
  (h2 : p.b_investment = 63000)
  (h3 : p.total_profit = 60000)
  (h4 : p.c_profit = 24000)
  (h5 : p.c_profit * (p.a_investment + p.b_investment + p.c_investment) = p.total_profit * p.c_investment) :
  p.c_investment = 72000 := by
  sorry


end NUMINAMATH_CALUDE_c_investment_value_l2039_203939


namespace NUMINAMATH_CALUDE_max_value_of_f_l2039_203962

-- Define the function
def f (x : ℝ) : ℝ := -3 * x^2 + 6

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2039_203962


namespace NUMINAMATH_CALUDE_kevin_cards_l2039_203908

theorem kevin_cards (initial : ℕ) (found : ℕ) (total : ℕ) : 
  initial = 7 → found = 47 → total = initial + found → total = 54 := by
  sorry

end NUMINAMATH_CALUDE_kevin_cards_l2039_203908


namespace NUMINAMATH_CALUDE_pencils_sold_is_24_l2039_203984

/-- The total number of pencils sold in a school store sale -/
def total_pencils_sold : ℕ :=
  let first_group := 2  -- number of students in the first group
  let second_group := 6 -- number of students in the second group
  let third_group := 2  -- number of students in the third group
  let pencils_first := 2  -- pencils bought by each student in the first group
  let pencils_second := 3 -- pencils bought by each student in the second group
  let pencils_third := 1  -- pencils bought by each student in the third group
  first_group * pencils_first + second_group * pencils_second + third_group * pencils_third

/-- Theorem stating that the total number of pencils sold is 24 -/
theorem pencils_sold_is_24 : total_pencils_sold = 24 := by
  sorry

end NUMINAMATH_CALUDE_pencils_sold_is_24_l2039_203984


namespace NUMINAMATH_CALUDE_sum_of_three_different_digits_is_18_l2039_203912

/-- Represents a non-zero digit (1-9) -/
def NonZeroDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The sum of three different non-zero digits is 18 -/
theorem sum_of_three_different_digits_is_18 :
  ∃ (a b c : NonZeroDigit), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a.val + b.val + c.val = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_different_digits_is_18_l2039_203912


namespace NUMINAMATH_CALUDE_no_arithmetic_sequence_with_sum_n_cubed_l2039_203990

theorem no_arithmetic_sequence_with_sum_n_cubed :
  ¬ ∃ (a₁ d : ℝ), ∀ (n : ℕ), n > 0 →
    (n : ℝ) / 2 * (2 * a₁ + (n - 1) * d) = (n : ℝ)^3 :=
sorry

end NUMINAMATH_CALUDE_no_arithmetic_sequence_with_sum_n_cubed_l2039_203990


namespace NUMINAMATH_CALUDE_alex_more_pens_than_jane_l2039_203963

def alex_pens (week : Nat) : Nat :=
  4 * 2^(week - 1)

def jane_pens : Nat := 16

theorem alex_more_pens_than_jane :
  alex_pens 4 - jane_pens = 16 := by
  sorry

end NUMINAMATH_CALUDE_alex_more_pens_than_jane_l2039_203963


namespace NUMINAMATH_CALUDE_midpoint_one_sixth_to_five_sixths_l2039_203952

theorem midpoint_one_sixth_to_five_sixths :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 6
  (a + b) / 2 = (1 : ℚ) / 2 := by
sorry

end NUMINAMATH_CALUDE_midpoint_one_sixth_to_five_sixths_l2039_203952


namespace NUMINAMATH_CALUDE_parallelogram_height_l2039_203992

theorem parallelogram_height (base area : ℝ) (h_base : base = 28) (h_area : area = 896) :
  area / base = 32 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2039_203992


namespace NUMINAMATH_CALUDE_largest_angle_cosine_in_triangle_l2039_203942

theorem largest_angle_cosine_in_triangle (a b c : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : c = 7) :
  let cos_largest_angle := min (min ((a^2 + b^2 - c^2) / (2*a*b)) ((b^2 + c^2 - a^2) / (2*b*c))) ((c^2 + a^2 - b^2) / (2*c*a))
  cos_largest_angle = -1/2 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_cosine_in_triangle_l2039_203942


namespace NUMINAMATH_CALUDE_custodian_jugs_l2039_203997

/-- The number of cups a full jug can hold -/
def jug_capacity : ℕ := 40

/-- The number of students -/
def num_students : ℕ := 200

/-- The number of cups each student drinks per day -/
def cups_per_student : ℕ := 10

/-- Calculates the number of jugs needed to provide water for all students -/
def jugs_needed : ℕ := (num_students * cups_per_student) / jug_capacity

theorem custodian_jugs : jugs_needed = 50 := by
  sorry

end NUMINAMATH_CALUDE_custodian_jugs_l2039_203997


namespace NUMINAMATH_CALUDE_max_handshakes_networking_event_l2039_203934

/-- Calculate the number of handshakes in a group -/
def handshakesInGroup (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculate the total number of handshakes -/
def totalHandshakes (total : ℕ) (groupA : ℕ) (groupB : ℕ) (groupC : ℕ) : ℕ :=
  handshakesInGroup total - (handshakesInGroup groupA + handshakesInGroup groupB + handshakesInGroup groupC)

/-- Theorem stating the maximum number of handshakes under given conditions -/
theorem max_handshakes_networking_event :
  let total := 100
  let groupA := 30
  let groupB := 35
  let groupC := 35
  totalHandshakes total groupA groupB groupC = 3325 := by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_networking_event_l2039_203934


namespace NUMINAMATH_CALUDE_largest_common_term_up_to_150_l2039_203931

theorem largest_common_term_up_to_150 :
  ∀ k ∈ Finset.range 151,
    (∃ n : ℕ, k = 2 + 8 * n) ∧
    (∃ m : ℕ, k = 3 + 9 * m) →
    k ≤ 138 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_up_to_150_l2039_203931


namespace NUMINAMATH_CALUDE_train_passing_time_l2039_203973

/-- Theorem: Time taken for slower train to pass faster train's driver -/
theorem train_passing_time
  (train_length : ℝ)
  (fast_train_speed slow_train_speed : ℝ)
  (h1 : train_length = 500)
  (h2 : fast_train_speed = 45)
  (h3 : slow_train_speed = 15) :
  (train_length / ((fast_train_speed + slow_train_speed) * (1000 / 3600))) = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2039_203973


namespace NUMINAMATH_CALUDE_total_mowings_is_30_l2039_203913

/-- Represents the number of times Ned mowed a lawn in each season -/
structure SeasonalMowing where
  spring : Nat
  summer : Nat
  fall : Nat

/-- Calculates the total number of mowings for a lawn across all seasons -/
def totalMowings (s : SeasonalMowing) : Nat :=
  s.spring + s.summer + s.fall

/-- The number of times Ned mowed his front lawn in each season -/
def frontLawnMowings : SeasonalMowing :=
  { spring := 6, summer := 5, fall := 4 }

/-- The number of times Ned mowed his backyard lawn in each season -/
def backyardLawnMowings : SeasonalMowing :=
  { spring := 5, summer := 7, fall := 3 }

/-- Theorem: The total number of times Ned mowed his lawns is 30 -/
theorem total_mowings_is_30 :
  totalMowings frontLawnMowings + totalMowings backyardLawnMowings = 30 := by
  sorry


end NUMINAMATH_CALUDE_total_mowings_is_30_l2039_203913


namespace NUMINAMATH_CALUDE_N_is_composite_l2039_203948

theorem N_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (2011 * 2012 * 2013 * 2014 + 1 = a * b) := by
  sorry

end NUMINAMATH_CALUDE_N_is_composite_l2039_203948


namespace NUMINAMATH_CALUDE_max_color_transitions_l2039_203928

/-- Represents a strategy for painting fence sections -/
def PaintingStrategy := Nat → Bool

/-- The number of fence sections -/
def numSections : Nat := 100

/-- Counts the number of color transitions in a given painting strategy -/
def countTransitions (strategy : PaintingStrategy) : Nat :=
  (List.range (numSections - 1)).filter (fun i => strategy i ≠ strategy (i + 1)) |>.length

/-- Theorem stating that the maximum number of guaranteed color transitions is 49 -/
theorem max_color_transitions :
  ∃ (strategy : PaintingStrategy),
    ∀ (otherStrategy : PaintingStrategy),
      countTransitions (fun i => if i % 2 = 0 then strategy (i / 2) else otherStrategy (i / 2)) ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_max_color_transitions_l2039_203928


namespace NUMINAMATH_CALUDE_only_negative_sqrt_two_less_than_zero_l2039_203925

theorem only_negative_sqrt_two_less_than_zero :
  let numbers : List ℝ := [5, 2, 0, -Real.sqrt 2]
  (∀ x ∈ numbers, x < 0) ↔ (x = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_sqrt_two_less_than_zero_l2039_203925


namespace NUMINAMATH_CALUDE_johns_age_is_20_l2039_203989

-- Define John's age and his dad's age
def johns_age : ℕ := sorry
def dads_age : ℕ := sorry

-- State the theorem
theorem johns_age_is_20 :
  (johns_age + 30 = dads_age) →
  (johns_age + dads_age = 70) →
  johns_age = 20 := by
sorry

end NUMINAMATH_CALUDE_johns_age_is_20_l2039_203989


namespace NUMINAMATH_CALUDE_fermat_point_theorem_l2039_203995

/-- Represents a line in a plane --/
structure Line where
  -- Define a line using two points it passes through
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Represents a triangle --/
structure Triangle where
  -- Define a triangle using its three vertices
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  vertex3 : ℝ × ℝ

/-- Get the orthocenter of a triangle --/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Get the circumcenter of a triangle --/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Get the perpendicular bisector of a line segment --/
def perpendicularBisector (p1 p2 : ℝ × ℝ) : Line := sorry

/-- Get the triangles formed by four lines --/
def getTriangles (l1 l2 l3 l4 : Line) : List Triangle := sorry

/-- Check if a point lies on a line --/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- The Fermat point theorem --/
theorem fermat_point_theorem (l1 l2 l3 l4 : Line) : 
  ∃! fermatPoint : ℝ × ℝ, 
    ∀ t ∈ getTriangles l1 l2 l3 l4, 
      pointOnLine fermatPoint (perpendicularBisector (orthocenter t) (circumcenter t)) := by
  sorry

end NUMINAMATH_CALUDE_fermat_point_theorem_l2039_203995


namespace NUMINAMATH_CALUDE_interval_for_720_recordings_l2039_203968

/-- Calculates the time interval between recordings given the number of recordings in an hour -/
def timeInterval (recordings : ℕ) : ℚ :=
  3600 / recordings

/-- Theorem stating that 720 recordings in an hour results in a 5-second interval -/
theorem interval_for_720_recordings :
  timeInterval 720 = 5 := by
  sorry

end NUMINAMATH_CALUDE_interval_for_720_recordings_l2039_203968


namespace NUMINAMATH_CALUDE_three_part_division_l2039_203919

theorem three_part_division (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) 
  (h4 : A + B + C = 782) (h5 : C = 306) : 
  ∃ (k : ℝ), k > 0 ∧ A = k * A ∧ B = k * B ∧ C = k * 306 ∧ A + B = 476 := by
  sorry

end NUMINAMATH_CALUDE_three_part_division_l2039_203919


namespace NUMINAMATH_CALUDE_point_light_source_theorem_l2039_203977

/-- Represents a person with their height and shadow length -/
structure Person where
  height : ℝ
  shadowLength : ℝ

/-- Represents different types of light sources -/
inductive LightSource
  | Point
  | Other

/-- Given two people under the same light source, 
    if the shorter person has a longer shadow, 
    then the light source must be a point light -/
theorem point_light_source_theorem 
  (personA personB : Person) 
  (light : LightSource) 
  (h1 : personA.height < personB.height) 
  (h2 : personA.shadowLength > personB.shadowLength) : 
  light = LightSource.Point := by
  sorry

end NUMINAMATH_CALUDE_point_light_source_theorem_l2039_203977


namespace NUMINAMATH_CALUDE_function_value_equality_l2039_203947

theorem function_value_equality (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = 2^x - 5) → f m = 3 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_equality_l2039_203947


namespace NUMINAMATH_CALUDE_egg_difference_l2039_203930

/-- The number of eggs needed for one chocolate cake -/
def chocolate_cake_eggs : ℕ := 3

/-- The number of eggs needed for one cheesecake -/
def cheesecake_eggs : ℕ := 8

/-- The number of chocolate cakes -/
def num_chocolate_cakes : ℕ := 5

/-- The number of cheesecakes -/
def num_cheesecakes : ℕ := 9

/-- Theorem: The difference in eggs needed for 9 cheesecakes and 5 chocolate cakes is 57 -/
theorem egg_difference : 
  num_cheesecakes * cheesecake_eggs - num_chocolate_cakes * chocolate_cake_eggs = 57 := by
  sorry

end NUMINAMATH_CALUDE_egg_difference_l2039_203930


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l2039_203949

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

theorem systematic_sample_theorem (s : SystematicSample)
    (h_pop : s.population_size = 48)
    (h_sample : s.sample_size = 4)
    (h_interval : s.interval = s.population_size / s.sample_size)
    (h5 : s.contains 5)
    (h29 : s.contains 29)
    (h41 : s.contains 41) :
    s.contains 17 := by
  sorry

#check systematic_sample_theorem

end NUMINAMATH_CALUDE_systematic_sample_theorem_l2039_203949


namespace NUMINAMATH_CALUDE_parabola_c_is_one_l2039_203946

/-- A parabola with equation y = 2x^2 + c and vertex at (0, 1) -/
structure Parabola where
  c : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  eq_vertex : vertex_y = 2 * vertex_x^2 + c
  is_vertex_zero_one : vertex_x = 0 ∧ vertex_y = 1

/-- The value of c for a parabola with equation y = 2x^2 + c and vertex at (0, 1) is 1 -/
theorem parabola_c_is_one (p : Parabola) : p.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_is_one_l2039_203946


namespace NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l2039_203938

theorem greatest_perimeter_of_special_triangle :
  ∀ (a b c : ℕ+),
  (a = 4 * b ∨ b = 4 * a ∨ a = 4 * c ∨ c = 4 * a ∨ b = 4 * c ∨ c = 4 * b) →
  (a = 18 ∨ b = 18 ∨ c = 18) →
  (a + b > c) →
  (a + c > b) →
  (b + c > a) →
  (a + b + c ≤ 43) :=
by sorry

end NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l2039_203938


namespace NUMINAMATH_CALUDE_money_exchange_problem_l2039_203936

/-- Proves that given 100 one-hundred-yuan bills exchanged for twenty-yuan and fifty-yuan bills
    totaling 260 bills, the number of twenty-yuan bills is 100 and the number of fifty-yuan bills is 160. -/
theorem money_exchange_problem (x y : ℕ) 
  (h1 : x + y = 260)
  (h2 : 20 * x + 50 * y = 100 * 100) :
  x = 100 ∧ y = 160 := by
  sorry

end NUMINAMATH_CALUDE_money_exchange_problem_l2039_203936


namespace NUMINAMATH_CALUDE_dance_girls_fraction_l2039_203915

theorem dance_girls_fraction (colfax_total : ℕ) (winthrop_total : ℕ)
  (colfax_boy_ratio colfax_girl_ratio : ℕ)
  (winthrop_boy_ratio winthrop_girl_ratio : ℕ)
  (h1 : colfax_total = 270)
  (h2 : winthrop_total = 180)
  (h3 : colfax_boy_ratio = 5 ∧ colfax_girl_ratio = 4)
  (h4 : winthrop_boy_ratio = 4 ∧ winthrop_girl_ratio = 5) :
  let colfax_girls := colfax_total * colfax_girl_ratio / (colfax_boy_ratio + colfax_girl_ratio)
  let winthrop_girls := winthrop_total * winthrop_girl_ratio / (winthrop_boy_ratio + winthrop_girl_ratio)
  let total_girls := colfax_girls + winthrop_girls
  let total_students := colfax_total + winthrop_total
  (total_girls : ℚ) / total_students = 22 / 45 := by
sorry

end NUMINAMATH_CALUDE_dance_girls_fraction_l2039_203915


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2039_203932

theorem absolute_value_inequality (x : ℝ) : 
  |((7 - x) / 5)| < 3 ↔ -8 < x ∧ x < 22 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2039_203932


namespace NUMINAMATH_CALUDE_kaylee_biscuits_l2039_203996

def biscuit_problem (total_needed : ℕ) (lemon_sold : ℕ) (chocolate_sold : ℕ) (oatmeal_sold : ℕ) : ℕ :=
  total_needed - (lemon_sold + chocolate_sold + oatmeal_sold)

theorem kaylee_biscuits :
  biscuit_problem 33 12 5 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_kaylee_biscuits_l2039_203996


namespace NUMINAMATH_CALUDE_worm_length_difference_l2039_203918

def worm_lengths : List ℝ := [0.8, 0.1, 1.2, 0.4, 0.7]

theorem worm_length_difference : 
  let max_length := worm_lengths.maximum?
  let min_length := worm_lengths.minimum?
  ∀ max min, max_length = some max → min_length = some min →
    max - min = 1.1 := by sorry

end NUMINAMATH_CALUDE_worm_length_difference_l2039_203918


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2039_203900

-- Define the quadratic polynomial q(x)
def q (x : ℚ) : ℚ := -15/14 * x^2 - 75/14 * x + 180/7

-- Theorem stating that q(x) satisfies the given conditions
theorem quadratic_polynomial_satisfies_conditions :
  q (-8) = 0 ∧ q 3 = 0 ∧ q 6 = -45 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2039_203900


namespace NUMINAMATH_CALUDE_intersection_points_form_rectangle_l2039_203985

/-- The set of points satisfying xy = 18 and x^2 + y^2 = 45 -/
def IntersectionPoints : Set (ℝ × ℝ) :=
  {p | p.1 * p.2 = 18 ∧ p.1^2 + p.2^2 = 45}

/-- A function to check if four points form a rectangle -/
def IsRectangle (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let d12 := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2
  let d23 := (p2.1 - p3.1)^2 + (p2.2 - p3.2)^2
  let d34 := (p3.1 - p4.1)^2 + (p3.2 - p4.2)^2
  let d41 := (p4.1 - p1.1)^2 + (p4.2 - p1.2)^2
  let d13 := (p1.1 - p3.1)^2 + (p1.2 - p3.2)^2
  let d24 := (p2.1 - p4.1)^2 + (p2.2 - p4.2)^2
  (d12 = d34 ∧ d23 = d41) ∧ (d13 = d24)

theorem intersection_points_form_rectangle :
  ∃ p1 p2 p3 p4 : ℝ × ℝ, p1 ∈ IntersectionPoints ∧ p2 ∈ IntersectionPoints ∧
    p3 ∈ IntersectionPoints ∧ p4 ∈ IntersectionPoints ∧
    IsRectangle p1 p2 p3 p4 :=
  sorry

end NUMINAMATH_CALUDE_intersection_points_form_rectangle_l2039_203985


namespace NUMINAMATH_CALUDE_truck_speed_problem_l2039_203971

/-- Proves that the speed of Truck Y is 53 miles per hour given the problem conditions -/
theorem truck_speed_problem (initial_distance : ℝ) (truck_x_speed : ℝ) (overtake_time : ℝ) (final_lead : ℝ) :
  initial_distance = 13 →
  truck_x_speed = 47 →
  overtake_time = 3 →
  final_lead = 5 →
  (initial_distance + truck_x_speed * overtake_time + final_lead) / overtake_time = 53 := by
  sorry

#check truck_speed_problem

end NUMINAMATH_CALUDE_truck_speed_problem_l2039_203971


namespace NUMINAMATH_CALUDE_third_car_year_l2039_203970

def first_car_year : ℕ := 1970
def years_between_first_and_second : ℕ := 10
def years_between_second_and_third : ℕ := 20

theorem third_car_year :
  first_car_year + years_between_first_and_second + years_between_second_and_third = 2000 :=
by sorry

end NUMINAMATH_CALUDE_third_car_year_l2039_203970


namespace NUMINAMATH_CALUDE_andrew_stickers_distribution_l2039_203979

/-- The number of stickers Andrew bought -/
def total_stickers : ℕ := 750

/-- The number of stickers Andrew kept -/
def kept_stickers : ℕ := 130

/-- The number of additional stickers Fred received compared to Daniel -/
def extra_stickers : ℕ := 120

/-- The number of stickers Daniel received -/
def daniel_stickers : ℕ := 250

theorem andrew_stickers_distribution :
  daniel_stickers + (daniel_stickers + extra_stickers) + kept_stickers = total_stickers :=
by sorry

end NUMINAMATH_CALUDE_andrew_stickers_distribution_l2039_203979


namespace NUMINAMATH_CALUDE_min_value_theorem_l2039_203988

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y - 2 = 0) :
  2/x + 9/y ≥ 25/2 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2039_203988


namespace NUMINAMATH_CALUDE_complex_on_negative_diagonal_l2039_203982

/-- A complex number z = a - ai corresponds to a point on the line y = -x in the complex plane. -/
theorem complex_on_negative_diagonal (a : ℝ) : 
  let z : ℂ := a - a * I
  (z.re, z.im) ∈ {p : ℝ × ℝ | p.2 = -p.1} :=
by
  sorry

end NUMINAMATH_CALUDE_complex_on_negative_diagonal_l2039_203982


namespace NUMINAMATH_CALUDE_max_sides_with_four_obtuse_angles_l2039_203980

-- Define a convex polygon type
structure ConvexPolygon where
  sides : ℕ
  interior_angles : Fin sides → Real
  is_convex : Bool
  obtuse_count : ℕ

-- Define the theorem
theorem max_sides_with_four_obtuse_angles 
  (p : ConvexPolygon) 
  (h1 : p.is_convex = true) 
  (h2 : p.obtuse_count = 4) 
  (h3 : ∀ i, 0 < p.interior_angles i ∧ p.interior_angles i < 180) 
  (h4 : (Finset.sum Finset.univ p.interior_angles) = (p.sides - 2) * 180) :
  p.sides ≤ 7 := by
  sorry


end NUMINAMATH_CALUDE_max_sides_with_four_obtuse_angles_l2039_203980


namespace NUMINAMATH_CALUDE_bisection_method_accuracy_l2039_203954

theorem bisection_method_accuracy (initial_interval_width : ℝ) (desired_accuracy : ℝ) : 
  initial_interval_width = 2 →
  desired_accuracy = 0.1 →
  ∃ n : ℕ, (n ≥ 5 ∧ initial_interval_width / (2^n : ℝ) < desired_accuracy) ∧
           ∀ m : ℕ, m < 5 → initial_interval_width / (2^m : ℝ) ≥ desired_accuracy :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_accuracy_l2039_203954


namespace NUMINAMATH_CALUDE_fencemaker_problem_l2039_203921

theorem fencemaker_problem (length width : ℝ) : 
  width = 40 → 
  length * width = 200 → 
  2 * length + width = 50 := by
sorry

end NUMINAMATH_CALUDE_fencemaker_problem_l2039_203921


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l2039_203956

/-- Represents a person in the arrangement -/
inductive Person
| Man (n : Fin 4)
| Woman (n : Fin 4)

/-- A circular arrangement of people -/
def CircularArrangement := List Person

/-- Checks if two people can be adjacent in the arrangement -/
def canBeAdjacent (p1 p2 : Person) : Bool :=
  match p1, p2 with
  | Person.Man _, Person.Woman _ => true
  | Person.Woman _, Person.Man _ => true
  | _, _ => false

/-- Checks if a circular arrangement is valid -/
def isValidArrangement (arr : CircularArrangement) : Bool :=
  arr.length = 8 ∧
  arr.all (fun p => match p with
    | Person.Man n => n.val < 4
    | Person.Woman n => n.val < 4) ∧
  (List.zip arr (arr.rotateLeft 1)).all (fun (p1, p2) => canBeAdjacent p1 p2) ∧
  (List.zip arr (arr.rotateLeft 1)).all (fun (p1, p2) =>
    match p1, p2 with
    | Person.Man n1, Person.Woman n2 => n1 ≠ n2
    | Person.Woman n1, Person.Man n2 => n1 ≠ n2
    | _, _ => true)

/-- Counts the number of valid circular arrangements -/
def countValidArrangements : Nat :=
  (List.filter isValidArrangement (List.permutations (List.map Person.Man (List.range 4) ++ List.map Person.Woman (List.range 4)))).length / 8

theorem valid_arrangements_count :
  countValidArrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l2039_203956


namespace NUMINAMATH_CALUDE_min_cuts_for_ten_pieces_l2039_203958

/-- The number of pieces resulting from n vertical cuts on a cylindrical cake -/
def num_pieces (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The minimum number of vertical cuts needed to divide a cylindrical cake into at least 10 pieces -/
theorem min_cuts_for_ten_pieces : ∃ (n : ℕ), n ≥ 4 ∧ num_pieces n ≥ 10 ∧ ∀ (m : ℕ), m < n → num_pieces m < 10 :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_ten_pieces_l2039_203958


namespace NUMINAMATH_CALUDE_olivias_remaining_money_l2039_203907

-- Define the given values
def initial_amount : ℝ := 500
def groceries_cost : ℝ := 125
def shoe_original_price : ℝ := 150
def shoe_discount : ℝ := 0.2
def belt_price : ℝ := 35
def jacket_price : ℝ := 85
def exchange_rate : ℝ := 1.2

-- Define the calculation steps
def discounted_shoe_price : ℝ := shoe_original_price * (1 - shoe_discount)
def total_clothing_cost : ℝ := (discounted_shoe_price + belt_price + jacket_price) * exchange_rate
def total_spent : ℝ := groceries_cost + total_clothing_cost
def remaining_amount : ℝ := initial_amount - total_spent

-- Theorem statement
theorem olivias_remaining_money :
  remaining_amount = 87 :=
by sorry

end NUMINAMATH_CALUDE_olivias_remaining_money_l2039_203907


namespace NUMINAMATH_CALUDE_system_solution_l2039_203941

theorem system_solution :
  ∀ x y z : ℚ,
  (x * y + 1 = 2 * z ∧
   y * z + 1 = 2 * x ∧
   z * x + 1 = 2 * y) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨
   (x = -2 ∧ y = -2 ∧ z = 5/2) ∨
   (x = -2 ∧ y = 5/2 ∧ z = -2) ∨
   (x = 5/2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2039_203941


namespace NUMINAMATH_CALUDE_angle_4_value_l2039_203966

theorem angle_4_value (angle_1 angle_2 angle_3 angle_4 angle_A angle_B : ℝ) :
  angle_1 + angle_2 = 180 →
  angle_3 = angle_4 →
  angle_3 = (1 / 2) * angle_4 →
  angle_A = 80 →
  angle_B = 50 →
  angle_4 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_4_value_l2039_203966


namespace NUMINAMATH_CALUDE_statue_weight_theorem_l2039_203998

/-- The weight of a statue after three weeks of carving --/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  initial_weight * (1 - 0.3) * (1 - 0.2) * (1 - 0.25)

/-- Theorem stating the weight of the final statue --/
theorem statue_weight_theorem :
  final_statue_weight 250 = 105 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_theorem_l2039_203998


namespace NUMINAMATH_CALUDE_money_sharing_l2039_203933

theorem money_sharing (jessica kevin laura total : ℕ) : 
  jessica + kevin + laura = total →
  jessica = 45 →
  3 * kevin = 4 * jessica →
  3 * laura = 9 * jessica →
  total = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_money_sharing_l2039_203933


namespace NUMINAMATH_CALUDE_i_13_times_1_plus_i_l2039_203986

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem i_13_times_1_plus_i : i^13 * (1 + i) = -1 + i := by sorry

end NUMINAMATH_CALUDE_i_13_times_1_plus_i_l2039_203986


namespace NUMINAMATH_CALUDE_only_solution_is_five_l2039_203916

theorem only_solution_is_five (n : ℤ) : 
  (⌊(n^2 : ℚ) / 5⌋ - ⌊(n : ℚ) / 2⌋^2 = 1) ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_only_solution_is_five_l2039_203916


namespace NUMINAMATH_CALUDE_spade_calculation_l2039_203922

-- Define the ♠ operation
def spade (x y : ℝ) : ℝ := (x + 2*y)^2 * (x - y)

-- State the theorem
theorem spade_calculation : spade 3 (spade 2 3) = 1046875 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l2039_203922


namespace NUMINAMATH_CALUDE_R_value_at_seven_l2039_203953

-- Define the function R in terms of g and S
def R (g : ℝ) (S : ℝ) : ℝ := 2 * g * S + 3

-- State the theorem
theorem R_value_at_seven :
  ∃ (g : ℝ), (R g 5 = 23) → (R g 7 = 31) := by
  sorry

end NUMINAMATH_CALUDE_R_value_at_seven_l2039_203953


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2039_203910

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 3680 ∧ percentage = 84.3 ∧ final = initial * (1 + percentage / 100) →
  final = 6782.64 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2039_203910


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l2039_203975

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₄ + a₅ = -31 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l2039_203975


namespace NUMINAMATH_CALUDE_raccoon_nut_distribution_l2039_203957

/-- Represents the number of nuts taken by each raccoon -/
structure NutDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given distribution satisfies all conditions -/
def isValidDistribution (d : NutDistribution) : Prop :=
  -- First raccoon's final nuts
  let first_final := d.first * 5 / 6 + d.second / 18 + d.third * 7 / 48
  -- Second raccoon's final nuts
  let second_final := d.first / 9 + d.second / 3 + d.third * 7 / 48
  -- Third raccoon's final nuts
  let third_final := d.first / 9 + d.second / 9 + d.third / 8
  -- All distributions result in whole numbers
  (d.first * 5 % 6 = 0) ∧ (d.second % 18 = 0) ∧ (d.third * 7 % 48 = 0) ∧
  (d.first % 9 = 0) ∧ (d.second % 3 = 0) ∧
  (d.first % 9 = 0) ∧ (d.second % 9 = 0) ∧ (d.third % 8 = 0) ∧
  -- Final ratio is 4:3:2
  (3 * first_final = 4 * second_final) ∧ (3 * first_final = 6 * third_final)

/-- The minimum total number of nuts -/
def minTotalNuts : ℕ := 864

theorem raccoon_nut_distribution :
  ∃ (d : NutDistribution), isValidDistribution d ∧
    d.first + d.second + d.third = minTotalNuts ∧
    (∀ (d' : NutDistribution), isValidDistribution d' →
      d'.first + d'.second + d'.third ≥ minTotalNuts) :=
  sorry


end NUMINAMATH_CALUDE_raccoon_nut_distribution_l2039_203957


namespace NUMINAMATH_CALUDE_man_speed_man_speed_result_l2039_203993

/-- Calculates the speed of a man given the parameters of a train passing him --/
theorem man_speed (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed := train_length / passing_time
  let man_speed_mps := relative_speed - train_speed_mps
  let man_speed_kmph := man_speed_mps * 3600 / 1000
  man_speed_kmph

/-- The speed of the man is approximately 5.976 kmph --/
theorem man_speed_result : 
  ∃ ε > 0, |man_speed 605 60 33 - 5.976| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_result_l2039_203993


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_l2039_203937

theorem sqrt_twelve_minus_sqrt_three : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_l2039_203937


namespace NUMINAMATH_CALUDE_muffin_price_is_four_l2039_203951

/-- Represents the number of muffins made by each person and the total contribution --/
structure MuffinSale where
  sasha : ℕ
  melissa : ℕ
  tiffany : ℕ
  contribution : ℕ

/-- Calculates the price per muffin given the sale information --/
def price_per_muffin (sale : MuffinSale) : ℚ :=
  sale.contribution / (sale.sasha + sale.melissa + sale.tiffany)

/-- Theorem stating that the price per muffin is $4 given the conditions --/
theorem muffin_price_is_four :
  ∀ (sale : MuffinSale),
    sale.sasha = 30 →
    sale.melissa = 4 * sale.sasha →
    sale.tiffany = (sale.sasha + sale.melissa) / 2 →
    sale.contribution = 900 →
    price_per_muffin sale = 4 := by
  sorry

end NUMINAMATH_CALUDE_muffin_price_is_four_l2039_203951


namespace NUMINAMATH_CALUDE_inequality_proof_l2039_203935

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : a + 2*b + 3*c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2039_203935


namespace NUMINAMATH_CALUDE_coconut_to_mango_ratio_l2039_203914

/-- Proves that the ratio of coconut trees to mango trees is 1:2 given the conditions --/
theorem coconut_to_mango_ratio :
  ∀ (mango_trees coconut_trees total_trees : ℕ) (ratio : ℚ),
    mango_trees = 60 →
    total_trees = 85 →
    coconut_trees = mango_trees * ratio - 5 →
    total_trees = mango_trees + coconut_trees →
    coconut_trees * 2 = mango_trees := by
  sorry

end NUMINAMATH_CALUDE_coconut_to_mango_ratio_l2039_203914


namespace NUMINAMATH_CALUDE_iris_pants_purchase_l2039_203904

/-- Represents the number of pairs of pants Iris bought -/
def num_pants : ℕ := sorry

/-- The cost of each jacket -/
def jacket_cost : ℕ := 10

/-- The number of jackets bought -/
def num_jackets : ℕ := 3

/-- The cost of each pair of shorts -/
def shorts_cost : ℕ := 6

/-- The number of pairs of shorts bought -/
def num_shorts : ℕ := 2

/-- The cost of each pair of pants -/
def pants_cost : ℕ := 12

/-- The total amount spent -/
def total_spent : ℕ := 90

theorem iris_pants_purchase :
  num_pants = 4 ∧
  num_pants * pants_cost + num_jackets * jacket_cost + num_shorts * shorts_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_iris_pants_purchase_l2039_203904


namespace NUMINAMATH_CALUDE_ones_digit_of_power_l2039_203906

-- Define a function to get the ones digit of a natural number
def onesDigit (n : ℕ) : ℕ := n % 10

-- Define the exponent
def exponent : ℕ := 22 * (11^11)

-- Theorem statement
theorem ones_digit_of_power : onesDigit (22^exponent) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_power_l2039_203906


namespace NUMINAMATH_CALUDE_function_inequality_l2039_203905

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * (deriv f x) > -f x) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2039_203905


namespace NUMINAMATH_CALUDE_xyz_divides_product_l2039_203923

/-- A proposition stating that if x, y, and z are distinct positive integers
    such that xyz divides (xy-1)(yz-1)(zx-1), then (x, y, z) is a permutation of (2, 3, 5) -/
theorem xyz_divides_product (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 → 
  x ≠ y → y ≠ z → x ≠ z →
  (x * y * z) ∣ ((x * y - 1) * (y * z - 1) * (z * x - 1)) →
  (x = 2 ∧ y = 3 ∧ z = 5) ∨ 
  (x = 2 ∧ y = 5 ∧ z = 3) ∨ 
  (x = 3 ∧ y = 2 ∧ z = 5) ∨ 
  (x = 3 ∧ y = 5 ∧ z = 2) ∨ 
  (x = 5 ∧ y = 2 ∧ z = 3) ∨ 
  (x = 5 ∧ y = 3 ∧ z = 2) := by
  sorry

#check xyz_divides_product

end NUMINAMATH_CALUDE_xyz_divides_product_l2039_203923


namespace NUMINAMATH_CALUDE_unique_5digit_number_l2039_203929

/-- A function that generates all 3-digit numbers from a list of 5 digits -/
def generate_3digit_numbers (digits : List Nat) : List Nat :=
  sorry

/-- The sum of all 3-digit numbers generated from the digits of a 5-digit number -/
def sum_3digit_numbers (n : Nat) : Nat :=
  sorry

/-- Checks if a number has 5 different non-zero digits -/
def has_5_different_nonzero_digits (n : Nat) : Prop :=
  sorry

theorem unique_5digit_number : 
  ∃! n : Nat, 
    10000 ≤ n ∧ n < 100000 ∧
    has_5_different_nonzero_digits n ∧
    n = sum_3digit_numbers n ∧
    n = 35964 :=
  sorry

end NUMINAMATH_CALUDE_unique_5digit_number_l2039_203929


namespace NUMINAMATH_CALUDE_problem_l2039_203961

def p : Prop := ∀ x : ℝ, (x > 3 ↔ x^2 > 9)
def q : Prop := ∀ a b : ℝ, (a^2 > b^2 ↔ a > b)

theorem problem : ¬(p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_problem_l2039_203961


namespace NUMINAMATH_CALUDE_monotonicity_and_range_l2039_203991

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + a * Real.log x

theorem monotonicity_and_range :
  ∀ (a : ℝ), a ≤ 0 →
  (∀ (x : ℝ), x > 0 → f (-2) x < f (-2) (1 + Real.sqrt 2) → x < 1 + Real.sqrt 2) ∧
  (∀ (x : ℝ), x > 0 → f (-2) x > f (-2) (1 + Real.sqrt 2) → x > 1 + Real.sqrt 2) ∧
  (∀ (x : ℝ), x > 0 → f a x > (1/2)*(2*Real.exp 1 + 1)*a ↔ a ∈ Set.Ioo (-2*(Real.exp 1)^2/(2*Real.exp 1 + 1)) 0) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_and_range_l2039_203991


namespace NUMINAMATH_CALUDE_power_difference_l2039_203940

theorem power_difference (a m n : ℝ) (hm : a ^ m = 3) (hn : a ^ n = 5) : 
  a ^ (m - n) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l2039_203940


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2039_203927

theorem pure_imaginary_condition (m : ℝ) : 
  (∀ z : ℂ, z = (m^2 - m : ℝ) + (m^2 - 3*m + 2 : ℝ) * I → z.re = 0 ∧ z.im ≠ 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2039_203927


namespace NUMINAMATH_CALUDE_alcohol_remaining_l2039_203974

/-- The amount of alcohol remaining after a series of pours and refills -/
def remaining_alcohol (initial_volume : ℚ) (pour_out1 : ℚ) (refill1 : ℚ) 
  (pour_out2 : ℚ) (refill2 : ℚ) (pour_out3 : ℚ) (refill3 : ℚ) : ℚ :=
  initial_volume * (1 - pour_out1) * (1 - pour_out2) * (1 - pour_out3)

/-- Theorem stating the final amount of alcohol in the bottle -/
theorem alcohol_remaining :
  remaining_alcohol 1 (1/3) (1/3) (1/3) (1/3) 1 1 = 8/27 := by
  sorry


end NUMINAMATH_CALUDE_alcohol_remaining_l2039_203974


namespace NUMINAMATH_CALUDE_athleteHitsBullseyeUncertain_l2039_203981

-- Define the type for different kinds of events
inductive EventType
  | Certain
  | Impossible
  | Uncertain

-- Define the event
def athleteHitsBullseye : EventType := EventType.Uncertain

-- Theorem statement
theorem athleteHitsBullseyeUncertain : athleteHitsBullseye = EventType.Uncertain := by
  sorry

end NUMINAMATH_CALUDE_athleteHitsBullseyeUncertain_l2039_203981


namespace NUMINAMATH_CALUDE_hexagon_regular_iff_equiangular_l2039_203924

/-- A hexagon is a polygon with 6 sides -/
structure Hexagon where
  sides : Fin 6 → ℝ
  angles : Fin 6 → ℝ

/-- A hexagon is equiangular if all its angles are equal -/
def is_equiangular (h : Hexagon) : Prop :=
  ∀ i j : Fin 6, h.angles i = h.angles j

/-- A hexagon is equilateral if all its sides are equal -/
def is_equilateral (h : Hexagon) : Prop :=
  ∀ i j : Fin 6, h.sides i = h.sides j

/-- A hexagon is regular if it is both equiangular and equilateral -/
def is_regular (h : Hexagon) : Prop :=
  is_equiangular h ∧ is_equilateral h

/-- Theorem: A hexagon is regular if and only if it is equiangular -/
theorem hexagon_regular_iff_equiangular (h : Hexagon) :
  is_regular h ↔ is_equiangular h :=
sorry

end NUMINAMATH_CALUDE_hexagon_regular_iff_equiangular_l2039_203924


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l2039_203926

/-- Given a line y = mx + 5 intersecting the ellipse 9x^2 + 16y^2 = 144,
    prove that the possible slopes m satisfy m ∈ (-∞,-1] ∪ [1,∞). -/
theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 9 * x^2 + 16 * y^2 = 144 ∧ y = m * x + 5) ↔ m ≤ -1 ∨ m ≥ 1 := by
  sorry

#check line_ellipse_intersection_slopes

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l2039_203926


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2039_203945

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) →
  (a = 1 ∧ b = 2) ∧
  (∀ c : ℝ, 
    (∀ x, x^2 - (c + 2) * x + 2 * c ≤ 0 ↔ 
      (c < 2 ∧ c ≤ x ∧ x ≤ 2) ∨
      (c = 2 ∧ x = 2) ∨
      (c > 2 ∧ 2 ≤ x ∧ x ≤ c))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2039_203945


namespace NUMINAMATH_CALUDE_smallest_even_integer_abs_inequality_l2039_203902

theorem smallest_even_integer_abs_inequality :
  ∃ (x : ℤ), 
    (∀ (y : ℤ), (y % 2 = 0 ∧ |3*y - 4| ≤ 20) → x ≤ y) ∧
    (x % 2 = 0) ∧
    (|3*x - 4| ≤ 20) ∧
    x = -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_even_integer_abs_inequality_l2039_203902


namespace NUMINAMATH_CALUDE_middle_number_proof_l2039_203964

theorem middle_number_proof (x y z : ℕ) 
  (h1 : x < y) (h2 : y < z) 
  (h3 : x + y = 14) (h4 : x + z = 20) (h5 : y + z = 22) : 
  y = 8 := by
sorry

end NUMINAMATH_CALUDE_middle_number_proof_l2039_203964


namespace NUMINAMATH_CALUDE_probability_yellow_ball_l2039_203920

theorem probability_yellow_ball (total_balls yellow_balls : ℕ) 
  (h1 : total_balls = 8)
  (h2 : yellow_balls = 5) :
  (yellow_balls : ℚ) / total_balls = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_ball_l2039_203920


namespace NUMINAMATH_CALUDE_largest_integer_less_than_150_with_remainder_5_mod_8_l2039_203909

theorem largest_integer_less_than_150_with_remainder_5_mod_8 :
  ∃ n : ℕ, n < 150 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 150 ∧ m % 8 = 5 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_150_with_remainder_5_mod_8_l2039_203909


namespace NUMINAMATH_CALUDE_system_solution_l2039_203917

theorem system_solution : 
  ∃! (x y : ℚ), (4 * x - 3 * y = -17) ∧ (5 * x + 6 * y = -4) ∧ 
  (x = -74/13) ∧ (y = -25/13) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2039_203917


namespace NUMINAMATH_CALUDE_derivative_sin_2x_minus_1_l2039_203960

theorem derivative_sin_2x_minus_1 (x : ℝ) :
  deriv (λ x => Real.sin (2 * x - 1)) x = 2 * Real.cos (2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_2x_minus_1_l2039_203960


namespace NUMINAMATH_CALUDE_plot_length_is_64_l2039_203994

/-- Proves that the length of a rectangular plot is 64 meters given the specified conditions -/
theorem plot_length_is_64 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 28 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  cost_per_meter * perimeter = total_cost →
  length = 64 := by
sorry

end NUMINAMATH_CALUDE_plot_length_is_64_l2039_203994


namespace NUMINAMATH_CALUDE_overlap_area_bound_l2039_203901

open Set

-- Define the type for rectangles
structure Rectangle where
  area : ℝ

-- Define the large rectangle
def largeRectangle : Rectangle :=
  { area := 5 }

-- Define the set of smaller rectangles
def smallRectangles : Set Rectangle :=
  { r : Rectangle | r.area = 1 }

-- State the theorem
theorem overlap_area_bound (n : ℕ) (h : n = 9) :
  ∃ (r₁ r₂ : Rectangle),
    r₁ ∈ smallRectangles ∧
    r₂ ∈ smallRectangles ∧
    r₁ ≠ r₂ ∧
    (∃ (overlap : Rectangle), overlap.area ≥ 1/9) :=
sorry

end NUMINAMATH_CALUDE_overlap_area_bound_l2039_203901


namespace NUMINAMATH_CALUDE_vector_magnitude_l2039_203944

/-- The magnitude of a 2D vector (1, 2) is √5 -/
theorem vector_magnitude : ∀ (a : ℝ × ℝ), a = (1, 2) → ‖a‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2039_203944


namespace NUMINAMATH_CALUDE_total_miles_theorem_l2039_203959

/-- The total miles run by Bill and Julia on Saturday and Sunday -/
def total_miles (bill_sunday : ℕ) : ℕ :=
  let bill_saturday := bill_sunday - 4
  let julia_sunday := 2 * bill_sunday
  bill_saturday + bill_sunday + julia_sunday

/-- Theorem: Given the conditions, Bill and Julia ran 36 miles in total -/
theorem total_miles_theorem (bill_sunday : ℕ) 
  (h1 : bill_sunday = 10) : total_miles bill_sunday = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_theorem_l2039_203959


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2039_203999

def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 3]

theorem perpendicular_vectors (m : ℝ) :
  (∀ i : Fin 2, (a i) * (a i + m * (b i)) = 0) →
  m = 5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2039_203999


namespace NUMINAMATH_CALUDE_min_value_expression_l2039_203972

theorem min_value_expression (a b c k : ℝ) 
  (h1 : a < c) (h2 : c < b) (h3 : b = k * c) (h4 : k > 1) :
  ∃ (min : ℝ), min = k^2/3 + 2 ∧ 
  ∀ (x : ℝ), ((k*c - a)^2 + (a + c)^2 + (c - a)^2) / c^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2039_203972


namespace NUMINAMATH_CALUDE_f_2_equals_13_l2039_203983

def a (k n : ℕ) : ℕ := 10^(k+1) + n^3

def b (k n : ℕ) : ℕ := (a k n) / (10^k)

def f (k : ℕ) : ℕ := sorry

theorem f_2_equals_13 : f 2 = 13 := by sorry

end NUMINAMATH_CALUDE_f_2_equals_13_l2039_203983
