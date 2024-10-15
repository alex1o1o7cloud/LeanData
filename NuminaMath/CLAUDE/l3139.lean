import Mathlib

namespace NUMINAMATH_CALUDE_distance_between_points_l3139_313900

/-- The distance between points (3, 24) and (10, 0) is 25. -/
theorem distance_between_points : Real.sqrt ((10 - 3)^2 + (24 - 0)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3139_313900


namespace NUMINAMATH_CALUDE_one_square_remains_l3139_313979

/-- Represents a grid with its dimensions and number of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (squares : ℕ)

/-- Represents the state of points on the grid -/
structure GridState :=
  (grid : Grid)
  (removed_points : ℕ)
  (remaining_squares : ℕ)

/-- Function to calculate the number of additional points to remove -/
def pointsToRemove (initial : GridState) (target : ℕ) : ℕ :=
  sorry

theorem one_square_remains (g : Grid) (initial : GridState) : 
  g.rows = 4 ∧ g.cols = 4 ∧ g.squares = 30 ∧ 
  initial.grid = g ∧ initial.removed_points = 4 →
  pointsToRemove initial 1 = 4 :=
sorry

end NUMINAMATH_CALUDE_one_square_remains_l3139_313979


namespace NUMINAMATH_CALUDE_min_value_expression_l3139_313934

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 4) :
  (x + 28 * y + 4) / (x * y) ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3139_313934


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_plus_constant_l3139_313930

def repeat_digit (d : Nat) (n : Nat) : Nat :=
  d * (10^n - 1) / 9

def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_product_plus_constant : 
  sum_of_digits ((repeat_digit 9 47 * repeat_digit 4 47) + 100000) = 424 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_plus_constant_l3139_313930


namespace NUMINAMATH_CALUDE_irrational_expression_l3139_313986

theorem irrational_expression (x : ℝ) : 
  Irrational ((x - 3 * Real.sqrt (x^2 + 4)) / 2) := by sorry

end NUMINAMATH_CALUDE_irrational_expression_l3139_313986


namespace NUMINAMATH_CALUDE_circle_radius_l3139_313971

/-- Given a circle with equation x^2 + y^2 - 2ax + 2 = 0 and center (2, 0), its radius is √2 -/
theorem circle_radius (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x + 2 = 0 ↔ (x - 2)^2 + y^2 = 2) → 
  (∃ r : ℝ, r > 0 ∧ r^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l3139_313971


namespace NUMINAMATH_CALUDE_inequality_proof_l3139_313939

theorem inequality_proof (a b c d : ℝ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (sum_condition : a/b + b/c + c/d + d/a = 4)
  (product_condition : a*c = b*d) :
  (a/c + b/d + c/a + d/b ≤ -12) ∧
  (∀ k : ℝ, (∀ a' b' c' d' : ℝ, 
    a'/b' + b'/c' + c'/d' + d'/a' = 4 → 
    a'*c' = b'*d' → 
    a'/c' + b'/d' + c'/a' + d'/b' ≤ k) → 
  k ≤ -12) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3139_313939


namespace NUMINAMATH_CALUDE_f_properties_l3139_313928

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem f_properties :
  (¬ (∀ x, f (-x) = -f x) ∧ ¬ (∀ x, f (-x) = f x)) ∧
  (∃ y, f 1 ≤ f y ∧ ∀ x, f 1 ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3139_313928


namespace NUMINAMATH_CALUDE_complex_roots_sum_of_absolute_values_l3139_313901

theorem complex_roots_sum_of_absolute_values (a : ℝ) (x₁ x₂ : ℂ) : 
  (∀ x : ℂ, x^2 - 2*a*x + a^2 - 4*a + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  Complex.abs x₁ + Complex.abs x₂ = 3 →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_sum_of_absolute_values_l3139_313901


namespace NUMINAMATH_CALUDE_y_intercept_range_l3139_313960

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the line y = kx + 1
def line1 (k x y : ℝ) : Prop := y = k * x + 1

-- Define the line l passing through (-2, 0) and the midpoint of AB
def line_l (m b x y : ℝ) : Prop := y = m * x + b

-- Define the condition that k is in the valid range
def valid_k (k : ℝ) : Prop := 1 < k ∧ k < Real.sqrt 2

-- Define the range of b
def b_range (b : ℝ) : Prop := b < -2 - Real.sqrt 2 ∨ b > 2

-- Main theorem
theorem y_intercept_range (k m b : ℝ) : 
  valid_k k →
  (∃ x1 y1 x2 y2 : ℝ, 
    hyperbola x1 y1 ∧ hyperbola x2 y2 ∧
    line1 k x1 y1 ∧ line1 k x2 y2 ∧
    x1 < 0 ∧ x2 < 0 ∧
    line_l m b (-2) 0 ∧
    line_l m b ((x1 + x2) / 2) ((y1 + y2) / 2)) →
  b_range b :=
sorry

end NUMINAMATH_CALUDE_y_intercept_range_l3139_313960


namespace NUMINAMATH_CALUDE_james_running_distance_l3139_313976

/-- Proves that given the conditions of the problem, the initial running distance was 600 miles per week -/
theorem james_running_distance (initial_distance : ℝ) : 
  (initial_distance + 40 * 3 = 1.2 * initial_distance) → 
  initial_distance = 600 := by
  sorry

end NUMINAMATH_CALUDE_james_running_distance_l3139_313976


namespace NUMINAMATH_CALUDE_right_angle_times_l3139_313909

/-- Represents a time on a 12-hour analog clock -/
structure ClockTime where
  hour : Fin 12
  minute : Fin 60

/-- Calculates the angle between hour and minute hands at a given time -/
def angleBetweenHands (time : ClockTime) : ℝ :=
  sorry

/-- Checks if the angle between hands is a right angle (90 degrees) -/
def isRightAngle (time : ClockTime) : Prop :=
  angleBetweenHands time = 90

/-- The theorem stating that when the hands form a right angle, the time is either 3:00 or 9:00 -/
theorem right_angle_times :
  ∀ (time : ClockTime), isRightAngle time →
    (time.hour = 3 ∧ time.minute = 0) ∨ (time.hour = 9 ∧ time.minute = 0) :=
  sorry

end NUMINAMATH_CALUDE_right_angle_times_l3139_313909


namespace NUMINAMATH_CALUDE_candy_distribution_l3139_313984

theorem candy_distribution (x : ℚ) 
  (h1 : 3 * x = mia_candies)
  (h2 : 4 * mia_candies = noah_candies)
  (h3 : 6 * noah_candies = olivia_candies)
  (h4 : x + mia_candies + noah_candies + olivia_candies = 468) :
  x = 117 / 22 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3139_313984


namespace NUMINAMATH_CALUDE_lemon_heads_package_count_l3139_313981

/-- The number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := 54

/-- The number of whole boxes Louis finished -/
def boxes_finished : ℕ := 9

/-- The number of Lemon Heads left after eating -/
def lemon_heads_left : ℕ := 0

/-- The number of Lemon Heads per package -/
def lemon_heads_per_package : ℕ := total_lemon_heads / boxes_finished

theorem lemon_heads_package_count : lemon_heads_per_package = 6 := by
  sorry

end NUMINAMATH_CALUDE_lemon_heads_package_count_l3139_313981


namespace NUMINAMATH_CALUDE_water_level_correct_water_level_rate_initial_water_level_l3139_313974

/-- Represents the water level function in a reservoir -/
def water_level (x : ℝ) : ℝ := 6 + 0.3 * x

theorem water_level_correct (x : ℝ) (h : x ≥ 0) :
  water_level x = 6 + 0.3 * x :=
by sorry

/-- The water level rises at a constant rate of 0.3 meters per hour -/
theorem water_level_rate (x y : ℝ) (hx : x ≥ 0) (hy : y > x) :
  (water_level y - water_level x) / (y - x) = 0.3 :=
by sorry

/-- The initial water level is 6 meters -/
theorem initial_water_level : water_level 0 = 6 :=
by sorry

end NUMINAMATH_CALUDE_water_level_correct_water_level_rate_initial_water_level_l3139_313974


namespace NUMINAMATH_CALUDE_box_volume_increase_l3139_313923

/-- 
A rectangular box with length l, width w, and height h.
Given the conditions:
1. Volume is 5400 cubic inches
2. Surface area is 2352 square inches
3. Sum of the lengths of its 12 edges is 240 inches
Prove that increasing each dimension by 1 inch results in a volume of 6637 cubic inches
-/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5400)
  (surface_area : 2 * l * w + 2 * w * h + 2 * h * l = 2352)
  (edge_sum : 4 * l + 4 * w + 4 * h = 240) :
  (l + 1) * (w + 1) * (h + 1) = 6637 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l3139_313923


namespace NUMINAMATH_CALUDE_prime_and_even_under_10_composite_and_odd_under_10_l3139_313911

def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k
def isOdd (n : ℕ) : Prop := ¬(isEven n)
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬(isPrime n)

theorem prime_and_even_under_10 : ∃! n, n < 10 ∧ isPrime n ∧ isEven n :=
sorry

theorem composite_and_odd_under_10 : ∃! n, n < 10 ∧ isComposite n ∧ isOdd n :=
sorry

end NUMINAMATH_CALUDE_prime_and_even_under_10_composite_and_odd_under_10_l3139_313911


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l3139_313914

/-- Represents a cube with holes -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculate the total surface area of a cube with holes -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length ^ 2
  let hole_area := 6 * cube.hole_side_length ^ 2
  let exposed_internal_area := 6 * 4 * cube.hole_side_length ^ 2
  original_surface_area - hole_area + exposed_internal_area

/-- Theorem stating the total surface area of the specific cube with holes -/
theorem cube_with_holes_surface_area :
  let cube : CubeWithHoles := { edge_length := 4, hole_side_length := 2 }
  total_surface_area cube = 168 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l3139_313914


namespace NUMINAMATH_CALUDE_reunion_boys_l3139_313916

/-- The number of handshakes when n people each shake hands with everyone else exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- There were 8 boys at the reunion -/
theorem reunion_boys : ∃ n : ℕ, n > 0 ∧ handshakes n = 28 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_reunion_boys_l3139_313916


namespace NUMINAMATH_CALUDE_bob_salary_last_year_l3139_313987

/-- Mario's salary this year -/
def mario_salary_this_year : ℝ := 4000

/-- Mario's salary increase percentage -/
def mario_increase_percentage : ℝ := 0.40

/-- Bob's salary last year as a multiple of Mario's salary this year -/
def bob_salary_multiple : ℝ := 3

theorem bob_salary_last_year :
  let mario_salary_last_year := mario_salary_this_year / (1 + mario_increase_percentage)
  let bob_salary_last_year := bob_salary_multiple * mario_salary_this_year
  bob_salary_last_year = 12000 := by sorry

end NUMINAMATH_CALUDE_bob_salary_last_year_l3139_313987


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l3139_313968

/-- Given a hyperbola and a circle intersecting to form a square, 
    prove the equation of the asymptotes of the hyperbola. -/
theorem hyperbola_asymptotes_equation 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c = Real.sqrt (a^2 + b^2)) 
  (h_hyperbola : ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1 → x^2 + y^2 = c^2 → x^2 = y^2) :
  ∃ k : ℝ, k = Real.sqrt (Real.sqrt 2 - 1) ∧ 
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1 → y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l3139_313968


namespace NUMINAMATH_CALUDE_work_by_concurrent_forces_l3139_313945

/-- Work done by concurrent forces -/
theorem work_by_concurrent_forces :
  let F₁ : ℝ × ℝ := (Real.log 2, Real.log 2)
  let F₂ : ℝ × ℝ := (Real.log 5, Real.log 2)
  let s : ℝ × ℝ := (2 * Real.log 5, 1)
  let F : ℝ × ℝ := (F₁.1 + F₂.1, F₁.2 + F₂.2)
  let W : ℝ := F.1 * s.1 + F.2 * s.2
  W = 2 :=
by sorry

end NUMINAMATH_CALUDE_work_by_concurrent_forces_l3139_313945


namespace NUMINAMATH_CALUDE_marks_profit_l3139_313956

/-- The profit Mark makes from selling a Magic card -/
def profit (initial_cost : ℝ) (value_multiplier : ℝ) : ℝ :=
  initial_cost * value_multiplier - initial_cost

/-- Theorem stating that Mark's profit is $200 -/
theorem marks_profit : profit 100 3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_marks_profit_l3139_313956


namespace NUMINAMATH_CALUDE_buyer_ratio_l3139_313904

/-- Represents the number of buyers on a given day -/
structure BuyerCount where
  count : ℕ

/-- Represents the buyer counts for three consecutive days -/
structure ThreeDayBuyers where
  dayBeforeYesterday : BuyerCount
  yesterday : BuyerCount
  today : BuyerCount

/-- The conditions given in the problem -/
def storeConditions (buyers : ThreeDayBuyers) : Prop :=
  buyers.today.count = buyers.yesterday.count + 40 ∧
  buyers.dayBeforeYesterday.count + buyers.yesterday.count + buyers.today.count = 140 ∧
  buyers.dayBeforeYesterday.count = 50

/-- The theorem to prove -/
theorem buyer_ratio (buyers : ThreeDayBuyers) 
  (h : storeConditions buyers) : 
  buyers.yesterday.count * 2 = buyers.dayBeforeYesterday.count := by
  sorry


end NUMINAMATH_CALUDE_buyer_ratio_l3139_313904


namespace NUMINAMATH_CALUDE_at_least_one_trinomial_has_two_roots_l3139_313950

theorem at_least_one_trinomial_has_two_roots 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : a₁ * a₂ * a₃ = b₁ * b₂ * b₃) 
  (h2 : b₁ * b₂ * b₃ > 1) : 
  ∃ (i : Fin 3), 
    let f := fun x => x^2 + 2 * ([a₁, a₂, a₃].get i) * x + ([b₁, b₂, b₃].get i)
    (∃ (x y : ℝ), x ≠ y ∧ f x = 0 ∧ f y = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_trinomial_has_two_roots_l3139_313950


namespace NUMINAMATH_CALUDE_first_pair_price_is_22_l3139_313996

/-- The price of the first pair of shoes -/
def first_pair_price : ℝ := 22

/-- The price of the second pair of shoes -/
def second_pair_price : ℝ := 1.5 * first_pair_price

/-- The total price of both pairs of shoes -/
def total_price : ℝ := 55

/-- Theorem stating that the price of the first pair of shoes is $22 -/
theorem first_pair_price_is_22 :
  first_pair_price = 22 ∧
  second_pair_price = 1.5 * first_pair_price ∧
  total_price = first_pair_price + second_pair_price :=
by sorry

end NUMINAMATH_CALUDE_first_pair_price_is_22_l3139_313996


namespace NUMINAMATH_CALUDE_slope_movement_l3139_313918

theorem slope_movement (hypotenuse : ℝ) (ratio : ℝ) : 
  hypotenuse = 100 * Real.sqrt 5 →
  ratio = 1 / 2 →
  ∃ (x : ℝ), x^2 + (ratio * x)^2 = hypotenuse^2 ∧ x = 100 :=
by sorry

end NUMINAMATH_CALUDE_slope_movement_l3139_313918


namespace NUMINAMATH_CALUDE_complex_product_modulus_l3139_313927

theorem complex_product_modulus (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = t - 3 * Complex.I →
  t = 6 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_complex_product_modulus_l3139_313927


namespace NUMINAMATH_CALUDE_shooter_probabilities_l3139_313958

/-- A shooter has a probability of hitting the target in a single shot -/
def hit_probability : ℝ := 0.5

/-- The number of shots taken -/
def num_shots : ℕ := 4

/-- The probability of hitting the target exactly k times in n shots -/
def prob_exact_hits (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * hit_probability ^ k * (1 - hit_probability) ^ (n - k)

/-- The probability of hitting the target at least once in n shots -/
def prob_at_least_one_hit (n : ℕ) : ℝ :=
  1 - (1 - hit_probability) ^ n

theorem shooter_probabilities :
  (prob_exact_hits num_shots 3 = 1/4) ∧
  (prob_at_least_one_hit num_shots = 15/16) := by
  sorry

end NUMINAMATH_CALUDE_shooter_probabilities_l3139_313958


namespace NUMINAMATH_CALUDE_max_five_cent_coins_l3139_313966

theorem max_five_cent_coins (x y z : ℕ) : 
  x + y + z = 25 →
  x + 2*y + 5*z = 60 →
  z ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_five_cent_coins_l3139_313966


namespace NUMINAMATH_CALUDE_systematic_sample_correct_l3139_313943

/-- Given a total number of students, sample size, and first drawn number,
    returns the list of remaining numbers in the systematic sampling sequence. -/
def systematicSample (totalStudents : Nat) (sampleSize : Nat) (firstNumber : Nat) : List Nat :=
  let interval := totalStudents / sampleSize
  List.range (sampleSize - 1) |>.map (fun i => (firstNumber + (i + 1) * interval) % totalStudents)

/-- Theorem stating that for the given conditions, the systematic sampling
    produces the expected sequence of numbers. -/
theorem systematic_sample_correct :
  systematicSample 60 5 4 = [16, 28, 40, 52] := by
  sorry

#eval systematicSample 60 5 4

end NUMINAMATH_CALUDE_systematic_sample_correct_l3139_313943


namespace NUMINAMATH_CALUDE_ladder_length_l3139_313980

theorem ladder_length (a b : ℝ) (ha : a = 20) (hb : b = 15) :
  Real.sqrt (a^2 + b^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ladder_length_l3139_313980


namespace NUMINAMATH_CALUDE_arrangements_of_distinct_letters_l3139_313993

-- Define the number of distinct letters
def num_distinct_letters : ℕ := 7

-- Define the function to calculate the number of arrangements
def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Theorem statement
theorem arrangements_of_distinct_letters : 
  num_arrangements num_distinct_letters = 5040 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_of_distinct_letters_l3139_313993


namespace NUMINAMATH_CALUDE_odd_number_probability_l3139_313978

-- Define a fair six-sided die
def FairDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set of odd numbers on the die
def OddNumbers : Finset ℕ := {1, 3, 5}

-- Theorem: The probability of rolling an odd number is 1/2
theorem odd_number_probability :
  (Finset.card OddNumbers : ℚ) / (Finset.card FairDie : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_probability_l3139_313978


namespace NUMINAMATH_CALUDE_all_log_monotonic_exists_divisible_by_2_and_5_exists_log2_positive_all_statements_true_l3139_313944

-- Define logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- 1. All logarithmic functions are monotonic
theorem all_log_monotonic (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  StrictMono (log a) := by sorry

-- 2. There exists an integer divisible by both 2 and 5
theorem exists_divisible_by_2_and_5 :
  ∃ n : ℤ, 2 ∣ n ∧ 5 ∣ n := by sorry

-- 3. There exists a real number x such that log₂x > 0
theorem exists_log2_positive :
  ∃ x : ℝ, log 2 x > 0 := by sorry

-- All statements are true
theorem all_statements_true :
  (∀ a : ℝ, a > 0 → a ≠ 1 → StrictMono (log a)) ∧
  (∃ n : ℤ, 2 ∣ n ∧ 5 ∣ n) ∧
  (∃ x : ℝ, log 2 x > 0) := by sorry

end NUMINAMATH_CALUDE_all_log_monotonic_exists_divisible_by_2_and_5_exists_log2_positive_all_statements_true_l3139_313944


namespace NUMINAMATH_CALUDE_laticia_socks_count_l3139_313961

/-- The number of pairs of socks Laticia knitted in the first week -/
def first_week : ℕ := 12

/-- The number of pairs of socks Laticia knitted in the second week -/
def second_week : ℕ := first_week + 4

/-- The number of pairs of socks Laticia knitted in the third week -/
def third_week : ℕ := (first_week + second_week) / 2

/-- The number of pairs of socks Laticia knitted in the fourth week -/
def fourth_week : ℕ := third_week - 3

/-- The total number of pairs of socks Laticia knitted over four weeks -/
def total_socks : ℕ := first_week + second_week + third_week + fourth_week

theorem laticia_socks_count : total_socks = 53 := by
  sorry

end NUMINAMATH_CALUDE_laticia_socks_count_l3139_313961


namespace NUMINAMATH_CALUDE_percentage_difference_l3139_313994

theorem percentage_difference : 
  (38 / 100 : ℚ) * 80 - (12 / 100 : ℚ) * 160 = 11.2 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l3139_313994


namespace NUMINAMATH_CALUDE_min_value_of_f_l3139_313964

def f (x : ℝ) := 3 * x^2 - 6 * x + 9

theorem min_value_of_f : 
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3139_313964


namespace NUMINAMATH_CALUDE_xy_system_implies_x2_plus_y2_l3139_313969

theorem xy_system_implies_x2_plus_y2 (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 110) : 
  x^2 + y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_xy_system_implies_x2_plus_y2_l3139_313969


namespace NUMINAMATH_CALUDE_steve_reading_time_l3139_313948

/-- Calculates the number of weeks needed to read a book given the total pages and pages read per week. -/
def weeks_to_read (total_pages : ℕ) (pages_per_day : ℕ) (reading_days_per_week : ℕ) : ℕ :=
  total_pages / (pages_per_day * reading_days_per_week)

/-- Proves that it takes 7 weeks to read a 2100-page book when reading 100 pages on 3 days per week. -/
theorem steve_reading_time : weeks_to_read 2100 100 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_steve_reading_time_l3139_313948


namespace NUMINAMATH_CALUDE_population_sum_theorem_l3139_313908

/-- The population of Springfield -/
def springfield_population : ℕ := 482653

/-- The difference in population between Springfield and Greenville -/
def population_difference : ℕ := 119666

/-- The total population of Springfield and Greenville -/
def total_population : ℕ := 845640

/-- Theorem stating that the sum of Springfield's population and a city with 119,666 fewer people equals the total population -/
theorem population_sum_theorem : 
  springfield_population + (springfield_population - population_difference) = total_population := by
  sorry

end NUMINAMATH_CALUDE_population_sum_theorem_l3139_313908


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3139_313973

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites
    and their y-coordinates are equal -/
def symmetric_wrt_y_axis (a b : ℝ × ℝ) : Prop :=
  a.1 = -b.1 ∧ a.2 = b.2

theorem symmetric_points_sum (x y : ℝ) :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (x - 4, 6 + y)
  symmetric_wrt_y_axis a b → x + y = -3 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3139_313973


namespace NUMINAMATH_CALUDE_circle_theorem_l3139_313957

/-- The circle passing through points A(1, 4) and B(3, 2) with its center on the line y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 4.5)^2 + y^2 = 28.25

/-- Point A -/
def point_A : ℝ × ℝ := (1, 4)

/-- Point B -/
def point_B : ℝ × ℝ := (3, 2)

/-- Point P -/
def point_P : ℝ × ℝ := (2, 4)

/-- A point is inside the circle if the left side of the equation is less than the right side -/
def is_inside_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - 4.5)^2 + p.2^2 < 28.25

theorem circle_theorem :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  is_inside_circle point_P :=
by sorry

end NUMINAMATH_CALUDE_circle_theorem_l3139_313957


namespace NUMINAMATH_CALUDE_alternating_draw_probability_l3139_313926

/-- Represents the number of white balls in the box -/
def white_balls : ℕ := 4

/-- Represents the number of black balls in the box -/
def black_balls : ℕ := 4

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- Represents the probability of drawing all balls in alternating colors -/
def alternating_probability : ℚ := 1 / 35

/-- Theorem stating that the probability of drawing all balls in alternating colors is 1/35 -/
theorem alternating_draw_probability :
  alternating_probability = 1 / 35 := by sorry

end NUMINAMATH_CALUDE_alternating_draw_probability_l3139_313926


namespace NUMINAMATH_CALUDE_intersection_point_l3139_313937

-- Define the linear functions
def f1 (a b x : ℝ) : ℝ := a * x + b + 3
def f2 (a b x : ℝ) : ℝ := -b * x + a - 2
def f3 (x : ℝ) : ℝ := 2 * x - 8

-- State the theorem
theorem intersection_point (a b : ℝ) :
  (∃ y, f1 a b 0 = f2 a b 0 ∧ y = f1 a b 0) ∧  -- First and second functions intersect on y-axis
  (∃ x, f2 a b x = f3 x ∧ f2 a b x = 0) →      -- Second and third functions intersect on x-axis
  (∃ x y, f1 a b x = f3 x ∧ y = f1 a b x ∧ x = -3 ∧ y = -14) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3139_313937


namespace NUMINAMATH_CALUDE_multiplication_equations_l3139_313983

theorem multiplication_equations : 
  (30 * 30 = 900) ∧
  (30 * 40 = 1200) ∧
  (40 * 70 = 2800) ∧
  (50 * 70 = 3500) ∧
  (60 * 70 = 4200) ∧
  (4 * 90 = 360) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equations_l3139_313983


namespace NUMINAMATH_CALUDE_f_13_equals_214_l3139_313977

/-- The function f defined as f(n) = n^2 + 2n + 19 -/
def f (n : ℕ) : ℕ := n^2 + 2*n + 19

/-- Theorem stating that f(13) equals 214 -/
theorem f_13_equals_214 : f 13 = 214 := by
  sorry

end NUMINAMATH_CALUDE_f_13_equals_214_l3139_313977


namespace NUMINAMATH_CALUDE_pams_apples_l3139_313962

theorem pams_apples (pam_bags : ℕ) (gerald_apples_per_bag : ℕ) 
  (h1 : pam_bags = 10)
  (h2 : gerald_apples_per_bag = 40) :
  pam_bags * (3 * gerald_apples_per_bag) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_pams_apples_l3139_313962


namespace NUMINAMATH_CALUDE_no_integer_solution_l3139_313931

theorem no_integer_solution : ¬ ∃ (x y : ℤ), 21 * x - 35 * y = 59 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3139_313931


namespace NUMINAMATH_CALUDE_polynomial_existence_l3139_313975

theorem polynomial_existence : ∃ (P : ℤ → ℤ), 
  (∃ (a b c d e f g h i : ℤ), ∀ x, P x = a*x^8 + b*x^7 + c*x^6 + d*x^5 + e*x^4 + f*x^3 + g*x^2 + h*x + i) ∧ 
  (∀ x : ℤ, P x ≠ 0) ∧
  (∀ n : ℕ, n > 0 → ∃ x : ℤ, (n : ℤ) ∣ P x) := by
sorry

end NUMINAMATH_CALUDE_polynomial_existence_l3139_313975


namespace NUMINAMATH_CALUDE_february_discount_correct_l3139_313992

/-- Represents the discount percentage applied in February -/
def discount_percentage : ℝ := 7

/-- Represents the initial markup percentage -/
def initial_markup : ℝ := 20

/-- Represents the New Year markup percentage -/
def new_year_markup : ℝ := 25

/-- Represents the profit percentage in February -/
def february_profit : ℝ := 39.5

/-- Theorem stating that the discount percentage in February is correct given the markups and profit -/
theorem february_discount_correct :
  let cost := 100 -- Assuming a base cost of 100 for simplicity
  let initial_price := cost * (1 + initial_markup / 100)
  let new_year_price := initial_price * (1 + new_year_markup / 100)
  let final_price := new_year_price * (1 - discount_percentage / 100)
  final_price - cost = february_profit * cost / 100 :=
sorry


end NUMINAMATH_CALUDE_february_discount_correct_l3139_313992


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_equals_square_l3139_313915

theorem power_of_two_plus_one_equals_square (m n : ℕ) : 2^n + 1 = m^2 ↔ m = 3 ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_equals_square_l3139_313915


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_products_min_value_is_achievable_l3139_313995

def is_permutation_of_1_to_9 (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℕ) : Prop :=
  ({a₁, a₂, a₃, b₁, b₂, b₃, c₁, c₂, c₃} : Finset ℕ) = Finset.range 9

theorem min_value_of_sum_of_products 
  (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℕ) 
  (h : is_permutation_of_1_to_9 a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃) : 
  a₁ * a₂ * a₃ + b₁ * b₂ * b₃ + c₁ * c₂ * c₃ ≥ 214 :=
sorry

theorem min_value_is_achievable : 
  ∃ a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℕ, 
    is_permutation_of_1_to_9 a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ ∧ 
    a₁ * a₂ * a₃ + b₁ * b₂ * b₃ + c₁ * c₂ * c₃ = 214 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_products_min_value_is_achievable_l3139_313995


namespace NUMINAMATH_CALUDE_arrangement_count_is_twelve_l3139_313953

/-- The number of elements to be arranged -/
def n : ℕ := 4

/-- The condition that A is adjacent to B -/
def adjacent_condition : Prop := true  -- We don't need to define this explicitly in Lean

/-- The number of ways to arrange n elements with the adjacent condition -/
def arrangement_count (n : ℕ) (adjacent_condition : Prop) : ℕ := sorry

/-- Theorem stating that the number of arrangements is 12 -/
theorem arrangement_count_is_twelve :
  arrangement_count n adjacent_condition = 12 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_twelve_l3139_313953


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l3139_313940

-- Define the hyperbola equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m - 1) = 1

-- Define the condition that foci are on x-axis
def foci_on_x_axis (m : ℝ) : Prop :=
  m + 2 > 0 ∧ m - 1 > 0

-- Theorem statement
theorem hyperbola_m_range (m : ℝ) :
  is_hyperbola m ∧ foci_on_x_axis m → m > 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l3139_313940


namespace NUMINAMATH_CALUDE_remaining_value_proof_l3139_313941

theorem remaining_value_proof (x : ℝ) (h : 0.36 * x = 2376) : 4500 - 0.7 * x = -120 := by
  sorry

end NUMINAMATH_CALUDE_remaining_value_proof_l3139_313941


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_bound_l3139_313951

def is_permutation (σ : Fin 100 → ℕ) : Prop :=
  Function.Bijective σ ∧ ∀ i, σ i ∈ Finset.range 101

def consecutive_sum (σ : Fin 100 → ℕ) (start : Fin 91) : ℕ :=
  (Finset.range 10).sum (λ i => σ (start + i))

theorem largest_consecutive_sum_bound :
  (∃ A : ℕ, A = 505 ∧
    (∀ σ : Fin 100 → ℕ, is_permutation σ →
      ∃ start : Fin 91, consecutive_sum σ start ≥ A) ∧
    ∀ B : ℕ, B > A →
      ∃ σ : Fin 100 → ℕ, is_permutation σ ∧
        ∀ start : Fin 91, consecutive_sum σ start < B) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_bound_l3139_313951


namespace NUMINAMATH_CALUDE_intersection_point_l3139_313925

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = 2 * x - 1
def line2 (x y : ℝ) : Prop := y = -3 * x + 4
def line3 (x y m : ℝ) : Prop := y = 4 * x + m

-- Theorem statement
theorem intersection_point (m : ℝ) : 
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line3 x y m) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3139_313925


namespace NUMINAMATH_CALUDE_polynomial_at_negative_two_l3139_313933

def polynomial (x : ℝ) : ℝ := 2 * x^4 - 3 * x^3 + x^2 - 2 * x + 4

theorem polynomial_at_negative_two :
  polynomial (-2) = 68 := by sorry

end NUMINAMATH_CALUDE_polynomial_at_negative_two_l3139_313933


namespace NUMINAMATH_CALUDE_abc_inequality_l3139_313959

/-- Given a + 2b + 3c = 4, prove two statements about a, b, and c -/
theorem abc_inequality (a b c : ℝ) (h : a + 2*b + 3*c = 4) :
  (∀ (ha : a > 0) (hb : b > 0) (hc : c > 0), 1/a + 2/b + 3/c ≥ 9) ∧
  (∃ (m : ℝ), m = 4/3 ∧ ∀ (x y z : ℝ), x + 2*y + 3*z = 4 → |1/2*x + y| + |z| ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3139_313959


namespace NUMINAMATH_CALUDE_intersection_A_B_l3139_313967

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3139_313967


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l3139_313905

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.6 * P) 
  (hP : P ≠ 0) : 
  M / N = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l3139_313905


namespace NUMINAMATH_CALUDE_second_shot_probability_l3139_313921

theorem second_shot_probability 
  (p_first : ℝ) 
  (p_consecutive : ℝ) 
  (h1 : p_first = 0.75) 
  (h2 : p_consecutive = 0.6) : 
  p_consecutive / p_first = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_second_shot_probability_l3139_313921


namespace NUMINAMATH_CALUDE_museum_trip_l3139_313991

theorem museum_trip (first_bus : ℕ) (second_bus : ℕ) (third_bus : ℕ) (fourth_bus : ℕ) :
  first_bus = 12 →
  second_bus = 2 * first_bus →
  third_bus = second_bus - 6 →
  first_bus + second_bus + third_bus + fourth_bus = 75 →
  fourth_bus - first_bus = 9 := by
sorry

end NUMINAMATH_CALUDE_museum_trip_l3139_313991


namespace NUMINAMATH_CALUDE_problem_statement_l3139_313903

theorem problem_statement :
  (∀ x : ℝ, 1 + 2 * x^4 ≥ 2 * x^3 + x^2) ∧
  (∀ x y z : ℝ, x + 2*y + 3*z = 6 →
    x^2 + y^2 + z^2 ≥ 18/7 ∧
    ∃ x y z : ℝ, x + 2*y + 3*z = 6 ∧ x^2 + y^2 + z^2 = 18/7) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3139_313903


namespace NUMINAMATH_CALUDE_complex_equation_l3139_313907

theorem complex_equation (z : ℂ) (h : z = 1 + I) : z^2 + 2/z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_l3139_313907


namespace NUMINAMATH_CALUDE_infinite_square_free_sequences_l3139_313919

def x_seq (a b n : ℕ) : ℕ := a * n + b
def y_seq (c d n : ℕ) : ℕ := c * n + d

def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p) ∣ n → False

theorem infinite_square_free_sequences
  (a b c d : ℕ) 
  (h1 : Nat.gcd a b = 1) 
  (h2 : Nat.gcd c d = 1) :
  ∃ S : Set ℕ, Set.Infinite S ∧ 
    ∀ n ∈ S, is_square_free (x_seq a b n) ∧ is_square_free (y_seq c d n) := by
  sorry

end NUMINAMATH_CALUDE_infinite_square_free_sequences_l3139_313919


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3139_313912

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + k }

theorem parabola_shift_theorem (p : Parabola) :
  let original := { a := -2, b := 0, c := 1 : Parabola }
  let shifted := shift_parabola original 1 2
  shifted = { a := -2, b := 4, c := 3 : Parabola } := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3139_313912


namespace NUMINAMATH_CALUDE_root_expression_equals_five_l3139_313946

theorem root_expression_equals_five (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (h1 : a - 5 * Real.sqrt a + 2 = 0)
  (h2 : b - 5 * Real.sqrt b + 2 = 0) :
  (a * Real.sqrt a + b * Real.sqrt b) / (a - b) *
  (2 / Real.sqrt a - 2 / Real.sqrt b) /
  (Real.sqrt a - (a + b) / Real.sqrt b) +
  5 * (5 * Real.sqrt a - a) / (b + 2) = 5 := by
sorry

end NUMINAMATH_CALUDE_root_expression_equals_five_l3139_313946


namespace NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l3139_313910

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The period of the decimal representation of a rational number -/
def decimal_period (q : ℚ) : ℕ := sorry

/-- The count of a specific digit in one period of the decimal representation -/
def digit_count_in_period (q : ℚ) (d : ℕ) : ℕ := sorry

/-- The probability of randomly selecting a specific digit from the decimal representation -/
def digit_probability (q : ℚ) (d : ℕ) : ℚ :=
  (digit_count_in_period q d : ℚ) / (decimal_period q : ℚ)

theorem probability_of_two_in_three_elevenths :
  digit_probability (3/11) 2 = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l3139_313910


namespace NUMINAMATH_CALUDE_three_Z_five_equals_fourteen_l3139_313997

-- Define the operation Z
def Z (a b : ℤ) : ℤ := b + 11*a - a^2 - a*b

-- Theorem statement
theorem three_Z_five_equals_fourteen : Z 3 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_three_Z_five_equals_fourteen_l3139_313997


namespace NUMINAMATH_CALUDE_odd_side_length_l3139_313924

/-- A triangle with two known sides and an odd third side -/
structure OddTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  h1 : side1 = 2
  h2 : side2 = 5
  h3 : ∃ k : ℕ, side3 = 2 * k + 1

/-- The triangle inequality theorem -/
axiom triangle_inequality (t : OddTriangle) : 
  t.side1 + t.side2 > t.side3 ∧ 
  t.side1 + t.side3 > t.side2 ∧ 
  t.side2 + t.side3 > t.side1

theorem odd_side_length (t : OddTriangle) : t.side3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_odd_side_length_l3139_313924


namespace NUMINAMATH_CALUDE_radii_of_circles_l3139_313920

/-- Two circles lying outside each other -/
structure TwoCircles where
  center_distance : ℝ
  external_tangent : ℝ
  internal_tangent : ℝ

/-- The radii of two circles -/
structure CircleRadii where
  r₁ : ℝ
  r₂ : ℝ

/-- Given the properties of two circles, compute their radii -/
def compute_radii (circles : TwoCircles) : CircleRadii :=
  { r₁ := 38, r₂ := 22 }

/-- Theorem stating that for the given circle properties, the radii are 38 and 22 -/
theorem radii_of_circles (circles : TwoCircles) 
    (h1 : circles.center_distance = 65) 
    (h2 : circles.external_tangent = 63) 
    (h3 : circles.internal_tangent = 25) : 
    compute_radii circles = { r₁ := 38, r₂ := 22 } := by
  sorry

end NUMINAMATH_CALUDE_radii_of_circles_l3139_313920


namespace NUMINAMATH_CALUDE_quadratic_properties_l3139_313985

def f (x : ℝ) := x^2 - 6*x + 8

theorem quadratic_properties :
  (∀ x, f x = (x - 2) * (x - 4)) ∧
  (∀ x, f x ≥ f 3) ∧
  (f 3 = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3139_313985


namespace NUMINAMATH_CALUDE_peanut_price_is_correct_l3139_313952

/-- The price of cashews per pound in dollars -/
def cashew_price : ℚ := 5

/-- The total weight of the mixture in pounds -/
def total_weight : ℚ := 25

/-- The total cost of the mixture in dollars -/
def total_cost : ℚ := 92

/-- The weight of cashews used in the mixture in pounds -/
def cashew_weight : ℚ := 11

/-- The price of peanuts per pound in dollars -/
def peanut_price : ℚ := (total_cost - cashew_price * cashew_weight) / (total_weight - cashew_weight)

theorem peanut_price_is_correct : peanut_price = 264/100 := by
  sorry

end NUMINAMATH_CALUDE_peanut_price_is_correct_l3139_313952


namespace NUMINAMATH_CALUDE_part_one_part_two_l3139_313902

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
def B : Set ℝ := {x | x^2 + x - 2 < 0}

-- Part 1
theorem part_one : A 0 ∩ (Set.univ \ B) = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2
theorem part_two (a : ℝ) : (∀ x ∈ A a, x ∉ B) ↔ (a ≤ -4 ∨ a ≥ 1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3139_313902


namespace NUMINAMATH_CALUDE_sector_angle_and_area_l3139_313935

/-- Given a sector with radius 8 and arc length 12, prove its central angle and area -/
theorem sector_angle_and_area :
  let r : ℝ := 8
  let l : ℝ := 12
  let α : ℝ := l / r
  let S : ℝ := (1 / 2) * l * r
  α = 3 / 2 ∧ S = 48 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_and_area_l3139_313935


namespace NUMINAMATH_CALUDE_intersection_probability_l3139_313998

/-- A regular decagon is a 10-sided polygon with all sides equal and all angles equal. -/
def RegularDecagon : Type := Unit

/-- The number of vertices in a regular decagon. -/
def num_vertices : ℕ := 10

/-- The number of diagonals in a regular decagon. -/
def num_diagonals (d : RegularDecagon) : ℕ := 35

/-- The number of ways to choose 2 diagonals from a regular decagon. -/
def num_diagonal_pairs (d : RegularDecagon) : ℕ := 595

/-- The number of sets of 4 points that determine intersecting diagonals. -/
def num_intersecting_sets (d : RegularDecagon) : ℕ := 210

/-- The probability that two randomly chosen diagonals of a regular decagon intersect inside the decagon. -/
theorem intersection_probability (d : RegularDecagon) : 
  (num_intersecting_sets d : ℚ) / (num_diagonal_pairs d) = 210 / 595 := by sorry

end NUMINAMATH_CALUDE_intersection_probability_l3139_313998


namespace NUMINAMATH_CALUDE_mass_of_compound_l3139_313982

/-- Molar mass of potassium in g/mol -/
def molar_mass_K : ℝ := 39.10

/-- Molar mass of aluminum in g/mol -/
def molar_mass_Al : ℝ := 26.98

/-- Molar mass of sulfur in g/mol -/
def molar_mass_S : ℝ := 32.07

/-- Molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Molar mass of hydrogen in g/mol -/
def molar_mass_H : ℝ := 1.01

/-- Number of moles of the compound -/
def num_moles : ℝ := 15

/-- Molar mass of potassium aluminum sulfate dodecahydrate (KAl(SO4)2·12H2O) in g/mol -/
def molar_mass_compound : ℝ := 
  molar_mass_K + molar_mass_Al + 2 * molar_mass_S + 32 * molar_mass_O + 24 * molar_mass_H

/-- Mass of the compound in grams -/
def mass_compound : ℝ := num_moles * molar_mass_compound

theorem mass_of_compound : mass_compound = 9996.9 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_compound_l3139_313982


namespace NUMINAMATH_CALUDE_f_negative_when_x_greater_than_one_third_l3139_313965

def f (x : ℝ) := -3 * x + 1

theorem f_negative_when_x_greater_than_one_third :
  ∀ x : ℝ, x > 1/3 → f x < 0 := by
sorry

end NUMINAMATH_CALUDE_f_negative_when_x_greater_than_one_third_l3139_313965


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3139_313913

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) :
  (∀ a b, a^3 * (b^(1/2)) = k) →  -- The cube of a and square root of b vary inversely
  (3^3 * (64^(1/2)) = k) →        -- a = 3 when b = 64
  (a * b = 36) →                  -- Given condition ab = 36
  (b = 6) :=                      -- Prove that b = 6
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3139_313913


namespace NUMINAMATH_CALUDE_ball_prices_theorem_l3139_313917

/-- Represents the prices and quantities of soccer balls and volleyballs -/
structure BallPrices where
  soccer_price : ℝ
  volleyball_price : ℝ
  total_balls : ℕ
  max_cost : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (bp : BallPrices) : Prop :=
  bp.soccer_price = bp.volleyball_price + 15 ∧
  480 / bp.soccer_price = 390 / bp.volleyball_price ∧
  bp.total_balls = 100

/-- The theorem to be proven -/
theorem ball_prices_theorem (bp : BallPrices) 
  (h : satisfies_conditions bp) : 
  bp.soccer_price = 80 ∧ 
  bp.volleyball_price = 65 ∧ 
  ∃ (m : ℕ), m ≤ bp.total_balls ∧ 
             m * bp.soccer_price + (bp.total_balls - m) * bp.volleyball_price ≤ bp.max_cost ∧
             ∀ (n : ℕ), n > m → 
               n * bp.soccer_price + (bp.total_balls - n) * bp.volleyball_price > bp.max_cost :=
by
  sorry

end NUMINAMATH_CALUDE_ball_prices_theorem_l3139_313917


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3139_313972

theorem smallest_multiplier_for_perfect_square (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 2 → ¬(∃ k : ℕ, 1152 * m = k * k)) ∧ 
  (∃ k : ℕ, 1152 * 2 = k * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3139_313972


namespace NUMINAMATH_CALUDE_john_used_16_bulbs_l3139_313932

/-- The number of light bulbs John used -/
def bulbs_used : ℕ := sorry

/-- The initial number of light bulbs -/
def initial_bulbs : ℕ := 40

/-- The number of light bulbs John has left after giving some away -/
def remaining_bulbs : ℕ := 12

theorem john_used_16_bulbs : 
  bulbs_used = 16 ∧ 
  (initial_bulbs - bulbs_used) / 2 = remaining_bulbs :=
sorry

end NUMINAMATH_CALUDE_john_used_16_bulbs_l3139_313932


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3139_313990

theorem complex_equation_solution (a : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a + i) * (1 + i) = 2 * i) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3139_313990


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3139_313929

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x - 1) ≥ 2

-- Define the solution set
def solution_set : Set ℝ := { x | 0 ≤ x ∧ x < 1 }

-- Theorem stating that the solution set is correct
theorem inequality_solution_set :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x ∧ x ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3139_313929


namespace NUMINAMATH_CALUDE_carwash_problem_l3139_313947

theorem carwash_problem (car_price truck_price suv_price : ℕ) 
  (total_raised num_suvs num_trucks : ℕ) : 
  car_price = 5 → 
  truck_price = 6 → 
  suv_price = 7 → 
  total_raised = 100 → 
  num_suvs = 5 → 
  num_trucks = 5 → 
  ∃ num_cars : ℕ, 
    num_cars * car_price + num_trucks * truck_price + num_suvs * suv_price = total_raised ∧ 
    num_cars = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_carwash_problem_l3139_313947


namespace NUMINAMATH_CALUDE_initial_player_count_l3139_313936

/-- Represents a server in the Minecraft scenario -/
structure Server :=
  (players : ℕ)

/-- Represents the state of the two servers at a given time -/
structure GameState :=
  (server1 : Server)
  (server2 : Server)

/-- Simulates a single step of the game, where a player may switch servers -/
def step (state : GameState) : GameState :=
  if state.server1.players > state.server2.players
  then { server1 := ⟨state.server1.players - 1⟩, server2 := ⟨state.server2.players + 1⟩ }
  else if state.server2.players > state.server1.players
  then { server1 := ⟨state.server1.players + 1⟩, server2 := ⟨state.server2.players - 1⟩ }
  else state

/-- Simulates the entire game for a given number of steps -/
def simulate (initial : GameState) (steps : ℕ) : GameState :=
  match steps with
  | 0 => initial
  | n + 1 => step (simulate initial n)

/-- The theorem stating the possible initial player counts -/
theorem initial_player_count (initial : GameState) :
  (simulate initial 2023).server1.players + (simulate initial 2023).server2.players = initial.server1.players + initial.server2.players →
  (∀ i : ℕ, i ≤ 2023 → (simulate initial i).server1.players ≠ 0) →
  (∀ i : ℕ, i ≤ 2023 → (simulate initial i).server2.players ≠ 0) →
  initial.server1.players = 1011 ∨ initial.server1.players = 1012 :=
sorry

end NUMINAMATH_CALUDE_initial_player_count_l3139_313936


namespace NUMINAMATH_CALUDE_expression_simplification_l3139_313954

theorem expression_simplification (x y : ℝ) (hx : x = 2) (hy : y = 1/2) :
  (x + y) * (x - y) + (x - y)^2 - (x^2 - 3*x*y) = 5 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3139_313954


namespace NUMINAMATH_CALUDE_sheep_in_pen_l3139_313949

theorem sheep_in_pen (total : ℕ) (rounded_up : ℕ) (wandered_off : ℕ) : 
  wandered_off = 9 →
  wandered_off = total / 10 →
  rounded_up = total * 9 / 10 →
  rounded_up = 81 := by
sorry

end NUMINAMATH_CALUDE_sheep_in_pen_l3139_313949


namespace NUMINAMATH_CALUDE_mardi_gras_necklaces_l3139_313999

/-- Proves that the total number of necklaces caught is 49 given the problem conditions --/
theorem mardi_gras_necklaces : 
  ∀ (boudreaux rhonda latch cecilia : ℕ),
  boudreaux = 12 →
  rhonda = boudreaux / 2 →
  latch = 3 * rhonda - 4 →
  cecilia = latch + 3 →
  ∃ (k : ℕ), boudreaux + rhonda + latch + cecilia = 7 * k →
  boudreaux + rhonda + latch + cecilia = 49 := by
sorry

end NUMINAMATH_CALUDE_mardi_gras_necklaces_l3139_313999


namespace NUMINAMATH_CALUDE_count_triangles_eq_29_l3139_313989

/-- The number of non-similar triangles with angles (in degrees) that are distinct
    positive integers in an arithmetic progression with an even common difference -/
def count_triangles : ℕ :=
  let angle_sum := 180
  let middle_angle := angle_sum / 3
  let max_difference := middle_angle - 1
  (max_difference / 2)

theorem count_triangles_eq_29 : count_triangles = 29 := by
  sorry

end NUMINAMATH_CALUDE_count_triangles_eq_29_l3139_313989


namespace NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l3139_313922

-- Problem 1
def solution_set (x : ℝ) : Prop := x < -7 ∨ x > 5/3

theorem inequality_solution : 
  ∀ x : ℝ, |2*x + 1| - |x - 4| > 2 ↔ solution_set x := by sorry

-- Problem 2
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by sorry

end NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l3139_313922


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l3139_313942

/-- Represents the number of students in each grade --/
structure Students :=
  (ninth : ℕ)
  (seventh : ℕ)
  (fifth : ℕ)

/-- The ratio of 9th-graders to 7th-graders is 7:4 --/
def ratio_ninth_seventh (s : Students) : Prop :=
  7 * s.seventh = 4 * s.ninth

/-- The ratio of 7th-graders to 5th-graders is 6:5 --/
def ratio_seventh_fifth (s : Students) : Prop :=
  6 * s.fifth = 5 * s.seventh

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.ninth + s.seventh + s.fifth

/-- The theorem stating the smallest possible number of students --/
theorem smallest_number_of_students :
  ∃ (s : Students),
    ratio_ninth_seventh s ∧
    ratio_seventh_fifth s ∧
    (∀ (t : Students),
      ratio_ninth_seventh t ∧ ratio_seventh_fifth t →
      total_students s ≤ total_students t) ∧
    total_students s = 43 :=
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l3139_313942


namespace NUMINAMATH_CALUDE_pamphlet_printing_speed_ratio_l3139_313938

theorem pamphlet_printing_speed_ratio : 
  ∀ (mike_speed : ℕ) (mike_hours_before_break : ℕ) (mike_hours_after_break : ℕ) 
    (leo_speed_multiplier : ℕ) (total_pamphlets : ℕ),
  mike_speed = 600 →
  mike_hours_before_break = 9 →
  mike_hours_after_break = 2 →
  total_pamphlets = 9400 →
  (mike_speed * mike_hours_before_break + 
   (mike_speed / 3) * mike_hours_after_break + 
   (leo_speed_multiplier * mike_speed) * (mike_hours_before_break / 3) = total_pamphlets) →
  leo_speed_multiplier = 2 := by
sorry

end NUMINAMATH_CALUDE_pamphlet_printing_speed_ratio_l3139_313938


namespace NUMINAMATH_CALUDE_prop_one_prop_two_prop_three_prop_four_l3139_313906

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define symmetry about x = 1
def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 + x) = f (1 - x)

-- Proposition 1
theorem prop_one (h : ∀ x : ℝ, f (1 + 2*x) = f (1 - 2*x)) : 
  symmetric_about_one f := sorry

-- Proposition 2
theorem prop_two : 
  (∀ x : ℝ, f (x - 1) = f (1 - x)) → symmetric_about_one f := sorry

-- Proposition 3
theorem prop_three (h1 : ∀ x : ℝ, f x = f (-x)) 
  (h2 : ∀ x : ℝ, f (1 + x) = -f x) : symmetric_about_one f := sorry

-- Proposition 4
theorem prop_four (h1 : ∀ x : ℝ, f x = -f (-x)) 
  (h2 : ∀ x : ℝ, f x = f (-x - 2)) : symmetric_about_one f := sorry

end NUMINAMATH_CALUDE_prop_one_prop_two_prop_three_prop_four_l3139_313906


namespace NUMINAMATH_CALUDE_origin_outside_circle_l3139_313970

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 2*a*x + 2*y + (a - 1)^2 = 0}
  (0, 0) ∉ circle := by
sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l3139_313970


namespace NUMINAMATH_CALUDE_expand_expression_l3139_313963

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3139_313963


namespace NUMINAMATH_CALUDE_two_rotations_top_left_to_top_right_l3139_313988

/-- Represents the corners of a rectangle --/
inductive Corner
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

/-- Represents the rotation of a rectangle around a regular pentagon --/
def rotateAroundPentagon (n : ℕ) (startCorner : Corner) : Corner :=
  match n % 4 with
  | 0 => startCorner
  | 1 => match startCorner with
    | Corner.TopLeft => Corner.TopRight
    | Corner.TopRight => Corner.BottomRight
    | Corner.BottomRight => Corner.BottomLeft
    | Corner.BottomLeft => Corner.TopLeft
  | 2 => match startCorner with
    | Corner.TopLeft => Corner.BottomRight
    | Corner.TopRight => Corner.BottomLeft
    | Corner.BottomRight => Corner.TopLeft
    | Corner.BottomLeft => Corner.TopRight
  | 3 => match startCorner with
    | Corner.TopLeft => Corner.BottomLeft
    | Corner.TopRight => Corner.TopLeft
    | Corner.BottomRight => Corner.TopRight
    | Corner.BottomLeft => Corner.BottomRight
  | _ => startCorner  -- This case should never occur due to % 4

/-- Theorem stating that after two full rotations, an object at the top left corner ends up at the top right corner --/
theorem two_rotations_top_left_to_top_right :
  rotateAroundPentagon 2 Corner.TopLeft = Corner.TopRight :=
by sorry


end NUMINAMATH_CALUDE_two_rotations_top_left_to_top_right_l3139_313988


namespace NUMINAMATH_CALUDE_associative_property_only_l3139_313955

theorem associative_property_only (a b c : ℕ) : 
  (a + b) + c = a + (b + c) ↔ 
  ∃ (x y z : ℕ), x + y + z = x + (y + z) ∧ x = 57 ∧ y = 24 ∧ z = 76 :=
by sorry

end NUMINAMATH_CALUDE_associative_property_only_l3139_313955
