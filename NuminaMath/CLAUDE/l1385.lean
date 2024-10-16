import Mathlib

namespace NUMINAMATH_CALUDE_second_person_speed_l1385_138512

/-- Given two people walking in the same direction, this theorem proves
    the speed of the second person given the conditions of the problem. -/
theorem second_person_speed
  (time : ℝ)
  (distance : ℝ)
  (speed1 : ℝ)
  (h1 : time = 9.5)
  (h2 : distance = 9.5)
  (h3 : speed1 = 4.5)
  : ∃ (speed2 : ℝ), speed2 = 5.5 ∧ distance = (speed2 - speed1) * time :=
by
  sorry

#check second_person_speed

end NUMINAMATH_CALUDE_second_person_speed_l1385_138512


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_with_quarter_circles_l1385_138513

/-- The area of the shaded region in a rectangle with quarter circles at corners -/
theorem shaded_area_rectangle_with_quarter_circles 
  (length : ℝ) (width : ℝ) (radius : ℝ) 
  (h_length : length = 12) 
  (h_width : width = 8) 
  (h_radius : radius = 4) : 
  length * width - π * radius^2 = 96 - 16 * π := by
sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_with_quarter_circles_l1385_138513


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_3_7_plus_6_6_l1385_138532

theorem greatest_prime_factor_of_3_7_plus_6_6 :
  ∃ p : ℕ, p.Prime ∧ p ∣ (3^7 + 6^6) ∧ ∀ q : ℕ, q.Prime → q ∣ (3^7 + 6^6) → q ≤ p ∧ p = 67 :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_3_7_plus_6_6_l1385_138532


namespace NUMINAMATH_CALUDE_average_velocity_proof_l1385_138545

/-- The average velocity of a particle with motion equation s(t) = 4 - 2t² 
    over the time interval [1, 1+Δt] is equal to -4 - 2Δt. -/
theorem average_velocity_proof (Δt : ℝ) : 
  let s (t : ℝ) := 4 - 2 * t^2
  let v_avg := (s (1 + Δt) - s 1) / Δt
  v_avg = -4 - 2 * Δt :=
by sorry

end NUMINAMATH_CALUDE_average_velocity_proof_l1385_138545


namespace NUMINAMATH_CALUDE_min_distance_squared_l1385_138549

/-- Given real numbers a, b, c, d satisfying |b+a^2-4ln a|+|2c-d+2|=0,
    the minimum value of (a-c)^2+(b-d)^2 is 5. -/
theorem min_distance_squared (a b c d : ℝ) 
    (h : |b + a^2 - 4*Real.log a| + |2*c - d + 2| = 0) : 
  (∀ x y z w : ℝ, |w + x^2 - 4*Real.log x| + |2*y - z + 2| = 0 →
    (a - c)^2 + (b - d)^2 ≤ (x - y)^2 + (w - z)^2) ∧
  (∃ x y z w : ℝ, |w + x^2 - 4*Real.log x| + |2*y - z + 2| = 0 ∧
    (a - c)^2 + (b - d)^2 = (x - y)^2 + (w - z)^2) ∧
  (a - c)^2 + (b - d)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_squared_l1385_138549


namespace NUMINAMATH_CALUDE_coloring_periodicity_l1385_138555

-- Define a circle with n equal arcs
def Circle (n : ℕ) := Fin n

-- Define a coloring of the circle
def Coloring (n : ℕ) := Circle n → ℕ

-- Define a rotation of the circle
def rotate (n : ℕ) (k : ℕ) (i : Circle n) : Circle n :=
  ⟨(i.val + k) % n, by sorry⟩

-- Define when two arcs are identically colored
def identically_colored (n : ℕ) (c : Coloring n) (i j k l : Circle n) : Prop :=
  ∃ m : ℕ, ∀ t : ℕ, c (rotate n m ⟨(i.val + t) % n, by sorry⟩) = c ⟨(k.val + t) % n, by sorry⟩

-- Define the condition for each division point
def condition_for_each_point (n : ℕ) (c : Coloring n) : Prop :=
  ∀ k : Circle n, ∃ i j : Circle n, 
    i ≠ j ∧ 
    identically_colored n c k i k j ∧
    (∀ t : ℕ, t < i.val - k.val → c ⟨(k.val + t) % n, by sorry⟩ ≠ c ⟨(k.val + t + j.val - i.val) % n, by sorry⟩)

-- Define periodicity of the coloring
def is_periodic (n : ℕ) (c : Coloring n) : Prop :=
  ∃ p : ℕ, p > 0 ∧ p < n ∧ ∀ i : Circle n, c i = c ⟨(i.val + p) % n, by sorry⟩

-- The main theorem
theorem coloring_periodicity (n : ℕ) (c : Coloring n) :
  condition_for_each_point n c → is_periodic n c :=
by sorry

end NUMINAMATH_CALUDE_coloring_periodicity_l1385_138555


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_a_equals_one_l1385_138543

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem purely_imaginary_implies_a_equals_one (a : ℝ) :
  let z : ℂ := Complex.mk (a - 1) 1
  is_purely_imaginary z → a = 1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_a_equals_one_l1385_138543


namespace NUMINAMATH_CALUDE_money_distribution_l1385_138521

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500)
  (h2 : B + C = 340)
  (h3 : C = 40) :
  A + C = 200 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1385_138521


namespace NUMINAMATH_CALUDE_tile_arrangement_theorem_l1385_138523

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Checks if a new configuration is valid given an original configuration -/
def is_valid_new_configuration (original new : TileConfiguration) : Prop :=
  new.tiles = original.tiles + 3 ∧ 
  new.perimeter < original.perimeter + 6

theorem tile_arrangement_theorem : ∃ (original new : TileConfiguration), 
  original.tiles = 10 ∧ 
  original.perimeter = 18 ∧
  new.tiles = 13 ∧
  new.perimeter = 17 ∧
  is_valid_new_configuration original new :=
sorry

end NUMINAMATH_CALUDE_tile_arrangement_theorem_l1385_138523


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1385_138540

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  let r := n % d
  (∀ k : Nat, k < r → ¬(d ∣ (n - k))) ∧ (d ∣ (n - r)) :=
by sorry

theorem problem_solution :
  let initial_number := 427398
  let divisor := 15
  let remainder := initial_number % divisor
  remainder = 3 ∧
  (∀ k : Nat, k < remainder → ¬(divisor ∣ (initial_number - k))) ∧
  (divisor ∣ (initial_number - remainder)) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1385_138540


namespace NUMINAMATH_CALUDE_mr_slinkums_count_correct_l1385_138541

/-- The initial number of Mr. Slinkums in the order -/
def initial_mr_slinkums : ℕ := 147

/-- The percentage of Mr. Slinkums displayed on shelves -/
def displayed_percentage : ℚ := 25 / 100

/-- The number of Mr. Slinkums left in storage -/
def storage_mr_slinkums : ℕ := 110

/-- Theorem stating that the initial number of Mr. Slinkums is correct -/
theorem mr_slinkums_count_correct :
  (1 - displayed_percentage) * initial_mr_slinkums = storage_mr_slinkums := by
  sorry

#check mr_slinkums_count_correct

end NUMINAMATH_CALUDE_mr_slinkums_count_correct_l1385_138541


namespace NUMINAMATH_CALUDE_swimmers_passing_theorem_l1385_138511

/-- Represents the number of times two swimmers pass each other in a pool --/
def swimmers_passing_count (pool_length : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of times swimmers pass each other under given conditions --/
theorem swimmers_passing_theorem :
  let pool_length : ℝ := 100
  let speed1 : ℝ := 4
  let speed2 : ℝ := 3
  let total_time : ℝ := 15 * 60  -- 15 minutes in seconds
  swimmers_passing_count pool_length speed1 speed2 total_time = 27 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_passing_theorem_l1385_138511


namespace NUMINAMATH_CALUDE_mrs_hilt_pizzas_l1385_138564

theorem mrs_hilt_pizzas (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 8) (h2 : total_slices = 16) :
  total_slices / slices_per_pizza = 2 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_pizzas_l1385_138564


namespace NUMINAMATH_CALUDE_edward_garage_sale_games_l1385_138599

/-- The number of games Edward bought at the garage sale -/
def garage_sale_games : ℕ := 14

/-- The number of games Edward bought from a friend -/
def friend_games : ℕ := 41

/-- The number of games that didn't work -/
def bad_games : ℕ := 31

/-- The number of good games Edward ended up with -/
def good_games : ℕ := 24

theorem edward_garage_sale_games :
  garage_sale_games = (good_games + bad_games) - friend_games :=
by sorry

end NUMINAMATH_CALUDE_edward_garage_sale_games_l1385_138599


namespace NUMINAMATH_CALUDE_length_of_AE_l1385_138509

/-- Given four points A, B, C, D on a 2D plane, and E as the intersection of segments AB and CD,
    prove that the length of AE is 5√5/3. -/
theorem length_of_AE (A B C D E : ℝ × ℝ) : 
  A = (0, 3) →
  B = (6, 0) →
  C = (4, 2) →
  D = (2, 0) →
  E.1 = 10/3 →
  E.2 = 4/3 →
  (E.2 - A.2) / (E.1 - A.1) = (B.2 - A.2) / (B.1 - A.1) →
  (E.2 - C.2) / (E.1 - C.1) = (D.2 - C.2) / (D.1 - C.1) →
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 5 * Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AE_l1385_138509


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1385_138516

/-- Given a line segment with one endpoint (-2, 5) and midpoint (1, 0),
    the sum of the coordinates of the other endpoint is -1. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  ((-2 + x) / 2 = 1 ∧ (5 + y) / 2 = 0) → 
  x + y = -1 :=
by sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1385_138516


namespace NUMINAMATH_CALUDE_append_two_digit_numbers_formula_l1385_138547

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≥ 0 ∧ units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- Appends one two-digit number after another -/
def append_two_digit_numbers (n1 n2 : TwoDigitNumber) : Nat :=
  1000 * n1.tens + 100 * n1.units + 10 * n2.tens + n2.units

/-- Theorem: Appending two two-digit numbers results in the expected formula -/
theorem append_two_digit_numbers_formula (n1 n2 : TwoDigitNumber) :
  append_two_digit_numbers n1 n2 = 1000 * n1.tens + 100 * n1.units + 10 * n2.tens + n2.units :=
by sorry

end NUMINAMATH_CALUDE_append_two_digit_numbers_formula_l1385_138547


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_l1385_138548

theorem sqrt_meaningful_iff (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 1) ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_l1385_138548


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l1385_138581

theorem sqrt_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l1385_138581


namespace NUMINAMATH_CALUDE_squared_plus_greater_than_self_l1385_138530

-- Define a monotonically increasing function on R
def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem squared_plus_greater_than_self
  (f : ℝ → ℝ) (h_monotone : monotone_increasing f) (t : ℝ) (h_t : t ≠ 0) :
  f (t^2 + t) > f t :=
sorry

end NUMINAMATH_CALUDE_squared_plus_greater_than_self_l1385_138530


namespace NUMINAMATH_CALUDE_remainder_problem_l1385_138585

theorem remainder_problem (x : ℤ) :
  x % 3 = 2 → x % 4 = 1 → x % 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1385_138585


namespace NUMINAMATH_CALUDE_total_blue_balloons_l1385_138546

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 9

/-- The number of blue balloons Sally has -/
def sally_balloons : ℕ := 5

/-- The number of blue balloons Jessica has -/
def jessica_balloons : ℕ := 2

/-- The total number of blue balloons -/
def total_balloons : ℕ := joan_balloons + sally_balloons + jessica_balloons

theorem total_blue_balloons : total_balloons = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_balloons_l1385_138546


namespace NUMINAMATH_CALUDE_marble_ratio_l1385_138582

/-- Proves that the ratio of marbles in a clay pot to marbles in a jar is 3:1 -/
theorem marble_ratio (jars : ℕ) (clay_pots : ℕ) (marbles_per_jar : ℕ) (total_marbles : ℕ) :
  jars = 16 →
  jars = 2 * clay_pots →
  marbles_per_jar = 5 →
  total_marbles = 200 →
  ∃ (marbles_per_pot : ℕ), 
    marbles_per_pot * clay_pots + marbles_per_jar * jars = total_marbles ∧
    marbles_per_pot / marbles_per_jar = 3 :=
by sorry

end NUMINAMATH_CALUDE_marble_ratio_l1385_138582


namespace NUMINAMATH_CALUDE_inequality_proof_l1385_138506

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + 2*b + 3*c = 9) : 1/a + 1/b + 1/c ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1385_138506


namespace NUMINAMATH_CALUDE_coins_after_fifth_hour_l1385_138589

def coins_in_jar (hour1 : ℕ) (hour2_3 : ℕ) (hour4 : ℕ) (taken_out : ℕ) : ℕ :=
  hour1 + 2 * hour2_3 + hour4 - taken_out

theorem coins_after_fifth_hour :
  coins_in_jar 20 30 40 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_coins_after_fifth_hour_l1385_138589


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1385_138522

theorem point_in_fourth_quadrant (a : ℤ) : 
  (2*a + 6 > 0) ∧ (3*a + 3 < 0) → (2*a + 6 = 2 ∧ 3*a + 3 = -3) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1385_138522


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l1385_138574

/-- Given a square C with perimeter 40 cm and a square D with area equal to one-third the area of square C, 
    the perimeter of square D is (40√3)/3 cm. -/
theorem square_perimeter_relation (C D : Real) : 
  (C = 10) →  -- Side length of square C (derived from perimeter 40)
  (D^2 = (C^2) / 3) →  -- Area of D is one-third of area of C
  (4 * D = (40 * Real.sqrt 3) / 3) :=  -- Perimeter of D
by sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l1385_138574


namespace NUMINAMATH_CALUDE_lamp_post_height_l1385_138529

/-- The height of a lamp post given specific conditions --/
theorem lamp_post_height (cable_ground_distance : ℝ) (person_distance : ℝ) (person_height : ℝ)
  (h1 : cable_ground_distance = 4)
  (h2 : person_distance = 3)
  (h3 : person_height = 1.6)
  (h4 : person_distance < cable_ground_distance) :
  ∃ (post_height : ℝ),
    post_height = (cable_ground_distance * person_height) / (cable_ground_distance - person_distance) ∧
    post_height = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_lamp_post_height_l1385_138529


namespace NUMINAMATH_CALUDE_optimal_landing_point_l1385_138500

/-- The optimal landing point for a messenger traveling from a boat to a camp on shore -/
theorem optimal_landing_point (boat_distance : ℝ) (camp_distance : ℝ) 
  (row_speed : ℝ) (walk_speed : ℝ) : ℝ :=
let landing_point := 12
let total_time (x : ℝ) := 
  (Real.sqrt (boat_distance^2 + x^2)) / row_speed + (camp_distance - x) / walk_speed
have h1 : boat_distance = 9 := by sorry
have h2 : camp_distance = 15 := by sorry
have h3 : row_speed = 4 := by sorry
have h4 : walk_speed = 5 := by sorry
have h5 : ∀ x, total_time landing_point ≤ total_time x := by sorry
landing_point

#check optimal_landing_point

end NUMINAMATH_CALUDE_optimal_landing_point_l1385_138500


namespace NUMINAMATH_CALUDE_units_digit_of_product_l1385_138514

theorem units_digit_of_product (n m : ℕ) : (5^7 * 6^4) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l1385_138514


namespace NUMINAMATH_CALUDE_median_siblings_is_two_l1385_138551

/-- Represents the number of students for each sibling count -/
def sibling_distribution : List (Nat × Nat) :=
  [(0, 2), (1, 3), (2, 2), (3, 1), (4, 2), (5, 1)]

/-- Calculates the total number of students -/
def total_students : Nat :=
  sibling_distribution.foldl (fun acc (_, count) => acc + count) 0

/-- Finds the median position -/
def median_position : Nat :=
  (total_students + 1) / 2

/-- Theorem: The median number of siblings in Mrs. Thompson's History class is 2 -/
theorem median_siblings_is_two :
  let cumulative_count := sibling_distribution.foldl
    (fun acc (siblings, count) => 
      match acc with
      | [] => [(siblings, count)]
      | (_, prev_count) :: _ => (siblings, prev_count + count) :: acc
    ) []
  cumulative_count.reverse.find? (fun (_, count) => count ≥ median_position)
    = some (2, 7) := by sorry

end NUMINAMATH_CALUDE_median_siblings_is_two_l1385_138551


namespace NUMINAMATH_CALUDE_factors_of_prime_factorization_l1385_138560

def prime_factorization := 2^3 * 3^5 * 5^4 * 7^2 * 11^6

def number_of_factors (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (3 + 1) * (5 + 1) * (4 + 1) * (2 + 1) * (6 + 1)

theorem factors_of_prime_factorization :
  number_of_factors prime_factorization = 2520 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_prime_factorization_l1385_138560


namespace NUMINAMATH_CALUDE_not_all_square_roots_irrational_l1385_138534

theorem not_all_square_roots_irrational : ¬ (∀ x : ℝ, ∃ y : ℝ, y ^ 2 = x → ¬ (∃ a b : ℤ, x = a / b ∧ b ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_square_roots_irrational_l1385_138534


namespace NUMINAMATH_CALUDE_davids_age_l1385_138570

/-- Given the age relationships between Anna, Ben, Carla, and David, prove David's age. -/
theorem davids_age 
  (anna ben carla david : ℕ)  -- Define variables for ages
  (h1 : anna = ben - 5)       -- Anna is five years younger than Ben
  (h2 : ben = carla + 2)      -- Ben is two years older than Carla
  (h3 : david = carla + 4)    -- David is four years older than Carla
  (h4 : anna = 12)            -- Anna is 12 years old
  : david = 19 :=             -- Prove David is 19 years old
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_davids_age_l1385_138570


namespace NUMINAMATH_CALUDE_books_read_per_year_l1385_138577

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  48 * c * s

/-- Theorem stating the total number of books read by the student body in one year -/
theorem books_read_per_year (c s : ℕ) :
  let books_per_month := 4
  let months_per_year := 12
  let students_per_class := s
  let num_classes := c
  total_books_read c s = books_per_month * months_per_year * students_per_class * num_classes :=
by sorry

end NUMINAMATH_CALUDE_books_read_per_year_l1385_138577


namespace NUMINAMATH_CALUDE_denver_birdhouse_profit_l1385_138567

/-- Represents the profit calculation for Denver's birdhouse business -/
theorem denver_birdhouse_profit :
  ∀ (wood_pieces : ℕ) (wood_cost : ℚ) (sale_price : ℚ),
    wood_pieces = 7 →
    wood_cost = 3/2 →
    sale_price = 32 →
    (sale_price / 2) - (wood_pieces : ℚ) * wood_cost = 11/2 :=
by
  sorry

end NUMINAMATH_CALUDE_denver_birdhouse_profit_l1385_138567


namespace NUMINAMATH_CALUDE_rope_cutting_l1385_138503

/-- Given two ropes of lengths 18 and 24 meters, this theorem proves that
    the maximum length of equal segments that can be cut from both ropes
    without remainder is 6 meters, and the total number of such segments is 7. -/
theorem rope_cutting (rope1 : ℕ) (rope2 : ℕ) 
  (h1 : rope1 = 18) (h2 : rope2 = 24) : 
  ∃ (segment_length : ℕ) (total_segments : ℕ),
    segment_length = 6 ∧ 
    total_segments = 7 ∧
    rope1 % segment_length = 0 ∧
    rope2 % segment_length = 0 ∧
    rope1 / segment_length + rope2 / segment_length = total_segments ∧
    ∀ (l : ℕ), l > segment_length → (rope1 % l ≠ 0 ∨ rope2 % l ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_rope_cutting_l1385_138503


namespace NUMINAMATH_CALUDE_function_properties_l1385_138587

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- State the theorem
theorem function_properties
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < π)
  (h_sym1 : ∀ x, f ω φ x = f ω φ ((2 * π) / 3 - x))
  (h_sym2 : ∀ x, f ω φ x = -f ω φ (π - x))
  (h_period : ∃ T > π / 2, ∀ x, f ω φ (x + T) = f ω φ x) :
  (∀ x, f ω φ (x + (2 * π) / 3) = f ω φ x) ∧
  (∀ x, f ω φ x = f ω φ (-x)) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1385_138587


namespace NUMINAMATH_CALUDE_bus_problem_l1385_138544

/-- Calculates the number of students remaining on a bus after a given number of stops,
    where half of the students get off at each stop. -/
def studentsRemaining (initial : ℕ) (stops : ℕ) : ℕ :=
  initial / (2 ^ stops)

/-- Theorem: If a bus starts with 48 students and half of the remaining students get off
    at each of three consecutive stops, then 6 students will remain on the bus after the third stop. -/
theorem bus_problem : studentsRemaining 48 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l1385_138544


namespace NUMINAMATH_CALUDE_max_non_managers_l1385_138535

/-- Given a department with 8 managers and a ratio of managers to non-managers greater than 7:24,
    the maximum number of non-managers is 27. -/
theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 → 
  (managers : ℚ) / non_managers > 7 / 24 →
  non_managers ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l1385_138535


namespace NUMINAMATH_CALUDE_diamond_three_eq_six_implies_a_eq_eight_l1385_138575

/-- Define the diamond operation -/
def diamond (a b : ℝ) : ℝ := 3 * a - 2 * b^2

/-- Theorem stating that if a ◇ 3 = 6, then a = 8 -/
theorem diamond_three_eq_six_implies_a_eq_eight :
  ∀ a : ℝ, diamond a 3 = 6 → a = 8 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_eq_six_implies_a_eq_eight_l1385_138575


namespace NUMINAMATH_CALUDE_angle_triple_supplement_l1385_138592

theorem angle_triple_supplement (x : ℝ) : x = 3 * (180 - x) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_l1385_138592


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1385_138588

theorem polynomial_factorization 
  (P Q R : Polynomial ℝ) 
  (h : P^4 + Q^4 = R^2) : 
  ∃ (p q r : ℝ) (S : Polynomial ℝ), 
    P = p • S ∧ Q = q • S ∧ R = r • S^2 :=
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1385_138588


namespace NUMINAMATH_CALUDE_base_conversion_440_to_octal_l1385_138526

theorem base_conversion_440_to_octal :
  (440 : ℕ) = 6 * 8^2 + 7 * 8^1 + 0 * 8^0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_440_to_octal_l1385_138526


namespace NUMINAMATH_CALUDE_M_not_finite_union_of_aps_l1385_138593

-- Define the set M
def M : Set ℕ := {n : ℕ | ∀ x y : ℕ, (1 : ℚ) / x + (1 : ℚ) / y ≠ 3 / n}

-- Define what it means for a set to be representable as a finite union of arithmetic progressions
def is_finite_union_of_aps (S : Set ℕ) : Prop :=
  ∃ (n : ℕ) (a d : Fin n → ℕ), S = ⋃ i, {k : ℕ | ∃ j : ℕ, k = a i + j * d i}

-- State the theorem
theorem M_not_finite_union_of_aps :
  (∀ n : ℕ, n ∉ M → ∀ m : ℕ, m * n ∉ M) →
  (∀ k : ℕ, k > 0 → (7 : ℕ) ^ k ∈ M) →
  ¬ is_finite_union_of_aps M :=
sorry

end NUMINAMATH_CALUDE_M_not_finite_union_of_aps_l1385_138593


namespace NUMINAMATH_CALUDE_club_leader_selection_l1385_138558

/-- Represents a club with members of two genders, some wearing glasses -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat
  boys_with_glasses : Nat
  girls_with_glasses : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def ways_to_choose_leaders (c : Club) : Nat :=
  (c.boys_with_glasses * (c.boys_with_glasses - 1)) +
  (c.girls_with_glasses * (c.girls_with_glasses - 1))

/-- The main theorem to prove -/
theorem club_leader_selection (c : Club) 
  (h1 : c.total_members = 24)
  (h2 : c.boys = 12)
  (h3 : c.girls = 12)
  (h4 : c.boys_with_glasses = 6)
  (h5 : c.girls_with_glasses = 6) :
  ways_to_choose_leaders c = 60 := by
  sorry

#eval ways_to_choose_leaders { total_members := 24, boys := 12, girls := 12, boys_with_glasses := 6, girls_with_glasses := 6 }

end NUMINAMATH_CALUDE_club_leader_selection_l1385_138558


namespace NUMINAMATH_CALUDE_maze_side_length_l1385_138594

/-- Represents a maze on a square grid -/
structure Maze where
  sideLength : ℕ
  wallLength : ℕ

/-- Checks if the maze satisfies the unique path property -/
def hasUniquePaths (m : Maze) : Prop :=
  m.sideLength ^ 2 = 2 * m.sideLength * (m.sideLength - 1) - m.wallLength + 1

theorem maze_side_length (m : Maze) :
  m.wallLength = 400 → hasUniquePaths m → m.sideLength = 21 := by
  sorry

end NUMINAMATH_CALUDE_maze_side_length_l1385_138594


namespace NUMINAMATH_CALUDE_multiplicative_inverse_7_mod_31_l1385_138568

theorem multiplicative_inverse_7_mod_31 : ∃ x : ℕ, x < 31 ∧ (7 * x) % 31 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_7_mod_31_l1385_138568


namespace NUMINAMATH_CALUDE_f_minus_one_lt_f_one_l1385_138537

theorem f_minus_one_lt_f_one
  (f : ℝ → ℝ)
  (h_diff : Differentiable ℝ f)
  (h_eq : ∀ x, f x = x^2 + 2 * x * (deriv f 2)) :
  f (-1) < f 1 := by
sorry

end NUMINAMATH_CALUDE_f_minus_one_lt_f_one_l1385_138537


namespace NUMINAMATH_CALUDE_coin_flip_solution_l1385_138556

def coin_flip_problem (n : ℕ) : Prop :=
  let p_tails : ℚ := 1/2
  let p_sequence : ℚ := 0.0625
  (p_tails ^ 2 * (1 - p_tails) ^ 2 = p_sequence) ∧ (n = 4)

theorem coin_flip_solution :
  ∃ n : ℕ, coin_flip_problem n :=
sorry

end NUMINAMATH_CALUDE_coin_flip_solution_l1385_138556


namespace NUMINAMATH_CALUDE_geometric_sequence_special_case_l1385_138559

/-- A geometric sequence with first term 1 and nth term equal to the product of the first 5 terms has n = 11 -/
theorem geometric_sequence_special_case (a : ℕ → ℝ) (n : ℕ) : 
  (∀ k, a (k + 1) / a k = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                            -- first term is 1
  a n = a 1 * a 2 * a 3 * a 4 * a 5 →   -- nth term equals product of first 5 terms
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_case_l1385_138559


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1385_138563

-- Define the variables and conditions
theorem sqrt_inequality (C : ℝ) (hC : C > 1) :
  Real.sqrt (C + 1) - Real.sqrt C < Real.sqrt C - Real.sqrt (C - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1385_138563


namespace NUMINAMATH_CALUDE_no_divisible_lilac_flowers_l1385_138578

theorem no_divisible_lilac_flowers : ¬∃ (q c : ℕ), 
  (∃ (p₁ p₂ : ℕ), q + c = p₂^2 ∧ 4*q + 5*c = p₁^2) ∧ 
  (∃ (x : ℕ), q = c * x) := by
sorry

end NUMINAMATH_CALUDE_no_divisible_lilac_flowers_l1385_138578


namespace NUMINAMATH_CALUDE_sabrina_can_finish_series_l1385_138565

theorem sabrina_can_finish_series 
  (total_books : Nat) 
  (pages_per_book : Nat) 
  (books_read_first_month : Nat) 
  (reading_speed : Nat) 
  (total_days : Nat) 
  (h1 : total_books = 14)
  (h2 : pages_per_book = 200)
  (h3 : books_read_first_month = 4)
  (h4 : reading_speed = 40)
  (h5 : total_days = 60) :
  ∃ (pages_read : Nat), pages_read ≥ total_books * pages_per_book := by
  sorry

#check sabrina_can_finish_series

end NUMINAMATH_CALUDE_sabrina_can_finish_series_l1385_138565


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_120_4620_l1385_138539

theorem gcd_lcm_sum_120_4620 : Nat.gcd 120 4620 + Nat.lcm 120 4620 = 4680 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_120_4620_l1385_138539


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1385_138504

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the property of being a pure imaginary number
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem complex_magnitude_problem (z : ℂ) (a : ℝ) 
  (h1 : is_pure_imaginary z) 
  (h2 : (2 + i) * z = 1 + a * i^3) : 
  Complex.abs (a + z) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1385_138504


namespace NUMINAMATH_CALUDE_expression_simplification_l1385_138505

theorem expression_simplification (m n : ℝ) 
  (h : Real.sqrt (m - 1/2) + (n + 2)^2 = 0) : 
  ((3*m + n) * (m + n) - (2*m - n)^2 + (m + 2*n) * (m - 2*n)) / (2*n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1385_138505


namespace NUMINAMATH_CALUDE_irrational_element_existence_l1385_138501

open Set Real

theorem irrational_element_existence
  (a b : ℚ)
  (M : Set ℝ)
  (hab : 0 < a ∧ a < b)
  (hM : ∀ (x y : ℝ), x ∈ M → y ∈ M → Real.sqrt (x * y) ∈ M)
  (haM : (a : ℝ) ∈ M)
  (hbM : (b : ℝ) ∈ M) :
  ∀ (c d : ℝ), (a : ℝ) < c → c < d → d < (b : ℝ) →
  ∃ (m : ℝ), m ∈ M ∧ Irrational m ∧ c < m ∧ m < d :=
sorry

end NUMINAMATH_CALUDE_irrational_element_existence_l1385_138501


namespace NUMINAMATH_CALUDE_lemonade_consumption_l1385_138554

/-- Represents the lemonade consumption problem -/
theorem lemonade_consumption (x : ℝ) 
  (h1 : x > 0)  -- Ed's initial lemonade amount is positive
  (h2 : x / 2 + x / 4 + 3 = 2 * x - (x / 4 + 3)) -- Equation representing equal consumption
  : x + 2 * x = 18 := by
  sorry

#check lemonade_consumption

end NUMINAMATH_CALUDE_lemonade_consumption_l1385_138554


namespace NUMINAMATH_CALUDE_lucien_ball_count_l1385_138580

/-- Proves that Lucien has 200 balls given the conditions of the problem -/
theorem lucien_ball_count :
  ∀ (lucca_balls lucca_basketballs lucien_basketballs : ℕ) 
    (lucien_balls : ℕ),
  lucca_balls = 100 →
  lucca_basketballs = lucca_balls / 10 →
  lucien_basketballs = lucien_balls / 5 →
  lucca_basketballs + lucien_basketballs = 50 →
  lucien_balls = 200 := by
sorry

end NUMINAMATH_CALUDE_lucien_ball_count_l1385_138580


namespace NUMINAMATH_CALUDE_ratio_sum_squares_to_sum_l1385_138579

theorem ratio_sum_squares_to_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : b = 2 * a) (h5 : c = 4 * a) (h6 : a^2 + b^2 + c^2 = 1701) : 
  a + b + c = 63 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_to_sum_l1385_138579


namespace NUMINAMATH_CALUDE_certain_number_proof_l1385_138597

theorem certain_number_proof (N : ℝ) : 
  (2 / 5 : ℝ) * N - (3 / 5 : ℝ) * 125 = 45 → N = 300 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1385_138597


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1385_138584

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) :
  k ≠ 0 →
  p ≠ 1 →
  r ≠ 1 →
  p ≠ r →
  a₂ = k * p →
  a₃ = k * p^2 →
  b₂ = k * r →
  b₃ = k * r^2 →
  a₃ - b₃ = 4 * (a₂ - b₂) →
  p + r = 4 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1385_138584


namespace NUMINAMATH_CALUDE_nine_balls_distribution_l1385_138527

/-- The number of ways to distribute n identical objects into 3 distinct boxes,
    where box i must contain at least i objects (for i = 1, 2, 3) -/
def distribute_balls (n : ℕ) : ℕ := Nat.choose (n - 1 - 2 - 3 + 3 - 1) 3

/-- Theorem stating that there are 10 ways to distribute 9 balls into 3 boxes
    with the given constraints -/
theorem nine_balls_distribution : distribute_balls 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_nine_balls_distribution_l1385_138527


namespace NUMINAMATH_CALUDE_kindergarten_tissues_l1385_138595

/-- The number of tissues in each mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The number of students in each kindergartner group -/
def group_sizes : List ℕ := [9, 10, 11]

/-- The total number of tissues brought by all kindergartner groups -/
def total_tissues : ℕ := (group_sizes.sum) * tissues_per_box

theorem kindergarten_tissues :
  total_tissues = 1200 :=
by sorry

end NUMINAMATH_CALUDE_kindergarten_tissues_l1385_138595


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l1385_138557

theorem degree_to_radian_conversion (π : ℝ) :
  (1 : ℝ) * π / 180 = π / 180 →
  (-150 : ℝ) * π / 180 = -5 * π / 6 :=
by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l1385_138557


namespace NUMINAMATH_CALUDE_james_height_l1385_138573

theorem james_height (tree_height : ℝ) (tree_shadow : ℝ) (james_shadow : ℝ) :
  tree_height = 60 →
  tree_shadow = 20 →
  james_shadow = 25 →
  (tree_height / tree_shadow) * james_shadow = 75 := by
  sorry

end NUMINAMATH_CALUDE_james_height_l1385_138573


namespace NUMINAMATH_CALUDE_carolyn_sum_is_18_l1385_138586

/-- Represents the game state -/
structure GameState where
  remaining : List Nat
  carolyn_sum : Nat

/-- Represents a player's move -/
inductive Move
  | Remove (n : Nat)

/-- Applies Carolyn's move to the game state -/
def apply_carolyn_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Remove n =>
    { remaining := state.remaining.filter (· ≠ n),
      carolyn_sum := state.carolyn_sum + n }

/-- Applies Paul's move to the game state -/
def apply_paul_move (state : GameState) (move : List Move) : GameState :=
  match move with
  | [] => state
  | (Move.Remove n) :: rest =>
    apply_paul_move
      { remaining := state.remaining.filter (· ≠ n),
        carolyn_sum := state.carolyn_sum }
      rest

/-- Checks if a number has a divisor in the list -/
def has_divisor_in_list (n : Nat) (list : List Nat) : Bool :=
  list.any (fun m => m ≠ n && n % m == 0)

/-- Simulates the game -/
def play_game (initial_state : GameState) : Nat :=
  let state1 := apply_carolyn_move initial_state (Move.Remove 4)
  let state2 := apply_paul_move state1 [Move.Remove 1, Move.Remove 2]
  let state3 := apply_carolyn_move state2 (Move.Remove 6)
  let state4 := apply_paul_move state3 [Move.Remove 3]
  let state5 := apply_carolyn_move state4 (Move.Remove 8)
  let final_state := apply_paul_move state5 [Move.Remove 5, Move.Remove 7]
  final_state.carolyn_sum

theorem carolyn_sum_is_18 :
  let initial_state : GameState := { remaining := [1, 2, 3, 4, 5, 6, 7, 8], carolyn_sum := 0 }
  play_game initial_state = 18 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_sum_is_18_l1385_138586


namespace NUMINAMATH_CALUDE_jacks_kids_l1385_138538

theorem jacks_kids (shirts_per_kid : ℕ) (buttons_per_shirt : ℕ) (total_buttons : ℕ) : 
  shirts_per_kid = 3 → buttons_per_shirt = 7 → total_buttons = 63 →
  ∃ (num_kids : ℕ), num_kids * shirts_per_kid * buttons_per_shirt = total_buttons ∧ num_kids = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jacks_kids_l1385_138538


namespace NUMINAMATH_CALUDE_cake_remainder_cake_problem_l1385_138502

theorem cake_remainder (john_ate : ℚ) (emily_took_half : Bool) : ℚ :=
  by
    -- Define John's portion
    have john_portion : ℚ := 3/5
    
    -- Define the remaining portion after John ate
    have remaining_after_john : ℚ := 1 - john_portion
    
    -- Define Emily's portion
    have emily_portion : ℚ := remaining_after_john / 2
    
    -- Calculate the final remaining portion
    have final_remaining : ℚ := remaining_after_john - emily_portion
    
    -- Prove that the final remaining portion is 1/5 (20%)
    sorry

-- State the theorem
theorem cake_problem : cake_remainder (3/5) true = 1/5 :=
  by sorry

end NUMINAMATH_CALUDE_cake_remainder_cake_problem_l1385_138502


namespace NUMINAMATH_CALUDE_cubic_common_root_identity_l1385_138528

theorem cubic_common_root_identity (p p' q q' : ℝ) (x : ℝ) :
  (x^3 + p*x + q = 0) ∧ (x^3 + p'*x + q' = 0) →
  (p*q' - q*p') * (p - p')^2 = (q - q')^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_common_root_identity_l1385_138528


namespace NUMINAMATH_CALUDE_sin_theta_in_terms_of_x_l1385_138552

theorem sin_theta_in_terms_of_x (θ : Real) (x : Real) (h_acute : 0 < θ ∧ θ < π / 2) 
  (h_cos : Real.cos (θ / 2) = Real.sqrt (x / (2 * x + 1))) :
  Real.sin θ = (2 * Real.sqrt (x * (x + 1))) / (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_in_terms_of_x_l1385_138552


namespace NUMINAMATH_CALUDE_n_equals_six_l1385_138550

/-- The number of coins flipped simultaneously -/
def n : ℕ := sorry

/-- The probability of exactly two tails when flipping n coins -/
def prob_two_tails (n : ℕ) : ℚ := n * (n - 1) / (2^(n + 1))

/-- Theorem stating that n equals 6 when the probability of two tails is 5/32 -/
theorem n_equals_six : 
  (prob_two_tails n = 5/32) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_n_equals_six_l1385_138550


namespace NUMINAMATH_CALUDE_unique_odd_pair_divisibility_l1385_138518

theorem unique_odd_pair_divisibility : 
  ∀ (a b : ℤ), 
    Odd a → Odd b →
    (∃ (c : ℕ), ∀ (n : ℕ), ∃ (k : ℤ), (c^n + 1 : ℤ) = k * (2^n * a + b)) →
    a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_odd_pair_divisibility_l1385_138518


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1385_138520

/-- The eccentricity of the hyperbola (x²/4) - (y²/2) = 1 is √6/2 -/
theorem hyperbola_eccentricity : 
  let C : Set (ℝ × ℝ) := {(x, y) | x^2/4 - y^2/2 = 1}
  ∃ e : ℝ, e = Real.sqrt 6 / 2 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ C → 
      e = Real.sqrt ((x^2/4 + y^2/2) / (x^2/4)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1385_138520


namespace NUMINAMATH_CALUDE_quadratic_equation_at_most_one_solution_l1385_138553

theorem quadratic_equation_at_most_one_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a ≥ 9/8 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_at_most_one_solution_l1385_138553


namespace NUMINAMATH_CALUDE_cirrus_count_l1385_138572

/-- The number of cumulonimbus clouds -/
def cumulonimbus : ℕ := 3

/-- The number of cumulus clouds -/
def cumulus : ℕ := 12 * cumulonimbus

/-- The number of cirrus clouds -/
def cirrus : ℕ := 4 * cumulus

/-- The number of altostratus clouds -/
def altostratus : ℕ := 6 * (cirrus + cumulus)

/-- Theorem stating that the number of cirrus clouds is 144 -/
theorem cirrus_count : cirrus = 144 := by sorry

end NUMINAMATH_CALUDE_cirrus_count_l1385_138572


namespace NUMINAMATH_CALUDE_inequality_theorem_l1385_138517

theorem inequality_theorem (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) :
  (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b ↔
    ((a ≠ b ∧ (b < -1 ∨ b > 0)) ∨ (a = b ∧ a ≠ -1 ∧ a ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1385_138517


namespace NUMINAMATH_CALUDE_halfway_point_l1385_138569

theorem halfway_point (a b : ℚ) (ha : a = 1/12) (hb : b = 1/10) :
  (a + b) / 2 = 11/120 := by
sorry

end NUMINAMATH_CALUDE_halfway_point_l1385_138569


namespace NUMINAMATH_CALUDE_rohan_salary_l1385_138590

/-- Rohan's monthly expenses and savings -/
structure RohanFinances where
  salary : ℝ
  food_percentage : ℝ
  rent_percentage : ℝ
  entertainment_percentage : ℝ
  conveyance_percentage : ℝ
  savings : ℝ

/-- Theorem stating Rohan's monthly salary given his expenses and savings -/
theorem rohan_salary (r : RohanFinances) 
  (h1 : r.food_percentage = 0.4)
  (h2 : r.rent_percentage = 0.2)
  (h3 : r.entertainment_percentage = 0.1)
  (h4 : r.conveyance_percentage = 0.1)
  (h5 : r.savings = 2000)
  (h6 : r.savings = r.salary * (1 - (r.food_percentage + r.rent_percentage + r.entertainment_percentage + r.conveyance_percentage))) :
  r.salary = 10000 := by
  sorry

#check rohan_salary

end NUMINAMATH_CALUDE_rohan_salary_l1385_138590


namespace NUMINAMATH_CALUDE_fraction_value_implies_x_l1385_138542

theorem fraction_value_implies_x (x : ℝ) : 2 / (x - 3) = 2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_implies_x_l1385_138542


namespace NUMINAMATH_CALUDE_largest_valid_code_l1385_138598

def is_power_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5^k

def is_power_of_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def digits_to_nat (a b c d e : ℕ) : ℕ := a * 10000 + b * 1000 + c * 100 + d * 10 + e

def is_valid_code (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ,
    n = digits_to_nat a b c d e ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    is_power_of_5 (a * 10 + b) ∧
    is_power_of_2 (d * 10 + e) ∧
    ∃ k : ℕ, c = 3 * k ∧
    (a + b + c + d + e) % 2 = 1

theorem largest_valid_code :
  ∀ n : ℕ, is_valid_code n → n ≤ 25916 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_code_l1385_138598


namespace NUMINAMATH_CALUDE_lesser_fraction_l1385_138515

theorem lesser_fraction (x y : ℚ) 
  (sum_eq : x + y = 13/14)
  (prod_eq : x * y = 1/8) :
  min x y = (13 - Real.sqrt 57) / 28 := by
sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1385_138515


namespace NUMINAMATH_CALUDE_geralds_toy_cars_l1385_138507

theorem geralds_toy_cars (initial_cars : ℕ) : 
  (initial_cars : ℚ) * (3/4 : ℚ) = 15 → initial_cars = 20 := by
  sorry

end NUMINAMATH_CALUDE_geralds_toy_cars_l1385_138507


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1385_138531

/-- A line is tangent to a parabola if and only if there exists a point (x₀, y₀) that satisfies
    the following conditions:
    1. The point lies on the line: x₀ - y₀ - 1 = 0
    2. The point lies on the parabola: y₀ = a * x₀^2
    3. The slope of the tangent line equals the derivative of the parabola at that point: 1 = 2 * a * x₀
-/
theorem line_tangent_to_parabola (a : ℝ) :
  (∃ x₀ y₀ : ℝ, x₀ - y₀ - 1 = 0 ∧ y₀ = a * x₀^2 ∧ 1 = 2 * a * x₀) ↔ a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1385_138531


namespace NUMINAMATH_CALUDE_seven_grades_needed_l1385_138576

/-- Represents the initial grades and calculates the number of additional
    grades of 5 needed to achieve an average of 4 -/
def min_grades_needed (initial_threes : Nat) (initial_twos : Nat) : Nat :=
  let initial_sum := 3 * initial_threes + 2 * initial_twos
  let initial_count := initial_threes + initial_twos
  let x := (4 * (initial_count + 1) - initial_sum) / (5 - 4)
  x

/-- Theorem stating that given the initial grades, 7 additional grades of 5
    are needed to achieve an average of exactly 4 -/
theorem seven_grades_needed :
  min_grades_needed 3 2 = 7 := by
  sorry

#eval min_grades_needed 3 2

end NUMINAMATH_CALUDE_seven_grades_needed_l1385_138576


namespace NUMINAMATH_CALUDE_merchant_tea_cups_l1385_138591

theorem merchant_tea_cups (S O P : ℕ) 
  (h1 : S + O = 11) 
  (h2 : P + O = 15) 
  (h3 : S + P = 14) : 
  S + O + P = 20 := by
sorry

end NUMINAMATH_CALUDE_merchant_tea_cups_l1385_138591


namespace NUMINAMATH_CALUDE_fruit_difference_l1385_138571

theorem fruit_difference (apples : ℕ) (peach_multiplier : ℕ) : 
  apples = 60 → peach_multiplier = 3 → 
  (peach_multiplier * apples) - apples = 120 := by
  sorry

end NUMINAMATH_CALUDE_fruit_difference_l1385_138571


namespace NUMINAMATH_CALUDE_min_squares_13x13_l1385_138533

/-- Represents a square on a grid -/
structure GridSquare where
  size : Nat
  deriving Repr

/-- The original square size -/
def originalSize : Nat := 13

/-- A list of squares that the original square is divided into -/
def divisionList : List GridSquare := [
  {size := 6},
  {size := 5},
  {size := 4},
  {size := 3},
  {size := 2},
  {size := 2},
  {size := 1},
  {size := 1},
  {size := 1},
  {size := 1},
  {size := 1}
]

/-- The number of squares in the division -/
def numSquares : Nat := divisionList.length

/-- Checks if the division is valid (covers the entire original square) -/
def isValidDivision (list : List GridSquare) : Prop :=
  list.foldl (fun acc square => acc + square.size * square.size) 0 = originalSize * originalSize

/-- Theorem: The minimum number of squares a 13x13 square can be divided into is 11 -/
theorem min_squares_13x13 :
  (isValidDivision divisionList) ∧
  (∀ (otherList : List GridSquare), isValidDivision otherList → otherList.length ≥ numSquares) :=
sorry

end NUMINAMATH_CALUDE_min_squares_13x13_l1385_138533


namespace NUMINAMATH_CALUDE_largest_initial_number_l1385_138566

theorem largest_initial_number :
  ∃ (a b c d e : ℕ),
    189 + a + b + c + d + e = 200 ∧
    189 % a ≠ 0 ∧ 189 % b ≠ 0 ∧ 189 % c ≠ 0 ∧ 189 % d ≠ 0 ∧ 189 % e ≠ 0 ∧
    ∀ (n : ℕ), n > 189 →
      ¬∃ (x y z w v : ℕ),
        n + x + y + z + w + v = 200 ∧
        n % x ≠ 0 ∧ n % y ≠ 0 ∧ n % z ≠ 0 ∧ n % w ≠ 0 ∧ n % v ≠ 0 :=
by sorry


end NUMINAMATH_CALUDE_largest_initial_number_l1385_138566


namespace NUMINAMATH_CALUDE_peanut_plantation_revenue_l1385_138596

-- Define the plantation and region sizes
def plantation_size : ℕ × ℕ := (500, 500)
def region_a_size : ℕ × ℕ := (200, 300)
def region_b_size : ℕ × ℕ := (200, 200)
def region_c_size : ℕ × ℕ := (100, 500)

-- Define production rates (grams per square foot)
def region_a_rate : ℕ := 60
def region_b_rate : ℕ := 45
def region_c_rate : ℕ := 30

-- Define peanut butter production rate
def peanut_to_butter_ratio : ℚ := 5 / 20

-- Define monthly selling prices (dollars per kg)
def monthly_prices : List ℚ := [12, 10, 14, 8, 11]

-- Function to calculate area
def area (size : ℕ × ℕ) : ℕ := size.1 * size.2

-- Function to calculate peanut production for a region
def region_production (size : ℕ × ℕ) (rate : ℕ) : ℕ := area size * rate

-- Calculate total peanut production
def total_peanut_production : ℕ :=
  region_production region_a_size region_a_rate +
  region_production region_b_size region_b_rate +
  region_production region_c_size region_c_rate

-- Calculate peanut butter production in kg
def peanut_butter_production : ℚ :=
  (total_peanut_production : ℚ) * peanut_to_butter_ratio / 1000

-- Calculate total revenue
def total_revenue : ℚ :=
  monthly_prices.foldl (fun acc price => acc + price * peanut_butter_production) 0

-- Theorem statement
theorem peanut_plantation_revenue :
  total_revenue = 94875 := by sorry

end NUMINAMATH_CALUDE_peanut_plantation_revenue_l1385_138596


namespace NUMINAMATH_CALUDE_sqrt_of_squared_negative_l1385_138510

theorem sqrt_of_squared_negative : Real.sqrt ((-5)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_squared_negative_l1385_138510


namespace NUMINAMATH_CALUDE_faye_crayons_count_l1385_138525

/-- The number of rows of crayons --/
def num_rows : ℕ := 15

/-- The number of crayons in each row --/
def crayons_per_row : ℕ := 42

/-- The total number of crayons --/
def total_crayons : ℕ := num_rows * crayons_per_row

theorem faye_crayons_count : total_crayons = 630 := by
  sorry

end NUMINAMATH_CALUDE_faye_crayons_count_l1385_138525


namespace NUMINAMATH_CALUDE_prob_8_or_9_is_half_l1385_138508

/-- The probability of hitting the 10 ring in one shot -/
def prob_10 : ℝ := 0.3

/-- The probability of hitting the 9 ring in one shot -/
def prob_9 : ℝ := 0.3

/-- The probability of hitting the 8 ring in one shot -/
def prob_8 : ℝ := 0.2

/-- The probability of hitting the 8 or 9 rings in one shot -/
def prob_8_or_9 : ℝ := prob_9 + prob_8

theorem prob_8_or_9_is_half : prob_8_or_9 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_prob_8_or_9_is_half_l1385_138508


namespace NUMINAMATH_CALUDE_cafeteria_pies_correct_l1385_138524

def cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies_correct :
  cafeteria_pies 50 5 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_correct_l1385_138524


namespace NUMINAMATH_CALUDE_total_cost_of_balls_l1385_138583

theorem total_cost_of_balls (basketball_price : ℕ) (volleyball_price : ℕ) 
  (basketball_quantity : ℕ) (volleyball_quantity : ℕ) :
  basketball_price = 48 →
  basketball_price = volleyball_price + 18 →
  basketball_quantity = 3 →
  volleyball_quantity = 5 →
  basketball_price * basketball_quantity + volleyball_price * volleyball_quantity = 294 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_balls_l1385_138583


namespace NUMINAMATH_CALUDE_inverse_function_inequality_solution_set_l1385_138561

/-- Given two functions f and g that intersect at two points, 
    prove the solution set of the inequality between their inverse functions. -/
theorem inverse_function_inequality_solution_set 
  (f g : ℝ → ℝ)
  (h_f : ∃ k b : ℝ, ∀ x, f x = k * x + b)
  (h_g : ∀ x, g x = 2^x + 1)
  (h_intersect : ∃ x₁ x₂ : ℝ, 
    f x₁ = g x₁ ∧ f x₁ = 2 ∧ 
    f x₂ = g x₂ ∧ f x₂ = 4 ∧
    x₁ < x₂)
  (f_inv g_inv : ℝ → ℝ)
  (h_f_inv : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x)
  (h_g_inv : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x)
  : {x : ℝ | f_inv x ≥ g_inv x} = Set.Ici 4 ∪ Set.Ioc 1 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_inequality_solution_set_l1385_138561


namespace NUMINAMATH_CALUDE_square_diagonal_ratio_l1385_138519

theorem square_diagonal_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (area_ratio : a^2 / b^2 = 49 / 64) : 
  (a * Real.sqrt 2) / (b * Real.sqrt 2) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_ratio_l1385_138519


namespace NUMINAMATH_CALUDE_largest_a_value_l1385_138536

theorem largest_a_value : ∃ (a_max : ℚ), 
  (∀ a : ℚ, (3 * a + 4) * (a - 2) = 7 * a → a ≤ a_max) ∧ 
  ((3 * a_max + 4) * (a_max - 2) = 7 * a_max) ∧
  a_max = 4 := by
sorry

end NUMINAMATH_CALUDE_largest_a_value_l1385_138536


namespace NUMINAMATH_CALUDE_guest_payment_divisibility_l1385_138562

theorem guest_payment_divisibility (A : Nat) (h1 : A < 10) : 
  (100 + 10 * A + 2) % 11 = 0 ↔ A = 3 := by
  sorry

end NUMINAMATH_CALUDE_guest_payment_divisibility_l1385_138562
