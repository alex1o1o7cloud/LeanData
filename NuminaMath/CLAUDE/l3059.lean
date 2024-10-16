import Mathlib

namespace NUMINAMATH_CALUDE_first_day_of_month_l3059_305954

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the day after n days
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (dayAfter d n)

-- Theorem statement
theorem first_day_of_month (d : DayOfWeek) :
  dayAfter d 22 = DayOfWeek.Wednesday → d = DayOfWeek.Tuesday :=
by sorry

end NUMINAMATH_CALUDE_first_day_of_month_l3059_305954


namespace NUMINAMATH_CALUDE_four_double_prime_value_l3059_305994

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- State the theorem
theorem four_double_prime_value : prime (prime 4) = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_double_prime_value_l3059_305994


namespace NUMINAMATH_CALUDE_deck_size_proof_l3059_305984

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b : ℚ) = 1/3 → 
  (r : ℚ) / (r + b + 4 : ℚ) = 1/4 → 
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_proof_l3059_305984


namespace NUMINAMATH_CALUDE_star_arrangements_l3059_305974

/-- The number of points on a regular ten-pointed star -/
def num_points : ℕ := 20

/-- The number of rotational symmetries of a regular ten-pointed star -/
def num_rotations : ℕ := 10

/-- The number of reflectional symmetries of a regular ten-pointed star -/
def num_reflections : ℕ := 2

/-- The total number of symmetries of a regular ten-pointed star -/
def total_symmetries : ℕ := num_rotations * num_reflections

/-- The number of distinct arrangements of objects on a regular ten-pointed star -/
def distinct_arrangements : ℕ := Nat.factorial num_points / total_symmetries

theorem star_arrangements :
  distinct_arrangements = Nat.factorial (num_points - 1) := by
  sorry

end NUMINAMATH_CALUDE_star_arrangements_l3059_305974


namespace NUMINAMATH_CALUDE_lyceum_students_count_l3059_305946

theorem lyceum_students_count :
  ∀ n : ℕ,
  (1000 < n ∧ n < 2000) →
  (n * 76 % 100 = 0) →
  (n * 5 % 37 = 0) →
  n = 1850 :=
by
  sorry

end NUMINAMATH_CALUDE_lyceum_students_count_l3059_305946


namespace NUMINAMATH_CALUDE_product_divisible_by_twelve_l3059_305933

theorem product_divisible_by_twelve (a b c d : ℤ) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) : 
  12 ∣ ((a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d)) := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_twelve_l3059_305933


namespace NUMINAMATH_CALUDE_union_of_sets_l3059_305930

theorem union_of_sets : 
  let M : Set ℕ := {0, 1, 3}
  let N : Set ℕ := {x | x ∈ ({0, 3, 9} : Set ℕ)}
  M ∪ N = {0, 1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l3059_305930


namespace NUMINAMATH_CALUDE_constant_calculation_l3059_305917

theorem constant_calculation (N : ℝ) (C : ℝ) : 
  N = 12.0 → C + 0.6667 * N = 0.75 * N → C = 0.9996 := by
  sorry

end NUMINAMATH_CALUDE_constant_calculation_l3059_305917


namespace NUMINAMATH_CALUDE_cat_leash_max_distance_l3059_305951

theorem cat_leash_max_distance :
  let center : ℝ × ℝ := (6, 2)
  let radius : ℝ := 15
  let origin : ℝ × ℝ := (0, 0)
  let max_distance := radius + Real.sqrt ((center.1 - origin.1)^2 + (center.2 - origin.2)^2)
  max_distance = 15 + 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_cat_leash_max_distance_l3059_305951


namespace NUMINAMATH_CALUDE_sum_minus_k_equals_ten_l3059_305953

theorem sum_minus_k_equals_ten (n k : ℕ) (a : ℕ) (h1 : 1 < k) (h2 : k < n) 
  (h3 : (n * (n + 1) / 2 - k) / (n - 1) = 10) (h4 : n + k = a) : a = 29 := by
  sorry

end NUMINAMATH_CALUDE_sum_minus_k_equals_ten_l3059_305953


namespace NUMINAMATH_CALUDE_square_area_from_corners_l3059_305950

/-- The area of a square with adjacent corners at (4, -1) and (-1, 3) on a Cartesian coordinate plane is 41. -/
theorem square_area_from_corners : 
  let p1 : ℝ × ℝ := (4, -1)
  let p2 : ℝ × ℝ := (-1, 3)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  side_length^2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_corners_l3059_305950


namespace NUMINAMATH_CALUDE_subset_condition_l3059_305966

def A : Set ℝ := {x | x < -1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 ≤ 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ -1/3 ≤ a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l3059_305966


namespace NUMINAMATH_CALUDE_chord_inscribed_squares_side_difference_l3059_305904

/-- Given a circle with radius r and a chord at distance h from the center,
    prove that the difference in side lengths of two squares inscribed in the segments
    formed by the chord is 8h/5. -/
theorem chord_inscribed_squares_side_difference
  (r h : ℝ) (hr : r > 0) (hh : 0 < h ∧ h < r) :
  ∃ (a b : ℝ),
    (a > 0 ∧ b > 0) ∧
    (a - h)^2 = r^2 - (a^2 / 4) ∧
    (b + h)^2 = r^2 - (b^2 / 4) ∧
    b - a = (8 * h) / 5 :=
sorry

end NUMINAMATH_CALUDE_chord_inscribed_squares_side_difference_l3059_305904


namespace NUMINAMATH_CALUDE_pencil_cost_to_selling_ratio_l3059_305986

/-- The ratio of the cost to the selling price for 70 pencils -/
theorem pencil_cost_to_selling_ratio 
  (C : ℝ) -- Cost price of one pencil
  (S : ℝ) -- Selling price of one pencil
  (h1 : C > 0) -- Assumption that cost is positive
  (h2 : S > 0) -- Assumption that selling price is positive
  (h3 : C > (2/7) * S) -- Assumption that cost is greater than 2/7 of selling price
  : (70 * C) / (70 * C - 20 * S) = C / (C - 2 * S / 7) :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_to_selling_ratio_l3059_305986


namespace NUMINAMATH_CALUDE_exists_solution_l3059_305961

theorem exists_solution : ∃ (a b c d : ℕ+), 2014 = (a^2 + b^2) * (c^3 - d^3) := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_l3059_305961


namespace NUMINAMATH_CALUDE_water_depth_multiple_of_height_l3059_305919

theorem water_depth_multiple_of_height (ron_height : ℕ) (water_depth : ℕ) :
  ron_height = 13 →
  water_depth = 208 →
  ∃ k : ℕ, water_depth = k * ron_height →
  water_depth / ron_height = 16 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_multiple_of_height_l3059_305919


namespace NUMINAMATH_CALUDE_no_solution_exists_l3059_305909

theorem no_solution_exists : ¬∃ x : ℝ, (x / (-4) ≥ 3 + x) ∧ (|2 * x - 1| < 4 + 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3059_305909


namespace NUMINAMATH_CALUDE_quinton_cupcakes_l3059_305941

/-- The number of cupcakes Quinton brought to school -/
def total_cupcakes : ℕ := sorry

/-- The number of students in Ms. Delmont's class -/
def delmont_students : ℕ := 18

/-- The number of students in Mrs. Donnelly's class -/
def donnelly_students : ℕ := 16

/-- The number of staff members who received a cupcake -/
def staff_members : ℕ := 4

/-- The number of cupcakes left over -/
def leftover_cupcakes : ℕ := 2

/-- Theorem stating that the total number of cupcakes Quinton brought to school is 40 -/
theorem quinton_cupcakes : 
  total_cupcakes = delmont_students + donnelly_students + staff_members + leftover_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_quinton_cupcakes_l3059_305941


namespace NUMINAMATH_CALUDE_yunas_average_score_l3059_305914

/-- Given Yuna's average score for May and June and her July score, 
    calculate her average score over the three months. -/
theorem yunas_average_score 
  (may_june_avg : ℝ) 
  (july_score : ℝ) 
  (h1 : may_june_avg = 84) 
  (h2 : july_score = 96) : 
  (2 * may_june_avg + july_score) / 3 = 88 := by
  sorry

#eval (2 * 84 + 96) / 3  -- This should evaluate to 88

end NUMINAMATH_CALUDE_yunas_average_score_l3059_305914


namespace NUMINAMATH_CALUDE_first_obtuse_triangle_l3059_305925

/-- Represents a triangle with three angles -/
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

/-- Constructs the pedal triangle of a given triangle -/
def pedal_triangle (t : Triangle) : Triangle :=
  { angle1 := 180 - 2 * t.angle1,
    angle2 := 180 - 2 * t.angle2,
    angle3 := 180 - 2 * t.angle3 }

/-- Checks if a triangle is obtuse -/
def is_obtuse (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

/-- Generates the nth pedal triangle in the sequence -/
def nth_pedal_triangle (n : Nat) : Triangle :=
  match n with
  | 0 => { angle1 := 59.5, angle2 := 60, angle3 := 60.5 }
  | n + 1 => pedal_triangle (nth_pedal_triangle n)

theorem first_obtuse_triangle :
  ∀ n : Nat, n < 6 → ¬(is_obtuse (nth_pedal_triangle n)) ∧
  is_obtuse (nth_pedal_triangle 6) :=
by sorry

end NUMINAMATH_CALUDE_first_obtuse_triangle_l3059_305925


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l3059_305928

theorem smaller_solution_quadratic (x : ℝ) : 
  (x^2 - 7*x - 18 = 0) → (x = -2 ∨ x = 9) → -2 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l3059_305928


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l3059_305960

def total_players : ℕ := 15
def quadruplets : ℕ := 4
def starters : ℕ := 6
def max_quadruplets_in_lineup : ℕ := 2

theorem basketball_lineup_combinations : 
  (Nat.choose (total_players - quadruplets) starters) + 
  (quadruplets * Nat.choose (total_players - quadruplets) (starters - 1)) + 
  (Nat.choose quadruplets 2 * Nat.choose (total_players - quadruplets) (starters - 2)) = 4290 :=
sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l3059_305960


namespace NUMINAMATH_CALUDE_polynomial_factorization_and_trig_inequality_l3059_305985

theorem polynomial_factorization_and_trig_inequality :
  (∀ x : ℂ, x^12 + x^9 + x^6 + x^3 + 1 = (x^4 + x^3 + x^2 + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1)) ∧
  (∀ θ : ℝ, 5 + 8 * Real.cos θ + 4 * Real.cos (2 * θ) + Real.cos (3 * θ) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_and_trig_inequality_l3059_305985


namespace NUMINAMATH_CALUDE_special_arithmetic_sequence_2007th_term_l3059_305990

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The 3rd, 5th, and 11th terms form a geometric sequence -/
  geometric_property : (a + 2*d) * (a + 10*d) = (a + 4*d)^2
  /-- The 4th term is 6 -/
  fourth_term : a + 3*d = 6

/-- The nth term of an arithmetic sequence -/
def arithmetic_term (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1) * seq.d

/-- The main theorem -/
theorem special_arithmetic_sequence_2007th_term 
  (seq : SpecialArithmeticSequence) : 
  arithmetic_term seq 2007 = 6015 := by
  sorry

end NUMINAMATH_CALUDE_special_arithmetic_sequence_2007th_term_l3059_305990


namespace NUMINAMATH_CALUDE_last_locker_opened_l3059_305955

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Toggles the state of a locker -/
def toggleLocker (state : LockerState) : LockerState :=
  match state with
  | LockerState.Open => LockerState.Closed
  | LockerState.Closed => LockerState.Open

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

/-- The main theorem stating that the last locker opened is 509 -/
theorem last_locker_opened :
  ∀ (n : Nat), n ≤ 512 →
    (isPerfectSquare n ↔ (
      ∀ (k : Nat), k ≤ 512 →
        (n % k = 0 → toggleLocker (
          if k = 1 then LockerState.Closed
          else if k < n then toggleLocker LockerState.Closed
          else LockerState.Closed
        ) = LockerState.Open)
    )) →
  (∀ m : Nat, m > 509 ∧ m ≤ 512 →
    ¬(∀ (k : Nat), k ≤ 512 →
      (m % k = 0 → toggleLocker (
        if k = 1 then LockerState.Closed
        else if k < m then toggleLocker LockerState.Closed
        else LockerState.Closed
      ) = LockerState.Open))) →
  isPerfectSquare 509 :=
by
  sorry


end NUMINAMATH_CALUDE_last_locker_opened_l3059_305955


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l3059_305926

theorem simplify_complex_fraction :
  (1 / ((1 / (Real.sqrt 5 + 2)) + (2 / (Real.sqrt 7 - 2)))) =
  ((6 * Real.sqrt 7 + 9 * Real.sqrt 5 + 6) / (118 + 12 * Real.sqrt 35)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l3059_305926


namespace NUMINAMATH_CALUDE_haleigh_leggings_count_l3059_305944

/-- The number of leggings needed for Haleigh's pets -/
def total_leggings : ℕ :=
  let num_dogs := 4
  let num_cats := 3
  let num_spiders := 2
  let num_parrots := 1
  let dog_legs := 4
  let cat_legs := 4
  let spider_legs := 8
  let parrot_legs := 2
  num_dogs * dog_legs + num_cats * cat_legs + num_spiders * spider_legs + num_parrots * parrot_legs

theorem haleigh_leggings_count : total_leggings = 46 := by
  sorry

end NUMINAMATH_CALUDE_haleigh_leggings_count_l3059_305944


namespace NUMINAMATH_CALUDE_max_k_value_l3059_305949

theorem max_k_value (x y k : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_k : k > 0)
  (h_eq : 3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3059_305949


namespace NUMINAMATH_CALUDE_probability_of_double_l3059_305969

-- Define the range of integers for the mini-domino set
def dominoRange : ℕ := 7

-- Define a function to calculate the total number of pairings
def totalPairings (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the number of doubles in the set
def numDoubles : ℕ := dominoRange

-- Theorem statement
theorem probability_of_double :
  (numDoubles : ℚ) / (totalPairings dominoRange : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_double_l3059_305969


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l3059_305965

theorem solution_to_system_of_equations :
  let system (x y z : ℝ) : Prop :=
    x^2 - y^2 + z = 64 / (x * y) ∧
    y^2 - z^2 + x = 64 / (y * z) ∧
    z^2 - x^2 + y = 64 / (x * z)
  ∀ x y z : ℝ, system x y z →
    ((x = 4 ∧ y = 4 ∧ z = 4) ∨
     (x = -4 ∧ y = -4 ∧ z = 4) ∨
     (x = -4 ∧ y = 4 ∧ z = -4) ∨
     (x = 4 ∧ y = -4 ∧ z = -4)) :=
by
  sorry

#check solution_to_system_of_equations

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l3059_305965


namespace NUMINAMATH_CALUDE_submarine_hit_guaranteed_l3059_305911

-- Define the type for the submarine's position and velocity
def Submarine := ℕ × ℕ+

-- Define the type for the firing sequence
def FiringSequence := ℕ → ℕ

-- The theorem statement
theorem submarine_hit_guaranteed :
  ∀ (sub : Submarine), ∃ (fire : FiringSequence), ∃ (t : ℕ),
    fire t = (sub.2 : ℕ) * t + sub.1 :=
by sorry

end NUMINAMATH_CALUDE_submarine_hit_guaranteed_l3059_305911


namespace NUMINAMATH_CALUDE_johns_game_percentage_l3059_305959

theorem johns_game_percentage (shots_per_foul : ℕ) (fouls_per_game : ℕ) (total_games : ℕ) (actual_shots : ℕ) :
  shots_per_foul = 2 →
  fouls_per_game = 5 →
  total_games = 20 →
  actual_shots = 112 →
  (actual_shots : ℚ) / ((shots_per_foul * fouls_per_game * total_games) : ℚ) * 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_johns_game_percentage_l3059_305959


namespace NUMINAMATH_CALUDE_inverse_of_5_mod_34_l3059_305942

theorem inverse_of_5_mod_34 : ∃ x : ℕ, x < 34 ∧ (5 * x) % 34 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_5_mod_34_l3059_305942


namespace NUMINAMATH_CALUDE_reconstruct_coordinate_system_l3059_305979

-- Define a parabola as a function
def parabola (x : ℝ) : ℝ := x^2

-- Define a point on the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line on the coordinate plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the y-axis (line of symmetry)
def y_axis : Line := sorry

-- Define the x-axis
def x_axis : Line := sorry

-- Define the unit length
def unit_length : ℝ := sorry

-- Theorem stating that the y-axis, x-axis, and unit length can be uniquely determined
theorem reconstruct_coordinate_system (p : Point → Prop) :
  (∀ x, p ⟨x, parabola x⟩) →  -- Given only points on the parabola
  ∃! (y_axis : Line) (x_axis : Line) (unit : ℝ),
    (y_axis.a = 1 ∧ y_axis.b = 0) ∧  -- y-axis is vertical
    (x_axis.a = 0 ∧ x_axis.b = 1) ∧  -- x-axis is horizontal
    (y_axis.c = 0) ∧  -- y-axis passes through origin
    (x_axis.c = 0) ∧  -- x-axis passes through origin
    (unit = 1) :=  -- unit length is 1
by sorry

end NUMINAMATH_CALUDE_reconstruct_coordinate_system_l3059_305979


namespace NUMINAMATH_CALUDE_production_days_l3059_305902

theorem production_days (n : ℕ) 
  (h1 : (n * 40 + 90) / (n + 1) = 45) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l3059_305902


namespace NUMINAMATH_CALUDE_extra_workers_for_deeper_hole_extra_workers_needed_l3059_305940

/-- Represents the number of workers needed for a digging task. -/
def workers_needed (initial_workers : ℕ) (initial_depth : ℕ) (initial_hours : ℕ) 
                   (target_depth : ℕ) (target_hours : ℕ) : ℕ :=
  (initial_workers * initial_hours * target_depth) / (initial_depth * target_hours)

/-- Theorem stating the number of workers needed for the new digging task. -/
theorem extra_workers_for_deeper_hole 
  (initial_workers : ℕ) (initial_depth : ℕ) (initial_hours : ℕ)
  (target_depth : ℕ) (target_hours : ℕ) :
  initial_workers = 45 → 
  initial_depth = 30 → 
  initial_hours = 8 → 
  target_depth = 70 → 
  target_hours = 5 → 
  workers_needed initial_workers initial_depth initial_hours target_depth target_hours = 168 :=
by
  sorry

/-- Calculates the extra workers needed based on the initial and required number of workers. -/
def extra_workers (initial : ℕ) (required : ℕ) : ℕ :=
  required - initial

/-- Theorem stating the number of extra workers needed for the new digging task. -/
theorem extra_workers_needed 
  (initial_workers : ℕ) (required_workers : ℕ) :
  initial_workers = 45 →
  required_workers = 168 →
  extra_workers initial_workers required_workers = 123 :=
by
  sorry

end NUMINAMATH_CALUDE_extra_workers_for_deeper_hole_extra_workers_needed_l3059_305940


namespace NUMINAMATH_CALUDE_altitude_of_triangle_on_rectangle_diagonal_l3059_305972

theorem altitude_of_triangle_on_rectangle_diagonal (l : ℝ) (h : l > 0) :
  let w := l * Real.sqrt 2 / 2
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := diagonal * altitude / 2
  triangle_area = rectangle_area →
  altitude = l * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_altitude_of_triangle_on_rectangle_diagonal_l3059_305972


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3059_305988

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- Definition of the line that intersects C -/
def L (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

/-- Definition of points A and B as intersections of C and L -/
def intersectionPoints (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂

/-- Condition for OA ⊥ OB -/
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

/-- Theorem stating the conditions for perpendicularity and the length of AB -/
theorem ellipse_intersection_theorem :
  ∀ k : ℝ, intersectionPoints k →
    (k = 1/2 ∨ k = -1/2) ↔
      (∃ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂ ∧
        perpendicular x₁ y₁ x₂ y₂ ∧
        ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2) = 4*(65^(1/2))/17) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3059_305988


namespace NUMINAMATH_CALUDE_min_students_glasses_and_pet_l3059_305980

theorem min_students_glasses_and_pet 
  (total : ℕ) (glasses : ℕ) (pet : ℕ) 
  (h1 : total = 35) 
  (h2 : glasses = 18) 
  (h3 : pet = 25) :
  glasses + pet - total ≤ (glasses + pet - total).max 0 ∧
  (glasses + pet - total).max 0 = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_students_glasses_and_pet_l3059_305980


namespace NUMINAMATH_CALUDE_chord_equation_l3059_305991

/-- The equation of the line on which the chord common to two circles lies -/
theorem chord_equation (r : ℝ) (ρ θ : ℝ) (h : r > 0) :
  (ρ = r ∨ ρ = -2 * r * Real.sin (θ + π/4)) →
  Real.sqrt 2 * ρ * (Real.sin θ + Real.cos θ) = -r :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l3059_305991


namespace NUMINAMATH_CALUDE_mike_grew_four_onions_l3059_305936

/-- The number of onions grown by Mike given the number of onions grown by Nancy, Dan, and the total number of onions. -/
def mikes_onions (nancy_onions dan_onions total_onions : ℕ) : ℕ :=
  total_onions - (nancy_onions + dan_onions)

/-- Theorem stating that Mike grew 4 onions given the conditions. -/
theorem mike_grew_four_onions :
  mikes_onions 2 9 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mike_grew_four_onions_l3059_305936


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_regular_triangular_pyramid_volume_is_correct_l3059_305989

/-- The volume of a regular triangular pyramid with specific properties -/
theorem regular_triangular_pyramid_volume 
  (r : ℝ) -- Length of the perpendicular from the base of the height to a lateral edge
  (α : ℝ) -- Dihedral angle between the lateral face and the base of the pyramid
  (h1 : 0 < r) -- r is positive
  (h2 : 0 < α) -- α is positive
  (h3 : α < π / 2) -- α is less than 90 degrees
  : ℝ :=
  let volume := (Real.sqrt 3 * r^3 * Real.sqrt ((4 + Real.tan α ^ 2) ^ 3)) / (8 * Real.tan α ^ 2)
  volume

#check regular_triangular_pyramid_volume

theorem regular_triangular_pyramid_volume_is_correct
  (r : ℝ) -- Length of the perpendicular from the base of the height to a lateral edge
  (α : ℝ) -- Dihedral angle between the lateral face and the base of the pyramid
  (h1 : 0 < r) -- r is positive
  (h2 : 0 < α) -- α is positive
  (h3 : α < π / 2) -- α is less than 90 degrees
  : regular_triangular_pyramid_volume r α h1 h2 h3 = 
    (Real.sqrt 3 * r^3 * Real.sqrt ((4 + Real.tan α ^ 2) ^ 3)) / (8 * Real.tan α ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_regular_triangular_pyramid_volume_is_correct_l3059_305989


namespace NUMINAMATH_CALUDE_option_d_most_suitable_for_comprehensive_survey_l3059_305943

/-- Represents a survey option -/
inductive SurveyOption
| A : SurveyOption  -- Investigating the service life of a batch of infrared thermometers
| B : SurveyOption  -- Investigating the travel methods of the people of Henan during the Spring Festival
| C : SurveyOption  -- Investigating the viewership of the Henan TV program "Li Yuan Chun"
| D : SurveyOption  -- Investigating the heights of all classmates

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  population_size : ℕ
  precision : ℝ

/-- Defines what makes a survey suitable for a comprehensive survey -/
def is_suitable_for_comprehensive_survey (s : SurveyCharacteristics) : Prop :=
  s.population_size ≤ 1000 ∧ s.precision ≥ 0.99

/-- Associates survey options with their characteristics -/
def survey_characteristics : SurveyOption → SurveyCharacteristics
| SurveyOption.A => ⟨10000, 0.9⟩
| SurveyOption.B => ⟨20000000, 0.8⟩
| SurveyOption.C => ⟨5000000, 0.85⟩
| SurveyOption.D => ⟨50, 0.99⟩

/-- Theorem: Option D is the most suitable for a comprehensive survey -/
theorem option_d_most_suitable_for_comprehensive_survey :
  ∀ (o : SurveyOption), o ≠ SurveyOption.D →
    is_suitable_for_comprehensive_survey (survey_characteristics SurveyOption.D) ∧
    ¬is_suitable_for_comprehensive_survey (survey_characteristics o) :=
by sorry


end NUMINAMATH_CALUDE_option_d_most_suitable_for_comprehensive_survey_l3059_305943


namespace NUMINAMATH_CALUDE_unique_solution_l3059_305963

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = -f (-x)) ∧
  (∀ x, f (x + 1) = f x + 1) ∧
  (∀ x ≠ 0, f (1 / x) = (1 / x^2) * f x)

/-- Theorem stating that the only function satisfying the conditions is f(x) = x -/
theorem unique_solution (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3059_305963


namespace NUMINAMATH_CALUDE_l2_passes_through_point_perpendicular_implies_a_value_max_distance_to_l1_l3059_305983

-- Define the lines l1 and l2
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := 3 * x + (a - 1) * y + 3 - a = 0

-- Define the point P
def P : ℝ × ℝ := (1, 3)

-- Statement 1
theorem l2_passes_through_point : ∀ a : ℝ, l2 a (-2/3) 1 := by sorry

-- Statement 2
theorem perpendicular_implies_a_value : 
  ∀ a : ℝ, (∀ x y : ℝ, l1 a x y → l2 a x y → (a * 3 + 2 * (a - 1) = 0)) → a = 2/5 := by sorry

-- Statement 3
theorem max_distance_to_l1 : 
  ∀ a : ℝ, ∃ x y : ℝ, l1 a x y ∧ Real.sqrt ((x - P.1)^2 + (y - P.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_l2_passes_through_point_perpendicular_implies_a_value_max_distance_to_l1_l3059_305983


namespace NUMINAMATH_CALUDE_coefficient_of_3x2y_l3059_305929

/-- The coefficient of a monomial is the numerical factor that multiplies the variable part. -/
def coefficient (m : ℕ → ℕ → ℚ) : ℚ := m 0 0

/-- A monomial is represented as a function from ℕ × ℕ to ℚ, where m i j represents the coefficient of x^i * y^j. -/
def monomial_3x2y : ℕ → ℕ → ℚ := fun i j => if i = 2 ∧ j = 1 then 3 else 0

theorem coefficient_of_3x2y :
  coefficient monomial_3x2y = 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_3x2y_l3059_305929


namespace NUMINAMATH_CALUDE_rectangular_park_diagonal_l3059_305923

theorem rectangular_park_diagonal (x y : ℝ) (h_positive : x > 0 ∧ y > 0) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x / y = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_park_diagonal_l3059_305923


namespace NUMINAMATH_CALUDE_polygon_area_theorem_l3059_305962

/-- The area of a polygon with given vertices -/
def polygonArea (vertices : List (ℤ × ℤ)) : ℚ :=
  sorry

/-- The number of integer points strictly inside a polygon -/
def interiorPoints (vertices : List (ℤ × ℤ)) : ℕ :=
  sorry

/-- The number of integer points on the boundary of a polygon -/
def boundaryPoints (vertices : List (ℤ × ℤ)) : ℕ :=
  sorry

theorem polygon_area_theorem :
  let vertices : List (ℤ × ℤ) := [(0, 1), (1, 2), (3, 2), (4, 1), (2, 0)]
  polygonArea vertices = 15/2 ∧
  interiorPoints vertices = 6 ∧
  boundaryPoints vertices = 5 :=
by sorry

end NUMINAMATH_CALUDE_polygon_area_theorem_l3059_305962


namespace NUMINAMATH_CALUDE_infinite_sum_solution_l3059_305968

theorem infinite_sum_solution (k : ℝ) (h1 : k > 2) 
  (h2 : (∑' n, (6 * n + 2) / k^n) = 15) : 
  k = (38 + 2 * Real.sqrt 46) / 30 := by
sorry

end NUMINAMATH_CALUDE_infinite_sum_solution_l3059_305968


namespace NUMINAMATH_CALUDE_H_points_infinite_but_not_all_l3059_305901

-- Define the curve C
def C : Set (ℝ × ℝ) := {p | p.1^2 / 4 + p.2^2 = 1}

-- Define the line l
def l : Set (ℝ × ℝ) := {p | p.1 = 4}

-- Define what it means to be an H point
def is_H_point (P : ℝ × ℝ) : Prop :=
  P ∈ C ∧ ∃ (A B : ℝ × ℝ) (k m : ℝ),
    A ∈ C ∧ B ∈ l ∧
    (∀ x y, y = k * x + m ↔ (x, y) ∈ ({P, A, B} : Set (ℝ × ℝ))) ∧
    (dist P A = dist P B ∨ dist P A = dist A B)

-- Define the set of H points
def H_points : Set (ℝ × ℝ) := {P | is_H_point P}

-- The theorem to be proved
theorem H_points_infinite_but_not_all :
  Set.Infinite H_points ∧ H_points ≠ C :=
sorry


end NUMINAMATH_CALUDE_H_points_infinite_but_not_all_l3059_305901


namespace NUMINAMATH_CALUDE_students_not_playing_sports_l3059_305915

theorem students_not_playing_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (basketball : ℕ)
  (football_tennis : ℕ) (football_basketball : ℕ) (tennis_basketball : ℕ) (all_three : ℕ)
  (h_total : total = 50)
  (h_football : football = 26)
  (h_tennis : tennis = 20)
  (h_basketball : basketball = 15)
  (h_football_tennis : football_tennis = 9)
  (h_football_basketball : football_basketball = 7)
  (h_tennis_basketball : tennis_basketball = 6)
  (h_all_three : all_three = 4) :
  total - (football + tennis + basketball - football_tennis - football_basketball - tennis_basketball + all_three) = 7 := by
sorry

end NUMINAMATH_CALUDE_students_not_playing_sports_l3059_305915


namespace NUMINAMATH_CALUDE_range_of_a_part1_range_of_a_part2_l3059_305916

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

-- Theorem for the first part of the problem
theorem range_of_a_part1 (a : ℝ) :
  (∃ x ∈ Set.Icc 1 3, f a x > 0) → a < 4 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_a_part2 (a : ℝ) :
  (∀ x ∈ Set.Icc 1 4, f a x ≥ -a) → a ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_part1_range_of_a_part2_l3059_305916


namespace NUMINAMATH_CALUDE_min_cuts_correct_l3059_305922

/-- The minimum number of cuts required to divide a cube of edge length 4 into 64 unit cubes -/
def min_cuts : ℕ := 6

/-- The edge length of the initial cube -/
def initial_edge_length : ℕ := 4

/-- The number of smaller cubes we want to create -/
def target_num_cubes : ℕ := 64

/-- The edge length of the smaller cubes -/
def target_edge_length : ℕ := 1

/-- Theorem stating that min_cuts is the minimum number of cuts required -/
theorem min_cuts_correct :
  (2 ^ min_cuts = target_num_cubes) ∧
  (∀ n : ℕ, n < min_cuts → 2 ^ n < target_num_cubes) :=
sorry

end NUMINAMATH_CALUDE_min_cuts_correct_l3059_305922


namespace NUMINAMATH_CALUDE_hundredth_term_equals_30503_l3059_305970

/-- A sequence of geometric designs -/
def f (n : ℕ) : ℕ := 3 * n^2 + 5 * n + 3

/-- The theorem stating that the 100th term of the sequence equals 30503 -/
theorem hundredth_term_equals_30503 :
  f 0 = 3 ∧ f 1 = 11 ∧ f 2 = 25 ∧ f 3 = 45 → f 100 = 30503 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_equals_30503_l3059_305970


namespace NUMINAMATH_CALUDE_b_approximation_l3059_305992

/-- Given that a = 2.68 * 0.74, prove that b = a^2 + cos(a) is approximately 2.96535 -/
theorem b_approximation (a : ℝ) (h : a = 2.68 * 0.74) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ |a^2 + Real.cos a - 2.96535| < ε :=
sorry

end NUMINAMATH_CALUDE_b_approximation_l3059_305992


namespace NUMINAMATH_CALUDE_f_derivative_at_2_when_a_0_f_minimum_at_0_iff_a_lt_2_g_not_tangent_to_line_with_slope_3_2_l3059_305999

noncomputable section

open Real

/-- The base of the natural logarithm -/
def e : ℝ := exp 1

/-- The function f(x) = (x^2 + ax + a)e^(-x) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x + a) * (e^(-x))

/-- The function g(x) = (4 - x)e^(x - 2) for x < 2 -/
def g (x : ℝ) : ℝ := (4 - x) * (e^(x - 2))

theorem f_derivative_at_2_when_a_0 :
  (deriv (f 0)) 2 = 0 := by sorry

theorem f_minimum_at_0_iff_a_lt_2 (a : ℝ) :
  (∀ x, f a 0 ≤ f a x) ↔ a < 2 := by sorry

theorem g_not_tangent_to_line_with_slope_3_2 :
  ¬ ∃ (c : ℝ), ∃ (x : ℝ), x < 2 ∧ g x = (3/2) * x + c ∧ (deriv g) x = 3/2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_when_a_0_f_minimum_at_0_iff_a_lt_2_g_not_tangent_to_line_with_slope_3_2_l3059_305999


namespace NUMINAMATH_CALUDE_percent_error_multiplication_l3059_305910

theorem percent_error_multiplication (x : ℝ) (h : x > 0) : 
  (|12 * x - x / 3| / (x / 3)) * 100 = 3500 := by
sorry

end NUMINAMATH_CALUDE_percent_error_multiplication_l3059_305910


namespace NUMINAMATH_CALUDE_range_of_f_l3059_305997

def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

theorem range_of_f :
  Set.range f = Set.Icc (-8 : ℝ) 8 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3059_305997


namespace NUMINAMATH_CALUDE_count_negative_numbers_l3059_305932

def number_list : List ℚ := [-2 - 2/3, 9/14, -3, 5/2, 0, -48/10, 5, -1]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l3059_305932


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3059_305931

/-- The repeating decimal 5.8̄ -/
def repeating_decimal : ℚ := 5 + 8/9

/-- The fraction 53/9 -/
def target_fraction : ℚ := 53/9

/-- Theorem stating that the repeating decimal 5.8̄ is equal to the fraction 53/9 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3059_305931


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3059_305907

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The common ratio of a geometric sequence -/
def CommonRatio (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h1 : GeometricSequence a)
    (h2 : a 2 + a 4 = 3)
    (h3 : a 3 * a 5 = 1) :
    ∃ q : ℝ, CommonRatio a q ∧ q = Real.sqrt 2 / 2 ∧
    ∀ n : ℕ, a n = 2 ^ ((n + 2 : ℝ) / 2) :=
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3059_305907


namespace NUMINAMATH_CALUDE_complex_number_equation_l3059_305913

theorem complex_number_equation (z : ℂ) 
  (h : 15 * Complex.normSq z = 5 * Complex.normSq (z + 1) + Complex.normSq (z^2 - 1) + 44) : 
  z^2 + 36 / z^2 = 60 := by sorry

end NUMINAMATH_CALUDE_complex_number_equation_l3059_305913


namespace NUMINAMATH_CALUDE_length_EC_l3059_305906

-- Define the points
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : ∃ t : ℝ, C = A + t • (C - A) ∧ D = B + t • (D - B))
variable (h2 : ‖A - E‖ = ‖A - B‖ - 1)
variable (h3 : ‖A - E‖ = ‖D - C‖)
variable (h4 : ‖A - D‖ = ‖B - E‖)
variable (h5 : angle A D C = angle D E C)

-- The theorem to prove
theorem length_EC : ‖E - C‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_length_EC_l3059_305906


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l3059_305978

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

/-- Theorem: The 50th term of the arithmetic sequence starting with 2 and incrementing by 5 is 247 -/
theorem fiftieth_term_of_sequence : arithmetic_sequence 2 5 50 = 247 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l3059_305978


namespace NUMINAMATH_CALUDE_wade_sandwich_cost_l3059_305945

def sandwich_cost (total_spent : ℚ) (num_sandwiches : ℕ) (num_drinks : ℕ) (drink_cost : ℚ) : ℚ :=
  (total_spent - (num_drinks : ℚ) * drink_cost) / (num_sandwiches : ℚ)

theorem wade_sandwich_cost :
  sandwich_cost 26 3 2 4 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_wade_sandwich_cost_l3059_305945


namespace NUMINAMATH_CALUDE_complex_power_equality_l3059_305900

theorem complex_power_equality (n : ℕ) (hn : n ≤ 1000) :
  ∀ t : ℝ, (Complex.cos t - Complex.I * Complex.sin t) ^ n = Complex.cos (n * t) - Complex.I * Complex.sin (n * t) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_equality_l3059_305900


namespace NUMINAMATH_CALUDE_balls_without_holes_count_l3059_305964

/-- The number of soccer balls Matthias has -/
def total_soccer_balls : ℕ := 40

/-- The number of basketballs Matthias has -/
def total_basketballs : ℕ := 15

/-- The number of soccer balls with holes -/
def soccer_balls_with_holes : ℕ := 30

/-- The number of basketballs with holes -/
def basketballs_with_holes : ℕ := 7

/-- The total number of balls without holes -/
def total_balls_without_holes : ℕ := 
  (total_soccer_balls - soccer_balls_with_holes) + (total_basketballs - basketballs_with_holes)

theorem balls_without_holes_count : total_balls_without_holes = 18 := by
  sorry

end NUMINAMATH_CALUDE_balls_without_holes_count_l3059_305964


namespace NUMINAMATH_CALUDE_infinite_primes_dividing_sequence_l3059_305927

theorem infinite_primes_dividing_sequence (a b c : ℕ) (ha : a ≠ c) (hb : b ≠ c) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p ∣ a^n + b^n - c^n} := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_dividing_sequence_l3059_305927


namespace NUMINAMATH_CALUDE_fraction_equality_l3059_305977

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (1/a + 1/b) / (1/a - 1/b) = 1001) : 
  (a + b) / (a - b) = 1001 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3059_305977


namespace NUMINAMATH_CALUDE_largest_n_is_max_factorization_exists_l3059_305924

/-- The largest value of n for which 4x^2 + nx + 96 can be factored as two linear factors with integer coefficients -/
def largest_n : ℕ := 385

/-- A structure representing the factorization of 4x^2 + nx + 96 -/
structure Factorization where
  a : ℤ
  b : ℤ
  h1 : (4 * X + a) * (X + b) = 4 * X^2 + largest_n * X + 96

/-- Theorem stating that largest_n is indeed the largest value for which the factorization exists -/
theorem largest_n_is_max :
  ∀ n : ℕ, n > largest_n →
    ¬∃ (f : Factorization), (4 * X + f.a) * (X + f.b) = 4 * X^2 + n * X + 96 :=
by sorry

/-- Theorem stating that a factorization exists for largest_n -/
theorem factorization_exists : ∃ (f : Factorization), True :=
by sorry

end NUMINAMATH_CALUDE_largest_n_is_max_factorization_exists_l3059_305924


namespace NUMINAMATH_CALUDE_bookstore_ratio_l3059_305920

theorem bookstore_ratio : 
  ∀ (sarah_paperback sarah_hardback brother_total : ℕ),
    sarah_paperback = 6 →
    sarah_hardback = 4 →
    brother_total = 10 →
    ∃ (brother_paperback brother_hardback : ℕ),
      brother_paperback = sarah_paperback / 3 →
      brother_hardback + brother_paperback = brother_total →
      (brother_hardback : ℚ) / sarah_hardback = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_bookstore_ratio_l3059_305920


namespace NUMINAMATH_CALUDE_smithtown_population_ratio_l3059_305908

/-- Represents the population of Smithtown -/
structure Population where
  total : ℝ
  rightHanded : ℝ
  leftHanded : ℝ
  men : ℝ
  women : ℝ
  leftHandedWomen : ℝ

/-- The conditions given in the problem -/
def populationConditions (p : Population) : Prop :=
  p.rightHanded / p.leftHanded = 3 ∧
  p.leftHandedWomen / p.total = 0.2500000000000001 ∧
  p.rightHanded = p.men

/-- The theorem to be proved -/
theorem smithtown_population_ratio
  (p : Population)
  (h : populationConditions p) :
  p.men / p.women = 3 := by
  sorry

end NUMINAMATH_CALUDE_smithtown_population_ratio_l3059_305908


namespace NUMINAMATH_CALUDE_division_problem_l3059_305981

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 1 / 2)
  : c / a = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3059_305981


namespace NUMINAMATH_CALUDE_initial_amount_equals_sum_l3059_305938

/-- The amount of money Agatha initially had to spend on the bike. -/
def initial_amount : ℕ := 60

/-- The amount Agatha spent on the frame. -/
def frame_cost : ℕ := 15

/-- The amount Agatha spent on the front wheel. -/
def front_wheel_cost : ℕ := 25

/-- The amount Agatha has left for the seat and handlebar tape. -/
def remaining_amount : ℕ := 20

/-- Theorem stating that the initial amount equals the sum of all expenses and remaining amount. -/
theorem initial_amount_equals_sum :
  initial_amount = frame_cost + front_wheel_cost + remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_equals_sum_l3059_305938


namespace NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l3059_305982

theorem irrational_sqrt_three_rational_others : 
  (Irrational (Real.sqrt 3)) ∧ 
  (¬ Irrational (-8 : ℝ)) ∧ 
  (¬ Irrational (0.3070809 : ℝ)) ∧ 
  (¬ Irrational (22 / 7 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l3059_305982


namespace NUMINAMATH_CALUDE_donut_problem_l3059_305921

theorem donut_problem (D : ℕ) : (D - 6) / 2 = 22 ↔ D = 50 := by
  sorry

end NUMINAMATH_CALUDE_donut_problem_l3059_305921


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_28_l3059_305948

def digit_sum (n : ℕ) : ℕ := sorry

def is_prime (n : ℕ) : Prop := sorry

theorem smallest_prime_with_digit_sum_28 :
  (is_prime 1999) ∧ 
  (digit_sum 1999 = 28) ∧ 
  (∀ m : ℕ, m < 1999 → (is_prime m ∧ digit_sum m = 28) → False) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_28_l3059_305948


namespace NUMINAMATH_CALUDE_additional_gas_needed_l3059_305947

/-- Calculates the additional gallons of gas needed for a truck to reach its destination. -/
theorem additional_gas_needed
  (miles_per_gallon : ℝ)
  (total_distance : ℝ)
  (current_gas : ℝ)
  (h1 : miles_per_gallon = 3)
  (h2 : total_distance = 90)
  (h3 : current_gas = 12) :
  (total_distance - current_gas * miles_per_gallon) / miles_per_gallon = 18 := by
  sorry

end NUMINAMATH_CALUDE_additional_gas_needed_l3059_305947


namespace NUMINAMATH_CALUDE_additional_workers_needed_l3059_305967

/-- Represents the problem of calculating additional workers needed to complete a construction project on time -/
theorem additional_workers_needed
  (total_days : ℕ) 
  (initial_workers : ℕ) 
  (days_passed : ℕ) 
  (work_completed : ℚ) 
  (h1 : total_days = 50)
  (h2 : initial_workers = 20)
  (h3 : days_passed = 25)
  (h4 : work_completed = 2/5)
  : ℕ := by
  sorry

#check additional_workers_needed

end NUMINAMATH_CALUDE_additional_workers_needed_l3059_305967


namespace NUMINAMATH_CALUDE_sum_of_roots_times_two_l3059_305935

theorem sum_of_roots_times_two (a b : ℝ) : 
  (a^2 + a - 6 = 0) → (b^2 + b - 6 = 0) → (2*a + 2*b = -2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_times_two_l3059_305935


namespace NUMINAMATH_CALUDE_used_car_clients_l3059_305957

theorem used_car_clients (num_cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) :
  num_cars = 18 →
  selections_per_client = 3 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / selections_per_client = 18 := by
  sorry

end NUMINAMATH_CALUDE_used_car_clients_l3059_305957


namespace NUMINAMATH_CALUDE_max_sum_circle_50_l3059_305912

/-- The maximum sum of x and y for integer solutions of x^2 + y^2 = 50 -/
theorem max_sum_circle_50 : 
  ∀ x y : ℤ, x^2 + y^2 = 50 → x + y ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_circle_50_l3059_305912


namespace NUMINAMATH_CALUDE_caitlin_age_l3059_305998

theorem caitlin_age (anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ)
  (h1 : anna_age = 42)
  (h2 : brianna_age = anna_age / 2)
  (h3 : caitlin_age = brianna_age - 5) :
  caitlin_age = 16 := by
sorry

end NUMINAMATH_CALUDE_caitlin_age_l3059_305998


namespace NUMINAMATH_CALUDE_vector_subtraction_l3059_305958

/-- Given vectors a and b in ℝ², prove that a - 2b equals (6, -7) -/
theorem vector_subtraction (a b : ℝ × ℝ) 
  (ha : a = (2, -1)) (hb : b = (-2, 3)) : 
  a - 2 • b = (6, -7) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3059_305958


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3059_305918

/-- Given an arithmetic sequence {a_n} where a₁₀ = 30 and a₂₀ = 50,
    the general term is a_n = 2n + 10 -/
theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)  -- The sequence
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence condition
  (h_10 : a 10 = 30)  -- Given condition
  (h_20 : a 20 = 50)  -- Given condition
  : ∀ n : ℕ, a n = 2 * n + 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3059_305918


namespace NUMINAMATH_CALUDE_intersection_M_N_l3059_305976

-- Define the sets M and N
def M : Set ℝ := {x | 2 * x - x^2 ≥ 0}
def N : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - x^2)}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3059_305976


namespace NUMINAMATH_CALUDE_average_of_three_l3059_305975

theorem average_of_three (y : ℝ) : (15 + 24 + y) / 3 = 20 → y = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_l3059_305975


namespace NUMINAMATH_CALUDE_percent_of_y_l3059_305952

theorem percent_of_y (y : ℝ) (h : y > 0) : ((6 * y) / 20 + (3 * y) / 10) / y = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l3059_305952


namespace NUMINAMATH_CALUDE_vector_equation_m_range_l3059_305903

theorem vector_equation_m_range :
  ∀ (m n x : ℝ),
  (∃ x, (n + 2, n - Real.cos x ^ 2) = (2 * m, m + Real.sin x)) →
  (∀ m', (∃ n' x', (n' + 2, n' - Real.cos x' ^ 2) = (2 * m', m' + Real.sin x')) → 
    0 ≤ m' ∧ m' ≤ 4) ∧
  (∃ n₁ x₁, (n₁ + 2, n₁ - Real.cos x₁ ^ 2) = (2 * 0, 0 + Real.sin x₁)) ∧
  (∃ n₂ x₂, (n₂ + 2, n₂ - Real.cos x₂ ^ 2) = (2 * 4, 4 + Real.sin x₂)) :=
by sorry

end NUMINAMATH_CALUDE_vector_equation_m_range_l3059_305903


namespace NUMINAMATH_CALUDE_proposition_and_variants_are_false_l3059_305993

theorem proposition_and_variants_are_false :
  (¬ ∀ (a b : ℝ), ab ≤ 0 → a ≤ 0 ∨ b ≤ 0) ∧
  (¬ ∀ (a b : ℝ), (a ≤ 0 ∨ b ≤ 0) → ab ≤ 0) ∧
  (¬ ∀ (a b : ℝ), ab > 0 → a > 0 ∧ b > 0) ∧
  (¬ ∀ (a b : ℝ), (a > 0 ∧ b > 0) → ab > 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_variants_are_false_l3059_305993


namespace NUMINAMATH_CALUDE_sasha_guessing_game_l3059_305939

theorem sasha_guessing_game (X : ℕ) (hX : X ≤ 100) :
  ∃ (questions : List (ℕ × ℕ)),
    questions.length ≤ 7 ∧
    (∀ (M N : ℕ), (M, N) ∈ questions → M < 100 ∧ N < 100) ∧
    ∀ (Y : ℕ), Y ≤ 100 →
      (∀ (M N : ℕ), (M, N) ∈ questions →
        Nat.gcd (X + M) N = Nat.gcd (Y + M) N) →
      X = Y :=
by sorry

end NUMINAMATH_CALUDE_sasha_guessing_game_l3059_305939


namespace NUMINAMATH_CALUDE_lanas_final_pages_l3059_305956

def lanas_pages (initial_pages : ℕ) (duanes_pages : ℕ) : ℕ :=
  initial_pages + duanes_pages / 2

theorem lanas_final_pages :
  lanas_pages 8 42 = 29 := by sorry

end NUMINAMATH_CALUDE_lanas_final_pages_l3059_305956


namespace NUMINAMATH_CALUDE_polar_eq_of_cartesian_line_l3059_305996

/-- The polar coordinate equation ρ cos θ = 1 represents the line x = 1 in Cartesian coordinates -/
theorem polar_eq_of_cartesian_line (ρ θ : ℝ) :
  (ρ * Real.cos θ = 1) ↔ (ρ * Real.cos θ = 1) :=
by sorry

end NUMINAMATH_CALUDE_polar_eq_of_cartesian_line_l3059_305996


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_l3059_305971

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line passing through the focus with inclination angle π/4
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  line A.1 A.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem parabola_intersection_distance 
  (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_l3059_305971


namespace NUMINAMATH_CALUDE_max_distinct_letters_exists_table_with_11_letters_l3059_305934

/-- Represents a 5x5 table of letters -/
def LetterTable := Fin 5 → Fin 5 → Char

/-- Checks if a row contains at most 3 different letters -/
def rowValid (table : LetterTable) (row : Fin 5) : Prop :=
  (Finset.image (λ col => table row col) Finset.univ).card ≤ 3

/-- Checks if a column contains at most 3 different letters -/
def colValid (table : LetterTable) (col : Fin 5) : Prop :=
  (Finset.image (λ row => table row col) Finset.univ).card ≤ 3

/-- Checks if the entire table is valid -/
def tableValid (table : LetterTable) : Prop :=
  (∀ row, rowValid table row) ∧ (∀ col, colValid table col)

/-- Counts the number of different letters in the table -/
def distinctLetters (table : LetterTable) : ℕ :=
  (Finset.image (λ (row, col) => table row col) (Finset.univ.product Finset.univ)).card

/-- The main theorem stating that the maximum number of distinct letters is 11 -/
theorem max_distinct_letters :
  ∀ (table : LetterTable), tableValid table → distinctLetters table ≤ 11 :=
sorry

/-- There exists a valid table with exactly 11 distinct letters -/
theorem exists_table_with_11_letters :
  ∃ (table : LetterTable), tableValid table ∧ distinctLetters table = 11 :=
sorry

end NUMINAMATH_CALUDE_max_distinct_letters_exists_table_with_11_letters_l3059_305934


namespace NUMINAMATH_CALUDE_pencil_count_l3059_305937

/-- Represents the number of items in a stationery store -/
structure StationeryStore where
  pens : ℕ
  pencils : ℕ
  erasers : ℕ

/-- Conditions for the stationery store inventory -/
def validInventory (s : StationeryStore) : Prop :=
  ∃ (x : ℕ), 
    s.pens = 5 * x ∧
    s.pencils = 6 * x ∧
    s.erasers = 10 * x ∧
    s.pencils = s.pens + 6 ∧
    s.erasers = 2 * s.pens

theorem pencil_count (s : StationeryStore) (h : validInventory s) : s.pencils = 36 := by
  sorry

#check pencil_count

end NUMINAMATH_CALUDE_pencil_count_l3059_305937


namespace NUMINAMATH_CALUDE_int_roots_count_l3059_305995

/-- A polynomial of degree 4 with integer coefficients -/
structure IntPoly4 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The number of integer roots of a polynomial, counting multiplicity -/
def num_int_roots (p : IntPoly4) : ℕ := sorry

/-- Theorem stating the possible values for the number of integer roots -/
theorem int_roots_count (p : IntPoly4) : 
  num_int_roots p = 0 ∨ num_int_roots p = 1 ∨ num_int_roots p = 2 ∨ num_int_roots p = 4 :=
sorry

end NUMINAMATH_CALUDE_int_roots_count_l3059_305995


namespace NUMINAMATH_CALUDE_sugar_left_l3059_305973

/-- Given a recipe requiring 2 cups of sugar, if you can make 0.165 of the recipe,
    then you have 0.33 cups of sugar left. -/
theorem sugar_left (full_recipe : ℝ) (fraction_possible : ℝ) (sugar_left : ℝ) :
  full_recipe = 2 →
  fraction_possible = 0.165 →
  sugar_left = full_recipe * fraction_possible →
  sugar_left = 0.33 := by
sorry

end NUMINAMATH_CALUDE_sugar_left_l3059_305973


namespace NUMINAMATH_CALUDE_total_wings_is_14_l3059_305987

/-- Represents the types of birds available for purchase. -/
inductive BirdType
| Parrot
| Pigeon
| Canary

/-- Represents the money received from each grandparent. -/
def grandparentMoney : List ℕ := [45, 60, 55, 50]

/-- Represents the cost of each bird type. -/
def birdCost : BirdType → ℕ
| BirdType.Parrot => 35
| BirdType.Pigeon => 25
| BirdType.Canary => 20

/-- Represents the number of birds in a discounted set for each bird type. -/
def discountSet : BirdType → ℕ
| BirdType.Parrot => 3
| BirdType.Pigeon => 4
| BirdType.Canary => 5

/-- Represents the cost of a discounted set for each bird type. -/
def discountSetCost : BirdType → ℕ
| BirdType.Parrot => 35 * 2 + 35 / 2
| BirdType.Pigeon => 25 * 3
| BirdType.Canary => 20 * 4

/-- Represents the number of wings each bird has. -/
def wingsPerBird : ℕ := 2

/-- Represents the total money John has to spend. -/
def totalMoney : ℕ := grandparentMoney.sum

/-- Theorem stating that the total number of wings of all birds John bought is 14. -/
theorem total_wings_is_14 :
  ∃ (parrot pigeon canary : ℕ),
    parrot > 0 ∧ pigeon > 0 ∧ canary > 0 ∧
    parrot * birdCost BirdType.Parrot +
    pigeon * birdCost BirdType.Pigeon +
    canary * birdCost BirdType.Canary = totalMoney ∧
    (parrot + pigeon + canary) * wingsPerBird = 14 :=
  sorry

end NUMINAMATH_CALUDE_total_wings_is_14_l3059_305987


namespace NUMINAMATH_CALUDE_largest_divisible_by_8_l3059_305905

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

def number_format (a : ℕ) : ℕ := 365000 + a * 100 + 20

theorem largest_divisible_by_8 :
  ∀ a : ℕ, a ≤ 9 →
    is_divisible_by_8 (number_format 9) ∧
    (is_divisible_by_8 (number_format a) → a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_by_8_l3059_305905
