import Mathlib

namespace NUMINAMATH_CALUDE_infinite_non_fractional_numbers_l266_26621

/-- A number is p-good if it cannot be expressed as p^x * (p^(yz) - 1) / (p^y - 1) for any nonnegative integers x, y, z -/
def IsPGood (n : ℕ) (p : ℕ) : Prop :=
  ∀ x y z : ℕ, n ≠ p^x * (p^(y*z) - 1) / (p^y - 1)

/-- The set of numbers that cannot be expressed as (p^a - p^b) / (p^c - p^d) for any prime p and integers a, b, c, d -/
def NonFractionalSet : Set ℕ :=
  {n : ℕ | ∀ p : ℕ, Prime p → IsPGood n p}

theorem infinite_non_fractional_numbers : Set.Infinite NonFractionalSet := by
  sorry

end NUMINAMATH_CALUDE_infinite_non_fractional_numbers_l266_26621


namespace NUMINAMATH_CALUDE_cubic_root_identity_l266_26680

theorem cubic_root_identity (a b c : ℂ) (n m : ℕ) :
  (∃ x : ℂ, x^3 = 1 ∧ a * x^(3*n + 2) + b * x^(3*m + 1) + c = 0) →
  a^3 + b^3 + c^3 - 3*a*b*c = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_identity_l266_26680


namespace NUMINAMATH_CALUDE_min_n_for_constant_term_l266_26622

theorem min_n_for_constant_term (n : ℕ) : 
  (∃ k : ℕ, 2 * n = 5 * k) ∧ (∀ m : ℕ, m < n → ¬∃ k : ℕ, 2 * m = 5 * k) ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_n_for_constant_term_l266_26622


namespace NUMINAMATH_CALUDE_disjunction_true_l266_26635

theorem disjunction_true : 
  (∃ α : ℝ, Real.cos (π - α) = Real.cos α) ∨ 
  (∀ x : ℝ, x^2 + 1 > 0) := by
sorry

end NUMINAMATH_CALUDE_disjunction_true_l266_26635


namespace NUMINAMATH_CALUDE_root_product_equals_two_l266_26602

theorem root_product_equals_two : 
  ∃ (x₁ x₂ x₃ : ℝ), 
    (Real.sqrt 4050 * x₁^3 - 8101 * x₁^2 + 4 = 0) ∧
    (Real.sqrt 4050 * x₂^3 - 8101 * x₂^2 + 4 = 0) ∧
    (Real.sqrt 4050 * x₃^3 - 8101 * x₃^2 + 4 = 0) ∧
    (x₁ < x₂) ∧ (x₂ < x₃) ∧
    (x₂ * (x₁ + x₃) = 2) := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_two_l266_26602


namespace NUMINAMATH_CALUDE_box_height_is_nine_l266_26640

/-- A rectangular box containing spheres -/
structure SphereBox where
  height : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ
  large_sphere_count : ℕ
  small_sphere_count : ℕ

/-- The specific box described in the problem -/
def problem_box : SphereBox :=
  { height := 9,
    large_sphere_radius := 3,
    small_sphere_radius := 1.5,
    large_sphere_count := 1,
    small_sphere_count := 8 }

/-- Theorem stating that the height of the box must be 9 -/
theorem box_height_is_nine (box : SphereBox) :
  box.height = 9 ∧
  box.large_sphere_radius = 3 ∧
  box.small_sphere_radius = 1.5 ∧
  box.large_sphere_count = 1 ∧
  box.small_sphere_count = 8 →
  box = problem_box :=
sorry

end NUMINAMATH_CALUDE_box_height_is_nine_l266_26640


namespace NUMINAMATH_CALUDE_hexomino_min_containing_rectangle_area_l266_26647

/-- A hexomino is a polyomino of 6 connected unit squares. -/
def Hexomino : Type := Unit  -- Placeholder definition

/-- The minimum area of a rectangle that contains a given hexomino. -/
def minContainingRectangleArea (h : Hexomino) : ℝ := sorry

/-- Theorem: The minimum area of any rectangle containing a hexomino is 21/2. -/
theorem hexomino_min_containing_rectangle_area (h : Hexomino) :
  minContainingRectangleArea h = 21 / 2 := by sorry

end NUMINAMATH_CALUDE_hexomino_min_containing_rectangle_area_l266_26647


namespace NUMINAMATH_CALUDE_duke_of_york_men_percentage_l266_26638

/-- The percentage of men remaining after two consecutive losses -/
theorem duke_of_york_men_percentage : 
  let initial_men : ℕ := 10000
  let first_loss_rate : ℚ := 1/10
  let second_loss_rate : ℚ := 3/20
  let remaining_men : ℚ := initial_men * (1 - first_loss_rate) * (1 - second_loss_rate)
  let percentage_remaining : ℚ := remaining_men / initial_men * 100
  percentage_remaining = 76.5 := by
  sorry

end NUMINAMATH_CALUDE_duke_of_york_men_percentage_l266_26638


namespace NUMINAMATH_CALUDE_actual_distance_traveled_prove_actual_distance_l266_26655

/-- The actual distance traveled by a person, given two different walking speeds and a distance difference. -/
theorem actual_distance_traveled (speed1 speed2 distance_diff : ℝ) (h1 : speed1 > 0) (h2 : speed2 > 0) 
  (h3 : speed2 > speed1) (h4 : distance_diff > 0) : ℝ :=
  let time := distance_diff / (speed2 - speed1)
  let actual_distance := speed1 * time
  actual_distance

/-- Proves that the actual distance traveled is 20 km under the given conditions. -/
theorem prove_actual_distance : 
  actual_distance_traveled 10 20 20 (by norm_num) (by norm_num) (by norm_num) (by norm_num) = 20 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_prove_actual_distance_l266_26655


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l266_26685

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x > 2}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l266_26685


namespace NUMINAMATH_CALUDE_no_natural_divisible_by_49_l266_26648

theorem no_natural_divisible_by_49 : ∀ n : ℕ, ¬(49 ∣ (n^2 + 5*n + 1)) := by sorry

end NUMINAMATH_CALUDE_no_natural_divisible_by_49_l266_26648


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l266_26659

theorem fraction_sum_equality : (3 : ℚ) / 8 + 9 / 12 - 1 / 6 = 23 / 24 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l266_26659


namespace NUMINAMATH_CALUDE_one_sport_count_l266_26628

/-- The number of members who play only one sport (badminton, tennis, or basketball) -/
def members_one_sport (total members badminton tennis basketball badminton_tennis badminton_basketball tennis_basketball all_three none : ℕ) : ℕ :=
  let badminton_only := badminton - badminton_tennis - badminton_basketball + all_three
  let tennis_only := tennis - badminton_tennis - tennis_basketball + all_three
  let basketball_only := basketball - badminton_basketball - tennis_basketball + all_three
  badminton_only + tennis_only + basketball_only

theorem one_sport_count :
  members_one_sport 150 65 80 60 20 15 25 10 12 = 115 := by
  sorry

end NUMINAMATH_CALUDE_one_sport_count_l266_26628


namespace NUMINAMATH_CALUDE_sum_of_ages_in_two_years_l266_26642

def Matt_age (Fem_age : ℕ) : ℕ := 4 * Fem_age

def current_Fem_age : ℕ := 11

def future_age (current_age : ℕ) : ℕ := current_age + 2

theorem sum_of_ages_in_two_years :
  future_age (Matt_age current_Fem_age) + future_age current_Fem_age = 59 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_in_two_years_l266_26642


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l266_26601

theorem quadratic_root_difference (a b c d e : ℝ) :
  (2 * a^2 - 5 * a + 2 = 3 * a + 24) →
  ∃ x y : ℝ, (x ≠ y) ∧ 
             (2 * x^2 - 5 * x + 2 = 3 * x + 24) ∧ 
             (2 * y^2 - 5 * y + 2 = 3 * y + 24) ∧ 
             (abs (x - y) = 2 * Real.sqrt 15) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l266_26601


namespace NUMINAMATH_CALUDE_clean_80_cars_per_day_l266_26691

/-- Represents the Super Clean Car Wash Company's operations -/
structure CarWash where
  price_per_car : ℕ  -- Price per car in dollars
  total_revenue : ℕ  -- Total revenue in dollars
  num_days : ℕ       -- Number of days

/-- Calculates the number of cars cleaned per day -/
def cars_per_day (cw : CarWash) : ℕ :=
  (cw.total_revenue / cw.price_per_car) / cw.num_days

/-- Theorem stating that the number of cars cleaned per day is 80 -/
theorem clean_80_cars_per_day (cw : CarWash)
  (h1 : cw.price_per_car = 5)
  (h2 : cw.total_revenue = 2000)
  (h3 : cw.num_days = 5) :
  cars_per_day cw = 80 := by
  sorry

#eval cars_per_day ⟨5, 2000, 5⟩

end NUMINAMATH_CALUDE_clean_80_cars_per_day_l266_26691


namespace NUMINAMATH_CALUDE_identity_function_divisibility_l266_26666

theorem identity_function_divisibility (f : ℕ → ℕ) :
  (∀ m n : ℕ, (f m + f n) ∣ (m + n)) →
  ∀ m : ℕ, f m = m :=
by sorry

end NUMINAMATH_CALUDE_identity_function_divisibility_l266_26666


namespace NUMINAMATH_CALUDE_ratio_antecedent_l266_26692

theorem ratio_antecedent (ratio_a ratio_b consequent : ℚ) : 
  ratio_a / ratio_b = 4 / 6 →
  consequent = 45 →
  ratio_a / ratio_b = ratio_a / consequent →
  ratio_a = 30 := by
  sorry

end NUMINAMATH_CALUDE_ratio_antecedent_l266_26692


namespace NUMINAMATH_CALUDE_smallest_multiple_of_7_4_5_l266_26658

theorem smallest_multiple_of_7_4_5 : ∃ n : ℕ+, (∀ m : ℕ+, m.val % 7 = 0 ∧ m.val % 4 = 0 ∧ m.val % 5 = 0 → n ≤ m) ∧ n.val % 7 = 0 ∧ n.val % 4 = 0 ∧ n.val % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_7_4_5_l266_26658


namespace NUMINAMATH_CALUDE_fraction_problem_l266_26631

theorem fraction_problem (a b : ℤ) (ha : a > 0) (hb : b > 0) :
  (a : ℚ) / (b + 6) = 1 / 6 ∧ (a + 4 : ℚ) / b = 1 / 4 →
  (a : ℚ) / b = 11 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l266_26631


namespace NUMINAMATH_CALUDE_tire_cost_l266_26630

theorem tire_cost (total_cost : ℝ) (num_tires : ℕ) (cost_per_tire : ℝ) :
  total_cost = 4 →
  num_tires = 8 →
  cost_per_tire = total_cost / num_tires →
  cost_per_tire = 0.50 := by
sorry

end NUMINAMATH_CALUDE_tire_cost_l266_26630


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l266_26690

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 19*x - 48 = 0 → x ≤ 24 :=
by
  sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l266_26690


namespace NUMINAMATH_CALUDE_final_block_count_l266_26683

theorem final_block_count :
  let initial_blocks : ℕ := 250
  let added_blocks : ℕ := 13
  let intermediate_blocks : ℕ := initial_blocks + added_blocks
  let doubling_factor : ℕ := 2
  let final_blocks : ℕ := intermediate_blocks * doubling_factor
  final_blocks = 526 := by sorry

end NUMINAMATH_CALUDE_final_block_count_l266_26683


namespace NUMINAMATH_CALUDE_brownies_needed_l266_26603

/-- Represents the amount of frosting used for different baked goods -/
structure FrostingUsage where
  layerCake : ℝ
  singleCake : ℝ
  panBrownies : ℝ
  dozenCupcakes : ℝ

/-- Represents the quantities of baked goods Paul needs to prepare -/
structure BakedGoods where
  layerCakes : ℕ
  singleCakes : ℕ
  dozenCupcakes : ℕ

def totalFrostingNeeded : ℝ := 21

theorem brownies_needed (usage : FrostingUsage) (goods : BakedGoods) 
  (h1 : usage.layerCake = 1)
  (h2 : usage.singleCake = 0.5)
  (h3 : usage.panBrownies = 0.5)
  (h4 : usage.dozenCupcakes = 0.5)
  (h5 : goods.layerCakes = 3)
  (h6 : goods.singleCakes = 12)
  (h7 : goods.dozenCupcakes = 6) :
  (totalFrostingNeeded - 
   (goods.layerCakes * usage.layerCake + 
    goods.singleCakes * usage.singleCake + 
    goods.dozenCupcakes * usage.dozenCupcakes)) / usage.panBrownies = 18 := by
  sorry

end NUMINAMATH_CALUDE_brownies_needed_l266_26603


namespace NUMINAMATH_CALUDE_triangle_two_solutions_l266_26667

theorem triangle_two_solutions (a b : ℝ) (A B : ℝ) :
  b = 2 →
  B = π / 4 →
  (∃ (C : ℝ), 0 < C ∧ C < π ∧ A + B + C = π ∧ a / Real.sin A = b / Real.sin B) →
  (∃ (C' : ℝ), 0 < C' ∧ C' < π ∧ C' ≠ C ∧ A + B + C' = π ∧ a / Real.sin A = b / Real.sin B) →
  2 < a ∧ a < 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_l266_26667


namespace NUMINAMATH_CALUDE_rectangle_area_l266_26663

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 246) : L * B = 3650 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l266_26663


namespace NUMINAMATH_CALUDE_train_speed_problem_l266_26634

theorem train_speed_problem (length1 length2 speed1 time : ℝ) 
  (h1 : length1 = 145)
  (h2 : length2 = 165)
  (h3 : speed1 = 60)
  (h4 : time = 8)
  (h5 : speed1 > 0) :
  let total_length := length1 + length2
  let relative_speed := total_length / time
  let speed2 := relative_speed - speed1
  speed2 = 79.5 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l266_26634


namespace NUMINAMATH_CALUDE_infinite_centers_of_symmetry_l266_26697

/-- A type representing a geometric figure. -/
structure Figure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A type representing a point in the figure. -/
structure Point where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a symmetry operation on a figure. -/
def SymmetryOperation : Type := Figure → Figure

/-- Represents a center of symmetry for a figure. -/
def CenterOfSymmetry (f : Figure) : Type := Point

/-- Composition of symmetry operations. -/
def composeSymmetry (s1 s2 : SymmetryOperation) : SymmetryOperation :=
  fun f => s1 (s2 f)

/-- 
  If a figure has more than one center of symmetry, 
  it must have infinitely many centers of symmetry.
-/
theorem infinite_centers_of_symmetry (f : Figure) :
  (∃ (c1 c2 : CenterOfSymmetry f), c1 ≠ c2) →
  ∀ n : ℕ, ∃ (centers : Finset (CenterOfSymmetry f)), centers.card > n :=
sorry

end NUMINAMATH_CALUDE_infinite_centers_of_symmetry_l266_26697


namespace NUMINAMATH_CALUDE_parallel_transitivity_l266_26609

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define the parallel relation between lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l266_26609


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l266_26668

/-- Calculate the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The problem statement -/
theorem ball_bounce_distance :
  let initialHeight : ℝ := 150
  let reboundFactor : ℝ := 3/4
  let bounces : ℕ := 5
  totalDistance initialHeight reboundFactor bounces = 765.703125 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l266_26668


namespace NUMINAMATH_CALUDE_william_arrival_time_l266_26639

/-- Represents time as hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : minutes < 60

/-- Adds hours and minutes to a given time -/
def addTime (t : Time) (h : ℕ) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + h * 60 + m
  let newHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  ⟨newHours % 24, newMinutes, by sorry⟩

/-- Calculates arrival time given departure time, total time on road, and total stop time -/
def calculateArrivalTime (departureTime : Time) (totalTimeOnRoad : ℕ) (totalStopTime : ℕ) : Time :=
  let actualDrivingTime := totalTimeOnRoad - (totalStopTime / 60)
  addTime departureTime actualDrivingTime 0

theorem william_arrival_time :
  let departureTime : Time := ⟨7, 0, by sorry⟩
  let totalTimeOnRoad : ℕ := 12
  let stopTimes : List ℕ := [25, 10, 25]
  let totalStopTime : ℕ := stopTimes.sum
  let arrivalTime := calculateArrivalTime departureTime totalTimeOnRoad totalStopTime
  arrivalTime = ⟨18, 0, by sorry⟩ := by sorry

end NUMINAMATH_CALUDE_william_arrival_time_l266_26639


namespace NUMINAMATH_CALUDE_king_can_equalize_l266_26616

/-- Represents a chessboard square --/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the state of the chessboard --/
def Chessboard := Square → ℤ

/-- Represents a sequence of king's moves --/
def KingPath := List Square

/-- Checks if a move between two squares is valid for a king --/
def isValidKingMove (s1 s2 : Square) : Prop :=
  (abs (s1.row - s2.row) ≤ 1) ∧ (abs (s1.col - s2.col) ≤ 1)

/-- Applies a sequence of king's moves to a chessboard --/
def applyMoves (board : Chessboard) (path : KingPath) : Chessboard :=
  sorry

/-- The main theorem --/
theorem king_can_equalize (initial : Chessboard) :
  ∃ (path : KingPath), ∀ (s1 s2 : Square), (applyMoves initial path s1) = (applyMoves initial path s2) :=
sorry

end NUMINAMATH_CALUDE_king_can_equalize_l266_26616


namespace NUMINAMATH_CALUDE_sequence_sum_l266_26629

theorem sequence_sum (a : ℕ → ℕ) : 
  (a 1 = 1) → 
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2^n) → 
  a 10 = 1023 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l266_26629


namespace NUMINAMATH_CALUDE_inverse_one_implies_one_l266_26653

theorem inverse_one_implies_one (a : ℝ) (h : a ≠ 0) : a⁻¹ = (-1)^0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_one_implies_one_l266_26653


namespace NUMINAMATH_CALUDE_fraction_subtraction_l266_26625

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l266_26625


namespace NUMINAMATH_CALUDE_machinery_expenditure_l266_26688

theorem machinery_expenditure (C : ℝ) (h : C > 0) : 
  let raw_material_cost : ℝ := (1 / 4) * C
  let remaining_after_raw : ℝ := C - raw_material_cost
  let final_remaining : ℝ := 0.675 * C
  let machinery_cost : ℝ := remaining_after_raw - final_remaining
  machinery_cost / remaining_after_raw = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_machinery_expenditure_l266_26688


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l266_26694

/-- Given plane vectors a, b, and c, if a + 2b is parallel to c, then the x-coordinate of c is -2. -/
theorem vector_parallel_condition (a b c : ℝ × ℝ) :
  a = (3, 1) →
  b = (-1, 1) →
  c.2 = -6 →
  (∃ (k : ℝ), k • (a + 2 • b) = c) →
  c.1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l266_26694


namespace NUMINAMATH_CALUDE_hypotenuse_length_l266_26671

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  -- Side lengths
  a : ℝ  -- Length of the side opposite to the 30° angle
  b : ℝ  -- Length of the side opposite to the 60° angle
  c : ℝ  -- Length of the hypotenuse (opposite to the 90° angle)
  -- Properties of a 30-60-90 triangle
  h1 : a = c / 2
  h2 : b = a * Real.sqrt 3

/-- Theorem: In a 30-60-90 triangle with side length opposite to 60° angle equal to 12, 
    the length of the hypotenuse is 8√3 -/
theorem hypotenuse_length (t : Triangle30_60_90) (h : t.b = 12) : t.c = 8 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_hypotenuse_length_l266_26671


namespace NUMINAMATH_CALUDE_only_prime_perfect_square_l266_26681

theorem only_prime_perfect_square : 
  ∀ p : ℕ, Prime p → (∃ k : ℕ, 5^p + 12^p = k^2) → p = 2 :=
by sorry

end NUMINAMATH_CALUDE_only_prime_perfect_square_l266_26681


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_49_l266_26626

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials_49 :
  units_digit (sum_factorials 49) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_49_l266_26626


namespace NUMINAMATH_CALUDE_circle_intersection_range_l266_26649

/-- The range of m for which two circles intersect -/
theorem circle_intersection_range :
  let circle1 : ℝ → ℝ → ℝ → Prop := λ x y m ↦ x^2 + y^2 = m
  let circle2 : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 - 6*x + 8*y - 24 = 0
  ∀ m : ℝ, (∃ x y : ℝ, circle1 x y m ∧ circle2 x y) ↔ 4 < m ∧ m < 144 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l266_26649


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l266_26675

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l266_26675


namespace NUMINAMATH_CALUDE_g_minimum_value_l266_26641

open Real

noncomputable def g (x : ℝ) : ℝ :=
  x + (2*x)/(x^2 + 1) + (x*(x + 3))/(x^2 + 3) + (3*(x + 1))/(x*(x^2 + 3))

theorem g_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_g_minimum_value_l266_26641


namespace NUMINAMATH_CALUDE_angle_FAH_is_45_degrees_l266_26670

/-- Given a unit square ABCD with EF parallel to AB, GH parallel to BC, BF = 1/4, and BF + DH = FH, 
    the measure of angle FAH is 45 degrees. -/
theorem angle_FAH_is_45_degrees (A B C D E F G H : ℝ × ℝ) : 
  -- Unit square ABCD
  A = (0, 1) ∧ B = (0, 0) ∧ C = (1, 0) ∧ D = (1, 1) →
  -- EF is parallel to AB
  (E.2 - F.2) / (E.1 - F.1) = (A.2 - B.2) / (A.1 - B.1) →
  -- GH is parallel to BC
  (G.2 - H.2) / (G.1 - H.1) = (B.2 - C.2) / (B.1 - C.1) →
  -- BF = 1/4
  F = (1/4, 0) →
  -- BF + DH = FH
  Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2) + 
  Real.sqrt ((D.1 - H.1)^2 + (D.2 - H.2)^2) = 
  Real.sqrt ((F.1 - H.1)^2 + (F.2 - H.2)^2) →
  -- Angle FAH is 45 degrees
  Real.arctan (((A.2 - F.2) / (A.1 - F.1) - (A.2 - H.2) / (A.1 - H.1)) / 
    (1 + (A.2 - F.2) / (A.1 - F.1) * (A.2 - H.2) / (A.1 - H.1))) * (180 / Real.pi) = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_FAH_is_45_degrees_l266_26670


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_T_l266_26633

def K : ℚ := (1:ℚ)/1 + (1:ℚ)/3 + (1:ℚ)/5 + (1:ℚ)/7 + (1:ℚ)/9

def T (n : ℕ) : ℚ := n * (5^(n-1) : ℚ) * K

def is_integer (q : ℚ) : Prop := ∃ (z : ℤ), q = z

theorem smallest_n_for_integer_T :
  ∀ n : ℕ, (n > 0 ∧ is_integer (T n)) → n ≥ 63 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_T_l266_26633


namespace NUMINAMATH_CALUDE_combined_selling_price_l266_26699

/-- Calculate the combined selling price of three articles given their cost prices and profit/loss percentages. -/
theorem combined_selling_price (cost1 cost2 cost3 : ℝ) : 
  cost1 = 70 →
  cost2 = 120 →
  cost3 = 150 →
  ∃ (sell1 sell2 sell3 : ℝ),
    (2/3 * sell1 = 0.85 * cost1) ∧
    (sell2 = cost2 * 1.3) ∧
    (sell3 = cost3 * 0.8) ∧
    (sell1 + sell2 + sell3 = 365.25) := by
  sorry

#check combined_selling_price

end NUMINAMATH_CALUDE_combined_selling_price_l266_26699


namespace NUMINAMATH_CALUDE_optical_mice_ratio_l266_26664

theorem optical_mice_ratio (total_mice : ℕ) (trackball_mice : ℕ) : 
  total_mice = 80 →
  trackball_mice = 20 →
  (total_mice / 2 : ℚ) = total_mice / 2 →
  (total_mice - total_mice / 2 - trackball_mice : ℚ) / total_mice = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_optical_mice_ratio_l266_26664


namespace NUMINAMATH_CALUDE_no_real_roots_iff_k_less_than_negative_one_l266_26610

theorem no_real_roots_iff_k_less_than_negative_one (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_k_less_than_negative_one_l266_26610


namespace NUMINAMATH_CALUDE_max_four_digit_prime_product_l266_26606

def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem max_four_digit_prime_product :
  ∃ (m x y z : Nat),
    isPrime x ∧ isPrime y ∧ isPrime z ∧
    x < 10 ∧ y < 10 ∧ z < 10 ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    isPrime (10 * x + y) ∧
    isPrime (10 * z + x) ∧
    m = x * y * (10 * x + y) ∧
    m ≥ 1000 ∧ m < 10000 ∧
    (∀ (m' x' y' z' : Nat),
      isPrime x' ∧ isPrime y' ∧ isPrime z' ∧
      x' < 10 ∧ y' < 10 ∧ z' < 10 ∧
      x' ≠ y' ∧ y' ≠ z' ∧ x' ≠ z' ∧
      isPrime (10 * x' + y') ∧
      isPrime (10 * z' + x') ∧
      m' = x' * y' * (10 * x' + y') ∧
      m' ≥ 1000 ∧ m' < 10000 →
      m' ≤ m) ∧
    m = 1533 :=
by sorry

end NUMINAMATH_CALUDE_max_four_digit_prime_product_l266_26606


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l266_26620

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l266_26620


namespace NUMINAMATH_CALUDE_pie_slices_left_l266_26656

theorem pie_slices_left (total_slices : ℕ) (half_given : ℚ) (quarter_given : ℚ) : 
  total_slices = 8 → half_given = 1/2 → quarter_given = 1/4 → 
  total_slices - (half_given * total_slices + quarter_given * total_slices) = 2 := by
sorry

end NUMINAMATH_CALUDE_pie_slices_left_l266_26656


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l266_26650

theorem shaded_area_theorem (r R : ℝ) (h1 : R > 0) (h2 : r > 0) : 
  (π * R^2 = 100 * π) → (r = R / 2) → 
  (π * R^2 / 2 + π * r^2 / 4 = 31.25 * π) := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l266_26650


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l266_26657

/-- Given a 9x16 rectangle that is cut into two congruent quadrilaterals
    which can be repositioned to form a square, the side length z of
    one quadrilateral is 12. -/
theorem quadrilateral_side_length (z : ℝ) : z = 12 :=
  let rectangle_area : ℝ := 9 * 16
  let square_side : ℝ := Real.sqrt rectangle_area
  sorry


end NUMINAMATH_CALUDE_quadrilateral_side_length_l266_26657


namespace NUMINAMATH_CALUDE_positive_A_value_l266_26604

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l266_26604


namespace NUMINAMATH_CALUDE_fraction_equality_l266_26637

theorem fraction_equality (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 1 / 3) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l266_26637


namespace NUMINAMATH_CALUDE_osmanthus_price_is_300_l266_26646

/-- The unit price of osmanthus trees given the following conditions:
  - Total amount raised is 7000 yuan
  - Total number of trees is 30
  - Cost of osmanthus trees is 3000 yuan
  - Unit price of osmanthus trees is 50% higher than cherry trees
-/
def osmanthus_price : ℝ :=
  let total_amount : ℝ := 7000
  let total_trees : ℝ := 30
  let osmanthus_cost : ℝ := 3000
  let price_ratio : ℝ := 1.5
  300

theorem osmanthus_price_is_300 :
  osmanthus_price = 300 := by sorry

end NUMINAMATH_CALUDE_osmanthus_price_is_300_l266_26646


namespace NUMINAMATH_CALUDE_inscribed_circle_l266_26632

-- Define the triangle vertices
def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (2, 5)
def C : ℝ × ℝ := (5, 2)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 2

-- State the theorem
theorem inscribed_circle :
  ∃ (x y : ℝ), circle_equation x y ∧
  (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    ((x - A.1)^2 + (y - A.2)^2 = (t * (B.1 - A.1))^2 + (t * (B.2 - A.2))^2) ∧
    ((x - B.1)^2 + (y - B.2)^2 = (t * (C.1 - B.1))^2 + (t * (C.2 - B.2))^2) ∧
    ((x - C.1)^2 + (y - C.2)^2 = (t * (A.1 - C.1))^2 + (t * (A.2 - C.2))^2)) :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_l266_26632


namespace NUMINAMATH_CALUDE_solution_set_characterization_l266_26617

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem solution_set_characterization (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_nonneg : ∀ x ≥ 0, f x = x^3 - 8) :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l266_26617


namespace NUMINAMATH_CALUDE_rent_calculation_l266_26644

def problem (salary : ℕ) (milk groceries education petrol misc rent : ℕ) : Prop :=
  let savings := salary / 10
  let other_expenses := milk + groceries + education + petrol + misc
  salary = savings + rent + other_expenses ∧
  milk = 1500 ∧
  groceries = 4500 ∧
  education = 2500 ∧
  petrol = 2000 ∧
  misc = 2500 ∧
  savings = 2000

theorem rent_calculation :
  ∀ salary milk groceries education petrol misc rent,
    problem salary milk groceries education petrol misc rent →
    rent = 5000 := by
  sorry

end NUMINAMATH_CALUDE_rent_calculation_l266_26644


namespace NUMINAMATH_CALUDE_shelbys_journey_l266_26693

/-- Shelby's scooter journey with varying weather conditions -/
theorem shelbys_journey 
  (speed_sunny : ℝ) 
  (speed_rainy : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (break_time : ℝ)
  (h1 : speed_sunny = 40)
  (h2 : speed_rainy = 15)
  (h3 : total_distance = 20)
  (h4 : total_time = 50)
  (h5 : break_time = 5) :
  ∃ (rainy_time : ℝ),
    rainy_time = 24 ∧
    speed_sunny * (total_time - rainy_time - break_time) / 60 + 
    speed_rainy * rainy_time / 60 = total_distance :=
by sorry

end NUMINAMATH_CALUDE_shelbys_journey_l266_26693


namespace NUMINAMATH_CALUDE_digit_456_is_8_l266_26651

/-- The decimal representation of 17/59 has a repeating cycle of 29 digits -/
def decimal_cycle : List Nat := [2, 8, 8, 1, 3, 5, 5, 9, 3, 2, 2, 0, 3, 3, 8, 9, 8, 3, 0, 5, 0, 8, 4, 7, 4, 5, 7, 6, 2, 7, 1, 1]

/-- The length of the repeating cycle in the decimal representation of 17/59 -/
def cycle_length : Nat := 29

/-- The 456th digit after the decimal point in the representation of 17/59 -/
def digit_456 : Nat := decimal_cycle[(456 % cycle_length) - 1]

theorem digit_456_is_8 : digit_456 = 8 := by sorry

end NUMINAMATH_CALUDE_digit_456_is_8_l266_26651


namespace NUMINAMATH_CALUDE_initial_data_points_l266_26624

theorem initial_data_points (x : ℝ) : 
  (1.20 * x - 0.25 * (1.20 * x) = 180) → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_initial_data_points_l266_26624


namespace NUMINAMATH_CALUDE_two_intersecting_lines_determine_plane_l266_26682

-- Define the basic types
def Point : Type := sorry
def Line : Type := sorry
def Plane : Type := sorry

-- Define the axioms of solid geometry (focusing on Axiom 3)
axiom intersecting_lines (l1 l2 : Line) : Prop
axiom determine_plane (l1 l2 : Line) (p : Plane) : Prop

-- Axiom 3: Two intersecting lines determine a plane
axiom axiom_3 (l1 l2 : Line) (p : Plane) : 
  intersecting_lines l1 l2 → determine_plane l1 l2 p

-- Theorem to prove
theorem two_intersecting_lines_determine_plane (l1 l2 : Line) :
  intersecting_lines l1 l2 → ∃ p : Plane, determine_plane l1 l2 p :=
sorry

end NUMINAMATH_CALUDE_two_intersecting_lines_determine_plane_l266_26682


namespace NUMINAMATH_CALUDE_animal_ages_sum_l266_26660

/-- Represents the ages of the animals in the problem -/
structure AnimalAges where
  porcupine : ℕ
  owl : ℕ
  lion : ℕ

/-- Defines the conditions given in the problem -/
def valid_ages (ages : AnimalAges) : Prop :=
  ages.owl = 2 * ages.porcupine ∧
  ages.owl = ages.lion + 2 ∧
  ages.lion = ages.porcupine + 4

/-- The theorem to be proved -/
theorem animal_ages_sum (ages : AnimalAges) :
  valid_ages ages → ages.porcupine + ages.owl + ages.lion = 28 := by
  sorry


end NUMINAMATH_CALUDE_animal_ages_sum_l266_26660


namespace NUMINAMATH_CALUDE_max_x_value_l266_26643

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 5) 
  (prod_sum_eq : x*y + x*z + y*z = 8) : 
  x ≤ 7/3 :=
sorry

end NUMINAMATH_CALUDE_max_x_value_l266_26643


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l266_26698

theorem lcm_gcf_problem (n m : ℕ) (h1 : Nat.lcm n m = 48) (h2 : Nat.gcd n m = 18) (h3 : m = 16) : n = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l266_26698


namespace NUMINAMATH_CALUDE_monday_distance_l266_26673

/-- Debby's jogging distances over three days -/
structure JoggingDistances where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  total : ℝ

/-- The jogging distances satisfy the given conditions -/
def satisfies_conditions (d : JoggingDistances) : Prop :=
  d.tuesday = 5 ∧ d.wednesday = 9 ∧ d.total = 16 ∧ d.monday + d.tuesday + d.wednesday = d.total

/-- Theorem: Debby jogged 2 kilometers on Monday -/
theorem monday_distance (d : JoggingDistances) (h : satisfies_conditions d) : d.monday = 2 := by
  sorry

end NUMINAMATH_CALUDE_monday_distance_l266_26673


namespace NUMINAMATH_CALUDE_max_quad_area_l266_26618

/-- The ellipse defined by x²/8 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- The foci of the ellipse -/
def foci : ℝ × ℝ × ℝ × ℝ := sorry

/-- A point on the ellipse -/
def point_on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

/-- Segment AB passes through the center of the ellipse -/
def segment_through_center (a b : ℝ × ℝ) : Prop := sorry

/-- The area of quadrilateral F₁AF₂B -/
def quad_area (a b : ℝ × ℝ) : ℝ := sorry

theorem max_quad_area :
  ∀ (a b : ℝ × ℝ),
    point_on_ellipse a →
    point_on_ellipse b →
    segment_through_center a b →
    quad_area a b ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_quad_area_l266_26618


namespace NUMINAMATH_CALUDE_expression_equality_l266_26672

theorem expression_equality : 4 + 3/10 + 9/1000 = 4.309 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l266_26672


namespace NUMINAMATH_CALUDE_inequality_solution_l266_26665

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (4 * x + 6) ≤ 5 ↔ x < -3/2 ∨ x > -1/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l266_26665


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_cube_l266_26684

theorem smallest_multiplier_for_cube (n : ℕ) : 
  (∀ m : ℕ, m < 300 → ¬∃ k : ℕ, 720 * m = k^3) ∧ 
  (∃ k : ℕ, 720 * 300 = k^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_cube_l266_26684


namespace NUMINAMATH_CALUDE_specific_plate_probability_l266_26615

/-- Represents the set of vowels used in license plates -/
def Vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

/-- Represents the set of non-vowel letters used in license plates -/
def NonVowels : Finset Char := {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'}

/-- Represents the set of even digits used in license plates -/
def EvenDigits : Finset Char := {'0', '2', '4', '6', '8'}

/-- Represents a license plate -/
structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Char
  fifth : Char
  h1 : first ∈ Vowels
  h2 : second ∈ Vowels
  h3 : first ≠ second
  h4 : third ∈ NonVowels
  h5 : fourth ∈ NonVowels
  h6 : third ≠ fourth
  h7 : fifth ∈ EvenDigits

/-- The probability of a specific license plate occurring -/
def licensePlateProbability (plate : LicensePlate) : ℚ :=
  1 / (Vowels.card * (Vowels.card - 1) * NonVowels.card * (NonVowels.card - 1) * EvenDigits.card)

theorem specific_plate_probability :
  ∃ (plate : LicensePlate), licensePlateProbability plate = 1 / 50600 :=
sorry

end NUMINAMATH_CALUDE_specific_plate_probability_l266_26615


namespace NUMINAMATH_CALUDE_triangular_pyramid_theorem_l266_26636

/-- A triangular pyramid with face areas S₁, S₂, S₃, S₄, distances H₁, H₂, H₃, H₄ 
    from any internal point to the faces, volume V, and constant k. -/
structure TriangularPyramid where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  S₄ : ℝ
  H₁ : ℝ
  H₂ : ℝ
  H₃ : ℝ
  H₄ : ℝ
  V : ℝ
  k : ℝ
  h_positive : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 ∧ H₁ > 0 ∧ H₂ > 0 ∧ H₃ > 0 ∧ H₄ > 0 ∧ V > 0 ∧ k > 0
  h_ratio : S₁ / 1 = S₂ / 2 ∧ S₂ / 2 = S₃ / 3 ∧ S₃ / 3 = S₄ / 4 ∧ S₄ / 4 = k

/-- The theorem to be proved -/
theorem triangular_pyramid_theorem (p : TriangularPyramid) : 
  1 * p.H₁ + 2 * p.H₂ + 3 * p.H₃ + 4 * p.H₄ = 3 * p.V / p.k := by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_theorem_l266_26636


namespace NUMINAMATH_CALUDE_initial_cows_l266_26654

theorem initial_cows (cows dogs : ℕ) : 
  cows = 2 * dogs →
  (3 / 4 : ℚ) * cows + (1 / 4 : ℚ) * dogs = 161 →
  cows = 184 := by
  sorry

end NUMINAMATH_CALUDE_initial_cows_l266_26654


namespace NUMINAMATH_CALUDE_solution_is_two_l266_26676

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (y : ℝ) : Prop :=
  lg (y - 1) - lg y = lg (2 * y - 2) - lg (y + 2)

-- Theorem statement
theorem solution_is_two :
  ∃ y : ℝ, y > 1 ∧ y + 2 > 0 ∧ equation y ∧ y = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_is_two_l266_26676


namespace NUMINAMATH_CALUDE_tangent_squares_roots_l266_26611

theorem tangent_squares_roots : ∃ (a b c : ℝ),
  a + b + c = 33 ∧
  a * b * c = 33 ∧
  a * b + b * c + c * a = 27 ∧
  ∀ (x : ℝ), x^3 - 33*x^2 + 27*x - 33 = 0 ↔ (x = a ∨ x = b ∨ x = c) := by
  sorry

end NUMINAMATH_CALUDE_tangent_squares_roots_l266_26611


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l266_26678

theorem rectangle_side_lengths :
  ∀ x y : ℝ,
  x > 0 →
  y > 0 →
  y = 2 * x →
  x * y = 2 * (x + y) →
  (x = 3 ∧ y = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l266_26678


namespace NUMINAMATH_CALUDE_f_of_3_eq_19_l266_26612

/-- The function f(x) = 2x^2 + 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- Theorem: f(3) = 19 -/
theorem f_of_3_eq_19 : f 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_eq_19_l266_26612


namespace NUMINAMATH_CALUDE_projection_matrix_values_l266_26687

def isProjectionMatrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 20/36; c, 16/36]

theorem projection_matrix_values :
  ∀ a c : ℚ, isProjectionMatrix (P a c) → a = 1/27 ∧ c = 5/27 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l266_26687


namespace NUMINAMATH_CALUDE_fiftieth_ring_squares_l266_26645

/-- Represents the number of unit squares in the nth ring of the described square array. -/
def ring_squares (n : ℕ) : ℕ := 32 * n - 16

/-- The theorem states that the 50th ring contains 1584 unit squares. -/
theorem fiftieth_ring_squares : ring_squares 50 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_ring_squares_l266_26645


namespace NUMINAMATH_CALUDE_only_B_on_line_l266_26613

-- Define the points
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (-2, 1)
def C : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (2, -9)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - 3 * y + 7 = 0

-- Theorem statement
theorem only_B_on_line :
  line_equation B.1 B.2 ∧
  ¬line_equation A.1 A.2 ∧
  ¬line_equation C.1 C.2 ∧
  ¬line_equation D.1 D.2 := by
  sorry

end NUMINAMATH_CALUDE_only_B_on_line_l266_26613


namespace NUMINAMATH_CALUDE_function_difference_bound_l266_26607

theorem function_difference_bound 
  (f : Set.Icc 0 1 → ℝ) 
  (h1 : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h2 : ∀ (x y : Set.Icc 0 1), x ≠ y → |f x - f y| < |x.val - y.val|) :
  ∀ (x y : Set.Icc 0 1), |f x - f y| < (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_function_difference_bound_l266_26607


namespace NUMINAMATH_CALUDE_fib_last_four_zeros_exist_l266_26662

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Last four digits of a natural number -/
def lastFourDigits (n : ℕ) : ℕ :=
  n % 10000

/-- Theorem: There exists a term in the first 100,000,001 Fibonacci numbers whose last four digits are all zeros -/
theorem fib_last_four_zeros_exist : ∃ n : ℕ, n < 100000001 ∧ lastFourDigits (fib n) = 0 := by
  sorry


end NUMINAMATH_CALUDE_fib_last_four_zeros_exist_l266_26662


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l266_26614

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < 2 ↔ x ∈ Set.Ioo (-2 : ℝ) (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l266_26614


namespace NUMINAMATH_CALUDE_next_meeting_after_105_days_l266_26689

/-- Represents the number of days between cinema visits for each boy -/
structure VisitIntervals :=
  (kolya : ℕ)
  (seryozha : ℕ)
  (vanya : ℕ)

/-- The least number of days after which all three boys meet at the cinema again -/
def nextMeeting (intervals : VisitIntervals) : ℕ :=
  Nat.lcm intervals.kolya (Nat.lcm intervals.seryozha intervals.vanya)

/-- Theorem stating that the next meeting of all three boys occurs after 105 days -/
theorem next_meeting_after_105_days :
  let intervals : VisitIntervals := { kolya := 3, seryozha := 7, vanya := 5 }
  nextMeeting intervals = 105 := by sorry

end NUMINAMATH_CALUDE_next_meeting_after_105_days_l266_26689


namespace NUMINAMATH_CALUDE_tees_per_member_l266_26600

/-- The number of people in Bill's golfing group -/
def group_size : ℕ := 4

/-- The number of tees in a generic package -/
def generic_package_size : ℕ := 12

/-- The number of tees in an aero flight package -/
def aero_package_size : ℕ := 2

/-- The maximum number of generic packages Bill can buy -/
def max_generic_packages : ℕ := 2

/-- The number of aero flight packages Bill must purchase -/
def aero_packages : ℕ := 28

/-- Theorem stating that the number of golf tees per member is 20 -/
theorem tees_per_member :
  (max_generic_packages * generic_package_size + aero_packages * aero_package_size) / group_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_tees_per_member_l266_26600


namespace NUMINAMATH_CALUDE_inequality_system_solution_l266_26623

theorem inequality_system_solution : 
  {x : ℝ | (8 * x - 3 ≤ 13) ∧ ((x - 1) / 3 - 2 < x - 1)} = {x : ℝ | -2 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l266_26623


namespace NUMINAMATH_CALUDE_pyramid_display_rows_l266_26605

/-- Represents the number of cans in a pyramid display. -/
def pyramid_display (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem stating that a pyramid display with 210 cans has 14 rows. -/
theorem pyramid_display_rows :
  ∃ (n : ℕ), pyramid_display n = 210 ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_display_rows_l266_26605


namespace NUMINAMATH_CALUDE_fly_probabilities_l266_26696

def fly_path (n m : ℕ) : ℕ := Nat.choose (n + m) n

theorem fly_probabilities :
  let p1 := (fly_path 8 10 : ℚ) / 2^18
  let p2 := ((fly_path 5 6 : ℚ) * (fly_path 2 4) : ℚ) / 2^18
  let p3 := (2 * (fly_path 2 7 : ℚ) * (fly_path 6 3) + 
             2 * (fly_path 3 6 : ℚ) * (fly_path 5 4) + 
             (fly_path 4 5 : ℚ) * (fly_path 4 5)) / 2^18
  (p1 = (fly_path 8 10 : ℚ) / 2^18) ∧
  (p2 = ((fly_path 5 6 : ℚ) * (fly_path 2 4) : ℚ) / 2^18) ∧
  (p3 = (2 * (fly_path 2 7 : ℚ) * (fly_path 6 3) + 
         2 * (fly_path 3 6 : ℚ) * (fly_path 5 4) + 
         (fly_path 4 5 : ℚ) * (fly_path 4 5)) / 2^18) :=
by sorry

end NUMINAMATH_CALUDE_fly_probabilities_l266_26696


namespace NUMINAMATH_CALUDE_money_distribution_l266_26669

/-- Given three people A, B, and C with a total of 1000 rupees between them,
    where B and C together have 600 rupees, and C has 300 rupees,
    prove that A and C together have 700 rupees. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 1000 →
  B + C = 600 →
  C = 300 →
  A + C = 700 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l266_26669


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l266_26652

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l266_26652


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l266_26674

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ ¬(a > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l266_26674


namespace NUMINAMATH_CALUDE_inverse_g_at_505_l266_26627

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 5

-- State the theorem
theorem inverse_g_at_505 : g⁻¹ 505 = 5 := by sorry

end NUMINAMATH_CALUDE_inverse_g_at_505_l266_26627


namespace NUMINAMATH_CALUDE_lucy_groceries_l266_26695

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 12

/-- The number of packs of noodles Lucy bought -/
def noodles : ℕ := 16

/-- The total number of grocery packs Lucy bought -/
def total_groceries : ℕ := cookies + noodles

theorem lucy_groceries : total_groceries = 28 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l266_26695


namespace NUMINAMATH_CALUDE_no_real_roots_iff_m_zero_l266_26677

theorem no_real_roots_iff_m_zero (m : ℝ) :
  (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) ↔ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_m_zero_l266_26677


namespace NUMINAMATH_CALUDE_initial_money_calculation_l266_26608

/-- The amount of money Monica and Sheila's mother gave them initially -/
def initial_money : ℝ := 50

/-- The cost of toilet paper -/
def toilet_paper_cost : ℝ := 12

/-- The cost of groceries -/
def groceries_cost : ℝ := 2 * toilet_paper_cost

/-- The amount of money left after buying toilet paper and groceries -/
def money_left : ℝ := initial_money - (toilet_paper_cost + groceries_cost)

/-- The cost of one pair of boots -/
def boot_cost : ℝ := 3 * money_left

/-- The additional money needed to buy two pairs of boots -/
def additional_money : ℝ := 2 * 35

theorem initial_money_calculation :
  initial_money = toilet_paper_cost + groceries_cost + money_left ∧
  2 * boot_cost = 2 * 3 * money_left ∧
  2 * boot_cost = 2 * 3 * money_left ∧
  2 * boot_cost - money_left = additional_money := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l266_26608


namespace NUMINAMATH_CALUDE_companion_numbers_example_companion_numbers_expression_l266_26619

/-- Two numbers are companion numbers if their sum equals their product. -/
def CompanionNumbers (a b : ℝ) : Prop := a + b = a * b

theorem companion_numbers_example : CompanionNumbers (-1) (1/2) := by sorry

theorem companion_numbers_expression (m n : ℝ) (h : CompanionNumbers m n) :
  -2 * m * n + 1/2 * (3 * m + 2 * (1/2 * n - m) + 3 * m * n - 6) = -3 := by sorry

end NUMINAMATH_CALUDE_companion_numbers_example_companion_numbers_expression_l266_26619


namespace NUMINAMATH_CALUDE_local_max_value_l266_26661

/-- The function f(x) = x³ - 12x --/
def f (x : ℝ) : ℝ := x^3 - 12*x

/-- m is the point of local maximum for f --/
def is_local_max (m : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - m| < δ → f x ≤ f m

theorem local_max_value :
  ∃ m : ℝ, is_local_max m ∧ m = -2 :=
sorry

end NUMINAMATH_CALUDE_local_max_value_l266_26661


namespace NUMINAMATH_CALUDE_section_through_center_l266_26679

-- Define a cube
def Cube := Set (ℝ × ℝ × ℝ)

-- Define a plane section
def PlaneSection := Set (ℝ × ℝ × ℝ)

-- Define the center of a cube
def centerOfCube (c : Cube) : ℝ × ℝ × ℝ := sorry

-- Define the volume of a set in 3D space
def volume (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define what it means for a plane to pass through a point
def passesThrough (p : PlaneSection) (point : ℝ × ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem section_through_center (c : Cube) (s : PlaneSection) :
  (∃ (A B : Set (ℝ × ℝ × ℝ)), A ∪ B = c ∧ A ∩ B = s ∧ volume A = volume B) →
  passesThrough s (centerOfCube c) := by sorry

end NUMINAMATH_CALUDE_section_through_center_l266_26679


namespace NUMINAMATH_CALUDE_min_rectangles_cover_square_l266_26686

/-- The smallest number of 2-by-3 non-overlapping rectangles needed to cover a 12-by-12 square exactly -/
def min_rectangles : ℕ := 24

/-- The side length of the square -/
def square_side : ℕ := 12

/-- The width of the rectangle -/
def rect_width : ℕ := 2

/-- The height of the rectangle -/
def rect_height : ℕ := 3

/-- The area of the square -/
def square_area : ℕ := square_side ^ 2

/-- The area of a single rectangle -/
def rect_area : ℕ := rect_width * rect_height

theorem min_rectangles_cover_square :
  min_rectangles * rect_area = square_area ∧
  ∃ (rows columns : ℕ),
    rows * columns = min_rectangles ∧
    rows * rect_height = square_side ∧
    columns * rect_width = square_side :=
by sorry

end NUMINAMATH_CALUDE_min_rectangles_cover_square_l266_26686
