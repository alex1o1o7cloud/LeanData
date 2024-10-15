import Mathlib

namespace NUMINAMATH_CALUDE_unique_perpendicular_line_parallel_intersections_perpendicular_line_in_plane_l2723_272387

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships
variable (outside : Point → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (in_plane : Point → Plane → Prop)
variable (in_line : Point → Line → Prop)

-- Theorem 1
theorem unique_perpendicular_line 
  (p : Point) (π : Plane) (h : outside p π) :
  ∃! l : Line, perpendicular l π ∧ in_line p l :=
sorry

-- Theorem 2
theorem parallel_intersections 
  (π₁ π₂ π₃ : Plane) (h : parallel_planes π₁ π₂) :
  parallel (intersect π₁ π₃) (intersect π₂ π₃) :=
sorry

-- Theorem 3
theorem perpendicular_line_in_plane 
  (π₁ π₂ : Plane) (p : Point) (l : Line)
  (h₁ : perpendicular_planes π₁ π₂) (h₂ : in_plane p π₁)
  (h₃ : perpendicular l π₂) (h₄ : in_line p l) :
  ∀ q : Point, in_line q l → in_plane q π₁ :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_parallel_intersections_perpendicular_line_in_plane_l2723_272387


namespace NUMINAMATH_CALUDE_min_sum_floor_l2723_272351

theorem min_sum_floor (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (⌊(x^2 + y^2) / z⌋ + ⌊(y^2 + z^2) / x⌋ + ⌊(z^2 + x^2) / y⌋ = 4) ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    ⌊(a^2 + b^2) / c⌋ + ⌊(b^2 + c^2) / a⌋ + ⌊(c^2 + a^2) / b⌋ ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_floor_l2723_272351


namespace NUMINAMATH_CALUDE_football_players_l2723_272328

theorem football_players (total : ℕ) (cricket : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 420)
  (h2 : cricket = 175)
  (h3 : both = 130)
  (h4 : neither = 50) :
  total - neither - (cricket - both) = 325 :=
sorry

end NUMINAMATH_CALUDE_football_players_l2723_272328


namespace NUMINAMATH_CALUDE_f_composition_value_l2723_272367

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 0
  else if x = 0 then Real.pi
  else Real.pi^2 + 1

theorem f_composition_value : f (f (f 1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l2723_272367


namespace NUMINAMATH_CALUDE_a_b_parallel_opposite_l2723_272378

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -4)

/-- Predicate to check if two vectors are parallel and in opposite directions -/
def parallel_opposite (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ w = (k * v.1, k * v.2)

/-- Theorem stating that vectors a and b are parallel and in opposite directions -/
theorem a_b_parallel_opposite : parallel_opposite a b := by
  sorry

end NUMINAMATH_CALUDE_a_b_parallel_opposite_l2723_272378


namespace NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l2723_272353

theorem percentage_of_red_non_honda_cars 
  (total_cars : ℕ) 
  (honda_cars : ℕ) 
  (honda_red_ratio : ℚ) 
  (total_red_ratio : ℚ) 
  (h1 : total_cars = 900) 
  (h2 : honda_cars = 500) 
  (h3 : honda_red_ratio = 90 / 100) 
  (h4 : total_red_ratio = 60 / 100) :
  (((total_red_ratio * total_cars) - (honda_red_ratio * honda_cars)) / (total_cars - honda_cars)) = 225 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l2723_272353


namespace NUMINAMATH_CALUDE_max_integer_k_l2723_272352

theorem max_integer_k (x y k : ℝ) : 
  x - 4*y = k - 1 →
  2*x + y = k →
  x - y ≤ 0 →
  ∀ m : ℤ, m ≤ k → m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_k_l2723_272352


namespace NUMINAMATH_CALUDE_cows_count_l2723_272338

/-- Represents the number of animals in the farm -/
structure FarmAnimals where
  ducks : ℕ
  cows : ℕ
  spiders : ℕ

/-- Checks if the given farm animals satisfy all the conditions -/
def satisfiesConditions (animals : FarmAnimals) : Prop :=
  let totalLegs := 2 * animals.ducks + 4 * animals.cows + 8 * animals.spiders
  let totalHeads := animals.ducks + animals.cows + animals.spiders
  totalLegs = 2 * totalHeads + 72 ∧
  animals.spiders = 2 * animals.ducks ∧
  totalHeads ≤ 40

/-- Theorem stating that the number of cows is 30 given the conditions -/
theorem cows_count (animals : FarmAnimals) :
  satisfiesConditions animals → animals.cows = 30 := by
  sorry

end NUMINAMATH_CALUDE_cows_count_l2723_272338


namespace NUMINAMATH_CALUDE_f_equals_negative_two_iff_b_equals_negative_one_l2723_272384

def f (x : ℝ) : ℝ := 5 * x + 3

theorem f_equals_negative_two_iff_b_equals_negative_one :
  ∀ b : ℝ, f b = -2 ↔ b = -1 := by sorry

end NUMINAMATH_CALUDE_f_equals_negative_two_iff_b_equals_negative_one_l2723_272384


namespace NUMINAMATH_CALUDE_radish_patch_area_l2723_272390

theorem radish_patch_area (pea_patch : ℝ) (radish_patch : ℝ) : 
  pea_patch = 2 * radish_patch →
  pea_patch / 6 = 5 →
  radish_patch = 15 := by
  sorry

end NUMINAMATH_CALUDE_radish_patch_area_l2723_272390


namespace NUMINAMATH_CALUDE_probability_sum_10_15_18_l2723_272393

def num_dice : ℕ := 3
def faces_per_die : ℕ := 6

def total_outcomes : ℕ := faces_per_die ^ num_dice

def sum_10_outcomes : ℕ := 27
def sum_15_outcomes : ℕ := 9
def sum_18_outcomes : ℕ := 1

def favorable_outcomes : ℕ := sum_10_outcomes + sum_15_outcomes + sum_18_outcomes

theorem probability_sum_10_15_18 : 
  (favorable_outcomes : ℚ) / total_outcomes = 37 / 216 := by sorry

end NUMINAMATH_CALUDE_probability_sum_10_15_18_l2723_272393


namespace NUMINAMATH_CALUDE_cone_height_equals_six_l2723_272355

/-- Proves that given a cylinder M with base radius 2 and height 6, and a cone N whose base diameter equals its slant height, if their volumes are equal, then the height of cone N is 6. -/
theorem cone_height_equals_six (r : ℝ) (h : ℝ) :
  (2 : ℝ) ^ 2 * 6 = (1 / 3) * r ^ 2 * h ∧ 
  h = r * Real.sqrt 3 →
  h = 6 := by sorry

end NUMINAMATH_CALUDE_cone_height_equals_six_l2723_272355


namespace NUMINAMATH_CALUDE_second_player_wins_l2723_272399

/-- A game where players take turns removing coins from a pile. -/
structure CoinGame where
  coins : ℕ              -- Number of coins in the pile
  max_take : ℕ           -- Maximum number of coins a player can take in one turn
  min_take : ℕ           -- Minimum number of coins a player can take in one turn

/-- Represents a player in the game. -/
inductive Player
| First
| Second

/-- Defines a winning strategy for a player. -/
def has_winning_strategy (game : CoinGame) (player : Player) : Prop :=
  ∃ (strategy : ℕ → ℕ), 
    (∀ n, game.min_take ≤ strategy n ∧ strategy n ≤ game.max_take) ∧
    (player = Player.First → strategy game.coins = game.coins) ∧
    (player = Player.Second → 
      ∀ first_move, game.min_take ≤ first_move ∧ first_move ≤ game.max_take →
        strategy (game.coins - first_move) = game.coins - first_move)

/-- The main theorem stating that the second player has a winning strategy in the specific game. -/
theorem second_player_wins :
  let game : CoinGame := { coins := 2016, max_take := 3, min_take := 1 }
  has_winning_strategy game Player.Second :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l2723_272399


namespace NUMINAMATH_CALUDE_six_digit_number_theorem_l2723_272394

def is_valid_six_digit_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def extract_digits (n : ℕ) : Fin 6 → ℕ
| 0 => n / 100000
| 1 => (n / 10000) % 10
| 2 => (n / 1000) % 10
| 3 => (n / 100) % 10
| 4 => (n / 10) % 10
| 5 => n % 10

theorem six_digit_number_theorem (n : ℕ) (hn : is_valid_six_digit_number n) :
  (extract_digits n 0 = 1) →
  (3 * n = (n % 100000) * 10 + 1) →
  (extract_digits n 1 + extract_digits n 2 + extract_digits n 3 + 
   extract_digits n 4 + extract_digits n 5 = 26) := by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_theorem_l2723_272394


namespace NUMINAMATH_CALUDE_complex_power_four_equals_negative_four_l2723_272388

theorem complex_power_four_equals_negative_four : 
  (1 + (1 / Complex.I)) ^ 4 = (-4 : ℂ) := by sorry

end NUMINAMATH_CALUDE_complex_power_four_equals_negative_four_l2723_272388


namespace NUMINAMATH_CALUDE_warehouse_space_theorem_l2723_272375

/-- Represents the warehouse with two floors and some occupied space -/
structure Warehouse :=
  (second_floor : ℝ)
  (first_floor : ℝ)
  (occupied_space : ℝ)

/-- The remaining available space in the warehouse -/
def remaining_space (w : Warehouse) : ℝ :=
  w.first_floor + w.second_floor - w.occupied_space

/-- The theorem stating the remaining available space in the warehouse -/
theorem warehouse_space_theorem (w : Warehouse) 
  (h1 : w.first_floor = 2 * w.second_floor)
  (h2 : w.occupied_space = w.second_floor / 4)
  (h3 : w.occupied_space = 5000) : 
  remaining_space w = 55000 := by
  sorry

#check warehouse_space_theorem

end NUMINAMATH_CALUDE_warehouse_space_theorem_l2723_272375


namespace NUMINAMATH_CALUDE_midpoint_coordinate_relation_l2723_272339

/-- Given two points D and E in the plane, if F is their midpoint,
    then 3 times the x-coordinate of F minus 5 times the y-coordinate of F equals 9. -/
theorem midpoint_coordinate_relation :
  let D : ℝ × ℝ := (30, 10)
  let E : ℝ × ℝ := (6, 8)
  let F : ℝ × ℝ := ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
  3 * F.1 - 5 * F.2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_relation_l2723_272339


namespace NUMINAMATH_CALUDE_train_length_l2723_272358

/-- Calculates the length of a train given its speed, platform length, and time to cross the platform. -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 108 →
  platform_length = 300.06 →
  crossing_time = 25 →
  (train_speed * 1000 / 3600 * crossing_time) - platform_length = 449.94 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2723_272358


namespace NUMINAMATH_CALUDE_overlap_area_is_one_l2723_272307

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three vertices -/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Calculate the area of overlap between two triangles on a 3x3 grid -/
def triangleOverlapArea (t1 t2 : Triangle) : ℝ :=
  sorry

/-- Theorem stating that the area of overlap between the given triangles is 1 square unit -/
theorem overlap_area_is_one :
  let t1 : Triangle := { v1 := {x := 0, y := 2}, v2 := {x := 2, y := 1}, v3 := {x := 0, y := 0} }
  let t2 : Triangle := { v1 := {x := 2, y := 2}, v2 := {x := 0, y := 1}, v3 := {x := 2, y := 0} }
  triangleOverlapArea t1 t2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_is_one_l2723_272307


namespace NUMINAMATH_CALUDE_guitar_center_shipping_fee_l2723_272308

/-- The shipping fee of Guitar Center given the conditions of the guitar purchase --/
theorem guitar_center_shipping_fee :
  let suggested_price : ℚ := 1000
  let guitar_center_discount : ℚ := 15 / 100
  let sweetwater_discount : ℚ := 10 / 100
  let savings : ℚ := 50
  let guitar_center_price := suggested_price * (1 - guitar_center_discount)
  let sweetwater_price := suggested_price * (1 - sweetwater_discount)
  guitar_center_price + (sweetwater_price - guitar_center_price - savings) = guitar_center_price :=
by sorry

end NUMINAMATH_CALUDE_guitar_center_shipping_fee_l2723_272308


namespace NUMINAMATH_CALUDE_sum_of_fifth_and_eighth_l2723_272345

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fifth_and_eighth (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 - 3*(a 3) - 5 = 0 →
  (a 10)^2 - 3*(a 10) - 5 = 0 →
  a 5 + a 8 = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fifth_and_eighth_l2723_272345


namespace NUMINAMATH_CALUDE_fixed_point_and_min_product_l2723_272397

/-- The line l passing through a fixed point P -/
def line_l (m x y : ℝ) : Prop := (3*m + 1)*x + (2 + 2*m)*y - 8 = 0

/-- The fixed point P -/
def point_P : ℝ × ℝ := (-4, 6)

/-- Line l₁ -/
def line_l1 (x : ℝ) : Prop := x = -1

/-- Line l₂ -/
def line_l2 (y : ℝ) : Prop := y = -1

/-- Theorem stating that P is the fixed point and the minimum value of |PM| · |PN| -/
theorem fixed_point_and_min_product :
  (∀ m : ℝ, line_l m (point_P.1) (point_P.2)) ∧
  (∃ min : ℝ, min = 42 ∧
    ∀ m : ℝ, ∀ M N : ℝ × ℝ,
      line_l m M.1 M.2 → line_l1 M.1 →
      line_l m N.1 N.2 → line_l2 N.2 →
      (M.1 - point_P.1)^2 + (M.2 - point_P.2)^2 *
      (N.1 - point_P.1)^2 + (N.2 - point_P.2)^2 ≥ min^2) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_and_min_product_l2723_272397


namespace NUMINAMATH_CALUDE_omega_squared_plus_7omega_plus_40_abs_l2723_272382

def ω : ℂ := 4 + 3 * Complex.I

theorem omega_squared_plus_7omega_plus_40_abs : 
  Complex.abs (ω^2 + 7*ω + 40) = 15 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_omega_squared_plus_7omega_plus_40_abs_l2723_272382


namespace NUMINAMATH_CALUDE_possible_m_values_l2723_272316

-- Define set A
def A : Set ℤ := {-1, 1}

-- Define set B
def B (m : ℤ) : Set ℤ := {x | m * x = 1}

-- Theorem statement
theorem possible_m_values :
  ∀ m : ℤ, B m ⊆ A → (m = 0 ∨ m = 1 ∨ m = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_possible_m_values_l2723_272316


namespace NUMINAMATH_CALUDE_combination_problem_l2723_272385

theorem combination_problem (m : ℕ) : 
  (1 : ℚ) / Nat.choose 5 m - (1 : ℚ) / Nat.choose 6 m = (7 : ℚ) / (10 * Nat.choose 7 m) → 
  Nat.choose 8 m = 28 := by
  sorry

end NUMINAMATH_CALUDE_combination_problem_l2723_272385


namespace NUMINAMATH_CALUDE_melanie_brownies_batches_l2723_272340

/-- Represents the number of brownies in each batch -/
def brownies_per_batch : ℕ := 20

/-- Represents the fraction of brownies set aside for the bake sale -/
def bake_sale_fraction : ℚ := 3/4

/-- Represents the fraction of remaining brownies put in a container -/
def container_fraction : ℚ := 3/5

/-- Represents the number of brownies given out -/
def brownies_given_out : ℕ := 20

/-- Proves that Melanie baked 10 batches of brownies -/
theorem melanie_brownies_batches :
  ∃ (batches : ℕ),
    batches = 10 ∧
    (brownies_per_batch * batches : ℚ) * (1 - bake_sale_fraction) * (1 - container_fraction) =
      brownies_given_out :=
by sorry

end NUMINAMATH_CALUDE_melanie_brownies_batches_l2723_272340


namespace NUMINAMATH_CALUDE_range_of_f_l2723_272324

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 3

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.Icc (-5 : ℝ) 13, ∃ x ∈ Set.Icc 2 5, f x = y ∧
  ∀ x ∈ Set.Icc 2 5, f x ∈ Set.Icc (-5 : ℝ) 13 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2723_272324


namespace NUMINAMATH_CALUDE_work_completion_proof_l2723_272347

/-- A's work rate in days -/
def a_rate : ℚ := 1 / 15

/-- B's work rate in days -/
def b_rate : ℚ := 1 / 20

/-- The fraction of work left after A and B work together -/
def work_left : ℚ := 65 / 100

/-- The number of days A and B worked together -/
def days_worked : ℚ := 3

theorem work_completion_proof :
  (a_rate + b_rate) * days_worked = 1 - work_left :=
sorry

end NUMINAMATH_CALUDE_work_completion_proof_l2723_272347


namespace NUMINAMATH_CALUDE_polynomial_arrangement_l2723_272330

-- Define the original polynomial
def original_polynomial (x : ℝ) : ℝ := 3*x^2 - x + x^3 - 1

-- Define the arranged polynomial
def arranged_polynomial (x : ℝ) : ℝ := -1 - x + 3*x^2 + x^3

-- Theorem stating that the original polynomial is equal to the arranged polynomial
theorem polynomial_arrangement :
  ∀ x : ℝ, original_polynomial x = arranged_polynomial x :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_arrangement_l2723_272330


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2723_272342

/-- For a right circular cone with volume 16π cubic centimeters and height 6 cm,
    the circumference of its base is 4√2π cm. -/
theorem cone_base_circumference :
  ∀ (r : ℝ), 
    (1 / 3 * π * r^2 * 6 = 16 * π) →
    (2 * π * r = 4 * Real.sqrt 2 * π) :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2723_272342


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l2723_272395

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 20) :
  let side_length := face_perimeter / 4
  (side_length ^ 3) = 125 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l2723_272395


namespace NUMINAMATH_CALUDE_bus_overlap_count_l2723_272300

-- Define the bus schedules
def busA_interval : ℕ := 6
def busB_interval : ℕ := 10
def busC_interval : ℕ := 14

-- Define the time range in minutes (5:00 PM to 10:00 PM)
def start_time : ℕ := 240  -- 4 hours after 1:00 PM
def end_time : ℕ := 540    -- 9 hours after 1:00 PM

-- Function to calculate the number of overlaps between two buses
def count_overlaps (interval1 interval2 start_time end_time : ℕ) : ℕ :=
  (end_time - start_time) / Nat.lcm interval1 interval2 + 1

-- Function to calculate the total number of distinct overlaps
def total_distinct_overlaps (start_time end_time : ℕ) : ℕ :=
  let ab_overlaps := count_overlaps busA_interval busB_interval start_time end_time
  let bc_overlaps := count_overlaps busB_interval busC_interval start_time end_time
  let ac_overlaps := count_overlaps busA_interval busC_interval start_time end_time
  ab_overlaps + bc_overlaps + ac_overlaps - 2  -- Subtracting 2 for common overlaps

-- The main theorem
theorem bus_overlap_count : 
  total_distinct_overlaps start_time end_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_bus_overlap_count_l2723_272300


namespace NUMINAMATH_CALUDE_no_roots_in_interval_l2723_272381

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - 4*x^3 + 10*x^2

-- State the theorem
theorem no_roots_in_interval :
  ∀ x ∈ Set.Icc 1 2, f x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_no_roots_in_interval_l2723_272381


namespace NUMINAMATH_CALUDE_membership_change_l2723_272396

theorem membership_change (initial_members : ℝ) : 
  let fall_increase := 0.07
  let spring_decrease := 0.19
  let fall_members := initial_members * (1 + fall_increase)
  let spring_members := fall_members * (1 - spring_decrease)
  let total_change_percentage := (spring_members / initial_members - 1) * 100
  total_change_percentage = -13.33 := by
sorry

end NUMINAMATH_CALUDE_membership_change_l2723_272396


namespace NUMINAMATH_CALUDE_rectangular_garden_length_l2723_272329

/-- Theorem: For a rectangular garden with a perimeter of 600 m and a breadth of 95 m, the length is 205 m. -/
theorem rectangular_garden_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 600) 
  (h2 : breadth = 95) :
  2 * (breadth + 205) = perimeter := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_length_l2723_272329


namespace NUMINAMATH_CALUDE_prime_dates_in_leap_year_l2723_272368

def isPrimeMonth (m : Nat) : Bool :=
  m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 7 ∨ m = 11

def isPrimeDay (d : Nat) : Bool :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 11 ∨ d = 13 ∨ d = 17 ∨ d = 19 ∨ d = 23 ∨ d = 29 ∨ d = 31

def daysInMonth (m : Nat) : Nat :=
  if m = 2 then 29
  else if m = 4 ∨ m = 11 then 30
  else 31

def countPrimeDates : Nat :=
  (List.range 12).filter isPrimeMonth
    |>.map (fun m => (List.range (daysInMonth m)).filter isPrimeDay |>.length)
    |>.sum

theorem prime_dates_in_leap_year :
  countPrimeDates = 63 := by
  sorry

end NUMINAMATH_CALUDE_prime_dates_in_leap_year_l2723_272368


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2723_272362

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    x1^2 + 2*m*x1 + m^2 + m = 0 ∧ 
    x2^2 + 2*m*x2 + m^2 + m = 0 ∧ 
    x1^2 + x2^2 = 12) → 
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2723_272362


namespace NUMINAMATH_CALUDE_center_trajectory_is_parabola_l2723_272302

/-- A circle passing through a point and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passesThrough : center.1^2 + (center.2 - 3)^2 = radius^2
  tangentToLine : center.2 + radius = 3

/-- The trajectory of the center of a moving circle -/
def centerTrajectory (c : TangentCircle) : Prop :=
  c.center.1^2 = 12 * c.center.2

/-- Theorem: The trajectory of the center of a circle passing through (0, 3) 
    and tangent to y + 3 = 0 is described by x^2 = 12y -/
theorem center_trajectory_is_parabola :
  ∀ c : TangentCircle, centerTrajectory c :=
sorry

end NUMINAMATH_CALUDE_center_trajectory_is_parabola_l2723_272302


namespace NUMINAMATH_CALUDE_negation_of_implication_l2723_272398

theorem negation_of_implication :
  (¬(∀ x : ℝ, x > 1 → x^2 > 1)) ↔ (∀ x : ℝ, x ≤ 1 → x^2 ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2723_272398


namespace NUMINAMATH_CALUDE_exists_expr_2023_l2723_272319

/-- An arithmetic expression without parentheses -/
inductive ArithExpr
  | Const : ℤ → ArithExpr
  | Add : ArithExpr → ArithExpr → ArithExpr
  | Sub : ArithExpr → ArithExpr → ArithExpr
  | Mul : ArithExpr → ArithExpr → ArithExpr
  | Div : ArithExpr → ArithExpr → ArithExpr

/-- Evaluation function for ArithExpr -/
def eval : ArithExpr → ℤ
  | ArithExpr.Const n => n
  | ArithExpr.Add a b => eval a + eval b
  | ArithExpr.Sub a b => eval a - eval b
  | ArithExpr.Mul a b => eval a * eval b
  | ArithExpr.Div a b => eval a / eval b

/-- Theorem stating the existence of an arithmetic expression evaluating to 2023 -/
theorem exists_expr_2023 : ∃ e : ArithExpr, eval e = 2023 := by
  sorry


end NUMINAMATH_CALUDE_exists_expr_2023_l2723_272319


namespace NUMINAMATH_CALUDE_spurs_team_size_l2723_272392

/-- The number of basketballs each player has -/
def basketballs_per_player : ℕ := 11

/-- The total number of basketballs -/
def total_basketballs : ℕ := 242

/-- The number of players on the team -/
def number_of_players : ℕ := total_basketballs / basketballs_per_player

theorem spurs_team_size :
  number_of_players = 22 :=
by sorry

end NUMINAMATH_CALUDE_spurs_team_size_l2723_272392


namespace NUMINAMATH_CALUDE_profit_and_max_profit_l2723_272333

/-- Initial profit per visitor in yuan -/
def initial_profit_per_visitor : ℝ := 10

/-- Initial daily visitor count -/
def initial_visitor_count : ℝ := 500

/-- Visitor loss per yuan of price increase -/
def visitor_loss_per_yuan : ℝ := 20

/-- Calculate profit based on price increase -/
def profit (price_increase : ℝ) : ℝ :=
  (initial_profit_per_visitor + price_increase) * (initial_visitor_count - visitor_loss_per_yuan * price_increase)

/-- Ticket price increase for 6000 yuan daily profit -/
def price_increase_for_target_profit : ℝ := 10

/-- Ticket price increase for maximum profit -/
def price_increase_for_max_profit : ℝ := 7.5

theorem profit_and_max_profit :
  (profit price_increase_for_target_profit = 6000) ∧
  (∀ x : ℝ, profit x ≤ profit price_increase_for_max_profit) := by
  sorry

end NUMINAMATH_CALUDE_profit_and_max_profit_l2723_272333


namespace NUMINAMATH_CALUDE_vector_combination_equality_l2723_272374

/-- Given vectors a, b, and c in ℝ³, prove that 2a - 3b + 4c equals (16, 0, -19) -/
theorem vector_combination_equality (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (3, 5, 1)) 
  (hb : b = (2, 2, 3)) 
  (hc : c = (4, -1, -3)) : 
  (2 : ℝ) • a - (3 : ℝ) • b + (4 : ℝ) • c = (16, 0, -19) := by
  sorry

end NUMINAMATH_CALUDE_vector_combination_equality_l2723_272374


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2723_272323

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) : 
  z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2723_272323


namespace NUMINAMATH_CALUDE_part_one_part_two_l2723_272343

/-- Definition of arithmetic sequence sum -/
def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

/-- Theorem for part (I) -/
theorem part_one :
  ∃! k : ℕ+, arithmetic_sum (3/2) 1 (k^2) = (arithmetic_sum (3/2) 1 k)^2 :=
sorry

/-- Definition of arithmetic sequence -/
def arithmetic_seq (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

/-- Theorem for part (II) -/
theorem part_two :
  ∀ a₁ d : ℚ, (∀ k : ℕ+, arithmetic_sum a₁ d (k^2) = (arithmetic_sum a₁ d k)^2) ↔
    ((∀ n, arithmetic_seq a₁ d n = 0) ∨
     (∀ n, arithmetic_seq a₁ d n = 1) ∨
     (∀ n, arithmetic_seq a₁ d n = 2 * n - 1)) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2723_272343


namespace NUMINAMATH_CALUDE_expand_expression_l2723_272336

theorem expand_expression (x : ℝ) : 20 * (3 * x + 4) - 10 = 60 * x + 70 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2723_272336


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2723_272327

theorem inequality_and_equality_condition (a b c : ℝ) :
  (5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * a * c + 4 * b * c) ∧
  (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * a * c + 4 * b * c ↔ a = 0 ∧ b = 0 ∧ c = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2723_272327


namespace NUMINAMATH_CALUDE_derivative_sin_plus_exp_cos_l2723_272344

theorem derivative_sin_plus_exp_cos (x : ℝ) :
  let y : ℝ → ℝ := λ x => Real.sin x + Real.exp x * Real.cos x
  deriv y x = (1 + Real.exp x) * Real.cos x - Real.exp x * Real.sin x :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_plus_exp_cos_l2723_272344


namespace NUMINAMATH_CALUDE_probability_twelve_no_consecutive_ones_l2723_272377

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def validSequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

/-- Probability of a valid sequence of length n -/
def probability (n : ℕ) : ℚ :=
  (validSequences n : ℚ) / (totalSequences n : ℚ)

theorem probability_twelve_no_consecutive_ones :
  probability 12 = 377 / 4096 :=
sorry

end NUMINAMATH_CALUDE_probability_twelve_no_consecutive_ones_l2723_272377


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l2723_272365

-- Define the sets A and B
def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ -1}
def B : Set ℝ := {x | x + 4 ≥ 0}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ -5} := by sorry

-- Theorem for ∁ᵤ(A ∩ B)
theorem complement_of_intersection_A_and_B : (A ∩ B)ᶜ = {x : ℝ | x < -4 ∨ x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l2723_272365


namespace NUMINAMATH_CALUDE_pattern_cost_is_15_l2723_272373

/-- The cost of a sewing pattern given the total spent, fabric cost per yard, yards of fabric bought,
    thread cost per spool, and number of thread spools bought. -/
def pattern_cost (total_spent fabric_cost_per_yard yards_fabric thread_cost_per_spool num_thread_spools : ℕ) : ℕ :=
  total_spent - (fabric_cost_per_yard * yards_fabric + thread_cost_per_spool * num_thread_spools)

/-- Theorem stating that the pattern cost is $15 given the specific conditions. -/
theorem pattern_cost_is_15 :
  pattern_cost 141 24 5 3 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_pattern_cost_is_15_l2723_272373


namespace NUMINAMATH_CALUDE_problem_solution_l2723_272356

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1/p + 1/q = 1) 
  (h4 : p*q = 12) : 
  q = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2723_272356


namespace NUMINAMATH_CALUDE_triangle_area_l2723_272364

theorem triangle_area (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → 
  b = 5 → 
  5 * cos_theta^2 - 7 * cos_theta - 6 = 0 →
  abs (1/2 * a * b * cos_theta) = 4.5 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l2723_272364


namespace NUMINAMATH_CALUDE_arccos_sin_five_equals_five_minus_pi_over_two_l2723_272326

theorem arccos_sin_five_equals_five_minus_pi_over_two :
  Real.arccos (Real.sin 5) = (5 - Real.pi) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_five_equals_five_minus_pi_over_two_l2723_272326


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l2723_272332

theorem sum_of_solutions_squared_equation (x₁ x₂ : ℝ) :
  (x₁ - 8)^2 = 49 → (x₂ - 8)^2 = 49 → x₁ + x₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l2723_272332


namespace NUMINAMATH_CALUDE_min_value_expression_l2723_272311

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sqrt : y^2 = x) :
  (x^2 + y^4) / (x * y^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2723_272311


namespace NUMINAMATH_CALUDE_absolute_value_expression_l2723_272369

theorem absolute_value_expression (x : ℤ) (h : x = 1999) :
  |4*x^2 - 5*x + 1| - 4*|x^2 + 2*x + 2| + 3*x + 7 = -19990 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l2723_272369


namespace NUMINAMATH_CALUDE_yanna_apples_l2723_272320

theorem yanna_apples (apples_to_zenny apples_to_andrea apples_kept : ℕ) : 
  apples_to_zenny = 18 → 
  apples_to_andrea = 6 → 
  apples_kept = 36 → 
  apples_to_zenny + apples_to_andrea + apples_kept = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_yanna_apples_l2723_272320


namespace NUMINAMATH_CALUDE_unique_integer_with_conditions_l2723_272304

theorem unique_integer_with_conditions : ∃! n : ℤ,
  50 ≤ n ∧ n ≤ 100 ∧
  n % 7 = 0 ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n = 84 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_with_conditions_l2723_272304


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l2723_272348

theorem power_mod_seventeen : 5^2021 ≡ 11 [ZMOD 17] := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l2723_272348


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l2723_272322

/-- Represents a 9x9 checkerboard filled with numbers 1 through 81 -/
def Checkerboard := Fin 9 → Fin 9 → Nat

/-- The number at position (i, j) on the checkerboard -/
def number_at (board : Checkerboard) (i j : Fin 9) : Nat :=
  9 * i.val + j.val + 1

/-- The sum of numbers in the four corners of the checkerboard -/
def corner_sum (board : Checkerboard) : Nat :=
  number_at board 0 0 + number_at board 0 8 + 
  number_at board 8 0 + number_at board 8 8

/-- Theorem stating that the sum of numbers in the four corners is 164 -/
theorem corner_sum_is_164 (board : Checkerboard) : corner_sum board = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l2723_272322


namespace NUMINAMATH_CALUDE_layla_earnings_correct_l2723_272309

/-- Calculates the babysitting earnings for a given family -/
def family_earnings (base_rate : ℚ) (hours : ℚ) (bonus_threshold : ℚ) (bonus_amount : ℚ) 
  (discount_rate : ℚ) (flat_rate : ℚ) (is_weekend : Bool) (past_midnight : Bool) : ℚ :=
  sorry

/-- Calculates Layla's total babysitting earnings -/
def layla_total_earnings : ℚ :=
  let donaldsons := family_earnings 15 7 5 5 0 0 false false
  let merck := family_earnings 18 6 3 0 0.1 0 false false
  let hille := family_earnings 20 3 0 10 0 0 false true
  let johnson := family_earnings 22 4 4 0 0 80 false false
  let ramos := family_earnings 25 2 0 20 0 0 true false
  donaldsons + merck + hille + johnson + ramos

theorem layla_earnings_correct : layla_total_earnings = 435.2 := by
  sorry

end NUMINAMATH_CALUDE_layla_earnings_correct_l2723_272309


namespace NUMINAMATH_CALUDE_young_photographer_club_l2723_272346

theorem young_photographer_club (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  ∃ (mixed_groups : ℕ),
    mixed_groups = 72 ∧
    mixed_groups * 2 = total_groups * group_size - boy_boy_photos - girl_girl_photos :=
by sorry

end NUMINAMATH_CALUDE_young_photographer_club_l2723_272346


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2723_272389

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {3, 4, 5}

-- Theorem statement
theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2723_272389


namespace NUMINAMATH_CALUDE_no_valid_distribution_of_skittles_l2723_272306

theorem no_valid_distribution_of_skittles : ¬ ∃ (F : ℕ+), 
  (14 - 3 * F.val ≥ 3) ∧ (14 - 3 * F.val) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_distribution_of_skittles_l2723_272306


namespace NUMINAMATH_CALUDE_same_color_probability_l2723_272357

/-- The probability of drawing two balls of the same color from a bag with green and white balls -/
theorem same_color_probability (green white : ℕ) (h : green = 5 ∧ white = 9) :
  let total := green + white
  let p_green := green / total
  let p_white := white / total
  let p_same_color := p_green * ((green - 1) / (total - 1)) + p_white * ((white - 1) / (total - 1))
  p_same_color = 46 / 91 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2723_272357


namespace NUMINAMATH_CALUDE_xy_value_given_condition_l2723_272359

theorem xy_value_given_condition (x y : ℝ) : 
  |x - 2| + Real.sqrt (y + 3) = 0 → x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_given_condition_l2723_272359


namespace NUMINAMATH_CALUDE_units_digit_problem_l2723_272314

theorem units_digit_problem : ∃ n : ℕ, (3 * 19 * 1933 - 3^4) % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_problem_l2723_272314


namespace NUMINAMATH_CALUDE_total_dress_designs_l2723_272360

/-- Represents the number of fabric color choices --/
def num_colors : Nat := 5

/-- Represents the number of pattern choices --/
def num_patterns : Nat := 4

/-- Represents the number of sleeve length choices --/
def num_sleeve_lengths : Nat := 2

/-- Theorem stating the total number of possible dress designs --/
theorem total_dress_designs : num_colors * num_patterns * num_sleeve_lengths = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l2723_272360


namespace NUMINAMATH_CALUDE_max_value_cos_theta_l2723_272366

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos x - Real.sin x

theorem max_value_cos_theta (θ : ℝ) 
  (h : ∀ x, f x ≤ f θ) : 
  Real.cos θ = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_theta_l2723_272366


namespace NUMINAMATH_CALUDE_pet_shop_kittens_l2723_272335

/-- Calculates the number of kittens in a pet shop given the following conditions:
  * The pet shop has 2 puppies
  * A puppy costs $20
  * A kitten costs $15
  * The total stock is worth $100
-/
theorem pet_shop_kittens (num_puppies : ℕ) (puppy_cost kitten_cost total_stock : ℚ) : 
  num_puppies = 2 → 
  puppy_cost = 20 → 
  kitten_cost = 15 → 
  total_stock = 100 → 
  (total_stock - num_puppies * puppy_cost) / kitten_cost = 4 := by
  sorry

#check pet_shop_kittens

end NUMINAMATH_CALUDE_pet_shop_kittens_l2723_272335


namespace NUMINAMATH_CALUDE_incorrect_games_proportion_l2723_272318

/-- Represents a chess tournament -/
structure ChessTournament where
  N : ℕ  -- number of players
  incorrect_games : ℕ  -- number of incorrect games

/-- Definition of a round-robin tournament -/
def is_round_robin (t : ChessTournament) : Prop :=
  t.incorrect_games ≤ t.N * (t.N - 1) / 2

/-- The main theorem: incorrect games are less than 75% of total games -/
theorem incorrect_games_proportion (t : ChessTournament) 
  (h : is_round_robin t) : 
  (4 * t.incorrect_games : ℚ) < (3 * t.N * (t.N - 1) : ℚ) := by
  sorry


end NUMINAMATH_CALUDE_incorrect_games_proportion_l2723_272318


namespace NUMINAMATH_CALUDE_sin_alpha_value_l2723_272386

-- Define the angle α
def α : Real := sorry

-- Define the point P on the terminal side of α
def P : ℝ × ℝ := (-2, 1)

-- Theorem statement
theorem sin_alpha_value :
  (α.sin = -2 * Real.sqrt 5 / 5) ∧
  (α.cos ≥ 0) ∧
  (α.sin * 2 + α.cos * (-2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l2723_272386


namespace NUMINAMATH_CALUDE_twenty_percent_equals_fiftyfour_l2723_272371

theorem twenty_percent_equals_fiftyfour (x : ℝ) : (20 / 100) * x = 54 → x = 270 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_equals_fiftyfour_l2723_272371


namespace NUMINAMATH_CALUDE_line_length_difference_l2723_272312

-- Define the lengths of the lines in inches
def white_line_inches : ℝ := 7.666666666666667
def blue_line_inches : ℝ := 3.3333333333333335

-- Define conversion rates
def inches_to_cm : ℝ := 2.54
def cm_to_mm : ℝ := 10

-- Theorem statement
theorem line_length_difference : 
  (white_line_inches * inches_to_cm - blue_line_inches * inches_to_cm) * cm_to_mm = 110.05555555555553 := by
  sorry

end NUMINAMATH_CALUDE_line_length_difference_l2723_272312


namespace NUMINAMATH_CALUDE_inequality_proof_l2723_272379

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2723_272379


namespace NUMINAMATH_CALUDE_plan_b_more_cost_effective_l2723_272391

/-- Plan A's cost per megabyte in cents -/
def plan_a_cost_per_mb : ℚ := 12

/-- Plan B's setup fee in cents -/
def plan_b_setup_fee : ℚ := 3000

/-- Plan B's cost per megabyte in cents -/
def plan_b_cost_per_mb : ℚ := 8

/-- The minimum number of megabytes for Plan B to be more cost-effective -/
def min_mb_for_plan_b : ℕ := 751

theorem plan_b_more_cost_effective :
  (↑min_mb_for_plan_b * plan_b_cost_per_mb + plan_b_setup_fee < ↑min_mb_for_plan_b * plan_a_cost_per_mb) ∧
  ∀ m : ℕ, m < min_mb_for_plan_b →
    (↑m * plan_b_cost_per_mb + plan_b_setup_fee ≥ ↑m * plan_a_cost_per_mb) :=
by sorry

end NUMINAMATH_CALUDE_plan_b_more_cost_effective_l2723_272391


namespace NUMINAMATH_CALUDE_books_sold_l2723_272334

theorem books_sold (initial_books : ℕ) (remaining_books : ℕ) : initial_books = 136 → remaining_books = 27 → initial_books - remaining_books = 109 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l2723_272334


namespace NUMINAMATH_CALUDE_spade_calculation_l2723_272354

-- Define the spade operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_calculation : spade 3 (spade 5 (spade 7 10)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l2723_272354


namespace NUMINAMATH_CALUDE_parabola_transformation_l2723_272310

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = -2x^2 + 1 -/
def original_parabola : Parabola := { a := -2, b := 0, c := 1 }

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * h, c := p.a * h^2 + p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- The resulting parabola after transformations -/
def transformed_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 3) (-1)

theorem parabola_transformation :
  transformed_parabola = { a := -2, b := 12, c := -18 } :=
sorry

end NUMINAMATH_CALUDE_parabola_transformation_l2723_272310


namespace NUMINAMATH_CALUDE_gift_packaging_combinations_l2723_272361

/-- The number of varieties of wrapping paper. -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of colors of ribbon. -/
def ribbon_colors : ℕ := 5

/-- The number of types of gift tags. -/
def gift_tag_types : ℕ := 6

/-- The total number of possible gift packaging combinations. -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_tag_types

/-- Theorem stating that the total number of gift packaging combinations is 300. -/
theorem gift_packaging_combinations :
  total_combinations = 300 :=
by sorry

end NUMINAMATH_CALUDE_gift_packaging_combinations_l2723_272361


namespace NUMINAMATH_CALUDE_range_of_sum_l2723_272331

theorem range_of_sum (x y : ℝ) (h1 : x - y = 4) (h2 : x > 3) (h3 : y < 1) :
  2 < x + y ∧ x + y < 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l2723_272331


namespace NUMINAMATH_CALUDE_sum_of_fraction_parts_is_correct_l2723_272350

/-- The repeating decimal 3.71717171... -/
def repeating_decimal : ℚ := 3 + 71/99

/-- The sum of the numerator and denominator of the fraction representing
    the repeating decimal 3.71717171... in its lowest terms -/
def sum_of_fraction_parts : ℕ := 467

/-- Theorem stating that the sum of the numerator and denominator of the fraction
    representing 3.71717171... in its lowest terms is 467 -/
theorem sum_of_fraction_parts_is_correct :
  ∃ (n d : ℕ), d ≠ 0 ∧ repeating_decimal = n / d ∧ Nat.gcd n d = 1 ∧ n + d = sum_of_fraction_parts := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fraction_parts_is_correct_l2723_272350


namespace NUMINAMATH_CALUDE_Φ_is_connected_l2723_272363

-- Define the set Φ
def Φ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               Real.sqrt (y^2 - 8*x^2 - 6*y + 9) ≤ 3*y - 1 ∧
               x^2 + y^2 ≤ 9}

-- Theorem statement
theorem Φ_is_connected : IsConnected Φ := by
  sorry

end NUMINAMATH_CALUDE_Φ_is_connected_l2723_272363


namespace NUMINAMATH_CALUDE_coat_price_and_tax_l2723_272303

/-- Represents the price of a coat -/
structure CoatPrice where
  original : ℝ
  discounted : ℝ
  taxRate : ℝ

/-- Calculates the tax amount based on the original price and tax rate -/
def calculateTax (price : CoatPrice) : ℝ :=
  price.original * price.taxRate

theorem coat_price_and_tax (price : CoatPrice) 
  (h1 : price.discounted = 72)
  (h2 : price.discounted = (2/5) * price.original)
  (h3 : price.taxRate = 0.05) :
  price.original = 180 ∧ calculateTax price = 9 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_and_tax_l2723_272303


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2723_272321

theorem tangent_line_problem (k a : ℝ) : 
  (∃ b : ℝ, (3 = 4 + a / 2 + 1) ∧ 
             (3 = 2 * k + b) ∧ 
             (k = 2 * 2 - a / 4)) → 
  (∃ b : ℝ, (3 = 4 + a / 2 + 1) ∧ 
             (3 = 2 * k + b) ∧ 
             (k = 2 * 2 - a / 4) ∧ 
             b = -7) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l2723_272321


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2723_272341

theorem lcm_hcf_problem (a b : ℕ+) :
  Nat.lcm a b = 2310 →
  Nat.gcd a b = 47 →
  a = 210 →
  b = 517 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2723_272341


namespace NUMINAMATH_CALUDE_combination_problem_l2723_272376

/-- Given that 1/C_5^m - 1/C_6^m = 7/(10C_7^m), prove that C_{21}^m = 210 -/
theorem combination_problem (m : ℕ) : 
  (1 / (Nat.choose 5 m : ℚ) - 1 / (Nat.choose 6 m : ℚ) = 7 / (10 * (Nat.choose 7 m : ℚ))) → 
  Nat.choose 21 m = 210 := by
sorry

end NUMINAMATH_CALUDE_combination_problem_l2723_272376


namespace NUMINAMATH_CALUDE_smallest_with_specific_divisor_counts_l2723_272301

/-- Count of positive odd integer divisors of n -/
def oddDivisorsCount (n : ℕ+) : ℕ := sorry

/-- Count of positive even integer divisors of n -/
def evenDivisorsCount (n : ℕ+) : ℕ := sorry

/-- Predicate to check if a number satisfies the divisor count conditions -/
def satisfiesDivisorCounts (n : ℕ+) : Prop :=
  oddDivisorsCount n = 7 ∧ evenDivisorsCount n = 14

theorem smallest_with_specific_divisor_counts :
  satisfiesDivisorCounts 1458 ∧
  ∀ m : ℕ+, m < 1458 → ¬satisfiesDivisorCounts m := by
  sorry

end NUMINAMATH_CALUDE_smallest_with_specific_divisor_counts_l2723_272301


namespace NUMINAMATH_CALUDE_smallest_angle_in_3_4_5_ratio_triangle_l2723_272315

theorem smallest_angle_in_3_4_5_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    (b = (4/3) * a) → (c = (5/3) * a) →
    (a + b + c = 180) →
    a = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_3_4_5_ratio_triangle_l2723_272315


namespace NUMINAMATH_CALUDE_final_state_is_green_l2723_272337

/-- Represents the colors of chameleons -/
inductive Color
  | Yellow
  | Red
  | Green

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons -/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 34

/-- Represents a color change event between two chameleons -/
def colorChange (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.Yellow, Color.Red => Color.Green
  | Color.Yellow, Color.Green => Color.Red
  | Color.Red, Color.Yellow => Color.Green
  | Color.Red, Color.Green => Color.Yellow
  | Color.Green, Color.Yellow => Color.Red
  | Color.Green, Color.Red => Color.Yellow
  | _, _ => c1  -- No change if colors are the same

/-- Theorem: The only possible final state is all chameleons being green -/
theorem final_state_is_green (finalState : ChameleonState) :
  (finalState.yellow + finalState.red + finalState.green = totalChameleons) →
  (∀ (c1 c2 : Color), colorChange c1 c2 = colorChange c2 c1) →
  (finalState.yellow = 0 ∧ finalState.red = 0 ∧ finalState.green = totalChameleons) :=
by sorry

#check final_state_is_green

end NUMINAMATH_CALUDE_final_state_is_green_l2723_272337


namespace NUMINAMATH_CALUDE_correct_operation_l2723_272380

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - b * a^2 = 2 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2723_272380


namespace NUMINAMATH_CALUDE_x_minus_y_equals_three_l2723_272313

theorem x_minus_y_equals_three (x y : ℝ) 
  (eq1 : 3 * x - 5 * y = 5)
  (eq2 : x / (x + y) = 5 / 7) :
  x - y = 3 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_three_l2723_272313


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2723_272372

theorem arithmetic_calculation : 4 * (8 - 3) + 6 / 2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2723_272372


namespace NUMINAMATH_CALUDE_min_teams_for_employees_l2723_272383

theorem min_teams_for_employees (total_employees : ℕ) (max_team_size : ℕ) (h1 : total_employees = 36) (h2 : max_team_size = 12) : 
  (total_employees + max_team_size - 1) / max_team_size = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_teams_for_employees_l2723_272383


namespace NUMINAMATH_CALUDE_complex_simplification_l2723_272317

theorem complex_simplification : (4 - 3*I)^2 + (1 + 2*I) = 8 - 22*I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l2723_272317


namespace NUMINAMATH_CALUDE_tens_digit_of_19_power_2021_l2723_272305

theorem tens_digit_of_19_power_2021 : ∃ n : ℕ, 19^2021 ≡ 10*n + 1 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_power_2021_l2723_272305


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2723_272370

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 4 * x * y) : 1 / x + 1 / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2723_272370


namespace NUMINAMATH_CALUDE_cube_root_sum_of_threes_l2723_272325

theorem cube_root_sum_of_threes : Real.sqrt (3^3 + 3^3 + 3^3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sum_of_threes_l2723_272325


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l2723_272349

theorem arithmetic_geometric_progression (k : ℝ) :
  ∃ (x y z : ℝ),
    x + y + z = k ∧
    y - x = z - y ∧
    y^2 = x * (z + k/6) ∧
    ((x = k/6 ∧ y = k/3 ∧ z = k/2) ∨ (x = 2*k/3 ∧ y = k/3 ∧ z = 0)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l2723_272349
