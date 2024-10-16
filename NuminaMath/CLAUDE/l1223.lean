import Mathlib

namespace NUMINAMATH_CALUDE_four_circles_theorem_l1223_122376

/-- Represents a square piece of paper -/
structure Paper :=
  (side : ℝ)
  (is_square : side > 0)

/-- Represents the state of the paper after folding and corner removal -/
structure FoldedPaper :=
  (original : Paper)
  (num_folds : ℕ)
  (corner_removed : Bool)

/-- Calculates the number of layers after folding -/
def num_layers (fp : FoldedPaper) : ℕ :=
  2^(fp.num_folds)

/-- Represents the hole pattern after unfolding -/
structure HolePattern :=
  (num_circles : ℕ)

/-- Function to determine the hole pattern after unfolding -/
def unfold_pattern (fp : FoldedPaper) : HolePattern :=
  { num_circles := if fp.corner_removed then (num_layers fp) / 4 else 0 }

theorem four_circles_theorem (p : Paper) :
  let fp := FoldedPaper.mk p 4 true
  (unfold_pattern fp).num_circles = 4 := by sorry

end NUMINAMATH_CALUDE_four_circles_theorem_l1223_122376


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_l1223_122327

theorem right_triangle_acute_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- sum of angles in a triangle is 180°
  α = 90 →           -- one angle is 90° (right angle)
  β = 20 →           -- given angle is 20°
  γ = 70 :=          -- prove that the other acute angle is 70°
by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_l1223_122327


namespace NUMINAMATH_CALUDE_day_temperature_difference_l1223_122312

def temperature_difference (lowest highest : ℤ) : ℤ :=
  highest - lowest

theorem day_temperature_difference :
  let lowest : ℤ := -15
  let highest : ℤ := 3
  temperature_difference lowest highest = 18 := by
  sorry

end NUMINAMATH_CALUDE_day_temperature_difference_l1223_122312


namespace NUMINAMATH_CALUDE_tetrahedron_pigeonhole_l1223_122351

/-- Represents the three possible states a point can be in -/
inductive PointState
  | Type1
  | Type2
  | Outside

/-- Represents a tetrahedron with vertices labeled by their state -/
structure Tetrahedron :=
  (vertices : Fin 4 → PointState)

/-- Theorem statement -/
theorem tetrahedron_pigeonhole (t : Tetrahedron) : 
  ∃ (i j : Fin 4), i ≠ j ∧ t.vertices i = t.vertices j :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_pigeonhole_l1223_122351


namespace NUMINAMATH_CALUDE_expression_evaluation_l1223_122356

theorem expression_evaluation : (-3)^6 / 3^4 - 4^3 * 2^2 + 9^2 = -166 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1223_122356


namespace NUMINAMATH_CALUDE_quiz_probability_l1223_122378

theorem quiz_probability : 
  let mcq_prob : ℚ := 1 / 3  -- Probability of correct MCQ answer
  let tf_prob : ℚ := 1 / 2   -- Probability of correct True/False answer
  mcq_prob * tf_prob * tf_prob = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_quiz_probability_l1223_122378


namespace NUMINAMATH_CALUDE_apple_trees_count_l1223_122374

theorem apple_trees_count (total_trees orange_trees : ℕ) 
  (h1 : total_trees = 74)
  (h2 : orange_trees = 27) :
  total_trees - orange_trees = 47 := by
  sorry

end NUMINAMATH_CALUDE_apple_trees_count_l1223_122374


namespace NUMINAMATH_CALUDE_mark_vaccine_waiting_time_l1223_122396

/-- Calculates the total waiting time in minutes for Mark's vaccine appointments and effectiveness periods -/
def total_waiting_time : ℕ :=
  let first_vaccine_wait := 4
  let second_vaccine_wait := 20
  let secondary_first_dose_wait := 30 + 10
  let secondary_second_dose_wait := 14 + 3
  let effectiveness_wait := 21
  let total_days := first_vaccine_wait + second_vaccine_wait + secondary_first_dose_wait + 
                    secondary_second_dose_wait + effectiveness_wait
  total_days * 24 * 60

theorem mark_vaccine_waiting_time :
  total_waiting_time = 146880 := by
  sorry

end NUMINAMATH_CALUDE_mark_vaccine_waiting_time_l1223_122396


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1223_122347

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 6*x₁ + 5 = 0) → (x₂^2 - 6*x₂ + 5 = 0) → x₁ + x₂ = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1223_122347


namespace NUMINAMATH_CALUDE_no_solution_exists_l1223_122313

theorem no_solution_exists : 
  ¬ ∃ (a b : ℕ+), a * b + 75 = 15 * Nat.lcm a b + 10 * Nat.gcd a b :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1223_122313


namespace NUMINAMATH_CALUDE_circle_radius_ratio_l1223_122387

theorem circle_radius_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (area_ratio : π * r₂^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_ratio_l1223_122387


namespace NUMINAMATH_CALUDE_rotated_square_height_l1223_122388

theorem rotated_square_height (square_side : Real) (rotation_angle : Real) : 
  square_side = 2 ∧ rotation_angle = π / 6 →
  let diagonal := square_side * Real.sqrt 2
  let height_above_center := (diagonal / 2) * Real.sin rotation_angle
  let initial_center_height := square_side / 2
  initial_center_height + height_above_center = 1 + Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_rotated_square_height_l1223_122388


namespace NUMINAMATH_CALUDE_time_to_peak_for_given_velocity_l1223_122382

/-- The time it takes for a ball to reach its peak height when thrown upwards -/
def time_to_peak_height (v : ℝ) : ℝ := 4 * v^2

/-- Theorem: The time to reach peak height for a ball thrown with initial velocity 1.25 m/s is 6.25 seconds -/
theorem time_to_peak_for_given_velocity :
  time_to_peak_height 1.25 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_time_to_peak_for_given_velocity_l1223_122382


namespace NUMINAMATH_CALUDE_three_even_out_of_five_probability_l1223_122342

/-- A fair 10-sided die -/
def TenSidedDie : Type := Fin 10

/-- The probability of rolling an even number on a 10-sided die -/
def probEven : ℚ := 1/2

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of dice that should show an even number -/
def numEven : ℕ := 3

/-- The probability of exactly three out of five 10-sided dice showing an even number -/
def probThreeEvenOutOfFive : ℚ := 5/16

theorem three_even_out_of_five_probability :
  (Nat.choose numDice numEven : ℚ) * probEven^numEven * (1 - probEven)^(numDice - numEven) = probThreeEvenOutOfFive :=
sorry

end NUMINAMATH_CALUDE_three_even_out_of_five_probability_l1223_122342


namespace NUMINAMATH_CALUDE_ryan_english_hours_l1223_122325

/-- The number of hours Ryan spends on learning Spanish -/
def spanish_hours : ℕ := 4

/-- The additional hours Ryan spends on learning English compared to Spanish -/
def additional_english_hours : ℕ := 3

/-- The number of hours Ryan spends on learning English -/
def english_hours : ℕ := spanish_hours + additional_english_hours

theorem ryan_english_hours : english_hours = 7 := by
  sorry

end NUMINAMATH_CALUDE_ryan_english_hours_l1223_122325


namespace NUMINAMATH_CALUDE_parallel_line_slope_l1223_122384

/-- Given a line parallel to 3x - 6y = 12, its slope is 1/2 -/
theorem parallel_line_slope :
  ∀ (m : ℚ) (b : ℚ), (∃ (k : ℚ), 3 * x - 6 * (m * x + b) = k) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l1223_122384


namespace NUMINAMATH_CALUDE_tan_pi_fourth_equals_one_l1223_122328

theorem tan_pi_fourth_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_fourth_equals_one_l1223_122328


namespace NUMINAMATH_CALUDE_first_digit_base_9_of_628_l1223_122364

/-- The first digit of the base 9 representation of a number -/
def first_digit_base_9 (n : ℕ) : ℕ :=
  if n < 9 then n else first_digit_base_9 (n / 9)

/-- The number in base 10 -/
def number : ℕ := 628

theorem first_digit_base_9_of_628 :
  first_digit_base_9 number = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_base_9_of_628_l1223_122364


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1223_122397

/-- The lateral surface area of a cylinder with a square axial cross-section -/
theorem cylinder_lateral_surface_area (s : ℝ) (h : s = 10) :
  let circumference := s * Real.pi
  let height := s
  height * circumference = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1223_122397


namespace NUMINAMATH_CALUDE_train_and_car_numbers_l1223_122353

/-- Represents a digit in the range 0 to 9 -/
def Digit := Fin 10

/-- Represents a mapping from characters to digits -/
def CodeMap := Char → Digit

/-- Checks if a CodeMap is valid (injective) -/
def isValidCodeMap (m : CodeMap) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → m c1 ≠ m c2

/-- Converts a string to a number using a CodeMap -/
def stringToNumber (s : String) (m : CodeMap) : ℕ :=
  s.foldl (fun acc c => acc * 10 + (m c).val) 0

/-- The main theorem -/
theorem train_and_car_numbers : ∃ (m : CodeMap),
  isValidCodeMap m ∧
  stringToNumber "SECRET" m - stringToNumber "OPEN" m = stringToNumber "ANSWER" m - stringToNumber "YOUR" m ∧
  stringToNumber "SECRET" m - stringToNumber "OPENED" m = 20010 ∧
  stringToNumber "TRAIN" m = 392 ∧
  stringToNumber "CAR" m = 2 := by
  sorry

end NUMINAMATH_CALUDE_train_and_car_numbers_l1223_122353


namespace NUMINAMATH_CALUDE_max_value_fraction_sum_l1223_122373

theorem max_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_sum_l1223_122373


namespace NUMINAMATH_CALUDE_rectangle_rhombus_ratio_l1223_122357

theorem rectangle_rhombus_ratio {a b c : ℝ} (h_perimeter : a + b = 2*c) (h_area : a*b = c^2/2) :
  a/b = 3 + 2*Real.sqrt 2 ∨ a/b = 3 - 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_rhombus_ratio_l1223_122357


namespace NUMINAMATH_CALUDE_sum_of_roots_l1223_122303

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 9*a^2 + 26*a - 40 = 0)
  (hb : 2*b^3 - 18*b^2 + 22*b - 30 = 0) : 
  a + b = Real.rpow 45 (1/3) + Real.rpow 22.5 (1/3) + 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1223_122303


namespace NUMINAMATH_CALUDE_exponential_inequality_l1223_122380

theorem exponential_inequality (a b c : ℝ) : 
  a^b > a^c ∧ a^c > 1 ∧ b < c → b < c ∧ c < 0 ∧ 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1223_122380


namespace NUMINAMATH_CALUDE_trig_values_for_point_l1223_122393

/-- Given a point P(-√3, m) on the terminal side of angle α, where m ≠ 0 and sin α = (√2 * m) / 4,
    prove the values of m, cos α, and tan α. -/
theorem trig_values_for_point (m : ℝ) (α : ℝ) (h1 : m ≠ 0) (h2 : Real.sin α = (Real.sqrt 2 * m) / 4) :
  (m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
  Real.cos α = -Real.sqrt 6 / 4 ∧
  (m > 0 → Real.tan α = -Real.sqrt 15 / 3) ∧
  (m < 0 → Real.tan α = Real.sqrt 15 / 3) := by
  sorry

end NUMINAMATH_CALUDE_trig_values_for_point_l1223_122393


namespace NUMINAMATH_CALUDE_dress_trim_cuff_length_l1223_122320

/-- Proves that the length of each cuff is 50 cm given the dress trimming conditions --/
theorem dress_trim_cuff_length :
  let hem_length : ℝ := 300
  let waist_length : ℝ := hem_length / 3
  let neck_ruffles : ℕ := 5
  let ruffle_length : ℝ := 20
  let lace_cost_per_meter : ℝ := 6
  let total_spent : ℝ := 36
  let total_lace_length : ℝ := total_spent / lace_cost_per_meter * 100
  let hem_waist_neck_length : ℝ := hem_length + waist_length + (neck_ruffles : ℝ) * ruffle_length
  let cuff_total_length : ℝ := total_lace_length - hem_waist_neck_length
  cuff_total_length / 2 = 50 := by sorry

end NUMINAMATH_CALUDE_dress_trim_cuff_length_l1223_122320


namespace NUMINAMATH_CALUDE_right_triangle_trig_identity_l1223_122311

theorem right_triangle_trig_identity (A B C : ℝ) (h1 : 0 < A) (h2 : A < π / 2) :
  B = π / 2 →
  3 * Real.sin A = 4 * Real.cos A + Real.tan A →
  Real.sin A = 2 * Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_trig_identity_l1223_122311


namespace NUMINAMATH_CALUDE_vector_parallelism_l1223_122375

theorem vector_parallelism (m : ℚ) : 
  let a : Fin 2 → ℚ := ![(-1), 2]
  let b : Fin 2 → ℚ := ![m, 1]
  (∃ (k : ℚ), k ≠ 0 ∧ (a + 2 • b) = k • (2 • a - b)) → m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallelism_l1223_122375


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l1223_122335

theorem consecutive_even_numbers_sum (a : ℤ) : 
  (∃ (x : ℤ), 
    (x = a) ∧ 
    (x + (x + 2) + (x + 4) + (x + 6) = 52)) → 
  (a + 4 = 14) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l1223_122335


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1223_122350

theorem complex_equation_solution (b : ℝ) : (2 + b * Complex.I) * Complex.I = 2 + 2 * Complex.I → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1223_122350


namespace NUMINAMATH_CALUDE_rectangle_area_l1223_122300

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 5 → length = 4 * width → width * length = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1223_122300


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1223_122399

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1223_122399


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l1223_122339

theorem empty_solution_set_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l1223_122339


namespace NUMINAMATH_CALUDE_odd_square_sum_parity_l1223_122354

theorem odd_square_sum_parity (n m : ℤ) (h : Odd (n^2 + m^2)) :
  ¬(Even n ∧ Even m) ∧ ¬(Odd n ∧ Odd m) := by
  sorry

end NUMINAMATH_CALUDE_odd_square_sum_parity_l1223_122354


namespace NUMINAMATH_CALUDE_valid_squares_count_l1223_122336

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  position : Nat × Nat

/-- Represents a 10x10 checkerboard with alternating black and white squares -/
def Checkerboard : Type := Unit

/-- Returns true if the given square contains at least 7 black squares -/
def hasAtLeastSevenBlackSquares (board : Checkerboard) (square : Square) : Bool :=
  sorry

/-- Counts the number of distinct squares on the board that contain at least 7 black squares -/
def countValidSquares (board : Checkerboard) : Nat :=
  sorry

/-- The main theorem stating that there are 116 valid squares -/
theorem valid_squares_count (board : Checkerboard) :
  countValidSquares board = 116 := by
  sorry

end NUMINAMATH_CALUDE_valid_squares_count_l1223_122336


namespace NUMINAMATH_CALUDE_vasya_driving_distance_l1223_122341

theorem vasya_driving_distance 
  (total_distance : ℝ) 
  (anton_distance vasya_distance sasha_distance dima_distance : ℝ)
  (h1 : anton_distance = vasya_distance / 2)
  (h2 : sasha_distance = anton_distance + dima_distance)
  (h3 : dima_distance = total_distance / 10)
  (h4 : anton_distance + vasya_distance + sasha_distance + dima_distance = total_distance)
  : vasya_distance = (2 / 5) * total_distance := by
  sorry

end NUMINAMATH_CALUDE_vasya_driving_distance_l1223_122341


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1223_122365

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1223_122365


namespace NUMINAMATH_CALUDE_diana_erasers_l1223_122394

/-- Given that Diana shares her erasers among 48 friends and each friend gets 80 erasers,
    prove that Diana has 3840 erasers. -/
theorem diana_erasers : ℕ → ℕ → ℕ → Prop :=
  fun num_friends erasers_per_friend total_erasers =>
    (num_friends = 48) →
    (erasers_per_friend = 80) →
    (total_erasers = num_friends * erasers_per_friend) →
    total_erasers = 3840

/-- Proof of the theorem -/
lemma diana_erasers_proof : diana_erasers 48 80 3840 := by
  sorry

end NUMINAMATH_CALUDE_diana_erasers_l1223_122394


namespace NUMINAMATH_CALUDE_profit_percentage_l1223_122381

theorem profit_percentage (C S : ℝ) (h : 19 * C = 16 * S) : 
  (S - C) / C * 100 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l1223_122381


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1223_122389

theorem trigonometric_identity : 
  100 * (Real.sin (253 * π / 180) * Real.sin (313 * π / 180) + 
         Real.sin (163 * π / 180) * Real.sin (223 * π / 180)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1223_122389


namespace NUMINAMATH_CALUDE_mountain_elevation_difference_l1223_122379

/-- The elevation difference between two mountains -/
def elevation_difference (h b : ℕ) : ℕ := h - b

/-- Proves that the elevation difference between two mountains is 2500 feet -/
theorem mountain_elevation_difference :
  ∃ (h b : ℕ),
    h = 10000 ∧
    3 * h = 4 * b ∧
    elevation_difference h b = 2500 := by
  sorry

end NUMINAMATH_CALUDE_mountain_elevation_difference_l1223_122379


namespace NUMINAMATH_CALUDE_impossible_tiling_after_replacement_l1223_122385

/-- Represents a tile type -/
inductive Tile
| TwoByTwo
| OneByFour

/-- Represents a tiling of a rectangular grid -/
def Tiling := List Tile

/-- Represents a rectangular grid -/
structure Grid :=
(rows : Nat)
(cols : Nat)

/-- Checks if a tiling is valid for a given grid -/
def isValidTiling (g : Grid) (t : Tiling) : Prop :=
  -- Definition omitted
  sorry

/-- Checks if a grid can be tiled with 2x2 and 1x4 tiles -/
def canBeTiled (g : Grid) : Prop :=
  ∃ t : Tiling, isValidTiling g t

/-- Represents the operation of replacing one 2x2 tile with a 1x4 tile -/
def replaceTile (t : Tiling) : Tiling :=
  -- Definition omitted
  sorry

/-- Main theorem: If a grid can be tiled, replacing one 2x2 tile with a 1x4 tile makes it impossible to tile -/
theorem impossible_tiling_after_replacement (g : Grid) :
  canBeTiled g → ¬(∃ t : Tiling, isValidTiling g (replaceTile t)) :=
by
  sorry

end NUMINAMATH_CALUDE_impossible_tiling_after_replacement_l1223_122385


namespace NUMINAMATH_CALUDE_cubic_equation_one_real_root_l1223_122337

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, x^3 - 6*x^2 + 9*x - 10 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_one_real_root_l1223_122337


namespace NUMINAMATH_CALUDE_no_constant_difference_integer_l1223_122314

theorem no_constant_difference_integer (x : ℤ) : 
  ¬∃ (k : ℤ), 
    (x^2 - 4*x + 5) - (2*x - 6) = k ∧ 
    (4*x - 8) - (x^2 - 4*x + 5) = k ∧ 
    (3*x^2 - 12*x + 11) - (4*x - 8) = k :=
by sorry

end NUMINAMATH_CALUDE_no_constant_difference_integer_l1223_122314


namespace NUMINAMATH_CALUDE_h_of_two_equals_fifteen_l1223_122318

theorem h_of_two_equals_fifteen (h : ℝ → ℝ) 
  (h_def : ∀ x : ℝ, h (3 * x - 4) = 4 * x + 7) : 
  h 2 = 15 := by
sorry

end NUMINAMATH_CALUDE_h_of_two_equals_fifteen_l1223_122318


namespace NUMINAMATH_CALUDE_high_school_harriers_loss_percentage_l1223_122306

theorem high_school_harriers_loss_percentage
  (total_games : ℝ)
  (games_won : ℝ)
  (games_lost : ℝ)
  (games_tied : ℝ)
  (h1 : games_won / games_lost = 5 / 3)
  (h2 : games_tied = 0.2 * total_games)
  (h3 : total_games = games_won + games_lost + games_tied) :
  games_lost / total_games = 0.3 := by
sorry

end NUMINAMATH_CALUDE_high_school_harriers_loss_percentage_l1223_122306


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l1223_122383

theorem quadratic_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Ioo 2 3, Monotone (fun x => x^2 - 2*a*x + 1))
  ↔ (a ≤ 2 ∨ a ≥ 3) := by sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l1223_122383


namespace NUMINAMATH_CALUDE_equation_solution_l1223_122377

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (9 * x)^18 - (18 * x)^9 = 0 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1223_122377


namespace NUMINAMATH_CALUDE_triangle_product_inequality_l1223_122324

/-- Triangle structure with sides a, b, c, perimeter P, and inscribed circle radius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  P : ℝ
  r : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_P : 0 < P
  pos_r : 0 < r
  perimeter_def : P = a + b + c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The product of any two sides of a triangle is not less than
    the product of its perimeter and the radius of its inscribed circle -/
theorem triangle_product_inequality (t : Triangle) : t.a * t.b ≥ t.P * t.r := by
  sorry

end NUMINAMATH_CALUDE_triangle_product_inequality_l1223_122324


namespace NUMINAMATH_CALUDE_isabel_spending_ratio_l1223_122343

/-- Given Isabel's initial amount, toy purchase, and final remaining amount,
    prove that the ratio of book cost to money after toy purchase is 1:2 -/
theorem isabel_spending_ratio (initial_amount : ℕ) (remaining_amount : ℕ)
    (h1 : initial_amount = 204)
    (h2 : remaining_amount = 51) :
  let toy_cost : ℕ := initial_amount / 2
  let after_toy : ℕ := initial_amount - toy_cost
  let book_cost : ℕ := after_toy - remaining_amount
  (book_cost : ℚ) / after_toy = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_isabel_spending_ratio_l1223_122343


namespace NUMINAMATH_CALUDE_blocks_color_theorem_l1223_122390

theorem blocks_color_theorem (total_blocks : ℕ) (blocks_per_color : ℕ) (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) :
  total_blocks / blocks_per_color = 7 := by
  sorry

end NUMINAMATH_CALUDE_blocks_color_theorem_l1223_122390


namespace NUMINAMATH_CALUDE_original_salary_approximation_l1223_122346

/-- Calculates the final salary after applying a sequence of percentage changes --/
def final_salary (original : ℝ) : ℝ :=
  original * 1.12 * 0.93 * 1.09 * 0.94

/-- Theorem stating that the original salary is approximately 981.47 --/
theorem original_salary_approximation :
  ∃ (S : ℝ), S > 0 ∧ final_salary S = 1212 ∧ abs (S - 981.47) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_original_salary_approximation_l1223_122346


namespace NUMINAMATH_CALUDE_count_odd_numbers_300_to_600_l1223_122352

theorem count_odd_numbers_300_to_600 : 
  (Finset.filter (fun n => n % 2 = 1 ∧ n > 300 ∧ n < 600) (Finset.range 600)).card = 149 := by
  sorry

end NUMINAMATH_CALUDE_count_odd_numbers_300_to_600_l1223_122352


namespace NUMINAMATH_CALUDE_valid_arrays_count_l1223_122305

/-- A 3x3 array with entries of 1 or -1 -/
def ValidArray : Type := Matrix (Fin 3) (Fin 3) Int

/-- Predicate to check if an entry is valid (1 or -1) -/
def isValidEntry (x : Int) : Prop := x = 1 ∨ x = -1

/-- Predicate to check if all entries in the array are valid -/
def hasValidEntries (arr : ValidArray) : Prop :=
  ∀ i j, isValidEntry (arr i j)

/-- Predicate to check if the sum of a row is zero -/
def rowSumZero (arr : ValidArray) (i : Fin 3) : Prop :=
  (arr i 0) + (arr i 1) + (arr i 2) = 0

/-- Predicate to check if the sum of a column is zero -/
def colSumZero (arr : ValidArray) (j : Fin 3) : Prop :=
  (arr 0 j) + (arr 1 j) + (arr 2 j) = 0

/-- Predicate to check if an array satisfies all conditions -/
def isValidArray (arr : ValidArray) : Prop :=
  hasValidEntries arr ∧
  (∀ i, rowSumZero arr i) ∧
  (∀ j, colSumZero arr j)

/-- The main theorem: there are exactly 6 valid arrays -/
theorem valid_arrays_count :
  ∃! (s : Finset ValidArray), (∀ arr ∈ s, isValidArray arr) ∧ s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_valid_arrays_count_l1223_122305


namespace NUMINAMATH_CALUDE_simplify_expression_calculate_expression_find_expression_value_l1223_122366

-- Question 1
theorem simplify_expression (a b : ℝ) :
  2 * (a - b)^2 - 4 * (a - b)^2 + 7 * (a - b)^2 = 5 * (a - b)^2 := by sorry

-- Question 2
theorem calculate_expression (a b : ℝ) (h : a^2 - 2*b^2 - 3 = 0) :
  -3*a^2 + 6*b^2 + 2032 = 2023 := by sorry

-- Question 3
theorem find_expression_value (a b : ℝ) (h1 : a^2 + 2*a*b = 15) (h2 : b^2 + 2*a*b = 6) :
  2*a^2 - 4*b^2 - 4*a*b = 6 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_calculate_expression_find_expression_value_l1223_122366


namespace NUMINAMATH_CALUDE_equation_solution_l1223_122326

theorem equation_solution : 
  ∃ x : ℝ, (2 / (x + 1) = 3 / (4 - x)) ∧ (x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1223_122326


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1223_122371

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | x + 1 ≥ 0 ∧ (x - 1) / 2 < 1}
  S = {x | -1 ≤ x ∧ x < 3} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1223_122371


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1223_122319

def A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1223_122319


namespace NUMINAMATH_CALUDE_chessboard_sum_zero_l1223_122386

/-- Represents a chessboard with signed numbers -/
def SignedChessboard := Fin 8 → Fin 8 → Int

/-- Checks if a row has exactly four positive and four negative numbers -/
def valid_row (board : SignedChessboard) (row : Fin 8) : Prop :=
  (Finset.filter (λ col => board row col > 0) Finset.univ).card = 4 ∧
  (Finset.filter (λ col => board row col < 0) Finset.univ).card = 4

/-- Checks if a column has exactly four positive and four negative numbers -/
def valid_column (board : SignedChessboard) (col : Fin 8) : Prop :=
  (Finset.filter (λ row => board row col > 0) Finset.univ).card = 4 ∧
  (Finset.filter (λ row => board row col < 0) Finset.univ).card = 4

/-- Checks if the board contains numbers from 1 to 64 with signs -/
def valid_numbers (board : SignedChessboard) : Prop :=
  ∀ n : Fin 64, ∃ (i j : Fin 8), |board i j| = n.val + 1

/-- The main theorem: sum of all numbers on a valid chessboard is zero -/
theorem chessboard_sum_zero (board : SignedChessboard)
  (h_rows : ∀ row, valid_row board row)
  (h_cols : ∀ col, valid_column board col)
  (h_nums : valid_numbers board) :
  (Finset.univ.sum (λ (i : Fin 8) => Finset.univ.sum (λ (j : Fin 8) => board i j))) = 0 :=
sorry

end NUMINAMATH_CALUDE_chessboard_sum_zero_l1223_122386


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l1223_122309

/-- Prove that for a hyperbola with given properties, its parameters satisfy specific values -/
theorem hyperbola_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a^2 + b^2) / a^2 = 4 →  -- eccentricity is 2
  (a * b / Real.sqrt (a^2 + b^2))^2 = 3 →  -- asymptote is tangent to the circle
  a^2 = 4 ∧ b^2 = 12 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l1223_122309


namespace NUMINAMATH_CALUDE_semicircle_radius_l1223_122398

/-- The radius of a semicircle with perimeter 180 cm is equal to 180 / (π + 2) cm. -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 180) :
  ∃ r : ℝ, r = perimeter / (Real.pi + 2) ∧ r * (Real.pi + 2) = perimeter := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l1223_122398


namespace NUMINAMATH_CALUDE_product_of_square_roots_l1223_122315

theorem product_of_square_roots (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (7 * q^3) * Real.sqrt (8 * q^5) = 210 * q^4 * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l1223_122315


namespace NUMINAMATH_CALUDE_sin_cos_tan_product_l1223_122304

theorem sin_cos_tan_product : 
  Real.sin (4/3 * Real.pi) * Real.cos (5/6 * Real.pi) * Real.tan (-4/3 * Real.pi) = -(3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_tan_product_l1223_122304


namespace NUMINAMATH_CALUDE_gift_packaging_combinations_l1223_122358

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

end NUMINAMATH_CALUDE_gift_packaging_combinations_l1223_122358


namespace NUMINAMATH_CALUDE_squareable_numbers_l1223_122330

/-- A natural number is squareable if the numbers from 1 to n can be arranged
    such that each number plus its index is a perfect square. -/
def Squareable (n : ℕ) : Prop :=
  ∃ (σ : Fin n → Fin n), Function.Bijective σ ∧
    ∀ (i : Fin n), ∃ (k : ℕ), (σ i).val + i.val + 1 = k^2

theorem squareable_numbers : 
  ¬ Squareable 7 ∧ Squareable 9 ∧ ¬ Squareable 11 ∧ Squareable 15 :=
sorry

end NUMINAMATH_CALUDE_squareable_numbers_l1223_122330


namespace NUMINAMATH_CALUDE_fraction_equality_l1223_122367

theorem fraction_equality : (1 / 5 - 1 / 6) / (1 / 3 - 1 / 4) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1223_122367


namespace NUMINAMATH_CALUDE_cos120_plus_sin_neg45_l1223_122349

theorem cos120_plus_sin_neg45 : 
  Real.cos (120 * π / 180) + Real.sin (-45 * π / 180) = - (1 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos120_plus_sin_neg45_l1223_122349


namespace NUMINAMATH_CALUDE_range_of_m_for_trig_equation_l1223_122310

theorem range_of_m_for_trig_equation :
  ∀ α m : ℝ,
  (∃ α, Real.cos α - Real.sqrt 3 * Real.sin α = (4 * m - 6) / (4 - m)) →
  -1 ≤ m ∧ m ≤ 7/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_trig_equation_l1223_122310


namespace NUMINAMATH_CALUDE_sqrt_pattern_l1223_122372

theorem sqrt_pattern (n : ℕ) (hn : n > 0) : 
  Real.sqrt (1 + 1 / (n^2 : ℝ) + 1 / ((n+1)^2 : ℝ)) = (n^2 + n + 1 : ℝ) / (n * (n+1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l1223_122372


namespace NUMINAMATH_CALUDE_parallel_line_plane_condition_l1223_122333

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (subset_of : Line → Plane → Prop)

-- State the theorem
theorem parallel_line_plane_condition
  (m n : Line) (α : Plane)
  (h1 : subset_of n α)
  (h2 : ¬ subset_of m α) :
  (∀ m n, parallel_lines m n → parallel_line_plane m α) ∧
  ¬(∀ m α, parallel_line_plane m α → parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_condition_l1223_122333


namespace NUMINAMATH_CALUDE_prob_at_least_two_girls_is_two_sevenths_l1223_122340

def total_students : ℕ := 8
def boys : ℕ := 5
def girls : ℕ := 3
def selected : ℕ := 3

def prob_at_least_two_girls : ℚ :=
  (Nat.choose girls 2 * Nat.choose boys 1 + Nat.choose girls 3 * Nat.choose boys 0) /
  Nat.choose total_students selected

theorem prob_at_least_two_girls_is_two_sevenths :
  prob_at_least_two_girls = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_girls_is_two_sevenths_l1223_122340


namespace NUMINAMATH_CALUDE_expression_evaluation_l1223_122392

theorem expression_evaluation (x y : ℚ) (hx : x = 5) (hy : y = 6) :
  (2 / y) / (2 / x) * 3 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1223_122392


namespace NUMINAMATH_CALUDE_prime_equation_solution_l1223_122334

theorem prime_equation_solution (p : ℕ) (hp : Prime p) :
  (∃ (n : ℤ) (k m : ℕ+), (m * k^2 + 2) * p - (m^2 + 2 * k^2) = n^2 * (m * p + 2)) →
  p = 3 ∨ p % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l1223_122334


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l1223_122345

theorem tan_seventeen_pi_fourths : Real.tan (17 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l1223_122345


namespace NUMINAMATH_CALUDE_quadratic_intersection_l1223_122344

/-- 
Given two functions f(x) = bx² + 5x + 2 and g(x) = -2x - 2,
prove that they intersect at exactly one point when b = 49/16.
-/
theorem quadratic_intersection (b : ℝ) : 
  (∃! x : ℝ, b * x^2 + 5 * x + 2 = -2 * x - 2) ↔ b = 49/16 := by
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l1223_122344


namespace NUMINAMATH_CALUDE_set_intersection_complement_equality_l1223_122355

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 3}

-- Define set N
def N : Set ℝ := {x | x ≤ 2}

-- Theorem statement
theorem set_intersection_complement_equality :
  M ∩ (U \ N) = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_set_intersection_complement_equality_l1223_122355


namespace NUMINAMATH_CALUDE_carrot_weight_problem_l1223_122369

/-- Prove that given 20 carrots weighing 3.64 kg in total, and 4 carrots with an average weight of 190 grams are removed, the average weight of the remaining 16 carrots is 180 grams. -/
theorem carrot_weight_problem (total_weight : ℝ) (removed_avg : ℝ) :
  total_weight = 3.64 →
  removed_avg = 190 →
  (total_weight * 1000 - 4 * removed_avg) / 16 = 180 := by
sorry

end NUMINAMATH_CALUDE_carrot_weight_problem_l1223_122369


namespace NUMINAMATH_CALUDE_popton_bus_toes_l1223_122362

/-- Represents a race on planet Popton -/
inductive Race
  | Hoopit
  | Neglart

/-- Number of hands for each race -/
def hands (r : Race) : Nat :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5

/-- Number of toes per hand for each race -/
def toes_per_hand (r : Race) : Nat :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of students of each race on the bus -/
def students (r : Race) : Nat :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8

/-- Total number of toes for all students of a given race on the bus -/
def total_toes (r : Race) : Nat :=
  students r * hands r * toes_per_hand r

/-- The total number of toes on the Popton school bus -/
theorem popton_bus_toes :
  total_toes Race.Hoopit + total_toes Race.Neglart = 164 := by
  sorry

end NUMINAMATH_CALUDE_popton_bus_toes_l1223_122362


namespace NUMINAMATH_CALUDE_system_solution_l1223_122391

theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, 2 * x - y = 5 * k + 6 ∧ 4 * x + 7 * y = k ∧ x + y = 2023) → k = 2022 :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1223_122391


namespace NUMINAMATH_CALUDE_road_building_time_l1223_122308

/-- Given that 60 workers can build a road in 5 days, prove that 40 workers
    working at the same rate will take 7.5 days to build the same road. -/
theorem road_building_time (workers_initial : ℕ) (days_initial : ℝ)
    (workers_new : ℕ) (days_new : ℝ) : 
    workers_initial = 60 → days_initial = 5 → workers_new = 40 → 
    (workers_initial : ℝ) * days_initial = workers_new * days_new →
    days_new = 7.5 := by
  sorry

#check road_building_time

end NUMINAMATH_CALUDE_road_building_time_l1223_122308


namespace NUMINAMATH_CALUDE_knocks_to_knicks_conversion_l1223_122302

/-- Conversion rate between knicks and knacks -/
def knicks_to_knacks : ℚ := 3 / 8

/-- Conversion rate between knacks and knocks -/
def knacks_to_knocks : ℚ := 6 / 5

/-- The number of knocks we want to convert -/
def target_knocks : ℚ := 30

theorem knocks_to_knicks_conversion :
  target_knocks * knacks_to_knocks⁻¹ * knicks_to_knacks⁻¹ = 200 / 3 :=
sorry

end NUMINAMATH_CALUDE_knocks_to_knicks_conversion_l1223_122302


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l1223_122307

theorem continued_fraction_evaluation :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = (3 + Real.sqrt 69) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l1223_122307


namespace NUMINAMATH_CALUDE_rational_coefficient_terms_count_l1223_122332

theorem rational_coefficient_terms_count :
  let expression := (x : ℝ) * (5 ^ (1/4 : ℝ)) + (y : ℝ) * (7 ^ (1/2 : ℝ))
  let power := 500
  let is_rational_coeff (k : ℕ) := (k % 4 = 0) ∧ ((power - k) % 2 = 0)
  (Finset.filter is_rational_coeff (Finset.range (power + 1))).card = 126 :=
by sorry

end NUMINAMATH_CALUDE_rational_coefficient_terms_count_l1223_122332


namespace NUMINAMATH_CALUDE_decoration_price_increase_l1223_122338

def price_1990 : ℝ := 11500
def increase_1990_to_1996 : ℝ := 0.13
def increase_1996_to_2001 : ℝ := 0.20

def price_2001 : ℝ :=
  price_1990 * (1 + increase_1990_to_1996) * (1 + increase_1996_to_2001)

theorem decoration_price_increase : price_2001 = 15594 := by
  sorry

end NUMINAMATH_CALUDE_decoration_price_increase_l1223_122338


namespace NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l1223_122322

/-- An arithmetic progression where the sum of the first twenty terms
    is six times the sum of the first ten terms -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_condition : (20 * a + 190 * d) = 6 * (10 * a + 45 * d)

/-- The ratio of the first term to the common difference is 2 -/
theorem ratio_first_term_to_common_difference
  (ap : ArithmeticProgression) : ap.a / ap.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l1223_122322


namespace NUMINAMATH_CALUDE_trip_speed_calculation_l1223_122316

theorem trip_speed_calculation (v : ℝ) : 
  v > 0 → -- Ensuring speed is positive
  (35 / v + 35 / 24 = 70 / 32) → -- Average speed equation
  v = 48 := by
sorry

end NUMINAMATH_CALUDE_trip_speed_calculation_l1223_122316


namespace NUMINAMATH_CALUDE_investment_ratio_is_two_to_three_l1223_122348

/-- A partnership problem with three investors A, B, and C. -/
structure Partnership where
  /-- B's investment amount -/
  b_investment : ℝ
  /-- Total profit earned -/
  total_profit : ℝ
  /-- B's share of the profit -/
  b_share : ℝ
  /-- A's investment is 3 times B's investment -/
  a_investment_prop : ℝ := 3 * b_investment
  /-- Assumption that total_profit and b_share are positive -/
  h_positive : 0 < total_profit ∧ 0 < b_share

/-- The ratio of B's investment to C's investment in the partnership -/
def investment_ratio (p : Partnership) : ℚ × ℚ :=
  (2, 3)

/-- Theorem stating that the investment ratio is 2:3 given the partnership conditions -/
theorem investment_ratio_is_two_to_three (p : Partnership)
  (h1 : p.total_profit = 3300)
  (h2 : p.b_share = 600) :
  investment_ratio p = (2, 3) := by
  sorry

#check investment_ratio_is_two_to_three

end NUMINAMATH_CALUDE_investment_ratio_is_two_to_three_l1223_122348


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1223_122368

theorem pure_imaginary_complex_number (a : ℝ) : 
  (((2 : ℂ) - a * Complex.I) / (1 + Complex.I)).re = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1223_122368


namespace NUMINAMATH_CALUDE_pascal_row_12_left_half_sum_l1223_122361

/-- The sum of the left half of a row in Pascal's Triangle -/
def pascal_left_half_sum (n : ℕ) : ℕ :=
  2^n

/-- Row 12 of Pascal's Triangle -/
def pascal_row_12 : ℕ := 12

theorem pascal_row_12_left_half_sum :
  pascal_left_half_sum pascal_row_12 = 2048 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row_12_left_half_sum_l1223_122361


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l1223_122331

theorem initial_number_of_persons 
  (average_weight_increase : ℝ) 
  (weight_of_leaving_person : ℝ) 
  (weight_of_new_person : ℝ) : 
  average_weight_increase = 4.5 ∧ 
  weight_of_leaving_person = 65 ∧ 
  weight_of_new_person = 74 → 
  (weight_of_new_person - weight_of_leaving_person) / average_weight_increase = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l1223_122331


namespace NUMINAMATH_CALUDE_complex_expression_proof_l1223_122321

theorem complex_expression_proof :
  let A : ℂ := 5 - 2*I
  let M : ℂ := -3 + 2*I
  let S : ℂ := 2*I
  let P : ℂ := 3
  2 * (A - M + S - P) = 10 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_proof_l1223_122321


namespace NUMINAMATH_CALUDE_carnival_earnings_example_l1223_122301

/-- Represents the earnings of a carnival snack booth over a period of days -/
def carnival_earnings (popcorn_sales : ℕ) (cotton_candy_multiplier : ℕ) (days : ℕ) (rent : ℕ) (ingredients_cost : ℕ) : ℕ :=
  let daily_total := popcorn_sales + popcorn_sales * cotton_candy_multiplier
  let total_revenue := daily_total * days
  let total_expenses := rent + ingredients_cost
  total_revenue - total_expenses

/-- Theorem stating that the carnival snack booth's earnings after expenses for 5 days is $895 -/
theorem carnival_earnings_example : carnival_earnings 50 3 5 30 75 = 895 := by
  sorry

end NUMINAMATH_CALUDE_carnival_earnings_example_l1223_122301


namespace NUMINAMATH_CALUDE_college_students_count_l1223_122360

theorem college_students_count :
  ∀ (students professors : ℕ),
  students = 15 * professors →
  students + professors = 40000 →
  students = 37500 :=
by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l1223_122360


namespace NUMINAMATH_CALUDE_line_circle_intersection_sufficient_not_necessary_condition_l1223_122323

theorem line_circle_intersection (k : ℝ) : 
  (∃ x y : ℝ, y = k * (x + 1) ∧ x^2 + y^2 - 2*x = 0) ↔ 
  (-Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3) :=
sorry

theorem sufficient_not_necessary_condition : 
  (∀ k : ℝ, -Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3 → 
    ∃ x y : ℝ, y = k * (x + 1) ∧ x^2 + y^2 - 2*x = 0) ∧
  (∃ k : ℝ, (k = -Real.sqrt 3 / 3 ∨ k = Real.sqrt 3 / 3) ∧
    ∃ x y : ℝ, y = k * (x + 1) ∧ x^2 + y^2 - 2*x = 0) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_sufficient_not_necessary_condition_l1223_122323


namespace NUMINAMATH_CALUDE_simon_practice_requirement_l1223_122317

def week1_hours : ℝ := 12
def week2_hours : ℝ := 16
def week3_hours : ℝ := 14
def total_weeks : ℝ := 4
def required_average : ℝ := 15

def fourth_week_hours : ℝ := 18

theorem simon_practice_requirement :
  (week1_hours + week2_hours + week3_hours + fourth_week_hours) / total_weeks = required_average :=
by sorry

end NUMINAMATH_CALUDE_simon_practice_requirement_l1223_122317


namespace NUMINAMATH_CALUDE_stock_value_change_l1223_122395

/-- Theorem: Stock Value Change over Two Days
    Given a stock that decreases in value by 25% on the first day and
    increases by 40% on the second day, prove that the overall
    percentage change is a 5% increase. -/
theorem stock_value_change (initial_value : ℝ) (initial_value_pos : initial_value > 0) :
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  (day2_value - initial_value) / initial_value = 0.05 := by
sorry

end NUMINAMATH_CALUDE_stock_value_change_l1223_122395


namespace NUMINAMATH_CALUDE_drink_ticket_cost_l1223_122359

/-- Proves that the cost of each drink ticket is $7 given Jenna's income and spending constraints -/
theorem drink_ticket_cost 
  (concert_ticket_cost : ℕ)
  (hourly_wage : ℕ)
  (weekly_hours : ℕ)
  (spending_percentage : ℚ)
  (num_drink_tickets : ℕ)
  (h1 : concert_ticket_cost = 181)
  (h2 : hourly_wage = 18)
  (h3 : weekly_hours = 30)
  (h4 : spending_percentage = 1/10)
  (h5 : num_drink_tickets = 5) :
  (((hourly_wage * weekly_hours * 4) * spending_percentage - concert_ticket_cost : ℚ) / num_drink_tickets : ℚ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_drink_ticket_cost_l1223_122359


namespace NUMINAMATH_CALUDE_circle_area_ratio_after_radius_increase_l1223_122363

theorem circle_area_ratio_after_radius_increase (r : ℝ) (hr : r > 0) :
  let original_area := π * r^2
  let new_radius := 1.5 * r
  let new_area := π * new_radius^2
  (original_area / new_area : ℝ) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_after_radius_increase_l1223_122363


namespace NUMINAMATH_CALUDE_root_transformation_l1223_122370

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (∀ x, x^3 - 5*x^2 + 10 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (∀ x, x^3 - 15*x^2 + 270 = 0 ↔ x = 3*r₁ ∨ x = 3*r₂ ∨ x = 3*r₃) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l1223_122370


namespace NUMINAMATH_CALUDE_triangle_theorem_l1223_122329

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sin t.A > Real.sin t.C)
  (h2 : t.a * t.c * Real.cos t.B = 2)
  (h3 : Real.cos t.B = 1/3)
  (h4 : t.b = 3) :
  t.a = 3 ∧ t.c = 2 ∧ Real.cos (t.B - t.C) = 23/27 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1223_122329
