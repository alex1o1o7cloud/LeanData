import Mathlib

namespace NUMINAMATH_CALUDE_f_properties_l2745_274588

def f (x m : ℝ) : ℝ := |x + 1| + |x + m + 1|

theorem f_properties (m : ℝ) :
  (∀ x, f x m ≥ |m - 2|) ↔ m ≥ 1 ∧
  (m ≤ 0 → ∀ x, ¬(f (-x) m < 2*m)) ∧
  (m > 0 → ∀ x, f (-x) m < 2*m ↔ 1 - m/2 < x ∧ x < 3*m/2 + 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2745_274588


namespace NUMINAMATH_CALUDE_f_zero_eq_zero_l2745_274555

-- Define the function f
def f : ℝ → ℝ := fun x => sorry

-- State the theorem
theorem f_zero_eq_zero :
  (∀ x : ℝ, f (x + 1) = x^2 + 2*x + 1) →
  f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_f_zero_eq_zero_l2745_274555


namespace NUMINAMATH_CALUDE_wood_length_proof_l2745_274547

/-- The original length of a piece of wood, given the length sawed off and the remaining length. -/
def original_length (sawed_off : ℝ) (remaining : ℝ) : ℝ := sawed_off + remaining

/-- Theorem stating that the original length of the wood is 0.41 meters. -/
theorem wood_length_proof :
  let sawed_off : ℝ := 0.33
  let remaining : ℝ := 0.08
  original_length sawed_off remaining = 0.41 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_proof_l2745_274547


namespace NUMINAMATH_CALUDE_no_nonnegative_solutions_l2745_274552

theorem no_nonnegative_solutions : ¬∃ x : ℝ, x ≥ 0 ∧ x^2 + 6*x + 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonnegative_solutions_l2745_274552


namespace NUMINAMATH_CALUDE_center_is_four_l2745_274530

-- Define the grid as a 3x3 matrix of natural numbers
def Grid := Matrix (Fin 3) (Fin 3) Nat

-- Define a predicate for consecutive numbers being adjacent
def consecutive_adjacent (g : Grid) : Prop := sorry

-- Define a function to get the edge numbers (excluding corners)
def edge_numbers (g : Grid) : List Nat := sorry

-- Define the sum of edge numbers
def edge_sum (g : Grid) : Nat := (edge_numbers g).sum

-- Define a predicate for the grid containing all numbers from 1 to 9
def contains_one_to_nine (g : Grid) : Prop := sorry

-- Define a function to get the center number
def center_number (g : Grid) : Nat := g 1 1

-- Main theorem
theorem center_is_four (g : Grid) 
  (h1 : consecutive_adjacent g)
  (h2 : edge_sum g = 28)
  (h3 : contains_one_to_nine g)
  (h4 : Even (center_number g)) :
  center_number g = 4 := by sorry

end NUMINAMATH_CALUDE_center_is_four_l2745_274530


namespace NUMINAMATH_CALUDE_perimeter_of_square_region_l2745_274545

theorem perimeter_of_square_region (total_area : ℝ) (num_squares : ℕ) (perimeter : ℝ) :
  total_area = 588 →
  num_squares = 14 →
  perimeter = 15 * Real.sqrt 42 :=
by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_square_region_l2745_274545


namespace NUMINAMATH_CALUDE_first_square_perimeter_l2745_274564

/-- Given two squares and a third square with specific properties, 
    prove that the perimeter of the first square is 24 meters. -/
theorem first_square_perimeter : 
  ∀ (s₁ s₂ s₃ : ℝ),
  (4 * s₂ = 32) →  -- Perimeter of second square is 32 m
  (4 * s₃ = 40) →  -- Perimeter of third square is 40 m
  (s₃^2 = s₁^2 + s₂^2) →  -- Area of third square equals sum of areas of first two squares
  (4 * s₁ = 24) :=  -- Perimeter of first square is 24 m
by
  sorry

#check first_square_perimeter

end NUMINAMATH_CALUDE_first_square_perimeter_l2745_274564


namespace NUMINAMATH_CALUDE_no_real_roots_m_range_l2745_274526

/-- A quadratic function with parameter m -/
def f (m x : ℝ) : ℝ := x^2 + m*x + (m+3)

/-- The discriminant of the quadratic function -/
def discriminant (m : ℝ) : ℝ := m^2 - 4*(m+3)

theorem no_real_roots_m_range (m : ℝ) :
  (∀ x, f m x ≠ 0) → m ∈ Set.Ioo (-2 : ℝ) 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_m_range_l2745_274526


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2745_274519

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 2}

-- Define set B
def B : Finset Nat := {2, 3, 4}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2745_274519


namespace NUMINAMATH_CALUDE_rod_length_theorem_l2745_274556

/-- The length of a rod in meters, given the number of pieces it can be cut into and the length of each piece in centimeters. -/
def rod_length_meters (num_pieces : ℕ) (piece_length_cm : ℕ) : ℚ :=
  (num_pieces * piece_length_cm : ℚ) / 100

/-- Theorem stating that a rod that can be cut into 50 pieces of 85 cm each is 42.5 meters long. -/
theorem rod_length_theorem : rod_length_meters 50 85 = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_theorem_l2745_274556


namespace NUMINAMATH_CALUDE_f_value_at_3_l2745_274537

theorem f_value_at_3 (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^7 + a*x^5 + b*x - 5) 
  (h2 : f (-3) = 5) : f 3 = -15 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l2745_274537


namespace NUMINAMATH_CALUDE_semicircle_triangle_area_ratio_l2745_274543

/-- Given a triangle ABC with sides in ratio 2:3:4 and an inscribed semicircle
    with diameter on the longest side, the ratio of the area of the semicircle
    to the area of the triangle is π√15 / 12 -/
theorem semicircle_triangle_area_ratio (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ratio : a / b = 2 / 3 ∧ b / c = 3 / 4) (h_triangle : (a + b > c) ∧ (b + c > a) ∧ (c + a > b)) :
  let s := (a + b + c) / 2
  let triangle_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let semicircle_area := π * c^2 / 8
  semicircle_area / triangle_area = π * Real.sqrt 15 / 12 := by
sorry

end NUMINAMATH_CALUDE_semicircle_triangle_area_ratio_l2745_274543


namespace NUMINAMATH_CALUDE_star_one_one_eq_neg_eleven_l2745_274569

/-- Definition of the * operation for rational numbers -/
def star (a b c : ℚ) (x y : ℚ) : ℚ := a * x + b * y + c

/-- Theorem: Given the conditions, 1 * 1 = -11 -/
theorem star_one_one_eq_neg_eleven 
  (a b c : ℚ) 
  (h1 : star a b c 3 5 = 15) 
  (h2 : star a b c 4 7 = 28) : 
  star a b c 1 1 = -11 := by
  sorry

end NUMINAMATH_CALUDE_star_one_one_eq_neg_eleven_l2745_274569


namespace NUMINAMATH_CALUDE_no_natural_number_with_digit_product_6552_l2745_274566

theorem no_natural_number_with_digit_product_6552 :
  ¬ ∃ n : ℕ, (n.digits 10).prod = 6552 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_number_with_digit_product_6552_l2745_274566


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l2745_274596

theorem smallest_number_divisibility (h : ℕ) : 
  (∀ k < 259, ¬(∃ n : ℕ, k + 5 = 8 * n ∧ k + 5 = 11 * n ∧ k + 5 = 3 * n)) ∧
  (∃ n : ℕ, 259 + 5 = 8 * n) ∧
  (∃ n : ℕ, 259 + 5 = 11 * n) ∧
  (∃ n : ℕ, 259 + 5 = 3 * n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l2745_274596


namespace NUMINAMATH_CALUDE_orchard_pure_gala_count_l2745_274509

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  cross_pollinated : ℕ
  pure_gala : ℕ

/-- The number of pure Gala trees in an orchard satisfying specific conditions -/
def pure_gala_count (o : Orchard) : Prop :=
  o.pure_fuji + o.cross_pollinated = 204 ∧
  o.pure_fuji = (3 * o.total) / 4 ∧
  o.cross_pollinated = o.total / 10 ∧
  o.pure_gala = 36

/-- Theorem stating that an orchard satisfying the given conditions has 36 pure Gala trees -/
theorem orchard_pure_gala_count :
  ∃ (o : Orchard), pure_gala_count o :=
sorry

end NUMINAMATH_CALUDE_orchard_pure_gala_count_l2745_274509


namespace NUMINAMATH_CALUDE_machines_needed_for_faster_job_additional_machines_needed_l2745_274522

theorem machines_needed_for_faster_job (initial_machines : ℕ) (initial_days : ℕ) : ℕ :=
  let total_machine_days := initial_machines * initial_days
  let new_days := initial_days * 3 / 4
  let new_machines := total_machine_days / new_days
  new_machines - initial_machines

theorem additional_machines_needed :
  machines_needed_for_faster_job 12 40 = 4 := by
  sorry

end NUMINAMATH_CALUDE_machines_needed_for_faster_job_additional_machines_needed_l2745_274522


namespace NUMINAMATH_CALUDE_pencil_cost_l2745_274548

theorem pencil_cost (total_students : ℕ) (total_cost : ℚ) : ∃ (buyers pencils_per_student pencil_cost : ℕ),
  total_students = 30 ∧
  total_cost = 1771 / 100 ∧
  buyers > total_students / 2 ∧
  buyers ≤ total_students ∧
  pencils_per_student > 1 ∧
  pencil_cost > pencils_per_student ∧
  buyers * pencils_per_student * pencil_cost = 1771 ∧
  pencil_cost = 11 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_l2745_274548


namespace NUMINAMATH_CALUDE_marco_card_ratio_l2745_274512

/-- Represents the number of cards in Marco's collection -/
def total_cards : ℕ := 500

/-- Represents the number of new cards Marco received in the trade -/
def new_cards : ℕ := 25

/-- Calculates the number of duplicate cards before the trade -/
def duplicate_cards : ℕ := 5 * new_cards

/-- Represents the ratio of duplicate cards to total cards -/
def duplicate_ratio : ℚ := duplicate_cards / total_cards

theorem marco_card_ratio : duplicate_ratio = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_marco_card_ratio_l2745_274512


namespace NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_roots_l2745_274511

theorem unique_magnitude_of_quadratic_roots (w : ℂ) : 
  w^2 - 6*w + 40 = 0 → ∃! x : ℝ, ∃ w : ℂ, w^2 - 6*w + 40 = 0 ∧ Complex.abs w = x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_roots_l2745_274511


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l2745_274535

/-- The volume of a sphere inscribed in a cube with edge length 8 inches -/
theorem inscribed_sphere_volume :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (256 / 3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l2745_274535


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l2745_274503

/-- A trapezoid with given properties -/
structure Trapezoid where
  longer_base : ℝ
  midpoints_distance : ℝ
  shorter_base : ℝ

/-- The theorem stating the relationship between the bases and the midpoints distance -/
theorem trapezoid_shorter_base (t : Trapezoid) 
  (h1 : t.longer_base = 24)
  (h2 : t.midpoints_distance = 4) : 
  t.shorter_base = 16 := by
  sorry

#check trapezoid_shorter_base

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l2745_274503


namespace NUMINAMATH_CALUDE_probability_theorem_l2745_274561

/-- Parallelogram with given vertices -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The specific parallelogram ABCD from the problem -/
def ABCD : Parallelogram :=
  { A := (9, 4)
    B := (3, -2)
    C := (-3, -2)
    D := (3, 4) }

/-- Probability of a point in the parallelogram being not above the x-axis -/
def probability_not_above_x_axis (p : Parallelogram) : ℚ :=
  1/2

/-- Theorem stating the probability for the given parallelogram -/
theorem probability_theorem :
  probability_not_above_x_axis ABCD = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2745_274561


namespace NUMINAMATH_CALUDE_cost_of_planting_flowers_l2745_274528

/-- The cost of planting flowers given the prices of flowers, clay pot, and soil bag. -/
theorem cost_of_planting_flowers
  (flower_cost : ℕ)
  (clay_pot_cost : ℕ)
  (soil_bag_cost : ℕ)
  (h1 : flower_cost = 9)
  (h2 : clay_pot_cost = flower_cost + 20)
  (h3 : soil_bag_cost = flower_cost - 2) :
  flower_cost + clay_pot_cost + soil_bag_cost = 45 := by
  sorry

#check cost_of_planting_flowers

end NUMINAMATH_CALUDE_cost_of_planting_flowers_l2745_274528


namespace NUMINAMATH_CALUDE_expression_value_l2745_274550

theorem expression_value (x y z : ℝ) 
  (h1 : (1 / (y + z)) + (1 / (x + z)) + (1 / (x + y)) = 5)
  (h2 : x + y + z = 2) :
  (x / (y + z)) + (y / (x + z)) + (z / (x + y)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2745_274550


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2745_274533

/-- The time required for a train to cross a platform -/
theorem train_platform_crossing_time
  (train_speed : Real)
  (man_crossing_time : Real)
  (platform_length : Real)
  (h1 : train_speed = 72 / 3.6) -- 72 kmph converted to m/s
  (h2 : man_crossing_time = 18)
  (h3 : platform_length = 340) :
  let train_length := train_speed * man_crossing_time
  let total_distance := train_length + platform_length
  total_distance / train_speed = 35 := by sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2745_274533


namespace NUMINAMATH_CALUDE_sum_of_products_l2745_274524

theorem sum_of_products : 
  12345 * 5 + 23451 * 4 + 34512 * 3 + 45123 * 2 + 51234 * 1 = 400545 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l2745_274524


namespace NUMINAMATH_CALUDE_temperature_difference_problem_l2745_274590

theorem temperature_difference_problem (M L N : ℝ) : 
  M = L + N →                           -- Minneapolis is N degrees warmer at noon
  (M - 8) - (L + 6) = 4 ∨ (M - 8) - (L + 6) = -4 →  -- Temperature difference at 6:00 PM
  (N = 18 ∨ N = 10) ∧ N * N = 180 := by
sorry

end NUMINAMATH_CALUDE_temperature_difference_problem_l2745_274590


namespace NUMINAMATH_CALUDE_identity_condition_l2745_274585

/-- 
Proves that the equation (3x-a)(2x+5)-x = 6x^2+2(5x-b) is an identity 
for all x if and only if a = 2 and b = 5.
-/
theorem identity_condition (a b : ℝ) : 
  (∀ x : ℝ, (3*x - a)*(2*x + 5) - x = 6*x^2 + 2*(5*x - b)) ↔ (a = 2 ∧ b = 5) := by
  sorry

end NUMINAMATH_CALUDE_identity_condition_l2745_274585


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l2745_274516

noncomputable def f (x : ℝ) : ℝ := 6 * (x^(1/3)) - (16/3) * (x^(1/4))

theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) → y = (2/3) * x :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l2745_274516


namespace NUMINAMATH_CALUDE_cone_slant_height_l2745_274531

/-- Given a cone with lateral area 10π cm² and base radius 2 cm, 
    the slant height of the cone is 5 cm. -/
theorem cone_slant_height (lateral_area base_radius : ℝ) : 
  lateral_area = 10 * Real.pi ∧ base_radius = 2 → 
  lateral_area = (1 / 2) * (2 * Real.pi * base_radius) * 5 := by
sorry

end NUMINAMATH_CALUDE_cone_slant_height_l2745_274531


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l2745_274518

/-- Calculates the total number of heartbeats during a race --/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  let race_duration := pace * race_distance
  race_duration * heart_rate

/-- Theorem: The athlete's heart beats 24300 times during the 30-mile race --/
theorem athlete_heartbeats :
  let heart_rate : ℕ := 135  -- beats per minute
  let pace : ℕ := 6          -- minutes per mile
  let race_distance : ℕ := 30 -- miles
  total_heartbeats heart_rate pace race_distance = 24300 :=
by
  sorry

end NUMINAMATH_CALUDE_athlete_heartbeats_l2745_274518


namespace NUMINAMATH_CALUDE_warehouse_worker_wage_l2745_274529

/-- Represents the problem of calculating warehouse workers' hourly wage --/
theorem warehouse_worker_wage :
  let num_warehouse_workers : ℕ := 4
  let num_managers : ℕ := 2
  let manager_hourly_wage : ℚ := 20
  let fica_tax_rate : ℚ := 1/10
  let days_per_month : ℕ := 25
  let hours_per_day : ℕ := 8
  let total_monthly_cost : ℚ := 22000

  let total_hours : ℕ := days_per_month * hours_per_day
  let manager_monthly_wage : ℚ := num_managers * manager_hourly_wage * total_hours
  
  ∃ (warehouse_hourly_wage : ℚ),
    warehouse_hourly_wage = 15 ∧
    total_monthly_cost = (1 + fica_tax_rate) * (num_warehouse_workers * warehouse_hourly_wage * total_hours + manager_monthly_wage) :=
by sorry

end NUMINAMATH_CALUDE_warehouse_worker_wage_l2745_274529


namespace NUMINAMATH_CALUDE_equal_probability_for_all_positions_l2745_274560

/-- Represents a lottery draw with n tickets, where one is winning. -/
structure LotteryDraw (n : ℕ) where
  tickets : Fin n → Bool
  winning_exists : ∃ t, tickets t = true
  only_one_winning : ∀ t₁ t₂, tickets t₁ = true → tickets t₂ = true → t₁ = t₂

/-- The probability of drawing the winning ticket in any position of a sequence of n draws. -/
def winning_probability (n : ℕ) (pos : Fin n) (draw : LotteryDraw n) : ℚ :=
  1 / n

/-- Theorem stating that the probability of drawing the winning ticket is equal for all positions in a sequence of 5 draws. -/
theorem equal_probability_for_all_positions (draw : LotteryDraw 5) :
    ∀ pos₁ pos₂ : Fin 5, winning_probability 5 pos₁ draw = winning_probability 5 pos₂ draw :=
  sorry

end NUMINAMATH_CALUDE_equal_probability_for_all_positions_l2745_274560


namespace NUMINAMATH_CALUDE_range_of_f_l2745_274557

noncomputable def f (x : ℝ) : ℝ := (Real.arccos x)^4 + (Real.arcsin x)^4

theorem range_of_f :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  ∃ y ∈ Set.Icc 0 (π^4/8),
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc 0 (π^4/8) := by
sorry

end NUMINAMATH_CALUDE_range_of_f_l2745_274557


namespace NUMINAMATH_CALUDE_dave_guitar_strings_l2745_274573

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 2

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 12

/-- The total number of guitar strings Dave needs to replace -/
def total_strings_replaced : ℕ := strings_per_night * shows_per_week * total_weeks

theorem dave_guitar_strings :
  total_strings_replaced = 144 :=
by sorry

end NUMINAMATH_CALUDE_dave_guitar_strings_l2745_274573


namespace NUMINAMATH_CALUDE_kyoko_balls_correct_l2745_274570

/-- The number of balls Kyoko bought -/
def num_balls : ℕ := 3

/-- The cost of each ball in dollars -/
def cost_per_ball : ℚ := 154/100

/-- The total amount Kyoko paid in dollars -/
def total_paid : ℚ := 462/100

/-- Theorem stating that the number of balls Kyoko bought is correct -/
theorem kyoko_balls_correct : 
  (cost_per_ball * num_balls : ℚ) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_kyoko_balls_correct_l2745_274570


namespace NUMINAMATH_CALUDE_decimal_digits_sum_l2745_274580

theorem decimal_digits_sum (a : ℕ) : ∃ (n m : ℕ),
  (10^(n-1) ≤ a ∧ a < 10^n) ∧
  (10^(3*(n-1)) ≤ a^3 ∧ a^3 < 10^(3*n)) ∧
  (3*n - 2 ≤ m ∧ m ≤ 3*n) →
  n + m ≠ 2001 := by
sorry

end NUMINAMATH_CALUDE_decimal_digits_sum_l2745_274580


namespace NUMINAMATH_CALUDE_sand_bucket_calculation_l2745_274574

theorem sand_bucket_calculation (bucket_weight : ℕ) (total_weight : ℕ) (h1 : bucket_weight = 2) (h2 : total_weight = 34) :
  total_weight / bucket_weight = 17 := by
  sorry

end NUMINAMATH_CALUDE_sand_bucket_calculation_l2745_274574


namespace NUMINAMATH_CALUDE_inserted_digit_divisible_by_seven_l2745_274541

theorem inserted_digit_divisible_by_seven :
  ∀ x : ℕ, x < 10 →
    (20000 + x * 100 + 6) % 7 = 0 ↔ x = 0 ∨ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_inserted_digit_divisible_by_seven_l2745_274541


namespace NUMINAMATH_CALUDE_two_rats_through_wall_l2745_274582

/-- The sum of lengths burrowed by two rats in n days -/
def S (n : ℕ) : ℚ :=
  (2^n - 1) + (2 - 1/(2^(n-1)))

/-- The problem statement -/
theorem two_rats_through_wall : S 5 = 32 + 15/16 := by
  sorry

end NUMINAMATH_CALUDE_two_rats_through_wall_l2745_274582


namespace NUMINAMATH_CALUDE_angle_of_inclination_x_eq_neg_one_l2745_274565

-- Define a vertical line
def vertical_line (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 = a}

-- Define the angle of inclination for a vertical line
def angle_of_inclination_vertical (l : Set (ℝ × ℝ)) : ℝ := 90

-- Theorem statement
theorem angle_of_inclination_x_eq_neg_one :
  angle_of_inclination_vertical (vertical_line (-1)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_x_eq_neg_one_l2745_274565


namespace NUMINAMATH_CALUDE_largest_prime_factor_l2745_274506

def numbers : List Nat := [55, 63, 95, 133, 143]

theorem largest_prime_factor :
  ∃ (n : Nat), n ∈ numbers ∧ 19 ∣ n ∧
  ∀ (m : Nat), m ∈ numbers → ∀ (p : Nat), Prime p → p ∣ m → p ≤ 19 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l2745_274506


namespace NUMINAMATH_CALUDE_inequality_proof_l2745_274597

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (2 * a^2 / (1 + a + a * b)^2 + 
   2 * b^2 / (1 + b + b * c)^2 + 
   2 * c^2 / (1 + c + c * a)^2 + 
   9 / ((1 + a + a * b) * (1 + b + b * c) * (1 + c + c * a))) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2745_274597


namespace NUMINAMATH_CALUDE_odd_function_extension_l2745_274542

-- Define the function f on the positive real numbers
def f_pos (x : ℝ) : ℝ := x * (x - 1)

-- State the theorem
theorem odd_function_extension {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) 
  (h_pos : ∀ x > 0, f x = f_pos x) : 
  ∀ x < 0, f x = x * (x + 1) := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l2745_274542


namespace NUMINAMATH_CALUDE_max_distance_difference_l2745_274567

-- Define the curve C₂
def C₂ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 1)

-- Define the distance function
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem max_distance_difference :
  ∃ (max : ℝ), max = 2 + 2 * Real.sqrt 39 ∧
  ∀ (P : ℝ × ℝ), C₂ P.1 P.2 →
    dist_squared P A - dist_squared P B ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_distance_difference_l2745_274567


namespace NUMINAMATH_CALUDE_evening_temp_calculation_l2745_274554

/-- Given a noon temperature and a temperature drop, calculate the evening temperature. -/
def evening_temperature (noon_temp : ℤ) (temp_drop : ℕ) : ℤ :=
  noon_temp - temp_drop

/-- Theorem: If the noon temperature is 2°C and it drops by 3°C, the evening temperature is -1°C. -/
theorem evening_temp_calculation :
  evening_temperature 2 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_evening_temp_calculation_l2745_274554


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2745_274571

-- Define the hyperbola equation
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 4 = 1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop :=
  y = 2 * x

-- Define the real axis length
def real_axis_length (a : ℝ) : ℝ :=
  2 * a

-- Theorem statement
theorem hyperbola_real_axis_length :
  ∃ a : ℝ, (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) →
  real_axis_length a = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l2745_274571


namespace NUMINAMATH_CALUDE_walter_bus_time_l2745_274536

def wake_up_time : Nat := 6 * 60 + 30
def leave_time : Nat := 7 * 60 + 30
def return_time : Nat := 16 * 60 + 30
def num_classes : Nat := 7
def class_duration : Nat := 45
def lunch_duration : Nat := 40
def additional_time : Nat := 150

def total_away_time : Nat := return_time - leave_time
def school_time : Nat := num_classes * class_duration + lunch_duration + additional_time

theorem walter_bus_time :
  total_away_time - school_time = 35 := by
  sorry

end NUMINAMATH_CALUDE_walter_bus_time_l2745_274536


namespace NUMINAMATH_CALUDE_coefficient_x_squared_proof_l2745_274534

/-- The coefficient of x^2 in the expansion of (1-3x)^7 -/
def coefficient_x_squared : ℕ := 7

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_x_squared_proof :
  coefficient_x_squared = binomial 7 6 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_proof_l2745_274534


namespace NUMINAMATH_CALUDE_reading_time_calculation_l2745_274589

/-- Calculates the time needed to read a book given the reading speed and book properties -/
theorem reading_time_calculation (reading_speed : ℕ) (paragraphs_per_page : ℕ) 
  (sentences_per_paragraph : ℕ) (total_pages : ℕ) : 
  reading_speed = 200 →
  paragraphs_per_page = 20 →
  sentences_per_paragraph = 10 →
  total_pages = 50 →
  (total_pages * paragraphs_per_page * sentences_per_paragraph) / reading_speed = 50 := by
  sorry

#check reading_time_calculation

end NUMINAMATH_CALUDE_reading_time_calculation_l2745_274589


namespace NUMINAMATH_CALUDE_complement_of_union_l2745_274586

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union : (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l2745_274586


namespace NUMINAMATH_CALUDE_music_store_purchase_total_l2745_274540

def trumpet_price : ℝ := 149.16
def music_tool_price : ℝ := 9.98
def song_book_price : ℝ := 4.14
def accessories_price : ℝ := 21.47
def valve_oil_original_price : ℝ := 8.20
def tshirt_price : ℝ := 14.95
def valve_oil_discount_rate : ℝ := 0.20
def sales_tax_rate : ℝ := 0.065

def total_spent : ℝ := 219.67

theorem music_store_purchase_total :
  let valve_oil_price := valve_oil_original_price * (1 - valve_oil_discount_rate)
  let subtotal := trumpet_price + music_tool_price + song_book_price + 
                  accessories_price + valve_oil_price + tshirt_price
  let sales_tax := subtotal * sales_tax_rate
  subtotal + sales_tax = total_spent := by sorry

end NUMINAMATH_CALUDE_music_store_purchase_total_l2745_274540


namespace NUMINAMATH_CALUDE_real_roots_iff_k_nonzero_l2745_274553

theorem real_roots_iff_k_nonzero (K : ℝ) :
  (∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3)) ↔ K ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_iff_k_nonzero_l2745_274553


namespace NUMINAMATH_CALUDE_radiator_water_fraction_l2745_274538

/-- The fraction of water remaining after n replacements in a radiator -/
def water_fraction (radiator_capacity : ℚ) (replacement_volume : ℚ) (n : ℕ) : ℚ :=
  (1 - replacement_volume / radiator_capacity) ^ n

theorem radiator_water_fraction :
  let radiator_capacity : ℚ := 20
  let replacement_volume : ℚ := 5
  let num_replacements : ℕ := 5
  water_fraction radiator_capacity replacement_volume num_replacements = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_radiator_water_fraction_l2745_274538


namespace NUMINAMATH_CALUDE_largest_kappa_l2745_274584

theorem largest_kappa : ∃ κ : ℝ, κ = 2 ∧ 
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + d^2 = b^2 + c^2 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*c + κ*b*d + a*d) ∧ 
  (∀ κ' : ℝ, κ' > κ → 
    ∃ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
      a^2 + d^2 = b^2 + c^2 ∧ 
      a^2 + b^2 + c^2 + d^2 < a*c + κ'*b*d + a*d) :=
by sorry

end NUMINAMATH_CALUDE_largest_kappa_l2745_274584


namespace NUMINAMATH_CALUDE_missing_integers_count_l2745_274595

theorem missing_integers_count (n : ℕ) (h : n = 2017) : 
  n - (n - n / 3 + n / 6 - n / 54) = 373 :=
by sorry

end NUMINAMATH_CALUDE_missing_integers_count_l2745_274595


namespace NUMINAMATH_CALUDE_compare_large_exponents_l2745_274549

theorem compare_large_exponents :
  20^(19^20) > 19^(20^19) := by
  sorry

end NUMINAMATH_CALUDE_compare_large_exponents_l2745_274549


namespace NUMINAMATH_CALUDE_larger_number_is_eight_l2745_274578

theorem larger_number_is_eight (x y : ℕ) (h1 : x * y = 24) (h2 : x + y = 11) : 
  max x y = 8 := by
sorry

end NUMINAMATH_CALUDE_larger_number_is_eight_l2745_274578


namespace NUMINAMATH_CALUDE_arithmetic_sequence_of_squares_l2745_274594

theorem arithmetic_sequence_of_squares (a b c x y : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- a, b, c are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ -- a, b, c are distinct
  b - a = c - b ∧ -- a, b, c form an arithmetic sequence
  x^2 = a * b ∧ -- x is the geometric mean of a and b
  y^2 = b * c -- y is the geometric mean of b and c
  → 
  (y^2 - b^2 = b^2 - x^2) ∧ -- x^2, b^2, y^2 form an arithmetic sequence
  ¬(y^2 / b^2 = b^2 / x^2) -- x^2, b^2, y^2 do not form a geometric sequence
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_of_squares_l2745_274594


namespace NUMINAMATH_CALUDE_five_not_in_A_and_B_l2745_274593

universe u

def U : Set Nat := {1, 2, 3, 4, 5}

theorem five_not_in_A_and_B
  (A B : Set Nat)
  (h_subset : A ⊆ U ∧ B ⊆ U)
  (h_inter : A ∩ B = {2, 4})
  (h_union : A ∪ B = {1, 2, 3, 4}) :
  5 ∉ A ∧ 5 ∉ B := by
  sorry


end NUMINAMATH_CALUDE_five_not_in_A_and_B_l2745_274593


namespace NUMINAMATH_CALUDE_bridget_profit_is_fifty_l2745_274546

/-- Calculates Bridget's profit from baking and selling bread --/
def bridget_profit (total_loaves : ℕ) (cost_per_loaf : ℚ) 
  (morning_price : ℚ) (late_afternoon_price : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3
  let morning_revenue := morning_sales * morning_price
  let afternoon_remaining := total_loaves - morning_sales
  let afternoon_sales := afternoon_remaining / 2
  let afternoon_revenue := afternoon_sales * (morning_price / 2)
  let late_afternoon_remaining := afternoon_remaining - afternoon_sales
  let late_afternoon_sales := (late_afternoon_remaining * 2) / 3
  let late_afternoon_revenue := late_afternoon_sales * late_afternoon_price
  let evening_sales := late_afternoon_remaining - late_afternoon_sales
  let evening_revenue := evening_sales * cost_per_loaf
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue + evening_revenue
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

/-- Bridget's profit is $50 --/
theorem bridget_profit_is_fifty :
  bridget_profit 60 1 3 1 = 50 :=
by sorry

end NUMINAMATH_CALUDE_bridget_profit_is_fifty_l2745_274546


namespace NUMINAMATH_CALUDE_registration_methods_l2745_274523

/-- The number of ways to distribute n distinct objects into k non-empty distinct groups -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 students and 3 courses -/
def num_students : ℕ := 5
def num_courses : ℕ := 3

/-- Each student signs up for exactly one course -/
axiom one_course_per_student : distribute num_students num_courses > 0

/-- Each course must have at least one student enrolled -/
axiom non_empty_courses : ∀ (i : Fin num_courses), ∃ (student : Fin num_students), sorry

/-- The number of different registration methods is 150 -/
theorem registration_methods : distribute num_students num_courses = 150 := by sorry

end NUMINAMATH_CALUDE_registration_methods_l2745_274523


namespace NUMINAMATH_CALUDE_smallest_n_for_divisibility_by_1991_l2745_274501

theorem smallest_n_for_divisibility_by_1991 :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (1991 ∣ (a + b) ∨ 1991 ∣ (a - b))) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℤ), T.card = m ∧
      ∀ (a b : ℤ), a ∈ T → b ∈ T → a ≠ b → ¬(1991 ∣ (a + b)) ∧ ¬(1991 ∣ (a - b))) ∧
  n = 997 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisibility_by_1991_l2745_274501


namespace NUMINAMATH_CALUDE_sum_difference_theorem_l2745_274514

def sara_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (x : ℕ) : ℕ :=
  5 * ((x + 2) / 5)

def mike_sum (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_five (List.range n))

theorem sum_difference_theorem :
  sara_sum 120 - mike_sum 120 = 6900 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_theorem_l2745_274514


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2745_274539

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (a 3)^2 - 3 * (a 3) - 5 = 0 →
  (a 11)^2 - 3 * (a 11) - 5 = 0 →
  a 5 + a 6 + a 10 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2745_274539


namespace NUMINAMATH_CALUDE_right_triangle_set_l2745_274591

/-- Checks if a set of three numbers can form a right-angled triangle --/
def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The sets of sticks given in the problem --/
def stickSets : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7)]

theorem right_triangle_set :
  ∃! (a b c : ℕ), (a, b, c) ∈ stickSets ∧ isRightTriangle a b c :=
by
  sorry

#check right_triangle_set

end NUMINAMATH_CALUDE_right_triangle_set_l2745_274591


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l2745_274508

theorem inequality_and_minimum_value (a b c : ℝ) 
  (ha : 1 < a ∧ a < Real.sqrt 7)
  (hb : 1 < b ∧ b < Real.sqrt 7)
  (hc : 1 < c ∧ c < Real.sqrt 7) :
  (1 / (a^2 - 1) + 1 / (7 - a^2) ≥ 2/3) ∧
  (1 / Real.sqrt ((a^2 - 1) * (7 - b^2)) + 
   1 / Real.sqrt ((b^2 - 1) * (7 - c^2)) + 
   1 / Real.sqrt ((c^2 - 1) * (7 - a^2)) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l2745_274508


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l2745_274579

def is_valid_representation (n : ℕ) (a b : ℕ) : Prop :=
  a > 2 ∧ b > 2 ∧ 2 * a + 1 = n ∧ b + 2 = n

theorem smallest_dual_base_representation : 
  (∃ (a b : ℕ), is_valid_representation 7 a b) ∧ 
  (∀ (n : ℕ), n < 7 → ¬∃ (a b : ℕ), is_valid_representation n a b) :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l2745_274579


namespace NUMINAMATH_CALUDE_base_2_representation_of_123_l2745_274513

theorem base_2_representation_of_123 : 
  (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_123_l2745_274513


namespace NUMINAMATH_CALUDE_cos_sin_transformation_l2745_274592

theorem cos_sin_transformation (x : ℝ) : 
  3 * Real.cos x = 3 * Real.sin (2 * (x + 2 * Real.pi / 3) - Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_transformation_l2745_274592


namespace NUMINAMATH_CALUDE_cab_speed_fraction_l2745_274575

/-- Proves that for a cab with a usual journey time of 40 minutes, if it's 8 minutes late at a reduced speed, then the reduced speed is 5/6 of its usual speed. -/
theorem cab_speed_fraction (usual_time : ℕ) (delay : ℕ) : 
  usual_time = 40 → delay = 8 → (usual_time : ℚ) / (usual_time + delay) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cab_speed_fraction_l2745_274575


namespace NUMINAMATH_CALUDE_second_question_percentage_l2745_274568

theorem second_question_percentage 
  (first_correct : Real) 
  (neither_correct : Real) 
  (both_correct : Real)
  (h1 : first_correct = 0.75)
  (h2 : neither_correct = 0.2)
  (h3 : both_correct = 0.6) :
  ∃ second_correct : Real, 
    second_correct = 0.65 ∧ 
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by sorry

end NUMINAMATH_CALUDE_second_question_percentage_l2745_274568


namespace NUMINAMATH_CALUDE_equation_solution_l2745_274559

theorem equation_solution (x : ℝ) : 
  Real.sqrt (5 * x - 4) + 15 / Real.sqrt (5 * x - 4) = 8 → x = 29/5 ∨ x = 13/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2745_274559


namespace NUMINAMATH_CALUDE_min_cuts_3x3x3_cube_l2745_274558

/-- Represents a 3D cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Represents a cut along a plane -/
inductive Cut
  | X : ℕ → Cut
  | Y : ℕ → Cut
  | Z : ℕ → Cut

/-- The minimum number of cuts required to divide a cube into unit cubes -/
def min_cuts (c : Cube 3) : ℕ := 6

/-- Theorem stating that the minimum number of cuts to divide a 3x3x3 cube into 27 unit cubes is 6 -/
theorem min_cuts_3x3x3_cube (c : Cube 3) :
  min_cuts c = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_cuts_3x3x3_cube_l2745_274558


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2745_274517

theorem sqrt_meaningful_range (a : ℝ) : (∃ x : ℝ, x^2 = a - 2) ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2745_274517


namespace NUMINAMATH_CALUDE_eds_walking_speed_l2745_274562

/-- Proves that Ed's walking speed is 4 km/h given the specified conditions -/
theorem eds_walking_speed (total_distance : ℝ) (sandys_speed : ℝ) (sandys_distance : ℝ) (time_difference : ℝ) :
  total_distance = 52 →
  sandys_speed = 6 →
  sandys_distance = 36 →
  time_difference = 2 →
  ∃ (eds_speed : ℝ), eds_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_eds_walking_speed_l2745_274562


namespace NUMINAMATH_CALUDE_mrs_brown_utility_bill_l2745_274598

/-- Calculates the actual utility bill amount given the initial payment and returned amount -/
def actualUtilityBill (initialPayment returnedAmount : ℕ) : ℕ :=
  initialPayment - returnedAmount

/-- Theorem stating that Mrs. Brown's actual utility bill is $710 -/
theorem mrs_brown_utility_bill :
  let initialPayment := 4 * 100 + 5 * 50 + 7 * 20
  let returnedAmount := 3 * 20 + 2 * 10
  actualUtilityBill initialPayment returnedAmount = 710 := by
  sorry

#eval actualUtilityBill (4 * 100 + 5 * 50 + 7 * 20) (3 * 20 + 2 * 10)

end NUMINAMATH_CALUDE_mrs_brown_utility_bill_l2745_274598


namespace NUMINAMATH_CALUDE_cylinder_volume_problem_l2745_274520

theorem cylinder_volume_problem (h₁ : ℝ) (h₂ : ℝ) (r₁ r₂ : ℝ) :
  r₁ = 7 →
  r₂ = 1.2 * r₁ →
  h₂ = 0.85 * h₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  π * r₁^2 * h₁ = 49 * π * h₁ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_problem_l2745_274520


namespace NUMINAMATH_CALUDE_book_cost_range_l2745_274577

theorem book_cost_range (p : ℝ) 
  (h1 : 11 * p < 15)
  (h2 : 12 * p > 16) : 
  4 / 3 < p ∧ p < 15 / 11 :=
by sorry

end NUMINAMATH_CALUDE_book_cost_range_l2745_274577


namespace NUMINAMATH_CALUDE_students_per_bus_l2745_274502

theorem students_per_bus (total_students : ℕ) (num_buses : ℕ) 
  (h1 : total_students = 360) (h2 : num_buses = 8) :
  total_students / num_buses = 45 := by
  sorry

end NUMINAMATH_CALUDE_students_per_bus_l2745_274502


namespace NUMINAMATH_CALUDE_sqrt_eight_times_half_minus_sqrt_three_power_zero_l2745_274527

theorem sqrt_eight_times_half_minus_sqrt_three_power_zero :
  Real.sqrt 8 * (1 / 2) - (Real.sqrt 3) ^ 0 = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_times_half_minus_sqrt_three_power_zero_l2745_274527


namespace NUMINAMATH_CALUDE_opposite_to_83_is_84_l2745_274581

/-- Represents a circle divided into 100 equal arcs with numbers assigned to each arc. -/
structure NumberedCircle where
  /-- The assignment of numbers to arcs, represented as a function from arc index to number. -/
  number_assignment : Fin 100 → Fin 100
  /-- The assignment is a bijection (each number is used exactly once). -/
  bijective : Function.Bijective number_assignment

/-- Checks if numbers less than k are evenly distributed on both sides of the diameter through k. -/
def evenlyDistributed (c : NumberedCircle) (k : Fin 100) : Prop :=
  ∀ (i : Fin 100), c.number_assignment i < k →
    (∃ (j : Fin 100), c.number_assignment j < k ∧ (i + 50) % 100 = j)

/-- The main theorem stating that if numbers are evenly distributed for all k,
    then the number opposite to 83 is 84. -/
theorem opposite_to_83_is_84 (c : NumberedCircle) 
    (h : ∀ (k : Fin 100), evenlyDistributed c k) :
    ∃ (i : Fin 100), c.number_assignment i = 83 ∧ c.number_assignment ((i + 50) % 100) = 84 := by
  sorry

end NUMINAMATH_CALUDE_opposite_to_83_is_84_l2745_274581


namespace NUMINAMATH_CALUDE_bivalent_metal_relative_atomic_mass_l2745_274525

-- Define the bivalent metal
structure BivalentMetal where
  relative_atomic_mass : ℝ

-- Define the reaction conditions
def hcl_moles : ℝ := 0.25

-- Define the reaction properties
def incomplete_reaction (m : BivalentMetal) : Prop :=
  3.5 / m.relative_atomic_mass > hcl_moles / 2

def complete_reaction (m : BivalentMetal) : Prop :=
  2.5 / m.relative_atomic_mass < hcl_moles / 2

-- Theorem to prove
theorem bivalent_metal_relative_atomic_mass :
  ∃ (m : BivalentMetal), 
    m.relative_atomic_mass = 24 ∧ 
    incomplete_reaction m ∧ 
    complete_reaction m :=
by
  sorry

end NUMINAMATH_CALUDE_bivalent_metal_relative_atomic_mass_l2745_274525


namespace NUMINAMATH_CALUDE_tensor_product_of_A_and_B_l2745_274521

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {y : ℝ | y ≥ 0}

-- Define the ⊗ operation
def tensorProduct (X Y : Set ℝ) : Set ℝ := (X ∪ Y) \ (X ∩ Y)

-- Theorem statement
theorem tensor_product_of_A_and_B :
  tensorProduct A B = {x : ℝ | x = 0 ∨ x ≥ 2} := by
  sorry

end NUMINAMATH_CALUDE_tensor_product_of_A_and_B_l2745_274521


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_l2745_274544

/-- Represents a club with members of different characteristics -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_non_jazz : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club)
  (h1 : c.total_members = 30)
  (h2 : c.left_handed = 11)
  (h3 : c.jazz_lovers = 20)
  (h4 : c.right_handed_non_jazz = 4)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members) :
  c.left_handed + c.jazz_lovers - c.total_members + c.right_handed_non_jazz = 5 := by
  sorry

#check left_handed_jazz_lovers

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_l2745_274544


namespace NUMINAMATH_CALUDE_vector_to_line_parallel_l2745_274505

/-- A vector pointing from the origin to a line parallel to another vector -/
theorem vector_to_line_parallel (t : ℝ) : ∃ (k : ℝ), ∃ (a b : ℝ),
  (a = 3 * t + 1 ∧ b = t + 1) ∧  -- Point on the line
  (∃ (c : ℝ), a = 3 * c ∧ b = c) ∧  -- Parallel to (3, 1)
  a = 3 * k - 2 ∧ b = k :=  -- The form of the vector
by sorry

end NUMINAMATH_CALUDE_vector_to_line_parallel_l2745_274505


namespace NUMINAMATH_CALUDE_mrs_hilt_hotdog_cost_l2745_274510

/-- The total cost in cents for a given number of hot dogs at a given price per hot dog -/
def total_cost (num_hotdogs : ℕ) (price_per_hotdog : ℕ) : ℕ :=
  num_hotdogs * price_per_hotdog

/-- Proof that Mrs. Hilt paid 300 cents for 6 hot dogs at 50 cents each -/
theorem mrs_hilt_hotdog_cost : total_cost 6 50 = 300 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_hotdog_cost_l2745_274510


namespace NUMINAMATH_CALUDE_decreasing_linear_function_negative_slope_l2745_274500

/-- A linear function y = kx - 5 where y decreases as x increases -/
def decreasing_linear_function (k : ℝ) : ℝ → ℝ := λ x ↦ k * x - 5

/-- Theorem: If y decreases as x increases in a linear function y = kx - 5, then k < 0 -/
theorem decreasing_linear_function_negative_slope (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → decreasing_linear_function k x₁ > decreasing_linear_function k x₂) →
  k < 0 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_negative_slope_l2745_274500


namespace NUMINAMATH_CALUDE_distribute_negative_three_l2745_274504

theorem distribute_negative_three (a : ℝ) : -3 * (a - 1) = 3 - 3 * a := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_three_l2745_274504


namespace NUMINAMATH_CALUDE_volumes_equal_l2745_274576

/-- The volume of a solid obtained by rotating a region around the y-axis -/
noncomputable def rotationVolume (f : ℝ → ℝ → Prop) : ℝ := sorry

/-- The region enclosed by the curves x² = 4y, x² = -4y, x = 4, and x = -4 -/
def region1 (x y : ℝ) : Prop :=
  (x^2 = 4*y ∨ x^2 = -4*y) ∧ -4 ≤ x ∧ x ≤ 4

/-- The region represented by points (x, y) that satisfy x² + y² ≤ 16, x² + (y - 2)² ≥ 4, and x² + (y + 2)² ≥ 4 -/
def region2 (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 16 ∧ x^2 + (y - 2)^2 ≥ 4 ∧ x^2 + (y + 2)^2 ≥ 4

/-- The theorem stating that the volumes of the two solids are equal -/
theorem volumes_equal : rotationVolume region1 = rotationVolume region2 := by
  sorry

end NUMINAMATH_CALUDE_volumes_equal_l2745_274576


namespace NUMINAMATH_CALUDE_alternating_seating_theorem_l2745_274572

/-- The number of ways to arrange n girls and n boys alternately around a round table with 2n seats -/
def alternating_seating_arrangements (n : ℕ) : ℕ :=
  2 * (n.factorial)^2

/-- Theorem stating that the number of alternating seating arrangements
    for n girls and n boys around a round table with 2n seats is 2(n!)^2 -/
theorem alternating_seating_theorem (n : ℕ) :
  alternating_seating_arrangements n = 2 * (n.factorial)^2 := by
  sorry

end NUMINAMATH_CALUDE_alternating_seating_theorem_l2745_274572


namespace NUMINAMATH_CALUDE_boisjoli_farm_egg_production_l2745_274532

/-- The number of eggs each hen lays per day at Boisjoli farm -/
theorem boisjoli_farm_egg_production 
  (num_hens : ℕ) 
  (num_days : ℕ) 
  (num_boxes : ℕ) 
  (eggs_per_box : ℕ) 
  (h_hens : num_hens = 270) 
  (h_days : num_days = 7) 
  (h_boxes : num_boxes = 315) 
  (h_eggs_per_box : eggs_per_box = 6) : 
  (num_boxes * eggs_per_box) / (num_hens * num_days) = 1 := by
  sorry

#check boisjoli_farm_egg_production

end NUMINAMATH_CALUDE_boisjoli_farm_egg_production_l2745_274532


namespace NUMINAMATH_CALUDE_four_cube_painted_subcubes_l2745_274599

/-- Represents a cube with some faces painted -/
structure PaintedCube where
  size : ℕ
  painted_faces : ℕ
  unpainted_faces : ℕ

/-- Calculates the number of subcubes with at least one painted face -/
def subcubes_with_paint (c : PaintedCube) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 cube with 4 painted faces has 48 subcubes with paint -/
theorem four_cube_painted_subcubes :
  let c : PaintedCube := { size := 4, painted_faces := 4, unpainted_faces := 2 }
  subcubes_with_paint c = 48 := by
  sorry

end NUMINAMATH_CALUDE_four_cube_painted_subcubes_l2745_274599


namespace NUMINAMATH_CALUDE_difference_in_amounts_l2745_274551

/-- Represents the three products A, B, and C --/
inductive Product
| A
| B
| C

/-- The initial price of a product --/
def initialPrice (p : Product) : ℝ :=
  match p with
  | Product.A => 100
  | Product.B => 150
  | Product.C => 200

/-- The price increase percentage for a product --/
def priceIncrease (p : Product) : ℝ :=
  match p with
  | Product.A => 0.10
  | Product.B => 0.15
  | Product.C => 0.20

/-- The quantity bought after price increase as a fraction of initial quantity --/
def quantityAfterIncrease (p : Product) : ℝ :=
  match p with
  | Product.A => 0.90
  | Product.B => 0.85
  | Product.C => 0.80

/-- The discount percentage --/
def discount : ℝ := 0.05

/-- The additional quantity bought on discount day as a fraction of initial quantity --/
def additionalQuantity (p : Product) : ℝ :=
  match p with
  | Product.A => 0.10
  | Product.B => 0.15
  | Product.C => 0.20

/-- The total amount paid on the price increase day --/
def amountOnIncreaseDay : ℝ :=
  (initialPrice Product.A * (1 + priceIncrease Product.A) * quantityAfterIncrease Product.A) +
  (initialPrice Product.B * (1 + priceIncrease Product.B) * quantityAfterIncrease Product.B) +
  (initialPrice Product.C * (1 + priceIncrease Product.C) * quantityAfterIncrease Product.C)

/-- The total amount paid on the discount day --/
def amountOnDiscountDay : ℝ :=
  (initialPrice Product.A * (1 - discount) * (1 + additionalQuantity Product.A)) +
  (initialPrice Product.B * (1 - discount) * (1 + additionalQuantity Product.B)) +
  (initialPrice Product.C * (1 - discount) * (1 + additionalQuantity Product.C))

/-- The theorem stating the difference in amounts paid --/
theorem difference_in_amounts : amountOnIncreaseDay - amountOnDiscountDay = 58.75 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_amounts_l2745_274551


namespace NUMINAMATH_CALUDE_cube_surface_area_l2745_274563

theorem cube_surface_area (d : ℝ) (h : d = 8 * Real.sqrt 2) : 
  6 * (d / Real.sqrt 2) ^ 2 = 384 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2745_274563


namespace NUMINAMATH_CALUDE_range_of_a_l2745_274515

-- Define the inequalities p and q
def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the theorem
theorem range_of_a :
  (∃ x : ℝ, (¬(p x) ∧ q x a) ∨ (p x ∧ ¬(q x a))) →
  (a ∈ Set.Icc (0 : ℝ) (1/2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2745_274515


namespace NUMINAMATH_CALUDE_prime_pairs_theorem_l2745_274507

theorem prime_pairs_theorem : 
  ∀ p q : ℕ, 
    Prime p → Prime q → Prime (p * q + p - 6) → 
    ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_theorem_l2745_274507


namespace NUMINAMATH_CALUDE_max_z_value_l2745_274587

theorem max_z_value (x y z : ℝ) 
  (sum_eq : x + y + z = 7)
  (prod_sum_eq : x * y + x * z + y * z = 12)
  (x_pos : x > 0)
  (y_pos : y > 0)
  (z_pos : z > 0) :
  z ≤ 1 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ + 1 = 7 ∧ x₀ * y₀ + x₀ * 1 + y₀ * 1 = 12 :=
sorry

end NUMINAMATH_CALUDE_max_z_value_l2745_274587


namespace NUMINAMATH_CALUDE_no_double_apply_1987_function_l2745_274583

theorem no_double_apply_1987_function :
  ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 1987 := by
  sorry

end NUMINAMATH_CALUDE_no_double_apply_1987_function_l2745_274583
