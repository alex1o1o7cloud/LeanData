import Mathlib

namespace NUMINAMATH_CALUDE_smallest_item_is_a5_l2394_239483

def sequence_a (n : ℕ) : ℚ :=
  2 * n^2 - 21 * n + 40

theorem smallest_item_is_a5 :
  ∀ n : ℕ, n ≥ 1 → sequence_a 5 ≤ sequence_a n :=
sorry

end NUMINAMATH_CALUDE_smallest_item_is_a5_l2394_239483


namespace NUMINAMATH_CALUDE_square_area_proof_l2394_239464

theorem square_area_proof (x : ℝ) : 
  (5 * x - 18 = 27 - 4 * x) → 
  ((5 * x - 18)^2 : ℝ) = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l2394_239464


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l2394_239454

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, IsPerfectSquareTrinomial 4 m 121 → (m = 44 ∨ m = -44) :=
by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l2394_239454


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l2394_239414

/-- Calculates the total distance walked in miles given the number of blocks walked east and north, and the length of each block in miles. -/
def total_distance_miles (blocks_east : ℕ) (blocks_north : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_east + blocks_north : ℚ) * miles_per_block

/-- Theorem stating that walking 6 blocks east and 12 blocks north, with each block being one-third of a mile, results in a total distance of 6 miles. -/
theorem arthur_walk_distance :
  total_distance_miles 6 12 (1/3) = 6 := by sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l2394_239414


namespace NUMINAMATH_CALUDE_closest_integer_to_ten_minus_sqrt_thirteen_l2394_239416

theorem closest_integer_to_ten_minus_sqrt_thirteen :
  let sqrt_13 : ℝ := Real.sqrt 13
  ∀ n : ℤ, n ∈ ({4, 5, 7} : Set ℤ) →
    3 < sqrt_13 ∧ sqrt_13 < 4 →
    |10 - sqrt_13 - 6| < |10 - sqrt_13 - ↑n| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_ten_minus_sqrt_thirteen_l2394_239416


namespace NUMINAMATH_CALUDE_integer_solution_less_than_one_l2394_239479

theorem integer_solution_less_than_one :
  ∃ (x : ℤ), x - 1 < 0 :=
by
  use 0
  sorry

end NUMINAMATH_CALUDE_integer_solution_less_than_one_l2394_239479


namespace NUMINAMATH_CALUDE_roots_of_h_l2394_239456

/-- Given that x = 1 is a root of f(x) = a/x + b and a ≠ 0, 
    prove that the roots of h(x) = ax^2 + bx are 0 and 1. -/
theorem roots_of_h (a b : ℝ) (ha : a ≠ 0) 
  (hf : a / 1 + b = 0) : 
  ∀ x : ℝ, ax^2 + bx = 0 ↔ x = 0 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_of_h_l2394_239456


namespace NUMINAMATH_CALUDE_original_polygon_sides_l2394_239401

-- Define the number of sides of the original polygon
def n : ℕ := sorry

-- Define the sum of interior angles of the new polygon
def new_polygon_angle_sum : ℝ := 2520

-- Theorem statement
theorem original_polygon_sides :
  (n + 1 - 2) * 180 = new_polygon_angle_sum → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_polygon_sides_l2394_239401


namespace NUMINAMATH_CALUDE_stanley_tires_l2394_239415

/-- The number of tires Stanley bought -/
def num_tires : ℕ := 240 / 60

/-- The cost of each tire in dollars -/
def cost_per_tire : ℕ := 60

/-- The total amount Stanley spent in dollars -/
def total_spent : ℕ := 240

theorem stanley_tires :
  num_tires = 4 ∧ cost_per_tire * num_tires = total_spent :=
sorry

end NUMINAMATH_CALUDE_stanley_tires_l2394_239415


namespace NUMINAMATH_CALUDE_sculpture_surface_area_l2394_239485

/-- Represents a step in the staircase sculpture -/
structure Step where
  cubes : ℕ
  exposed_front : ℕ

/-- Represents the staircase sculpture -/
def Sculpture : List Step := [
  { cubes := 6, exposed_front := 6 },
  { cubes := 5, exposed_front := 5 },
  { cubes := 4, exposed_front := 4 },
  { cubes := 2, exposed_front := 2 },
  { cubes := 1, exposed_front := 5 }
]

/-- Calculates the total exposed surface area of the sculpture -/
def total_exposed_area (sculpture : List Step) : ℕ :=
  let top_area := sculpture.map (·.cubes) |>.sum
  let side_area := sculpture.map (·.exposed_front) |>.sum
  top_area + side_area

/-- Theorem: The total exposed surface area of the sculpture is 40 square meters -/
theorem sculpture_surface_area :
  total_exposed_area Sculpture = 40 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_surface_area_l2394_239485


namespace NUMINAMATH_CALUDE_odd_divisors_implies_perfect_square_l2394_239480

theorem odd_divisors_implies_perfect_square (n : ℕ) : 
  (Odd (Nat.card {d : ℕ | d ∣ n})) → ∃ m : ℕ, n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisors_implies_perfect_square_l2394_239480


namespace NUMINAMATH_CALUDE_pizza_slices_l2394_239427

theorem pizza_slices (buzz_ratio waiter_ratio : ℕ) 
  (h1 : buzz_ratio = 5)
  (h2 : waiter_ratio = 8)
  (h3 : waiter_ratio * x - 20 = 28)
  (x : ℕ) : 
  buzz_ratio * x + waiter_ratio * x = 78 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l2394_239427


namespace NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l2394_239419

/-- Represents a labeling of a tetrahedron's vertices -/
def TetrahedronLabeling := Fin 4 → Fin 4

/-- Checks if a labeling uses each number exactly once -/
def is_valid_labeling (l : TetrahedronLabeling) : Prop :=
  ∀ i : Fin 4, ∃! j : Fin 4, l j = i

/-- Represents a face of the tetrahedron -/
def Face := Fin 3 → Fin 4

/-- Gets the sum of labels on a face -/
def face_sum (l : TetrahedronLabeling) (f : Face) : Nat :=
  (f 0).val + (f 1).val + (f 2).val

/-- Checks if all faces have the same sum -/
def all_faces_equal_sum (l : TetrahedronLabeling) : Prop :=
  ∀ f₁ f₂ : Face, face_sum l f₁ = face_sum l f₂

theorem no_valid_tetrahedron_labeling :
  ¬∃ l : TetrahedronLabeling, is_valid_labeling l ∧ all_faces_equal_sum l := by
  sorry

end NUMINAMATH_CALUDE_no_valid_tetrahedron_labeling_l2394_239419


namespace NUMINAMATH_CALUDE_sum_three_not_all_less_than_one_l2394_239482

theorem sum_three_not_all_less_than_one (a b c : ℝ) (h : a + b + c = 3) :
  ¬(a < 1 ∧ b < 1 ∧ c < 1) := by sorry

end NUMINAMATH_CALUDE_sum_three_not_all_less_than_one_l2394_239482


namespace NUMINAMATH_CALUDE_intersection_points_roots_l2394_239404

theorem intersection_points_roots (x y : ℝ) : 
  (∃ x, x^2 - 3*x = 0 ∧ x ≠ 0 ∧ x ≠ 3) ∨
  (∀ x, x = x - 3 → x^2 - 3*x ≠ 0) :=
by sorry

#check intersection_points_roots

end NUMINAMATH_CALUDE_intersection_points_roots_l2394_239404


namespace NUMINAMATH_CALUDE_taxi_service_comparison_l2394_239472

-- Define the taxi services
structure TaxiService where
  initialFee : ℚ
  chargePerUnit : ℚ
  unitDistance : ℚ

def jimTaxi : TaxiService := { initialFee := 2.25, chargePerUnit := 0.35, unitDistance := 2/5 }
def susanTaxi : TaxiService := { initialFee := 3.00, chargePerUnit := 0.40, unitDistance := 1/3 }
def johnTaxi : TaxiService := { initialFee := 1.75, chargePerUnit := 0.30, unitDistance := 1/4 }

-- Function to calculate total charge
def totalCharge (service : TaxiService) (distance : ℚ) : ℚ :=
  service.initialFee + (distance / service.unitDistance).ceil * service.chargePerUnit

-- Theorem to prove
theorem taxi_service_comparison :
  let tripDistance : ℚ := 3.6
  let jimCharge := totalCharge jimTaxi tripDistance
  let susanCharge := totalCharge susanTaxi tripDistance
  let johnCharge := totalCharge johnTaxi tripDistance
  (jimCharge < johnCharge) ∧ (johnCharge < susanCharge) := by sorry

end NUMINAMATH_CALUDE_taxi_service_comparison_l2394_239472


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_water_percentage_is_90_l2394_239423

/-- The percentage of water in fresh grapes, given the conditions of the drying process -/
theorem water_percentage_in_fresh_grapes : ℝ → Prop :=
  fun p =>
    let fresh_weight : ℝ := 25
    let dried_weight : ℝ := 3.125
    let dried_water_percentage : ℝ := 20
    let fresh_solid_content : ℝ := fresh_weight * (100 - p) / 100
    let dried_solid_content : ℝ := dried_weight * (100 - dried_water_percentage) / 100
    fresh_solid_content = dried_solid_content →
    p = 90

/-- The theorem stating that the water percentage in fresh grapes is 90% -/
theorem water_percentage_is_90 : water_percentage_in_fresh_grapes 90 := by
  sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_water_percentage_is_90_l2394_239423


namespace NUMINAMATH_CALUDE_three_cuts_make_5x5_mat_l2394_239432

/-- Represents a rectangular piece of cloth -/
structure Cloth where
  rows : ℕ
  cols : ℕ
  checkered : Bool

/-- Represents a cut on the cloth -/
inductive Cut
  | Vertical (col : ℕ)
  | Horizontal (row : ℕ)

/-- Represents the result of cutting a cloth -/
def cut_result (c : Cloth) (cut : Cut) : Cloth × Cloth :=
  match cut with
  | Cut.Vertical col => ⟨⟨c.rows, col, c.checkered⟩, ⟨c.rows, c.cols - col, c.checkered⟩⟩
  | Cut.Horizontal row => ⟨⟨row, c.cols, c.checkered⟩, ⟨c.rows - row, c.cols, c.checkered⟩⟩

/-- Checks if a cloth can form a 5x5 mat -/
def is_5x5_mat (c : Cloth) : Bool :=
  c.rows = 5 && c.cols = 5 && c.checkered

/-- The main theorem -/
theorem three_cuts_make_5x5_mat :
  ∃ (cut1 cut2 cut3 : Cut),
    let initial_cloth := Cloth.mk 6 7 true
    let (c1, c2) := cut_result initial_cloth cut1
    let (c3, c4) := cut_result c1 cut2
    let (c5, c6) := cut_result c2 cut3
    ∃ (final_cloth : Cloth),
      is_5x5_mat final_cloth ∧
      (final_cloth.rows * final_cloth.cols =
       c3.rows * c3.cols + c4.rows * c4.cols + c5.rows * c5.cols + c6.rows * c6.cols) :=
by
  sorry


end NUMINAMATH_CALUDE_three_cuts_make_5x5_mat_l2394_239432


namespace NUMINAMATH_CALUDE_square_root_of_a_minus_b_l2394_239463

theorem square_root_of_a_minus_b (a b : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (a + 3)^2 = x ∧ (2*a - 6)^2 = x) →
  (b = -8) →
  Real.sqrt (a - b) = 3 := by sorry

end NUMINAMATH_CALUDE_square_root_of_a_minus_b_l2394_239463


namespace NUMINAMATH_CALUDE_new_yellow_tint_percentage_l2394_239439

/-- Calculates the new percentage of yellow tint after adding more yellow tint to a mixture -/
theorem new_yellow_tint_percentage
  (initial_volume : ℝ)
  (initial_yellow_percentage : ℝ)
  (added_yellow_volume : ℝ)
  (h1 : initial_volume = 40)
  (h2 : initial_yellow_percentage = 0.25)
  (h3 : added_yellow_volume = 10) :
  let initial_yellow_volume := initial_volume * initial_yellow_percentage
  let new_yellow_volume := initial_yellow_volume + added_yellow_volume
  let new_total_volume := initial_volume + added_yellow_volume
  new_yellow_volume / new_total_volume = 0.4 := by
sorry


end NUMINAMATH_CALUDE_new_yellow_tint_percentage_l2394_239439


namespace NUMINAMATH_CALUDE_alexanders_paintings_l2394_239429

/-- The number of paintings at each new gallery given Alexander's drawing conditions -/
theorem alexanders_paintings (first_gallery_paintings : ℕ) (new_galleries : ℕ) 
  (pencils_per_painting : ℕ) (signing_pencils_per_gallery : ℕ) (total_pencils : ℕ) :
  first_gallery_paintings = 9 →
  new_galleries = 5 →
  pencils_per_painting = 4 →
  signing_pencils_per_gallery = 2 →
  total_pencils = 88 →
  ∃ (paintings_per_new_gallery : ℕ),
    paintings_per_new_gallery = 2 ∧
    total_pencils = 
      first_gallery_paintings * pencils_per_painting + 
      new_galleries * paintings_per_new_gallery * pencils_per_painting +
      (new_galleries + 1) * signing_pencils_per_gallery :=
by sorry

end NUMINAMATH_CALUDE_alexanders_paintings_l2394_239429


namespace NUMINAMATH_CALUDE_salary_change_percentage_l2394_239490

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 84 / 100 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l2394_239490


namespace NUMINAMATH_CALUDE_inverse_inequality_l2394_239462

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l2394_239462


namespace NUMINAMATH_CALUDE_experts_win_probability_l2394_239424

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Viewers winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def expert_score : ℕ := 3

/-- The current score of Viewers -/
def viewer_score : ℕ := 4

/-- The number of rounds needed to win the game -/
def winning_score : ℕ := 6

/-- The probability of Experts winning the game from the current state -/
theorem experts_win_probability : 
  p^4 + 4 * p^3 * q = 0.4752 := by sorry

end NUMINAMATH_CALUDE_experts_win_probability_l2394_239424


namespace NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l2394_239488

theorem ceiling_fraction_evaluation : 
  (⌈(23 : ℝ) / 9 - ⌈(35 : ℝ) / 23⌉⌉) / (⌈(35 : ℝ) / 9 + ⌈(9 : ℝ) * 23 / 35⌉⌉) = (1 : ℝ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l2394_239488


namespace NUMINAMATH_CALUDE_interval_equivalence_l2394_239474

theorem interval_equivalence (x : ℝ) : 
  (1/4 < x ∧ x < 1/2) ↔ (1 < 5*x ∧ 5*x < 3) ∧ (2 < 8*x ∧ 8*x < 4) := by
  sorry

end NUMINAMATH_CALUDE_interval_equivalence_l2394_239474


namespace NUMINAMATH_CALUDE_coefficient_x3y3_in_x_plus_y_to_6_l2394_239475

theorem coefficient_x3y3_in_x_plus_y_to_6 :
  (Finset.range 7).sum (fun k => (Nat.choose 6 k : ℕ) * 
    (if k = 3 then 1 else 0)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y3_in_x_plus_y_to_6_l2394_239475


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2394_239457

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2394_239457


namespace NUMINAMATH_CALUDE_carpooling_arrangements_count_l2394_239443

/-- Represents the last digit of a license plate -/
inductive LicensePlateEnding
| Nine
| Zero
| Two
| One
| Five

/-- Represents a day in the carpooling period -/
inductive Day
| Five
| Six
| Seven
| Eight
| Nine

def is_odd_day (d : Day) : Bool :=
  match d with
  | Day.Five | Day.Seven | Day.Nine => true
  | _ => false

def is_even_ending (e : LicensePlateEnding) : Bool :=
  match e with
  | LicensePlateEnding.Zero | LicensePlateEnding.Two => true
  | _ => false

def is_valid_car (d : Day) (e : LicensePlateEnding) : Bool :=
  (is_odd_day d && !is_even_ending e) || (!is_odd_day d && is_even_ending e)

/-- Represents a carpooling arrangement for the 5-day period -/
def CarpoolingArrangement := Day → LicensePlateEnding

def is_valid_arrangement (arr : CarpoolingArrangement) : Prop :=
  (∀ d, is_valid_car d (arr d)) ∧
  (∃! d, arr d = LicensePlateEnding.Nine)

def number_of_arrangements : ℕ := sorry

theorem carpooling_arrangements_count :
  number_of_arrangements = 80 := by sorry

end NUMINAMATH_CALUDE_carpooling_arrangements_count_l2394_239443


namespace NUMINAMATH_CALUDE_not_decreasing_on_interval_l2394_239450

-- Define a real-valued function on the real line
variable (f : ℝ → ℝ)

-- State the theorem
theorem not_decreasing_on_interval (h : f (-1) < f 1) :
  ¬(∀ x y : ℝ, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → x ≤ y → f x ≥ f y) :=
by sorry

end NUMINAMATH_CALUDE_not_decreasing_on_interval_l2394_239450


namespace NUMINAMATH_CALUDE_evelyns_marbles_l2394_239499

/-- The number of marbles Evelyn has in total -/
def total_marbles (initial : ℕ) (from_henry : ℕ) (from_grace : ℕ) (cards : ℕ) (marbles_per_card : ℕ) : ℕ :=
  initial + from_henry + from_grace + cards * marbles_per_card

/-- Theorem stating that Evelyn's total number of marbles is 140 -/
theorem evelyns_marbles :
  total_marbles 95 9 12 6 4 = 140 := by
  sorry

#eval total_marbles 95 9 12 6 4

end NUMINAMATH_CALUDE_evelyns_marbles_l2394_239499


namespace NUMINAMATH_CALUDE_function_composition_ratio_l2394_239409

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l2394_239409


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2394_239431

theorem quadratic_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 11 = 25) → 
  (d^2 - 6*d + 11 = 25) → 
  c ≥ d → 
  c + 2*d = 9 - Real.sqrt 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2394_239431


namespace NUMINAMATH_CALUDE_f_property_implies_n_times_s_eq_14_l2394_239426

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the main property of f
axiom f_property (x y z : ℝ) : f (x^2 + y * f z) = x * f x + z * f y + y^2

-- Define n as the number of possible values of f(5)
def n : ℕ := sorry

-- Define s as the sum of all possible values of f(5)
def s : ℝ := sorry

-- State the theorem to be proved
theorem f_property_implies_n_times_s_eq_14 : n * s = 14 := by sorry

end NUMINAMATH_CALUDE_f_property_implies_n_times_s_eq_14_l2394_239426


namespace NUMINAMATH_CALUDE_type_T_machine_time_l2394_239405

-- Define the time for a type B machine to complete the job
def time_B : ℝ := 7

-- Define the time for 2 type T machines and 3 type B machines to complete the job together
def time_combined : ℝ := 1.2068965517241381

-- Define the time for a type T machine to complete the job
def time_T : ℝ := 5

-- Theorem statement
theorem type_T_machine_time : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |time_T - (1 / ((1 / time_combined) - (3 / (2 * time_B))))| < ε :=
sorry

end NUMINAMATH_CALUDE_type_T_machine_time_l2394_239405


namespace NUMINAMATH_CALUDE_age_difference_z_younger_than_x_l2394_239455

-- Define variables for ages
variable (X Y Z : ℕ)

-- Define the condition from the problem
def age_condition (X Y Z : ℕ) : Prop := X + Y = Y + Z + 19

-- Theorem to prove
theorem age_difference (h : age_condition X Y Z) : X - Z = 19 :=
by sorry

-- Convert years to decades
def years_to_decades (years : ℕ) : ℚ := (years : ℚ) / 10

-- Theorem to prove the final result
theorem z_younger_than_x (h : age_condition X Y Z) : 
  years_to_decades (X - Z) = 1.9 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_z_younger_than_x_l2394_239455


namespace NUMINAMATH_CALUDE_red_bottles_count_l2394_239440

/-- The number of red water bottles in the fridge -/
def red_bottles : ℕ := 2

/-- The number of black water bottles in the fridge -/
def black_bottles : ℕ := 3

/-- The number of blue water bottles in the fridge -/
def blue_bottles : ℕ := 4

/-- The total number of water bottles initially in the fridge -/
def total_bottles : ℕ := 9

theorem red_bottles_count : red_bottles + black_bottles + blue_bottles = total_bottles := by
  sorry

end NUMINAMATH_CALUDE_red_bottles_count_l2394_239440


namespace NUMINAMATH_CALUDE_point_position_on_line_l2394_239478

/-- Given five points on a line, prove the position of a point P satisfying a specific ratio condition -/
theorem point_position_on_line (a b c d : ℝ) :
  let O := (0 : ℝ)
  let A := (2*a : ℝ)
  let B := (3*b : ℝ)
  let C := (4*c : ℝ)
  let D := (5*d : ℝ)
  ∃ P : ℝ, B ≤ P ∧ P ≤ C ∧
    (P - A)^2 * (C - P) = (D - P)^2 * (P - B) →
    P = (8*a*c - 15*b*d) / (8*c - 15*d - 6*b + 4*a) :=
by sorry

end NUMINAMATH_CALUDE_point_position_on_line_l2394_239478


namespace NUMINAMATH_CALUDE_amount_of_b_l2394_239493

theorem amount_of_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 100) (h4 : (3/10) * a = (1/5) * b) : b = 60 := by
  sorry

end NUMINAMATH_CALUDE_amount_of_b_l2394_239493


namespace NUMINAMATH_CALUDE_sin_deg_rad_solutions_l2394_239430

def sin_deg_rad_eq (x : ℝ) : Prop := Real.sin x = Real.sin (x * Real.pi / 180)

theorem sin_deg_rad_solutions :
  ∃ (S : Finset ℝ), S.card = 10 ∧
    (∀ x ∈ S, 0 ≤ x ∧ x ≤ 90 ∧ sin_deg_rad_eq x) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 90 ∧ sin_deg_rad_eq x → x ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_sin_deg_rad_solutions_l2394_239430


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l2394_239420

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β)
  (h3 : perpendicular m α) (h4 : parallel m β) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l2394_239420


namespace NUMINAMATH_CALUDE_point_on_y_axis_l2394_239425

/-- If a point P(a-1, a^2-9) lies on the y-axis, then its coordinates are (0, -8). -/
theorem point_on_y_axis (a : ℝ) :
  (a - 1 = 0) → (a - 1, a^2 - 9) = (0, -8) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l2394_239425


namespace NUMINAMATH_CALUDE_optimal_tic_tac_toe_draw_l2394_239452

/-- Represents a player in Tic-Tac-Toe -/
inductive Player : Type
| X : Player
| O : Player

/-- Represents a position on the Tic-Tac-Toe board -/
inductive Position : Type
| one | two | three | four | five | six | seven | eight | nine

/-- Represents the state of a Tic-Tac-Toe game -/
structure GameState :=
  (board : Position → Option Player)
  (currentPlayer : Player)

/-- Represents an optimal move in Tic-Tac-Toe -/
def OptimalMove : GameState → Position → Prop := sorry

/-- Represents the outcome of a Tic-Tac-Toe game -/
inductive GameOutcome : Type
| Draw : GameOutcome
| Win : Player → GameOutcome

/-- Plays a full game of Tic-Tac-Toe with optimal moves -/
def playOptimalGame : GameState → GameOutcome := sorry

/-- Theorem: Every game of Tic-Tac-Toe between optimal players ends in a draw -/
theorem optimal_tic_tac_toe_draw :
  ∀ (initialState : GameState),
  (∀ (state : GameState) (move : Position), OptimalMove state move → 
    playOptimalGame (sorry : GameState) = playOptimalGame state) →
  playOptimalGame initialState = GameOutcome.Draw :=
sorry

end NUMINAMATH_CALUDE_optimal_tic_tac_toe_draw_l2394_239452


namespace NUMINAMATH_CALUDE_silver_beads_count_l2394_239476

/-- Represents the number of beads in a necklace. -/
structure BeadCount where
  total : Nat
  blue : Nat
  red : Nat
  white : Nat
  silver : Nat

/-- Conditions for Michelle's necklace. -/
def michellesNecklace : BeadCount where
  total := 40
  blue := 5
  red := 2 * 5
  white := 5 + (2 * 5)
  silver := 40 - (5 + (2 * 5) + (5 + (2 * 5)))

/-- Theorem stating that the number of silver beads in Michelle's necklace is 10. -/
theorem silver_beads_count : michellesNecklace.silver = 10 := by
  sorry

#eval michellesNecklace.silver

end NUMINAMATH_CALUDE_silver_beads_count_l2394_239476


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l2394_239445

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle2D where
  v1 : Point2D
  v2 : Point2D
  v3 : Point2D

/-- Check if three lines are parallel -/
def are_parallel (l1 l2 l3 : Line2D) : Prop :=
  ∃ (k1 k2 : ℝ), l1.a = k1 * l2.a ∧ l1.b = k1 * l2.b ∧
                 l1.a = k2 * l3.a ∧ l1.b = k2 * l3.b

/-- Check if a point lies on a line -/
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a line passes through a point -/
def line_through_point (l : Line2D) (p : Point2D) : Prop :=
  point_on_line p l

/-- Check if a triangle's vertices lie on given lines -/
def triangle_vertices_on_lines (t : Triangle2D) (l1 l2 l3 : Line2D) : Prop :=
  (point_on_line t.v1 l1 ∨ point_on_line t.v1 l2 ∨ point_on_line t.v1 l3) ∧
  (point_on_line t.v2 l1 ∨ point_on_line t.v2 l2 ∨ point_on_line t.v2 l3) ∧
  (point_on_line t.v3 l1 ∨ point_on_line t.v3 l2 ∨ point_on_line t.v3 l3)

/-- Check if a triangle's sides (or extensions) pass through given points -/
def triangle_sides_through_points (t : Triangle2D) (p1 p2 p3 : Point2D) : Prop :=
  ∃ (l1 l2 l3 : Line2D),
    (point_on_line t.v1 l1 ∧ point_on_line t.v2 l1 ∧ line_through_point l1 p1) ∧
    (point_on_line t.v2 l2 ∧ point_on_line t.v3 l2 ∧ line_through_point l2 p2) ∧
    (point_on_line t.v3 l3 ∧ point_on_line t.v1 l3 ∧ line_through_point l3 p3)

theorem triangle_construction_theorem 
  (l1 l2 l3 : Line2D) 
  (p1 p2 p3 : Point2D) 
  (h_parallel : are_parallel l1 l2 l3) :
  ∃ (t : Triangle2D), 
    triangle_vertices_on_lines t l1 l2 l3 ∧ 
    triangle_sides_through_points t p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l2394_239445


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2394_239406

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Real.sqrt 2 + Complex.I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2394_239406


namespace NUMINAMATH_CALUDE_sarah_tuesday_pencils_l2394_239444

/-- The number of pencils Sarah bought on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah bought on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := 92

/-- Theorem: Sarah bought 18 pencils on Tuesday -/
theorem sarah_tuesday_pencils :
  monday_pencils + tuesday_pencils + 3 * tuesday_pencils = total_pencils :=
by sorry

end NUMINAMATH_CALUDE_sarah_tuesday_pencils_l2394_239444


namespace NUMINAMATH_CALUDE_four_folds_result_l2394_239489

/-- Represents a square piece of paper. -/
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Represents a fold on the paper. -/
inductive Fold
  | Diagonal
  | Perpendicular

/-- Represents the pattern of creases on the unfolded paper. -/
structure CreasePattern :=
  (folds : List Fold)
  (is_symmetrical : Bool)
  (center_at_mean : Bool)

/-- Function to perform a single fold. -/
def fold (s : Square) : CreasePattern :=
  sorry

/-- Function to perform four folds. -/
def four_folds (s : Square) : CreasePattern :=
  sorry

/-- Theorem stating the result of folding a square paper four times. -/
theorem four_folds_result (s : Square) :
  let pattern := four_folds s
  pattern.is_symmetrical ∧ 
  pattern.center_at_mean ∧ 
  (∃ (d p : Fold), d = Fold.Diagonal ∧ p = Fold.Perpendicular ∧ d ∈ pattern.folds ∧ p ∈ pattern.folds) :=
by sorry

end NUMINAMATH_CALUDE_four_folds_result_l2394_239489


namespace NUMINAMATH_CALUDE_only_event1_is_random_l2394_239491

-- Define the possible types of events
inductive EventType
  | Random
  | Certain
  | Impossible

-- Define the events
def event1 : EventType := EventType.Random
def event2 : EventType := EventType.Certain
def event3 : EventType := EventType.Impossible

-- Define a function to check if an event is random
def isRandomEvent (e : EventType) : Prop :=
  e = EventType.Random

-- Theorem statement
theorem only_event1_is_random :
  isRandomEvent event1 ∧ ¬isRandomEvent event2 ∧ ¬isRandomEvent event3 :=
by
  sorry


end NUMINAMATH_CALUDE_only_event1_is_random_l2394_239491


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l2394_239449

/-- Given two vectors a and b in ℝ³, where a = (-2, 3, 1) and b = (4, m, n),
    if a is parallel to b, then m + n = -8 -/
theorem parallel_vectors_sum (m n : ℝ) : 
  let a : ℝ × ℝ × ℝ := (-2, 3, 1)
  let b : ℝ × ℝ × ℝ := (4, m, n)
  (∃ (k : ℝ), b = k • a) → m + n = -8 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l2394_239449


namespace NUMINAMATH_CALUDE_min_value_expression_l2394_239448

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 1) :
  ∃ (min_val : ℝ), 
    (∀ c', c' > 2 → (3*a*c'/b + c'/(a*b) + 6/(c'-2) ≥ min_val)) ∧ 
    (∃ c'', c'' > 2 ∧ 3*a*c''/b + c''/(a*b) + 6/(c''-2) = min_val) ∧
    min_val = 1 / (a * (1 - a)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2394_239448


namespace NUMINAMATH_CALUDE_line_parameterization_l2394_239458

/-- Given a line y = 2x - 40 parameterized by (x, y) = (g(t), 20t - 14), 
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ x y, y = 2*x - 40 ↔ ∃ t, x = g t ∧ y = 20*t - 14) →
  ∀ t, g t = 10*t + 13 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l2394_239458


namespace NUMINAMATH_CALUDE_parabola_directrix_l2394_239492

/-- The equation of the directrix of the parabola x² = 4y is y = -1 -/
theorem parabola_directrix (x y : ℝ) : 
  (∀ x y, x^2 = 4*y → ∃ k, y = -k ∧ k = 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2394_239492


namespace NUMINAMATH_CALUDE_books_read_during_travel_l2394_239451

theorem books_read_during_travel (total_distance : ℝ) (distance_per_book : ℝ) : 
  total_distance = 6987.5 → 
  distance_per_book = 482.3 → 
  ⌊total_distance / distance_per_book⌋ = 14 := by
sorry

end NUMINAMATH_CALUDE_books_read_during_travel_l2394_239451


namespace NUMINAMATH_CALUDE_curve_intersection_property_m_range_l2394_239495

/-- The curve C defined by y² = 4x for x > 0 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4*p.1 ∧ p.1 > 0}

/-- The line passing through (m, 0) with slope 1/t -/
def line (m t : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = t*p.2 + m}

/-- The dot product of vectors FA and FB where F is (1, 0) -/
def dot_product (A B : ℝ × ℝ) : ℝ := (A.1 - 1)*(B.1 - 1) + A.2*B.2

theorem curve_intersection_property :
  ∃ (m : ℝ), m > 0 ∧
  ∀ (t : ℝ), ∀ (A B : ℝ × ℝ),
    A ∈ C → B ∈ C → A ∈ line m t → B ∈ line m t → A ≠ B →
    dot_product A B < 0 :=
sorry

theorem m_range (m : ℝ) :
  (∀ (t : ℝ), ∀ (A B : ℝ × ℝ),
    A ∈ C → B ∈ C → A ∈ line m t → B ∈ line m t → A ≠ B →
    dot_product A B < 0) ↔
  3 - 2*Real.sqrt 2 < m ∧ m < 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_curve_intersection_property_m_range_l2394_239495


namespace NUMINAMATH_CALUDE_horizontal_row_different_l2394_239412

/-- Represents the weight of a row of apples -/
def RowWeight : Type := ℝ

/-- Represents the arrangement of apples -/
structure AppleArrangement where
  total_apples : ℕ
  rows : ℕ
  apples_per_row : ℕ
  diagonal_weights : Fin 3 → RowWeight
  vertical_weights : Fin 3 → RowWeight
  horizontal_weight : RowWeight

/-- The given arrangement of apples satisfies the problem conditions -/
def valid_arrangement (a : AppleArrangement) : Prop :=
  a.total_apples = 9 ∧
  a.rows = 10 ∧
  a.apples_per_row = 3 ∧
  ∃ (t : RowWeight),
    (∀ i : Fin 3, a.diagonal_weights i = t) ∧
    (∀ i : Fin 3, a.vertical_weights i = t) ∧
    a.horizontal_weight ≠ t

theorem horizontal_row_different (a : AppleArrangement) 
  (h : valid_arrangement a) : 
  ∃ (t : RowWeight), 
    (∀ i : Fin 3, a.diagonal_weights i = t) ∧ 
    (∀ i : Fin 3, a.vertical_weights i = t) ∧ 
    a.horizontal_weight ≠ t := by
  sorry

#check horizontal_row_different

end NUMINAMATH_CALUDE_horizontal_row_different_l2394_239412


namespace NUMINAMATH_CALUDE_equation_solution_l2394_239486

theorem equation_solution : ∃ x : ℝ, 9 - x - 2 * (31 - x) = 27 ∧ x = 80 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2394_239486


namespace NUMINAMATH_CALUDE_average_height_of_trees_l2394_239442

theorem average_height_of_trees (elm_height oak_height pine_height : ℚ) : 
  elm_height = 35 / 3 →
  oak_height = 107 / 6 →
  pine_height = 31 / 2 →
  (elm_height + oak_height + pine_height) / 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_height_of_trees_l2394_239442


namespace NUMINAMATH_CALUDE_phone_reps_calculation_l2394_239418

/-- The number of hours each phone rep works per day -/
def hours_per_day : ℕ := 8

/-- The hourly wage of each phone rep in dollars -/
def hourly_wage : ℚ := 14

/-- The number of days worked -/
def days_worked : ℕ := 5

/-- The total payment for all new employees after 5 days in dollars -/
def total_payment : ℚ := 28000

/-- The number of new phone reps the company wants to hire -/
def num_phone_reps : ℕ := 50

theorem phone_reps_calculation :
  (hours_per_day * hourly_wage * days_worked : ℚ) * num_phone_reps = total_payment :=
by sorry

end NUMINAMATH_CALUDE_phone_reps_calculation_l2394_239418


namespace NUMINAMATH_CALUDE_negation_equivalence_l2394_239459

-- Define the universe of discourse
variable (Teacher : Type)

-- Define the predicates
variable (loves_math : Teacher → Prop)
variable (dislikes_math : Teacher → Prop)

-- State the theorem
theorem negation_equivalence :
  (∃ t : Teacher, dislikes_math t) ↔ ¬(∀ t : Teacher, loves_math t) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2394_239459


namespace NUMINAMATH_CALUDE_number_calculation_l2394_239438

theorem number_calculation (n : ℝ) : 
  0.1 * 0.3 * ((Real.sqrt (0.5 * n))^2) = 90 → n = 6000 := by
sorry

end NUMINAMATH_CALUDE_number_calculation_l2394_239438


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2394_239466

theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), y = k * x + 1 ∧ x^2 + y^2 = 2 ∧ (x ≠ 0 ∨ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2394_239466


namespace NUMINAMATH_CALUDE_exam_pass_rate_l2394_239477

theorem exam_pass_rate (hindi : ℝ) (english : ℝ) (math : ℝ) 
  (hindi_english : ℝ) (hindi_math : ℝ) (english_math : ℝ) (all_three : ℝ)
  (h1 : hindi = 25) (h2 : english = 48) (h3 : math = 35)
  (h4 : hindi_english = 27) (h5 : hindi_math = 20) (h6 : english_math = 15)
  (h7 : all_three = 10) :
  100 - (hindi + english + math - hindi_english - hindi_math - english_math + all_three) = 44 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_rate_l2394_239477


namespace NUMINAMATH_CALUDE_polynomial_factors_imply_absolute_value_l2394_239434

theorem polynomial_factors_imply_absolute_value (h k : ℝ) : 
  (∀ x, (x + 2) * (x - 1) ∣ (3 * x^3 - h * x + k)) →
  |3 * h - 2 * k| = 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factors_imply_absolute_value_l2394_239434


namespace NUMINAMATH_CALUDE_triangle_diagram_solutions_l2394_239435

theorem triangle_diagram_solutions : 
  ∃! (solutions : List (ℕ × ℕ × ℕ)), 
    solutions.length = 6 ∧ 
    ∀ (a b c : ℕ), (a, b, c) ∈ solutions ↔ 
      (14 * 4 * a = 14 * 6 * c ∧ 
       14 * 4 * a = a * b * c ∧ 
       a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_diagram_solutions_l2394_239435


namespace NUMINAMATH_CALUDE_collinear_implies_coplanar_not_coplanar_implies_not_collinear_l2394_239417

-- Define a point in space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define collinearity for three points
def collinear (p q r : Point3D) : Prop := sorry

-- Define coplanarity for four points
def coplanar (p q r s : Point3D) : Prop := sorry

-- Theorem 1: If three points are collinear, then four points are coplanar
theorem collinear_implies_coplanar (p q r s : Point3D) :
  (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) →
  coplanar p q r s := by sorry

-- Theorem 2: If four points are not coplanar, then no three points are collinear
theorem not_coplanar_implies_not_collinear (p q r s : Point3D) :
  ¬(coplanar p q r s) →
  ¬(collinear p q r) ∧ ¬(collinear p q s) ∧ ¬(collinear p r s) ∧ ¬(collinear q r s) := by sorry

end NUMINAMATH_CALUDE_collinear_implies_coplanar_not_coplanar_implies_not_collinear_l2394_239417


namespace NUMINAMATH_CALUDE_beef_stew_duration_l2394_239441

/-- The number of days the beef stew lasts for 2 people -/
def days_for_two : ℝ := 7

/-- The number of days the beef stew lasts for 5 people -/
def days_for_five : ℝ := 2.8

/-- The number of people in the original scenario -/
def original_people : ℕ := 2

/-- The number of people in the new scenario -/
def new_people : ℕ := 5

theorem beef_stew_duration :
  days_for_two * original_people = days_for_five * new_people :=
by sorry

end NUMINAMATH_CALUDE_beef_stew_duration_l2394_239441


namespace NUMINAMATH_CALUDE_base_10_to_base_5_l2394_239411

theorem base_10_to_base_5 : ∃ (a b c d : ℕ), 
  255 = a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0 ∧ 
  a = 2 ∧ b = 1 ∧ c = 0 ∧ d = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_5_l2394_239411


namespace NUMINAMATH_CALUDE_triangle_theorem_l2394_239461

theorem triangle_theorem (a b c A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = π) (h5 : 2 * c * Real.cos C + b * Real.cos A + a * Real.cos B = 0) :
  C = 2 * π / 3 ∧ 
  (c = 3 → A = π / 6 → (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2394_239461


namespace NUMINAMATH_CALUDE_biscuits_butter_cookies_difference_l2394_239465

-- Define the number of cookies baked in the morning and afternoon
def morning_butter_cookies : ℕ := 20
def morning_biscuits : ℕ := 40
def afternoon_butter_cookies : ℕ := 10
def afternoon_biscuits : ℕ := 20

-- Define the total number of each type of cookie
def total_butter_cookies : ℕ := morning_butter_cookies + afternoon_butter_cookies
def total_biscuits : ℕ := morning_biscuits + afternoon_biscuits

-- Theorem statement
theorem biscuits_butter_cookies_difference :
  total_biscuits - total_butter_cookies = 30 := by
  sorry

end NUMINAMATH_CALUDE_biscuits_butter_cookies_difference_l2394_239465


namespace NUMINAMATH_CALUDE_cricket_average_increase_l2394_239433

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  newInningsRuns : ℕ

/-- Calculates the increase in average runs per innings -/
def averageIncrease (player : CricketPlayer) : ℚ :=
  let oldAverage : ℚ := player.totalRuns / player.innings
  let newTotal : ℕ := player.totalRuns + player.newInningsRuns
  let newAverage : ℚ := newTotal / (player.innings + 1)
  newAverage - oldAverage

/-- Theorem stating the increase in average for the given scenario -/
theorem cricket_average_increase :
  ∀ (player : CricketPlayer),
  player.innings = 10 →
  player.totalRuns = 370 →
  player.newInningsRuns = 81 →
  averageIncrease player = 4 := by
  sorry


end NUMINAMATH_CALUDE_cricket_average_increase_l2394_239433


namespace NUMINAMATH_CALUDE_inequality_solution_l2394_239400

theorem inequality_solution (x : ℝ) : 
  3/20 + |x - 13/60| < 7/30 ↔ 2/15 < x ∧ x < 3/10 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2394_239400


namespace NUMINAMATH_CALUDE_complex_fraction_equals_point_l2394_239446

theorem complex_fraction_equals_point : (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_point_l2394_239446


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2394_239460

-- Define the polynomial
def p (x : ℝ) : ℝ := -3*(x^8 - 2*x^5 + 4*x^3 - 6) + 5*(x^4 + 3*x^2) - 4*(x^6 - 5)

-- Theorem: The sum of coefficients of p is 45
theorem sum_of_coefficients : p 1 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2394_239460


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2394_239468

theorem quadratic_inequality (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ ≠ x₂ →
  y₁ = -x₁^2 →
  y₂ = -x₂^2 →
  x₁ * x₂ > x₂^2 →
  y₁ < y₂ := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2394_239468


namespace NUMINAMATH_CALUDE_gasoline_consumption_rate_l2394_239453

/-- Represents the gasoline consumption problem --/
structure GasolineProblem where
  initial_gasoline : ℝ
  supermarket_distance : ℝ
  farm_distance : ℝ
  partial_farm_trip : ℝ
  final_gasoline : ℝ

/-- Calculates the total distance traveled --/
def total_distance (p : GasolineProblem) : ℝ :=
  2 * p.supermarket_distance + 2 * p.partial_farm_trip + p.farm_distance

/-- Calculates the total gasoline consumed --/
def gasoline_consumed (p : GasolineProblem) : ℝ :=
  p.initial_gasoline - p.final_gasoline

/-- Theorem stating the gasoline consumption rate --/
theorem gasoline_consumption_rate (p : GasolineProblem) 
  (h1 : p.initial_gasoline = 12)
  (h2 : p.supermarket_distance = 5)
  (h3 : p.farm_distance = 6)
  (h4 : p.partial_farm_trip = 2)
  (h5 : p.final_gasoline = 2) :
  total_distance p / gasoline_consumed p = 2 := by sorry

end NUMINAMATH_CALUDE_gasoline_consumption_rate_l2394_239453


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2394_239410

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  Real.sqrt ((2 - (-6))^2 + (y - 5)^2) = 10 → 
  y = 11 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2394_239410


namespace NUMINAMATH_CALUDE_symmetric_trapezoid_feasibility_l2394_239498

/-- Represents a symmetric trapezoid with one parallel side equal to the legs -/
structure SymmetricTrapezoid where
  /-- Length of the legs -/
  a : ℝ
  /-- Distance from the intersection point of the diagonals to one endpoint of the other parallel side -/
  b : ℝ
  /-- Assumption that a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0

/-- Theorem stating the feasibility condition for constructing the symmetric trapezoid -/
theorem symmetric_trapezoid_feasibility (t : SymmetricTrapezoid) :
  (∃ (trapezoid : SymmetricTrapezoid), trapezoid.a = t.a ∧ trapezoid.b = t.b) ↔ 3 * t.b > 2 * t.a := by
  sorry

end NUMINAMATH_CALUDE_symmetric_trapezoid_feasibility_l2394_239498


namespace NUMINAMATH_CALUDE_matthew_hotdogs_l2394_239408

/-- The number of hotdogs Matthew needs to cook for his children -/
def total_hotdogs : ℕ :=
  let ella_emma_hotdogs := 2 + 2
  let luke_hotdogs := 2 * ella_emma_hotdogs
  let hunter_hotdogs := (3 * ella_emma_hotdogs) / 2
  ella_emma_hotdogs + luke_hotdogs + hunter_hotdogs

theorem matthew_hotdogs : total_hotdogs = 18 := by
  sorry

end NUMINAMATH_CALUDE_matthew_hotdogs_l2394_239408


namespace NUMINAMATH_CALUDE_bread_cost_l2394_239494

/-- The cost of each loaf of bread given the total number of loaves, 
    cost of peanut butter, initial amount of money, and amount left over. -/
theorem bread_cost (num_loaves : ℕ) (peanut_butter_cost initial_money leftover : ℚ) :
  num_loaves = 3 ∧ 
  peanut_butter_cost = 2 ∧ 
  initial_money = 14 ∧ 
  leftover = 5.25 →
  (initial_money - leftover - peanut_butter_cost) / num_loaves = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_l2394_239494


namespace NUMINAMATH_CALUDE_counterexample_exists_l2394_239402

-- Define the set of numbers to check
def numbers : List Nat := [25, 35, 39, 49, 51]

-- Define what it means for a number to be composite
def isComposite (n : Nat) : Prop := ¬ Nat.Prime n

-- Define the counterexample property
def isCounterexample (n : Nat) : Prop := isComposite n ∧ Nat.Prime (n - 2)

-- Theorem to prove
theorem counterexample_exists : ∃ n ∈ numbers, isCounterexample n := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2394_239402


namespace NUMINAMATH_CALUDE_room_width_calculation_l2394_239484

/-- Given a room with length 5.5 m and a floor paving cost of 400 Rs per sq metre
    resulting in a total cost of 8250 Rs, prove that the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) :
  length = 5.5 →
  cost_per_sqm = 400 →
  total_cost = 8250 →
  width = total_cost / cost_per_sqm / length →
  width = 3.75 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l2394_239484


namespace NUMINAMATH_CALUDE_smallest_angle_satisfying_condition_l2394_239473

theorem smallest_angle_satisfying_condition : 
  ∃ (x : ℝ), x > 0 ∧ x < (π / 180) * 360 ∧ 
  Real.sin (4 * x) * Real.sin (5 * x) = Real.cos (4 * x) * Real.cos (5 * x) ∧
  (∀ (y : ℝ), 0 < y ∧ y < x → 
    Real.sin (4 * y) * Real.sin (5 * y) ≠ Real.cos (4 * y) * Real.cos (5 * y)) ∧
  x = (π / 180) * 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_satisfying_condition_l2394_239473


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l2394_239469

theorem condition_neither_sufficient_nor_necessary
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ)
  (ha₁ : a₁ ≠ 0) (hb₁ : b₁ ≠ 0) (hc₁ : c₁ ≠ 0)
  (ha₂ : a₂ ≠ 0) (hb₂ : b₂ ≠ 0) (hc₂ : c₂ ≠ 0) :
  ¬(((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)) ↔
    (∀ x, a₁ * x^2 + b₁ * x + c₁ > 0 ↔ a₂ * x^2 + b₂ * x + c₂ > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l2394_239469


namespace NUMINAMATH_CALUDE_cubic_greater_than_quadratic_l2394_239447

theorem cubic_greater_than_quadratic (x : ℝ) (h : x > 1) : x^3 > x^2 - x + 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_greater_than_quadratic_l2394_239447


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2394_239413

theorem quadratic_inequality_equivalence : 
  ∀ x : ℝ, x * (2 * x + 3) < -2 ↔ x ∈ Set.Ioo (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2394_239413


namespace NUMINAMATH_CALUDE_lemniscate_symmetric_origin_lemniscate_max_distance_squared_lemniscate_unique_equidistant_point_l2394_239421

-- Define the lemniscate curve
def Lemniscate (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; ((x + a)^2 + y^2) * ((x - a)^2 + y^2) = a^4}

-- Statement 1: Symmetry with respect to the origin
theorem lemniscate_symmetric_origin (a : ℝ) (h : a > 0) :
  ∀ (p : ℝ × ℝ), p ∈ Lemniscate a ↔ (-p.1, -p.2) ∈ Lemniscate a :=
sorry

-- Statement 2: Maximum value of |PO|^2 - a^2
theorem lemniscate_max_distance_squared (a : ℝ) (h : a > 0) :
  ∃ (p : ℝ × ℝ), p ∈ Lemniscate a ∧
    ∀ (q : ℝ × ℝ), q ∈ Lemniscate a → (p.1^2 + p.2^2) - a^2 ≥ (q.1^2 + q.2^2) - a^2 ∧
    (p.1^2 + p.2^2) - a^2 = a^2 :=
sorry

-- Statement 3: Unique point equidistant from focal points
theorem lemniscate_unique_equidistant_point (a : ℝ) (h : a > 0) :
  ∃! (p : ℝ × ℝ), p ∈ Lemniscate a ∧
    (p.1 + a)^2 + p.2^2 = (p.1 - a)^2 + p.2^2 :=
sorry

end NUMINAMATH_CALUDE_lemniscate_symmetric_origin_lemniscate_max_distance_squared_lemniscate_unique_equidistant_point_l2394_239421


namespace NUMINAMATH_CALUDE_roots_can_change_l2394_239470

-- Define the concept of a root being lost
def root_lost (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x ∧ ¬(Real.tan (f x) = Real.tan (g x))

-- Define the concept of an extraneous root appearing
def extraneous_root (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x ≠ g x ∧ Real.tan (f x) = Real.tan (g x)

-- Theorem stating that roots can be lost and extraneous roots can appear
theorem roots_can_change (f g : ℝ → ℝ) : 
  (root_lost f g) ∧ (extraneous_root f g) := by
  sorry


end NUMINAMATH_CALUDE_roots_can_change_l2394_239470


namespace NUMINAMATH_CALUDE_waiter_earnings_problem_l2394_239436

/-- Calculates the total earnings of a waiter during a shift --/
def waiter_earnings (customers : ℕ) (no_tip : ℕ) (tip_3 : ℕ) (tip_4 : ℕ) (tip_5 : ℕ) 
  (couple_groups : ℕ) (pool_contribution_rate : ℚ) (meal_cost : ℚ) : ℚ :=
  let total_tips := 3 * tip_3 + 4 * tip_4 + 5 * tip_5
  let net_tips := total_tips * (1 - pool_contribution_rate)
  net_tips - meal_cost

/-- Theorem stating that the waiter's earnings are $64.20 given the problem conditions --/
theorem waiter_earnings_problem : 
  waiter_earnings 25 5 8 6 6 2 (1/10) 6 = 321/5 := by
  sorry

end NUMINAMATH_CALUDE_waiter_earnings_problem_l2394_239436


namespace NUMINAMATH_CALUDE_reciprocal_sum_l2394_239422

theorem reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 = b^2 + b*c) (h2 : b^2 = c^2 + a*c) : 
  1/c = 1/a + 1/b := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l2394_239422


namespace NUMINAMATH_CALUDE_print_shop_copies_l2394_239487

theorem print_shop_copies (x_price y_price difference : ℚ) (h1 : x_price = 1.25)
  (h2 : y_price = 2.75) (h3 : difference = 90) :
  ∃ n : ℚ, n * y_price = n * x_price + difference ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_copies_l2394_239487


namespace NUMINAMATH_CALUDE_scheduleArrangements_eq_180_l2394_239497

/-- The number of ways to schedule 4 out of 6 people over 3 days -/
def scheduleArrangements : ℕ :=
  Nat.choose 6 1 * Nat.choose 5 1 * Nat.choose 4 2

/-- Theorem stating that the number of scheduling arrangements is 180 -/
theorem scheduleArrangements_eq_180 : scheduleArrangements = 180 := by
  sorry

end NUMINAMATH_CALUDE_scheduleArrangements_eq_180_l2394_239497


namespace NUMINAMATH_CALUDE_union_equals_S_l2394_239467

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem union_equals_S : S ∪ T = S := by sorry

end NUMINAMATH_CALUDE_union_equals_S_l2394_239467


namespace NUMINAMATH_CALUDE_excess_amount_correct_l2394_239471

/-- The amount in excess of which the import tax is applied -/
def excess_amount : ℝ := 1000

/-- The total value of the item -/
def total_value : ℝ := 2560

/-- The import tax rate -/
def tax_rate : ℝ := 0.07

/-- The amount of import tax paid -/
def tax_paid : ℝ := 109.20

/-- Theorem stating that the excess amount is correct given the conditions -/
theorem excess_amount_correct : 
  tax_rate * (total_value - excess_amount) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_excess_amount_correct_l2394_239471


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l2394_239407

theorem wire_ratio_proof (total_length : ℝ) (short_length : ℝ) :
  total_length = 70 →
  short_length = 20 →
  short_length / (total_length - short_length) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l2394_239407


namespace NUMINAMATH_CALUDE_negation_equivalence_l2394_239496

theorem negation_equivalence :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2394_239496


namespace NUMINAMATH_CALUDE_difference_of_squares_2006_l2394_239481

def is_difference_of_squares (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 - b^2

theorem difference_of_squares_2006 :
  ¬(is_difference_of_squares 2006) ∧
  (is_difference_of_squares 2004) ∧
  (is_difference_of_squares 2005) ∧
  (is_difference_of_squares 2007) :=
sorry

end NUMINAMATH_CALUDE_difference_of_squares_2006_l2394_239481


namespace NUMINAMATH_CALUDE_f_satisfies_all_points_l2394_239428

/-- Function representing the relationship between x and y -/
def f (x : ℝ) : ℝ := 200 - 40*x - 10*x^2

/-- The set of points given in the table -/
def points : List (ℝ × ℝ) := [(0, 200), (1, 160), (2, 80), (3, 0), (4, -120)]

/-- Theorem stating that the function f satisfies all points in the given table -/
theorem f_satisfies_all_points : ∀ (p : ℝ × ℝ), p ∈ points → f p.1 = p.2 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_all_points_l2394_239428


namespace NUMINAMATH_CALUDE_probability_through_D_l2394_239437

/-- Represents a point on the grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points --/
def numPaths (start finish : Point) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The probability of choosing a specific path --/
def pathProbability (start finish : Point) : ℚ :=
  (1 / 2) ^ (finish.x - start.x + finish.y - start.y)

theorem probability_through_D (A D B : Point)
  (hA : A = ⟨0, 0⟩)
  (hD : D = ⟨3, 1⟩)
  (hB : B = ⟨6, 3⟩) :
  (numPaths A D * numPaths D B : ℚ) * pathProbability A B / numPaths A B = 20 / 63 :=
sorry

end NUMINAMATH_CALUDE_probability_through_D_l2394_239437


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l2394_239403

theorem power_tower_mod_500 : 4^(4^(4^4)) ≡ 36 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l2394_239403
