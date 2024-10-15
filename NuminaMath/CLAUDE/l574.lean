import Mathlib

namespace NUMINAMATH_CALUDE_kids_meals_sold_l574_57489

theorem kids_meals_sold (kids_meals : ℕ) (adult_meals : ℕ) : 
  (kids_meals : ℚ) / (adult_meals : ℚ) = 2 / 1 →
  kids_meals + adult_meals = 12 →
  kids_meals = 8 := by
sorry

end NUMINAMATH_CALUDE_kids_meals_sold_l574_57489


namespace NUMINAMATH_CALUDE_not_all_even_numbers_multiple_of_eight_l574_57463

theorem not_all_even_numbers_multiple_of_eight : ¬ (∀ n : ℕ, 2 ∣ n → 8 ∣ n) := by
  sorry

#check not_all_even_numbers_multiple_of_eight

end NUMINAMATH_CALUDE_not_all_even_numbers_multiple_of_eight_l574_57463


namespace NUMINAMATH_CALUDE_not_periodic_x_plus_cos_x_l574_57455

theorem not_periodic_x_plus_cos_x : ¬∃ (T : ℝ), T ≠ 0 ∧ ∀ (x : ℝ), x + T + Real.cos (x + T) = x + Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_not_periodic_x_plus_cos_x_l574_57455


namespace NUMINAMATH_CALUDE_wavelength_in_scientific_notation_l574_57453

/-- Converts nanometers to meters -/
def nm_to_m (nm : ℝ) : ℝ := nm * 0.000000001

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem wavelength_in_scientific_notation :
  let wavelength_nm : ℝ := 688
  let wavelength_m : ℝ := nm_to_m wavelength_nm
  let scientific : ScientificNotation := to_scientific_notation wavelength_m
  scientific.coefficient = 6.88 ∧ scientific.exponent = -7 :=
sorry

end NUMINAMATH_CALUDE_wavelength_in_scientific_notation_l574_57453


namespace NUMINAMATH_CALUDE_kaleb_initial_savings_l574_57487

/-- The amount of money Kaleb had initially saved up. -/
def initial_savings : ℕ := sorry

/-- The cost of each toy. -/
def toy_cost : ℕ := 6

/-- The number of toys Kaleb can buy after receiving his allowance. -/
def num_toys : ℕ := 6

/-- The amount of allowance Kaleb received. -/
def allowance : ℕ := 15

/-- Theorem stating that Kaleb's initial savings were $21. -/
theorem kaleb_initial_savings :
  initial_savings = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_kaleb_initial_savings_l574_57487


namespace NUMINAMATH_CALUDE_probability_distinct_numbers_value_l574_57471

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The probability of rolling six distinct numbers with six eight-sided dice -/
def probability_distinct_numbers : ℚ :=
  (num_sides.factorial / (num_sides - num_dice).factorial) / num_sides ^ num_dice

theorem probability_distinct_numbers_value :
  probability_distinct_numbers = 315 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_probability_distinct_numbers_value_l574_57471


namespace NUMINAMATH_CALUDE_simplify_fraction_l574_57446

theorem simplify_fraction : 18 * (8 / 15) * (3 / 4) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l574_57446


namespace NUMINAMATH_CALUDE_harriet_round_trip_l574_57434

/-- Harriet's round trip between A-ville and B-town -/
theorem harriet_round_trip 
  (speed_to_b : ℝ) 
  (speed_from_b : ℝ) 
  (time_to_b_minutes : ℝ) 
  (h1 : speed_to_b = 110) 
  (h2 : speed_from_b = 140) 
  (h3 : time_to_b_minutes = 168) : 
  let time_to_b := time_to_b_minutes / 60
  let distance := speed_to_b * time_to_b
  let time_from_b := distance / speed_from_b
  time_to_b + time_from_b = 5 := by
  sorry

end NUMINAMATH_CALUDE_harriet_round_trip_l574_57434


namespace NUMINAMATH_CALUDE_triangle_count_is_83_l574_57411

/-- Represents a rectangle divided into triangles -/
structure DividedRectangle where
  width : ℕ
  height : ℕ
  horizontal_divisions : ℕ
  vertical_divisions : ℕ

/-- Counts the number of triangles in the divided rectangle -/
def count_triangles (r : DividedRectangle) : ℕ :=
  sorry

/-- The specific rectangle from the problem -/
def problem_rectangle : DividedRectangle :=
  { width := 4
  , height := 5
  , horizontal_divisions := 4
  , vertical_divisions := 5 }

theorem triangle_count_is_83 : 
  count_triangles problem_rectangle = 83 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_is_83_l574_57411


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l574_57450

/-- Given a right pyramid with a square base, proves that the side length of the base is 10 meters 
    when the area of one lateral face is 120 square meters and the slant height is 24 meters. -/
theorem pyramid_base_side_length : 
  ∀ (base_side_length slant_height lateral_face_area : ℝ),
  slant_height = 24 →
  lateral_face_area = 120 →
  lateral_face_area = (1/2) * base_side_length * slant_height →
  base_side_length = 10 := by
sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l574_57450


namespace NUMINAMATH_CALUDE_parallel_perpendicular_lines_l574_57470

/-- Given a point P and a line l, prove the equations of parallel and perpendicular lines through P -/
theorem parallel_perpendicular_lines
  (P : ℝ × ℝ)  -- Point P
  (l : ℝ → ℝ → Prop)  -- Line l
  (hl : ∀ x y, l x y ↔ 3 * x - 2 * y - 7 = 0)  -- Equation of line l
  (hP : P = (-4, 2))  -- Coordinates of point P
  : 
  -- 1. Equation of parallel line through P
  (∀ x y, (3 * x - 2 * y + 16 = 0) ↔ 
    (∃ k, k ≠ 0 ∧ ∀ a b, l a b → (3 * x - 2 * y = 3 * a - 2 * b + k))) ∧
    (3 * P.1 - 2 * P.2 + 16 = 0) ∧

  -- 2. Equation of perpendicular line through P
  (∀ x y, (2 * x + 3 * y + 2 = 0) ↔ 
    (∀ a b, l a b → (3 * (x - a) + 2 * (y - b) = 0))) ∧
    (2 * P.1 + 3 * P.2 + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_lines_l574_57470


namespace NUMINAMATH_CALUDE_robin_camera_pictures_l574_57476

/-- The number of pictures Robin uploaded from her camera -/
def camera_pictures (phone_pictures total_albums pictures_per_album : ℕ) : ℕ :=
  total_albums * pictures_per_album - phone_pictures

/-- Proof that Robin uploaded 5 pictures from her camera -/
theorem robin_camera_pictures :
  camera_pictures 35 5 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_robin_camera_pictures_l574_57476


namespace NUMINAMATH_CALUDE_noras_age_l574_57472

/-- Represents a person's age --/
structure Person where
  age : ℕ

/-- Proves that Nora's current age is 10 years old --/
theorem noras_age (terry nora : Person) : 
  (terry.age + 10 = 4 * nora.age) → 
  (terry.age = 30) → 
  (nora.age = 10) := by
sorry

end NUMINAMATH_CALUDE_noras_age_l574_57472


namespace NUMINAMATH_CALUDE_second_round_difference_l574_57496

/-- Bowling game results -/
structure BowlingGame where
  patrick_first : ℕ
  richard_first : ℕ
  patrick_second : ℕ
  richard_second : ℕ

/-- Conditions of the bowling game -/
def bowling_conditions (game : BowlingGame) : Prop :=
  game.patrick_first = 70 ∧
  game.richard_first = game.patrick_first + 15 ∧
  game.patrick_second = 2 * game.richard_first ∧
  game.richard_second < game.patrick_second ∧
  game.richard_first + game.richard_second = game.patrick_first + game.patrick_second + 12

/-- Theorem: The difference between Patrick's and Richard's knocked down pins in the second round is 3 -/
theorem second_round_difference (game : BowlingGame) 
  (h : bowling_conditions game) : 
  game.patrick_second - game.richard_second = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_round_difference_l574_57496


namespace NUMINAMATH_CALUDE_last_digit_of_max_value_l574_57431

/-- Operation that combines two numbers a and b into a * b + 1 -/
def combine (a b : ℕ) : ℕ := a * b + 1

/-- Type representing the state of the blackboard -/
def Blackboard := List ℕ

/-- Function to perform one step of the operation -/
def performStep (board : Blackboard) : Blackboard :=
  match board with
  | a :: b :: rest => combine a b :: rest
  | _ => board

/-- Function to perform n steps of the operation -/
def performNSteps (n : ℕ) (board : Blackboard) : Blackboard :=
  match n with
  | 0 => board
  | n + 1 => performNSteps n (performStep board)

/-- The maximum possible value after 127 operations -/
def maxFinalValue : ℕ :=
  let initialBoard : Blackboard := List.replicate 128 1
  (performNSteps 127 initialBoard).head!

theorem last_digit_of_max_value :
  maxFinalValue % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_max_value_l574_57431


namespace NUMINAMATH_CALUDE_not_divisible_by_2n_plus_65_l574_57423

theorem not_divisible_by_2n_plus_65 (n : ℕ+) : ¬(2^n.val + 65 ∣ 5^n.val - 3^n.val) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_2n_plus_65_l574_57423


namespace NUMINAMATH_CALUDE_g_at_negative_two_l574_57475

/-- The function g(x) = 2x^2 + 3x + 1 -/
def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

/-- Theorem: g(-2) = 3 -/
theorem g_at_negative_two : g (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_at_negative_two_l574_57475


namespace NUMINAMATH_CALUDE_shopping_trip_tax_percentage_l574_57425

/-- Represents the spending distribution and tax rates for a shopping trip -/
structure ShoppingTrip where
  clothing_percent : ℝ
  food_percent : ℝ
  other_percent : ℝ
  clothing_tax_rate : ℝ
  food_tax_rate : ℝ
  other_tax_rate : ℝ

/-- Calculates the total tax percentage for a given shopping trip -/
def totalTaxPercentage (trip : ShoppingTrip) : ℝ :=
  trip.clothing_percent * trip.clothing_tax_rate +
  trip.food_percent * trip.food_tax_rate +
  trip.other_percent * trip.other_tax_rate

/-- Theorem stating that for the given shopping trip, the total tax is 5% of the total amount spent excluding taxes -/
theorem shopping_trip_tax_percentage :
  let trip : ShoppingTrip := {
    clothing_percent := 0.5,
    food_percent := 0.2,
    other_percent := 0.3,
    clothing_tax_rate := 0.04,
    food_tax_rate := 0,
    other_tax_rate := 0.1
  }
  totalTaxPercentage trip = 0.05 := by
  sorry


end NUMINAMATH_CALUDE_shopping_trip_tax_percentage_l574_57425


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l574_57408

theorem sphere_volume_ratio (r R : ℝ) (h : r > 0) (H : R > 0) :
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 4 / 9 →
  ((4 / 3) * Real.pi * r^3) / ((4 / 3) * Real.pi * R^3) = 8 / 27 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l574_57408


namespace NUMINAMATH_CALUDE_catch_up_time_tom_catches_jerry_l574_57422

/-- Represents the figure-eight track --/
structure Track :=
  (small_loop : ℝ)
  (large_loop : ℝ)
  (large_loop_eq : large_loop = (4 / 3) * small_loop)

/-- Represents the runners Tom and Jerry --/
structure Runner :=
  (speed : ℝ)

/-- The problem setup --/
def problem (t : Track) (tom jerry : Runner) : Prop :=
  tom.speed > jerry.speed ∧
  tom.speed * 20 = t.large_loop ∧
  jerry.speed * 20 = t.small_loop ∧
  tom.speed * 15 = t.small_loop

/-- The theorem to be proved --/
theorem catch_up_time (t : Track) (tom jerry : Runner) 
  (h : problem t tom jerry) : ℝ := by
  sorry

/-- The main theorem stating that Tom catches up with Jerry in 80 minutes --/
theorem tom_catches_jerry (t : Track) (tom jerry : Runner) 
  (h : problem t tom jerry) : catch_up_time t tom jerry h = 80 := by
  sorry

end NUMINAMATH_CALUDE_catch_up_time_tom_catches_jerry_l574_57422


namespace NUMINAMATH_CALUDE_intersection_with_complement_l574_57442

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 3, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l574_57442


namespace NUMINAMATH_CALUDE_surface_area_increase_after_removal_l574_57466

/-- Represents a rectangular solid with length, width, and height -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the change in surface area after removal of a smaller prism -/
def surfaceAreaChange (larger : RectangularSolid) (smaller : RectangularSolid) : ℝ :=
  (smaller.length * smaller.width + smaller.length * smaller.height + smaller.width * smaller.height) * 2 -
  smaller.length * smaller.width

theorem surface_area_increase_after_removal :
  let larger := RectangularSolid.mk 5 3 2
  let smaller := RectangularSolid.mk 2 1 1
  surfaceAreaChange larger smaller = 4 := by
  sorry


end NUMINAMATH_CALUDE_surface_area_increase_after_removal_l574_57466


namespace NUMINAMATH_CALUDE_paddington_washington_goats_difference_l574_57486

theorem paddington_washington_goats_difference 
  (washington_goats : ℕ) 
  (total_goats : ℕ) 
  (h1 : washington_goats = 140)
  (h2 : total_goats = 320)
  (h3 : washington_goats < total_goats - washington_goats) : 
  total_goats - washington_goats - washington_goats = 40 := by
  sorry

end NUMINAMATH_CALUDE_paddington_washington_goats_difference_l574_57486


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l574_57445

theorem unique_two_digit_integer (s : ℕ) : 
  (∃! s, 10 ≤ s ∧ s < 100 ∧ 13 * s % 100 = 52) :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l574_57445


namespace NUMINAMATH_CALUDE_range_of_a_l574_57493

/-- The range of values for a given the conditions -/
theorem range_of_a (p q : ℝ → Prop) (a : ℝ) : 
  (∀ x, p x → q x) →  -- p is sufficient for q
  (∃ x, q x ∧ ¬(p x)) →  -- p is not necessary for q
  (∀ x, p x ↔ (x^2 - 2*x - 3 < 0)) →  -- definition of p
  (∀ x, q x ↔ (x > a)) →  -- definition of q
  a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l574_57493


namespace NUMINAMATH_CALUDE_solve_p_q_system_l574_57451

theorem solve_p_q_system (p q : ℝ) (hp : p > 1) (hq : q > 1)
  (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_p_q_system_l574_57451


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l574_57440

theorem regular_octagon_interior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let sum_of_interior_angles : ℝ := (n - 2) * 180
  let interior_angle : ℝ := sum_of_interior_angles / n
  135

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l574_57440


namespace NUMINAMATH_CALUDE_problem_bottle_height_l574_57401

/-- Represents a bottle constructed from two cylinders -/
structure Bottle where
  small_radius : ℝ
  large_radius : ℝ
  water_height_right_side_up : ℝ
  water_height_upside_down : ℝ

/-- Calculates the total height of the bottle -/
def total_height (b : Bottle) : ℝ :=
  sorry

/-- The specific bottle from the problem -/
def problem_bottle : Bottle :=
  { small_radius := 1
  , large_radius := 3
  , water_height_right_side_up := 20
  , water_height_upside_down := 28 }

/-- Theorem stating that the total height of the problem bottle is 29 cm -/
theorem problem_bottle_height :
  total_height problem_bottle = 29 := by sorry

end NUMINAMATH_CALUDE_problem_bottle_height_l574_57401


namespace NUMINAMATH_CALUDE_triangle_theorem_l574_57497

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c^2 = t.a^2 + t.b^2 - t.a * t.b ∧
  t.b = 2 ∧
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 ∧ t.a = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l574_57497


namespace NUMINAMATH_CALUDE_interior_angle_sum_l574_57448

theorem interior_angle_sum (n : ℕ) : 
  (180 * (n - 2) = 2160) → (180 * ((n + 3) - 2) = 2700) := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_l574_57448


namespace NUMINAMATH_CALUDE_inequality_proof_l574_57473

theorem inequality_proof (a b : ℝ) : 
  |a + b| / (1 + |a + b|) ≤ |a| / (1 + |a|) + |b| / (1 + |b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l574_57473


namespace NUMINAMATH_CALUDE_beth_double_age_in_8_years_l574_57441

/-- The number of years until Beth is twice her sister's age -/
def years_until_double_age (beth_age : ℕ) (sister_age : ℕ) : ℕ :=
  beth_age + sister_age

theorem beth_double_age_in_8_years (beth_age : ℕ) (sister_age : ℕ) 
  (h1 : beth_age = 18) (h2 : sister_age = 5) :
  years_until_double_age beth_age sister_age = 8 :=
sorry

#check beth_double_age_in_8_years

end NUMINAMATH_CALUDE_beth_double_age_in_8_years_l574_57441


namespace NUMINAMATH_CALUDE_mass_percentage_Al_approx_l574_57481

-- Define atomic masses
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_S : ℝ := 32.06
def atomic_mass_Ca : ℝ := 40.08
def atomic_mass_C : ℝ := 12.01
def atomic_mass_O : ℝ := 16.00
def atomic_mass_K : ℝ := 39.10
def atomic_mass_Cl : ℝ := 35.45

-- Define molar masses of compounds
def molar_mass_Al2S3 : ℝ := 2 * atomic_mass_Al + 3 * atomic_mass_S
def molar_mass_CaCO3 : ℝ := atomic_mass_Ca + atomic_mass_C + 3 * atomic_mass_O
def molar_mass_KCl : ℝ := atomic_mass_K + atomic_mass_Cl

-- Define moles of compounds in the mixture
def moles_Al2S3 : ℝ := 2
def moles_CaCO3 : ℝ := 3
def moles_KCl : ℝ := 5

-- Define total mass of the mixture
def total_mass : ℝ := moles_Al2S3 * molar_mass_Al2S3 + moles_CaCO3 * molar_mass_CaCO3 + moles_KCl * molar_mass_KCl

-- Define mass of Al in the mixture
def mass_Al : ℝ := 2 * moles_Al2S3 * atomic_mass_Al

-- Theorem: The mass percentage of Al in the mixture is approximately 11.09%
theorem mass_percentage_Al_approx (ε : ℝ) (h : ε > 0) : 
  ∃ δ : ℝ, δ > 0 ∧ |mass_Al / total_mass * 100 - 11.09| < δ :=
sorry

end NUMINAMATH_CALUDE_mass_percentage_Al_approx_l574_57481


namespace NUMINAMATH_CALUDE_number_of_blue_balls_l574_57469

/-- Given a set of balls with red, blue, and green colors, prove the number of blue balls. -/
theorem number_of_blue_balls
  (total : ℕ)
  (green : ℕ)
  (h1 : total = 40)
  (h2 : green = 7)
  (h3 : ∃ (blue : ℕ), total = green + blue + 2 * blue) :
  ∃ (blue : ℕ), blue = 11 ∧ total = green + blue + 2 * blue :=
sorry

end NUMINAMATH_CALUDE_number_of_blue_balls_l574_57469


namespace NUMINAMATH_CALUDE_exists_table_with_square_corner_sums_l574_57483

/-- Represents a 100 x 100 table of natural numbers -/
def Table := Fin 100 → Fin 100 → ℕ

/-- Checks if all numbers in the same row or column are different -/
def all_different (t : Table) : Prop :=
  ∀ i j i' j', i ≠ i' ∨ j ≠ j' → t i j ≠ t i' j'

/-- Checks if the sum of numbers in angle cells of a square submatrix is a square number -/
def corner_sum_is_square (t : Table) : Prop :=
  ∀ i j n, ∃ k : ℕ, 
    t i j + t i (j + n) + t (i + n) j + t (i + n) (j + n) = k * k

/-- The main theorem stating the existence of a table satisfying all conditions -/
theorem exists_table_with_square_corner_sums : 
  ∃ t : Table, all_different t ∧ corner_sum_is_square t := by
  sorry

end NUMINAMATH_CALUDE_exists_table_with_square_corner_sums_l574_57483


namespace NUMINAMATH_CALUDE_f_composition_theorem_l574_57457

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the condition that |f(x)| ≤ 1/2 for all x in [2, 4]
def f_condition (p q : ℝ) : Prop :=
  ∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → |f p q x| ≤ 1/2

-- Define the n-fold composition of f
def f_compose (p q : ℝ) : ℕ → ℝ → ℝ
  | 0, x => x
  | n+1, x => f p q (f_compose p q n x)

-- The theorem to prove
theorem f_composition_theorem (p q : ℝ) (h : f_condition p q) :
  f_compose p q 2017 ((5 - Real.sqrt 11) / 2) = (5 + Real.sqrt 11) / 2 :=
sorry

end NUMINAMATH_CALUDE_f_composition_theorem_l574_57457


namespace NUMINAMATH_CALUDE_license_plate_difference_l574_57492

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible Florida license plates -/
def florida_plates : ℕ := num_letters^6 * num_digits^2

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := num_letters^3 * num_digits^4

/-- The difference between Florida and Texas license plate possibilities -/
def plate_difference : ℕ := florida_plates - texas_plates

theorem license_plate_difference :
  plate_difference = 54293545536 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l574_57492


namespace NUMINAMATH_CALUDE_sqrt_real_range_l574_57418

theorem sqrt_real_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = 4 + 2 * x) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_real_range_l574_57418


namespace NUMINAMATH_CALUDE_initial_amount_is_100_l574_57456

/-- The amount of money Jasmine spent on fruits -/
def spent_on_fruits : ℝ := 15

/-- The amount of money Jasmine had left to spend -/
def money_left : ℝ := 85

/-- The initial amount of money Jasmine's mom gave her -/
def initial_amount : ℝ := spent_on_fruits + money_left

/-- Theorem stating that the initial amount of money Jasmine's mom gave her is $100.00 -/
theorem initial_amount_is_100 : initial_amount = 100 := by sorry

end NUMINAMATH_CALUDE_initial_amount_is_100_l574_57456


namespace NUMINAMATH_CALUDE_work_multiple_proof_l574_57499

/-- Represents the time taken to complete a job given the number of workers and the fraction of the job to be completed -/
def time_to_complete (num_workers : ℕ) (job_fraction : ℚ) (base_time : ℕ) : ℚ :=
  (job_fraction * base_time) / num_workers

theorem work_multiple_proof (base_workers : ℕ) (h : base_workers > 0) :
  time_to_complete base_workers 1 12 = 12 →
  time_to_complete (2 * base_workers) (1/2) 12 = 3 := by
sorry

end NUMINAMATH_CALUDE_work_multiple_proof_l574_57499


namespace NUMINAMATH_CALUDE_arithmetic_progression_reciprocals_squares_l574_57438

theorem arithmetic_progression_reciprocals_squares (a b c : ℝ) :
  (2 / (c + a) = 1 / (b + c) + 1 / (b + a)) →
  (a^2 + c^2 = 2 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_reciprocals_squares_l574_57438


namespace NUMINAMATH_CALUDE_initial_boys_count_l574_57435

/-- Given a school with an initial number of girls and boys, and after some additions,
    prove that the initial number of boys was 214. -/
theorem initial_boys_count (initial_girls : ℕ) (initial_boys : ℕ) 
  (added_girls : ℕ) (added_boys : ℕ) (final_boys : ℕ) : 
  initial_girls = 135 → 
  added_girls = 496 → 
  added_boys = 910 → 
  final_boys = 1124 → 
  initial_boys + added_boys = final_boys → 
  initial_boys = 214 := by
sorry

end NUMINAMATH_CALUDE_initial_boys_count_l574_57435


namespace NUMINAMATH_CALUDE_percentage_calculation_l574_57468

theorem percentage_calculation : 
  (0.2 * 120 + 0.25 * 250 + 0.15 * 80) - 0.1 * 600 = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l574_57468


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l574_57444

theorem triangle_max_perimeter (a b y : ℕ) (ha : a = 7) (hb : b = 9) :
  (∃ (y : ℕ), a + b + y = (a + b + y).max (a + b + (a + b - 1))) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l574_57444


namespace NUMINAMATH_CALUDE_diamond_calculation_l574_57488

def diamond (A B : ℚ) : ℚ := (A - B) / 5

theorem diamond_calculation : (diamond (diamond 7 15) 2) = -18/25 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l574_57488


namespace NUMINAMATH_CALUDE_sequence_2023rd_term_l574_57436

def sequence_term (n : ℕ) : ℚ := (-1)^n * (n : ℚ) / (n + 1)

theorem sequence_2023rd_term : sequence_term 2023 = -2023 / 2024 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2023rd_term_l574_57436


namespace NUMINAMATH_CALUDE_lcm_520_693_l574_57478

theorem lcm_520_693 : Nat.lcm 520 693 = 360360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_520_693_l574_57478


namespace NUMINAMATH_CALUDE_birds_in_tree_l574_57414

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29)
  (h2 : final_birds = 42) :
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l574_57414


namespace NUMINAMATH_CALUDE_chicken_feed_requirement_l574_57416

/-- Represents the problem of calculating chicken feed requirements --/
theorem chicken_feed_requirement 
  (chicken_price : ℝ) 
  (feed_price : ℝ) 
  (feed_weight : ℝ) 
  (num_chickens : ℕ) 
  (profit : ℝ) 
  (h1 : chicken_price = 1.5)
  (h2 : feed_price = 2)
  (h3 : feed_weight = 20)
  (h4 : num_chickens = 50)
  (h5 : profit = 65) :
  (num_chickens * chicken_price - profit) / feed_price * feed_weight / num_chickens = 2 := by
  sorry

#check chicken_feed_requirement

end NUMINAMATH_CALUDE_chicken_feed_requirement_l574_57416


namespace NUMINAMATH_CALUDE_reflection_in_first_quadrant_l574_57490

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the reflection across y-axis
def reflect_y (p : Point) : Point :=
  (-p.1, p.2)

-- Define the first quadrant
def in_first_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 > 0

-- Theorem statement
theorem reflection_in_first_quadrant :
  let P : Point := (-3, 1)
  in_first_quadrant (reflect_y P) := by sorry

end NUMINAMATH_CALUDE_reflection_in_first_quadrant_l574_57490


namespace NUMINAMATH_CALUDE_fourteen_segments_l574_57403

/-- A right triangle with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- The number of distinct integer lengths of line segments from E to DF -/
def numIntegerSegments (t : RightTriangle) : ℕ := sorry

/-- Our specific right triangle -/
def triangle : RightTriangle := { de := 24, ef := 25 }

/-- The theorem to prove -/
theorem fourteen_segments : numIntegerSegments triangle = 14 := by sorry

end NUMINAMATH_CALUDE_fourteen_segments_l574_57403


namespace NUMINAMATH_CALUDE_solution_set_implies_a_minus_b_l574_57407

/-- The solution set of a quadratic inequality -/
def SolutionSet (a b : ℝ) : Set ℝ :=
  {x | a * x^2 + b * x + 2 > 0}

/-- The theorem stating the relationship between the solution set and the value of a - b -/
theorem solution_set_implies_a_minus_b (a b : ℝ) :
  SolutionSet a b = {x | -1/2 < x ∧ x < 1/3} → a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_minus_b_l574_57407


namespace NUMINAMATH_CALUDE_coin_division_l574_57461

theorem coin_division (n : ℕ) : 
  (n > 0) →
  (n % 6 = 4) → 
  (n % 5 = 3) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 6 ≠ 4 ∨ m % 5 ≠ 3)) →
  (n % 7 = 0) := by
sorry

end NUMINAMATH_CALUDE_coin_division_l574_57461


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l574_57479

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (4, -1)

-- Define the radius of the new circle
def new_radius : ℝ := 1

-- Define the possible equations of the new circle
def new_circle_1 (x y : ℝ) : Prop := (x - 5)^2 + (y + 1)^2 = 1
def new_circle_2 (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 1

-- Theorem statement
theorem tangent_circle_equation : 
  ∃ (x y : ℝ), (circle_C x y ∧ (x, y) = tangent_point) → 
  (new_circle_1 x y ∨ new_circle_2 x y) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l574_57479


namespace NUMINAMATH_CALUDE_number_of_hens_l574_57452

/-- Given the following conditions:
  - The total cost of pigs and hens is 1200 Rs
  - There are 3 pigs
  - The average price of a hen is 30 Rs
  - The average price of a pig is 300 Rs
  Prove that the number of hens bought is 10. -/
theorem number_of_hens (total_cost : ℕ) (num_pigs : ℕ) (hen_price : ℕ) (pig_price : ℕ) 
  (h1 : total_cost = 1200)
  (h2 : num_pigs = 3)
  (h3 : hen_price = 30)
  (h4 : pig_price = 300) :
  ∃ (num_hens : ℕ), num_hens * hen_price + num_pigs * pig_price = total_cost ∧ num_hens = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_hens_l574_57452


namespace NUMINAMATH_CALUDE_tile_difference_is_88_l574_57427

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The difference in the number of tiles between the 7th and 5th squares -/
def tile_difference : ℕ := tiles_in_square 7 - tiles_in_square 5

theorem tile_difference_is_88 : tile_difference = 88 := by
  sorry

end NUMINAMATH_CALUDE_tile_difference_is_88_l574_57427


namespace NUMINAMATH_CALUDE_soccer_team_activities_l574_57405

/-- The number of activities required for a soccer team practice --/
def total_activities (total_players : ℕ) (goalies : ℕ) : ℕ :=
  let non_goalie_activities := goalies * (total_players - 1)
  2 * non_goalie_activities

theorem soccer_team_activities :
  total_activities 25 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_activities_l574_57405


namespace NUMINAMATH_CALUDE_orange_juice_mix_l574_57491

/-- Given the conditions for preparing orange juice, prove that 3 cans of water are needed per can of concentrate. -/
theorem orange_juice_mix (servings : ℕ) (serving_size : ℚ) (concentrate_cans : ℕ) (concentrate_size : ℚ) 
  (h1 : servings = 200)
  (h2 : serving_size = 6)
  (h3 : concentrate_cans = 60)
  (h4 : concentrate_size = 5) : 
  (servings * serving_size - concentrate_cans * concentrate_size) / (concentrate_cans * concentrate_size) = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_mix_l574_57491


namespace NUMINAMATH_CALUDE_v_2007_equals_1_l574_57419

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 2
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 5
| (n + 1) => g (v n)

-- Theorem statement
theorem v_2007_equals_1 : v 2007 = 1 := by
  sorry

end NUMINAMATH_CALUDE_v_2007_equals_1_l574_57419


namespace NUMINAMATH_CALUDE_no_positive_integer_perfect_square_l574_57449

theorem no_positive_integer_perfect_square : 
  ∀ n : ℕ+, ¬∃ y : ℤ, (n : ℤ)^2 - 21*(n : ℤ) + 110 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_perfect_square_l574_57449


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l574_57462

/-- Given a rectangular prism with base edge length 3 cm and lateral face diagonal 3√5 cm,
    prove that its volume is 54 cm³ -/
theorem rectangular_prism_volume (base_edge : ℝ) (lateral_diagonal : ℝ) (volume : ℝ) :
  base_edge = 3 →
  lateral_diagonal = 3 * Real.sqrt 5 →
  volume = base_edge * base_edge * Real.sqrt (lateral_diagonal^2 - base_edge^2) →
  volume = 54 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l574_57462


namespace NUMINAMATH_CALUDE_cube_face_sum_l574_57477

theorem cube_face_sum (a b c d e f : ℕ+) :
  (a * b * c + a * e * c + a * b * f + a * e * f +
   d * b * c + d * e * c + d * b * f + d * e * f) = 1287 →
  (a + d) + (b + e) + (c + f) = 33 := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l574_57477


namespace NUMINAMATH_CALUDE_january_salary_l574_57474

/-- Represents the salary structure for a person over 5 months -/
structure SalaryStructure where
  jan : ℝ
  feb : ℝ
  mar : ℝ
  apr : ℝ
  may : ℝ
  bonus : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem january_salary (s : SalaryStructure) 
  (avg_jan_apr : (s.jan + s.feb + s.mar + s.apr) / 4 = 8000)
  (avg_feb_may : (s.feb + s.mar + s.apr + s.may) / 4 = 8400)
  (may_salary : s.may = 6500)
  (apr_raise : s.apr = 1.05 * s.feb)
  (mar_bonus : s.mar = s.feb + s.bonus) :
  s.jan = 4900 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l574_57474


namespace NUMINAMATH_CALUDE_max_sum_of_product_2401_l574_57406

theorem max_sum_of_product_2401 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2401 →
  A + B + C ≤ 351 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_product_2401_l574_57406


namespace NUMINAMATH_CALUDE_inequalities_system_k_range_l574_57494

theorem inequalities_system_k_range :
  ∀ k : ℚ,
  (∀ x : ℤ, x^2 - x - 2 > 0 ∧ 2*x^2 + (5 + 2*k)*x + 5 < 0 ↔ x = -2) →
  3/4 < k ∧ k ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_inequalities_system_k_range_l574_57494


namespace NUMINAMATH_CALUDE_no_equal_distribution_l574_57415

/-- Represents the number of glasses --/
def num_glasses : ℕ := 2018

/-- Represents the total amount of champagne --/
def total_champagne : ℕ := 2019

/-- Represents a distribution of champagne among glasses --/
def Distribution := Fin num_glasses → ℚ

/-- Checks if a distribution is valid (sums to total_champagne) --/
def is_valid_distribution (d : Distribution) : Prop :=
  (Finset.sum Finset.univ (λ i => d i)) = total_champagne

/-- Represents the equalization operation on two glasses --/
def equalize (d : Distribution) (i j : Fin num_glasses) : Distribution :=
  λ k => if k = i ∨ k = j then (d i + d j) / 2 else d k

/-- Represents the property of all glasses having equal integer amount --/
def all_equal_integer (d : Distribution) : Prop :=
  ∃ n : ℕ, ∀ i : Fin num_glasses, d i = n

/-- The main theorem stating that no initial distribution can result in
    all glasses having equal integer amount after repeated equalization --/
theorem no_equal_distribution :
  ¬∃ (d : Distribution), is_valid_distribution d ∧
    ∃ (seq : ℕ → Fin num_glasses × Fin num_glasses),
      ∃ (n : ℕ), all_equal_integer (Nat.iterate (λ d' => equalize d' (seq n).1 (seq n).2) n d) := by
  sorry

end NUMINAMATH_CALUDE_no_equal_distribution_l574_57415


namespace NUMINAMATH_CALUDE_journey_distance_l574_57495

/-- Prove that given a journey with specified conditions, the total distance traveled is 270 km. -/
theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 15 →
  speed1 = 45 →
  speed2 = 30 →
  (total_time * speed1 * speed2) / (speed1 + speed2) = 270 := by
  sorry

#check journey_distance

end NUMINAMATH_CALUDE_journey_distance_l574_57495


namespace NUMINAMATH_CALUDE_diameter_line_equation_l574_57402

/-- Given a circle and a point inside it, prove the equation of the line containing the diameter through the point. -/
theorem diameter_line_equation (x y : ℝ) :
  (x - 1)^2 + y^2 = 4 →  -- Circle equation
  (2 : ℝ) - 1 < 2 →      -- Point (2,1) is inside the circle
  ∃ (m b : ℝ), x - y - 1 = 0 ∧ 
    (∀ (x' y' : ℝ), (x' - 1)^2 + y'^2 = 4 → (y' - 1) = m * (x' - 2) + b) :=
by sorry

end NUMINAMATH_CALUDE_diameter_line_equation_l574_57402


namespace NUMINAMATH_CALUDE_glee_club_female_members_l574_57426

theorem glee_club_female_members 
  (total_members : ℕ) 
  (female_ratio : ℕ) 
  (male_ratio : ℕ) 
  (h1 : total_members = 18)
  (h2 : female_ratio = 2)
  (h3 : male_ratio = 1)
  (h4 : female_ratio * male_members + male_ratio * male_members = total_members)
  : female_ratio * male_members = 12 :=
by
  sorry

#check glee_club_female_members

end NUMINAMATH_CALUDE_glee_club_female_members_l574_57426


namespace NUMINAMATH_CALUDE_registration_count_l574_57409

/-- The number of ways two students can register for universities --/
def registration_possibilities : ℕ :=
  let n_universities := 3
  let n_students := 2
  let choose_one := n_universities
  let choose_two := n_universities.choose 2
  (choose_one ^ n_students) + 
  (choose_two ^ n_students) + 
  (2 * choose_one * choose_two)

theorem registration_count : registration_possibilities = 36 := by
  sorry

end NUMINAMATH_CALUDE_registration_count_l574_57409


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_two_l574_57410

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_two :
  lg 2 * lg 2 + lg 2 * lg 5 + lg 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_two_l574_57410


namespace NUMINAMATH_CALUDE_train_speed_l574_57458

/-- Proves that a train of given length passing a point in a given time has a specific speed in kmph -/
theorem train_speed (train_length : Real) (time_to_pass : Real) (speed_kmph : Real) : 
  train_length = 20 →
  time_to_pass = 1.9998400127989762 →
  speed_kmph = (train_length / time_to_pass) * 3.6 →
  speed_kmph = 36.00287986320432 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l574_57458


namespace NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l574_57464

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 3 = 10) :
  a 2 = 5 :=
sorry

end NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l574_57464


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l574_57430

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 7
def quadruplets_in_lineup : ℕ := 3

theorem volleyball_team_selection :
  (Nat.choose quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 1980 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l574_57430


namespace NUMINAMATH_CALUDE_base_conversion_equality_l574_57439

/-- Given that 32₄ = 120ᵦ, prove that the unique positive integer b satisfying this equation is 2. -/
theorem base_conversion_equality (b : ℕ) : b > 0 ∧ (3 * 4 + 2) = (1 * b^2 + 2 * b + 0) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l574_57439


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l574_57421

/-- 
Given an arithmetic progression with first term a₁ and common difference d,
if the product of the 3rd and 6th terms is 406,
and the 9th term divided by the 4th term gives a quotient of 2 with remainder 6,
then the first term is 4 and the common difference is 5.
-/
theorem arithmetic_progression_problem (a₁ d : ℚ) : 
  (a₁ + 2*d) * (a₁ + 5*d) = 406 →
  (a₁ + 8*d) = 2*(a₁ + 3*d) + 6 →
  a₁ = 4 ∧ d = 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_progression_problem_l574_57421


namespace NUMINAMATH_CALUDE_product_plus_twenty_l574_57432

theorem product_plus_twenty : ∃ n : ℕ, n = 5 * 7 ∧ n + 12 + 8 = 55 := by sorry

end NUMINAMATH_CALUDE_product_plus_twenty_l574_57432


namespace NUMINAMATH_CALUDE_exchange_calculation_l574_57454

/-- Exchange rate between Canadian and American dollars -/
def exchange_rate : ℚ := 120 / 80

/-- Amount of American dollars to be exchanged -/
def american_dollars : ℚ := 50

/-- Function to calculate Canadian dollars given American dollars -/
def exchange (usd : ℚ) : ℚ := usd * exchange_rate

theorem exchange_calculation :
  exchange american_dollars = 75 := by
  sorry

end NUMINAMATH_CALUDE_exchange_calculation_l574_57454


namespace NUMINAMATH_CALUDE_power_of_same_base_power_of_different_base_l574_57482

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem 1: Condition for representing a^n as (a^p)^q
theorem power_of_same_base (a n : ℕ) (h : n > 1) :
  (∃ p q : ℕ, p > 1 ∧ q > 1 ∧ a^n = (a^p)^q) ↔ ¬(isPrime n) :=
sorry

-- Theorem 2: Condition for representing a^n as b^m with a different base
theorem power_of_different_base (a n : ℕ) (h : n > 0) :
  (∃ b m : ℕ, b ≠ a ∧ m > 0 ∧ a^n = b^m) ↔
  (∃ k : ℕ, k > 0 ∧ ∃ b : ℕ, b ≠ a ∧ a^n = (b^k)^(n/k)) :=
sorry

end NUMINAMATH_CALUDE_power_of_same_base_power_of_different_base_l574_57482


namespace NUMINAMATH_CALUDE_min_triangle_area_l574_57498

/-- The minimum non-zero area of a triangle with vertices (0,0), (50,20), and (p,q),
    where p and q are integers. -/
theorem min_triangle_area :
  ∀ p q : ℤ,
  let area := (1/2 : ℝ) * |20 * p - 50 * q|
  ∃ p' q' : ℤ,
    (area > 0 → area ≥ 15) ∧
    (∃ a : ℝ, a > 0 ∧ a < 15 → ¬∃ p'' q'' : ℤ, (1/2 : ℝ) * |20 * p'' - 50 * q''| = a) :=
by sorry

end NUMINAMATH_CALUDE_min_triangle_area_l574_57498


namespace NUMINAMATH_CALUDE_complex_multiplication_l574_57428

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 - 2*i) = 2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l574_57428


namespace NUMINAMATH_CALUDE_sum_of_products_l574_57437

theorem sum_of_products (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 1) (h2 : a + b + c = 0) :
  a * b + a * c + b * c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l574_57437


namespace NUMINAMATH_CALUDE_marathon_time_proof_l574_57467

theorem marathon_time_proof (dean_time jake_time micah_time : ℝ) : 
  dean_time = 9 →
  micah_time = (2/3) * dean_time →
  jake_time = micah_time + (1/3) * micah_time →
  micah_time + dean_time + jake_time = 23 := by
sorry

end NUMINAMATH_CALUDE_marathon_time_proof_l574_57467


namespace NUMINAMATH_CALUDE_expression_factorization_l574_57420

theorem expression_factorization (x y z : ℤ) :
  x^2 - y^2 - z^2 + 2*y*z + x + y - z = (x + y - z) * (x - y + z + 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l574_57420


namespace NUMINAMATH_CALUDE_gumball_count_l574_57485

/-- Represents a gumball machine with red, green, and blue gumballs. -/
structure GumballMachine where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Creates a gumball machine with the given conditions. -/
def createMachine (redCount : ℕ) : GumballMachine :=
  let blueCount := redCount / 2
  let greenCount := blueCount * 4
  { red := redCount, blue := blueCount, green := greenCount }

/-- Calculates the total number of gumballs in the machine. -/
def totalGumballs (machine : GumballMachine) : ℕ :=
  machine.red + machine.blue + machine.green

/-- Theorem stating that a machine with 16 red gumballs has 56 gumballs in total. -/
theorem gumball_count : totalGumballs (createMachine 16) = 56 := by
  sorry

end NUMINAMATH_CALUDE_gumball_count_l574_57485


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l574_57480

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def is_geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, b (n + 1) = q * b n

/-- The theorem statement -/
theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  is_arithmetic_sequence a →
  is_geometric_sequence (λ n => a (2*n - 1) - (2*n - 1)) q →
  q = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l574_57480


namespace NUMINAMATH_CALUDE_julia_garden_area_l574_57447

/-- Represents a rectangular garden with given walking constraints -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  length_walk : length * 30 = 1500
  perimeter_walk : (length + width) * 2 * 12 = 1500

/-- The area of Julia's garden is 625 square meters -/
theorem julia_garden_area (garden : RectangularGarden) : garden.length * garden.width = 625 := by
  sorry

#check julia_garden_area

end NUMINAMATH_CALUDE_julia_garden_area_l574_57447


namespace NUMINAMATH_CALUDE_complex_equation_solution_l574_57417

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : i * z = 1) : z = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l574_57417


namespace NUMINAMATH_CALUDE_min_value_theorem_l574_57413

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 / a + 1 / b + 2 * Real.sqrt (a * b)) ≥ 4 ∧
  (1 / a + 1 / b + 2 * Real.sqrt (a * b) = 4 ↔ a = b) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l574_57413


namespace NUMINAMATH_CALUDE_probability_of_one_in_pascal_triangle_l574_57404

def pascalTriangleElements (n : ℕ) : ℕ := n * (n + 1) / 2

def onesInPascalTriangle (n : ℕ) : ℕ := 1 + 2 * (n - 1)

theorem probability_of_one_in_pascal_triangle : 
  (onesInPascalTriangle 20 : ℚ) / (pascalTriangleElements 20 : ℚ) = 39 / 210 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_in_pascal_triangle_l574_57404


namespace NUMINAMATH_CALUDE_freshman_class_size_l574_57460

theorem freshman_class_size :
  ∃! n : ℕ, n < 400 ∧ n % 26 = 17 ∧ n % 24 = 6 :=
by
  use 379
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l574_57460


namespace NUMINAMATH_CALUDE_rd_investment_exceeds_target_l574_57433

def initial_investment : ℝ := 1.3
def annual_increase : ℝ := 0.12
def target_investment : ℝ := 2.0
def start_year : ℕ := 2015
def target_year : ℕ := 2019

theorem rd_investment_exceeds_target :
  (initial_investment * (1 + annual_increase) ^ (target_year - start_year) > target_investment) ∧
  (∀ y : ℕ, y < target_year → initial_investment * (1 + annual_increase) ^ (y - start_year) ≤ target_investment) :=
by sorry

end NUMINAMATH_CALUDE_rd_investment_exceeds_target_l574_57433


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l574_57443

theorem arithmetic_mean_problem (original_list : List ℝ) 
  (a b c : ℝ) : 
  (original_list.length = 20) →
  (original_list.sum / original_list.length = 45) →
  (let new_list := original_list ++ [a, b, c]
   new_list.sum / new_list.length = 50) →
  (a + b + c) / 3 = 250 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l574_57443


namespace NUMINAMATH_CALUDE_cornelia_age_proof_l574_57465

/-- Cornelia's current age -/
def cornelia_age : ℕ := 80

/-- Kilee's current age -/
def kilee_age : ℕ := 20

/-- In 10 years, Cornelia will be three times as old as Kilee -/
theorem cornelia_age_proof :
  cornelia_age + 10 = 3 * (kilee_age + 10) :=
by sorry

end NUMINAMATH_CALUDE_cornelia_age_proof_l574_57465


namespace NUMINAMATH_CALUDE_largest_two_digit_number_with_condition_l574_57459

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_valid : tens ≥ 1 ∧ tens ≤ 9
  units_valid : units ≥ 0 ∧ units ≤ 9

/-- Checks if a two-digit number satisfies the given condition -/
def satisfiesCondition (n : TwoDigitNumber) : Prop :=
  n.tens * n.units = n.tens + n.units + 17

/-- Theorem: 74 is the largest two-digit number satisfying the condition -/
theorem largest_two_digit_number_with_condition :
  ∀ n : TwoDigitNumber, satisfiesCondition n → n.tens * 10 + n.units ≤ 74 := by
  sorry

#check largest_two_digit_number_with_condition

end NUMINAMATH_CALUDE_largest_two_digit_number_with_condition_l574_57459


namespace NUMINAMATH_CALUDE_legendre_symbol_three_l574_57400

-- Define the Legendre symbol
noncomputable def legendre_symbol (a p : ℕ) : ℤ := sorry

-- Define the theorem
theorem legendre_symbol_three (p : ℕ) (h_prime : Nat.Prime p) :
  (p % 12 = 1 ∨ p % 12 = 11 → legendre_symbol 3 p = 1) ∧
  (p % 12 = 5 ∨ p % 12 = 7 → legendre_symbol 3 p = -1) := by
  sorry

end NUMINAMATH_CALUDE_legendre_symbol_three_l574_57400


namespace NUMINAMATH_CALUDE_cycling_trip_distances_l574_57412

-- Define the total route distance
def total_distance : ℝ := 120

-- Define the distances traveled each day
def day1_distance : ℝ := 36
def day2_distance : ℝ := 40
def day3_distance : ℝ := 44

-- Theorem statement
theorem cycling_trip_distances :
  -- Day 1 condition
  day1_distance = total_distance / 3 - 4 ∧
  -- Day 2 condition
  day2_distance = (total_distance - day1_distance) / 2 - 2 ∧
  -- Day 3 condition
  day3_distance = (total_distance - day1_distance - day2_distance) * 10 / 11 + 4 ∧
  -- Total distance is the sum of all days
  total_distance = day1_distance + day2_distance + day3_distance :=
by sorry


end NUMINAMATH_CALUDE_cycling_trip_distances_l574_57412


namespace NUMINAMATH_CALUDE_sara_score_is_26_l574_57424

/-- Represents a mathematics contest with a specific scoring system -/
structure MathContest where
  total_questions : Nat
  correct_points : Int
  incorrect_points : Int
  unanswered_points : Int

/-- Represents a contestant's performance in the math contest -/
structure ContestPerformance where
  correct_answers : Nat
  incorrect_answers : Nat
  unanswered : Nat

/-- Calculates the total score for a contestant given their performance and the contest rules -/
def calculate_score (contest : MathContest) (performance : ContestPerformance) : Int :=
  performance.correct_answers * contest.correct_points +
  performance.incorrect_answers * contest.incorrect_points +
  performance.unanswered * contest.unanswered_points

/-- The specific math contest Sara participated in -/
def sara_contest : MathContest :=
  { total_questions := 30
  , correct_points := 2
  , incorrect_points := -1
  , unanswered_points := 0 }

/-- Sara's performance in the contest -/
def sara_performance : ContestPerformance :=
  { correct_answers := 18
  , incorrect_answers := 10
  , unanswered := 2 }

/-- Theorem stating that Sara's score in the contest is 26 points -/
theorem sara_score_is_26 :
  calculate_score sara_contest sara_performance = 26 := by
  sorry

end NUMINAMATH_CALUDE_sara_score_is_26_l574_57424


namespace NUMINAMATH_CALUDE_proposition_form_l574_57484

theorem proposition_form : 
  ∃ (p q : Prop), (12 % 4 = 0 ∧ 12 % 3 = 0) ↔ (p ∧ q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_form_l574_57484


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l574_57429

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 = 5 → b^2 = 12 → c^2 = a^2 + b^2 →
  c^2 = 17 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l574_57429
