import Mathlib

namespace NUMINAMATH_CALUDE_coconut_trips_proof_l3659_365942

def coconut_problem (total_coconuts : ℕ) (barbie_capacity : ℕ) (bruno_capacity : ℕ) : ℕ :=
  (total_coconuts + barbie_capacity + bruno_capacity - 1) / (barbie_capacity + bruno_capacity)

theorem coconut_trips_proof :
  coconut_problem 144 4 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_coconut_trips_proof_l3659_365942


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3659_365938

/-- Quadratic function -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_function_properties (b c : ℝ) :
  (∀ x, f b c x ≥ f b c 1) →  -- minimum at x = 1
  f b c 1 = 3 →              -- minimum value is 3
  f b c 2 = 4 →              -- f(2) = 4
  b = -2 ∧ c = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3659_365938


namespace NUMINAMATH_CALUDE_petting_zoo_theorem_l3659_365923

theorem petting_zoo_theorem (total_animals : ℕ) (carrot_eaters : ℕ) (hay_eaters : ℕ) (both_eaters : ℕ) :
  total_animals = 75 →
  carrot_eaters = 26 →
  hay_eaters = 56 →
  both_eaters = 14 →
  total_animals - (carrot_eaters + hay_eaters - both_eaters) = 7 :=
by sorry

end NUMINAMATH_CALUDE_petting_zoo_theorem_l3659_365923


namespace NUMINAMATH_CALUDE_family_reunion_children_l3659_365916

theorem family_reunion_children (adults children : ℕ) : 
  adults = children / 3 →
  adults / 3 + 10 = adults →
  children = 45 := by
sorry

end NUMINAMATH_CALUDE_family_reunion_children_l3659_365916


namespace NUMINAMATH_CALUDE_regular_iff_all_face_angles_equal_exists_non_regular_with_five_equal_angles_l3659_365961

-- Define a tetrahedron
structure Tetrahedron where
  A : Point
  B : Point
  C : Point
  D : Point

-- Define a function to calculate the angle between two faces of a tetrahedron
def angleBetweenFaces (t : Tetrahedron) (face1 : Fin 4) (face2 : Fin 4) : ℝ := sorry

-- Define what it means for a tetrahedron to be regular
def isRegular (t : Tetrahedron) : Prop := sorry

-- Define what it means for all face angles to be equal
def allFaceAnglesEqual (t : Tetrahedron) : Prop :=
  ∀ (i j k l : Fin 4), i ≠ j ∧ k ≠ l → angleBetweenFaces t i j = angleBetweenFaces t k l

-- Define what it means for five out of six face angles to be equal
def fiveFaceAnglesEqual (t : Tetrahedron) : Prop :=
  ∃ (i j k l m n : Fin 4), i ≠ j ∧ k ≠ l ∧ m ≠ n ∧
    angleBetweenFaces t i j = angleBetweenFaces t k l ∧
    angleBetweenFaces t i j = angleBetweenFaces t m n ∧
    (∀ (a b : Fin 4), a ≠ b → 
      angleBetweenFaces t a b = angleBetweenFaces t i j ∨
      angleBetweenFaces t a b = angleBetweenFaces t k l ∨
      angleBetweenFaces t a b = angleBetweenFaces t m n)

-- Theorem 1: A tetrahedron is regular if and only if all face angles are equal
theorem regular_iff_all_face_angles_equal (t : Tetrahedron) :
  isRegular t ↔ allFaceAnglesEqual t := by sorry

-- Theorem 2: There exists a non-regular tetrahedron with five equal face angles
theorem exists_non_regular_with_five_equal_angles :
  ∃ (t : Tetrahedron), fiveFaceAnglesEqual t ∧ ¬isRegular t := by sorry

end NUMINAMATH_CALUDE_regular_iff_all_face_angles_equal_exists_non_regular_with_five_equal_angles_l3659_365961


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3659_365907

theorem quadratic_discriminant :
  let a : ℝ := 2
  let b : ℝ := 2 + Real.sqrt 2
  let c : ℝ := 1/2
  (b^2 - 4*a*c) = 2 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3659_365907


namespace NUMINAMATH_CALUDE_complex_fraction_product_l3659_365998

theorem complex_fraction_product (a b : ℝ) : 
  (1 + Complex.I) / (1 - Complex.I) = Complex.mk a b → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_product_l3659_365998


namespace NUMINAMATH_CALUDE_polynomial_equality_l3659_365982

theorem polynomial_equality : 99^5 - 5*99^4 + 10*99^3 - 10*99^2 + 5*99 - 1 = 98^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3659_365982


namespace NUMINAMATH_CALUDE_shortest_player_height_l3659_365962

theorem shortest_player_height (tallest_height : Float) (height_difference : Float) :
  tallest_height = 77.75 →
  height_difference = 9.5 →
  tallest_height - height_difference = 68.25 := by
  sorry

end NUMINAMATH_CALUDE_shortest_player_height_l3659_365962


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3659_365988

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + c^2*b^2 ≥ 15/16 ∧ 
  (a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + c^2*b^2 = 15/16 ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3659_365988


namespace NUMINAMATH_CALUDE_max_value_polynomial_l3659_365983

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  ∃ M : ℝ, M = (6084 : ℝ) / 17 ∧
  ∀ z w : ℝ, z + w = 5 →
    z^4*w + z^3*w + z^2*w + z*w + z*w^2 + z*w^3 + z*w^4 ≤ M ∧
    ∃ a b : ℝ, a + b = 5 ∧
      a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l3659_365983


namespace NUMINAMATH_CALUDE_prob_not_all_same_l3659_365980

-- Define a fair 6-sided die
def fair_die : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the probability of all dice showing the same number
def prob_all_same : ℚ := 1 / 1296

-- Theorem statement
theorem prob_not_all_same (d : ℕ) (n : ℕ) (p : ℚ) 
  (hd : d = fair_die) (hn : n = num_dice) (hp : p = prob_all_same) : 
  1 - p = 1295 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_all_same_l3659_365980


namespace NUMINAMATH_CALUDE_problem_statement_l3659_365929

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) := ∀ x, g (-x) = g x

theorem problem_statement (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g)
  (h1 : f (-1) + g 1 = 2) (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3659_365929


namespace NUMINAMATH_CALUDE_number_added_before_division_l3659_365963

theorem number_added_before_division (x : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) → 
  (∃ n : ℤ, ∃ m : ℤ, x + n = 41 * m + 18) → 
  (∃ n : ℤ, x + n ≡ 18 [ZMOD 41] ∧ n = 5) :=
by sorry

end NUMINAMATH_CALUDE_number_added_before_division_l3659_365963


namespace NUMINAMATH_CALUDE_mary_stickers_l3659_365986

/-- The number of stickers Mary brought to class -/
def stickers_brought : ℕ := 50

/-- The number of Mary's friends -/
def num_friends : ℕ := 5

/-- The number of stickers Mary gave to each friend -/
def stickers_per_friend : ℕ := 4

/-- The number of stickers Mary gave to each other student -/
def stickers_per_other : ℕ := 2

/-- The number of stickers Mary has left over -/
def stickers_leftover : ℕ := 8

/-- The total number of students in the class, including Mary -/
def total_students : ℕ := 17

theorem mary_stickers :
  stickers_brought =
    num_friends * stickers_per_friend +
    (total_students - 1 - num_friends) * stickers_per_other +
    stickers_leftover :=
by sorry

end NUMINAMATH_CALUDE_mary_stickers_l3659_365986


namespace NUMINAMATH_CALUDE_proposition_counterexample_l3659_365970

theorem proposition_counterexample : 
  ∃ (α β : Real), 
    α > β ∧ 
    0 < α ∧ α < Real.pi / 2 ∧
    0 < β ∧ β < Real.pi / 2 ∧
    Real.tan α ≤ Real.tan β :=
by sorry

end NUMINAMATH_CALUDE_proposition_counterexample_l3659_365970


namespace NUMINAMATH_CALUDE_sqrt_15_minus_one_over_three_lt_one_l3659_365909

theorem sqrt_15_minus_one_over_three_lt_one :
  (Real.sqrt 15 - 1) / 3 < 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_15_minus_one_over_three_lt_one_l3659_365909


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3659_365981

theorem max_value_quadratic (f : ℝ → ℝ) (h : ∀ x, f x = -x^2 + 2*x + 3) :
  (∀ x ∈ Set.Icc 2 3, f x ≤ 3) ∧ (∃ x ∈ Set.Icc 2 3, f x = 3) := by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3659_365981


namespace NUMINAMATH_CALUDE_vector_angle_obtuse_m_values_l3659_365951

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, -3)

def angle_obtuse (x y : ℝ × ℝ) : Prop :=
  let dot_product := x.1 * y.1 + x.2 * y.2
  let magnitude_x := Real.sqrt (x.1^2 + x.2^2)
  let magnitude_y := Real.sqrt (y.1^2 + y.2^2)
  dot_product < 0 ∧ dot_product ≠ -magnitude_x * magnitude_y

theorem vector_angle_obtuse_m_values :
  ∀ m : ℝ, angle_obtuse a (b m) → m = -4 ∨ m = 7/4 :=
sorry

end NUMINAMATH_CALUDE_vector_angle_obtuse_m_values_l3659_365951


namespace NUMINAMATH_CALUDE_polynomial_zero_l3659_365911

-- Define the polynomial
def P (x : ℂ) (p q α β : ℤ) : ℂ := 
  (x - p) * (x - q) * (x^2 + α*x + β)

-- State the theorem
theorem polynomial_zero (p q : ℤ) : 
  ∃ (α β : ℤ), P ((3 + Complex.I * Real.sqrt 15) / 2) p q α β = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_zero_l3659_365911


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l3659_365993

/-- Represents the dimensions and properties of a metallic sheet and the box formed from it. -/
structure MetallicSheet where
  length : ℝ
  width : ℝ
  cutSquareSize : ℝ
  boxVolume : ℝ

/-- Calculates the volume of the box formed from the metallic sheet. -/
def boxVolumeCalc (sheet : MetallicSheet) : ℝ :=
  (sheet.length - 2 * sheet.cutSquareSize) * 
  (sheet.width - 2 * sheet.cutSquareSize) * 
  sheet.cutSquareSize

/-- Theorem stating the width of the metallic sheet given the conditions. -/
theorem metallic_sheet_width 
  (sheet : MetallicSheet)
  (h1 : sheet.length = 48)
  (h2 : sheet.cutSquareSize = 8)
  (h3 : sheet.boxVolume = 5120)
  (h4 : boxVolumeCalc sheet = sheet.boxVolume) :
  sheet.width = 36 :=
sorry

end NUMINAMATH_CALUDE_metallic_sheet_width_l3659_365993


namespace NUMINAMATH_CALUDE_initial_water_percentage_l3659_365969

theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_volume = 70)
  (h2 : added_water = 14)
  (h3 : final_water_percentage = 25)
  (h4 : (initial_volume * x / 100 + added_water) / (initial_volume + added_water) = final_water_percentage / 100) :
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l3659_365969


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3659_365960

def sachin_age : ℚ := 24.5
def age_difference : ℕ := 7

theorem age_ratio_proof :
  let rahul_age : ℚ := sachin_age + age_difference
  (sachin_age / rahul_age) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3659_365960


namespace NUMINAMATH_CALUDE_star_commutative_star_not_distributive_l3659_365955

/-- Binary operation ⋆ -/
def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

/-- Commutativity of ⋆ -/
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

/-- Non-distributivity of ⋆ over addition -/
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

end NUMINAMATH_CALUDE_star_commutative_star_not_distributive_l3659_365955


namespace NUMINAMATH_CALUDE_rectangle_formation_count_l3659_365910

/-- The number of ways to choose 2 items from a set of 5 items -/
def choose_two_from_five : ℕ := 10

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 5

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 5

/-- The number of ways to choose 4 lines to form a rectangle -/
def ways_to_form_rectangle : ℕ := choose_two_from_five * choose_two_from_five

theorem rectangle_formation_count :
  ways_to_form_rectangle = 100 :=
sorry

end NUMINAMATH_CALUDE_rectangle_formation_count_l3659_365910


namespace NUMINAMATH_CALUDE_band_member_earnings_l3659_365932

theorem band_member_earnings (attendees : ℕ) (ticket_price : ℝ) (band_share : ℝ) (band_members : ℕ) : 
  attendees = 500 → 
  ticket_price = 30 → 
  band_share = 0.7 → 
  band_members = 4 → 
  (attendees * ticket_price * band_share) / band_members = 2625 := by
sorry

end NUMINAMATH_CALUDE_band_member_earnings_l3659_365932


namespace NUMINAMATH_CALUDE_midpoint_fraction_l3659_365973

theorem midpoint_fraction : 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 6
  (a + b) / 2 = (19 : ℚ) / 24 := by
sorry

end NUMINAMATH_CALUDE_midpoint_fraction_l3659_365973


namespace NUMINAMATH_CALUDE_intersection_M_N_l3659_365976

def M : Set ℕ := {1, 2, 3, 4}

def N : Set ℕ := {x | ∃ n ∈ M, x = n^2}

theorem intersection_M_N : M ∩ N = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3659_365976


namespace NUMINAMATH_CALUDE_square_root_sum_simplification_l3659_365906

theorem square_root_sum_simplification :
  Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + 
  Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9) - 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_simplification_l3659_365906


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3659_365994

theorem opposite_of_negative_fraction :
  ∀ (x : ℚ), x = -6/7 → (x + 6/7 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l3659_365994


namespace NUMINAMATH_CALUDE_manager_wage_l3659_365927

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure Wages where
  manager : ℝ
  chef : ℝ
  dishwasher : ℝ

/-- The conditions for wages at Joe's Steakhouse -/
def wage_conditions (w : Wages) : Prop :=
  w.chef = w.dishwasher * 1.2 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.manager - 3.4

/-- The theorem stating that under the given conditions, the manager's hourly wage is $8.50 -/
theorem manager_wage (w : Wages) (h : wage_conditions w) : w.manager = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_manager_wage_l3659_365927


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3659_365985

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℚ) (d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a : ℚ) (d : ℚ) (h1 : a = 1/2) (h2 : d = 2/3) :
  arithmetic_sequence a d 10 = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3659_365985


namespace NUMINAMATH_CALUDE_monkey_peach_problem_l3659_365925

/-- The number of peaches the monkey's mother originally had -/
def mothers_original_peaches (little_monkey_initial : ℕ) (peaches_given : ℕ) (mother_ratio : ℕ) : ℕ :=
  (little_monkey_initial + peaches_given) * mother_ratio + peaches_given

theorem monkey_peach_problem :
  mothers_original_peaches 6 3 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_monkey_peach_problem_l3659_365925


namespace NUMINAMATH_CALUDE_extension_point_coordinates_l3659_365957

/-- Given points A and B, and a point C on the extension of AB such that BC = 2/3 * AB,
    prove that the coordinates of C are (53/3, 17/3). -/
theorem extension_point_coordinates (A B C : ℝ × ℝ) : 
  A = (1, -1) →
  B = (11, 3) →
  C - B = 2/3 • (B - A) →
  C = (53/3, 17/3) := by
sorry

end NUMINAMATH_CALUDE_extension_point_coordinates_l3659_365957


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l3659_365922

theorem bobby_candy_problem (x : ℕ) :
  x + 17 = 43 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l3659_365922


namespace NUMINAMATH_CALUDE_max_ones_in_table_l3659_365975

/-- Represents a table with rows and columns -/
structure Table :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the constraints for the table -/
structure TableConstraints :=
  (table : Table)
  (row_sum_mod_3 : ℕ)
  (col_sum_mod_3 : ℕ)

/-- The maximum number of 1's that can be placed in the table -/
def max_ones (constraints : TableConstraints) : ℕ :=
  sorry

/-- The specific constraints for our problem -/
def our_constraints : TableConstraints :=
  { table := { rows := 2005, cols := 2006 },
    row_sum_mod_3 := 0,
    col_sum_mod_3 := 0 }

/-- The theorem to be proved -/
theorem max_ones_in_table :
  max_ones our_constraints = 1336 :=
sorry

end NUMINAMATH_CALUDE_max_ones_in_table_l3659_365975


namespace NUMINAMATH_CALUDE_george_exchange_rate_l3659_365948

/-- The amount George will receive for each special bill he exchanges on his 25th birthday. -/
def exchange_rate (total_years : ℕ) (spent_percentage : ℚ) (total_exchange_amount : ℚ) : ℚ :=
  let total_bills := total_years
  let remaining_bills := total_bills - (spent_percentage * total_bills)
  total_exchange_amount / remaining_bills

/-- Theorem stating that George will receive $1.50 for each special bill he exchanges. -/
theorem george_exchange_rate :
  exchange_rate 10 (1/5) 12 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_george_exchange_rate_l3659_365948


namespace NUMINAMATH_CALUDE_closed_triangular_path_steps_divisible_by_three_l3659_365956

/-- A closed path on a triangular lattice -/
structure TriangularPath where
  steps : ℕ
  is_closed : Bool

/-- Theorem: The number of steps in a closed path on a triangular lattice is divisible by 3 -/
theorem closed_triangular_path_steps_divisible_by_three (path : TriangularPath) 
  (h : path.is_closed = true) : 
  ∃ k : ℕ, path.steps = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_closed_triangular_path_steps_divisible_by_three_l3659_365956


namespace NUMINAMATH_CALUDE_shark_sighting_relationship_l3659_365958

/-- The relationship between shark sightings in Cape May and Daytona Beach --/
theorem shark_sighting_relationship (total_sightings cape_may_sightings : ℕ) 
  (h1 : total_sightings = 40)
  (h2 : cape_may_sightings = 24)
  (h3 : ∃ R : ℕ, cape_may_sightings = R - 8) :
  ∃ R : ℕ, R = 32 ∧ cape_may_sightings = R - 8 := by
  sorry

end NUMINAMATH_CALUDE_shark_sighting_relationship_l3659_365958


namespace NUMINAMATH_CALUDE_remainder_after_adding_5000_l3659_365902

theorem remainder_after_adding_5000 (n : ℤ) (h : n % 6 = 4) : (n + 5000) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_5000_l3659_365902


namespace NUMINAMATH_CALUDE_tower_of_hanoi_l3659_365903

/-- The minimal number of moves required to transfer n disks
    from one rod to another in the Tower of Hanoi game. -/
def minMoves : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2 * minMoves n + 1

/-- Theorem stating that the minimal number of moves for n disks
    in the Tower of Hanoi game is 2^n - 1. -/
theorem tower_of_hanoi (n : ℕ) : minMoves n = 2^n - 1 := by
  sorry

#eval minMoves 3  -- Expected output: 7
#eval minMoves 4  -- Expected output: 15

end NUMINAMATH_CALUDE_tower_of_hanoi_l3659_365903


namespace NUMINAMATH_CALUDE_valid_purchase_options_l3659_365967

/-- Represents the price of an item in kopecks -/
def ItemPrice : ℕ → Prop := λ p => ∃ (a : ℕ), p = 100 * a + 99

/-- The total cost of the purchase in kopecks -/
def TotalCost : ℕ := 20083

/-- Proposition that n is a valid number of items purchased -/
def ValidPurchase (n : ℕ) : Prop :=
  ∃ (p : ℕ), ItemPrice p ∧ n * p = TotalCost

theorem valid_purchase_options :
  ∀ n : ℕ, ValidPurchase n ↔ (n = 17 ∨ n = 117) :=
sorry

end NUMINAMATH_CALUDE_valid_purchase_options_l3659_365967


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3659_365921

theorem inequality_system_solution :
  let S : Set ℝ := {x | 2 * x + 1 > 0 ∧ (x + 1) / 3 > x - 1}
  S = {x | -1/2 < x ∧ x < 2} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3659_365921


namespace NUMINAMATH_CALUDE_smallest_positive_e_l3659_365933

def is_root (p : ℝ → ℝ) (r : ℝ) : Prop := p r = 0

theorem smallest_positive_e (a b c d e : ℤ) : 
  let p := fun (x : ℝ) => (a : ℝ) * x^4 + (b : ℝ) * x^3 + (c : ℝ) * x^2 + (d : ℝ) * x + (e : ℝ)
  (is_root p (-3) ∧ is_root p 6 ∧ is_root p 10 ∧ is_root p (-1/2)) →
  (e > 0) →
  (∀ e' : ℤ, e' > 0 → 
    let p' := fun (x : ℝ) => (a : ℝ) * x^4 + (b : ℝ) * x^3 + (c : ℝ) * x^2 + (d : ℝ) * x + (e' : ℝ)
    (is_root p' (-3) ∧ is_root p' 6 ∧ is_root p' 10 ∧ is_root p' (-1/2)) → e' ≥ e) →
  e = 180 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_e_l3659_365933


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l3659_365935

theorem slope_angle_of_line (x y : ℝ) :
  y = -Real.sqrt 3 * x + 1 → Real.arctan (-Real.sqrt 3) * (180 / Real.pi) = 120 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l3659_365935


namespace NUMINAMATH_CALUDE_no_prime_pair_divisibility_l3659_365912

theorem no_prime_pair_divisibility : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ (p * q ∣ (2^p - 1) * (2^q - 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_pair_divisibility_l3659_365912


namespace NUMINAMATH_CALUDE_circle_M_equation_l3659_365936

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 5

-- Define the point M
def point_M : ℝ × ℝ := (1, -1)

-- Theorem statement
theorem circle_M_equation :
  (∃ (x y : ℝ), line_equation x y ∧ (x, y) = point_M) ∧
  circle_equation 3 0 ∧
  circle_equation 0 1 →
  ∀ (x y : ℝ), circle_equation x y ↔ (x - 1)^2 + (y + 1)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_circle_M_equation_l3659_365936


namespace NUMINAMATH_CALUDE_quadrilateral_is_rectangle_l3659_365944

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral in the plane -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Distance squared between two points -/
def distanceSquared (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- A quadrilateral is a rectangle if its diagonals bisect each other -/
def isRectangle (quad : Quadrilateral) : Prop :=
  let midpointAC := Point.mk ((quad.A.x + quad.C.x) / 2) ((quad.A.y + quad.C.y) / 2)
  let midpointBD := Point.mk ((quad.B.x + quad.D.x) / 2) ((quad.B.y + quad.D.y) / 2)
  midpointAC = midpointBD

/-- Main theorem -/
theorem quadrilateral_is_rectangle (quad : Quadrilateral) :
  (∀ M N P : Point, ¬collinear M N P →
    distanceSquared M quad.A + distanceSquared M quad.C =
    distanceSquared M quad.B + distanceSquared M quad.D) →
  isRectangle quad :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_is_rectangle_l3659_365944


namespace NUMINAMATH_CALUDE_emmas_garden_area_l3659_365946

theorem emmas_garden_area :
  ∀ (short_posts long_posts : ℕ) (short_side long_side : ℝ),
  short_posts > 1 ∧
  long_posts > 1 ∧
  short_posts + long_posts = 12 ∧
  long_posts = 3 * short_posts ∧
  short_side = 6 * (short_posts - 1) ∧
  long_side = 6 * (long_posts - 1) →
  short_side * long_side = 576 := by
sorry

end NUMINAMATH_CALUDE_emmas_garden_area_l3659_365946


namespace NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l3659_365972

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fourth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 0 = 23)
  (h_last : a 5 = 59) :
  a 3 = 41 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l3659_365972


namespace NUMINAMATH_CALUDE_sues_waiting_time_l3659_365996

/-- Proves that Sue's waiting time in New York is 16 hours given the travel conditions -/
theorem sues_waiting_time (total_time : ℝ) (ny_to_sf_time : ℝ) (no_to_ny_ratio : ℝ) 
  (h1 : total_time = 58)
  (h2 : ny_to_sf_time = 24)
  (h3 : no_to_ny_ratio = 3/4)
  : total_time - (no_to_ny_ratio * ny_to_sf_time) - ny_to_sf_time = 16 := by
  sorry

#check sues_waiting_time

end NUMINAMATH_CALUDE_sues_waiting_time_l3659_365996


namespace NUMINAMATH_CALUDE_pedro_gifts_l3659_365915

theorem pedro_gifts (total : ℕ) (emilio : ℕ) (jorge : ℕ) 
  (h1 : total = 21)
  (h2 : emilio = 11)
  (h3 : jorge = 6) :
  total - (emilio + jorge) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pedro_gifts_l3659_365915


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3659_365931

theorem complex_magnitude_problem (z : ℂ) (h : (1 - 2*I) * z = 5*I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3659_365931


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3659_365919

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (9 + 3 * z) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3659_365919


namespace NUMINAMATH_CALUDE_sum_of_special_sequence_l3659_365943

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) - a n = 5

def geometric_subsequence (a : ℕ → ℚ) : Prop :=
  (a 2)^2 = a 1 * a 5

def sum_of_first_six (a : ℕ → ℚ) : ℚ :=
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6)

theorem sum_of_special_sequence :
  ∀ a : ℕ → ℚ,
  arithmetic_sequence a →
  geometric_subsequence a →
  sum_of_first_six a = 90 :=
sorry

end NUMINAMATH_CALUDE_sum_of_special_sequence_l3659_365943


namespace NUMINAMATH_CALUDE_x_value_l3659_365991

theorem x_value (x y : ℝ) (h1 : 2 * x - y = 14) (h2 : y = 2) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3659_365991


namespace NUMINAMATH_CALUDE_max_value_sin_cos_sum_l3659_365913

/-- The function f(x) = 6 sin x + 8 cos x has a maximum value of 10. -/
theorem max_value_sin_cos_sum :
  ∃ (M : ℝ), M = 10 ∧ ∀ x, 6 * Real.sin x + 8 * Real.cos x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_sum_l3659_365913


namespace NUMINAMATH_CALUDE_expand_polynomial_l3659_365934

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 5 * x - 6) = 4 * x^3 + 7 * x^2 - 21 * x - 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3659_365934


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3659_365937

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the set C with parameters a and b
def C (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- Theorem statement
theorem intersection_implies_sum (a b : ℝ) :
  C a b = A ∩ B → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l3659_365937


namespace NUMINAMATH_CALUDE_set_intersection_complement_empty_l3659_365901

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x | ∃ a : ℕ, x = a - 1}

theorem set_intersection_complement_empty : A ∩ (Set.univ \ B) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_complement_empty_l3659_365901


namespace NUMINAMATH_CALUDE_inequality_proof_l3659_365953

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 7/4 ≥ a*b + 2*a + b/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3659_365953


namespace NUMINAMATH_CALUDE_tan_585_degrees_l3659_365926

theorem tan_585_degrees : Real.tan (585 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_585_degrees_l3659_365926


namespace NUMINAMATH_CALUDE_kevin_initial_cards_l3659_365941

/-- The number of cards Kevin lost -/
def lost_cards : ℝ := 7.0

/-- The number of cards Kevin has after losing some -/
def remaining_cards : ℕ := 40

/-- The initial number of cards Kevin found -/
def initial_cards : ℝ := remaining_cards + lost_cards

theorem kevin_initial_cards : initial_cards = 47.0 := by
  sorry

end NUMINAMATH_CALUDE_kevin_initial_cards_l3659_365941


namespace NUMINAMATH_CALUDE_chip_credit_card_balance_l3659_365924

/-- Calculates the balance on a credit card after two months, given an initial balance,
    monthly interest rate, and an additional charge in the second month. -/
def balance_after_two_months (initial_balance : ℝ) (interest_rate : ℝ) (additional_charge : ℝ) : ℝ :=
  let balance_after_first_month := initial_balance * (1 + interest_rate)
  let balance_before_second_interest := balance_after_first_month + additional_charge
  balance_before_second_interest * (1 + interest_rate)

/-- Theorem stating that given the specific conditions of Chip's credit card,
    the balance after two months is $48.00. -/
theorem chip_credit_card_balance :
  balance_after_two_months 50 0.2 20 = 48 :=
by sorry

end NUMINAMATH_CALUDE_chip_credit_card_balance_l3659_365924


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3659_365918

/-- An isosceles triangle with two sides of length 7 and one side of length 3 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 7 ∧ b = 7 ∧ c = 3 → -- Two sides are 7cm and one side is 3cm
  a + b > c ∧ b + c > a ∧ c + a > b → -- Triangle inequality
  (a = b ∨ b = c ∨ c = a) → -- Isosceles condition
  a + b + c = 17 := by -- Perimeter is 17cm
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3659_365918


namespace NUMINAMATH_CALUDE_function_condition_implies_a_range_l3659_365914

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (x + 1) - a * x

theorem function_condition_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → (f a x + a * x) / exp x ≤ a * x) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_function_condition_implies_a_range_l3659_365914


namespace NUMINAMATH_CALUDE_v_2002_equals_4_l3659_365917

def g : ℕ → ℕ
  | 1 => 3
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | 5 => 5
  | _ => 0  -- default case for completeness

def v : ℕ → ℕ
  | 0 => 3
  | n + 1 => g (v n)

theorem v_2002_equals_4 : v 2002 = 4 := by
  sorry

end NUMINAMATH_CALUDE_v_2002_equals_4_l3659_365917


namespace NUMINAMATH_CALUDE_inequality_impossibility_l3659_365974

theorem inequality_impossibility (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(a > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_impossibility_l3659_365974


namespace NUMINAMATH_CALUDE_max_value_of_f_l3659_365979

/-- The function f(x) = x^3 - 3ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem max_value_of_f (a : ℝ) :
  (∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f a x ≥ f a 2) →
  (∃ x, f a x = 18 ∧ ∀ y, f a y ≤ f a x) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3659_365979


namespace NUMINAMATH_CALUDE_apps_after_deletion_l3659_365995

/-- Represents the number of apps on Faye's phone. -/
structure PhoneApps where
  total : ℕ
  gaming : ℕ
  utility : ℕ
  gaming_deleted : ℕ
  utility_deleted : ℕ

/-- Calculates the number of remaining apps after deletion. -/
def remaining_apps (apps : PhoneApps) : ℕ :=
  apps.total - (apps.gaming_deleted + apps.utility_deleted)

/-- Theorem stating the number of remaining apps after deletion. -/
theorem apps_after_deletion (apps : PhoneApps)
  (h1 : apps.total = 12)
  (h2 : apps.gaming = 5)
  (h3 : apps.utility = apps.total - apps.gaming)
  (h4 : apps.gaming_deleted = 4)
  (h5 : apps.utility_deleted = 3)
  (h6 : apps.gaming - apps.gaming_deleted ≥ 1)
  (h7 : apps.utility - apps.utility_deleted ≥ 1) :
  remaining_apps apps = 5 := by
  sorry


end NUMINAMATH_CALUDE_apps_after_deletion_l3659_365995


namespace NUMINAMATH_CALUDE_part1_part2_l3659_365949

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + a*x + 3

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2) 2, f a x ≥ a) ↔ a ∈ Set.Icc (-7) 2 :=
sorry

-- Part 2
theorem part2 (x : ℝ) :
  (∀ a ∈ Set.Icc 4 6, f a x ≥ 0) ↔ 
  x ∈ Set.Iic (-3 - Real.sqrt 6) ∪ Set.Ici (-3 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3659_365949


namespace NUMINAMATH_CALUDE_green_balls_removal_l3659_365971

theorem green_balls_removal (total : ℕ) (initial_green_percentage : ℚ) 
  (final_green_percentage : ℚ) (removed : ℕ) : 
  total = 600 →
  initial_green_percentage = 7/10 →
  final_green_percentage = 3/5 →
  removed = 150 →
  (initial_green_percentage * total - removed) / (total - removed) = final_green_percentage := by
sorry

end NUMINAMATH_CALUDE_green_balls_removal_l3659_365971


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3659_365966

theorem trigonometric_inequality (α : ℝ) : 4 * Real.sin (3 * α) + 5 ≥ 4 * Real.cos (2 * α) + 5 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3659_365966


namespace NUMINAMATH_CALUDE_average_income_Q_R_l3659_365928

theorem average_income_Q_R (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (P + R) / 2 = 5200 →
  P = 4000 →
  (Q + R) / 2 = 6250 := by
sorry

end NUMINAMATH_CALUDE_average_income_Q_R_l3659_365928


namespace NUMINAMATH_CALUDE_max_min_values_l3659_365977

noncomputable def f (x a : ℝ) : ℝ := -x^2 + 2*x + a

theorem max_min_values (a : ℝ) (h : a ≠ 0) :
  ∃ (m n : ℝ),
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x a ≤ m) ∧
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x a = m) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → n ≤ f x a) ∧
    (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x a = n) ∧
    m = 1 + a ∧
    n = -3 + a :=
by
  sorry

end NUMINAMATH_CALUDE_max_min_values_l3659_365977


namespace NUMINAMATH_CALUDE_number_equation_l3659_365997

theorem number_equation (x : ℤ) : 8 * x + 64 = 336 ↔ x = 34 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3659_365997


namespace NUMINAMATH_CALUDE_sine_cosine_shift_l3659_365939

/-- The shift amount between two trigonometric functions -/
def shift_amount (f g : ℝ → ℝ) : ℝ :=
  sorry

theorem sine_cosine_shift :
  let f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x
  let g (x : ℝ) := Real.sin x - Real.sqrt 3 * Real.cos x
  let φ := shift_amount f g
  0 < φ ∧ φ < 2 * Real.pi → φ = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_shift_l3659_365939


namespace NUMINAMATH_CALUDE_triangle_side_length_l3659_365900

/-- In a triangle ABC, given specific angle and side length conditions, prove the length of side b. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = 4 * A ∧  -- Given angle condition
  a = 20 ∧  -- Given side length
  c = 40 ∧  -- Given side length
  a / Real.sin A = b / Real.sin B ∧  -- Law of Sines
  a / Real.sin A = c / Real.sin C  -- Law of Sines
  →
  b = 20 * (16 * (9 * Real.sqrt 3 / 16) - 20 * (3 * Real.sqrt 3 / 4) + 5 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3659_365900


namespace NUMINAMATH_CALUDE_triangle_area_l3659_365947

/-- The area of a triangle with vertices at (3, -3), (3, 4), and (8, -3) is 17.5 square units -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (3, -3)
  let v2 : ℝ × ℝ := (3, 4)
  let v3 : ℝ × ℝ := (8, -3)
  let area := abs ((v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2)) / 2)
  area = 17.5 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_l3659_365947


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l3659_365990

def visitors_previous_day : ℕ := 100
def additional_visitors : ℕ := 566

theorem buckingham_palace_visitors :
  visitors_previous_day + additional_visitors = 666 := by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l3659_365990


namespace NUMINAMATH_CALUDE_julia_bought_496_balls_l3659_365940

/-- The number of balls Julia bought -/
def total_balls : ℕ :=
  let red_packs : ℕ := 3
  let yellow_packs : ℕ := 10
  let green_packs : ℕ := 8
  let blue_packs : ℕ := 5
  let red_balls_per_pack : ℕ := 22
  let yellow_balls_per_pack : ℕ := 19
  let green_balls_per_pack : ℕ := 15
  let blue_balls_per_pack : ℕ := 24
  red_packs * red_balls_per_pack +
  yellow_packs * yellow_balls_per_pack +
  green_packs * green_balls_per_pack +
  blue_packs * blue_balls_per_pack

theorem julia_bought_496_balls : total_balls = 496 := by
  sorry

end NUMINAMATH_CALUDE_julia_bought_496_balls_l3659_365940


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l3659_365959

theorem hot_dogs_remainder : 25197629 % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l3659_365959


namespace NUMINAMATH_CALUDE_outfit_count_l3659_365964

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 8

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pants available -/
def num_pants : ℕ := 8

/-- The number of hats available -/
def num_hats : ℕ := 8

/-- A function that calculates the number of valid outfits -/
def valid_outfits : ℕ := 
  num_colors * num_colors * num_colors - 
  (num_colors * (num_colors - 1) * 3)

/-- Theorem stating that the number of valid outfits is 344 -/
theorem outfit_count : valid_outfits = 344 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l3659_365964


namespace NUMINAMATH_CALUDE_pictures_per_album_l3659_365999

/-- Given the number of pictures uploaded from a phone and a camera, and the number of albums,
    prove that the number of pictures in each album is correct. -/
theorem pictures_per_album
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (num_albums : ℕ)
  (h1 : phone_pics = 35)
  (h2 : camera_pics = 5)
  (h3 : num_albums = 5)
  (h4 : num_albums > 0) :
  (phone_pics + camera_pics) / num_albums = 8 := by
sorry

end NUMINAMATH_CALUDE_pictures_per_album_l3659_365999


namespace NUMINAMATH_CALUDE_parallelogram_intersection_l3659_365930

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : Point) (b : Point)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A : Point) (B : Point) (C : Point) (D : Point)

/-- Checks if a point is inside a parallelogram -/
def isInside (p : Point) (para : Parallelogram) : Prop := sorry

/-- Checks if two lines are parallel -/
def isParallel (l1 l2 : Line) : Prop := sorry

/-- Checks if a point lies on a line -/
def isOnLine (p : Point) (l : Line) : Prop := sorry

/-- Checks if three lines intersect at a single point -/
def intersectAtOnePoint (l1 l2 l3 : Line) : Prop := sorry

theorem parallelogram_intersection 
  (ABCD : Parallelogram) 
  (M : Point) 
  (P Q R S : Point)
  (PR QS BS PD MC : Line)
  (h1 : isInside M ABCD)
  (h2 : isParallel PR (Line.mk ABCD.B ABCD.C))
  (h3 : isParallel QS (Line.mk ABCD.A ABCD.B))
  (h4 : isOnLine P (Line.mk ABCD.A ABCD.B))
  (h5 : isOnLine Q (Line.mk ABCD.B ABCD.C))
  (h6 : isOnLine R (Line.mk ABCD.C ABCD.D))
  (h7 : isOnLine S (Line.mk ABCD.D ABCD.A))
  (h8 : PR = Line.mk P R)
  (h9 : QS = Line.mk Q S)
  (h10 : BS = Line.mk ABCD.B S)
  (h11 : PD = Line.mk P ABCD.D)
  (h12 : MC = Line.mk M ABCD.C)
  : intersectAtOnePoint BS PD MC := sorry

end NUMINAMATH_CALUDE_parallelogram_intersection_l3659_365930


namespace NUMINAMATH_CALUDE_triathlon_completion_time_l3659_365905

/-- A triathlon participant's speeds and completion time -/
theorem triathlon_completion_time 
  (swim_dist : ℝ) 
  (cycle_dist : ℝ) 
  (run_dist : ℝ) 
  (swim_speed : ℝ) 
  (h1 : swim_dist = 1.5) 
  (h2 : cycle_dist = 40) 
  (h3 : run_dist = 10) 
  (h4 : swim_speed > 0) 
  (h5 : swim_speed * 5 * 2.5 * (swim_dist / swim_speed + run_dist / (5 * swim_speed)) = 
        cycle_dist + swim_speed * 5 * 2.5 * 6) : 
  swim_dist / swim_speed + cycle_dist / (swim_speed * 5 * 2.5) + run_dist / (swim_speed * 5) = 134 :=
by sorry

end NUMINAMATH_CALUDE_triathlon_completion_time_l3659_365905


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3659_365904

theorem complex_fraction_sum : 
  let U := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
           1 / (Real.sqrt 14 - 3) - 1 / (3 - Real.sqrt 12) + 
           1 / (Real.sqrt 12 - Real.sqrt 11)
  U = 10 + Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3659_365904


namespace NUMINAMATH_CALUDE_alyssa_games_this_year_l3659_365987

/-- The number of soccer games Alyssa attended over three years -/
def total_games : ℕ := 39

/-- The number of games Alyssa attended last year -/
def last_year_games : ℕ := 13

/-- The number of games Alyssa plans to attend next year -/
def next_year_games : ℕ := 15

/-- The number of games Alyssa attended this year -/
def this_year_games : ℕ := total_games - last_year_games - next_year_games

theorem alyssa_games_this_year :
  this_year_games = 11 := by sorry

end NUMINAMATH_CALUDE_alyssa_games_this_year_l3659_365987


namespace NUMINAMATH_CALUDE_inequality_proof_l3659_365908

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3659_365908


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l3659_365954

theorem relationship_between_exponents 
  (a b c : ℝ) (x y q z : ℝ) 
  (h1 : a^x = c^q) (h2 : a^x = b^2) (h3 : c^q = b^2)
  (h4 : c^y = a^z) (h5 : c^y = b^3) (h6 : a^z = b^3) :
  x * q = y * z := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l3659_365954


namespace NUMINAMATH_CALUDE_mary_fruit_change_l3659_365950

/-- The change Mary received after buying fruits -/
theorem mary_fruit_change (berries_cost peaches_cost payment : ℚ) 
  (h1 : berries_cost = 719 / 100)
  (h2 : peaches_cost = 683 / 100)
  (h3 : payment = 20) :
  payment - (berries_cost + peaches_cost) = 598 / 100 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_change_l3659_365950


namespace NUMINAMATH_CALUDE_product_inequality_l3659_365989

theorem product_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  min (a * (1 - b)) (min (b * (1 - c)) (c * (1 - a))) ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3659_365989


namespace NUMINAMATH_CALUDE_function_representation_flexibility_l3659_365978

-- Define a function type
def Function (α : Type) (β : Type) := α → β

-- State the theorem
theorem function_representation_flexibility 
  {α β : Type} (f : Function α β) : 
  ¬ (∀ (formula : α → β), f = formula) :=
sorry

end NUMINAMATH_CALUDE_function_representation_flexibility_l3659_365978


namespace NUMINAMATH_CALUDE_maud_olive_flea_multiple_l3659_365984

/-- The number of fleas on Gertrude -/
def gertrude_fleas : ℕ := 10

/-- The number of fleas on Olive -/
def olive_fleas : ℕ := gertrude_fleas / 2

/-- The total number of fleas on all chickens -/
def total_fleas : ℕ := 40

/-- The number of fleas on Maud -/
def maud_fleas : ℕ := total_fleas - gertrude_fleas - olive_fleas

/-- The multiple of fleas Maud has compared to Olive -/
def maud_olive_multiple : ℕ := maud_fleas / olive_fleas

theorem maud_olive_flea_multiple :
  maud_olive_multiple = 5 := by sorry

end NUMINAMATH_CALUDE_maud_olive_flea_multiple_l3659_365984


namespace NUMINAMATH_CALUDE_circles_separated_l3659_365945

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Define the centers of the circles
def center₁ : ℝ × ℝ := (-2, -1)
def center₂ : ℝ × ℝ := (2, 1)

-- Define the radius of the circles
def radius : ℝ := 2

-- Theorem: The circles C₁ and C₂ are separated
theorem circles_separated : 
  ∀ (x y : ℝ), (C₁ x y ∧ C₂ x y) → 
  (center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2 > (radius + radius)^2 :=
sorry

end NUMINAMATH_CALUDE_circles_separated_l3659_365945


namespace NUMINAMATH_CALUDE_cube_preserves_inequality_l3659_365965

theorem cube_preserves_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_inequality_l3659_365965


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3659_365920

theorem sum_of_three_numbers : 731 + 672 + 586 = 1989 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3659_365920


namespace NUMINAMATH_CALUDE_pet_center_final_count_l3659_365992

/-- 
Given:
- initial_dogs: The initial number of dogs in the pet center
- initial_cats: The initial number of cats in the pet center
- adopted_dogs: The number of dogs adopted
- new_cats: The number of new cats collected

Prove that the final number of pets in the pet center is 57.
-/
theorem pet_center_final_count 
  (initial_dogs : ℕ) 
  (initial_cats : ℕ) 
  (adopted_dogs : ℕ) 
  (new_cats : ℕ) 
  (h1 : initial_dogs = 36)
  (h2 : initial_cats = 29)
  (h3 : adopted_dogs = 20)
  (h4 : new_cats = 12) :
  initial_dogs - adopted_dogs + initial_cats + new_cats = 57 :=
by
  sorry


end NUMINAMATH_CALUDE_pet_center_final_count_l3659_365992


namespace NUMINAMATH_CALUDE_perfect_square_identity_l3659_365952

theorem perfect_square_identity (x y : ℝ) : x^2 + 2*x*y + y^2 = (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_identity_l3659_365952


namespace NUMINAMATH_CALUDE_exact_rolls_probability_l3659_365968

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The number of dice rolled -/
def dice : ℕ := 8

/-- The number of dice we want to show a specific number -/
def target : ℕ := 4

/-- The probability of rolling exactly 'target' number of twos 
    when rolling 'dice' number of 'sides'-sided dice -/
def probability : ℚ := 168070 / 16777216

theorem exact_rolls_probability : 
  (Nat.choose dice target * (1 / sides) ^ target * ((sides - 1) / sides) ^ (dice - target)) = probability := by
  sorry

end NUMINAMATH_CALUDE_exact_rolls_probability_l3659_365968
