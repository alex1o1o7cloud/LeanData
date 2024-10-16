import Mathlib

namespace NUMINAMATH_CALUDE_box_dimensions_l952_95210

def is_valid_box (a b : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ b ∧ a^2 * b = a^2 + 4*a*b

theorem box_dimensions :
  ∀ a b : ℕ, is_valid_box a b ↔ (a = 8 ∧ b = 2) ∨ (a = 5 ∧ b = 5) :=
sorry

end NUMINAMATH_CALUDE_box_dimensions_l952_95210


namespace NUMINAMATH_CALUDE_equilateral_triangle_circumcircle_area_l952_95221

/-- The area of the circumcircle of an equilateral triangle with side length 4√3 is 16π -/
theorem equilateral_triangle_circumcircle_area :
  let side_length : ℝ := 4 * Real.sqrt 3
  let triangle_area : ℝ := (side_length ^ 2 * Real.sqrt 3) / 4
  let circumradius : ℝ := side_length / Real.sqrt 3
  circumradius ^ 2 * Real.pi = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circumcircle_area_l952_95221


namespace NUMINAMATH_CALUDE_cube_root_of_four_condition_l952_95235

theorem cube_root_of_four_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_four_condition_l952_95235


namespace NUMINAMATH_CALUDE_units_digit_square_equal_l952_95281

theorem units_digit_square_equal (a b : ℕ) (h : (a % 10 + b % 10) = 10) : 
  (a^2 % 10) = (b^2 % 10) := by
sorry

end NUMINAMATH_CALUDE_units_digit_square_equal_l952_95281


namespace NUMINAMATH_CALUDE_loss_per_metre_cloth_l952_95294

/-- Calculates the loss per metre of cloth for a shopkeeper. -/
theorem loss_per_metre_cloth (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) 
  (h1 : total_metres = 300)
  (h2 : total_selling_price = 9000)
  (h3 : cost_price_per_metre = 36) :
  (cost_price_per_metre * total_metres - total_selling_price) / total_metres = 6 := by
  sorry

#check loss_per_metre_cloth

end NUMINAMATH_CALUDE_loss_per_metre_cloth_l952_95294


namespace NUMINAMATH_CALUDE_triangle_theorem_l952_95208

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the condition for an acute-angled triangle
def isAcuteAngled (t : Triangle) : Prop := sorry

-- Define the point P on AC
def pointOnAC (t : Triangle) (P : ℝ × ℝ) : Prop := sorry

-- Define the condition 2AP = BC
def conditionAP (t : Triangle) (P : ℝ × ℝ) : Prop := sorry

-- Define points X and Y symmetric to P with respect to A and C
def symmetricPoints (t : Triangle) (P X Y : ℝ × ℝ) : Prop := sorry

-- Define the condition BX = BY
def equalDistances (t : Triangle) (X Y : ℝ × ℝ) : Prop := sorry

-- Define the angle BCA
def angleBCA (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_theorem (t : Triangle) (P X Y : ℝ × ℝ) :
  isAcuteAngled t →
  pointOnAC t P →
  conditionAP t P →
  symmetricPoints t P X Y →
  equalDistances t X Y →
  angleBCA t = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l952_95208


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l952_95265

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x > 0) ↔ (∃ x : ℝ, x^2 - 2*x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l952_95265


namespace NUMINAMATH_CALUDE_angle_measure_problem_l952_95243

theorem angle_measure_problem (C D : ℝ) 
  (h1 : C + D = 360)
  (h2 : C = 5 * D) : 
  C = 300 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l952_95243


namespace NUMINAMATH_CALUDE_room_width_proof_l952_95252

theorem room_width_proof (room_length : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_length = 20 →
  veranda_width = 2 →
  veranda_area = 144 →
  ∃ room_width : ℝ,
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = veranda_area ∧
    room_width = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_width_proof_l952_95252


namespace NUMINAMATH_CALUDE_no_integer_solution_l952_95201

theorem no_integer_solution : ¬ ∃ (x : ℤ), (x + 12 > 15) ∧ (-3*x > -9) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l952_95201


namespace NUMINAMATH_CALUDE_increasing_quadratic_coefficient_range_l952_95279

def f (m : ℝ) (x : ℝ) := 3 * x^2 + m * x + 2

theorem increasing_quadratic_coefficient_range (m : ℝ) :
  (∀ x ≥ 1, ∀ y > x, f m y > f m x) →
  m ≥ -6 :=
by sorry

end NUMINAMATH_CALUDE_increasing_quadratic_coefficient_range_l952_95279


namespace NUMINAMATH_CALUDE_weight_of_barium_iodide_l952_95296

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of moles of Barium iodide -/
def moles_BaI2 : ℝ := 4

/-- The molecular weight of Barium iodide (BaI2) in g/mol -/
def molecular_weight_BaI2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_I

/-- The total weight of Barium iodide in grams -/
def total_weight_BaI2 : ℝ := moles_BaI2 * molecular_weight_BaI2

theorem weight_of_barium_iodide :
  total_weight_BaI2 = 1564.52 := by sorry

end NUMINAMATH_CALUDE_weight_of_barium_iodide_l952_95296


namespace NUMINAMATH_CALUDE_total_coins_l952_95260

def total_value : ℚ := 71
def value_20_paise : ℚ := 20 / 100
def value_25_paise : ℚ := 25 / 100
def num_20_paise : ℕ := 260

theorem total_coins : ∃ (num_25_paise : ℕ), 
  (num_20_paise : ℚ) * value_20_paise + (num_25_paise : ℚ) * value_25_paise = total_value ∧
  num_20_paise + num_25_paise = 336 :=
sorry

end NUMINAMATH_CALUDE_total_coins_l952_95260


namespace NUMINAMATH_CALUDE_lcm_1332_888_l952_95274

theorem lcm_1332_888 : Nat.lcm 1332 888 = 2664 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1332_888_l952_95274


namespace NUMINAMATH_CALUDE_swim_meet_cars_l952_95227

theorem swim_meet_cars (num_vans : ℕ) (people_per_car : ℕ) (people_per_van : ℕ) 
  (max_per_car : ℕ) (max_per_van : ℕ) (extra_capacity : ℕ) :
  num_vans = 3 →
  people_per_car = 5 →
  people_per_van = 3 →
  max_per_car = 6 →
  max_per_van = 8 →
  extra_capacity = 17 →
  ∃ (num_cars : ℕ), 
    num_cars * people_per_car + num_vans * people_per_van + extra_capacity = 
    num_cars * max_per_car + num_vans * max_per_van ∧
    num_cars = 2 :=
by sorry

end NUMINAMATH_CALUDE_swim_meet_cars_l952_95227


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l952_95211

theorem largest_multiple_of_9_under_100 : ∃ n : ℕ, n * 9 = 99 ∧ 
  99 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ 99 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l952_95211


namespace NUMINAMATH_CALUDE_range_of_y_over_x_l952_95270

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom inequality_condition : ∀ x y : ℝ, f (x^2 - 2*x) ≤ -f (2*y - y^2)
axiom symmetry_condition : ∀ x : ℝ, f (x - 1) = f (1 - x)

-- Define the theorem
theorem range_of_y_over_x :
  (∀ x y : ℝ, 1 ≤ x → x ≤ 4 → f x = y → -1/2 ≤ y/x ∧ y/x ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_l952_95270


namespace NUMINAMATH_CALUDE_square_difference_l952_95214

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l952_95214


namespace NUMINAMATH_CALUDE_find_M_l952_95228

theorem find_M : ∃ M : ℕ, (1001 + 1003 + 1005 + 1007 + 1009 = 5100 - M) → M = 75 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l952_95228


namespace NUMINAMATH_CALUDE_max_area_right_triangle_l952_95250

/-- Given two positive real numbers a and b, the area of a triangle with sides a and b
    is maximized when these sides are perpendicular. -/
theorem max_area_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ θ : ℝ, 0 < θ ∧ θ < π →
    (1/2) * a * b * Real.sin θ ≤ (1/2) * a * b :=
by sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_l952_95250


namespace NUMINAMATH_CALUDE_fuchsia_purple_or_blue_count_l952_95220

/-- Represents the survey results about fuchsia color perception --/
structure FuchsiaSurvey where
  total : ℕ
  like_pink : ℕ
  like_pink_and_purple : ℕ
  like_none : ℕ
  like_all : ℕ

/-- Calculates the number of people who believe fuchsia is "like purple" or "like blue" --/
def purple_or_blue (survey : FuchsiaSurvey) : ℕ :=
  survey.total - survey.like_none - (survey.like_pink - survey.like_pink_and_purple)

/-- Theorem stating that for the given survey results, 64 people believe fuchsia is "like purple" or "like blue" --/
theorem fuchsia_purple_or_blue_count :
  let survey : FuchsiaSurvey := {
    total := 150,
    like_pink := 90,
    like_pink_and_purple := 47,
    like_none := 23,
    like_all := 20
  }
  purple_or_blue survey = 64 := by
  sorry

end NUMINAMATH_CALUDE_fuchsia_purple_or_blue_count_l952_95220


namespace NUMINAMATH_CALUDE_twentieth_number_in_base6_l952_95275

-- Define a function to convert decimal to base 6
def decimalToBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

-- State the theorem
theorem twentieth_number_in_base6 :
  decimalToBase6 20 = [3, 2] :=
sorry

end NUMINAMATH_CALUDE_twentieth_number_in_base6_l952_95275


namespace NUMINAMATH_CALUDE_trigonometric_simplification_trigonometric_evaluation_l952_95284

-- Part 1
theorem trigonometric_simplification (α : ℝ) : 
  (Real.cos (α - π/2)) / (Real.sin (5*π/2 + α)) * Real.sin (α - 2*π) * Real.cos (2*π - α) = Real.sin α ^ 2 := by
  sorry

-- Part 2
theorem trigonometric_evaluation : 
  Real.sin (25*π/6) + Real.cos (25*π/3) + Real.tan (-25*π/4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_trigonometric_evaluation_l952_95284


namespace NUMINAMATH_CALUDE_belmont_basketball_winning_percentage_l952_95212

theorem belmont_basketball_winning_percentage 
  (X : ℕ) (Y Z : ℝ) (h1 : 0 < Y) (h2 : Y < 100) (h3 : 0 < Z) (h4 : Z < 100) :
  let G := X * ((Y / 100) - (Z / 100)) / (Z / 100 - 1)
  ∃ (G : ℝ), (Z / 100) * (X + G) = (Y / 100) * X + G :=
by sorry

end NUMINAMATH_CALUDE_belmont_basketball_winning_percentage_l952_95212


namespace NUMINAMATH_CALUDE_expenditure_ratio_l952_95233

theorem expenditure_ratio (income : ℝ) (h : income > 0) :
  let savings_rate := 0.35
  let income_increase := 0.35
  let savings_increase := 1.0

  let savings_year1 := savings_rate * income
  let expenditure_year1 := income - savings_year1

  let income_year2 := income * (1 + income_increase)
  let savings_year2 := savings_year1 * (1 + savings_increase)
  let expenditure_year2 := income_year2 - savings_year2

  let total_expenditure := expenditure_year1 + expenditure_year2

  (total_expenditure / expenditure_year1) = 2
  := by sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l952_95233


namespace NUMINAMATH_CALUDE_factor_expression_l952_95209

theorem factor_expression (x : ℝ) : 72 * x^5 - 162 * x^9 = -18 * x^5 * (9 * x^4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l952_95209


namespace NUMINAMATH_CALUDE_constant_white_sectors_neighboring_minutes_l952_95222

/-- Represents the color of a sector -/
inductive Color
| White
| Red

/-- Represents the state of the circle -/
def CircleState := Vector Color 1000

/-- Repaints 500 consecutive sectors in the circle -/
def repaint (state : CircleState) (start : Fin 1000) : CircleState :=
  sorry

/-- Counts the number of white sectors in the circle -/
def countWhite (state : CircleState) : Nat :=
  sorry

theorem constant_white_sectors_neighboring_minutes 
  (initial : CircleState)
  (repaints : ℕ → Fin 1000) 
  (n : ℕ) :
  (countWhite (repaint (repaint (repaint initial (repaints (n-1))) (repaints n)) (repaints (n+1))) = 
   countWhite (repaint (repaint initial (repaints (n-1))) (repaints n))) →
  ((countWhite (repaint (repaint initial (repaints (n-1))) (repaints n)) = 
    countWhite (repaint initial (repaints (n-1))))
   ∨ 
   (countWhite (repaint (repaint (repaint initial (repaints (n-1))) (repaints n)) (repaints (n+1))) = 
    countWhite (repaint (repaint initial (repaints (n-1))) (repaints n)))) :=
  sorry

#check constant_white_sectors_neighboring_minutes

end NUMINAMATH_CALUDE_constant_white_sectors_neighboring_minutes_l952_95222


namespace NUMINAMATH_CALUDE_jasmine_cut_length_l952_95240

def ribbon_length : ℕ := 10
def janice_cut_length : ℕ := 2

theorem jasmine_cut_length :
  ∀ (cut_length : ℕ),
    cut_length ≠ janice_cut_length →
    cut_length > 0 →
    ribbon_length % cut_length = 0 →
    cut_length ∣ ribbon_length →
    cut_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_cut_length_l952_95240


namespace NUMINAMATH_CALUDE_miami_ny_temp_difference_l952_95204

/-- Represents the temperatures of three cities and their relationships -/
structure CityTemperatures where
  new_york : ℝ
  miami : ℝ
  san_diego : ℝ
  ny_temp_is_80 : new_york = 80
  miami_cooler_than_sd : miami = san_diego - 25
  average_temp : (new_york + miami + san_diego) / 3 = 95

/-- The temperature difference between Miami and New York -/
def temp_difference (ct : CityTemperatures) : ℝ :=
  ct.miami - ct.new_york

/-- Theorem stating that the temperature difference between Miami and New York is 10 degrees -/
theorem miami_ny_temp_difference (ct : CityTemperatures) : temp_difference ct = 10 := by
  sorry

end NUMINAMATH_CALUDE_miami_ny_temp_difference_l952_95204


namespace NUMINAMATH_CALUDE_total_socks_l952_95258

def sock_problem (red_socks black_socks white_socks : ℕ) : Prop :=
  red_socks = 40 ∧
  black_socks = red_socks / 2 ∧
  white_socks = 2 * (red_socks + black_socks)

theorem total_socks (red_socks black_socks white_socks : ℕ) 
  (h : sock_problem red_socks black_socks white_socks) : 
  red_socks + black_socks + white_socks = 180 :=
by sorry

end NUMINAMATH_CALUDE_total_socks_l952_95258


namespace NUMINAMATH_CALUDE_mod_power_seventeen_seven_l952_95264

theorem mod_power_seventeen_seven (m : ℕ) : 
  17^7 % 11 = m ∧ 0 ≤ m ∧ m < 11 → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_mod_power_seventeen_seven_l952_95264


namespace NUMINAMATH_CALUDE_inscribed_circle_square_area_l952_95241

theorem inscribed_circle_square_area (r : ℝ) (h : r > 0) :
  π * r^2 = 9 * π → 4 * r^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_area_l952_95241


namespace NUMINAMATH_CALUDE_square_roots_values_l952_95255

theorem square_roots_values (m : ℝ) (a : ℝ) (h1 : a > 0) 
  (h2 : (3 * m - 1)^2 = a) (h3 : (-2 * m - 2)^2 = a) :
  a = 64 ∨ a = 64/25 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_values_l952_95255


namespace NUMINAMATH_CALUDE_smallest_n_is_five_l952_95238

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def satisfies_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ is_divisible (n^2 - n + 1) k ∧
  ∃ m : ℕ, 1 ≤ m ∧ m ≤ n + 1 ∧ ¬is_divisible (n^2 - n + 1) m

theorem smallest_n_is_five :
  satisfies_condition 5 ∧
  ∀ n : ℕ, 0 < n ∧ n < 5 → ¬satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_smallest_n_is_five_l952_95238


namespace NUMINAMATH_CALUDE_jack_and_jill_speed_l952_95259

/-- The speed of Jack and Jill's walk, given their conditions -/
theorem jack_and_jill_speed :
  ∀ x : ℝ,
  let jack_speed := x^2 - 13*x - 30
  let jill_distance := x^2 - 5*x - 84
  let jill_time := x + 7
  let jill_speed := jill_distance / jill_time
  (jack_speed = jill_speed) → (jack_speed = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_jack_and_jill_speed_l952_95259


namespace NUMINAMATH_CALUDE_doll_ratio_is_two_to_one_l952_95261

/-- Given the number of Dina's dolls -/
def dinas_dolls : ℕ := 60

/-- Given the fraction of Ivy's dolls that are collectors editions -/
def ivy_collectors_fraction : ℚ := 2/3

/-- Given the number of Ivy's collectors edition dolls -/
def ivy_collectors : ℕ := 20

/-- Calculate the total number of Ivy's dolls -/
def ivys_dolls : ℕ := ivy_collectors * 3 / 2

/-- The ratio of Dina's dolls to Ivy's dolls -/
def doll_ratio : ℚ := dinas_dolls / ivys_dolls

theorem doll_ratio_is_two_to_one : doll_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_doll_ratio_is_two_to_one_l952_95261


namespace NUMINAMATH_CALUDE_sequence_sum_product_l952_95249

theorem sequence_sum_product (n : ℕ+) : 
  let S : ℕ+ → ℚ := λ k => k / (k + 1)
  S n * S (n + 1) = 3/4 → n = 6 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_product_l952_95249


namespace NUMINAMATH_CALUDE_specific_pyramid_surface_area_l952_95297

/-- A right rectangular pyramid with square bases -/
structure RightRectangularPyramid where
  upperBaseEdge : ℝ
  lowerBaseEdge : ℝ
  sideEdge : ℝ

/-- Calculate the surface area of a right rectangular pyramid -/
def surfaceArea (p : RightRectangularPyramid) : ℝ :=
  -- Surface area calculation
  sorry

/-- The theorem stating the surface area of the specific pyramid -/
theorem specific_pyramid_surface_area :
  let p : RightRectangularPyramid := {
    upperBaseEdge := 2,
    lowerBaseEdge := 4,
    sideEdge := 2
  }
  surfaceArea p = 10 * Real.sqrt 3 + 20 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_surface_area_l952_95297


namespace NUMINAMATH_CALUDE_remaining_black_cards_l952_95217

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : ℕ)
  (black_cards : ℕ)
  (red_cards : ℕ)

/-- Defines the properties of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    black_cards := 26,
    red_cards := 26 }

/-- Theorem: After removing 4 black cards from a standard deck, 22 black cards remain -/
theorem remaining_black_cards (d : Deck) (h1 : d = standard_deck) (h2 : d.black_cards = d.red_cards) :
  d.black_cards - 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_black_cards_l952_95217


namespace NUMINAMATH_CALUDE_existence_of_n_for_prime_p_l952_95287

theorem existence_of_n_for_prime_p (p : ℕ) (hp : Prime p) : ∃ n : ℕ, p ∣ (2022^n - n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_for_prime_p_l952_95287


namespace NUMINAMATH_CALUDE_max_product_xy_l952_95215

theorem max_product_xy (x y : ℝ) :
  (Real.sqrt (x + y - 1) + x^4 + y^4 - 1/8 ≤ 0) →
  (x * y ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_max_product_xy_l952_95215


namespace NUMINAMATH_CALUDE_no_point_M_exists_line_EF_exists_l952_95226

-- Define the ellipse C
def C (x y : ℝ) : Prop := (x - 2)^2 + y^2/4 = 1

-- Define the line l
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define point R
def R : ℝ × ℝ := (1, 4)

-- Theorem 1: No point M exists inside C satisfying the given condition
theorem no_point_M_exists : ¬ ∃ M : ℝ × ℝ, 
  C M.1 M.2 ∧ 
  (∀ Q A B : ℝ × ℝ, 
    l Q.1 Q.2 → 
    C A.1 A.2 → 
    C B.1 B.2 → 
    (∃ t : ℝ, M.1 = t * (Q.1 - M.1) + M.1 ∧ M.2 = t * (Q.2 - M.2) + M.2) →
    (A.1 - M.1)^2 + (A.2 - M.2)^2 = (B.1 - M.1)^2 + (B.2 - M.2)^2 →
    (A.1 - M.1)^2 + (A.2 - M.2)^2 = (A.1 - Q.1)^2 + (A.2 - Q.2)^2) :=
sorry

-- Theorem 2: Line EF exists and has the given equations
theorem line_EF_exists : ∃ E F : ℝ × ℝ,
  C E.1 E.2 ∧ 
  C F.1 F.2 ∧
  (R.1 - E.1)^2 + (R.2 - E.2)^2 = (E.1 - F.1)^2 + (E.2 - F.2)^2 ∧
  (2 * E.1 + E.2 - 6 = 0 ∨ 14 * E.1 + E.2 - 18 = 0) ∧
  (2 * F.1 + F.2 - 6 = 0 ∨ 14 * F.1 + F.2 - 18 = 0) :=
sorry

end NUMINAMATH_CALUDE_no_point_M_exists_line_EF_exists_l952_95226


namespace NUMINAMATH_CALUDE_question_mark_value_l952_95288

theorem question_mark_value (question_mark : ℝ) : 
  question_mark * 240 = 173 * 240 → question_mark = 173 := by
  sorry

end NUMINAMATH_CALUDE_question_mark_value_l952_95288


namespace NUMINAMATH_CALUDE_sock_order_ratio_l952_95271

theorem sock_order_ratio (black_pairs blue_pairs : ℕ) (price_blue : ℝ) :
  black_pairs = 4 →
  (4 * 2 * price_blue + blue_pairs * price_blue) * 1.5 = blue_pairs * 2 * price_blue + 4 * price_blue →
  blue_pairs = 16 :=
by sorry

end NUMINAMATH_CALUDE_sock_order_ratio_l952_95271


namespace NUMINAMATH_CALUDE_lizard_adoption_rate_l952_95248

def initial_dogs : ℕ := 30
def initial_cats : ℕ := 28
def initial_lizards : ℕ := 20
def dog_adoption_rate : ℚ := 1/2
def cat_adoption_rate : ℚ := 1/4
def new_pets : ℕ := 13
def total_pets_after_month : ℕ := 65

theorem lizard_adoption_rate : 
  let dogs_adopted := (initial_dogs : ℚ) * dog_adoption_rate
  let cats_adopted := (initial_cats : ℚ) * cat_adoption_rate
  let remaining_dogs := initial_dogs - dogs_adopted.floor
  let remaining_cats := initial_cats - cats_adopted.floor
  let total_before_lizard_adoption := remaining_dogs + remaining_cats + initial_lizards + new_pets
  let lizards_adopted := total_before_lizard_adoption - total_pets_after_month
  lizards_adopted / initial_lizards = 1/5 := by sorry

end NUMINAMATH_CALUDE_lizard_adoption_rate_l952_95248


namespace NUMINAMATH_CALUDE_building_heights_sum_l952_95278

/-- The sum of heights of four buildings with specific height relationships -/
theorem building_heights_sum : 
  let tallest : ℝ := 100
  let second : ℝ := tallest / 2
  let third : ℝ := second / 2
  let fourth : ℝ := third / 5
  tallest + second + third + fourth = 180 := by sorry

end NUMINAMATH_CALUDE_building_heights_sum_l952_95278


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l952_95272

theorem termite_ridden_not_collapsing (total_homes : ℕ) (termite_ridden : ℕ) (collapsing : ℕ)
  (h1 : termite_ridden = total_homes / 3)
  (h2 : collapsing = termite_ridden * 5 / 8) :
  (termite_ridden - collapsing : ℚ) / total_homes = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l952_95272


namespace NUMINAMATH_CALUDE_sequence_increasing_iff_k_greater_than_neg_three_l952_95282

theorem sequence_increasing_iff_k_greater_than_neg_three (k : ℝ) :
  (∀ n : ℕ, (n^2 + k*n + 2) < ((n+1)^2 + k*(n+1) + 2)) ↔ k > -3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_iff_k_greater_than_neg_three_l952_95282


namespace NUMINAMATH_CALUDE_calculation_proof_l952_95237

theorem calculation_proof :
  (1 * (-8) - (-6) + (-3) = -5) ∧
  (5 / 13 - 3.7 + 8 / 13 + 1.7 = -1) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l952_95237


namespace NUMINAMATH_CALUDE_intersection_of_lines_l952_95206

/-- Given two lines m and n that intersect at (2, 7), 
    where m has equation y = 2x + 3 and n has equation y = kx + 1,
    prove that k = 3. -/
theorem intersection_of_lines (k : ℝ) : 
  (∀ x y : ℝ, y = 2*x + 3 → y = k*x + 1 → x = 2 ∧ y = 7) → 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l952_95206


namespace NUMINAMATH_CALUDE_out_of_pocket_calculation_l952_95293

def out_of_pocket (initial_purchase : ℝ) (tv_return : ℝ) (bike_return : ℝ) (toaster_purchase : ℝ) : ℝ :=
  let total_return := tv_return + bike_return
  let sold_bike_cost := bike_return * 1.2
  let sold_bike_price := sold_bike_cost * 0.8
  initial_purchase - total_return - sold_bike_price + toaster_purchase

theorem out_of_pocket_calculation :
  out_of_pocket 3000 700 500 100 = 1420 := by
  sorry

end NUMINAMATH_CALUDE_out_of_pocket_calculation_l952_95293


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_problem_statement_l952_95280

theorem odd_even_sum_difference : ℕ → Prop :=
  fun n =>
    let odd_sum := (n^2 + 2*n + 1)^2
    let even_sum := n * (n + 1) * (2*n + 2)
    odd_sum - even_sum = 3057

theorem problem_statement : odd_even_sum_difference 1012 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_problem_statement_l952_95280


namespace NUMINAMATH_CALUDE_ross_breaths_per_minute_l952_95234

/-- Calculates the number of breaths per minute given the air inhaled per breath and total air inhaled in 24 hours. -/
def breaths_per_minute (air_per_breath : ℚ) (total_air_24h : ℚ) : ℚ :=
  (total_air_24h / air_per_breath) / (24 * 60)

/-- Theorem stating that Ross takes 17 breaths per minute. -/
theorem ross_breaths_per_minute :
  breaths_per_minute (5/9) 13600 = 17 := by
  sorry

#eval breaths_per_minute (5/9) 13600

end NUMINAMATH_CALUDE_ross_breaths_per_minute_l952_95234


namespace NUMINAMATH_CALUDE_class_size_l952_95285

/-- Represents the number of students excelling in various combinations of sports -/
structure SportExcellence where
  sprint : ℕ
  swimming : ℕ
  basketball : ℕ
  sprint_swimming : ℕ
  swimming_basketball : ℕ
  sprint_basketball : ℕ
  all_three : ℕ

/-- The total number of students in the class -/
def total_students (se : SportExcellence) (non_excellent : ℕ) : ℕ :=
  se.sprint + se.swimming + se.basketball
  - se.sprint_swimming - se.swimming_basketball - se.sprint_basketball
  + se.all_three + non_excellent

/-- The theorem stating the total number of students in the class -/
theorem class_size (se : SportExcellence) (non_excellent : ℕ) : 
  se.sprint = 17 → se.swimming = 18 → se.basketball = 15 →
  se.sprint_swimming = 6 → se.swimming_basketball = 6 →
  se.sprint_basketball = 5 → se.all_three = 2 → non_excellent = 4 →
  total_students se non_excellent = 39 := by
  sorry

/-- Example usage of the theorem -/
example : ∃ (se : SportExcellence) (non_excellent : ℕ), 
  total_students se non_excellent = 39 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l952_95285


namespace NUMINAMATH_CALUDE_piglets_count_l952_95200

/-- Calculates the number of piglets given the total number of straws and straws per piglet -/
def number_of_piglets (total_straws : ℕ) (straws_per_piglet : ℕ) : ℕ :=
  let straws_for_adult_pigs := (3 * total_straws) / 5
  let straws_for_piglets := straws_for_adult_pigs
  straws_for_piglets / straws_per_piglet

/-- Proves that the number of piglets is 30 given the problem conditions -/
theorem piglets_count : number_of_piglets 300 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_piglets_count_l952_95200


namespace NUMINAMATH_CALUDE_equipment_purchase_problem_l952_95219

/-- Equipment purchase problem -/
theorem equipment_purchase_problem 
  (price_A : ℕ)
  (price_B : ℕ)
  (discount_B : ℕ)
  (total_units : ℕ)
  (min_B : ℕ)
  (h1 : price_A = 40)
  (h2 : 30 * price_B - 5 * discount_B = 1425)
  (h3 : 50 * price_B - 25 * discount_B = 2125)
  (h4 : total_units = 90)
  (h5 : min_B = 15) :
  ∃ (units_A units_B : ℕ),
    units_A + units_B = total_units ∧
    units_B ≥ min_B ∧
    units_B ≤ 2 * units_A ∧
    units_A * price_A + (min units_B 25) * price_B + 
      (max (units_B - 25) 0) * (price_B - discount_B) = 3675 ∧
    ∀ (a b : ℕ),
      a + b = total_units →
      b ≥ min_B →
      b ≤ 2 * a →
      a * price_A + (min b 25) * price_B + 
        (max (b - 25) 0) * (price_B - discount_B) ≥ 3675 := by
  sorry

end NUMINAMATH_CALUDE_equipment_purchase_problem_l952_95219


namespace NUMINAMATH_CALUDE_smallest_solution_l952_95290

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation in the problem -/
def equation (x : ℝ) : Prop :=
  floor (x^2) - (floor x)^2 = 17

/-- Theorem stating that 7√2 is the smallest solution -/
theorem smallest_solution :
  ∀ x : ℝ, equation x → x ≥ 7 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_l952_95290


namespace NUMINAMATH_CALUDE_tax_free_amount_correct_l952_95289

/-- The tax-free amount for goods purchased in country B -/
def tax_free_amount : ℝ := 600

/-- The total value of goods purchased -/
def total_value : ℝ := 1720

/-- The tax rate applied to the excess amount -/
def tax_rate : ℝ := 0.07

/-- The amount of tax paid -/
def tax_paid : ℝ := 78.4

/-- Theorem stating that the tax-free amount satisfies the given conditions -/
theorem tax_free_amount_correct : 
  tax_rate * (total_value - tax_free_amount) = tax_paid := by
  sorry

end NUMINAMATH_CALUDE_tax_free_amount_correct_l952_95289


namespace NUMINAMATH_CALUDE_remaining_amount_after_ten_months_l952_95276

/-- Represents a loan scenario where a person borrows money and pays it back in monthly installments. -/
structure LoanScenario where
  /-- The total amount borrowed -/
  borrowed_amount : ℝ
  /-- The fixed amount paid back each month -/
  monthly_payment : ℝ
  /-- Assumption that the borrowed amount is positive -/
  borrowed_positive : borrowed_amount > 0
  /-- Assumption that the monthly payment is positive -/
  payment_positive : monthly_payment > 0
  /-- After 6 months, half of the borrowed amount has been paid back -/
  half_paid_after_six_months : 6 * monthly_payment = borrowed_amount / 2

/-- Theorem stating that the remaining amount owed after 10 months is equal to
    the borrowed amount minus 10 times the monthly payment. -/
theorem remaining_amount_after_ten_months (scenario : LoanScenario) :
  scenario.borrowed_amount - 10 * scenario.monthly_payment =
  scenario.borrowed_amount - (6 * scenario.monthly_payment + 4 * scenario.monthly_payment) :=
by sorry

end NUMINAMATH_CALUDE_remaining_amount_after_ten_months_l952_95276


namespace NUMINAMATH_CALUDE_train_length_problem_l952_95268

theorem train_length_problem (faster_speed slower_speed : ℝ) 
  (passing_time : ℝ) (h1 : faster_speed = 47) (h2 : slower_speed = 36) 
  (h3 : passing_time = 36) : ∃ (train_length : ℝ), train_length = 55 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l952_95268


namespace NUMINAMATH_CALUDE_distinct_digits_base_eight_l952_95257

/-- The number of three-digit numbers with distinct digits in base b -/
def distinctDigitNumbers (b : ℕ) : ℕ := (b - 1) * (b - 1) * (b - 2)

/-- Theorem stating that there are 250 three-digit numbers with distinct digits in base 8 -/
theorem distinct_digits_base_eight :
  distinctDigitNumbers 8 = 250 := by
  sorry

end NUMINAMATH_CALUDE_distinct_digits_base_eight_l952_95257


namespace NUMINAMATH_CALUDE_manufacturer_not_fraudulent_l952_95225

/-- Represents the mass of a bread bag -/
structure BreadBag where
  labeledMass : ℝ
  tolerance : ℝ
  measuredMass : ℝ

/-- Determines if the manufacturer has engaged in fraudulent behavior -/
def isFraudulent (bag : BreadBag) : Prop :=
  bag.measuredMass < bag.labeledMass - bag.tolerance ∨ 
  bag.measuredMass > bag.labeledMass + bag.tolerance

theorem manufacturer_not_fraudulent (bag : BreadBag) 
  (h1 : bag.labeledMass = 200)
  (h2 : bag.tolerance = 3)
  (h3 : bag.measuredMass = 198) : 
  ¬(isFraudulent bag) := by
  sorry

#check manufacturer_not_fraudulent

end NUMINAMATH_CALUDE_manufacturer_not_fraudulent_l952_95225


namespace NUMINAMATH_CALUDE_triangular_sum_iff_squares_sum_l952_95245

/-- A triangular number is a positive integer of the form n * (n + 1) / 2 -/
def IsTriangular (k : ℕ) : Prop :=
  ∃ n : ℕ, k = n * (n + 1) / 2

/-- m is a sum of two triangular numbers -/
def IsSumOfTwoTriangular (m : ℕ) : Prop :=
  ∃ a b : ℕ, IsTriangular a ∧ IsTriangular b ∧ m = a + b

/-- n is a sum of two squares -/
def IsSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ x y : ℤ, n = x^2 + y^2

/-- Main theorem: m is a sum of two triangular numbers if and only if 4m + 1 is a sum of two squares -/
theorem triangular_sum_iff_squares_sum (m : ℕ) :
  IsSumOfTwoTriangular m ↔ IsSumOfTwoSquares (4 * m + 1) :=
sorry

end NUMINAMATH_CALUDE_triangular_sum_iff_squares_sum_l952_95245


namespace NUMINAMATH_CALUDE_prob_six_heads_and_return_l952_95291

/-- The number of nodes in the circular arrangement -/
def num_nodes : ℕ := 5

/-- The total number of coin flips -/
def num_flips : ℕ := 12

/-- The number of heads we're interested in -/
def target_heads : ℕ := 6

/-- Represents the movement on the circular arrangement -/
def net_movement (heads : ℕ) : ℤ :=
  (heads : ℤ) - (num_flips - heads : ℤ)

/-- The condition for returning to the starting node -/
def returns_to_start (heads : ℕ) : Prop :=
  net_movement heads % (num_nodes : ℤ) = 0

/-- The probability of flipping exactly 'heads' number of heads in 'num_flips' flips -/
def prob_heads (heads : ℕ) : ℚ :=
  (Nat.choose num_flips heads : ℚ) / 2^num_flips

/-- The main theorem to prove -/
theorem prob_six_heads_and_return :
  returns_to_start target_heads ∧ prob_heads target_heads = 231 / 1024 := by
  sorry


end NUMINAMATH_CALUDE_prob_six_heads_and_return_l952_95291


namespace NUMINAMATH_CALUDE_unique_prime_triple_l952_95298

theorem unique_prime_triple : 
  ∃! (x y z : ℕ), 
    (Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z) ∧ 
    (x > y ∧ y > z) ∧
    (Nat.Prime (x - y) ∧ Nat.Prime (y - z) ∧ Nat.Prime (x - z)) ∧
    (x = 7 ∧ y = 5 ∧ z = 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l952_95298


namespace NUMINAMATH_CALUDE_range_of_a_l952_95231

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) → 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * a - 1) ^ x₁ > (2 * a - 1) ^ x₂) → 
  1/2 < a ∧ a ≤ 2/3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l952_95231


namespace NUMINAMATH_CALUDE_dataset_transformation_l952_95263

/-- Represents a dataset with mean and variance -/
structure Dataset where
  mean : ℝ
  variance : ℝ

/-- Represents the transformation of adding a constant to each data point -/
def add_constant (d : Dataset) (c : ℝ) : Dataset :=
  { mean := d.mean + c,
    variance := d.variance }

theorem dataset_transformation (d : Dataset) :
  d.mean = 2.8 →
  d.variance = 3.6 →
  (add_constant d 60).mean = 62.8 ∧ (add_constant d 60).variance = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_dataset_transformation_l952_95263


namespace NUMINAMATH_CALUDE_value_of_expression_l952_95246

theorem value_of_expression : 8 + 2 * (3^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l952_95246


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_range_l952_95267

theorem empty_solution_set_iff_a_range (a : ℝ) :
  (∀ x : ℝ, |x - 1| + |x - 3| > a^2 - 2*a - 1) ↔ (-1 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_range_l952_95267


namespace NUMINAMATH_CALUDE_f_2023_equals_2_l952_95242

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2023_equals_2 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_symmetry : ∀ x, f (1 + x) = f (1 - x))
  (h_interval : ∀ x ∈ Set.Icc 0 1, f x = 2^x) :
  f 2023 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2023_equals_2_l952_95242


namespace NUMINAMATH_CALUDE_quadratic_decreasing_range_l952_95229

/-- Given a quadratic function y = (x-m)^2 - 1, if y decreases as x increases when x ≤ 3, then m ≥ 3 -/
theorem quadratic_decreasing_range (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≤ 3 ∧ x₂ ≤ 3 ∧ x₁ < x₂ → (x₁ - m)^2 - 1 > (x₂ - m)^2 - 1) →
  m ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_range_l952_95229


namespace NUMINAMATH_CALUDE_entertainment_percentage_l952_95232

def monthly_salary : ℝ := 5000
def food_percentage : ℝ := 40
def rent_percentage : ℝ := 20
def conveyance_percentage : ℝ := 10
def savings : ℝ := 1000

theorem entertainment_percentage :
  let total_known_expenses := food_percentage + rent_percentage + conveyance_percentage
  let remaining_percentage := 100 - total_known_expenses
  let expected_savings := (remaining_percentage / 100) * monthly_salary
  let entertainment_expense := expected_savings - savings
  entertainment_expense / monthly_salary * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_entertainment_percentage_l952_95232


namespace NUMINAMATH_CALUDE_age_sum_l952_95251

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 18 years old
  Prove that the sum of their ages is 47 years. -/
theorem age_sum (a b c : ℕ) : 
  b = 18 → 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 47 := by sorry

end NUMINAMATH_CALUDE_age_sum_l952_95251


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l952_95262

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2*I) = 5) : z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l952_95262


namespace NUMINAMATH_CALUDE_distance_P_to_xoy_is_3_l952_95295

/-- The distance from a point to the xOy plane in 3D Cartesian coordinates --/
def distance_to_xoy_plane (p : ℝ × ℝ × ℝ) : ℝ :=
  |p.2.2|

/-- The point P with coordinates (1, -2, 3) --/
def P : ℝ × ℝ × ℝ := (1, -2, 3)

/-- Theorem: The distance from point P(1,-2,3) to the xOy plane is 3 --/
theorem distance_P_to_xoy_is_3 : distance_to_xoy_plane P = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_xoy_is_3_l952_95295


namespace NUMINAMATH_CALUDE_tetrahedron_volume_relation_l952_95239

/-- A tetrahedron with volume V, face areas S_i, and distances H_i from an internal point to each face. -/
structure Tetrahedron where
  V : ℝ
  S : Fin 4 → ℝ
  H : Fin 4 → ℝ
  K : ℝ
  h_positive : V > 0
  S_positive : ∀ i, S i > 0
  H_positive : ∀ i, H i > 0
  K_positive : K > 0
  h_relation : ∀ i : Fin 4, S i / (i.val + 1 : ℝ) = K

theorem tetrahedron_volume_relation (t : Tetrahedron) :
  t.H 0 + 2 * t.H 1 + 3 * t.H 2 + 4 * t.H 3 = 3 * t.V / t.K := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_relation_l952_95239


namespace NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l952_95236

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 13) (h2 : b = 24) :
  (a + b + x = 78) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l952_95236


namespace NUMINAMATH_CALUDE_average_seashells_per_person_l952_95273

/-- The number of seashells found by Sally -/
def sally_shells : ℕ := 9

/-- The number of seashells found by Tom -/
def tom_shells : ℕ := 7

/-- The number of seashells found by Jessica -/
def jessica_shells : ℕ := 5

/-- The number of seashells found by Alex -/
def alex_shells : ℕ := 12

/-- The total number of people who found seashells -/
def total_people : ℕ := 4

/-- The average number of seashells found per person -/
def average_shells : ℚ := (sally_shells + tom_shells + jessica_shells + alex_shells : ℚ) / total_people

theorem average_seashells_per_person :
  average_shells = 33 / 4 :=
by sorry

end NUMINAMATH_CALUDE_average_seashells_per_person_l952_95273


namespace NUMINAMATH_CALUDE_percentage_defective_meters_l952_95202

def total_meters : ℕ := 4000
def rejected_meters : ℕ := 2

theorem percentage_defective_meters :
  (rejected_meters : ℝ) / total_meters * 100 = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_percentage_defective_meters_l952_95202


namespace NUMINAMATH_CALUDE_largest_integer_with_gcd_18_6_l952_95218

theorem largest_integer_with_gcd_18_6 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 150 → Nat.gcd m 18 = 6 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_gcd_18_6_l952_95218


namespace NUMINAMATH_CALUDE_hex_to_decimal_conversion_l952_95269

/-- Given that the hexadecimal number (3m502_(16)) is equal to 4934 in decimal,
    prove that m = 4. -/
theorem hex_to_decimal_conversion (m : ℕ) : 
  (3 * 16^4 + m * 16^3 + 5 * 16^2 + 0 * 16^1 + 2 * 16^0 = 4934) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_hex_to_decimal_conversion_l952_95269


namespace NUMINAMATH_CALUDE_andrew_kept_130_stickers_l952_95292

def andrew_stickers : ℕ := 750
def daniel_stickers : ℕ := 250
def fred_extra_stickers : ℕ := 120

def fred_stickers : ℕ := daniel_stickers + fred_extra_stickers
def shared_stickers : ℕ := daniel_stickers + fred_stickers
def andrew_kept_stickers : ℕ := andrew_stickers - shared_stickers

theorem andrew_kept_130_stickers : andrew_kept_stickers = 130 := by
  sorry

end NUMINAMATH_CALUDE_andrew_kept_130_stickers_l952_95292


namespace NUMINAMATH_CALUDE_difference_of_squares_72_48_l952_95253

theorem difference_of_squares_72_48 : 72^2 - 48^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_72_48_l952_95253


namespace NUMINAMATH_CALUDE_equation_satisfies_condition_l952_95213

theorem equation_satisfies_condition (x y z : ℤ) : 
  x = z ∧ y = x - 1 → x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfies_condition_l952_95213


namespace NUMINAMATH_CALUDE_function_bound_on_unit_interval_l952_95216

theorem function_bound_on_unit_interval 
  (f : Set.Icc 0 1 → ℝ)
  (h₁ : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h₂ : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₁ - f x₂| < |x₁.1 - x₂.1|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₁ - f x₂| < (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_function_bound_on_unit_interval_l952_95216


namespace NUMINAMATH_CALUDE_new_players_joined_new_players_joined_game_l952_95244

theorem new_players_joined (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let new_players := (total_lives - initial_players * lives_per_player) / lives_per_player
  new_players

theorem new_players_joined_game : new_players_joined 8 6 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_new_players_joined_new_players_joined_game_l952_95244


namespace NUMINAMATH_CALUDE_octahedron_ant_path_probability_l952_95299

/-- Represents a vertex in the octahedron --/
inductive Vertex
| Top
| Bottom
| Middle1
| Middle2
| Middle3
| Middle4

/-- Represents an octahedron --/
structure Octahedron where
  vertices : List Vertex
  edges : List (Vertex × Vertex)
  is_regular : Bool

/-- Represents the ant's path --/
structure AntPath where
  start : Vertex
  a : Vertex
  b : Vertex
  c : Vertex

/-- Function to check if a vertex is in the middle ring --/
def is_middle_ring (v : Vertex) : Bool :=
  match v with
  | Vertex.Middle1 | Vertex.Middle2 | Vertex.Middle3 | Vertex.Middle4 => true
  | _ => false

/-- Function to get adjacent vertices --/
def get_adjacent_vertices (o : Octahedron) (v : Vertex) : List Vertex :=
  sorry

/-- Function to calculate the probability of returning to the start --/
def return_probability (o : Octahedron) (path : AntPath) : Rat :=
  sorry

theorem octahedron_ant_path_probability (o : Octahedron) (path : AntPath) :
  o.is_regular = true →
  is_middle_ring path.start = true →
  path.a ∈ get_adjacent_vertices o path.start →
  path.b ∈ get_adjacent_vertices o path.a →
  path.c ∈ get_adjacent_vertices o path.b →
  return_probability o path = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_octahedron_ant_path_probability_l952_95299


namespace NUMINAMATH_CALUDE_coefficient_d_nonzero_l952_95247

/-- A polynomial of degree 5 with specific properties -/
def Q (a b c d f : ℝ) (x : ℝ) : ℝ :=
  x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + f

/-- Theorem: For a polynomial Q with five distinct x-intercepts, including (0,0) and (1,0), 
    the coefficient d must be non-zero -/
theorem coefficient_d_nonzero 
  (a b c d f : ℝ) 
  (h1 : Q a b c d f 0 = 0)
  (h2 : Q a b c d f 1 = 0)
  (h3 : ∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ 
       p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ p ≠ 1 ∧ q ≠ 1 ∧ r ≠ 1 ∧
       ∀ x : ℝ, Q a b c d f x = x * (x - 1) * (x - p) * (x - q) * (x - r)) : 
  d ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_coefficient_d_nonzero_l952_95247


namespace NUMINAMATH_CALUDE_triangle_theorem_l952_95277

/-- Given a triangle ABC with sides a, b, c, inradius r, and exradii r₁, r₂, r₃ opposite vertices A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ

/-- Conditions for the triangle -/
def ValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b ∧
  t.a > t.r₁ ∧ t.b > t.r₂ ∧ t.c > t.r₃

/-- Definition of an acute triangle -/
def IsAcute (t : Triangle) : Prop :=
  t.a^2 + t.b^2 > t.c^2 ∧ t.b^2 + t.c^2 > t.a^2 ∧ t.c^2 + t.a^2 > t.b^2

/-- The main theorem to be proved -/
theorem triangle_theorem (t : Triangle) (h : ValidTriangle t) :
  IsAcute t ∧ t.a + t.b + t.c > t.r + t.r₁ + t.r₂ + t.r₃ := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l952_95277


namespace NUMINAMATH_CALUDE_seating_theorem_l952_95205

/-- The number of ways to seat n people around a round table -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange a pair of people -/
def pairArrangements : ℕ := 2

/-- The number of ways to seat 10 people around a round table with two specific people next to each other -/
def seatingArrangements : ℕ := roundTableArrangements 9 * pairArrangements

theorem seating_theorem : seatingArrangements = 80640 := by sorry

end NUMINAMATH_CALUDE_seating_theorem_l952_95205


namespace NUMINAMATH_CALUDE_smallest_divisible_term_l952_95207

/-- An integer sequence satisfying the given recurrence relation -/
def IntegerSequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → (n - 1) * a (n + 1) = (n + 1) * a n - 2 * (n - 1)

/-- The property that 2008 divides a_2007 -/
def DivisibilityCondition (a : ℕ → ℤ) : Prop :=
  2008 ∣ a 2007

/-- The main theorem statement -/
theorem smallest_divisible_term
  (a : ℕ → ℤ)
  (h_seq : IntegerSequence a)
  (h_div : DivisibilityCondition a) :
  (∀ n : ℕ, 2 ≤ n ∧ n < 501 → ¬(2008 ∣ a n)) ∧
  (2008 ∣ a 501) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_term_l952_95207


namespace NUMINAMATH_CALUDE_rectangle_area_l952_95266

theorem rectangle_area (x y : ℝ) (h1 : y = (7/3) * x) (h2 : 2 * (x + y) = 40) : x * y = 84 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l952_95266


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l952_95254

theorem ice_cream_flavors (n k : ℕ) (hn : n = 4) (hk : k = 4) :
  (n + k - 1).choose (k - 1) = 35 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l952_95254


namespace NUMINAMATH_CALUDE_sum_of_odds_is_even_product_zero_implies_factor_zero_exists_even_prime_l952_95230

-- Definition of odd number
def isOdd (n : Int) : Prop := ∃ k : Int, n = 2 * k + 1

-- Statement 1
theorem sum_of_odds_is_even (x y : Int) (hx : isOdd x) (hy : isOdd y) : 
  ∃ k : Int, x + y = 2 * k := by sorry

-- Statement 2
theorem product_zero_implies_factor_zero (x y : ℝ) (h : x * y = 0) : 
  x = 0 ∨ y = 0 := by sorry

-- Definition of prime number
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

-- Statement 3
theorem exists_even_prime : ∃ p : Nat, isPrime p ∧ ¬isOdd p := by sorry

end NUMINAMATH_CALUDE_sum_of_odds_is_even_product_zero_implies_factor_zero_exists_even_prime_l952_95230


namespace NUMINAMATH_CALUDE_student_age_l952_95286

theorem student_age (student_age man_age : ℕ) : 
  man_age = student_age + 26 →
  man_age + 2 = 2 * (student_age + 2) →
  student_age = 24 := by
sorry

end NUMINAMATH_CALUDE_student_age_l952_95286


namespace NUMINAMATH_CALUDE_cube_product_three_six_l952_95223

theorem cube_product_three_six : 3^3 * 6^3 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_three_six_l952_95223


namespace NUMINAMATH_CALUDE_special_polynomial_max_value_l952_95283

/-- A polynomial with real coefficients satisfying the given condition -/
def SpecialPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, P t = P 1 * t^2 + P (P 1) * t + P (P (P 1))

/-- The theorem stating the maximum value of P(P(P(P(1)))) -/
theorem special_polynomial_max_value (P : ℝ → ℝ) (h : SpecialPolynomial P) :
    ∃ M : ℝ, M = (1 : ℝ) / 9 ∧ P (P (P (P 1))) ≤ M ∧ 
    ∃ P₀ : ℝ → ℝ, SpecialPolynomial P₀ ∧ P₀ (P₀ (P₀ (P₀ 1))) = M :=
by sorry

end NUMINAMATH_CALUDE_special_polynomial_max_value_l952_95283


namespace NUMINAMATH_CALUDE_block_stacks_ratio_l952_95203

theorem block_stacks_ratio : 
  ∀ (stack1 stack2 stack3 stack4 stack5 : ℕ),
  stack1 = 7 →
  stack2 = stack1 + 3 →
  stack3 = stack2 - 6 →
  stack4 = stack3 + 10 →
  stack1 + stack2 + stack3 + stack4 + stack5 = 55 →
  stack5 / stack2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_block_stacks_ratio_l952_95203


namespace NUMINAMATH_CALUDE_probability_after_removal_l952_95224

theorem probability_after_removal (total : ℕ) (blue : ℕ) (removed : ℕ) 
  (h1 : total = 25)
  (h2 : blue = 9)
  (h3 : removed = 5)
  (h4 : removed < blue)
  (h5 : removed < total) :
  (blue - removed : ℚ) / (total - removed) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_after_removal_l952_95224


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l952_95256

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l952_95256
