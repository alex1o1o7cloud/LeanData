import Mathlib

namespace NUMINAMATH_CALUDE_linen_tablecloth_cost_is_25_l3221_322137

/-- Represents the cost structure for wedding reception decorations --/
structure WeddingDecorations where
  num_tables : ℕ
  place_settings_per_table : ℕ
  place_setting_cost : ℕ
  roses_per_centerpiece : ℕ
  lilies_per_centerpiece : ℕ
  rose_cost : ℕ
  lily_cost : ℕ
  total_decoration_cost : ℕ

/-- Calculates the cost of a single linen tablecloth --/
def linen_tablecloth_cost (d : WeddingDecorations) : ℕ :=
  let place_settings_cost := d.num_tables * d.place_settings_per_table * d.place_setting_cost
  let centerpiece_cost := d.num_tables * (d.roses_per_centerpiece * d.rose_cost + d.lilies_per_centerpiece * d.lily_cost)
  let tablecloth_total_cost := d.total_decoration_cost - (place_settings_cost + centerpiece_cost)
  tablecloth_total_cost / d.num_tables

/-- Theorem stating that the cost of a single linen tablecloth is $25 --/
theorem linen_tablecloth_cost_is_25 (d : WeddingDecorations)
  (h1 : d.num_tables = 20)
  (h2 : d.place_settings_per_table = 4)
  (h3 : d.place_setting_cost = 10)
  (h4 : d.roses_per_centerpiece = 10)
  (h5 : d.lilies_per_centerpiece = 15)
  (h6 : d.rose_cost = 5)
  (h7 : d.lily_cost = 4)
  (h8 : d.total_decoration_cost = 3500) :
  linen_tablecloth_cost d = 25 := by
  sorry

end NUMINAMATH_CALUDE_linen_tablecloth_cost_is_25_l3221_322137


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3221_322170

theorem fraction_sum_equality (p q r s : ℝ) 
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) + s / (40 - s) = 9) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) + 8 / (40 - s) = 7.6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3221_322170


namespace NUMINAMATH_CALUDE_a_formula_l3221_322117

noncomputable section

/-- The function f(x) = x / sqrt(1 + x^2) -/
def f (x : ℝ) : ℝ := x / Real.sqrt (1 + x^2)

/-- The sequence a_n defined recursively -/
def a (x : ℝ) : ℕ → ℝ
  | 0 => f x
  | n + 1 => f (a x n)

/-- The theorem stating the general formula for a_n -/
theorem a_formula (x : ℝ) (h : x > 0) (n : ℕ) :
  a x n = x / Real.sqrt (1 + n * x^2) := by
  sorry

end

end NUMINAMATH_CALUDE_a_formula_l3221_322117


namespace NUMINAMATH_CALUDE_max_brownies_144_l3221_322102

/-- Represents the dimensions of a rectangular pan -/
structure PanDimensions where
  m : ℕ
  n : ℕ

/-- Calculates the number of interior pieces in the pan -/
def interiorPieces (d : PanDimensions) : ℕ := (d.m - 2) * (d.n - 2)

/-- Calculates the number of perimeter pieces in the pan -/
def perimeterPieces (d : PanDimensions) : ℕ := 2 * d.m + 2 * d.n - 4

/-- Represents the condition that interior pieces are twice the perimeter pieces -/
def interiorTwicePerimeter (d : PanDimensions) : Prop :=
  interiorPieces d = 2 * perimeterPieces d

/-- The theorem stating that the maximum number of brownies is 144 -/
theorem max_brownies_144 :
  ∃ (d : PanDimensions), interiorTwicePerimeter d ∧
  (∀ (d' : PanDimensions), interiorTwicePerimeter d' → d.m * d.n ≥ d'.m * d'.n) ∧
  d.m * d.n = 144 := by
  sorry

end NUMINAMATH_CALUDE_max_brownies_144_l3221_322102


namespace NUMINAMATH_CALUDE_num_divisors_2310_l3221_322139

/-- The number of positive divisors of a positive integer n -/
def numPositiveDivisors (n : ℕ+) : ℕ := sorry

/-- 2310 as a positive integer -/
def n : ℕ+ := 2310

/-- Theorem: The number of positive divisors of 2310 is 32 -/
theorem num_divisors_2310 : numPositiveDivisors n = 32 := by sorry

end NUMINAMATH_CALUDE_num_divisors_2310_l3221_322139


namespace NUMINAMATH_CALUDE_marble_247_is_white_l3221_322134

/-- Represents the color of a marble -/
inductive MarbleColor
| Gray
| White
| Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 12 with
  | 0 | 1 | 2 | 3 => MarbleColor.Gray
  | 4 | 5 | 6 | 7 | 8 => MarbleColor.White
  | _ => MarbleColor.Black

/-- Theorem stating that the 247th marble is white -/
theorem marble_247_is_white : marbleColor 247 = MarbleColor.White := by
  sorry


end NUMINAMATH_CALUDE_marble_247_is_white_l3221_322134


namespace NUMINAMATH_CALUDE_shaded_area_of_circumscribed_circles_shaded_area_equals_135π_l3221_322105

/-- The area of the shaded region between a circle circumscribing two externally tangent circles with radii 3 and 5 -/
theorem shaded_area_of_circumscribed_circles (π : ℝ) : ℝ := by
  -- Define the radii of the two smaller circles
  let r₁ : ℝ := 3
  let r₂ : ℝ := 5

  -- Define the radius of the larger circumscribing circle
  let R : ℝ := r₁ + r₂ + r₂

  -- Define the areas of the circles
  let A₁ : ℝ := π * r₁^2
  let A₂ : ℝ := π * r₂^2
  let A_large : ℝ := π * R^2

  -- Define the shaded area
  let shaded_area : ℝ := A_large - A₁ - A₂

  -- Prove that the shaded area equals 135π
  sorry

/-- The main theorem stating that the shaded area is equal to 135π -/
theorem shaded_area_equals_135π (π : ℝ) : shaded_area_of_circumscribed_circles π = 135 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_circumscribed_circles_shaded_area_equals_135π_l3221_322105


namespace NUMINAMATH_CALUDE_original_price_after_discount_l3221_322124

theorem original_price_after_discount (P : ℝ) : 
  P * (1 - 0.2) = P - 50 → P = 250 := by
  sorry

end NUMINAMATH_CALUDE_original_price_after_discount_l3221_322124


namespace NUMINAMATH_CALUDE_percentage_less_l3221_322146

theorem percentage_less (x y : ℝ) (h : x = 5 * y) : (x - y) / x * 100 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_less_l3221_322146


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3221_322107

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (n : ℕ) :
  a 1 = 1 →
  a 2 = 4 →
  (∀ k ≥ 2, 2 * a k = a (k - 1) + a (k + 1)) →
  a n = 301 →
  n = 101 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3221_322107


namespace NUMINAMATH_CALUDE_absolute_sum_of_roots_greater_than_four_l3221_322185

theorem absolute_sum_of_roots_greater_than_four 
  (p : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂) 
  (h2 : x₁^2 + p*x₁ + 4 = 0) 
  (h3 : x₂^2 + p*x₂ + 4 = 0) : 
  |x₁ + x₂| > 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_sum_of_roots_greater_than_four_l3221_322185


namespace NUMINAMATH_CALUDE_circle_equation_tangent_lines_l3221_322112

-- Define the circle C
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the point M
def M : ℝ × ℝ := (0, 2)

-- Define the point P
def P : ℝ × ℝ := (3, 2)

-- Theorem for the equation of circle C
theorem circle_equation : 
  ∃ (r : ℝ), M ∈ Circle r ∧ Circle r = {p : ℝ × ℝ | p.1^2 + p.2^2 = 4} :=
sorry

-- Define a tangent line
def TangentLine (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | k * p.1 - p.2 - 3 * k + 2 = 0}

-- Theorem for the equations of tangent lines
theorem tangent_lines :
  ∃ (k₁ k₂ : ℝ), 
    (TangentLine k₁ = {p : ℝ × ℝ | p.2 = 2}) ∧
    (TangentLine k₂ = {p : ℝ × ℝ | 12 * p.1 - 5 * p.2 - 26 = 0}) ∧
    P ∈ TangentLine k₁ ∧ P ∈ TangentLine k₂ ∧
    (∀ (p : ℝ × ℝ), p ∈ TangentLine k₁ ∩ Circle 2 → p = P) ∧
    (∀ (p : ℝ × ℝ), p ∈ TangentLine k₂ ∩ Circle 2 → p = P) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_lines_l3221_322112


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3221_322108

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≥ 1 ∧ y ≥ 2 → x + y ≥ 3) ∧
  ∃ x y, x + y ≥ 3 ∧ ¬(x ≥ 1 ∧ y ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3221_322108


namespace NUMINAMATH_CALUDE_max_m_and_min_sum_l3221_322160

theorem max_m_and_min_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ m : ℝ, (3 / a + 1 / b ≥ m / (a + 3 * b)) → m ≤ 12) ∧
  (a + 2 * b + 2 * a * b = 8 → a + 2 * b ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_max_m_and_min_sum_l3221_322160


namespace NUMINAMATH_CALUDE_rectangle_perimeter_is_46_l3221_322135

/-- A rectangle dissection puzzle with seven squares -/
structure RectangleDissection where
  b₁ : ℕ
  b₂ : ℕ
  b₃ : ℕ
  b₄ : ℕ
  b₅ : ℕ
  b₆ : ℕ
  b₇ : ℕ
  rel₁ : b₁ + b₂ = b₃
  rel₂ : b₁ + b₃ = b₄
  rel₃ : b₃ + b₄ = b₅
  rel₄ : b₄ + b₅ = b₆
  rel₅ : b₂ + b₅ = b₇
  b₁_eq_one : b₁ = 1
  b₂_eq_two : b₂ = 2

/-- The perimeter of the rectangle in the dissection puzzle -/
def perimeter (r : RectangleDissection) : ℕ :=
  2 * (r.b₆ + r.b₇)

/-- Theorem stating that the perimeter of the rectangle is 46 -/
theorem rectangle_perimeter_is_46 (r : RectangleDissection) : perimeter r = 46 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_is_46_l3221_322135


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3221_322136

/-- Atomic weight in atomic mass units (amu) -/
def atomic_weight (element : String) : Float :=
  match element with
  | "N" => 14.01
  | "H" => 1.01
  | "Br" => 79.90
  | "O" => 16.00
  | "C" => 12.01
  | _ => 0  -- Default case for unknown elements

/-- Number of atoms for each element in the compound -/
def atom_count (element : String) : Nat :=
  match element with
  | "N" => 2
  | "H" => 6
  | "Br" => 1
  | "O" => 1
  | "C" => 3
  | _ => 0  -- Default case for elements not in the compound

/-- Calculate the molecular weight of the compound -/
def molecular_weight : Float :=
  ["N", "H", "Br", "O", "C"].map (fun e => (atomic_weight e) * (atom_count e).toFloat)
    |> List.sum

/-- Theorem: The molecular weight of the compound is approximately 166.01 amu -/
theorem compound_molecular_weight :
  (molecular_weight - 166.01).abs < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3221_322136


namespace NUMINAMATH_CALUDE_cost_of_five_cds_l3221_322114

/-- The cost of a certain number of identical CDs -/
def cost_of_cds (n : ℕ) : ℚ :=
  28 * (n / 2 : ℚ)

/-- Theorem stating that the cost of five CDs is 70 dollars -/
theorem cost_of_five_cds : cost_of_cds 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_five_cds_l3221_322114


namespace NUMINAMATH_CALUDE_slower_train_speed_theorem_l3221_322158

/-- The speed of the faster train in km/hr -/
def faster_train_speed : ℝ := 46

/-- The time taken for the faster train to completely pass the slower train in seconds -/
def passing_time : ℝ := 72

/-- The length of each train in meters -/
def train_length : ℝ := 100

/-- The speed of the slower train in km/hr -/
def slower_train_speed : ℝ := 36

theorem slower_train_speed_theorem :
  ∃ (v : ℝ), v = slower_train_speed ∧
  (2 * train_length) = (faster_train_speed - v) * (passing_time / 3600) * 1000 := by
  sorry

end NUMINAMATH_CALUDE_slower_train_speed_theorem_l3221_322158


namespace NUMINAMATH_CALUDE_abhinav_bhupathi_total_money_l3221_322189

/-- The problem of calculating the total amount of money Abhinav and Bhupathi have together. -/
theorem abhinav_bhupathi_total_money (abhinav_amount bhupathi_amount : ℚ) : 
  (4 : ℚ) / 15 * abhinav_amount = (2 : ℚ) / 5 * bhupathi_amount →
  bhupathi_amount = 484 →
  abhinav_amount + bhupathi_amount = 1210 := by
  sorry

#check abhinav_bhupathi_total_money

end NUMINAMATH_CALUDE_abhinav_bhupathi_total_money_l3221_322189


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l3221_322116

theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 30 →
  offset1 = 9 →
  area = 225 →
  ∃ offset2 : ℝ, 
    offset2 = 6 ∧ 
    area = (1/2) * diagonal * offset1 + (1/2) * diagonal * offset2 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_offset_l3221_322116


namespace NUMINAMATH_CALUDE_parabola_directrix_parameter_l3221_322168

/-- 
For a parabola y = ax^2 with directrix y = 1, the value of a is -1/4.
-/
theorem parabola_directrix_parameter (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Condition 1: Parabola equation
  (∃ y : ℝ, y = 1 ∧ ∀ x : ℝ, y = 1 → (x, y) ∉ {(x, y) | y = a * x^2}) →  -- Condition 2: Directrix equation
  a = -1/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_parameter_l3221_322168


namespace NUMINAMATH_CALUDE_largest_ball_radius_largest_ball_touches_plane_largest_ball_on_z_axis_l3221_322164

/-- Represents a torus formed by revolving a circle about the z-axis. -/
structure Torus where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents a spherical ball. -/
structure Ball where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The largest ball that can be positioned on top of the torus. -/
def largest_ball (t : Torus) : Ball :=
  { center := (0, 0, 4),
    radius := 4 }

/-- Theorem stating that the largest ball has radius 4. -/
theorem largest_ball_radius (t : Torus) :
  (t.center = (4, 0, 1) ∧ t.radius = 1) →
  (largest_ball t).radius = 4 := by
  sorry

/-- Theorem stating that the largest ball touches the horizontal plane. -/
theorem largest_ball_touches_plane (t : Torus) :
  (t.center = (4, 0, 1) ∧ t.radius = 1) →
  (largest_ball t).center.2.1 = (largest_ball t).radius := by
  sorry

/-- Theorem stating that the largest ball is centered on the z-axis. -/
theorem largest_ball_on_z_axis (t : Torus) :
  (t.center = (4, 0, 1) ∧ t.radius = 1) →
  (largest_ball t).center.1 = 0 ∧ (largest_ball t).center.2.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_ball_radius_largest_ball_touches_plane_largest_ball_on_z_axis_l3221_322164


namespace NUMINAMATH_CALUDE_bingley_has_six_bracelets_l3221_322122

/-- The number of bracelets Bingley has remaining after exchanges with Kelly and his sister -/
def bingleysRemainingBracelets (bingleyInitial : ℕ) (kellyInitial : ℕ) : ℕ :=
  let bingleyAfterKelly := bingleyInitial + kellyInitial / 4
  bingleyAfterKelly - bingleyAfterKelly / 3

/-- Theorem stating that Bingley has 6 bracelets remaining -/
theorem bingley_has_six_bracelets :
  bingleysRemainingBracelets 5 16 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bingley_has_six_bracelets_l3221_322122


namespace NUMINAMATH_CALUDE_floor_times_self_equals_54_l3221_322109

theorem floor_times_self_equals_54 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 54 ∧ x = 54 / 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_equals_54_l3221_322109


namespace NUMINAMATH_CALUDE_pet_ownership_percentage_l3221_322191

theorem pet_ownership_percentage (total_students : ℕ) (cat_owners : ℕ) (dog_owners : ℕ) (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 150)
  (h3 : dog_owners = 100)
  (h4 : both_owners = 40) :
  (cat_owners + dog_owners - both_owners) / total_students = 42 / 100 := by
  sorry

end NUMINAMATH_CALUDE_pet_ownership_percentage_l3221_322191


namespace NUMINAMATH_CALUDE_infinite_divisors_of_power_plus_one_l3221_322154

theorem infinite_divisors_of_power_plus_one (a : ℕ) (h1 : a > 1) (h2 : Even a) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, n ∣ a^n + 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_divisors_of_power_plus_one_l3221_322154


namespace NUMINAMATH_CALUDE_hyperbola_quadrilateral_area_ratio_max_l3221_322149

theorem hyperbola_quadrilateral_area_ratio_max (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x : ℝ), x = a * b / (a^2 + b^2) ∧ ∀ (y : ℝ), y = a * b / (a^2 + b^2) → x ≥ y) →
  a * b / (a^2 + b^2) ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_quadrilateral_area_ratio_max_l3221_322149


namespace NUMINAMATH_CALUDE_smallest_number_negative_l3221_322156

theorem smallest_number_negative (a : ℝ) :
  (∀ x : ℝ, min (2^(x-1) - 3^(4-x) + a) (a + 5 - x^3 - 2*x) < 0) ↔ a < -7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_negative_l3221_322156


namespace NUMINAMATH_CALUDE_greatest_x_value_l3221_322167

theorem greatest_x_value (x : ℝ) : 
  x ≠ 6 → x ≠ -3 → (x^2 - x - 30) / (x - 6) = 5 / (x + 3) → 
  x ≤ -2 ∧ ∃ y, y = -2 ∧ (y^2 - y - 30) / (y - 6) = 5 / (y + 3) := by
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3221_322167


namespace NUMINAMATH_CALUDE_subset_condition_l3221_322141

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) / (x - 4) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 5) > 0}

-- State the theorem
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 4 ≤ a ∧ a < 5 := by sorry

end NUMINAMATH_CALUDE_subset_condition_l3221_322141


namespace NUMINAMATH_CALUDE_limit_ratio_sevens_to_total_l3221_322111

/-- Count of digit 7 occurrences in decimal representation of numbers from 1 to n -/
def count_sevens (n : ℕ) : ℕ := sorry

/-- Total count of digits in decimal representation of numbers from 1 to n -/
def total_digits (n : ℕ) : ℕ := sorry

/-- The theorem stating that the limit of the ratio of 7's to total digits is 1/10 -/
theorem limit_ratio_sevens_to_total (ε : ℝ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ n ≥ N, |((count_sevens n : ℝ) / (total_digits n : ℝ)) - (1 / 10)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_ratio_sevens_to_total_l3221_322111


namespace NUMINAMATH_CALUDE_yellow_red_ball_arrangements_l3221_322195

theorem yellow_red_ball_arrangements :
  let total_balls : ℕ := 7
  let yellow_balls : ℕ := 4
  let red_balls : ℕ := 3
  Nat.choose total_balls yellow_balls = 35 := by sorry

end NUMINAMATH_CALUDE_yellow_red_ball_arrangements_l3221_322195


namespace NUMINAMATH_CALUDE_first_hour_premium_l3221_322118

/-- A psychologist charges different rates for the first hour and additional hours of therapy. -/
structure TherapyRates where
  /-- The charge for the first hour of therapy -/
  first_hour : ℝ
  /-- The charge for each additional hour of therapy -/
  additional_hour : ℝ
  /-- The total charge for 5 hours of therapy is $375 -/
  five_hour_total : first_hour + 4 * additional_hour = 375
  /-- The total charge for 2 hours of therapy is $174 -/
  two_hour_total : first_hour + additional_hour = 174

/-- The difference between the first hour charge and additional hour charge is $40 -/
theorem first_hour_premium (rates : TherapyRates) : 
  rates.first_hour - rates.additional_hour = 40 := by
  sorry

end NUMINAMATH_CALUDE_first_hour_premium_l3221_322118


namespace NUMINAMATH_CALUDE_savings_after_expense_l3221_322120

def weekly_savings (n : ℕ) : ℕ := 20 + 10 * n

def total_savings (weeks : ℕ) : ℕ :=
  (List.range weeks).map weekly_savings |>.sum

theorem savings_after_expense (weeks : ℕ) (expense : ℕ) : 
  weeks = 4 → expense = 75 → total_savings weeks - expense = 65 := by
  sorry

end NUMINAMATH_CALUDE_savings_after_expense_l3221_322120


namespace NUMINAMATH_CALUDE_sum_interior_angles_octagon_l3221_322123

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The sum of the interior angles of an octagon is 1080° -/
theorem sum_interior_angles_octagon :
  sum_interior_angles octagon_sides = 1080 := by
  sorry


end NUMINAMATH_CALUDE_sum_interior_angles_octagon_l3221_322123


namespace NUMINAMATH_CALUDE_compound_proposition_1_compound_proposition_2_compound_proposition_3_l3221_322165

-- Define the propositions
def smallest_angle_not_greater_than_60 (α : Real) : Prop :=
  (∀ β γ : Real, α + β + γ = 180 → α ≤ β ∧ α ≤ γ) → α ≤ 60

def isosceles_right_triangle (α β γ : Real) : Prop :=
  α + β + γ = 180 ∧ α = 90 ∧ β = 45 ∧ (γ = α ∨ γ = β) ∧ α = 90

def triangle_with_60_degree (α β γ : Real) : Prop :=
  α + β + γ = 180 ∧ (α = 60 ∨ β = 60 ∨ γ = 60)

-- Theorem statements
theorem compound_proposition_1 (α : Real) :
  smallest_angle_not_greater_than_60 α ↔ 
  ¬(∀ β γ : Real, α + β + γ = 180 → α ≤ β ∧ α ≤ γ → α > 60) :=
sorry

theorem compound_proposition_2 (α β γ : Real) :
  isosceles_right_triangle α β γ ↔
  (α + β + γ = 180 ∧ α = 90 ∧ β = 45 ∧ (γ = α ∨ γ = β)) ∧
  (α + β + γ = 180 ∧ α = 90 ∧ β = 45) :=
sorry

theorem compound_proposition_3 (α β γ : Real) :
  triangle_with_60_degree α β γ ↔
  (α + β + γ = 180 ∧ α = 60 ∧ β = 60 ∧ γ = 60) ∨
  (α + β + γ = 180 ∧ (α = 60 ∨ β = 60 ∨ γ = 60) ∧ (α = 90 ∨ β = 90 ∨ γ = 90)) :=
sorry

end NUMINAMATH_CALUDE_compound_proposition_1_compound_proposition_2_compound_proposition_3_l3221_322165


namespace NUMINAMATH_CALUDE_factorization_of_4a_squared_minus_1_l3221_322157

theorem factorization_of_4a_squared_minus_1 (a : ℝ) : 4 * a^2 - 1 = (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4a_squared_minus_1_l3221_322157


namespace NUMINAMATH_CALUDE_max_value_on_sphere_l3221_322132

theorem max_value_on_sphere (x y z : ℝ) (h : x^2 + y^2 + 4*z^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 6 ∧ ∀ (a b c : ℝ), a^2 + b^2 + 4*c^2 = 1 → a + b + 4*c ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_sphere_l3221_322132


namespace NUMINAMATH_CALUDE_paint_remaining_l3221_322110

theorem paint_remaining (num_statues : ℕ) (paint_per_statue : ℚ) (h1 : num_statues = 3) (h2 : paint_per_statue = 1/6) : 
  num_statues * paint_per_statue = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_l3221_322110


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3221_322144

theorem quadratic_roots_product (x : ℝ) : 
  (x - 4) * (2 * x + 10) = x^2 - 15 * x + 56 → 
  ∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ (c / a) + 6 = -90 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3221_322144


namespace NUMINAMATH_CALUDE_motorcycle_wheels_l3221_322179

theorem motorcycle_wheels (total_wheels : ℕ) (num_cars : ℕ) (num_motorcycles : ℕ) 
  (wheels_per_car : ℕ) (h1 : total_wheels = 117) (h2 : num_cars = 19) 
  (h3 : num_motorcycles = 11) (h4 : wheels_per_car = 5) :
  (total_wheels - num_cars * wheels_per_car) / num_motorcycles = 2 :=
by sorry

end NUMINAMATH_CALUDE_motorcycle_wheels_l3221_322179


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3221_322176

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^3 + y^3 = 85/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3221_322176


namespace NUMINAMATH_CALUDE_dusty_paid_hundred_l3221_322145

/-- Represents the cost and quantity of cake slices --/
structure CakeOrder where
  single_layer_cost : ℕ
  double_layer_cost : ℕ
  single_layer_quantity : ℕ
  double_layer_quantity : ℕ

/-- Calculates the total cost of the cake order --/
def total_cost (order : CakeOrder) : ℕ :=
  order.single_layer_cost * order.single_layer_quantity +
  order.double_layer_cost * order.double_layer_quantity

/-- Represents Dusty's cake purchase and change received --/
structure DustysPurchase where
  order : CakeOrder
  change_received : ℕ

/-- Theorem: Given Dusty's cake purchase and change received, prove that he paid $100 --/
theorem dusty_paid_hundred (purchase : DustysPurchase)
  (h1 : purchase.order.single_layer_cost = 4)
  (h2 : purchase.order.double_layer_cost = 7)
  (h3 : purchase.order.single_layer_quantity = 7)
  (h4 : purchase.order.double_layer_quantity = 5)
  (h5 : purchase.change_received = 37) :
  total_cost purchase.order + purchase.change_received = 100 := by
  sorry


end NUMINAMATH_CALUDE_dusty_paid_hundred_l3221_322145


namespace NUMINAMATH_CALUDE_palindrome_decomposition_l3221_322172

/-- A word is a list of characters -/
def Word := List Char

/-- A palindrome is a word that reads the same forward and backward -/
def isPalindrome (w : Word) : Prop :=
  w = w.reverse

/-- X is a word of length 2014 consisting of only 'A' and 'B' -/
def X : Word :=
  List.replicate 2014 'A'  -- Example word, actual content doesn't matter for the theorem

/-- Theorem: There exist at least 806 palindromes whose concatenation forms X -/
theorem palindrome_decomposition :
  ∃ (palindromes : List Word),
    palindromes.length ≥ 806 ∧
    (∀ p ∈ palindromes, isPalindrome p) ∧
    palindromes.join = X :=
  sorry


end NUMINAMATH_CALUDE_palindrome_decomposition_l3221_322172


namespace NUMINAMATH_CALUDE_evaluate_g_l3221_322183

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

-- State the theorem
theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l3221_322183


namespace NUMINAMATH_CALUDE_acceleration_at_two_seconds_l3221_322190

-- Define the distance function
def s (t : ℝ) : ℝ := 2 * t^3 - 5 * t^2 + 2

-- Define the velocity function as the derivative of the distance function
def v (t : ℝ) : ℝ := 6 * t^2 - 10 * t

-- Define the acceleration function as the derivative of the velocity function
def a (t : ℝ) : ℝ := 12 * t - 10

-- Theorem: The acceleration at t = 2 seconds is 14 m/s²
theorem acceleration_at_two_seconds : a 2 = 14 := by sorry

end NUMINAMATH_CALUDE_acceleration_at_two_seconds_l3221_322190


namespace NUMINAMATH_CALUDE_token_count_after_removal_l3221_322106

/-- Represents a token on the board -/
inductive Token
| White
| Black
| Empty

/-- Represents the board state -/
def Board (n : ℕ) := Fin (2*n) → Fin (2*n) → Token

/-- Counts the number of tokens of a specific type on the board -/
def countTokens (b : Board n) (t : Token) : ℕ := sorry

/-- Performs the token removal process -/
def removeTokens (b : Board n) : Board n := sorry

theorem token_count_after_removal (n : ℕ) (initial_board : Board n) :
  let final_board := removeTokens initial_board
  (countTokens final_board Token.Black ≤ n^2) ∧ 
  (countTokens final_board Token.White ≤ n^2) := by
  sorry

end NUMINAMATH_CALUDE_token_count_after_removal_l3221_322106


namespace NUMINAMATH_CALUDE_number_problem_l3221_322177

theorem number_problem (x : ℝ) : (0.5 * x - 10 = 25) → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3221_322177


namespace NUMINAMATH_CALUDE_triangle_side_length_l3221_322142

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  C = 2 * A ∧
  Real.cos A = 3/4 ∧
  a * c * Real.cos B = 27/2 →
  b = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3221_322142


namespace NUMINAMATH_CALUDE_max_distance_circle_centers_l3221_322180

/-- The maximum distance between the centers of two circles with 8-inch diameters
    placed within a 16-inch by 20-inch rectangle is 4√13 inches. -/
theorem max_distance_circle_centers (rect_width rect_height circle_diameter : ℝ)
  (hw : rect_width = 20)
  (hh : rect_height = 16)
  (hd : circle_diameter = 8)
  (h_nonneg : rect_width > 0 ∧ rect_height > 0 ∧ circle_diameter > 0) :
  Real.sqrt ((rect_width - circle_diameter) ^ 2 + (rect_height - circle_diameter) ^ 2) = 4 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_centers_l3221_322180


namespace NUMINAMATH_CALUDE_max_ab_value_l3221_322115

theorem max_ab_value (a b : ℝ) (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) (h3 : 3 ≤ a + b) (h4 : a + b ≤ 4) :
  ∃ (m : ℝ), m = 15/4 ∧ ab ≤ m ∧ ∃ (a' b' : ℝ), 1 ≤ a' - b' ∧ a' - b' ≤ 2 ∧ 3 ≤ a' + b' ∧ a' + b' ≤ 4 ∧ a' * b' = m :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l3221_322115


namespace NUMINAMATH_CALUDE_increasing_interval_of_sine_function_l3221_322129

open Real

theorem increasing_interval_of_sine_function 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (ω : ℝ) :
  (ω > 0) →
  (∀ x, f x = 2 * sin (ω * x + π / 4)) →
  (∀ x, g x = 2 * cos (2 * x - π / 4)) →
  (∀ x, f (x + π / ω) = f x) →
  (∀ x, g (x + π) = g x) →
  (Set.Icc 0 (π / 8) : Set ℝ) = {x | x ∈ Set.Icc 0 π ∧ ∀ y ∈ Set.Icc 0 x, f y ≤ f x} :=
sorry

end NUMINAMATH_CALUDE_increasing_interval_of_sine_function_l3221_322129


namespace NUMINAMATH_CALUDE_chess_tournament_matches_l3221_322127

/-- The number of matches in a chess tournament -/
def tournament_matches (n : ℕ) (matches_per_pair : ℕ) : ℕ :=
  matches_per_pair * n * (n - 1) / 2

/-- Theorem: In a chess tournament with 150 players, where each player plays 3 matches
    against every other player, the total number of matches is 33,750 -/
theorem chess_tournament_matches :
  tournament_matches 150 3 = 33750 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_matches_l3221_322127


namespace NUMINAMATH_CALUDE_system_solution_l3221_322161

theorem system_solution (x y : ℝ) : 
  (x^2 + x*y + y^2 = 23) ∧ (x^4 + x^2*y^2 + y^4 = 253) →
  ((x = Real.sqrt 29 ∧ y = Real.sqrt 5) ∨ 
   (x = Real.sqrt 29 ∧ y = -Real.sqrt 5) ∨
   (x = -Real.sqrt 29 ∧ y = Real.sqrt 5) ∨
   (x = -Real.sqrt 29 ∧ y = -Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3221_322161


namespace NUMINAMATH_CALUDE_tanya_work_days_l3221_322194

/-- Given Sakshi can do a piece of work in 20 days and Tanya is 25% more efficient than Sakshi,
    prove that Tanya will take 16 days to do the same piece of work. -/
theorem tanya_work_days (sakshi_days : ℕ) (tanya_efficiency : ℚ) :
  sakshi_days = 20 →
  tanya_efficiency = 125 / 100 →
  (sakshi_days : ℚ) / tanya_efficiency = 16 := by
  sorry

end NUMINAMATH_CALUDE_tanya_work_days_l3221_322194


namespace NUMINAMATH_CALUDE_two_white_prob_correct_at_least_one_white_prob_correct_l3221_322173

/-- Represents the outcome of drawing a ball -/
inductive Ball
| White
| Black

/-- Represents the state of the bag of balls -/
structure BagState where
  total : Nat
  white : Nat
  black : Nat

/-- The initial state of the bag -/
def initialBag : BagState :=
  { total := 5, white := 3, black := 2 }

/-- Calculates the probability of drawing two white balls in succession -/
def probTwoWhite (bag : BagState) : Rat :=
  (bag.white / bag.total) * ((bag.white - 1) / (bag.total - 1))

/-- Calculates the probability of drawing at least one white ball in two draws -/
def probAtLeastOneWhite (bag : BagState) : Rat :=
  1 - (bag.black / bag.total) * ((bag.black - 1) / (bag.total - 1))

theorem two_white_prob_correct :
  probTwoWhite initialBag = 3 / 10 := by sorry

theorem at_least_one_white_prob_correct :
  probAtLeastOneWhite initialBag = 9 / 10 := by sorry

end NUMINAMATH_CALUDE_two_white_prob_correct_at_least_one_white_prob_correct_l3221_322173


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l3221_322121

theorem sqrt_expression_equality : 
  Real.sqrt 12 - Real.sqrt (1/3) - Real.sqrt 2 * Real.sqrt 6 = -(Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l3221_322121


namespace NUMINAMATH_CALUDE_jimmy_remaining_cards_l3221_322138

/-- Calculates the number of cards Jimmy has left after giving cards to Bob and Mary. -/
def cards_left (initial_cards : ℕ) (cards_to_bob : ℕ) : ℕ :=
  initial_cards - cards_to_bob - (2 * cards_to_bob)

/-- Theorem stating that Jimmy has 9 cards left after giving cards to Bob and Mary. -/
theorem jimmy_remaining_cards :
  cards_left 18 3 = 9 := by
  sorry

#eval cards_left 18 3

end NUMINAMATH_CALUDE_jimmy_remaining_cards_l3221_322138


namespace NUMINAMATH_CALUDE_probability_between_C_and_E_l3221_322147

/-- Given points A, B, C, D, E on a line segment AB, prove that the probability
    of a randomly selected point on AB being between C and E is 1/24. -/
theorem probability_between_C_and_E (A B C D E : ℝ) : 
  A < C ∧ C < E ∧ E < D ∧ D < B →  -- Points are ordered on the line
  B - A = 4 * (D - A) →            -- AB = 4AD
  B - A = 8 * (C - B) →            -- AB = 8BC
  B - E = 2 * (E - C) →            -- BE = 2CE
  (E - C) / (B - A) = 1 / 24 := by
    sorry

end NUMINAMATH_CALUDE_probability_between_C_and_E_l3221_322147


namespace NUMINAMATH_CALUDE_function_constancy_l3221_322159

def is_constant {α : Type*} (f : α → ℕ) : Prop :=
  ∀ x y, f x = f y

theorem function_constancy (f : ℤ × ℤ → ℕ) 
  (h : ∀ (x y : ℤ), 4 * f (x, y) = f (x - 1, y) + f (x, y + 1) + f (x + 1, y) + f (x, y - 1)) :
  is_constant f := by
  sorry

end NUMINAMATH_CALUDE_function_constancy_l3221_322159


namespace NUMINAMATH_CALUDE_carpet_transformation_possible_l3221_322162

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a cut piece of a rectangle -/
structure CutPiece where
  width : ℕ
  height : ℕ

/-- Represents the state of the carpet after being cut -/
structure DamagedCarpet where
  original : Rectangle
  cutOut : CutPiece

/-- Function to check if a transformation from a damaged carpet to a new rectangle is possible -/
def canTransform (damaged : DamagedCarpet) (new : Rectangle) : Prop :=
  damaged.original.width * damaged.original.height - 
  damaged.cutOut.width * damaged.cutOut.height = 
  new.width * new.height

/-- The main theorem to prove -/
theorem carpet_transformation_possible : 
  ∃ (damaged : DamagedCarpet) (new : Rectangle),
    damaged.original = ⟨9, 12⟩ ∧ 
    damaged.cutOut = ⟨1, 8⟩ ∧
    new = ⟨10, 10⟩ ∧
    canTransform damaged new :=
sorry

end NUMINAMATH_CALUDE_carpet_transformation_possible_l3221_322162


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l3221_322181

theorem max_value_cos_sin (θ : Real) (h : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  (Real.cos (θ / 2))^2 * (1 - Real.sin θ) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l3221_322181


namespace NUMINAMATH_CALUDE_smallest_consecutive_digit_sum_divisible_by_7_l3221_322100

-- Define a function to calculate the digit sum of a natural number
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digitSum (n / 10)

-- Define a predicate for consecutive numbers with digit sums divisible by 7
def consecutiveDigitSumDivisibleBy7 (n : ℕ) : Prop :=
  (digitSum n) % 7 = 0 ∧ (digitSum (n + 1)) % 7 = 0

-- Theorem statement
theorem smallest_consecutive_digit_sum_divisible_by_7 :
  ∀ n : ℕ, n < 69999 → ¬(consecutiveDigitSumDivisibleBy7 n) ∧
  consecutiveDigitSumDivisibleBy7 69999 :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_digit_sum_divisible_by_7_l3221_322100


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3221_322169

/-- The speed of a boat in still water, given its downstream travel time and distance, and the stream's speed. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 7)
  (h3 : downstream_distance = 147) :
  downstream_distance = (boat_speed + stream_speed) * downstream_time →
  boat_speed = 16 :=
by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3221_322169


namespace NUMINAMATH_CALUDE_s_square_sum_l3221_322128

/-- The sequence s_n is defined by the power series expansion of 1 / (1 - 2x - x^2) -/
noncomputable def s : ℕ → ℝ := sorry

/-- The power series expansion of 1 / (1 - 2x - x^2) -/
axiom power_series_expansion (x : ℝ) (h : x ≠ 0) : 
  (1 : ℝ) / (1 - 2*x - x^2) = ∑' (n : ℕ), s n * x^n

/-- The main theorem: s_n^2 + s_{n+1}^2 = s_{2n+2} for all non-negative integers n -/
theorem s_square_sum (n : ℕ) : (s n)^2 + (s (n+1))^2 = s (2*n+2) := by sorry

end NUMINAMATH_CALUDE_s_square_sum_l3221_322128


namespace NUMINAMATH_CALUDE_milk_consumption_l3221_322126

theorem milk_consumption (bottle_milk : ℚ) (pour_fraction : ℚ) (drink_fraction : ℚ) :
  bottle_milk = 3/4 →
  pour_fraction = 1/2 →
  drink_fraction = 1/3 →
  drink_fraction * (pour_fraction * bottle_milk) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_milk_consumption_l3221_322126


namespace NUMINAMATH_CALUDE_sundae_price_l3221_322104

/-- Given a caterer's order of ice-cream bars and sundaes, calculate the price of each sundae. -/
theorem sundae_price
  (ice_cream_bars : ℕ)
  (sundaes : ℕ)
  (total_price : ℚ)
  (ice_cream_bar_price : ℚ)
  (h1 : ice_cream_bars = 125)
  (h2 : sundaes = 125)
  (h3 : total_price = 250)
  (h4 : ice_cream_bar_price = 0.6) :
  (total_price - ice_cream_bars * ice_cream_bar_price) / sundaes = 1.4 := by
  sorry

#check sundae_price

end NUMINAMATH_CALUDE_sundae_price_l3221_322104


namespace NUMINAMATH_CALUDE_sum_of_roots_l3221_322196

/-- Given a quadratic function f(x) = x^2 - 2016x + 2015 and two distinct points a and b
    where f(a) = f(b), prove that a + b = 2016 -/
theorem sum_of_roots (a b : ℝ) (ha : a ≠ b) :
  (a^2 - 2016*a + 2015 = b^2 - 2016*b + 2015) →
  a + b = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3221_322196


namespace NUMINAMATH_CALUDE_x_value_proof_l3221_322175

theorem x_value_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^4 / y = 2) (h2 : y^3 / z = 6) (h3 : z^2 / x = 8) :
  x = (18432 : ℝ)^(1/23) := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l3221_322175


namespace NUMINAMATH_CALUDE_fraction_equality_l3221_322188

theorem fraction_equality : 2 / 3 = (2 + 4) / (3 + 6) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3221_322188


namespace NUMINAMATH_CALUDE_base_conversion_three_digits_l3221_322198

theorem base_conversion_three_digits : 
  ∃ (b : ℕ), b > 1 ∧ b^2 ≤ 256 ∧ 256 < b^3 ∧ ∀ (x : ℕ), 1 < x ∧ x < b → (x^2 > 256 ∨ x^3 ≤ 256) :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_three_digits_l3221_322198


namespace NUMINAMATH_CALUDE_symmetric_seven_zeros_sum_l3221_322187

/-- A function representing |(1-x^2)(x^2+ax+b)| - c -/
def f (a b c x : ℝ) : ℝ := |(1 - x^2) * (x^2 + a*x + b)| - c

/-- Symmetry condition: f is symmetric about x = -2 -/
def is_symmetric (a b c : ℝ) : Prop :=
  ∀ x, f a b c (x + 2) = f a b c (-x - 2)

/-- The function has exactly 7 zeros -/
def has_seven_zeros (a b c : ℝ) : Prop :=
  ∃! (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, f a b c x = 0

theorem symmetric_seven_zeros_sum (a b c : ℝ) :
  is_symmetric a b c →
  has_seven_zeros a b c →
  c ≠ 0 →
  a + b + c = 32 := by sorry

end NUMINAMATH_CALUDE_symmetric_seven_zeros_sum_l3221_322187


namespace NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l3221_322151

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) : a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l3221_322151


namespace NUMINAMATH_CALUDE_outfits_count_l3221_322155

/-- The number of possible outfits with different colored shirt and hat -/
def number_of_outfits (red_shirts green_shirts pants green_hats red_hats : ℕ) : ℕ :=
  (red_shirts * green_hats + green_shirts * red_hats) * pants

/-- Theorem stating the number of outfits given the specific quantities -/
theorem outfits_count : number_of_outfits 6 4 7 10 9 = 672 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3221_322155


namespace NUMINAMATH_CALUDE_abc_product_l3221_322101

theorem abc_product (a b c : ℝ) 
  (eq1 : b + c = 16) 
  (eq2 : c + a = 17) 
  (eq3 : a + b = 18) : 
  a * b * c = 606.375 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l3221_322101


namespace NUMINAMATH_CALUDE_cube_arrangement_theorem_l3221_322199

/-- Represents a cube with colored faces -/
structure Cube where
  blue_faces : Nat
  red_faces : Nat

/-- Represents an arrangement of cubes into a larger cube -/
structure CubeArrangement where
  cubes : List Cube
  visible_red_faces : Nat
  visible_blue_faces : Nat

/-- The theorem to be proved -/
theorem cube_arrangement_theorem 
  (cubes : List Cube) 
  (first_arrangement : CubeArrangement) :
  (cubes.length = 8) →
  (∀ c ∈ cubes, c.blue_faces = 2 ∧ c.red_faces = 4) →
  (first_arrangement.cubes = cubes) →
  (first_arrangement.visible_red_faces = 8) →
  (first_arrangement.visible_blue_faces = 16) →
  (∃ second_arrangement : CubeArrangement,
    second_arrangement.cubes = cubes ∧
    second_arrangement.visible_red_faces = 24 ∧
    second_arrangement.visible_blue_faces = 0) :=
by sorry

end NUMINAMATH_CALUDE_cube_arrangement_theorem_l3221_322199


namespace NUMINAMATH_CALUDE_power_calculation_l3221_322150

theorem power_calculation : 
  (27 : ℝ)^3 * 9^2 / 3^17 = 1/81 :=
by
  have h1 : (27 : ℝ) = 3^3 := by sorry
  have h2 : (9 : ℝ) = 3^2 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_calculation_l3221_322150


namespace NUMINAMATH_CALUDE_infinitely_many_square_sum_square_no_zero_l3221_322119

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Check if a number contains no zero digits -/
def no_zero_digits (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem infinitely_many_square_sum_square_no_zero :
  ∃ f : ℕ → ℕ, 
    (∀ m : ℕ, ∃ k : ℕ, f m = k^2) ∧ 
    (∀ m : ℕ, ∃ l : ℕ, S (f m) = l^2) ∧
    (∀ m : ℕ, no_zero_digits (f m)) ∧
    Function.Injective f :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_square_sum_square_no_zero_l3221_322119


namespace NUMINAMATH_CALUDE_p_recurrence_l3221_322166

/-- The probability of getting a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℝ := sorry

/-- The recurrence relation for p_{n,k} -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + 1 / 2^k := by sorry

end NUMINAMATH_CALUDE_p_recurrence_l3221_322166


namespace NUMINAMATH_CALUDE_vertical_angles_are_equal_converse_is_false_l3221_322140

-- Define what it means for angles to be vertical
def are_vertical_angles (α β : Real) : Prop := sorry

-- Define what it means for angles to be equal
def are_equal_angles (α β : Real) : Prop := α = β

-- Theorem stating that vertical angles are equal
theorem vertical_angles_are_equal (α β : Real) : 
  are_vertical_angles α β → are_equal_angles α β := by sorry

-- Theorem stating that the converse is false
theorem converse_is_false : 
  ¬(∀ α β : Real, are_equal_angles α β → are_vertical_angles α β) := by sorry

end NUMINAMATH_CALUDE_vertical_angles_are_equal_converse_is_false_l3221_322140


namespace NUMINAMATH_CALUDE_total_tulips_count_l3221_322131

def tulips_per_eye : ℕ := 8
def number_of_eyes : ℕ := 2
def tulips_for_smile : ℕ := 18
def background_multiplier : ℕ := 9

def total_tulips : ℕ := 
  (tulips_per_eye * number_of_eyes + tulips_for_smile) + 
  (background_multiplier * tulips_for_smile)

theorem total_tulips_count : total_tulips = 196 := by
  sorry

end NUMINAMATH_CALUDE_total_tulips_count_l3221_322131


namespace NUMINAMATH_CALUDE_square_value_theorem_l3221_322125

theorem square_value_theorem (a b : ℝ) (h : a > b) :
  ∃ square : ℝ, (-2*a - 1 < -2*b + square) ∧ (square = 0) := by
sorry

end NUMINAMATH_CALUDE_square_value_theorem_l3221_322125


namespace NUMINAMATH_CALUDE_symmetry_yoz_plane_l3221_322143

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The yoz plane in 3D space -/
def yozPlane : Set Point3D := {p : Point3D | p.x = 0}

/-- Symmetry with respect to the yoz plane -/
def symmetricPointYOZ (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

theorem symmetry_yoz_plane :
  let p : Point3D := ⟨2, 3, 5⟩
  symmetricPointYOZ p = ⟨-2, 3, 5⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetry_yoz_plane_l3221_322143


namespace NUMINAMATH_CALUDE_log_equation_sum_l3221_322148

theorem log_equation_sum (A B C : ℕ+) : 
  (Nat.gcd A.val (Nat.gcd B.val C.val) = 1) →
  (A : ℝ) * (Real.log 5 / Real.log 100) + (B : ℝ) * (Real.log 2 / Real.log 100) = C →
  A + B + C = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_sum_l3221_322148


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3221_322163

/-- A quadratic function f with parameter t -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (x - (t + 2) / 2)^2 - t^2 / 4

/-- The theorem stating the properties of the quadratic function and the value of t -/
theorem quadratic_function_properties (t : ℝ) :
  t ≠ 0 ∧
  f t ((t + 2) / 2) = -t^2 / 4 ∧
  f t 1 = 0 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) (1/2), f t x ≥ -5) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) (1/2), f t x = -5) →
  t = -9/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3221_322163


namespace NUMINAMATH_CALUDE_greatest_b_value_l3221_322182

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 12 ≥ 0 → x ≤ 6) ∧ 
  (-6^2 + 8*6 - 12 ≥ 0) := by
sorry

end NUMINAMATH_CALUDE_greatest_b_value_l3221_322182


namespace NUMINAMATH_CALUDE_rectangle_length_equality_l3221_322130

/-- Given a figure composed of rectangles with right angles, prove that the unknown length Y is 1 cm --/
theorem rectangle_length_equality (Y : ℝ) : Y = 1 := by
  -- Define the sum of top segment lengths
  let top_sum := 3 + 2 + 3 + 4 + Y
  -- Define the sum of bottom segment lengths
  let bottom_sum := 7 + 4 + 2
  -- Assert that the sums are equal (property of rectangles)
  have sum_equality : top_sum = bottom_sum := by sorry
  -- Solve for Y
  sorry


end NUMINAMATH_CALUDE_rectangle_length_equality_l3221_322130


namespace NUMINAMATH_CALUDE_car_travel_time_fraction_l3221_322152

theorem car_travel_time_fraction (distance : ℝ) (original_time : ℝ) (new_speed : ℝ) 
  (h1 : distance = 432)
  (h2 : original_time = 6)
  (h3 : new_speed = 48) : 
  (distance / new_speed) / original_time = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_time_fraction_l3221_322152


namespace NUMINAMATH_CALUDE_max_digits_product_5_and_3_l3221_322192

theorem max_digits_product_5_and_3 : 
  ∀ a b : ℕ, 
  10000 ≤ a ∧ a ≤ 99999 → 
  100 ≤ b ∧ b ≤ 999 → 
  a * b < 100000000 := by
sorry

end NUMINAMATH_CALUDE_max_digits_product_5_and_3_l3221_322192


namespace NUMINAMATH_CALUDE_largest_710_double_correct_l3221_322193

/-- Converts a positive integer to its base-7 representation as a list of digits --/
def toBase7 (n : ℕ+) : List ℕ :=
  sorry

/-- Converts a list of digits to a base-10 number --/
def toBase10 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if a positive integer is a 7-10 double --/
def is710Double (n : ℕ+) : Prop :=
  toBase10 (toBase7 n) = 2 * n

/-- The largest 7-10 double --/
def largest710Double : ℕ+ := 315

theorem largest_710_double_correct :
  is710Double largest710Double ∧
  ∀ n : ℕ+, n > largest710Double → ¬is710Double n :=
sorry

end NUMINAMATH_CALUDE_largest_710_double_correct_l3221_322193


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l3221_322184

theorem inserted_numbers_sum : ∃ (a b : ℝ), 
  4 < a ∧ a < b ∧ b < 16 ∧ 
  (b - a = a - 4) ∧
  (b * b = a * 16) ∧
  a + b = 20 := by
  sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l3221_322184


namespace NUMINAMATH_CALUDE_product_zero_from_sum_and_cube_sum_l3221_322133

theorem product_zero_from_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 125) : 
  a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_from_sum_and_cube_sum_l3221_322133


namespace NUMINAMATH_CALUDE_max_value_a_plus_b_l3221_322171

/-- Given that -1/4 * x^2 ≤ ax + b ≤ e^x for all x ∈ ℝ, the maximum value of a + b is 2 -/
theorem max_value_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, -1/4 * x^2 ≤ a * x + b ∧ a * x + b ≤ Real.exp x) → 
  (∀ c d : ℝ, (∀ x : ℝ, -1/4 * x^2 ≤ c * x + d ∧ c * x + d ≤ Real.exp x) → a + b ≥ c + d) →
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_a_plus_b_l3221_322171


namespace NUMINAMATH_CALUDE_youngest_child_age_l3221_322113

/-- Represents a family with its members and ages -/
structure Family where
  members : ℕ
  totalAge : ℕ

/-- Calculates the average age of a family -/
def averageAge (f : Family) : ℚ :=
  f.totalAge / f.members

theorem youngest_child_age (initialFamily : Family) 
  (finalFamily : Family) (timePassed : ℕ) (ageDifference : ℕ) :
  initialFamily.members = 4 →
  averageAge initialFamily = 24 →
  timePassed = 10 →
  finalFamily.members = initialFamily.members + 2 →
  ageDifference = 2 →
  averageAge finalFamily = 24 →
  ∃ (youngestAge : ℕ), 
    youngestAge = 3 ∧ 
    finalFamily.totalAge = initialFamily.totalAge + timePassed * initialFamily.members + youngestAge + (youngestAge + ageDifference) :=
by sorry


end NUMINAMATH_CALUDE_youngest_child_age_l3221_322113


namespace NUMINAMATH_CALUDE_triangle_similarity_from_arithmetic_sides_l3221_322174

/-- Two triangles with sides in arithmetic progression and one equal angle are similar -/
theorem triangle_similarity_from_arithmetic_sides (a b c a₁ b₁ c₁ : ℝ) 
  (angleCAB angleCBA angleABC angleC₁A₁B₁ angleC₁B₁A₁ angleA₁B₁C₁ : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < a₁ ∧ 0 < b₁ ∧ 0 < c₁ →
  b - a = c - b →
  b₁ - a₁ = c₁ - b₁ →
  angleCAB + angleCBA + angleABC = π →
  angleC₁A₁B₁ + angleC₁B₁A₁ + angleA₁B₁C₁ = π →
  angleCAB = angleC₁A₁B₁ →
  ∃ (k : ℝ), k > 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁ :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_from_arithmetic_sides_l3221_322174


namespace NUMINAMATH_CALUDE_minimum_artists_count_l3221_322153

theorem minimum_artists_count : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 5 = 1 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 3 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 5 = 1 ∧ m % 6 = 2 ∧ m % 8 = 3 → m ≥ n) ∧
  n = 236 := by
  sorry

end NUMINAMATH_CALUDE_minimum_artists_count_l3221_322153


namespace NUMINAMATH_CALUDE_time_to_work_l3221_322178

def round_trip_time : ℝ := 2
def speed_to_work : ℝ := 80
def speed_to_home : ℝ := 120

theorem time_to_work :
  let distance := (round_trip_time * speed_to_work * speed_to_home) / (speed_to_work + speed_to_home)
  let time_to_work := distance / speed_to_work
  time_to_work * 60 = 72 := by sorry

end NUMINAMATH_CALUDE_time_to_work_l3221_322178


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l3221_322197

theorem negation_of_existence (f : ℝ → ℝ) : 
  (¬ ∃ x : ℝ, f x = 0) ↔ (∀ x : ℝ, f x ≠ 0) := by sorry

theorem negation_of_cubic_equation :
  (¬ ∃ x : ℝ, x^3 + 5*x - 2 = 0) ↔ (∀ x : ℝ, x^3 + 5*x - 2 ≠ 0) := by
  apply negation_of_existence (λ x => x^3 + 5*x - 2)

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l3221_322197


namespace NUMINAMATH_CALUDE_min_b_value_l3221_322103

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/4) * x + 3/(4*x) - 1

def g (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 4

theorem min_b_value (b : ℝ) :
  (∀ x₁ ∈ Set.Ioo 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g b x₂) →
  b ≥ 17/8 := by
  sorry

end NUMINAMATH_CALUDE_min_b_value_l3221_322103


namespace NUMINAMATH_CALUDE_ax_plus_by_fifth_power_l3221_322186

theorem ax_plus_by_fifth_power (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 6)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = -360 := by sorry

end NUMINAMATH_CALUDE_ax_plus_by_fifth_power_l3221_322186
