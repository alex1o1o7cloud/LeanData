import Mathlib

namespace NUMINAMATH_CALUDE_coach_mike_change_l646_64649

/-- The amount of change Coach Mike received when buying lemonade -/
theorem coach_mike_change (cost : ℕ) (given : ℕ) (change : ℕ) : 
  cost = 58 → given = 75 → change = given - cost → change = 17 := by
  sorry

end NUMINAMATH_CALUDE_coach_mike_change_l646_64649


namespace NUMINAMATH_CALUDE_sum_of_coefficients_3x_minus_4y_power_20_l646_64630

theorem sum_of_coefficients_3x_minus_4y_power_20 :
  let f : ℝ → ℝ → ℝ := λ x y => (3*x - 4*y)^20
  (f 1 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_3x_minus_4y_power_20_l646_64630


namespace NUMINAMATH_CALUDE_rhombus_side_length_l646_64661

/-- A rhombus with perimeter 60 cm has sides of length 15 cm each. -/
theorem rhombus_side_length (perimeter : ℝ) (side_length : ℝ) : 
  perimeter = 60 → side_length * 4 = perimeter → side_length = 15 := by
  sorry

#check rhombus_side_length

end NUMINAMATH_CALUDE_rhombus_side_length_l646_64661


namespace NUMINAMATH_CALUDE_sector_arc_length_l646_64654

/-- Given a sector with area 9 and central angle 2 radians, its arc length is 6. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 9 → angle = 2 → arc_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l646_64654


namespace NUMINAMATH_CALUDE_coin_flip_probability_l646_64668

theorem coin_flip_probability :
  let p : ℝ := 1/3  -- Probability of getting heads in a single flip
  let q : ℝ := 1 - p  -- Probability of getting tails in a single flip
  let num_players : ℕ := 4  -- Number of players
  let prob_same_flips : ℝ := (p^num_players) * (∑' n, q^(num_players * n)) -- Probability all players flip same number of times
  prob_same_flips = 1/65
  := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l646_64668


namespace NUMINAMATH_CALUDE_statement_d_not_always_true_l646_64618

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- Two lines are different if they are not equal -/
def different_lines (m n : Line) : Prop := m ≠ n

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perpendicular (l1 l2 : Line) : Prop := sorry

/-- Statement D is not always true -/
theorem statement_d_not_always_true 
  (α : Plane) (m n : Line) 
  (h1 : different_lines m n) 
  (h2 : lines_perpendicular m n) 
  (h3 : line_perp_plane m α) : 
  ¬ (line_parallel_plane n α) := sorry

end NUMINAMATH_CALUDE_statement_d_not_always_true_l646_64618


namespace NUMINAMATH_CALUDE_geometric_series_example_l646_64610

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  (a - a * r^n) / (1 - r)

theorem geometric_series_example : 
  let a : ℚ := 1/5
  let r : ℚ := -1/5
  let n : ℕ := 6
  geometric_series_sum a r n = 1562/9375 := by sorry

end NUMINAMATH_CALUDE_geometric_series_example_l646_64610


namespace NUMINAMATH_CALUDE_three_integers_sum_l646_64629

theorem three_integers_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 125 →
  (a : ℕ) + b + c = 31 := by
  sorry

end NUMINAMATH_CALUDE_three_integers_sum_l646_64629


namespace NUMINAMATH_CALUDE_possible_pen_counts_l646_64615

def total_money : ℕ := 11
def pen_cost : ℕ := 3
def notebook_cost : ℕ := 1

def valid_pen_count (x : ℕ) : Prop :=
  ∃ y : ℕ, x * pen_cost + y * notebook_cost = total_money

theorem possible_pen_counts : 
  (valid_pen_count 1 ∧ valid_pen_count 2 ∧ valid_pen_count 3) ∧
  (∀ x : ℕ, valid_pen_count x → x = 1 ∨ x = 2 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_possible_pen_counts_l646_64615


namespace NUMINAMATH_CALUDE_hexagons_in_nth_ring_hexagons_in_100th_ring_l646_64698

/-- The number of hexagons in the nth ring of a hexagonal array -/
def hexagons_in_ring (n : ℕ) : ℕ := 6 * n

/-- Theorem: The number of hexagons in the nth ring is 6n -/
theorem hexagons_in_nth_ring (n : ℕ) :
  hexagons_in_ring n = 6 * n := by sorry

/-- Corollary: The number of hexagons in the 100th ring is 600 -/
theorem hexagons_in_100th_ring :
  hexagons_in_ring 100 = 600 := by sorry

end NUMINAMATH_CALUDE_hexagons_in_nth_ring_hexagons_in_100th_ring_l646_64698


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l646_64687

theorem arithmetic_equalities :
  (-16 - (-12) - 24 + 18 = -10) ∧
  (-6^2 * (-7/12 - 13/36 + 5/6) = 4) := by sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l646_64687


namespace NUMINAMATH_CALUDE_kindergarten_group_divisibility_l646_64633

theorem kindergarten_group_divisibility (n : ℕ) (a : ℕ) (h1 : n = 3 * a / 2) 
  (h2 : a % 2 = 0) (h3 : a % 4 = 0) : n % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_group_divisibility_l646_64633


namespace NUMINAMATH_CALUDE_cakes_after_school_l646_64657

theorem cakes_after_school (croissants_per_person breakfast_people pizzas_per_person bedtime_people total_food : ℕ) 
  (h1 : croissants_per_person = 7)
  (h2 : breakfast_people = 2)
  (h3 : pizzas_per_person = 30)
  (h4 : bedtime_people = 2)
  (h5 : total_food = 110) :
  ∃ (cakes_per_person : ℕ), 
    croissants_per_person * breakfast_people + cakes_per_person * 2 + pizzas_per_person * bedtime_people = total_food ∧ 
    cakes_per_person = 18 := by
  sorry

end NUMINAMATH_CALUDE_cakes_after_school_l646_64657


namespace NUMINAMATH_CALUDE_reflect_parabola_x_axis_l646_64611

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflects a parabola across the x-axis -/
def reflect_x_axis (p : Parabola) : Parabola :=
  { a := -p.a, b := -p.b, c := -p.c }

/-- The original parabola y = x^2 - x - 1 -/
def original_parabola : Parabola :=
  { a := 1, b := -1, c := -1 }

theorem reflect_parabola_x_axis :
  reflect_x_axis original_parabola = { a := -1, b := 1, c := 1 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_parabola_x_axis_l646_64611


namespace NUMINAMATH_CALUDE_S_not_equal_T_l646_64621

def S : Set Int := {x | ∃ n : Int, x = 2 * n}
def T : Set Int := {x | ∃ k : Int, x = 4 * k + 1 ∨ x = 4 * k - 1}

theorem S_not_equal_T : S ≠ T := by
  sorry

end NUMINAMATH_CALUDE_S_not_equal_T_l646_64621


namespace NUMINAMATH_CALUDE_square_circle_distance_sum_constant_l646_64692

/-- Given a square ABCD with side length 2a and a circle k centered at the center of the square with radius R, 
    the sum of squared distances from any point P on the circle to the vertices of the square is constant. -/
theorem square_circle_distance_sum_constant 
  (a R : ℝ) 
  (A B C D : ℝ × ℝ) 
  (k : Set (ℝ × ℝ)) 
  (h_square : A = (-a, a) ∧ B = (a, a) ∧ C = (a, -a) ∧ D = (-a, -a))
  (h_circle : k = {P : ℝ × ℝ | P.1^2 + P.2^2 = R^2}) :
  ∀ P ∈ k, 
    (P.1 - A.1)^2 + (P.2 - A.2)^2 + 
    (P.1 - B.1)^2 + (P.2 - B.2)^2 + 
    (P.1 - C.1)^2 + (P.2 - C.2)^2 + 
    (P.1 - D.1)^2 + (P.2 - D.2)^2 = 4*R^2 + 8*a^2 :=
by sorry

end NUMINAMATH_CALUDE_square_circle_distance_sum_constant_l646_64692


namespace NUMINAMATH_CALUDE_deposit_percentage_l646_64684

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) :
  deposit = 150 →
  remaining = 1350 →
  (deposit / (deposit + remaining)) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_deposit_percentage_l646_64684


namespace NUMINAMATH_CALUDE_dice_tosses_probability_l646_64693

/-- The number of sides on the dice -/
def num_sides : ℕ := 8

/-- The probability of rolling a 3 on a single toss -/
def p_roll_3 : ℚ := 1 / num_sides

/-- The target probability of rolling a 3 at least once -/
def target_prob : ℚ := 111328125 / 1000000000

/-- The number of tosses -/
def num_tosses : ℕ := 7

theorem dice_tosses_probability :
  1 - (1 - p_roll_3) ^ num_tosses = target_prob := by sorry

end NUMINAMATH_CALUDE_dice_tosses_probability_l646_64693


namespace NUMINAMATH_CALUDE_total_fat_served_l646_64622

/-- The amount of fat in ounces for each type of fish --/
def herring_fat : ℕ := 40
def eel_fat : ℕ := 20
def pike_fat : ℕ := eel_fat + 10
def salmon_fat : ℕ := 35
def halibut_fat : ℕ := 50

/-- The number of each type of fish served --/
def herring_count : ℕ := 40
def eel_count : ℕ := 30
def pike_count : ℕ := 25
def salmon_count : ℕ := 20
def halibut_count : ℕ := 15

/-- The total amount of fat served --/
def total_fat : ℕ := 
  herring_fat * herring_count +
  eel_fat * eel_count +
  pike_fat * pike_count +
  salmon_fat * salmon_count +
  halibut_fat * halibut_count

theorem total_fat_served : total_fat = 4400 := by
  sorry

end NUMINAMATH_CALUDE_total_fat_served_l646_64622


namespace NUMINAMATH_CALUDE_tile_border_ratio_l646_64642

theorem tile_border_ratio (n s d : ℝ) (h1 : n = 30) 
  (h2 : (n^2 * s^2) / ((n*s + 2*n*d)^2) = 0.81) : d/s = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l646_64642


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l646_64659

theorem smallest_n_for_candy_purchase : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (r g b : ℕ), r > 0 → g > 0 → b > 0 → 10 * r = 18 * g ∧ 18 * g = 20 * b ∧ 20 * b = 24 * n) ∧
  (∀ (m : ℕ), m > 0 → 
    (∀ (r g b : ℕ), r > 0 → g > 0 → b > 0 → 10 * r = 18 * g ∧ 18 * g = 20 * b ∧ 20 * b = 24 * m) → 
    n ≤ m) ∧
  n = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l646_64659


namespace NUMINAMATH_CALUDE_factorization_x_10_minus_1024_l646_64648

theorem factorization_x_10_minus_1024 (x : ℝ) : 
  x^10 - 1024 = (x^5 + 32) * (x^5 - 32) :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_x_10_minus_1024_l646_64648


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l646_64616

theorem rectangle_shorter_side (a b d : ℝ) : 
  a / b = 3 / 4 →  -- ratio of sides is 3:4
  d^2 = a^2 + b^2 →  -- Pythagorean theorem
  d = 9 →  -- diagonal is 9
  a = 5.4 :=  -- shorter side is 5.4
by sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l646_64616


namespace NUMINAMATH_CALUDE_park_conditions_l646_64696

-- Define the basic conditions
def temperature_at_least_70 : Prop := sorry
def is_sunny : Prop := sorry
def park_is_packed : Prop := sorry

-- Define the main theorem
theorem park_conditions :
  (temperature_at_least_70 ∧ is_sunny → park_is_packed) →
  (¬park_is_packed → ¬temperature_at_least_70 ∨ ¬is_sunny) :=
by sorry

end NUMINAMATH_CALUDE_park_conditions_l646_64696


namespace NUMINAMATH_CALUDE_opposite_blue_is_black_l646_64636

-- Define the colors
inductive Color
| Blue | Yellow | Orange | Black | Silver | Gold

-- Define a cube
structure Cube where
  faces : Fin 6 → Color

-- Define the views
structure View where
  top : Color
  front : Color
  right : Color

-- Define the problem setup
def cube_problem (c : Cube) (v1 v2 v3 : View) : Prop :=
  -- All faces have different colors
  (∀ i j : Fin 6, i ≠ j → c.faces i ≠ c.faces j) ∧
  -- The views are consistent with the cube
  (v1.top = Color.Gold ∧ v1.front = Color.Black ∧ v1.right = Color.Orange) ∧
  (v2.top = Color.Gold ∧ v2.front = Color.Yellow ∧ v2.right = Color.Orange) ∧
  (v3.top = Color.Gold ∧ v3.front = Color.Silver ∧ v3.right = Color.Orange)

-- The theorem to prove
theorem opposite_blue_is_black (c : Cube) (v1 v2 v3 : View) 
  (h : cube_problem c v1 v2 v3) : 
  ∃ (i j : Fin 6), c.faces i = Color.Blue ∧ c.faces j = Color.Black ∧ 
  (i.val + j.val = 5 ∨ i.val + j.val = 7) :=
sorry

end NUMINAMATH_CALUDE_opposite_blue_is_black_l646_64636


namespace NUMINAMATH_CALUDE_max_value_of_z_l646_64646

theorem max_value_of_z (x y : ℝ) (h1 : x + 2*y - 5 ≥ 0) (h2 : x - 2*y + 3 ≥ 0) (h3 : x - 5 ≤ 0) :
  ∀ x' y', x' + 2*y' - 5 ≥ 0 → x' - 2*y' + 3 ≥ 0 → x' - 5 ≤ 0 → x + y ≥ x' + y' ∧ x + y ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_z_l646_64646


namespace NUMINAMATH_CALUDE_distance_calculation_l646_64671

/-- Represents the distance to the destination in kilometers -/
def distance : ℝ := 96

/-- Represents the rowing speed in still water in km/h -/
def rowing_speed : ℝ := 10

/-- Represents the current velocity in km/h -/
def current_velocity : ℝ := 2

/-- Represents the total round trip time in hours -/
def total_time : ℝ := 20

/-- Theorem stating that the given conditions result in the correct distance -/
theorem distance_calculation :
  let speed_with_current := rowing_speed + current_velocity
  let speed_against_current := rowing_speed - current_velocity
  (distance / speed_with_current) + (distance / speed_against_current) = total_time :=
sorry

end NUMINAMATH_CALUDE_distance_calculation_l646_64671


namespace NUMINAMATH_CALUDE_integer_solutions_for_k_l646_64667

theorem integer_solutions_for_k (k : ℤ) : 
  (∃ x : ℤ, 9 * x - 3 = k * x + 14) ↔ k ∈ ({8, 10, -8, 26} : Set ℤ) := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_for_k_l646_64667


namespace NUMINAMATH_CALUDE_ancient_greek_lifespan_l646_64686

/-- Represents a year in the BC/AD calendar system -/
inductive Year
| BC (n : ℕ)  -- n years Before Christ
| AD (n : ℕ)  -- n years Anno Domini

/-- Calculates the number of years between two dates, accounting for no year 0 -/
def yearsBetween (birth : Year) (death : Year) : ℕ :=
  match birth, death with
  | Year.BC b, Year.AD d => b + d - 1
  | _, _ => 0  -- Other cases are not relevant for this problem

/-- Theorem stating the lifespan of the ancient Greek -/
theorem ancient_greek_lifespan :
  yearsBetween (Year.BC 40) (Year.AD 40) = 79 := by
  sorry

end NUMINAMATH_CALUDE_ancient_greek_lifespan_l646_64686


namespace NUMINAMATH_CALUDE_y_value_proof_l646_64652

theorem y_value_proof (y : ℝ) (h : 9 / (y^2) = y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l646_64652


namespace NUMINAMATH_CALUDE_swan_count_l646_64676

/-- The number of swans in a lake that has "a pair plus two more" -/
def pair_plus_two (x : ℕ) : Prop := ∃ n : ℕ, x = 2 * n + 2

/-- The number of swans in a lake that has "three minus three" -/
def three_minus_three (x : ℕ) : Prop := ∃ m : ℕ, x = 3 * m - 3

/-- The total number of swans satisfies both conditions -/
theorem swan_count : ∃ x : ℕ, pair_plus_two x ∧ three_minus_three x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_swan_count_l646_64676


namespace NUMINAMATH_CALUDE_binomial_identity_l646_64626

theorem binomial_identity (n k : ℕ) (hn : n > 1) (hk : k > 1) (hkn : k ≤ n) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_identity_l646_64626


namespace NUMINAMATH_CALUDE_area_of_M_l646_64614

-- Define the region M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (|y| + |4 - y| ≤ 4) ∧
               ((y^2 + x - 4*y + 1) / (2*y + x - 7) ≤ 0)}

-- State the theorem
theorem area_of_M : MeasureTheory.volume M = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_of_M_l646_64614


namespace NUMINAMATH_CALUDE_first_day_is_sunday_l646_64637

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after n days -/
def afterDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (afterDays d n)

/-- Theorem: If the 18th day of a month is a Wednesday, then the 1st day of that month is a Sunday -/
theorem first_day_is_sunday (d : DayOfWeek) (h : afterDays d 17 = DayOfWeek.Wednesday) :
  d = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_first_day_is_sunday_l646_64637


namespace NUMINAMATH_CALUDE_simplify_expression_l646_64601

theorem simplify_expression : 15 * (7 / 10) * (1 / 9) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l646_64601


namespace NUMINAMATH_CALUDE_problem_solution_l646_64634

theorem problem_solution (x : ℝ) : 3 ∈ ({x + 2, x^2 + 2*x} : Set ℝ) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l646_64634


namespace NUMINAMATH_CALUDE_trapezoid_circle_radii_l646_64669

/-- An isosceles trapezoid with inscribed and circumscribed circles -/
structure IsoscelesTrapezoid where
  -- Base lengths
  BC : ℝ
  AD : ℝ
  -- Inscribed circle exists
  has_inscribed_circle : Bool
  -- Circumscribed circle exists
  has_circumscribed_circle : Bool

/-- The radii of inscribed and circumscribed circles of an isosceles trapezoid -/
def circle_radii (t : IsoscelesTrapezoid) : ℝ × ℝ :=
  sorry

theorem trapezoid_circle_radii (t : IsoscelesTrapezoid) 
  (h1 : t.BC = 4)
  (h2 : t.AD = 16)
  (h3 : t.has_inscribed_circle = true)
  (h4 : t.has_circumscribed_circle = true) :
  circle_radii t = (4, 5 * Real.sqrt 41 / 4) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_circle_radii_l646_64669


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_for_greater_than_two_l646_64604

theorem necessary_but_not_sufficient_condition_for_greater_than_two (a : ℝ) :
  (a ≥ 2 → a > 2 → True) ∧ ¬(a ≥ 2 → a > 2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_for_greater_than_two_l646_64604


namespace NUMINAMATH_CALUDE_silver_zinc_battery_properties_l646_64665

/-- Represents an electrode in the battery -/
inductive Electrode
| Zinc
| SilverOxide

/-- Represents the direction of current flow -/
inductive CurrentFlow
| FromZincToSilverOxide
| FromSilverOxideToZinc

/-- Represents the change in OH⁻ concentration -/
inductive OHConcentrationChange
| Increase
| Decrease
| NoChange

/-- Models a silver-zinc battery -/
structure SilverZincBattery where
  negativeElectrode : Electrode
  positiveElectrode : Electrode
  zincReaction : String
  silverOxideReaction : String
  currentFlow : CurrentFlow
  ohConcentrationChange : OHConcentrationChange

/-- Theorem about the properties of a silver-zinc battery -/
theorem silver_zinc_battery_properties (battery : SilverZincBattery) 
  (h1 : battery.zincReaction = "Zn + 2OH⁻ - 2e⁻ = Zn(OH)₂")
  (h2 : battery.silverOxideReaction = "Ag₂O + H₂O + 2e⁻ = 2Ag + 2OH⁻") :
  battery.negativeElectrode = Electrode.Zinc ∧
  battery.positiveElectrode = Electrode.SilverOxide ∧
  battery.ohConcentrationChange = OHConcentrationChange.Increase ∧
  battery.currentFlow = CurrentFlow.FromSilverOxideToZinc :=
sorry

end NUMINAMATH_CALUDE_silver_zinc_battery_properties_l646_64665


namespace NUMINAMATH_CALUDE_plane_perpendicular_sufficient_condition_l646_64655

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- Define the "contained in" relation for lines and planes
variable (containedIn : Line → Plane → Prop)

-- Define the intersecting relation for lines
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem plane_perpendicular_sufficient_condition
  (α β : Plane) (m n l₁ l₂ : Line)
  (h1 : m ≠ n)
  (h2 : containedIn m α)
  (h3 : containedIn n α)
  (h4 : containedIn l₁ β)
  (h5 : containedIn l₂ β)
  (h6 : intersect l₁ l₂)
  (h7 : perpendicular l₁ m)
  (h8 : perpendicular l₂ m) :
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicular_sufficient_condition_l646_64655


namespace NUMINAMATH_CALUDE_johns_former_wage_l646_64658

/-- Represents John's work schedule and wage information -/
structure WorkInfo where
  hours_per_workday : ℕ
  days_between_workdays : ℕ
  monthly_pay : ℕ
  days_in_month : ℕ
  raise_percentage : ℚ

/-- Calculates the former hourly wage given the work information -/
def former_hourly_wage (info : WorkInfo) : ℚ :=
  let days_worked := info.days_in_month / (info.days_between_workdays + 1)
  let total_hours := days_worked * info.hours_per_workday
  let current_hourly_wage := info.monthly_pay / total_hours
  current_hourly_wage / (1 + info.raise_percentage)

/-- Theorem stating that John's former hourly wage was $20 -/
theorem johns_former_wage (info : WorkInfo) 
  (h1 : info.hours_per_workday = 12)
  (h2 : info.days_between_workdays = 1)
  (h3 : info.monthly_pay = 4680)
  (h4 : info.days_in_month = 30)
  (h5 : info.raise_percentage = 3/10) :
  former_hourly_wage info = 20 := by
  sorry

end NUMINAMATH_CALUDE_johns_former_wage_l646_64658


namespace NUMINAMATH_CALUDE_geometric_sequence_difference_l646_64600

theorem geometric_sequence_difference (q : ℝ) (hq1 : q ≠ 1) (hq0 : q ≠ 0) :
  let a : ℕ → ℝ := λ n => q^(n-1)
  let b : ℕ → ℝ := λ n => a (n+1) - a n
  (∀ n : ℕ, b (n+1) / b n = q) ∧
  (b 1 = q - 1) ∧
  (∀ n : ℕ, b n = (q - 1) * q^(n-1)) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_difference_l646_64600


namespace NUMINAMATH_CALUDE_circle_properties_l646_64666

-- Define the line
def line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the first circle (the one we're proving)
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

-- Define what it means for a point to be on a circle
def on_circle (x y : ℝ) (circle : ℝ → ℝ → Prop) : Prop := circle x y

-- Define what it means for a line to be tangent to a circle
def is_tangent (circle : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), on_circle x y circle ∧ line x y ∧
  ∀ (x' y' : ℝ), line x' y' → (x' - x)^2 + (y' - y)^2 ≥ 0

-- Define what it means for two circles to intersect
def circles_intersect (circle1 circle2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), on_circle x y circle1 ∧ on_circle x y circle2

-- State the theorem
theorem circle_properties :
  is_tangent circle1 line ∧ circles_intersect circle1 circle2 := by sorry

end NUMINAMATH_CALUDE_circle_properties_l646_64666


namespace NUMINAMATH_CALUDE_constrained_words_count_l646_64679

/-- The number of possible letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- A five-letter word with the given constraints -/
structure ConstrainedWord :=
  (first : Fin alphabet_size)
  (second : Fin alphabet_size)
  (third : Fin alphabet_size)

/-- The total number of constrained words -/
def total_constrained_words : ℕ := alphabet_size ^ 3

theorem constrained_words_count :
  total_constrained_words = 17576 := by
  sorry

end NUMINAMATH_CALUDE_constrained_words_count_l646_64679


namespace NUMINAMATH_CALUDE_exercise_book_count_l646_64682

/-- Given a ratio of pencils to pens to exercise books and the number of pencils,
    calculate the number of exercise books. -/
theorem exercise_book_count (pencil_ratio : ℕ) (pen_ratio : ℕ) (book_ratio : ℕ) 
    (pencil_count : ℕ) (h1 : pencil_ratio = 14) (h2 : pen_ratio = 4) (h3 : book_ratio = 3) 
    (h4 : pencil_count = 140) : 
    (pencil_count * book_ratio) / pencil_ratio = 30 := by
  sorry

end NUMINAMATH_CALUDE_exercise_book_count_l646_64682


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l646_64619

theorem cube_volume_surface_area (x : ℝ) :
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) →
  x = Real.sqrt 3 / 72 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l646_64619


namespace NUMINAMATH_CALUDE_hyperbola_focus_l646_64672

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  (x - 5)^2 / 9^2 - (y - 20)^2 / 15^2 = 1

def is_focus (x y : ℝ) : Prop :=
  hyperbola_equation x y ∧ 
  ∃ (x' y' : ℝ), hyperbola_equation x' y' ∧ 
  (x - 5)^2 + (y - 20)^2 = (x' - 5)^2 + (y' - 20)^2 ∧ 
  (x ≠ x' ∨ y ≠ y')

theorem hyperbola_focus :
  ∃ (x y : ℝ), is_focus x y ∧ 
  (∀ (x' y' : ℝ), is_focus x' y' → x' ≤ x) ∧
  x = 5 + Real.sqrt 306 ∧ y = 20 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l646_64672


namespace NUMINAMATH_CALUDE_lcm_24_150_l646_64685

theorem lcm_24_150 : Nat.lcm 24 150 = 600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_150_l646_64685


namespace NUMINAMATH_CALUDE_fundraiser_total_l646_64620

def fundraiser (sasha_muffins : ℕ) (sasha_price : ℕ)
               (melissa_multiplier : ℕ) (melissa_price : ℕ)
               (tiffany_price : ℕ)
               (sarah_muffins : ℕ) (sarah_price : ℕ)
               (damien_dozens : ℕ) (damien_price : ℕ) : ℕ :=
  let melissa_muffins := melissa_multiplier * sasha_muffins
  let tiffany_muffins := (sasha_muffins + melissa_muffins) / 2
  let damien_muffins := damien_dozens * 12
  (sasha_muffins * sasha_price) +
  (melissa_muffins * melissa_price) +
  (tiffany_muffins * tiffany_price) +
  (sarah_muffins * sarah_price) +
  (damien_muffins * damien_price)

theorem fundraiser_total :
  fundraiser 30 4 4 3 5 50 2 2 6 = 1099 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_total_l646_64620


namespace NUMINAMATH_CALUDE_product_of_solutions_l646_64612

theorem product_of_solutions (x₁ x₂ : ℝ) 
  (h₁ : x₁ * Real.exp x₁ = Real.exp 2)
  (h₂ : x₂ * Real.log x₂ = Real.exp 2) :
  x₁ * x₂ = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l646_64612


namespace NUMINAMATH_CALUDE_arnold_protein_consumption_l646_64635

/-- Calculates the total protein consumed given the protein content of different food items. -/
def total_protein_consumed (collagen_protein_per_2_scoops : ℕ) (protein_powder_per_scoop : ℕ) (steak_protein : ℕ) : ℕ :=
  let collagen_protein := collagen_protein_per_2_scoops / 2
  collagen_protein + protein_powder_per_scoop + steak_protein

/-- Proves that the total protein consumed is 86 grams given the specific food items. -/
theorem arnold_protein_consumption : 
  total_protein_consumed 18 21 56 = 86 := by
  sorry

end NUMINAMATH_CALUDE_arnold_protein_consumption_l646_64635


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_l646_64647

/-- 
A function that checks if a quadratic expression x^2 + bx + c 
can be factored into two binomials with integer coefficients
-/
def is_factorizable (b : ℤ) (c : ℤ) : Prop :=
  ∃ (r s : ℤ), c = r * s ∧ b = r + s

/-- 
The smallest positive integer b for which x^2 + bx + 1890 
factors into a product of two binomials with integer coefficients
-/
theorem smallest_factorizable_b : 
  (∀ b : ℤ, b > 0 ∧ b < 141 → ¬(is_factorizable b 1890)) ∧ 
  (is_factorizable 141 1890) := by
  sorry

#check smallest_factorizable_b

end NUMINAMATH_CALUDE_smallest_factorizable_b_l646_64647


namespace NUMINAMATH_CALUDE_two_pairs_more_likely_than_three_of_a_kind_l646_64624

def num_dice : ℕ := 5
def faces_per_die : ℕ := 6

def total_outcomes : ℕ := faces_per_die ^ num_dice

def two_pairs_outcomes : ℕ := 
  num_dice * faces_per_die * (num_dice - 1).choose 2 * (faces_per_die - 1) * (faces_per_die - 2)

def three_of_a_kind_outcomes : ℕ := 
  num_dice.choose 3 * faces_per_die * (faces_per_die - 1) * (faces_per_die - 2)

theorem two_pairs_more_likely_than_three_of_a_kind :
  (two_pairs_outcomes : ℚ) / total_outcomes > (three_of_a_kind_outcomes : ℚ) / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_two_pairs_more_likely_than_three_of_a_kind_l646_64624


namespace NUMINAMATH_CALUDE_cubic_root_difference_l646_64628

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + a

/-- The derivative of the cubic function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

theorem cubic_root_difference (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  (∃! (x₁ x₂ : ℝ), f a x₁ = 0 ∧ f a x₂ = 0) →
  x₂ - x₁ = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_difference_l646_64628


namespace NUMINAMATH_CALUDE_cubic_equation_one_root_l646_64662

theorem cubic_equation_one_root (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ x^3 - a*x^2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_one_root_l646_64662


namespace NUMINAMATH_CALUDE_haleigh_leggings_needed_l646_64689

/-- The number of pairs of leggings needed for pets -/
def leggings_needed (num_dogs : ℕ) (num_cats : ℕ) (legs_per_animal : ℕ) (legs_per_legging : ℕ) : ℕ :=
  ((num_dogs + num_cats) * legs_per_animal) / legs_per_legging

/-- Theorem: Haleigh needs 14 pairs of leggings for her pets -/
theorem haleigh_leggings_needed :
  leggings_needed 4 3 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_haleigh_leggings_needed_l646_64689


namespace NUMINAMATH_CALUDE_project_completion_time_l646_64699

theorem project_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let time_together := (a * b) / (a + b)
  time_together > 0 ∧ 
  (1 / a + 1 / b) * time_together = 1 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l646_64699


namespace NUMINAMATH_CALUDE_car_distance_proof_l646_64663

/-- Calculates the total distance traveled by a car over a given number of hours,
    where the car's speed increases by a fixed amount each hour. -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc h => acc + initialSpeed + h * speedIncrease) 0

/-- Proves that a car traveling 45 km in the first hour and increasing speed by 2 km/h
    each hour will travel 672 km in 12 hours. -/
theorem car_distance_proof :
  totalDistance 45 2 12 = 672 := by
  sorry

#eval totalDistance 45 2 12

end NUMINAMATH_CALUDE_car_distance_proof_l646_64663


namespace NUMINAMATH_CALUDE_test_questions_count_l646_64681

theorem test_questions_count :
  ∀ (total_questions : ℕ),
    total_questions % 5 = 0 →
    32 > (70 * total_questions) / 100 →
    32 < (77 * total_questions) / 100 →
    total_questions = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l646_64681


namespace NUMINAMATH_CALUDE_multiple_problem_l646_64644

theorem multiple_problem (n m : ℝ) : n = 5 → m * n - 15 = 2 * n + 10 → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l646_64644


namespace NUMINAMATH_CALUDE_percentage_problem_l646_64625

theorem percentage_problem (x : ℝ) :
  (0.15 * 0.30 * 0.50 * x = 117) → (x = 5200) :=
by sorry

end NUMINAMATH_CALUDE_percentage_problem_l646_64625


namespace NUMINAMATH_CALUDE_remainder_n_squared_plus_2n_plus_3_l646_64664

theorem remainder_n_squared_plus_2n_plus_3 (n : ℤ) (a : ℤ) (h : n = 100 * a - 1) :
  (n^2 + 2*n + 3) % 100 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_n_squared_plus_2n_plus_3_l646_64664


namespace NUMINAMATH_CALUDE_smallest_c_for_max_at_zero_l646_64680

/-- Given a function y = a * cos(b * x + c) where a, b, and c are positive constants,
    and the graph reaches a maximum at x = 0, the smallest possible value of c is 0. -/
theorem smallest_c_for_max_at_zero (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≤ a * Real.cos c) →
  (∀ ε > 0, ∃ x, a * Real.cos (b * x + (c - ε)) > a * Real.cos (c - ε)) →
  c = 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_max_at_zero_l646_64680


namespace NUMINAMATH_CALUDE_john_card_expenditure_l646_64608

/-- The number of thank you cards John sent for Christmas gifts -/
def christmas_cards : ℕ := 20

/-- The number of thank you cards John sent for birthday gifts -/
def birthday_cards : ℕ := 15

/-- The cost of each thank you card in dollars -/
def card_cost : ℕ := 2

/-- The total cost of all thank you cards John bought -/
def total_cost : ℕ := (christmas_cards + birthday_cards) * card_cost

theorem john_card_expenditure :
  total_cost = 70 := by sorry

end NUMINAMATH_CALUDE_john_card_expenditure_l646_64608


namespace NUMINAMATH_CALUDE_range_of_m_l646_64639

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_domain : ∀ x, f x ≠ 0 → x ∈ Set.Icc (-2 : ℝ) 2
axiom f_decreasing : ∀ x y, x < y ∧ x ∈ Set.Icc (-2 : ℝ) 0 → f x > f y

-- Define the inequality condition
def inequality_condition (m : ℝ) : Prop := f (1 - m) + f (1 - m^2) < 0

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, inequality_condition m ↔ m ∈ Set.Icc (-1 : ℝ) 1 ∧ m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l646_64639


namespace NUMINAMATH_CALUDE_angle_sum_pi_half_l646_64660

theorem angle_sum_pi_half (θ₁ θ₂ : Real) (h_acute₁ : 0 < θ₁ ∧ θ₁ < π/2) (h_acute₂ : 0 < θ₂ ∧ θ₂ < π/2)
  (h_eq : (Real.sin θ₁)^2020 / (Real.cos θ₂)^2018 + (Real.cos θ₁)^2020 / (Real.sin θ₂)^2018 = 1) :
  θ₁ + θ₂ = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_pi_half_l646_64660


namespace NUMINAMATH_CALUDE_max_value_sum_of_fractions_l646_64640

theorem max_value_sum_of_fractions (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_fractions_l646_64640


namespace NUMINAMATH_CALUDE_tournament_matches_divisible_by_two_l646_64677

/-- Represents a single elimination tennis tournament -/
structure TennisTournament where
  total_players : ℕ
  bye_players : ℕ
  first_round_players : ℕ

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : TennisTournament) : ℕ :=
  t.total_players - 1

/-- Theorem: The total number of matches in the specified tournament is divisible by 2 -/
theorem tournament_matches_divisible_by_two :
  ∃ (t : TennisTournament), 
    t.total_players = 128 ∧ 
    t.bye_players = 32 ∧ 
    t.first_round_players = 96 ∧ 
    ∃ (k : ℕ), total_matches t = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_tournament_matches_divisible_by_two_l646_64677


namespace NUMINAMATH_CALUDE_parallel_linear_function_b_value_l646_64656

/-- A linear function y = kx + b whose graph is parallel to y = 3x and passes through (1, -1) has b = -4 -/
theorem parallel_linear_function_b_value (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b) →  -- Definition of linear function
  k = 3 →  -- Parallel to y = 3x
  -1 = k * 1 + b →  -- Passes through (1, -1)
  b = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_linear_function_b_value_l646_64656


namespace NUMINAMATH_CALUDE_root_of_inverse_point_l646_64691

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverse functions
variable (h_inverse : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x)

-- Assume f_inv(0) = 2
variable (h_f_inv_zero : f_inv 0 = 2)

-- Theorem: If f_inv(0) = 2, then f(2) = 0
theorem root_of_inverse_point (f f_inv : ℝ → ℝ) 
  (h_inverse : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x) 
  (h_f_inv_zero : f_inv 0 = 2) : 
  f 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_root_of_inverse_point_l646_64691


namespace NUMINAMATH_CALUDE_multiplication_correction_l646_64643

theorem multiplication_correction (a b c d e : ℕ) : 
  (a = 4 ∧ b = 5 ∧ c = 4 ∧ d = 5 ∧ e = 4) →
  (a * b * c * d * e = 2247) →
  ∃ (x : ℕ), (x = 5 ∨ x = 7) ∧
    ((4 * x * 4 * 5 * 4 = 2240) ∨ (4 * 5 * 4 * x * 4 = 2240)) :=
by sorry

end NUMINAMATH_CALUDE_multiplication_correction_l646_64643


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l646_64638

/-- The total surface area of a cylinder with height 15 cm and radius 5 cm is 200π square cm. -/
theorem cylinder_surface_area :
  let h : ℝ := 15
  let r : ℝ := 5
  let total_area : ℝ := 2 * Real.pi * r * r + 2 * Real.pi * r * h
  total_area = 200 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l646_64638


namespace NUMINAMATH_CALUDE_yulia_lemonade_stand_expenses_l646_64617

/-- Yulia's lemonade stand financial calculation -/
theorem yulia_lemonade_stand_expenses 
  (net_profit : ℝ) 
  (lemonade_revenue : ℝ) 
  (babysitting_earnings : ℝ) 
  (h1 : net_profit = 44)
  (h2 : lemonade_revenue = 47)
  (h3 : babysitting_earnings = 31) :
  lemonade_revenue + babysitting_earnings - net_profit = 34 :=
by
  sorry

#check yulia_lemonade_stand_expenses

end NUMINAMATH_CALUDE_yulia_lemonade_stand_expenses_l646_64617


namespace NUMINAMATH_CALUDE_find_k_value_l646_64609

theorem find_k_value (k : ℝ) (h1 : k ≠ 0) 
  (h2 : ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 - k * (x^2 + x + 3)) : 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l646_64609


namespace NUMINAMATH_CALUDE_division_result_approx_point_zero_seven_l646_64673

-- Define the approximation tolerance
def tolerance : ℝ := 0.001

-- Define the condition that 35 divided by x is approximately 500
def divisionApprox (x : ℝ) : Prop := 
  abs (35 / x - 500) < tolerance

-- Theorem statement
theorem division_result_approx_point_zero_seven :
  ∃ x : ℝ, divisionApprox x ∧ abs (x - 0.07) < tolerance :=
sorry

end NUMINAMATH_CALUDE_division_result_approx_point_zero_seven_l646_64673


namespace NUMINAMATH_CALUDE_caravan_keeper_count_l646_64674

/-- Represents the number of keepers in the caravan -/
def num_keepers : ℕ := 10

/-- Represents the number of hens in the caravan -/
def num_hens : ℕ := 60

/-- Represents the number of goats in the caravan -/
def num_goats : ℕ := 35

/-- Represents the number of camels in the caravan -/
def num_camels : ℕ := 6

/-- Represents the number of feet for a hen -/
def hen_feet : ℕ := 2

/-- Represents the number of feet for a goat or camel -/
def goat_camel_feet : ℕ := 4

/-- Represents the number of feet for a keeper -/
def keeper_feet : ℕ := 2

/-- Represents the difference between total feet and total heads -/
def feet_head_difference : ℕ := 193

theorem caravan_keeper_count :
  num_keepers * keeper_feet +
  num_hens * hen_feet +
  num_goats * goat_camel_feet +
  num_camels * goat_camel_feet =
  (num_keepers + num_hens + num_goats + num_camels + feet_head_difference) :=
by sorry

end NUMINAMATH_CALUDE_caravan_keeper_count_l646_64674


namespace NUMINAMATH_CALUDE_f_is_locally_odd_l646_64641

/-- Definition of a locally odd function -/
def LocallyOdd (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

/-- The quadratic function we're examining -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 4 * a

/-- Theorem: The function f is locally odd for any real a -/
theorem f_is_locally_odd (a : ℝ) : LocallyOdd (f a) := by
  sorry


end NUMINAMATH_CALUDE_f_is_locally_odd_l646_64641


namespace NUMINAMATH_CALUDE_train_speed_l646_64675

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 400) (h2 : time = 10) :
  length / time = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l646_64675


namespace NUMINAMATH_CALUDE_alyssas_allowance_proof_l646_64697

/-- Alyssa's weekly allowance in dollars -/
def weekly_allowance : ℝ := 240

theorem alyssas_allowance_proof :
  ∃ (a : ℝ),
    a > 0 ∧
    a / 2 + a / 5 + a / 4 + 12 = a ∧
    a = weekly_allowance := by
  sorry

end NUMINAMATH_CALUDE_alyssas_allowance_proof_l646_64697


namespace NUMINAMATH_CALUDE_phone_call_probability_l646_64690

/-- The probability of answering a phone call at the first ring -/
def p_first : ℝ := 0.1

/-- The probability of answering a phone call at the second ring -/
def p_second : ℝ := 0.3

/-- The probability of answering a phone call at the third ring -/
def p_third : ℝ := 0.4

/-- The probability of answering a phone call at the fourth ring -/
def p_fourth : ℝ := 0.1

/-- The events of answering at each ring are mutually exclusive -/
axiom mutually_exclusive : True

/-- The probability of answering within the first four rings -/
def p_within_four : ℝ := p_first + p_second + p_third + p_fourth

theorem phone_call_probability :
  p_within_four = 0.9 :=
sorry

end NUMINAMATH_CALUDE_phone_call_probability_l646_64690


namespace NUMINAMATH_CALUDE_min_value_exponential_function_l646_64670

theorem min_value_exponential_function (x : ℝ) : Real.exp x + 4 * Real.exp (-x) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_function_l646_64670


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_count_l646_64695

theorem absolute_value_equation_solution_count :
  ∃! (s : Finset ℤ), (∀ a ∈ s, |3*a+7| + |3*a-5| = 12) ∧ s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_count_l646_64695


namespace NUMINAMATH_CALUDE_percent_problem_l646_64631

theorem percent_problem (x : ℝ) : (0.15 * 40 = 0.25 * x + 2) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l646_64631


namespace NUMINAMATH_CALUDE_march_production_l646_64653

/-- Represents the monthly production function -/
def production_function (x : ℝ) : ℝ := x + 1

/-- March is represented by the number 3 -/
def march : ℝ := 3

/-- Theorem stating that the estimated production for March is 4 -/
theorem march_production :
  production_function march = 4 := by sorry

end NUMINAMATH_CALUDE_march_production_l646_64653


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l646_64632

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ n - 17 ≠ 0 ∧ 7*n + 8 ≠ 0 ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (7*n + 8)) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n →
    m - 17 = 0 ∨ 7*m + 8 = 0 ∨
    (∀ (j : ℕ), j > 1 → ¬(j ∣ (m - 17) ∧ j ∣ (7*m + 8)))) ∧
  n = 144 :=
by sorry


end NUMINAMATH_CALUDE_least_reducible_fraction_l646_64632


namespace NUMINAMATH_CALUDE_lottery_probability_l646_64645

def megaball_count : ℕ := 30
def winnerball_count : ℕ := 50
def ordered_winnerball_count : ℕ := 2
def unordered_winnerball_count : ℕ := 5

def megaball_prob : ℚ := 1 / megaball_count
def ordered_winnerball_prob : ℚ := 1 / (winnerball_count * (winnerball_count - 1))
def unordered_winnerball_prob : ℚ := 1 / (Nat.choose (winnerball_count - ordered_winnerball_count) unordered_winnerball_count)

theorem lottery_probability :
  megaball_prob * ordered_winnerball_prob * unordered_winnerball_prob = 1 / 125703480000 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l646_64645


namespace NUMINAMATH_CALUDE_natasha_speed_over_limit_l646_64650

/-- Proves that Natasha was going 10 mph over the speed limit -/
theorem natasha_speed_over_limit (distance : ℝ) (time : ℝ) (speed_limit : ℝ) :
  distance = 60 ∧ time = 1 ∧ speed_limit = 50 →
  distance / time - speed_limit = 10 := by
  sorry

end NUMINAMATH_CALUDE_natasha_speed_over_limit_l646_64650


namespace NUMINAMATH_CALUDE_kibble_remaining_is_seven_l646_64607

/-- The amount of kibble remaining in Luna's bag after one day of feeding. -/
def kibble_remaining (initial_amount : ℕ) (mary_morning : ℕ) (mary_evening : ℕ) (frank_afternoon : ℕ) : ℕ :=
  initial_amount - (mary_morning + mary_evening + frank_afternoon + 2 * frank_afternoon)

/-- Theorem stating that the amount of kibble remaining is 7 cups. -/
theorem kibble_remaining_is_seven :
  kibble_remaining 12 1 1 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_kibble_remaining_is_seven_l646_64607


namespace NUMINAMATH_CALUDE_simplify_fraction_l646_64627

theorem simplify_fraction : 15 * (18 / 11) * (-42 / 45) = -23 - (1 / 11) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l646_64627


namespace NUMINAMATH_CALUDE_add_point_four_to_fifty_six_point_seven_l646_64694

theorem add_point_four_to_fifty_six_point_seven :
  0.4 + 56.7 = 57.1 := by sorry

end NUMINAMATH_CALUDE_add_point_four_to_fifty_six_point_seven_l646_64694


namespace NUMINAMATH_CALUDE_constant_sum_of_powers_l646_64603

/-- S_n is constant for real x, y, z with xyz = 1 and x + y + z = 0 iff n = 1 or n = 3 -/
theorem constant_sum_of_powers (n : ℕ+) :
  (∀ x y z : ℝ, x * y * z = 1 → x + y + z = 0 → 
    ∃ c : ℝ, ∀ x' y' z' : ℝ, x' * y' * z' = 1 → x' + y' + z' = 0 → 
      x'^(n : ℕ) + y'^(n : ℕ) + z'^(n : ℕ) = c) ↔ 
  n = 1 ∨ n = 3 := by
sorry

end NUMINAMATH_CALUDE_constant_sum_of_powers_l646_64603


namespace NUMINAMATH_CALUDE_max_sequence_length_l646_64683

theorem max_sequence_length (x : ℕ) : 
  (68000 - 55 * x > 0) ∧ (34 * x - 42000 > 0) ↔ x = 1236 :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_l646_64683


namespace NUMINAMATH_CALUDE_solution_value_l646_64605

theorem solution_value (a : ℝ) : (∃ x : ℝ, x = 1 ∧ a * x + 2 * x = 3) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l646_64605


namespace NUMINAMATH_CALUDE_cubic_three_roots_l646_64606

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x - a

/-- The derivative of f with respect to x -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The second derivative of f with respect to x -/
def f'' (x : ℝ) : ℝ := 6*x

/-- The value of f at x = 1 -/
def f_at_1 (a : ℝ) : ℝ := -2 - a

/-- The value of f at x = -1 -/
def f_at_neg_1 (a : ℝ) : ℝ := 2 - a

/-- Theorem: The cubic function f(x) = x^3 - 3x - a has three distinct real roots 
    if and only if a is in the open interval (-2, 2) -/
theorem cubic_three_roots (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ 
  -2 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_roots_l646_64606


namespace NUMINAMATH_CALUDE_range_of_m_l646_64688

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 8 * x / y) > m^2 + 2*m) → 
  -4 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l646_64688


namespace NUMINAMATH_CALUDE_goldfish_preference_total_l646_64678

/-- Calculates the total number of students preferring goldfish across three classes -/
theorem goldfish_preference_total (class_size : ℕ) 
  (johnson_fraction : ℚ) (feldstein_fraction : ℚ) (henderson_fraction : ℚ)
  (h1 : class_size = 30)
  (h2 : johnson_fraction = 1 / 6)
  (h3 : feldstein_fraction = 2 / 3)
  (h4 : henderson_fraction = 1 / 5) :
  ⌊class_size * johnson_fraction⌋ + ⌊class_size * feldstein_fraction⌋ + ⌊class_size * henderson_fraction⌋ = 31 :=
by sorry

end NUMINAMATH_CALUDE_goldfish_preference_total_l646_64678


namespace NUMINAMATH_CALUDE_ideal_gas_entropy_change_l646_64623

/-- Entropy change for an ideal gas under different conditions -/
theorem ideal_gas_entropy_change
  (m μ R Cp Cv : ℝ)
  (P V T P1 P2 V1 V2 T1 T2 : ℝ)
  (h_ideal_gas : P * V = (m / μ) * R * T)
  (h_m_pos : m > 0)
  (h_μ_pos : μ > 0)
  (h_R_pos : R > 0)
  (h_Cp_pos : Cp > 0)
  (h_Cv_pos : Cv > 0)
  (h_P_pos : P > 0)
  (h_V_pos : V > 0)
  (h_T_pos : T > 0)
  (h_P1_pos : P1 > 0)
  (h_P2_pos : P2 > 0)
  (h_V1_pos : V1 > 0)
  (h_V2_pos : V2 > 0)
  (h_T1_pos : T1 > 0)
  (h_T2_pos : T2 > 0) :
  (∃ ΔS : ℝ,
    (P = P1 ∧ P = P2 → ΔS = (m / μ) * Cp * Real.log (V2 / V1)) ∧
    (V = V1 ∧ V = V2 → ΔS = (m / μ) * Cv * Real.log (P2 / P1)) ∧
    (T = T1 ∧ T = T2 → ΔS = (m / μ) * R * Real.log (V2 / V1))) :=
by sorry

end NUMINAMATH_CALUDE_ideal_gas_entropy_change_l646_64623


namespace NUMINAMATH_CALUDE_function_extension_l646_64602

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem function_extension (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_symmetry : ∀ x, f (2 - x) = f x)
  (h_base : ∀ x ∈ Set.Ioo 0 1, f x = Real.log x) :
  (∀ x ∈ Set.Icc (-1) 0, f x = -Real.log (-x)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Ioo (4 * k) (4 * k + 1), f x = Real.log (x - 4 * k)) :=
sorry

end NUMINAMATH_CALUDE_function_extension_l646_64602


namespace NUMINAMATH_CALUDE_well_depth_equation_l646_64613

theorem well_depth_equation (d : ℝ) (u : ℝ) (h : u = Real.sqrt d) : 
  d = 14 * (10 - d / 1200)^2 → 14 * u^2 + 1200 * u - 12000 * Real.sqrt 14 = 0 := by
  sorry

#check well_depth_equation

end NUMINAMATH_CALUDE_well_depth_equation_l646_64613


namespace NUMINAMATH_CALUDE_min_value_implies_a_l646_64651

/-- The function f(x) defined as |x+1| + |2x+a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |2*x + a|

/-- The theorem stating that if the minimum value of f(x) is 3, then a = -4 or a = 8 -/
theorem min_value_implies_a (a : ℝ) : (∀ x, f a x ≥ 3) ∧ (∃ x, f a x = 3) → a = -4 ∨ a = 8 := by
  sorry


end NUMINAMATH_CALUDE_min_value_implies_a_l646_64651
