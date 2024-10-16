import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequalities_l3558_355816

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set condition
def solution_set (a b c : ℝ) : Set ℝ := {x : ℝ | x ≤ -2 ∨ x ≥ 3}

-- Define the theorem
theorem quadratic_inequalities 
  (a b c : ℝ) 
  (h : ∀ x, x ∈ solution_set a b c ↔ f a b c x ≤ 0) :
  (a < 0) ∧ 
  ({x : ℝ | a * x + c > 0} = {x : ℝ | x < 6}) ∧
  ({x : ℝ | c * x^2 + b * x + a < 0} = {x : ℝ | -1/2 < x ∧ x < 1/3}) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l3558_355816


namespace NUMINAMATH_CALUDE_employee_payment_proof_l3558_355811

/-- The weekly payment for employee B -/
def payment_B : ℝ := 180

/-- The weekly payment for employee A -/
def payment_A : ℝ := 1.5 * payment_B

/-- The total weekly payment for both employees -/
def total_payment : ℝ := 450

theorem employee_payment_proof :
  payment_A + payment_B = total_payment :=
by sorry

end NUMINAMATH_CALUDE_employee_payment_proof_l3558_355811


namespace NUMINAMATH_CALUDE_min_interval_number_bound_l3558_355899

/-- Represents a football tournament schedule -/
structure TournamentSchedule (n : ℕ) where
  -- n is the number of teams
  teams : Fin n
  -- schedule is a list of pairs of teams representing matches
  schedule : List (Fin n × Fin n)
  -- Each pair of teams plays exactly one match
  one_match : ∀ i j, i ≠ j → (i, j) ∈ schedule ∨ (j, i) ∈ schedule
  -- One match is scheduled each day
  one_per_day : schedule.length = (n.choose 2)

/-- The interval number between two matches of a team -/
def intervalNumber (s : TournamentSchedule n) (team : Fin n) : ℕ → ℕ → ℕ :=
  sorry

/-- The minimum interval number for a given schedule -/
def minIntervalNumber (s : TournamentSchedule n) : ℕ :=
  sorry

/-- Theorem: The minimum interval number does not exceed ⌊(n-3)/2⌋ -/
theorem min_interval_number_bound {n : ℕ} (hn : n ≥ 5) (s : TournamentSchedule n) :
  minIntervalNumber s ≤ (n - 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_interval_number_bound_l3558_355899


namespace NUMINAMATH_CALUDE_forty_five_million_scientific_notation_l3558_355890

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem forty_five_million_scientific_notation :
  toScientificNotation 45000000 = ScientificNotation.mk 4.5 7 sorry := by sorry

end NUMINAMATH_CALUDE_forty_five_million_scientific_notation_l3558_355890


namespace NUMINAMATH_CALUDE_students_with_both_fruits_l3558_355837

theorem students_with_both_fruits (apples bananas only_one : ℕ) 
  (h1 : apples = 12)
  (h2 : bananas = 8)
  (h3 : only_one = 10) :
  apples + bananas - only_one = 5 := by
  sorry

end NUMINAMATH_CALUDE_students_with_both_fruits_l3558_355837


namespace NUMINAMATH_CALUDE_unique_albums_count_l3558_355846

/-- Represents the album collections of Andrew, John, and Bella -/
structure AlbumCollections where
  andrew_total : ℕ
  andrew_john_shared : ℕ
  john_unique : ℕ
  bella_andrew_overlap : ℕ

/-- Calculates the number of unique albums not shared among any two people -/
def unique_albums (collections : AlbumCollections) : ℕ :=
  (collections.andrew_total - collections.andrew_john_shared) + collections.john_unique

/-- Theorem stating that the number of unique albums is 18 given the problem conditions -/
theorem unique_albums_count (collections : AlbumCollections)
  (h1 : collections.andrew_total = 20)
  (h2 : collections.andrew_john_shared = 10)
  (h3 : collections.john_unique = 8)
  (h4 : collections.bella_andrew_overlap = 5)
  (h5 : collections.bella_andrew_overlap ≤ collections.andrew_total - collections.andrew_john_shared) :
  unique_albums collections = 18 := by
  sorry

#eval unique_albums { andrew_total := 20, andrew_john_shared := 10, john_unique := 8, bella_andrew_overlap := 5 }

end NUMINAMATH_CALUDE_unique_albums_count_l3558_355846


namespace NUMINAMATH_CALUDE_sector_central_angle_l3558_355835

-- Define the sector
structure Sector where
  perimeter : ℝ
  area : ℝ

-- Theorem statement
theorem sector_central_angle (s : Sector) (h1 : s.perimeter = 6) (h2 : s.area = 2) :
  ∃ θ : ℝ, (θ = 1 ∨ θ = 4) ∧ 
  (∃ r : ℝ, r > 0 ∧ θ * r + 2 * r = s.perimeter ∧ 1/2 * r^2 * θ = s.area) :=
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3558_355835


namespace NUMINAMATH_CALUDE_sum_of_possible_x_values_l3558_355882

theorem sum_of_possible_x_values (x : ℝ) (h : |x - 12| = 100) : 
  ∃ (x₁ x₂ : ℝ), |x₁ - 12| = 100 ∧ |x₂ - 12| = 100 ∧ x₁ + x₂ = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_values_l3558_355882


namespace NUMINAMATH_CALUDE_min_tiles_to_cover_classroom_l3558_355874

/-- Represents the dimensions of a rectangular area in centimeters -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Represents a tile with its area in square centimeters -/
structure Tile where
  area : ℕ

def classroom : Dimensions := ⟨624, 432⟩
def rectangular_tile : Tile := ⟨60 * 80⟩
def triangular_tile : Tile := ⟨40 * 40 / 2⟩

def tiles_needed (room : Dimensions) (tile : Tile) : ℕ :=
  (room.length * room.width + tile.area - 1) / tile.area

theorem min_tiles_to_cover_classroom :
  min (tiles_needed classroom rectangular_tile) (tiles_needed classroom triangular_tile) = 57 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_to_cover_classroom_l3558_355874


namespace NUMINAMATH_CALUDE_batsman_score_l3558_355880

theorem batsman_score (T : ℝ) : 
  (5 * 4 + 5 * 6 : ℝ) + (2/3) * T = T → T = 150 := by sorry

end NUMINAMATH_CALUDE_batsman_score_l3558_355880


namespace NUMINAMATH_CALUDE_point_coordinates_l3558_355861

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance from a point to x-axis and y-axis
def distToXAxis (p : Point2D) : ℝ := |p.y|
def distToYAxis (p : Point2D) : ℝ := |p.x|

-- Define the set of possible coordinates
def possibleCoordinates : Set Point2D :=
  {⟨2, 1⟩, ⟨2, -1⟩, ⟨-2, 1⟩, ⟨-2, -1⟩}

-- Theorem statement
theorem point_coordinates (M : Point2D) :
  distToXAxis M = 1 ∧ distToYAxis M = 2 → M ∈ possibleCoordinates := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3558_355861


namespace NUMINAMATH_CALUDE_situps_problem_l3558_355869

/-- Situps problem -/
theorem situps_problem (diana_rate hani_rate total_situps : ℕ) 
  (h1 : hani_rate = diana_rate + 3)
  (h2 : diana_rate = 4)
  (h3 : total_situps = 110) :
  diana_rate * (total_situps / (diana_rate + hani_rate)) = 40 := by
  sorry

#check situps_problem

end NUMINAMATH_CALUDE_situps_problem_l3558_355869


namespace NUMINAMATH_CALUDE_expected_pairs_value_l3558_355836

/-- The number of boys in the lineup -/
def num_boys : ℕ := 9

/-- The number of girls in the lineup -/
def num_girls : ℕ := 15

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of adjacent pairs in the lineup -/
def num_pairs : ℕ := total_people - 1

/-- The probability of a boy-girl or girl-boy pair at any given adjacent position -/
def pair_probability : ℚ := 
  (num_boys * num_girls + num_girls * num_boys) / (total_people * (total_people - 1))

/-- The expected number of boy-girl or girl-boy pairs in a random permutation -/
def expected_pairs : ℚ := num_pairs * pair_probability

theorem expected_pairs_value : expected_pairs = 3105 / 276 := by sorry

end NUMINAMATH_CALUDE_expected_pairs_value_l3558_355836


namespace NUMINAMATH_CALUDE_inequality_proof_l3558_355831

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3558_355831


namespace NUMINAMATH_CALUDE_equation_solution_l3558_355892

theorem equation_solution (x : ℝ) : 
  (24 : ℝ) / 36 = Real.sqrt (x / 36) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3558_355892


namespace NUMINAMATH_CALUDE_worker_delay_l3558_355889

/-- Proves that reducing speed to 5/6 of normal results in a 12-minute delay -/
theorem worker_delay (usual_time : ℝ) (speed_ratio : ℝ) 
  (h1 : usual_time = 60)
  (h2 : speed_ratio = 5 / 6) : 
  (usual_time / speed_ratio) - usual_time = 12 := by
  sorry

#check worker_delay

end NUMINAMATH_CALUDE_worker_delay_l3558_355889


namespace NUMINAMATH_CALUDE_total_problems_l3558_355818

def math_pages : ℕ := 6
def reading_pages : ℕ := 4
def problems_per_page : ℕ := 3

theorem total_problems : math_pages + reading_pages * problems_per_page = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_l3558_355818


namespace NUMINAMATH_CALUDE_first_expression_value_l3558_355817

theorem first_expression_value (a : ℝ) (E : ℝ) : 
  a = 26 → (E + (3 * a - 8)) / 2 = 69 → E = 68 := by sorry

end NUMINAMATH_CALUDE_first_expression_value_l3558_355817


namespace NUMINAMATH_CALUDE_man_rowing_speed_l3558_355809

/-- Calculates the speed of a man rowing upstream given his speed in still water and downstream speed -/
def speed_upstream (speed_still : ℝ) (speed_downstream : ℝ) : ℝ :=
  2 * speed_still - speed_downstream

/-- Theorem stating that given a man's speed in still water is 32 kmph and his speed downstream is 42 kmph, his speed upstream is 22 kmph -/
theorem man_rowing_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 32) 
  (h2 : speed_downstream = 42) : 
  speed_upstream speed_still speed_downstream = 22 := by
  sorry

#eval speed_upstream 32 42

end NUMINAMATH_CALUDE_man_rowing_speed_l3558_355809


namespace NUMINAMATH_CALUDE_solve_equation_l3558_355805

theorem solve_equation : ∃! y : ℚ, 2 * y + 3 * y = 500 - (4 * y + 5 * y) ∧ y = 250 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3558_355805


namespace NUMINAMATH_CALUDE_complex_cube_root_magnitude_l3558_355878

theorem complex_cube_root_magnitude (w : ℂ) (h : w^3 = 64 - 48*I) : 
  Complex.abs w = 2 * Real.rpow 10 (1/3) := by
sorry

end NUMINAMATH_CALUDE_complex_cube_root_magnitude_l3558_355878


namespace NUMINAMATH_CALUDE_circle_radius_range_equivalence_l3558_355849

/-- A circle in a 2D Cartesian coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle has exactly two points at distance 1 from x-axis -/
def has_two_points_at_distance_one (c : Circle) : Prop :=
  ∃ (p1 p2 : ℝ × ℝ),
    (p1 ≠ p2) ∧
    (p1.1 - c.center.1)^2 + (p1.2 - c.center.2)^2 = c.radius^2 ∧
    (p2.1 - c.center.1)^2 + (p2.2 - c.center.2)^2 = c.radius^2 ∧
    (abs p1.2 = 1 ∨ abs p2.2 = 1) ∧
    (∀ (p : ℝ × ℝ), 
      (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 → 
      abs p.2 = 1 → (p = p1 ∨ p = p2))

/-- The main theorem stating the equivalence -/
theorem circle_radius_range_equivalence :
  ∀ (c : Circle),
    c.center = (3, -5) →
    (has_two_points_at_distance_one c ↔ (4 < c.radius ∧ c.radius < 6)) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_range_equivalence_l3558_355849


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l3558_355886

theorem solution_satisfies_equations :
  ∃ (j k : ℚ), 5 * j - 42 * k = 1 ∧ 2 * k - j = 3 ∧ j = -4 ∧ k = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l3558_355886


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_l3558_355885

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  focus1 : Point
  focus2 : Point
  passingPoint : Point

/-- Checks if a point satisfies the ellipse equation -/
def satisfiesEllipseEquation (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.h)^2 / e.a^2 + (p.y - e.k)^2 / e.b^2 = 1

/-- The main theorem -/
theorem ellipse_a_plus_k (e : Ellipse) : 
  e.focus1 = ⟨1, 1⟩ →
  e.focus2 = ⟨1, 3⟩ →
  e.passingPoint = ⟨-4, 2⟩ →
  e.a > 0 →
  e.b > 0 →
  satisfiesEllipseEquation e e.passingPoint →
  e.a + e.k = 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_l3558_355885


namespace NUMINAMATH_CALUDE_zeta_sum_seventh_power_l3558_355879

theorem zeta_sum_seventh_power (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 1)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 3)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 7) :
  ζ₁^7 + ζ₂^7 + ζ₃^7 = 71 := by
  sorry

end NUMINAMATH_CALUDE_zeta_sum_seventh_power_l3558_355879


namespace NUMINAMATH_CALUDE_initial_gifts_count_l3558_355865

/-- The number of gifts sent to the orphanage -/
def gifts_sent : ℕ := 66

/-- The number of gifts left under the tree -/
def gifts_left : ℕ := 11

/-- The initial number of gifts -/
def initial_gifts : ℕ := gifts_sent + gifts_left

theorem initial_gifts_count : initial_gifts = 77 := by
  sorry

end NUMINAMATH_CALUDE_initial_gifts_count_l3558_355865


namespace NUMINAMATH_CALUDE_chocolate_cost_l3558_355858

theorem chocolate_cost (candy_cost : ℕ) (candy_count : ℕ) (chocolate_count : ℕ) (price_difference : ℕ) :
  candy_cost = 530 →
  candy_count = 12 →
  chocolate_count = 8 →
  price_difference = 5400 →
  candy_count * candy_cost = chocolate_count * (candy_count * candy_cost / chocolate_count - price_difference / chocolate_count) + price_difference →
  candy_count * candy_cost / chocolate_count - price_difference / chocolate_count = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_l3558_355858


namespace NUMINAMATH_CALUDE_cow_to_horse_ratio_l3558_355823

def total_animals : ℕ := 168
def num_cows : ℕ := 140

theorem cow_to_horse_ratio :
  let num_horses := total_animals - num_cows
  num_cows / num_horses = 5 := by
sorry

end NUMINAMATH_CALUDE_cow_to_horse_ratio_l3558_355823


namespace NUMINAMATH_CALUDE_zeros_imply_sum_l3558_355847

/-- A quadratic function with zeros at -2 and 3 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- Theorem stating that if f has zeros at -2 and 3, then a + b = -7 -/
theorem zeros_imply_sum (a b : ℝ) :
  f a b (-2) = 0 ∧ f a b 3 = 0 → a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_zeros_imply_sum_l3558_355847


namespace NUMINAMATH_CALUDE_percentage_decrease_of_b_l3558_355868

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 → 
  b > 0 → 
  a / b = 4 / 5 → 
  x = a * 1.25 → 
  m = b * (1 - p / 100) → 
  m / x = 0.6 → 
  p = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_decrease_of_b_l3558_355868


namespace NUMINAMATH_CALUDE_gcd_problem_l3558_355800

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 2 * 947 * k) : 
  Nat.gcd (Int.natAbs (3 * a^2 + 47 * a + 101)) (Int.natAbs (a + 19)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3558_355800


namespace NUMINAMATH_CALUDE_five_digit_divisibility_l3558_355840

/-- A function that removes the middle digit of a five-digit number -/
def removeMidDigit (n : ℕ) : ℕ :=
  (n / 10000) * 1000 + (n % 1000)

/-- A predicate that checks if a number is five-digit -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem five_digit_divisibility (A : ℕ) :
  isFiveDigit A →
  (∃ k : ℕ, A = k * (removeMidDigit A)) ↔ (∃ m : ℕ, A = m * 1000) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_l3558_355840


namespace NUMINAMATH_CALUDE_min_value_fraction_l3558_355830

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = 1) :
  (x + y) / (x * y) ≥ 9 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 1 ∧ (x + y) / (x * y) = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3558_355830


namespace NUMINAMATH_CALUDE_precision_of_0_598_l3558_355871

/-- Represents the precision of a decimal number -/
inductive Precision
  | Whole
  | Tenth
  | Hundredth
  | Thousandth
  | TenThousandth
  deriving Repr

/-- Determines the precision of an approximate number -/
def precision (x : Float) : Precision :=
  match x.toString.split (· == '.') with
  | [_, decimal] =>
    match decimal.length with
    | 1 => Precision.Tenth
    | 2 => Precision.Hundredth
    | 3 => Precision.Thousandth
    | 4 => Precision.TenThousandth
    | _ => Precision.Whole
  | _ => Precision.Whole

theorem precision_of_0_598 :
  precision 0.598 = Precision.Thousandth := by
  sorry

end NUMINAMATH_CALUDE_precision_of_0_598_l3558_355871


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3558_355804

theorem max_value_trig_expression (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) = Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3558_355804


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l3558_355841

/-- The average of an arithmetic sequence with 21 terms, 
    starting at -180 and ending at 180, with a common difference of 6, is 0. -/
theorem arithmetic_sequence_average : 
  let first_term : ℤ := -180
  let last_term : ℤ := 180
  let num_terms : ℕ := 21
  let common_diff : ℤ := 6
  let sequence := fun i => first_term + (i : ℤ) * common_diff
  (first_term + last_term) / 2 = 0 ∧ 
  last_term = first_term + (num_terms - 1 : ℕ) * common_diff :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l3558_355841


namespace NUMINAMATH_CALUDE_retailer_profit_calculation_l3558_355806

theorem retailer_profit_calculation 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (actual_profit_percentage : ℝ) 
  (h1 : markup_percentage = 60) 
  (h2 : discount_percentage = 25) 
  (h3 : actual_profit_percentage = 20) : 
  markup_percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_retailer_profit_calculation_l3558_355806


namespace NUMINAMATH_CALUDE_basketball_team_math_enrollment_l3558_355850

theorem basketball_team_math_enrollment (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) :
  total_players = 25 →
  physics_players = 12 →
  both_subjects = 5 →
  (∃ (math_players : ℕ), math_players = total_players - physics_players + both_subjects ∧ math_players = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_team_math_enrollment_l3558_355850


namespace NUMINAMATH_CALUDE_whistle_cost_l3558_355851

theorem whistle_cost (total_cost yoyo_cost : ℕ) (h1 : total_cost = 38) (h2 : yoyo_cost = 24) :
  total_cost - yoyo_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_whistle_cost_l3558_355851


namespace NUMINAMATH_CALUDE_fraction_calculation_l3558_355876

theorem fraction_calculation : (5 / 3 : ℚ) ^ 3 * (2 / 5 : ℚ) = 50 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3558_355876


namespace NUMINAMATH_CALUDE_range_of_a_l3558_355844

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3558_355844


namespace NUMINAMATH_CALUDE_set_operations_l3558_355802

theorem set_operations (A B : Set ℕ) (hA : A = {3, 5, 6, 8}) (hB : B = {4, 5, 7, 8}) :
  (A ∩ B = {5, 8}) ∧ (A ∪ B = {3, 4, 5, 6, 7, 8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3558_355802


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l3558_355848

theorem product_ratio_theorem (a b c d e f : ℝ) (X : ℝ) 
  (h1 : a * b * c = X)
  (h2 : b * c * d = X)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l3558_355848


namespace NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l3558_355895

/-- Given a triangle DEF with inradius r, circumradius R, and angles D, E, F,
    prove that if r = 2, R = 9, and 2cos(E) = cos(D) + cos(F), then the area of the triangle is 54. -/
theorem triangle_area_with_given_conditions (D E F : Real) (r R : Real) :
  r = 2 →
  R = 9 →
  2 * Real.cos E = Real.cos D + Real.cos F →
  ∃ (area : Real), area = 54 ∧ area = r * (Real.sin D + Real.sin E + Real.sin F) * R / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l3558_355895


namespace NUMINAMATH_CALUDE_framing_needed_photo_framing_proof_l3558_355872

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered photo -/
theorem framing_needed (original_width original_height : ℕ) 
  (enlargement_factor : ℕ) (border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (final_width + final_height)
  let perimeter_feet := (perimeter_inches + 11) / 12  -- Rounding up to nearest foot
  perimeter_feet

/-- Proves that a 5x7 inch photo, quadrupled and with 3-inch border, requires 10 feet of framing -/
theorem photo_framing_proof :
  framing_needed 5 7 4 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_framing_needed_photo_framing_proof_l3558_355872


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l3558_355884

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l3558_355884


namespace NUMINAMATH_CALUDE_x_power_ten_equals_fifty_plus_twenty_five_sqrt_five_over_two_l3558_355883

theorem x_power_ten_equals_fifty_plus_twenty_five_sqrt_five_over_two 
  (x : ℝ) (h : x + 1/x = Real.sqrt 5) : 
  x^10 = (50 + 25 * Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_x_power_ten_equals_fifty_plus_twenty_five_sqrt_five_over_two_l3558_355883


namespace NUMINAMATH_CALUDE_probability_two_yellow_marbles_l3558_355814

/-- The probability of drawing two yellow marbles successively from a jar -/
theorem probability_two_yellow_marbles 
  (blue : ℕ) (yellow : ℕ) (black : ℕ) 
  (h_blue : blue = 3)
  (h_yellow : yellow = 4)
  (h_black : black = 8) :
  let total := blue + yellow + black
  (yellow / total) * ((yellow - 1) / (total - 1)) = 2 / 35 := by
sorry

end NUMINAMATH_CALUDE_probability_two_yellow_marbles_l3558_355814


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_specific_case_l3558_355888

def binomial_expansion (a b : ℝ) (n : ℕ) := (a + b)^n

def fourth_term_coefficient (a b : ℝ) (n : ℕ) : ℝ :=
  Nat.choose n 3 * a^(n-3) * b^3

theorem fourth_term_coefficient_specific_case :
  fourth_term_coefficient (1/2 * Real.sqrt x) (2/(3*x)) 6 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_specific_case_l3558_355888


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l3558_355829

/-- The parabolas y = (x - 1)^2 and x - 3 = (y + 2)^2 intersect at four points that lie on a circle with radius squared equal to 1/2 -/
theorem intersection_points_on_circle : ∃ (c : ℝ × ℝ) (r : ℝ),
  (∀ (p : ℝ × ℝ), (p.2 = (p.1 - 1)^2 ∧ p.1 - 3 = (p.2 + 2)^2) →
    ((p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2)) ∧
  r^2 = (1 : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l3558_355829


namespace NUMINAMATH_CALUDE_square_difference_262_258_l3558_355873

theorem square_difference_262_258 : 262^2 - 258^2 = 2080 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_262_258_l3558_355873


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_31_over_6_l3558_355821

theorem greatest_integer_less_than_negative_31_over_6 :
  ⌊-31/6⌋ = -6 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_31_over_6_l3558_355821


namespace NUMINAMATH_CALUDE_smallest_multiple_40_over_100_l3558_355859

theorem smallest_multiple_40_over_100 : ∀ n : ℕ, n > 0 ∧ 40 ∣ n ∧ n > 100 → n ≥ 120 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_40_over_100_l3558_355859


namespace NUMINAMATH_CALUDE_game_cost_l3558_355862

theorem game_cost (initial_money : ℕ) (num_toys : ℕ) (toy_price : ℕ) (game_cost : ℕ) : 
  initial_money = 57 → 
  num_toys = 5 → 
  toy_price = 6 → 
  initial_money = game_cost + (num_toys * toy_price) → 
  game_cost = 27 := by
sorry

end NUMINAMATH_CALUDE_game_cost_l3558_355862


namespace NUMINAMATH_CALUDE_solution_set_equality_l3558_355824

/-- The set of real numbers a for which the solution set of |x - 2| < a is a subset of (-2, 1] -/
def A : Set ℝ := {a | ∀ x, |x - 2| < a → -2 < x ∧ x ≤ 1}

/-- The theorem stating that A is equal to (-∞, 0] -/
theorem solution_set_equality : A = Set.Iic 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3558_355824


namespace NUMINAMATH_CALUDE_max_students_in_dance_l3558_355877

theorem max_students_in_dance (x : ℕ) : 
  x < 100 ∧ 
  x % 8 = 5 ∧ 
  x % 5 = 3 →
  x ≤ 93 ∧ 
  ∃ y : ℕ, y = 93 ∧ 
    y < 100 ∧ 
    y % 8 = 5 ∧ 
    y % 5 = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_students_in_dance_l3558_355877


namespace NUMINAMATH_CALUDE_lindas_tv_cost_l3558_355813

/-- The cost of Linda's TV purchase, given her original savings and furniture expenses -/
theorem lindas_tv_cost (original_savings : ℝ) (furniture_fraction : ℝ) : 
  original_savings = 800 →
  furniture_fraction = 3/4 →
  original_savings * (1 - furniture_fraction) = 200 := by
sorry

end NUMINAMATH_CALUDE_lindas_tv_cost_l3558_355813


namespace NUMINAMATH_CALUDE_possible_integer_roots_l3558_355857

def polynomial (x b₂ b₁ : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 30

def is_root (x b₂ b₁ : ℤ) : Prop := polynomial x b₂ b₁ = 0

def divisors_of_30 : Set ℤ := {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30}

theorem possible_integer_roots (b₂ b₁ : ℤ) :
  {x : ℤ | ∃ (b₂ b₁ : ℤ), is_root x b₂ b₁} = divisors_of_30 := by sorry

end NUMINAMATH_CALUDE_possible_integer_roots_l3558_355857


namespace NUMINAMATH_CALUDE_quadratic_polynomial_determination_l3558_355896

/-- A type representing quadratic polynomials -/
def QuadraticPolynomial := ℝ → ℝ

/-- A function that evaluates a quadratic polynomial at a given point -/
def evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ := p x

/-- A function that checks if two polynomials agree at a given point -/
def agree (p q : QuadraticPolynomial) (x : ℝ) : Prop := evaluate p x = evaluate q x

theorem quadratic_polynomial_determination (n : ℕ) (h : n > 1) :
  ∃ (C : ℝ), C > 0 ∧
  ∀ (polynomials : Finset QuadraticPolynomial),
  polynomials.card = n →
  ∃ (points : Finset ℝ),
  points.card = 2 * n^2 + 1 ∧
  ∃ (p : QuadraticPolynomial),
  p ∈ polynomials ∧
  ∀ (q : QuadraticPolynomial),
  (∀ (x : ℝ), x ∈ points → agree p q x) → p = q :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_determination_l3558_355896


namespace NUMINAMATH_CALUDE_broccoli_area_l3558_355893

theorem broccoli_area (current_production : ℕ) (increase : ℕ) : 
  current_production = 2601 →
  increase = 101 →
  ∃ (previous_side : ℕ) (current_side : ℕ),
    previous_side ^ 2 + increase = current_side ^ 2 ∧
    current_side ^ 2 = current_production ∧
    (current_side ^ 2 : ℚ) / current_production = 1 :=
by sorry

end NUMINAMATH_CALUDE_broccoli_area_l3558_355893


namespace NUMINAMATH_CALUDE_selection_theorem_l3558_355807

/-- The number of ways to select 4 students from 7 students (4 boys and 3 girls), 
    ensuring that the selection includes both boys and girls -/
def selection_ways : ℕ :=
  Nat.choose 7 4 - Nat.choose 4 4

theorem selection_theorem : selection_ways = 34 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l3558_355807


namespace NUMINAMATH_CALUDE_inequality_proof_l3558_355828

theorem inequality_proof (x : ℝ) (hx : x > 0) : 1 + x^2018 ≥ (2*x)^2017 / (1 + x)^2016 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3558_355828


namespace NUMINAMATH_CALUDE_min_operations_rectangle_l3558_355815

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Measures the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Checks if two numbers are equal -/
def compare (a b : ℝ) : Bool :=
  sorry

/-- Checks if a quadrilateral is a rectangle -/
def isRectangle (q : Quadrilateral) : Bool :=
  sorry

/-- Counts the number of operations needed to determine if a quadrilateral is a rectangle -/
def countOperations (q : Quadrilateral) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of operations to determine if a quadrilateral is a rectangle is 9 -/
theorem min_operations_rectangle (q : Quadrilateral) : 
  (countOperations q = 9) ∧ (∀ n : ℕ, n < 9 → ¬(∀ q' : Quadrilateral, isRectangle q' ↔ countOperations q' ≤ n)) :=
  sorry

end NUMINAMATH_CALUDE_min_operations_rectangle_l3558_355815


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3558_355812

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^3 > x^2) ↔ ∃ x : ℝ, x^3 ≤ x^2 := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3558_355812


namespace NUMINAMATH_CALUDE_goldfish_pond_problem_l3558_355843

theorem goldfish_pond_problem :
  ∀ (x : ℕ),
  (x > 0) →
  (3 * x / 7 : ℚ) + (4 * x / 7 : ℚ) = x →
  (5 * x / 8 : ℚ) + (3 * x / 8 : ℚ) = x →
  (5 * x / 8 : ℚ) - (3 * x / 7 : ℚ) = 33 →
  x = 168 := by
sorry

end NUMINAMATH_CALUDE_goldfish_pond_problem_l3558_355843


namespace NUMINAMATH_CALUDE_arithmetic_expression_value_l3558_355863

theorem arithmetic_expression_value :
  ∀ (A B C : Nat),
    A ≠ B → A ≠ C → B ≠ C →
    A < 10 → B < 10 → C < 10 →
    3 * C % 10 = C →
    (2 * B + 1) % 10 = B →
    300 + 10 * B + C = 395 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_value_l3558_355863


namespace NUMINAMATH_CALUDE_vector_subtraction_magnitude_l3558_355875

def vector_a : ℝ × ℝ := (-2, 2)

theorem vector_subtraction_magnitude (b : ℝ × ℝ) 
  (h1 : ‖b‖ = 1) 
  (h2 : vector_a • b = 2) : ‖vector_a - 2 • b‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_magnitude_l3558_355875


namespace NUMINAMATH_CALUDE_days_off_per_month_l3558_355838

def total_holidays : ℕ := 36
def months_in_year : ℕ := 12

theorem days_off_per_month :
  total_holidays / months_in_year = 3 := by sorry

end NUMINAMATH_CALUDE_days_off_per_month_l3558_355838


namespace NUMINAMATH_CALUDE_fruit_store_problem_l3558_355820

/-- The number of watermelons in a fruit store. -/
def num_watermelons : ℕ := by sorry

theorem fruit_store_problem :
  let apples : ℕ := 82
  let pears : ℕ := 90
  let tangerines : ℕ := 88
  let melons : ℕ := 84
  let total_fruits : ℕ := apples + pears + tangerines + melons + num_watermelons
  (total_fruits % 88 = 0) ∧ (total_fruits / 88 = 5) →
  num_watermelons = 96 := by sorry

end NUMINAMATH_CALUDE_fruit_store_problem_l3558_355820


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3558_355842

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1)}

def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem set_intersection_theorem : N ∩ (U \ M) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3558_355842


namespace NUMINAMATH_CALUDE_triangle_city_population_l3558_355898

theorem triangle_city_population : ∃ (x y z : ℕ+), 
  x^2 + 50 = y^2 + 1 ∧ 
  y^2 + 351 = z^2 ∧ 
  x^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_triangle_city_population_l3558_355898


namespace NUMINAMATH_CALUDE_line_intersection_xz_plane_l3558_355881

/-- The line passing through two points intersects the xz-plane at a specific point -/
theorem line_intersection_xz_plane (p₁ p₂ : ℝ × ℝ × ℝ) (intersection : ℝ × ℝ × ℝ) : 
  p₁ = (2, 3, 2) → 
  p₂ = (6, -1, 7) → 
  intersection.2 = 0 → 
  ∃ t : ℝ, intersection = (2 + 4*t, 3 - 4*t, 2 + 5*t) ∧ 
        intersection = (5, 0, 23/4) := by
  sorry

#check line_intersection_xz_plane

end NUMINAMATH_CALUDE_line_intersection_xz_plane_l3558_355881


namespace NUMINAMATH_CALUDE_ball_probabilities_l3558_355860

def total_balls : ℕ := 6
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def black_balls : ℕ := 1
def drawn_balls : ℕ := 3

def prob_one_red_one_white : ℚ := 3 / 10
def prob_at_least_two_red : ℚ := 1 / 2
def prob_no_black : ℚ := 1 / 2

theorem ball_probabilities :
  (total_balls = red_balls + white_balls + black_balls) →
  (drawn_balls ≤ total_balls) →
  (prob_one_red_one_white = 3 / 10) ∧
  (prob_at_least_two_red = 1 / 2) ∧
  (prob_no_black = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l3558_355860


namespace NUMINAMATH_CALUDE_cement_truck_loads_l3558_355825

theorem cement_truck_loads (total material_truck_loads sand_truck_loads dirt_truck_loads : ℚ)
  (h1 : total = 0.67)
  (h2 : sand_truck_loads = 0.17)
  (h3 : dirt_truck_loads = 0.33)
  : total - (sand_truck_loads + dirt_truck_loads) = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_cement_truck_loads_l3558_355825


namespace NUMINAMATH_CALUDE_product_of_one_plus_tangents_17_and_28_l3558_355855

theorem product_of_one_plus_tangents_17_and_28 :
  (1 + Real.tan (17 * π / 180)) * (1 + Real.tan (28 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_tangents_17_and_28_l3558_355855


namespace NUMINAMATH_CALUDE_pentagonal_prism_sum_l3558_355810

/-- A pentagonal prism is a three-dimensional geometric shape with specific properties. -/
structure PentagonalPrism where
  /-- The number of faces in a pentagonal prism -/
  faces : ℕ
  /-- The number of edges in a pentagonal prism -/
  edges : ℕ
  /-- The number of vertices in a pentagonal prism -/
  vertices : ℕ
  /-- A pentagonal prism has 7 faces (2 pentagonal bases + 5 rectangular lateral faces) -/
  faces_count : faces = 7
  /-- A pentagonal prism has 15 edges (5 for each base + 5 connecting edges) -/
  edges_count : edges = 15
  /-- A pentagonal prism has 10 vertices (5 for each base) -/
  vertices_count : vertices = 10

/-- The sum of faces, edges, and vertices of a pentagonal prism is 32. -/
theorem pentagonal_prism_sum (p : PentagonalPrism) : p.faces + p.edges + p.vertices = 32 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_prism_sum_l3558_355810


namespace NUMINAMATH_CALUDE_dads_borrowed_nickels_l3558_355866

/-- The number of nickels Mike's dad borrowed -/
def nickels_borrowed (initial_nickels remaining_nickels : ℕ) : ℕ :=
  initial_nickels - remaining_nickels

theorem dads_borrowed_nickels :
  let initial_nickels : ℕ := 87
  let remaining_nickels : ℕ := 12
  nickels_borrowed initial_nickels remaining_nickels = 75 := by
sorry

end NUMINAMATH_CALUDE_dads_borrowed_nickels_l3558_355866


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l3558_355803

/-- The equation represents two lines -/
theorem equation_represents_two_lines :
  ∃ (a b c d : ℝ), a ≠ c ∧ b ≠ d ∧
  ∀ (x y : ℝ), x^2 - 50*y^2 - 10*x + 25 = 0 ↔ 
  ((x = a*y + b) ∨ (x = c*y + d)) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l3558_355803


namespace NUMINAMATH_CALUDE_no_special_multiple_l3558_355826

/-- Calculates the digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

/-- Generates a repunit with m ones -/
def repunit (m : ℕ) : ℕ :=
  if m = 0 then 0 else (10^m - 1) / 9

/-- The main theorem -/
theorem no_special_multiple :
  ¬ ∃ (n m : ℕ), 
    (∃ k : ℕ, n = k * (10 * 94)) ∧
    (n % repunit m = 0) ∧
    (digit_sum n < m) :=
sorry

end NUMINAMATH_CALUDE_no_special_multiple_l3558_355826


namespace NUMINAMATH_CALUDE_system_solution_exists_l3558_355852

theorem system_solution_exists : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  (4 * x₁ - 3 * y₁ = -3 ∧ 8 * x₁ + 5 * y₁ = 11 + x₁^2) ∧
  (4 * x₂ - 3 * y₂ = -3 ∧ 8 * x₂ + 5 * y₂ = 11 + x₂^2) ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_exists_l3558_355852


namespace NUMINAMATH_CALUDE_airport_distance_l3558_355819

/-- Represents the problem of calculating the distance to the airport --/
def airport_distance_problem (initial_speed : ℝ) (speed_increase : ℝ) (late_time : ℝ) : Prop :=
  ∃ (distance : ℝ) (initial_time : ℝ),
    -- If he continued at initial speed, he'd be 1 hour late
    distance = initial_speed * (initial_time + 1) ∧
    -- The remaining distance at increased speed
    (distance - initial_speed) = (initial_speed + speed_increase) * (initial_time - late_time) ∧
    -- The total distance is 70 miles
    distance = 70

/-- The theorem stating that the airport is 70 miles away --/
theorem airport_distance :
  airport_distance_problem 40 20 (1/4) :=
sorry


end NUMINAMATH_CALUDE_airport_distance_l3558_355819


namespace NUMINAMATH_CALUDE_equal_area_line_slope_l3558_355801

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Calculates the distance from a point to a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Determines if a line divides a circle into equal areas -/
def divideCircleEqually (c : Circle) (l : Line) : Prop :=
  sorry

theorem equal_area_line_slope :
  let c1 : Circle := ⟨(10, 40), 5⟩
  let c2 : Circle := ⟨(15, 30), 5⟩
  let p : ℝ × ℝ := (12, 20)
  ∃ (l : Line),
    l.slope = (5 - Real.sqrt 73) / 2 ∨ l.slope = (5 + Real.sqrt 73) / 2 ∧
    (p.1 * l.slope + l.yIntercept = p.2) ∧
    divideCircleEqually c1 l ∧
    divideCircleEqually c2 l :=
  sorry

end NUMINAMATH_CALUDE_equal_area_line_slope_l3558_355801


namespace NUMINAMATH_CALUDE_john_walks_farther_l3558_355894

/-- John's walking distance to school in miles -/
def john_distance : ℝ := 1.74

/-- Nina's walking distance to school in miles -/
def nina_distance : ℝ := 1.235

/-- The difference between John's and Nina's walking distances -/
def distance_difference : ℝ := john_distance - nina_distance

theorem john_walks_farther : distance_difference = 0.505 := by sorry

end NUMINAMATH_CALUDE_john_walks_farther_l3558_355894


namespace NUMINAMATH_CALUDE_parabola_tangent_circle_problem_l3558_355864

-- Define the parabola T₀: y = x²
def T₀ (x : ℝ) : ℝ := x^2

-- Define point P
def P : ℝ × ℝ := (1, -1)

-- Define the tangent line passing through P and intersecting T₀
def tangent_line (x₁ x₂ : ℝ) : Prop :=
  x₁ < x₂ ∧ 
  T₀ x₁ = (x₁ - 1) * 2 * x₁ + (-1) ∧
  T₀ x₂ = (x₂ - 1) * 2 * x₂ + (-1)

-- Define circle E with center at P and tangent to line MN
def circle_E (r : ℝ) : Prop :=
  r = (4 : ℝ) / Real.sqrt 5

-- Define chords AC and BD passing through origin and perpendicular in circle E
def chords_ABCD (d₁ d₂ : ℝ) : Prop :=
  d₁^2 + d₂^2 = 2

-- Main theorem
theorem parabola_tangent_circle_problem :
  ∃ (x₁ x₂ r d₁ d₂ : ℝ),
    tangent_line x₁ x₂ ∧
    circle_E r ∧
    chords_ABCD d₁ d₂ ∧
    x₁ = 1 - Real.sqrt 2 ∧
    x₂ = 1 + Real.sqrt 2 ∧
    r^2 * Real.pi = (16 : ℝ) / 5 ∧
    2 * r^2 - (d₁^2 + d₂^2) ≤ (22 : ℝ) / 5 :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_circle_problem_l3558_355864


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l3558_355833

def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

theorem solution_set_implies_m_value (m : ℝ) :
  (∀ x : ℝ, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l3558_355833


namespace NUMINAMATH_CALUDE_soccer_league_games_l3558_355853

theorem soccer_league_games (C D : ℕ) : 
  (3 * C = 4 * (C - (C / 4))) →  -- Team C has won 3/4 of its games
  (2 * (C + 6) = 3 * ((C + 6) - ((C + 6) / 3))) →  -- Team D has won 2/3 of its games
  (C + 6 = D) →  -- Team D has played 6 more games than team C
  (C = 12) :=  -- Prove that team C has played 12 games
by sorry

end NUMINAMATH_CALUDE_soccer_league_games_l3558_355853


namespace NUMINAMATH_CALUDE_abc_inequality_l3558_355808

theorem abc_inequality (a b c : ℝ) (sum_eq_one : a + b + c = 1) (prod_pos : a * b * c > 0) :
  a * b + b * c + c * a < Real.sqrt (a * b * c) / 2 + 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3558_355808


namespace NUMINAMATH_CALUDE_harry_pizza_order_cost_l3558_355822

def pizza_order_cost (large_pizza_cost : ℚ) (topping_cost : ℚ) 
  (num_pizzas : ℕ) (num_toppings : ℕ) (tip_percentage : ℚ) : ℚ :=
  let base_cost := num_pizzas * (large_pizza_cost + num_toppings * topping_cost)
  let tip := tip_percentage * base_cost
  base_cost + tip

theorem harry_pizza_order_cost :
  pizza_order_cost 14 2 2 3 (1/4) = 50 := by
  sorry

end NUMINAMATH_CALUDE_harry_pizza_order_cost_l3558_355822


namespace NUMINAMATH_CALUDE_hotel_room_number_contradiction_l3558_355897

theorem hotel_room_number_contradiction : 
  ¬ ∃ (a b c : ℕ), 
    0 < a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    0 ≤ c ∧ c ≤ 9 ∧ 
    100 * a + 10 * b + c = (a + 1) * (b + 1) * c :=
by sorry

end NUMINAMATH_CALUDE_hotel_room_number_contradiction_l3558_355897


namespace NUMINAMATH_CALUDE_paper_folding_cutting_perimeter_ratio_l3558_355867

theorem paper_folding_cutting_perimeter_ratio :
  let original_length : ℝ := 10
  let original_width : ℝ := 8
  let folded_length : ℝ := original_length / 2
  let folded_width : ℝ := original_width
  let small_rectangle_length : ℝ := folded_length
  let small_rectangle_width : ℝ := folded_width / 2
  let large_rectangle_length : ℝ := folded_length
  let large_rectangle_width : ℝ := folded_width
  let small_rectangle_perimeter : ℝ := 2 * (small_rectangle_length + small_rectangle_width)
  let large_rectangle_perimeter : ℝ := 2 * (large_rectangle_length + large_rectangle_width)
  small_rectangle_perimeter / large_rectangle_perimeter = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_cutting_perimeter_ratio_l3558_355867


namespace NUMINAMATH_CALUDE_range_of_y_over_x_for_unit_modulus_complex_l3558_355839

theorem range_of_y_over_x_for_unit_modulus_complex (x y : ℝ) :
  (x - 2)^2 + y^2 = 1 →
  y ≠ 0 →
  ∃ k : ℝ, y = k * x ∧ k ∈ Set.Ioo (-Real.sqrt 3 / 3) 0 ∪ Set.Ioo 0 (Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_for_unit_modulus_complex_l3558_355839


namespace NUMINAMATH_CALUDE_train_crossing_time_l3558_355870

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 120 → 
  train_speed_kmh = 48 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 9 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3558_355870


namespace NUMINAMATH_CALUDE_thirtieth_set_sum_l3558_355891

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of consecutive integers from a to b, inclusive -/
def sum_consecutive (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ :=
  let first := triangular_number (n - 1) + 1
  let last := triangular_number n
  sum_consecutive first last

theorem thirtieth_set_sum : S 30 = 13515 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_set_sum_l3558_355891


namespace NUMINAMATH_CALUDE_product_when_c_is_one_l3558_355887

theorem product_when_c_is_one (a b c : ℕ+) (h1 : a * b * c = a * b ^ 3) (h2 : c = 1) :
  a * b * c = a := by
  sorry

end NUMINAMATH_CALUDE_product_when_c_is_one_l3558_355887


namespace NUMINAMATH_CALUDE_triangle_expression_range_l3558_355856

theorem triangle_expression_range (A B C a b c : ℝ) : 
  0 < A → A < 3 * π / 4 →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  c * Real.sin A = a * Real.cos C →
  1 < Real.sqrt 3 * Real.sin A - Real.cos (B + π / 4) ∧ 
  Real.sqrt 3 * Real.sin A - Real.cos (B + π / 4) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_range_l3558_355856


namespace NUMINAMATH_CALUDE_factorization_equality_l3558_355834

theorem factorization_equality (a : ℝ) : -9 - a^2 + 6*a = -(a - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3558_355834


namespace NUMINAMATH_CALUDE_plumbing_job_washers_l3558_355845

/-- Calculates the number of remaining washers after a plumbing job --/
def remaining_washers (total_pipe_length : ℕ) (pipe_per_bolt : ℕ) (washers_per_bolt : ℕ) (total_washers : ℕ) : ℕ :=
  let bolts_needed := total_pipe_length / pipe_per_bolt
  let washers_used := bolts_needed * washers_per_bolt
  total_washers - washers_used

/-- Theorem stating that for the given plumbing job, 4 washers will remain --/
theorem plumbing_job_washers :
  remaining_washers 40 5 2 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_plumbing_job_washers_l3558_355845


namespace NUMINAMATH_CALUDE_community_pantry_fraction_l3558_355832

theorem community_pantry_fraction (total_donation : ℚ) 
  (crisis_fund_fraction : ℚ) (livelihood_fraction : ℚ) (contingency_amount : ℚ) :
  total_donation = 240 →
  crisis_fund_fraction = 1/2 →
  livelihood_fraction = 1/4 →
  contingency_amount = 30 →
  (total_donation - crisis_fund_fraction * total_donation - 
   livelihood_fraction * (total_donation - crisis_fund_fraction * total_donation) - 
   contingency_amount) / total_donation = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_community_pantry_fraction_l3558_355832


namespace NUMINAMATH_CALUDE_debbys_flour_amount_l3558_355854

/-- Proves that Debby's total flour amount is correct given her initial amount and purchase. -/
theorem debbys_flour_amount (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 12 → bought = 4 → total = initial + bought → total = 16 := by
  sorry

end NUMINAMATH_CALUDE_debbys_flour_amount_l3558_355854


namespace NUMINAMATH_CALUDE_ernies_income_ratio_l3558_355827

def ernies_previous_income : ℕ := 6000
def jacks_income : ℕ := 2 * ernies_previous_income
def combined_income : ℕ := 16800
def ernies_current_income : ℕ := combined_income - jacks_income

theorem ernies_income_ratio :
  (ernies_current_income : ℚ) / ernies_previous_income = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ernies_income_ratio_l3558_355827
