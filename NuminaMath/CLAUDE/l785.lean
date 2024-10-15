import Mathlib

namespace NUMINAMATH_CALUDE_lunchroom_students_l785_78556

theorem lunchroom_students (students_per_table : ℕ) (num_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : num_tables = 34) : 
  students_per_table * num_tables = 204 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l785_78556


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l785_78558

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (5 * bowling_ball_weight = 3 * canoe_weight) →
    (3 * canoe_weight = 105) →
    bowling_ball_weight = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l785_78558


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_216_l785_78517

theorem closest_integer_to_cube_root_216 : 
  ∀ n : ℤ, |n - (216 : ℝ)^(1/3)| ≥ |6 - (216 : ℝ)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_216_l785_78517


namespace NUMINAMATH_CALUDE_xyz_equals_five_l785_78504

theorem xyz_equals_five
  (a b c x y z : ℂ)
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq_a : a = (b^2 + c^2) / (x - 3))
  (eq_b : b = (a^2 + c^2) / (y - 3))
  (eq_c : c = (a^2 + b^2) / (z - 3))
  (sum_prod : x*y + y*z + z*x = 11)
  (sum : x + y + z = 5) :
  x * y * z = 5 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_five_l785_78504


namespace NUMINAMATH_CALUDE_expression_value_l785_78581

theorem expression_value (m n : ℝ) (h : m + 2*n = 1) : 3*m^2 + 6*m*n + 6*n = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l785_78581


namespace NUMINAMATH_CALUDE_value_of_m_area_of_triangle_max_y_intercept_l785_78580

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 - x - m^2 + 6*m - 7

-- Theorem 1: If the graph passes through A(-1, 2), then m = 5
theorem value_of_m (m : ℝ) : f m (-1) = 2 → m = 5 := by sorry

-- Theorem 2: If m = 5, the area of triangle ABC is 5/3
theorem area_of_triangle : 
  let m := 5
  let x1 := (- 2/3 : ℝ)  -- x-coordinate of point C
  let x2 := (1 : ℝ)      -- x-coordinate of point B
  (1/2 : ℝ) * |x2 - x1| * 2 = 5/3 := by sorry

-- Theorem 3: The maximum y-coordinate of the y-intercept is 2
theorem max_y_intercept : 
  ∃ (m : ℝ), ∀ (m' : ℝ), f m' 0 ≤ f m 0 ∧ f m 0 = 2 := by sorry

end NUMINAMATH_CALUDE_value_of_m_area_of_triangle_max_y_intercept_l785_78580


namespace NUMINAMATH_CALUDE_smallest_bob_number_l785_78545

/-- Alice's number -/
def alice_number : ℕ := 45

/-- Bob's number is a natural number -/
def bob_number : ℕ := sorry

/-- Every prime factor of Alice's number is also a prime factor of Bob's number -/
axiom bob_has_alice_prime_factors :
  ∀ p : ℕ, Prime p → p ∣ alice_number → p ∣ bob_number

/-- Bob's number is the smallest possible given the conditions -/
axiom bob_number_is_smallest :
  ∀ n : ℕ, (∀ p : ℕ, Prime p → p ∣ alice_number → p ∣ n) → bob_number ≤ n

theorem smallest_bob_number : bob_number = 15 := by sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l785_78545


namespace NUMINAMATH_CALUDE_single_elimination_matches_l785_78543

/-- The number of matches required in a single-elimination tournament -/
def matches_required (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with n participants,
    the number of matches required to determine the winner is n - 1 -/
theorem single_elimination_matches (n : ℕ) (h : n > 0) : 
  matches_required n = n - 1 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_matches_l785_78543


namespace NUMINAMATH_CALUDE_oranges_per_box_l785_78552

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (oranges_per_box : ℕ) : 
  total_oranges = 24 → num_boxes = 3 → oranges_per_box * num_boxes = total_oranges → oranges_per_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l785_78552


namespace NUMINAMATH_CALUDE_construction_delay_l785_78579

/-- Represents the construction project -/
structure ConstructionProject where
  total_days : ℕ
  initial_workers : ℕ
  additional_workers : ℕ
  days_before_addition : ℕ

/-- Calculates the delay in days if additional workers were not added -/
def calculate_delay (project : ConstructionProject) : ℕ :=
  let total_work := project.total_days * project.initial_workers
  let work_done_before_addition := project.days_before_addition * project.initial_workers
  let remaining_work := total_work - work_done_before_addition
  let days_with_additional_workers := project.total_days - project.days_before_addition
  let work_done_after_addition := days_with_additional_workers * (project.initial_workers + project.additional_workers)
  (remaining_work + project.initial_workers - 1) / project.initial_workers - days_with_additional_workers

theorem construction_delay (project : ConstructionProject) 
  (h1 : project.total_days = 100)
  (h2 : project.initial_workers = 100)
  (h3 : project.additional_workers = 100)
  (h4 : project.days_before_addition = 20) :
  calculate_delay project = 80 := by
  sorry

#eval calculate_delay { total_days := 100, initial_workers := 100, additional_workers := 100, days_before_addition := 20 }

end NUMINAMATH_CALUDE_construction_delay_l785_78579


namespace NUMINAMATH_CALUDE_candle_count_l785_78591

def total_candles (bedroom_candles : ℕ) (additional_candles : ℕ) : ℕ :=
  bedroom_candles + (bedroom_candles / 2) + additional_candles

theorem candle_count : total_candles 20 20 = 50 := by
  sorry

end NUMINAMATH_CALUDE_candle_count_l785_78591


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l785_78521

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.arctan (b / a) = π / 6) →
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l785_78521


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l785_78572

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = {x : ℝ | x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l785_78572


namespace NUMINAMATH_CALUDE_james_course_cost_l785_78566

/-- Represents the cost per unit for James's community college courses. -/
def cost_per_unit (units_per_semester : ℕ) (total_cost : ℕ) (num_semesters : ℕ) : ℚ :=
  total_cost / (units_per_semester * num_semesters)

/-- Theorem stating that the cost per unit is $50 given the conditions. -/
theorem james_course_cost : 
  cost_per_unit 20 2000 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_james_course_cost_l785_78566


namespace NUMINAMATH_CALUDE_original_number_of_people_l785_78593

theorem original_number_of_people (x : ℕ) : 
  (x / 3 : ℚ) = 18 → x = 54 := by sorry

end NUMINAMATH_CALUDE_original_number_of_people_l785_78593


namespace NUMINAMATH_CALUDE_greatest_power_of_eleven_l785_78550

theorem greatest_power_of_eleven (n : ℕ+) : 
  (Finset.card (Nat.divisors n) = 72) →
  (Finset.card (Nat.divisors (11 * n)) = 96) →
  (∃ k : ℕ, 11^k ∣ n ∧ ∀ m : ℕ, 11^m ∣ n → m ≤ k) →
  (∃ k : ℕ, 11^k ∣ n ∧ ∀ m : ℕ, 11^m ∣ n → m ≤ k) ∧ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_eleven_l785_78550


namespace NUMINAMATH_CALUDE_chord_passes_through_fixed_point_l785_78557

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Parabola C with equation x^2 = 4y -/
def parabolaC (p : Point) : Prop :=
  p.x^2 = 4 * p.y

/-- Dot product of two vectors represented by points -/
def dotProduct (p1 p2 : Point) : ℝ :=
  p1.x * p2.x + p1.y * p2.y

/-- Condition that the dot product of OA and OB is -4 -/
def dotProductCondition (a b : Point) : Prop :=
  dotProduct a b = -4

/-- Line passes through a point -/
def linePassesThrough (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Theorem stating that if a chord AB of parabola C satisfies the dot product condition,
    then the line AB always passes through the point (0, 2) -/
theorem chord_passes_through_fixed_point 
  (a b : Point) (l : Line) 
  (h1 : parabolaC a) 
  (h2 : parabolaC b) 
  (h3 : dotProductCondition a b) 
  (h4 : linePassesThrough l a) 
  (h5 : linePassesThrough l b) : 
  linePassesThrough l (Point.mk 0 2) :=
sorry

end NUMINAMATH_CALUDE_chord_passes_through_fixed_point_l785_78557


namespace NUMINAMATH_CALUDE_number_of_cars_in_race_l785_78578

/-- The number of cars in a race where:
  1. Each car starts with 3 people.
  2. After the halfway point, each car has 4 people.
  3. At the end of the race, there are 80 people in total. -/
theorem number_of_cars_in_race : ℕ :=
  let initial_people_per_car : ℕ := 3
  let final_people_per_car : ℕ := 4
  let total_people_at_end : ℕ := 80
  20

#check number_of_cars_in_race

end NUMINAMATH_CALUDE_number_of_cars_in_race_l785_78578


namespace NUMINAMATH_CALUDE_triangle_side_length_l785_78586

/-- Given a triangle ABC with ∠A = 40°, ∠B = 90°, and AC = 6, prove that BC = 6 * sin(40°) -/
theorem triangle_side_length (A B C : ℝ × ℝ) : 
  let angle (P Q R : ℝ × ℝ) := Real.arccos ((Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2)) / 
    (Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) * Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2))
  let dist (P Q : ℝ × ℝ) := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  angle B A C = Real.pi / 4.5 →  -- 40°
  angle A B C = Real.pi / 2 →    -- 90°
  dist A C = 6 →
  dist B C = 6 * Real.sin (Real.pi / 4.5) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l785_78586


namespace NUMINAMATH_CALUDE_farmer_plant_beds_l785_78530

theorem farmer_plant_beds (bean_seedlings : ℕ) (bean_per_row : ℕ) 
  (pumpkin_seeds : ℕ) (pumpkin_per_row : ℕ) 
  (radishes : ℕ) (radish_per_row : ℕ) 
  (rows_per_bed : ℕ) : 
  bean_seedlings = 64 → 
  bean_per_row = 8 → 
  pumpkin_seeds = 84 → 
  pumpkin_per_row = 7 → 
  radishes = 48 → 
  radish_per_row = 6 → 
  rows_per_bed = 2 → 
  (bean_seedlings / bean_per_row + 
   pumpkin_seeds / pumpkin_per_row + 
   radishes / radish_per_row) / rows_per_bed = 14 := by
  sorry

#check farmer_plant_beds

end NUMINAMATH_CALUDE_farmer_plant_beds_l785_78530


namespace NUMINAMATH_CALUDE_pretzel_problem_l785_78536

theorem pretzel_problem (john_pretzels alan_pretzels marcus_pretzels initial_pretzels : ℕ) :
  john_pretzels = 28 →
  alan_pretzels = john_pretzels - 9 →
  marcus_pretzels = john_pretzels + 12 →
  marcus_pretzels = 40 →
  initial_pretzels = john_pretzels + alan_pretzels + marcus_pretzels →
  initial_pretzels = 87 := by
  sorry

end NUMINAMATH_CALUDE_pretzel_problem_l785_78536


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l785_78529

theorem weight_loss_challenge (initial_weight : ℝ) (clothes_weight_percentage : ℝ) 
  (h1 : clothes_weight_percentage > 0) 
  (h2 : initial_weight > 0) : 
  (0.85 * initial_weight + clothes_weight_percentage * 0.85 * initial_weight) / initial_weight = 0.867 → 
  clothes_weight_percentage = 0.02 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l785_78529


namespace NUMINAMATH_CALUDE_quadratic_equation_magnitude_unique_l785_78553

theorem quadratic_equation_magnitude_unique :
  ∃! m : ℝ, ∀ z : ℂ, z^2 - 10*z + 50 = 0 → Complex.abs z = m :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_magnitude_unique_l785_78553


namespace NUMINAMATH_CALUDE_dons_pizza_consumption_l785_78506

/-- Don's pizza consumption problem -/
theorem dons_pizza_consumption (darias_consumption : ℝ) (total_consumption : ℝ) 
  (h1 : darias_consumption = 2.5 * (total_consumption - darias_consumption))
  (h2 : total_consumption = 280) : 
  total_consumption - darias_consumption = 80 := by
  sorry

end NUMINAMATH_CALUDE_dons_pizza_consumption_l785_78506


namespace NUMINAMATH_CALUDE_negative_to_even_power_l785_78537

theorem negative_to_even_power (a : ℝ) : (-a)^4 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_negative_to_even_power_l785_78537


namespace NUMINAMATH_CALUDE_sum_of_a_values_l785_78567

theorem sum_of_a_values : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, (∃! (sol : Finset ℤ), 
    (∀ x ∈ sol, (4 * x - a ≥ 1 ∧ (x + 13) / 2 ≥ x + 2)) ∧ 
    sol.card = 6)) ∧ 
  S.sum id = 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_values_l785_78567


namespace NUMINAMATH_CALUDE_cone_volume_l785_78574

/-- The volume of a cone with base radius 1 and slant height 2√7 is √3π. -/
theorem cone_volume (r h s : ℝ) : 
  r = 1 → s = 2 * Real.sqrt 7 → h^2 + r^2 = s^2 → (1/3) * π * r^2 * h = Real.sqrt 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l785_78574


namespace NUMINAMATH_CALUDE_average_and_variance_after_adding_datapoint_l785_78594

def initial_average : ℝ := 4
def initial_variance : ℝ := 2
def initial_count : ℕ := 7
def new_datapoint : ℝ := 4
def new_count : ℕ := initial_count + 1

def new_average (x : ℝ) : Prop :=
  x = (initial_count * initial_average + new_datapoint) / new_count

def new_variance (s : ℝ) : Prop :=
  s = (initial_count * initial_variance + (new_datapoint - initial_average)^2) / new_count

theorem average_and_variance_after_adding_datapoint :
  ∃ (x s : ℝ), new_average x ∧ new_variance s ∧ x = initial_average ∧ s < initial_variance :=
sorry

end NUMINAMATH_CALUDE_average_and_variance_after_adding_datapoint_l785_78594


namespace NUMINAMATH_CALUDE_meat_division_l785_78507

theorem meat_division (pot1_weight pot2_weight total_meat : ℕ) 
  (h1 : pot1_weight = 645)
  (h2 : pot2_weight = 237)
  (h3 : total_meat = 1000) :
  ∃ (meat1 meat2 : ℕ),
    meat1 + meat2 = total_meat ∧
    pot1_weight + meat1 = pot2_weight + meat2 ∧
    meat1 = 296 ∧
    meat2 = 704 := by
  sorry

#check meat_division

end NUMINAMATH_CALUDE_meat_division_l785_78507


namespace NUMINAMATH_CALUDE_lucky_years_2023_to_2027_l785_78523

def isLuckyYear (year : Nat) : Prop :=
  ∃ (month day : Nat), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

theorem lucky_years_2023_to_2027 : 
  ¬(isLuckyYear 2023) ∧
  (isLuckyYear 2024) ∧
  (isLuckyYear 2025) ∧
  (isLuckyYear 2026) ∧
  (isLuckyYear 2027) := by
  sorry

end NUMINAMATH_CALUDE_lucky_years_2023_to_2027_l785_78523


namespace NUMINAMATH_CALUDE_four_player_tournament_games_l785_78564

/-- The number of games in a round-robin tournament with n players -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 4 players, where each player plays against 
    every other player exactly once, the total number of games is 6 -/
theorem four_player_tournament_games : 
  num_games 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_four_player_tournament_games_l785_78564


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l785_78516

theorem decimal_sum_to_fraction :
  (0.3 : ℚ) + 0.04 + 0.005 + 0.0006 + 0.00007 = 34567 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l785_78516


namespace NUMINAMATH_CALUDE_inverse_mod_89_l785_78563

theorem inverse_mod_89 (h : (5⁻¹ : ZMod 89) = 39) : (25⁻¹ : ZMod 89) = 8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_89_l785_78563


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l785_78548

/-- Given a cube with surface area 294 square centimeters, its volume is 343 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ, s > 0 → 6 * s^2 = 294 → s^3 = 343 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l785_78548


namespace NUMINAMATH_CALUDE_golf_club_average_rounds_l785_78513

/-- Represents the data for golfers and rounds played -/
structure GolfData where
  rounds : List Nat
  golfers : List Nat

/-- Calculates the average rounds played and rounds to the nearest whole number -/
def averageRoundsRounded (data : GolfData) : Nat :=
  let totalRounds := (List.zip data.rounds data.golfers).map (fun (r, g) => r * g) |>.sum
  let totalGolfers := data.golfers.sum
  Int.toNat ((totalRounds * 2 + totalGolfers) / (2 * totalGolfers))

/-- Theorem stating that for the given golf data, the rounded average is 3 -/
theorem golf_club_average_rounds : 
  averageRoundsRounded { rounds := [1, 2, 3, 4, 5], golfers := [6, 3, 2, 4, 4] } = 3 := by
  sorry

end NUMINAMATH_CALUDE_golf_club_average_rounds_l785_78513


namespace NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l785_78575

theorem cos_pi_sixth_minus_alpha (α : ℝ) 
  (h : Real.sin (α + π / 6) + Real.cos α = -Real.sqrt 3 / 3) : 
  Real.cos (π / 6 - α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_minus_alpha_l785_78575


namespace NUMINAMATH_CALUDE_complex_moduli_product_l785_78533

theorem complex_moduli_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_moduli_product_l785_78533


namespace NUMINAMATH_CALUDE_min_y_value_l785_78540

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 56*y) : 
  y ≥ 28 - 2 * Real.sqrt 212 := by
sorry

end NUMINAMATH_CALUDE_min_y_value_l785_78540


namespace NUMINAMATH_CALUDE_constant_rate_walking_l785_78518

/-- Given a constant walking rate where 600 metres are covered in 4 minutes,
    prove that the distance covered in 6 minutes is 900 metres. -/
theorem constant_rate_walking (rate : ℝ) (h1 : rate > 0) (h2 : rate * 4 = 600) :
  rate * 6 = 900 := by
  sorry

end NUMINAMATH_CALUDE_constant_rate_walking_l785_78518


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l785_78569

theorem complex_expression_evaluation :
  let a : ℝ := 3.67
  let b : ℝ := 4.83
  let c : ℝ := 2.57
  let d : ℝ := -0.12
  let x : ℝ := 7.25
  let y : ℝ := -0.55
  
  let expression : ℝ := (3*a * (4*b - 2*y)^2) / (5*c * d^3 * 0.5*x) - (2*x * y^3) / (a * b^2 * c)
  
  ∃ ε > 0, |expression - (-57.179729)| < ε ∧ ε < 0.000001 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l785_78569


namespace NUMINAMATH_CALUDE_flagpole_shadow_length_l785_78532

theorem flagpole_shadow_length 
  (flagpole_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : building_height = 28)
  (h3 : building_shadow = 70)
  : ∃ (flagpole_shadow : ℝ), 
    flagpole_height / flagpole_shadow = building_height / building_shadow ∧ 
    flagpole_shadow = 45 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_shadow_length_l785_78532


namespace NUMINAMATH_CALUDE_middle_number_proof_l785_78565

theorem middle_number_proof (A B C : ℝ) (hC : C = 56) (hDiff : C - A = 32) (hRatio : B / C = 5 / 7) : B = 40 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l785_78565


namespace NUMINAMATH_CALUDE_inscribed_cylinder_properties_l785_78522

/-- A right circular cylinder inscribed in a right circular cone -/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ
  cylinder_height : ℝ
  /-- The cylinder's diameter equals its height -/
  height_eq_diameter : cylinder_height = 2 * cylinder_radius
  /-- The axes of the cylinder and cone coincide -/
  axes_coincide : True

/-- The space left in the cone above the cylinder -/
def space_above_cylinder (c : InscribedCylinder) : ℝ :=
  c.cone_altitude - c.cylinder_height

theorem inscribed_cylinder_properties (c : InscribedCylinder) 
  (h1 : c.cone_diameter = 16) 
  (h2 : c.cone_altitude = 20) : 
  c.cylinder_radius = 40 / 9 ∧ space_above_cylinder c = 100 / 9 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_cylinder_properties_l785_78522


namespace NUMINAMATH_CALUDE_remainder_theorem_l785_78570

theorem remainder_theorem (n : ℤ) (h : n % 11 = 5) : (4 * n - 9) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l785_78570


namespace NUMINAMATH_CALUDE_max_distance_P_to_D_l785_78538

/-- A square with side length 1 in a 2D plane -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_square : side_length = 1)

/-- A point P in the same plane as the square -/
def P : ℝ × ℝ := sorry

/-- Distance between two points in 2D plane -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the maximum distance between P and D -/
theorem max_distance_P_to_D (square : Square) 
  (h1 : distance P square.A = u)
  (h2 : distance P square.B = v)
  (h3 : distance P square.C = w)
  (h4 : u^2 + w^2 = v^2) : 
  ∃ (max_dist : ℝ), max_dist = Real.sqrt 2 ∧ 
    ∀ (P' : ℝ × ℝ), 
      distance P' square.A = u → 
      distance P' square.B = v → 
      distance P' square.C = w → 
      u^2 + w^2 = v^2 → 
      distance P' square.D ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_max_distance_P_to_D_l785_78538


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l785_78547

/-- Given a geometric sequence {a_n} with positive common ratio q,
    if a_3 · a_9 = (a_5)^2, then q = 1. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  q > 0 →
  (∀ n, a (n + 1) = a n * q) →
  a 3 * a 9 = (a 5)^2 →
  q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l785_78547


namespace NUMINAMATH_CALUDE_height_after_growth_spurt_height_approximation_l785_78526

/-- Calculates the height of a person after a year of growth with specific conditions. -/
theorem height_after_growth_spurt (initial_height : ℝ) 
  (initial_growth_rate : ℝ) (initial_growth_months : ℕ) 
  (growth_increase_rate : ℝ) (total_months : ℕ) : ℝ :=
  let inches_to_meters := 0.0254
  let height_after_initial_growth := initial_height + initial_growth_rate * initial_growth_months
  let remaining_months := total_months - initial_growth_months
  let first_variable_growth := initial_growth_rate * (1 + growth_increase_rate)
  let variable_growth_sum := first_variable_growth * 
    (1 - (1 + growth_increase_rate) ^ remaining_months) / growth_increase_rate
  (height_after_initial_growth + variable_growth_sum) * inches_to_meters

/-- The height after growth spurt is approximately 2.59 meters. -/
theorem height_approximation : 
  ∃ ε > 0, |height_after_growth_spurt 66 2 3 0.1 12 - 2.59| < ε :=
sorry

end NUMINAMATH_CALUDE_height_after_growth_spurt_height_approximation_l785_78526


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l785_78562

theorem magical_red_knights_fraction 
  (total_knights : ℕ) 
  (total_knights_pos : total_knights > 0)
  (red_knights : ℕ) 
  (blue_knights : ℕ) 
  (magical_knights : ℕ) 
  (red_knights_fraction : red_knights = (3 * total_knights) / 8)
  (blue_knights_fraction : blue_knights = total_knights - red_knights)
  (magical_knights_fraction : magical_knights = total_knights / 4)
  (magical_ratio : ∃ (p q : ℕ) (p_pos : p > 0) (q_pos : q > 0), 
    red_knights * p * 3 = blue_knights * p * q ∧ 
    red_knights * p + blue_knights * p = magical_knights * q) :
  ∃ (p q : ℕ) (p_pos : p > 0) (q_pos : q > 0), 
    7 * p = 3 * q ∧ 
    red_knights * p = magical_knights * q := by
  sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l785_78562


namespace NUMINAMATH_CALUDE_f_extrema_l785_78510

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem f_extrema :
  let a : ℝ := 0
  let b : ℝ := Real.pi / 2
  (∀ x ∈ Set.Icc a b, f x ≤ f (Real.pi / 6)) ∧
  (∀ x ∈ Set.Icc a b, f (Real.pi / 2) ≤ f x) := by
  sorry

#check f_extrema

end NUMINAMATH_CALUDE_f_extrema_l785_78510


namespace NUMINAMATH_CALUDE_solution_set_implies_m_range_l785_78505

open Real

theorem solution_set_implies_m_range (m : ℝ) :
  (∀ x : ℝ, |x - 3| - 2 - (-|x + 1| + 4) ≥ m + 1) →
  m ≤ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_range_l785_78505


namespace NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l785_78511

/-- A fair die with sides numbered from 1 to 6 -/
def Die : Type := Fin 6

/-- The set of possible outcomes when rolling two dice -/
def TwoRolls : Type := Die × Die

/-- Function to check if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The sum of two dice rolls -/
def rollSum (roll : TwoRolls) : ℕ := sorry

/-- The set of all possible outcomes when rolling two dice -/
def allOutcomes : Finset TwoRolls := sorry

/-- The set of outcomes where the sum is prime -/
def primeOutcomes : Finset TwoRolls := sorry

/-- Theorem: The probability of rolling a prime sum with two fair dice is 5/12 -/
theorem probability_prime_sum_two_dice : 
  (Finset.card primeOutcomes : ℚ) / (Finset.card allOutcomes : ℚ) = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l785_78511


namespace NUMINAMATH_CALUDE_no_five_digit_sum_20_div_9_l785_78524

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem no_five_digit_sum_20_div_9 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → ¬(n % 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_five_digit_sum_20_div_9_l785_78524


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l785_78585

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l785_78585


namespace NUMINAMATH_CALUDE_complement_of_P_is_singleton_two_l785_78551

def U : Set Int := {-1, 0, 1, 2}

def P : Set Int := {x ∈ U | x^2 < 2}

theorem complement_of_P_is_singleton_two :
  (U \ P) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_is_singleton_two_l785_78551


namespace NUMINAMATH_CALUDE_solution_set_inequality_l785_78587

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l785_78587


namespace NUMINAMATH_CALUDE_discount_rate_calculation_l785_78596

def marked_price : ℝ := 240
def selling_price : ℝ := 120

theorem discount_rate_calculation : 
  (marked_price - selling_price) / marked_price * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_discount_rate_calculation_l785_78596


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l785_78541

theorem fraction_equation_solution (x : ℝ) : (x - 3) / (x + 3) = 2 → x = -9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l785_78541


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l785_78549

theorem sum_of_roots_quadratic : 
  let a : ℝ := 1
  let b : ℝ := -8
  let c : ℝ := -7
  let sum_of_roots := -b / a
  sum_of_roots = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l785_78549


namespace NUMINAMATH_CALUDE_oabc_shape_oabc_not_rhombus_l785_78514

/-- Given distinct points A, B, and C on a coordinate plane with origin O,
    prove that OABC can form either a parallelogram or a straight line, but not a rhombus. -/
theorem oabc_shape (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (2*x₁ - x₂, 2*y₁ - y₂) ∧ (x₂, y₂) ≠ (2*x₁ - x₂, 2*y₁ - y₂)) :
  (∃ (k : ℝ), k ≠ 0 ∧ k ≠ 1 ∧ x₂ = k * x₁ ∧ y₂ = k * y₁) ∨ 
  (x₁ + x₂ = 2*x₁ - x₂ ∧ y₁ + y₂ = 2*y₁ - y₂) :=
by sorry

/-- The figure OABC cannot form a rhombus. -/
theorem oabc_not_rhombus (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (2*x₁ - x₂, 2*y₁ - y₂) ∧ (x₂, y₂) ≠ (2*x₁ - x₂, 2*y₁ - y₂)) :
  ¬(x₁^2 + y₁^2 = x₂^2 + y₂^2 ∧ 
    x₁^2 + y₁^2 = (2*x₁ - x₂)^2 + (2*y₁ - y₂)^2 ∧ 
    x₂^2 + y₂^2 = (2*x₁ - x₂)^2 + (2*y₁ - y₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_oabc_shape_oabc_not_rhombus_l785_78514


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l785_78519

theorem sum_of_coefficients_zero (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 2) = 2 * x^2 + 5 * x + 3) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l785_78519


namespace NUMINAMATH_CALUDE_fraction_equality_l785_78555

theorem fraction_equality (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m^4 / n^5) * (n^4 / m^3) = m / n := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l785_78555


namespace NUMINAMATH_CALUDE_total_books_l785_78571

theorem total_books (tim_books sam_books : ℕ) 
  (h1 : tim_books = 44) 
  (h2 : sam_books = 52) : 
  tim_books + sam_books = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l785_78571


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l785_78515

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation z(1+i³) = i
def equation (z : ℂ) : Prop := z * (1 + i^3) = i

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l785_78515


namespace NUMINAMATH_CALUDE_correct_aprons_tomorrow_l785_78544

def aprons_to_sew_tomorrow (total : ℕ) (already_sewn : ℕ) (today_multiplier : ℕ) : ℕ :=
  let today_sewn := already_sewn * today_multiplier
  let total_sewn := already_sewn + today_sewn
  let remaining := total - total_sewn
  remaining / 2

theorem correct_aprons_tomorrow :
  aprons_to_sew_tomorrow 150 13 3 = 49 := by
  sorry

end NUMINAMATH_CALUDE_correct_aprons_tomorrow_l785_78544


namespace NUMINAMATH_CALUDE_max_regions_intersected_by_line_l785_78500

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a line in 3D space -/
structure Line where
  -- Add necessary fields here
  mk :: -- Constructor

/-- The number of regions that the planes of a tetrahedron divide space into -/
def num_regions_tetrahedron : ℕ := 15

/-- The maximum number of regions a line can intersect -/
def max_intersected_regions (t : Tetrahedron) (l : Line) : ℕ := sorry

/-- Theorem stating the maximum number of regions a line can intersect -/
theorem max_regions_intersected_by_line (t : Tetrahedron) :
  ∃ l : Line, max_intersected_regions t l = 5 ∧
  ∀ l' : Line, max_intersected_regions t l' ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_regions_intersected_by_line_l785_78500


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l785_78554

theorem quadratic_equation_result (x : ℝ) (h : x^2 - x - 1 = 0) : 2*x^2 - 2*x + 2021 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l785_78554


namespace NUMINAMATH_CALUDE_problem_1_l785_78559

theorem problem_1 : (5 / 17) * (-4) - (5 / 17) * 15 + (-5 / 17) * (-2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l785_78559


namespace NUMINAMATH_CALUDE_town_population_is_300_l785_78561

/-- The number of females attending the meeting -/
def females_attending : ℕ := 50

/-- The number of males attending the meeting -/
def males_attending : ℕ := 2 * females_attending

/-- The total number of people attending the meeting -/
def total_attending : ℕ := females_attending + males_attending

/-- The total population of the town -/
def town_population : ℕ := 2 * total_attending

theorem town_population_is_300 : town_population = 300 := by
  sorry

end NUMINAMATH_CALUDE_town_population_is_300_l785_78561


namespace NUMINAMATH_CALUDE_cos_double_angle_with_tan_l785_78512

theorem cos_double_angle_with_tan (α : Real) (h : Real.tan α = 1/2) : 
  Real.cos (2 * α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_with_tan_l785_78512


namespace NUMINAMATH_CALUDE_system_solutions_correct_l785_78520

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℚ, x - y = 2 ∧ 2*x + y = 7 ∧ x = 3 ∧ y = 1) ∧
  -- System 2
  (∃ x y : ℚ, x - 2*y = 3 ∧ (1/2)*x + (3/4)*y = 13/4 ∧ x = 5 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l785_78520


namespace NUMINAMATH_CALUDE_secret_spread_days_l785_78535

/-- The number of people who know the secret after n days -/
def people_knowing_secret (n : ℕ) : ℕ :=
  (3^(n+1) - 1) / 2

/-- The proposition that it takes 7 days for at least 2186 people to know the secret -/
theorem secret_spread_days : ∃ n : ℕ, n = 7 ∧ 
  people_knowing_secret (n - 1) < 2186 ∧ people_knowing_secret n ≥ 2186 :=
sorry

end NUMINAMATH_CALUDE_secret_spread_days_l785_78535


namespace NUMINAMATH_CALUDE_age_difference_l785_78584

theorem age_difference (louis_age jerica_age matilda_age : ℕ) : 
  louis_age = 14 →
  jerica_age = 2 * louis_age →
  matilda_age = 35 →
  matilda_age - jerica_age = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l785_78584


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l785_78577

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l785_78577


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l785_78509

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) →
  (3 * q^2 + 9 * q - 21 = 0) →
  (3*p - 4) * (6*q - 8) = 122 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l785_78509


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_at_least_one_false_l785_78583

theorem not_p_and_q_implies_at_least_one_false (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_at_least_one_false_l785_78583


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l785_78503

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_positive : ∀ n, 0 < a n) 
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q) 
  (h_sum_1_2 : a 1 + a 2 = 3/4)
  (h_sum_3_to_6 : a 3 + a 4 + a 5 + a 6 = 15) :
  a 7 + a 8 + a 9 = 112 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l785_78503


namespace NUMINAMATH_CALUDE_number_of_balls_in_box_l785_78582

theorem number_of_balls_in_box : ∃ x : ℕ, x > 20 ∧ x < 30 ∧ (x - 20 = 30 - x) ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_of_balls_in_box_l785_78582


namespace NUMINAMATH_CALUDE_probability_second_red_given_first_red_is_five_ninths_l785_78502

/-- Represents the probability of drawing a red ball as the second ball, given that the first ball drawn was red, from a set of 10 balls containing 6 red balls and 4 white balls. -/
def probability_second_red_given_first_red (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) : ℚ :=
  if total_balls = red_balls + white_balls ∧ red_balls > 0 then
    (red_balls - 1) / (total_balls - 1)
  else
    0

/-- Theorem stating that the probability of drawing a red ball as the second ball, given that the first ball drawn was red, from a set of 10 balls containing 6 red balls and 4 white balls, is 5/9. -/
theorem probability_second_red_given_first_red_is_five_ninths :
  probability_second_red_given_first_red 10 6 4 = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_red_given_first_red_is_five_ninths_l785_78502


namespace NUMINAMATH_CALUDE_forty_fifth_turn_turning_position_1978_to_2010_l785_78597

-- Define the sequence of turning positions
def turningPosition (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    (1 + n / 2) * (n / 2) + 1
  else
    ((n + 1) / 2)^2 + 1

-- Theorem for the 45th turning position
theorem forty_fifth_turn : turningPosition 45 = 530 := by
  sorry

-- Theorem for the turning position between 1978 and 2010
theorem turning_position_1978_to_2010 :
  ∃ n : ℕ, turningPosition n = 1981 ∧
    1978 < turningPosition n ∧ turningPosition n < 2010 ∧
    ∀ m : ℕ, m ≠ n →
      (1978 < turningPosition m → turningPosition m ≥ 2010) ∨
      (turningPosition m ≤ 1978) := by
  sorry

end NUMINAMATH_CALUDE_forty_fifth_turn_turning_position_1978_to_2010_l785_78597


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l785_78539

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a → sum_condition a → 2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l785_78539


namespace NUMINAMATH_CALUDE_system_one_solution_set_system_two_solution_set_l785_78534

-- System 1
theorem system_one_solution_set :
  {x : ℝ | 3*x > x + 6 ∧ (1/2)*x < -x + 5} = {x : ℝ | 3 < x ∧ x < 10/3} := by sorry

-- System 2
theorem system_two_solution_set :
  {x : ℝ | 2*x - 1 < 5 - 2*(x-1) ∧ (3+5*x)/3 > 1} = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_system_one_solution_set_system_two_solution_set_l785_78534


namespace NUMINAMATH_CALUDE_four_sticks_impossible_other_sticks_possible_lolly_stick_triangle_l785_78508

/-- A function that checks if it's possible to form a triangle with given number of lolly sticks -/
def can_form_triangle (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a + b + c = n ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that it's impossible to form a triangle with 4 lolly sticks -/
theorem four_sticks_impossible : ¬ can_form_triangle 4 :=
sorry

/-- Theorem stating that it's possible to form triangles with 3, 5, 6, and 7 lolly sticks -/
theorem other_sticks_possible :
  can_form_triangle 3 ∧ can_form_triangle 5 ∧ can_form_triangle 6 ∧ can_form_triangle 7 :=
sorry

/-- Main theorem combining the above results -/
theorem lolly_stick_triangle :
  ¬ can_form_triangle 4 ∧
  (can_form_triangle 3 ∧ can_form_triangle 5 ∧ can_form_triangle 6 ∧ can_form_triangle 7) :=
sorry

end NUMINAMATH_CALUDE_four_sticks_impossible_other_sticks_possible_lolly_stick_triangle_l785_78508


namespace NUMINAMATH_CALUDE_max_visible_cubes_is_400_l785_78525

/-- The dimension of the cube --/
def n : ℕ := 12

/-- The number of unit cubes on one face of the cube --/
def face_count : ℕ := n^2

/-- The number of unit cubes along one edge of the cube --/
def edge_count : ℕ := n

/-- The number of visible faces from a corner --/
def visible_faces : ℕ := 3

/-- The number of edges shared between two visible faces --/
def shared_edges : ℕ := 3

/-- The maximum number of visible unit cubes from a single point --/
def max_visible_cubes : ℕ := visible_faces * face_count - shared_edges * (edge_count - 1) + 1

/-- Theorem stating that the maximum number of visible unit cubes is 400 --/
theorem max_visible_cubes_is_400 : max_visible_cubes = 400 := by
  sorry

end NUMINAMATH_CALUDE_max_visible_cubes_is_400_l785_78525


namespace NUMINAMATH_CALUDE_units_digit_17_25_l785_78501

theorem units_digit_17_25 : 17^25 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_25_l785_78501


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l785_78592

theorem fuel_tank_capacity : ∃ (C : ℝ), C = 204 ∧ C > 0 := by
  -- Define the ethanol content of fuels A and B
  let ethanol_A : ℝ := 0.12
  let ethanol_B : ℝ := 0.16

  -- Define the volume of fuel A added
  let volume_A : ℝ := 66

  -- Define the total ethanol volume in the full tank
  let total_ethanol : ℝ := 30

  -- The capacity C satisfies the equation:
  -- ethanol_A * volume_A + ethanol_B * (C - volume_A) = total_ethanol
  
  sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l785_78592


namespace NUMINAMATH_CALUDE_circle_circumference_l785_78527

theorem circle_circumference (r : ℝ) (h : r > 0) : 
  (2 * r^2 = π * r^2) → (2 * π * r = 4 * r) :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_l785_78527


namespace NUMINAMATH_CALUDE_sand_lost_during_journey_l785_78531

theorem sand_lost_during_journey (initial_sand final_sand : ℝ) 
  (h1 : initial_sand = 4.1)
  (h2 : final_sand = 1.7) :
  initial_sand - final_sand = 2.4 := by
sorry

end NUMINAMATH_CALUDE_sand_lost_during_journey_l785_78531


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l785_78590

/-- The area of wrapping paper required to wrap a box on a pedestal -/
theorem wrapping_paper_area (w h p : ℝ) (hw : w > 0) (hh : h > 0) (hp : p > 0) :
  let paper_area := 4 * w * (p + h)
  paper_area = 4 * w * (p + h) :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l785_78590


namespace NUMINAMATH_CALUDE_sequence_contains_30_l785_78573

theorem sequence_contains_30 : ∃ n : ℕ+, n * (n + 1) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sequence_contains_30_l785_78573


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l785_78576

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the perpendicular bisector
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersecting_circles :
  ∀ (A B : ℝ × ℝ),
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B →
  perp_bisector ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l785_78576


namespace NUMINAMATH_CALUDE_inequality_proof_l785_78598

theorem inequality_proof (x y z w : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0)
  (hxy : x + y ≠ 0) (hzw : z + w ≠ 0) (hxyzw : x * y + z * w ≥ 0) :
  ((x + y) / (z + w) + (z + w) / (x + y))⁻¹ + 1 / 2 ≥ 
  (x / z + z / x)⁻¹ + (y / w + w / y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l785_78598


namespace NUMINAMATH_CALUDE_sqrt_31_between_5_and_6_l785_78595

theorem sqrt_31_between_5_and_6 : 5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_31_between_5_and_6_l785_78595


namespace NUMINAMATH_CALUDE_ellipse_ratio_l785_78542

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and semi-latus rectum c,
    if a² + b² - 3c² = 0, then (a + c) / (a - c) = 3 + 2√2 -/
theorem ellipse_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 - 3*c^2 = 0) : (a + c) / (a - c) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_ratio_l785_78542


namespace NUMINAMATH_CALUDE_det_A_eq_46_l785_78589

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -1; 7, 4, -3; 2, 2, 5]

theorem det_A_eq_46 : A.det = 46 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_46_l785_78589


namespace NUMINAMATH_CALUDE_four_letter_words_with_a_l785_78546

theorem four_letter_words_with_a (n : ℕ) (total_letters : ℕ) (letters_without_a : ℕ) : 
  n = 4 → 
  total_letters = 5 → 
  letters_without_a = 4 → 
  (total_letters ^ n) - (letters_without_a ^ n) = 369 :=
by sorry

end NUMINAMATH_CALUDE_four_letter_words_with_a_l785_78546


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l785_78560

theorem fuel_tank_capacity 
  (fuel_a_ethanol_percentage : ℝ)
  (fuel_b_ethanol_percentage : ℝ)
  (total_ethanol : ℝ)
  (fuel_a_volume : ℝ)
  (h1 : fuel_a_ethanol_percentage = 0.12)
  (h2 : fuel_b_ethanol_percentage = 0.16)
  (h3 : total_ethanol = 28)
  (h4 : fuel_a_volume = 99.99999999999999)
  : ∃ (capacity : ℝ), 
    fuel_a_ethanol_percentage * fuel_a_volume + 
    fuel_b_ethanol_percentage * (capacity - fuel_a_volume) = total_ethanol ∧
    capacity = 200 :=
by sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l785_78560


namespace NUMINAMATH_CALUDE_maintenance_check_time_l785_78528

theorem maintenance_check_time (initial_time : ℝ) : 
  (initial_time * 1.2 = 30) → initial_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_time_l785_78528


namespace NUMINAMATH_CALUDE_remainder_theorem_l785_78599

/-- The polynomial f(x) = x^5 - 8x^4 + 15x^3 + 20x^2 - 5x - 20 -/
def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 15*x^3 + 20*x^2 - 5*x - 20

/-- The theorem statement -/
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, f = fun x ↦ (x - 4) * q x + 216 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l785_78599


namespace NUMINAMATH_CALUDE_initial_water_amount_l785_78568

/-- 
Given a bucket with an initial amount of water, prove that this amount is 3 gallons
when adding 6.8 gallons results in a total of 9.8 gallons.
-/
theorem initial_water_amount (initial_amount : ℝ) : 
  initial_amount + 6.8 = 9.8 → initial_amount = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_amount_l785_78568


namespace NUMINAMATH_CALUDE_downstream_distance_is_100_l785_78588

/-- Represents the properties of a boat traveling in a stream -/
structure BoatTravel where
  downstream_time : ℝ
  upstream_distance : ℝ
  upstream_time : ℝ
  stream_speed : ℝ

/-- Calculates the downstream distance given boat travel properties -/
def downstream_distance (bt : BoatTravel) : ℝ :=
  sorry

/-- Theorem stating that the downstream distance is 100 km given specific conditions -/
theorem downstream_distance_is_100 (bt : BoatTravel) 
  (h1 : bt.downstream_time = 10)
  (h2 : bt.upstream_distance = 200)
  (h3 : bt.upstream_time = 25)
  (h4 : bt.stream_speed = 1) :
  downstream_distance bt = 100 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_is_100_l785_78588
