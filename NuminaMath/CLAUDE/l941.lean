import Mathlib

namespace NUMINAMATH_CALUDE_books_in_history_section_l941_94132

/-- Calculates the number of books shelved in the history section. -/
def books_shelved_in_history (initial_books : ℕ) (fiction_books : ℕ) (children_books : ℕ) 
  (misplaced_books : ℕ) (books_left : ℕ) : ℕ :=
  initial_books - fiction_books - children_books + misplaced_books - books_left

/-- Theorem stating the number of books shelved in the history section. -/
theorem books_in_history_section :
  books_shelved_in_history 51 19 8 4 16 = 12 := by
  sorry

#eval books_shelved_in_history 51 19 8 4 16

end NUMINAMATH_CALUDE_books_in_history_section_l941_94132


namespace NUMINAMATH_CALUDE_square_sum_zero_iff_both_zero_l941_94192

theorem square_sum_zero_iff_both_zero (x y : ℝ) : x^2 + y^2 = 0 ↔ x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_iff_both_zero_l941_94192


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l941_94195

def stewart_farm (horse_food_per_day : ℕ) (total_horse_food : ℕ) (num_sheep : ℕ) : Prop :=
  ∃ (num_horses : ℕ),
    horse_food_per_day * num_horses = total_horse_food ∧
    (num_sheep : ℚ) / num_horses = 5 / 7

theorem stewart_farm_ratio :
  stewart_farm 230 12880 40 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l941_94195


namespace NUMINAMATH_CALUDE_joyces_property_size_l941_94181

theorem joyces_property_size (new_property_size old_property_size pond_size suitable_land : ℝ) : 
  new_property_size = 10 * old_property_size →
  pond_size = 1 →
  suitable_land = 19 →
  new_property_size = suitable_land + pond_size →
  old_property_size = 2 := by
sorry

end NUMINAMATH_CALUDE_joyces_property_size_l941_94181


namespace NUMINAMATH_CALUDE_ethanol_percentage_fuel_B_l941_94183

-- Define the constants
def tank_capacity : ℝ := 212
def fuel_A_ethanol_percentage : ℝ := 0.12
def fuel_A_volume : ℝ := 98
def total_ethanol : ℝ := 30

-- Define the theorem
theorem ethanol_percentage_fuel_B :
  let ethanol_A := fuel_A_ethanol_percentage * fuel_A_volume
  let ethanol_B := total_ethanol - ethanol_A
  let fuel_B_volume := tank_capacity - fuel_A_volume
  (ethanol_B / fuel_B_volume) * 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ethanol_percentage_fuel_B_l941_94183


namespace NUMINAMATH_CALUDE_green_yards_calculation_l941_94197

/-- The number of yards of silk dyed green in a factory order -/
def green_yards (total_yards pink_yards : ℕ) : ℕ :=
  total_yards - pink_yards

/-- Theorem stating that the number of yards dyed green is 61921 -/
theorem green_yards_calculation :
  green_yards 111421 49500 = 61921 := by
  sorry

end NUMINAMATH_CALUDE_green_yards_calculation_l941_94197


namespace NUMINAMATH_CALUDE_cube_of_product_l941_94174

theorem cube_of_product (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_l941_94174


namespace NUMINAMATH_CALUDE_order_of_7_wrt_g_l941_94115

def g (x : ℕ) : ℕ := x^2 % 13

def g_iter (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => g (g_iter n x)

theorem order_of_7_wrt_g :
  (∀ k < 12, g_iter k 7 ≠ 7) ∧ g_iter 12 7 = 7 :=
sorry

end NUMINAMATH_CALUDE_order_of_7_wrt_g_l941_94115


namespace NUMINAMATH_CALUDE_min_product_abc_l941_94110

theorem min_product_abc (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 1 → 
  a ≤ 3*b → a ≤ 3*c → b ≤ 3*a → b ≤ 3*c → c ≤ 3*a → c ≤ 3*b → 
  a * b * c ≥ 9 / 343 := by
sorry

end NUMINAMATH_CALUDE_min_product_abc_l941_94110


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l941_94152

theorem imaginary_part_of_complex_number :
  let z : ℂ := 1 - 2 * Complex.I
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l941_94152


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l941_94185

theorem largest_divisor_of_n (n : ℕ+) (h : 450 ∣ n^2) : 
  ∀ d : ℕ, d ∣ n → d ≤ 30 ∧ 30 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l941_94185


namespace NUMINAMATH_CALUDE_total_paintable_area_l941_94165

/-- Represents the dimensions of a bedroom -/
structure Bedroom where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the wall area of a bedroom -/
def wallArea (b : Bedroom) : ℝ :=
  2 * (b.length * b.height + b.width * b.height)

/-- Calculates the paintable area of a bedroom -/
def paintableArea (b : Bedroom) (unpaintableArea : ℝ) : ℝ :=
  wallArea b - unpaintableArea

/-- The four bedrooms in Isabella's house -/
def isabellasBedrooms : List Bedroom := [
  { length := 14, width := 12, height := 9 },
  { length := 13, width := 11, height := 9 },
  { length := 15, width := 10, height := 9 },
  { length := 12, width := 12, height := 9 }
]

/-- The area occupied by doorways and windows in each bedroom -/
def unpaintableAreaPerRoom : ℝ := 70

/-- Theorem: The total area of walls to be painted in Isabella's house is 1502 square feet -/
theorem total_paintable_area :
  (isabellasBedrooms.map (fun b => paintableArea b unpaintableAreaPerRoom)).sum = 1502 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_l941_94165


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l941_94155

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 < 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l941_94155


namespace NUMINAMATH_CALUDE_problem_solution_l941_94153

theorem problem_solution (x y : ℚ) (hx : x = 3/4) (hy : y = 4/3) :
  (1/3 * x^7 * y^6) * 4 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l941_94153


namespace NUMINAMATH_CALUDE_chess_tournament_games_l941_94176

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 5 → total_games = 10 → (n * (n - 1)) / 2 = total_games → n - 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l941_94176


namespace NUMINAMATH_CALUDE_smallest_possible_a_l941_94121

theorem smallest_possible_a (P : ℤ → ℤ) (a : ℕ+) : 
  (∀ x, ∃ (c : ℤ), P x = c) →  -- P has integer coefficients
  (P 1 = a) →
  (P 3 = a) →
  (P 2 = -a) →
  (P 4 = -a) →
  (P 6 = -a) →
  (P 8 = -a) →
  (∀ b : ℕ+, b < 105 → 
    ¬(∃ Q : ℤ → ℤ, 
      (∀ x, ∃ (c : ℤ), Q x = c) ∧  -- Q has integer coefficients
      (Q 1 = b) ∧
      (Q 3 = b) ∧
      (Q 2 = -b) ∧
      (Q 4 = -b) ∧
      (Q 6 = -b) ∧
      (Q 8 = -b)
    )
  ) →
  a = 105 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l941_94121


namespace NUMINAMATH_CALUDE_line_intercepts_sum_zero_l941_94134

/-- Given a line l with equation 2x + (k - 3)y - 2k + 6 = 0, where k ≠ 3,
    if the sum of its x-intercept and y-intercept is 0, then k = 1. -/
theorem line_intercepts_sum_zero (k : ℝ) (h : k ≠ 3) :
  let l := {(x, y) : ℝ × ℝ | 2 * x + (k - 3) * y - 2 * k + 6 = 0}
  let x_intercept := (k - 3 : ℝ)
  let y_intercept := (2 : ℝ)
  x_intercept + y_intercept = 0 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_zero_l941_94134


namespace NUMINAMATH_CALUDE_geometric_properties_l941_94123

/-- A geometric figure -/
structure Figure where
  -- Add necessary properties here
  mk :: -- Constructor

/-- Defines when two figures can overlap perfectly -/
def can_overlap (f1 f2 : Figure) : Prop :=
  sorry

/-- Defines congruence between two figures -/
def congruent (f1 f2 : Figure) : Prop :=
  sorry

/-- The area of a figure -/
def area (f : Figure) : ℝ :=
  sorry

/-- The perimeter of a figure -/
def perimeter (f : Figure) : ℝ :=
  sorry

theorem geometric_properties :
  (∀ f1 f2 : Figure, can_overlap f1 f2 → congruent f1 f2) ∧
  (∀ f1 f2 : Figure, congruent f1 f2 → area f1 = area f2) ∧
  (∃ f1 f2 : Figure, area f1 = area f2 ∧ ¬congruent f1 f2) ∧
  (∃ f1 f2 : Figure, perimeter f1 = perimeter f2 ∧ ¬congruent f1 f2) :=
sorry

end NUMINAMATH_CALUDE_geometric_properties_l941_94123


namespace NUMINAMATH_CALUDE_fishing_competition_result_l941_94166

/-- The total number of days in the fishing season -/
def season_days : ℕ := 213

/-- The number of fish caught per day by the first fisherman -/
def first_fisherman_rate : ℕ := 3

/-- The number of days the second fisherman catches 1 fish per day -/
def second_fisherman_phase1_days : ℕ := 30

/-- The number of days the second fisherman catches 2 fish per day -/
def second_fisherman_phase2_days : ℕ := 60

/-- The number of fish caught per day by the second fisherman in phase 1 -/
def second_fisherman_phase1_rate : ℕ := 1

/-- The number of fish caught per day by the second fisherman in phase 2 -/
def second_fisherman_phase2_rate : ℕ := 2

/-- The number of fish caught per day by the second fisherman in phase 3 -/
def second_fisherman_phase3_rate : ℕ := 4

/-- The total number of fish caught by the first fisherman -/
def first_fisherman_total : ℕ := first_fisherman_rate * season_days

/-- The total number of fish caught by the second fisherman -/
def second_fisherman_total : ℕ :=
  second_fisherman_phase1_rate * second_fisherman_phase1_days +
  second_fisherman_phase2_rate * second_fisherman_phase2_days +
  second_fisherman_phase3_rate * (season_days - second_fisherman_phase1_days - second_fisherman_phase2_days)

theorem fishing_competition_result :
  second_fisherman_total - first_fisherman_total = 3 := by sorry

end NUMINAMATH_CALUDE_fishing_competition_result_l941_94166


namespace NUMINAMATH_CALUDE_equation_equivalence_l941_94196

theorem equation_equivalence : ∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ (x - 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l941_94196


namespace NUMINAMATH_CALUDE_april_roses_problem_l941_94142

theorem april_roses_problem (rose_price : ℕ) (roses_left : ℕ) (earnings : ℕ) :
  rose_price = 7 →
  roses_left = 4 →
  earnings = 35 →
  ∃ (initial_roses : ℕ), initial_roses = (earnings / rose_price) + roses_left ∧ initial_roses = 9 :=
by sorry

end NUMINAMATH_CALUDE_april_roses_problem_l941_94142


namespace NUMINAMATH_CALUDE_fencing_final_probability_l941_94141

theorem fencing_final_probability (p_a : ℝ) (h1 : p_a = 0.41) :
  let p_b := 1 - p_a
  p_b = 0.59 := by
sorry

end NUMINAMATH_CALUDE_fencing_final_probability_l941_94141


namespace NUMINAMATH_CALUDE_seating_arrangements_l941_94169

structure Table :=
  (chairs : ℕ)
  (couples : ℕ)

def valid_seating (t : Table) (arrangements : ℕ) : Prop :=
  t.chairs = 12 ∧
  t.couples = 6 ∧
  arrangements = 43200

theorem seating_arrangements (t : Table) :
  valid_seating t 43200 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l941_94169


namespace NUMINAMATH_CALUDE_max_value_of_function_l941_94104

theorem max_value_of_function (x : ℝ) : 
  (Real.sin x * (2 - Real.cos x)) / (5 - 4 * Real.cos x) ≤ Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l941_94104


namespace NUMINAMATH_CALUDE_circle_radius_condition_l941_94137

theorem circle_radius_condition (x y c : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 2*y + c = 0 → (x + 4)^2 + (y + 1)^2 = 25) → c = -8 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_condition_l941_94137


namespace NUMINAMATH_CALUDE_triangle_inequality_l941_94170

theorem triangle_inequality (A B C m n l : ℝ) (h : A + B + C = π) : 
  (m^2 + Real.tan (A/2) * Real.tan (B/2))^(1/2) + 
  (n^2 + Real.tan (B/2) * Real.tan (C/2))^(1/2) + 
  (l^2 + Real.tan (C/2) * Real.tan (A/2))^(1/2) ≤ 
  (3 * (m^2 + n^2 + l^2 + 1))^(1/2) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l941_94170


namespace NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_alpha_l941_94147

theorem cos_two_pi_thirds_minus_alpha (α : ℝ) (h : Real.sin (α - π/6) = 3/5) :
  Real.cos (2*π/3 - α) = 3/5 := by sorry

end NUMINAMATH_CALUDE_cos_two_pi_thirds_minus_alpha_l941_94147


namespace NUMINAMATH_CALUDE_round_to_nearest_whole_number_l941_94136

theorem round_to_nearest_whole_number : 
  let x : ℝ := 6703.4999
  ‖x - 6703‖ < ‖x - 6704‖ :=
by sorry

end NUMINAMATH_CALUDE_round_to_nearest_whole_number_l941_94136


namespace NUMINAMATH_CALUDE_inequality_proof_l941_94194

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^2 + b^2) > (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l941_94194


namespace NUMINAMATH_CALUDE_car_speed_from_tire_rotation_l941_94156

/-- Given a tire rotating at a certain rate with a specific circumference,
    calculate the speed of the car in km/h. -/
theorem car_speed_from_tire_rotation 
  (revolutions_per_minute : ℝ) 
  (tire_circumference : ℝ) 
  (h1 : revolutions_per_minute = 400) 
  (h2 : tire_circumference = 5) : 
  (revolutions_per_minute * tire_circumference * 60) / 1000 = 120 := by
  sorry

#check car_speed_from_tire_rotation

end NUMINAMATH_CALUDE_car_speed_from_tire_rotation_l941_94156


namespace NUMINAMATH_CALUDE_investment_calculation_l941_94128

/-- Calculates the investment amount given dividend information -/
theorem investment_calculation (face_value premium dividend_rate dividend_received : ℚ) : 
  face_value = 100 →
  premium = 20 / 100 →
  dividend_rate = 5 / 100 →
  dividend_received = 600 →
  (dividend_received / (face_value * dividend_rate)) * (face_value * (1 + premium)) = 14400 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l941_94128


namespace NUMINAMATH_CALUDE_min_nSn_value_l941_94148

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))

/-- The main theorem -/
theorem min_nSn_value (seq : ArithmeticSequence) 
    (h1 : seq.S 10 = 0) 
    (h2 : seq.S 15 = 25) : 
  ∃ n : ℕ, ∀ m : ℕ, n * seq.S n ≤ m * seq.S m ∧ n * seq.S n = -49 := by
  sorry

end NUMINAMATH_CALUDE_min_nSn_value_l941_94148


namespace NUMINAMATH_CALUDE_equation_equivalence_l941_94179

theorem equation_equivalence (a b c : ℝ) (h : b > 0) :
  (a / Real.sqrt (18 * b)) * (c / Real.sqrt (72 * b)) = 1 →
  a * c = 36 * b :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l941_94179


namespace NUMINAMATH_CALUDE_line_intersects_circle_l941_94113

/-- The line l in polar coordinates -/
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = Real.sqrt 2

/-- The circle C in Cartesian coordinates -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The line l in Cartesian coordinates -/
def line_l_cartesian (x y : ℝ) : Prop := x + y = 2

/-- Theorem stating that the line l intersects the circle C -/
theorem line_intersects_circle :
  ∃ (x y : ℝ), line_l_cartesian x y ∧ circle_C x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l941_94113


namespace NUMINAMATH_CALUDE_impossible_odd_sum_arrangement_l941_94111

theorem impossible_odd_sum_arrangement : 
  ¬ ∃ (seq : Fin 2018 → ℕ), 
    (∀ i : Fin 2018, 1 ≤ seq i ∧ seq i ≤ 2018) ∧ 
    (∀ i : Fin 2018, seq i ≠ seq ((i + 1) % 2018)) ∧
    (∀ i : Fin 2018, Odd (seq i + seq ((i + 1) % 2018) + seq ((i + 2) % 2018))) :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_odd_sum_arrangement_l941_94111


namespace NUMINAMATH_CALUDE_sophomores_in_sample_l941_94193

-- Define the total number of students in each grade
def freshmen : ℕ := 400
def sophomores : ℕ := 600
def juniors : ℕ := 500

-- Define the total sample size
def sample_size : ℕ := 100

-- Theorem to prove
theorem sophomores_in_sample :
  (sophomores * sample_size) / (freshmen + sophomores + juniors) = 40 := by
  sorry

end NUMINAMATH_CALUDE_sophomores_in_sample_l941_94193


namespace NUMINAMATH_CALUDE_train_length_l941_94190

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) : 
  speed_kmph = 72 → crossing_time = 7 → 
  (speed_kmph * 1000 / 3600) * crossing_time = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l941_94190


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l941_94109

theorem arithmetic_expression_equality : 54 + (42 / 14) + (27 * 17) - 200 - (360 / 6) + 2^4 = 272 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l941_94109


namespace NUMINAMATH_CALUDE_mixture_composition_l941_94108

theorem mixture_composition (water_percent_1 water_percent_2 mixture_percent : ℝ)
  (parts_1 : ℝ) (h1 : water_percent_1 = 0.20)
  (h2 : water_percent_2 = 0.35) (h3 : parts_1 = 10)
  (h4 : mixture_percent = 0.24285714285714285) : ∃ (parts_2 : ℝ),
  parts_2 = 4 ∧ 
  (water_percent_1 * parts_1 + water_percent_2 * parts_2) / (parts_1 + parts_2) = mixture_percent :=
by sorry

end NUMINAMATH_CALUDE_mixture_composition_l941_94108


namespace NUMINAMATH_CALUDE_one_line_passes_through_trisection_point_l941_94198

-- Define the points
def A : ℝ × ℝ := (-3, 6)
def B : ℝ × ℝ := (6, -3)
def P : ℝ × ℝ := (2, 3)

-- Define the trisection points
def T₁ : ℝ × ℝ := (0, 3)
def T₂ : ℝ × ℝ := (3, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + y - 9 = 0

-- Theorem statement
theorem one_line_passes_through_trisection_point :
  ∃ T, (T = T₁ ∨ T = T₂) ∧ 
       line_equation P.1 P.2 ∧
       line_equation T.1 T.2 :=
sorry

end NUMINAMATH_CALUDE_one_line_passes_through_trisection_point_l941_94198


namespace NUMINAMATH_CALUDE_x_range_for_equation_l941_94154

theorem x_range_for_equation (x y : ℝ) (h : x / y = x - y) : x ≥ 4 ∨ x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_equation_l941_94154


namespace NUMINAMATH_CALUDE_sum_consecutive_odd_integers_divisible_by_16_l941_94173

theorem sum_consecutive_odd_integers_divisible_by_16 :
  let start := 2101
  let count := 15
  let sequence := List.range count |>.map (fun i => start + 2 * i)
  sequence.sum % 16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_odd_integers_divisible_by_16_l941_94173


namespace NUMINAMATH_CALUDE_robotics_club_subjects_l941_94149

theorem robotics_club_subjects (total : ℕ) (math : ℕ) (physics : ℕ) (cs : ℕ) 
  (math_physics : ℕ) (math_cs : ℕ) (physics_cs : ℕ) (all_three : ℕ) : 
  total = 60 ∧ 
  math = 42 ∧ 
  physics = 35 ∧ 
  cs = 15 ∧ 
  math_physics = 25 ∧ 
  math_cs = 10 ∧ 
  physics_cs = 5 ∧ 
  all_three = 4 → 
  total - (math + physics + cs - math_physics - math_cs - physics_cs + all_three) = 0 :=
by sorry

end NUMINAMATH_CALUDE_robotics_club_subjects_l941_94149


namespace NUMINAMATH_CALUDE_complex_modulus_one_minus_i_l941_94189

theorem complex_modulus_one_minus_i :
  let z : ℂ := 1 - I
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_one_minus_i_l941_94189


namespace NUMINAMATH_CALUDE_banana_distribution_l941_94139

theorem banana_distribution (total_bananas : Nat) (num_groups : Nat) (bananas_per_group : Nat) :
  total_bananas = 407 →
  num_groups = 11 →
  bananas_per_group = total_bananas / num_groups →
  bananas_per_group = 37 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l941_94139


namespace NUMINAMATH_CALUDE_cos_2theta_value_l941_94160

theorem cos_2theta_value (θ : ℝ) :
  let a : ℝ × ℝ := (1, Real.cos (2 * x))
  let b : ℝ × ℝ := (Real.sin (2 * x), -Real.sqrt 3)
  let f : ℝ → ℝ := λ x => a.1 * b.1 + a.2 * b.2
  f (θ / 2 + 2 * Real.pi / 3) = 6 / 5 →
  Real.cos (2 * θ) = 7 / 25 :=
by sorry

end NUMINAMATH_CALUDE_cos_2theta_value_l941_94160


namespace NUMINAMATH_CALUDE_no_common_root_l941_94112

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ x : ℝ, x^2 + b*x + c = 0 ∧ x^2 + a*x + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_root_l941_94112


namespace NUMINAMATH_CALUDE_base_equation_solution_l941_94159

/-- Converts a list of digits in base b to its decimal representation -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if a list of digits is valid in base b -/
def valid_digits (digits : List Nat) (b : Nat) : Prop :=
  digits.all (· < b)

theorem base_equation_solution :
  ∃! b : Nat, b > 1 ∧
    valid_digits [3, 4, 6, 4] b ∧
    valid_digits [4, 6, 2, 3] b ∧
    valid_digits [1, 0, 0, 0, 0] b ∧
    to_decimal [3, 4, 6, 4] b + to_decimal [4, 6, 2, 3] b = to_decimal [1, 0, 0, 0, 0] b :=
by
  sorry

end NUMINAMATH_CALUDE_base_equation_solution_l941_94159


namespace NUMINAMATH_CALUDE_triangle_area_l941_94172

/-- The area of a triangle with base 15 and height p is equal to 15p/2 -/
theorem triangle_area (p : ℝ) : 
  (1/2 : ℝ) * 15 * p = 15 * p / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l941_94172


namespace NUMINAMATH_CALUDE_square_not_always_positive_l941_94124

theorem square_not_always_positive : ∃ (a : ℝ), ¬(a^2 > 0) := by sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l941_94124


namespace NUMINAMATH_CALUDE_expression_value_l941_94138

theorem expression_value (a b : ℤ) (ha : a = 4) (hb : b = -3) : -a - b^3 + a*b = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l941_94138


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l941_94130

/-- Given an arithmetic sequence {a_n} with common difference d,
    S_n is the sum of the first n terms. -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d ∧ S n = n * a 1 + n * (n - 1) * d / 2

theorem arithmetic_sequence_ratio
  (a : ℕ → ℚ) (d : ℚ) (S : ℕ → ℚ)
  (h : arithmetic_sequence a d S)
  (h_ratio : S 5 / S 3 = 3) :
  a 5 / a 3 = 17 / 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l941_94130


namespace NUMINAMATH_CALUDE_target_probabilities_l941_94150

/-- The probability of hitting the target for both A and B -/
def p : ℝ := 0.6

/-- The probability that both A and B hit the target -/
def prob_both_hit : ℝ := p * p

/-- The probability that exactly one of A and B hits the target -/
def prob_one_hit : ℝ := 2 * p * (1 - p)

/-- The probability that at least one of A and B hits the target -/
def prob_at_least_one_hit : ℝ := 1 - (1 - p) * (1 - p)

theorem target_probabilities :
  prob_both_hit = 0.36 ∧
  prob_one_hit = 0.48 ∧
  prob_at_least_one_hit = 0.84 := by
  sorry

end NUMINAMATH_CALUDE_target_probabilities_l941_94150


namespace NUMINAMATH_CALUDE_element_four_in_B_l941_94140

def U : Set ℕ := {x | x ≤ 7}

theorem element_four_in_B (A B : Set ℕ) 
  (h1 : U = A ∪ B) 
  (h2 : A ∩ (Bᶜ) = {2, 3, 5, 7}) : 
  4 ∈ B := by
  sorry

end NUMINAMATH_CALUDE_element_four_in_B_l941_94140


namespace NUMINAMATH_CALUDE_ramesh_refrigerator_price_l941_94177

/-- The price Ramesh paid for the refrigerator --/
def price_paid (labelled_price : ℝ) : ℝ :=
  0.8 * labelled_price + 125 + 250

/-- The theorem stating the price Ramesh paid for the refrigerator --/
theorem ramesh_refrigerator_price :
  ∃ (labelled_price : ℝ),
    1.2 * labelled_price = 19200 ∧
    price_paid labelled_price = 13175 := by
  sorry

end NUMINAMATH_CALUDE_ramesh_refrigerator_price_l941_94177


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_seven_l941_94199

theorem sqrt_expression_equals_seven :
  (Real.sqrt 3 - 2)^2 + Real.sqrt 12 + 6 * Real.sqrt (1/3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_seven_l941_94199


namespace NUMINAMATH_CALUDE_not_necessarily_p_or_q_l941_94131

theorem not_necessarily_p_or_q (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (p q : Prop), (¬p ∧ ¬(p ∧ q)) → (p ∨ q) :=
by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_p_or_q_l941_94131


namespace NUMINAMATH_CALUDE_triangle_circle_areas_l941_94184

theorem triangle_circle_areas (r s t : ℝ) : 
  r + s = 6 →
  r + t = 8 →
  s + t = 10 →
  r > 0 →
  s > 0 →
  t > 0 →
  π * r^2 + π * s^2 + π * t^2 = 36 * π :=
by sorry

end NUMINAMATH_CALUDE_triangle_circle_areas_l941_94184


namespace NUMINAMATH_CALUDE_probability_b_greater_than_a_l941_94171

def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {1, 2, 3}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (A.product B).filter (fun p => p.2 > p.1)

theorem probability_b_greater_than_a :
  (favorable_outcomes.card : ℚ) / ((A.card * B.card) : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_b_greater_than_a_l941_94171


namespace NUMINAMATH_CALUDE_seashell_count_l941_94186

/-- Given a collection of seashells with specific counts for different colors,
    calculate the number of shells that are not red, green, or blue. -/
theorem seashell_count (total : ℕ) (red green blue : ℕ) 
    (h_total : total = 501)
    (h_red : red = 123)
    (h_green : green = 97)
    (h_blue : blue = 89) :
    total - (red + green + blue) = 192 := by
  sorry

end NUMINAMATH_CALUDE_seashell_count_l941_94186


namespace NUMINAMATH_CALUDE_cube_dimension_reduction_l941_94188

theorem cube_dimension_reduction (initial_face_area : ℝ) (reduction : ℝ) : 
  initial_face_area = 36 ∧ reduction = 1 → 
  (3 : ℝ) * (Real.sqrt initial_face_area - reduction) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_dimension_reduction_l941_94188


namespace NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l941_94101

/-- Calculates Bhanu's expenditure on house rent given his spending patterns and petrol expense -/
def house_rent_expenditure (total_income : ℝ) (petrol_percentage : ℝ) (rent_percentage : ℝ) (petrol_expense : ℝ) : ℝ :=
  let remaining_income := total_income - petrol_expense
  rent_percentage * remaining_income

/-- Proves that Bhanu's expenditure on house rent is 210 given his spending patterns and petrol expense -/
theorem bhanu_house_rent_expenditure :
  ∀ (total_income : ℝ),
    total_income > 0 →
    house_rent_expenditure total_income 0.3 0.3 300 = 210 :=
by
  sorry

#eval house_rent_expenditure 1000 0.3 0.3 300

end NUMINAMATH_CALUDE_bhanu_house_rent_expenditure_l941_94101


namespace NUMINAMATH_CALUDE_loaves_delivered_correct_evening_delivery_l941_94144

/-- Given the initial number of loaves, the number of loaves sold, and the final number of loaves,
    calculate the number of loaves delivered. -/
theorem loaves_delivered (initial : ℕ) (sold : ℕ) (final : ℕ) :
  final - (initial - sold) = final - initial + sold :=
by sorry

/-- The number of loaves delivered in the evening -/
def evening_delivery : ℕ := 2215 - (2355 - 629)

theorem correct_evening_delivery : evening_delivery = 489 :=
by sorry

end NUMINAMATH_CALUDE_loaves_delivered_correct_evening_delivery_l941_94144


namespace NUMINAMATH_CALUDE_find_A_in_terms_of_B_and_C_l941_94119

/-- Given two functions f and g, and constants A, B, and C, prove that A can be expressed in terms of B and C. -/
theorem find_A_in_terms_of_B_and_C
  (f g : ℝ → ℝ)
  (A B C : ℝ)
  (h₁ : ∀ x, f x = A * x^2 - 3 * B * C)
  (h₂ : ∀ x, g x = C * x^2)
  (h₃ : B ≠ 0)
  (h₄ : C ≠ 0)
  (h₅ : f (g 2) = A - 3 * C) :
  A = (3 * C * (B - 1)) / (16 * C^2 - 1) := by
sorry


end NUMINAMATH_CALUDE_find_A_in_terms_of_B_and_C_l941_94119


namespace NUMINAMATH_CALUDE_mona_monday_miles_l941_94116

/-- Represents Mona's biking schedule for a week --/
structure BikingWeek where
  total_miles : ℝ
  monday_miles : ℝ
  wednesday_miles : ℝ
  saturday_miles : ℝ
  steep_trail_speed : ℝ
  flat_road_speed : ℝ
  saturday_speed_reduction : ℝ

/-- Theorem stating that Mona biked 6 miles on Monday --/
theorem mona_monday_miles (week : BikingWeek) 
  (h1 : week.total_miles = 30)
  (h2 : week.wednesday_miles = 12)
  (h3 : week.saturday_miles = 2 * week.monday_miles)
  (h4 : week.steep_trail_speed = 6)
  (h5 : week.flat_road_speed = 15)
  (h6 : week.saturday_speed_reduction = 0.2)
  (h7 : week.total_miles = week.monday_miles + week.wednesday_miles + week.saturday_miles) :
  week.monday_miles = 6 := by
  sorry

#check mona_monday_miles

end NUMINAMATH_CALUDE_mona_monday_miles_l941_94116


namespace NUMINAMATH_CALUDE_product_of_real_and_imaginary_parts_l941_94120

theorem product_of_real_and_imaginary_parts : ∃ (z : ℂ), z = (2 + Complex.I) * Complex.I ∧ (z.re * z.im = -2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_and_imaginary_parts_l941_94120


namespace NUMINAMATH_CALUDE_height_comparison_l941_94175

theorem height_comparison (a b : ℝ) (h : a = 0.8 * b) : b = 1.25 * a := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l941_94175


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l941_94168

theorem sum_of_x_and_y_is_two (x y : ℝ) 
  (eq1 : x^3 + y^3 = 98) 
  (eq2 : x^2*y + x*y^2 = -30) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l941_94168


namespace NUMINAMATH_CALUDE_sum_specific_sequence_l941_94158

theorem sum_specific_sequence : 
  (1000 + 20 + 1000 + 30 + 1000 + 40 + 1000 + 10) = 4100 := by
  sorry

end NUMINAMATH_CALUDE_sum_specific_sequence_l941_94158


namespace NUMINAMATH_CALUDE_correct_average_calculation_l941_94122

theorem correct_average_calculation (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 16 →
  incorrect_num = 25 →
  correct_num = 35 →
  (n : ℚ) * initial_avg + (correct_num - incorrect_num) = n * 17 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l941_94122


namespace NUMINAMATH_CALUDE_parabola_distance_sum_lower_bound_l941_94163

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point N
def N : ℝ × ℝ := (2, 2)

-- Statement of the theorem
theorem parabola_distance_sum_lower_bound :
  ∀ (M : ℝ × ℝ), parabola M.1 M.2 →
  dist M focus + dist M N ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_distance_sum_lower_bound_l941_94163


namespace NUMINAMATH_CALUDE_restaurant_tax_calculation_l941_94118

-- Define the tax calculation function
def calculate_tax (turnover : ℕ) : ℕ :=
  if turnover ≤ 1000 then
    300
  else
    300 + (turnover - 1000) * 4 / 100

-- Theorem statement
theorem restaurant_tax_calculation :
  calculate_tax 35000 = 1660 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_tax_calculation_l941_94118


namespace NUMINAMATH_CALUDE_fraction_denominator_proof_l941_94162

theorem fraction_denominator_proof (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (9 * y) / 20 + (3 * y) / x = 0.75 * y) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_proof_l941_94162


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l941_94157

/-- Calculates the discount percentage given item costs and final amount spent -/
theorem discount_percentage_calculation 
  (hand_mitts_cost apron_cost utensils_cost final_amount : ℚ)
  (nieces : ℕ)
  (h1 : hand_mitts_cost = 14)
  (h2 : apron_cost = 16)
  (h3 : utensils_cost = 10)
  (h4 : nieces = 3)
  (h5 : final_amount = 135) :
  let knife_cost := 2 * utensils_cost
  let single_set_cost := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
  let total_cost := nieces * single_set_cost
  let discount_amount := total_cost - final_amount
  let discount_percentage := (discount_amount / total_cost) * 100
  discount_percentage = 25 := by
sorry


end NUMINAMATH_CALUDE_discount_percentage_calculation_l941_94157


namespace NUMINAMATH_CALUDE_mork_tax_rate_l941_94126

/-- Proves that Mork's tax rate is 10% given the specified conditions --/
theorem mork_tax_rate (mork_income : ℝ) (mork_tax_rate : ℝ) 
  (h1 : mork_income > 0)
  (h2 : mork_tax_rate > 0)
  (h3 : mork_tax_rate < 1)
  (h4 : (mork_tax_rate * mork_income + 3 * 0.2 * mork_income) / (4 * mork_income) = 0.175) :
  mork_tax_rate = 0.1 := by
sorry


end NUMINAMATH_CALUDE_mork_tax_rate_l941_94126


namespace NUMINAMATH_CALUDE_selection_ways_l941_94107

def club_size : ℕ := 20
def co_presidents : ℕ := 2
def treasurers : ℕ := 1

theorem selection_ways : 
  (club_size.choose co_presidents * (club_size - co_presidents).choose treasurers) = 3420 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_l941_94107


namespace NUMINAMATH_CALUDE_vegetable_ghee_mixture_weight_l941_94103

/-- Calculates the weight of a mixture of two brands of vegetable ghee -/
theorem vegetable_ghee_mixture_weight
  (weight_a : ℝ)
  (weight_b : ℝ)
  (ratio_a : ℝ)
  (ratio_b : ℝ)
  (total_volume : ℝ)
  (h_weight_a : weight_a = 800)
  (h_weight_b : weight_b = 850)
  (h_ratio_a : ratio_a = 3)
  (h_ratio_b : ratio_b = 2)
  (h_total_volume : total_volume = 3) :
  let volume_a := (ratio_a / (ratio_a + ratio_b)) * total_volume
  let volume_b := (ratio_b / (ratio_a + ratio_b)) * total_volume
  let total_weight := (weight_a * volume_a + weight_b * volume_b) / 1000
  total_weight = 2.46 := by
  sorry


end NUMINAMATH_CALUDE_vegetable_ghee_mixture_weight_l941_94103


namespace NUMINAMATH_CALUDE_total_students_l941_94129

theorem total_students (S : ℕ) (T : ℕ) : 
  T = 6 * S - 78 →
  T - S = 2222 →
  T = 2682 := by
sorry

end NUMINAMATH_CALUDE_total_students_l941_94129


namespace NUMINAMATH_CALUDE_roots_sum_product_l941_94182

theorem roots_sum_product (a b : ℂ) : 
  (a ≠ b) → 
  (a^3 + 3*a^2 + a + 1 = 0) → 
  (b^3 + 3*b^2 + b + 1 = 0) → 
  (a^2 * b + a * b^2 + 3*a*b = 1) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_product_l941_94182


namespace NUMINAMATH_CALUDE_smallest_d_value_l941_94164

def no_triangle (a b c : ℝ) : Prop :=
  a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

theorem smallest_d_value (c d : ℝ) 
  (h1 : 2 < c ∧ c < d)
  (h2 : no_triangle 2 c d)
  (h3 : no_triangle (1/d) (1/c) 2) :
  d = 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_d_value_l941_94164


namespace NUMINAMATH_CALUDE_medicine_dose_per_kg_l941_94106

theorem medicine_dose_per_kg (child_weight : ℝ) (dose_parts : ℕ) (dose_per_part : ℝ) :
  child_weight = 30 →
  dose_parts = 3 →
  dose_per_part = 50 →
  (dose_parts * dose_per_part) / child_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_medicine_dose_per_kg_l941_94106


namespace NUMINAMATH_CALUDE_defeat_dragon_l941_94105

/-- Represents the three heroes --/
inductive Hero
| Ilya
| Dobrynya
| Alyosha

/-- Calculates the number of heads removed by a hero's strike --/
def headsRemoved (hero : Hero) (h : ℕ) : ℕ :=
  match hero with
  | Hero.Ilya => (h / 2) + 1
  | Hero.Dobrynya => (h / 3) + 2
  | Hero.Alyosha => (h / 4) + 3

/-- Represents a sequence of strikes by the heroes --/
def Strike := List Hero

/-- Applies a sequence of strikes to the initial number of heads --/
def applyStrikes (initialHeads : ℕ) (strikes : Strike) : ℕ :=
  strikes.foldl (fun remaining hero => remaining - (headsRemoved hero remaining)) initialHeads

/-- Theorem: For any initial number of heads, there exists a sequence of strikes that reduces it to zero --/
theorem defeat_dragon (initialHeads : ℕ) : ∃ (strikes : Strike), applyStrikes initialHeads strikes = 0 :=
sorry


end NUMINAMATH_CALUDE_defeat_dragon_l941_94105


namespace NUMINAMATH_CALUDE_odd_function_property_y_value_at_3_l941_94125

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x

theorem odd_function_property (a b c : ℝ) :
  ∀ x, f a b c (-x) = -(f a b c x) :=
sorry

theorem y_value_at_3 (a b c : ℝ) :
  f a b c (-3) - 5 = 7 → f a b c 3 - 5 = -17 :=
sorry

end NUMINAMATH_CALUDE_odd_function_property_y_value_at_3_l941_94125


namespace NUMINAMATH_CALUDE_winnie_kept_balloons_l941_94102

/-- The number of balloons Winnie keeps for herself after distributing
    as many as possible equally among her friends -/
def balloons_kept (total_balloons : ℕ) (num_friends : ℕ) : ℕ :=
  total_balloons % num_friends

theorem winnie_kept_balloons :
  balloons_kept 200 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_winnie_kept_balloons_l941_94102


namespace NUMINAMATH_CALUDE_water_distribution_l941_94145

theorem water_distribution (total_water : ℕ) (eight_oz_glasses : ℕ) (four_oz_glasses : ℕ) 
  (h1 : total_water = 122)
  (h2 : eight_oz_glasses = 4)
  (h3 : four_oz_glasses = 15) : 
  (total_water - (8 * eight_oz_glasses + 4 * four_oz_glasses)) / 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_water_distribution_l941_94145


namespace NUMINAMATH_CALUDE_solve_for_y_l941_94127

theorem solve_for_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l941_94127


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l941_94191

def U : Set ℝ := Set.univ

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x : ℝ | x ≥ 2}

theorem intersection_complement_theorem : A ∩ (U \ B) = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l941_94191


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_is_56_l941_94161

/-- The number of people the Ferris wheel can seat -/
def ferris_wheel_capacity (total_waiting : ℕ) (not_riding : ℕ) : ℕ :=
  total_waiting - not_riding

/-- Theorem: The Ferris wheel capacity is 56 people given the problem conditions -/
theorem ferris_wheel_capacity_is_56 :
  ferris_wheel_capacity 92 36 = 56 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_is_56_l941_94161


namespace NUMINAMATH_CALUDE_max_discount_rate_l941_94180

/-- Represents the maximum discount rate problem -/
theorem max_discount_rate 
  (cost_price : ℝ) 
  (original_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 4) 
  (h2 : original_price = 5) 
  (h3 : min_profit_margin = 0.1) : 
  ∃ (max_discount : ℝ), 
    max_discount = 12 ∧ 
    ∀ (discount : ℝ), 
      discount ≤ max_discount → 
      original_price * (1 - discount / 100) - cost_price ≥ min_profit_margin * cost_price :=
sorry

end NUMINAMATH_CALUDE_max_discount_rate_l941_94180


namespace NUMINAMATH_CALUDE_doubling_base_and_exponent_l941_94135

theorem doubling_base_and_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (2*a)^(2*b) = a^b * y^b → y = 4*a :=
by sorry

end NUMINAMATH_CALUDE_doubling_base_and_exponent_l941_94135


namespace NUMINAMATH_CALUDE_digits_of_3_power_20_times_5_power_15_l941_94178

theorem digits_of_3_power_20_times_5_power_15 : ∃ n : ℕ, 
  (10 ^ (n - 1) ≤ 3^20 * 5^15) ∧ (3^20 * 5^15 < 10^n) ∧ (n = 16) := by sorry

end NUMINAMATH_CALUDE_digits_of_3_power_20_times_5_power_15_l941_94178


namespace NUMINAMATH_CALUDE_second_valid_number_is_068_l941_94151

/-- Represents a random number table as a list of natural numbers. -/
def RandomNumberTable : List ℕ := [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76]

/-- Represents the total number of units. -/
def TotalUnits : ℕ := 200

/-- Represents the starting column in the random number table. -/
def StartColumn : ℕ := 5

/-- Checks if a number is valid (i.e., between 1 and TotalUnits). -/
def isValidNumber (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ TotalUnits

/-- Finds the nth valid number in the random number table. -/
def nthValidNumber (n : ℕ) : ℕ := sorry

/-- The main theorem stating that the second valid number is 068. -/
theorem second_valid_number_is_068 : nthValidNumber 2 = 68 := by sorry

end NUMINAMATH_CALUDE_second_valid_number_is_068_l941_94151


namespace NUMINAMATH_CALUDE_problem_proof_l941_94146

theorem problem_proof : (-2)^0 - 3 * Real.tan (30 * π / 180) - |Real.sqrt 3 - 2| = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l941_94146


namespace NUMINAMATH_CALUDE_total_fish_count_l941_94167

/-- Given 261 fishbowls with 23 fish each, prove that the total number of fish is 6003. -/
theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1 : num_fishbowls = 261) 
  (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l941_94167


namespace NUMINAMATH_CALUDE_prime_product_square_l941_94114

theorem prime_product_square (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p ≠ q → p ≠ r → q ≠ r →
  (p * q * r) % (p + q + r) = 0 →
  ∃ n : ℕ, (p - 1) * (q - 1) * (r - 1) + 1 = n ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_product_square_l941_94114


namespace NUMINAMATH_CALUDE_gcd_7488_12467_l941_94117

theorem gcd_7488_12467 : Nat.gcd 7488 12467 = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7488_12467_l941_94117


namespace NUMINAMATH_CALUDE_range_of_a_l941_94187

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |2*x - a|

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ (1/4)*a^2 + 1) → a ∈ Set.Icc (-2) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l941_94187


namespace NUMINAMATH_CALUDE_tangent_circles_distance_l941_94133

/-- The distance between the centers of two tangent circles with radii 1 and 7 is either 6 or 8 -/
theorem tangent_circles_distance (r₁ r₂ d : ℝ) : 
  r₁ = 1 → r₂ = 7 → (d = r₁ + r₂ ∨ d = |r₂ - r₁|) → d = 6 ∨ d = 8 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_distance_l941_94133


namespace NUMINAMATH_CALUDE_stock_price_decrease_l941_94100

theorem stock_price_decrease (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_2006 := initial_price * 1.3
  let price_after_2007 := price_after_2006 * 1.2
  let decrease_percentage := (price_after_2007 - initial_price) / price_after_2007 * 100
  ∃ ε > 0, abs (decrease_percentage - 35.9) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l941_94100


namespace NUMINAMATH_CALUDE_tan_family_total_cost_l941_94143

/-- Represents the composition of the group visiting the amusement park -/
structure GroupComposition where
  children : Nat
  adults : Nat
  seniors : Nat

/-- Represents the discount rules for the amusement park -/
structure DiscountRules where
  seniorDiscount : Rat
  childDiscount : Rat
  groupDiscountThreshold : Nat
  groupDiscount : Rat

/-- Calculates the total cost for a group visiting the amusement park -/
def calculateTotalCost (composition : GroupComposition) (rules : DiscountRules) (adultPrice : Rat) : Rat :=
  sorry

/-- Theorem stating that the total cost for the Tan family's tickets is $45 -/
theorem tan_family_total_cost :
  let composition : GroupComposition := { children := 2, adults := 2, seniors := 2 }
  let rules : DiscountRules := { seniorDiscount := 3/10, childDiscount := 1/5, groupDiscountThreshold := 5, groupDiscount := 1/10 }
  let adultPrice : Rat := 10
  calculateTotalCost composition rules adultPrice = 45 := by
  sorry

end NUMINAMATH_CALUDE_tan_family_total_cost_l941_94143
