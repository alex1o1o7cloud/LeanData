import Mathlib

namespace emily_annual_holidays_l789_78923

theorem emily_annual_holidays 
    (holidays_per_month : ℕ) 
    (months_in_year : ℕ) 
    (h1: holidays_per_month = 2)
    (h2: months_in_year = 12)
    : holidays_per_month * months_in_year = 24 := 
by
  sorry

end emily_annual_holidays_l789_78923


namespace total_marbles_l789_78934

variable (b : ℝ)
variable (r : ℝ) (g : ℝ)
variable (h₁ : r = 1.3 * b)
variable (h₂ : g = 1.5 * b)

theorem total_marbles (b : ℝ) (r : ℝ) (g : ℝ) (h₁ : r = 1.3 * b) (h₂ : g = 1.5 * b) : r + b + g = 3.8 * b :=
by
  sorry

end total_marbles_l789_78934


namespace find_missing_dimension_l789_78968

-- Definitions based on conditions
def is_dimension_greatest_area (x : ℝ) : Prop :=
  max (2 * x) (max (3 * x) 6) = 15

-- The final statement to prove
theorem find_missing_dimension (x : ℝ) (h1 : is_dimension_greatest_area x) : x = 5 :=
sorry

end find_missing_dimension_l789_78968


namespace average_of_remaining_two_numbers_l789_78950

theorem average_of_remaining_two_numbers :
  ∀ (a b c d e f : ℝ),
    (a + b + c + d + e + f) / 6 = 3.95 →
    (a + b) / 2 = 3.6 →
    (c + d) / 2 = 3.85 →
    ((e + f) / 2 = 4.4) :=
by
  intros a b c d e f h1 h2 h3
  have h4 : a + b + c + d + e + f = 23.7 := sorry
  have h5 : a + b = 7.2 := sorry
  have h6 : c + d = 7.7 := sorry
  have h7 : e + f = 8.8 := sorry
  exact sorry

end average_of_remaining_two_numbers_l789_78950


namespace average_percentage_taller_l789_78992

theorem average_percentage_taller 
  (h1 b1 h2 b2 h3 b3 : ℝ)
  (h1_eq : h1 = 228) (b1_eq : b1 = 200)
  (h2_eq : h2 = 120) (b2_eq : b2 = 100)
  (h3_eq : h3 = 147) (b3_eq : b3 = 140) :
  ((h1 - b1) / b1 * 100 + (h2 - b2) / b2 * 100 + (h3 - b3) / b3 * 100) / 3 = 13 := by
  rw [h1_eq, b1_eq, h2_eq, b2_eq, h3_eq, b3_eq]
  sorry

end average_percentage_taller_l789_78992


namespace vector_parallel_y_value_l789_78935

theorem vector_parallel_y_value (y : ℝ) 
  (a : ℝ × ℝ := (3, 2)) 
  (b : ℝ × ℝ := (6, y)) 
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) : 
  y = 4 :=
by sorry

end vector_parallel_y_value_l789_78935


namespace flyers_left_l789_78983

theorem flyers_left (total_flyers : ℕ) (jack_flyers : ℕ) (rose_flyers : ℕ) (h1 : total_flyers = 1236) (h2 : jack_flyers = 120) (h3 : rose_flyers = 320) : (total_flyers - (jack_flyers + rose_flyers) = 796) := 
by
  sorry

end flyers_left_l789_78983


namespace center_coordinates_l789_78955

noncomputable def center_of_circle (x y : ℝ) : Prop := 
  x^2 + y^2 + 2*x - 4*y = 0

theorem center_coordinates : center_of_circle (-1) 2 :=
by sorry

end center_coordinates_l789_78955


namespace physics_marks_l789_78901

theorem physics_marks
  (P C M : ℕ)
  (h1 : P + C + M = 240)
  (h2 : P + M = 180)
  (h3 : P + C = 140) :
  P = 80 :=
by
  sorry

end physics_marks_l789_78901


namespace evaluate_expression_l789_78981

theorem evaluate_expression : 202 - 101 + 9 = 110 :=
by
  sorry

end evaluate_expression_l789_78981


namespace embankment_height_bounds_l789_78945

theorem embankment_height_bounds
  (a : ℝ) (b : ℝ) (h : ℝ)
  (a_eq : a = 5)
  (b_lower_bound : 2 ≤ b)
  (vol_lower_bound : 400 ≤ (25 * (a^2 - b^2)))
  (vol_upper_bound : (25 * (a^2 - b^2)) ≤ 500) :
  1 ≤ h ∧ h ≤ (5 - Real.sqrt 5) / 2 :=
by
  sorry

end embankment_height_bounds_l789_78945


namespace sum_of_squares_of_roots_l789_78944

theorem sum_of_squares_of_roots :
  (∃ x1 x2 : ℝ, 5 * x1^2 - 3 * x1 - 11 = 0 ∧ 5 * x2^2 - 3 * x2 - 11 = 0 ∧ x1 ≠ x2) →
  (x1 + x2 = 3 / 5 ∧ x1 * x2 = -11 / 5) →
  (x1^2 + x2^2 = 119 / 25) :=
by intro h1 h2; sorry

end sum_of_squares_of_roots_l789_78944


namespace find_equidistant_point_l789_78927

theorem find_equidistant_point :
  ∃ (x z : ℝ),
    ((x - 1)^2 + 4^2 + z^2 = (x - 2)^2 + 2^2 + (z - 3)^2) ∧
    ((x - 1)^2 + 4^2 + z^2 = (x - 3)^2 + 9 + (z + 2)^2) ∧
    (x + 2 * z = 5) ∧
    (x = 15 / 8) ∧
    (z = 5 / 8) :=
by
  sorry

end find_equidistant_point_l789_78927


namespace april_rainfall_correct_l789_78902

-- Define the constants for the rainfalls in March and the difference in April
def march_rainfall : ℝ := 0.81
def rain_difference : ℝ := 0.35

-- Define the expected April rainfall based on the conditions
def april_rainfall : ℝ := march_rainfall - rain_difference

-- Theorem to prove that the April rainfall is 0.46 inches
theorem april_rainfall_correct : april_rainfall = 0.46 :=
by
  -- Placeholder for the proof
  sorry

end april_rainfall_correct_l789_78902


namespace second_tap_empties_cistern_l789_78980

theorem second_tap_empties_cistern (t_fill: ℝ) (x: ℝ) (t_net: ℝ) : 
  (1 / 6) - (1 / x) = (1 / 12) → x = 12 := 
by
  sorry

end second_tap_empties_cistern_l789_78980


namespace rogers_spending_l789_78984

theorem rogers_spending (B m p : ℝ) (H1 : m = 0.25 * (B - p)) (H2 : p = 0.10 * (B - m)) : 
  m + p = (4 / 13) * B :=
sorry

end rogers_spending_l789_78984


namespace count_four_digit_numbers_l789_78940

open Int

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9

def valid_combination (a b c d : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  a ≠ 0 ∧
  a + b + c + d = 10 ∧
  (a + c) - (b + d) % 11 = 0

theorem count_four_digit_numbers :
  ∃ n, n > 0 ∧ ∀ a b c d : ℕ, valid_combination a b c d → n = sorry :=
sorry

end count_four_digit_numbers_l789_78940


namespace equal_distribution_of_drawings_l789_78918

theorem equal_distribution_of_drawings (total_drawings : ℕ) (neighbors : ℕ) (drawings_per_neighbor : ℕ)
  (h1 : total_drawings = 54)
  (h2 : neighbors = 6)
  (h3 : total_drawings = neighbors * drawings_per_neighbor) :
  drawings_per_neighbor = 9 :=
by
  rw [h1, h2] at h3
  linarith

end equal_distribution_of_drawings_l789_78918


namespace probability_of_at_least_one_accurate_forecast_l789_78967

theorem probability_of_at_least_one_accurate_forecast (PA PB : ℝ) (hA : PA = 0.8) (hB : PB = 0.75) :
  1 - ((1 - PA) * (1 - PB)) = 0.95 :=
by
  rw [hA, hB]
  sorry

end probability_of_at_least_one_accurate_forecast_l789_78967


namespace part1_part2_l789_78914

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -1)
noncomputable def vec_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (x / 2), Real.cos (x / 2) ^ 2)
noncomputable def f (x : ℝ) : ℝ := (vec_m x).1 * (vec_n x).1 + (vec_m x).2 * (vec_n x).2 + 1

-- Part 1
theorem part1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) (hf : f x = 11 / 10) : 
  x = (Real.pi / 6) + Real.arcsin (3 / 5) :=
sorry

-- Part 2
theorem part2 {A B C a b c : ℝ} 
  (hABC : A + B + C = Real.pi) 
  (habc : 2 * b * Real.cos A ≤ 2 * c - Real.sqrt 3 * a) : 
  (0 < B ∧ B ≤ Real.pi / 6) → 
  ∃ y, (0 < y ∧ y ≤ 1 / 2 ∧ f B = y) :=
sorry

end part1_part2_l789_78914


namespace solution_quadrant_I_l789_78961

theorem solution_quadrant_I (c x y : ℝ) :
  (x - y = 2 ∧ c * x + y = 3 ∧ x > 0 ∧ y > 0) ↔ (-1 < c ∧ c < 3/2) := by
  sorry

end solution_quadrant_I_l789_78961


namespace Chloe_initial_picked_carrots_l789_78936

variable (x : ℕ)

theorem Chloe_initial_picked_carrots :
  (x - 45 + 42 = 45) → (x = 48) :=
by
  intro h
  sorry

end Chloe_initial_picked_carrots_l789_78936


namespace find_number_l789_78957

-- Define the conditions.
def condition (x : ℚ) : Prop := x - (1 / 3) * x = 16 / 3

-- Define the theorem from the translated (question, conditions, correct answer) tuple
theorem find_number : ∃ x : ℚ, condition x ∧ x = 8 :=
by
  sorry

end find_number_l789_78957


namespace probability_X_interval_l789_78917

noncomputable def fx (x c : ℝ) : ℝ :=
  if -c ≤ x ∧ x ≤ c then (1 / c) * (1 - (|x| / c))
  else 0

theorem probability_X_interval (c : ℝ) (hc : 0 < c) :
  (∫ x in (c / 2)..c, fx x c) = 1 / 8 :=
sorry

end probability_X_interval_l789_78917


namespace range_of_a_l789_78954

-- Define the negation of the original proposition as a function
def negated_prop (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 4 * x + a > 0

-- State the theorem to be proven
theorem range_of_a (a : ℝ) (h : ¬∃ x : ℝ, a * x^2 + 4 * x + a ≤ 0) : a > 2 :=
  by
  -- Using the assumption to conclude the negated proposition holds
  let h_neg : negated_prop a := sorry
  
  -- Prove the range of a based on h_neg
  sorry

end range_of_a_l789_78954


namespace infinite_set_P_l789_78924

-- Define the condition as given in the problem
def has_property_P (P : Set ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → (∀ p : ℕ, p.Prime → p ∣ k^3 + 6 → p ∈ P)

-- State the proof problem
theorem infinite_set_P (P : Set ℕ) (h : has_property_P P) : ∃ p : ℕ, p ∉ P → false :=
by
  -- The statement asserts that the set P described by has_property_P is infinite.
  sorry

end infinite_set_P_l789_78924


namespace no_natural_n_for_perfect_square_l789_78964

theorem no_natural_n_for_perfect_square :
  ¬ ∃ n : ℕ, ∃ k : ℕ, 2007 + 4^n = k^2 :=
by {
  sorry  -- Proof omitted
}

end no_natural_n_for_perfect_square_l789_78964


namespace shaded_area_l789_78912

-- Definitions and conditions from the problem
def Square1Side := 4 -- in inches
def Square2Side := 12 -- in inches
def Triangle_DGF_similar_to_Triangle_AHF : Prop := (4 / 12) = (3 / 16)

theorem shaded_area
  (h1 : Square1Side = 4)
  (h2 : Square2Side = 12)
  (h3 : Triangle_DGF_similar_to_Triangle_AHF) :
  ∃ shaded_area : ℕ, shaded_area = 10 :=
by
  -- Calculation steps here
  sorry

end shaded_area_l789_78912


namespace sum_positive_implies_at_least_one_positive_l789_78990

variables {a b : ℝ}

theorem sum_positive_implies_at_least_one_positive (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l789_78990


namespace water_tank_capacity_l789_78970

theorem water_tank_capacity
  (tank_capacity : ℝ)
  (h : 0.30 * tank_capacity = 0.90 * tank_capacity - 54) :
  tank_capacity = 90 :=
by
  -- proof goes here
  sorry

end water_tank_capacity_l789_78970


namespace missing_digit_first_digit_l789_78985

-- Definitions derived from conditions
def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_divisible_by_six (n : ℕ) : Prop := n % 6 = 0
def multiply_by_two (d : ℕ) : ℕ := 2 * d

-- Main statement to prove
theorem missing_digit_first_digit (d : ℕ) (n : ℕ) 
  (h1 : multiply_by_two d = n) 
  (h2 : is_three_digit_number n) 
  (h3 : is_divisible_by_six n)
  (h4 : d = 2)
  : n / 100 = 2 :=
sorry

end missing_digit_first_digit_l789_78985


namespace richard_twice_as_old_as_scott_in_8_years_l789_78915

theorem richard_twice_as_old_as_scott_in_8_years :
  (richard_age - david_age = 6) ∧ (david_age - scott_age = 8) ∧ (david_age = 14) →
  (richard_age + 8 = 2 * (scott_age + 8)) :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end richard_twice_as_old_as_scott_in_8_years_l789_78915


namespace T_description_l789_78905

-- Definitions of conditions
def T (x y : ℝ) : Prop :=
  (x + 3 = 4 ∧ y ≤ 9) ∨
  (y - 5 = 4 ∧ x ≤ 1) ∨
  (x + 3 = y - 5 ∧ x ≥ 1)

-- The problem statement in Lean: Prove that T describes three rays with a common point (1, 9)
theorem T_description :
  ∀ x y, T x y ↔ 
    ((x = 1 ∧ y ≤ 9) ∨
     (x ≤ 1 ∧ y = 9) ∨
     (x ≥ 1 ∧ y = x + 8)) :=
by sorry

end T_description_l789_78905


namespace find_m_intersection_points_l789_78947

theorem find_m (m : ℝ) (hp : 2^2 + 2 * m + m^2 - 3 = 4) (h_pos : m > 0) : m = 1 := 
by
  sorry

theorem intersection_points (m : ℝ) (hp : 2^2 + 2 * m + m^2 - 3 = 4) (h_pos : m > 0) 
  (hm : m = 1) : ∃ x1 x2 : ℝ, (x^2 + x - 2 = 0) ∧ x1 ≠ x2 :=
by
  sorry

end find_m_intersection_points_l789_78947


namespace solve_for_x_l789_78956

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (5 * x + 2)) (h2 : y = 2) : x = -3 / 10 :=
by
  sorry

end solve_for_x_l789_78956


namespace carl_max_value_l789_78993

-- Definitions based on problem conditions.
def value_of_six_pound_rock : ℕ := 20
def weight_of_six_pound_rock : ℕ := 6
def value_of_three_pound_rock : ℕ := 9
def weight_of_three_pound_rock : ℕ := 3
def value_of_two_pound_rock : ℕ := 4
def weight_of_two_pound_rock : ℕ := 2
def max_weight_carl_can_carry : ℕ := 24

/-- Proves that Carl can carry rocks worth maximum 80 dollars given the conditions. -/
theorem carl_max_value : ∃ (n m k : ℕ),
    n * weight_of_six_pound_rock + m * weight_of_three_pound_rock + k * weight_of_two_pound_rock ≤ max_weight_carl_can_carry ∧
    n * value_of_six_pound_rock + m * value_of_three_pound_rock + k * value_of_two_pound_rock = 80 :=
by
  sorry

end carl_max_value_l789_78993


namespace carrie_hours_per_week_l789_78991

variable (H : ℕ)

def carrie_hourly_wage : ℕ := 8
def cost_of_bike : ℕ := 400
def amount_left_over : ℕ := 720
def weeks_worked : ℕ := 4
def total_earnings : ℕ := cost_of_bike + amount_left_over

theorem carrie_hours_per_week :
  (weeks_worked * H * carrie_hourly_wage = total_earnings) →
  H = 35 := by
  sorry

end carrie_hours_per_week_l789_78991


namespace most_convincing_method_l789_78911

-- Defining the survey data
def male_participants : Nat := 4258
def male_believe_doping : Nat := 2360
def female_participants : Nat := 3890
def female_believe_framed : Nat := 2386

-- Defining the question-to-answer equivalence related to the most convincing method
theorem most_convincing_method :
  "Independence Test" = "Independence Test" := 
by
  sorry

end most_convincing_method_l789_78911


namespace isosceles_triangle_base_length_l789_78973

theorem isosceles_triangle_base_length
  (perimeter : ℝ)
  (side1 side2 base : ℝ)
  (h_perimeter : perimeter = 18)
  (h_side1 : side1 = 4)
  (h_isosceles : side1 = side2 ∨ side1 = base ∨ side2 = base)
  (h_triangle : side1 + side2 + base = 18) :
  base = 7 := 
sorry

end isosceles_triangle_base_length_l789_78973


namespace exists_pentagon_from_midpoints_l789_78931

noncomputable def pentagon_from_midpoints (A1 B1 C1 D1 E1 : ℝ × ℝ) : Prop :=
  ∃ (A B C D E : ℝ × ℝ), 
    (A1 = (A + B) / 2) ∧ 
    (B1 = (B + C) / 2) ∧ 
    (C1 = (C + D) / 2) ∧ 
    (D1 = (D + E) / 2) ∧ 
    (E1 = (E + A) / 2)

-- statement of the theorem
theorem exists_pentagon_from_midpoints (A1 B1 C1 D1 E1 : ℝ × ℝ) :
  pentagon_from_midpoints A1 B1 C1 D1 E1 :=
sorry

end exists_pentagon_from_midpoints_l789_78931


namespace single_digit_pairs_l789_78994

theorem single_digit_pairs:
  ∃ x y: ℕ, x ≠ 1 ∧ x ≠ 9 ∧ y ≠ 1 ∧ y ≠ 9 ∧ x < 10 ∧ y < 10 ∧ 
  (x * y < 100 ∧ ((x * y) % 10 + (x * y) / 10 == x ∨ (x * y) % 10 + (x * y) / 10 == y))
  → (x, y) ∈ [(3, 4), (3, 7), (6, 4), (6, 7)] :=
by
  sorry

end single_digit_pairs_l789_78994


namespace liangliang_distance_to_school_l789_78986

theorem liangliang_distance_to_school :
  (∀ (t : ℕ), (40 * t = 50 * (t - 5)) → (40 * 25 = 1000)) :=
sorry

end liangliang_distance_to_school_l789_78986


namespace candy_in_one_bowl_l789_78987

theorem candy_in_one_bowl (total_candies : ℕ) (eaten_candies : ℕ) (bowls : ℕ) (taken_per_bowl : ℕ) 
  (h1 : total_candies = 100) (h2 : eaten_candies = 8) (h3 : bowls = 4) (h4 : taken_per_bowl = 3) :
  (total_candies - eaten_candies) / bowls - taken_per_bowl = 20 :=
by
  sorry

end candy_in_one_bowl_l789_78987


namespace final_temperature_l789_78933

theorem final_temperature (initial_temp cost_per_tree spent amount temperature_drop : ℝ) 
  (h1 : initial_temp = 80) 
  (h2 : cost_per_tree = 6)
  (h3 : spent = 108) 
  (h4 : temperature_drop = 0.1) 
  (trees_planted : ℝ) 
  (h5 : trees_planted = spent / cost_per_tree) 
  (temp_reduction : ℝ) 
  (h6 : temp_reduction = trees_planted * temperature_drop) 
  (final_temp : ℝ) 
  (h7 : final_temp = initial_temp - temp_reduction) : 
  final_temp = 78.2 := 
by
  sorry

end final_temperature_l789_78933


namespace cube_of_prism_volume_l789_78908

theorem cube_of_prism_volume (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y) * (y * z) * (z * x) = (x * y * z)^3 :=
by
  sorry

end cube_of_prism_volume_l789_78908


namespace parabola_translation_l789_78903

-- Define the initial equation of the parabola
def initial_parabola (x : ℝ) : ℝ := x^2 - 2

-- Define the transformation: translate one unit to the right
def translate_right (x : ℝ) : ℝ := initial_parabola (x - 1)

-- Define the transformation: move up three units
def move_up (y : ℝ) : ℝ := y + 3

-- Define the resulting equation after the transformations
def resulting_parabola (x : ℝ) : ℝ := move_up (translate_right x)

-- Define the target equation
def target_parabola (x : ℝ) : ℝ := (x - 1)^2 + 1

-- Formalize the proof problem
theorem parabola_translation :
  ∀ x : ℝ, resulting_parabola x = target_parabola x :=
by
  -- Proof steps go here
  sorry

end parabola_translation_l789_78903


namespace factorization_correct_l789_78928

theorem factorization_correct :
  ∀ (x y : ℝ), 
    (¬ ( (y - 1) * (y + 1) = y^2 - 1 ) ) ∧
    (¬ ( x^2 * y + x * y^2 - 1 = x * y * (x + y) - 1 ) ) ∧
    (¬ ( (x - 2) * (x - 3) = (3 - x) * (2 - x) ) ) ∧
    ( x^2 - 4 * x + 4 = (x - 2)^2 ) :=
by
  intros x y
  repeat { constructor }
  all_goals { sorry }

end factorization_correct_l789_78928


namespace sufficient_not_necessary_for_ellipse_l789_78921

-- Define the conditions
def positive_denominator_m (m : ℝ) : Prop := m > 0
def positive_denominator_2m_minus_1 (m : ℝ) : Prop := 2 * m - 1 > 0
def denominators_not_equal (m : ℝ) : Prop := m ≠ 1

-- Define the question
def is_ellipse_condition (m : ℝ) : Prop := m > 1

-- The main theorem
theorem sufficient_not_necessary_for_ellipse (m : ℝ) :
  positive_denominator_m m ∧ positive_denominator_2m_minus_1 m ∧ denominators_not_equal m → is_ellipse_condition m :=
by
  -- Proof omitted
  sorry

end sufficient_not_necessary_for_ellipse_l789_78921


namespace ratio_of_integers_l789_78910

theorem ratio_of_integers (a b : ℤ) (h : 1996 * a + b / 96 = a + b) : a / b = 1 / 2016 ∨ b / a = 2016 :=
by
  sorry

end ratio_of_integers_l789_78910


namespace percentage_increase_is_20_l789_78959

-- Defining the original cost and new cost
def original_cost := 200
def new_total_cost := 480

-- Doubling the capacity means doubling the original cost
def doubled_old_cost := 2 * original_cost

-- The increase in cost
def increase_cost := new_total_cost - doubled_old_cost

-- The percentage increase in cost
def percentage_increase := (increase_cost / doubled_old_cost) * 100

-- The theorem we need to prove
theorem percentage_increase_is_20 : percentage_increase = 20 :=
  by
  sorry

end percentage_increase_is_20_l789_78959


namespace compute_expression_l789_78926

theorem compute_expression : 2 + 8 * 3 - 4 + 6 * 5 / 2 - 3 ^ 2 = 28 := by
  sorry

end compute_expression_l789_78926


namespace new_person_weight_l789_78958

-- Define the initial conditions
def initial_average_weight (w : ℕ) := 6 * w -- The total weight of 6 persons

-- Define the scenario where the average weight increases by 2 kg
def total_weight_increase := 6 * 2 -- The total increase in weight due to an increase of 2 kg in average weight

def person_replaced := 75 -- The weight of the person being replaced

-- Define the expected condition on the weight of the new person
theorem new_person_weight (w_new : ℕ) :
  initial_average_weight person_replaced + total_weight_increase = initial_average_weight (w_new / 6) →
  w_new = 87 :=
sorry

end new_person_weight_l789_78958


namespace sum_of_squares_of_coefficients_l789_78969

theorem sum_of_squares_of_coefficients :
  ∃ (a b c d e f : ℤ), (∀ x : ℤ, 8 * x ^ 3 + 64 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) ∧ 
  (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 + e ^ 2 + f ^ 2 = 356) := 
by
  sorry

end sum_of_squares_of_coefficients_l789_78969


namespace cuboid_height_l789_78925

/-- Given a cuboid with surface area 2400 cm², length 15 cm, and breadth 10 cm,
    prove that the height is 42 cm. -/
theorem cuboid_height (SA l w : ℝ) (h : ℝ) : 
  SA = 2400 → l = 15 → w = 10 → 2 * (l * w + l * h + w * h) = SA → h = 42 :=
by
  intros hSA hl hw hformula
  sorry

end cuboid_height_l789_78925


namespace line_circle_intersect_l789_78977

theorem line_circle_intersect (a : ℝ) (h : a > 1) :
  ∃ x y : ℝ, (x - a)^2 + (y - 1)^2 = 2 ∧ x - a * y - 2 = 0 :=
sorry

end line_circle_intersect_l789_78977


namespace remainder_of_x_pow_77_eq_6_l789_78982

theorem remainder_of_x_pow_77_eq_6 (x : ℤ) (h : x^77 % 7 = 6) : x^77 % 7 = 6 :=
by
  sorry

end remainder_of_x_pow_77_eq_6_l789_78982


namespace union_inter_complement_l789_78962

open Set

variable (U : Set ℝ := univ)
variable (A : Set ℝ := {x | abs (x - 2) > 3})
variable (B : Set ℝ := {x | x * (-2 - x) > 0})

theorem union_inter_complement 
  (C_U_A : Set ℝ := compl A)
  (A_def : A = {x | abs (x - 2) > 3})
  (B_def : B = {x | x * (-2 - x) > 0})
  (C_U_A_def : C_U_A = compl A) :
  (A ∪ B = {x : ℝ | x < 0} ∪ {x : ℝ | x > 5}) ∧ 
  ((C_U_A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 0}) :=
by
  sorry

end union_inter_complement_l789_78962


namespace singers_in_choir_l789_78963

variable (X : ℕ)

/-- In the first verse, only half of the total singers sang -/ 
def first_verse_not_singing (X : ℕ) : ℕ := X / 2

/-- In the second verse, a third of the remaining singers joined in -/
def second_verse_joining (X : ℕ) : ℕ := (X / 2) / 3

/-- In the final third verse, 10 people joined so that the whole choir sang together -/
def remaining_singers_after_second_verse (X : ℕ) : ℕ := first_verse_not_singing X - second_verse_joining X

def final_verse_joining_condition (X : ℕ) : Prop := remaining_singers_after_second_verse X = 10

theorem singers_in_choir : ∃ (X : ℕ), final_verse_joining_condition X ∧ X = 30 :=
by
  sorry

end singers_in_choir_l789_78963


namespace composite_numbers_l789_78997

theorem composite_numbers (n : ℕ) (hn : n > 0) :
  (∃ p q, p > 1 ∧ q > 1 ∧ 2 * 2^(2^n) + 1 = p * q) ∧ 
  (∃ p q, p > 1 ∧ q > 1 ∧ 3 * 2^(2*n) + 1 = p * q) :=
sorry

end composite_numbers_l789_78997


namespace sum_of_geometric_sequence_l789_78907

theorem sum_of_geometric_sequence :
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 3280 / 6561 :=
by
  let a : ℚ := 1 / 3
  let r : ℚ := 1 / 3
  let n : ℕ := 8
  let S_n := a * (1 - r^n) / (1 - r)
  sorry

end sum_of_geometric_sequence_l789_78907


namespace min_value_expression_l789_78949

theorem min_value_expression (x y : ℝ) : 
  (∃ (x_min y_min : ℝ), 
  (x_min = 1/2 ∧ y_min = 0) ∧ 
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = 39/4) :=
by
  sorry

end min_value_expression_l789_78949


namespace garden_length_is_60_l789_78930

noncomputable def garden_length (w l : ℕ) : Prop :=
  l = 2 * w ∧ 2 * w + 2 * l = 180

theorem garden_length_is_60 (w l : ℕ) (h : garden_length w l) : l = 60 :=
by
  sorry

end garden_length_is_60_l789_78930


namespace total_persimmons_l789_78976

-- Definitions based on conditions in a)
def totalWeight (kg : ℕ) := kg = 3
def weightPerFivePersimmons (kg : ℕ) := kg = 1

-- The proof problem
theorem total_persimmons (k : ℕ) (w : ℕ) (x : ℕ) (h1 : totalWeight k) (h2 : weightPerFivePersimmons w) : x = 15 :=
by
  -- With the definitions totalWeight and weightPerFivePersimmons given in the conditions
  -- we aim to prove that the number of persimmons, x, is 15.
  sorry

end total_persimmons_l789_78976


namespace count_triples_l789_78972

open Set

theorem count_triples 
  (A B C : Set ℕ) 
  (h_union : A ∪ B ∪ C = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
  (h_inter : A ∩ B ∩ C = ∅) :
  (∃ n : ℕ, n = 60466176) :=
by
  -- Proof can be filled in here
  sorry

end count_triples_l789_78972


namespace maximum_regular_hours_is_40_l789_78913

-- Definitions based on conditions
def regular_pay_per_hour := 3
def overtime_pay_per_hour := 6
def total_payment_received := 168
def overtime_hours := 8
def overtime_earnings := overtime_hours * overtime_pay_per_hour
def regular_earnings := total_payment_received - overtime_earnings
def maximum_regular_hours := regular_earnings / regular_pay_per_hour

-- Lean theorem statement corresponding to the proof problem
theorem maximum_regular_hours_is_40 : maximum_regular_hours = 40 := by
  sorry

end maximum_regular_hours_is_40_l789_78913


namespace number_of_students_l789_78948

variable (F S J R T : ℕ)

axiom freshman_more_than_junior : F = (5 * J) / 4
axiom sophomore_fewer_than_freshman : S = 9 * F / 10
axiom total_students : T = F + S + J + R
axiom seniors_total : R = T / 5
axiom given_sophomores : S = 144

theorem number_of_students (T : ℕ) : T = 540 :=
by 
  sorry

end number_of_students_l789_78948


namespace alice_weight_l789_78909

theorem alice_weight (a c : ℝ) (h1 : a + c = 200) (h2 : a - c = a / 3) : a = 120 :=
by
  sorry

end alice_weight_l789_78909


namespace problem_f_prime_at_zero_l789_78965

noncomputable def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5) + 6

theorem problem_f_prime_at_zero : deriv f 0 = 120 :=
by
  -- Proof omitted
  sorry

end problem_f_prime_at_zero_l789_78965


namespace sum_of_integers_with_largest_proper_divisor_55_l789_78916

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def largest_proper_divisor (n d : ℕ) : Prop :=
  (d ∣ n) ∧ (d < n) ∧ ∀ e, (e ∣ n ∧ e < n ∧ e > d) → False

theorem sum_of_integers_with_largest_proper_divisor_55 : 
  (∀ n : ℕ, largest_proper_divisor n 55 → n = 110 ∨ n = 165 ∨ n = 275) →
  110 + 165 + 275 = 550 :=
by
  sorry

end sum_of_integers_with_largest_proper_divisor_55_l789_78916


namespace charles_earnings_l789_78937

def housesit_rate : ℝ := 15
def dog_walk_rate : ℝ := 22
def hours_housesit : ℝ := 10
def num_dogs : ℝ := 3

theorem charles_earnings :
  housesit_rate * hours_housesit + dog_walk_rate * num_dogs = 216 :=
by
  sorry

end charles_earnings_l789_78937


namespace other_root_eq_six_l789_78975

theorem other_root_eq_six (a : ℝ) (x1 : ℝ) (x2 : ℝ) 
  (h : x1 = -2) 
  (eqn : ∀ x, x^2 - a * x - 3 * a = 0 → (x = x1 ∨ x = x2)) :
  x2 = 6 :=
by
  sorry

end other_root_eq_six_l789_78975


namespace quadratic_roots_condition_l789_78971

theorem quadratic_roots_condition (a : ℝ) :
  (∃ α : ℝ, 5 * α = -(a - 4) ∧ 4 * α^2 = a - 5) ↔ (a = 7 ∨ a = 5) :=
by
  sorry

end quadratic_roots_condition_l789_78971


namespace probability_of_girls_under_18_l789_78943

theorem probability_of_girls_under_18
  (total_members : ℕ)
  (girls : ℕ)
  (boys : ℕ)
  (underaged_girls : ℕ)
  (two_members_chosen : ℕ)
  (total_ways_to_choose_two : ℕ)
  (ways_to_choose_two_girls : ℕ)
  (ways_to_choose_at_least_one_underaged : ℕ)
  (prob : ℚ)
  : 
  total_members = 15 →
  girls = 8 →
  boys = 7 →
  underaged_girls = 3 →
  two_members_chosen = 2 →
  total_ways_to_choose_two = (Nat.choose total_members two_members_chosen) →
  ways_to_choose_two_girls = (Nat.choose girls two_members_chosen) →
  ways_to_choose_at_least_one_underaged = 
    (Nat.choose underaged_girls 1 * Nat.choose (girls - underaged_girls) 1 + Nat.choose underaged_girls 2) →
  prob = (ways_to_choose_at_least_one_underaged : ℚ) / (total_ways_to_choose_two : ℚ) →
  prob = 6 / 35 :=
by
  intros
  sorry

end probability_of_girls_under_18_l789_78943


namespace kenneth_left_with_amount_l789_78999

theorem kenneth_left_with_amount (total_earnings : ℝ) (percentage_spent : ℝ) (amount_left : ℝ) 
    (h_total_earnings : total_earnings = 450) (h_percentage_spent : percentage_spent = 0.10) 
    (h_spent_amount : total_earnings * percentage_spent = 45) : 
    amount_left = total_earnings - total_earnings * percentage_spent :=
by sorry

end kenneth_left_with_amount_l789_78999


namespace probability_of_odd_score_l789_78942

noncomputable def dartboard : Type := sorry

variables (r_inner r_outer : ℝ)
variables (inner_values outer_values : Fin 3 → ℕ)
variables (P_odd : ℚ)

-- Conditions
def dartboard_conditions (r_inner r_outer : ℝ) (inner_values outer_values : Fin 3 → ℕ) : Prop :=
  r_inner = 4 ∧ r_outer = 8 ∧
  inner_values 0 = 3 ∧ inner_values 1 = 1 ∧ inner_values 2 = 1 ∧
  outer_values 0 = 3 ∧ outer_values 1 = 2 ∧ outer_values 2 = 2

-- Correct Answer
def correct_odds_probability (P_odd : ℚ) : Prop :=
  P_odd = 4 / 9

-- Main Statement
theorem probability_of_odd_score (r_inner r_outer : ℝ) (inner_values outer_values : Fin 3 → ℕ) (P_odd : ℚ) :
  dartboard_conditions r_inner r_outer inner_values outer_values →
  correct_odds_probability P_odd :=
sorry

end probability_of_odd_score_l789_78942


namespace slope_of_line_determined_by_any_two_solutions_l789_78998

theorem slope_of_line_determined_by_any_two_solutions 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 4 / x₁ + 5 / y₁ = 0) 
  (h₂ : 4 / x₂ + 5 / y₂ = 0) 
  (h_distinct : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -5 / 4 := 
sorry

end slope_of_line_determined_by_any_two_solutions_l789_78998


namespace sum_of_possible_values_l789_78929

theorem sum_of_possible_values (x : ℝ) (h : (x + 3) * (x - 5) = 20) : x = -2 ∨ x = 7 :=
sorry

end sum_of_possible_values_l789_78929


namespace parallelogram_area_l789_78900

theorem parallelogram_area (b h : ℝ) (hb : b = 20) (hh : h = 4) : b * h = 80 := by
  sorry

end parallelogram_area_l789_78900


namespace distinct_prime_factors_90_l789_78978

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l789_78978


namespace find_mn_l789_78951

variable (OA OB OC : EuclideanSpace ℝ (Fin 3))
variable (AOC BOC : ℝ)

axiom length_OA : ‖OA‖ = 2
axiom length_OB : ‖OB‖ = 2
axiom length_OC : ‖OC‖ = 2 * Real.sqrt 3
axiom tan_angle_AOC : Real.tan AOC = 3 * Real.sqrt 3
axiom angle_BOC : BOC = Real.pi / 3

theorem find_mn : ∃ m n : ℝ, OC = m • OA + n • OB ∧ m = 5 / 3 ∧ n = 2 * Real.sqrt 3 := by
  sorry

end find_mn_l789_78951


namespace find_number_l789_78966

theorem find_number (x : ℝ) (h : x - (3/5 : ℝ) * x = 60) : x = 150 :=
sorry

end find_number_l789_78966


namespace charlyn_visible_area_l789_78988

noncomputable def visible_area (side_length vision_distance : ℝ) : ℝ :=
  let outer_rectangles_area := 4 * (side_length * vision_distance)
  let outer_squares_area := 4 * (vision_distance * vision_distance)
  let inner_square_area := 
    let inner_side_length := side_length - 2 * vision_distance
    inner_side_length * inner_side_length
  let total_walk_area := side_length * side_length
  total_walk_area - inner_square_area + outer_rectangles_area + outer_squares_area

theorem charlyn_visible_area :
  visible_area 10 2 = 160 := by
  sorry

end charlyn_visible_area_l789_78988


namespace maximum_value_condition_l789_78995

open Real

theorem maximum_value_condition {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h1 : x + y = 16) (h2 : x = 2 * y) :
  (1 / x + 1 / y) = 9 / 32 :=
by
  sorry

end maximum_value_condition_l789_78995


namespace rational_square_root_l789_78996

theorem rational_square_root {x y : ℚ} 
  (h : (x^2 + y^2 - 2) * (x + y)^2 + (xy + 1)^2 = 0) : 
  ∃ r : ℚ, r * r = 1 + x * y := 
sorry

end rational_square_root_l789_78996


namespace sequence_eventually_periodic_l789_78952

open Nat

noncomputable def sum_prime_factors_plus_one (K : ℕ) : ℕ := 
  (K.factors.sum) + 1

theorem sequence_eventually_periodic (K : ℕ) (hK : K ≥ 9) :
  ∃ m n : ℕ, m ≠ n ∧ sum_prime_factors_plus_one^[m] K = sum_prime_factors_plus_one^[n] K := 
sorry

end sequence_eventually_periodic_l789_78952


namespace base_subtraction_problem_l789_78979

theorem base_subtraction_problem (b : ℕ) (C_b : ℕ) (hC : C_b = 12) : 
  b = 15 :=
by
  sorry

end base_subtraction_problem_l789_78979


namespace maximum_profit_l789_78989

def cost_price_per_unit : ℕ := 40
def initial_selling_price_per_unit : ℕ := 50
def units_sold_per_month : ℕ := 210
def price_increase_effect (x : ℕ) : ℕ := units_sold_per_month - 10 * x
def profit_function (x : ℕ) : ℕ := (price_increase_effect x) * (initial_selling_price_per_unit + x - cost_price_per_unit)

theorem maximum_profit :
  profit_function 5 = 2400 ∧ profit_function 6 = 2400 :=
by
  sorry

end maximum_profit_l789_78989


namespace capacity_of_each_bucket_in_second_case_final_proof_l789_78938

def tank_volume (buckets: ℕ) (bucket_capacity: ℝ) : ℝ := buckets * bucket_capacity

theorem capacity_of_each_bucket_in_second_case
  (total_volume: ℝ)
  (first_case_buckets : ℕ)
  (first_case_capacity : ℝ)
  (second_case_buckets : ℕ) :
  first_case_buckets * first_case_capacity = total_volume → 
  (total_volume / second_case_buckets) = 9 :=
by
  intros h
  sorry

-- Given the conditions:
noncomputable def total_volume := tank_volume 28 13.5

theorem final_proof :
  (tank_volume 28 13.5 = total_volume) → 
  (total_volume / 42 = 9) :=
by
  intro h
  exact capacity_of_each_bucket_in_second_case total_volume 28 13.5 42 h

end capacity_of_each_bucket_in_second_case_final_proof_l789_78938


namespace mr_william_land_percentage_l789_78904

/--
Given:
1. Farm tax is levied on 90% of the cultivated land.
2. The tax department collected a total of $3840 through the farm tax from the village.
3. Mr. William paid $480 as farm tax.

Prove: The percentage of total land of Mr. William over the total taxable land of the village is 12.5%.
-/
theorem mr_william_land_percentage (T W : ℝ) 
  (h1 : 0.9 * W = 480) 
  (h2 : 0.9 * T = 3840) : 
  (W / T) * 100 = 12.5 :=
by
  sorry

end mr_william_land_percentage_l789_78904


namespace no_rain_five_days_l789_78919

-- Define the problem conditions and the required result.
def prob_rain := (2 / 3)
def prob_no_rain := (1 - prob_rain)
def prob_no_rain_five_days := prob_no_rain^5

theorem no_rain_five_days : 
  prob_no_rain_five_days = (1 / 243) :=
by
  sorry

end no_rain_five_days_l789_78919


namespace father_current_age_l789_78946

variable (M F : ℕ)

/-- The man's current age is (2 / 5) of the age of his father. -/
axiom man_age : M = (2 / 5) * F

/-- After 12 years, the man's age will be (1 / 2) of his father's age. -/
axiom age_relation_in_12_years : (M + 12) = (1 / 2) * (F + 12)

/-- Prove that the father's current age, F, is 60. -/
theorem father_current_age : F = 60 :=
by
  sorry

end father_current_age_l789_78946


namespace distance_between_parallel_lines_l789_78974

-- Definitions
def line_eq1 (x y : ℝ) : Prop := 3 * x - 4 * y - 12 = 0
def line_eq2 (x y : ℝ) : Prop := 6 * x - 8 * y + 11 = 0

-- Statement of the problem
theorem distance_between_parallel_lines :
  (∀ x y : ℝ, line_eq1 x y ↔ line_eq2 x y) →
  (∃ d : ℝ, d = 7 / 2) :=
by
  sorry

end distance_between_parallel_lines_l789_78974


namespace triangle_statements_l789_78960

-- Define the fundamental properties of the triangle
noncomputable def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a = 45 ∧ a = 2 ∧ b = 2 * Real.sqrt 2 ∧ 
  (a - b = c * Real.cos B - c * Real.cos A)

-- Statement A
def statement_A (A B C a b c : ℝ) (h : triangle A B C a b c) : Prop :=
  ∃ B, Real.sin B = 1

-- Statement B
def statement_B (A B C : ℝ) (v_AC v_AB : ℝ) : Prop :=
  v_AC * v_AB > 0 → Real.cos A > 0

-- Statement C
def statement_C (A B : ℝ) (a b : ℝ) : Prop :=
  Real.sin A > Real.sin B → a > b

-- Statement D
def statement_D (A B C a b c : ℝ) (h : triangle A B C a b c) : Prop :=
  (a - b = c * Real.cos B - c * Real.cos A) →
  (a = b ∨ c^2 = a^2 + b^2)

-- Final proof statement
theorem triangle_statements (A B C a b c : ℝ) (v_AC v_AB : ℝ) 
  (h_triangle : triangle A B C a b c) :
  (statement_A A B C a b c h_triangle) ∧
  ¬(statement_B A B C v_AC v_AB) ∧
  (statement_C A B a b) ∧
  (statement_D A B C a b c h_triangle) :=
by sorry

end triangle_statements_l789_78960


namespace range_of_a_l789_78941

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x > 2 ↔ x < 2 / (a - 1)) → a < 1 :=
by
  sorry

end range_of_a_l789_78941


namespace total_money_is_correct_l789_78920

-- Define the values of different types of coins and the amount of each.
def gold_value : ℕ := 75
def silver_value : ℕ := 40
def bronze_value : ℕ := 20
def titanium_value : ℕ := 10

def gold_count : ℕ := 6
def silver_count : ℕ := 8
def bronze_count : ℕ := 10
def titanium_count : ℕ := 4
def cash : ℕ := 45

-- Define the total amount of money.
def total_money : ℕ :=
  (gold_count * gold_value) +
  (silver_count * silver_value) +
  (bronze_count * bronze_value) +
  (titanium_count * titanium_value) + cash

-- The proof statement
theorem total_money_is_correct : total_money = 1055 := by
  sorry

end total_money_is_correct_l789_78920


namespace initial_incorrect_average_l789_78953

theorem initial_incorrect_average (S_correct S_wrong : ℝ) :
  (S_correct = S_wrong - 26 + 36) →
  (S_correct / 10 = 19) →
  (S_wrong / 10 = 18) :=
by
  sorry

end initial_incorrect_average_l789_78953


namespace num_of_triangles_with_perimeter_10_l789_78922

theorem num_of_triangles_with_perimeter_10 :
  ∃ (triangles : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → 
      a + b + c = 10 ∧ 
      a + b > c ∧ 
      a + c > b ∧ 
      b + c > a) ∧ 
    triangles.card = 4 := sorry

end num_of_triangles_with_perimeter_10_l789_78922


namespace find_square_value_l789_78906

variable (a b : ℝ)
variable (square : ℝ)

-- Conditions: Given the equation square * 3 * a = -3 * a^2 * b
axiom condition : square * 3 * a = -3 * a^2 * b

-- Theorem: Prove that square = -a * b
theorem find_square_value (a b : ℝ) (square : ℝ) (h : square * 3 * a = -3 * a^2 * b) : 
    square = -a * b :=
by
  exact sorry

end find_square_value_l789_78906


namespace hyperbola_equation_l789_78932

-- Define the conditions
def hyperbola_eq := ∀ (x y a b : ℝ), a > 0 ∧ b > 0 → x^2 / a^2 - y^2 / b^2 = 1
def parabola_eq := ∀ (x y : ℝ), y^2 = (2 / 5) * x
def intersection_point_M := ∃ (x : ℝ), ∀ (y : ℝ), y = 1 → y^2 = (2 / 5) * x
def line_intersect_N := ∀ (F₁ M N : ℝ × ℝ), 
  (N.1 = -1 / 10) ∧ (F₁.1 ≠ M.1) ∧ (N.2 = 0)

-- State the proof problem
theorem hyperbola_equation 
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (hyp_eq : hyperbola_eq)
  (par_eq : parabola_eq)
  (int_pt_M : intersection_point_M)
  (line_int_N : line_intersect_N) :
  ∀ (x y : ℝ), x^2 / 5 - y^2 / 4 = 1 :=
by sorry

end hyperbola_equation_l789_78932


namespace solve_system_1_solve_system_2_l789_78939

theorem solve_system_1 (x y : ℤ) (h1 : y = 2 * x - 3) (h2 : 3 * x + 2 * y = 8) : x = 2 ∧ y = 1 :=
by {
  sorry
}

theorem solve_system_2 (x y : ℤ) (h1 : 2 * x + 3 * y = 7) (h2 : 3 * x - 2 * y = 4) : x = 2 ∧ y = 1 :=
by {
  sorry
}

end solve_system_1_solve_system_2_l789_78939
