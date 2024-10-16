import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_sum_l2169_216918

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a + 3 * i) / i = b - 2 * i → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2169_216918


namespace NUMINAMATH_CALUDE_expression_value_l2169_216970

theorem expression_value (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2169_216970


namespace NUMINAMATH_CALUDE_work_division_proof_l2169_216941

/-- The number of days it takes x to finish the entire work -/
def x_total_days : ℝ := 18

/-- The number of days it takes y to finish the entire work -/
def y_total_days : ℝ := 15

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining_days : ℝ := 12

/-- The number of days y worked before leaving the job -/
def y_worked_days : ℝ := 5

theorem work_division_proof :
  let total_work : ℝ := 1
  let x_rate : ℝ := total_work / x_total_days
  let y_rate : ℝ := total_work / y_total_days
  y_worked_days * y_rate + x_remaining_days * x_rate = total_work :=
by sorry

end NUMINAMATH_CALUDE_work_division_proof_l2169_216941


namespace NUMINAMATH_CALUDE_value_range_sqrt_16_minus_4_pow_x_l2169_216965

theorem value_range_sqrt_16_minus_4_pow_x :
  ∀ x : ℝ, 0 ≤ Real.sqrt (16 - 4^x) ∧ Real.sqrt (16 - 4^x) < 4 := by
  sorry

end NUMINAMATH_CALUDE_value_range_sqrt_16_minus_4_pow_x_l2169_216965


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2169_216955

theorem imaginary_part_of_z (z : ℂ) (h : (3 + 4*I)*z = 5) : 
  z.im = -4/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2169_216955


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l2169_216942

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, StrictMonoOn f (Set.Ioo 0 2) := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l2169_216942


namespace NUMINAMATH_CALUDE_game_lives_theorem_l2169_216998

/-- Given a game with initial players, players who quit, and total remaining lives,
    calculate the number of lives per remaining player. -/
def lives_per_player (initial_players : ℕ) (players_quit : ℕ) (total_lives : ℕ) : ℕ :=
  total_lives / (initial_players - players_quit)

/-- Theorem: In a game with 16 initial players, 7 players quitting, and 72 total remaining lives,
    each remaining player has 8 lives. -/
theorem game_lives_theorem :
  lives_per_player 16 7 72 = 8 := by
  sorry

end NUMINAMATH_CALUDE_game_lives_theorem_l2169_216998


namespace NUMINAMATH_CALUDE_patanjali_walk_l2169_216940

/-- Represents the walking scenario of Patanjali over three days --/
structure WalkingScenario where
  hours_day1 : ℕ
  speed_day1 : ℕ
  total_distance : ℕ

/-- Calculates the distance walked on the first day given a WalkingScenario --/
def distance_day1 (scenario : WalkingScenario) : ℕ :=
  scenario.hours_day1 * scenario.speed_day1

/-- Calculates the total distance walked over three days given a WalkingScenario --/
def total_distance (scenario : WalkingScenario) : ℕ :=
  (distance_day1 scenario) + 
  (scenario.hours_day1 - 1) * (scenario.speed_day1 + 1) + 
  scenario.hours_day1 * (scenario.speed_day1 + 1)

/-- Theorem stating that given the conditions, the distance walked on the first day is 18 miles --/
theorem patanjali_walk (scenario : WalkingScenario) 
  (h1 : scenario.speed_day1 = 3) 
  (h2 : total_distance scenario = 62) : 
  distance_day1 scenario = 18 := by
  sorry

#eval distance_day1 { hours_day1 := 6, speed_day1 := 3, total_distance := 62 }

end NUMINAMATH_CALUDE_patanjali_walk_l2169_216940


namespace NUMINAMATH_CALUDE_john_bought_three_tshirts_l2169_216957

/-- The number of t-shirts John bought -/
def num_tshirts : ℕ := 3

/-- The cost of each t-shirt in dollars -/
def tshirt_cost : ℕ := 20

/-- The amount spent on pants in dollars -/
def pants_cost : ℕ := 50

/-- The total amount spent in dollars -/
def total_spent : ℕ := 110

theorem john_bought_three_tshirts :
  num_tshirts * tshirt_cost + pants_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_john_bought_three_tshirts_l2169_216957


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2169_216935

theorem equal_roots_quadratic (p : ℝ) : 
  (∃! p, ∀ x, x^2 - (p + 1) * x + p = 0 → (∃! x, x^2 - (p + 1) * x + p = 0)) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2169_216935


namespace NUMINAMATH_CALUDE_bob_overspent_l2169_216927

theorem bob_overspent (necklace_cost book_cost total_spent limit : ℕ) : 
  necklace_cost = 34 →
  book_cost = necklace_cost + 5 →
  total_spent = necklace_cost + book_cost →
  limit = 70 →
  total_spent - limit = 3 := by
  sorry

end NUMINAMATH_CALUDE_bob_overspent_l2169_216927


namespace NUMINAMATH_CALUDE_exists_M_with_properties_l2169_216931

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The theorem stating the existence of M with the required properties -/
theorem exists_M_with_properties : 
  ∃ M : ℕ, M^2 = 36^50 * 50^36 ∧ sum_of_digits M = 36 := by sorry

end NUMINAMATH_CALUDE_exists_M_with_properties_l2169_216931


namespace NUMINAMATH_CALUDE_distribute_five_into_three_l2169_216922

/-- The number of ways to distribute n distinct items into k identical bags -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 3 identical bags -/
theorem distribute_five_into_three : distribute 5 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_five_into_three_l2169_216922


namespace NUMINAMATH_CALUDE_smallest_linear_combination_l2169_216963

theorem smallest_linear_combination : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∃ (m n p : ℤ), k = 2010 * m + 44550 * n + 100 * p) ∧
  (∀ (j : ℕ), j > 0 → (∃ (x y z : ℤ), j = 2010 * x + 44550 * y + 100 * z) → j ≥ k) ∧
  k = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_l2169_216963


namespace NUMINAMATH_CALUDE_town_population_problem_l2169_216962

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 1200) : ℚ) * (1 - 11/100) : ℚ).floor = original_population - 32 → 
  original_population = 10000 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l2169_216962


namespace NUMINAMATH_CALUDE_gcf_of_lcm_sum_and_difference_l2169_216938

theorem gcf_of_lcm_sum_and_difference : Nat.gcd (Nat.lcm 9 15 + 5) (Nat.lcm 10 21 - 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcm_sum_and_difference_l2169_216938


namespace NUMINAMATH_CALUDE_fedya_statement_possible_l2169_216903

/-- Represents a date with year, month, and day -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Represents Fedya's age on a given date -/
def age (birthdate : Date) (currentDate : Date) : ℕ := sorry

/-- Returns the date one year after the given date -/
def nextYear (d : Date) : Date := sorry

/-- Returns the date two days before the given date -/
def twoDaysAgo (d : Date) : Date := sorry

/-- Theorem stating that Fedya's statement could be true -/
theorem fedya_statement_possible : ∃ (birthdate currentDate : Date),
  age birthdate (twoDaysAgo currentDate) = 10 ∧
  age birthdate (nextYear currentDate) = 13 :=
sorry

end NUMINAMATH_CALUDE_fedya_statement_possible_l2169_216903


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2169_216953

theorem concentric_circles_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * Real.pi * r₁) = 48 / 360 * (2 * Real.pi * r₂)) →
  (r₁ / r₂ = 4 / 5 ∧ (r₁^2 / r₂^2 = 16 / 25)) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l2169_216953


namespace NUMINAMATH_CALUDE_integral_points_on_line_segment_l2169_216928

def is_on_line_segment (x y : ℤ) : Prop :=
  ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧
  x = (22 : ℤ) + t * ((16 : ℤ) - (22 : ℤ)) ∧
  y = (12 : ℤ) + t * ((17 : ℤ) - (12 : ℤ))

theorem integral_points_on_line_segment :
  ∃! p : ℤ × ℤ, 
    is_on_line_segment p.1 p.2 ∧
    10 ≤ p.1 ∧ p.1 ≤ 30 ∧
    10 ≤ p.2 ∧ p.2 ≤ 30 :=
sorry

end NUMINAMATH_CALUDE_integral_points_on_line_segment_l2169_216928


namespace NUMINAMATH_CALUDE_matching_probability_is_one_third_l2169_216958

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans a person has -/
def JellyBeans.total (jb : JellyBeans) : ℕ := jb.blue + jb.green + jb.yellow

/-- Abe's jelly beans -/
def abe : JellyBeans := { blue := 2, green := 2, yellow := 0 }

/-- Bob's jelly beans -/
def bob : JellyBeans := { blue := 3, green := 1, yellow := 2 }

/-- The probability of two people showing matching color jelly beans -/
def matchingProbability (person1 person2 : JellyBeans) : ℚ :=
  let totalProb : ℚ := 
    (person1.blue * person2.blue + person1.green * person2.green) / 
    (person1.total * person2.total)
  totalProb

theorem matching_probability_is_one_third : 
  matchingProbability abe bob = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_matching_probability_is_one_third_l2169_216958


namespace NUMINAMATH_CALUDE_specific_right_triangle_l2169_216956

/-- A right triangle with specific side lengths -/
structure RightTriangle where
  -- The length of the hypotenuse
  ab : ℝ
  -- The length of one of the other sides
  ac : ℝ
  -- The length of the remaining side
  bc : ℝ
  -- Constraint that this is a right triangle (Pythagorean theorem)
  pythagorean : ab ^ 2 = ac ^ 2 + bc ^ 2

/-- Theorem: In a right triangle with hypotenuse 5 and one side 4, the other side is 3 -/
theorem specific_right_triangle :
  ∃ (t : RightTriangle), t.ab = 5 ∧ t.ac = 4 ∧ t.bc = 3 := by
  sorry


end NUMINAMATH_CALUDE_specific_right_triangle_l2169_216956


namespace NUMINAMATH_CALUDE_special_number_not_perfect_square_l2169_216943

/-- A number composed of exactly 100 zeros, 100 ones, and 100 twos -/
def special_number : ℕ :=
  -- We don't need to define the exact number, just its properties
  sorry

/-- The sum of digits of the special number -/
def sum_of_digits : ℕ := 300

/-- Theorem: The special number is not a perfect square -/
theorem special_number_not_perfect_square :
  ∀ n : ℕ, n ^ 2 ≠ special_number := by
  sorry

end NUMINAMATH_CALUDE_special_number_not_perfect_square_l2169_216943


namespace NUMINAMATH_CALUDE_carnival_wait_time_l2169_216993

/-- Carnival Ride Wait Time Problem -/
theorem carnival_wait_time (total_time roller_coaster_wait giant_slide_wait : ℕ)
  (roller_coaster_rides tilt_a_whirl_rides giant_slide_rides : ℕ)
  (h1 : total_time = 4 * 60)
  (h2 : roller_coaster_wait = 30)
  (h3 : giant_slide_wait = 15)
  (h4 : roller_coaster_rides = 4)
  (h5 : tilt_a_whirl_rides = 1)
  (h6 : giant_slide_rides = 4) :
  ∃ tilt_a_whirl_wait : ℕ,
    total_time = roller_coaster_wait * roller_coaster_rides +
                 tilt_a_whirl_wait * tilt_a_whirl_rides +
                 giant_slide_wait * giant_slide_rides ∧
    tilt_a_whirl_wait = 60 :=
by sorry

end NUMINAMATH_CALUDE_carnival_wait_time_l2169_216993


namespace NUMINAMATH_CALUDE_fraction_equality_l2169_216937

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a/b = 8)
  (h2 : c/b = 4)
  (h3 : c/d = 2/3) :
  d/a = 3/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2169_216937


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2169_216911

theorem complex_fraction_equality : ∃ (i : ℂ), i * i = -1 ∧ (7 + i) / (3 + 4 * i) = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2169_216911


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2169_216910

/-- Triangle ABC with side lengths x+1, x, x-1 opposite to angles A, B, C respectively --/
structure Triangle (x : ℝ) where
  sideA : ℝ := x + 1
  sideB : ℝ := x
  sideC : ℝ := x - 1

/-- The angle A is twice the angle C --/
axiom angle_relation {x : ℝ} (t : Triangle x) :
  ∃ (A C : ℝ), A = 2 * C

/-- The Law of Sines holds for the triangle --/
axiom law_of_sines {x : ℝ} (t : Triangle x) :
  ∃ (A B C : ℝ), (t.sideA / Real.sin A) = (t.sideB / Real.sin B) ∧ 
                  (t.sideB / Real.sin B) = (t.sideC / Real.sin C)

/-- The Law of Cosines holds for the triangle --/
axiom law_of_cosines {x : ℝ} (t : Triangle x) :
  ∃ (C : ℝ), Real.cos C = (t.sideA^2 + t.sideB^2 - t.sideC^2) / (2 * t.sideA * t.sideB)

/-- The perimeter of the triangle is 15 --/
theorem triangle_perimeter {x : ℝ} (t : Triangle x) : t.sideA + t.sideB + t.sideC = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2169_216910


namespace NUMINAMATH_CALUDE_min_value_theorem_l2169_216901

theorem min_value_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0)
  (h_solution : ∀ x, x^2 - 4*a*x + 3*a^2 < 0 ↔ x ∈ Set.Ioo x₁ x₂) :
  ∀ y, x₁ + x₂ + a / (x₁ * x₂) ≥ y → y ≤ 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2169_216901


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2169_216961

theorem hyperbola_equation (P Q : ℝ × ℝ) : 
  P = (-3, 2 * Real.sqrt 7) → 
  Q = (-6 * Real.sqrt 2, 7) → 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), (y^2 / a^2) - (x^2 / b^2) = 1 ↔ 
      ((x, y) = P ∨ (x, y) = Q)) ∧
    a^2 = 25 ∧ b^2 = 75 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2169_216961


namespace NUMINAMATH_CALUDE_chord_inequality_l2169_216923

/-- Given a semicircle with unit radius and four consecutive chords with lengths a, b, c, d,
    prove that a^2 + b^2 + c^2 + d^2 + abc + bcd < 4 -/
theorem chord_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hchords : ∃ (A B C D E : ℝ × ℝ), 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = a^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = b^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = c^2 ∧
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = d^2 ∧
    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧ 
    D.1^2 + D.2^2 = 1 ∧ E.1^2 + E.2^2 = 1 ∧
    (A.2 ≥ 0 ∧ B.2 ≥ 0 ∧ C.2 ≥ 0 ∧ D.2 ≥ 0 ∧ E.2 ≥ 0)) :
  a^2 + b^2 + c^2 + d^2 + a*b*c + b*c*d < 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_inequality_l2169_216923


namespace NUMINAMATH_CALUDE_xy_range_l2169_216989

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2/x + 3*y + 4/y = 10) : 
  1 ≤ x*y ∧ x*y ≤ 8/3 := by
sorry

end NUMINAMATH_CALUDE_xy_range_l2169_216989


namespace NUMINAMATH_CALUDE_equation_solution_exists_l2169_216936

theorem equation_solution_exists : ∃ x : ℤ, 
  |x - ((1125 - 500 + 660 - 200) * (3/2) * (3/4) / 45)| ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l2169_216936


namespace NUMINAMATH_CALUDE_keith_cd_player_cost_l2169_216944

/-- The amount Keith spent on speakers -/
def speakers_cost : ℚ := 136.01

/-- The amount Keith spent on new tires -/
def tires_cost : ℚ := 112.46

/-- The total amount Keith spent -/
def total_cost : ℚ := 387.85

/-- The amount Keith spent on the CD player -/
def cd_player_cost : ℚ := total_cost - (speakers_cost + tires_cost)

theorem keith_cd_player_cost :
  cd_player_cost = 139.38 := by sorry

end NUMINAMATH_CALUDE_keith_cd_player_cost_l2169_216944


namespace NUMINAMATH_CALUDE_triangle_angle_sum_special_case_l2169_216980

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (angle_sum : A + B + C = π)
  (law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C))

-- State the theorem
theorem triangle_angle_sum_special_case (t : Triangle) 
  (h : (t.a + t.c - t.b) * (t.a + t.c + t.b) = 3 * t.a * t.c) : 
  t.A + t.C = 2 * π / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_special_case_l2169_216980


namespace NUMINAMATH_CALUDE_sum_in_M_l2169_216912

/-- Define the set Mα for a positive real number α -/
def M (α : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), x2 > x1 → 
    -α * (x2 - x1) < f x2 - f x1 ∧ f x2 - f x1 < α * (x2 - x1)

/-- Theorem: If f ∈ Mα1 and g ∈ Mα2, then f + g ∈ Mα1+α2 -/
theorem sum_in_M (α1 α2 : ℝ) (f g : ℝ → ℝ) 
    (hα1 : α1 > 0) (hα2 : α2 > 0) 
    (hf : M α1 f) (hg : M α2 g) : 
  M (α1 + α2) (f + g) := by
  sorry

end NUMINAMATH_CALUDE_sum_in_M_l2169_216912


namespace NUMINAMATH_CALUDE_tomato_plants_count_l2169_216917

/-- Represents the number of vegetables harvested from each surviving plant. -/
def vegetables_per_plant : ℕ := 7

/-- Represents the total number of vegetables harvested. -/
def total_vegetables : ℕ := 56

/-- Represents the number of eggplant plants. -/
def eggplant_plants : ℕ := 2

/-- Represents the initial number of pepper plants. -/
def initial_pepper_plants : ℕ := 4

/-- Represents the number of pepper plants that died. -/
def dead_pepper_plants : ℕ := 1

theorem tomato_plants_count (T : ℕ) : 
  (T / 2 + eggplant_plants + (initial_pepper_plants - dead_pepper_plants)) * vegetables_per_plant = total_vegetables → 
  T = 6 := by
sorry

end NUMINAMATH_CALUDE_tomato_plants_count_l2169_216917


namespace NUMINAMATH_CALUDE_sum_of_squares_minus_linear_l2169_216920

theorem sum_of_squares_minus_linear : ∀ x y : ℝ, 
  x ≠ y → 
  x^2 - 2000*x = y^2 - 2000*y → 
  x + y = 2000 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_minus_linear_l2169_216920


namespace NUMINAMATH_CALUDE_joker_prob_is_one_twentyseventh_l2169_216945

/-- A standard deck of cards with jokers -/
structure Deck :=
  (total_cards : ℕ)
  (jokers : ℕ)
  (h_total : total_cards = 54)
  (h_jokers : jokers = 2)

/-- The probability of drawing a joker from the top of the deck -/
def joker_probability (d : Deck) : ℚ :=
  d.jokers / d.total_cards

/-- Theorem: The probability of drawing a joker from a standard 54-card deck with 2 jokers is 1/27 -/
theorem joker_prob_is_one_twentyseventh (d : Deck) : joker_probability d = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_joker_prob_is_one_twentyseventh_l2169_216945


namespace NUMINAMATH_CALUDE_derivative_exp_cos_l2169_216959

/-- The derivative of e^x * cos(x) is e^x * (cos(x) - sin(x)) -/
theorem derivative_exp_cos (x : ℝ) : 
  deriv (λ x => Real.exp x * Real.cos x) x = Real.exp x * (Real.cos x - Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_exp_cos_l2169_216959


namespace NUMINAMATH_CALUDE_log_32_2_l2169_216974

theorem log_32_2 : Real.log 2 / Real.log 32 = 1 / 5 := by
  have h : 32 = 2^5 := by sorry
  sorry

end NUMINAMATH_CALUDE_log_32_2_l2169_216974


namespace NUMINAMATH_CALUDE_polynomial_roots_theorem_l2169_216900

theorem polynomial_roots_theorem (a b c : ℝ) : 
  (∃ (r s t : ℝ), 
    (∀ x : ℝ, x^4 - a*x^3 + b*x^2 - c*x + a = 0 ↔ x = 0 ∨ x = r ∨ x = s ∨ x = t) ∧
    (a > 0) ∧
    (∀ a' : ℝ, a' > 0 → a' ≥ a)) →
  a = 3 * Real.sqrt 3 ∧ b = 9 ∧ c = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_theorem_l2169_216900


namespace NUMINAMATH_CALUDE_initial_amount_is_21_l2169_216994

/-- Represents the money transactions between three people A, B, and C. -/
structure MoneyTransaction where
  a_initial : ℚ
  b_initial : ℚ := 5
  c_initial : ℚ := 9

/-- Calculates the final amounts after all transactions. -/
def final_amounts (mt : MoneyTransaction) : ℚ × ℚ × ℚ :=
  let a1 := mt.a_initial - (mt.b_initial + mt.c_initial)
  let b1 := 2 * mt.b_initial
  let c1 := 2 * mt.c_initial
  
  let a2 := a1 + (a1 / 2)
  let b2 := b1 - ((a1 / 2) + (c1 / 2))
  let c2 := c1 + (c1 / 2)
  
  let a3 := a2 + 3 * a2 + 3 * b2
  let b3 := b2 + 3 * b2 + 3 * c2
  let c3 := c2 - (3 * a2 + 3 * b2)
  
  (a3, b3, c3)

/-- Theorem stating that if the final amounts are (24, 16, 8), then A started with 21 cents. -/
theorem initial_amount_is_21 (mt : MoneyTransaction) : 
  final_amounts mt = (24, 16, 8) → mt.a_initial = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_is_21_l2169_216994


namespace NUMINAMATH_CALUDE_product_arrangement_count_l2169_216978

/-- The number of products to arrange -/
def n : ℕ := 5

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldr (· * ·) 1

/-- The number of arrangements with A and B together -/
def arrangementsWithABTogether : ℕ := 2 * factorial (n - 1)

/-- The number of arrangements with C and D together -/
def arrangementsWithCDTogether : ℕ := 2 * 2 * factorial (n - 2)

/-- The total number of valid arrangements -/
def validArrangements : ℕ := arrangementsWithABTogether - arrangementsWithCDTogether

theorem product_arrangement_count : validArrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_product_arrangement_count_l2169_216978


namespace NUMINAMATH_CALUDE_largest_three_digit_geometric_l2169_216925

/-- Checks if a number is a three-digit integer -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Extracts the hundreds digit of a three-digit number -/
def hundredsDigit (n : ℕ) : ℕ := n / 100

/-- Extracts the tens digit of a three-digit number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- Extracts the ones digit of a three-digit number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- Checks if the digits of a three-digit number are distinct -/
def hasDistinctDigits (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  h ≠ t ∧ t ≠ o ∧ h ≠ o

/-- Checks if the digits of a three-digit number form a geometric sequence -/
def isGeometricSequence (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  ∃ r : ℚ, r ≠ 0 ∧ t = h / r ∧ o = t / r

theorem largest_three_digit_geometric : 
  ∀ n : ℕ, isThreeDigit n → hasDistinctDigits n → isGeometricSequence n → hundredsDigit n = 8 → n ≤ 842 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_geometric_l2169_216925


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2169_216972

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) : 
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3/2 :=
sorry

theorem equality_condition (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) : 
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) = 3/2 ↔ 
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2169_216972


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2169_216934

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) ≥ 7.5 :=
sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) = 7.5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2169_216934


namespace NUMINAMATH_CALUDE_q_investment_l2169_216985

/-- Given two business partners P and Q, where:
  * P invested 75000 rupees
  * The profit is divided between P and Q in the ratio of 5:1
  * The profit ratio is equal to the investment ratio
Prove that Q invested 15000 rupees. -/
theorem q_investment (p_investment : ℕ) (profit_ratio : ℚ) (h1 : p_investment = 75000) (h2 : profit_ratio = 5/1) : 
  let q_investment := p_investment / profit_ratio.num
  q_investment = 15000 := by
sorry

end NUMINAMATH_CALUDE_q_investment_l2169_216985


namespace NUMINAMATH_CALUDE_binary_of_34_l2169_216902

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Theorem: The binary representation of 34 is 100010 -/
theorem binary_of_34 : toBinary 34 = [false, true, false, false, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_of_34_l2169_216902


namespace NUMINAMATH_CALUDE_line_equation_proof_l2169_216988

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (p : Point) (result_line : Line) : 
  given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 5 ∧
  p.x = -2 ∧ p.y = 1 ∧
  result_line.a = 2 ∧ result_line.b = -3 ∧ result_line.c = 7 →
  pointOnLine p result_line ∧ parallel given_line result_line :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2169_216988


namespace NUMINAMATH_CALUDE_fourth_term_of_specific_sequence_l2169_216996

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  first_term : ℝ
  common_ratio : ℝ

/-- The nth term of a geometric sequence -/
def nth_term (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem fourth_term_of_specific_sequence :
  ∃ (seq : GeometricSequence),
    seq.first_term = 512 ∧
    nth_term seq 6 = 32 ∧
    nth_term seq 4 = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_specific_sequence_l2169_216996


namespace NUMINAMATH_CALUDE_power_four_mod_nine_l2169_216913

theorem power_four_mod_nine : 4^215 % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_four_mod_nine_l2169_216913


namespace NUMINAMATH_CALUDE_james_socks_count_l2169_216904

/-- The number of pairs of red socks James has -/
def red_pairs : ℕ := 20

/-- The number of red socks James has -/
def red_socks : ℕ := red_pairs * 2

/-- The number of black socks James has -/
def black_socks : ℕ := red_socks / 2

/-- The number of red and black socks combined -/
def red_black_socks : ℕ := red_socks + black_socks

/-- The number of white socks James has -/
def white_socks : ℕ := red_black_socks * 2

/-- The total number of socks James has -/
def total_socks : ℕ := red_socks + black_socks + white_socks

theorem james_socks_count : total_socks = 180 := by
  sorry

end NUMINAMATH_CALUDE_james_socks_count_l2169_216904


namespace NUMINAMATH_CALUDE_average_temperature_l2169_216967

def temperatures : List ℝ := [60, 59, 56, 53, 49, 48, 46]

theorem average_temperature : 
  (List.sum temperatures) / temperatures.length = 53 :=
by sorry

end NUMINAMATH_CALUDE_average_temperature_l2169_216967


namespace NUMINAMATH_CALUDE_fishing_ratio_l2169_216971

/-- Given that Jordan caught 4 fish and after losing one-fourth of their total catch, 
    9 fish remain, prove that the ratio of Perry's catch to Jordan's catch is 2:1 -/
theorem fishing_ratio : 
  let jordan_catch : ℕ := 4
  let remaining_fish : ℕ := 9
  let total_catch : ℕ := remaining_fish * 4 / 3
  let perry_catch : ℕ := total_catch - jordan_catch
  (perry_catch : ℚ) / jordan_catch = 2 / 1 := by
sorry


end NUMINAMATH_CALUDE_fishing_ratio_l2169_216971


namespace NUMINAMATH_CALUDE_gcd_2028_2100_l2169_216986

theorem gcd_2028_2100 : Nat.gcd 2028 2100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2028_2100_l2169_216986


namespace NUMINAMATH_CALUDE_dart_board_partitions_l2169_216939

def partition_count (n : ℕ) (k : ℕ) : ℕ := 
  sorry

theorem dart_board_partitions : partition_count 5 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dart_board_partitions_l2169_216939


namespace NUMINAMATH_CALUDE_three_thousand_six_hundred_factorization_l2169_216921

theorem three_thousand_six_hundred_factorization (a b c d : ℕ+) 
  (h1 : 3600 = 2^(a.val) * 3^(b.val) * 4^(c.val) * 5^(d.val))
  (h2 : a.val + b.val + c.val + d.val = 7) : c.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_thousand_six_hundred_factorization_l2169_216921


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l2169_216914

theorem greatest_integer_fraction_inequality : 
  ∀ x : ℤ, (8 : ℚ) / 11 > (x : ℚ) / 15 ↔ x ≤ 10 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l2169_216914


namespace NUMINAMATH_CALUDE_divisibility_implies_sum_product_l2169_216916

theorem divisibility_implies_sum_product (p q r s : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*r*x + s = 
    (x^4 + 4*x^3 + 6*x^2 + 4*x + 1) * k) →
  (p + q + r) * s = -2.2 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_sum_product_l2169_216916


namespace NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l2169_216975

theorem geometric_sequence_eighth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^5 = 32) 
  (h3 : r > 0) : 
  a * r^7 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l2169_216975


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2169_216991

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2169_216991


namespace NUMINAMATH_CALUDE_inequality_proof_l2169_216960

theorem inequality_proof (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (2*a - b)^2 / (a - b)^2 + (2*b - c)^2 / (b - c)^2 + (2*c - a)^2 / (c - a)^2 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2169_216960


namespace NUMINAMATH_CALUDE_eccentric_annulus_area_l2169_216992

/-- Eccentric annulus area theorem -/
theorem eccentric_annulus_area 
  (R r d : ℝ) 
  (h1 : R > r) 
  (h2 : d < R) : 
  Real.pi * (R - r - d^2 / (R - r)) = 
    Real.pi * R^2 - Real.pi * r^2 :=
sorry

end NUMINAMATH_CALUDE_eccentric_annulus_area_l2169_216992


namespace NUMINAMATH_CALUDE_l_structure_surface_area_l2169_216926

/-- Represents the L-shaped structure -/
structure LStructure where
  bottom_length : ℕ
  bottom_width : ℕ
  stack_height : ℕ

/-- Calculates the surface area of the L-shaped structure -/
def surface_area (l : LStructure) : ℕ :=
  let bottom_area := l.bottom_length * l.bottom_width
  let bottom_perimeter := 2 * l.bottom_length + l.bottom_width
  let stack_side_area := 2 * l.stack_height
  let stack_top_area := 1
  bottom_area + bottom_perimeter + stack_side_area + stack_top_area

/-- The specific L-shaped structure in the problem -/
def problem_structure : LStructure :=
  { bottom_length := 3
  , bottom_width := 3
  , stack_height := 6 }

theorem l_structure_surface_area :
  surface_area problem_structure = 29 := by
  sorry

end NUMINAMATH_CALUDE_l_structure_surface_area_l2169_216926


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2169_216990

/-- The standard equation of a hyperbola with given asymptotes and passing through a specific point -/
theorem hyperbola_standard_equation (x y : ℝ) :
  (∀ (t : ℝ), y = (2/3) * t ∨ y = -(2/3) * t) →  -- Asymptotes condition
  (x = Real.sqrt 6 ∧ y = 2) →                    -- Point condition
  (3 * y^2 / 4) - (x^2 / 3) = 1 :=               -- Standard equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2169_216990


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2169_216995

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def has_four_consecutive_terms (b : ℕ → ℝ) (S : Set ℝ) : Prop :=
  ∃ k : ℕ, (b k ∈ S) ∧ (b (k + 1) ∈ S) ∧ (b (k + 2) ∈ S) ∧ (b (k + 3) ∈ S)

theorem geometric_sequence_ratio (a b : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  (∀ n : ℕ, b n = a n + 1) →
  has_four_consecutive_terms b {-53, -23, 19, 37, 82} →
  abs q > 1 →
  q = -3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2169_216995


namespace NUMINAMATH_CALUDE_range_of_a_l2169_216999

open Real

theorem range_of_a (m : ℝ) (hm : m > 0) :
  (∃ x : ℝ, x + a * (2*x + 2*m - 4*Real.exp 1*x) * (log (x + m) - log x) = 0) →
  a ∈ Set.Iic 0 ∪ Set.Ici (1 / (2 * Real.exp 1)) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2169_216999


namespace NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l2169_216933

-- Define the propositions p and q
def p (a b : ℝ) : Prop := a + b > 0
def q (a b : ℝ) : Prop := a * b > 0

-- Theorem stating that p is neither sufficient nor necessary for q
theorem p_neither_sufficient_nor_necessary_for_q :
  (∃ a b : ℝ, p a b ∧ ¬q a b) ∧ (∃ a b : ℝ, q a b ∧ ¬p a b) :=
sorry

end NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l2169_216933


namespace NUMINAMATH_CALUDE_function_properties_l2169_216987

theorem function_properties :
  (∃ x : ℝ, (10 : ℝ) ^ x = x) ∧
  (∃ x : ℝ, (10 : ℝ) ^ x = x ^ 2) ∧
  (¬ ∀ x : ℝ, (10 : ℝ) ^ x > x) ∧
  (¬ ∀ x : ℝ, x > 0 → (10 : ℝ) ^ x > x ^ 2) ∧
  (¬ ∃ x y : ℝ, x ≠ y ∧ (10 : ℝ) ^ x = -x ∧ (10 : ℝ) ^ y = -y) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2169_216987


namespace NUMINAMATH_CALUDE_correct_production_l2169_216964

/-- Represents the production of each shift in a car manufacturing plant. -/
structure ShiftProduction where
  day : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given shift production satisfies the problem conditions. -/
def satisfiesConditions (p : ShiftProduction) : Prop :=
  p.day = 4 * p.second ∧
  p.third = (3 * p.second) / 2 ∧
  p.day + p.second + p.third = 8000

/-- Theorem stating that the given production numbers satisfy the problem conditions. -/
theorem correct_production : satisfiesConditions ⟨4923, 1231, 1846⟩ := by
  sorry

#check correct_production

end NUMINAMATH_CALUDE_correct_production_l2169_216964


namespace NUMINAMATH_CALUDE_puzzle_solution_l2169_216930

theorem puzzle_solution :
  ∀ (S I A L T : ℕ),
  S ≠ 0 →
  S ≠ I ∧ S ≠ A ∧ S ≠ L ∧ S ≠ T ∧
  I ≠ A ∧ I ≠ L ∧ I ≠ T ∧
  A ≠ L ∧ A ≠ T ∧
  L ≠ T →
  10 * S + I < 100 →
  1000 * S + 100 * A + 10 * L + T < 10000 →
  (10 * S + I) * (10 * S + I) = 1000 * S + 100 * A + 10 * L + T →
  S = 9 ∧ I = 8 ∧ A = 6 ∧ L = 0 ∧ T = 4 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2169_216930


namespace NUMINAMATH_CALUDE_parabola_max_vertex_sum_l2169_216981

theorem parabola_max_vertex_sum (a S : ℤ) (h : S ≠ 0) :
  let parabola (x y : ℚ) := ∃ b c : ℚ, y = a * x^2 + b * x + c
  let passes_through (x y : ℚ) := parabola x y
  let vertex_sum := 
    let x₀ : ℚ := (3 * S : ℚ) / 2
    let y₀ : ℚ := -((9 * S^2 : ℚ) / 4) * a
    x₀ + y₀
  (passes_through 0 0) ∧ 
  (passes_through (3 * S) 0) ∧ 
  (passes_through (3 * S - 2) 35) →
  (∀ M : ℚ, vertex_sum ≤ M → M ≤ 1485/4)
  :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_vertex_sum_l2169_216981


namespace NUMINAMATH_CALUDE_factorial_6_l2169_216973

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_6 : factorial 6 = 720 := by sorry

end NUMINAMATH_CALUDE_factorial_6_l2169_216973


namespace NUMINAMATH_CALUDE_complex_magnitude_l2169_216919

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2169_216919


namespace NUMINAMATH_CALUDE_real_part_of_complex_expression_l2169_216946

theorem real_part_of_complex_expression :
  Complex.re ((1 - 2 * Complex.I)^2 + Complex.I) = -3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_expression_l2169_216946


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2169_216979

theorem complex_equation_solution (z : ℂ) : (3 - 4*I)*z = 25 → z = 3 + 4*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2169_216979


namespace NUMINAMATH_CALUDE_reappearance_line_l2169_216969

def letter_cycle_length : ℕ := 5
def digit_cycle_length : ℕ := 4

theorem reappearance_line : Nat.lcm letter_cycle_length digit_cycle_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_reappearance_line_l2169_216969


namespace NUMINAMATH_CALUDE_inequalities_representation_l2169_216947

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a shape in 2D space -/
class Shape where
  contains : Point2D → Prop

/-- Diamond shape defined by |x| + |y| ≤ r -/
def Diamond (r : ℝ) : Shape where
  contains p := abs p.x + abs p.y ≤ r

/-- Circle shape defined by x² + y² ≤ r² -/
def Circle (r : ℝ) : Shape where
  contains p := p.x^2 + p.y^2 ≤ r^2

/-- Hexagon shape defined by 3Max(|x|, |y|) ≤ r -/
def Hexagon (r : ℝ) : Shape where
  contains p := 3 * max (abs p.x) (abs p.y) ≤ r

/-- Theorem stating that the inequalities represent the described geometric shapes -/
theorem inequalities_representation (r : ℝ) (p : Point2D) :
  (Diamond r).contains p → (Circle r).contains p → (Hexagon r).contains p :=
by sorry

end NUMINAMATH_CALUDE_inequalities_representation_l2169_216947


namespace NUMINAMATH_CALUDE_y_work_time_l2169_216948

/-- Given workers x, y, and z, and their work rates, prove that y alone takes 24 hours to complete the work. -/
theorem y_work_time (x y z : ℝ) (hx : x = 1 / 8) (hyz : y + z = 1 / 6) (hxz : x + z = 1 / 4) :
  1 / y = 24 := by
  sorry

end NUMINAMATH_CALUDE_y_work_time_l2169_216948


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2169_216906

theorem inequality_system_solution :
  let S : Set ℝ := {x | (x - 1) / 2 + 2 > x ∧ 2 * (x - 2) ≤ 3 * x - 5}
  S = {x | 1 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2169_216906


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2169_216976

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 10/21) 
  (h2 : x - y = 1/63) : 
  x^2 - y^2 = 10/1323 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2169_216976


namespace NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l2169_216982

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The daily water consumption of the first sibling -/
def sibling1_daily_consumption : ℕ := 8

/-- The daily water consumption of the second sibling -/
def sibling2_daily_consumption : ℕ := 7

/-- The daily water consumption of the third sibling -/
def sibling3_daily_consumption : ℕ := 9

/-- The total water consumption of all siblings in one week -/
def total_weekly_consumption : ℕ :=
  (sibling1_daily_consumption + sibling2_daily_consumption + sibling3_daily_consumption) * days_in_week

theorem siblings_weekly_water_consumption :
  total_weekly_consumption = 168 := by
  sorry

end NUMINAMATH_CALUDE_siblings_weekly_water_consumption_l2169_216982


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l2169_216984

theorem quadratic_root_implies_a_value (a : ℝ) :
  (2 : ℝ)^2 - a * 2 + 6 = 0 → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l2169_216984


namespace NUMINAMATH_CALUDE_davids_trip_money_l2169_216951

theorem davids_trip_money (initial_amount spent_amount remaining_amount : ℕ) :
  remaining_amount = 500 →
  spent_amount = remaining_amount + 800 →
  initial_amount = spent_amount + remaining_amount →
  initial_amount = 1800 :=
by sorry

end NUMINAMATH_CALUDE_davids_trip_money_l2169_216951


namespace NUMINAMATH_CALUDE_second_street_sales_l2169_216932

/-- Represents the sales data for a door-to-door salesman selling security systems. -/
structure SalesData where
  commission_per_sale : ℕ
  total_commission : ℕ
  streets : Fin 4 → ℕ
  second_street_sales : ℕ

/-- The conditions of the sales problem. -/
def sales_conditions (data : SalesData) : Prop :=
  data.commission_per_sale = 25 ∧
  data.total_commission = 175 ∧
  data.streets 0 = data.second_street_sales / 2 ∧
  data.streets 1 = data.second_street_sales ∧
  data.streets 2 = 0 ∧
  data.streets 3 = 1

/-- Theorem stating that under the given conditions, the number of security systems sold on the second street is 4. -/
theorem second_street_sales (data : SalesData) :
  sales_conditions data → data.second_street_sales = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_street_sales_l2169_216932


namespace NUMINAMATH_CALUDE_depak_money_problem_l2169_216983

theorem depak_money_problem :
  ∀ x : ℕ, 
    (x + 1) % 6 = 0 ∧ 
    x % 6 ≠ 0 ∧
    ∀ y : ℕ, y > x → (y + 1) % 6 ≠ 0 ∨ y % 6 = 0
    → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_depak_money_problem_l2169_216983


namespace NUMINAMATH_CALUDE_rectangle_length_l2169_216950

theorem rectangle_length (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth →
  area = length * breadth →
  area = 6075 →
  length = 135 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l2169_216950


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2169_216954

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ, 7^x + 1 = 3^y + 5^z →
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2169_216954


namespace NUMINAMATH_CALUDE_independence_test_not_always_correct_l2169_216907

-- Define the independence test
def independence_test (sample : Type) : Prop := True

-- Define the principle of small probability
def principle_of_small_probability : Prop := True

-- Define the concept of different samples
def different_samples (s1 s2 : Type) : Prop := s1 ≠ s2

-- Define the concept of different conclusions
def different_conclusions (c1 c2 : Prop) : Prop := c1 ≠ c2

-- Define other methods for determining categorical variable relationships
def other_methods_exist : Prop := True

-- Theorem statement
theorem independence_test_not_always_correct :
  (∀ (s : Type), independence_test s → principle_of_small_probability) →
  (∀ (s1 s2 : Type), different_samples s1 s2 → 
    ∃ (c1 c2 : Prop), different_conclusions c1 c2) →
  other_methods_exist →
  ¬(∀ (s : Type), independence_test s → 
    ∀ (conclusion : Prop), conclusion) :=
by sorry

end NUMINAMATH_CALUDE_independence_test_not_always_correct_l2169_216907


namespace NUMINAMATH_CALUDE_sum_of_integers_l2169_216909

theorem sum_of_integers (a b : ℕ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) 
  (h4 : a^2 - b^2 = 2018 - 2*a) : a + b = 672 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2169_216909


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l2169_216908

/-- Represents the savings from Coupon A (20% off total price) -/
def savingsA (price : ℝ) : ℝ := 0.2 * price

/-- Represents the savings from Coupon B (flat $40 discount) -/
def savingsB : ℝ := 40

/-- Represents the savings from Coupon C (30% off amount exceeding $120) -/
def savingsC (price : ℝ) : ℝ := 0.3 * (price - 120)

/-- Theorem stating the difference between max and min prices where Coupon A is optimal -/
theorem coupon_savings_difference (minPrice maxPrice : ℝ) : 
  (minPrice > 120) →
  (maxPrice > 120) →
  (∀ p : ℝ, minPrice ≤ p → p ≤ maxPrice → 
    savingsA p ≥ max (savingsB) (savingsC p)) →
  (∃ p : ℝ, p > maxPrice → 
    savingsA p < max (savingsB) (savingsC p)) →
  (∃ p : ℝ, p < minPrice → p > 120 → 
    savingsA p < max (savingsB) (savingsC p)) →
  maxPrice - minPrice = 160 := by
sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l2169_216908


namespace NUMINAMATH_CALUDE_estimate_pi_random_simulation_l2169_216968

/-- Estimate pi using a random simulation method with a square paper and inscribed circle -/
theorem estimate_pi_random_simulation (total_seeds : ℕ) (seeds_in_circle : ℕ) :
  total_seeds = 1000 →
  seeds_in_circle = 778 →
  ∃ (pi_estimate : ℝ), pi_estimate = 4 * (seeds_in_circle : ℝ) / (total_seeds : ℝ) ∧ 
                        abs (pi_estimate - 3.112) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_estimate_pi_random_simulation_l2169_216968


namespace NUMINAMATH_CALUDE_division_problem_l2169_216949

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2169_216949


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l2169_216905

theorem baker_cakes_sold (initial_cakes : ℕ) (bought_cakes : ℕ) (remaining_cakes : ℕ) :
  initial_cakes = 121 →
  bought_cakes = 170 →
  remaining_cakes = 186 →
  ∃ (sold_cakes : ℕ), sold_cakes = 105 ∧ initial_cakes - sold_cakes + bought_cakes = remaining_cakes :=
by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l2169_216905


namespace NUMINAMATH_CALUDE_uncool_parents_count_l2169_216929

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total = 40)
  (h2 : cool_dads = 20)
  (h3 : cool_moms = 25)
  (h4 : both_cool = 10) :
  total - (cool_dads + cool_moms - both_cool) = 5 := by
  sorry

end NUMINAMATH_CALUDE_uncool_parents_count_l2169_216929


namespace NUMINAMATH_CALUDE_reach_power_of_three_l2169_216924

/-- Represents the possible operations on the blackboard -/
inductive Operation
  | triple_minus_one : Operation  -- 3k - 1
  | double_plus_one : Operation   -- 2k + 1
  | half : Operation              -- k / 2

/-- Applies an operation to a number if the result is an integer -/
def apply_operation (k : ℤ) (op : Operation) : Option ℤ :=
  match op with
  | Operation.triple_minus_one => some (3 * k - 1)
  | Operation.double_plus_one => some (2 * k + 1)
  | Operation.half => if k % 2 = 0 then some (k / 2) else none

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a number -/
def apply_sequence (n : ℤ) (seq : OperationSequence) : Option ℤ :=
  seq.foldl (fun acc op => acc.bind (fun k => apply_operation k op)) (some n)

/-- The main theorem -/
theorem reach_power_of_three (n : ℤ) (h : n ≥ 1) :
  ∃ (seq : OperationSequence), apply_sequence n seq = some (3^2023) :=
sorry

end NUMINAMATH_CALUDE_reach_power_of_three_l2169_216924


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2169_216952

theorem smallest_number_divisible (n : ℕ) : n = 1013 ↔ 
  (∀ m : ℕ, m < n → 
    ¬(∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
      m - 5 = 12 * k₁ ∧
      m - 5 = 16 * k₂ ∧
      m - 5 = 18 * k₃ ∧
      m - 5 = 21 * k₄ ∧
      m - 5 = 28 * k₅)) ∧
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    n - 5 = 12 * k₁ ∧
    n - 5 = 16 * k₂ ∧
    n - 5 = 18 * k₃ ∧
    n - 5 = 21 * k₄ ∧
    n - 5 = 28 * k₅) :=
by sorry


end NUMINAMATH_CALUDE_smallest_number_divisible_l2169_216952


namespace NUMINAMATH_CALUDE_competition_results_l2169_216997

/-- Represents the score for a single competition -/
structure CompetitionScore where
  first : ℕ+
  second : ℕ+
  third : ℕ+
  first_gt_second : first > second
  second_gt_third : second > third

/-- Represents the results of all competitions -/
structure CompetitionResults where
  score : CompetitionScore
  num_competitions : ℕ+
  a_total_score : ℕ
  b_total_score : ℕ
  c_total_score : ℕ
  b_first_place_count : ℕ

/-- The main theorem statement -/
theorem competition_results 
  (res : CompetitionResults)
  (h1 : res.num_competitions = 6)
  (h2 : res.a_total_score = 26)
  (h3 : res.b_total_score = 11)
  (h4 : res.c_total_score = 11)
  (h5 : res.b_first_place_count = 1) :
  ∃ (b_third_place_count : ℕ), b_third_place_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_competition_results_l2169_216997


namespace NUMINAMATH_CALUDE_inequality_proof_l2169_216915

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  b^2 / a < a^2 / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2169_216915


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l2169_216966

-- Define the number of herbs and gems
def num_herbs : ℕ := 4
def num_gems : ℕ := 6

-- Define the number of incompatible combinations for one gem
def incompatible_combinations : ℕ := 3

-- Define the number of herbs that can be used with the specific gem
def specific_gem_combinations : ℕ := 1

-- Theorem statement
theorem wizard_elixir_combinations :
  let total_combinations := num_herbs * num_gems
  let remaining_after_incompatible := total_combinations - incompatible_combinations
  let valid_combinations := remaining_after_incompatible - (num_herbs - specific_gem_combinations)
  valid_combinations = 18 := by
  sorry


end NUMINAMATH_CALUDE_wizard_elixir_combinations_l2169_216966


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l2169_216977

theorem at_least_one_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 1/y ≥ 2) ∨ (y + 1/z ≥ 2) ∨ (z + 1/x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l2169_216977
